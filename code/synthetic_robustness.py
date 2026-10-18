"""
synthetic_robustness.py - Robustness runs for the synthetic-world experiment.

Uses the world, networks and metric from synthetic_world.py (keep both files
in the same folder) and adds three things:

  1. SEEDS. Every setting is run with several seeds. A seed changes the world's
     random "physics", the networks' starting weights, the training streams and
     the test objects.
  2. A PRE-SPECIFIED COMPARISON OF LEARNING RULES, fixed before running.
     Each rule changes one thing from the one below it:
       recon           reconstruct the (noise-corrupted) input, fresh objects
       contrast_fixed  match two glimpses of the same object, fixed set of
                       6000 objects per sense, object does not change
       contrast_fresh  same, but fresh objects every step
       contrast_drift  same, and the object drifts slightly between glimpses
                       (the rule used in the main experiment)
  3. PERMUTATION-CALIBRATED SCORES (after Groger et al., ICML 2026). For each
     measurement we shuffle the pairing 20 times, repeating the full
     max-over-layer-pairs search each time, to get a null distribution:
       calibrated = (score - null_mean) / (1 - null_mean)
       p          = (1 + #null >= score) / (1 + 20)

Parts (all use width 32 and 8000 steps unless noted):
  A  overlap sweep     shared 0,2,4,6,8 of 8   rule contrast_drift
  B  learning rules    shared 8 of 8           all four rules
  C  width sweep       shared 4 of 8           widths 4, 16, 64 (32 is in A)

Results are appended to robustness_results.csv after every run; if the
script is interrupted, run it again and it continues where it stopped.

Usage:
  python synthetic_robustness.py                 # everything, seeds 0 1 2
  python synthetic_robustness.py --parts A       # one part
  python synthetic_robustness.py --seeds 0 1 2 3 4
  python synthetic_robustness.py --summary       # just print the tables
"""
import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_world import (DEVICE, D, DRIFT, N_TEST, K, World, EyeNet, EarNet,
                             same_object_loss, features, knn_mask, mutual_knn,
                             is_degenerate, subsample_sd)

STEPS = 8000
BS = 256
N_FIXED = 6000
N_NULL = 20
OUT = "robustness_results.csv"
FIELDS = ["part", "rule", "n_shared", "width", "seed", "step",
          "alignment", "sd", "null_mean", "null_p95", "calibrated", "p_value",
          "eye_truth", "ear_truth", "eye_truth_null", "ear_truth_null",
          "loss_eye", "loss_ear", "seconds"]


# --------------------------------------------------------------------------
# Metric with a permutation null (full layer search repeated per shuffle)
# --------------------------------------------------------------------------
def layer_masks(layers):
    return [None if is_degenerate(x) else knn_mask(x.to(DEVICE)) for x in layers]


def best_from_masks(masks_a, masks_b):
    best, pair = -1.0, (None, None)
    for j, mb in enumerate(masks_b):
        if mb is None:
            continue
        for i, ma in enumerate(masks_a):
            if ma is None:
                continue
            s = mutual_knn(ma, mb)
            if s > best:
                best, pair = s, (i, j)
    return best, pair


def calibrated_alignment(layers_a, layers_b, n_null, seed):
    ma = layer_masks(layers_a)
    real, pair = best_from_masks(ma, layer_masks(layers_b))
    g = torch.Generator().manual_seed(10_000 + seed)
    null = []
    for _ in range(n_null):
        perm = torch.randperm(layers_b[0].shape[0], generator=g)
        null.append(best_from_masks(ma, layer_masks([x[perm] for x in layers_b]))[0])
    null = torch.tensor(null)
    mu = float(null.mean())
    return dict(score=real, pair=pair, null_mean=mu,
                null_p95=float(null.quantile(0.95)),
                calibrated=(real - mu) / (1 - mu),
                p_value=(1 + int((null >= real).sum())) / (1 + n_null))


# --------------------------------------------------------------------------
# Decoders (only used by the 'recon' rule)
# --------------------------------------------------------------------------
class EyeDecoder(nn.Module):
    def __init__(self, w, bott=32):
        super().__init__()
        self.w = w
        self.fc = nn.Linear(bott, 4 * w * 16)
        self.net = nn.Sequential(
            nn.GELU(), nn.ConvTranspose2d(4 * w, 2 * w, 4, 2, 1),
            nn.GELU(), nn.ConvTranspose2d(2 * w, w, 4, 2, 1),
            nn.GELU(), nn.ConvTranspose2d(w, 3, 4, 2, 1))

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 4 * self.w, 4, 4))


class EarDecoder(nn.Module):
    """Reconstructs the cochlea output (what the ear network actually sees)."""
    def __init__(self, w, bott=32):
        super().__init__()
        self.w = w
        self.fc = nn.Linear(bott, 4 * w * 32)
        self.net = nn.Sequential(
            nn.GELU(), nn.ConvTranspose2d(4 * w, 2 * w, 4, 2, 1),
            nn.GELU(), nn.ConvTranspose2d(2 * w, w, 4, 2, 1),
            nn.GELU(), nn.ConvTranspose2d(w, 1, 4, 2, 1))

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 4 * self.w, 8, 4))


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def run(rule, n_shared, width, seed, steps=STEPS, drift=DRIFT, checkpoints=None, return_features=False, return_models=False):
    world = World(n_shared, seed=seed)
    shared_t, z_eye_t, z_ear_t = world.sample(N_TEST, seed=999 + seed)
    gt = torch.Generator(device=DEVICE).manual_seed(12345 + seed)
    eye_test, ear_test = world.see(z_eye_t, gt), world.hear(z_ear_t, gt)
    shared_t = shared_t.cpu()

    torch.manual_seed(seed)
    eye, ear = EyeNet(width).to(DEVICE), EarNet(width).to(DEVICE)
    params_e, params_a = list(eye.parameters()), list(ear.parameters())
    if rule == "recon":
        dec_e, dec_a = EyeDecoder(width).to(DEVICE), EarDecoder(width).to(DEVICE)
        params_e += list(dec_e.parameters())
        params_a += list(dec_a.parameters())
    opt_e = torch.optim.Adam(params_e, lr=1e-3)
    opt_a = torch.optim.Adam(params_a, lr=1e-3)
    g = torch.Generator(device=DEVICE).manual_seed(seed + 7)

    if rule == "contrast_fixed":
        # each sense gets its own fixed, disjoint set of objects
        fixed_e = torch.rand(N_FIXED, D, generator=g, device=DEVICE)
        fixed_a = torch.rand(N_FIXED, D, generator=g, device=DEVICE)

    saved = {}

    def measure(step, loss_e, loss_a):
        fe, fa = features(eye, eye_test), features(ear, ear_test)
        if return_features:
            saved[step] = (fe[-1].clone(), fa[-1].clone())   # final 32-d embedding layer
        c = calibrated_alignment(fe, fa, N_NULL, seed)
        out = dict(step=step, alignment=c["score"], null_mean=c["null_mean"],
                   null_p95=c["null_p95"], calibrated=c["calibrated"], p_value=c["p_value"],
                   sd=subsample_sd(fe[c["pair"][0]], fa[c["pair"][1]]),
                   loss_eye=loss_e, loss_ear=loss_a)
        if n_shared > 0:
            ce = calibrated_alignment(fe, [shared_t], 5, seed)
            ca = calibrated_alignment(fa, [shared_t], 5, seed)
            out.update(eye_truth=ce["score"], ear_truth=ca["score"],
                       eye_truth_null=ce["null_mean"], ear_truth_null=ca["null_mean"])
        else:
            out.update(eye_truth=float("nan"), ear_truth=float("nan"),
                       eye_truth_null=float("nan"), ear_truth_null=float("nan"))
        return out

    checkpoints = set(checkpoints) if checkpoints else {0, steps // 2, steps}
    curve, loss_e, loss_a = [], float("nan"), float("nan")
    for step in range(steps + 1):
        if step in checkpoints:
            curve.append(measure(step, loss_e, loss_a))
        if step == steps:
            break

        if rule == "contrast_fixed":
            ze = fixed_e[torch.randint(0, N_FIXED, (BS,), device=DEVICE, generator=g)]
            za = fixed_a[torch.randint(0, N_FIXED, (BS,), device=DEVICE, generator=g)]
        else:
            ze = world.fresh(BS, g, eye_side=True)
            za = world.fresh(BS, g, eye_side=False)

        if rule == "recon":
            xe = world.see(ze, g)
            xa = world.hear(za, g)
            ze_hat = eye.encode(xe + 0.1 * torch.randn(xe.shape, generator=g, device=DEVICE))[0]
            za_hat = ear.encode(xa + 0.1 * torch.randn(xa.shape, generator=g, device=DEVICE))[0]
            le = F.mse_loss(dec_e(ze_hat), xe)
            with torch.no_grad():
                target_a = ear.cochlea(xa)
            la = F.mse_loss(dec_a(za_hat), target_a)
        else:
            if rule == "contrast_drift":
                ze2 = (ze + drift * torch.randn(ze.shape, generator=g, device=DEVICE)).clamp(0, 1)
                za2 = (za + drift * torch.randn(za.shape, generator=g, device=DEVICE)).clamp(0, 1)
            else:
                ze2, za2 = ze, za
            le = same_object_loss(eye(world.see(ze, g)), eye(world.see(ze2, g)))
            la = same_object_loss(ear(world.hear(za, g)), ear(world.hear(za2, g)))

        opt_e.zero_grad(); le.backward(); opt_e.step()
        opt_a.zero_grad(); la.backward(); opt_a.step()
        loss_e, loss_a = le.item(), la.item()
    if return_models:
        return curve, eye, ear, world
    if return_features:
        return curve, saved, shared_t, z_eye_t.cpu(), z_ear_t.cpu()
    return curve


# --------------------------------------------------------------------------
# Plan, resume, summary
# --------------------------------------------------------------------------
def plan(parts, seeds):
    jobs = []
    for seed in seeds:
        if "A" in parts:
            jobs += [("A", "contrast_drift", s, 32, seed) for s in [0, 2, 4, 6, 8]]
        if "B" in parts:
            jobs += [("B", r, 8, 32, seed) for r in ["recon", "contrast_fixed", "contrast_fresh"]]
        if "C" in parts:
            jobs += [("C", "contrast_drift", 4, w, seed) for w in [4, 16, 64]]
    return jobs


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def done_keys(rows):
    return {(r["rule"], int(r["n_shared"]), int(r["width"]), int(r["seed"])) for r in rows}


def mean_sd(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, s


def cell(rows, rule, s, w, key, step):
    return [float(r[key]) for r in rows if r["rule"] == rule and int(r["n_shared"]) == s
            and int(r["width"]) == w and int(r["step"]) == step]


def f(m, s):
    if m != m:
        return "      -      "
    return f"{m:.3f} ± {s:.3f}" if s == s else f"{m:.3f}        "


def summary(path=OUT):
    rows = load(path)
    if not rows:
        print("No results yet.")
        return
    last = max(int(r["step"]) for r in rows)
    print(f"\nMean ± SD across seeds (final step {last}). Calibrated: 0 = shuffled-null level.\n")
    hdr = f"{'':22s}{'n':>3s}{'eye<->ear':>16s}{'calibrated':>16s}{'max p':>7s}{'untrained':>16s}{'eye<->truth':>16s}{'ear<->truth':>16s}"

    def line(label, rule, s, w):
        fin = cell(rows, rule, s, w, "alignment", last)
        if not fin:
            return
        ps = cell(rows, rule, s, w, "p_value", last)
        print(f"{label:22s}{len(fin):>3d}{f(*mean_sd(fin)):>16s}"
              f"{f(*mean_sd(cell(rows, rule, s, w, 'calibrated', last))):>16s}"
              f"{max(ps):>7.3f}"
              f"{f(*mean_sd(cell(rows, rule, s, w, 'alignment', 0))):>16s}"
              f"{f(*mean_sd(cell(rows, rule, s, w, 'eye_truth', last))):>16s}"
              f"{f(*mean_sd(cell(rows, rule, s, w, 'ear_truth', last))):>16s}")

    print("A) OVERLAP SWEEP (contrast_drift, width 32)")
    print(hdr)
    for s in [0, 2, 4, 6, 8]:
        line(f"shared {s}/8", "contrast_drift", s, 32)
    print("\nB) LEARNING RULES (shared 8/8, width 32)")
    print(hdr)
    for r in ["recon", "contrast_fixed", "contrast_fresh", "contrast_drift"]:
        line(r, r, 8, 32)
    print("\nC) WIDTH SWEEP (contrast_drift, shared 4/8)")
    print(hdr)
    for w in [4, 16, 32, 64]:
        line(f"width {w}", "contrast_drift", 4, w)
    print("\n'max p' is the largest permutation p-value across seeds (0.048 = beat all 20 shuffles).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summary(args.out)
        return

    jobs = plan(args.parts.upper(), args.seeds)
    done = done_keys(load(args.out))
    todo = [j for j in jobs if (j[1], j[2], j[3], j[4]) not in done]
    # a job may appear in two parts (e.g. contrast_drift at 8/8); run it once
    seen, unique = set(), []
    for j in todo:
        k = (j[1], j[2], j[3], j[4])
        if k not in seen:
            seen.add(k)
            unique.append(j)
    print(f"Device: {DEVICE} | {len(unique)} runs to do ({len(jobs) - len(todo)} already done)")

    new_file = not os.path.exists(args.out)
    for n, (part, rule, s, w, seed) in enumerate(unique, 1):
        t = time.time()
        print(f"[{n}/{len(unique)}] part {part}  rule={rule}  shared={s}/8  width={w}  seed={seed}",
              flush=True)
        curve = run(rule, s, w, seed, steps=args.steps)
        secs = time.time() - t
        fin = curve[-1]
        print(f"    alignment={fin['alignment']:.3f} (null {fin['null_mean']:.3f}, "
              f"calibrated {fin['calibrated']:.3f}, p={fin['p_value']:.3f})  "
              f"untrained={curve[0]['alignment']:.3f}  ({secs:.0f}s)", flush=True)
        with open(args.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                wr.writeheader()
                new_file = False
            for m in curve:
                row = {k: m.get(k, "") for k in FIELDS}
                row.update(part=part, rule=rule, n_shared=s, width=w, seed=seed, seconds=round(secs))
                for k in FIELDS:
                    if isinstance(row[k], float):
                        row[k] = round(row[k], 5)
                wr.writerow(row)
    summary(args.out)


if __name__ == "__main__":
    main()
