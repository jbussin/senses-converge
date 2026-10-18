"""
unpaired_alignment.py - Can the correspondence between the eye and the ear be
recovered WITHOUT any paired data, and does that depend on the drift?

For each drift sigma (shared 8/8, width 32, 8000 steps, seeds 0 1 2) we train
the eye and ear exactly as in drift_sweep.py, take their final 32-d embeddings
of the 1024 test objects, THROW AWAY the pairing, and ask an unpaired matching
method to find which sound goes with which image.

Matching method: entropic Gromov-Wasserstein optimal transport (compares only
the internal distance structure of each space). Three variants are run and
the valid one with the lowest Gromov-Wasserstein cost is kept - an
unsupervised choice that never looks at the true pairing (a coupling is valid
if it is finite with total mass within 1% of 1; log-domain Sinkhorn is used):
  (a) epsilon 0.005, (b) epsilon 0.02, (c) annealed 0.1 -> 0.005.
Distances: cosine distance between embeddings, divided by its mean.

Scores (using the true pairing only for evaluation):
  top1 / top10 / top50 : true partner is the best / in the top 10 / top 50
                         matches (chance 0.001 / 0.010 / 0.049)
  quality              : 1 - mean true-latent distance between each object and
                         its match, divided by the mean distance between random
                         objects (0 = random matching, 1 = perfect)

Pre-registered predictions (written before running):
  1. Matching quality follows the same inverted U in drift as paired alignment.
  2. Untrained networks and drift 0 match at or near chance.
  3. Because the matcher needs near-faithful geometry, absolute quality may be
     low everywhere; a null result is possible and will be reported as such.

Needs synthetic_world.py and the updated synthetic_robustness.py in the same
folder, and the POT library:   python -m pip install pot
Appends to unpaired_results.csv after every run; rerun to resume.

Usage:
  python unpaired_alignment.py
  python unpaired_alignment.py --summary
  python unpaired_alignment.py --calibrate   # matcher vs. controlled noise (Section 4.5)
"""
import argparse, csv, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
import ot

from synthetic_robustness import run, DEVICE

OUT = "unpaired_results.csv"
DRIFTS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
FIELDS = ["drift", "seed", "stage", "top1", "top10", "top50", "quality", "gw_cost",
          "variant", "paired_alignment", "seconds"]


def cos_dist(X):
    X = F.normalize(X.double(), dim=1)
    D = (1 - X @ X.T).clamp_min(0)
    return D / D.mean()


def gw_cost(C1, C2, p, q, T):
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, "square_loss")
    return float(ot.gromov.gwloss(constC, hC1, hC2, T))


def valid(T):
    """Reject numerically collapsed couplings (NaN, or total mass far from 1)."""
    return bool(torch.isfinite(T).all()) and abs(float(T.sum()) - 1.0) < 0.01


def egw(C1, C2, p, q, eps, G0=None, max_iter=200, tol=1e-9):
    """Entropic Gromov-Wasserstein (square loss), same projected-gradient
    updates as POT's entropic_gromov_wasserstein, but each step is solved with
    log-domain Sinkhorn so small epsilon cannot underflow."""
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, "square_loss")
    T = torch.outer(p, q) if G0 is None else G0
    for _ in range(max_iter):
        T_prev = T
        tens = ot.gromov.gwggrad(constC, hC1, hC2, T)
        T = ot.sinkhorn(p, q, tens, eps, method="sinkhorn_log", numItermax=1000)
        if float((T - T_prev).abs().sum()) < tol:
            break
    return T


def match(fe, fa):
    dev = DEVICE
    C1, C2 = cos_dist(fe).to(dev), cos_dist(fa).to(dev)
    n = C1.shape[0]
    p = torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)
    q = p.clone()
    cands = []
    for name, eps in [("eps0.005", 5e-3), ("eps0.02", 2e-2)]:
        T = egw(C1, C2, p, q, eps)
        if valid(T):
            cands.append((gw_cost(C1, C2, p, q, T), name, T))
    T = None
    for eps in [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]:
        T_new = egw(C1, C2, p, q, eps, G0=T)
        if not valid(T_new):
            break
        T = T_new
    if T is not None:
        cands.append((gw_cost(C1, C2, p, q, T), "annealed", T))
    if not cands:  # nothing valid: fall back to the uninformative coupling
        T = torch.outer(p, q)
        cands.append((gw_cost(C1, C2, p, q, T), "none_valid", T))
    cost, name, T = min(cands, key=lambda c: c[0])
    return T.detach().cpu().numpy(), cost, name


def score(T, shared):
    n = T.shape[0]
    rank = np.argsort(-T, axis=1)
    idx = np.arange(n)
    Sd = torch.cdist(shared.double(), shared.double()).numpy()
    m = rank[:, 0]
    return dict(top1=float((m == idx).mean()),
                top10=float(np.mean([i in rank[i, :10] for i in range(n)])),
                top50=float(np.mean([i in rank[i, :50] for i in range(n)])),
                quality=float(1 - Sd[idx, m].mean() / Sd.mean()))


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def ms(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, s


def summary(path=OUT):
    rows = load(path)
    if not rows:
        print("No results yet."); return
    print("\nUNPAIRED MATCHING (shared 8/8, width 32; mean ± SD over seeds)")
    print("chance: top1 0.001, top10 0.010, top50 0.049, quality 0.000\n")
    print(f"{'drift':>6s} {'stage':>8s}{'top1':>15s}{'top10':>15s}{'top50':>15s}{'quality':>16s}{'paired M':>15s}")
    for d in DRIFTS:
        for stage in (["untrained", "trained"] if d == 0.0 else ["trained"]):
            sel = [r for r in rows if float(r["drift"]) == d and r["stage"] == stage]
            if not sel:
                continue
            f = lambda k: "%.3f ± %.3f" % ms([float(r[k]) for r in sel])
            print(f"{('-' if stage == 'untrained' else '%.2f' % d):>6s} {stage:>8s}{f('top1'):>15s}{f('top10'):>15s}{f('top50'):>15s}{f('quality'):>16s}{f('paired_alignment'):>15s}")


def calibrate(n=400, noise=(0.05, 0.08, 0.10, 0.12, 0.15, 0.2)):
    """How much paired agreement does the matcher need? Match true latents Z
    against Z + Gaussian noise (all 8 properties shared) and report paired
    alignment next to unpaired match quality. Writes unpaired_calibration.csv."""
    from synthetic_world import World, best_alignment
    _, ze, _ = World(8, seed=0).sample(n, 999)
    shared, Z = ze.cpu(), ze.cpu()
    g = torch.Generator().manual_seed(3)
    with open("unpaired_calibration.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["noise_sd", "paired_alignment", "quality", "top1", "top10", "variant", "seconds"])
        for sd in noise:
            B = Z + sd * torch.randn(Z.shape, generator=g)
            M = best_alignment([Z], [B])[0]
            t = time.time()
            T, _, var = match(Z, B)
            sc = score(T, shared)
            row = [sd, round(M, 4), round(sc["quality"], 4), round(sc["top1"], 4), round(sc["top10"], 4),
                   var, round(time.time() - t)]
            wr.writerow(row)
            print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out); return
    if args.calibrate:
        calibrate(); return
    done = {(float(r["drift"]), int(r["seed"])) for r in load(args.out)}
    todo = [(d, s) for s in args.seeds for d in DRIFTS if (d, s) not in done]
    print(f"Device: {DEVICE} | {len(todo)} runs to do")
    new_file = not os.path.exists(args.out)
    for n, (d, seed) in enumerate(todo, 1):
        t = time.time()
        rule = "contrast_fresh" if d == 0.0 else "contrast_drift"
        print(f"[{n}/{len(todo)}] drift={d} seed={seed}: training...", flush=True)
        curve, saved, shared, _, _ = run(rule, 8, 32, seed, steps=args.steps, drift=d,
                                         checkpoints=[0, args.steps], return_features=True)
        results = []
        # untrained networks are identical across drift values for a given seed,
        # so the untrained stage is matched once per seed (at drift 0)
        stages = [("trained", args.steps, curve[-1])]
        if d == 0.0:
            stages.insert(0, ("untrained", 0, curve[0]))
        for stage, step, m in stages:
            fe, fa = saved[step]
            T, cost, variant = match(fe, fa)
            sc = score(T, shared)
            results.append(dict(drift=d, seed=seed, stage=stage, gw_cost=round(cost, 6), variant=variant,
                                paired_alignment=round(m["alignment"], 5), **{k: round(v, 5) for k, v in sc.items()}))
            print(f"    {stage:>9s}: top1={sc['top1']:.3f} top10={sc['top10']:.3f} top50={sc['top50']:.3f} "
                  f"quality={sc['quality']:.3f}  (paired M={m['alignment']:.3f}, {variant})", flush=True)
        secs = round(time.time() - t)
        with open(args.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                wr.writeheader(); new_file = False
            for r in results:
                r["seconds"] = secs
                wr.writerow(r)
    summary(args.out)


if __name__ == "__main__":
    main()
