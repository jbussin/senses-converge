"""
scale_eval.py - Do trained learners show the pattern Koepke et al. (2026)
report for language/vision models: fine-grained agreement falling as the
evaluation set grows, while coarse agreement holds?

For shared S in {8, 6, 4} of 8 properties (contrast_drift, sigma 0.05,
width 32, 8000 steps, seeds 0 1 2) we train the eye and ear, then evaluate
them on N = 256, 1024, 4096, 8192 NEW objects (one glimpse each), with
  k = 10        (fixed neighborhood size: "fine")
  k = N / 100   (fixed fraction of the data: "coarse")
We report layer-maximized alignment M, its permutation null (10 shuffles,
full layer search), the calibrated score, the perfect-learner level
M(Z_e, Z_a) at the same N and k, and each learner's alignment with the truth.
S = 8 is the control: with nothing private, any fall with N comes from
learning error rather than private properties.

Exploratory: 1024 objects x 4 glimpses per sense (4096 items; eye glimpse g
paired with ear glimpse g of the same object) versus 4096 single-glimpse
objects.

Pre-registered predictions (written before running):
  1. With k = 10, learned alignment falls with N at every S, including S = 8
     (learning error behaves like noise).
  2. With k = N/100, learned alignment is roughly constant in N.
  3. For S < 8, the learned curve falls roughly in proportion to the
     perfect-learner curve.
  4. Multiple glimpses: no confident prediction.

Needs synthetic_world.py and the updated synthetic_robustness.py.
Appends to scale_results.csv after every trained pair; rerun to resume.
Usage:  python scale_eval.py        |   python scale_eval.py --summary
"""
import argparse, csv, math, os, time

import torch
import torch.nn.functional as F

from synthetic_world import DEVICE, is_degenerate, features
from synthetic_robustness import run

OUT = "scale_results.csv"
SHARED = [8, 6, 4]
NS = [256, 1024, 4096, 8192]
N_NULL = 10
FIELDS = ["n_shared", "seed", "eval", "N", "k_mode", "k", "alignment", "null_mean", "calibrated",
          "perfect", "eye_truth", "ear_truth", "seconds"]


def knn_mask(x, k):
    f = F.normalize(x.to(DEVICE).float(), dim=-1)
    sim = f @ f.T
    sim.fill_diagonal_(-float("inf"))
    idx = sim.topk(k, dim=1).indices
    m = torch.zeros_like(sim)
    m.scatter_(1, idx, 1.0)
    return m


def overlap(ma, mb, k):
    return ((ma * mb).sum(1) / k).mean().item()


def masks(layers, k):
    return [None if is_degenerate(x) else knn_mask(x, k) for x in layers]


def best(ma_list, mb_list, k):
    top = -1.0
    for mb in mb_list:
        if mb is None:
            continue
        for ma in ma_list:
            if ma is not None:
                top = max(top, overlap(ma, mb, k))
    return top


def aligned(layers_a, layers_b, k, seed, n_null=N_NULL):
    ma = masks(layers_a, k)
    real = best(ma, masks(layers_b, k), k)
    g = torch.Generator().manual_seed(20_000 + seed)
    null = []
    for _ in range(n_null):
        perm = torch.randperm(layers_b[0].shape[0], generator=g)
        null.append(best(ma, masks([x[perm] for x in layers_b], k), k))
    mu = sum(null) / len(null)
    del ma
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return real, mu, (real - mu) / (1 - mu)


def evaluate(eye, ear, fe, fa, ze, za, zsh, n_shared, seed, label, rows, t0):
    n = fe[0].shape[0]
    for k_mode, k in [("fixed10", 10), ("frac1pct", max(1, round(n / 100)))]:
        a, mu, cal = aligned(fe, fa, k, seed)
        perfect = overlap(knn_mask(ze, k), knn_mask(za, k), k)
        if n_shared > 0:
            et = best(masks(fe, k), [knn_mask(zsh, k)], k)
            at = best(masks(fa, k), [knn_mask(zsh, k)], k)
        else:
            et = at = float("nan")
        rows.append(dict(n_shared=n_shared, seed=seed, eval=label, N=n, k_mode=k_mode, k=k,
                         alignment=round(a, 5), null_mean=round(mu, 5), calibrated=round(cal, 5),
                         perfect=round(perfect, 5), eye_truth=round(et, 5), ear_truth=round(at, 5),
                         seconds=round(time.time() - t0)))
        print(f"      {label:>8s} N={n:<5d} k={k:<3d} M={a:.3f} (null {mu:.3f}, cal {cal:.3f})  "
              f"perfect={perfect:.3f}  eye/ear-truth={et:.3f}/{at:.3f}", flush=True)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def ms(xs):
    xs = [x for x in xs if x == x]
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, s


def summary(path=OUT):
    rows = load(path)
    if not rows:
        print("No results yet."); return
    for k_mode, title in [("fixed10", "k = 10 (fine)"), ("frac1pct", "k = N/100 (coarse)")]:
        print(f"\n{title}: calibrated eye<->ear  |  perfect-learner level   (mean over seeds)")
        print("N      " + "".join(f"   S={s}/8 learned   perfect" for s in SHARED))
        for n in NS:
            line = f"{n:<7d}"
            for s in SHARED:
                sel = [r for r in rows if r["eval"] == "single" and int(r["N"]) == n and r["k_mode"] == k_mode
                       and int(r["n_shared"]) == s]
                if sel:
                    c, _ = ms([float(r["calibrated"]) for r in sel]); p, _ = ms([float(r["perfect"]) for r in sel])
                    line += f"        {c:.3f}     {p:.3f}"
                else:
                    line += "            -         -"
            print(line)
    print("\nExploratory multi-glimpse (N = 4096 items): calibrated eye<->ear, single vs 4 glimpses")
    for k_mode in ["fixed10", "frac1pct"]:
        for s in SHARED:
            one = [float(r["calibrated"]) for r in rows if r["eval"] == "single" and int(r["N"]) == 4096
                   and r["k_mode"] == k_mode and int(r["n_shared"]) == s]
            four = [float(r["calibrated"]) for r in rows if r["eval"] == "multi4" and r["k_mode"] == k_mode
                    and int(r["n_shared"]) == s]
            if one and four:
                print(f"  {k_mode:>8s} S={s}/8: single {ms(one)[0]:.3f}   4 glimpses {ms(four)[0]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--ns", type=int, nargs="+", default=NS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out); return
    done = {(int(r["n_shared"]), int(r["seed"])) for r in load(args.out)}
    todo = [(s, sd) for sd in args.seeds for s in SHARED if (s, sd) not in done]
    print(f"Device: {DEVICE} | {len(todo)} trained pairs to do")
    new_file = not os.path.exists(args.out)
    for i, (s, seed) in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] shared={s}/8 seed={seed}: training...", flush=True)
        _, eye, ear, world = run("contrast_drift", s, 32, seed, steps=args.steps, return_models=True)
        rows = []
        # single-glimpse test sets: one big draw, nested prefixes
        nmax = max(args.ns)
        zsh, ze, za = world.sample(nmax, seed=5000 + seed)
        g = torch.Generator(device=DEVICE).manual_seed(777 + seed)
        fe_all, fa_all = features(eye, world.see(ze, g)), features(ear, world.hear(za, g))
        for n in args.ns:
            evaluate(eye, ear, [x[:n] for x in fe_all], [x[:n] for x in fa_all], ze[:n].cpu(), za[:n].cpu(),
                     zsh[:n].cpu(), s, seed, "single", rows, t0)
        del fe_all, fa_all
        # exploratory: 1024 objects x 4 glimpses
        zsh4, ze4, za4 = world.sample(1024, seed=6000 + seed)
        ze4, za4, zsh4 = ze4.repeat(4, 1), za4.repeat(4, 1), zsh4.repeat(4, 1)
        g4 = torch.Generator(device=DEVICE).manual_seed(888 + seed)
        fe4, fa4 = features(eye, world.see(ze4, g4)), features(ear, world.hear(za4, g4))
        evaluate(eye, ear, fe4, fa4, ze4.cpu(), za4.cpu(), zsh4.cpu(), s, seed, "multi4", rows, t0)
        del fe4, fa4
        with open(args.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                wr.writeheader(); new_file = False
            wr.writerows(rows)
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
    summary(args.out)


if __name__ == "__main__":
    main()
