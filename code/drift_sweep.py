"""
drift_sweep.py - How much should an object change between two glimpses?

Pre-registered predictions (written before running):
  1. Inverted U: alignment peaks at an intermediate drift. Too little drift
     collapses toward 'contrast_fresh' (latents encoded, geometry not tied);
     too much makes the two glimpses nearly unrelated objects.
  2. The ear is more sensitive to drift than the eye.
  3. Training 3x longer at drift 0.05 closes some, not most, of the gap to
     the perfect-learner reference levels.

Parts (shared 8/8, width 32, seeds 0 1 2):
  D  drift in {0.01, 0.02, 0.1, 0.2, 0.4} for 8000 steps
     (drift 0 = contrast_fresh and drift 0.05 = contrast_drift are already in
      robustness_results.csv and are merged in the summary)
  E  drift 0.05 for 24000 steps, checkpoints at 0, 4000, 8000, 16000, 24000

Needs synthetic_world.py and the updated synthetic_robustness.py in the same
folder. Appends to drift_results.csv after every run; rerun to resume.

Usage:
  python drift_sweep.py              # parts D and E (~55 min on an RTX 4070 Ti)
  python drift_sweep.py --parts D
  python drift_sweep.py --summary
"""
import argparse, csv, math, os, time

from synthetic_robustness import run, FIELDS

OUT = "drift_results.csv"
FIELDS_D = ["drift"] + FIELDS
DRIFTS = [0.01, 0.02, 0.1, 0.2, 0.4]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plan(parts, seeds):
    jobs = []
    for seed in seeds:
        if "D" in parts:
            jobs += [("D", d, 8000, seed) for d in DRIFTS]
        if "E" in parts:
            jobs.append(("E", 0.05, 24000, seed))
    return jobs


def ms(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, s


def summary(path=OUT, base="robustness_results.csv"):
    rows = load(path)
    old = load(base)
    table = {}
    for d, rule in [(0.0, "contrast_fresh"), (0.05, "contrast_drift")]:
        sel = [r for r in old if r["rule"] == rule and r["n_shared"] == "8" and r["width"] == "32"
               and r["step"] == "8000"]
        if sel:
            table[d] = sel
    for d in DRIFTS:
        sel = [r for r in rows if r["part"] == "D" and float(r["drift"]) == d and r["step"] == "8000"]
        if sel:
            table[d] = sel
    print("\nD) DRIFT SWEEP (shared 8/8, width 32, 8000 steps; mean ± SD over seeds)")
    print(f"{'drift':>7s}{'n':>3s}{'eye<->ear':>18s}{'calibrated':>18s}{'eye<->truth':>18s}{'ear<->truth':>18s}")
    for d in sorted(table):
        sel = table[d]
        f = lambda k: "%.3f ± %.3f" % ms([float(r[k]) for r in sel])
        print(f"{d:>7.2f}{len(sel):>3d}{f('alignment'):>18s}{f('calibrated'):>18s}{f('eye_truth'):>18s}{f('ear_truth'):>18s}")
    e = [r for r in rows if r["part"] == "E"]
    if e:
        print("\nE) LONGER TRAINING (drift 0.05, shared 8/8, width 32)")
        print(f"{'step':>7s}{'n':>3s}{'eye<->ear':>18s}{'calibrated':>18s}{'eye<->truth':>18s}{'ear<->truth':>18s}")
        for st in sorted({int(r["step"]) for r in e}):
            sel = [r for r in e if int(r["step"]) == st]
            f = lambda k: "%.3f ± %.3f" % ms([float(r[k]) for r in sel])
            print(f"{st:>7d}{len(sel):>3d}{f('alignment'):>18s}{f('calibrated'):>18s}{f('eye_truth'):>18s}{f('ear_truth'):>18s}")
    print("\nPerfect-learner reference at 8/8 shared: 1.000 for all three columns.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="DE")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--steps", type=int, default=None, help="override steps (testing only)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    done = {(r["part"], float(r["drift"]), int(r["seed"])) for r in load(args.out)}
    todo = [j for j in plan(args.parts.upper(), args.seeds) if (j[0], j[1], j[3]) not in done]
    print(f"{len(todo)} runs to do")
    new_file = not os.path.exists(args.out)
    for n, (part, drift, steps, seed) in enumerate(todo, 1):
        steps = args.steps or steps
        cps = [0, steps // 6, steps // 3, 2 * steps // 3, steps] if part == "E" else None
        t = time.time()
        print(f"[{n}/{len(todo)}] part {part}  drift={drift}  steps={steps}  seed={seed}", flush=True)
        curve = run("contrast_drift", 8, 32, seed, steps=steps, drift=drift, checkpoints=cps)
        secs = time.time() - t
        fin = curve[-1]
        print(f"    eye<->ear={fin['alignment']:.3f}  eye<->truth={fin['eye_truth']:.3f}  "
              f"ear<->truth={fin['ear_truth']:.3f}  ({secs:.0f}s)", flush=True)
        with open(args.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS_D)
            if new_file:
                wr.writeheader(); new_file = False
            for m in curve:
                row = {k: m.get(k, "") for k in FIELDS}
                row.update(drift=drift, part=part, rule="contrast_drift", n_shared=8, width=32,
                           seed=seed, seconds=round(secs))
                for k in FIELDS_D:
                    if isinstance(row.get(k), float):
                        row[k] = round(row[k], 5)
                wr.writerow(row)
    summary(args.out)


if __name__ == "__main__":
    main()
