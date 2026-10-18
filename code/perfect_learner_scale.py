"""
perfect_learner_scale.py - Reference calculation for the perfect-learner scale table (tab:scale).

Alignment between two PERFECT learners (features = their own latent vectors,
no learning error) as the number of evaluated objects N grows, for S of 8
latent properties shared. Uses only synthetic_world.py (no training, no GPU).
Fixed k=10 versus k = N/100 (a fixed fraction of the data). Three draws each.

Usage: python perfect_learner_scale.py
"""
import statistics as st
from synthetic_world import World, knn_mask, mutual_knn

NS, SS, SEEDS = [256, 1024, 4096, 8192], [8, 7, 6, 4], [0, 1, 2]
for label, kfun in [("k=10", lambda n: 10), ("k=N/100", lambda n: max(1, round(n / 100)))]:
    print(f"\n{label}")
    print("N     " + "".join(f"   S={s}/8         " for s in SS))
    for n in NS:
        k = kfun(n)
        cells = []
        for s in SS:
            vals = []
            for seed in SEEDS:
                _, ze, za = World(s, seed=seed).sample(n, 999 + seed)
                vals.append(mutual_knn(knn_mask(ze, k), knn_mask(za, k), k))
            cells.append(f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}   ")
        print(f"{n:<6d}" + "".join(cells))
