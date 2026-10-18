"""Figures and summary numbers for the revised paper.
Inputs (in data/): robustness_results.csv, results_calibrated.csv
Outputs: figures/fig_synth_overlap_rules.pdf, figures/fig_synth_width.pdf,
         figures/fig_textvision.pdf, and data/summary.txt (all numbers used in the text)."""
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"pdf.fonttype": 42, 
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.6, "savefig.bbox": "tight",
})
CH = 10 / 1023
out = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.append(line)


def ms(xs):
    xs = [x for x in xs if x == x]
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, s


# ------------------------------------------------------------------ synthetic
R = list(csv.DictReader(open(os.path.join(DATA, "robustness_results.csv"))))
LAST = max(int(r["step"]) for r in R)


def get(rule, s, w, key, step=None):
    step = LAST if step is None else step
    return [float(r[key]) for r in R if r["rule"] == rule and int(r["n_shared"]) == s
            and int(r["width"]) == w and int(r["step"]) == step]


def seeds(rule, s, w):
    return sorted({int(r["seed"]) for r in R if r["rule"] == rule and int(r["n_shared"]) == s
                   and int(r["width"]) == w})


KEYS = ["alignment", "null_mean", "calibrated", "p_value", "eye_truth", "ear_truth",
        "eye_truth_null", "ear_truth_null"]


def block(label, rule, s, w):
    if not get(rule, s, w, "alignment"):
        say(f"{label}: MISSING")
        return
    parts = [f"{label} (seeds {seeds(rule, s, w)})"]
    for k in KEYS:
        v = get(rule, s, w, k)
        if all(x != x for x in v):
            continue
        m, sd = ms(v)
        parts.append(f"{k}={m:.4f}±{sd:.4f}" + (f" max={max(v):.3f}" if k == "p_value" else ""))
    for k in ["alignment", "eye_truth", "ear_truth", "calibrated"]:
        v = get(rule, s, w, k, 0)
        if v and not all(x != x for x in v):
            m, sd = ms(v)
            parts.append(f"init_{k}={m:.4f}±{sd:.4f}")
    say("  ".join(parts))


say("=== A: overlap (contrast_drift, w32)")
for s in [0, 2, 4, 6, 8]:
    block(f"S={s}", "contrast_drift", s, 32)
say("=== B: rules (S8, w32)")
RULES = ["recon", "contrast_fixed", "contrast_fresh", "contrast_drift"]
for r in RULES:
    block(r, r, 8, 32)
say("=== C: width (contrast_drift, S4)")
for w in [4, 16, 32, 64]:
    block(f"w={w}", "contrast_drift", 4, w)

# per-seed monotonicity check for the overlap sweep
for sd in sorted({int(r["seed"]) for r in R}):
    seq = [get_seed for get_seed in
           [[float(r["alignment"]) for r in R if r["rule"] == "contrast_drift" and int(r["n_shared"]) == s
             and int(r["width"]) == 32 and int(r["step"]) == LAST and int(r["seed"]) == sd] for s in [0, 2, 4, 6, 8]]]
    if all(seq):
        vals = [v[0] for v in seq]
        say(f"seed {sd} overlap sweep: {['%.4f' % v for v in vals]} monotone={all(a < b for a, b in zip(vals, vals[1:]))}")


def errbar(ax, xs, key, rule, s_or_w, which, col, mk, lab, sweep):
    ys, es = [], []
    for x in xs:
        v = get(rule, x, 32, key) if sweep == "S" else get(rule, 4, x, key)
        m, sd = ms(v)
        ys.append(m); es.append(0 if sd != sd else sd)
        for vv in v:
            ax.scatter([x], [vv], s=6, color=col, alpha=0.35, lw=0, zorder=2)
    ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=4, capsize=2, elinewidth=0.8, label=lab, zorder=3)


# Figure: overlap sweep + rules (2x2: cross-model on top, truth below)
fig, axs = plt.subplots(2, 2, figsize=(6.8, 4.6))
(a, b), (c, d) = axs
S = [0, 2, 4, 6, 8]
errbar(a, S, "alignment", "contrast_drift", None, None, AQUA, "^", "Eye$\\leftrightarrow$ear (trained)", "S")
a.plot(S, [ms(get("contrast_drift", s, 32, "alignment", 0))[0] for s in S], color=AQUA, ls="--", lw=0.9,
       marker="^", mfc="white", ms=3.5, label="Eye$\\leftrightarrow$ear (untrained)")
a.plot(S, [ms(get("contrast_drift", s, 32, "null_mean"))[0] for s in S], color=MUTED, ls=":", lw=1.0,
       label="Permutation null")
a.set_xticks(S); a.set_ylim(0, None)
a.set_ylabel("Eye$\\leftrightarrow$ear alignment")
a.set_title("(a) Overlap sweep: learners vs each other", fontsize=8, color=TEXT)
a.legend(fontsize=6.3, loc="upper left")

errbar(c, [2, 4, 6, 8], "eye_truth", "contrast_drift", None, None, BLUE, "o", "Eye (trained)", "S")
errbar(c, [2, 4, 6, 8], "ear_truth", "contrast_drift", None, None, ORANGE, "s", "Ear (trained)", "S")
for key, col, mk, lab in [("eye_truth", BLUE, "o", "Eye (untrained)"), ("ear_truth", ORANGE, "s", "Ear (untrained)")]:
    c.plot([2, 4, 6, 8], [ms(get("contrast_drift", s, 32, key, 0))[0] for s in [2, 4, 6, 8]], color=col,
           ls="--", lw=0.9, marker=mk, mfc="white", ms=3.5, label=lab)
c.set_xticks(S); c.set_xlim(-0.4, 8.4); c.set_ylim(0, None)
c.set_xlabel("Properties shared by the two senses (of 8)")
c.set_ylabel("Alignment with true shared properties")
c.set_title("(c) Overlap sweep: learners vs the truth", fontsize=8, color=TEXT)
c.legend(fontsize=6.3, loc="upper left")

labels = ["recon", "contrast\nfixed", "contrast\nfresh", "contrast\ndrift"]
def rule_panel(ax, specs, ylabel, title):
    for key, col, mk, lab, dx in specs:
        for i, r in enumerate(RULES):
            v = get(r, 8, 32, key); v0 = get(r, 8, 32, key, 0)
            if not v:
                continue
            m, sd = ms(v); m0, _ = ms(v0)
            for vv in v:
                ax.scatter([i + dx], [vv], s=6, color=col, alpha=0.35, lw=0, zorder=2)
            ax.errorbar([i + dx], [m], yerr=[0 if sd != sd else sd], color=col, marker=mk, ms=4, capsize=2,
                        elinewidth=0.8, label=lab if i == 0 else None, zorder=3)
            ax.scatter([i + dx], [m0], marker=mk, facecolor="white", edgecolor=col, s=16, lw=0.9, zorder=4)
            ax.plot([i + dx, i + dx], [m0, m], color=col, lw=0.6, ls=":", zorder=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, None)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=8, color=TEXT)
    ax.legend(fontsize=6.3, loc="upper left")
rule_panel(b, [("alignment", AQUA, "^", "Eye$\\leftrightarrow$ear", 0.0)], "Eye$\\leftrightarrow$ear alignment (8/8)",
           "(b) Learning rules: learners vs each other")
b.axhline(ms([float(r["null_mean"]) for r in R if int(r["n_shared"]) == 8 and int(r["step"]) == LAST])[0],
          color=MUTED, ls=":", lw=1.0, label="Permutation null")
b.legend(fontsize=6.3, loc="upper left")
rule_panel(d, [("eye_truth", BLUE, "o", "Eye", -0.12), ("ear_truth", ORANGE, "s", "Ear", 0.12)],
           "Alignment with truth (8/8)", "(d) Learning rules: learners vs the truth")
fig.text(0.5, -0.01, "Open markers: untrained. Filled: trained, mean $\\pm$ SD over seeds; faint dots: individual seeds.",
         ha="center", fontsize=6.8, color=MUTED)
fig.tight_layout(w_pad=2.5, h_pad=1.8)
fig.savefig(os.path.join(FIG, "fig_synth_overlap_rules.pdf"))

# Figure: width
fig, ax = plt.subplots(figsize=(3.4, 2.4))
W = [4, 16, 32, 64]
errbar(ax, W, "alignment", "contrast_drift", None, None, AQUA, "^", "Eye$\\leftrightarrow$ear", "W")
errbar(ax, W, "eye_truth", "contrast_drift", None, None, BLUE, "o", "Eye$\\leftrightarrow$truth", "W")
errbar(ax, W, "ear_truth", "contrast_drift", None, None, ORANGE, "s", "Ear$\\leftrightarrow$truth", "W")
ax.plot(W, [ms(get("contrast_drift", 4, w, "null_mean"))[0] for w in W], color=MUTED, ls="--", lw=0.9,
        label="Permutation null")
ax.set_xscale("log", base=2); ax.set_xticks(W); ax.set_xticklabels([str(w) for w in W])
ax.set_xlabel("Network width (base channels)")
ax.set_ylabel("Alignment (4/8 shared)"); ax.set_ylim(0, None)
ax.legend(fontsize=6.3, loc="upper left")
fig.savefig(os.path.join(FIG, "fig_synth_width.pdf"))

# ------------------------------------------------------------------ text-vision
T = list(csv.DictReader(open(os.path.join(DATA, "results_calibrated.csv"))))
say("=== Text-vision (calibrated)")
for r in T:
    say(f"{r['text_model']:24s} {r['vision_model'].split('/')[-1]:14s} M={float(r['alignment']):.4f}"
        f"±{float(r['alignment_sd']):.4f} null={float(r['shuffled']):.4f} cal={float(r['calibrated']):.4f}"
        f" p={float(r['p_value']):.3f} layers={r.get('n_text_layers','?')}x{r.get('n_vision_layers','?')}")
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5))
for vm, col, mk, lab in [("dinov2-large", BLUE, "o", "DINOv2-L"), ("dinov2-base", ORANGE, "s", "DINOv2-B"),
                         ("dinov2-small", AQUA, "^", "DINOv2-S")]:
    r = [x for x in T if x["vision_model"].endswith(vm) and "UNTRAINED" not in x["text_model"]]
    px = [float(x["text_params_M"]) for x in r]
    a.errorbar(px, [float(x["alignment"]) for x in r], yerr=[2 * float(x["alignment_sd"]) for x in r],
               color=col, marker=mk, ms=4, capsize=2, elinewidth=0.8, label=lab)
    a.plot(px, [float(x["shuffled"]) for x in r], color=col, ls="--", lw=0.8)
    b.plot(px, [float(x["calibrated"]) for x in r], color=col, marker=mk, ms=4, label=lab)
    u = [x for x in T if x["vision_model"].endswith(vm) and "UNTRAINED" in x["text_model"]]
    if u:
        a.scatter([float(u[0]["text_params_M"])], [float(u[0]["alignment"])], marker="x", color=col, s=20, zorder=5)
        b.scatter([float(u[0]["text_params_M"])], [float(u[0]["calibrated"])], marker="x", color=col, s=20, zorder=5)
a.axhline(CH, color=MUTED, lw=0.8, ls=":")
for ax_ in (a, b):
    ax_.set_xscale("log"); ax_.set_xlabel("Text model parameters (millions)")
a.set_ylabel("Mutual $k$-NN alignment"); a.set_ylim(0, None)
a.set_title("(a) Raw (dashed: permutation null)", fontsize=8, color=TEXT)
b.set_ylabel("Calibrated alignment"); b.set_ylim(0, None)
b.set_title("(b) Calibrated (x: random-init 410M)", fontsize=8, color=TEXT)
a.legend(fontsize=6.3, loc="upper left")
fig.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(FIG, "fig_textvision.pdf"))

# log-linear fits of calibrated score on log2 params
for vm in ["dinov2-small", "dinov2-base", "dinov2-large"]:
    r = [x for x in T if x["vision_model"].endswith(vm) and "UNTRAINED" not in x["text_model"]]
    for key in ["alignment", "calibrated"]:
        x = [math.log2(float(q["text_params_M"])) for q in r]; y = [float(q[key]) for q in r]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sxy = sum((i - mx) * (j - my) for i, j in zip(x, y)); sxx = sum((i - mx) ** 2 for i in x)
        syy = sum((j - my) ** 2 for j in y)
        say(f"fit {vm} {key}: slope/doubling={sxy / sxx:.4f} r={sxy / math.sqrt(sxx * syy):.3f}")
open(os.path.join(DATA, "summary.txt"), "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------------ drift sweep
DR = list(csv.DictReader(open(os.path.join(DATA, "drift_results.csv"))))
DRIFTS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
def dvals(d, key):
    if d == 0.0:
        return get("contrast_fresh", 8, 32, key)
    if d == 0.05:
        return get("contrast_drift", 8, 32, key)
    return [float(r[key]) for r in DR if r["part"] == "D" and float(r["drift"]) == d and int(r["step"]) == 8000]
say("=== Drift sweep")
for d in DRIFTS:
    say(f"drift={d}: " + "  ".join(f"{k}={ms(dvals(d,k))[0]:.4f}±{ms(dvals(d,k))[1]:.4f}" for k in ["alignment", "calibrated", "eye_truth", "ear_truth"]))
E = [r for r in DR if r["part"] == "E"]
say("=== Longer training (drift 0.05)")
STEPS_E = sorted({int(r["step"]) for r in E})
for st_ in STEPS_E:
    sel = [r for r in E if int(r["step"]) == st_]
    say(f"step={st_}: " + "  ".join(f"{k}={ms([float(r[k]) for r in sel])[0]:.4f}±{ms([float(r[k]) for r in sel])[1]:.4f}" for k in ["alignment", "calibrated", "eye_truth", "ear_truth"]))

fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.9, 2.35))
xs = list(range(len(DRIFTS)))
def dplot(ax, key, col, mk, lab):
    ys, es = [], []
    for i, d in enumerate(DRIFTS):
        v = dvals(d, key); m, s_ = ms(v)
        ys.append(m); es.append(0 if s_ != s_ else s_)
        for vv in v:
            ax.scatter([i + 0.22], [vv], s=9, color=col, alpha=0.55, lw=0, zorder=2)
    ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=4, capsize=2, elinewidth=0.8, label=lab, zorder=3)
dplot(a, "alignment", AQUA, "^", "Eye$\\leftrightarrow$ear")
a.plot(xs, [ms(dvals(d, "null_mean"))[0] for d in DRIFTS], color=MUTED, ls=":", lw=1.0, label="Permutation null")
dplot(b, "eye_truth", BLUE, "o", "Eye")
dplot(b, "ear_truth", ORANGE, "s", "Ear")
for ax_ in (a, b):
    ax_.set_xticks(xs); ax_.set_xticklabels(["0", ".01", ".02", ".05", ".1", ".2", ".4"])
    ax_.set_xlabel("Drift $\\sigma$ between glimpses"); ax_.set_ylim(0, None)
    ax_.axvline(3, color=GRID, lw=3, zorder=0)
a.set_ylabel("Alignment (8/8 shared)")
a.set_title("(a) Learners vs each other", fontsize=8, color=TEXT)
b.set_title("(b) Learners vs the truth", fontsize=8, color=TEXT)
a.legend(fontsize=6.2, loc="upper left"); b.legend(fontsize=6.2, loc="upper left")
for key, col, mk, lab in [("alignment", AQUA, "^", "Eye$\\leftrightarrow$ear"), ("eye_truth", BLUE, "o", "Eye$\\leftrightarrow$truth"),
                          ("ear_truth", ORANGE, "s", "Ear$\\leftrightarrow$truth")]:
    ys, es = [], []
    for st_ in STEPS_E:
        m, s_ = ms([float(r[key]) for r in E if int(r["step"]) == st_]); ys.append(m); es.append(s_)
    c.errorbar([s_ / 1000 for s_ in STEPS_E], ys, yerr=es, color=col, marker=mk, ms=4, capsize=2, elinewidth=0.8, label=lab)
c.set_xlabel("Training steps (thousands)"); c.set_ylim(0, None)
c.set_title("(c) Longer training ($\\sigma=0.05$)", fontsize=8, color=TEXT)
c.set_ylim(0, 0.46); c.legend(fontsize=6.2, loc="upper left")
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(FIG, "fig_drift.pdf"))
open(os.path.join(DATA, "summary.txt"), "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------------ scale evaluation
SC = list(csv.DictReader(open(os.path.join(DATA, "scale_results.csv"))))
NSC = [256, 1024, 4096, 8192]
def scv(s, n, km, key):
    return [float(r[key]) for r in SC if r["eval"] == "single" and int(r["n_shared"]) == s
            and int(r["N"]) == n and r["k_mode"] == km]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for ax_, km, title in [(a, "fixed10", "(a) Fine: $k=10$"), (b, "frac1pct", "(b) Coarse: $k=N/100$")]:
    for s, col, mk in [(8, BLUE, "o"), (6, ORANGE, "s"), (4, AQUA, "^")]:
        ys, es = [], []
        for n in NSC:
            v = scv(s, n, km, "calibrated"); m, sd_ = ms(v); ys.append(m); es.append(0 if sd_ != sd_ else sd_)
            for vv in v:
                ax_.scatter([n * 1.12], [vv], s=9, color=col, alpha=0.55, lw=0, zorder=2)
        ax_.errorbar(NSC, ys, yerr=es, color=col, marker=mk, ms=4, capsize=2, elinewidth=0.8,
                     label=f"{s}/8 shared", zorder=3)
    ax_.set_xscale("log", base=2); ax_.set_xticks(NSC); ax_.set_xticklabels([str(n) for n in NSC])
    ax_.set_xlabel("Evaluated objects $N$"); ax_.set_title(title, fontsize=8, color=TEXT); ax_.set_ylim(0, None)
a.set_ylabel("Calibrated eye$\\leftrightarrow$ear alignment")
a.legend(fontsize=6.3, loc="upper right")
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_scale.pdf"))
