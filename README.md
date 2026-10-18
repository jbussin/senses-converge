# Separately Trained Senses Converge on Shared Latent Structure

**An Empirical Check of Identifiability Predictions Without Shared Data**
Jonathan Bussing · Independent researcher · jbussing@ucsd.edu

Two networks, an "eye" trained on rendered images and an "ear" trained on rendered sounds, are trained alone, without labels, on disjoint objects from a synthetic world. They converge on the part of the world they both perceive, and each moves toward the world's true latent structure. That happens only under the learning rule that approximates the assumptions of identifiability theory: contrastive learning between glimpses of an object that drifts between them.

- Paper: [`paper/senses_converge.pdf`](paper/senses_converge.pdf)
- Interactive summary: `index.html` (served with GitHub Pages)
- Preprint record and DOI: Zenodo (link added on release)

## Contents

```
code/synthetic_world.py        world, renderers, learners, mutual k-NN metric
code/synthetic_robustness.py   overlap, learning-rule and width sweeps (Section 4.3)
code/drift_sweep.py            drift sweep and 24k-step runs (Section 4.4)
code/unpaired_alignment.py     unpaired Gromov-Wasserstein matching (Section 4.5; --calibrate for Table 3)
code/scale_eval.py             trained learners evaluated at N = 256 to 8,192 (Section 4.6)
code/perfect_learner_scale.py  perfect-learner alignment vs evaluation size (Table 5; no training)
code/platonic_test.py          Pythia x DINOv2 check on WIT with permutation calibration (Section 5)
code/make_figures_v2.py        Figures 1-4 and the summary numbers in data/summary.txt
data/                          raw results for every run in the paper (three seeds per configuration)
paper/                         LaTeX source, bibliography, figures and compiled PDF
```

Data files:

| File | Produced by |
|---|---|
| `robustness_results.csv` | `synthetic_robustness.py` |
| `drift_results.csv` | `drift_sweep.py` |
| `unpaired_results.csv`, `unpaired_calibration.csv` | `unpaired_alignment.py` (with and without `--calibrate`) |
| `scale_results.csv` | `scale_eval.py` |
| `perfect_learner_scale.json` | `perfect_learner_scale.py` |
| `results_calibrated.csv` | `platonic_test.py --out results_calibrated.csv` |
| `synthetic_results.csv` | `synthetic_world.py` (single-seed check; matches seed 0 of the multi-seed runs to GPU nondeterminism) |
| `pythia_dinov2_results.csv` | earlier uncalibrated text-vision run, kept for reference |
| `summary.txt` | `make_figures_v2.py` |

## Reproducing

```
pip install -r requirements.txt
cd code
python synthetic_robustness.py        # writes robustness_results.csv here
python drift_sweep.py                 # writes drift_results.csv
python unpaired_alignment.py          # needs POT (pip install pot)
python scale_eval.py
python perfect_learner_scale.py
python platonic_test.py --out results_calibrated.csv
python make_figures_v2.py ../data ../paper/figures
```

The experiment scripts write their CSV files into the folder they run from and resume if interrupted. To regenerate the figures from new runs, point `make_figures_v2.py` at the folder holding them. The figures alone need only `matplotlib` and the included data.

All experiments ran on one NVIDIA RTX 4070 Ti (12 GB). A synthetic configuration takes about two minutes per 8,000 steps. Software: Python 3.14, PyTorch 2.11 (CUDA 12.8), Transformers 5.17.

## Citation

See `CITATION.cff`. The DOI will be added when the Zenodo record is published.

## License

Code: MIT (see `LICENSE`). Paper, figures and data: CC BY 4.0.

Experiment code, analysis and drafting were done with the assistance of Claude, an AI system developed by Anthropic. The author ran all experiments and is responsible for the content.
