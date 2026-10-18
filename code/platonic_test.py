"""
platonic_test.py - A small test of the "Platonic Representation Hypothesis".

Question: do a TEXT model and a VISION model, trained separately on different
data, organize the same image-caption pairs in a similar way? And does that
agreement grow as the text model gets bigger?

How it works:
  1. Load ~1000 image/caption pairs.
  2. Embed each caption with a family of text LLMs (Pythia: 70M -> 2.8B params).
  3. Embed each image with vision models (DINOv2 small/base/large) that never
     saw text.
  4. For each model, build a "who is my nearest neighbor" table.
  5. Score = mutual k-nearest-neighbor overlap: for each item, what fraction of
     its 10 nearest neighbors in text-space are ALSO its 10 nearest neighbors in
     image-space. Chance level is about k/N (~0.01 here).
  6. Control: shuffle the pairing (caption i matched with a random image) and
     score again. If the real score isn't far above the shuffled one, there's
     no alignment.

Prediction if the hypothesis holds: the score goes UP as the text model gets
bigger, and stays near chance for the shuffled control.

Usage:
  python platonic_test.py --selftest          # checks the metric, no downloads
  python platonic_test.py                     # full run (GPU recommended)
  python platonic_test.py --n 512 --quick     # smaller, faster run
"""
import argparse
import csv
import os
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*decompression bomb.*", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning, module="PIL")

import torch

# Control model: same architecture as pythia-410m but random weights, never
# trained. Shows how much alignment comes from architecture alone vs learning.
UNTRAINED_CONTROL = "EleutherAI/pythia-410m"

TEXT_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
]
VISION_MODELS = [
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# --------------------------------------------------------------------------
# The metric
# --------------------------------------------------------------------------
def knn_mask(feats: torch.Tensor, k: int) -> torch.Tensor:
    """feats: [N, D]. Returns a [N, N] 0/1 mask of each row's k nearest
    neighbors by cosine similarity (excluding itself)."""
    f = torch.nn.functional.normalize(feats.float(), dim=-1)
    sim = f @ f.T
    sim.fill_diagonal_(-float("inf"))
    idx = sim.topk(k, dim=1).indices
    mask = torch.zeros_like(sim)
    mask.scatter_(1, idx, 1.0)
    return mask


def mutual_knn(mask_a: torch.Tensor, mask_b: torch.Tensor, k: int) -> float:
    """Average fraction of shared nearest neighbors between two spaces."""
    return ((mask_a * mask_b).sum(dim=1) / k).mean().item()


def is_degenerate(x: torch.Tensor) -> bool:
    """True if every item has (almost) the same vector, e.g. a ViT's CLS token
    before the first layer. Such a layer has no neighbor structure at all, and
    its tied similarities make top-k return the same arbitrary indices for every
    row, which produces a fake above-chance score."""
    x = x.float()
    spread = (x - x.mean(0)).norm(dim=1).mean()
    return bool(spread <= 1e-4 * x.norm(dim=1).mean().clamp_min(1e-12))


def best_alignment(layers_a, layers_b, k, perm=None):
    """layers_a / layers_b: lists of [N, D] features, one per layer.
    Like the original paper, we score every layer pair and keep the best.
    perm: optional shuffle applied to B's rows (the control)."""
    masks_a = [None if is_degenerate(x) else knn_mask(x.to(DEVICE), k) for x in layers_a]
    best, best_pair = -1.0, None
    for j, xb in enumerate(layers_b):
        if is_degenerate(xb):
            continue
        xb = xb.to(DEVICE)
        if perm is not None:
            xb = xb[perm]
        mb = knn_mask(xb, k)
        for i, ma in enumerate(masks_a):
            if ma is None:
                continue
            s = mutual_knn(ma, mb, k)
            if s > best:
                best, best_pair = s, (i, j)
    return best, best_pair


def calibrated(layers_a, layers_b, k, n_null=20, seed=0):
    """Permutation null for the max-over-layer-pairs score (after Groger et al.,
    ICML 2026): shuffle the pairing n_null times, redo the full layer search
    each time. Returns real score, best pair, null mean, calibrated score
    (real - null_mean) / (1 - null_mean), and a permutation p-value."""
    masks_a = [None if is_degenerate(x) else knn_mask(x.to(DEVICE), k) for x in layers_a]

    def best(layers_b_perm):
        top, pair = -1.0, None
        for j, xb in enumerate(layers_b_perm):
            if is_degenerate(xb):
                continue
            mb = knn_mask(xb.to(DEVICE), k)
            for i, ma in enumerate(masks_a):
                if ma is None:
                    continue
                v = mutual_knn(ma, mb, k)
                if v > top:
                    top, pair = v, (i, j)
        return top, pair

    real, pair = best(layers_b)
    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_null):
        perm = torch.randperm(layers_b[0].shape[0], generator=g)
        null.append(best([x[perm] for x in layers_b])[0])
    null = torch.tensor(null)
    mu = float(null.mean())
    return real, pair, mu, (real - mu) / (1 - mu), (1 + int((null >= real).sum())) / (1 + n_null)


def subsample_sd(xa, xb, k, n_sub=30, frac=0.8, seed=0):
    """Error bar: re-score the chosen layer pair on n_sub random 80% subsets
    of the items and return the standard deviation of those scores. Tells you
    how much the number would wobble with a different sample of pairs."""
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    m = int(n * frac)
    xa, xb = xa.to(DEVICE), xb.to(DEVICE)
    scores = []
    for _ in range(n_sub):
        idx = torch.randperm(n, generator=g)[:m].to(DEVICE)
        scores.append(mutual_knn(knn_mask(xa[idx], k), knn_mask(xb[idx], k), k))
    return float(torch.tensor(scores).std())


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_pairs(n):
    from datasets import load_dataset

    # Try the dataset the original paper used (Wikipedia image-text pairs),
    # then fall back to Flickr30k captions.
    attempts = [
        dict(path="minhuh/prh", revision="wit_1024", split="train"),
        dict(path="nlphuji/flickr30k", split="test", streaming=True),
    ]
    for kw in attempts:
        try:
            ds = load_dataset(**kw)
            images, texts = [], []
            for row in ds:
                img = row.get("image")
                txt = row.get("text", row.get("caption"))
                if isinstance(txt, list):
                    txt = txt[0]
                if img is None or not txt:
                    continue
                images.append(img.convert("RGB"))
                texts.append(str(txt))
                if len(images) >= n:
                    break
            print(f"Loaded {len(images)} pairs from {kw['path']}")
            return images, texts
        except Exception as e:  # noqa: BLE001
            print(f"  could not load {kw['path']}: {e}")
    raise SystemExit("No dataset could be loaded. Check your internet / HF access.")


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
@torch.no_grad()
def text_features(name, texts, bs=16, max_len=256, untrained=False):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if untrained:
        torch.manual_seed(0)
        model = AutoModel.from_config(AutoConfig.from_pretrained(name))
        model = model.to(DEVICE, DTYPE).eval()
    else:
        model = AutoModel.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())

    per_layer = None
    for s in range(0, len(texts), bs):
        batch = tok(texts[s:s + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(DEVICE)
        out = model(**batch, output_hidden_states=True)
        m = batch["attention_mask"].unsqueeze(-1).to(out.hidden_states[0].dtype)
        # mean-pool over real tokens, for every layer
        pooled = [((h * m).sum(1) / m.sum(1)).float().cpu() for h in out.hidden_states]
        if per_layer is None:
            per_layer = [[p] for p in pooled]
        else:
            for L, p in enumerate(pooled):
                per_layer[L].append(p)
    del model
    torch.cuda.empty_cache()
    return [torch.cat(x) for x in per_layer], n_params


@torch.no_grad()
def vision_features(name, images, bs=32):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())

    per_layer = None
    for s in range(0, len(images), bs):
        px = proc(images=images[s:s + bs], return_tensors="pt")["pixel_values"]
        out = model(pixel_values=px.to(DEVICE, DTYPE), output_hidden_states=True)
        cls = [h[:, 0].float().cpu() for h in out.hidden_states]  # CLS token
        if per_layer is None:
            per_layer = [[c] for c in cls]
        else:
            for L, c in enumerate(cls):
                per_layer[L].append(c)
    del model
    torch.cuda.empty_cache()
    return [torch.cat(x) for x in per_layer], n_params


# --------------------------------------------------------------------------
# Self-test: checks the metric behaves sensibly, no downloads needed
# --------------------------------------------------------------------------
def selftest(k=10, n=1000):
    g = torch.Generator().manual_seed(0)
    world = torch.randn(n, 32, generator=g)              # hidden "reality"
    # Two different "models": different random projections of the same world
    a = world @ torch.randn(32, 256, generator=g)
    b = world @ torch.randn(32, 384, generator=g)
    noise = torch.randn(n, 384, generator=g)
    unrelated = torch.randn(n, 384, generator=g)
    perm = torch.randperm(n, generator=g)

    ma = knn_mask(a, k)
    cases = {
        "same world, different models": knn_mask(b, k),
        "same world + heavy noise": knn_mask(b + b.std() * 1.5 * noise, k),
        "unrelated features": knn_mask(unrelated, k),
        "same world, shuffled pairing": knn_mask(b[perm], k),
    }
    print(f"Self-test (chance ~ {k / n:.3f}):")
    for label, mb in cases.items():
        print(f"  {label:32s} {mutual_knn(ma, mb, k):.3f}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024, help="number of image-text pairs")
    ap.add_argument("--k", type=int, default=10, help="neighbors for mutual kNN")
    ap.add_argument("--quick", action="store_true", help="fewer models")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--null", type=int, default=20, help="shuffles for the permutation null")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.k)
        return

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()  # hide harmless "UNEXPECTED key" load reports
    except Exception:  # noqa: BLE001
        pass

    names = TEXT_MODELS[:4] if args.quick else TEXT_MODELS
    # (label, hub name, untrained?)
    text_models = [(UNTRAINED_CONTROL.split("/")[-1] + " UNTRAINED", UNTRAINED_CONTROL, True)]
    text_models += [(n.split("/")[-1], n, False) for n in names]
    vision_models = VISION_MODELS[:2] if args.quick else VISION_MODELS
    print(f"Device: {DEVICE}")

    images, texts = load_pairs(args.n)
    n = len(images)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))

    vis = {}
    for vm in vision_models:
        t = time.time()
        vis[vm] = vision_features(vm, images)
        print(f"  embedded images with {vm} ({time.time() - t:.0f}s)")

    rows = []
    for label, name, untrained in text_models:
        t = time.time()
        tfeats, tparams = text_features(name, texts, untrained=untrained)
        print(f"  embedded captions with {label} ({time.time() - t:.0f}s)")
        for vm, (vfeats, vparams) in vis.items():
            real, pair, shuf, cal, pval = calibrated(tfeats, vfeats, args.k, n_null=args.null)
            sd = subsample_sd(tfeats[pair[0]], vfeats[pair[1]], args.k)
            rows.append(dict(text_model=label, text_params_M=round(tparams / 1e6),
                             vision_model=vm, vision_params_M=round(vparams / 1e6),
                             alignment=round(real, 4), alignment_sd=round(sd, 4),
                             shuffled=round(shuf, 4),
                             margin=round(real - shuf, 4),
                             calibrated=round(cal, 4), p_value=round(pval, 4),
                             n_text_layers=len(tfeats), n_vision_layers=len(vfeats),
                             best_text_layer=pair[0], best_vision_layer=pair[1]))
            print(f"    vs {vm:24s} alignment={real:.3f} ±{sd:.3f}  null={shuf:.3f}  "
                  f"calibrated={cal:.3f}  p={pval:.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary table: rows = text model, columns = vision model
    print(f"\nMutual kNN alignment: real ±error / permutation-null mean "
          f"(k={args.k}, N={n}, chance ~ {args.k / n:.3f})")
    print(f"{'text model':26s}" + "".join(f"{v.split('/')[-1]:>24s}" for v in vision_models))
    for label, _, _ in text_models:
        cells = [r for r in rows if r["text_model"] == label]
        print(f"{label:26s}" + "".join(
            f"{c['alignment']:>9.3f} ±{c['alignment_sd']:.3f} / {c['shuffled']:.3f}"
            for c in cells))
    print("\nRule of thumb: two scores differ for real if the gap is bigger than "
          "about 2x their ± values.")
    print(f"\nFull results saved to {args.out}")
    print("\nIf the hypothesis holds: real scores rise going DOWN the trained models, "
          "sit well above the shuffled control, and the UNTRAINED row stays low.")


if __name__ == "__main__":
    main()
