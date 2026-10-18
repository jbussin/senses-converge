"""
synthetic_world.py - Do two "minds" with different senses converge on the same
map of a world, when NO human data is involved at all?

The world:
  Every object is defined by hidden properties. The "eye" sees 8 of them and the
  "ear" sees 8 of them. Some are SHARED (both senses pick them up), the rest are
  PRIVATE to one sense - like how humans and dolphins perceive some of the same
  things and some things only one species can sense.

  - Eye:  each object is rendered as a 32x32 color image (a few soft blobs
          whose positions, sizes and colors depend on the object).
  - Ear:  each object is rendered as a sound (a mix of decaying tones whose
          pitches, loudness and decay depend on the object).
  How properties turn into pixels / sound is a fixed random nonlinear function,
  so neither model gets the properties handed to it.

The minds:
  An image model and a sound model, each trained ONLY on its own sense, on
  its OWN stream of fresh objects (they never see a single shared example, and
  never see each other). Each object is perceived through varying glimpses
  (shifted, brighter/dimmer, delayed, phase-scrambled...) and drifts slightly
  between glimpses, like things do over time. Each mind learns one thing:
  "these two glimpses were the same object". No labels, no text, no pairing
  across senses. The ear gets a fixed 'cochlea' (frequency breakdown) first,
  like a real ear.
  (Earlier versions that learned by copying their input, or by matching
  glimpses of frozen objects, barely converged: what the mind learns from
  matters.)

The test:
  Show both trained models the same 1024 NEW objects (as an image and as a
  sound), and measure mutual k-nearest-neighbor alignment, exactly like
  platonic_test.py. Also measure each model's alignment with the TRUE shared
  properties, which is only possible because we built the world.

Predictions if "shared world -> shared map" is right:
  1. Overlap sweep: alignment is at chance when the senses share 0 properties
     and rises as they share more.
  2. Training curve: alignment starts low (untrained) and rises as the models
     learn, even though they learn separately.
  3. Width sweep: bigger models align at least as well as small ones.

Usage:
  python synthetic_world.py --quick      # ~minutes on CPU, seconds on GPU
  python synthetic_world.py              # full run (GPU recommended)
"""
import argparse
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 8            # properties each sense sees
IMG = 32         # image size
WAV = 1024       # sound length in samples
N_BLOBS = 6
N_TONES = 6
N_TEST = 1024
DRIFT = 0.05     # how much an object changes between two glimpses
K = 10


# --------------------------------------------------------------------------
# Metric (same as platonic_test.py)
# --------------------------------------------------------------------------
def knn_mask(feats, k=K):
    f = F.normalize(feats.float(), dim=-1)
    sim = f @ f.T
    sim.fill_diagonal_(-float("inf"))
    idx = sim.topk(k, dim=1).indices
    mask = torch.zeros_like(sim)
    mask.scatter_(1, idx, 1.0)
    return mask


def mutual_knn(ma, mb, k=K):
    return ((ma * mb).sum(dim=1) / k).mean().item()


def is_degenerate(x):
    x = x.float()
    spread = (x - x.mean(0)).norm(dim=1).mean()
    return bool(spread <= 1e-4 * x.norm(dim=1).mean().clamp_min(1e-12))


def best_alignment(layers_a, layers_b, perm=None):
    masks_a = [None if is_degenerate(x) else knn_mask(x.to(DEVICE)) for x in layers_a]
    best, pair = -1.0, (None, None)
    for j, xb in enumerate(layers_b):
        if is_degenerate(xb):
            continue
        xb = xb.to(DEVICE)
        if perm is not None:
            xb = xb[perm]
        mb = knn_mask(xb)
        for i, ma in enumerate(masks_a):
            if ma is None:
                continue
            s = mutual_knn(ma, mb)
            if s > best:
                best, pair = s, (i, j)
    return best, pair


def subsample_sd(xa, xb, n_sub=30, frac=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    m = int(n * frac)
    xa, xb = xa.to(DEVICE), xb.to(DEVICE)
    out = []
    for _ in range(n_sub):
        idx = torch.randperm(n, generator=g)[:m].to(DEVICE)
        out.append(mutual_knn(knn_mask(xa[idx]), knn_mask(xb[idx])))
    return float(torch.tensor(out).std())


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------
class World:
    """n_shared of the D properties each sense sees are the same properties.

    Every time an object is perceived, the glimpse differs a little (nuisance):
      eye: the whole scene shifts position, brightness changes, sensor noise
      ear: the sound starts at a different moment, tone phases and volume
           change, sensor noise
    The object's properties stay fixed. Learning what stays constant across
    glimpses is how each mind has to discover the object."""

    def __init__(self, n_shared, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.S, self.P = n_shared, D - n_shared
        # Fixed random nonlinear "physics" for each sense
        self.v1 = (torch.randn(D, 64, generator=g) * 0.9).to(DEVICE)
        self.vb = (torch.randn(64, generator=g) * 0.5).to(DEVICE)
        self.v2 = (torch.randn(64, N_BLOBS * 6, generator=g) * 0.35).to(DEVICE)
        self.a1 = (torch.randn(D, 64, generator=g) * 0.9).to(DEVICE)
        self.ab = (torch.randn(64, generator=g) * 0.5).to(DEVICE)
        self.a2 = (torch.randn(64, N_TONES * 3, generator=g) * 0.35).to(DEVICE)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, IMG), torch.linspace(0, 1, IMG), indexing="ij")
        self.xx, self.yy = xx.to(DEVICE), yy.to(DEVICE)
        self.t = torch.linspace(0, 1, WAV, device=DEVICE)

    def sample(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        shared = torch.rand(n, self.S, generator=g)
        priv_eye = torch.rand(n, self.P, generator=g)
        priv_ear = torch.rand(n, self.P, generator=g)
        z_eye = torch.cat([shared, priv_eye], 1).to(DEVICE)
        z_ear = torch.cat([shared, priv_ear], 1).to(DEVICE)
        return shared.to(DEVICE), z_eye, z_ear

    def fresh(self, n, g, eye_side):
        """New random objects for one sense (training only)."""
        return torch.rand(n, D, generator=g, device=DEVICE)

    def _u(self, n, g, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(n, *shape, generator=g, device=DEVICE)

    def see(self, z, g):
        n = z.shape[0]
        p = torch.sigmoid(torch.tanh((z * 2 - 1) @ self.v1 + self.vb) @ self.v2)
        p = p.view(-1, N_BLOBS, 6)
        dx, dy = self._u(n, g, -0.12, 0.12, 1, 1, 1), self._u(n, g, -0.12, 0.12, 1, 1, 1)
        x = 0.15 + 0.7 * p[..., 0, None, None] + dx
        y = 0.15 + 0.7 * p[..., 1, None, None] + dy
        sig = 0.04 + 0.16 * p[..., 2, None, None]
        rgb = p[..., 3:6]
        blob = torch.exp(-((self.xx - x) ** 2 + (self.yy - y) ** 2) / (2 * sig ** 2))
        img = torch.einsum("nbhw,nbc->nchw", blob, rgb)
        img = img / img.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        img = img * self._u(n, g, 0.7, 1.3, 1, 1, 1)                  # brightness
        return img + 0.05 * torch.randn(img.shape, generator=g, device=DEVICE)

    def hear(self, z, g):
        n = z.shape[0]
        p = torch.sigmoid(torch.tanh((z * 2 - 1) @ self.a1 + self.ab) @ self.a2)
        p = p.view(-1, N_TONES, 3)
        freq = 5 + 115 * p[..., 0, None]
        amp = p[..., 1, None]
        decay = 6 * p[..., 2, None]
        delay = self._u(n, g, 0.0, 0.15, 1, 1)                          # onset time
        phase = self._u(n, g, 0.0, 6.2832, N_TONES, 1)                  # tone phases
        tt = (self.t - delay).clamp_min(0)
        on = (self.t >= delay).float()
        wave = (amp * on * torch.exp(-decay * tt) * torch.sin(6.2832 * freq * tt + phase)).sum(1)
        wave = wave / wave.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        wave = wave * self._u(n, g, 0.7, 1.3, 1)                        # volume
        wave = wave + 0.05 * torch.randn(wave.shape, generator=g, device=DEVICE)
        return wave.unsqueeze(1)


# --------------------------------------------------------------------------
# The minds: one encoder per sense, trained only on its own sense.
# Learning rule: two glimpses of the same object should map to nearby points,
# glimpses of different objects to distant points (contrastive, like SimCLR).
# No labels, no text, no pairing across senses.
# --------------------------------------------------------------------------
class EyeNet(nn.Module):
    def __init__(self, w, bott=32):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, w, 4, 2, 1), nn.GELU()),          # 16x16
            nn.Sequential(nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GELU()),      # 8x8
            nn.Sequential(nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.GELU()),  # 4x4
        ])
        self.to_z = nn.Linear(4 * w * 16, bott)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(bott, 64))

    def encode(self, x):
        feats, h = [], x
        for blk in self.enc:
            h = blk(h)
            feats.append(h.flatten(1))
        z = self.to_z(h.flatten(1))
        feats.append(z)
        return z, feats

    def forward(self, x):
        return self.head(self.encode(x)[0])


class EarNet(nn.Module):
    """Sound mind. Like a real ear, it first passes sound through a fixed
    'cochlea' (a frequency-over-time breakdown, not learned), then learns."""

    def __init__(self, w, bott=32):
        super().__init__()
        self.register_buffer("window", torch.hann_window(128))
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, w, 4, 2, 1), nn.GELU()),          # 32x16
            nn.Sequential(nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GELU()),      # 16x8
            nn.Sequential(nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.GELU()),  # 8x4
        ])
        self.to_z = nn.Linear(4 * w * 32, bott)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(bott, 64))

    def cochlea(self, wave):
        spec = torch.stft(wave.squeeze(1), n_fft=128, hop_length=32, window=self.window,
                          return_complex=True).abs()                      # (n, 65, 33)
        spec = torch.log1p(10 * spec)[:, :64, :32]                      # (n, 64, 32)
        return spec.unsqueeze(1)

    def encode(self, x):
        feats, h = [], self.cochlea(x)
        for blk in self.enc:
            h = blk(h)
            feats.append(h.flatten(1))
        z = self.to_z(h.flatten(1))
        feats.append(z)
        return z, feats

    def forward(self, x):
        return self.head(self.encode(x)[0])


def same_object_loss(p1, p2, tau=0.2):
    """Each glimpse must pick out its partner glimpse among the whole batch."""
    z = F.normalize(torch.cat([p1, p2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(-float("inf"))
    n = p1.shape[0]
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def features(model, x, bs=256):
    model.eval()
    layers = None
    for s in range(0, len(x), bs):
        _, f = model.encode(x[s:s + bs])
        f = [t.float().cpu() for t in f]
        layers = [[t] for t in f] if layers is None else [l + [t] for l, t in zip(layers, f)]
    model.train()
    return [torch.cat(l) for l in layers]


# --------------------------------------------------------------------------
# One experiment: build a world, train both minds separately, measure
# --------------------------------------------------------------------------
def run(n_shared, width, steps, checkpoints, seed=0, verbose=False, bs=256):
    world = World(n_shared, seed=seed)
    # Test objects, perceived once by each sense
    shared_t, z_eye_t, z_ear_t = world.sample(N_TEST, seed=999)
    gt = torch.Generator(device=DEVICE).manual_seed(12345)
    eye_test, ear_test = world.see(z_eye_t, gt), world.hear(z_ear_t, gt)
    shared_t = shared_t.cpu()
    perm = torch.randperm(N_TEST, generator=torch.Generator().manual_seed(0))

    torch.manual_seed(seed)
    eye, ear = EyeNet(width).to(DEVICE), EarNet(width).to(DEVICE)
    opt_e = torch.optim.Adam(eye.parameters(), lr=1e-3)
    opt_a = torch.optim.Adam(ear.parameters(), lr=1e-3)
    g = torch.Generator(device=DEVICE).manual_seed(seed + 7)

    def measure():
        fe, fa = features(eye, eye_test), features(ear, ear_test)
        real, pair = best_alignment(fe, fa)
        out = dict(alignment=real)
        out["shuffled"], _ = best_alignment(fe, fa, perm=perm)
        out["sd"] = subsample_sd(fe[pair[0]], fa[pair[1]])
        if n_shared > 0:
            out["eye_truth"], _ = best_alignment(fe, [shared_t])
            out["ear_truth"], _ = best_alignment(fa, [shared_t])
        else:
            out["eye_truth"] = out["ear_truth"] = float("nan")
        return out

    curve = []
    loss_e = loss_a = float("nan")
    for step in range(steps + 1):
        if step in checkpoints:
            m = measure()
            m.update(step=step, loss_eye=loss_e, loss_ear=loss_a)
            curve.append(m)
            if verbose:
                print(f"      step {step:5d}  alignment={m['alignment']:.3f} ±{m['sd']:.3f}"
                      f"  shuffled={m['shuffled']:.3f}")
        if step == steps:
            break
        # Fresh objects every step (a world that keeps producing new things);
        # each sense draws its OWN objects - still no shared examples.
        # Two glimpses = same object a moment apart: it has drifted slightly,
        # and the view (nuisance) differs.
        ze = world.fresh(bs, g, eye_side=True)
        za = world.fresh(bs, g, eye_side=False)
        ze2 = (ze + DRIFT * torch.randn(ze.shape, generator=g, device=DEVICE)).clamp(0, 1)
        za2 = (za + DRIFT * torch.randn(za.shape, generator=g, device=DEVICE)).clamp(0, 1)
        le = same_object_loss(eye(world.see(ze, g)), eye(world.see(ze2, g)))
        la = same_object_loss(ear(world.hear(za, g)), ear(world.hear(za2, g)))
        opt_e.zero_grad(); le.backward(); opt_e.step()
        opt_a.zero_grad(); la.backward(); opt_a.step()
        loss_e, loss_a = le.item(), la.item()
    return curve


def fmt(x):
    return "   -  " if x != x else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="synthetic_results.csv")
    ap.add_argument("--steps", type=int, default=None, help="override training steps")
    args = ap.parse_args()

    if args.quick:
        overlaps, widths, main_w, main_s = [0, 4, 8], [4, 16], 16, 4
        steps = 1500
    else:
        overlaps, widths, main_w, main_s = [0, 2, 4, 6, 8], [4, 8, 16, 32, 64], 32, 4
        steps = 8000
    if args.steps:
        steps = args.steps
    checkpoints = sorted({0, steps // 20, steps // 5, steps // 2, steps})
    print(f"Device: {DEVICE}  | training steps={steps} (fresh objects each step)  test objects={N_TEST}")
    print(f"Chance level ~ {K / N_TEST:.3f}\n")

    results = {}

    def get(s, w):
        if (s, w) not in results:
            t = time.time()
            verbose = (s, w) == (main_s, main_w)
            print(f"  shared={s}/{D}  width={w}" + ("  (training curve)" if verbose else ""))
            results[(s, w)] = run(s, w, steps, checkpoints, verbose=verbose)
            f = results[(s, w)][-1]
            print(f"    -> alignment={f['alignment']:.3f} ±{f['sd']:.3f}  "
                  f"shuffled={f['shuffled']:.3f}  ({time.time() - t:.0f}s)")
        return results[(s, w)]

    for s in overlaps:
        get(s, main_w)
    for w in widths:
        get(main_s, w)

    rows = []
    for (s, w), curve in sorted(results.items()):
        for m in curve:
            rows.append(dict(n_shared=s, width=w, step=m["step"],
                             alignment=round(m["alignment"], 4), sd=round(m["sd"], 4),
                             shuffled=round(m["shuffled"], 4),
                             eye_truth=round(m["eye_truth"], 4), ear_truth=round(m["ear_truth"], 4),
                             loss_eye=round(m["loss_eye"], 5), loss_ear=round(m["loss_ear"], 5)))
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    hdr = f"{'':14s}{'eye<->ear':>16s}{'shuffled':>10s}{'untrained':>11s}{'eye<->truth':>13s}{'ear<->truth':>13s}"

    print(f"\n1) HOW MUCH OF THE WORLD THE SENSES SHARE  (width={main_w})")
    print(hdr)
    for s in overlaps:
        c = results[(s, main_w)]
        f, u = c[-1], c[0]
        print(f"{f'shared {s}/{D}':14s}{f['alignment']:>9.3f} ±{f['sd']:.3f}{f['shuffled']:>10.3f}"
              f"{u['alignment']:>11.3f}{fmt(f['eye_truth']):>13s}{fmt(f['ear_truth']):>13s}")

    print(f"\n2) MODEL SIZE  (shared={main_s}/{D})")
    print(hdr)
    for w in widths:
        c = results[(main_s, w)]
        f, u = c[-1], c[0]
        print(f"{f'width {w}':14s}{f['alignment']:>9.3f} ±{f['sd']:.3f}{f['shuffled']:>10.3f}"
              f"{u['alignment']:>11.3f}{fmt(f['eye_truth']):>13s}{fmt(f['ear_truth']):>13s}")

    print(f"\n3) TRAINING CURVE  (shared={main_s}/{D}, width={main_w})")
    print(f"{'step':>8s}{'eye<->ear':>16s}{'eye<->truth':>13s}{'ear<->truth':>13s}")
    for m in results[(main_s, main_w)]:
        print(f"{m['step']:>8d}{m['alignment']:>9.3f} ±{m['sd']:.3f}"
              f"{fmt(m['eye_truth']):>13s}{fmt(m['ear_truth']):>13s}")

    print(f"\nFull results saved to {args.out}")
    print("\nIf 'shared world -> shared map' holds: (1) eye<->ear is at chance with 0 "
          "shared and rises with more sharing,\n(3) it rises during training from the "
          "untrained level, and both minds move toward the true structure.")


if __name__ == "__main__":
    main()
