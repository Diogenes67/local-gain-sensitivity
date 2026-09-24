#!/usr/bin/env python3
"""
Synthetic sources with matched conditional entropy and varied required context, v2 (18 Sept 2026)
==============================================================================================
v2.1 after the first run: data are generated fresh every batch (the fixed 20,000-sequence corpus
was memorised below the entropy floor), V = 4,096 so the floor-to-ceiling range is ~6 nats
(with V = 64 whole-block rotation saturated at chance everywhere), sum-k is dropped (modular
sums of >= 3 tokens do not train in 10k steps), rotation is applied at doses 0.25 / 0.5 / 1.0
and identity skip is a co-primary readout.
The k-gram sweep varies three things at once: how far back the next token depends, how many
tokens it depends on, and the conditional entropy of the source. This experiment fixes the
alphabet and the conditional entropy exactly and varies only the required context.

Every source: alphabet V = 4,096, uniform marginals, and at each position with probability p the
next token is a deterministic function of the context, otherwise uniform over V. So the
conditional entropy given the full past is the same for every k and every family:
    H = -q log q - (V-1) r log r,  q = p + (1-p)/V,  r = (1-p)/V      (p = 0.8: 2.16 nats)
and the unigram entropy is log V = 8.32 nats. The achievable loss floor is H for every source.

Two families, both k-th order Markov:
  lag-k   x_t = x_{t-k}                         one required token, k positions back
  pair-k  x_t = (x_{t-1} + x_{t-k}) mod V       two required tokens, the farther k back
lag-k varies the distance at a fixed number of required tokens (one); pair-k adds a second
required token at fixed distance. Between them, context length and number of dependencies are
separated, with entropy fixed. k = 1 is the degenerate copy task in both (pair-1 = 2 x_{t-1}).

Per model (12-layer, d = 512, causal, same architecture/optimiser/schedule as the destroyed-data
and k-gram experiments; V = 4,096 so 42M rather than 63M parameters):
  * validation loss on 256 held-out sequences of the same source
  * interior perturbation sensitivity, primary: Haar rotation of the attention and feed-forward
    branch outputs (before the residual add) of blocks 4-7 (waist), 0-3 (early), 8-11 (late) at
    doses 0.25, 0.5 and 1.0, the intervention of the original d1/k-gram/census experiments;
    secondary: whole-block rotation at the same doses; dL in nats over 50 x 16 fresh sequences
  * per-block single branch rotation (dose 1) and per-block identity skip (12 + 12 passes)
  * sigma1 profile by survey_sigma1_v2's estimator (5 inputs x 8 positions, causal mask),
    pooled and two-arm rules
  * early-exit loss per block (final norm + head applied at each block)

Run order: lag family first, seed-major. One JSON per run in $SYN_OUT, resume-safe.
Needs survey_sigma1_v2.py beside it.
"""

import os, sys, json, time, math, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(os.environ.get("SYN_OUT", "synthetic_context"))

V = 4096
P_DET = 0.8
SEQ = 128
BATCH = 16
D_MODEL, N_LAYERS, N_HEADS = 512, 12, 8
STEPS = 10000
LR, WD, WARMUP, CLIP = 3e-4, 0.01, 500, 1.0
N_VAL_BATCHES, N_EVAL_BATCHES = 16, 50
SEEDS = [0, 1, 2, 3, 4]
FAMILIES = os.environ.get("SYN_FAMILIES", "lag pair").split()
KS = {"lag": [1, 2, 4, 8, 16, 32], "pair": [1, 2, 4, 8, 16, 32]}
DOSES = [0.25, 0.5, 1.0]
WAIST, EARLY, LATE = [4, 5, 6, 7], [0, 1, 2, 3], [8, 9, 10, 11]


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "", flush=True)


def cond_entropy(p=P_DET, v=V):
    q = p + (1 - p) / v; r = (1 - p) / v
    return float(-q * math.log(q) - (v - 1) * r * math.log(r))


# ---------------------------------------------------------------- sources
def generate(family, k, n_seqs, seed):
    """(n_seqs, SEQ) long tensor. Positions < k are uniform; after that the rule holds w.p. P_DET."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (n_seqs, SEQ), generator=g)
    det = torch.rand(n_seqs, SEQ, generator=g) < P_DET
    for t in range(k, SEQ):
        if family == "lag":
            target = x[:, t - k]
        elif family == "pair":
            target = (x[:, t - 1] + x[:, t - k]) % V
        else:
            raise ValueError(family)
        x[:, t] = torch.where(det[:, t], target, x[:, t])
    return x


# ---------------------------------------------------------------- model (grid architecture, V = 64)
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        return x + self.ff(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ, D_MODEL)
        self.layers = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.norm_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, V, bias=False)

    def embed(self, ids):
        T = ids.shape[1]
        return self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device)[None])

    def forward(self, ids, mask=None):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.head(self.norm_f(x))


def causal_mask(T, device):
    m = torch.zeros(T, T, device=device)
    return m.masked_fill(torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1), float("-inf"))


def nt_loss(logits, ids):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))


# ---------------------------------------------------------------- perturbations
def haar(d, seed):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=g))
    return Q * torch.sign(torch.diag(R))[None, :]


class Hooks:
    def __init__(self): self.h = []
    def add(self, x): self.h.append(x)
    def clear(self):
        for x in self.h: x.remove()
        self.h.clear()


@torch.no_grad()
def eval_loss(model, data, mask, setup=None):
    """Mean next-token loss over N_EVAL_BATCHES batches; per-batch list returned for the bootstrap."""
    model.eval(); hk = Hooks()
    try:
        if setup: setup(hk)
        out = []
        for b in range(len(data) // BATCH):
            ids = data[b * BATCH:(b + 1) * BATCH].to(DEVICE)
            out.append(nt_loss(model(ids, mask), ids).item())
        return out
    finally:
        hk.clear(); model.train()


def dosed(Q, dose):
    """Interpolate between I and Q and re-orthogonalise (QR), as in the d1 low-dose variants."""
    if dose >= 1.0:
        return Q
    M = (1 - dose) * torch.eye(Q.shape[0]) + dose * Q
    Qd, R = torch.linalg.qr(M)
    return Qd * torch.sign(torch.diag(R))[None, :]


def rotate_blocks(model, blocks, Qs, dose=1.0):
    """Whole-block rotation Q[h + F(h)] (secondary readout; saturates in trained models)."""
    def setup(hk):
        for li in blocks:
            Q = dosed(Qs[li], dose).to(DEVICE)
            hk.add(model.layers[li].register_forward_hook(lambda m, a, o, Q=Q: o @ Q.T))
    return setup


def rotate_branches(model, blocks, Qa, Qf, dose=1.0):
    """Branch rotation h + Q F(h): the attention output and the feed-forward output of each block are
    rotated before the residual add. This is the intervention of the original destroyed-data, k-gram
    and census experiments (d3_ngram_sweep.forward_with_intervention rotates attn_out and mlp_out)."""
    def setup(hk):
        for li in blocks:
            A = dosed(Qa[li], dose).to(DEVICE); Fq = dosed(Qf[li], dose).to(DEVICE)
            hk.add(model.layers[li].attn.register_forward_hook(lambda m, a, o, Q=A: (o[0] @ Q.T,) + tuple(o[1:])))
            hk.add(model.layers[li].ff.register_forward_hook(lambda m, a, o, Q=Fq: o @ Q.T))
    return setup


def skip_block(model, li):
    def setup(hk):
        hk.add(model.layers[li].register_forward_hook(lambda m, a, o: a[0]))
    return setup


def skip_blocks(model, blocks):
    def setup(hk):
        for li in blocks:
            hk.add(model.layers[li].register_forward_hook(lambda m, a, o: a[0]))
    return setup


@torch.no_grad()
def early_exit(model, data, mask):
    """Loss when norm_f + head are applied at the output of each block."""
    model.eval()
    tot = np.zeros(N_LAYERS)
    nb = min(10, N_EVAL_BATCHES)
    for b in range(nb):
        ids = data[b * BATCH:(b + 1) * BATCH].to(DEVICE)
        x = model.embed(ids)
        for li, layer in enumerate(model.layers):
            x = layer(x, mask=mask)
            tot[li] += nt_loss(model.head(model.norm_f(x)), ids).item()
    model.train()
    return (tot / nb).tolist()


def survey_profile(model, mask, ids):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    layers = list(model.layers)
    b = dict(model=model, layers=layers, dtype=torch.float32, cls_offset=0, forward=lambda m, inp: m(inp["input_ids"], mask))
    samples = [[] for _ in range(N_LAYERS)]
    for i in range(ids.shape[0]):
        cap = sv.capture_layer_inputs(b, {"input_ids": ids[i:i + 1].to(DEVICE)})
        for li in range(N_LAYERS):
            h0, rest, kw = cap[li]; lay = sv.Layout(h0)
            call = sv.make_layer_call(layers[li], rest, kw)
            pos = sv.choose_positions(lay, 8, 0, None)
            f = sv.make_f_incontext(call, h0, lay, pos, False, torch.float32)
            sig, _ = sv.sigma1_power_iteration(f, lay.get(h0.float(), pos), 4, 50, sv.PI_TOL, {})
            samples[li].extend(float(s) for s in sig)
    for p in model.parameters(): p.requires_grad_(True)
    model.train()
    prof = np.array([np.mean(s) for s in samples])
    return prof, samples


def two_arm(prof):
    p = np.asarray(prof, float)[1:]; L = len(p); n = L // 3
    e, m, l = p[:n].mean(), p[n:L - n].mean(), p[L - n:].mean(); R = m / ((e + l) / 2)
    return dict(R_ex0=float(R), early_over_mid=float(e / m), late_over_mid=float(l / m),
                pooled_rule=bool(R < 0.80 and l / m > 1), two_arm_rule=bool(R < 0.80 and e > m and l > m))


def region_dL(losses, base):
    a, b = np.array(losses), np.array(base); d = float(a.mean() - b.mean())
    rng = np.random.default_rng(0); n = len(a)
    boots = [a[i].mean() - b[i].mean() for i in (rng.integers(0, n, n) for _ in range(500))]
    return dict(dL=d, ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


# ---------------------------------------------------------------- one run
def run(family, k, seed):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    base_seed = 100000 + 1000 * KS[family].index(k) + seed + (0 if family == "lag" else 50000)
    val = generate(family, k, N_VAL_BATCHES * BATCH, base_seed + 1)
    evald = generate(family, k, N_EVAL_BATCHES * BATCH, base_seed + 2)
    model = Transformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    mask = causal_mask(SEQ, DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    log(f"  {family}-{k} seed {seed}: {n_params / 1e6:.1f}M params, H_cond {cond_entropy():.3f} nats, fresh data every batch")
    model.train(); last = None
    for step in range(STEPS):
        ids = generate(family, k, BATCH, base_seed + 10 + step).to(DEVICE)      # never repeats
        loss = nt_loss(model(ids, mask), ids)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step(); sched.step()
        last = loss.item()
        if (step + 1) % 2500 == 0:
            log(f"    step {step + 1}/{STEPS} loss {last:.3f} ({time.time() - t0:.0f} s)")
    vloss = float(np.mean(eval_loss(model, val, mask)[:N_VAL_BATCHES]))
    base = eval_loss(model, evald, mask)
    Qs = {li: haar(D_MODEL, 7000 + 13 * li + seed) for li in range(N_LAYERS)}
    Qa = {li: haar(D_MODEL, 7100 + 13 * li + seed) for li in range(N_LAYERS)}
    Qf = {li: haar(D_MODEL, 7200 + 13 * li + seed) for li in range(N_LAYERS)}
    # primary: branch rotation at three doses; secondary: whole-block rotation at three doses
    regions = {f"{name}@{d}": region_dL(eval_loss(model, evald, mask, rotate_branches(model, blk, Qa, Qf, d)), base)
               for name, blk in (("waist", WAIST), ("early", EARLY), ("late", LATE)) for d in DOSES}
    block_regions = {f"{name}@{d}": region_dL(eval_loss(model, evald, mask, rotate_blocks(model, blk, Qs, d)), base)
                     for name, blk in (("waist", WAIST), ("early", EARLY), ("late", LATE)) for d in DOSES}
    for name in ("waist", "early", "late"):
        regions[name] = regions[f"{name}@1.0"]; block_regions[name] = block_regions[f"{name}@1.0"]
    skip_regions = {name: region_dL(eval_loss(model, evald, mask, skip_blocks(model, blk)), base)
                    for name, blk in (("waist", WAIST), ("early", EARLY), ("late", LATE))}
    per_block = [region_dL(eval_loss(model, evald, mask, rotate_branches(model, [li], Qa, Qf)), base)["dL"] for li in range(N_LAYERS)]
    per_skip = [region_dL(eval_loss(model, evald, mask, skip_block(model, li)), base)["dL"] for li in range(N_LAYERS)]
    ee = early_exit(model, evald, mask)
    prof, samples = survey_profile(model, mask, evald[:5])
    s = two_arm(prof)
    edge = max(regions["early"]["dL"], regions["late"]["dL"], 1e-9)
    sedge = max(skip_regions["early"]["dL"], skip_regions["late"]["dL"], 1e-9)
    log(f"    -> val {vloss:.3f} (floor {cond_entropy():.3f}, ceiling {math.log(V):.3f})  branch@1 waist {regions['waist']['dL']:.3f} early {regions['early']['dL']:.3f} late {regions['late']['dL']:.3f}"
        f"  | branch@0.25 waist {regions['waist@0.25']['dL']:.3f} early {regions['early@0.25']['dL']:.3f} late {regions['late@0.25']['dL']:.3f}"
        f"  | block@1 waist {block_regions['waist']['dL']:.3f} early {block_regions['early']['dL']:.3f} late {block_regions['late']['dL']:.3f}"
        f"  | skip waist {skip_regions['waist']['dL']:.3f} early {skip_regions['early']['dL']:.3f} late {skip_regions['late']['dL']:.3f} (w/e {skip_regions['waist']['dL'] / sedge:.2f})"
        f"  | R_ex0 {s['R_ex0']:.3f} E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f}  ({time.time() - t0:.0f} s)")
    return dict(family=family, k=k, seed=seed, V=V, p_det=P_DET, cond_entropy=cond_entropy(), unigram_entropy=math.log(V),
                required_tokens=1 if family == "lag" else 2, required_distance=k, doses=DOSES,
                n_layers=N_LAYERS, d_model=D_MODEL, params_M=round(n_params / 1e6, 1), steps=STEPS, batch=BATCH, seq=SEQ,
                lr=LR, weight_decay=WD, warmup=WARMUP, data="fresh every batch", final_train_loss=round(last, 4), val_loss=round(vloss, 4),
                baseline_eval_loss=float(np.mean(base)), regions=regions, waist_over_edge=regions["waist"]["dL"] / edge,
                skip_regions=skip_regions, skip_waist_over_edge=skip_regions["waist"]["dL"] / sedge,
                block_regions=block_regions, block_waist_over_edge=block_regions["waist"]["dL"] / max(block_regions["early"]["dL"], block_regions["late"]["dL"], 1e-9),
                per_block_rotation_dL=per_block, per_block_skip_dL=per_skip, early_exit_loss=ee,
                sigma1_profile=[round(float(x), 4) for x in prof], sigma1_samples=[[round(float(x), 4) for x in q] for q in samples], **s,
                intervention="primary: Haar rotation of attention and feed-forward branch outputs before the residual add, doses 0.25/0.5/1 (interpolated, QR re-orthogonalised); secondary: whole-block rotation, same doses; identity skip; survey_sigma1_v2 estimator, causal mask",
                survey_version=sv.VERSION, time_s=round(time.time() - t0),
                gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", torch=torch.__version__,
                date=datetime.datetime.now().isoformat(timespec="seconds"))


def _ser(o):
    try: return o.item()
    except Exception: return str(o)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(f, k, s) for s in SEEDS for f in FAMILIES for k in KS[f]]
    log("=" * 70); log(f"synthetic context: families {FAMILIES}, {len(todo)} runs -> {OUT}; device {DEVICE}; H_cond {cond_entropy():.3f} nats, log V {math.log(V):.3f}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    log("=" * 70)
    t0 = time.time()
    for i, (f, k, s) in enumerate(todo):
        path = OUT / f"{f}{k}_s{s}.json"
        if path.exists():
            try:
                d = json.load(open(path)); assert "sigma1_profile" in d
                log(f"[{i + 1}/{len(todo)}] {f}-{k} s{s}: done (waist dL {d['regions']['waist']['dL']:.3f}), skipping"); continue
            except Exception:
                path.unlink()
        log(f"\n[{i + 1}/{len(todo)}] {f}-{k} seed {s}")
        res = run(f, k, s)
        tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1, default=_ser); tmp.replace(path)
        if DEVICE == "cuda": torch.cuda.empty_cache()
    log(f"\nall done in {(time.time() - t0) / 3600:.2f} h")
    summary()


def summary():
    rows = [json.load(open(f)) for f in sorted(OUT.glob("*_s*.json"))]
    if not rows: return
    print("\n%-5s%4s%3s %7s | %7s %7s %6s | %7s %6s | %7s %6s | %6s %5s" % ("fam", "k", "n", "val", "rot1 w", "rot1 e", "w/e", "rot.25w", "w/e", "skip w", "w/e", "R_ex0", "two"))
    for f in FAMILIES:
        for k in KS[f]:
            rs = [r for r in rows if r["family"] == f and r["k"] == k]
            if not rs: continue
            g = lambda key: np.mean([r[key] for r in rs]); rg = lambda key: np.mean([r["regions"][key]["dL"] for r in rs])
            re25 = max(rg("early@0.25"), rg("late@0.25"), 1e-9)
            print("%-5s%4d%3d %7.3f | %7.3f %7.3f %6.2f | %7.3f %6.2f | %7.3f %6.2f | %6.3f %3d/%d" % (
                f, k, len(rs), g("val_loss"), rg("waist"), rg("early"), g("waist_over_edge"), rg("waist@0.25"), rg("waist@0.25") / re25,
                np.mean([r["skip_regions"]["waist"]["dL"] for r in rs]), g("skip_waist_over_edge"), g("R_ex0"), sum(r["two_arm_rule"] for r in rs), len(rs)))
    print(f"floor {cond_entropy():.3f} nats, ceiling {math.log(V):.3f} nats, for every source")


if __name__ == "__main__":
    main()
