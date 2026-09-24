#!/usr/bin/env python3
"""
Mask x loss factorial, v2 (18 Sept 2026)
========================================
Re-run of b1_objective_factorial with two changes that matter and nothing else:

  1. The prefix condition is leakage-free. Prefix positions attend within the prefix only;
     suffix positions attend to the whole prefix and causally within the suffix. The March
     mask let prefix rows attend to everything, so suffix predictions could see later suffix
     tokens through prefix representations (val loss 0.4-0.6, near-memorisation).

  2. Profiling uses survey_sigma1_v2's own estimator through its own hooks, so the numbers
     sit on the same code path as the 45 pretrained models: token-local J^T J power
     iteration, 5 WikiText-103 validation inputs x 8 positions from T/8 to T-1, 4 restarts,
     50 iterations, tol 1e-6, the condition's own attention mask at profile time, float32,
     TF32 off. The final per-block profile and all 40 samples are stored so the two-arm rule
     (R_ex0 < 0.80 and both edge thirds above the middle) can be applied.

     As a diagnostic, every model is also profiled under the causal mask regardless of its
     training condition, so the effect of the profile-time mask is visible.

Model, data, optimiser and schedule follow the grid scripts: 12-layer pre-norm transformer,
d = 512, 8 heads, GPT-2 vocabulary, WikiText-103 train tokenised once into 128-token
sequences, batch 16, AdamW lr 3e-4 wd 0.01, 500 warm-up steps, grad clip 1.0, 10,000 steps.
The last 256 sequences of the tokenisation are held out for validation.

Conditions x seeds 0-4, decisive comparison first:
  AR    causal mask, next-token loss at every position
  MLM   no mask, 15% positions corrupted (80% EOS / 10% random / 10% kept), loss at those
  CMLM  causal mask, same corruption and loss
  PLM   leakage-free prefix mask (P = 64), next-token loss on the suffix

One JSON per run in $FACT_OUT, atomic, resume-safe.  Needs survey_sigma1_v2.py beside it.
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
OUT = Path(os.environ.get("FACT_OUT", "factorial_v2"))
DATA_CACHE = Path(os.environ.get("GRID_OUT", "grid_v2")) / "_wikitext103_128.pt"

VOCAB = 50257
SEQ = 128
BATCH = 16
D_MODEL, N_LAYERS, N_HEADS = 512, 12, 8
STEPS = 10000
LR, WD, WARMUP, CLIP = 3e-4, 0.01, 500, 1.0
PREFIX = 64
MASK_PROB = 0.15
EOS = 50256
N_VAL = 256
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ["AR", "MLM", "CMLM", "PLM"]      # decisive pair first
PROFILE_AT = [2000, 5000]                        # plus final


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "", flush=True)


# ---------------------------------------------------------------- model (grid architecture)
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)        # need_weights default True: no fused fast path, matches the grid
        x = x + h
        return x + self.ff(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ, D_MODEL)
        self.layers = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.norm_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, ids, mask=None):
        T = ids.shape[1]
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device)[None])
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.head(self.norm_f(x))


# ---------------------------------------------------------------- masks (additive float, -inf = blocked)
def causal_mask(T, device):
    m = torch.zeros(T, T, device=device)
    return m.masked_fill(torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1), float("-inf"))


def prefix_mask(T, P, device):
    """Leakage-free prefix-LM mask: rows i < P attend to j < P only; rows i >= P attend causally."""
    m = torch.zeros(T, T, device=device)
    m[:P, P:] = float("-inf")
    for i in range(P, T):
        m[i, i + 1:] = float("-inf")
    return m


def mask_for(cond, T, device):
    if cond in ("AR", "CMLM"):
        return causal_mask(T, device)
    if cond == "MLM":
        return torch.zeros(T, T, device=device)   # all-attend as an explicit zero mask, so every condition takes the same attention code path
    if cond == "PLM":
        return prefix_mask(T, PREFIX, device)
    raise ValueError(cond)


# ---------------------------------------------------------------- losses
def corrupt(ids, gen):
    """BERT-style: 15% selected; of those 80% -> EOS, 10% -> random, 10% unchanged."""
    sel = torch.rand(ids.shape, generator=gen, device=ids.device) < MASK_PROB
    r = torch.rand(ids.shape, generator=gen, device=ids.device)
    x = ids.clone()
    x[sel & (r < 0.8)] = EOS
    rnd = sel & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(0, VOCAB, (int(rnd.sum()),), generator=gen, device=ids.device)
    return x, sel


def loss_for(cond, model, ids, mask, gen):
    if cond == "AR":
        logits = model(ids, mask)
        return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    if cond in ("MLM", "CMLM"):
        x, sel = corrupt(ids, gen)
        logits = model(x, mask)
        return F.cross_entropy(logits[sel], ids[sel])
    if cond == "PLM":
        logits = model(ids, mask)
        # positions P-1 .. T-2 predict tokens P .. T-1 (the suffix)
        return F.cross_entropy(logits[:, PREFIX - 1:-1].reshape(-1, VOCAB), ids[:, PREFIX:].reshape(-1))
    raise ValueError(cond)


@torch.no_grad()
def validate(cond, model, val, mask):
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(12345)
    tot, n = 0.0, 0
    for i in range(0, len(val), BATCH):
        ids = val[i:i + BATCH].to(DEVICE)
        tot += float(loss_for(cond, model, ids, mask, gen)) * len(ids); n += len(ids)
    model.train()
    return tot / n


# ---------------------------------------------------------------- survey-estimator profiling
def survey_profile(model, mask, ids, n_positions=8, restarts=4, iters=50):
    """Token-local sigma1 per block through survey_sigma1_v2's hooks and power iteration.
    Returns (profile[L], samples[L][n_inputs*n_positions], mean_iters)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = list(model.layers)
    b = dict(model=model, layers=layers, dtype=torch.float32, cls_offset=0,
             forward=lambda m, inp: m(inp["input_ids"], mask))
    L = len(layers)
    samples = [[] for _ in range(L)]
    its = []
    for i in range(ids.shape[0]):
        captured = sv.capture_layer_inputs(b, {"input_ids": ids[i:i + 1].to(DEVICE)})
        for li in range(L):
            hidden0, rest, kw = captured[li]
            layout = sv.Layout(hidden0)
            call = sv.make_layer_call(layers[li], rest, kw)
            positions = sv.choose_positions(layout, n_positions, 0, None)
            x0 = layout.get(hidden0.float(), positions)
            f = sv.make_f_incontext(call, hidden0, layout, positions, False, torch.float32)
            sig, n_it = sv.sigma1_power_iteration(f, x0, restarts, iters, sv.PI_TOL, {})
            samples[li].extend(float(s) for s in sig)
            its.append(n_it)
        del captured
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    prof = np.array([np.mean(s) for s in samples])
    return prof, samples, float(np.mean(its))


def two_arm(prof):
    p = np.asarray(prof, float)[1:]; L = len(p); n = L // 3
    e, m, l = p[:n].mean(), p[n:L - n].mean(), p[L - n:].mean()
    R = m / ((e + l) / 2)
    return dict(R_ex0=float(R), early_over_mid=float(e / m), late_over_mid=float(l / m),
                pooled_rule=bool(R < 0.80 and l / m > 1.0), two_arm_rule=bool(R < 0.80 and e > m and l > m))


# ---------------------------------------------------------------- data
def load_data():
    if DATA_CACHE.exists():
        log(f"data: {DATA_CACHE}")
        return torch.load(DATA_CACHE)
    log("tokenising wikitext-103 train (about 10 min the first time)...")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = " ".join(t for t in ds["text"] if len(t.strip()) > 50)
    ids = tok.encode(text, add_special_tokens=False)
    n = len(ids) // SEQ
    data = torch.tensor(ids[:n * SEQ]).reshape(-1, SEQ)
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, DATA_CACHE)
    return data


# ---------------------------------------------------------------- one run
def run(cond, seed, train, val, probe_ids):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(10_000 + seed)
    model = Transformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  {cond} seed {seed}: {n_params / 1e6:.1f}M params")
    mask = mask_for(cond, SEQ, DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    perm = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=perm)
    model.train()
    t0 = time.time(); step = 0; last = None; traj = {}
    while step < STEPS:
        for i in range(0, len(order) - BATCH + 1, BATCH):
            if step >= STEPS:
                break
            if step in PROFILE_AT:
                prof, _, _ = survey_profile(model, mask, probe_ids)
                s = two_arm(prof); traj[step] = s
                log(f"    step {step}: R_ex0 {s['R_ex0']:.3f}  E/M {s['early_over_mid']:.2f}  L/M {s['late_over_mid']:.2f}  loss {last:.3f}")
            ids = train[order[i:i + BATCH]].to(DEVICE)
            loss = loss_for(cond, model, ids, mask, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); sched.step()
            step += 1; last = loss.item()
            if step % 2500 == 0:
                log(f"    step {step}/{STEPS} loss {last:.3f} ({time.time() - t0:.0f} s)")
    vloss = validate(cond, model, val, mask)
    log(f"    final: train loss {last:.3f}, val loss {vloss:.3f}; profiling with the survey estimator...")
    prof, samples, mean_it = survey_profile(model, mask, probe_ids)
    s = two_arm(prof)
    prof_causal, _, _ = survey_profile(model, causal_mask(SEQ, DEVICE), probe_ids) if cond != "AR" and cond != "CMLM" else (prof, None, None)
    sc = two_arm(prof_causal)
    log(f"    -> R_ex0 {s['R_ex0']:.3f}  E/M {s['early_over_mid']:.2f}  L/M {s['late_over_mid']:.2f}  "
        f"pooled {s['pooled_rule']}  two-arm {s['two_arm_rule']}  | causal-mask profile R {sc['R_ex0']:.3f}  ({time.time() - t0:.0f} s)")
    return dict(
        condition=cond, seed=seed, mask={"AR": "causal", "CMLM": "causal", "MLM": "none (bidirectional)", "PLM": f"prefix P={PREFIX}, leakage-free"}[cond],
        loss={"AR": "next-token, all positions", "CMLM": "masked-token 15% (80/10/10)", "MLM": "masked-token 15% (80/10/10)", "PLM": "next-token on suffix"}[cond],
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, params_M=round(n_params / 1e6, 1),
        steps=STEPS, batch=BATCH, seq=SEQ, lr=LR, weight_decay=WD, warmup=WARMUP,
        final_train_loss=round(last, 4), val_loss=round(vloss, 4),
        estimator="survey_sigma1_v2 token-local J^T J power iteration, condition mask at profile time",
        survey_version=sv.VERSION, n_inputs=int(probe_ids.shape[0]), n_positions=8, restarts=4, iters=50, mean_iters=mean_it,
        final_profile=[round(float(x), 4) for x in prof], samples=[[round(float(x), 4) for x in s_] for s_ in samples],
        **s, trajectory=traj,
        causal_mask_profile=[round(float(x), 4) for x in prof_causal], causal_mask_stats=sc,
        time_s=round(time.time() - t0), gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
        torch=torch.__version__, date=datetime.datetime.now().isoformat(timespec="seconds"),
    )


def _ser(o):
    try:
        return o.item()
    except Exception:
        return str(o)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log(f"factorial v2: {len(CONDITIONS)} conditions x {len(SEEDS)} seeds -> {OUT}")
    log(f"device {DEVICE}; torch {torch.__version__}; survey {sv.VERSION}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    data = load_data()
    train, val = data[:-N_VAL], data[-N_VAL:]
    log(f"data {tuple(data.shape)}: train {len(train)}, val {len(val)}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    probe_ids = sv.natural_text_ids(tok, 5)                # the survey's own 5 WikiText-103 validation inputs
    log(f"probe inputs {tuple(probe_ids.shape)} (survey natural_text_ids)")
    log("=" * 70)
    todo = [(c, s) for s in SEEDS for c in CONDITIONS]     # seed-major so every condition gets a seed early
    todo.sort(key=lambda cs: (cs[1], CONDITIONS.index(cs[0])))
    t0 = time.time()
    for k, (cond, seed) in enumerate(todo):
        f = OUT / f"{cond}_s{seed}.json"
        if f.exists():
            try:
                d = json.load(open(f)); assert "final_profile" in d
                log(f"[{k + 1}/{len(todo)}] {cond} s{seed}: done (R {d['R_ex0']:.3f}), skipping"); continue
            except Exception:
                f.unlink()
        log(f"\n[{k + 1}/{len(todo)}] {cond} seed {seed}")
        res = run(cond, seed, train, val, probe_ids)
        tmp = f.with_suffix(".tmp")
        json.dump(res, open(tmp, "w"), indent=1, default=_ser); tmp.replace(f)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    log(f"\nall done in {(time.time() - t0) / 3600:.2f} h")
    summary()


def summary():
    rows = [json.load(open(f)) for f in sorted(OUT.glob("*_s*.json"))]
    if not rows:
        return
    print("\n%-6s%-5s%8s%7s%7s  pooled two  %8s  causal-R" % ("cond", "seed", "R_ex0", "E/M", "L/M", "val"))
    for r in sorted(rows, key=lambda r: (CONDITIONS.index(r["condition"]), r["seed"])):
        print("%-6s%-5d%8.3f%7.2f%7.2f  %4s  %3s  %8.3f  %6.3f" % (
            r["condition"], r["seed"], r["R_ex0"], r["early_over_mid"], r["late_over_mid"],
            "+" if r["pooled_rule"] else ".", "+" if r["two_arm_rule"] else ".", r["val_loss"], r["causal_mask_stats"]["R_ex0"]))
    print()
    for c in CONDITIONS:
        rs = [r for r in rows if r["condition"] == c]
        if rs:
            R = [r["R_ex0"] for r in rs]
            print("%-5s n=%d  R %.3f +/- %.3f  pooled %d/%d  two-arm %d/%d  val %.3f" % (
                c, len(rs), np.mean(R), np.std(R), sum(r["pooled_rule"] for r in rs), len(rs),
                sum(r["two_arm_rule"] for r in rs), len(rs), np.mean([r["val_loss"] for r in rs])))


if __name__ == "__main__":
    main()
