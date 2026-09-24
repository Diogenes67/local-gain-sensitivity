#!/usr/bin/env python3
"""
Re-profile the topology, residual-alpha and k-gram experiments with the canonical estimator
==========================================================================================
The March scripts (b2_topology_intervention.py, e15_bypass_alpha_sweep.py, d3_ngram_sweep.py)
profiled with a jvp-only iteration on the whole-sequence block Jacobian at uniform-random token
ids and without the attention mask, which estimates the spectral radius, not sigma_1. Their
checkpoints were not kept. This script retrains the same models (same architecture, corpus,
optimiser, schedule and seeds as the originals) and measures

  geometry     token-local sigma_1 per block by survey_sigma1_v2 (J^T J power iteration, causal
               mask, 5 inputs x 8 positions, 4 restarts) on natural WikiText text AND on
               in-distribution text; R_ex0, two-arm rule, original C/U-fit; at init, 5k, 10k
  sensitivity  branch rotation (attention and feed-forward outputs before the residual add, the
               d1/k-gram/census intervention) by fixed region (early 0-3, waist 4-7, late 8-11) at
               doses 0.25/0.5/1 (linear interpolation as in d3) and by the original fitted-waist
               regions; per-block branch rotation; per-block identity skip; skip by region
  early exit   final norm + tied head applied at every block output

Experiments (REPRO_EXP or --exp):
  topology  6 conditions x 3 seeds (baseline, no_residual, gated_middle, highway, multi_highway,
            unet_skip), WikiText-103 20k-sequence corpus, AdamW wd 0.1 betas (0.9, 0.95), cosine
  alpha     alpha in {0, 0.25, 0.5, 0.75, 1} x 3 seeds, same training as topology
  kgram     k in {1, 2, 3, 4, 5, 8} x 5 seeds, k-gram back-off sources fitted to the same corpus
            (top-200 successors per context), AdamW wd 0.01, cosine; held-out sequences for eval

Every run writes one JSON atomically, resume-safe; fp16 checkpoints in <OUT>/ckpt; results zip
written at the end. Needs survey_sigma1_v2.py beside it.
"""

import os, sys, json, time, math, gc, datetime, argparse, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(os.environ.get("REPRO_OUT", "reprofile_v2"))
SEED_BASE = 42
VOCAB_SIZE = 50257
CFG = dict(n_layers=12, d_model=512, n_heads=8, d_ff=2048, seq_len=128, batch_size=16, n_steps=10000, lr=3e-4, warmup_steps=500)
PROFILE_AT = [5000]
EARLY, WAIST, LATE = [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
DOSES = [0.25, 0.5, 1.0]
N_EVAL_BATCHES, EVAL_BATCH = 50, 16
N_BOOT = 500
TOPOLOGIES = ["baseline", "no_residual", "gated_middle", "highway", "multi_highway", "unet_skip"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_VALUES = [1, 2, 3, 4, 5, 8]
N_TRAIN_SEQS, TOP_K_NGRAM = 20000, 200
CONTRAST_WAIST_W, BOUNDARY_CUT = 0.20, 0.15


def log(msg=""):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================ models (verbatim from the March scripts)
class B2Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, residual_scale=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self._gate_param = None
        if residual_scale is None or (not isinstance(residual_scale, nn.Parameter) and residual_scale == 1.0):
            self._residual_mode = "standard"
        elif not isinstance(residual_scale, nn.Parameter) and residual_scale == 0.0:
            self._residual_mode = "none"
        elif isinstance(residual_scale, nn.Parameter):
            self._residual_mode = "gated"; self._gate_param = residual_scale
        else:
            self._residual_mode = "fixed"; self.register_buffer("_fixed_scale", torch.tensor(float(residual_scale)))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        if self._residual_mode == "none": x2 = attn_out
        elif self._residual_mode == "gated": x2 = torch.sigmoid(self._gate_param) * x + attn_out
        elif self._residual_mode == "fixed": x2 = self._fixed_scale * x + attn_out
        else: x2 = x + attn_out
        h = self.ln2(x2)
        mlp_out = self.mlp(h)
        if self._residual_mode == "none": return mlp_out
        elif self._residual_mode == "gated": return torch.sigmoid(self._gate_param) * x2 + mlp_out
        elif self._residual_mode == "fixed": return self._fixed_scale * x2 + mlp_out
        return x2 + mlp_out


class B2Transformer(nn.Module):
    def __init__(self, topology="baseline"):
        super().__init__()
        cfg = CFG; self.cfg = cfg; self.topology = topology
        d, n_layers = cfg["d_model"], cfg["n_layers"]
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d); self.pos_emb = nn.Embedding(cfg["seq_len"], d); self.drop = nn.Dropout(0.1)
        blocks = []; self.gate_params = nn.ParameterList()
        for i in range(n_layers):
            if topology == "no_residual":
                block = B2Block(d, cfg["n_heads"], cfg["d_ff"], residual_scale=0.0)
            elif topology == "gated_middle" and 4 <= i <= 8:
                gate = nn.Parameter(torch.tensor(3.0)); self.gate_params.append(gate)
                block = B2Block(d, cfg["n_heads"], cfg["d_ff"], residual_scale=gate)
            else:
                block = B2Block(d, cfg["n_heads"], cfg["d_ff"])
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False); self.head.weight = self.tok_emb.weight
        if topology == "highway":
            self.highway_proj = nn.Linear(d, d, bias=False); nn.init.zeros_(self.highway_proj.weight)
        if topology == "multi_highway":
            self.mh_projs = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in ("1_7", "3_9", "5_11")})
            for p in self.mh_projs.values(): nn.init.zeros_(p.weight)
        if topology == "unet_skip":
            self.unet_projs = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in ("1_10", "2_9", "3_8")})
            for p in self.unet_projs.values(): nn.init.zeros_(p.weight)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            skip_projs = set()
            if hasattr(self, "highway_proj"): skip_projs.add(self.highway_proj)
            if hasattr(self, "mh_projs"): skip_projs.update(self.mh_projs.values())
            if hasattr(self, "unet_projs"): skip_projs.update(self.unet_projs.values())
            if module in skip_projs: return
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        if self.topology in ("highway", "multi_highway", "unet_skip"):
            lo = {}
            for i, block in enumerate(self.blocks):
                if self.topology == "highway":
                    if i == 9 and 2 in lo: x = x + self.highway_proj(lo[2])
                elif self.topology == "multi_highway":
                    if i == 7 and 1 in lo: x = x + self.mh_projs["1_7"](lo[1])
                    if i == 9 and 3 in lo: x = x + self.mh_projs["3_9"](lo[3])
                    if i == 11 and 5 in lo: x = x + self.mh_projs["5_11"](lo[5])
                elif self.topology == "unet_skip":
                    if i == 8 and 3 in lo: x = x + self.unet_projs["3_8"](lo[3])
                    if i == 9 and 2 in lo: x = x + self.unet_projs["2_9"](lo[2])
                    if i == 10 and 1 in lo: x = x + self.unet_projs["1_10"](lo[1])
                if self.topology == "highway" and i == 2: lo[2] = x.clone()
                elif self.topology == "multi_highway" and i in (1, 3, 5): lo[i] = x.clone()
                elif self.topology == "unet_skip" and i in (1, 2, 3): lo[i] = x.clone()
                x = block(x, attn_mask=attn_mask)
        else:
            for block in self.blocks:
                x = block(x, attn_mask=attn_mask)
        return self.head(self.ln_f(x))

    def get_gate_values(self):
        return [float(torch.sigmoid(g).item()) for g in self.gate_params] if self.topology == "gated_middle" else None


class AlphaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.ln1 = nn.LayerNorm(d_model); self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model); self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x); attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask); x2 = self.alpha * x + attn_out
        h = self.ln2(x2); return self.alpha * x2 + self.mlp(h)


class AlphaTransformer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        cfg = CFG; self.cfg = cfg; self.alpha = alpha; d = cfg["d_model"]
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d); self.pos_emb = nn.Embedding(cfg["seq_len"], d); self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([AlphaBlock(d, cfg["n_heads"], cfg["d_ff"], alpha=alpha) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB_SIZE, bias=False); self.head.weight = self.tok_emb.weight
        self.apply(_init_plain)

    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device).unsqueeze(0)))
        for block in self.blocks: x = block(x, attn_mask=attn_mask)
        return self.head(self.ln_f(x))


def _init_plain(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)


class D3Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model); self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x); attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask); x = x + attn_out
        h = self.ln2(x); return x + self.mlp(h)


class D3Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = CFG; self.cfg = cfg; d = cfg["d_model"]
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d); self.pos_emb = nn.Embedding(cfg["seq_len"], d); self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([D3Block(d, cfg["n_heads"], cfg["d_ff"]) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(VOCAB_SIZE, d, bias=False); self.head.weight = self.tok_emb.weight
        self.apply(_init_plain)

    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device).unsqueeze(0)))
        for block in self.blocks: x = block(x, attn_mask=attn_mask)
        return F.linear(self.ln_f(x), self.tok_emb.weight)


def head_logits(model, h):
    return F.linear(model.ln_f(h), model.tok_emb.weight)


# ================================================================ data
def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


def load_wikitext_corpus():
    """The March corpus: first 20,000 WikiText-103 train texts (>100 chars) that tokenise to >= 128
    tokens, truncated to 128; validation from WikiText-2. Cached under ROOT/data."""
    cache = ROOT / "data" / "wikitext_20k_128.pt"
    if cache.exists():
        d = torch.load(cache); log(f"corpus cache: {cache} ({len(d['train'])} train, {len(d['val'])} val)"); return d["train"], d["val"]
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    log("tokenising the March corpus (a few minutes the first time)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    train = []
    for t in ds["text"]:
        if len(t.strip()) <= 100: continue
        ids = tok(t, truncation=True, max_length=CFG["seq_len"], return_tensors="pt").input_ids[0]
        if len(ids) == CFG["seq_len"]: train.append(ids)
        if len(train) >= N_TRAIN_SEQS: break
    vds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    val = []
    for t in vds["text"]:
        if len(t.strip()) > 50:
            ids = tok(t, truncation=True, max_length=CFG["seq_len"], return_tensors="pt").input_ids[0]
            if len(ids) == CFG["seq_len"]: val.append(ids)
    train, val = torch.stack(train), torch.stack(val)
    cache.parent.mkdir(parents=True, exist_ok=True); torch.save(dict(train=train, val=val), cache)
    log(f"  train {tuple(train.shape)}, val {tuple(val.shape)}")
    return train, val


def build_ngram_models(all_ids):
    """k-gram tables as in d3_ngram_sweep.build_ngram_model: top-200 successors per context, stored as
    (token array, cumulative-probability array) for fast sampling."""
    models = {}
    for k in sorted(set(K_VALUES)):
        if k == 1: models[1] = {}; continue
        c = k - 1
        log(f"  building {k}-gram table (context {c})...")
        counts = defaultdict(lambda: defaultdict(int))
        for i in range(c, len(all_ids)):
            counts[tuple(all_ids[i - c:i])][all_ids[i]] += 1
        table = {}
        for ctx, nxt in counts.items():
            top = sorted(nxt.items(), key=lambda x: -x[1])[:TOP_K_NGRAM]
            tot = sum(v for _, v in top)
            table[ctx] = (np.array([t for t, _ in top], dtype=np.int64), np.cumsum([v / tot for _, v in top]))
        models[k] = table
        log(f"    {len(table)} contexts")
        del counts
    return models


def generate_kgram(k, models, unigram_cum, n_seqs, seq_len, seed):
    """d3_ngram_sweep.generate_kgram_sequences: back-off k -> k-1 -> ... -> unigram; first k-1 tokens
    from the highest-order table that fits."""
    rng = np.random.RandomState(seed)
    out = np.zeros((n_seqs, seq_len), dtype=np.int64)
    U = rng.random_sample((n_seqs, seq_len))
    for s in range(n_seqs):
        seq = []
        for pos in range(seq_len):
            u = U[s, pos]; tok = None
            hi = min(pos + 1, k) if pos < k - 1 else k
            for bk in range(hi, 0, -1):
                if bk == 1:
                    tok = int(np.searchsorted(unigram_cum, u)); break
                c = bk - 1
                if len(seq) >= c and bk in models:
                    ent = models[bk].get(tuple(seq[-c:]))
                    if ent is not None:
                        toks, cum = ent; tok = int(toks[min(np.searchsorted(cum, u), len(toks) - 1)]); break
            if tok is None: tok = int(np.searchsorted(unigram_cum, u))
            seq.append(min(tok, VOCAB_SIZE - 1))
        out[s] = seq
    return torch.from_numpy(out)


def kgram_data(k, seed, corpus, models, unigram_cum):
    cache = ROOT / "data" / f"kgram_k{k}_s{seed}.pt"
    if cache.exists(): return torch.load(cache)
    t0 = time.time()
    train = generate_kgram(k, models, unigram_cum, N_TRAIN_SEQS, CFG["seq_len"], seed=k * 1000 + seed)
    held = generate_kgram(k, models, unigram_cum, N_EVAL_BATCHES * EVAL_BATCH, CFG["seq_len"], seed=k * 1000 + seed + 777)
    d = dict(train=train, held=held); cache.parent.mkdir(parents=True, exist_ok=True); torch.save(d, cache)
    log(f"  generated {k}-gram data seed {seed} ({time.time() - t0:.0f} s)")
    return d


# ================================================================ measurement
def survey_profile(model, mask, ids, n_positions=8, restarts=4, iters=50):
    model.eval()
    # parameters keep requires_grad=True: with it off, nn.MultiheadAttention takes its fused fast path,
    # which forward-mode AD does not support, and the estimator falls back to the slower double-backward JVP
    layers = list(model.blocks)
    b = dict(model=model, layers=layers, dtype=torch.float32, cls_offset=0, forward=lambda m, inp: m(inp["input_ids"], attn_mask=mask))
    L = len(layers); samples = [[] for _ in range(L)]; its = []
    for i in range(ids.shape[0]):
        captured = sv.capture_layer_inputs(b, {"input_ids": ids[i:i + 1].to(DEVICE)})
        for li in range(L):
            hidden0, rest, kw = captured[li]
            layout = sv.Layout(hidden0); call = sv.make_layer_call(layers[li], rest, kw)
            positions = sv.choose_positions(layout, n_positions, 0, None)
            x0 = layout.get(hidden0.float(), positions)
            f = sv.make_f_incontext(call, hidden0, layout, positions, False, torch.float32)
            sig, n_it = sv.sigma1_power_iteration(f, x0, restarts, iters, sv.PI_TOL, {})
            samples[li].extend(float(s) for s in sig); its.append(n_it)
        del captured
    model.train()
    return np.array([np.mean(s) for s in samples]), samples, float(np.mean(its))


def two_arm(prof):
    p = np.asarray(prof, float)[1:]; L = len(p); n = L // 3
    e, m, l = p[:n].mean(), p[n:L - n].mean(), p[L - n:].mean(); R = m / ((e + l) / 2)
    pf = np.asarray(prof, float); nf = len(pf) // 3
    C = float((np.concatenate([pf[:nf], pf[-nf:]]).mean() - pf[nf:-nf].mean()) / pf.mean())
    return dict(R_ex0=float(R), early_over_mid=float(e / m), late_over_mid=float(l / m),
                pooled_rule=bool(R < 0.80 and l > m), two_arm_rule=bool(R < 0.80 and e > m and l > m), contrast_C=C)


def fit_ushape(profile):
    from scipy import stats
    prof = np.array(profile); n = len(prof); lp = np.log(prof + 1e-12); d = np.linspace(0, 1, n)
    X = np.column_stack([d ** 2, d, np.ones(n)]); beta = np.linalg.lstsq(X, lp, rcond=None)[0]; a = beta[0]
    d_star = np.clip(-beta[1] / (2 * a), 0.0, 1.0) if a > 0 else 0.5
    wm = (np.abs(d - d_star) <= CONTRAST_WAIST_W) & (np.arange(n) > 0)
    im, om = d <= BOUNDARY_CUT, d >= 1 - BOUNDARY_CUT
    med = lambda m: float(np.median(lp[m])) if m.sum() > 0 else 0.0
    dE, dT = med(im) - med(wm), med(om) - med(wm)
    waist = [i for i in range(n) if wm[i]]
    w = len(waist)
    early = [i for i in range(1, n) if waist and i < min(waist)][-w:] if w else []
    late = [i for i in range(n) if waist and i > max(waist)][:w] if w else []
    return dict(a=float(a), d_star=float(d_star), delta_E=float(dE), delta_T=float(dT), C_fit=float(min(dE, dT)), waist=waist, early=early, late=late)


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


def rotate_branches(model, blocks, Qa, Qf, dose=1.0):
    """d3.forward_with_intervention: u <- (1 - dose) u + dose (u Q^T) on attention and MLP outputs."""
    def setup(hk):
        for li in blocks:
            A = Qa[li].to(DEVICE); Fq = Qf[li].to(DEVICE)
            hk.add(model.blocks[li].attn.register_forward_hook(lambda m, a, o, Q=A: ((1 - dose) * o[0] + dose * (o[0] @ Q.T),) + tuple(o[1:])))
            hk.add(model.blocks[li].mlp.register_forward_hook(lambda m, a, o, Q=Fq: (1 - dose) * o + dose * (o @ Q.T)))
    return setup


def skip_blocks(model, blocks):
    def setup(hk):
        for li in blocks:
            hk.add(model.blocks[li].register_forward_hook(lambda m, a, o: a[0]))
    return setup


@torch.no_grad()
def eval_losses(model, data, setup=None):
    model.eval(); model.drop.p = 0.0; hk = Hooks(); mask = causal_mask(CFG["seq_len"], DEVICE)
    try:
        if setup: setup(hk)
        out = []
        for b in range(min(N_EVAL_BATCHES, len(data) // EVAL_BATCH)):
            ids = data[b * EVAL_BATCH:(b + 1) * EVAL_BATCH].to(DEVICE)
            logits = model(ids, attn_mask=mask)
            out.append(F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE), ids[:, 1:].reshape(-1)).item())
        return out
    finally:
        hk.clear(); model.drop.p = 0.1; model.train()


def dstat(losses, base):
    a, b = np.array(losses), np.array(base); rng = np.random.default_rng(0); n = len(a)
    boots = [a[i].mean() - b[i].mean() for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return dict(dL=float(a.mean() - b.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


@torch.no_grad()
def early_exit(model, data):
    model.eval(); model.drop.p = 0.0; L = len(model.blocks); mask = causal_mask(CFG["seq_len"], DEVICE)
    outs = [None] * L; hk = Hooks()
    for li in range(L):
        hk.add(model.blocks[li].register_forward_hook(lambda m, a, o, li=li: outs.__setitem__(li, o)))
    tot = np.zeros(L); n = 0
    try:
        for b in range(min(N_EVAL_BATCHES, len(data) // EVAL_BATCH)):
            ids = data[b * EVAL_BATCH:(b + 1) * EVAL_BATCH].to(DEVICE)
            model(ids, attn_mask=mask)
            for li in range(L):
                lg = head_logits(model, outs[li])
                tot[li] += F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB_SIZE), ids[:, 1:].reshape(-1)).item()
            n += 1
    finally:
        hk.clear(); model.drop.p = 0.1; model.train()
    return (tot / max(n, 1)).tolist()


def measure(model, seed, eval_data, probe_nat, probe_dist):
    """Everything measured on one trained model."""
    L = len(model.blocks); out = {}
    mask = causal_mask(CFG["seq_len"], DEVICE)
    prof_nat, samp_nat, it1 = survey_profile(model, mask, probe_nat)
    prof_dist, samp_dist, it2 = survey_profile(model, mask, probe_dist)
    out["profile_natural"], out["profile_indist"] = prof_nat.tolist(), prof_dist.tolist()
    out["samples_natural"], out["samples_indist"] = [[round(x, 4) for x in s] for s in samp_nat], [[round(x, 4) for x in s] for s in samp_dist]
    out["stats_natural"], out["stats_indist"] = two_arm(prof_nat), two_arm(prof_dist)
    out["ufit_indist"] = fit_ushape(prof_dist); out["ufit_natural"] = fit_ushape(prof_nat)
    out["power_iters"] = [it1, it2]
    base = eval_losses(model, eval_data); out["baseline_eval_loss"] = float(np.mean(base))
    Qa = {li: haar(CFG["d_model"], SEED_BASE + li * 100 + seed) for li in range(L)}
    Qf = {li: haar(CFG["d_model"], SEED_BASE + li * 100 + 10 + seed) for li in range(L)}
    regions = {}
    for rn, blk in (("early", EARLY), ("waist", WAIST), ("late", LATE)):
        for dose in DOSES:
            regions[f"{rn}@{dose}"] = dstat(eval_losses(model, eval_data, rotate_branches(model, blk, Qa, Qf, dose)), base)
        regions[rn] = regions[f"{rn}@1.0"]
    out["regions_branch"] = regions
    edge = max(regions["early"]["dL"], regions["late"]["dL"], 1e-9); out["waist_over_edge_branch"] = regions["waist"]["dL"] / edge
    fit = out["ufit_indist"]; fr = {}
    for rn in ("waist", "early", "late"):
        if fit[rn]:
            fr[rn] = dstat(eval_losses(model, eval_data, rotate_branches(model, fit[rn], Qa, Qf, 1.0)), base)
    out["regions_branch_fitted_waist"] = fr
    out["skip_regions"] = {rn: dstat(eval_losses(model, eval_data, skip_blocks(model, blk)), base) for rn, blk in (("early", EARLY), ("waist", WAIST), ("late", LATE))}
    sedge = max(out["skip_regions"]["early"]["dL"], out["skip_regions"]["late"]["dL"], 1e-9); out["skip_waist_over_edge"] = out["skip_regions"]["waist"]["dL"] / sedge
    out["per_block_branch_dL"] = [dstat(eval_losses(model, eval_data, rotate_branches(model, [li], Qa, Qf, 1.0)), base)["dL"] for li in range(L)]
    out["per_block_skip_dL"] = [dstat(eval_losses(model, eval_data, skip_blocks(model, [li])), base)["dL"] for li in range(L)]
    out["early_exit_loss"] = early_exit(model, eval_data)
    return out


# ================================================================ training
def get_lr(step, warm_from_zero):
    w, n, lr = CFG["warmup_steps"], CFG["n_steps"], CFG["lr"]
    if step < w: return lr * step / w
    return lr * 0.5 * (1 + math.cos(math.pi * (step - w) / (n - w)))


def train_model(model, train, seed, exp, probe_cb):
    """topology/alpha: sequential batches over the fixed corpus, AdamW wd 0.1 betas (0.9, 0.95);
    kgram: permuted epochs, AdamW wd 0.01. Cosine schedule with 500 warm-up, clip 1.0, as the originals."""
    if exp == "kgram":
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.1, betas=(0.9, 0.95))
    mask = causal_mask(CFG["seq_len"], DEVICE); bs = CFG["batch_size"]; model.train()
    step, t0, last = 0, time.time(), None
    if exp == "kgram":
        gperm = torch.Generator().manual_seed(SEED_BASE + seed)
        while step < CFG["n_steps"]:
            perm = torch.randperm(len(train), generator=gperm)
            for i in range(0, len(perm) - bs, bs):
                if step >= CFG["n_steps"]: break
                if step in PROFILE_AT: probe_cb(step)
                ids = train[perm[i:i + bs]].to(DEVICE)
                for pg in opt.param_groups: pg["lr"] = get_lr(step, True)
                loss = F.cross_entropy(model(ids, attn_mask=mask)[:, :-1].reshape(-1, VOCAB_SIZE), ids[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                step += 1; last = loss.item()
                if step % 2500 == 0: log(f"    step {step}/{CFG['n_steps']} loss {last:.3f} ({time.time() - t0:.0f} s)")
    else:
        data_idx = 0
        for step in range(1, CFG["n_steps"] + 1):
            if step - 1 in PROFILE_AT: probe_cb(step - 1)
            for pg in opt.param_groups: pg["lr"] = get_lr(step, True)
            ids = torch.stack([train[(data_idx + i) % len(train)] for i in range(bs)]).to(DEVICE); data_idx += bs
            loss = F.cross_entropy(model(ids, attn_mask=mask)[:, :-1].reshape(-1, VOCAB_SIZE), ids[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            last = loss.item()
            if step % 2500 == 0: log(f"    step {step}/{CFG['n_steps']} loss {last:.3f} ({time.time() - t0:.0f} s)")
    return last


def run_one(exp, cond, seed, train, eval_data, probe_nat, probe_dist, ckpt_dir):
    t0 = time.time()
    torch.manual_seed(SEED_BASE + seed * 1000 if exp != "kgram" else SEED_BASE + seed); np.random.seed(SEED_BASE + seed)
    if exp == "topology": model = B2Transformer(topology=cond)
    elif exp == "alpha": model = AlphaTransformer(alpha=float(cond))
    else: model = D3Transformer()
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  {exp} {cond} seed {seed}: {n_params / 1e6:.1f}M params")
    mask = causal_mask(CFG["seq_len"], DEVICE)
    traj = {}
    def probe(step):
        p_nat, _, _ = survey_profile(model, mask, probe_nat); p_dist, _, _ = survey_profile(model, mask, probe_dist)
        base = eval_losses(model, eval_data)
        traj[step] = dict(profile_natural=p_nat.tolist(), profile_indist=p_dist.tolist(), stats_natural=two_arm(p_nat), stats_indist=two_arm(p_dist), eval_loss=float(np.mean(base)))
        s = traj[step]["stats_indist"]
        log(f"    step {step}: R_ex0 in-dist {s['R_ex0']:.3f} (natural {traj[step]['stats_natural']['R_ex0']:.3f}) E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f} eval {np.mean(base):.3f}")
    probe(0)
    last = train_model(model, train, seed, exp, probe)
    m = measure(model, seed, eval_data, probe_nat, probe_dist)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ck = ckpt_dir / f"{exp}_{cond}_s{seed}.pt"
    torch.save({k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}, ck)
    s = m["stats_indist"]; r = m["regions_branch"]
    extra = {}
    if exp == "topology" and cond == "gated_middle": extra["final_gates"] = model.get_gate_values()
    if exp == "topology" and hasattr(model, "highway_proj"): extra["highway_weight_norm"] = float(model.highway_proj.weight.norm())
    if exp == "topology" and hasattr(model, "mh_projs"): extra["skip_weight_norms"] = {k: float(p.weight.norm()) for k, p in model.mh_projs.items()}
    if exp == "topology" and hasattr(model, "unet_projs"): extra["skip_weight_norms"] = {k: float(p.weight.norm()) for k, p in model.unet_projs.items()}
    log(f"    -> eval {m['baseline_eval_loss']:.3f}  R_ex0 in-dist {s['R_ex0']:.3f} (natural {m['stats_natural']['R_ex0']:.3f}) E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f} two-arm {s['two_arm_rule']} C {s['contrast_C']:.3f}  "
        f"branch waist {r['waist']['dL']:.3f} early {r['early']['dL']:.3f} late {r['late']['dL']:.3f} (w/e {m['waist_over_edge_branch']:.2f})  skip waist {m['skip_regions']['waist']['dL']:.3f}  ({time.time() - t0:.0f} s)")
    return dict(experiment=exp, condition=str(cond), seed=seed, params_M=round(n_params / 1e6, 1), final_train_loss=round(last, 4), trajectory=traj, **m, **extra,
                checkpoint=str(ck), estimator="survey_sigma1_v2 token-local J^T J, causal mask, 5 inputs x 8 positions", survey_version=sv.VERSION,
                intervention="branch rotation (attention and MLP outputs, linear dose interpolation as d3); identity skip; early exit", cfg=CFG,
                gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", torch=torch.__version__, time_s=round(time.time() - t0),
                date=datetime.datetime.now().isoformat(timespec="seconds"))


# ================================================================ driver
def runs_for(exp):
    if exp == "topology": return [(t, s) for s in range(3) for t in TOPOLOGIES]
    if exp == "alpha": return [(a, s) for s in range(3) for a in ALPHAS]
    return [(k, s) for s in range(5) for k in K_VALUES]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", default=os.environ.get("REPRO_EXP", "kgram"), choices=["topology", "alpha", "kgram"])
    args = ap.parse_args(argv); exp = args.exp
    OUT = ROOT / exp; OUT.mkdir(parents=True, exist_ok=True); CK = OUT / "ckpt"
    todo = runs_for(exp)
    log("=" * 70); log(f"reprofile v2: {exp}, {len(todo)} runs -> {OUT}; device {DEVICE}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    log("=" * 70)
    train, val = load_wikitext_corpus()
    from transformers import AutoTokenizer
    probe_nat = sv.natural_text_ids(AutoTokenizer.from_pretrained("gpt2"), 5)
    models = unigram_cum = None
    if exp == "kgram":
        all_ids = train.reshape(-1).tolist()
        counts = np.bincount(np.array(all_ids), minlength=VOCAB_SIZE).astype(np.float64); unigram_cum = np.cumsum(counts / counts.sum())
        models = build_ngram_models(all_ids)
    t0 = time.time()
    for i, (cond, seed) in enumerate(todo):
        path = OUT / f"{exp}_{cond}_s{seed}.json"
        if path.exists():
            try:
                d = json.load(open(path)); assert "profile_indist" in d
                log(f"[{i + 1}/{len(todo)}] {cond} s{seed}: done (R {d['stats_indist']['R_ex0']:.3f}), skipping"); continue
            except Exception:
                path.unlink()
        log(f"\n[{i + 1}/{len(todo)}] {exp} {cond} seed {seed}")
        if exp == "kgram":
            d = kgram_data(cond, seed, train, models, unigram_cum); tr, ev = d["train"], d["held"]; probe_dist = ev[-5:]
        else:
            tr, ev = train, val[:N_EVAL_BATCHES * EVAL_BATCH]; probe_dist = val[-5:]
        res = run_one(exp, cond, seed, tr, ev, probe_nat, probe_dist, CK)
        tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1); tmp.replace(path)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    log(f"\nall done in {(time.time() - t0) / 3600:.2f} h"); summary(exp); export(exp)


def summary(exp):
    OUT = ROOT / exp
    rows = [json.load(open(f)) for f in sorted(OUT.glob(f"{exp}_*_s*.json"))]
    if not rows: return
    conds = TOPOLOGIES if exp == "topology" else ([str(a) for a in ALPHAS] if exp == "alpha" else [str(k) for k in K_VALUES])
    print("\n%-14s%3s %7s | %7s %7s %6s %6s %5s | %7s %7s %7s %5s | %7s %5s | %7s" % ("cond", "n", "eval", "R indist", "R nat", "E/M", "L/M", "two", "br w", "br e", "br l", "w/e", "skip w", "w/e", "C_fit"))
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        if not rs: continue
        g = lambda f: np.mean([f(r) for r in rs])
        print("%-14s%3d %7.3f | %7.3f %7.3f %6.2f %6.2f %2d/%d | %7.3f %7.3f %7.3f %5.2f | %7.3f %5.2f | %7.3f" % (
            c, len(rs), g(lambda r: r["baseline_eval_loss"]), g(lambda r: r["stats_indist"]["R_ex0"]), g(lambda r: r["stats_natural"]["R_ex0"]),
            g(lambda r: r["stats_indist"]["early_over_mid"]), g(lambda r: r["stats_indist"]["late_over_mid"]), sum(r["stats_indist"]["two_arm_rule"] for r in rs), len(rs),
            g(lambda r: r["regions_branch"]["waist"]["dL"]), g(lambda r: r["regions_branch"]["early"]["dL"]), g(lambda r: r["regions_branch"]["late"]["dL"]), g(lambda r: r["waist_over_edge_branch"]),
            g(lambda r: r["skip_regions"]["waist"]["dL"]), g(lambda r: r["skip_waist_over_edge"]), g(lambda r: r["ufit_indist"]["C_fit"])))


def export(exp):
    OUT = ROOT / exp
    stage = ROOT / f"{exp}_json"; stage.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json"): shutil.copy(f, stage / f.name)
    z = shutil.make_archive(str(ROOT / f"reprofile_{exp}_results"), "zip", str(stage))
    here = Path.cwd() / f"reprofile_{exp}_results.zip"
    if Path(z).resolve() != here.resolve(): shutil.copy(z, here)
    log(f"results zip: {z} and {here} -- on Drive already if ROOT is on Drive")


if __name__ == "__main__":
    main()
