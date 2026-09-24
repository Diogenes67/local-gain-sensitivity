"""
B1 Grid Follow-Up: Strengthen the width-gates-hourglass finding
================================================================
Three additions to the original 3×3×2 grid:

1. w=768 column at d=6,12,24 × 2 seeds = 6 runs
   → Pins down the HG threshold between w=512 (non-HG) and w=1024 (HG)

2. Extra seeds for borderline cells: 3 new seeds each for
   d6×w1024 (split 0.82 vs 1.00 on 2 seeds) and d24×w512 (transient HG)
   → 2 cells × 3 seeds = 6 runs

3. d24×w1024 with lower LR (1e-4 instead of 3e-4), 2 seeds
   → Fix the divergence so the strong HG result isn't dismissible
   → 2 runs

Total: 14 new runs, ~3-4 hours on T4/A100.
Keeps same architecture, data, profiling as original grid.
"""

import subprocess, sys
for pkg in ["datasets", "transformers"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import gc
import time
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path(os.environ.get("GRID_OUT", "grid_v2"))

VOCAB_SIZE = 50257
SEQ_LEN = 128
BATCH_SIZE = 16
N_HEADS = 8
TOTAL_STEPS = 20000
DEFAULT_LR = 3e-4
WEIGHT_DECAY = 0.01
HG_THRESHOLD = 0.80

PROFILE_STEPS = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 20000]

# ── Run definitions ──
# Each tuple: (depth, width, seed, lr, label)


# ── Model (identical to original grid) ──
class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class GridTransformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, vocab_size, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, input_ids, causal=True):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = None
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            mask = mask.float().masked_fill(mask, float('-inf'))
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm_f(x)
        return self.head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Data ──
def load_wikitext():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        text = " ".join([t for t in ds["text"] if len(t.strip()) > 50])
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_seqs = len(tokens) // SEQ_LEN
        tokens = tokens[:n_seqs * SEQ_LEN]
        return torch.tensor(tokens).reshape(-1, SEQ_LEN)
    except Exception as e:
        print(f"Warning: Could not load WikiText ({e}), using random data")
        return torch.randint(0, VOCAB_SIZE, (10000, SEQ_LEN))


class TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# ── Sigma1 profiling (same as original) ──
def profile_sigma1(model, data, n_inputs=5, n_iter=50, n_probes=16):
    model.eval()
    device = next(model.parameters()).device

    try:
        from torch.func import jvp as func_jvp, vjp as func_vjp
    except ImportError:
        from functorch import jvp as func_jvp, vjp as func_vjp

    all_profiles = []
    for inp_idx in range(min(n_inputs, len(data))):
        input_ids = data[inp_idx].unsqueeze(0).to(device)
        B, T = input_ids.shape
        t = T // 2

        with torch.no_grad():
            pos = torch.arange(T, device=device).unsqueeze(0)
            x = model.tok_emb(input_ids) + model.pos_emb(pos)
            mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            mask = mask.float().masked_fill(mask, float('-inf'))

        layer_sigma1s = []
        for layer_idx, layer in enumerate(model.layers):
            x_in = x.detach()

            def layer_fn(x_token):
                full_x = x_in.clone()
                full_x[0, t] = x_token
                out = layer(full_x, mask=mask)
                return out[0, t]

            d = x_in.shape[-1]
            best_s1 = 0.0

            for probe in range(n_probes):
                v = torch.randn(d, device=device)
                v = v / v.norm()

                for it in range(n_iter):
                    x_t = x_in[0, t].detach()
                    y_t, Jv = func_jvp(layer_fn, (x_t,), (v,))
                    _, vjp_fn = func_vjp(layer_fn, x_t)
                    JtJv = vjp_fn(Jv)[0]
                    v = JtJv / (JtJv.norm() + 1e-10)

                s1 = Jv.norm().item()
                best_s1 = max(best_s1, s1)

            layer_sigma1s.append(best_s1)

            with torch.no_grad():
                x = layer(x.detach(), mask=mask)

        all_profiles.append(layer_sigma1s)

    return np.mean(all_profiles, axis=0).tolist()


def compute_R_ex0(profile):
    p = profile[1:]
    n = len(p)
    third = n // 3
    if third == 0:
        return 1.0
    edge = p[:third] + p[-third:]
    mid = p[third:-third]
    if not edge or not mid:
        return 1.0
    return np.mean(mid) / np.mean(edge)


# ── Training ──
def train_one_run(n_layers, d_model, seed, lr, data_tensor):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GridTransformer(n_layers, d_model, N_HEADS, VOCAB_SIZE, SEQ_LEN).to(DEVICE)
    n_params = model.param_count()
    print(f"  Model: {n_layers}L × d={d_model}, {n_params/1e6:.1f}M params, lr={lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    dataset = TokenDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    trajectory = {}
    crossing_step = None
    next_profile_idx = 0

    model.train()
    step = 0
    last_loss = None

    while step < TOTAL_STEPS:
        for batch in loader:
            if step >= TOTAL_STEPS:
                break

            if next_profile_idx < len(PROFILE_STEPS) and step >= PROFILE_STEPS[next_profile_idx]:
                print(f"    Profiling at step {step}...", flush=True)
                profile = profile_sigma1(model, data_tensor, n_inputs=3, n_iter=30, n_probes=8)
                R_ex0 = compute_R_ex0(profile)
                is_hg = R_ex0 < HG_THRESHOLD
                trajectory[step] = round(R_ex0, 4)
                print(f"    step={step}: R_ex0={R_ex0:.4f}, HG={is_hg}, loss={last_loss:.4f}" if last_loss else
                      f"    step={step}: R_ex0={R_ex0:.4f}, HG={is_hg}", flush=True)

                if is_hg and crossing_step is None:
                    crossing_step = step

                next_profile_idx += 1
                model.train()

            batch = batch.to(DEVICE)
            logits = model(batch, causal=True)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                batch[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            last_loss = loss.item()

            if step % 5000 == 0:
                print(f"    step {step}/{TOTAL_STEPS}, loss={last_loss:.4f}", flush=True)

    # Final profile
    print(f"    Final profiling at step {TOTAL_STEPS}...", flush=True)
    profile = profile_sigma1(model, data_tensor, n_inputs=5, n_iter=50, n_probes=16)
    R_ex0_final = compute_R_ex0(profile)
    trajectory[TOTAL_STEPS] = round(R_ex0_final, 4)

    if R_ex0_final < HG_THRESHOLD and crossing_step is None:
        crossing_step = TOTAL_STEPS

    return {
        "depth": n_layers,
        "width": d_model,
        "seed": seed,
        "lr": lr,
        "params_M": round(n_params / 1e6, 1),
        "final_r_ex0": round(R_ex0_final, 4),
        "final_hg": R_ex0_final < HG_THRESHOLD,
        "crossing_step": crossing_step,
        "trajectory": trajectory,
        "final_profile": [round(x, 3) for x in profile],
        "final_loss": round(last_loss, 4) if last_loss else None,
    }


# ══════════════════════════════════════════════════════════════════════════
# Standalone runner for the last 4 of the 9 outstanding grid runs.
# Everything above this line is b1_grid_followup.py verbatim, except:
#   - load_dataset("wikitext") -> "Salesforce/wikitext"  (newer datasets needs the namespace)
#   - RESULTS_DIR from $GRID_OUT
#   - the RUNS list moved down here
# so the numerics match the 22 runs already done on Colab.
#
#   pip install torch transformers datasets numpy
#   GRID_OUT=/workspace/grid_v2 nohup python -u b1_grid_vast.py > grid.log 2>&1 &
#   tail -f grid.log
#
# One JSON per run in $GRID_OUT; finished runs are skipped, so just restart it
# if the box dies. ~4 h on an A100, expect 2-3 on an H100.
# ══════════════════════════════════════════════════════════════════════════

RUNS = [
    (24, 512, 137, DEFAULT_LR, "original-seed"),   # March gave 0.822; 3 new seeds gave 0.779-0.785
    (24, 256, 42,  DEFAULT_LR, "original-seed"),   # completes width 256
    (24, 256, 137, DEFAULT_LR, "original-seed"),
    (6,  768, 42,  DEFAULT_LR, "lost-to-crash"),   # completes the d6 w768 cell
]
# If you would rather put the heavy pair on the H100 and leave these to Colab,
# swap in:  (24, 1024, 42, DEFAULT_LR, "divergence-check"),
#           (24, 1024, 137, DEFAULT_LR, "divergence-check"),
#           (12, 1024, 42, DEFAULT_LR, "band-top"),
#           (12, 1024, 137, DEFAULT_LR, "band-top"),


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else "", flush=True)


def _ser(o):
    try:
        return o.item()
    except Exception:
        return str(o)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # pin fp32 so the H100 matches the A100 runs (these are the torch defaults;
    # setting them explicitly guards against a different default in the vast image)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    log("=" * 70)
    log(f"B1 grid, vast runner: {len(RUNS)} runs -> {RESULTS_DIR.resolve()}")
    log(f"device {DEVICE}; torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        vram = getattr(p, "total_memory", None)
        log(f"GPU {p.name}, {vram / 1e9:.0f} GB, sm_{p.major}{p.minor}")
    log(f"steps {TOTAL_STEPS}, batch {BATCH_SIZE}, seq {SEQ_LEN}, heads {N_HEADS}, "
        f"threshold {HG_THRESHOLD}")
    log(f"profile checkpoints: {PROFILE_STEPS}")
    log("=" * 70)

    cache = RESULTS_DIR / "_wikitext103_128.pt"
    if cache.exists():
        data = torch.load(cache)
        log(f"data from cache: {tuple(data.shape)}")
    else:
        log("tokenising wikitext-103 train (about 10 min the first time)...")
        data = load_wikitext()
        torch.save(data, cache)
        log(f"data: {tuple(data.shape)} (cached)")

    for i, (depth, width, seed, lr, why) in enumerate(RUNS, 1):
        key = f"d{depth}_w{width}_s{seed}" + (f"_lr{lr}" if lr != DEFAULT_LR else "")
        f = RESULTS_DIR / f"{key}.json"
        if f.exists():
            log(f"[{i}/{len(RUNS)}] skip {key} (already on disk)")
            continue
        log("")
        log("-" * 70)
        log(f"[{i}/{len(RUNS)}] {key}  ({why})")
        log("-" * 70)
        t0 = time.time()
        res = train_one_run(depth, width, seed, lr, data)
        res.update(label=why, run_key=key, time_s=round(time.time() - t0),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE,
                   torch=torch.__version__, script="b1_grid_vast.py (b1_grid_followup.py core)",
                   total_steps=TOTAL_STEPS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
                   n_heads=N_HEADS, hg_threshold=HG_THRESHOLD,
                   n_sequences=int(data.shape[0]), host="vast.ai")
        tmp = f.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=1, default=_ser)
        tmp.replace(f)                              # atomic, so a kill never leaves a half file
        log(f"  -> R_ex0 {res['final_r_ex0']}  HG {res['final_hg']}  "
            f"loss {res['final_loss']}  cross {res['crossing_step']}  {res['time_s']}s")
        del res
        gc.collect()
        torch.cuda.empty_cache()

    done = sorted(p.stem for p in RESULTS_DIR.glob("d*.json"))
    log("")
    log(f"done: {len(done)}/{len(RUNS)} requested; {len(done)} files in {RESULTS_DIR}")
    for k in done:
        d = json.load(open(RESULTS_DIR / f"{k}.json"))
        log(f"  {k:22s} R_ex0 {d['final_r_ex0']:.4f}  HG {str(d['final_hg']):5s}  "
            f"loss {d['final_loss']}")


if __name__ == "__main__":
    main()
