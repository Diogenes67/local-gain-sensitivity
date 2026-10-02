"""
B1 Width × Depth Onset-Time Grid
=================================
Tests whether hourglass crossing time increases with width and decreases
with depth, turning the width/duration narrative into a quantitative result.

Small grid: 3 widths × 3 depths × 2 seeds = 18 runs.
Each run trains AR for 20K steps, profiling at 10 log-spaced checkpoints
to find the crossing time (step at which R_ex0 first drops below 0.80).

Grid:
  Depth: 6, 12, 24 layers
  Width: 256, 512, 1024 (d_model)
  All with 8 heads, AR objective, WikiText-103

Expected: deeper → earlier crossing, wider → later crossing (or no crossing
within budget).

Run on GPU: ~4-6 hours total. VRAM: max ~8GB for the 24L×1024 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Grid ──
DEPTHS = [6, 12, 24]
WIDTHS = [256, 512, 1024]
N_HEADS = 8
VOCAB_SIZE = 50257
SEQ_LEN = 128
BATCH_SIZE = 16
TOTAL_STEPS = 20000
LR = 3e-4
WEIGHT_DECAY = 0.01
SEEDS = [42, 137]

# Checkpoint steps for profiling (log-spaced)
PROFILE_STEPS = [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 20000]
HG_THRESHOLD = 0.80


# ── Model (same architecture as factorial) ──
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
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
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


# ── Sigma1 profiling (torch.func) ──
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


# ── Training + profiling at checkpoints ──
def train_one_run(n_layers, d_model, seed, data_tensor):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GridTransformer(n_layers, d_model, N_HEADS, VOCAB_SIZE, SEQ_LEN).to(DEVICE)
    n_params = model.param_count()
    print(f"  Model: {n_layers}L × d={d_model}, {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TokenDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    checkpoint_results = []
    crossing_step = None  # first step where R_ex0 < threshold
    next_profile_idx = 0

    model.train()
    step = 0
    epoch = 0

    while step < TOTAL_STEPS:
        epoch += 1
        for batch in loader:
            if step >= TOTAL_STEPS:
                break

            # Check if we should profile
            if next_profile_idx < len(PROFILE_STEPS) and step >= PROFILE_STEPS[next_profile_idx]:
                print(f"    Profiling at step {step}...")
                profile = profile_sigma1(model, data_tensor, n_inputs=3, n_iter=30, n_probes=8)
                R_ex0 = compute_R_ex0(profile)
                is_hg = R_ex0 < HG_THRESHOLD
                checkpoint_results.append({
                    "step": step,
                    "R_ex0": round(R_ex0, 4),
                    "is_hourglass": is_hg,
                })
                print(f"    step={step}: R_ex0={R_ex0:.4f}, HG={is_hg}")

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

            if step % 2000 == 0:
                print(f"    step {step}/{TOTAL_STEPS}, loss={loss.item():.4f}")

    # Final profile
    print(f"    Final profiling at step {TOTAL_STEPS}...")
    profile = profile_sigma1(model, data_tensor, n_inputs=5, n_iter=50, n_probes=16)
    R_ex0_final = compute_R_ex0(profile)
    is_hg_final = R_ex0_final < HG_THRESHOLD

    if is_hg_final and crossing_step is None:
        crossing_step = TOTAL_STEPS

    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "seed": seed,
        "n_params_M": round(n_params / 1e6, 1),
        "depth_to_width": round(n_layers / d_model, 4),
        "R_ex0_final": round(R_ex0_final, 4),
        "is_hourglass_final": is_hg_final,
        "crossing_step": crossing_step,
        "trajectory": checkpoint_results,
        "final_profile": [round(x, 3) for x in profile],
    }


def main():
    print("=" * 60)
    print("B1 Width × Depth Onset-Time Grid")
    print(f"Device: {DEVICE}")
    print(f"Depths: {DEPTHS}, Widths: {WIDTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(DEPTHS) * len(WIDTHS) * len(SEEDS)}")
    print("=" * 60)

    print("\nLoading data...")
    data = load_wikitext()
    print(f"Data: {data.shape[0]} sequences of length {SEQ_LEN}")

    results = []
    for depth in DEPTHS:
        for width in WIDTHS:
            for seed in SEEDS:
                print(f"\n--- depth={depth}, width={width}, seed={seed} ---")
                try:
                    result = train_one_run(depth, width, seed, data)
                    results.append(result)
                    print(f"  RESULT: R_ex0={result['R_ex0_final']}, "
                          f"HG={result['is_hourglass_final']}, "
                          f"crossing_step={result['crossing_step']}")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  OOM: depth={depth}, width={width} — skipping")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    else:
                        raise

    # Summary table
    print("\n" + "=" * 70)
    print("WIDTH × DEPTH ONSET-TIME GRID — SUMMARY")
    print("=" * 70)
    print(f"{'Depth':>6} {'Width':>6} {'Params':>8} {'D/W':>6} {'R_ex0':>8} {'HG':>4} {'Cross':>8}")
    print("-" * 70)
    for depth in DEPTHS:
        for width in WIDTHS:
            runs = [r for r in results if r["n_layers"] == depth and r["d_model"] == width]
            if not runs:
                continue
            R_vals = [r["R_ex0_final"] for r in runs]
            hg_count = sum(1 for r in runs if r["is_hourglass_final"])
            cross_steps = [r["crossing_step"] for r in runs if r["crossing_step"] is not None]
            cross_str = f"{np.mean(cross_steps):.0f}" if cross_steps else "never"
            print(f"{depth:>6} {width:>6} {runs[0]['n_params_M']:>7.1f}M "
                  f"{runs[0]['depth_to_width']:>6.3f} "
                  f"{np.mean(R_vals):>7.3f} {hg_count:>3}/{len(runs)} {cross_str:>8}")
        print()

    # Save
    out = {"experiment": "B1_width_depth_grid", "results": results}
    out_path = RESULTS_DIR / "b1_width_depth_grid.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    txt_path = RESULTS_DIR / "b1_width_depth_grid.txt"
    with open(txt_path, "w") as f:
        f.write("B1 Width × Depth Onset-Time Grid\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Depth':>6} {'Width':>6} {'Params':>8} {'D/W':>6} {'R_ex0':>8} {'HG':>4} {'Cross':>8}\n")
        f.write("-" * 55 + "\n")
        for depth in DEPTHS:
            for width in WIDTHS:
                runs = [r for r in results if r["n_layers"] == depth and r["d_model"] == width]
                if not runs:
                    continue
                R_vals = [r["R_ex0_final"] for r in runs]
                hg_count = sum(1 for r in runs if r["is_hourglass_final"])
                cross_steps = [r["crossing_step"] for r in runs if r["crossing_step"] is not None]
                cross_str = f"{np.mean(cross_steps):.0f}" if cross_steps else "never"
                f.write(f"{depth:>6} {width:>6} {runs[0]['n_params_M']:>7.1f}M "
                        f"{runs[0]['depth_to_width']:>6.3f} "
                        f"{np.mean(R_vals):>7.3f} {hg_count:>3}/{len(runs)} {cross_str:>8}\n")
            f.write("\n")


if __name__ == "__main__":
    main()