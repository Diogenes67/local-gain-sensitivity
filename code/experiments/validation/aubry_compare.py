#!/usr/bin/env python3
"""
Matched comparison with the Aubry et al. (ICLR 2025) block-Jacobian implementation
===================================================================================
Same model, same inputs, same token position, two implementations of the same object.

Aubry et al. (github.com/sugolov/coupling, coupling/jacobian.py + coupling/main.py):
  J = d hidden_states[j+1][0, -1, :] / d hidden_states[j][0, -1, :], materialised as a d x d matrix
  by vectorising reverse-mode VJPs over the rows of the identity (chunk_vmap), then J - I for the
  block's residual contribution f (their "block Jacobian"), then torch.linalg.svd. Token index -1.
  HF hidden_states are used as the block boundaries; in GPT-NeoX/GPT-2/Llama the LAST entry of
  hidden_states carries the final layer norm, so their last-block Jacobian includes it.

Ours (survey_sigma1_v2): the same token-local map through forward hooks on the block itself,
  sigma_1 by power iteration on J^T J (one JVP + one VJP per step), block (J) or branch (J - I).

Three measurements per (model, dtype, input, block):
  A  Aubry as released: hidden_states boundaries, full Jacobian, full SVD -> sigma_1(J), sigma_1(J-I)
  B  Aubry's Jacobian + SVD on the block's own input/output tensors (removes the final-norm
     discrepancy at the last block; otherwise identical to A)
  C  ours: power iteration at the same position; also our standard 8-position profile.
Reported: per-block relative differences, Spearman of the depth profiles, R_ex0 under each,
wall time and peak GPU memory per path, and the JVP method the power iteration used.

  python aubry_compare.py --models EleutherAI/pythia-410m --dtypes float32 bfloat16 --n-inputs 5

Writes one JSON per (model, dtype) plus a summary; results zipped beside the output directory.
Needs survey_sigma1_v2.py beside it.
"""

import os, sys, json, time, math, datetime, argparse, shutil
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv

DEVICE = sv.DEVICE
log = sv.log
OUT = Path(os.environ.get("AUBRY_OUT", "aubry_compare"))

try:
    from functorch.experimental import chunk_vmap as _chunk_vmap  # what the released code imports
    CHUNK_VMAP = "functorch.experimental.chunk_vmap"
except Exception:
    _chunk_vmap = None
    CHUNK_VMAP = "torch.autograd.grad(is_grads_batched=True) in chunks (functorch.experimental.chunk_vmap unavailable)"


# ---------------------------------------------------------------- Aubry et al. jacobian(), verbatim in substance
def aubry_jacobian(output, input, index=-1, chunks=17, index_in=None):
    """coupling/jacobian.py::jacobian. output/input: (1, T, d) tensors in one autograd graph.
    Returns d output[0, index, :] / d input[0, index_in, :] as a (d, d) matrix, row i = grad of output_i."""
    from torch.autograd import grad
    out = output[0, index, :]
    d = out.numel()
    I_N = torch.eye(d, device=out.device, dtype=out.dtype)
    index_in = index if index_in is None else index_in
    if _chunk_vmap is not None:
        def get_vjp(v):
            return grad(out, input, v, retain_graph=True)[0][0, index_in, :]
        return _chunk_vmap(get_vjp, chunks=chunks)(I_N)
    rows = []
    step = int(math.ceil(d / chunks))
    for s in range(0, d, step):
        g = grad(out, input, grad_outputs=I_N[s:s + step], retain_graph=True, is_grads_batched=True)[0]
        rows.append(g[:, 0, index_in, :])
    return torch.cat(rows, 0)


def svd_top(J):
    J32 = J.detach().float()
    s = torch.linalg.svdvals(J32)
    return s[0].item(), s[1].item()


# ---------------------------------------------------------------- one (model, dtype)
def run_one(cfg, dtype_name, args):
    hf_id = cfg["hf_id"]
    b = sv.load_bundle(cfg, dtype_name)
    model, layers = b["model"], b["layers"]
    L = len(layers)
    inputs = sv.prepare_inputs(b, "natural", args.n_inputs)
    T = int(inputs[0]["input_ids"].shape[1])
    chunks = 2 * (T // 20) + 5            # their run_coupling_hf formula (i = 0)
    resident = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else float("nan")
    log(f"  {hf_id} {dtype_name}: {L} blocks, T={T}, resident {resident:.2f} GB, chunks {chunks}, vmap via {CHUNK_VMAP}")

    per_input = []
    tA = tB = tC = tC8 = 0.0
    memA = memB = memC = 0.0
    n_iter_all = []
    jvp_state = {}
    for ii, inp in enumerate(inputs):
        ids = inp["input_ids"]
        rec = dict(input=ii, A_block=[], A_branch=[], B_block=[], B_branch=[], C_block=[], C_branch=[], C8_profile=[], C_iters=[])

        # ---- A + B: one forward with the graph kept; params need grad as in their demo (from_pretrained default)
        for p in model.parameters():
            p.requires_grad_(True)
        cap_in, cap_out, hooks = [None] * L, [None] * L, []
        for li, layer in enumerate(layers):
            def pre(mod, a, kw, li=li):
                cap_in[li] = a[0] if len(a) else kw["hidden_states"]
            def post(mod, a, o, li=li):
                cap_out[li] = o[0] if isinstance(o, (tuple, list)) else o
            hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            hooks.append(layer.register_forward_hook(post))
        if DEVICE == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        outputs = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states
        assert len(hs) == L + 1, (len(hs), L)
        for j in range(L):
            J = aubry_jacobian(hs[j + 1], hs[j], index=-1, chunks=chunks)
            d = J.shape[0]
            s1, _ = svd_top(J)
            s1b, _ = svd_top(J - torch.eye(d, device=J.device, dtype=J.dtype))
            rec["A_block"].append(s1); rec["A_branch"].append(s1b)
            del J
        if DEVICE == "cuda": torch.cuda.synchronize()
        tA += time.time() - t0
        memA = max(memA, torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0)

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for j in range(L):
            J = aubry_jacobian(cap_out[j], cap_in[j], index=-1, chunks=chunks)
            d = J.shape[0]
            s1, _ = svd_top(J)
            s1b, _ = svd_top(J - torch.eye(d, device=J.device, dtype=J.dtype))
            rec["B_block"].append(s1); rec["B_branch"].append(s1b)
            del J
        if DEVICE == "cuda": torch.cuda.synchronize()
        tB += time.time() - t0 + 0.0
        memB = max(memB, torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0)
        for h in hooks: h.remove()
        del outputs, hs, cap_in, cap_out
        for p in model.parameters():
            p.requires_grad_(False)
        if DEVICE == "cuda": torch.cuda.empty_cache()

        # ---- C: ours, same position (T-1), block and branch
        if DEVICE == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        captured = sv.capture_layer_inputs(b, inp)
        for li in range(L):
            hidden0, rest, kw = captured[li]
            layout = sv.Layout(hidden0)
            call = sv.make_layer_call(layers[li], rest, kw)
            x0 = layout.get(hidden0.float(), [T - 1])
            f = sv.make_f_incontext(call, hidden0, layout, [T - 1], False, b["dtype"])
            s, n = sv.sigma1_power_iteration(f, x0, sv.N_RESTARTS, sv.PI_ITERS, sv.PI_TOL, jvp_state)
            fb = sv.make_f_incontext(call, hidden0, layout, [T - 1], True, b["dtype"])
            sb, nb = sv.sigma1_power_iteration(fb, x0, sv.N_RESTARTS, sv.PI_ITERS, sv.PI_TOL, jvp_state)
            rec["C_block"].append(s.item()); rec["C_branch"].append(sb.item()); rec["C_iters"].append([n, nb])
        if DEVICE == "cuda": torch.cuda.synchronize()
        tC += time.time() - t0
        memC = max(memC, torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0)

        # ---- C8: our standard protocol, 8 positions, block
        t0 = time.time()
        for li in range(L):
            hidden0, rest, kw = captured[li]
            layout = sv.Layout(hidden0)
            call = sv.make_layer_call(layers[li], rest, kw)
            positions = sv.choose_positions(layout, sv.N_POSITIONS, b.get("cls_offset", 0), inp.get("valid_T"))
            x0 = layout.get(hidden0.float(), positions)
            f = sv.make_f_incontext(call, hidden0, layout, positions, False, b["dtype"])
            s, n = sv.sigma1_power_iteration(f, x0, sv.N_RESTARTS, sv.PI_ITERS, sv.PI_TOL, jvp_state)
            rec["C8_profile"].append([float(v) for v in s])
        tC8 += time.time() - t0
        del captured
        if DEVICE == "cuda": torch.cuda.empty_cache()

        A, B, C = np.array(rec["A_block"]), np.array(rec["B_block"]), np.array(rec["C_block"])
        Ab, Bb, Cb = np.array(rec["A_branch"]), np.array(rec["B_branch"]), np.array(rec["C_branch"])
        rec["rel_CB_block"] = (np.abs(C - B) / B).tolist(); rec["rel_CB_branch"] = (np.abs(Cb - Bb) / Bb).tolist()
        rec["rel_AB_block"] = (np.abs(A - B) / B).tolist()
        per_input.append(rec)
        log(f"    input {ii + 1}/{len(inputs)}: |C-B|/B block median {np.median(rec['rel_CB_block']):.1e} max {np.max(rec['rel_CB_block']):.1e}; "
            f"branch median {np.median(rec['rel_CB_branch']):.1e} max {np.max(rec['rel_CB_branch']):.1e}; "
            f"A vs B last block {rec['rel_AB_block'][-1]:.2e} (others max {np.max(rec['rel_AB_block'][:-1]):.1e}); "
            f"iters median {int(np.median(np.array(rec['C_iters'])))}")

    # ---- aggregate
    from scipy import stats
    def prof(key): return np.mean([r[key] for r in per_input], axis=0)
    A, B, C = prof("A_block"), prof("B_block"), prof("C_block")
    Ab, Bb, Cb = prof("A_branch"), prof("B_branch"), prof("C_branch")
    C8 = np.mean([[np.mean(v) for v in r["C8_profile"]] for r in per_input], axis=0)
    relCB = np.concatenate([r["rel_CB_block"] for r in per_input]); relCBb = np.concatenate([r["rel_CB_branch"] for r in per_input])
    relAB = np.concatenate([r["rel_AB_block"][:-1] for r in per_input]); relAB_last = [r["rel_AB_block"][-1] for r in per_input]
    R = lambda p: sv.R_thirds(np.asarray(p), exclude0=True)[0]
    summary = dict(
        hf_id=hf_id, label=cfg["label"], dtype=dtype_name, n_blocks=L, d=int(d), T=T, n_inputs=len(inputs), position=T - 1,
        chunk_vmap=CHUNK_VMAP, jvp_method=jvp_state.get("name"),
        agreement=dict(
            block_rel_median=float(np.median(relCB)), block_rel_max=float(np.max(relCB)),
            branch_rel_median=float(np.median(relCBb)), branch_rel_max=float(np.max(relCBb)),
            pearson_block=float(np.corrcoef(C, B)[0, 1]), spearman_block=float(stats.spearmanr(C, B)[0]),
            spearman_block_vs_released=float(stats.spearmanr(C, A)[0]),
            released_vs_blockboundary_rel_max_except_last=float(np.max(relAB)),
            released_vs_blockboundary_rel_last_block=float(np.mean(relAB_last)),
            spearman_lastpos_vs_8pos=float(stats.spearmanr(C, C8)[0])),
        R_ex0=dict(aubry_released_block=R(A), aubry_released_branch=R(Ab), aubry_blockboundary_block=R(B), ours_lastpos_block=R(C),
                   ours_lastpos_branch=R(Cb), ours_8pos_block=R(C8)),
        time_s=dict(aubry_released=tA, aubry_blockboundary=tB, ours_lastpos_block_and_branch=tC, ours_8pos_block=tC8,
                    per_block_aubry=tB / (L * len(inputs)), per_block_ours_lastpos=tC / (2 * L * len(inputs))),
        peak_gb=dict(resident=resident, aubry_released=memA, aubry_blockboundary=memB, ours=memC),
        profiles=dict(aubry_released_block=A.tolist(), aubry_released_branch=Ab.tolist(), aubry_blockboundary_block=B.tolist(),
                      aubry_blockboundary_branch=Bb.tolist(), ours_lastpos_block=C.tolist(), ours_lastpos_branch=Cb.tolist(), ours_8pos_block=C8.tolist()),
        per_input=per_input, survey_version=sv.VERSION, torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", date=datetime.datetime.now().isoformat(timespec="seconds"))
    log(f"  => {cfg['label']} {dtype_name}: block rel diff median {summary['agreement']['block_rel_median']:.1e} max {summary['agreement']['block_rel_max']:.1e}, "
        f"branch median {summary['agreement']['branch_rel_median']:.1e} max {summary['agreement']['branch_rel_max']:.1e}, Spearman {summary['agreement']['spearman_block']:.4f}; "
        f"R_ex0 theirs {R(B):.3f} ours {R(C):.3f} (8-pos {R(C8):.3f}); time theirs {tB:.0f} s ours {tC:.0f} s; peak GB theirs {memB:.2f} ours {memC:.2f}")
    del b, model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["EleutherAI/pythia-410m"])
    ap.add_argument("--dtypes", nargs="*", default=["float32", "bfloat16"])
    ap.add_argument("--n-inputs", type=int, default=5)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    log("=" * 70); log(f"Aubry et al. matched comparison -> {OUT}; models {args.models}; dtypes {args.dtypes}; {args.n_inputs} inputs; device {DEVICE}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    log("=" * 70)
    rows = []
    for hf_id in args.models:
        cfg = sv.CFG[hf_id]
        for dt in args.dtypes:
            path = OUT / f"aubry_{sv.safe_name(hf_id)}_{dt}.json"
            if path.exists():
                rows.append(json.load(open(path))); log(f"{hf_id} {dt}: done, skipping"); continue
            res = run_one(cfg, dt, args)
            tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1); tmp.replace(path)
            rows.append(res)
    table(rows); export()


def table(rows):
    print("\n%-14s %-9s | %9s %9s | %9s %9s | %8s | %6s %6s %6s | %7s %7s | %6s %6s" % (
        "model", "dtype", "blk med", "blk max", "br med", "br max", "Spearman", "R thr", "R ours", "R 8pos", "t thr", "t ours", "GB thr", "GB our"))
    for r in rows:
        a, R, t, m = r["agreement"], r["R_ex0"], r["time_s"], r["peak_gb"]
        print("%-14s %-9s | %9.1e %9.1e | %9.1e %9.1e | %8.4f | %6.3f %6.3f %6.3f | %7.0f %7.0f | %6.2f %6.2f" % (
            r["label"], r["dtype"], a["block_rel_median"], a["block_rel_max"], a["branch_rel_median"], a["branch_rel_max"], a["spearman_block"],
            R["aubry_blockboundary_block"], R["ours_lastpos_block"], R["ours_8pos_block"], t["aubry_blockboundary"], t["ours_lastpos_block_and_branch"],
            m["aubry_blockboundary"], m["ours"]))
        print("%-14s %-9s   released hidden_states vs block boundary: last block rel %.2e, other blocks max %.1e; Spearman ours vs released %.4f; JVP %s" % (
            "", "", a["released_vs_blockboundary_rel_last_block"], a["released_vs_blockboundary_rel_max_except_last"], a["spearman_block_vs_released"], r["jvp_method"]))


def export():
    stage = OUT.parent / "aubry_compare_json"; stage.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json"): shutil.copy(f, stage / f.name)
    z = shutil.make_archive(str(OUT.parent / "aubry_compare_results"), "zip", str(stage))
    here = Path.cwd() / "aubry_compare_results.zip"
    if Path(z).resolve() != here.resolve(): shutil.copy(z, here)
    log(f"results zip: {z} and {here} -- download it before the instance goes")


if __name__ == "__main__":
    main()
