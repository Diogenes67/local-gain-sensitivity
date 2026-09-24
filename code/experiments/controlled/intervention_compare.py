#!/usr/bin/env python3
"""
Intervention comparison on eight pretrained models (18 Sept 2026)
================================================================
The controlled experiments rotate a block's OUTPUT (the residual state); the pretrained census
rotates each sublayer's BRANCH output and leaves the residual stream unrotated. Those are
different interventions. This script applies both, plus identity skipping, block by block, on the
eight-model protocol panel, and records for each the loss change AND the actual perturbation
magnitude at the block output relative to the clean residual-stream norm, so equal rotation dose
is no longer assumed to be equal intervention strength.

Per block l, four passes over the evaluation set:
  block   Q [h_l + F_l(h_l)]          Haar rotation of the block output. Residual norm preserved.
  branch  h_l + Q F_l(h_l)            Haar rotation of each sublayer's output (the census rule).
                                      Branch norm preserved; residual norm not.
  skip    h_l                          block output replaced by its input (computation removed,
                                      alignment preserved).
  plumb   Q at block l output, Q^T at block l+1 input.  Must reproduce the clean loss; a check
                                      that the hooks act where they claim to (blocks 0..L-2).

Readout: next-token cross-entropy on WikiText-103 validation (30 x 16 x 128 tokens) for language
models; KL(clean || perturbed) over the ImageNet head on CIFAR-100 for ConvNeXt. Same data and
sizes as the census. Magnitude: ||h'_{l+1} - h_{l+1}|| / ||h_{l+1}|| on the first evaluation
batch, averaged over tokens, where h_{l+1} is the block output on the clean pass.

Reuses scramble_census_april_panel.py (must sit beside this file) for model loading, layer and
sublayer lookup, Haar rotations and the evaluation data. One JSON per model, resume-safe.

    HF_TOKEN=... python3 intervention_compare.py --out results/intervention_compare
"""

import os, sys, json, time, datetime, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scramble_census_april_panel as sc

PANEL = [
    ("gpt2",                        "gpt2"),
    ("EleutherAI/pythia-160m",      "pythia"),
    ("state-spaces/mamba-130m-hf",  "mamba"),
    ("facebook/convnext-tiny-224",  "convnext"),
    ("facebook/convnext-base-224",  "convnext"),
    ("meta-llama/Llama-3.2-1B",     "llama"),
    ("EleutherAI/pythia-6.9b",      "pythia"),
    ("Qwen/Qwen2.5-7B",             "qwen"),
]
DEVICE = sc.DEVICE


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "", flush=True)


# ---------------------------------------------------------------- hooks
def _hidden_of(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _with_hidden(output, h):
    return (h,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else h


def _input_hidden(args, kwargs):
    return args[0] if len(args) else kwargs["hidden_states"]


def rot(x, Q, axis):
    """Rotate x along `axis` by Q (d x d)."""
    Q = Q.to(x.device, dtype=x.dtype)
    x = x.movedim(axis, -1)
    y = torch.einsum("ij,...j->...i", Q, x)
    return y.movedim(-1, axis)


class Hooks:
    def __init__(self):
        self.h = []

    def add(self, handle):
        self.h.append(handle)

    def clear(self):
        for x in self.h:
            x.remove()
        self.h.clear()


def block_axis(arch):
    return 1 if arch == "convnext" else -1        # ConvNeXt block outputs are (N, C, H, W)


# ---------------------------------------------------------------- one model
def run_model(hf_id, arch, args):
    model, tok_or_proc, d_model = sc.load_model({"hf_id": hf_id, "arch": arch})
    layers = sc.get_layers(model, arch)
    L = len(layers)
    is_vision = arch == "convnext"
    n_eval = args.n_eval
    if is_vision:
        data = sc.prepare_image_eval(tok_or_proc, n_eval * sc.EVAL_BATCH_SIZE)
    else:
        data = sc.prepare_text_eval(tok_or_proc, n_eval * sc.EVAL_BATCH_SIZE)
    bs = sc.EVAL_BATCH_SIZE
    log(f"  {L} blocks; {n_eval} eval batches of {bs}")

    # block-output width per block (ConvNeXt changes width across stages)
    widths = []
    with torch.no_grad():
        caps = {}
        hk = Hooks()
        for li in range(L):
            hk.add(layers[li].register_forward_hook(lambda m, a, o, li=li: caps.__setitem__(li, _hidden_of(o).detach())))
        b0 = data[:bs].to(DEVICE)
        sc.forward_input(model, arch, b0)
        hk.clear()
    for li in range(L):
        widths.append(caps[li].shape[block_axis(arch)])
    clean_out0 = {li: caps[li].float() for li in range(L)}     # clean block outputs, batch 0
    del caps

    Qb = {li: sc.random_orthogonal(widths[li], sc.SEED + 1000 + li) for li in range(L)}
    Qs = {}
    for li in range(L):
        for si, sm in enumerate(sc.get_sublayer_modules(layers[li], arch)):
            Qs[(li, si)] = sc.random_orthogonal(sc.sublayer_width(sm, layers[li], arch, d_model), sc.SEED + li * 100 + si * 10)

    clean_logp = [None] * n_eval

    @torch.no_grad()
    def eval_pass(setup, capture_block=None):
        """setup(hk) registers the intervention hooks. Returns (per-batch losses, magnitude at capture_block on batch 0)."""
        hk = Hooks()
        mag = None
        try:
            setup(hk)
            store = {}
            if capture_block is not None:
                hk.add(layers[capture_block].register_forward_hook(lambda m, a, o: store.__setitem__("h", _hidden_of(o).detach().float())))
            losses = []
            for b in range(n_eval):
                batch = data[b * bs:(b + 1) * bs].to(DEVICE)
                if is_vision:
                    logp = F.log_softmax(model(pixel_values=batch).logits.float(), dim=-1)
                    if clean_logp[b] is None:
                        clean_logp[b] = logp.clone()
                    cl = clean_logp[b]
                    losses.append((cl.exp() * (cl - logp)).sum(-1).mean().item())
                else:
                    losses.append(model(input_ids=batch, labels=batch, use_cache=False).loss.item())
                if b == 0 and capture_block is not None:
                    h1, h0 = store["h"], clean_out0[capture_block]
                    ax = block_axis(arch)
                    diff = (h1 - h0).movedim(ax, -1).flatten(0, -2).norm(dim=-1)
                    base = h0.movedim(ax, -1).flatten(0, -2).norm(dim=-1)
                    mag = float((diff / base.clamp_min(1e-12)).mean())
            return losses, mag
        finally:
            hk.clear()

    base, _ = eval_pass(lambda hk: None)
    baseline = float(np.mean(base)); base_arr = np.array(base)
    log(f"  baseline {'KL' if is_vision else 'loss'} {baseline:.4f}")

    def mk_block(li):
        def setup(hk):
            hk.add(layers[li].register_forward_hook(lambda m, a, o: _with_hidden(o, rot(_hidden_of(o), Qb[li], block_axis(arch)))))
        return setup

    def mk_branch(li):
        def setup(hk):
            for si, sm in enumerate(sc.get_sublayer_modules(layers[li], arch)):
                hk.add(sm.register_forward_hook(lambda m, a, o, Q=Qs[(li, si)]: _with_hidden(o, rot(_hidden_of(o), Q, -1))))
        return setup

    def mk_skip(li):
        def setup(hk):
            def fh(m, a, kw, o):
                return _with_hidden(o, _input_hidden(a, kw))
            hk.add(layers[li].register_forward_hook(fh, with_kwargs=True))
        return setup

    def mk_plumb(li):
        def setup(hk):
            hk.add(layers[li].register_forward_hook(lambda m, a, o: _with_hidden(o, rot(_hidden_of(o), Qb[li], block_axis(arch)))))
            QT = Qb[li].T.contiguous()

            def pre(m, a, kw):
                if len(a):
                    return (rot(a[0], QT, block_axis(arch)),) + tuple(a[1:]), kw
                kw = dict(kw); kw["hidden_states"] = rot(kw["hidden_states"], QT, block_axis(arch)); return a, kw
            hk.add(layers[li + 1].register_forward_pre_hook(pre, with_kwargs=True))
        return setup

    results = {k: [] for k in ("block", "branch", "skip", "plumb")}
    mags = {k: [] for k in ("block", "branch", "skip")}
    t0 = time.time()
    for li in range(L):
        row = {}
        for name, mk in (("block", mk_block), ("branch", mk_branch), ("skip", mk_skip)):
            ls, mag = eval_pass(mk(li), capture_block=li)
            arr = np.array(ls); d = float(arr.mean() - baseline)
            rng = np.random.RandomState(sc.SEED + li)
            boots = [arr[i].mean() - base_arr[i].mean() for i in (rng.randint(0, n_eval, n_eval) for _ in range(sc.N_BOOTSTRAP))]
            results[name].append(dict(block=li, dL=d, ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], magnitude=mag))
            mags[name].append(mag); row[name] = (d, mag)
        if li < L - 1 and widths[li] == widths[li + 1]:
            # the control needs block l+1 to receive block l's output unchanged; at a ConvNeXt
            # stage boundary a downsampling layer sits between them, so the check is skipped there
            ls, _ = eval_pass(mk_plumb(li))
            dp = float(np.mean(ls) - baseline); results["plumb"].append(dict(block=li, dL=dp))
        else:
            dp = float("nan")
        log(f"    L{li:2d}  block dL {row['block'][0]:+.4f} (|dh|/|h| {row['block'][1]:.2f})   branch {row['branch'][0]:+.4f} ({row['branch'][1]:.2f})   "
            f"skip {row['skip'][0]:+.4f} ({row['skip'][1]:.2f})   plumb {dp:+.1e}   [{time.time() - t0:.0f} s]")

    def thirds(v):
        v = np.asarray(v, float)[1:L - 1]; n = len(v) // 3
        return dict(early=float(v[:n].mean()), waist=float(v[n:len(v) - n].mean()), late=float(v[len(v) - n:].mean()))
    summary = {}
    for name in ("block", "branch", "skip"):
        dl = [r["dL"] for r in results[name]]; t = thirds(dl)
        summary[name] = dict(**t, waist_over_edge=t["waist"] / max(t["early"], t["late"], 1e-12),
                             magnitude_median=float(np.nanmedian(mags[name])), magnitude_range=[float(np.nanmin(mags[name])), float(np.nanmax(mags[name]))])
    plumb_max = float(np.max(np.abs([r["dL"] for r in results["plumb"]]))) if results["plumb"] else None
    from scipy import stats
    rho_bb = stats.spearmanr([r["dL"] for r in results["block"]][1:L - 1], [r["dL"] for r in results["branch"]][1:L - 1])[0]
    rho_bs = stats.spearmanr([r["dL"] for r in results["block"]][1:L - 1], [r["dL"] for r in results["skip"]][1:L - 1])[0]
    log(f"  waist/edge  block {summary['block']['waist_over_edge']:.3f}  branch {summary['branch']['waist_over_edge']:.3f}  skip {summary['skip']['waist_over_edge']:.3f}"
        f"  | median |dh|/|h|  block {summary['block']['magnitude_median']:.2f} branch {summary['branch']['magnitude_median']:.2f} skip {summary['skip']['magnitude_median']:.2f}"
        f"  | plumbing max |dL| {plumb_max:.1e}  | Spearman block~branch {rho_bb:.2f}, block~skip {rho_bs:.2f}")
    out = dict(hf_id=hf_id, arch=arch, n_layers=L, baseline=baseline, n_eval_batches=n_eval, batch=bs, seq_len=sc.SEQ_LEN,
               readout="KL(clean||perturbed), ImageNet head, CIFAR-100" if is_vision else "next-token CE, WikiText-103 validation",
               widths=widths, per_block=results, summary=summary, plumbing_max_abs_dL=plumb_max,
               spearman_block_vs_branch=float(rho_bb), spearman_block_vs_skip=float(rho_bs),
               time_s=time.time() - t0, gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
               torch=torch.__version__, date=datetime.datetime.now().isoformat(timespec="seconds"))
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=os.environ.get("IC_OUT", "results/intervention_compare"))
    ap.add_argument("--n-eval", type=int, default=int(os.environ.get("IC_N_EVAL", sc.N_EVAL_BATCHES)))
    ap.add_argument("--no-skip", action="store_true")
    args, _ = ap.parse_known_args(argv)
    if os.environ.get("IC_MODELS"):
        args.models = os.environ["IC_MODELS"].split()
    sc._resolve_hf_token()
    os.makedirs(args.out, exist_ok=True)
    todo = [(h, a) for h, a in PANEL if not args.models or h in args.models]
    log(f"intervention comparison on {DEVICE}: {[h for h, _ in todo]}")
    log(f"HF token: {'found' if os.environ.get('HF_TOKEN') else 'none (Llama will fail)'}")
    for k, (hf_id, arch) in enumerate(todo):
        path = os.path.join(args.out, f"ic_{hf_id.replace('/', '_')}.json")
        if os.path.exists(path) and not args.no_skip:
            log(f"[{k + 1}/{len(todo)}] {hf_id}: done, skipping"); continue
        log(f"\n[{k + 1}/{len(todo)}] {hf_id} ({arch})")
        try:
            res = run_model(hf_id, arch, args)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"  FAILED {hf_id}: {e}"); continue
        tmp = path + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp, path)
    summary(args.out)


def summary(out):
    rows = [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f.startswith("ic_") and f.endswith(".json")]
    if not rows:
        return
    print("\n%-28s %8s | %-22s | %-22s | %-22s | %9s" % ("model", "base", "block  w/e  |dh|/|h|", "branch w/e  |dh|/|h|", "skip   w/e  |dh|/|h|", "plumb max"))
    for r in rows:
        s = r["summary"]
        print("%-28s %8.3f | %6.3f %6.2f      | %6.3f %6.2f      | %6.3f %6.2f      | %9.1e" % (
            r["hf_id"], r["baseline"], s["block"]["waist_over_edge"], s["block"]["magnitude_median"],
            s["branch"]["waist_over_edge"], s["branch"]["magnitude_median"], s["skip"]["waist_over_edge"], s["skip"]["magnitude_median"],
            r["plumbing_max_abs_dL"] or float("nan")))


if __name__ == "__main__":
    main()
