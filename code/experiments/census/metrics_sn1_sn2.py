#!/usr/bin/env python3
"""
SN1 / SN2 metric comparison, recomputed under the v2 survey protocol
====================================================================

Replaces the April five-script numbers behind two claims in the Results:

  SN1  "On the same five Pythia models, centred kernel alignment gives an inverted
        profile, effective rank is flat and the Frobenius norm shows only weak depth
        dependence."
  SN2  "Normalising sigma_1 by the Frobenius norm flattens the pretrained profiles."

Four per-block quantities, all on the canonical protocol (natural WikiText-103 inputs,
token in context, float32, eager attention, TF32 off) so they are directly comparable
with sigma1_*.json from survey_sigma1_v2.py:

  cka[l]      linear CKA between the block's input representation h_l and its output
              h_{l+1}, both (T x d) over the whole sequence, features centred.
              Low CKA = the block changes the representation a lot.
  erank[l]    effective rank of h_l: exp(-sum p_i log p_i) with p_i = s_i / sum(s),
              s the singular values of the centred (T x d) representation.
  frob[l]     ||J_l||_F of the SAME token-local Jacobian sigma_1 is taken from, by
              Hutchinson: for v ~ N(0, I_d), E||J v||^2 = ||J||_F^2. Unbiased, needs
              only JVPs (no vjp), so it is far cheaper than the power iteration.
  sigma1[l]   read from the existing survey JSON, not recomputed.

and the derived ratio sigma1[l] / frob[l].

Every profile gets the same summary as the survey, R_ex0 = mean(middle third) /
mean(edge thirds) over blocks 1..L-1, via survey_sigma1_v2.profile_stats.

Usage
-----
    export SURVEY_DIR=results/survey_sigma1_v2/incontext-natural-block-float32
    python3 metrics_sn1_sn2.py --out results/metrics_sn1_sn2

    # defaults to the five Pythia models of the SN1 sentence; override with
    python3 metrics_sn1_sn2.py --models gpt2 EleutherAI/pythia-160m

One JSON per model, written atomically; finished models are skipped on a re-run.
Cost is dominated by n_probes JVPs per (block, input): about 6 min for the five
Pythia models on an H100 at the defaults, against ~2 h for a sigma_1 survey pass.
"""

import os, sys, json, time, argparse, datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv

PANEL = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
         "EleutherAI/pythia-1.4b", "EleutherAI/pythia-6.9b"]


def log(m=""):
    print(f"[{time.strftime('%H:%M:%S')}] {m}" if m else "", flush=True)


# ---------------------------------------------------------------- representation metrics
def _as_tokens(h):
    """(1,T,d) or (1,d,H,W) -> (N, d) float64 on cpu, features centred."""
    h = h.detach().float()
    if h.dim() == 4:
        x = h[0].flatten(1).T                      # (H*W, d)
    else:
        x = h[0]                                   # (T, d)
    x = x.double().cpu()
    return x - x.mean(0, keepdim=True)


def linear_cka(X, Y):
    """Linear CKA between two centred (N, d) matrices. Computed through the Gram
    form ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), which is O(N d^2), no N x N matrix."""
    xty = X.T @ Y
    num = float((xty ** 2).sum())
    dx = float(((X.T @ X) ** 2).sum()) ** 0.5
    dy = float(((Y.T @ Y) ** 2).sum()) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def effective_rank(X):
    """exp(Shannon entropy of the normalised singular-value spectrum) of a centred (N, d)."""
    s = torch.linalg.svdvals(X)
    s = s[s > 0]
    if s.numel() == 0:
        return float("nan")
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


# ---------------------------------------------------------------- Frobenius norm
def frobenius_hutchinson(f, x0, n_probes, seed, chunk=8):
    """||J||_F for the map f at base points x0 (B0, d). For v ~ N(0, I_d),
    E||J v||^2 = ||J||_F^2, so the mean over probes is unbiased. Returns (B0,).

    Probes run in chunks because f expands the whole sequence to batch B0 * rep, which
    at B0 = 8 positions and d = 4,096 is a gigabyte of activation per 64 probes."""
    B0 = x0.shape[0]
    g = torch.Generator(device=x0.device).manual_seed(seed)
    acc = torch.zeros(B0, device=x0.device)
    done = 0
    while done < n_probes:
        rep = min(chunk, n_probes - done)
        x = x0.repeat_interleave(rep, dim=0)
        v = torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32)
        Jv = None
        for name, fn in sv.JVP_METHODS:
            try:
                Jv = fn(f, x, v).float()
                if not torch.isfinite(Jv).all():
                    raise RuntimeError("non-finite JVP")
                break
            except Exception:
                Jv = None
        if Jv is None:
            return torch.full((B0,), float("nan"), device=x0.device)
        acc += (Jv.flatten(1) ** 2).sum(1).view(B0, rep).sum(1)
        done += rep
        del Jv, x, v
    return (acc / n_probes).sqrt()


# ---------------------------------------------------------------- one model
def run_model(cfg, args, survey_dir):
    b = sv.load_bundle(cfg, "float32")
    layers, L = b["layers"], len(b["layers"])
    inputs = sv.prepare_inputs(b, "natural", args.n_inputs)

    cka_s = [[] for _ in range(L)]
    er_s = [[] for _ in range(L)]
    fr_s = [[] for _ in range(L)]
    t0 = time.time()

    for ii, inp in enumerate(inputs):
        captured = sv.capture_layer_inputs(b, inp)
        for li in range(L):
            hidden0, rest, kw = captured[li]
            layout = sv.Layout(hidden0)
            layer_call = sv.make_layer_call(layers[li], rest, kw)

            # representation metrics: h_l against the block's own output
            X = _as_tokens(hidden0)
            if li + 1 < L:
                Y = _as_tokens(captured[li + 1][0])
            else:
                with torch.no_grad():
                    Y = _as_tokens(layer_call(hidden0.to(b["dtype"])))
            er_s[li].append(effective_rank(X))
            cka_s[li].append(linear_cka(X, Y))

            # ||J||_F on the token-local map, same positions the survey uses
            positions = sv.choose_positions(layout, args.n_positions, b.get("cls_offset", 0),
                                            inp.get("valid_T"))
            x0 = layout.get(hidden0.float(), positions)
            f = sv.make_f_incontext(layer_call, hidden0, layout, positions, False, b["dtype"])
            fr = frobenius_hutchinson(f, x0, args.n_probes, sv.SEED + ii)
            fr_s[li].extend([float(v) for v in fr])

        del captured
        if sv.DEVICE == "cuda":
            torch.cuda.empty_cache()
        log(f"    input {ii + 1}/{len(inputs)} ({time.time() - t0:.0f} s); "
            f"block0 |J|_F {np.mean(fr_s[0]):.3g}, mid {np.mean(fr_s[L // 2]):.3g}, "
            f"last {np.mean(fr_s[-1]):.3g}")

    cka = np.array([np.nanmean(s) for s in cka_s])
    erank = np.array([np.nanmean(s) for s in er_s])
    frob = np.array([np.nanmean(s) for s in fr_s])

    # sigma_1 from the survey run, not recomputed
    sig = None
    sp = os.path.join(survey_dir, f"sigma1_{sv.safe_name(cfg['hf_id'])}.json") if survey_dir else None
    if sp and os.path.exists(sp):
        sig = np.array(json.load(open(sp))["sigma1_profile"], dtype=float)
        if len(sig) != L:
            log(f"    survey profile has {len(sig)} blocks, model has {L}; ratio skipped")
            sig = None
    else:
        log(f"    no survey JSON at {sp}; sigma1/||J||_F ratio skipped")

    out = dict(
        hf_id=cfg["hf_id"], label=cfg["label"], arch=cfg["arch"], family=cfg["family"],
        params_M=cfg["params_M"], n_layers=L,
        version="metrics-sn1-sn2-v1", survey_version=sv.VERSION,
        protocol="incontext-natural-block-float32",
        n_inputs=args.n_inputs, n_positions=args.n_positions, n_probes=args.n_probes,
        definitions=dict(
            cka="linear CKA between h_l and the block output, whole sequence, features centred",
            erank="exp(entropy of normalised singular values) of the centred h_l",
            frob="||J||_F of d h_{l+1,t}/d h_{l,t} by Hutchinson, v ~ N(0, I_d)",
            sigma1="read from the survey JSON, not recomputed"),
        cka_profile=cka.tolist(), erank_profile=erank.tolist(), frob_profile=frob.tolist(),
        cka_samples=[list(map(float, s)) for s in cka_s],
        erank_samples=[list(map(float, s)) for s in er_s],
        frob_samples=[list(map(float, s)) for s in fr_s],
        cka_stats=sv.profile_stats(cka), erank_stats=sv.profile_stats(erank),
        frob_stats=sv.profile_stats(frob),
        time_s=time.time() - t0, torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if sv.DEVICE == "cuda" else "cpu",
        date=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if sig is not None:
        ratio = sig / frob
        out.update(sigma1_profile=sig.tolist(), sigma1_stats=sv.profile_stats(sig),
                   sigma1_over_frob=ratio.tolist(), sigma1_over_frob_stats=sv.profile_stats(ratio))
    del b
    if sv.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--survey-dir", default=os.environ.get("SURVEY_DIR"))
    ap.add_argument("--out", default=os.environ.get("METRICS_OUT", "results/metrics_sn1_sn2"))
    ap.add_argument("--n-inputs", type=int, default=5)
    ap.add_argument("--n-positions", type=int, default=8)
    ap.add_argument("--n-probes", type=int, default=64)
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sv._resolve_hf_token()

    models = args.models or PANEL
    unknown = [m for m in models if m not in sv.CFG]
    if unknown:
        raise SystemExit(f"unknown model ids (add to survey REGISTRY): {unknown}")
    os.makedirs(args.out, exist_ok=True)
    log(f"metrics-sn1-sn2-v1 on {sv.DEVICE}; torch {torch.__version__}")
    log(f"models ({len(models)}): {models}")
    log(f"survey dir: {args.survey_dir or '(none: sigma1 ratio will be skipped)'}")
    log(f"settings: {args.n_inputs} inputs x {args.n_positions} positions x {args.n_probes} probes")

    t0 = time.time()
    for i, m in enumerate(models):
        path = os.path.join(args.out, f"metrics_{sv.safe_name(m)}.json")
        if os.path.exists(path) and not args.no_skip:
            log(f"[{i + 1}/{len(models)}] {m}: already done, skipping")
            continue
        log(f"\n{'=' * 70}\n[{i + 1}/{len(models)}] {m}\n{'=' * 70}")
        try:
            res = run_model(sv.CFG[m], args, args.survey_dir)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"  FAILED: {type(e).__name__}: {e}")
            continue
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(res, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp, path)
        s = res["cka_stats"]["R_ex0_thirds"], res["erank_stats"]["R_ex0_thirds"], res["frob_stats"]["R_ex0_thirds"]
        r = res.get("sigma1_over_frob_stats", {}).get("R_ex0_thirds")
        log(f"  R_ex0  CKA {s[0]:.3f} | erank {s[1]:.3f} | ||J||_F {s[2]:.3f}"
            + (f" | sigma1/||J||_F {r:.3f}" if r is not None else "")
            + (f" | sigma1 {res['sigma1_stats']['R_ex0_thirds']:.3f}" if "sigma1_stats" in res else ""))

    # summary table
    rows = []
    for fn in sorted(os.listdir(args.out)):
        if fn.startswith("metrics_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(args.out, fn)))
            rows.append(r)
    if rows:
        log("\n" + "=" * 78)
        log(f"{'model':<16}{'sigma1':>9}{'CKA':>9}{'erank':>9}{'|J|_F':>9}{'s1/|J|_F':>11}")
        for r in sorted(rows, key=lambda x: x["params_M"]):
            g = lambda k: r.get(k, {}).get("R_ex0_thirds")
            fmt = lambda v: f"{v:>9.3f}" if isinstance(v, float) and np.isfinite(v) else f"{'-':>9}"
            log(f"{r['label']:<16}{fmt(g('sigma1_stats'))}{fmt(g('cka_stats'))}{fmt(g('erank_stats'))}"
                f"{fmt(g('frob_stats'))}{fmt(g('sigma1_over_frob_stats')):>11}")
        log("=" * 78)
        log("R_ex0 < 0.80 is a valley (hourglass); ~1.0 is flat; > 1.0 is inverted.")
    log(f"\nTotal {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
