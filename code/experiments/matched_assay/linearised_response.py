"""Downstream linearised response: pilot (21 Sept 2026).

For a probe (block l, input i, position t) and a direction u at the block input, the small-dose divergence at the
same position is, to second order,

    D_KL(p_clean || p_perturbed) ~= 1/2 * eps^2 * ||h_{l,t}||^2 * g^T F g,    g = A u,

where A = d logits_t / d h_{l,t} is the Jacobian of the position-t logits with respect to the block input at t (the
whole downstream stack, block l included), and F = diag(p) - p p^T is the Fisher metric of the output distribution.
g = A u is one forward-mode product through the model, so no eigendecomposition is needed. The pilot asks

  1. does the linearised prediction reproduce the measured same-position divergence at eps = 0.01, 0.03, 0.1;
  2. how the consequence along v_1 splits into downstream amplification ||g||^2 / ||u||^2 and output alignment
     g^T F g / ||g||^2, and how each compares with a random direction;
  3. whether the split, not just the product, differs between training conditions.

Runs on the same subset as the sampling check (real s0-1, shuffled s0-1, k = 1 s0-1, k = 8 s0-1), five inputs x
four positions as in the main assay, v_1 and four random directions per probe. Reuses matched_perturbation.py
(adapters, block-input capture, top singular vector). Output: one JSON per model in LINRESP_OUT (resume-safe) and a zip.

Colab:
    import linearised_response as lr; lr.OUT = H / 'linresp'; lr.main()
"""
import os, sys, json, time, gc, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_perturbation as mp

OUT = Path(os.environ.get("LINRESP_OUT", "linresp"))
SUBSET = [("d1", "real", 0), ("d1", "real", 1), ("d1", "shuffled", 0), ("d1", "shuffled", 1), ("kgram", "1", 0), ("kgram", "1", 1), ("kgram", "8", 0), ("kgram", "8", 1)]
EPS = [0.01, 0.03, 0.1]
N_RAND = 4
DEVICE = mp.DEVICE
log = mp.log


def logit_f_with_ids(ad, li, hidden0, t, ids_1):
    base = hidden0.clone()

    def f(x):
        def pre(m, a):
            h = base.clone(); h[0, t] = x; return (h,) + tuple(a[1:])
        hk = ad.layers[li].register_forward_pre_hook(pre)
        try:
            logits = ad.forward(ids_1)
        finally:
            hk.remove()
        return logits[0, t]
    return f


def run_model(ad, ids, xs, sels):
    L = len(ad.layers); positions = mp.POSITIONS; n_in = ids.shape[0]
    g_rng = torch.Generator(device="cpu").manual_seed(mp.SEED)
    rows = []
    for li in range(L):
        for i in range(n_in):
            ids_1 = (xs[i:i + 1] if xs is not None else ids[i:i + 1]).to(DEVICE)
            x_in = mp.capture_block_input(ad, li, ids_1)
            f_blk = mp.make_token_local_f(ad, li, x_in, positions)
            x0 = x_in[0, positions].clone()
            sig, v1, _ = mp.top_singular(f_blk, x0)
            for p, t in enumerate(positions):
                fl = logit_f_with_ids(ad, li, x_in, t, ids_1)
                h0 = x_in[0, t].clone(); hn = float(h0.norm())
                with torch.no_grad():
                    logits0 = fl(h0); p0 = torch.softmax(logits0.float(), -1)
                dirs = [("v1", v1[p])]
                for r in range(N_RAND):
                    u = torch.randn(h0.shape[0], generator=g_rng).to(DEVICE); dirs.append((f"rand{r}", u / u.norm()))
                for name, u in dirs:
                    _, g = torch.func.jvp(fl, (h0,), (u,)); g = g.float()
                    Fg = p0 * g - p0 * float((p0 * g).sum())          # F g with F = diag(p) - p p^T
                    quad = float((g * Fg).sum()); amp = float((g * g).sum())
                    pred = {str(e): 0.5 * (e * hn) ** 2 * quad for e in EPS}
                    meas = {}
                    with torch.no_grad():
                        for e in EPS:
                            kls = []
                            for sgn in (1.0, -1.0):
                                lg = fl(h0 + sgn * e * hn * u).float(); q = torch.log_softmax(lg, -1)
                                kls.append(float((p0 * (torch.log(p0 + 1e-30) - q)).sum()))
                            meas[str(e)] = float(np.mean(kls))
                    rows.append(dict(block=li, input=i, position=t, direction=name, sigma1=float(sig[p]), h_norm=hn,
                                     amplification=amp, alignment=quad / max(amp, 1e-30), quad=quad, kl_pred=pred, kl_meas=meas))
        log(f"  block {li}: {len(rows)} rows so far")
    return rows


def summarise(rows):
    """Per block: v1 vs random amplification and alignment; prediction ratio at each eps."""
    L = max(r["block"] for r in rows) + 1; out = []
    for li in range(L):
        v = [r for r in rows if r["block"] == li and r["direction"] == "v1"]; rd = [r for r in rows if r["block"] == li and r["direction"] != "v1"]
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        out.append(dict(block=li, sigma1=m(v, "sigma1"), amp_v1=m(v, "amplification"), amp_rand=m(rd, "amplification"), align_v1=m(v, "alignment"), align_rand=m(rd, "alignment"),
                        quad_v1=m(v, "quad"), quad_rand=m(rd, "quad"),
                        pred_over_meas={e: float(np.median([r["kl_pred"][e] / max(r["kl_meas"][e], 1e-30) for r in v])) for e in map(str, EPS)}))
    return out


def run_job(exp, cond, seed, res_dir):
    fpath = res_dir / f"linresp_{exp}_{cond}_s{seed}.json"
    if fpath.exists():
        log(f"[SKIP] {exp} {cond} s{seed}"); return
    t0 = time.time(); log(f"\n{exp} {cond} seed {seed}")
    if exp == "d1":
        model = mp.load_d1(cond, seed); ad = mp.Adapter("d1", model, cond); ids, xs, sels = mp.d1_inputs(cond, seed)
    else:
        model = mp.load_kgram(int(cond), seed); ad = mp.Adapter("kgram", model); ids, xs, sels = mp.kgram_inputs(int(cond), seed)
    rows = run_model(ad, ids, xs, sels); per_block = summarise(rows)
    out = dict(experiment=exp, condition=cond, seed=seed, version="linresp-pilot-2026-09-21", eps=EPS, positions=mp.POSITIONS, n_inputs=int(ids.shape[0]), n_random=N_RAND,
               definition="KL_pred = 1/2 eps^2 ||h||^2 g^T F g, g = A u the forward-mode product of the position-t logits w.r.t. the block-l input at t; amplification = ||g||^2/||u||^2; alignment = g^T F g/||g||^2; kl_meas = same-position KL, signs averaged",
               per_block=per_block, raw=rows, elapsed_seconds=time.time() - t0)
    mp.write_json(fpath, out); log(f"  saved {fpath} ({time.time() - t0:.0f} s)")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    res = OUT / "results"; res.mkdir(parents=True, exist_ok=True)
    for exp, cond, seed in SUBSET:
        try:
            run_job(exp, cond, seed, res)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"  FAILED {exp} {cond} s{seed}: {e}")
    z = shutil.make_archive(str(OUT / "linresp_results"), "zip", res); log(f"zip {z}")
    return res


if __name__ == "__main__":
    main()
