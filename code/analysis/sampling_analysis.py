"""Sampling check of the matched assay (21 Sept 2026): 20 inputs x 6 positions x 8 random directions on ten existing checkpoints
(real s0-1, shuffled s0-1, k = 1 s0-1, k = 8 s0-1 at 12 layers; real s0 and shuffled s0 at 24 layers), against the main assay
(5 inputs x 4 positions x 4 random directions). Nothing retrained.

For each model: R_ex0 of sigma1, the within-model Spearman correlation between sigma1 and the downstream divergence (relative dose
eps = 0.1, and per unit squared absolute displacement), the middle-block divergence, the v1 / random ratio (all positions and same
position) and the excess over the isotropic prediction, from both runs; plus a bootstrap over inputs (inputs resampled with
replacement, 2,000 draws) in the sampling run, which is the uncertainty across independent inputs that the main assay could not give.

Run: python sampling_analysis.py -> sampling_check.json
"""
import json, glob, os
import numpy as np
from scipy import stats

R = "/mnt/user-data/outputs/ncs"
HERE = os.path.dirname(os.path.abspath(__file__))
SAMP = {12: os.path.join(HERE, "..", "sampling", "unz12"), 24: os.path.join(HERE, "..", "sampling", "unz")}
MAIN = {12: f"{R}/matched/results", 24: f"{R}/scale/results"}
E = 0.1; IE = 2


def per_block_from_raw(d, inputs=None):
    """Per-block means over probes (optionally a subset of inputs): sigma1, kl_seq v1/rand, kl_t v1/rand, gain v1/rand, h_norm, S."""
    L = len(d["per_block"]); acc = {k: np.zeros(L) for k in ("s1", "ksv", "ksr", "ktv", "ktr", "gv", "gr", "hn", "S")}; n = np.zeros(L)
    for q in d["raw"]:
        if inputs is not None and q["input"] not in inputs: continue
        b = q["block"]; n[b] += 1
        acc["s1"][b] += q["sigma1"]; acc["hn"][b] += q["h_norm"]
        acc["ksv"][b] += q["kl_seq"]["v1"][IE]; acc["ksr"][b] += q["kl_seq"]["rand"][IE]
        acc["ktv"][b] += q["kl_t"]["v1"][IE]; acc["ktr"][b] += q["kl_t"]["rand"][IE]
        acc["gv"][b] += q["gain"]["v1"][IE]; acc["gr"][b] += q["gain"]["rand"][IE]
        acc["S"][b] += q["kl_seq"]["v1"][IE] / (E * q["h_norm"]) ** 2
    return {k: v / np.maximum(n, 1) for k, v in acc.items()}, L


def stats_for(pb, L):
    lo, hi = 1, L - 2 if L == 24 else 10   # blocks 1-10 (12 layers) or 1-22 (24 layers), as in the paper
    sl = slice(lo, hi + 1)
    if L == 12: mid = slice(4, 8)
    else: mid = slice(8, 16)
    p = pb["s1"][1:]; n = len(p) // 3
    R_ex0 = p[n:len(p) - n].mean() / ((p[:n].mean() + p[len(p) - n:].mean()) / 2)
    gr2 = (pb["gv"][sl] / pb["gr"][sl]) ** 2
    return dict(R_ex0=float(R_ex0), rho_rel=float(stats.spearmanr(pb["s1"][sl], pb["ksv"][sl])[0]), rho_abs=float(stats.spearmanr(pb["s1"][sl], pb["S"][sl])[0]),
                mid_kl=float(pb["ksv"][mid].mean()), ratio_seq=float(np.median(pb["ksv"][sl] / pb["ksr"][sl])), ratio_t=float(np.median(pb["ktv"][sl] / pb["ktr"][sl])),
                excess_seq=float(np.median((pb["ksv"][sl] / pb["ksr"][sl]) / gr2)), excess_t=float(np.median((pb["ktv"][sl] / pb["ktr"][sl]) / gr2)))


def main():
    out = []
    rng = np.random.default_rng(0)
    for L, folder in SAMP.items():
        for f in sorted(glob.glob(os.path.join(folder, "matched_*.json"))):
            d = json.load(open(f)); name = os.path.basename(f)
            m = json.load(open(os.path.join(MAIN[L], name)))
            pbs, _ = per_block_from_raw(d); pbm, _ = per_block_from_raw(m)
            ss, sm = stats_for(pbs, L), stats_for(pbm, L)
            # bootstrap over inputs in the sampling run
            n_in = d["n_inputs"]; boots = {k: [] for k in ("rho_rel", "rho_abs", "mid_kl", "R_ex0")}
            for _ in range(2000):
                ids = set(rng.integers(0, n_in, n_in).tolist())
                # resampling with replacement means weights; approximate by weighted means over the multiset
                draw = rng.integers(0, n_in, n_in)
                acc = {k: np.zeros(L) for k in ("s1", "ksv", "S")}; n = np.zeros(L)
                by_in = {}
                for q in d["raw"]: by_in.setdefault(q["input"], []).append(q)
                for i in draw:
                    for q in by_in[i]:
                        b = q["block"]; n[b] += 1; acc["s1"][b] += q["sigma1"]; acc["ksv"][b] += q["kl_seq"]["v1"][IE]; acc["S"][b] += q["kl_seq"]["v1"][IE] / (E * q["h_norm"]) ** 2
                pb = {k: v / np.maximum(n, 1) for k, v in acc.items()}
                lo, hi = 1, (L - 2 if L == 24 else 10); sl = slice(lo, hi + 1); mid = slice(4, 8) if L == 12 else slice(8, 16)
                p = pb["s1"][1:]; t = len(p) // 3
                boots["R_ex0"].append(p[t:len(p) - t].mean() / ((p[:t].mean() + p[len(p) - t:].mean()) / 2))
                boots["rho_rel"].append(stats.spearmanr(pb["s1"][sl], pb["ksv"][sl])[0]); boots["rho_abs"].append(stats.spearmanr(pb["s1"][sl], pb["S"][sl])[0])
                boots["mid_kl"].append(pb["ksv"][mid].mean())
            ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
            out.append(dict(model=name.replace("matched_", "").replace(".json", ""), L=L, n_probes_main=len(m["raw"]), n_probes_sampling=len(d["raw"]),
                            n_random_main=m["n_random_directions"], n_random_sampling=d["n_random_directions"], main=sm, sampling=ss, sampling_ci_over_inputs=ci))
    json.dump(dict(generated="2026-09-21", rows=out), open(os.path.join(HERE, "sampling_check.json"), "w"), indent=1)
    print(f"{'model':16s} {'L':>2} | {'R main':>7} {'R samp':>7} {'CI':>14} | {'rho_rel m':>9} {'samp':>6} {'CI':>14} | {'rho_abs m':>9} {'samp':>6} | {'midKL m':>9} {'samp':>9} | {'v1/rand seq m':>13} {'samp':>6} | {'excess_t m':>10} {'samp':>6}")
    for r in out:
        a, b, c = r["main"], r["sampling"], r["sampling_ci_over_inputs"]
        print(f"{r['model']:16s} {r['L']:>2} | {a['R_ex0']:7.3f} {b['R_ex0']:7.3f} {c['R_ex0'][0]:6.3f}-{c['R_ex0'][1]:6.3f} | {a['rho_rel']:9.2f} {b['rho_rel']:6.2f} {c['rho_rel'][0]:6.2f}-{c['rho_rel'][1]:6.2f} | {a['rho_abs']:9.2f} {b['rho_abs']:6.2f} | {a['mid_kl']:9.2e} {b['mid_kl']:9.2e} | {a['ratio_seq']:13.1f} {b['ratio_seq']:6.1f} | {a['excess_t']:10.1f} {b['excess_t']:6.1f}")
    d_rel = [r["sampling"]["rho_rel"] - r["main"]["rho_rel"] for r in out]; d_R = [r["sampling"]["R_ex0"] - r["main"]["R_ex0"] for r in out]
    print("max |delta rho_rel| %.2f, max |delta R| %.3f, mid-KL ratio sampling/main %.2f-%.2f" % (max(map(abs, d_rel)), max(map(abs, d_R)), min(r["sampling"]["mid_kl"] / r["main"]["mid_kl"] for r in out), max(r["sampling"]["mid_kl"] / r["main"]["mid_kl"] for r in out)))


if __name__ == "__main__":
    main()
