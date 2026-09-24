"""Extra analyses for the pretrained matched assay: input-resampled CIs for the within-model correlations, the
direction excess with counts, and the decomposition with the random-class alignment as a ratio of means (so that the
three factors multiply exactly to the linearised same-position ratio)."""
import json, glob, os
import numpy as np
from scipy import stats
IE = 2; E0 = 0
out = {}
for f in sorted(glob.glob("results/mp_*.json")):
    d = json.load(open(f)); L = d["n_layers"]; lab = d["label"]
    raw = [r for r in d["raw"] if 1 <= r["block"] <= L - 2]
    blocks = sorted({r["block"] for r in raw}); inputs = sorted({r["input"] for r in raw})
    arr = {}
    for r in raw:
        arr.setdefault((r["block"], r["input"]), []).append(r)
    def rhos(inp):
        s1, rel, S = [], [], []
        for b in blocks:
            rs = [q for i in inp for q in arr[(b, i)]]
            s1.append(np.mean([q["sigma1"] for q in rs])); rel.append(np.mean([q["kl_seq"]["v1"][IE] for q in rs]))
            S.append(np.mean([q["kl_seq"]["v1"][IE] / (0.1 * q["h_norm"]) ** 2 for q in rs]))
        return stats.spearmanr(s1, rel)[0], stats.spearmanr(s1, S)[0]
    r0 = rhos(inputs); rng = np.random.default_rng(0); bs = []
    for _ in range(500):
        bs.append(rhos(list(rng.choice(inputs, len(inputs), replace=True))))
    bs = np.array(bs)
    # decomposition, ratio-of-means convention
    lr = [x for x in d["lin_raw"] if 1 <= x["block"] <= L - 2]
    loc, aft, ali, meas, lin = [], [], [], [], []
    for x in lr:
        s1 = x["sigma1"]; g_r = np.mean(x["gain_rand_each"][E0]); amp_r = np.mean(x["amp_rand"]); quad_r = np.mean(x["quad_rand"])
        loc.append((s1 / g_r) ** 2); aft.append((x["amp_v1"] / s1 ** 2) / (amp_r / g_r ** 2))
        ali.append((x["quad_v1"] / x["amp_v1"]) / (quad_r / amp_r))
        meas.append(x["kl_t_v1"][E0] / np.mean(x["kl_t_rand_each"][E0])); lin.append(x["quad_v1"] / quad_r)
    gm = lambda v: float(np.exp(np.mean(np.log(np.clip(v, 1e-300, None)))))
    out[lab] = dict(rho_rel=float(r0[0]), rho_rel_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                    rho_abs=float(r0[1]), rho_abs_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                    gm_local_sq=gm(loc), gm_after=gm(aft), gm_align=gm(ali), gm_meas_ratio=gm(meas), gm_lin_ratio=gm(lin),
                    frac_after_above_1=float(np.mean(np.array(aft) > 1)), median_lin_over_meas=float(np.median(np.array(lin) / np.array(meas))))
    o = out[lab]
    print(f"{lab:14s} rho_rel {o['rho_rel']:+.2f} [{o['rho_rel_ci'][0]:+.2f},{o['rho_rel_ci'][1]:+.2f}]  rho_abs {o['rho_abs']:+.2f} [{o['rho_abs_ci'][0]:+.2f},{o['rho_abs_ci'][1]:+.2f}] | "
          f"local^2 {o['gm_local_sq']:.1f} x after {o['gm_after']:.2f} x align {o['gm_align']:.2f} = {o['gm_local_sq']*o['gm_after']*o['gm_align']:.1f}; measured {o['gm_meas_ratio']:.1f}; after>1 {100*o['frac_after_above_1']:.0f}%")
json.dump(out, open("e1_extra.json", "w"), indent=1)
