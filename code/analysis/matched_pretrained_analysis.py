"""Analysis of the pretrained matched assay (matched_pretrained.py outputs; 23 Sept 2026).

Mirrors the controlled-model analyses so the numbers are directly comparable:
  scale        within-model Spearman over blocks 1..L-2 between sigma1 and consequence, at the relative dose (KL_seq along
               v1 at eps = 0.1) and per unit squared absolute displacement (S = KL_seq / (eps ||h||)^2); activation-norm growth
               (matched_analysis.py, Perturbation-scaling check)
  pooled       log y = beta log sigma1 + gamma depth + alpha_model, depth = block / (L - 1) because L differs between models;
               cluster-robust SE by model; per-model depth trends as a sensitivity fit (regression_check.py M0 and M2)
  transfer     out-of-model prediction of the within-model-centred profile from beta and gamma fitted on the other models:
               leave one model out and leave one family out, against depth alone (regression_check.py L1O, loco_check.py)
  direction    same-position KL along v1 over a random direction against the squared gain ratio, per block at eps = 0.1
               (matched_analysis.py, direction specificity)
  decomposition second-order prediction against measured KL at eps = 0.01; local-gain, after-block and alignment factors
               (linresp_analysis.py, linresp_afterblock.py)

Run: python matched_pretrained_analysis.py <results dir> -> matched_pretrained_stats.json beside this script
"""
import json, glob, os, sys
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
E = 0.1


def load(res_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(res_dir, "mp_*.json"))):
        if f.endswith("_partial.json"):
            continue
        d = json.load(open(f))
        if d.get("quick"):
            continue
        out.append(d)
    return out


def per_model(d):
    ie = d["eps"].index(E); L = d["n_layers"]
    B = [x for x in d["per_block"] if 1 <= x["block"] <= L - 2]
    s1 = np.array([x["sigma1"] for x in B]); rel = np.array([x["kl_seq_v1"][ie] for x in B]); S = np.array([x["S_abs"][ie] for x in B])
    h = np.array([x["h_norm"] for x in B]); blk = np.array([x["block"] for x in B])
    klr = np.array([x["kl_t_v1"][ie] / x["kl_t_rand"][ie] for x in B]); gr2 = np.array([(x["gain_v1"][ie] / x["gain_rand"][ie]) ** 2 for x in B])
    r = dict(label=d["label"], family=d["family"], n_layers=L, n_blocks=len(B),
             gain_over_sigma1_eps001=float(np.mean([x["gain_v1"][0] / x["sigma1"] for x in d["per_block"]])),
             rho_rel=float(stats.spearmanr(s1, rel)[0]), rho_abs=float(stats.spearmanr(s1, S)[0]),
             rho_block_sigma1=float(stats.spearmanr(blk, s1)[0]), rho_block_hnorm=float(stats.spearmanr(blk, h)[0]),
             hnorm_ratio_last_over_first=float(h[-1] / h[0]),
             r2_log_rel_on_log_h=float(stats.linregress(np.log(h), np.log(rel)).rvalue ** 2),
             kl_t_ratio_median=float(np.median(klr)), gain_ratio_sq_median=float(np.median(gr2)), excess_median=float(np.median(klr / gr2)),
             n_excess_above_1=int((klr / gr2 > 1).sum()))
    frame = pd.DataFrame(dict(model=d["label"], family=d["family"], ls1=np.log(s1), depth=blk / (L - 1), y_rel=np.log(rel), y_abs=np.log(S)))
    return r, frame


def decomposition(d):
    lr = [x for x in d["lin_raw"] if 1 <= x["block"] <= d["n_layers"] - 2]
    if not lr:
        return {}
    ie0 = 0  # eps = 0.01
    rows = []
    for x in lr:
        s1 = x["sigma1"]; g_r = float(np.mean(x["gain_rand_each"][ie0]))   # [eps][direction]
        amp_r = np.array(x["amp_rand"]); quad_r = np.array(x["quad_rand"])
        al_v = x["quad_v1"] / x["amp_v1"]; al_r = float(np.mean(quad_r / amp_r))
        kl_v = x["kl_t_v1"][ie0]; kl_r = float(np.mean(x["kl_t_rand_each"][ie0]))
        rows.append(dict(block=x["block"], sigma1=s1, local_sq=(s1 / g_r) ** 2, after=(x["amp_v1"] / s1 ** 2) / (float(np.mean(amp_r)) / g_r ** 2),
                         after_v1=x["amp_v1"] / s1 ** 2, align_ratio=al_v / al_r, total_amp=x["amp_v1"] / float(np.mean(amp_r)),
                         kl_ratio=kl_v / max(kl_r, 1e-300), pred_over_meas=0.5 * (0.01 * x["h_norm"]) ** 2 * x["quad_v1"] / max(kl_v, 1e-300)))
    df = pd.DataFrame(rows)
    gm = lambda k: float(np.exp(df.groupby("block")[k].apply(lambda v: np.log(v.clip(lower=1e-300)).mean()).mean()))
    by_b = df.groupby("block").mean(numeric_only=True)
    recon = float(np.median(df["local_sq"] * df["after"] * df["align_ratio"] / df["kl_ratio"]))
    return dict(n_probes=len(df), pred_over_meas_median=float(df.pred_over_meas.median()), gm_local_sq=gm("local_sq"), gm_after=gm("after"),
                gm_align=gm("align_ratio"), gm_total_amp=gm("total_amp"), reconstruction_median=recon, frac_after_above_1=float((df.after > 1).mean()),
                rho_sigma1_after_v1=float(stats.spearmanr(by_b.sigma1, by_b.after_v1)[0]))


def centred_r2(df, y, test_col, with_gain=True):
    num = den = 0.0
    for key in df[test_col].unique():
        tr, te = df[df[test_col] != key], df[df[test_col] == key]
        if tr.model.nunique() < 2:
            continue
        f = f"{y} ~ ls1 + depth + C(model)" if with_gain else f"{y} ~ depth + C(model)"
        m = smf.ols(f, tr).fit()
        for _, t in te.groupby("model"):
            p = m.params["depth"] * t.depth + (m.params["ls1"] * t.ls1 if with_gain else 0)
            yc = t[y] - t[y].mean(); pc = p - p.mean()
            num += float(((yc - pc) ** 2).sum()); den += float((yc ** 2).sum())
    return 1 - num / den


def main(res_dir):
    ds = load(res_dir)
    rows, frames, dec = [], [], {}
    for d in ds:
        r, fr = per_model(d); rows.append(r); frames.append(fr); dec[d["label"]] = decomposition(d)
    df = pd.concat(frames, ignore_index=True)
    out = dict(n_models=len(ds), models=rows, decomposition=dec, pooled={}, transfer={})
    for y in ("y_abs", "y_rel"):
        m0 = smf.ols(f"{y} ~ ls1 + depth + C(model)", df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
        m2 = smf.ols(f"{y} ~ ls1 + C(model) + C(model):depth", df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
        out["pooled"][y] = dict(slope=float(m0.params["ls1"]), se_cluster=float(m0.bse["ls1"]), depth=float(m0.params["depth"]),
                                slope_model_trends=float(m2.params["ls1"]), se_model_trends=float(m2.bse["ls1"]))
        out["transfer"][y] = dict(l1o_gain=centred_r2(df, y, "model"), l1o_depth=centred_r2(df, y, "model", False),
                                  lofo_gain=centred_r2(df, y, "family"), lofo_depth=centred_r2(df, y, "family", False))
    json.dump(out, open(os.path.join(HERE, "matched_pretrained_stats.json"), "w"), indent=1)
    print(f"{'model':16s} {'L':>3s} {'g/s1':>6s} {'h ratio':>8s} {'rho rel':>8s} {'rho abs':>8s} {'KLt v1/rnd':>10s} {'(s1/gr)^2':>9s} {'excess':>7s} {'>1':>6s}")
    for r in rows:
        print(f"{r['label']:16s} {r['n_layers']:3d} {r['gain_over_sigma1_eps001']:6.3f} {r['hnorm_ratio_last_over_first']:8.1f} {r['rho_rel']:+8.2f} {r['rho_abs']:+8.2f} "
              f"{r['kl_t_ratio_median']:10.1f} {r['gain_ratio_sq_median']:9.2f} {r['excess_median']:7.2f} {r['n_excess_above_1']:3d}/{r['n_blocks']}")
    for y in ("y_abs", "y_rel"):
        p, t = out["pooled"][y], out["transfer"][y]
        print(f"{y}: pooled slope {p['slope']:+.2f} (cluster s.e. {p['se_cluster']:.2f}; with per-model depth trends {p['slope_model_trends']:+.2f} +/- {p['se_model_trends']:.2f}) | "
              f"L1O R2 {t['l1o_gain']:.2f} vs depth {t['l1o_depth']:.2f} | leave-one-family-out {t['lofo_gain']:.2f} vs {t['lofo_depth']:.2f}")
    for k, v in dec.items():
        if v:
            print(f"{k:16s} pred/meas {v['pred_over_meas_median']:.3f} | local^2 {v['gm_local_sq']:.2f} x after {v['gm_after']:.2f} x align {v['gm_align']:.2f} "
                  f"(total amp {v['gm_total_amp']:.1f}; recon {v['reconstruction_median']:.2f}) | after>1 {100 * v['frac_after_above_1']:.0f}% | rho(s1, after_v1) {v['rho_sigma1_after_v1']:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results"))
