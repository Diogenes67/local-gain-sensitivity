"""Downstream linearised response pilot (21 Sept 2026): analysis of linresp_*.json (eight checkpoints).

Per probe the pilot records g = A u (position-t logits differentiated along u through the downstream stack), the
amplification ||g||^2 (u is a unit vector), the alignment g^T F g / ||g||^2 with F the output Fisher metric, and the
predicted and measured same-position divergence at eps in {0.01, 0.03, 0.1}. This script reports, per model:
prediction accuracy; the within-model rank correlation of sigma1 with amplification, alignment and their product over
blocks 1-10; the share of the block-to-block variance of log consequence carried by each factor; and the v1-to-random
ratio of each factor (geometric mean over blocks 1-10).

Run: python linresp_analysis.py -> linresp_stats.json
"""
import json, glob, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "linresp", "unz")
L = 12; SL = slice(1, 11); BL = np.arange(1, 11)


def mean_log(raw, key, v1):
    out = np.zeros(L); n = np.zeros(L)
    for r in raw:
        if (r["direction"] == "v1") == v1:
            out[r["block"]] += np.log(max(r[key], 1e-300)); n[r["block"]] += 1
    return out / n


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "linresp_*.json"))):
        d = json.load(open(f)); raw = d["raw"]
        s1 = np.array([b["sigma1"] for b in d["per_block"]]); ls1 = np.log(s1[SL])
        la_v, lal_v, lq_v = mean_log(raw, "amplification", True), mean_log(raw, "alignment", True), mean_log(raw, "quad", True)
        la_r, lal_r, lq_r = mean_log(raw, "amplification", False), mean_log(raw, "alignment", False), mean_log(raw, "quad", False)
        pred = {e: float(np.median([r["kl_pred"][e] / r["kl_meas"][e] for r in raw if r["direction"] == "v1" and r["kl_meas"][e] > 0])) for e in ("0.01", "0.03", "0.1")}
        va, val_, cov = np.var(la_v[SL]), np.var(lal_v[SL]), np.cov(la_v[SL], lal_v[SL], bias=True)[0, 1]; tot = va + val_ + 2 * cov
        rows.append(dict(model=os.path.basename(f)[8:-5], cond=d["condition"], n_probes=len(raw), pred_over_meas=pred,
                         rho_sigma1_amp=float(stats.spearmanr(ls1, la_v[SL])[0]), rho_sigma1_align=float(stats.spearmanr(ls1, lal_v[SL])[0]), rho_sigma1_quad=float(stats.spearmanr(ls1, lq_v[SL])[0]),
                         var_share_amp=float(va / tot), var_share_align=float(val_ / tot), var_share_cov=float(2 * cov / tot),
                         v1_over_rand_amp=float(np.exp(np.mean(la_v[SL] - la_r[SL]))), v1_over_rand_align=float(np.exp(np.mean(lal_v[SL] - lal_r[SL]))), v1_over_rand_quad=float(np.exp(np.mean(lq_v[SL] - lq_r[SL]))),
                         depth_trend_amp=float(stats.spearmanr(BL, la_v[SL])[0]), depth_trend_align=float(stats.spearmanr(BL, lal_v[SL])[0]),
                         align_v1_blocks=np.exp(lal_v).tolist(), amp_v1_blocks=np.exp(la_v).tolist()))
    json.dump(dict(generated="2026-09-21", rows=rows), open(os.path.join(HERE, "linresp_stats.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['model']:16s} pred/meas {r['pred_over_meas']['0.01']:.3f} {r['pred_over_meas']['0.03']:.3f} {r['pred_over_meas']['0.1']:.3f} | rho(s1; amp {r['rho_sigma1_amp']:+.2f} align {r['rho_sigma1_align']:+.2f} quad {r['rho_sigma1_quad']:+.2f}) | var share amp {r['var_share_amp']:.2f} align {r['var_share_align']:.2f} | v1/rand amp {r['v1_over_rand_amp']:.1f} align {r['v1_over_rand_align']:.2f} quad {r['v1_over_rand_quad']:.1f} | align_v1 {min(r['align_v1_blocks'][1:11]):.2e}-{max(r['align_v1_blocks'][1:11]):.2e}")


if __name__ == "__main__":
    main()
