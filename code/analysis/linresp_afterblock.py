"""After-block amplification (22 Sept 2026; coauthor review 3.4).

The linearised response records, per probe and direction u, the total input-to-logit amplification ||A u||^2 (A from the
block-l input at position t to the position-t logits, block l included) and the output alignment g^T F g / ||g||^2. The
matched assay records, at the same probes (same inputs, positions, seed and random-direction generator), the local gain
along v1 (sigma1) and the mean local gain along the same four random directions (eps = 0.01, the linear regime). Dividing
the total amplification by the squared local gain gives the amplification by the blocks after block l:

    after(u) = ||A u||^2 / ||J u||^2,    with ||J v1|| = sigma1 and ||J u_rand|| = the matched assay's random gain

and the same-position consequence ratio factorises exactly (second order, causal mask) as

    KL(v1) / KL(rand) = (sigma1 / g_rand)^2 x after(v1) / after(rand) x align(v1) / align(rand).

For the random class the ratio of means is used (mean ||A u||^2 over the four directions divided by the squared mean
gain), which is the quantity the assay recorded. Reported per model over blocks 1-10: the three factors (median over
probes), the reconstruction of the measured ratio, the within-model rank correlation of sigma1 with after-block
amplification along v1, and the depth trend of after-block amplification.

Run: python linresp_afterblock.py -> linresp_afterblock.json
"""
import json, glob, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "linresp", "unz")
MATCHED = "/mnt/user-data/outputs/ncs/matched/results"
IE = 0  # eps = 0.01 in the matched assay's eps list [0.01, 0.03, 0.1, 0.3, 1.0]
BL = list(range(1, 11))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "linresp_*.json"))):
        d = json.load(open(f)); name = os.path.basename(f)[8:-5]
        m = json.load(open(os.path.join(MATCHED, f"matched_{name}.json")))
        mix = {(q["block"], q["input"], q["position"]): q for q in m["raw"]}
        probes = {}
        for r in d["raw"]:
            probes.setdefault((r["block"], r["input"], r["position"]), {}).setdefault("v1" if r["direction"] == "v1" else "rand", []).append(r)
        per = []
        for key, pr in probes.items():
            q = mix[key]; v = pr["v1"][0]; rs = pr["rand"]
            s1 = v["sigma1"]; g_r = q["gain"]["rand"][IE]
            amp_v, amp_r = v["amplification"], float(np.mean([x["amplification"] for x in rs]))
            al_v, al_r = v["alignment"], float(np.mean([x["alignment"] for x in rs]))
            quad_v, quad_r = v["quad"], float(np.mean([x["quad"] for x in rs]))
            after_v, after_r = amp_v / s1 ** 2, amp_r / g_r ** 2
            kl_v, kl_r = v["kl_meas"]["0.01"], float(np.mean([x["kl_meas"]["0.01"] for x in rs]))
            # consistency: linresp random KL vs the matched assay's own random KL at the same probe and eps (same directions?)
            per.append(dict(block=key[0], local_sq=(s1 / g_r) ** 2, after=after_v / after_r, align=al_v / al_r, after_v1=after_v, after_rand=after_r,
                            kl_ratio_meas=kl_v / max(kl_r, 1e-300), kl_ratio_lin=quad_v / max(quad_r, 1e-300),
                            rand_kl_match=q["kl_t"]["rand"][IE] / max(kl_r, 1e-300), sigma1=s1))
        sel = [p for p in per if p["block"] in BL]
        med = lambda k: float(np.median([p[k] for p in sel]))
        # geometric mean over probes then over blocks, the convention of linresp_analysis.py, so the three factors
        # multiply to the total amplification and alignment ratios already reported
        def gm(k):
            per_block = [np.mean([np.log(max(p[k], 1e-300)) for p in sel if p["block"] == b]) for b in BL]
            return float(np.exp(np.mean(per_block)))
        recon = float(np.median([p["local_sq"] * p["after"] * p["align"] / p["kl_ratio_meas"] for p in sel]))
        # per-block means for the within-model correlations
        blk = {b: [p for p in sel if p["block"] == b] for b in BL}
        s1_b = [np.mean([p["sigma1"] for p in blk[b]]) for b in BL]
        aft_b = [np.mean([p["after_v1"] for p in blk[b]]) for b in BL]
        aftr_b = [np.mean([p["after_rand"] for p in blk[b]]) for b in BL]
        ratio_b = [np.mean([p["after"] for p in blk[b]]) for b in BL]
        rows.append(dict(model=name, cond=d["condition"], n_probes=len(sel),
                         local_sq=med("local_sq"), after=med("after"), align=med("align"), product_over_measured=recon,
                         total_amp=float(np.median([p["local_sq"] * p["after"] for p in sel])),
                         gm_local_sq=gm("local_sq"), gm_after=gm("after"), gm_align=gm("align"), gm_total_amp=gm("local_sq") * gm("after"),
                         kl_ratio_meas=med("kl_ratio_meas"), kl_ratio_lin=med("kl_ratio_lin"),
                         n_after_above_1=int(sum(p["after"] > 1 for p in sel)), n_align_above_1=int(sum(p["align"] > 1 for p in sel)),
                         rand_kl_match_median=med("rand_kl_match"),
                         rho_sigma1_after_v1=float(stats.spearmanr(s1_b, aft_b)[0]), rho_sigma1_after_ratio=float(stats.spearmanr(s1_b, ratio_b)[0]),
                         depth_trend_after_v1=float(stats.spearmanr(BL, aft_b)[0]), depth_trend_after_rand=float(stats.spearmanr(BL, aftr_b)[0]),
                         after_v1_blocks=[float(x) for x in aft_b], after_rand_blocks=[float(x) for x in aftr_b], after_ratio_blocks=[float(x) for x in ratio_b]))
    json.dump(dict(generated="2026-09-22", rows=rows), open(os.path.join(HERE, "linresp_afterblock.json"), "w"), indent=1)
    print(f"{'model':16s} {'(s1/gr)^2':>9} {'after':>6} {'align':>6} {'prod/meas':>9} {'KLratio':>8} {'after>1':>8} {'randKL match':>12} {'rho(s1,after_v1)':>16} {'rho(s1,ratio)':>13} {'depth after_v1':>14}")
    for r in rows:
        print(f"{r['model']:16s} {r['local_sq']:9.2f} {r['after']:6.2f} {r['align']:6.2f} | gm {r['gm_local_sq']:5.2f} x {r['gm_after']:6.2f} x {r['gm_align']:5.2f} (total amp {r['gm_total_amp']:5.1f}) {r['product_over_measured']:9.3f} {r['kl_ratio_meas']:8.2f} {r['n_after_above_1']:4d}/{r['n_probes']:<3d} {r['rand_kl_match_median']:12.3f} {r['rho_sigma1_after_v1']:16.2f} {r['rho_sigma1_after_ratio']:13.2f} {r['depth_trend_after_v1']:14.2f}")


if __name__ == "__main__":
    main()
