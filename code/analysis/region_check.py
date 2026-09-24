"""Block-index lists for every 'middle' used in the paper, and the census association recomputed with matched regions.

The survey ratio R_ex0 takes thirds of blocks 1..L-1 (remainder to the middle); the census sensitivity regions take thirds of
blocks 1..L-2 (block 0 and the last block reported apart); the controlled intervention rotates blocks 4-7 of 12. This script
prints the block lists for every depth in the paper and recomputes (i) the census two-arm count and the association between
R and relative interior sensitivity with geometry taken over 1..L-2, and (ii) the controlled-model ratio with the middle
matched to the intervention (4-7; edges 1-3 and 8-11), which matched_analysis.py also reports (R_m47).

Run: python region_check.py -> region_check.json
"""
import json, glob, os
import numpy as np
from scipy import stats
from make_figs import R_thirds

R = "/mnt/user-data/outputs/ncs"
HERE = os.path.dirname(os.path.abspath(__file__))


def thirds(lo, hi):
    n = hi - lo + 1; t = n // 3
    return list(range(lo, lo + t)), list(range(lo + t, hi + 1 - t)), list(range(hi + 1 - t, hi + 1))


def ratio(p, lo, hi):
    q = np.asarray(p, float)[lo:hi + 1]; n = len(q); t = n // 3
    e, m, l = q[:t].mean(), q[t:n - t].mean(), q[n - t:].mean()
    return m / ((e + l) / 2), e / m, l / m


out = {"block_lists": {}}
for L in sorted({6, 12, 24, 16, 28, 32, 36, 48}):
    out["block_lists"][L] = dict(gain_R_ex0=dict(zip(("early", "middle", "late"), thirds(1, L - 1))),
                                 census_sensitivity=dict(zip(("early", "middle", "late"), thirds(1, L - 2))))
out["block_lists"][12]["controlled_intervention"] = dict(early=[0, 1, 2, 3], middle=[4, 5, 6, 7], late=[8, 9, 10, 11])
out["block_lists"][12]["matched_region_ratio_R_m47"] = dict(middle=[4, 5, 6, 7], edges=[1, 2, 3, 8, 9, 10, 11])
out["block_lists"][24]["controlled_intervention"] = dict(early=list(range(0, 8)), middle=list(range(8, 16)), late=list(range(16, 24)))

# census with geometry over 1..L-2
cen = json.load(open(f"{R}/census_v4/census_v4.json"))
surv = {}
for f in glob.glob(f"{R}/survey_v2/incontext-natural-block-float32/sigma1_*.json"):
    d = json.load(open(f)); surv[d["hf_id"].lower()] = d
rows = []
for r in cen:
    p = np.array(surv[r["hf_id"].lower()]["sigma1_profile"]); L = len(p)
    Rg = R_thirds(p)[0]; Rm, EM, LM = ratio(p, 1, L - 2)
    rows.append(dict(label=r["label"], L=L, R_1_Lm1=float(Rg), R_1_Lm2=float(Rm), two_arm_1_Lm1=bool(r["two_arm"]), two_arm_1_Lm2=bool(Rm < 0.80 and EM > 1 and LM > 1),
                     valid=bool(r["valid_readout"]), rel_interior_sensitivity=r["waist_over_edge_int"]))
v = [x for x in rows if x["valid"]]
out["census"] = dict(n=len(rows), two_arm_1_Lm1=sum(x["two_arm_1_Lm1"] for x in rows), two_arm_1_Lm2=sum(x["two_arm_1_Lm2"] for x in rows),
                     changed=[(x["label"], round(x["R_1_Lm1"], 2), round(x["R_1_Lm2"], 2)) for x in rows if x["two_arm_1_Lm1"] != x["two_arm_1_Lm2"]],
                     max_abs_R_diff=max(abs(x["R_1_Lm1"] - x["R_1_Lm2"]) for x in rows),
                     spearman_relsens_vs_R_1_Lm1=float(stats.spearmanr([x["rel_interior_sensitivity"] for x in v], [x["R_1_Lm1"] for x in v])[0]),
                     spearman_relsens_vs_R_1_Lm2=float(stats.spearmanr([x["rel_interior_sensitivity"] for x in v], [x["R_1_Lm2"] for x in v])[0]), n_valid=len(v), rows=rows)
json.dump(out, open(os.path.join(HERE, "region_check.json"), "w"), indent=1)
for L, d in out["block_lists"].items(): print(L, d)
c = out["census"]; print({k: v for k, v in c.items() if k != "rows"})
