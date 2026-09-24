"""Matched token-local perturbation assay: analysis of the 50 controlled checkpoints (d1 v2.1 x 20, k-gram x 30).

Reads matched/results/matched_*.json (per block: sigma1, gain and KL/dCE along v1 and along random unit
directions at eps in {0.01, 0.03, 0.1, 0.3, 1}), the d1 v2.1 JSONs and the k-gram JSONs (branch-rotation
waist dL, R_ex0), and writes matched/matched_stats.json plus fig/edfig7_new.{png,pdf}.

Run: python matched_analysis.py
"""
import json, glob, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, CAT, RAMP6, OUT

R = "/mnt/user-data/outputs/ncs"
EPS = [0.01, 0.03, 0.1, 0.3, 1.0]; E = 0.1; IE = EPS.index(E)
WAIST, EARLY, LATE = [4, 5, 6, 7], [0, 1, 2, 3], [8, 9, 10, 11]
D1_LAB = {"real": "real text", "shuffled": "shuffled", "randlab": "random labels", "bidir": "bidirectional"}
D1_ORDER = ["real", "bidir", "shuffled", "randlab"]
KS = [1, 2, 3, 4, 5, 8]


def thirds_ex0(v):
    """mid / mean(early, late) over blocks 1..L-2 (the R_ex0 convention: block 0 dropped, thirds of the rest)."""
    p = np.asarray(v, float)[1:]; L = len(p); n = L // 3
    e, m, l = p[:n].mean(), p[n:L - n].mean(), p[L - n:].mean()
    return m / ((e + l) / 2), e / m, l / m


def load():
    rows = []
    for f in sorted(glob.glob(f"{R}/matched/results/matched_*.json")):
        d = json.load(open(f)); exp, cond, seed = d["experiment"], d["condition"], d["seed"]
        pb = d["per_block"]
        r = dict(exp=exp, cond=cond, seed=seed, sigma1=np.array([b["sigma1"] for b in pb]),
                 gain_v1=np.array([b["gain_v1"] for b in pb]), gain_rand=np.array([b["gain_rand"] for b in pb]),
                 kl_seq_v1=np.array([b["kl_seq_v1"] for b in pb]), kl_seq_rand=np.array([b["kl_seq_rand"] for b in pb]),
                 kl_t_v1=np.array([b["kl_t_v1"] for b in pb]), kl_t_rand=np.array([b["kl_t_rand"] for b in pb]),
                 dce_v1=np.array([b["dce_v1"] for b in pb]), dce_rand=np.array([b["dce_rand"] for b in pb]),
                 raw=d["raw"], elapsed=d["elapsed_seconds"], gpu=d["prov_gpu"])
        if exp == "d1":
            o = json.load(open(f"{R}/d1/results_v21/{cond}_s{seed}.json"))
            r.update(rot_waist=o["regions"]["waist"]["dL"], rot_early=o["regions"]["early"]["dL"], rot_late=o["regions"]["late"]["dL"],
                     rot_per_block=np.array(o["per_block_rotation_dL"]), R_ex0_survey=o["R_ex0"], sigma1_survey=np.array(o["sigma1_profile"]), label=D1_LAB[cond], group=cond)
        else:
            o = json.load(open(f"{R}/reprofile/results_kgram/kgram_{cond}_s{seed}.json"))
            r.update(rot_waist=o["regions_branch"]["waist"]["dL"], rot_early=o["regions_branch"]["early"]["dL"], rot_late=o["regions_branch"]["late"]["dL"],
                     rot_per_block=np.array(o["per_block_branch_dL"]), R_ex0_survey=o["stats_indist"]["R_ex0"], sigma1_survey=np.array(o["profile_indist"]), label=f"k = {cond}", group=f"k{cond}")
        r["R_ex0_matched"] = thirds_ex0(r["sigma1"])[0]
        for key in ("kl_seq_v1", "kl_seq_rand", "dce_v1", "kl_t_v1"):
            v = r[key][:, IE]
            r[f"waist_{key}"] = float(v[WAIST].mean()); r[f"early_{key}"] = float(v[EARLY].mean()); r[f"late_{key}"] = float(v[LATE].mean())
            r[f"R_{key}"] = thirds_ex0(v)[0]
        # perturbation-scaling check: KL per unit squared absolute displacement, S = KL / ||delta||^2 with ||delta|| = eps ||h_{l,t}||,
        # averaged over the 20 probes of each block (raw entries carry h_norm per probe)
        L = len(pb); hn = np.zeros(L); S = np.zeros(L); n = np.zeros(L); grel = np.zeros(L); ng = np.zeros(L)
        rawix = {(q["block"], q["input"], q["position"]): q for q in d["raw"]}
        for (b, i_, p_), q in rawix.items():
            hn[b] += q["h_norm"]; S[b] += q["kl_seq"]["v1"][IE] / (E * q["h_norm"]) ** 2; n[b] += 1
            if (b + 1, i_, p_) in rawix:   # activation-normalised gain, sigma1 * ||h_l|| / ||h_{l+1}||, per probe
                grel[b] += q["sigma1"] * q["h_norm"] / rawix[(b + 1, i_, p_)]["h_norm"]; ng[b] += 1
        r["h_norm"] = hn / n; r["S_abs"] = S / n; r["g_rel"] = grel / np.maximum(ng, 1)
        r["spearman_blocks_grel_kl"] = float(stats.spearmanr(r["g_rel"][1:11], r["kl_seq_v1"][1:11, IE])[0])
        r["R_m47"] = float(r["sigma1"][4:8].mean() / ((r["sigma1"][1:4].mean() + r["sigma1"][8:12].mean()) / 2))   # middle matched to the intervention (4-7)
        r["spearman_blocks_sigma1_kl"] = float(stats.spearmanr(r["sigma1"][1:11], r["kl_seq_v1"][1:11, IE])[0])
        r["spearman_blocks_sigma1_S"] = float(stats.spearmanr(r["sigma1"][1:11], r["S_abs"][1:11])[0])
        r["spearman_block_sigma1"] = float(stats.spearmanr(np.arange(1, 11), r["sigma1"][1:11])[0])
        r["spearman_block_S"] = float(stats.spearmanr(np.arange(1, 11), r["S_abs"][1:11])[0])
        r["spearman_block_hnorm"] = float(stats.spearmanr(np.arange(1, 11), r["h_norm"][1:11])[0])
        r["hnorm_ratio_10_over_1"] = float(r["h_norm"][10] / r["h_norm"][1])
        r["spearman_blocks_gain_kl"] = float(stats.spearmanr(r["gain_v1"][1:11, IE], r["kl_seq_v1"][1:11, IE])[0])
        r["spearman_blocks_kl_rot"] = float(stats.spearmanr(r["kl_seq_v1"][1:11, IE], r["rot_per_block"][1:11])[0])
        rows.append(r)
    return rows


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with the (ranked) covariate z partialled out."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T; return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return float(stats.pearsonr(resid(rx, rz), resid(ry, rz))[0])


def depth_adjusted(rows):
    """What gain adds beyond block position: block-partialled rank correlations, first differences over adjacent blocks, and pooled
    log-log regressions with model fixed effects (blocks 1-10), for the relative-dose divergence and the per-unit-displacement S."""
    import pandas as pd, statsmodels.formula.api as smf
    bl = np.arange(1, 11); out = {"by_group": {}}; frames = []
    for i, r in enumerate(rows):
        s1, kl, S = r["sigma1"][1:11], r["kl_seq_v1"][1:11, IE], r["S_abs"][1:11]
        r["partial_rel"] = partial_spearman(s1, kl, bl); r["partial_abs"] = partial_spearman(s1, S, bl)
        r["diff_rel"] = float(stats.spearmanr(np.diff(np.log(s1)), np.diff(np.log(kl)))[0]); r["diff_abs"] = float(stats.spearmanr(np.diff(np.log(s1)), np.diff(np.log(S)))[0])
        frames.append(pd.DataFrame(dict(model=i, ls1=np.log(s1), lkl=np.log(kl), lS=np.log(S), lh=np.log(r["h_norm"][1:11]), block=bl)))
    for k in ("partial_rel", "partial_abs", "diff_rel", "diff_abs"):
        a = np.array([r[k] for r in rows]); out[k] = dict(median=float(np.median(a)), n_positive=int((a > 0).sum()), min=float(a.min()), max=float(a.max()))
    for g in D1_ORDER + [f"k{k}" for k in KS]:
        out["by_group"][g] = {k: [r[k] for r in rows if r["group"] == g] for k in ("partial_rel", "partial_abs", "diff_rel", "diff_abs")}
    df = pd.concat(frames)
    for y, name in (("lkl", "relative"), ("lS", "absolute")):
        m = smf.ols(f"{y} ~ ls1 + block + C(model)", df).fit(); m0 = smf.ols(f"{y} ~ ls1 + C(model)", df).fit()
        out[f"pooled_{name}"] = dict(slope_log_sigma1=float(m.params["ls1"]), se=float(m.bse["ls1"]), block=float(m.params["block"]), block_se=float(m.bse["block"]),
                                    slope_without_block=float(m0.params["ls1"]), se_without_block=float(m0.bse["ls1"]))
    m = smf.ols("lkl ~ lh + C(model)", df).fit(); out["logkl_on_lognorm"] = dict(slope=float(m.params["lh"]), se=float(m.bse["lh"]), r2=float(m.rsquared))
    return out


def ci_spearman(x, y, n_boot=5000, seed=0):
    x, y = np.asarray(x), np.asarray(y); rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x)); bs.append(stats.spearmanr(x[i], y[i])[0])
    rho, p = stats.spearmanr(x, y)
    return dict(rho=float(rho), p=float(p), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(len(x)))


def analyse(rows):
    st = {"eps_primary": E, "n_models": len(rows), "elapsed_total_min": round(sum(r["elapsed"] for r in rows) / 60, 1), "gpu": sorted(set(r["gpu"] for r in rows))}
    # A. the assay reproduces sigma_1 at the smallest dose
    ratio = np.concatenate([r["gain_v1"][:, 0] / r["sigma1"] for r in rows])
    st["gain_eps0.01_over_sigma1"] = dict(median=float(np.median(ratio)), p5=float(np.percentile(ratio, 5)), p95=float(np.percentile(ratio, 95)), min=float(ratio.min()), max=float(ratio.max()))
    # A2. matched sigma_1 (4 positions x 5 inputs) against the survey sigma_1 (8 positions x 5 inputs): R_ex0 agreement
    st["R_ex0_matched_vs_survey"] = dict(spearman=float(stats.spearmanr([r["R_ex0_matched"] for r in rows], [r["R_ex0_survey"] for r in rows])[0]),
                                         max_abs_diff=float(max(abs(r["R_ex0_matched"] - r["R_ex0_survey"]) for r in rows)))
    # A3. dose scaling of the KL readout: log-log slope 0.03 -> 0.3 over waist blocks (2 = quadratic)
    slopes = []
    for r in rows:
        for b in WAIST:
            k = r["kl_seq_v1"][b]; slopes.append((np.log(k[3]) - np.log(k[1])) / (np.log(0.3) - np.log(0.03)))
    st["kl_seq_loglog_slope_0.03_0.3"] = dict(median=float(np.median(slopes)), p5=float(np.percentile(slopes, 5)), p95=float(np.percentile(slopes, 95)))
    sat = [r["kl_seq_v1"][b, 4] / r["kl_seq_v1"][b, 3] for r in rows for b in WAIST]
    st["kl_seq_ratio_eps1_over_eps0.3"] = dict(median=float(np.median(sat)), quadratic_would_be=float((1.0 / 0.3) ** 2))
    # B. condition contrasts (d1) and the k ladder, matched waist effect vs branch-rotation waist dL
    d1 = [r for r in rows if r["exp"] == "d1"]; kg = [r for r in rows if r["exp"] == "kgram"]
    conds = {}
    for c in D1_ORDER:
        rs = [r for r in d1 if r["cond"] == c]
        conds[c] = dict(n=len(rs), waist_kl=[r["waist_kl_seq_v1"] for r in rs], waist_dce=[r["waist_dce_v1"] for r in rs], rot_waist=[r["rot_waist"] for r in rs],
                        R_ex0=[r["R_ex0_matched"] for r in rs], R_kl=[r["R_kl_seq_v1"] for r in rs], waist_ratio_v1_rand=[float(np.mean(r["kl_seq_v1"][WAIST, IE] / r["kl_seq_rand"][WAIST, IE])) for r in rs],
                        blocks_rho_sigma1_kl=[r["spearman_blocks_sigma1_kl"] for r in rs])
        for k2 in list(conds[c].keys()):
            if isinstance(conds[c][k2], list): conds[c][k2 + "_mean"] = float(np.mean(conds[c][k2]))
    st["d1"] = conds
    st["d1_ratio_real_over"] = {c: float(np.mean(conds["real"]["waist_kl"]) / np.mean(conds[c]["waist_kl"])) for c in ("shuffled", "randlab", "bidir")}
    st["d1_rot_ratio_real_over"] = {c: float(np.mean(conds["real"]["rot_waist"]) / max(np.mean(conds[c]["rot_waist"]), 1e-9)) for c in ("shuffled", "randlab", "bidir")}
    ks = {}
    for k in KS:
        rs = [r for r in kg if int(r["cond"]) == k]
        ks[k] = dict(n=len(rs), waist_kl=[r["waist_kl_seq_v1"] for r in rs], waist_dce=[r["waist_dce_v1"] for r in rs], rot_waist=[r["rot_waist"] for r in rs],
                     R_ex0=[r["R_ex0_matched"] for r in rs], R_kl=[r["R_kl_seq_v1"] for r in rs], waist_ratio_v1_rand=[float(np.mean(r["kl_seq_v1"][WAIST, IE] / r["kl_seq_rand"][WAIST, IE])) for r in rs],
                     blocks_rho_sigma1_kl=[r["spearman_blocks_sigma1_kl"] for r in rs])
        for k2 in list(ks[k].keys()):
            if isinstance(ks[k][k2], list): ks[k][k2 + "_mean"] = float(np.mean(ks[k][k2]))
    st["kgram"] = ks
    st["kgram_spearman_k_vs_waist_kl"] = ci_spearman([int(r["cond"]) for r in kg], [r["waist_kl_seq_v1"] for r in kg])
    st["kgram_spearman_k_vs_waist_dce"] = ci_spearman([int(r["cond"]) for r in kg], [r["waist_dce_v1"] for r in kg])
    st["kgram_spearman_k_vs_rot_waist"] = ci_spearman([int(r["cond"]) for r in kg], [r["rot_waist"] for r in kg])
    st["kgram_ladder_ratio_k8_over_k1"] = dict(kl=float(np.mean(ks[8]["waist_kl"]) / np.mean(ks[1]["waist_kl"])), rot=float(np.mean(ks[8]["rot_waist"]) / np.mean(ks[1]["rot_waist"])))
    # C. across the 50 checkpoints: matched waist effect vs rotation waist dL (assay agreement) and vs geometry
    for name, sub in (("all", rows), ("d1", d1), ("kgram", kg)):
        st[f"spearman_waistkl_vs_rotwaist_{name}"] = ci_spearman([r["waist_kl_seq_v1"] for r in sub], [r["rot_waist"] for r in sub])
        st[f"spearman_waistkl_vs_Rex0_{name}"] = ci_spearman([r["waist_kl_seq_v1"] for r in sub], [r["R_ex0_matched"] for r in sub])
        st[f"spearman_waistdce_vs_Rex0_{name}"] = ci_spearman([r["waist_dce_v1"] for r in sub], [r["R_ex0_matched"] for r in sub])
        st[f"spearman_Rkl_vs_Rex0_{name}"] = ci_spearman([r["R_kl_seq_v1"] for r in sub], [r["R_ex0_matched"] for r in sub])
    # D. within-model, across blocks: sigma_1 vs task effect
    b_rho = np.array([r["spearman_blocks_sigma1_kl"] for r in rows]); st["blocks_rho_sigma1_kl"] = dict(median=float(np.median(b_rho)), min=float(b_rho.min()), max=float(b_rho.max()),
                                                                                                        n_positive=int((b_rho > 0).sum()))
    st["blocks_rho_sigma1_kl"]["n_above_0.5"] = int((b_rho > 0.5).sum()); st["blocks_rho_sigma1_kl"]["n_below_-0.5"] = int((b_rho < -0.5).sum())
    b_S = np.array([r["spearman_blocks_sigma1_S"] for r in rows])
    st["blocks_rho_sigma1_S_abs"] = dict(median=float(np.median(b_S)), min=float(b_S.min()), max=float(b_S.max()), n_positive=int((b_S > 0).sum()),
                                         spearman_with_relative=float(stats.spearmanr(b_rho, b_S)[0]),
                                         hnorm_ratio_10_over_1_min=float(min(r["hnorm_ratio_10_over_1"] for r in rows)), hnorm_ratio_10_over_1_max=float(max(r["hnorm_ratio_10_over_1"] for r in rows)),
                                         n_hnorm_increasing=int(sum(r["spearman_block_hnorm"] > 0.9 for r in rows)),
                                         n_S_declining=int(sum(r["spearman_block_S"] < 0 for r in rows)),
                                         by_group={g: dict(rel=[r["spearman_blocks_sigma1_kl"] for r in rows if r["group"] == g], abs=[r["spearman_blocks_sigma1_S"] for r in rows if r["group"] == g],
                                                          rho_block_sigma1=[r["spearman_block_sigma1"] for r in rows if r["group"] == g], hnorm_ratio=[r["hnorm_ratio_10_over_1"] for r in rows if r["group"] == g])
                                                   for g in D1_ORDER + [f"k{k}" for k in KS]})
    b_g = np.array([r["spearman_blocks_grel_kl"] for r in rows])
    st["blocks_rho_grel_kl"] = dict(median=float(np.median(b_g)), min=float(b_g.min()), max=float(b_g.max()), n_positive=int((b_g > 0).sum()),
                                    spearman_with_sigma1_version=float(stats.spearmanr(b_rho, b_g)[0]),
                                    by_group={g: [r["spearman_blocks_grel_kl"] for r in rows if r["group"] == g] for g in D1_ORDER + [f"k{k}" for k in KS]})
    # matched-region ratio (middle 4-7, edges 1-3 and 8-11) against the survey convention (middle 4-8)
    Rm = np.array([r["R_m47"] for r in rows]); Rx = np.array([r["R_ex0_matched"] for r in rows])
    st["R_m47_vs_R_ex0"] = dict(spearman=float(stats.spearmanr(Rm, Rx)[0]), max_abs_diff=float(np.abs(Rm - Rx).max()), n_below_08_ex0=int((Rx < 0.8).sum()), n_below_08_m47=int((Rm < 0.8).sum()),
                                spearman_waistkl_vs_R_m47=ci_spearman([r["waist_kl_seq_v1"] for r in rows], Rm), spearman_rotwaist_vs_R_m47=ci_spearman([r["rot_waist"] for r in rows], Rm),
                                spearman_rotwaist_vs_R_ex0=ci_spearman([r["rot_waist"] for r in rows], Rx))
    st["depth_adjusted"] = depth_adjusted(rows)
    b_rho2 = np.array([r["spearman_blocks_kl_rot"] for r in rows]); st["blocks_rho_kl_rot"] = dict(median=float(np.median(b_rho2)), min=float(b_rho2.min()), max=float(b_rho2.max()), n_positive=int((b_rho2 > 0).sum()))
    # E. direction specificity: KL along v1 / KL along random, against the squared gain ratio (isotropic prediction)
    klr = np.concatenate([(r["kl_seq_v1"][1:11, IE] / r["kl_seq_rand"][1:11, IE]) for r in rows])
    gr2 = np.concatenate([(r["gain_v1"][1:11, IE] / r["gain_rand"][1:11, IE]) ** 2 for r in rows])
    st["direction_specificity"] = dict(kl_ratio_median=float(np.median(klr)), kl_ratio_p5=float(np.percentile(klr, 5)), kl_ratio_p95=float(np.percentile(klr, 95)),
                                       gain_ratio_sq_median=float(np.median(gr2)), excess_median=float(np.median(klr / gr2)), excess_p5=float(np.percentile(klr / gr2, 5)), excess_p95=float(np.percentile(klr / gr2, 95)),
                                       spearman_log=float(stats.spearmanr(np.log(klr), np.log(gr2))[0]), n=int(len(klr)))
    causal = [r for r in rows if r["cond"] != "bidir"]
    klt = np.concatenate([(r["kl_t_v1"][1:11, IE] / r["kl_t_rand"][1:11, IE]) for r in causal]); gr2c = np.concatenate([(r["gain_v1"][1:11, IE] / r["gain_rand"][1:11, IE]) ** 2 for r in causal])
    kls = np.concatenate([(r["kl_seq_v1"][1:11, IE] / r["kl_seq_rand"][1:11, IE]) for r in causal])
    st["direction_specificity_samepos"] = dict(n_models=len(causal), n=int(len(klt)), kl_t_ratio_median=float(np.median(klt)), kl_seq_ratio_median_causal=float(np.median(kls)),
                                               gain_ratio_sq_median=float(np.median(gr2c)), excess_t_median=float(np.median(klt / gr2c)), excess_t_p5=float(np.percentile(klt / gr2c, 5)), excess_t_p95=float(np.percentile(klt / gr2c, 95)),
                                               excess_seq_median_causal=float(np.median(kls / gr2c)), n_excess_t_above_1=int((klt / gr2c > 1).sum()),
                                               by_group={g: float(np.median(np.concatenate([(r["kl_t_v1"][1:11, IE] / r["kl_t_rand"][1:11, IE]) / (r["gain_v1"][1:11, IE] / r["gain_rand"][1:11, IE]) ** 2 for r in causal if r["group"] == g]))) for g in ["real", "shuffled", "randlab"] + [f"k{k}" for k in KS]})
    for name, sub in (("real", [r for r in d1 if r["cond"] == "real"]), ("shuffled", [r for r in d1 if r["cond"] == "shuffled"]), ("k8", [r for r in kg if r["cond"] == "8"]), ("k1", [r for r in kg if r["cond"] == "1"])):
        a = np.concatenate([(r["kl_seq_v1"][WAIST, IE] / r["kl_seq_rand"][WAIST, IE]) for r in sub]); g = np.concatenate([(r["gain_v1"][WAIST, IE] / r["gain_rand"][WAIST, IE]) ** 2 for r in sub])
        st["direction_specificity"][f"waist_{name}"] = dict(kl_ratio_median=float(np.median(a)), gain_ratio_sq_median=float(np.median(g)), excess_median=float(np.median(a / g)))
    return st


def figure(rows, st):
    d1 = [r for r in rows if r["exp"] == "d1"]; kg = [r for r in rows if r["exp"] == "kgram"]
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 3, left=0.07, right=0.99, top=0.97, bottom=0.075, wspace=0.45, hspace=0.5)
    blocks = np.arange(12)
    col_d1 = {"real": CAT[0], "bidir": CAT[3], "shuffled": CAT[1], "randlab": CAT[2]}
    # a: sigma_1 by block, d1 conditions; b: matched task effect by block; c: kgram task effect by block
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    for c in D1_ORDER:
        rs = [r for r in d1 if r["cond"] == c]; M = np.array([r["sigma1"] for r in rs])
        ax.plot(blocks, M.mean(0), color=col_d1[c], lw=0.9, marker="o", ms=1.8, mew=0, label=D1_LAB[c])
        ax.fill_between(blocks, M.min(0), M.max(0), color=col_d1[c], alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (matched probes)"); tidy(ax); ax.legend(loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8)
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    for c in D1_ORDER:
        rs = [r for r in d1 if r["cond"] == c]; M = np.array([r["kl_seq_v1"][:, IE] for r in rs])
        ax.plot(blocks, M.mean(0), color=col_d1[c], lw=0.9, marker="o", ms=1.8, mew=0, label=D1_LAB[c])
        ax.fill_between(blocks, M.min(0), M.max(0), color=col_d1[c], alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel(f"downstream KL along $v_1$, $\\epsilon$ = {E} (nats)"); tidy(ax)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    for i, k in enumerate(KS):
        rs = [r for r in kg if int(r["cond"]) == k]; M = np.array([r["kl_seq_v1"][:, IE] for r in rs])
        ax.plot(blocks, M.mean(0), color=RAMP6[i], lw=0.9, marker="o", ms=1.8, mew=0, label=f"k = {k}")
        ax.fill_between(blocks, M.min(0), M.max(0), color=RAMP6[i], alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel(f"downstream KL along $v_1$, $\\epsilon$ = {E} (nats)"); tidy(ax); ax.legend(loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.8)
    # d: matched waist effect vs branch-rotation waist dL (50 models); e: vs R_ex0; f: direction specificity
    def scatter(ax, xk, yk, xl, yl, letter, logx=False, logy=True):
        panel(ax, letter)
        for c in D1_ORDER:
            rs = [r for r in d1 if r["cond"] == c]
            ax.plot([r[xk] for r in rs], [r[yk] for r in rs], "o", color=col_d1[c], ms=3.6, mew=0.5, mec="white", ls="none", label=D1_LAB[c], zorder=4)
        for i, k in enumerate(KS):
            rs = [r for r in kg if int(r["cond"]) == k]
            ax.plot([r[xk] for r in rs], [r[yk] for r in rs], "s", color=RAMP6[i], ms=3.4, mew=0.5, mec="white", ls="none", label=f"k = {k}", zorder=4)
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        ax.set_xlabel(xl); ax.set_ylabel(yl); tidy(ax, grid="both")
    ax = fig.add_subplot(gs[1, 0]); scatter(ax, "rot_waist", "waist_kl_seq_v1", "branch-rotation waist $\\Delta L$, dose 1 (nats)", f"matched waist KL, $\\epsilon$ = {E} (nats)", "d", logx=True)
    s = st["spearman_waistkl_vs_rotwaist_all"]; ax.text(0.03, 0.97, f"$\\rho$ = {s['rho']:.2f} (CI {s['ci'][0]:.2f} to {s['ci'][1]:.2f}), n = 50", transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    ax = fig.add_subplot(gs[1, 1]); scatter(ax, "R_ex0_matched", "waist_kl_seq_v1", "$R_{\\mathrm{ex0}}$ (matched probes)", f"matched waist KL, $\\epsilon$ = {E} (nats)", "e")
    s = st["spearman_waistkl_vs_Rex0_all"]; ax.text(0.03, 0.97, f"$\\rho$ = {s['rho']:.2f} (CI {s['ci'][0]:.2f} to {s['ci'][1]:.2f}), n = 50", transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    ax.axvline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    handles = [Line2D([], [], marker="o", color=col_d1[c], ls="none", ms=3.2, mew=0.5, mec="white", label=D1_LAB[c]) for c in D1_ORDER]
    handles += [Line2D([], [], marker="s", color=RAMP6[i], ls="none", ms=3.0, mew=0.5, mec="white", label=f"k = {k}") for i, k in enumerate(KS)]
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "f")
    klr = np.concatenate([(r["kl_seq_v1"][1:11, IE] / r["kl_seq_rand"][1:11, IE]) for r in rows]); gr2 = np.concatenate([(r["gain_v1"][1:11, IE] / r["gain_rand"][1:11, IE]) ** 2 for r in rows])
    ax.plot(gr2, klr, "o", color=CAT[0], ms=1.6, mew=0, alpha=0.45, ls="none", zorder=3)
    lo, hi = min(gr2.min(), klr.min()) * 0.8, max(gr2.max(), klr.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("(gain along $v_1$ / gain along random)$^2$"); ax.set_ylabel("KL along $v_1$ / KL along random"); tidy(ax, grid="both")
    ds = st["direction_specificity"]; ax.text(0.03, 0.97, f"all reachable positions, blocks 1–10, 50 models:\nmedian {ds['kl_ratio_median']:.0f}× random; {ds['excess_median']:.1f}× the squared\ngain ratio (5th–95th {ds['excess_p5']:.1f}–{ds['excess_p95']:.1f}).\nDashed line exact only at the\nperturbed position (Fig. 3a).", transform=ax.transAxes, fontsize=5.6, color=INK2, va="top", linespacing=1.25, zorder=5, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    # g: per-block, real vs shuffled: sigma_1 and KL side by side as ratios real/shuffled; h: dose curves; i: legend + block-level rho histogram
    ax = fig.add_subplot(gs[2, 0]); panel(ax, "g")
    re = np.array([r["kl_seq_v1"][:, IE] for r in d1 if r["cond"] == "real"]); sh = np.array([r["kl_seq_v1"][:, IE] for r in d1 if r["cond"] == "shuffled"])
    re_s = np.array([r["sigma1"] for r in d1 if r["cond"] == "real"]); sh_s = np.array([r["sigma1"] for r in d1 if r["cond"] == "shuffled"])
    ax.plot(blocks, re.mean(0) / sh.mean(0), color=CAT[0], lw=0.9, marker="o", ms=1.8, mew=0, label="downstream KL along $v_1$")
    ax.plot(blocks, re_s.mean(0) / sh_s.mean(0), color=CAT[1], lw=0.9, marker="o", ms=1.8, mew=0, label="$\\sigma_1$")
    ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel("real / shuffled (ratio of means)"); tidy(ax); ax.legend(loc="lower right", handlelength=1.2)
    ax = fig.add_subplot(gs[2, 1]); panel(ax, "h")
    for c, colr in (("real", CAT[0]), ("shuffled", CAT[1])):
        rs = [r for r in d1 if r["cond"] == c]
        M = np.array([r["kl_seq_v1"][WAIST].mean(0) for r in rs]); Mr = np.array([r["kl_seq_rand"][WAIST].mean(0) for r in rs])
        ax.plot(EPS, M.mean(0), color=colr, lw=0.9, marker="o", ms=2.2, mew=0, label=f"{D1_LAB[c]}, $v_1$")
        ax.plot(EPS, Mr.mean(0), color=colr, lw=0.9, marker="o", ms=2.2, mew=0, ls=(0, (3, 2)), label=f"{D1_LAB[c]}, random")
    e = np.array(EPS); ax.plot(e, M.mean(0)[2] * (e / E) ** 2, color=MUTED, lw=0.5, ls=":", label="$\\epsilon^2$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("relative dose $\\epsilon$"); ax.set_ylabel("waist downstream KL (nats)"); tidy(ax, grid="both"); ax.legend(loc="upper left", handlelength=1.4, fontsize=5.5)
    ax = fig.add_subplot(gs[2, 2]); panel(ax, "i")
    b_rho = np.array([r["spearman_blocks_sigma1_kl"] for r in rows])
    ax.hist(b_rho, bins=np.linspace(-1, 1, 21), color=CAT[0], edgecolor="white", lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_xlabel("$\\rho$($\\sigma_1$, KL along $v_1$), blocks 1–10"); ax.set_ylabel("models"); tidy(ax)
    ax.text(0.03, 0.97, f"median {np.median(b_rho):.2f}\n{int((b_rho > 0).sum())} of 50 positive", transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=5.5, handletextpad=0.3, columnspacing=0.9, bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.savefig(os.path.join(OUT, "edfig7_new.png")); fig.savefig(os.path.join(OUT, "edfig7_new.pdf")); plt.close(fig)


def figure_main(rows, st):
    """Fig. 4: the central matched result. a sigma_1 per block (d1), b downstream KL per block (d1), c within-model rank
    correlation, d across-model association with R_ex0 (populations marked), e direction specificity."""
    d1 = [r for r in rows if r["exp"] == "d1"]; kg = [r for r in rows if r["exp"] == "kgram"]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.99, top=0.95, bottom=0.12, wspace=0.45, hspace=0.55)
    blocks = np.arange(12)
    col_d1 = {"real": CAT[0], "bidir": CAT[3], "shuffled": CAT[1], "randlab": CAT[2]}
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    for c in D1_ORDER:
        rs = [r for r in d1 if r["cond"] == c]; M = np.array([r["sigma1"] for r in rs])
        ax.plot(blocks, M.mean(0), color=col_d1[c], lw=0.9, marker="o", ms=1.8, mew=0, label=D1_LAB[c])
        ax.fill_between(blocks, M.min(0), M.max(0), color=col_d1[c], alpha=0.15, lw=0)
    ax.axvspan(3.5, 7.5, color="#f3f3f0", zorder=0)
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel("local gain $\\sigma_1$"); tidy(ax); ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, fontsize=5.5)
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    for c in D1_ORDER:
        rs = [r for r in d1 if r["cond"] == c]; M = np.array([r["kl_seq_v1"][:, IE] for r in rs])
        ax.plot(blocks, M.mean(0), color=col_d1[c], lw=0.9, marker="o", ms=1.8, mew=0)
        ax.fill_between(blocks, M.min(0), M.max(0), color=col_d1[c], alpha=0.15, lw=0)
    ax.axvspan(3.5, 7.5, color="#f3f3f0", zorder=0)
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel(f"downstream KL along $v_1$ (nats)"); tidy(ax)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    groups = [(c, D1_LAB[c], col_d1[c]) for c in D1_ORDER] + [(f"k{k}", f"k = {k}", RAMP6[i]) for i, k in enumerate(KS)]
    rng = np.random.default_rng(0)
    for gi, (g, lab, colr) in enumerate(groups):
        rs = [r for r in rows if r["group"] == g]
        x = gi + rng.uniform(-0.18, 0.18, len(rs))
        ax.plot(x, [r["spearman_blocks_sigma1_kl"] for r in rs], "o", color=colr, ms=3.0, mew=0.4, mec="white", ls="none", zorder=4)
        ax.plot(x, [r["spearman_blocks_sigma1_S"] for r in rs], "o", mfc="none", mec=colr, ms=3.0, mew=0.7, ls="none", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1); ax.axvline(3.5, color=GRID, lw=0.5, zorder=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([lab for _, lab, _ in groups], rotation=60, ha="right", fontsize=5.2)
    b_rho = np.array([r["spearman_blocks_sigma1_kl"] for r in rows]); b_S = np.array([r["spearman_blocks_sigma1_S"] for r in rows])
    ax.set_ylim(-1.05, 1.6); ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_ylabel("$\\rho$($\\sigma_1$, downstream effect), blocks 1–10"); tidy(ax)
    hs = [Line2D([], [], marker="o", color=INK2, ls="none", ms=3.0, mew=0.4, mec="white", label=f"relative dose $\\epsilon$: median {np.median(b_rho):.2f}, {int((b_rho > 0).sum())} of 50 > 0"),
          Line2D([], [], marker="o", mfc="none", mec=INK2, ls="none", ms=3.0, mew=0.7, label=f"per unit $\\|\\delta\\|^2$: median {np.median(b_S):.2f}, {int((b_S > 0).sum())} of 50 > 0")]
    ax.legend(handles=hs, loc="upper left", fontsize=5.2, handletextpad=0.3, frameon=False, borderaxespad=0.2)
    ax = fig.add_subplot(gs[1, 0:2]); panel(ax, "d", dx=-0.075, dy=1.06)
    for c in D1_ORDER:
        rs = [r for r in d1 if r["cond"] == c]
        ax.plot([r["R_ex0_matched"] for r in rs], [r["waist_kl_seq_v1"] for r in rs], "o", color=col_d1[c], ms=3.8, mew=0.5, mec="white", ls="none", label=D1_LAB[c], zorder=4)
    for i, k in enumerate(KS):
        rs = [r for r in kg if int(r["cond"]) == k]
        ax.plot([r["R_ex0_matched"] for r in rs], [r["waist_kl_seq_v1"] for r in rs], "s", color=RAMP6[i], ms=3.4, mew=0.5, mec="white", ls="none", label=f"k = {k}", zorder=4)
    ax.axvline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log"); ax.set_xlabel("$R_{\\mathrm{ex0}}$ (matched probes)"); ax.set_ylabel(f"middle-block KL, $\\epsilon$ = {E} (nats)"); tidy(ax, grid="both")
    sa, sk, sd = st["spearman_waistkl_vs_Rex0_all"], st["spearman_waistkl_vs_Rex0_kgram"], st["spearman_waistkl_vs_Rex0_d1"]
    ax.text(0.02, 0.97, f"all 50: $\\rho$ = {sa['rho']:.2f} (CI {sa['ci'][0]:.2f} to {sa['ci'][1]:.2f})\nk-gram: {sk['rho']:.2f} ({sk['ci'][0]:.2f} to {sk['ci'][1]:.2f}); training conditions: {sd['rho']:.2f} ({sd['ci'][0]:.2f} to {sd['ci'][1]:.2f})",
            transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    ax.legend(loc="lower right", ncol=5, fontsize=5.3, handletextpad=0.3, columnspacing=0.7, frameon=False, bbox_to_anchor=(1.0, 0.16)); ax.set_ylim(5e-7, 2e-2)
    # e: direction specificity at the perturbed position, where the squared gain ratio is the exact isotropic
    # prediction under a causal mask (the five bidirectional models are excluded for that reason)
    ax = fig.add_subplot(gs[1, 2]); panel(ax, "e", dx=-0.17, dy=1.06)
    causal = [r for r in rows if r["cond"] != "bidir"]
    klt = np.concatenate([(r["kl_t_v1"][1:11, IE] / r["kl_t_rand"][1:11, IE]) for r in causal])
    gr2 = np.concatenate([(r["gain_v1"][1:11, IE] / r["gain_rand"][1:11, IE]) ** 2 for r in causal])
    ax.plot(gr2, klt, "o", color=CAT[0], ms=1.6, mew=0, alpha=0.45, ls="none", zorder=3)
    lo, hi = min(gr2.min(), klt.min()) * 0.8, max(gr2.max(), klt.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("(gain along $v_1$ / gain along random)$^2$"); ax.set_ylabel("same-position KL, $v_1$ / random"); tidy(ax, grid="both")
    ds = st["direction_specificity_samepos"]
    ax.text(0.03, 0.97, f"median {ds['kl_t_ratio_median']:.1f}x a random direction\n{ds['excess_t_median']:.1f}x the isotropic prediction\n{ds['n_models']} causal models, blocks 1-10", transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    fig.savefig(os.path.join(OUT, "fig4_matched.png")); fig.savefig(os.path.join(OUT, "fig4_matched.pdf")); plt.close(fig)


if __name__ == "__main__":
    rows = load(); st = analyse(rows)
    os.makedirs(f"{R}/matched", exist_ok=True)
    json.dump(st, open(f"{R}/matched/matched_stats.json", "w"), indent=1, default=float)
    figure(rows, st); figure_main(rows, st)
    def f(d): return f"{d['rho']:.2f} (CI {d['ci'][0]:.2f} to {d['ci'][1]:.2f}, p {d['p']:.3g}, n {d['n']})"
    print("models", st["n_models"], "GPU", st["gpu"], "total", st["elapsed_total_min"], "min")
    print("gain(eps 0.01)/sigma1:", st["gain_eps0.01_over_sigma1"])
    print("R_ex0 matched vs survey:", st["R_ex0_matched_vs_survey"])
    print("KL dose slope 0.03-0.3:", st["kl_seq_loglog_slope_0.03_0.3"], "eps1/eps0.3:", st["kl_seq_ratio_eps1_over_eps0.3"])
    print("\nd1 conditions (waist KL along v1 at eps 0.1 | dCE | rotation waist dL | R_ex0 matched | R of KL profile | v1/rand ratio | blocks rho sigma1-KL):")
    for c in D1_ORDER:
        d = st["d1"][c]; print(f"  {c:9s} n={d['n']}  KL {d['waist_kl_mean']:.5f} [{min(d['waist_kl']):.5f}-{max(d['waist_kl']):.5f}]  dCE {d['waist_dce_mean']:+.5f}  rot {d['rot_waist_mean']:.3f}  R {d['R_ex0_mean']:.2f}  R_kl {d['R_kl_mean']:.2f}  v1/rand {d['waist_ratio_v1_rand_mean']:.1f}  rho_blocks {d['blocks_rho_sigma1_kl_mean']:+.2f}")
    print("real / other, matched KL:", {k: round(v, 1) for k, v in st["d1_ratio_real_over"].items()}, " rotation:", {k: round(v, 1) for k, v in st["d1_rot_ratio_real_over"].items()})
    print("\nk-gram:")
    for k in KS:
        d = st["kgram"][k]; print(f"  k={k} n={d['n']}  KL {d['waist_kl_mean']:.5f} [{min(d['waist_kl']):.5f}-{max(d['waist_kl']):.5f}]  dCE {d['waist_dce_mean']:+.5f}  rot {d['rot_waist_mean']:.3f}  R {d['R_ex0_mean']:.2f}  R_kl {d['R_kl_mean']:.2f}  v1/rand {d['waist_ratio_v1_rand_mean']:.1f}  rho_blocks {d['blocks_rho_sigma1_kl_mean']:+.2f}")
    print("k vs waist KL:", f(st["kgram_spearman_k_vs_waist_kl"]), "| k vs dCE:", f(st["kgram_spearman_k_vs_waist_dce"]), "| k vs rot:", f(st["kgram_spearman_k_vs_rot_waist"]), "| k8/k1", st["kgram_ladder_ratio_k8_over_k1"])
    for name in ("all", "d1", "kgram"):
        print(f"{name}: waistKL vs rotWaist {f(st[f'spearman_waistkl_vs_rotwaist_{name}'])} | waistKL vs R_ex0 {f(st[f'spearman_waistkl_vs_Rex0_{name}'])} | waist dCE vs R_ex0 {f(st[f'spearman_waistdce_vs_Rex0_{name}'])} | R_kl vs R_ex0 {f(st[f'spearman_Rkl_vs_Rex0_{name}'])}")
    print("blocks rho sigma1-KL:", st["blocks_rho_sigma1_kl"], "\nblocks rho KL-rotation:", st["blocks_rho_kl_rot"])
    a = st["blocks_rho_sigma1_S_abs"]; print("perturbation-scaling check (S = KL/||delta||^2):", {k: v for k, v in a.items() if k != "by_group"})
    g = st["blocks_rho_grel_kl"]; print("activation-normalised gain vs relative KL:", {k: v for k, v in g.items() if k != "by_group"}); print("  by group:", {k: (round(min(v), 2), round(max(v), 2)) for k, v in g["by_group"].items()})
    da = st["depth_adjusted"]; print("depth-adjusted:", {k: v for k, v in da.items() if k != "by_group"})
    for g, v in da["by_group"].items(): print(f"  {g:9s} " + "  ".join(f"{k} {min(x):+.2f}..{max(x):+.2f}" for k, x in v.items()))
    print("matched-region ratio (middle 4-7):", json.dumps(st["R_m47_vs_R_ex0"], default=float)[:600])
    print("direction specificity, same position, causal models:", json.dumps(st["direction_specificity_samepos"], default=float)[:900])
    for g, v in a["by_group"].items(): print(f"  {g:9s} rel {min(v['rel']):+.2f}..{max(v['rel']):+.2f}  abs {min(v['abs']):+.2f}..{max(v['abs']):+.2f}  rho(block,sigma1) {min(v['rho_block_sigma1']):+.2f}..{max(v['rho_block_sigma1']):+.2f}  ||h|| b10/b1 {min(v['hnorm_ratio']):.1f}..{max(v['hnorm_ratio']):.1f}")
    print("direction specificity:", json.dumps(st["direction_specificity"], indent=0)[:900])
