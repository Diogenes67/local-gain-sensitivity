"""Extended Data Tables 3 (NMI numbering), 4 and 5 for the NMI manuscript (23 Sept 2026).

ED Table 1  ed_table1.md with the Supplementary Note pointer replaced (ed_table1_nmi.md)
ED Table 3  provenance table from make_tables.py (tables/ed_table3.md) with display-item references mapped to the NMI layout (Pythia-410M checkpoint row from E7, 24 Sept 2026),
            the pruning row removed and rows added for the pretrained matched assay, the 30-input survey and the Pythia-410M
            checkpoints (retained summary, pending E7)
ED Table 4  gain against consequence beyond depth: held-out profile prediction (controlled L1O, LOCO, leave-one-set-out;
            pretrained leave-one-model-out and leave-one-family-out), pooled slopes under three depth specifications,
            condition-specific and per-model slopes, and within-model correlations (Spearman, block-partialled, first differences)
ED Table 5  condition-level gain ratios and middle-block intervention costs: the four training conditions, the two switching
            conditions, training-step trajectories, the k-gram and entropy-matched families and the 24-layer repeat

Sources: regression_check.json, loco_check.json, matched_pretrained/matched_pretrained_stats.json, fig2_scale_nmi_check.json,
matched_analysis.load() + depth_adjusted() (NCS_R = folder holding matched/results, d1/results_v21, reprofile/results_kgram),
markov/markov_stats.json, scale/scale_stats.json.
Run: NCS_R=<folder> python make_ed_tables_nmi.py -> tables/ed_table3_nmi.md, ed_table4.md, ed_table5.md, ed_tables45_check.json
"""
import os, sys, json, re
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matched_analysis as ma
if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
OUT = os.path.join(HERE, "tables")
J = lambda p: json.load(open(os.path.join(HERE, p)))
M = "−"


def f(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", M) if x < 0 else s


def pair(a, b, d=2):
    return f"{f(a, d)} / {f(b, d)}"


def rng(v, d=None):
    v = np.asarray(v, float)
    if d is None:
        d = 2 if np.abs(v).max() >= 1 else 3
    return f"{f(v.min(), d)} to {f(v.max(), d)}" if v.min() < 0 else f"{f(v.min(), d)}–{f(v.max(), d)}"


def msd(v, d=2):
    v = np.asarray(v, float)
    return f"{f(v.mean(), d)} ± {f(v.std(ddof=1), d)}"


# ---------------------------------------------------------------- ED Table 4
def ed_table4(rows, dep):
    rc, lc = J("regression_check.json"), J("loco_check.json")
    mp = J("matched_pretrained/matched_pretrained_stats.json")
    GL = {"real": "real text", "bidir": "bidirectional", "shuffled": "shuffled text", "randlab": "random labels",
          "k1": "k = 1", "k2": "k = 2", "k3": "k = 3", "k4": "k = 4", "k5": "k = 5", "k8": "k = 8"}
    T = []
    add = lambda *c: T.append("| " + " | ".join(c) + " |")
    A, R_ = lc["y_abs"], lc["y_rel"]
    # a. held-out prediction
    add("**a** Held-out profile prediction", "controlled, leave one model out", "50 folds", pair(A["L1O"]["r2_gain"], A["L1O"]["r2_block_only"]), pair(R_["L1O"]["r2_gain"], R_["L1O"]["r2_block_only"]), "centred R², gain + depth / depth only")
    add("", "controlled, leave one condition out", "10 folds pooled", pair(A["LOCO"]["r2_gain"], A["LOCO"]["r2_block_only"]), pair(R_["LOCO"]["r2_gain"], R_["LOCO"]["r2_block_only"]), "")
    add("", "", "4 text conditions pooled", pair(A["LOCO"]["by_set"]["training_conditions"]["r2_gain"], A["LOCO"]["by_set"]["training_conditions"]["r2_block_only"]),
        pair(R_["LOCO"]["by_set"]["training_conditions"]["r2_gain"], R_["LOCO"]["by_set"]["training_conditions"]["r2_block_only"]), "each held out once")
    add("", "", "6 k-gram orders pooled", pair(A["LOCO"]["by_set"]["kgram"]["r2_gain"], A["LOCO"]["by_set"]["kgram"]["r2_block_only"]),
        pair(R_["LOCO"]["by_set"]["kgram"]["r2_gain"], R_["LOCO"]["by_set"]["kgram"]["r2_block_only"]), "other orders in training")
    for g in GL:
        a_, r_ = A["LOCO"]["by_condition"][g], R_["LOCO"]["by_condition"][g]
        add("", "", f"held out: {GL[g]}", pair(a_["r2_gain"], a_["r2_block_only"]), pair(r_["r2_gain"], r_["r2_block_only"]),
            f"slope fitted without it {f(A['LOCO']['train_slope'][g])} / {f(R_['LOCO']['train_slope'][g])}")
    add("", "controlled, leave one set out", "fit 20 text models, predict 30 k-gram", pair(A["LOSO"]["by_set"]["kgram"]["r2_gain"], A["LOSO"]["by_set"]["kgram"]["r2_block_only"]),
        pair(R_["LOSO"]["by_set"]["kgram"]["r2_gain"], R_["LOSO"]["by_set"]["kgram"]["r2_block_only"]), f"slope fitted on text {f(A['LOSO']['train_slope']['kgram'])} / {f(R_['LOSO']['train_slope']['kgram'])}")
    add("", "", "fit 30 k-gram models, predict 20 text", pair(A["LOSO"]["by_set"]["training_conditions"]["r2_gain"], A["LOSO"]["by_set"]["training_conditions"]["r2_block_only"]),
        pair(R_["LOSO"]["by_set"]["training_conditions"]["r2_gain"], R_["LOSO"]["by_set"]["training_conditions"]["r2_block_only"]), f"slope fitted on k-gram {f(A['LOSO']['train_slope']['training_conditions'])} / {f(R_['LOSO']['train_slope']['training_conditions'])}")
    ta, tr = mp["transfer"]["y_abs"], mp["transfer"]["y_rel"]
    add("", "pretrained, leave one model out", "6 folds", pair(ta["l1o_gain"], ta["l1o_depth"]), pair(tr["l1o_gain"], tr["l1o_depth"]), "depth = block / (L − 1)")
    add("", "pretrained, leave one family out", "5 folds (two Pythias together)", pair(ta["lofo_gain"], ta["lofo_depth"]), pair(tr["lofo_gain"], tr["lofo_depth"]), "")
    # b. slopes
    a0, r0 = rc["y_abs"], rc["y_rel"]
    ci = lambda m: f"{f(m['boot']['ci'][0][0])} to {f(m['boot']['ci'][1][0])}"
    add("**b** Pooled log–log slope β", "controlled, model effects + linear block (reported)", "50 models, 500 rows",
        f"{f(a0['M0']['slope'])} (s.e. {f(a0['M0']['se_cluster_model'])}; {ci(a0['M0'])})", f"{f(r0['M0']['slope'])} (s.e. {f(r0['M0']['se_cluster_model'])}; {ci(r0['M0'])})",
        "model-clustered s.e.; bootstrap 95% CI over models")
    add("", "controlled, separate effect per block", "", f"{f(a0['M1']['slope'])} (s.e. {f(a0['M1']['se_cluster_model'])})", f"{f(r0['M1']['slope'])} (s.e. {f(r0['M1']['se_cluster_model'])})", "")
    add("", "controlled, linear depth trend per model", "", f"{f(a0['M2']['slope'])} (s.e. {f(a0['M2']['se_cluster_model'])})", f"{f(r0['M2']['slope'])} (s.e. {f(r0['M2']['se_cluster_model'])})", "relative-dose sign depends on the depth specification")
    for g in GL:
        a_, r_ = a0["M3"]["by_group"][g], r0["M3"]["by_group"][g]
        add("", "within one condition" if g == "real" else "", GL[g], f"{f(a_['slope'])} ({f(a_['boot_ci'][0])} to {f(a_['boot_ci'][1])})",
            f"{f(r_['slope'])} ({f(r_['boot_ci'][0])} to {f(r_['boot_ci'][1])})", "five models; bootstrap over models" if g == "real" else "")
    wa, wr = a0["W"], r0["W"]
    add("", "per model, block as covariate", "50 models", f"median {f(wa['median'])} (IQR {f(wa['q25'])} to {f(wa['q75'])}); {wa['n_positive']} positive, {wa['n_above_1']} above 1",
        f"median {f(wr['median'])} (IQR {f(wr['q25'])} to {f(wr['q75'])}); {wr['n_positive']} positive", "")
    pa, pr = mp["pooled"]["y_abs"], mp["pooled"]["y_rel"]
    add("", "pretrained, model effects + linear depth", "6 models", f"{f(pa['slope'])} (s.e. {f(pa['se_cluster'])})", f"{f(pr['slope'])} (s.e. {f(pr['se_cluster'])})", "six clusters")
    add("", "pretrained, linear depth trend per model", "", f"{f(pa['slope_model_trends'])} (s.e. {f(pa['se_model_trends'])})", f"{f(pr['slope_model_trends'])} (s.e. {f(pr['se_model_trends'])})", "")
    # c. within-model correlations
    sa = np.array([r["spearman_blocks_sigma1_S"] for r in rows]); sr = np.array([r["spearman_blocks_sigma1_kl"] for r in rows])
    mc = lambda v: f"median {f(np.median(v))}; {int((np.asarray(v) > 0).sum())} of {len(v)} positive"
    add("**c** Within-model rank correlation, blocks 1–10", "controlled, Spearman σ₁ with consequence", "50 models", mc(sa), mc(sr), "")
    add("", "controlled, block partialled out", "", mc([r["partial_abs"] for r in rows]), mc([r["partial_rel"] for r in rows]), "")
    add("", "controlled, first differences over adjacent blocks", "", mc([r["diff_abs"] for r in rows]), mc([r["diff_rel"] for r in rows]), "")
    add("", "pretrained, Spearman σ₁ with consequence", "6 models, blocks 1 to L − 2", mc([m["rho_abs"] for m in mp["models"]]), mc([m["rho_rel"] for m in mp["models"]]), "input-resampled CIs in Fig. 2f")
    # d. predictor sets on identical folds (predictor_check.py; review 5, item 1) and e. same-position readout (item 4)
    pc = J("predictor_check.json"); rc_ = J("readout_check.json")
    S4 = ("depth", "depth+norm", "depth+randgain", "depth+gain")
    four = lambda e: " / ".join(f(e[n]["r2"]) for n in S4)
    four_set = lambda e, key, s_: " / ".join(f(e[n][key][s_]) for n in S4)
    dl = lambda e: (lambda d_: f"Δ(σ₁ − random gain) {f(d_['point'])} ({f(d_['ci'][0])} to {f(d_['ci'][1])})")(e["depth+randgain"]["delta_r2_depth_gain_minus_this"])
    C, P = pc["controlled"], pc["pretrained"]
    add("**d** Predictor sets on identical folds", "controlled, leave one model out", "50 folds", four(C["y_abs"]["L1O"]), four(C["y_rel"]["L1O"]), "depth / + norm / + random-direction gain / + σ₁; " + dl(C["y_abs"]["L1O"]))
    add("", "controlled, leave one condition out", "10 folds pooled", four(C["y_abs"]["LOCO"]), four(C["y_rel"]["LOCO"]), dl(C["y_abs"]["LOCO"]))
    add("", "", "4 text conditions pooled", four_set(C["y_abs"]["LOCO"], "by_set", "training_conditions"), four_set(C["y_rel"]["LOCO"], "by_set", "training_conditions"), "")
    add("", "", "6 k-gram orders pooled", four_set(C["y_abs"]["LOCO"], "by_set", "kgram"), four_set(C["y_rel"]["LOCO"], "by_set", "kgram"), "")
    add("", "controlled, leave one set out", "fit 20 text models, predict 30 k-gram", four_set(C["y_abs"]["LOSO"], "by_set", "kgram"), four_set(C["y_rel"]["LOSO"], "by_set", "kgram"), "")
    add("", "", "fit 30 k-gram models, predict 20 text", four_set(C["y_abs"]["LOSO"], "by_set", "training_conditions"), four_set(C["y_rel"]["LOSO"], "by_set", "training_conditions"), "")
    add("", "pretrained, leave one model out", "6 folds", four(P["y_abs"]["L1O"]), four(P["y_rel"]["L1O"]), dl(P["y_abs"]["L1O"]))
    add("", "pretrained, leave one family out", "5 folds", four(P["y_abs"]["LOFO"]), four(P["y_rel"]["LOFO"]), dl(P["y_abs"]["LOFO"]) + "; per-family values with intervals in Fig. 2g")
    add("**e** Same-position readout D~t~", "controlled, leave one model out", "50 folds", four(C["y_abs_t"]["L1O"]), four(C["y_rel_t"]["L1O"]), "the four sets as in d; " + dl(C["y_abs_t"]["L1O"]))
    add("", "controlled, leave one condition out", "10 folds pooled", four(C["y_abs_t"]["LOCO"]), four(C["y_rel_t"]["LOCO"]), dl(C["y_abs_t"]["LOCO"]))
    add("", "controlled, leave one set out", "fit 20 text models, predict 30 k-gram", four_set(C["y_abs_t"]["LOSO"], "by_set", "kgram"), four_set(C["y_rel_t"]["LOSO"], "by_set", "kgram"), "")
    add("", "", "fit 30 k-gram models, predict 20 text", four_set(C["y_abs_t"]["LOSO"], "by_set", "training_conditions"), four_set(C["y_rel_t"]["LOSO"], "by_set", "training_conditions"), "")
    add("", "pretrained, leave one family out", "5 folds", four(P["y_abs_t"]["LOFO"]), four(P["y_rel_t"]["LOFO"]), dl(P["y_abs_t"]["LOFO"]))
    ct, pt = rc_["controlled"], rc_["pretrained"]
    add("", "controlled, pooled slope and within-model ρ", "50 models", f"{f(ct['y_abs_t']['slope'])} (s.e. {f(ct['y_abs_t']['se'])}); median ρ {f(ct['y_abs_t']['rho_median'])}, {ct['y_abs_t']['n_pos']} of 50 positive",
        f"{f(ct['y_rel_t']['slope'])} (s.e. {f(ct['y_rel_t']['se'])}); median ρ {f(ct['y_rel_t']['rho_median'])}, {ct['y_rel_t']['n_pos']} of 50 positive", "all-positions values in b and c")
    add("", "pretrained, pooled slope and within-model ρ", "6 models", f"{f(pt['y_abs_t']['slope'])} (s.e. {f(pt['y_abs_t']['se'])}); median ρ {f(pt['y_abs_t']['rho_median'])}, {pt['y_abs_t']['n_pos']} of 6 positive",
        f"{f(pt['y_rel_t']['slope'])} (s.e. {f(pt['y_rel_t']['se'])}); median ρ {f(pt['y_rel_t']['rho_median'])}, {pt['y_rel_t']['n_pos']} of 6 positive", "")
    hdr = "| Section | Analysis | Subset | Normalised sensitivity | Relative dose (ε = 0.1) | Note |\n|---|---|---|---|---|---|\n"
    foot = ("\n\nNormalised sensitivity is the same-sequence divergence along v₁ divided by the squared displacement; relative dose is the divergence "
            "at a displacement of 0.1‖h‖. Centred R² is 1 − SS(residual)/SS(total) of the within-model-centred log profile pooled over the held-out models, "
            "with β and γ fitted on the remaining models (loco_check.py, matched_pretrained_analysis.py); it measures the shape of a profile, not its level. "
            "Leave one condition out asks whether a condition absent from training is predicted when the other nine are present; the text-to-k-gram fold asks "
            "whether a whole source family is predicted from the other. Pooled slopes: regression_check.py; per-model and correlation values: matched_analysis.py. "
            "Section d compares four predictor sets on identical folds (predictor_check.py): block position; block position with the log activation norm; with the log gain "
            "along a random unit direction at ε = 0.01; and with log σ₁. Δ is the pooled R² of the σ₁ set minus that of the random-direction set on the same folds, with a "
            "95% percentile interval from 2,000 resamples of the held-out models. Section e repeats d for the divergence read at the perturbed position only (D~t~, the "
            "Fig. 3 readout) in place of the mean over the perturbed and later positions (D̄, the readout of a–c and of Fig. 2); readout_check.py.")
    md = hdr + "\n".join(T) + foot
    open(os.path.join(OUT, "ed_table4.md"), "w").write(md)
    return dict(n_rows=len(T), spearman_abs_median=float(np.median(sa)), spearman_rel_median=float(np.median(sr)),
                partial=dict(abs=float(np.median([r["partial_abs"] for r in rows])), rel=float(np.median([r["partial_rel"] for r in rows]))),
                diff=dict(abs=float(np.median([r["diff_abs"] for r in rows])), rel=float(np.median([r["diff_rel"] for r in rows]))),
                loso=dict(text_to_kgram=A["LOSO"]["by_set"]["kgram"], kgram_to_text=A["LOSO"]["by_set"]["training_conditions"]))


# ---------------------------------------------------------------- ED Table 5
def ed_table5():
    R = ma.R; T = []; chk = {}
    add = lambda *c: T.append("| " + " | ".join(c) + " |")
    def d1(cond, seeds):
        return [json.load(open(f"{R}/d1/results_v21/{cond}_s{s}.json")) for s in seeds]
    lab = {"real": "real text", "shuffled": "token-shuffled text", "randlab": "random labels", "bidir": "bidirectional masked text",
           "real2shuf": "real → shuffled at step 5,000", "shuf2real": "shuffled → real at step 5,000"}
    for i, cond in enumerate(("real", "shuffled", "randlab", "bidir")):
        ds = d1(cond, range(5))
        add("12 blocks, training conditions" if i == 0 else "", lab[cond], "5", msd([d["R_ex0"] for d in ds]), msd([d["early_over_mid"] for d in ds]),
            msd([d["late_over_mid"] for d in ds]), rng([d["regions"]["waist"]["dL"] for d in ds]), rng([d["skip_regions"]["waist"]["dL"] for d in ds]))
        chk[cond] = dict(R=[d["R_ex0"] for d in ds], rot=[d["regions"]["waist"]["dL"] for d in ds])
    for i, cond in enumerate(("real2shuf", "shuf2real")):
        ds = d1(cond, range(3))
        add("12 blocks, condition switching" if i == 0 else "", lab[cond], "3", msd([d["R_ex0"] for d in ds]), msd([d["early_over_mid"] for d in ds]),
            msd([d["late_over_mid"] for d in ds]), rng([d["regions"]["waist"]["dL"] for d in ds]), rng([d["skip_regions"]["waist"]["dL"] for d in ds]))
    first = True
    for cond in ("real", "shuffled"):
        ds = d1(cond, range(5))
        for step in sorted({int(s) for d in ds for s in d["trajectory"]}):
            tr = [d["trajectory"][str(step)] for d in ds if str(step) in d["trajectory"]]
            add("12 blocks, during training" if first else "", f"{lab[cond]}, step {step:,}", str(len(tr)), msd([t["R_ex0"] for t in tr]), msd([t["early_over_mid"] for t in tr]),
                msd([t["late_over_mid"] for t in tr]), rng([t["waist_dL_branch"] for t in tr]), "–")
            first = False
        add("", f"{lab[cond]}, step 10,000", "5", msd([d["R_ex0"] for d in ds]), msd([d["early_over_mid"] for d in ds]), msd([d["late_over_mid"] for d in ds]),
            rng([d["regions"]["waist"]["dL"] for d in ds]), "–")
    for i, k in enumerate((1, 2, 3, 4, 5, 8)):
        ds = [json.load(open(f"{R}/reprofile/results_kgram/kgram_{k}_s{s}.json")) for s in range(5)]
        st = [d["stats_indist"] for d in ds]
        add("12 blocks, k-gram sources" if i == 0 else "", f"k = {k}", "5", msd([s["R_ex0"] for s in st]), msd([s["early_over_mid"] for s in st]), msd([s["late_over_mid"] for s in st]),
            rng([d["regions_branch"]["waist"]["dL"] for d in ds]), rng([d["skip_regions"]["waist"]["dL"] for d in ds]))
        chk[f"k{k}"] = dict(R=[s["R_ex0"] for s in st], rot=[d["regions_branch"]["waist"]["dL"] for d in ds])
    mk = J("markov/markov_stats.json")["per_order"]
    for i, m in enumerate(sorted(mk, key=int)):
        o = mk[m]
        add("12 blocks, entropy-matched (1.00 nats)" if i == 0 else "", f"m = {m} ({o['n_states']:,} contexts)", str(o["n"]), msd(o["R"]), msd(o["EM"]), msd(o["LM"]), rng(o["rot_mid"]), rng(o["skip_mid"]))
        chk[f"m{m}"] = dict(R=o["R"], rot=o["rot_mid"])
    sc = J("scale/scale_stats.json")
    for i, cond in enumerate(("real", "shuffled")):
        o = sc[cond]
        add("24 blocks" if i == 0 else "", lab[cond], str(len(o["R"])), msd(o["R"]), msd(o["EM"]), msd(o["LM"]), rng(o["rot_mid"]), rng(o["skip_mid"]))
        chk[f"L24_{cond}"] = dict(R=o["R"], rot=o["rot_mid"])
    hdr = ("| Setting | Condition | n | Middle-to-edge gain ratio | Early / middle | Late / middle | Middle-block rotation cost (nats) | Middle-block skip cost (nats) |\n"
           "|---|---|---|---|---|---|---|---|\n")
    foot = ("\n\nGain ratios are mean ± s.d. over seeds under the canonical estimator (thirds of blocks 1 to L − 1: 1–3, 4–8, 9–11 at 12 blocks; 1–7, 8–16, "
            "17–23 at 24). Intervention costs are the range over seeds of the loss increase when the branch outputs of the middle blocks are rotated (Haar "
            "rotation, dose 1) or the blocks are skipped (identity), blocks 4–7 at 12 blocks and 8–15 at 24. Switching models are read out on the objective "
            "of their final phase. Trajectory values are from the checkpoints saved during the same runs. Entropy-matched sources have conditional entropy "
            "1.00 nats at every order m over eight tokens. Sources: d1/results_v21, reprofile/results_kgram, markov/markov_stats.json, scale/scale_stats.json.")
    open(os.path.join(OUT, "ed_table5.md"), "w").write(hdr + "\n".join(T) + foot)
    return dict(n_rows=len(T), values=chk)


# ---------------------------------------------------------------- ED Table 3 (NMI references)
def ed_table3_nmi():
    s = open(os.path.join(OUT, "ed_table3.md")).read()
    lines = s.split("\n")
    MAP = [("Estimator validation (Fig. 1b; Methods)", "Estimator validation (Fig. 1d; Methods)"),
           ("Random-direction probing (Fig. 1c)", "Random-direction probing (Fig. 1e)"),
           ("Survey of 45 pretrained models (Fig. 5a–e; ED Table 1)", "Survey of 45 pretrained models (Fig. 5a,b; ED Table 1)"),
           ("Destroyed-data experiment (Fig. 2a–d)", "Destroyed-data experiment (Fig. 4a,b; ED Table 5)"),
           ("Condition switching (Fig. 2d)", "Condition switching (ED Table 5)"),
           ("Dependency-order sweep (Fig. 3)", "Dependency-order sweep (Fig. 4c; ED Table 5)"),
           ("Matched token-local perturbation assay (Fig. 4; ED Fig. 7; SN 17)", "Matched token-local perturbation assay (Figs 2a–d and 3a; ED Fig. 7; ED Table 4)"),
           ("Downstream linearised response, pilot (Methods)", "Downstream linearised response (Fig. 3c–e; Methods)"),
           ("Entropy-matched Markov family (Fig. 3d–f)", "Entropy-matched Markov family (Fig. 4d,e; ED Table 5)"),
           ("Scale point in depth (Fig. 2e–g)", "Scale point in depth (Fig. 2e; ED Table 5)"),
           ("Pretrained census (Fig. 5f; ED Table 2; ED Fig. 6)", "Pretrained census (Fig. 5b; ED Table 2; ED Fig. 6)")]
    out, dropped = [], []
    for l in lines:
        if l.startswith("| Layer and unstructured pruning"):
            dropped.append(l[:60]); continue
        for a, b in MAP:
            if l.startswith("| " + a):
                l = l.replace(a, b, 1)
        out.append(l)
    # rows added after the census row
    ncol = out[0].count("|") - 1
    new = [["Matched assay in pretrained decoders (Figs 2f and 3b–e; ED Table 4)", "σ~1~ and matched displacement along v~1~ and norm-matched random directions", "token-local, natural context, natural inputs, 20 inputs × 6 positions",
            "displacement at ε ∈ {0.01, 0.03, 0.1, 0.3}; linearised response", "float32", "GPT-2 124M, Pythia-410M, Pythia-1.4B, Llama-3.2-1B, Qwen2.5-1.5B, Gemma-2-2B",
            "matched_pretrained.py, matched_pretrained_analysis.py, e1_extra.py", "yes", "blocks 1 to L − 2; WikiText-103 validation; 500 input resamples"],
           ["30-input survey (Fig. 5a; Methods)", "σ~1~", "token-local, natural context, natural inputs, 30 inputs × 8 positions", "–", "float32",
            "11 models of Figs 2, 3 and 5", "survey_sigma1_v2.py (SURVEY_N_INPUTS = 30), e3_analysis.py", "yes", "whole-input bootstrap; five-input subsets"],
           ["Pythia-410M public checkpoints (Fig. 5c; ED Fig. 4c,d)", "σ~1~; branch-rotation cost", "token-local, natural context, natural inputs, 30 inputs × 8 positions",
            "branch rotation, dose 1, one block at a time; next-token loss (census protocol)", "float32", "11 public checkpoints, steps 0 to 143,000",
            "e7_checkpoints.py (imports survey_sigma1_v2.py; census rotation, data and seeds)", "yes",
            "30 batches of 16 × 128 tokens, WikiText-103 validation; 95% CI of the gain ratio by input bootstrap; step 143,000 reproduces the census file and the 30-input survey"]]
    i = next(j for j, l in enumerate(out) if l.startswith("| Pretrained census"))
    for r in reversed(new):
        r = (r + ["–"] * ncol)[:ncol]
        out.insert(i + 1, "| " + " | ".join(r) + " |")
    t3 = "\n".join(out)
    for x, y in (("(SN 1–2; ED Fig. 3)", "(ED Fig. 3)"), ("(SN 6; not part of the evidential argument)", "(SN 2; not part of the evidential argument)"),
                 ("cell means transcribed in SN 6", "cell means transcribed in SN 2"), ("Window additivity (SN 16)", "Window additivity (Methods)"),
                 ("(ED Fig. 8b; SN 13)", "(ED Fig. 8b; SN 4)"), ("(ED Fig. 8a; SN 4)", "(ED Fig. 8a; SN 1)"),
                 ("checkpoints (SN 4, 6, 13, 16; ED Fig. 8)", "checkpoints (SN 1, 2 and 4; ED Fig. 8)"), ("Supplementary Note 17", "Supplementary Note 5"),
                 ("| Emergence during training (ED Fig. 4) |", "| Emergence during training (ED Fig. 4a,b) |"),
                 ("profile thirds 1–7, 8–15, 16–22", "profile thirds 1–7, 8–16, 17–23"),
                 ("| 26 checkpoints, d1_v21 (Drive) |", "| 20 checkpoints (the six switching checkpoints are in the next row), d1_v21 (Drive) |"),
                 ("| token-local, one position | – | float32 | 12 runs, leaking prefix mask |", "| token-local, one position, random token ids, no attention mask at profile time | – | float32 | 12 runs, leaking prefix mask |"),
                 ("| branch rotation of blocks 4–7 | float32 | 60 runs (March 2026; checkpoints and per-model files not retained) |", "| branch rotation, together, of the middle blocks identified from each model's σ~1~ profile | float32 | 60 runs (May 2026; checkpoints and per-model files not retained) |"),
                 ("k-gram models (March 2026) | d3f_windowed_scrambling.py", "k-gram models (May 2026) | d3f_windowed_scrambling.py"),
                 ("| branch rotation of block 5, then 500 fine-tuning steps of the downstream blocks | float32 | k = 8 models (March 2026) |", "| rotation of the residual stream at the output of block 5, then 500 fine-tuning steps of the downstream blocks | float32 | k = 8 models (May 2026) |"),
                 ("| top five right singular vectors of the block-5 token-local Jacobian by randomised SVD | token-local |", "| top five right singular vectors of the block-5 Jacobian over the whole input, by randomised SVD | whole-sequence, causal mask |"),
                 ("nine k-gram models (March 2026) | exp4a_singular_direction_perturbation.py", "nine k-gram models (May 2026) | exp4a_singular_direction_perturbation.py"),
                 ("k-gram models (March 2026) | d3r_residual_gain_stress.py", "k-gram models (May 2026) | d3r_residual_gain_stress.py"),
                 ("| Dyck-2 probes (ED Fig. 8a; SN 1) | – | – | per-block branch rotation; linear probes; participation ratio | float32 | four Dyck-2 models (March 2026) |", "| Bracket-nesting probes (ED Fig. 8a; SN 1) | – | – | rotation of the residual stream at the input of one block at a time; linear probes; participation ratio | float32 | four bracket-nesting models (May 2026) |"),
                 ("The March 2026 routing, skip-coefficient and k-gram profiles, taken by forward-mode iteration on J over the whole layer input at random token ids, are superseded", "The routing and skip-coefficient profiles of March 2026 and the k-gram profiles of May 2026, taken by forward-mode iteration on J over the whole layer input at random token ids, are superseded"),
                 ("the March mechanism-experiment checkpoints (SN 1, 2 and 4; ED Fig. 8) were not retained", "the May 2026 mechanism-experiment checkpoints (SN 1 to 4; ED Fig. 8) were not retained"),
                 ("three 1.3B seeds, 354M controls, Pythia-410M public checkpoints |", "three 1.3B seeds, 354M controls |"),
                 ("; s2_pythia_developmental.py (checkpoints; canonical estimator)", "")):
        assert t3.count(x) == 1, x
        t3 = t3.replace(x, y)
    assert not re.search(r"SN ([6-9]|1[0-9])\b", t3), re.findall(r"SN \d+", t3)
    open(os.path.join(OUT, "ed_table3_nmi.md"), "w").write(t3)
    t1 = open(os.path.join(OUT, "ed_table1.md")).read()
    x = "are in ed_table1.csv and Supplementary Note 10."
    assert t1.count(x) == 1
    open(os.path.join(OUT, "ed_table1_nmi.md"), "w").write(t1.replace(x, "are in ed_table1.csv; sweeping the ratio threshold from 0.60 to 0.95 gives 6 to 22 interior-valley models (14 at 0.80)."))
    return dict(dropped=dropped, n_cols=ncol)


def main():
    rows = ma.load(); dep = ma.depth_adjusted(rows)
    chk = dict(t4=ed_table4(rows, dep), t5=ed_table5(), t3=ed_table3_nmi(), dep_summary={k: dep[k] for k in ("partial_rel", "partial_abs", "diff_rel", "diff_abs")})
    json.dump(chk, open(os.path.join(OUT, "ed_tables45_check.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in chk.items() if k != "t5"}, indent=1, default=float)[:3000])
    for k, v in chk["t5"]["values"].items():
        print(k, "R", np.round(np.mean(v["R"]), 3), "rot", np.round(np.min(v["rot"]), 3), np.round(np.max(v["rot"]), 3))


if __name__ == "__main__":
    main()
