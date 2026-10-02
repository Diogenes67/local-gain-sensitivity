"""Predictive value of local gain against alternative block-level predictors on identical held-out folds (24 Sept 2026; review 5, item 1).

Panel: the 50 controlled models x blocks 1-10 of regression_check.py, with four block-level predictors per model:
  block   linear depth (the depth baseline of the paper)            C(block)  one coefficient per block ("flexible depth")
  lh      log activation norm ||h_l|| at the block input             lgr       log gain along a random unit direction at eps = 0.01
  ls1     log sigma_1 (local gain)                                   (lgr equals the Frobenius gain up to a constant that centring removes)
Outcomes: normalised sensitivity log(KL / ||delta||^2) and relative-dose divergence log KL, each for the all-positions readout
(kl_seq, the paper's Fig. 2 convention) and the same-position readout (kl_t, the Fig. 3a convention).

For every predictor set, coefficients (with a per-model intercept) are fitted on the training models of a fold and the
within-model-centred profile of each held-out model is predicted from the non-model terms; R^2 = 1 - SS(resid)/SS(total)
pooled over held-out models. Folds are identical across predictor sets: leave-one-model-out (50), leave-one-condition-out (10)
and leave-one-set-out (text conditions <-> k-gram). Paired uncertainty: the pooled R^2 difference between two predictor sets on
the same folds, bootstrapped over held-out models (2,000 resamples).

The same comparison is run on the six pretrained decoders (matched_pretrained/results, 20 inputs x 6 positions, blocks 1 to L-2)
with relative depth, its square (flexible depth), lh, lgr and ls1; folds: leave-one-model-out and leave-one-family-out.

Run: NCS_R=<folder holding matched/, d1/, reprofile/> MP_RESULTS=<matched_pretrained/results> python predictor_check.py
26 Sept 2026: joint baseline (depth + norm + random-direction gain) with and without sigma1, paired against each other.
  -> predictor_check.json
"""
import json, os, sys, glob
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matched_analysis as ma
if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
MP = os.environ.get("MP_RESULTS", os.path.join(HERE, "matched_pretrained", "results"))
E = 0.1; BL = np.arange(1, 11)
GROUPS = ma.D1_ORDER + [f"k{k}" for k in ma.KS]
SETS = {"training_conditions": ma.D1_ORDER, "kgram": [f"k{k}" for k in ma.KS]}
N_BOOT = 2000

SPECS = {                                   # name -> formula terms (a per-model intercept C(model) is always added for fitting)
    "depth": "block", "depth_flex": "C(block)", "norm": "lh", "randgain": "lgr", "gain": "ls1",
    "depth+norm": "block + lh", "depth+randgain": "block + lgr", "depth+gain": "block + ls1",
    "flex+norm": "C(block) + lh", "flex+randgain": "C(block) + lgr", "flex+gain": "C(block) + ls1",
    "depth+norm+gain": "block + lh + ls1", "depth+randgain+gain": "block + lgr + ls1", "depth+norm+randgain": "block + lh + lgr",
    "all": "C(block) + lh + lgr + ls1", "all_but_gain": "C(block) + lh + lgr",
    "joint": "block + lh + lgr", "joint+gain": "block + lh + lgr + ls1",     # joint baseline (linear depth) and the same with sigma1
}
SPECS_PRE = {
    "depth": "depth", "depth_flex": "depth + depth2", "norm": "lh", "randgain": "lgr", "gain": "ls1",
    "depth+norm": "depth + lh", "depth+randgain": "depth + lgr", "depth+gain": "depth + ls1",
    "flex+gain": "depth + depth2 + ls1", "depth+norm+gain": "depth + lh + ls1", "depth+randgain+gain": "depth + lgr + ls1",
    "all": "depth + depth2 + lh + lgr + ls1", "all_but_gain": "depth + depth2 + lh + lgr",
    "joint": "depth + lh + lgr", "joint+gain": "depth + lh + lgr + ls1",
}
# paired-bootstrap comparisons: (reference set, compared set or None for every other set, key)
COMPARISONS = [("depth+gain", None, "delta_r2_depth_gain_minus_this"), ("joint+gain", "joint", "delta_r2_joint_gain_minus_joint"),
               ("all", "all_but_gain", "delta_r2_all_minus_all_but_gain")]


def panel_controlled(rows):
    frames = []
    for i, r in enumerate(rows):
        # per-probe same-position normalised sensitivity, averaged per block as S_abs is in matched_analysis.load
        L = len(r["sigma1"]); St = np.zeros(L); n = np.zeros(L)
        for q in r["raw"]:
            b = q["block"]; St[b] += q["kl_t"]["v1"][ma.IE] / (E * q["h_norm"]) ** 2; n[b] += 1
        St = St / n
        frames.append(pd.DataFrame(dict(model=i, group=r["group"], block=BL,
                                        ls1=np.log(r["sigma1"][1:11]), lh=np.log(r["h_norm"][1:11]), lgr=np.log(r["gain_rand"][1:11, 0]),
                                        y_abs=np.log(r["S_abs"][1:11]), y_rel=np.log(r["kl_seq_v1"][1:11, ma.IE]),
                                        y_abs_t=np.log(St[1:11]), y_rel_t=np.log(r["kl_t_v1"][1:11, ma.IE]))))
    return pd.concat(frames, ignore_index=True)


def panel_pretrained():
    frames = []
    for f in sorted(glob.glob(os.path.join(MP, "mp_*.json"))):
        d = json.load(open(f)); ie = d["eps"].index(E); L = d["n_layers"]
        B = [x for x in d["per_block"] if 1 <= x["block"] <= L - 2]
        blk = np.array([x["block"] for x in B]); dep = blk / (L - 1)
        St = {}; n = {}
        for q in d["raw"]:
            b = q["block"]; St[b] = St.get(b, 0) + q["kl_t"]["v1"][ie] / (E * q["h_norm"]) ** 2; n[b] = n.get(b, 0) + 1
        frames.append(pd.DataFrame(dict(model=d["label"], group=d["family"], block=blk, depth=dep, depth2=dep ** 2,
                                        ls1=np.log([x["sigma1"] for x in B]), lh=np.log([x["h_norm"] for x in B]), lgr=np.log([x["gain_rand"][0] for x in B]),
                                        y_abs=np.log([x["S_abs"][ie] for x in B]), y_rel=np.log([x["kl_seq_v1"][ie] for x in B]),
                                        y_abs_t=np.log([St[x["block"]] / n[x["block"]] for x in B]), y_rel_t=np.log([x["kl_t_v1"][ie] for x in B]))))
    return pd.concat(frames, ignore_index=True)


def predict(df, y, terms, train_mask, test_mask):
    """Fit y ~ terms + C(model) on the training rows; return per held-out model (model, group, SSres, SStot) of the centred prediction."""
    tr, te = df[train_mask], df[test_mask]
    m = smf.ols(f"{y} ~ {terms} + C(model)", tr).fit()
    # design matrix of the non-model terms for the held-out rows (same column names as in the fitted model); the model
    # intercepts are not used, and the constant drops out of the within-model centring
    from patsy import dmatrix
    out = []
    for mid, t in te.groupby("model", sort=False):
        X = dmatrix(terms, t, return_type="dataframe")
        keep = [c for c in X.columns if c != "Intercept"]
        p = X[keep].values @ m.params[keep].values
        yc = t[y].values - t[y].values.mean(); pc = p - p.mean()
        out.append((mid, t.group.iloc[0], float(((yc - pc) ** 2).sum()), float((yc ** 2).sum())))
    return out


def r2(parts, keep=None):
    p = [x for x in parts if keep is None or x[1] in keep]
    return 1 - sum(x[2] for x in p) / sum(x[3] for x in p)


def folds_controlled(df):
    F = [("L1O", [(f"m{m}", df.model == m) for m in df.model.unique()]),
         ("LOCO", [(g, df.group == g) for g in GROUPS]),
         ("LOSO", [(s, df.group.isin(v)) for s, v in SETS.items()])]
    return F


def folds_pretrained(df):
    return [("L1O", [(m, df.model == m) for m in df.model.unique()]),
            ("LOFO", [(g, df.group == g) for g in df.group.unique()])]


def run(df, specs, folds, ys):
    res = {}
    for y in ys:
        res[y] = {}
        for scheme, fl in folds:
            parts = {}
            for name, terms in specs.items():
                parts[name] = sum((predict(df, y, terms, ~test, test) for _, test in fl), [])
            entry = {name: dict(r2=r2(p)) for name, p in parts.items()}
            if scheme in ("LOCO", "L1O") and "group" in df:
                for name, p in parts.items():
                    entry[name]["by_condition"] = {g: r2(p, [g]) for g in sorted(df.group.unique(), key=str)}
                    if scheme == "LOCO":
                        entry[name]["by_set"] = {s: r2(p, v) for s, v in SETS.items()}   # pooled over the held-out conditions of each set
            if scheme == "LOSO":
                for name, p in parts.items():
                    entry[name]["by_set"] = {s: r2(p, v) for s, v in SETS.items()}
            if scheme == "LOFO":
                for name, p in parts.items():
                    entry[name]["by_family"] = {g: r2(p, [g]) for g in df.group.unique()}
            # paired bootstrap over held-out models of the pooled R^2 difference: depth+gain against every other set,
            # joint+gain against the joint baseline (depth, norm, random-direction gain), and all against all_but_gain
            idx = {name: {x[0]: x for x in p} for name, p in parts.items()}
            for ref, other, key in COMPARISONS:
                if ref not in parts: continue
                models = [x[0] for x in parts[ref]]
                for name in specs:
                    if name == ref or (other is not None and name != other): continue
                    rng = np.random.default_rng(0); diffs = []      # the same 2,000 resamples for every comparison (and in make_fig2_scale.py)
                    for _ in range(N_BOOT):
                        draw = rng.choice(models, len(models), replace=True)
                        a = 1 - sum(idx[ref][m][2] for m in draw) / sum(idx[ref][m][3] for m in draw)
                        b = 1 - sum(idx[name][m][2] for m in draw) / sum(idx[name][m][3] for m in draw)
                        diffs.append(a - b)
                    d = np.array(diffs)
                    entry[name][key] = dict(point=entry[ref]["r2"] - entry[name]["r2"], ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                            frac_boot_positive=float((d > 0).mean()))
                    if other is not None and "by_set" in entry[name]:
                        # the same difference within each held-out set (LOCO, LOSO), with its own paired interval
                        entry[name][key + "_by_set"] = {}
                        for sname, members in SETS.items():
                            sub = [m for m in models if idx[ref][m][1] in members]
                            if not sub: continue
                            rng = np.random.default_rng(0); dd = []
                            for _ in range(N_BOOT):
                                draw = rng.choice(sub, len(sub), replace=True)
                                a = 1 - sum(idx[ref][m][2] for m in draw) / sum(idx[ref][m][3] for m in draw)
                                b = 1 - sum(idx[name][m][2] for m in draw) / sum(idx[name][m][3] for m in draw)
                                dd.append(a - b)
                            dd = np.array(dd)
                            entry[name][key + "_by_set"][sname] = dict(point=entry[ref]["by_set"][sname] - entry[name]["by_set"][sname],
                                                                        ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
            res[y][scheme] = entry
    return res


def main():
    rows = ma.load(); dfc = panel_controlled(rows)
    out = dict(n_controlled=len(rows), specs=SPECS, specs_pretrained=SPECS_PRE, eps=E, n_boot=N_BOOT,
               controlled=run(dfc, SPECS, folds_controlled(dfc), ["y_abs", "y_rel", "y_abs_t", "y_rel_t"]))
    dfp = panel_pretrained(); out["n_pretrained"] = int(dfp.model.nunique())
    out["pretrained"] = run(dfp, SPECS_PRE, folds_pretrained(dfp), ["y_abs", "y_rel", "y_abs_t", "y_rel_t"])
    # correlations among the predictors (within model, pooled), for the text
    def within(df, a, b):
        return float(np.median([np.corrcoef(t[a], t[b])[0, 1] for _, t in df.groupby("model")]))
    out["predictor_correlations_within_model_median"] = dict(
        controlled={f"{a}~{b}": within(dfc, a, b) for a, b in (("ls1", "lh"), ("ls1", "lgr"), ("lh", "lgr"), ("ls1", "block"), ("lh", "block"), ("lgr", "block"))},
        pretrained={f"{a}~{b}": within(dfp, a, b) for a, b in (("ls1", "lh"), ("ls1", "lgr"), ("lh", "lgr"), ("ls1", "depth"), ("lh", "depth"), ("lgr", "depth"))})
    json.dump(out, open(os.path.join(HERE, "predictor_check.json"), "w"), indent=1)
    for pop in ("controlled", "pretrained"):
        for y in ("y_abs", "y_abs_t", "y_rel"):
            print(f"\n== {pop} {y}")
            for scheme, entry in out[pop][y].items():
                line = "  ".join(f"{n} {v['r2']:+.2f}" for n, v in entry.items())
                print(f"  {scheme:5s} {line}")


if __name__ == "__main__":
    main()
