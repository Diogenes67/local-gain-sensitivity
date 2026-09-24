"""Pooled gain-consequence regression: specification and uncertainty (22 Sept 2026; coauthor review 3.5).

Panel: the 50 controlled models x blocks 1-10 (500 rows) from the matched assay, as in matched_analysis.depth_adjusted:
  y_rel = log KL_seq along v1 at relative dose eps = 0.1 (the assay's own convention)
  y_abs = log S, S = KL_seq / (eps ||h||)^2 (per unit squared absolute displacement)
  x = log sigma1; block = 1..10; model = one of 50 (fixed effect); group = condition (real, bidir, shuffled, randlab, k1..k8)

Reported for each outcome:
  M0  y ~ log sigma1 + block + C(model)            plain OLS SE (what the draft reported), cluster-robust SE by model,
                                                  and a model-level bootstrap (models resampled with replacement, 2,000 draws)
  M1  y ~ log sigma1 + C(block) + C(model)         arbitrary shared depth effect per block
  M2  y ~ log sigma1 + C(model) + C(model):block   per-model intercept and per-model linear depth trend
  M3  M0 fitted within each condition (five models each): condition-specific slopes with cluster-robust SE and a model bootstrap
  W   per-model OLS slope of y on log sigma1 with block as covariate (10 rows each): distribution over the 50 models
  L1O leave-one-model-out prediction of y from M0 (R^2 of out-of-model prediction after removing the model intercept),
      descriptive only

Run: python regression_check.py -> regression_check.json
"""
import json, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from matched_analysis import load, IE, D1_ORDER, KS

HERE = os.path.dirname(os.path.abspath(__file__))
BL = np.arange(1, 11)


def panel(rows):
    frames = []
    for i, r in enumerate(rows):
        s1, kl, S = r["sigma1"][1:11], r["kl_seq_v1"][1:11, IE], r["S_abs"][1:11]
        frames.append(pd.DataFrame(dict(model=i, group=r["group"], ls1=np.log(s1), y_rel=np.log(kl), y_abs=np.log(S), lh=np.log(r["h_norm"][1:11]), block=BL)))
    return pd.concat(frames, ignore_index=True)


def fit(formula, df, cluster=True):
    if cluster:
        return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
    return smf.ols(formula, df).fit()


def boot_models(df, formula, term, n=2000, seed=0, groups=None):
    """Resample models with replacement; refit; percentile CI of the coefficient `term` (or of each term in `groups`)."""
    rng = np.random.default_rng(seed); models = df["model"].unique(); out = []
    for _ in range(n):
        draw = rng.choice(models, len(models), replace=True)
        # give each drawn copy a new model id so the fixed effects stay distinct
        parts = [df[df.model == m].assign(model=f"{m}_{j}") for j, m in enumerate(draw)]
        b = pd.concat(parts, ignore_index=True)
        try:
            m = smf.ols(formula, b).fit()
            out.append([m.params[t] for t in (groups or [term])])
        except Exception:
            continue
    a = np.array(out)
    return dict(n_ok=len(out), median=np.median(a, 0).tolist(), ci=[np.percentile(a, 2.5, 0).tolist(), np.percentile(a, 97.5, 0).tolist()])


def main():
    rows = load(); df = panel(rows); out = dict(n_models=len(rows), n_rows=len(df))
    for y in ("y_rel", "y_abs"):
        o = {}
        f0 = f"{y} ~ ls1 + block + C(model)"
        m_plain = fit(f0, df, cluster=False); m_cl = fit(f0, df)
        o["M0"] = dict(formula=f0, slope=float(m_cl.params["ls1"]), se_ols=float(m_plain.bse["ls1"]), se_cluster_model=float(m_cl.bse["ls1"]),
                       ci_cluster=[float(x) for x in m_cl.conf_int().loc["ls1"]], block=float(m_cl.params["block"]), block_se_cluster=float(m_cl.bse["block"]),
                       r2=float(m_cl.rsquared), boot=boot_models(df, f0, "ls1"))
        f1 = f"{y} ~ ls1 + C(block) + C(model)"; m1 = fit(f1, df)
        o["M1"] = dict(formula=f1, slope=float(m1.params["ls1"]), se_cluster_model=float(m1.bse["ls1"]), ci_cluster=[float(x) for x in m1.conf_int().loc["ls1"]])
        f2 = f"{y} ~ ls1 + C(model) + C(model):block"; m2 = fit(f2, df)   # per-model intercept and per-model linear depth trend
        o["M2"] = dict(formula=f2, slope=float(m2.params["ls1"]), se_cluster_model=float(m2.bse["ls1"]), ci_cluster=[float(x) for x in m2.conf_int().loc["ls1"]])
        # condition-specific slopes
        groups = D1_ORDER + [f"k{k}" for k in KS]
        fb = f"{y} ~ ls1 + block + C(model)"
        o["M3"] = dict(formula=fb + " fitted within each condition (five models; cluster-robust SE with five clusters is rough, the model bootstrap is over five models)", by_group={})
        for g in groups:
            sub = df[df.group == g]; mg = fit(fb, sub)
            bb = boot_models(sub, fb, "ls1", n=1000, seed=1)
            o["M3"]["by_group"][g] = dict(slope=float(mg.params["ls1"]), se_cluster_model=float(mg.bse["ls1"]), n_models=int(sub.model.nunique()), boot_ci=[bb["ci"][0][0], bb["ci"][1][0]])
        # within-model slopes
        ws = []
        for m in df.model.unique():
            sub = df[df.model == m]; mm = smf.ols(f"{y} ~ ls1 + block", sub).fit(); ws.append(float(mm.params["ls1"]))
        ws = np.array(ws)
        o["W"] = dict(median=float(np.median(ws)), q25=float(np.percentile(ws, 25)), q75=float(np.percentile(ws, 75)), n_positive=int((ws > 0).sum()),
                      n_above_1=int((ws > 1).sum()), min=float(ws.min()), max=float(ws.max()), by_group={g: [float(x) for x in ws[df.groupby('model').group.first().values == g]] for g in groups})
        # leave-one-model-out: predict the within-model centred outcome from ls1 and block coefficients fitted on the other 49 models
        num = den = 0.0
        for m in df.model.unique():
            tr, te = df[df.model != m], df[df.model == m]
            mm = smf.ols(f"{y} ~ ls1 + block + C(model)", tr).fit()
            pred = mm.params["ls1"] * te.ls1 + mm.params["block"] * te.block; yc = te[y] - te[y].mean(); pc = pred - pred.mean()
            num += float(((yc - pc) ** 2).sum()); den += float((yc ** 2).sum())
        o["L1O"] = dict(r2_within_model_centred=1 - num / den)
        # the same without gain (block only), to show what gain adds out of model
        num = 0.0
        for m in df.model.unique():
            tr, te = df[df.model != m], df[df.model == m]
            mm = smf.ols(f"{y} ~ block + C(model)", tr).fit()
            pred = mm.params["block"] * te.block; yc = te[y] - te[y].mean(); pc = pred - pred.mean()
            num += float(((yc - pc) ** 2).sum())
        o["L1O"]["r2_block_only"] = 1 - num / den
        out[y] = o
    json.dump(out, open(os.path.join(HERE, "regression_check.json"), "w"), indent=1)
    for y in ("y_rel", "y_abs"):
        o = out[y]; print(f"\n== {y}")
        print(f"M0 slope {o['M0']['slope']:+.2f}  OLS se {o['M0']['se_ols']:.2f}  cluster se {o['M0']['se_cluster_model']:.2f}  cluster CI {o['M0']['ci_cluster'][0]:+.2f}..{o['M0']['ci_cluster'][1]:+.2f}  boot CI {o['M0']['boot']['ci'][0][0]:+.2f}..{o['M0']['boot']['ci'][1][0]:+.2f}  block {o['M0']['block']:+.2f}")
        print(f"M1 (C(block)) slope {o['M1']['slope']:+.2f} se {o['M1']['se_cluster_model']:.2f}   M2 (+model trends) slope {o['M2']['slope']:+.2f} se {o['M2']['se_cluster_model']:.2f}")
        for g, v in o["M3"]["by_group"].items():
            print(f"   {g:9s} slope {v['slope']:+.2f} se {v['se_cluster_model']:.2f} boot {v['boot_ci'][0]:+.2f}..{v['boot_ci'][1]:+.2f}")
        print(f"W within-model slopes median {o['W']['median']:+.2f} IQR {o['W']['q25']:+.2f}..{o['W']['q75']:+.2f} positive {o['W']['n_positive']}/50 >1 {o['W']['n_above_1']}/50")
        print(f"L1O R2 within-model centred {o['L1O']['r2_within_model_centred']:.2f}  block only {o['L1O']['r2_block_only']:.2f}")


if __name__ == "__main__":
    main()
