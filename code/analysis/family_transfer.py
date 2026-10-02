"""Per-family held-out prediction in the six pretrained decoders, with uncertainty (24 Sept 2026; review 5, item 2).

For each family fold (GPT-2, Pythia [410M and 1.4B], Llama, Qwen, Gemma) the block-level model y ~ terms + C(model) is fitted on
the other families and the within-model-centred profile of each held-out model is predicted from the non-model terms.
Reported per fold: R^2 for depth alone, depth + activation norm, depth + random-direction gain and depth + gain, on the
all-positions normalised sensitivity (Fig. 2 convention) and on the same-position readout, with a 95% interval from
resampling the 20 inputs of the held-out model(s) (500 resamples; the training fit is held fixed, so the interval reflects
the sampling noise of the held-out profile). Also per model: leave-one-model-out R^2 with the same interval, and the
within-model Spearman correlation of sigma_1 with the two outcomes.

Run: MP_RESULTS=<matched_pretrained/results> python family_transfer.py -> family_transfer.json
26 Sept 2026: joint baseline (depth + norm + random-direction gain) with and without sigma1 added (own resampling stream).
"""
import json, os, glob
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.environ.get("MP_RESULTS", os.path.join(HERE, "matched_pretrained", "results"))
E = 0.1; N_BOOT = 500
SPECS = {"depth": "depth", "depth+norm": "depth + lh", "depth+randgain": "depth + lgr", "depth+gain": "depth + ls1", "all_but_gain": "depth + depth2 + lh + lgr",
         "joint": "depth + lh + lgr", "joint+gain": "depth + lh + lgr + ls1"}
# 26 Sept 2026: the joint baseline (depth + norm + random-direction gain) and the same with sigma1, added for the Patterns version.
# They draw their input resamples from a second generator so that the intervals of the original sets are unchanged.
NEW = {"joint", "joint+gain"}


def load():
    models = {}
    for f in sorted(glob.glob(os.path.join(MP, "mp_*.json"))):
        d = json.load(open(f)); ie = d["eps"].index(E); L = d["n_layers"]
        blocks = list(range(1, L - 1)); inputs = sorted({q["input"] for q in d["raw"]})
        # per-probe quantities keyed by (block, input): lists over positions
        per = {}
        for q in d["raw"]:
            if q["block"] in blocks:
                per.setdefault((q["block"], q["input"]), []).append(dict(
                    s1=q["sigma1"], h=q["h_norm"], gr=np.mean(q["gain"]["rand"][0]) if isinstance(q["gain"]["rand"][0], list) else q["gain"]["rand"][0],
                    S=q["kl_seq"]["v1"][ie] / (E * q["h_norm"]) ** 2, St=q["kl_t"]["v1"][ie] / (E * q["h_norm"]) ** 2))
        models[d["label"]] = dict(family=d["family"], L=L, blocks=blocks, inputs=inputs, per=per)
    return models


def profile(m, inputs):
    """Block-level panel of one model from a set of inputs (with repeats)."""
    rows = []
    for b in m["blocks"]:
        rs = [r for i in inputs for r in m["per"][(b, i)]]
        rows.append(dict(block=b, depth=b / (m["L"] - 1), depth2=(b / (m["L"] - 1)) ** 2, ls1=np.log(np.mean([r["s1"] for r in rs])),
                         lh=np.log(np.mean([r["h"] for r in rs])), lgr=np.log(np.mean([r["gr"] for r in rs])),
                         y_abs=np.log(np.mean([r["S"] for r in rs])), y_abs_t=np.log(np.mean([r["St"] for r in rs]))))
    return pd.DataFrame(rows)


def centred_parts(fit, terms, t, y):
    X = dmatrix(terms, t, return_type="dataframe"); keep = [c for c in X.columns if c != "Intercept"]
    p = X[keep].values @ fit.params[keep].values
    yc = t[y].values - t[y].values.mean(); pc = p - p.mean()
    return float(((yc - pc) ** 2).sum()), float((yc ** 2).sum())


def main():
    models = load(); labels = list(models); fams = sorted({m["family"] for m in models.values()})
    base = {lab: profile(m, m["inputs"]).assign(model=lab, family=m["family"]) for lab, m in models.items()}
    df_all = pd.concat(base.values(), ignore_index=True)
    rng = np.random.default_rng(0); rng2 = np.random.default_rng(1)
    out = dict(eps=E, n_boot=N_BOOT, models={lab: dict(family=m["family"], n_layers=m["L"], n_inputs=len(m["inputs"])) for lab, m in models.items()}, folds={}, l1o={})
    for y in ("y_abs", "y_abs_t"):
        out["folds"][y] = {}
        for fam in fams:
            held = [lab for lab in labels if models[lab]["family"] == fam]
            tr = df_all[~df_all.model.isin(held)]
            res = {}
            for name, terms in SPECS.items():
                fit = smf.ols(f"{y} ~ {terms} + C(model)", tr).fit()
                parts = [centred_parts(fit, terms, base[lab], y) for lab in held]
                point = 1 - sum(p[0] for p in parts) / sum(p[1] for p in parts)
                boots = []
                r = rng2 if name in NEW else rng
                for _ in range(N_BOOT):
                    bp = []
                    for lab in held:
                        m = models[lab]; draw = list(r.choice(m["inputs"], len(m["inputs"]), replace=True))
                        bp.append(centred_parts(fit, terms, profile(m, draw), y))
                    boots.append(1 - sum(p[0] for p in bp) / sum(p[1] for p in bp))
                res[name] = dict(r2=point, ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                 per_model={lab: 1 - p[0] / p[1] for lab, p in zip(held, parts)})
            res["improvement_gain_over_depth"] = res["depth+gain"]["r2"] - res["depth"]["r2"]
            out["folds"][y][fam] = res
        # leave-one-model-out with the same interval, and within-model Spearman
        out["l1o"][y] = {}
        for lab in labels:
            tr = df_all[df_all.model != lab]; res = {}
            for name, terms in SPECS.items():
                fit = smf.ols(f"{y} ~ {terms} + C(model)", tr).fit()
                p = centred_parts(fit, terms, base[lab], y)
                boots = []; r = rng2 if name in NEW else rng
                for _ in range(N_BOOT):
                    m = models[lab]; draw = list(r.choice(m["inputs"], len(m["inputs"]), replace=True))
                    bp = centred_parts(fit, terms, profile(m, draw), y); boots.append(1 - bp[0] / bp[1])
                res[name] = dict(r2=1 - p[0] / p[1], ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
            rho = stats.spearmanr(base[lab].ls1, base[lab][y])[0]
            rb = []
            for _ in range(N_BOOT):
                m = models[lab]; draw = list(rng.choice(m["inputs"], len(m["inputs"]), replace=True)); pr = profile(m, draw)
                rb.append(stats.spearmanr(pr.ls1, pr[y])[0])
            res["spearman_sigma1"] = dict(rho=float(rho), ci95=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))])
            out["l1o"][y][lab] = res
    # pooled R^2 over the family folds and over the model folds (1 - sum SSres / sum SStot over held-out models), with the same
    # input-resampling interval: one draw per held-out model per resample, shared by the predictor sets (fits held fixed)
    out["pooled"] = {}
    for y in ("y_abs", "y_abs_t"):
        out["pooled"][y] = {}
        for scheme in ("family", "model"):
            fits = {}   # (held-out model, spec) -> fitted model
            for lab in labels:
                held = [l for l in labels if models[l]["family"] == models[lab]["family"]] if scheme == "family" else [lab]
                tr = df_all[~df_all.model.isin(held)]
                for name, terms in SPECS.items():
                    fits[(lab, name)] = (fit := smf.ols(f"{y} ~ {terms} + C(model)", tr).fit(), terms)
            point = {}
            for name in SPECS:
                parts = [centred_parts(fits[(lab, name)][0], fits[(lab, name)][1], base[lab], y) for lab in labels]
                point[name] = 1 - sum(p[0] for p in parts) / sum(p[1] for p in parts)
            boots = {name: [] for name in SPECS}
            for _ in range(N_BOOT):
                prof = {lab: profile(models[lab], list(rng.choice(models[lab]["inputs"], len(models[lab]["inputs"]), replace=True))) for lab in labels}
                for name in SPECS:
                    bp = [centred_parts(fits[(lab, name)][0], fits[(lab, name)][1], prof[lab], y) for lab in labels]
                    boots[name].append(1 - sum(p[0] for p in bp) / sum(p[1] for p in bp))
            out["pooled"][y][scheme] = {name: dict(r2=point[name], ci95=[float(np.percentile(boots[name], 2.5)), float(np.percentile(boots[name], 97.5))]) for name in SPECS}
    json.dump(out, open(os.path.join(HERE, "family_transfer.json"), "w"), indent=1)
    for y in ("y_abs", "y_abs_t"):
        for scheme in ("family", "model"):
            print(f"== {y}: pooled, {scheme} held out: " + "  ".join(f"{n} {v['r2']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" for n, v in out["pooled"][y][scheme].items()))
    for y in ("y_abs", "y_abs_t"):
        print(f"\n== {y}: family folds (R2, 95% input-resampling interval)")
        for fam, res in out["folds"][y].items():
            print(f"  {fam:7s} " + "  ".join(f"{n} {v['r2']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" for n, v in res.items() if isinstance(v, dict)))
        print(f"== {y}: per model (L1O)")
        for lab, res in out["l1o"][y].items():
            print(f"  {lab:14s} depth {res['depth']['r2']:+.2f}  depth+gain {res['depth+gain']['r2']:+.2f} [{res['depth+gain']['ci95'][0]:+.2f},{res['depth+gain']['ci95'][1]:+.2f}]  rho(sigma1,y) {res['spearman_sigma1']['rho']:+.2f} [{res['spearman_sigma1']['ci95'][0]:+.2f},{res['spearman_sigma1']['ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
