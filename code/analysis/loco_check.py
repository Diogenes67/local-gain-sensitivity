"""Leave-one-condition-out prediction of the within-model consequence profile (23 Sept 2026; NMI review 2, item 5).

Same panel and model as regression_check.py (M0: y ~ log sigma1 + block + C(model), 50 models x blocks 1-10).
For each held-out set, beta (log sigma1) and gamma (block) are fitted on the remaining models, and the within-model-centred
profile of every held-out model is predicted from beta*log sigma1 + gamma*block. R^2 = 1 - SS(resid)/SS(total), pooled over
the held-out models. The block-only model (y ~ block + C(model)) is the depth baseline.

Folds reported:
  L1O   leave one model out (50 folds; reproduces regression_check.json)
  LOCO  leave one condition out (10 folds: real, bidir, shuffled, randlab, k1, k2, k3, k4, k5, k8)
  LOSO  leave one set out (2 folds: train on the 20 training-condition models, predict the 30 k-gram models, and the reverse)
Per-condition R^2 is reported for every fold scheme so that a loss on any condition is visible, and R^2 pooled over the held-out\nmodels of each set (by_set; for LOSO this separates text -> k-gram from k-gram -> text).

Run: NCS_R=<folder holding matched/, d1/, reprofile/> python loco_check.py -> loco_check.json
"""
import json, os, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))   # current matched_analysis first; src only for make_figs
import matched_analysis as ma
if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
from regression_check import panel

GROUPS = ma.D1_ORDER + [f"k{k}" for k in ma.KS]
SETS = {"training_conditions": ma.D1_ORDER, "kgram": [f"k{k}" for k in ma.KS]}


def predict(df, y, train_mask, test_mask, with_gain=True):
    tr, te = df[train_mask], df[test_mask]
    f = f"{y} ~ ls1 + block + C(model)" if with_gain else f"{y} ~ block + C(model)"
    m = smf.ols(f, tr).fit()
    out = []
    for mid, t in te.groupby("model"):
        p = m.params["block"] * t.block + (m.params["ls1"] * t.ls1 if with_gain else 0)
        yc = t[y] - t[y].mean(); pc = p - p.mean()
        out.append((t.group.iloc[0], float(((yc - pc) ** 2).sum()), float((yc ** 2).sum())))
    return out, (float(m.params["ls1"]) if with_gain else None)


def r2(parts, group=None):
    p = [x for x in parts if group is None or x[0] == group]
    return 1 - sum(x[1] for x in p) / sum(x[2] for x in p)


def run(df, y, folds):
    gain, base, slopes = [], [], {}
    for name, test in folds:
        g, b_ = predict(df, y, ~test, test), predict(df, y, ~test, test, with_gain=False)
        gain += g[0]; base += b_[0]; slopes[name] = g[1]
    rs = lambda parts, groups: 1 - sum(x[1] for x in parts if x[0] in groups) / sum(x[2] for x in parts if x[0] in groups)
    return dict(r2_gain=r2(gain), r2_block_only=r2(base),
                by_condition={c: dict(r2_gain=r2(gain, c), r2_block_only=r2(base, c)) for c in GROUPS},
                by_set={s: dict(r2_gain=rs(gain, v), r2_block_only=rs(base, v)) for s, v in SETS.items()},
                train_slope=slopes)


def main():
    rows = ma.load(); df = panel(rows); out = dict(n_models=len(rows), n_rows=len(df))
    for y in ("y_abs", "y_rel"):
        o = {}
        o["L1O"] = run(df, y, [(str(m), df.model == m) for m in df.model.unique()])
        o["LOCO"] = run(df, y, [(g, df.group == g) for g in GROUPS])
        o["LOSO"] = run(df, y, [(s, df.group.isin(v)) for s, v in SETS.items()])
        out[y] = o
    json.dump(out, open(os.path.join(HERE, "loco_check.json"), "w"), indent=1)
    for y in ("y_abs", "y_rel"):
        print(f"\n== {y}  ({'per unit squared displacement' if y == 'y_abs' else 'relative dose eps = 0.1'})")
        for k in ("L1O", "LOCO", "LOSO"):
            o = out[y][k]; print(f"{k:5s} R2 gain+block {o['r2_gain']:.2f}   block only {o['r2_block_only']:.2f}")
        print("  LOCO by held-out condition (gain+block / block only / slope fitted without it)")
        for c in GROUPS:
            v = out[y]["LOCO"]["by_condition"][c]
            print(f"   {c:9s} {v['r2_gain']:+.2f} / {v['r2_block_only']:+.2f} / {out[y]['LOCO']['train_slope'][c]:+.2f}")
        print("  LOSO train slope", {k: round(v, 2) for k, v in out[y]["LOSO"]["train_slope"].items()})
        for k in ("L1O", "LOCO", "LOSO"):
            print(f"  {k} pooled by held-out set (gain+block / block only):",
                  {s: (round(v['r2_gain'], 2), round(v['r2_block_only'], 2)) for s, v in out[y][k]["by_set"].items()})


if __name__ == "__main__":
    main()
