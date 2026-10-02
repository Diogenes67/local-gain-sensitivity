"""Same-position versions of the main gain-consequence statistics (24 Sept 2026; review 5, item 4).

The paper's Fig. 2 statistics use the mean divergence over the positions a perturbation reaches (kl_seq). This script
repeats the pooled log-log slope (model and block effects, model-clustered s.e.; and with a depth trend per model) and the
within-model Spearman correlations for the same-position divergence (kl_t), for the 50 controlled models and the six
pretrained decoders, at eps = 0.1. Held-out R^2 for both readouts is in predictor_check.json.

Run: NCS_R=<folder> MP_RESULTS=<matched_pretrained/results> python readout_check.py -> readout_check.json
"""
import os, sys, json
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matched_analysis as ma
if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
import predictor_check as pc


def main():
    rows = ma.load(); df = pc.panel_controlled(rows); out = {}
    for y in ("y_abs", "y_abs_t", "y_rel", "y_rel_t"):
        m = smf.ols(f"{y} ~ ls1 + block + C(model)", df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
        m2 = smf.ols(f"{y} ~ ls1 + C(model) + C(model):block", df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
        ws = np.array([stats.spearmanr(t.ls1, t[y])[0] for _, t in df.groupby("model")]); grp = df.groupby("model").group.first().values
        out[y] = dict(slope=float(m.params["ls1"]), se=float(m.bse["ls1"]), slope_trends=float(m2.params["ls1"]), se_trends=float(m2.bse["ls1"]),
                      rho_median=float(np.median(ws)), n_pos=int((ws > 0).sum()), rho_by_group={g: [float(x) for x in ws[grp == g]] for g in pc.GROUPS})
    dfp = pc.panel_pretrained(); outp = {}
    for y in ("y_abs", "y_abs_t", "y_rel", "y_rel_t"):
        m = smf.ols(f"{y} ~ ls1 + depth + C(model)", dfp).fit(cov_type="cluster", cov_kwds={"groups": dfp["model"]})
        ws = {lab: float(stats.spearmanr(t.ls1, t[y])[0]) for lab, t in dfp.groupby("model")}
        outp[y] = dict(slope=float(m.params["ls1"]), se=float(m.bse["ls1"]), rho=ws, rho_median=float(np.median(list(ws.values()))), n_pos=int(sum(v > 0 for v in ws.values())))
    json.dump(dict(readouts=dict(y_abs="log(kl_seq / |delta|^2)", y_abs_t="log(kl_t / |delta|^2)", y_rel="log kl_seq", y_rel_t="log kl_t"), controlled=out, pretrained=outp),
              open(os.path.join(HERE, "readout_check.json"), "w"), indent=1)
    for y, o in out.items():
        print(f"controlled {y:8s} slope {o['slope']:+.2f} ({o['se']:.2f}); trends {o['slope_trends']:+.2f} ({o['se_trends']:.2f}); rho median {o['rho_median']:+.2f}, {o['n_pos']}/50 positive")
    for y, o in outp.items():
        print(f"pretrained {y:8s} slope {o['slope']:+.2f} ({o['se']:.2f}); rho median {o['rho_median']:+.2f}, {o['n_pos']}/6 positive")


if __name__ == "__main__":
    main()
