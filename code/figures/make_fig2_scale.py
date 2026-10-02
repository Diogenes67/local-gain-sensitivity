"""Fig. 2 (NMI layout, 25 Sept 2026): perturbation scale decides whether gain ranks consequence.

a  input norm per block relative to block 1, 50 controlled models
b  within-model Spearman(sigma1, consequence) over blocks 1-10, relative dose and per unit squared displacement, paired
c  per-unit consequence against sigma1 after removing model and block effects (Frisch-Waugh residuals of regression_check M0)
d  held-out R^2 of the within-model normalised-sensitivity profile from four predictor sets on identical folds: depth, depth +
   activation norm, depth + random-direction gain, depth + gain (predictor_check.py); error bar on the last, the paired bootstrap
   interval of its difference from depth + random-direction gain
e  the panel-b correlations in the 24-layer real and shuffled models
f  the panel-b correlations in the six pretrained decoders (e1_extra.py)
g  held-out prediction in the pretrained decoders per model, for the Pythia pair and pooled, with input-resampling intervals
   (family_transfer.py)

Run: NCS_R=<folder with matched/, d1/, reprofile/, scale/> python make_fig2_scale.py -> fig/fig2_scale_nmi.{png,pdf}
"""
import os, sys, json
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matched_analysis as ma
import predictor_check as pc
from regression_check import panel as reg_panel
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, RAMP6

if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
OUT = os.path.join(HERE, "fig")

GROUPS = ["real", "bidir", "shuffled", "randlab"] + [f"k{k}" for k in ma.KS]
TEXT_COL = {"real": "#eb6834", "bidir": "#e87ba4", "shuffled": "#1baf7a", "randlab": "#4a3aa7"}
COL = dict(TEXT_COL, **{f"k{k}": c for k, c in zip(ma.KS, RAMP6)})
LAB = {"real": "real text", "bidir": "bidirectional", "shuffled": "shuffled text", "randlab": "random labels", **{f"k{k}": f"k = {k}" for k in ma.KS}}
BL = np.arange(1, 11)
SETS = {"depth": "block", "depth+norm": "block + lh", "depth+randgain": "block + lgr", "depth+gain": "block + ls1"}
SET_LAB = {"depth": "depth", "depth+norm": "depth + norm", "depth+randgain": "depth + random-direction gain", "depth+gain": "depth + gain"}
SET_COL = {"depth": "#d3d2ce", "depth+norm": "#a6a5a0", "depth+randgain": "#66655f", "depth+gain": INK}
PT = [("GPT-2 124M", "#8a5a00"), ("Pythia-410M", "#c7254e"), ("Pythia-1.4B", "#7a1734"), ("Llama-3.2-1B", "#2a78d6"), ("Qwen2.5-1.5B", "#1baf7a"), ("Gemma-2-2B", "#4a3aa7")]
N_BOOT = 2000


def scale_rows():
    R = os.path.join(ma.R, "scale", "results"); L = 24; IE = 2; out = []
    for c in ("real", "shuffled"):
        for s in range(3):
            m = json.load(open(f"{R}/matched_d1_{c}_s{s}.json")); pb = m["per_block"]
            sig = np.array([b["sigma1"] for b in pb]); kl = np.array([b["kl_seq_v1"][IE] for b in pb])
            hn = np.zeros(L); S = np.zeros(L); n = np.zeros(L)
            for q in m["raw"]:
                b = q["block"]; hn[b] += q["h_norm"]; S[b] += q["kl_seq"]["v1"][IE] / (0.1 * q["h_norm"]) ** 2; n[b] += 1
            S = S / n
            out.append(dict(cond=c, seed=s, rho_rel=float(stats.spearmanr(sig[1:L - 1], kl[1:L - 1])[0]), rho_abs=float(stats.spearmanr(sig[1:L - 1], S[1:L - 1])[0])))
    return out


def heldout_controlled(df, y="y_abs"):
    """Pooled centred R^2 of the four predictor sets on identical folds, and the paired bootstrap (over held-out models) of the
    difference depth + gain minus depth + random-direction gain."""
    KG = [f"k{k}" for k in ma.KS]
    schemes = {"model\nheld out": [df.model == m for m in df.model.unique()],
               "text condition\nheld out": [df.group == g for g in ma.D1_ORDER],
               "k-gram order\nheld out": [df.group == g for g in KG],
               "text → k-gram": [df.group.isin(KG)],
               "k-gram → text": [df.group.isin(ma.D1_ORDER)]}
    out = {}
    for scheme, folds in schemes.items():
        parts = {name: sum((pc.predict(df, y, terms, ~t, t) for t in folds), []) for name, terms in SETS.items()}
        res = {name: pc.r2(p) for name, p in parts.items()}
        models = [x[0] for x in parts["depth+gain"]]; rng = np.random.default_rng(0)
        idx = {name: {x[0]: x for x in p} for name, p in parts.items()}
        diffs = []
        for _ in range(N_BOOT):
            draw = rng.choice(models, len(models), replace=True)
            a = 1 - sum(idx["depth+gain"][m][2] for m in draw) / sum(idx["depth+gain"][m][3] for m in draw)
            b = 1 - sum(idx["depth+randgain"][m][2] for m in draw) / sum(idx["depth+randgain"][m][3] for m in draw)
            diffs.append(a - b)
        res["delta_gain_minus_randgain"] = dict(point=res["depth+gain"] - res["depth+randgain"], ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
        res["n_folds"] = len(folds); res["n_heldout_models"] = len(models)
        out[scheme.replace("\n", " ")] = res
    return out


def bars_with_offscale(ax, x, vals, w, color, ymin, label=None, zorder=2):
    """Bars clipped at ymin; an off-scale value is printed below the axis."""
    v = np.array(vals, float); clipped = np.maximum(v, ymin)
    ax.bar(x, clipped, w, color=color, label=label, zorder=zorder, lw=0)
    for xx, val, c in zip(x, v, clipped):
        if val < ymin:
            ax.text(xx, ymin + 0.02, f"{val:.1f}".replace("-", "−"), rotation=90, ha="center", va="bottom", fontsize=4.2, color="white" if color == INK else INK, zorder=zorder + 1)


def main():
    rows = ma.load(); df = reg_panel(rows)
    dfc = pc.panel_controlled(rows)
    ho = heldout_controlled(dfc)
    ft = json.load(open(os.path.join(HERE, "family_transfer.json")))
    sc = scale_rows()
    check = {}

    fig = plt.figure(figsize=(7.08, 6.9))
    gs = fig.add_gridspec(3, 3, left=0.07, right=0.985, top=0.965, bottom=0.075, wspace=0.42, hspace=0.62, height_ratios=[1, 1, 1])

    # a: norm growth
    ax = fig.add_subplot(gs[0, 0])
    ratios = []
    for r in rows:
        h = r["h_norm"][1:11] / r["h_norm"][1]; ratios.append(r["h_norm"][10] / r["h_norm"][1])
        ax.plot(BL, h, color=COL[r["group"]], lw=0.6, alpha=0.85)
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel("‖h‖ relative to block 1")
    ax.set_xticks([1, 4, 7, 10]); tidy(ax)
    check["a_norm_ratio_range"] = [float(min(ratios)), float(max(ratios))]
    ax.text(0.03, 0.97, f"block 10 / block 1\n{min(ratios):.1f}–{max(ratios):.0f}-fold", transform=ax.transAxes, va="top", fontsize=6, color=INK2)
    panel(ax, "a")

    # b: paired correlations
    ax = fig.add_subplot(gs[0, 1])
    rel = np.array([r["spearman_blocks_sigma1_kl"] for r in rows]); ab = np.array([r["spearman_blocks_sigma1_S"] for r in rows])
    rng = np.random.default_rng(0); jit = rng.uniform(-0.08, 0.08, len(rows))
    for i, r in enumerate(rows):
        ax.plot([0 + jit[i], 1 + jit[i]], [rel[i], ab[i]], color=COL[r["group"]], lw=0.4, alpha=0.6)
        ax.scatter([0 + jit[i], 1 + jit[i]], [rel[i], ab[i]], s=5, color=COL[r["group"]], lw=0, zorder=3)
    for x, v in ((0, rel), (1, ab)):
        ax.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, color=INK, lw=1.2, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.5, ls="--")
    m = lambda v: f"{np.median(v):+.2f}".replace("-", "−")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"relative dose\nmedian {m(rel)}\n{(rel > 0).sum()}/50 > 0", f"per unit ‖δ‖²\nmedian {m(ab)}\n{(ab > 0).sum()}/50 > 0"], fontsize=5.5)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-1.05, 1.05); ax.set_ylabel("within-model ρ(σ₁, consequence)"); tidy(ax)
    check["b"] = dict(rel_median=float(np.median(rel)), rel_pos=int((rel > 0).sum()), abs_median=float(np.median(ab)), abs_pos=int((ab > 0).sum()))
    panel(ax, "b")

    # c: Frisch-Waugh residual plot for the per-unit slope
    ax = fig.add_subplot(gs[0, 2])
    ry = smf.ols("y_abs ~ block + C(model)", df).fit().resid; rx = smf.ols("ls1 ~ block + C(model)", df).fit().resid
    m0 = smf.ols("y_abs ~ ls1 + block + C(model)", df).fit(cov_type="cluster", cov_kwds={"groups": df["model"]})
    for g in GROUPS:
        sel = df.group == g
        ax.scatter(rx[sel], ry[sel], s=3, color=COL[g], lw=0, alpha=0.8)
    xs = np.linspace(rx.min(), rx.max(), 10); ax.plot(xs, m0.params["ls1"] * xs, color=INK, lw=1.0)
    ax.set_xlabel("log σ₁ (model and block removed)"); ax.set_ylabel("log consequence per unit ‖δ‖²\n(model and block removed)"); tidy(ax, grid="both")
    reg = json.load(open(os.path.join(HERE, "regression_check.json")))
    bci = reg["y_abs"]["M0"]["boot"]["ci"]; rel_s = reg["y_rel"]["M0"]["slope"]; rel_se = reg["y_rel"]["M0"]["se_cluster_model"]
    ax.text(0.03, 0.97, (f"slope {m0.params['ls1']:.2f} (95% CI {bci[0][0]:.2f}–{bci[1][0]:.2f})\nrelative dose: {rel_s:+.2f} (s.e. {rel_se:.2f})").replace("-", "−"), transform=ax.transAxes, va="top", fontsize=5.5, color=INK2)
    check["c"] = dict(slope=float(m0.params["ls1"]), ci=[bci[0][0], bci[1][0]], rel_slope=rel_s)
    panel(ax, "c")

    # d: held-out R2 from four predictor sets on identical folds (controlled)
    ax = fig.add_subplot(gs[1, 0:2])
    schemes = list(ho); x = np.arange(len(schemes)); w = 0.19; ymin = -0.5
    for j, name in enumerate(SETS):
        bars_with_offscale(ax, x + (j - 1.5) * w, [ho[s][name] for s in schemes], w, SET_COL[name], ymin, label=SET_LAB[name])
    for i, s in enumerate(schemes):
        d = ho[s]["delta_gain_minus_randgain"]; base = ho[s]["depth+randgain"]
        ax.plot([x[i] + 1.5 * w] * 2, [max(base + d["ci"][0], ymin), base + d["ci"][1]], color="#eb6834", lw=0.9, zorder=4)
        ax.plot([x[i] + 1.5 * w - 0.05, x[i] + 1.5 * w + 0.05], [base + d["ci"][1]] * 2, color="#eb6834", lw=0.7, zorder=4)
        ax.plot([x[i] + 1.5 * w - 0.05, x[i] + 1.5 * w + 0.05], [max(base + d["ci"][0], ymin)] * 2, color="#eb6834", lw=0.7, zorder=4)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" held out", "\nheld out") for s in schemes], fontsize=5.5); ax.set_ylabel("held-out R², per unit ‖δ‖²"); ax.set_ylim(ymin, 0.95)
    ax.set_xlim(-0.55, len(schemes) - 0.45)
    ax.legend(fontsize=5, loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.8, borderaxespad=0.3); tidy(ax)
    check["d"] = {s: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ho[s].items()} for s in schemes}
    panel(ax, "d", dx=-0.09)

    # e: 24-layer
    ax = fig.add_subplot(gs[1, 2])
    for r in sc:
        c = TEXT_COL[r["cond"]]
        ax.plot([0, 1], [r["rho_rel"], r["rho_abs"]], color=c, lw=0.6); ax.scatter([0, 1], [r["rho_rel"], r["rho_abs"]], s=8, color=c, lw=0, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.5, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["relative dose", "per unit ‖δ‖²"]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("within-model ρ, 24 layers"); tidy(ax)
    ax.text(1.12, 0.93, "real", color=TEXT_COL["real"], fontsize=5.5, va="center")
    ax.text(1.12, 0.80, "shuffled", color=TEXT_COL["shuffled"], fontsize=5.5, va="center")
    check["e"] = {f"{r['cond']}_s{r['seed']}": [round(r["rho_rel"], 2), round(r["rho_abs"], 2)] for r in sc}
    panel(ax, "e")

    # f: pretrained decoders (matched_pretrained.py; e1_extra.py input-resampled 95% CIs)
    ax = fig.add_subplot(gs[2, 0])
    ex = json.load(open(os.path.join(HERE, "matched_pretrained", "e1_extra.json")))
    for j, (lab, c) in enumerate(PT):
        v = ex[lab]; dx = (j - 2.5) * 0.05
        ax.plot([0 + dx, 1 + dx], [v["rho_rel"], v["rho_abs"]], color=c, lw=0.6)
        for xx, key in ((0 + dx, "rho_rel"), (1 + dx, "rho_abs")):
            lo, hi = v[key + "_ci"]; ax.plot([xx, xx], [lo, hi], color=c, lw=0.6); ax.scatter([xx], [v[key]], s=8, color=c, lw=0, zorder=3)
    rr = np.array([ex[l]["rho_rel"] for l, _ in PT]); aa = np.array([ex[l]["rho_abs"] for l, _ in PT])
    order = sorted(range(len(PT)), key=lambda j: -aa[j]); ypos = []
    for j in order:
        y = aa[j] if not ypos or ypos[-1] - aa[j] >= 0.075 else ypos[-1] - 0.075
        ypos.append(y); ax.text(1.18, y, PT[j][0], color=PT[j][1], fontsize=4.6, va="center")
    m6 = lambda v: f"{np.median(v):+.2f}".replace("-", "−")
    ax.axhline(0, color=MUTED, lw=0.5, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"relative dose\nmedian {m6(rr)}\n{(rr > 0).sum()}/6 > 0", f"per unit ‖δ‖²\nmedian {m6(aa)}\n{(aa > 0).sum()}/6 > 0"], fontsize=5.5)
    ax.set_xlim(-0.4, 1.9); ax.set_ylim(-1.05, 1.05); ax.set_ylabel("within-model ρ, pretrained"); tidy(ax)
    check["f"] = {l: [round(ex[l]["rho_rel"], 2), round(ex[l]["rho_abs"], 2)] for l, _ in PT}
    panel(ax, "f")

    # g: held-out prediction in the pretrained decoders (family_transfer.py; input-resampling 95% intervals, fits held fixed)
    ax = fig.add_subplot(gs[2, 1:3])
    l1o = ft["l1o"]["y_abs"]; fam = ft["folds"]["y_abs"]; pooled = ft["pooled"]["y_abs"]
    SHORT = {"GPT-2 124M": "GPT-2\n124M", "Pythia-410M": "Pythia\n410M", "Pythia-1.4B": "Pythia\n1.4B", "Llama-3.2-1B": "Llama\n3.2-1B", "Qwen2.5-1.5B": "Qwen2.5\n1.5B", "Gemma-2-2B": "Gemma-2\n2B"}
    groups = [(lab, l1o[lab], c) for lab, c in PT] + [("Pythia pair\nheld out", fam["Pythia"], INK2), ("pooled,\nfamily\nheld out", pooled["family"], INK2), ("pooled,\nmodel\nheld out", pooled["model"], INK2)]
    x = np.arange(len(groups)); w = 0.19; ymin = -1.6
    for j, name in enumerate(SETS):
        vals = [g[1][name]["r2"] for g in groups]
        bars_with_offscale(ax, x + (j - 1.5) * w, vals, w, SET_COL[name], ymin, label=SET_LAB[name])
        for i, g in enumerate(groups):
            lo, hi = g[1][name]["ci95"]; xx = x[i] + (j - 1.5) * w
            ax.plot([xx, xx], [max(lo, ymin), hi], color="#eb6834" if name == "depth+gain" else INK2, lw=0.7, zorder=4)
    ax.axhline(0, color=INK2, lw=0.5); ax.axvline(len(PT) - 0.5, color=GRID, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT.get(g[0], g[0]) for g in groups], fontsize=5.2)
    for t, g in zip(ax.get_xticklabels(), groups):
        t.set_color(g[2])
    ax.set_ylabel("held-out R², per unit ‖δ‖²"); ax.set_ylim(ymin, 1.0); ax.set_xlim(-0.55, len(groups) - 0.45); tidy(ax)
    check["g"] = {g[0].replace("\n", " ").replace("held out", "held out").strip(): {name: [round(g[1][name]["r2"], 3), [round(v, 3) for v in g[1][name]["ci95"]]] for name in SETS} for g in groups}
    panel(ax, "g", dx=-0.09)

    # shared colour key
    handles = [plt.Line2D([], [], color=COL[g], lw=1.5, label=LAB[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=5.3, frameon=False, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_scale_nmi.{ext}"), dpi=300)
    json.dump(check, open(os.path.join(HERE, "fig2_scale_nmi_check.json"), "w"), indent=1)
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
