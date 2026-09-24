"""Fig. 2 (NMI layout, 23 Sept 2026): perturbation scale decides whether gain ranks consequence.

a  input norm per block relative to block 1, 50 controlled models
b  within-model Spearman(sigma1, consequence) over blocks 1-10, relative dose and per unit squared displacement, paired
c  per-unit consequence against sigma1 after removing model and block effects (Frisch-Waugh residuals of regression_check M0)
d  out-of-model R^2 of the within-model profile: leave one model out, leave one condition out (text / k-gram folds)
e  the panel-b correlations in the 24-layer real and shuffled models
f  reserved for the pretrained decoders (experiment E1)

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
import loco_check as lc
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


def loco_split(df, y):
    res = {}
    for name, gs in (("text", ma.D1_ORDER), ("kgram", [f"k{k}" for k in ma.KS])):
        g, b = [], []
        for c in gs:
            t = df.group == c
            g += lc.predict(df, y, ~t, t)[0]; b += lc.predict(df, y, ~t, t, with_gain=False)[0]
        res[name] = (lc.r2(g), lc.r2(b))
    return res


def main():
    rows = ma.load(); df = reg_panel(rows)
    lo = json.load(open(os.path.join(HERE, "loco_check.json")))
    split = {y: loco_split(df, y) for y in ("y_abs", "y_rel")}
    sc = scale_rows()
    check = {}

    fig = plt.figure(figsize=(7.08, 4.6))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.95, bottom=0.10, wspace=0.42, hspace=0.55)

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
    m = lambda v: f"{np.median(v):+.2f}".replace("-", "\u2212")
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
    ax.text(0.03, 0.97, (f"slope {m0.params['ls1']:.2f} (95% CI {bci[0][0]:.2f}–{bci[1][0]:.2f})\nrelative dose: {rel_s:+.2f} (s.e. {rel_se:.2f})").replace("-", "\u2212"), transform=ax.transAxes, va="top", fontsize=5.5, color=INK2)
    check["c"] = dict(slope=float(m0.params["ls1"]), ci=[bci[0][0], bci[1][0]], rel_slope=rel_s)
    panel(ax, "c")

    # d: out-of-model R2
    ax = fig.add_subplot(gs[1, 0])
    cats = [("model\nheld out", lo["y_abs"]["L1O"]["r2_gain"], lo["y_abs"]["L1O"]["r2_block_only"], lo["y_rel"]["L1O"]["r2_gain"], lo["y_rel"]["L1O"]["r2_block_only"]),
            ("text condition\nheld out", *split["y_abs"]["text"], *split["y_rel"]["text"]),
            ("k-gram order\nheld out", *split["y_abs"]["kgram"], *split["y_rel"]["kgram"])]
    x = np.arange(len(cats)); w = 0.2
    ax.bar(x - 1.5 * w, [c[1] for c in cats], w, color=INK, label="gain + depth, per unit")
    ax.bar(x - 0.5 * w, [c[2] for c in cats], w, color=MUTED, label="depth only, per unit")
    ax.bar(x + 0.5 * w, [c[3] for c in cats], w, color=INK, alpha=0.35, hatch="////", edgecolor="white", lw=0, label="gain + depth, relative")
    ax.bar(x + 1.5 * w, [c[4] for c in cats], w, color=MUTED, alpha=0.35, label="depth only, relative")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cats], fontsize=5.5); ax.set_ylabel("out-of-model R²"); ax.set_ylim(-0.45, 0.9)
    ax.legend(fontsize=5, loc="upper right", ncol=1, handlelength=1.2); tidy(ax)
    check["d"] = {c[0].replace("\n", " "): [round(v, 2) for v in c[1:]] for c in cats}
    panel(ax, "d")

    # e: 24-layer
    ax = fig.add_subplot(gs[1, 1])
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
    ax = fig.add_subplot(gs[1, 2])
    ex = json.load(open(os.path.join(HERE, "matched_pretrained", "e1_extra.json")))
    PT = [("GPT-2 124M", "#8a5a00"), ("Pythia-410M", "#c7254e"), ("Pythia-1.4B", "#7a1734"), ("Llama-3.2-1B", "#2a78d6"), ("Qwen2.5-1.5B", "#1baf7a"), ("Gemma-2-2B", "#4a3aa7")]
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
    m6 = lambda v: f"{np.median(v):+.2f}".replace("-", "\u2212")
    ax.axhline(0, color=MUTED, lw=0.5, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"relative dose\nmedian {m6(rr)}\n{(rr > 0).sum()}/6 > 0", f"per unit ‖δ‖²\nmedian {m6(aa)}\n{(aa > 0).sum()}/6 > 0"], fontsize=5.5)
    ax.set_xlim(-0.4, 1.9); ax.set_ylim(-1.05, 1.05); ax.set_ylabel("within-model ρ, pretrained"); tidy(ax)
    check["f"] = {l: [round(ex[l]["rho_rel"], 2), round(ex[l]["rho_abs"], 2)] for l, _ in PT}
    panel(ax, "f")

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
