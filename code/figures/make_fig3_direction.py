"""Fig. 3 (NMI layout, 25 Sept 2026): where the consequence of the leading local direction arises.

a  controlled: same-position KL along v1 over a random direction against the squared gain ratio, blocks 1-10, 45 causal models (eps = 0.1)
b  pretrained decoders: the same, blocks 1..L-2 (eps = 0.1)
c  factors of the v1-to-random ratio per model with 95% intervals: squared local gain and the downstream factor g^T F g / ||J u||^2
   (downstream_factor.py); the after-block amplification and the output alignment in the mean-centred logit gauge
   (linresp_centred_stats.json, from the gauge-recorded re-run; raw-gauge values in Methods); and the measured excess over the
   squared gain ratio on the same probes at eps = 0.01 (per-probe ratios, geometric mean) and at eps = 0.1 under the panel a
   convention (ratio of block means, median over blocks)
d  downstream factor per probe (v1 over random), per model, with the fraction above one
e  within-model Spearman correlation across blocks between sigma1 and the downstream factor of v1 itself, g^T F g / sigma1^2

Run: NCS_R=<folder with matched/results> LINRESP=<matched/linresp> MP_RESULTS=<matched_pretrained/results> python make_fig3_direction.py
  -> fig/fig3_direction_nmi.{png,pdf}, fig3_direction_nmi_check.json
"""
import os, sys, json
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matched_analysis as ma
import downstream_factor as dfm
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, RAMP6
if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
OUT = os.path.join(HERE, "fig"); IE = 2
TEXT_COL = {"real": "#eb6834", "bidir": "#e87ba4", "shuffled": "#1baf7a", "randlab": "#4a3aa7"}
COL = dict(TEXT_COL, **{f"k{k}": c for k, c in zip(ma.KS, RAMP6)})
PT_COL = {"GPT-2 124M": "#8a5a00", "Pythia-410M": "#c7254e", "Pythia-1.4B": "#7a1734", "Llama-3.2-1B": "#2a78d6", "Qwen2.5-1.5B": "#1baf7a", "Gemma-2-2B": "#4a3aa7"}
PT_ORDER = ["GPT-2 124M", "Pythia-410M", "Pythia-1.4B", "Llama-3.2-1B", "Qwen2.5-1.5B", "Gemma-2-2B"]
C_LAB = {"d1_real": ("real", "real"), "d1_shuffled": ("shuffled", "shuffled"), "kgram_1": ("k = 1", "k1"), "kgram_8": ("k = 8", "k8")}
gmean = lambda v: float(np.exp(np.mean(np.log(np.clip(np.asarray(v, float), 1e-300, None)))))


def per_model_rows():
    """name -> dict(rows, color, label) for the eight controlled checkpoints then the six decoders (per-probe rows of downstream_factor)."""
    out = {}
    for name, (per, blocks, cond) in dfm.controlled_rows().items():
        base, seed = name.rsplit("_s", 1); lab, grp = C_LAB[base]
        out[f"{lab} s{seed}"] = dict(rows=per, blocks=blocks, color=COL[grp], key=name)
    pre = dfm.pretrained_rows()
    for lab in PT_ORDER:
        per, blocks, fam = pre[lab]
        out[lab] = dict(rows=per, blocks=blocks, color=PT_COL[lab], key=lab)
    return out


def block_profile(rows, blocks, key):
    return [np.mean([r[key] for r in rows if r["block"] == b]) for b in blocks]


def main():
    rows = ma.load(); causal = [r for r in rows if r["group"] != "bidir"]
    M = per_model_rows(); names = list(M); nC = 8
    DF = json.load(open(os.path.join(HERE, "downstream_factor.json")))
    S = {r["model"]: r for r in DF["controlled"]} | {r["model"]: r for r in DF["pretrained"]}
    LC = json.load(open(os.path.join(HERE, "linresp_centred_stats.json")))       # mean-centred after-block factor and alignment
    G = {r["model"]: r for r in LC["controlled"]} | {r["model"]: r for r in LC["pretrained"]}
    check = {}
    fig = plt.figure(figsize=(7.08, 7.4))
    gs = fig.add_gridspec(3, 6, left=0.075, right=0.985, top=0.965, bottom=0.085, wspace=1.6, hspace=0.72, height_ratios=[1, 1.05, 1])

    # a controlled scatter
    ax = fig.add_subplot(gs[0, 0:3])
    xs, ys = [], []
    for r in causal:
        x = (r["gain_v1"][1:11, IE] / r["gain_rand"][1:11, IE]) ** 2; y = r["kl_t_v1"][1:11, IE] / r["kl_t_rand"][1:11, IE]
        ax.scatter(x, y, s=3, color=COL[r["group"]], lw=0, alpha=0.8); xs += list(x); ys += list(y)
    xs, ys = np.array(xs), np.array(ys); lim = [0.8, max(ys.max(), xs.max()) * 1.3]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("(gain along v₁ / random)²"); ax.set_ylabel("same-position KL, v₁ / random"); tidy(ax, grid="both")
    exc = ys / xs
    ax.text(0.97, 0.03, f"controlled, 45 models\nmedian ratio {np.median(ys):.1f}\nexcess {np.median(exc):.1f}; {int((exc > 1).sum())}/{len(exc)} above", transform=ax.transAxes, va="bottom", ha="right", fontsize=5.3, color=INK2)
    check["a"] = dict(median_ratio=float(np.median(ys)), median_excess=float(np.median(exc)), n_above=int((exc > 1).sum()), n=len(exc))
    panel(ax, "a", dx=-0.13)

    # b pretrained scatter
    ax = fig.add_subplot(gs[0, 3:6])
    xs, ys, per = [], [], {}
    for lab in PT_ORDER:
        pb = pretrained_per_block(lab)
        x = np.array([(b["gain_v1"][IE] / b["gain_rand"][IE]) ** 2 for b in pb]); y = np.array([b["kl_t_v1"][IE] / b["kl_t_rand"][IE] for b in pb])
        ax.scatter(x, y, s=4, color=PT_COL[lab], lw=0, alpha=0.9, label=lab); xs += list(x); ys += list(y)
        per[lab] = dict(ratio=float(np.median(y)), gain_sq=float(np.median(x)), excess=float(np.median(y / x)), n_above=int((y / x > 1).sum()), n=len(x))
    xs, ys = np.array(xs), np.array(ys); lim = [0.8, max(ys.max(), xs.max()) * 1.3]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("(gain along v₁ / random)²"); ax.set_ylabel("same-position KL, v₁ / random"); tidy(ax, grid="both")
    exc = ys / xs
    ax.text(0.97, 0.03, f"pretrained, 6 decoders\nmedian ratio {np.median(ys):.1f}\nexcess {np.median(exc):.1f}; {int((exc > 1).sum())}/{len(exc)} above", transform=ax.transAxes, va="bottom", ha="right", fontsize=5.3, color=INK2)
    ax.legend(fontsize=4.6, loc="upper left", handletextpad=0.1, markerscale=1.3, borderaxespad=0.2)
    check["b"] = dict(median_ratio=float(np.median(ys)), median_excess=float(np.median(exc)), n_above=int((exc > 1).sum()), n=len(exc), per_model=per)
    panel(ax, "b", dx=-0.13)

    # c factors per model with 95% intervals
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(names))
    spec = [("local_sq", -0.30, "o", 1.0), ("after_centred", -0.15, "s", 0.45), ("align_centred", 0.0, "^", 0.45), ("combined", 0.15, "D", 1.0)]
    for key, dx, mk, alpha in spec:
        for i, n in enumerate(names):
            if key in ("local_sq", "combined"):
                s = S[M[n]["key"]]; v = s["summary"][key]["gm"]; lo, hi = s["ci95"][f"{key}.gm"]
            else:
                g = G[M[n]["key"]]; v = g["summary"][key]; lo, hi = g["ci95"][key]
            c = M[n]["color"]
            ax.plot([x[i] + dx] * 2, [lo, hi], color=c, lw=0.7, alpha=alpha, zorder=3)
            ax.scatter([x[i] + dx], [v], marker=mk, s=11 if mk != "D" else 13, color=c, alpha=alpha, lw=0.5 if mk == "D" else 0, edgecolor=INK if mk == "D" else "none", zorder=4)
    # the downstream factor from the gauge-recorded run must equal the one from the assay-based decomposition (same probes)
    for n in names:
        assert abs(G[M[n]["key"]]["summary"]["combined"] / S[M[n]["key"]]["summary"]["combined"]["gm"] - 1) < 0.02, n
    # measured excess on the same probes: eps 0.01 per-probe ratios (B), and the panel a/b convention at eps 0.1 (A)
    for i, n in enumerate(names):
        s = S[M[n]["key"]]
        v, (lo, hi) = s["summary"]["excess_001"]["B_gm"], s["ci95"]["excess_001.B_gm"]
        ax.plot([x[i] + 0.30] * 2, [lo, hi], color=INK, lw=0.7, zorder=3); ax.scatter([x[i] + 0.30], [v], marker="D", s=13, facecolor="white", edgecolor=INK, lw=0.6, zorder=4)
        v, (lo, hi) = s["summary"]["excess_01"]["A_median"], s["ci95"]["excess_01.A_median"]
        ax.plot([x[i] + 0.45] * 2, [lo, hi], color=INK2, lw=0.6, zorder=3); ax.scatter([x[i] + 0.45], [v], marker="x", s=12, color=INK2, lw=0.7, zorder=4)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--"); ax.axvline(nC - 0.5, color=GRID, lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=5)
    for t, n in zip(ax.get_xticklabels(), names):
        if n == "Pythia-410M": t.set_fontweight("bold")
    ax.set_ylabel("v₁ / random, factor"); ax.set_ylim(0.15, 300); ax.set_xlim(-0.6, len(names) - 0.3); tidy(ax)
    hl = [plt.Line2D([], [], marker="o", ls="", color=INK2, ms=3, label="local gain²"),
          plt.Line2D([], [], marker="s", ls="", color=INK2, alpha=0.45, ms=3, label="after block (mean-centred logits)"),
          plt.Line2D([], [], marker="^", ls="", color=INK2, alpha=0.45, ms=3, label="alignment (mean-centred logits)"),
          plt.Line2D([], [], marker="D", ls="", color=INK2, markeredgecolor=INK, ms=3, label="downstream factor (product)"),
          plt.Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK, ms=3, label="measured excess, ε = 0.01"),
          plt.Line2D([], [], marker="x", ls="", color=INK2, ms=3.5, label="measured excess, ε = 0.1, as in a, b")]
    ax.legend(handles=hl, fontsize=4.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, handletextpad=0.1, columnspacing=0.9, borderaxespad=0.0, frameon=False)
    ax.text(nC / 2 - 0.5, 150, "controlled", ha="center", fontsize=5.5, color=INK2); ax.text(nC + 3 - 0.5, 150, "pretrained", ha="center", fontsize=5.5, color=INK2)
    check["c"] = {n: {k: [round(S[M[n]["key"]]["summary"][k][a], 2), [round(v, 2) for v in S[M[n]["key"]]["ci95"][f"{k}.{a}"]]]
                      for k, a in (("local_sq", "gm"), ("after", "gm"), ("align", "gm"), ("combined", "gm"), ("excess_001", "B_gm"), ("excess_01", "A_median"))}
                  | {k: [round(G[M[n]["key"]]["summary"][k], 2), [round(v, 2) for v in G[M[n]["key"]]["ci95"][k]]] for k in ("after_centred", "align_centred")} for n in names}
    panel(ax, "c", dx=-0.04, dy=1.10)

    # d downstream factor per probe
    ax = fig.add_subplot(gs[2, 0:3])
    data = [np.log10([r["combined"] for r in M[n]["rows"]]) for n in names]
    parts = ax.violinplot(data, positions=x, widths=0.8, showextrema=False, showmedians=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(M[names[i]]["color"]); b.set_alpha(0.55); b.set_edgecolor("none")
    parts["cmedians"].set_color(INK); parts["cmedians"].set_linewidth(0.8)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--"); ax.axvline(nC - 0.5, color=GRID, lw=0.8)
    frac = {n: float(np.mean([r["combined"] > 1 for r in M[n]["rows"]])) for n in names}
    for i, n in enumerate(names):
        ax.text(i, 2.35, f"{100 * frac[n]:.0f}%", ha="center", fontsize=4.6, color=INK2)
    ax.set_ylim(-1.6, 2.6); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=4.8)
    ax.set_ylabel("log₁₀ downstream factor per probe\n(v₁ / random)"); tidy(ax)
    check["d"] = {n: round(frac[n], 2) for n in names}
    panel(ax, "d", dx=-0.13)

    # e rho(sigma1, downstream factor of v1) across blocks
    ax = fig.add_subplot(gs[2, 3:6])
    rho = {}
    for i, n in enumerate(names):
        rws, bl = M[n]["rows"], M[n]["blocks"]
        s1 = block_profile(rws, bl, "sigma1"); dv = block_profile(rws, bl, "down_v1"); av = block_profile(rws, bl, "after_v1")
        rho[n] = dict(down=float(stats.spearmanr(s1, dv)[0]), after=float(stats.spearmanr(s1, av)[0]))
        ax.scatter(i, rho[n]["down"], s=14, color=M[n]["color"], lw=0, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--"); ax.axvline(nC - 0.5, color=GRID, lw=0.8)
    ax.set_ylim(-1.05, 1.05); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=4.8)
    ax.set_ylabel("within-model ρ(σ₁, downstream\nfactor of v₁) across blocks"); tidy(ax)
    ax.text(nC / 2 - 0.5, 0.92, "controlled", ha="center", fontsize=5.5, color=INK2); ax.text(nC + 3 - 0.5, 0.92, "pretrained", ha="center", fontsize=5.5, color=INK2)
    check["e"] = {n: {k: round(v, 2) for k, v in rho[n].items()} for n in names}
    panel(ax, "e", dx=-0.13)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_direction_nmi.{ext}"), dpi=300)
    json.dump(check, open(os.path.join(HERE, "fig3_direction_nmi_check.json"), "w"), indent=1)
    print(json.dumps(check, indent=1))


_PB = {}


def pretrained_per_block(label):
    """per_block records (blocks 1..L-2) of one decoder from the assay results."""
    if not _PB:
        import glob
        for f in glob.glob(os.path.join(dfm.MP, "mp_*.json")):
            d = json.load(open(f)); L = d["n_layers"]
            _PB[d["label"]] = [b for b in d["per_block"] if 1 <= b["block"] <= L - 2]
    return _PB[label]


if __name__ == "__main__":
    main()
