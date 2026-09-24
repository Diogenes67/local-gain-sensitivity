"""Fig. 3 (NMI layout, 23 Sept 2026): where the consequence of the leading local direction arises.

a  controlled: same-position KL along v1 over a random direction against the squared gain ratio, blocks 1-10, 45 causal models (eps = 0.1)
b  pretrained decoders: the same, blocks 1..L-2 (eps = 0.1)
c  factors of the v1-to-random ratio per model: squared local gain, the map after the block (later blocks, final norm, head), alignment
   (geometric mean over probes within block, then over blocks); 8 controlled checkpoints and 6 pretrained decoders
d  after-map factor per probe (v1 over random), per model, with the fraction above one
e  within-model Spearman correlation between sigma1 and the after-map amplification of v1 across blocks

Controlled factors use the matched assay's random-direction gain at eps = 0.01 and the linearised response (linresp_afterblock.py
convention); pretrained factors come from matched_pretrained.py, which recorded both in one run.
Run: NCS_R=<folder with matched/results> python make_fig3_direction.py -> fig/fig3_direction_nmi.{png,pdf}, fig3_direction_nmi_check.json
"""
import os, sys, json, glob
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matched_analysis as ma
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, RAMP6
if os.environ.get("NCS_R"):
    ma.R = os.environ["NCS_R"]
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
OUT = os.path.join(HERE, "fig"); IE = 2
TEXT_COL = {"real": "#eb6834", "bidir": "#e87ba4", "shuffled": "#1baf7a", "randlab": "#4a3aa7"}
COL = dict(TEXT_COL, **{f"k{k}": c for k, c in zip(ma.KS, RAMP6)})
PT_COL = {"GPT-2 124M": "#8a5a00", "Pythia-410M": "#c7254e", "Pythia-1.4B": "#7a1734", "Llama-3.2-1B": "#2a78d6", "Qwen2.5-1.5B": "#1baf7a", "Gemma-2-2B": "#4a3aa7"}
PT_ORDER = ["GPT-2 124M", "Pythia-410M", "Pythia-1.4B", "Llama-3.2-1B", "Qwen2.5-1.5B", "Gemma-2-2B"]
gmean = lambda v: float(np.exp(np.mean(np.log(np.clip(np.asarray(v, float), 1e-300, None)))))


def controlled_factors():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "matched", "linresp", "linresp_*.json"))):
        d = json.load(open(f)); name = os.path.basename(f)[8:-5]
        m = json.load(open(os.path.join(ma.R, "matched", "results", f"matched_{name}.json")))
        mix = {(q["block"], q["input"], q["position"]): q for q in m["raw"]}
        probes = {}
        for r in d["raw"]:
            probes.setdefault((r["block"], r["input"], r["position"]), {}).setdefault("v1" if r["direction"] == "v1" else "rand", []).append(r)
        rows = []
        for key, pr in probes.items():
            if not 1 <= key[0] <= 10:
                continue
            q = mix[key]; v = pr["v1"][0]; rs = pr["rand"]; s1 = v["sigma1"]; g_r = q["gain"]["rand"][0]
            amp_r = np.mean([x["amplification"] for x in rs]); al_r = np.mean([x["alignment"] for x in rs])
            rows.append(dict(block=key[0], sigma1=s1, local=(s1 / g_r) ** 2, after=(v["amplification"] / s1 ** 2) / (amp_r / g_r ** 2),
                             after_v1=v["amplification"] / s1 ** 2, align=v["alignment"] / al_r))
        lab = {"d1_real": "real", "d1_shuffled": "shuffled", "kgram_1": "k = 1", "kgram_8": "k = 8"}[name.rsplit("_s", 1)[0]] + " s" + name.rsplit("_s", 1)[1]
        grp = {"d1_real": "real", "d1_shuffled": "shuffled", "kgram_1": "k1", "kgram_8": "k8"}[name.rsplit("_s", 1)[0]]
        out[lab] = dict(rows=rows, color=COL[grp])
    return out


def pretrained_factors():
    out = {}
    for f in glob.glob(os.path.join(HERE, "matched_pretrained", "results", "mp_*.json")):
        d = json.load(open(f)); L = d["n_layers"]; rows = []
        for x in d["lin_raw"]:
            if not 1 <= x["block"] <= L - 2:
                continue
            s1 = x["sigma1"]; g_r = np.mean(x["gain_rand_each"][0]); amp_r = np.mean(x["amp_rand"]); al_r = np.mean(np.array(x["quad_rand"]) / np.array(x["amp_rand"]))
            rows.append(dict(block=x["block"], sigma1=s1, local=(s1 / g_r) ** 2, after=(x["amp_v1"] / s1 ** 2) / (amp_r / g_r ** 2),
                             after_v1=x["amp_v1"] / s1 ** 2, align=(x["quad_v1"] / x["amp_v1"]) / al_r))
        pb = [b for b in d["per_block"] if 1 <= b["block"] <= L - 2]
        out[d["label"]] = dict(rows=rows, color=PT_COL[d["label"]], per_block=pb)
    return {k: out[k] for k in PT_ORDER}


def summarise(rows):
    blocks = sorted({r["block"] for r in rows})
    g = lambda k: gmean([gmean([r[k] for r in rows if r["block"] == b]) for b in blocks])
    s1 = [np.mean([r["sigma1"] for r in rows if r["block"] == b]) for b in blocks]
    av = [np.mean([r["after_v1"] for r in rows if r["block"] == b]) for b in blocks]
    return dict(local=g("local"), after=g("after"), align=g("align"), frac_after_gt1=float(np.mean([r["after"] > 1 for r in rows])),
                rho_s1_after=float(stats.spearmanr(s1, av)[0]))


def main():
    rows = ma.load(); causal = [r for r in rows if r["group"] != "bidir"]
    C = controlled_factors(); P = pretrained_factors()
    check = {}
    fig = plt.figure(figsize=(7.08, 5.4))
    gs = fig.add_gridspec(2, 6, left=0.075, right=0.985, top=0.95, bottom=0.12, wspace=1.6, hspace=0.75)

    # a controlled scatter
    ax = fig.add_subplot(gs[0, 0:2])
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
    panel(ax, "a")

    # b pretrained scatter
    ax = fig.add_subplot(gs[0, 2:4])
    xs, ys, per = [], [], {}
    for lab, v in P.items():
        pb = v["per_block"]
        x = np.array([(b["gain_v1"][IE] / b["gain_rand"][IE]) ** 2 for b in pb]); y = np.array([b["kl_t_v1"][IE] / b["kl_t_rand"][IE] for b in pb])
        ax.scatter(x, y, s=4, color=v["color"], lw=0, alpha=0.9, label=lab); xs += list(x); ys += list(y)
        per[lab] = dict(ratio=float(np.median(y)), gain_sq=float(np.median(x)), excess=float(np.median(y / x)), n_above=int((y / x > 1).sum()), n=len(x))
    xs, ys = np.array(xs), np.array(ys); lim = [0.8, max(ys.max(), xs.max()) * 1.3]
    ax.plot(lim, lim, color=MUTED, ls="--", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("(gain along v₁ / random)²"); ax.set_ylabel("same-position KL, v₁ / random"); tidy(ax, grid="both")
    exc = ys / xs
    ax.text(0.97, 0.03, f"pretrained, 6 decoders\nmedian ratio {np.median(ys):.1f}\nexcess {np.median(exc):.1f}; {int((exc > 1).sum())}/{len(exc)} above", transform=ax.transAxes, va="bottom", ha="right", fontsize=5.3, color=INK2)
    ax.legend(fontsize=4.6, loc="upper left", handletextpad=0.1, markerscale=1.3, borderaxespad=0.2)
    check["b"] = dict(median_ratio=float(np.median(ys)), median_excess=float(np.median(exc)), n_above=int((exc > 1).sum()), n=len(exc), per_model=per)
    panel(ax, "b")

    # c factors per model
    ax = fig.add_subplot(gs[0, 4:6])
    names = list(C) + list(P); S = {k: summarise((C.get(k) or P.get(k))["rows"]) for k in names}
    x = np.arange(len(names))
    for key, mk, lab in (("local", "o", "squared local gain"), ("after", "s", "after the block"), ("align", "^", "alignment")):
        ax.scatter(x, [S[n][key] for n in names], marker=mk, s=10, color=[(C.get(n) or P.get(n))["color"] for n in names], lw=0, label=lab, zorder=3)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--"); ax.axvline(len(C) - 0.5, color=GRID, lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=4.8); ax.set_ylabel("v₁ / random, factor"); tidy(ax)
    hl = [plt.Line2D([], [], marker=m, ls="", color=INK2, ms=3, label=l) for m, l in (("o", "local gain²"), ("s", "after block"), ("^", "alignment"))]
    ax.set_ylim(0.12, 300); ax.legend(handles=hl, fontsize=4.8, loc="upper left", ncol=3, handletextpad=0.1, columnspacing=0.6)
    check["c"] = {n: {k: round(S[n][k], 2) for k in ("local", "after", "align")} for n in names}
    panel(ax, "c")

    # d after-map distributions
    ax = fig.add_subplot(gs[1, 0:3])
    data = [np.log10([r["after"] for r in (C.get(n) or P.get(n))["rows"]]) for n in names]
    parts = ax.violinplot(data, positions=x, widths=0.8, showextrema=False, showmedians=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor((C.get(names[i]) or P.get(names[i]))["color"]); b.set_alpha(0.55); b.set_edgecolor("none")
    parts["cmedians"].set_color(INK); parts["cmedians"].set_linewidth(0.8)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--"); ax.axvline(len(C) - 0.5, color=GRID, lw=0.8)
    for i, n in enumerate(names):
        ax.text(i, 2.5, f"{100 * S[n]['frac_after_gt1']:.0f}%", ha="center", fontsize=4.6, color=INK2)
    ax.set_ylim(-1.6, 2.7); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=4.8)
    ax.set_ylabel("log₁₀ after-block factor (v₁ / random)"); tidy(ax)
    check["d"] = {n: round(S[n]["frac_after_gt1"], 2) for n in names}
    panel(ax, "d", dx=-0.1)

    # e rho(sigma1, after_v1)
    ax = fig.add_subplot(gs[1, 3:6])
    for i, n in enumerate(names):
        ax.scatter(i, S[n]["rho_s1_after"], s=14, color=(C.get(n) or P.get(n))["color"], lw=0, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--"); ax.axvline(len(C) - 0.5, color=GRID, lw=0.8)
    ax.set_ylim(-1.05, 1.05); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=4.8)
    ax.set_ylabel("within-model ρ(σ₁, after-block\namplification of v₁)"); tidy(ax)
    ax.text(len(C) / 2 - 0.5, 0.92, "controlled", ha="center", fontsize=5.5, color=INK2); ax.text(len(C) + len(P) / 2 - 0.5, 0.92, "pretrained", ha="center", fontsize=5.5, color=INK2)
    check["e"] = {n: round(S[n]["rho_s1_after"], 2) for n in names}
    panel(ax, "e", dx=-0.1)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_direction_nmi.{ext}"), dpi=300)
    json.dump(check, open(os.path.join(HERE, "fig3_direction_nmi_check.json"), "w"), indent=1)
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
