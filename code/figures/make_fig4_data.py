"""Fig. 4 (NMI layout, 23 Sept 2026): dependency structure in the data sets middle-layer intervention sensitivity.

a  sigma1 per block (blocks 1-11), four training conditions, mean +/- s.d. of five seeds (d1 v2.1)
b  middle-block (4-7) branch-rotation cost (filled) and identity-skip cost (open) per seed, four training conditions
c  the same against k for the k-gram sources (reprofile_v2 k-gram, five seeds per k)
d  the same against order m for the entropy-matched Markov family (markov_v1, 10,000 steps)
e  middle-to-edge gain ratio against middle-block rotation cost, all 80 models
Run: NCS_R=<folder with d1/, reprofile/, markov/> python make_fig4_data.py -> fig/fig4_data_nmi.{png,pdf}, fig4_data_nmi_check.json
"""
import os, sys, json, glob
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, RAMP6
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
R = os.environ.get("NCS_R", "/mnt/user-data/outputs/ncs"); OUT = os.path.join(HERE, "fig")
COND = ["real", "shuffled", "randlab", "bidir"]
LAB = {"real": "real text", "shuffled": "shuffled text", "randlab": "random labels", "bidir": "bidirectional"}
CC = {"real": "#eb6834", "shuffled": "#1baf7a", "randlab": "#4a3aa7", "bidir": "#e87ba4"}
KS = [1, 2, 3, 4, 5, 8]; MS = [1, 2, 3, 4, 5, 6]
KC = dict(zip(KS, RAMP6))
MC = dict(zip(MS, ["#9fd9b9", "#72c49a", "#45ad7c", "#23925f", "#0d7646", "#005a30"]))


def load():
    d1 = {c: [json.load(open(f"{R}/d1/results_v21/{c}_s{s}.json")) for s in range(5)] for c in COND}
    kg = {k: [json.load(open(f"{R}/reprofile/results_kgram/kgram_{k}_s{s}.json")) for s in range(5)] for k in KS}
    mk = {m: [json.load(open(f"{R}/markov/results/markov_{m}_s{s}.json")) for s in range(5)] for m in MS}
    return d1, kg, mk


def main():
    d1, kg, mk = load(); check = {}
    fig = plt.figure(figsize=(7.08, 4.9))
    gs = fig.add_gridspec(2, 6, left=0.075, right=0.985, top=0.95, bottom=0.09, wspace=1.7, hspace=0.55)

    # a
    ax = fig.add_subplot(gs[0, 0:2]); B = np.arange(1, 12)
    ax.axvspan(3.5, 7.5, color=GRID, alpha=0.6, lw=0)
    for c in COND:
        P = np.array([r["sigma1_profile"][1:12] for r in d1[c]]); m, s = P.mean(0), P.std(0, ddof=1)
        ax.fill_between(B, m - s, m + s, color=CC[c], alpha=0.18, lw=0); ax.plot(B, m, color=CC[c], lw=1.0, label=LAB[c])
        Rv = [r["R_ex0"] for r in d1[c]]; check.setdefault("a_R", {})[c] = [round(float(np.mean(Rv)), 3), round(float(np.std(Rv, ddof=1)), 3)]
    ax.set_yscale("log"); ax.set_xlabel("block"); ax.set_ylabel("σ₁"); ax.set_xticks([1, 4, 7, 10]); tidy(ax)
    ax.legend(fontsize=5, loc="upper right", handlelength=1.2)
    from matplotlib.ticker import LogLocator, FormatStrFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5))); ax.yaxis.set_major_formatter(FormatStrFormatter("%g")); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.text(5.5, 0.02, "blocks 4–7", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5, color=INK2)
    panel(ax, "a")

    # b
    ax = fig.add_subplot(gs[0, 2:4]); rng = np.random.default_rng(1)
    for i, c in enumerate(COND):
        rot = [r["regions"]["waist"]["dL"] for r in d1[c]]; sk = [r["skip_regions"]["waist"]["dL"] for r in d1[c]]
        j = rng.uniform(-0.12, 0.12, 5)
        ax.scatter(i - 0.15 + j, rot, s=10, color=CC[c], lw=0); ax.scatter(i + 0.15 + j, sk, s=10, facecolor="white", edgecolor=CC[c], lw=0.7)
        check.setdefault("b", {})[c] = dict(rot=[round(min(rot), 3), round(max(rot), 3)], skip=[round(min(sk), 3), round(max(sk), 3)])
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[c].replace(" ", "\n") for c in COND], fontsize=5.5)
    ax.set_ylabel("middle-block cost (nats)"); tidy(ax)
    ax.scatter([], [], s=10, color=INK2, label="rotation"); ax.scatter([], [], s=10, facecolor="white", edgecolor=INK2, label="skip"); ax.legend(fontsize=5, loc="upper right")
    panel(ax, "b")

    def ladder(ax, groups, keyfun, colors, xlabel, xt):
        means = []
        for x, g in zip(xt, groups):
            rot = [keyfun(r)[0] for r in g]; sk = [keyfun(r)[1] for r in g]; j = rng.uniform(-0.12, 0.12, len(g))
            ax.scatter(x - 0.15 + j, rot, s=9, color=colors[x], lw=0); ax.scatter(x + 0.15 + j, sk, s=9, facecolor="white", edgecolor=colors[x], lw=0.7)
            means.append(np.mean(rot))
        ax.plot(xt, means, color=INK2, lw=0.7, zorder=1); ax.set_xlabel(xlabel); ax.set_xticks(xt); ax.set_ylabel("middle-block cost (nats)"); tidy(ax)
        return means

    # c k-gram
    ax = fig.add_subplot(gs[0, 4:6])
    kx = {k: i for i, k in enumerate(KS)}
    cols = {i: KC[k] for k, i in kx.items()}
    means = ladder(ax, [kg[k] for k in KS], lambda r: (r["regions_branch"]["waist"]["dL"], r["skip_regions"]["waist"]["dL"]), cols, "k (k-gram source)", list(range(6)))
    ax.set_xticklabels([str(k) for k in KS])
    rho = stats.spearmanr([k for k in KS for _ in range(5)], [r["regions_branch"]["waist"]["dL"] for k in KS for r in kg[k]])[0]
    ax.text(0.03, 0.97, "means ordered as k\npermutation p = 0.0014", transform=ax.transAxes, va="top", fontsize=5.3, color=INK2)
    check["c"] = dict(means=[round(m, 3) for m in means], rho_all=round(float(rho), 3))
    panel(ax, "c")

    # d Markov
    ax = fig.add_subplot(gs[1, 0:3])
    mx = {m: m for m in MS}
    means = ladder(ax, [mk[m] for m in MS], lambda r: (r["regions_branch"]["waist"]["dL"], r["skip_regions"]["waist"]["dL"]), MC, "order m (entropy fixed at 1.00 nats)", MS)
    orders = [m for m in MS for _ in range(5)]; cost = [r["regions_branch"]["waist"]["dL"] for m in MS for r in mk[m]]
    rho = stats.spearmanr(orders, cost)[0]
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f} (30 models)", transform=ax.transAxes, va="top", fontsize=5.3, color=INK2)
    check["d"] = dict(means=[round(m, 3) for m in means], rho=round(float(rho), 3))
    panel(ax, "d", dx=-0.1)

    # e all 80
    ax = fig.add_subplot(gs[1, 3:6]); pts = []
    for c in COND:
        for r in d1[c]:
            ax.scatter(max(r["regions"]["waist"]["dL"], 1e-3), r["R_ex0"], s=12, marker="o", color=CC[c], lw=0); pts.append((r["regions"]["waist"]["dL"], r["R_ex0"]))
    for k in KS:
        for r in kg[k]:
            ax.scatter(r["regions_branch"]["waist"]["dL"], r["stats_indist"]["R_ex0"], s=12, marker="s", color=KC[k], lw=0); pts.append((r["regions_branch"]["waist"]["dL"], r["stats_indist"]["R_ex0"]))
    for m in MS:
        for r in mk[m]:
            ax.scatter(r["regions_branch"]["waist"]["dL"], r["stats_indist"]["R_ex0"], s=12, marker="^", color=MC[m], lw=0); pts.append((r["regions_branch"]["waist"]["dL"], r["stats_indist"]["R_ex0"]))
    ax.axhline(0.80, color=MUTED, ls="--", lw=0.6); ax.set_xscale("log"); ax.set_xlabel("middle-block rotation cost (nats)"); ax.set_ylabel("middle-to-edge gain ratio"); tidy(ax, grid="both")
    hl = [plt.Line2D([], [], marker=mk_, ls="", color=INK2, ms=3.5, label=l) for mk_, l in (("o", "training conditions"), ("s", "k-gram"), ("^", "entropy-matched"))]
    ax.legend(handles=hl, fontsize=5, loc="lower right")
    check["e"] = dict(n=len(pts), n_below_080=int(sum(p[1] < 0.8 for p in pts)))
    panel(ax, "e", dx=-0.1)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig4_data_nmi.{ext}"), dpi=300)
    json.dump(check, open(os.path.join(HERE, "fig4_data_nmi_check.json"), "w"), indent=1)
    print(json.dumps(check))


if __name__ == "__main__":
    main()
