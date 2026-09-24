"""Figures 3, 4 and Extended Data Fig. 5 for the NCS submission, drawn from the
result JSONs of the September 2026 runs (d1 v2.1, factorial v2, grid v2,
reprofile_v2 topology / alpha / kgram) and the March factorial file.

Run:  python make_figs.py            -> fig/fig3_new.png, fig/fig4_new.png, fig/edfig5_new.png
      python make_figs.py --check    -> prints the numbers drawn, for comparison with the text
"""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

R = "/mnt/user-data/outputs/ncs"
MARCH = "/mnt/user-data/uploads/Desktop/Interesting/PCI/results/b1_objective_factorial/factorial_partial_results.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.frameon": False,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "lines.linewidth": 0.9, "lines.markersize": 3.2,
    "axes.grid": False, "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e5e5e2"

# validated categorical order (validate_palette.js, adjacent pairs, light surface):
# blue, orange, aqua, violet, magenta, green
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#008300"]
# validated ordinal blue ramps (--ordinal): six steps for k, five for alpha
RAMP6 = ["#76acfc", "#5892e8", "#3c79d1", "#2460b7", "#06489c", "#003083"]
RAMP5 = ["#76acfc", "#518ce3", "#306dc4", "#0f4ea3", "#003083"]

D1_COND = ["real", "shuffled", "randlab", "bidir", "shuf2real", "real2shuf"]
D1_LAB = {"real": "real text", "shuffled": "shuffled text", "randlab": "random labels",
          "bidir": "bidirectional", "shuf2real": "shuffled → real", "real2shuf": "real → shuffled"}
D1_COL = dict(zip(D1_COND, CAT))
D1_MK = dict(zip(D1_COND, ["o", "s", "^", "D", "v", "P"]))

FAC_COND = ["AR", "CMLM", "MLM", "PLM"]
FAC_LAB = {"AR": "autoregressive", "CMLM": "causal masked-token", "MLM": "bidirectional masked-token",
           "PLM": "prefix, leakage-free"}
FAC_COL = dict(zip(FAC_COND, CAT[:4]))

TOPO = ["baseline", "gated_middle", "highway", "multi_highway", "unet", "no_residual"]
TOPO_LAB = {"baseline": "baseline residual", "gated_middle": "gated middle", "highway": "highway",
            "multi_highway": "multi-highway", "unet": "U-Net skips", "no_residual": "no residual"}
TOPO_COL = dict(zip(TOPO, CAT))

KS = [1, 2, 3, 4, 5, 8]
K_COL = dict(zip(KS, RAMP6))
ALPHAS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
A_COL = dict(zip(ALPHAS, RAMP5))


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def R_thirds(profile, exclude0=True):
    p = np.asarray(profile, dtype=float)
    p = p[1:] if exclude0 else p
    n = len(p); t = n // 3
    early, mid, late = p[:t], p[t:n - t], p[n - t:]
    edge = np.concatenate([early, late]).mean()
    return mid.mean() / edge, early.mean() / mid.mean(), late.mean() / mid.mean(), t


def panel(ax, letter, dx=-0.2, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def tidy(ax, grid="y"):
    ax.tick_params(length=2, pad=1.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def band(ax, x, ys, color, label=None, ls="-", lw=0.9, marker=None, ms=2.2, zorder=3, alpha_band=0.16):
    ys = np.asarray(ys, dtype=float)
    m, s = ys.mean(0), ys.std(0, ddof=1) if len(ys) > 1 else np.zeros(ys.shape[1])
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha_band, linewidth=0, zorder=zorder - 1)
    ax.plot(x, m, color=color, ls=ls, lw=lw, label=label, marker=marker, ms=ms, mew=0, zorder=zorder,
            solid_capstyle="round", solid_joinstyle="round")
    return m, s


def dots(ax, x, vals, color, filled=True, marker="o", jitter=0.0, ms=3.2, zorder=4, lw=0.7):
    xs = x + (np.linspace(-jitter, jitter, len(vals)) if jitter and len(vals) > 1 else 0)
    if filled:
        ax.plot(xs, vals, marker, color=color, ms=ms, mew=0.6, mec="white", zorder=zorder, ls="none")
    else:
        ax.plot(xs, vals, marker, mfc="white", mec=color, ms=ms, mew=lw, zorder=zorder, ls="none")
    return xs


def mean_bar(ax, x, vals, color=INK, half=0.22, lw=1.0, zorder=5):
    m = float(np.mean(vals))
    ax.plot([x - half, x + half], [m, m], color=color, lw=lw, zorder=zorder, solid_capstyle="butt")
    return m


# ------------------------------------------------------------------ data
def d1_data():
    out = {}
    for c in D1_COND:
        runs = load(f"{R}/d1/results_v21/{c}_s*.json")
        assert runs, c
        out[c] = dict(
            seeds=[r["seed"] for r in runs],
            prof=np.array([r["sigma1_profile"] for r in runs]),
            R=np.array([r["R_ex0"] for r in runs]),
            EM=np.array([r["early_over_mid"] for r in runs]),
            LM=np.array([r["late_over_mid"] for r in runs]),
            two_arm=[r["two_arm_rule"] for r in runs],
            branch=np.array([r["regions"]["waist"]["dL"] for r in runs]),
            skip=np.array([r["skip_regions"]["waist"]["dL"] for r in runs]),
            traj_R=np.array([[r["trajectory"][s]["R_ex0"] for s in ("1000", "3000", "5000")] + [r["R_ex0"]] for r in runs]),
            traj_dL=np.array([[r["trajectory"][s]["waist_dL_branch"] for s in ("1000", "3000", "5000")]
                              + [r["regions"]["waist"]["dL"]] for r in runs]),
        )
    return out


def fac_data():
    out = {}
    for c in FAC_COND:
        runs = load(f"{R}/factorial/results/{c}_s*.json")
        assert len(runs) == 5, c
        out[c] = dict(prof=np.array([r["final_profile"] for r in runs]),
                      R=np.array([r["R_ex0"] for r in runs]),
                      EM=np.array([r["early_over_mid"] for r in runs]),
                      LM=np.array([r["late_over_mid"] for r in runs]),
                      two_arm=[r["two_arm_rule"] for r in runs])
    return out


def march_data():
    d = json.load(open(MARCH))["conditions"]
    m = {"AR": "AR", "CMLM": "Causal_MLM", "MLM": "MLM", "PLM": "Prefix_LM"}
    return {c: np.array([r["R_ex0"] for r in d[m[c]]["results"]]) for c in FAC_COND}


def grid_data():
    runs = load(f"{R}/grid_v2/*.json")
    cells = {}
    for r in runs:
        Rr, EM, LM, t = R_thirds(r["final_profile"])
        key = (r["depth"], r["width"])
        cells.setdefault(key, []).append(dict(
            seed=r["seed"], lr=r["lr"], prof=np.array(r["final_profile"], float), R=Rr, EM=EM, LM=LM, t=t,
            loss=r["final_loss"], diverged=r["final_loss"] > 5.5,
            two_arm=bool(Rr < 0.80 and EM > 1 and LM > 1)))
    return cells


def kgram_data():
    out = {}
    for k in KS:
        runs = load(f"{R}/reprofile/results_kgram/kgram_{k}_s*.json")
        assert len(runs) == 5, k
        out[k] = dict(
            prof=np.array([r["profile_indist"] for r in runs]),
            R=np.array([r["stats_indist"]["R_ex0"] for r in runs]),
            EM=np.array([r["stats_indist"]["early_over_mid"] for r in runs]),
            LM=np.array([r["stats_indist"]["late_over_mid"] for r in runs]),
            two_arm=[r["stats_indist"]["two_arm_rule"] for r in runs],
            branch=np.array([r["regions_branch"]["waist"]["dL"] for r in runs]),
            skip=np.array([r["skip_regions"]["waist"]["dL"] for r in runs]),
            exit=np.array([r["early_exit_loss"] for r in runs]),
            base=np.array([r["baseline_eval_loss"] for r in runs]),
            pb=np.array([r["per_block_branch_dL"] for r in runs]),
        )
    return out


def topo_data():
    out = {}
    for c in TOPO:
        runs = load(f"{R}/reprofile/results_topology/topology_{c}_s*.json")
        assert len(runs) == 3, c
        out[c] = dict(prof=np.array([r["profile_indist"] for r in runs]),
                      R=np.array([r["stats_indist"]["R_ex0"] for r in runs]),
                      EM=np.array([r["stats_indist"]["early_over_mid"] for r in runs]),
                      LM=np.array([r["stats_indist"]["late_over_mid"] for r in runs]),
                      two_arm=[r["stats_indist"]["two_arm_rule"] for r in runs],
                      branch=np.array([r["regions_branch"]["waist"]["dL"] for r in runs]),
                      skip=np.array([r["skip_regions"]["waist"]["dL"] for r in runs]),
                      loss=np.array([r["baseline_eval_loss"] for r in runs]))
    return out


def alpha_data():
    out = {}
    for a in ALPHAS:
        runs = load(f"{R}/reprofile/results_alpha/alpha_{a}_s*.json")
        assert len(runs) == 3, a
        out[a] = dict(prof=np.array([r["profile_indist"] for r in runs]),
                      R=np.array([r["stats_indist"]["R_ex0"] for r in runs]),
                      two_arm=[r["stats_indist"]["two_arm_rule"] for r in runs],
                      loss=np.array([r["baseline_eval_loss"] for r in runs]))
    return out


# ------------------------------------------------------------------ Fig. 3
def fig3():
    d1, fac, cells = d1_data(), fac_data(), grid_data()
    fig = plt.figure(figsize=(7.2, 8.6))
    gs1 = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.975, bottom=0.745, wspace=0.42)
    gs2 = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.665, bottom=0.44, wspace=0.42, width_ratios=[1, 1, 1.05])
    gs3 = fig.add_gridspec(3, 4, left=0.065, right=0.99, top=0.335, bottom=0.045, wspace=0.14, hspace=0.55)
    blocks = np.arange(12)

    # a: per-block sigma1 (blocks 1-11, log), four pure conditions
    ax = fig.add_subplot(gs1[0, 0]); panel(ax, "a")
    b11 = np.arange(1, 12)
    handles = []
    for c in D1_COND[:4]:
        band(ax, b11, d1[c]["prof"][:, 1:], D1_COL[c], marker="o", ms=1.8)
        Rm, Rs = d1[c]["R"].mean(), d1[c]["R"].std(ddof=1)
        handles.append(Line2D([], [], color=D1_COL[c], lw=1.2, label=f"{D1_LAB[c]}, $R$ = {Rm:.2f} \u00b1 {Rs:.2f}"))
    ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(0.9, 11); ax.set_yticks([1, 2, 3, 5, 10]); ax.set_yticklabels(["1", "2", "3", "5", "10"])
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (token-local, causal mask), log")
    ax.set_xticks(range(1, 12, 2)); ax.set_xlim(0.7, 11.3)
    tidy(ax)
    ax.legend(handles=handles, loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)

    # b: waist dL per seed, branch (filled) and skip (open)
    ax = fig.add_subplot(gs1[0, 1]); panel(ax, "b")
    for i, c in enumerate(D1_COND[:4]):
        dots(ax, i - 0.17, d1[c]["branch"], D1_COL[c], filled=True, jitter=0.06)
        dots(ax, i + 0.17, d1[c]["skip"], D1_COL[c], filled=False, jitter=0.06)
        mb = mean_bar(ax, i - 0.17, d1[c]["branch"], color=D1_COL[c], half=0.13)
        ms_ = mean_bar(ax, i + 0.17, d1[c]["skip"], color=D1_COL[c], half=0.13)
        if c == "real":
            ax.annotate(f"{mb:.2f}", (i - 0.17, d1[c]["branch"].max()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=5.5, color=INK2)
            ax.annotate(f"{ms_:.2f}", (i + 0.17, ms_), xytext=(7, -2), textcoords="offset points", ha="left", fontsize=5.5, color=INK2)
        else:
            ax.annotate(f"{mb:.2f}", (i - 0.17, mb), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(["real", "shuffled", "random\nlabels", "bidir."])
    ax.set_ylabel("interior dependence, $\\Delta L$ (nats)"); ax.set_ylim(-0.03, 1.32); ax.set_xlim(-0.6, 3.6)
    tidy(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", ms=3.2, label="branch rotation, blocks 4–7"),
                       Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=3.2, label="identity skip, blocks 4–7")],
              loc="upper right", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)

    # c: R vs dL scatter, all six conditions
    ax = fig.add_subplot(gs1[0, 2]); panel(ax, "c")
    ax.axvline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(0.80, 1.28, "$R$ = 0.80", fontsize=5.5, color=INK2, ha="center", va="bottom")
    for c in D1_COND:
        ax.plot(d1[c]["R"], d1[c]["branch"], D1_MK[c], color=D1_COL[c], ms={"^": 3.8, "P": 4.4, "v": 3.8}.get(D1_MK[c], 3.4),
                mew=0.6, mec="white", ls="none", zorder=4)
    ax.annotate("real", (d1["real"]["R"].mean(), d1["real"]["branch"].mean()), xytext=(-4, 10), textcoords="offset points", fontsize=5.5, color=INK2, ha="right")
    ax.annotate("shuffled → real", (d1["shuf2real"]["R"].mean(), d1["shuf2real"]["branch"].mean()), xytext=(6, 4), textcoords="offset points", fontsize=5.5, color=INK2)
    ax.annotate("real → shuffled", (d1["real2shuf"]["R"].mean(), d1["real2shuf"]["branch"].mean()), xytext=(-6, 5), textcoords="offset points", fontsize=5.5, color=INK2, ha="right")
    ax.annotate("bidirectional", (d1["bidir"]["R"].mean(), d1["bidir"]["branch"].max()), xytext=(5, 6), textcoords="offset points", fontsize=5.5, color=INK2)
    ax.annotate("shuffled", (d1["shuffled"]["R"].mean(), d1["shuffled"]["branch"].mean()), xytext=(0, 6), textcoords="offset points", fontsize=5.5, color=INK2, ha="center")
    ax.annotate("random labels", (d1["randlab"]["R"].mean(), d1["randlab"]["branch"].min()), xytext=(0, -8), textcoords="offset points", fontsize=5.5, color=INK2, ha="center")
    ax.set_xlabel("$R_{\\mathrm{ex0}}$"); ax.set_ylabel("interior dependence, $\\Delta L$ (nats)")
    ax.set_xlim(0.42, 1.12); ax.set_ylim(-0.09, 1.32); tidy(ax, grid="both")

    # d: trajectories (two sub-axes)
    steps = np.array([1000, 3000, 5000, 10000])
    axd1 = fig.add_subplot(gs2[0, 0]); panel(axd1, "d")
    axd2 = fig.add_subplot(gs2[0, 1])
    for ax_, key, ylab in ((axd1, "traj_R", "$R_{\\mathrm{ex0}}$"), (axd2, "traj_dL", "interior dependence, $\\Delta L$ (nats)")):
        ax_.axvline(5000, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
        for c in D1_COND:
            ls = "-" if c in D1_COND[:4] else (0, (4, 1.5))
            band(ax_, steps, d1[c][key], D1_COL[c], ls=ls, marker="o", ms=1.8)
        ax_.set_xticks(steps); ax_.set_xticklabels(["1k", "3k", "5k", "10k"]); ax_.set_xlabel("training step")
        ax_.set_ylabel(ylab); ax_.set_xlim(500, 10500); tidy(ax_)
    axd1.text(5000, 1.19, "switch", fontsize=5.5, color=INK2, ha="center", va="bottom")
    axd1.set_ylim(0.45, 1.2); axd2.set_ylim(-0.03, 1.32)
    axd1.axhline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    axd1.text(600, 0.805, "0.80", fontsize=5.5, color=INK2, va="bottom")
    axd2.legend(handles=[Line2D([], [], color=D1_COL[c], lw=1.2, ls="-" if c in D1_COND[:4] else (0, (3, 1.2)), label=D1_LAB[c]) for c in D1_COND],
                loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)

    # e: factorial profiles (blocks 1-11, log) + R strip
    FAC_SHORT = {"AR": "AR", "CMLM": "causal MT", "MLM": "bidir. MT", "PLM": "prefix"}
    gse = gs2[0, 2].subgridspec(1, 2, width_ratios=[2.2, 1], wspace=0.5)
    ax = fig.add_subplot(gse[0, 0]); panel(ax, "e")
    for c in FAC_COND:
        band(ax, b11, fac[c]["prof"][:, 1:], FAC_COL[c], marker="o", ms=1.8)
    ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(0.9, 11); ax.set_yticks([1, 2, 3, 5, 10]); ax.set_yticklabels(["1", "2", "3", "5", "10"])
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (token-local, condition mask), log")
    ax.set_xticks(range(1, 12, 2)); ax.set_xlim(0.7, 11.3); tidy(ax)
    ax.legend(handles=[Line2D([], [], color=FAC_COL[c], lw=1.2, label=FAC_SHORT[c]) for c in FAC_COND],
              loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
    ax2 = fig.add_subplot(gse[0, 1])
    ax2.axhline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for i, c in enumerate(FAC_COND):
        dots(ax2, i, fac[c]["R"], FAC_COL[c], jitter=0.12)
        mean_bar(ax2, i, fac[c]["R"], color=FAC_COL[c], half=0.28)
    ax2.set_xticks(range(4)); ax2.set_xticklabels([FAC_SHORT[c] for c in FAC_COND], rotation=60, ha="right", rotation_mode="anchor"); ax2.set_xlim(-0.6, 3.6)
    ax2.set_ylabel("$R_{\\mathrm{ex0}}$ per seed"); ax2.set_ylim(0.65, 1.1); tidy(ax2)
    ax2.text(-0.55, 0.805, "0.80", fontsize=5.5, color=INK2, va="bottom", ha="left")

    # f: width x depth grid, small multiples of normalised profiles
    depths, widths = [6, 12, 24], [256, 512, 768, 1024]
    for i, dep in enumerate(depths):
        for j, wid in enumerate(widths):
            ax = fig.add_subplot(gs3[i, j])
            if i == 0 and j == 0:
                fig.text(0.012, 0.36, "f", fontsize=8, fontweight="bold", va="bottom", ha="left")
            runs = cells.get((dep, wid), [])
            n = dep - 1; t = n // 3
            x = np.arange(1, dep)
            ax.axhspan(0, 100, xmin=(t) / n, xmax=(n - t) / n, color="#f3f3f0", zorder=0)
            ax.axhline(1.0, color=MUTED, lw=0.5, zorder=1)
            Rtxt, ua, div = [], 0, 0
            for r in sorted(runs, key=lambda r: (r["lr"] < 3e-4, r["seed"])):
                p = r["prof"][1:] / r["prof"][1:][t:n - t].mean()
                if r["diverged"]:
                    ax.plot(x, p, color=MUTED, lw=0.7, ls=(0, (2, 1.5)), zorder=2); div += 1
                elif r["lr"] < 3e-4:
                    ax.plot(x, p, color=CAT[1], lw=0.8, zorder=3)
                else:
                    ax.plot(x, p, color=CAT[0], lw=0.8, zorder=3)
                Rtxt.append(r["R"]); ua += r["two_arm"] and not r["diverged"]
            main = [r for r in runs if r["lr"] >= 3e-4]
            low = [r for r in runs if r["lr"] < 3e-4]
            txt = "$R$ " + ", ".join(f"{r['R']:.2f}" for r in sorted(main, key=lambda r: r["seed"]))
            if low:
                txt += "\nlr 10$^{-4}$: " + ", ".join(f"{r['R']:.2f}" for r in low)
            conv = [r for r in main if not r["diverged"]]
            ntwo = sum(r["two_arm"] for r in conv)
            if ntwo:
                txt += f"\ntwo-arm U {ntwo}/{len(conv)}"
            if div:
                txt += f"\n{div}/{len(main)} diverged"
            ax.set_title(txt, fontsize=5.2, color=INK2, loc="left", pad=2, linespacing=1.1)
            ax.set_yscale("log"); ax.set_ylim(0.4, 9); ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
            ax.minorticks_off()
            ax.set_xlim(0.5, dep - 0.5)
            ax.set_xticks([1, dep - 1] if dep > 6 else [1, 5])
            tidy(ax, grid=None)
            if i == 0:
                ax.text(0.5, 1.3, f"width {wid:,}", transform=ax.transAxes, fontsize=6.5, ha="center", va="bottom")
            if j == 0:
                ax.set_ylabel(f"depth {dep}\n$\\sigma_1$ / mid-third mean")
            if i == 2:
                ax.set_xlabel("block")
    fig.savefig(os.path.join(OUT, "fig3_new.png"))
    fig.savefig(os.path.join(OUT, "fig3_new.pdf"))
    plt.close(fig)
    return d1, fac, cells


# ------------------------------------------------------------------ Fig. 4
def fig4():
    kg = kgram_data()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs1 = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.95, bottom=0.6, wspace=0.42)
    gs2 = fig.add_gridspec(1, 2, left=0.065, right=0.99, top=0.46, bottom=0.1, wspace=0.35)
    b = np.arange(1, 12)

    ax = fig.add_subplot(gs1[0, 0]); panel(ax, "a")
    for k in KS:
        m, s = band(ax, b, kg[k]["prof"][:, 1:], K_COL[k], marker="o", ms=1.8)
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (token-local, causal mask)")
    ax.set_xticks(range(1, 12, 2)); ax.set_xlim(0.7, 11.3); tidy(ax)
    ax.legend(handles=[Line2D([], [], color=K_COL[k], lw=1.2, label=f"$k$ = {k}") for k in KS],
              loc="upper right", ncol=2, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25, columnspacing=0.8)

    ax = fig.add_subplot(gs1[0, 1]); panel(ax, "b")
    for i, k in enumerate(KS):
        dots(ax, i - 0.17, kg[k]["branch"], INK2, filled=True, jitter=0.06, ms=2.8)
        dots(ax, i + 0.17, kg[k]["skip"], INK2, filled=False, jitter=0.06, ms=2.8)
        mb = mean_bar(ax, i - 0.17, kg[k]["branch"], color=INK, half=0.13)
        mean_bar(ax, i + 0.17, kg[k]["skip"], color=INK, half=0.13)
        ax.annotate(f"{mb:.2f}", (i - 0.17, kg[k]["branch"].max()), xytext=(-1, 4), textcoords="offset points", ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(range(6)); ax.set_xticklabels([str(k) for k in KS]); ax.set_xlabel("dependency order $k$")
    ax.set_ylabel("interior dependence, $\\Delta L$ (nats)"); ax.set_ylim(0, 1.05); ax.set_xlim(-0.6, 5.6); tidy(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", ms=3, label="branch rotation, blocks 4–7"),
                       Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=3, label="identity skip, blocks 4–7")],
              loc="upper left", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)

    ax = fig.add_subplot(gs1[0, 2]); panel(ax, "c")
    ax.axvline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(0.80, 0.053, "$R$ = 0.80", fontsize=5.5, color=INK2, ha="center", va="bottom")
    for k in KS:
        ax.plot(kg[k]["R"], kg[k]["branch"], "o", color=K_COL[k], ms=3.4, mew=0.6, mec="white", ls="none", zorder=4, label=f"$k$ = {k}")
    ax.set_yscale("log"); ax.set_ylim(0.05, 1.3); ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xlim(0.72, 1.18); ax.set_xlabel("$R_{\\mathrm{ex0}}$"); ax.set_ylabel("interior dependence, $\\Delta L$ (nats), log")
    tidy(ax, grid="both")
    ax.legend(loc="upper left", ncol=2, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, columnspacing=0.8)

    ax = fig.add_subplot(gs2[0, 0]); panel(ax, "d", dx=-0.13)
    ax.axhline(1.10, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for k in KS:
        ratio = kg[k]["exit"] / kg[k]["base"][:, None]
        band(ax, np.arange(12), ratio, K_COL[k], marker="o", ms=1.8)
    ax.set_xlabel("block at which the final normalisation and head are applied"); ax.set_ylabel("early-exit loss / final loss")
    ax.set_xticks(range(0, 12)); ax.set_xlim(-0.3, 11.3); ax.set_ylim(0.95, 2.45); tidy(ax)
    ax.legend(handles=[Line2D([], [], color=K_COL[k], lw=1.2, label=f"$k$ = {k}") for k in KS]
              + [Line2D([], [], color=MUTED, lw=0.5, ls=(0, (3, 2)), label="1.10 \u00d7 final loss")],
              loc="upper right", ncol=4, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25, columnspacing=0.8)

    ax = fig.add_subplot(gs2[0, 1]); panel(ax, "e", dx=-0.13)
    for k in KS:
        pb = kg[k]["pb"][:, 1:]
        m = pb.mean(0)
        ax.plot(b, np.clip(m, 1e-3, None), color=K_COL[k], lw=0.9, marker="o", ms=1.8, mew=0, zorder=3)
        for row in pb:
            ax.plot(b, np.clip(row, 1e-3, None), color=K_COL[k], lw=0.4, alpha=0.4, zorder=2)
    ax.set_yscale("log"); ax.set_ylim(0.004, 1.0); ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1]); ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
    ax.minorticks_off()
    ax.set_xlabel("block rotated alone (embedding block excluded)"); ax.set_ylabel("per-block rotation cost, $\\Delta L$ (nats), log")
    ax.set_xticks(range(1, 12)); ax.set_xlim(0.7, 11.3); tidy(ax)
    ax.legend(handles=[Line2D([], [], color=K_COL[k], lw=1.2, label=f"$k$ = {k}") for k in KS],
              loc="upper right", ncol=3, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25, columnspacing=0.8)
    fig.savefig(os.path.join(OUT, "fig4_new.png"))
    fig.savefig(os.path.join(OUT, "fig4_new.pdf"))
    plt.close(fig)
    return kg


# ------------------------------------------------------------------ ED Fig. 5
def edfig5():
    tp, al, fac, mar = topo_data(), alpha_data(), fac_data(), march_data()
    FAC_SHORT = {"AR": "AR", "CMLM": "causal MT", "MLM": "bidir. MT", "PLM": "prefix"}
    fig = plt.figure(figsize=(7.2, 9.0))
    gsa1 = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.975, bottom=0.79, wspace=0.4, width_ratios=[1.5, 1, 1.1])
    gsa2 = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.72, bottom=0.55, wspace=0.4)
    gsb = fig.add_gridspec(1, 3, left=0.065, right=0.99, top=0.47, bottom=0.29, wspace=0.4, width_ratios=[1, 1, 1.5])
    gsc = fig.add_gridspec(1, 2, left=0.065, right=0.99, top=0.2, bottom=0.05, wspace=0.1, width_ratios=[1, 1.6])
    b = np.arange(1, 12)
    TSHORT = ["base", "gated", "hwy", "multi", "U-Net", "no res."]

    # a1 residual designs, linear
    ax = fig.add_subplot(gsa1[0, 0]); fig.text(0.012, 0.975, "a", fontsize=8, fontweight="bold", va="bottom", ha="left")
    for c in TOPO[:5]:
        band(ax, b, tp[c]["prof"][:, 1:], TOPO_COL[c], marker="o", ms=1.8)
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (blocks 1\u201311)")
    ax.set_xticks(range(1, 12, 2)); ax.set_xlim(0.7, 11.3); ax.set_ylim(1.0, 5.6); tidy(ax)
    ax.legend(handles=[Line2D([], [], color=TOPO_COL[c], lw=1.2, label=TOPO_LAB[c]) for c in TOPO[:5]],
              loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.2, ncol=2, columnspacing=0.8)
    # a2 no residual, log
    ax = fig.add_subplot(gsa1[0, 1])
    p = np.clip(tp["no_residual"]["prof"][:, 1:], 1e-3, None)
    ax.plot(b, np.exp(np.log(p).mean(0)), color=TOPO_COL["no_residual"], lw=0.9, marker="o", ms=1.8, mew=0, zorder=3)
    for row in p:
        ax.plot(b, row, color=TOPO_COL["no_residual"], lw=0.4, alpha=0.5, zorder=2)
    ax.axhline(1.0, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(1e-3, 10); ax.set_yticks([0.001, 0.01, 0.1, 1, 10]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (blocks 1\u201311), log"); ax.set_xticks(range(1, 12, 2)); ax.set_xlim(0.7, 11.3); tidy(ax)
    ax.legend(handles=[Line2D([], [], color=TOPO_COL["no_residual"], lw=1.2, label="no residual (three seeds)")],
              loc="upper center", handlelength=1.2, borderaxespad=0.2)
    # a3 sensitivity
    ax = fig.add_subplot(gsa1[0, 2])
    for i, c in enumerate(TOPO):
        dots(ax, i - 0.17, tp[c]["branch"], TOPO_COL[c], filled=True, jitter=0.06, ms=2.8)
        dots(ax, i + 0.17, tp[c]["skip"], TOPO_COL[c], filled=False, jitter=0.06, ms=2.8)
        mean_bar(ax, i - 0.17, tp[c]["branch"], color=TOPO_COL[c], half=0.13)
        mean_bar(ax, i + 0.17, tp[c]["skip"], color=TOPO_COL[c], half=0.13)
    ax.set_xticks(range(6)); ax.set_xticklabels(TSHORT, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, 5.6); ax.set_ylabel("interior dependence, $\\Delta L$ (nats)"); ax.set_ylim(-0.1, 4.2); tidy(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="none", ms=3, label="branch rotation, blocks 4\u20137"),
                       Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", ms=3, label="identity skip, blocks 4\u20137")],
              loc="upper left", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    # a4-a6 strips: R, E/M, L/M
    for col, key, ylab, ref, log in ((0, "R", "$R_{\\mathrm{ex0}}$", 0.80, False), (1, "EM", "early third / middle third", 1.0, True), (2, "LM", "late third / middle third", 1.0, True)):
        ax = fig.add_subplot(gsa2[0, col])
        ax.axhline(ref, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
        for i, c in enumerate(TOPO):
            dots(ax, i, tp[c][key], TOPO_COL[c], jitter=0.14, ms=2.8)
            mean_bar(ax, i, tp[c][key], color=TOPO_COL[c], half=0.3)
        ax.set_xticks(range(6)); ax.set_xticklabels(TSHORT, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, 5.6); ax.set_ylabel(ylab); tidy(ax)
        if log:
            ax.set_yscale("log"); ax.minorticks_off()
            ax.set_yticks([0.3, 1, 3, 10, 30, 100]); ax.set_yticklabels(["0.3", "1", "3", "10", "30", "100"]); ax.set_ylim(0.2, 120)
        else:
            ax.set_ylim(0, 1.25)
            ax.text(5.55, 0.805, "0.80", fontsize=5.5, color=INK2, va="bottom", ha="right")
            ax.set_title("two-arm U per design: " + ", ".join(f"{sum(tp[c]['two_arm'])}/3" for c in TOPO), fontsize=5.5, color=INK2, loc="left", pad=2)

    # b: alpha sweep
    av = np.array([float(a) for a in ALPHAS])
    ax = fig.add_subplot(gsb[0, 0]); fig.text(0.012, 0.47, "b", fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.axhline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for a, x in zip(ALPHAS, av):
        dots(ax, x, al[a]["R"], A_COL[a], jitter=0.03, ms=2.8)
        ax.text(x, 0.02, f"U {sum(al[a]['two_arm'])}/3", fontsize=5.2, color=INK2, ha="center", va="bottom")
    ax.plot(av, [al[a]["R"].mean() for a in ALPHAS], color=INK2, lw=0.7, zorder=2)
    ax.set_xlabel("skip coefficient $\\alpha$"); ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_xticks(av); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_ylim(0, 1.05); tidy(ax)
    ax.text(0.0, 0.815, "0.80", fontsize=5.5, color=INK2, va="bottom", ha="left")
    ax = fig.add_subplot(gsb[0, 1])
    for a, x in zip(ALPHAS, av):
        dots(ax, x, al[a]["loss"], A_COL[a], jitter=0.03, ms=2.8, marker="o")
    ax.plot(av, [al[a]["loss"].mean() for a in ALPHAS], color=INK2, lw=0.7, zorder=2)
    ax.set_xlabel("skip coefficient $\\alpha$"); ax.set_ylabel("validation loss (nats)"); ax.set_xticks(av); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_ylim(5, 8); tidy(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=A_COL[a], ls="none", ms=3, label=f"$\\alpha$ = {a.rstrip('0').rstrip('.') if a != '0.0' else '0'}") for a in ALPHAS],
              loc="upper right", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    ax = fig.add_subplot(gsb[0, 2])
    for a in ALPHAS:
        p = np.clip(al[a]["prof"][:, 1:], 1e-3, None)
        m = np.exp(np.log(p).mean(0))
        ax.plot(b, m, color=A_COL[a], lw=0.9, marker="o", ms=1.8, mew=0, zorder=3)
        for row in p:
            ax.plot(b, row, color=A_COL[a], lw=0.35, alpha=0.3, zorder=2)
    ax.axhline(1.0, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log"); ax.minorticks_off(); ax.set_ylim(0.01, 20); ax.set_yticks([0.01, 0.1, 1, 10]); ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (blocks 1\u201311), log"); ax.set_xticks(range(1, 12, 2)); ax.set_xlim(0.7, 11.3); tidy(ax)

    # c: March vs September factorial
    ax = fig.add_subplot(gsc[0, 0]); fig.text(0.012, 0.2, "c", fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.axvline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for i, c in enumerate(FAC_COND):
        y = 3 - i
        ax.plot(mar[c], np.full(len(mar[c]), y + 0.16), "o", mfc="white", mec=MUTED, ms=3, mew=0.7, ls="none", zorder=4)
        ax.plot(fac[c]["R"], np.full(5, y - 0.16), "o", color=FAC_COL[c], ms=3, mew=0.6, mec="white", ls="none", zorder=4)
        ax.plot([mar[c].mean(), fac[c]["R"].mean()], [y + 0.16, y - 0.16], color=MUTED, lw=0.5, zorder=2)
    ax.set_yticks(range(4)); ax.set_yticklabels([FAC_SHORT[c] for c in FAC_COND[::-1]])
    ax.set_xlabel("$R_{\\mathrm{ex0}}$ per seed"); ax.set_xlim(0.55, 1.2); ax.set_ylim(-0.6, 3.6); tidy(ax, grid="x")
    ax.text(0.80, 3.55, "0.80", fontsize=5.5, color=INK2, ha="center", va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", mfc="white", mec=MUTED, ls="none", ms=3, label="March 2026: three seeds, the factorial script's single-position profiler, leaking prefix mask"),
                       Line2D([], [], marker="o", color=INK2, ls="none", ms=3, label="September 2026: five seeds, canonical estimator, leakage-free prefix (colours as Fig. 3e)")],
              loc="upper left", bbox_to_anchor=(1.04, 1.0), handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    fig.savefig(os.path.join(OUT, "edfig5_new.png"))
    fig.savefig(os.path.join(OUT, "edfig5_new.pdf"))
    plt.close(fig)
    return tp, al, fac, mar


def check():
    d1, fac, cells = d1_data(), fac_data(), grid_data()
    kg, tp, al, mar = kgram_data(), topo_data(), alpha_data(), march_data()
    print("d1 v2.1: condition  R per seed | branch waist | skip waist")
    for c in D1_COND:
        d = d1[c]
        print(f"  {c:10s} R {np.round(d['R'], 3).tolist()} branch {np.round(d['branch'], 3).tolist()} skip {np.round(d['skip'], 3).tolist()} two-arm {sum(d['two_arm'])}/{len(d['two_arm'])}")
        print(f"             traj R {np.round(d['traj_R'].mean(0), 3).tolist()} traj dL {np.round(d['traj_dL'].mean(0), 3).tolist()}")
    print("factorial v2:")
    for c in FAC_COND:
        print(f"  {c:5s} R {np.round(fac[c]['R'], 3).tolist()} E/M {np.round(fac[c]['EM'], 2).tolist()} L/M {np.round(fac[c]['LM'], 2).tolist()} two-arm {sum(fac[c]['two_arm'])}")
    print("March factorial:", {c: np.round(v, 3).tolist() for c, v in mar.items()})
    print("grid v2:")
    for key in sorted(cells):
        rs = sorted(cells[key], key=lambda r: (r["lr"] < 3e-4, r["seed"]))
        print(f"  d{key[0]:<3d} w{key[1]:<5d}", " | ".join(f"s{r['seed']}{' lr1e-4' if r['lr']<3e-4 else ''} R {r['R']:.2f} E/M {r['EM']:.2f} L/M {r['LM']:.2f} loss {r['loss']:.2f}{' DIV' if r['diverged'] else ''}{' U' if r['two_arm'] else ''}" for r in rs))
    print("kgram:")
    for k in KS:
        d = kg[k]
        ratio = d["exit"] / d["base"][:, None]
        within = [int(np.argmax(r <= 1.10)) for r in ratio]
        print(f"  k={k} R {np.round(d['R'], 3).tolist()} branch mean {d['branch'].mean():.3f} ({d['branch'].min():.2f}-{d['branch'].max():.2f}) skip mean {d['skip'].mean():.3f} ratio {(d['skip']/d['branch']).mean():.3f} exit-within-10% block {within} pb k mean {np.round(d['pb'].mean(0)[1:], 3).tolist()}")
    print("topology:")
    for c in TOPO:
        d = tp[c]
        print(f"  {c:14s} R {np.round(d['R'], 3).tolist()} E/M {np.round(d['EM'], 2).tolist()} L/M {np.round(d['LM'], 2).tolist()} branch {np.round(d['branch'], 3).tolist()} skip {np.round(d['skip'], 3).tolist()} loss {np.round(d['loss'], 2).tolist()}")
    print("alpha:")
    for a in ALPHAS:
        print(f"  a={a} R {np.round(al[a]['R'], 3).tolist()} loss {np.round(al[a]['loss'], 2).tolist()} two-arm {sum(al[a]['two_arm'])}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        fig3(); fig4(); edfig5()
        print("written", OUT)
