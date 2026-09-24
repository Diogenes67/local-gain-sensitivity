"""Fig. 1 (NMI layout, 23 Sept 2026): two conventions for sizing a perturbation, and the validated gain estimator.

a  schematic: relative displacement delta_l = eps ||h_l|| v1 grows with the residual norm; absolute displacement is fixed
b  illustrative values (no data): gain falling with depth, norm growing; with the downstream factor held constant,
   normalised sensitivity follows sigma1^2 and consequence at a relative dose follows sigma1^2 ||h_l||^2
c  the estimator: hook, token-local map, power iteration on J^T J by one forward-mode and one reverse-mode product
d  power-iteration sigma1 against exact SVD of the materialised token-local Jacobian, 210 pairs in seven models
e  the same 210 Jacobians under four estimators, as a fraction of exact sigma1
Data: validation/validation_cellH.json (GPT-2 124M, Pythia-160M, Pythia-410M, Phi-2; parsed from cellH.log) and
validation/validation_convnext_mamba_from_log.json (ConvNeXt-tiny, ConvNeXt-base, Mamba-130M; parsed from the printed output
of validate_convnext_mamba.ipynb, 18 Sept 2026). Both at the printed precision of three decimals.
Precision (float16, bfloat16) is in Extended Data Fig. 2 and Table 1, not here.
Run: python make_fig1_nmi.py -> fig/fig1_nmi.{png,pdf}, fig1_nmi_check.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, CAT
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
OUT = os.path.join(HERE, "fig")
REL, ABS, GAIN, NORM = "#c2571a", "#1f5f99", "#333333", "#8a8a8a"
CLS_COL = {"transformer": CAT[0], "convolutional": CAT[1], "state-space": CAT[2]}
CLS_MK = {"transformer": "o", "convolutional": "s", "state-space": "^"}
LAB = {"gpt2": "GPT-2 124M", "EleutherAI/pythia-160m": "Pythia-160M", "EleutherAI/pythia-410m": "Pythia-410M", "microsoft/phi-2": "Phi-2"}


def load_validation():
    rows = []
    for r in json.load(open(os.path.join(HERE, "validation", "validation_cellH.json"))):
        rows.append(dict(model=LAB[r["model"]], cls="transformer", exact=r["sigma1"], est=r["sigma1_est"], rho=r["rho_est"],
                         branch=r["sigma1_branch"], probe=r["probes64"]))
    d = json.load(open(os.path.join(HERE, "validation", "validation_convnext_mamba_from_log.json")))
    for m in d["models"]:
        cls = "state-space" if "Mamba" in m["label"] else "convolutional"
        for r in m["rows"]:
            rows.append(dict(model=m["label"], cls=cls, exact=r["sigma1_exact"], est=r["sigma1_est"], rho=r["rho_est"],
                             branch=r["sigma1_branch_exact"], probe=r["random_probe_64"]))
    return rows, {m["label"]: m["summary_printed"] for m in d["models"]}


def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=5.3, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.25)


def arrow(ax, p0, p1, color=INK2, lw=0.7, cs="arc3", ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, connectionstyle=cs, shrinkA=0, shrinkB=0))


def panel_a(ax):
    L = 10; blocks = np.arange(1, L + 1); hnorm = np.geomspace(1.0, 10.0, L)
    ax.set_xlim(0.2, L + 0.9); ax.set_ylim(-3.4, 3.9); ax.axis("off")
    x = np.linspace(0.5, L + 0.5, 200); w = 0.12 + 0.05 * np.interp(x, blocks, hnorm)
    ax.fill_between(x, -w, w, color="#e9e9e9", lw=0)
    ax.annotate("", xy=(L + 0.85, 0), xytext=(L + 0.45, 0), arrowprops=dict(arrowstyle="-|>", color=NORM, lw=0.8))
    ax.text(0.3, -0.52, "residual stream", ha="left", va="top", color=NORM, fontsize=5.8)
    for b in blocks:
        ax.add_patch(FancyBboxPatch((b - 0.3, -0.28), 0.6, 0.56, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=GAIN, lw=0.6))
        ax.text(b, 0, str(b), ha="center", va="center", fontsize=6)
    ax.text(0.3, 0.52, "block", ha="left", va="bottom", fontsize=5.8, color=GAIN)
    ax.text(5.5, 0.66 + 0.05 * hnorm[4], "$\\Vert h_l \\Vert$ grows with depth", ha="center", va="bottom", fontsize=6, color=NORM)
    for b, h in zip(blocks, hnorm):
        ax.add_patch(FancyArrowPatch((b, 1.35), (b, 1.35 + 0.19 * h), arrowstyle="-|>", mutation_scale=5, color=REL, lw=0.9))
    ax.text(0.3, 3.45, "Relative convention   $\\delta_l = \\epsilon \\Vert h_l \\Vert v_1$", color=REL, fontsize=6.6, fontweight="bold", va="top")
    for b in blocks:
        ax.add_patch(FancyArrowPatch((b, -1.35), (b, -2.0), arrowstyle="-|>", mutation_scale=5, color=ABS, lw=0.9))
    ax.text(0.3, -2.35, "Absolute convention   $\\Vert \\delta_l \\Vert$ fixed", color=ABS, fontsize=6.6, fontweight="bold", va="top")
    ax.text(0.3, -2.85, "Same ε, same direction v$_1$; only the size of the displacement differs.", fontsize=5.8, color=GAIN, va="top")


def panel_b(ax):
    L = 10; blocks = np.arange(1, L + 1); sigma = np.linspace(6.0, 3.0, L); hnorm = np.geomspace(1.0, 10.0, L); eps = 0.1
    nrm = lambda v: v / v.max()
    per_unit = sigma ** 2; rel_dose = sigma ** 2 * (eps * hnorm) ** 2
    ax.plot(blocks, nrm(sigma), color=GAIN, lw=1.0)
    ax.plot(blocks, nrm(hnorm), color=NORM, lw=1.0, ls=":")
    ax.plot(blocks, nrm(per_unit), color=ABS, lw=1.4)
    ax.plot(blocks, nrm(rel_dose), color=REL, lw=1.4)
    ax.set_yscale("log"); ax.set_ylim(0.04, 1.6); ax.set_xlim(0.6, 10.1); ax.set_xticks([1, 5, 10])
    ax.set_xlabel("block"); ax.set_ylabel("value / maximum (log)"); tidy(ax, grid=None)
    kw = dict(fontsize=5.6, va="center", clip_on=False)
    ax.text(10.5, nrm(sigma)[-1], "gain σ$_1$", color=GAIN, **kw)
    ax.text(10.5, nrm(per_unit)[-1], "normalised sensitivity\n(ranks with gain)", color=ABS, **kw)
    ax.text(10.5, nrm(rel_dose)[-1] * 1.05, "consequence at a\nrelative dose\n(ranking reversed)", color=REL, **kw)
    ax.annotate("norm $\\Vert h_l \\Vert$", xy=(3.0, nrm(hnorm)[2]), xytext=(1.3, 0.34), color=NORM, fontsize=5.6,
                arrowprops=dict(arrowstyle="-", color=NORM, lw=0.5))
    ax.set_title("Illustrative values; downstream factor held constant", fontsize=5.6, color=NORM, loc="left", pad=3)


def panel_c(ax):
    ax.set_xlim(-0.25, 10.25); ax.set_ylim(0, 10); ax.axis("off")
    y1, h = 6.9, 2.2
    box(ax, 0.0, y1, 2.9, h, "natural forward\npass, all positions", fc="#f3f3f0")
    box(ax, 3.55, y1, 2.9, h, "pre-hook on block $l$\nstores its input $h_l$\nand keyword args", fc="#f3f3f0")
    box(ax, 7.1, y1, 2.9, h, "$f_t(x)$: block $l$ output\nat $t$ with input $t$ = $x$,\nothers held fixed", fc="#eaf1fb", ec=CAT[0])
    arrow(ax, (2.9, y1 + h / 2), (3.55, y1 + h / 2)); arrow(ax, (6.45, y1 + h / 2), (7.1, y1 + h / 2))
    y2 = 2.6
    box(ax, 0.0, y2, 2.9, h, "$u = Jv$\nforward-mode JVP", fc="white", ec=CAT[0])
    box(ax, 3.55, y2, 2.9, h, "$w = J^{\\mathsf{T}}u$\nreverse-mode VJP", fc="white", ec=CAT[0])
    box(ax, 7.1, y2, 2.9, h, "$v \\leftarrow w / \\|w\\|$\n$\\sigma_1 \\approx \\|u\\|$", fc="white", ec=CAT[0])
    arrow(ax, (8.55, y1), (8.55, y2 + h), color=CAT[0])
    ax.text(8.75, (y1 + y2 + h) / 2, "$J = \\partial f_t / \\partial x$", fontsize=5.6, color=CAT[0], va="center")
    arrow(ax, (2.9, y2 + h / 2), (3.55, y2 + h / 2), color=CAT[0]); arrow(ax, (6.45, y2 + h / 2), (7.1, y2 + h / 2), color=CAT[0])
    arrow(ax, (8.55, y2), (1.45, y2), color=CAT[0], cs="arc3,rad=-0.28")
    ax.text(5.0, 0.35, "iterate to tolerance 10$^{-6}$ (at most 50 steps), four restarts;\nno d × d Jacobian is formed", ha="center", va="bottom",
            fontsize=5.4, color=INK2, linespacing=1.3)


def main():
    V, summ = load_validation()
    ex = np.array([r["exact"] for r in V]); es = np.array([r["est"] for r in V])
    err = np.abs(es / ex - 1)
    fig = plt.figure(figsize=(7.08, 5.5))
    axa = fig.add_axes([0.0, 0.53, 0.56, 0.44]); panel_a(axa); axa.text(0.0, 0.985, "a", transform=axa.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axb = fig.add_axes([0.655, 0.60, 0.17, 0.33]); panel_b(axb); panel(axb, "b", dx=-0.42, dy=1.07)
    axc = fig.add_axes([0.01, 0.06, 0.36, 0.40]); panel_c(axc); axc.text(0.0, 1.0, "c", transform=axc.transAxes, fontsize=8, fontweight="bold", va="bottom")
    # d
    ax = fig.add_axes([0.47, 0.10, 0.22, 0.33]); panel(ax, "d", dx=-0.3)
    for cls in CLS_COL:
        rs = [r for r in V if r["cls"] == cls]
        ax.plot([r["exact"] for r in rs], [r["est"] for r in rs], CLS_MK[cls], color=CLS_COL[cls], ms=2.6, mew=0.3, mec="white", ls="none",
                label=f"{cls} ({len({r['model'] for r in rs})})", zorder=3)
    lo, hi = ex.min() * 0.7, ex.max() * 1.4
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("exact σ$_1$ (materialised Jacobian)"); ax.set_ylabel("power-iteration σ$_1$"); tidy(ax, grid="both")
    r = np.corrcoef(ex, es)[0, 1]
    ax.text(0.04, 0.97, f"n = {len(V)}, seven models\nPearson r = {r:.3f}\nmax. relative error {err.max() * 1e4:.0f} × 10$^{{-4}}$", transform=ax.transAxes,
            fontsize=5.4, color=INK2, va="top")
    ax.legend(loc="lower right", handlelength=0.8, fontsize=5.2, borderaxespad=0.2, handletextpad=0.3)
    # e
    ax = fig.add_axes([0.78, 0.10, 0.21, 0.33]); panel(ax, "e", dx=-0.3)
    keys = [("est", "J$^\\mathsf{T}$J\niteration"), ("branch", "branch\nJ − I"), ("rho", "iteration\non J"), ("probe", "64 random\nprobes")]
    rng = np.random.default_rng(0); check = {}
    for i, (k, lab) in enumerate(keys):
        frac = np.array([r[k] / r["exact"] for r in V])
        for cls in CLS_COL:
            f = np.array([r[k] / r["exact"] for r in V if r["cls"] == cls])
            ax.plot(i + rng.uniform(-0.2, 0.2, len(f)), f, CLS_MK[cls], color=CLS_COL[cls], ms=1.5, mew=0, alpha=0.5, ls="none", zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(frac)] * 2, color=INK, lw=1.0, zorder=4)
        check[k] = dict(median=float(np.median(frac)), min=float(frac.min()), max=float(frac.max()))
    ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in keys], fontsize=5.2); ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("fraction of exact σ$_1$"); tidy(ax)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_nmi.{ext}"), dpi=300)
    out = dict(n=len(V), pearson_r=float(r), max_rel_err_printed=float(err.max()), median_rel_err_printed=float(np.median(err)),
               max_rel_err_summary_lines=summ, estimators=check,
               probe_underestimate_pct=[float(100 * (1 - check["probe"]["max"])), float(100 * (1 - check["probe"]["min"]))],
               branch_abs_diff_max=float(max(abs(r_["branch"] - r_["exact"]) for r_ in V)))
    json.dump(out, open(os.path.join(HERE, "fig1_nmi_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
