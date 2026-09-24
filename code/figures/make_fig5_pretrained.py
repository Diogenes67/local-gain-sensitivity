"""Fig. 5 (NMI layout, 23 Sept 2026): pretrained models.

a  sigma1 per block (block 0 omitted) for four models, 30 inputs x 8 positions (survey_30inputs, E3); ribbon 16th-84th percentile
b  census: relative middle-block sensitivity (waist-to-edge ratio, blocks 1..L-2) against the five-input survey gain ratio, 44 models, by family
c  Pythia-410M public checkpoints (E7): gain ratio (30 inputs, 95% CI over inputs) and middle-third branch-rotation cost (census protocol)
Run: python make_fig5_pretrained.py -> fig/fig5_pretrained_nmi.{png,pdf}, fig5_pretrained_nmi_check.json
"""
import os, sys, json, csv
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from make_figs import panel, tidy, INK, INK2, MUTED, GRID
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
OUT = os.path.join(HERE, "fig"); MAC = os.path.dirname(HERE)
FAM_COL = {"AR decoder": "#2a78d6", "SSM": "#eb6834", "ConvNet (vision)": "#1baf7a", "MLP-Mixer (vision)": "#008300", "Vision supervised": "#4a3aa7",
           "Vision SSL": "#8e7cc3", "Masked encoder": "#e87ba4", "Encoder-decoder": "#8a5a00", "Audio encoder": "#52514e"}
FAM_LAB = {"AR decoder": "autoregressive decoder", "SSM": "state-space", "ConvNet (vision)": "ConvNeXt", "MLP-Mixer (vision)": "MLP-Mixer", "Vision supervised": "ViT",
           "Vision SSL": "DINOv2", "Masked encoder": "masked encoder", "Encoder-decoder": "T5 encoder", "Audio encoder": "audio encoder"}
A_MODELS = [("EleutherAI_pythia-1_4b", "Pythia-1.4B", "AR decoder"), ("facebook_convnext-base-224", "ConvNeXt-base", "ConvNet (vision)"),
            ("state-spaces_mamba-130m-hf", "Mamba-130M", "SSM"), ("google_gemma-2-2b", "Gemma-2-2B", "AR decoder")]


def thirds(p):
    p = np.asarray(p, float)[1:]; n = len(p); t = n // 3
    return p[t:n - t].mean() / ((p[:t].mean() + p[n - t:].mean()) / 2), t


def main():
    check = {}
    fig = plt.figure(figsize=(7.08, 4.9))
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.945, top=0.94, bottom=0.17, wspace=0.45, hspace=0.55)
    # a
    for i, (fn, lab, fam) in enumerate(A_MODELS):
        d = json.load(open(os.path.join(HERE, "survey_30inputs", "results", f"sigma1_{fn}.json")))
        S = d["sigma1_samples"][1:]; L = len(S); x = np.arange(1, L + 1)
        m = np.array([np.mean(s) for s in S]); lo = np.array([np.percentile(s, 16) for s in S]); hi = np.array([np.percentile(s, 84) for s in S])
        R, t = thirds(d["sigma1_profile"])
        ax = fig.add_subplot(gs[0, i]); ax.axvspan(t + 0.5, L - t + 0.5, color=GRID, alpha=0.7, lw=0)
        ax.fill_between(x, lo, hi, color=FAM_COL[fam], alpha=0.2, lw=0); ax.plot(x, m, color=FAM_COL[fam], lw=0.9)
        ax.set_title(f"{lab}, ratio {R:.2f}", fontsize=6.3, loc="left"); ax.set_xlabel("block"); tidy(ax)
        if i == 0:
            ax.set_ylabel("σ₁"); panel(ax, "a")
        check.setdefault("a", {})[lab] = round(float(R), 3)
    # b census
    ax = fig.add_subplot(gs[1, 0:2])
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "census_v4", "census_v4.csv"))) if r["valid_readout"] == "True" and r["ceiling_ok"] == "True"]
    xs, ys = [], []
    for r in rows:
        x_, y_ = float(r["R_ex0"]), float(r["waist_over_edge_int"]); xs.append(x_); ys.append(y_)
        mk = "o" if r["two_arm"] == "True" else "o"
        ax.scatter(x_, min(y_, 1.45), s=16, color=FAM_COL[r["family"]], lw=0.6 if r["two_arm"] == "True" else 0, edgecolor=INK if r["two_arm"] == "True" else "none", zorder=3)
        if y_ > 1.45:
            ax.annotate(f"{r['label']} ({y_:.1f})", (x_, 1.45), xytext=(4, -2), textcoords="offset points", fontsize=4.8, color=INK2)
    rho = stats.spearmanr(xs, ys)[0]
    ax.axvline(0.80, color=MUTED, ls="--", lw=0.6); ax.set_ylim(-0.05, 1.5)
    ax.set_xlabel("middle-to-edge gain ratio"); ax.set_ylabel("middle-block rotation cost /\nlarger edge-third cost"); tidy(ax, grid="both")
    ax.text(0.98, 0.80, f"Spearman ρ = {rho:.2f}, n = {len(xs)}\n95% CI −0.25 to 0.36", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=INK2, linespacing=1.3)
    fams = [f for f in FAM_COL if any(r["family"] == f for r in rows)]
    hl = [plt.Line2D([], [], marker="o", ls="", color=FAM_COL[f], ms=3.5, label=FAM_LAB[f]) for f in fams]
    hl.append(plt.Line2D([], [], marker="o", ls="", color="white", markeredgecolor=INK, ms=3.5, label="interior valley"))
    ax.legend(handles=hl, fontsize=4.8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), handletextpad=0.2, columnspacing=0.8, frameon=False)
    check["b"] = dict(rho=round(float(rho), 3), n=len(xs))
    panel(ax, "b", dx=-0.12)
    # c Pythia-410M public checkpoints (E7, 23 Sept 2026): canonical survey sigma1 on 30 inputs; census branch rotation, middle third
    rows = sorted(json.load(open(os.path.join(HERE, "e7_checkpoints", "results", "e7_summary.json")))["rows"], key=lambda r: r["step"])
    st = np.array([r["step"] for r in rows], float); Rv = np.array([r["R_ex0"] for r in rows]); dl = np.array([r["dL_middle"] for r in rows])
    err = np.array([[r["R_ex0"] - r["R_ci"][0] for r in rows], [r["R_ci"][1] - r["R_ex0"] for r in rows]])
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_xscale("symlog", linthresh=1000, linscale=0.6)
    ax.axhline(0.80, color=MUTED, ls="--", lw=0.6)
    ax.errorbar(st, Rv, yerr=err, color="#2a78d6", lw=0.9, marker="o", ms=2.5, mew=0, elinewidth=0.6, capsize=0)
    ax.set_ylim(0.5, 1.1); ax.set_ylabel("middle-to-edge gain ratio", color="#2a78d6")
    ax2 = ax.twinx(); ax2.plot(st, dl, color="#eb6834", lw=0.9, marker="s", ms=2.5, mew=0)
    ax2.set_ylim(-0.02, 0.45); ax2.set_ylabel("middle-third rotation cost (nats)", color="#eb6834")
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False); ax2.tick_params(length=2)
    ax.set_xlim(-150, 2.2e5); ax.set_xticks([0, 1e3, 1e4, 1e5]); ax.set_xticklabels(["0", "10³", "10⁴", "10⁵"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([k * 10 ** e for e in (3, 4) for k in range(2, 10)] + [2e5]))
    ax.set_xlabel("Pythia-410M training step (11 public checkpoints)"); tidy(ax)
    check["c"] = dict(steps=[int(v) for v in st], R=[round(float(v), 3) for v in Rv], R_ci=[[round(v, 3) for v in r["R_ci"]] for r in rows],
                      dL_middle=[round(float(v), 3) for v in dl], dL_early=[round(r["dL_early"], 3) for r in rows], dL_late=[round(r["dL_late"], 3) for r in rows],
                      mid_over_edge=[round(r["dL_middle_over_larger_edge"], 3) for r in rows], clean_loss=[round(r["clean_loss"], 3) for r in rows])
    panel(ax, "c", dx=-0.12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig5_pretrained_nmi.{ext}"), dpi=300)
    json.dump(check, open(os.path.join(HERE, "fig5_pretrained_nmi_check.json"), "w"), indent=1)
    print(json.dumps(check))


if __name__ == "__main__":
    main()
