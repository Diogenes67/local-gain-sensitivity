"""Extended Data Fig. 4 (NMI layout, 24 Sept 2026): emergence during training.

a  R_ex0 against step, three 1.3B seeds (March 2026 single-position estimator; b1v2 trajectories)
b  R_ex0 against step, matched 354M autoregressive and masked-token runs (March 2026; b1_duration_comparison.txt)
c  Pythia-410M public checkpoints (E7): sigma1 per block, 30 inputs x 8 positions, canonical survey estimator (block 0 omitted)
d  Pythia-410M public checkpoints (E7): branch-rotation cost per block, census protocol (blocks 1..L-2, the census regions)
Replaces edfig4() of make_edfigs.py, whose panel c was the retained May 2026 summary.
Run: python make_edfig4_nmi.py -> fig/edfig4_nmi.{png,pdf}, edfig4_nmi_check.json
"""
import os, sys, json, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); MAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figs import panel, INK2, MUTED, GRID, CAT
plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
OUT = os.path.join(HERE, "fig"); E7 = os.path.join(HERE, "e7_checkpoints", "results")


def tidy(ax):
    ax.tick_params(length=2, width=0.5)


def lab(s):
    return f"{s // 1000}k" if s >= 1000 else str(s)


def main():
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.86, top=0.95, bottom=0.1, wspace=0.28, hspace=0.5)
    notes = {}
    # a: 1.3B seeds
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", dx=-0.14)
    finals = {}
    for i, (seed, f) in enumerate((("42", "b1v2_seed42_trajectory.json"), ("1042", "b1v2_seed1042_trajectory.json"), ("2042", "b1v2_seed2042_trajectory.json"))):
        d = json.load(open(os.path.join(MAC, "results", f)))
        tr = d.get("profile_history") or d["trajectory"]
        st = [t["step"] for t in tr]; Rv = [t["R_ex0"] for t in tr]
        ax.plot(np.array(st) / 1000, Rv, color=CAT[i], lw=0.9, marker="o", ms=1.8, mew=0, label=f"seed {seed}")
        finals[seed] = dict(final=round(Rv[-1], 3), at8k=round(Rv[st.index(8000)], 3), at32k=round(Rv[st.index(32000)], 3), first_below_080=next((s for s, r in zip(st, Rv) if r < 0.8), None))
    ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_xlabel("training step (thousands)"); ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_ylim(0.5, 1.7); tidy(ax)
    ax.legend(fontsize=5.6, handlelength=1.2); ax.set_title("1.3B, 24 layers, three seeds", fontsize=6.5, loc="left")
    notes["a"] = finals
    # b: 354M AR vs MLM
    txt = open(os.path.join(MAC, "results", "b1_duration_comparison.txt")).read()
    def block(name):
        sec = txt.split(name)[1].split("\n\n")[0]
        rows = re.findall(r"^\s*(\d+)\s+([\d.]+)", sec, flags=re.M)
        return [int(a) for a, _ in rows], [float(b) for _, b in rows]
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", dx=-0.14)
    sa, ra = block("AR (16K steps):"); sm, rm = block("MLM (20K steps):")
    ax.plot(np.array(sa) / 1000, ra, color=CAT[0], lw=0.9, marker="o", ms=1.8, mew=0, label="autoregressive")
    ax.plot(np.array(sm) / 1000, rm, color=CAT[2], lw=0.9, marker="s", ms=1.8, mew=0, label="masked-token")
    ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_xlabel("training step (thousands)"); ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_ylim(0.5, 1.2); tidy(ax)
    ax.legend(fontsize=5.6, handlelength=1.2); ax.set_title("354M, 24 layers, one seed", fontsize=6.5, loc="left")
    notes["b"] = dict(AR=list(zip(sa, ra)), MLM=list(zip(sm, rm)))
    # c, d: Pythia-410M public checkpoints (E7)
    steps = sorted(json.load(open(os.path.join(E7, "e7_summary.json")))["rows"], key=lambda r: r["step"])
    steps = [r["step"] for r in steps]
    cmap = plt.get_cmap("viridis"); col = {s: cmap(0.05 + 0.85 * i / (len(steps) - 1)) for i, s in enumerate(steps)}
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "c", dx=-0.14)
    nc = {}
    for s in steps:
        d = json.load(open(os.path.join(E7, "sigma1", f"sigma1_step{s}.json")))
        p = np.array(d["sigma1_profile"]); L = len(p) - 1; t = L // 3
        ax.plot(np.arange(1, L + 1), p[1:], color=col[s], lw=0.8, marker="o", ms=1.3, mew=0)
        nc[s] = dict(R=round(d["R_ex0_thirds"], 3), block0=round(float(p[0]), 1), max_block=int(np.argmax(p[1:]) + 1), max_val=round(float(p[1:].max()), 2), converged_frac=round(d["converged_frac"], 2))
    ax.axvspan(t + 0.5, L - t + 0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.set_xlabel("block"); ax.set_ylabel("σ₁"); ax.set_xlim(0.3, L + 0.7); tidy(ax)
    ax.set_title("Pythia-410M, σ₁ per block (block 0 omitted)", fontsize=6.5, loc="left")
    notes["c"] = nc
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "d", dx=-0.14)
    nd = {}
    for s in steps:
        d = json.load(open(os.path.join(E7, "census", f"census_step{s}.json")))
        p = np.array(d["delta_L_profile"]); L = len(p); inner = p[1:L - 1]
        nd[s] = dict(block0=round(float(p[0]), 3), last=round(float(p[-1]), 3), abs_max_inner=round(float(np.abs(inner).max()), 3), max_block=int(np.argmax(inner) + 1),
                     max_val=round(float(inner.max()), 3), regions={k: round(v, 3) for k, v in d["regions"].items()})
        if s == 0:
            continue                                   # all per-block costs within +-0.04 nats at initialisation; not drawn on the log axis
        ax.plot(np.arange(1, L - 1), np.clip(inner, 1e-3, None), color=col[s], lw=0.8, marker="o", ms=1.3, mew=0)
    n = L - 2; t = n // 3
    ax.axvspan(t + 0.5, n - t + 0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 10); ax.set_xlim(0.3, n + 0.7); ax.set_xticks([5, 10, 15, 20])
    ax.set_xlabel("block"); ax.set_ylabel("branch-rotation cost (nats)"); tidy(ax)
    ax.set_title("Pythia-410M, rotation cost per block (blocks 1 to L − 2)", fontsize=6.5, loc="left")
    notes["d"] = nd
    hl = [plt.Line2D([], [], color=col[s], lw=1.2, label=lab(s)) for s in steps]
    fig.legend(handles=hl, title="step", title_fontsize=6, fontsize=5.6, loc="center left", bbox_to_anchor=(0.87, 0.28), frameon=False, handlelength=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"edfig4_nmi.{ext}"), dpi=300)
    json.dump(notes, open(os.path.join(HERE, "edfig4_nmi_check.json"), "w"), indent=1, default=str)
    print(json.dumps({k: notes[k] for k in ("c", "d")}, default=str)[:3000])


if __name__ == "__main__":
    main()
