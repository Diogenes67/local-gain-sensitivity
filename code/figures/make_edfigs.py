"""Extended Data Figs. 2, 3, 4 and 8 for the NCS submission, drawn from result files on disk.

ED Fig. 2  protocol and precision dependence   survey_v2/* (Sept 2026), survey_v2/precision/* (Drive eps sweeps)
ED Fig. 3  representational measures            metrics/results/metrics_*.json (metrics_sn1_sn2, 18 Sept 2026)
ED Fig. 4  moved to make_edfig4_nmi.py (panels a,b from the March 2026 files; c,d from the E7 checkpoint rerun)
ED Fig. 8  what perturbation sensitivity measures  March 2026 files (exp2a, d3l, exp4a, d3r)

Run: python make_edfigs.py          -> fig/edfig2_new, edfig3_new, edfig8_new (.png/.pdf)
     python make_edfigs.py --check  -> prints the numbers quoted in the legends
"""
import json, glob, os, sys, math, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from make_figs import panel, INK, INK2, MUTED, GRID, CAT, RAMP6, RAMP5, OUT, R_thirds

R = "/mnt/user-data/outputs/ncs"
SV = f"{R}/survey_v2"
MAC = "/mnt/user-data/uploads/Desktop/Interesting/PCI"
CHECK = "--check" in sys.argv


def tidy(ax):
    ax.tick_params(length=2, width=0.5)


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/sigma1_*.json")):
        j = json.load(open(f)); out[j["hf_id"]] = j
    return out


def summary(d):
    return {r["hf_id"]: r for r in json.load(open(f"{d}/survey_summary.json"))}


def nanfloat(x):
    return float("nan") if x is None else float(x)


# ============================================================ ED Fig. 2
def edfig2():
    can = load_dir(f"{SV}/incontext-natural-block-float32")
    rho = load_dir(f"{SV}/incontext-natural-block-float32-rho")
    notes = {}
    fig = plt.figure(figsize=(7.2, 8.4))
    gs0 = fig.add_gridspec(4, 6, left=0.08, right=0.99, top=0.975, bottom=0.05, hspace=0.7, wspace=0.9, height_ratios=[1.1, 1, 1, 0.95])
    class _GS:
        def __getitem__(self, k):
            r, c = k
            if isinstance(c, slice): return gs0[r, :]
            return gs0[r, c * 3:(c + 1) * 3]
    gs = _GS()
    # ---- a: rho vs sigma1 per block, eight models (small multiples in one panel: log-log scatter)
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", dx=-0.16)
    order = ["gpt2", "EleutherAI/pythia-160m", "meta-llama/Llama-3.2-1B", "EleutherAI/pythia-6.9b", "Qwen/Qwen2.5-7B",
             "facebook/convnext-tiny-224", "facebook/convnext-base-224", "state-spaces/mamba-130m-hf"]
    cols = CAT + ["#8a8985", "#b07d2b"]
    ratios, changes = [], []
    for i, hf in enumerate(order):
        s = np.array(can[hf]["sigma1_profile"]); r = np.array(rho[hf]["sigma1_profile"])
        ax.plot(s[1:], r[1:], "o", ms=2.6, mew=0.3, mec="white", color=cols[i], label=rho[hf]["label"], zorder=3)
        ax.plot(s[:1], r[:1], "s", ms=3.0, mew=0.3, mec="white", color=cols[i], zorder=3)
        q = s / r; ratios.append((rho[hf]["label"], q.min(), q.max(), np.median(q), q[0]))
        # two-arm rule (R_ex0 < 0.80 with early/middle > 1 and late/middle > 1) recomputed from both profiles; the stored
        # 'hourglass' flag in the survey files is the earlier pooled-edge rule (22 Sept 2026 fix)
        def _two(p):
            R, em, lm, _ = R_thirds(np.array(p)); return bool(R < 0.80 and em > 1 and lm > 1)
        changes.append((rho[hf]["label"], can[hf]["R_ex0_thirds"], rho[hf]["R_ex0_thirds"], _two(can[hf]["sigma1_profile"]), _two(rho[hf]["sigma1_profile"])))
    lim = [0.3, 300]; ax.plot(lim, lim, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for k, ls in ((3, (0, (1, 1.5))), (10, (0, (1, 1.5)))):
        ax.plot(lim, [l / k for l in lim], color=GRID, lw=0.5, ls=ls, zorder=1)
    ax.text(200, 200 / 3, "1/3", fontsize=5.5, color=MUTED, va="bottom"); ax.text(200, 200 / 10, "1/10", fontsize=5.5, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(0.3, 100)
    ax.set_xlabel("$\\sigma_1(J)$ per block"); ax.set_ylabel("$\\rho(J)$ per block"); tidy(ax)
    ax.legend(loc="upper left", fontsize=5.2, ncol=2, handlelength=1.0, columnspacing=0.6, borderaxespad=0.2)
    allq = np.concatenate([np.array(can[h]["sigma1_profile"]) / np.array(rho[h]["sigma1_profile"]) for h in order])
    allq_ex0 = np.concatenate([(np.array(can[h]["sigma1_profile"]) / np.array(rho[h]["sigma1_profile"]))[1:] for h in order])
    nchg = sum(1 for c in changes if bool(c[3]) != bool(c[4]))
    notes["a"] = dict(ratio_all=(allq.min(), allq.max(), np.median(allq)), ratio_ex0=(allq_ex0.min(), allq_ex0.max(), np.median(allq_ex0)),
                      block0_ratio=[(x[0], round(x[4], 1)) for x in ratios], changes=[(c[0], round(c[1], 2), round(c[2], 2)) for c in changes], n_change=nchg)
    # ---- a2 (right of a): R_ex0 under sigma1 and rho
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", dx=-0.16)
    labs = [rho[h]["label"] for h in order]; x = np.arange(len(order))
    Rs = [can[h]["R_ex0_thirds"] for h in order]; Rr = [rho[h]["R_ex0_thirds"] for h in order]
    ax.bar(x - 0.18, Rs, 0.34, color=CAT[0], label="$\\sigma_1$ (canonical)"); ax.bar(x + 0.18, Rr, 0.34, color=CAT[1], label="$\\rho$")
    ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_xticks(x); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_ylim(0, 1.3); tidy(ax); ax.legend(loc="upper left", fontsize=5.5, ncol=2, handlelength=1.0)
    ax.text(0.99, 0.96, f"classification changes in {nchg} of 8", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=INK2)
    # ---- c: four probe protocols, 13 models
    protos = [("incontext-natural-block-float32", "canonical", CAT[0], "o"), ("incontext-random-block-float32", "random token ids", CAT[2], "s"),
              ("isolated-random-block-float32", "isolated token", CAT[3], "^"), ("fullseq-random-block-float32", "whole-sequence Jacobian", CAT[1], "D")]
    sums = {p[0]: summary(f"{SV}/{p[0]}") for p in protos}
    models = [h for h in sums["incontext-random-block-float32"] if h in sums["incontext-natural-block-float32"]]
    models.sort(key=lambda h: sums["incontext-natural-block-float32"][h]["params_M"])
    ax = fig.add_subplot(gs[1, :]); panel(ax, "c", dx=-0.07)
    x = np.arange(len(models)); w = 0.2
    for j, (p, lab, col, mk) in enumerate(protos):
        vals = [nanfloat(sums[p].get(h, {}).get("R_ex0_thirds")) for h in models]
        ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=lab)
    ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels([sums["incontext-natural-block-float32"][h]["label"] for h in models], rotation=35, ha="right", fontsize=5.5)
    ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_ylim(0, 1.35); tidy(ax); ax.legend(loc="upper left", ncol=4, fontsize=5.6, handlelength=1.0, columnspacing=0.8)
    can_R = {h: sums["incontext-natural-block-float32"][h]["R_ex0_thirds"] for h in models}
    d_rand = [sums["incontext-random-block-float32"][h]["R_ex0_thirds"] - can_R[h] for h in models]
    iso = {h: sums["isolated-random-block-float32"][h]["R_ex0_thirds"] for h in models if h in sums["isolated-random-block-float32"]}
    full = {h: sums["fullseq-random-block-float32"][h]["R_ex0_thirds"] for h in models if h in sums["fullseq-random-block-float32"]}
    notes["c"] = dict(n=len(models), median_abs_d_random=float(np.median(np.abs(d_rand))), max_abs_d_random=float(np.max(np.abs(d_rand))),
                      isolated=[(sums["incontext-natural-block-float32"][h]["label"], round(can_R[h], 2), round(iso[h], 2)) for h in iso],
                      fullseq=[(sums["incontext-natural-block-float32"][h]["label"], round(can_R[h], 2), round(full[h], 2)) for h in full],
                      n_iso=len(iso), n_full=len(full))
    # ---- d: branch Jacobian J - I against block Jacobian J
    br = load_dir(f"{SV}/incontext-natural-branch-float32")
    ax = fig.add_subplot(gs[2, 0]); panel(ax, "d", dx=-0.16)
    dmax, Rd = 0, []
    for i, h in enumerate(sorted(br, key=lambda h: br[h]["params_M"])):
        if h not in can: continue
        s = np.array(can[h]["sigma1_profile"]); b = np.array(br[h]["sigma1_profile"])
        ax.plot(s, b, "o", ms=2.4, mew=0.3, mec="white", color=(CAT + [MUTED])[i % 7], label=br[h]["label"], zorder=3)
        dmax = max(dmax, np.abs(b - s).max()); Rd.append((br[h]["label"], can[h]["R_ex0_thirds"], br[h]["R_ex0_thirds"]))
    lim = [0.5, 500]; ax.plot(lim, lim, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$\\sigma_1(J)$ per block"); ax.set_ylabel("$\\sigma_1(J-I)$ per block"); tidy(ax); ax.legend(fontsize=5.2, loc="upper left", handlelength=1.0)
    notes["d"] = dict(n=len(Rd), max_abs_diff_sigma1=float(dmax), max_abs_diff_R=float(max(abs(a - b) for _, a, b in Rd)), R=[(l, round(a, 3), round(b, 3)) for l, a, b in Rd])
    # ---- e: float16 profiles against float32 for the three large Pythia models
    f16 = load_dir(f"{SV}/incontext-natural-block-float16")
    ax = fig.add_subplot(gs[2, 1]); panel(ax, "e", dx=-0.16)
    first_nf = {}; hs = []
    for i, h in enumerate(sorted(f16, key=lambda h: f16[h]["params_M"])):
        p16 = np.array([nanfloat(v) for v in f16[h]["sigma1_profile"]]); p32 = np.array(can[h]["sigma1_profile"])
        fin = [all(c["finite"][k] for c in f16[h]["capture_diagnostics"]) for k in range(len(p16))]
        nf = fin.index(False) if False in fin else None; first_nf[f16[h]["label"]] = nf
        col = [CAT[0], CAT[1], CAT[3]][i]
        ax.plot(np.arange(1, len(p32)), p32[1:], color=col, lw=0.9)
        ax.plot(np.arange(1, len(p16)), p16[1:], color=col, lw=0.9, ls=(0, (2, 1.5)))
        if nf is not None: ax.axvspan(nf - 0.5, len(p16) - 0.5, color=col, alpha=0.07, lw=0)
        hs.append(Line2D([], [], color=col, lw=0.9, label=f"{f16[h]['label']} (from block {nf})"))
    hs += [Line2D([], [], color=INK2, lw=0.9, label="float32, AD"), Line2D([], [], color=INK2, lw=0.9, ls=(0, (2, 1.5)), label="float16, finite differences")]
    ax.set_yscale("log"); ax.set_ylim(2.5, 40); ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$"); tidy(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks([3, 5, 10, 20, 40])
    ax.legend(handles=hs, fontsize=5.2, ncol=2, handlelength=1.4, columnspacing=0.8, loc="upper left")
    notes["e"] = dict(first_nonfinite=first_nf)
    # ---- f, g, h: precision sweep, R_ex0 under fd float32/float16/bfloat16 at three eps against float32 AD
    P = f"{SV}/precision"
    gs = [gs0[3, 0:2], gs0[3, 2:4], gs0[3, 4:6]]
    small = ["gpt2", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "Qwen/Qwen2.5-1.5B", "gpt2-xl"]
    adR = {h: can[h]["R_ex0_thirds"] for h in small}
    prec_notes = {}
    for j, (tag, lab) in enumerate((("float32fd", "float32, finite differences"), ("float16", "float16, finite differences"), ("bfloat16", "bfloat16, finite differences"))):
        ax = fig.add_subplot(gs[j]); panel(ax, "fgh"[j], dx=-0.28)
        x = np.arange(len(small)); w = 0.26
        ax.bar(x - 1.5 * w, [adR[h] for h in small], w, color=INK2, label="float32 AD")
        nonfinite = {}
        for k, eps in enumerate(("0.001", "0.01", "0.1")):
            f = f"{P}/{tag}_eps{eps}.json"
            if not os.path.exists(f): continue
            S = {r["hf_id"]: r for r in json.load(open(f))}
            vals = [nanfloat(S[h]["R_ex0_thirds"]) if h in S else float("nan") for h in small]
            nonfinite[eps] = [S[h]["label"] for h in small if h in S and math.isnan(nanfloat(S[h]["R_ex0_thirds"]))]
            ax.bar(x + (k - 0.5) * w, vals, w, color=RAMP5[1 + k], label=f"$\\varepsilon$ = {eps}")
            for xi, v in zip(x, vals):
                if math.isnan(v): ax.text(xi + (k - 0.5) * w, 0.02, "×", ha="center", va="bottom", fontsize=6, color=RAMP5[1 + k])
            prec_notes[f"{tag}_eps{eps}"] = dict(vals=[(S[h]["label"], None if math.isnan(nanfloat(S[h]["R_ex0_thirds"])) else round(S[h]["R_ex0_thirds"], 3)) for h in small if h in S],
                                              max_abs_dev=float(np.nanmax(np.abs(np.array(vals) - np.array([adR[h] for h in small])))))
        ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_ylim(0, 1.25 if tag != "bfloat16" else 2.8)
        ax.set_xticks(x); ax.set_xticklabels([can[h]["label"] for h in small], rotation=35, ha="right", fontsize=5.3); ax.set_title(lab, fontsize=6.5, loc="left")
        if j == 0: ax.set_ylabel("$R_{\\mathrm{ex0}}$")
        tidy(ax); ax.legend(fontsize=5.0, ncol=2, handlelength=1.0, columnspacing=0.6, loc="upper left" if tag != "bfloat16" else "upper right")
    notes["fgh"] = prec_notes
    fig.savefig(os.path.join(OUT, "edfig2_new.png")); fig.savefig(os.path.join(OUT, "edfig2_new.pdf")); plt.close(fig)
    return notes


# ============================================================ ED Fig. 3
def edfig3():
    files = sorted(glob.glob(f"{R}/metrics/results/metrics_*.json"), key=lambda f: json.load(open(f))["params_M"])
    M = [json.load(open(f)) for f in files]
    metrics = [("sigma1", "$\\sigma_1$", CAT[0]), ("cka", "CKA", CAT[1]), ("erank", "effective rank", CAT[2]), ("frob", "$\\|J\\|_F$", CAT[3]), ("sigma1_over_frob", "$\\sigma_1/\\|J\\|_F$", CAT[4])]
    fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 5, left=0.07, right=0.99, top=0.95, bottom=0.1, hspace=0.55, wspace=0.45)
    ax = fig.add_subplot(gs[0, :3]); panel(ax, "a", dx=-0.1)
    x = np.arange(len(M)); w = 0.16
    for j, (key, lab, col) in enumerate(metrics):
        vals = [m[f"{key}_stats"]["R_ex0_thirds"] for m in M]
        ax.bar(x + (j - 2) * w, vals, w, color=col, label=lab)
    ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.axhline(1.0, color=GRID, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([m["label"] for m in M]); ax.set_ylabel("middle-to-edge ratio $R_{\\mathrm{ex0}}$"); ax.set_ylim(0, 2.2); tidy(ax)
    ax.legend(ncol=5, fontsize=5.6, handlelength=1.0, columnspacing=0.8, loc="upper left")
    ax = fig.add_subplot(gs[0, 3:]); panel(ax, "b", dx=-0.2)
    for i, m in enumerate(M):
        p = np.array(m["sigma1_over_frob"]); ax.plot(np.arange(1, len(p)), p[1:], color=RAMP5[i], lw=0.9, marker="o", ms=1.8, mew=0, label=m["label"])
    ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1/\\|J\\|_F$"); tidy(ax); ax.legend(fontsize=5.4, handlelength=1.2)
    # per-metric profiles for the five models (row 2)
    for j, (key, lab, col) in enumerate(metrics[:4]):
        ax = fig.add_subplot(gs[1, j]); panel(ax, "cdef"[j], dx=-0.3)
        for i, m in enumerate(M):
            p = np.array(m[f"{key}_profile" if key != "sigma1_over_frob" else "sigma1_over_frob"]); n = len(p)
            xs = np.linspace(0, 1, n - 1); ax.plot(xs, p[1:], color=RAMP5[i], lw=0.8)
        if key in ("sigma1", "frob"):
            ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yticks([3, 5, 10, 20] if key == "sigma1" else [20, 30, 50, 70])
        ax.set_title(lab, fontsize=6.5, loc="left"); ax.set_xlabel("relative depth"); tidy(ax)
    ax = fig.add_subplot(gs[1, 4]); panel(ax, "g", dx=-0.3)
    for i, m in enumerate(M):
        ax.plot([0, 1], [m["sigma1_stats"]["R_ex0_thirds"], m["sigma1_over_frob_stats"]["R_ex0_thirds"]], color=RAMP5[i], lw=0.9, marker="o", ms=2.2, mew=0)
    ax.axhline(0.8, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_xticks([0, 1]); ax.set_xticklabels(["$\\sigma_1$", "$\\sigma_1/\\|J\\|_F$"]); ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_title("normalisation", fontsize=6.5, loc="left"); tidy(ax)
    fig.savefig(os.path.join(OUT, "edfig3_new.png")); fig.savefig(os.path.join(OUT, "edfig3_new.pdf")); plt.close(fig)
    notes = {}
    for m in M:
        notes[m["label"]] = {k: round(m[f"{k}_stats"]["R_ex0_thirds"], 3) for k, _, _ in metrics}
        notes[m["label"]]["norm_minus_sigma1"] = round(m["sigma1_over_frob_stats"]["R_ex0_thirds"] - m["sigma1_stats"]["R_ex0_thirds"], 3)
    order_s = [m["label"] for m in sorted(M, key=lambda m: m["sigma1_stats"]["R_ex0_thirds"])]
    order_n = [m["label"] for m in sorted(M, key=lambda m: m["sigma1_over_frob_stats"]["R_ex0_thirds"])]
    notes["order_sigma1"] = order_s; notes["order_normalised"] = order_n; notes["order_same"] = order_s == order_n
    return notes


# ED Fig. 4 moved to make_edfig4_nmi.py (24 Sept 2026): panels c and d are drawn from the E7 checkpoint rerun.


# ============================================================ ED Fig. 8
def edfig8():
    fig = plt.figure(figsize=(7.2, 2.4)); gs = fig.add_gridspec(1, 4, left=0.06, right=0.99, top=0.92, bottom=0.24, wspace=0.55)
    notes = {}
    # a: Dyck-2 probes
    E = [json.load(open(f)) for f in sorted(glob.glob(f"{MAC}/results/exp2a_bracket_seed*.json"))]
    r2 = np.array([e["probes"]["depth_r2"] for e in E]); dL = np.array([e["delta_L_per_layer"] for e in E]); pr = np.array([e["dimensionality"]["participation_ratio"] for e in E])
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", dx=-0.3)
    xb = np.arange(12); ax.bar(xb, dL.mean(0), 0.7, color=CAT[2], yerr=dL.std(0), error_kw=dict(lw=0.5, capsize=1.5), label="per-block rotation ΔL (nats)")
    ax.set_xlabel("block (−1, embedding)"); ax.set_ylabel("ΔL (nats)", color=CAT[2]); tidy(ax)
    ax2 = ax.twinx(); ax2.plot(np.arange(13) - 1, r2.mean(0), color=CAT[1], lw=0.9, marker="o", ms=2.0, mew=0, label="probe $R^2$ (nesting depth)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("probe $R^2$", color=CAT[1]); ax2.tick_params(colors=CAT[1], length=2, width=0.5); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(CAT[1]); ax2.spines["top"].set_visible(False)
    ax.set_title(f"Dyck-2 models (n = {len(E)})", fontsize=6.5, loc="left")
    notes["a"] = dict(n=len(E), r2_by_position=[round(v, 3) for v in r2.mean(0)], dL_by_block=[round(v, 2) for v in dL.mean(0)], pr_by_position=[round(v, 1) for v in pr.mean(0)])
    # b: downstream-only adaptation
    d = json.load(open(f"{MAC}/results/d3l_alignment_adaptation/d3l_aggregate.json"))
    seeds = sorted(d["per_seed"]); before = [d["per_seed"][s]["dL_target_before"] for s in seeds]; after = [d["per_seed"][s]["dL_target_after"] for s in seeds]; ctrl = [d["per_seed"][s]["control_dL_target"] for s in seeds]
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", dx=-0.3)
    for i, (lab, vals, col) in enumerate((("rotated, before", before, CAT[0]), ("rotated, after 500 steps", after, CAT[1]), ("no rotation, after 500 steps", ctrl, MUTED))):
        ax.bar(i, np.mean(vals), 0.6, color=col); ax.plot(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, "o", ms=2.2, mec="white", mew=0.3, color=INK, zorder=4)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["rotated,\nbefore", "rotated,\nfine-tuned", "no rotation,\nfine-tuned"], fontsize=5.5); ax.set_ylabel("ΔL at block 5 (nats)"); tidy(ax)
    ax.text(1, np.mean(after) + 0.25, f"{d['aggregate']['mean_reduction_pct']:.0f}% recovered", ha="center", fontsize=5.6, color=INK2)
    ax.set_title(f"k = 8, block 5 (n = {len(seeds)} seeds)", fontsize=6.5, loc="left")
    notes["b"] = dict(mean_reduction_pct=d["aggregate"]["mean_reduction_pct"], range=(d["aggregate"]["min_reduction_pct"], d["aggregate"]["max_reduction_pct"]), before=np.mean(before), after=np.mean(after), control=np.mean(ctrl), clean_nll_change=d["aggregate"]["mean_clean_nll_change"])
    # c: singular-direction perturbation
    files = sorted(glob.glob(f"{MAC}/results/exp4a_singular_perturbation/exp4a_controlled_*gram_seed*.json"))
    per_k = {}
    for f in files:
        j = json.load(open(f)); e = j["perturbation"]["eps_0.1"]
        per_k.setdefault(j["k"], []).append((np.mean(e["singular_dL"]), e["random_dL_mean"], e["v1_over_random"], e["singular_dL"][0]))
    ks = sorted(per_k)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c", dx=-0.3)
    x = np.arange(len(ks)); w = 0.36
    sv = [np.mean([v[0] for v in per_k[k]]) for k in ks]; rv = [np.mean([v[1] for v in per_k[k]]) for k in ks]; ratio = [np.mean([v[2] for v in per_k[k]]) for k in ks]
    ax.bar(x - w / 2, sv, w, color=CAT[0], label="top five right singular directions"); ax.bar(x + w / 2, rv, w, color=MUTED, label="norm-matched random")
    for xi, s, r in zip(x, sv, ratio): ax.text(xi, s * 1.15, f"{r:.0f}×", ha="center", fontsize=5.6, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 2e-2); ax.set_xticks(x); ax.set_xticklabels([f"k = {k}" for k in ks]); ax.set_ylabel("loss increase (nats)"); tidy(ax); ax.legend(fontsize=5.2, loc="upper left", handlelength=1.0)
    ax.set_title(f"block 5, ε = 0.1 (n = {len(files)})", fontsize=6.5, loc="left")
    notes["c"] = dict(n=len(files), ks=ks, singular=[float(v) for v in sv], random=[float(v) for v in rv], v1_over_random_mean=[float(v) for v in ratio], overall_mean_ratio=float(np.mean([v[2] for k in ks for v in per_k[k]])))
    # d: residual-gain curvature by region
    d = json.load(open(f"{MAC}/results/d3r_residual_gain_stress/d3r_aggregate.json"))
    ax = fig.add_subplot(gs[0, 3]); panel(ax, "d", dx=-0.3)
    regions = ["early", "body", "late"]; x = np.arange(len(regions)); w = 0.26
    for j, k in enumerate(d["k_values"]):
        m = [d["curvature_by_k_region"][f"k{k}_{r}"]["mean"] for r in regions]; s = [d["curvature_by_k_region"][f"k{k}_{r}"]["std"] for r in regions]
        ax.bar(x + (j - 1) * w, m, w, yerr=s, color=RAMP6[[1, 2, 5][j]], error_kw=dict(lw=0.5, capsize=1.5), label=f"k = {k}")
    ax.set_xticks(x); ax.set_xticklabels(["early\n(0–3)", "middle\n(4–7)", "late\n(8–11)"], fontsize=5.5); ax.set_ylabel("residual-gain curvature $S_r$"); tidy(ax); ax.legend(fontsize=5.4, handlelength=1.0)
    body = d["curvature_by_k_region"]; notes["d"] = dict(body_k2=body["k2_body"]["mean"], body_k8=body["k8_body"]["mean"], ratio=body["k8_body"]["mean"] / body["k2_body"]["mean"], alphas=d["alphas"], n_seeds=d["n_seeds"])
    fig.savefig(os.path.join(OUT, "edfig8_new.png")); fig.savefig(os.path.join(OUT, "edfig8_new.pdf")); plt.close(fig)
    return notes


if __name__ == "__main__":
    n2 = edfig2(); n3 = edfig3(); n8 = edfig8()
    if CHECK:
        import pprint
        for name, n in (("ED2", n2), ("ED3", n3), ("ED8", n8)):
            print("=" * 20, name); pprint.pprint(n, width=160, compact=True)
    print("wrote edfig2_new, edfig3_new, edfig8_new")
