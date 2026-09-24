"""Scale point in depth (24-layer real vs shuffled, 3 + 3 seeds) with the matched assay: stats and ED Fig. 10.

Run: python scale_analysis.py -> scale/scale_stats.json, fig/edfig10_new.{png,pdf}
"""
import json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, CAT, OUT

R = "/mnt/user-data/outputs/ncs/scale/results"
L = 24; EARLY, MID, LATE = list(range(0, 8)), list(range(8, 16)), list(range(16, 24)); IE = 2   # eps = 0.1
COL = {"real": CAT[0], "shuffled": CAT[1]}


def load():
    out = {}
    for c in ("real", "shuffled"):
        rows = []
        for s in range(3):
            d = json.load(open(f"{R}/{c}_s{s}.json")); m = json.load(open(f"{R}/matched_d1_{c}_s{s}.json")); pb = m["per_block"]
            sig = np.array([b["sigma1"] for b in pb]); kl = np.array([b["kl_seq_v1"][IE] for b in pb]); klr = np.array([b["kl_seq_rand"][IE] for b in pb])
            g = np.array([b["gain_v1"][IE] for b in pb]); gr = np.array([b["gain_rand"][IE] for b in pb])
            hn = np.zeros(L); S = np.zeros(L); n = np.zeros(L)
            for q in m["raw"]:
                b = q["block"]; hn[b] += q["h_norm"]; S[b] += q["kl_seq"]["v1"][IE] / (0.1 * q["h_norm"]) ** 2; n[b] += 1
            hn, S = hn / n, S / n
            rows.append(dict(seed=s, eval=d["baseline_eval_loss"], R=d["R_ex0"], EM=d["early_over_mid"], LM=d["late_over_mid"], two=d["two_arm_rule"],
                             prof=np.array(d["sigma1_profile"]), rot_mid=d["regions"]["waist"]["dL"], rot_early=d["regions"]["early"]["dL"], rot_late=d["regions"]["late"]["dL"],
                             skip_mid=d["skip_regions"]["waist"]["dL"], per_rot=np.array(d["per_block_rotation_dL"]), sig=sig, kl=kl, klr=klr,
                             mid_kl=float(kl[MID].mean()), rho=float(stats.spearmanr(sig[1:L - 1], kl[1:L - 1])[0]), rho_abs=float(stats.spearmanr(sig[1:L - 1], S[1:L - 1])[0]), hnorm_ratio=float(hn[L - 2] / hn[1]), peak_block=int(np.argmax(kl[1:L - 1]) + 1),
                             v1_over_rand=float(np.median(kl[1:L - 1] / klr[1:L - 1])), excess=float(np.median((kl[1:L - 1] / klr[1:L - 1]) / (g[1:L - 1] / gr[1:L - 1]) ** 2)),
                             gain_over_sig=float(np.mean([b["gain_v1"][0] / b["sigma1"] for b in pb])), time_s=d["time_s"], gpu=d["gpu"], steps=d["steps"], params=d["params_M"]))
        out[c] = rows
    return out


def analyse(D):
    st = {}
    for c, rows in D.items():
        st[c] = {k: [float(r[k]) if not isinstance(r[k], bool) else r[k] for r in rows] for k in ("eval", "R", "EM", "LM", "two", "rot_mid", "rot_early", "rot_late", "skip_mid", "mid_kl", "rho", "rho_abs", "hnorm_ratio", "peak_block", "v1_over_rand", "excess", "gain_over_sig")}
        st[c]["params_M"] = rows[0]["params"]; st[c]["steps"] = rows[0]["steps"]; st[c]["time_s"] = [r["time_s"] for r in rows]
    st["real_over_shuffled_rot_mid"] = float(np.mean(st["real"]["rot_mid"]) / np.mean(st["shuffled"]["rot_mid"]))
    st["real_over_shuffled_mid_kl"] = float(np.mean(st["real"]["mid_kl"]) / np.mean(st["shuffled"]["mid_kl"]))
    st["real_over_shuffled_skip_mid"] = float(np.mean(st["real"]["skip_mid"]) / np.mean(st["shuffled"]["skip_mid"]))
    return st


def figure(D):
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.93, bottom=0.2, wspace=0.42)
    b = np.arange(L)
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    for c in ("real", "shuffled"):
        P = np.array([r["prof"] for r in D[c]])
        ax.plot(b[1:], P.mean(0)[1:], color=COL[c], lw=0.9, marker="o", ms=1.8, mew=0, label=f"{c} text, $R_{{\\mathrm{{ex0}}}}$ = {np.mean([r['R'] for r in D[c]]):.2f}")
        ax.fill_between(b[1:], P.min(0)[1:], P.max(0)[1:], color=COL[c], alpha=0.15, lw=0)
    ax.axvspan(7.5, 15.5, color="#f3f3f0", zorder=0); ax.set_xlabel("block"); ax.set_ylabel("$\\sigma_1$ (blocks 1–23)"); tidy(ax); ax.legend(loc="upper right", handlelength=1.2, fontsize=5.8)
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    for c in ("real", "shuffled"):
        P = np.array([r["per_rot"] for r in D[c]])
        ax.plot(b, P.mean(0), color=COL[c], lw=0.9, marker="o", ms=1.8, mew=0, label=c)
        ax.fill_between(b, P.min(0), P.max(0), color=COL[c], alpha=0.15, lw=0)
    ax.axvspan(7.5, 15.5, color="#f3f3f0", zorder=0); ax.set_yscale("log"); ax.set_ylim(5e-4, 2)
    ax.set_xlabel("block"); ax.set_ylabel("per-block branch-rotation $\\Delta L$ (nats)"); tidy(ax)
    ax.text(0.97, 0.97, "middle third (8–15)\nreal 0.79–0.94 nats\nshuffled 0.035–0.053", transform=ax.transAxes, fontsize=5.8, color=INK2, va="top", ha="right")
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    for c in ("real", "shuffled"):
        P = np.array([r["kl"] for r in D[c]])
        ax.plot(b, P.mean(0), color=COL[c], lw=0.9, marker="o", ms=1.8, mew=0)
        ax.fill_between(b, P.min(0), P.max(0), color=COL[c], alpha=0.15, lw=0)
    ax.axvspan(7.5, 15.5, color="#f3f3f0", zorder=0); ax.set_yscale("log")
    ax.set_xlabel("block"); ax.set_ylabel("downstream KL along $v_1$, $\\epsilon$ = 0.1 (nats)"); tidy(ax)
    rr = [r["rho"] for r in D["real"]]; rs = [r["rho"] for r in D["shuffled"]]
    ax.text(0.97, 0.97, f"$\\rho$($\\sigma_1$, KL) over blocks 1–22\nreal {min(rr):+.2f} to {max(rr):+.2f}\nshuffled {min(rs):+.2f} to {max(rs):+.2f}", transform=ax.transAxes, fontsize=5.8, color=INK2, va="top", ha="right")
    fig.savefig(os.path.join(OUT, "edfig10_new.png")); fig.savefig(os.path.join(OUT, "edfig10_new.pdf")); plt.close(fig)


if __name__ == "__main__":
    D = load(); st = analyse(D)
    os.makedirs("/mnt/user-data/outputs/ncs/scale", exist_ok=True)
    json.dump(st, open("/mnt/user-data/outputs/ncs/scale/scale_stats.json", "w"), indent=1)
    figure(D)
    for c in ("real", "shuffled"):
        s = st[c]; print(c, {k: (np.round(v, 3).tolist() if isinstance(v, list) and not isinstance(v[0], bool) else v) for k, v in s.items()})
    print("real/shuffled: rot mid", round(st["real_over_shuffled_rot_mid"], 1), "skip mid", round(st["real_over_shuffled_skip_mid"], 1), "matched mid KL", round(st["real_over_shuffled_mid_kl"], 1))
