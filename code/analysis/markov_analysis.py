"""Entropy-matched random Markov family (orders 1-6, V = 8, H = 1.0 nats; 30 models): stats and ED Fig. 11.

Run: python markov_analysis.py -> markov/markov_stats.json, fig/edfig11_new.{png,pdf}
"""
import json, glob, os, itertools
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figs import panel, tidy, INK, INK2, MUTED, GRID, CAT, RAMP6, OUT

R = "/mnt/user-data/outputs/ncs/markov/results"
ORDERS = [1, 2, 3, 4, 5, 6]


def load():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/markov_*.json")) if not f.endswith(".old.json")]
    for r in rows: r["m"] = int(r["condition"])
    return rows


def ci_spearman(x, y, n_boot=5000, seed=0):
    x, y = np.asarray(x), np.asarray(y); rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x)); bs.append(stats.spearmanr(x[i], y[i])[0])
    rho, p = stats.spearmanr(x, y)
    return dict(rho=float(rho), p=float(p), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(len(x)))


def analyse(rows):
    st = dict(n=len(rows), n_steps=sorted(set(r["cfg"]["n_steps"] for r in rows)), gpu=rows[0]["gpu"], mean_time_min=float(np.mean([r["time_s"] for r in rows]) / 60))
    per = {}
    for m in ORDERS:
        rs = [r for r in rows if r["m"] == m]
        per[m] = dict(n=len(rs), n_states=rs[0]["source"]["n_states"], floor=[r["source"]["floor_tail"] for r in rs], held=[r["tail_loss"] for r in rs], gap=[r["learned_gap"] for r in rs],
                      margin_below_prev=[r["margin_below_prev_order"] for r in rs], floor_prev=[r["source"]["floors_by_order"][-2] for r in rs], floor_uni=[r["source"]["floors_by_order"][0] for r in rs], MI=[r["source"]["mutual_information"] for r in rs], H_marg=[r["source"]["H_marginal"] for r in rs],
                      R=[r["stats_indist"]["R_ex0"] for r in rs], EM=[r["stats_indist"]["early_over_mid"] for r in rs], LM=[r["stats_indist"]["late_over_mid"] for r in rs],
                      two_arm=int(sum(r["stats_indist"]["two_arm_rule"] for r in rs)),
                      rot_mid=[r["regions_branch"]["waist"]["dL"] for r in rs], rot_early=[r["regions_branch"]["early"]["dL"] for r in rs], rot_late=[r["regions_branch"]["late"]["dL"] for r in rs],
                      skip_mid=[r["skip_regions"]["waist"]["dL"] for r in rs], w_over_e=[r["waist_over_edge_branch"] for r in rs],
                      per_block_rot_mean=np.mean([r["per_block_branch_dL"] for r in rs], 0).tolist(), early_exit_mean=np.mean([r["early_exit_loss"] for r in rs], 0).tolist(),
                      floors_by_order=rs[0]["source"]["floors_by_order"])
    st["per_order"] = per
    m = np.array([r["m"] for r in rows]); w = np.array([r["regions_branch"]["waist"]["dL"] for r in rows]); sk = np.array([r["skip_regions"]["waist"]["dL"] for r in rows]); Rx = np.array([r["stats_indist"]["R_ex0"] for r in rows])
    for name, sel in (("all", m >= 1), ("orders_1_5", m <= 5)):
        st[f"spearman_order_rot_{name}"] = ci_spearman(m[sel], w[sel]); st[f"spearman_order_skip_{name}"] = ci_spearman(m[sel], sk[sel]); st[f"spearman_order_R_{name}"] = ci_spearman(m[sel], Rx[sel])
    means = [np.mean(w[m == k]) for k in range(1, 6)]
    st["means_1_5"] = [float(x) for x in means]; st["monotone_1_5"] = bool(all(np.diff(means) > 0))
    st["perm_p_1_5"] = float(sum(all(np.diff([means[i] for i in p]) > 0) for p in itertools.permutations(range(5))) / 120)
    st["ratio_5_over_1"] = float(means[4] / means[0]); st["skip_over_rot"] = [float(np.mean(sk[m == k]) / np.mean(w[m == k])) for k in ORDERS]
    st["R_range"] = [float(Rx.min()), float(Rx.max())]; st["two_arm_total"] = int(sum(r["stats_indist"]["two_arm_rule"] for r in rows))
    return st


def figure(rows, st):
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.99, top=0.93, bottom=0.2, wspace=0.42)
    per = st["per_order"]
    # a: middle-block rotation and skip vs order, per seed
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "a")
    for i, m in enumerate(ORDERS):
        ax.plot(m + np.linspace(-0.12, 0.12, 5), per[m]["rot_mid"], "o", color=CAT[0], ms=3.0, mew=0.4, mec="white", ls="none", zorder=4)
        ax.plot(m + np.linspace(-0.12, 0.12, 5), per[m]["skip_mid"], "s", mfc="white", mec=CAT[0], ms=2.8, mew=0.7, ls="none", zorder=4)
    ax.plot(ORDERS, [np.mean(per[m]["rot_mid"]) for m in ORDERS], color=CAT[0], lw=0.8, zorder=3, label="branch rotation")
    ax.plot(ORDERS, [np.mean(per[m]["skip_mid"]) for m in ORDERS], color=CAT[0], lw=0.8, ls=(0, (3, 2)), zorder=3, label="identity skip")
    ax.set_xlabel("order $m$ of the source"); ax.set_ylabel("middle-block $\\Delta L$ (nats)"); ax.set_xticks(ORDERS); tidy(ax); ax.legend(loc="upper left", handlelength=1.4, fontsize=5.8, bbox_to_anchor=(0.0, 1.02)); ax.set_ylim(-0.02, 0.5)
    s = st["spearman_order_rot_orders_1_5"]
    ax.text(0.03, 0.80, f"$\\rho$(order, $\\Delta L$) = {s['rho']:.2f} (CI {s['ci'][0]:.2f} to {s['ci'][1]:.2f})\nover orders 1–5, learned to the floor;\norder 6 partly learned", transform=ax.transAxes, fontsize=5.6, color=INK2, ha="left", va="top")
    # b: per-block rotation by order
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "b")
    for i, m in enumerate(ORDERS):
        ax.plot(range(12), per[m]["per_block_rot_mean"], color=RAMP6[i], lw=0.9, marker="o", ms=1.8, mew=0, label=f"$m$ = {m}")
    ax.axvspan(3.5, 7.5, color="#f3f3f0", zorder=0); ax.set_yscale("log"); ax.set_ylim(3e-4, 5)
    ax.set_xlabel("block"); ax.set_ylabel("per-block branch-rotation $\\Delta L$ (nats)"); tidy(ax); ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, fontsize=5.6)
    # c: R_ex0 vs order, per seed, and learned gap
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "c")
    for m in ORDERS:
        ax.plot(m + np.linspace(-0.12, 0.12, 5), per[m]["R"], "o", color=CAT[0], ms=3.0, mew=0.4, mec="white", ls="none", zorder=4)
    ax.axhline(0.80, color=MUTED, lw=0.5, ls=(0, (3, 2))); ax.set_ylim(0.45, 1.25)
    ax.set_xlabel("order $m$ of the source"); ax.set_ylabel("$R_{\\mathrm{ex0}}$"); ax.set_xticks(ORDERS); tidy(ax)
    s = st["spearman_order_R_all"]
    ax.text(0.03, 0.97, f"no hourglass in 30 of 30\n$\\rho$(order, $R_{{\\mathrm{{ex0}}}}$) = {s['rho']:.2f} (CI {s['ci'][0]:.2f} to {s['ci'][1]:.2f})", transform=ax.transAxes, fontsize=5.6, color=INK2, va="top")
    ins = ax.inset_axes([0.3, 0.14, 0.45, 0.3])
    ins.plot(ORDERS, [np.mean(per[m]["gap"]) for m in ORDERS], color=CAT[1], lw=0.8, marker="o", ms=2.0, mew=0)
    ins.set_xticks(ORDERS); ins.tick_params(labelsize=5, length=2); ins.set_ylabel("held loss − floor\n(nats)", fontsize=5); ins.set_xlabel("order", fontsize=5)
    for sp in ("top", "right"): ins.spines[sp].set_visible(False)
    fig.savefig(os.path.join(OUT, "edfig11_new.png")); fig.savefig(os.path.join(OUT, "edfig11_new.pdf")); plt.close(fig)


if __name__ == "__main__":
    rows = load(); st = analyse(rows)
    json.dump(st, open("/mnt/user-data/outputs/ncs/markov/markov_stats.json", "w"), indent=1)
    figure(rows, st)
    for m in ORDERS:
        p = st["per_order"][m]
        print(f"m={m} states {p['n_states']:6d} gap {np.mean(p['gap']):+.3f} [{min(p['gap']):+.3f},{max(p['gap']):+.3f}] margin {np.mean(p['margin_below_prev']):+.3f} MI {min(p['MI']):.2f}-{max(p['MI']):.2f} | R {min(p['R']):.2f}-{max(p['R']):.2f} two {p['two_arm']} | rot mid {np.mean(p['rot_mid']):.3f} [{min(p['rot_mid']):.3f}-{max(p['rot_mid']):.3f}] early {np.mean(p['rot_early']):.2f} late {np.mean(p['rot_late']):.3f} skip {np.mean(p['skip_mid']):.3f} w/e {np.mean(p['w_over_e']):.2f}")
    for k in ("spearman_order_rot_all", "spearman_order_rot_orders_1_5", "spearman_order_skip_orders_1_5", "spearman_order_R_all", "spearman_order_R_orders_1_5"): print(k, st[k])
    print("means 1-5", np.round(st["means_1_5"], 3), "monotone", st["monotone_1_5"], "perm p", st["perm_p_1_5"], "5/1", round(st["ratio_5_over_1"], 1), "skip/rot", np.round(st["skip_over_rot"], 2), "R range", st["R_range"], "time/run min", round(st["mean_time_min"], 1))
