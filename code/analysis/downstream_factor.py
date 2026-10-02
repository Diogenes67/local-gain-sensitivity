"""Combined downstream factor and reconciliation of the Fig. 3 factors on identical probes (24 Sept 2026; review 5, item 3).

For a probe (block l, input, position) and a direction u at the block input, the linearised same-position consequence per
unit squared displacement is 1/2 g^T F g with g = A u. Dividing by the squared local gain ||J u||^2 leaves the downstream
part of the map:

    combined(u) = g^T F g / ||J u||^2 = after(u) x align(u),   after = ||A u||^2 / ||J u||^2,  align = g^T F g / ||g||^2.

after and align each depend on the logit gauge (adding a constant to every logit changes ||g|| but not g^T F g); combined
does not. Reported per model as ratios of v_1 to the random-direction class: local (sigma_1 / g_rand)^2, after, align,
combined, and their product, the linearised same-position ratio KL(v_1)/KL(rand).

Reconciliation with Fig. 3a on the same probes: the measured same-position ratio KL_t(v_1)/KL_t(rand), the squared
finite-difference gain ratio and their quotient (the measured excess) at eps = 0.01 (linear regime) and 0.1 (the Fig. 3a
dose), under both aggregations used in the paper:
  A  Fig. 3a: ratio of per-block means (mean over probes of KL and of gain, ratio per block), then the median over blocks;
  B  Fig. 3c: per-probe ratios, geometric mean over probes within a block, then over blocks.
At eps = 0.01 the random-direction KL is 1e-5 to 1e-8 nats and float32 rounding makes a few values zero or negative; those
probes are excluded from B (counted in n_excluded) and enter A through the block means. The per-probe measured-to-linearised
ratio is reported as a median (Methods: 1.000 at eps = 0.01).
Uncertainty: 95% intervals from resampling probes within blocks (controlled, 20 per block) or inputs (pretrained, five
linearised inputs), 500 resamples.

Controlled: the eight linearised checkpoints (real s0-1, shuffled s0-1, k = 1 s0-1, k = 8 s0-1), blocks 1-10, from
matched/linresp (linearised response) and matched/results (assay), same probes and random directions.
Pretrained: the six decoders, blocks 1 to L-2, from matched_pretrained/results (lin_raw: five inputs x six positions,
eight random directions with per-direction random gain and KL).

Run: LINRESP=<matched/linresp> MATCHED=<matched/results> MP_RESULTS=<matched_pretrained/results> python downstream_factor.py
  -> downstream_factor.json
"""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LINRESP = os.environ.get("LINRESP", os.path.join(HERE, "matched", "linresp"))
MATCHED = os.environ.get("MATCHED", os.path.join(HERE, "matched", "results"))
MP = os.environ.get("MP_RESULTS", os.path.join(HERE, "matched_pretrained", "results"))
N_BOOT = 500; EPS_ASSAY = [0.01, 0.03, 0.1, 0.3, 1.0]; EPS_PRE = [0.01, 0.03, 0.1, 0.3]


def gm(v):
    v = np.asarray(v, float); v = v[v > 0]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def summarise(per, blocks):
    out = {}
    pos = lambda p, k: p[k] > 0
    # linearised factors, aggregation B (per-probe ratios, gm over probes then blocks)
    for k in ("local_sq", "after", "align", "combined", "lin_ratio"):
        out[k] = dict(gm=gm([gm([p[k] for p in per if p["block"] == b]) for b in blocks]),
                      median=float(np.median([p[k] for p in per])))
    out["combined_from_parts_gm"] = out["after"]["gm"] * out["align"]["gm"]
    for e in ("001", "01"):
        # A: ratio of per-block means, median over blocks
        rb, gb = [], []
        for b in blocks:
            pb = [p for p in per if p["block"] == b]
            rb.append(np.mean([p[f"kl_v_{e}"] for p in pb]) / np.mean([p[f"kl_r_{e}"] for p in pb]))
            gb.append((np.mean([p[f"gain_v_{e}"] for p in pb]) / np.mean([p[f"gain_r_{e}"] for p in pb])) ** 2)
        rb, gb = np.array(rb), np.array(gb)
        out[f"meas_ratio_{e}"] = dict(A_median=float(np.median(rb)))
        out[f"gain_sq_{e}"] = dict(A_median=float(np.median(gb)))
        out[f"excess_{e}"] = dict(A_median=float(np.median(rb / gb)))
        # B: per-probe ratios (positive measured KL only), gm over probes then blocks
        ok = [p for p in per if p[f"kl_v_{e}"] > 0 and p[f"kl_r_{e}"] > 0]
        out[f"n_excluded_{e}"] = len(per) - len(ok)
        for k, num, den in (("meas_ratio", f"kl_v_{e}", f"kl_r_{e}"), ("gain_sq", f"gain_v_{e}", f"gain_r_{e}")):
            vals = {b: [(p[num] / p[den]) ** (2 if k == "gain_sq" else 1) for p in ok if p["block"] == b] for b in blocks}
            out[f"{k}_{e}"]["B_gm"] = gm([gm(v) for v in vals.values()])
        out[f"excess_{e}"]["B_gm"] = out[f"meas_ratio_{e}"]["B_gm"] / out[f"gain_sq_{e}"]["B_gm"]
        # per-probe measured / linearised ratio (median)
        out[f"meas_over_lin_{e}"] = dict(median=float(np.median([(p[f"kl_v_{e}"] / p[f"kl_r_{e}"]) / p["lin_ratio"] for p in ok])),
                                         frac_within_2fold=float(np.mean([0.5 < (p[f"kl_v_{e}"] / p[f"kl_r_{e}"]) / p["lin_ratio"] < 2 for p in ok])))
    return out


CI_KEYS = [("after", "gm"), ("align", "gm"), ("combined", "gm"), ("local_sq", "gm"), ("excess_001", "A_median"), ("excess_001", "B_gm"), ("excess_01", "A_median"), ("excess_01", "B_gm")]


def boot(per, blocks, unit_key, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed); units = sorted({p[unit_key] for p in per}); by_unit = {u: [p for p in per if p[unit_key] == u] for u in units}
    acc = {f"{k}.{a}": [] for k, a in CI_KEYS}
    for _ in range(n):
        draw = rng.choice(units, len(units), replace=True); s = summarise([p for u in draw for p in by_unit[u]], blocks)
        for k, a in CI_KEYS: acc[f"{k}.{a}"].append(s[k][a])
    return {key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for key, v in acc.items()}


def controlled_rows():
    """Per-probe rows (blocks 1-10) of the eight linearised controlled checkpoints: name -> (per, blocks, condition)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(LINRESP, "linresp_*.json"))):
        d = json.load(open(f)); name = os.path.basename(f)[8:-5]
        m = json.load(open(os.path.join(MATCHED, f"matched_{name}.json")))
        mix = {(q["block"], q["input"], q["position"]): q for q in m["raw"]}
        probes = {}
        for r in d["raw"]:
            probes.setdefault((r["block"], r["input"], r["position"]), {}).setdefault("v1" if r["direction"] == "v1" else "rand", []).append(r)
        per = []; i001, i01 = EPS_ASSAY.index(0.01), EPS_ASSAY.index(0.1)
        for key, pr in probes.items():
            if not 1 <= key[0] <= 10: continue
            q = mix[key]; v = pr["v1"][0]; rs = pr["rand"]
            s1 = v["sigma1"]; g_r = q["gain"]["rand"][i001]
            amp_v, amp_r = v["amplification"], float(np.mean([x["amplification"] for x in rs]))
            quad_v, quad_r = v["quad"], float(np.mean([x["quad"] for x in rs]))
            after = (amp_v / s1 ** 2) / (amp_r / g_r ** 2); align = (quad_v / amp_v) / (quad_r / amp_r)
            per.append(dict(block=key[0], input=key[1], probe=f"{key[1]}_{key[2]}", sigma1=s1, local_sq=(s1 / g_r) ** 2, after=after, align=align, combined=after * align, lin_ratio=quad_v / quad_r,
                            after_v1=amp_v / s1 ** 2, down_v1=quad_v / s1 ** 2,
                            kl_v_001=q["kl_t"]["v1"][i001], kl_r_001=q["kl_t"]["rand"][i001], gain_v_001=q["gain"]["v1"][i001], gain_r_001=q["gain"]["rand"][i001],
                            kl_v_01=q["kl_t"]["v1"][i01], kl_r_01=q["kl_t"]["rand"][i01], gain_v_01=q["gain"]["v1"][i01], gain_r_01=q["gain"]["rand"][i01]))
        out[name] = (per, list(range(1, 11)), d["condition"])
    return out


def controlled():
    rows = []
    for name, (per, blocks, cond) in controlled_rows().items():
        rows.append(dict(model=name, condition=cond, n_probes=len(per), summary=summarise(per, blocks), ci95=boot(per, blocks, "probe"), ci_unit="probe"))
    return rows


def pretrained_rows():
    """Per-probe rows (blocks 1 to L-2) of the six decoders: label -> (per, blocks, family)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(MP, "mp_*.json"))):
        d = json.load(open(f)); L = d["n_layers"]; i001, i01 = EPS_PRE.index(0.01), EPS_PRE.index(0.1)
        per = []
        for x in d["lin_raw"]:
            if not 1 <= x["block"] <= L - 2: continue
            s1 = x["sigma1"]; g_r = float(np.mean(x["gain_rand_each"][i001])); amp_r = float(np.mean(x["amp_rand"])); quad_r = float(np.mean(x["quad_rand"]))
            after = (x["amp_v1"] / s1 ** 2) / (amp_r / g_r ** 2); align = (x["quad_v1"] / x["amp_v1"]) / (quad_r / amp_r)
            per.append(dict(block=x["block"], input=x["input"], probe=f"{x['input']}_{x['position']}", sigma1=s1, local_sq=(s1 / g_r) ** 2, after=after, align=align, combined=after * align,
                            lin_ratio=x["quad_v1"] / quad_r, after_v1=x["amp_v1"] / s1 ** 2, down_v1=x["quad_v1"] / s1 ** 2,
                            kl_v_001=x["kl_t_v1"][i001], kl_r_001=float(np.mean(x["kl_t_rand_each"][i001])), gain_v_001=x["gain_v1"][i001], gain_r_001=g_r,
                            kl_v_01=x["kl_t_v1"][i01], kl_r_01=float(np.mean(x["kl_t_rand_each"][i01])), gain_v_01=x["gain_v1"][i01], gain_r_01=float(np.mean(x["gain_rand_each"][i01]))))
        out[d["label"]] = (per, sorted({p["block"] for p in per}), d["family"])
    return out


def pretrained():
    rows = []
    for label, (per, blocks, fam) in pretrained_rows().items():
        rows.append(dict(model=label, family=fam, n_probes=len(per), summary=summarise(per, blocks), ci95=boot(per, blocks, "input"), ci_unit="input"))
    return rows


def main():
    out = dict(controlled=controlled(), pretrained=pretrained(), n_boot=N_BOOT)
    json.dump(out, open(os.path.join(HERE, "downstream_factor.json"), "w"), indent=1)
    for pop in ("controlled", "pretrained"):
        print(f"\n== {pop}: linearised factors (gm over probes then blocks, [95%]) and measured excess under aggregation A (Fig. 3a) and B (Fig. 3c)")
        print(f"{'model':15s} {'local^2':>7} {'after':>17} {'align':>17} {'combined':>17} {'lin':>6} | {'excess .01 A':>12} {'B':>6} | {'excess .1 A':>11} {'B':>6} | {'meas/lin .01':>12} {'.1':>6} {'excl':>5}")
        for r in out[pop]:
            s = r["summary"]; c = r["ci95"]
            f = lambda k: f"{s[k]['gm']:5.2f} [{c[k+'.gm'][0]:4.2f},{c[k+'.gm'][1]:4.2f}]"
            print(f"{r['model']:15s} {s['local_sq']['gm']:7.2f} {f('after'):>17} {f('align'):>17} {f('combined'):>17} {s['lin_ratio']['gm']:6.1f} | "
                  f"{s['excess_001']['A_median']:12.2f} {s['excess_001']['B_gm']:6.2f} | {s['excess_01']['A_median']:11.2f} {s['excess_01']['B_gm']:6.2f} | "
                  f"{s['meas_over_lin_001']['median']:12.3f} {s['meas_over_lin_01']['median']:6.2f} {s['n_excluded_001']:5d}")


if __name__ == "__main__":
    main()
