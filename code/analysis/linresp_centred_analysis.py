"""Amplification and alignment in three logit gauges (25 Sept 2026; review 5, item 3).

Reads the gauge-recorded linearised-response runs (linresp_centred.py for the controlled checkpoints, linresp_pretrained_centred.py
for the six decoders) and reports, per model and per gauge, the factors of the v_1-to-random ratio with the conventions of
downstream_factor.py: after-block factor (amp_v1 / sigma_1^2) / (mean_r amp / g_rand^2) and alignment
(quad_v1 / amp_v1) / (mean_r quad / mean_r amp), so that local^2 x after x align = quad_v1 / mean_r quad on every probe in every
gauge. Gauges: raw amp = ||g||^2; mean-centred amp_c = ||g - mean(g) 1||^2; p-centred amp_p = ||g - (p^T g) 1||^2. Geometric
mean over probes within a block and then over blocks (1-10 controlled; 1 to L-2 pretrained); 95% percentile intervals from
500 resamples of probes within blocks (controlled) or of inputs (pretrained). Also reported: the constant component of the
response, V mean(g)^2 / ||g||^2, along v_1 and averaged over the random directions (median over probes).

Run: LC_CONTROLLED=<linresp_centred/results> LC_PRETRAINED=<matched_pretrained_centred/results> python linresp_centred_analysis.py
  -> linresp_centred_stats.json
"""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LC_C = os.environ.get("LC_CONTROLLED", os.path.join(HERE, "matched", "linresp_centred", "results"))
LC_P = os.environ.get("LC_PRETRAINED", os.path.join(HERE, "matched_pretrained_centred", "results"))
N_BOOT = 500
GAUGES = (("raw", "amplification", "alignment"), ("centred", "amplification_centred", "alignment_centred"), ("pcentred", "amplification_pcentred", "alignment_pcentred"))


def gm(v):
    v = np.asarray(v, float); v = v[v > 0]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def summarise(per, blocks):
    out = {}
    for g, a, al in GAUGES:
        for k in ("amp", "after", "align"):
            out[f"{k}_{g}"] = gm([gm([p[f"{k}_{g}"] for p in per if p["block"] == b]) for b in blocks])
        out[f"product_{g}"] = out[f"after_{g}"] * out[f"align_{g}"]
    out["local_sq"] = gm([gm([p["local_sq"] for p in per if p["block"] == b]) for b in blocks])
    out["combined"] = gm([gm([p["combined"] for p in per if p["block"] == b]) for b in blocks])
    out["const_share_v1"] = float(np.median([p["const_v1"] for p in per])); out["const_share_rand"] = float(np.median([p["const_rand"] for p in per]))
    return out


def boot(per, blocks, unit_key):
    rng = np.random.default_rng(0); units = sorted({p[unit_key] for p in per}); by = {u: [p for p in per if p[unit_key] == u] for u in units}
    keys = [f"{k}_{g}" for g, _, _ in GAUGES for k in ("amp", "after", "align")] + ["combined", "local_sq"]
    acc = {k: [] for k in keys}
    for _ in range(N_BOOT):
        s = summarise([p for u in rng.choice(units, len(units), replace=True) for p in by[u]], blocks)
        for k in keys: acc[k].append(s[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc.items()}


def rows_pretrained(d):
    L = d["n_layers"]; per = []
    for r in d["raw"]:
        if not 1 <= r["block"] <= L - 2: continue
        rec = dict(block=r["block"], input=r["input"], probe=f"{r['input']}_{r['position']}")
        quad_v, quad_r = r["quad"][0], float(np.mean(r["quad"][1:])); s1, g_r = r["sigma1"], g_rand_of(r)
        rec["local_sq"] = (s1 / g_r) ** 2
        for g, a, al in GAUGES:
            amp_v, amp_r = r[a][0], float(np.mean(r[a][1:]))
            rec[f"amp_{g}"] = amp_v / amp_r; rec[f"after_{g}"] = (amp_v / s1 ** 2) / (amp_r / g_r ** 2); rec[f"align_{g}"] = (quad_v / amp_v) / (quad_r / amp_r)
        # downstream factor with the local gain: sigma1 for v1, the assay's random-direction gain at eps 0.01 for the random class
        rec["combined"] = (quad_v / quad_r) / (s1 / g_r) ** 2
        rec["const_v1"] = r["V"] * r["g_mean"][0] ** 2 / r["amplification"][0]
        rec["const_rand"] = float(np.mean([r["V"] * r["g_mean"][j] ** 2 / r["amplification"][j] for j in range(1, len(r["g_mean"]))]))
        per.append(rec)
    return per, sorted({p["block"] for p in per})


_MP = {}


def g_rand_of(r):
    """The assay's mean random-direction gain at eps = 0.01 on this probe (from the record's assay block when present)."""
    if "assay_gain_rand" in r:
        return r["assay_gain_rand"]
    raise KeyError("assay gain missing")


def attach_assay_gain(d, mp_dir):
    """Attach the matched assay's random-direction gain at eps 0.01 per probe (needed for the local factor)."""
    import re
    f = os.path.join(mp_dir, f"mp_{re.sub(r'[^A-Za-z0-9._-]', '_', d['hf_id']).replace('.', '_')}.json")
    cands = glob.glob(os.path.join(mp_dir, "mp_*.json"))
    m = None
    for c in cands:
        dd = json.load(open(c))
        if dd["label"] == d["label"]: m = dd; break
    assert m is not None, d["label"]
    lin = {(x["block"], x["input"], x["position"]): x for x in m["lin_raw"]}
    for r in d["raw"]:
        x = lin[(r["block"], r["input"], r["position"])]
        r["assay_gain_rand"] = float(np.mean(x["gain_rand_each"][0]))


def rows_controlled(d, matched_dir):
    name = f"{d['experiment']}_{d['condition']}_s{d['seed']}"
    m = json.load(open(os.path.join(matched_dir, f"matched_{name}.json")))
    mix = {(q["block"], q["input"], q["position"]): q for q in m["raw"]}
    probes = {}
    for r in d["raw"]:
        probes.setdefault((r["block"], r["input"], r["position"]), {}).setdefault("v1" if r["direction"] == "v1" else "rand", []).append(r)
    per = []
    for key, pr in probes.items():
        if not 1 <= key[0] <= 10: continue
        v = pr["v1"][0]; rs = pr["rand"]; q = mix[key]; g_r = q["gain"]["rand"][0]; s1 = v["sigma1"]
        rec = dict(block=key[0], input=key[1], probe=f"{key[1]}_{key[2]}")
        quad_v, quad_r = v["quad"], float(np.mean([x["quad"] for x in rs])); rec["local_sq"] = (s1 / g_r) ** 2
        for g, a, al in GAUGES:
            amp_v, amp_r = v[a], float(np.mean([x[a] for x in rs]))
            rec[f"amp_{g}"] = amp_v / amp_r; rec[f"after_{g}"] = (amp_v / s1 ** 2) / (amp_r / g_r ** 2); rec[f"align_{g}"] = (quad_v / amp_v) / (quad_r / amp_r)
        rec["combined"] = (quad_v / quad_r) / (s1 / g_r) ** 2
        rec["const_v1"] = v["V"] * v["g_mean"] ** 2 / v["amplification"]
        rec["const_rand"] = float(np.mean([x["V"] * x["g_mean"] ** 2 / x["amplification"] for x in rs]))
        per.append(rec)
    return name, per, list(range(1, 11))


def main():
    out = dict(controlled=[], pretrained=[], n_boot=N_BOOT)
    mp_dir = os.environ.get("MP_RESULTS", os.path.join(HERE, "matched_pretrained", "results"))
    for f in sorted(glob.glob(os.path.join(LC_P, "lc_*.json"))):
        d = json.load(open(f)); attach_assay_gain(d, mp_dir)
        per, blocks = rows_pretrained(d)
        out["pretrained"].append(dict(model=d["label"], n_probes=len(per), summary=summarise(per, blocks), ci95=boot(per, blocks, "input"), ci_unit="input",
                                      elapsed_min=d["elapsed_seconds"] / 60, gpu=d.get("prov_gpu"), version=d["version"]))
    matched_dir = os.environ.get("MATCHED", os.path.join(HERE, "matched", "results"))
    for f in sorted(glob.glob(os.path.join(LC_C, "linresp_*.json"))):
        d = json.load(open(f))
        if d.get("quick"): continue
        name, per, blocks = rows_controlled(d, matched_dir)
        out["controlled"].append(dict(model=name, n_probes=len(per), summary=summarise(per, blocks), ci95=boot(per, blocks, "probe"), ci_unit="probe",
                                      elapsed_min=d["elapsed_seconds"] / 60, gpu=d.get("prov_gpu"), version=d["version"]))
    json.dump(out, open(os.path.join(HERE, "linresp_centred_stats.json"), "w"), indent=1)
    for pop in ("controlled", "pretrained"):
        if not out[pop]: continue
        print(f"\n== {pop}: v1/random factors, gm over probes then blocks [95%]; after x align = downstream factor in every gauge")
        print(f"{'model':15s} {'local2':>6s} | {'raw after':>17s} {'align':>17s} | {'centred after':>17s} {'align':>17s} | {'p-centred after':>17s} {'align':>17s} | {'down':>5s} | const v1 / rand")
        for r in out[pop]:
            s, c = r["summary"], r["ci95"]
            f_ = lambda k: f"{s[k]:5.2f} [{c[k][0]:5.2f},{c[k][1]:5.2f}]"
            print(f"{r['model']:15s} {s['local_sq']:6.2f} | {f_('after_raw'):>17s} {f_('align_raw'):>17s} | {f_('after_centred'):>17s} {f_('align_centred'):>17s} | {f_('after_pcentred'):>17s} {f_('align_pcentred'):>17s} | "
                  f"{s['combined']:5.2f} | {s['const_share_v1']:.3f} / {s['const_share_rand']:.3f}")


if __name__ == "__main__":
    main()
