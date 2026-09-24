"""E3: survey input resampling (30 inputs per model) against the five-input survey (23 Sept 2026).
For each of the 11 models: R_ex0 (middle-to-edge gain ratio) with the input-clustered CI from 30 inputs, early/middle and
late/middle, two-arm (interior valley) status; the five-input survey value and CI (ed_table1.csv); the spread of R_ex0 over
six disjoint five-input subsets of the 30. Then the census association (Spearman between R_ex0 and the waist-to-edge ratio
over the 44 valid models) recomputed with the 30-input values substituted for these 11 models, with a 5,000-resample CI."""
import json, glob, csv
import numpy as np
from scipy import stats

def thirds(p):
    p = np.asarray(p, float)[1:]; n = len(p); t = n // 3
    e, m, l = p[:t].mean(), p[t:n - t].mean(), p[n - t:].mean()
    return m / ((e + l) / 2), e / m, l / m

ed = {r["hf_id"]: r for r in csv.DictReader(open("ed_table1.csv"))}
out = {}
print(f"{'model':16s} {'R30':>6s} {'CI30':>14s} {'E/M':>5s} {'L/M':>5s} {'valley30':>8s} | {'R5':>5s} {'CI5':>12s} {'valley5':>7s} | 5-input subsets")
for f in sorted(glob.glob("results/sigma1_*.json")):
    d = json.load(open(f)); hf = d["hf_id"]; S = d["sigma1_samples"]; L = d["n_layers"]; npos = d["n_positions"]; nin = d["n_inputs"]
    R, em, lm = thirds(d["sigma1_profile"]); valley = bool(R < 0.80 and em > 1 and lm > 1)
    arr = np.array([np.array(s).reshape(nin, npos) for s in S])          # (L, inputs, positions)
    subs = []
    for k in range(nin // 5):
        idx = list(range(5 * k, 5 * k + 5)); subs.append(thirds(arr[:, idx, :].mean(axis=(1, 2)))[0])
    e = ed.get(hf)
    R5, lo5, hi5 = float(e["R"]), float(e["lo"]), float(e["hi"]); v5 = e["two_arm"] == "True"
    out[hf] = dict(label=d["label"], R30=R, ci30=d["R_ex0_thirds_ci95"], em=em, lm=lm, valley30=valley, R5=R5, ci5=[lo5, hi5], valley5=v5,
                   subsets5=subs, subset_range=[min(subs), max(subs)], outside_ci5=bool(R < lo5 or R > hi5))
    o = out[hf]
    print(f"{o['label']:16s} {R:6.3f} [{o['ci30'][0]:.3f},{o['ci30'][1]:.3f}] {em:5.2f} {lm:5.2f} {str(valley):>8s} | {R5:5.3f} [{lo5:.2f},{hi5:.2f}] {str(v5):>7s} | {min(subs):.3f}-{max(subs):.3f}" + ("  R30 outside 5-input CI" if o["outside_ci5"] else ""))

# census association with the 30-input values substituted
rows = [r for r in csv.DictReader(open("census_v4.csv")) if r["valid_readout"] == "True" and r["ceiling_ok"] == "True"]
def assoc(sub):
    x = np.array([out[r["hf_id"]]["R30"] if (sub and r["hf_id"] in out) else float(r["R_ex0"]) for r in rows])
    y = np.array([float(r["waist_over_edge_int"]) for r in rows])
    rho = stats.spearmanr(x, y)[0]; rng = np.random.default_rng(0); bs = []
    for _ in range(5000):
        i = rng.integers(0, len(x), len(x)); bs.append(stats.spearmanr(x[i], y[i])[0])
    return float(rho), np.nanpercentile(bs, [2.5, 97.5]).tolist(), len(x)
a0, a1 = assoc(False), assoc(True)
print(f"\ncensus association, five-input survey: rho {a0[0]:+.3f} CI {a0[1][0]:+.2f} to {a0[1][1]:+.2f} (n = {a0[2]})")
print(f"census association, 30-input values for the 11 models: rho {a1[0]:+.3f} CI {a1[1][0]:+.2f} to {a1[1][1]:+.2f}")
dec = [r for r in rows if r["family"] == "AR decoder"]
def assoc_dec(sub):
    x = np.array([out[r["hf_id"]]["R30"] if (sub and r["hf_id"] in out) else float(r["R_ex0"]) for r in dec]); y = np.array([float(r["waist_over_edge_int"]) for r in dec])
    return float(stats.spearmanr(x, y)[0]), len(x)
print("decoders only:", assoc_dec(False), assoc_dec(True))
json.dump(dict(models=out, census_5=a0, census_30=a1, decoders_5=assoc_dec(False), decoders_30=assoc_dec(True)), open("e3_stats.json", "w"), indent=1)
