"""Census v4: merge the 45 per-layer branch-rotation files (31 from the April-panel and May
runs that hooked branches correctly and read out a task loss, 14 re-run on 20 Sept 2026 with
branch hooks and task readouts on natural data) with the survey geometry (canonical estimator,
two-arm rule) and compute the census statistics.

Run: python census_merge_v4.py -> /mnt/user-data/outputs/ncs/census_v4/census_v4.json, .csv, stats.json
"""
import json, glob, os, csv, sys
import numpy as np
from scipy import stats
from make_figs import R_thirds

R = "/mnt/user-data/outputs/ncs"
PERLAYER = "/tmp/claude-0/-home-claude/5c80d2c8-c253-5f39-a63d-00ab0becc130/scratchpad/hg/census/merged_v4"
SURVEY = f"{R}/survey_v2/incontext-natural-block-float32"
OUT = f"{R}/census_v4"
DECODER_ARCHES = ("pythia", "gpt2", "qwen", "phi", "llama", "mistral", "gemma2")
FAMILY_OF_ARCH = {"pythia": "AR decoder", "gpt2": "AR decoder", "qwen": "AR decoder", "phi": "AR decoder", "llama": "AR decoder",
                  "mistral": "AR decoder", "gemma2": "AR decoder", "mamba": "SSM", "convnext": "ConvNet", "mlp_mixer": "MLP-Mixer",
                  "mixer": "MLP-Mixer", "vit": "ViT", "dinov2": "DINOv2", "bert": "Masked encoder", "roberta": "Masked encoder",
                  "t5": "T5 encoder", "whisper": "Audio encoder", "wav2vec2": "Audio encoder"}


def thirds(vals):
    n = len(vals); t = n // 3; r = n % 3
    return float(np.mean(vals[:t])), float(np.mean(vals[t:2 * t + r])), float(np.mean(vals[2 * t + r:]))


VOCAB = {"pythia": 50304, "gpt2": 50257, "qwen": 151936, "phi": 51200, "llama": 128256, "mistral": 32000, "gemma2": 256000,
         "mamba": 50280, "bert": 30522, "roberta": 50265, "t5": 32128, "whisper": 51865, "wav2vec2": 32}


def sensitivity(dL, baseline=None, arch=None, readout_is_ce=True):
    L = len(dL)
    e5, w5, l5 = thirds(dL)
    ei, wi, li = thirds(dL[1:L - 1])
    body_mean = float(np.mean(dL[1:]))
    valid = body_mean >= -0.01
    # criterion 3 (added 20 Sept 2026): a cross-entropy readout is interpretable only below the uniform-prediction
    # ceiling ln V; Flan-T5-large's span-corruption loss (13.6 nats against ln 32128 = 10.4) fails it
    ceiling_ok = True
    if readout_is_ce and baseline is not None and arch in VOCAB:
        ceiling_ok = baseline <= np.log(VOCAB[arch])
    valid = valid and ceiling_ok
    F5 = valid and w5 > 0.01 and w5 > 0.1 * max(e5, l5)
    Fi = valid and wi > 0.01 and wi > 0.1 * max(ei, li)
    return dict(dL_block0=float(dL[0]), dL_last=float(dL[-1]), body_mean=body_mean, valid_readout=bool(valid), ceiling_ok=bool(ceiling_ok),
                dL_early_v5=e5, dL_waist_v5=w5, dL_late_v5=l5, F_waist_v5=bool(F5),
                dL_early_int=ei, dL_waist_int=wi, dL_late_int=li, F_waist_int=bool(Fi),
                waist_over_edge_int=float(wi / max(ei, li)) if max(ei, li) > 0 else float("nan"))


def main():
    os.makedirs(OUT, exist_ok=True)
    survey = {}
    for f in glob.glob(f"{SURVEY}/sigma1_*.json"):
        d = json.load(open(f)); survey[d["hf_id"]] = d
    rows = []
    for f in sorted(glob.glob(f"{PERLAYER}/perlayer_*.json")):
        d = json.load(open(f))
        s = survey[d["hf_id"]]
        Rv, EM, LM, _ = R_thirds(s["sigma1_profile"])
        readout = d.get("readout", "next-token cross-entropy, WikiText-103 validation")
        sens = sensitivity(np.array(d["delta_L_profile"], float), d.get("baseline"), d["arch"], readout_is_ce=("KL" not in readout))
        row = dict(hf_id=d["hf_id"], label=s["label"], arch=d["arch"], family=s["family"], n_layers=d["n_layers"], params_M=d.get("params_M"),
                   readout=d.get("readout", "next-token cross-entropy, WikiText-103 validation"), protocol=d.get("protocol", "may-2026"),
                   baseline=d.get("baseline"), R_ex0=Rv, early_over_mid=EM, late_over_mid=LM, two_arm=bool(Rv < 0.80 and EM > 1 and LM > 1),
                   pooled=bool(Rv < 0.80 and LM > 1), source_file=os.path.basename(f), **sens)
        if "kl_profile" in d:
            kl = np.array(d["kl_profile"], float); ks = sensitivity(kl, readout_is_ce=False)
            row.update(kl_waist_int=ks["dL_waist_int"], kl_waist_over_edge_int=ks["waist_over_edge_int"], kl_valid=ks["valid_readout"])
        G, F = row["two_arm"], row["F_waist_int"]
        row["cell_int"] = "excluded" if not row["valid_readout"] else ("geometry+sensitivity" if (G and F) else "geometry only" if G else "sensitivity only" if F else "neither")
        rows.append(row)
    rows.sort(key=lambda r: (["AR decoder", "SSM", "ConvNet (vision)", "MLP-Mixer (vision)", "Vision supervised", "Vision SSL", "Masked encoder",
                              "Encoder-decoder", "Audio encoder"].index(r["family"]), r["params_M"] or 0))
    json.dump(rows, open(f"{OUT}/census_v4.json", "w"), indent=1)
    with open(f"{OUT}/census_v4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    valid = [r for r in rows if r["valid_readout"]]
    excl = [r for r in rows if not r["valid_readout"]]
    st = {}
    st["n_models"] = len(rows); st["n_valid"] = len(valid); st["excluded"] = [(r["label"], round(r["body_mean"], 3), round(r["waist_over_edge_int"], 3), "ceiling" if not r["ceiling_ok"] else "negative interior") for r in excl]
    st["n_sensitive"] = sum(r["F_waist_int"] for r in valid)
    st["hg_positive_valid"] = sum(r["two_arm"] for r in valid)
    st["hg_pos_sensitive"] = sum(r["two_arm"] and r["F_waist_int"] for r in valid)
    st["hg_neg_sensitive"] = sum((not r["two_arm"]) and r["F_waist_int"] for r in valid)
    st["hg_neg_total"] = sum(not r["two_arm"] for r in valid)
    x = np.array([r["R_ex0"] for r in valid]); y = np.array([r["waist_over_edge_int"] for r in valid])
    rho, p = stats.spearmanr(x, y); st["spearman_all"] = (float(rho), float(p))
    rng = np.random.default_rng(0); bs = []
    for _ in range(5000):
        idx = rng.integers(0, len(valid), len(valid)); bs.append(stats.spearmanr(x[idx], y[idx])[0])
    st["spearman_all_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    dec = [r for r in valid if r["arch"] in DECODER_ARCHES]
    st["n_decoders"] = len(dec)
    st["spearman_decoders"] = tuple(map(float, stats.spearmanr([r["R_ex0"] for r in dec], [r["waist_over_edge_int"] for r in dec])))
    xd = np.array([r["R_ex0"] for r in dec]); yd = np.array([r["waist_over_edge_int"] for r in dec]); bs = []
    for _ in range(5000):
        idx = rng.integers(0, len(dec), len(dec)); bs.append(stats.spearmanr(xd[idx], yd[idx])[0])
    st["spearman_decoders_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    fams = sorted(set(FAMILY_OF_ARCH[r["arch"]] for r in valid))
    loo = {}
    for fam in fams:
        sub = [r for r in valid if FAMILY_OF_ARCH[r["arch"]] != fam]
        loo[fam] = float(stats.spearmanr([r["R_ex0"] for r in sub], [r["waist_over_edge_int"] for r in sub])[0])
    st["leave_one_family_out"] = loo; st["max_abs_loo"] = float(max(abs(v) for v in loo.values()))
    within = {}
    for fam in fams:
        sub = [r for r in valid if FAMILY_OF_ARCH[r["arch"]] == fam]
        if len(sub) >= 4:
            rr, pp = stats.spearmanr([r["R_ex0"] for r in sub], [r["waist_over_edge_int"] for r in sub]); within[fam] = (len(sub), float(rr), float(pp))
    st["within_family"] = within
    # KL-readout version over the models that have it (the 14 re-run plus the four ConvNeXts whose primary readout is KL)
    klrows = [r for r in valid if "kl_waist_over_edge_int" in r]
    st["kl_readout_n"] = len(klrows)
    st["cells"] = {c: sum(r["cell_int"] == c for r in rows) for c in ("geometry+sensitivity", "geometry only", "sensitivity only", "neither", "excluded")}
    st["insensitive_valid"] = [(r["label"], round(r["dL_waist_int"], 3), round(r["waist_over_edge_int"], 3)) for r in valid if not r["F_waist_int"]]
    for r in rows:
        if r["hf_id"] in ("EleutherAI/pythia-12b", "microsoft/phi-2"):
            st[r["label"]] = dict(R=round(r["R_ex0"], 3), waist=round(r["dL_waist_int"], 3), we=round(r["waist_over_edge_int"], 3))
    json.dump(st, open(f"{OUT}/census_v4_stats.json", "w"), indent=1)
    for k, v in st.items():
        print(f"{k}: {v}")
    print(f"\n{'label':26s} {'fam':16s} {'R':>5s} {'2arm':>4s} {'waist':>7s} {'w/e':>6s} {'valid':>5s} {'sens':>4s} {'cell':22s} {'proto':18s}")
    for r in rows:
        print(f"{r['label']:26s} {r['family'][:16]:16s} {r['R_ex0']:5.2f} {str(r['two_arm'])[0]:>4s} {r['dL_waist_int']:7.3f} {r['waist_over_edge_int']:6.2f} {str(r['valid_readout'])[0]:>5s} {str(r['F_waist_int'])[0]:>4s} {r['cell_int']:22s} {r['protocol']:18s}")


if __name__ == "__main__":
    main()
