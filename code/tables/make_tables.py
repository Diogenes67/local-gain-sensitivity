"""Extended Data Tables 1-3 as markdown, from the survey JSONs (ED Table 1), census_v4 (ED Table 2)
and the experiment record (ED Table 3). build.py inserts each after its legend.

Run: python make_tables.py -> tables/ed_table1.md, ed_table2.md, ed_table3.md, ed_table1.csv, ed_table2.csv
"""
import json, glob, os, csv
import numpy as np
from make_figs import R_thirds

R = "/mnt/user-data/outputs/ncs"
SURVEY = f"{R}/survey_v2/incontext-natural-block-float32"
CENSUS = f"{R}/census_v4/census_v4.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT, exist_ok=True)
SEED, NBOOT = 42, 2000

FAMILY_ORDER = ["AR decoder", "SSM", "ConvNet (vision)", "MLP-Mixer (vision)", "Vision supervised", "Vision SSL",
                "Masked encoder", "Encoder-decoder", "Audio encoder"]
FAMILY_LABEL = {"AR decoder": "AR decoder", "SSM": "State-space", "ConvNet (vision)": "ConvNet", "MLP-Mixer (vision)": "MLP-Mixer",
                "Vision supervised": "ViT (supervised)", "Vision SSL": "ViT (self-supervised)", "Masked encoder": "Masked encoder",
                "Encoder-decoder": "T5 encoder", "Audio encoder": "Audio encoder"}


def clustered_ci(samples, n_inputs, n_positions, nboot=NBOOT, seed=SEED):
    S = np.asarray(samples, float)
    L, n = S.shape
    assert n == n_inputs * n_positions
    Sc = S.reshape(L, n_inputs, n_positions)
    rng = np.random.default_rng(seed)
    Rs = []
    for _ in range(nboot):
        idx = rng.integers(0, n_inputs, n_inputs)
        Rs.append(R_thirds(Sc[:, idx, :].reshape(L, -1).mean(1))[0])
    return np.percentile(Rs, [2.5, 97.5])


def survey_rows():
    rows = []
    for f in sorted(glob.glob(f"{SURVEY}/sigma1_*.json")):
        d = json.load(open(f))
        Rv, EM, LM, _ = R_thirds(d["sigma1_profile"])
        assert abs(Rv - d["R_ex0_thirds"]) < 1e-9
        lo, hi = clustered_ci(d["sigma1_samples"], d["n_inputs"], d["n_positions"])
        rows.append(dict(hf_id=d["hf_id"], label=d["label"], family=d["family"], L=d["n_layers"], params=d["params_M"],
                         R=Rv, lo=lo, hi=hi, EM=EM, LM=LM, two_arm=bool(Rv < 0.80 and EM > 1 and LM > 1),
                         pooled=bool(Rv < 0.80 and LM > 1), C=d["contrast_C_ex0"], a=d["quad_a_ex0"],
                         block0=d["sigma1_profile"][0]))
    rows.sort(key=lambda r: (FAMILY_ORDER.index(r["family"]), r["params"]))
    return rows


def ed_table1(rows):
    # NBH: non-breaking hyphen inside the table so that a confidence interval is never split across lines
    NBH = "\u2011"
    hdr = ("| Model | Family | Blocks | Params (M) | R~ex0~ | 95% CI | Early/middle | Late/middle | Two-arm | Pooled-edge |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        body += (f"| {r['label']} | {FAMILY_LABEL[r['family']]} | {r['L']} | {r['params']:,} | {r['R']:.2f} | {r['lo']:.2f}{NBH}{r['hi']:.2f} | "
                 f"{r['EM']:.2f} | {r['LM']:.2f} | {'positive' if r['two_arm'] else '–'} | {'positive' if r['pooled'] else '–'} |\n")
    n2, np_ = sum(r["two_arm"] for r in rows), sum(r["pooled"] for r in rows)
    strad = [r["label"] for r in rows if r["lo"] < 0.80 < r["hi"]]
    hw = np.median([(r["hi"] - r["lo"]) / 2 for r in rows])
    foot = (f"\nTwo-arm rule: R~ex0~ < 0.80 with early/middle > 1 and late/middle > 1 ({n2} of 45 positive). Pooled-edge rule: "
            f"R~ex0~ < 0.80 with late/middle > 1 ({np_} of 45). 95% CI: input-clustered percentile bootstrap, the five inputs resampled "
            f"with replacement, each carrying its eight positions, {NBOOT:,} resamples (median half-width {hw:.3f}); {len(strad)} intervals "
            f"straddle 0.80 ({', '.join(strad)}). Block 0 is excluded from every column. The spectral contrast C and the quadratic coefficient a of the "
            f"earlier classification are in ed_table1.csv and Supplementary Note 10.\n")
    open(os.path.join(OUT, "ed_table1.md"), "w").write(hdr + body + foot)
    with open(os.path.join(OUT, "ed_table1.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return n2, np_, strad, hw


READOUT_SHORT = {"next-token": "next-token CE", "masked-LM": "masked-token CE", "span-corruption": "span-corruption CE", "KL(clean": "KL, ImageNet head",
                 "CTC": "CTC loss", "teacher-forced": "decoder CE"}


def readout_short(r):
    for k, v in READOUT_SHORT.items():
        if r.startswith(k) or k in r[:40]:
            return v
    return r[:24]


def ed_table2(rows):
    cen = json.load(open(f"{R}/census_v4/census_v4.json"))
    hdr = ("| Model | Family | Blocks | Readout | Clean loss | R~ex0~ | Two-arm | ΔL early | ΔL middle | ΔL late | Middle/edge | ΔL block 0 | ΔL last block | Interior-sensitive | Cell | Excluded |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in cen:
        crit = []
        if not r["valid_readout"]:
            if r["body_mean"] < -0.01: crit.append("1")
            if not (r["waist_over_edge_int"] == r["waist_over_edge_int"]): crit.append("2")
            if not r["ceiling_ok"]: crit.append("3")
        G = r["two_arm"]
        if r["valid_readout"]:
            F, Fv = r["F_waist_int"], r["F_waist_v5"]
            cell_int = "both" if (G and F) else "geometry" if G else "sensitivity" if F else "neither"
            cell_v5 = "both" if (G and Fv) else "geometry" if G else "sensitivity" if Fv else "neither"
        else:
            cell_int = cell_v5 = "excluded"
        we = r["waist_over_edge_int"]; we_s = f"{we:.2f}" if we == we else "undefined"
        base = "0 (KL)" if "KL" in r["readout"][:12] else f"{r['baseline']:.2f}"
        body += (f"| {r['label']} | {FAMILY_LABEL[r['family']]} | {r['n_layers']} | {readout_short(r['readout'])} | {base} | {r['R_ex0']:.2f} | "
                 f"{'positive' if G else '–'} | {r['dL_early_int']:.3f} | {r['dL_waist_int']:.3f} | {r['dL_late_int']:.3f} | {we_s} | "
                 f"{r['dL_block0']:.2f} | {r['dL_last']:.2f} | {'yes' if (r['valid_readout'] and r['F_waist_int']) else ('no' if r['valid_readout'] else '–')} | "
                 f"{cell_int} | {', '.join(crit) if crit else '–'} |\n")
    foot = ("\nΔL, increase of the model's own loss in nats (next-token, masked-token, span-corruption, CTC or teacher-forced decoder "
            "cross-entropy) or the KL divergence from the clean logits over the ImageNet-1k head (vision models) when the branch outputs of "
            "a block are rotated by a Haar-random orthogonal matrix, dose 1, one block at a time; regional means over thirds of blocks 1 to "
            "L − 2, with block 0 and the last block reported separately. Middle/edge, middle mean divided by the larger edge-third mean. "
            "Interior-sensitive: middle mean > 0.01 nats and > 10% of the larger edge-third mean. Cell crosses the two-arm geometry flag with "
            "the interior-sensitivity flag (both, geometry only, sensitivity only, neither); the assignment under the earlier all-blocks "
            "sensitivity rule, which took thirds over every block including block 0, is in ed_table2.csv. Excluded: 1, mean interior ΔL below −0.01; 2, middle/edge undefined because no edge third "
            "is positive; 3, clean cross-entropy above the uniform-prediction baseline ln V (Flan-T5-large, 13.6 nats against 10.4). The 14 "
            "non-decoder models were re-run on 20 September 2026 with branch-level hooks and the readouts listed (Methods), the two audio models again on 21 September on utterances of at most 10 s; the other 31 "
            "files are from the April and May runs, whose hooks and readouts were as described.\n")
    open(os.path.join(OUT, "ed_table2.md"), "w").write(hdr + body + foot)
    with open(os.path.join(OUT, "ed_table2.csv"), "w", newline="") as f:
        keys = [k for k in cen[0].keys()]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(cen)
    return cen


ED3 = [
 # (result / display item, operator, Jacobian domain and probe, intervention, precision, models or checkpoints, script, commensurable)
 ("Estimator validation (Fig. 1b; Methods)", "σ~1~ by J^T^J power iteration against exact SVD of the materialised Jacobian", "token-local, natural context, natural inputs", "–", "float32", "GPT-2 124M, Pythia-160M, Pythia-410M, Phi-2, ConvNeXt-tiny/base, Mamba-130M (210 pairs)", "a0_sigma1_validation_v2.py (transformers); validate_convnext_mamba.ipynb with survey_sigma1_v2 v2.3 (ConvNeXt, Mamba)", "yes"),
 ("Random-direction probing (Fig. 1c)", "maximum gain over 64 random unit directions", "token-local, as above", "–", "float32", "the same 210 pairs", "a0_sigma1_validation_v2.py", "yes"),
 ("Comparison with Aubry et al. (Methods)", "their materialised Jacobian and full SVD against our power iteration", "token-local, last position, natural inputs", "–", "float32 and bfloat16", "GPT-2 124M, Pythia-410M", "aubry_compare.py; github.com/sugolov/coupling", "yes"),
 ("Operator dependence, ρ against σ~1~ (ED Fig. 2a,b)", "spectral radius by iteration on J", "token-local, natural context", "–", "float32", "eight models", "survey_sigma1_v2.py --estimator rho", "no (ρ, reported as a comparison)"),
 ("Probe-context dependence (ED Fig. 2c)", "σ~1~", "isolated token; whole-sequence Jacobian; random token ids", "–", "float32", "eight models plus five decoders", "survey_sigma1_v2.py --context isolated|fullseq, --inputs random", "no (alternative protocols)"),
 ("Branch against block Jacobian (ED Fig. 2d)", "σ~1~ of J − I", "token-local, natural context", "–", "float32", "seven decoders", "survey_sigma1_v2.py --branch", "yes"),
 ("Precision dependence (ED Fig. 2e–h)", "σ~1~ with finite-difference Jv", "token-local, natural context", "–", "float16, bfloat16 (ε ∈ {10^−3^, 10^−2^, 10^−1^})", "seven decoders of 124M–1.5B; Pythia-2.8B, 6.9B, 12B", "survey_sigma1_v2.py --dtype float16|bfloat16", "no (reduced precision)"),
 ("Survey of 45 pretrained models (Fig. 5a–e; ED Table 1)", "σ~1~", "token-local, natural context, natural inputs, five inputs × eight positions, four restarts", "–", "float32", "45 models, Hugging Face Hub", "survey_sigma1_v2.py (incontext-natural-block-float32)", "canonical"),
 ("Metric comparison, CKA, effective rank, ‖J‖~F~ (SN 1–2; ED Fig. 3)", "CKA, effective rank, Hutchinson ‖J‖~F~ alongside the survey σ~1~", "token-local, natural context", "–", "float32", "Pythia-70M, 160M, 410M, 1.4B, 6.9B", "metrics_sn1_sn2.py", "yes"),
 ("Destroyed-data experiment (Fig. 2a–d)", "σ~1~", "token-local, the condition's own mask (causal; none for the bidirectional models), canonical protocol", "branch rotation (doses 0.25, 0.5, 1), identity skip, per block and by region", "float32", "26 checkpoints, d1_v21 (Drive)", "d1_v2.py v2.1 (Colab)", "canonical"),
 ("Condition switching (Fig. 2d)", "σ~1~", "as above", "as above", "float32", "six checkpoints, d1_v21", "d1_v2.py v2.1", "canonical"),
 ("Mask × loss design (ED Fig. 9a; ED Fig. 5c)", "σ~1~", "token-local, the condition's own mask at profile time", "–", "float32", "20 runs, factorial_v2", "factorial_v2.py, profiling through survey_sigma1_v2's estimator", "canonical"),
 ("Mask × loss design, March run (ED Fig. 5c)", "σ~1~ by the training script's single-position J^T^J iteration", "token-local, one position", "–", "float32", "12 runs, leaking prefix mask", "b1_objective_factorial.py (March 2026)", "no (superseded; shown for the record)"),
 ("Width × depth grid (ED Fig. 9b)", "σ~1~ by the training script's J^T^J iteration", "token-local, position T/2, causal mask", "–", "float32 (TF32 off)", "31 runs at 3 × 10^−4^ plus two at 10^−4^, grid_v2", "b1_grid_followup.py (Sept 2026 re-run)", "yes (single position)"),
 ("Emergence during training (ED Fig. 4)", "σ~1~ by the training script's J^T^J iteration", "token-local, position T/2", "–", "mixed-precision training, float32 profiling", "three 1.3B seeds, 354M controls, Pythia-410M public checkpoints", "b1_1b_duration_seeds_v3.py, b1_1b_seed42_v4.py (1.3B); b1_duration_ar20k.py, b1_duration_mlm.py (354M); s2_pythia_developmental.py (checkpoints; canonical estimator)", "yes (single position)"),
 ("Routing topology (ED Fig. 5a)", "σ~1~", "token-local, causal mask, canonical protocol", "branch rotation, identity skip, per block and by region; early exit", "float32", "18 checkpoints, reprofile_v2/topology (Drive)", "reprofile_v2.py --exp topology (Sept 2026 retrain)", "canonical"),
 ("Skip-coefficient sweep (ED Fig. 5b)", "σ~1~", "as above", "as above", "float32", "15 checkpoints, reprofile_v2/alpha", "reprofile_v2.py --exp alpha", "canonical"),
 ("Dependency-order sweep (Fig. 3)", "σ~1~", "as above, on held-out sequences of the model's own source and on WikiText-103", "branch rotation, identity skip, per block and by region; early exit", "float32", "30 checkpoints, reprofile_v2/kgram", "reprofile_v2.py --exp kgram", "canonical"),
 ("Entropy-perturbation control (SN 6; not part of the evidential argument)", "–", "–", "branch rotation of blocks 4–7", "float32", "60 runs (March 2026; checkpoints and per-model files not retained)", "d3j_confidence_decouple.py, d3j_completion.py (cell means transcribed in SN 6 from the notebook log; not reproducible from retained files)", "not applicable"),
 ("Window additivity (SN 16)", "–", "–", "branch rotation of contiguous windows of 2–4 blocks", "float32", "k-gram models (March 2026)", "d3f_windowed_scrambling.py", "not applicable"),
 ("One-hop synthetic sources (Methods)", "σ~1~", "token-local, causal mask, canonical protocol", "identity skip per block; early exit (whole-block rotation by region measured and not used)", "float32", "60 runs, synthetic_context_v2", "synthetic_context.py (v2, Sept 2026)", "canonical"),
 ("Downstream-only adaptation (ED Fig. 8b; SN 13)", "–", "–", "branch rotation of block 5, then 500 fine-tuning steps of the downstream blocks", "float32", "k = 8 models (March 2026)", "d3l_alignment_adaptation.py (d3l_aggregate.json)", "not applicable"),
 ("Singular-direction perturbation (ED Fig. 8c)", "top five right singular vectors of the block-5 token-local Jacobian by randomised SVD", "token-local", "input perturbation ±ε v, ε = 0.1 × mean activation norm, against norm-matched random directions", "float32", "nine k-gram models (March 2026)", "exp4a_singular_direction_perturbation.py (nine result files)", "not compared with the survey"),
 ("Matched token-local perturbation assay (Fig. 4; ED Fig. 7; SN 17)", "σ~1~ and v~1~ by the survey's J^T^J iteration on the block input at one position", "token-local, causal mask (the condition's own mask), four positions × five held-out inputs", "input displacement ±ε‖h‖ along v~1~ and along four random unit directions, ε ∈ {0.01, 0.03, 0.1, 0.3, 1}; readouts gain, KL at the position, downstream KL, own loss", "float32", "50 checkpoints: d1 v2.1 (20) and k-gram (30), 20 Sept 2026", "matched_perturbation.py; matched_analysis.py; regression_check.py (pooled regressions, 22 Sept 2026)", "yes (same operator and probe)"),
 ("Matched assay, sampling check (Methods)", "σ~1~ and v~1~ as above", "token-local, six positions × 20 held-out inputs, eight random directions per probe", "as above", "float32", "ten checkpoints (real s0–1, shuffled s0–1, k = 1 s0–1, k = 8 s0–1; 24-layer real s0, shuffled s0), reloaded from float16", "matched_sampling_colab.ipynb (21 Sept 2026); sampling_analysis.py", "yes"),
 ("Downstream linearised response, pilot (Methods)", "σ~1~ and v~1~ as above; g = Au by one forward-mode product to the position-t logits", "token-local, five inputs × four positions, v~1~ and four random directions", "displacement ε‖h‖u, ε ∈ {0.01, 0.03, 0.1}, same-position KL against ½ε^2^‖h‖^2^ g^T^Fg", "float32", "eight checkpoints (real s0–1, shuffled s0–1, k = 1 s0–1, k = 8 s0–1)", "linearised_response.py (21 Sept 2026); linresp_analysis.py; linresp_afterblock.py (after-block split, 22 Sept 2026)", "yes"),
 ("Entropy-matched Markov family (Fig. 3d–f)", "σ~1~ (canonical)", "token-local, causal mask, held-out sequences of the model's own source", "branch rotation, identity skip, per block and by region; early exit", "float32", "30 checkpoints, markov_v1_10k (orders 1–6 × 5 seeds), 20 Sept 2026", "markov_v1.py, markov_source.py", "canonical"),
 ("Scale point in depth (Fig. 2e–g)", "σ~1~ (canonical) and the matched assay", "token-local, causal mask, canonical protocol", "branch rotation by third and per block, identity skip, matched displacement along v~1~", "float32", "six 24-layer checkpoints (real and shuffled text, three seeds each), 20 Sept 2026", "scale_v1.py (d1_v2.py + matched_perturbation.py)", "canonical"),
 ("Residual-gain stress (ED Fig. 8d)", "–", "–", "residual branch rescaled by α ∈ {0.75, 1, 1.25} by region", "float32", "k-gram models (March 2026)", "d3r_residual_gain_stress.py (d3r_aggregate.json)", "not applicable"),
 ("Dyck-2 probes (ED Fig. 8a; SN 4)", "–", "–", "per-block branch rotation; linear probes; participation ratio", "float32", "four Dyck-2 models (March 2026)", "exp2a_bracket_probes.py (four result files)", "not applicable"),
 ("Pretrained census (Fig. 5f; ED Table 2; ED Fig. 6)", "survey σ~1~ (canonical)", "token-local, natural context", "branch rotation, dose 1, one block at a time; readout the model's own loss or KL over the classification head; Gemma-2-2B with a BOS token", "float32", "45 models; 31 per-layer files from the April and May runs (decoders, Mamba, ConvNeXt) and 14 from the 20 Sept 2026 re-run (masked encoders, T5, ViT, DINOv2, MLP-Mixer, audio)", "scramble_census_april_panel.py, delta_l_census_reclass.py (decoders only), scramble_census_v4.py; merged by census_merge_v4.py", "canonical"),
 ("Intervention comparison (Methods)", "–", "–", "whole-block rotation, branch rotation, identity skip, plumbing control", "float32", "eight models", "intervention_compare.py", "not applicable"),
 ("Layer and unstructured pruning (SN 7, 9)", "survey σ~1~ for selection", "token-local", "layer removal by Block Influence, cosine distance or σ~1~; Wanda, magnitude and σ~1~-guided sparsity", "float32", "Pythia-12B; Pythia and Qwen models", "s1_pruning_protocol.py, d3n_pruning_baselines.py, m6_wanda_pruning.py, e10_layer_skip_importance.py", "yes (selection only)"),
]


DESIGN = {  # seeds and inferential unit; evaluation data; block indices; magnitude
 "Survey of 45": "one run per model; five inputs × eight positions (40 probes per block), four restarts; R over thirds of blocks 1 to L − 1; CI over inputs",
 "Destroyed-data": "five training seeds per condition (shuffle seed tied to the training seed); 800 held-out sequences (intervention set), 256 (validation); profile thirds 1–3, 4–8, 9–11; intervention 0–3, 4–7, 8–11; one Haar draw per block and seed, dose 1 (0.25, 0.5 also); 500-resample bootstrap over batches within a run; unit for condition contrasts, the run",
 "Condition switching": "three seeds per switching condition; otherwise as above",
 "Mask × loss design (ED Fig. 9a": "five seeds per cell (20 runs); profile thirds as above; unit, the run",
 "Width × depth grid": "1–5 seeds per cell (31 runs at 3 × 10^−4^, two at 10^−4^); thirds of blocks 1 to L − 1 per depth; unit, the run",
 "Routing topology": "three seeds per topology; regions as the destroyed-data experiment; unit, the run",
 "Skip-coefficient": "three seeds per α (15 runs); as above",
 "Dependency-order sweep": "five seeds per k (source seed tied to the training seed); 800 held-out sequences of the model's own source; regions as above; one Haar draw per block and seed; primary test on the six group means (exact permutation over 720 orderings, prespecified)",
 "One-hop synthetic": "five seeds per lag (60 runs); as above",
 "Downstream linearised": "eight checkpoints; 20 probes per block; amplification ||g||^2 and alignment g^T F g/||g||^2 per probe; unit, the model",
 "Matched assay, sampling": "ten checkpoints; 20 inputs × 6 positions × (v~1~ + 8 random); bootstrap over inputs (2,000 draws); unit, the model",
 "Matched token-local": "the 50 checkpoints above, reloaded from float16; five held-out inputs × four positions (t = 16, 48, 80, 112) per block; v~1~ and four random unit directions per probe; ε ∈ {0.01, 0.03, 0.1, 0.3, 1} × ‖h~l,t~‖, signs averaged; blocks 1–10 for within-model correlations, 4–7 for the middle-block mean; unit, the model",
 "Entropy-matched Markov": "five seeds per order (source drawn per seed); 800 held-out sequences, positions t ≥ 8; regions and draws as the k-gram sweep; ordering statistics over all 30 and over the 25 within 0.025 nats of the floor (split made after inspection)",
 "Scale point": "three seeds per condition; regions early 0–7, middle 8–15, late 16–23; profile thirds 1–7, 8–15, 16–22; matched assay as above over blocks 1–22",
 "Pretrained census": "one run per model; 30 batches of 16 (four utterances for audio); one Haar draw per block; sensitivity thirds of blocks 1 to L − 2, block 0 and the last block apart; 200-resample bootstrap over batches; unit, the model",
}


def design_for(title):
    for k, v in DESIGN.items():
        if title.startswith(k): return v
    return "–"


def ed_table3():
    hdr = ("| Result (display item) | Operator estimated | Jacobian domain and probe | Intervention | Precision | Models or checkpoints | Script | Commensurable with the survey | Seeds, evaluation data, block indices, magnitude, inferential unit |\n"
           "|---|---|---|---|---|---|---|---|---|\n")
    body = "".join("| " + " | ".join(r) + " | " + design_for(r[0]) + " |\n" for r in ED3)
    foot = ("\nCanonical: σ~1~ of the token-local block Jacobian by J^T^J power iteration, token in its natural context, natural inputs, "
            "five inputs × eight positions, four restarts, float32, with the model's own attention mask. Rows marked 'yes (single position)' "
            "use the same operator at one position and are compared with the survey on that basis. Supplementary Note 17 gives the provenance of "
            "every row: the run each result comes from, the runs it replaces, and whether it is regenerable from the deposited files. The March 2026 routing, skip-coefficient "
            "and k-gram profiles, taken by forward-mode iteration on J over the whole layer input at random token ids, are superseded by the "
            "September re-runs listed here and appear nowhere in the paper. Checkpoints of every September run are on persistent storage; "
            "the March mechanism-experiment checkpoints (SN 4, 6, 13, 16; ED Fig. 8) were not retained, and their results do not depend on the estimator. "
            "The last column records the design of the experiments the argument rests on: training seeds and the inferential unit, the evaluation data, "
            "the block indices of every region, the intervention magnitude and the number of random draws.\n")
    open(os.path.join(OUT, "ed_table3.md"), "w").write(hdr + body + foot)


if __name__ == "__main__":
    rows = survey_rows()
    n2, np_, strad, hw = ed_table1(rows)
    cen = ed_table2(rows)
    ed_table3()
    print(f"ED Table 1: {len(rows)} rows, two-arm {n2}, pooled {np_}, straddle {len(strad)}: {strad}, median half-width {hw:.3f}")
    valid = [r for r in cen if r["valid_readout"]]
    print(f"ED Table 2: {len(cen)} rows, valid {len(valid)}, sensitive {sum(r['F_waist_int'] for r in valid)}")
    print("written", OUT)
