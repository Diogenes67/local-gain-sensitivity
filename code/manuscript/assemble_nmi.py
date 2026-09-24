"""Assemble the NMI manuscript (23 Sept 2026): the rewritten main text (HOURGLASS/NMI_revision/main_text_v2.md),
the current Methods with the plan's section-3 edits, the new main legends and Table 1, and the Extended Data legends
with figure references renumbered. Writes nmi/main_nmi.md, nmi/methods_nmi.md, nmi/legends_nmi.md; build_nmi.py builds the docx.
"""
import re, os
D = os.path.dirname(os.path.abspath(__file__)); NCS = os.path.dirname(D)
MT = os.path.expanduser("~/mnt/HOURGLASS/NMI_revision/main_text_v2.md")

# ---- number -> key mapping of the current (NCS) build, by first appearance in main.md + methods.md
old_body = open(os.path.join(NCS, "main.md")).read() + "\n\n" + open(os.path.join(NCS, "methods.md")).read()
order = []
for m in re.finditer(r'\{\{([a-z0-9]+)\}\}', old_body):
    if m.group(1) not in order:
        order.append(m.group(1))
num2key = {i + 1: k for i, k in enumerate(order)}

def expand(s):
    out = []
    for part in s.split(","):
        if "–" in part:
            a, b = part.split("–"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

t = open(MT).read()
t = re.sub(r'^---\ntitle: (.*)\n---\n', r'# \1\n', t)
head, rest = t.split("## Main", 1)
main_part, legends_part = rest.split("## Figure legends", 1)
main_part = re.sub(r'\^([0-9][0-9,–]*)\^', lambda m: "".join("{{%s}}" % num2key[n] for n in expand(m.group(1))), main_part)
open(os.path.join(D, "main_nmi.md"), "w").write(head + "## Main" + main_part.rstrip() + "\n")

# ---- Methods with the plan's edits
mt = open(os.path.join(NCS, "methods.md")).read()
edits = [
    ("### Mid-to-edge ratio and classification", "### Middle-to-edge gain ratio and interior valley"),
    ("A model is hourglass-positive if R~ex0~ < 0.80 and both edge thirds exceed the middle third",
     "A profile has an interior valley if R~ex0~ < 0.80 and both edge thirds exceed the middle third"),
    ("Continuous R~ex0~ is primary and the classification secondary.", "The continuous ratio is primary and the valley classification secondary; in the main text R~ex0~ is called the middle-to-edge gain ratio."),
    ("Stability with more inputs was not checked and five clusters is a small number for a percentile bootstrap.",
     "Five clusters is a small number for a percentile bootstrap. The 11 models of Figs 2, 3 and 5 were profiled again with 30 inputs (survey_sigma1_v2 unchanged except for the number of inputs; e3_analysis.py): the ratio moved by at most 0.08 (median 0.02), no two-arm classification changed, and the 30-input value lay outside the five-input interval in three models (Pythia-1.4B, ConvNeXt-base and Mamba-130M), so five-input intervals understate the sampling spread. Six disjoint five-input subsets of the 30 inputs spread by up to 0.17 (ViT-base). With the 30-input values substituted, the census association is 0.04 (95% CI −0.26 to 0.34) and the decoder-only association is unchanged."),
    ("A model is interior-sensitive if the waist mean exceeds 0.01 nats and 10% of the larger edge-third mean. The earlier rule, which took thirds over all blocks including block 0, is reported alongside in Extended Data Table 2.",
     "Relative interior sensitivity is the waist mean divided by the larger edge-third mean (the waist-to-edge ratio)."),
    ("All eight models are interior-sensitive by the census criterion under all three interventions; the waist-to-edge ratio depends on the intervention",
     "The waist-to-edge ratio depends on the intervention"),
    ("the relative-dose result is reported in the main text, and both are shown in Fig. 4c.", "both are reported in the main text (Fig. 2b)."),
    ("(Fig. 3e,g)", "(Fig. 4d)"),
    ("the token-local Mamba validation in Fig. 1b is regenerable",
     "the exact-decomposition values in Fig. 1d,e are parsed at three decimals from the printed logs of the validation runs (the per-pair ConvNeXt and Mamba files were not kept; the notebook with its output is deposited), so they can be re-plotted and the runs repeated with survey_sigma1_v2.py --validate"),
    ("Partial correlations: relative dose, median 0.19, 35 of 50 positive (real text 0.09–0.53, shuffled 0.05–0.25, random labels −0.34 to 0.12, bidirectional −0.39 to 0.49, k = 1–3 −0.87 to 0.64, k = 4 and 5 0.59–0.89, k = 8 −0.07 to 0.80); per unit displacement, median 0.32, 35 of 50 positive (real 0.36–0.77, shuffled −0.41 to 0.29, random labels −0.38 to 0.46, k = 4 and 5 0.54–0.85).",
     "Partial correlations: relative dose, median 0.20, 32 of 50 positive (real text 0.09–0.53, shuffled −0.04 to 0.25, random labels −0.34 to 0.12, bidirectional −0.39 to 0.49, k = 1–3 −0.73 to 0.78, k = 4 and 5 0.59–0.89, k = 8 −0.07 to 0.80); per unit displacement, median 0.32, 34 of 50 positive (real 0.36–0.77, shuffled −0.41 to 0.02, random labels −0.38 to 0.46, k = 4 and 5 0.54–0.85)."),
    ("0.90–0.93 (real) and 0.82–0.97 (shuffled)", "0.90–0.93 (real) and 0.82–0.96 (shuffled)"),
    ("0.78–0.84 at 5,000 and 1.05–1.16 at 10,000", "0.78–0.84 at 5,000 and 1.05–1.15 at 10,000"),
    ("(five models each; bootstrap over the five models)", "(five models each; bootstrap over the five models; Extended Data Table 4)"),
    ("Layer-pruning comparisons on Pythia-12B (Block Influence, cosine distance and σ~1~ selection at 1–8 layers removed) and unstructured pruning (Wanda, σ~1~-guided and magnitude pruning to 50% sparsity) are described in Supplementary Note 7.",
     "Eleven public Pythia-410M checkpoints (steps 0, 512, 1,000, 2,000, 4,000, 8,000, 16,000, 32,000, 64,000, 100,000 and 143,000) were profiled with the survey estimator on the inputs of the 30-input survey and with the census branch rotation on the census data, seeds and loss (e7_checkpoints.py; Fig. 5c and Extended Data Fig. 4c,d). At step 143,000 the clean loss and the per-block rotation costs reproduce the census file (largest difference 5 × 10^−5^ nats) and R~ex0~ reproduces the 30-input survey (0.837). Within 50 iterations the power iteration reached its tolerance in 0% and 1% of batches at steps 0 and 512, against 46–79% from step 1,000; ‖Jv‖ does not decrease under the iteration, so the σ~1~ values at those two steps are lower bounds."),
    ("Extended Data Figs. 2–7, 9 and 10 and the three Extended Data tables are generated from the manifest", "Extended Data Figs. 2–7, 9 and 10 and Extended Data Tables 1–5 are generated from the manifest"),
    ("will be released at https://github.com/Diogenes67/hourglass-structure with a Zenodo DOI on publication.",
     "will be released at https://github.com/Diogenes67/local-gain-sensitivity under the MIT licence, with a Zenodo DOI on publication."),
    ("available to reviewers at submission and with a Zenodo DOI on publication.",
     "available to reviewers at submission and, on publication, on Zenodo under a CC BY 4.0 licence with a DOI."),
    ("the metric-comparison outputs and the per-run grid outputs are deposited", "the metric-comparison outputs, the per-run grid outputs and the per-checkpoint Pythia-410M survey and census outputs are deposited"),
    ("the checkpoints of the September destroyed-data, switching, mask × loss, routing,", "the checkpoints of the September destroyed-data, switching, routing,"),
]
for a, b in edits:
    assert a in mt, a
    mt = mt.replace(a, b)
# drop the census base-rate sentences
mt, n = re.subn(r'The base rate of interior sensitivity in the panel is 43 of 44.*?is the one model that fails the 10% clause\. ', '', mt)
assert n == 1
# leave-one-condition-out after the leave-one-model-out sentence
a = "this is a descriptive check of transfer across models, not a predictive evaluation on new populations."
assert a in mt
mt = mt.replace(a, a + " Leave-one-condition-out (loco_check.py): with each of the ten conditions held out in turn and β and γ fitted on the other nine, R^2^ is 0.51 against 0.29 from the block term alone per unit displacement and −0.11 against −0.09 at the relative dose; pooled over the four held-out text conditions it is 0.76 against 0.42, over the six held-out k-gram orders 0.04 against 0.03. Fitted on the 20 text models, gain and depth predict the 30 k-gram profiles at 0.17 against −0.28 from the block term alone, and fitted on the 30 k-gram models they predict the 20 text profiles at 0.51 against 0.33 (Extended Data Table 4). The slope fitted on the k-gram models alone is 0.53, against 2.19 on the text models.")
# review 4: logit-coordinate convention of the decomposition
a = "and an output alignment g^T^Fg/‖g‖^2^."
assert a in mt
mt = mt.replace(a, a + " Adding a constant to every logit changes ‖g‖ but not g^T^Fg, because F annihilates the constant vector, so the split into amplification and alignment is specific to the logit coordinates; their product, which gives the divergence, is not. The after-block factor defined below includes the final normalisation and the output head as well as the later blocks.")
# E1 methods section, before Block regions
e1 = """### Matched assay in pretrained decoders

GPT-2 124M, Pythia-410M, Pythia-1.4B, Llama-3.2-1B, Qwen2.5-1.5B and Gemma-2-2B (a beginning-of-sequence token prepended for Gemma-2-2B, as in the census) were probed with the matched assay and the linearised response in one run (matched_pretrained.py; float32, eager attention, TF32 disabled, one NVIDIA A100 40 GB, 8–198 min per model). Inputs were 20 WikiText-103 validation sequences of 128 tokens spread evenly through the split, at positions t ∈ {8, 32, 56, 80, 104, 120}, for every block. σ~1~ and v~1~ came from the *J*^T^*J* power iteration on the survey's in-context token-local map (two restarts, 50 iterations, tolerance 10^−6^). The block input at t was displaced by ±ε‖h~l,t~‖u with u = v~1~ or one of eight random unit vectors drawn once per input and position, ε ∈ {0.01, 0.03, 0.1, 0.3}, signs averaged, with the four readouts of the controlled assay. The linearised response was computed on the first five inputs for v~1~ and the same eight random directions. Within-model correlations are over blocks 1 to L − 2, with 95% intervals from 500 resamples of whole inputs; pooled regressions express depth as block/(L − 1) because L differs between models, with standard errors clustered on the six models; out-of-model prediction holds out one model or one family, the two Pythia models forming one family (matched_pretrained_analysis.py, e1_extra.py). Gain along v~1~ at ε = 0.01 reproduced σ~1~ to a mean ratio of 0.96–1.00 per model, and the linearised same-position divergence matched the measured value to a median ratio of 1.000–1.001. At ε = 0.01 a minority of probes in GPT-2, Qwen2.5 and Gemma-2 gave random-direction divergences near the float32 resolution of the log-softmax over their large vocabularies, so ratios of measured divergences are summarised at ε = 0.1, as in the controlled models, and the decomposition factors use the gains at ε = 0.01 with the analytic linearised response.

"""
a = "### Block regions and checkpoints"
assert a in mt
mt = mt.replace(a, e1 + a)

# ---- Supplementary Information renumbering (make_supplement_nmi.py): old 4 -> 1, 6 -> 2, 7 -> 3, 13 -> 4, 17 -> 5; others redirected
SN_SPECIFIC = [
    ("(Supplementary Note 11)", "(Extended Data Fig. 1)"),
    (" (Supplementary Note 12)", ""),
    ("and a threshold sweep from 0.60 to 0.95 is in Supplementary Note 10.", "and sweeping the threshold from 0.60 to 0.95 gives 6 to 22 interior-valley models (14 at 0.80)."),
    ("summarised with the same middle-to-edge ratio (Supplementary Note 1).", "summarised with the same middle-to-edge ratio (Extended Data Fig. 3)."),
    ("rather than recomputed (Supplementary Note 2).", "rather than recomputed (Extended Data Fig. 3)."),
    ("and the confidence-adjusted association by partial Spearman correlation (ρ = 0.68 given confidence, p < 0.001). For the entropy-perturbation control, effect estimates and intervals for noise within each k are given in Supplementary Note 6 alongside the ANOVA;",
     "and the confidence-adjusted association by partial Spearman correlation, which was inconclusive because confidence and order share most of their rank variance (partial ρ 0.33–0.37, p 0.05–0.09); the cell means and the ANOVA are in Supplementary Note 6;"),
    (" Exact p values and sample sizes are in Supplementary Table 10.", ""),
    ("Supplementary Notes 4, 6, 13 and 16", "Supplementary Notes 1, 2 and 4"),
]
SN_MAP = {"17": "5", "13": "4", "7": "3", "6": "2", "4": "1"}


def sn_remap(txt, specific):
    for a, b in specific:
        assert a in txt, a
        txt = txt.replace(a, b)
    def f(m):
        assert m.group(1) in SN_MAP, m.group(0)
        return "Supplementary Note " + SN_MAP[m.group(1)]
    return re.sub(r"Supplementary Note (\d+)", f, txt)

mt = sn_remap(mt, SN_SPECIFIC)
open(os.path.join(D, "methods_nmi.md"), "w").write(mt)

# ---- legends: new main legends + Table 1, then Extended Data with renumbered references
lg = open(os.path.join(NCS, "legends.md")).read()
ed = "## Extended Data" + lg.split("## Extended Data", 1)[1]
ed_edits = [
    ("validated exactly in Fig. 1b", "validated exactly in Fig. 1d"),
    ("Companion to Fig. 4. **a**–**c**, As Fig. 4a,b with the k-gram models added", "Companion to Figs. 2 and 3. **a**–**c**, Per-block σ~1~ and downstream divergence along v~1~ at ε = 0.1 (formerly main-text panels), with the k-gram models added"),
    ("**e**, As Fig. 4d.", "**e**, Middle-block downstream divergence at ε = 0.1 against the middle-to-edge gain ratio of the matched probes (Spearman ρ = 0.80, 95% CI 0.67 to 0.88, n = 50); Fig. 4e shows the same relation for rotation cost in 80 models."), ("**f**, As Fig. 4e", "**f**, As Fig. 3a"),
    ("same-position readout of Fig. 4e", "same-position readout of Fig. 3a"), ("**i**, As Fig. 4c.", "**i**, As Fig. 2b."),
    ("the 50-model matched assay of Fig. 4 supersedes it", "the 50-model matched assay of Figs. 2 and 3 supersedes it"),
    ("five seeds per value (Fig. 3a–c)", "five seeds per value (Fig. 4c)"),
    ("four from hourglass to not", "four losing the interior valley"),
    ("and stays above 0.98 thereafter", "and stays at 0.98–0.99 thereafter (means of four seeds)"),
    ("after the September re-runs every profile in the paper is on the canonical estimator, and the March routing, residual-sweep and k-gram profiles appear nowhere.",
     "after the September re-runs every profile in the paper is on the canonical estimator except Extended Data Fig. 4a,b (March 2026 single-position estimator, stated in its legend)."),
    ("**c**, Pythia-410M public checkpoints from step 0 to 143,000 (float32, canonical estimator, May 2026): R~ex0~ stays at 0.82–0.96 while the middle-block rotation cost rises from −0.04 nats at initialisation to 4.5 nats by step 2,000 and 7.1–7.3 nats from step 32,000.",
     "**c**, σ~1~ per block for 11 public Pythia-410M checkpoints from step 0 to 143,000 (colour, step; block 0 omitted; 30 inputs × 8 positions, survey estimator; middle third shaded). Block 0 is 8.0 at initialisation and 41–68 from step 2,000. The gain ratio falls after step 32,000 because the late third rises (mean σ~1~ 3.0 to 4.5) faster than the middle third (2.9 to 3.4). **d**, Branch-rotation cost per block for the same checkpoints (census protocol; blocks 1 to L − 2; middle third shaded; logarithmic axis). At initialisation every per-block cost is within ±0.04 nats and is not drawn; the step-512 value at block 19 (3 × 10^−4^ nats) is drawn at 10^−3^. Block 5 carries the largest cost from step 2,000 (0.87 nats, rising to 5.5 nats at step 143,000). Fig. 5c summarises panels c and d."),
]
for a, b in ed_edits:
    assert a in ed, a
    ed = ed.replace(a, b)
# strip provenance tags at the end of each legend, except ED Fig. 1 (its note is a disclosure) and the tables
paras = ed.split("\n\n"); out = []
for p in paras:
    if p.startswith("**Extended Data Fig.") and not p.startswith("**Extended Data Fig. 1 |"):
        p = re.sub(r'\s*\[[^\[\]]*\]\s*$', '', p)
    out.append(p)
ed = "\n\n".join(out)
ed = sn_remap(ed, [])
ed = ed.rstrip() + "\n\n" + ("**Extended Data Table 4 | Gain and consequence beyond depth.** **a**, Held-out prediction of the within-model profile of consequence, as centred R² from gain and depth against depth alone, for normalised sensitivity and for the divergence at the relative dose: controlled models with one model, one training condition or one source family held out, and pretrained decoders with one model or one family held out. **b**, Pooled log–log slopes of consequence on σ~1~ under three depth specifications, fitted within each condition, per model, and in the pretrained decoders. **c**, Within-model rank correlations over blocks, raw, with block partialled out and over first differences.") + "\n\n" + ("**Extended Data Table 5 | Condition-level gain ratios and middle-block intervention costs.** Middle-to-edge gain ratio, early-third and late-third ratios to the middle, and the loss increase under branch rotation and identity skipping of the middle blocks, for the four training conditions, the two switching conditions, checkpoints during training, the k-gram and entropy-matched families at 12 blocks, and the real-text and shuffled-text repeat at 24 blocks.")
open(os.path.join(D, "legends_nmi.md"), "w").write("## Figure legends" + legends_part.rstrip() + "\n\n" + ed)

for f in ("main_nmi.md", "methods_nmi.md", "legends_nmi.md"):
    print(f, len(open(os.path.join(D, f)).read().split()), "words")
print("refs mapped:", sorted({k for k in re.findall(r'\{\{([a-z0-9]+)\}\}', open(os.path.join(D, 'main_nmi.md')).read())}))
