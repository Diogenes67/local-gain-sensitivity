"""Supplementary Information for the NMI manuscript (23 Sept 2026; E7 provenance 24 Sept), built from supplement.md (May 2026 notes + Note 17).

Kept and renumbered (old -> new): 4 -> 1 (bracket-nesting probes), 6 -> 2 (entropy and confidence controls), 7 -> 3 (position
confound; the pruning comparisons are withdrawn), 13 -> 4 (intervention construct validity), 17 -> 5 (provenance).
Dropped: 1, 2, 3, 5, 8, 9, 10, 11, 12, 14, 15, 16 and the May supplementary figures and tables. Their material is superseded by the
September re-runs and is either in the Methods, the Extended Data or Note 5, or withdrawn (Note 14, the May Pythia checkpoint
panel; Note 16, the March layer-skip ratio, superseded by Extended Data Table 5).
Values in Note 1 and in Note 4, Experiment 2, are recomputed from the retained per-seed files (exp2a_bracket_seed0-3.json,
d3l_alignment_adaptation/d3l_aggregate.json).
Run: python make_supplement_nmi.py -> nmi/supplement_nmi.md (then pandoc -> nmi/hourglass_NMI_supplement.docx)
"""
import os, re, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); MAC = os.path.dirname(HERE)
SRC = open(os.path.join(HERE, "supplement.md")).read()


def notes(src):
    body = src.split("# Supplementary Notes", 1)[1].split("# Supplementary figures", 1)[0]
    parts = re.split(r'(?m)^\*\*Supplementary Note (\d+): ', body)
    out = {}
    for i in range(1, len(parts), 2):
        n = int(parts[i]); txt = parts[i + 1]
        title, rest = txt.split("**", 1)
        out[n] = (" ".join(title.split()), rest)
    return out


def unwrap(txt):
    paras = [p for p in re.split(r'\n\s*\n', txt.strip())]
    out = []
    for p in paras:
        if p.lstrip().startswith(">") or p.lstrip().startswith("|") or p.lstrip().startswith("---"):
            out.append(p.strip()); continue
        out.append(" ".join(l.strip() for l in p.split("\n")))
    return "\n\n".join(out)


def clean(t):
    t = t.replace(" \\-\\-- ", ", ").replace("\\-\\-- ", ", ").replace(" \\-\\--", ",").replace(" --- ", ", ").replace("---", ",")
    t = t.replace("\\--", "–").replace("--", "–")
    t = t.replace("\\\\\\_", "_").replace("\\_", "_").replace("\\'", "'").replace('\\"', '"').replace("\\<", "<").replace("\\>", ">")
    t = t.replace("\\[", "[").replace("\\]", "]").replace("\\|", "|").replace("R\\^2\\^", "R²")
    return t


def span(t, start, end, new):
    """replace t[start anchor .. end anchor) with new; both anchors must occur once"""
    assert t.count(start) == 1, start[:50]
    i = t.index(start)
    j = t.index(end, i) if end else len(t)
    return t[:i] + new + t[j:]


def rep(t, a, b):
    assert t.count(a) == 1, a[:60]
    return t.replace(a, b)


N = {k: (title, clean(unwrap(body))) for k, (title, body) in notes(SRC).items()}

# ---------------------------------------------------------------- Note 1 (old 4): bracket-nesting probes
bs = [json.load(open(os.path.join(MAC, "results", f"exp2a_bracket_seed{s}.json"))) for s in range(4)]
r2 = np.array([b["probes"]["depth_r2"] for b in bs]); ts = np.array([b["probes"]["top_stack_acc"] for b in bs])
pr = np.array([b["dimensionality"]["participation_ratio"] for b in bs]); dl = np.array([b["delta_L_per_layer"] for b in bs])
chk1 = dict(r2_emb=r2[:, 0].mean(), r2_b0=r2[:, 1].mean(), r2_rest=(r2[:, 2:].mean(0).min(), r2[:, 2:].mean(0).max()), r2_min_any=r2[:, 1:].min(),
            ts_b1=ts[:, 2].mean(), ts_from=int(np.argmax(ts.mean(0) > 0.995)) - 1, pr_emb=pr[:, 0].mean(), pr_blocks=(pr[:, 1:].mean(0).min(), pr[:, 1:].mean(0).max()),
            dl0=dl[:, 0].mean(), dl11=dl[:, -1].mean(), dl_min=dl.mean(0).min())
t = N[4][1]
t = rep(t, "This note reports the bracket-nesting probe experiment referenced in the main text (Figure 6A).",
        "*Provenance: March 2026 run of four models; the per-seed result files are retained (exp2a_bracket_seed0–3.json) and the checkpoints are not. The values below are recomputed from those files. Shown in Extended Data Fig. 8a.*\n\nThis note reports the bracket-nesting probe experiment of Extended Data Fig. 8a.")
t = span(t, "**Results.**", "**Interpretation.**",
         f"**Results.** Values are means over the four seeds. A linear probe for nesting depth reached R² = {chk1['r2_b0']:.2f} at the output of block 0 "
         f"({chk1['r2_emb']:.2f} at the embedding) and stayed at {chk1['r2_rest'][0]:.2f}–{chk1['r2_rest'][1]:.2f} through block 11 (at least {chk1['r2_min_any']:.2f} in every seed). "
         f"The top-of-stack probe reached {chk1['ts_b1']:.2f} accuracy after block 1 and 1.00 from block {chk1['ts_from']} onward. Per-block rotation cost fell from "
         f"{chk1['dl0']:.2f} nats at block 0 to {chk1['dl11']:.2f} nats at block 11 and was no lower at any block. The participation ratio of the "
         f"representation fell from {chk1['pr_emb']:.0f} at the embedding to {chk1['pr_blocks'][0]:.0f}–{chk1['pr_blocks'][1]:.0f} across the blocks.\n\n")
t = rep(t, "by layer 2–3.", "by block 1 or 2.")
t = rep(t, "(see Note 13), and that the hourglass interior propagates", "(Supplementary Note 4), and that the interior blocks propagate")
N1 = ("Bracket-nesting probes and compact state representation", t)

# ---------------------------------------------------------------- Note 2 (old 6): entropy and confidence controls
t = N[6][1]
t = rep(t, "A reviewer asked whether alternative confidence measures produce the same result. We repeated the partial-Spearman analysis using two additional covariates:",
        "To test whether alternative confidence measures give the same result, the partial-Spearman analysis was repeated with two additional covariates:")
t = rep(t, "comes from the convergent metrics reported in the main text.", "comes from the convergent metrics of the March 2026 analysis.")
t = rep(t, "The main dependency-order sweep (Figure 3) confounds", "The March 2026 dependency-order sweep confounded")
t = re.sub(r"k noise = 0\.0 noise = 0\.3 noise = 0\.6 noise = 0\.9 mean.*?1\.453",
           "Cell means (± s.d. over five seeds) of the middle-block cost in nats at noise 0, 0.3, 0.6 and 0.9: k = 5, 1.317 ± 0.079, 1.310 ± 0.116, "
           "1.265 ± 0.093 and 1.273 ± 0.084 (mean 1.291); k = 8, 1.472 ± 0.064, 1.467 ± 0.050, 1.427 ± 0.069 and 1.446 ± 0.068 (mean 1.453); k = 2, "
           "mean 0.493 (the cell values are not recoverable from the log).", t, flags=re.S)
t = t.replace("Delta-L_waist", "ΔL~waist~").replace("ΔL_waist", "ΔL~waist~").replace("H_pred", "H~pred~").replace("H_cond", "H~cond~")
t = t.replace("10\\^-21", "10^−21^").replace("k >= 3", "k ≥ 3").replace("partial rho", "partial ρ").replace("rho(", "ρ(").replace("rho = ", "ρ = ").replace("(rho >", "(ρ >").replace("= -0.987", "= −0.987")
N2 = ("Entropy, confidence and robustness controls", t)

# ---------------------------------------------------------------- Note 3 (old 7): position confound only
t = N[7][1]
i = t.index("**Position-corrected functional measures (D3m).**"); j = t.index("**Unstructured pruning: Wanda baseline.**")
d3m = t[i:j].strip()
d3m = d3m.replace("waist depth sensitivity", "middle-block sensitivity").replace("waist computation", "middle-block computation")
d3m = span(d3m, "These results validate the position-confound argument", None,
           "Per-block rotation cost therefore mixes a block's contribution with its position, but the confound does not change the middle-block comparison across dependency orders.")
t = ("*Provenance: March 2026 k-gram models, retained aggregate outputs. Earlier versions of this note also reported layer-removal and unstructured-pruning "
     "comparisons on Pythia-12B; they are withdrawn (Supplementary Note 5).*\n\nPer-block rotation cost is confounded with block position: a block near the "
     "input has more downstream blocks through which a disruption can propagate than a block near the output. Removing the blocks with the lowest per-block "
     "cost therefore selects edge blocks for their position, not for a small contribution.\n\n" + d3m)
N3 = ("Position confound in per-block rotation cost", t)

# ---------------------------------------------------------------- Note 4 (old 13): intervention construct validity
ad = json.load(open(os.path.join(MAC, "results", "d3l_alignment_adaptation", "d3l_aggregate.json")))
ps = list(ad["per_seed"].values())
b = np.array([p["dL_target_before"] for p in ps]); a = np.array([p["dL_target_after"] for p in ps]); c = np.array([p["control_dL_target"] for p in ps])
red = np.array([p["reduction_pct"] for p in ps]); cl = np.array([p["clean_nll_after"] - p["clean_nll_before"] for p in ps]); cr = (b - c) / b * 100
sd = lambda v: f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"
chk4 = dict(before=sd(b), after=sd(a), control=sd(c), reduction=f"{red.mean():.1f} ± {red.std(ddof=1):.1f}", red_range=(red.min(), red.max()),
            clean=(cl.min(), cl.max()), control_recovery=f"{cr.mean():.1f} ± {cr.std(ddof=1):.1f}")
t = N[13][1]
t = ("*Provenance: March 2026 runs on the k-gram models. Experiment 1 is reported from its retained aggregate values; Experiment 2 is recomputed from the "
     "retained per-seed file (d3l_alignment_adaptation/d3l_aggregate.json). Experiment 2 is shown in Extended Data Fig. 8b.*\n\n") + t
t = span(t, "**Pre-specified thresholds**:", "**Results.** The mean Spearman",
         "**Thresholds set before the run.** A mean ρ(scramble, resample) above 0.85 would indicate the same construct as resample ablation, 0.5–0.85 a "
         "correlated but distinct construct and below 0.5 a different construct.\n\n")
t = rep(t, "(Table S11a)", "(Supplementary Table 1)")
t = rep(t, "\n\n\\*\\*\n\nablation, per model.\\*\\*", "")
t = span(t, "Five k = 8 models (seeds 0–4) were used.", "**Pre-specified threshold**",
         "Five k = 8 models (seeds 0–4) were used. A fixed random orthogonal rotation was applied to the output of block 5; blocks 0–5 were frozen and "
         "blocks 6–11 were fine-tuned with AdamW (learning rate 1 × 10⁻⁴) for 500 steps, after which the cost of the rotation was measured again. In a "
         "control arm the same blocks were fine-tuned for the same number of steps without the rotation, and the cost of then applying it was measured.\n\n")
t = span(t, "**Pre-specified threshold**", "**Combined interpretation.**",
         "**Threshold set before the run.** A reduction of the cost by more than 70% would indicate that the intervention measures alignment-disruption "
         "sensitivity rather than the necessity of the rotated block's computation.\n\n"
         f"**Results.** Adaptation reduced the cost of rotating block 5 from {chk4['before']} to {chk4['after']} nats, a reduction of {chk4['reduction']}% "
         f"({chk4['red_range'][0]:.1f}–{chk4['red_range'][1]:.1f}% across seeds; Supplementary Table 2). Fine-tuning without the rotation left the cost at "
         f"{chk4['control']} nats ({chk4['control_recovery'].replace('-', '−')}% change). The clean loss rose by {chk4['clean'][0]:.2f}–{chk4['clean'][1]:.2f} nats across seeds. "
         "Earlier drafts reported a two-condition pilot (5.25 to 0.09 nats) and a three-condition rerun (5.48 to 0.29 nats, 94.7%) whose result files were "
         "not retained; the values here are from the retained file.\n\n"
         "**Interpretation.** The reduction exceeds the threshold, and the control shows that the recovery is specific to adaptation to the rotated basis. "
         "Most of the cost of a rotation therefore reflects the dependence of later blocks on the basis established upstream, not the necessity of the "
         "rotated block's computation, and the intervention is interpreted as alignment-disruption sensitivity. The recovery does not show that the full "
         "residual state was preserved.\n\n")
t = span(t, "This reframing does not undermine the main results of the paper.", "**Off-manifold displacement.**", "")
t = t.replace("waist layers", "middle blocks").replace("waist layer", "middle block")
N4 = ("Intervention construct validity", t)

# ---------------------------------------------------------------- Note 5 (old 17): provenance
t = N[17][1]
t = rep(t, "Every number, table and figure in the main text, Extended Data Figs. 2 to 7, 9 and 10, and the three Extended Data tables are produced from the frozen results manifest by the scripts named in Extended Data Table 3.",
        "Every number, table and figure in the main text, Extended Data Figs. 2 to 7, 9 and 10, and Extended Data Tables 1 to 5 are produced from the frozen results manifest by the scripts named in Extended Data Table 3. The exact-decomposition values of Fig. 1d,e are parsed at three decimals from the printed logs of the validation runs; the per-pair ConvNeXt and Mamba files were written to the instance disk and not kept, and the notebook with its output is deposited.")
t = rep(t, "The March 2026 mechanism experiments (Extended Data Fig. 8; Supplementary Notes 4, 6, 13 and 16)",
        "The March 2026 mechanism experiments (Extended Data Fig. 8; Supplementary Notes 1, 2 and 4)")
t = rep(t, "the token-local Mamba validation in Fig. 1b is regenerable.", "the token-local Mamba validation in Fig. 1d is parsed from the printed log of its run.")
t = rep(t, "The Pythia-410M checkpoint trajectory in the same figure (Extended Data Fig. 4c) is canonical.",
        "The Pythia-410M checkpoint trajectory in the same figure (Extended Data Fig. 4c,d; also Fig. 5c) is canonical. Eleven public checkpoints were profiled with the survey estimator and the census branch rotation (e7_checkpoints.py), and at the final checkpoint both reproduce the deposited census and 30-input survey files.")
t = span(t, "**Superseded single-position estimates.**", "**Matched perturbation assay, doses", "")
t = t.replace("a two-arm hourglass on real text", "a two-arm interior valley on real text").replace("no hourglass in any pure condition", "no interior valley in any pure condition")
t = t.replace("reported the hourglass in every causal-mask run", "reported an interior valley in every causal-mask run")
t += ("\n\n**Withdrawn results.** Earlier versions reported layer-removal and unstructured-pruning comparisons on Pythia-12B in which σ₁ selected layers "
      "or weights less well than Block Influence, cosine distance and Wanda. They are withdrawn. In the unstructured comparison the Jacobian–vector product "
      "failed at every layer at that scale and the σ₁-guided condition fell back to per-layer magnitude pruning, so σ₁ was not used; the layer-removal results "
      "are in a file without model, estimator or protocol provenance; and the worked pruning protocol was reconstructed from terminal output. The main text "
      "makes no pruning claim. The May 2026 layer-skip decomposition (skip cost 0.60 of scramble cost) is superseded by the September identity-skip "
      "measurements (Extended Data Table 5; Table 1). A May 2026 Pythia-410M checkpoint summary (middle-block rotation cost 7.1–7.3 nats from step 32,000) used a rotation that "
      "differed from the census (7.2 nats in the middle third at the final checkpoint, against 0.38 nats in the census file); it is withdrawn and replaced by the rerun "
      "under the census protocol (Fig. 5c; Extended Data Fig. 4c,d).")
N5 = ("Provenance of every reported result", t)

# ---------------------------------------------------------------- assemble
def sweep(t):
    t = re.sub(r"gain-hourglass", "interior-valley", t); t = re.sub(r"\bHG\+", "interior-valley", t)
    t = t.replace("hourglass", "interior valley").replace("Waist depth sensitivity", "Middle-block sensitivity").replace("waist depth sensitivity", "middle-block sensitivity")
    return t

front = ("# Supplementary Information\n\n**When local amplification predicts downstream sensitivity in neural networks**\n\n"
         "Anthony Porter, Hemanth Saratchandran, Nadhir Hassen, Zhibin Liao, Johan Verjans\n\n"
         "This Supplementary Information contains five Supplementary Notes and two Supplementary Tables. Notes 1 to 4 report retained analyses from the "
         "March 2026 runs that the Methods cite for design details and controls; the main results do not rest on them, and the provenance of each is stated "
         "at its head. Note 5 records which run each reported result comes from, what it replaced, what can be recomputed from the deposited files and what "
         "has been withdrawn.\n\n")
out = [front, "# Supplementary Notes\n"]
for i, (title, body) in enumerate((N1, N2, N3, N4, N5), 1):
    out.append(f"**Supplementary Note {i}: {title}**\n\n{sweep(body).strip()}\n")
tab = ("# Supplementary Tables\n\n**Supplementary Table 1 | Rank correlation between per-block rotation cost and other ablations.** Ten k-gram models (k = 5 and "
       "k = 8, five seeds each), Spearman ρ between the per-block cost vectors (12 blocks) of each pair of interventions; March 2026, retained aggregate values.\n\n"
       "| Comparison | Mean ρ ± s.d. | Range |\n|---|---|---|\n| Rotation against resample ablation | −0.031 ± 0.378 | −0.783 to 0.573 |\n"
       "| Rotation against mean ablation | 0.006 ± 0.137 | −0.189 to 0.273 |\n| Rotation against zero ablation | −0.009 ± 0.213 | −0.364 to 0.329 |\n\n"
       "**Supplementary Table 2 | Downstream-only adaptation.** Five k = 8 models; rotation of block 5, blocks 6–11 fine-tuned for 500 steps; from "
       "d3l_aggregate.json (per-seed values).\n\n| Seed | Cost before (nats) | Cost after (nats) | Reduction (%) | Control cost (nats) | Clean-loss change (nats) |\n|---|---|---|---|---|---|\n")
for k, p in ad["per_seed"].items():
    tab += f"| {k[-1]} | {p['dL_target_before']:.3f} | {p['dL_target_after']:.3f} | {p['reduction_pct']:.1f} | {p['control_dL_target']:.3f} | {p['clean_nll_after'] - p['clean_nll_before']:.3f} |\n"
tab += f"| mean ± s.d. | {chk4['before']} | {chk4['after']} | {chk4['reduction']} | {chk4['control']} | {cl.mean():.2f} ± {cl.std(ddof=1):.2f} |\n"
out.append(tab)
md = "\n".join(out)
left = sorted(set(re.findall(r"(?i)hourglass|HG\+|Figure \d|Table S\d+|Note 1[3-7]|\\-", md)))
os.makedirs(os.path.join(HERE, "nmi"), exist_ok=True)
import sys; sys.path.insert(0, os.path.join(HERE, "nmi")); import copyedit_patch
md = copyedit_patch.apply(md, "supp")   # Nature copy-edit, 24 Sept 2026
open(os.path.join(HERE, "nmi", "supplement_nmi.md"), "w").write(md)
json.dump(dict(note1=chk1, note4=chk4, leftover_tokens=left, words=len(md.split())), open(os.path.join(HERE, "nmi", "supplement_nmi_check.json"), "w"), indent=1, default=float)
print(json.dumps(dict(note1=chk1, note4=chk4, leftover=left, words=len(md.split())), indent=1, default=float))
