# Code and results for "When local amplification predicts downstream sensitivity in neural networks"

Repository: https://github.com/Diogenes67/local-gain-sensitivity

Code for the Nature Machine Intelligence submission. 50 scripts and notebooks; 178 result files listed with SHA-256 in
`results_manifest_nmi.md`.

## Layout

- `code/experiments/` GPU experiments, each written to save its outputs to persistent storage as it runs and to resume after a disconnect.
  `survey/survey_sigma1_v2.py` is the canonical gain estimator (token-local Jacobian, J^T J power iteration by forward- and reverse-mode
  products, float32, natural inputs); every profile in the paper comes from it or imports it, except Extended Data Fig. 4a,b
  (March 2026 single-position estimator, stated in the legend).
- `code/analysis/` statistics read from the result files (matched assay, pooled regressions, held-out prediction, linearised response,
  pretrained assay, region lists, count check).
- `code/figures/`, `code/tables/` build every main and Extended Data figure and table from the result files.
- `code/manuscript/` assembles the manuscript and the Supplementary Information from markdown sources.

## Display items

| Display item | Script | Result folder |
|---|---|---|
| Fig. 1a–e | make_fig1_nmi.py | validation/ |
| Fig. 2 | make_fig2_scale.py | matched/, scale/, matched_pretrained/, loco_check.json |
| Fig. 3 | make_fig3_direction.py | matched/, matched/linresp/, matched_pretrained/ |
| Fig. 4 | make_fig4_data.py | d1/results_v21, reprofile/, markov/ |
| Fig. 5 | make_fig5_pretrained.py | survey_30inputs/, census_v4/, e7_checkpoints/results (panel c) |
| Table 1 | values from the analyses above | see Extended Data Table 3 |
| ED Fig. 4 | make_edfig4_nmi.py | results/ (March 2026 trajectories, panels a,b), e7_checkpoints/results (panels c,d) |
| ED Figs 1–3, 5–10 | make_edfigs.py, matched_analysis.py | see Extended Data Table 3 |
| ED Tables 1–3 | make_tables.py, make_ed_tables_nmi.py | survey, census_v4 |
| ED Tables 4, 5 | make_ed_tables_nmi.py | regression_check.json, loco_check.json, matched_pretrained/, d1/, reprofile/, markov/, scale/ |

Extended Data Table 3 gives, for every result, the operator, Jacobian domain, intervention, precision, models, script and evaluation data.
Supplementary Note 5 gives the class of evidence for each result (regenerable, retained summary or image only) and what was withdrawn.

## Requirements

Python 3.11, torch >= 2.6, transformers >= 5.16, numpy, scipy, pandas, statsmodels, matplotlib, huggingface_hub, datasets, pyarrow;
timm for MLP-Mixer; torchvision for CIFAR-100. The experiments ran on single NVIDIA A100 or RTX PRO 6000 GPUs.

## Rebuilding

    python matched_analysis.py            # needs the unzipped matched/, d1/ and reprofile/ results (NCS_R)
    python regression_check.py; python loco_check.py
    python make_fig1_nmi.py; python make_fig2_scale.py; python make_fig3_direction.py; python make_fig4_data.py; python make_fig5_pretrained.py; python make_edfig4_nmi.py
    NCS_R=<results> python make_ed_tables_nmi.py
    python nmi/assemble_nmi.py; python nmi/build_nmi.py; python make_supplement_nmi.py

## Licence

Code: MIT (`LICENSE`). Result files and the results manifest: CC BY 4.0 (`DATA_LICENSE.md`).

## Before release

Tag a release and mint a Zenodo DOI through the GitHub integration; cite the DOI in the Code availability statement.
