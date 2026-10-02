# Code and results for "When local amplification predicts downstream sensitivity in neural networks"

Code and results for the Patterns submission (Porter, Hassen, Saratchandran, Liao, Verjans and van den Hengel, 2026).
98 scripts and notebooks under `code/`; 260 result files (107 MB) under `results/`, each listed with its SHA-256 in
`results_manifest.md`. Table S7 of the paper gives, for every result, the operator, Jacobian domain, intervention, precision,
models, script and evaluation data; Supplemental note 5 gives the class of evidence for each result and what was withdrawn.

## Layout

- `code/experiments/` GPU experiments. Each writes its outputs to persistent storage as it runs and resumes after a disconnect.
  `survey/survey_sigma1_v2.py` is the canonical gain estimator (token-local Jacobian, J^T J power iteration by forward- and
  reverse-mode products, float32, natural inputs); every profile in the paper comes from it or imports it. `validation/` holds the
  exact-decomposition validation (25 Sept 2026 rerun) and the Mamba full-scan check; `controlled/` the training conditions, k-gram,
  entropy-matched, lag, 24-layer, mask x loss, grid, topology and skip-coefficient runs and the intervention comparison;
  `matched_assay/` and `pretrained_assay/` the matched token-local perturbation assay and the linearised response;
  `census/` the 45-model branch-rotation census and the representational-measure comparison; `checkpoints/` the Pythia-410M trajectory.
- `code/analysis/` statistics read from the result files: within-model correlations and pooled regressions (`matched_analysis.py`,
  `regression_check.py`), held-out prediction (`loco_check.py`, `predictor_check.py` with the joint baseline, `family_transfer.py`,
  `readout_check.py`), the linearised response and downstream factor (`linresp_*.py`, `downstream_factor.py`), the pretrained assay
  (`matched_pretrained_analysis.py`, `e1_extra.py`), region lists (`region_check.py`) and the count check.
- `code/figures/`, `code/tables/` build the main and supplemental figures and tables from the result files
  (`make_fig1_jmlr.py` is Figure 1, `make_fig2_patterns.py` Figure 2, `make_fig3_direction.py` Figure 3, `make_fig4_data.py` Figure 4,
  `make_fig5_pretrained.py` Figure 5, `make_edfigs.py` and `make_edfig4_nmi.py` Figures S1–S11, `make_tables.py` and
  `make_ed_tables_nmi.py` Tables S1–S4 and S7).
- `code/manuscript/` builds the submitted documents from markdown (`build.sh`, `build_supp.py`, `update_table_s3.py`).
- `results/` the result files, in their original folder structure.

## Display items

| Display item | Script | Result folder |
|---|---|---|
| Figure 1 | make_fig1_jmlr.py | validation_rerun/results/validation |
| Figure 2 | make_fig2_patterns.py (predictor_check.py, family_transfer.py, regression_check.py) | matched/, scale/, matched_pretrained/, analysis/ |
| Figure 3 | make_fig3_direction.py (downstream_factor.py, linresp_centred_analysis.py) | matched/, matched/linresp, matched/linresp_centred, matched_pretrained/, matched_pretrained_centred/ |
| Figure 4 | make_fig4_data.py | d1/results_v21, reprofile/, markov/ |
| Figure 5 | make_fig5_pretrained.py | survey_30inputs/, census_v4/, e7_checkpoints/ |
| Table 1 | values from the analyses above | Table S7 |
| Table 2 | values from the analyses above | Table S3 |
| Figure S1 | mamba_fullscan_validation.py | validation_rerun/results/fullscan |
| Figures S2–S11 | make_edfigs.py, make_edfig4_nmi.py | Table S7 |
| Tables S1, S2, S7 | make_tables.py | survey, census_v4 |
| Tables S3, S4 | make_ed_tables_nmi.py, update_table_s3.py | regression_check.json, loco_check.json, predictor_check.json, matched_pretrained/, d1/, reprofile/, markov/, scale/ |

## Requirements

Python 3.11, torch >= 2.6, transformers >= 5.16, numpy, scipy, pandas, statsmodels, matplotlib, huggingface_hub, datasets, pyarrow;
timm for MLP-Mixer; torchvision for CIFAR-100. The experiments ran on single NVIDIA A100, H200 or RTX PRO 6000 GPUs.
The manuscript build needs pandoc >= 3 with citeproc, xelatex and python3.

## Rebuilding the analyses from results/

    NCS_R=results/NCS python matched_analysis.py        # unzip matched/matched_results.zip and the d1, reprofile and scale zips first
    python regression_check.py; python loco_check.py
    NCS_R=results/NCS MP_RESULTS=results/NCS/matched_pretrained/results python predictor_check.py
    MP_RESULTS=results/NCS/matched_pretrained/results python family_transfer.py
    python make_fig1_jmlr.py; NCS_R=results/NCS python make_fig2_patterns.py; python make_fig3_direction.py; python make_fig4_data.py; python make_fig5_pretrained.py
    NCS_R=results/NCS python make_ed_tables_nmi.py; python update_table_s3.py
    bash build.sh                                        # main_patterns.{docx,pdf}, supplement_patterns.{docx,pdf}

## Licence and citation

MIT licence (code). Results are released under CC BY 4.0. Cite the paper and the Zenodo DOI of the tagged release.
