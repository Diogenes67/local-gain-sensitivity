# Patterns submission tree (26 Sept 2026)

Two documents for Cell Press Patterns: `main_patterns.{docx,pdf}` (Summary, Highlights, The bigger picture with the data
science maturity level, Introduction, Results, Discussion ending with guidance for perturbation studies, Experimental
procedures with Resource availability, back matter, references in Cell numbered style, figure legends, Tables 1 and 2; the
PDF is a review copy with line numbers and the five figures appended) and `supplement_patterns.{docx,pdf}` (Document S1:
Figures S1–S11, Tables S1–S7, Supplemental notes 1–5 including the experimental record, supplemental methods, supplemental
references [S1]–[S4]).

Build: `bash build.sh` (needs pandoc >= 3 with citeproc, xelatex with lmodern, python3, DejaVu fonts). `build_supp.py`
assembles Document S1 from the markdown sources in `src/`; `update_table_s3.py` rewrites sections d and e of Table S3
(`src/ed_table4.md`) from `analysis/predictor_check.json`.

- `src/main_patterns.md`: the paper, written for this version (pandoc citations `[@key]`, resolved from `src/refs.bib` with
  `src/cell.csl`).
- `src/record_patterns.md`: the experimental record (Supplemental note 5, first part), written for this version.
- `src/legends_nmi.md`, `src/supplement_nmi.md`, `src/methods_nmi.md`, `src/ed_table*.md`: the JMLR copies of the NMI
  sources; the supplemental figure and table legends, notes 1–4, the provenance note, the tables and the full methods
  sections come from these, with cross-references renamed to the Cell forms by `build_supp.py`.
- `fig/`: PNG renderings (300 dpi) of the figures; `fig2_scale_patterns` is the Figure 2 of this version (six predictor
  sets in panels d and g), from `analysis/make_fig2_patterns.py`.
- `analysis/`: the joint-baseline analysis of 26 Sept 2026: `predictor_check.py/.json` (controlled and pretrained folds,
  paired bootstraps), `family_transfer.py/.json` (pretrained folds with input-resampling intervals),
  `make_fig2_patterns.py` and its check file.

Differences from the JMLR version: Results reordered (sizing conventions and held-out prediction first, estimator
validation reduced to one paragraph), a joint predictor baseline (depth + activation norm + random-direction gain, with and
without σ₁) in Figure 2d,g, Table 2 and Table S3, a Table 2 comparing the controlled and pretrained panels, a Discussion
that ends with guidance on perturbation size, interpreting gain and downstream checks, and Cell Press structure and
naming ("supplemental", Figure S1, Table S1, related-to statements).

Open before submission: the Patterns author guidelines could not be fetched from here, so the limits assumed (Summary
150 words, Highlights 85 characters, The bigger picture 120 words, seven display items, Experimental procedures outside
the word count) should be checked on cell.com; the DSML statement; co-author sign-off; Zenodo DOI at acceptance.
