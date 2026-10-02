#!/bin/bash
# Build the Patterns submission package: main_patterns.{docx,pdf} and supplement_patterns.{docx,pdf} (Document S1).
# Needs pandoc (>= 3, with citeproc), xelatex, python3. Run from anywhere: bash build.sh
set -e
cd "$(dirname "$0")"
python3 build_supp.py

# figure pages appended to the main text (initial submission: figures embedded after the tables)
cat > src/main_figs.md <<'EOF'

## Figures

![](fig/fig1_jmlr.png){width=100%}

Figure 1

\newpage

![](fig/fig2_scale_patterns.png){width=100%}

Figure 2

\newpage

![](fig/fig3_direction_nmi.png){width=100%}

Figure 3

\newpage

![](fig/fig4_data_nmi.png){width=100%}

Figure 4

\newpage

![](fig/fig5_pretrained_nmi.png){width=100%}

Figure 5
EOF

cat > src/header_main.tex <<'EOF'
\usepackage{pdflscape}
\usepackage{lineno}
\linenumbers
\usepackage{setspace}
\onehalfspacing
\setlength{\emergencystretch}{3em}
\usepackage{longtable,booktabs}
\let\oldlongtable\longtable
EOF
cat > src/header_supp.tex <<'EOF'
\usepackage{pdflscape}
\pagenumbering{gobble}
\setlength{\emergencystretch}{3em}
EOF

COMMON="--from markdown+superscript+subscript+pipe_tables+raw_tex --resource-path=.:src:fig"
PDFOPTS=(--pdf-engine=xelatex -V "mainfont=DejaVu Sans" -V "mathfont=DejaVu Math TeX Gyre" -V geometry:margin=2.2cm -V papersize=a4 -V colorlinks=false)

# main text: docx and PDF (the PDF is a review copy with line numbers); both carry the five figures after the tables
cat src/main_patterns.md src/main_figs.md > src/_main_build.md
pandoc src/_main_build.md $COMMON --citeproc -o main_patterns.docx
pandoc src/_main_build.md $COMMON --citeproc "${PDFOPTS[@]}" -V fontsize=11pt -H src/header_main.tex -o main_patterns.pdf
rm src/_main_build.md

# Document S1
pandoc src/supplement_patterns.md $COMMON -o supplement_patterns.docx
pandoc src/supplement_patterns.md $COMMON "${PDFOPTS[@]}" -V fontsize=10pt -H src/header_supp.tex -o supplement_patterns.pdf

for f in main_patterns.pdf supplement_patterns.pdf; do echo "$f: $(pdfinfo $f | grep Pages)"; done
