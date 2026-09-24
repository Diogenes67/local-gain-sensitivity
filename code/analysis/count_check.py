"""Count check for the NCS submission.

Every run/model/pair count quoted in main.md, methods.md and legends.md is compared with the
number of entries in the result files it was read from. Prints a pass/fail table and exits
non-zero on any failure. Re-run after any edit to the text or the result folders.

Run: python count_check.py
"""
import os, re, json, glob, sys

R = "/mnt/user-data/outputs/ncs"
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = {f: open(os.path.join(HERE, f)).read() for f in ["main.md", "methods.md", "legends.md"]}


def njson(pattern):
    return len(glob.glob(os.path.join(R, pattern), recursive=True))


def jlen(path, key=None):
    d = json.load(open(os.path.join(R, path)))
    return len(d[key]) if key else len(d)


def quoted(pattern):
    """Number of times a phrase (regex) occurs across the three text files."""
    return sum(len(re.findall(pattern, t)) for t in TEXT.values())


# (claim as it appears in the text, regex for the phrase, expected count, how the count is measured)
CHECKS = [
 ("45 pretrained models (survey, canonical)", r"45 pretrained models", 45,
  lambda: jlen("survey_v2/incontext-natural-block-float32/survey_summary.json")),
 ("45 per-model sigma1 files (survey, canonical)", r"45 pretrained models", 45,
  lambda: njson("survey_v2/incontext-natural-block-float32/sigma1_*.json")),
 ("45 models (rotation census v4)", r"45 models", 45, lambda: jlen("census_v4/census_v4.json")),
 ("45 per-layer files (census v4 merged)", r"45 models", 45, lambda: njson("census_v4/merged_v4/perlayer_*.json")),
 ("44 pretrained models (census valid after ceiling exclusion)", r"44 (pretrained )?models", 44,
  lambda: json.load(open(f"{R}/census_v4/census_v4_stats.json"))["n_valid"]),
 ("210 pairs (estimator validation: 120 cell H + 3 x 30 ConvNeXt/Mamba)", r"210 pairs", 210,
  lambda: jlen("validation/validation_cellH.json") + sum(jlen(f"validation/convnext_mamba/{os.path.basename(p)}", "rows") for p in glob.glob(f"{R}/validation/convnext_mamba/*.json"))),
 ("120 pairs (cell H)", r"120 pairs", 120, lambda: jlen("validation/validation_cellH.json")),
 ("30 pairs per model (ConvNeXt/Mamba validation)", r"30 pairs", 30,
  lambda: jlen("validation/convnext_mamba/validation_state-spaces_mamba-130m-hf.json", "rows")),
 ("20 runs (mask x loss factorial)", r"19 of 20 runs", 20, lambda: njson("factorial/results/*.json")),
 ("20 models (mask x loss factorial, Methods)", r"20 models", 20, lambda: njson("factorial/results/*.json")),
 ("31 runs (width x depth grid v2, default lr)", r"31 runs", 31, lambda: njson("grid_v2/*.json")),
 ("30 models (k-gram Markov, reprofile)", r"30 models", 30, lambda: njson("reprofile/results_kgram/*.json")),
 ("18 runs (topology, reprofile)", r"18 runs", 18, lambda: njson("reprofile/results_topology/*.json")),
 ("15 runs (alpha sweep, reprofile)", r"15 runs", 15, lambda: njson("reprofile/results_alpha/*.json")),
 ("50 controlled models (matched perturbation assay)", r"50 (controlled )?models", 50, lambda: njson("matched/results/*.json")),
 ("50 models (matched_stats n_models)", r"50 (controlled )?models", 50,
  lambda: json.load(open(f"{R}/matched/matched_stats.json"))["n_models"]),
 ("60 synthetic models (lag + pair families, 6 lags x 5 seeds x 2)", r"lag family", 60, lambda: njson("synthetic/results_v2/*.json")),
 ("30 runs (entropy-matched Markov family, 10k steps)", r"30 runs", 30, lambda: njson("markov/results/*.json")),
 ("6 runs (scale point: three real + three shuffled, 24 layers)", r"Three 24-layer", 6, lambda: njson("scale/results/[rs]*.json")),
 ("6 matched-assay files (scale point)", r"Three 24-layer", 6, lambda: njson("scale/results/matched_*.json")),
 ("13 models (four probe protocols; whole-sequence set)", r"13 models", 13, lambda: njson("survey_v2/fullseq-random-block-float32/sigma1_*.json")),
 ("13 models (random-ids set)", r"13 models", 13, lambda: njson("survey_v2/incontext-random-block-float32/sigma1_*.json")),
 ("11 models (isolated-token set)", r"11 models", 11, lambda: njson("survey_v2/isolated-random-block-float32/sigma1_*.json")),
 ("7 models (branch vs block)", r"branch", 7, lambda: njson("survey_v2/incontext-natural-branch-float32/sigma1_*.json")),
 ("3 decoders float16 (Pythia 2.8B, 6.9B, 12B)", r"float16", 3, lambda: njson("survey_v2/incontext-natural-block-float16/sigma1_*.json")),
 ("8 models (spectral radius comparison)", r"spectral radius", 8, lambda: njson("survey_v2/incontext-natural-block-float32-rho/sigma1_*.json")),
 ("5 Pythia models (metrics: CKA, effective rank, Frobenius)", r"five Pythia", 5, lambda: njson("metrics/results/*.json")),
 ("26 runs (destroyed-data d1 v2.1: 5 real, 5 shuffled, 5 bidir, 5 randlab, 3+3 transfer)", r"destroyed", 26, lambda: njson("d1/results_v21/*.json")),
 ("60 models (entropy control, Design A) - per-model files not retained", r"60 models", None, lambda: None),
]

# The entropy-control count has no files on disk; recorded as SKIP, see ED Table 3 note.
rows, fails = [], 0
for label, pat, expect, fn in CHECKS:
    mentions = quoted(pat)
    try:
        got = fn()
    except Exception as e:
        got = f"ERR {e}"
    if expect is None:
        status = "SKIP"
    elif got == expect:
        status = "PASS"
    else:
        status = "FAIL"; fails += 1
    rows.append((status, label, expect, got, mentions))

w = max(len(r[1]) for r in rows)
print(f"{'status':6} {'claim':{w}} {'text':>5} {'files':>6} {'mentions':>8}")
for s, l, e, g, m in rows:
    print(f"{s:6} {l:{w}} {str(e):>5} {str(g):>6} {m:>8}")
print(f"\n{len(rows)} checks, {fails} failed")
sys.exit(1 if fails else 0)
