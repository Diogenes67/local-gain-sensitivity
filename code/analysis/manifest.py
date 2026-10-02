"""Frozen results manifest for the NCS submission.

Walks the result folders that the figures, tables and text are drawn from, records every
JSON/CSV/log file with its size and SHA-256, and writes results_manifest.json and
results_manifest.md next to this script. Re-run after any result folder changes; the
manifest is the record of which files the numbers were read from.

Run: python manifest.py
"""
import os, json, hashlib, glob, datetime

R = "/mnt/user-data/outputs/ncs"
MAC = "/mnt/user-data/uploads/Desktop/Interesting/PCI"
HERE = os.path.dirname(os.path.abspath(__file__))

# (label, root, glob patterns, used by)
SETS = [
 ("Survey v2 (canonical + protocols, precision, rho)", f"{R}/survey_v2", ["**/*.json", "**/*.csv"], "Fig. 1, Fig. 5, ED Fig. 2, ED Table 1; make_figs2.py, make_edfigs.py, make_tables.py"),
 ("Estimator validation (exact SVD, cell H; ConvNeXt/Mamba)", f"{R}/validation", ["*.json", "**/*.json", "*.log"], "Fig. 1b, Methods; parse.py"),
 ("Metrics (CKA, effective rank, Frobenius)", f"{R}/metrics/results", ["*.json"], "ED Fig. 3; make_edfigs.py"),
 ("Destroyed-data v2.1 (d1)", f"{R}/d1", ["results*/**/*.json"], "Fig. 2, ED Fig. 6; make_main23.py, make_figs.py"),
 ("Mask x loss factorial v2", f"{R}/factorial", ["results/**/*.json"], "ED Fig. 9a; make_main23.py"),
 ("Width x depth grid v2", f"{R}/grid_v2", ["*.json"], "ED Fig. 9b; make_main23.py"),
 ("Width x depth grid (March reconstruction)", f"{R}/grid", ["*.json", "*.txt"], "ED Fig. 9b (March seeds); make_main23.py"),
 ("Reprofile v2 (topology, alpha, k-gram)", f"{R}/reprofile", ["results_*/**/*.json"], "Fig. 3, ED Figs. 5 and 10; make_main23.py, make_figs.py"),
 ("Intervention comparison", f"{R}/intervention", ["results/**/*.json"], "Methods (rotation vs skip); intervention_compare.py"),
 ("Synthetic lag family", f"{R}/synthetic", ["results_v2/**/*.json"], "Results (lag family); synthetic_context.py"),
 ("Census v4 (pretrained rotation)", f"{R}/census_v4", ["results/**/*.json", "merged_v4/**/*.json", "*.json", "*.csv"], "Fig. 5, ED Fig. 6, ED Table 2; census_merge_v4.py"),
 ("Matched perturbation assay", f"{R}/matched", ["results/**/*.json", "matched_stats.json"], "Fig. 4 (incl. perturbation-scaling check), ED Fig. 7, SN 17; matched_analysis.py"),
 ("Scale point (24 layers)", f"{R}/scale", ["results/**/*.json", "scale_stats.json"], "Fig. 2e–g; scale_analysis.py, make_main23.py"),
 ("Matched assay sampling check (12 and 24 layers)", R, ["matched/sampling/*.json", "scale/sampling/*.json"], "Methods (Sampling check); sampling_analysis.py"),
 ("Downstream linearised response pilot", R, ["matched/linresp/*.json"], "Methods (Downstream linearised response); linresp_analysis.py"),
 ("Entropy-matched Markov family (10k steps)", f"{R}/markov", ["results/**/*.json", "markov_stats.json"], "Fig. 3d–f; markov_analysis.py, make_main23.py"),
 ("Aubry comparison", f"{R}/aubry", ["results*/**/*.json"], "Methods (Aubry et al.); aubry_compare.py"),
 ("Per-layer scramble (ConvNeXt, Mamba)", f"{R}/scramble/v2_results", ["*.json"], "ED Table 2 notes"),
 ("March 2026 files used for ED Figs. 4 and 8 (from the Mac)", f"{MAC}/results", [
     "b1v2_seed*_trajectory.json", "b1_duration_comparison.txt", "b1_billion_scale.txt", "b1_5seed_factorial.txt",
     "s2_developmental/*.json", "exp2a_bracket_seed*.json", "d3l_alignment_adaptation/d3l_aggregate.json",
     "exp4a_singular_perturbation/*.json", "d3r_residual_gain_stress/d3r_aggregate.json", "b1_objective_factorial/*.json",
     "d3h_low_dose/d3h_summary.json", "s6_covariates/s6_combined.json", "d3_ngram/d3b_entropy_report.json"], "ED Fig. 4, ED Fig. 8, ED Table 3; make_edfigs.py"),
 ("March 2026 image reproduced", MAC, ["mamba_validation_figure.png"], "ED Fig. 1"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for label, root, pats, used in SETS:
    files = set()
    for pat in pats:
        files.update(glob.glob(os.path.join(root, pat), recursive=True))
    files = sorted(f for f in files if os.path.isfile(f))
    for f in files:
        rows.append(dict(set=label, path=os.path.relpath(f, "/mnt/user-data"), bytes=os.path.getsize(f), sha256=sha(f), used_by=used))
    print(f"{label}: {len(files)} files")

out = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), n_files=len(rows), files=rows)
json.dump(out, open(os.path.join(HERE, "results_manifest.json"), "w"), indent=1)
with open(os.path.join(HERE, "results_manifest.md"), "w") as fh:
    fh.write(f"# Results manifest\n\nGenerated {out['generated']}; {len(rows)} files. Paths are relative to /mnt/user-data (outputs/ncs = the NCS working folder on the Mac; uploads/Desktop/Interesting/PCI = the March 2026 folder).\n\n")
    cur = None
    for r in rows:
        if r["set"] != cur:
            cur = r["set"]; fh.write(f"\n## {cur}\n\nUsed by: {r['used_by']}\n\n| file | bytes | sha256 |\n|---|---:|---|\n")
        fh.write(f"| {r['path']} | {r['bytes']} | {r['sha256'][:16]}… |\n")
print("wrote results_manifest.json / .md", len(rows))
