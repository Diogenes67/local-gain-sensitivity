"""
validation_summary.py  --  statistics of the exact token-local validation rerun
============================================================================

Reads <results_dir>/validation/validation_<model>.json (written by survey_sigma1_v2.py --validate),
prints per-model and pooled statistics in the form the paper reports them, sets them beside the
values retained from the September 2026 three-decimal logs, and writes
<results_dir>/validation_rerun_summary.json. With --old <dir>, also matches every pair to the
retained rows (validation_cellH.json, validation_convnext_mamba_from_log.json) and reports the
largest relative difference in exact sigma_1, which is the check that the rerun reproduces the
logged run.

Usage:
  python validation_summary.py <results_dir> [--old NCS/validation]
"""
import os, sys, json, glob
import numpy as np

# values in the paper, parsed at three decimals from the printed logs (Appendix H)
REPORTED = {
    "GPT-2 124M":    dict(group="transformer"),
    "Pythia-160M":   dict(group="transformer"),
    "Pythia-410M":   dict(group="transformer"),
    "Phi-2":         dict(group="transformer"),
    "ConvNeXt-tiny": dict(group="other", median=1.8e-5, max=9.4e-5, rho_max=0.08, ratio=(1.5, 26.0)),
    "ConvNeXt-base": dict(group="other", median=2.3e-5, max=1.2e-4, rho_max=0.11, ratio=(1.6, 11.3)),
    "Mamba-130M":    dict(group="other", median=5.4e-5, max=1.0e-4, rho_max=0.09, ratio=(1.6, 8.9)),
}
REPORTED_POOLED = dict(transformers_median=7e-5, transformers_max=5e-4, pooled_max=4.8e-4, pearson=1.000,
                       rho_median=3e-3, rho_max_transformers=0.21, probes_underestimate_pct=(44, 92),
                       ratio_transformers=(1.5, 11.3), branch_absdiff_max=0.52, iters_median_range=(10, 18))

OLD_FILES = ("validation_cellH.json", "validation_convnext_mamba_from_log.json")


def load_new(results_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "validation", "validation_*.json"))):
        d = json.load(open(f))
        out[d["summary"]["label"]] = d
    return out


def per_model(d):
    rows = d["rows"]
    a = lambda k: np.array([r[k] for r in rows], dtype=float)
    s1, s1e, rho, rhoe, sb, pr, it = a("sigma1_exact"), a("sigma1_est"), a("rho_exact"), a("rho_est"), a("sigma1_branch_exact"), a("random_probe_64"), a("sigma1_est_iters")
    err = np.abs(s1e / s1 - 1)
    return dict(hf_id=d["summary"]["hf_id"], label=d["summary"]["label"], n=len(rows), d=sorted({r["d"] for r in rows}),
                blocks=d["summary"]["blocks"],
                sigma1_rel_err_median=float(np.median(err)), sigma1_rel_err_max=float(err.max()),
                pearson_r=float(np.corrcoef(s1, s1e)[0, 1]),
                rho_rel_err_median=float(np.median(np.abs(rhoe / rho - 1))), rho_rel_err_max=float(np.abs(rhoe / rho - 1).max()),
                iters_median=float(np.median(it)), iters_max=int(it.max()),
                probe_fraction_range=[float((pr / s1).min()), float((pr / s1).max())],
                probe_underestimate_pct_range=[float(100 * (1 - pr / s1).max()), float(100 * (1 - pr / s1).min())][::-1],
                sigma1_over_rho_range=[float((s1 / rho).min()), float((s1 / rho).max())],
                branch_absdiff_max=float(np.abs(sb - s1).max()),
                sigma1_range=[float(s1.min()), float(s1.max())], time_s=d["summary"].get("time_s"),
                version=d["summary"].get("version"))


def pooled(new):
    rows = [(lab, r) for lab, d in new.items() for r in d["rows"]]
    a = lambda k: np.array([r[k] for _, r in rows], dtype=float)
    s1, s1e, rho, rhoe, sb, pr = a("sigma1_exact"), a("sigma1_est"), a("rho_exact"), a("rho_est"), a("sigma1_branch_exact"), a("random_probe_64")
    err = np.abs(s1e / s1 - 1)
    grp = np.array([REPORTED.get(lab, {}).get("group", "?") for lab, _ in rows])
    tr, ot = grp == "transformer", grp == "other"
    return dict(n=len(rows), models=sorted(new), sigma1_rel_err_median=float(np.median(err)), sigma1_rel_err_max=float(err.max()),
                transformers_median=float(np.median(err[tr])) if tr.any() else None, transformers_max=float(err[tr].max()) if tr.any() else None,
                others_median=float(np.median(err[ot])) if ot.any() else None, others_max=float(err[ot].max()) if ot.any() else None,
                pearson_r=float(np.corrcoef(s1, s1e)[0, 1]),
                rho_rel_err_median=float(np.median(np.abs(rhoe / rho - 1))), rho_rel_err_max=float(np.abs(rhoe / rho - 1).max()),
                rho_rel_err_max_transformers=float(np.abs(rhoe / rho - 1)[tr].max()) if tr.any() else None,
                probes_underestimate_pct=[float(100 * (1 - pr / s1).min()), float(100 * (1 - pr / s1).max())],
                probes_fraction_transformers=[float((pr / s1)[tr].min()), float((pr / s1)[tr].max())] if tr.any() else None,
                probes_fraction_others=[float((pr / s1)[ot].min()), float((pr / s1)[ot].max())] if ot.any() else None,
                sigma1_over_rho_transformers=[float((s1 / rho)[tr].min()), float((s1 / rho)[tr].max())] if tr.any() else None,
                branch_absdiff_max=float(np.abs(sb - s1).max()),
                iters_median_by_model={lab: float(np.median([r["sigma1_est_iters"] for r in d["rows"]])) for lab, d in new.items()})


def load_old(old_dir):
    """Retained three-decimal rows keyed by (hf_id, input, block, position) -> exact sigma_1."""
    old = {}
    f = os.path.join(old_dir, "validation_cellH.json")
    if os.path.exists(f):
        for r in json.load(open(f)):
            old[(r["model"], r["inp"], r["block"], r["pos"])] = r["sigma1"]
    f = os.path.join(old_dir, "validation_convnext_mamba_from_log.json")
    if os.path.exists(f):
        for m in json.load(open(f))["models"]:
            for r in m["rows"]:
                old[(m["hf_id"], r["input"], r["block"], r["pos"])] = r["sigma1_exact"]
    return old


def compare_old(new, old):
    out = {}
    for lab, d in new.items():
        hf = d["summary"]["hf_id"]
        diffs, missing = [], 0
        for r in d["rows"]:
            k = (hf, r["input"], r["block"], r["position"])
            if k in old:
                diffs.append(abs(r["sigma1_exact"] - old[k]) / old[k])
            else:
                missing += 1
        out[lab] = dict(n_matched=len(diffs), n_unmatched=missing,
                        max_rel_diff=float(max(diffs)) if diffs else None, median_rel_diff=float(np.median(diffs)) if diffs else None)
    return out


def summarise(results_dir, old_dir=None):
    new = load_new(results_dir)
    if not new:
        print("no validation files yet in", os.path.join(results_dir, "validation")); return None
    per = {lab: per_model(d) for lab, d in new.items()}
    print(f"{'model':<14} {'n':>3} {'d':<18} {'s1 err med':>10} {'max':>8} {'r':>9} {'rho err max':>11} {'iters med':>9} {'probes %':>10} {'s1/rho':>11} {'|d branch|':>10}  reported (3 d.p. logs)")
    for lab in sorted(per, key=lambda l: list(REPORTED).index(l) if l in REPORTED else 99):
        p, rep = per[lab], REPORTED.get(lab, {})
        rs = (f"med {rep['median']:.1e} max {rep['max']:.1e} rho {rep['rho_max']:.2f} s1/rho {rep['ratio'][0]}-{rep['ratio'][1]}"
              if "median" in rep else "transformers pooled: med 7e-05 max 5e-04")
        print(f"{lab:<14} {p['n']:>3} {str(p['d']):<18} {p['sigma1_rel_err_median']:>10.1e} {p['sigma1_rel_err_max']:>8.1e} {p['pearson_r']:>9.6f} "
              f"{p['rho_rel_err_max']:>11.2f} {p['iters_median']:>9.0f} {p['probe_underestimate_pct_range'][0]:>4.0f}-{p['probe_underestimate_pct_range'][1]:<4.0f} "
              f"{p['sigma1_over_rho_range'][0]:>5.1f}-{p['sigma1_over_rho_range'][1]:<5.1f} {p['branch_absdiff_max']:>10.3f}  {rs}")
    po = pooled(new)
    R = REPORTED_POOLED
    print(f"\npooled: n = {po['n']}; sigma1 relative error median {po['sigma1_rel_err_median']:.1e}, max {po['sigma1_rel_err_max']:.1e} (reported max {R['pooled_max']:.1e}); "
          f"Pearson r {po['pearson_r']:.6f}")
    if po["transformers_median"] is not None:
        print(f"  transformers: median {po['transformers_median']:.1e}, max {po['transformers_max']:.1e} (reported {R['transformers_median']:.0e}, {R['transformers_max']:.0e})")
    if po["others_median"] is not None:
        print(f"  ConvNeXt and Mamba: median {po['others_median']:.1e}, max {po['others_max']:.1e}")
    print(f"  rho estimate: median {po['rho_rel_err_median']:.1e}, max {po['rho_rel_err_max']:.2f}"
          + (f" (transformers {po['rho_rel_err_max_transformers']:.2f}; reported median {R['rho_median']:.0e}, max {R['rho_max_transformers']})" if po['rho_rel_err_max_transformers'] is not None else ""))
    print(f"  random probes underestimate sigma1 by {po['probes_underestimate_pct'][0]:.0f}-{po['probes_underestimate_pct'][1]:.0f}% (reported {R['probes_underestimate_pct'][0]}-{R['probes_underestimate_pct'][1]}%)"
          + (f"; fraction returned: transformers {100 * po['probes_fraction_transformers'][0]:.0f}-{100 * po['probes_fraction_transformers'][1]:.0f}%" if po['probes_fraction_transformers'] else "")
          + (f", others {100 * po['probes_fraction_others'][0]:.0f}-{100 * po['probes_fraction_others'][1]:.0f}%" if po['probes_fraction_others'] else ""))
    if po["sigma1_over_rho_transformers"]:
        print(f"  sigma1/rho in the transformers {po['sigma1_over_rho_transformers'][0]:.1f}-{po['sigma1_over_rho_transformers'][1]:.1f} (reported {R['ratio_transformers'][0]}-{R['ratio_transformers'][1]})")
    print(f"  |sigma1(J-I) - sigma1(J)| max {po['branch_absdiff_max']:.3f} (reported {R['branch_absdiff_max']})")
    im = po["iters_median_by_model"]
    print(f"  per-model median iterations {min(im.values()):.0f}-{max(im.values()):.0f} (reported {R['iters_median_range'][0]}-{R['iters_median_range'][1]})")
    out = dict(per_model=per, pooled=po, reported=dict(per_model=REPORTED, pooled=REPORTED_POOLED))
    if old_dir:
        old = load_old(old_dir)
        if old:
            out["against_retained_logs"] = compare_old(new, old)
            print("\nagainst the retained three-decimal rows (relative difference in exact sigma1, per pair):")
            for lab, c in out["against_retained_logs"].items():
                print(f"  {lab:<14} matched {c['n_matched']:>3} unmatched {c['n_unmatched']:>3}  max {c['max_rel_diff']:.1e}  median {c['median_rel_diff']:.1e}" if c["n_matched"] else f"  {lab:<14} no matched pairs")
    path = os.path.join(results_dir, "validation_rerun_summary.json")
    json.dump(out, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)
    print("\nsaved", path)
    return out


if __name__ == "__main__":
    rd = sys.argv[1]
    old = sys.argv[sys.argv.index("--old") + 1] if "--old" in sys.argv else None
    summarise(rd, old)
