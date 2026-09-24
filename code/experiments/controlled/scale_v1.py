#!/usr/bin/env python3
"""
Scale point in depth: the real-text against shuffled-text contrast and the matched assay at 24 layers
=====================================================================================================
The causal claims of the paper rest on 12-layer models. This experiment repeats the central contrast
at twice the depth with everything else the d1 v2.1 setup (d = 512, 8 heads, untied 50,257-token head,
WikiText-103, AdamW, 10,000 steps, batch 16 x 128), three seeds of real text and three of shuffled text,
and then runs the matched token-local perturbation assay on the six checkpoints.

Measured per model (d1_v2.run, unchanged code with the region lists rescaled to 24 blocks): canonical
sigma_1 profile, R_ex0 and two-arm rule; branch rotation of the middle third (blocks 8-15) at doses
0.25/0.5/1, the early (0-7) and late (16-23) thirds, whole-block rotation, identity skipping by region
and per block, profile trajectory at steps 1,000/3,000/5,000, fp16 checkpoint. Then matched_perturbation
.run_job on each checkpoint: sigma_1 and v_1 per block, gain and downstream KL / dCE along v_1 and along
random directions at five doses (per-block JSON, as the 50-model assay).

Outputs (SCALE_OUT on Drive): d1/{cond}_s{seed}.json + d1/ckpt/*.pt, matched/results/matched_d1_*.json,
scale_v1_results.zip (both sets). Resume-safe: finished runs are skipped; a disconnect costs one run.

  python scale_v1.py               6 training runs + 6 matched assays
  python scale_v1.py --plumbing    tiny models on random data, end to end

Needs survey_sigma1_v2.py, factorial_v2.py, d1_v2.py, reprofile_v2.py and matched_perturbation.py beside it.
Environment: SCALE_OUT, D1_DATA (the tokenised WikiText-103 cache), SCALE_LAYERS (default 24).
"""
import os, sys, json, time, shutil, datetime, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv
import factorial_v2 as fv
import d1_v2 as d1
import matched_perturbation as mp

DEVICE = fv.DEVICE
ROOT = Path(os.environ.get("SCALE_OUT", "scale_v1"))
N_LAYERS = int(os.environ.get("SCALE_LAYERS", 24))
CONDS = {"real": 3, "shuffled": 3}
log = d1.log


def configure(n_layers, plumbing=False):
    """Point the d1 and matched modules at this experiment: depth, regions, output folders."""
    global N_LAYERS
    N_LAYERS = n_layers; fv.N_LAYERS = n_layers
    t = n_layers // 3
    d1.EARLY, d1.WAIST, d1.LATE = list(range(0, t)), list(range(t, 2 * t)), list(range(2 * t, n_layers))
    d1.OUT = ROOT / "d1"; d1.CKPT = d1.OUT / "ckpt"; d1.CONDS = dict(CONDS)
    mp.D1_CKPT = d1.CKPT; mp.OUT = ROOT / "matched"
    if plumbing:
        fv.D_MODEL, fv.N_HEADS, fv.VOCAB, fv.EOS = 64, 4, 2000, 1999
        d1.STEPS = 20; d1.PROFILE_AT = [10]; d1.N_EVAL = 3; d1.N_VAL = 32
        mp.POSITIONS = mp.POSITIONS[:2]; mp.N_INPUTS = 2
    log(f"scale v1: {n_layers} layers, regions early {d1.EARLY[0]}-{d1.EARLY[-1]} middle {d1.WAIST[0]}-{d1.WAIST[-1]} late {d1.LATE[0]}-{d1.LATE[-1]}; "
        f"steps {d1.STEPS}; out {ROOT}")


def train_all(real_train, real_val, real_eval, probe_ids):
    d1.OUT.mkdir(parents=True, exist_ok=True)
    todo = [(c, s) for s in range(max(CONDS.values())) for c, n in CONDS.items() if s < n]
    for i, (c, s) in enumerate(todo):
        path = d1.OUT / f"{c}_s{s}.json"
        if path.exists():
            try:
                d = json.load(open(path)); assert "sigma1_profile" in d and d["n_layers"] == N_LAYERS and d["steps"] == d1.STEPS
                log(f"[{i + 1}/{len(todo)}] {c} s{s}: done (R {d['R_ex0']:.3f}, middle dL {d['regions']['waist']['dL']:.3f}), skipping"); continue
            except Exception as e:
                log(f"  {e}"); path.rename(path.with_suffix(".old.json"))
        log(f"\n[{i + 1}/{len(todo)}] {c} seed {s} ({N_LAYERS} layers)")
        res = d1.run(c, s, real_train, real_val, real_eval, probe_ids)
        res["experiment"] = "scale_v1"; res["n_layers"] = N_LAYERS
        tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1, default=fv._ser); tmp.replace(path)
        if DEVICE == "cuda": torch.cuda.empty_cache()


def matched_all(real_eval):
    res_dir = mp.OUT / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    mp._D1_DATA["eval"] = real_eval.clone()
    for c, n in CONDS.items():
        for s in range(n):
            mp.run_job("d1", c, s, res_dir, quick=False)
    return res_dir


def summary():
    rows = []
    for f in sorted((ROOT / "d1").glob("*_s*.json")):
        if f.name.endswith(".old.json"): continue
        d = json.load(open(f)); m = ROOT / "matched" / "results" / f"matched_d1_{d['condition']}_s{d['seed']}.json"
        ms = json.load(open(m))["summary"] if m.exists() else {}
        rows.append((d["condition"], d["seed"], d["n_layers"], d["baseline_eval_loss"], d["R_ex0"], d["early_over_mid"], d["late_over_mid"], d["two_arm_rule"],
                     d["regions"]["waist"]["dL"], d["regions"]["early"]["dL"], d["regions"]["late"]["dL"], d["skip_regions"]["waist"]["dL"],
                     ms.get("R_ex0_sigma1", float("nan")), ms.get("waist_kl_seq_v1_eps0.1", float("nan")), ms.get("spearman_sigma1_vs_kl_seq_v1_eps0.1_ex0", float("nan"))))
    print("\n%-9s%2s %3s %6s | %6s %5s %5s %4s | %7s %7s %7s | %7s | %7s %10s %7s" % ("cond", "s", "L", "eval", "R_ex0", "E/M", "L/M", "two", "rot mid", "rot ely", "rot late", "skip mid", "R(mtch)", "KL mid v1", "rho blk"))
    for r in rows:
        print("%-9s%2d %3d %6.3f | %6.3f %5.2f %5.2f %4s | %7.3f %7.3f %7.3f | %7.3f | %7.3f %10.3e %7.2f" % r)
    return rows


def export():
    stage = ROOT / "scale_v1_json"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in list((ROOT / "d1").glob("*.json")) + list((ROOT / "matched" / "results").glob("*.json")):
        shutil.copy(f, stage / f.name)
    z = shutil.make_archive(str(ROOT / "scale_v1_results"), "zip", str(stage))
    here = Path.cwd() / "scale_v1_results.zip"
    if Path(z).resolve() != here.resolve(): shutil.copy(z, here)
    log(f"results zip: {z} ({os.path.getsize(z) / 1e6:.2f} MB) and {here}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--plumbing", action="store_true"); args = ap.parse_args(argv)
    configure(N_LAYERS if not args.plumbing else 6, plumbing=args.plumbing)
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    t0 = time.time()
    if args.plumbing:
        torch.manual_seed(0)
        data = torch.randint(0, 1000, (400, fv.SEQ))
        real_train, real_val, real_eval = data[:300], data[300:332], data[-(d1.N_EVAL * fv.BATCH):]
        probe_ids = data[:2]
    else:
        data = fv.load_data()
        real_train, real_val, real_eval = data[:-(d1.N_VAL + d1.N_EVAL * fv.BATCH)], data[-(d1.N_VAL + d1.N_EVAL * fv.BATCH):-(d1.N_EVAL * fv.BATCH)], data[-(d1.N_EVAL * fv.BATCH):]
        from transformers import AutoTokenizer
        probe_ids = sv.natural_text_ids(AutoTokenizer.from_pretrained("gpt2"), 5)
    train_all(real_train, real_val, real_eval, probe_ids)
    matched_all(real_eval)
    log(f"\nall done in {(time.time() - t0) / 3600:.2f} h"); summary(); export()


if __name__ == "__main__":
    main()
