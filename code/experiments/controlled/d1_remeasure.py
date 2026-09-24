#!/usr/bin/env python3
"""
Re-measure d1 v2 checkpoints with BRANCH rotation (18 Sept 2026)
================================================================
d1_v2.py trained with whole-block rotation as its sensitivity readout. The original destroyed-data,
k-gram and census experiments rotated the attention and feed-forward BRANCH outputs before the
residual add (d3_ngram_sweep.forward_with_intervention), and whole-block rotation saturates at the
loss ceiling in trained models. This script loads every saved checkpoint in $D1_OUT/ckpt and adds:

  regions_branch      waist / early / late branch rotation at doses 0.25, 0.5, 1.0, bootstrap CI
  per_block_branch    single-block branch rotation, dose 1
  skip_regions        waist / early / late identity skip
  (per_block_skip is already in the d1_v2 JSON; recomputed here for completeness)

Same evaluation data as d1_v2 (regenerated from the same seeds), same readout rule (own objective,
randlab read out on real text). Writes <cond>_s<seed>.json in $D1_OUT/branch/. No training.
Needs d1_v2.py, factorial_v2.py and survey_sigma1_v2.py beside it.
"""

import os, sys, json, time, datetime
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factorial_v2 as fv
import d1_v2 as d1

DEVICE = fv.DEVICE
OUT = Path(os.environ.get("D1_OUT", "d1_v2")); CKPT = OUT / "ckpt"; BR = OUT / "branch"
DOSES = [0.25, 0.5, 1.0]
log = fv.log


def dosed(Q, dose):
    if dose >= 1.0:
        return Q
    M = (1 - dose) * torch.eye(Q.shape[0]) + dose * Q
    Qd, R = torch.linalg.qr(M)
    return Qd * torch.sign(torch.diag(R))[None, :]


def rotate_branches(model, blocks, Qa, Qf, dose=1.0):
    def setup(hk):
        for li in blocks:
            A = dosed(Qa[li], dose).to(DEVICE); Fq = dosed(Qf[li], dose).to(DEVICE)
            hk.add(model.layers[li].attn.register_forward_hook(lambda m, a, o, Q=A: (o[0] @ Q.T,) + tuple(o[1:])))
            hk.add(model.layers[li].ff.register_forward_hook(lambda m, a, o, Q=Fq: o @ Q.T))
    return setup


def skip_blocks(model, blocks):
    def setup(hk):
        for li in blocks:
            hk.add(model.layers[li].register_forward_hook(lambda m, a, o: a[0]))
    return setup


def main():
    BR.mkdir(parents=True, exist_ok=True)
    data = fv.load_data()
    N_VAL, N_EVAL, BATCH = d1.N_VAL, d1.N_EVAL, d1.BATCH
    real_eval = data[-(N_EVAL * BATCH):]
    ckpts = sorted(CKPT.glob("*.pt"))
    log(f"re-measuring {len(ckpts)} checkpoints with branch rotation -> {BR}")
    for i, ck in enumerate(ckpts):
        name = ck.stem; cond, s = name.rsplit("_s", 1); seed = int(s)
        out = BR / f"{name}.json"
        if out.exists():
            log(f"[{i + 1}/{len(ckpts)}] {name}: done, skipping"); continue
        t0 = time.time()
        srcp = OUT / f"{name}.json"
        src = json.load(open(srcp)) if srcp.exists() else {}
        phase = src.get("final_phase") or d1.phase_condition(cond, d1.STEPS - 1); rp = "real" if phase == "randlab" else phase
        mask = d1.mask_for(rp)
        ev = d1.shuffle_within(real_eval, 500 + seed) if rp == "shuffled" else real_eval
        model = fv.Transformer().to(DEVICE); model.load_state_dict(torch.load(ck, map_location=DEVICE)); model.eval()
        base = d1.eval_losses(model, rp, ev, mask)
        Qa = {li: d1.haar(fv.D_MODEL, 9100 + 17 * li + seed) for li in range(fv.N_LAYERS)}
        Qf = {li: d1.haar(fv.D_MODEL, 9200 + 17 * li + seed) for li in range(fv.N_LAYERS)}
        regions = {}
        for rn, blk in (("waist", d1.WAIST), ("early", d1.EARLY), ("late", d1.LATE)):
            for dose in DOSES:
                regions[f"{rn}@{dose}"] = d1.dstat(d1.eval_losses(model, rp, ev, mask, rotate_branches(model, blk, Qa, Qf, dose)), base)
            regions[rn] = regions[f"{rn}@1.0"]
        skip_regions = {rn: d1.dstat(d1.eval_losses(model, rp, ev, mask, skip_blocks(model, blk)), base)
                        for rn, blk in (("waist", d1.WAIST), ("early", d1.EARLY), ("late", d1.LATE))}
        per_block = [d1.dstat(d1.eval_losses(model, rp, ev, mask, rotate_branches(model, [li], Qa, Qf)), base)["dL"] for li in range(fv.N_LAYERS)]
        per_skip = [d1.dstat(d1.eval_losses(model, rp, ev, mask, d1.skip(model, li)), base)["dL"] for li in range(fv.N_LAYERS)]
        edge = max(regions["early"]["dL"], regions["late"]["dL"], 1e-9); sedge = max(skip_regions["early"]["dL"], skip_regions["late"]["dL"], 1e-9)
        res = dict(condition=cond, seed=seed, final_phase=phase, readout=src.get("readout"), baseline_eval_loss=float(np.mean(base)),
                   intervention="Haar rotation of attention and feed-forward branch outputs before the residual add (the d1/k-gram/census intervention); identity skip",
                   regions_branch=regions, waist_over_edge_branch=regions["waist"]["dL"] / edge,
                   skip_regions=skip_regions, skip_waist_over_edge=skip_regions["waist"]["dL"] / sedge,
                   per_block_branch_dL=per_block, per_block_skip_dL=per_skip,
                   R_ex0=src.get("R_ex0", float("nan")), early_over_mid=src.get("early_over_mid", float("nan")), late_over_mid=src.get("late_over_mid", float("nan")),
                   contrast_C=src.get("contrast_C", float("nan")), two_arm_rule=src.get("two_arm_rule", False),
                   checkpoint=str(ck), time_s=round(time.time() - t0), date=datetime.datetime.now().isoformat(timespec="seconds"))
        tmp = out.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1, default=fv._ser); tmp.replace(out)
        log(f"[{i + 1}/{len(ckpts)}] {name}: base {np.mean(base):.3f}  branch@1 waist {regions['waist']['dL']:.3f} early {regions['early']['dL']:.3f} late {regions['late']['dL']:.3f} (w/e {regions['waist']['dL'] / edge:.2f})"
            f"  branch@0.25 waist {regions['waist@0.25']['dL']:.3f}  skip waist {skip_regions['waist']['dL']:.3f} (w/e {skip_regions['waist']['dL'] / sedge:.2f})  [{time.time() - t0:.0f} s]")
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    summary()


def summary():
    rows = [json.load(open(f)) for f in sorted(BR.glob("*_s*.json"))]
    if not rows: return
    print("\n%-10s%3s %7s | %7s %7s %7s %6s | %8s | %7s %6s | %6s %5s %6s" % ("cond", "n", "base", "br w", "br e", "br l", "w/e", "br.25 w", "skip w", "w/e", "R_ex0", "two", "C"))
    for c in d1.CONDS:
        rs = [r for r in rows if r["condition"] == c]
        if not rs: continue
        g = lambda k: np.mean([r["regions_branch"][k]["dL"] for r in rs])
        print("%-10s%3d %7.3f | %7.3f %7.3f %7.3f %6.2f | %8.3f | %7.3f %6.2f | %6.3f %3d/%d %6.3f" % (
            c, len(rs), np.mean([r["baseline_eval_loss"] for r in rs]), g("waist"), g("early"), g("late"), np.mean([r["waist_over_edge_branch"] for r in rs]),
            g("waist@0.25"), np.mean([r["skip_regions"]["waist"]["dL"] for r in rs]), np.mean([r["skip_waist_over_edge"] for r in rs]),
            np.mean([r["R_ex0"] for r in rs]), sum(r["two_arm_rule"] for r in rs), len(rs), np.mean([r["contrast_C"] for r in rs])))


if __name__ == "__main__":
    main()
