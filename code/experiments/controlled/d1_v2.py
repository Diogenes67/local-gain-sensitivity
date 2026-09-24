#!/usr/bin/env python3
"""
Destroyed-data experiment, v2.1 (19 Sept 2026)
==============================================
Re-run of d1 with the canonical estimator, BRANCH rotation sensitivity (the d1/k-gram/census
intervention: attention and feed-forward outputs rotated before the residual add, doses
0.25/0.5/1), whole-block rotation as a saturating comparison, identity skip, and checkpoints
saved. v2 (18 Sept) measured whole-block rotation only, which saturates; its checkpoints were
lost with the instance. The results zip is written automatically at the end of the run. Same 12-layer d=512 architecture, optimiser and schedule as factorial_v2 (imported).

Conditions (five seeds each unless noted):
  real       causal, next-token loss, WikiText-103
  shuffled   causal, next-token loss, tokens permuted once within each sequence (unigram
             statistics preserved, order destroyed; permutation fixed per sequence)
  randlab    causal, next-token loss, real inputs with every target replaced by a uniformly
             random token resampled every batch
  bidir      no mask, masked-token loss on real text (15% corruption, 80/10/10)
  shuf2real  shuffled for 5,000 steps then real for 5,000 (3 seeds; optimiser state kept)
  real2shuf  the reverse (3 seeds)

Per model, at steps 1,000, 3,000, 5,000 and 10,000: sigma1 profile through survey_sigma1_v2's
estimator (5 validation inputs x 8 positions, condition mask), pooled and two-arm rules, and
spectral contrast C as d1 defined it; interior sensitivity as dL under branch rotation of blocks
4-7 at dose 1 (whole-block dose 1 alongside). At 10,000 additionally waist/early/late regions
under branch rotation at doses 0.25, 0.5 and 1, whole-block rotation at dose 1, identity skip by
region, and per-block branch rotation and skip, all with bootstrap CIs over 50 x 16 sequences.

Readout for sensitivity: the model's own objective on its own held-out distribution (shuffled
models on shuffled text, bidir on masked real text), except randlab, whose objective is
unlearnable and which is read out with next-token cross-entropy on real text.

Final weights saved to $D1_OUT/ckpt/<cond>_s<seed>.pt so nothing here needs retraining to be
re-profiled. Resume-safe. Needs survey_sigma1_v2.py and factorial_v2.py beside it.
"""

import os, sys, json, time, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factorial_v2 as fv
import survey_sigma1_v2 as sv

DEVICE = fv.DEVICE
OUT = Path(os.environ.get("D1_OUT", "d1_v21"))
CKPT = OUT / "ckpt"
V, SEQ, BATCH, STEPS = fv.VOCAB, fv.SEQ, fv.BATCH, fv.STEPS
N_VAL, N_EVAL = 256, 50
SWITCH = 5000
PROFILE_AT = [1000, 3000, 5000]
CONDS = {"real": 5, "shuffled": 5, "randlab": 5, "bidir": 5, "shuf2real": 3, "real2shuf": 3}
WAIST, EARLY, LATE = [4, 5, 6, 7], [0, 1, 2, 3], [8, 9, 10, 11]
log = fv.log


# ---------------------------------------------------------------- data variants
def shuffle_within(x, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.argsort(torch.rand(x.shape, generator=g), dim=1)
    return torch.gather(x, 1, idx)


def phase_condition(cond, step):
    if cond == "shuf2real":
        return "shuffled" if step < SWITCH else "real"
    if cond == "real2shuf":
        return "real" if step < SWITCH else "shuffled"
    return cond


def loss_for(phase, model, ids, mask, gen):
    if phase in ("real", "shuffled"):
        return fv.loss_for("AR", model, ids, mask, gen)
    if phase == "bidir":
        return fv.loss_for("MLM", model, ids, mask, gen)
    if phase == "randlab":
        logits = model(ids, mask)
        tgt = torch.randint(0, V, ids[:, 1:].shape, generator=gen, device=ids.device)
        return F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt.reshape(-1))
    raise ValueError(phase)


def mask_for(phase):
    return fv.mask_for("MLM" if phase == "bidir" else "AR", SEQ, DEVICE)


# ---------------------------------------------------------------- perturbation
def haar(d, seed):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=g))
    return Q * torch.sign(torch.diag(R))[None, :]


class Hooks:
    def __init__(self): self.h = []
    def add(self, x): self.h.append(x)
    def clear(self):
        for x in self.h: x.remove()
        self.h.clear()


@torch.no_grad()
def eval_losses(model, phase, data, mask, setup=None):
    model.eval(); hk = Hooks(); gen = torch.Generator(device=DEVICE).manual_seed(777)
    try:
        if setup: setup(hk)
        out = []
        for b in range(min(N_EVAL, len(data) // BATCH)):
            ids = data[b * BATCH:(b + 1) * BATCH].to(DEVICE)
            out.append(loss_for(phase, model, ids, mask, gen).item())
        return out
    finally:
        hk.clear(); model.train()


def rotate(model, blocks, Qs):
    """whole-block: rotate the residual state at the block output (saturates in trained models; comparison only)"""
    def setup(hk):
        for li in blocks:
            hk.add(model.layers[li].register_forward_hook(lambda m, a, o, Q=Qs[li].to(DEVICE): o @ Q.T))
    return setup


def dosed(Q, dose):
    if dose >= 1.0:
        return Q
    M = (1 - dose) * torch.eye(Q.shape[0]) + dose * Q
    Qd, R = torch.linalg.qr(M)
    return Qd * torch.sign(torch.diag(R))[None, :]


def rotate_branches(model, blocks, Qa, Qf, dose=1.0):
    """branch: rotate the attention output and the feed-forward output before each residual add
    (the intervention of the original d1, the k-gram sweep and the census)"""
    def setup(hk):
        for li in blocks:
            A = dosed(Qa[li], dose).to(DEVICE); Fq = dosed(Qf[li], dose).to(DEVICE)
            hk.add(model.layers[li].attn.register_forward_hook(lambda m, a, o, Q=A: (o[0] @ Q.T,) + tuple(o[1:])))
            hk.add(model.layers[li].ff.register_forward_hook(lambda m, a, o, Q=Fq: o @ Q.T))
    return setup


def skip(model, li):
    def setup(hk):
        hk.add(model.layers[li].register_forward_hook(lambda m, a, o: a[0]))
    return setup


def skip_blocks(model, blocks):
    def setup(hk):
        for li in blocks:
            hk.add(model.layers[li].register_forward_hook(lambda m, a, o: a[0]))
    return setup


def dstat(losses, base):
    a, b = np.array(losses), np.array(base); rng = np.random.default_rng(0); n = len(a)
    boots = [a[i].mean() - b[i].mean() for i in (rng.integers(0, n, n) for _ in range(500))]
    return dict(dL=float(a.mean() - b.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def contrast_C(prof):
    """d1's spectral contrast: (mean edge sigma1 - mean interior sigma1) / mean sigma1, thirds over the 12 blocks."""
    p = np.asarray(prof, float); L = len(p); n = L // 3
    edge = np.concatenate([p[:n], p[L - n:]]).mean(); mid = p[n:L - n].mean()
    return float((edge - mid) / p.mean())


def two_arm(prof):
    p = np.asarray(prof, float)[1:]; L = len(p); n = L // 3
    e, m, l = p[:n].mean(), p[n:L - n].mean(), p[L - n:].mean(); R = m / ((e + l) / 2)
    return dict(R_ex0=float(R), early_over_mid=float(e / m), late_over_mid=float(l / m),
                pooled_rule=bool(R < 0.80 and l / m > 1), two_arm_rule=bool(R < 0.80 and e > m and l > m), contrast_C=contrast_C(prof))


# ---------------------------------------------------------------- one run
def run(cond, seed, real_train, real_val, real_eval, probe_ids):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(20_000 + seed)
    shuf_train = shuffle_within(real_train, 300 + seed)
    shuf_val, shuf_eval = shuffle_within(real_val, 400 + seed), shuffle_within(real_eval, 500 + seed)
    model = fv.Transformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=fv.LR, weight_decay=fv.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / fv.WARMUP))
    order = torch.randperm(len(real_train), generator=torch.Generator().manual_seed(seed))
    log(f"  {cond} seed {seed}: {n_params / 1e6:.1f}M params")

    def eval_set(phase):
        return (shuf_val, shuf_eval) if phase == "shuffled" else (real_val, real_eval)

    def readout_phase(phase):
        return "real" if phase == "randlab" else phase

    Qs = {li: haar(fv.D_MODEL, 9000 + 17 * li + seed) for li in range(fv.N_LAYERS)}
    Qa = {li: haar(fv.D_MODEL, 9100 + 17 * li + seed) for li in range(fv.N_LAYERS)}
    Qf = {li: haar(fv.D_MODEL, 9200 + 17 * li + seed) for li in range(fv.N_LAYERS)}
    traj = {}

    def checkpoint_measure(step):
        phase = phase_condition(cond, step - 1 if step > 0 else 0)
        rp = readout_phase(phase); mask = mask_for(rp); _, ev = eval_set(rp)
        prof, _, _ = fv.survey_profile(model, mask_for(phase), probe_ids)
        base = eval_losses(model, rp, ev, mask)
        wb = dstat(eval_losses(model, rp, ev, mask, rotate_branches(model, WAIST, Qa, Qf)), base)
        wk = dstat(eval_losses(model, rp, ev, mask, rotate(model, WAIST, Qs)), base)
        s = two_arm(prof)
        traj[step] = dict(**s, waist_dL_branch=wb["dL"], waist_ci_branch=wb["ci"], waist_dL_block=wk["dL"], baseline=float(np.mean(base)), profile=[round(float(x), 4) for x in prof])
        log(f"    step {step}: R {s['R_ex0']:.3f} E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f} C {s['contrast_C']:.3f}  waist dL branch {wb['dL']:.3f} (block {wk['dL']:.3f})  base {np.mean(base):.3f}")

    model.train(); step = 0; last = None
    while step < STEPS:
        for i in range(0, len(order) - BATCH + 1, BATCH):
            if step >= STEPS: break
            if step in PROFILE_AT:
                checkpoint_measure(step)
            phase = phase_condition(cond, step)
            src = shuf_train if phase == "shuffled" else real_train
            ids = src[order[i:i + BATCH]].to(DEVICE)
            loss = loss_for(phase, model, ids, mask_for(phase), gen)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fv.CLIP); opt.step(); sched.step()
            step += 1; last = loss.item()
            if step % 2500 == 0:
                log(f"    step {step}/{STEPS} loss {last:.3f} ({time.time() - t0:.0f} s)")

    # final measurements
    phase = phase_condition(cond, STEPS - 1); rp = readout_phase(phase); mask = mask_for(rp); val, ev = eval_set(rp)
    vloss = float(np.mean(eval_losses(model, rp, val[:N_VAL], mask)))
    prof, samples, _ = fv.survey_profile(model, mask_for(phase), probe_ids)
    s = two_arm(prof)
    base = eval_losses(model, rp, ev, mask)
    REG = (("waist", WAIST), ("early", EARLY), ("late", LATE))
    regions = {}
    for n, blk in REG:
        for dose in (0.25, 0.5, 1.0):
            regions[f"{n}@{dose}"] = dstat(eval_losses(model, rp, ev, mask, rotate_branches(model, blk, Qa, Qf, dose)), base)
        regions[n] = regions[f"{n}@1.0"]
    regions_block = {n: dstat(eval_losses(model, rp, ev, mask, rotate(model, blk, Qs)), base) for n, blk in REG}
    skip_regions = {n: dstat(eval_losses(model, rp, ev, mask, skip_blocks(model, blk)), base) for n, blk in REG}
    per_rot = [dstat(eval_losses(model, rp, ev, mask, rotate_branches(model, [li], Qa, Qf)), base)["dL"] for li in range(fv.N_LAYERS)]
    per_skip = [dstat(eval_losses(model, rp, ev, mask, skip(model, li)), base)["dL"] for li in range(fv.N_LAYERS)]
    edge = max(regions["early"]["dL"], regions["late"]["dL"], 1e-9); sedge = max(skip_regions["early"]["dL"], skip_regions["late"]["dL"], 1e-9)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}, CKPT / f"{cond}_s{seed}.pt")  # fp16, ~180 MB
    log(f"    -> val {vloss:.3f}  R {s['R_ex0']:.3f} E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f} C {s['contrast_C']:.3f} two-arm {s['two_arm_rule']}  "
        f"branch dL waist {regions['waist']['dL']:.3f} early {regions['early']['dL']:.3f} late {regions['late']['dL']:.3f} (w/e {regions['waist']['dL'] / edge:.2f}; @0.25 waist {regions['waist@0.25']['dL']:.3f})  "
        f"skip waist {skip_regions['waist']['dL']:.3f} (w/e {skip_regions['waist']['dL'] / sedge:.2f})  block waist {regions_block['waist']['dL']:.3f}  ({time.time() - t0:.0f} s)")
    return dict(condition=cond, seed=seed, final_phase=phase, readout=f"{rp} objective on {'shuffled' if rp == 'shuffled' else 'real'} held-out text",
                n_layers=fv.N_LAYERS, d_model=fv.D_MODEL, params_M=round(n_params / 1e6, 1), steps=STEPS, switch_step=SWITCH if "2" in cond else None,
                final_train_loss=round(last, 4), val_loss=round(vloss, 4), baseline_eval_loss=float(np.mean(base)),
                sigma1_profile=[round(float(x), 4) for x in prof], sigma1_samples=[[round(float(x), 4) for x in q] for q in samples], **s,
                regions=regions, waist_over_edge=regions["waist"]["dL"] / edge, regions_block=regions_block, skip_regions=skip_regions,
                skip_waist_over_edge=skip_regions["waist"]["dL"] / sedge, per_block_rotation_dL=per_rot, per_block_skip_dL=per_skip,
                trajectory=traj, checkpoint=str(CKPT / f"{cond}_s{seed}.pt"),
                estimator="survey_sigma1_v2 token-local J^T J, condition mask", intervention="branch: Haar rotation of attention and feed-forward outputs before the residual add, doses 0.25/0.5/1 (QR-re-orthogonalised interpolation); whole-block rotation dose 1 as comparison; identity skip",
                survey_version=sv.VERSION, time_s=round(time.time() - t0), gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
                torch=torch.__version__, date=datetime.datetime.now().isoformat(timespec="seconds"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = fv.load_data()
    real_train, real_val, real_eval = data[:-(N_VAL + N_EVAL * BATCH)], data[-(N_VAL + N_EVAL * BATCH):-(N_EVAL * BATCH)], data[-(N_EVAL * BATCH):]
    from transformers import AutoTokenizer
    probe_ids = sv.natural_text_ids(AutoTokenizer.from_pretrained("gpt2"), 5)
    todo = [(c, s) for s in range(5) for c, n in CONDS.items() if s < n]
    log("=" * 70); log(f"d1 v2: {len(todo)} runs -> {OUT}; device {DEVICE}; train {len(real_train)} val {len(real_val)} eval {len(real_eval)}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    log("=" * 70); t0 = time.time()
    for i, (c, s) in enumerate(todo):
        path = OUT / f"{c}_s{s}.json"
        if path.exists():
            try:
                d = json.load(open(path)); assert "sigma1_profile" in d
                log(f"[{i + 1}/{len(todo)}] {c} s{s}: done (R {d['R_ex0']:.3f}, waist dL {d['regions']['waist']['dL']:.3f}), skipping"); continue
            except Exception:
                path.unlink()
        log(f"\n[{i + 1}/{len(todo)}] {c} seed {s}")
        res = run(c, s, real_train, real_val, real_eval, probe_ids)
        tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1, default=fv._ser); tmp.replace(path)
        if DEVICE == "cuda": torch.cuda.empty_cache()
    log(f"\nall done in {(time.time() - t0) / 3600:.2f} h"); summary(); export()


def export():
    """JSON results zipped beside OUT and into the current directory, so they can be downloaded before the instance goes."""
    import shutil
    stage = OUT.parent / "d1_v21_json"; stage.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json"): shutil.copy(f, stage / f.name)
    z = shutil.make_archive(str(OUT.parent / "d1_v21_results"), "zip", str(stage))
    try:
        here = Path.cwd() / "d1_v21_results.zip"
        if Path(z).resolve() != here.resolve(): shutil.copy(z, here)
        log(f"results zip: {z}  and  {here}  ({Path(z).stat().st_size / 1e3:.0f} kB) -- DOWNLOAD IT NOW; the instance disk is not persistent")
    except Exception as e:
        log(f"results zip: {z} ({e})")


def summary():
    rows = [json.load(open(f)) for f in sorted(OUT.glob("*_s*.json"))]
    if not rows: return
    print("\n%-10s%3s %7s %7s %7s %7s %6s  %7s | %7s %7s %7s %5s | %7s %5s | %7s" % ("cond", "n", "val", "R_ex0", "E/M", "L/M", "two", "C", "br w", "br e", "br l", "w/e", "skip w", "w/e", "block w"))
    for c in CONDS:
        rs = [r for r in rows if r["condition"] == c]
        if not rs: continue
        g = lambda k: np.mean([r[k] for r in rs]); rg = lambda k: np.mean([r["regions"][k]["dL"] for r in rs])
        print("%-10s%3d %7.3f %7.3f %7.2f %7.2f %3d/%d  %7.3f | %7.3f %7.3f %7.3f %5.2f | %7.3f %5.2f | %7.3f" % (c, len(rs), g("val_loss"), g("R_ex0"), g("early_over_mid"), g("late_over_mid"),
              sum(r["two_arm_rule"] for r in rs), len(rs), g("contrast_C"), rg("waist"), rg("early"), rg("late"), g("waist_over_edge"),
              np.mean([r["skip_regions"]["waist"]["dL"] for r in rs]), g("skip_waist_over_edge"), np.mean([r["regions_block"]["waist"]["dL"] for r in rs])))


if __name__ == "__main__":
    main()
