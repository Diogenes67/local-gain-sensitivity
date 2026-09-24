#!/usr/bin/env python3
"""
Second task family: random-transition m-th-order Markov sources, small vocabulary, entropy matched
===================================================================================================
The k-gram sweep varies context length, predictive dependence and difficulty together (the sources are
fitted to WikiText-103 over a 50,257-token vocabulary and the models never reach the entropy floor).
This family holds the vocabulary (V = 8) and the conditional entropy (H = 1.0 nats, matched by
temperature per source) fixed and varies only the order m, the number of preceding tokens the next token
depends on, m in {1, 2, 3, 4, 5, 6}, five seeds each (30 models). The transition logits of a context are
the sum over r = 1..m of independent standard-normal random tables indexed by the last r tokens (a
random back-off source): every order contributes, the full order-m table is needed to reach the floor,
and the floor a predictor reading only the last r tokens could reach, H(X_t | last r), is computed
exactly for every r (floors_by_order). A dense alternative, one independent Dirichlet row per context,
was tried first and rejected: it has no structure below order m, so the gradient signal for a partial
context of r tokens scales as V^-(m-r)/2, and a pilot at m = 5 sat at the uniform ceiling for 1,250
steps (markov_source.raw_table keeps both). The marginal entropy is held within 0.1 nats of ln V by
redrawing low-order tables, so mutual information is matched too (0.99-1.06 nats). Data are fresh every
batch (160,000 training sequences used once); the held-out set is 800 sequences.

Models, optimiser, schedule and every measurement are those of the k-gram sweep (reprofile_v2.py):
12-layer d = 512 pre-norm transformer (D3Transformer, tied 8-token embedding), AdamW wd 0.01, lr 3e-4,
500 warm-up, cosine, 10,000 steps, batch 16 x 128, dropout 0.1; canonical sigma_1 profile
(survey_sigma1_v2, in-distribution inputs; there is no natural-text probe for an 8-token model);
branch rotation by region at doses 0.25/0.5/1, fitted-waist rotation, identity skip by region,
per-block rotation and skip, early exit. Added for this family: the exact floor of the held-out set
under the true source and the learned gap (model loss minus floor over positions t >= 8, valid for every
m <= 8), a held-out learning curve every 500 steps, and the source statistics (tau, conditional and
marginal entropy, mutual information, number of contexts).

Outputs (MARKOV_OUT, on Drive): one JSON per run written atomically, resume-safe; fp16 checkpoint per run
in ckpt/; source and held-out cache in data/; results zip at the end (export()). Runs are ordered
seed-major (all orders at seed 0, then seed 1, ...) so that a partial sweep is a complete ladder.

  python markov_v1.py               full sweep (30 runs)
  python markov_v1.py --quick       learnability check: full model, orders 4/5/6, seed 0, 2,000 steps
  python markov_v1.py --plumbing    tiny models, 20 steps, end to end incl. summary and export

Needs survey_sigma1_v2.py, reprofile_v2.py and markov_source.py beside it.
"""
import os, sys, json, time, math, gc, datetime, argparse, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv
import reprofile_v2 as rp
import markov_source as ms

DEVICE = rp.DEVICE
ROOT = Path(os.environ.get("MARKOV_OUT", "markov_v1"))
V, H_TARGET, ALPHA = 8, 1.0, 0.5
ORDERS, N_SEEDS = [1, 2, 3, 4, 5, 6], 5
GAP_POS, EVAL_EVERY = 8, 500
N_HELD = rp.N_EVAL_BATCHES * rp.EVAL_BATCH      # 800 held-out sequences, as the k-gram sweep
rp.VOCAB_SIZE = V                                # every model, loss and probe in reprofile_v2 reads this global
CFG = rp.CFG
FULL = dict(CFG)                                 # the k-gram configuration, restored at every full-sweep start (the quick and plumbing modes mutate the shared dict)
log = rp.log


# ================================================================ data
def source_data(m, seed):
    """Source (tempered transition table, stationary distribution, entropies) and held-out set, cached on
    the persistent root; the training stream is regenerated deterministically at run start (20 MB per run
    otherwise)."""
    cache = ROOT / "data" / f"markov_m{m}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        src = dict(V=int(z["V"]), m=int(z["m"]), seed=int(z["seed"]), alpha=float(z["alpha"]), H_target=float(z["H_target"]), P=z["P"].astype(np.float64), pi=z["pi"],
                   tau=float(z["tau"]), H_cond=float(z["H_cond"]), H_cond_untempered=float(z["H_cond_untempered"]), H_marginal=float(z["H_marginal"]),
                   mutual_information=float(z["mutual_information"]), marginal=z["marginal"], n_states=int(z["n_states"]), stationary_iters=int(z["stationary_iters"]),
                   structure=str(z["structure"]) if "structure" in z else "hier", draw_attempts=int(z["draw_attempts"]) if "draw_attempts" in z else 1,
                   floors_by_order=z["floors_by_order"].tolist() if "floors_by_order" in z else None)
        return src, z["held"].astype(np.int64), z["floor_held"]
    t0 = time.time()
    src = ms.make_source(V=V, m=m, seed=seed, alpha=ALPHA, H_target=H_TARGET)
    held = ms.generate(src, N_HELD, CFG["seq_len"], seed=10_000 * m + seed + 777)
    floor_held = ms.floor_per_position(src, held).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, held=held.astype(np.uint8), floor_held=floor_held, P=src["P"].astype(np.float32), **{k: v for k, v in src.items() if k != "P"})
    log(f"  source m={m} seed {seed} ({src['structure']}): {src['n_states']} contexts, tau {src['tau']:.3f}, H {src['H_cond']:.4f} (untempered {src['H_cond_untempered']:.3f}), "
        f"H_marg {src['H_marginal']:.3f}, MI {src['mutual_information']:.3f}, floors by order {np.round(src['floors_by_order'], 3).tolist()}, "
        f"held floor(t>={GAP_POS}) {np.nanmean(floor_held[:, GAP_POS:]):.4f} ({time.time() - t0:.0f} s)")
    return src, held, floor_held


def training_stream(src, seed, n_steps):
    return torch.from_numpy(ms.generate(src, n_steps * CFG["batch_size"], CFG["seq_len"], seed=10_000 * src["m"] + seed + 1))


# ================================================================ measurement
@torch.no_grad()
def tail_loss(model, held):
    """Held-out loss over targets t >= GAP_POS, the positions at which every order <= GAP_POS has its full context."""
    model.eval(); model.drop.p = 0.0; mask = rp.causal_mask(CFG["seq_len"], DEVICE); tot = 0.0; n = 0
    try:
        for b in range(0, len(held), rp.EVAL_BATCH):
            ids = held[b:b + rp.EVAL_BATCH].to(DEVICE); lg = model(ids, attn_mask=mask)
            tot += F.cross_entropy(lg[:, GAP_POS - 1:-1].reshape(-1, V), ids[:, GAP_POS:].reshape(-1)).item(); n += 1
        return tot / max(n, 1)
    finally:
        model.drop.p = 0.1; model.train()


def measure(model, seed, held, probe_dist):
    """rp.measure without the natural-text profile (an 8-token model has none)."""
    L = len(model.blocks); out = {}
    mask = rp.causal_mask(CFG["seq_len"], DEVICE)
    prof, samp, it1 = rp.survey_profile(model, mask, probe_dist)
    out["profile_indist"] = prof.tolist(); out["samples_indist"] = [[round(x, 4) for x in s] for s in samp]
    out["stats_indist"] = rp.two_arm(prof); out["ufit_indist"] = rp.fit_ushape(prof); out["power_iters"] = [it1]
    base = rp.eval_losses(model, held); out["baseline_eval_loss"] = float(np.mean(base))
    Qa = {li: rp.haar(CFG["d_model"], rp.SEED_BASE + li * 100 + seed) for li in range(L)}
    Qf = {li: rp.haar(CFG["d_model"], rp.SEED_BASE + li * 100 + 10 + seed) for li in range(L)}
    regions = {}
    for rn, blk in (("early", rp.EARLY), ("waist", rp.WAIST), ("late", rp.LATE)):
        for dose in rp.DOSES:
            regions[f"{rn}@{dose}"] = rp.dstat(rp.eval_losses(model, held, rp.rotate_branches(model, blk, Qa, Qf, dose)), base)
        regions[rn] = regions[f"{rn}@1.0"]
    out["regions_branch"] = regions
    edge = max(regions["early"]["dL"], regions["late"]["dL"], 1e-9); out["waist_over_edge_branch"] = regions["waist"]["dL"] / edge
    fit = out["ufit_indist"]; fr = {}
    for rn in ("waist", "early", "late"):
        if fit[rn]:
            fr[rn] = rp.dstat(rp.eval_losses(model, held, rp.rotate_branches(model, fit[rn], Qa, Qf, 1.0)), base)
    out["regions_branch_fitted_waist"] = fr
    out["skip_regions"] = {rn: rp.dstat(rp.eval_losses(model, held, rp.skip_blocks(model, blk)), base) for rn, blk in (("early", rp.EARLY), ("waist", rp.WAIST), ("late", rp.LATE))}
    sedge = max(out["skip_regions"]["early"]["dL"], out["skip_regions"]["late"]["dL"], 1e-9); out["skip_waist_over_edge"] = out["skip_regions"]["waist"]["dL"] / sedge
    out["per_block_branch_dL"] = [rp.dstat(rp.eval_losses(model, held, rp.rotate_branches(model, [li], Qa, Qf, 1.0)), base)["dL"] for li in range(L)]
    out["per_block_skip_dL"] = [rp.dstat(rp.eval_losses(model, held, rp.skip_blocks(model, [li])), base)["dL"] for li in range(L)]
    out["early_exit_loss"] = rp.early_exit(model, held)
    return out


# ================================================================ training
def train_model(model, stream, seed, held, probe_cb, curve_cb):
    """The k-gram optimiser (AdamW wd 0.01, cosine with 500 warm-up, clip 1.0) on fresh batches."""
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    mask = rp.causal_mask(CFG["seq_len"], DEVICE); bs = CFG["batch_size"]; model.train()
    t0, last = time.time(), None
    for step in range(CFG["n_steps"]):
        if step in rp.PROFILE_AT: probe_cb(step)
        if step % EVAL_EVERY == 0: curve_cb(step)
        ids = stream[step * bs:(step + 1) * bs].to(DEVICE)
        for pg in opt.param_groups: pg["lr"] = rp.get_lr(step, True)
        loss = F.cross_entropy(model(ids, attn_mask=mask)[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        last = loss.item()
        if (step + 1) % 2500 == 0: log(f"    step {step + 1}/{CFG['n_steps']} loss {last:.3f} ({time.time() - t0:.0f} s)")
    curve_cb(CFG["n_steps"])
    return last


def run_one(m, seed, ckpt_dir, quick=False):
    t0 = time.time()
    src, held_np, floor_held = source_data(m, seed)
    held = torch.from_numpy(held_np); probe_dist = held[-5:]
    floor_tail = float(np.nanmean(floor_held[:, GAP_POS:])); floor_all = float(np.nanmean(floor_held[:, 1:]))
    torch.manual_seed(rp.SEED_BASE + seed); np.random.seed(rp.SEED_BASE + seed)
    model = rp.D3Transformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  markov m={m} seed {seed}: {n_params / 1e6:.1f}M params; floor(t>={GAP_POS}) {floor_tail:.4f}")
    stream = training_stream(src, seed, CFG["n_steps"])
    mask = rp.causal_mask(CFG["seq_len"], DEVICE)
    traj, curve = {}, []
    def probe(step):
        p, _, _ = rp.survey_profile(model, mask, probe_dist); base = rp.eval_losses(model, held)
        traj[step] = dict(profile_indist=p.tolist(), stats_indist=rp.two_arm(p), eval_loss=float(np.mean(base)), tail_loss=tail_loss(model, held))
        s = traj[step]["stats_indist"]
        log(f"    step {step}: R_ex0 {s['R_ex0']:.3f} E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f} eval {traj[step]['eval_loss']:.3f} gap {traj[step]['tail_loss'] - floor_tail:+.4f}")
    def on_curve(step):
        tl = tail_loss(model, held); curve.append([step, round(tl, 5)])
        if step % 2000 == 0: log(f"    step {step}: held(t>={GAP_POS}) {tl:.4f} gap {tl - floor_tail:+.4f}")
    if not quick: probe(0)
    last = train_model(model, stream, seed, held, probe if not quick else (lambda s: None), on_curve)
    tl = tail_loss(model, held); gap = tl - floor_tail
    src_stats = dict(V=V, order=m, structure=src["structure"], n_states=src["n_states"], tau=src["tau"], H_cond=src["H_cond"], H_cond_untempered=src["H_cond_untempered"],
                     H_marginal=src["H_marginal"], mutual_information=src["mutual_information"], floors_by_order=src["floors_by_order"], draw_attempts=src["draw_attempts"],
                     floor_tail=floor_tail, floor_all_positions=floor_all, uniform_ceiling=float(np.log(V)), stationary_iters=src["stationary_iters"])
    below_prev = (src["floors_by_order"][-2] - tl) if src["floors_by_order"] and m >= 1 else float("nan")   # margin below the best order-(m-1) predictor
    if quick:
        log(f"    -> quick m={m}: held(t>={GAP_POS}) {tl:.4f} floor {floor_tail:.4f} gap {gap:+.4f}; below the order-{m - 1} floor by {below_prev:+.4f} ({time.time() - t0:.0f} s)")
        return dict(experiment="markov_quick", condition=str(m), seed=seed, tail_loss=tl, learned_gap=gap, margin_below_prev_order=below_prev, learning_curve=curve, source=src_stats, cfg=dict(CFG),
                    gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", torch=torch.__version__, time_s=round(time.time() - t0), date=datetime.datetime.now().isoformat(timespec="seconds"))
    mres = measure(model, seed, held, probe_dist)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ck = ckpt_dir / f"markov_{m}_s{seed}.pt"
    torch.save({k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()}, ck)
    s = mres["stats_indist"]; r = mres["regions_branch"]
    log(f"    -> eval {mres['baseline_eval_loss']:.3f} tail {tl:.4f} gap {gap:+.4f} (below order-{m - 1} floor by {below_prev:+.4f})  R_ex0 {s['R_ex0']:.3f} E/M {s['early_over_mid']:.2f} L/M {s['late_over_mid']:.2f} two-arm {s['two_arm_rule']} C {s['contrast_C']:.3f}  "
        f"branch waist {r['waist']['dL']:.3f} early {r['early']['dL']:.3f} late {r['late']['dL']:.3f} (w/e {mres['waist_over_edge_branch']:.2f})  skip waist {mres['skip_regions']['waist']['dL']:.3f}  ({time.time() - t0:.0f} s)")
    return dict(experiment="markov", condition=str(m), seed=seed, params_M=round(n_params / 1e6, 1), final_train_loss=round(last, 4), trajectory=traj,
                tail_loss=tl, learned_gap=gap, margin_below_prev_order=below_prev, learning_curve=curve, source=src_stats, **mres,
                checkpoint=str(ck), estimator="survey_sigma1_v2 token-local J^T J, causal mask, 5 inputs x 8 positions", survey_version=sv.VERSION,
                intervention="branch rotation (attention and MLP outputs, linear dose interpolation as d3); identity skip; early exit",
                data=f"random-transition order-{m} Markov source ({src['structure']}), V={V}, tempered to H={H_TARGET} nats; fresh batches; {N_HELD} held-out sequences",
                cfg=dict(CFG), gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu", torch=torch.__version__, time_s=round(time.time() - t0),
                date=datetime.datetime.now().isoformat(timespec="seconds"))


# ================================================================ driver
def runs():
    return [(m, s) for s in range(N_SEEDS) for m in ORDERS]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--plumbing", action="store_true")
    args = ap.parse_args(argv)
    global ORDERS, N_SEEDS
    if args.plumbing:
        CFG.update(n_layers=6, d_model=32, n_heads=4, d_ff=64, n_steps=20, warmup_steps=5); rp.PROFILE_AT[:] = [10]
        rp.EARLY, rp.WAIST, rp.LATE = [0, 1], [2, 3], [4, 5]; rp.N_EVAL_BATCHES = 5
        ORDERS, N_SEEDS = [1, 3], 1
    if args.quick:
        CFG.update(n_steps=2000); ORDERS, N_SEEDS = [int(x) for x in os.environ.get("MARKOV_QUICK_ORDERS", "4,5,6").split(",")], 1
    if not (args.quick or args.plumbing):
        CFG.update(FULL); rp.EARLY, rp.WAIST, rp.LATE = [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]; rp.N_EVAL_BATCHES = 50; rp.PROFILE_AT[:] = [5000]
    exp = "markov_quick" if args.quick else "markov"
    OUT = ROOT / exp; OUT.mkdir(parents=True, exist_ok=True); CK = OUT / "ckpt"
    todo = runs()
    log("=" * 70); log(f"markov v1: {exp}, {len(todo)} runs -> {OUT}; device {DEVICE}; V={V}, H={H_TARGET}, orders {ORDERS}, seeds {N_SEEDS}, n_steps {CFG['n_steps']}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name}, {p.total_memory / 1e9:.0f} GB")
    log("=" * 70)
    t0 = time.time()
    for i, (m, seed) in enumerate(todo):
        path = OUT / f"{exp}_{m}_s{seed}.json"
        if path.exists():
            try:
                d = json.load(open(path)); assert ("profile_indist" in d) or args.quick
                assert d["cfg"]["n_steps"] == CFG["n_steps"], f"{path.name} was trained for {d['cfg']['n_steps']} steps, this sweep uses {CFG['n_steps']}: re-running it"
                log(f"[{i + 1}/{len(todo)}] m={m} s{seed}: done (gap {d['learned_gap']:+.4f}, {d['cfg']['n_steps']} steps), skipping"); continue
            except Exception as e:
                log(f"  {e}"); path.rename(path.with_suffix(".old.json"))
        log(f"\n[{i + 1}/{len(todo)}] markov m={m} seed {seed} ({CFG['n_steps']} steps)")
        res = run_one(m, seed, CK, quick=args.quick)
        tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w"), indent=1); tmp.replace(path)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    log(f"\nall done in {(time.time() - t0) / 3600:.2f} h"); summary(exp); export(exp)


def summary(exp="markov"):
    OUT = ROOT / exp
    rows = [json.load(open(f)) for f in sorted(OUT.glob(f"{exp}_*_s*.json"))]
    if not rows: print("no results yet"); return
    if exp == "markov_quick":
        print("\n%-6s%3s %8s %8s %8s | learning curve (step: held loss, t>=8)" % ("order", "n", "floor", "held", "gap"))
        for r in sorted(rows, key=lambda r: (int(r["condition"]), r["seed"])):
            print("%-6s%3d %8.4f %8.4f %+8.4f | %s" % (r["condition"], 1, r["source"]["floor_tail"], r["tail_loss"], r["learned_gap"],
                  "  ".join(f"{s}:{v:.3f}" for s, v in r["learning_curve"][::2])))
        return
    print("\n%-6s%3s %7s %7s %7s %7s | %7s %6s %6s %5s | %7s %7s %7s %5s | %7s %5s | %7s" % ("order", "n", "floor", "held", "gap", "m-1 mrg", "R_ex0", "E/M", "L/M", "two", "br w", "br e", "br l", "w/e", "skip w", "w/e", "C_fit"))
    for m in sorted(set(int(r["condition"]) for r in rows)):
        rs = [r for r in rows if int(r["condition"]) == m]
        g = lambda f: np.mean([f(r) for r in rs])
        print("%-6d%3d %7.4f %7.4f %+7.4f %+7.4f | %7.3f %6.2f %6.2f %2d/%d | %7.3f %7.3f %7.3f %5.2f | %7.3f %5.2f | %7.3f" % (
            m, len(rs), g(lambda r: r["source"]["floor_tail"]), g(lambda r: r["tail_loss"]), g(lambda r: r["learned_gap"]), g(lambda r: r.get("margin_below_prev_order", float("nan"))),
            g(lambda r: r["stats_indist"]["R_ex0"]), g(lambda r: r["stats_indist"]["early_over_mid"]), g(lambda r: r["stats_indist"]["late_over_mid"]),
            sum(r["stats_indist"]["two_arm_rule"] for r in rs), len(rs),
            g(lambda r: r["regions_branch"]["waist"]["dL"]), g(lambda r: r["regions_branch"]["early"]["dL"]), g(lambda r: r["regions_branch"]["late"]["dL"]), g(lambda r: r["waist_over_edge_branch"]),
            g(lambda r: r["skip_regions"]["waist"]["dL"]), g(lambda r: r["skip_waist_over_edge"]), g(lambda r: r["ufit_indist"]["C_fit"])))
    print("n_steps per run:", sorted(set(r["cfg"]["n_steps"] for r in rows)), "| output folder:", OUT)
    print("per-run gaps:", "  ".join(f"m{r['condition']}s{r['seed']}:{r['learned_gap']:+.3f}" for r in sorted(rows, key=lambda r: (int(r["condition"]), r["seed"]))))


def export(exp="markov"):
    OUT = ROOT / exp
    stage = ROOT / f"{exp}_json"; stage.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json"): shutil.copy(f, stage / f.name)
    z = shutil.make_archive(str(ROOT / f"{exp}_v1_results"), "zip", str(stage))
    here = Path.cwd() / f"{exp}_v1_results.zip"
    if Path(z).resolve() != here.resolve(): shutil.copy(z, here)
    log(f"results zip: {z} ({os.path.getsize(z) / 1e6:.2f} MB) and {here} -- on Drive already if MARKOV_OUT is on Drive")


if __name__ == "__main__":
    main()
