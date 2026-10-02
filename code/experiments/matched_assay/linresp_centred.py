"""Downstream linearised response with the logit gauge recorded (25 Sept 2026; review 5, item 3).

Same probes, directions and quantities as the 21 Sept pilot (linearised_response.py): for a probe (block l, input i,
position t) and a direction u at the block input, g = A u is the forward-mode product of the position-t logits with
respect to the block-l input at t, and the small-dose same-position divergence is 1/2 eps^2 ||h||^2 g^T F g with
F = diag(p) - p p^T. The pilot recorded ||g||^2 (amplification) and g^T F g / ||g||^2 (alignment). Both depend on the
logit gauge: adding a constant to every logit changes ||g|| and not g^T F g. This run also records, per probe and
direction, mean(g) over the vocabulary, p^T g and the vocabulary size V, from which the split is available in three
gauges without another forward pass:

    raw           amp = ||g||^2,                              align = g^T F g / amp
    mean-centred  amp_c = ||g - mean(g) 1||^2 = amp - V mean(g)^2, align_c = g^T F g / amp_c
    p-centred     amp_p = ||g - (p^T g) 1||^2,                 align_p = g^T F g / amp_p

The product g^T F g, and therefore the combined downstream factor g^T F g / ||J u||^2, is the same in every gauge.
The measured same-position KL at eps in {0.01, 0.03, 0.1} is recorded again for the check against the prediction.

Subset (as the pilot): d1 real s0-1, shuffled s0-1, k-gram k = 1 s0-1, k = 8 s0-1; five inputs x four positions; v_1 and
four random directions per probe (same generator and seed, so the random directions are those of the matched assay).
Output: one JSON per model in LINRESP_CENTRED_OUT/results (atomic, resume-safe), a summary from disk, and a zip beside
the results and in the working directory. Needs matched_perturbation.py (and its imports) beside it.

Run: python linresp_centred.py            # LINRESP_CENTRED_OUT, D1_CKPT, D1_DATA, REPRO_ROOT as for the matched assay
     python linresp_centred.py --plumbing # random-weight models, synthetic inputs, CPU, no checkpoints
"""
import os, sys, json, time, gc, shutil, argparse, datetime, platform
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_perturbation as mp

OUT = Path(os.environ.get("LINRESP_CENTRED_OUT", "linresp_centred"))
SUBSET = [("d1", "real", 0), ("d1", "real", 1), ("d1", "shuffled", 0), ("d1", "shuffled", 1), ("kgram", "1", 0), ("kgram", "1", 1), ("kgram", "8", 0), ("kgram", "8", 1)]
EPS = [0.01, 0.03, 0.1]
N_RAND = 4
VERSION = "linresp-centred-2026-09-25"
DEVICE = mp.DEVICE
log = mp.log


def logit_f_with_ids(ad, li, hidden0, t, ids_1):
    base = hidden0.clone()

    def f(x):
        def pre(m, a):
            h = base.clone(); h[0, t] = x; return (h,) + tuple(a[1:])
        hk = ad.layers[li].register_forward_pre_hook(pre)
        try:
            logits = ad.forward(ids_1)
        finally:
            hk.remove()
        return logits[0, t]
    return f


def gauge_stats(g, p0):
    """amp, quad and the two centred amplifications from one logit response g (V,) and the clean distribution p0."""
    V = g.numel(); gm = float(g.mean()); gp = float((p0 * g).sum())
    amp = float((g * g).sum()); quad = float((g * g * p0).sum() - gp ** 2)
    amp_c = float(((g - gm) ** 2).sum()); amp_p = float(((g - gp) ** 2).sum())
    return dict(amplification=amp, quad=quad, alignment=quad / max(amp, 1e-30), g_mean=gm, g_pmean=gp, V=int(V),
                amplification_centred=amp_c, alignment_centred=quad / max(amp_c, 1e-30),
                amplification_pcentred=amp_p, alignment_pcentred=quad / max(amp_p, 1e-30))


def run_model(ad, ids, xs, sels, quick=False):
    # same loop order and generator as matched_perturbation.run_model, so the random directions are the assay's
    # (full run: every block, every input; quick: three blocks, one input, two positions, as the assay's quick mode)
    L = len(ad.layers); positions = mp.POSITIONS if not quick else mp.POSITIONS[:2]; n_in = ids.shape[0] if not quick else 1
    blocks = range(L) if not quick else [0, L // 2, L - 1]
    g_rng = torch.Generator(device="cpu").manual_seed(mp.SEED)
    rows = []
    for li in blocks:
        for i in range(n_in):
            ids_1 = (xs[i:i + 1] if xs is not None else ids[i:i + 1]).to(DEVICE)
            x_in = mp.capture_block_input(ad, li, ids_1)
            f_blk = mp.make_token_local_f(ad, li, x_in, positions)
            x0 = x_in[0, positions].clone()
            sig, v1, _ = mp.top_singular(f_blk, x0)
            for p, t in enumerate(positions):
                fl = logit_f_with_ids(ad, li, x_in, t, ids_1)
                h0 = x_in[0, t].clone(); hn = float(h0.norm())
                with torch.no_grad():
                    logits0 = fl(h0); p0 = torch.softmax(logits0.float(), -1)
                dirs = [("v1", v1[p])]
                for r in range(N_RAND):
                    u = torch.randn(h0.shape[0], generator=g_rng).to(DEVICE); dirs.append((f"rand{r}", u / u.norm()))
                for name, u in dirs:
                    _, g = torch.func.jvp(fl, (h0,), (u,)); g = g.detach().float()
                    st = gauge_stats(g, p0)
                    pred = {str(e): 0.5 * (e * hn) ** 2 * st["quad"] for e in EPS}
                    meas = {}
                    with torch.no_grad():
                        for e in EPS:
                            kls = []
                            for sgn in (1.0, -1.0):
                                lg = fl(h0 + sgn * e * hn * u).float(); q = torch.log_softmax(lg, -1)
                                kls.append(float((p0 * (torch.log(p0 + 1e-30) - q)).sum()))
                            meas[str(e)] = float(np.mean(kls))
                    rows.append(dict(block=li, input=i, position=t, direction=name, sigma1=float(sig[p]), h_norm=hn, kl_pred=pred, kl_meas=meas, **st))
        log(f"  block {li}: {len(rows)} rows so far")
    return rows


def summarise(rows):
    L = max(r["block"] for r in rows) + 1; out = []
    for li in range(L):
        v = [r for r in rows if r["block"] == li and r["direction"] == "v1"]; rd = [r for r in rows if r["block"] == li and r["direction"] != "v1"]
        if not v:
            continue
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        rec = dict(block=li, sigma1=m(v, "sigma1"), quad_v1=m(v, "quad"), quad_rand=m(rd, "quad"),
                   pred_over_meas={e: float(np.median([r["kl_pred"][e] / max(r["kl_meas"][e], 1e-30) for r in v])) for e in map(str, EPS)})
        for gauge, a in (("", "amplification"), ("_centred", "amplification_centred"), ("_pcentred", "amplification_pcentred")):
            rec[f"amp_v1{gauge}"] = m(v, a); rec[f"amp_rand{gauge}"] = m(rd, a)
            rec[f"align_v1{gauge}"] = rec["quad_v1"] / rec[f"amp_v1{gauge}"]; rec[f"align_rand{gauge}"] = rec["quad_rand"] / rec[f"amp_rand{gauge}"]
        out.append(rec)
    return out


def provenance():
    p = {"torch": torch.__version__, "python": platform.python_version(), "date": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE}
    if torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name()
    return p


def run_job(exp, cond, seed, res_dir, quick=False):
    fpath = res_dir / f"linresp_{exp}_{cond}_s{seed}{'_quick' if quick else ''}.json"
    if fpath.exists():
        try:
            if json.load(open(fpath)).get("version") == VERSION:
                log(f"[SKIP] {exp} {cond} s{seed}"); return
        except Exception:
            pass
    t0 = time.time(); log(f"\n{exp} {cond} seed {seed}")
    if exp == "d1":
        model = mp.load_d1(cond, seed); ad = mp.Adapter("d1", model, cond); ids, xs, sels = mp.d1_inputs(cond, seed)
    else:
        model = mp.load_kgram(int(cond), seed); ad = mp.Adapter("kgram", model); ids, xs, sels = mp.kgram_inputs(int(cond), seed)
    rows = run_model(ad, ids, xs, sels, quick); per_block = summarise(rows)
    out = dict(experiment=exp, condition=cond, seed=seed, version=VERSION, eps=EPS, positions=mp.POSITIONS if not quick else mp.POSITIONS[:2],
               n_inputs=int(ids.shape[0]), n_random=N_RAND, quick=quick,
               definition=("g = A u, forward-mode product of the position-t logits w.r.t. the block-l input at t (block l included); "
                           "amplification = ||g||^2; quad = g^T F g with F = diag(p) - p p^T; alignment = quad / amplification; "
                           "amplification_centred = ||g - mean(g) 1||^2; amplification_pcentred = ||g - (p^T g) 1||^2; "
                           "kl_pred = 1/2 eps^2 ||h||^2 quad; kl_meas = same-position KL(clean || perturbed), signs averaged"),
               per_block=per_block, raw=rows, elapsed_seconds=time.time() - t0, **{f"prov_{k}": v for k, v in provenance().items()})
    mp.write_json(fpath, out); log(f"  saved {fpath} ({time.time() - t0:.0f} s)")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


PILOT = Path(os.environ.get("LINRESP_PILOT", str(OUT.parent / "linresp" / "results")))


def pilot_deviation(d, name):
    """Median relative deviation of quad and amplification from the 21 Sept pilot on the same probes, if its file exists."""
    f = PILOT / f"linresp_{name}.json"
    if not f.exists():
        return None
    old = {(r["block"], r["input"], r["position"], r["direction"]): r for r in json.load(open(f))["raw"]}
    dev = {k: [] for k in ("quad", "amplification")}
    for r in d["raw"]:
        o = old.get((r["block"], r["input"], r["position"], r["direction"]))
        if o:
            for k in dev: dev[k].append(abs(r[k] / o[k] - 1) if o[k] else 0.0)
    return {k: float(np.median(v)) if v else float("nan") for k, v in dev.items()} | {"n": len(dev["quad"])}


def summary(res_dir):
    """From the JSONs on disk: per model, the v1/random ratios of amplification and alignment in the three gauges (geometric
    mean over probes then blocks 1-10), the product (which is gauge-free), the prediction check at eps = 0.01 and the
    median relative deviation of quad from the 21 Sept pilot on the same probes (blank when the pilot file is absent)."""
    files = sorted(Path(res_dir).glob("linresp_*.json"))
    log(f"\n{'model':22s} {'amp v1/rand':>11s} {'align':>7s} | {'centred amp':>11s} {'align':>7s} | {'p-centred amp':>13s} {'align':>7s} | {'quad ratio':>10s} {'pred/meas .01':>13s} {'dev. pilot':>10s}")
    rows = []
    for f in files:
        d = json.load(open(f)); raw = [r for r in d["raw"] if 1 <= r["block"] <= 10] or d["raw"]
        probes = {}
        for r in raw:
            probes.setdefault((r["block"], r["input"], r["position"]), {}).setdefault("v1" if r["direction"] == "v1" else "rand", []).append(r)
        def ratio(key):
            per = {}
            for k, pr in probes.items():
                v = pr["v1"][0][key]; rd = float(np.mean([x[key] for x in pr["rand"]]))
                per.setdefault(k[0], []).append(np.log(max(v, 1e-300) / max(rd, 1e-300)))
            return float(np.exp(np.mean([np.mean(v) for v in per.values()])))
        rec = dict(model=f.stem[8:], amp=ratio("amplification"), align=ratio("alignment"), amp_c=ratio("amplification_centred"), align_c=ratio("alignment_centred"),
                   amp_p=ratio("amplification_pcentred"), align_p=ratio("alignment_pcentred"), quad=ratio("quad"),
                   pred_over_meas=float(np.median([r["kl_pred"]["0.01"] / max(r["kl_meas"]["0.01"], 1e-30) for r in raw])), pilot=pilot_deviation(d, f.stem[8:]))
        rows.append(rec)
        dev = f"{rec['pilot']['quad']:10.2e}" if rec["pilot"] else f"{'':10s}"
        log(f"{rec['model']:22s} {rec['amp']:11.2f} {rec['align']:7.2f} | {rec['amp_c']:11.2f} {rec['align_c']:7.2f} | {rec['amp_p']:13.2f} {rec['align_p']:7.2f} | {rec['quad']:10.2f} {rec['pred_over_meas']:13.3f} {dev}")
    return rows


def export(res_dir):
    res_dir = Path(res_dir); stage = res_dir.parent / "_stage"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir()
    for f in res_dir.glob("linresp_*.json"): shutil.copy(f, stage / f.name)
    z = shutil.make_archive(str(res_dir.parent / "linresp_centred_results"), "zip", str(stage))
    here = Path.cwd() / "linresp_centred_results.zip"
    if Path(z).resolve() != here.resolve(): shutil.copy(z, here)
    shutil.rmtree(stage)
    log(f"results zip: {z} ({os.path.getsize(z)} bytes) and {here}")
    return [z, str(here)]


def plumbing():
    """Random-weight d1 and k-gram models, synthetic inputs, CPU: checks the gauge identities on real records and that
    the second-order prediction matches the measured KL at eps = 0.01, then summary() and export()."""
    import factorial_v2 as fv, reprofile_v2 as rp
    torch.manual_seed(0); fv.VOCAB = 2000; rp.VOCAB_SIZE = 2000; fv.EOS = 1999
    res_dir = OUT / "results_plumbing"; res_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("d1", "kgram"):
        ids = torch.randint(0, 1000, (2, fv.SEQ))
        if kind == "d1":
            model = fv.Transformer().to(DEVICE).eval(); ad = mp.Adapter("d1", model, "real")
        else:
            model = rp.D3Transformer().to(DEVICE).eval(); model.drop.p = 0.0; ad = mp.Adapter("kgram", model)
        rows = run_model(ad, ids, None, None, quick=True)
        for r in rows:
            assert abs(r["amplification_centred"] - (r["amplification"] - r["V"] * r["g_mean"] ** 2)) < 1e-3 * max(r["amplification"], 1e-6), "centred identity"
            assert r["amplification_centred"] <= r["amplification"] * (1 + 1e-6) and r["amplification_pcentred"] >= r["amplification_centred"] * (1 - 1e-6), "gauge ordering"
        ratio = float(np.median([r["kl_pred"]["0.01"] / max(r["kl_meas"]["0.01"], 1e-30) for r in rows if r["direction"] == "v1"]))
        log(f"  {kind}: linearised/measured KL at eps 0.01 = {ratio:.4f} (expect ~1); mean(g) magnitude / rms(g) = "
            f"{np.median([abs(r['g_mean']) / np.sqrt(r['amplification'] / r['V']) for r in rows]):.3f}")
        assert 0.8 < ratio < 1.25, "linearised prediction does not match the measured KL"
        mp.write_json(res_dir / f"linresp_{kind}_plumbing_s0_quick.json", dict(experiment=kind, condition="plumbing", seed=0, version=VERSION, eps=EPS, per_block=summarise(rows), raw=rows, quick=True))
    summary(res_dir); zs = export(res_dir); assert all(os.path.getsize(z) > 0 for z in zs); log("plumbing OK")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--plumbing", action="store_true"); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.plumbing:
        plumbing(); return
    res = OUT / "results"; res.mkdir(parents=True, exist_ok=True)
    for exp, cond, seed in SUBSET:
        try:
            run_job(exp, cond, seed, res, args.quick)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"  FAILED {exp} {cond} s{seed}: {e}")
    summary(res); export(res)


if __name__ == "__main__":
    main()
