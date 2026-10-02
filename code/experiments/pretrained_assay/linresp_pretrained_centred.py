"""Downstream linearised response in the six pretrained decoders with the logit gauge recorded (25 Sept 2026; review 5, item 3).

Same probes and directions as the linearised part of the 23 Sept matched assay (matched_pretrained.py): the first five of
the 20 WikiText-103 inputs, positions {8, 32, 56, 80, 104, 120}, every block; v_1 from the token-local J^T J power iteration
and the same eight random unit directions (drawn once per (input, position) from the assay's generator and seed). For each
probe and direction g = A u is one batched forward-mode product of the position-t logits with respect to the block-l input
at t (block l included). Recorded per direction: amplification ||g||^2, quad g^T F g with F = diag(p) - p p^T, alignment
quad / amplification, and the gauge statistics mean(g), p^T g and V, from which

    mean-centred  amp_c = ||g - mean(g) 1||^2 = amp - V mean(g)^2      align_c = quad / amp_c
    p-centred     amp_p = ||g - (p^T g) 1||^2                          align_p = quad / amp_p

quad and the combined downstream factor quad / ||J u||^2 are the same in every gauge. Two checks are recorded: the measured
same-position KL along v_1 at eps = 0.01 (two forward passes) against the second-order prediction 1/2 eps^2 ||h||^2 quad,
and, when the assay's result file is available (MP_RESULTS), the assay's sigma_1, amplification and quad on the same probes.

Outputs: one JSON per model in $MPC_OUT/results (atomic; a partial file after every block, so a disconnect loses at most
one block), a summary from disk, and linresp_pretrained_centred_results.zip in $MPC_OUT and in the working directory.
Needs matched_pretrained.py and survey_sigma1_v2.py beside it.

Usage: python linresp_pretrained_centred.py [--models gpt2 ...] [--quick] [--plumbing]
Environment: MPC_OUT (default ./matched_pretrained_centred), MP_RESULTS (assay results folder, optional), HF_TOKEN,
             MP_LOGIT_BUDGET_GB (default 6), MP_PURGE_CACHE
"""
import os, sys, json, time, gc, argparse, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_pretrained as mpp
import survey_sigma1_v2 as sv

DEVICE = mpp.DEVICE
OUT = os.environ.get("MPC_OUT", "matched_pretrained_centred")
MP_RESULTS = os.environ.get("MP_RESULTS", "")
VERSION = "linresp-pretrained-centred-2026-09-25"
EPS_CHECK = 0.01
log = mpp.log


def logit_response(b, li, ids_1, h0, t, dirs):
    """g = A u for every direction in one batched forward-mode product. Returns G (B, V) float32, p0 (V,), the method."""
    layer = b["layers"][li]
    U = torch.stack([u for _, u in dirs]); B = U.shape[0]
    X0 = h0[None].expand(B, -1).clone()

    def fl(x):
        def pre(m, args, kwargs):
            h = mpp._hidden_from(args, kwargs)
            h = torch.cat([h[:, :t], x[:, None, :].to(h.dtype), h[:, t + 1:]], 1)
            return mpp._with_hidden(args, kwargs, h)
        hk = layer.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            logits = mpp._first(b["forward"](b["model"], {"input_ids": ids_1.expand(B, -1)}).logits)
        finally:
            hk.remove()
        return logits[:, t].float()

    method = "torch.func.jvp"
    try:
        logits0, G = torch.func.jvp(fl, (X0,), (U,))
    except Exception as e:
        method = f"central finite difference (jvp failed: {type(e).__name__})"
        with torch.no_grad():
            s = 1e-3 * float(h0.norm())
            G = (fl(X0 + s * U) - fl(X0 - s * U)) / (2 * s); logits0 = fl(X0)
    with torch.no_grad():
        p0 = torch.softmax(logits0[0].float(), -1)
    return G.detach().float(), p0, method


def gauge_stats(G, p0):
    """Per direction: amplification, quad, alignment, mean(g), p^T g, V and the two centred amplifications and alignments."""
    with torch.no_grad():
        V = G.shape[1]; gm = G.mean(1); gp = (G * p0).sum(1)
        amp = (G * G).sum(1); quad = (G * G * p0).sum(1) - gp ** 2
        amp_c = ((G - gm[:, None]) ** 2).sum(1); amp_p = ((G - gp[:, None]) ** 2).sum(1)
    f = lambda x: [float(v) for v in x.cpu()]
    return dict(amplification=f(amp), quad=f(quad), alignment=f(quad / amp.clamp_min(1e-30)), g_mean=f(gm), g_pmean=f(gp), V=int(V),
                amplification_centred=f(amp_c), alignment_centred=f(quad / amp_c.clamp_min(1e-30)),
                amplification_pcentred=f(amp_p), alignment_pcentred=f(quad / amp_p.clamp_min(1e-30)))


@torch.no_grad()
def kl_v1_check(b, li, ids_1, h_t, t, v1, eps=EPS_CHECK):
    """Measured same-position KL(clean || perturbed) along v1 at eps, signs averaged (three rows in one forward)."""
    norm = float(h_t.norm())
    delta = torch.stack([torch.zeros_like(h_t), eps * norm * v1, -eps * norm * v1])
    layer = b["layers"][li]

    def pre(m, args, kwargs):
        h = mpp._hidden_from(args, kwargs).clone()
        h[:, t] = h[:, t] + delta.to(h.dtype)
        return mpp._with_hidden(args, kwargs, h)

    hk = layer.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        logits = mpp._first(b["forward"](b["model"], {"input_ids": ids_1.expand(3, -1)}).logits)[:, t].float()
    finally:
        hk.remove()
    logp = torch.log_softmax(logits, -1); cl = logp[0:1]
    kl = (cl.exp() * (cl - logp)).sum(-1)
    return float(kl[1:].mean())


def assay_records(hf_id):
    """The assay's lin_raw records for this model keyed by probe, if its result file is available."""
    if not MP_RESULTS:
        return {}
    f = mpp.result_path(MP_RESULTS, hf_id)
    if not os.path.exists(f):
        log(f"  (no assay file {f}; reproduction check skipped)"); return {}
    d = json.load(open(f))
    return {(x["block"], x["input"], x["position"]): x for x in d["lin_raw"]}


def run_model(b, ids, part_path=None, quick=False, assay=None):
    L = len(b["layers"])
    blocks = list(range(L)) if not quick else [0, L // 2, L - 1]
    positions = mpp.POSITIONS if not quick else mpp.POSITIONS[:2]
    n_in = ids.shape[0] if not quick else 2
    n_lin = min(mpp.N_LIN_INPUTS, n_in) if not quick else 1
    assay = assay or {}
    g = torch.Generator(device="cpu").manual_seed(mpp.SEED)
    rows, done_blocks, methods = [], set(), set()
    if part_path and os.path.exists(part_path):
        try:
            p = json.load(open(part_path)); rows = p["rows"]; done_blocks = set(p["done_blocks"]); methods = set(p.get("methods", []))
            log(f"  resuming: {len(done_blocks)} blocks already done")
        except Exception:
            pass
    t0 = time.time()
    # identical draw order to matched_pretrained.run_model: all n_in inputs x positions, N_RAND directions each
    d = b["model"].config.hidden_size
    rand_dirs = {}
    for i in range(n_in):
        for t in positions:
            U = torch.randn(mpp.N_RAND, d, generator=g); rand_dirs[(i, t)] = U / U.norm(dim=1, keepdim=True)
    for li in blocks:
        if li in done_blocks:
            continue
        tb = time.time(); n0 = len(rows)
        for i in range(n_lin):
            ids_1 = ids[i:i + 1].to(DEVICE)
            captured = sv.capture_layer_inputs(b, {"input_ids": ids_1})
            hidden0 = captured[li][0]
            sig, v1, n_it = mpp.sigma1_block(b, li, captured, positions)
            for p, t in enumerate(positions):
                h_t = hidden0[0, t].float(); hn = float(h_t.norm())
                dirs = [("v1", v1[p])] + [(f"rand{r}", rand_dirs[(i, t)][r].to(DEVICE)) for r in range(mpp.N_RAND)]
                G, p0, method = logit_response(b, li, ids_1, h_t, t, dirs); methods.add(method)
                st = gauge_stats(G, p0)
                kl_meas = kl_v1_check(b, li, ids_1, h_t, t, v1[p])
                row = dict(block=li, input=i, position=t, sigma1=float(sig[p]), power_iters=n_it, h_norm=hn, directions=[n for n, _ in dirs],
                           kl_pred_v1=0.5 * (EPS_CHECK * hn) ** 2 * st["quad"][0], kl_meas_v1=kl_meas, **st)
                a = assay.get((li, i, t))
                if a:
                    row["assay"] = dict(sigma1=a["sigma1"], amp_v1=a["amp_v1"], quad_v1=a["quad_v1"], amp_rand=a["amp_rand"], quad_rand=a["quad_rand"])
                rows.append(row)
                del G
            del captured
        done_blocks.add(li)
        rb = rows[n0:]
        rep = ""
        if rb and "assay" in rb[0]:
            dq = np.median([abs(r["quad"][0] / r["assay"]["quad_v1"] - 1) for r in rb])
            ds = np.median([abs(r["sigma1"] / r["assay"]["sigma1"] - 1) for r in rb]); rep = f"  |dev| vs assay: quad {dq:.1e} sigma1 {ds:.1e}"
        ratio = np.median([r["kl_pred_v1"] / r["kl_meas_v1"] for r in rb if r["kl_meas_v1"] > 0])
        c = np.median([abs(r["g_mean"][0]) / np.sqrt(r["amplification"][0] / r["V"]) for r in rb])
        log(f"    block {li:2d}: sigma1 {np.mean([r['sigma1'] for r in rb]):7.3f}  pred/meas KL(v1, .01) {ratio:.3f}  |mean g|/rms g (v1) {c:.3f}"
            f"  centred/raw amp (v1) {np.median([r['amplification_centred'][0] / r['amplification'][0] for r in rb]):.3f}{rep}  ({time.time() - tb:.0f} s)")
        if part_path:
            mpp.write_json(part_path, dict(rows=rows, done_blocks=sorted(done_blocks), methods=sorted(methods)))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    rows.sort(key=lambda r: (r["block"], r["input"], r["position"]))
    return rows, sorted(methods), time.time() - t0


def per_block_summary(rows):
    out = []
    for li in sorted({r["block"] for r in rows}):
        rb = [r for r in rows if r["block"] == li]
        rec = dict(block=li, sigma1=float(np.mean([r["sigma1"] for r in rb])), n=len(rb),
                   pred_over_meas_v1=float(np.median([r["kl_pred_v1"] / r["kl_meas_v1"] for r in rb if r["kl_meas_v1"] > 0])))
        for k in ("amplification", "quad", "alignment", "amplification_centred", "alignment_centred", "amplification_pcentred", "alignment_pcentred"):
            rec[f"{k}_v1"] = float(np.mean([r[k][0] for r in rb])); rec[f"{k}_rand"] = float(np.mean([np.mean(r[k][1:]) for r in rb]))
        out.append(rec)
    return out


def result_path(res_dir, hf_id, quick=False):
    return os.path.join(res_dir, f"lc_{sv.safe_name(hf_id)}{'_quick' if quick else ''}.json")


def run_job(hf_id, res_dir, quick=False, bundle=None):
    path = result_path(res_dir, hf_id, quick)
    if os.path.exists(path):
        try:
            if json.load(open(path)).get("version") == VERSION:
                log(f"[SKIP] {hf_id}: done"); return
        except Exception:
            pass
    log("\n" + "-" * 78 + f"\n{hf_id}\n" + "-" * 78)
    b = bundle or mpp.load(hf_id)
    ids = mpp.model_inputs(b, mpp.N_INPUTS)
    rows, methods, secs = run_model(b, ids, path.replace(".json", "_partial.json"), quick, assay_records(hf_id) if not quick else {})
    out = dict(hf_id=hf_id, label=b["cfg"]["label"], arch=b["arch"], family=mpp.FAMILY.get(b["arch"], b["arch"]), n_layers=len(b["layers"]),
               d_model=int(b["model"].config.hidden_size), version=VERSION, assay_version=mpp.VERSION, eps_check=EPS_CHECK,
               positions=mpp.POSITIONS if not quick else mpp.POSITIONS[:2], n_inputs_drawn=int(ids.shape[0]) if not quick else 2,
               n_lin_inputs=mpp.N_LIN_INPUTS if not quick else 1, n_random_directions=mpp.N_RAND, bos_prepended=b["arch"] in mpp.BOS_ARCHES,
               inputs="WikiText-103 validation, survey_sigma1_v2.natural_text_ids, 128 tokens" + (" (BOS prepended, last token dropped)" if b["arch"] in mpp.BOS_ARCHES else ""),
               estimator="token-local J^T J power iteration on the block (survey_sigma1_v2 in-context map), jvp + vjp, float32, 2 restarts, 50 iterations",
               directions="v_1 and 8 random unit vectors drawn once per (input, position) with the assay's generator and seed",
               definition=("g = A u, forward-mode product of the position-t logits w.r.t. the block-l input at t (block l included); "
                           "amplification = ||g||^2; quad = g^T F g, F = diag(p) - p p^T; alignment = quad / amplification; "
                           "amplification_centred = ||g - mean(g) 1||^2; amplification_pcentred = ||g - (p^T g) 1||^2; "
                           "kl_pred_v1 = 1/2 eps^2 ||h||^2 quad(v1); kl_meas_v1 = same-position KL(clean || perturbed) along v1, signs averaged; "
                           "assay = the 23 Sept matched assay's values on the same probe"),
               lin_methods=methods, per_block=per_block_summary(rows), raw=rows, quick=quick, elapsed_seconds=secs,
               **{f"prov_{k}": v for k, v in mpp.provenance().items()})
    mpp.write_json(path, out)
    pp = path.replace(".json", "_partial.json")
    if os.path.exists(pp):
        os.remove(pp)
    log(f"  saved {path} ({secs / 60:.1f} min)")
    del b; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def summary(res_dir):
    """From the JSONs on disk: per model, the v1/random ratios (geometric mean over probes then blocks 1 to L-2) of amplification
    and alignment in the raw, mean-centred and p-centred gauges, the quad ratio (gauge-free), the prediction check along v1 and
    the median relative deviation of quad from the assay on the same probes."""
    log(f"\n{'model':16s} {'L':>3s} {'amp v1/rand':>11s} {'align':>7s} | {'centred amp':>11s} {'align':>7s} | {'p-centred amp':>13s} {'align':>7s} | {'quad ratio':>10s} {'pred/meas':>9s} {'dev. assay':>10s}")
    rows = []
    for fn in sorted(os.listdir(res_dir)):
        if not (fn.startswith("lc_") and fn.endswith(".json")) or fn.endswith("_partial.json"):
            continue
        d = json.load(open(os.path.join(res_dir, fn))); L = d["n_layers"]
        raw = [r for r in d["raw"] if 1 <= r["block"] <= L - 2] or d["raw"]

        def ratio(key):
            per = {}
            for r in raw:
                per.setdefault(r["block"], []).append(np.log(max(r[key][0], 1e-300) / max(float(np.mean(r[key][1:])), 1e-300)))
            return float(np.exp(np.mean([np.mean(v) for v in per.values()])))
        dev = [abs(r["quad"][0] / r["assay"]["quad_v1"] - 1) for r in raw if "assay" in r]
        rec = dict(model=d["label"], L=L, amp=ratio("amplification"), align=ratio("alignment"), amp_c=ratio("amplification_centred"), align_c=ratio("alignment_centred"),
                   amp_p=ratio("amplification_pcentred"), align_p=ratio("alignment_pcentred"), quad=ratio("quad"),
                   pred_over_meas=float(np.median([r["kl_pred_v1"] / r["kl_meas_v1"] for r in raw if r["kl_meas_v1"] > 0])),
                   dev_assay_quad=float(np.median(dev)) if dev else None)
        rows.append(rec)
        dev_s = f"{rec['dev_assay_quad']:10.1e}" if rec["dev_assay_quad"] is not None else f"{'':10s}"
        log(f"{rec['model']:16s} {L:3d} {rec['amp']:11.2f} {rec['align']:7.2f} | {rec['amp_c']:11.2f} {rec['align_c']:7.2f} | {rec['amp_p']:13.2f} {rec['align_p']:7.2f} | "
            f"{rec['quad']:10.2f} {rec['pred_over_meas']:9.3f} {dev_s}")
    return rows


def export(res_dir):
    stage = os.path.join(OUT, "lc_export")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    n = 0
    for fn in os.listdir(res_dir):
        if fn.endswith(".json") and not fn.endswith("_partial.json"):
            shutil.copy(os.path.join(res_dir, fn), stage); n += 1
    zs = [shutil.make_archive(os.path.join(OUT, "linresp_pretrained_centred_results"), "zip", stage)]
    if os.path.abspath(os.getcwd()) != os.path.abspath(OUT):
        zs.append(shutil.make_archive(os.path.join(os.getcwd(), "linresp_pretrained_centred_results"), "zip", stage))
    for z in zs:
        log(f"export: {n} files -> {z} ({os.path.getsize(z) / 1e6:.1f} MB)")
    return zs


def plumbing():
    """Tiny random gpt2 / pythia / llama / qwen / gemma2 (as matched_pretrained.plumbing), quick mode: checks the gauge identities,
    that amplification and quad equal matched_pretrained.linresp on the same directions, that the second-order prediction matches
    the measured KL along v1 at eps 0.01, then summary() and export()."""
    from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM,
                              Qwen2Config, Qwen2ForCausalLM, Gemma2Config, Gemma2ForCausalLM)
    torch.manual_seed(0); V = 512
    common = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=256)

    def eager(c):
        c._attn_implementation = "eager"; return c
    builders = {
        "gpt2": lambda: GPT2LMHeadModel(eager(GPT2Config(vocab_size=V, n_embd=64, n_layer=4, n_head=4, n_positions=256))),
        "pythia": lambda: GPTNeoXForCausalLM(eager(GPTNeoXConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                                                 num_attention_heads=4, max_position_embeddings=256, rotary_pct=0.25))),
        "llama": lambda: LlamaForCausalLM(eager(LlamaConfig(**common))),
        "qwen": lambda: Qwen2ForCausalLM(eager(Qwen2Config(**common))),
        "gemma2": lambda: Gemma2ForCausalLM(eager(Gemma2Config(**common, head_dim=16, sliding_window=16))),
    }
    res_dir = os.path.join(OUT, "results_plumbing"); os.makedirs(res_dir, exist_ok=True)
    checks = {}
    for arch, build in builders.items():
        model = build().to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cfg = dict(hf_id=f"plumbing/{arch}", arch=arch, label=f"plumbing-{arch}", family=mpp.FAMILY[arch], params_M=0, april=False)
        sv.CFG[cfg["hf_id"]] = cfg
        b = dict(arch=arch, hf_id=cfg["hf_id"], cfg=cfg, dtype=torch.float32, kind="text", vocab_size=V, model=model,
                 layers=sv.get_layers(model, arch), forward=lambda m, inp: m(input_ids=inp["input_ids"], use_cache=False))
        ids = torch.randint(0, V, (2, sv.SEQ_LEN), generator=torch.Generator().manual_seed(1))
        rows, methods, secs = run_model(b, ids, None, quick=True)
        # gauge identities and ordering
        for r in rows:
            for j in range(len(r["directions"])):
                a, ac, ap, gm, gp = r["amplification"][j], r["amplification_centred"][j], r["amplification_pcentred"][j], r["g_mean"][j], r["g_pmean"][j]
                assert abs(ac - (a - r["V"] * gm ** 2)) < 1e-3 * max(a, 1e-6), "centred identity"
                assert ac <= a * (1 + 1e-5) and ap >= ac * (1 - 1e-5), "gauge ordering"
                assert abs(r["quad"][j] / ac - r["alignment_centred"][j]) < 1e-6 * max(abs(r["alignment_centred"][j]), 1e-9), "alignment_centred"
        # same amplification and quad as the assay's linresp on the same directions (first row of the run)
        r0 = rows[0]; ids_1 = ids[0:1].to(DEVICE); captured = sv.capture_layer_inputs(b, {"input_ids": ids_1})
        li, t = r0["block"], r0["position"]; hidden0 = captured[li][0]; h_t = hidden0[0, t].float()
        sig, v1, _ = mpp.sigma1_block(b, li, captured, mpp.POSITIONS[:2])
        g = torch.Generator(device="cpu").manual_seed(mpp.SEED); U = torch.randn(mpp.N_RAND, b["model"].config.hidden_size, generator=g); U = U / U.norm(dim=1, keepdim=True)
        dirs = [("v1", v1[0])] + [(f"rand{k}", U[k].to(DEVICE)) for k in range(mpp.N_RAND)]
        amp_a, quad_a, _ = mpp.linresp(b, li, ids_1, h_t, t, dirs)
        d_amp = float(np.max(np.abs(np.array(r0["amplification"]) / amp_a - 1))); d_quad = float(np.max(np.abs(np.array(r0["quad"]) / quad_a - 1)))
        pred_ratio = float(np.median([r["kl_pred_v1"] / r["kl_meas_v1"] for r in rows if r["kl_meas_v1"] > 0]))
        checks[arch] = dict(max_dev_amp_vs_assay=d_amp, max_dev_quad_vs_assay=d_quad, pred_over_meas=pred_ratio, lin_methods=methods, secs=secs)
        log(f"  {arch}: amp/quad vs matched_pretrained.linresp max |dev| {d_amp:.1e} / {d_quad:.1e}; linearised/measured KL(v1, eps 0.01) = {pred_ratio:.4f} (expect ~1); "
            f"via {methods}; {secs:.0f} s")
        assert d_amp < 1e-3 and d_quad < 1e-3, f"{arch}: does not reproduce the assay's linearised response"
        assert 0.8 < pred_ratio < 1.25, f"{arch}: linearised prediction does not match the measured KL"
        out = dict(hf_id=cfg["hf_id"], label=cfg["label"], arch=arch, family=mpp.FAMILY[arch], n_layers=len(b["layers"]), version=VERSION,
                   per_block=per_block_summary(rows), raw=rows, quick=True)
        mpp.write_json(os.path.join(res_dir, f"lc_plumbing_{arch}.json"), out)
    summary(res_dir)
    zs = export(res_dir)
    assert all(os.path.getsize(z) > 0 for z in zs)
    mpp.write_json(os.path.join(res_dir, "plumbing_checks.json"), checks)
    log("plumbing OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", default=bool(os.environ.get("MP_QUICK")))
    ap.add_argument("--plumbing", action="store_true")
    args, _ = ap.parse_known_args(argv)
    log(f"linearised response, pretrained decoders, gauge recorded | {VERSION} | {mpp.provenance()} | out={OUT} | assay={MP_RESULTS or '-'}")
    if args.plumbing:
        plumbing(); return
    sv._resolve_hf_token()
    res_dir = os.path.join(OUT, "results"); os.makedirs(res_dir, exist_ok=True)
    models = args.models or (os.environ.get("MP_MODELS", "").split() or mpp.MODELS)
    for hf_id in models:
        try:
            run_job(hf_id, res_dir, args.quick)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"  FAILED {hf_id}: {e}")
        if os.environ.get("MP_PURGE_CACHE"):
            sv._purge_hub_cache(hf_id)
    summary(res_dir); export(res_dir)


if __name__ == "__main__":
    main()
