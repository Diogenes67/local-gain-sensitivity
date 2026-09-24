"""
Matched token-local perturbation assay and downstream linearised response in pretrained decoders (23 Sept 2026)
==============================================================================================================
Question (NMI review 2, item 4). The scale result (gain predicts consequence per unit squared displacement, not at a
fixed fraction of the activation norm) and the direction result (v_1 costs more than a random direction, and the excess
is downstream amplification) were measured in 50 small controlled transformers. This script repeats both measurements,
with the same definitions, in six pretrained causal decoders from five families.

Models: gpt2, EleutherAI/pythia-410m, EleutherAI/pythia-1.4b, meta-llama/Llama-3.2-1B, Qwen/Qwen2.5-1.5B,
google/gemma-2-2b (BOS prepended, as in the census). Llama and Gemma are gated: the HF token needs access.

Per model: 20 WikiText-103 validation sequences of 128 tokens (survey_sigma1_v2.natural_text_ids), positions
t in {8, 32, 56, 80, 104, 120}, every block. Per (block, input, position):
  sigma_1, v_1   token-local J^T J power iteration on the block (survey_sigma1_v2's in-context map; 2 restarts, 50 iterations)
  matched assay  h_{l,t} += sign * eps * ||h_{l,t}|| * u, u in {v_1, 8 random unit vectors}, eps in {0.01, 0.03, 0.1, 0.3},
                 signs averaged; readouts: local gain ||dy_t|| / ||delta||, KL at t, mean KL over t..T-2, change of the
                 next-token loss over t..T-2 (identical to matched_perturbation.py)
  linearised     on the first 5 inputs: g = A u (forward-mode product of the position-t logits w.r.t. the block-l input at t,
  response       whole downstream stack including block l) for v_1 and the same 8 random directions; amplification ||g||^2,
                 alignment g^T F g / ||g||^2 with F = diag(p) - p p^T (identical to linearised_response.py)

Outputs: one JSON per model in $MP_OUT/results (atomic; a partial file is written after every block, so a disconnect
loses at most one block), a summary table from disk, and matched_pretrained_results.zip in $MP_OUT and in the working
directory. Needs survey_sigma1_v2.py beside it.

Usage: python matched_pretrained.py [--models gpt2 ...] [--plumbing]
Environment: MP_OUT (default ./matched_pretrained), HF_TOKEN, MP_LOGIT_BUDGET_GB (default 6), MP_QUICK=1
"""
import os, sys, json, time, gc, argparse, shutil, datetime, platform
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as sv

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

DEVICE = sv.DEVICE
OUT = os.environ.get("MP_OUT", "matched_pretrained")
VERSION = "matched-pretrained-v1-2026-09-23"
SEED = 42
MODELS = ["gpt2", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B", "google/gemma-2-2b"]
FAMILY = {"gpt2": "GPT-2", "pythia": "Pythia", "llama": "Llama", "qwen": "Qwen", "gemma2": "Gemma"}
BOS_ARCHES = {"gemma2"}
EPS = [0.01, 0.03, 0.1, 0.3]
N_RAND = 8
N_INPUTS = 20
POSITIONS = [8, 32, 56, 80, 104, 120]
N_LIN_INPUTS = 5
PI_ITERS, PI_RESTARTS, PI_TOL = 50, 2, 1e-6
LOGIT_BUDGET = float(os.environ.get("MP_LOGIT_BUDGET_GB", "6")) * 1e9


def log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------- helpers around survey_sigma1_v2
def _hidden_from(args, kwargs):
    return args[0] if len(args) else kwargs["hidden_states"]


def _with_hidden(args, kwargs, h):
    if len(args):
        return (h,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs); kwargs["hidden_states"] = h
    return args, kwargs


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def top_singular(f, x0, iters=PI_ITERS, restarts=PI_RESTARTS, tol=PI_TOL):
    """Batched power iteration on J^T J (rows independent). Returns sigma_1 (P,), v_1 (P, d), mean iterations."""
    P, d = x0.shape
    best_s = torch.zeros(P, device=x0.device); best_v = torch.zeros_like(x0); its = []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(restarts):
        v = torch.randn(P, d, generator=g).to(x0.device); v = v / v.norm(dim=1, keepdim=True)
        n_it = iters
        for it in range(iters):
            _, Jv = torch.func.jvp(f, (x0,), (v,))
            _, vjp_fn = torch.func.vjp(f, x0)
            JtJv = vjp_fn(Jv)[0]
            v_new = JtJv / (JtJv.norm(dim=1, keepdim=True) + 1e-30)
            cos = (v_new * v).sum(1).abs(); v = v_new
            if bool((1 - cos).max() < tol):
                n_it = it + 1; break
        _, Jv = torch.func.jvp(f, (x0,), (v,))
        s = Jv.norm(dim=1); better = s > best_s
        best_s = torch.where(better, s, best_s); best_v[better] = v[better]; its.append(n_it)
    return best_s.detach(), best_v.detach(), float(np.mean(its))


def sigma1_block(b, li, captured, positions):
    hidden0, rest, kw = captured[li]
    call = sv.make_layer_call(b["layers"][li], rest, kw)
    layout = sv.Layout(hidden0)
    x0 = layout.get(hidden0.float(), positions).clone()
    try:
        f = sv.make_f_incontext(call, hidden0, layout, positions, False, torch.float32)
        return top_singular(f, x0)
    except Exception as e:
        log(f"    batched positions failed at block {li} ({type(e).__name__}: {str(e)[:80]}); one position at a time")
        out = [top_singular(sv.make_f_incontext(call, hidden0, layout, [t], False, torch.float32), x0[j:j + 1]) for j, t in enumerate(positions)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]), float(np.mean([o[2] for o in out]))


# ---------------------------------------------------------------- matched perturbation
@torch.no_grad()
def perturb_and_read(b, li, ids_1, h_t, t, dirs):
    """Batched forwards, each chunk starting with a clean row. Returns per perturbed row: local gain, KL at t,
    mean KL over t..T-2 and delta-CE over t..T-2, with meta (direction, eps, sign)."""
    norm = float(h_t.norm())
    rows, meta = [], []
    for name, u in dirs:
        for e in EPS:
            for sgn in (1, -1):
                rows.append(sgn * e * norm * u); meta.append((name, e, sgn))
    delta_all = torch.stack(rows)
    T = ids_1.shape[1]; V = b["vocab_size"]
    per_row = T * V * 4 * 3
    chunk = max(1, int(LOGIT_BUDGET // per_row) - 1)
    layer = b["layers"][li]
    res = {k: [] for k in ("gain", "kl_t", "kl_seq", "dce")}
    clean_loss = None
    for c0 in range(0, len(rows), chunk):
        delta = torch.cat([torch.zeros_like(h_t)[None], delta_all[c0:c0 + chunk]])
        B = delta.shape[0]
        store = {}

        def pre(m, args, kwargs):
            h = _hidden_from(args, kwargs).clone()
            h[:, t] = h[:, t] + delta.to(h.dtype)
            return _with_hidden(args, kwargs, h)

        hp = layer.register_forward_pre_hook(pre, with_kwargs=True)
        ho = layer.register_forward_hook(lambda m, a, o: store.__setitem__("y", _first(o)[:, t].detach().float().clone()))
        try:
            logits = _first(b["forward"](b["model"], {"input_ids": ids_1.expand(B, -1)}).logits).float()
        finally:
            hp.remove(); ho.remove()
        y = store["y"]
        gain = (y[1:] - y[0:1]).norm(dim=1) / delta[1:].norm(dim=1)
        logp = F.log_softmax(logits, -1); cl = logp[0:1]
        kl_all = (cl.exp() * (cl - logp)).sum(-1)
        lg = logits[:, t:-1]; tg = ids_1[:, t + 1:].expand(B, -1)
        loss = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1), reduction="none").view(B, -1).mean(1)
        res["gain"].append(gain.cpu()); res["kl_t"].append(kl_all[1:, t].cpu()); res["kl_seq"].append(kl_all[1:, t:-1].mean(1).cpu())
        res["dce"].append((loss[1:] - loss[0]).cpu()); clean_loss = float(loss[0])
        del logits, logp, kl_all
    out = {k: torch.cat(v).numpy() for k, v in res.items()}
    out.update(meta=meta, h_norm=norm, clean_loss=clean_loss)
    return out


def aggregate(meta, arr):
    """Sign-averaged mean per (class, eps) for v1 and the random class, plus the per-random-direction values."""
    out = {"v1": [], "rand": [], "rand_each": []}
    names = sorted({n for n, _, _ in meta if n != "v1"})
    for e in EPS:
        out["v1"].append(float(np.mean([arr[i] for i, (n, ee, _) in enumerate(meta) if n == "v1" and ee == e])))
        each = [float(np.mean([arr[i] for i, (n, ee, _) in enumerate(meta) if n == r and ee == e])) for r in names]
        out["rand_each"].append(each); out["rand"].append(float(np.mean(each)))
    return out


# ---------------------------------------------------------------- linearised downstream response
def linresp(b, li, ids_1, h0, t, dirs):
    """g = A u for every direction in one batched forward-mode product; returns amplification, alignment and quad."""
    layer = b["layers"][li]
    U = torch.stack([u for _, u in dirs]); B = U.shape[0]
    X0 = h0[None].expand(B, -1).clone()

    def fl(x):
        def pre(m, args, kwargs):
            h = _hidden_from(args, kwargs)
            h = torch.cat([h[:, :t], x[:, None, :].to(h.dtype), h[:, t + 1:]], 1)
            return _with_hidden(args, kwargs, h)
        hk = layer.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            logits = _first(b["forward"](b["model"], {"input_ids": ids_1.expand(B, -1)}).logits)
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
        G = G.float(); amp = (G * G).sum(1)
        pg = (G * p0).sum(1); quad = (G * G * p0).sum(1) - pg ** 2
    return amp.cpu().numpy(), quad.cpu().numpy(), method


# ---------------------------------------------------------------- one model
def load(hf_id):
    cfg = sv.CFG[hf_id]
    b = sv.load_bundle(cfg, "float32")
    return b


def model_inputs(b, n):
    ids = sv.natural_text_ids(b["tokenizer"], n)
    if b["arch"] in BOS_ARCHES and b["tokenizer"].bos_token_id is not None:
        ids = torch.cat([torch.full((ids.shape[0], 1), b["tokenizer"].bos_token_id, dtype=ids.dtype), ids[:, :-1]], 1)
    return ids


def run_model(b, ids, part_path=None, quick=False):
    L = len(b["layers"])
    blocks = list(range(L)) if not quick else [0, L // 2, L - 1]
    positions = POSITIONS if not quick else POSITIONS[:2]
    n_in = ids.shape[0] if not quick else 2
    n_lin = min(N_LIN_INPUTS, n_in) if not quick else 1
    g = torch.Generator(device="cpu").manual_seed(SEED)
    raw, lin_raw, per_block = [], [], []
    done_blocks = set()
    if part_path and os.path.exists(part_path):
        try:
            p = json.load(open(part_path)); raw, lin_raw, per_block = p["raw"], p["lin_raw"], p["per_block"]
            done_blocks = {pb["block"] for pb in per_block}
            log(f"  resuming: {len(done_blocks)} blocks already done")
        except Exception:
            pass
    lin_methods = set()
    t0 = time.time()
    # random directions are drawn once per (input, position) and reused for every block, so the random class is the
    # same set of vectors at every block (as the matched assay drew them per probe, the distribution is identical)
    d = b["model"].config.hidden_size
    rand_dirs = {}
    for i in range(n_in):
        for t in positions:
            U = torch.randn(N_RAND, d, generator=g); rand_dirs[(i, t)] = U / U.norm(dim=1, keepdim=True)
    for li in blocks:
        if li in done_blocks:
            continue
        tb = time.time(); rows_b = []
        for i in range(n_in):
            ids_1 = ids[i:i + 1].to(DEVICE)
            captured = sv.capture_layer_inputs(b, {"input_ids": ids_1})
            hidden0 = captured[li][0]
            sig, v1, n_it = sigma1_block(b, li, captured, positions)
            for p, t in enumerate(positions):
                h_t = hidden0[0, t].float()
                dirs = [("v1", v1[p])] + [(f"rand{r}", rand_dirs[(i, t)][r].to(DEVICE)) for r in range(N_RAND)]
                res = perturb_and_read(b, li, ids_1, h_t, t, dirs)
                ag = {k: aggregate(res["meta"], res[k]) for k in ("gain", "kl_t", "kl_seq", "dce")}
                row = dict(block=li, input=i, position=t, sigma1=float(sig[p]), power_iters=n_it, h_norm=res["h_norm"], clean_loss=res["clean_loss"],
                           **{k: dict(v1=ag[k]["v1"], rand=ag[k]["rand"]) for k in ag})
                row["kl_t"]["rand_each"] = ag["kl_t"]["rand_each"]; row["gain"]["rand_each"] = ag["gain"]["rand_each"]
                raw.append(row); rows_b.append(row)
                if i < n_lin:
                    amp, quad, method = linresp(b, li, ids_1, h_t, t, dirs); lin_methods.add(method)
                    lin_raw.append(dict(block=li, input=i, position=t, sigma1=float(sig[p]), h_norm=res["h_norm"],
                                        amp_v1=float(amp[0]), quad_v1=float(quad[0]), amp_rand=amp[1:].tolist(), quad_rand=quad[1:].tolist(),
                                        kl_t_v1=ag["kl_t"]["v1"], kl_t_rand_each=ag["kl_t"]["rand_each"],
                                        gain_v1=ag["gain"]["v1"], gain_rand_each=ag["gain"]["rand_each"]))
            del captured
        m = lambda k, c: np.mean([r[k][c] for r in rows_b], 0).tolist()
        pb = dict(block=li, sigma1=float(np.mean([r["sigma1"] for r in rows_b])), h_norm=float(np.mean([r["h_norm"] for r in rows_b])),
                  gain_v1=m("gain", "v1"), gain_rand=m("gain", "rand"), kl_t_v1=m("kl_t", "v1"), kl_t_rand=m("kl_t", "rand"),
                  kl_seq_v1=m("kl_seq", "v1"), kl_seq_rand=m("kl_seq", "rand"), dce_v1=m("dce", "v1"), dce_rand=m("dce", "rand"),
                  S_abs=[float(np.mean([r["kl_seq"]["v1"][k] / (e * r["h_norm"]) ** 2 for r in rows_b])) for k, e in enumerate(EPS)])
        per_block.append(pb)
        ie = EPS.index(0.1)
        log(f"    block {li:2d}: sigma1 {pb['sigma1']:7.3f}  gain(v1,.01)/sigma1 {pb['gain_v1'][0] / pb['sigma1']:.3f}  ||h|| {pb['h_norm']:9.2f}  "
            f"KLseq v1 {pb['kl_seq_v1'][ie]:.2e} rand {pb['kl_seq_rand'][ie]:.2e}  ({time.time() - tb:.0f} s)")
        if part_path:
            write_json(part_path, dict(raw=raw, lin_raw=lin_raw, per_block=per_block))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    per_block.sort(key=lambda x: x["block"])
    return per_block, raw, lin_raw, sorted(lin_methods), time.time() - t0


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def provenance():
    p = {"torch": torch.__version__, "transformers": sv._tf_version(), "python": platform.python_version(),
         "date": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE}
    if torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name()
    return p


def result_path(res_dir, hf_id, quick=False):
    return os.path.join(res_dir, f"mp_{sv.safe_name(hf_id)}{'_quick' if quick else ''}.json")


def run_job(hf_id, res_dir, quick=False, bundle=None):
    path = result_path(res_dir, hf_id, quick)
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            if d.get("version") == VERSION:
                log(f"[SKIP] {hf_id}: done"); return
        except Exception:
            pass
    log("\n" + "-" * 78 + f"\n{hf_id}\n" + "-" * 78)
    b = bundle or load(hf_id)
    ids = model_inputs(b, N_INPUTS)
    per_block, raw, lin_raw, lin_methods, secs = run_model(b, ids, path.replace(".json", "_partial.json"), quick)
    out = dict(hf_id=hf_id, label=b["cfg"]["label"], arch=b["arch"], family=FAMILY.get(b["arch"], b["arch"]), n_layers=len(b["layers"]),
               d_model=int(b["model"].config.hidden_size), version=VERSION, eps=EPS, positions=POSITIONS if not quick else POSITIONS[:2],
               n_inputs=int(ids.shape[0]) if not quick else 2, n_lin_inputs=N_LIN_INPUTS if not quick else 1, n_random_directions=N_RAND,
               bos_prepended=b["arch"] in BOS_ARCHES,
               inputs="WikiText-103 validation, survey_sigma1_v2.natural_text_ids, 128 tokens" + (" (BOS prepended, last token dropped)" if b["arch"] in BOS_ARCHES else ""),
               estimator="token-local J^T J power iteration on the block (survey_sigma1_v2 in-context map), jvp + vjp, float32, 2 restarts, 50 iterations",
               perturbation="h_{l,t} += sign * eps * ||h_{l,t}|| * u, u in {v_1, 8 random unit vectors drawn once per (input, position)}, signs averaged; pre-hook on the block input",
               readouts=dict(gain="||block output change at t|| / ||delta||", kl_t="KL(clean || perturbed) at t", kl_seq="mean KL over t..T-2",
                             dce="change of next-token CE over t..T-2", S_abs="kl_seq(v1) / (eps ||h||)^2 averaged over probes",
                             linresp="g = A u, forward-mode product of the position-t logits w.r.t. the block-l input at t; amp = ||g||^2; quad = g^T F g, F = diag(p) - p p^T"),
               lin_methods=lin_methods, per_block=per_block, raw=raw, lin_raw=lin_raw, quick=quick, elapsed_seconds=secs,
               **{f"prov_{k}": v for k, v in provenance().items()})
    write_json(path, out)
    pp = path.replace(".json", "_partial.json")
    if os.path.exists(pp):
        os.remove(pp)
    log(f"  saved {path} ({secs / 60:.1f} min)")
    del b; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary and export (from disk)
def summary(res_dir):
    from scipy import stats
    ie = EPS.index(0.1)
    log(f"\n{'model':22s} {'L':>3s} {'g/s1':>6s} {'h ratio':>8s} {'rho rel':>8s} {'rho abs':>8s} {'v1/rand KLt':>11s} {'(s1/gr)^2':>9s}")
    rows = []
    for fn in sorted(os.listdir(res_dir)):
        if not (fn.startswith("mp_") and fn.endswith(".json")) or fn.endswith("_partial.json"):
            continue
        d = json.load(open(os.path.join(res_dir, fn))); pb = d["per_block"]; B = [x for x in pb if 1 <= x["block"] <= d["n_layers"] - 2]
        if len(B) < 3:
            continue
        s1 = [x["sigma1"] for x in B]; rel = [x["kl_seq_v1"][ie] for x in B]; S = [x["S_abs"][ie] for x in B]
        klr = np.median([x["kl_t_v1"][ie] / x["kl_t_rand"][ie] for x in B]); gr2 = np.median([(x["gain_v1"][ie] / x["gain_rand"][ie]) ** 2 for x in B])
        r = (d["label"], d["n_layers"], np.mean([x["gain_v1"][0] / x["sigma1"] for x in pb]), B[-1]["h_norm"] / B[0]["h_norm"],
             stats.spearmanr(s1, rel)[0], stats.spearmanr(s1, S)[0], klr, gr2)
        rows.append(r)
        log(f"{r[0]:22s} {r[1]:3d} {r[2]:6.3f} {r[3]:8.1f} {r[4]:+8.2f} {r[5]:+8.2f} {r[6]:11.1f} {r[7]:9.1f}")
    return rows


def export(res_dir):
    stage = os.path.join(OUT, "mp_export")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    n = 0
    for fn in os.listdir(res_dir):
        if fn.endswith(".json"):
            shutil.copy(os.path.join(res_dir, fn), stage); n += 1
    zs = [shutil.make_archive(os.path.join(OUT, "matched_pretrained_results"), "zip", stage)]
    if os.path.abspath(os.getcwd()) != os.path.abspath(OUT):
        zs.append(shutil.make_archive(os.path.join(os.getcwd(), "matched_pretrained_results"), "zip", stage))
    for z in zs:
        log(f"export: {n} files -> {z} ({os.path.getsize(z) / 1e6:.1f} MB)")
    return zs


# ---------------------------------------------------------------- plumbing test (tiny random models, no Hub access)
def plumbing():
    """Tiny randomly initialised gpt2 / pythia / llama / qwen / gemma2; checks that the finite gain along v1 at eps 0.01
    reproduces sigma1, that the linearised prediction matches the measured same-position KL at eps 0.01, and that
    summary() and export() run."""
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
        cfg = dict(hf_id=f"plumbing/{arch}", arch=arch, label=f"plumbing-{arch}", family=FAMILY[arch], params_M=0, april=False)
        sv.CFG[cfg["hf_id"]] = cfg
        b = dict(arch=arch, hf_id=cfg["hf_id"], cfg=cfg, dtype=torch.float32, kind="text", vocab_size=V, model=model,
                 layers=sv.get_layers(model, arch), forward=lambda m, inp: m(input_ids=inp["input_ids"], use_cache=False))
        ids = torch.randint(0, V, (2, sv.SEQ_LEN), generator=torch.Generator().manual_seed(1))
        per_block, raw, lin_raw, methods, secs = run_model(b, ids, None, quick=True)
        g_over_s = float(np.mean([pb["gain_v1"][0] / pb["sigma1"] for pb in per_block]))
        # second-order prediction vs measured same-position KL at eps 0.01 (v1)
        ratios = [0.5 * (0.01 * r["h_norm"]) ** 2 * r["quad_v1"] / r["kl_t_v1"][0] for r in lin_raw if r["kl_t_v1"][0] > 0]
        pred_ratio = float(np.median(ratios))
        checks[arch] = dict(gain_over_sigma1=g_over_s, pred_over_meas=pred_ratio, lin_methods=methods, secs=secs)
        log(f"  {arch}: gain(v1, eps 0.01)/sigma1 = {g_over_s:.4f} (expect ~1); linearised/measured KL at eps 0.01 = {pred_ratio:.4f} (expect ~1); "
            f"linresp via {methods}; {secs:.0f} s")
        assert 0.9 < g_over_s < 1.1, f"{arch}: finite perturbation along v1 does not reproduce sigma1"
        assert 0.8 < pred_ratio < 1.25, f"{arch}: linearised prediction does not match the measured KL"
        out = dict(hf_id=cfg["hf_id"], label=cfg["label"], arch=arch, family=FAMILY[arch], n_layers=len(b["layers"]), version=VERSION, eps=EPS,
                   per_block=per_block, raw=raw, lin_raw=lin_raw, quick=True)
        write_json(os.path.join(res_dir, f"mp_plumbing_{arch}.json"), out)
    summary(res_dir)
    zs = export(res_dir)
    assert all(os.path.getsize(z) > 0 for z in zs)
    write_json(os.path.join(res_dir, "plumbing_checks.json"), checks)
    log("plumbing OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", default=bool(os.environ.get("MP_QUICK")))
    ap.add_argument("--plumbing", action="store_true")
    args, _ = ap.parse_known_args(argv)
    log(f"matched assay, pretrained decoders | {VERSION} | {provenance()} | out={OUT}")
    if args.plumbing:
        plumbing(); return
    sv._resolve_hf_token()
    res_dir = os.path.join(OUT, "results"); os.makedirs(res_dir, exist_ok=True)
    models = args.models or (os.environ.get("MP_MODELS", "").split() or MODELS)
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
