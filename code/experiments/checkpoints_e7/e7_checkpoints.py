"""E7: Pythia-410M public checkpoints under the canonical protocols (NMI revision, 23 Sept 2026).

Fig. 5c currently rests on a May 2026 file reconstructed from terminal output, whose rotation cost at the final
checkpoint (7.2 nats in the middle third) does not match the census value for the same model (0.38 nats), so the
intervention differed. This script remeasures every checkpoint with the two protocols used elsewhere in the paper.

For each public checkpoint (Hub revision stepN) of EleutherAI/pythia-410m:
  census   per-block branch rotation, exactly as the decoder census: a Haar-random orthogonal rotation of the attention
           and MLP outputs of one block at a time before the residual add (dose 1, seed 42 + 100 l + 10 s), next-token
           cross-entropy on 30 batches of 16 consecutive 128-token WikiText-103 validation windows, 200-sample bootstrap
           interval over batches; plumbing checks (dose 0 reproduces the clean loss, dose 1 on block 1 changes it)
  sigma1   the canonical survey profile (survey_sigma1_v2.py, imported unchanged: token in context, natural inputs,
           float32, 8 positions, 4 restarts, 50 iterations) with E7_N_INPUTS inputs (default 30, as E3)
The tokenizer is the released one (identical across checkpoints); only the weights change with the revision.

Built-in checks at step 143000 (the released model): the census profile must reproduce
census_v4/merged_v4/perlayer_eleutherai_pythia-410m.json (clean loss 3.4932 nats), and R_ex0 the E3 30-input value 0.837.

Outputs, written atomically as each (checkpoint, stage) finishes, resume-safe:
  <E7_OUT>/census/census_step<N>.json   <E7_OUT>/sigma1/sigma1_step<N>.json   <E7_OUT>/e7_summary.json
Environment: E7_OUT, E7_STEPS (space-separated), E7_N_INPUTS, E7_STAGES ("census sigma1"), E7_QUICK=1 (2 batches, quick
power iteration; plumbing only), E7_PLUMBING=1 (tiny random GPT-NeoX saved locally, no network; container test).
"""
import os, sys, json, time, gc, shutil, datetime, platform, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import survey_sigma1_v2 as sv          # sets TF32 off and DEVICE; imported unchanged

VERSION = "e7-2026-09-23"
HF_ID = "EleutherAI/pythia-410m"
OUT = os.environ.get("E7_OUT", "results/e7")
STEPS = [int(s) for s in os.environ.get("E7_STEPS", "143000 0 512 2000 8000 32000 64000 100000 1000 4000 16000").split()]
STAGES = os.environ.get("E7_STAGES", "census sigma1").split()
N_INPUTS = int(os.environ.get("E7_N_INPUTS", "30"))
QUICK = os.environ.get("E7_QUICK") == "1"
PLUMBING = os.environ.get("E7_PLUMBING") == "1"
DEVICE = sv.DEVICE
SEED, SEQ_LEN, BS, N_BOOT, DOSE = 42, 128, 16, 200, 1.0
N_BATCHES = 2 if (QUICK or PLUMBING) else 30

# census_v4/merged_v4/perlayer_eleutherai_pythia-410m.json (May 2026 decoder census, same protocol)
CENSUS_REF = dict(baseline=3.493244433403015, delta_L_profile=[
    7.4894, 1.7206, 0.7026, 1.1184, 0.7548, 5.4702, 2.4683, 0.6672, 0.5375, 0.3034, 0.3022, 0.3526,
    0.3417, 0.4874, 0.3044, 0.3829, 0.9024, 0.5537, 0.5459, 0.4888, 0.4224, 0.479, 0.6016, 1.31])
E3_REF_R = 0.8368                     # survey_30inputs/results/sigma1_EleutherAI_pythia-410m.json

REV = {"rev": None}
_orig_fp = sv._from_pretrained


def _fp(cls, hf_id, dtype, **kw):
    """sv._from_pretrained with the current Hub revision added (weights only)."""
    if REV["rev"] is not None:
        kw["revision"] = REV["rev"]
    return _orig_fp(cls, hf_id, dtype, **kw)


sv._from_pretrained = _fp


def log(msg):
    print(msg, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def provenance():
    import transformers
    p = dict(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
             date=datetime.datetime.now().isoformat(timespec="seconds"), device=DEVICE)
    if torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name()
    return p


def thirds(v):
    """Early, middle, late means over blocks 1..L-2 (the census regions) and middle / larger edge."""
    v = np.asarray(v, float); inner = v[1:len(v) - 1]; n = len(inner); t = n // 3
    e, w, l = inner[:t].mean(), inner[t:n - t].mean(), inner[n - t:].mean()
    return dict(early=float(e), middle=float(w), late=float(l), middle_over_larger_edge=float(w / max(e, l)) if max(e, l) > 0 else None)


# ---------------------------------------------------------------- plumbing (container test, no network)
def setup_plumbing():
    global HF_ID
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    d = os.path.join(OUT, "_tiny_neox")
    if not os.path.isdir(d):
        torch.manual_seed(0)
        cfg = GPTNeoXConfig(vocab_size=512, hidden_size=64, num_hidden_layers=6, num_attention_heads=4, intermediate_size=256,
                            max_position_embeddings=256, use_parallel_residual=True, rotary_pct=0.25)
        GPTNeoXForCausalLM(cfg).save_pretrained(d)
        words = ["w%d" % i for i in range(500)]
        vocab = {w: i for i, w in enumerate(words + ["[UNK]"])}
        tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]").save_pretrained(d)
    rng = np.random.RandomState(0)
    fake = [" ".join("w%d" % i for i in rng.randint(0, 500, 400)) for _ in range(300)]
    sv._wikitext_validation_texts = lambda: fake
    sv.CFG[d] = dict(sv.CFG[HF_ID], hf_id=d)
    HF_ID = d
    log(f"PLUMBING: tiny random GPT-NeoX at {d}; synthetic text; revisions ignored")


# ---------------------------------------------------------------- census (branch rotation)
def random_orthogonal(d, seed):
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(H)
    return torch.from_numpy(Q @ np.diag(np.sign(np.diag(R))))


def rot_hook(R, dose):
    def hook(mod, inp, output):
        if isinstance(output, tuple):
            h = output[0]
            return ((1 - dose) * h + dose * torch.einsum("ij,...j->...i", R.to(h.device, h.dtype), h),) + tuple(output[1:])
        return (1 - dose) * output + dose * torch.einsum("ij,...j->...i", R.to(output.device, output.dtype), output)
    return hook


_IDS = {}


def census_ids(tok):
    if "ids" not in _IDS:
        texts = sv._wikitext_validation_texts()
        text = "\n".join(t for t in texts if len(t.strip()) > 50)
        ids = tok.encode(text)                      # as the May census (no BOS for the GPT-NeoX tokenizer)
        seqs = [ids[i:i + SEQ_LEN] for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN)][:N_BATCHES * BS]
        _IDS["ids"] = torch.tensor(seqs, dtype=torch.long)
        _IDS["first_token"] = int(ids[0]); _IDS["n_tokens"] = len(ids)
        log(f"  census data: {len(ids)} tokens -> {len(seqs)} windows of {SEQ_LEN}; first token id {ids[0]} (bos {tok.bos_token_id})")
    return _IDS["ids"]


@torch.no_grad()
def run_census(step):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(HF_ID, token=os.environ.get("HF_TOKEN"))
    model = _fp(AutoModelForCausalLM, HF_ID, torch.float32, token=os.environ.get("HF_TOKEN"), attn_implementation="eager").to(DEVICE).eval()
    layers = list(model.gpt_neox.layers); L = len(layers); d = model.config.hidden_size
    subs = [(l.attention, l.mlp) for l in layers]
    R = {(li, si): random_orthogonal(d, SEED + li * 100 + si * 10) for li in range(L) for si in range(2)}
    ids = census_ids(tok)

    def evaluate(targets, dose=DOSE):
        hs = [mod.register_forward_hook(rot_hook(R[(li, si)], dose)) for li in targets for si, mod in enumerate(subs[li])]
        try:
            out = []
            for b in range(N_BATCHES):
                x = ids[b * BS:(b + 1) * BS].to(DEVICE)
                out.append(model(input_ids=x, labels=x, use_cache=False).loss.item())
            return np.array(out)
        finally:
            for h in hs:
                h.remove()

    base = evaluate([]); baseline = float(base.mean())
    log(f"  clean loss {baseline:.4f} nats")
    p0 = float(abs(evaluate([1], 0.0).mean() - baseline)); p1 = float(evaluate([1], 1.0).mean() - baseline)
    log(f"  plumbing: dose 0 on block 1 |dL| = {p0:.1e}; dose 1 dL = {p1:+.4f}")
    assert p0 < 1e-5 * max(1.0, baseline), "dose-0 hook changed the loss"
    dl, ci = [], []
    for li in range(L):
        arr = evaluate([li]); delta = float(arr.mean() - baseline)
        rng = np.random.RandomState(SEED + li)
        boots = [arr[i].mean() - base[i].mean() for i in (rng.randint(0, N_BATCHES, N_BATCHES) for _ in range(N_BOOT))]
        dl.append(delta); ci.append([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
        log(f"    L{li:2d}: dL = {delta:+.4f} [{ci[-1][0]:+.4f}, {ci[-1][1]:+.4f}]")
    res = dict(experiment="E7 census", version=VERSION, hf_id=HF_ID, revision=REV["rev"], step=step, n_layers=L, d_model=d,
               protocol="branch rotation (attention and MLP outputs), dose 1, as the decoder census", baseline=baseline,
               delta_L_profile=dl, delta_L_ci=ci, regions=thirds(dl), n_batches=N_BATCHES, batch_size=BS, seq_len=SEQ_LEN,
               seed=SEED, n_bootstrap=N_BOOT, plumbing_dose0_abs_dL=p0, plumbing_dose1_dL_block1=p1,
               data=dict(n_tokens=_IDS["n_tokens"], first_token=_IDS["first_token"]), quick=QUICK, plumbing=PLUMBING,
               time_s=time.time() - t0, provenance=provenance())
    if step == 143000 and not (QUICK or PLUMBING):
        ref = np.array(CENSUS_REF["delta_L_profile"])
        res["check_vs_census"] = dict(baseline_diff=baseline - CENSUS_REF["baseline"], max_abs_diff=float(np.abs(np.array(dl) - ref).max()))
        log(f"  CHECK vs census file: clean loss diff {res['check_vs_census']['baseline_diff']:+.4f}; max |dL diff| {res['check_vs_census']['max_abs_diff']:.4f}")
    del model; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- sigma1 (canonical survey)
def run_sigma1(step):
    args = sv.parse_args([])
    if not (QUICK or PLUMBING):
        args.n_inputs = N_INPUTS
    else:
        args.n_inputs, args.n_positions, args.restarts, args.iters = 2, 2, 1, 6
    b = sv.load_bundle(dict(sv.CFG[HF_ID]), "float32")
    try:
        inputs = sv.prepare_inputs(b, "natural", args.n_inputs)
        res = sv.profile_model(b, "incontext", "natural", False, args, inputs=inputs)
    finally:
        del b; gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    res.update(experiment="E7 sigma1", e7_version=VERSION, revision=REV["rev"], step=step, quick=QUICK, plumbing=PLUMBING)
    if step == 143000 and not (QUICK or PLUMBING) and args.n_inputs == 30:
        res["check_vs_e3"] = dict(R_diff=res["R_ex0_thirds"] - E3_REF_R)
        log(f"  CHECK vs E3: R_ex0 {res['R_ex0_thirds']:.4f} against {E3_REF_R} (diff {res['R_ex0_thirds'] - E3_REF_R:+.4f})")
    return res


def purge(hf_id):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = os.path.join(HF_HUB_CACHE, "models--" + hf_id.replace("/", "--"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        log(f"  (cache purge failed: {e})")


def summary():
    rows = []
    for s in sorted(set(STEPS)):
        c = os.path.join(OUT, "census", f"census_step{s}.json"); g = os.path.join(OUT, "sigma1", f"sigma1_step{s}.json")
        r = dict(step=s)
        if os.path.exists(c):
            d = json.load(open(c)); r.update(clean_loss=d["baseline"], **{f"dL_{k}": v for k, v in d["regions"].items()})
        if os.path.exists(g):
            d = json.load(open(g)); r.update(R_ex0=d["R_ex0_thirds"], R_ci=d["R_ex0_thirds_ci95"], n_inputs=d["n_inputs"])
        rows.append(r)
    write_json(os.path.join(OUT, "e7_summary.json"), dict(version=VERSION, rows=rows))
    log(f"\n{'step':>7s} {'clean CE':>9s} {'dL early':>9s} {'dL mid':>8s} {'dL late':>8s} {'mid/edge':>9s} {'R_ex0':>7s}  CI")
    f = lambda r, k, w, p: (f"{r[k]:{w}.{p}f}" if r.get(k) is not None else " " * (w - 1) + "-")
    for r in rows:
        log(f"{r['step']:7d} {f(r, 'clean_loss', 9, 3)} {f(r, 'dL_early', 9, 3)} {f(r, 'dL_middle', 8, 3)} {f(r, 'dL_late', 8, 3)} "
            f"{f(r, 'dL_middle_over_larger_edge', 9, 3)} {f(r, 'R_ex0', 7, 3)}  {r.get('R_ci', '')}")


def main():
    sv._resolve_hf_token()
    if PLUMBING:
        setup_plumbing()
    log(f"{VERSION}; {sv.VERSION}; device {DEVICE}; steps {STEPS}; stages {STAGES}; inputs {N_INPUTS}; batches {N_BATCHES}; out {OUT}")
    if DEVICE == "cuda":
        log(f"GPU {torch.cuda.get_device_name()}; TF32 matmul {torch.backends.cuda.matmul.allow_tf32}, conv {torch.backends.cudnn.allow_tf32}")
    t_all = time.time()
    for i, step in enumerate(STEPS):
        REV["rev"] = None if PLUMBING else f"step{step}"
        log(f"\n{'=' * 70}\n[{i + 1}/{len(STEPS)}] {HF_ID} step {step}\n{'=' * 70}")
        for stage in STAGES:
            path = os.path.join(OUT, stage, f"{stage}_step{step}.json")
            if os.path.exists(path):
                log(f"  skip {stage} (exists)"); continue
            try:
                t0 = time.time()
                res = run_census(step) if stage == "census" else run_sigma1(step)
                write_json(path, res)
                log(f"  saved {path} ({time.time() - t0:.0f} s)")
            except Exception as e:
                log(f"  FAILED {stage} step {step}: {type(e).__name__}: {e}"); traceback.print_exc()
        if not PLUMBING:
            purge(HF_ID)
    summary()
    log(f"\nTotal {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
