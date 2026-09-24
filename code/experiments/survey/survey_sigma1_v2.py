"""
survey_sigma1_v2.py  --  one profiler, one protocol, every model
================================================================

Measures the quantity the paper defines, for every model in the survey and census
panels, with one script:

    sigma_1 of the token-local Jacobian  J_l,t = d h_{l+1,t} / d h_{l,t}

i.e. perturb the hidden state at ONE position t at the input of block l, with every
other position held fixed at its natural-forward value, and read the change at the
same position t at the output of block l. sigma_1 is the leading singular value of
that d x d map, found by power iteration on J^T J with exact automatic
differentiation (torch.func.jvp for J v, reverse-mode autograd for J^T u), 50
iterations, four random restarts per position, max over restarts.

Inputs are natural: WikiText-103 validation text for language models (5 sequences of
128 tokens, 8 probe positions each), CIFAR-100 test images for ConvNeXt, ViT, DINOv2
and MLP-Mixer, LibriSpeech test-clean for Whisper and wav2vec2. Float32, TF32 off.

The profile is over blocks (block 0 = first transformer block / first ConvNeXt block).
R_ex0 = mean sigma_1 over the middle third of blocks 1..L-1 divided by the mean over
the two edge thirds; late uptick = mean(last third) / mean(middle third); a model is
hourglass-positive if R_ex0 < 0.80 and late uptick > 1.0 (the paper's rule). The May
census vertex/median R_ex0, spectral contrast C and quadratic curvature a are written
too, so the scrambling census 2x2 can use this geometry.

Why this exists: the April survey values came from five or more profiling scripts
whose sigma_1 values are not comparable; the 29 April reprofile pushed each token
through its layer alone (sequence length 1) on random ids and iterated on J^T rather
than J^T J; the May census used finite-difference sigma_1 of the whole-sequence layer
Jacobian on random ids. Three different quantities, three different classifications.
This script replaces all of them and, with --protocol-panel, measures the alternative
protocols on eight models so the dependence can be shown (Extended Data figure).

Modes
  --dtype float16        same inputs and settings, model in float16 (Fig. 1d); uses the
                         finite-difference protocol of the April note (relative step --fd-eps,
                         default 1e-3) because forward-mode AD is unreliable in half precision,
                         and records non-finite outputs instead of stopping
  --branch               sigma_1 of J - I, the branch Jacobian (Extended Data Fig. 5);
                         decoders, Mamba and ConvNeXt only (pre-norm residual blocks)
  --context incontext    canonical (default)
  --context isolated     token pushed through the block alone, kwargs sliced to t
  --context fullseq      sigma_1 of the whole-sequence layer Jacobian
  --inputs natural       canonical (default)      --inputs random   random ids/pixels
  --protocol-panel       the 8-model x 4-protocol sub-run
  --estimator rho        power iteration on J instead of J^T J: the spectral radius, which is
                         what the earlier finite-difference variant measured (2-9x smaller than
                         sigma_1 on these layers, and it flattens profiles); tag suffix -rho
  --validate             exact Jacobians at 5 blocks x 3 positions x 2 inputs: exact sigma_1 and rho
                         against both estimators and 64 random probes (Fig. 1b,c); sequence models
  --panel april|union    the 33 April survey models, or April + May census extras (45)

Usage (shell)
  python survey_sigma1_v2.py --panel union --results-dir results/survey_sigma1_v2
  python survey_sigma1_v2.py --models gpt2 --smoke            # reference check, prints Fig. 1a anchors
  python survey_sigma1_v2.py --protocol-panel
  python survey_sigma1_v2.py --dtype float16 --models EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b
  python survey_sigma1_v2.py --branch --models gpt2 microsoft/phi-2 EleutherAI/pythia-410m ...

Usage (Colab cell): set env vars SURVEY_PANEL, SURVEY_MODELS (space-separated),
SURVEY_RESULTS_DIR, SURVEY_DTYPE, SURVEY_BRANCH=1, SURVEY_CONTEXT, SURVEY_INPUTS,
SURVEY_PROTOCOL_PANEL=1, SURVEY_SMOKE=1, SURVEY_QUICK=1, SURVEY_NO_SKIP=1, SURVEY_ZIP=1,
SURVEY_FD_EPS, SURVEY_PURGE_CACHE=1 (delete each model's Hub cache after use; Colab's disk fills
at about 100 GB of weights),
then exec(open("survey_sigma1_v2.py").read()). The notebook's -f argument is ignored.

Resume-safe: an existing JSON for (model, protocol) is skipped unless --no-skip.
Output: <results-dir>/<protocol>/sigma1_<model>.json, plus survey_summary.json/.csv/.md
per protocol directory, plus survey_results.zip with --zip.
"""

import os, sys, json, time, gc, math, argparse, shutil, datetime, traceback
import numpy as np
import torch

# ---------------------------------------------------------------- precision
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SEQ_LEN = 128
N_INPUTS = 5
N_POSITIONS = 8
N_RESTARTS = 4
PI_ITERS = 50
PI_TOL = 1e-6
N_BOOT = 500
AUDIO_SECONDS = 10.0
VERSION = "survey-v2.3-2026-09-18"


def log(msg):
    print(msg, flush=True)


def _resolve_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
    except Exception:
        tok = None
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:
            tok = None
    if tok:
        os.environ["HF_TOKEN"] = tok
    return tok


# ---------------------------------------------------------------- registry
# (hf_id, arch, label, family, params_M, april_panel)
REGISTRY = [
    ("EleutherAI/pythia-70m",   "pythia", "Pythia-70M",   "AR decoder", 70,    True),
    ("EleutherAI/pythia-160m",  "pythia", "Pythia-160M",  "AR decoder", 160,   True),
    ("EleutherAI/pythia-410m",  "pythia", "Pythia-410M",  "AR decoder", 410,   True),
    ("EleutherAI/pythia-1b",    "pythia", "Pythia-1B",    "AR decoder", 1000,  True),
    ("EleutherAI/pythia-1.4b",  "pythia", "Pythia-1.4B",  "AR decoder", 1400,  True),
    ("EleutherAI/pythia-2.8b",  "pythia", "Pythia-2.8B",  "AR decoder", 2800,  True),
    ("EleutherAI/pythia-6.9b",  "pythia", "Pythia-6.9B",  "AR decoder", 6900,  True),
    ("EleutherAI/pythia-12b",   "pythia", "Pythia-12B",   "AR decoder", 12000, True),
    ("gpt2",                    "gpt2",   "GPT-2 124M",   "AR decoder", 124,   True),
    ("gpt2-large",              "gpt2",   "GPT-2 Large",  "AR decoder", 774,   True),
    ("gpt2-xl",                 "gpt2",   "GPT-2 XL",     "AR decoder", 1558,  True),
    ("microsoft/phi-2",         "phi",    "Phi-2",        "AR decoder", 2700,  True),
    ("mistralai/Mistral-7B-v0.1", "mistral", "Mistral-7B", "AR decoder", 7200, True),
    ("Qwen/Qwen2.5-0.5B",       "qwen",   "Qwen2.5-0.5B", "AR decoder", 494,   True),
    ("Qwen/Qwen2.5-1.5B",       "qwen",   "Qwen2.5-1.5B", "AR decoder", 1540,  True),
    ("Qwen/Qwen2.5-3B",         "qwen",   "Qwen2.5-3B",   "AR decoder", 3090,  True),
    ("Qwen/Qwen2.5-7B",         "qwen",   "Qwen2.5-7B",   "AR decoder", 7610,  True),
    ("Qwen/Qwen2.5-14B",        "qwen",   "Qwen2.5-14B",  "AR decoder", 14700, True),
    ("meta-llama/Llama-3.2-1B", "llama",  "Llama-3.2-1B", "AR decoder", 1240,  True),
    ("meta-llama/Llama-3.2-3B", "llama",  "Llama-3.2-3B", "AR decoder", 3210,  True),
    ("google/gemma-2-2b",       "gemma2", "Gemma-2-2B",   "AR decoder", 2610,  True),
    ("facebook/convnext-tiny-224",  "convnext", "ConvNeXt-tiny",  "ConvNet (vision)", 28,  True),
    ("facebook/convnext-small-224", "convnext", "ConvNeXt-small", "ConvNet (vision)", 50,  True),
    ("facebook/convnext-base-224",  "convnext", "ConvNeXt-base",  "ConvNet (vision)", 89,  True),
    ("facebook/convnext-large-224", "convnext", "ConvNeXt-large", "ConvNet (vision)", 198, True),
    ("state-spaces/mamba-130m-hf",  "mamba", "Mamba-130M", "SSM", 130,  True),
    ("state-spaces/mamba-1.4b-hf",  "mamba", "Mamba-1.4B", "SSM", 1400, True),
    ("state-spaces/mamba-2.8b-hf",  "mamba", "Mamba-2.8B", "SSM", 2800, True),
    ("google-bert/bert-base-uncased", "bert",    "BERT-base",    "Masked encoder", 110, True),
    ("FacebookAI/roberta-base",      "roberta", "RoBERTa-base", "Masked encoder", 125, True),
    ("google/vit-base-patch16-224",  "vit",     "ViT-base",     "Vision supervised", 86, True),
    ("facebook/dinov2-base",         "dinov2",  "DINOv2-base",  "Vision SSL", 86, True),
    ("openai/whisper-small",         "whisper", "Whisper-small", "Audio encoder", 88, True),
    # May census extras
    ("google-t5/t5-small",   "t5", "T5-small (encoder)",   "Encoder-decoder", 60,  False),
    ("google-t5/t5-base",    "t5", "T5-base (encoder)",    "Encoder-decoder", 220, False),
    ("google-t5/t5-large",   "t5", "T5-large (encoder)",   "Encoder-decoder", 770, False),
    ("google/flan-t5-large", "t5", "Flan-T5-large (encoder)", "Encoder-decoder", 780, False),
    ("facebook/wav2vec2-base-960h", "wav2vec2", "wav2vec2-base", "Audio encoder", 95, False),
    ("google-bert/bert-large-uncased", "bert",    "BERT-large",    "Masked encoder", 340, False),
    ("FacebookAI/roberta-large",      "roberta", "RoBERTa-large", "Masked encoder", 355, False),
    ("state-spaces/mamba-370m-hf",    "mamba",   "Mamba-370M",    "SSM", 370, False),
    ("Qwen/Qwen2-0.5B",               "qwen",    "Qwen2-0.5B",    "AR decoder", 494, False),
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral", "Mistral-7B-Instruct", "AR decoder", 7200, False),
    ("timm/mixer_b16_224.goog_in21k_ft_in1k", "mlp_mixer", "MLP-Mixer-B", "MLP-Mixer (vision)", 60,  False),
    ("timm/mixer_l16_224.goog_in21k_ft_in1k", "mlp_mixer", "MLP-Mixer-L", "MLP-Mixer (vision)", 208, False),
]
CFG = {r[0]: dict(hf_id=r[0], arch=r[1], label=r[2], family=r[3], params_M=r[4], april=r[5]) for r in REGISTRY}

PROTOCOL_PANEL_MODELS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-6.9b", "gpt2", "Qwen/Qwen2.5-7B",
                         "meta-llama/Llama-3.2-1B", "state-spaces/mamba-130m-hf",
                         "facebook/convnext-tiny-224", "facebook/convnext-base-224"]
PROTOCOL_PANEL = [("incontext", "natural"), ("incontext", "random"), ("isolated", "random"), ("fullseq", "random")]

DECODERS = ("gpt2", "pythia", "llama", "qwen", "mistral", "gemma2", "phi")
BRANCH_OK = DECODERS + ("mamba", "convnext")
TEXT_ARCHES = DECODERS + ("mamba", "bert", "roberta", "t5")
IMAGE_ARCHES = ("convnext", "vit", "dinov2", "mlp_mixer")
AUDIO_ARCHES = ("whisper", "wav2vec2")

# v35 Fig. 1a, exact full-Jacobian sigma_1 on GPT-2 124M (text inputs), for the smoke check
GPT2_REFERENCE = {0: 9.8, 6: 1.1, 11: 2.5}


# ---------------------------------------------------------------- arguments
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--panel", default=None, choices=["april", "union"])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--branch", action="store_true")
    ap.add_argument("--context", default=None, choices=["incontext", "isolated", "fullseq"])
    ap.add_argument("--inputs", default=None, choices=["natural", "random"])
    ap.add_argument("--protocol-panel", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--n-inputs", type=int, default=None)
    ap.add_argument("--n-positions", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--fd-eps", type=float, default=None, help="relative finite-difference step for reduced-precision runs")
    ap.add_argument("--purge-cache", action="store_true", help="delete each model's Hub cache after profiling it")
    ap.add_argument("--force-fd", action="store_true", help="use the finite-difference protocol in float32 too (matched control for --dtype float16)")
    ap.add_argument("--estimator", default=None, choices=["sigma1", "rho"],
                    help="sigma1 (power iteration on J^T J, default) or rho (power iteration on J: the spectral radius the earlier finite-difference variant estimated)")
    ap.add_argument("--validate", action="store_true",
                    help="materialise the exact token-local Jacobian at five blocks x three positions x two inputs and compare exact sigma1/rho with the estimators and a 64-direction random-probe baseline (Fig. 1b,c)")
    args, unknown = ap.parse_known_args(argv)
    env = os.environ
    if args.models is None and env.get("SURVEY_MODELS"):
        args.models = env["SURVEY_MODELS"].split()
    args.panel = args.panel or env.get("SURVEY_PANEL") or "union"
    args.results_dir = args.results_dir or env.get("SURVEY_RESULTS_DIR") or "results/survey_sigma1_v2"
    args.dtype = args.dtype or env.get("SURVEY_DTYPE") or "float32"
    args.branch = args.branch or env.get("SURVEY_BRANCH") == "1"
    args.context = args.context or env.get("SURVEY_CONTEXT") or "incontext"
    args.inputs = args.inputs or env.get("SURVEY_INPUTS") or "natural"
    args.protocol_panel = args.protocol_panel or env.get("SURVEY_PROTOCOL_PANEL") == "1"
    args.smoke = args.smoke or env.get("SURVEY_SMOKE") == "1"
    args.quick = args.quick or env.get("SURVEY_QUICK") == "1"
    args.no_skip = args.no_skip or env.get("SURVEY_NO_SKIP") == "1"
    args.zip = args.zip or env.get("SURVEY_ZIP") == "1"
    args.n_inputs = args.n_inputs or int(env.get("SURVEY_N_INPUTS", N_INPUTS))
    args.n_positions = args.n_positions or int(env.get("SURVEY_N_POSITIONS", N_POSITIONS))
    args.restarts = args.restarts or int(env.get("SURVEY_RESTARTS", N_RESTARTS))
    args.iters = args.iters or int(env.get("SURVEY_ITERS", PI_ITERS))
    args.fd_eps = args.fd_eps or float(env.get("SURVEY_FD_EPS", 1e-3))
    args.purge_cache = args.purge_cache or env.get("SURVEY_PURGE_CACHE") == "1"
    args.force_fd = args.force_fd or env.get("SURVEY_FORCE_FD") == "1"
    args.estimator = args.estimator or env.get("SURVEY_ESTIMATOR") or "sigma1"
    args.validate = args.validate or env.get("SURVEY_VALIDATE") == "1"
    if args.quick:
        args.n_inputs, args.n_positions, args.restarts, args.iters = 1, 2, 1, 6
    return args


def protocol_tag(context, inputs, branch, dtype, fd=False, estimator="sigma1"):
    return (f"{context}-{inputs}-{'branch' if branch else 'block'}-{dtype}" + ("-fd" if fd else "")
            + ("-rho" if estimator == "rho" else ""))


def safe_name(hf_id):
    return hf_id.replace("/", "_").replace(".", "_")


# ---------------------------------------------------------------- data
def _wikitext_validation_texts():
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
                            repo_type="dataset", token=token)
        return pd.read_parquet(p)["text"].tolist()
    except Exception as e:
        log(f"  parquet download failed ({e}); falling back to datasets")
        from datasets import load_dataset
        return list(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"])


_TEXT_CACHE = {}


def natural_text_ids(tokenizer, n_seqs, seq_len=SEQ_LEN):
    """n_seqs sequences of seq_len tokens, spread evenly through WikiText-103 validation."""
    if "text" not in _TEXT_CACHE:
        texts = _wikitext_validation_texts()
        _TEXT_CACHE["text"] = "\n".join(t for t in texts if len(t.strip()) > 50)
    text = _TEXT_CACHE["text"]
    # tokenise a slice around each of n_seqs evenly spaced character offsets (cheap)
    n_chars = len(text)
    offsets = np.linspace(0.05, 0.95, n_seqs) * n_chars
    seqs = []
    for off in offsets:
        chunk = text[int(off): int(off) + 20 * seq_len]
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        if len(ids) < seq_len:
            ids = tokenizer.encode(text[int(off): int(off) + 200 * seq_len], add_special_tokens=False)
        seqs.append(ids[:seq_len])
    return torch.tensor(seqs, dtype=torch.long)


def random_text_ids(vocab_size, n_seqs, seq_len=SEQ_LEN):
    g = torch.Generator().manual_seed(SEED)
    return torch.randint(0, vocab_size, (n_seqs, seq_len), generator=g)


def _cifar100_test_images(n):
    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100(root="./data_cifar100", train=False, download=True)
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        return [ds[int(i)][0].convert("RGB") for i in idx]
    except Exception as e:
        log(f"  torchvision CIFAR-100 failed ({e}); falling back to datasets")
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar100", split="test")
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        return [ds[int(i)]["img"].convert("RGB") for i in idx]


def _librispeech_waveforms(n, seconds=AUDIO_SECONDS):
    """n LibriSpeech test-clean utterances >= 4 s, truncated to `seconds`, 16 kHz float tensors."""
    out = []
    try:
        import torchaudio
        ds = torchaudio.datasets.LIBRISPEECH("./data_librispeech", url="test-clean", download=True)
        idx = np.linspace(0, len(ds) - 1, 4 * n).astype(int)
        for i in idx:
            wav, sr = ds[int(i)][0], ds[int(i)][1]
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav[0]
            if wav.numel() >= 4 * 16000:
                out.append(wav[: int(seconds * 16000)].float())
            if len(out) == n:
                break
    except Exception as e:
        log(f"  torchaudio LibriSpeech failed ({e}); falling back to datasets streaming")
        from datasets import load_dataset
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
        for ex in ds:
            wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
            if ex["audio"]["sampling_rate"] != 16000:
                continue
            if wav.numel() >= 4 * 16000:
                out.append(wav[: int(seconds * 16000)])
            if len(out) == n:
                break
    return out


# ---------------------------------------------------------------- loading
def _from_pretrained(cls, hf_id, dtype, **kw):
    """dtype= on transformers >= 4.56 else torch_dtype=; drops attn_implementation if refused."""
    for dkey in ("dtype", "torch_dtype"):
        try:
            return cls.from_pretrained(hf_id, **{dkey: dtype}, **kw)
        except TypeError as e:
            if "attn_implementation" in str(e) and "attn_implementation" in kw:
                kw = {k: v for k, v in kw.items() if k != "attn_implementation"}
                return cls.from_pretrained(hf_id, **{dkey: dtype}, **kw)
            if dkey == "torch_dtype":
                raise
        except ValueError as e:
            if "attn_implementation" in kw and ("eager" in str(e) or "attn_implementation" in str(e)):
                kw = {k: v for k, v in kw.items() if k != "attn_implementation"}
                return cls.from_pretrained(hf_id, **{dkey: dtype}, **kw)
            raise


LAYER_PATHS = {
    "gpt2": ["transformer.h", "h"],
    "llama": ["model.layers", "layers"], "gemma2": ["model.layers", "layers"], "phi": ["model.layers", "layers"],
    "qwen": ["model.layers", "layers"], "mistral": ["model.layers", "layers"],
    "pythia": ["gpt_neox.layers", "layers"],
    "mamba": ["backbone.layers", "layers"],
    "bert": ["encoder.layer", "encoder.layers", "layers", "layer"], "roberta": ["encoder.layer", "encoder.layers", "layers", "layer"],
    "vit": ["encoder.layer", "encoder.layers", "layers", "layer"], "dinov2": ["encoder.layer", "encoder.layers", "layers", "layer"],
    "t5": ["encoder.block", "block", "encoder.layers"],
    "whisper": ["layers", "encoder.layers"],
    "wav2vec2": ["encoder.layers", "layers"],
    "mlp_mixer": ["blocks"],
}


def _get_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def get_layers(model, arch):
    """The block list of a model, tolerant of the attribute renames across transformers versions."""
    if arch == "convnext":
        enc = model.convnext.encoder if hasattr(model, "convnext") else model.encoder
        return [blk for stage in enc.stages for blk in stage.layers]
    for path in LAYER_PATHS[arch]:
        try:
            layers = _get_path(model, path)
            if isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential)) and len(layers) > 0:
                return list(layers)
        except AttributeError:
            continue
    # generic fallback: the longest ModuleList of block-like modules
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)) and len(mod) > 1 and \
                any(k in type(mod[0]).__name__ for k in ("Layer", "Block", "Decoder", "Encoder")):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise ValueError(f"cannot find the block list for {arch}")
    log(f"  (block list found by search at '{best[0]}')")
    return list(best[1])


def load_bundle(cfg, dtype_name):
    """Returns dict(model, layers, arch, kind, forward, tokenizer/processor, vocab_size, cls_offset)."""
    arch, hf_id = cfg["arch"], cfg["hf_id"]
    token = os.environ.get("HF_TOKEN")
    dtype = getattr(torch, dtype_name)
    log(f"  Loading {hf_id} ({arch}, {dtype_name})...")
    b = dict(arch=arch, hf_id=hf_id, cfg=cfg, dtype=dtype, cls_offset=0)
    if arch in DECODERS or arch == "mamba":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = dict(token=token)
        if arch != "mamba":
            kw["attn_implementation"] = "eager"
        model = _from_pretrained(AutoModelForCausalLM, hf_id, dtype, **kw)
        tok = AutoTokenizer.from_pretrained(hf_id, token=token)
        b.update(kind="text", tokenizer=tok, vocab_size=int(model.config.vocab_size),
                 forward=lambda m, inp: m(input_ids=inp["input_ids"], use_cache=False))
    elif arch in ("bert", "roberta"):
        from transformers import AutoModel, AutoTokenizer
        model = _from_pretrained(AutoModel, hf_id, dtype, token=token, attn_implementation="eager")
        tok = AutoTokenizer.from_pretrained(hf_id, token=token)
        b.update(kind="text", tokenizer=tok, vocab_size=int(model.config.vocab_size),
                 forward=lambda m, inp: m(input_ids=inp["input_ids"]))
    elif arch == "t5":
        from transformers import T5EncoderModel, AutoTokenizer
        model = _from_pretrained(T5EncoderModel, hf_id, dtype, token=token)
        tok = AutoTokenizer.from_pretrained(hf_id, token=token)
        b.update(kind="text", tokenizer=tok, vocab_size=int(model.config.vocab_size),
                 forward=lambda m, inp: m(input_ids=inp["input_ids"]))
    elif arch in ("vit", "dinov2"):
        from transformers import AutoModel, AutoImageProcessor
        model = _from_pretrained(AutoModel, hf_id, dtype, token=token, attn_implementation="eager")
        proc = AutoImageProcessor.from_pretrained(hf_id, token=token)
        b.update(kind="image", processor=proc, cls_offset=1,
                 forward=lambda m, inp: m(pixel_values=inp["pixel_values"]))
        if arch == "dinov2":
            b["cls_offset"] = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    elif arch == "convnext":
        from transformers import ConvNextForImageClassification, AutoImageProcessor
        model = _from_pretrained(ConvNextForImageClassification, hf_id, dtype, token=token)
        proc = AutoImageProcessor.from_pretrained(hf_id, token=token)
        b.update(kind="image", processor=proc, forward=lambda m, inp: m(pixel_values=inp["pixel_values"]))
    elif arch == "mlp_mixer":
        import timm
        name = hf_id.replace("timm/", "")
        model = timm.create_model(name, pretrained=True)
        if dtype != torch.float32:
            model = model.to(dtype)
        data_cfg = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**data_cfg)
        b.update(kind="image", timm_transform=transform, timm_input_size=data_cfg.get("input_size", (3, 224, 224)),
                 forward=lambda m, inp: m(inp["pixel_values"]))
    elif arch == "whisper":
        from transformers import WhisperModel, WhisperFeatureExtractor
        full = _from_pretrained(WhisperModel, hf_id, dtype, token=token, attn_implementation="eager")
        model = full.encoder
        fe = WhisperFeatureExtractor.from_pretrained(hf_id, token=token)
        b.update(kind="audio", feature_extractor=fe, forward=lambda m, inp: m(input_features=inp["input_features"]))
    elif arch == "wav2vec2":
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        model = _from_pretrained(Wav2Vec2Model, hf_id, dtype, token=token, attn_implementation="eager")
        fe = Wav2Vec2FeatureExtractor.from_pretrained(hf_id, token=token)
        b.update(kind="audio", feature_extractor=fe, forward=lambda m, inp: m(input_values=inp["input_values"]))
    else:
        raise ValueError(arch)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    b["model"] = model
    b["layers"] = get_layers(model, arch)
    log(f"  {len(b['layers'])} blocks; {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
    return b


# ---------------------------------------------------------------- inputs
def prepare_inputs(b, inputs_kind, n_inputs):
    """List of input dicts for the model forward, each with batch 1, plus 'valid_T' hints."""
    arch, kind = b["arch"], b["kind"]
    out = []
    if kind == "text":
        if inputs_kind == "natural":
            ids = natural_text_ids(b["tokenizer"], n_inputs)
        else:
            ids = random_text_ids(b["vocab_size"], n_inputs)
        for i in range(ids.shape[0]):
            out.append({"input_ids": ids[i:i + 1].to(DEVICE)})
        desc = f"{'WikiText-103 validation' if inputs_kind == 'natural' else 'uniform random ids'}, {n_inputs} x {SEQ_LEN} tokens"
    elif kind == "image":
        if inputs_kind == "natural":
            imgs = _cifar100_test_images(n_inputs)
            if arch == "mlp_mixer":
                px = torch.stack([b["timm_transform"](im) for im in imgs])
            else:
                px = b["processor"](images=imgs, return_tensors="pt")["pixel_values"]
            desc = f"CIFAR-100 test images, {n_inputs}, {tuple(px.shape[1:])}"
        else:
            g = torch.Generator().manual_seed(SEED)
            shape = (3, 224, 224)
            if arch == "mlp_mixer":
                shape = tuple(b.get("timm_input_size", shape))
            elif arch in ("vit", "dinov2", "convnext"):
                proc = b["processor"]
                cs = getattr(proc, "crop_size", None) or getattr(proc, "size", None) or {}
                h = cs.get("height", cs.get("shortest_edge", 224)) if isinstance(cs, dict) else 224
                shape = (3, int(h), int(h))
            px = torch.randn((n_inputs,) + shape, generator=g)
            desc = f"random pixels N(0,1) in normalised space, {n_inputs}, {shape}"
        for i in range(px.shape[0]):
            out.append({"pixel_values": px[i:i + 1].to(DEVICE, dtype=b["dtype"])})
    elif kind == "audio":
        fe = b["feature_extractor"]
        if inputs_kind == "natural":
            wavs = _librispeech_waveforms(n_inputs)
            desc = f"LibriSpeech test-clean, {n_inputs} utterances, first {AUDIO_SECONDS:.0f} s"
        else:
            g = torch.Generator().manual_seed(SEED)
            wavs = [0.05 * torch.randn(int(AUDIO_SECONDS * 16000), generator=g) for _ in range(n_inputs)]
            desc = f"white noise, {n_inputs} x {AUDIO_SECONDS:.0f} s"
        for w in wavs:
            if arch == "whisper":
                feats = fe(w.numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
                valid_T = int(math.ceil(w.numel() / 16000 / 30.0 * 1500))
                out.append({"input_features": feats.to(DEVICE, dtype=b["dtype"]), "valid_T": valid_T})
            else:
                vals = fe(w.numpy(), sampling_rate=16000, return_tensors="pt")["input_values"]
                out.append({"input_values": vals.to(DEVICE, dtype=b["dtype"])})
    else:
        raise ValueError(kind)
    b["input_desc"] = desc
    log(f"  Inputs: {desc}")
    return out


# ---------------------------------------------------------------- capture
def _strip_kwargs(kw):
    out = {}
    for k, v in kw.items():
        if "past" in k or k in ("use_cache", "cache_params", "layer_past"):
            continue
        out[k] = v
    return out


def capture_layer_inputs(b, inp):
    """One forward with pre-hooks; returns per-layer (hidden[1,...], extra_args, kwargs)."""
    layers = b["layers"]
    captured = [None] * len(layers)
    hooks = []

    def make_hook(i):
        def fn(mod, args, kwargs):
            if len(args) == 0:
                kwargs = dict(kwargs)
                h = kwargs.pop("hidden_states")
                rest = ()
            else:
                h, rest = args[0], tuple(args[1:])
            captured[i] = (h.detach(), rest, _strip_kwargs(kwargs))
        return fn

    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(make_hook(i), with_kwargs=True))
    try:
        with torch.no_grad():
            b["forward"](b["model"], inp)
    finally:
        for h in hooks:
            h.remove()
    missing = [i for i, c in enumerate(captured) if c is None]
    if missing:
        raise RuntimeError(f"pre-hook never fired for blocks {missing}")
    return captured


def make_layer_call(layer, rest, kw):
    def call(X):
        out = layer(X, *rest, **kw)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out
    return call


# ---------------------------------------------------------------- token layout
class Layout:
    """Where the tokens are in a block's hidden tensor: (1,T,d) sequences or (1,C,H,W) maps."""

    def __init__(self, hidden):
        self.dim = hidden.dim()
        if self.dim == 3:
            _, self.T, self.d = hidden.shape
        elif self.dim == 4:
            _, self.d, self.H, self.W = hidden.shape
            self.T = self.H * self.W
        else:
            raise ValueError(f"unsupported hidden shape {tuple(hidden.shape)}")

    def hw(self, p):
        return divmod(int(p), self.W)

    def get(self, hidden, positions):
        if self.dim == 3:
            return hidden[0, positions, :]                      # (B, d)
        hs, ws = zip(*[self.hw(p) for p in positions])
        return hidden[0, :, list(hs), list(ws)].T               # (B, d)

    def mask(self, positions, device):
        B = len(positions)
        if self.dim == 3:
            M = torch.zeros(B, self.T, 1, device=device)
            M[torch.arange(B), torch.as_tensor(positions)] = 1.0
        else:
            M = torch.zeros(B, 1, self.H, self.W, device=device)
            for i, p in enumerate(positions):
                h, w = self.hw(p)
                M[i, 0, h, w] = 1.0
        return M

    def broadcast(self, xb):
        return xb[:, None, :] if self.dim == 3 else xb[:, :, None, None]

    def gather(self, Y, positions):
        B = len(positions)
        if self.dim == 3:
            return Y[torch.arange(B, device=Y.device), torch.as_tensor(positions, device=Y.device), :]
        hs, ws = zip(*[self.hw(p) for p in positions])
        return Y[torch.arange(B, device=Y.device), :, list(hs), list(ws)]


def choose_positions(layout, n_pos, cls_offset=0, valid_T=None):
    """n_pos probe positions: sequences, evenly spaced from T/8 to T-1 after the CLS/register
    tokens (and within the valid span for padded audio); maps, an interior grid."""
    if layout.dim == 3:
        T = layout.T if valid_T is None else min(layout.T, valid_T)
        lo = max(cls_offset, T // 8)
        hi = T - 1
        if hi <= lo:
            return list(range(max(cls_offset, 0), T))[:n_pos] or [0]
        pos = np.unique(np.round(np.linspace(lo, hi, n_pos)).astype(int)).tolist()
        return pos
    H, W = layout.H, layout.W
    k = int(math.ceil(math.sqrt(n_pos)))
    hs = np.round(np.linspace(0.2 * (H - 1), 0.8 * (H - 1), k)).astype(int)
    ws = np.round(np.linspace(0.2 * (W - 1), 0.8 * (W - 1), k)).astype(int)
    grid = [int(h * W + w) for h in hs for w in ws]
    grid = list(dict.fromkeys(grid))
    return grid[:n_pos]


# ---------------------------------------------------------------- Jacobian products
def _flat_norm(x):
    return x.flatten(1).norm(dim=1).view(-1, *([1] * (x.dim() - 1)))


def jvp_func(f, x, v):
    return torch.func.jvp(f, (x,), (v,))[1]


def jvp_double_backward(f, x, v):
    x = x.detach().requires_grad_(True)
    y = f(x)
    u = torch.zeros_like(y, requires_grad=True)
    g = torch.autograd.grad(y, x, grad_outputs=u, create_graph=True)[0]
    return torch.autograd.grad(g, u, grad_outputs=v.to(g.dtype))[0].detach()


def jvp_fd(f, x, v, eps=1e-3):
    with torch.no_grad():
        s = _flat_norm(x).clamp_min(1e-8) * eps
        return (f(x + s * v) - f(x)) / s


def vjp(f, x, u):
    x = x.detach().requires_grad_(True)
    y = f(x)
    return torch.autograd.grad(y, x, grad_outputs=u.to(y.dtype))[0].detach()


JVP_METHODS = [("torch.func.jvp", jvp_func), ("double-backward", jvp_double_backward), ("finite-difference", jvp_fd)]


def sigma1_power_iteration(f, x0, n_restarts, iters, tol, jvp_state):
    """Batched power iteration on J^T J. x0: (B0, ...) base points; returns (sigma[B0], n_iter).
    jvp_state is a dict caching which JVP implementation works for this model."""
    B0 = x0.shape[0]
    x = x0.repeat_interleave(n_restarts, dim=0)
    g = torch.Generator(device=x.device).manual_seed(SEED)
    v = torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32)
    v = v / _flat_norm(v)
    sigma = torch.zeros(x.shape[0], device=x.device)
    n_done = 0
    methods = jvp_state.get("methods", JVP_METHODS)
    for it in range(iters):
        Jv = None
        start = jvp_state.get("idx", 0)
        for k in range(start, len(methods)):
            name, fn = methods[k]
            try:
                Jv = fn(f, x, v).float()
                if not torch.isfinite(Jv).all():
                    raise RuntimeError("non-finite JVP")
                if k != start:
                    log(f"    JVP via {name} (earlier methods failed)")
                jvp_state["idx"] = k
                jvp_state["name"] = name
                break
            except Exception as e:
                jvp_state.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
                Jv = None
        if Jv is None:
            if jvp_state.get("tolerate_nonfinite"):
                jvp_state["n_nonfinite"] = jvp_state.get("n_nonfinite", 0) + 1
                return torch.full((B0,), float("nan"), device=x.device), it
            raise RuntimeError("all JVP methods failed: " + " | ".join(jvp_state.get("errors", [])))
        s_new = _flat_norm(Jv).flatten()
        u = Jv / _flat_norm(Jv).clamp_min(1e-30)
        JTu = vjp(f, x, u).float()
        v = JTu / _flat_norm(JTu).clamp_min(1e-30)
        n_done = it + 1
        if it >= 4:
            rel = ((s_new - sigma).abs() / s_new.clamp_min(1e-30)).max().item()
            sigma = s_new
            if rel < tol:
                break
        else:
            sigma = s_new
    sigma = sigma.view(B0, n_restarts).max(dim=1).values
    return sigma, n_done


def rho_power_iteration(f, x0, n_restarts, iters, tol, jvp_state):
    """Power iteration on J itself: v <- J v / |J v|. Converges to the spectral radius |lambda_max|
    when the dominant eigenvalue is real; with a complex dominant pair the iterate oscillates, so
    the value reported is the median of the last five |J v| per restart, max over restarts.
    This is the quantity the earlier finite-difference variant estimated."""
    B0 = x0.shape[0]
    x = x0.repeat_interleave(n_restarts, dim=0)
    g = torch.Generator(device=x.device).manual_seed(SEED)
    v = torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32)
    v = v / _flat_norm(v)
    methods = jvp_state.get("methods", JVP_METHODS)
    hist = []
    n_done = 0
    for it in range(iters):
        Jv = None
        start = jvp_state.get("idx", 0)
        for k in range(start, len(methods)):
            name, fn = methods[k]
            try:
                Jv = fn(f, x, v).float()
                if not torch.isfinite(Jv).all():
                    raise RuntimeError("non-finite JVP")
                jvp_state["idx"] = k; jvp_state["name"] = name
                break
            except Exception as e:
                jvp_state.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
                Jv = None
        if Jv is None:
            if jvp_state.get("tolerate_nonfinite"):
                jvp_state["n_nonfinite"] = jvp_state.get("n_nonfinite", 0) + 1
                return torch.full((B0,), float("nan"), device=x.device), it
            raise RuntimeError("all JVP methods failed: " + " | ".join(jvp_state.get("errors", [])))
        r = _flat_norm(Jv).flatten()
        hist.append(r)
        v = Jv / _flat_norm(Jv).clamp_min(1e-30)
        n_done = it + 1
        if it >= 6:
            last = torch.stack(hist[-3:])
            if ((last.max(0).values - last.min(0).values) / last.mean(0).clamp_min(1e-30)).max().item() < tol:
                break
    tail = torch.stack(hist[-5:]) if len(hist) >= 5 else torch.stack(hist)
    rho = tail.median(dim=0).values.view(B0, n_restarts).max(dim=1).values
    return rho, n_done


# ---------------------------------------------------------------- the maps
def make_f_incontext(layer_call, hidden0, layout, positions, branch, model_dtype):
    """f: (B, d) -> (B, d) with B a multiple of len(positions); row i perturbs position
    positions[i // rep] (rows are ordered position-major, matching repeat_interleave)."""
    P = len(positions)
    base1 = hidden0.float()
    M1 = layout.mask(positions, hidden0.device)
    cache = {}

    def f(xb):
        B = xb.shape[0]
        rep = B // P
        if rep not in cache:
            pos_rep = [p for p in positions for _ in range(rep)]
            cache[rep] = (base1.expand(B, *base1.shape[1:]), M1.repeat_interleave(rep, dim=0), pos_rep)
        base, M, pos_rep = cache[rep]
        X = base * (1.0 - M) + layout.broadcast(xb) * M
        Y = layer_call(X.to(model_dtype))
        y = layout.gather(Y, pos_rep).float()
        return y - xb if branch else y
    return f


def make_f_fullseq(layer_call, branch, model_dtype):
    def f(X):
        Y = layer_call(X.to(model_dtype)).float()
        return Y - X if branch else Y
    return f


def _slice_to_position(obj, t, T):
    """Slice every tensor dim of size T down to [t:t+1] (masks, rotary tables, position ids)."""
    if torch.is_tensor(obj):
        if obj.dim() == 0:
            return obj
        out = obj
        for dim in range(obj.dim()):
            if out.shape[dim] == T and T > 1:
                out = out.narrow(dim, t, 1)
        return out
    if isinstance(obj, (tuple, list)):
        return type(obj)(_slice_to_position(o, t, T) for o in obj)
    if isinstance(obj, dict):
        return {k: _slice_to_position(v, t, T) for k, v in obj.items()}
    return obj


def make_f_isolated(layer, rest, kw, layout, t, branch, model_dtype):
    rest_t = _slice_to_position(rest, t, layout.T)
    kw_t = _slice_to_position(kw, t, layout.T)
    call = make_layer_call(layer, rest_t, kw_t)

    def f(xb):                      # (B, d) -> (B, d); each row is a length-1 sequence
        Y = call(xb[:, None, :].to(model_dtype))
        y = Y[:, 0, :].float()
        return y - xb if branch else y
    return f


# ---------------------------------------------------------------- profile
def profile_model(b, context, inputs_kind, branch, args, inputs=None):
    """Returns the per-model result dict for one protocol."""
    arch = b["arch"]
    if branch and arch not in BRANCH_OK:
        raise ValueError(f"--branch is defined for pre-norm residual blocks only ({BRANCH_OK}); {arch} is not")
    if context == "isolated" and b["kind"] != "text":
        raise ValueError("isolated context is defined for sequence models only")
    layers = b["layers"]
    L = len(layers)
    if inputs is None:
        inputs = prepare_inputs(b, inputs_kind, args.n_inputs)
    model_dtype = b["dtype"]
    reduced = model_dtype != torch.float32 or getattr(args, "force_fd", False)
    if reduced:
        # Forward-mode AD is not reliable in half precision, so the reduced-precision comparison
        # uses the finite-difference protocol of the April float16 note (relative step fd_eps)
        # with the block run in the reduced dtype; non-finite outputs are recorded, not fatal.
        from functools import partial
        eps = getattr(args, "fd_eps", 1e-3)
        jvp_state = {"methods": [(f"finite-difference eps={eps:g} ({str(model_dtype).split('.')[-1]})", partial(jvp_fd, eps=eps))],
                     "tolerate_nonfinite": True, "fd_eps": eps}
        log(f"    reduced precision ({str(model_dtype).split('.')[-1]}): finite-difference JVP, relative step {eps:g}")
    else:
        jvp_state = {}
    use_rho = getattr(args, "estimator", "sigma1") == "rho"
    iterate = rho_power_iteration if use_rho else sigma1_power_iteration
    capture_diag = []
    samples = [[] for _ in range(L)]        # per layer: list over (input, position) or (input,) for fullseq
    iters_used = []
    positions_log = []
    t0 = time.time()
    for ii, inp in enumerate(inputs):
        captured = capture_layer_inputs(b, inp)
        with torch.no_grad():
            hmax = [float(c[0].float().abs().max()) for c in captured]
            hfin = [bool(torch.isfinite(c[0]).all()) for c in captured]
        capture_diag.append({"max_abs_hidden": hmax, "finite": hfin})
        if not all(hfin):
            log(f"    input {ii + 1}: non-finite captured hidden state at blocks {[i for i, ok in enumerate(hfin) if not ok]}")
        elif ii == 0:
            log(f"    captured hidden states: max |h| {max(hmax):.3g} (block {int(np.argmax(hmax))}), all finite")
        for li in range(L):
            hidden0, rest, kw = captured[li]
            layout = Layout(hidden0)
            layer_call = make_layer_call(layers[li], rest, kw)
            if context == "fullseq":
                f = make_f_fullseq(layer_call, branch, model_dtype)
                x0 = hidden0.float()
                sig, n_it = iterate(f, x0, args.restarts, args.iters, PI_TOL, jvp_state)
                samples[li].append(float(sig[0]))
                iters_used.append(n_it)
                continue
            positions = choose_positions(layout, args.n_positions, b.get("cls_offset", 0), inp.get("valid_T"))
            if ii == 0 and li in (0, L - 1):
                positions_log.append((li, tuple(hidden0.shape), positions))
            if context == "isolated":
                for t in positions:
                    f = make_f_isolated(layers[li], rest, kw, layout, t, branch, model_dtype)
                    x0 = layout.get(hidden0.float(), [t])
                    sig, n_it = iterate(f, x0, args.restarts, args.iters, PI_TOL, jvp_state)
                    samples[li].append(float(sig[0]))
                    iters_used.append(n_it)
                continue
            # in-context: all positions in one batch, falling back to one position at a time
            x0 = layout.get(hidden0.float(), positions)
            try:
                f = make_f_incontext(layer_call, hidden0, layout, positions, branch, model_dtype)
                sig, n_it = iterate(f, x0, args.restarts, args.iters, PI_TOL, jvp_state)
                samples[li].extend([float(s) for s in sig])
                iters_used.append(n_it)
            except Exception as e:
                if li == 0 and ii == 0:
                    log(f"    batched positions failed ({type(e).__name__}: {str(e)[:100]}); one position at a time")
                jvp_state.pop("idx", None)
                for j, t in enumerate(positions):
                    f = make_f_incontext(layer_call, hidden0, layout, [t], branch, model_dtype)
                    sig, n_it = iterate(f, x0[j:j + 1], args.restarts, args.iters, PI_TOL, jvp_state)
                    samples[li].append(float(sig[0]))
                    iters_used.append(n_it)
        del captured
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        done = [np.mean(s) for s in samples]
        log(f"    input {ii + 1}/{len(inputs)} done ({time.time() - t0:.0f} s); "
            f"block0 {done[0]:.3g}, mid {done[L // 2]:.3g}, last {done[-1]:.3g}")
    prof = np.array([np.nanmean(s) if np.isfinite(s).any() else np.nan for s in samples])
    sd = np.array([np.nanstd(s) if np.isfinite(s).any() else np.nan for s in samples])
    med = np.array([np.nanmedian(s) if np.isfinite(s).any() else np.nan for s in samples])
    if jvp_state.get("n_nonfinite"):
        log(f"    non-finite JVP in {jvp_state['n_nonfinite']} probe batches; affected blocks: "
            f"{[i for i, s in enumerate(samples) if not np.isfinite(s).all()]}")
    stats = profile_stats(prof)
    ci = bootstrap_R_ci(samples)
    res = dict(
        hf_id=b["hf_id"], label=b["cfg"]["label"], arch=arch, family=b["cfg"]["family"],
        params_M=b["cfg"]["params_M"], in_april_panel=b["cfg"]["april"], n_layers=L,
        version=VERSION, protocol=protocol_tag(context, inputs_kind, branch, str(model_dtype).split(".")[-1], getattr(args, "force_fd", False),
                                               getattr(args, "estimator", "sigma1")),
        estimator=(("rho: power iteration on J" if use_rho else "sigma1: power iteration on J^T J")
                   + (" via finite differences" if reduced else " via AD (jvp/vjp)")),
        context=context, inputs=inputs_kind, branch=branch, dtype=str(model_dtype).split(".")[-1],
        definition="sigma_1 of d h_{l+1,t}/d h_{l,t}, other positions held fixed" if context == "incontext" else
                   ("sigma_1 of the block Jacobian for a length-1 sequence at position t" if context == "isolated" else
                    "sigma_1 of the whole-sequence block Jacobian"),
        jvp_method=jvp_state.get("name"), jvp_errors=jvp_state.get("errors", [])[:5],
        fd_eps=jvp_state.get("fd_eps"), n_nonfinite_probe_batches=jvp_state.get("n_nonfinite", 0),
        capture_diagnostics=capture_diag,
        n_inputs=len(inputs), n_positions=args.n_positions, n_restarts=args.restarts, pi_iters=args.iters,
        pi_tol=PI_TOL, mean_iters=float(np.mean(iters_used)) if iters_used else None,
        converged_frac=float(np.mean([n < args.iters for n in iters_used])) if iters_used else None,
        input_desc=b.get("input_desc"), positions_example=positions_log,
        sigma1_profile=prof.tolist(), sigma1_sd=sd.tolist(), sigma1_median=med.tolist(),
        sigma1_samples=[list(map(float, s)) for s in samples],
        **stats, R_ex0_thirds_ci95=ci,
        time_s=time.time() - t0, torch=torch.__version__, transformers=_tf_version(),
        gpu=torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
        date=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    return res


def _tf_version():
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return None


# ---------------------------------------------------------------- statistics
def R_thirds(profile, exclude0=True):
    p = np.asarray(profile, dtype=float)
    p = p[1:] if exclude0 else p
    n = len(p)
    if n < 3:
        return float("nan"), float("nan")
    t = n // 3
    early, mid, late = p[:t], p[t:n - t], p[n - t:]
    edge = np.concatenate([early, late]).mean()
    R = mid.mean() / edge if edge > 0 else float("nan")
    uptick = late.mean() / mid.mean() if mid.mean() > 0 else float("nan")
    return float(R), float(uptick)


def R_vertex(profile):
    """May-census R_ex0: waist median / edge median about the log-quadratic vertex, block 0 excluded."""
    prof = np.asarray(profile, dtype=float)
    n = len(prof)
    if n < 3:
        return float("nan")
    d = np.linspace(0, 1, n)
    X = np.column_stack([d ** 2, d, np.ones(n)])
    beta = np.linalg.lstsq(X, np.log(prof + 1e-12), rcond=None)[0]
    a = beta[0]
    d_star = np.clip(-beta[1] / (2 * a), 0.0, 1.0) if a > 0 else 0.5
    waist = (np.abs(d - d_star) <= 0.20) & (np.arange(n) > 0)
    edge = ((d <= 0.15) | (d >= 0.85)) & (np.arange(n) > 0)
    if waist.sum() == 0 or edge.sum() == 0:
        return float("nan")
    e = np.median(prof[edge])
    return float(np.median(prof[waist]) / e) if e > 0 else float("nan")


def contrast_and_curvature(profile):
    """Census classifier geometry (d3g v5): quadratic coefficient a on [0,1] and
    spectral contrast C = (edge - waist) / edge with 3-block edge means and a 3-block waist."""
    s1 = np.asarray(profile, dtype=float)
    if len(s1) < 4:
        return float("nan"), float("nan")
    x = np.linspace(0, 1, len(s1))
    a = float(np.polyfit(x, s1, 2)[0])
    mid = len(s1) // 2
    edge = (np.mean(s1[:3]) + np.mean(s1[-3:])) / 2
    waist = np.mean(s1[mid - 1:mid + 2])
    C = float((edge - waist) / edge) if edge > 0 else float("nan")
    return C, a


def profile_stats(prof):
    R, up = R_thirds(prof, exclude0=True)
    R_incl0, up_incl0 = R_thirds(prof, exclude0=False)
    C_ex0, a_ex0 = contrast_and_curvature(prof[1:])
    C_incl0, a_incl0 = contrast_and_curvature(prof)
    return dict(R_ex0_thirds=R, late_uptick=up, hourglass=bool(R < 0.80 and up > 1.0) if np.isfinite(R) else None,
                R_thirds_incl0=R_incl0, R_ex0_vertex=R_vertex(prof),
                contrast_C_ex0=C_ex0, quad_a_ex0=a_ex0, contrast_C_incl0=C_incl0, quad_a_incl0=a_incl0,
                block0_over_block1=float(prof[0] / prof[1]) if len(prof) > 1 and prof[1] > 0 else None)


def bootstrap_R_ci(samples, n_boot=N_BOOT):
    """95% CI of R_ex0_thirds, resampling the (input, position) samples with replacement."""
    L = len(samples)
    n = min(len(s) for s in samples)
    if n < 2:
        return [None, None]
    arr = np.array([s[:n] for s in samples], dtype=float)          # (L, n)
    rng = np.random.default_rng(SEED)
    Rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        with np.errstate(all="ignore"):
            prof = np.nanmean(arr[:, idx], axis=1)
        Rs.append(R_thirds(prof)[0])
    Rs = np.array([r for r in Rs if np.isfinite(r)])
    if len(Rs) == 0:
        return [None, None]
    return [float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))]


# ---------------------------------------------------------------- io
def result_path(results_dir, tag, hf_id):
    return os.path.join(results_dir, tag, f"sigma1_{safe_name(hf_id)}.json")


def write_summary(results_dir, tag):
    d = os.path.join(results_dir, tag)
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("sigma1_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(d, fn)))
            rows.append({k: r.get(k) for k in (
                "label", "hf_id", "family", "arch", "n_layers", "params_M", "in_april_panel", "R_ex0_thirds",
                "R_ex0_thirds_ci95", "late_uptick", "hourglass", "R_ex0_vertex", "contrast_C_ex0", "quad_a_ex0",
                "contrast_C_incl0", "quad_a_incl0", "block0_over_block1", "jvp_method", "converged_frac",
                "mean_iters", "time_s", "protocol")})
    if not rows:
        return
    for r in rows:
        for k in ("R_ex0_thirds", "late_uptick", "R_ex0_vertex", "contrast_C_ex0", "quad_a_ex0"):
            if r.get(k) is None:
                r[k] = float("nan")
    rows.sort(key=lambda r: (not r["in_april_panel"], r["family"], r["params_M"] or 0))
    json.dump(rows, open(os.path.join(d, "survey_summary.json"), "w"), indent=2)
    import csv
    with open(os.path.join(d, "survey_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["R_ex0_thirds_ci95"] = "" if not r["R_ex0_thirds_ci95"] or r["R_ex0_thirds_ci95"][0] is None else \
                f"{r['R_ex0_thirds_ci95'][0]:.3f}-{r['R_ex0_thirds_ci95'][1]:.3f}"
            w.writerow(rr)
    with open(os.path.join(d, "survey_summary.md"), "w") as f:
        f.write(f"Protocol {tag}\n\n| Model | Family | Blocks | R_ex0 | 95% CI | uptick | HG | R_vertex | C_ex0 | a_ex0 |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            ci = r["R_ex0_thirds_ci95"]
            ci_s = f"{ci[0]:.2f}-{ci[1]:.2f}" if ci and ci[0] is not None else ""
            f.write(f"| {r['label']} | {r['family']} | {r['n_layers']} | {r['R_ex0_thirds']:.3f} | {ci_s} | "
                    f"{r['late_uptick']:.2f} | {'yes' if r['hourglass'] else 'no'} | {r['R_ex0_vertex']:.3f} | "
                    f"{r['contrast_C_ex0']:.3f} | {r['quad_a_ex0']:.3f} |\n")
    april = [r for r in rows if r["in_april_panel"]]
    log(f"\n  {tag}: {len(rows)} models; April panel {sum(bool(r['hourglass']) for r in april)}/{len(april)} hourglass-positive;"
        f" all {sum(bool(r['hourglass']) for r in rows)}/{len(rows)}")
    log(f"  {'model':26s} {'R_ex0':>6s} {'uptick':>6s} {'HG':>3s} {'C_ex0':>6s}")
    for r in rows:
        log(f"  {r['label']:26s} {r['R_ex0_thirds']:6.3f} {r['late_uptick']:6.2f} {'yes' if r['hourglass'] else 'no':>3s} {r['contrast_C_ex0']:6.3f}")


# ---------------------------------------------------------------- main
def run_validation(cfg, args):
    """Exact token-local Jacobians at five blocks x three positions x two natural inputs:
    exact sigma_1 (SVD), exact rho (eigenvalues), the sigma_1 and rho power-iteration estimates,
    sigma_1 of J - I, and a 64-direction random-probe estimate (max |J v| over random unit v)."""
    out_dir = os.path.join(args.results_dir, "validation")
    path = os.path.join(out_dir, f"validation_{safe_name(cfg['hf_id'])}.json")
    if os.path.exists(path) and not args.no_skip:
        log(f"  skip validation {cfg['hf_id']} (exists)"); return
    b = load_bundle(cfg, "float32")
    if b["kind"] == "audio":
        log("  validation mode: audio models not supported"); return
    # 18 Sept: image models allowed. Layout handles (1,C,H,W) maps; the token is a spatial
    # position, d = channels, and choose_positions gives an interior grid.
    inputs = prepare_inputs(b, "natural", 2)
    L = len(b["layers"])
    blocks = sorted(set([0, L // 4, L // 2, (3 * L) // 4, L - 1]))
    rows = []
    t0 = time.time()
    for ii, inp in enumerate(inputs):
        captured = capture_layer_inputs(b, inp)
        for li in blocks:
            hidden0, rest, kw = captured[li]
            layout = Layout(hidden0)
            call = make_layer_call(b["layers"][li], rest, kw)
            positions = choose_positions(layout, 3, 0, inp.get("valid_T"))
            for t in positions:
                f = make_f_incontext(call, hidden0, layout, [t], False, torch.float32)
                x0 = layout.get(hidden0.float(), [t])
                d = x0.shape[1]
                try:
                    J = torch.autograd.functional.jacobian(f, x0, vectorize=True)
                except Exception:
                    J = torch.autograd.functional.jacobian(f, x0)
                J = J.reshape(d, d).detach()
                sv = torch.linalg.svdvals(J)
                ev = torch.linalg.eigvals(J).abs()
                s1_exact, s2_exact, rho_exact = sv[0].item(), sv[1].item(), ev.max().item()
                s1_branch = torch.linalg.svdvals(J - torch.eye(d, device=J.device))[0].item()
                s1_est, n1 = sigma1_power_iteration(f, x0, args.restarts, args.iters, PI_TOL, {})
                rho_est, n2 = rho_power_iteration(f, x0, args.restarts, args.iters, PI_TOL, {})
                g = torch.Generator(device=J.device).manual_seed(SEED)
                V = torch.randn(64, d, generator=g, device=J.device); V = V / V.norm(dim=1, keepdim=True)
                probe = (V @ J.T).norm(dim=1).max().item()
                rows.append(dict(input=ii, block=li, position=int(t), d=d, sigma1_exact=s1_exact, sigma2_exact=s2_exact,
                                 rho_exact=rho_exact, sigma1_branch_exact=s1_branch, sigma1_est=float(s1_est[0]),
                                 sigma1_est_iters=n1, rho_est=float(rho_est[0]), rho_est_iters=n2, random_probe_64=probe))
                log(f"    input {ii} block {li:3d} pos {t:3d}: sigma1 {s1_exact:8.3f} (est {float(s1_est[0]):8.3f}, {n1} it)  "
                    f"rho {rho_exact:7.3f} (est {float(rho_est[0]):7.3f})  sigma1(J-I) {s1_branch:7.3f}  probes64 {probe:7.3f}  "
                    f"sigma1/rho {s1_exact / max(rho_exact, 1e-9):5.2f}")
        del captured
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    s1e = np.array([r["sigma1_exact"] for r in rows]); s1p = np.array([r["sigma1_est"] for r in rows])
    rhe = np.array([r["rho_exact"] for r in rows]); rhp = np.array([r["rho_est"] for r in rows]); pr = np.array([r["random_probe_64"] for r in rows])
    summary = dict(hf_id=cfg["hf_id"], label=cfg["label"], n=len(rows), blocks=blocks,
                   sigma1_max_rel_err=float(np.max(np.abs(s1p - s1e) / s1e)), sigma1_pearson_r=float(np.corrcoef(s1e, s1p)[0, 1]),
                   rho_max_rel_err=float(np.max(np.abs(rhp - rhe) / rhe)),
                   probe_underestimate_range=[float(np.min(1 - pr / s1e)), float(np.max(1 - pr / s1e))],
                   sigma1_over_rho_range=[float(np.min(s1e / rhe)), float(np.max(s1e / rhe))],
                   time_s=time.time() - t0, version=VERSION)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(summary=summary, rows=rows), open(path, "w"), indent=1)
    log(f"  validation: sigma1 estimator max rel err {summary['sigma1_max_rel_err']:.2e} (r = {summary['sigma1_pearson_r']:.4f}); "
        f"rho estimator max rel err {summary['rho_max_rel_err']:.2e}; random probes underestimate by "
        f"{100 * summary['probe_underestimate_range'][0]:.0f}-{100 * summary['probe_underestimate_range'][1]:.0f}%; "
        f"sigma1/rho {summary['sigma1_over_rho_range'][0]:.1f}-{summary['sigma1_over_rho_range'][1]:.1f}; saved {path}")
    del b; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _purge_hub_cache(hf_id):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = os.path.join(HF_HUB_CACHE, "models--" + hf_id.replace("/", "--"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            log(f"  purged Hub cache for {hf_id}")
    except Exception as e:
        log(f"  (cache purge failed: {e})")


def _print_smoke_anchors(path):
    p = json.load(open(path))["sigma1_profile"]
    log("  smoke reference (v35 Fig. 1a exact-Jacobian sigma_1 on GPT-2 124M): " +
        ", ".join(f"L{k} {v:.1f} (now {p[k]:.2f})" for k, v in GPT2_REFERENCE.items()))


def run_one(cfg, protocols, args):
    """Load a model once and run every requested protocol on it."""
    todo = []
    if args.validate:
        run_validation(cfg, args)
        return
    for (context, inputs_kind, branch, dtype_name) in protocols:
        tag = protocol_tag(context, inputs_kind, branch, dtype_name, args.force_fd, args.estimator)
        path = result_path(args.results_dir, tag, cfg["hf_id"])
        if os.path.exists(path) and not args.no_skip:
            log(f"  skip {cfg['hf_id']} [{tag}] (exists)")
            if args.smoke and cfg["hf_id"] == "gpt2" and context == "incontext" and not branch:
                _print_smoke_anchors(path)
            continue
        if branch and cfg["arch"] not in BRANCH_OK:
            log(f"  skip {cfg['hf_id']} [{tag}]: branch mode not defined for {cfg['arch']}")
            continue
        if context == "isolated" and cfg["arch"] not in TEXT_ARCHES:
            log(f"  skip {cfg['hf_id']} [{tag}]: isolated context not defined for {cfg['arch']}")
            continue
        todo.append((context, inputs_kind, branch, dtype_name, tag, path))
    if not todo:
        return
    by_dtype = {}
    for item in todo:
        by_dtype.setdefault(item[3], []).append(item)
    for dtype_name, items in by_dtype.items():
        b = None
        try:
            b = load_bundle(cfg, dtype_name)
            input_cache = {}
            for (context, inputs_kind, branch, _, tag, path) in items:
                log(f"  [{tag}]")
                if inputs_kind not in input_cache:
                    input_cache[inputs_kind] = prepare_inputs(b, inputs_kind, args.n_inputs)
                res = profile_model(b, context, inputs_kind, branch, args, inputs=input_cache[inputs_kind])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(res, open(path, "w"), indent=1)
                log(f"  saved {path}\n    R_ex0={res['R_ex0_thirds']:.3f} (CI {res['R_ex0_thirds_ci95']}), uptick={res['late_uptick']:.2f}, "
                    f"hourglass={res['hourglass']}, R_vertex={res['R_ex0_vertex']:.3f}, C_ex0={res['contrast_C_ex0']:.3f}, "
                    f"jvp={res['jvp_method']}, {res['time_s']:.0f} s")
                if args.smoke and cfg["hf_id"] == "gpt2" and context == "incontext" and not branch:
                    _print_smoke_anchors(path)
        except Exception as e:
            log(f"  FAILED {cfg['hf_id']} [{dtype_name}]: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            if b is not None:
                del b
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    if args.purge_cache:
        _purge_hub_cache(cfg["hf_id"])


def main(argv=None):
    args = parse_args(argv)
    _resolve_hf_token()
    log(f"{VERSION}; device {DEVICE}; torch {torch.__version__}; transformers {_tf_version()}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        a = torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")
        log(f"GPU: {p.name} ({p.total_memory / 1e9:.0f} GB, sm_{p.major}{p.minor}); TF32 conv={torch.backends.cudnn.allow_tf32} "
            f"matmul={torch.backends.cuda.matmul.allow_tf32}; HF token: {'found' if os.environ.get('HF_TOKEN') else 'none'}")
    log(f"settings: inputs {args.n_inputs} x positions {args.n_positions}, restarts {args.restarts}, iters {args.iters}, tol {PI_TOL}")

    if args.protocol_panel:
        models = args.models or PROTOCOL_PANEL_MODELS
        protocols = [(c, i, False, "float32") for (c, i) in PROTOCOL_PANEL]
    else:
        if args.models:
            models = args.models
        elif args.smoke:
            models = ["gpt2"]
        else:
            models = [r[0] for r in REGISTRY if (args.panel == "union" or r[5])]
        protocols = [(args.context, args.inputs, args.branch, args.dtype)]
    unknown = [m for m in models if m not in CFG]
    if unknown:
        raise SystemExit(f"unknown model ids (add to REGISTRY): {unknown}")
    order = {r[0]: i for i, r in enumerate(REGISTRY)}
    models = sorted(models, key=lambda m: (CFG[m]["params_M"], order[m]))
    log(f"models ({len(models)}): {models}")
    log(f"protocols: {[protocol_tag(*p, args.force_fd, args.estimator) for p in protocols]}" + (" (validation mode)" if args.validate else ""))
    os.makedirs(args.results_dir, exist_ok=True)
    t0 = time.time()
    for i, m in enumerate(models):
        log(f"\n{'=' * 78}\n[{i + 1}/{len(models)}] {m} ({CFG[m]['arch']}, ~{CFG[m]['params_M']}M)\n{'=' * 78}")
        run_one(CFG[m], protocols, args)
    for p in protocols:
        tag = protocol_tag(*p, args.force_fd, args.estimator)
        if os.path.isdir(os.path.join(args.results_dir, tag)):
            write_summary(args.results_dir, tag)
    log(f"\nTotal {time.time() - t0:.0f} s")
    if args.zip:
        z = shutil.make_archive("survey_results", "zip", args.results_dir)
        log(f"zipped: {z}")


if __name__ == "__main__":
    main()
