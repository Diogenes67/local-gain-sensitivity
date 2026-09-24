"""
Per-layer branch-rotation census, v4: the 14 non-decoder models re-run with
branch-level hooks and task readouts on natural inputs.
=========================================================================
Why. The May 2026 run (delta_l_census_reclass.py) that supplied 28 of the 45 census
files treated the 14 non-decoder models incorrectly: ViT, DINOv2, MLP-Mixer, Whisper
and wav2vec2 were fed random tensors and read out by the variance of the final hidden
state; T5 was read out by the variance of its encoder output; BERT, RoBERTa, T5 and the
ViT feed-forward were hooked on Hugging Face module outputs that already include the
residual add, so those were residual-state rotations, which saturate. This script
re-measures all 14 with hooks on the branch output before the residual add and with a
task readout on natural data:

  BERT-base/large, RoBERTa-base/large   masked-LM cross-entropy, WikiText-103 validation, 15% masking (80/10/10)
  T5-small/base/large, Flan-T5-large    span-corruption (denoising) cross-entropy, WikiText-103, 15% noise, mean span 3;
                                        encoder blocks are scrambled, the decoder is untouched
  ViT-base, DINOv2-base (linear head),  KL(clean || scrambled) over the ImageNet-1k head on CIFAR-100 test images,
  MLP-Mixer B/16, L/16                  as the ConvNeXt census already does (self-label cross-entropy recorded alongside)
  wav2vec2-base-960h                    CTC loss against the reference transcript, LibriSpeech test-clean (first 10 s)
  Whisper-small                         teacher-forced decoder cross-entropy on the reference transcript, LibriSpeech;
                                        encoder blocks are scrambled, the decoder is untouched

Every model also records KL(clean || scrambled) over its own output distribution, so the
whole census can be analysed under one label-free readout.

Branch hooks (verified on transformers 5.17; the resolver falls back to the 4.x names):
  BERT/RoBERTa  layer.attention.output.dense, layer.output.dense      (Linear outputs, before LayerNorm(h + x))
  T5 encoder    block.layer[0].SelfAttention (tuple[0]), block.layer[1].DenseReluDense
  ViT           layer.attention (tuple[0]), layer.mlp                  (4.x: layer.attention, layer.output.dense)
  DINOv2        layer.attention, layer.mlp                             (before layer-scale and the residual add)
  MLP-Mixer     block.mlp_tokens (channel axis), block.mlp_channels
  wav2vec2      layer.attention (tuple[0]), layer.feed_forward
  Whisper enc.  layer.self_attn (tuple[0]), layer.fc2

Protocol otherwise as the census: Haar-random orthogonal rotation, dose 1, seed 42, one
block at a time, 30 evaluation batches (16 sequences or images; 4 utterances), 200-sample
bootstrap CI on delta-L over batches. Two plumbing checks per model: dose-0 hooks reproduce
the clean loss to 1e-6, and dose-1 hooks on block 1 change it.

Outputs: one JSON per model, written atomically, resume-safe (a finished model is skipped);
summary table from disk; zip export. sigma1 profiles are NOT recomputed here: the census
takes geometry from the survey JSONs (survey_sigma1_v2, canonical protocol).

Usage:
  python scramble_census_v4.py                       # all 14, resume-safe
  python scramble_census_v4.py --models google-bert/bert-base-uncased
  python scramble_census_v4.py --quick               # 2 batches per model, plumbing only
  python scramble_census_v4.py --plumbing            # tiny random models, no network (container test)
Environment variables (for notebooks): CENSUS_OUT, CENSUS_MODELS, CENSUS_QUICK=1, CENSUS_DATA.
"""
import os, sys, json, time, gc, argparse, shutil, datetime, platform
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SEQ_LEN = 128
N_EVAL_BATCHES = 30
EVAL_BATCH_SIZE = 16
AUDIO_BATCH_SIZE = 4
AUDIO_MAX_SEC = 10.0
N_BOOTSTRAP = 200
DOSE = 1.0
PROTOCOL = "branch-rotation-v4"
VERSION = "census-v4-2026-09-20"


def log(msg):
    print(msg, flush=True)


MODELS = [
    {"hf_id": "google-bert/bert-base-uncased",  "arch": "bert",     "n_layers": 12, "params_M": 110, "family": "Masked encoder"},
    {"hf_id": "google-bert/bert-large-uncased", "arch": "bert",     "n_layers": 24, "params_M": 335, "family": "Masked encoder"},
    {"hf_id": "FacebookAI/roberta-base",        "arch": "roberta",  "n_layers": 12, "params_M": 125, "family": "Masked encoder"},
    {"hf_id": "FacebookAI/roberta-large",       "arch": "roberta",  "n_layers": 24, "params_M": 355, "family": "Masked encoder"},
    {"hf_id": "google-t5/t5-small",             "arch": "t5",       "n_layers": 6,  "params_M": 60,  "family": "Encoder-decoder"},
    {"hf_id": "google-t5/t5-base",              "arch": "t5",       "n_layers": 12, "params_M": 220, "family": "Encoder-decoder"},
    {"hf_id": "google-t5/t5-large",             "arch": "t5",       "n_layers": 24, "params_M": 770, "family": "Encoder-decoder"},
    {"hf_id": "google/flan-t5-large",           "arch": "t5",       "n_layers": 24, "params_M": 780, "family": "Encoder-decoder"},
    {"hf_id": "google/vit-base-patch16-224",    "arch": "vit",      "n_layers": 12, "params_M": 86,  "family": "Vision supervised"},
    {"hf_id": "facebook/dinov2-base",           "arch": "dinov2",   "n_layers": 12, "params_M": 86,  "family": "Vision SSL",
     "head_id": "facebook/dinov2-base-imagenet1k-1-layer"},
    {"hf_id": "timm/mixer_b16_224.goog_in21k_ft_in1k", "arch": "mixer", "n_layers": 12, "params_M": 59,  "family": "MLP-Mixer (vision)"},
    {"hf_id": "timm/mixer_l16_224.goog_in21k_ft_in1k", "arch": "mixer", "n_layers": 24, "params_M": 207, "family": "MLP-Mixer (vision)"},
    {"hf_id": "facebook/wav2vec2-base-960h",    "arch": "wav2vec2", "n_layers": 12, "params_M": 95,  "family": "Audio encoder"},
    {"hf_id": "openai/whisper-small",           "arch": "whisper",  "n_layers": 12, "params_M": 244, "family": "Audio encoder"},
]

TEXT_ARCHES = ("bert", "roberta")
VISION_ARCHES = ("vit", "dinov2", "mixer")
AUDIO_ARCHES = ("wav2vec2", "whisper")


def _resolve_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
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


# ---------------------------------------------------------------- layers and branch hooks
def _get(obj, path):
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _has(obj, path):
    try:
        _get(obj, path)
        return True
    except (AttributeError, IndexError, KeyError):
        return False


def get_layers(model, arch):
    if arch == "bert":
        return list(model.bert.encoder.layer)
    if arch == "roberta":
        return list(model.roberta.encoder.layer)
    if arch == "t5":
        return list(model.encoder.block)
    if arch == "vit":
        core = model.vit
        return list(core.layers) if _has(core, "layers") else list(core.encoder.layer)
    if arch == "dinov2":
        core = model.dinov2
        return list(core.encoder.layer) if _has(core, "encoder.layer") else list(core.layers)
    if arch == "mixer":
        return list(model.blocks)
    if arch == "wav2vec2":
        return list(model.wav2vec2.encoder.layers)
    if arch == "whisper":
        return list(model.model.encoder.layers)
    raise ValueError(arch)


def branch_hooks(layer, arch):
    """(module, path, tuple_index, axis) for the two branch outputs of a block, before the residual add."""
    if arch in ("bert", "roberta"):
        return [(_get(layer, "attention.output.dense"), "attention.output.dense", None, -1),
                (_get(layer, "output.dense"), "output.dense", None, -1)]
    if arch == "t5":
        return [(_get(layer, "layer.0.SelfAttention"), "layer.0.SelfAttention", 0, -1),
                (_get(layer, "layer.1.DenseReluDense"), "layer.1.DenseReluDense", None, -1)]
    if arch == "vit":
        if _has(layer, "mlp"):                                   # transformers 5.x
            return [(layer.attention, "attention", 0, -1), (layer.mlp, "mlp", None, -1)]
        return [(layer.attention, "attention", 0, -1),            # 4.x: ViTOutput adds the residual, hook its dense
                (_get(layer, "output.dense"), "output.dense", None, -1)]
    if arch == "dinov2":
        return [(layer.attention, "attention", None, -1), (layer.mlp, "mlp", None, -1)]
    if arch == "mixer":
        return [(layer.mlp_tokens, "mlp_tokens", None, 1),        # output is (B, C, N): channels on axis 1
                (layer.mlp_channels, "mlp_channels", None, -1)]
    if arch == "wav2vec2":
        return [(layer.attention, "attention", 0, -1), (layer.feed_forward, "feed_forward", None, -1)]
    if arch == "whisper":
        return [(layer.self_attn, "self_attn", 0, -1), (layer.fc2, "fc2", None, -1)]
    raise ValueError(arch)


def random_orthogonal(d, seed):
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(H)
    return torch.from_numpy(Q @ np.diag(np.sign(np.diag(R))))


def rotate(x, R, dose, axis):
    R = R.to(x.device, dtype=x.dtype)
    xm = x.movedim(axis, -1)
    rot = torch.einsum("ij,...j->...i", R, xm)
    out = (1 - dose) * xm + dose * rot
    return out.movedim(-1, axis)


class HookManager:
    def __init__(self):
        self.handles = []

    def register(self, module, R, dose, tuple_index=None, axis=-1):
        def hook(mod, inp, output):
            if tuple_index is not None:
                out = list(output)
                out[tuple_index] = rotate(out[tuple_index], R, dose, axis)
                return tuple(out)
            return rotate(output, R, dose, axis)
        self.handles.append(module.register_forward_hook(hook))

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------- data
def wikitext_validation_texts():
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


def cifar100_test_images(n, root):
    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
        return [ds[i][0].convert("RGB") for i in range(n)]
    except Exception as e:
        log(f"  torchvision CIFAR-100 failed ({e}); falling back to datasets")
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar100", split="test")
        return [ds[i]["img"].convert("RGB") for i in range(n)]


def librispeech_test_clean(n, root, cache):
    """First n utterances of LibriSpeech test-clean, cropped to AUDIO_MAX_SEC, with transcripts; cached as .pt."""
    if cache and os.path.exists(cache):
        d = torch.load(cache)
        if len(d["wav"]) >= n:
            log(f"  LibriSpeech: {n} utterances from cache {cache}")
            return d["wav"][:n], d["text"][:n]
    wavs, texts = [], []
    try:
        import torchaudio
        ds = torchaudio.datasets.LIBRISPEECH(root=root, url="test-clean", download=True)
        for i in range(n):
            w, sr, text = ds[i][0], ds[i][1], ds[i][2]
            assert sr == 16000
            wavs.append(w[0, : int(AUDIO_MAX_SEC * sr)].clone()); texts.append(text)
    except Exception as e:
        log(f"  torchaudio LibriSpeech failed ({e}); falling back to datasets streaming")
        from datasets import load_dataset
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
        for ex in ds:
            a = ex["audio"]; assert a["sampling_rate"] == 16000
            wavs.append(torch.tensor(a["array"][: int(AUDIO_MAX_SEC * 16000)], dtype=torch.float32)); texts.append(ex["text"])
            if len(wavs) >= n:
                break
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"wav": wavs, "text": texts}, cache)
        log(f"  LibriSpeech: {len(wavs)} utterances cached to {cache}")
    return wavs, texts


def text_windows(tok, n_seqs, add_special, cls_id=None, sep_id=None):
    texts = wikitext_validation_texts()
    text = "\n".join(t for t in texts if len(t.strip()) > 50)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    win = SEQ_LEN - 2 if add_special else SEQ_LEN
    seqs = [ids[i:i + win] for i in range(0, len(ids) - win, win)][:n_seqs]
    if add_special:
        seqs = [[cls_id] + s + [sep_id] for s in seqs]
    log(f"  WikiText-103 validation: {len(ids)} tokens -> {len(seqs)} windows of {SEQ_LEN}")
    return torch.tensor(seqs, dtype=torch.long)


# ---------------------------------------------------------------- corruption (deterministic per batch)
def mlm_mask(batch, tok, seed):
    g = torch.Generator().manual_seed(seed)
    special = torch.zeros_like(batch, dtype=torch.bool)
    for sid in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
        if sid is not None:
            special |= batch == sid
    prob = torch.full(batch.shape, 0.15); prob[special] = 0.0
    masked = torch.bernoulli(prob, generator=g).bool()
    labels = batch.clone(); labels[~masked] = -100
    inp = batch.clone()
    r = torch.rand(batch.shape, generator=g)
    inp[masked & (r < 0.8)] = tok.mask_token_id
    rnd = masked & (r >= 0.8) & (r < 0.9)
    inp[rnd] = torch.randint(len(tok), batch.shape, generator=g)[rnd]
    return inp, labels, masked


def random_spans_noise_mask(length, rng, noise_density=0.15, mean_span=3.0):
    """T5's span-corruption mask (Raffel et al.; the HF run_t5_mlm implementation)."""
    num_noise = int(round(length * noise_density)); num_noise = min(max(num_noise, 1), length - 1)
    num_spans = max(int(round(num_noise / mean_span)), 1)
    num_nonnoise = length - num_noise

    def seg(num_items, num_segments):
        mask = np.arange(num_items - 1) < (num_segments - 1)
        rng.shuffle(mask)
        first = np.pad(mask, [[1, 0]])
        seg_id = np.cumsum(first)
        _, lengths = np.unique(seg_id, return_counts=True)
        return lengths

    noise_len = seg(num_noise, num_spans); nonnoise_len = seg(num_nonnoise, num_spans)
    inter = np.reshape(np.stack([nonnoise_len, noise_len], axis=1), [num_spans * 2])
    starts = np.cumsum(inter)[:-1]
    ind = np.zeros((length,), dtype=np.int8); ind[starts] = 1
    return (np.cumsum(ind) % 2 == 1)


def t5_corrupt(batch, tok, seed, sentinel0, eos_id, pad_id):
    """Inputs with each noise span replaced by one sentinel; targets sentinel + span, ending with EOS."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for row in batch.tolist():
        m = random_spans_noise_mask(len(row), rng)
        inp, tgt, k, prev = [], [], 0, False
        for tkn, is_noise in zip(row, m):
            if is_noise:
                if not prev:
                    inp.append(sentinel0 - k); tgt.append(sentinel0 - k); k += 1
                tgt.append(tkn)
            else:
                inp.append(tkn)
            prev = is_noise
        tgt.append(sentinel0 - k)
        inputs.append(inp + [eos_id]); targets.append(tgt + [eos_id])
    li, lt = max(map(len, inputs)), max(map(len, targets))
    inp_t = torch.full((len(inputs), li), pad_id, dtype=torch.long); att = torch.zeros((len(inputs), li), dtype=torch.long)
    lab_t = torch.full((len(targets), lt), -100, dtype=torch.long)
    for i, (a, b) in enumerate(zip(inputs, targets)):
        inp_t[i, :len(a)] = torch.tensor(a); att[i, :len(a)] = 1; lab_t[i, :len(b)] = torch.tensor(b)
    return inp_t, att, lab_t


# ---------------------------------------------------------------- readouts
def kl_from_logp(clean_logp, logp, mask=None):
    """KL(clean || current) averaged over positions (masked by `mask` if given)."""
    kl = (clean_logp.exp() * (clean_logp - logp)).sum(-1)
    if mask is not None:
        return (kl * mask).sum() / mask.sum().clamp(min=1)
    return kl.mean()


class Readout:
    """One object per model: prepares batches once, then evaluates loss and KL under any hooks."""

    def __init__(self, cfg, model, aux, n_batches, data_root, cache_dir):
        self.cfg, self.model, self.aux, self.arch = cfg, model, aux, cfg["arch"]
        self.n = n_batches
        self.batches = []
        self.clean_logp = [None] * n_batches
        a = self.arch
        if a in TEXT_ARCHES:
            tok = aux
            ids = text_windows(tok, n_batches * EVAL_BATCH_SIZE, True, tok.cls_token_id, tok.sep_token_id)
            for b in range(n_batches):
                chunk = ids[b * EVAL_BATCH_SIZE:(b + 1) * EVAL_BATCH_SIZE]
                inp, labels, masked = mlm_mask(chunk, tok, SEED + b)
                self.batches.append((inp, labels, masked))
            self.desc = "masked-LM cross-entropy at masked positions, WikiText-103 validation, 128-token windows, 15% masking (80/10/10), fixed masks"
        elif a == "t5":
            tok = aux
            ids = text_windows(tok, n_batches * EVAL_BATCH_SIZE, False)
            s0 = tok.convert_tokens_to_ids("<extra_id_0>")
            for b in range(n_batches):
                chunk = ids[b * EVAL_BATCH_SIZE:(b + 1) * EVAL_BATCH_SIZE]
                self.batches.append(t5_corrupt(chunk, tok, SEED + b, s0, tok.eos_token_id, tok.pad_token_id))
            self.desc = "span-corruption (denoising) cross-entropy over target tokens, WikiText-103 validation, 128-token windows, 15% noise, mean span 3, fixed spans; encoder scrambled, decoder untouched"
        elif a in VISION_ARCHES:
            imgs = cifar100_test_images(n_batches * EVAL_BATCH_SIZE, os.path.join(data_root, "cifar100"))
            if a == "mixer":
                import timm
                cfg_t = timm.data.resolve_data_config({}, model=model)
                tf = timm.data.create_transform(**cfg_t)
                px = torch.stack([tf(im) for im in imgs])
            else:
                px = aux(images=imgs, return_tensors="pt")["pixel_values"]
            for b in range(n_batches):
                self.batches.append((px[b * EVAL_BATCH_SIZE:(b + 1) * EVAL_BATCH_SIZE],))
            self.desc = "KL(clean || scrambled) over the ImageNet-1k head on CIFAR-100 test images (primary); self-label cross-entropy alongside"
        elif a in AUDIO_ARCHES:
            wavs, texts = librispeech_test_clean(n_batches * AUDIO_BATCH_SIZE, os.path.join(data_root, "librispeech"),
                                                 os.path.join(cache_dir, f"librispeech_test_clean_{n_batches * AUDIO_BATCH_SIZE}.pt"))
            proc = aux
            for b in range(n_batches):
                w = [x.numpy() for x in wavs[b * AUDIO_BATCH_SIZE:(b + 1) * AUDIO_BATCH_SIZE]]
                t = texts[b * AUDIO_BATCH_SIZE:(b + 1) * AUDIO_BATCH_SIZE]
                if a == "wav2vec2":
                    feats = proc(w, sampling_rate=16000, return_tensors="pt", padding=True)
                    lab = proc.tokenizer(t, return_tensors="pt", padding=True)
                    labels = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
                    lengths = torch.tensor([len(x) for x in w])
                    self.batches.append((feats["input_values"], labels, lengths))
                else:
                    feats = proc.feature_extractor(w, sampling_rate=16000, return_tensors="pt")
                    lab = proc.tokenizer([s.lower() for s in t], return_tensors="pt", padding=True)
                    labels = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
                    dst = model.config.decoder_start_token_id
                    if (labels[:, 0] == dst).all():
                        labels = labels[:, 1:]
                    self.batches.append((feats["input_features"], labels))
            self.desc = ("CTC loss against the reference transcript, LibriSpeech test-clean, first 10 s of each utterance" if a == "wav2vec2"
                         else "teacher-forced decoder cross-entropy on the reference transcript (lower-cased), LibriSpeech test-clean, first 10 s; encoder scrambled, decoder untouched")

    @torch.no_grad()
    def evaluate(self):
        """Returns per-batch (loss, kl, extra) under whatever hooks are registered."""
        losses, kls, extras = [], [], []
        m, a = self.model, self.arch
        for b, batch in enumerate(self.batches):
            if a in TEXT_ARCHES:
                inp, labels, masked = (x.to(DEVICE) for x in batch)
                out = m(input_ids=inp, labels=labels)
                logp = F.log_softmax(out.logits.float(), -1)
                loss = out.loss.item(); extra = None   # masked positions carry the labels
            elif a == "t5":
                inp, att, labels = (x.to(DEVICE) for x in batch)
                out = m(input_ids=inp, attention_mask=att, labels=labels)
                logp = F.log_softmax(out.logits.float(), -1); masked = labels != -100
                loss = out.loss.item(); extra = None
            elif a in VISION_ARCHES:
                px = batch[0].to(DEVICE)
                logits = m(px) if a == "mixer" else m(pixel_values=px).logits
                logp = F.log_softmax(logits.float(), -1); masked = None
                if self.clean_logp[b] is None:
                    self.clean_logp[b] = logp.detach().clone()
                cl = self.clean_logp[b]
                loss = kl_from_logp(cl, logp).item()
                extra = {"selflabel_ce": F.nll_loss(logp, cl.argmax(-1)).item()}
            elif a == "wav2vec2":
                x, labels, lengths = batch
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                out = m(input_values=x, labels=labels)
                logp = F.log_softmax(out.logits.float(), -1)
                n_frames = m._get_feat_extract_output_lengths(lengths).to(DEVICE)
                masked = torch.arange(logp.shape[1], device=DEVICE)[None, :] < n_frames[:, None]
                loss = out.loss.item(); extra = None
            elif a == "whisper":
                x, labels = (t.to(DEVICE) for t in batch)
                out = m(input_features=x, labels=labels)
                logp = F.log_softmax(out.logits.float(), -1); masked = labels != -100
                loss = out.loss.item(); extra = None
            else:
                raise ValueError(a)
            if a not in VISION_ARCHES:
                sel = logp[masked] if masked is not None else logp.reshape(-1, logp.shape[-1])   # scored positions only (memory)
                if self.clean_logp[b] is None:
                    self.clean_logp[b] = sel.detach().clone()
                kl = kl_from_logp(self.clean_logp[b], sel).item()
            else:
                kl = loss
            losses.append(loss); kls.append(kl); extras.append(extra)
        return losses, kls, extras


# ---------------------------------------------------------------- loading
def _from_pretrained(cls, hf_id, **kw):
    try:
        return cls.from_pretrained(hf_id, dtype=torch.float32, **kw)
    except TypeError:
        return cls.from_pretrained(hf_id, torch_dtype=torch.float32, **kw)


def load_model(cfg):
    arch, hf_id = cfg["arch"], cfg["hf_id"]
    token = os.environ.get("HF_TOKEN")
    log(f"  Loading {hf_id} ({arch})...")
    if arch in ("bert", "roberta"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        model = _from_pretrained(AutoModelForMaskedLM, hf_id, token=token, attn_implementation="eager")
        aux = AutoTokenizer.from_pretrained(hf_id, token=token)
    elif arch == "t5":
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        model = _from_pretrained(T5ForConditionalGeneration, hf_id, token=token)
        aux = AutoTokenizer.from_pretrained(hf_id, token=token)
    elif arch == "vit":
        from transformers import ViTForImageClassification, AutoImageProcessor
        model = _from_pretrained(ViTForImageClassification, hf_id, token=token, attn_implementation="eager")
        aux = AutoImageProcessor.from_pretrained(hf_id, token=token)
    elif arch == "dinov2":
        from transformers import Dinov2ForImageClassification, AutoImageProcessor, AutoModel
        model = _from_pretrained(Dinov2ForImageClassification, cfg["head_id"], token=token, attn_implementation="eager")
        aux = AutoImageProcessor.from_pretrained(cfg["head_id"], token=token)
        # the linear-head checkpoint must carry the survey's backbone weights: check a few tensors
        try:
            bb = _from_pretrained(AutoModel, hf_id, token=token)
            sd_h, sd_b = model.dinov2.state_dict(), bb.state_dict()
            diffs = [float((sd_h[k].float() - sd_b[k].float()).abs().max()) for k in list(sd_b)[:40] if k in sd_h]
            log(f"  DINOv2 backbone parity with {hf_id}: max |diff| over {len(diffs)} tensors = {max(diffs):.2e}")
            cfg["backbone_parity_max_abs_diff"] = max(diffs)
            del bb
        except Exception as e:
            log(f"  backbone parity check skipped: {e}")
    elif arch == "mixer":
        import timm
        model = timm.create_model(hf_id.replace("timm/", "hf-hub:timm/"), pretrained=True)
        aux = None
    elif arch == "wav2vec2":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        model = _from_pretrained(Wav2Vec2ForCTC, hf_id, token=token, attn_implementation="eager")
        model.config.ctc_loss_reduction = "mean"; model.config.ctc_zero_infinity = True
        aux = Wav2Vec2Processor.from_pretrained(hf_id, token=token)
    elif arch == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        model = _from_pretrained(WhisperForConditionalGeneration, hf_id, token=token, attn_implementation="eager")
        aux = WhisperProcessor.from_pretrained(hf_id, token=token)
        aux.tokenizer.set_prefix_tokens(language="english", task="transcribe")
        model.config.forced_decoder_ids = None
    else:
        raise ValueError(arch)
    model = model.to(DEVICE).eval().float()
    return model, aux


# ---------------------------------------------------------------- tiny models for the container plumbing test
def tiny_models():
    """Random-weight miniatures of each architecture plus synthetic batches: no network, CPU in seconds."""
    from transformers import (BertConfig, BertForMaskedLM, RobertaConfig, RobertaForMaskedLM, T5Config, T5ForConditionalGeneration,
                              ViTConfig, ViTForImageClassification, Dinov2Config, Dinov2ForImageClassification,
                              Wav2Vec2Config, Wav2Vec2ForCTC, WhisperConfig, WhisperForConditionalGeneration)
    import timm
    torch.manual_seed(0)
    out = []
    out.append(("bert", BertForMaskedLM(BertConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, vocab_size=120))))
    out.append(("roberta", RobertaForMaskedLM(RobertaConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, vocab_size=120))))
    out.append(("t5", T5ForConditionalGeneration(T5Config(d_model=32, d_ff=64, num_layers=3, num_decoder_layers=2, num_heads=2, d_kv=16, vocab_size=200, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))))
    out.append(("vit", ViTForImageClassification(ViTConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=8, num_labels=10))))
    out.append(("dinov2", Dinov2ForImageClassification(Dinov2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=8, num_labels=10))))
    from timm.models.mlp_mixer import MlpMixer
    mx = MlpMixer(num_classes=10, img_size=64, patch_size=16, num_blocks=3, embed_dim=32)
    torch.nn.init.normal_(mx.head.weight, std=0.05)      # timm zero-initialises the head; give it a random readout
    out.append(("mixer", mx))
    out.append(("wav2vec2", Wav2Vec2ForCTC(Wav2Vec2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64, vocab_size=32,
                                                           conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, ctc_loss_reduction="mean", ctc_zero_infinity=True))))
    out.append(("whisper", WhisperForConditionalGeneration(WhisperConfig(d_model=32, encoder_layers=3, decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2,
                                                                          encoder_ffn_dim=64, decoder_ffn_dim=64, vocab_size=100, num_mel_bins=8, max_source_positions=50,
                                                                          max_target_positions=20, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1))))
    return out


class TinyReadout(Readout):
    """Synthetic batches with the same evaluate() code path."""

    def __init__(self, arch, model, n_batches):
        self.cfg, self.model, self.aux, self.arch, self.n = {"arch": arch}, model, None, arch, n_batches
        self.batches, self.clean_logp = [], [None] * n_batches
        g = torch.Generator().manual_seed(1)
        for b in range(n_batches):
            if arch in TEXT_ARCHES:
                ids = torch.randint(5, 120, (4, 16), generator=g)
                labels = ids.clone(); masked = torch.rand((4, 16), generator=g) < 0.2; labels[~masked] = -100
                inp = ids.clone(); inp[masked] = 3
                self.batches.append((inp, labels, masked))
            elif arch == "t5":
                ids = torch.randint(5, 190, (4, 16), generator=g); att = torch.ones_like(ids)
                lab = torch.randint(5, 190, (4, 8), generator=g)
                self.batches.append((ids, att, lab))
            elif arch in VISION_ARCHES:
                size = 64 if arch == "mixer" else 32
                self.batches.append((torch.randn(4, 3, size, size, generator=g),))
            elif arch == "wav2vec2":
                x = torch.randn(4, 4000, generator=g); lab = torch.randint(1, 30, (4, 6), generator=g)
                self.batches.append((x, lab, torch.full((4,), 4000)))
            elif arch == "whisper":
                x = torch.randn(4, 8, 100, generator=g); lab = torch.randint(3, 100, (4, 10), generator=g)
                self.batches.append((x, lab))
        self.desc = "synthetic plumbing batches"


# ---------------------------------------------------------------- assay
def run_census(cfg, model, readout, n_batches):
    arch = cfg["arch"]
    layers = get_layers(model, arch)
    assert len(layers) == cfg["n_layers"], f"layer count {len(layers)} != {cfg['n_layers']}"
    hooks = [branch_hooks(l, arch) for l in layers]
    d_model = None
    R = {}
    for li, hl in enumerate(hooks):
        for si, (mod, path, ti, axis) in enumerate(hl):
            R[(li, si)] = None      # width resolved on first call below

    # widths: run one clean batch with recording hooks to read the branch output widths
    widths = {}
    rec = []
    for li, hl in enumerate(hooks):
        for si, (mod, path, ti, axis) in enumerate(hl):
            def mk(li=li, si=si, ti=ti, axis=axis):
                def h(m, i, o):
                    t = o[ti] if ti is not None else o
                    widths[(li, si)] = t.shape[axis]
                return h
            rec.append(mod.register_forward_hook(mk()))
    base_losses, base_kls, base_extra = readout.evaluate()
    for h in rec:
        h.remove()
    for key, w in widths.items():
        li, si = key
        R[key] = random_orthogonal(w, SEED + li * 100 + si * 10)
    d_model = widths[(0, 0)]
    log(f"  branch widths: block 0 -> {[widths[(0, s)] for s in range(len(hooks[0]))]}; hooks {[p for _, p, _, _ in hooks[0]]}")

    baseline = float(np.mean(base_losses)); base_arr = np.array(base_losses)
    log(f"  Baseline loss: {baseline:.4f}" + (f"  (self-label CE {np.mean([e['selflabel_ce'] for e in base_extra]):.4f})" if base_extra[0] else ""))

    def eval_with(target_layers, dose=DOSE):
        mgr = HookManager()
        try:
            for li in target_layers:
                for si, (mod, path, ti, axis) in enumerate(hooks[li]):
                    mgr.register(mod, R[(li, si)], dose, ti, axis)
            return readout.evaluate()
        finally:
            mgr.remove_all()

    # plumbing: dose 0 reproduces the baseline; dose 1 on block 1 (or 0) changes it
    probe = 1 if len(layers) > 1 else 0
    l0, _, _ = eval_with([probe], dose=0.0)
    plumb0 = float(abs(np.mean(l0) - baseline))
    l1, _, _ = eval_with([probe], dose=1.0)
    plumb1 = float(np.mean(l1) - baseline)
    log(f"  plumbing: dose-0 hook on block {probe} -> |dL| = {plumb0:.2e}; dose-1 -> dL = {plumb1:+.4f}")
    assert plumb0 < 1e-5 * max(1.0, abs(baseline)), "dose-0 hook changed the loss: hook is not on the forward path as expected"

    dl, ci, kl_prof, extra_prof = [], [], [], []
    for li in range(len(layers)):
        ls, kls, ex = eval_with([li])
        arr = np.array(ls)
        delta = float(arr.mean() - baseline)
        rng = np.random.RandomState(SEED + li)
        boots = [arr[idx].mean() - base_arr[idx].mean() for idx in (rng.randint(0, n_batches, n_batches) for _ in range(N_BOOTSTRAP))]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        dl.append(delta); ci.append([lo, hi]); kl_prof.append(float(np.mean(kls)))
        extra_prof.append({"selflabel_ce": float(np.mean([e["selflabel_ce"] for e in ex]))} if ex[0] else None)
        log(f"    L{li:2d}: dL={delta:+.4f} [{lo:+.4f},{hi:+.4f}]  KL={kl_prof[-1]:.4f}")

    out = {
        "hf_id": cfg["hf_id"], "arch": arch, "family": cfg.get("family"), "n_layers": len(layers), "d_model": int(d_model),
        "params_M": cfg["params_M"], "protocol": PROTOCOL, "version": VERSION,
        "baseline": baseline, "delta_L_profile": dl, "delta_L_ci": ci,
        "kl_profile": kl_prof, "baseline_kl": 0.0,
        "readout": readout.desc,
        "kl_readout": "KL(clean || scrambled) over the model's own output distribution (masked/target positions for text, valid frames for CTC, all classes for vision)",
        "hooks": [p for _, p, _, _ in hooks[0]], "branch_widths_block0": [widths[(0, s)] for s in range(len(hooks[0]))],
        "dose": DOSE, "seed": SEED, "n_batches": n_batches,
        "batch_size": AUDIO_BATCH_SIZE if arch in AUDIO_ARCHES else EVAL_BATCH_SIZE,
        "plumbing_dose0_abs_dL": plumb0, "plumbing_dose1_dL_block": probe, "plumbing_dose1_dL": plumb1,
        "sigma1_source": "survey_sigma1_v2 (incontext-natural-block-float32); not recomputed here",
    }
    if extra_prof[0]:
        out["vision_extra"] = extra_prof
        out["baseline_selflabel_ce"] = float(np.mean([e["selflabel_ce"] for e in base_extra]))
    if "backbone_parity_max_abs_diff" in cfg:
        out["backbone_parity_max_abs_diff"] = cfg["backbone_parity_max_abs_diff"]
    return out


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def provenance():
    import transformers
    p = {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
         "date": datetime.datetime.now().isoformat(timespec="seconds"), "device": DEVICE}
    try:
        import timm; p["timm"] = timm.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p["gpu"] = torch.cuda.get_device_name()
    return p


def summary(out_dir):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("perlayer_") and fn.endswith(".json"):
            d = json.load(open(os.path.join(out_dir, fn)))
            dL = np.array(d["delta_L_profile"]); L = len(dL)
            inner = dL[1:L - 1]; n = len(inner); t = n // 3
            e, w, l = inner[:t].mean(), inner[t:n - t].mean(), inner[n - t:].mean()
            we = w / max(e, l) if max(e, l) > 0 else float("nan")
            rows.append((d["hf_id"], d["arch"], L, d["baseline"], e, w, l, we, dL[0], dL[-1], d.get("plumbing_dose0_abs_dL"), d.get("quick", False)))
    log(f"\n{'model':44s} {'L':>3s} {'base':>8s} {'early':>8s} {'waist':>8s} {'late':>8s} {'w/e':>6s} {'blk0':>8s} {'last':>8s} {'plumb0':>8s}")
    for r in rows:
        log(f"{r[0]:44s} {r[2]:3d} {r[3]:8.4f} {r[4]:8.4f} {r[5]:8.4f} {r[6]:8.4f} {r[7]:6.2f} {r[8]:8.4f} {r[9]:8.4f} {r[10]:8.1e}" + ("  QUICK" if r[11] else ""))
    return rows


def export(out_dir):
    stage = os.path.join(os.path.dirname(out_dir.rstrip("/")), "census_v4_export")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    n = 0
    for fn in os.listdir(out_dir):
        if fn.endswith(".json") or fn.endswith(".txt"):
            shutil.copy(os.path.join(out_dir, fn), stage); n += 1
    z1 = shutil.make_archive(os.path.join(os.path.dirname(out_dir.rstrip("/")), "census_v4_results"), "zip", stage)
    z2 = shutil.make_archive(os.path.join(os.getcwd(), "census_v4_results"), "zip", stage)
    for z in (z1, z2):
        log(f"export: {n} files -> {z} ({os.path.getsize(z) / 1e3:.0f} kB)")
    return z1, z2


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("CENSUS_OUT", "results/census_v4"))
    ap.add_argument("--data", default=os.environ.get("CENSUS_DATA", "data"))
    ap.add_argument("--models", nargs="*", default=(os.environ.get("CENSUS_MODELS") or "").split() or None)
    ap.add_argument("--quick", action="store_true", default=bool(os.environ.get("CENSUS_QUICK")))
    ap.add_argument("--plumbing", action="store_true", help="tiny random models, synthetic data, no network")
    ap.add_argument("--no-skip", action="store_true")
    args, _ = ap.parse_known_args()
    _resolve_hf_token()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.data, exist_ok=True)
    n_batches = 2 if (args.quick or args.plumbing) else N_EVAL_BATCHES
    suffix = "_quick" if args.quick else ("_plumbing" if args.plumbing else "")
    prov = provenance()
    log(f"census v4 | {prov} | out={args.out} | batches={n_batches}")
    if torch.cuda.is_available():
        torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")

    if args.plumbing:
        todo = [({"hf_id": f"tiny/{a}", "arch": a, "n_layers": 3, "params_M": 0, "family": "tiny"}, m) for a, m in tiny_models()]
    else:
        todo = [(c, None) for c in MODELS if not args.models or c["hf_id"] in args.models]
    for cfg, tiny in todo:
        fpath = os.path.join(args.out, f"perlayer_{cfg['hf_id'].replace('/', '_').lower()}{suffix}.json")
        if os.path.exists(fpath) and not args.no_skip:
            try:
                d = json.load(open(fpath))
                if d.get("protocol") == PROTOCOL and len(d.get("delta_L_profile", [])) == cfg["n_layers"]:
                    log(f"[SKIP] {cfg['hf_id']}: done"); continue
            except Exception:
                pass
        log("\n" + "-" * 78 + f"\n{cfg['hf_id']} ({cfg['arch']}, {cfg['n_layers']} blocks)\n" + "-" * 78)
        t0 = time.time(); model = None
        try:
            if tiny is not None:
                model = tiny.to(DEVICE).eval().float()
                readout = TinyReadout(cfg["arch"], model, n_batches)
            else:
                model, aux = load_model(cfg)
                readout = Readout(cfg, model, aux, n_batches, args.data, os.path.join(args.out, "..", "data_cache"))
            res = run_census(cfg, model, readout, n_batches)
            res.update({"quick": bool(args.quick or args.plumbing), "elapsed_seconds": time.time() - t0, **{f"prov_{k}": v for k, v in prov.items()}})
            write_json(fpath, res)
            log(f"  saved {fpath} ({time.time() - t0:.0f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"  FAILED {cfg['hf_id']}: {e}")
        finally:
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary(args.out)
    export(args.out)


if __name__ == "__main__":
    main()
