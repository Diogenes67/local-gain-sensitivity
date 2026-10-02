"""
Plumbing and correctness test for survey_sigma1_v2.py without Hub access.

Builds a tiny randomly initialised model of every architecture in the registry, runs
the canonical protocol (in-context, token-local) plus the branch, fullseq and isolated
variants where defined, and checks:
  1. every architecture's capture-and-replay path runs and gives finite profiles
  2. the power-iteration sigma_1 matches the exact SVD of the materialised token-local
     Jacobian (relative error < 1e-3) on several architectures
  3. batched positions give the same sigma_1 as one position at a time (broadcasting of
     captured kwargs across the batch is sound)
  4. the JVP used is exact AD (torch.func.jvp or double-backward), not finite differences
Numbers are meaningless; only the code path and the estimator are being tested.
"""
import os, sys, json, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_sigma1_v2 as S

from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM,
                          LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM, Gemma2Config, Gemma2ForCausalLM,
                          PhiConfig, PhiForCausalLM, MambaConfig, MambaForCausalLM,
                          ConvNextConfig, ConvNextForImageClassification,
                          BertConfig, BertModel, RobertaConfig, RobertaModel,
                          T5Config, T5EncoderModel, ViTConfig, ViTModel, Dinov2Config, Dinov2Model,
                          WhisperConfig, WhisperModel, Wav2Vec2Config, Wav2Vec2Model)

torch.manual_seed(0)
V = 512
T = 24
common = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)


def eager(cfg):
    cfg._attn_implementation = "eager"
    return cfg


def text_forward(m, inp):
    return m(input_ids=inp["input_ids"], use_cache=False)


def enc_forward(m, inp):
    return m(input_ids=inp["input_ids"])


builders = {
    "gpt2":     lambda: (GPT2LMHeadModel(eager(GPT2Config(vocab_size=V, n_embd=64, n_layer=4, n_head=4, n_positions=256))), text_forward),
    "pythia":   lambda: (GPTNeoXForCausalLM(eager(GPTNeoXConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                                               num_attention_heads=4, max_position_embeddings=256, rotary_pct=0.25))), text_forward),
    "llama":    lambda: (LlamaForCausalLM(eager(LlamaConfig(**common))), text_forward),
    "mistral":  lambda: (MistralForCausalLM(eager(MistralConfig(**common, sliding_window=16))), text_forward),
    "qwen":     lambda: (Qwen2ForCausalLM(eager(Qwen2Config(**common))), text_forward),
    "gemma2":   lambda: (Gemma2ForCausalLM(eager(Gemma2Config(**common, head_dim=16, sliding_window=16))), text_forward),
    "phi":      lambda: (PhiForCausalLM(eager(PhiConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                                       num_attention_heads=4, max_position_embeddings=256))), text_forward),
    "mamba":    lambda: (MambaForCausalLM(MambaConfig(vocab_size=V, hidden_size=64, num_hidden_layers=4, state_size=8)), text_forward),
    "bert":     lambda: (BertModel(eager(BertConfig(vocab_size=V, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                                                    intermediate_size=128, max_position_embeddings=256))), enc_forward),
    "roberta":  lambda: (RobertaModel(eager(RobertaConfig(vocab_size=V, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                                                          intermediate_size=128, max_position_embeddings=256 + 2))), enc_forward),
    "t5":       lambda: (T5EncoderModel(T5Config(vocab_size=V, d_model=64, d_kv=16, d_ff=128, num_layers=4, num_heads=4)), enc_forward),
    "vit":      lambda: (ViTModel(eager(ViTConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128,
                                                  image_size=32, patch_size=8))), lambda m, inp: m(pixel_values=inp["pixel_values"])),
    "dinov2":   lambda: (Dinov2Model(eager(Dinov2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128,
                                                        image_size=32, patch_size=8))), lambda m, inp: m(pixel_values=inp["pixel_values"])),
    "convnext": lambda: (ConvNextForImageClassification(ConvNextConfig(num_channels=3, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 2, 1],
                                                                       num_labels=10, layer_scale_init_value=1.0)), lambda m, inp: m(pixel_values=inp["pixel_values"])),
    "whisper":  lambda: (WhisperModel(eager(WhisperConfig(vocab_size=V, d_model=64, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128,
                                                          decoder_layers=1, decoder_attention_heads=4, decoder_ffn_dim=128, num_mel_bins=80,
                                                          max_source_positions=50, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                                          decoder_start_token_id=1))).encoder,
                         lambda m, inp: m(input_features=inp["input_features"])),
    "wav2vec2": lambda: (Wav2Vec2Model(eager(Wav2Vec2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
                                                            conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4))),
                         lambda m, inp: m(input_values=inp["input_values"])),
}
try:
    import timm

    def _mixer():
        from timm.models.mlp_mixer import MlpMixer
        m = MlpMixer(img_size=32, patch_size=8, num_blocks=2, embed_dim=32, num_classes=10)
        return m, (lambda mm, inp: mm(inp["pixel_values"]))
    builders["mlp_mixer"] = _mixer
except Exception as e:
    print("timm not available, MLP-Mixer path not tested:", e)


def make_inputs(arch, n=2):
    g = torch.Generator().manual_seed(1)
    out = []
    for _ in range(n):
        if arch in S.TEXT_ARCHES:
            out.append({"input_ids": torch.randint(0, V, (1, T), generator=g)})
        elif arch == "convnext":
            out.append({"pixel_values": torch.randn(1, 3, 64, 64, generator=g)})
        elif arch in ("vit", "dinov2", "mlp_mixer"):
            out.append({"pixel_values": torch.randn(1, 3, 32, 32, generator=g)})
        elif arch == "whisper":
            out.append({"input_features": torch.randn(1, 80, 100, generator=g), "valid_T": 30})
        elif arch == "wav2vec2":
            out.append({"input_values": torch.randn(1, 8000, generator=g)})
    return out


class A:
    n_inputs, n_positions, restarts, iters = 2, 3, 2, 30
    fd_eps, force_fd, estimator = 1e-3, False, "sigma1"


class ARho(A):
    estimator = "rho"


def bundle_for(arch):
    model, fwd = builders[arch]()
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cls_offset = 1 if arch in ("vit", "dinov2") else 0
    kind = "text" if arch in S.TEXT_ARCHES else ("image" if arch in S.IMAGE_ARCHES else "audio")
    return dict(model=model, layers=S.get_layers(model, arch), arch=arch, hf_id=f"tiny/{arch}", kind=kind,
                cfg=dict(label=arch, family="test", params_M=0, april=False), dtype=torch.float32,
                forward=fwd, cls_offset=cls_offset)


def exact_check(b, inputs, li=1):
    """sigma_1 by materialised Jacobian + SVD vs the power iteration, one position."""
    captured = S.capture_layer_inputs(b, inputs[0])
    hidden0, rest, kw = captured[li]
    layout = S.Layout(hidden0)
    positions = S.choose_positions(layout, 3, b.get("cls_offset", 0), inputs[0].get("valid_T"))
    t = positions[1]
    call = S.make_layer_call(b["layers"][li], rest, kw)
    f = S.make_f_incontext(call, hidden0, layout, [t], False, torch.float32)
    x0 = layout.get(hidden0.float(), [t])
    J = torch.autograd.functional.jacobian(f, x0).reshape(x0.shape[1], x0.shape[1])
    s_exact = torch.linalg.svdvals(J)[0].item()
    st = {}
    s_pi, n_it = S.sigma1_power_iteration(f, x0, 3, 200, 1e-10, st)
    rel = abs(s_pi.item() - s_exact) / s_exact
    print(f"  exact sigma_1 at block {li} pos {t}: SVD {s_exact:.6f}, power iteration {s_pi.item():.6f} "
          f"({n_it} iters, rel err {rel:.1e}, jvp={st.get('name')})")
    assert rel < 1e-3, f"estimator disagrees with exact SVD for {b['arch']}: {rel}"
    assert st.get("name") != "finite-difference", f"{b['arch']} fell back to finite differences"
    return rel


def batched_vs_loop_check(b, inputs, li=1):
    """Batched positions against the exact SVD of each position's materialised Jacobian,
    which also checks that captured kwargs broadcast correctly across the batch."""
    captured = S.capture_layer_inputs(b, inputs[0])
    hidden0, rest, kw = captured[li]
    layout = S.Layout(hidden0)
    positions = S.choose_positions(layout, 4, b.get("cls_offset", 0), inputs[0].get("valid_T"))
    call = S.make_layer_call(b["layers"][li], rest, kw)
    x0 = layout.get(hidden0.float(), positions)
    fb = S.make_f_incontext(call, hidden0, layout, positions, False, torch.float32)
    sb, n_it = S.sigma1_power_iteration(fb, x0, 3, 400, 1e-10, {})
    exact = []
    for j, t in enumerate(positions):
        f1 = S.make_f_incontext(call, hidden0, layout, [t], False, torch.float32)
        J = torch.autograd.functional.jacobian(f1, x0[j:j + 1]).reshape(x0.shape[1], x0.shape[1])
        exact.append(torch.linalg.svdvals(J)[0].item())
    exact = torch.tensor(exact)
    rel = ((sb - exact).abs() / exact).max().item()
    print(f"  batched positions {positions} vs exact SVD: max rel err {rel:.1e} ({n_it} iters)")
    assert rel < 2e-3, f"batched positions disagree with exact SVD for {b['arch']}: {rel}"


ok = True
t_all = time.time()
for arch in builders:
    print(f"\n=== {arch}")
    try:
        b = bundle_for(arch)
        inputs = make_inputs(arch)
        # canonical
        res = S.profile_model(b, "incontext", "random", False, A, inputs=inputs)
        prof = res["sigma1_profile"]
        assert len(prof) == len(b["layers"]) and all(np.isfinite(prof)), f"bad profile {prof}"
        assert res["jvp_method"] in ("torch.func.jvp", "double-backward"), f"jvp method {res['jvp_method']}"
        print(f"  canonical: {len(prof)} blocks, sigma1 {['%.3g' % v for v in prof]}, R_ex0={res['R_ex0_thirds']:.3f}, "
              f"uptick={res['late_uptick']:.2f}, jvp={res['jvp_method']}, conv={res['converged_frac']:.2f}, {res['time_s']:.1f}s")
        json.dumps(res)
        # branch
        if arch in S.BRANCH_OK:
            rb = S.profile_model(b, "incontext", "random", True, A, inputs=inputs)
            assert all(np.isfinite(rb["sigma1_profile"]))
            print(f"  branch:    sigma1 {['%.3g' % v for v in rb['sigma1_profile']]}")
        # spectral radius: must be finite and no larger than sigma_1 (rho <= sigma_1 for any square J)
        rr = S.profile_model(b, "incontext", "random", False, ARho, inputs=inputs[:1])
        assert all(np.isfinite(rr["sigma1_profile"])) and rr["protocol"].endswith("-rho")
        assert all(rr["sigma1_profile"][i] <= 1.05 * prof[i] for i in range(len(prof))), "rho exceeds sigma_1"
        print(f"  rho:       {['%.3g' % v for v in rr['sigma1_profile']]}")
        # fullseq
        rf = S.profile_model(b, "fullseq", "random", False, A, inputs=inputs[:1])
        assert all(np.isfinite(rf["sigma1_profile"]))
        assert all(rf["sigma1_profile"][i] >= 0.999 * prof[i] for i in range(len(prof))), \
            "whole-sequence sigma_1 should be >= token-local sigma_1"
        print(f"  fullseq:   sigma1 {['%.3g' % v for v in rf['sigma1_profile']]}")
        # isolated
        if arch in S.TEXT_ARCHES:
            ri = S.profile_model(b, "isolated", "random", False, A, inputs=inputs[:1])
            assert all(np.isfinite(ri["sigma1_profile"]))
            print(f"  isolated:  sigma1 {['%.3g' % v for v in ri['sigma1_profile']]}")
        # exact and batching checks
        exact_check(b, inputs)
        batched_vs_loop_check(b, inputs)
    except Exception as e:
        ok = False
        import traceback
        traceback.print_exc()
        print(f"  FAILED {arch}: {type(e).__name__}: {e}")

# io: save one result the way run_one does and build the summary tables
try:
    import tempfile, shutil
    d = tempfile.mkdtemp()
    b = bundle_for("gpt2")
    res = S.profile_model(b, "incontext", "random", False, A, inputs=make_inputs("gpt2"))
    res["in_april_panel"] = True
    tag = S.protocol_tag("incontext", "random", False, "float32")
    path = S.result_path(d, tag, "gpt2")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    S.write_summary(d, tag)
    assert os.path.exists(os.path.join(d, tag, "survey_summary.csv"))
    print("  io/summary path ok:", sorted(os.listdir(os.path.join(d, tag))))
    shutil.rmtree(d)
    a = S.parse_args(["-f", "kernel.json", "--quick"])
    assert a.n_positions == 2 and a.iters == 6, a
    print("  argparse ignores -f kernel.json; --quick applied")
except Exception as e:
    ok = False
    import traceback; traceback.print_exc()

print(f"\n{'ALL ARCHITECTURES PASSED' if ok else 'FAILED'} ({time.time() - t_all:.0f} s)")
