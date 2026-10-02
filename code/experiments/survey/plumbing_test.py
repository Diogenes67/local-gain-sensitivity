"""
Plumbing test for scramble_census_april_panel.py without Hub access.

Builds a tiny randomly initialised model of every architecture in the April
completion list from its config class, then runs the sigma1 profiler and the
per-layer scrambling assay on random inputs. Checks that hooks fire, the
captured-kwargs layer replay works for each architecture, rotations preserve
norm, and delta-L is finite. Numbers are meaningless; only the code path is
being tested.
"""
import sys, os, json, types
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scramble_census_april_panel as S

from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM,
                          Gemma2Config, Gemma2ForCausalLM, PhiConfig, PhiForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM, MambaConfig, MambaForCausalLM,
                          ConvNextConfig, ConvNextForImageClassification)

torch.manual_seed(0)
V = 512
common = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)

builders = {
    "gpt2":     lambda: GPT2LMHeadModel(GPT2Config(vocab_size=V, n_embd=64, n_layer=4, n_head=4, n_positions=256)),
    "llama":    lambda: LlamaForCausalLM(LlamaConfig(**common)),
    "gemma2":   lambda: Gemma2ForCausalLM(Gemma2Config(**common, head_dim=16, sliding_window=64)),
    "phi":      lambda: PhiForCausalLM(PhiConfig(vocab_size=V, hidden_size=64, intermediate_size=128,
                                                  num_hidden_layers=4, num_attention_heads=4, max_position_embeddings=256)),
    "qwen":     lambda: Qwen2ForCausalLM(Qwen2Config(**common)),
    "mamba":    lambda: MambaForCausalLM(MambaConfig(vocab_size=V, hidden_size=64, num_hidden_layers=4, state_size=8)),
    "convnext": lambda: ConvNextForImageClassification(ConvNextConfig(num_channels=3, hidden_sizes=[16, 32, 64, 128],
                                                                       depths=[1, 1, 2, 1], num_labels=10)),
}

S.N_BOOTSTRAP = 20
S.PROBE_DIM = 4
S.PI_ITERS = 3
n_eval = 2
ok = True
for arch, build in builders.items():
    print(f"\n=== {arch}")
    model = build().float().eval()
    if arch != "convnext" and arch != "mamba":
        model.config._attn_implementation = "eager"
    layers = S.get_layers(model, arch)
    d_model = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if arch == "convnext":
        eval_data = torch.randn(n_eval * S.EVAL_BATCH_SIZE, 3, 64, 64)
        s1 = S.compute_sigma1_profile(model, layers, arch, None, 2, image_shape=(3, 64, 64))
    else:
        eval_data = torch.randint(0, V, (n_eval * S.EVAL_BATCH_SIZE, S.SEQ_LEN))
        s1 = S.compute_sigma1_profile(model, layers, arch, V, 2)
    assert len(s1) == len(layers) and all(np.isfinite(s1)) and all(v != 1.0 for v in s1), f"sigma1 replay failed for {arch}: {s1}"
    # rotation preserves norm
    R = S.random_orthogonal(d_model if arch != "convnext" else 16, 1)
    x = torch.randn(2, 5, R.shape[0]) if arch != "convnext" else torch.randn(2, 16, 4, 4)
    y = S.rotate(x, R, 1.0)
    assert torch.allclose(x.flatten(1).norm(dim=1), y.flatten(1).norm(dim=1), atol=1e-4), "rotation not norm-preserving"
    res = S.run_perlayer_scrambling(model, layers, arch, d_model, eval_data, s1, n_eval)
    assert len(res["delta_L_profile"]) == len(layers) and all(np.isfinite(res["delta_L_profile"])), arch
    print(f"  R_ex0 vertex={S.compute_R_ex0(s1):.3f} thirds={S.compute_R_ex0_thirds(s1):.3f}; dL={['%.3f' % v for v in res['delta_L_profile']]}")
    json.dumps(res)
print("\nALL ARCHITECTURES PASSED" if ok else "FAILED")
