| Model | Family | Blocks | Readout | Clean loss | R~ex0~ | Two-arm | ΔL early | ΔL middle | ΔL late | Middle/edge | ΔL block 0 | ΔL last block | Interior-sensitive | Cell | Excluded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pythia-70M | AR decoder | 6 | next-token CE | 4.54 | 0.42 | – | 3.383 | 4.296 | 3.651 | 1.18 | 8.02 | 23.22 | yes | sensitivity | – |
| GPT-2 124M | AR decoder | 12 | next-token CE | 4.03 | 0.71 | positive | 2.239 | 0.294 | 0.519 | 0.13 | 6.35 | 6.20 | yes | both | – |
| Pythia-160M | AR decoder | 12 | next-token CE | 3.99 | 0.76 | – | 3.022 | 1.345 | 1.768 | 0.45 | 6.68 | 12.01 | yes | sensitivity | – |
| Pythia-410M | AR decoder | 24 | next-token CE | 3.49 | 0.84 | – | 1.843 | 0.377 | 0.571 | 0.20 | 7.49 | 1.31 | yes | sensitivity | – |
| Qwen2-0.5B | AR decoder | 24 | next-token CE | 3.15 | 0.86 | – | 1.885 | 0.260 | 0.605 | 0.14 | 7.34 | 3.07 | yes | sensitivity | – |
| Qwen2.5-0.5B | AR decoder | 24 | next-token CE | 3.18 | 0.88 | – | 2.408 | 0.263 | 0.729 | 0.11 | 7.07 | 5.20 | yes | sensitivity | – |
| GPT-2 Large | AR decoder | 36 | next-token CE | 3.58 | 0.92 | – | 0.151 | 0.054 | 0.082 | 0.35 | 6.36 | 0.07 | yes | sensitivity | – |
| Pythia-1B | AR decoder | 16 | next-token CE | 3.30 | 0.81 | – | 1.724 | 0.559 | 0.689 | 0.32 | 5.80 | 1.14 | yes | sensitivity | – |
| Llama-3.2-1B | AR decoder | 16 | next-token CE | 2.99 | 0.82 | – | 2.640 | 1.038 | 0.975 | 0.39 | 7.85 | 3.75 | yes | sensitivity | – |
| Pythia-1.4B | AR decoder | 24 | next-token CE | 3.19 | 0.70 | positive | 1.469 | 0.309 | 0.321 | 0.21 | 6.76 | 1.50 | yes | both | – |
| Qwen2.5-1.5B | AR decoder | 28 | next-token CE | 2.89 | 0.86 | – | 1.282 | 0.149 | 0.531 | 0.12 | 8.44 | 1.64 | yes | sensitivity | – |
| GPT-2 XL | AR decoder | 48 | next-token CE | 3.45 | 0.92 | – | 0.091 | 0.032 | 0.055 | 0.35 | 8.62 | 0.02 | yes | sensitivity | – |
| Gemma-2-2B | AR decoder | 26 | next-token CE | 2.97 | 0.96 | – | 1.304 | 0.424 | 0.747 | 0.33 | 3.56 | 1.01 | yes | sensitivity | – |
| Phi-2 | AR decoder | 32 | next-token CE | 3.05 | 0.80 | – | 0.473 | 0.038 | 0.089 | 0.08 | 7.08 | 0.27 | no | neither | – |
| Pythia-2.8B | AR decoder | 32 | next-token CE | 3.05 | 0.78 | positive | 1.014 | 0.202 | 0.252 | 0.20 | 7.31 | 0.96 | yes | both | – |
| Qwen2.5-3B | AR decoder | 36 | next-token CE | 2.76 | 0.92 | – | 1.022 | 0.117 | 0.401 | 0.11 | 7.44 | 1.54 | yes | sensitivity | – |
| Llama-3.2-3B | AR decoder | 28 | next-token CE | 2.74 | 0.97 | – | 1.134 | 0.290 | 0.308 | 0.26 | 6.92 | 1.60 | yes | sensitivity | – |
| Pythia-6.9B | AR decoder | 32 | next-token CE | 2.93 | 0.80 | positive | 1.424 | 0.301 | 0.095 | 0.21 | 6.81 | 1.17 | yes | both | – |
| Mistral-7B-Instruct | AR decoder | 32 | next-token CE | 2.65 | 0.92 | – | 0.695 | 0.108 | 0.130 | 0.15 | 8.15 | 0.33 | yes | sensitivity | – |
| Mistral-7B | AR decoder | 32 | next-token CE | 2.40 | 0.92 | – | 0.711 | 0.128 | 0.148 | 0.18 | 6.96 | 0.51 | yes | sensitivity | – |
| Qwen2.5-7B | AR decoder | 28 | next-token CE | 2.61 | 0.75 | positive | 0.492 | 0.133 | 0.347 | 0.27 | 10.14 | 5.20 | yes | both | – |
| Pythia-12B | AR decoder | 36 | next-token CE | 2.86 | 0.93 | – | 1.500 | 0.280 | 0.084 | 0.19 | 7.28 | 0.77 | yes | sensitivity | – |
| Qwen2.5-14B | AR decoder | 48 | next-token CE | 2.45 | 0.97 | – | 0.360 | 0.054 | 0.173 | 0.15 | 9.23 | 0.98 | yes | sensitivity | – |
| Mamba-130M | State-space | 24 | next-token CE | 3.68 | 0.51 | positive | 0.279 | 0.186 | 0.765 | 0.24 | 5.16 | 21.91 | yes | both | – |
| Mamba-370M | State-space | 48 | next-token CE | 3.31 | 0.75 | positive | 0.085 | 0.067 | 0.256 | 0.26 | 4.06 | 8.57 | yes | both | – |
| Mamba-1.4B | State-space | 48 | next-token CE | 3.05 | 0.91 | – | 0.050 | 0.055 | 0.150 | 0.37 | 2.05 | 2.59 | yes | sensitivity | – |
| Mamba-2.8B | State-space | 64 | next-token CE | 2.93 | 0.92 | – | 0.042 | 0.038 | 0.094 | 0.40 | 0.31 | 2.83 | yes | sensitivity | – |
| ConvNeXt-tiny | ConvNet | 18 | KL, ImageNet head | 0 (KL) | 0.35 | positive | 1.142 | 0.549 | 0.955 | 0.48 | 2.60 | 1.07 | yes | both | – |
| ConvNeXt-small | ConvNet | 36 | KL, ImageNet head | 0 (KL) | 0.58 | positive | 0.449 | 0.104 | 0.100 | 0.23 | 2.20 | 0.17 | yes | both | – |
| ConvNeXt-base | ConvNet | 36 | KL, ImageNet head | 0 (KL) | 0.49 | positive | 0.360 | 0.100 | 0.056 | 0.28 | 3.25 | 0.06 | yes | both | – |
| ConvNeXt-large | ConvNet | 36 | KL, ImageNet head | 0 (KL) | 0.52 | positive | 0.289 | 0.067 | 0.033 | 0.23 | 2.75 | 0.02 | yes | both | – |
| MLP-Mixer-B | MLP-Mixer | 12 | KL, ImageNet head | 0 (KL) | 0.55 | positive | 4.483 | 3.992 | 2.446 | 0.89 | 4.67 | 1.61 | yes | both | – |
| MLP-Mixer-L | MLP-Mixer | 24 | KL, ImageNet head | 0 (KL) | 0.75 | positive | 2.955 | 2.107 | 2.485 | 0.71 | 5.19 | 1.38 | yes | both | – |
| ViT-base | ViT (supervised) | 12 | KL, ImageNet head | 0 (KL) | 0.81 | – | 2.468 | 1.673 | 1.420 | 0.68 | 4.80 | 2.44 | yes | sensitivity | – |
| DINOv2-base | ViT (self-supervised) | 12 | KL, ImageNet head | 0 (KL) | 0.77 | positive | 4.678 | 6.531 | 5.667 | 1.15 | 6.07 | 5.19 | yes | both | – |
| BERT-base | Masked encoder | 12 | masked-token CE | 2.26 | 0.97 | – | 0.471 | 0.488 | 0.522 | 0.93 | 0.57 | 0.33 | yes | sensitivity | – |
| RoBERTa-base | Masked encoder | 12 | masked-token CE | 2.00 | 1.13 | – | 0.720 | 0.450 | 0.512 | 0.63 | 1.95 | 6.85 | yes | sensitivity | – |
| BERT-large | Masked encoder | 24 | masked-token CE | 2.09 | 0.99 | – | 0.429 | 0.235 | 0.563 | 0.42 | 0.07 | 0.28 | yes | sensitivity | – |
| RoBERTa-large | Masked encoder | 24 | masked-token CE | 1.98 | 1.34 | – | 0.583 | 0.475 | 1.024 | 0.46 | 6.81 | 4.83 | yes | sensitivity | – |
| T5-small (encoder) | T5 encoder | 6 | span-corruption CE | 3.18 | 1.09 | – | 0.209 | 0.136 | 0.295 | 0.46 | 5.50 | 0.99 | yes | sensitivity | – |
| T5-base (encoder) | T5 encoder | 12 | span-corruption CE | 2.70 | 1.01 | – | 0.046 | 0.031 | 0.075 | 0.41 | 2.72 | 0.18 | yes | sensitivity | – |
| T5-large (encoder) | T5 encoder | 24 | span-corruption CE | 2.39 | 0.97 | – | 0.008 | 0.012 | 0.018 | 0.63 | 3.30 | -0.02 | yes | sensitivity | – |
| Flan-T5-large (encoder) | T5 encoder | 24 | span-corruption CE | 13.64 | 0.22 | – | 2.050 | -0.216 | -0.163 | -0.11 | 0.28 | 0.04 | – | excluded | 3 |
| wav2vec2-base | Audio encoder | 12 | CTC loss | 0.60 | 0.93 | – | 1.742 | 1.454 | 2.976 | 0.49 | 3.53 | 8.89 | yes | sensitivity | – |
| Whisper-small | Audio encoder | 12 | decoder CE | 0.85 | 0.86 | – | -0.000 | 5.284 | 0.905 | 5.84 | 7.06 | 3.51 | yes | sensitivity | – |

ΔL, increase of the model's own loss in nats (next-token, masked-token, span-corruption, CTC or teacher-forced decoder cross-entropy) or the KL divergence from the clean logits over the ImageNet-1k head (vision models) when the branch outputs of a block are rotated by a Haar-random orthogonal matrix, dose 1, one block at a time; regional means over thirds of blocks 1 to L − 2, with block 0 and the last block reported separately. Middle/edge, middle mean divided by the larger edge-third mean. Interior-sensitive: middle mean > 0.01 nats and > 10% of the larger edge-third mean. Cell crosses the two-arm geometry flag with the interior-sensitivity flag (both, geometry only, sensitivity only, neither); the assignment under the earlier all-blocks sensitivity rule, which took thirds over every block including block 0, is in ed_table2.csv. Excluded: 1, mean interior ΔL below −0.01; 2, middle/edge undefined because no edge third is positive; 3, clean cross-entropy above the uniform-prediction baseline ln V (Flan-T5-large, 13.6 nats against 10.4). The 14 non-decoder models were re-run on 20 September 2026 with branch-level hooks and the readouts listed (Methods), the two audio models again on 21 September on utterances of at most 10 s; the other 31 files are from the April and May runs, whose hooks and readouts were as described.
