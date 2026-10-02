| Model | Family | Blocks | Params (M) | R~ex0~ | 95% CI | Early/middle | Late/middle | Two-arm | Pooled-edge |
|---|---|---|---|---|---|---|---|---|---|
| Pythia-70M | AR decoder | 6 | 70 | 0.42 | 0.40‑0.44 | 0.96 | 3.82 | – | positive |
| GPT-2 124M | AR decoder | 12 | 124 | 0.71 | 0.68‑0.75 | 1.32 | 1.49 | positive | positive |
| Pythia-160M | AR decoder | 12 | 160 | 0.76 | 0.71‑0.81 | 0.75 | 1.89 | – | positive |
| Pythia-410M | AR decoder | 24 | 410 | 0.84 | 0.81‑0.87 | 1.08 | 1.30 | – | – |
| Qwen2-0.5B | AR decoder | 24 | 494 | 0.86 | 0.81‑0.90 | 1.31 | 1.02 | – | – |
| Qwen2.5-0.5B | AR decoder | 24 | 494 | 0.88 | 0.85‑0.92 | 1.27 | 1.00 | – | – |
| GPT-2 Large | AR decoder | 36 | 774 | 0.92 | 0.89‑0.94 | 1.10 | 1.09 | – | – |
| Pythia-1B | AR decoder | 16 | 1,000 | 0.81 | 0.79‑0.82 | 1.17 | 1.31 | – | – |
| Llama-3.2-1B | AR decoder | 16 | 1,240 | 0.82 | 0.81‑0.83 | 1.36 | 1.10 | – | – |
| Pythia-1.4B | AR decoder | 24 | 1,400 | 0.70 | 0.69‑0.72 | 1.09 | 1.75 | positive | positive |
| Qwen2.5-1.5B | AR decoder | 28 | 1,540 | 0.86 | 0.82‑0.89 | 1.30 | 1.04 | – | – |
| GPT-2 XL | AR decoder | 48 | 1,558 | 0.92 | 0.89‑0.95 | 1.07 | 1.11 | – | – |
| Gemma-2-2B | AR decoder | 26 | 2,610 | 0.96 | 0.91‑1.01 | 1.03 | 1.05 | – | – |
| Phi-2 | AR decoder | 32 | 2,700 | 0.80 | 0.78‑0.82 | 1.27 | 1.22 | – | – |
| Pythia-2.8B | AR decoder | 32 | 2,800 | 0.78 | 0.76‑0.81 | 1.14 | 1.41 | positive | positive |
| Qwen2.5-3B | AR decoder | 36 | 3,090 | 0.92 | 0.88‑0.95 | 1.20 | 0.99 | – | – |
| Llama-3.2-3B | AR decoder | 28 | 3,210 | 0.97 | 0.94‑1.00 | 1.25 | 0.81 | – | – |
| Pythia-6.9B | AR decoder | 32 | 6,900 | 0.80 | 0.77‑0.83 | 1.11 | 1.39 | positive | positive |
| Mistral-7B-Instruct | AR decoder | 32 | 7,200 | 0.92 | 0.91‑0.94 | 1.31 | 0.85 | – | – |
| Mistral-7B | AR decoder | 32 | 7,200 | 0.92 | 0.90‑0.93 | 1.35 | 0.83 | – | – |
| Qwen2.5-7B | AR decoder | 28 | 7,610 | 0.75 | 0.71‑0.78 | 1.37 | 1.28 | positive | positive |
| Pythia-12B | AR decoder | 36 | 12,000 | 0.93 | 0.91‑0.96 | 0.98 | 1.16 | – | – |
| Qwen2.5-14B | AR decoder | 48 | 14,700 | 0.97 | 0.92‑1.02 | 1.09 | 0.97 | – | – |
| Mamba-130M | State-space | 24 | 130 | 0.51 | 0.50‑0.53 | 1.27 | 2.62 | positive | positive |
| Mamba-370M | State-space | 48 | 370 | 0.75 | 0.72‑0.78 | 1.12 | 1.55 | positive | positive |
| Mamba-1.4B | State-space | 48 | 1,400 | 0.91 | 0.87‑0.94 | 0.97 | 1.23 | – | – |
| Mamba-2.8B | State-space | 64 | 2,800 | 0.92 | 0.90‑0.94 | 0.97 | 1.20 | – | – |
| ConvNeXt-tiny | ConvNet | 18 | 28 | 0.35 | 0.33‑0.37 | 1.82 | 3.88 | positive | positive |
| ConvNeXt-small | ConvNet | 36 | 50 | 0.58 | 0.54‑0.62 | 1.59 | 1.86 | positive | positive |
| ConvNeXt-base | ConvNet | 36 | 89 | 0.49 | 0.47‑0.51 | 2.13 | 1.95 | positive | positive |
| ConvNeXt-large | ConvNet | 36 | 198 | 0.52 | 0.49‑0.55 | 1.48 | 2.40 | positive | positive |
| MLP-Mixer-B | MLP-Mixer | 12 | 60 | 0.55 | 0.47‑0.64 | 1.38 | 2.26 | positive | positive |
| MLP-Mixer-L | MLP-Mixer | 24 | 208 | 0.75 | 0.72‑0.78 | 1.43 | 1.25 | positive | positive |
| ViT-base | ViT (supervised) | 12 | 86 | 0.81 | 0.68‑0.98 | 0.89 | 1.59 | – | – |
| DINOv2-base | ViT (self-supervised) | 12 | 86 | 0.77 | 0.72‑0.82 | 1.11 | 1.50 | positive | positive |
| BERT-base | Masked encoder | 12 | 110 | 0.97 | 0.93‑1.02 | 1.08 | 0.99 | – | – |
| RoBERTa-base | Masked encoder | 12 | 125 | 1.13 | 1.09‑1.18 | 1.00 | 0.77 | – | – |
| BERT-large | Masked encoder | 24 | 340 | 0.99 | 0.93‑1.04 | 0.83 | 1.18 | – | – |
| RoBERTa-large | Masked encoder | 24 | 355 | 1.34 | 1.24‑1.44 | 0.91 | 0.58 | – | – |
| T5-small (encoder) | T5 encoder | 6 | 60 | 1.09 | 0.86‑1.49 | 1.02 | 0.82 | – | – |
| T5-base (encoder) | T5 encoder | 12 | 220 | 1.01 | 0.95‑1.07 | 1.03 | 0.95 | – | – |
| T5-large (encoder) | T5 encoder | 24 | 770 | 0.97 | 0.91‑1.03 | 0.96 | 1.10 | – | – |
| Flan-T5-large (encoder) | T5 encoder | 24 | 780 | 0.22 | 0.12‑0.82 | 8.26 | 0.99 | – | – |
| Whisper-small | Audio encoder | 12 | 88 | 0.86 | 0.82‑0.91 | 0.82 | 1.50 | – | – |
| wav2vec2-base | Audio encoder | 12 | 95 | 0.93 | 0.83‑1.03 | 0.76 | 1.40 | – | – |

Two-arm rule: R~ex0~ < 0.80 with early/middle > 1 and late/middle > 1 (14 of 45 positive). Pooled-edge rule: R~ex0~ < 0.80 with late/middle > 1 (16 of 45). 95% CI: input-clustered percentile bootstrap, the five inputs resampled with replacement, each carrying its eight positions, 2,000 resamples (median half-width 0.034); 8 intervals straddle 0.80 (Pythia-160M, Pythia-1B, Phi-2, Pythia-2.8B, Pythia-6.9B, ViT-base, DINOv2-base, Flan-T5-large (encoder)). Block 0 is excluded from every column. The spectral contrast C and the quadratic coefficient a of the earlier classification are in ed_table1.csv; sweeping the ratio threshold from 0.60 to 0.95 gives 6 to 22 interior-valley models (14 at 0.80).
