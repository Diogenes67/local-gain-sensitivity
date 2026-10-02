| Setting | Condition | n | Middle-to-edge gain ratio | Early / middle | Late / middle | Middle-block rotation cost (nats) | Middle-block skip cost (nats) |
|---|---|---|---|---|---|---|---|
| 12 blocks, training conditions | real text | 5 | 1.00 ± 0.03 | 1.12 ± 0.04 | 0.88 ± 0.03 | 1.05–1.15 | 0.751–0.845 |
|  | token-shuffled text | 5 | 0.55 ± 0.06 | 2.76 ± 0.40 | 0.91 ± 0.02 | 0.039–0.055 | 0.036–0.057 |
|  | random labels | 5 | 0.84 ± 0.08 | 1.45 ± 0.26 | 0.96 ± 0.03 | −0.009 to 0.013 | −0.002 to 0.005 |
|  | bidirectional masked text | 5 | 0.84 ± 0.04 | 1.49 ± 0.09 | 0.89 ± 0.02 | 0.105–0.160 | 0.069–0.122 |
| 12 blocks, condition switching | real → shuffled at step 5,000 | 3 | 0.78 ± 0.03 | 1.55 ± 0.08 | 1.03 ± 0.04 | 0.181–0.205 | 0.129–0.151 |
|  | shuffled → real at step 5,000 | 3 | 0.84 ± 0.03 | 1.49 ± 0.06 | 0.89 ± 0.02 | 0.359–0.464 | 0.309–0.395 |
| 12 blocks, during training | real text, step 1,000 | 5 | 1.18 ± 0.05 | 0.93 ± 0.04 | 0.76 ± 0.03 | 0.238–0.290 | – |
|  | real text, step 3,000 | 5 | 1.08 ± 0.03 | 1.06 ± 0.03 | 0.79 ± 0.02 | 0.554–0.649 | – |
|  | real text, step 5,000 | 5 | 1.03 ± 0.02 | 1.11 ± 0.03 | 0.82 ± 0.01 | 0.775–0.839 | – |
|  | real text, step 10,000 | 5 | 1.00 ± 0.03 | 1.12 ± 0.04 | 0.88 ± 0.03 | 1.05–1.15 | – |
|  | token-shuffled text, step 1,000 | 5 | 0.64 ± 0.02 | 2.17 ± 0.11 | 0.95 ± 0.01 | 0.014–0.024 | – |
|  | token-shuffled text, step 3,000 | 5 | 0.57 ± 0.05 | 2.60 ± 0.33 | 0.94 ± 0.01 | 0.028–0.050 | – |
|  | token-shuffled text, step 5,000 | 5 | 0.56 ± 0.02 | 2.67 ± 0.17 | 0.93 ± 0.01 | 0.047–0.064 | – |
|  | token-shuffled text, step 10,000 | 5 | 0.55 ± 0.06 | 2.76 ± 0.40 | 0.91 ± 0.02 | 0.039–0.055 | – |
| 12 blocks, k-gram sources | k = 1 | 5 | 0.97 ± 0.04 | 1.13 ± 0.07 | 0.93 ± 0.02 | 0.072–0.101 | 0.069–0.104 |
|  | k = 2 | 5 | 0.92 ± 0.05 | 1.33 ± 0.12 | 0.84 ± 0.04 | 0.142–0.160 | 0.116–0.125 |
|  | k = 3 | 5 | 0.98 ± 0.02 | 1.09 ± 0.04 | 0.95 ± 0.01 | 0.317–0.372 | 0.253–0.290 |
|  | k = 4 | 5 | 1.06 ± 0.04 | 0.96 ± 0.04 | 0.93 ± 0.04 | 0.537–0.588 | 0.427–0.480 |
|  | k = 5 | 5 | 1.08 ± 0.03 | 0.90 ± 0.05 | 0.95 ± 0.02 | 0.673–0.828 | 0.554–0.677 |
|  | k = 8 | 5 | 1.01 ± 0.04 | 0.98 ± 0.05 | 0.99 ± 0.03 | 0.813–0.952 | 0.623–0.749 |
| 12 blocks, entropy-matched (1.00 nats) | m = 1 (8 contexts) | 5 | 0.94 ± 0.01 | 1.19 ± 0.03 | 0.95 ± 0.01 | 0.003–0.006 | 0.001–0.004 |
|  | m = 2 (64 contexts) | 5 | 1.06 ± 0.08 | 1.01 ± 0.12 | 0.88 ± 0.04 | 0.026–0.078 | 0.017–0.076 |
|  | m = 3 (512 contexts) | 5 | 0.92 ± 0.08 | 1.32 ± 0.22 | 0.87 ± 0.06 | 0.075–0.345 | 0.051–0.304 |
|  | m = 4 (4,096 contexts) | 5 | 0.93 ± 0.06 | 1.34 ± 0.11 | 0.82 ± 0.04 | 0.169–0.307 | 0.137–0.257 |
|  | m = 5 (32,768 contexts) | 5 | 0.98 ± 0.08 | 1.30 ± 0.14 | 0.77 ± 0.03 | 0.184–0.396 | 0.154–0.344 |
|  | m = 6 (262,144 contexts) | 5 | 0.89 ± 0.05 | 1.51 ± 0.13 | 0.74 ± 0.02 | 0.258–0.313 | 0.223–0.265 |
| 24 blocks | real text | 3 | 0.97 ± 0.03 | 1.27 ± 0.05 | 0.79 ± 0.02 | 0.789–0.938 | 0.663–0.768 |
|  | token-shuffled text | 3 | 0.74 ± 0.03 | 1.77 ± 0.12 | 0.96 ± 0.01 | 0.035–0.053 | 0.034–0.052 |

Gain ratios are mean ± s.d. over seeds under the canonical estimator (thirds of blocks 1 to L − 1: 1–3, 4–8, 9–11 at 12 blocks; 1–7, 8–16, 17–23 at 24). Intervention costs are the range over seeds of the loss increase when the branch outputs of the middle blocks are rotated (Haar rotation, dose 1) or the blocks are skipped (identity), blocks 4–7 at 12 blocks and 8–15 at 24. Switching models are read out on the objective of their final phase. Trajectory values are from the checkpoints saved during the same runs. Entropy-matched sources have conditional entropy 1.00 nats at every order m over eight tokens. Sources: d1/results_v21, reprofile/results_kgram, markov/markov_stats.json, scale/scale_stats.json.