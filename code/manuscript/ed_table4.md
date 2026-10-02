| Section | Analysis | Subset | Normalised sensitivity | Relative dose (ε = 0.1) | Note |
|---|---|---|---|---|---|
| **a** Held-out profile prediction | controlled, leave one model out | 50 folds | 0.56 / 0.33 | 0.05 / 0.04 | centred R², gain + depth / depth only |
|  | controlled, leave one condition out | 10 folds pooled | 0.51 / 0.29 | −0.11 / −0.09 |  |
|  |  | 4 text conditions pooled | 0.76 / 0.42 | −0.05 / −0.01 | each held out once |
|  |  | 6 k-gram orders pooled | 0.04 / 0.03 | −0.32 / −0.38 | other orders in training |
|  |  | held out: real text | 0.49 / 0.39 | −0.08 / −0.02 | slope fitted without it 1.96 / −0.72 |
|  |  | held out: bidirectional | 0.86 / 0.63 | −3.01 / −1.90 | slope fitted without it 1.96 / −0.88 |
|  |  | held out: shuffled text | 0.90 / 0.37 | −0.72 / 0.27 | slope fitted without it 1.87 / −1.21 |
|  |  | held out: random labels | 0.67 / 0.37 | 0.03 / 0.02 | slope fitted without it 1.99 / −0.26 |
|  |  | held out: k = 1 | −1.71 / −2.05 | 0.23 / 0.27 | slope fitted without it 1.91 / −0.71 |
|  |  | held out: k = 2 | 0.68 / 0.76 | −1.21 / −1.09 | slope fitted without it 2.04 / −0.63 |
|  |  | held out: k = 3 | 0.75 / 0.62 | −3.04 / −3.54 | slope fitted without it 2.01 / −0.56 |
|  |  | held out: k = 4 | 0.58 / 0.64 | −1.21 / −1.44 | slope fitted without it 2.08 / −0.50 |
|  |  | held out: k = 5 | 0.51 / 0.53 | −0.91 / −1.11 | slope fitted without it 2.09 / −0.49 |
|  |  | held out: k = 8 | 0.44 / 0.71 | −1.27 / −1.55 | slope fitted without it 2.14 / −0.40 |
|  | controlled, leave one set out | fit 20 text models, predict 30 k-gram | 0.17 / −0.28 | −1.49 / −1.15 | slope fitted on text 2.19 / 0.49 |
|  |  | fit 30 k-gram models, predict 20 text | 0.51 / 0.33 | 0.05 / 0.01 | slope fitted on k-gram 0.53 / −0.81 |
|  | pretrained, leave one model out | 6 folds | 0.24 / 0.08 | −0.01 / −0.02 | depth = block / (L − 1) |
|  | pretrained, leave one family out | 5 folds (two Pythias together) | 0.32 / 0.02 | 0.24 / 0.03 |  |
| **b** Pooled log–log slope β | controlled, model effects + linear block (reported) | 50 models, 500 rows | 2.01 (s.e. 0.15; 1.72 to 2.29) | −0.61 (s.e. 0.30; −1.35 to −0.06) | model-clustered s.e.; bootstrap 95% CI over models |
|  | controlled, separate effect per block |  | 2.08 (s.e. 0.17) | −0.62 (s.e. 0.35) |  |
|  | controlled, linear depth trend per model |  | 2.35 (s.e. 0.19) | 0.49 (s.e. 0.33) | relative-dose sign depends on the depth specification |
|  | within one condition | real text | 5.39 (3.70 to 7.53) | 5.88 (3.68 to 8.68) | five models; bootstrap over models |
|  |  | bidirectional | 2.23 (1.83 to 2.57) | 1.10 (0.93 to 1.27) |  |
|  |  | shuffled text | 2.13 (1.66 to 2.54) | −0.40 (−0.70 to −0.16) |  |
|  |  | random labels | 2.45 (1.10 to 3.27) | 1.33 (−0.59 to 1.94) |  |
|  |  | k = 1 | 2.38 (1.17 to 4.47) | 0.51 (−0.67 to 2.77) |  |
|  |  | k = 2 | −0.15 (−1.23 to 1.39) | −0.44 (−1.49 to 1.10) |  |
|  |  | k = 3 | 0.56 (−1.12 to 1.57) | −0.59 (−2.63 to 0.79) |  |
|  |  | k = 4 | 4.82 (3.84 to 6.33) | 4.73 (3.90 to 6.09) |  |
|  |  | k = 5 | 4.72 (3.82 to 5.54) | 4.49 (3.49 to 5.37) |  |
|  |  | k = 8 | 2.75 (1.44 to 3.30) | 2.21 (0.93 to 2.80) |  |
|  | per model, block as covariate | 50 models | median 2.52 (IQR 1.37 to 3.75); 42 positive, 40 above 1 | median 1.39 (IQR −0.20 to 3.88); 36 positive |  |
|  | pretrained, model effects + linear depth | 6 models | 2.12 (s.e. 0.73) | 1.89 (s.e. 0.99) | six clusters |
|  | pretrained, linear depth trend per model |  | 0.85 (s.e. 0.72) | 0.88 (s.e. 0.66) |  |
| **c** Within-model rank correlation, blocks 1–10 | controlled, Spearman σ₁ with consequence | 50 models | median 0.60; 44 of 50 positive | median −0.02; 25 of 50 positive |  |
|  | controlled, block partialled out |  | median 0.32; 35 of 50 positive | median 0.19; 35 of 50 positive |  |
|  | controlled, first differences over adjacent blocks |  | median 0.55; 41 of 50 positive | median 0.24; 35 of 50 positive |  |
|  | pretrained, Spearman σ₁ with consequence | 6 models, blocks 1 to L − 2 | median 0.56; 5 of 6 positive | median −0.11; 2 of 6 positive | input-resampled CIs in Fig. 2f |
| **d** Predictor sets on identical folds | controlled, leave one model out | 50 folds | 0.33 / 0.34 / 0.42 / 0.56; joint 0.42 → 0.55 | 0.04 / 0.60 / 0.04 / 0.05; joint 0.65 → 0.73 | depth / + norm / + random-direction gain / + σ₁; joint = depth + norm + random-direction gain, → with σ₁; Δ(σ₁ − random gain) 0.13 (0.06 to 0.20); Δ(joint + σ₁ − joint) 0.14 (0.07 to 0.21) |
|  | controlled, leave one condition out | 10 folds pooled | 0.29 / −0.02 / 0.30 / 0.51; joint 0.18 → 0.13 | −0.09 / 0.34 / −0.14 / −0.11; joint 0.46 → 0.52 | Δ(σ₁ − random gain) 0.21 (0.11 to 0.32); Δ(joint + σ₁ − joint) −0.05 (−0.40 to 0.22) |
|  |  | 4 text conditions pooled | 0.42 / −0.05 / 0.42 / 0.76; joint 0.25 → 0.19 | −0.01 / 0.41 / −0.06 / −0.05; joint 0.56 → 0.64 | Δ(joint + σ₁ − joint) −0.06 (−0.67 to 0.34) |
|  |  | 6 k-gram orders pooled | 0.03 / 0.04 / 0.07 / 0.04; joint 0.04 → 0.01 | −0.38 / 0.09 / −0.44 / −0.32; joint 0.11 → 0.08 | Δ(joint + σ₁ − joint) −0.03 (−0.14 to 0.05) |
|  | controlled, leave one set out | fit 20 text models, predict 30 k-gram | −0.28 / −0.12 / 0.11 / 0.17; joint 0.17 → 0.08 | −1.15 / −0.10 / −2.65 / −1.49; joint 0.22 → 0.12 | Δ(joint + σ₁ − joint) −0.09 (−0.28 to 0.05) |
|  |  | fit 30 k-gram models, predict 20 text | 0.33 / −3.07 / 0.10 / 0.51; joint −6.61 → −8.46 | 0.01 / −0.64 / −0.37 / 0.05; joint −2.30 → −3.00 | Δ(joint + σ₁ − joint) −1.85 (−3.49 to −0.80) |
|  | pretrained, leave one model out | 6 folds | 0.08 / 0.06 / 0.07 / 0.24; joint 0.05 → 0.11 | −0.02 / 0.01 / −0.02 / −0.01; joint −0.01 → 0.02 | Δ(σ₁ − random gain) 0.17 (−0.19 to 0.39); Δ(joint + σ₁ − joint) 0.06 (−0.29 to 0.23) |
|  | pretrained, leave one family out | 5 folds | 0.02 / 0.04 / −0.04 / 0.32; joint −0.00 → 0.22 | 0.03 / −0.01 / −0.01 / 0.24; joint −0.06 → 0.18 | Δ(σ₁ − random gain) 0.36 (0.08 to 0.70); Δ(joint + σ₁ − joint) 0.22 (−0.01 to 0.42); per-family values with intervals in Figure 2g |
| **e** Same-position readout D~t~ | controlled, leave one model out | 50 folds | 0.23 / 0.28 / 0.50 / 0.62; joint 0.52 → 0.64 | 0.17 / 0.74 / 0.16 / 0.18; joint 0.82 → 0.86 | the six sets as in d; Δ(σ₁ − random gain) 0.12 (0.02 to 0.21); Δ(joint + σ₁ − joint) 0.12 (0.05 to 0.19) |
|  | controlled, leave one condition out | 10 folds pooled | 0.15 / −0.62 / 0.35 / 0.57; joint 0.14 → 0.53 | 0.04 / 0.42 / −0.01 / 0.03; joint 0.69 → 0.82 | Δ(σ₁ − random gain) 0.22 (0.07 to 0.39); Δ(joint + σ₁ − joint) 0.38 (0.19 to 0.66) |
|  |  | 4 text conditions pooled | 0.35 / −0.73 / 0.53 / 0.85; joint 0.26 → 0.79 | 0.09 / 0.46 / 0.06 / 0.06; joint 0.78 → 0.93 | Δ(joint + σ₁ − joint) 0.53 (0.29 to 1.02) |
|  |  | 6 k-gram orders pooled | −0.45 / −0.29 / −0.18 / −0.28; joint −0.20 → −0.27 | −0.27 / 0.18 / −0.41 / −0.16; joint 0.24 → 0.21 | Δ(joint + σ₁ − joint) −0.07 (−0.19 to 0.02) |
|  | controlled, leave one set out | fit 20 text models, predict 30 k-gram | −1.75 / −1.34 / −0.28 / −0.24; joint −0.08 → −0.34 | −1.31 / −0.41 / −3.88 / −1.52; joint 0.32 → 0.19 | Δ(joint + σ₁ − joint) −0.26 (−0.43 to −0.12) |
|  |  | fit 30 k-gram models, predict 20 text | 0.15 / −1.18 / 0.29 / 0.41; joint −2.44 → −3.26 | 0.14 / 0.57 / −0.07 / 0.18; joint 0.25 → 0.12 | Δ(joint + σ₁ − joint) −0.82 (−1.49 to −0.45) |
|  | pretrained, leave one model out | 6 folds | 0.34 / 0.36 / 0.33 / 0.40; joint 0.34 → 0.37 | 0.27 / 0.25 / 0.26 / 0.26; joint 0.23 → 0.23 | Δ(σ₁ − random gain) 0.07 (−0.02 to 0.15); Δ(joint + σ₁ − joint) 0.04 (−0.09 to 0.13) |
|  | pretrained, leave one family out | 5 folds | 0.28 / 0.30 / 0.24 / 0.39; joint 0.26 → 0.33 | 0.27 / 0.19 / 0.24 / 0.29; joint 0.14 → 0.18 | Δ(σ₁ − random gain) 0.15 (0.02 to 0.40); Δ(joint + σ₁ − joint) 0.07 (−0.07 to 0.23) |
|  | controlled, pooled slope and within-model ρ | 50 models | 2.05 (s.e. 0.11); median ρ 0.59, 42 of 50 positive | −0.72 (s.e. 0.29); median ρ −0.25, 18 of 50 positive | all-positions values in b and c |
|  | pretrained, pooled slope and within-model ρ | 6 models | 0.98 (s.e. 0.42); median ρ 0.38, 5 of 6 positive | 0.60 (s.e. 0.44); median ρ −0.04, 3 of 6 positive |  |

Normalised sensitivity is the same-sequence divergence along v₁ divided by the squared displacement; relative dose is the divergence at a displacement of 0.1‖h‖. Centred R² is 1 − SS(residual)/SS(total) of the within-model-centred log profile pooled over the held-out models, with β and γ fitted on the remaining models (loco_check.py, matched_pretrained_analysis.py); it measures the shape of a profile, not its level. Leave one condition out asks whether a condition absent from training is predicted when the other nine are present; the text-to-k-gram fold asks whether a whole source family is predicted from the other. Pooled slopes: regression_check.py; per-model and correlation values: matched_analysis.py. Section d compares six predictor sets on identical folds (predictor_check.py): block position; block position with the log activation norm; with the log gain along a random unit direction at ε = 0.01; with log σ₁; the joint baseline of block position with the norm and the random-direction gain; and the joint baseline with log σ₁. Δ(σ₁ − random gain) is the pooled R² of the σ₁ set minus that of the random-direction set on the same folds, and Δ(joint + σ₁ − joint) that of the joint baseline with σ₁ minus without, each with a 95% percentile interval from 2,000 resamples of the held-out models. Section e repeats d for the divergence read at the perturbed position only (D~t~, the Fig. 3 readout) in place of the mean over the perturbed and later positions (D̄, the readout of a–c and of Fig. 2); readout_check.py.