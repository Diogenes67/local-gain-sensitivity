# Results manifest (NMI submission)

178 files; SHA-256 of each file read by the analyses. Paths are relative to the PCI folder.


## Survey, canonical protocol (45 models, five inputs)

Used by: Fig. 5a,b; ED Table 1; ED Fig. 2

| File | Bytes | SHA-256 |
|---|---|---|
| results/survey_sigma1_v2/survey_results_v2_15sept.zip | 1288072 | 665a3d3567563ec0… |

## Survey, 30 inputs (11 models)

Used by: Fig. 5a; Methods

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/survey_30inputs/results/sigma1_EleutherAI_pythia-1_4b.json | 156674 | ba71d3c1dff5edae… |
| NCS/survey_30inputs/results/sigma1_EleutherAI_pythia-410m.json | 157385 | 474d5b4bce2333ed… |
| NCS/survey_30inputs/results/sigma1_FacebookAI_roberta-large.json | 157870 | de24b88c4bdfd274… |
| NCS/survey_30inputs/results/sigma1_Qwen_Qwen2_5-1_5B.json | 180509 | a088ef79ca468e54… |
| NCS/survey_30inputs/results/sigma1_facebook_convnext-base-224.json | 235765 | ec9fa9a9f957cb40… |
| NCS/survey_30inputs/results/sigma1_facebook_dinov2-base.json | 80070 | 4d300ebb75481706… |
| NCS/survey_30inputs/results/sigma1_google_gemma-2-2b.json | 169075 | b0d6ec4d2a8d6db8… |
| NCS/survey_30inputs/results/sigma1_google_vit-base-patch16-224.json | 80495 | 9d6e1e87bd390fc5… |
| NCS/survey_30inputs/results/sigma1_gpt2.json | 79579 | d7bd7f855dea7adb… |
| NCS/survey_30inputs/results/sigma1_meta-llama_Llama-3_2-1B.json | 105081 | 4032710aa796fd1e… |
| NCS/survey_30inputs/results/sigma1_state-spaces_mamba-130m-hf.json | 157957 | 6de18d00632839b0… |
| NCS/survey_30inputs/results/survey_summary.json | 9051 | 31d27bc21a6c5dda… |

## Estimator validation (exact SVD)

Used by: Fig. 1d,e

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/validation/cellH.log | 18494 | 444eb5ff318d5ba7… |
| NCS/validation/validate_convnext_mamba.ipynb | 290977 | 5964a8e55291c9a2… |
| NCS/validation/validation_cellH.json | 28037 | bb7bff483bc32817… |
| NCS/validation/validation_convnext_mamba_from_log.json | 23435 | 166389584a25fd66… |

## Comparison with Aubry et al.

Used by: Methods

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/aubry/results/aubry_EleutherAI_pythia-410m_bfloat16.json | 59698 | dfb4139a4c9ebecb… |
| NCS/aubry/results/aubry_EleutherAI_pythia-410m_float32.json | 59770 | ed32e865a8aad8c9… |
| NCS/aubry/results/aubry_gpt2_bfloat16.json | 31289 | e79bed7880bb4d26… |
| NCS/aubry/results/aubry_gpt2_float32.json | 31296 | e2b7529e4773f87f… |
| NCS/aubry/results_blackwell/aubry_EleutherAI_pythia-410m_bfloat16.json | 59636 | 9ba5803c2c327fd2… |
| NCS/aubry/results_blackwell/aubry_EleutherAI_pythia-410m_float32.json | 59813 | 261c8bf22551dfdb… |
| NCS/aubry/results_blackwell/aubry_gpt2_bfloat16.json | 31346 | 5fff16020d3b9f66… |
| NCS/aubry/results_blackwell/aubry_gpt2_float32.json | 31356 | 101c625514e4e295… |

## Destroyed-data and switching (d1 v2.1)

Used by: Fig. 4a,b; ED Table 5

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/d1/results_v21/d1_v21_results.zip | 101031 | fc0f49bb5acd3a3e… |

## Mask x loss (factorial v2; per-model summary parsed from the run log)

Used by: ED Fig. 9a; Methods

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/factorial/factorial_v2.zip | 48903 | 08ba9b00c3cc86c3… |
| NCS/factorial/factorial_v2_from_log.json | 4906 | f2ef90bb9e910bee… |
| NCS/factorial/factorial_v2_with_output.ipynb | 258658 | dc56acfe47e17f24… |
| NCS/factorial/results/AR_s0.json | 7022 | f5098ee9e3225587… |
| NCS/factorial/results/AR_s1.json | 7033 | 9283cf1f77c378fa… |
| NCS/factorial/results/AR_s2.json | 7027 | 83cff38b2e28b5b0… |
| NCS/factorial/results/AR_s3.json | 7025 | 2f739af85b9e8e80… |
| NCS/factorial/results/AR_s4.json | 7018 | ae3b5ad7fa6f80b8… |
| NCS/factorial/results/CMLM_s0.json | 7038 | 64a305988a79d709… |
| NCS/factorial/results/CMLM_s1.json | 7058 | cd4e438da3bc2745… |
| NCS/factorial/results/CMLM_s2.json | 7025 | 70e5c1aae6b148fc… |
| NCS/factorial/results/CMLM_s3.json | 7033 | 644e37ad7c31618d… |
| NCS/factorial/results/CMLM_s4.json | 7038 | cb748c55fdf6c9a7… |
| NCS/factorial/results/MLM_s0.json | 7043 | abf2b1bac9acdc3d… |
| NCS/factorial/results/MLM_s1.json | 7033 | 5d8bd4328386c18e… |
| NCS/factorial/results/MLM_s2.json | 7062 | a1d7b9e9c76ab45c… |
| NCS/factorial/results/MLM_s3.json | 7048 | 6f0d9847206630f9… |
| NCS/factorial/results/MLM_s4.json | 7035 | 38f3f4d8d7824fd7… |
| NCS/factorial/results/PLM_s0.json | 7022 | 636eaad7294ab042… |
| NCS/factorial/results/PLM_s1.json | 7045 | e015e9f30bf9f7fc… |
| NCS/factorial/results/PLM_s2.json | 7052 | b541542377fcf744… |
| NCS/factorial/results/PLM_s3.json | 7040 | 5dfc4d62114f5401… |
| NCS/factorial/results/PLM_s4.json | 7030 | b59e6b34f9d69878… |

## Width x depth grid

Used by: ED Fig. 9b

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/grid/b1_grid_output.txt | 22151 | b84be3b7be62fa8c… |
| NCS/grid/b1_width_depth_grid_reconstructed.json | 23039 | f0333f35f71bbf68… |

## Reprofile v2 (topology, skip coefficient, k-gram)

Used by: Fig. 4c; ED Figs 5 and 10; ED Table 5

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/reprofile/results_alpha.zip | 86997 | 98987ce66cddbff4… |
| NCS/reprofile/results_kgram.zip | 194426 | 31905f1161650998… |
| NCS/reprofile/results_topology.zip | 116354 | 92152ff481a65462… |

## Entropy-matched Markov family

Used by: Fig. 4d,e; ED Table 5

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/markov/markov_stats.json | 20307 | 45f1be4de8208705… |
| NCS/markov/markov_v1_results_10k.zip | 144599 | 9bfa2f0b9a7f7628… |
| NCS/markov/markov_v1_results_2k.zip | 134477 | 69ddd7ce936e7fcb… |

## Scale point, 24 layers

Used by: Fig. 2e; ED Table 5

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/scale/scale_stats.json | 3169 | 26fe915d4e4aa825… |
| NCS/scale/scale_v1_results.zip | 1360414 | a33e1a09a6ef7b37… |

## One-hop synthetic sources

Used by: Methods

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/synthetic/synthetic_context_v2_results.zip | 194957 | 647adfb893d82a30… |

## Intervention comparison

Used by: Table 1; Methods

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/intervention/results/intervention_compare_8models_summary.txt | 4855 | ec91d89d51b34dcb… |

## Matched assay, controlled (50 models) and sampling check

Used by: Figs 2a–e and 3a,c–e; ED Fig. 7; ED Table 4

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/matched/linresp/linresp_d1_real_s0.json | 617563 | e0cbfce5ba0ec4d8… |
| NCS/matched/linresp/linresp_d1_real_s1.json | 617124 | af5b9e14ea2841bf… |
| NCS/matched/linresp/linresp_d1_shuffled_s0.json | 620381 | 3983faadc6f5d88a… |
| NCS/matched/linresp/linresp_d1_shuffled_s1.json | 620283 | b20637daf6c78c4c… |
| NCS/matched/linresp/linresp_kgram_1_s0.json | 619138 | 8480a01c5b6c9cc8… |
| NCS/matched/linresp/linresp_kgram_1_s1.json | 619652 | d8fe42037c90f48d… |
| NCS/matched/linresp/linresp_kgram_8_s0.json | 612356 | 78b0a3292dc7f2bd… |
| NCS/matched/linresp/linresp_kgram_8_s1.json | 612984 | 4ef154e7cf447958… |
| NCS/matched/matched_results.zip | 5627806 | f146d3557766f5a4… |
| NCS/matched/matched_stats.json | 35388 | 7df5d07ae6fc6719… |
| NCS/matched/sampling/matched_d1_real_s0.json | 2151334 | c958be1a13b55534… |
| NCS/matched/sampling/matched_d1_real_s1.json | 2151956 | 9b470cd3aa6d1af0… |
| NCS/matched/sampling/matched_d1_shuffled_s0.json | 2163554 | 981d68b698c8835d… |
| NCS/matched/sampling/matched_d1_shuffled_s1.json | 2162705 | 254b9c37d4a0ceb1… |
| NCS/matched/sampling/matched_kgram_1_s0.json | 2158831 | a8a5a44665b3f9ec… |
| NCS/matched/sampling/matched_kgram_1_s1.json | 2158648 | c6c417043c18cbf6… |
| NCS/matched/sampling/matched_kgram_8_s0.json | 2139455 | 31340e21016c3e3e… |
| NCS/matched/sampling/matched_kgram_8_s1.json | 2139091 | 48f30719587a94a0… |

## Matched assay, pretrained decoders

Used by: Figs 2f and 3b–e; ED Table 4

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/matched_pretrained/e1_extra.json | 2975 | 2ab9b8a73e6a0a11… |
| NCS/matched_pretrained/matched_pretrained_stats.json | 6401 | 5004540fb7223ffc… |
| NCS/matched_pretrained/plumbing_checks.json | 757 | 1f88fe7abdf47f27… |
| NCS/matched_pretrained/results/mp_EleutherAI_pythia-1_4b.json | 8383764 | cf60adb9b0e89ca0… |
| NCS/matched_pretrained/results/mp_EleutherAI_pythia-410m.json | 8410872 | 57931bf1e8c773a1… |
| NCS/matched_pretrained/results/mp_Qwen_Qwen2_5-1_5B.json | 9840769 | 16b4bc79ab609da6… |
| NCS/matched_pretrained/results/mp_google_gemma-2-2b.json | 9167767 | d54b021bf8fb5469… |
| NCS/matched_pretrained/results/mp_gpt2.json | 4193960 | 3fa6085df4ff7e88… |
| NCS/matched_pretrained/results/mp_meta-llama_Llama-3_2-1B.json | 5611714 | 38baa0e43671cba9… |

## Census v4 (branch rotation)

Used by: Fig. 5b; ED Table 2; ED Fig. 6

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/census_v4/census_v4.csv | 23255 | 02e9e84fa46f1386… |
| NCS/census_v4/census_v4.json | 47253 | 05e4b37497f396ea… |
| NCS/census_v4/census_v4_results.zip | 23190 | ef81847a8e3dc5bb… |
| NCS/census_v4/census_v4_stats.json | 1524 | e1a43587eae71221… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-1.4b.json | 3259 | e83d798cbb70fdbc… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-12b.json | 4660 | 782cfd8602154c7a… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-160m.json | 1898 | 079f9cb46c8c0bd7… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-1b.json | 2357 | 5a6ad8bad4504279… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-2.8b.json | 4181 | 4d438a350de4cd7a… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-410m.json | 3256 | 455678c337dec50f… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-6.9b.json | 4207 | fff1e66a922c2ae1… |
| NCS/census_v4/merged_v4/perlayer_eleutherai_pythia-70m.json | 1228 | 71f1628d4fde9bf5… |
| NCS/census_v4/merged_v4/perlayer_facebook_convnext-base-224.json | 9857 | 3d7720eddff4a3e5… |
| NCS/census_v4/merged_v4/perlayer_facebook_convnext-large-224.json | 9880 | b5893e593d7cbc2f… |
| NCS/census_v4/merged_v4/perlayer_facebook_convnext-small-224.json | 9836 | a3049da315e755e9… |
| NCS/census_v4/merged_v4/perlayer_facebook_convnext-tiny-224.json | 5279 | a2560faaf02572f2… |
| NCS/census_v4/merged_v4/perlayer_facebook_dinov2-base.json | 3028 | 70e599e7a8d9f4d8… |
| NCS/census_v4/merged_v4/perlayer_facebook_wav2vec2-base-960h.json | 2414 | dbe5ace050902a7b… |
| NCS/census_v4/merged_v4/perlayer_facebookai_roberta-base.json | 2468 | 6277bba624006876… |
| NCS/census_v4/merged_v4/perlayer_facebookai_roberta-large.json | 3646 | dbcfc2176dfd3629… |
| NCS/census_v4/merged_v4/perlayer_google-bert_bert-base-uncased.json | 2481 | 062b3fff13e455fb… |
| NCS/census_v4/merged_v4/perlayer_google-bert_bert-large-uncased.json | 3695 | f471af3ca67a483b… |
| NCS/census_v4/merged_v4/perlayer_google-t5_t5-base.json | 2566 | 4a16db77056eb6d2… |
| NCS/census_v4/merged_v4/perlayer_google-t5_t5-large.json | 3866 | 3a197905119bda2f… |
| NCS/census_v4/merged_v4/perlayer_google-t5_t5-small.json | 1938 | be4a8eab29493406… |
| NCS/census_v4/merged_v4/perlayer_google_flan-t5-large.json | 3820 | e28864118480cf6a… |
| NCS/census_v4/merged_v4/perlayer_google_gemma-2-2b.json | 1395 | 945f05e1029652bd… |
| NCS/census_v4/merged_v4/perlayer_google_vit-base-patch16-224.json | 3032 | f5ad70f5b689c4f0… |
| NCS/census_v4/merged_v4/perlayer_gpt2-large.json | 4851 | fc5aed458994cd1c… |
| NCS/census_v4/merged_v4/perlayer_gpt2-xl.json | 6261 | 2e2d00bea39fb640… |
| NCS/census_v4/merged_v4/perlayer_gpt2.json | 1895 | 3aba23b447b493fa… |
| NCS/census_v4/merged_v4/perlayer_meta-llama_llama-3.2-1b.json | 2530 | 8419939cfeaddfed… |
| NCS/census_v4/merged_v4/perlayer_meta-llama_llama-3.2-3b.json | 3906 | 9e538fd30766af7a… |
| NCS/census_v4/merged_v4/perlayer_microsoft_phi-2.json | 4389 | 72f05e237bc5dbfa… |
| NCS/census_v4/merged_v4/perlayer_mistralai_mistral-7b-instruct-v0.2.json | 4225 | 6b3d7e2afea91f9b… |
| NCS/census_v4/merged_v4/perlayer_mistralai_mistral-7b-v0.1.json | 4218 | a53ee14ae3d86b28… |
| NCS/census_v4/merged_v4/perlayer_openai_whisper-small.json | 2499 | 1a43a4eac0ab7962… |
| NCS/census_v4/merged_v4/perlayer_qwen_qwen2-0.5b.json | 3256 | bc1fc340ca6b1d75… |
| NCS/census_v4/merged_v4/perlayer_qwen_qwen2.5-0.5b.json | 3245 | 4c55bb74489910fb… |
| NCS/census_v4/merged_v4/perlayer_qwen_qwen2.5-1.5b.json | 3717 | cdad85c6c23b236c… |
| NCS/census_v4/merged_v4/perlayer_qwen_qwen2.5-14b.json | 6192 | c8fc93eb8503e98c… |
| NCS/census_v4/merged_v4/perlayer_qwen_qwen2.5-3b.json | 4777 | 130760989ca6fdd3… |
| NCS/census_v4/merged_v4/perlayer_qwen_qwen2.5-7b.json | 3883 | 0828dccb55d16c1a… |
| NCS/census_v4/merged_v4/perlayer_state-spaces_mamba-1.4b-hf.json | 6268 | 1caff0834ab68737… |
| NCS/census_v4/merged_v4/perlayer_state-spaces_mamba-130m-hf.json | 3479 | 9010311e87b5ccb4… |
| NCS/census_v4/merged_v4/perlayer_state-spaces_mamba-2.8b-hf.json | 8139 | df22f06b62486a94… |
| NCS/census_v4/merged_v4/perlayer_state-spaces_mamba-370m-hf.json | 6228 | 7c66f17c44cfa87f… |
| NCS/census_v4/merged_v4/perlayer_timm_mixer_b16_224.goog_in21k_ft_in1k.json | 3036 | 67ba816a10126504… |
| NCS/census_v4/merged_v4/perlayer_timm_mixer_l16_224.goog_in21k_ft_in1k.json | 4772 | 20e11c683036e4fd… |

## Analysis outputs

Used by: Figs 1–5; ED Tables 4 and 5

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/fig1_nmi_check.json | 1072 | 1a7b618a5177046f… |
| NCS/fig2_scale_nmi_check.json | 1055 | d7cced200bf2196b… |
| NCS/fig3_direction_nmi_check.json | 2906 | 6ef30ad2d1f3d184… |
| NCS/fig4_data_nmi_check.json | 820 | 1b222197994dc499… |
| NCS/fig5_pretrained_nmi_check.json | 1418 | f2be99d50ef66647… |
| NCS/linresp_afterblock.json | 12673 | 7c76c041cd308d46… |
| NCS/linresp_stats.json | 10734 | 5e50ed125604dc4c… |
| NCS/loco_check.json | 12039 | 00da11563cce88f6… |
| NCS/region_check.json | 17240 | 6c7ad03d37871d46… |
| NCS/regression_check.json | 9855 | 107496c1d81d73c2… |
| NCS/sampling_check.json | 11015 | a48ec51068b498d4… |
| NCS/tables/ed_tables45_check.json | 6497 | baad3671f84f1500… |

## Retained March 2026 files (mechanism experiments)

Used by: ED Fig. 8; SN 1 and 4

| File | Bytes | SHA-256 |
|---|---|---|
| results/d3l_alignment_adaptation/d3l_aggregate.json | 3145 | e3c99ecab0f57a99… |
| results/exp2a_bracket_seed0.json | 5310 | d299315368eb27c0… |
| results/exp2a_bracket_seed1.json | 5316 | f1597a25b4931bbb… |
| results/exp2a_bracket_seed2.json | 5294 | 16aa07c59317b843… |
| results/exp2a_bracket_seed3.json | 5297 | 6368da75086d0f3a… |

## Pythia-410M public checkpoints (E7, 23 Sept 2026)

Used by: Fig. 5c; ED Fig. 4c,d; ED Table 3

| File | Bytes | SHA-256 |
|---|---|---|
| NCS/e7_checkpoints/results/census/census_step0.json | 2982 | 3536cc2224d6d3ee… |
| NCS/e7_checkpoints/results/census/census_step1000.json | 2881 | b71caa1843418045… |
| NCS/e7_checkpoints/results/census/census_step100000.json | 2816 | 4c0ecb9517c29251… |
| NCS/e7_checkpoints/results/census/census_step143000.json | 2911 | f1b2a4e27f5fdb5e… |
| NCS/e7_checkpoints/results/census/census_step16000.json | 2832 | 884f28eecac9dd5b… |
| NCS/e7_checkpoints/results/census/census_step2000.json | 2852 | 7e0cef2ab562002f… |
| NCS/e7_checkpoints/results/census/census_step32000.json | 2834 | 9854ae394d94d3a5… |
| NCS/e7_checkpoints/results/census/census_step4000.json | 2848 | 0535858a04ad3c4e… |
| NCS/e7_checkpoints/results/census/census_step512.json | 2916 | 064140598c34a1d2… |
| NCS/e7_checkpoints/results/census/census_step64000.json | 2826 | eb533f9cf6f57332… |
| NCS/e7_checkpoints/results/census/census_step8000.json | 2837 | 24d40eafa888083e… |
| NCS/e7_checkpoints/results/e7_summary.json | 3784 | 0c0c514814b0a9bf… |
| NCS/e7_checkpoints/results/run_log.txt | 43813 | 995659adf7f27178… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step0.json | 160420 | 8c0eba14a6ca1328… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step1000.json | 160293 | 9d5df71a02652653… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step100000.json | 157881 | 6a4bfb2c4cca98c6… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step143000.json | 157583 | b6f6dd50c65757fe… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step16000.json | 158514 | d84db650eaf6a1c3… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step2000.json | 159478 | ecc74641da77b708… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step32000.json | 158083 | d91df7c17f9631be… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step4000.json | 158809 | 909b878467a37fb1… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step512.json | 160262 | 66d5d1d3291be528… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step64000.json | 157944 | c77ce29959c6cfed… |
| NCS/e7_checkpoints/results/sigma1/sigma1_step8000.json | 158827 | 0be0d4e4ca31437f… |
