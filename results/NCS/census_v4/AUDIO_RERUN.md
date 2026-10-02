# Audio rows of the census: re-run needed (21 Sept 2026)

scramble_census_v4.py cropped LibriSpeech test-clean audio to 10 s but kept the full transcript, so for utterances
longer than 10 s the CTC (wav2vec2) and teacher-forced (Whisper) readouts compared cropped audio with an uncropped
transcript. scramble_census_v4_audiofix.py keeps only utterances of at most 10 s (uncropped) and skips the rest.

Steps (Colab, the census notebook with the fixed script in place of the original):
1. delete the cached audio file under CENSUS_DATA (the .pt written by librispeech_test_clean);
2. CENSUS_MODELS="facebook/wav2vec2-base-960h openai/whisper-small" python scramble_census_v4_audiofix.py --no-skip
3. python census_merge_v4.py  (rebuilds census_v4.json / census_v4_stats.json / merged_v4)
4. python make_tables.py; python make_figs2.py; python build.py  (ED Table 2, ED Fig. 6, Fig. 5f)
Done 21 Sept 2026 20:10 ACST: both models re-run, merged (census_v4.json, census_v4_stats.json), tables and figures rebuilt; old files in results/pre_audiofix.
