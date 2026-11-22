# Contribution record — Ersin Ege Tezgüler

## Project direction

The project evaluates ASR on singing. Genius is used for song candidate
metadata; LRCLIB `plainLyrics`/`syncedLyrics` are assumed gold references. The
corpus is Japanese-only. Ad-libs are never manually added; omissions and
version mismatches are disclosed as limitations rather than patched.

## Work completed

- Genius API candidate discovery, query provenance, deduplication, and
  merge-existing behavior.
- LRCLIB matching by title/artist variants, synced-lyrics preservation,
  resumable fetching, and unresolved-match reporting.
- yt-dlp video-search metadata discovery and a local audio-only WAV
  downloader with source IDs/checksums.
- English purge (`lyrics-asr-purge-english`): removed every English row from
  every tracked and local file, reproducibly and idempotently (CI asserts a
  rerun removes nothing). 4,146 JSONL rows, 20 search queries, 12 selection
  groups and 29 English audio files were removed.
- Japanese corpus construction (`lyrics-asr-ja-corpus`), seed 42: unions the
  100-song Japanese expansion pool with the surviving pilot rows (108
  candidates, 22 speaker groups), applies a language/script/timing/duration/
  speaker-cap gate chain, and logs all 36 exclusions with a machine-readable
  reason. Result: **72 songs, 20 speaker groups**. Four test speaker groups
  (`aimer`, `king-gnu`, `satoshi-fujihara`, `vaundy`) are pinned to preserve
  the earlier cached zero-shot baselines on those 9 songs; seed 42 assigns
  every other group to train or validation.
- Reference/audio alignment (`lyrics-asr-ja-align`): a `faster-whisper-base`
  probe plus a 0.5 s grid search recovers one scalar offset per song between
  the LRCLIB timestamp base and the released video, falling back to a
  duration prior below a confidence threshold. No song is ever dropped on
  this score, so the frozen test split stays independent of any model's
  accuracy.
- Utterance segmentation (`lyrics-asr-ja-segment`): cuts offset-corrected,
  ≤28 s utterance clips at 16 kHz mono from merged LRCLIB lines. **658
  utterances** (408 train / 110 validation / 140 test) from 4.33 h of used
  audio (5.07 h of source audio over the 72 songs).
- Made the evaluator Japanese-only: CER + Character F1 (micro-aggregated,
  NFKC + casefold + punctuation/whitespace normalization), with WER dropped
  because Japanese has no whitespace-delimited word boundaries. Added a
  dependency-free standalone reimplementation for the Colab path, verified
  byte-identical to the project scorer.
- Configured and ran **7 systems in 3 tiers** on a free Colab T4: 4
  multilingual Whisper baselines (74M–809M, CTranslate2/float16), 2
  already-trained Japanese-only checkpoints (`kotoba-whisper-v2.0`,
  `anime-whisper`), and a LoRA fine-tune of `whisper-small` on the `train`
  split (`r=32`, `alpha=64`, `q_proj`/`v_proj`, 4 epochs, lr `5e-5`, epoch 2
  selected by lowest `validation` CER, `test` decoded once).
- Built a self-contained `colab_handoff/` package (notebook + runner +
  standalone scorer + instructions + a 89.9 MB Opus-encoded clip bundle) so
  the GPU run needs no repository clone.
- Independently re-scored every Colab prediction file with the project's own
  `lyrics_asr.ja_evaluate` before publishing a single number, to confirm the
  Colab run's numbers rather than just trust them.
- 68 unit/integration tests, synthetic and real-fixture; GitHub Actions CI on
  Python 3.10/3.12 plus a corpus-purge regression check; release-artifact
  workflow.
- Rewrote `README.md` for the Japanese-only, now-complete study: split
  statistics (real, generated — not hand-typed), exclusion tables, the 7-row
  final results table for both `test` and `validation`, and an explicit
  pairwise contribution write-up.
- Rebuilt the 10-slide presentation deck (`presentation/Can_ASR_Hear_Sung_Lyrics.pptx`)
  for the Japanese-only, final study: all counts, the split table, the
  system-tier taxonomy, and the benchmark-results slide were replaced with
  the real 72-song corpus and the full 7-system test table.

## Final results (test split: 140 utterances, 15 songs, 4 speaker groups)

| Tier | System | CER ↓ | Character F1 ↑ |
|---|---|---:|---:|
| baseline | `faster-whisper-large-v3-turbo` (809M) | **34.90%** | **73.08%** |
| baseline | `faster-whisper-medium` (769M) | 36.34% | 71.83% |
| baseline | `faster-whisper-small` (244M) | 41.04% | 67.49% |
| contribution | `whisper-small-lora-ja` (ours) | 42.28% | 66.20% |
| japanese-pretrained | `kotoba-whisper-v2.0` (756M) | 49.16% | 60.55% |
| baseline | `faster-whisper-base` (74M) | 60.55% | 53.13% |
| japanese-pretrained | `anime-whisper` (756M) | 61.64% | 52.92% |

Scale dominates: the 4 baselines rank exactly by parameter count, and neither
Japanese-only pretrained checkpoint beats the two largest multilingual
baselines on sung audio. The LoRA contribution beats its own base checkpoint
(`faster-whisper-small`) on `validation` (its model-selection split: 48.51%
vs. 51.22% CER) but not on `test` (42.28% vs. 41.04%) — reported as the
honest finding, not hidden behind the selection split. Full per-song/
per-speaker breakdowns, environment details, and the fine-tune epoch history
are in `reports/`, `final_metrics.json`, and `finetune_report.json`.

## Remaining work

- None required for the Japanese-only benchmark as scoped; it is complete
  and the results above are real, not placeholders.
- Possible future extensions, not started: vocal/instrumental source
  separation before scoring; dialect-stratified analysis (currently
  `unassessed` for every row, by design); expanding past 20 speaker groups.

The 72-song corpus's full audio, LRCLIB lyric text, and predictions are
local-only and not included in the repository. The tracked
selection/exclusion manifests and lyrics-free metadata provide provenance for
every inclusion and exclusion decision.
