# Cleaned dataset manifests

Japanese-only. `lyrics-asr-purge-english` is the single reproducible place where
that constraint is enforced; English rows are removed rather than filtered on
read.

## Files

- `genius_candidates_cleaned.jsonl` — 630 machine-filtered Japanese candidates.
  `metadata_review_status` still requires review of candidate identity. Pageview
  terciles are a preliminary relative popularity signal only; missing counts and
  tiers need curation. The declared `reference_policy: lrclib_assumed_gold` is
  the study assumption, not a lyric-level verification claim.
- `lrclib_matches.jsonl` — 539 LRCLIB matches as a lyrics-free index (IDs,
  title/artist, duration, availability flags, retrieval time). No lyric text.
- `lrclib_pending.jsonl` — 90 unresolved candidates, reconciled so that every row
  is also present in the cleaned candidate list.
- `ja_corpus.jsonl` — the corpus actually used: 72 songs, 20 speaker groups, with
  the seed-42 speaker-disjoint split and every quality-gate measurement
  (`japanese_script_ratio`, `synced_line_count`, `sync_coverage`,
  `source_duration_delta_seconds`). No local paths, no lyric text.
- `ja_corpus_excluded.jsonl` — all 36 excluded songs with an explicit
  `exclusion_reason`, so inclusion decisions are auditable rather than implicit.
- `benchmark_selection.jsonl` / `benchmark_video_sources.jsonl` — the 35 Japanese
  rows surviving from the earlier bilingual pilot. The 9 test songs here are the
  frozen test songs whose zero-shot baselines were already computed; the corpus
  builder pins their speakers into `test`.
- `japanese_benchmark_selection.jsonl` — the 100-song Japanese expansion pool the
  corpus builder draws from.

## Rules

The manifests and their provenance belong in the repository; lyrics and audio do
not. Only commit media if redistribution is authorized — in that case update
`.gitignore` intentionally and document the license and source. Otherwise keep
them in local/private storage and keep stable paths and IDs in the manifest.

Document every inclusion/exclusion, language correction, popularity-tier
decision, and LRCLIB match provenance. Run `lyrics-asr-validate` before splitting
and again with `--require-split` afterwards. Regenerate the corpus and its split
with `lyrics-asr-ja-corpus`, which is deterministic at seed 42.

The full-corpus 80/10/10 split remains future work; `ja_corpus.jsonl` is the
corpus in use, split 62/20/18 by song count so the frozen 15-song test split and
a usable 13-song development split both fit.
