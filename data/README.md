# Dataset files

**This corpus is Japanese-only.** English rows are not filtered at read time —
they are absent. The purge is
reproducible via `lyrics-asr-purge-english`, which drops any row whose language
signal names `en` and never names `ja`, reconciles the LRCLIB pending list
against the surviving candidates, and deletes the corresponding local audio.
Bilingual rows that name `ja` are kept, because the label is the sung language.

Keep raw Genius candidate metadata and cleaned metadata manifests in version
control. Audio, cut clips, and LRCLIB lyric text are excluded by default through
`.gitignore`. A public endpoint is technically accessible, but that does not by
itself grant permission to redistribute its lyrics.

## LRCLIB is the annotation source

Every reference in this dataset is annotated from LRCLIB, and nothing else. The
project treats LRCLIB `plainLyrics` as the assumed gold reference and uses
`syncedLyrics` for timing:

- `text` — the reference transcript, from `plainLyrics` (or `syncedLyrics` with
  the line timestamps stripped).
- `synced_lyrics` — the timestamped lines, which are what make utterance-level
  segmentation possible.
- `reference_policy` / `reference_status` — always `lrclib_assumed_gold`.
- `lrclib_id`, `lrclib_url`, `retrieved_at_utc` — provenance for every row.

Do not add ad-libs manually, and do not edit a lyric to fit an ASR hypothesis.
LRCLIB may omit performed ad-libs or describe a different release of the same
song; both are reported as limitations rather than silently patched. Never use
YouTube's existing captions as a reference or as a system prediction.

Two LRCLIB-specific quality gates matter enough to state here:

1. **Script check.** LRCLIB sometimes stores a romanized or translated version of
   a Japanese lyric. Scoring kana/kanji ASR output against romaji would report a
   near-total error rate for a reference defect, so a reference whose script is
   mostly Latin is excluded (`japanese_script_ratio` ≥ 0.60).
2. **Timing check.** A row needs enough timestamped lines, and a lyric timespan
   that plausibly covers the track, before it can be segmented.

Store language and dialect separately. Language is assigned from song/reference
metadata; dialect requires listening to the sung voice. Until reviewed, `dialect`
stays `unassessed` and `dialect_status` stays `requires_audio_review` rather than
being guessed from an artist's birthplace or name.

## Layout

| Path | Tracked | Contents |
|---|:---:|---|
| `raw/genius_candidates.jsonl` | yes | Japanese Genius discovery metadata (1,713 rows) |
| `cleaned/genius_candidates_cleaned.jsonl` | yes | Filtered candidates (630) |
| `cleaned/lrclib_matches.jsonl` | yes | LRCLIB match index, **no lyric text** (539) |
| `cleaned/lrclib_pending.jsonl` | yes | Unresolved LRCLIB candidates (90) |
| `cleaned/ja_corpus.jsonl` | yes | The Japanese corpus and its seed-42 split (72 songs) |
| `cleaned/ja_corpus_excluded.jsonl` | yes | Every excluded song with its reason (36) |
| `references/lrclib_lyrics.jsonl` | no | LRCLIB plain + synced text |
| `audio/` | no | Full-song WAVs and yt-dlp provenance |
| `clips/ja/` | no | Cut 16 kHz mono utterance clips |
| `private/` | no | Utterance manifest, alignment offsets, predictions |

Use UTF-8 JSONL, one row per record. A song row in `ja_corpus.jsonl` carries IDs,
URLs, speaker/recording groups, the chosen video source with its duration delta,
the quality-gate measurements, and the split — but no local paths and no lyric
text. An utterance row in `private/ja_utterances.jsonl` adds the clip path, the
audio window, the applied alignment offset, and the reference text:

```json
{"clip_id":"genius-123-u03","song_id":"genius-123","genius_id":123,"lrclib_id":456,"language":"ja","language_status":"curated_from_song_and_reference_metadata","dialect":"unassessed","dialect_status":"requires_audio_review","speaker_group":"vocalist-id","recording_group":"genius-123","audio_path":"data/clips/ja/genius-123/genius-123-u03.wav","source_audio_path":"data/audio/ja/VIDEOID.wav","lyric_start_seconds":57.6,"audio_start_seconds":71.6,"duration_seconds":27.4,"alignment_offset_seconds":14.0,"text":"...","reference_policy":"lrclib_assumed_gold","split":"test"}
```

`lyrics-asr-fetch-lrclib` retrieves `plainLyrics` (or derives plain text from
`syncedLyrics`) into the gitignored `references/` directory using LRCLIB's public
API and an identifying User-Agent. Matches are accepted only when title
similarity and artist metadata meet the configured threshold; unresolved
candidates are recorded separately for review. Keep lyrics local unless
redistribution permission is clear.
