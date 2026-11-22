# Test-split reference/audio audit

How the 140 LRCLIB reference utterances in the frozen `test` split (15 songs,
4 speaker groups, 0.92 h) were checked against the audio before any system
was scored on them. All numbers below are read directly from
`data/private/ja_alignment.jsonl` and `data/private/ja_corpus_local.jsonl`
and are reproducible with `lyrics-asr-ja-align` + `lyrics-asr-ja-stats`;
nothing here is hand-typed.

## Why this is needed

LRCLIB timestamps are relative to the lyric database's own reference
recording, not to whatever YouTube upload this project downloaded. A music
video frequently prepends an intro the reference does not count, so
`lyric_time + offset = audio_time` for some per-song `offset` that is not
knowable in advance. Scoring without correcting for it would hold the wrong
~20 s of audio against a lyric line's reference text and report that as
model error.

## Method

1. **Probe.** `Systran/faster-whisper-base` (`ebe41f70`) decodes the first
   150 s of the song and returns timestamped segments. It never sees the
   lyric text, so this step cannot leak lexical information into the
   benchmark.
2. **Grid search.** A 0.5 s step search finds the offset that maximizes
   agreement between each LRCLIB line's window and the audio beneath it,
   scored by overlap-weighted soft character F1 (partial overlap contributes
   partial credit; without that weighting the objective is flat for several
   seconds and the search stops on the plateau).
3. **Confidence gate.** If the best score is below **0.35**, the estimate is
   not trusted. The offset instead falls back to the **duration prior**,
   `max(0, video_duration − reference_duration)` — the typical length of an
   added intro. **No song is ever dropped for a low score.** The frozen test
   split's membership is fixed before this step runs and stays independent
   of it; a low-confidence song is flagged in the manifest, not removed.

## Per-song result, test split (15 songs)

Sorted by confidence, lowest first:

| clip_id | artist | confidence | method | offset (s) | video−ref delta (s) | windows |
|---|---|---:|---|---:|---:|---:|
| genius-3003020 | Aimer | 0.231 | duration_prior_fallback | +0.0 | +0.0 | 12 |
| genius-2930936 | Aimer | 0.259 | duration_prior_fallback | +0.0 | +0.0 | 13 |
| genius-12856071 | King Gnu | 0.270 | duration_prior_fallback | +22.0 | +22.0 | 9 |
| genius-9389282 | King Gnu | 0.289 | duration_prior_fallback | +0.0 | +0.0 | 10 |
| genius-7396632 | Vaundy | 0.355 | asr_grid_search | +6.0 | +16.0 | 9 |
| genius-7525914 | Vaundy | 0.386 | asr_grid_search | −2.5 | −1.0 | 9 |
| genius-11011149 | Official髭男dism | 0.387 | asr_grid_search | −0.5 | +0.0 | 11 |
| genius-4427549 | King Gnu | 0.397 | asr_grid_search | −2.5 | −1.0 | 10 |
| genius-5114680 | King Gnu | 0.404 | asr_grid_search | +5.0 | +14.0 | 7 |
| genius-6582902 | Official髭男dism | 0.412 | asr_grid_search | −0.5 | −2.0 | 10 |
| genius-3003007 | Aimer | 0.448 | asr_grid_search | −1.0 | +2.0 | 8 |
| genius-5067991 | Vaundy | 0.475 | asr_grid_search | −0.5 | +1.0 | 7 |
| genius-3002994 | Aimer | 0.501 | asr_grid_search | +13.0 | +14.0 | 8 |
| genius-4873539 | Official髭男dism | 0.528 | asr_grid_search | −0.5 | +2.1 | 11 |
| genius-7809295 | Official髭男dism | 0.534 | asr_grid_search | −0.5 | −1.0 | 8 |

**11 of 15** songs clear the 0.35 confidence gate and use the searched
offset; **4 of 15** (2 Aimer, 2 King Gnu) fall back to the duration prior.
**12 of 15** have a nonzero offset applied at all — a song with a 0 s delta
and a video upload that matches the reference recording exactly needs no
correction, and the search/fallback both correctly return ≈0 for those.

### Worked example: a fallback case

King Gnu "AIZO" (`genius-12856071`): reference 216 s, video 238 s, a +22 s
delta consistent with an intro. The grid search's best score (0.270) is below
the 0.35 gate, so the offset used is the **duration-prior fallback**, +22.0 s
— which happens to equal the raw delta here. This is reported in the main
README as the canonical low-confidence example; it is reproduced here with
the actual confidence score for traceability.

### Worked example: a searched case

Aimer "カタオモイ" (`genius-3002994`): reference 207 s, video 221 s, a +14 s
delta. The grid search clears the gate at confidence 0.501 and returns
+13.0 s — close to, but not identical to, the raw duration delta, which is
expected: the delta is only a prior for where to start searching, not the
answer itself.

## Handling missing lines, uncovered audio, and chorus repeats

**No test-split utterance is manually corrected, patched, or dropped for a
content mismatch.** Three structural choices do the work instead, and all
three are applied uniformly — they do not single out hard cases:

1. **Sync coverage is reported, not assumed.** `sync_coverage` (LRCLIB's
   synced-lyric timespan ÷ video duration) ranges from **0.839 to 0.995**
   across the 15 test songs (mean 0.909). The remaining 0.5–16 % of each
   song's duration is intro, outro, or an instrumental passage no LRCLIB line
   claims to describe. That audio is never cut into a clip and never scored
   — there is no reference to score it against, so silently including it
   would either require inventing a reference or charging the model for
   audio it was never asked about. Both are avoided.
2. **A 5 s silence gap closes the current utterance window.** If two
   consecutive LRCLIB lines are more than 5 s apart, segmentation treats
   everything between them as unreferenced (most commonly an instrumental
   break or a long ad-lib) and starts a new window after the gap, rather
   than stretching one clip across it. This is what keeps instrumental
   breaks out of scored audio without needing a per-song manual boundary.
3. **A repeated chorus is scored exactly as many times as LRCLIB's
   `syncedLyrics` timestamps it, not as many times as it is actually sung.**
   When a lyric service collapses a repeated chorus into a single timestamped
   occurrence (a common shorthand), the later repeat in the audio is
   unreferenced audio and is handled by rule 1 above — it falls outside
   `sync_coverage` and is never cut or scored. When `syncedLyrics` *does*
   give the repeat its own timestamp, it is cut and scored like any other
   line. Which case applies is a property of the specific LRCLIB entry, not
   a judgment call made per song; the project does not special-case any
   track's chorus structure by hand.

Combined, these three rules are the entire explanation for the *source → used*
audio gap reported in [`README.md` → Split statistics](../README.md#split-statistics):
on the test split, 1.05 h of source audio yields 0.92 h of scored audio (88%) —
the other 12% is exactly the uncovered audio described above, and it is
excluded from scoring by construction rather than filtered after the fact.

## What this audit does not claim

- It does not verify that LRCLIB's lyric *text* matches what is actually sung
  in this specific release (ad-libs, alternate takes) — that is a separate,
  acknowledged limitation (see `README.md` → Limitations,
  "Reference quality is an assumption, not a verification").
- It verifies **timing** alignment (which audio span a line's text is scored
  against), not reference *correctness*.
- The 0.35 confidence threshold and 5 s gap/12 s cap constants are the
  project's own choices, not externally validated; they are documented here
  and in `src/lyrics_asr/ja_align.py` / `ja_segment.py` so they are checkable
  and changeable, not implicit.
