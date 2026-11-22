# Japanese Sung-Lyrics ASR: Dataset and Benchmark

This project builds a reproducible evaluation set for **ASR on singing**, and benchmarks several already-trained ASR systems on it. Genius is used to discover candidate songs; **LRCLIB is the sole annotation source** for the lyric references. Every system is scored on the same frozen clips, the same references, and the same normalization.

> **Japanese-only scope.** This benchmark covers Japanese-language songs only. English rows are not filtered at read time — they have been removed from every tracked and local file. See [Japanese-only enforcement](#japanese-only-enforcement).

---

## Table of contents

- [What this benchmark measures](#what-this-benchmark-measures)
- [Japanese-only enforcement](#japanese-only-enforcement)
- [LRCLIB as the annotation source](#lrclib-as-the-annotation-source)
- [Source attribution and licensing](#source-attribution-and-licensing)
- [Corpus construction](#corpus-construction)
- [Reference/audio alignment](#referenceaudio-alignment)
- [Test-split reference audit](docs/test_reference_audit.md)
- [The evaluation unit: utterances, not whole songs](#the-evaluation-unit-utterances-not-whole-songs)
- [Metrics: CER and Character F1](#metrics-cer-and-character-f1)
- [Systems benchmarked](#systems-benchmarked)
- [Models excluded for lacking Japanese support](#models-excluded-for-lacking-japanese-support)
- [Split statistics](#split-statistics)
- [Results](#results)
- [Running the benchmark on a free Colab T4](#running-the-benchmark-on-a-free-colab-t4)
- [Reproducing everything](#reproducing-everything)
- [Limitations](#limitations)
- [Tests and CI](#tests-and-ci)

---

## What this benchmark measures

Can general-purpose and Japanese-specialised ASR systems transcribe **sung** Japanese lyrics from previously unheard singers? The corpus is deliberately hard: real commercial recordings with instrumental backing, melodic delivery, and studio processing — conditions these models were not primarily trained on.

Three design choices make the numbers mean something:

1. **Speaker-disjoint splits.** No vocalist or recording appears in more than one split, so test performance estimates generalization to unseen voices rather than memorized singers.
2. **Utterance-level scoring.** Each scored item is a ~28-second window whose reference is the lyric actually sung in that window, recovered from LRCLIB timestamps. Scoring a 4-minute song against a whole lyric sheet conflates mistranscription with never having been told the song's structure.
3. **Rule-based corpus selection.** Every inclusion and exclusion is produced by a documented gate and recorded with a reason, before any system runs.

## Japanese-only enforcement

`lyrics-asr-purge-english` is the single reproducible place where the constraint lives. A row is English when its language signal names `en` and never names `ja`; rows that name `ja` are kept, including bilingual ones, because the label is the sung language. Rerunning it on the current tree removes nothing, and CI asserts exactly that.

| File | Before | After |
|---|---:|---:|
| `data/raw/genius_candidates.jsonl` | 2,807 | 1,713 |
| `data/cleaned/genius_candidates_cleaned.jsonl` | 1,561 | 630 |
| `data/cleaned/lrclib_matches.jsonl` | 1,420 | 539 |
| `data/cleaned/lrclib_pending.jsonl` | 141 | 90 |
| `data/cleaned/benchmark_selection.jsonl` | 63 | 35 |
| `data/references/lrclib_lyrics.jsonl` | 1,420 | 539 |
| `config/searches.json` queries | 64 | 44 |
| Local full-song WAVs | 64 | 35 → 72 after expansion |

4,146 JSONL rows, 20 search queries, 12 selection groups and 29 English audio files were removed. The earlier bilingual Whisper results were discarded; only the 9 Japanese test songs' cached predictions were kept (see [Results](#results)).

Because English rows were removed from the raw Genius snapshot too, that file no longer reproduces the original bilingual discovery run. That is a deliberate consequence of the constraint, recorded here rather than hidden.

## LRCLIB as the annotation source

Every reference is annotated from LRCLIB and nothing else. The project treats LRCLIB `plainLyrics` as the **assumed gold** reference and uses `syncedLyrics` for timing. Each row preserves `lrclib_id`, `lrclib_url`, and `retrieved_at_utc`, and declares `reference_policy: lrclib_assumed_gold`.

Lyrics are never hand-edited to fit a hypothesis, ad-libs are never added, and YouTube's own captions are never used — not as a reference and not as a prediction.

Two LRCLIB-specific defects are screened out, because both would otherwise be reported as model error:

- **Romanized or translated references.** LRCLIB sometimes stores a Japanese song's lyric in Latin script. Scoring kana/kanji output against romaji yields a near-total error rate caused entirely by the reference. The gate `japanese_script_ratio ≥ 0.60` removed **23 songs** — including YOASOBI "Into the Night", Utada Hikaru "Sanctuary" and Vaundy "怪獣の花唄", all of which had a romanized LRCLIB entry.
- **Unusable timing.** A row needs ≥ 8 timestamped lines and a lyric timespan that plausibly covers the track (0.55–1.08 of duration) before it can be segmented.

## Source attribution and licensing

No song audio and no full lyric text is tracked by git, anywhere in this repository, for any song — see [Reproducing everything → Data layout](#data-layout) for the exact gitignored paths (`data/audio/`, `data/clips/`, `data/references/`, `data/private/`, `colab_handoff/colab_bundle.zip`, `colab_handoff/final_predictions/`). Only metadata is tracked: video/LRCLIB IDs, derived counts, and code. `git add -A --dry-run` is checked before every push to confirm no gitignored path slips through.

**Every one of the 72 source videos was checked individually against YouTube's own license metadata** (`yt_dlp`, metadata-only — no video or audio was downloaded to run this check). Result: **all 72 are Standard YouTube License (All Rights Reserved)**. **Zero are Creative Commons.** This is expected for commercial J-pop/J-rock, and it is the reason nothing but identifiers is ever committed — the corpus is fully reproducible from `data/manifest.csv` plus `lyrics-asr-download-audio` / `lyrics-asr-fetch-lrclib`, but the media itself never passes through this repository or its Git history.

The "Owner" column is the uploading YouTube channel at check time, not a legal determination of copyright ownership — most are the artist's/label's official channel; a few are third-party uploads used when no official-channel source existed within this project's duration-matching tolerance (see [Corpus construction](#corpus-construction)). All 72 remain Standard YouTube License regardless of uploader. The same table is tracked machine-readably at `data/manifest.csv`, and is duplicated in `LICENSE` so the licensing claim travels with the code even if this README changes.

| Split | Artist | Title | Owner (YouTube channel) | YouTube video ID | LRCLIB ID | License |
|---|---|---|---|---|---:|---|
| train | Ado | うっせぇわ | Ado | `Qp3b-RXtz4w` | 7351545 | Standard YouTube License |
| train | Ado | 新時代 | NL77 | `Yku9pi8oDK0` | 7333157 | Standard YouTube License |
| train | Aimyon | ざらめ | あいみょん | `WLCw46JqbLk` | 11175590 | Standard YouTube License |
| train | Aimyon | ハルノヒ | あいみょん | `pfGI91CFtRg` | 5170613 | Standard YouTube License |
| train | Aimyon | マリーゴールド | JM GACK | `dEDyNfhA0Xs` | 3014963 | Standard YouTube License |
| train | Aimyon | 愛を伝えたいだとか | あいみょん | `9qRCARM_LfE` | 3015025 | Standard YouTube License |
| train | Aimyon | 裸の心 | あいみょん | `yOAwvRmVIyo` | 14568975 | Standard YouTube License |
| train | Mrs. GREEN APPLE | Magic | AOG Music | `nacrCj-Swj8` | 7235177 | Standard YouTube License |
| train | Mrs. GREEN APPLE | Variety | Mrs. GREEN APPLE | `HlrQ3ZTdN78` | 38948990 | Standard YouTube License |
| train | Mrs. GREEN APPLE | 慶びの種 | Mrs. GREEN APPLE | `t_H7arA7OvM` | 31235787 | Standard YouTube License |
| train | Mrs. Green Apple | Soranji | Mrs. GREEN APPLE | `GNQyYTdJ4l8` | 3871576 | Standard YouTube License |
| train | Superfly | My Best Of My Life | Superfly | `_Q-QFSJQ580` | 3226902 | Standard YouTube License |
| train | Superfly | 愛をこめて花束を | Superfly | `gU5oN0KVofU` | 1071526 | Standard YouTube License |
| train | Superfly | 覚醒 | Superfly | `7u052cW1e1Y` | 1051947 | Standard YouTube License |
| train | YOASOBI | Idol | YOASOBI | `ZRtdQ81jPUQ` | 8022411 | Standard YouTube License |
| train | YOASOBI | 夜に駆ける | YOASOBI | `by4SYYWlhEs` | 7058562 | Standard YouTube License |
| train | YOASOBI | 怪物 | YOASOBI | `dy90tA3TT1c` | 7907369 | Standard YouTube License |
| train | YOASOBI | 群青 | YOASOBI | `Y4nEEZwckuU` | 5557410 | Standard YouTube License |
| train | ずっと真夜中でいいのに。 | TAIDADA | ずっと真夜中でいいのに。 ZUTOMAYO | `IeyCdm9WwXM` | 30198989 | Standard YouTube License |
| train | ずっと真夜中でいいのに。 | あいつら全員同窓会 | ずっと真夜中でいいのに。 ZUTOMAYO | `ZUwaudw8ht0` | 3431920 | Standard YouTube License |
| train | ずっと真夜中でいいのに。 | お勉強しといてよ | ずっと真夜中でいいのに。 ZUTOMAYO | `Atvsg_zogxo` | 3431927 | Standard YouTube License |
| train | ずっと真夜中でいいのに。 | 不法侵入 | ずっと真夜中でいいのに。 ZUTOMAYO | `SAdkxVFyAyc` | 3595876 | Standard YouTube License |
| train | ずっと真夜中でいいのに。 | 勘冴えて悔しいわ | ずっと真夜中でいいのに。 ZUTOMAYO | `4QePrv24TBU` | 3595882 | Standard YouTube License |
| train | キタニタツヤ | いらないもの | キタニタツヤ / Tatsuya Kitani | `_L_M8kIynhE` | 21085075 | Standard YouTube License |
| train | キタニタツヤ | ずうっといっしょ! | 종따에요 (エヨ) | `4R8u7FVPwrM` | 22558635 | Standard YouTube License |
| train | キタニタツヤ | 次回予告 | キタニタツヤ / Tatsuya Kitani | `nAxSTbugJ_U` | 8989272 | Standard YouTube License |
| train | キタニタツヤ | 私が明日死ぬなら | キタニタツヤ / Tatsuya Kitani | `P8ODCzh51WE` | 5668082 | Standard YouTube License |
| train | キタニタツヤ | 青のすみか | Rad1o Music | `Nm31tVsq-ZQ` | 5668099 | Standard YouTube License |
| train | ヨルシカ | ただ君に晴れ | ヨルシカ / n-buna Official | `-VKIqrvVOpo` | 7330892 | Standard YouTube License |
| train | ヨルシカ | ヒッチコック | ヨルシカ / n-buna Official | `t7MBzMP4OzY` | 3566889 | Standard YouTube License |
| train | ヨルシカ | 晴る | Jamong | `uF7eT3nhyZ0` | 5376018 | Standard YouTube License |
| train | ヨルシカ | 花に亡霊 | ヨルシカ / n-buna Official | `9lVPAWLWtWc` | 7331096 | Standard YouTube License |
| train | ヨルシカ | 言って。 | ヨルシカ / n-buna Official | `F64yFFnZfkI` | 7331106 | Standard YouTube License |
| train | 宇多田ヒカル | Passion | Hikaru Utada | `j01KLlxfoN4` | 2403225 | Standard YouTube License |
| train | 宇多田ヒカル | 誓い | HTMSD | `Xpy3pGXg5CU` | 31319869 | Standard YouTube License |
| train | 米津玄師 | Azalea | World of Japan | `EDJFgBdAy5w` | 17980888 | Standard YouTube License |
| train | 米津玄師 | IRIS OUT | Kenshi Yonezu 米津玄師 | `ux3QETpLcPs` | 30754808 | Standard YouTube License |
| train | 米津玄師 | KICK BACK | Kenshi Yonezu 米津玄師 | `M2cckDmNLMI` | 1491439 | Standard YouTube License |
| train | 米津玄師 | Pale Blue | Kenshi Yonezu 米津玄師 | `7WZ1Kt3zraY` | 3122057 | Standard YouTube License |
| train | 米津玄師 | 地球儀 | Kenshi Yonezu 米津玄師 | `VUsURj_OYdA` | 5299681 | Standard YouTube License |
| train | 藍井エイル | IGNITE | Eir Aoi Official YouTube Channel | `hoxUfOTE5II` | 7967405 | Standard YouTube License |
| train | 藍井エイル | Lament | Eir Aoi Official YouTube Channel | `652DpTJ0g30` | 6864582 | Standard YouTube License |
| train | 藍井エイル | MEMORIA | angel heart | `GZ7g_my5OaY` | 2926296 | Standard YouTube License |
| train | 藍井エイル | 鼓動 | Eir Aoi Official YouTube Channel | `FLuFszuT8xM` | 6724324 | Standard YouTube License |
| validation | ClariS | コネクト | ClariS Official YouTube Channel | `7EuTPTVpuNI` | 5833592 | Standard YouTube License |
| validation | ClariS | ヒトリゴト | ClariS Official YouTube Channel | `DVt13wPJqmw` | 4810519 | Standard YouTube License |
| validation | Fujii Kaze | grace | Fujii Kaze | `rSYoIuyks8g` | 3871589 | Standard YouTube License |
| validation | Fujii Kaze | まつり | Fujii Kaze | `NwOvu-j_WjY` | 3429465 | Standard YouTube License |
| validation | Fujii Kaze | 死ぬのがいいわ | ルビーChan | `63jl1n5mN30` | 2739687 | Standard YouTube License |
| validation | Reol | ゆーれいずみー | Reol Official | `3mUXwYx1CQs` | 3092809 | Standard YouTube License |
| validation | Reol | ヒビカセ | Reol Official | `TkroHwQYpFE` | 2695447 | Standard YouTube License |
| validation | Reol | 十中八九 | Reol Official | `8PiCsyCexRA` | 4143297 | Standard YouTube License |
| validation | いよわ | きゅうくらりん | いよわ/ iyowa | `2b1IexhKPz4` | 24023279 | Standard YouTube License |
| validation | れをる | アシンメトリー | Reol Official | `sz-yVfChwNU` | 2406256 | Standard YouTube License |
| validation | 稲葉曇 | リレイアウター (Relayouter) | inabakumori | `b56xjtP6Qac` | 30430688 | Standard YouTube License |
| validation | 稲葉曇 | ロストアンブレラ | inabakumori | `DeKLpgzh-qQ` | 31916154 | Standard YouTube License |
| validation | 稲葉曇 | 私は雨 | inabakumori | `EEk4JGzqoFg` | 18370205 | Standard YouTube License |
| test | Aimer | Brave Shine | Aimer Official YouTube Channel | `XeI8E20ZUE4` | 7274081 | Standard YouTube License |
| test | Aimer | RE:I AM | Aimer Official YouTube Channel | `ptwH69_0EAo` | 7231408 | Standard YouTube License |
| test | Aimer | StarRingChild | Garden of Melody | `D16AGQQ0tdM` | 7826774 | Standard YouTube License |
| test | Aimer | カタオモイ | Aimer Official YouTube Channel | `kxs9Su_mbpU` | 8138770 | Standard YouTube License |
| test | King Gnu | AIZO | King Gnu official YouTube channel | `zz2a9Q2Wru0` | 30253574 | Standard YouTube License |
| test | King Gnu | SPECIALZ | King Gnu official YouTube channel | `fhzKLBZJC3w` | 8426482 | Standard YouTube License |
| test | King Gnu | どろん | King Gnu official YouTube channel | `gc0_Acq8dV4` | 5815949 | Standard YouTube License |
| test | King Gnu | 白日 | King Gnu official YouTube channel | `Tll_g1Gx7I8` | 2738272 | Standard YouTube License |
| test | Official髭男dism | 50% | OFFICIAL HIGE DANDISM | `ea-WMqlmZ1E` | 17980804 | Standard YouTube License |
| test | Official髭男dism | Cry Baby | OFFICIAL HIGE DANDISM | `O1bhZgkC4Gw` | 2521639 | Standard YouTube License |
| test | Official髭男dism | ミックスナッツ | mysticholy | `Wf8XuAoCjN8` | 4744403 | Standard YouTube License |
| test | Official髭男dism | 宿命 | OFFICIAL HIGE DANDISM | `-kgOFJG881I` | 6658846 | Standard YouTube License |
| test | Vaundy | 不可幸力 | Vaundy | `Gbz2C2gQREI` | 4970187 | Standard YouTube License |
| test | Vaundy | 裸の勇者 | Vaundy | `FT0GKCuSaW0` | 5206919 | Standard YouTube License |
| test | Vaundy | 踊り子 | Vaundy | `7HgJIAUtICU` | 5206921 | Standard YouTube License |

`data/manifest.csv` carries the same rows plus `utterances` and `duration_seconds` per song, frozen at the point this table was generated.

## Corpus construction

`lyrics-asr-ja-corpus` is deterministic at **seed 42** and takes the union of the 100-song Japanese expansion pool and the surviving Japanese rows of the earlier pilot — 108 candidate songs over 22 speaker groups — then applies:

| Gate | Rule | Why |
|---|---|---|
| Language | `language == "ja"` | project scope: Japanese only |
| Reference script | `japanese_script_ratio ≥ 0.60` | rejects romanized/translated LRCLIB entries |
| Synced lines | `≥ 8` timestamped non-empty lines | needed to cut utterances |
| Sync coverage | lyric timespan / duration in `[0.55, 1.08]` | rejects implausible timing |
| Source title | blocklist: cover, live, karaoke, instrumental, remix, THE FIRST TAKE, acoustic, … | the reference describes the studio recording |
| Source duration | `|video − LRCLIB| ≤ 45 s` | rejects different edits; residual offset is corrected, not assumed away |
| Source preference | official/artist channel, then smallest duration delta | best available provenance |
| Speaker cap | 5 songs per speaker (4 for pinned test speakers) | stops one vocalist dominating; keeps test in the 15–18 song budget |

All **36 exclusions** are written to `data/cleaned/ja_corpus_excluded.jsonl` with a machine-readable `exclusion_reason`, so inclusion decisions are auditable:

| Reason | Songs |
|---|---:|
| LRCLIB stored a romanized/translated reference | 23 |
| Over the per-speaker song cap | 9 |
| Implausible synced-lyric coverage | 2 |
| Too few timestamped lines | 1 |
| No studio-recording source within the duration rule | 1 |

The result is **72 songs over 20 speaker groups**, split 62/20/18 by song count.

### Preserving the frozen test set

Four speaker groups — `aimer`, `king-gnu`, `satoshi-fujihara`, `vaundy` — are **pinned into test**. They are the speakers of the 9 Japanese songs whose zero-shot baselines were already computed, so pinning them preserves that cached result instead of discarding it. Seed 42 then assigns every remaining speaker group to train or validation. `lyrics-asr-split --pin-test-speakers` implements the rule generically and raises an error on an unknown speaker rather than silently ignoring it.

Speaker/recording groups are disjoint across all three splits, verified by `lyrics-asr-validate --require-split` and by unit test.

## Reference/audio alignment

LRCLIB timestamps are relative to the released master. A music video for the same recording often prepends an intro, so lyric time `t` corresponds to audio time `t + offset`. Uncorrected, a cut utterance would hold the wrong lyric: in this corpus 19 of 72 sources differ from the LRCLIB duration by more than 5 s, up to +45 s.

`lyrics-asr-ja-align` estimates **one scalar per song** in two stages, so the expensive part is paid once:

1. **Probe.** `Systran/faster-whisper-base` (`ebe41f70`) decodes the first 150 s of each song and stores timestamped segments. It is never shown the lyric.
2. **Search.** A 0.5 s grid search finds the offset maximizing agreement between the lyric windows and the audio beneath them, scored by overlap-weighted soft character F1. Partially overlapping segments contribute in proportion to their overlap — without that weighting the objective is flat across several seconds and the search stops early on the plateau.

When confidence falls below 0.35 the estimate is not trusted and the offset falls back to the duration prior `max(0, video − reference)`, the usual length of an added intro. **No song is ever dropped on the basis of this score**, so the frozen test split stays independent of any model's accuracy. Only a single number per song comes from this pass, which is why using an ASR model here does not leak lexical information into the benchmark.

Worked examples: Aimer "カタオモイ" has a +14 s duration delta and the search recovered +13.0 s; King Gnu "白日" has a −1 s delta and the search returned −2.5 s; King Gnu "AIZO" (+22 s delta) scored below threshold and fell back to the +22 s prior.

For the full per-song confidence/offset table on the frozen `test` split, and how missing lines, instrumental gaps, and chorus repeats are handled during scoring, see [`docs/test_reference_audit.md`](docs/test_reference_audit.md).

## The evaluation unit: utterances, not whole songs

Each utterance is a window of at most ~28 s, built by merging consecutive LRCLIB lines, shifted by the song's offset, and cut to 16 kHz mono WAV. A line's span is capped at 12 s and a gap over 5 s closes the window, so instrumental passages are not charged to the preceding line and never enter the cut audio.

This also matches the window Whisper was trained on (30-second clips), so no system is penalized for a context length it never saw in training. Because uncovered audio is skipped, the corpus reports both the **source duration** and the smaller **used duration** — see [Split statistics](#split-statistics).

## Metrics: CER and Character F1

WER is **not** reported. Japanese is written without spaces, so there is no uncontroversial word tokenization, and a word-level error rate would end up measuring the choice of segmenter as much as the model itself. CER sidesteps the problem by operating directly on characters, which is well defined regardless of segmentation.

- **CER** = (substitutions + deletions + insertions) / reference characters. Lower is better. It is an error *rate*, so it is unbounded above — a system that hallucinates long text can exceed 100%.
- **Character F1** is the harmonic mean of character precision and recall from the same edit alignment. It is bounded in [0, 1], and it charges hallucinated text to *precision* instead of folding everything into one rate. Reported **alongside** CER, never instead of it, because an alignment-based F1 is order-insensitive enough to hide some sequencing errors.

Aggregation is micro: counts are summed over utterances and then divided, so a long utterance counts in proportion to its length. Normalization is Unicode NFKC, case folding, punctuation removal and whitespace removal, applied identically to every system. Per-speaker and per-song tables are emitted too, since one hard song can dominate a 15-song split.

## Systems benchmarked

Seven systems, three tiers. Every one of them decodes the **same** clips with the **same** settings — `language="ja"`, `task="transcribe"`, beam 5, no VAD, `condition_on_previous_text=False` — so the checkpoint is the only variable in the results table. Revisions are pinned to exact commit SHAs, because Hugging Face branches move and a benchmark that cannot be tied to a commit cannot be reproduced.

| Tier | System | Checkpoint | Revision | Params | Family |
|---|---|---|---|---:|---|
| baseline | `faster-whisper-base` | [`Systran/faster-whisper-base`](https://huggingface.co/Systran/faster-whisper-base) | `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` | 74M | multilingual Whisper (CTranslate2) |
| baseline | `faster-whisper-small` | [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small) | `536b0662742c02347bc0e980a01041f333bce120` | 244M | multilingual Whisper (CTranslate2) |
| baseline | `faster-whisper-medium` | [`Systran/faster-whisper-medium`](https://huggingface.co/Systran/faster-whisper-medium) | `08e178d48790749d25932bbc082711ddcfdfbc4f` | 769M | multilingual Whisper (CTranslate2) |
| baseline | `faster-whisper-large-v3-turbo` | [`deepdml/faster-whisper-large-v3-turbo-ct2`](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` | 809M | Whisper large-v3-turbo, CTranslate2 conversion of [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) |
| japanese-pretrained | `kotoba-whisper-v2.0` | [`kotoba-tech/kotoba-whisper-v2.0-faster`](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) | `f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc` | 756M | **Japanese-only** distilled Whisper — distilled from large-v3 on [ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech) |
| japanese-pretrained | `anime-whisper` | [`litagin/anime-whisper`](https://huggingface.co/litagin/anime-whisper) | `22e2008a8182b357da3922a6308d095008f72973` | 756M | **Japanese-only** — `kotoba-whisper-v2.0` fine-tuned on ~5,300 h of expressive Japanese speech |
| contribution | `whisper-small-lora-ja` | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) + LoRA | trained here, seed 42 | 244M + ~7M adapter | our own LoRA fine-tune on the Japanese `train` split |

The three tiers answer three different questions, and keeping them apart is the point:

- **baseline** — how far does *scale alone* get a general multilingual model on sung Japanese? Four sizes spanning 74M → 809M, so the scaling trend is visible rather than asserted.
- **japanese-pretrained** — do *already-trained* Japanese-only checkpoints beat a larger multilingual one? Neither was fit by this project, so this tier is a clean read on language specialisation. `anime-whisper` is included because it is further fine-tuned on deliberately expressive, non-neutral Japanese delivery, which is closer to singing than read speech is.
- **contribution** — what does ~2.7 h of in-domain *sung* Japanese buy on top of `whisper-small`? LoRA (`r=32`, `alpha=64`, on `q_proj`/`v_proj`) trains ~7M parameters for four epochs, the epoch is chosen on `validation` (epoch 2 of 4, by lowest CER), and the chosen adapter decodes `test` exactly once.

A fair warning on the contribution tier, stated up front rather than discovered in the results: 2.7 h over 11 singers cannot out-train a model distilled on thousands of hours of Japanese, so `whisper-small-lora-ja` is **not** expected to top the table. The honest comparison is `whisper-small-lora-ja` against `faster-whisper-small` — same base checkpoint, same size, one of them adapted to sung Japanese. That pairwise delta is what the fine-tune actually measures, and it is why both the plain and the adapted `small` rows are reported.

## Models excluded for lacking Japanese support

Two candidate systems are **excluded rather than run under a wrong language setting**. Both model cards enumerate exactly 25 European languages and never mention Japanese (verified against the cards' YAML `language:` lists and full text):

| Model | Model card | Revision | Declared languages | Japanese? |
|---|---|---|---:|:---:|
| `nvidia/canary-1b-v2` | <https://huggingface.co/nvidia/canary-1b-v2> | `d455706339a6b32e1aa40f82c713a482a0c938e2` | 25 (bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk) | **no** |
| `nvidia/parakeet-tdt-0.6b-v3` | <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | 25 (en, es, fr, de, bg, hr, cs, da, nl, et, fi, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, sv, ru, uk) | **no** |

Reporting a Japanese score for a model that does not support Japanese would measure the mismatch, not the model. The exclusion, its reason, and both revisions are recorded in `config/benchmark_models.json` under `excluded_systems`.

## Split statistics

Regenerate everything below with `lyrics-asr-ja-stats`, which writes the full version to `reports/ja_corpus_stats.md` (+ `.jsonl`). Nothing here is hand-typed.

### Audio budget: exact measured hours

No planning estimate is reported next to a measured one — both rows below are measured directly from the tracked corpus and `data/manifest.csv`, not projected.

| Quantity | Hours | What it is |
|---|---:|---|
| Source | 5.07 | Full duration of the 72 songs that passed every corpus gate |
| Used (scored) | **4.33** | Duration actually cut into scored utterance windows (85% of source) |

36 songs were rejected by the corpus gates before reaching this count — most because LRCLIB stored a romanized or translated reference rather than kana/kanji, and the per-speaker cap dropped 9 more so no single vocalist dominates (see [Corpus construction](#corpus-construction)). The gap between *source* and *used* above is audio that no lyric line covers: intros, instrumental breaks and outros are never cut, so they are never scored against a reference that does not describe them.

### Splits

| Split | Speakers | Songs | Utterances | Source audio (h) | Used audio (h) | Used share | Mean utt (s) | Reference chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 11 | 44 | 408 | 3.17 | 2.68 | 85% | 23.7 | 25,351 |
| validation | 5 | 13 | 110 | 0.86 | 0.72 | 84% | 23.7 | 7,399 |
| test | 4 | 15 | 140 | 1.05 | 0.92 | 88% | 23.7 | 9,452 |
| **total** | **20** | **72** | **658** | **5.07** | **4.33** | **85%** | **23.7** | **42,202** |

### Speaker groups per split (disjoint by construction, seed 42)

| Split | Count | Speaker groups |
|---|---:|---|
| train | 11 | `ado`, `aimyon`, `eir-aoi`, `hikaru-utada`, `ikura-yoasobi`, `kenshi-yonezu`, `motoki-omori`, `suis-yorushika`, `superfly`, `tatsuya-kitani`, `zutomayo` |
| validation | 5 | `claris`, `fujii-kaze`, `kaai-yuki`, `kafu`, `reol` |
| test | 4 | `aimer`, `king-gnu`, `satoshi-fujihara`, `vaundy` |

The four test speaker groups are **pinned**: they are the vocalists of the 9 Japanese songs whose zero-shot baselines were already computed, so pinning them preserves that cached result instead of discarding it. Seed 42 assigns every other speaker group to train or validation.

### Exact train/test overlap

| Measure | Count | Share of test |
|---|---:|---:|
| Distinct lyric lines in test | 545 | — |
| …also appearing verbatim in train | 1 | 0.18% |
| Distinct full utterance texts in test | 139 | — |
| …also appearing verbatim in train | 0 | 0.00% |
| Distinct lyric lines in validation also in train | 0 | 0.00% |
| Speaker groups shared between train and test | 0 | 0.00% |
| Recording groups shared between train and test | 0 | 0.00% |

Overlap is measured after the same NFKC + case-fold + punctuation-removal + whitespace-removal normalization used for scoring. The single overlapping line is a 4-character phrase — a common fixed expression, which is the expected and benign source of line-level overlap. **No** full utterance text is shared, and no speaker or recording group crosses a split boundary. That last property is what the split actually protects: a clip-level split could put the same singer, the same mix, or the same repeated chorus on both sides and inflate the scores.

### Reference/audio alignment quality

| Split | Songs | Mean confidence | Confident (≥0.35) | Non-zero offset | Largest offset (s) |
|---|---:|---:|---:|---:|---:|
| train | 44 | 0.408 | 28/44 | 27 | 34.0 |
| validation | 13 | 0.379 | 9/13 | 3 | 16.0 |
| test | 15 | 0.392 | 11/15 | 7 | 22.0 |

Confidence is the overlap-weighted soft character F1 between a lyric window and the audio beneath it at the chosen offset. Below 0.35 the scalar offset falls back to the duration prior. **No song is removed on the basis of this score**, so the frozen test split stays independent of any model's accuracy.

### LRCLIB reference provenance

| Property | Value |
|---|---|
| Songs with an LRCLIB `plainLyrics`/`syncedLyrics` reference | 72/72 |
| Reference policy | `lrclib_assumed_gold` for every row |
| Mean Japanese-script ratio of references | 0.934 |
| Minimum Japanese-script ratio in the corpus (gate ≥ 0.60) | 0.62 |
| Mean synced-lyric coverage of track duration | 0.89 |
| Total timestamped lyric lines used | 3,486 |

## Results

The frozen test split is 140 utterances / 15 songs / 4 held-out speaker groups / 0.92 h of scored audio / 9,452 reference characters. Produced by `run_colab.py --stage all --device cuda` on a free-tier **Tesla T4** (`torch 2.14.1+cu130`, Python 3.13.16, `float16`); the full per-split/per-system record, including inference minutes and the exact environment, is archived in `final_metrics.json`. Independently re-scored from the raw predictions with the project's own `lyrics_asr.ja_evaluate` (not just the Colab-bundled standalone copy) to confirm these numbers before they went in this table.

| Tier | System | Checkpoint | Revision | CER ↓ | Character F1 ↑ | Precision | Recall | Exact | RTF |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `faster-whisper-large-v3-turbo` | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f75` | **34.90%** | **73.08%** | 74.21% | 71.98% | 5.00% | 0.03× |
| baseline | `faster-whisper-medium` | `Systran/faster-whisper-medium` | `08e178d4` | 36.34% | 71.83% | 71.64% | 72.02% | 2.14% | 0.06× |
| baseline | `faster-whisper-small` | `Systran/faster-whisper-small` | `536b0662` | 41.04% | 67.49% | 67.78% | 67.20% | 1.43% | 0.03× |
| contribution | `whisper-small-lora-ja` | `openai/whisper-small` + LoRA | `lora-r32-seed42-epoch2` | 42.28% | 66.20% | 68.86% | 63.74% | 1.43% | 0.03× |
| japanese-pretrained | `kotoba-whisper-v2.0` | `kotoba-tech/kotoba-whisper-v2.0-faster` | `f44edd35` | 49.16% | 60.55% | 64.51% | 57.05% | 2.14% | 0.03× |
| baseline | `faster-whisper-base` | `Systran/faster-whisper-base` | `ebe41f70` | 60.55% | 53.13% | 52.74% | 53.52% | 0.71% | 0.03× |
| japanese-pretrained | `anime-whisper` | `litagin/anime-whisper` | `22e2008a` | 61.64% | 52.92% | 53.69% | 52.17% | 0.00% | 0.02× |

Scale alone is the strongest effect in this table: the four baselines rank exactly by parameter count, and `large-v3-turbo` (809M) wins outright. Neither Japanese-only pretrained checkpoint beats the two largest multilingual baselines on sung audio — `kotoba-whisper-v2.0` lands mid-table and `anime-whisper` lands last, despite both being Japanese-specialised and `anime-whisper` specifically fine-tuned on expressive speech. Full per-utterance, per-song and per-speaker breakdowns are in `reports/ja_test_benchmark_*.csv`.

### The contribution, honestly

`whisper-small-lora-ja` is LoRA-adapted `whisper-small` (`r=32`, `alpha=64`, `q_proj`/`v_proj`, AdamW lr `5e-5`, batch 8, seed 42) trained for 4 epochs in **10.1 minutes** on the T4. Epoch **2 of 4** was selected by lowest `validation` CER (48.51%); epochs 3–4 drifted slightly worse (48.57%, 49.63%), which is exactly the overfitting risk flagged above for 2.7 h over 11 train speakers. The selected adapter decoded `test` exactly once — see `finetune_report.json` for the full epoch history.

The fair comparison is against `faster-whisper-small`, same base checkpoint and size, and the result is mixed rather than a clean win:

| Split | System | CER ↓ | Character F1 ↑ | Recall |
|---|---|---:|---:|---:|
| validation | `faster-whisper-small` | 51.22% | 57.39% | 57.48% |
| validation | `whisper-small-lora-ja` | **48.51%** | **59.04%** | 56.68% |
| test | `faster-whisper-small` | **41.04%** | **67.49%** | 67.20% |
| test | `whisper-small-lora-ja` | 42.28% | 66.20% | 63.74% |

The adapter improves both CER and Character F1 on `validation` (its own model-selection split) but is slightly worse than the unmodified base on `test` — on both splits it also has *lower* recall than the base model, which is the opposite of what more in-domain training data is usually expected to buy. The honest reading is that 2.7 h over 11 singers was enough to measurably shift the model, but not enough to generalize past the 5 validation speaker groups to the 4 held-out test ones. This is reported as the finding, not hidden behind the model-selection split.

Two numbers from an earlier pilot are **not** from this run, kept only as a sanity-check reference point, from the cached whole-song baselines on the 9 pinned test songs (whole-song scoring, not the utterance protocol above — see `reports/ja_legacy_whole_song_*`):

| System | Unit | CER ↓ | Character F1 ↑ |
|---|---|---:|---:|
| `faster-whisper-small` | whole song | 44.44% | 63.60% |
| `faster-whisper-base` | whole song | 67.20% | 44.83% |

Whole-song scoring charges a model for every instrumental bar and every unsung repeat, so these are not comparable to the utterance-level table above; the utterance protocol is the one the benchmark uses.

### Development-split numbers

`reports/ja_validation_benchmark_table.md` scores the same seven systems on the 110-utterance `validation` split (13 songs, 5 speaker groups, 0.72 h, 7,399 reference characters) — this is the split the LoRA epoch was selected on, so it is not a second held-out estimate:

| Tier | System | Checkpoint | Revision | CER ↓ | Character F1 ↑ | Precision | Recall | Exact | RTF |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | `faster-whisper-medium` | `Systran/faster-whisper-medium` | `08e178d4` | **40.11%** | **66.49%** | 67.29% | 65.71% | 0.91% | 0.07× |
| baseline | `faster-whisper-large-v3-turbo` | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f75` | 41.14% | 65.50% | 66.05% | 64.97% | 1.82% | 0.04× |
| contribution | `whisper-small-lora-ja` | `openai/whisper-small` + LoRA | `lora-r32-seed42-epoch2` | 48.51% | 59.04% | 61.59% | 56.68% | 0.00% | 0.04× |
| japanese-pretrained | `kotoba-whisper-v2.0` | `kotoba-tech/kotoba-whisper-v2.0-faster` | `f44edd35` | 50.44% | 57.65% | 61.66% | 54.13% | 0.91% | 0.03× |
| baseline | `faster-whisper-small` | `Systran/faster-whisper-small` | `536b0662` | 51.22% | 57.39% | 57.30% | 57.48% | 0.91% | 0.04× |
| japanese-pretrained | `anime-whisper` | `litagin/anime-whisper` | `22e2008a` | 64.24% | 49.79% | 51.70% | 48.02% | 0.91% | 0.02× |
| baseline | `faster-whisper-base` | `Systran/faster-whisper-base` | `ebe41f70` | 66.54% | 43.48% | 43.32% | 43.65% | 0.00% | 0.05× |

Note `faster-whisper-medium` beats `large-v3-turbo` here but not on `test` — a 15-vs-13-song ranking flip, which is the per-song variance the next section warns about.

### What to report alongside the table

- **CER and Character F1 together.** CER alone cannot distinguish a system that stopped early from one that invented text; precision and recall separate them, which is the entire reason both columns exist.
- **The `faster-whisper-small` vs `whisper-small-lora-ja` pair**, on `test`, not `validation` — see [The contribution, honestly](#the-contribution-honestly) above.
- **Per-song spread** (`reports/ja_test_benchmark_per_song.csv`). With 15 test songs, one unusually hard track can move the corpus number by several points, and `faster-whisper-medium`/`large-v3-turbo` already swap rank between `validation` and `test` above.
- **The selected epoch** (2 of 4, validation CER 48.51%) from `finetune_report.json`, so the checkpoint-selection claim is checkable.

## Running the benchmark on a free Colab T4

CPU `int8` inference is the bottleneck in this project — six pretrained systems over 1.64 h of `validation` + `test` audio measured out to roughly six hours of laptop wall clock, and LoRA fine-tuning on CPU is not practical at all. The same work on a free Colab T4 at `float16` is well under an hour. `colab_handoff/` holds everything needed, and **the repository does not have to be cloned on Colab** — the bundle is the only input.

| File | Purpose |
|---|---|
| `colab_handoff/colab_runner.ipynb` | Drop-in notebook: GPU check → upload → install → one-clip smoke test → full run → results |
| `colab_handoff/run_colab.py` | The actual pipeline. Three resumable stages: `baselines`, `finetune`, `evaluate` |
| `colab_handoff/ja_eval_standalone.py` | CER + Character F1 with zero project imports — same arithmetic as `lyrics_asr.ja_evaluate`, verified identical by `tests/test_project.py` |
| `colab_handoff/colab_instructions.md` | Copy-paste cells, expected timings, and troubleshooting (incl. the CTranslate2 cuDNN fix) |
| `colab_handoff/colab_bundle.zip` | 89.9 MB: manifest + 658 cut clips + the three files above. Gitignored; rebuild with one command |

Build the bundle, then upload it:

```powershell
lyrics-asr-package-colab
```

```python
# in Colab, after Runtime > Change runtime type > T4 GPU
!cd /content && unzip -q -o colab_bundle.zip -d bundle
!pip install -q "faster-whisper==1.2.1" "transformers>=4.44,<5" "peft>=0.13" "accelerate>=0.34" soundfile av
!cd /content/bundle && python run_colab.py --stage all --device cuda
```

Outputs: `final_metrics.json` (every number, per split, per system, plus environment and decode settings), `final_predictions/<split>__<system>.jsonl`, `reports/*.md` + `*.csv`, `finetune_report.json`, and `lora_adapter/best/`.

**Audio in the bundle.** The cut clips are 16 kHz mono PCM, which is 256 kbit/s — 476 MB for 658 clips. FLAC only shaves ~17% off dense music, so the bundle instead ships **Opus 48 kbit/s mono** and lands at 89.9 MB. At a 16 kHz sampling rate the signal occupies only 8 kHz of bandwidth, which Opus encodes essentially transparently at that rate, and every system reads the identical encoded clips, so no row is advantaged. Use `lyrics-asr-package-colab --codec flac` (lossless) or `--codec wav` (byte-identical to the local clips) when upload size does not matter. Full-length source songs are deliberately **not** in the bundle; only the cut utterance windows travel.

**Resumability.** Free sessions get reclaimed. Prediction files are append-only and already-transcribed clips are skipped, so rerunning the identical command continues where it stopped and costs at most one clip. Pass `--output /content/drive/MyDrive/ja_asr_out` to write results straight to Drive.

## Reproducing everything

Requires Python 3.12 and FFmpeg on `PATH`. Discovery/cleaning utilities use only the standard library; media and ASR stages need the extras.

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[media,asr]"
```

The Japanese pipeline runs end to end as six commands. Each stage is resumable and writes its own provenance.

```powershell
# 0. Enforce the Japanese-only constraint (dry run first; --apply to write)
lyrics-asr-purge-english --root .
lyrics-asr-purge-english --root . --apply

# 1. Select the corpus and assign the seed-42 speaker-disjoint split
lyrics-asr-ja-corpus

# 2. Fetch only the Japanese WAVs that are missing
lyrics-asr-download-audio --input data/private/ja_pending_sources.jsonl `
  --audio-dir data/audio/ja `
  --archive data/private/ja-yt-dlp-archive.txt `
  --manifest data/private/ja_audio_manifest.jsonl

# 3. Re-run step 1 so the new audio is picked up, then estimate per-song offsets
lyrics-asr-ja-corpus
lyrics-asr-ja-align

# 4. Cut the offset-corrected utterance clips
lyrics-asr-ja-segment
lyrics-asr-validate data/private/ja_utterances.jsonl --require-split

# 5. Benchmark the CTranslate2 systems on development, then once on the frozen test split
lyrics-asr-ja-benchmark --split validation
lyrics-asr-ja-benchmark --split test

# 6. Score and report
lyrics-asr-ja-evaluate --references data/private/ja_utterances.jsonl --split test `
  --prediction faster-whisper-base=data/private/ja_test_predictions_faster-whisper-base.jsonl `
  --prediction faster-whisper-small=data/private/ja_test_predictions_faster-whisper-small.jsonl `
  --prediction faster-whisper-medium=data/private/ja_test_predictions_faster-whisper-medium.jsonl `
  --prediction faster-whisper-large-v3-turbo=data/private/ja_test_predictions_faster-whisper-large-v3-turbo.jsonl `
  --prediction kotoba-whisper-v2.0=data/private/ja_test_predictions_kotoba-whisper-v2.0.jsonl `
  --output reports/ja_test_benchmark
lyrics-asr-ja-stats

# 7. Package the GPU handoff (manifest + cut clips + runner + scorer)
lyrics-asr-package-colab
```

Step 5 runs the five **CTranslate2** systems, which need no PyTorch. The sixth pretrained system (`litagin/anime-whisper`) ships only in `transformers` format, and the LoRA contribution needs a GPU to be worth running at all — both live in the Colab path, which also re-runs the five CTranslate2 systems at `float16` so all seven rows of the results table come from one environment. Step 7 builds that handoff; see [Running the benchmark on a free Colab T4](#running-the-benchmark-on-a-free-colab-t4).

`lyrics-asr-ja-download-audio` uses yt-dlp with `bestaudio`, no playlist expansion, and FFmpeg WAV conversion. It records video ID, source URL, duration, yt-dlp version and SHA-256 per file so any source can be audited. It uses no cookies, no login sessions, and no DRM or access-control bypass; only download videos you are authorized to download, and follow YouTube's current terms. Media, cut clips, references and predictions are all gitignored and never committed.

### Data layout

```text
config/benchmark_models.json          Systems, pinned revisions, metric policy, exclusions
config/searches.json                  44 Japanese artist/song searches
data/raw/genius_candidates.jsonl      Japanese Genius discovery snapshot (1,713)
data/cleaned/lrclib_matches.jsonl     LRCLIB match index, no lyric text (539)
data/cleaned/ja_corpus.jsonl          The corpus + seed-42 split (72 songs)  <- tracked
data/cleaned/ja_corpus_excluded.jsonl Every exclusion with its reason (36)   <- tracked
data/manifest.csv                     Frozen per-song manifest: video/LRCLIB ID,
                                       artist, title, license, split, utterances,
                                       duration_seconds (72 songs)             <- tracked
data/references/lrclib_lyrics.jsonl   LRCLIB plain + synced text             (gitignored)
data/audio/ja/                        Full-song WAVs + yt-dlp provenance     (gitignored)
data/clips/ja/                        Cut 16 kHz mono utterance clips        (gitignored)
data/private/ja_corpus_local.jsonl    Corpus with resolved audio paths       (gitignored)
data/private/ja_alignment.jsonl       Per-song offset, confidence, method    (gitignored)
data/private/ja_utterances.jsonl      The evaluation manifest + references   (gitignored)
reports/                              Scored CSVs and Markdown tables
```

## Limitations

- **Reference quality is an assumption, not a verification.** LRCLIB may omit performed ad-libs, include lines not sung in this release, or describe a different edit of the same song. Scores include that error.
- **20 speaker groups is small.** Four of them carry the whole test split. Per-speaker results are reported precisely because the corpus-level number can be dominated by one singer.
- **Alignment is a scalar, not a forced alignment.** A single offset cannot fix a song whose internal structure differs from the master (an extended bridge, a radio edit). Low-confidence songs are flagged in the manifest rather than removed.
- **`dialect` is `unassessed` for every row** and `dialect_status` is `requires_audio_review`, so no dialect-stratified claim is made. Dialect must not be inferred from an artist's birthplace or name; singing can neutralize or stylize regional features.
- **Speaker groups are curated and provisional.** Producer credits and collaborations need care, and Vocaloid recordings use voicebanks rather than human singers — the Inabakumori tracks are grouped under `kaai-yuki` and `kafu` by voicebank.
- **Backing music is not removed.** No source separation is applied, so scores reflect vocals plus instrumental. Source separation is the most promising next step for this corpus.
- **These are results on a 72-song corpus**, not a full corpus. Six of the seven systems are zero-shot; only `whisper-small-lora-ja` trained on the `train` split, and only on `train`.

## Tests and CI

```powershell
python -m unittest discover -s tests -v
lyrics-asr-validate tests/fixtures/manifest.jsonl
```

59 tests cover discovery/merge, candidate filtering, LRCLIB match/retry, lyrics-free metadata export, yt-dlp URL/search/download handling, the English purge and its bilingual-row rule, the Japanese corpus gates and source picking, window construction, offset recovery from a known synthetic shift, clip cutting against the real FFmpeg binary, Japanese CER/Character-F1 behaviour including hallucination accounting, pinned-speaker split determinism, and leakage detection. Tests make no network calls; the two FFmpeg-backed tests skip when the binary is absent.

GitHub Actions runs the suite and CLI smoke checks on Python 3.10 and 3.12, and asserts that re-running the English purge removes nothing — so an English row cannot re-enter the corpus unnoticed. `v*` tags build wheel/source artifacts without publishing them.
