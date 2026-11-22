# Raw candidate metadata

Store versioned Genius API candidate metadata snapshots here, for example
`genius_candidates.jsonl`. The files are intentionally tracked by Git so the
query results used to curate the corpus can be audited and reproduced. They
contain discovery metadata, not scraped lyric text. Never include API tokens.

The merged snapshot, `genius_candidates.jsonl`, contains 1,713 unique Japanese
search results from 44 Japanese artist/song searches plus one personal-interest
search. English results were removed outright by `lyrics-asr-purge-english` to
keep the project Japanese-only, so the snapshot no longer reproduces the
original bilingual discovery run. It has not been fully artist-verified.
Do not put audio or lyric text in this directory.
