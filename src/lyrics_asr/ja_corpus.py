"""Select the Japanese-only song corpus and assign its speaker-disjoint split.

Every quality gate is a module-level constant so the corpus is defined by a rule
rather than by hand-picking, and so the gates can be quoted in the write-up. The
gates are applied before any ASR system runs, which keeps the test split frozen
and independent of any score.

Gates, and why each one exists:

``MIN_JA_SCRIPT_RATIO``
    LRCLIB sometimes stores a romanized or translated version of a Japanese
    lyric. Scoring kana/kanji ASR output against romaji would report ~100% CER
    for a reference problem, not a model problem, so a reference whose script is
    mostly Latin is dropped.
``MIN_SYNCED_LINES`` / ``SYNC_COVERAGE``
    Utterance references are cut from LRCLIB ``syncedLyrics`` timestamps. A row
    with too few lines, or whose lyric timespan does not plausibly cover the
    track, cannot be segmented reliably.
``MAX_SOURCE_DELTA_SECONDS``
    A candidate video whose duration is far from the LRCLIB duration is usually a
    different edit (TV size, extended, medley). A moderate difference is fine
    because :mod:`lyrics_asr.ja_align` estimates a per-song time offset.
``TITLE_BLOCKLIST``
    Covers, lives, karaoke and instrumentals are not the studio recording the
    LRCLIB reference describes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

from .split import assign_splits

MIN_JA_SCRIPT_RATIO = 0.60
MIN_SYNCED_LINES = 8
SYNC_COVERAGE = (0.55, 1.08)
MAX_SOURCE_DELTA_SECONDS = 45.0
SONGS_PER_SPEAKER_CAP = 5
TEST_SONGS_PER_SPEAKER_CAP = 4
SEED = 42
RATIOS = (0.62, 0.20, 0.18)

TITLE_BLOCKLIST = re.compile(
    r"cover|live|remix|karaoke|instrumental|off\s*vocal|sped\s*up|slowed|nightcore|8d\s*audio"
    r"|reaction|tv\s*size|short\s*ver|acoustic|piano|orgel|music\s*box|a\s*cappella|mashup"
    r"|first\s*take|unplugged|session|rehearsal|\bmedley\b"
    r"|カラオケ|カバー|歌って|歌わせ|ライブ|弾いて|演奏|オルゴール|耳コピ|アカペラ",
    re.I,
)

KANA_KANJI = re.compile(r"[぀-ゟ゠-ヿ一-鿿々ー]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
TIMESTAMP = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")

SELECTION_SOURCES = (
    "data/cleaned/japanese_benchmark_selection.jsonl",
    "data/cleaned/benchmark_selection.jsonl",
)
# Only this selection's test rows are frozen: they are the songs whose zero-shot
# baseline predictions are already cached, so their speakers must stay in test.
FROZEN_TEST_SELECTION = "data/cleaned/benchmark_selection.jsonl"
CANDIDATE_SOURCES = (
    "data/private/japanese_youtube_candidates_final.jsonl",
    "data/private/japanese_youtube_candidates.jsonl",
    "data/private/benchmark_youtube_candidates_full.jsonl",
    "data/private/benchmark_youtube_candidates.jsonl",
    "data/cleaned/youtube_candidates.jsonl",
)
APPROVED_SOURCES = (
    "data/cleaned/benchmark_video_sources.jsonl",
    "data/private/benchmark_youtube_sources.jsonl",
    "data/private/benchmark_test_sources.jsonl",
    "data/private/benchmark_kenshi_train_sources.jsonl",
)
AUDIO_MANIFESTS = (
    "data/private/benchmark_sample_audio_manifest.jsonl",
    "data/private/benchmark_sample_audio_manifest_v2.jsonl",
    "data/private/benchmark_extra_audio_manifest.jsonl",
    "data/private/benchmark_audio_manifest.jsonl",
    "data/private/benchmark_test_audio_manifest.jsonl",
    "data/private/ja_audio_manifest.jsonl",
)
AUDIO_DIRS = ("data/audio/benchmark", "data/audio/benchmark_extra", "data/audio/benchmark_test", "data/audio/ja")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def japanese_script_ratio(text: str) -> float:
    core = re.sub(r"\s+", "", text)
    kana_kanji = len(KANA_KANJI.findall(core))
    latin = len(LATIN_LETTER.findall(core))
    return kana_kanji / (kana_kanji + latin) if (kana_kanji + latin) else 0.0


def parse_synced(synced: str) -> list[tuple[float, str]]:
    lines: list[tuple[float, str]] = []
    for raw in (synced or "").splitlines():
        match = TIMESTAMP.match(raw.strip())
        if not match:
            continue
        seconds = int(match.group(1)) * 60 + float(match.group(2))
        lines.append((seconds, match.group(3).strip()))
    return [line for line in lines if line[1]]


def load_candidates(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Merge every stored yt-dlp candidate list and approved source into one pool."""
    pool: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()

    def add(clip_id: str, candidate: dict[str, Any]) -> None:
        video_id = str(candidate.get("video_id") or "")
        if not video_id or (clip_id, video_id) in seen:
            return
        seen.add((clip_id, video_id))
        pool.setdefault(clip_id, []).append(candidate)

    for name in CANDIDATE_SOURCES:
        for row in read_jsonl(root / name):
            raw = row.get("yt_dlp_candidates")
            if isinstance(raw, str):
                try:
                    raw = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    continue
            for candidate in raw or []:
                add(str(row["clip_id"]), dict(candidate))
    for name in APPROVED_SOURCES:
        for row in read_jsonl(root / name):
            video_id = row.get("video_id")
            if not video_id:
                continue
            add(str(row["clip_id"]), {
                "video_id": video_id,
                "youtube_url": row.get("youtube_url") or f"https://www.youtube.com/watch?v={video_id}",
                "video_title": row.get("video_title") or "",
                "channel": row.get("channel") or "",
                "duration_seconds": row.get("video_duration_seconds") or row.get("duration_seconds"),
                "previously_approved": True,
            })
    return pool


def load_cached_audio(root: Path) -> dict[str, dict[str, Any]]:
    """Map clip_id to an already-downloaded WAV, so nothing is fetched twice."""
    on_disk: dict[str, Path] = {}
    for directory in AUDIO_DIRS:
        folder = root / directory
        if folder.is_dir():
            for wav in sorted(folder.glob("*.wav")):
                on_disk[wav.stem] = wav
    cached: dict[str, dict[str, Any]] = {}
    for name in AUDIO_MANIFESTS:
        for row in read_jsonl(root / name):
            clip_id, video_id = str(row.get("clip_id") or ""), str(row.get("video_id") or "")
            if not clip_id or video_id not in on_disk or clip_id in cached:
                continue
            cached[clip_id] = {
                "video_id": video_id,
                "youtube_url": row.get("youtube_url"),
                "video_title": row.get("video_title") or "",
                "channel": row.get("channel") or "",
                "video_duration_seconds": row.get("duration_seconds") or row.get("video_duration_seconds"),
                "audio_path": str(on_disk[video_id]),
                "audio_sha256": row.get("audio_sha256"),
            }
    return cached


def pick_source(
    candidates: list[dict[str, Any]],
    reference_duration: float,
    artist: str,
) -> tuple[dict[str, Any] | None, str]:
    """Choose the candidate most likely to be the studio recording.

    Preference order: title not blocklisted, duration close to the LRCLIB
    duration, and an official/artist channel as the tie-break.
    """
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for candidate in candidates:
        duration = candidate.get("duration_seconds")
        title = str(candidate.get("video_title") or "")
        if duration is None:
            continue
        if TITLE_BLOCKLIST.search(title):
            continue
        delta = abs(float(duration) - reference_duration)
        if delta > MAX_SOURCE_DELTA_SECONDS:
            continue
        channel = str(candidate.get("channel") or "").casefold()
        official = "official" in channel or (artist and artist.casefold() in channel)
        scored.append((delta - (3.0 if official else 0.0), delta, candidate))
    if not scored:
        return None, "no_candidate_within_duration_and_title_rules"
    scored.sort(key=lambda item: (item[0], str(item[2].get("video_id"))))
    return scored[0][2], "title_blocklist; duration match; official-channel tie-break"


def build(root: Path, corpus_output: Path, sources_output: Path, excluded_output: Path) -> None:
    lyrics = {str(row["clip_id"]): row for row in read_jsonl(root / "data/references/lrclib_lyrics.jsonl")}
    candidates = load_candidates(root)
    cached = load_cached_audio(root)

    # Union every Japanese selection we have, and remember the frozen test songs.
    pool: dict[str, dict[str, Any]] = {}
    pinned_test_clips: set[str] = set()
    for name in SELECTION_SOURCES:
        for row in read_jsonl(root / name):
            if str(row.get("language")) != "ja":
                continue
            clip_id = str(row["clip_id"])
            entry = pool.setdefault(clip_id, {})
            entry.update({
                "clip_id": clip_id,
                "genius_id": row.get("genius_id"),
                "lrclib_id": row.get("lrclib_id"),
                "genius_url": row.get("genius_url"),
                "lrclib_url": row.get("lrclib_url"),
                "title": row.get("title") or row.get("song_title"),
                "artist": row.get("artist") or (row.get("artists") or [""])[0],
                "artists": row.get("artists") or [],
                "artist_hint": row.get("artist_hint"),
                "speaker_group": row.get("speaker_group"),
                "recording_group": row.get("recording_group") or clip_id,
                "popularity_tier": row.get("popularity_tier") or "unknown",
                "pageviews_at_snapshot": row.get("pageviews_at_snapshot"),
            })
            if row.get("split") == "test" and name == FROZEN_TEST_SELECTION:
                pinned_test_clips.add(clip_id)

    excluded: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for clip_id, entry in sorted(pool.items()):
        reason = None
        lyric = lyrics.get(clip_id)
        if lyric is None:
            reason = "no_lrclib_reference_in_local_cache"
        else:
            text = str(lyric.get("text") or "")
            ratio = japanese_script_ratio(text)
            synced = parse_synced(str(lyric.get("synced_lyrics") or ""))
            duration = float(lyric.get("duration_seconds") or 0.0)
            span = (synced[-1][0] - synced[0][0]) if len(synced) >= 2 else 0.0
            coverage = span / duration if duration else 0.0
            if ratio < MIN_JA_SCRIPT_RATIO:
                reason = f"reference_not_japanese_script (ratio={ratio:.2f})"
            elif len(synced) < MIN_SYNCED_LINES:
                reason = f"too_few_synced_lines ({len(synced)})"
            elif not duration:
                reason = "no_lrclib_duration"
            elif not SYNC_COVERAGE[0] <= coverage <= SYNC_COVERAGE[1]:
                reason = f"implausible_sync_coverage ({coverage:.2f})"
            else:
                entry.update({
                    "lrclib_title": lyric.get("lrclib_title"),
                    "lrclib_artist": lyric.get("lrclib_artist"),
                    "reference_duration_seconds": duration,
                    "japanese_script_ratio": round(ratio, 4),
                    "synced_line_count": len(synced),
                    "sync_coverage": round(coverage, 4),
                })
        if reason:
            excluded.append({**entry, "exclusion_reason": reason})
            continue

        cache = cached.get(clip_id)
        if cache:
            entry.update({
                "video_id": cache["video_id"],
                "youtube_url": cache["youtube_url"] or f"https://www.youtube.com/watch?v={cache['video_id']}",
                "video_title": cache["video_title"],
                "channel": cache["channel"],
                "video_duration_seconds": cache["video_duration_seconds"],
                "audio_path": cache["audio_path"],
                "audio_status": "cached",
                "source_selection_method": "previously approved and downloaded",
            })
        else:
            chosen, method = pick_source(candidates.get(clip_id, []), entry["reference_duration_seconds"], str(entry.get("artist") or ""))
            if chosen is None:
                excluded.append({**entry, "exclusion_reason": method})
                continue
            entry.update({
                "video_id": chosen["video_id"],
                "youtube_url": chosen.get("youtube_url") or f"https://www.youtube.com/watch?v={chosen['video_id']}",
                "video_title": chosen.get("video_title") or "",
                "channel": chosen.get("channel") or "",
                "video_duration_seconds": chosen.get("duration_seconds"),
                "audio_path": None,
                "audio_status": "missing",
                "source_selection_method": method,
            })
        entry["source_duration_delta_seconds"] = round(
            float(entry["video_duration_seconds"] or 0) - entry["reference_duration_seconds"], 1
        )
        entry["is_pinned_test"] = clip_id in pinned_test_clips
        eligible.append(entry)

    # Cap songs per speaker so no single vocalist dominates and so the frozen test
    # speakers stay inside the requested 15-18 song test budget.
    pinned_speakers = frozenset(
        str(entry["speaker_group"]) for entry in eligible if entry["is_pinned_test"]
    )
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for entry in eligible:
        by_speaker.setdefault(str(entry["speaker_group"]), []).append(entry)

    selected: list[dict[str, Any]] = []
    for speaker, entries in sorted(by_speaker.items()):
        cap = TEST_SONGS_PER_SPEAKER_CAP if speaker in pinned_speakers else SONGS_PER_SPEAKER_CAP
        entries.sort(key=lambda item: (
            not item["is_pinned_test"],                      # frozen test songs first
            item["audio_status"] != "cached",                # then anything already downloaded
            abs(float(item["source_duration_delta_seconds"])),  # then best-aligned source
            -float(item["japanese_script_ratio"]),
            str(item["clip_id"]),
        ))
        keep, drop = entries[:cap], entries[cap:]
        selected.extend(keep)
        excluded.extend({**entry, "exclusion_reason": f"over_speaker_cap ({cap})"} for entry in drop)

    for entry in selected:
        entry["language"] = "ja"
        entry["language_status"] = "curated_from_song_and_reference_metadata"
        entry["dialect"] = "unassessed"
        entry["dialect_status"] = "requires_audio_review"
        entry["reference_status"] = "lrclib_assumed_gold"
        entry["selection_seed"] = SEED
        entry["selection_method"] = (
            f"japanese-only; ja_script>={MIN_JA_SCRIPT_RATIO}; synced>={MIN_SYNCED_LINES} lines; "
            f"source |dur delta|<={MAX_SOURCE_DELTA_SECONDS}s; speaker cap {SONGS_PER_SPEAKER_CAP}"
            f" ({TEST_SONGS_PER_SPEAKER_CAP} for frozen test speakers)"
        )

    splits = assign_splits(selected, SEED, RATIOS, pinned_speakers)
    for entry in selected:
        entry["split"] = splits[entry["clip_id"]]

    ordered_keys = [
        "clip_id", "genius_id", "lrclib_id", "genius_url", "lrclib_url", "title", "artist", "artists",
        "artist_hint", "lrclib_title", "lrclib_artist", "language", "language_status", "dialect",
        "dialect_status", "popularity_tier", "pageviews_at_snapshot", "reference_duration_seconds",
        "japanese_script_ratio", "synced_line_count", "sync_coverage", "video_id", "youtube_url",
        "video_title", "channel", "video_duration_seconds", "source_duration_delta_seconds",
        "source_selection_method", "audio_status", "split", "speaker_group", "recording_group",
        "is_pinned_test", "reference_status", "selection_method", "selection_seed",
    ]
    corpus_rows = [{key: entry.get(key) for key in ordered_keys} for entry in selected]
    corpus_rows.sort(key=lambda row: (row["split"], str(row["speaker_group"]), str(row["clip_id"])))
    write_jsonl(corpus_output, corpus_rows)

    # The tracked corpus carries no local paths. Downstream stages need them, so a
    # parallel local-only copy adds audio_path for the songs already on disk.
    by_clip = {entry["clip_id"]: entry for entry in selected}
    local_rows = [
        {**row, "audio_path": by_clip[row["clip_id"]].get("audio_path"),
         "audio_sha256": cached.get(row["clip_id"], {}).get("audio_sha256")}
        for row in corpus_rows
        if by_clip[row["clip_id"]].get("audio_path")
    ]
    write_jsonl(corpus_output.parent.parent / "private" / "ja_corpus_local.jsonl", local_rows)

    pending = [
        {
            "clip_id": entry["clip_id"], "genius_id": entry.get("genius_id"), "lrclib_id": entry.get("lrclib_id"),
            "split": entry["split"], "speaker_group": entry["speaker_group"], "recording_group": entry["recording_group"],
            "language": "ja", "language_status": entry["language_status"], "dialect": entry["dialect"],
            "dialect_status": entry["dialect_status"], "youtube_url": entry["youtube_url"],
            "video_id": entry["video_id"], "video_title": entry.get("video_title"), "channel": entry.get("channel"),
            "video_duration_seconds": entry.get("video_duration_seconds"),
            "reference_duration_seconds": entry["reference_duration_seconds"],
            "title": entry.get("title"), "artist": entry.get("artist"),
            "source_selection_method": entry["source_selection_method"],
        }
        for entry in selected if entry["audio_status"] == "missing"
    ]
    write_jsonl(sources_output, pending)
    write_jsonl(excluded_output, excluded)

    print(f"Japanese pool considered: {len(pool)} songs")
    print(f"Excluded by quality gates / caps: {len(excluded)}")
    print(f"Corpus: {len(corpus_rows)} songs, {len({row['speaker_group'] for row in corpus_rows})} speaker groups")
    for name in ("train", "validation", "test"):
        subset = [row for row in corpus_rows if row["split"] == name]
        hours = sum(float(row["reference_duration_seconds"]) for row in subset) / 3600
        print(f"  {name:11s} {len(subset):3d} songs  {len({row['speaker_group'] for row in subset}):2d} speakers  {hours:.2f} reference hours")
    pinned_in_test = sum(1 for row in corpus_rows if row["is_pinned_test"] and row["split"] == "test")
    print(f"Frozen test songs preserved in test: {pinned_in_test}/{len(pinned_test_clips)}")
    print(f"Audio still to download: {len(pending)} (cached: {sum(1 for r in selected if r['audio_status'] == 'cached')})")
    print(f"Wrote {corpus_output}, {sources_output}, {excluded_output}, "
          f"{corpus_output.parent.parent / 'private' / 'ja_corpus_local.jsonl'} ({len(local_rows)} rows with audio)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--corpus-output", type=Path, default=Path("data/cleaned/ja_corpus.jsonl"))
    parser.add_argument("--sources-output", type=Path, default=Path("data/private/ja_pending_sources.jsonl"))
    parser.add_argument("--excluded-output", type=Path, default=Path("data/cleaned/ja_corpus_excluded.jsonl"))
    args = parser.parse_args()
    try:
        build(args.root, args.corpus_output, args.sources_output, args.excluded_output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
