"""Remove every English song from the tracked and local corpus files.

The course constraint for this project is Japanese-only data. This module is the
single reproducible place where that constraint is enforced, so the purge can be
re-run and audited instead of being a one-off manual edit.

A row is English when its language signal names ``en`` and never names ``ja``.
Rows whose signal names ``ja`` are always kept, including bilingual rows, because
the project treats the song's sung language as the label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

LANGUAGE_FIELDS = ("language", "language_hint", "language_hints", "lang")

# Tracked and local JSONL files that carry one song (or candidate) per line.
JSONL_TARGETS = (
    "data/raw/genius_candidates.jsonl",
    "data/cleaned/genius_candidates_cleaned.jsonl",
    "data/cleaned/lrclib_matches.jsonl",
    "data/cleaned/benchmark_selection.jsonl",
    "data/cleaned/benchmark_video_sources.jsonl",
    "data/cleaned/youtube_candidates.jsonl",
    "data/cleaned/japanese_benchmark_selection.jsonl",
    "data/private/japanese_benchmark_references.jsonl",
    "data/private/japanese_youtube_candidates.jsonl",
    "data/private/japanese_youtube_candidates_final.jsonl",
    "data/references/lrclib_lyrics.jsonl",
    "data/references/genius_lyrics.jsonl",
    "data/private/benchmark_manifest.jsonl",
    "data/private/benchmark_references.jsonl",
    "data/private/benchmark_references_v2.jsonl",
    "data/private/benchmark_selection.jsonl",
    "data/private/benchmark_sample_audio_manifest.jsonl",
    "data/private/benchmark_sample_audio_manifest_v2.jsonl",
    "data/private/benchmark_audio_manifest.jsonl",
    "data/private/benchmark_extra_audio_manifest.jsonl",
    "data/private/benchmark_test_audio_manifest.jsonl",
    "data/private/benchmark_test_references.jsonl",
    "data/private/benchmark_test_sources.jsonl",
    "data/private/benchmark_sources_missing.jsonl",
    "data/private/benchmark_youtube_candidates.jsonl",
    "data/private/benchmark_youtube_candidates_full.jsonl",
    "data/private/benchmark_youtube_sources.jsonl",
    "data/private/benchmark_kenshi_train_sources.jsonl",
    "data/private/benchmark_predictions_whisper_base.jsonl",
    "data/private/benchmark_predictions_whisper_small.jsonl",
    "data/private/benchmark_smoke_audio.jsonl",
    "data/private/benchmark_smoke_reference.jsonl",
    "data/private/benchmark_smoke_prediction.jsonl",
)

# Search/selection configs with a per-entry language field.
CONFIG_TARGETS = (
    ("config/searches.json", "queries"),
    ("config/personal_requests.json", "queries"),
    ("config/benchmark_sample.json", "groups"),
    ("config/japanese_benchmark_sample.json", "groups"),
    ("config/japanese_expansion.json", "queries"),
)

# English-only artefacts that have no Japanese counterpart.
DELETE_FILES = (
    "data/private/benchmark_predictions_whisper_base.jsonl.en.bak",
    "reports/benchmark_asr_per_clip.csv",
    "reports/benchmark_asr_summary.csv",
    "reports/benchmark_whisper_base_per_clip.csv",
    "reports/benchmark_whisper_base_summary.csv",
    "reports/benchmark_whisper_small_per_clip.csv",
    "reports/benchmark_whisper_small_summary.csv",
)


def language_signal(row: dict[str, Any]) -> list[str]:
    """Collect every language code mentioned by a row."""
    codes: list[str] = []
    for field in LANGUAGE_FIELDS:
        if field not in row:
            continue
        value = row[field]
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if item is None:
                continue
            code = str(item).strip().lower().split("-")[0]
            if code:
                codes.append(code)
    return codes


def is_english_only(row: dict[str, Any]) -> bool:
    codes = language_signal(row)
    return bool(codes) and "ja" not in codes and "en" in codes


def filter_jsonl(path: Path, english_clip_ids: set[str], apply: bool) -> tuple[int, int]:
    """Drop English-only rows; also drop rows keyed to an English clip_id."""
    kept: list[str] = []
    removed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            clip_id = str(row.get("clip_id") or (f"genius-{row['genius_id']}" if row.get("genius_id") else ""))
            if is_english_only(row) or (clip_id and clip_id in english_clip_ids):
                removed += 1
                continue
            kept.append(json.dumps(row, ensure_ascii=False))
    if apply and removed:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")
    return removed, len(kept)


def filter_config(path: Path, key: str, apply: bool) -> tuple[int, int]:
    config = json.loads(path.read_text(encoding="utf-8"))
    entries = config.get(key)
    if not isinstance(entries, list):
        return 0, 0
    kept = [entry for entry in entries if not is_english_only(entry)]
    removed = len(entries) - len(kept)
    if apply and removed:
        config[key] = kept
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return removed, len(kept)


def collect_english_clip_ids(root: Path) -> tuple[set[str], set[str]]:
    """Return English clip IDs and the video IDs whose audio should be deleted.

    Video IDs are only returned when no Japanese row claims the same file, so a
    source shared with a Japanese song is never deleted.
    """
    english: set[str] = set()
    japanese: set[str] = set()
    english_videos: set[str] = set()
    japanese_videos: set[str] = set()
    for name in JSONL_TARGETS:
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            clip_id = str(row.get("clip_id") or (f"genius-{row['genius_id']}" if row.get("genius_id") else ""))
            video_id = str(row.get("video_id") or "")
            if not video_id:
                audio_path = str(row.get("audio_path") or "")
                if audio_path.endswith(".wav"):
                    video_id = Path(audio_path.replace("\\", "/")).stem
            codes = language_signal(row)
            if "ja" in codes:
                if clip_id:
                    japanese.add(clip_id)
                if video_id:
                    japanese_videos.add(video_id)
            elif "en" in codes:
                if clip_id:
                    english.add(clip_id)
                if video_id:
                    english_videos.add(video_id)
    return english - japanese, english_videos - japanese_videos


def delete_audio(root: Path, video_ids: Iterable[str], apply: bool) -> list[Path]:
    removed: list[Path] = []
    audio_root = root / "data" / "audio"
    if not audio_root.is_dir():
        return removed
    wanted = set(video_ids)
    for path in sorted(audio_root.rglob("*")):
        if not path.is_file():
            continue
        stem = path.name.split(".")[0]
        if stem in wanted:
            removed.append(path)
            if apply:
                path.unlink()
    return removed


def purge(root: Path, apply: bool) -> None:
    english_clips, english_videos = collect_english_clip_ids(root)
    print(f"English-only clip IDs: {len(english_clips)}")
    print(f"English-only source video IDs: {len(english_videos)}")

    total_removed = 0
    for name in JSONL_TARGETS:
        path = root / name
        if not path.is_file():
            continue
        removed, kept = filter_jsonl(path, english_clips, apply)
        total_removed += removed
        if removed:
            print(f"  {name}: removed {removed}, kept {kept}")
    for name, key in CONFIG_TARGETS:
        path = root / name
        if not path.is_file():
            continue
        removed, kept = filter_config(path, key, apply)
        if removed:
            print(f"  {name}[{key}]: removed {removed}, kept {kept}")

    # lrclib_pending.jsonl carries no language field, so it is reconciled by key:
    # a candidate only belongs there if it survived the cleaned-candidate purge.
    cleaned_path = root / "data/cleaned/genius_candidates_cleaned.jsonl"
    pending_path = root / "data/cleaned/lrclib_pending.jsonl"
    if cleaned_path.is_file() and pending_path.is_file():
        cleaned_ids = {
            str(json.loads(line)["genius_id"])
            for line in cleaned_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        kept, dropped = [], 0
        for line in pending_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("genius_id")) in cleaned_ids:
                kept.append(json.dumps(row, ensure_ascii=False))
            else:
                dropped += 1
        if dropped:
            print(f"  data/cleaned/lrclib_pending.jsonl: removed {dropped}, kept {len(kept)} (reconciled to cleaned candidates)")
            if apply:
                pending_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")

    audio = delete_audio(root, english_videos, apply)
    print(f"  audio files {'deleted' if apply else 'to delete'}: {len(audio)}")

    for name in DELETE_FILES:
        path = root / name
        if path.is_file():
            print(f"  {name}: {'deleted' if apply else 'to delete'} (English-era report)")
            if apply:
                path.unlink()

    print(f"\nTotal JSONL rows removed: {total_removed}")
    if not apply:
        print("Dry run. Re-run with --apply to write the changes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--apply", action="store_true", help="Write changes; without it the run is a dry run")
    args = parser.parse_args()
    try:
        purge(args.root, args.apply)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
