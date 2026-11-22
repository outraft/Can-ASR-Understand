"""Assemble a local audio/reference manifest from the tracked selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def read_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row_key = str(row[key])
        if row_key in rows:
            raise ValueError(f"{path}:{line_number}: duplicate {key} {row_key!r}")
        rows[row_key] = row
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_benchmark(
    selection_path: Path,
    lyrics_path: Path,
    audio_manifest_path: Path,
    manifest_output: Path,
    references_output: Path,
    test_references_output: Path,
) -> tuple[int, int]:
    selection = [json.loads(line) for line in selection_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lyrics = read_index(lyrics_path, "clip_id")
    audio = read_index(audio_manifest_path, "clip_id")
    manifest_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    test_reference_rows: list[dict[str, Any]] = []
    for selected in selection:
        clip_id = str(selected["clip_id"])
        if clip_id not in lyrics:
            raise ValueError(f"Missing local LRCLIB lyric reference for {clip_id}")
        if clip_id not in audio:
            raise ValueError(f"Missing local audio manifest row for {clip_id}")
        lyric, audio_row = lyrics[clip_id], audio[clip_id]
        audio_path = Path(str(audio_row["audio_path"]))
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing local audio for {clip_id}: {audio_path}")
        if selected.get("lrclib_id") != lyric.get("lrclib_id"):
            raise ValueError(f"LRCLIB ID mismatch for {clip_id}")
        if selected.get("video_id") and audio_row.get("video_id") and selected["video_id"] != audio_row["video_id"]:
            raise ValueError(f"YouTube video ID mismatch for {clip_id}")
        reference = {
            "clip_id": clip_id,
            "language": selected["language"],
            "language_status": selected.get("language_status", "curated_from_song_and_reference_metadata"),
            "dialect": selected.get("dialect", "unassessed"),
            "dialect_status": selected.get("dialect_status", "requires_audio_review"),
            "text": lyric["text"],
            "lrclib_id": lyric.get("lrclib_id"),
            "reference_policy": "lrclib_assumed_gold",
            "split": selected["split"],
        }
        reference_rows.append(reference)
        if selected["split"] == "test":
            test_reference_rows.append(reference)
        manifest_rows.append({
            "clip_id": clip_id,
            "song_id": f"genius-{selected['genius_id']}",
            "genius_id": selected["genius_id"],
            "lrclib_id": selected.get("lrclib_id"),
            "genius_url": selected.get("genius_url"),
            "lrclib_url": selected.get("lrclib_url"),
            "song_title": selected.get("title"),
            "artists": selected.get("artists", []),
            "language": selected["language"],
            "dialect": selected.get("dialect", "unassessed"),
            "dialect_status": selected.get("dialect_status", "requires_audio_review"),
            "popularity_tier": selected.get("popularity_tier") or "unknown",
            "pageviews_at_selection": selected.get("pageviews_at_snapshot"),
            "popularity_tier_method": "pageview tercile" if selected.get("popularity_tier") else "pageview unavailable; left unranked",
            "youtube_url": audio_row.get("youtube_url"),
            "audio_path": str(audio_path),
            "reference_path": str(references_output),
            "reference_status": "lrclib_assumed_gold",
            "speaker_group": selected["speaker_group"],
            "recording_group": selected["recording_group"],
            "duration_seconds": audio_row.get("duration_seconds") or selected.get("duration_seconds"),
            "split": selected["split"],
        })
    write_jsonl(manifest_output, manifest_rows)
    write_jsonl(references_output, reference_rows)
    write_jsonl(test_references_output, test_reference_rows)
    return len(manifest_rows), len(test_reference_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("data/cleaned/benchmark_selection.jsonl"))
    parser.add_argument("--lyrics", type=Path, default=Path("data/references/lrclib_lyrics.jsonl"))
    parser.add_argument("--audio-manifest", type=Path, default=Path("data/private/benchmark_sample_audio_manifest.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("data/private/benchmark_manifest.jsonl"))
    parser.add_argument("--references-output", type=Path, default=Path("data/private/benchmark_references.jsonl"))
    parser.add_argument("--test-references-output", type=Path, default=Path("data/private/benchmark_test_references.jsonl"))
    args = parser.parse_args()
    try:
        total, test = prepare_benchmark(args.selection, args.lyrics, args.audio_manifest, args.manifest_output, args.references_output, args.test_references_output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    print(f"Prepared {total} local benchmark clips ({test} test references); text/media remain gitignored.")


if __name__ == "__main__":
    main()
