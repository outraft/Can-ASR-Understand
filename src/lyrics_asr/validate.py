"""Validate a curated JSONL dataset manifest and split integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

REQUIRED = {
    "clip_id", "song_id", "genius_id", "genius_url", "song_title", "artists",
    "language", "popularity_tier", "youtube_url", "audio_path", "reference_path",
    "reference_status", "speaker_group", "recording_group", "duration_seconds", "dialect", "dialect_status",
}
TIERS = {"popular", "mid", "niche", "unknown"}
SPLITS = {"train", "validation", "test"}


def validate_rows(rows: list[dict[str, Any]], require_split: bool = False) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    speakers: dict[str, str] = {}
    recordings: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        prefix = f"row {index}"
        missing = sorted(REQUIRED - row.keys())
        if missing:
            errors.append(f"{prefix}: missing fields: {', '.join(missing)}")
            continue
        clip_id = str(row["clip_id"])
        if clip_id in seen:
            errors.append(f"{prefix}: duplicate clip_id {clip_id!r}")
        seen.add(clip_id)
        if not str(row["language"]).strip() or row["language"] == "und":
            errors.append(f"{prefix}: language must be curated")
        if row["popularity_tier"] not in TIERS:
            errors.append(f"{prefix}: popularity_tier must be one of {sorted(TIERS)}")
        if not isinstance(row["artists"], list) or not row["artists"]:
            errors.append(f"{prefix}: artists must be a non-empty list")
        if row["reference_status"] not in {"checked", "lrclib_assumed_gold"}:
            errors.append(f"{prefix}: reference_status must be 'checked' or 'lrclib_assumed_gold'")
        try:
            if float(row["duration_seconds"]) <= 0:
                errors.append(f"{prefix}: duration_seconds must be positive")
        except (TypeError, ValueError):
            errors.append(f"{prefix}: duration_seconds must be numeric")
        for field in ("genius_url", "youtube_url"):
            if urlparse(str(row[field])).scheme not in {"http", "https"}:
                errors.append(f"{prefix}: {field} must be an http(s) URL")
        if require_split and row.get("split") not in SPLITS:
            errors.append(f"{prefix}: split must be one of {sorted(SPLITS)}")
        split = row.get("split")
        for key, mapping, label in ((str(row["speaker_group"]), speakers, "speaker"), (str(row["recording_group"]), recordings, "recording")):
            if split:
                previous = mapping.setdefault(key, str(split))
                if previous != split:
                    errors.append(f"{prefix}: {label} group {key!r} crosses splits {previous!r} and {split!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-split", action="store_true")
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        errors = validate_rows(rows, args.require_split)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print(f"Valid manifest: {len(rows)} clips")


if __name__ == "__main__":
    main()
