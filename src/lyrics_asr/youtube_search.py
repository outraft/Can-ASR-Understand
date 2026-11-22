"""Search YouTube for candidate video metadata using yt-dlp (no media download)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def candidate_searches(matches: list[dict[str, Any]], search_fn, results_per_song: int = 5, limit: int | None = None, delay: float = 1.0) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    selected = matches[:limit] if limit else matches
    for record in selected:
        title = str(record.get("lrclib_title") or record.get("title") or record.get("genius_title") or "").strip()
        artist = str(record.get("lrclib_artist") or record.get("artist") or record.get("genius_primary_artist") or "").strip()
        query = " ".join(part for part in (title, artist) if part)
        if not query:
            continue
        results = search_fn(query, results_per_song)
        result_row = {
            "clip_id": record.get("clip_id", f"genius-{record['genius_id']}"),
            "genius_id": record["genius_id"],
            "lrclib_id": record.get("lrclib_id"),
            "language": record.get("language"),
            "reference_duration_seconds": record.get("duration_seconds"),
            "track_title": title,
            "artist": artist,
            "reference_duration_seconds": record.get("duration_seconds"),
            "query": query,
            "search_status": "needs_manual_video_review",
            "yt_dlp_candidates": results,
        }
        for field in ("split", "speaker_group", "recording_group"):
            if field in record:
                result_row[field] = record[field]
        if record.get("artist_hint"):
            result_row["artist_hint"] = record["artist_hint"]
        output.append(result_row)
        time.sleep(max(0.0, delay))
    return output


def search_youtube(query: str, count: int = 5) -> list[dict[str, Any]]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install the optional media dependency: python -m pip install -e .[media]") from exc
    options = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"ytsearch{count}:{query}", download=False)
    candidates: list[dict[str, Any]] = []
    for item in (info or {}).get("entries") or []:
        video_id = item.get("id")
        if not video_id:
            continue
        candidates.append({
            "video_id": str(video_id),
            "youtube_url": item.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
            "video_title": item.get("title"),
            "channel": item.get("channel") or item.get("uploader"),
            "duration_seconds": item.get("duration"),
        })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/cleaned/lrclib_matches.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/youtube_candidates.jsonl"))
    parser.add_argument("--results-per-song", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Search only the first N songs as a trial")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    if args.results_per_song < 1 or (args.limit is not None and args.limit < 1):
        sys.exit("--results-per-song and --limit must be positive")
    try:
        matches = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            version_info = version("yt-dlp")
        except PackageNotFoundError:
            version_info = "unknown"
        rows = candidate_searches(matches, lambda query, count: search_youtube(query, count), args.results_per_song, args.limit, args.delay)
        for row in rows:
            row["yt_dlp_version"] = version_info
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        sys.exit(str(exc))
    print(f"Saved {len(rows)} video-search candidate records to {args.output}; manually select the exact recording before downloading.")


if __name__ == "__main__":
    main()
