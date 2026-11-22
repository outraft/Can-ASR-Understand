"""Download authorized YouTube audio URLs into local ignored storage with yt-dlp."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


def validate_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and (parsed.hostname or "").lower() in YOUTUBE_HOSTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(
    sources: list[dict[str, Any]],
    audio_dir: Path,
    archive: Path,
    manifest_path: Path,
    downloader_factory=None,
    delay: float = 1.0,
) -> tuple[int, int]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install the optional media dependency: python -m pip install -e .[media]") from exc
    factory = downloader_factory or yt_dlp.YoutubeDL
    audio_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    old = json.loads(line)
                    existing[str(old["clip_id"])] = old
                except (KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid audio manifest at {manifest_path}:{line_number}") from exc
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writeinfojson": True,
        "download_archive": str(archive),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    downloaded = failed = 0
    with manifest_path.open("a", encoding="utf-8", newline="\n") as manifest:
        for row in sources:
            url = str(row.get("youtube_url", ""))
            if not validate_youtube_url(url):
                raise ValueError(f"{row.get('clip_id', '<unknown>')}: expected a direct https YouTube watch URL")
            clip_id = str(row["clip_id"])
            prior = existing.get(clip_id)
            if prior and prior.get("youtube_url") != url:
                raise ValueError(f"{clip_id}: source URL changed; use a new clip_id to preserve provenance")
            if prior and Path(str(prior.get("audio_path", ""))).is_file():
                continue
            with factory(options) as ydl:
                try:
                    info = ydl.extract_info(url, download=True)
                except yt_dlp.utils.DownloadError as exc:
                    match = re.search(r"HTTP Error (\d{3})", str(exc))
                    status = int(match.group(1)) if match else None
                    manifest.write(json.dumps({
                        "clip_id": clip_id,
                        "genius_id": row.get("genius_id"),
                        "lrclib_id": row.get("lrclib_id"),
                        "split": row.get("split"),
                        "speaker_group": row.get("speaker_group"),
                        "youtube_url": url,
                        "download_status": "blocked_or_unavailable",
                        "http_status": status,
                        "yt_dlp_version": yt_dlp.version.__version__,
                        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")
                    manifest.flush()
                    failed += 1
                    # Stop on an access/rate-limit response; don't keep probing.
                    if status in {401, 403, 429}:
                        break
                    time.sleep(max(0.0, delay))
                    continue
                video_id = str((info or {}).get("id") or "")
                if not video_id:
                    raise RuntimeError(f"yt-dlp returned no video ID for {row.get('clip_id', '<unknown>')}")
                audio_path = audio_dir / f"{video_id}.wav"
                if not audio_path.is_file():
                    raise RuntimeError(f"Expected WAV output was not produced for video ID {video_id}; check ffmpeg installation")
                record = {
                    "clip_id": row["clip_id"],
                    "genius_id": row.get("genius_id"),
                    "lrclib_id": row.get("lrclib_id"),
                    "split": row.get("split"),
                    "speaker_group": row.get("speaker_group"),
                    "recording_group": row.get("recording_group"),
                    "language": row.get("language"),
                    "language_status": row.get("language_status"),
                    "dialect": row.get("dialect", "unassessed"),
                    "dialect_status": row.get("dialect_status", "requires_audio_review"),
                    "youtube_url": url,
                    "video_id": video_id,
                    "audio_path": str(audio_path),
                    "duration_seconds": (info or {}).get("duration"),
                    "audio_sha256": sha256_file(audio_path),
                    "download_status": "downloaded",
                    "yt_dlp_version": yt_dlp.version.__version__,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                existing[clip_id] = record
                downloaded += 1
            time.sleep(max(0.0, delay))
    return downloaded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/cleaned/youtube_sources.jsonl"), help="Manually curated, permission-checked URLs")
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--archive", type=Path, default=Path("data/private/yt-dlp-archive.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/audio/download_manifest.jsonl"))
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between sequential video downloads")
    args = parser.parse_args()
    try:
        sources = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        downloaded, failed = download_sources(sources, args.audio_dir, args.archive, args.manifest, delay=args.delay)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        sys.exit(str(exc))
    print(f"Downloaded {downloaded} audio files; {failed} unavailable/blocked. Local media and manifest are gitignored.")


if __name__ == "__main__":
    main()
