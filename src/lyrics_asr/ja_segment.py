"""Cut offset-corrected utterance clips and their LRCLIB references.

Scoring a whole 4-minute song against a whole lyric sheet conflates two different
failures: mistranscription, and the model never being given the lyric's structure
(repeats, instrumental passages, ad-libs). The Hugging Face Audio Course's ASR
chapter evaluates short utterances that each carry their own reference, which is
also the unit Whisper was trained on (30-second windows). This module produces
that unit for sung Japanese by cutting each song at LRCLIB ``syncedLyrics``
boundaries, shifted by the per-song offset from :mod:`lyrics_asr.ja_align`.

Audio that no lyric line covers - intros, solos, outros - is never cut, so the
corpus reports both the full source duration and the smaller *used* duration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ja_align import build_windows
from .ja_corpus import parse_synced, read_jsonl, write_jsonl
from .evaluate import normalize

MIN_CLIP_SECONDS = 2.5
MAX_CLIP_SECONDS = 30.0
MIN_REFERENCE_CHARACTERS = 6
EDGE_PADDING_SECONDS = 0.25


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[:200]}")
    return float(result.stdout.strip())


def cut_clip(source: Path, destination: Path, start: float, duration: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"ffmpeg failed for {destination.name}: {result.stderr.strip()[:200]}")


def segment(root: Path, corpus_path: Path, alignment_path: Path, clips_dir: Path, output: Path, force: bool) -> None:
    corpus = read_jsonl(corpus_path)
    if not corpus:
        raise ValueError(f"No corpus rows at {corpus_path}")
    alignment = {str(row["clip_id"]): row for row in read_jsonl(alignment_path)}
    lyrics = {str(row["clip_id"]): row for row in read_jsonl(root / "data/references/lrclib_lyrics.jsonl")}

    utterances: list[dict[str, Any]] = []
    source_seconds = 0.0
    skipped: dict[str, int] = {}
    for entry in corpus:
        clip_id = str(entry["clip_id"])
        audio_path = Path(str(entry["audio_path"]))
        if not audio_path.is_file():
            skipped["audio_missing"] = skipped.get("audio_missing", 0) + 1
            continue
        lyric = lyrics.get(clip_id)
        if lyric is None:
            skipped["no_reference"] = skipped.get("no_reference", 0) + 1
            continue
        audio_duration = probe_duration(audio_path)
        source_seconds += audio_duration
        offset = float(alignment.get(clip_id, {}).get("offset_seconds", 0.0))
        confidence = float(alignment.get(clip_id, {}).get("alignment_confidence", 0.0))
        method = str(alignment.get(clip_id, {}).get("alignment_method", "none"))

        lines = parse_synced(str(lyric.get("synced_lyrics") or ""))
        windows = build_windows(lines, float(entry["reference_duration_seconds"]))
        index = 0
        for lyric_start, lyric_end, text in windows:
            reference = re.sub(r"\s+", "", normalize(text))
            if len(reference) < MIN_REFERENCE_CHARACTERS:
                skipped["reference_too_short"] = skipped.get("reference_too_short", 0) + 1
                continue
            start = max(0.0, lyric_start + offset - EDGE_PADDING_SECONDS)
            end = min(audio_duration, lyric_end + offset + EDGE_PADDING_SECONDS)
            duration = end - start
            if duration < MIN_CLIP_SECONDS:
                skipped["clip_too_short"] = skipped.get("clip_too_short", 0) + 1
                continue
            duration = min(duration, MAX_CLIP_SECONDS)
            index += 1
            utterance_id = f"{clip_id}-u{index:02d}"
            destination = clips_dir / clip_id / f"{utterance_id}.wav"
            if force or not destination.is_file():
                cut_clip(audio_path, destination, start, duration)
            utterances.append({
                "clip_id": utterance_id,
                "song_id": clip_id,
                "genius_id": entry.get("genius_id"),
                "lrclib_id": entry.get("lrclib_id"),
                "split": entry["split"],
                "speaker_group": entry["speaker_group"],
                "recording_group": entry.get("recording_group") or clip_id,
                "language": "ja",
                "language_status": entry.get("language_status"),
                "dialect": entry.get("dialect", "unassessed"),
                "dialect_status": entry.get("dialect_status", "requires_audio_review"),
                "song_title": entry.get("title"),
                "artist": entry.get("artist"),
                "artists": entry.get("artists") or ([entry["artist"]] if entry.get("artist") else []),
                "genius_url": entry.get("genius_url"),
                "lrclib_url": entry.get("lrclib_url"),
                "popularity_tier": entry.get("popularity_tier") or "unknown",
                "text": "\n".join(line for line in text.splitlines() if line.strip()),
                "reference_characters": len(reference),
                "reference_policy": "lrclib_assumed_gold",
                "reference_status": entry.get("reference_status", "lrclib_assumed_gold"),
                "reference_path": str(output),
                "audio_path": str(destination),
                "source_audio_path": str(audio_path),
                "youtube_url": entry.get("youtube_url"),
                "lyric_start_seconds": round(lyric_start, 2),
                "lyric_end_seconds": round(lyric_end, 2),
                "audio_start_seconds": round(start, 2),
                "duration_seconds": round(duration, 2),
                "alignment_offset_seconds": round(offset, 2),
                "alignment_confidence": round(confidence, 4),
                "alignment_method": method,
            })

    write_jsonl(output, utterances)

    used = sum(row["duration_seconds"] for row in utterances)
    print(f"Wrote {len(utterances)} utterances from {len(corpus)} songs -> {output}")
    print(f"Clips under {clips_dir} (16 kHz mono WAV, gitignored)")
    if skipped:
        print("Skipped:", skipped)
    if source_seconds:
        print(f"\nSource audio: {source_seconds / 3600:.2f} h | used (cut) audio: {used / 3600:.2f} h "
              f"({100 * used / source_seconds:.1f}% of source)")
    header = f"{'split':12s}{'songs':>6s}{'speakers':>10s}{'utts':>6s}{'used h':>9s}{'mean s':>8s}{'ref chars':>11s}"
    print("\n" + header)
    print("-" * len(header))
    for split in ("train", "validation", "test"):
        subset = [row for row in utterances if row["split"] == split]
        if not subset:
            continue
        hours = sum(row["duration_seconds"] for row in subset) / 3600
        print(f"{split:12s}{len({row['song_id'] for row in subset}):>6d}"
              f"{len({row['speaker_group'] for row in subset}):>10d}{len(subset):>6d}{hours:>9.2f}"
              f"{sum(row['duration_seconds'] for row in subset) / len(subset):>8.1f}"
              f"{sum(row['reference_characters'] for row in subset):>11d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--corpus", type=Path, default=Path("data/private/ja_corpus_local.jsonl"))
    parser.add_argument("--alignment", type=Path, default=Path("data/private/ja_alignment.jsonl"))
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips/ja"))
    parser.add_argument("--output", type=Path, default=Path("data/private/ja_utterances.jsonl"))
    parser.add_argument("--force", action="store_true", help="Re-cut clips that already exist")
    args = parser.parse_args()
    try:
        segment(args.root, args.corpus, args.alignment, args.clips_dir, args.output, args.force)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
