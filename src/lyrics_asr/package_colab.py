"""Build the self-contained Colab bundle: manifest + transcoded clips + runner.

CPU inference on a laptop is the bottleneck in this project, not the data: six
pretrained systems over ~1.6 h of utterance audio is several hours at int8 on
CPU and under half an hour on a free Colab T4 at float16. This module packages
everything the T4 needs into one zip, and *nothing else* - no full-length source
songs, no absolute local paths, no English rows (there are none left to carry).

Clip codec
----------
The cut clips are 16 kHz mono 16-bit PCM, which is 256 kbit/s. FLAC only shaves
~17% off dense music, so a lossless bundle is still hundreds of megabytes. The
default is therefore Opus at 48 kbit/s mono: at a 16 kHz sampling rate the signal
occupies 8 kHz of bandwidth, which Opus encodes essentially transparently at that
rate, and the bundle drops to tens of megabytes. Every system in the benchmark
reads the same encoded clips, so the comparison between rows is unaffected; pass
``--codec flac`` (lossless) or ``--codec wav`` (byte-identical to local) when the
upload size does not matter and you want the audio untouched.

The manifest's ``audio_path`` is rewritten to a bundle-relative path and
local-only fields are dropped, so the zip is portable and leaks no filesystem
layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from .ja_corpus import read_jsonl

# Fields that describe the utterance and its reference. Everything else in
# ja_utterances.jsonl is local bookkeeping (absolute paths, source WAV names) and
# has no business travelling to a cloud VM.
PORTABLE_FIELDS = (
    "clip_id", "song_id", "genius_id", "lrclib_id", "split", "speaker_group",
    "recording_group", "language", "language_status", "dialect", "dialect_status",
    "song_title", "artist", "artists", "genius_url", "lrclib_url", "popularity_tier",
    "text", "reference_characters", "reference_policy", "reference_status",
    "lyric_start_seconds", "lyric_end_seconds", "audio_start_seconds",
    "duration_seconds", "alignment_offset_seconds", "alignment_confidence",
    "alignment_method",
)

CODECS = {
    "opus": {"extension": ".opus", "args": ["-c:a", "libopus", "-vbr", "on", "-application", "audio"]},
    "flac": {"extension": ".flac", "args": ["-c:a", "flac", "-compression_level", "8"]},
    "wav": {"extension": ".wav", "args": ["-c:a", "pcm_s16le"]},
}

HANDOFF_FILES = ("run_colab.py", "ja_eval_standalone.py", "colab_instructions.md", "colab_runner.ipynb")


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transcode(source: Path, destination: Path, codec: str, bitrate: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    settings = CODECS[codec]
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "16000"]
    command += settings["args"]
    if codec == "opus":
        command += ["-b:a", bitrate]
    command.append(str(destination))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def portable_row(row: dict[str, Any], audio_path: str) -> dict[str, Any]:
    out = {field: row[field] for field in PORTABLE_FIELDS if field in row}
    out["audio_path"] = audio_path
    return out


def package(
    root: Path,
    utterances_path: Path,
    output: Path,
    handoff_dir: Path,
    codec: str,
    bitrate: str,
    splits: list[str],
    force: bool,
) -> Path:
    rows = read_jsonl(utterances_path)
    if not rows:
        raise ValueError(f"No utterances in {utterances_path}")
    foreign = sorted({str(row.get("language", "und")) for row in rows} - {"ja"})
    if foreign:
        raise ValueError(f"This corpus is Japanese-only but the manifest contains {foreign}")
    selected = [row for row in rows if str(row.get("split")) in splits]
    if not selected:
        raise ValueError(f"No utterances in splits {splits}")

    stage = root / "data" / "private" / "colab_stage" / codec
    extension = CODECS[codec]["extension"]
    manifest: list[dict[str, Any]] = []
    source_bytes = staged_bytes = 0

    for index, row in enumerate(selected, start=1):
        source = root / str(row["audio_path"])
        if not source.exists():
            raise FileNotFoundError(f"missing clip {source} for {row['clip_id']}; run lyrics-asr-ja-segment")
        relative = f"clips/{row['song_id']}/{row['clip_id']}{extension}"
        destination = stage / row["song_id"] / f"{row['clip_id']}{extension}"
        if force or not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
            transcode(source, destination, codec, bitrate)
        source_bytes += source.stat().st_size
        staged_bytes += destination.stat().st_size
        manifest.append(portable_row(row, relative))
        if index % 100 == 0 or index == len(selected):
            print(f"  encoded {index}/{len(selected)} ({codec})")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("ja_utterances.jsonl", manifest_text)
        for name in HANDOFF_FILES:
            path = handoff_dir / name
            if path.exists():
                archive.write(path, name)
        config = root / "config" / "benchmark_models.json"
        if config.exists():
            archive.write(config, "benchmark_models.json")
        for row in manifest:
            relative = str(row["audio_path"])
            # Audio is already compressed; storing it skips a pointless deflate pass.
            archive.write(stage / relative[len("clips/"):], relative,
                          compress_type=zipfile.ZIP_STORED if codec != "wav" else None)

    counts: dict[str, int] = {}
    seconds: dict[str, float] = {}
    for row in manifest:
        split = str(row.get("split"))
        counts[split] = counts.get(split, 0) + 1
        seconds[split] = seconds.get(split, 0.0) + float(row.get("duration_seconds") or 0.0)
    print(f"\nBundle: {output}")
    print(f"  codec        {codec}" + (f" @ {bitrate}" if codec == "opus" else " (lossless)"))
    print(f"  clips        {len(manifest)} from {len({row['song_id'] for row in manifest})} songs")
    print(f"  source PCM   {human(source_bytes)}")
    print(f"  encoded      {human(staged_bytes)} ({staged_bytes / source_bytes * 100:.1f}% of PCM)")
    print(f"  zip          {human(output.stat().st_size)}")
    print(f"  sha256       {sha256(output)}")
    for split in ("train", "validation", "test"):
        if split in counts:
            print(f"  {split:11s}  {counts[split]:4d} clips  {seconds[split] / 3600:.2f} h")
    print("\nNext: upload the zip to Colab (or Google Drive) and follow colab_instructions.md")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--utterances", type=Path, default=Path("data/private/ja_utterances.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("colab_handoff/colab_bundle.zip"))
    parser.add_argument("--handoff-dir", type=Path, default=Path("colab_handoff"))
    parser.add_argument("--codec", default="opus", choices=sorted(CODECS))
    parser.add_argument("--bitrate", default="48k", help="Opus bitrate (ignored for flac/wav)")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--force", action="store_true", help="Re-encode clips that already exist")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required to transcode clips; install it and retry")
    try:
        package(
            args.root, args.utterances, args.output, args.handoff_dir, args.codec,
            args.bitrate, [part.strip() for part in args.splits.split(",") if part.strip()],
            args.force,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
