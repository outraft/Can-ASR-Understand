"""Transcribe local audio with faster-whisper and save benchmark hypotheses."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transcribe_manifest(
    audio_manifest: Path,
    references: Path,
    output: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    split: str = "test",
    cpu_threads: int = 0,
    model_revision: str | None = None,
    model_factory=None,
) -> int:
    if model_factory is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Install the optional ASR dependency with `python -m pip install -e ".[asr]"`') from exc
        model_factory = WhisperModel
    audio_rows = read_rows(audio_manifest)
    reference_rows = {str(row["clip_id"]): row for row in read_rows(references)}
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if output.exists():
        existing_ids = {str(row["clip_id"]) for row in read_rows(output)}
    selected = [row for row in audio_rows if split == "all" or row.get("split") == split]
    pending = [row for row in selected if str(row["clip_id"]) not in existing_ids]
    if not pending:
        return 0
    model_id = f"Systran/faster-whisper-{model_size}"
    model_source: str = model_id
    if model_revision and model_factory.__module__.startswith("faster_whisper"):
        from huggingface_hub import snapshot_download
        model_source = snapshot_download(repo_id=model_id, revision=model_revision)
    model = model_factory(model_source, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    try:
        whisper_version = version("faster-whisper")
    except PackageNotFoundError:
        whisper_version = "unknown"
    written = 0
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for audio_row in pending:
            clip_id = str(audio_row["clip_id"])
            if clip_id not in reference_rows:
                raise ValueError(f"No LRCLIB reference for clip_id {clip_id}")
            reference = reference_rows[clip_id]
            language = str(reference.get("language") or audio_row.get("language") or "")
            if not language:
                raise ValueError(f"No language for clip_id {clip_id}")
            audio_path = Path(str(audio_row["audio_path"]))
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file missing for {clip_id}: {audio_path}")
            started = time.perf_counter()
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            record = {
                "clip_id": clip_id,
                "language": language,
                "dialect": reference.get("dialect", "unassessed"),
                "dialect_status": reference.get("dialect_status", "requires_audio_review"),
                "text": text,
                "model": model_id,
                "model_revision": model_revision,
                "faster_whisper_version": whisper_version,
                "device": device,
                "compute_type": compute_type,
                "beam_size": beam_size,
                "detected_language": getattr(info, "language", None),
                "audio_sha256": audio_row.get("audio_sha256"),
                "inference_seconds": round(time.perf_counter() - started, 3),
                "transcribed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", type=Path, default=Path("data/audio/benchmark_manifest.jsonl"))
    parser.add_argument("--references", type=Path, default=Path("data/private/benchmark_references.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/private/benchmark_predictions_whisper_small.jsonl"))
    parser.add_argument("--model", default="small", help="faster-whisper model size (multilingual Whisper checkpoint)")
    parser.add_argument("--revision", help="Pinned Hugging Face model revision/commit hash")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    try:
        count = transcribe_manifest(args.audio_manifest, args.references, args.output, args.model, args.device, args.compute_type, args.beam_size, args.split, args.cpu_threads, model_revision=args.revision)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        sys.exit(str(exc))
    print(f"Transcribed {count} {args.split} audio clips with faster-whisper {args.model}; hypotheses are local-only at {args.output}.")


if __name__ == "__main__":
    main()
