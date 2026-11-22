"""Run every eligible Japanese ASR system over the same frozen utterance clips.

All systems in ``config/benchmark_models.json`` are CTranslate2 checkpoints, so one
faster-whisper code path covers the multilingual Whisper baselines and the
Japanese-specialised contribution models. Each system decodes with
``language="ja"`` and ``task="transcribe"`` pinned, at a pinned revision, over the
identical clip list - that is what makes the numbers comparable.

Writing is append-only and keyed by utterance id, so an interrupted run resumes
without re-decoding anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .ja_corpus import read_jsonl


def load_systems(config_path: Path, only: list[str]) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    systems = config["systems"]
    if only:
        names = {name.casefold() for name in only}
        systems = [system for system in systems if system["name"].casefold() in names]
        missing = names - {system["name"].casefold() for system in systems}
        if missing:
            raise ValueError(f"Unknown system name(s): {sorted(missing)}")
    return systems


def run_system(system: dict[str, Any], utterances: list[dict[str, Any]], output: Path, decode: dict[str, Any]) -> tuple[int, float]:
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    done = {str(row["clip_id"]) for row in read_jsonl(output)}
    pending = [row for row in utterances if str(row["clip_id"]) not in done]
    if not pending:
        print(f"  {system['name']}: already complete ({len(done)} utterances)")
        return 0, 0.0

    local = snapshot_download(repo_id=system["repo_id"], revision=system["revision"])
    model = WhisperModel(
        local,
        device=system.get("device", "cpu"),
        compute_type=system.get("compute_type", "int8"),
        cpu_threads=int(system.get("cpu_threads", 0)),
    )
    try:
        faster_whisper_version = version("faster-whisper")
    except PackageNotFoundError:
        faster_whisper_version = "unknown"

    output.parent.mkdir(parents=True, exist_ok=True)
    written, spent = 0, 0.0
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for row in pending:
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.is_file():
                raise FileNotFoundError(f"{row['clip_id']}: clip missing at {audio_path}")
            started = time.perf_counter()
            segments, info = model.transcribe(
                str(audio_path),
                language=decode["language"],
                task=decode.get("task", "transcribe"),
                beam_size=int(system.get("beam_size", decode.get("beam_size", 5))),
                vad_filter=bool(decode.get("vad_filter", False)),
                condition_on_previous_text=bool(decode.get("condition_on_previous_text", False)),
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            elapsed = time.perf_counter() - started
            spent += elapsed
            handle.write(json.dumps({
                "clip_id": str(row["clip_id"]),
                "song_id": row.get("song_id"),
                "split": row.get("split"),
                "speaker_group": row.get("speaker_group"),
                "language": decode["language"],
                "dialect": row.get("dialect", "unassessed"),
                "dialect_status": row.get("dialect_status", "requires_audio_review"),
                "text": text,
                "system": system["name"],
                "model": system["repo_id"],
                "model_revision": system["revision"],
                "model_card": system.get("model_card"),
                "tier": system.get("tier"),
                "decode_language": decode["language"],
                "decode_task": decode.get("task", "transcribe"),
                "beam_size": int(system.get("beam_size", decode.get("beam_size", 5))),
                "device": system.get("device", "cpu"),
                "compute_type": system.get("compute_type", "int8"),
                "faster_whisper_version": faster_whisper_version,
                "detected_language": getattr(info, "language", None),
                "audio_seconds": float(row.get("duration_seconds") or 0.0),
                "inference_seconds": round(elapsed, 3),
                "transcribed_at_utc": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 25 == 0 or written == len(pending):
                print(f"    {system['name']}: {written}/{len(pending)} utterances "
                      f"({spent / 60:.1f} min so far)", flush=True)
    return written, spent


def benchmark(config_path: Path, utterance_path: Path, split: str, output_dir: Path, only: list[str]) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decode = config["decode_settings"]
    systems = load_systems(config_path, only)
    utterances = [row for row in read_jsonl(utterance_path) if split == "all" or row.get("split") == split]
    if not utterances:
        raise ValueError(f"No {split} utterances in {utterance_path}")
    audio_seconds = sum(float(row.get("duration_seconds") or 0.0) for row in utterances)
    print(f"{len(utterances)} {split} utterances ({audio_seconds / 3600:.2f} h) x {len(systems)} systems")

    for system in systems:
        output = output_dir / f"ja_{split}_predictions_{system['name']}.jsonl"
        written, spent = run_system(system, utterances, output, decode)
        if written:
            print(f"  {system['name']}: wrote {written} -> {output} "
                  f"({spent / 60:.1f} min, {spent / audio_seconds:.2f}x real time)")

    print("\nPrediction files:")
    for system in systems:
        output = output_dir / f"ja_{split}_predictions_{system['name']}.jsonl"
        print(f"  --prediction {system['name']}={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("config/benchmark_models.json"))
    parser.add_argument("--utterances", type=Path, default=Path("data/private/ja_utterances.jsonl"))
    parser.add_argument("--split", default="test", choices=("train", "validation", "test", "all"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/private"))
    parser.add_argument("--system", action="append", default=[], help="Run only this system name; repeatable")
    args = parser.parse_args()
    try:
        benchmark(args.models, args.utterances, args.split, args.output_dir, args.system)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
