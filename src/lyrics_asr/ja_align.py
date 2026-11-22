"""Estimate the per-song time offset between LRCLIB timestamps and the local audio.

LRCLIB ``syncedLyrics`` timestamps are relative to the released master. A YouTube
music video for the same recording often prepends an intro, so lyric time ``t``
corresponds to audio time ``t + offset``. Without correcting that offset, a cut
utterance would hold the wrong lyric and every metric would measure a reference
bug instead of a model.

The estimate is deliberately a *single scalar per song*. The run has two stages so
the expensive part is paid once:

1. ``transcribe`` — one resumable pass of a small zero-shot model over each full
   song, storing timestamped segments. Nothing about the lyric is given to it.
2. ``offset`` — a pure post-processing grid search for the offset that best lines
   the stored segments up with the lyric lines, scored by bag-of-characters F1
   (order-insensitive, so it measures time alignment rather than transcription
   quality).

When the best score is below ``MIN_ALIGNMENT_CONFIDENCE`` the estimate is not
trusted and the offset falls back to the duration prior
``max(0, video_duration - reference_duration)``, which is the usual length of an
added intro. Songs are never dropped on the basis of this score, so the frozen
test split stays independent of any model's accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ja_corpus import parse_synced, read_jsonl, write_jsonl
from .evaluate import normalize

ALIGNMENT_MODEL = "Systran/faster-whisper-base"
ALIGNMENT_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
# Only the opening of each track is decoded. Estimating one scalar needs a handful
# of overlapping windows, not the whole song, and this keeps the pass affordable.
PROBE_SECONDS = 150.0
SAMPLE_RATE = 16000
OFFSET_STEP = 0.5
OFFSET_MARGIN = 8.0
MAX_OFFSET = 80.0
MIN_ALIGNMENT_CONFIDENCE = 0.35
WINDOW_SECONDS = 28.0
MAX_LINE_SECONDS = 12.0
MAX_LINE_GAP_SECONDS = 5.0


def build_windows(
    lines: list[tuple[float, str]],
    song_duration: float,
    window_seconds: float = WINDOW_SECONDS,
) -> list[tuple[float, float, str]]:
    """Merge consecutive timestamped lines into windows of at most ``window_seconds``.

    A line ends when the next line starts, capped at ``MAX_LINE_SECONDS`` so a long
    instrumental break is not charged to the previous line. A gap longer than
    ``MAX_LINE_GAP_SECONDS`` closes the window, which keeps instrumental passages
    out of the cut audio.
    """
    windows: list[tuple[float, float, str]] = []
    start: float | None = None
    end: float = 0.0
    texts: list[str] = []
    for index, (timestamp, text) in enumerate(lines):
        following = lines[index + 1][0] if index + 1 < len(lines) else min(timestamp + 6.0, song_duration)
        line_end = min(max(following, timestamp + 1.0), timestamp + MAX_LINE_SECONDS)
        if start is None:
            start, end, texts = timestamp, line_end, [text]
            continue
        if (line_end - start) <= window_seconds and (timestamp - end) <= MAX_LINE_GAP_SECONDS:
            end, _ = line_end, texts.append(text)
        else:
            windows.append((start, end, "\n".join(texts)))
            start, end, texts = timestamp, line_end, [text]
    if start is not None:
        windows.append((start, end, "\n".join(texts)))
    return windows


def bag_f1(reference: str, hypothesis: str) -> float:
    """Order-insensitive character F1, used only as an alignment score."""
    ref = Counter(normalize(reference).replace(" ", ""))
    hyp = Counter(normalize(hypothesis).replace(" ", ""))
    if not ref or not hyp:
        return 0.0
    overlap = sum((ref & hyp).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def transcribe_songs(
    root: Path,
    corpus: list[dict[str, Any]],
    output: Path,
    device: str,
    compute_type: str,
    probe_seconds: float = PROBE_SECONDS,
) -> int:
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        raise RuntimeError('Install the ASR extra: python -m pip install -e ".[asr]"') from exc
    from huggingface_hub import snapshot_download

    done = {str(row["clip_id"]) for row in read_jsonl(output)}
    pending = [row for row in corpus if str(row["clip_id"]) not in done and row.get("audio_path")]
    if not pending:
        return 0
    model = WhisperModel(
        snapshot_download(repo_id=ALIGNMENT_MODEL, revision=ALIGNMENT_REVISION),
        device=device,
        compute_type=compute_type,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for row in pending:
            audio_path = Path(str(row["audio_path"]))
            if not audio_path.is_file():
                raise FileNotFoundError(f"{row['clip_id']}: audio missing at {audio_path}")
            started = time.perf_counter()
            audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
            if probe_seconds > 0:
                audio = audio[: int(probe_seconds * SAMPLE_RATE)]
            segments, _ = model.transcribe(
                audio, language="ja", beam_size=1, vad_filter=False, condition_on_previous_text=False
            )
            stored = [
                {"start": round(segment.start, 3), "end": round(segment.end, 3), "text": segment.text.strip()}
                for segment in segments
                if segment.text.strip()
            ]
            handle.write(json.dumps({
                "clip_id": str(row["clip_id"]),
                "audio_path": str(audio_path),
                "model": ALIGNMENT_MODEL,
                "model_revision": ALIGNMENT_REVISION,
                "probe_seconds": probe_seconds,
                "segments": stored,
                "inference_seconds": round(time.perf_counter() - started, 2),
                "transcribed_at_utc": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(f"  [{written}/{len(pending)}] {row['clip_id']} {str(row.get('title'))[:28]} "
                  f"({len(stored)} segments, {time.perf_counter() - started:.0f}s)", flush=True)
    return written


def window_score(
    segments: list[tuple[float, float, Counter]],
    start: float,
    end: float,
    reference: Counter,
) -> float:
    """Soft character F1 between a window's reference and the audio under it.

    A segment that only partly overlaps the window contributes its characters in
    proportion to the overlap. Without that weighting the score would be flat
    across several seconds of candidate offsets - a segment is either "inside" the
    window or not - and the search would stop at the first offset on the plateau
    instead of the true shift.
    """
    hypothesis: Counter = Counter()
    for segment_start, segment_end, counts in segments:
        overlap = min(end, segment_end) - max(start, segment_start)
        if overlap <= 0:
            continue
        fraction = overlap / max(segment_end - segment_start, 1e-6)
        for character, count in counts.items():
            hypothesis[character] += count * fraction
    hypothesis_total = sum(hypothesis.values())
    reference_total = sum(reference.values())
    if not hypothesis_total or not reference_total:
        return 0.0
    shared = sum(min(count, hypothesis.get(character, 0.0)) for character, count in reference.items())
    if not shared:
        return 0.0
    precision, recall = shared / hypothesis_total, shared / reference_total
    return 2 * precision * recall / (precision + recall)


def estimate_offset(
    segments: list[dict[str, Any]],
    windows: list[tuple[float, float, str]],
    duration_delta: float,
) -> tuple[float, float]:
    """Grid-search the offset that best aligns lyric windows to stored segments."""
    prepared = [
        (float(segment["start"]), float(segment["end"]), Counter(normalize(str(segment["text"])).replace(" ", "")))
        for segment in segments
        if float(segment["end"]) > float(segment["start"])
    ]
    prepared = [item for item in prepared if item[2]]
    references = [
        (start, end, Counter(normalize(text).replace(" ", "")))
        for start, end, text in windows
    ]
    references = [item for item in references if item[2]]
    if not prepared or not references:
        return max(0.0, duration_delta), 0.0

    prior = max(0.0, duration_delta)
    low = min(-OFFSET_MARGIN, prior - OFFSET_MARGIN)
    high = min(MAX_OFFSET, max(OFFSET_MARGIN, prior + OFFSET_MARGIN))
    best_offset, best_score = prior, -1.0
    steps = int(round((high - low) / OFFSET_STEP)) + 1
    for step in range(steps):
        offset = low + step * OFFSET_STEP
        weighted = weight = 0.0
        for start, end, reference in references:
            size = sum(reference.values())
            weighted += window_score(prepared, start + offset, end + offset, reference) * size
            weight += size
        score = weighted / weight if weight else 0.0
        if score > best_score:
            best_offset, best_score = offset, score
    return best_offset, max(best_score, 0.0)


def align(root: Path, corpus_path: Path, song_asr_path: Path, output: Path, device: str, compute_type: str, stage: str, probe_seconds: float = PROBE_SECONDS) -> None:
    corpus = read_jsonl(root / corpus_path if not corpus_path.is_absolute() else corpus_path)
    if not corpus:
        raise ValueError(f"No corpus rows at {corpus_path}")
    lyrics = {str(row["clip_id"]): row for row in read_jsonl(root / "data/references/lrclib_lyrics.jsonl")}

    if stage in {"transcribe", "all"}:
        print(f"Alignment pass with {ALIGNMENT_MODEL} @ {ALIGNMENT_REVISION[:8]} "
              f"({device}/{compute_type}, first {probe_seconds:.0f}s per song)")
        written = transcribe_songs(root, corpus, song_asr_path, device, compute_type, probe_seconds)
        print(f"Transcribed {written} new songs into {song_asr_path}")
    if stage == "transcribe":
        return

    stored = {str(row["clip_id"]): row for row in read_jsonl(song_asr_path)}
    rows: list[dict[str, Any]] = []
    for entry in corpus:
        clip_id = str(entry["clip_id"])
        lyric = lyrics.get(clip_id)
        if lyric is None:
            continue
        reference_duration = float(entry["reference_duration_seconds"])
        lines = parse_synced(str(lyric.get("synced_lyrics") or ""))
        windows = build_windows(lines, reference_duration)
        delta = float(entry.get("source_duration_delta_seconds") or 0.0)
        asr = stored.get(clip_id)
        if asr and asr.get("segments"):
            offset, confidence = estimate_offset(asr["segments"], windows, delta)
            method = "asr_grid_search"
            if confidence < MIN_ALIGNMENT_CONFIDENCE:
                offset, method = max(0.0, delta), "duration_prior_fallback"
        else:
            offset, confidence, method = max(0.0, delta), 0.0, "duration_prior_no_asr"
        rows.append({
            "clip_id": clip_id,
            "split": entry["split"],
            "speaker_group": entry["speaker_group"],
            "title": entry.get("title"),
            "artist": entry.get("artist"),
            "reference_duration_seconds": reference_duration,
            "video_duration_seconds": entry.get("video_duration_seconds"),
            "source_duration_delta_seconds": delta,
            "offset_seconds": round(offset, 2),
            "alignment_confidence": round(confidence, 4),
            "alignment_method": method,
            "alignment_model": ALIGNMENT_MODEL,
            "alignment_model_revision": ALIGNMENT_REVISION,
            "window_count": len(windows),
        })
    write_jsonl(output, rows)

    print(f"\nAligned {len(rows)} songs -> {output}")
    print("method:", dict(Counter(row["alignment_method"] for row in rows)))
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        confident = sum(1 for row in subset if row["alignment_confidence"] >= MIN_ALIGNMENT_CONFIDENCE)
        mean = sum(row["alignment_confidence"] for row in subset) / len(subset)
        shifted = sum(1 for row in subset if abs(row["offset_seconds"]) >= 1.0)
        print(f"  {split:11s} {len(subset):3d} songs  mean confidence {mean:.3f}  "
              f"confident {confident}/{len(subset)}  non-zero offset {shifted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--corpus", type=Path, default=Path("data/private/ja_corpus_local.jsonl"))
    parser.add_argument("--song-asr", type=Path, default=Path("data/private/ja_song_asr_base.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/private/ja_alignment.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--stage", choices=("transcribe", "offset", "all"), default="all")
    parser.add_argument("--probe-seconds", type=float, default=PROBE_SECONDS,
                        help="Decode only this many opening seconds per song; 0 decodes the whole track")
    args = parser.parse_args()
    try:
        align(args.root, args.corpus, args.song_asr, args.output, args.device, args.compute_type, args.stage, args.probe_seconds)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
