"""Japanese sung-lyrics scorer: CER + Character F1, with zero project imports.

This is the *same* arithmetic as ``lyrics_asr.ja_evaluate`` and
``lyrics_asr.evaluate``, inlined into one dependency-free file so the Colab run
and the local run cannot drift apart. Only the standard library is used, so it
imports on a bare Colab VM before any ``pip install`` has run.

Why CER and Character F1, and no WER
------------------------------------
Japanese is written without spaces between words, so there is no uncontroversial
word tokenization: WER would measure the segmenter as much as the model. The
Hugging Face Audio Course, Chapter 5 ("Evaluation metrics for ASR"), states it
outright:

    "Certain languages, such as Mandarin and Japanese, have no notion of
    'words', and so the WER is meaningless. Here, we revert to using the CER."
    -- https://huggingface.co/learn/audio-course/chapter5/evaluation

* **CER** = (substitutions + deletions + insertions) / reference characters.
  Lower is better. It is an error *rate*, so it is unbounded above: a system that
  hallucinates long text can exceed 100%.
* **Character F1** is the harmonic mean of character precision and recall taken
  from the same edit alignment. It is bounded in [0, 1], and because it charges
  insertions to precision instead of folding them into one rate, it separates
  "missed the lyric" from "invented extra text".

Aggregation is micro: counts are summed over utterances and then divided, so a
long utterance counts proportionally to its length.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

COUNT_KEYS = ("correct", "substitutions", "deletions", "insertions", "reference_characters", "exact_match")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    """NFKC, case-fold, punctuation -> space, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).casefold()
    chars = [" " if unicodedata.category(char).startswith("P") else char for char in text]
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def characters(text: str) -> list[str]:
    """Normalized character tokens: NFKC, case-folded, punctuation and spaces removed."""
    return list(normalize(text).replace(" ", ""))


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    """Return substitutions, deletions, insertions, and exact aligned tokens."""
    rows, cols = len(reference), len(hypothesis)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        dp[i][0] = i
    for j in range(cols + 1):
        dp[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    substitutions = deletions = insertions = correct = 0
    i, j = rows, cols
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            if reference[i - 1] == hypothesis[j - 1]:
                correct += 1
            else:
                substitutions += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions, correct


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = characters(reference), characters(hypothesis)
    subs, dels, ins, correct = edit_counts(ref, hyp)
    errors = subs + dels + ins
    precision_denominator = correct + subs + ins
    recall_denominator = correct + subs + dels
    precision = correct / precision_denominator if precision_denominator else 0.0
    recall = correct / recall_denominator if recall_denominator else 0.0
    return {
        "cer": errors / len(ref) if ref else (0.0 if not hyp else 1.0),
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "correct": correct,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "reference_characters": len(ref),
        "hypothesis_characters": len(hyp),
        "exact_match": int(ref == hyp),
    }


def summarize(counts: dict[str, float], utterances: int) -> dict[str, Any]:
    reference_characters = counts["reference_characters"]
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    precision_denominator = counts["correct"] + counts["substitutions"] + counts["insertions"]
    recall_denominator = counts["correct"] + counts["substitutions"] + counts["deletions"]
    precision = counts["correct"] / precision_denominator if precision_denominator else 0.0
    recall = counts["correct"] / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "utterances": utterances,
        "reference_characters": int(reference_characters),
        "cer": errors / reference_characters if reference_characters else 0.0,
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": f1,
        "exact_match_rate": counts["exact_match"] / utterances if utterances else 0.0,
        "substitutions": int(counts["substitutions"]),
        "deletions": int(counts["deletions"]),
        "insertions": int(counts["insertions"]),
    }


def write_markdown(corpus_rows: list[dict[str, Any]], output_stem: Path, caption: str = "") -> Path:
    """Emit the results as a Markdown table, grouped baseline-first then contribution."""
    order = {"baseline": 0, "japanese-pretrained": 1, "contribution": 2}
    rows = sorted(corpus_rows, key=lambda row: (order.get(str(row.get("tier")), 3), row["cer"]))
    best = min(row["cer"] for row in rows) if rows else None
    lines = [
        "| Tier | System | Checkpoint | Revision | CER ↓ | Character F1 ↑ | Precision | Recall | Exact | RTF |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cer = f"{row['cer'] * 100:.2f}%"
        f1 = f"{row['character_f1'] * 100:.2f}%"
        if best is not None and row["cer"] == best:
            cer, f1 = f"**{cer}**", f"**{f1}**"
        revision = str(row.get("model_revision") or "")
        factor = row.get("real_time_factor")
        cells = [
            str(row.get("tier") or "—"),
            f"`{row['system']}`",
            f"`{row.get('model') or '—'}`",
            f"`{revision[:8]}`" if revision else "—",
            cer,
            f1,
            f"{row['character_precision'] * 100:.2f}%",
            f"{row['character_recall'] * 100:.2f}%",
            f"{row['exact_match_rate'] * 100:.2f}%",
            f"{factor:.2f}×" if factor else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    reference = rows[0] if rows else {}
    lines.append("")
    lines.append(
        f"{caption}Scored on {reference.get('utterances', 0)} utterances from "
        f"{reference.get('songs', 0)} songs by {reference.get('speakers', 0)} speaker groups, "
        f"{reference.get('audio_hours', 0):.2f} h of audio, "
        f"{reference.get('reference_characters', 0):,} reference characters."
    )
    path = Path(output_stem).with_name(Path(output_stem).stem + "_table.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {path}")
    return path


def evaluate_japanese(
    reference_path: Path,
    predictions: list[str],
    output_stem: Path,
    split: str = "all",
) -> list[dict[str, Any]]:
    """Score every NAME=PATH prediction file against one frozen reference split."""
    references = {
        str(row["clip_id"]): row
        for row in read_jsonl(reference_path)
        if split == "all" or str(row.get("split")) == split
    }
    if not references:
        raise ValueError(f"No {split} reference utterances in {reference_path}")
    foreign = sorted({str(row.get("language", "und")) for row in references.values()} - {"ja"})
    if foreign:
        raise ValueError(f"This evaluator is Japanese-only; references contain language(s) {foreign}")

    parsed: list[tuple[str, Path]] = []
    for argument in predictions:
        if "=" not in argument:
            raise ValueError(f"Prediction must be NAME=PATH, got {argument!r}")
        name, path = argument.split("=", 1)
        parsed.append((name, Path(path)))

    per_utterance: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    song_rows: list[dict[str, Any]] = []

    for system, path in parsed:
        hypotheses = {str(row["clip_id"]): row for row in read_jsonl(path)}
        missing = sorted(references.keys() - hypotheses.keys())
        extra = sorted(hypotheses.keys() - references.keys())
        if missing or extra:
            raise ValueError(
                f"{system}: must cover exactly the frozen utterance set; "
                f"missing={len(missing)} {missing[:3]}, extra={len(extra)} {extra[:3]}"
            )
        totals: dict[str, float] = defaultdict(float)
        by_speaker: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        by_song: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        speaker_utterances: dict[str, int] = defaultdict(int)
        song_utterances: dict[str, int] = defaultdict(int)
        song_examples: dict[str, dict[str, Any]] = {}
        audio_seconds = inference_seconds = 0.0
        tier = model = revision = ""

        for clip_id in sorted(references):
            reference, hypothesis = references[clip_id], hypotheses[clip_id]
            metrics = score_pair(str(reference["text"]), str(hypothesis.get("text") or ""))
            speaker = str(reference.get("speaker_group", "unknown"))
            song = str(reference.get("song_id", clip_id))
            tier = tier or str(hypothesis.get("tier") or "")
            model = model or str(hypothesis.get("model") or "")
            revision = revision or str(hypothesis.get("model_revision") or "")
            audio_seconds += float(reference.get("duration_seconds") or 0.0)
            inference_seconds += float(hypothesis.get("inference_seconds") or 0.0)
            per_utterance.append({
                "system": system, "clip_id": clip_id, "song_id": song, "speaker_group": speaker,
                "split": reference.get("split"), "song_title": reference.get("song_title"),
                "duration_seconds": reference.get("duration_seconds"),
                "alignment_confidence": reference.get("alignment_confidence"),
                **{key: metrics[key] for key in (
                    "cer", "character_f1", "character_precision", "character_recall",
                    "reference_characters", "hypothesis_characters", "substitutions",
                    "deletions", "insertions", "exact_match")},
            })
            speaker_utterances[speaker] += 1
            song_utterances[song] += 1
            song_examples.setdefault(song, reference)
            for key in COUNT_KEYS:
                totals[key] += metrics[key]
                by_speaker[speaker][key] += metrics[key]
                by_song[song][key] += metrics[key]

        summary = summarize(totals, len(references))
        corpus_rows.append({
            "system": system, "tier": tier, "model": model, "model_revision": revision,
            "split": split, "scope": "corpus",
            "songs": len(song_utterances), "speakers": len(speaker_utterances),
            **summary,
            "audio_hours": round(audio_seconds / 3600, 4),
            "inference_minutes": round(inference_seconds / 60, 2),
            "real_time_factor": round(inference_seconds / audio_seconds, 4) if audio_seconds else None,
        })
        for speaker, counts in sorted(by_speaker.items()):
            speaker_rows.append({"system": system, "scope": "speaker", "speaker_group": speaker,
                                 **summarize(counts, speaker_utterances[speaker])})
        for song, counts in sorted(by_song.items()):
            example = song_examples[song]
            song_rows.append({"system": system, "scope": "song", "song_id": song,
                              "song_title": example.get("song_title"), "artist": example.get("artist"),
                              "speaker_group": example.get("speaker_group"),
                              **summarize(counts, song_utterances[song])})

    def dump(rows: list[dict[str, Any]], suffix: str) -> None:
        if not rows:
            return
        path = Path(output_stem).with_name(Path(output_stem).stem + suffix + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")

    dump(per_utterance, "_per_utterance")
    dump(corpus_rows, "_summary")
    dump(speaker_rows, "_per_speaker")
    dump(song_rows, "_per_song")
    write_markdown(corpus_rows, Path(output_stem))

    ranked = sorted(corpus_rows, key=lambda row: row["cer"])
    width = max(len(row["system"]) for row in ranked)
    header = (f"{'system':{width}s} {'tier':20s} {'CER':>8s} {'CharF1':>8s} "
              f"{'P':>7s} {'R':>7s} {'exact':>7s} {'RTF':>6s}")
    print("\n" + header)
    print("-" * len(header))
    for row in ranked:
        print(f"{row['system']:{width}s} {row['tier']:20s} {row['cer'] * 100:7.2f}% "
              f"{row['character_f1'] * 100:7.2f}% {row['character_precision'] * 100:6.2f}% "
              f"{row['character_recall'] * 100:6.2f}% {row['exact_match_rate'] * 100:6.2f}% "
              f"{row['real_time_factor'] or 0:5.2f}x")
    print(f"\n{ranked[0]['utterances']} utterances, {ranked[0]['songs']} songs, "
          f"{ranked[0]['speakers']} speakers, {ranked[0]['audio_hours']:.2f} h scored audio, "
          f"{ranked[0]['reference_characters']} reference characters")
    return corpus_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True, help="Japanese utterance manifest with text")
    parser.add_argument("--prediction", action="append", required=True, help="NAME=JSONL; repeat per system")
    parser.add_argument("--output", type=Path, default=Path("reports/ja_benchmark"))
    parser.add_argument("--split", default="all", choices=("train", "validation", "test", "all"))
    args = parser.parse_args()
    try:
        evaluate_japanese(args.references, args.prediction, args.output, args.split)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
