"""Compare multiple transcript systems against the same references."""

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

UNSEGMENTED = {"ja", "zh", "th", "lo", "km", "my"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    chars = [" " if unicodedata.category(char).startswith("P") else char for char in text]
    return re.sub(r"\s+", " ", "".join(chars)).strip()


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


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = str(row["clip_id"])
            if clip_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate clip_id {clip_id!r}")
            if "text" not in row:
                raise ValueError(f"{path}:{line_number}: missing text")
            rows[clip_id] = row
    return rows


def score(reference: str, hypothesis: str, language: str) -> dict[str, Any]:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    lang = language.lower().split("-")[0]
    unsegmented = lang in UNSEGMENTED
    ref_tokens = list(ref.replace(" ", "")) if unsegmented else ref.split()
    hyp_tokens = list(hyp.replace(" ", "")) if unsegmented else hyp.split()
    subs, dels, ins, correct = edit_counts(ref_tokens, hyp_tokens)
    token_errors = subs + dels + ins
    precision_denominator = correct + subs + ins
    recall_denominator = correct + subs + dels
    precision = correct / precision_denominator if precision_denominator else (1.0 if not ref_tokens and not hyp_tokens else 0.0)
    recall = correct / recall_denominator if recall_denominator else (1.0 if not ref_tokens and not hyp_tokens else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    char_subs, char_dels, char_ins, _ = edit_counts(ref_chars, hyp_chars)
    return {
        "wer": None if unsegmented else (token_errors / len(ref_tokens) if ref_tokens else (0.0 if not hyp_tokens else 1.0)),
        "token_error_rate": token_errors / len(ref_tokens) if ref_tokens else (0.0 if not hyp_tokens else 1.0),
        "token_f1": f1,
        "cer": (char_subs + char_dels + char_ins) / len(ref_chars) if ref_chars else (0.0 if not hyp_chars else 1.0),
        "exact_match": int(ref == hyp),
        "reference_tokens": len(ref_tokens),
        "correct_tokens": correct,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "char_errors": char_subs + char_dels + char_ins,
        "reference_characters": len(ref_chars),
    }


def evaluate(reference_path: Path, predictions: list[str], output_path: Path) -> None:
    references = read_jsonl(reference_path)
    if not references:
        raise ValueError("Reference file is empty")
    parsed: list[tuple[str, Path]] = []
    for argument in predictions:
        if "=" not in argument:
            raise ValueError(f"Prediction must be NAME=PATH, got {argument!r}")
        name, path = argument.split("=", 1)
        parsed.append((name, Path(path)))

    per_clip: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for system, path in parsed:
        hypotheses = read_jsonl(path)
        missing = sorted(references.keys() - hypotheses.keys())
        extra = sorted(hypotheses.keys() - references.keys())
        if missing or extra:
            raise ValueError(f"{system}: clip IDs must exactly match references; missing={missing[:5]}, extra={extra[:5]}")
        totals: dict[str, float] = defaultdict(float)
        language_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        language_rows: dict[str, int] = defaultdict(int)
        dialect_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        dialect_rows: dict[tuple[str, str], int] = defaultdict(int)
        has_unsegmented = False
        for clip_id in sorted(references):
            ref_row, hyp_row = references[clip_id], hypotheses[clip_id]
            language = str(ref_row.get("language", "und"))
            dialect = str(ref_row.get("dialect", "unassessed"))
            has_unsegmented |= language.lower().split("-")[0] in UNSEGMENTED
            metrics = score(str(ref_row["text"]), str(hyp_row["text"]), language)
            per_clip.append({"system": system, "clip_id": clip_id, "language": language, "dialect": dialect, **metrics})
            language_rows[language] += 1
            if dialect not in {"", "und", "unknown", "unassessed", "unassigned"}:
                dialect_rows[(language, dialect)] += 1
            for key in ("substitutions", "deletions", "insertions", "reference_tokens", "correct_tokens", "char_errors", "reference_characters", "exact_match"):
                totals[key] += metrics[key]
                language_totals[language][key] += metrics[key]
                if dialect not in {"", "und", "unknown", "unassessed", "unassigned"}:
                    dialect_totals[(language, dialect)][key] += metrics[key]

        def summarize(scope: str, counts: dict[str, float], language: str, utterances: int, disable_token_metrics: bool = False, dialect: str = "") -> dict[str, Any]:
            ref_tokens, correct = counts["reference_tokens"], counts["correct_tokens"]
            precision_den = correct + counts["substitutions"] + counts["insertions"]
            recall_den = correct + counts["substitutions"] + counts["deletions"]
            precision = correct / precision_den if precision_den else 1.0
            recall = correct / recall_den if recall_den else 1.0
            token_errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
            return {
                "system": system,
                "scope": scope,
                "language": language,
                "dialect": dialect,
                "wer": None if disable_token_metrics or language.split("-")[0] in UNSEGMENTED else (token_errors / ref_tokens if ref_tokens else 0.0),
                "token_error_rate": None if disable_token_metrics else (token_errors / ref_tokens if ref_tokens else 0.0),
                "token_f1": None if disable_token_metrics else (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
                "cer": counts["char_errors"] / counts["reference_characters"] if counts["reference_characters"] else 0.0,
                "exact_match_rate": counts["exact_match"] / utterances if utterances else 0.0,
                "substitutions": int(counts["substitutions"]),
                "deletions": int(counts["deletions"]),
                "insertions": int(counts["insertions"]),
                "utterances": utterances,
            }

        summaries.append(summarize("overall", totals, "all", len(references), has_unsegmented))
        for language, counts in sorted(language_totals.items()):
            summaries.append(summarize("language", counts, language, language_rows[language]))
        for (language, dialect), counts in sorted(dialect_totals.items()):
            summaries.append(summarize("dialect", counts, language, dialect_rows[(language, dialect)], dialect=dialect))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path = output_path.with_name(output_path.stem + "_per_clip.csv")
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    with clip_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_clip[0]))
        writer.writeheader()
        writer.writerows(per_clip)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote per-clip metrics to {clip_path}")
    print(f"Wrote corpus and per-language metrics to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=JSONL; repeat per system")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.csv"), help="Output filename stem")
    args = parser.parse_args()
    try:
        evaluate(args.references, args.prediction, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
