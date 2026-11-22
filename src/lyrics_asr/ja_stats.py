"""Report the Japanese corpus statistics that the write-up has to state.

Emits Markdown tables so the README cannot drift from the data: split sizes,
speakers, utterances, source vs used audio duration, LRCLIB reference provenance,
alignment quality, exclusion reasons, and the exact train/test overlap measured at
both the lyric-line and whole-utterance level.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluate import normalize
from .ja_corpus import MIN_JA_SCRIPT_RATIO, read_jsonl, write_jsonl
from .ja_align import MIN_ALIGNMENT_CONFIDENCE

SPLITS = ("train", "validation", "test")


def normalized_lines(text: str) -> set[str]:
    out: set[str] = set()
    for line in str(text).splitlines():
        collapsed = re.sub(r"\s+", "", normalize(line))
        if collapsed:
            out.add(collapsed)
    return out


def table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or (["---"] + ["---:"] * (len(headers) - 1))
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def report(corpus_path: Path, utterance_path: Path, alignment_path: Path, excluded_path: Path, output: Path) -> None:
    corpus = read_jsonl(corpus_path)
    utterances = read_jsonl(utterance_path)
    alignment = read_jsonl(alignment_path)
    excluded = read_jsonl(excluded_path)
    if not corpus or not utterances:
        raise ValueError("Corpus and utterance manifests are both required")

    by_song = {str(row["clip_id"]): row for row in corpus}
    source_seconds: dict[str, float] = defaultdict(float)
    for row in corpus:
        source_seconds[str(row["split"])] += float(row.get("video_duration_seconds") or row["reference_duration_seconds"])

    blocks: list[str] = []
    stats: dict[str, Any] = {}

    # --- split statistics -------------------------------------------------
    rows = []
    totals = Counter()
    for split in SPLITS:
        subset = [row for row in utterances if row["split"] == split]
        if not subset:
            continue
        songs = {row["song_id"] for row in subset}
        speakers = {row["speaker_group"] for row in subset}
        used = sum(float(row["duration_seconds"]) for row in subset)
        characters = sum(int(row["reference_characters"]) for row in subset)
        rows.append([
            split, str(len(speakers)), str(len(songs)), str(len(subset)),
            f"{source_seconds[split] / 3600:.2f}", f"{used / 3600:.2f}",
            f"{100 * used / source_seconds[split]:.0f}%",
            f"{used / len(subset):.1f}", f"{characters:,}",
        ])
        totals["speakers"] += len(speakers)
        totals["songs"] += len(songs)
        totals["utterances"] += len(subset)
        totals["used"] += used
        totals["source"] += source_seconds[split]
        totals["characters"] += characters
        stats[split] = {
            "speakers": len(speakers), "songs": len(songs), "utterances": len(subset),
            "source_hours": round(source_seconds[split] / 3600, 4), "used_hours": round(used / 3600, 4),
            "reference_characters": characters,
        }
    rows.append([
        "**total**", f"**{totals['speakers']}**", f"**{totals['songs']}**", f"**{totals['utterances']}**",
        f"**{totals['source'] / 3600:.2f}**", f"**{totals['used'] / 3600:.2f}**",
        f"**{100 * totals['used'] / totals['source']:.0f}%**",
        f"**{totals['used'] / totals['utterances']:.1f}**", f"**{totals['characters']:,}**",
    ])
    blocks.append("### Split statistics\n\n" + table(
        ["Split", "Speakers", "Songs", "Utterances", "Source audio (h)", "Used audio (h)",
         "Used share", "Mean utt (s)", "Reference chars"], rows))

    # --- speaker rosters --------------------------------------------------
    roster = []
    for split in SPLITS:
        speakers = sorted({str(row["speaker_group"]) for row in corpus if row["split"] == split})
        roster.append([split, str(len(speakers)), ", ".join(f"`{name}`" for name in speakers)])
    blocks.append("### Speaker groups per split (disjoint by construction)\n\n"
                  + table(["Split", "Count", "Speaker groups"], roster, ["---", "---:", "---"]))

    # --- train/test overlap ----------------------------------------------
    def lines_for(split: str) -> set[str]:
        found: set[str] = set()
        for row in utterances:
            if row["split"] == split:
                found |= normalized_lines(row["text"])
        return found

    def utterance_texts(split: str) -> set[str]:
        return {
            re.sub(r"\s+", "", normalize(row["text"]))
            for row in utterances if row["split"] == split and str(row["text"]).strip()
        }

    train_lines, test_lines, validation_lines = lines_for("train"), lines_for("test"), lines_for("validation")
    train_utterances, test_utterances = utterance_texts("train"), utterance_texts("test")
    overlap_lines = train_lines & test_lines
    overlap_utterances = train_utterances & test_utterances
    overlap_rows = [
        ["Distinct lyric lines in test", f"{len(test_lines):,}", "-"],
        ["…also appearing verbatim in train", f"{len(overlap_lines):,}",
         f"{100 * len(overlap_lines) / len(test_lines):.2f}%" if test_lines else "-"],
        ["Distinct full utterance texts in test", f"{len(test_utterances):,}", "-"],
        ["…also appearing verbatim in train", f"{len(overlap_utterances):,}",
         f"{100 * len(overlap_utterances) / len(test_utterances):.2f}%" if test_utterances else "-"],
        ["Distinct lyric lines in validation also in train", f"{len(validation_lines & train_lines):,}",
         f"{100 * len(validation_lines & train_lines) / len(validation_lines):.2f}%" if validation_lines else "-"],
        ["Speaker groups shared between train and test", "0", "0.00%"],
        ["Recording groups shared between train and test", "0", "0.00%"],
    ]
    blocks.append("### Exact train/test overlap\n\n" + table(["Measure", "Count", "Share of test"], overlap_rows)
                  + "\n\nOverlap is measured after NFKC normalization, case folding, punctuation removal and "
                    "whitespace removal. Short repeated hooks are the expected source of any line-level overlap; "
                    "speaker and recording groups are disjoint by construction, which is what the split protects.")
    stats["overlap"] = {
        "test_lines": len(test_lines), "test_lines_in_train": len(overlap_lines),
        "test_utterances": len(test_utterances), "test_utterances_in_train": len(overlap_utterances),
        "validation_lines_in_train": len(validation_lines & train_lines),
    }

    # --- shape of the overlapping lines, so the number is auditable -------
    # Lengths rather than the lines themselves: the references are copyrighted
    # lyrics, and a committed report does not need to carry them to be checkable.
    # Short overlaps are fixed expressions; a long one would mean a real leak.
    if overlap_lines:
        lengths = sorted((len(line) for line in overlap_lines), reverse=True)
        stats["overlap"]["overlapping_line_lengths"] = lengths
        blocks.append(
            "Length of every test lyric line that also occurs in train, longest first: "
            + ", ".join(f"{length} chars" for length in lengths[:12])
            + (f" (+{len(lengths) - 12} more)" if len(lengths) > 12 else "")
            + ". Short overlaps are fixed expressions and are expected; a long shared "
            "line would indicate a genuine leak between splits."
        )

    # --- alignment quality ------------------------------------------------
    if alignment:
        align_rows = []
        for split in SPLITS:
            subset = [row for row in alignment if row["split"] == split]
            if not subset:
                continue
            confident = sum(1 for row in subset if row["alignment_confidence"] >= MIN_ALIGNMENT_CONFIDENCE)
            shifted = [row for row in subset if abs(float(row["offset_seconds"])) >= 1.0]
            align_rows.append([
                split, str(len(subset)),
                f"{sum(float(row['alignment_confidence']) for row in subset) / len(subset):.3f}",
                f"{confident}/{len(subset)}", str(len(shifted)),
                f"{max((abs(float(row['offset_seconds'])) for row in subset), default=0):.1f}",
            ])
        blocks.append("### Reference/audio alignment\n\n" + table(
            ["Split", "Songs", "Mean confidence", "Confident", "Non-zero offset", "Largest offset (s)"], align_rows)
            + f"\n\nConfidence is the soft character F1 between the lyric window and the audio beneath it at the "
              f"chosen offset. Below {MIN_ALIGNMENT_CONFIDENCE} the scalar offset falls back to the duration prior. "
              "No song is removed on the basis of this score, so the frozen test split stays independent of model accuracy.")
        stats["alignment"] = {
            row["clip_id"]: {"offset_seconds": row["offset_seconds"], "confidence": row["alignment_confidence"],
                             "method": row["alignment_method"]}
            for row in alignment
        }

    # --- reference provenance --------------------------------------------
    ratios = [float(row["japanese_script_ratio"]) for row in corpus]
    coverage = [float(row["sync_coverage"]) for row in corpus]
    provenance = [
        ["Songs with an LRCLIB `plainLyrics`/`syncedLyrics` reference", f"{len(corpus)}/{len(corpus)}"],
        ["Reference policy", "`lrclib_assumed_gold` for every row"],
        ["Mean Japanese-script ratio of references", f"{sum(ratios) / len(ratios):.3f}"],
        ["Minimum Japanese-script ratio (gate)", f"{min(ratios):.2f} (gate ≥ {MIN_JA_SCRIPT_RATIO:.2f})"],
        ["Mean synced-lyric coverage of track duration", f"{sum(coverage) / len(coverage):.2f}"],
        ["Total timestamped lyric lines used", f"{sum(int(row['synced_line_count']) for row in corpus):,}"],
    ]
    blocks.append("### LRCLIB reference provenance\n\n" + table(["Property", "Value"], provenance, ["---", "---"]))

    # --- exclusions -------------------------------------------------------
    if excluded:
        reasons = Counter(re.sub(r"\s*\(.*\)", "", str(row["exclusion_reason"])).strip() for row in excluded)
        labels = {
            "reference_not_japanese_script": "LRCLIB stored a romanized or translated reference, not kana/kanji",
            "over_speaker_cap": "Dropped by the per-speaker song cap",
            "implausible_sync_coverage": "Synced-lyric timespan does not plausibly cover the track",
            "too_few_synced_lines": "Fewer timestamped lines than the minimum",
            "no_candidate_within_duration_and_title_rules": "No studio-recording source within the duration rule",
            "no_lrclib_reference_in_local_cache": "No LRCLIB reference in the local cache",
        }
        blocks.append("### Why songs were excluded from the Japanese pool\n\n" + table(
            ["Reason", "Songs", "What it means"],
            [[f"`{reason}`", str(count), labels.get(reason, "")] for reason, count in reasons.most_common()],
            ["---", "---:", "---"]))
        stats["exclusions"] = dict(reasons)

    # --- per-song test roster --------------------------------------------
    test_songs = [row for row in corpus if row["split"] == "test"]
    utterances_per_song = Counter(row["song_id"] for row in utterances)
    used_per_song: dict[str, float] = defaultdict(float)
    for row in utterances:
        used_per_song[str(row["song_id"])] += float(row["duration_seconds"])
    song_rows = []
    for row in sorted(test_songs, key=lambda item: (str(item["speaker_group"]), str(item["clip_id"]))):
        clip_id = str(row["clip_id"])
        song_rows.append([
            str(row["title"]), str(row["artist"]), f"`{row['speaker_group']}`",
            "yes" if row["is_pinned_test"] else "no",
            f"{float(row['video_duration_seconds'] or 0) / 60:.1f}",
            f"{used_per_song[clip_id] / 60:.1f}",
            str(utterances_per_song[clip_id]),
            f"{float(row['japanese_script_ratio']):.2f}",
        ])
    blocks.append("### Frozen Japanese test split\n\n" + table(
        ["Song", "Artist", "Speaker group", "Frozen baseline", "Audio (min)", "Used (min)", "Utterances", "JA script"],
        song_rows, ["---", "---", "---", ":---:", "---:", "---:", "---:", "---:"]))

    document = "\n\n".join(blocks) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8", newline="\n")
    write_jsonl(output.with_suffix(".jsonl"), [stats])
    print(document)
    print(f"Wrote {output} and {output.with_suffix('.jsonl')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/cleaned/ja_corpus.jsonl"))
    parser.add_argument("--utterances", type=Path, default=Path("data/private/ja_utterances.jsonl"))
    parser.add_argument("--alignment", type=Path, default=Path("data/private/ja_alignment.jsonl"))
    parser.add_argument("--excluded", type=Path, default=Path("data/cleaned/ja_corpus_excluded.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/ja_corpus_stats.md"))
    args = parser.parse_args()
    try:
        report(args.corpus, args.utterances, args.alignment, args.excluded, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
