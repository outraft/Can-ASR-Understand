"""Create deterministic speaker/recording-disjoint splits from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def assign_splits(
    rows: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float],
    pin_test_speakers: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Map every row's group component to a split name.

    Groups are speaker/recording components, so all clips by one vocalist and all
    clips from one recording land in the same split. ``pin_test_speakers`` forces
    the components containing those speakers into ``test`` before the seeded
    assignment balances the remaining components over the other splits; this keeps
    a previously frozen test set intact while the rest stays reproducible.
    """
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Train/validation/test ratios must be positive and sum to 1")

    groups = UnionFind()
    for row in rows:
        speaker, recording = str(row.get("speaker_group", "")).strip(), str(row.get("recording_group", "")).strip()
        if not speaker or not recording:
            raise ValueError(f"{row.get('clip_id', '<unknown>')}: speaker_group and recording_group are required")
        groups.union("speaker:" + speaker, "recording:" + recording)

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        components[groups.find("speaker:" + str(row["speaker_group"]).strip())].append(row)

    pinned = {groups.find("speaker:" + speaker.strip()) for speaker in pin_test_speakers if ("speaker:" + speaker.strip()) in groups.parent}
    missing = {s for s in pin_test_speakers if ("speaker:" + s.strip()) not in groups.parent}
    if missing:
        raise ValueError(f"Pinned test speakers absent from the manifest: {sorted(missing)}")

    names, total = ["train", "validation", "test"], len(rows)
    targets = dict(zip(names, ratios))
    counts = {name: 0 for name in names}
    assignment: dict[str, str] = {}
    for component in pinned:
        assignment[component] = "test"
        counts["test"] += len(components[component])

    # Pinned components already fill test, so balance the rest over train/validation
    # in proportion to their requested shares.
    free_names = [name for name in names if name != "test"] if pinned else names
    remaining = [item for item in components.items() if item[0] not in assignment]
    random.Random(seed).shuffle(remaining)
    remaining.sort(key=lambda item: len(item[1]), reverse=True)
    for component, clips in remaining:
        split_name = min(
            free_names,
            key=lambda name: (counts[name] + len(clips) - targets[name] * total) ** 2
            - (counts[name] - targets[name] * total) ** 2,
        )
        assignment[component] = split_name
        counts[split_name] += len(clips)

    return {row["clip_id"]: assignment[groups.find("speaker:" + str(row["speaker_group"]).strip())] for row in rows}


def make_splits(
    input_path: Path,
    output_path: Path,
    seed: int,
    ratios: tuple[float, float, float],
    pin_test_speakers: frozenset[str] = frozenset(),
) -> None:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Manifest is empty")

    by_clip = assign_splits(rows, seed, ratios, pin_test_speakers)
    for row in rows:
        row["split"] = by_clip[row["clip_id"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def reference_text(row: dict[str, Any]) -> str:
        if row.get("reference_text") is not None:
            return str(row["reference_text"])
        reference_path = row.get("reference_path")
        if reference_path:
            path = Path(reference_path)
            if not path.is_absolute():
                path = input_path.parent / path
            return path.read_text(encoding="utf-8")
        return ""

    def sentences(split_name: str) -> set[str]:
        """Normalized reference lines, which is the unit an overlap audit needs."""
        found: set[str] = set()
        for row in rows:
            if row["split"] != split_name:
                continue
            for line in reference_text(row).splitlines():
                collapsed = " ".join(line.casefold().split())
                if collapsed:
                    found.add(collapsed)
        return found

    train_sentences, test_sentences = sentences("train"), sentences("test")
    print(f"Wrote {len(rows)} clips to {output_path} (seed={seed})")
    for name in ["train", "validation", "test"]:
        subset = [row for row in rows if row["split"] == name]
        speakers = {row["speaker_group"] for row in subset}
        hours = sum(float(row.get("duration_seconds", 0)) for row in subset) / 3600
        print(f"{name}: {len(speakers)} speakers, {len(subset)} utterances, {hours:.2f} hours")
    shared = train_sentences & test_sentences
    print(f"Distinct test reference sentences: {len(test_sentences)}")
    print(f"Exact test sentences also in train: {len(shared)}"
          + (f" ({100 * len(shared) / len(test_sentences):.1f}%)" if test_sentences else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument(
        "--pin-test-speakers",
        nargs="*",
        default=[],
        help="Speaker groups forced into test so an already-frozen test set is preserved",
    )
    args = parser.parse_args()
    try:
        make_splits(args.input, args.output, args.seed, tuple(args.ratios), frozenset(args.pin_test_speakers))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
