"""Create a reproducible singer-disjoint local benchmark sample from LRCLIB matches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

EXCLUDED_TITLE = re.compile(r"\b(remix|live|karaoke|instrumental|demo|cover|romanized|translation)\b", re.I)
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in value if char.isalnum())


def stable_key(seed: int, artist: str, genius_id: int) -> str:
    return hashlib.sha256(f"{seed}|{artist}|{genius_id}".encode("utf-8")).hexdigest()


def select_rows(matches: list[dict[str, Any]], candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_by_id = {int(row["genius_id"]): row for row in candidates}
    match_by_id = {int(row["genius_id"]): row for row in matches}
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    speaker_splits: dict[str, str] = {}
    seed = int(config.get("seed", 2026))

    def record(match: dict[str, Any], candidate: dict[str, Any], split: str, speaker: str, language: str, method: str, dialect: str = "unassessed", dialect_status: str = "requires_audio_review") -> dict[str, Any]:
        genius_id = int(match["genius_id"])
        return {
            "clip_id": str(match.get("clip_id", f"genius-{genius_id}")),
            "genius_id": genius_id,
            "lrclib_id": match.get("lrclib_id"),
            "genius_url": match.get("genius_url") or candidate.get("url"),
            "lrclib_url": match.get("lrclib_url"),
            "title": match.get("lrclib_title") or candidate.get("title"),
            "artist": match.get("lrclib_artist") or candidate.get("primary_artist"),
            "duration_seconds": match.get("duration_seconds"),
            "artists": candidate.get("artists", []),
            "artist_hint": (candidate.get("matched_artist_hints") or [None])[0],
            "language": language,
            "language_status": "curated_from_song_and_reference_metadata",
            "dialect": dialect,
            "dialect_status": dialect_status,
            "popularity_tier": candidate.get("popularity_tier"),
            "pageviews_at_snapshot": candidate.get("pageviews_at_snapshot"),
            "split": split,
            "speaker_group": speaker,
            "recording_group": f"genius-{genius_id}",
            "reference_status": "lrclib_assumed_gold",
            "selection_method": method,
            "selection_seed": seed,
        }

    for group in config.get("groups", []):
        artist_hint = str(group["artist_hint"])
        split = str(group["split"])
        speaker = str(group["speaker_group"])
        language = str(group["language"])
        aliases = {normalize(str(alias)) for alias in group.get("aliases", [])}
        if split not in SPLIT_ORDER:
            raise ValueError(f"Unsupported split {split!r}")
        if speaker in speaker_splits and speaker_splits[speaker] != split:
            raise ValueError(f"Speaker group {speaker!r} is assigned to multiple splits")
        speaker_splits[speaker] = split

        eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for match in matches:
            genius_id = int(match["genius_id"])
            candidate = candidate_by_id.get(genius_id)
            if not candidate or genius_id in selected_ids:
                continue
            if artist_hint not in candidate.get("matched_artist_hints", []):
                continue
            if (candidate.get("language_hints") or []) != [language]:
                continue
            if normalize(str(match.get("lrclib_artist") or "")) not in aliases:
                continue
            if len(candidate.get("artists", [])) != 1:
                continue
            title = str(match.get("lrclib_title") or candidate.get("title") or "")
            if EXCLUDED_TITLE.search(title) or not match.get("lrclib_id"):
                continue
            eligible.append((match, candidate))

        # Prefer a spread over relative popularity bands, then deterministically
        # fill from remaining eligible songs using the fixed seed.
        tiers = ("popular", "niche", "mid", None)
        bands: dict[str | None, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for tier in tiers:
            band = [(match, candidate) for match, candidate in eligible if candidate.get("popularity_tier") == tier]
            band.sort(key=lambda pair: stable_key(seed, artist_hint, int(pair[0]["genius_id"])))
            bands[tier] = band
        ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
        while any(bands.values()):
            for tier in tiers:
                if bands[tier]:
                    ordered.append(bands[tier].pop(0))
        requested_ids = group.get("genius_ids")
        if requested_ids is not None:
            eligible_by_id = {int(match["genius_id"]): (match, candidate) for match, candidate in eligible}
            missing_ids = [int(genius_id) for genius_id in requested_ids if int(genius_id) not in eligible_by_id]
            if missing_ids:
                raise ValueError(f"{artist_hint}: configured Genius IDs are not eligible under the artist/language/solo filters: {missing_ids}")
            chosen = [eligible_by_id[int(genius_id)] for genius_id in requested_ids]
            count = len(chosen)
        else:
            count = int(group["count"])
            if len(ordered) < count:
                raise ValueError(f"{artist_hint}: requested {count} clips but only {len(ordered)} eligible exact-artist songs are available")
            chosen = ordered[:count]
        for match, candidate in chosen:
            selected.append(record(match, candidate, split, speaker, language, "single-credited-artist; seeded tier-balanced sample", str(group.get("dialect", "unassessed")), str(group.get("dialect_status", "requires_audio_review"))))
            selected_ids.add(int(match["genius_id"]))

    for fixed in config.get("fixed", []):
        genius_id = int(fixed["genius_id"])
        if genius_id in selected_ids:
            raise ValueError(f"Fixed Genius ID {genius_id} is already selected")
        if genius_id not in match_by_id or genius_id not in candidate_by_id:
            raise ValueError(f"Fixed Genius ID {genius_id} has no candidate/LRCLIB match")
        split, speaker, language = str(fixed["split"]), str(fixed["speaker_group"]), str(fixed["language"])
        if split not in SPLIT_ORDER:
            raise ValueError(f"Unsupported fixed split {split!r}")
        if speaker in speaker_splits and speaker_splits[speaker] != split:
            raise ValueError(f"Speaker group {speaker!r} is assigned to multiple splits")
        speaker_splits[speaker] = split
        selected.append(record(match_by_id[genius_id], candidate_by_id[genius_id], split, speaker, language, "fixed-personal-request; vocalist/voicebank assignment", str(fixed.get("dialect", "unassessed")), str(fixed.get("dialect_status", "requires_audio_review"))))
        selected_ids.add(genius_id)

    return sorted(selected, key=lambda row: (SPLIT_ORDER[row["split"]], row["speaker_group"], row["genius_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=Path("data/cleaned/lrclib_matches.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("data/cleaned/genius_candidates_cleaned.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config/benchmark_sample.json"))
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/benchmark_selection.jsonl"))
    args = parser.parse_args()
    try:
        matches = [json.loads(line) for line in args.matches.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidates = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
        config = json.loads(args.config.read_text(encoding="utf-8"))
        rows = select_rows(matches, candidates, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    print(f"Selected {len(rows)} metadata-only benchmark clips to {args.output}; no audio/lyrics were written.")


if __name__ == "__main__":
    main()
