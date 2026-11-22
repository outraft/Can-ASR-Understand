"""Filter Genius search noise into a reviewable, metadata-only candidate list."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any


def normalize_name(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def artist_matches(artist_names: list[str], aliases: list[str]) -> bool:
    normalized_aliases = {normalize_name(alias) for alias in aliases if normalize_name(alias)}
    for artist in artist_names:
        # Genius sometimes combines the Japanese and romanized artist names,
        # e.g. "稲葉曇 (inabakumori)". Compare each display-name component,
        # not substrings (which would falsely match "Future" in "Odd Future").
        base_name = re.sub(r"[（(][^）)]*[）)]", " ", artist)
        components = [artist, base_name, *re.findall(r"[（(]([^）)]+)[）)]", artist)]
        if any(normalize_name(component) in normalized_aliases for component in components):
            return True
    return False


def is_translation_or_romanization(row: dict[str, Any]) -> bool:
    primary = normalize_name(str(row.get("primary_artist", "")))
    return any(marker in primary for marker in ("geniusromanizations", "geniusenglishtranslations", "geniustranslations"))


def clean_candidates(raw_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    query_config = {str(item["query"]): item for item in config.get("queries", [])}
    cleaned: list[dict[str, Any]] = []
    for row in raw_rows:
        if is_translation_or_romanization(row):
            continue
        artist_names = [str(name) for name in row.get("artists", [])]
        matched_queries: list[str] = []
        matched_artists: set[str] = set()
        matched_languages: set[str] = set()
        for query in row.get("discovered_by_queries", []):
            settings = query_config.get(str(query))
            if not settings:
                continue
            aliases = settings.get("artist_aliases") or [settings.get("artist_hint", query)]
            if artist_matches(artist_names, [str(alias) for alias in aliases]):
                matched_queries.append(str(query))
                matched_artists.add(str(settings.get("artist_hint", query)))
                matched_languages.add(str(settings.get("language", "und")))
        if not matched_queries:
            continue
        cleaned.append({
            "genius_id": row.get("genius_id"),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "primary_artist": row.get("primary_artist", ""),
            "artists": artist_names,
            "release_date": row.get("release_date"),
            "pageviews_at_snapshot": row.get("pageviews"),
            "language_hints": sorted(matched_languages),
            "matched_artist_hints": sorted(matched_artists),
            "matched_queries": sorted(set(matched_queries)),
            "fetched_at_utc": row.get("fetched_at_utc"),
            "metadata_review_status": "needs_manual_review",
            "reference_policy": "lrclib_assumed_gold",
            "popularity_tier": None,
            "popularity_tier_method": None,
        })

    # Relative terciles are a reproducible starting point for sampling, not an
    # objective or permanent popularity label. Ambiguous-language/missing-view
    # rows remain unranked for later curation.
    by_language: dict[str, list[int]] = {}
    for index, row in enumerate(cleaned):
        if len(row["language_hints"]) == 1 and row["pageviews_at_snapshot"] is not None:
            by_language.setdefault(row["language_hints"][0], []).append(index)
    for language, indices in by_language.items():
        values = [int(cleaned[index]["pageviews_at_snapshot"]) for index in indices]
        if len(values) < 3:
            continue
        low_cut, high_cut = statistics.quantiles(values, n=3, method="inclusive")
        for index, value in zip(indices, values):
            tier = "niche" if value <= low_cut else "mid" if value <= high_cut else "popular"
            cleaned[index]["popularity_tier"] = tier
            cleaned[index]["popularity_tier_method"] = f"relative pageview tercile within {language} candidate pool at snapshot"
    for row in cleaned:
        if row["popularity_tier_method"] is None:
            if row["pageviews_at_snapshot"] is None:
                row["popularity_tier_method"] = "unavailable pageview count; tier pending curation"
            elif len(row["language_hints"]) != 1:
                row["popularity_tier_method"] = "ambiguous language hints; tier pending curation"
            else:
                row["popularity_tier_method"] = "fewer than three ranked candidates for language; tier pending curation"
    return sorted(cleaned, key=lambda row: (row["genius_id"] is None, row["genius_id"] or 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/genius_candidates.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("config/searches.json"))
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/genius_candidates_cleaned.jsonl"))
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        config = json.loads(args.config.read_text(encoding="utf-8"))
        cleaned = clean_candidates(rows, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in cleaned:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    print(f"Saved {len(cleaned)} metadata candidates to {args.output}; review candidate metadata, not the assumed LRCLIB gold text.")


if __name__ == "__main__":
    main()
