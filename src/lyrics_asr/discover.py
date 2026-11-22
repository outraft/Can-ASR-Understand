"""Discover candidate songs with the Genius API; this does not fetch lyrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://api.genius.com/search"


def load_env_file(path: Path = Path(".env")) -> None:
    """Load Genius credentials from a simple local .env without overriding the process env."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in {"GENIUS_ACCESS_TOKEN", "GENIUS_ACCESS"}:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def request_page(query: str, page: int, token: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "page": page})
    request = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "lyrics-asr-study/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("meta", {}).get("status") != 200:
        raise RuntimeError(f"Genius API returned an unexpected response for {query!r}")
    return payload


def normalize_song(song: dict[str, Any], query: dict[str, str], fetched_at: str) -> dict[str, Any]:
    primary = song.get("primary_artist") or {}
    featured = song.get("featured_artists") or []
    names = [primary.get("name", ""), *(artist.get("name", "") for artist in featured)]
    stats = song.get("stats") or {}
    return {
        "genius_id": song.get("id"),
        "title": song.get("title", ""),
        "url": song.get("url", ""),
        "primary_artist": primary.get("name", ""),
        "artists": [name for name in names if name],
        "release_date": song.get("release_date_for_display"),
        "pageviews": stats.get("pageviews"),
        "language_hints": [query.get("language", "und")],
        "artist_hints": [query.get("artist_hint", "")],
        "discovered_by_queries": [query.get("query", "")],
        "fetched_at_utc": fetched_at,
    }


def discover(config_path: Path, output_path: Path, token: str, merge_existing: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pages = max(1, int(config.get("pages_per_query", 1)))
    delay = 1.0 / max(0.1, float(config.get("requests_per_second", 1.0)))
    queries = config.get("queries", [])
    if not queries:
        raise ValueError("No queries configured")

    results: dict[int, dict[str, Any]] = {}
    if merge_existing and output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("genius_id") is not None:
                    results[int(row["genius_id"])] = row
    for query in queries:
        for page in range(1, pages + 1):
            try:
                payload = request_page(query["query"], page, token)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Genius API HTTP {exc.code}: {detail}") from exc
            fetched_at = datetime.now(timezone.utc).isoformat()
            hits = payload.get("response", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                song = hit.get("result") or {}
                if song.get("id") is None:
                    continue
                row = normalize_song(song, query, fetched_at)
                existing = results.get(row["genius_id"])
                if existing:
                    existing["language_hints"] = sorted(set(existing["language_hints"]) | set(row["language_hints"]))
                    existing["artist_hints"] = sorted(set(existing["artist_hints"]) | set(row["artist_hints"]))
                    existing["discovered_by_queries"] = sorted(set(existing["discovered_by_queries"]) | set(row["discovered_by_queries"]))
                    existing["last_seen_at_utc"] = fetched_at
                else:
                    results[row["genius_id"]] = row
            time.sleep(delay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for song_id in sorted(results):
            handle.write(json.dumps(results[song_id], ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} candidate songs to {output_path}")
    print("Candidates require manual verification of artist, language, recording, and popularity tier.")
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/searches.json"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/genius_candidates.jsonl"))
    parser.add_argument("--merge-existing", action="store_true", help="Merge new results into an existing snapshot by Genius ID")
    args = parser.parse_args()
    load_env_file(Path(".env"))
    token = os.environ.get("GENIUS_ACCESS_TOKEN") or os.environ.get("GENIUS_ACCESS")
    if not token:
        sys.exit("Set GENIUS_ACCESS_TOKEN in the environment or project .env; do not commit API tokens.")
    try:
        discover(args.config, args.output, token, args.merge_existing)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
