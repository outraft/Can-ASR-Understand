"""Fetch LRCLIB plain/synced lyric references for the cleaned candidate list."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = "https://lrclib.net/api/search"
USER_AGENT = "lyrics-asr-study/0.1 (https://lrclib.net/docs)"
TIMESTAMP = re.compile(r"^\s*\[\d{2}:\d{2}(?:\.\d{1,3})?\]\s*")
VERSION_MARKER = re.compile(r"\b(remix|live|clean|explicit|demo|original|instrumental|acoustic|radio edit|sped up|slowed|extended)\b", re.I)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in value if char.isalnum())


def metadata_variants(value: str, *, title: bool = False) -> list[str]:
    raw = value.strip()
    parenthetical = re.findall(r"[（(]([^）)]+)[）)]", raw)
    if title and any(VERSION_MARKER.search(part) for part in parenthetical):
        return [raw]
    base = re.sub(r"[（(][^）)]*[）)]", " ", raw)
    variants = [base.strip(), raw]
    if not title:
        variants.extend(parenthetical)
    elif any(any(ord(char) > 127 for char in part) for part in parenthetical):
        # Japanese titles often carry a romanization/translation in parentheses.
        variants.extend(parenthetical)
    output: list[str] = []
    for variant in variants:
        if variant and normalize(variant) and normalize(variant) not in {normalize(item) for item in output}:
            output.append(variant)
    return output


def artist_matches(candidate: dict[str, Any], lrclib_artist: str) -> bool:
    lrclib_artists = [
        component.strip()
        for component in re.split(r"\s*(?:,|&|\bfeat\.?\b|\bft\.?\b)\s*", lrclib_artist, flags=re.I)
        if component.strip()
    ]
    candidate_names = [str(candidate.get("primary_artist", ""))]
    candidate_names.extend(str(name) for name in candidate.get("artists", []))
    candidate_names.extend(str(name) for name in candidate.get("matched_artist_hints", []))
    candidate_components = {
        normalize(component)
        for name in candidate_names
        for component in metadata_variants(name)
        if normalize(component)
    }
    return any(normalize(name) in candidate_components for name in lrclib_artists)


def lyric_text(record: dict[str, Any]) -> str:
    plain = record.get("plainLyrics")
    if plain:
        return str(plain).strip()
    synced = record.get("syncedLyrics")
    if not synced:
        return ""
    lines = [TIMESTAMP.sub("", line).strip() for line in str(synced).splitlines()]
    return "\n".join(line for line in lines if line)


def match_records(candidate: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float, bool]:
    titles = metadata_variants(str(candidate.get("title", "")), title=True)
    best: tuple[dict[str, Any], float, bool] | None = None
    for record in records:
        candidate_title = str(record.get("trackName") or record.get("name") or "")
        title_similarity = max(
            (difflib.SequenceMatcher(None, normalize(title), normalize(result_title)).ratio()
             for title in titles for result_title in metadata_variants(candidate_title, title=True)),
            default=0.0,
        )
        artist_ok = artist_matches(candidate, str(record.get("artistName", "")))
        ranked = (record, title_similarity, artist_ok)
        if best is None or (title_similarity, artist_ok) > (best[1], best[2]):
            best = ranked
    if best is None:
        return None, 0.0, False
    return best


def search_lrclib(
    candidate: dict[str, Any],
    user_agent: str = USER_AGENT,
    timeout: int = 30,
    track_name: str | None = None,
    artist_name: str | None = None,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "track_name": track_name or candidate.get("title", ""),
        "artist_name": artist_name or candidate.get("primary_artist", "") or (candidate.get("matched_artist_hints") or [""])[0],
    })
    request = urllib.request.Request(f"{API_URL}?{params}", headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("LRCLIB search response was not a JSON array")
    return payload


def search_with_retry(search_fn, candidate: dict[str, Any], user_agent: str, track_name: str, artist_name: str, attempts: int = 3) -> list[dict[str, Any]]:
    for attempt in range(attempts):
        try:
            return search_fn(candidate, user_agent=user_agent, track_name=track_name, artist_name=artist_name)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 429}:
                raise RuntimeError(f"LRCLIB returned HTTP {exc.code}; stopping without bypassing the response") from exc
            if exc.code < 500 or attempt + 1 == attempts:
                raise RuntimeError(f"LRCLIB returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"LRCLIB network request failed after {attempts} attempts") from exc
        time.sleep(2**attempt)
    raise RuntimeError("LRCLIB request failed")


def read_processed_ids(output_path: Path | None, pending_path: Path | None) -> set[int]:
    processed: set[int] = set()
    for path in (output_path, pending_path):
        if path is None or not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    processed.add(int(json.loads(line)["genius_id"]))
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return processed


def export_match_metadata(lyrics_path: Path, candidates_path: Path, metadata_path: Path) -> int:
    """Write a versionable match index without any lyric text or timestamps."""
    candidates = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidate_by_id = {int(row["genius_id"]): row for row in candidates}
    metadata: list[dict[str, Any]] = []
    if lyrics_path.exists():
        with lyrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                genius_id = int(row["genius_id"])
                candidate = candidate_by_id.get(genius_id, {})
                metadata.append({
                    "clip_id": row.get("clip_id", f"genius-{genius_id}"),
                    "genius_id": genius_id,
                    "lrclib_id": row.get("lrclib_id"),
                    "genius_url": row.get("genius_url") or candidate.get("url"),
                    "lrclib_url": row.get("lrclib_url"),
                    "genius_title": candidate.get("title"),
                    "genius_primary_artist": candidate.get("primary_artist"),
                    "lrclib_title": row.get("lrclib_title"),
                    "lrclib_artist": row.get("lrclib_artist"),
                    "album": row.get("album"),
                    "duration_seconds": row.get("duration_seconds"),
                    "language_hint": row.get("language"),
                    "language_status": row.get("language_status"),
                    "matched_artist_hints": candidate.get("matched_artist_hints", []),
                    "popularity_tier": candidate.get("popularity_tier"),
                    "pageviews_at_snapshot": candidate.get("pageviews_at_snapshot"),
                    "reference_policy": row.get("reference_policy"),
                    "match_status": row.get("match_status"),
                    "title_similarity": row.get("title_similarity"),
                    "plain_lyrics_available": bool(row.get("text")),
                    "synced_lyrics_available": bool(row.get("synced_lyrics")),
                    "retrieved_at_utc": row.get("retrieved_at_utc"),
                })
    metadata.sort(key=lambda row: row["genius_id"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(metadata)


def fetch_candidates(
    input_path: Path,
    output_path: Path,
    pending_path: Path,
    limit: int | None = None,
    delay: float = 0.5,
    user_agent: str = USER_AGENT,
    search_fn=search_lrclib,
    retry_pending: bool = False,
) -> tuple[int, int, int]:
    candidates = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    successful_ids = read_processed_ids(output_path, None)
    pending_ids = read_processed_ids(None, pending_path)
    processed = successful_ids if retry_pending else successful_ids | pending_ids
    if retry_pending:
        candidates = [candidate for candidate in candidates if int(candidate["genius_id"]) in pending_ids and int(candidate["genius_id"]) not in successful_ids]
    matched = pending = attempted = 0
    pending_write_path = pending_path.with_suffix(pending_path.suffix + ".retry") if retry_pending else pending_path
    if retry_pending:
        pending_write_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as out, pending_write_path.open("w" if retry_pending else "a", encoding="utf-8", newline="\n") as unresolved:
        for candidate in candidates:
            genius_id = int(candidate["genius_id"])
            if genius_id in processed:
                continue
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            candidate_title = str(candidate.get("title", ""))
            candidate_artist = str(candidate.get("primary_artist", ""))
            title_variants = metadata_variants(candidate_title, title=True)
            artist_variants = metadata_variants(candidate_artist)
            artist_variants.extend(str(name) for name in candidate.get("matched_artist_hints", []) if name)
            query_pairs: list[tuple[str, str]] = []
            for title_variant in title_variants:
                for artist_variant in artist_variants or [candidate_artist]:
                    pair = (title_variant, artist_variant)
                    if pair not in query_pairs:
                        query_pairs.append(pair)
            records_by_id: dict[Any, dict[str, Any]] = {}
            best: dict[str, Any] | None = None
            title_similarity = 0.0
            artist_ok = False
            for query_index, (track_name, artist_name) in enumerate(query_pairs):
                results = search_with_retry(search_fn, candidate, user_agent, track_name, artist_name)
                for result in results:
                    records_by_id[result.get("id", (result.get("trackName"), result.get("artistName")))] = result
                best, title_similarity, artist_ok = match_records(candidate, list(records_by_id.values()))
                if best and title_similarity >= 0.90 and artist_ok:
                    break
                if query_index + 1 < len(query_pairs):
                    time.sleep(max(0.0, delay))
            records = list(records_by_id.values())
            text = lyric_text(best) if best else ""
            # Do not silently associate a lyric with the wrong song. Auto-match
            # only close title and artist matches; save unresolved IDs/metadata
            # separately so they cannot leak into evaluation references.
            if best and text and title_similarity >= 0.90 and artist_ok:
                out.write(json.dumps({
                    "clip_id": f"genius-{genius_id}",
                    "genius_id": genius_id,
                    "lrclib_id": best.get("id"),
                    "genius_url": candidate.get("url"),
                    "lrclib_title": best.get("trackName") or best.get("name"),
                    "lrclib_artist": best.get("artistName"),
                    "album": best.get("albumName"),
                    "duration_seconds": best.get("duration"),
                    "lrclib_url": f"https://lrclib.net/api/get/{best['id']}" if best.get("id") is not None else None,
                    "language": (candidate.get("language_hints") or ["und"])[0],
                    "language_status": "query_hint_needs_review",
                    "reference_policy": "lrclib_assumed_gold",
                    "match_status": "title_artist_match",
                    "title_similarity": round(title_similarity, 4),
                    "text": text,
                    "synced_lyrics": best.get("syncedLyrics"),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                out.flush()
                matched += 1
            else:
                unresolved.write(json.dumps({
                    "genius_id": genius_id,
                    "genius_title": candidate.get("title"),
                    "genius_artist": candidate.get("primary_artist"),
                    "candidate_result_count": len(records),
                    "best_lrclib_id": best.get("id") if best else None,
                    "best_lrclib_title": (best.get("trackName") or best.get("name")) if best else None,
                    "best_lrclib_artist": best.get("artistName") if best else None,
                    "title_similarity": round(title_similarity, 4),
                    "artist_match": artist_ok,
                    "status": "no_match_or_needs_review",
                }, ensure_ascii=False) + "\n")
                unresolved.flush()
                pending += 1
            processed.add(genius_id)
            time.sleep(max(0.0, delay))
    if retry_pending and pending_write_path.exists():
        if limit is None or attempted >= len(candidates):
            pending_write_path.replace(pending_path)
        else:
            pending_write_path.unlink()
    return matched, pending, attempted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/cleaned/genius_candidates_cleaned.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/references/lrclib_lyrics.jsonl"))
    parser.add_argument("--pending", type=Path, default=Path("data/cleaned/lrclib_pending.jsonl"))
    parser.add_argument("--metadata-output", type=Path, default=Path("data/cleaned/lrclib_matches.jsonl"))
    parser.add_argument("--limit", type=int, help="Maximum new candidate IDs to query")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between sequential API requests")
    parser.add_argument("--retry-pending", action="store_true", help="Retry only IDs in the existing pending file and refresh that file")
    parser.add_argument("--user-agent", default=USER_AGENT, help="Identifying User-Agent sent to LRCLIB")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        sys.exit("--limit must be positive")
    try:
        matched, pending, attempted = fetch_candidates(args.input, args.output, args.pending, args.limit, args.delay, args.user_agent, retry_pending=args.retry_pending)
        metadata_count = export_match_metadata(args.output, args.input, args.metadata_output)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    print(f"Queried {attempted} candidates: {matched} matched, {pending} unresolved.")
    print(f"Lyrics are local at {args.output} (gitignored); unresolved metadata is at {args.pending}.")
    print(f"Wrote {metadata_count} lyrics-free match records to {args.metadata_output}.")


if __name__ == "__main__":
    main()
