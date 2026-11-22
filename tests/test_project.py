from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lyrics_asr.clean_candidates import artist_matches, clean_candidates
from lyrics_asr.discover import discover, load_env_file, normalize_song
from lyrics_asr.evaluate import edit_counts, evaluate, normalize, score
from lyrics_asr.fetch_lrclib import artist_matches as lrclib_artist_matches
from lyrics_asr.fetch_lrclib import export_match_metadata, fetch_candidates, lyric_text, match_records, search_with_retry
from lyrics_asr.prepare_benchmark import prepare_benchmark
from lyrics_asr.split import make_splits
from lyrics_asr.validate import validate_rows
from lyrics_asr.youtube_search import candidate_searches
from lyrics_asr.download_audio import download_sources, validate_youtube_url
from lyrics_asr.transcribe import transcribe_manifest
from lyrics_asr.select_benchmark import select_rows
from lyrics_asr.split import assign_splits
from lyrics_asr.purge_english import is_english_only, language_signal
from lyrics_asr.ja_corpus import TITLE_BLOCKLIST, japanese_script_ratio, parse_synced, pick_source
from lyrics_asr.ja_align import bag_f1, build_windows, estimate_offset
from lyrics_asr.ja_evaluate import characters, evaluate_japanese, score_pair
from lyrics_asr.ja_segment import probe_duration, segment

ROOT = Path(__file__).parent
FIXTURE = ROOT / "fixtures" / "manifest.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DiscoveryTests(unittest.TestCase):
    def test_loads_dotenv_access_alias_without_overriding_process_env(self):
        with tempfile.TemporaryDirectory() as temp:
            env_file = Path(temp) / ".env"
            env_file.write_text('GENIUS_ACCESS = "dotenv-cat"\n', encoding="utf-8")
            with patch.dict(os.environ, {"GENIUS_ACCESS_TOKEN": "process-cat"}, clear=True):
                load_env_file(env_file)
                self.assertEqual(os.environ["GENIUS_ACCESS"], "dotenv-cat")
                self.assertEqual(os.environ["GENIUS_ACCESS_TOKEN"], "process-cat")

    def test_normalizes_metadata_without_lyrics(self):
        row = normalize_song(
            {"id": 12, "title": "Song", "url": "https://genius.com/song", "primary_artist": {"name": "Artist"}, "featured_artists": [{"name": "Guest"}], "stats": {"pageviews": 99}},
            {"query": "Artist", "language": "en", "artist_hint": "Artist"},
            "2026-11-22T00:00:00+00:00",
        )
        self.assertEqual(row["genius_id"], 12)
        self.assertEqual(row["artists"], ["Artist", "Guest"])
        self.assertEqual(row["pageviews"], 99)
        self.assertNotIn("lyrics", row)

    def test_discovery_deduplicates_hits_and_preserves_query_hints(self):
        payload = {"meta": {"status": 200}, "response": {"hits": [{"result": {"id": 4, "title": "Song", "url": "https://genius.com/song", "primary_artist": {"name": "Artist"}}}]}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = folder / "config.json"
            output = folder / "candidates.jsonl"
            config.write_text(json.dumps({"pages_per_query": 1, "requests_per_second": 100, "queries": [
                {"query": "Artist", "language": "en", "artist_hint": "Artist"},
                {"query": "アーティスト", "language": "ja", "artist_hint": "Artist"},
            ]}), encoding="utf-8")
            output.write_text(json.dumps({
                "genius_id": 4, "title": "Song", "url": "https://genius.com/song",
                "primary_artist": "Artist", "artists": ["Artist"], "language_hints": ["en"],
                "artist_hints": ["Artist"], "discovered_by_queries": ["Artist"],
                "fetched_at_utc": "2026-01-01T00:00:00+00:00",
            }) + "\n", encoding="utf-8")
            with patch("lyrics_asr.discover.request_page", return_value=payload), patch("lyrics_asr.discover.time.sleep"):
                self.assertEqual(discover(config, output, "token", merge_existing=True), 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["language_hints"], ["en", "ja"])
            self.assertEqual(len(row["discovered_by_queries"]), 2)


class CandidateCleaningTests(unittest.TestCase):
    def test_artist_matching_avoids_substring_false_positive_and_handles_aliases(self):
        self.assertFalse(artist_matches(["Odd Future"], ["Future"]))
        self.assertTrue(artist_matches(["稲葉曇 (inabakumori)"], ["Inabakumori", "稲葉曇"]))
        self.assertTrue(artist_matches(["Eve (JPN)"], ["Eve"]))

    def test_clean_candidates_filters_search_noise_and_translation_entries(self):
        config = {"queries": [
            {"query": "Future", "language": "en", "artist_hint": "Future", "artist_aliases": ["Future"]},
            {"query": "Inabakumori", "language": "ja", "artist_hint": "Inabakumori", "artist_aliases": ["Inabakumori", "稲葉曇"]},
        ]}
        rows = [
            {"genius_id": 1, "title": "Song", "primary_artist": "Future", "artists": ["Future"], "discovered_by_queries": ["Future"], "pageviews": 42},
            {"genius_id": 2, "title": "Song", "primary_artist": "Odd Future", "artists": ["Odd Future"], "discovered_by_queries": ["Future"], "pageviews": 13},
            {"genius_id": 3, "title": "曲", "primary_artist": "稲葉曇 (inabakumori)", "artists": ["稲葉曇 (inabakumori)"], "discovered_by_queries": ["Inabakumori"], "pageviews": 7},
            {"genius_id": 4, "title": "Translation", "primary_artist": "Genius English Translations", "artists": ["Genius English Translations"], "discovered_by_queries": ["Inabakumori"], "pageviews": 1},
        ]
        cleaned = clean_candidates(rows, config)
        self.assertEqual([row["genius_id"] for row in cleaned], [1, 3])
        self.assertTrue(all(row["metadata_review_status"] == "needs_manual_review" for row in cleaned))
        self.assertTrue(all(row["reference_policy"] == "lrclib_assumed_gold" for row in cleaned))

    def test_assigns_reproducible_relative_pageview_terciles(self):
        config = {"queries": [{"query": "Future", "language": "en", "artist_hint": "Future", "artist_aliases": ["Future"]}]}
        rows = [
            {"genius_id": index, "primary_artist": "Future", "artists": ["Future"], "discovered_by_queries": ["Future"], "pageviews": views}
            for index, views in enumerate((10, 20, 30), 1)
        ]
        cleaned = clean_candidates(rows, config)
        self.assertEqual([row["popularity_tier"] for row in cleaned], ["niche", "mid", "popular"])


class BenchmarkSelectionTests(unittest.TestCase):
    def test_selection_is_speaker_disjoint_reproducible_and_lyrics_free(self):
        candidates=[]
        matches=[]
        for genius_id,artist,tier in ((1,'Alpha','popular'),(2,'Alpha','niche'),(3,'Beta','mid'),(4,'Personal',None)):
            candidates.append({'genius_id':genius_id,'matched_artist_hints':[artist],'language_hints':['en'],'artists':[artist],'title':f'Song {genius_id}','primary_artist':artist,'popularity_tier':tier})
            matches.append({'genius_id':genius_id,'clip_id':f'g-{genius_id}','lrclib_id':genius_id+10,'lrclib_title':f'Song {genius_id}','lrclib_artist':artist})
        config={'seed':2026,'groups':[
            {'artist_hint':'Alpha','speaker_group':'alpha','language':'en','split':'train','count':2,'aliases':['Alpha']},
            {'artist_hint':'Beta','speaker_group':'beta','language':'en','split':'test','count':1,'aliases':['Beta']},
        ],'fixed':[{'genius_id':4,'speaker_group':'personal-voice','language':'en','split':'train'}]}
        first=select_rows(matches,candidates,config)
        self.assertEqual(first,select_rows(matches,candidates,config))
        self.assertEqual(Counter(row['split'] for row in first),{'train':3,'test':1})
        self.assertFalse(any('text' in row for row in first))
        self.assertTrue(all(row['dialect']=='unassessed' for row in first))
        self.assertTrue(all(row['dialect_status']=='requires_audio_review' for row in first))


class BenchmarkAssemblyTests(unittest.TestCase):
    def test_prepare_local_benchmark_joins_audio_and_reference_without_copying_lyrics_to_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            audio=folder/'audio.wav'; audio.write_bytes(b'synthetic wav')
            selection=folder/'selection.jsonl'; lyrics=folder/'lyrics.jsonl'; audio_index=folder/'audio.jsonl'
            manifest=folder/'manifest.jsonl'; references=folder/'refs.jsonl'; test_refs=folder/'test_refs.jsonl'
            selection.write_text(json.dumps({
                'clip_id':'clip-1','genius_id':1,'lrclib_id':11,'genius_url':'https://genius.com/song',
                'lrclib_url':'https://lrclib.net/api/get/11','title':'Song','artists':['Singer'],
                'language':'en','popularity_tier':'popular','pageviews_at_snapshot':100,
                'dialect':'unassessed','dialect_status':'requires_audio_review',
                'speaker_group':'singer-1','recording_group':'recording-1','duration_seconds':12,
                'split':'test'
            })+'\n',encoding='utf-8')
            lyrics.write_text(json.dumps({'clip_id':'clip-1','genius_id':1,'lrclib_id':11,'language':'en','text':'gold words'})+'\n',encoding='utf-8')
            audio_index.write_text(json.dumps({'clip_id':'clip-1','genius_id':1,'lrclib_id':11,'video_id':'v1','youtube_url':'https://youtu.be/v1','audio_path':str(audio),'duration_seconds':12,'split':'test'})+'\n',encoding='utf-8')
            self.assertEqual(prepare_benchmark(selection,lyrics,audio_index,manifest,references,test_refs),(1,1))
            row=json.loads(manifest.read_text(encoding='utf-8'))
            self.assertNotIn('text',row)
            self.assertEqual(row['reference_status'],'lrclib_assumed_gold')
            self.assertEqual(row['dialect'],'unassessed')
            self.assertEqual(validate_rows([row],require_split=True),[])
            self.assertEqual(json.loads(test_refs.read_text(encoding='utf-8'))['text'],'gold words')


class LrclibFetchTests(unittest.TestCase):
    def test_transient_network_failure_retries_boundedly(self):
        calls = []

        def flaky_search(candidate, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.URLError("connection reset")
            return []

        with patch("lyrics_asr.fetch_lrclib.time.sleep"):
            result = search_with_retry(flaky_search, {}, "test-agent", "title", "artist", attempts=2)
        self.assertEqual(result, [])
        self.assertEqual(len(calls), 2)

    def test_uses_plain_lyrics_or_strips_synced_timestamps(self):
        self.assertEqual(lyric_text({"plainLyrics": "plain text", "syncedLyrics": "[00:01.00] synced"}), "plain text")
        self.assertEqual(lyric_text({"syncedLyrics": "[00:01.00] first\n[00:02.20] second"}), "first\nsecond")

    def test_match_uses_title_and_artist(self):
        candidate = {"title": "Song", "primary_artist": "Artist", "artists": ["Artist"]}
        records = [{"id": 10, "trackName": "Song", "artistName": "Artist"}]
        record, similarity, artist_ok = match_records(candidate, records)
        self.assertEqual(record["id"], 10)
        self.assertEqual(similarity, 1.0)
        self.assertTrue(artist_ok)

    def test_matches_japanese_title_variants_and_featured_artist(self):
        japanese = {"title": "ロストアンブレラ (Lost Umbrella)", "primary_artist": "稲葉曇 (inabakumori)", "artists": ["稲葉曇 (inabakumori)"]}
        record, similarity, artist_ok = match_records(japanese, [{"id": 11, "trackName": "ロストアンブレラ", "artistName": "稲葉曇"}])
        self.assertEqual(record["id"], 11)
        self.assertEqual(similarity, 1.0)
        self.assertTrue(artist_ok)
        self.assertTrue(lrclib_artist_matches({"primary_artist": "Future", "artists": ["Future"]}, "Future, Lil Wayne"))

    def test_fetch_resumes_and_separates_unresolved_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, output, pending = folder / "candidates.jsonl", folder / "lyrics.jsonl", folder / "pending.jsonl"
            candidates = [
                {"genius_id": 1, "title": "Song", "primary_artist": "Artist", "artists": ["Artist"], "matched_artist_hints": ["Artist"], "language_hints": ["en"], "url": "genius-url"},
                {"genius_id": 2, "title": "Other song", "primary_artist": "Artist", "artists": ["Artist"], "matched_artist_hints": ["Artist"], "language_hints": ["en"]},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8")
            def fake_search(candidate, **kwargs):
                if candidate["genius_id"] == 1:
                    return [{"id": 10, "trackName": "Song", "artistName": "Artist", "plainLyrics": "gold text"}]
                return [{"id": 20, "trackName": "Different", "artistName": "Artist", "plainLyrics": "not this song"}]

            result = fetch_candidates(source, output, pending, delay=0, search_fn=fake_search)
            self.assertEqual(result, (1, 1, 2))
            fetched_row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(fetched_row["reference_policy"], "lrclib_assumed_gold")
            self.assertEqual(fetched_row["text"], "gold text")
            self.assertEqual(len(pending.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(fetch_candidates(source, output, pending, delay=0, search_fn=fake_search), (0, 0, 0))

    def test_retry_pending_only_retries_unresolved_and_refreshes_pending_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, output, pending = folder / "candidates.jsonl", folder / "lyrics.jsonl", folder / "pending.jsonl"
            candidates = [
                {"genius_id": 1, "title": "Already", "primary_artist": "Artist", "artists": ["Artist"], "language_hints": ["en"]},
                {"genius_id": 2, "title": "Retry", "primary_artist": "Artist", "artists": ["Artist"], "language_hints": ["en"]},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8")
            output.write_text(json.dumps({"genius_id": 1, "text": "prior"}) + "\n", encoding="utf-8")
            pending.write_text(json.dumps({"genius_id": 2}) + "\n", encoding="utf-8")
            calls = []

            def fake_search(candidate, **kwargs):
                calls.append(candidate["genius_id"])
                return [{"id": 22, "trackName": "Retry", "artistName": "Artist", "plainLyrics": "text"}]

            result = fetch_candidates(source, output, pending, delay=0, search_fn=fake_search, retry_pending=True)
            self.assertEqual(result, (1, 0, 1))
            self.assertEqual(calls, [2])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(pending.read_text(encoding="utf-8"), "")


class YouTubeMediaTests(unittest.TestCase):
    def test_downloaded_media_and_private_archive_are_gitignored(self):
        ignore = (Path(__file__).parent.parent / ".gitignore").read_text(encoding="utf-8")
        for path in ("data/audio/", "data/references/", "data/private/"):
            self.assertIn(path, ignore)

    def test_search_candidates_preserves_song_provenance(self):
        matches = [{"genius_id": 4, "lrclib_id": 44, "title": "Song", "artist": "Singer", "speaker_group": "singer", "split": "test"}]
        def fake_search(query, count):
            self.assertEqual(query, "Song Singer")
            self.assertEqual(count, 2)
            return [{"video_id": "video-1", "youtube_url": "https://www.youtube.com/watch?v=video-1"}]
        with patch("lyrics_asr.youtube_search.time.sleep"):
            rows = candidate_searches(matches, fake_search, results_per_song=2, delay=0)
        self.assertEqual(rows[0]["genius_id"], 4)
        self.assertEqual(rows[0]["lrclib_id"], 44)
        self.assertEqual(rows[0]["search_status"], "needs_manual_video_review")
        self.assertEqual(rows[0]["speaker_group"], "singer")
        self.assertEqual(rows[0]["split"], "test")

    def test_youtube_url_allowlist(self):
        self.assertTrue(validate_youtube_url("https://www.youtube.com/watch?v=abc"))
        self.assertTrue(validate_youtube_url("https://youtu.be/abc"))
        self.assertFalse(validate_youtube_url("https://notyoutube.com/watch?v=abc"))
        self.assertFalse(validate_youtube_url("http://youtube.com/watch?v=abc"))

    def test_download_records_local_audio_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            audio_dir, archive, manifest = folder / "audio", folder / "private" / "archive.txt", folder / "audio" / "manifest.jsonl"

            class FakeYDL:
                def __init__(self, options):
                    self.options = options

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def extract_info(self, url, download):
                    (audio_dir / "video-1.wav").write_bytes(b"synthetic audio")
                    return {"id": "video-1", "duration": 2.0}

            fake_module = SimpleNamespace(version=SimpleNamespace(__version__="test"))
            sources = [{"clip_id": "clip-1", "genius_id": 1, "youtube_url": "https://youtu.be/video-1"}]
            with patch.dict(sys.modules, {"yt_dlp": fake_module}):
                self.assertEqual(download_sources(sources, audio_dir, archive, manifest, lambda options: FakeYDL(options), delay=0), (1, 0))
                self.assertEqual(download_sources(sources, audio_dir, archive, manifest, lambda options: FakeYDL(options), delay=0), (0, 0))
            row = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(row["video_id"], "video-1")
            self.assertEqual(len(row["audio_sha256"]), 64)


class TranscriptionTests(unittest.TestCase):
    def test_transcriber_writes_prediction_and_resumes_by_clip_id(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            audio=folder/'clip.wav'
            audio.write_bytes(b'fake audio')
            audio_manifest=folder/'audio.jsonl'
            references=folder/'references.jsonl'
            output=folder/'predictions.jsonl'
            audio_manifest.write_text(json.dumps({"clip_id":"clip-1","audio_path":str(audio),"split":"test","audio_sha256":"digest"})+'\n',encoding='utf-8')
            references.write_text(json.dumps({"clip_id":"clip-1","language":"en","text":"reference"})+'\n',encoding='utf-8')

            class FakeModel:
                def __init__(self,*args,**kwargs): pass
                def transcribe(self,*args,**kwargs):
                    return iter([SimpleNamespace(text=" hypothesis ")]), SimpleNamespace(language="en")

            self.assertEqual(transcribe_manifest(audio_manifest,references,output,model_factory=FakeModel),1)
            prediction=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(prediction['text'],'hypothesis')
            self.assertEqual(prediction['language'],'en')
            self.assertEqual(prediction['dialect'],'unassessed')
            self.assertEqual(transcribe_manifest(audio_manifest,references,output,model_factory=FakeModel),0)

    def test_match_metadata_export_excludes_lyrics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            lyrics, candidates, metadata = folder / "lyrics.jsonl", folder / "candidates.jsonl", folder / "metadata.jsonl"
            lyrics.write_text(json.dumps({
                "genius_id": 9, "clip_id": "genius-9", "lrclib_id": 90,
                "text": "private lyric text", "synced_lyrics": "[00:01] private lyric text",
                "lrclib_title": "Title", "lrclib_artist": "Artist", "reference_policy": "lrclib_assumed_gold",
            }) + "\n", encoding="utf-8")
            candidates.write_text(json.dumps({"genius_id": 9, "title": "Title", "matched_artist_hints": ["Artist"]}) + "\n", encoding="utf-8")
            self.assertEqual(export_match_metadata(lyrics, candidates, metadata), 1)
            row = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertTrue(row["plain_lyrics_available"])
            self.assertTrue(row["synced_lyrics_available"])
            self.assertNotIn("text", row)
            self.assertNotIn("synced_lyrics", row)


class MetricTests(unittest.TestCase):
    def test_normalization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(normalize("Hello, WORLD!"), "hello world")

    def test_edit_counts_include_substitution_deletion_and_insertion(self):
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "x", "c", "d"]), (1, 0, 1, 2))

    def test_english_wer_and_f1(self):
        result = score("we sing tonight", "we sang tonight", "en")
        self.assertEqual(result["wer"], 1 / 3)
        self.assertEqual(result["token_f1"], 2 / 3)
        self.assertEqual(result["substitutions"], 1)

    def test_japanese_uses_character_tokens_and_omits_wer(self):
        result = score("あおいそら", "あおいはな", "ja")
        self.assertIsNone(result["wer"])
        self.assertEqual(result["token_error_rate"], 2 / 5)
        self.assertEqual(result["cer"], 2 / 5)

    def test_empty_reference(self):
        result = score("", "extra", "en")
        self.assertEqual(result["wer"], 1.0)
        self.assertEqual(result["token_f1"], 0.0)

    def test_evaluator_writes_comparable_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            refs = folder / "refs.jsonl"
            pred = folder / "pred.jsonl"
            refs.write_text('{"clip_id":"a","language":"en","text":"hello world"}\n', encoding="utf-8")
            pred.write_text('{"clip_id":"a","text":"hello word"}\n', encoding="utf-8")
            evaluate(refs, [f"baseline={pred}"], folder / "metrics.csv")
            summary = (folder / "metrics_summary.csv").read_text(encoding="utf-8")
            self.assertIn("baseline", summary)
            self.assertIn("0.5", summary)

    def test_evaluator_rejects_different_clip_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            refs, pred = folder / "refs.jsonl", folder / "pred.jsonl"
            refs.write_text('{"clip_id":"a","language":"en","text":"hello"}\n', encoding="utf-8")
            pred.write_text('{"clip_id":"b","text":"hello"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clip IDs"):
                evaluate(refs, [f"asr={pred}"], folder / "out.csv")

    def test_evaluator_reports_dialect_groups_when_annotated(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            refs, pred = folder / "refs.jsonl", folder / "pred.jsonl"
            refs.write_text(
                '{"clip_id":"a","language":"en","dialect":"en-US-general","text":"hello there"}\n'
                '{"clip_id":"b","language":"en","dialect":"en-GB-southern","text":"good evening"}\n',
                encoding="utf-8",
            )
            pred.write_text(
                '{"clip_id":"a","text":"hello there"}\n'
                '{"clip_id":"b","text":"good evening"}\n',
                encoding="utf-8",
            )
            evaluate(refs, [f"asr={pred}"], folder / "dialect.csv")
            summary = (folder / "dialect_summary.csv").read_text(encoding="utf-8")
            self.assertIn("asr,dialect,en,en-US-general", summary)
            self.assertIn("asr,dialect,en,en-GB-southern", summary)


class DatasetTests(unittest.TestCase):
    def test_example_manifest_is_valid(self):
        self.assertEqual(validate_rows(load_jsonl(FIXTURE)), [])

    def test_collected_dataset_snapshot_is_complete_and_lyrics_free(self):
        data_root = ROOT.parent / "data"
        raw = load_jsonl(data_root / "raw" / "genius_candidates.jsonl")
        cleaned = load_jsonl(data_root / "cleaned" / "genius_candidates_cleaned.jsonl")
        matched = load_jsonl(data_root / "cleaned" / "lrclib_matches.jsonl")
        pending = load_jsonl(data_root / "cleaned" / "lrclib_pending.jsonl")
        raw_ids = [row["genius_id"] for row in raw]
        cleaned_ids = {row["genius_id"] for row in cleaned}
        matched_ids = {row["genius_id"] for row in matched}
        pending_ids = {row["genius_id"] for row in pending}
        self.assertEqual(len(raw_ids), len(set(raw_ids)))
        self.assertGreaterEqual(len(raw), 1000)
        self.assertGreaterEqual(len(cleaned), 500)
        self.assertGreater(len(matched), 400)
        self.assertGreater(len(pending), 0)
        self.assertTrue(cleaned_ids.issubset(set(raw_ids)))
        self.assertFalse(matched_ids & pending_ids)
        # Both outcome files are drawn from the cleaned candidates and nothing else.
        self.assertTrue((matched_ids | pending_ids).issubset(cleaned_ids))
        self.assertTrue(all("text" not in row and "synced_lyrics" not in row for row in matched))

    def test_no_english_song_survives_anywhere_in_the_tracked_corpus(self):
        """The course constraint is Japanese-only, so English must be absent, not filtered at read time."""
        data_root = ROOT.parent / "data"
        for name in ("cleaned/benchmark_selection.jsonl", "cleaned/benchmark_video_sources.jsonl",
                     "cleaned/lrclib_matches.jsonl", "cleaned/ja_corpus.jsonl",
                     "cleaned/japanese_benchmark_selection.jsonl"):
            rows = load_jsonl(data_root / name)
            self.assertTrue(rows, f"{name} is empty")
            english = [row for row in rows if is_english_only(row)]
            self.assertEqual(english, [], f"{name} still has {len(english)} English rows")
        for name in ("searches.json", "benchmark_sample.json", "japanese_benchmark_sample.json"):
            config = json.loads((ROOT.parent / "config" / name).read_text(encoding="utf-8"))
            entries = config.get("queries") or config.get("groups") or []
            self.assertTrue(entries, f"{name} has no entries")
            self.assertEqual([e for e in entries if is_english_only(e)], [], f"{name} still targets English")

    def test_language_signal_keeps_bilingual_rows_and_drops_english_only(self):
        self.assertTrue(is_english_only({"language": "en"}))
        self.assertTrue(is_english_only({"language_hints": ["en"]}))
        self.assertFalse(is_english_only({"language": "ja"}))
        self.assertFalse(is_english_only({"language_hints": ["en", "ja"]}))
        self.assertFalse(is_english_only({}), "a row with no language signal is not evidence of English")
        self.assertEqual(language_signal({"language": "ja-JP"}), ["ja"])

    def test_japanese_corpus_is_speaker_disjoint_and_keeps_the_frozen_test_songs(self):
        data_root = ROOT.parent / "data"
        corpus = load_jsonl(data_root / "cleaned" / "ja_corpus.jsonl")
        self.assertTrue(all(row["language"] == "ja" for row in corpus))
        split_speakers = {
            split: {row["speaker_group"] for row in corpus if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        self.assertFalse(split_speakers["train"] & split_speakers["validation"])
        self.assertFalse(split_speakers["train"] & split_speakers["test"])
        self.assertFalse(split_speakers["validation"] & split_speakers["test"])
        recordings = {split: {row["recording_group"] for row in corpus if row["split"] == split} for split in split_speakers}
        self.assertFalse(recordings["train"] & recordings["test"])
        test_songs = [row for row in corpus if row["split"] == "test"]
        self.assertGreaterEqual(len(test_songs), 15)
        self.assertLessEqual(len(test_songs), 18)
        # Every song whose zero-shot baseline is already cached must still be in test.
        frozen = [row for row in corpus if row["is_pinned_test"]]
        self.assertEqual(len(frozen), 9)
        self.assertTrue(all(row["split"] == "test" for row in frozen))
        self.assertTrue(all(row["selection_seed"] == 42 for row in corpus))
        self.assertTrue(all(row["japanese_script_ratio"] >= 0.60 for row in corpus))
        self.assertTrue(all("audio_path" not in row for row in corpus), "tracked corpus must not leak local paths")

    def test_rejects_duplicate_ids_and_bad_tier(self):
        rows = load_jsonl(FIXTURE)[:2]
        rows[1]["clip_id"] = rows[0]["clip_id"]
        rows[1]["popularity_tier"] = "huge"
        errors = validate_rows(rows)
        self.assertTrue(any("duplicate clip_id" in error for error in errors))
        self.assertTrue(any("popularity_tier" in error for error in errors))

    def test_split_is_deterministic_and_keeps_groups_together(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            first, second = folder / "first.jsonl", folder / "second.jsonl"
            make_splits(FIXTURE, first, 17, (0.5, 0.25, 0.25))
            make_splits(FIXTURE, second, 17, (0.5, 0.25, 0.25))
            first_rows, second_rows = load_jsonl(first), load_jsonl(second)
            self.assertEqual([row["split"] for row in first_rows], [row["split"] for row in second_rows])
            self.assertEqual(validate_rows(first_rows, require_split=True), [])

    def test_split_keeps_collaboration_recording_together(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, output = folder / "collab.jsonl", folder / "split.jsonl"
            rows = load_jsonl(FIXTURE)[:2]
            rows[1]["recording_group"] = rows[0]["recording_group"]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            make_splits(source, output, 2026, (0.5, 0.25, 0.25))
            split_rows = load_jsonl(output)
            self.assertEqual(split_rows[0]["split"], split_rows[1]["split"])

    def test_split_validator_detects_recording_leakage(self):
        rows = load_jsonl(FIXTURE)[:2]
        rows[0]["split"], rows[1]["split"] = "train", "test"
        rows[1]["recording_group"] = rows[0]["recording_group"]
        errors = validate_rows(rows, require_split=True)
        self.assertTrue(any("recording group" in error and "crosses splits" in error for error in errors))

    def test_split_validator_detects_speaker_leakage(self):
        rows = load_jsonl(FIXTURE)[:2]
        rows[0]["split"], rows[1]["split"] = "train", "test"
        rows[1]["speaker_group"] = rows[0]["speaker_group"]
        errors = validate_rows(rows, require_split=True)
        self.assertTrue(any("speaker group" in error and "crosses splits" in error for error in errors))


class JapaneseCorpusGateTests(unittest.TestCase):
    def test_script_ratio_separates_kana_kanji_from_romanized_references(self):
        self.assertEqual(japanese_script_ratio("君の名を呼ぶ"), 1.0)
        self.assertEqual(japanese_script_ratio("Into the Night"), 0.0)
        self.assertEqual(japanese_script_ratio(""), 0.0)
        self.assertLess(japanese_script_ratio("Hi! ula-la-la 春が"), 0.60)

    def test_title_blocklist_rejects_non_studio_recordings(self):
        for title in ("踊り子 - Vaundy【Acoustic Duet Cover】", "Vaundy 踊り子 INSTRUMENTAL",
                      "Aimer - Kataomoi / THE FIRST TAKE", "カラオケ 宿命", "King Gnu - Doron LIVE MV"):
            self.assertTrue(TITLE_BLOCKLIST.search(title), title)
        for title in ("King Gnu - AIZO", "ClariS『hitorigoto』Music Video Full", "YOASOBI「怪物」Official Music Video"):
            self.assertIsNone(TITLE_BLOCKLIST.search(title), title)

    def test_source_pick_prefers_duration_match_and_skips_blocked_titles(self):
        candidates = [
            {"video_id": "live", "video_title": "Song (LIVE)", "channel": "Artist", "duration_seconds": 200},
            {"video_id": "far", "video_title": "Song", "channel": "random", "duration_seconds": 400},
            {"video_id": "good", "video_title": "Song", "channel": "Artist Official", "duration_seconds": 202},
        ]
        chosen, method = pick_source(candidates, 200.0, "Artist")
        self.assertEqual(chosen["video_id"], "good")
        self.assertIn("duration match", method)

    def test_source_pick_reports_when_nothing_qualifies(self):
        chosen, method = pick_source([{"video_id": "x", "video_title": "Song cover", "duration_seconds": 200}], 200.0, "Artist")
        self.assertIsNone(chosen)
        self.assertIn("no_candidate", method)

    def test_parse_synced_drops_blank_and_unstamped_lines(self):
        lines = parse_synced("[00:01.50] first\n[00:05.00] \nnot timestamped\n[01:00.25] second")
        self.assertEqual(lines, [(1.5, "first"), (60.25, "second")])


class JapaneseWindowAndAlignmentTests(unittest.TestCase):
    def test_windows_respect_the_length_cap_and_keep_order(self):
        lines = parse_synced("\n".join(f"[00:{index * 5:02d}.00] line{index}" for index in range(12)))
        windows = build_windows(lines, 70.0, window_seconds=28.0)
        self.assertTrue(all(end - start <= 28.0 + 1e-6 for start, end, _ in windows))
        self.assertEqual([start for start, _, _ in windows], sorted(start for start, _, _ in windows))
        self.assertEqual("\n".join(text for _, _, text in windows).count("line"), 12)

    def test_window_does_not_charge_a_long_instrumental_gap_to_the_previous_line(self):
        windows = build_windows(parse_synced("[00:00.00] a\n[01:30.00] b"), 120.0)
        self.assertEqual(len(windows), 2)
        self.assertLessEqual(windows[0][1] - windows[0][0], 12.0)

    def test_bag_f1_is_order_insensitive(self):
        self.assertEqual(bag_f1("あいうえお", "おえういあ"), 1.0)
        self.assertEqual(bag_f1("あいうえお", "かきくけこ"), 0.0)
        self.assertEqual(bag_f1("あい", ""), 0.0)

    def test_offset_search_recovers_a_known_intro_shift(self):
        windows = [(0.0, 10.0, "あいうえお"), (12.0, 22.0, "かきくけこ")]
        shift = 14.0
        segments = [
            {"start": 0.0 + shift, "end": 10.0 + shift, "text": "あいうえお"},
            {"start": 12.0 + shift, "end": 22.0 + shift, "text": "かきくけこ"},
        ]
        offset, confidence = estimate_offset(segments, windows, duration_delta=shift)
        self.assertAlmostEqual(offset, shift, delta=1.0)
        self.assertGreater(confidence, 0.9)

    def test_offset_search_is_reported_as_low_confidence_when_audio_does_not_match(self):
        windows = [(0.0, 10.0, "あいうえお")]
        segments = [{"start": 0.0, "end": 10.0, "text": "まったく違う文字列"}]
        _, confidence = estimate_offset(segments, windows, duration_delta=0.0)
        self.assertLess(confidence, 0.35)


class JapaneseMetricTests(unittest.TestCase):
    def test_characters_normalize_width_case_punctuation_and_whitespace(self):
        self.assertEqual(characters("ＡＢＣ　あい"), ["a", "b", "c", "あ", "い"])
        self.assertEqual(characters("カタオモイ、です！\nほんと"), list("カタオモイですほんと"))

    def test_perfect_and_empty_hypotheses(self):
        perfect = score_pair("君の名を呼ぶ", "君の名を呼ぶ")
        self.assertEqual((perfect["cer"], perfect["character_f1"], perfect["exact_match"]), (0.0, 1.0, 1))
        empty = score_pair("君の名を呼ぶ", "")
        self.assertEqual((empty["cer"], empty["character_f1"], empty["deletions"]), (1.0, 0.0, 6))

    def test_hallucination_is_charged_to_precision_not_recall(self):
        result = score_pair("君の名", "君の名" + "あ" * 30)
        self.assertEqual(result["character_recall"], 1.0)
        self.assertLess(result["character_precision"], 0.2)
        self.assertGreater(result["cer"], 1.0, "CER is a rate and may exceed 100%")

    def test_japanese_evaluator_rejects_non_japanese_references(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            references, predictions = folder / "ref.jsonl", folder / "hyp.jsonl"
            references.write_text('{"clip_id":"a","language":"en","text":"hello"}\n', encoding="utf-8")
            predictions.write_text('{"clip_id":"a","text":"hello"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                evaluate_japanese(references, [f"asr={predictions}"], folder / "out")
            self.assertIn("Japanese-only", str(caught.exception))

    def test_japanese_evaluator_requires_the_exact_frozen_clip_set(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            references, predictions = folder / "ref.jsonl", folder / "hyp.jsonl"
            references.write_text(
                '{"clip_id":"a","language":"ja","text":"あい"}\n{"clip_id":"b","language":"ja","text":"うえ"}\n',
                encoding="utf-8")
            predictions.write_text('{"clip_id":"a","text":"あい"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                evaluate_japanese(references, [f"asr={predictions}"], folder / "out")
            self.assertIn("frozen utterance set", str(caught.exception))

    def test_japanese_evaluator_writes_per_utterance_speaker_and_song_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            references, predictions = folder / "ref.jsonl", folder / "hyp.jsonl"
            references.write_text(
                '{"clip_id":"s1-u01","song_id":"s1","speaker_group":"alpha","split":"test","language":"ja",'
                '"duration_seconds":10,"text":"君の名を呼ぶ"}\n'
                '{"clip_id":"s2-u01","song_id":"s2","speaker_group":"beta","split":"test","language":"ja",'
                '"duration_seconds":10,"text":"あいうえお"}\n',
                encoding="utf-8")
            predictions.write_text(
                '{"clip_id":"s1-u01","text":"君の名を呼ぶ","tier":"baseline","inference_seconds":1}\n'
                '{"clip_id":"s2-u01","text":"","tier":"baseline","inference_seconds":1}\n',
                encoding="utf-8")
            rows = evaluate_japanese(references, [f"asr={predictions}"], folder / "out")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["songs"], 2)
            self.assertEqual(rows[0]["speakers"], 2)
            # 6 of 11 reference characters deleted.
            self.assertAlmostEqual(rows[0]["cer"], 5 / 11, places=6)
            self.assertAlmostEqual(rows[0]["real_time_factor"], 2 / 20, places=6)
            for suffix in ("_per_utterance", "_summary", "_per_speaker", "_per_song"):
                self.assertTrue((folder / f"out{suffix}.csv").is_file(), suffix)
            self.assertNotIn("wer", (folder / "out_summary.csv").read_text(encoding="utf-8"))


class JapaneseSegmentationTests(unittest.TestCase):
    """Cutting clips shells out to ffmpeg, so this exercises the real binary when present."""

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def test_segmenter_applies_the_offset_and_reports_used_versus_source_audio(self):
        if not self.ffmpeg_available():
            self.skipTest("ffmpeg/ffprobe not installed")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            audio = folder / "song.wav"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=440:duration=90", "-ac", "1", "-ar", "16000", str(audio)],
                check=True, capture_output=True,
            )
            (folder / "data" / "references").mkdir(parents=True)
            # Lyric lines at 0/10/20 s; the audio carries a 15 s intro.
            synced = "[00:00.00] あいうえお\n[00:10.00] かきくけこ\n[00:20.00] さしすせそ"
            (folder / "data" / "references" / "lrclib_lyrics.jsonl").write_text(
                json.dumps({"clip_id": "song-1", "lrclib_id": 1, "text": "あいうえお",
                            "synced_lyrics": synced, "duration_seconds": 75.0}, ensure_ascii=False) + "\n",
                encoding="utf-8")
            corpus = folder / "corpus.jsonl"
            corpus.write_text(json.dumps({
                "clip_id": "song-1", "genius_id": 1, "lrclib_id": 1, "split": "test",
                "speaker_group": "alpha", "recording_group": "song-1", "title": "Song", "artist": "Artist",
                "artists": ["Artist"], "genius_url": "https://genius.com/x", "lrclib_url": "https://lrclib.net/api/get/1",
                "popularity_tier": "mid", "youtube_url": "https://www.youtube.com/watch?v=abc",
                "language_status": "curated_from_song_and_reference_metadata", "dialect": "unassessed",
                "dialect_status": "requires_audio_review", "reference_status": "lrclib_assumed_gold",
                "reference_duration_seconds": 75.0, "video_duration_seconds": 90.0,
                "audio_path": str(audio),
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            alignment = folder / "alignment.jsonl"
            alignment.write_text(json.dumps({
                "clip_id": "song-1", "offset_seconds": 15.0, "alignment_confidence": 0.8,
                "alignment_method": "asr_grid_search",
            }) + "\n", encoding="utf-8")
            output = folder / "utterances.jsonl"
            segment(folder, corpus, alignment, folder / "clips", output, force=False)

            rows = load_jsonl(output)
            self.assertTrue(rows)
            self.assertTrue(all(row["language"] == "ja" for row in rows))
            self.assertTrue(all(row["alignment_offset_seconds"] == 15.0 for row in rows))
            # The first window starts at lyric 0 s, so its audio starts at the offset.
            self.assertAlmostEqual(rows[0]["audio_start_seconds"], 15.0 - 0.25, places=2)
            for row in rows:
                clip = Path(row["audio_path"])
                self.assertTrue(clip.is_file(), clip)
                self.assertAlmostEqual(probe_duration(clip), row["duration_seconds"], delta=0.2)
                self.assertLessEqual(row["duration_seconds"], 30.0)
            # Cut audio must stay inside the real file rather than running past its end.
            self.assertLessEqual(max(row["audio_start_seconds"] + row["duration_seconds"] for row in rows), 90.5)
            self.assertEqual(validate_rows(rows, require_split=True), [])

    def test_segmenter_is_idempotent_and_keeps_clip_ids_stable(self):
        if not self.ffmpeg_available():
            self.skipTest("ffmpeg/ffprobe not installed")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            audio = folder / "song.wav"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=440:duration=40", "-ac", "1", "-ar", "16000", str(audio)],
                check=True, capture_output=True,
            )
            (folder / "data" / "references").mkdir(parents=True)
            (folder / "data" / "references" / "lrclib_lyrics.jsonl").write_text(
                json.dumps({"clip_id": "song-1", "lrclib_id": 1, "text": "あ",
                            "synced_lyrics": "[00:00.00] あいうえお\n[00:08.00] かきくけこ",
                            "duration_seconds": 40.0}, ensure_ascii=False) + "\n", encoding="utf-8")
            corpus = folder / "corpus.jsonl"
            corpus.write_text(json.dumps({
                "clip_id": "song-1", "genius_id": 1, "lrclib_id": 1, "split": "train",
                "speaker_group": "alpha", "recording_group": "song-1", "title": "Song", "artist": "Artist",
                "artists": ["Artist"], "genius_url": "https://genius.com/x", "lrclib_url": "https://lrclib.net/api/get/1",
                "popularity_tier": "mid", "youtube_url": "https://www.youtube.com/watch?v=abc",
                "reference_duration_seconds": 40.0, "video_duration_seconds": 40.0, "audio_path": str(audio),
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            alignment = folder / "alignment.jsonl"
            alignment.write_text(json.dumps({"clip_id": "song-1", "offset_seconds": 0.0,
                                             "alignment_confidence": 0.9, "alignment_method": "asr_grid_search"}) + "\n",
                                 encoding="utf-8")
            first, second = folder / "a.jsonl", folder / "b.jsonl"
            segment(folder, corpus, alignment, folder / "clips", first, force=False)
            segment(folder, corpus, alignment, folder / "clips", second, force=False)
            left, right = load_jsonl(first), load_jsonl(second)
            self.assertEqual([row["clip_id"] for row in left], [row["clip_id"] for row in right])
            self.assertEqual(left[0]["clip_id"], "song-1-u01")


class PinnedSplitTests(unittest.TestCase):
    def rows(self):
        return [
            {"clip_id": f"{speaker}-{index}", "speaker_group": speaker, "recording_group": f"{speaker}-{index}"}
            for speaker, count in (("alpha", 4), ("beta", 3), ("gamma", 3), ("delta", 2), ("epsilon", 2))
            for index in range(count)
        ]

    def test_pinned_speakers_land_in_test_and_never_elsewhere(self):
        assignment = assign_splits(self.rows(), 42, (0.62, 0.20, 0.18), frozenset({"alpha"}))
        self.assertTrue(all(split == "test" for clip, split in assignment.items() if clip.startswith("alpha")))
        self.assertTrue(all(not clip.startswith("alpha") for clip, split in assignment.items() if split != "test"))
        self.assertNotIn("test", {split for clip, split in assignment.items() if not clip.startswith("alpha")})

    def test_assignment_is_deterministic_for_a_seed_and_keeps_speakers_together(self):
        first = assign_splits(self.rows(), 42, (0.62, 0.20, 0.18), frozenset({"alpha"}))
        second = assign_splits(self.rows(), 42, (0.62, 0.20, 0.18), frozenset({"alpha"}))
        self.assertEqual(first, second)
        by_speaker = {}
        for clip, split in first.items():
            by_speaker.setdefault(clip.rsplit("-", 1)[0], set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in by_speaker.values()))

    def test_unknown_pinned_speaker_is_an_error_rather_than_silently_ignored(self):
        with self.assertRaises(ValueError) as caught:
            assign_splits(self.rows(), 42, (0.62, 0.20, 0.18), frozenset({"nobody"}))
        self.assertIn("nobody", str(caught.exception))


class ColabHandoffTests(unittest.TestCase):
    """The GPU handoff is a second copy of the scoring path, so it has to be pinned.

    ``colab_handoff/`` exists because CPU inference is too slow, and it deliberately
    duplicates the scorer so a bare Colab VM needs no project install. A duplicate
    is only safe while it stays identical, which is what these tests enforce.
    """

    HANDOFF = ROOT.parent / "colab_handoff"

    @classmethod
    def load(cls, name: str):
        import importlib.util

        if str(cls.HANDOFF) not in sys.path:
            sys.path.insert(0, str(cls.HANDOFF))
        spec = importlib.util.spec_from_file_location(name, cls.HANDOFF / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(name, module)
        spec.loader.exec_module(module)
        return module

    def test_standalone_scorer_matches_the_project_scorer_exactly(self):
        standalone = self.load("ja_eval_standalone")
        cases = [
            ("ひらがなのれんしゅう", "ひらがなのれんしゅう"),
            ("ひらがなのれんしゅう", "ひらがなの"),
            ("ひらがな", "ひらがなのれんしゅうです"),
            ("カタカナと漢字", "かたかなと漢字"),
            ("ＡＢＣ　１２３", "abc 123"),
            ("句読点、あり。", "句読点あり"),
            ("", "なにか"),
            ("なにか", ""),
            ("", ""),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(
                    score_pair(reference, hypothesis),
                    standalone.score_pair(reference, hypothesis),
                )
                self.assertEqual(characters(reference), standalone.characters(reference))

    def test_standalone_normalization_folds_width_case_and_punctuation(self):
        standalone = self.load("ja_eval_standalone")
        # NFKC collapses full-width to half-width, so these must score as identical.
        self.assertEqual(standalone.score_pair("ＡＢＣ", "abc")["cer"], 0.0)
        # Punctuation and spaces are removed before counting, not substituted.
        self.assertEqual(standalone.score_pair("あ、い。う", "あいう")["cer"], 0.0)

    def test_runner_checkpoints_and_revisions_match_the_tracked_config(self):
        runner = self.load("run_colab")
        config = json.loads(
            (ROOT.parent / "config" / "benchmark_models.json").read_text(encoding="utf-8"))
        tracked = {entry["name"]: entry for entry in config["systems"]}
        for system in runner.SYSTEMS:
            with self.subTest(system=system["name"]):
                self.assertIn(system["name"], tracked)
                self.assertEqual(system["repo_id"], tracked[system["name"]]["repo_id"])
                self.assertEqual(system["revision"], tracked[system["name"]]["revision"])
                self.assertEqual(system["tier"], tracked[system["name"]]["tier"])
                self.assertRegex(system["revision"], r"^[0-9a-f]{40}$")
        # The LoRA contribution is trained by the runner, so it is configured but not
        # listed as a pretrained system to download.
        self.assertNotIn("whisper-small-lora-ja", {s["name"] for s in runner.SYSTEMS})
        self.assertEqual(runner.FINETUNE_BASE, tracked["whisper-small-lora-ja"]["repo_id"])

    def test_runner_pins_japanese_and_never_language_detects(self):
        runner = self.load("run_colab")
        self.assertEqual(runner.LANGUAGE, "ja")
        self.assertEqual(runner.TASK, "transcribe")
        config = json.loads(
            (ROOT.parent / "config" / "benchmark_models.json").read_text(encoding="utf-8"))
        self.assertEqual(runner.BEAM_SIZE, config["decode_settings"]["beam_size"])

    def test_excluded_nvidia_models_are_absent_from_the_runner(self):
        runner = self.load("run_colab")
        repos = {system["repo_id"] for system in runner.SYSTEMS}
        for excluded in ("nvidia/canary-1b-v2", "nvidia/parakeet-tdt-0.6b-v3"):
            self.assertNotIn(excluded, repos)

    def test_bundle_manifest_drops_local_paths_and_keeps_the_reference(self):
        from lyrics_asr.package_colab import PORTABLE_FIELDS, portable_row

        row = {
            "clip_id": "genius-1-u01", "song_id": "genius-1", "split": "test",
            "speaker_group": "someone", "language": "ja", "text": "テスト",
            "duration_seconds": 12.5, "reference_characters": 3,
            "audio_path": r"data\clips\ja\genius-1\genius-1-u01.wav",
            "source_audio_path": r"data\audio\ja\abc.wav",
            "reference_path": r"data\private\ja_utterances.jsonl",
        }
        out = portable_row(row, "clips/genius-1/genius-1-u01.opus")
        self.assertEqual(out["audio_path"], "clips/genius-1/genius-1-u01.opus")
        self.assertNotIn("source_audio_path", out)
        self.assertNotIn("reference_path", out)
        self.assertEqual(out["text"], "テスト")
        self.assertNotIn("source_audio_path", PORTABLE_FIELDS)
        self.assertFalse([key for key in out if "\\" in str(out[key])])

    def test_packager_refuses_a_manifest_that_is_not_japanese_only(self):
        from lyrics_asr.package_colab import package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "utterances.jsonl"
            manifest.write_text(
                json.dumps({"clip_id": "a", "song_id": "s", "split": "test",
                            "language": "en", "text": "hello",
                            "audio_path": "missing.wav"}) + "\n",
                encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                package(root, manifest, root / "out.zip", self.HANDOFF, "wav", "48k",
                        ["test"], False)
            self.assertIn("Japanese-only", str(caught.exception))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to build a bundle")
    def test_packager_builds_a_portable_self_contained_zip(self):
        import zipfile

        from lyrics_asr.package_colab import package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "data" / "clips" / "ja" / "genius-1" / "genius-1-u01.wav"
            clip.parent.mkdir(parents=True)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=440:duration=3", "-ac", "1", "-ar", "16000",
                 str(clip)], check=True)
            manifest = root / "utterances.jsonl"
            manifest.write_text(json.dumps({
                "clip_id": "genius-1-u01", "song_id": "genius-1", "split": "test",
                "speaker_group": "someone", "language": "ja", "text": "テスト",
                "duration_seconds": 3.0, "reference_characters": 3,
                "audio_path": "data/clips/ja/genius-1/genius-1-u01.wav",
                "source_audio_path": "data/audio/ja/abc.wav",
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            output = package(root, manifest, root / "bundle.zip", self.HANDOFF,
                             "opus", "48k", ["test"], False)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                rows = [json.loads(line) for line in
                        archive.read("ja_utterances.jsonl").decode("utf-8").splitlines()]
            # The runner and the scorer must travel with the data, or a bare Colab
            # VM cannot run anything.
            self.assertIn("run_colab.py", names)
            self.assertIn("ja_eval_standalone.py", names)
            self.assertEqual(rows[0]["audio_path"], "clips/genius-1/genius-1-u01.opus")
            self.assertIn(rows[0]["audio_path"], names)
            self.assertNotIn("source_audio_path", rows[0])

    def test_notebook_is_valid_json_and_requests_a_gpu(self):
        notebook = json.loads(
            (self.HANDOFF / "colab_runner.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(notebook["metadata"]["colab"]["gpuType"], "T4")
        self.assertTrue(notebook["cells"])
        for cell in notebook["cells"]:
            self.assertIn(cell["cell_type"], {"code", "markdown"})
            self.assertIsInstance(cell["source"], list)


if __name__ == "__main__":
    unittest.main()
