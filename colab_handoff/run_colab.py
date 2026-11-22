"""One-file Colab T4 runner: 6 pretrained Japanese-capable ASR systems + a LoRA fine-tune.

Run the whole study on a free Colab T4 instead of a laptop CPU::

    python run_colab.py --stage all

Stages (each is independently resumable, because Colab disconnects):

``baselines``
    Transcribe the frozen Japanese utterance clips with six already-trained
    systems. Five are CTranslate2 (``faster-whisper``) checkpoints run at
    ``float16`` on CUDA; one (``litagin/anime-whisper``) ships only in
    ``transformers`` format and is run through ``WhisperForConditionalGeneration``
    at ``float16``. Every system decodes with ``language="ja"``,
    ``task="transcribe"``, ``num_beams=5`` and no VAD, so the only thing that
    varies between rows of the results table is the checkpoint.

``finetune``
    LoRA fine-tune ``openai/whisper-small`` on the Japanese ``train`` split
    (``language="ja"``, ``task="transcribe"``), score the adapter on
    ``validation`` after every epoch, keep the lowest-CER epoch, then decode the
    ``test`` split exactly once with that adapter. On a T4 this is a few minutes:
    the train split is ~400 utterances and only the LoRA matrices get gradients.

``evaluate``
    Score every prediction file with ``ja_eval_standalone.py`` (CER + Character
    F1, no WER - see that file's docstring for the Hugging Face Audio Course
    citation) and write ``final_metrics.json`` plus Markdown tables.

Outputs land in ``final_predictions/`` (one JSONL per system per split),
``reports/`` (CSV + Markdown) and ``final_metrics.json``.

Nothing here reads the repository: the bundle produced by
``lyrics-asr-package-colab`` is the only input, and ``ja_eval_standalone.py``
duplicates the local scorer's arithmetic so Colab numbers and laptop numbers are
comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator

import ja_eval_standalone as scorer

SAMPLE_RATE = 16000
BEAM_SIZE = 5
LANGUAGE = "ja"
TASK = "transcribe"

# Pinned checkpoints. `revision` is the exact commit SHA each number was produced
# with: Hugging Face branches move, and a benchmark table that cannot be tied to a
# commit cannot be reproduced.
SYSTEMS: list[dict[str, Any]] = [
    {
        "name": "faster-whisper-base",
        "repo_id": "Systran/faster-whisper-base",
        "revision": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
        "backend": "ct2",
        "tier": "baseline",
        "parameters": "74M",
    },
    {
        "name": "faster-whisper-small",
        "repo_id": "Systran/faster-whisper-small",
        "revision": "536b0662742c02347bc0e980a01041f333bce120",
        "backend": "ct2",
        "tier": "baseline",
        "parameters": "244M",
    },
    {
        "name": "faster-whisper-medium",
        "repo_id": "Systran/faster-whisper-medium",
        "revision": "08e178d48790749d25932bbc082711ddcfdfbc4f",
        "backend": "ct2",
        "tier": "baseline",
        "parameters": "769M",
    },
    {
        "name": "faster-whisper-large-v3-turbo",
        "repo_id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "revision": "4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
        "backend": "ct2",
        "tier": "baseline",
        "parameters": "809M",
    },
    {
        "name": "kotoba-whisper-v2.0",
        "repo_id": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "revision": "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc",
        "backend": "ct2",
        "tier": "japanese-pretrained",
        "parameters": "756M",
    },
    {
        "name": "anime-whisper",
        "repo_id": "litagin/anime-whisper",
        "revision": "22e2008a8182b357da3922a6308d095008f72973",
        "backend": "hf",
        "tier": "japanese-pretrained",
        "parameters": "756M",
    },
]

FINETUNE_BASE = "openai/whisper-small"
FINETUNE_NAME = "whisper-small-lora-ja"

PIP_PACKAGES = [
    "faster-whisper==1.2.1",
    "transformers>=4.44,<5",
    "peft>=0.13",
    "accelerate>=0.34",
    "soundfile",
    "av",
]


# --------------------------------------------------------------------------- io


def log(message: str) -> None:
    print(message, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return scorer.read_jsonl(path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_audio(path: Path) -> "Any":
    """Decode any clip container to mono float32 at 16 kHz.

    The bundle ships Opus by default (lossless PCM is ~5x larger for the same
    audio), so the loader has to cope with a compressed container. PyAV, which
    ``faster-whisper`` already depends on, is tried first; the rest are fallbacks
    so a missing optional dependency does not end the run.
    """
    try:
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    except Exception:
        pass
    try:
        import numpy as np
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if rate != SAMPLE_RATE:
            target = int(round(len(data) * SAMPLE_RATE / rate))
            data = np.interp(
                np.linspace(0, len(data) - 1, target, dtype="float64"),
                np.arange(len(data), dtype="float64"),
                data,
            ).astype("float32")
        return data
    except Exception:
        pass
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return np.frombuffer(raw, dtype="float32")


def resolve_clip(bundle: Path, row: dict[str, Any]) -> Path:
    candidate = bundle / str(row["audio_path"])
    if candidate.exists():
        return candidate
    direct = Path(str(row["audio_path"]))
    if direct.exists():
        return direct
    raise FileNotFoundError(
        f"clip for {row['clip_id']} not found at {candidate}; "
        f"is --bundle pointing at the unzipped colab_bundle directory?"
    )


# ----------------------------------------------------------------- environment


def ensure_dependencies(install: bool) -> None:
    if not install:
        return
    log("Installing dependencies (quiet; a few minutes on a cold VM)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES],
        check=True,
    )
    log("Dependencies installed.")


def describe_device(requested: str) -> tuple[str, str, dict[str, Any]]:
    """Return (device, compute_type, environment report).

    Falls back to CPU rather than crashing, but says so loudly: a silent CPU run
    would take hours and the reported real-time factors would not mean what the
    table claims.
    """
    report: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    device = requested
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            log("!! --device cuda was asked for but no GPU is visible.")
            log("!! In Colab: Runtime > Change runtime type > T4 GPU, then rerun.")
            device = "cpu"
    except ImportError:
        device = "cpu" if device == "auto" else device
    compute_type = "float16" if device == "cuda" else "int8"
    report["device"], report["compute_type"] = device, compute_type
    if device == "cpu":
        log("!! Running on CPU. This is ~20x slower than a T4; expect hours, not minutes.")
    else:
        log(f"Device: {device} ({report.get('gpu', '?')}, {compute_type})")
    return device, compute_type, report


# ------------------------------------------------------------------- inference


def pending_rows(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Rows whose prediction is not already on disk, so a disconnect costs one clip."""
    if not path.exists():
        return rows
    done = {str(row.get("clip_id")) for row in read_jsonl(path)}
    remaining = [row for row in rows if str(row["clip_id"]) not in done]
    if remaining:
        log(f"  resuming: {len(done)} already done, {len(remaining)} to go")
    return remaining


def transcribe_ct2(
    system: dict[str, Any],
    rows: list[dict[str, Any]],
    bundle: Path,
    device: str,
    compute_type: str,
) -> Iterator[dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        system["repo_id"],
        revision=system["revision"],
        device=device,
        compute_type=compute_type,
    )
    for index, row in enumerate(rows, start=1):
        audio = load_audio(resolve_clip(bundle, row))
        started = time.perf_counter()
        segments, _ = model.transcribe(
            audio,
            language=LANGUAGE,
            task=TASK,
            beam_size=BEAM_SIZE,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        elapsed = time.perf_counter() - started
        if index % 25 == 0 or index == len(rows):
            log(f"  [{index}/{len(rows)}] {row['clip_id']} {elapsed:.1f}s")
        yield {"clip_id": row["clip_id"], "text": text, "inference_seconds": round(elapsed, 3)}
    del model
    free_gpu()


def transcribe_hf(
    system: dict[str, Any],
    rows: list[dict[str, Any]],
    bundle: Path,
    device: str,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    """Batched ``generate`` for checkpoints that ship only in transformers format.

    ``model.generate`` is called directly rather than via ``pipeline`` so this
    path uses the same decoding call as the LoRA model below - one less way for
    the fine-tune and the pretrained rows to become incomparable.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        system["repo_id"], revision=system["revision"], language="Japanese", task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        system["repo_id"], revision=system["revision"], torch_dtype=dtype
    ).to(device).eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        audio = [load_audio(resolve_clip(bundle, row)) for row in batch]
        started = time.perf_counter()
        texts = generate_texts(model, processor, audio, device, dtype)
        elapsed = time.perf_counter() - started
        share = elapsed / len(batch)
        log(f"  [{min(start + batch_size, len(rows))}/{len(rows)}] batch {elapsed:.1f}s")
        for row, text in zip(batch, texts):
            yield {"clip_id": row["clip_id"], "text": text, "inference_seconds": round(share, 3)}
    del model
    free_gpu()


def generate_texts(model, processor, audio: list, device: str, dtype) -> list[str]:
    """Shared beam-search decode for every transformers-backed Whisper here."""
    import torch

    features = processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(device=device, dtype=dtype)
    with torch.no_grad():
        tokens = model.generate(
            features,
            language=LANGUAGE,
            task=TASK,
            num_beams=BEAM_SIZE,
            max_new_tokens=200,
        )
    return [text.strip() for text in processor.batch_decode(tokens, skip_special_tokens=True)]


def free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def amp(device: str, torch_module) -> Any:
    """fp16 autocast on CUDA, a no-op elsewhere.

    ``torch.autocast("cuda", enabled=False)`` is not reliably harmless on a
    CPU-only build, so the CPU path gets a plain null context instead.
    """
    if device != "cuda":
        return nullcontext()
    return torch_module.autocast("cuda", dtype=torch_module.float16)


def grad_scaler(device: str, torch_module) -> Any:
    """``torch.amp.GradScaler`` moved namespace in torch 2.3; support both."""
    enabled = device == "cuda"
    try:
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(enabled=enabled)


def run_baselines(
    bundle: Path,
    utterances: list[dict[str, Any]],
    splits: list[str],
    names: list[str],
    predictions_dir: Path,
    device: str,
    compute_type: str,
    batch_size: int,
) -> None:
    chosen = [system for system in SYSTEMS if system["name"] in names]
    missing = sorted(set(names) - {system["name"] for system in chosen})
    if missing:
        raise SystemExit(f"Unknown system(s) {missing}; available: {[s['name'] for s in SYSTEMS]}")
    for split in splits:
        rows = [row for row in utterances if str(row.get("split")) == split]
        if not rows:
            log(f"No {split} utterances; skipping.")
            continue
        for system in chosen:
            path = predictions_dir / f"{split}__{system['name']}.jsonl"
            todo = pending_rows(rows, path)
            if not todo:
                log(f"{split}/{system['name']}: complete ({len(rows)} clips), skipping")
                continue
            log(f"\n=== {split}/{system['name']} ({system['parameters']}, {system['backend']}) "
                f"{len(todo)} clips ===")
            if system["backend"] == "ct2":
                stream = transcribe_ct2(system, todo, bundle, device, compute_type)
            else:
                stream = transcribe_hf(system, todo, bundle, device, batch_size)
            started = time.perf_counter()
            for record in stream:
                append_jsonl(path, [{
                    **record,
                    "split": split,
                    "system": system["name"],
                    "model": system["repo_id"],
                    "model_revision": system["revision"],
                    "tier": system["tier"],
                    "device": device,
                    "compute_type": compute_type if system["backend"] == "ct2" else "float16",
                    "beam_size": BEAM_SIZE,
                    "language": LANGUAGE,
                }])
            log(f"=== {split}/{system['name']} done in {(time.perf_counter() - started) / 60:.1f} min "
                f"-> {path}")


# -------------------------------------------------------------------- finetune


def finetune(
    bundle: Path,
    utterances: list[dict[str, Any]],
    predictions_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    lora_rank: int,
    seed: int,
    adapter_dir: Path,
) -> dict[str, Any]:
    """LoRA fine-tune whisper-small on train, pick the best epoch on validation, decode test once.

    Only the LoRA matrices receive gradients, so the optimizer state is a few MB
    and a T4 finishes well inside a free session. The epoch is chosen on
    ``validation`` and the chosen adapter touches ``test`` exactly once, which is
    the whole point of keeping three splits.
    """
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device != "cuda":
        log("!! Fine-tuning on CPU is impractically slow. Use a T4 runtime.")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = [row for row in utterances if str(row.get("split")) == "train"]
    validation = [row for row in utterances if str(row.get("split")) == "validation"]
    test = [row for row in utterances if str(row.get("split")) == "test"]
    if not train or not validation or not test:
        raise SystemExit("finetune needs train, validation and test utterances in the bundle")
    log(f"\n=== LoRA fine-tune {FINETUNE_BASE}: train={len(train)} "
        f"validation={len(validation)} test={len(test)} ===")

    processor = WhisperProcessor.from_pretrained(FINETUNE_BASE, language="Japanese", task=TASK)
    tokenizer = processor.tokenizer

    def features_for(rows: list[dict[str, Any]]) -> list:
        """Mel features are computed once and cached; 3 epochs would otherwise redecode."""
        out = []
        for index, row in enumerate(rows, start=1):
            audio = load_audio(resolve_clip(bundle, row))
            mel = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
            out.append(mel.astype("float16"))
            if index % 100 == 0 or index == len(rows):
                log(f"  features {index}/{len(rows)}")
        return out

    log("Extracting log-mel features (cached in RAM for all epochs)...")
    train_features = features_for(train)
    train_labels = [tokenizer(str(row["text"]), truncation=True, max_length=448).input_ids
                    for row in train]

    model = WhisperForConditionalGeneration.from_pretrained(FINETUNE_BASE)
    # Let `generate` build the ja/transcribe prompt itself; a stale forced_decoder_ids
    # on the config silently overrides the language argument on some versions.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    decoder_start = model.config.decoder_start_token_id

    model = get_peft_model(model, LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    ))
    model.print_trainable_parameters()
    model = model.to(device)

    def set_cache(flag: bool) -> None:
        inner = model.base_model.model
        inner.config.use_cache = flag
        inner.generation_config.use_cache = flag

    def collate(indices: list[int]) -> tuple:
        mels = torch.tensor(np.stack([train_features[i] for i in indices]), dtype=torch.float32)
        rows = [train_labels[i] for i in indices]
        width = max(len(row) for row in rows)
        padded = torch.tensor([row + [-100] * (width - len(row)) for row in rows], dtype=torch.long)
        # The model prepends decoder_start_token_id itself when shifting labels right,
        # so the tokenizer's leading <|startoftranscript|> has to come off.
        if bool((padded[:, 0] == decoder_start).all()):
            padded = padded[:, 1:]
        return mels.to(device), padded.to(device)

    def score(rows: list[dict[str, Any]], batch: int, tag: str) -> tuple[float, float, list[dict[str, Any]]]:
        set_cache(True)
        model.eval()
        records: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            audio = [load_audio(resolve_clip(bundle, row)) for row in chunk]
            began = time.perf_counter()
            with amp(device, torch):
                texts = generate_texts(model, processor, audio, device, torch.float32)
            share = (time.perf_counter() - began) / len(chunk)
            for row, text in zip(chunk, texts):
                records.append({"clip_id": row["clip_id"], "text": text,
                                "inference_seconds": round(share, 3)})
        totals: dict[str, float] = {key: 0.0 for key in scorer.COUNT_KEYS}
        for row, record in zip(rows, records):
            metrics = scorer.score_pair(str(row["text"]), record["text"])
            for key in scorer.COUNT_KEYS:
                totals[key] += metrics[key]
        summary = scorer.summarize(totals, len(rows))
        log(f"  {tag}: CER {summary['cer'] * 100:.2f}%  CharF1 {summary['character_f1'] * 100:.2f}%")
        set_cache(False)
        model.train()
        return summary["cer"], summary["character_f1"], records

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=0.0,
    )
    order = np.arange(len(train))
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(0.1 * total_steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
        (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    ))
    scaler = grad_scaler(device, torch)

    set_cache(False)
    model.train()
    history: list[dict[str, Any]] = []
    best = {"cer": float("inf"), "epoch": None}
    adapter_dir.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        np.random.shuffle(order)
        running, batches = 0.0, 0
        epoch_started = time.perf_counter()
        for start in range(0, len(order), batch_size):
            mels, labels = collate([int(i) for i in order[start:start + batch_size]])
            with amp(device, torch):
                loss = model(input_features=mels, labels=labels).loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += float(loss.detach())
            batches += 1
        log(f"epoch {epoch}/{epochs}: loss {running / max(1, batches):.4f} "
            f"({time.perf_counter() - epoch_started:.0f}s)")
        cer, f1, _ = score(validation, eval_batch_size, f"validation epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": round(running / max(1, batches), 4),
                        "validation_cer": cer, "validation_character_f1": f1})
        if cer < best["cer"]:
            best = {"cer": cer, "character_f1": f1, "epoch": epoch}
            target = adapter_dir / "best"
            if target.exists():
                shutil.rmtree(target)
            model.save_pretrained(str(target))
            log(f"  new best epoch ({cer * 100:.2f}% CER) -> {target}")

    minutes = (time.perf_counter() - training_started) / 60
    log(f"Training finished in {minutes:.1f} min; best epoch {best['epoch']} "
        f"at {best['cer'] * 100:.2f}% validation CER")

    # Reload the selected epoch so `test` is decoded by the checkpoint validation
    # actually chose, not by whatever the last epoch happened to leave in memory.
    del model
    free_gpu()
    base = WhisperForConditionalGeneration.from_pretrained(FINETUNE_BASE)
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    base.generation_config.forced_decoder_ids = None
    base.generation_config.language = LANGUAGE
    base.generation_config.task = TASK
    model = PeftModel.from_pretrained(base, str(adapter_dir / "best")).to(device).eval()

    meta = {
        "split": None, "system": FINETUNE_NAME, "model": FINETUNE_BASE,
        "model_revision": f"lora-r{lora_rank}-seed{seed}-epoch{best['epoch']}",
        "tier": "contribution", "device": device, "compute_type": "float16",
        "beam_size": BEAM_SIZE, "language": LANGUAGE,
    }
    for split, rows in (("validation", validation), ("test", test)):
        path = predictions_dir / f"{split}__{FINETUNE_NAME}.jsonl"
        if path.exists():
            path.unlink()
        records: list[dict[str, Any]] = []
        for start in range(0, len(rows), eval_batch_size):
            chunk = rows[start:start + eval_batch_size]
            audio = [load_audio(resolve_clip(bundle, row)) for row in chunk]
            began = time.perf_counter()
            with amp(device, torch):
                texts = generate_texts(model, processor, audio, device, torch.float32)
            share = (time.perf_counter() - began) / len(chunk)
            for row, text in zip(chunk, texts):
                records.append({"clip_id": row["clip_id"], "text": text,
                                "inference_seconds": round(share, 3), **meta, "split": split})
            log(f"  {split} [{min(start + eval_batch_size, len(rows))}/{len(rows)}]")
        append_jsonl(path, records)
        log(f"Wrote {path}")

    report = {
        "base_model": FINETUNE_BASE, "system": FINETUNE_NAME, "epochs": epochs,
        "lora_rank": lora_rank, "lora_alpha": lora_rank * 2,
        "target_modules": ["q_proj", "v_proj"], "learning_rate": learning_rate,
        "batch_size": batch_size, "seed": seed, "language": LANGUAGE, "task": TASK,
        "train_utterances": len(train), "validation_utterances": len(validation),
        "test_utterances": len(test), "training_minutes": round(minutes, 2),
        "selected_epoch": best["epoch"], "selection_metric": "validation CER",
        "validation_cer_at_selected_epoch": best["cer"],
        "history": history,
    }
    (predictions_dir.parent / "finetune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    free_gpu()
    return report


# -------------------------------------------------------------------- evaluate


def evaluate(
    manifest: Path,
    predictions_dir: Path,
    reports_dir: Path,
    splits: list[str],
    environment: dict[str, Any],
    finetune_report: dict[str, Any] | None,
) -> dict[str, Any]:
    utterances = read_jsonl(manifest)
    metrics: dict[str, Any] = {
        "task": "Japanese sung-lyrics transcription",
        "metrics": {
            "primary": ["cer", "character_f1"],
            "wer_reported": False,
            "justification": (
                "Japanese is written without spaces between words, so word-level "
                "tokenization is not well defined and WER would measure the segmenter "
                "as much as the model."
            ),
            "quote": ("Certain languages, such as Mandarin and Japanese, have no notion of "
                      "'words', and so the WER is meaningless. Here, we revert to using the CER."),
            "reference": "https://huggingface.co/learn/audio-course/chapter5/evaluation",
        },
        "decode_settings": {"language": LANGUAGE, "task": TASK, "beam_size": BEAM_SIZE,
                            "vad_filter": False, "condition_on_previous_text": False},
        "environment": environment,
        "finetune": finetune_report,
        "splits": {},
    }
    for split in splits:
        expected = {str(row["clip_id"]) for row in utterances if str(row.get("split")) == split}
        if not expected:
            continue
        arguments: list[str] = []
        for path in sorted(predictions_dir.glob(f"{split}__*.jsonl")):
            name = path.name[len(split) + 2:-len(".jsonl")]
            covered = {str(row.get("clip_id")) for row in read_jsonl(path)}
            if covered != expected:
                log(f"!! skipping {name} on {split}: covers {len(covered)}/{len(expected)} clips "
                    f"(rerun --stage baselines to finish it)")
                continue
            arguments.append(f"{name}={path}")
        if not arguments:
            log(f"!! no complete prediction files for {split}")
            continue
        log(f"\n=== scoring {split}: {len(arguments)} systems ===")
        rows = scorer.evaluate_japanese(
            manifest, arguments, reports_dir / f"ja_{split}_benchmark", split)
        metrics["splits"][split] = rows
    return metrics


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=Path("."),
                        help="Unzipped colab_bundle directory (holds ja_utterances.jsonl + clips/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write predictions/reports (default: --bundle)")
    parser.add_argument("--stage", default="all",
                        choices=("all", "baselines", "finetune", "evaluate"))
    parser.add_argument("--splits", default="validation,test",
                        help="Comma-separated splits to transcribe and score")
    parser.add_argument("--systems", default=",".join(system["name"] for system in SYSTEMS),
                        help="Comma-separated pretrained systems to run")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8, help="Batch for transformers inference")
    parser.add_argument("--install", action="store_true", help="pip install dependencies first")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dependencies(args.install)
    bundle = args.bundle.resolve()
    output = (args.output or bundle).resolve()
    manifest = bundle / "ja_utterances.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found. Unzip colab_bundle.zip and pass --bundle <dir>.")

    utterances = read_jsonl(manifest)
    foreign = sorted({str(row.get("language", "und")) for row in utterances} - {"ja"})
    if foreign:
        raise SystemExit(f"This study is Japanese-only but the manifest contains {foreign}")
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    names = [part.strip() for part in args.systems.split(",") if part.strip()]
    predictions_dir = output / "final_predictions"
    reports_dir = output / "reports"
    counts = {split: sum(1 for row in utterances if str(row.get("split")) == split)
              for split in ("train", "validation", "test")}
    log(f"Bundle: {bundle}")
    log(f"Utterances: {len(utterances)} total "
        + " ".join(f"{split}={count}" for split, count in counts.items()))

    device, compute_type, environment = describe_device(args.device)
    started = time.perf_counter()

    if args.stage in ("all", "baselines"):
        run_baselines(bundle, utterances, splits, names, predictions_dir,
                      device, compute_type, args.batch_size)

    finetune_report = None
    report_path = output / "finetune_report.json"
    if args.stage in ("all", "finetune"):
        finetune_report = finetune(
            bundle, utterances, predictions_dir, device, args.epochs,
            args.train_batch_size, args.eval_batch_size, args.learning_rate,
            args.lora_rank, args.seed, output / "lora_adapter")
    elif report_path.exists():
        finetune_report = json.loads(report_path.read_text(encoding="utf-8"))

    if args.stage in ("all", "evaluate"):
        if finetune_report is None and report_path.exists():
            finetune_report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = evaluate(manifest, predictions_dir, reports_dir, splits,
                           environment, finetune_report)
        metrics["wall_clock_minutes"] = round((time.perf_counter() - started) / 60, 2)
        destination = output / "final_metrics.json"
        destination.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\nWrote {destination}")
        log(f"Markdown tables in {reports_dir}")
        log("Paste reports/ja_test_benchmark_table.md into the README under '## Results'.")

    log(f"\nTotal wall clock: {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
