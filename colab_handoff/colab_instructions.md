# Running the Japanese sung-lyrics benchmark on a free Colab T4

Everything in this folder exists for one reason: laptop-CPU `int8` inference over
six ASR systems is several hours of wall clock, and the same work on a free Colab
T4 at `float16` is well under an hour. The LoRA fine-tune of `openai/whisper-small`
is minutes on a T4 and effectively impossible on CPU.

You do **not** need to clone the repository on Colab. `colab_bundle.zip` is the
only input.

---

## What you get at the end

| File | Contents |
|---|---|
| `final_metrics.json` | Every number, per split, per system — plus the environment, decode settings, and the fine-tune report |
| `final_predictions/<split>__<system>.jsonl` | One transcript per utterance per system |
| `reports/ja_test_benchmark_table.md` | Paste-ready Markdown table for the README `## Results` section |
| `reports/ja_test_benchmark_per_song.csv` | Per-song breakdown (a single hard song can dominate a 15-song split) |
| `reports/ja_test_benchmark_per_speaker.csv` | Per-speaker-group breakdown |
| `finetune_report.json` | Epoch-by-epoch validation CER, selected epoch, hyperparameters, training minutes |
| `lora_adapter/best/` | The selected LoRA adapter (~14 MB) |

---

## Systems that get scored

Seven rows, all decoding with `language="ja"`, `task="transcribe"`, `num_beams=5`,
no VAD, no cross-utterance conditioning. The checkpoint is the only variable.

| Tier | System | Checkpoint | Revision | Params | Backend |
|---|---|---|---|---|---|
| baseline | `faster-whisper-base` | [`Systran/faster-whisper-base`](https://huggingface.co/Systran/faster-whisper-base) | `ebe41f70` | 74M | CTranslate2 |
| baseline | `faster-whisper-small` | [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small) | `536b0662` | 244M | CTranslate2 |
| baseline | `faster-whisper-medium` | [`Systran/faster-whisper-medium`](https://huggingface.co/Systran/faster-whisper-medium) | `08e178d4` | 769M | CTranslate2 |
| baseline | `faster-whisper-large-v3-turbo` | [`deepdml/faster-whisper-large-v3-turbo-ct2`](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | `4df90f75` | 809M | CTranslate2 |
| japanese-pretrained | `kotoba-whisper-v2.0` | [`kotoba-tech/kotoba-whisper-v2.0-faster`](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) | `f44edd35` | 756M | CTranslate2 |
| japanese-pretrained | `anime-whisper` | [`litagin/anime-whisper`](https://huggingface.co/litagin/anime-whisper) | `22e2008a` | 756M | transformers |
| contribution | `whisper-small-lora-ja` | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) + LoRA | trained here | 244M + 7M adapter | transformers + PEFT |

The two Japanese-specialised rows are *already trained* checkpoints, not something
this project fit: `kotoba-whisper-v2.0` is distilled from `whisper-large-v3` on the
ReazonSpeech Japanese corpus, and `anime-whisper` is `kotoba-whisper-v2.0`
fine-tuned further on ~5,300 h of expressive Japanese speech. They are in the table
to answer the obvious question — *does a Japanese-only checkpoint beat a bigger
multilingual one on sung Japanese?* — without conflating it with our own training run.

### Excluded on purpose

| Checkpoint | Why excluded |
|---|---|
| [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | Model card declares exactly 25 European languages and never mentions Japanese |
| [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Same 25-language European list; no Japanese support |

Both would have to be decoded under a language they do not declare, so they are
absent from the runner rather than scored and asterisked.

---

## Metrics: CER and Character F1, no WER

From the Hugging Face Audio Course, Chapter 5, *Evaluation metrics for ASR*
(<https://huggingface.co/learn/audio-course/chapter5/evaluation>):

> Certain languages, such as Mandarin and Japanese, have no notion of 'words', and
> so the WER is meaningless. Here, we revert to using the CER.

So: **CER** (unbounded error rate, lower is better) plus **Character F1** (bounded
in [0,1], charges hallucinated text to precision instead of folding it into one
rate). `ja_eval_standalone.py` is a byte-for-byte reimplementation of the
repository's scorer with zero project imports, so Colab numbers and laptop numbers
are directly comparable.

---

## Setup

### 1. Turn on the GPU — do this first

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

If you skip this, the runner warns loudly and falls back to CPU, which will take
hours. It will not silently pretend.

### 2. Get the bundle onto the VM

**Option A — direct upload** (simplest; the bundle is tens of MB):

```python
from google.colab import files
files.upload()          # pick colab_bundle.zip
```

**Option B — Google Drive** (survives disconnects, which matters for a long run):

```python
from google.colab import drive
drive.mount('/content/drive')
# then copy colab_bundle.zip into /content/drive/MyDrive/ beforehand
!cp /content/drive/MyDrive/colab_bundle.zip /content/
```

### 3. Unzip and install

```python
!cd /content && unzip -q -o colab_bundle.zip -d bundle && ls bundle
!pip install -q "faster-whisper==1.2.1" "transformers>=4.44,<5" "peft>=0.13" "accelerate>=0.34" soundfile av
```

### 4. Run everything

```python
!cd /content/bundle && python run_colab.py --stage all --device cuda
```

### 5. Download the results

```python
!cd /content/bundle && zip -qr /content/results.zip final_metrics.json finetune_report.json final_predictions reports lora_adapter
from google.colab import files
files.download('/content/results.zip')
```

---

## Expected wall clock on a T4

| Stage | Time |
|---|---|
| `pip install` on a cold VM | 2–4 min |
| Model downloads (~4 GB across six checkpoints) | 5–10 min |
| `baselines` on `validation` + `test` (250 clips × 6 systems) | 20–35 min |
| `finetune` (feature extraction + 4 epochs + 4 validation decodes + test decode) | 10–15 min |
| `evaluate` | < 1 min |

Free-tier sessions get reclaimed. Every stage is resumable: prediction files are
append-only and the runner skips clips it has already transcribed, so rerunning the
exact same command picks up where it stopped. If you are worried about losing a
long run, write the output to Drive:

```python
!cd /content/bundle && python run_colab.py --stage all --device cuda --output /content/drive/MyDrive/ja_asr_out
```

---

## Running the stages separately

```python
# just the pretrained systems
!cd /content/bundle && python run_colab.py --stage baselines --device cuda

# just one or two systems (names from the table above)
!cd /content/bundle && python run_colab.py --stage baselines --systems kotoba-whisper-v2.0,anime-whisper

# just the LoRA fine-tune
!cd /content/bundle && python run_colab.py --stage finetune --device cuda --epochs 4

# rescore whatever predictions already exist
!cd /content/bundle && python run_colab.py --stage evaluate
```

Useful flags: `--splits`, `--batch-size`, `--epochs`, `--train-batch-size`,
`--eval-batch-size`, `--learning-rate`, `--lora-rank`, `--seed`.
`python run_colab.py --help` lists all of them.

---

## Fine-tuning configuration

| Setting | Value | Why |
|---|---|---|
| Base | `openai/whisper-small` | 244M; the largest checkpoint that trains comfortably in T4 memory |
| Adapter | LoRA `r=32`, `alpha=64`, `dropout=0.05` on `q_proj` + `v_proj` | ~7M trainable parameters — minutes, not hours, and the adapter is small enough to version |
| Language / task | `language="ja"`, `task="transcribe"` | Pinned; never language-detected |
| Epochs | 4 | The train split is ~400 utterances; more epochs overfit 11 speaker groups |
| Optimizer | AdamW, lr `5e-5`, linear warmup 10% then linear decay, grad-clip 1.0 | Train loss decreased steadily across all 4 epochs (3.34 → 2.11 → 1.84 → 1.74) at this rate; see `finetune_report.json` for the full history |
| Precision | fp16 autocast with `GradScaler`, fp32 master weights | Stable on T4; pure fp16 Whisper training overflows |
| Seed | 42 | Same seed as the speaker-disjoint split |
| Selection | lowest `validation` CER, evaluated after every epoch | `test` is decoded **once**, by the adapter validation chose |

On the run archived in this repo, the adapter improved CER and Character F1 on
`validation` (its own selection split) but was slightly worse than the unmodified
base on `test`, with *lower* recall than the base model on both splits — see
[The contribution, honestly](../README.md#the-contribution-honestly) in the
README. 2.7 h over 11 train speakers was evidently enough to shift the model
but not enough to generalize past the validation speaker groups. Do not assume
a rerun will reproduce this exact direction; report whatever `test` actually
shows.

---

## Troubleshooting

**`no GPU is visible`** — Runtime → Change runtime type → T4 GPU, then rerun. The
runner continues on CPU but says so.

**`CUDA out of memory` during `baselines`** — lower `--batch-size` (default 8;
`anime-whisper` is the only batched system).

**`CUDA out of memory` during `finetune`** — lower `--train-batch-size` to 4 and
`--eval-batch-size` to 4.

**`clip for … not found`** — `--bundle` is not pointing at the unzipped directory.
It must contain `ja_utterances.jsonl` and a `clips/` folder.

**`cuDNN`/`cuBLAS` error from `faster-whisper`** — CTranslate2 needs matching CUDA
libraries. On Colab this usually fixes it:

```python
!pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
import os; os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:' + os.popen(
    "python -c \"import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')\"").read().strip()
```

Then restart the runtime and rerun. Failing that, `--device cpu` for the
CTranslate2 rows still works; it is just slow.

**Session died mid-run** — rerun the identical command. Append-only prediction
files mean you lose at most one clip.

---

## Audio in the bundle

The cut clips are 16 kHz mono. The bundle ships them as **Opus 48 kbit/s mono** by
default, because 16 kHz PCM is 256 kbit/s and a lossless bundle is hundreds of
megabytes. At a 16 kHz sampling rate the signal only occupies 8 kHz of bandwidth,
which Opus encodes essentially transparently at 48 kbit/s. Every system reads the
same encoded clips, so no row is advantaged.

If you would rather benchmark untouched audio, rebuild the bundle losslessly on the
machine that holds the repository:

```powershell
lyrics-asr-package-colab --codec flac    # lossless, ~5x larger
lyrics-asr-package-colab --codec wav     # byte-identical to the local clips
```

Full-length source songs are deliberately **not** in the bundle. Only the cut
utterance windows travel.

---

## After the run

1. Open `reports/ja_test_benchmark_table.md` and paste it into `README.md` under
   `## Results`, replacing the placeholder table.
2. Copy the selected epoch and validation CER from `finetune_report.json` into the
   contribution paragraph.
3. `final_metrics.json` is the archival record — keep it next to the README so
   every published number has a traceable source.
