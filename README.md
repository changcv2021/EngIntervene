# EngIntervene: Benchmarking Multimodal Engineering State Understanding and Design Intervention Reasoning

## Dataset on Hugging Face

**The EngIntervene dataset is available at the anonymous Hugging Face repository:**

### [benchmarkanon/EngIntervene](https://huggingface.co/datasets/benchmarkanon/EngIntervene)

This is the download and access location for the complete **3,229-item benchmark
input dataset**, covering **seven industrial domains and tasks T1–T4**. It includes
the questions, options where applicable, response requirements and input images.
The Hugging Face `test` split contains the full benchmark, not only the SFT
experiment's held-out subset.

This GitHub repository provides the **training and evaluation code**; download
the dataset from the Hugging Face link above. Gold reference answers and scoring
rubrics are kept separately and are not included in that input release.


## Experiment code

Official experiment code for EngIntervene. This repository reuses the validated
code supplement; publishing updates its title, publication note and file hashes,
not the training, inference or scoring algorithms.

Code supplement for multimodal engineering understanding, relation modeling,
diagnosis and design-change reasoning. This package contains **no author names,
affiliations, institutional endpoints, credentials, dataset answers, model
weights, historical job logs or Git history**.

## Scope

- Full-benchmark and held-out local-model inference.
- The completed single-seed **Format-SFT** and **Industry-SFT** procedures for
  Qwen3.5-4B, InternVL3.5-8B-HF and Qwen2.5-VL-7B-Instruct.
- The exact frozen source-grouped split and task-balanced sampling IDs.
- T1 deterministic choice scoring and separate Qwen3.5-4B T2–T4 rubric judging.
- Both T4 **Atomic Rubric Score** and **Strict Revision Success**.
- Task/domain summaries, held-out Challenge evaluation, and portable full /
  text-only / shuffled-image ablation wrappers.

The training loop, label masks, optimizer schedule, answer targets, task prompts,
T1 parser and judge algorithms are carried over from the completed experiments.
Portable data/CLI/orchestration wrappers are newly organized for release; this
packaging is **not a new six-run GPU reproduction**. See `VALIDATION.md` for checks
actually performed and `CODE_PROVENANCE.json` for original algorithm hashes.

This is not the entire private experiment workspace. Site-specific dispatchers,
mail notification code, old failed/retry pipelines, cloud account deployments,
dataset collection tools and trained checkpoints are intentionally excluded.

## 1. Environment

Python 3.12 was used. Create your own environment and install:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The historical core was PyTorch 2.8.0, PEFT 0.20.0 and Transformers
5.16.0.dev0 at the exact Git commit pinned in `requirements.txt`. Choose a
PyTorch CUDA build compatible with your hardware. Secondary dependencies are
not a bit-for-bit historical environment lock. No installation or model download
is triggered by importing this package.

Place downloaded model snapshots in the paths in `configs/training.json`, or
edit those paths. The runtime uses `local_files_only=True`; obtain model files
and accept any upstream terms separately. One CUDA device must be visible per
training/inference/judge process. Parallelize independent models or inference
shards across GPUs. Never run GPU work on an HPC login node.

## 2. Inputs and scoring references are separate

Reuse the existing [EngIntervene dataset](https://huggingface.co/datasets/benchmarkanon/EngIntervene).
The input snapshot checked during packaging is commit
`0c914706e4f7d08a109ec1817eebbddd2080337c`. Download/access it separately,
then point the loader at its local directory. This code supplement neither
repackages the dataset nor changes its visibility, contents or license.

The input loader accepts either:

1. The local EngIntervene Hugging Face-format directory, with
   `data/test-*.parquet` and embedded original image bytes; or
2. The original release schema with `internal_items.jsonl`, `assets.jsonl`
   and referenced images. Use `--asset-root` for images outside the release.

The public input dataset alone **cannot** supply SFT targets or grading golds.
Obtain the corresponding scoring-side `labels.jsonl` and `rubrics.jsonl`
separately, then keep them outside the public code/input directory.
The original equivalents are `private_labels.jsonl` / `private_rubrics.jsonl`.
If you already hold the anonymous data delivery, use its existing
`private_scoring/labels.jsonl` and `private_scoring/rubrics.jsonl`; do not
regenerate answers or upload that directory with this code. The dataset's
single Hub `test` split contains all 3,229 items, whereas this code's adaptation
Test split contains the frozen 1,306 held-out items.
No repository access token is embedded; access permissions remain your own.

```bash
python -m engintervene.data --dataset data/EngIntervene --output work/inputs
```

This writes input-only `items.jsonl` and, for the matching full release,
`train.jsonl`, `validation.jsonl`, `test.jsonl`. It preserves question wording,
all image references, option IDs and response requirements. Auxiliary context,
source excerpts and gold references are not silently appended to prompts.

## 3. Prepare and train SFT

```bash
python -m engintervene.prepare_sft \
  --items work/inputs/items.jsonl \
  --labels private_scoring/labels.jsonl \
  --output work/data/internal_grouped_v1/sft_train_validation_only

# model-index: 0=Qwen3.5-4B, 1=InternVL3.5-8B-HF, 2=Qwen2.5-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python -m engintervene.train \
  --model-index 0 --condition format --mode train
CUDA_VISIBLE_DEVICES=0 python -m engintervene.train \
  --model-index 0 --condition industry --mode train
```

Repeat the two conditions for indices 1 and 2. These are **six independent
single-seed runs**, each starting from its base model. Do not train Industry-SFT
on top of a Format-SFT adapter. `--mode smoke` is a separate diagnostic run;
its checkpoint is never used to initialize formal training.

Frozen split: Train **1,615**, Validation **308**, Test **1,306**, based on
**227 connected source groups**. Source/image isolation takes precedence over
matching 50/10/40 exactly. Some domain × task cells are absent from held-out
Test; report them as N/A. The split files reproduce the existing assignment;
this package does not invent a new grouping algorithm or item-random split.

Shared parameters: seed42, 2 epochs, microbatch1, accumulation16, 202 optimizer
updates, AdamW lr1e-4/weight-decay0.01, six-step warmup factor multiplied by
cosine decay, gradient-norm clipping1.0, LoRA r16/alpha32/dropout0.05,
BF16 base weights, language-layer adapters only, no 4-/8-bit quantization.
Image area is constrained to 65,536–1,048,576 pixels before the processor.
Maximum training sequence length is 32,768; overlong inputs fail without silent
truncation. Thinking is disabled.

Each epoch draws 1,615 samples **with replacement**: T1/T2/T3 each404 and T4
403. Thus two epochs do not mean that every training item was seen twice.
Format-SFT supervises only JSON wrappers and turn endings; answer-content
positions are masked. Industry-SFT supervises full frozen assistant references.
**Format-SFT still exposes gold answer content as teacher-forcing context. It
is a structural-loss control, not a knowledge-free training condition.**

Evaluate validation token-weighted NLL after each epoch and select its minimum.
There is no early stopping in these completed two-epoch runs. All three models
selected epoch2 for Format-SFT and epoch1 for Industry-SFT. No Test scores or
answers enter checkpoint selection. Complete historical module targets and
trainable parameter counts are in `configs/historical_run_parameters.json`.

## 4. Evaluate a base or adapted model

```bash
# SFT before/after comparison: the same held-out 1,306 items for all conditions.
CUDA_VISIBLE_DEVICES=0 python -m engintervene.inference \
  --items work/inputs/test.jsonl --model-key qwen35_4b --condition base \
  --output work/base_generations.jsonl
CUDA_VISIBLE_DEVICES=0 python -m engintervene.inference \
  --items work/inputs/test.jsonl --model-key qwen35_4b --condition industry \
  --output work/industry_generations.jsonl
```

`format` is also supported. Adapted runs load `best_checkpoint.json` from a
successfully completed training directory, never select on Test. The `--adapter-run`
argument accommodates moved checkpoint directories; update runtime checkpoint
paths after moving them. For **full-benchmark base-model evaluation**, replace
`test.jsonl` with `items.jsonl`; do not call adapted full-data scores held-out.

Use `--num-shards N --shard-index i --output work/run_i.jsonl` for independent
GPU workers. Every worker needs a distinct output path. Each answer is generated
once, with a hard limit of 256 tokens for T1 and 4096 for T2–T4. Sampling uses
temperature 0.7, top-p 0.8, top-k 20, with matched per-item seeds across conditions.
The actual capped output is preserved and graded; it is not automatically
replaced by zero solely because it reached the token cap.

For image ablations add `--visual-mode text-only` or `--visual-mode shuffled-image`.
Text-only sends **no images and no added image descriptions**, without changing
the dataset. Shuffled-image replaces the image bundle with a deterministic donor
from the same task/domain, with no shared image byte hashes. It fails explicitly
if no valid donor exists. This is a portable ablation implementation; it does not
claim to reproduce any unbundled historical donor mapping or other-model backend.

## 5. Score and summarize

```bash
CUDA_VISIBLE_DEVICES=0 python -m engintervene.score \
  --items work/inputs/test.jsonl \
  --labels private_scoring/labels.jsonl --rubrics private_scoring/rubrics.jsonl \
  --generations work/industry_generations.jsonl \
  --judge-model models/Qwen3.5-4B --output work/industry_scores.jsonl

python -m engintervene.summarize \
  --items work/inputs/test.jsonl --scores work/industry_scores.jsonl \
  --output work/industry_summary.json
```

Both scorer and inference support independent shards. `--generations` and
`--scores` accept multiple JSONL files. Missing/duplicate IDs and infrastructure
errors block aggregation. An absent task is N/A, not zero. The summarizer writes
JSON plus Markdown, with task/domain cells and held-out Challenge statistics.

T1 uses the historical Unicode-normalized choice-set parser. T2–T4 use an
**unadapted, separate Qwen3.5-4B** as a text-only per-criterion judge, not the
candidate's SFT adapter. It is greedy, non-thinking, with up to three judgment
attempts (160/128/128 tokens). Positive evidence must be a candidate-answer span.
The historical evidence check removes whitespace and case-folds; it is not a
byte-identical comparison. Invalid judgments finally fall back to `met=false`
and are counted. This does not constitute expert adjudication or calibration.

Atomic Score = clip(weighted required credit − critical-error penalties,
0, score cap) / score cap. Strict Success requires all specified required items
and no forbidden error. Omitting a correct condition is not itself an explicitly
endorsed critical error. Existing rubric contents/limitations are not repaired
or relabeled by this code release. Report both T4 metrics.

`Industry-SFT → Challenge` uses the **249 original Challenge items in held-out
Test**, not a third training run and not all original Challenge items.

## External/API model outputs

The same scorer can grade API outputs prepared separately. Supply JSONL records
with unique `item_id`, `response` (the actual emitted text) and `error: null`,
covering exactly the IDs in `--items`. Retain raw responses, token usage, model
version and provider protocol in your private experiment logs. Institutional
cloud endpoints, Batch submission scripts, authentication files and deployment
names are **not bundled**, and this package makes no calls to paid services.
New API results require their own inference-protocol description.

## HPC, anonymity and release

`scripts/run_job.sbatch` is a scheduler-neutral command wrapper. Supply your
own verified partition/account/resources/log path; it does not submit anything
by itself. Do not include your edited local paths, generated work directories,
credentials or logs in a paper attachment. Use the audited ZIP, not a zip of
your working directory. No automatic email, upload or external telemetry runs.

See `ANONYMITY.md`, `VALIDATION.md`, `NOTICE.md` and `docs/README_ZH.md`.
The code is released under the **MIT License**, with anonymous contributor
attribution; see `LICENSE.txt`. Third-party dependencies, datasets and pretrained
models retain their respective licenses and are not relicensed by this package.
