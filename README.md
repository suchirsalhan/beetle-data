# Beetle-Data

Modular large-scale preprocessing library for streaming, decontaminating, scoring, and pretokenizing multilingual training data. Produces Arrow datasets compatible with the [BeetleLM](../beetlelm) training framework. Supports both **static mode** (original 3-stage pipeline) and **curriculum mode** (BeetleStream v2, 7-stage pipeline with quality scoring, topic clustering, and difficulty grading). Also supports streaming held-out evaluation data for [learning curve analysis](../beetle-analyze/aoa/).

## Overview

Beetle-Data prepares 24B-token bilingual (L1 + English) training corpora for 125M-parameter language models following Chinchilla scaling laws. The pipeline streams data from FineWeb-Edu (English) and FineWeb-2 (non-English), removes evaluation benchmark contamination via 13-gram overlap detection, and pretokenizes using pre-trained bilingual BPE tokenizers.

**BeetleStream v2** extends the original pipeline with four new curriculum stages (A, B, C, D) that add pedagogical quality scoring, topic clustering, and difficulty grading. A teacher LLM (Llama-3-70B) annotates a sample of documents; a lightweight student model distills those annotations to score the full corpus; and an indexing pass partitions every document by topic, quality, and difficulty for curriculum-aware training.

### Architecture

```
 STATIC MODE (original 3-stage pipeline)
 ═════════════════════════════════════════

 Stage 1                   Stage 2                         Stage 3
 Benchmark Index     -->   Streaming Decontamination  -->  Pretokenization
 (one-time, ~5 min)        (~5 hrs, 4 nodes)               (~2 hrs, 4 nodes)

 HF benchmarks             FineWeb-2 / FineWeb-Edu         Clean Parquet
     |                          |                               |
     v                          v                               v
 13-gram index (.pkl)      Clean Parquet shards            Arrow datasets
                           + manifest.json                 (beetlelm-compatible)


 CURRICULUM MODE (BeetleStream v2, 7-stage pipeline)
 ═══════════════════════════════════════════════════════

 Stage 1       Stage 2       Stage A        Stage B         Stage C
 Benchmark --> Streaming --> Teacher     --> Feature     --> Student
 Index         Decontam.     Annotation     Transform       Model
 (5 min)       (5 hrs)       (7 hrs)        (10 min)        (10 min)
     |              |             |               |               |
     v              v             v               v               v
 13-gram        Clean         Annotation     Feature         Trained
 index          Parquet       JSONL          Parquet         sklearn
 (.pkl)         shards        (rubric)       (labels)        model


                Stage D              Stage 3
            --> Score + Index    --> Pretokenize
                (10 hrs)             (2 hrs)
                    |                    |
                    v                    v
                Indexed              Curriculum Arrow
                Parquet shards       (quality / difficulty /
                (topic-partitioned)   topic_id columns)


 Stage 4 (optional, both modes)
 Held-Out Streaming
 (~10 min per language)

 FineWeb-2 (beyond training cutoff)
     |
     v
 Held-out Parquet shards (for learning curve eval)
```

Stages are decoupled so you can re-run pretokenization when a tokenizer changes without re-streaming 560B tokens, re-run decontamination against new benchmarks without re-tokenizing, or re-run scoring without re-annotating.

## Languages (38 + English)

| Group | Languages |
|-------|-----------|
| Core (20) | Polish (pl), Dutch (nl), Spanish (es), Greek (el), Japanese (ja), French (fr), Chinese (zh), German (de), Italian (it), Basque (eu), Turkish (tr), Indonesian (id), Tagalog (tl), Persian (fa), Hindi (hi), Tamil (ta), Swedish (sv), Russian (ru), Catalan (ca), Arabic (ar) |
| Extension (8) | Urdu (ur), Bengali (bn), Czech (cs), Gujarati (gu), Thai (th), Vietnamese (vi), Korean (ko), Danish (da) |
| Low-Resource European (5) | Hungarian (hu), Bulgarian (bg), Croatian (hr), Ukrainian (uk), Slovenian (sl) |
| Low-Resource African (4) | Somali (so), Amharic (am), Yoruba (yo), Wolof (wo) |
| Bilingual partner | English (en) via FineWeb-Edu |

Each non-English language is paired with English using a bilingual tokenizer from HuggingFace (`Beetle-Data/tokenizer-{lang}-en`).

## Quick Start

### Prerequisites

```bash
python3 -m venv venvs/demo; source venvs/demo/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your-token>
```

### Static Mode (Original Pipeline)

The original 3-stage pipeline. Processes each language sequentially: decontaminate, pretokenize, upload to HuggingFace, then delete local files. Peak disk usage stays under 300 GB.

```bash
cd beetle-data

# All 19 languages (default) -- uploads each to HF, cleans up locally
bash scripts/launch_full_pipeline.sh

# Specific languages only
bash scripts/launch_full_pipeline.sh --lang fr de es zh ja
bash scripts/launch_full_pipeline.sh --lang nl it ru pl tr
bash scripts/launch_full_pipeline.sh --lang tl hi ta eu ar
bash scripts/launch_full_pipeline.sh --lang sv el ca fa id

# Custom output directory
OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_full_pipeline.sh

# Without HF upload (keep local files -- needs ~3.2 TB disk)
bash scripts/launch_full_pipeline.sh --no-upload

# Extension languages (separate pipeline)
bash scripts/launch_extensions.sh

# Low-resource languages
bash scripts/launch_low_resource.sh
```

### Curriculum Mode (BeetleStream v2)

The full 7-stage pedagogical pipeline. Adds quality scoring, topic clustering, and difficulty grading on top of the static pipeline. Requires GPUs for teacher annotation (Stage A) and student model scoring (Stage D).

```bash
cd beetle-data

# Default: full curriculum pipeline, all core languages
bash scripts/launch_beetlestream.sh

# Specific languages
bash scripts/launch_beetlestream.sh --lang fr de es zh ja

# Custom output directory
OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_beetlestream.sh

# Use the smaller 8B teacher model (faster, lower annotation quality)
bash scripts/launch_beetlestream.sh --teacher-model meta-llama/Meta-Llama-3-8B-Instruct
```

## Curriculum Mode: Execution Sequence

The full BeetleStream v2 pipeline runs in 8 steps. The default launch script (`bash scripts/launch_beetlestream.sh`) executes all of them, but each step can also be run individually.

### Step 1: Build Benchmark Index (5 min, 1 node)

Builds the 13-gram index from all evaluation benchmarks. Run once; shared across all languages and both pipelines.

```bash
export OUTPUT_DIR="${OUTPUT_DIR:-pipeline_output}"

python -m pipeline.run_pipeline --stage 1 \
    --project-root . --output-dir "$OUTPUT_DIR" --skip-disk-check
```

### Step 2: Decontaminate (5 hrs, 4 nodes)

Streams ~28B tokens per language from FineWeb-2/FineWeb-Edu, discards any document containing a 13-gram overlap with evaluation benchmarks, and writes clean Parquet shards.

```bash
# Node 0 (also decontaminates English):
python -m pipeline.run_pipeline --stage 2 --node-id 0 \
    --index "$OUTPUT_DIR/benchmark_13gram.pkl" \
    --output-dir "$OUTPUT_DIR" --project-root . --skip-disk-check

# Nodes 1-3:
python -m pipeline.run_pipeline --stage 2 --node-id 1 \
    --index "$OUTPUT_DIR/benchmark_13gram.pkl" \
    --output-dir "$OUTPUT_DIR" --project-root . --skip-disk-check

python -m pipeline.run_pipeline --stage 2 --node-id 2 \
    --index "$OUTPUT_DIR/benchmark_13gram.pkl" \
    --output-dir "$OUTPUT_DIR" --project-root . --skip-disk-check

python -m pipeline.run_pipeline --stage 2 --node-id 3 \
    --index "$OUTPUT_DIR/benchmark_13gram.pkl" \
    --output-dir "$OUTPUT_DIR" --project-root . --skip-disk-check
```

### Step 3: Teacher Annotation (7 hrs, Node 0, 8 GPUs via vLLM)

Launches a vLLM server with tensor parallelism across 8 A100 GPUs, reservoir-samples 500K documents across all languages, and annotates them with a pedagogical rubric (quality, difficulty, vocabulary, engagement, topic). Calibration examples are loaded from KidLM-corpus and CLC-L1-CEFR on HuggingFace.

```bash
# Launch vLLM in the background (Node 0 only)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --port 8000 &

# Wait for server readiness, then run Stage A
python -m pipeline.run_pipeline --stage A \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config configs/beetlestream_curriculum.yaml \
    --teacher-model meta-llama/Meta-Llama-3-70B-Instruct \
    --vllm-url http://localhost:8000/v1

# Stop vLLM after annotation completes
kill %1
```

### Step 4: Feature Transform (10 min, CPU)

Converts teacher annotation JSONL into structured feature Parquet files. Extracts numeric quality scores (0-5), difficulty levels (1-3), vocabulary complexity flags, engagement flags, and topic labels.

```bash
python -m pipeline.run_pipeline --stage B --output-dir "$OUTPUT_DIR"
```

### Step 5: Student Model (10 min, 1 GPU)

Trains lightweight sklearn regressors and classifiers on frozen multilingual-e5-base embeddings (768-dim) to approximate the teacher LLM's pedagogical scores. Outputs a trained student model for Stage D scoring.

```bash
python -m pipeline.run_pipeline --stage C \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config configs/beetlestream_curriculum.yaml
```

### Step 6: Score + Index (10 hrs, 4 nodes, 8 GPUs each)

Applies heuristic filters (stopword density, readability, script consistency, repetition), then embeds surviving documents with the student model, predicts quality/difficulty, clusters into 200 topics via k-means, and writes Hive-partitioned indexed Parquet shards.

```bash
# Node 0:
python -m pipeline.run_pipeline --stage D --node-id 0 \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config configs/beetlestream_curriculum.yaml

# Node 1:
python -m pipeline.run_pipeline --stage D --node-id 1 \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config configs/beetlestream_curriculum.yaml

# Node 2:
python -m pipeline.run_pipeline --stage D --node-id 2 \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config configs/beetlestream_curriculum.yaml

# Node 3:
python -m pipeline.run_pipeline --stage D --node-id 3 \
    --output-dir "$OUTPUT_DIR" \
    --beetlestream-config configs/beetlestream_curriculum.yaml
```

### Step 7: Pretokenize (2 hrs, 4 nodes)

Tokenizes indexed shards into Arrow datasets with 513-token chunks (512 + 1). In curriculum mode, the output Arrow includes extra columns (quality, difficulty, topic_id) for curriculum-aware data loading.

```bash
# Node 0:
python -m pipeline.run_pipeline --stage 3 --node-id 0 \
    --output-dir "$OUTPUT_DIR" --stream-mode curriculum

# Node 1:
python -m pipeline.run_pipeline --stage 3 --node-id 1 \
    --output-dir "$OUTPUT_DIR" --stream-mode curriculum

# Node 2:
python -m pipeline.run_pipeline --stage 3 --node-id 2 \
    --output-dir "$OUTPUT_DIR" --stream-mode curriculum

# Node 3:
python -m pipeline.run_pipeline --stage 3 --node-id 3 \
    --output-dir "$OUTPUT_DIR" --stream-mode curriculum
```

### Step 8: Upload + Cleanup

After pretokenization completes, indexed shards and curriculum Arrow datasets are uploaded to HuggingFace and local files are deleted. This is handled automatically by the pipeline when `upload_to_hf: true` (the default). To run the full pipeline end-to-end including upload and cleanup:

```bash
bash scripts/launch_beetlestream.sh
```

## Compute Budget

Estimated wall-clock time and GPU usage for the full curriculum pipeline across 20 core languages.

| Step | Stage | What | Wall-clock | A100-hrs | Nodes |
|------|-------|------|-----------|----------|-------|
| 1 | 1 | Build benchmark index | 5 min | 0 | 1 |
| 2 | 2 | Stream + decontaminate | 5 hrs | 0 | 4 |
| 3 | A | Teacher annotation (vLLM) | 7 hrs | 56 | 1 (8 GPUs) |
| 4 | B | Feature transform | 10 min | 0 | 1 (CPU) |
| 5 | C | Student model training | 10 min | ~0.2 | 1 (1 GPU) |
| 6 | D | Score + index | 10 hrs | ~180 | 4 (8 GPUs each) |
| 7 | 3 | Pretokenize | 2 hrs | 0 | 4 |
| 8 | -- | Upload + cleanup | ~30 min | 0 | 4 |
| **Total** | | | **~25 hrs** | **~240** | **4 nodes** |

Wall-clock is dominated by the sequential dependency between Steps 3-5 (teacher annotation, feature transform, student model) on Node 0 and Step 6 (scoring) across all nodes. Steps 2 and 7 run in parallel across 4 nodes.

**Static mode** (Stages 1, 2, 3 only): ~7 hours wall-clock, 0 A100-hours.

### Hardware

- 4 nodes x 8 A100-80GB GPUs each
- 128 CPUs per node (Intel Xeon Platinum 8358 @ 2.60 GHz)
- ~512 GB RAM per node

## Disk Budget

| Phase | Cumulative on disk | Notes |
|-------|-------------------|-------|
| After Step 2 (decontamination) | ~1.47 TB | Clean Parquet shards for 20 languages + English |
| After Step 6 (score + index) | ~2.5 TB | Indexed Parquet + annotations + student model |
| Peak (Step 7, pretokenization) | ~3.8 TB | Indexed shards + curriculum Arrow being built |
| After cleanup (Step 8) | ~0 | All uploaded to HuggingFace, local files deleted |

**Static mode** peak: ~270 GB (storage-optimized, processes one language at a time). Without upload (`--no-upload`): ~3.2 TB total.

### Recommended Mount Points

| Mount | Use for | Reason |
|-------|---------|--------|
| `/mnt/ssd-3` (19 TB free) | `OUTPUT_DIR` | Fast SSD, ample space |
| `/mnt/ssd-cluster` (500 GB) | Alternative for single-lang runs | Fast Ceph SSD |
| `/` overlay (1.4 TB) | Code, venvs | Per-node NVMe |

```bash
OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_beetlestream.sh
```

## HuggingFace Datasets

All outputs are uploaded to the `Beetle-Data` organization on HuggingFace.

### Static Mode Datasets

```
Beetle-Data/{lang}-24B              # Pretokenized Arrow (L1 side)
Beetle-Data/en-for-{lang}-24B       # Pretokenized Arrow (English side, tokenized with {lang}-en tokenizer)
```

### Curriculum Mode Datasets

```
Beetle-Data/{lang}-indexed-24B      # Indexed Parquet shards per language
                                    #   Hive-partitioned by topic: lang={xx}/topic={0..199}/shard_*.parquet
                                    #   Columns: text, url, doc_id, quality, difficulty, topic_id

Beetle-Data/{lang}-curriculum-24B   # Pretokenized curriculum Arrow per language
                                    #   Columns: input_ids (513-token chunks), quality, difficulty, topic_id

Beetle-Data/beetlestream-annotations  # Teacher annotation JSONL + feature Parquet
                                      #   Rubric scores for 500K sampled documents

Beetle-Data/beetlestream-student-model  # Trained student model artifacts
                                        #   sklearn regressors, embedder config, cluster centroids
```

### Loading Datasets

```python
from datasets import load_dataset, load_from_disk

# Static mode
ds = load_dataset("Beetle-Data/pl-24B", split="train")

# Curriculum mode (indexed shards)
ds = load_dataset("Beetle-Data/pl-indexed-24B", split="train")

# Curriculum mode (pretokenized)
ds = load_dataset("Beetle-Data/pl-curriculum-24B", split="train")
```

## Output Formats

### Static Mode

#### Stage 2 Parquet (decontaminated)

```
columns: text (utf8), url (utf8), doc_id (int64), word_count (int32)
compression: Snappy
shard size: 50,000 documents per file
manifest: {lang}_manifest.json -> {shard_file: [first_doc_id, last_doc_id]}
```

#### Stage 3 Arrow (beetlelm-compatible)

Each Arrow dataset contains `input_ids` sequences of exactly **513 tokens** (512 + 1 for input + label), matching `beetlelm/src/bilingual/data/pretokenize.py` format:
- `tokenizer.encode(text, add_special_tokens=False)`
- No cross-document token bleeding (remainder tokens discarded at document boundaries)
- Loaded via `datasets.load_from_disk()` in beetlelm's `PretokenizedMultilingualDataset`

### Curriculum Mode

#### Indexed Parquet (Stage D output)

```
columns: text (utf8), url (utf8), doc_id (int64),
         quality (float32), difficulty (int8), topic_id (int16)
partitioning: Hive (lang={xx}/topic={0..199}/shard_*.parquet)
shard size: 10,000 documents per file
metadata: manifest.json, cluster_centroids.pkl, topic_distribution.json
```

#### Curriculum Arrow (Stage 3 output, curriculum mode)

```
columns: input_ids (list<int32>, length 513),
         quality (float32), difficulty (int8), topic_id (int16)
format: Arrow dataset, compatible with datasets.load_from_disk()
```

### Token Budget per Bilingual Pair

| Side | Ratio | Tokens (24B total) |
|------|-------|-------------------|
| L1 | 1/2 | 12B |
| English | 1/2 | 12B |

**Trilingual budget**: 8B per language (3 x 8B = 24B). **Monolingual budget**: 24B single language.

## Evaluation Benchmarks (Strict Decontamination)

All evaluation data is indexed as 13-grams. Any training document containing a 13-gram overlap with any benchmark sentence is **discarded entirely**.

### Minimal Pairs (BLiMP Family)

| Benchmark | HuggingFace ID | Language | Good Column | Bad Column |
|-----------|---------------|----------|-------------|------------|
| BLiMP | `nyu-mll/blimp` | English | `sentence_good` | `sentence_bad` |
| ZhoBLiMP | `Junrui1202/zhoblimp` | Chinese | `sentence_good` | `sentence_bad` |
| BLiMP-NL | `juletxara/blimp-nl` | Dutch | `sentence_good` | `sentence_bad` |
| RuBLiMP | `RussianNLP/rublimp` | Russian | `source_sentence` | `target_sentence` |
| TurBLiMP | `juletxara/turblimp` | Turkish | `sentence_good` | `sentence_bad` |
| JBLiMP | `polm-stability/jblimp` | Japanese | `good_sentence` | `bad_sentence` |
| SLING | `suchirsalhan/SLING` | Chinese | `sentence_good` | `sentence_bad` |
| CLiMP | `suchirsalhan/CLiMP` | Chinese | column 2 (label=1) | column 2 (label=0) |
| MultiBLiMP | `jumelet/multiblimp` | NL, DE, FR, FA, BG | `sen` | `wrong_sen` |

### Other Benchmarks

| Benchmark | HuggingFace ID | Type | Text Columns |
|-----------|---------------|------|--------------|
| FLORES-200 | `crystina-z/flores200` | Perplexity (devtest) | `sentence_{lang_tag}` per language |
| XCOMPS | `fpadovani/xcomps-dataset` | Minimal pairs | `acceptable_sent`, `unacceptable_sent` |
| XNLI | `xnli` | NLI (validation) | `premise`, `hypothesis` |
| UD Treebanks | `universal_dependencies` | Syntax (per-language) | `text` |
| MECO-L2 | Local TSV | Reading times | `FullText` |

## Smoke Test (Local CPU, ~10 min)

Run a minimal end-to-end check before launching large-scale jobs. Exercises all stages with tiny data volumes -- no GPU, no HF upload, and minimal disk needed.

### Static Mode Smoke Test

```bash
# Stage 1: Build the full benchmark index (~5 min)
python -m pipeline.run_pipeline --stage 1 --project-root . --skip-disk-check

# Stage 2: Decontaminate a tiny sample for one language (~2 min)
python -m pipeline.run_pipeline --stage 2 --lang pl --target-words 10000 \
  --index pipeline_output/benchmark_13gram.pkl --project-root . \
  --num-workers 4 --skip-disk-check

# Stage 3: Pretokenize the small output, skip upload (~1 min)
python -m pipeline.run_pipeline --stage 3 --lang pl --no-upload --no-cleanup \
  --project-root . --num-workers 4 --skip-disk-check
```

Expected outputs:
- `pipeline_output/benchmark_13gram.pkl` -- benchmark index
- `pipeline_output/decontaminated/pl/*.parquet` -- at least 1 Parquet shard
- `pipeline_output/pretokenized/pl-en/` -- Arrow dataset

### Curriculum Mode Smoke Test

```bash
# Steps 1-2: Build index + decontaminate (as above)
python -m pipeline.run_pipeline --stage 1 --project-root . --skip-disk-check
python -m pipeline.run_pipeline --stage 2 --lang pl --target-words 10000 \
  --index pipeline_output/benchmark_13gram.pkl --project-root . \
  --num-workers 4 --skip-disk-check

# Step 4: Feature transform (requires annotation JSONL from Step 3; skip if no GPU)
python -m pipeline.run_pipeline --stage B --output-dir pipeline_output

# Step 7: Pretokenize in curriculum mode
python -m pipeline.run_pipeline --stage 3 --lang pl --no-upload --no-cleanup \
  --project-root . --num-workers 4 --skip-disk-check --stream-mode curriculum
```

Note: Steps 3 and 5-6 (teacher annotation, student model, score + index) require GPUs and cannot be smoke-tested on CPU alone. The smoke test for curriculum mode validates the non-GPU stages.

## Run Individual Stages

### Static Mode

```bash
# Stage 1: Build 13-gram benchmark index (~5 minutes)
python -m pipeline.benchmark_index \
    --output pipeline_output/benchmark_13gram.pkl \
    --project-root /path/to/PHD

# Stage 2: Stream + decontaminate one language (~45-60 min)
python -m pipeline.decontaminate_stream \
    --lang pl \
    --index pipeline_output/benchmark_13gram.pkl \
    --output-dir pipeline_output

# Stage 3: Pretokenize + upload to HF + cleanup (~20 min)
python -m pipeline.pretokenize_arrow \
    --lang pl \
    --output-dir pipeline_output \
    --hf-user Beetle-Data
```

### Curriculum Stages

```bash
# Stage A: Teacher annotation (requires vLLM server running)
python -m pipeline.teacher_annotate \
    --output-dir pipeline_output \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --vllm-url http://localhost:8000/v1

# Stage B: Feature transform
python -m pipeline.feature_transform --output-dir pipeline_output

# Stage C: Student model training
python -m pipeline.student_model --output-dir pipeline_output

# Stage D: Score + index
python -m pipeline.score_and_index \
    --output-dir pipeline_output \
    --beetlestream-config configs/beetlestream_curriculum.yaml
```

## Run on SLURM Cluster (4 Nodes)

### Static Mode

```bash
# Stages 1+2 across 4 nodes
sbatch scripts/launch_decontaminate.sh

# Stage 3 across 4 nodes
sbatch scripts/launch_pretokenize.sh

# Or the full storage-optimized pipeline
sbatch scripts/launch_full_pipeline.sh
```

### Curriculum Mode

```bash
# Full BeetleStream v2 pipeline
OUTPUT_DIR=/mnt/ssd-3/beetle-data sbatch scripts/launch_beetlestream.sh
```

### Node Assignments

```
Core:
Node 0: pl, nl, es, el, ja
Node 1: fr, zh, de, it, eu
Node 2: tr, id, tl, fa, hi
Node 3: ta, sv, ru, ca, ar
Node 4: en

Extension (separate runs):
Node 0: ur, bn, cs, gu
Node 1: th, vi, ko, da

Low-Resource (separate runs):
Node 0: hu, bg, hr, uk, sl
Node 1: so, am, yo, wo
```

## Run via Orchestrator

```bash
# Full static pipeline, all languages
python -m pipeline.run_pipeline --project-root /path/to/PHD

# Full static pipeline, specific languages
python -m pipeline.run_pipeline --lang pl nl es --project-root /path/to/PHD

# Full static pipeline, SLURM node
python -m pipeline.run_pipeline --node-id 0 --project-root /path/to/PHD

# Specific stages only
python -m pipeline.run_pipeline --stage 2 3 --node-id 0 \
    --index pipeline_output/benchmark_13gram.pkl

# Curriculum stages only
python -m pipeline.run_pipeline --stage A B C D --node-id 0 \
    --beetlestream-config configs/beetlestream_curriculum.yaml

# No upload, no cleanup (for debugging)
python -m pipeline.run_pipeline --lang pl --no-upload --no-cleanup
```

## Stream Held-Out Evaluation Data (Stage 4)

Streams FineWeb-2 documents **beyond the training cutoff** for use in learning curve analysis (Chang & Bergen, 2024). Reads `docs_streamed` from training stats to ensure no overlap with training data. Works with both static and curriculum modes.

```bash
# Stream 10,000 held-out documents for German
python -m pipeline.stream_held_out --lang de --output-dir pipeline_output --n-docs 10000

# Stream for multiple languages
for lang in de nl ja zh es ru; do
    python -m pipeline.stream_held_out --lang $lang --output-dir pipeline_output --n-docs 10000
done
```

Output: `pipeline_output/held_out/{lang}/*.parquet` + `{lang}_held_out_stats.json`

The held-out data is consumed by `beetle-analyze/aoa/prepare_eval_data.py` for constructing evaluation sequences that are guaranteed unseen during training.

## Post-Hoc Contamination Analysis

For Infinigram-style querying to analyze whether model performance correlates with training data presence:

```bash
# Check if a string would have been flagged as contaminated
python -m pipeline.post_hoc check \
    --index pipeline_output/benchmark_13gram.pkl \
    --text "The cat sat on the mat and watched the birds fly."

# Search the training corpus for n-gram matches
python -m pipeline.post_hoc scan \
    --lang pl --text "example query text" \
    --output-dir pipeline_output

# Batch analysis from a file
python -m pipeline.post_hoc batch \
    --index pipeline_output/benchmark_13gram.pkl \
    --input eval_strings.txt \
    --lang pl --output-dir pipeline_output
```

## Verify Output

```bash
# Full verification (checks chunk lengths, token ranges, contamination)
python scripts/verify_output.py --output-dir pipeline_output --hf-user Beetle-Data

# Quick check (skip contamination scan)
python scripts/verify_output.py --output-dir pipeline_output --quick

# Specific languages only
python scripts/verify_output.py --output-dir pipeline_output --langs pl nl es
```

## Project Structure

```
beetle-data/
  pipeline/
    __init__.py
    config.py                  # Language registry, benchmark defs, pipeline + BeetleStream config
    utils.py                   # Text normalization, n-gram extraction
    benchmark_index.py         # Stage 1: Build 13-gram index from benchmarks
    decontaminate_stream.py    # Stage 2: Stream + decontaminate + write Parquet
    teacher_annotate.py        # Stage A: LLM pedagogical annotation via vLLM
    feature_transform.py       # Stage B: Annotation JSONL -> feature Parquet
    student_model.py           # Stage C: Train embedding-based quality approximator
    heuristic_filters.py       # Stage D (pass 1): Fast text triage filters
    score_and_index.py         # Stage D (pass 2): Full corpus scoring + indexed shards
    pretokenize_arrow.py       # Stage 3: Tokenize + pack + write Arrow (static + curriculum)
    stream_held_out.py         # Stage 4: Stream held-out data beyond training cutoff
    post_hoc.py                # Post-hoc contamination analysis
    run_pipeline.py            # CLI orchestrator (all stages)
  configs/
    beetlestream_curriculum.yaml  # BeetleStream v2 curriculum pipeline config
    fineweb_bilingual.yaml        # 12B per language bilingual config
    fineweb_monolingual.yaml      # 24B monolingual experiment config
    fineweb_trilingual.yaml       # 8B per language trilingual config
    fineweb_tokenizer.yaml        # Tokenizer training config
  scripts/
    launch_beetlestream.sh     # BeetleStream v2: full curriculum pipeline (all steps)
    launch_decontaminate.sh    # SLURM: Stage 1+2 across 4 nodes
    launch_pretokenize.sh      # SLURM: Stage 3 across 4 nodes
    launch_full_pipeline.sh    # Single-node full static pipeline
    launch_extensions.sh       # Extension language pipeline
    launch_low_resource.sh     # Low-resource language pipeline
    run_all_experiments.sh     # Experiment runner
    verify_output.py           # End-to-end validation
  tok/                         # Tokenizer training scripts (already complete)
  data/                        # Legacy decontamination scripts
  stream/                      # Streaming utilities
```

## FineWeb-2 Language Mappings

Source data identifiers (matching `tok/multi-train-tok.py`):

```
pl -> pol_Latn    nl -> nld_Latn    es -> spa_Latn    el -> ell_Grek
ja -> jpn_Jpan    fr -> fra_Latn    zh -> cmn_Hani    de -> deu_Latn
it -> ita_Latn    eu -> eus_Latn    tr -> tur_Latn    id -> ind_Latn
tl -> fil_Latn    fa -> fas_Arab    hi -> hin_Deva    ta -> tam_Taml
sv -> swe_Latn    ru -> rus_Cyrl    ca -> cat_Latn
ar -> arb_Arab
en -> FineWeb-Edu (HuggingFaceFW/fineweb-edu)
```

Extension and low-resource languages: see `pipeline/config.py` `LANG_REGISTRY` for full mappings.

## Configuration

All pipeline settings are in `pipeline/config.py`:

- `PipelineConfig`: token targets, sequence length, worker count, shard sizes, stream mode
- `BeetleStreamConfig`: teacher model, student model, heuristic thresholds, indexing params
- `StreamMode`: enum for `static`, `random_stream`, `curriculum`
- `LANG_REGISTRY`: language metadata (FineWeb-2 names, FLORES tags, tier)
- `BENCHMARK_DEFS`: evaluation benchmarks and their HuggingFace paths/columns
- `NODE_ASSIGNMENTS`: SLURM node-to-language mapping (core, extension, low-resource)

Key defaults:
- Sequence length: 512 (chunk length: 513 = seq_len + 1)
- N-gram size: 13
- Target words per language: ~22B (overshoots to ~28B tokens, yields ~24B after decontamination)
- Bilingual ratio: 1/2 L1, 1/2 English (50:50 split)
- Workers: 24 per stage
- Shard size: 50,000 documents per Parquet file (static), 10,000 per indexed shard (curriculum)
- Teacher model: `meta-llama/Meta-Llama-3-70B-Instruct`
- Student embedding: `intfloat/multilingual-e5-base` (768-dim)
- Topic clusters: 200 per language (k-means)
- Teacher sample size: 500,000 documents across all languages

Curriculum-specific configuration lives in `configs/beetlestream_curriculum.yaml`.
