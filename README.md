# Beetle-Data

Modular large-scale preprocessing library for streaming, decontaminating, and pretokenizing multilingual training data. Produces Arrow datasets compatible with the [BeetleLM](../beetlelm) training framework.

## Overview

Beetle-Data prepares 24B-token bilingual (L1 + English) training corpora for 125M-parameter language models following Chinchilla scaling laws. The pipeline streams data from FineWeb-Edu (English) and FineWeb-2 (non-English), removes evaluation benchmark contamination via 13-gram overlap detection, and pretokenizes using pre-trained bilingual BPE tokenizers.

### Architecture: Three Decoupled Stages

```
Stage 1                    Stage 2                         Stage 3
Benchmark Index     -->    Streaming Decontamination  -->  Pretokenization
(one-time, ~5 min)         (~5 hrs, 4 nodes)               (~2 hrs, 4 nodes)

HF benchmarks              FineWeb-2 / FineWeb-Edu         Clean Parquet
    |                           |                               |
    v                           v                               v
13-gram index (.pkl)       Clean Parquet shards            Arrow datasets
                           + manifest.json                 (beetlelm-compatible)
```

Stages are decoupled so you can re-run pretokenization when a tokenizer changes without re-streaming 560B tokens, or re-run decontamination against new benchmarks without re-tokenizing.

## Languages (38 + English)

| Group | Languages |
|-------|-----------|
| Core (20) | Polish (pl), Dutch (nl), Spanish (es), Greek (el), Japanese (ja), French (fr), Chinese (zh), German (de), Italian (it), Basque (eu), Turkish (tr), Indonesian (id), Tagalog (tl), Persian (fa), Hindi (hi), Tamil (ta), Swedish (sv), Russian (ru), Catalan (ca), Arabic (ar) |
| Extension (8) | Urdu (ur), Bengali (bn), Czech (cs), Gujarati (gu), Thai (th), Vietnamese (vi), Korean (ko), Danish (da) |
| Low-Resource European (5) | Hungarian (hu), Bulgarian (bg), Croatian (hr), Ukrainian (uk), Slovenian (sl) |
| Low-Resource African (4) | Somali (so), Amharic (am), Yoruba (yo), Wolof (wo) |
| Bilingual partner | English (en) via FineWeb-Edu |

Each non-English language is paired with English using a bilingual tokenizer from HuggingFace (`Beetle-Data/tokenizer-{lang}-en`).

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

## Quick Start

### Prerequisites

```bash
python3 -m venv venvs/demo; source venvs/demo/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your-token>
```

### Run the Full Pipeline (Storage-Optimized)

The default mode processes each language sequentially: decontaminate, pretokenize, upload to HuggingFace (`Beetle-Data/`), then delete local files. Peak disk usage stays under 300 GB.

```bash
cd beetle-data

# All 19 languages (default) — uploads each to HF, cleans up locally
bash scripts/launch_full_pipeline.sh

# Specific languages only
bash scripts/launch_full_pipeline.sh --lang pl nl es

# Custom output directory (e.g., on an SSD mount with more space)
OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_full_pipeline.sh

# Without HF upload (keep local files — needs ~3.2 TB disk)
bash scripts/launch_full_pipeline.sh --no-upload

# Extension languages (lower priority, separate pipeline)
bash scripts/launch_extensions.sh

# Low-resource languages (lowest priority)
bash scripts/launch_low_resource.sh
```

### Run Individual Stages

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

### Run on SLURM Cluster (4 Nodes)

Each node processes its assigned languages with the storage-optimized flow:

```bash
# Full pipeline across 4 nodes (index build on node 0, all nodes process)
sbatch scripts/launch_decontaminate.sh
```

Set the output directory to a shared SSD mount:
```bash
OUTPUT_DIR=/mnt/ssd-3/beetle-data sbatch scripts/launch_decontaminate.sh
```

Node assignments:
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

### Run via Orchestrator

```bash
# Full pipeline, all languages
python -m pipeline.run_pipeline --project-root /path/to/PHD

# Full pipeline, specific languages
python -m pipeline.run_pipeline --lang pl nl es --project-root /path/to/PHD

# Full pipeline, SLURM node
python -m pipeline.run_pipeline --node-id 0 --project-root /path/to/PHD

# Specific stages only
python -m pipeline.run_pipeline --stage 2 3 --node-id 0 \
    --index pipeline_output/benchmark_13gram.pkl

# No upload, no cleanup (for debugging)
python -m pipeline.run_pipeline --lang pl --no-upload --no-cleanup
```

### Verify Output

```bash
# Full verification (checks chunk lengths, token ranges, contamination)
python scripts/verify_output.py --output-dir pipeline_output --hf-user Beetle-Data

# Quick check (skip contamination scan)
python scripts/verify_output.py --output-dir pipeline_output --quick

# Specific languages only
python scripts/verify_output.py --output-dir pipeline_output --langs pl nl es
```

## Output Format

### On HuggingFace (permanent storage)

Each pretokenized dataset is uploaded to `https://huggingface.co/datasets/Beetle-Data/`:

```
Beetle-Data/pl-24B              # Polish pretokenized Arrow dataset
Beetle-Data/en-for-pl-24B       # English (tokenized with pl-en tokenizer)
Beetle-Data/nl-24B
Beetle-Data/en-for-nl-24B
...
```

To load for training:
```python
from datasets import load_dataset
ds = load_dataset("Beetle-Data/pl-24B", split="train")
```

### On local disk (temporary during processing)

```
pipeline_output/
  benchmark_13gram.pkl          # Stage 1 index (~200 MB, kept)
  decontaminated/
    en/                         # EN Parquet (~70 GB, kept until all pairs done)
    pl/                         # L1 Parquet (~70 GB, deleted after pretokenization)
  pretokenized/
    pl/                         # Arrow (~64 GB, deleted after HF upload)
    en_for_pl/                  # Arrow (~32 GB, deleted after HF upload)
```

### Stage 2 Parquet Format

```
columns: text (utf8), url (utf8), doc_id (int64), word_count (int32)
compression: Snappy
shard size: 50,000 documents per file
manifest: {lang}_manifest.json → {shard_file: [first_doc_id, last_doc_id]}
```

### Stage 3 Arrow Format (beetlelm-compatible)

Each Arrow dataset contains `input_ids` sequences of exactly **513 tokens** (512 + 1 for input + label), matching `beetlelm/src/bilingual/data/pretokenize.py` format:
- `tokenizer.encode(text, add_special_tokens=False)`
- No cross-document token bleeding (remainder tokens discarded at document boundaries)
- Loaded via `datasets.load_from_disk()` in beetlelm's `PretokenizedMultilingualDataset`

### Token Budget per Bilingual Pair

| Side | Ratio | Tokens (24B total) |
|------|-------|-------------------|
| L1 | 1/2 | 12B |
| English | 1/2 | 12B |

**Trilingual budget**: 8B per language (3 x 8B = 24B). **Monolingual budget**: 24B single language.

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

## Compute Requirements

### Hardware

- 4 nodes x 8 A100-80GB GPUs each (GPUs are idle -- pipeline is CPU-only)
- 128 CPUs per node (Intel Xeon Platinum 8358 @ 2.60GHz)
- ~512 GB RAM per node

### Time and Cost Estimates

| Stage | Wall-clock (4 nodes) | GPU-hours | CPU-hours |
|-------|---------------------|-----------|-----------|
| 1. Index Build | 5 min | 0 | ~1 |
| 2. Stream + Decontaminate | ~5 hours | 0 | ~2,560 |
| 3. Pretokenize | ~2 hours | 0 | ~1,024 |
| **Total** | **~7 hours** | **0** | **~3,585** |

**Node-hours**: 4 nodes x 7 hours = **28 node-hours**.

### Memory Budget per Node

| Component | Stage 2 | Stage 3 |
|-----------|---------|---------|
| BenchmarkIndex | ~0.2 GB | -- |
| HF streaming buffers | ~2 GB | -- |
| Parquet I/O buffers | ~4 GB | ~4 GB |
| Tokenizer instances | -- | ~6 GB |
| Arrow write buffers | -- | ~10 GB |
| Worker overhead | ~20 GB | ~20 GB |
| **Total** | **~35 GB** | **~48 GB** |

### Disk Budget (Storage-Optimized Mode)

The pipeline processes languages one at a time and uploads Arrow datasets to HuggingFace after each, keeping local disk usage minimal.

| Phase | On disk | Notes |
|-------|---------|-------|
| EN decontamination | ~70 GB | Kept until all 19 pairs done |
| L1 decontamination (current lang) | ~70 GB | Deleted after pretokenization |
| Arrow `_parts/` temp (during build) | ~110 GB | Cleaned up per-language |
| Arrow dataset (before upload) | ~96 GB | Deleted after HF upload |
| HF cache + index | ~20 GB | Persistent |
| **Peak at any point** | **~270 GB** | |
| **After each language completes** | **~90 GB** | EN Parquet + index + cache |
| **After all languages complete** | **~20 GB** | Index + cache only |

Without `--no-upload` (keeping all local files): ~3.2 TB total.

#### Recommended mount points

| Mount | Use for | Reason |
|-------|---------|--------|
| `/mnt/ssd-3` (19 TB free) | `OUTPUT_DIR` | Fast SSD, ample space |
| `/mnt/ssd-cluster` (500 GB) | Alternative for single-lang runs | Fast Ceph SSD |
| `/` overlay (1.4 TB) | Code, venvs | Per-node NVMe |

```bash
# Use SSD mount for output
OUTPUT_DIR=/mnt/ssd-3/beetle-data python -m pipeline.run_pipeline ...
```

## Project Structure

```
beetle-data/
  pipeline/
    __init__.py
    config.py                  # Language registry, benchmark defs, pipeline config
    utils.py                   # Text normalization, n-gram extraction
    benchmark_index.py         # Stage 1: Build 13-gram index from benchmarks
    decontaminate_stream.py    # Stage 2: Stream + decontaminate + write Parquet
    pretokenize_arrow.py       # Stage 3: Tokenize + pack + write Arrow
    post_hoc.py                # Post-hoc contamination analysis
    run_pipeline.py            # CLI orchestrator
  configs/
    fineweb_monolingual.yaml   # 24B monolingual experiment config
    fineweb_bilingual.yaml     # 12B per language bilingual config
    fineweb_trilingual.yaml    # 8B per language trilingual config
    fineweb_tokenizer.yaml     # Tokenizer training config
  scripts/
    launch_decontaminate.sh    # SLURM: Stage 1+2 across 4 nodes
    launch_pretokenize.sh      # SLURM: Stage 3 across 4 nodes
    launch_full_pipeline.sh    # Single-node full pipeline
    launch_extensions.sh       # Extension language pipeline
    launch_low_resource.sh     # Low-resource language pipeline
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

- `PipelineConfig`: token targets, sequence length, worker count, shard sizes
- `LANG_REGISTRY`: language metadata (FineWeb-2 names, FLORES tags)
- `BENCHMARK_DEFS`: evaluation benchmarks and their HuggingFace paths/columns
- `NODE_ASSIGNMENTS`: SLURM node-to-language mapping

Key defaults:
- Sequence length: 512 (chunk length: 513 = seq_len + 1)
- N-gram size: 13
- Target words per language: ~22B (overshoots to ~28B tokens, yields ~24B after decontamination)
- Bilingual ratio: 1/2 L1, 1/2 English (50:50 split)
- Workers: 24 per stage
- Shard size: 50,000 documents per Parquet file
