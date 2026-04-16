# Beetle-Data

Modular large-scale preprocessing library for streaming, decontaminating, scoring, and pretokenizing multilingual training data. Produces Arrow datasets compatible with the [BeetleLM](../beetlelm) training framework. Supports **static mode** (3-stage pipeline) and **curriculum mode** (BeetleStream v2, 7-stage pipeline with pedagogical quality scoring, topic clustering, and difficulty grading).

## Overview

Beetle-Data prepares bilingual (L1 + English) training corpora for 125M-parameter language models following Chinchilla scaling laws. The pipeline streams raw data from FineWeb-Edu (English) and FineWeb-2 (non-English), collecting **28B clean tokens per language side** after decontamination, then cuts to a **24B-token training target** (12B L1 + 12B EN) at the pretokenization stage.

Evaluation benchmark contamination is removed via 13-gram overlap detection across 19 benchmarks including BLiMP, FLORES-200, XNLI, UD Treebanks, and MECO-L2 (the reading-time stimuli loaded directly from [HuggingFace](https://huggingface.co/datasets/suchirsalhan/MECO)).

**BeetleStream v2** extends the pipeline with four curriculum stages (A→B→C→D) that add pedagogical quality scoring, topic clustering, and difficulty grading for curriculum-aware training.

## Quick Customization

To change the token budget or swap the training dataset, edit four constants at the top of `pipeline/config.py`. No other files need changing.

```python
# pipeline/config.py — USER-CONFIGURABLE section

STREAM_TOKENS_PER_LANG = 28_000_000_000   # Clean tokens to collect per language
TARGET_TOKENS_PER_LANG = 24_000_000_000   # Tokens used for training (24B bilingual)

TRAINING_DATASET_L1 = "HuggingFaceFW/fineweb-2"    # Non-English source
TRAINING_DATASET_EN = "HuggingFaceFW/fineweb-edu"  # English source
TRAINING_TEXT_FIELD = "text"                        # Document text field
```

The pipeline derives all internal word-count targets from these constants. Swap `TRAINING_DATASET_L1` / `TRAINING_DATASET_EN` to any HuggingFace dataset (same `name=` subset convention for L1, plain `split="train"` for EN).

### Token Budget Explained

```
FineWeb-2 / FineWeb-Edu stream
        │
        ▼
  13-gram decontamination (Stage 2)
        │   collect until 28B clean tokens per language side
        │   (≈ 21.5B whitespace words at 1.3 BPE tokens/word)
        ▼
  Raw decontaminated Parquet shards
        │
        ▼
  Pretokenization (Stage 3)
        │   stop at 24B tokens per bilingual pair
        │   → 12B L1 tokens + 12B EN tokens
        ▼
  Arrow datasets  ──▶  BeetleLM training
```

---

## Architecture

```
STATIC MODE (3-stage pipeline)
══════════════════════════════════════════════════════════

Stage 1            Stage 2                       Stage 3
Benchmark      →   Stream + Decontaminate    →   Pretokenize
Index              (28B tokens / lang)            (cut to 24B)
(one-time)         (~5 hrs, 4 nodes)              (~2 hrs, 4 nodes)

HF benchmarks      FineWeb-2 / FineWeb-Edu         Clean Parquet
MECO-L2 (HF)            │                               │
    │                   ▼                               ▼
    ▼              Clean Parquet               Arrow datasets (beetlelm)
13-gram index      + manifest.json


CURRICULUM MODE (BeetleStream v2, 7-stage pipeline)
══════════════════════════════════════════════════════════════════

 Stage 1 → Stage 2 → Stage A     → Stage B    → Stage C
 Index      Decontam  Teacher        Feature      Student
 (5 min)    (5 hrs)   Annotation     Transform    Model
                      (7 hrs,        (10 min)     (10 min)
                       1 node)

 Stage D              Stage 3                 Stage 4 (optional)
 Score + Index   →   Pretokenize         →   Held-Out Streaming
 (10 hrs, 4 nodes)   (2 hrs, 4 nodes)        (~2 hrs, 1 node)

 Hive-partitioned     Curriculum Arrow         FineWeb-2 beyond
 Parquet (topic/      (quality, difficulty,    training cutoff
  quality/difficulty)  topic_id, input_ids)    (for AoA eval)
```

Stages are decoupled: re-run pretokenization when a tokenizer changes without re-streaming 28B tokens; re-run decontamination against new benchmarks without re-tokenizing.

---

## Languages (38 + English)

| Group | Languages |
|-------|-----------|
| Core (20) | Polish (pl), Dutch (nl), Spanish (es), Greek (el), Japanese (ja), French (fr), Chinese (zh), German (de), Italian (it), Basque (eu), Turkish (tr), Indonesian (id), Tagalog (tl), Persian (fa), Hindi (hi), Tamil (ta), Swedish (sv), Russian (ru), Catalan (ca), Arabic (ar) |
| Extension (8) | Urdu (ur), Bengali (bn), Czech (cs), Gujarati (gu), Thai (th), Vietnamese (vi), Korean (ko), Danish (da) |
| Low-Resource European (5) | Hungarian (hu), Bulgarian (bg), Croatian (hr), Ukrainian (uk), Slovenian (sl) |
| Low-Resource African (4) | Somali (so), Amharic (am), Yoruba (yo), Wolof (wo) |
| Bilingual partner | English (en) via FineWeb-Edu |

---

## Curriculum Mode: The Idea

Standard language model pretraining presents documents in random order. For models trained to study language acquisition and second-language learning, the **ordering and composition** of training data matters. BeetleStream v2 introduces a pedagogically-motivated curriculum:

**Why curriculum ordering?** Evidence from cognitive science and L2 acquisition research suggests that learners benefit from progressing from simpler to more complex material, from high-frequency to rare vocabulary, and from well-structured to noisier text. BeetleStream v2 encodes these dimensions as document-level signals and uses them to partition and order the training corpus.

**How it works:**

1. **Teacher annotation (Stage A)**: A large teacher LLM (Llama-3-70B-Instruct) evaluates a sample of 500K documents using a pedagogical rubric covering five dimensions — quality (0–5), difficulty (1–3), vocabulary complexity, engagement, and topic coherence. Calibration examples from KidLM-corpus (child-directed text) and CLC-L1-CEFR (graded L2 writing) anchor the rubric to real proficiency levels.

2. **Student distillation (Stages B→C)**: Because running the 70B teacher on billions of documents is too costly, a lightweight student model is trained to mimic the teacher. Documents are embedded with `multilingual-e5-base` (768-dim), and sklearn regressors predict quality and difficulty scores for every document in the corpus.

3. **Topic clustering (Stage D)**: Documents are clustered into 200 topic groups via k-means on the same embeddings. Each document receives a `topic_id`, enabling topic-balanced sampling during training — preventing any single domain from dominating training batches.

4. **Curriculum Arrow (Stage 3, curriculum mode)**: The pretokenized output includes `quality`, `difficulty`, and `topic_id` alongside `input_ids`. BeetleLM's data loader uses these to implement custom sampling schedules: easy → hard, or topic-diverse → topic-focused, depending on training phase.

The result is a training corpus with the same raw text as the static pipeline but annotated with pedagogical metadata that enables controlled experiments in curriculum learning for language acquisition research.

---

## Prerequisites

```bash
python3 -m venv venvs/demo; source venvs/demo/bin/activate
pip install -r requirements.txt
export HF_TOKEN=<your-token>
```

---

## Step 0: Train Tokenizers

Bilingual tokenizers must be trained and published to HuggingFace **before** running the pipeline (Stage 3 loads them for pretokenization). Each tokenizer is trained on 2M sentences streamed from the same FineWeb sources used by the pipeline: `HuggingFaceFW/fineweb-2` (L1) and `HuggingFaceFW/fineweb-edu` (English).

**Train all 20 languages** (sequential, ~30 min per language):

```bash
bash tok/run.sh
```

**Train a single language** (e.g., Arabic):

```bash
python tok/multi-train-tok.py --lang ar --hf-user Beetle-Data --vocab-size 50000 --sentences 2000000
```

**Train a subset** (e.g., batch 4 languages):

```bash
for lang in sv el ca fa id; do
  python tok/multi-train-tok.py --lang "$lang" --hf-user Beetle-Data --vocab-size 50000 --sentences 2000000
done
```

Output: `Beetle-Data/tokenizer-{lang}-en` on HuggingFace Hub (e.g., `Beetle-Data/tokenizer-ar-en`).

Japanese and Chinese have dedicated scripts (`tok/ja-en-tok.py`, `tok/zh-en-tok.py`) that are called automatically by `tok/run.sh`.

| Language Group | Tokenizer Model | Normalization | Notes |
|----------------|-----------------|---------------|-------|
| Most Latin-script, Greek, Russian | BPE | NFKC | Standard ByteLevel pre-tokenization |
| Arabic | Unigram | NFKC + tatweel removal | UnicodeScripts + Metaspace pre-tokenization |
| Persian | BPE | NFKC + tatweel removal | ByteLevel pre-tokenization |
| Chinese | BPE | NFC | Prefix space, trim offsets |
| Japanese | BPE | NFC | mecab-based pre-tokenization (separate script) |

---

## Step 0: Train Tokenizers

Bilingual tokenizers must be trained and published to HuggingFace **before** running the pipeline (Stage 3 loads tokenizers from HF). Each tokenizer is trained on 2M sentences streamed from the same FineWeb sources used by the pipeline (`HuggingFaceFW/fineweb-2` for L1, `HuggingFaceFW/fineweb-edu` for English).

**Train all 20 languages** (sequential, ~30 min per language):
```bash
bash tok/run.sh
```

**Train a single language** (e.g., Arabic):
```bash
python tok/multi-train-tok.py --lang ar --hf-user Beetle-Data --vocab-size 50000 --sentences 2000000
```

**Output**: `Beetle-Data/tokenizer-{lang}-en` on HuggingFace Hub (e.g., `Beetle-Data/tokenizer-ar-en`).

The pipeline's pretokenization stage (`pipeline/pretokenize_arrow.py`) loads the tokenizer via `config.tokenizer_repo(lang)` which resolves to these HF repos. If a tokenizer is missing, Stage 3 will fail with a download error.

Supported languages and tokenizer types are defined in `tok/multi-train-tok.py:LANG_CONFIGS`. Japanese and Chinese use separate training scripts (`tok/ja-en-tok.py`, `tok/zh-en-tok.py`) but are included in the batch `run.sh`.

---

## Human-Scale BabyBabel Pretokenization

Pretokenizes BabyBabel raw text for human-scale mono/bi/trilingual experiments in BeetleLM. Produces Arrow datasets compatible with `beetlelm`'s `PretokenizedMultilingualDataset` and pushes them to **`Beetle-HumanScale/`** on HuggingFace Hub.

> **HF org change:** Human-scale data and tokenizers now live under [`Beetle-HumanScale`](https://huggingface.co/Beetle-HumanScale) (replacing `Beetle-Data` for BabyBabel). The `Beetle-Data` org continues to host the large-scale FineWeb pipeline outputs.

### Why pretokenize in beetle-data?

Human-scale experiments (Experiment 1) run hundreds of sweep configs. Without pretokenization, each training run re-tokenizes the same BabyBabel text on-the-fly. Pretokenizing once in beetle-data and loading from HF (`pretokenized_source: "hf_arrow"`) eliminates this overhead across all runs.

### Token Budgets

| Mode | Per-language budget | Total tokens |
|------|-------------------|--------------|
| Monolingual | 100M | 100M |
| Bilingual (balanced) | 50M x 2 | 100M |
| Bilingual B4 Classroom | 80M L1 + 20M L2 | 100M |
| Trilingual | 33M x 3 | ~100M |

### Dataset Naming Conventions

All datasets are pushed to the `Beetle-HumanScale` HF org. The naming scheme encodes the language, token budget, and tokenizer used:

| Mode | L1 dataset name | L2 / other dataset names |
|------|----------------|--------------------------|
| **Monolingual** | `Beetle-HumanScale/{lang}-100M` | -- |
| **Bilingual (balanced)** | `Beetle-HumanScale/{l1}-50M-{tok_pair}` | `Beetle-HumanScale/{l2}-for-{l1}-50M-{tok_pair}` |
| **Bilingual B4 Classroom** | `Beetle-HumanScale/{l1}-80M-{tok_pair}` | `Beetle-HumanScale/{l2}-for-{l1}-20M-{tok_pair}` |
| **Trilingual** | `Beetle-HumanScale/{lang}-33M-{tok_triple}` | (one dataset per language in the triple) |

Examples:
- Monolingual Dutch: `Beetle-HumanScale/nld-100M`
- Bilingual Dutch L1: `Beetle-HumanScale/nld-50M-eng-nld`
- Bilingual English L2 for Dutch: `Beetle-HumanScale/eng-for-nld-50M-eng-nld`
- B4 Classroom Dutch L1: `Beetle-HumanScale/nld-80M-eng-nld`
- B4 Classroom English L2 for Dutch: `Beetle-HumanScale/eng-for-nld-20M-eng-nld`
- Trilingual Dutch: `Beetle-HumanScale/nld-33M-eng-nld-zho`

### Tokenizer Naming Conventions

Tokenizers are also hosted under `Beetle-HumanScale`. Naming follows the language combination in sorted order:

| Mode | Tokenizer repo |
|------|---------------|
| Monolingual | `Beetle-HumanScale/bpe-humanscale-{lang}` |
| Bilingual | `Beetle-HumanScale/bpe-humanscale-{sorted_pair}` |
| Trilingual | `Beetle-HumanScale/bpe-humanscale-{sorted_triple}` |

Examples:
- Monolingual Dutch: `Beetle-HumanScale/bpe-humanscale-nld`
- Bilingual eng+nld: `Beetle-HumanScale/bpe-humanscale-eng-nld`
- Trilingual eng+nld+zho: `Beetle-HumanScale/bpe-humanscale-eng-nld-zho`

### Automatic Tokenizer Training (`ensure_tokenizer()`)

Tokenizers are auto-detected and auto-trained if missing on the HF Hub. When a pretokenization run begins, `ensure_tokenizer()` checks whether the required tokenizer repo exists on `Beetle-HumanScale`. If the tokenizer is not found, it is trained on the fly and pushed to the Hub before pretokenization proceeds. No manual tokenizer training step is needed for human-scale experiments.

### Input / Output

| | Description |
|---|---|
| **Input** | Raw text from `BabyLM-community/babylm-{lang}` (9 languages: zho, fas, eng, nld, bul, fra, ind, deu, ukr) |
| **Tokenizer** | `Beetle-HumanScale/bpe-humanscale-{lang(s)}` (auto-trained if missing) |
| **Output** | Arrow datasets on `Beetle-HumanScale/` with `input_ids` column (chunk_len=513) |
| **Format** | Exactly matches `beetlelm/src/bilingual/data/pretokenize.py`: `encode(text, add_special_tokens=False)`, pack into 513-token chunks, no cross-document token bleeding |

Since tokenizers encode the language combination, the same raw text tokenized with different tokenizers produces different `input_ids`. Each output dataset encodes the tokenizer used in its name (e.g., `nld-50M-eng-nld` = Dutch text tokenized with the eng-nld tokenizer).

### Usage

```bash
# Monolingual (100M tokens)
python -m pipeline.pretokenize_babybabel --mono --lang nld --target 100M

# Bilingual (50M per side)
python -m pipeline.pretokenize_babybabel --pair eng nld

# B4 Classroom (80M L1 + 20M L2)
python -m pipeline.pretokenize_babybabel --pair eng nld --l1-target 80M --l2-target 20M

# Trilingual (33M per side)
python -m pipeline.pretokenize_babybabel --triple eng nld zho --target 33M

# Pilot (3 MECO pairs)
python -m pipeline.pretokenize_babybabel --pilot

# All 36 pairs (72 Arrow datasets)
python -m pipeline.pretokenize_babybabel --all

# Skip HuggingFace upload (local Arrow only)
python -m pipeline.pretokenize_babybabel --all --no-upload
```

### SLURM Submission

Human-scale data is small (~100M tokens total), so a single node suffices:

```bash
# All 36 pairs (under 1 hour)
sbatch scripts/launch_pretokenize_babybabel.sh

# Pilot only
MODE=pilot sbatch scripts/launch_pretokenize_babybabel.sh

# Single pair
L1=eng L2=nld sbatch scripts/launch_pretokenize_babybabel.sh
```

### Integration with BeetleLM

BeetleLM training configs reference these datasets with `pretokenized_source: "hf_arrow"`:

```yaml
# In a beetlelm training config:
data:
  lang_sources: ["Beetle-HumanScale/nld-50M-eng-nld"]
  l2_streams: ["Beetle-HumanScale/eng-for-nld-50M-eng-nld"]
  pretokenized: true
  pretokenized_source: "hf_arrow"
```

The config generator (`beetlelm/configs/generate_human_scale_bilingual.py`) produces these references automatically. Run pretokenization **before** launching training jobs.

---

## Execution: 4-Node Cluster

All four steps below are designed for 4 nodes of 8 × A100-80GB GPUs each with a shared filesystem (e.g., `/mnt/ssd-3`).

### STEP 1 — Static Pipeline: Stages 1 + 2 + 3 (4 nodes in parallel)

Each node runs independently with a disjoint language set. Stage 1 (benchmark index) is built once by whichever node runs first; all subsequent nodes skip it automatically. Streams 28B clean tokens per language, then pretokenizes to 24B-token bilingual pairs.

```bash
# Node 0
bash scripts/launch_full_pipeline.sh --lang fr de es zh ja

# Node 1
bash scripts/launch_full_pipeline.sh --lang nl it ru pl tr

# Node 2
bash scripts/launch_full_pipeline.sh --lang tl hi ta eu ar

# Node 3
bash scripts/launch_full_pipeline.sh --lang sv el ca fa id
```

To prepare for curriculum mode later (uploads raw Parquet shards to HuggingFace):

```bash
# Add --curriculum-prep to any node's command
bash scripts/launch_full_pipeline.sh --lang fr de es zh ja --curriculum-prep
```

Custom output directory:
```bash
OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_full_pipeline.sh --lang fr de es zh ja
```

---

### STEP 2 — Stream Held-Out Evaluation Data (1 node)

Run after STEP 1 completes across all 4 nodes. Streams FineWeb-2 documents beyond the training cutoff for all 20 core languages, for use in learning curve / AoA analysis.

```bash
bash scripts/launch_held_out.sh
```

Options:
```bash
N_DOCS=50000 bash scripts/launch_held_out.sh           # 50K held-out docs/lang
OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_held_out.sh
```

Output: `pipeline_output/held_out/{lang}/*.parquet`

---

### Curriculum STEP 3 — Stages A + B + C (1 node, ~8 hrs)

Run on one node after STEP 1 completes. Annotates 500K sampled documents with a pedagogical rubric (Stage A), extracts features (Stage B), and trains the student scoring model (Stage C). Uploads outputs to HuggingFace for STEP 4 nodes to download.

```bash
bash scripts/launch_curriculum_abc.sh
```

Options:
```bash
# Smaller 8B teacher (faster, lower annotation quality)
bash scripts/launch_curriculum_abc.sh --teacher-model meta-llama/Meta-Llama-3-8B-Instruct

OUTPUT_DIR=/mnt/ssd-3/beetle-data bash scripts/launch_curriculum_abc.sh
```

Uploads:
- `Beetle-Data/beetlestream-annotations` — teacher annotation JSONL + feature Parquet
- `Beetle-Data/beetlestream-student-model` — sklearn model + embedder config + cluster centroids

---

### Curriculum STEP 4 — Stages D + 3 (4 nodes in parallel)

Downloads Stage A–C outputs from HuggingFace, scores and clusters the full corpus (Stage D), then pretokenizes to curriculum Arrow with `quality`, `difficulty`, and `topic_id` columns (Stage 3, curriculum mode). Cuts to 24B tokens per bilingual pair.

```bash
# Node 0
bash scripts/launch_curriculum_d3.sh --lang fr de es zh ja

# Node 1
bash scripts/launch_curriculum_d3.sh --lang nl it ru pl tr

# Node 2
bash scripts/launch_curriculum_d3.sh --lang tl hi ta eu ar

# Node 3
bash scripts/launch_curriculum_d3.sh --lang sv el ca fa id
```

Each node downloads raw Parquet from `Beetle-Data/{lang}-raw-28B`, runs Stages D+3, uploads curriculum Arrow to `Beetle-Data/{lang}-curriculum-28B`, then cleans up local files.

---

## Compute Budget

### Static Mode (Stages 1, 2, 3)

| Step | Stage | What | Wall-clock | A100-hrs | Nodes |
|------|-------|------|------------|----------|-------|
| 1 | 1+2+3 | Benchmark index + decontaminate + pretokenize | ~7 hrs | 0 | 4 |
| 2 | 4 | Stream held-out (all 20 languages) | ~2 hrs | 0 | 1 |
| **Total** | | | **~9 hrs** | **0** | **4** |

### Curriculum Mode (all stages)

| Step | Stage | What | Wall-clock | A100-hrs | Nodes |
|------|-------|------|------------|----------|-------|
| 1 | 1+2+3 | Static pipeline (per node, 4 in parallel) | ~7 hrs | 0 | 4 |
| 2 | 4 | Stream held-out (1 node, all languages) | ~2 hrs | 0 | 1 |
| 3 | A+B+C | Teacher annotation + student model | ~8 hrs | 58 | 1 (8 GPUs) |
| 4 | D+3 | Score + index + pretokenize (4 in parallel) | ~12 hrs | 96/node | 4 (8 GPUs each) |
| **Total** | | | **~29 hrs** | **~442** | **4 nodes** |

Wall-clock is dominated by the sequential dependency of STEP 3 (curriculum A+B+C on one node) before STEP 4 can begin. STEPs 1, 2, and 4 are fully parallel across nodes.

### Hardware

- 4 nodes × 8 A100-80GB GPUs
- 128 CPUs per node (Intel Xeon Platinum 8358 @ 2.60 GHz)
- ~512 GB RAM per node

---

## Disk Budget

| Phase | Disk usage | Notes |
|-------|-----------|-------|
| After STEP 1 (decontamination) | ~1.47 TB total (across 4 nodes) | ~370 GB/node, 20 languages + English |
| With `--curriculum-prep` | +1.47 TB HF upload | Raw Parquet uploaded to HF before deletion |
| After curriculum Stage D | ~2.5 TB | Indexed Parquet + annotations + student model |
| Peak during Stage 3 (curriculum pretokenize) | ~3.8 TB | Indexed shards + Arrow being built |
| After cleanup | ~0 | All uploaded to HuggingFace, local files deleted |

**Static mode per-node peak**: ~320 GB (storage-optimized, processes one language at a time). Without upload (`--no-upload`): ~880 GB/node.

### Recommended Mount Points

| Mount | Use for |
|-------|---------|
| `/mnt/ssd-3` (19 TB free) | `OUTPUT_DIR` — fast SSD, ample space |
| `/mnt/ssd-cluster` (500 GB) | Single-language smoke tests |

---

## HuggingFace Datasets

Large-scale FineWeb outputs are uploaded to the `Beetle-Data` organization. Human-scale BabyBabel data and tokenizers are uploaded to `Beetle-HumanScale`.

### Human-Scale (Beetle-HumanScale)

```
Beetle-HumanScale/{lang}-100M                         # Monolingual Arrow (100M tokens)
Beetle-HumanScale/{l1}-50M-{tok_pair}                  # Bilingual L1 Arrow (50M tokens)
Beetle-HumanScale/{l2}-for-{l1}-50M-{tok_pair}         # Bilingual L2 Arrow (50M tokens)
Beetle-HumanScale/{l1}-80M-{tok_pair}                  # B4 Classroom L1 Arrow (80M tokens)
Beetle-HumanScale/{l2}-for-{l1}-20M-{tok_pair}         # B4 Classroom L2 Arrow (20M tokens)
Beetle-HumanScale/{lang}-33M-{tok_triple}              # Trilingual Arrow (33M tokens each)
Beetle-HumanScale/bpe-humanscale-{lang}                # Monolingual tokenizer
Beetle-HumanScale/bpe-humanscale-{sorted_pair}         # Bilingual tokenizer
Beetle-HumanScale/bpe-humanscale-{sorted_triple}       # Trilingual tokenizer
```

### Static Mode

```
Beetle-Data/{lang}-28B              # Pretokenized Arrow (L1 side, 28B streamed)
Beetle-Data/en-for-{lang}-28B       # Pretokenized Arrow (English side)
Beetle-Data/{lang}-raw-28B          # Raw decontaminated Parquet (with --curriculum-prep)
```

### Curriculum Mode

```
Beetle-Data/{lang}-indexed-28B      # Hive-partitioned Parquet
                                    #   lang={xx}/topic={0..199}/shard_*.parquet
                                    #   Columns: text, url, doc_id, quality, difficulty, topic_id
Beetle-Data/{lang}-curriculum-28B   # Pretokenized curriculum Arrow
                                    #   Columns: input_ids (513-tok chunks), quality, difficulty, topic_id
Beetle-Data/beetlestream-annotations    # Teacher annotation JSONL + feature Parquet (500K docs)
Beetle-Data/beetlestream-student-model  # sklearn regressors, embedder config, cluster centroids
```

### Loading Datasets

```python
from datasets import load_dataset, load_from_disk

# Static pretokenized
ds = load_dataset("Beetle-Data/pl-28B", split="train")

# Curriculum indexed shards (with pedagogical metadata)
ds = load_dataset("Beetle-Data/pl-indexed-28B", split="train")

# Curriculum pretokenized (for BeetleLM curriculum training)
ds = load_dataset("Beetle-Data/pl-curriculum-28B", split="train")
```

---

## Evaluation Benchmarks (Decontamination)

All evaluation text is indexed as 13-grams. Any training document containing a 13-gram overlap with any benchmark sentence is **discarded entirely**. The benchmark index is built in Stage 1 and shared across all pipeline runs.

### Minimal Pairs (BLiMP Family)

| Benchmark | HuggingFace ID | Language |
|-----------|---------------|----------|
| BLiMP | `nyu-mll/blimp` | English |
| ZhoBLiMP | `Junrui1202/zhoblimp` | Chinese |
| BLiMP-NL | `juletxara/blimp-nl` | Dutch |
| RuBLiMP | `RussianNLP/rublimp` | Russian |
| TurBLiMP | `juletxara/turblimp` | Turkish |
| JBLiMP | `polm-stability/jblimp` | Japanese |
| SLING | `suchirsalhan/SLING` | Chinese |
| CLiMP | `suchirsalhan/CLiMP` | Chinese |
| MultiBLiMP | `jumelet/multiblimp` | NL, DE, FR, FA, BG |

### Other Benchmarks

| Benchmark | Source | Type | Text Columns |
|-----------|--------|------|--------------|
| FLORES-200 | `crystina-z/flores200` | Perplexity (devtest) | `sentence_{lang_tag}` |
| XCOMPS | `fpadovani/xcomps-dataset` | Minimal pairs | `acceptable_sent`, `unacceptable_sent` |
| XNLI | `xnli` | NLI (validation) | `premise`, `hypothesis` |
| UD Treebanks | `universal_dependencies` | Syntax | `text` |
| MECO-L2 | [`suchirsalhan/MECO`](https://huggingface.co/datasets/suchirsalhan/MECO) | Reading times | `FullText` |

MECO-L2 is downloaded directly from HuggingFace at Stage 1 — no local file or `--project-root` required. Columns available: `itemid`, `wordnum`, `word`, `FullText`, `FullTextMarked`.

---

## Post-Hoc Contamination Analysis (Infinigram)

Infinigram-style querying enables analysis of whether model performance on MECO reading times correlates with training data presence — i.e., whether words or passages that appear in the training corpus are processed faster.

```bash
# 1. Check if a string would have been flagged as contaminated
python -m pipeline.post_hoc check \
    --index pipeline_output/benchmark_13gram.pkl \
    --text "The cat sat on the mat and watched the birds."

# 2a. Scan local corpus (immediately after Stage 2, before HF upload)
python -m pipeline.post_hoc scan \
    --lang de --text "example eval stimulus text" \
    --output-dir pipeline_output

# 2b. Scan HuggingFace corpus (after upload, no local files needed)
python -m pipeline.post_hoc scan \
    --lang de --text "example eval stimulus text" \
    --hf-user Beetle-Data --hf-suffix 28B

# 3. Batch analysis from a file (one text per line)
python -m pipeline.post_hoc batch \
    --index pipeline_output/benchmark_13gram.pkl \
    --input meco_stimuli.txt \
    --lang de --output-dir pipeline_output \
    --output results.json
```

The `check` mode verifies whether a given text's 13-grams appear in the benchmark index (i.e., whether it was excluded from training). The `scan` mode searches the training corpus for n-gram matches and returns document IDs, URLs, and surrounding context — enabling Infinigram-style frequency analysis over the actual training data.

---

## Output Formats

### Stage 2 Parquet (decontaminated, 28B tokens per language)

```
columns:     text (utf8), url (utf8), doc_id (int64), word_count (int32)
compression: Snappy
shard size:  50,000 documents per file
manifest:    {lang}_manifest.json → {shard_file: [first_doc_id, last_doc_id]}
stats:       {lang}_stats.json → contamination rate, words_accumulated, etc.
```

### Stage 3 Arrow — Static Mode (24B token bilingual pair)

```
columns:    input_ids (list<int32>, length 513)
format:     HuggingFace Arrow (datasets.load_from_disk compatible)
chunk:      512 tokens input + 1 label = 513 total
packing:    no cross-document token bleeding (remainder discarded at doc boundary)
```

Token budget per bilingual pair:

| Side | Ratio | Tokens (24B total) |
|------|-------|-------------------|
| L1 | 1/2 | 12B |
| English | 1/2 | 12B |

### Stage 3 Arrow — Curriculum Mode

```
columns:    input_ids (list<int32>, length 513),
            quality (float32, 0-5),
            difficulty (int8, 1-3),
            topic_id (int16, 0-199)
```

### Curriculum Indexed Parquet (Stage D output)

```
columns:     text, url, doc_id, quality (float32), difficulty (int8), topic_id (int16)
partitioning: Hive: lang={xx}/topic={0..199}/shard_*.parquet
shard size:  10,000 documents per file
metadata:    manifest.json, cluster_centroids.pkl, topic_distribution.json
```

---

## Smoke Test (Local CPU, ~10 min)

Exercises all stages with tiny data volumes — no GPU, no HF upload, minimal disk.

```bash
# Stage 1: Build benchmark index (loads MECO from HuggingFace, ~5 min)
python -m pipeline.run_pipeline --stage 1 --skip-disk-check

# Stage 2: Decontaminate a small sample (~2 min)
python -m pipeline.run_pipeline --stage 2 --lang pl --target-words 10000 \
    --index pipeline_output/benchmark_13gram.pkl \
    --num-workers 4 --skip-disk-check

# Stage 3: Pretokenize, skip upload (~1 min)
python -m pipeline.run_pipeline --stage 3 --lang pl --no-upload --no-cleanup \
    --num-workers 4 --skip-disk-check

# Post-hoc check
python -m pipeline.post_hoc check \
    --index pipeline_output/benchmark_13gram.pkl \
    --text "The quick brown fox jumps over the lazy dog."
```

Expected outputs:
- `pipeline_output/benchmark_13gram.pkl` — benchmark index (includes MECO from HF)
- `pipeline_output/decontaminated/pl/*.parquet` — at least 1 Parquet shard
- `pipeline_output/pretokenized/pl-en/` — Arrow dataset

---

## Run Individual Stages

```bash
# Stage 1: Build benchmark index (~5 min)
python -m pipeline.benchmark_index \
    --output pipeline_output/benchmark_13gram.pkl

# Stage 2: Stream + decontaminate one language (~45-60 min)
python -m pipeline.decontaminate_stream \
    --lang pl --index pipeline_output/benchmark_13gram.pkl \
    --output-dir pipeline_output

# Stage 3: Pretokenize + upload + cleanup (~20 min)
python -m pipeline.pretokenize_arrow \
    --lang pl --output-dir pipeline_output --hf-user Beetle-Data

# Stage 4: Stream held-out data
python -m pipeline.stream_held_out \
    --lang de --output-dir pipeline_output --n-docs 10000

# Curriculum Stage A (requires vLLM server)
python -m pipeline.teacher_annotate \
    --output-dir pipeline_output \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --vllm-url http://localhost:8000/v1

# Curriculum Stage B: Feature transform
python -m pipeline.feature_transform --output-dir pipeline_output

# Curriculum Stage C: Student model training
python -m pipeline.student_model --output-dir pipeline_output

# Curriculum Stage D: Score + index
python -m pipeline.score_and_index \
    --output-dir pipeline_output \
    --beetlestream-config configs/beetlestream_curriculum.yaml
```

---

## Run via Orchestrator

```bash
# Full static pipeline, all languages
python -m pipeline.run_pipeline

# Full static pipeline, specific languages
python -m pipeline.run_pipeline --lang pl nl es

# Stage 1 only (build benchmark index)
python -m pipeline.run_pipeline --stage 1 --skip-disk-check

# Stage 2 only (decontaminate), specific language
python -m pipeline.run_pipeline --stage 2 --lang pl \
    --index pipeline_output/benchmark_13gram.pkl

# Curriculum stages only
python -m pipeline.run_pipeline --stage A B C \
    --beetlestream-config configs/beetlestream_curriculum.yaml

# No upload, no cleanup (for debugging)
python -m pipeline.run_pipeline --lang pl --no-upload --no-cleanup

# Curriculum prep: also upload raw Parquet to HF after Stage 2
python -m pipeline.run_pipeline --lang pl --curriculum-prep
```

---

## SLURM Cluster

```bash
# Full static pipeline across 4 nodes
sbatch scripts/launch_decontaminate.sh
sbatch scripts/launch_pretokenize.sh

# Or the full BeetleStream v2 curriculum pipeline (sequential steps)
OUTPUT_DIR=/mnt/ssd-3/beetle-data sbatch scripts/launch_beetlestream.sh
```

Node assignments (core languages):
```
Node 0: pl, nl, es, el, ja
Node 1: fr, zh, de, it, eu
Node 2: tr, id, tl, fa, hi
Node 3: ta, sv, ru, ca, ar
```

---

## Configuration

All pipeline settings live in `pipeline/config.py`:

**User-configurable constants** (top of file — only these need changing):
- `STREAM_TOKENS_PER_LANG` — clean tokens to collect per language side (default: 28B)
- `TARGET_TOKENS_PER_LANG` — tokens used for training per pair (default: 24B)
- `TRAINING_DATASET_L1` — non-English HuggingFace dataset (default: FineWeb-2)
- `TRAINING_DATASET_EN` — English HuggingFace dataset (default: FineWeb-Edu)
- `TRAINING_TEXT_FIELD` — field name for document text (default: `"text"`)

**Internal configuration** (dataclasses):
- `PipelineConfig` — token targets, sequence length, worker count, shard sizes
- `BeetleStreamConfig` — teacher model, student model, heuristic thresholds, indexing params
- `StreamMode` — enum for `static`, `random_stream`, `curriculum`
- `LANG_REGISTRY` — per-language metadata (FineWeb-2 names, FLORES tags, tier)
- `BENCHMARK_DEFS` — evaluation benchmarks and their HuggingFace IDs
- `NODE_ASSIGNMENTS` — SLURM node-to-language mapping

Key defaults:
- Sequence length: 512 (chunk length: 513 = seq_len + 1)
- N-gram size: 13 (for decontamination)
- Workers: 24 per stage
- Shard size: 50,000 documents/file (static), 10,000/file (curriculum indexed)
- Teacher model: `meta-llama/Meta-Llama-3-70B-Instruct`
- Student embedding: `intfloat/multilingual-e5-base` (768-dim)
- Topic clusters: 200 per language

---

## FineWeb-2 Language Mappings

```
pl → pol_Latn    nl → nld_Latn    es → spa_Latn    el → ell_Grek
ja → jpn_Jpan    fr → fra_Latn    zh → cmn_Hani    de → deu_Latn
it → ita_Latn    eu → eus_Latn    tr → tur_Latn    id → ind_Latn
tl → fil_Latn    fa → fas_Arab    hi → hin_Deva    ta → tam_Taml
sv → swe_Latn    ru → rus_Cyrl    ca → cat_Latn    ar → arb_Arab
en → FineWeb-Edu (HuggingFaceFW/fineweb-edu)
```

---

## Project Structure

```
beetle-data/
  pipeline/
    config.py                  # USER-CONFIGURABLE constants + language registry
    utils.py                   # Text normalization, n-gram extraction
    benchmark_index.py         # Stage 1: 13-gram index (MECO loaded from HF)
    decontaminate_stream.py    # Stage 2: Stream + decontaminate + Parquet
    pretokenize_arrow.py       # Stage 3: Tokenize + pack + Arrow (24B cutoff)
    stream_held_out.py         # Stage 4: Held-out data beyond training cutoff
    teacher_annotate.py        # Stage A: Pedagogical annotation via vLLM
    feature_transform.py       # Stage B: Annotation JSONL → feature Parquet
    student_model.py           # Stage C: Train embedding-based quality scorer
    heuristic_filters.py       # Stage D pass 1: Fast text triage
    score_and_index.py         # Stage D pass 2: Scoring + topic clustering
    post_hoc.py                # Infinigram-style contamination analysis (local + HF)
    run_pipeline.py            # CLI orchestrator (all stages)
    __init__.py
  configs/
    beetlestream_curriculum.yaml  # BeetleStream v2 curriculum pipeline config
    fineweb_bilingual.yaml        # 12B per language bilingual tokenizer config
    fineweb_monolingual.yaml      # 24B monolingual experiment config
    fineweb_trilingual.yaml       # 8B per language trilingual config
    fineweb_tokenizer.yaml        # Tokenizer training config
  scripts/
    launch_full_pipeline.sh       # STEP 1: Static pipeline per node (Stages 1+2+3)
    launch_held_out.sh            # STEP 2: Held-out streaming, all languages, 1 node
    launch_curriculum_abc.sh      # Curriculum STEP 3: Stages A+B+C, 1 node
    launch_curriculum_d3.sh       # Curriculum STEP 4: Stages D+3, per node
    launch_beetlestream.sh        # Full BeetleStream v2 end-to-end (SLURM)
    launch_decontaminate.sh       # SLURM: Stages 1+2 across 4 nodes
    launch_pretokenize.sh         # SLURM: Stage 3 across 4 nodes
    launch_extensions.sh          # Extension language pipeline
    launch_low_resource.sh        # Low-resource language pipeline
    verify_output.py              # End-to-end validation
  tok/                         # Tokenizer training scripts (Step 0 — run before Stage 3)
  data/                        # Legacy decontamination scripts
```
