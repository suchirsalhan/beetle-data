# Beetle-Data: {lang_name} ({lang_code}) — Indexed 24B

Quality-scored, topic-clustered Parquet dataset for BeetleStream v2 curriculum training.

## Overview

| Property | Value |
|----------|-------|
| Language | {lang_name} ({lang_code}) |
| Total documents | {total_docs:,} |
| Total tokens (estimated) | ~{total_tokens} |
| Quality score range | 0-5 (additive pedagogical quality) |
| Difficulty levels | 1 (elementary), 2 (middle), 3 (high) |
| Topic clusters | {n_topics} |
| Shard size | ~10,000 documents |

## Data Source

Filtered and quality-scored from FineWeb-2 (`{fw2_name}`) via the BeetleStream v2 pipeline:
1. Decontaminated against 16+ evaluation benchmarks (13-gram overlap detection)
2. Heuristic-filtered (stopword density, readability, script consistency, repetition)
3. Quality-scored by a student model trained on Llama-3-70B teacher annotations
4. Topic-clustered via k-means on multilingual-e5-base embeddings

## Schema

```
text:       utf8       — Document text
url:        utf8       — Source URL
doc_id:     int64      — Unique document identifier
quality:    float32    — Pedagogical quality score (0-5)
difficulty: int8       — Readability difficulty level (1-3)
topic_id:   int16      — Topic cluster ID (0-{max_topic_id})
engagement: float32    — Student engagement score (0-1)
```

## Directory Structure

```
{lang_code}-indexed-24B/
  topic=0/
    shard_00000.parquet
    shard_00001.parquet
  topic=1/
    shard_00000.parquet
  ...
  topic={max_topic_id}/
    shard_00000.parquet
```

## Quality Distribution

| Score | Count | Percentage |
|-------|-------|-----------|
{quality_distribution_table}

## Difficulty Distribution

| Level | Count | Percentage |
|-------|-------|-----------|
{difficulty_distribution_table}

## Topic Distribution (Top 20)

| Topic ID | Documents | Avg Quality | Difficulty 1 | Difficulty 2 | Difficulty 3 |
|----------|-----------|-------------|-------------|-------------|-------------|
{topic_distribution_table}

## Cross-Lingual Topic Coverage

Topics with >=1,000 documents in both {lang_name} and English:
{cross_lingual_topics}

## Usage

### Load with HuggingFace Datasets

```python
from datasets import load_dataset

# Load all data
ds = load_dataset("Beetle-Data/{lang_code}-indexed-24B")

# Load specific topic
ds_topic = load_dataset("Beetle-Data/{lang_code}-indexed-24B",
                        data_files="topic=42/*.parquet")

# Filter by quality
high_quality = ds.filter(lambda x: x["quality"] >= 4)
```

### For BeetleLM Training

Use the pretokenized version:
```python
ds = load_dataset("Beetle-Data/{lang_code}-curriculum-24B")
```

## License

This dataset is derived from FineWeb-2 and subject to its license terms.

## Citation

```bibtex
@misc{{beetlestream2024,
  title={{BeetleStream v2: Pedagogical Multilingual Curriculum Engine}},
  author={{Beetle-Data}},
  year={{2024}},
  howpublished={{https://huggingface.co/datasets/Beetle-Data/{lang_code}-indexed-24B}}
}}
```
