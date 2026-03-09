"""
build_babybabel.py
==================
Stream all BabyLM source datasets, tokenize in parallel across all CPU cores,
slice each language independently at 11 token thresholds, then push 99
monolingual datasets to HuggingFace as:

  BeetleLM/BabyBabel-{lang}-{target}
  e.g. BeetleLM/BabyBabel-bul-33M, BeetleLM/BabyBabel-eng-100M, ...

  9 languages × 11 targets = 99 datasets

Usage
-----
  pip install datasets huggingface_hub tiktoken tqdm
  huggingface-cli login          # or set HF_TOKEN env var
  python build_babybabel.py

Runtime estimate (128-core Xeon, 8× A100, data-centre uplink)
--------------------------------------------------------------
  Step 1+2  stream + tokenize + slice   ~3–5 min   (CPU-bound, 126 workers)
  Step 3    pre-create 99 repos         ~15–30 sec  (pure REST, 99 threads)
  Step 4a   stage 1 — small  (36 repos) ~6–10 min  (8 upload threads)
  Step 4b   stage 2 — medium (36 repos) ~8–14 min  (8 upload threads)
  Step 4c   stage 3 — large  (27 repos) ~8–14 min  (8 upload threads)
  ─────────────────────────────────────────────────────────────────────
  Total                                 ~25–45 min

  A100s: not used — tiktoken and HF uploads are CPU/network-bound.
  They will be useful when training on the resulting datasets.

Push strategy
-------------
  1. Pre-create all 99 repos in parallel (99 threads, pure API — no data).
     This removes create_repo() latency from inside each upload thread.
  2. Push in 3 stages sorted by dataset size (small → large).
     Smaller datasets go first so you get early validation and partial
     results faster.  Within each stage, 8 concurrent upload threads.

  Stage 1  targets 10M–33M   → 36 repos  (≤33M tokens each)
  Stage 2  targets 40M–66M   → 36 repos  (40M–66M tokens each)
  Stage 3  targets 70M–100M  → 27 repos  (70M–100M tokens each)
"""

import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

HF_ORG = "BeetleLM"

SOURCES = {
    "zho": "BabyLM-community/babylm-zho",
    "fas": "BabyLM-community/babylm-fas",
    "eng": "BabyLM-community/babylm-eng",
    "nld": "BabyLM-community/babylm-nld",
    "bul": "BabyLM-community/babylm-bul",
    "fra": "BabyLM-community/babylm-fra",
    "ind": "BabyLM-community/babylm-ind",
    "deu": "BabyLM-community/babylm-deu",
    "ukr": "BabyLM-community/babylm-ukr",
}

# Must be defined in ascending order — slice_language() depends on this.
TARGETS = {
    "10M":  10_000_000,
    "20M":  20_000_000,
    "30M":  30_000_000,
    "33M":  33_000_000,
    "40M":  40_000_000,
    "50M":  50_000_000,
    "60M":  60_000_000,
    "66M":  66_000_000,
    "70M":  70_000_000,
    "80M":  80_000_000,
    "100M": 100_000_000,
}

# Push stages — group targets by size so small datasets go first.
# Each stage is pushed fully before the next begins.
PUSH_STAGES = [
    ["10M", "20M", "30M", "33M"],          # stage 1 — small
    ["40M", "50M", "60M", "66M"],          # stage 2 — medium
    ["70M", "80M", "100M"],                # stage 3 — large
]

TEXT_FIELD        = "text"
BATCH_SIZE        = 16_000  # large batches → less IPC overhead on 128-core machine
NUM_WORKERS       = max(1, mp.cpu_count() - 2)
PUSH_THREADS      = 8       # concurrent upload threads per stage
PRECREATE_THREADS = 99      # one per repo — pure REST calls, very fast
SHUFFLE_SEED      = 42


# ── Tokenizer worker initializer ──────────────────────────────────────────────

ENC = None  # global per-worker encoder

def _init_worker():
    """Load tiktoken once per worker process — not once per batch."""
    global ENC
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")

def _count_batch(texts: list[str]) -> list[int]:
    """Tokenize a batch of texts. ENC is pre-loaded by _init_worker."""
    return [len(ENC.encode(t, disallowed_special=())) for t in texts]


# ── Step 1+2 — Stream, tokenize, and slice one language ──────────────────────

def process_language(
    lang: str,
    ds_id: str,
    pool: mp.Pool,
) -> dict[str, list[dict]]:
    """
    Stream one language, tokenize via the shared pool, accumulate into
    per-threshold buffers without ever holding the full dataset in RAM.

    No-overshoot rule: a row is only added to bucket[key] if
    running + n_tok <= threshold.  This keeps each dataset strictly
    within its token budget.

    Returns {target_key: [rows]}.
    """
    sorted_targets = sorted(TARGETS.items(), key=lambda kv: kv[1])

    buffers:  dict[str, list[dict]] = {k: [] for k in TARGETS}
    filled:   set[str] = set()
    running   = 0
    rows_seen = 0

    ds = load_dataset(ds_id, split="train", trust_remote_code=True, streaming=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=10_000)

    pending: list[dict] = []

    def flush(batch: list[dict]) -> bool:
        nonlocal running

        texts = [r["text"] for r in batch]
        counts: list[int] = []
        sub_batches = [texts[i : i + BATCH_SIZE]
                       for i in range(0, len(texts), BATCH_SIZE)]
        for result in pool.imap(_count_batch, sub_batches):
            counts.extend(result)

        for row, n_tok in zip(batch, counts):
            if n_tok == 0:
                continue

            # Add to every open bucket that won't be pushed over threshold
            for key, threshold in sorted_targets:
                if key not in filled and running + n_tok <= threshold:
                    buffers[key].append(row)

            running += n_tok

            for key, threshold in sorted_targets:
                if key not in filled and running >= threshold:
                    filled.add(key)

            if len(filled) == len(TARGETS):
                return True

        return False

    for hf_row in ds:
        text = hf_row.get(TEXT_FIELD, "").strip()
        if not text:
            continue
        rows_seen += 1
        pending.append({"text": text, "language": lang, "source": ds_id})

        if len(pending) >= BATCH_SIZE:
            if flush(pending):
                pending = []
                break
            pending = []

    if pending:
        flush(pending)

    for key, threshold in sorted_targets:
        if key not in filled:
            tqdm.write(
                f"  [{lang}] WARNING: corpus exhausted at {running:,} tok "
                f"-- {key} target not met, using all {len(buffers[key]):,} rows"
            )

    tqdm.write(
        f"  ✓ {lang:4s}  rows={rows_seen:,}  tokens≈{running:,}  "
        f"thresholds={len(filled)}/{len(TARGETS)}"
    )
    return buffers


# ── Step 3 — Pre-create all repos in parallel ─────────────────────────────────

def precreate_all_repos(all_buffers: dict[str, dict[str, list[dict]]]):
    """
    Fire off create_repo() for every non-empty repo concurrently.
    These are lightweight REST calls — no data transfer.
    All repos exist before any upload begins, so uploads never stall
    waiting for repo creation.
    """
    hf_token = os.environ.get("HF_TOKEN")

    repo_ids = [
        f"{HF_ORG}/BabyBabel-{lang}-{key}"
        for lang in all_buffers
        for key in TARGETS
        if all_buffers[lang].get(key)
    ]

    print(f"\n  Pre-creating {len(repo_ids)} repos ({PRECREATE_THREADS} threads) ...")
    t0 = time.time()

    def _create(repo_id: str):
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        exist_ok=True, token=hf_token)
        return repo_id

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=PRECREATE_THREADS) as executor:
        futures = {executor.submit(_create, r): r for r in repo_ids}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  creating repos", unit="repo"):
            try:
                future.result()
                ok += 1
            except Exception as exc:
                tqdm.write(f"  [warn] {futures[future]}: {exc}")
                failed += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {ok} created, {failed} failed")


# ── Step 4 — Staged concurrent push ──────────────────────────────────────────

def _push_one(args: tuple) -> tuple[str, int]:
    lang, key, threshold, rows, hf_token = args
    repo_id = f"{HF_ORG}/BabyBabel-{lang}-{key}"

    ds_dict = DatasetDict({"train": Dataset.from_list(rows)})
    ds_dict.push_to_hub(
        repo_id,
        token=hf_token,
        commit_message=(
            f"Add BabyBabel-{lang}-{key} "
            f"({threshold // 1_000_000}M-token slice, seed={SHUFFLE_SEED})"
        ),
        max_shard_size="500MB",  # keep shards manageable
    )
    return repo_id, len(rows)


def push_staged(all_buffers: dict[str, dict[str, list[dict]]]):
    """
    Push datasets in 3 stages (small → medium → large).
    Within each stage, PUSH_THREADS uploads run concurrently.
    Staging keeps the upload queue predictable and makes early
    results available sooner.
    """
    hf_token = os.environ.get("HF_TOKEN")
    total_pushed = 0
    t_start = time.time()

    for stage_num, stage_keys in enumerate(PUSH_STAGES, 1):
        stage_jobs = [
            (lang, key, TARGETS[key], all_buffers[lang][key], hf_token)
            for lang in all_buffers
            for key in stage_keys
            if all_buffers[lang].get(key)
        ]

        if not stage_jobs:
            continue

        total_rows = sum(len(j[3]) for j in stage_jobs)
        print(
            f"\n  Stage {stage_num}/{len(PUSH_STAGES)}  "
            f"targets={stage_keys}  "
            f"repos={len(stage_jobs)}  "
            f"rows={total_rows:,}  "
            f"threads={PUSH_THREADS}"
        )
        t_stage = time.time()

        with ThreadPoolExecutor(max_workers=PUSH_THREADS) as executor:
            futures = {executor.submit(_push_one, job): job for job in stage_jobs}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  stage {stage_num}", unit="repo"):
                job = futures[future]
                lang, key = job[0], job[1]
                try:
                    repo_id, n_rows = future.result()
                    total_pushed += 1
                    tqdm.write(f"  ✓  {repo_id}  ({n_rows:,} rows)")
                except Exception as exc:
                    tqdm.write(f"  ✗  BabyBabel-{lang}-{key}: {exc}")

        stage_elapsed = time.time() - t_stage
        print(f"  Stage {stage_num} done in {stage_elapsed/60:.1f} min")

    total_elapsed = time.time() - t_start
    print(f"\n  All {total_pushed} repos pushed in {total_elapsed/60:.1f} min total")


# ── Orchestration ─────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN not set -- "
            "make sure you are logged in via `huggingface-cli login`\n"
        )

    print(
        f"Config\n"
        f"  Languages      : {list(SOURCES)}\n"
        f"  Targets        : {list(TARGETS)}  ({len(TARGETS)} thresholds)\n"
        f"  Max repos      : {len(SOURCES) * len(TARGETS)}\n"
        f"  CPU workers    : {NUM_WORKERS}\n"
        f"  Batch size     : {BATCH_SIZE:,} rows\n"
        f"  Shuffle seed   : {SHUFFLE_SEED}\n"
        f"  Upload threads : {PUSH_THREADS} per stage\n"
        f"  Push stages    : {PUSH_STAGES}\n"
    )

    # ── Step 1+2 — Stream, tokenize, slice ────────────────────────────────────
    print("Step 1+2/4  Stream → tokenize → slice  (single persistent pool)\n")
    t0 = time.time()

    with mp.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        all_buffers: dict[str, dict[str, list[dict]]] = {}
        for lang, ds_id in SOURCES.items():
            tqdm.write(f"\n  [{lang}]  {ds_id}")
            all_buffers[lang] = process_language(lang, ds_id, pool)

    tok_elapsed = time.time() - t0
    print(f"\n  Tokenize+slice done in {tok_elapsed/60:.1f} min")

    # Summary table
    print("\n  Row counts per language × target:")
    print(f"  {'lang':<6}" + "".join(f"{k:>8}" for k in TARGETS))
    for lang in SOURCES:
        row_counts = [len(all_buffers[lang].get(k, [])) for k in TARGETS]
        print(f"  {lang:<6}" + "".join(f"{n:>8,}" for n in row_counts))

    # ── Step 3 — Pre-create all repos ─────────────────────────────────────────
    print("\nStep 3/4  Pre-creating all repos in parallel ...")
    precreate_all_repos(all_buffers)

    # ── Step 4 — Staged push ──────────────────────────────────────────────────
    print("\nStep 4/4  Staged push to HuggingFace ...")
    push_staged(all_buffers)

    print("\nAll done!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
