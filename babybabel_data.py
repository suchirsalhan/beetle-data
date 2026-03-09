"""
build_babybabel.py
==================
Download all BabyLM source datasets, tokenize in parallel across all CPU cores,
slice each language independently at 33M / 50M / 100M token thresholds, then
push 27 monolingual datasets to HuggingFace as:

  BeetleLM/BabyBabel-{lang}-33
  BeetleLM/BabyBabel-{lang}-50
  BeetleLM/BabyBabel-{lang}-100

  e.g. BeetleLM/BabyBabel-bul-33, BeetleLM/BabyBabel-eng-100, ...

Usage
-----
  pip install datasets huggingface_hub tiktoken tqdm
  huggingface-cli login          # or set HF_TOKEN env var
  python build_babybabel.py

Parallelism strategy (128-core Xeon Platinum 8358, 2 NUMA nodes)
-----------------------------------------------------------------
  * Downloads     — 9 threads (ThreadPoolExecutor); I/O bound, no GIL issue
  * Tokenization  — single persistent Pool(126 workers) shared across all langs;
                    avoids 9× spawn overhead; large BATCH_SIZE (16k) keeps
                    workers fed and amortises IPC overhead
  * HF pushes     — up to 9 concurrent threads; network-bound
  * A100s         — tiktoken is CPU-only; GPUs are not used here.
                    They will be useful when you train on these datasets.
"""

import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import tiktoken
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

TARGETS = {
    "33":  33_000_000,
    "50":  50_000_000,
    "100": 100_000_000,
}

TEXT_FIELD   = "text"
BATCH_SIZE   = 16_000                      # large batches → less IPC overhead
NUM_WORKERS  = max(1, mp.cpu_count() - 2)  # leave 2 cores free for I/O / main
MAX_PUSH_THREADS = 9                        # one per language; HF rate-limits above ~10


# ── Token counting (runs in worker processes) ─────────────────────────────────

def _count_batch(texts: list[str]) -> list[int]:
    """Count tokens for a batch of texts. Runs inside persistent worker processes."""
    enc = tiktoken.get_encoding("cl100k_base")
    return [len(enc.encode(t, disallowed_special=())) for t in texts]


# ── Step 1 — Parallel download ────────────────────────────────────────────────

def _download_one(lang_and_id: tuple[str, str]) -> tuple[str, list[dict]]:
    lang, ds_id = lang_and_id
    ds = load_dataset(ds_id, split="train", trust_remote_code=True)
    rows = [
        {"text": row[TEXT_FIELD], "language": lang, "source": ds_id}
        for row in ds
        if row.get(TEXT_FIELD, "").strip()
    ]
    return lang, rows


def download_all() -> dict[str, list[dict]]:
    """
    Download all 9 datasets concurrently (I/O bound → ThreadPoolExecutor).
    HuggingFace caches shards locally; subsequent runs skip the download.
    """
    all_data: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as executor:
        futures = {
            executor.submit(_download_one, item): item[0]
            for item in SOURCES.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  downloading", unit="lang"):
            lang = futures[future]
            try:
                lang, rows = future.result()
                all_data[lang] = rows
                tqdm.write(f"  ✓ {lang:4s}  {len(rows):>10,} rows")
            except Exception as exc:
                tqdm.write(f"  ✗ {lang:4s}  FAILED: {exc}")
    return all_data


# ── Step 2 — Parallel tokenization (single persistent pool) ──────────────────

def tokenize_all(
    all_data: dict[str, list[dict]],
    pool: mp.Pool,
) -> dict[str, list[int]]:
    """
    Tokenize every language using one shared Pool — avoids 9× spawn overhead.
    Returns {lang: [token_count_per_row]}.
    """
    token_counts: dict[str, list[int]] = {}
    total_rows = sum(len(v) for v in all_data.values())

    with tqdm(total=total_rows, unit="rows", desc="  tokenising") as pbar:
        for lang, rows in all_data.items():
            texts   = [r["text"] for r in rows]
            batches = [texts[i : i + BATCH_SIZE]
                       for i in range(0, len(texts), BATCH_SIZE)]
            counts: list[int] = []
            for batch_counts in pool.imap(_count_batch, batches):
                counts.extend(batch_counts)
                pbar.update(len(batch_counts))
            token_counts[lang] = counts

    return token_counts


# ── Step 3 — Per-language slice ───────────────────────────────────────────────

def slice_language(
    rows: list[dict],
    counts: list[int],
) -> dict[str, list[dict]]:
    """
    For a single language, accumulate rows until each token threshold is hit.
    Returns {target_key: [rows]}.  If corpus is exhausted before a threshold,
    returns all available rows for that target (with a warning).
    """
    sorted_targets = sorted(TARGETS.items(), key=lambda kv: kv[1])
    buffers: dict[str, list[dict]] = {k: [] for k in TARGETS}
    filled:  set[str] = set()
    running = 0

    for row, n_tok in zip(rows, counts):
        if n_tok == 0:
            continue
        running += n_tok

        for key, _ in sorted_targets:
            if key not in filled:
                buffers[key].append(row)

        for key, threshold in sorted_targets:
            if key not in filled and running >= threshold:
                filled.add(key)

        if len(filled) == len(TARGETS):
            break

    # Warn for any threshold not reached
    for key, threshold in sorted_targets:
        if key not in filled:
            tqdm.write(
                f"    WARNING: corpus exhausted at {running:,} tok "
                f"-- {key}M target not met, using all {len(buffers[key]):,} rows"
            )

    return buffers


# ── Step 4 — Push to Hub (concurrent) ────────────────────────────────────────

def _push_one(args: tuple) -> str:
    lang, key, threshold, rows, token = args
    repo_id = f"{HF_ORG}/BabyBabel-{lang}-{key}"
    api = HfApi()

    ds_dict = DatasetDict({"train": Dataset.from_list(rows)})
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as exc:
        tqdm.write(f"    [warn] create_repo {repo_id}: {exc}")

    ds_dict.push_to_hub(
        repo_id,
        token=token,
        commit_message=f"Add BabyBabel-{lang}-{key} ({threshold // 1_000_000}M-token slice)",
    )
    return repo_id


def push_all(all_buffers: dict[str, dict[str, list[dict]]]):
    """Push all 27 datasets concurrently (network-bound → ThreadPoolExecutor)."""
    token = os.environ.get("HF_TOKEN")

    push_jobs = [
        (lang, key, threshold, all_buffers[lang][key], token)
        for lang in all_buffers
        for key, threshold in TARGETS.items()
    ]

    print(f"\n  Pushing {len(push_jobs)} datasets ({MAX_PUSH_THREADS} concurrent threads) ...")
    with ThreadPoolExecutor(max_workers=MAX_PUSH_THREADS) as executor:
        futures = {executor.submit(_push_one, job): job for job in push_jobs}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  pushing", unit="repo"):
            job = futures[future]
            lang, key = job[0], job[1]
            try:
                repo_id = future.result()
                tqdm.write(f"  ✓ pushed {repo_id}  ({len(job[3]):,} rows)")
            except Exception as exc:
                tqdm.write(f"  ✗ FAILED {HF_ORG}/BabyBabel-{lang}-{key}: {exc}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN not set -- "
            "make sure you are logged in via `huggingface-cli login`\n"
        )

    # 1. Download all languages concurrently ───────────────────────────────────
    print(f"Step 1/4  Downloading {len(SOURCES)} datasets concurrently ...")
    all_data = download_all()
    total_rows = sum(len(v) for v in all_data.values())
    print(f"\n  Total rows: {total_rows:,}  |  Workers: {NUM_WORKERS}\n")

    # 2. Tokenise with a single persistent pool ────────────────────────────────
    print(f"Step 2/4  Tokenising ({NUM_WORKERS} workers, batch={BATCH_SIZE:,}) ...")
    with mp.Pool(processes=NUM_WORKERS) as pool:
        token_counts = tokenize_all(all_data, pool)

    lang_totals = {lang: sum(tc) for lang, tc in token_counts.items()}
    grand_total = sum(lang_totals.values())
    print(f"\n  Corpus total: {grand_total:,} tokens")
    for lang, n in sorted(lang_totals.items(), key=lambda kv: -kv[1]):
        can_hit = [k for k, t in TARGETS.items() if n >= t]
        hits = ", ".join(can_hit) if can_hit else "none (below 33M)"
        print(f"    {lang}: {n:>14,} tok  →  targets reachable: {hits}")

    # 3. Slice each language independently ─────────────────────────────────────
    print("\nStep 3/4  Slicing per language ...")
    all_buffers: dict[str, dict[str, list[dict]]] = {}
    for lang, rows in all_data.items():
        all_buffers[lang] = slice_language(rows, token_counts[lang])
        sizes = {k: len(v) for k, v in all_buffers[lang].items()}
        print(f"  {lang}: " + "  ".join(f"{k}M→{v:,}rows" for k, v in sizes.items()))

    # 4. Push all 27 datasets concurrently ─────────────────────────────────────
    print("\nStep 4/4  Pushing to HuggingFace ...")
    push_all(all_buffers)

    print("\nAll done!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe on macOS/Windows/Linux
    main()
