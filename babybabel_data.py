"""
build_babybabel.py
==================
Download all BabyLM source datasets up-front, tokenize in parallel across all
CPU cores, accumulate to 33M / 50M / 100M token thresholds, then push each
slice to HuggingFace as:
  BeetleLM/BabyBabel-fr-33
  BeetleLM/BabyBabel-fr-50
  BeetleLM/BabyBabel-fr-100

Usage
-----
  pip install datasets huggingface_hub tiktoken tqdm
  huggingface-cli login          # or set HF_TOKEN env var
  python build_babybabel.py

Speed-ups vs the streaming version
-----------------------------------
  * Pre-download  — all shards cached locally; no repeated HF round-trips
  * Multiprocessing — tokenization spread across all CPU cores via Pool.imap
    (tiktoken releases the GIL, so true parallelism is achieved)
"""

import os
import multiprocessing as mp

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

TEXT_FIELD  = "text"
BATCH_SIZE  = 2_000                       # rows per multiprocessing chunk
NUM_WORKERS = max(1, mp.cpu_count() - 1)  # leave 1 core free for I/O


# ── Token counting (runs in worker processes) ─────────────────────────────────

def _count_batch(texts: list[str]) -> list[int]:
    """Count tokens for a batch of texts. Called inside worker processes."""
    enc = tiktoken.get_encoding("cl100k_base")
    return [len(enc.encode(t, disallowed_special=())) for t in texts]


# ── Step 1 — Pre-download ─────────────────────────────────────────────────────

def download_all() -> dict[str, list[dict]]:
    """
    Download every source dataset in full (non-streaming) and return as a dict
    of {lang: list_of_rows}. HuggingFace caches shards locally so re-runs
    skip the download entirely.
    """
    all_data: dict[str, list[dict]] = {}
    for lang, ds_id in SOURCES.items():
        print(f"  down  {ds_id} ...", end=" ", flush=True)
        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            rows = [
                {"text": row[TEXT_FIELD], "language": lang, "source": ds_id}
                for row in ds
                if row.get(TEXT_FIELD, "").strip()
            ]
            all_data[lang] = rows
            print(f"{len(rows):,} rows")
        except Exception as exc:
            print(f"FAILED -- {exc}")
    return all_data


# ── Step 2 — Parallel tokenization ───────────────────────────────────────────

def tokenize_parallel(rows: list[dict]) -> list[int]:
    """
    Return a list of token counts (one per row) using a multiprocessing Pool.
    Processes rows in BATCH_SIZE chunks spread across NUM_WORKERS cores.
    """
    texts   = [r["text"] for r in rows]
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    counts: list[int] = []
    with mp.Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(texts), unit="rows", desc="    tokenising", leave=False) as pbar:
            for batch_counts in pool.imap(_count_batch, batches):
                counts.extend(batch_counts)
                pbar.update(len(batch_counts))
    return counts


# ── Step 3 — Interleave & slice ───────────────────────────────────────────────

def interleave_and_slice(
    all_data: dict[str, list[dict]],
    token_counts: dict[str, list[int]],
) -> dict[str, list[dict]]:
    """
    Round-robin across languages and accumulate rows until each token
    threshold is reached. Returns {target_key: [rows]}.
    """
    sorted_targets = sorted(TARGETS.items(), key=lambda kv: kv[1])
    max_tokens     = sorted_targets[-1][1]

    buffers: dict[str, list[dict]] = {k: [] for k in TARGETS}
    filled:  set[str]              = set()
    total_tokens = 0

    iters = {
        lang: iter(zip(rows, token_counts[lang]))
        for lang, rows in all_data.items()
    }

    print(f"\n  Round-robining {len(iters)} languages ...")
    with tqdm(total=max_tokens, unit="tok", unit_scale=True, desc="  accumulating") as pbar:
        while iters:
            exhausted = []
            for lang, it in list(iters.items()):
                try:
                    row, n_tok = next(it)
                except StopIteration:
                    exhausted.append(lang)
                    continue

                if n_tok == 0:
                    continue

                total_tokens += n_tok
                pbar.update(n_tok)

                for key, _ in sorted_targets:
                    if key not in filled:
                        buffers[key].append(row)

                for key, threshold in sorted_targets:
                    if key not in filled and total_tokens >= threshold:
                        filled.add(key)
                        tqdm.write(
                            f"  threshold {key}M reached  "
                            f"({total_tokens:,} tokens, {len(buffers[key]):,} rows)"
                        )

                if len(filled) == len(TARGETS):
                    return buffers

            for lang in exhausted:
                del iters[lang]

    # Corpus exhausted before every threshold was met
    for key, threshold in sorted_targets:
        if key not in filled:
            tqdm.write(
                f"  WARNING: corpus exhausted at {total_tokens:,} tok "
                f"-- {key}M target not met, using all available data"
            )

    return buffers


# ── Step 4 — Push to Hub ──────────────────────────────────────────────────────

def push_datasets(buffers: dict[str, list[dict]]):
    api   = HfApi()
    token = os.environ.get("HF_TOKEN")

    for key, threshold in sorted(TARGETS.items(), key=lambda kv: kv[1]):
        repo_id = f"{HF_ORG}/BabyBabel-fr-{key}"
        rows    = buffers[key]

        print(f"\n  pushing {repo_id}  ({len(rows):,} rows) ...")

        ds_dict = DatasetDict({"train": Dataset.from_list(rows)})

        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as exc:
            print(f"    [warn] create_repo: {exc}")

        ds_dict.push_to_hub(
            repo_id,
            token=token,
            commit_message=f"Add BabyBabel-fr-{key} ({threshold // 1_000_000}M tokens)",
        )
        print(f"    pushed {repo_id}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN not set -- "
            "make sure you are logged in via `huggingface-cli login`\n"
        )

    # 1. Download ──────────────────────────────────────────────────────────────
    print(f"Step 1/4  Downloading {len(SOURCES)} datasets (cached after first run) ...")
    all_data = download_all()

    total_rows = sum(len(v) for v in all_data.values())
    print(f"\n  Total rows downloaded: {total_rows:,}")
    print(f"  Tokenization workers:  {NUM_WORKERS}\n")

    # 2. Tokenise in parallel ──────────────────────────────────────────────────
    print("Step 2/4  Parallel tokenization ...")
    token_counts: dict[str, list[int]] = {}
    for lang, rows in all_data.items():
        print(f"  {lang}  ({len(rows):,} rows)")
        token_counts[lang] = tokenize_parallel(rows)

    lang_totals = {lang: sum(tc) for lang, tc in token_counts.items()}
    grand_total = sum(lang_totals.values())
    print(f"\n  Corpus total: {grand_total:,} tokens")
    for lang, n in sorted(lang_totals.items(), key=lambda kv: -kv[1]):
        print(f"    {lang}: {n:,}")

    if grand_total < TARGETS["100"]:
        print(
            f"\n  WARNING: full corpus ({grand_total:,} tok) is smaller than "
            "the 100M target. BabyBabel-fr-100 will contain the full corpus."
        )

    # 3. Interleave & slice ────────────────────────────────────────────────────
    print("\nStep 3/4  Interleaving and slicing ...")
    buffers = interleave_and_slice(all_data, token_counts)

    # 4. Push ──────────────────────────────────────────────────────────────────
    print("\nStep 4/4  Pushing to HuggingFace ...")
    push_datasets(buffers)

    print("\nAll done!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # safe on macOS/Windows/Linux
    main()
