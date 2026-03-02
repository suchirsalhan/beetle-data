#!/usr/bin/env python3
import os
import time
import queue
import multiprocessing as mp
from pathlib import Path
from threading import Thread

import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer # Changed to AutoTokenizer for "Fast" support

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
SEQ_LEN = 512

# Total targets (36 Billion tokens)
TARGETS = {
    "es": 4_000_000_000, "fr": 4_000_000_000, "de": 4_000_000_000,
    "pl": 4_000_000_000, "tr": 4_000_000_000, "ar": 4_000_000_000,
    "zh": 4_000_000_000, "en": 8_000_000_000,
}

TMP = Path("ultra_tmp")
TMP.mkdir(exist_ok=True)

# Use the massive CPU count of the A100 node
CPU = os.cpu_count()
TOKEN_WORKERS = max(8, CPU - 8) # Leave a few cores for I/O and OS

api = HfApi()

# =====================================================
# WORKERS
# =====================================================

def stream(lang):
    """Generator for streaming dataset text."""
    ds_name = "HuggingFaceFW/fineweb-edu" if lang == "en" else "uonlp/CulturaX"
    ds_args = {"split": "train", "streaming": True}
    if lang != "en": ds_args["path"] = lang
    
    ds = load_dataset(ds_name, **ds_args)
    for ex in ds:
        t = ex.get("text")
        if t: yield t.replace("\n", " ")

def tokenizer_worker(in_q, out_q, tokenizer_path):
    """Uses Fast Tokenizer (Rust) for 10x speed over standard LlamaTokenizer."""
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    while True:
        text = in_q.get()
        if text is None: break
        # Fast tokenizer handles batching/multithreading internally if needed
        ids = tok.encode(text, add_special_tokens=False)
        out_q.put(ids)

def packer(token_q, block_q):
    buffer = []
    while True:
        ids = token_q.get()
        if ids is None: break
        buffer.extend(ids)
        while len(buffer) >= SEQ_LEN:
            block_q.put(np.array(buffer[:SEQ_LEN], dtype=np.int32))
            buffer = buffer[SEQ_LEN:]

def arrow_writer(lang, block_q, upload_q):
    shard_id, shard_tokens = 0, 0
    adaptive_target = 10_000_000 # Start with 10M tokens per shard
    arrays = []

    while True:
        block = block_q.get()
        if block is None: break
        arrays.append(pa.array([block]))
        shard_tokens += SEQ_LEN

        if shard_tokens >= adaptive_target:
            table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
            fname = TMP / f"{lang}_{shard_id}.arrow"
            with ipc.new_file(str(fname), table.schema) as writer:
                writer.write(table)
            upload_q.put((lang, shard_id, fname))
            shard_id += 1
            shard_tokens = 0
            arrays = []
            adaptive_target = min(int(adaptive_target * 1.1), 100_000_000) # Cap shard size at 100M tokens

def uploader(upload_q):
    while True:
        item = upload_q.get()
        if item is None: break
        lang, shard, fname = item
        repo = f"{HF_USER}/{lang}-512"
        create_repo(repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(fname), path_in_repo=f"train_{shard}.arrow", repo_id=repo, repo_type="dataset")
        os.remove(fname)

# =====================================================
# EXECUTION
# =====================================================

def run_language(lang):
    print(f"🚀 [STARTING] Language: {lang} | Target: {TARGETS[lang]} tokens")
    text_q, token_q, block_q, upload_q = mp.Queue(5000), mp.Queue(5000), mp.Queue(5000), mp.Queue()
    
    tokenizer_path = f"{HF_USER}/tokenizer-{lang}"
    
    # Spin up workers
    tok_procs = [mp.Process(target=tokenizer_worker, args=(text_q, token_q, tokenizer_path)) for _ in range(TOKEN_WORKERS)]
    for w in tok_procs: w.start()
    
    p_proc = mp.Process(target=packer, args=(token_q, block_q))
    w_proc = mp.Process(target=arrow_writer, args=(lang, block_q, upload_q))
    p_proc.start(); w_proc.start()
    
    up_thread = Thread(target=uploader, args=(upload_q,), daemon=True)
    up_thread.start()

    tokens_processed = 0
    start_time = time.time()

    for text in stream(lang):
        text_q.put(text)
        tokens_processed += (len(text) // 4) # Rough estimate (4 chars per token)
        if tokens_processed >= TARGETS[lang]: break

    # Shutdown sequence
    for _ in tok_procs: text_q.put(None)
    for w in tok_procs: w.join()
    token_q.put(None); p_proc.join()
    block_q.put(None); w_proc.join()
    upload_q.put(None)
    
    print(f"✅ [FINISHED] {lang} in {(time.time()-start_time)/3600:.2f} hours")

def main():
    # RUN SEQUENTIALLY to avoid crashing the node and OOMing RAM
    for lang in TARGETS:
        run_language(lang)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
