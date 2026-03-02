#!/usr/bin/env python3
import os
import time
import multiprocessing as mp
from threading import Thread
from pathlib import Path
import sentencepiece as spm
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_folder

# =====================================================
# CONFIG
# =====================================================
HF_USER = "RA-ALTA"
HF_TOKEN = os.environ["HF_TOKEN"]

LANGS = ["es","fr","de","pl","tr","ar","zh","en"]
VOCAB_SIZE = 50_000
SEQ_LEN = 512
BOOTSTRAP_SENTENCES = 500_000
TMP = Path("tmp_factory")
TMP.mkdir(exist_ok=True)

TOTAL_TOKENS = {
    "5B": 5_000_000_000,
    "10B": 10_000_000_000,
    "15B": 15_000_000_000,
    "20B": 20_000_000_000,
}

RATIO_L1 = 2/3
RATIO_L2 = 1/3

SHARD_BLOCKS = 10_000  # blocks per shard
api = HfApi(token=HF_TOKEN)

# =====================================================
# STREAMS
# =====================================================
def culturax_stream(lang):
    ds = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n"," ")

def fineweb_stream():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    for ex in ds:
        t = ex.get("text")
        if t:
            yield t.replace("\n"," ")

def bilingual_stream(lang):
    """Balanced L1 + English"""
    l1_gen = culturax_stream(lang) if lang != "en" else fineweb_stream()
    en_gen = fineweb_stream()
    while True:
        for t in l1_gen:
            yield t, "L1"
        for t in en_gen:
            yield t, "L2"

# =====================================================
# TOKENIZER BOOTSTRAP
# =====================================================
def train_tokenizer(lang, sample):
    out_dir = TMP / f"tokenizer_{lang}"
    out_dir.mkdir(exist_ok=True)
    prefix = out_dir / "spm"
    print(f"🚀 Training tokenizer for {lang}")
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sample),
        model_prefix=str(prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        num_threads=os.cpu_count(),
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
    )
    return str(prefix) + ".model"

# =====================================================
# TOKENIZE + ARROW WRITER
# =====================================================
def tokenizer_worker(text_q, block_q, sp_model, max_tokens_L1, max_tokens_L2, counter):
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    buffer = []
    count_L1 = 0
    count_L2 = 0
    start_time = time.time()

    while True:
        item = text_q.get()
        if item is None:
            break
        text, tag = item
        tokens = sp.encode(text, out_type=int)

        # enforce token limits per L1/L2
        if tag == "L1":
            if count_L1 >= max_tokens_L1:
                continue
            remaining = max_tokens_L1 - count_L1
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
            count_L1 += len(tokens)
        else:  # L2
            if count_L2 >= max_tokens_L2:
                continue
            remaining = max_tokens_L2 - count_L2
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
            count_L2 += len(tokens)

        buffer.extend(tokens)
        while len(buffer) >= SEQ_LEN:
            block_q.put(np.array(buffer[:SEQ_LEN], dtype=np.int32))
            buffer = buffer[SEQ_LEN:]
            with counter.get_lock():
                counter.value += SEQ_LEN
                elapsed = time.time() - start_time
                eta = ((max_tokens_L1 + max_tokens_L2) - counter.value) / (counter.value / elapsed + 1e-9)
                print(f"\rTokens processed: {counter.value / 1e9:.2f}B | ETA: {eta/60:.1f} min", end="")

        # stop if both quotas reached
        if count_L1 >= max_tokens_L1 and count_L2 >= max_tokens_L2:
            break

    print()  # newline after finishing

def arrow_writer(block_q, upload_q, lang, suffix, counter):
    shard = 0
    arrays = []
    while True:
        block = block_q.get()
        if block is None:
            break
        arrays.append(pa.array([block]))
        if len(arrays) >= SHARD_BLOCKS:
            table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
            fname = TMP / f"{lang}_train_{suffix}_{shard}.arrow"
            with ipc.new_file(fname, table.schema) as w:
                w.write(table)
            upload_q.put(fname)
            arrays = []
            shard += 1
            print(f"Shard {shard} written for {lang}-{suffix} | Tokens so far: {counter.value/1e9:.2f}B")
    # leftover
    if arrays:
        table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
        fname = TMP / f"{lang}_train_{suffix}_{shard}.arrow"
        with ipc.new_file(fname, table.schema) as w:
            w.write(table)
        upload_q.put(fname)
        print(f"Final shard {shard} written for {lang}-{suffix} | Tokens so far: {counter.value/1e9:.2f}B")

# =====================================================
# UPLOADER
# =====================================================
def uploader(upload_q, dataset_repo):
    create_repo(dataset_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    while True:
        fname = upload_q.get()
        if fname is None:
            break
        api.upload_file(
            path_or_fileobj=str(fname),
            path_in_repo=fname.name,
            repo_id=dataset_repo,
            repo_type="dataset",
        )
        os.remove(fname)

# =====================================================
# LANGUAGE PIPELINE
# =====================================================
def process_language(lang):
    print(f"🔹 Starting pipeline for {lang}")
    stream = bilingual_stream(lang)
    sample = [next(stream)[0] for _ in range(BOOTSTRAP_SENTENCES)]
    sp_model = train_tokenizer(lang, sample)

    tokenizer_repo = f"{HF_USER}/tokenizer-{lang}"
    create_repo(tokenizer_repo, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(folder_path=TMP / f"tokenizer_{lang}", repo_id=tokenizer_repo,
                  repo_type="model", token=HF_TOKEN)
    print(f"✅ Uploaded tokenizer for {lang}")

    for suffix, total in TOTAL_TOKENS.items():
        print(f"🟢 Processing {suffix} dataset for {lang}")
        max_tokens_L1 = int(total * RATIO_L1)
        max_tokens_L2 = int(total * RATIO_L2)

        counter = mp.Value("i", 0)  # shared token counter
        text_q = mp.Queue(2000)
        block_q = mp.Queue(2000)
        upload_q = mp.Queue()

        tok_proc = mp.Process(target=tokenizer_worker,
                              args=(text_q, block_q, sp_model, max_tokens_L1, max_tokens_L2, counter))
        writer_proc = mp.Process(target=arrow_writer, args=(block_q, upload_q, lang, suffix, counter))
        up_thread = Thread(target=uploader, args=(upload_q, f"{HF_USER}/{lang}-{suffix}"))

        tok_proc.start()
        writer_proc.start()
        up_thread.start()

        stream = bilingual_stream(lang)
        for item in stream:
            text_q.put(item)
        text_q.put(None)
        tok_proc.join()

        block_q.put(None)
        writer_proc.join()

        upload_q.put(None)
        up_thread.join()
        print(f"🎉 Finished {suffix} dataset for {lang}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    processes = []
    for lang in LANGS:
        p = mp.Process(target=process_language, args=(lang,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    print("✅ All languages finished")
