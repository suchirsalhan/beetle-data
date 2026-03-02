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

# Each language will have its own tokenizer + dataset repo
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
            yield t
        for t in en_gen:
            yield t

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
def tokenizer_worker(text_q, block_q, sp_model):
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    buffer = []
    while True:
        text = text_q.get()
        if text is None:
            break
        buffer.extend(sp.encode(text, out_type=int))
        while len(buffer) >= SEQ_LEN:
            block_q.put(np.array(buffer[:SEQ_LEN], dtype=np.int32))
            buffer = buffer[SEQ_LEN:]

def arrow_writer(block_q, upload_q, lang):
    shard = 0
    arrays = []
    while True:
        block = block_q.get()
        if block is None:
            break
        arrays.append(pa.array([block]))
        if len(arrays) >= 10_000:
            table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
            fname = TMP / f"{lang}_train_{shard}.arrow"
            with ipc.new_file(fname, table.schema) as w:
                w.write(table)
            upload_q.put(fname)
            arrays = []
            shard += 1
    # write leftover
    if arrays:
        table = pa.Table.from_arrays([pa.concat_arrays(arrays)], names=["input_ids"])
        fname = TMP / f"{lang}_train_{shard}.arrow"
        with ipc.new_file(fname, table.schema) as w:
            w.write(table)
        upload_q.put(fname)

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

    # 1️⃣ Bootstrap sample for tokenizer
    stream = bilingual_stream(lang)
    sample = [next(stream) for _ in range(BOOTSTRAP_SENTENCES)]
    sp_model = train_tokenizer(lang, sample)

    # Upload tokenizer
    tokenizer_repo = f"{HF_USER}/tokenizer-{lang}"
    create_repo(tokenizer_repo, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(folder_path=TMP / f"tokenizer_{lang}", repo_id=tokenizer_repo,
                  repo_type="model", token=HF_TOKEN)
    print(f"✅ Uploaded tokenizer for {lang}")

    # 2️⃣ Streaming tokenization
    text_q = mp.Queue(2000)
    block_q = mp.Queue(2000)
    upload_q = mp.Queue()

    tok_proc = mp.Process(target=tokenizer_worker, args=(text_q, block_q, sp_model))
    writer_proc = mp.Process(target=arrow_writer, args=(block_q, upload_q, lang))
    up_thread = Thread(target=uploader, args=(upload_q, f"{HF_USER}/bilingual-{lang}-512"))

    tok_proc.start()
    writer_proc.start()
    up_thread.start()

    # Feed stream
    for text in stream:
        text_q.put(text)

    text_q.put(None)
    tok_proc.join()

    block_q.put(None)
    writer_proc.join()

    upload_q.put(None)
    up_thread.join()
    print(f"🎉 Finished pipeline for {lang}")

# =====================================================
# MAIN: PARALLEL EXECUTION
# =====================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    processes = []
    for i, lang in enumerate(LANGS):
        p = mp.Process(target=process_language, args=(lang,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    print("✅ All languages finished")
