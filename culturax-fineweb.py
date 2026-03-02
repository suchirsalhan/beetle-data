#!/usr/bin/env python3
import os
import time
import queue
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

LANG = "es"
VOCAB_SIZE = 50_000
SEQ_LEN = 512
BOOTSTRAP_SENTENCES = 500_000

TMP = Path("tmp_ultra")
TMP.mkdir(exist_ok=True)

api = HfApi(token=HF_TOKEN)

# =====================================================
# STREAMS
# =====================================================

def culturax():
    ds = load_dataset("uonlp/CulturaX", LANG,
                      split="train", streaming=True)
    for ex in ds:
        yield ex["text"].replace("\n"," ")

def fineweb():
    ds = load_dataset("HuggingFaceFW/fineweb-edu",
                      split="train", streaming=True)
    for ex in ds:
        yield ex["text"].replace("\n"," ")

def bilingual_stream():
    while True:
        for t in culturax():
            yield t
        for t in fineweb():
            yield t

# =====================================================
# 1️⃣ TOKENIZER BOOTSTRAP
# =====================================================

def train_tokenizer(sample):

    print("🚀 Training SentencePiece tokenizer")

    prefix = TMP / "spm"

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
# 2️⃣ TOKENIZE + WRITE SHARDS
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
            block = np.array(buffer[:SEQ_LEN], dtype=np.int32)
            buffer = buffer[SEQ_LEN:]
            block_q.put(block)


def arrow_writer(block_q, upload_q):

    shard = 0
    arrays = []

    while True:
        block = block_q.get()
        if block is None:
            break

        arrays.append(pa.array([block]))

        if len(arrays) >= 10_000:

            table = pa.Table.from_arrays(
                [pa.concat_arrays(arrays)],
                names=["input_ids"]
            )

            fname = TMP / f"train_{shard}.arrow"

            with ipc.new_file(fname, table.schema) as w:
                w.write(table)

            upload_q.put(fname)
            arrays = []
            shard += 1


# =====================================================
# 3️⃣ UPLOADER
# =====================================================

def uploader(upload_q):

    dataset_repo = f"{HF_USER}/bilingual-{LANG}-512"

    create_repo(dataset_repo,
                repo_type="dataset",
                exist_ok=True,
                token=HF_TOKEN)

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
# MAIN PIPELINE
# =====================================================

def main():

    stream = bilingual_stream()

    # ---------------------------
    # Bootstrap tokenizer
    # ---------------------------

    sample = []
    for _ in range(BOOTSTRAP_SENTENCES):
        sample.append(next(stream))

    sp_model = train_tokenizer(sample)

    # Upload tokenizer
    tokenizer_repo = f"{HF_USER}/tokenizer-en-{LANG}"

    create_repo(tokenizer_repo,
                repo_type="model",
                exist_ok=True,
                token=HF_TOKEN)

    upload_folder(
        folder_path=TMP,
        repo_id=tokenizer_repo,
        repo_type="model",
        token=HF_TOKEN,
    )

    print("✅ Tokenizer uploaded")

    # ---------------------------
    # Streaming tokenization
    # ---------------------------

    text_q = mp.Queue(2000)
    block_q = mp.Queue(2000)
    upload_q = mp.Queue()

    tok_proc = mp.Process(
        target=tokenizer_worker,
        args=(text_q, block_q, sp_model))
    writer_proc = mp.Process(
        target=arrow_writer,
        args=(block_q, upload_q))

    tok_proc.start()
    writer_proc.start()

    up_thread = Thread(target=uploader, args=(upload_q,))
    up_thread.start()

    print("🔥 Continuous dataset building")

    for text in stream:
        text_q.put(text)

    text_q.put(None)
    tok_proc.join()

    block_q.put(None)
    writer_proc.join()

    upload_q.put(None)
    up_thread.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
