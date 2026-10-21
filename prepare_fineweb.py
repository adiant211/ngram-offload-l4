"""Tokenize a FineWeb sample-10BT prefix into uint16 train/held-out files for training runs.

Held-out documents come first in the stream and training documents after them, so no
document appears in both. Same pinned revision and tokenizer as prepare_ab.py, but
batched multithreaded encoding, so a few hundred million tokens take minutes.
Usage: python prepare_fineweb.py --train-tokens 300000000 --val-tokens 2097153 --prefix fineweb-train
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=int, default=300_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_097_153)
    parser.add_argument("--prefix", default="fineweb-train")
    parser.add_argument("--batch-docs", type=int, default=2048)
    parser.add_argument("--tokenizer", default="/content/good_tokenizer/tokenizer.json")
    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<eos>")
    revision = "9bb295ddab0e05d785b879661af7260fed5140fc"
    stream = iter(load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train",
                               streaming=True, revision=revision))
    manifest = dict(dataset="HuggingFaceFW/fineweb", config="sample-10BT", revision=revision,
                    eos=eos, add_special_tokens=False, tokenizer_sha256=sha256(tokenizer_path))
    pending = []  # encoded documents not yet written (carried from the held-out split)
    begin = time.time()
    for split, target in (("val", args.val_tokens), ("train", args.train_tokens)):
        path = Path(f"{args.prefix}-{split}.bin")
        partial = path.with_suffix(".partial")
        count = documents = 0
        with partial.open("wb") as output:
            while count < target:
                if not pending:
                    texts = [row["text"] for _, row in zip(range(args.batch_docs), stream)]
                    if not texts:
                        raise RuntimeError("FineWeb stream exhausted")
                    pending = [e.ids + [eos] for e in
                               tokenizer.encode_batch(texts, add_special_tokens=False)]
                    pending.reverse()
                ids = pending.pop()[:target-count]
                np.asarray(ids, dtype=np.uint16).tofile(output)
                count += len(ids)
                documents += 1
                if documents % 100_000 == 0:
                    print("PROGRESS", split, count, f"{count/(time.time()-begin):.0f} tok/s", flush=True)
        partial.rename(path)  # a finished file never looks partial, and vice versa
        manifest[split] = dict(path=str(path), tokens=count, documents=documents, sha256=sha256(path))
    Path(f"{args.prefix}-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("FINEWEB_FIXTURES", json.dumps(manifest), f"{time.time()-begin:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    # The streaming reader's background threads can keep the interpreter alive after
    # an aborted request; the files and manifest are complete by now.
    sys.stdout.flush()
    os._exit(0)
