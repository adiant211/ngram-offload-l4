"""Disjoint FineWeb train/held-out fixtures for short quality A/B runs.

Held-out documents come first in the stream, training documents after them, so no
document appears in both. Same pinned revision and tokenizer as prepare800k.py.
Usage: python prepare_ab.py --train-tokens 50000000 --val-tokens 1048577
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_048_577)
    parser.add_argument("--prefix", default="fineweb-ab")
    args = parser.parse_args()
    tokenizer_path = Path("/content/good_tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<eos>")
    revision = "9bb295ddab0e05d785b879661af7260fed5140fc"
    stream = iter(load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train",
                               streaming=True, revision=revision))
    manifest = dict(dataset="HuggingFaceFW/fineweb", config="sample-10BT", revision=revision,
                    eos=eos, add_special_tokens=False,
                    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
    for split, target in (("val", args.val_tokens), ("train", args.train_tokens)):
        path = Path(f"{args.prefix}-{split}.bin")
        count = documents = 0
        with path.open("wb") as output:
            while count < target:
                ids = tokenizer.encode(next(stream)["text"], add_special_tokens=False).ids + [eos]
                ids = ids[:target-count]
                np.asarray(ids, dtype=np.uint16).tofile(output)
                count += len(ids)
                documents += 1
        manifest[split] = dict(path=str(path), tokens=count, documents=documents,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    Path(f"{args.prefix}-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("AB_FIXTURES", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
