"""Stream a bounded 8 MiB FineWeb English fixture, large enough for five 786k updates."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def main():
    path = Path("fineweb-800k-confirm.bin")
    manifest_path = Path("fineweb-800k-confirm-manifest.json")
    target = 4_194_305
    tokenizer_path = Path("/content/good_tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None and max(tokenizer.get_vocab().values()) < 65536
    revision = "9bb295ddab0e05d785b879661af7260fed5140fc"
    stream = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train",
                          streaming=True, revision=revision)
    started = time.perf_counter()
    count = documents = 0
    with path.open("wb") as output:
        for document in stream:
            tokens = tokenizer.encode(document["text"], add_special_tokens=False).ids + [eos]
            tokens = tokens[:target-count]
            np.asarray(tokens, dtype=np.uint16).tofile(output)
            count += len(tokens)
            documents += 1
            if count >= target:
                break
    assert count == target
    metadata = dict(dataset="HuggingFaceFW/fineweb", config="sample-10BT", split="train",
                    revision=revision, streaming=True, source_documents=documents,
                    token_count=count, bytes=path.stat().st_size, eos=eos,
                    add_special_tokens=False,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                    preparation_seconds=time.perf_counter()-started)
    manifest_path.write_text(json.dumps(metadata, indent=2))
    print("CONFIRMATION_FIXTURE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
