"""Canonical token ids for n-gram hashing (Engram-style tokenizer compression).

Tokens whose text is equal after normalization share one id for hashing only, so "The", " the"
and "THE" read the same n-gram rows. The model's input embedding and targets keep the original ids.

Normalization follows Engram (github.com/deepseek-ai/Engram, engram_demo_v1.py):
NFKC, NFD, strip accents, lowercase, collapse whitespace runs to one space, a lone space is kept as
a space, strip. Each normalized string maps to its smallest original id, so the map is deterministic.
Special tokens, tokens that decode to incomplete UTF-8 (byte-fallback pieces print as U+FFFD) and
tokens that normalize to an empty string keep their own id.
"""
import hashlib
import json

import torch

SENTINEL = ""


def normalizer():
    from tokenizers import Regex, normalizers
    return normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), SENTINEL),
        normalizers.Strip(), normalizers.Replace(SENTINEL, " ")])


def build_map(tokenizer_path):
    """[vocab] int64 tensor: original id -> canonical id."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    special = {t["id"] for t in json.loads(tokenizer.to_str()).get("added_tokens", []) if t.get("special")}
    norm = normalizer()
    first = {}
    canon = torch.arange(vocab, dtype=torch.int64)
    for token in range(vocab):
        if token in special:
            continue
        text = tokenizer.decode([token], skip_special_tokens=False)
        if "�" in text:
            continue
        key = norm.normalize_str(text)
        if not key:
            continue
        canon[token] = first.setdefault(key, token)
    return canon


def checksum(canon):
    return hashlib.sha256(canon.numpy().tobytes()).hexdigest()


def attach(memory, canon):
    """Use `canon` when hashing in both the RAM path (pinned_ngram) and the GPU reference path."""
    memory._canon = canon.cpu()
    memory._canon_cache = {}


def canonical(memory, ids):
    """ids mapped to canonical ids on their own device; unchanged when no map is attached."""
    canon = getattr(memory, "_canon", None)
    if canon is None:
        return ids
    if ids.device.type == "cpu":
        return canon[ids]
    cached = memory._canon_cache.get(ids.device)
    if cached is None:
        cached = memory._canon_cache[ids.device] = canon.to(ids.device)
    return cached[ids]


def check(tokenizer_path):
    a, b = build_map(tokenizer_path), build_map(tokenizer_path)
    assert torch.equal(a, b), "map is not deterministic"
    assert bool((a <= torch.arange(len(a))).all()), "canonical id must be the smallest of its class"
    assert torch.equal(a[a], a), "map must be idempotent"
    distinct = int(torch.unique(a).numel())
    print("NGRAM_CANON", json.dumps(dict(vocab=len(a), distinct=distinct, folded=round(1 - distinct / len(a), 4),
                                         sha256=checksum(a)[:16])), flush=True)
    return a


if __name__ == "__main__":
    import sys
    check(sys.argv[1] if len(sys.argv) > 1 else "/content/good_tokenizer/tokenizer.json")
