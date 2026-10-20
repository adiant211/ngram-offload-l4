#!/usr/bin/env bash
# Iteration 3 recommended recipe: 8 x 1,024 x 96 = 786,432 tokens/update.
# Measured on a Colab L4: 29,178 tokens/s, 6.53 GiB allocated / 6.80 GiB reserved peak.
# For the lowest-memory profile (6.30 / 6.58 GiB, 28,895 tokens/s) use --loss-chunks 8.
set -euo pipefail
cd /content/a100stack
python -u bench100m.py --tokens 8192 --seq-len 1024 --grad-accum 96 --ngram-device cpu --ram-mode staged --sparse-update blocks --update-threads "$(nproc)" --attn native --pack --compile tail --ckpt none --fused-norms --fast-ce --loss-chunks 4 --token-weighted --defer-loss-sync --data fineweb-800k-confirm.bin --warmup 1 --steps 4 --output results-iter3.jsonl
