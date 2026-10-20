#!/usr/bin/env bash
set -euo pipefail
cd /content/a100stack
python -u bench100m.py --tokens 8192 --seq-len 1024 --grad-accum 96 --ngram-device cpu --ram-mode staged --attn native --pack --compile mlp --ckpt none --fused-norms --loss-chunks 8 --token-weighted --defer-loss-sync --data fineweb-800k-confirm.bin --warmup 1 --steps 4 --output results-800k-7gb.jsonl
