#!/usr/bin/env bash
# Supervised 700M dense + 1.2B n-gram run on one GPU. After any crash it restarts from the
# newest complete checkpoint (--resume auto). A clean finish or a graceful stop exits 0:
#   pkill -TERM -f trainer.py   -> checkpoint after the current update, then stop.
# Extra arguments are passed to trainer.py (later flags override earlier ones).
# For durability across runtime loss, point CKPT_DIR at mounted Google Drive or other storage.
set -uo pipefail
cd /content/a100stack
MAX_RESTARTS=${MAX_RESTARTS:-20}
# Compiled kernels: a warm cache turns a ~12-minute first update into ~1 minute after a
# restart. Keep it next to the checkpoints so a fresh runtime can reuse it too.
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR:-/content/compile-cache}/inductor"
export TRITON_CACHE_DIR="${CACHE_DIR:-/content/compile-cache}/triton"
ARGS=(--preset 700m --ngram-params 1200000000 --micro-tokens 16384 --grad-accum 32
      --steps 10000 --warmup-steps 300 --ckpt-dir "${CKPT_DIR:-ckpt700m}" --ckpt-minutes 30
      --keep 2 --resume auto --eval-every 250 --eval-exit-at 24
      --metrics metrics700m.jsonl --output results700m.jsonl "$@")
for attempt in $(seq 0 "$MAX_RESTARTS"); do
  python -u trainer.py "${ARGS[@]}" --run-name "main-a$attempt"
  code=$?
  if [ "$code" -eq 0 ]; then echo "SUPERVISOR finished"; exit 0; fi
  echo "SUPERVISOR trainer exited with $code; restart $((attempt+1)) from the last checkpoint"
  sleep 15
done
echo "SUPERVISOR giving up after $MAX_RESTARTS restarts"
exit 1
