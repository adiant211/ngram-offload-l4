#!/usr/bin/env bash
# Throughput sweep for 700M dense + 1.2B n-gram on one A100 80GB, 786,432-token updates.
# Each arm: 8 updates of real FineWeb data, the first 2 excluded (compilation, warmup).
# Usage: bash a100_sweep.sh [arm ...]   (default: all arms)
set -uo pipefail
cd /content/a100stack
COMMON=(--preset 700m --ngram-params 1200000000 --steps 8 --warmup-steps 2 --timing-skip 2
        --eval-data "" --resume none --output results-a100.jsonl --metrics metrics-a100.jsonl)
arm() {  # name, extra args...
  local name=$1; shift
  echo "START $name $(date +%T)"
  python -u trainer.py "${COMMON[@]}" --run-name "$name" "$@" > "a100-$name.log" 2>&1 \
    || { tail -15 "a100-$name.log"; echo "FAILED $name"; return 0; }
  python - "$name" <<'EOF'
import json, sys
r = json.loads(next(l[7:] for l in reversed(open(f"a100-{sys.argv[1]}.log").read().splitlines()) if l.startswith("RESULT ")))
keys = ("tokens_per_second", "median_update_s", "mean_mfu", "mean_dense_opt_s", "mean_sparse_s",
        "peak_alloc_gib", "peak_reserved_gib", "peak_device_used_gib", "host_peak_rss_gib", "final_loss", "cpu_input_fallbacks",
        "layer_drop_realized_rate")
print("SUMMARY", sys.argv[1], json.dumps({k: r.get(k) for k in keys}))
EOF
}
run_arm() {
  case "$1" in
    m16k)        arm m16k --micro-tokens 16384 --grad-accum 48 ;;
    m24k)        arm m24k --micro-tokens 24576 --grad-accum 32 ;;
    m24k-ce2)    arm m24k-ce2 --micro-tokens 24576 --grad-accum 32 --loss-chunks 2 ;;
    m8k)         arm m8k --micro-tokens 8192 --grad-accum 96 ;;
    m12k)        arm m12k --micro-tokens 12288 --grad-accum 64 ;;
    m12k-ce8)    arm m12k-ce8 --micro-tokens 12288 --grad-accum 64 --loss-chunks 8 ;;
    m8k-ld02)    arm m8k-ld02 --micro-tokens 8192 --grad-accum 96 --layer-drop 0.2 --layer-drop-schedule constant ;;
    m12k-ld02)   arm m12k-ld02 --micro-tokens 12288 --grad-accum 64 --layer-drop 0.2 --layer-drop-schedule constant ;;
    tinytable)   arm tinytable --micro-tokens 16384 --grad-accum 48 --ngram-params 20000000 ;;
    ld02)        arm ld02 --micro-tokens 16384 --grad-accum 48 --layer-drop 0.2 --layer-drop-schedule constant ;;
    ld04)        arm ld04 --micro-tokens 16384 --grad-accum 48 --layer-drop 0.4 --layer-drop-schedule constant ;;
    *) echo "unknown arm $1"; return 1 ;;
  esac
}
for a in "${@:-m16k m24k tinytable ld02}"; do
  for b in $a; do run_arm "$b"; done
done
echo SWEEP_DONE
