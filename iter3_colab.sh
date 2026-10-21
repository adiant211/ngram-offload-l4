#!/usr/bin/env bash
# Iteration 3 on a fresh Colab L4 runtime. Expects in /content:
#   a100stack.zip, good_tokenizer.zip, and either iter3-scripts.zip or this repository's
#   .py files next to this script.
# Usage: bash iter3_colab.sh setup | parity | sweep | confirm | fastce | profile [mlp|tail]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd /content
BASE=(--tokens 8192 --seq-len 1024 --grad-accum 96 --ngram-device cpu --ram-mode staged
      --attn native --pack --ckpt none --fused-norms --loss-chunks 8 --token-weighted
      --defer-loss-sync --data fineweb-800k-confirm.bin --warmup 1 --steps 4)

run() {  # name, extra args...
  local name=$1; shift
  echo "START $name $(date +%T)"
  python -u bench100m.py "${BASE[@]}" "$@" --output results-iter3.jsonl > "iter3-$name.log" 2>&1 \
    || { tail -20 "iter3-$name.log"; echo "FAILED $name"; return 1; }
  python - "$name" <<'EOF'
import json, sys
r = json.loads(next(l[7:] for l in reversed(open(f"iter3-{sys.argv[1]}.log").read().splitlines()) if l.startswith("RESULT ")))
keys = ("tokens_per_second", "median_step_ms", "forward_backward_clip_ms", "sparse_optimizer_ms",
        "peak_allocated_gib", "peak_reserved_gib", "final_loss", "mean_touched_rows", "cpu_input_fallbacks")
print("SUMMARY", sys.argv[1], json.dumps({k: r.get(k) for k in keys}))
print("DETAIL", sys.argv[1], json.dumps({k: round(v) for k, v in r.get("pin_detail_ms_per_step", {}).items()}))
EOF
}

case "${1:-}" in
setup)
  pip -q install flash-linear-attention==0.5.2 fla-core==0.5.2 einops
  python - <<'EOF'
import zipfile
for name in ("a100stack.zip", "good_tokenizer.zip"):
    zipfile.ZipFile(f"/content/{name}").extractall("/content")
EOF
  if [ -f /content/iter3-scripts.zip ]; then
    python -c "import zipfile; zipfile.ZipFile('/content/iter3-scripts.zip').extractall('/content/a100stack')"
  else
    cp "$HERE"/*.py /content/a100stack/
  fi
  cd /content/a100stack
  python - <<'EOF'
from pathlib import Path
p = Path("model.py")
s = p.read_text()
s = s.replace("self.loss_fn(x.unsqueeze(0), y.unsqueeze(0), self.lm_head.weight)", "self.loss_fn(x, y, self.lm_head.weight)")
s = s.replace("F.conv1d(xi, conv.weight, groups=x.shape[1])",
              "F.conv1d(xi, conv.weight.unsqueeze(1) if conv.weight.ndim == 2 else conv.weight, groups=x.shape[1])")
p.write_text(s)
EOF
  python -u prepare800k.py
  python - <<'EOF'
import hashlib, json
digest = hashlib.sha256(open("fineweb-800k-confirm.bin", "rb").read()).hexdigest()
assert digest == "a5ff5040206ad44e68c317cc286124a1d44c86502594951bed0ac703223c1f7f", digest
print("FIXTURE_SHA_MATCHES_ITERATION_2")
EOF
  nproc; free -g; nvidia-smi --query-gpu=name,memory.total --format=csv
  ;;
parity)
  cd /content/a100stack
  python -u pinned_ngram.py
  python -c "from ram_ngram import check_ram_ngram; check_ram_ngram()"
  ;;
sweep)
  cd /content/a100stack
  run baseline --compile mlp
  run blocks --compile mlp --sparse-update blocks
  run blocks-threads --compile mlp --sparse-update blocks --update-threads "$(nproc)"
  run tail --compile tail --sparse-update blocks --update-threads "$(nproc)"
  ;;
confirm)  # after the scratch-buffer block update
  cd /content/a100stack
  run final-mlp --compile mlp --sparse-update blocks --update-threads "$(nproc)"
  run final-tail --compile tail --sparse-update blocks --update-threads "$(nproc)"
  run final-tail-4chunks --compile tail --sparse-update blocks --update-threads "$(nproc)" --loss-chunks 4
  ;;
fastce)  # adds the patched FLA fused cross-entropy
  cd /content/a100stack
  python -u fast_flce.py
  run fastce-tail --compile tail --sparse-update blocks --update-threads "$(nproc)" --fast-ce
  run fastce-tail-4chunks --compile tail --sparse-update blocks --update-threads "$(nproc)" --fast-ce --loss-chunks 4
  ;;
dion)  # Dion3 on one GPU, against AdamW with the same concat packing
  cd /content/a100stack
  FAST=(--compile tail --sparse-update blocks --update-threads "$(nproc)" --fast-ce --loss-chunks 4 --pack-mode concat)
  run adamw-concat "${FAST[@]}"
  run dion3 "${FAST[@]}" --optimizer dion3
  ;;
dist)  # 1 rank x accumulation 4 must match 2 gloo ranks x accumulation 2 on one GPU
  cd /content/a100stack
  SMALL=(--tokens 8192 --seq-len 1024 --ngram-device cpu --ram-mode staged --sparse-update blocks
         --attn native --pack --pack-mode concat --ckpt none --fused-norms --fast-ce --loss-chunks 4
         --compile tail --token-weighted --defer-loss-sync --data fineweb-800k-confirm.bin
         --warmup 1 --steps 2 --optimizer dion3 --checksums --output results-dist.jsonl)
  python -u bench100m.py "${SMALL[@]}" --grad-accum 4 > dist-world1.log 2>&1
  torchrun --standalone --nproc_per_node 2 bench100m.py "${SMALL[@]}" --grad-accum 2 \
    --dist-backend gloo > dist-world2.log 2>&1
  python - <<'EOF'
import json
rows = [json.loads(l) for l in open("results-dist.jsonl")][-2:]
for r in rows:
    print("DIST", r["world_size"], json.dumps({k: r.get(k) for k in ("first_loss", "final_loss",
          "dense_checksum", "table_checksum", "row_state_checksum", "checksum_rank_spread")}))
a, b = rows
rel = lambda k: abs(a[k]-b[k])/max(abs(a[k]), 1e-12)
print("DIST_RELATIVE", json.dumps({k: rel(k) for k in ("final_loss", "dense_checksum",
      "table_checksum", "row_state_checksum")}))
EOF
  ;;
depthab)  # Depth-Attention quality A/B: same data, held-out loss, plus a second seed for noise
  cd /content/a100stack
  [ -f fineweb-ab-train.bin ] || python -u prepare_ab.py --train-tokens 25000000
  AB=(--tokens 8192 --seq-len 1024 --grad-accum 8 --ngram-device cpu --ram-mode staged
      --sparse-update blocks --update-threads "$(nproc)" --attn native --pack --pack-mode concat
      --ckpt none --fused-norms --fast-ce --loss-chunks 4 --compile tail --token-weighted
      --defer-loss-sync --optimizer dion3 --data fineweb-ab-train.bin --eval-data fineweb-ab-val.bin
      --warmup 1 --steps "${AB_STEPS:-300}" --eval-exit-at 15 --output results-depthab.jsonl)
  for arm in "base 1234 off 0" "depth 1234 softmax 0" "layerdrop 1234 off 0.2" "base-seed2 4321 off 0"; do
    set -- $arm
    python -u bench100m.py "${AB[@]}" --seed "$2" --depth-attn "$3" --layer-drop "$4" \
      > "depthab-$1.log" 2>&1 || { tail -20 "depthab-$1.log"; echo "FAILED $1"; }
    grep -h EVAL "depthab-$1.log" | sed "s/^/$1 /"
  done
  ;;
layerdrop)  # throughput of the 786k recipe with constant increasing-depth layer dropout
  cd /content/a100stack
  FAST=(--compile tail --sparse-update blocks --update-threads "$(nproc)" --fast-ce --loss-chunks 4
        --pack-mode concat --optimizer dion3 --layer-drop-schedule constant)
  run ld-0.2 "${FAST[@]}" --layer-drop 0.2
  run ld-0.4 "${FAST[@]}" --layer-drop 0.4
  ;;
profile)
  cd /content/a100stack
  # A later --steps overrides the one in BASE.
  python -u bench100m.py "${BASE[@]}" --compile "${2:-mlp}" --sparse-update blocks \
    --steps 1 --profile-kernels 4 --output results-iter3-profile.jsonl > iter3-profile.log 2>&1
  grep -A45 "Self CPU" iter3-profile.log | head -50; grep KERNEL_PROFILE iter3-profile.log
  ;;
*) echo "usage: $0 setup|parity|sweep|confirm|fastce|dion|dist|depthab|layerdrop|profile [mlp|tail]"; exit 2 ;;
esac
