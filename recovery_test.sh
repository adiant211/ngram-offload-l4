#!/usr/bin/env bash
# End-to-end recovery test at full model size (700M dense + 1.2B table), small updates.
#   ref, ref2 : two uninterrupted runs of 9 updates (their gap is the run-to-run noise)
#   crash     : checkpoints every 3 updates, then
#               1. hard exit while checkpoint 6 is being written -> resumes from 3
#               2. hard exit in the middle of update 8           -> resumes from 6
#               3. finishes at update 9
#   sigterm   : resumes the finished run with a longer schedule, gets SIGTERM, must
#               checkpoint and exit 0
# The crash run's losses after each resume must match the reference within the noise.
set -uo pipefail
cd /content/a100stack
rm -rf ckpt-recovery metrics-recovery-*.jsonl results-recovery.jsonl
COMMON=(--preset 700m --ngram-params 1200000000 --micro-tokens 16384 --grad-accum 4
        --steps 9 --warmup-steps 2 --eval-data "" --timing-skip 1 --output results-recovery.jsonl)
python -u trainer.py "${COMMON[@]}" --resume none --run-name ref --metrics metrics-recovery-ref.jsonl \
  > recovery-ref.log 2>&1; echo "ref exit $?"
python -u trainer.py "${COMMON[@]}" --resume none --run-name ref2 --metrics metrics-recovery-ref2.jsonl \
  > recovery-ref2.log 2>&1; echo "ref2 exit $?"
CRASH=("${COMMON[@]}" --ckpt-dir ckpt-recovery --ckpt-every 3 --keep 3 --metrics metrics-recovery-crash.jsonl)
python -u trainer.py "${CRASH[@]}" --resume auto --run-name crash1 --crash-at-step 6 --crash-point during-save \
  > recovery-crash1.log 2>&1; echo "crash1 exit $? (expect 17)"; ls ckpt-recovery
python -u trainer.py "${CRASH[@]}" --resume auto --run-name crash2 --crash-at-step 7 --crash-point mid-update \
  > recovery-crash2.log 2>&1; echo "crash2 exit $? (expect 17)"; ls ckpt-recovery
python -u trainer.py "${CRASH[@]}" --resume auto --run-name crash3 > recovery-crash3.log 2>&1
echo "crash3 exit $? (expect 0)"; ls ckpt-recovery
grep -h "RESUMED\|RESUME_SKIPPED\|CRASH_TEST\|CHECKPOINT" recovery-crash*.log
python -u trainer.py "${CRASH[@]}" --steps 40 --resume auto --run-name sigterm > recovery-sigterm.log 2>&1 &
pid=$!
until [ "$(grep -c '^STEP' recovery-sigterm.log)" -ge 2 ]; do sleep 2; kill -0 $pid 2>/dev/null || break; done
kill -TERM $pid; wait $pid; echo "sigterm exit $? (expect 0)"; ls ckpt-recovery
grep -h "RESUMED\|STOPPED\|CHECKPOINT" recovery-sigterm.log
python - <<'EOF'
import json
def losses(name):
    rows = [json.loads(l) for l in open(f"metrics-recovery-{name}.jsonl")]
    return rows
ref = {r["step"]: r["loss"] for r in losses("ref")}
ref2 = {r["step"]: r["loss"] for r in losses("ref2")}
crash = {}
for r in losses("crash"):
    crash[r["step"]] = (r["loss"], r["run"])  # later (resumed) records overwrite earlier ones
rel = lambda a, b: abs(a-b)/abs(b)
noise = max(rel(ref2[s], ref[s]) for s in ref)
gap = {s: (round(rel(crash[s][0], ref[s]), 7), crash[s][1]) for s in sorted(crash) if s in ref}
print("RECOVERY_NOISE ref2_vs_ref_max_rel", noise)
print("RECOVERY_GAP crash_vs_ref", json.dumps(gap))
worst = max(v[0] for v in gap.values())
print("RECOVERY_VERDICT", "PASS" if worst <= max(3*noise, 1e-5) else "CHECK", "worst", worst)
EOF
