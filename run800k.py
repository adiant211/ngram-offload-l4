"""Run one controlled 700k--800k-token update throughput trial in the Colab project."""
import json
import subprocess
from pathlib import Path

trials800 = {}


def run800(name, batch, seq, accum, extra=(), warmup=1, steps=2, device="cuda"):
    assert 700000 <= batch*seq*accum <= 800000
    cmd = ["python", "-u", "bench100m.py", "--tokens", str(batch*seq), "--seq-len", str(seq),
           "--grad-accum", str(accum), "--ngram-device", device,
           "--attn", "native", "--pack", "--compile", "mlp", "--ckpt", "none",
           "--gdn-head-dim", "128", "--token-weighted", "--defer-loss-sync",
           "--warmup", str(warmup), "--steps", str(steps),
           "--output", "results-800k-7gb.jsonl", *extra]
    if device == "cuda" and "--host-accum" not in extra:
        cmd.append("--cuda-accum")
    print("START", name, "shape", batch, seq, accum, flush=True)
    logpath = Path(name+".log")
    with logpath.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("PLACEMENT", "STEP")):
                print(line.rstrip(), flush=True)
        code = proc.wait()
    lines = logpath.read_text().splitlines()
    if code:
        print("\n".join(lines[-14:]), flush=True)
        raise RuntimeError(name+" failed")
    record = json.loads(next(x[7:] for x in reversed(lines) if x.startswith("RESULT ")))
    trials800[name] = record
    fields = ["tokens_per_second", "peak_allocated_gib", "peak_reserved_gib", "median_step_ms",
              "effective_batch_tokens", "final_loss", "finite"]
    print("SUMMARY", name, json.dumps({k: record[k] for k in fields}), flush=True)
    return record
