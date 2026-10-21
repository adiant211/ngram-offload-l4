#!/usr/bin/env python3
"""Launch the 700M dense + 1.2B n-gram run on this host's GPUs, tuned from measured hardware.

Tuned for 2x NVIDIA CMP 170HX (GA100 with 70 SMs, ~1.5 TB/s HBM2e, PCIe Gen1 x4, no NVLink),
here with 64 GB per card; it adapts to other GA100-class GPUs too.

1. probe  (in a child process, so this launcher never holds GPU memory): per GPU memory, SMs,
          pinned host<->device bandwidth, BF16 and FP32 (no TF32) GEMM throughput, peer access;
          host RAM, cores and checkpoint disk space.
2. plan   microbatch from memory (measured A100 memory model), accumulation for ~1M-token updates,
          and for slow PCIe: the n-gram accumulator reduced in host memory over gloo, BF16
          gradient all-reduce, Dion3 replicated on every rank (no optimizer all-gather),
          per-rank CPU thread split.
3. run    torchrun trainer.py under a supervisor: after a crash it restarts with --resume auto;
          Ctrl-C / SIGTERM stops the trainer after its current update with a checkpoint.

Usage:
  python run.py --probe-only                 # measure and print the plan
  python run.py --dry-run                    # also print the torchrun command
  python run.py                              # train (resumes automatically)
  python run.py -- --layer-drop 0.2          # extra trainer.py arguments after --
Setup: put this repository's .py files next to the a100stack source (model.py, config.py, ...),
       pip install flash-linear-attention==0.5.2 fla-core==0.5.2 "git+https://github.com/microsoft/dion.git@58d38ad"
Data:  python prepare_fineweb.py --tokenizer good_tokenizer/tokenizer.json --train-tokens 3000000000
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Measured on an A100-SXM4-80GB, 700m preset (iteration 5): peak allocated GiB is close to
# FIXED + PER_K * (microbatch tokens / 1024), and the device adds ~0.9 GiB (context, libraries).
FIXED_GIB, PER_K_GIB, DEVICE_EXTRA_GIB = 9.9, 1.95, 0.9
DDP_EXTRA_GIB = 1.0                       # gradient buckets and compression buffers
# Forward/backward only (788,480 tokens in 28.3 s at the default config); BF16 8192^3 GEMM rate.
A100_FWD_BWD_TOKENS_PER_S, A100_BF16_TFLOPS = 27_860, 265.0


def probe_worker():
    """Runs in a child process: measure every visible GPU and print one JSON line."""
    import torch
    gpus = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        props = torch.cuda.get_device_properties(index)
        size = 256 << 20
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(size, dtype=torch.uint8, device="cuda")

        def bandwidth(copy):
            best = float("inf")
            for _ in range(3):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                copy()
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - begin)
            return size / best / 1e9

        h2d = bandwidth(lambda: device.copy_(host, non_blocking=True))
        d2h = bandwidth(lambda: host.copy_(device, non_blocking=True))

        def gemm_tflops(n, dtype, reps):
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            a @ b
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for _ in range(reps):
                a @ b
            torch.cuda.synchronize()
            return 2 * n**3 * reps / (time.perf_counter() - begin) / 1e12

        torch.backends.cuda.matmul.allow_tf32 = False
        bf16 = gemm_tflops(8192, torch.bfloat16, 20)
        fp32 = gemm_tflops(4096, torch.float32, 5)
        peers = [j for j in range(torch.cuda.device_count())
                 if j != index and torch.cuda.can_device_access_peer(index, j)]
        gpus.append(dict(index=index, name=props.name, memory_gib=props.total_memory / 2**30,
                         sms=props.multi_processor_count, h2d_gbs=h2d, d2h_gbs=d2h,
                         bf16_tflops=bf16, fp32_tflops=fp32, peers=peers))
        del host, device
        torch.cuda.empty_cache()
    print("PROBE " + json.dumps(dict(gpus=gpus, torch=torch.__version__)), flush=True)


def probe():
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe-worker"],
                         capture_output=True, text=True, check=True).stdout
    result = json.loads(next(l[6:] for l in out.splitlines() if l.startswith("PROBE ")))
    result["host_ram_gib"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    result["cpu_cores"] = os.cpu_count()
    return result


def plan(hw, args):
    gpus = hw["gpus"][:args.gpus] if args.gpus else hw["gpus"]
    if not gpus:
        raise SystemExit("no CUDA GPUs visible")
    world = len(gpus)
    memory = min(g["memory_gib"] for g in gpus)
    pcie = min(min(g["h2d_gbs"], g["d2h_gbs"]) for g in gpus)
    bf16 = min(g["bf16_tflops"] for g in gpus)
    extra = DDP_EXTRA_GIB if world > 1 else 0.0
    budget = memory * args.memory_fraction

    def device_gib(micro):
        return FIXED_GIB + PER_K_GIB * micro / 1024 + DEVICE_EXTRA_GIB + extra

    fitting = [m for m in (8192, 10240, 12288, 16384, 20480, 24576) if device_gib(m) <= budget]
    if args.micro_tokens:
        micro = args.micro_tokens
    elif fitting:
        micro = fitting[-1]
    else:
        raise SystemExit(f"{memory:.1f} GiB per GPU is too small for the 700m preset: even 8,192-token "
                         f"microbatches need ~{device_gib(8192):.0f} GiB. (Stock CMP 170HX cards have 8 GB.)")
    slow_pcie = pcie < 8.0
    target = args.update_tokens or (1_048_576 if slow_pcie and world > 1 else 786_432)
    accum = max(1, round(target / (world * micro)))
    update_tokens = world * micro * accum
    steps = math.ceil(args.total_tokens / update_tokens)
    warmup = max(1, round(236e6 / update_tokens))       # same token budget as the A100 schedule

    # Rough per-update time budget (estimates scaled from A100 measurements, not measurements).
    per_gpu = A100_FWD_BWD_TOKENS_PER_S * min(1.0, bf16 / A100_BF16_TFLOPS)
    h2d = min(g["h2d_gbs"] for g in gpus) * 1e9
    local_tokens = micro * accum
    budget_s = dict(
        forward_backward=local_tokens / per_gpu,
        # n-gram rows (4 x 512 BF16 per token) are copied in-stream before each forward;
        # their FP32 gradients return on a side stream and overlap with compute.
        ngram_rows_h2d=local_tokens * 4 * 512 * 2 / h2d,
        # DDP reduces gradients during the last microbatch's backward; most of it is exposed.
        dense_allreduce=(2 * 716e6 * (2 if slow_pcie else 4) * (world - 1) / world / (pcie * 1e9)
                         if world > 1 else 0.0),
        # gloo all-reduce of the 4.8 GB FP32 accumulator in host memory (~3 GB/s loopback).
        ngram_reduce=2 * 4.8 * (world - 1) / world / 3.0 if world > 1 else 0.0,
        # every rank streams the whole table; ranks share the host's memory bandwidth.
        table_update=1.4 * world,
        dion3=0.16 * A100_BF16_TFLOPS / bf16,
        # replicated Dion3: rank 0 broadcasts FP32 weights + momentum (~5.4 GB) every 50 updates.
        replica_resync=5.4e9 / (pcie * 1e9) / 50 if world > 1 and slow_pcie else 0.0,
    )
    total_s = sum(budget_s.values())
    estimate = dict(seconds_per_update={k: round(v, 1) for k, v in budget_s.items()},
                    total_s=round(total_s, 1), tokens_s=round(update_tokens / total_s),
                    per_gpu_compute_tokens_s=round(per_gpu))
    ram_need = 35 + 20 * (world - 1)                    # measured 35 GiB peak RSS for one rank
    warnings = []
    if slow_pcie:
        warnings.append(f"host link {pcie:.1f} GB/s (PCIe Gen1 x4 is ~0.8): n-gram rows and gradients and the "
                        "dense all-reduce cross it; using gloo n-gram sync, BF16 all-reduce and ~1M-token updates")
    if any(g["fp32_tflops"] < 2.0 for g in gpus):
        warnings.append("FP32 (non-tensor-core) GEMMs are throttled on this GPU; the model's GEMMs use BF16 "
                        "tensor cores, but FP32 elementwise/optimizer work may run slower than on an A100")
    if hw["host_ram_gib"] < ram_need:
        warnings.append(f"host RAM {hw['host_ram_gib']:.0f} GiB < ~{ram_need} GiB needed "
                        f"(table + FP32 accumulator per rank, checkpoint snapshots)")
    if world > 1 and not all(gpus[0]["index"] in g["peers"] or g is gpus[0] for g in gpus):
        warnings.append("no GPU peer access: NCCL routes through host memory (expected without NVLink)")
    config = dict(world=world, micro_tokens=micro, grad_accum=accum, update_tokens=update_tokens,
                  steps=steps, warmup_steps=warmup,
                  gpu_mem_cap_gib=round(min(memory - 1.0, device_gib(micro) - DEVICE_EXTRA_GIB + 2.0), 1),
                  update_threads=max(1, hw["cpu_cores"] // world),
                  ngram_sync="gloo" if world > 1 else "default",
                  ddp_comm_hook="bf16" if world > 1 and slow_pcie else "none",
                  dion_replicated=world > 1 and slow_pcie,
                  predicted_device_gib=round(device_gib(micro), 1))
    return config, estimate, warnings


def command(config, args, extra):
    trainer = [
        "trainer.py", "--preset", "700m", "--ngram-params", str(args.ngram_params),
        "--micro-tokens", str(config["micro_tokens"]), "--grad-accum", str(config["grad_accum"]),
        "--steps", str(config["steps"]), "--warmup-steps", str(config["warmup_steps"]),
        "--gpu-mem-cap-gib", str(config["gpu_mem_cap_gib"]),
        "--update-threads", str(config["update_threads"]),
        "--data", args.data, "--eval-data", args.eval_data, "--tokenizer", args.tokenizer,
        "--ckpt-dir", args.ckpt_dir, "--ckpt-minutes", str(args.ckpt_minutes), "--keep", "2",
        "--resume", "auto", "--eval-every", str(args.eval_every), "--eval-exit-at", "24",
        "--metrics", "metrics-run.jsonl", "--output", "results-run.jsonl"]
    if config["world"] > 1:
        trainer += ["--ngram-sync", config["ngram_sync"], "--ddp-comm-hook", config["ddp_comm_hook"]]
        if config["dion_replicated"]:
            trainer += ["--dion-replicated"]
        return ["torchrun", "--standalone", "--nproc_per_node", str(config["world"])] + trainer + extra
    return [sys.executable, "-u"] + trainer + extra


def ensure_model_patched():
    """The two source fixes iter3_colab.sh applies on Colab (flat fused-loss call, 2-D conv weights)."""
    path = HERE / "model.py"
    if not path.exists():
        raise SystemExit("run.py must sit in the a100stack source directory with this repository's .py files")
    source = patched = path.read_text()
    patched = patched.replace("self.loss_fn(x.unsqueeze(0), y.unsqueeze(0), self.lm_head.weight)",
                              "self.loss_fn(x, y, self.lm_head.weight)")
    patched = patched.replace(
        "F.conv1d(xi, conv.weight, groups=x.shape[1])",
        "F.conv1d(xi, conv.weight.unsqueeze(1) if conv.weight.ndim == 2 else conv.weight, groups=x.shape[1])")
    if patched != source:
        path.write_text(patched)
        print("PATCHED model.py (loss call, conv weights)", flush=True)


def supervise(cmd, config, args):
    env = dict(os.environ)
    cache = Path(args.cache_dir)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
    env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    env["OMP_NUM_THREADS"] = str(config["update_threads"])   # torchrun would otherwise set 1
    env.setdefault("NCCL_DEBUG", "WARN")
    if config["world"] > 1:
        env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(config["world"])))
    state = dict(child=None, stop=False)

    def forward(signum, frame):
        state["stop"] = True
        if state["child"] is not None and state["child"].poll() is None:
            os.killpg(state["child"].pid, signal.SIGTERM)    # trainer: finish update, checkpoint, exit 0
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    for attempt in range(args.max_restarts + 1):
        print(f"SUPERVISOR start {attempt} {time.strftime('%H:%M:%S')}", flush=True)
        state["child"] = subprocess.Popen(cmd + ["--run-name", f"run-a{attempt}"], cwd=HERE, env=env,
                                          start_new_session=True)
        code = state["child"].wait()
        if code == 0 or state["stop"]:
            print(f"SUPERVISOR done (exit {code})", flush=True)
            return code
        print(f"SUPERVISOR trainer exited with {code}; resuming from the last checkpoint in 15 s", flush=True)
        time.sleep(15)
    print("SUPERVISOR giving up", flush=True)
    return 1


def main():
    if "--probe-worker" in sys.argv:
        return probe_worker()
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, default=0, help="use the first N GPUs (default: all)")
    p.add_argument("--micro-tokens", type=int, default=0, help="override the memory-based choice")
    p.add_argument("--update-tokens", type=int, default=0, help="target tokens per update")
    p.add_argument("--memory-fraction", type=float, default=0.85, help="share of GPU memory to plan for")
    p.add_argument("--total-tokens", type=float, default=10e9, help="schedule length in tokens")
    p.add_argument("--ngram-params", type=int, default=1_200_000_000)
    p.add_argument("--tokenizer", default=next(
        (str(t) for t in (HERE / "good_tokenizer/tokenizer.json", HERE.parent / "good_tokenizer/tokenizer.json",
                          Path("/content/good_tokenizer/tokenizer.json")) if t.exists()),
        "good_tokenizer/tokenizer.json"))
    p.add_argument("--data", default="fineweb-train-train.bin")
    p.add_argument("--eval-data", default="fineweb-train-val.bin")
    p.add_argument("--ckpt-dir", default="ckpt-run")
    p.add_argument("--cache-dir", default="compile-cache")
    p.add_argument("--ckpt-minutes", type=float, default=30)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--max-restarts", type=int, default=20)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    hw = probe()
    config, estimate, warnings = plan(hw, args)
    print("HARDWARE " + json.dumps(hw, indent=1), flush=True)
    print("PLAN " + json.dumps(config, indent=1), flush=True)
    print("ESTIMATE (rough, from A100 measurements) " + json.dumps(estimate, indent=1), flush=True)
    for warning in warnings:
        print("WARNING " + warning, flush=True)
    if args.probe_only:
        return 0
    ensure_model_patched()
    if not Path(args.tokenizer).exists():
        raise SystemExit(f"tokenizer not found at {args.tokenizer}; pass --tokenizer")
    for path in (args.data, args.eval_data):
        if not (HERE / path).exists():
            raise SystemExit(f"missing {path}: run prepare_fineweb.py first or pass --data/--eval-data")
    disk = shutil.disk_usage(HERE).free / 2**30
    need = 3 * (8.8 + 3.2 * (config["world"] - 1))
    if disk < need:
        print(f"WARNING {disk:.0f} GiB free disk; keeping 2 checkpoints (+1 in flight) needs ~{need:.0f} GiB")
    cmd = command(config, args, extra)
    print("COMMAND " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    return supervise(cmd, config, args)


if __name__ == "__main__":
    sys.exit(main() or 0)
