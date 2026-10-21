"""Recoverable training for a100stack: dense hybrid LM + hashed n-gram table in pinned RAM.

Built from the iteration-3/4 harness pieces (packing, compiled tails, fast CE, the staged
n-gram path, Dion3) plus what a long run needs:

* Deterministic streaming data: update s, rank r and microbatch m always map to the same
  window, so a resumed run sees exactly the data the uninterrupted run would have.
* WSD learning-rate schedule, also applied to the sparse table optimizer.
* Atomic checkpoints written by a background thread: dense weights, n-gram table and row
  state, optimizer state per rank, RNG streams, layer-dropout RNG and counters. FP64
  checksums are stored and re-verified against the live state after a resume.
* --resume auto continues from the newest complete checkpoint and falls back to older ones
  if a load fails. SIGTERM/SIGINT and --max-minutes stop after the current update with a
  checkpoint. Nonfinite updates are skipped (dense gradients and the n-gram accumulator are
  discarded); too many in a row exits with code 3 so a supervisor restarts from a checkpoint.

One GPU:    python -u trainer.py --data fineweb-train-train.bin --eval-data fineweb-train-val.bin
Supervised: bash train700m.sh   (restarts with --resume auto after a crash)
Multi-GPU:  torchrun --nproc_per_node N trainer.py ...
"""
import argparse
import contextlib
import json
import math
import os
import queue
import random
import shutil
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

import bench100m as bench
import model as stack
from config import ModelConfig, NgramConfig

NGRAM_KEYS = ("ngram.table", "ngram.row_v")


def model_config(name, ngram_params):
    ngram = NgramConfig(dim=512, total_params=ngram_params, table_dtype="bfloat16", inject_layers=(1,))
    if name == "700m":
        # The source's 500m preset (32 layers, 3:1 DeltaNet:attention, head dim 128), widened to 1,280.
        return ModelConfig(d_model=1280, n_layers=32, n_heads=10, n_kv_heads=2, head_dim=128,
                           gdn_heads=10, gdn_head_dim=128, ffn_dim=3200, ngram=ngram)
    if name == "500m":
        return ModelConfig(ngram=ngram)
    if name == "100m":
        c = bench.config()
        c.ngram = ngram
        return c
    raise ValueError(name)


# ----------------------------------------------------------------------------- data
class Windows:
    """Global microbatch g -> fixed (tokens+1)-token window, with a fresh block order per epoch."""

    def __init__(self, path, tokens, seed):
        self.raw = np.memmap(path, dtype=np.uint16, mode="r")
        self.tokens, self.seed = tokens, seed
        self.blocks = (len(self.raw) - 1) // tokens
        assert self.blocks > 0, f"{path} is smaller than one microbatch"
        self._epoch, self._order = None, None

    def __call__(self, g):
        epoch, i = divmod(g, self.blocks)
        if epoch != self._epoch:
            self._epoch = epoch
            self._order = np.random.default_rng(self.seed + epoch).permutation(self.blocks)
        start = int(self._order[i]) * self.tokens
        return self.raw[start:start + self.tokens + 1]


def host_batch(window, eos, max_doc):
    """bench100m.build_batch on the host: pinned tensors, so the copies can be asynchronous."""
    ids, targets = window[:-1].astype(np.int64), window[1:].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(ids[:-1] == eos) + 1]
    ends = np.r_[starts[1:], len(ids)]
    starts = np.concatenate([np.arange(s, e, max_doc) for s, e in zip(starts, ends)])
    cu = np.r_[starts, len(ids)].astype(np.int32)
    pos = np.arange(len(ids)) - np.repeat(starts, np.diff(cu))
    targets[ids == eos] = -100
    cu_cpu = torch.from_numpy(cu)
    batch = dict(ids=torch.from_numpy(ids).pin_memory(), targets=torch.from_numpy(targets).pin_memory(),
                 pos=torch.from_numpy(pos).pin_memory(), cu=cu_cpu.pin_memory(),
                 cu_cpu=cu_cpu, cu_cpu_long=cu_cpu.long(), max_len=int(np.diff(cu).max()),
                 valid_tokens=int((targets != -100).sum()))
    batch["ids_cpu"], batch["pos_cpu"] = batch["ids"], batch["pos"]  # the n-gram hash runs on the host
    return batch


def to_device(batch):
    out = dict(batch)
    for key in ("ids", "targets", "pos", "cu"):
        out[key] = batch[key].to("cuda", non_blocking=True)
    return out


class Loader:
    """Background thread: builds this rank's microbatches and each update's global token count."""

    def __init__(self, windows, eos, max_doc, start_step, accum, rank, world):
        self.windows, self.eos, self.max_doc = windows, eos, max_doc
        self.step, self.accum, self.rank, self.world = start_step, accum, rank, world
        self.queue = queue.Queue(maxsize=2 * accum)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            per = self.world * self.accum
            while True:
                # g = (step*world + rank)*accum + micro, as in bench100m: W ranks x accumulation A
                # see the same tokens per update as one rank with accumulation A*W.
                windows = [self.windows(self.step * per + j) for j in range(per)]
                valid = sum(len(w) - 1 - int((w[:-1] == self.eos).sum()) for w in windows)
                for micro in range(self.accum):
                    batch = host_batch(windows[self.rank * self.accum + micro], self.eos, self.max_doc)
                    self.queue.put((self.step, micro, batch, valid))
                self.step += 1
        except BaseException as error:  # surfaced in the training thread
            self.queue.put(error)

    def next(self, step, micro):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        assert item[:2] == (step, micro), (item[:2], step, micro)
        return to_device(item[2]), item[3]


def lr_multiplier(step, args):
    """Warmup-stable-decay, as in the source's optim.wsd_lr."""
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    decay_start = int(args.steps * (1 - args.decay_frac))
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / max(1, args.steps - decay_start)
    return args.min_lr_frac + (1 - args.min_lr_frac) * (1 - math.sqrt(min(1.0, t)))


# ----------------------------------------------------------------------------- checkpoints
def to_host(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {key: to_host(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(value) for value in obj)
    return obj


def checksum(obj):
    """FP64 sum over every tensor in a nested structure (no full-size FP64 temporaries)."""
    if torch.is_tensor(obj):
        tensor = obj.to(torch.uint8) if obj.dtype == torch.bool else obj
        return float(torch.sum(tensor, dtype=torch.float64))
    if isinstance(obj, dict):
        return sum(checksum(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(checksum(value) for value in obj)
    return 0.0


def same_sum(a, b):
    return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def save_file(obj, path):
    with open(path, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())


class Checkpointer:
    """step_XXXXXXXX.partial/ -> files -> per-rank markers -> meta.json, COMPLETE -> rename.

    A directory without COMPLETE (or still named .partial) is never resumed from.
    Rank 0 renames once every rank's marker exists, so no collective runs off the main thread.
    """

    def __init__(self, root, keep, rank, world):
        self.root, self.keep, self.rank, self.world = Path(root), keep, rank, world
        self.thread, self.error = None, None
        if rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            for stale in self.root.glob("*.partial"):
                shutil.rmtree(stale, ignore_errors=True)

    def complete(self):
        return sorted((d for d in self.root.glob("step_*")
                       if d.is_dir() and not d.name.endswith(".partial") and (d / "COMPLETE").exists()),
                      key=lambda d: d.name)

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("checkpoint write failed") from error

    def save(self, step, files, meta, blocking=False):
        self.wait()
        final = self.root / f"step_{step:08d}"
        partial = self.root / f"step_{step:08d}.partial"

        def write():
            try:
                begin = time.perf_counter()
                partial.mkdir(parents=True, exist_ok=True)
                sums = {}
                for name, obj in files.items():
                    save_file(obj, partial / f"{name}.pt")
                    sums[name] = checksum(obj)
                (partial / f"rank{self.rank}.json").write_text(json.dumps(sums))
                if self.rank != 0:
                    return
                deadline = time.time() + 1800
                while not all((partial / f"rank{r}.json").exists() for r in range(self.world)):
                    if time.time() > deadline:
                        raise TimeoutError(f"ranks did not finish writing {partial}")
                    time.sleep(0.2)
                meta["checksums"] = {}
                for r in range(self.world):
                    meta["checksums"].update(json.loads((partial / f"rank{r}.json").read_text()))
                meta["write_seconds"] = time.perf_counter() - begin
                (partial / "meta.json").write_text(json.dumps(meta, indent=2))
                with open(partial / "COMPLETE", "w") as handle:
                    handle.write(str(step))
                    handle.flush()
                    os.fsync(handle.fileno())
                if final.exists():  # same step saved again (e.g. stop right after a resume)
                    shutil.rmtree(final)
                partial.rename(final)
                for old in self.complete()[:-self.keep]:
                    shutil.rmtree(old, ignore_errors=True)
                print("CHECKPOINT", json.dumps(dict(step=step, path=str(final),
                                                    seconds=round(meta["write_seconds"], 1))), flush=True)
            except BaseException as error:
                self.error = error

        self.thread = threading.Thread(target=write, name=f"ckpt-{step}")
        self.thread.start()
        if blocking:
            self.wait()


def discard_sparse(memory):
    """Drop this update's n-gram gradients (after a nonfinite loss or gradient norm)."""
    for slot in memory._pin_slots:
        if slot["future"] is not None:
            slot["future"].result()
    memory._host_grad.zero_()
    memory._host_touched.zero_()


# ----------------------------------------------------------------------------- main
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("700m", "500m", "100m"), default="700m")
    p.add_argument("--ngram-params", type=int, default=1_200_000_000)
    p.add_argument("--data", default="fineweb-train-train.bin")
    p.add_argument("--eval-data", default="fineweb-train-val.bin")
    p.add_argument("--micro-tokens", type=int, default=16384)
    p.add_argument("--seq-len", type=int, default=1024, help="maximum packed document length")
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--steps", type=int, default=10_000, help="schedule length in updates")
    p.add_argument("--stop-at", type=int, default=0, help="end this invocation after this update")
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--decay-frac", type=float, default=0.2)
    p.add_argument("--min-lr-frac", type=float, default=0.0)
    p.add_argument("--optimizer", choices=("dion3", "adamw"), default="dion3")
    p.add_argument("--dion-lr", type=float, default=0.02)
    p.add_argument("--dion-fraction", type=float, default=0.5)
    p.add_argument("--dion-mu", type=float, default=0.95)
    p.add_argument("--dion-weight-decay", type=float, default=0.01)
    p.add_argument("--dion-triton", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scalar-lr", type=float, default=3e-3)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--compile", choices=("none", "mlp", "tail"), default="tail")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--loss-chunks", type=int, default=4)
    p.add_argument("--fast-ce", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--update-threads", type=int, default=os.cpu_count())
    p.add_argument("--layer-drop", type=float, default=0.0)
    p.add_argument("--layer-drop-dist", choices=("ild", "uniform", "ald"), default="ild")
    p.add_argument("--layer-drop-schedule", choices=("decreasing", "constant", "increasing"),
                   default="decreasing")
    p.add_argument("--depth-attn", choices=("off", "softmax", "uniform"), default="off")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-tokens", type=int, default=1_048_576)
    p.add_argument("--eval-exit-at", type=int, default=0)
    p.add_argument("--ckpt-dir", default="ckpt700m")
    p.add_argument("--ckpt-every", type=int, default=0, help="updates between checkpoints (0: off)")
    p.add_argument("--ckpt-minutes", type=float, default=0, help="also checkpoint after this much time")
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--resume", default="auto", help="auto | none | checkpoint directory")
    p.add_argument("--max-minutes", type=float, default=0, help="stop (with a checkpoint) after this long")
    p.add_argument("--max-skips", type=int, default=3, help="consecutive nonfinite updates before exit 3")
    p.add_argument("--metrics", default="metrics700m.jsonl")
    p.add_argument("--output", default="results700m.jsonl")
    p.add_argument("--run-name", default="run")
    p.add_argument("--timing-skip", type=int, default=2, help="updates excluded from the throughput summary")
    p.add_argument("--peak-tflops", type=float, default=312.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl")
    p.add_argument("--crash-at-step", type=int, default=-1, help="test hook: hard exit during this update")
    p.add_argument("--crash-point", choices=("mid-update", "during-save"), default="mid-update")
    return p.parse_args()


def main():
    args = parse()
    distributed = "RANK" in os.environ
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(args.dist_backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert stack.USE_FLA, "FLA required"

    stop = dict(reason=None)

    def request_stop(signum, frame):
        stop["reason"] = signal.Signals(signum).name
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    c = model_config(args.preset, args.ngram_params)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/content/good_tokenizer/tokenizer.json")
    assert max(tok.get_vocab().values()) + 1 == c.vocab_size
    eos = tok.token_to_id("<eos>")

    torch.manual_seed(args.seed)
    with torch.device("cuda"):
        model = stack.HybridLM(c)
    if distributed:
        model.ngram.init_sync()
    model._loss = MethodType(bench.flat_fused_loss, model)
    if args.fast_ce:
        import fast_flce
        fast_flce.install()
        model.register_forward_pre_hook(lambda module, inputs: fast_flce.set_next_total(inputs[0]["valid_tokens"]))
    group = dist.group.WORLD if distributed else None
    from ram_ngram import enable_ram_ngram
    from pinned_ngram import attach_cpu_inputs, enable_pinned
    enable_ram_ngram(model.ngram)
    collect = enable_pinned(model.ngram, staged=True, sparse_update="blocks",
                            update_threads=args.update_threads, process_group=group)
    if not distributed:
        attach_cpu_inputs(model)
    torch.cuda.empty_cache()
    model.train()
    report = model.param_report()
    log("MODEL", json.dumps(dict(preset=args.preset, **report, rows_per_table=c.ngram.rows_per_table)))
    opts = SimpleNamespace(pack=True, pack_mode="concat", attn="native", compile=args.compile,
                           compile_mode=args.compile_mode, ckpt="none", residual_dtype="float32",
                           fused_norms=True, loss_chunks=args.loss_chunks)
    from optim800k import apply_optimizations
    apply_optimizations(model, opts)
    bench.optimize(model, opts)
    if args.depth_attn != "off":
        from depth_attention import enable_depth_attention
        log("DEPTH_ATTENTION", json.dumps(enable_depth_attention(model, args.depth_attn)))
    layer_drop = None
    if args.layer_drop > 0 or args.eval_exit_at:
        import layer_dropout
        layer_dropout.install(model, args.compile_mode)
        if args.layer_drop > 0:
            layer_drop = layer_dropout.LayerDropout(len(model.blocks), args.layer_drop, args.layer_drop_dist,
                                                    args.layer_drop_schedule, args.steps, args.seed)

    # ---- resume: weights and table before DDP and the optimizer, optimizer state after
    ckpt = Checkpointer(args.ckpt_dir, args.keep, rank, world)
    if distributed:
        dist.barrier()
    candidates = []
    if args.resume == "auto":
        candidates = ckpt.complete()[::-1]
    elif args.resume != "none":
        candidates = [Path(args.resume)]
    resumed, own = None, None
    for directory in candidates:
        try:
            meta = json.loads((directory / "meta.json").read_text())
            assert meta["world_size"] == world, f"saved with {meta['world_size']} ranks, running {world}"
            assert meta["preset"] == args.preset and meta["ngram_params"] == args.ngram_params
            dense = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
            ngram = torch.load(directory / "ngram.pt", map_location="cpu", weights_only=False)
            own = {name: torch.load(directory / f"{name}_rank{rank}.pt", map_location="cpu", weights_only=False)
                   for name in ("optim", "state")}
            missing, unexpected = model.load_state_dict(dense, strict=False)
            assert not unexpected and sorted(missing) == sorted(NGRAM_KEYS), (missing, unexpected)
            model.ngram.table.copy_(ngram["table"])
            model.ngram.row_v.copy_(ngram["row_v"])
            resumed = (directory, meta)
            del dense, ngram
            break
        except Exception as error:  # a damaged checkpoint: try the next older one
            log("RESUME_SKIPPED", str(directory), repr(error)[:300])
            own = None
    if candidates and resumed is None:
        raise RuntimeError("no loadable checkpoint among " + ", ".join(map(str, candidates)))

    train_model = model
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        DDP._set_params_and_buffers_to_ignore_for_model(model, list(NGRAM_KEYS))
        train_model = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False,
                          gradient_as_bucket_view=True, bucket_cap_mb=64)
        attach_cpu_inputs(train_model, model.ngram)
    if args.optimizer == "dion3":
        from dion_optim import build_dion3
        optimizer, groups = build_dion3(model, args, train_model.process_group if distributed else None)
        log("OPTIMIZER", json.dumps(dict(name="dion3", world=world, **groups)))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr, betas=(0.9, 0.95),
                                      weight_decay=0.01, fused=True)
    for group_ in optimizer.param_groups:
        group_["base_lr"] = group_["lr"]

    start_step, tokens_seen, train_seconds, skipped = 0, 0, 0.0, 0
    if resumed is not None:
        directory, meta = resumed
        optimizer.load_state_dict(own["optim"])
        state = own["state"]
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state(state["torch_cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if layer_drop is not None and state.get("layer_drop"):
            layer_drop.rng.bit_generator.state = state["layer_drop"]["rng"]
            layer_drop.dropped, layer_drop.drawn = state["layer_drop"]["dropped"], state["layer_drop"]["drawn"]
        start_step, tokens_seen = meta["step"], meta["tokens_seen"]
        train_seconds, skipped = meta["train_seconds"], meta["skipped_updates"]
        # The live state must reproduce the checksums computed when the files were written.
        live = dict(model=checksum({k: v for k, v in model.state_dict().items() if k not in NGRAM_KEYS}),
                    ngram=checksum(dict(table=model.ngram.table, row_v=model.ngram.row_v)))
        live[f"optim_rank{rank}"] = checksum(optimizer.state_dict())
        bad = {k: (v, meta["checksums"][k]) for k, v in live.items() if not same_sum(v, meta["checksums"][k])}
        assert not bad, f"resumed state differs from the checkpoint: {bad}"
        print("RESUMED", json.dumps(dict(rank=rank, path=str(directory), step=start_step,
                                         tokens_seen=tokens_seen, verified=sorted(live))), flush=True)
        del own

    def snapshot(step):
        """CPU copies taken at an update boundary; the write happens in the background."""
        torch.cuda.synchronize()
        files = {f"optim_rank{rank}": to_host(optimizer.state_dict()),
                 f"state_rank{rank}": dict(
                     torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(),
                     numpy=np.random.get_state(), python=random.getstate(),
                     layer_drop=None if layer_drop is None else dict(
                         rng=layer_drop.rng.bit_generator.state, dropped=layer_drop.dropped,
                         drawn=layer_drop.drawn))}
        if rank == 0:
            files["model"] = {k: v.detach().to("cpu", copy=True)
                              for k, v in model.state_dict().items() if k not in NGRAM_KEYS}
            files["ngram"] = dict(table=model.ngram.table.clone(), row_v=model.ngram.row_v.clone())
        meta = dict(step=step, tokens_seen=tokens_seen, train_seconds=train_seconds,
                    skipped_updates=skipped, world_size=world, preset=args.preset,
                    ngram_params=args.ngram_params, args=vars(args), model_config=asdict(c),
                    saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        return files, meta

    windows = Windows(args.data, args.micro_tokens, args.seed)
    loader = Loader(windows, eos, args.seq_len, start_step, args.grad_accum, rank, world)
    eval_batches = []
    if args.eval_data:
        held_out = np.memmap(args.eval_data, mode="r", dtype=np.uint16)
        count = min(args.eval_tokens, len(held_out) - 1) // args.micro_tokens
        eval_batches = [host_batch(held_out[w * args.micro_tokens:][:args.micro_tokens + 1], eos, args.seq_len)
                        for w in range(rank, count, world)]

    def evaluate():
        def held_out_loss(plan=None):
            model._layer_plan = plan
            total = torch.zeros(2, dtype=torch.float64, device="cuda")
            with torch.no_grad():
                for batch in eval_batches:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(to_device(batch))
                    total += torch.stack([loss.double() * batch["valid_tokens"],
                                          torch.tensor(float(batch["valid_tokens"]), device="cuda",
                                                       dtype=torch.float64)])
            if distributed:
                dist.all_reduce(total)
            model._layer_plan = None
            return float(total[0] / total[1])
        result = dict(eval_loss=held_out_loss())
        if args.eval_exit_at:
            from layer_dropout import exit_plan
            result["eval_loss_early_exit"] = held_out_loss(exit_plan(len(model.blocks), args.eval_exit_at))
        return result

    metrics = open(args.metrics, "a") if rank == 0 else None
    last_step = min(args.steps, args.stop_at or args.steps)
    log("CONFIG", json.dumps(dict(start_step=start_step, last_step=last_step, world=world,
                                  update_tokens=args.micro_tokens * args.grad_accum * world,
                                  data_blocks=windows.blocks, eval_microbatches=len(eval_batches))))
    timed, step_records = [], []
    last_ckpt_time, consecutive_skips, evaluation = time.time(), 0, {}
    peak_tflops = args.peak_tflops * 1e12
    for step in range(start_step, last_step):
        begin = time.perf_counter()
        mult = lr_multiplier(step, args)
        for group_ in optimizer.param_groups:
            group_["lr"] = group_["base_lr"] * mult
        optimizer.zero_grad(set_to_none=True)
        values, flops, update_tokens = [], 0.0, 0
        for micro in range(args.grad_accum):
            batch, step_valid = loader.next(step, micro)
            if step == args.crash_at_step and args.crash_point == "mid-update" and micro == args.grad_accum // 2:
                print("CRASH_TEST", json.dumps(dict(step=step, micro=micro)), flush=True)
                os._exit(17)
            sync = not distributed or micro == args.grad_accum - 1
            if layer_drop is not None:
                model._layer_plan = None if distributed and sync else layer_drop.plan(step)
            with contextlib.nullcontext() if sync else train_model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = train_model(batch)
                weight = batch["valid_tokens"] / step_valid * world
                (loss * weight).backward()
            collect(model.ngram)
            values.append(loss.detach() * weight)
            flops += model.flops_per_step(batch["cu_cpu"], args.micro_tokens)
            update_tokens += args.micro_tokens
        total = torch.stack(values).sum()
        if distributed:
            dist.all_reduce(total)
            total /= world
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=True)
        dense_begin = time.perf_counter()
        loss_value, gnorm_value = float(total), float(gnorm)
        if not (math.isfinite(loss_value) and math.isfinite(gnorm_value)):
            optimizer.zero_grad(set_to_none=True)
            discard_sparse(model.ngram)
            skipped += 1
            consecutive_skips += 1
            log("SKIPPED_UPDATE", json.dumps(dict(step=step, loss=loss_value, gnorm=gnorm_value)))
            if consecutive_skips > args.max_skips:
                log("TOO_MANY_NONFINITE_UPDATES: exiting for a restart from the last checkpoint")
                raise SystemExit(3)
            rows, dense_seconds, sparse_seconds = 0, 0.0, 0.0
        else:
            consecutive_skips = 0
            optimizer.step()
            torch.cuda.synchronize()
            sparse_begin = time.perf_counter()
            rows = model.ngram.sparse_step(mult)
            dense_seconds, sparse_seconds = sparse_begin - dense_begin, time.perf_counter() - sparse_begin
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        tokens_seen += update_tokens * world
        train_seconds += seconds
        done = step + 1
        record = dict(step=done, loss=loss_value, lr_mult=mult, gnorm=gnorm_value,
                      tok_s=update_tokens * world / seconds, mfu=flops / seconds / peak_tflops,
                      update_s=seconds, dense_opt_s=dense_seconds, sparse_s=sparse_seconds, rows=rows,
                      tokens_seen=tokens_seen, peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, run=args.run_name)
        if args.eval_every and done % args.eval_every == 0:
            record.update(evaluate())
        if step - start_step >= args.timing_skip:
            timed.append(record)
        step_records.append(record)
        log("STEP", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in record.items()}))
        if metrics:
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()

        # Collective decisions, so every rank checkpoints and stops at the same update.
        flags = torch.tensor([stop["reason"] is not None,
                              bool(args.max_minutes and time.time() - started > args.max_minutes * 60),
                              bool(args.ckpt_minutes and time.time() - last_ckpt_time > args.ckpt_minutes * 60)],
                             dtype=torch.int32, device="cuda")
        if distributed:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        stopping = bool(flags[0] or flags[1])
        due = bool(flags[2]) or (args.ckpt_every and done % args.ckpt_every == 0)
        final = done == last_step
        if due or stopping or (final and (args.ckpt_every or args.ckpt_minutes)):
            files, meta = snapshot(done)
            ckpt.save(done, files, meta, blocking=stopping or final)
            last_ckpt_time = time.time()
            if done == args.crash_at_step and args.crash_point == "during-save":
                print("CRASH_TEST", json.dumps(dict(step=done, point="during-save")), flush=True)
                os._exit(17)
        if stopping:
            log("STOPPED", json.dumps(dict(step=done, reason=stop["reason"] or "max-minutes")))
            break
    ckpt.wait()
    if eval_batches and not (args.eval_every and step_records and step_records[-1]["step"] % args.eval_every == 0):
        evaluation = evaluate()
    elif step_records:
        evaluation = {k: v for k, v in step_records[-1].items() if k.startswith("eval_")}
    if rank == 0 and step_records:
        result = dict(run=args.run_name, preset=args.preset, **report, world_size=world,
                      gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      update_tokens=args.micro_tokens * args.grad_accum * world,
                      micro_tokens=args.micro_tokens, grad_accum=args.grad_accum,
                      first_step=step_records[0]["step"], last_step=step_records[-1]["step"],
                      timed_updates=len(timed), skipped_updates=skipped,
                      tokens_per_second=(args.micro_tokens * args.grad_accum * world * len(timed)
                                         / sum(r["update_s"] for r in timed)) if timed else None,
                      median_update_s=float(np.median([r["update_s"] for r in timed])) if timed else None,
                      mean_mfu=float(np.mean([r["mfu"] for r in timed])) if timed else None,
                      mean_dense_opt_s=float(np.mean([r["dense_opt_s"] for r in timed])) if timed else None,
                      mean_sparse_s=float(np.mean([r["sparse_s"] for r in timed])) if timed else None,
                      peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      host_peak_rss_gib=__import__("resource").getrusage(
                          __import__("resource").RUSAGE_SELF).ru_maxrss / 2**20,
                      final_loss=step_records[-1]["loss"], losses=[r["loss"] for r in step_records],
                      cpu_input_fallbacks=model.ngram._cpu_input_fallbacks,
                      dynamo_unique_graphs=__import__("torch._dynamo.utils").
                      _dynamo.utils.counters["stats"]["unique_graphs"],
                      args=vars(args))
        result.update(evaluation)
        if layer_drop is not None:
            result.update(layer_drop.report())
        with open(args.output, "a") as handle:
            handle.write(json.dumps(result) + "\n")
        log("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("args", "losses")}))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
