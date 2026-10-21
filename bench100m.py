"""Short real-data throughput experiments for a100stack on a single GPU.

Run from the extracted a100stack directory, after installing FLA 0.5.2.
Uses the supplied tokenizer's bounded FineWeb sample, not random tokens.
Timing includes forward/backward, clipping, dense optimizer and sparse updates;
excludes sample preparation, warmup, compilation and evaluation. Batches are GPU resident.
"""
import argparse
import contextlib
import gc
import inspect
import json
import math
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model as stack
from config import ModelConfig, NgramConfig


def config():
    return ModelConfig(
        vocab_size=60800, d_model=512, n_layers=20, full_attn_every=4,
        n_heads=4, n_kv_heads=1, head_dim=128,
        gdn_heads=4, gdn_head_dim=128, ffn_dim=1408,
        ngram=NgramConfig(dim=512, total_params=500_000_000,
                          table_dtype="bfloat16", inject_layers=(1,)),
    )


def flat_fused_loss(self, x, targets):
    """FLA 0.5.2 accepts flat packed states/targets, including the original ZIP source."""
    return self.loss_fn(x, targets, self.lm_head.weight)


def native_attention(self, x, batch, force_ref=False):
    from torch.nn.attention.varlen import varlen_attn
    c, n = self.c, x.shape[0]
    if hasattr(self, "in_proj"):
        q, k, v, gate = self.in_proj(x).split(
            (c.n_heads*c.head_dim, c.n_kv_heads*c.head_dim,
             c.n_kv_heads*c.head_dim, c.n_heads*c.head_dim), -1)
    else:
        q, k, v, gate = self.q_proj(x), self.k_proj(x), self.v_proj(x), self.gate_proj(x)
    q = self.q_norm(q.view(n, c.n_heads, c.head_dim))
    k = self.k_norm(k.view(n, c.n_kv_heads, c.head_dim))
    v = v.view(n, c.n_kv_heads, c.head_dim)
    if hasattr(self, "_depth_index"):
        # Same-position RoPE rotations cancel in q.k, so depth scores use pre-RoPE states.
        from depth_attention import depth_mix
        v = depth_mix(self, q, k, v)
    q = stack.rope(q, batch["pos"], c.rope_theta)
    k = stack.rope(k, batch["pos"], c.rope_theta)
    # PyTorch 2.11 uses the right window boundary to express causality.
    out = varlen_attn(q, k, v, batch["cu"], batch["cu"],
                      batch["max_len"], batch["max_len"],
                      window_size=(-1, 0))
    out = out * gate.view_as(out).sigmoid()
    return self.o_proj(out.reshape(n, -1))


def pack_linear(module, names, attribute="in_proj"):
    weights = [getattr(module, name).weight.detach() for name in names]
    linear = nn.Linear(weights[0].shape[1], sum(w.shape[0] for w in weights),
                       bias=False, device=weights[0].device, dtype=weights[0].dtype)
    with torch.no_grad():
        linear.weight.copy_(torch.cat(weights, 0))
    for name in names:
        delattr(module, name)
    setattr(module, attribute, linear)


def packed_gdn(self, x, batch):
    n, h, dk, dv = x.shape[0], self.H, self.dk, self.dv
    q, k, v, a, beta, gate = self.in_proj(x).split((h*dk, h*dk, h*dv, h, h, h*dv), -1)
    q, _ = self.q_conv1d(x=q.unsqueeze(0), cu_seqlens=batch["cu"])
    k, _ = self.k_conv1d(x=k.unsqueeze(0), cu_seqlens=batch["cu"])
    v, _ = self.v_conv1d(x=v.unsqueeze(0), cu_seqlens=batch["cu"])
    out, _ = stack.chunk_gated_delta_rule(
        q=q.reshape(1, n, h, dk), k=k.reshape(1, n, h, dk),
        v=v.reshape(1, n, h, dv), g=a.unsqueeze(0), beta=beta.unsqueeze(0),
        A_log=self.A_log, dt_bias=self.dt_bias,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, state_v_first=True,
        cu_seqlens=batch["cu"], cu_seqlens_cpu=batch["cu_cpu_long"])
    out = self.o_norm(out.squeeze(0), gate.reshape(n, h, dv))
    return self.o_proj(out.reshape(n, -1))


def packed_mlp(self, x):
    gate, up = self.in_proj(x).chunk(2, -1)
    return self.w_down(F.silu(gate) * up)


def block_tail(self, x, mixed):
    """Both FP32 gated residual adds, norm2 and the MLP as one static-shape region."""
    x = self.res1(x, mixed)
    return self.res2(x, self.mlp(self.norm2(x)))


def tail_forward(self, x, batch):
    return self._tail(x, self.mixer(self.norm1(x), batch))


def optimize(model, args):
    import types
    assert args.compile != "tail" or args.ckpt == "none", "tail compile has no checkpointed variant"
    model.full_ckpt = False
    # "fused" merges the weights into one Parameter; "concat" keeps the original
    # Parameters (as matrix optimizers and checkpoints expect) and fuses only the GEMM.
    pack = pack_linear
    if args.pack and getattr(args, "pack_mode", "fused") == "concat":
        from dion_optim import concat_projections as pack
    for block in model.blocks:
        block.ckpt_mlp = args.ckpt == "mlp"
        mixer = block.mixer
        if args.pack:
            if block.is_attn:
                pack(mixer, ("q_proj", "k_proj", "v_proj", "gate_proj"))
            else:
                pack(mixer, ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj"))
                mixer.forward = types.MethodType(packed_gdn, mixer)
            pack(block.mlp, ("w_gate", "w_up"))
            block.mlp.forward = types.MethodType(packed_mlp, block.mlp)
        if block.is_attn and args.attn == "native":
            mixer.forward = types.MethodType(native_attention, mixer)
        if args.compile == "mlp":
            block._mlp = torch.compile(block._mlp, fullgraph=True, dynamic=False,
                                       mode=getattr(args, "compile_mode", "default"))
        if args.compile == "tail":
            block._tail = torch.compile(types.MethodType(block_tail, block), fullgraph=True,
                                        dynamic=False, mode=getattr(args, "compile_mode", "default"))
            block.forward = types.MethodType(tail_forward, block)
        if args.ckpt == "mlp":
            block.ckpt_mlp = True


def build_batch(raw, eos, max_doc=1024):
    ids, targets = raw[:-1].astype(np.int64), raw[1:].astype(np.int64).copy()
    starts = np.r_[0, np.flatnonzero(ids[:-1] == eos) + 1]
    ends = np.r_[starts[1:], len(ids)]
    starts = np.concatenate([np.arange(s, e, max_doc) for s, e in zip(starts, ends)])
    cu = torch.tensor(np.r_[starts, len(ids)], dtype=torch.int32)
    pos = np.arange(len(ids)) - np.repeat(starts, np.diff(cu.numpy()))
    targets[ids == eos] = -100
    valid_tokens = int((targets != -100).sum())
    batch = dict(ids=torch.from_numpy(ids), targets=torch.from_numpy(targets),
                 pos=torch.from_numpy(pos), cu=cu)
    batch = {key: value.to("cuda") for key, value in batch.items()}
    batch.update(cu_cpu=cu, cu_cpu_long=cu.long(), max_len=int(np.diff(cu.numpy()).max()),
                 valid_tokens=valid_tokens, ids_cpu=torch.from_numpy(ids),
                 pos_cpu=torch.from_numpy(pos))
    return batch


def check_attention():
    """One small output and input-gradient parity check, including GQA and boundaries."""
    import copy
    c = config()
    baseline = stack.GatedAttention(c).cuda()
    fast = copy.deepcopy(baseline)
    from types import MethodType
    fast.forward = MethodType(native_attention, fast)
    x = torch.randn(129, 512, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    cu = torch.tensor([0, 33, 97, 129], dtype=torch.int32)
    batch = dict(cu=cu.cuda(), cu_cpu=cu, cu_cpu_long=cu.long(), max_len=64,
                 pos=torch.cat([torch.arange(n) for n in (33, 64, 32)]).cuda())
    upstream = torch.randn_like(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = baseline(x, batch)
        b = fast(x2, batch)
    (a.float()*upstream).sum().backward()
    (b.float()*upstream).sum().backward()
    rel = lambda p, q: float((p.float()-q.float()).norm()/p.float().norm().clamp_min(1e-9))
    errors = dict(output=rel(a, b), input_gradient=rel(x.grad, x2.grad))
    assert max(errors.values()) < 0.03, errors
    print("ATTENTION_PARITY", json.dumps(errors), flush=True)


def check_packing(model, batch):
    """Preserved-weight whole-model loss comparison before optimizer creation."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = float(model(batch))
    return before


def check_gdn_packing(model, batch):
    import copy
    import types
    original = next(block.mixer for block in model.blocks if not block.is_attn)
    packed = copy.deepcopy(original)
    pack_linear(packed, ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj"))
    packed.forward = types.MethodType(packed_gdn, packed)
    n = 129
    cu = torch.tensor([0, 33, 97, n], dtype=torch.int32)
    small = dict(cu=cu.cuda(), cu_cpu=cu, cu_cpu_long=cu.long(), max_len=64,
                 pos=torch.cat([torch.arange(s) for s in (33, 64, 32)]).cuda())
    x = torch.randn(n, 512, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = original(x, small), packed(x2, small)
    (a.float()*upstream).sum().backward()
    (b.float()*upstream).sum().backward()
    relative = lambda p, q: float((p.float()-q.float()).norm()/p.float().norm().clamp_min(1e-9))
    errors = dict(output=relative(a, b), input_gradient=relative(x.grad, x2.grad))
    assert max(errors.values()) < 0.03, errors
    model.zero_grad(set_to_none=True)
    print("GDN_PACKING_PARITY", json.dumps(errors), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Maximum packed document length; --tokens sets microbatch size")
    parser.add_argument("--ngram-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ram-mode", choices=("original", "pin", "staged"), default="original")
    parser.add_argument("--profile-ram", action="store_true")
    parser.add_argument("--sparse-update", choices=("rows", "blocks"), default="rows",
                        help="RAM-table row update: gather touched rows, or stream masked row blocks")
    parser.add_argument("--update-threads", type=int, default=None,
                        help="CPU threads for the end-of-step sparse update (GPU is idle then)")
    parser.add_argument("--fast-ce", action="store_true",
                        help="FLA fused CE without its per-call sync; BF16 x BF16 -> FP32 weight-gradient GEMM")
    parser.add_argument("--profile-kernels", type=int, default=0,
                        help="After timing, profile this many microbatches and print top CUDA kernels")
    parser.add_argument("--cuda-accum", action="store_true")
    parser.add_argument("--host-accum", action="store_true")
    parser.add_argument("--residual-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--fused-norms", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "max-autotune-no-cudagraphs", "reduce-overhead"), default="default")
    parser.add_argument("--loss-chunks", type=int, default=8)
    parser.add_argument("--token-weighted", action="store_true")
    parser.add_argument("--defer-loss-sync", action="store_true")
    parser.add_argument("--check-ram", action="store_true")
    parser.add_argument("--attn", choices=("loop", "native"), default="native")
    parser.add_argument("--compile", choices=("none", "mlp", "tail"), default="none",
                        help="tail also folds both gated residual adds into the compiled MLP region")
    parser.add_argument("--ckpt", choices=("none", "mlp"), default="none")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--pack-mode", choices=("fused", "concat"), default="fused",
                        help="concat keeps separate Parameters and fuses only the GEMM (needed by Dion3)")
    parser.add_argument("--optimizer", choices=("adamw", "dion3"), default="adamw",
                        help="dense optimizer; the n-gram table always uses its sparse row RMSprop")
    parser.add_argument("--dion-lr", type=float, default=0.02)
    parser.add_argument("--dion-fraction", type=float, default=0.5)
    parser.add_argument("--dion-mu", type=float, default=0.95)
    parser.add_argument("--dion-weight-decay", type=float, default=0.01)
    parser.add_argument("--dion-triton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scalar-lr", type=float, default=3e-3,
                        help="AdamW lr for embeddings, norms and scalars under --optimizer dion3")
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default="nccl",
                        help="under torchrun; gloo lets several ranks share one GPU for testing")
    parser.add_argument("--depth-attn", choices=("off", "softmax", "uniform"), default="off",
                        help="Depth-Attention value mixing across the gated-attention layers")
    parser.add_argument("--layer-drop", type=float, default=0.0,
                        help="max layer-dropout rate p_max (0 disables)")
    parser.add_argument("--layer-drop-dist", choices=("ild", "uniform", "ald"), default="ild")
    parser.add_argument("--layer-drop-schedule", choices=("decreasing", "constant", "increasing"),
                        default="decreasing")
    parser.add_argument("--layer-drop-horizon", type=int, default=0,
                        help="steps over which the schedule runs; default warmup + steps")
    parser.add_argument("--eval-exit-at", type=int, default=0,
                        help="also evaluate with only the first N blocks (early exit)")
    parser.add_argument("--eval-data", default=None,
                        help="held-out uint16 token file; reports mean loss after training")
    parser.add_argument("--eval-tokens", type=int, default=1048576)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checksums", action="store_true",
                        help="report FP64 sums of dense parameters and the n-gram table after training")
    parser.add_argument("--gdn-head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="results100m.jsonl")
    parser.add_argument("--data", default="fineweb-bench.bin")
    args = parser.parse_args()
    assert not args.pack or args.attn == "native"
    assert args.steps > 0 and args.warmup > 0 and args.grad_accum > 0
    assert not (args.host_accum and args.cuda_accum)
    assert args.optimizer != "dion3" or not args.pack or args.pack_mode == "concat", \
        "Dion3 orthogonalizes whole Parameters: use --pack-mode concat to keep q/k/v/gate separate"
    distributed = "RANK" in os.environ
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(args.dist_backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        # With gloo, several test ranks may share one GPU; NCCL needs one GPU per rank.
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count())
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert stack.USE_FLA, "FLA required; do not benchmark a sequential reference GDN"
    if args.check:
        check_attention()
        gc.collect()
        torch.cuda.empty_cache()
    if args.check_ram:
        from ram_ngram import check_ram_ngram
        check_ram_ngram()
        gc.collect()
        torch.cuda.empty_cache()
    c = config()
    c.gdn_head_dim = args.gdn_head_dim
    c.gdn_heads = c.d_model // c.gdn_head_dim
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/content/good_tokenizer/tokenizer.json")
    assert max(tok.get_vocab().values()) + 1 == c.vocab_size
    eos = tok.token_to_id("<eos>")
    raw = np.memmap(args.data, mode="r", dtype=np.uint16)
    count = args.steps + args.warmup
    # Global microbatch g = (step*world + rank)*accum + micro: one rank with accumulation
    # A*W sees the same data per update as W ranks with accumulation A each.
    window = lambda g: raw[(g*args.tokens) % (len(raw)-args.tokens):][:args.tokens+1]
    local = [(i*world + rank)*args.grad_accum + m for i in range(count) for m in range(args.grad_accum)]
    batches = [build_batch(window(g), eos, args.seq_len) for g in local]
    # Supervised targets per update across all ranks, from the fixture itself (no collective):
    # build_batch ignores exactly the targets whose input token is EOS.
    per_update = world*args.grad_accum
    global_valid = [sum(args.tokens - int((window(g)[:-1] == eos).sum())
                        for g in range(i*per_update, (i+1)*per_update)) for i in range(count)]
    torch.manual_seed(args.seed)
    with torch.device("cuda"):
        model = stack.HybridLM(c)
    if distributed:
        model.ngram.init_sync()  # rank 0's table everywhere, before it moves to RAM
    if model.loss_fn is not None:
        model._loss = MethodType(flat_fused_loss, model)
    if args.fast_ce:
        assert model.loss_fn is not None, "--fast-ce patches FLA's fused loss"
        import fast_flce
        fast_flce.install()
        # The count is known on the CPU from batch preparation; no per-microbatch sync.
        model.register_forward_pre_hook(lambda module, inputs: fast_flce.set_next_total(inputs[0]["valid_tokens"]))
    staged = args.host_accum or (args.ngram_device == "cpu" and args.ram_mode == "staged")
    assert args.sparse_update == "rows" or (staged and not args.host_accum), \
        "--sparse-update blocks applies to --ram-mode staged with the RAM table"
    # Replicas stay identical only where the sparse step reduces gradients across ranks:
    # the staged paths here, or the original GPU table (its sparse_step all-gathers rows).
    assert not distributed or staged or (args.ngram_device == "cuda" and not args.cuda_accum), \
        "multi-GPU needs --ram-mode staged, --host-accum, or the plain GPU table"
    group = dist.group.WORLD if distributed else None
    if args.ngram_device == "cpu":
        from ram_ngram import enable_ram_ngram, collect_ram_grads
        enable_ram_ngram(model.ngram)
        if args.ram_mode != "original":
            from pinned_ngram import enable_pinned
            collect_ram_grads = enable_pinned(model.ngram, staged=staged, profile=args.profile_ram,
                                              sparse_update=args.sparse_update,
                                              update_threads=args.update_threads, process_group=group)
        gc.collect()
        torch.cuda.empty_cache()
    if args.cuda_accum:
        from cuda_ngram_accum import enable_cuda_accum, collect_cuda_grads
        enable_cuda_accum(model.ngram)
    if args.host_accum:
        from pinned_ngram import enable_pinned
        collect_ram_grads = enable_pinned(model.ngram, staged=True, gpu_table=True,
                                          update_threads=args.update_threads, process_group=group)
    if staged:
        from pinned_ngram import attach_cpu_inputs
        if not distributed:  # under DDP the hook goes on the wrapper, below
            attach_cpu_inputs(model)
    model.train()
    report = model.param_report()
    print("MODEL", json.dumps(report), flush=True)
    print("CONFIG", json.dumps(vars(args)), flush=True)
    kernels = stack.backend_summary()
    if args.attn == "native":
        kernels["attn"] = "torch-native-varlen"
    print("KERNELS", json.dumps(kernels), flush=True)
    if args.check and args.pack:
        check_gdn_packing(model, batches[0])
    before = check_packing(model, batches[0]) if args.check and args.pack else None
    from optim800k import apply_optimizations
    apply_optimizations(model, args)
    optimize(model, args)
    if args.depth_attn != "off":
        assert args.attn == "native", "Depth-Attention is implemented in the native attention path"
        from depth_attention import enable_depth_attention
        log("DEPTH_ATTENTION", json.dumps(enable_depth_attention(model, args.depth_attn)))
    layer_drop = None
    if args.layer_drop > 0 or args.eval_exit_at:
        import layer_dropout
        layer_dropout.install(model, args.compile_mode)
        if args.layer_drop > 0:
            layer_drop = layer_dropout.LayerDropout(
                len(model.blocks), args.layer_drop, args.layer_drop_dist, args.layer_drop_schedule,
                args.layer_drop_horizon or count, args.seed)
    print("PLACEMENT", dict(table=str(model.ngram.table.device), row_state=str(model.ngram.row_v.device),
                            table_bytes=model.ngram.table.numel()*model.ngram.table.element_size(),
                            checkpointing=any(b.ckpt_mlp for b in model.blocks) or model.full_ckpt,
                            maximum_document_tokens=max(b["max_len"] for b in batches)), flush=True)
    if before is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after = float(model(batches[0]))
        assert math.isfinite(after) and abs(after-before)/max(1., abs(before)) < 0.01, (before, after)
        print("PACKED_MODEL_LOSS_PARITY", before, after, flush=True)
    assert model.param_report() == report
    train_model = model
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # The n-gram table and row state are host buffers kept in sync by the sparse step.
        DDP._set_params_and_buffers_to_ignore_for_model(model, ["ngram.table", "ngram.row_v"])
        train_model = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False,
                          gradient_as_bucket_view=True, bucket_cap_mb=64)
        if staged:  # before DDP copies the batch's CPU token tensors to the GPU
            from pinned_ngram import attach_cpu_inputs
            attach_cpu_inputs(train_model, model.ngram)
    if args.optimizer == "dion3":
        from dion_optim import build_dion3
        optimizer, groups = build_dion3(model, args, train_model.process_group if distributed else None)
        log("OPTIMIZER", json.dumps(dict(name="dion3", world=world, **groups)))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                     weight_decay=0.01, fused=True)
    losses, seconds, touched_rows = [], [], []
    sparse_seconds, dense_seconds, forward_backward_seconds = [], [], []
    begin_wall = None
    for i in range(count):
        if i == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin_wall = time.perf_counter()
            if args.ngram_device == "cpu" or args.host_accum:
                model.ngram._ram_profile = dict(lookup=0.0, collect=0.0, update=0.0)
                if hasattr(model.ngram, "_pin_detail"):
                    model.ngram._pin_detail = {}
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        optimizer.zero_grad(set_to_none=True)
        values = []
        step_valid = global_valid[i]
        assert step_valid > 0
        for micro in range(args.grad_accum):
            batch = batches[i * args.grad_accum + micro]
            # DDP all-reduces only on the last microbatch of the update.
            sync = not distributed or micro == args.grad_accum-1
            if layer_drop is not None:
                # Under DDP the synchronizing microbatch runs every block, so no
                # parameter is unused in the backward that triggers the reduction.
                model._layer_plan = None if distributed and sync else layer_drop.plan(i)
            with contextlib.nullcontext() if sync else train_model.no_sync():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = train_model(batch)
                # DDP averages over ranks, so scale by world: the result is the global token mean.
                weight = (batch["valid_tokens"]/step_valid*world if args.token_weighted
                          else 1/args.grad_accum)
                value = loss.detach() if args.defer_loss_sync else float(loss.detach())
                if not args.defer_loss_sync and not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite loss at step {i}, micro {micro}: {value}")
                (loss * weight).backward()
            if args.cuda_accum:
                collect_cuda_grads(model.ngram)
            if args.ngram_device == "cpu" or args.host_accum:
                collect_ram_grads(model.ngram)
            values.append(value * weight)
        if distributed:
            total = (torch.stack(values).sum() if args.defer_loss_sync
                     else torch.tensor(sum(values), device="cuda"))
            dist.all_reduce(total)
            value = float(total)/world
        else:
            value = float(torch.stack(values).sum()) if args.defer_loss_sync else sum(values)
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite accumulated loss at step {i}: {value}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=True)
        events[1].record()
        optimizer.step()
        events[2].record()
        touched = model.ngram.sparse_step()
        events[3].record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            seconds.append(events[0].elapsed_time(events[3])/1000)
            forward_backward_seconds.append(events[0].elapsed_time(events[1])/1000)
            dense_seconds.append(events[1].elapsed_time(events[2])/1000)
            sparse_seconds.append(events[2].elapsed_time(events[3])/1000)
            losses.append(value)
            touched_rows.append(touched)
        if i in (0, args.warmup-1, count-1):
            log("STEP", i+1, "loss", value, "rows", touched)
    wall = time.perf_counter() - begin_wall
    assert torch.isfinite(model.ngram.row_v).all(), "Nonfinite sparse optimizer state"
    if world == 1:
        assert global_valid == [sum(b["valid_tokens"] for b in batches[j*args.grad_accum:(j+1)*args.grad_accum])
                                for j in range(count)], "fixture-derived token counts disagree with batches"
    evaluation = {}
    if layer_drop is not None:
        model._layer_plan = None
        evaluation.update(layer_drop.report())
    if args.eval_data:
        # Token-weighted mean loss over a held-out file, after timing. Train mode is kept
        # (no dropout here) so compiled regions are reused; no_grad skips n-gram gradients.
        held_out = np.memmap(args.eval_data, mode="r", dtype=np.uint16)
        windows = min(args.eval_tokens, len(held_out)-1)//args.tokens
        eval_batches = [build_batch(held_out[w*args.tokens:][:args.tokens+1], eos, args.seq_len)
                        for w in range(windows)]

        def held_out_loss(plan=None):
            model._layer_plan = plan
            total, supervised = 0.0, 0
            with torch.no_grad():
                for batch in eval_batches:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        total += float(model(batch))*batch["valid_tokens"]
                    supervised += batch["valid_tokens"]
            model._layer_plan = None
            return total/supervised, supervised
        loss, supervised = held_out_loss()
        evaluation.update(eval_loss=loss, eval_tokens_used=windows*args.tokens,
                          eval_supervised_tokens=supervised)
        if args.eval_exit_at:
            from layer_dropout import exit_plan
            evaluation["eval_loss_early_exit"] = held_out_loss(
                exit_plan(len(model.blocks), args.eval_exit_at))[0]
        log("EVAL", json.dumps(evaluation))
    update_tokens = args.tokens*args.grad_accum*world
    result = dict(
        **vars(args), **report, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        world_size=world, effective_batch_tokens=update_tokens,
        data_token_count=len(raw), fixture_reused=update_tokens*count > len(raw)-1,
        min_supervised_tokens_per_update=min(global_valid),
        tokens_per_second=update_tokens*args.steps/wall,
        gpu_tokens_per_second=update_tokens*args.steps/sum(seconds), measured_wall_seconds=wall,
        median_step_ms=float(np.median(seconds)*1000),
        forward_backward_clip_ms=float(np.mean(forward_backward_seconds)*1000),
        dense_optimizer_ms=float(np.mean(dense_seconds)*1000),
        sparse_optimizer_ms=float(np.mean(sparse_seconds)*1000),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        first_loss=losses[0], final_loss=losses[-1], finite=True,
        mean_touched_rows=float(np.mean(touched_rows)), resident_data=True,
        trainable_table=True, dtype="autocast_bfloat16_fp32_dense_parameters",
        ngram_table_device=str(model.ngram.table.device),
        ngram_table_pinned=model.ngram.table.is_pinned(),
        ngram_row_state_device=str(model.ngram.row_v.device),
        maximum_document_tokens=max(b["max_len"] for b in batches),
        checkpointing_enabled=any(b.ckpt_mlp for b in model.blocks) or model.full_ckpt,
        host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        losses=losses,
        dynamo_unique_graphs=__import__("torch._dynamo.utils").\
            _dynamo.utils.counters["stats"]["unique_graphs"],
    )
    # Merged, not keyword-expanded: an evaluation key may share a name with a CLI flag.
    result.update(evaluation)
    if args.ngram_device == "cpu" or args.host_accum:
        result["ram_operation_wall_ms_per_step"] = {
            key: value/args.steps*1000 for key, value in model.ngram._ram_profile.items()}
        if hasattr(model.ngram, "_pin_detail"):
            result["pin_detail_ms_per_step"] = {
                key: value/args.steps*1000 for key, value in model.ngram._pin_detail.items()}
    if staged:
        # Nonzero means some lookups hashed via a synchronizing GPU-to-CPU copy.
        result["cpu_input_fallbacks"] = model.ngram._cpu_input_fallbacks
    if args.checksums:
        with torch.no_grad():
            result["dense_checksum"] = float(sum(p.detach().double().sum() for p in model.parameters()))
            result["table_checksum"] = float(torch.sum(model.ngram.table, dtype=torch.float64))
            result["row_state_checksum"] = float(torch.sum(model.ngram.row_v, dtype=torch.float64))
        if distributed:  # replicas must agree; report the spread across ranks
            sums = torch.tensor([result["dense_checksum"], result["table_checksum"]],
                                dtype=torch.float64, device="cuda")
            spread = sums.clone()
            dist.all_reduce(sums, op=dist.ReduceOp.MAX)
            dist.all_reduce(spread, op=dist.ReduceOp.MIN)
            result["checksum_rank_spread"] = (sums-spread).tolist()
    if rank == 0:
        with open(args.output, "a") as handle:
            handle.write(json.dumps(result) + "\n")
        Path("config100m.json").write_text(json.dumps(asdict(c), indent=2))
    log("RESULT", json.dumps(result))
    if distributed:
        dist.destroy_process_group()
    if args.profile_kernels:
        profile_kernels(model, batches, args, collect_ram_grads if staged or args.ngram_device == "cpu"
                        else None, collect_cuda_grads if args.cuda_accum else None)


def profile_kernels(model, batches, args, collect_ram, collect_cuda):
    """Diagnostic only: run after timing so profiler overhead never enters RESULT."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for batch in batches[:args.profile_kernels]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)
            loss.backward()
            if collect_cuda is not None:
                collect_cuda(model.ngram)
            if collect_ram is not None:
                collect_ram(model.ngram)
        torch.cuda.synchronize()
    from torch.autograd import DeviceType
    events = prof.key_averages()
    # Operator rows repeat their kernels' device time; count kernel rows only.
    total = sum(e.self_device_time_total for e in events if e.device_type == DeviceType.CUDA)
    table = events.table(sort_by="self_device_time_total", row_limit=40, max_name_column_width=90)
    path = Path(args.output).with_suffix(".kernels.txt")
    path.write_text(table)
    print(table, flush=True)
    print("KERNEL_PROFILE", json.dumps(dict(microbatches=args.profile_kernels, path=str(path),
                                            device_ms_per_microbatch=total/1000/args.profile_kernels)),
          flush=True)


if __name__ == "__main__":
    main()
