"""Short real-data throughput experiments for a100stack on a single GPU.

Run from the extracted a100stack directory, after installing FLA 0.5.2.
Uses the supplied tokenizer's bounded FineWeb sample, not random tokens.
Timing includes forward/backward, clipping, dense optimizer and sparse updates;
excludes sample preparation, warmup, compilation and evaluation. Batches are GPU resident.
"""
import argparse
import gc
import inspect
import json
import math
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
    for block in model.blocks:
        block.ckpt_mlp = args.ckpt == "mlp"
        mixer = block.mixer
        if args.pack:
            if block.is_attn:
                pack_linear(mixer, ("q_proj", "k_proj", "v_proj", "gate_proj"))
            else:
                pack_linear(mixer, ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj"))
                mixer.forward = types.MethodType(packed_gdn, mixer)
            pack_linear(block.mlp, ("w_gate", "w_up"))
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
    parser.add_argument("--gdn-head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="results100m.jsonl")
    parser.add_argument("--data", default="fineweb-bench.bin")
    args = parser.parse_args()
    assert not args.pack or args.attn == "native"
    assert args.steps > 0 and args.warmup > 0 and args.grad_accum > 0
    assert not (args.host_accum and args.cuda_accum)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
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
    batches = [build_batch(raw[(i*args.tokens) % (len(raw)-args.tokens):][:args.tokens+1], eos, args.seq_len)
               for i in range(count * args.grad_accum)]
    torch.manual_seed(1234)
    with torch.device("cuda"):
        model = stack.HybridLM(c)
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
    if args.ngram_device == "cpu":
        from ram_ngram import enable_ram_ngram, collect_ram_grads
        enable_ram_ngram(model.ngram)
        if args.ram_mode != "original":
            from pinned_ngram import enable_pinned
            collect_ram_grads = enable_pinned(model.ngram, staged=staged, profile=args.profile_ram,
                                              sparse_update=args.sparse_update,
                                              update_threads=args.update_threads)
        gc.collect()
        torch.cuda.empty_cache()
    if args.cuda_accum:
        from cuda_ngram_accum import enable_cuda_accum, collect_cuda_grads
        enable_cuda_accum(model.ngram)
    if args.host_accum:
        from pinned_ngram import enable_pinned
        collect_ram_grads = enable_pinned(model.ngram, staged=True, gpu_table=True,
                                          update_threads=args.update_threads)
    if staged:
        from pinned_ngram import attach_cpu_inputs
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
        step_batches = batches[i*args.grad_accum:(i+1)*args.grad_accum]
        step_valid = sum(b["valid_tokens"] for b in step_batches)
        assert step_valid > 0
        for micro in range(args.grad_accum):
            batch = batches[i * args.grad_accum + micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)
            weight = batch["valid_tokens"]/step_valid if args.token_weighted else 1/args.grad_accum
            value = loss.detach() if args.defer_loss_sync else float(loss.detach())
            if not args.defer_loss_sync and not math.isfinite(value):
                raise RuntimeError(f"Nonfinite loss at step {i}, micro {micro}: {value}")
            (loss * weight).backward()
            if args.cuda_accum:
                collect_cuda_grads(model.ngram)
            if args.ngram_device == "cpu" or args.host_accum:
                collect_ram_grads(model.ngram)
            values.append(value * weight)
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
            print("STEP", i+1, "loss", value, "rows", touched, flush=True)
    wall = time.perf_counter() - begin_wall
    assert torch.isfinite(model.ngram.row_v).all(), "Nonfinite sparse optimizer state"
    result = dict(
        **vars(args), **report, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        effective_batch_tokens=args.tokens*args.grad_accum,
        data_token_count=len(raw), fixture_reused=args.tokens*args.grad_accum*count > len(raw)-1,
        min_supervised_tokens_per_update=min(sum(b["valid_tokens"] for b in
            batches[j*args.grad_accum:(j+1)*args.grad_accum]) for j in range(count)),
        tokens_per_second=args.tokens*args.grad_accum*args.steps/wall,
        gpu_tokens_per_second=args.tokens*args.grad_accum*args.steps/sum(seconds), measured_wall_seconds=wall,
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
    )
    if args.ngram_device == "cpu" or args.host_accum:
        result["ram_operation_wall_ms_per_step"] = {
            key: value/args.steps*1000 for key, value in model.ngram._ram_profile.items()}
        if hasattr(model.ngram, "_pin_detail"):
            result["pin_detail_ms_per_step"] = {
                key: value/args.steps*1000 for key, value in model.ngram._pin_detail.items()}
    if staged:
        # Nonzero means some lookups hashed via a synchronizing GPU-to-CPU copy.
        result["cpu_input_fallbacks"] = model.ngram._cpu_input_fallbacks
    with open(args.output, "a") as handle:
        handle.write(json.dumps(result) + "\n")
    Path("config100m.json").write_text(json.dumps(asdict(c), indent=2))
    print("RESULT", json.dumps(result), flush=True)
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
