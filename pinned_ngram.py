"""Pinned RAM n-gram table, transfer profiling, and bounded asynchronous CPU accumulation."""
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from ngram import ngram_hash
from ngram_canon import canonical

_detail_lock = threading.Lock()


def _add(memory, key, seconds):
    # The aggregation worker reports concurrently with the training thread.
    with _detail_lock:
        memory._pin_detail[key] = memory._pin_detail.get(key, 0.0) + seconds


def attach_cpu_inputs(model, memory=None):
    """Hand each forward's CPU ids/positions to the n-gram lookup.

    The handoff is consumed once per lookup. Unlike a cache keyed by CUDA
    data_ptr, it cannot return another batch's tokens after allocator reuse.
    Under DDP, pass the DDP wrapper as the model (and the n-gram module as
    memory): DDP copies every input tensor to the GPU before the inner
    module's hooks run, including these CPU copies.
    """
    memory = memory if memory is not None else model.ngram

    def hook(module, args):
        batch = args[0]
        if "ids_cpu" in batch:
            memory._cpu_next = (batch["ids_cpu"], batch["pos_cpu"])
    return model.register_forward_pre_hook(hook)


def _indices(memory, ids, pos):
    start = time.perf_counter()
    cached, memory._cpu_next = getattr(memory, "_cpu_next", None), None
    if cached is None or cached[0].numel() != ids.numel() or cached[0].device.type != "cpu":
        # Correct but synchronizing: waits for queued GPU work before hashing.
        memory._cpu_input_fallbacks = getattr(memory, "_cpu_input_fallbacks", 0) + 1
        cached = ids.detach().cpu(), pos.detach().cpu()
    cpu_ids, cpu_pos = cached
    cpu_ids = canonical(memory, cpu_ids)  # identity unless a canonical-token map is attached
    _add(memory, "input_d2h_wall", time.perf_counter()-start)
    start = time.perf_counter()
    indices, masks = [], []
    for order in memory.cfg.orders:
        for head in range(memory.cfg.heads_per_order):
            idx, valid = ngram_hash(cpu_ids, cpu_pos, order, head, memory.rows)
            indices.append(idx + len(indices)*memory.rows)
            masks.append(valid)
    result = torch.stack(indices, 1), torch.stack(masks, 1)
    _add(memory, "cpu_hash", time.perf_counter()-start)
    return result


def enable_pinned(memory, staged=False, profile=False, gpu_table=False,
                  sparse_update="rows", update_threads=None, process_group=None):
    assert memory.table.device.type == ("cuda" if gpu_table else "cpu")
    assert sparse_update in ("rows", "blocks")
    assert not (gpu_table and sparse_update == "blocks"), "block update needs the RAM table"
    memory._ram_pending = []
    memory._ram_grads = []
    memory._ram_profile = dict(lookup=0.0, collect=0.0, update=0.0)
    if not gpu_table:
        memory.table = memory.table.pin_memory()
    memory._pin_detail = {}
    memory._pin_profile = profile
    memory._sparse_update = sparse_update
    memory._update_threads = update_threads
    # ~8 MB FP32 scratch per block whatever the row width (4,096 rows at 512 wide): stays
    # cache- and malloc-friendly, and narrow tables don't pay Python overhead per tiny block.
    memory._update_block = max(4096, (1 << 21) // memory.cfg.dim)
    memory._process_group = process_group
    assert process_group is None or staged, "multi-rank n-gram sync is implemented for the staged path"
    if staged:
        # CPU inputs already exist in a real data loader; attach_cpu_inputs hands
        # them over. A CUDA round trip would wait for backward and kill overlap.
        memory._cpu_next = None
        memory._cpu_input_fallbacks = 0
        memory._host_grad = torch.zeros_like(memory.table, dtype=torch.float32, device="cpu")
        memory._host_touched = torch.zeros(memory.table.shape[0], dtype=torch.bool)
        memory._pin_stream = torch.cuda.Stream()
        memory._pin_worker = ThreadPoolExecutor(max_workers=1)
        memory._pin_slots = [dict(future=None), dict(future=None)]
        memory._pin_index = 0
        memory.lookup = types.MethodType(_staged_lookup, memory)
        memory.sparse_step = types.MethodType(_staged_step, memory)
        return collect_staged
    if profile:
        memory.lookup = types.MethodType(_profile_lookup, memory)
        memory.sparse_step = types.MethodType(_profile_step, memory)
        return collect_profile
    from ram_ngram import collect_ram_grads
    return collect_ram_grads


def _profile_lookup(memory, ids, pos):
    begin = time.perf_counter()
    idx, valid = _indices(memory, ids, pos)
    start = time.perf_counter()
    rows = F.embedding(idx, memory.table)
    _add(memory, "cpu_gather", time.perf_counter()-start)
    start = time.perf_counter()
    rows = rows.pin_memory()
    _add(memory, "pin_staging_copy", time.perf_counter()-start)
    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    gpu_rows = rows.to(ids.device, non_blocking=True)
    z.record()
    emb = gpu_rows.float()
    # This diagnostic deliberately fences so transfer and backward wait are distinct.
    z.synchronize()
    _add(memory, "rows_h2d_cuda", a.elapsed_time(z)/1000)
    if memory.training and torch.is_grad_enabled():
        emb.requires_grad_(True)
        memory._ram_pending.append((idx.reshape(-1), valid.reshape(-1), emb))
    result = (emb*valid.to(ids.device, non_blocking=True).unsqueeze(-1)).flatten(1)
    memory._ram_profile["lookup"] += time.perf_counter()-begin
    return result


@torch.no_grad()
def collect_profile(memory):
    begin = time.perf_counter()
    for idx, valid, emb in memory._ram_pending:
        start = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        _add(memory, "backward_queue_wait", time.perf_counter()-start)
        start = time.perf_counter()
        grad = emb.grad.detach().reshape(-1, memory.cfg.dim).cpu()
        _add(memory, "gradient_d2h_wall", time.perf_counter()-start)
        start = time.perf_counter()
        memory._ram_grads.append((idx[valid], grad[valid]))
        _add(memory, "cpu_filter", time.perf_counter()-start)
    memory._ram_pending.clear()
    memory._ram_profile["collect"] += time.perf_counter()-begin


@torch.no_grad()
def _profile_step(memory, lr_mult=1.0):
    begin = time.perf_counter()
    start = time.perf_counter()
    idx = torch.cat([p[0] for p in memory._ram_grads])
    grad = torch.cat([p[1] for p in memory._ram_grads])
    memory._ram_grads.clear()
    _add(memory, "cpu_concat", time.perf_counter()-start)
    start = time.perf_counter()
    uniq, inv = torch.unique(idx, return_inverse=True)
    _add(memory, "cpu_unique", time.perf_counter()-start)
    start = time.perf_counter()
    g = torch.zeros(uniq.numel(), memory.cfg.dim).index_add_(0, inv, grad)
    _add(memory, "cpu_reduce", time.perf_counter()-start)
    start = time.perf_counter()
    cfg = memory.cfg
    v = memory.row_v[uniq].mul_(cfg.beta2).add_(g.square().mean(1), alpha=1-cfg.beta2)
    memory.row_v[uniq] = v
    upd = g/(v.sqrt()+cfg.eps).unsqueeze(1)
    memory.table.index_add_(0, uniq, (upd*(-cfg.lr*lr_mult)).to(memory.table.dtype))
    _add(memory, "cpu_rmsprop", time.perf_counter()-start)
    memory._ram_profile["update"] += time.perf_counter()-begin
    return uniq.numel()


def _staged_lookup(memory, ids, pos):
    begin = time.perf_counter()
    idx, valid = _indices(memory, ids, pos)
    slot = memory._pin_slots[memory._pin_index % len(memory._pin_slots)]
    memory._pin_index += 1
    # The worker and DMA must finish before either pinned buffer is overwritten.
    if slot["future"] is not None:
        slot["future"].result()
    if "h2d_done" in slot:
        slot["h2d_done"].synchronize()
    shape = (idx.numel(), memory.cfg.dim)
    if "rows" not in slot or slot["rows"].shape != shape:
        slot["rows"] = torch.empty(shape, dtype=memory.table.dtype, pin_memory=True)
        slot["grad"] = torch.empty(shape, dtype=torch.float32, pin_memory=True)
        slot["indices"] = torch.empty_like(idx, pin_memory=True)
        slot["valid"] = torch.empty_like(valid, pin_memory=True)
    start = time.perf_counter()
    if memory.table.device.type == "cpu":
        torch.index_select(memory.table, 0, idx.reshape(-1), out=slot["rows"])
    else:
        slot["indices"].copy_(idx)
    # A pageable host-to-device copy waits for all queued GPU work before it starts,
    # which would drain the GPU queue once per microbatch; the mask goes through pinned memory.
    slot["valid"].copy_(valid)
    _add(memory, "cpu_gather", time.perf_counter()-start)
    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    if memory.table.device.type == "cpu":
        rows = slot["rows"].to(ids.device, non_blocking=True)
    else:
        rows = F.embedding(slot["indices"].to(ids.device, non_blocking=True), memory.table)
    valid_device = slot["valid"].to(ids.device, non_blocking=True)
    z.record()
    slot["h2d_done"] = z
    emb = rows.view(*idx.shape, memory.cfg.dim).float()
    if memory.training and torch.is_grad_enabled():
        emb.requires_grad_(True)
        memory._ram_pending.append((idx.reshape(-1), valid.reshape(-1), emb, slot, a, z))
    result = (emb*valid_device.unsqueeze(-1)).flatten(1)
    memory._ram_profile["lookup"] += time.perf_counter()-begin
    return result


@torch.no_grad()
def collect_staged(memory):
    begin = time.perf_counter()
    for idx, valid, emb, slot, h2d_a, h2d_z in memory._ram_pending:
        ready = torch.cuda.Event()
        ready.record()
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        grad = emb.grad.detach().reshape(-1, memory.cfg.dim)
        with torch.cuda.stream(memory._pin_stream):
            memory._pin_stream.wait_event(ready)
            a.record()
            slot["grad"].copy_(grad, non_blocking=True)
            z.record()
            grad.record_stream(memory._pin_stream)
        slot["future"] = memory._pin_worker.submit(_reduce_slot, memory, slot["grad"], idx, valid,
                                                    a, z, h2d_a, h2d_z)
        emb.grad = None
    memory._ram_pending.clear()
    memory._ram_profile["collect"] += time.perf_counter()-begin


def _reduce_slot(memory, gradient, idx, valid, a, z, h2d_a, h2d_z):
    start = time.perf_counter()
    z.synchronize()
    _add(memory, "worker_dma_wait", time.perf_counter()-start)
    _add(memory, "gradient_d2h_cuda", a.elapsed_time(z)/1000)
    _add(memory, "rows_h2d_cuda" if memory.table.device.type == "cpu" else "metadata_h2d_and_gpu_gather",
         h2d_a.elapsed_time(h2d_z)/1000)
    start = time.perf_counter()
    # Invalid lookup entries have exactly zero gradient because lookup masks them.
    memory._host_grad.index_add_(0, idx, gradient)
    memory._host_touched[idx[valid]] = True
    _add(memory, "cpu_incremental_reduce", time.perf_counter()-start)


@torch.no_grad()
def _staged_step(memory, lr_mult=1.0):
    begin = time.perf_counter()
    start = time.perf_counter()
    for slot in memory._pin_slots:
        if slot["future"] is not None:
            slot["future"].result()
    _add(memory, "worker_final_join", time.perf_counter()-start)
    if memory._process_group is not None:
        start = time.perf_counter()
        _allreduce_accumulator(memory)
        _add(memory, "cross_rank_reduce", time.perf_counter()-start)
    threads = torch.get_num_threads()
    if memory._update_threads:
        # The GPU idles during this phase, so the whole host can work on it.
        torch.set_num_threads(memory._update_threads)
    try:
        if memory._sparse_update == "blocks":
            touched = _block_update(memory, lr_mult)
        else:
            touched = _row_update(memory, lr_mult)
    finally:
        torch.set_num_threads(threads)
    memory._ram_profile["update"] += time.perf_counter()-begin
    return touched


@torch.no_grad()
def _allreduce_accumulator(memory, block=1 << 16):
    """Average row gradients and OR touched masks across ranks, matching DDP's averaging.

    Every rank then applies the identical sparse update, so the replicated tables stay
    bit-identical without broadcasting them. NCCL needs device tensors, so the 2 GB FP32
    accumulator is staged through the GPU in 128 MB blocks; gloo reduces host memory directly.
    """
    import torch.distributed as dist
    group = memory._process_group
    world = dist.get_world_size(group)
    if world == 1:
        return
    touched = memory._host_touched.view(torch.uint8)
    grad = memory._host_grad
    if dist.get_backend(group) == "gloo":
        dist.all_reduce(touched, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(grad, group=group)
        grad.div_(world)
        return
    device = torch.device("cuda", torch.cuda.current_device())
    mask = touched.to(device)
    dist.all_reduce(mask, op=dist.ReduceOp.MAX, group=group)
    touched.copy_(mask)
    staging = torch.empty(min(block, grad.shape[0]), grad.shape[1], device=device)
    for first in range(0, grad.shape[0], block):
        rows = grad[first:first+block]
        part = staging[:rows.shape[0]]
        part.copy_(rows)
        dist.all_reduce(part, op=dist.ReduceOp.AVG, group=group)
        rows.copy_(part)


def _splitmix(*values):
    z = 0x9E3779B97F4A7C15
    for value in values:
        z = (z ^ (int(value) & 0xFFFFFFFFFFFFFFFF)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
    return z


def _sr_noise(memory, count, block_index):
    """Uniform integers in [0, 2^16), one per element, from a fixed seeded buffer read at an offset
    derived from (seed, update, block): no free-running generator, so a resumed run makes the
    same rounding draws as the uninterrupted one."""
    size = max(1 << 22, 2*count)
    buffer = getattr(memory, "_sr_buffer", None)
    if buffer is None or buffer.numel() < size:
        generator = torch.Generator().manual_seed(int(getattr(memory, "_sr_seed", 0)) ^ 0x5EED)
        buffer = memory._sr_buffer = torch.randint(0, 1 << 16, (size,), dtype=torch.int32, generator=generator)
    offset = _splitmix(getattr(memory, "_sr_seed", 0), getattr(memory, "_sr_step", 0), block_index)         % (buffer.numel()-count)
    return buffer[offset:offset+count]


def stochastic_round_(target, value, noise):
    """target (BF16) <- value (FP32) rounded stochastically: add uniform noise to the 16 bits BF16
    drops, then truncate. Unbiased; values already representable in BF16 are unchanged. Overwrites
    value (contiguous FP32)."""
    bits = value.view(torch.int32)
    bits.add_(noise.view(bits.shape))
    bits.bitwise_and_(-65536)
    target.copy_(value)  # exact: the low 16 bits are zero


def _block_update(memory, lr_mult):
    """Same arithmetic as _row_update, over contiguous row blocks with a touched mask.

    Most rows are touched by a 786k-token update, so gathering and scattering them
    costs more than streaming every row once. The accumulator is cleared per block
    while it is still in cache, which replaces a separate pass over 2 GB.
    """
    start = time.perf_counter()
    cfg, table, touched, block = memory.cfg, memory.table, 0, memory._update_block
    scratch = getattr(memory, "_update_scratch", None)
    if scratch is None or scratch["work"].shape[0] != block:
        # Reused every step: fresh multi-MB temporaries are mmapped and faulted in each time.
        scratch = memory._update_scratch = dict(
            work=torch.empty(block, cfg.dim), low=torch.empty(block, cfg.dim, dtype=table.dtype),
            mean=torch.empty(block), fresh=torch.empty(block), denom=torch.empty(block),
            high=torch.empty(block, cfg.dim))
    stochastic = getattr(cfg, "stochastic_round", False) and table.dtype == torch.bfloat16
    for first in range(0, table.shape[0], block):
        mask = memory._host_touched[first:first+block]
        count = int(mask.sum())
        g = memory._host_grad[first:first+block]
        if count:
            touched += count
            n = g.shape[0]
            work, low = scratch["work"][:n], scratch["low"][:n]
            mean, fresh, denom = scratch["mean"][:n], scratch["fresh"][:n], scratch["denom"][:n]
            v = memory.row_v[first:first+block]
            torch.mean(torch.square(g, out=work), 1, out=mean)
            torch.mul(v, cfg.beta2, out=fresh).add_(mean, alpha=1-cfg.beta2)
            torch.sqrt(fresh, out=denom).add_(cfg.eps)
            torch.div(g, denom.unsqueeze(1), out=work).mul_(-cfg.lr*lr_mult)
            # Untouched rows keep their state; masking also isolates them from
            # any nonfinite value that reached a masked-out accumulator entry.
            v.copy_(torch.where(mask, fresh, v))
            work.masked_fill_(~mask.unsqueeze(1), 0.0)
            if stochastic:
                high = scratch["high"][:n]
                high.copy_(table[first:first+block]).add_(work)
                stochastic_round_(table[first:first+block], high, _sr_noise(memory, high.numel(), first // block))
            else:
                table[first:first+block].add_(low.copy_(work))
            g.zero_()
            mask.zero_()
        elif g.any():
            g.zero_()
    _add(memory, "cpu_rmsprop_and_zero_blocks", time.perf_counter()-start)
    return touched


def _row_update(memory, lr_mult):
    start = time.perf_counter()
    uniq = memory._host_touched.nonzero().flatten()
    cfg = memory.cfg
    for chunk, rows in enumerate(uniq.split(16384)):
        g = memory._host_grad[rows]
        if memory.table.device.type == "cuda":
            g = g.pin_memory().to(memory.table.device, non_blocking=True)
            rows = rows.to(memory.table.device, non_blocking=True)
        v = memory.row_v[rows].mul_(cfg.beta2).add_(g.square().mean(1), alpha=1-cfg.beta2)
        memory.row_v[rows] = v
        upd = g/(v.sqrt()+cfg.eps).unsqueeze(1)
        if getattr(cfg, "stochastic_round", False) and memory.table.dtype == torch.bfloat16:
            value = (memory.table[rows].float() + upd*(-cfg.lr*lr_mult)).contiguous()
            noise = _sr_noise(memory, value.numel(), chunk).to(value.device)
            rounded = torch.empty_like(value, dtype=torch.bfloat16)
            stochastic_round_(rounded, value, noise)
            memory.table[rows] = rounded
        else:
            memory.table.index_add_(0, rows, (upd*(-cfg.lr*lr_mult)).to(memory.table.dtype))
    _add(memory, "cpu_rmsprop", time.perf_counter()-start)
    start = time.perf_counter()
    memory._host_grad.zero_()
    memory._host_touched.zero_()
    _add(memory, "cpu_zero_accumulator", time.perf_counter()-start)
    return uniq.numel()


def check_pinned(gpu_table=False, canon=False, stochastic=False):
    """Pinned/staged paths against the GPU reference NgramMemory.

    canon: a synthetic canonical-token map on both sides (lookups must stay identical).
    stochastic: candidates round their BF16 table writes stochastically, the reference to nearest,
    so tables may differ by one BF16 ulp per update (inside the tolerance)."""
    import copy
    from config import NgramConfig
    from ngram import NgramMemory
    from ram_ngram import enable_ram_ngram
    import ngram_canon
    torch.set_num_threads(4)
    torch.manual_seed(792)
    reference = NgramMemory(NgramConfig(dim=16, total_params=65536, table_dtype="bfloat16"), 32, 1).cuda().train()
    if canon:
        folded = torch.arange(60800)
        folded[1::3] -= 1  # ids 3k+1 hash as 3k; idempotent
        ngram_canon.attach(reference, folded)
    modes = ("rows",) if gpu_table else ("rows", "blocks")
    candidates, collects = {}, {}
    for mode in modes:
        candidates[mode] = copy.deepcopy(reference)
        if stochastic:
            candidates[mode].cfg = copy.deepcopy(reference.cfg)
            candidates[mode].cfg.stochastic_round = True
            candidates[mode]._sr_seed = 11
        if not gpu_table:
            enable_ram_ngram(candidates[mode])
        collects[mode] = enable_pinned(candidates[mode], staged=True, gpu_table=gpu_table,
                                       sparse_update=mode, update_threads=2)
        # Several blocks, including a partial last block, on the 4,096-row test table.
        candidates[mode]._update_block = 1000
    # Stochastic rounding may land one BF16 ulp away from round-to-nearest on each update,
    # which also shows in the next update's looked-up rows.
    tol = dict(rtol=0.008, atol=0.001) if stochastic else dict(rtol=0.004, atol=0.0005)
    for step in range(2):
        for candidate in candidates.values():
            candidate._sr_step = step
        for micro in range(4):
            ids = torch.randint(0, 60800, (193,), device="cuda")
            pos = torch.arange(193, device="cuda") % 37
            a = reference.lookup(ids, pos)
            weight = torch.randn_like(a)
            (a*weight/4).sum().backward()
            for mode, candidate in candidates.items():
                if micro != 3:  # The last microbatch exercises the synchronizing fallback.
                    candidate._cpu_next = (ids.cpu(), pos.cpu())
                b = candidate.lookup(ids, pos)
                torch.testing.assert_close(a, b, **tol)
                (b*weight/4).sum().backward()
                collects[mode](candidate)
        touched = reference.sparse_step()
        for candidate in candidates.values():
            assert candidate.sparse_step() == touched
            torch.testing.assert_close(reference.table.cpu(), candidate.table.cpu(), **tol)
            torch.testing.assert_close(reference.row_v.cpu(), candidate.row_v.cpu(), rtol=1e-4, atol=1e-6)
            assert not candidate._host_touched.any() and not candidate._host_grad.any()
            assert candidate._cpu_next is None
        if len(candidates) == 2 and not stochastic:
            # Same formula in a different layout. Vectorized and scalar tails may round
            # add_(alpha=...) differently, so row_v may move by one FP32 ulp.
            assert torch.equal(candidates["rows"].table, candidates["blocks"].table)
            torch.testing.assert_close(candidates["rows"].row_v, candidates["blocks"].row_v,
                                       rtol=2.5e-7, atol=0)
    for candidate in candidates.values():
        assert candidate._cpu_input_fallbacks == 2
        candidate._pin_worker.shutdown()
    print("PINNED_PARITY: two updates, buffer reuse and CPU-input handoff passed; GPU table:",
          gpu_table, "updates:", modes, "canonical ids:", canon, "stochastic rounding:", stochastic,
          "block table bit-identical:", len(modes) == 2 and not stochastic, flush=True)


def check_stochastic_round():
    """Unbiased, exact for representable values, deterministic for (seed, step, block)."""
    from types import SimpleNamespace
    memory = SimpleNamespace(_sr_seed=3, _sr_step=0)
    count = 1 << 16
    # 1 + 1e-3: BF16 spacing at 1.0 is 2^-7, so round-to-nearest would always give exactly 1.0.
    target = torch.full((count,), 1.0 + 1e-3)
    draws = []
    for step in range(64):
        memory._sr_step = step
        out = torch.empty(count, dtype=torch.bfloat16)
        stochastic_round_(out, target.clone(), _sr_noise(memory, count, 0))
        draws.append(out.float())
    mean = torch.stack(draws).mean()
    assert abs(float(mean) - (1.0 + 1e-3)) < 2e-5, float(mean)
    assert set(torch.stack(draws).unique().tolist()) == {1.0, 1.0 + 2**-7}
    exact = torch.randn(count).bfloat16()
    out = torch.empty_like(exact)
    stochastic_round_(out, exact.float(), _sr_noise(memory, count, 5))
    assert torch.equal(out, exact), "representable values must not move"
    negative = torch.full((count,), -(1.0 + 1e-3))
    out = torch.empty(count, dtype=torch.bfloat16)
    stochastic_round_(out, negative, _sr_noise(memory, count, 9))
    assert abs(float(out.float().mean()) + 1.0 + 1e-3) < 2e-4, float(out.float().mean())
    memory._sr_step = 7
    a, b = _sr_noise(memory, count, 2).clone(), _sr_noise(memory, count, 2).clone()
    c = _sr_noise(memory, count, 3)
    assert torch.equal(a, b) and not torch.equal(a, c)
    print("STOCHASTIC_ROUND: unbiased (mean", round(float(mean), 6), "for 1.001), representable values "
          "unchanged, negative values unbiased, noise deterministic per (seed, step, block)", flush=True)


def _distributed_worker(rank, world, port, mode):
    import copy
    import torch.distributed as dist
    from config import NgramConfig
    from ngram import NgramMemory
    from ram_ngram import enable_ram_ngram
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.set_num_threads(2)
    torch.manual_seed(792)
    reference = NgramMemory(NgramConfig(dim=16, total_params=65536, table_dtype="bfloat16"), 32, 1).cuda().train()
    candidate = copy.deepcopy(reference)
    enable_ram_ngram(candidate)
    collect = enable_pinned(candidate, staged=True, sparse_update=mode, update_threads=2,
                            process_group=dist.group.WORLD)
    candidate._update_block = 1000
    data = torch.Generator().manual_seed(5)
    micro_per_rank = 2
    for _ in range(2):
        for micro in range(world*micro_per_rank):
            ids = torch.randint(0, 60800, (193,), generator=data)
            pos = torch.arange(193) % 37
            upstream = torch.randn(193, reference.cfg.num_tables*reference.cfg.dim, generator=data).cuda()
            scale = 1/(world*micro_per_rank)
            (reference.lookup(ids.cuda(), pos.cuda())*upstream*scale).sum().backward()
            if micro // micro_per_rank == rank:
                candidate._cpu_next = (ids, pos)
                # x world: the cross-rank step averages, as DDP does for dense gradients.
                (candidate.lookup(ids.cuda(), pos.cuda())*upstream*scale*world).sum().backward()
                collect(candidate)
        # Every rank's reference saw all microbatches; keep its step single-process
        # (with a process group initialized, ngram.sparse_step would all-gather rows).
        import ngram as ngram_module
        initialized = ngram_module.dist.is_initialized
        ngram_module.dist.is_initialized = lambda: False
        try:
            touched = reference.sparse_step()
        finally:
            ngram_module.dist.is_initialized = initialized
        assert candidate.sparse_step() == touched
        torch.testing.assert_close(reference.table.cpu(), candidate.table, rtol=0.004, atol=0.0005)
        torch.testing.assert_close(reference.row_v.cpu(), candidate.row_v, rtol=1e-4, atol=1e-6)
    sums = torch.tensor([torch.sum(candidate.table, dtype=torch.float64),
                         torch.sum(candidate.row_v, dtype=torch.float64)])
    gathered = [torch.empty_like(sums) for _ in range(world)]
    dist.all_gather(gathered, sums)
    assert all(torch.equal(gathered[0], g) for g in gathered), gathered
    candidate._pin_worker.shutdown()
    dist.destroy_process_group()
    if rank == 0:
        print("PINNED_DISTRIBUTED_PARITY: world", world, "update:", mode,
              "matches one process; replicas bit-identical", flush=True)


def check_pinned_distributed(world=2, mode="blocks"):
    """Ranks split the microbatches; the result must match one process seeing all of them."""
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_distributed_worker, args=(world, port, mode), nprocs=world, join=True)


if __name__ == "__main__":
    check_stochastic_round()
    check_pinned()
    check_pinned(gpu_table=True)
    check_pinned(canon=True)
    check_pinned(stochastic=True)
    check_pinned(gpu_table=True, stochastic=True)
    for mode in ("rows", "blocks"):
        check_pinned_distributed(mode=mode)
