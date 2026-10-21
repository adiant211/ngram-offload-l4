"""Remove FLA 0.5.2's per-microbatch CPU-GPU synchronizations (trainer --fla-nosync).

torch.cuda sync debugging found five per microbatch, all in FLA:

1. prepare_chunk_indices (varlen chunk table for the short convolutions and the chunked
   delta rule). Without a host copy of cu_seqlens it reads counts.sum() from the GPU;
   with one, it still copies the result from pageable memory, which waits for the stream.
   The replacement builds the table on the host from the batch's CPU cu_seqlens (handed
   over with set_cu_seqlens) and copies it through pinned memory without blocking.
2. fused_linear_cross_entropy_backward checks `do != 1` on the host to skip a multiply.
   The replacement always multiplies: identical results (x * 1.0 is exact), no host read.

The GPU otherwise drains its queue at each of these points; the patches change no values.
"""
import inspect
import sys
import textwrap

import torch

_handoff = None  # (cu_seqlens on the GPU, the same values on the host)
_cache = {}
stats = dict(host_built=0, fallback=0)
_CE_CHECK = "if torch.ne(do, torch.tensor(1.0, device=do.device)):"


def set_cu_seqlens(device_cu, host_cu):
    """Called per microbatch with the batch's device cu_seqlens and its host copy."""
    global _handoff
    _handoff = (device_cu, host_cu)


def install():
    import fla.ops.utils.index as index
    import fla.modules.fused_linear_cross_entropy as flce
    original = getattr(index, "_original_prepare_chunk_indices", index.prepare_chunk_indices)
    index._original_prepare_chunk_indices = original

    def prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
        host = cu_seqlens_cpu
        if host is None and _handoff is not None and cu_seqlens is _handoff[0]:
            host = _handoff[1]
        if host is None or host.device.type != "cpu":
            stats["fallback"] += 1
            return original(cu_seqlens, chunk_size, cu_seqlens_cpu)
        key = (chunk_size, id(cu_seqlens), id(host))
        hit = _cache.get(key)
        if hit is not None and hit[0] is cu_seqlens and hit[1] is host:
            return hit[2]
        stats["host_built"] += 1
        counts = (index.prepare_lens(host) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
        seg_id, intra = index._segmented_arange(counts)
        table = torch.stack([seg_id, intra], 1).to(cu_seqlens.dtype).pin_memory()
        result = table.to(cu_seqlens.device, non_blocking=True)
        if len(_cache) > 16:
            _cache.clear()
        _cache[key] = (cu_seqlens, host, result)  # references keep the ids valid
        return result

    replaced = 0
    for name, module in list(sys.modules.items()):
        if name.startswith("fla") and module is not None and \
                getattr(module, "prepare_chunk_indices", None) is original:
            module.prepare_chunk_indices = prepare_chunk_indices
            replaced += 1

    backward = getattr(flce, "_original_backward", flce.fused_linear_cross_entropy_backward)
    flce._original_backward = backward
    source = textwrap.dedent(inspect.getsource(backward))
    assert source.count(_CE_CHECK) == 1, "FLA source changed: cross-entropy backward check not found"
    source = source.replace(_CE_CHECK, "if True:  # always scale: exact for do == 1, no host read")
    source = "\n".join(line for line in source.splitlines() if not line.startswith("@"))
    exec(compile(source, flce.__file__, "exec"), flce.__dict__)
    # FusedLinearCrossEntropyFunction.backward looks the function up as a module global.
    return dict(chunk_index_modules=replaced)


def check():
    """Patched vs original: identical chunk tables and bit-identical cross-entropy gradients."""
    import fla.ops.utils.index as index
    import fla.modules.fused_linear_cross_entropy as flce
    install()
    torch.manual_seed(5)
    lengths = torch.tensor([1, 63, 64, 65, 700, 1024, 129, 3])
    host = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    device = host.to(torch.int32).cuda()
    for chunk in (16, 64):
        reference = index._original_prepare_chunk_indices(device, chunk)
        set_cu_seqlens(device, host)
        patched = index.prepare_chunk_indices(device, chunk)
        assert torch.equal(reference, patched) and reference.dtype == patched.dtype, chunk
    patched_backward = flce.fused_linear_cross_entropy_backward
    n, h, v = 2048, 256, 60800
    x = torch.randn(n, h, device="cuda")
    weight = (torch.randn(v, h, device="cuda") / h**0.5)
    target = torch.randint(0, v, (n,), device="cuda")
    for scale in (1.0, 0.37):
        grads = []
        for function in (flce._original_backward, patched_backward):
            flce.fused_linear_cross_entropy_backward = function
            xi, wi = x.clone().requires_grad_(True), weight.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flce.FusedLinearCrossEntropyLoss()(xi, target, wi)
            (loss * scale).backward()
            grads.append((xi.grad, wi.grad))
        assert all(torch.equal(a, b) for a, b in zip(*grads)), scale
    flce.fused_linear_cross_entropy_backward = patched_backward
    print("FLA_NOSYNC_PARITY chunk tables identical; cross-entropy gradients bit-identical", flush=True)


if __name__ == "__main__":
    check()
