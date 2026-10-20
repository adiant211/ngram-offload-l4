"""Single-GPU n-gram gradient accumulation with memory independent of accumulation count."""
import types
import torch
import torch.distributed as dist


def enable_cuda_accum(memory):
    if memory.table.device.type != "cuda":
        raise ValueError("CUDA accumulation requires a CUDA n-gram table")
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("This benchmark helper supports one GPU")
    memory._cuda_grad = torch.zeros_like(memory.table, dtype=torch.float32)
    memory._cuda_touched = torch.zeros(memory.table.shape[0], device=memory.table.device,
                                       dtype=torch.bool)
    memory.sparse_step = types.MethodType(_step, memory)


@torch.no_grad()
def collect_cuda_grads(memory):
    for idx, valid, emb in memory._pending:
        if emb.grad is None:
            continue
        rows = idx[valid]
        memory._cuda_grad.index_add_(0, rows, emb.grad.reshape(-1, memory.cfg.dim)[valid])
        memory._cuda_touched.index_fill_(0, rows, True)
        emb.grad = None
    memory._pending.clear()


@torch.no_grad()
def _step(memory, lr_mult=1.0):
    collect_cuda_grads(memory)
    touched = memory._cuda_touched.nonzero().flatten()
    cfg = memory.cfg
    # Bound update temporaries as well as gradients retained during accumulation.
    for rows in touched.split(16384):
        g = memory._cuda_grad[rows]
        v = memory.row_v[rows].mul_(cfg.beta2).add_(g.square().mean(1), alpha=1-cfg.beta2)
        memory.row_v[rows] = v
        upd = g / (v.sqrt() + cfg.eps).unsqueeze(1)
        memory.table.index_add_(0, rows, (upd * (-cfg.lr * lr_mult)).to(memory.table.dtype))
    memory._cuda_grad.zero_()
    memory._cuda_touched.zero_()
    return touched.numel()


def check_cuda_accum():
    """Compare two optimizer updates, each with two accumulated lookup gradients."""
    from config import NgramConfig
    from ngram import NgramMemory
    torch.manual_seed(321)
    cfg = NgramConfig(dim=16, total_params=65536, table_dtype="bfloat16")
    reference = NgramMemory(cfg, 32, 1).cuda()
    candidate = NgramMemory(cfg, 32, 1).cuda()
    candidate.load_state_dict(reference.state_dict())
    enable_cuda_accum(candidate)
    for _ in range(2):
        for _ in range(2):
            ids = torch.randint(0, 61, (193,), device="cuda")
            pos = torch.arange(193, device="cuda") % 37
            a, b = reference.lookup(ids, pos), candidate.lookup(ids, pos)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            weight = torch.randn_like(a)
            (a * weight / 2).sum().backward()
            (b * weight / 2).sum().backward()
            collect_cuda_grads(candidate)
            assert not candidate._pending
        assert reference.sparse_step() == candidate.sparse_step()
        torch.testing.assert_close(reference.table, candidate.table, rtol=2e-3, atol=5e-4)
        torch.testing.assert_close(reference.row_v, candidate.row_v, rtol=1e-4, atol=1e-6)
        assert not candidate._cuda_touched.any()
        assert not candidate._cuda_grad.any()
    print("CUDA_ACCUM_PARITY: two updates passed", flush=True)


if __name__ == "__main__":
    check_cuda_accum()
