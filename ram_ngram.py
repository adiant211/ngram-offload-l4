"""CPU-resident n-gram table and row optimizer; only gathered rows visit CUDA.

Dense projections remain on the GPU. Sparse leaf gradients are copied to RAM
after each microbatch, so gradient accumulation does not retain CUDA leaf buffers.
"""
import math
import time
import types

import torch
import torch.nn.functional as F

from ngram import NgramMemory, ngram_hash


def enable_ram_ngram(memory):
    if memory._pending:
        raise RuntimeError("Move the table before the first training forward")
    memory.table = memory.table.cpu()
    memory.row_v = memory.row_v.cpu()
    memory._ram_pending = []
    memory._ram_grads = []
    memory._ram_profile = dict(lookup=0.0, collect=0.0, update=0.0)
    memory.lookup = types.MethodType(ram_lookup, memory)
    memory.sparse_step = types.MethodType(ram_sparse_step, memory)
    assert memory.table.device.type == memory.row_v.device.type == "cpu"


def ram_lookup(self, ids, doc_pos):
    started = time.perf_counter()
    cpu_ids, cpu_pos = ids.detach().cpu(), doc_pos.detach().cpu()
    indices, masks = [], []
    table_number = 0
    for order in self.cfg.orders:
        for head in range(self.cfg.heads_per_order):
            idx, valid = ngram_hash(cpu_ids, cpu_pos, order, head, self.rows)
            indices.append(idx + table_number*self.rows)
            masks.append(valid)
            table_number += 1
    idx, valid = torch.stack(indices, 1), torch.stack(masks, 1)
    # The full table stays in ordinary host RAM. Pin only this batch's gathered rows.
    rows = F.embedding(idx, self.table).pin_memory()
    emb = rows.to(ids.device, non_blocking=True).float()
    if self.training and torch.is_grad_enabled():
        emb.requires_grad_(True)
        self._ram_pending.append((idx.reshape(-1), valid.reshape(-1), emb))
    result = (emb*valid.to(ids.device, non_blocking=True).unsqueeze(-1)).flatten(1)
    self._ram_profile["lookup"] += time.perf_counter()-started
    return result


@torch.no_grad()
def collect_ram_grads(memory):
    started = time.perf_counter()
    for idx, valid, emb in memory._ram_pending:
        if emb.grad is None:
            raise RuntimeError("Missing n-gram leaf gradient")
        grad = emb.grad.detach().reshape(-1, memory.cfg.dim).cpu()
        memory._ram_grads.append((idx[valid], grad[valid]))
    memory._ram_pending.clear()
    memory._ram_profile["collect"] += time.perf_counter()-started


@torch.no_grad()
def ram_sparse_step(self, lr_mult=1.0):
    if self._ram_pending:
        collect_ram_grads(self)
    if not self._ram_grads:
        return 0
    started = time.perf_counter()
    idx = torch.cat([row[0] for row in self._ram_grads])
    grad = torch.cat([row[1] for row in self._ram_grads])
    self._ram_grads.clear()
    uniq, inverse = torch.unique(idx, return_inverse=True)
    g = torch.zeros(uniq.numel(), self.cfg.dim, dtype=torch.float32)
    g.index_add_(0, inverse, grad)
    cfg = self.cfg
    state = self.row_v[uniq].mul_(cfg.beta2).add_(g.square().mean(1), alpha=1-cfg.beta2)
    self.row_v[uniq] = state
    update = g/(state.sqrt()+cfg.eps).unsqueeze(1)
    self.table.index_add_(0, uniq, (update*(-cfg.lr*lr_mult)).to(self.table.dtype))
    self._ram_profile["update"] += time.perf_counter()-started
    return uniq.numel()


def check_ram_ngram():
    """One small lookup/backward/sparse-update equivalence check, with accumulation."""
    import copy
    from config import NgramConfig
    cfg = NgramConfig(dim=16, total_params=65536, table_dtype="bfloat16")
    torch.manual_seed(1234)
    gpu = NgramMemory(cfg, d_model=32, n_inject=1).cuda().train()
    ram = copy.deepcopy(gpu)
    enable_ram_ngram(ram)
    for _ in range(2):
        ids = torch.randint(0, 60800, (193,), device="cuda")
        pos = torch.arange(193, device="cuda") % 64
        upstream = torch.randn(193, cfg.num_tables*cfg.dim, device="cuda")
        a, b = gpu.lookup(ids, pos), ram.lookup(ids, pos)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        (a*upstream/2).sum().backward()
        (b*upstream/2).sum().backward()
        collect_ram_grads(ram)
    gpu_rows, ram_rows = gpu.sparse_step(), ram.sparse_step()
    assert gpu_rows == ram_rows
    cpu_reference = gpu.table.cpu().float()
    error = float((cpu_reference-ram.table.float()).norm()/cpu_reference.norm())
    assert error < 0.005, error
    torch.testing.assert_close(gpu.row_v.cpu(), ram.row_v, rtol=0.0001, atol=0.000001)
    assert not ram._ram_pending and not ram._ram_grads
    print("RAM_NGRAM_PARITY", dict(lookup_exact=True, sparse_table_relative_error=error,
                                   touched_rows=ram_rows, accumulated_microbatches=2), flush=True)
