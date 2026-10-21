"""Local smoke test of dion_optim: grouping, ConcatProjection equivalence, two Dion3 steps.

Usage: python tools/check_dion_local.py <dir with a100stack source + dion_optim.py> <dion repo>
Uses the tiny preset and the reference (non-FLA) kernels, so it runs without Triton.
"""
import copy
import sys
import types
from types import SimpleNamespace

sys.path[:0] = [sys.argv[1], sys.argv[2]]
sys.modules.setdefault("resource", types.ModuleType("resource"))
import torch
import model as stack
from config import preset
from dion_optim import ConcatProjection, build_dion3, split_params

torch.manual_seed(3)
c, _ = preset("tiny")
c.ngram.enabled = False
net = stack.HybridLM(c).cuda().train()

# ConcatProjection: one GEMM, same output and per-matrix gradients as the separate layers.
mlp = net.blocks[0].mlp
x = torch.randn(64, c.d_model, device="cuda", requires_grad=True)
fused = ConcatProjection([mlp.w_gate, mlp.w_up])(x)
separate = torch.cat([mlp.w_gate(x), mlp.w_up(x)], -1)
torch.testing.assert_close(fused, separate)
upstream = torch.randn_like(fused)
g1 = torch.autograd.grad((fused*upstream).sum(), [mlp.w_gate.weight, mlp.w_up.weight])
g2 = torch.autograd.grad((separate*upstream).sum(), [mlp.w_gate.weight, mlp.w_up.weight])
for a, b in zip(g1, g2):
    torch.testing.assert_close(a, b)

matrix, embed, head, other = split_params(net)
names = {id(p): n for n, p in net.named_parameters()}
assert all(min(p.shape) >= 16 and p.dim() == 2 for p in matrix)
assert not head, "tied lm_head must be grouped with the embedding"
assert any(names[id(p)].endswith("a_proj.weight") for p in other), "small a/b projections go to AdamW"

args = SimpleNamespace(scalar_lr=3e-3, dion_lr=0.02, dion_fraction=0.5, dion_mu=0.95,
                       dion_weight_decay=0.01, dion_triton=False)
optimizer, report = build_dion3(net, args)
n = 96
cu = torch.tensor([0, 40, n], dtype=torch.int32)
batch = dict(ids=torch.randint(0, 512, (n,), device="cuda"), targets=torch.randint(0, 512, (n,), device="cuda"),
             pos=torch.cat([torch.arange(40), torch.arange(56)]).cuda(), cu=cu.cuda(), cu_cpu=cu,
             cu_cpu_long=cu.long(), max_len=56)
before = copy.deepcopy(net.state_dict())
losses = []
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = net(batch)
    loss.backward()
    optimizer.step()
    losses.append(float(loss))
moved = {k for k, v in net.state_dict().items() if not torch.equal(v, before[k])}
params = {n for n, _ in net.named_parameters()}
assert params <= moved, sorted(params - moved)
assert all(torch.isfinite(p).all() for p in net.parameters())
print("DION3_LOCAL", report, "losses", [round(v, 4) for v in losses], flush=True)
