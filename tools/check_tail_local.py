"""Local trace/numerics check for bench100m's tail region (aot_eager: no Triton on Windows)."""
import copy, sys, types
sys.path.insert(0, sys.argv[1])
sys.modules.setdefault("resource", types.ModuleType("resource"))
import torch
import bench100m as bench
import model as stack
from config import preset

torch.manual_seed(5)
c, _ = preset("tiny")
c.ngram.enabled = False
base = stack.HybridLM(c).cuda().train()
cand = copy.deepcopy(base)
for block in cand.blocks:
    block._tail = torch.compile(types.MethodType(bench.block_tail, block), fullgraph=True,
                                dynamic=False, backend="aot_eager")
    block.forward = types.MethodType(bench.tail_forward, block)
n = 96
cu = torch.tensor([0, 40, n], dtype=torch.int32)
b = dict(ids=torch.randint(0, 512, (n,), device="cuda"), targets=torch.randint(0, 512, (n,), device="cuda"),
         pos=torch.cat([torch.arange(40), torch.arange(56)]).cuda(), cu=cu.cuda(), cu_cpu=cu,
         cu_cpu_long=cu.long(), max_len=56)
for m in (base, cand):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(b)
    loss.backward()
    m.result = float(loss)
rel = max(float((p.grad-q.grad).norm()/p.grad.norm().clamp_min(1e-12))
          for p, q in zip(base.parameters(), cand.parameters()) if p.grad is not None)
print("TAIL_CHECK loss", base.result, cand.result, "max param-grad rel", rel)
assert abs(base.result-cand.result) < 1e-4 and rel < 1e-3
