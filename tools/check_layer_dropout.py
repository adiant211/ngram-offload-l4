"""Local checks for layer_dropout on the tiny preset (reference kernels, no compile).

Usage: python tools/check_layer_dropout.py <dir with a100stack source + layer_dropout.py>
"""
import sys
import types

sys.path.insert(0, sys.argv[1])
sys.modules.setdefault("resource", types.ModuleType("resource"))
import numpy as np
import torch
import model as stack
from config import preset
import layer_dropout as ld

torch.manual_seed(7)
c, _ = preset("tiny")
c.ngram.enabled = False
net = stack.HybridLM(c).cuda().train()
n = 96
cu = torch.tensor([0, 40, n], dtype=torch.int32)
batch = dict(ids=torch.randint(0, 512, (n,), device="cuda"), targets=torch.randint(0, 512, (n,), device="cuda"),
             pos=torch.cat([torch.arange(40), torch.arange(56)]).cuda(), cu=cu.cuda(), cu_cpu=cu,
             cu_cpu_long=cu.long(), max_len=56)
L = len(net.blocks)
reference = float(net(batch))
ld.install(net)
assert abs(float(net(batch)) - reference) < 1e-6, "no plan must equal the original forward"
net._layer_plan = ld.LayerPlan([True]*L, [1.0]*L)
assert abs(float(net(batch)) - reference) < 1e-6, "all kept, unscaled must equal the original"

# Drop block 2, scale block 3 by 1.25: compare with a hand-written forward.
net._layer_plan = ld.LayerPlan([True, True, False, True], [1.0, 1.0, 1.0, 1.25])
got = net(batch)
x = net.embed(batch["ids"])
for i, block in enumerate(net.blocks):
    if i == 2:
        continue
    if i == 3:
        x = block.res1(x, 1.25*block.mixer(block.norm1(x), batch))
        x = block.res2(x, 1.25*block.mlp(block.norm2(x)))
    else:
        x = block(x, batch)
expected = net._loss(net.norm(x), batch["targets"])
torch.testing.assert_close(got, expected)
net.zero_grad(set_to_none=True)
got.backward()
assert all(p.grad is None for p in net.blocks[2].parameters()), "a dropped block gets no gradient"
assert all(p.grad is not None for p in net.blocks[3].parameters() if p.requires_grad)

# Schedule: increasing-in-depth, decreasing-in-time averages p_max/4 (paper Eq. 9).
p_max, layers, horizon = 0.4, 20, 101
mean = np.mean([[ld.drop_rate(l, layers, t, horizon, p_max, "ild", "decreasing") for l in range(layers)]
                for t in range(horizon)])
assert abs(mean - p_max/4) < 1e-9, mean
sampler = ld.LayerDropout(layers, p_max, "ild", "constant", horizon, seed=3)
plans = [sampler.plan(0) for _ in range(4000)]
realized = sampler.report()["layer_drop_realized_rate"]
assert abs(realized - p_max/2) < 0.01, realized
assert all(p.keep[0] for p in plans), "ILD never drops the first block"
kept_last = [p.scale_values[-1] for p in plans if p.keep[-1]]
assert all(abs(s - 1/(1-p_max)) < 1e-12 for s in kept_last), "kept blocks scale by 1/(1-p)"
print("LAYER_DROPOUT_CHECK passed: identity paths, skip+scale forward, no grads for dropped blocks,",
      f"ILD+DTS mean rate {mean:.4f} = p_max/4, realized ILD rate {realized:.4f} ~ p_max/2", flush=True)
