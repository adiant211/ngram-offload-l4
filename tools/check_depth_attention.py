"""Unit checks for depth_attention.depth_mix (no model needed). Usage: python tools/check_depth_attention.py <dir>"""
import math
import sys
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
import torch
from depth_attention import depth_mix

torch.manual_seed(0)
n, heads, kv, d = 37, 4, 1, 16
store = {}
layers = [SimpleNamespace(_depth_index=i, _depth_sources=s, _depth_store=store, _depth_mixing="softmax")
          for i, s in enumerate([[], [0], [0], [0, 2], [0, 2]])]
qs = [torch.randn(n, heads, d, requires_grad=True) for _ in layers]
ks = [torch.randn(n, kv, d, requires_grad=True) for _ in layers]
vs = [torch.randn(n, kv, d, requires_grad=True) for _ in layers]
mixed = [depth_mix(layer, q, k, v) for layer, q, k, v in zip(layers, qs, ks, vs)]

assert torch.equal(mixed[0], vs[0]), "a layer with no sources keeps its value"
# Layer 1 by hand: softmax over [k0, k1] with the head-averaged query.
query = qs[1].view(n, kv, heads//kv, d).mean(2)
scores = torch.stack([(query*ks[0]).sum(-1), (query*ks[1]).sum(-1)], -1)/math.sqrt(d)
w = torch.softmax(scores, -1)
torch.testing.assert_close(mixed[1], w[..., 0, None]*mixed[0] + w[..., 1, None]*vs[1])
# Layer 3 reads the *mixed* values of layers 0 and 2.
mixed[4].sum().backward()
assert vs[0].grad is not None and ks[2].grad is not None and vs[1].grad is None, \
    "gradients reach exactly the source layers"

store.clear()
for layer in layers:
    layer._depth_mixing = "uniform"
mixed = [depth_mix(layer, q, k, v) for layer, q, k, v in zip(layers, qs, ks, vs)]
torch.testing.assert_close(mixed[1], (mixed[0]+vs[1])/2)
print("DEPTH_ATTENTION_CHECK passed: identity without sources, softmax by hand, mixed-value chaining, "
      "gradient routing, uniform averaging", flush=True)
