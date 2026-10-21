"""Depth-Attention (Zeng et al., arXiv:2606.05014) for the gated-attention layers (bench100m --depth-attn).

Before a softmax-attention layer attends over the sequence, its query attends along depth
over the keys of earlier attention layers at the same token, and mixes their depth-mixed
values into its own value:

    alpha_j = softmax_j(q . k_j / sqrt(d)),  j in D_m          (GQA: queries averaged per KV head)
    v~_m    = alpha_m v_m + sum_{j < m} alpha_j v~_j
    out     = CausalAttn(Q, K, V~)

No parameters are added and the value cache keeps its shape. Only the 5 softmax-attention
layers of the hybrid participate: Gated DeltaNet values feed a delta-rule state, which the
paper does not cover. D_m is the layer itself plus attention layers 0, s, 2s, ... below it,
with the paper's default half-depth stride s = ceil(n_attention / 2).
"""
import math

import torch


def enable_depth_attention(model, mixing="softmax"):
    """Number the attention layers and share one per-forward store among them."""
    assert mixing in ("softmax", "uniform")
    attention = [block.mixer for block in model.blocks if block.is_attn]
    stride = max(1, math.ceil(len(attention)/2))
    store = {}
    for index, mixer in enumerate(attention):
        mixer._depth_index = index
        mixer._depth_sources = [j for j in range(0, index, stride)]
        mixer._depth_store = store
        mixer._depth_mixing = mixing
    # Each forward starts with an empty store; earlier layers' tensors stay in the graph.
    model.register_forward_pre_hook(lambda module, inputs: store.clear())
    return dict(attention_layers=len(attention), stride=stride,
                sources={m._depth_index: m._depth_sources for m in attention})


def depth_mix(mixer, q, k, v):
    """q: [N, H, d], k and v: [N, KV, d] for this layer. Returns the depth-mixed value."""
    store = mixer._depth_store
    # A source layer skipped by layer dropout in this forward is simply not mixed in.
    sources = [j for j in mixer._depth_sources if j in store]
    keys = [store[j][0] for j in sources] + [k]
    values = [store[j][1] for j in sources] + [v]
    if len(keys) > 1:
        if mixer._depth_mixing == "uniform":
            weights = torch.full((*k.shape[:2], len(keys)), 1/len(keys), device=k.device)
        else:
            n, heads, dim = q.shape
            query = q.float().view(n, k.shape[1], heads//k.shape[1], dim).mean(2)   # [N, KV, d]
            scores = torch.stack([(query*key.float()).sum(-1) for key in keys], -1)  # [N, KV, S]
            weights = torch.softmax(scores/math.sqrt(dim), -1)
        mixed = sum(weights[..., i, None]*value.float() for i, value in enumerate(values))
        v = mixed.to(v.dtype)
    store[mixer._depth_index] = (k, v)
    return v
