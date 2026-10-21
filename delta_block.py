"""Delta Block attention residuals for a100stack (trainer --delta-block).

From "Delta Attention Residuals: Per-Sublayer Sources for Cross-Layer Information Flow"
(Luo, Cai, Hu; github.com/wdlctc/delta-attention-residuals-code, mode "delta_block").

Every sublayer (mixer and MLP) reads x + sum_i softmax_i(q . RMSNorm_w(s_i)) s_i instead of x,
where the sources s_i come from block boundaries (blocks of ceil(L / num_blocks) layers): at
the first layer of each block, before its mixer, a new source is closed from the residual
stream x. The residual stream itself is unchanged; only sublayer inputs see the mix. Each
sublayer has its own query q and RMSNorm weight w: 4 * d parameters per layer.

Source construction:
  "repo"  - exactly the published code: s_0 = x_0 (embedding), s_k = x_k - s_{k-1}. Beyond the
            second block these are alternating sums (s_2 = delta_2 + x_0, s_3 = delta_3 + delta_1).
            This is what produced the published results.
  "paper" - true block deltas, as described: s_k = x_k - x_{k-1} (x_k: stream at boundary k).

Port: the mixing is fused into the compiled regions (mixer input: mix + norm1; MLP input:
residual add + mix + norm2 + MLP + residual add), so no stacked [N, T, D] copies are built or
saved for backward. Logits are computed in FP32 (the published code's einsum runs in BF16
under autocast). Layer-dropout plans and early-exit evaluation keep working.
"""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def select(sources, query, weight, eps):
    """sum_i softmax_i(q . RMSNorm_w(s_i)) s_i, without materializing a stacked copy."""
    qw = weight * query
    logits = torch.stack([(s * qw).sum(-1) * torch.rsqrt(s.pow(2).mean(-1) + eps) for s in sources])
    probs = logits.softmax(0)
    out = probs[0].unsqueeze(-1) * sources[0]
    for p, s in zip(probs[1:], sources[1:]):
        out = out + p.unsqueeze(-1) * s
    return out


def mixer_input(self, x, *sources):
    h = x + select(sources, self.delta_attn_query, self.delta_attn_weight, self._delta_eps)
    return F.rms_norm(h, (h.shape[-1],), self.norm1.weight, self.norm1.eps)


def delta_tail(self, x, mixed, scale, *sources):
    x = self.res1(x, scale * mixed)
    h = x + select(sources, self.delta_mlp_query, self.delta_mlp_weight, self._delta_eps)
    return self.res2(x, scale * self.mlp(self.norm2(h)))


def run_block(block, x, batch, attn_sources, mlp_sources, scale):
    if attn_sources:
        normed = block._delta_mixer_input(x, *attn_sources)
    else:
        normed = block.norm1(x)
    return block._delta_tail(x, block.mixer(normed, batch), scale, *mlp_sources)


def delta_forward(self, batch):
    """HybridLM.forward with Delta Block sources (and an optional layer-dropout plan)."""
    plan = getattr(self, "_layer_plan", None)
    x = self.embed(batch["ids"])
    memory = self.ngram.lookup(batch["ids"], batch["pos"]) if self.ngram is not None else None
    sources, last_boundary = [], None
    for i, block in enumerate(self.blocks):
        if i in self.inject_at:
            x = self.ngram.inject(self.inject_at[i], x, memory)
        attn_sources = list(sources)
        if i % self._delta_layers_per_block == 0:
            if not sources:
                sources.append(x)
            elif self._delta_variant == "repo":
                sources.append(x - sources[-1])
            else:
                sources.append(x - last_boundary)
            last_boundary = x
        if plan is not None and not plan.keep[i]:
            continue
        scale = self._delta_one if plan is None else plan.scale[i]
        x = run_block(block, x, batch, attn_sources, list(sources), scale)
    return self._loss(self.norm(x), batch["targets"])


def install(model, num_blocks=4, variant="repo", compile_mode="default", compile=True):
    assert variant in ("repo", "paper")
    assert not model.full_ckpt and not any(b.ckpt_mlp for b in model.blocks), \
        "the Delta Block path does not implement activation checkpointing"
    layers = len(model.blocks)
    per_block = max(1, math.ceil(layers / num_blocks))
    d = model.c.d_model
    device = model.embed.weight.device
    generator = torch.Generator(device="cpu").manual_seed(4242)
    added = 0
    for i, block in enumerate(model.blocks):
        attn_sources = (i + per_block - 1) // per_block      # boundaries strictly before layer i
        if attn_sources:
            block.delta_attn_query = nn.Parameter(
                (torch.randn(d, generator=generator) * 0.02).to(device))
            block.delta_attn_weight = nn.Parameter(torch.ones(d, device=device))
            added += 2 * d
            fn = types.MethodType(mixer_input, block)
            block._delta_mixer_input = torch.compile(fn, fullgraph=True, dynamic=False,
                                                     mode=compile_mode) if compile else fn
        block.delta_mlp_query = nn.Parameter((torch.randn(d, generator=generator) * 0.02).to(device))
        block.delta_mlp_weight = nn.Parameter(torch.ones(d, device=device))
        added += 2 * d
        block._delta_eps = model.c.norm_eps
        fn = types.MethodType(delta_tail, block)
        block._delta_tail = torch.compile(fn, fullgraph=True, dynamic=False,
                                          mode=compile_mode) if compile else fn
    model._delta_layers_per_block, model._delta_variant = per_block, variant
    model._delta_one = torch.ones((), device=device)
    model._layer_plan = getattr(model, "_layer_plan", None)
    model.forward = types.MethodType(delta_forward, model)
    return dict(num_blocks=math.ceil(layers / per_block), layers_per_block=per_block,
                variant=variant, parameters=added,
                sources_per_layer=[min(num_blocks, i // per_block + 1) for i in range(layers)])


def check():
    """Compiled port vs the eager reference on the 100m model: loss and gradients, both variants."""
    import numpy as np
    from types import SimpleNamespace
    import bench100m as bench
    import model as stack
    from optim800k import apply_optimizations
    for variant in ("repo", "paper"):
        torch.manual_seed(7)
        c = bench.config()
        c.ngram.total_params = 1 << 22
        net = stack.HybridLM(c).cuda().train()
        net._loss = types.MethodType(bench.flat_fused_loss, net)
        opts = SimpleNamespace(pack=True, pack_mode="concat", attn="native", compile="tail",
                               compile_mode="default", ckpt="none", residual_dtype="float32",
                               fused_norms=True, loss_chunks=4)
        apply_optimizations(net, opts)
        bench.optimize(net, opts)
        report = install(net, num_blocks=4, variant=variant)
        with torch.no_grad():   # sharp, non-uniform routing so the mix actually matters
            for name, p in net.named_parameters():
                if name.endswith("_query"):
                    p.mul_(400)
        batch = bench.build_batch(np.random.default_rng(3).integers(4, 60800, 4097, dtype=np.uint16), 3, 1024)
        results = []
        for forward in (delta_forward, reference_forward):
            net.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward(net, batch)
            loss.backward()
            net.ngram._pending.clear()
            grads = {n: p.grad.detach().clone() for n, p in net.named_parameters()
                     if p.grad is not None and ("delta" in n or n.endswith(("blocks.9.mlp.w_down.weight",
                                                                            "blocks.3.mixer.o_proj.weight")))}
            results.append((float(loss), grads))
        (l0, g0), (l1, g1) = results
        rel = {n: float((g0[n] - g1[n]).norm() / g1[n].norm().clamp_min(1e-12)) for n in g1}
        worst = max(rel, key=rel.get)
        print("DELTA_BLOCK_PARITY", variant, dict(loss=l0, reference=l1, loss_rel=abs(l0 - l1) / abs(l1),
              grads=len(rel), worst_grad=worst, worst_rel=rel[worst],
              params=report["parameters"]), flush=True)
        assert abs(l0 - l1) / abs(l1) < 2e-3 and rel[worst] < 5e-2, (variant, l0, l1, worst, rel[worst])


def reference_forward(model, batch):
    """Plain eager implementation following the published code (stacked sources, RMSNorm
    module math, einsum), for the equivalence check only."""
    x = model.embed(batch["ids"])
    memory = model.ngram.lookup(batch["ids"], batch["pos"]) if model.ngram is not None else None
    sources, last = [], None
    eps = model.c.norm_eps

    def mix(srcs, query, weight):
        V = torch.stack(srcs, 0).float()
        K = V * torch.rsqrt(V.pow(2).mean(-1, keepdim=True) + eps) * weight
        weights = torch.einsum("d,ntd->nt", query.float(), K).softmax(0)
        return torch.einsum("nt,ntd->td", weights, V)

    for i, block in enumerate(model.blocks):
        if i in model.inject_at:
            x = model.ngram.inject(model.inject_at[i], x, memory)
        attn_sources = list(sources)
        if i % model._delta_layers_per_block == 0:
            sources.append(x if not sources else (x - sources[-1] if model._delta_variant == "repo"
                                                  else x - last))
            last = x
        h = x + mix(attn_sources, block.delta_attn_query, block.delta_attn_weight) if attn_sources else x
        x = block.res1(x, block.mixer(block.norm1(h), batch))
        h = x + mix(sources, block.delta_mlp_query, block.delta_mlp_weight)
        x = block.res2(x, block.mlp(block.norm2(h)))
    return model._loss(model.norm(x), batch["targets"])


if __name__ == "__main__":
    check()
