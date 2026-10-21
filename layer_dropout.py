"""Layer dropout / stochastic depth (Elhoushi et al., arXiv:2609.05275) for bench100m.

Paper recipe: drop whole blocks (mixer + MLP together), rate increasing with depth
(p_l = p_max * l/(L-1)), decreasing over training (p *= 1 - t/(T-1)), kept blocks'
residual branches scaled by r = 1/(1-p) during training and 1 at evaluation.

Adaptation to this stack: one Bernoulli draw per block per *microbatch* (the paper
prefers per sequence). A microbatch is one packed varlen stream, so per-sequence
skipping would change every layer's token count, recompiling the static-shape regions
and re-planning varlen kernels; 96 microbatches per update still give 96 independent
masks. Dropped blocks are not executed, so the FLOPs are really saved.
"""
import types

import numpy as np
import torch


def drop_rate(layer, layers, step, horizon, p_max, distribution, schedule, p_final=0.0):
    """Scheduled rate times the depth profile. "decreasing" moves linearly from p_max to p_final
    over `horizon` steps and holds p_final afterwards (p_final = 0: the paper's schedule)."""
    if distribution == "ild":
        depth = layer/max(1, layers-1)
    elif distribution == "ald":
        depth = 1.0 if layer % 2 == 1 else 0.0
    else:
        depth = 1.0
    progress = step/max(1, horizon-1)
    if schedule == "decreasing":
        p = p_final + (p_max-p_final)*max(0.0, 1-progress)
    elif schedule == "increasing":
        p = p_max*min(1.0, progress)
    else:
        p = p_max
    return p*depth


class LayerPlan:
    """Which blocks run for one microbatch, and each kept block's residual scale."""

    def __init__(self, keep, scale):
        self.keep = keep
        self.scale_values = scale
        values = torch.tensor(scale, dtype=torch.float32).pin_memory()
        self.scale = values.to("cuda", non_blocking=True)


class LayerDropout:
    def __init__(self, layers, p_max, distribution, schedule, horizon, seed, p_final=0.0):
        self.layers, self.p_max, self.horizon, self.p_final = layers, p_max, horizon, p_final
        self.distribution, self.schedule = distribution, schedule
        self.rng = np.random.default_rng(seed)
        self.dropped = self.drawn = 0

    def plan(self, step):
        rates = [drop_rate(l, self.layers, step, self.horizon, self.p_max,
                           self.distribution, self.schedule, self.p_final) for l in range(self.layers)]
        keep = [bool(self.rng.random() >= p) for p in rates]
        self.dropped += keep.count(False)
        self.drawn += len(keep)
        return LayerPlan(keep, [1/(1-p) if k and p > 0 else 1.0 for k, p in zip(keep, rates)])

    def report(self):
        return dict(layer_drop_realized_rate=self.dropped/max(1, self.drawn))


def exit_plan(layers, exit_at):
    """Evaluation: run the first exit_at blocks unscaled (r_eval = 1), skip the rest."""
    plan = LayerPlan([i < exit_at for i in range(layers)], [1.0]*layers)
    return plan


def scaled_tail(self, x, mixed, scale):
    x = self.res1(x, scale*mixed)
    return self.res2(x, scale*self.mlp(self.norm2(x)))


def scaled_block(self, x, batch, scale):
    if hasattr(self, "_tail_scaled"):
        return self._tail_scaled(x, self.mixer(self.norm1(x), batch), scale)
    x = self.res1(x, scale*self.mixer(self.norm1(x), batch))
    return self.res2(x, scale*self._mlp(x))


def layerdrop_forward(self, batch):
    """HybridLM.forward with an optional per-microbatch LayerPlan (no checkpointing)."""
    plan = getattr(self, "_layer_plan", None)
    x = self.embed(batch["ids"])
    memory = self.ngram.lookup(batch["ids"], batch["pos"]) if self.ngram is not None else None
    for i, block in enumerate(self.blocks):
        if i in self.inject_at:
            x = self.ngram.inject(self.inject_at[i], x, memory)
        if plan is None:
            x = block(x, batch)
        elif plan.keep[i]:
            x = (block(x, batch) if plan.scale_values[i] == 1.0
                 else scaled_block(block, x, batch, plan.scale[i]))
    return self._loss(self.norm(x), batch["targets"])


def install(model, compile_mode=None):
    assert not model.full_ckpt and not any(b.ckpt_mlp for b in model.blocks), \
        "layer dropout path does not implement activation checkpointing"
    model.forward = types.MethodType(layerdrop_forward, model)
    model._layer_plan = None
    for block in model.blocks:
        if hasattr(block, "_tail"):
            block._tail_scaled = torch.compile(types.MethodType(scaled_tail, block), fullgraph=True,
                                               dynamic=False, mode=compile_mode or "default")
