"""Optional precision and fused-normalization experiments for the 800k batch sweep."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


def bf16_embedding_output(module, inputs, output):
    return output.to(torch.bfloat16)


def apply_optimizations(model, args):
    if args.residual_dtype == "bfloat16":
        model.embed.register_forward_hook(bf16_embedding_output)
    if args.fused_norms:
        from fla.modules import RMSNorm
        # Reuse Parameter objects, including their FP32 dtype and initialization.
        for parent_path, parent in list(model.named_modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.RMSNorm):
                    # Inductor already fuses this norm inside the regional MLP.
                    # FLA 0.5.2's norm cannot be traced in that fullgraph region.
                    if args.compile in ("mlp", "tail") and name == "norm2" and parent_path.startswith("blocks."):
                        continue
                    fused = RMSNorm(child.weight.numel(), eps=child.eps,
                                    device=child.weight.device, dtype=child.weight.dtype)
                    fused.weight = child.weight
                    setattr(parent, name, fused)
    if model.loss_fn is not None:
        model.loss_fn.num_chunks = args.loss_chunks


def check_optimizations():
    import bench100m as bench
    import model as stack
    torch.set_num_threads(4)
    torch.manual_seed(654)
    c = bench.config()
    c.ngram.total_params = 65536
    baseline = stack.HybridLM(c).cuda().train()
    args = SimpleNamespace(pack=True, attn="native", compile="none", ckpt="none",
                           compile_mode="default", residual_dtype="float32",
                           fused_norms=False, loss_chunks=8)
    bench.optimize(baseline, args)
    batch = bench.build_batch(np.random.default_rng(987).integers(4, 60800, 258,
                             dtype=np.uint16), 3, 128)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = baseline(batch)
    a.backward()
    baseline.ngram._pending.clear()
    records = {}
    for name, dtype, fused in [("bf16_only", "bfloat16", False),
                                ("fused_only", "float32", True)]:
        candidate = copy.deepcopy(baseline)
        candidate.zero_grad(set_to_none=True)
        args.residual_dtype, args.fused_norms = dtype, fused
        apply_optimizations(candidate, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            b = candidate(batch)
        b.backward()
        loss_rel = float((a-b).abs()/a.abs())
        grad_rel = float((baseline.embed.weight.grad-candidate.embed.weight.grad).norm()
                         / baseline.embed.weight.grad.norm())
        assert torch.isfinite(b) and torch.isfinite(candidate.embed.weight.grad).all()
        record = dict(loss_relative_error=loss_rel, embedding_gradient_relative_error=grad_rel,
                      acceptable=loss_rel < 0.01 and grad_rel < 0.06,
                      dense_parameters=str(candidate.embed.weight.dtype), residual=dtype)
        records[name] = record
        print("OPTIM800K_ISOLATION", name, record, flush=True)
        del candidate
    Path("parity800k.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    check_optimizations()
