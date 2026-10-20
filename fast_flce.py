"""Opt-in speedups for FLA 0.5.2's fused linear cross-entropy (bench100m --fast-ce).

Two changes, applied by rewriting fused_linear_cross_entropy_forward's source:
  1. The supervised-token count comes from the CPU batch instead of
     target.ne(ignore_index).sum().item(), a GPU-to-CPU sync per microbatch.
  2. Each chunk's weight gradient is one BF16 x BF16 GEMM with FP32 accumulation
     written straight into the FP32 accumulator. FLA instead copies the [V, C] BF16
     logit gradient to FP32 and runs a TF32 GEMM. The loss and dx paths are untouched;
     only dw's x operand is rounded to BF16, as the logits GEMM already does.
"""
import inspect
import re
import textwrap

import torch
import fla.modules.fused_linear_cross_entropy as flce

_SYNC = "total = target.ne(ignore_index).sum().item()"
_DW = re.compile(r"torch\.addmm\(\s*input=dw,\s*mat1=c_logits\.t\(\)\.to\(dtype=grad_dtype\),"
                 r"\s*mat2=c_x,\s*out=dw,\s*\)")


def _known_total(target, ignore_index):
    total, flce._next_total = getattr(flce, "_next_total", None), None
    if total is None:
        return target.ne(ignore_index).sum().item()
    return total


def _dw_update(dw, grad_logits, x):
    with torch.autocast("cuda", enabled=False):
        if dw.dtype == torch.float32 and grad_logits.dtype in (torch.bfloat16, torch.float16):
            torch.addmm(dw, grad_logits.t(), x.to(grad_logits.dtype), out_dtype=torch.float32, out=dw)
        else:
            torch.addmm(dw, grad_logits.t().to(dw.dtype), x.to(dw.dtype), out=dw)


def install(known_total=True, bf16_weight_grad=True):
    original = getattr(flce, "_original_forward", flce.fused_linear_cross_entropy_forward)
    flce._original_forward = original
    source = textwrap.dedent(inspect.getsource(original))
    if known_total:
        assert source.count(_SYNC) == 1, "FLA source changed: token-count sync not found"
        source = source.replace(_SYNC, "total = _known_total(target, ignore_index)")
    if bf16_weight_grad:
        source, count = _DW.subn("_dw_update(dw, c_logits, c_x)", source)
        assert count == 1, "FLA source changed: weight-gradient addmm not found"
    flce._known_total, flce._dw_update = _known_total, _dw_update
    exec(compile(source, flce.__file__, "exec"), flce.__dict__)
    # FusedLinearCrossEntropyFunction looks the forward up as a module global at call time.


def set_next_total(total):
    """Supervised (non-ignored) target count for the next loss call; consumed once."""
    flce._next_total = int(total)


def check():
    """Patched vs original on full-size shapes: loss and dx, and dw relative error."""
    torch.manual_seed(11)
    n, h, v = 8192, 512, 60800
    x = torch.randn(n, h, device="cuda")
    weight = (torch.randn(v, h, device="cuda") / h**0.5).requires_grad_(True)
    target = torch.randint(0, v, (n,), device="cuda")
    target[torch.rand(n, device="cuda") < 0.01] = -100
    original = getattr(flce, "_original_forward", flce.fused_linear_cross_entropy_forward)
    results = []
    for patched in (False, True):
        flce.fused_linear_cross_entropy_forward = original
        if patched:
            install()
            set_next_total((target != -100).sum().item())
        xi = x.clone().requires_grad_(True)
        weight.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flce.FusedLinearCrossEntropyLoss()(xi, target, weight)
        loss.backward()
        results.append((float(loss), xi.grad.clone(), weight.grad.clone()))
    (l0, dx0, dw0), (l1, dx1, dw1) = results
    rel = lambda a, b: float((a-b).norm()/a.norm())
    errors = dict(loss_abs=abs(l0-l1), dx_rel=rel(dx0, dx1), dw_rel=rel(dw0, dw1))
    assert errors["loss_abs"] == 0 and errors["dx_rel"] == 0 and errors["dw_rel"] < 0.01, errors
    print("FAST_CE_PARITY", errors, flush=True)


if __name__ == "__main__":
    check()
