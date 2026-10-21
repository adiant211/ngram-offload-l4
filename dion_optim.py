"""Dion3 (microsoft/dion NorDion2) for bench100m, on one GPU or across DDP ranks.

Grouping follows a100stack's optim.py:
  2-D hidden matrices (min dim >= 16)               -> Dion3, algorithm "nordion2"
  embedding (tied lm_head), norms, gates, A_log,
  dt_bias, conv kernels, small a/b projections      -> AdamW inside Dion3
  n-gram table                                      -> its own sparse row RMSprop, not here

Pass the DDP process group as 'process_group': Dion3 then splits Newton-Schulz
work across ranks and all-gathers the results. DDP has already averaged gradients.

NorDion2 orthogonalizes each parameter as one matrix and does not support
'split_sizes', so a packed QKV/gate weight would get a different update from the
separate matrices the model defines. ConcatProjection keeps the parameters
separate and concatenates them only in the forward pass, so the GEMM stays fused.
"""
import torch
import torch.nn.functional as F


class ConcatProjection:
    """One fused GEMM over several bias-free Linear layers, without merging their parameters.

    A plain object rather than an nn.Module, so the parameters stay registered once,
    under their original names; state_dicts stay compatible with a100stack checkpoints.
    """

    def __init__(self, linears):
        assert all(linear.bias is None for linear in linears)
        self.linears = list(linears)

    def __call__(self, x):
        return F.linear(x, torch.cat([linear.weight for linear in self.linears]))


def concat_projections(module, names, attribute="in_proj"):
    setattr(module, attribute, ConcatProjection([getattr(module, name) for name in names]))


def split_params(model):
    matrix, embed, head, other = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("embed."):
            embed.append(param)
        elif name.startswith("lm_head."):
            head.append(param)
        elif param.dim() == 2 and min(param.shape) >= 16:
            matrix.append(param)
        else:
            other.append(param)
    return matrix, embed, head, other


def eager_dion():
    """Replace dion's torch.compile-wrapped helpers with the original functions.

    The compiled helpers are traced again for every megabatch shape in a fresh process,
    which costs minutes per (re)start of a large model. Returns how many were unwrapped.
    """
    import sys
    import dion  # noqa: F401  (loads the submodules)
    count = 0
    for name, module in list(sys.modules.items()):
        if not (name == "dion" or name.startswith("dion.")) or module is None:
            continue
        for attribute, value in list(vars(module).items()):
            original = getattr(value, "_torchdynamo_orig_callable", None)
            if callable(value) and original is not None:
                setattr(module, attribute, original)
                count += 1
    return count


def build_dion3(model, args, process_group=None):
    from dion import Dion3
    matrix, embed, head, other = split_params(model)
    scalar = dict(lr=args.scalar_lr, betas=(0.9, 0.95), weight_decay=0.0)
    groups = [dict(params=matrix, algorithm="nordion2")] + [
        dict(params=params, algorithm="adamw", **scalar) for params in (embed, head, other) if params]
    optimizer = Dion3(groups, distributed_mesh=process_group, lr=args.dion_lr,
                      fraction=args.dion_fraction, mu=args.dion_mu, muon_beta2=0.95,
                      betas=(0.9, 0.95), weight_decay=args.dion_weight_decay,
                      adjust_lr="spectral_norm", use_triton=args.dion_triton)
    report = dict(matrix_tensors=len(matrix), matrix_params=sum(p.numel() for p in matrix),
                  adamw_tensors=len(embed)+len(head)+len(other),
                  adamw_params=sum(p.numel() for p in embed+head+other))
    return optimizer, report
