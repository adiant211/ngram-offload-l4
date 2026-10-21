# Architecture and modification stack

## Model (700m preset)

| | |
|---|---|
| Layers | 32: 24 Gated DeltaNet + 8 gated softmax attention (every 4th layer) |
| Width | 1,280 |
| Attention | 10 query heads × 128, 2 KV heads, QK-norm, RoPE θ = 1e6, sigmoid output gate |
| Gated DeltaNet | 10 heads × 128, short convolution of width 4 (FLA kernels) |
| MLP | SwiGLU, 3,200 |
| Residuals | pre-norm (RMSNorm), gated residual adds (x + 2·sigmoid(a)·f(x)) |
| Vocabulary | 60,800, embedding tied to the output head |
| Dense parameters | 715,990,240 |
| N-gram memory | 1,199,998,976 BF16 parameters in pinned CPU RAM: bigram + trigram, 2 hash functions each = 4 tables × 585,937 rows × 512. The lookup is added before block 2 through a 2,048→1,280 projection with a context gate |

The source is a100stack (`code/model.py`, `config.py`, `ngram.py`). The `700m` preset is its `500m` preset
(32 × 1,024) widened to 1,280. `code/model.py` has two small fixes applied (a flat input to the fused
loss, the conv weight shape).

## Modification stack

Defaults in **bold** are on. Everything else is opt-in.

**1. Model definition**
- Architecture, weights and parameter names unchanged, so checkpoints stay compatible.
- Opt-in: layer dropout, Depth-Attention, Delta Block, other shapes via `[model.shape]`.

**2. Execution (same math, faster)**
- **Projections:** q/k/v/gate and gate/up each run as one GEMM; parameters stay separate.
- **Attention:** native PyTorch variable-length attention instead of a per-document loop.
- **DeltaNet:** packed forward with FLA kernels and a host copy of the sequence boundaries.
- **Norms:** FLA fused RMSNorm.
- **Compilation:** `torch.compile` per block over the residual adds + norm + MLP (static shapes).
- **Loss:**
  - fused linear cross-entropy in 4 chunks;
  - no per-microbatch sync for the token count;
  - head weight gradient as one BF16 GEMM.
- **FLA syncs:** the CPU–GPU syncs inside FLA (chunk tables, loss backward) are removed. Bit-identical.
- **Loss weighting:** token-weighted loss across microbatches, with the loss read back once per update.

**3. N-gram memory in CPU RAM**
- Token ids are hashed on the CPU, rows gathered into pinned buffers and copied to the GPU asynchronously.
- Row gradients return on a side stream and are accumulated in an FP32 host buffer by a worker thread.
- Once per update, a sparse row-RMSprop step streams the table in 4,096-row blocks over all CPU cores,
  scaled by the learning-rate schedule.

**4. Optimizer and schedule**
- Dion3 on the 257 weight matrices; AdamW inside Dion3 on the embedding, norms and gates.
- Warmup-stable-decay schedule.

**5. Memory**
- Expandable allocator segments, so reserved memory stays close to allocated.
- Optional hard cap on PyTorch's allocator.

**6. Training loop (`code/trainer.py`)**
- **Data:** deterministic and resumable. Update s, microbatch m always reads the same window (seeded block
  order per pass), so a resumed run sees exactly the data the uninterrupted run would have.
- **Checkpoints:** a background thread writes `step_N.partial/` and renames it only when complete.
  - Files: dense weights, n-gram table + row state, optimizer state, all RNG states, and `meta.json`
    with FP64 checksums.
  - Resume takes the newest complete checkpoint, restores everything, and refuses to continue if
    the checksums of the live state disagree.
- **Stops and failures:**
  - SIGTERM/SIGINT/`max_minutes`: finish the update, checkpoint, exit 0.
  - Non-finite updates are skipped; repeated ones exit for a restart.
  - `exp.py` restarts the trainer after any crash.

## Data flow of one update

```
for each of grad_accum microbatches:
    CPU: hash token ids -> gather 4 table rows per token -> pinned buffer -> async copy to GPU
    GPU: embed -> block 1 -> add n-gram memory -> blocks 2..32 -> fused loss -> backward
    GPU->CPU (side stream): row gradients -> worker thread adds them into the FP32 host buffer
clip gradients -> Dion3/AdamW step (GPU) -> sparse RMSprop over the touched table rows (CPU, all cores)
```

## Code map

| File | Role |
|---|---|
| `trainer.py` | Training loop, data loader, checkpoints, resume, signals, diagnostics |
| `model.py`, `config.py`, `ngram.py` | a100stack model source |
| `bench100m.py` | Execution patches (packing, native attention, packed DeltaNet, compiled tails) |
| `optim800k.py` | Fused norms, loss chunks |
| `pinned_ngram.py`, `ram_ngram.py` | N-gram table in CPU RAM: lookup, gradient return, sparse update, parity checks |
| `fast_flce.py`, `fla_nosync.py` | Loss and FLA patches, with parity checks |
| `dion_optim.py` | Dion3 parameter grouping, concatenated projections |
| `layer_dropout.py`, `depth_attention.py`, `delta_block.py` | Opt-in paper techniques |
| `prepare_fineweb.py` | FineWeb tokenization into train/held-out files |
| `a100stack_train_original.py`, `data.py`, `optim.py`, `launch.sh`, … | The original a100stack trainer and helpers, for reference |
