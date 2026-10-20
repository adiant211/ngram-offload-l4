# a100stack iteration 3: 786k-token updates at 29.2k tokens/s under 7 GiB

Recorded 21 October 2026 on a fresh Colab NVIDIA L4 runtime (12 vCPUs, 52 GB RAM, Torch 2.11.0+cu130, FLA/fla-core 0.5.2).

## Result

Same model, same 786,432-token update (device batch 8 × 1,024 × accumulation 96), same fixture, same optimizers:

| Configuration | Tokens/s | vs. baseline | Forward/backward per update | Sparse step | Allocated / reserved peak |
|---|---:|---:|---:|---:|---:|
| Iteration-2 recipe (baseline, re-measured) | 25,699 | — | 29.04 s | 1.55 s | 6.73 / 7.21 GiB |
| + block sparse update, 12 CPU threads | 26,649 | +3.7% | 28.94 s | 0.56 s | 6.73 / 7.25 GiB |
| + `tail` compile region | 27,163 | +5.7% | 28.37 s | 0.57 s | 6.41 / 6.92 GiB |
| + fast fused cross-entropy, 8 loss chunks | 28,895 | +12.4% | 26.65 s | 0.56 s | **6.30 / 6.58 GiB** |
| **+ fast fused cross-entropy, 4 loss chunks** | **29,178** | **+13.5%** | 26.38 s | 0.56 s | 6.53 / 6.80 GiB |

**Recommended:** the last row (`run-selected-iter3.sh`): fastest, and still under 7 GiB reserved. For maximum memory headroom, use 8 loss chunks (6.58 GiB reserved, −1.0% speed).

Every run: one warmup plus four timed updates (3,145,728 timed tokens), finite losses and sparse state, zero CPU-input fallbacks, 816,227 touched n-gram rows per update. Final training losses after five updates: 9.70910 (baseline), 9.70903 (tail), 9.70903 (fast CE, 8 chunks), 9.70716 (fast CE, 4 chunks; the chunk count changes floating-point summation order). The baseline here measured 25,699 tokens/s versus 26,229 in iteration 2 on a different runtime allocation; compare rows within this table only.

Fixture: the same bounded FineWeb `sample-10BT` sample as iteration 2, regenerated with `prepare800k.py`; its SHA-256 (`a5ff5040…c1f7f`) matched, so the data is byte-identical.

## What changed, and why it helped

### 1. Sparse n-gram update: masked contiguous blocks with reused scratch buffers

At 786k tokens per update, 816k of the table's 976,560 rows are touched (84%). The old update gathered touched rows, updated them and scattered them back with `index_add_`, then zeroed the 2 GB FP32 accumulator in a separate pass, all on 4 threads while the GPU idled.

The new `--sparse-update blocks` path streams the whole table in 4,096-row blocks. It applies the identical RMSprop arithmetic under the touched mask, writes into preallocated scratch buffers, and clears each accumulator block while it is still in cache. `--update-threads` gives this phase every CPU core, since the GPU is idle during it anyway.

The scratch buffers matter: the first block version allocated fresh 32 MB temporaries per block, which glibc serves with `mmap` and the kernel must fault in and zero every time. That version was *slower* than the row path (1.83 s). On the Colab VM, full-size CPU-only timings (`tools/bench_update.py`):

| Update path | Threads | Seconds |
|---|---:|---:|
| rows (old) | 4 | 1.54 |
| rows | 12 | 1.18 |
| blocks, scratch buffers | 4 | 0.51 |
| blocks, scratch buffers | 12 | 0.40 |

In training, the end-of-step sparse optimizer phase went from 1.55 s to 0.56 s per update.

**Numerics:** the BF16 table is bit-identical to the row path. Row-optimizer state can differ by at most one FP32 ulp, because vectorized and scalar loop tails round `add_(alpha=…)` differently; the check measured 1 element in 4,096 at 1e-7 relative.

### 2. `--compile tail`: residual adds and norm inside the compiled MLP region

Previously only `norm2 → MLP` was compiled. The `tail` region compiles `res1 → norm2 → MLP → res2` per block (the FP32 gated residual adds plus the MLP), so Inductor fuses the residual arithmetic with the norm and activation kernels. Forward/backward fell 2.2% and peak memory fell 0.31 GiB, because fewer FP32 residual-stream intermediates are saved for backward. The region traces as one full graph (local `aot_eager` check: identical loss and zero gradient difference).

### 3. `--fast-ce`: two fixes inside FLA 0.5.2's fused linear cross-entropy

A kernel profile (4 microbatches, `--profile-kernels`) showed the fused loss at about 25% of GPU time. Two causes:

- **A GPU→CPU sync per microbatch.** `total = target.ne(ignore_index).sum().item()` blocks the CPU until the GPU reaches the loss, 96 times per update, so the CPU loses its lead in launching work. The count is already known on the CPU from batch preparation, so it is now passed in.
- **An FP32 weight-gradient path.** For each chunk, FLA copied the `[60,800 × C]` BF16 logit gradient to FP32 (~250 MB written) and ran a TF32 `addmm` into the FP32 accumulator. Now one BF16 × BF16 GEMM accumulates in FP32 and writes straight into the FP32 accumulator (`torch.addmm(..., out_dtype=torch.float32)`).

`fast_flce.py` applies both changes by rewriting the FLA function's source at runtime. It asserts that each pattern it replaces is found exactly once, so a changed FLA version fails loudly instead of silently running the original path.

**Numerics (full-size check: 8,192 tokens × 512 × 60,800):** loss and input gradient are **bit-identical**. The lm-head weight gradient differs by 0.17% relative, because its activation operand is now rounded to BF16, as the logits GEMM already rounds it. That is still more precise than a plain autocast `nn.Linear`, whose weight gradient is produced in BF16.

GPU kernel time per microbatch fell from about 272 ms to 236 ms. The FP32 copy disappeared, so peak memory also fell.

## Stability fixes

- **CPU-input handoff (correctness).** The staged n-gram path found each batch's CPU token ids through a dict keyed by the CUDA tensor's `data_ptr()`. With a real data loader, the CUDA caching allocator reuses addresses, so a lookup could silently hash *another batch's* tokens. A model forward pre-hook now hands `(ids_cpu, pos_cpu)` to the lookup, consumed exactly once. If it is absent or the wrong size, the lookup falls back to a correct but synchronizing GPU→CPU copy and counts the fallback (`cpu_input_fallbacks` in every result; 0 in all runs here).
- **Thread-safe profiling counters.** The aggregation worker and the training thread both updated the stage-timing dict without a lock; read-modify-write races could drop samples. A lock now guards it.
- **Kernel profiler totals.** `--profile-kernels` now sums kernel rows only. Operator rows repeat their kernels' device time, which double-counted it.

## Where the time goes now (recommended recipe, 4 profiled microbatches)

| Kernel / operator | Share of GPU time |
|---|---:|
| Dense BF16 matmuls (`aten::mm`) | 32% |
| Gated DeltaNet chunk backward | 12% |
| Fused linear cross-entropy (forward-time gradients included) | 7% |
| Dtype-conversion copies (`aten::copy_`) | 7% |
| Causal conv1d backward | 5% |
| FP32 adds (gradient accumulation, residuals) | 4% |
| Compiled SwiGLU backward | 4% |
| logsumexp + cross-entropy kernels | 7% |

The remaining time is mostly model compute, at d_model 512 on a 300 GB/s card. Further gains likely need fewer FP32↔BF16 casts (FP32 master weights are cast per microbatch) or fused GDN preprocessing. Both change precision or kernel internals and were not attempted here.

## Verification

- `python pinned_ngram.py`: row and block updates against the reference GPU implementation, over two optimizer updates of four accumulated microbatches each, including buffer reuse, accumulator clearing and the synchronizing fallback path. Also the GPU-table/host-accumulator mode. Passed locally (RTX 3090) and on the L4.
- `python -c "from ram_ngram import check_ram_ngram; check_ram_ngram()"`: passed on the L4.
- `python fast_flce.py`: loss/dx bit-identical, dw relative error 0.0017. Passed on the L4.
- The `tail` region was checked locally with `aot_eager`; on the L4, Inductor's numerics show up as matching training losses.

Not done: held-out evaluation, long-run convergence, or multi-run variance. Timing excludes initialization, fixture preparation, staging, compilation and warmup, as in earlier iterations.

## Reproduce

On a Colab L4 runtime, with `a100stack.zip` and `good_tokenizer.zip` in `/content` and this repository's files copied next to `iter3_colab.sh`:

```bash
bash iter3_colab.sh setup     # FLA 0.5.2, source patches, FineWeb fixture (SHA-checked)
bash iter3_colab.sh parity    # n-gram parity checks
bash iter3_colab.sh fastce    # fast-CE parity + the two recommended-recipe runs
bash run-selected-iter3.sh    # the recommended recipe alone
```

Raw records: `results/results-iter3.jsonl` (nine runs, including the stage-timing breakdowns).
