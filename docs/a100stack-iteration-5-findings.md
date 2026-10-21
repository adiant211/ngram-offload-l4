# a100stack iteration 5: 700M dense + 1.2B n-gram table on one A100, recoverable

Recorded 21 October 2026 on Colab: NVIDIA A100-SXM4-80GB, 12 vCPUs, 167 GB RAM,
Torch 2.11.0+cu130, FLA 0.5.2, `microsoft/dion` at `58d38ad`.

## 1. Model

The source has no 700M preset. `trainer.py --preset 700m` widens its `500m` preset
(32 layers, 3:1 Gated DeltaNet : gated attention, head dim 128) from 1,024 to 1,280:

| | |
|---|---|
| Layers | 32: 24 Gated DeltaNet + 8 gated softmax attention (every 4th) |
| Width | 1,280 |
| Attention | 10 query heads × 128, 2 KV heads, QK-norm, RoPE θ = 1e6, sigmoid output gate |
| Gated DeltaNet | 10 heads × 128, short conv 4 |
| MLP | SwiGLU, 3,200 |
| Vocabulary | 60,800, tied embedding / output head |
| Dense parameters | **715,990,240** |
| N-gram table | **1,199,998,976** BF16 parameters (2.4 GB) in pinned host RAM: bigram + trigram × 2 hashes = 4 tables × 585,937 rows × 512, injected before block 2 |
| Optimizers | Dion3 on 257 matrices (637.0M); AdamW on 340 tensors (79.0M); sparse row RMSprop on the table |

## 2. Throughput and GPU memory

Same recipe as iteration 4 (packed projections, concat packing for Dion3, compiled block tails,
patched fused cross-entropy with 4 chunks, staged n-gram path with the blocked CPU update on all 12
threads), ~786k-token updates, real FineWeb data, 6 timed updates after 2 untimed ones.
Device memory is what `nvidia-smi` reports (allocator reservations plus CUDA context).

| Microbatch × accumulation | Tokens/s | MFU | vs best | Allocated / reserved peak | Device |
|---|---:|---:|---:|---:|---:|
| 16,384 × 48 | **27,276** | 38.6% | — | 41.1 / 41.5 GiB | 42.1 GiB |
| 12,288 × 64 | 26,185 | 37.0% | −4.0% | 33.6 / 33.8 GiB | 35.1 GiB |
| 12,288 × 64, BF16 residual | 26,209 | 37.1% | −3.9% | 33.5 / 33.7 GiB | 34.4 GiB |
| 10,240 × 77 | 24,835 | 35.1% | −9.0% | 29.8 / 30.0 GiB | 30.6 GiB |
| **10,240 × 77 + FLA sync removal (default, §3)** | **26,456** | **37.4%** | **−3.0%** | 29.8 / 30.0 GiB | **30.7 GiB** |
| 10,240 × 77 + layer dropout p_max 0.2 (constant, before §3) | 27,293 | — | — | 29.8 / 30.0 GiB | 30.6 GiB |
| 8,192 × 96 | 22,875 | 32.4% | −16.1% | 25.5 / 25.9 GiB | 26.5 GiB |

- **Target "< 34 GB GPU memory, ≤ 20% slower":** 10,240-token microbatches. The run script also
  sets `--gpu-mem-cap-gib 31`, so the allocator raises an error instead of growing past 31 GiB.
  Under the cap the device stays below 31.7 GiB, i.e. below 34 GB in decimal units too.
- Memory is close to linear in microbatch size: ≈ 9.4 GiB fixed (FP32 weights and gradients,
  Dion3 momentum, AdamW moments for the embedding, BF16 weight copies, CUDA context)
  + ≈ 2.1 GiB per 1,024 microbatch tokens.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by the trainer) keeps the reserved
  peak within 0.3 GiB of the allocated peak.
- The BF16 residual option saved almost nothing here: the n-gram injection before block 2
  multiplies by an FP32 gate and the stream returns to FP32. Not recommended.
- Per update (16k microbatches): forward/backward ≈ 27.2 s, Dion3 step 0.2 s, CPU table update
  1.4 s (streams 2.4 GB of table plus a 4.8 GB FP32 gradient accumulator).
- Layer dropout (iteration 4), up to 20% per block rising with depth (~10% of blocks skipped):
  +9.9% at the same memory (measured before the sync removal). Its MFU column is omitted
  because the FLOP count assumes every block runs.
- Rows other than the default were measured before the sync removal of section 3, which
  should lift them by a similar ~6%.

## 3. Kernel profile and the decision against kernel rewrites

CUDA kernel time per 10,240-token microbatch (`trainer.py --profile-micro 8`), 367 ms in total:

| Category | ms | Share | Assessment |
|---|---:|---:|---|
| GEMMs (cuBLAS/CUTLASS) | 207 | 56% | ≈ 212 TFLOP/s, 68% of A100 BF16 peak for K = 1,280 shapes: near the practical ceiling |
| FLA Gated DeltaNet path | 62 | 17% | 23 ms of it is the width-4 short convolution (forward recomputed in backward) |
| Other ATen elementwise | 41 | 11% | FP32 gradient accumulation (~10 ms), BF16 weight casts, per-microbatch weight concatenation (~5 ms) |
| Inductor fused tails | 30 | 8% | already fused |
| Copies (overlapped), cross-entropy, norms | 27 | 7% | small |

Decision: **no kernel rewrites.** The candidates (a fused short convolution without recompute,
FP32-accumulating weight-gradient GEMMs, faster DeltaNet kernels) are worth roughly 1–5% each and
need custom kernels or autograd plus gradient parity tests.

The profile also showed the GPU busy only 89% of the wall time. `--sync-debug` found five
CPU–GPU synchronizations per microbatch, all inside FLA 0.5.2:

- three from `prepare_chunk_indices` (varlen chunk tables for the short convolutions and the
  chunked delta rule): a `repeat_interleave` that reads a GPU sum, and a pageable host-to-device copy;
- two from the fused cross-entropy backward, which reads `do != 1` on the host to skip a multiply.

`fla_nosync.py` (on by default, `--no-fla-nosync` to disable):

- builds the chunk tables on the host from the batch's CPU `cu_seqlens`, copying them through pinned memory;
- always applies the upstream gradient in the cross-entropy backward. That is exact, since x · 1.0 = x.

Its parity check requires identical chunk tables and bit-identical cross-entropy gradients
(upstream gradient 1.0 and 0.37). Results:

- zero synchronizations per microbatch;
- 616 of 616 tables built on the host;
- GPU busy fraction 89.9% → 97.9%;
- wall time per microbatch 409 → 376 ms;
- **24,835 → 26,456 tokens/s (+6.5%)** at unchanged memory.

The n-gram validity mask now also goes through pinned memory (+0.7%, within noise).

## 4. Recoverability

`trainer.py` is a training loop, not a benchmark:

- **Data:** deterministic. Update s, rank r, microbatch m always read the same window of the
  token file (a seeded block permutation per epoch), built by a background thread into pinned
  memory. A resumed run reads exactly the data the uninterrupted run would have.
- **Checkpoints:** one directory per checkpoint, written by a background thread from CPU
  snapshots taken at an update boundary:
  `model.pt` (dense weights), `ngram.pt` (table + row state), `optim_rank{r}.pt` (Dion3/AdamW
  state, per rank), `state_rank{r}.pt` (CPU/CUDA/NumPy/Python RNG, layer-dropout RNG),
  `meta.json` (step, tokens, schedule position, configuration, FP64 checksums).
  The directory is `step_N.partial` until every file is fsynced and `COMPLETE` is written,
  then renamed. Partial directories are never resumed from and are cleaned up at start.
  8.8 GB each, written in 32 s in the background; `--keep` prunes old ones.
- **Resume:** `--resume auto` takes the newest complete checkpoint (an older one if loading
  fails), restores everything above, then recomputes the checksums from the *live* model,
  table and optimizer and refuses to continue on a mismatch.
- **Stopping:** SIGTERM/SIGINT or `--max-minutes` finish the current update, checkpoint, exit 0.
- **Nonfinite updates** are skipped (dense gradients and the n-gram accumulator discarded);
  more than `--max-skips` in a row exits with code 3.
- **Supervisor:** `train700m.sh` restarts the trainer with `--resume auto` after any nonzero exit
  and keeps the compile caches next to the checkpoints.

### Recovery test (`recovery_test.sh`, full model size, 65k-token updates)

| Scenario | Outcome |
|---|---|
| Hard exit while checkpoint 6 is being written | `step_6.partial` ignored; resumed from step 3, checksums verified |
| Hard exit in the middle of update 8 | Checkpoint 6 was still being written (32 s write), so again resumed from step 3, verified |
| Uninterrupted finish | Checkpoints 6 and 9 complete, exit 0 |
| SIGTERM after resuming from 9 | Finished update 12, wrote checkpoint 12, exit 0 |

Loss after a resume versus an uninterrupted run: at most 0.26% relative difference (steps 7–8).
Two uninterrupted runs differ by up to 0.10% from GPU nondeterminism alone (atomics in the
backward kernels), and these early updates move the loss by ~10% each, so trajectories are not
bit-reproducible in either case. The state itself is restored exactly: the checksums of the live
weights, table and optimizer state match the saved ones.

### Restart cost

The first update of a fresh process compiles everything: about 12 minutes on a cold cache,
8 of them in Dion3's compiled helpers (they are traced per megabatch shape). With warm caches a
restart reaches its first update in about 1 minute. `train700m.sh` keeps the Inductor and Triton
caches in `CACHE_DIR` (default `/content/compile-cache`); put it on persistent storage together
with `CKPT_DIR` to keep restarts on a fresh runtime short. `--no-dion-compile` runs Dion3's
helpers eagerly for even faster starts.

### Supervised run

MAIN_RUN

## 5. Limits and next steps

- Checkpoints in `/content` survive process crashes, not the loss of the Colab runtime. For
  that, mount Google Drive (or other storage) and pass `CKPT_DIR` and `CACHE_DIR` there.
- The CPU table update (1.4 s per update, ~4.5%) is serial with the GPU. Hiding it needs a
  one-update-stale table, which changes the algorithm; not done.
- Multi-GPU uses the iteration-4 paths (DDP, Dion3 process group, n-gram accumulator
  all-reduce); checkpoint files are per rank and resuming requires the same world size.
- These are throughput measurements and short runs, not a convergence study.
