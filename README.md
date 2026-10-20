# ngram-offload-l4

**Training-throughput engineering for a 100M hybrid language model with a 500M-parameter hashed n-gram memory, on a single 24 GB NVIDIA L4.**

The model pairs a 99.9M-parameter dense backbone (15 Gated DeltaNet layers, 5 gated attention layers, width 512, tied 60,800-token embedding) with a 500M-parameter hashed n-gram table (bigram and trigram, two hashes each, BF16). The table is trained with a sparse row-wise RMSprop optimizer. This repository holds the benchmark harness, the optimizations, and the measurement record for training it with ~786k-token updates inside a 7 GiB GPU memory budget.

## Headline

Same model, same 786,432-token update (8 × 1,024 tokens × 96 accumulation steps), same real FineWeb data, same optimizers, on a Colab L4:

| | Tokens/s | GPU memory (allocated / reserved peak) |
|---|---:|---:|
| Iteration-2 recipe | 25,699 | 6.73 / 7.21 GiB |
| **Iteration-3 recipe** | **29,178 (+13.5%)** | **6.53 / 6.80 GiB** |
| Iteration-3, lowest memory | 28,895 (+12.4%) | 6.30 / 6.58 GiB |

Loss and input gradients match the earlier implementation. The documented exceptions are the lm-head weight gradient (0.17% relative, from BF16 rounding of one GEMM operand) and one-ulp differences in sparse optimizer state. See [docs/a100stack-iteration-3-findings.md](docs/a100stack-iteration-3-findings.md).

## How it works

- **The n-gram table lives in pinned host RAM.** Each microbatch, the CPU hashes token ids, gathers the needed rows into reusable pinned buffers and ships them to the GPU asynchronously. The row gradients come back on a separate CUDA stream and a CPU worker accumulates them while the GPU keeps computing ([pinned_ngram.py](pinned_ngram.py)).
- **The sparse optimizer runs once per update**, streaming the table in masked 4,096-row blocks over all CPU cores, with reused scratch buffers (`--sparse-update blocks --update-threads N`).
- **Dense model:** packed projections, native PyTorch varlen attention, FLA Gated DeltaNet kernels, fused norms, and a compiled per-block region covering the residual adds, norm and SwiGLU MLP (`--compile tail`) ([bench100m.py](bench100m.py), [optim800k.py](optim800k.py)).
- **Loss:** FLA's fused linear cross-entropy, patched to drop a per-microbatch GPU→CPU sync and to compute the lm-head weight gradient as one BF16 GEMM with FP32 accumulation (`--fast-ce`, [fast_flce.py](fast_flce.py)).
- **Correct accumulation:** token-weighted loss normalization across microbatches with ignored boundary targets, and deferred loss synchronization.

## Repository layout

| Path | What it is |
|---|---|
| `bench100m.py` | Benchmark/training-step harness and all switches |
| `pinned_ngram.py` | Pinned-RAM n-gram lookup, async gradient return, CPU accumulation, sparse updates, parity check |
| `ram_ngram.py`, `cuda_ngram_accum.py` | Simpler RAM-table and GPU-accumulator variants with parity checks |
| `fast_flce.py` | FLA fused cross-entropy patch, with a full-size parity check |
| `optim800k.py` | Fused-norm swap and precision experiments |
| `prepare800k.py` | Builds the bounded, revision-pinned FineWeb fixture |
| `iter3_colab.sh` | Colab runner: `setup`, `parity`, `sweep`, `confirm`, `fastce`, `profile` |
| `run-selected-iter3.sh` | The recommended recipe |
| `results/results-iter3.jsonl` | Raw records for all nine iteration-3 runs, with stage timings |
| `docs/` | Findings from every iteration, oldest to newest: 100m → ram-2560 → batch-scaling → batch6 → 800k-7gb → iteration-3 |
| `tools/` | CPU-only sparse-update microbenchmark; local compile-region trace check |

## Requirements

This repository contains the harness and optimizations only. It does **not** include:

- the `a100stack` model source (`model.py`, `config.py`, `ngram.py`, …), which the harness imports, or
- the 60,800-token tokenizer (`good_tokenizer`).

Both are supplied separately as `a100stack.zip` and `good_tokenizer.zip`.

Tested environment: Google Colab, NVIDIA L4, Python 3.13, PyTorch 2.11.0+cu130 (keep Colab's build), `flash-linear-attention==0.5.2`, `fla-core==0.5.2`. `fast_flce.py` targets FLA 0.5.2's source and refuses to patch a version where its patterns are missing.

## Run

On a Colab L4 runtime, put `a100stack.zip` and `good_tokenizer.zip` in `/content` and this repository in `/content` as well. Then:

```bash
bash iter3_colab.sh setup        # install FLA, patch source, build and SHA-check the fixture
bash iter3_colab.sh parity       # n-gram correctness checks
bash run-selected-iter3.sh       # recommended recipe: one warmup + four timed updates
```

Each run prints a `RESULT` JSON line and appends it to `results-iter3.jsonl`, with throughput, memory peaks, stage timings and correctness flags. Add `--profile-kernels 4` for a post-timing CUDA kernel table.

## Scope and caveats

These are throughput measurements over five short updates, not a training run. No held-out evaluation, convergence study or multi-run variance analysis was performed. Timing excludes initialization, data preparation, compilation and warmup. Numbers from different Colab runtime allocations differ by a few percent; compare rows measured together.
