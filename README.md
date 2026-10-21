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

### Iteration 4: Dion3, multi-GPU, layer dropout

| Same recipe, 786k-token updates | Tokens/s | GPU memory |
|---|---:|---:|
| AdamW | 29,180 | 6.51 / 6.92 GiB |
| **Dion3** (`--optimizer dion3 --pack-mode concat`) | 28,826 | 6.26 / 6.57 GiB |
| Dion3 + layer dropout, 10% mean drop rate | 30,740 | 6.26 / 6.60 GiB |
| Dion3 + layer dropout, 20% mean drop rate | 33,409 | 6.26 / 6.61 GiB |
| Dion3 + Depth-Attention | 28,602 | 6.28 / 6.72 GiB |

- **Dion3** (microsoft/dion NorDion2) handles the 2-D matrices; AdamW inside Dion3 handles embeddings, norms and scalars.
- **Multi-GPU:** launch with `torchrun` to get DDP, a Dion3 process group, and cross-rank reduction of the n-gram gradients.
- **Opt-in paper techniques:** Depth-Attention (arXiv:2606.05014) and layer dropout (arXiv:2609.05275).

Details, equivalence tests and the quality A/B are in [docs/a100stack-iteration-4-findings.md](docs/a100stack-iteration-4-findings.md).

### Iteration 5: 700M dense + 1.2B n-gram table on one A100, recoverable

32 layers (24 Gated DeltaNet + 8 gated attention), width 1,280, 716.0M dense parameters, plus a
1.2B-parameter BF16 n-gram table in pinned host RAM. A100-SXM4-80GB, ~786k-token updates:

| Microbatch × accumulation | Tokens/s | MFU | GPU memory (`nvidia-smi`) |
|---|---:|---:|---:|
| 10,240 × 77 (34 GB ceiling) | 26,379 | 37.3% | 30.6 GiB |
| **16,384 × 48 (default, ~60 GB ceiling)** | **27,504** | 38.9% | **42.7 GiB** |
| 20,480 × 38 | 27,502 | 38.9% | 51.2 GiB |
| 24,576 × 32 | 28,145 | 39.8% | 58.1 GiB |
| 16,384 × 48 + layer dropout (p_max 0.2) | 29,858 | — | 42.1 GiB |
| 10,240 × 77 + Delta Block (opt-in) | 22,637 | — | 30.8 GiB |

- `trainer.py` is a recoverable training loop:
  - deterministic resumable data and a WSD schedule;
  - atomic background checkpoints (8.8 GB, 32 s) covering weights, table, optimizer and RNG state, with checksums re-verified on resume;
  - graceful stop on SIGTERM and skipping of nonfinite updates.
- `fla_nosync.py` removes five CPU–GPU syncs per microbatch inside FLA (+6.5%); a kernel profile showed custom kernels would add only 1–5% each.
- `train700m.sh` restarts it from the newest complete checkpoint after a crash.
- `recovery_test.sh` kills runs mid-update and mid-checkpoint and checks the resumed runs.

Throughput plateaus at 16k-token microbatches; 24k buys 2.3% for 15 GiB more. Layer dropout (p_max 0.2) adds 8.6%. Delta Block attention residuals (`--delta-block`, ported from Luo, Cai and Hu's published code) are available but cost 14% throughput in this stack. A supervised run was killed with `kill -9` and resumed from its checkpoint automatically, with matching losses. See [docs/a100stack-iteration-5-findings.md](docs/a100stack-iteration-5-findings.md).

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
| `dion_optim.py` | Dion3 parameter grouping (single GPU or DDP process group) and concat-at-forward projections |
| `depth_attention.py` | Depth-Attention value mixing across the gated-attention layers (`--depth-attn`) |
| `layer_dropout.py` | Layer dropout with depth/time schedules and early-exit evaluation (`--layer-drop`, `--eval-exit-at`) |
| `prepare_ab.py` | Disjoint FineWeb train/held-out fixtures for quality A/B runs |
| `run.py` | Hardware-probed launcher (tuned for 2x CMP 170HX: slow PCIe, no NVLink); supervised torchrun with auto-resume |
| `trainer.py` | Recoverable training loop (700m/500m/100m presets): streaming data, WSD schedule, atomic checkpoints, resume, signals |
| `train700m.sh` | Supervised 700M + 1.2B run under 34 GB GPU memory; restarts from the last checkpoint |
| `recovery_test.sh` | Crash/resume/SIGTERM test at full model size |
| `a100_sweep.sh` | A100 microbatch/memory/layer-dropout throughput sweep |
| `delta_block.py` | Delta Block attention residuals (opt-in `--delta-block`), fused into the compiled regions, with an FP32 equivalence check against the published code |
| `fla_nosync.py` | Removes FLA 0.5.2's per-microbatch CPU-GPU syncs (chunk tables, CE backward), with a parity check |
| `prepare_fineweb.py` | Batched FineWeb tokenization into disjoint train/held-out files (300M tokens in ~5 min) |
| `optim800k.py` | Fused-norm swap and precision experiments |
| `prepare800k.py` | Builds the bounded, revision-pinned FineWeb fixture |
| `iter3_colab.sh` | Colab runner: `setup`, `parity`, `sweep`, `confirm`, `fastce`, `dion`, `dist`, `layerdrop`, `depthab`, `profile` |
| `run-selected-iter3.sh` | The recommended recipe |
| `results/results-iter5.jsonl` | A100 sweep, kernel profiles, sync sites, recovery test and supervised-run records |
| `results/results-iter3.jsonl` | Raw records for all nine iteration-3 runs, with stage timings |
| `docs/` | Findings from every iteration, oldest to newest: 100m → ram-2560 → batch-scaling → batch6 → 800k-7gb → iteration-3 |
| `tools/` | CPU-only sparse-update microbenchmark; local checks for the compile region, Dion3 wiring, Depth-Attention and layer dropout |

## Requirements

This repository contains the harness and optimizations only. It does **not** include:

- the `a100stack` model source (`model.py`, `config.py`, `ngram.py`, …), which the harness imports, or
- the 60,800-token tokenizer (`good_tokenizer`).

Both are supplied separately as `a100stack.zip` and `good_tokenizer.zip`.

Tested environment: Google Colab, NVIDIA L4, Python 3.13, PyTorch 2.11.0+cu130 (keep Colab's build), `flash-linear-attention==0.5.2`, `fla-core==0.5.2`. For `--optimizer dion3`, install `dion` from `git+https://github.com/microsoft/dion.git@58d38ad`. `fast_flce.py` targets FLA 0.5.2's source and refuses to patch a version where its patterns are missing.

Multi-GPU: `torchrun --nproc_per_node N bench100m.py ... --ram-mode staged --pack-mode concat --optimizer dion3`. `--dist-backend gloo` lets several test ranks share one GPU. Each rank keeps its own pinned host copy of the n-gram table and accumulator, about 3 GB of host RAM per rank.

## Run

On a Colab L4 runtime, put `a100stack.zip` and `good_tokenizer.zip` in `/content` and this repository in `/content` as well. Then:

```bash
bash iter3_colab.sh setup        # install FLA, patch source, build and SHA-check the fixture
bash iter3_colab.sh parity       # n-gram correctness checks
bash run-selected-iter3.sh       # recommended recipe: one warmup + four timed updates
```

For the 700M + 1.2B training run on an A100 (after `setup` and `pip install "git+https://github.com/microsoft/dion.git@58d38ad"`):

```bash
cd /content/a100stack && python -u prepare_fineweb.py --train-tokens 300000000
bash train700m.sh                      # resumes automatically; pkill -TERM -f trainer.py stops cleanly
CKPT_DIR=/content/drive/MyDrive/ckpt700m CACHE_DIR=/content/drive/MyDrive/cache700m bash train700m.sh  # survives runtime loss
```

On your own multi-GPU host (e.g. 2x CMP 170HX), from the a100stack source directory with this repository's files:

```bash
python run.py --probe-only   # measure the GPUs and the host link, print the plan and a time estimate
python run.py                # supervised torchrun run with the chosen settings; resumes after a crash
```

Each benchmark run prints a `RESULT` JSON line and appends it to `results-iter3.jsonl`, with throughput, memory peaks, stage timings and correctness flags. Add `--profile-kernels 4` for a post-timing CUDA kernel table.

## Scope and caveats

These are throughput measurements over five short updates, not a training run. No held-out evaluation, convergence study or multi-run variance analysis was performed. Timing excludes initialization, data preparation, compilation and warmup. Numbers from different Colab runtime allocations differ by a few percent; compare rows measured together.
