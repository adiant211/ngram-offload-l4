# Measured results

Hardware: Colab NVIDIA A100-SXM4-80GB, 12 vCPUs, 167 GB RAM, PyTorch 2.11.0+cu130, FLA 0.5.2,
Dion3 at `microsoft/dion@58d38ad`. Data: real FineWeb. Throughput: 6 timed updates after 2 untimed ones,
including the CPU table step and the optimizer.

Raw records are in `docs/results-iteration-5.jsonl`; the full write-up is in
`docs/findings-iteration-5.md`.

## Throughput and GPU memory (700m preset, ~786k-token updates)

| Setup | Tokens/s | Hours per 1B tokens | GPU memory (`nvidia-smi`) |
|---|---:|---:|---:|
| 10k microbatches | 26,379 | 10.5 | 30.6 GiB |
| **16k microbatches (default)** | **27,504** | **10.1** | **42.7 GiB** |
| 20k microbatches | 27,502 | 10.1 | 51.2 GiB |
| 24k microbatches | 28,145 | 9.9 | 58.1 GiB |
| 16k + layer dropout 0.2 (constant) | 29,858 | 9.3 | 42.1 GiB |
| 10k + Delta Block | 22,637 | 12.3 | 30.8 GiB |

At the default speed the full 10B-token schedule takes about 4.2 days on one A100-80GB.

**Where an update's time goes** (16k, 786k tokens, 28.6 s):

| Phase | Time |
|---|---:|
| GPU forward + backward | ~27.0 s |
| CPU update of the 1.2B table | 1.4 s |
| Dion3 step | 0.2 s |

**GPU kernel time per 10k-token microbatch** (367 ms):

| Category | Share | Note |
|---|---:|---|
| Matrix multiplies | 56% | ~68% of A100 BF16 peak for these shapes |
| DeltaNet kernels (FLA) | 17% | |
| Elementwise | 11% | Gradient accumulation, weight casts |
| Compiled regions | 8% | |
| Loss, norms, copies | 7% | |

After removing FLA's CPU–GPU syncs, the GPU is busy 98% of the time.

## Recoverability

| Test (full 700M + 1.2B model) | Outcome |
|---|---|
| Hard kill while a checkpoint was being written | Half-written checkpoint ignored; resumed from the previous one, checksums verified |
| Hard kill in the middle of an update | Resumed from the newest complete checkpoint, checksums verified |
| SIGTERM | Finished the update, wrote a checkpoint, exit 0 |
| 40-min supervised run, `kill -9` at 22 min | Supervisor restarted it; resumed from update 35. The repeated updates matched: loss 11.47899 / 11.47947 vs 11.47900 / 11.47957 before the kill. Back to training ~80 s after the kill (warm compile cache) |

**Checkpoint costs:**

| | |
|---|---:|
| Size | 8.8 GB |
| Training pause | 4.6–4.9 s |
| Background write | ~35 s |

Losses are not bit-identical between runs: GPU kernels make two uninterrupted runs differ by up to 0.1%
early in training. The saved state itself is restored exactly; checksums are re-verified on every
resume.

## Tried and not adopted

| Idea | Result |
|---|---|
| Bigger microbatches than 16k | 20k: no gain. 24k: +2.3% for 15 GiB more |
| BF16 residual stream | No memory saved: the n-gram injection returns the stream to FP32 |
| Custom kernels (DeltaNet short convolution, FP32-accumulating GEMMs, etc.) | Profiled at 1–5% each; not worth the work |
| Delta Block attention residuals | −14% throughput here (the unavoidable cost is ~3.5%); kept opt-in |
| Depth-Attention | −0.8% throughput; gain not confirmed beyond seed noise (100M A/B) |
| FP8 | No FP8 on A100. Estimated +10–20% on H100/H200 for this model size |
| Two GPUs over slow PCIe (CMP 170HX) | Code paths exist (`--ngram-sync gloo`, `--dion-replicated`, …); single GPU is simpler at this size |

## Rough projections (not measured)

- **One H200 with the table on the GPU:** ~70–80k tokens/s.
- **A100 40GB with the 10k profile:** ~24–26k tokens/s.
