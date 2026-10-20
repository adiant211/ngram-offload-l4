# a100stack: 2,560 tokens, accumulation 16, CPU n-gram table

20 October 2026. Follow-up experiments in the existing [Colab notebook](https://colab.research.google.com/drive/1ATUMpru9oLbLjy7evcIO1DOoEDKiZbI-).

## Result

The requested RAM profile completed successfully at **10,895 tokens/sec**, with **3.51 GiB peak allocated GPU memory** and **4.55 GiB peak host-process RSS**. The full 499,998,720-parameter n-gram table and its row-wise optimizer state stayed on the CPU during training. The 99,879,672 dense parameters and their optimizer remained on the L4 GPU.

Against the matching 2,560-token GPU-table run, offloading saved **1.47 GiB of VRAM (29.5%)** and reduced throughput by **12.4%**. This is a lower-VRAM option. The earlier 8K microbatch configuration remains the faster tested recipe on this L4.

## Settings and matched comparison

The prior benchmark's `8192` setting was tokens per microbatch; packed documents were separately capped at 1,024 tokens. For this request, both microbatch tokens and maximum packed document length were set to **2,560**, and accumulation was increased to **16**. Activation checkpointing was already off in the previous best configuration; it was explicitly kept off in every new run.

All three runs used the same **40,960-token effective batch**, two warmup optimizer steps, and four timed optimizer steps. They processed the same bounded FineWeb sample in the same order: 81,920 warmup tokens and 163,840 timed tokens. The document and microbatch boundaries differ between the 8K and 2,560 profiles, so their training trajectories are not asserted to be identical.

| Profile | 8K GPU control | 2,560 GPU table | Requested RAM profile |
| --- | ---: | ---: | ---: |
| Tokens per microbatch | 8,192 | 2,560 | 2,560 |
| Maximum packed document tokens | 1,024 | 2,560 | 2,560 |
| Gradient accumulation | 5 | 16 | 16 |
| Effective batch tokens | 40,960 | 40,960 | 40,960 |
| Activation checkpointing | Off | Off | Off |
| N-gram table / sparse optimizer state | GPU / GPU | GPU / GPU | CPU / CPU |
| Tokens/sec | 28,210.59 | 12,432.61 | 10,895.03 |
| Measured time, seconds | 5.8077 | 13.1782 | 15.0380 |
| Median optimizer step, seconds | 1.4530 | 3.2967 | 3.7495 |
| Peak allocated GPU memory, GiB | 8.19 | 4.98 | 3.51 |
| Peak host-process RSS, GiB | 1.93 | 2.03 | 4.55 |
| Final training loss | 9.31060 | 9.31794 | 9.31842 |

Relative to the 8K control, the combined requested profile used about **57.1% less VRAM** and had about **61.4% lower throughput**. This comparison includes changes in both microbatch size and document cap. Moving only the n-gram table to RAM is isolated by the last two columns.

## Implementation and checks

`bench100m.py` now exposes `--seq-len`, `--ngram-device`, and `--check-ram`. `--tokens` still controls microbatch size. The existing GPU behavior remains the default. Checkpoint flags are set explicitly, and actual buffer placement and checkpoint status are recorded in each result.

The new `ram_ngram.py` keeps the full BF16 table in ordinary CPU RAM, along with FP32 row-wise RMSprop state. It hashes and gathers rows on the CPU, pins the gathered batch, and transfers only those rows to CUDA for the dense projection and backward pass. After each microbatch, sparse leaf gradients move back to RAM and GPU leaf references are released. CPU gradients accumulate until one sparse update per optimizer step. The table is trainable and is not frozen.

The table itself occupies **999,997,440 bytes** of CPU memory, plus approximately 3.9 MB of row state. Host RSS also includes the framework, staging buffers, accumulated sparse gradients, and temporary optimizer arrays. The table is initially constructed on the GPU and moved before training; the reported steady-state peaks exclude initialization. This implementation therefore still needs enough GPU memory for that initial allocation.

One targeted check compared CPU and GPU n-gram lookup, backward, and sparse updates after two accumulated microbatches. Lookups matched exactly; sparse table relative error was **0**, and row-wise optimizer state matched the test tolerance. The check touched 1,239 unique rows. In full training, CPU and GPU variants also reported the same mean touched-row count, **117,209.75** per timed optimizer step. Every run completed with finite loss, dense gradient norm, and sparse optimizer state.

CPU sparse updates averaged about **273 ms of host-call wall time per optimizer step**. Row gathering and gradient transfers add further work. The recorded gradient-collection wall time includes waits for CUDA backward completion and must not be interpreted as pure copy time.

## Reproduce

Use the same compatible Colab environment and bounded FineWeb data/tokenizer from the preceding experiment. Place the updated `bench100m.py` and `ram_ngram.py` together in `/content/a100stack`, then run:

```bash
cd /content/a100stack
python -u bench100m.py --tokens 2560 --seq-len 2560 --grad-accum 16 --steps 4 --warmup 2 --attn native --pack --compile mlp --ckpt none --ngram-device cpu --check-ram --output results-ram-2560.jsonl
```

Native variable-length attention, packed projections, regional MLP compilation, 128-dimensional GDN heads, BF16 autocast, fused AdamW, and sparse row-wise RMSprop were retained. Dense and n-gram parameter counts are unchanged.

Timing includes training, clipping, dense and sparse optimizer updates, and all CPU/GPU table transfers and synchronization within each step. It excludes sample streaming/tokenization, input staging, initialization, warmup, and compilation. Token batches are GPU resident; the n-gram table is CPU resident in the RAM case. These are short training-compute benchmarks, not live-stream throughput or quality evaluations.

## Saved evidence

The notebook contains the commands and outputs. Runtime files under `/content/a100stack` include:

- `results-ram-2560.jsonl`
- `ram-control-8192.log`
- `gpu-seq2560-acc16.log`
- `ram-seq2560-acc16.log`
- Updated `bench100m.py` and new `ram_ngram.py`
- `bench100m-before-ram.py`, preserving the prior benchmark

The verified runtime bundle is `/content/a100stack-ram-2560-experiment.zip`: **46 files, 1,359,160 bytes**. It includes the new `run-ram-2560.sh` runner, findings, source, exact results, logs, environment, bounded data, and tokenizer. The runner passed shell syntax validation and the ZIP passed its integrity check. Runtime files remain temporary until downloaded or copied elsewhere.

All training experiments have finished. No convergence run or checkpoint was produced. The original faster GPU profile remains available alongside the new RAM profile.
