# a100stack: device batch 6 × sequence length 1,024 × accumulation 128

Measured on the existing Colab NVIDIA L4, with the same 99,879,672 dense parameters and 499,998,720 n-gram parameters. The n-gram table and row optimizer state stayed on GPU. Gradient checkpointing was disabled. Packed native attention, packed projections, regional MLP compilation, GDN head dimension 128 and BF16 autocast remained enabled.

| Configuration | Grad accumulation | Tokens/update | Throughput | Peak allocated VRAM | Median update |
|---|---:|---:|---:|---:|---:|
| Earlier GPU-table best: 8 × 1,024 | 5 | 40,960 | 28,211 tokens/s | 8.19 GiB | 1.45 s |
| Requested: 6 × 1,024 | 128 | 786,432 | 26,257 tokens/s | 8.60 GiB | 29.95 s |

The requested configuration completed, but was 6.9% slower than the earlier best. This comparison changes device batch, accumulation and sparse-gradient implementation together; it does not isolate accumulation's effect. Packed batches contain variable-length document segments, so the displayed batch × sequence dimensions are equivalent token-budget shapes with the stated context cap.

## Required script correction

The original implementation failed during the first sparse update with CUDA out of memory, attempting an additional 6.00 GiB concatenation after retaining lookup embeddings and gradients across 128 backward passes. It had 14.63 GiB allocated and only 4.14 GiB free at the failed allocation.

`cuda_ngram_accum.py` now collects gradients after every backward pass into a fixed FP32 row accumulator (about 1.86 GiB), records touched rows, and frees pending lookup leaves. Once per optimizer update it applies the original row-wise RMSprop rule in bounded row chunks and clears the accumulator. The table still updates only after all 128 accumulation steps. This single-GPU helper explicitly rejects distributed execution.

`bench100m.py --cuda-accum` opts into this behavior; existing defaults are preserved. A targeted test compared lookup outputs, touched rows, BF16 table values and FP32 row optimizer state over two optimizer updates, each accumulating two backward passes. It also checked that the accumulator clears between updates. The test passed. The full requested run then completed with finite losses, dense gradients and sparse optimizer state.

## Measurement details

- Two warmup updates followed by four timed updates: 3,145,728 timed tokens in 119.805352832 seconds.
- Exact throughput: 26,256.99040685735 tokens/s; median update: 29,950.263671875 ms.
- Forward/backward, accumulation and clipping: mean 29,889.31689453125 ms/update.
- Dense optimizer: mean 12.08243203163147 ms/update; sparse optimizer: mean 49.678592681884766 ms/update.
- Peak allocated: 8.600389957427979 GiB; peak reserved: 9.0390625 GiB; host peak RSS: 1.9943199157714844 GiB.
- Average touched n-gram rows: 498,475 per measured update.
- First/final measured losses: 9.399976387619972 / 6.968680631369352. These are short repeated-sample training losses, not validation quality measurements.
- Same bounded, pre-staged FineWeb English sample and supplied tokenizer as previous experiments. The sample is smaller than one effective update and is reused. No additional corpus download.
- Timing includes real forward/backward, dense clipping/optimizer updates, row accumulation and n-gram updates. It excludes data preparation, initialization, compilation and warmup. No convergence claim or independent sequence-length optimization.

## Reproduce

Run in `/content/a100stack` using the existing Colab environment and sample:

```bash
python -u bench100m.py --tokens 6144 --seq-len 1024 --grad-accum 128 --ngram-device cuda --cuda-accum --attn native --pack --compile mlp --ckpt none --gdn-head-dim 128 --warmup 2 --steps 4 --output results-batch6-seq1024-acc128-fixed.jsonl
```

Runtime evidence: `batch6-seq1024-acc128.log` (original failure), `batch6-seq1024-acc128-fixed.log` (successful retry), `results-batch6-seq1024-acc128-fixed.jsonl`, and the preserved `bench100m-before-cuda-accum.py`. The Colab runtime is temporary; corrected helper and benchmark copies are also saved alongside this local report.
