# a100stack: 700k–800k tokens per update near 7 GiB

Recommended configuration near the 7 GiB memory target on the Colab NVIDIA L4: **device batch 8 × sequence length 1,024 × gradient accumulation 96 = 786,432 tokens/update**, a pinned RAM n-gram table, asynchronous transfer buffers, incremental CPU gradient aggregation, fused normalization outside the compiled MLP, and eight fused-loss chunks. Its fresh-data check reached **26,229 tokens/sec at 6.67 GiB allocated / 7.20 GiB reserved peak GPU memory**. The faster four-chunk alternative reached **26,688 tokens/sec at 7.07 GiB allocated / 7.60 GiB reserved** over four timed updates. The default recipe chooses the lower reserved-memory footprint; the shorter eight-chunk confirmation is less statistically precise.

The model remains 99,879,672 dense parameters plus 499,998,720 n-gram parameters. Dense weights, dense optimizer state and residuals remain FP32 under BF16 autocast; the n-gram table remains BF16. Dense fused AdamW and the original row-wise sparse RMSprop rules are unchanged. Activation checkpointing remains disabled.

## Fresh-data confirmations

| Device batch × seq_len | Accumulation | Tokens/update | Loss chunks | Tokens/sec | Allocated peak | Reserved peak | Median update |
|---|---:|---:|---:|---:|---:|---:|---:|
| **8 × 1,024** | **96** | **786,432** | **4** | **26,688** | **7.07 GiB** | **7.60 GiB** | **29.41 s** |
| 8 × 1,024 | 96 | 786,432 | 8 | 26,229 | 6.67 GiB | 7.20 GiB | 29.98 s |

The four-chunk result averaged four measured updates after one warmup: 3,145,728 timed tokens in 117.871145091 seconds. At least 785,309 tokens/update contributed to the loss after ignored boundary targets. The eight-chunk confirmation measured one update after one warmup, so the durations differ and the 1.7% speed difference has limited precision. Use eight chunks for additional memory headroom.

Both used the supplied tokenizer and a bounded 4,194,305-token FineWeb English fixture, large enough that neither confirmation reused fixture batches. Its first update touched 815,999 n-gram rows; the longer confirmation averaged 816,227 touched rows/update. This is heavier than the earlier repeated 262k-token fixture. The table and its row optimizer state stayed in RAM; the table's pinned status was checked. Host peak RSS for the longer confirmation was 5.09 GiB.

Memory figures are PyTorch allocated/reserved peaks during measured training, rather than total device usage reported by the driver. Input batches are pre-staged on GPU. Displayed batch shapes are packed-token equivalents; actual document segments vary in length and retain their boundaries.

## Controlled exploratory comparisons

These trials used the earlier 262,145-token fixture, which is reused within an effective batch. Keep their speed comparisons separate from the larger-fixture confirmations above.

| Placement / optimization | Device batch × seq_len | Accumulation | Tokens/update | Tokens/sec | Allocated peak |
|---|---:|---:|---:|---:|---:|
| Original RAM table | 8 × 1,024 | 96 | 786,432 | 20,950 | 6.93 GiB |
| Pin the full RAM table only | 8 × 1,024 | 96 | 786,432 | 21,032 | 6.93 GiB |
| Reusable pinned buffers + async gradient copies + CPU accumulator | 8 × 1,024 | 96 | 786,432 | 25,648 | 6.87 GiB |
| Also reuse CPU input IDs/positions | 8 × 1,024 | 96 | 786,432 | 26,431 | 6.87 GiB |
| Also fuse norms outside the compiled MLP | 8 × 1,024 | 96 | 786,432 | 26,977 | 6.69 GiB |
| Also use four loss chunks | 8 × 1,024 | 96 | 786,432 | 27,016 | 7.02 GiB |
| Pinned pipeline + fused norms, longer context | 4 × 2,048 | 96 | 786,432 | 26,547 | 6.67 GiB |
| Pinned pipeline + fused norms, longer context | 3 × 2,560 | 100 | 768,000 | 26,009 | 6.42 GiB |
| GPU table + asynchronous RAM gradient accumulator | 6 × 1,024 | 128 | 786,432 | 25,750 | 6.68 GiB |
| GPU table + RAM gradient accumulator, larger device workload | 6 × 1,152 | 112 | 774,144 | 26,012 | 7.12 GiB |
| GPU table + GPU gradient accumulator, smaller device batch | 4 × 1,024 | 192 | 786,432 | 18,712 | 7.19 GiB |

Pinning the full table alone changed speed by only 0.4%, within short-run variability. The pinned pipeline plus fused norms improved matched-fixture throughput by about 28.8%. The architecture and optimizers stayed fixed. Most trials used one warmup plus two measured updates; longer-context, loss-chunk and last GPU-table shape checks used one measured update. The original RAM control predates token-weighted loss normalization and deferred loss synchronization; later trials include both corrections.

## What was the bottleneck?

A separate, deliberately synchronized diagnostic measured the original pin-only path at 8 × 1,024 × 96. These are diagnostic stage times per update, not a throughput-winning configuration:

| Stage | Time/update |
|---|---:|
| Gradient GPU-to-CPU copy into ordinary RAM, including allocation/staging | 4.267 s |
| CPU gradient filtering | 1.144 s |
| CPU gradient concatenation | 1.152 s |
| CPU unique-row indexing | 0.094 s |
| CPU gradient reduction | 0.559 s |
| CPU row-wise RMSprop update | 0.716 s |
| CPU table row gather | 0.634 s |
| Copy gathered rows into pinned staging storage | 0.324 s |
| Gathered-row transfer to GPU, CUDA event time | 0.269 s |
| CPU n-gram hashing | 0.163 s |
| Waiting for queued backward work before gradient transfer | 4.443 s |

The principal offload costs were returning gradients through pageable memory and repeatedly allocating/filtering/concatenating large gradient tensors. Fetching table rows to GPU was much smaller. The backward-wait figure is queued model computation, not pure PCIe transfer time.

Pinning the source table does not automatically pin the separately gathered output tensor. The optimized implementation gathers directly into reusable pinned buffers, copies gradients into pinned buffers on a separate CUDA stream, and lets a CPU worker aggregate them while the GPU continues. Two buffer slots enforce completion before reuse. CPU inputs are retained from batch preparation to avoid a synchronizing GPU-to-CPU round trip for hashing. Hash computation itself is still timed.

A fixed FP32 CPU row accumulator replaces the growing gradient list and large end-of-step concatenation. Valid touched rows are tracked separately. Updates occur once after all 96 accumulation passes; row updates are processed in bounded chunks. The original RMSprop rule and learning rate are preserved.

On the larger-fixture confirmation, gradient DMA itself measured 0.572 seconds/update and row H2D transfer 0.319 seconds/update. CPU incremental reduction took 0.621 seconds and overlapped GPU work; the final worker join took only 0.00063 seconds. Those overlapped figures must not be added as independent wall-time costs. The end-of-step CPU update/clear took about 1.585 seconds; forward/backward and clipping occupied about 27.878 seconds of the roughly 29.47-second average step. Model forward/backward is now the dominant phase. This measurement does not identify which individual model kernel dominates.

## Documentation review

- **FlashQLA:** its supported hardware starts at SM90 and includes SM100/103/120/121. This L4 is SM89, so FlashQLA is not a supported backend for this runtime. It was not installed. [Official requirements](https://github.com/QwenLM/FlashQLA#installation).
- **Transformers:** regional compilation was already enabled. Its fused-kernel guidance identified normalization as an additional candidate; existing FLA kernels provided the implementation without installing another training framework. FLA normalization inside the fullgraph-compiled MLP failed to trace, so that norm stays in Torch's compiled path. Norms outside that region use the fused implementation. [Compilation guide](https://huggingface.co/docs/transformers/main/torch_compile), [training kernels](https://huggingface.co/docs/transformers/v5.19.0/kernels).
- **TRL:** packing/padding-free execution and BF16 were already present. Its asynchronous offload guidance motivated checking the data-transfer path. This custom n-gram implementation uses explicit pinned buffers and streams; it does not depend on a TRL Trainer flag. [Memory guidance](https://huggingface.co/docs/trl/reducing_memory_usage), [speed guidance](https://huggingface.co/docs/trl/speeding_up_training).
- **Unsloth:** fused, chunked linear cross-entropy was already active. Four versus eight chunks was tested; the short-fixture difference was negligible, while the fresh-data check favored four chunks by about 1.7% at higher memory. Correct token-weighted accumulation was added for ignored boundary targets. No LoRA or quantization was introduced. [Fused/chunked loss and offloading](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning), [accumulation normalization](https://unsloth.ai/blog/gradient).

## Corrections and verification

- `bench100m.py`: optional RAM modes and host accumulation, token-weighted loss normalization, deferred per-batch scalar synchronization, fused-normalization/precision options, tunable loss chunks, explicit fixture metadata, and a flat FLA 0.5.2 fused-loss adapter that also works with the original source ZIP.
- `pinned_ngram.py`: pinned table and transfer buffers, separate copy stream, CPU aggregation worker, safe repeated buffer reuse, bounded row updates, and stage profiling. Both RAM-table and GPU-table/RAM-gradient modes passed a targeted comparison against the original implementation over two optimizer updates with four accumulated backward passes each, including buffer reuse and accumulator clearing.
- `optim800k.py`: optional fused norms and BF16 residual experiment. BF16 residuals were rejected: loss stayed close but embedding gradients differed by about 20.8%. The fused-norm-only comparison had relative loss error 0.0000138 and embedding-gradient error 0.0362 under mixed precision. This is a numerical comparison, not proof of unchanged convergence.
- All completed throughput trials checked finite losses, dense gradients and sparse optimizer state. No held-out quality or long-run convergence study was performed. No persistent training run or model checkpoint was started.

Timing includes forward/backward, clipping, transfers and waits used during training, CPU aggregation and both optimizer updates. It excludes initialization, data preparation/staging, compilation and warmup. First-use compilation was substantial for some configurations.

## Reproduce and evidence

Use the existing Colab Torch 2.11.0+cu130 environment with FLA/fla-core 0.5.2. Put the benchmark and helpers alongside the supplied a100stack source files. The notebook contains `run-selected-800k.sh`, `selected800k.json`, all trial logs, `results-800k-7gb.jsonl`, the input manifests, and the complete experiment archive. Runtime files are temporary; the corrected scripts and this report are also saved locally in this output directory. The local benchmark files still require the supplied a100stack source and tokenizer.

The recommended command uses eight chunks for reserved-memory headroom. The script runs four timed updates; its recorded eight-chunk check used one timed update. Change the loss-chunk argument to four for the faster, higher-memory profile:

```bash
python -u bench100m.py --tokens 8192 --seq-len 1024 --grad-accum 96 --ngram-device cpu --ram-mode staged --attn native --pack --compile mlp --ckpt none --fused-norms --loss-chunks 8 --token-weighted --defer-loss-sync --data fineweb-800k-confirm.bin --warmup 1 --steps 4 --output results-800k-7gb.jsonl
```

Generate the larger bounded fixture with `python prepare800k.py`. Dataset: HuggingFaceFW/fineweb, sample-10BT, train, streaming, revision `9bb295ddab0e05d785b879661af7260fed5140fc`. The fixture contains 4,194,305 uint16 tokens from 5,829 documents and occupies 8,388,610 bytes. Its SHA-256 is `a5ff5040206ad44e68c317cc286124a1d44c86502594951bed0ac703223c1f7f`; tokenizer SHA-256 is `421f32bf48a5d4903aeb72de414c6ab6c30e94f8a749eea7b72221e602427d49`. EOS is 3 and automatic special-token insertion is disabled.
