# a100stack: larger GPU batches at a 2,560-token context cap

20 October 2026. Experiments in the existing [Colab notebook](https://colab.research.google.com/drive/1ATUMpru9oLbLjy7evcIO1DOoEDKiZbI-).

## Answer and result

**Longer sequences were not needed.** Keeping the maximum packed document length at **2,560**, doubling GPU microbatch tokens from **2,560 to 5,120** increased throughput from **10,895 to 16,673 tokens/sec**, approximately **53% faster**.

The best tested RAM-table configuration was **10,240 GPU microbatch tokens, accumulation 4**: **20,009 tokens/sec**, with **8.44 GiB peak allocated VRAM**. That is approximately **84% faster** than the original 2,560/16 RAM profile. A further doubling to 20,480 tokens used substantially more VRAM and was slightly slower, so filling GPU memory did not increase throughput indefinitely.

Checkpointing was already disabled in all these experiments, including the original 2,560-token profile. These results measure the effect of GPU microbatch scaling, not a benefit from disabling checkpointing.

## Three distinct knobs

- **Sequence/context cap:** how long each packed document segment can be. It remained 2,560 throughout.
- **GPU microbatch:** the total number of packed tokens processed in one forward/backward pass. Increasing this supplies more parallel work without extending any document's context.
- **Gradient accumulation:** how many microbatches contribute to one optimizer update. It builds the effective batch but does not make an individual microbatch more parallel. Accumulation was halved at each doubling to keep the effective batch at 40,960 tokens.

In general, activation checkpointing saves memory by recomputing activations during backward. Disabling it can avoid that recomputation when memory permits. It does not require a longer context; memory can instead be used for a larger microbatch. This task did not include a checkpointing-on comparison.

## Measured scaling

All rows use the same 99,879,672 dense parameters on the L4 GPU and 499,998,720 trainable n-gram parameters in CPU RAM. The table and its row-wise optimizer state stayed on the CPU; dense projections and dense optimization stayed on the GPU. Maximum document length was 2,560 and activation checkpointing was off in every row.

| GPU microbatch tokens | Accumulation | Effective batch tokens | Tokens/sec | Peak allocated VRAM, GiB | Peak host-process RSS, GiB | Measured time, seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,560 | 16 | 40,960 | 10,895 | 3.51 | 4.55 | 15.0380 |
| 5,120 | 8 | 40,960 | 16,673 | 5.14 | 4.09 | 9.8265 |
| **10,240** | **4** | **40,960** | **20,009** | **8.44** | **4.10** | **8.1882** |
| 20,480 | 2 | 40,960 | 19,321 | 14.94 | 4.20 | 8.4801 |

The 20,480-token case used about 6.51 GiB more VRAM than 10,240 and measured about 3.4% lower throughput. These short runs indicate a plateau near 19–20K tokens/sec among the larger tested batches. They do not establish a global optimum or directly measure peak SM utilization.

Every row processes 81,920 warmup tokens followed by 163,840 timed tokens: two warmup optimizer steps and four measured optimizer steps. The 2,560-token baseline is the completed RAM-table trial from the preceding comparison; the three larger cases are new trials. Final training losses were approximately 9.31842, 9.31154, 9.29852, and 9.31831 respectively. All completed with finite losses, dense gradient norms, and sparse optimizer state. Changing microbatch boundaries can change context resets, so exact training-trajectory equivalence is not claimed.

## Selected recipe

```bash
cd /content/a100stack
python -u bench100m.py --tokens 10240 --seq-len 2560 --grad-accum 4 --steps 4 --warmup 2 --attn native --pack --compile mlp --ckpt none --ngram-device cpu --output results-ram-batch-scaling.jsonl
```

The existing source already supported these settings; no model or optimizer algorithm changes were needed. Native variable-length attention, packed projections, regional MLP compilation, 128-dimensional GDN heads, BF16 autocast, fused AdamW, and sparse row-wise RMSprop were retained. No repeated parity tests were added; the training finite checks remained enabled and the CPU/GPU sparse parity check from the preceding task had already passed.

## Interpretation and limits

The GPU can process more independent packed tokens while each document stays within its context cap. Larger batches improved throughput strongly up to 10,240 tokens here. The 20,480-token result shows why unused VRAM alone is not evidence that a larger batch will run faster.

All these scaling tests retained CPU n-gram offload. They are not a comparison against a GPU-resident table at each larger microbatch. At the selected 10,240 setting, the recorded CPU sparse-update host-call time was about 290 ms per optimizer step; the optimizer and transfers remain part of the workload.

Timing includes forward/backward, clipping, dense/sparse optimization, and n-gram CPU/GPU transfers and synchronization. It excludes initialization, input staging, dataset streaming/tokenization, warmup, compilation, and evaluation. FineWeb token batches are GPU resident; the n-gram table is CPU resident. The measurements are short training-compute benchmarks, not a live-stream rate, convergence study, or quality evaluation.

## Evidence

The notebook contains all commands and outputs. Runtime files under `/content/a100stack` include `results-ram-batch-scaling.jsonl`, `ram-batch5120-acc8.log`, `ram-batch10240-acc4.log`, and `ram-batch20480-acc2.log`. The baseline remains in `results-ram-2560.jsonl` and `ram-seq2560-acc16.log`. All training experiments have finished; no long training job or checkpoint was produced.

The selected configuration is saved in `selected-ram-batch-profile.json` and `run-ram-batch-scaling.sh`. The runner passed shell syntax validation. The verified runtime bundle, `/content/a100stack-ram-batch-scaling-experiment.zip`, contains **53 files, 1,368,530 bytes**, including source, results, logs, findings, bounded sample data, tokenizer, and environment metadata. It passed its integrity check. Runtime files remain temporary until downloaded or copied elsewhere.
