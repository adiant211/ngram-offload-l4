# a100stack 100M + 500M n-gram throughput

20 October 2026. Notebook: [a100stack 100M + 500M n-gram — L4 throughput](https://colab.research.google.com/drive/1ATUMpru9oLbLjy7evcIO1DOoEDKiZbI-).

## Confirmed result

At the same **32,768-token effective batch**, measured training throughput increased from **8,142 to 27,627 tokens/second — 3.39× faster** on one NVIDIA L4. The optimized run peaked at **8.06 GiB** of PyTorch allocated GPU memory. Both configurations completed with finite losses and gradients. All training experiments have finished; no long training job was launched.

| Matched comparison | Baseline | Selected recipe |
| --- | ---: | ---: |
| Microbatch tokens | 2,048 | 8,192 |
| Gradient accumulation | 16 | 4 |
| Effective batch tokens | 32,768 | 32,768 |
| Attention | Per-document SDPA loop | Native variable-length attention |
| Packed projections | No | Yes |
| Regional MLP compilation | No | Yes |
| GDN head dimension | 128 | 128 |
| Tokens/sec | 8,142.48 | 27,627.14 |
| Measured tokens | 196,608 | 196,608 |
| Measured wall time, seconds | 24.1460 | 7.1165 |
| Peak allocated GPU memory, GiB | 4.95 | 8.06 |
| Final training loss | 9.25996 | 9.26152 |

These are **training-compute measurements with GPU-resident batches**. Timing includes forward/backward, gradient clipping, dense optimizer updates, and sparse n-gram updates. It excludes streaming/tokenization, GPU batch staging, warmup, compilation, and evaluation. The measurements do not establish an end-to-end live-stream rate or long-run model quality. Microbatch boundaries can alter context resets, so the two training trajectories are not asserted to be identical.

## Model, data, and runtime

- **99,879,672 dense parameters + 499,998,720 n-gram parameters**, totaling 599,878,392. The entire table is allocated and sparse updates are applied each optimizer step; it is not frozen.
- 20 layers: 15 Gated DeltaNet and 5 gated attention layers; width 512; FFN width 1,408; tied embedding/output head; vocabulary 60,800; maximum packed document length 1,024. Attention uses 4 query heads and 1 KV head, each width 128. GDN uses 4 heads of width 128. The n-gram configuration has four width-512 hash tables, injected before block index 1.
- NVIDIA L4 with 22.03 GiB visible VRAM; Python 3.13.15; existing PyTorch 2.11.0+cu130; CUDA 13.0; Triton 3.6.0. Installed `flash-linear-attention==0.5.2` and `fla-core==0.5.2`. No Torch replacement or FlashAttention source build was needed.
- BF16 autocast with FP32 dense parameters; BF16 n-gram table; FP32 sparse row state. Dense optimizer: fused AdamW, LR 0.0003, betas (0.9, 0.95), weight decay 0.01, global dense gradient clipping 1.0. Sparse table uses the supplied row-wise RMSprop settings: LR 0.02, beta2 0.99, epsilon 1e-8.
- FineWeb English, `HuggingFaceFW/fineweb`, `sample-10BT`, streamed using the supplied `good_tokenizer.zip`. EOS ID 3. Only a bounded **262,145-token uint16 sample**, 524,290 bytes, from 349 source documents was retained. Documents are packed with boundary metadata and position resets; targets after EOS inputs are masked. Benchmarks use real tokens, not random inputs.
- Seed 1234. Final matched runs each use two warmup optimizer steps, followed by six timed optimizer steps. Warmup processes 65,536 tokens; timing processes 196,608 tokens. No held-out evaluation was performed. The remote dataset revision was not pinned; sample and tokenizer hashes are saved in the bundle's manifest.

## What improved throughput

The main improvements were native packed attention and a larger microbatch. Projection packing gave a further useful gain and reduced memory. MLP compilation added a small steady-state improvement, with its initial compilation cost excluded from the timing.

The attention path uses [PyTorch's variable-length attention API](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html), with the actual 2.11 runtime signature inspected before use. The original per-document SDPA loop remains the measured baseline.

| Exploratory run, accumulation 1 | Micro tokens | Tokens/sec | Peak GiB |
| --- | ---: | ---: | ---: |
| Original attention loop | 2,048 | 6,903 | 4.14 |
| Native packed attention | 2,048 | 8,531 | 3.95 |
| Native packed attention | 8,192 | 24,426 | 8.50 |
| Native packed attention | 16,384 | 23,759 | 14.60 |
| Native attention + packed projections | 8,192 | 26,073 | 7.87 |
| Above + regional MLP compilation | 8,192 | 26,624 | 7.43 |
| Above + 64-dimensional GDN heads | 8,192 | 27,011 | 7.50 |

The 16K microbatch was slightly slower than 8K and consumed substantially more memory. Reducing GDN head width to 64 gave only about 1.5% additional speed in its short trial, while changing recurrent state capacity. That variant has 99,940,272 dense parameters. It was not selected; the recommended configuration retains 128-dimensional GDN heads. Quality of that architectural change remains untested.

The exploratory rows have different effective batches and shorter timing windows. Use the matched-batch comparison above for the confirmed speedup.

## Corrections and targeted checks

The original model's fused linear cross-entropy call was corrected to use flattened 2D inputs. The reference convolution now accepts FLA's weight layout. Original core files were saved under `original/` in the runtime.

The benchmark preserves weights while combining GDN input projections, attention input/gate projections, and MLP gate/up projections. It optionally compiles each block's MLP region. Document boundaries, gates, attention causality, the model parameter budget, and sparse table updates remain active.

- Native attention output/input-gradient relative errors: **0.002715 / 0.004995**, below the targeted 0.03 tolerance. The check includes unequal document lengths and GQA.
- Packed GDN output error: **0**; input-gradient relative error: **0.002897** for 128-dimensional heads and **0.002881** for the 64-dimensional trial.
- Preserved-weight full-model loss before/after packing: **11.485714 / 11.485784** for the selected architecture.
- Every training step checks finite loss and dense gradient norm; sparse optimizer state is checked at completion. The matched runs completed successfully.
- The saved shell runner passed `bash -n`; the 38-file ZIP passed its CRC/integrity check.

The earlier 500M/RTX 3090 numerical failure was not investigated in this task. Successful 100M/L4 runs do not resolve its root cause.

## Saved recipe and artifacts

The notebook was renamed, updated, and saved. Runtime files are under `/content/a100stack`; the verified **1,343,769-byte** archive is `/content/a100stack-100m-experiment.zip`, visible in the notebook's Files pane. It includes corrected source, benchmark, configuration, exact JSONL results, logs, environment metadata, sample/tokenizer hashes, bounded FineWeb data, tokenizer, requirements, findings, and shell runner.

The in-app browser did not return a completed download through either the notebook download helper or the Files pane action. The ZIP's existence and integrity in Colab are verified; a local ZIP download is not confirmed. Runtime files are temporary until copied or downloaded elsewhere. This local findings document and the companion `bench100m.py` are saved independently of Colab.

To reproduce from the bundle in a compatible Colab runtime, extract it under `/content`, retain PyTorch 2.11.0+cu130, install `a100stack/requirements100m.txt`, and run:

```bash
cd /content/a100stack
python -u bench100m.py --tokens 8192 --grad-accum 4 --steps 6 --warmup 2 --attn native --pack --compile mlp --gdn-head-dim 128
```

`run100m.sh` runs that same benchmark recipe and accepts additional CLI overrides. The original `train.py` and `launch.sh` were not converted into a validated 100M production trainer. No convergence run or training checkpoint was produced.
