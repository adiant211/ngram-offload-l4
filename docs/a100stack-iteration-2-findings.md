# a100stack iteration 2 — findings at stop

Recorded 19 October 2026. Work stopped at the user's request. No further training, tuning, or debugging is authorized by this report.

## Result

The scripts were substantially corrected and targeted checks passed, but iteration 2 is **incomplete**. The intended 500M model failed with nonfinite or extremely large intermediate values before a valid training throughput measurement. Its root cause remains unresolved.

The only end-to-end training-window measurement was **about 9,251 tokens/second for the tiny preset**, using real streamed FineWeb English and the supplied tokenizer. This is not a throughput estimate for the 500M model, an A100, or a two-GPU run. That initial tiny process subsequently aborted during loader shutdown; the loader shutdown problem was fixed and checked separately, but the tiny training benchmark was not rerun afterward.

The remaining FlashAttention build and its compiler workers were terminated. A final process check showed no active training/build processes matching this work and no GPU compute process. Source, logs, data samples, and checkpoints were retained. The container itself and Jupyter were left running.

## Actual environment

- One NVIDIA RTX 3090, 24 GiB VRAM; not the plan's two A100 GPUs.
- Existing `/venv/main`, Python 3.12.15. NVIDIA driver 565.77; driver-reported CUDA capability 12.7.
- Initial successful kernel checks used PyTorch 2.8.0+cu126, Triton 3.4, flash-linear-attention/FLA 0.5.2, and FlashAttention 2.8.3.post2.
- During diagnosis, PyTorch was upgraded to 2.14.1+cu126 and Triton 3.8.0. The upgrade did not eliminate the numerical failure.
- The old FlashAttention binary was incompatible after the PyTorch upgrade. A normal rebuild failed on C++17 versus newer Torch header requirements. A source rebuild with C++20 flags was interrupted at stop; successful installation was not established. The last model backend summary used FLA GDN, SDPA reference attention, and FLA fused linear cross-entropy.
- `logs/environment.txt` records the earlier environment, before the Torch upgrade. It must not be treated as a lockfile for the final environment.
- Last observed disk usage: approximately 12 GB used, 44 GB free.

## Data

Used `HuggingFaceFW/fineweb`, configuration `sample-10BT`, English, with streaming. Tokenizer: `/workspace/good_tokenizer`; vocabulary 60,800; EOS token ID 3. Full FineWeb was not downloaded.

Documents were separated by a stable BLAKE2b hash of document ID, URL, or text: bucket 0 of 100 held out; remaining buckets used for training. EOS was appended per document. Packed batches reset positions and prevent document-boundary attention/ngram leakage. Targets immediately following an EOS input were masked.

A bounded local uint16 sample was also prepared for GPU tests:

| File under `/workspace/a100stack/data/fineweb/` | Tokens | Bytes |
| --- | ---: | ---: |
| `train.bin` | 2,097,153 | 4,194,306 |
| `validation.bin` | 32,769 | 65,538 |

`manifest.json` records sample and tokenizer metadata and SHA256 hashes. Dataset revision was not pinned, so reproducing the exact remote stream later is not guaranteed.

## Corrections made in the container

Current source is in `/workspace/a100stack`. The original ZIP remains `/workspace/a100stack.zip`; original versions of the main edited source files are under `/workspace/a100stack/original/`.

- **`data.py`:** streamed FineWeb support, bounded token buffering, document-aware packing, explicit tokenizer metadata, held-out split, loader/RNG state, and consumed-batch checkpoint state for file prefetch. Streaming moved to a synchronous iterator with explicit generator close after a background-stream shutdown crash.
- **`train.py`:** streaming/file CLI options, configurable benchmarks, held-out evaluation, checkpoint/resume state for model/optimizer/data/RNG, atomic checkpoint replacement, finite-loss checks, warmup-excluded throughput, optimizer timing, GPU memory metrics, optional profiling, and chunked table checksums. CUDA reference fallbacks require explicit opt-in.
- **`model.py`:** reference convolution accepts the FLA weight layout; fused linear cross-entropy receives flattened 2D inputs; reference loss uses smaller chunks; reference loss can be forced for checking.
- **`optim.py`:** explicit AdamW `beta1`/`beta2` settings for Dion's scalar parameter groups.
- **`config.py`:** restored the missing `450m` preset. The existing `500m` preset represents the plan's approximately 497.2M dense parameters plus 1.5B hashed table parameters; it was not validated by successful full training.
- **`ngram.py`:** compact per-microstep sparse-gradient collection, deterministic merging of duplicate rows with sorting and segment reduction, and short-input-safe shifted hashes.
- **Added utilities:** `prepare_data.py`, `test_data.py`, `test_stream_resume.py`, `check_sparse_sync.py`, expanded kernel checks, and `sweep.py` for isolated throughput and learning-rate experiments.

The sweep harness was not run. `launch.sh`, `requirements.txt`, and `README.md` remain the original versions and do not describe or reproduce the modified runtime. Local `work/` source copies predate some later container edits and are not a verified current source export.

## Targeted verification and measurements

| Check | Observed result | Limit |
| --- | --- | --- |
| Data unit checks | 3 passed: EOS/packing, ngram boundaries, consumed-state file resume | Targeted checks only |
| FineWeb stream resume | Next batch matched exactly after state restore; clean exit | Loader check, not full training resume |
| Sparse update synchronization | Two CPU/Gloo processes produced identical table/state hashes after 3 updates | Not two-GPU NCCL/DDP validation |
| Isolated GPU kernel parity | GDN, attention, full GDN, fused linear CE passed relative-error threshold 0.05 | Passed before Torch upgrade; does not establish whole-model correctness |
| Standalone native BF16 GEMMs | 64 independent operations passed; relative error about 0.00287–0.00289 against FP32 | Does not reproduce the whole-model execution context |
| CUDA memory/init sanitizers | Reported zero errors in inspected runs | Does not exclude logical/numerical kernel problems |

In the small parity run, the largest reported relative error was about **1.82%** for a full-GDN parameter gradient. Full-width checks at 128 packed tokens also passed the 5% threshold. Separate forward/backward timings at width 1024 and 1024 packed tokens were roughly **5.1–5.5 ms for GDN** and **2.1–2.2 ms for attention**. These are isolated operation timings, not training tokens/second.

### Tiny training window

The four-layer, width-128 tiny preset used vocabulary 60,800, streamed FineWeb, Muon, 512 tokens per microbatch, gradient accumulation 1, and six optimizer steps. Two initial steps were excluded from benchmark timing.

- Measured tokens: 2,048; measured training-window time: about 0.221384 seconds.
- Aggregate throughput: **9,250.89 tokens/second**. Reported windows were approximately 9,040 and 9,471 tokens/second.
- Step-4/step-6 logged loss: 11.3900 / 11.2194. Held-out losses at steps 2/4/6: approximately 11.4548 / 11.3249 / 11.2319.
- Peak allocated GPU memory: approximately 0.204 GiB.
- Checkpoint saved: `/workspace/a100stack/ckpt/tiny/step6.pt`.
- Evidence: `logs/tiny-stream.jsonl`, `logs/tiny-stream.out`.

Throughput excludes warmup, evaluation, and checkpoint time. The process later aborted during streaming-loader cleanup; the subsequent loader-only fix passed exact-resume and clean-exit checks. Do not describe this as a cleanly completed post-fix training run.

### Full-model attempt

The `500m` trial used bounded FineWeb files, Muon, BF16 table storage, 2,048 tokens per microbatch, gradient accumulation 1, MLP activation checkpointing, and compilation disabled. It encountered nonfinite loss at its first step. **No valid 500M training throughput, stable optimizer comparison, or sustained memory measurement was obtained.** Evidence: `logs/500m-2048-mlp.out` and its metrics file.

## Numerical failure: observations and uncertainty

Whole-model probes sometimes produced intermediate values around 1e23–1e25 or NaNs after initially bounded activations. Hooks observed this in several linear projections, with ordinary-sized inputs and weights; FP32 recomputation of a suspect operation was around 5–6 in magnitude. In one probe, copying the suspect output to CPU confirmed a maximum around **3.99e25**, so that observation was not merely a GPU-side diagnostic reduction artifact.

Saved failing GEMM inputs replayed correctly in a fresh process. A separate test without model/FLA/FlashAttention imports passed 64 native BF16 GEMMs. These results do not support a confident claim of broken GPU hardware or universally incorrect native BF16 arithmetic.

Diagnostic changes including precision/reference paths, allocation settings, transfer synchronization, selected GDN changes, runtime upgrade, and explicit Triton cast experiments did not establish a reliable fix. The final diagnostic that bypassed both GDN and attention also stopped at a bad intermediate; this altered architecture is not a training result. The failure's origin remains unresolved in the model-run context.

Diagnostic scripts such as `safe_cast.py`, `first_bad*.py`, and `diagnose*.py` are experiments, not validated production fixes. In particular, the safe-cast experiment did not resolve the issue and was not integrated into production training.

Useful evidence under `/workspace/a100stack/logs/`: `parity.log`, `parity-fullwidth.log`, `stream-resume.log`, `sparse-sync.log`, `native-gemm-accuracy.log`, `cpu-truth.log`, `gemm-repro.log`, `all-native.log`, `sanitizer*.log`, `initcheck.log`, `torch-upgrade.log`, and `flash-attn-cxx20.log`. A failing-operation snapshot was retained as `logs/failing-gemm.pt`.

## Incomplete plan items

No throughput sweep or optimizer/scalar learning-rate sweep was completed. No best configuration was selected. Full-model checkpoint equivalence, two-A100/NCCL behavior, a sustained dry run, and the planned approximately 10B-token training run remain unverified or unstarted. No A100 throughput or total training-time projection is justified by these measurements.

Before any future training continuation, the numerical failure and the Torch/FlashAttention environment need to be resolved, then a short stable full-model run verified. This is a record of outstanding work, not a request to resume it.
