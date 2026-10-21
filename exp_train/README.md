# exp_train: 700M hybrid LM + 600M n-gram table, one GPU

A ready-to-run, easy-to-tweak training setup:

- **Model:** a 714M-parameter hybrid language model (24 Gated DeltaNet + 8 gated attention layers, width 1,280) with a 600M-parameter hashed n-gram memory kept in CPU RAM:
  - **Memory:** 8 hash tables of 32-wide rows, read with normalized token ids and added at two depths.
  - **On by default:** Delta Block attention residuals, and layer dropout falling from 0.25 to 0.05 by the start of the learning-rate decay.
- **Tokenizer:** the 60,800-token tokenizer, in `tokenizer/`.
- **Settings:** everything lives in **one file, `exp_config.toml`**.

On one A100-80GB the default trains at **~27,700 tokens/s at the start and ~26,400 averaged over a run, using
~42 GB of GPU memory**, and a crashed or stopped run resumes from its last checkpoint by itself.

## Quick start

```bash
bash setup.sh                 # FLA 0.5.2 and Dion3 on top of your PyTorch
python exp.py check           # GPU, packages and correctness checks (~5–10 min)
python exp.py prepare         # download and tokenize FineWeb into data/ (1B tokens, ~17 min)
python exp.py train           # train; Ctrl-C stops cleanly with a checkpoint
```

Run `python exp.py train` again (same `run.name`) to continue where it stopped.

## Requirements

| | Default settings | Smaller-memory profile (`micro_tokens = 10240`, `grad_accum = 77`) |
|---|---|---|
| GPU | NVIDIA, BF16, ≥ 48 GB (A100-80GB tested) | ≥ 34 GB |
| Host RAM | ≥ 32 GB (table 1.2 GB + 2.4 GB gradient buffer + checkpoint copies; 12.9 GiB peak measured between checkpoints) | same |
| CPU | the n-gram table update uses all cores; 12 vCPUs tested | same |
| Disk | data (2 bytes/token) + 2 checkpoints × 8.8 GB + compile cache ~2 GB | same |
| Software | Linux, Python ≥ 3.11, PyTorch 2.11 + CUDA, `flash-linear-attention==0.5.2`, Dion3 (`setup.sh`) | same |

## What's in here

| Path | What it is |
|---|---|
| `exp_config.toml` | **All settings.** Commented; explained in `SETTINGS.md` |
| `exp.py` | Launcher: `train`, `prepare`, `check`, `stop`; `--set section.key=value` for one-off changes |
| `setup.sh` | Installs FLA 0.5.2 and Dion3 |
| `tokenizer/` | The 60,800-token tokenizer (`tokenizer.json` is the one used) |
| `code/` | Model source (a100stack, with two small fixes applied) and the training stack |
| `runs/<name>/` | Created per experiment: config snapshot, `train.log`, `metrics.jsonl`, `results.jsonl`, `ckpt/` |
| `SETTINGS.md` | Every setting: what it does, measured effect, when to change it |
| `RESULTS.md` | Measured throughput and memory, recovery tests, what was tried and rejected |
| `ARCHITECTURE.md` | Model shape and every modification in the stack |
| `docs/` | Full iteration-5 findings, raw result records, the original a100stack README |

## Common tweaks

```bash
python exp.py train --set run.name=lowmem --set batch.micro_tokens=10240 --set batch.grad_accum=77 --set performance.gpu_mem_cap_gib=31
python exp.py train --set run.name=nodelta --set model.delta_block=false    # ~12% faster without Delta Block
python exp.py train --set run.name=nold --set model.layer_drop=0            # layer dropout off
python exp.py train --set run.name=wide --set 'model.shape."ngram.dim"=64' --set 'model.shape."ngram.heads_per_order"=2'   # 600M as 4 x 2.34M x 64
python exp.py train --set run.name=short --set schedule.max_minutes=60       # one hour, then checkpoint and stop
python exp.py train --dry-run                                                # show the command, run nothing
```

Use a new `run.name` for every changed experiment. A run only resumes from checkpoints made with
the same model configuration.

## Monitoring and stopping

- `runs/<name>/train.log`: everything the trainer prints. One `STEP {...}` line per update with
  loss, tokens/s, MFU, memory and timings; `CHECKPOINT`, `RESUMED` and `EVAL` lines.
- `runs/<name>/metrics.jsonl`: the same per-update records, one JSON object per line.
- `python exp.py stop` (or Ctrl-C, or SIGTERM): the trainer finishes the current update, writes a
  checkpoint and exits. A crash or `kill -9` instead triggers an automatic restart from the newest
  complete checkpoint.

## Notes

- The first update compiles kernels: about 12 minutes on a fresh machine, about 1 minute with the
  `compile-cache/` folder from an earlier run.
- The default schedule is 12,716 updates × 786,432 tokens = 10B tokens. `prepare` makes 1B tokens by
  default; the loader cycles through the file (a new shuffled block order per pass). For a full run,
  raise `prepare.train_tokens` (FineWeb sample-10BT has ~10B).
- Checkpoints survive crashes, not the loss of the machine's disk. On Colab, put `runs_dir` and
  `cache_dir` on mounted Google Drive.
