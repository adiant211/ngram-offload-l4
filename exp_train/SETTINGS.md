# Settings

All settings live in `exp_config.toml`. Override any of them for one run with
`python exp.py train --set section.key=value`.

Unknown keys are rejected, so a typo can't silently do nothing. All numbers are measured on one
A100-SXM4-80GB with the 700m preset and ~786k-token updates unless marked otherwise.

## Choosing a memory profile

GPU memory is set almost entirely by `batch.micro_tokens`. For the 700m preset, memory as `nvidia-smi`
reports it is about **11 GiB + 1.95 GiB per 1,024 microbatch tokens**.

| `micro_tokens` × `grad_accum` | Tokens/s | GPU memory | Use when |
|---|---:|---:|---|
| 8,192 × 96 | ~24,000 (est.) | 26.5 GiB | GPU ≤ 32 GB |
| 10,240 × 77 | 26,379 | 30.6 GiB | GPU ≤ 34–40 GB (set `gpu_mem_cap_gib = 31`) |
| **16,384 × 48** | **27,504** | **42.7 GiB** | **default**; GPU ≥ 48 GB |
| 20,480 × 38 | 27,502 | 51.2 GiB | never: no faster than 16k |
| 24,576 × 32 | 28,145 | 58.1 GiB | only if +2.3% is worth 15 GiB |

Keep `micro_tokens × grad_accum` near 786k when you switch profiles, so the training recipe stays the
same.

The table rows were measured with the previous 1.2B table and without Delta Block. The default 600M table
uses about the same GPU memory (42.0 vs 42.7 GiB at 16k); Delta Block adds ~0.2 GiB.

## [run]

| Key | Default | Meaning |
|---|---|---|
| `name` | `700m-base` | Experiment name; everything goes to `runs/<name>/`. Same name = continue that run. Change it for every new experiment. |
| `seed` | 1234 | Initialization, data order, layer-dropout draws. |
| `runs_dir` | `runs` | Put on persistent storage (e.g. mounted Google Drive) to survive losing the machine. |
| `cache_dir` | `compile-cache` | Compiled GPU kernels. With it, a restart reaches training in ~1–2 min; without it, ~12 min. |
| `max_restarts` | 20 | Automatic restarts after crashes. |

## [data] and [prepare]

| Key | Default | Meaning |
|---|---|---|
| `train` | `data/fineweb-train.bin` | Flat `uint16` token file, documents separated by `<eos>`. Any tokenized corpus in this format works. |
| `val` | `data/fineweb-val.bin` | Held-out documents for evaluation. If the file is missing, evaluation is skipped. |
| `tokenizer` | `tokenizer/tokenizer.json` | The 60,800-token tokenizer. The model's vocabulary size must match it. |
| `prepare.train_tokens` | 1,000,000,000 | Tokens `exp.py prepare` writes (~1M tokens/s, 2 bytes each). The loader cycles through the file with a new block order each pass. |
| `prepare.val_tokens` | 2,097,153 | Held-out tokens, from documents before the training ones in the stream. |

## [model]

| Key | Default | Meaning and measured effect |
|---|---|---|
| `preset` | `700m` | `700m`: 32 layers × 1,280, 716M dense. `500m`: 32 × 1,024, ~494M. `100m`: 20 × 512, 100M (the L4 experiments). |
| `ngram_params` | 600M | Size of the hashed bigram/trigram table in CPU RAM. Its shape is set in `[model.shape]` (default: 8 tables × 2,343,750 rows × 32). Costs 2 bytes/param for the table plus 4 bytes/param for the gradient buffer (3.6 GB at 600M). The previous default was 1.2B as 4 × 585,937 × 512. |
| `layer_drop` | 0.25 | Layer dropout (arXiv:2609.05275): starting drop rate of the deepest block. The rate rises linearly with depth from 0 at the first block (`ild`), so at the start ~12.5% of blocks are skipped per microbatch. Dropped blocks are not computed. Full-depth loss neutral (within seed noise) in an earlier 100M A/B; much better early-exit loss. At 700M: +11.3% throughput at the starting rate (12.3% of blocks skipped); about +6% averaged over a run with these defaults (estimate). 100M A/B: much better early-exit loss. |
| `layer_drop_dist` | `ild` | `ild`: drop rate rises linearly with depth (paper default). `uniform`; `ald`: every other layer. |
| `layer_drop_schedule` | `decreasing` | `decreasing`: linear from `layer_drop` to `layer_drop_final`, reached at `layer_drop_end`, then held. `constant`: `layer_drop` throughout. `increasing`. |
| `layer_drop_final` | 0.05 | Rate held from `layer_drop_end` to the end of training. 0 = the paper's schedule. |
| `layer_drop_end` | `"decay"` | When `layer_drop_final` is reached. `"decay"`: when the learning-rate decay starts (80% of the schedule with `decay_frac = 0.2`), or a fraction of the schedule (`1.0` = the end). Averaged over a whole run, ~6.5% of blocks are skipped with the defaults. |
| `depth_attn` | `off` | Depth-Attention (arXiv:2606.05014) on the 8 attention layers. −0.8% throughput (L4 measurement). Quality gain unconfirmed: −0.045 loss in one paired 100M A/B, inside the 0.025 seed noise. |
| `delta_block` | true | Delta Block attention residuals (Luo, Cai, Hu): each sublayer input mixes block-level depth sources. At 700M it costs ~12% throughput (with layer dropout on) and +0.15 GiB. In the 100M A/B, Delta Block + layer dropout were neutral at full depth (+0.003 loss) and −0.13 with early exit; Delta Block's own effect was not isolated. The authors report +0.6 points downstream accuracy (Qwen3-0.6B fine-tune) and 1.7–2.6% lower perplexity (533M–1B). |
| `delta_blocks` | 4 | Depth blocks (sources). 4 → blocks of 8 layers. |
| `delta_variant` | `repo` | `repo`: the published code (its sources are alternating sums beyond the second block). `paper`: true block deltas. |

### [model.shape]

These keys override fields of the preset. The n-gram keys are quoted because they contain a dot;
`--set` accepts them too: `--set 'model.shape."ngram.dim"=64'`.

| Key | Default | Meaning |
|---|---|---|
| `"ngram.heads_per_order"` | 4 | Hash functions per n-gram order. Bigram + trigram × 4 = 8 tables. More, narrower tables lower the share of each row's signal that comes from colliding n-grams. |
| `"ngram.dim"` | 32 | Row width. Rows = `ngram_params / (tables × dim)` = 2,343,750 per table. |
| `"ngram.inject_layers"` | [1, 15] | Where the memory is added to the residual stream: before block 2 and before block 16 (of 32), as in Engram's 2 + 15 of 30. One lookup is shared; each point has its own projection and gate. |
| `"ngram.canonical_tokens"` | true | Hash normalized token ids (Engram's tokenizer compression: NFKC, accents stripped, lowercase, whitespace collapsed). 60,800 ids fold to 45,455 for hashing only; the model's embedding and targets are unchanged. The map's checksum is stored in each checkpoint. |
| `"ngram.stochastic_round"` | true | Round the BF16 table updates stochastically instead of to nearest: unbiased, so small updates to large rows aren't lost. The draws are deterministic per (seed, update, block), so resumed runs repeat them. |
| `"ngram.orders"`, `"ngram.lr"`, `"ngram.beta2"` | (2, 3), 0.02, 0.99 | N-gram orders and the sparse RMSprop settings. |

The model itself can be changed with `d_model`, `n_layers`, `full_attn_every`, `n_heads`, `n_kv_heads`,
`head_dim`, `gdn_heads`, `gdn_head_dim`, `ffn_dim` and `rope_theta`. Rules:
- `n_heads × head_dim = d_model` and `gdn_heads × gdn_head_dim = d_model`;
- `n_heads` must be a multiple of `n_kv_heads`;
- `ffn_dim` should be a multiple of 128;
- `"ngram.inject_layers"` and `eval.eval_exit_at` must stay below `n_layers`.

After changing the model itself, the memory table and the A100 reference speed no longer apply. Re-measure with
a short run (`schedule.max_minutes = 15`).

## [batch]

| Key | Default | Meaning |
|---|---|---|
| `micro_tokens` | 16,384 | Tokens per forward/backward pass (documents packed into one stream). Sets GPU memory; see the table above. |
| `grad_accum` | 48 | Microbatches per optimizer update. Update size = `micro_tokens × grad_accum`. Larger updates spread the fixed per-update cost (1.4 s table step + 0.2 s Dion3, ~5% at 786k) but change the training recipe. |
| `seq_len` | 1,024 | Longest piece of one document; longer documents are split there. Larger values let attention see further; the cost grows with document length. |

## [schedule]

Warmup-stable-decay. The schedule also scales the n-gram table's learning rate.

| Key | Default | Meaning |
|---|---|---|
| `steps` | 12,716 | Schedule length in updates. Total tokens = `steps × micro_tokens × grad_accum` (10.0B by default). |
| `warmup_steps` | 300 | Linear warmup (236M tokens). Each update logs the effective matrix learning rate as `lr`. |
| `decay_frac` | 0.2 | The last 20% of updates decay to `min_lr_frac`. |
| `min_lr_frac` | 0.0 | Final learning-rate fraction. |
| `stop_at` | 0 | End this invocation after update N; resume later continues the same schedule. |
| `max_minutes` | 0 | Stop with a checkpoint after this long. |

## [optimizer]

| Key | Default | Meaning |
|---|---|---|
| `optimizer` | `dion3` | Dion3 (Microsoft `dion`, NorDion2) on the 257 weight matrices, AdamW inside it on the embedding, norms and gates (as in a100stack's `optim.py`). `adamw`: AdamW for everything. |
| `dion_lr` | 0.02 | Matrix learning rate (spectral-norm adjusted). From the a100stack preset; not tuned for 700M. |
| `dion_fraction`, `dion_mu`, `dion_weight_decay` | 0.5, 0.95, 0.01 | Dion3 rank fraction, momentum, weight decay. |
| `scalar_lr` | 3e-3 | AdamW learning rate for embedding/norms/gates. |
| `adamw_lr` | 3e-4 | Only with `optimizer = "adamw"`. |
| `grad_clip` | 1.0 | Global gradient-norm clip. |
| `dion_triton` | true | Triton Newton–Schulz kernels. |
| `dion_compile` | true | Dion3 compiles its helpers: ~8 min on the first update of a fresh machine, ~40 s with a warm cache. `false` skips that at slightly slower steps (not measured). |

## [performance]

Changing these doesn't change the math, except where noted.

| Key | Default | Meaning and measured effect |
|---|---|---|
| `compile` | `tail` | Compiles each block's residual adds + norm + MLP. `mlp`/`none` are slower. |
| `loss_chunks` | 4 | Fused cross-entropy chunks. More chunks use less memory and run slightly slower. |
| `fast_ce` | true | Patched fused loss without a per-microbatch sync. The head weight gradient becomes one BF16 GEMM (0.17% different). |
| `fla_nosync` | true | Removes five CPU–GPU syncs per microbatch inside FLA. Bit-identical results. **+6.5%** at 10k microbatches (GPU busy 90% → 98%). |
| `residual_dtype` | `float32` | `bfloat16` saved no memory in this model. |
| `update_threads` | 0 | CPU threads for the table step; 0 = all cores. |
| `gpu_mem_cap_gib` | 52 | Hard cap on PyTorch's GPU allocations: going over raises an error instead of using more memory. The device shows ~1–2 GiB more (CUDA context, kernels). 0 = no cap. |

## [eval]

| Key | Default | Meaning |
|---|---|---|
| `eval_every` | 250 | Held-out loss every N updates (and at the end). Costs ~15–20 s per evaluation at 1M tokens. |
| `eval_tokens` | 1,048,576 | Held-out tokens per evaluation. |
| `eval_exit_at` | 24 | Also report the loss using only the first N blocks (early exit). This is useful with layer dropout. 0 = off. |

## [checkpoint]

| Key | Default | Meaning |
|---|---|---|
| `ckpt_minutes` | 30 | Checkpoint interval. Each is 8.8 GB, pauses training ~5 s, and is written in the background in ~35 s. |
| `ckpt_every` | 0 | Or every N updates. |
| `keep` | 2 | Completed checkpoints kept; older ones are deleted. |
| `resume` | `auto` | `auto`: the newest complete checkpoint in `runs/<name>/ckpt` (an older one if loading fails). `none`: start fresh. Or a checkpoint directory. |
| `max_skips` | 3 | Updates with non-finite loss or gradients are skipped. After this many in a row the trainer exits so the supervisor restarts it from the last checkpoint. |

## Not exposed in exp_config.toml

The trainer also has multi-GPU options for `torchrun`: `--ngram-sync gloo`, `--ddp-comm-hook bf16`,
`--dion-replicated` and `--resync-every`. It also has diagnostics: `--profile-micro N` (kernel time
by category) and `--sync-debug` (lists CPU-GPU sync sites). See `python code/trainer.py --help`.
