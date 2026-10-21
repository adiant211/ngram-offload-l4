# a100stack iteration 4: Dion3, multi-GPU readiness, and three papers

Recorded 21 October 2026 on a Colab NVIDIA L4 (12 vCPUs, 52 GB RAM, Torch 2.11.0+cu130, FLA 0.5.2, `microsoft/dion` at commit `58d38ad`).

## 1. Dion3 replaces AdamW for the dense matrices

`--optimizer dion3` follows the original a100stack `optim.py`:

| Parameters | Optimizer |
|---|---|
| 2-D hidden matrices, min dim ≥ 16 (161 tensors, 68.6M params) | Dion3 = NorDion2, `algorithm="nordion2"`, lr 0.02, fraction 0.5, μ 0.95, weight decay 0.01, spectral-norm lr adjustment, Triton polar-express Newton–Schulz |
| Tied embedding/head, norms, gates, `A_log`, `dt_bias`, conv kernels, a/b projections (214 tensors, 31.3M params) | AdamW inside Dion3, lr 3e-3, betas (0.9, 0.95), no weight decay |
| 500M n-gram table | unchanged sparse row RMSprop |

**Packing.** NorDion2 orthogonalizes each Parameter as one matrix and rejects `split_sizes` ("supported only by Muon and NorMuon"). A packed Q/K/V/gate weight would therefore get a different update from the separate matrices the model defines. `--pack-mode concat` keeps the original Parameters and concatenates them only inside the forward pass, so the GEMM stays fused, Dion3 sees the real matrices, and state_dicts stay compatible with a100stack checkpoints. Concat packing costs nothing: AdamW measured 29,180 tokens/s with concat versus 29,178 with fused Parameters.

**Throughput and memory** (786,432-token updates, recommended iteration-3 recipe, one warmup + four timed updates):

| Dense optimizer | Tokens/s | Allocated / reserved peak | Loss after 5 updates |
|---|---:|---:|---:|
| AdamW (lr 3e-4) | 29,180 | 6.51 / 6.92 GiB | 9.707 |
| **Dion3** | **28,826 (−1.2%)** | **6.26 / 6.57 GiB** | **8.425** |

Dion3 keeps one momentum buffer per matrix instead of AdamW's two, hence the lower peak. The early loss gap partly reflects Dion3's much larger learning rate, so it is not a tuned comparison. The optimizer step itself is a small fraction of a 27-second update.

## 2. Multi-GPU readiness

Launch with `torchrun`. The same script then:

- Wraps the model in DDP (`broadcast_buffers=False`, bucket views). Gradients are all-reduced only on each update's last microbatch (`no_sync` before it). The n-gram table and row state are excluded from DDP.
- Gives Dion3 the DDP process group, so Newton–Schulz work is split across ranks.
- Splits data by global microbatch index, so W ranks × accumulation A see the same tokens per update as one rank × accumulation A·W. Token-weighted loss uses the global supervised-token count, computed from the fixture with no collective, scaled by W to undo DDP's averaging.
- All-reduces the 2 GB FP32 n-gram gradient accumulator (NCCL: staged through the GPU in 128 MB blocks with `AVG`; gloo: in host memory) and ORs the touched-row masks before the sparse update. Every rank then applies the identical update, so the replicated tables never need a broadcast. Rank 0's table is broadcast once at start.

**Bug found and fixed on the way:** DDP copies every tensor in the input dict to the GPU, including the CPU token ids that the n-gram hash needs. The CPU-input hook now sits on the DDP wrapper, before that copy. A lookup that receives device tensors falls back to the counted synchronizing copy instead of crashing.

**Verification:**

- `pinned_ngram.py` adds a 2-process gloo test. Each rank takes half the microbatches, and both update paths match a single process seeing all of them, with bit-identical replicas. It passed locally (RTX 3090) and on the L4.
- End to end, two gloo ranks share one L4 (accumulation 2 each) against one rank (accumulation 4), with Dion3 and all optimizations, over three updates:
  - Replicas are bit-identical: checksum spread `[0.0, 0.0]`.
  - Final loss differs by 0.06% and the dense-parameter checksum by 0.15%.
  - That is about 8× the run-to-run noise of two single-rank runs (0.008%). The most likely amplifier is Dion3's discrete top-k row selection reacting to reduction-order differences.
- Real multi-GPU NCCL runs were not possible on one L4. The NCCL code path is exercised only in structure; gloo was used for the tests.

Known limitation: each rank holds its own pinned 1 GB table and 2 GB accumulator in host RAM, so a single host with N GPUs needs N copies.

## 3. Papers

### Depth-Attention (arXiv:2606.05014): implemented, A/B below

Before each softmax-attention layer attends over the sequence, its query attends over the keys of earlier layers at the same token and mixes their (depth-mixed) values into its own value. There are no new parameters and the KV-cache shape is unchanged.

- **Reported:** Pile perplexity 8.17 → 7.77 (1.5B) and 7.55 → 7.25 (3B) at 32B tokens. At 500M, validation loss fell 0.023, and plain uniform averaging captured most of it. The gain held from 360M to 3B.
- **Fit:** only the 5 gated-attention layers of this 3:1 Gated DeltaNet hybrid participate, using the paper's half-depth stride over attention layers. DeltaNet values feed a recurrent state, which the paper does not cover. `--depth-attn {softmax,uniform}`; unit-checked in `tools/check_depth_attention.py`.
- **Cost on the 786k recipe with Dion3:** 28,602 tokens/s (−0.8%), 6.28 / 6.72 GiB peak, no extra compiled graphs.

### Layer dropout, "Don't Drop Dropout" (arXiv:2609.05275): implemented, measured

- **Recipe:**
  - drop whole blocks;
  - drop rate rising with depth, p_ℓ = ℓ/(L−1)·p_max;
  - drop rate decaying to 0 over training;
  - kept branches scaled by 1/(1−p).
- **Reported:** at 20 tokens/parameter, p_max 0.2 (5% FLOPs saved) gave +0.46% / +0.03% / −0.06% validation loss at 271M / 503M / 906M. Larger models tolerate p_max 0.6–0.99 (15–25% savings). Early exit and self-speculative decoding become viable (1.3–1.5×).
- **Fit:**
  - One mask draw per block per *microbatch*. The paper prefers per sequence, but per-sequence skipping would change each layer's packed token count and recompile the static-shape regions. 96 microbatches per update still give 96 independent masks.
  - Dropped blocks are not executed, so the FLOPs are really saved.
  - The scale is a tensor input to the compiled region: 21 graphs total, no recompile-limit hits.
  - Under DDP, the synchronizing microbatch runs dense, so no parameter is unused in the reducing backward.
  - A layer dropped this microbatch is simply not a Depth-Attention source.
  - `--layer-drop P --layer-drop-dist ild --layer-drop-schedule decreasing`; unit-checked in `tools/check_layer_dropout.py`.

Throughput, 786k-token recipe with Dion3, *constant* rising-with-depth schedule (the paper's decaying schedule averages p_max/4 over a run):

| p_max | Mean drop rate | Tokens/s | vs Dion3 |
|---:|---:|---:|---:|
| 0 | 0% | 28,826 | — |
| 0.2 | 10% (realized 10.4%) | 30,740 | +6.6% |
| 0.4 | 20% | 33,409 | +15.9% |

Memory is unchanged (6.26 / 6.6 GiB).

### Full-bandwidth transformer (arXiv:2608.08888): dropped

Latent feedback re-runs the stack 2–3× per step during a late-pretraining phase, and its benefit shows mainly in decoding. Dropped at the user's request.

## 4. Quality A/B (micro model)

Setup:
- **Training:** 301 updates of 65,536 tokens (8 × 8,192 × accumulation 8), about 19.7M tokens per arm, Dion3, the full iteration-3 recipe.
- **Data:** identical across arms, from a fresh 25M-token FineWeb sample.
- **Evaluation:** 1,048,576 held-out tokens from *different* documents (`prepare_ab.py`), at full depth and with early exit after 15 of 20 blocks.
- **Scope:** one run per arm, constant learning rates, no warmup schedule.

| Arm | Held-out loss | Last training step | Early exit @ 15/20 |
|---|---:|---:|---:|
| Baseline, seed 1234 | 5.1191 | 5.1500 | 5.3393 |
| Baseline, seed 4321 | 5.0942 | 5.1296 | 5.2821 |
| Depth-Attention (softmax), seed 1234 | **5.0741** | 5.1100 | 5.2888 |
| Layer dropout p_max 0.2, increasing in depth, decreasing in time, seed 1234 (realized 5.1%) | 5.1321 | 5.1677 | **5.2225** |

Reading it honestly:

- **Seed noise is large at this budget.** The two baselines differ by 0.025, so single-run differences below about 0.03 are not distinguishable.
- **Depth-Attention looks positive but is not established.**
  - It is −0.045 against its same-seed baseline (paired: identical initial weights and data order, since it adds no parameters), but only −0.020 against the other seed.
  - The direction and size match the paper's 500M ablation (−0.023).
  - Confirming it needs at least 3 seeds per arm or a longer run.
- **Layer dropout is neutral at full depth and clearly better for truncated inference.**
  - Full-depth loss is +0.013, inside the noise; the paper reports +0.46% at 271M and break-even from ~500M.
  - Early-exit loss after 15 of 20 blocks improves by 0.06–0.12 over both baselines, the depth robustness behind the paper's early-exit and self-speculative decoding results.
  - Its compute saving is only 5% with this schedule.
- **Implications for the planned ~700M model:**
  - Depth-Attention is nearly free, so it deserves a multi-seed check.
  - Layer dropout is worth carrying with the paper's recipe. At 700M, p_max 0.2–0.4 is in the range the paper found neutral or better (503M/906M at 0.2), with 5–10% of training compute saved and early exit as a bonus. Higher p_max was only validated at ≥1.8B.

Bugs found during the A/B, both fixed: two evaluation fields (`eval_tokens`, `eval_exit_at`) collided with CLI flag names when the result record was built, so three arms printed their `EVAL` line but lost their JSON record. Results now merge evaluation fields with `dict.update`.

## Reproduce

```bash
pip install "git+https://github.com/microsoft/dion.git@58d38ad"
bash iter3_colab.sh setup
bash iter3_colab.sh parity      # n-gram checks incl. 2-rank gloo equivalence
bash iter3_colab.sh dion        # AdamW-concat vs Dion3 throughput
bash iter3_colab.sh dist        # 1 rank x accum 4 vs 2 gloo ranks x accum 2
bash iter3_colab.sh layerdrop   # layer-dropout throughput
bash iter3_colab.sh depthab     # quality A/B (builds disjoint FineWeb train/held-out fixtures)
```
