# a100stack iteration 6: n-gram table upgrade, Delta Block and layer dropout on, a learning-rate fix

Recorded 21 October 2026 on Colab: NVIDIA A100-SXM4-80GB, 12 vCPUs, PyTorch 2.11.0+cu130, FLA 0.5.2,
`microsoft/dion@58d38ad`. Raw records: `results/results-iteration-6.jsonl`.

## 1. What changed

### N-gram table (from `ngram_handover.md`)

| Change | Where | Notes |
|---|---|---|
| 1.2B as 4 × 585,937 × 512 → **600M as 8 tables × 2,343,750 rows × 32** (bigram + trigram × 4 hashes) | config | Measured purity on 100M FineWeb tokens: bigram 0.62 → 0.79, trigram 0.17 → 0.31 |
| **Normalized token ids for hashing** (Engram tokenizer compression) | `ngram_canon.py`, `pinned_ngram._indices`, a100stack `NgramMemory.lookup` | NFKC, NFD, strip accents, lowercase, collapse whitespace. 60,800 ids fold into 45,455 classes (25.2%). Special tokens and incomplete UTF-8 pieces keep their own id. Hashing only; the map's SHA-256 is stored in checkpoints and verified on resume |
| **Second injection point**, `inject_layers = (1, 15)` | config | One lookup shared by both points; each has its own 256→1,280 projection and gate. Gradients from both sum into the shared leaf |
| **Stochastic rounding** of the BF16 table writes | `pinned_ngram` | Add uniform noise to the 16 bits BF16 drops, then truncate. Noise comes from a fixed seeded buffer at an offset derived from (seed, update, block), so resumed runs repeat the draws |
| **Gather path in the table step** | `pinned_ngram._block_update` | With 32-wide rows only ~18% of rows are touched per 786k-token update; blocks with < 35% touched rows update just those rows, and clear only those accumulator rows. Same arithmetic, bit-identical to the row and dense paths |
| Block size by row width | `pinned_ngram.enable_pinned` | ~8 MB of FP32 scratch per block at any width |

### Training recipe (package defaults)

- **Layer dropout:**
  - The deepest block's rate falls linearly from 0.25 to 0.05 by the start of the learning-rate decay (80% of the schedule), then holds at 0.05.
  - The rate rises with depth (`ild`).
  - New flags: `--layer-drop-final`, `--layer-drop-end`; in `exp_config.toml`, `layer_drop_end = "decay"`.
- **Delta Block attention residuals:** on.

### Fixes found while testing

1. **Compounding learning rate (`trainer.py`, since iteration 5).**
   - **Cause:** Dion3 stores each group's `lr` as a persistent device tensor and copies anything assigned to `group["lr"]` into it. The trainer kept that tensor as its base learning rate, so each update wrote base × multiplier into the base itself.
   - **Effect:** with a 300-update warmup the dense learning rates collapsed to ~0 within a few updates. Only the n-gram table trained, which is why the iteration-5 supervised run's loss barely moved.
   - **Fix:** the base rates are now plain floats from construction, and every update logs the effective `lr`.
   - **Verified:** the 100M model with a 30-update warmup reaches loss 6.99 at update 40 (11.48 before the fix), and the logged `lr` ramps exactly 0.00067 → 0.02.
   - **Not affected:** iteration-5 throughput, memory and recovery results, and the iteration-4 A/B (`bench100m.py` has no schedule).
2. **Recompile limit (Delta Block).**
   - **Cause:** all blocks share the two compiled Delta Block functions. With layer dropout and early-exit evaluation they need more than Dynamo's default 8 variants (1–4 sources × grad mode × the dropout scale input), and `fullgraph` makes that an error at the first evaluation.
   - **Fix:** `delta_block.install` raises the limit to 64.
3. **Supervisor loop.** That error repeated on every restart. `exp.py` now gives up after 3 consecutive failures that produce no new checkpoint.

## 2. Throughput (700m preset, 16k × 48 = 786k-token updates)

| Setup | Tokens/s | GPU (device) | Table step |
|---|---:|---:|---:|
| Previous defaults (1.2B table, 1 injection) | 27,493 | 42.1 GiB | 1.42 s |
| New table (before the gather path) | 27,804 | 43.4 GiB | 1.33 s |
| **New table** | **28,235** | 42.0 GiB | 0.80 s |
| New table + layer dropout at its starting rate (before the gather path) | 30,959 | 41.9 GiB | 1.34 s |
| **New defaults**: new table + Delta Block + layer dropout, start of run (12.3% of blocks skipped) | **27,701** | 42.2 GiB | 0.79 s |
| New defaults without stochastic rounding | 27,830 | 42.2 GiB | 0.68 s |

- **Delta Block** costs ~12% at 700M.
- **Layer dropout** gives +11.3% at its starting rate; averaged over a run (~6.5% of blocks skipped) the new defaults should run at about 26,400 tokens/s (estimate).
- **Stochastic rounding** costs 0.12 s per update (0.4%).
- **N-gram stages per update, old → new table:**

  | Stage | Old table | New table |
  |---|---:|---:|
  | CPU hashing | 143 ms | 266 ms (8 hashes, plus the canonical map) |
  | Row gather | 402 ms | 136 ms |
  | Row copies to GPU | 295 ms | 39 ms |
  | Gradient copies back | 535 ms | 67 ms |
  | Accumulation | 622 ms | 472 ms |
  | Peak host RAM | 21.9 GiB | 12.9 GiB |

## 3. Quality A/B (100m preset)

Setup:
- 300 updates × 65k tokens (~20M tokens); 30-update warmup, WSD with 20% decay.
- Held-out FineWeb loss on 1M tokens; early exit after 15 of 20 blocks.
- 2 seeds per arm.
- The new-table arms inject at blocks 2 and 11.

| Arm | Seed 1234 | Seed 4321 | **Mean** | Early exit (mean) | Speed |
|---|---:|---:|---:|---:|---:|
| Previous table (1.2B, 4 × 585,937 × 512, 1 injection) | 5.0442 | 5.0575 | 5.0509 | 5.2545 | ~37.7k tokens/s |
| New table (all four handover items) | 5.0286 | 5.0316 | **5.0301** | 5.2339 | ~43.8k tokens/s |
| New table + Delta Block + layer dropout 0.25 → 0.05 | 5.0349 | 5.0320 | 5.0335 | **5.1029** | ~36.2k tokens/s |

- **New table vs previous:** −0.021 at full depth, on both seeds (−0.016, −0.026); the previous table's seeds differ by 0.013. It is also 16% faster at this size.
- **Delta Block + layer dropout added:** +0.003 at full depth, i.e. neutral. Early-exit loss is −0.131. Layer dropout is the known source of the early-exit gain; Delta Block's own effect was not isolated.
- **Scope:** 20M tokens at 100M parameters, two seeds. The handover's four items were tested together, not separately.

## 4. Verification

- **`pinned_ngram.py` parity, all passing:**
  - **Row vs block vs gather paths:** tables bit-identical, plain and with canonical ids.
  - **GPU-table path:** passes.
  - **Stochastic rounding:**
    - against round-to-nearest: within one BF16 step at the update's scale, mean drift < 1e-3;
    - unbiasedness: mean 1.000998 for a target of 1.001 over 64 draws;
    - representable values unchanged;
    - draws deterministic per (seed, update, block).
  - **2-rank gloo tests (rows, blocks):** replicas bit-identical.
- **`ngram_canon.py`:** deterministic, idempotent, each class maps to its smallest id.
- **Crash and resume with every new feature on:**
  - `kill -9` after update 8; the supervisor resumed from checkpoint 6 with checksums verified, including the canonical-map hash.
  - Updates 7 and 8 repeated within 3e-6: loss 11.51249 / 11.51192 vs 11.51246 / 11.51191.
