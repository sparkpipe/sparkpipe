# K3 serial-TP16 replay plan (offline, band-down) + golden reference

Owner: K3 MODEL agent · Scope: the `iterative TP1` serial replay for K3 on the
idle spark — one rank shard at a time under the 108G cap, emulating the TP16
all-reduce by summing 16 rank partials. Plan only; no launches, no downloads, no
commits. Verified by grep/read against this clone at `afb43a8`; cited file:line.

---

## 0. Scout report (read-only — what actually exists)

**No K3 TP16 rank shards exist on any host.** The sharder accepts degree 16 only
on 32-element-tile packs (`tools/k3_shard.py`: `_expert_gate_up`/`_expert_down`
refuse when `k_tiles % degree != 0`; the module docstring: "TP16 is REFUSED unless
the pack carries 32-element tiles"). No `expert_tile_k=32` pack has been produced
("Producing the full-model TP16 pack is a packer run away", `docs/K3_TP4PP4_PREP.md`
"TP16 pack path COMPLETE at the code level").

What DOES exist (16-rank TP4xPP4, tile_k=128):

| Host(s) | Artifact | Size | vs 108G |
| --- | --- | --- | --- |
| spark0..sparkf | TP4xPP4 rank packs `k3.stage{s}.rank0{t}.pack` at `/home/sparkX/sparkdata/k3.mxfp4.tp4pp4/packs/` (spark[s*4+t]) | ~52-53 GB | **fits** |
| spark0 | stage-0 rank-0 (the "other idle spark" candidate) | ~53.6 GB | fits |
| spark2 | stage-0 rank-2 | ~53.6 GB | fits |
| spark1/4/8/c | PP stage packs in `/tmp/k3_stage_*` | ~350-393 GB | **over** |
| (gate scratch) | 4-layer slice pack (layers 0-3, ~65 GB, tile_k=128) from the TP4 equivalence gate | ~65 GB | fits (also under the ~96 GB `cudaHostRegister` budget, `docs/K3_TP4PP4_PREP.md:199`) |
| — | full model (MXFP4) | ~1.56 TB | **over** |

Host mapping (`tools/k3_deploy_ranks.sh`): stage s rank t → `spark[s*4+t]`;
spark0 = stage0.rank00, spark2 = stage0.rank02.

**Scout conclusion.** spark0/spark2 hold TP4xPP4 rank packs (NOT TP16). To do a
TP16 serial replay, the TP16 shards must first be produced (packer
`expert_tile_k=32` → `k3_shard.py ... 16`). The golden reference for the
equivalence must be a **slice pack**, not the full model — the full model (1.56 TB)
and even a stage pack (~350-393 GB) exceed both 108G and the ~96 GB registration
budget, exactly as the TP4 gate concluded (`docs/K3_TP4PP4_PREP.md:193-202`).

---

## 1. Prerequisite — produce the TP16 shards (packer + sharder; needs the host cleared)

1. Pack the SAME 4-layer slice the TP4 gate used (layers 0-3: dense-KDA, routed
   MLA, routed KDA, routed MLA) with `expert_tile_k=32` — the sharder only admits
   degree 16 on 32-element tiles (224 = 7×32 for w1 k, 192 = 6×32 for w2 k;
   `docs/K3_TP16_REPACK.md`). Slice size stays ~65 GB (tile_k changes nothing in
   total bytes); the 16 rank slices are ~4 GB each.
2. Slice 16 ways: `k3_shard.py SLICE.pack /tmp/k3_tp16_ 16` → 16 self-describing
   rank packs (~4 GB each).

---

## 2. Serial replay — shard order and run procedure

Run **serially, one at a time, on one idle spark**, correctness over speed:

1. **Golden leg first:** the full slice pack at `tp_degree 1` via the runner's
   `--pp1` derive mode (the TP4 gate's mode, `docs/K3_TP4PP4_PREP.md:202`).
   `K3_HOOK_DUMP=<prefix>_full`.
2. **Ranks 0..15 in order:** each `k3_tp16_/rank{NN}.pack` at `tp_degree 1`,
   `K3_HOOK_DUMP=<prefix>_r{NN}`. Each rank leg is ~4 GB — trivially under 108G.

The runner's dump (`spark_k3_resident_decode_stage_runner.cu:426-441`) writes raw
BF16 per layer/phase as
`<prefix>_r<rank>_l<layer>_p<phase>_<stage>_<name>.bin` (`name` ∈
{attention_out, hidden, shared_out, attnres_partial}). At `tp_degree 1` the layer
folds its own projections (`tp_sharded=0`) so the dump IS the rank's partial; the
sum of the 16 partials is the full hidden.

---

## 3. TP16 collective emulation spec

The TP16 all-reduce is `SparkTpCollectiveAllReduceSumBf16/SumF32` (or NCCL degree
16; `runner.cu:400,546,562,785`). Offline it is emulated element-wise in fp32:

- **SUM (the bulk):** `full[k] ≈ Σ_{r=0..15} rank_r[k]`, compared in fp32 with a
  BF16-rounding tolerance. Covers: the slot-encoded embedding (out-of-slice rank
  contributes 0, `runner.cu:367-369`), and the per-layer attention/MoE partials
  folded into the AttnRes by the two-phase hook (`runner.cu:411-421,537-562`).
- **MAX (the head, NOT a sum):** the lm_head is vocab-row-split; each rank emits its
  local (score, token) argmax and the collective is a **max-over-scores** merge
  (`K3RunnerHeadExchange`, `runner.cu:767-800`). Emulate by comparing the 16 rank
  (score, token) slots and taking the max score (tie-break by token id, matching the
  checker's deterministic tie rule).
- **Reuse `tools/k3_tp4_equivalence_check.py`** extended 4→16 ranks: it already
  sums in fp32 and compares with rel = Δ/max(|full|,0.25) < 0.03
  (`tools/k3_tp4_equivalence_check.py:57-74`). For 16 BF16 partials the worst-case
  rounding error is ~4x the 4-rank case — loosen the rel threshold to ~0.06 (or add a
  small absolute floor) rather than re-derive; the golden receipts (below) pin the
  contract as "4-rank sum == full to bf16 rounding".

---

## 4. Expert routing per rank (K3 geometry wiring for the shared harness)

All 16 ranks route IDENTICALLY: `router_weight`/`router_bias` are REPLICATED
(`tools/k3_shard.py` `REPLICATED` set), so each rank's `route_expert`/`route_weight`
(top-16 of 896) match. The sharding is the sharder's classification (`tools/k3_shard.py:41-68`):

| class | tensors | rank slice |
| --- | --- | --- |
| REPLICATED | norms, attnres_{attn,mlp,out}, router_w/b, kda_decay_down/bias, kda_head_log_scale, kda_out_norm, mla_q_down/norm, mla_kv_a/norm, routed_norm | whole |
| OUTPUT_HEADS | kda_qkv_beta (per-section), kda_q/k/v_conv, kda_decay_up, kda_gate, mla_q_up, mla_kv_b_value, mla_gate | 96/16 = 6 heads |
| INPUT_HEADS | kda_out_weight, mla_out_weight | input-split on heads → partial (SUM) |
| OUTPUT_DIM | routed_down_weight | latent/16 = 224 rows |
| INPUT_DIM | routed_up_weight | latent/16 = 224 cols → partial (SUM) |
| CONCAT_OUTPUT | shared_w1_weight, dense_gate_up_weight | each gate/up half output-split, re-concat |
| INPUT_DIM_PLAIN | shared_w2_weight, dense_down_weight | input-split → partial (SUM) |
| **EXPERT_CONCAT** | **expert_w1_weight** | **DIAGONAL**: rank's gate|up cells (12×2=24 cells = 384 out) × rank's latent k-tiles (7×32 = 224 in) |
| **EXPERT_INPUT** | **expert_w2_weight** | k-split: rank's intermediate k-tiles (6×32 = 192) — see §6 flag |
| vocab | model.embed_tokens.weight, lm_head.weight | vocab/16 = 10240 rows |

The MoE per-rank dataflow (degree 16): replicated router → routed_down (224) →
w1 diagonal (224→384) → SiTU (384→192) → w2 (192→224) → finalize (top-16 weighted
sum → latent 224) → routed_up (224→hidden partial) → AR sums the partial.

---

## 5. Golden reference (bit-determinism receipts from docs/K3_PERF.md)

- **4 ULP determinism everywhere** — "fresh-run determinism and capture fidelity now
  hold at 4 ULP everywhere" (`docs/K3_PERF.md:19`).
- **TP4 4-rank sum == full to bf16 rounding** — the exact contract the serial replay
  extends to 16 ranks (`docs/K3_PERF.md:19-20`).
- **Graph replay bit-identical** — 54.2 ms, 0 mismatches beyond ULP limits
  (`docs/K3_PERF.md:53,129`).
- **Single-spark bit-deterministic** — 0 mismatches, 6/6 identical dumps
  (`docs/K3_TP4PP4_PREP.md:153`).
- **TP4 offline equivalence PASS** — "the rank dumps SUM to the full dump (the exact
  contract the TP4 all-reduce ships)" (`docs/K3_TP4PP4_PREP.md:193-202`).

**Golden tolerance for TP16:** fp32 sum of 16 rank BF16 partials vs the full BF16,
rel ≤ ~0.06 (16-way worst-case ≈ 4x the 4-rank 0.03 floor).

---

## 6. Flags — verify before trusting the TP16 MoE numerics

1. **TP16 shards do not exist yet** — §1 is a packer + sharder run (compute), which
   needs the host cleared. Until then the only runnable replay is the existing TP4xPP4
   (4-way) via the current 4-rank checker.
2. **w2 sharding inconsistency (flagged, not yet closed).** The sharder's
   `_expert_down` input-splits w2 on k-tiles but keeps its OUTPUT at full latent
   (`tools/k3_shard.py` `_expert_down` → `_reprice_interleave(..., k_dim=...)` with
   out_dim unchanged), while the layer's finalize reads the w2 output at the rank's
   latent slice `moe_in`. Separately, the w2 GEMM launch passes
   `(moe_in, w2_in)` = (latent, intermediate) as (input, output), which under the
   GEMM's input=K/output=N convention (`runtime/gemm.cuh:229-230`,
   `k_tiles=input_dimension/tile_k`, `cells=output_dimension/16`) is transposed
   against the pack's w2 = [latent N, intermediate K] (`tools/k3_pack.py:621`). The
   determinism/equivalence receipts validate **sharding reconstruction**, not absolute
   numerics — the torch-reference numerical gate is still OPEN
   (`docs/K3_TP4PP4_PREP.md`). Close this with the torch-reference gate before the
   TP16 MoE replay is trusted; the serial replay itself is the right instrument to
   surface it (a transposed w2 would make the 16-rank sum diverge from the full).
3. **Equivalence ≠ torch parity.** The serial-TP16 replay proves the TP16 SHARDING
   (rank sum == full) — exactly this task's goal — but a torch-reference gate remains
   the closure for absolute numerics.

## What I need

- **Coordinator:** clear the idle spark for the §1 packer+sharder run (produce the
  tile_k=32 slice + 16 rank slices).
- **cuda-kernels agent:** the shared iterative-TP1 replay harness (the runner/checker
  shell used for the prior model); I supply the K3 geometry table (§4) and the golden
  refs/tolerances (§5), and extend `k3_tp4_equivalence_check.py` to 16 ranks + the
  head MAX merge.
