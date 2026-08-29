# K3 serial-TP16 replay — execution receipt (spark0, 2026-08-17)

Chain executed on spark0 (SSH from the workstation). No downloads; the checkpoint
shards were already present. Status per the directive's report order.

## 1. Pack — DONE
`tools/k3_pack.py <checkpoint> tp16_slice_l0-3.pack 0 4 32` → **87 tensors,
55.3 GB** ("packed 87 tensors, 55337441408 payload bytes"). Manifest:
`config.layers=4, first_layer=0`; every expert tensor carries
`interleave.tile_k=32` (verified per-tensor: `tile_k values {32}`).

## 2. Shards — DONE
`tools/k3_shard.py tp16_slice_l0-3.pack <prefix> 16` → **16 rank packs**
(~1.6 GB each, 25.7 GB total). Rank00 manifest: `tp_degree=16, tp_rank=0`;
w1 = **diagonal** `out_dim=384 × k_dim=224` (tile_k 32, 24 cells × 7 k-tiles);
w2 = **k-split only** `out_dim=3584 × k_dim=192` (224 cells × 6 k-tiles).

## 3. Per-rank replay — RAN, then FAILED at LM_LAUNCH_ERR_SHAPE
Golden (full slice, tp_degree 1) ran and captured 7168 bf16. Rank 0 then failed
during the slice dispatch:

    sparkpipe_k3: slice dispatch failed -41      # LM_LAUNCH_ERR_SHAPE (runtime/launch.h:69)
    SWEEP FAIL -4 (peak_held 1686452352)          # rank 0 shard = 1.69 GB, under budget

Root cause (verified against the rank00 manifest + `runtime/launch.h:176-177`):
the BF16 GEMM requires `input_dimension % tile_k == 0` with BF16 `tile_k=128`,
but the TP16 rank-sliced BF16 input dims are not whole 128-tiles:

| tensor (rank 0) | shape | input (K) | K % 128 |
| --- | --- | --- | --- |
| `dense_down_weight` (layer 0) | [7168, **2112**] | 2112 | **64** → FAILS first |
| `routed_up_weight` (layer 1) | [7168, **224**] | 224 | **96** → would fail |
| `shared_w2_weight` (layer 1) | [7168, 384] | 384 | 0 OK |
| `kda_out_weight` (layer 0) | [7168, 768] | 768 | 0 OK |

`dense_down` (layer 0) is the first non-whole-tile input, so the slice stops
**before any MoE layer** — the w2 path is never reached.

## 4. w2 verdict — MASKED (secondary to the BF16 K-tile blocker)
The w2 sharding inconsistency I flagged (`docs/SERIAL_TP16_K3.md` §6.2) is
CONFIRMED in the produced pack (w2 `out_dim=3584` full, NOT split to 224), and
the layer's w2 GEMM arg order remains transposed vs the pack's
[latent N, inter K] orientation. But the serial replay cannot reach the w2:
the BF16 projections fail at `dense_down` (layer 0) first. So the w2 verdict is
**NOT YET OBSERVABLE** — it is the *second* blocker, behind the BF16 K-tile
failure. Both must be fixed before TP16 MoE numerics can be judged.

## 5. Final sum-vs-golden — NOT REACHED
Ranks produce no partials (dispatch fails before the first token), so there is
no sum to compare. The golden alone is captured.

## Conclusion and the fix
TP16 is blocked by TWO independent issues, in order:

1. **BF16 GEMM K-tile (128) rejects the TP16 rank-sliced input dims.** 2112
   (dense_down), 224 (routed_up), 192 (expert_w2 k / shared paths) are all
   **multiples of 32, not 128**. This is the same class the expert tensors
   already solved with TILE_K=32 (`docs/K3_TP16_REPACK.md`); the BF16
   projections (`dense_down`, `routed_up`) need the same treatment — a
   TILE_K=32 BF16 GEMM variant, or a pack V3 that pads rank dims to 128-multiples.
   (The `runtime/launch.h:170-175` comment already warns this exact class: "a
   trailing partial tile is dropped... wrong output, no crash".)
2. **w2 sharding** (sharder does not output-split w2 to 224; layer passes
   transposed input/output) — still OPEN, masked by #1.

## What I need
- **cuda-kernels agent**: a TILE_K=32 (or 64) BF16 GEMM instantiation for the
  input-dimension-sharded dense/routed projections, mirroring the landed
  INTERLEAVED_B TILE_K=32 wave — OR confirm pack V3 padding as the chosen fix.
- **Coordinator**: the w2 sharder fix (output-split w2 to latent/16) is a
  model-file change (tools/k3_shard.py + layer.cuh arg order) I can draft next.
