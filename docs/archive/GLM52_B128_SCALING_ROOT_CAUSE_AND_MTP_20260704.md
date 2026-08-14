# GLM52 B128 Scaling Root Cause, Kernel Fix, and MTP Commit Path - 2026-07-04

## Measured refutation

B128 exact-PP13 stage sweep (operator-reported, best-of runs):

```text
stage 0:6   176.7 ms      stage 6:6   189.1 ms      stage 12:6  189.7 ms
B64 reference: ~100 ms per stage
```

Roughly 1.9x the time for 2x the tokens: per-token stage cost is flat from
B64 (1.56 ms/token) to B128 (1.48 ms/token). Batching stopped paying past
B64.

**Correction of GLM52_PP13_PRODUCTION_REVIEW_20260704.md:** that review
predicted ~2x aggregate at B128 from expert-coverage saturation (87% -> 98%).
The coverage arithmetic holds for the AOT MoE path, which reads each
activated expert once per pass. The error was assuming the QKVO and dense-MLP
projections were also weight-amortized across the batch. They were not, and
they dominate the marginal cost, so the prediction was wrong as stated.

## Root cause (structural, in code)

`SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearKernel` (the
built-in quantized tensor-core path used by all raw attention projections and
the dense MLP) ran as:

```text
grid = (output_dim / 16 N-tiles, ceil(batch / 16) M-tiles)
block = 32 threads (one warp), one 16x16 output tile
per K-slab: the block loads AND dequantizes its full 16-wide weight strip
```

Every 16-row M-tile is an independent block that re-reads and re-dequantizes
the identical K x 16 weight strip. Projection weight DRAM traffic and dequant
compute therefore scale with ceil(batch/16): 4x at B64, 8x at B128. The
per-projection "wide tile" variants (32/64 rows, still one warp) cut the
re-read but collapsed grid parallelism on small-N matrices (kv_a has only 36
N-tiles) and were measured slower, hence commit 274bd05's revert.

This fits the observed numbers: MoE cost is ~fixed past coverage saturation,
projection cost is ~linear in batch, and their sum reproduces both the
per-token flatness and stage `0:6` (3 dense + 3 routed layers) flipping from
slowest at B1 to fastest at B128.

## The fix in this change

`SparkGlm52ResidentDecodeStageSupportedQuantizedBf16WmmaLinearBatchKernel`:
256 threads (8 warps) per block; per K-slab the block cooperatively
dequantizes ONE 16x16 weight tile into shared memory and each warp mma-syncs
it against its own 16-row input fragment, so a block covers 128 rows with the
weights read and dequantized once. `grid.y = ceil(batch / 128)`.

```text
projection weight traffic + dequant:  ceil(B/16)x  ->  ceil(B/128)x
B128: 8x less;  B<=128: read-once;  B<=16: unchanged original kernel
```

The launcher routes `active_sequence_count > 16` to the batch kernel; the
qualified B1..B16 path is byte-identical. Wide-tile variants are deleted.

For the B256/512/1024 ladder (now open after the bucket-limit lift): at
B > 128 the kernel runs ceil(B/128) M-blocks per N-tile and the per-strip
re-read returns at that granularity, because CUDA schedules blockIdx.x
fastest and a full weight sweep separates same-N M-blocks, evicting the
strip from L2. The knob is grid rasterization - make M the fast grid
dimension so same-N M-blocks run adjacently and hit L2 on the ~100-400 KB
strips - plus, if needed, an in-block M-chunk loop. Measure at B256 first;
below B256 the expert sweep dominates either way.

## Expected numbers after the fix (derived; sweep required)

Routed layer at B128: attention ~165 MB + shared expert ~38 MB + routed MoE
~5.3 GB (98% coverage NVFP4) once each -> ~20 ms bandwidth floor plus MMA
compute -> ~21-23 ms/layer, ~130-140 ms/stage, **~950 tok/s at B128**, with
scaling restored because the fixed expert sweep now dominates: ~1800 tok/s at
B256 if compute stays subdominant. The B256/512/1024 ladder is worth building
only after this kernel lands; codex's B128 stage-plan cost profile
(`SparkGlm52StagePlanLoadEstimatedB128CostProfile`, hardcoded 2x B64) must be
replaced by a measured sweep.

## MTP commit path (this change, host contract complete)

The fused final token tail already commits MTP drafts on device every decode
pass (`LaunchFusedFinalTokenTail` in the stage submit flow writes
`mtp_committed_token_ids`, accept mask, event counters). The host stack
rejected the result: serving pinned DECODE_BATCH lanes to token_counts == 1
and `FinishSlotAfterDecode` hardcoded one budget unit.

Now: `CONFIGURATION_FLAG_MTP_COMMIT` arms decode dispatches with
`DISPATCH_FLAG_MTP_COMMIT` and `mtp_draft_token_budget = min(2, batch minimum
remaining budget - 1)` (skipped when any lane sets DISABLE_SPECULATION -
all-or-nothing per dispatch, the deliberate v1 simplification). Serving
validates per-lane commits in [1, 1 + budget] and copies actuals into
`dispatch->decode_committed_token_counts`; CompleteDispatch consumes the
committed amount per lane, re-validated against the slot's remaining budget.
The serving adapter reads back committed draft ids up to the first CANCELLED
sentinel, budget-capped. At acceptance a1 (+a1*a2), effective tokens per
weight sweep = 1 + a1 + a1*a2, multiplicative with the batch fix.

## Verification register

Host: full `make test` green including
`SparkTestRequestApiMtpCommitConsumesMultiTokenBudget` and
`SparkTestServingMtpCommitStreamsMultiTokenLanes`.

CUDA: clang CUDA syntax gate at sm_90 only (no GPU, no nvcc in this
environment). REQUIRED on spark2 before relying on any of this: (1) oracle
numeric validation of the batch projection kernel via the validation runner's
projection compare, (2) B64 + B128 stage re-sweep and a measured B128 cost
profile, (3) MTP end-to-end against the real epilogue including acceptance
counters, (4) confirm decode CUDA-graph capture replays the new kernel (the
batch kernel is a plain stream launch inside the captured region; grid shape
depends only on the bucket, so capture-per-bucket is unaffected - verify).

## Still open (unchanged from the production review)

The dspark draft network remains the stated boundary for the largest
single-stream lever. The node-context builder extraction from the validation
runner remains pending; the serving adapter covers dispatch-side execution
but building node contexts and exact plans from a resident pack still lives
only in the validator.
