# GLM-5.2 PP13 multirow linear-plan root cause and fix

Date: 2026-07-18. Base: origin/main 68af32e (after PR #481/#482). All live
numbers come from retained receipts; the fix below awaits the hardware A/B
defined at the end.

## Root cause

`SparkGlm52ResidentDecodeStageLinearPlanRequiredPreparedActiveRows` pinned
every F32-output cublasLt plan to a prepared M of one row:

```c
return plan->output_is_f32 != 0u ? 1u : active_sequence_count;
```

`SparkGlm52ResidentDecodeStageLaunchPreboundLinearPlan` then took the
mismatch branch:

```c
launch_count = prepared_active_sequence_count == active_sequence_count
    ? 1u : active_sequence_count;
for (row_index = 0u; row_index < launch_count; ++row_index)
    cublasLtMatmul(...);
```

Every dispatch with N active rows issued N separate M=1 cublasLt matmuls per
F32 plan invocation. The F32 plans are the bf16-weight projections with f32
outputs: MoE router logits on every routed layer, DSA scores, and the
MTP-head pieces. All prebound launches funnel through this one function
(router site plus the Maybe wrapper), so the loop fired on every routed
layer of every stage for every multi-row dispatch.

## What it explains

MTP cycle receipt (`docs/GLM52_MTP_CYCLE_DECOMPOSITION_20260718.md`):
six-row verify executes in 56-63 ms per middle rank against 16-17 ms for one
row, 3.579x total, 98% of flight in resident execution. Six rows times the
per-layer M=1 launches is exactly that shape, and the receipt named this
mechanism as one of two candidates.

July-14 B64 receipt (`diagnostics/glm52_b64_api_performance_20260714/`):
58.22 token/s where per-stage kernels are 15-19 ms. 64 lanes times six
routed layers of M=1 router launches adds roughly 60 ms per rank, giving
~76 ms per-rank occupancy. A full 13-packet ring at 76 ms per rank predicts
5 lanes / 76 ms = 66 token/s; 58 was measured. The earlier "ring occupancy"
derivation (reload interval 86-95 ms attributed to handoff overhead) is
REATTRIBUTED: the missing time was GPU execution inflated by this loop, not
host or transport handoff. The host stack was separately exonerated by
`tools/sparkpipe_glm52_pipesim.c`, which drives the real request API,
scheduler, and serving engine against an ideal 13-stage ring in virtual
time and reproduces live B1 exactly (4.0 vs 4.05 token/s) while keeping all
13 cohorts in flight (B64 ideal ~289 token/s at 16 ms stages).

## The fix

1. `RequiredPreparedActiveRows` now returns the padded batch bucket for all
   cublasLt plans, f32 and bf16 alike:
   `min(SparkStagePlanSelectBatchBucketValue(active), maximum)`.
   Bucket floor is 16, so the prepared shape changes only on bucket
   transitions, which also removes the per-dispatch descriptor+autotune
   re-prepare that bf16 plans previously paid whenever the adaptive cohort
   width flipped (5 to 4 at B64).
2. The BF16_ROW_MAJOR launch branch requires prepared == required and issues
   exactly ONE cublasLtMatmul at the prepared M. The per-row loop is deleted
   for this kind. Mismatch fails closed with MODULE_NOT_VALIDATED and the
   existing `linear_plan_active_rows_mismatch` line; there is no silent
   scalar fallback.
3. Padded rows beyond active are computed and ignored. All plan input and
   output buffers are allocated at maximum_active_sequence_count, every plan
   beta is zero, GEMM rows are independent, and every consumer kernel reads
   only active rows, so padded garbage cannot contaminate active output.
   The extra M cost is input reads only (weights dominate at these shapes).
4. The CUBLASLT_FP8_E4M3_ROW_MAJOR kind keeps its previous launch behavior
   unchanged. Its descriptors are never re-prepared by
   `PrepareActiveRows` (bf16-kind only), so fail-closing it would break an
   unmeasured path; it is not in the FP8 production plan set.

Graphs interact correctly: the graph signature mixes the matmul descriptor
pointer, so a bucket transition re-prepares the plan, changes the signature,
and forces recapture, while steady-state same-bucket dispatches replay.

## Also in this change

- Builder early input receive: `SparkGlm52Pp13BuilderSubmitWork` now posts
  the hidden-input receive immediately after the capacity checks and returns
  BUSY before any prep when the upstream frame has not arrived. On success
  it sets `..._DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED`, mapped to
  `..._FRAME_CONTEXT_FLAG_HIDDEN_INPUT_PRERECEIVED`, and the driver's
  `PostInputHiddenTransport` skips its own receive. This deletes the
  retry-time full rebuild (KV table, uploads, metadata launches) that every
  input-gated BUSY previously repeated on the 250 us / 1 ms retry cadence.
  Builder and driver artifacts must ship from the same release; a mixed pair
  double-receives and stalls, which the zero-drift release procedure
  already forbids.
- `tools/sparkpipe_glm52_pipesim.c` (build/sparkpipe_glm52_pipesim
  `<stage_us> <hop_us> <requests> <output_tokens>`): host-stack-in-the-loop
  ring simulator; the regression reference for host-side pipelining.
- `docs/GLM52_MEASURED_STATUS.md`: B4/B16/B64 corrected to MEASURED per the
  retained 20260714 receipts.

## Expected result, to be confirmed by hardware A/B

- Six-row verify per middle rank: from 56-63 ms toward the one-row 16-17 ms
  plus small M growth. MTP cycle from ~890 ms toward ~300 ms; at the
  measured 2.9 committed tokens per cycle that is ~10 token/s at B1 against
  4.05 plain.
- B64 plain decode per-rank occupancy: from ~76 ms toward 15-19 ms;
  aggregate toward the 260-330 token/s pipesim ideal.

## Required live gate

Deploy, then rerun both retained benchmarks unchanged: the MTP cycle prompt
with `SPARKPIPE_PP13_PACKET_TIMING=1` (token SHA-256 must still equal
`9a75ab164284854a2f35d4e24b9ad117727d7755fcbe26b6bb709d480a559990`) and the
July-14 B1/B4/B16/B64 sweep. Accept when verify per-rank execute is under
25 ms and B64 exceeds 150 token/s, or retain the counterexample with the
per-rank tables. Also fix, separately, the Spark2 rank-daemon READY-transition
exit documented in the cycle receipt: startup must wait for READY or retry.
