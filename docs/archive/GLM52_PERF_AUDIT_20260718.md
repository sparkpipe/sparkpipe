# GLM-5.2 PP13 performance audit, main 56d84c7c

Date: 2026-07-18. Scope: verify the merged performance chain will deliver
the stated goals before the ring A/B, and catch anything that silently
defeats it. Host-verifiable claims carry line receipts; archive and
kernel-time claims are marked as ring gates.

## PASS with receipts

1. F32 linear-plan fix intact end to end. `RequiredPreparedActiveRows`
   returns min(batch bucket, maximum) for all cublasLt plans
   (linear_plan.h); the BF16_ROW_MAJOR launch is a single matmul with
   fail-closed validation (sm121 ~13990); the FP8_E4M3 cublasLt kind keeps
   its prior behavior. `PrepareStageLinearPlanRows` runs in every submit
   (builder 9942) and `PrepareMtpLinearPlanRows` on the MTP paths
   (4938/7602/8548).
2. Bucket >= execution rows everywhere. All three packet builders pass
   execution_row_count (lanes x rows) to `SelectExecutionBatchBucket`
   (work_control 292/511, backend 1146) with widening to the next compiled
   bucket; prepare-time and launch-time buckets agree by idempotence of the
   bucket function.
3. Early input receive covers decode AND prefill: prefill chunks route
   through `BuilderSubmitWork` (SubmitPrefillChunk -> SubmitWork), so the
   pre-prep BUSY return applies to both; the driver skips its receive under
   HIDDEN_INPUT_PRERECEIVED.
4. Prefill wave is 256 (`PREFILL_WAVE_TOKENS` =
   MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH), scheduler normalizes multi-block
   steps, row capacity 512 covers the wave.
5. Adaptive MTP floor present and inert on healthy traffic (EMA seeded
   2900, threshold 1150).
6. Submit hot path is host-fill-then-bulk-copy: the per-lane loops in
   PrepareDeviceKvView / UploadWorkDecodePositions / TokenIdsAndEmbedding /
   MtpBudgets / DecodeMetadata / ExpandExecutionKvRows / MtpTreeShadowRows
   stage host buffers and issue 2-4 bulk cudaMemcpyAsync each; roughly
   15-20 CUDA ops per packet, ~1-2 ms host prep, no per-row device loops.
7. #490 KvStoreProgress costs zero when the kv-store tier is disabled
   (null-state early return) and a nonblocking cudaEventQuery when armed.
8. Residentd remains event-driven: driver wake pipe plus input/output
   transport event fds in the poll set.
9. Graph cache holds current + 12 LRU spares keyed on
   (active_sequence_count, specialization signature): producer and verify
   graphs coexist across alternation.
10. Host-stack ceiling reconfirmed by pipesim at stage 16 ms: B64 ideal
    289 tok/s aggregate, 13 cohorts in flight.

## FINDING (fixed in this change): bucket-transition autotune churn

At batched MTP the frame shapes alternate: producer decode at B64 is 64
rows (bucket 64) while six-row verify is 384 rows (bucket 384/512-capped).
`PrepareOne` re-created descriptors, re-ran `SelectAlgorithm` autotune
warmup/measurement GEMMs, and re-allocated workspace on every prepared-M
change, and the descriptor pointer feeds the graph signature, so every
producer<->verify alternation would have re-prepared and recaptured per
rank per cycle. B1 was unaffected (all shapes bucket to 16), so the
current A/B would not have shown it; the first batched MTP run would have.

Fix: prepared M is now grow-only. `PrepareActiveRows` skips any plan whose
prepared count already covers the requirement, and the BF16 launch accepts
prepared >= required, launching one matmul at the prepared M. Padded rows
remain compute-and-ignore (buffers at maximum, betas zero, GEMM rows
independent, consumers read active rows only). Descriptors freeze after
the largest shape has been seen once, signatures stabilize, and the graph
cache then serves both shapes with zero re-prepare. Marginal cost is input
reads at the larger M, sub-millisecond per stage at these dimensions.

## Ring-side gates (not checkable here)

- Archive stage plans must accept the widened buckets for batched verify
  (execution rows 384 at B64 x 6-row tree) and report
  maximum_active_sequence_count accordingly; load-time validation covers
  this, run one B64 MTP smoke after the B1 A/B.
- Isolated kernel times (15-19 ms/stage) and the MoE weight-read model
  behind the prefill law are retained-receipt and model respectively; the
  packet-timing run supplies ground truth.
- The two .cu edits in this change compile only under the ring nvcc gate.

## Unchanged expectations

B1 plain 4.05 -> MTP ~10+ tok/s post A/B; B64 decode toward 260-330;
prefill ~1,000-1,400 serialized at wave 256, per-rank wave times from
SPARKPIPE_PP13_PACKET_TIMING size the pipelining step.
