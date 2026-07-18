# GLM-5.2 PP13 prefill wave pipelining specification

Date: 2026-07-18. Status: SPEC. Wave 256 is merged and host-proven; this
document defines the multi-inflight wave change that follows the linear-plan
hardware A/B.

## Measured baseline and model

Live long-prompt prefill was ~100 token/s. Decomposition: waves were 64
tokens (`PREFILL_WAVE_TOKENS` pinned to one KV block) and strictly
serialized (slot state RUNNING_PREFILL admits one dispatch; wave N+1 enters
rank0 only after wave N's final event), and every 64-row wave paid the F32
linear-plan M=1 loop (~60 ms/rank pre-fix). One wave per ring traversal:
64 / ~0.7s.

Per-rank prefill wave time is bounded below by MoE weight reads: 6 layers x
256 experts x 3 x 6144 x 2048 FP8 = ~59 GB/rank; a full expert sweep at
~273 GB/s = ~215 ms. Effective bytes depend on routing skew (uniform-random
would give ~86% coverage at 64 tokens; real text is far more skewed), so the
per-rank number must come from `SPARKPIPE_PP13_PACKET_TIMING=1` at waves 64
and 256 before pipelining targets are set. Law once pipelined:
tokens/s ~= wave_tokens / per_rank_wave_time, compute crossover near
2k-token waves.

## Already merged

- Wave 64 -> 256: `SPARK_GLM52_PP13_SERVICE_BACKEND_PREFILL_WAVE_TOKENS` now
  equals `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH`. The whole
  chain already supported it: scheduler default step is 256 and normalizes
  arbitrary multi-block steps, wire packet cap is 256, builder staging is
  256, execution row capacity is 512 (64 lanes x 8 rows), archive plans are
  validated to 384+ rows by B64 MTP. Host-proven by
  `SparkTestRequestApiPrefillWaveSpansMultipleKvBlocks` (300-token prompt ->
  waves (0,256)+(256,44) -> decode).
- The M=1 fix applies to prefill frames (same `LaunchPreboundLinearPlan`
  path; bucket(256)=256, one GEMM per plan).

## The pipelining change

Verified seam: the scheduler is stateless per admit; the wave offset is the
`computed_prompt_token_count` the request API passes
(`decision->scheduled_prompt_token_offset = computed_prompt_token_count`,
scheduler.c:1078). Everything therefore localizes to the request API plus
completion tolerance:

1. Slot gains `dispatched_prompt_token_count` (u32). Invariant:
   computed <= dispatched <= prompt_token_count; equal when idle.
2. `SchedulePrefill` eligibility: a slot in RUNNING_PREFILL with
   dispatched < prompt_token_count remains prefill-schedulable, capped by
   `SPARK_GLM52_REQUEST_API_PREFILL_INFLIGHT_WAVE_LIMIT` (12: one below the
   ring depth so decode injection is never fully starved). The admit request
   passes dispatched as the offset source; on accept, dispatched +=
   scheduled_prompt_token_count.
3. KV and prefix checks (`ReservePrefillResidentKvBlocks`,
   `PrefillCachedBlocksAreResident`, `EnsurePrefillSlotKvCapacity`) switch
   from computed to dispatched so wave N+1's blocks are allocated while N is
   in flight. Prefix-cache hash chaining is deterministic from token
   content, so parent/result hashes for step N+1 are computable at dispatch.
4. Completion: waves of one sequence complete in order (daemon per-sequence
   FIFO plus single hidden-transport connection guarantee ring order; the
   backend pending table is keyed by request+position). CompleteDispatch
   advances computed by the wave's scheduled count and transitions to
   READY_DECODE only when computed == prompt AND no prefill dispatches
   remain in flight for the slot (track `inflight_prefill_dispatch_count`).
5. Failure/cancel: any wave failure or CancelDispatch rewinds
   dispatched = computed, releases blocks for the rewound range, and drains
   remaining in-flight waves of that slot as no-ops (completion for a
   position > computed after rewind is dropped with a counted trace, not an
   error). CancelRequest waits for inflight_prefill_dispatch_count == 0.
6. Scheduler admission bookkeeping needs no change: each wave is an
   independent admission against `spark_inflight_counts`
   (queue_depth_per_spark 14 bounds total ring occupancy; a saturating
   prefill can hold up to the wave limit, decode keeps at least two slots).
7. Backend/residentd need no change: the work queue is 256 deep, prefill
   packets ride the same per-sequence FIFO, and the single
   `pending_prefill_active` parking slot only engages at queue-full.

Causality: wave N+1 at layer L reads only KV that wave N wrote while
passing layer L on the same rank; per-rank in-order execution is exactly
the daemon FIFO. No cross-rank barrier is required.

## Expected result

Pipelined wave-256 throughput = 256 / per_rank_wave_time(256). At the
weight-sweep bound that is ~1,200 token/s; with measured routing skew,
1,500-2,500 is plausible. A later dispatch-cap hardfork (256 -> 1024:
packet arrays, builder staging, host_prefill stride) multiplies again
toward the ~5-10k compute/bandwidth crossover.

## Gate

Deploy after the linear-plan A/B. Measure: long-prompt (>= 4k tokens)
prefill token/s at wave 256 serialized (this release) and pipelined (next),
with per-rank wave times from packet timing, plus decode-latency impact of
a concurrent saturating prefill. Accept pipelining when prefill >= 4x the
serialized-256 number and B64 decode P95 step time degrades < 25% under
concurrent prefill; otherwise retain the counterexample tables.
