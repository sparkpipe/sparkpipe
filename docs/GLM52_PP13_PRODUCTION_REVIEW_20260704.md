# GLM52 PP13 Production Review and Speedup Plan - 2026-07-04

Scope: full host runtime read (`src/`, `include/`), firmware module host layer,
exact-PP13 launch path, validation runner structure, docs, Makefile, tests.
No GPU was available for this review; every millisecond figure below is either
from repo docs, from operator-reported measurements, or a derived roofline with
the arithmetic shown. Derived numbers are labeled as such.

## Verdict

The architecture is right. Compile-don't-interpret, exact module identity,
fail-closed on missing fast paths, one submit per stage, stage-level graph
replay. The 2x from the exact PP13 stage slice (see
`GLM52_PP13_2X_SPEEDUP_ROOT_CAUSE_20260704.md`) was the correct first move:
execution shape beats micro-tuning.

The next factor of 5-10x in aggregate tok/s does not come from kernels.
It comes from tokens-per-weight-sweep. The arithmetic:

## Where the machine sits against the memory roofline

Model constants from
`modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_resident_decode_stage_firmware.h`:

```text
hidden 6144, heads 64, MLA latent 512, rope 64
q_a 6144x2048, q_b 2048x16384, kv_a 6144x576, kv_b 512x28672, o 16384x6144
routed experts 256, top-k 8, expert intermediate 2048
dense intermediate 12288 (layers 0-2), layers 78, first routed layer 3
```

Per routed expert: 3 x 6144 x 2048 = 37.75M params.
Per layer routed weights: 256 x 37.75M = 9.66B params.
NVFP4 storage: 0.5 B/param + UE4M3 scale per 16-group = 0.5625 B/param.

### B64 decode stage (derived)

Expected expert coverage at 64 tokens x top-8 over 256 experts, assuming
independent routing: 1 - (248/256)^64 = 0.87.

```text
6 layers x 9.66B x 0.5625 B x 0.87 = 28.4 GB per stage pass
28.4 GB / 273 GB/s               = 104 ms
```

Operator-measured B64 stage time is ~100 ms. **B64 decode is already at the
NVFP4 expert-weight streaming roofline (within ~5%).** Kernel tuning at B64 is
near-exhausted; the stage pass is a DRAM sweep of the resident experts.
Every additional aggregate tok/s must come from more committed tokens per sweep.

### B1 decode stage (derived)

Active bytes per layer at B1: attention FP8 ~165 MB (q_a 12.6 + q_b 33.6 +
kv_a 3.5 + kv_b 14.7 + o 100.7 M params), routed top-8 NVFP4 ~170 MB, shared
expert FP8 ~38 MB, router ~1.6 MB. Per 6-layer stage: ~2.25 GB.

```text
2.25 GB / 273 GB/s = 8.2 ms floor
measured             16.0 - 20.7 ms  (GLM52_PP13_2X_SPEEDUP_ROOT_CAUSE_20260704.md)
gap                  1.9 - 2.5x
```

So B1 has ~2x of kernel/overlap headroom; B64 has almost none.

## Ordered speedup plan

### L1. Batch past 64 (largest aggregate lever, ~2x at B128, ~3.5x at B256)

At the weight-sweep roofline, aggregate tok/s scales almost linearly with
batch until KV/activation traffic or compute binds; at B128 coverage rises to
~98% so weight traffic grows only ~13% while tokens double. The ceiling is
currently welded in at several independent places (all must move together,
derived from one recipe constant, not hand-edited N times):

```text
firmware.h:74   SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 64
stage_plan.h    SPARK_STAGE_PLAN_BUCKET_B64 ladder ends at 64
stage_plan.c    SparkStagePlanSelectBatchBucket caps at B64
scheduler.h     SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT = BUCKET_B64
required_decode_stage.cu ~15004  exact plan pins batch_bucket to {16,32,64}
```

The AOT side is already ahead of the host: `B12X_AOT_TOKENS` defaults to
`1,2,4,8,16,32,64,96,128` (Makefile), and the generated kernel table is
bucket-keyed by `token_upper_bound`. Required work, in order: AOT-generate and
qualify B128 (then B256) MoE buckets on spark hardware; run
`tools/glm52_stage_bucket_sweep.py` to produce measured B128/B256 layer-cost
profiles; add the profiles to `spark_glm52_stage_plan.c`; lift the bucket
ladder and the exact-plan pin to recipe-driven values. This is exactly the
recipe -> one-time calibration -> firmware flow; nothing here is a runtime
search.

### L2. MTP commit in serving (~1.5-1.9x on top of any batch)

The fused final epilogue already selects and commits MTP draft tokens and
counters (`FUSED_FINAL_TOKEN_TAIL` + `BUILTIN_FUSED_FINAL_TOKEN_EPILOGUE`,
see `GLM52_FINAL_EPILOGUE_AND_PERSISTENT_TRANSPORT_20260703.md`), and
`SPARK_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT` is 2. Accepted MTP
tokens ride the same weight sweep for free. What remains is serving-side:
consume committed counts in the request API completion path and measure real
acceptance on target prompts. At acceptance a1 (+a1*a2 for the second draft),
effective tokens/step = 1 + a1 + a1*a2.

### L3. DSpark draft backend (largest single-stream lever)

The scheduler, request API states, tap ABI, and equal-length verify-lane
packing were completed in the historical DSpark implementation; its own
"current boundary" is the missing C/CUDA draft network under
`SparkGlm52DsparkDraftFunction`. A verify lane of up to 7 tokens costs
approximately one weight sweep, so committed tokens per sweep multiply by
(1 + accepted). This stacks with L1 and L2.

### L4. Measured stage rebalance (+20-25% B1, free)

Pipeline rate = 1 / slowest stage. Current evidence: stage `0:6` = 20.74 ms vs
median 16.6 ms; sum of the 13 published stage times is ~222 ms, so a perfectly
balanced 13-way split is ~17.1 ms -> ~58 tok/s vs today's 48.2. The balancer
already exists (`SparkGlm52StagePlanBuildMeasuredBalancedWithFinalCost`), but
the exact-PP13 CUDA contract hard-pins `first_layer = stage_index * 6` and
`layer_count == 6`:

```text
required_decode_stage.cu ~14984  expected_first_layer_index = stage_index * 6
required_decode_stage.cu ~14997  exact_stage_slice_plan->layer_count != 6u
module.c ~773-806                EXACT_PP13_FIXED6 + layer_count checks
```

Fix: the exact plan carries (first_layer_index, layer_count) from the recipe
and validation checks plan-vs-plan consistency instead of the constant 6.
This is also the prerequisite for the stated product requirement that
pipelines of different lengths (e.g. a 4-stage small-model pipeline) coexist.
Hardware-validated change; the host stage-plan side needs nothing new.

### L5. B1 kernel efficiency (up to ~2x B1, bounded at ~8-9 ms/stage)

The per-phase device profiler already exists
(`SparkValidationPrintExactPp13PhaseProfile`, validation runner ~7746, phases:
attention_norm, attention_projection, dsa_selection, rope_kv_write,
mla_attention, output_projection, post_attention_norm, local_moe,
completion_tail). Run it per layer at B1 and attack the top phase. Candidates,
in expected order: GEMV bandwidth efficiency on the o-projection (100.7M
params, the single largest attention matrix), local_moe top-8 GEMV grouping,
serialized norm/rope short kernels inside the graph. Do not start this before
L1/L4: B1 latency is not the money workload, and stage 0's dense layers will
be rebalanced away first.

### L6. NVFP4 attention weights (minor)

`PROJECTION_RAW_GLM_NVFP4_E2M1` exists in the projection-mode enum. Halving
attention weight bytes saves ~0.5 GB/stage at B1 (~2 ms); negligible at B64
where experts dominate. Do after L5 profiling confirms attention_projection
is bandwidth-bound.

## Prefill: honest status and the extraction plan

What exists and is real: chunked-prefill scheduling with prefix cache and KV
block tables in the host runtime (tests green), the paged bulk-prefill plan
ABI and capabilities in the firmware header, the prompt bridge
(`tools/glm52_prompt_pipeline_input.py`) that refuses to fake a full-prompt
pass, and the C dry-run checker proving token file -> prefill dispatches ->
decode_ready.

The blocker, stated exactly: a runnable prompt-serving binary needs the GLM52
node-context builder factored out of the 9.8k-line validation runner. Today
the only code that can take a resident pack and produce initialized
`SparkResidentDecodeStageNodeContext` + exact stage-slice plans lives in
`modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu`:

```text
~3018  SparkValidationConfigureNode            (node context field wiring)
~3181  SparkValidationEnableLayer3RouterTopK   (+ the Enable* family through ~3680)
~7373  SparkValidationAllocateExactFinalEpilogueWorkspace
~7400  SparkValidationInitializeExactPp13StageSlicePlan
~7496  SparkValidationPrepareExactPp13StageSliceLayer
plus the resident-pack weight binding those functions consume
```

Extraction target: one production translation unit
`modules/glm52_resident_decode_stage/source/spark_glm52_node_context_builder.cu`
exporting build-from-pack for a stage (pack root + stage range + bucket ->
node contexts, slice node context, exact plan, workspaces), consumed by BOTH
the validator and a new `tools/sparkpipe_glm52_serve.c`. One writer. Fixture
loading and oracle comparison stay in the validator. This is code motion plus
an interface, but it is CUDA-side and must be compile- and run-verified on
spark hardware with nvcc; it is deliberately not attempted in this PR from a
CPU-only environment.

The serve binary then composes existing pieces: token file / socket in ->
request API submit -> `ScheduleNext` loop -> exact PP13 submits -> persistent
hidden transport callbacks -> completion stream out. No new scheduling
machinery is needed; this PR hardened the pieces it will sit on.

## What this PR changes (host, test-verified)

1. Prefix-family cohort scan no longer re-hashes each prompt prefix from token
   zero at every block boundary on every `ScheduleNext`. Each slot carries a
   chained-hash memo (`prefix_scan_hashed_token_count`/`prefix_scan_hash`) and
   extends it one 16-token block at a time. Per tick per slot: O(new tokens)
   instead of O(prefix_tokens x boundaries). Hash values are bit-identical by
   the chain construction; pinned by
   `SparkTestPrefixCacheChainedBlockHashMatchesFullHash`.

2. Decode and speculative-verify batch packing no longer rescan all request
   slots once per selected member with a linear exclusion list
   (O(target^2 x capacity)). One pass collects the top-`batch_target` members
   into priority order (bounded insertion, leader pinned at index 0). Same
   selection, same order, same predicates; the old per-member argmax over a
   stable set is exactly top-k of a strict total order. Pinned by
   `SparkTestRequestApiDecodeBatchPacksTopPriorityMembersInOrder` plus the
   pre-existing batching tests.

At 4096 queued requests and target 64 this removes ~10^7 handle comparisons
plus ~10^5 full prompt re-hashes per scheduling tick from the host critical
path that feeds 13 stages.

## Not verified here (epistemic register)

No spark hardware in this environment: no CUDA compilation, no kernel
execution, no timing. The 273 GB/s figure is from
`GLM52_12X_SPARK_PERFORMANCE_MODEL.md`. Expert-coverage math assumes
independent uniform routing; real routing correlation moves B64 coverage and
therefore the roofline estimate by single-digit percent. The attention
projection inventory is read from node-context/pipeline-slot field names and
dimensions in the firmware header; the rope-projection weight (~8-25 MB/layer)
is the one term inferred rather than read. The ~100 ms B64 stage time is the
operator-reported measurement, consistent with the derived roofline.
