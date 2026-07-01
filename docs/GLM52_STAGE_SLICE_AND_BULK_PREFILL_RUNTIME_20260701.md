# GLM52 Stage-Slice And Bulk-Prefill Runtime 2026-07-01

This update adds production-facing runtime contracts for the two submission-count
problems that were still visible after the B32/B64 final validation fix.

## Stage-slice submit path

The resident decode-stage module can now be initialized with a
`SparkGlm52ResidentDecodeStageSliceNodeContext`.  The slice context owns an
ordered array of per-layer `SparkGlm52ResidentDecodeStageNodeContext` pointers
and submits the whole slice through one backend completion path.

The CUDA backend entry point is:

```text
SparkGlm52ResidentDecodeStageBackendSubmitStageSlice
```

The SM121 required CUDA entry point is:

```text
SparkGlm52Sm121RequiredDecodeStageLaunchStageSlice
```

The implementation captures/replays one CUDA graph for the whole layer slice
when `enable_cuda_graph_replay` is set on the first layer context. Intermediate
layers run hidden-only; the last layer runs hidden-only unless the slice context
marks it as the final-token stage.  The slice context now carries the first
layer index, so the dense-prefix slice can legally contain layers `0..8` or
`0..9` for the measured PP13 plan while routed-only slices remain capped at
eight layers.

This removes the production requirement to submit one frame per layer. It still
requires the packer/validator to provide per-layer resident contexts and
per-layer or ping-ponged hidden buffers; the old validator path still mutates one
context repeatedly and therefore cannot prove true stage-slice graph replay until
its fixture memory layout is split by layer.

## Bulk-prefill driver mode

The model-driver ABI now has a prefill frame flag:

```text
SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL
```

and a program capability flag:

```text
SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL
```

For a prefill frame, `new_token_count` is interpreted as prompt-token count
instead of MTP draft count. The generated driver skips the decode `max_new_tokens`
limit for prefill frames when the program declares `bulk_prefill`; the resident
module then validates the request against the attached bulk-prefill plan.

The resident firmware now exposes:

```text
SparkGlm52ResidentDecodeStageBulkPrefillPlan
SparkGlm52ResidentDecodeStageBackendSubmitBulkPrefill
SparkGlm52ResidentDecodeStageBackendSubmitStageSliceBulkPrefill
SparkGlm52Sm121RequiredDecodeStageLaunchBulkPrefill
```

The bulk-prefill plan is intentionally a hard contract: production prefill must
provide a stream-ordered tensor-core/block-KV implementation with zero host
staging and zero device memcpy capability bits. If the plan is required but not
attached, module initialization fails; if a prefill frame arrives without a
valid plan, admission and execute reject the frame.  For stage-slice node
contexts, prefill admission validates every layer in the slice and execution
uses one backend completion path for the whole slice instead of accidentally
launching only the first layer.

## Stage-plan and scheduler guardrails

`SparkGlm52StagePlanValidate` now encodes the current GLM-5.2 cut rules used by
the validator: 78 total layers, dense prefix layers 0-2, routed stages starting
at layer 3, routed slices of 1..8 layers, contiguous coverage, and exactly one
final-token stage.

`SparkGlm52StagePlanBuildBalanced` chooses contiguous stages from measured
per-layer costs while respecting the routed-layer-per-stage limit.  The same
module also has the B16/B32/B64 bucket selector for production scheduler code.
The production maximum stage count is now the current spark count: 13.
