# GLM-5.2 PP13 4-bit/8-bit runtime implementation notes, 2026-07-01

This branch keeps the production target at exactly 13 sparks. PP14 planning is intentionally not exposed because the current deployment only has 13 spark endpoints available.

## Implemented runtime variants

The resident decode-stage firmware now has an explicit model-quantization contract:

- `SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT`
- `SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT`

`AUTO` remains accepted for compatibility, but production contexts can set `SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION` to reject ambiguous contexts.

## 4-bit path

The 4-bit path uses NVFP4/MXFP4 projection-plan modes instead of pretending that raw BF16 projection pointers exist:

- raw Q/KV/O projection modes accept quantized prebound tensor-core plans;
- the CUDA reference fallback can decode NVFP4/MXFP4 packed nibbles through `SparkGlm52ResidentDecodeStageQuantizedLinearView`;
- routed NVFP4 layers remain bound to the existing FlashInfer B12x fused-MoE plan;
- module validation rejects an FP8-routed layer under the NVFP4 model-quantization mode.

## 8-bit path

The 8-bit path uses FP8 E4M3 projection-plan/direct-weight support plus an explicit FP8 MoE plan:

- raw Q/KV/O projection modes accept FP8 direct tensors or prebound FP8 tensor-core plans;
- routed FP8 layers use `SparkGlm52ResidentDecodeStageFp8MoePlan`;
- the CUDA path runs the router, dispatches the FP8 expert plan, and applies the residual add;
- module validation rejects an NVFP4-routed layer under the FP8 model-quantization mode.

## Stage-slice and prefill behavior

Stage-slice decode and stage-slice prefill both have one-submit paths. Bulk-prefill inside a stage slice now uses `SparkGlm52Sm121RequiredDecodeStageLaunchStageSliceBulkPrefill`, so the backend no longer submits one prefill operation per layer. CUDA graph signatures include model-quantization mode, linear-plan metadata, quantized payload/scale pointers, bulk-prefill plans, and FP8-MoE plans.

The remaining performance-critical code is the plan-provided tensor-core implementation. The reference quantized kernels are correctness/fallback paths; production performance should attach prebound tensor-core projection plans, FP8-MoE plans, NVFP4 B12x plans, stage-slice plans, and bulk-prefill plans.

## Current measured-balanced PP13 estimate

Using the measured 2026-07-01 profile and 13 sparks:

- B64 balanced PP13 slowest-stage estimate: 78.741333 ms, about 813 tok/s.
- B32 balanced PP13 slowest-stage estimate: 63.119000 ms, about 507 tok/s.

These estimates are stage-balance ceilings. They assume the per-stage one-submit path is used and that the attached plan launch functions are actually fast enough to reach the modeled per-layer costs.
