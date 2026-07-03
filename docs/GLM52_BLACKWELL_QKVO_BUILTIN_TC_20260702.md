# GLM52 Blackwell Q/KV/O built-in tensor-core projection path

This pass replaces the descriptor-only quantized Q/KV/O path with a built-in CUDA tensor-core projection implementation for Sparkpipe's resident GLM52 decode stage.

## Implemented path

The required SM121 CUDA module now contains a built-in quantized projection launcher for these linear plan kinds:

- `SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR`
- `SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR`
- `SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR`

The launcher consumes `SparkGlm52ResidentDecodeStageQuantizedLinearView` from `linear_plan->custom_state` and performs the projection with CUDA WMMA BF16 tensor-core tiles:

```text
BF16 activation tile + dequantized FP8/NVFP4/MXFP4 weight tile -> BF16/F32 output tile
```

The kernel computes one output-column tile for up to 64 active sequences per CTA. Four warps share the same dequantized weight tile, so B64 decode reuses the weight dequantization across the full graph bucket instead of redoing it once per 16-sequence tile.

## Why this replaces the previous weak path

The previous binding function required caller-provided cuBLASLt descriptors and did not own the CUDA projection implementation. That was not enough for Sparkpipe production because it let a tensor-core plan be mostly a descriptor shell.

Now a tensor-core projection plan is executable when either:

1. it supplies an external custom launch function, or
2. it supplies a valid quantized linear view and uses the built-in Sparkpipe CUDA launcher.

If neither is present, module validation rejects the plan.

## Scope and honesty

This is real CUDA tensor-core code, not the old scalar reference kernel. It is still a conservative WMMA/dequantize-to-BF16 implementation rather than a native Blackwell FP4/FP8 MMA mainloop. The next kernel step is to replace the WMMA dequantized-weight mainloop with native Blackwell block-scaled FP8/FP4 MMA while preserving the same plan validation and launch surface.
