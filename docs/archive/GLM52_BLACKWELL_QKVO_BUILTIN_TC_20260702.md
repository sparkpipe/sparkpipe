# GLM52 Blackwell Q/KV/O built-in tensor-core projection path

This pass replaces the descriptor-only quantized Q/KV/O path with a built-in CUDA tensor-core projection implementation for Sparkpipe's resident GLM52 decode stage.

## Implemented path

The required SM121 CUDA module now contains a built-in quantized projection launcher for these linear plan kinds:

- `SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR`
- `SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR`
- `SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR`

The launcher consumes `SparkResidentDecodeStageQuantizedLinearView` from `linear_plan->custom_state`. FP8 plans use the built-in FlashInfer/CUTLASS SM120 groupwise scaled GEMM path:

```text
BF16 activation
    -> stream-ordered FP8 E4M3 row quantization
    -> FP8 E4M3 x FP8 E4M3 block-scaled tensor-core GEMM
    -> BF16 output
```

The 576-output KV-A projection is padded once at plan creation to the required 640-output physical shape. Its stream-ordered output trim is part of the captured graph. NVFP4 and MXFP4 plans retain their separate BF16 WMMA implementation.

## Why this replaces the previous weak path

The previous binding function required caller-provided cuBLASLt descriptors and did not own the CUDA projection implementation. That was not enough for Sparkpipe production because it let a tensor-core plan be mostly a descriptor shell.

An FP8 tensor-core projection plan is executable only when it has:

1. a valid quantized linear view;
2. the built-in scaled-GEMM backend bound at resident initialization; and
3. enough activation, physical-tail, and backend workspace for its declared shape.

Missing FP8 backend state fails loudly. FP8 plans never fall through to the WMMA implementation.

## Scope and honesty

The scaled-GEMM path passed direct B1/B64 numerical gates, CUDA graph capture/replay, and the exact six-layer PP13 stage0 validator with real FP8 weights. The 4-bit WMMA path is a distinct implementation and is not evidence for FP8 performance.
