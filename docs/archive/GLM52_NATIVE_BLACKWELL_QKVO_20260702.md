# GLM52 native Blackwell Q/KV/O projection path, 2026-07-02

This pass replaces the previous dequantize-to-BF16 tensor-core projection implementation with a native SM12x low-precision MMA path for Q/KV/O projection plans.

## Implemented

The required SM121 CUDA module now owns native Blackwell projection kernels for these production plan kinds:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR
```

The launch path is now:

```text
BF16 active hidden rows
    -> per-sequence activation quantization into native E4M3 or E2M1 tiles
    -> activation scale staging in native scale-byte format
    -> native SM12x mma.sync Q/KV/O projection mainloop
    -> BF16/F32 output
```

The old WMMA BF16-dequantized projection kernel is no longer selected by the built-in Blackwell tensor-core launch path.

## Native MMA atoms

The FP8 path uses the SM12x low-precision `mma.sync` atom:

```text
mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
```

The NVFP4 path uses the SM12x block-scaled FP4 atom:

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
```

The MXFP4 path uses the corresponding E8M0-scale atom:

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0
```

## Workspace contract

Built-in native projection plans now require device workspace. The resident module validates that every built-in tensor-core Q/KV/O plan has enough workspace for:

```text
quantized activation payload
aligned scale offset
activation scale bytes
```

The required CUDA module exposes:

```c
SparkGlm52Sm121RequiredDecodeStageCalculateBlackwellNativeQuantizedTensorCoreWorkspaceBytes(...)
```

so model loading can size each projection plan workspace deterministically.

## No silent fallback

A tensor-core plan without either an external custom launch function or a valid built-in quantized view plus native workspace is rejected. The built-in path launches the native SM12x MMA kernels; it does not fall back to the CUDA scalar reference kernels or to the old dequantized-BF16 WMMA kernel.

## Validation

Local CPU/module tests passed:

```text
make clean
make -j2 test
```

The CUDA target still cannot build in this container because `nvcc` is unavailable:

```text
make cuda_glm52_resident_decode_stage
cuda_glm52_resident_decode_stage skipped: nvcc unavailable
```

Spark2 still needs to compile/debug the inline PTX operand layouts and measure the native kernels.
