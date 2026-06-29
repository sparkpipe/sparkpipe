# GLM52 SM121 FlashInfer B12x vendor integration

SparkPipe vendors the uploaded FlashInfer source under:

```text
third_party/flashinfer/
```

The production target is the SM120/SM121 B12x fused MoE path:

```text
third_party/flashinfer/flashinfer/fused_moe/cute_dsl/b12x_moe.py
third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/
```

FlashInfer attribution is retained in:

```text
third_party/flashinfer/LICENSE
third_party/flashinfer/NOTICE
third_party/flashinfer/licenses/
third_party/flashinfer/SPARKPIPE_PROVENANCE.md
third_party/flashinfer/SPARKPIPE_UPSTREAM_MANIFEST.sha256
```

## Implementation split

The FlashInfer B12x fast path is CuTe DSL Python source. SparkPipe uses that source only during the one-time AOT generation phase:

```text
one-time AOT generation:
    import vendored FlashInfer B12x source
    force SM121/NVFP4/BF16/GLM52 shape
    compile exact CuTe DSL kernels
    export TVM-FFI object files
    generate C/CUDA launch table
    benchmark exact token buckets
    write manifest and runtime link flags

production runtime:
    link strict adapter
    link generated kernel table archive
    link exported TVM-FFI objects
    restore already-converted B12x weights/scales
    launch from C/CUDA only
```

No serving process imports Python, Torch, FlashInfer, or CuTe DSL.

## Exact SparkPipe primitive ABI

The B12x MoE primitive ABI is defined in:

```text
include/sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h
```

The required contract is:

```text
arch: sm_121a
hidden_dimension: 6144
intermediate_dimension: 2048
experts: 256
top_k: 8
quant_mode: nvfp4
output_dtype: bf16
gate_up_order: up_gate
weight_layout: flashinfer_static_view
scale_layout: flashinfer_static_storage
fallback_allowed: false
runtime_backend_selection: forbidden
```

The strict adapter module is:

```text
modules/glm52_sm121_flashinfer_b12x_moe/
```

It requires these backend symbols:

```text
SparkFlashInferB12xCompiledMoeCreate
SparkFlashInferB12xCompiledMoeLaunch
SparkFlashInferB12xCompiledMoeDestroy
```

The compiled backend module is:

```text
modules/glm52_sm121_b12x_compiled_backend/
```

It requires generated AOT symbols:

```text
SparkGlm52Sm121B12xGeneratedManifestInstance
SparkGlm52Sm121B12xGeneratedLaunch
```

If the generated pack is absent, build/link fails. There is no simulated slow implementation.

## One-time AOT command

Run on `spark1` with CUDA, SM121, PyTorch, vendored FlashInfer dependencies, and `nvidia-cutlass-dsl` available:

```bash
make glm52_b12x_prepare_spark_env
make glm52_b12x_aot_compile
```

Then build the native runtime artifacts:

```bash
make glm52_flashinfer_b12x_moe_adapter
make glm52_b12x_compiled_backend NVCC=nvcc
```
