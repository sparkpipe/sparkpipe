# GLM52 SM121 B12x AOT native runtime path

SparkPipe permits Python, PyTorch, FlashInfer, and CuTe DSL only in the one-time qualification/build phase. The serving process does not import Python, Torch, FlashInfer, or CuTe DSL. It links the generated TVM-FFI object pack, the generated C/CUDA launch table, and the strict C adapter.

## Build/qualification phase

Run this on the target `spark1` software stack:

```bash
tools/glm52_b12x_prepare_spark_env.sh
. "$HOME/.config/sparkpipe/glm52_b12x_aot_env.sh"
./tools/glm52_b12x_aot_compile.py \
  --tokens 1,2,4,8,16,32,64,96,128 \
  --warmup 5 \
  --iterations 20 \
  --benchmark \
  --output-dir build/glm52_b12x_aot
```

The environment prep script installs the Python 3.12 development headers into a
user-owned sysroot. This satisfies Triton's AOT launcher compilation without
sudo or system package mutation.

The AOT compiler defaults to static/micro SM121 buckets. The dynamic backend is
behind `--allow-dynamic`; do not publish a dynamic bucket unless it passes on
the target GPU.

The tool uses vendored FlashInfer B12x CuTe DSL source to compile exact GLM52/SM121 buckets. It emits:

```text
build/glm52_b12x_aot/generated/aot_manifest.json
build/glm52_b12x_aot/generated/objects/*.o
build/glm52_b12x_aot/generated/spark_glm52_sm121_b12x_generated_kernel_table.cu
build/glm52_b12x_aot/generated/tvm_ffi_flags.mk
build/glm52_b12x_aot/generated/runtime_link_args.txt
```

The generated manifest contains the source-export hashes, exact shape, selected backend per token bucket, timing record, and manifest hash. The model recipe must use that manifest hash in `kernel_manifest_hash_low64`; otherwise the C backend refuses initialization.

## Runtime build phase

Build the strict adapter and compiled backend:

```bash
make glm52_flashinfer_b12x_moe_adapter
make glm52_b12x_compiled_backend NVCC=nvcc
```

The generated backend target intentionally fails unless the AOT output exists and contains exported FlashInfer kernel objects.

The production link must include:

```text
build/modules/glm52_sm121_flashinfer_b12x_moe/libglm52_sm121_flashinfer_b12x_moe_adapter.a
build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_compiled_backend.a
build/modules/glm52_sm121_b12x_compiled_backend/libglm52_sm121_b12x_generated_kernel_table.a
runtime libraries listed in build/glm52_b12x_aot/generated/runtime_link_args.txt
```

The runtime boundary is the required backend ABI:

```text
SparkFlashInferB12xCompiledMoeCreate
SparkFlashInferB12xCompiledMoeLaunch
SparkFlashInferB12xCompiledMoeDestroy
```

The compiled backend requires the generated manifest and launch table:

```text
SparkGlm52Sm121B12xGeneratedManifestInstance
SparkGlm52Sm121B12xGeneratedLaunch
```

If the AOT output is missing, the generated archive target fails. If the backend is linked without generated artifacts, unresolved symbols remain. There is no fallback implementation.

## Static contract

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
runtime_backend_selection: forbidden
fallback_allowed: false
```

## Runtime-restored weights and scales

Serving does not perform Python/Torch layout conversion. The one-time preparation phase must restore resident buffers that already match FlashInfer's static views:

```text
w1_weight_fp4_static_view:        [4096, 6144, 256] logical FP4 view
w1_scale_static_storage_ue4m3:    FlashInfer-converted static scale storage
w1_alpha_fp32_by_expert:          [256]
fc2_input_scale_fp32_by_expert:   [256]
w2_weight_fp4_static_view:        [6144, 2048, 256] logical FP4 view
w2_scale_static_storage_ue4m3:    FlashInfer-converted static scale storage
w2_alpha_fp32_by_expert:          [256]
```

The model loader must fail if it only has original checkpoint/modelopt/MMA layouts and no converted resident B12x layout.
