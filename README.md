# SparkPipe firmware compiler

SparkPipe compiles a model-description JSON into an exact model-specific firmware package. The runtime does not interpret model graphs, choose CUDA kernels, search for fallbacks, or substitute slow reference paths.

```text
model-description JSON
        +
validated module library
        |
        v
offline compiler
        |
        +-- generated direct-call stage driver
        +-- exact selected link units only
        +-- immutable package receipt
        v
model firmware package
        |
        v
small runtime orchestrator
```

## Rules

1. The JSON is the compile-time model authority.
2. Every stage names one target and every operation names one exact module ID.
3. A published module is a real link unit with ABI, target, entry symbols, validation recipe, and validator arguments.
4. Missing required CUDA is a link or validation error.
5. Production execution has no generic CUDA, host, scalar, or library fallback.
6. Warmup, autotune, layout conversion, tactic choice, and graph bucket choice are offline qualification work. Serving restores a qualified package and captures only process-local runtime objects.

## Build and test

```sh
make -j4 all
make -j4 test
```

## Offline products

```text
build/libsparkpipe_common.a
build/libsparkpipe_compiler.a
build/libsparkpipe_runtime.a
build/sparkpipe_module_publish
build/sparkpipe_model_compile
build/sparkpipe_driver_inspect
```

The serving process links only the runtime/common libraries plus the compiled model driver. JSON parsing, validation, artifact publication, SHA-256, and C generation remain offline.

## GLM 5.2 SM121 decode stage

`modules/glm52_resident_decode_stage/` contains the SparkPipe ABI boundary for the GLM 5.2 resident decode stage. It does not contain a slow substitute for the required CUDA implementation.

The required external CUDA module is:

```text
spark.glm52.sm121.required_decode_stage.b12x_fused.v1
```

It must export:

```text
SparkGlm52Sm121RequiredDecodeStageInitialize
SparkGlm52Sm121RequiredDecodeStageLaunch
SparkGlm52Sm121RequiredDecodeStageQuiesce
```

Build the archive:

```sh
make -C modules/glm52_resident_decode_stage archive CUDA_ARCH=sm_121a
```

Package with the required CUDA link unit:

```sh
make -C modules/glm52_resident_decode_stage package \
    CUDA_ARCH=sm_121a \
    MAX_STAGE_MICROSECONDS=<qualified-limit> \
    REQUIRED_CUDA_CC_ARGS="--cc-arg /path/to/libglm52_sm121_required_decode_stage.a" \
    GLM52_REQUIRED_CUDA_LINK_ARGS="/path/to/libglm52_sm121_required_decode_stage.a"
```

If the required library is omitted or does not define the required symbols, the package or validator link fails.

## GLM 5.2 SM121 B12x AOT primitive

The FlashInfer B12x fused-MoE source is vendored only for offline generation. Build/qualification may use Python, Torch, and CuTe DSL; serving must not. The runtime links the generated C/CUDA artifacts and required symbols only.

One-time AOT generation on `spark1`:

```sh
tools/glm52_b12x_prepare_spark_env.sh
. "$HOME/.config/sparkpipe/glm52_b12x_aot_env.sh"
./tools/glm52_b12x_aot_compile.py \
    --tokens 1,2,4,8,16,32,64,96,128 \
    --warmup 5 \
    --iterations 20 \
    --benchmark \
    --output-dir build/glm52_b12x_aot
```

The AOT tool defaults to static/micro SM121 buckets. Dynamic export is only
enabled with `--allow-dynamic`; it is not part of the production recipe unless
it has a passing target-hardware qualification.

Build the strict primitive adapter and compiled backend:

```sh
make glm52_flashinfer_b12x_moe_adapter
make glm52_b12x_compiled_backend NVCC=nvcc
```

If `build/glm52_b12x_aot/generated/spark_glm52_sm121_b12x_generated_kernel_table.cu`, `tvm_ffi_flags.mk`, or `objects/*.o` is missing, the generated backend target fails. The serving link must include the generated archive plus runtime libraries listed in `build/glm52_b12x_aot/generated/runtime_link_args.txt`.
