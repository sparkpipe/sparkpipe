# SparkPipe firmware compiler

SparkPipe compiles one model-description JSON into exact model-specific device drivers. The serving runtime receives completed firmware images; it does not interpret model graphs, choose kernels, validate modules, or search for fallbacks.

```text
model-description JSON
        +
validated firmware link-unit library
        |
        v
offline model compiler
        |
        +-- one generated direct-call driver per stage
        +-- only the exact selected link units
        +-- one immutable package receipt
        v
model firmware package
        |
        v
small SparkPipe orchestrator
        -> pre-resolved numeric route
        -> compatible node/replica
        -> exact model-specific program
```

A firmware module is one immutable **link unit**. It may be either:

- one relocatable object (`.o`); or
- one normal static archive (`.a`) containing the host ABI entry point, CUDA objects, device-link output, and private helpers.

A thin archive is rejected because it points outside itself and therefore is not deployable firmware.

The generated driver is the execution plan. The expected high-performance module is usually model-, shape-, layout-, quantization-, and GPU-specific and owns as much fusion, graph capture, stream scheduling, resident state, JIT KV-cache policy, private expert queues, and transport integration as measurement justifies.

The SparkPipe orchestrator performs route admission through the version-3 driver ABI. It can see neutral pressure and cost data, no-host-staging/no-device-memcpy counters, opaque dispatch slots, dispatch generations/cookies, graph replay counters, stale-admission counters, and completions. It does not see KV page layout, CUDA graph topology, MoE queue structure, token-selection internals, or model-specific scheduling hacks.

## Non-negotiable rules

1. The JSON is the sole compile-time model authority.
2. Every stage names one exact target; every operation names one exact module ID.
3. A published module is an exact link unit plus its ABI, target, entry symbols, validation recipe, and ordered validator contract arguments.
4. Validation runs once for a new artifact contract. Reusing it in another model, stage, or manifest does not run validation again.
5. Model compilation fails if any required module artifact is absent, corrupted, or ABI-incompatible.
6. Generated stage drivers contain explicit calls in JSON order and link only selected link units.
7. The runtime loads drivers once, resolves routes once, and submits through numeric handles.
8. Missing support is a build failure. Production execution has no generic or host fallback.
9. Model-specific CUDA programs and aggressive fusion are expected when they improve the complete model stage.

## Build and test

```sh
make -j4 all
make -j4 test
make demo
```

The tests include a two-member static archive whose firmware entry point calls a private helper from another archive member. This proves that a published module is a real library unit rather than one source file disguised as an architecture.

## Offline and runtime products

```text
build/libsparkpipe_common.a    status and filesystem support
build/libsparkpipe_compiler.a  JSON, module publication, and driver generation
build/libsparkpipe_runtime.a   driver loader and orchestrator only
build/sparkpipe_module_publish
build/sparkpipe_model_compile
build/sparkpipe_driver_inspect
```

The serving process links only `libsparkpipe_runtime.a` and `libsparkpipe_common.a`. JSON parsing, SHA-256, validator launching, module publication, and C generation remain offline.

## Publish a validated firmware module

A complete CUDA stage can be built as a normal static archive and published as one module:

```sh
build/sparkpipe_module_publish \
    --library build/modules \
    --module glm.decode.stage0.sm121.profile_a.v1 \
    --target cuda.sm121.gb10 \
    --link-unit build/libglm_decode_stage0_sm121.a \
    --recipe glm.decode.stage0.hardware.v1 \
    --initialize SparkGlmDecodeStage0Initialize \
    --execute SparkGlmDecodeStage0Execute \
    --destroy SparkGlmDecodeStage0Destroy \
    --validator build/validate_glm_decode_stage0
```

Publication performs one content-addressed transaction:

```text
copy exact link unit
    -> hash stored bytes
    -> reuse an existing passing record, or run the validator once
    -> reject validator mutation
    -> make the stored artifact read-only
    -> atomically activate module ID + target
```

Publishing the identical artifact contract again does not execute the validator. Ordered validator arguments are part of that identity, so changing a numerical tolerance, hardware condition, or stage-latency ceiling causes exactly one new validation.

## Exact CUDA firmware sources

`modules/glm52_resident_sparse_mla/` is the first preserved CUDA path converted to the firmware ABI. It is one fixed GLM 5.2 resident sparse-MLA program for the exact target and geometry encoded in its module ID:

```text
cuda.sm121.glm52.resident_sparse_mla.bf16
64 heads
512 latent elements
64 adjacent-pair RoPE elements
2048 selected context tokens
64 tokens per KV block
```

One submission launches a fused RoPE/current-token-KV placement kernel followed on the same caller-owned stream by sparse cached MLA, then emits one external completion. Streams, device buffers, paged KV storage, lookup tables, graph slots, and capacity bounds are bound once through the model-specific node context. The submit path performs no allocation, registry lookup, fallback search, or device-wide synchronization. The module now exposes direct admission/snapshot symbols and optional per-slot CUDA graph replay state.

`modules/glm52_resident_decode_stage/` expands that boundary into a correctness-first resident decode-stage firmware source:

```text
cuda.sm121.glm52.resident_decode_stage.bf16
6144 hidden elements
64 MLA heads
512 latent elements
64 adjacent-pair RoPE elements
2048 native-selected sparse context tokens
256 restricted-vocabulary logits
2 MXFP4 MTP draft tokens
```

The decode-stage source launches attention RMSNorm, raw BF16/FP8 q/kv projections, native sparse-token handling, RoPE plus latent/key-nope/value cache writes, value-head cached attention, real `[6144,16384]` `o_proj`, residual, local MoE progression, final RMSNorm, restricted-vocabulary logits and argmax, MXFP4 E2M1/E8M0 MTP draft logits, verify/commit/rollback counters, optional phase clock markers, and one stream-ordered external completion. The intended production sparse-index mode is preselected driver-owned DSA; the serial top-k mode is debug-only.

Build exact archives from empty module build directories:

```sh
make cuda_glm52_resident_sparse_mla
make cuda_glm52_resident_decode_stage
```

Publication requires an explicit full-stage wall-clock ceiling:

```sh
make cuda_glm52_resident_decode_stage_publish \
    CUDA_ARCH=sm_121 \
    MAX_STAGE_MICROSECONDS=<qualified-maximum>
```

A new archive contract is admitted only if its target-hardware validator passes numerical comparison and every measured submission-to-completion run is within that ceiling. This repository environment has no `nvcc` or compatible GPU, so the CUDA source is not claimed as validated or performant here. If it misses the threshold, it must be optimized rather than published.

## Compile a complete model package

```sh
build/sparkpipe_model_compile \
    --model model.json \
    --library build/modules \
    --output build/packages/glm-profile-a \
    --include include \
    --cc cc \
    --cc-arg -lcudart
```

Omitting `--stage` is the deployment path. The compiler parses the JSON once and builds every stage as one fail-closed package transaction:

```text
build/packages/glm-profile-a/
    model_package.json
    stages/
        stage_000/
            model_driver.so
            spark_model_driver_generated.c
            link_units/<artifact-sha256>.a
        stage_001/
            model_driver.so
            spark_model_driver_generated.c
            link_units/<artifact-sha256>.o
```

`model_package.json` is a build receipt, not a runtime plan. It embeds the original JSON, exact selected artifact identities, generated-program hashes, and driver hashes. The orchestrator never parses it.

For development inspection only, `--stage <name>` builds one stage and emits `compiled_manifest.json` beside it.

The retained generated C is the direct audit of the hot path: explicit calls, no operation loop, no module registry, and no runtime capability negotiation.

## Source layout

```text
include/sparkpipe/       compiler, module, driver, and orchestrator contracts
src/                     active implementation
tools/                   publication, package compilation, and inspection
schema/                  model-description JSON schema
examples/                model-description examples
tests/                   focused contract and end-to-end tests
modules/glm52_resident_sparse_mla/ first exact CUDA firmware source
modules/glm52_resident_decode_stage/ resident decode-stage CUDA firmware source
modules/cuda_candidates/ preserved CUDA source reservoir
```

The preserved CUDA sources are excluded from the active build. A candidate becomes selectable only after it is shaped into an exact firmware link unit, hardware-validated, and published.

`SPEC.md` is the architecture contract. `STATUS.md` states the current implementation boundary. The handoff docs under `docs/` describe the LLM driver ABI and CUDA firmware methods for the next target-hardware optimization pass.
