# GLM52 JIT KV backend and Blackwell Q/KV/O plan binding, 2026-07-02

This pass keeps the work narrow and connected.

## Finished: internal async JIT KV prefetch backend

Sparkpipe now has a concrete async KV prefetch backend behind the request API's internal JIT scheduling path.

The backend is initialized once with a backing store and then attached to `SparkRequestApiConfiguration` through:

```c
SparkRequestApiConfigurationUseAsyncKvCachePrefetchBackend(...)
```

Callers still submit normal requests. They do not inspect the queue and do not issue prefetches. Sparkpipe builds the critical prefetch plans for the selected near-future dispatch, starts/polls the backend internally, and greenlights dispatch only after the needed KV blocks are resident.

The backend supports:

```text
13-lane prefetch plans
multiple in-flight prefetch IDs
hash-addressed source entries using parent/block/content hash
memory backing stores for unit/integration testing
POSIX file-descriptor backing stores for NVMe-like block files
key/value block copies
per-poll lane progress
request-API async start/poll integration
generation-safe residency marking through the existing KV arena
```

The source map matters for external KV cache persistence. A prefetch block can be resolved by:

```text
content_hash
or block_hash + parent_hash
or physical block index fallback
```

That matches the intended hash-chain model:

```text
prev_hash + token_span -> block_hash / content_hash -> persistent KV block source
```

## Finished: no-fallback Blackwell quantized Q/KV/O launch binding

The required SM121 CUDA module now exports:

```c
SparkGlm52Sm121RequiredDecodeStageLaunchBlackwellQuantizedTensorCoreLinearPlan(...)
SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedTensorCoreLinearPlan(...)
SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedProjectionPlans(...)
```

These are for tensor-core plan kinds only:

```text
TENSOR_CORE_FP8_E4M3_ROW_MAJOR
TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR
TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR
```

The path is intentionally not a reference fallback. It requires:

```text
matching quantized linear view
cuBLASLt handle
matmul descriptor
input/weight/output layouts
selected algorithm
workspace state
```

If those are absent, the launch fails with `SPARK_STATUS_INVALID_ARGUMENT` rather than falling back to scalar/reference kernels. This keeps production honest while letting Spark2 bind real Blackwell/cuBLASLt FP8/FP4 descriptors for Q/KV/O projections.

## Validation

Local CPU tests passed:

```text
make clean
make -j2 test
```

CUDA target behavior in this container:

```text
make cuda_glm52_resident_decode_stage
cuda_glm52_resident_decode_stage skipped: nvcc unavailable
```

The new CUDA launch binding still needs Spark2 `nvcc` compile and hardware validation.
