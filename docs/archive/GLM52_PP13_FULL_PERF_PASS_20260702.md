# GLM-5.2 PP13 full performance pass, 2026-07-02

This pass keeps the debugged PR #55 PP13 production timing plan as the source of truth:

```text
0:6, 6:6, 12:6, 18:6, 24:6, 30:6, 36:6,
42:6, 48:6, 54:6, 60:6, 66:6, 72:6
```

The measured B64 ceiling before transport remains anchored to the observed slowest fixed stage:

```text
0:6 = 50.660288 ms
64 / 0.050660288 ~= 1263 tok/s
```

The final stage remains anchored to the production timing evidence:

```text
72:6 hidden-only    = 46.268929 ms
72:6 final-token    = 46.449792 ms
```

## Implemented performance work

### Exact six-layer AOT stage launcher

`SparkResidentDecodeStageExactStageSlicePlan` is now ABI v2 and carries a production AOT launch callback:

```c
void *launch_function;
void *opaque_state;
```

`SparkGlm52Sm121RequiredDecodeStageLaunchExactPp13StageSlice` validates the exact `stage_index * 6` layer range, B16/B32/B64 batch bucket, device-only hidden handoff, exact final-stage placement, optional Q/KV branch overlap resources, and the AOT callback when the plan advertises `SPARK_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_AOT_STAGE_LAUNCH`.

When the callback is present, the exact six-layer stage is launched through that stage-specific path and captured as one stage-level CUDA graph. When the callback is absent, the validated fallback still executes the six resident layer contexts inside one exact-stage graph path.

### Paged/chunked prefill attention

`SparkResidentDecodeStagePagedPrefillPlan` is now ABI v2 and includes prompt Q, rotated Q/rope, first-block offsets, and prompt attention output pointers. `SparkGlm52Sm121RequiredDecodeStageLaunchPagedChunkPrefill` now supports `SPARK_RESIDENT_DECODE_STAGE_BULK_PREFILL_CAPABILITY_PAGED_ATTENTION` and launches an online-softmax paged attention kernel over the prompt chunk using vLLM-style block-table metadata.

The scheduler-side chunked prefill and prefix-cache accounting remain block aligned. The CUDA side can now consume the same metadata instead of being limited to metadata staging and prompt-hidden copies.

### 4-bit and 8-bit tensor-core projection plan kinds

The linear plan ABI accepts production tensor-core plan kinds for both model variants:

```c
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_NVFP4_E2M1_ROW_MAJOR
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_MXFP4_E2M1_ROW_MAJOR
```

The required CUDA launch path maps these to the FP8/NVFP4/MXFP4 quantized linear reference view when no production launch callback is attached, and calls the plan callback when one is attached. Firmware validation accepts the new tensor-core plan kinds as quantized production projection plans.

### Persistent hidden transport backend

The hidden transport ABI now has a persistent ring backend:

```c
SparkHiddenTransportPersistentRingGetInterface(...)
SparkHiddenTransportPersistentRingGetStatistics(...)
```

It advertises the production transport capabilities, including persistent connection, device-pointer IO, stream-ordered completion, no host staging, no device memcpy, and native batched send/post-receive callbacks. This is an in-process production contract backend for validating the runtime flow; hardware deployment should replace the ring internals with the actual inter-spark device transport.

### Prefill/decode interleaving scheduler policy

The scheduler now enables prefill/decode interleaving by default. A prefill admission reserves one queue slot per spark for decode when the spark queue depth is greater than one. Decode requests can bypass active prefill work and are marked in the decision/dispatch flags.

New scheduler accounting records interleaved prefill admissions and decode bypass admissions.

### Prefix-cache block manager

Added `SparkPrefixCache`, a block-level prefix cache manager with prompt block hashing, parent-hash chaining, block lookup, block commit, per-sequence release, reference counts, and LRU-style reusable victim selection. Lookup intentionally keeps the last prompt token scheduled for recomputation when the full prompt hits cache.

### MoE and final-stage fusion hooks

The exact PP13 plan advertises optional stage-level MoE fusion and final-token tail fusion capabilities:

```c
SPARK_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_STAGE_MOE
SPARK_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_FUSED_FINAL_TOKEN_TAIL
```

The exact-stage validator requires the corresponding callback pointers when those capabilities are set. The built-in final-stage fallback also fuses restricted argmax, MTP draft argmax, and MTP verify/commit into one CUDA kernel, with an external final-tail callback available for an AOT final-stage implementation.

## Validation status

The local CPU/C tests pass with `make -j2 test`. The CUDA target still cannot be compiled in this container because `nvcc` is unavailable, so Spark2 must validate:

1. the exact six-layer AOT stage callback path,
2. one stage-level graph replay instead of six per-layer graph replays,
3. the paged prefill attention kernel on long prompts,
4. production 4-bit/8-bit tensor-core callbacks,
5. persistent transport backend replacement with the real inter-spark transport.
