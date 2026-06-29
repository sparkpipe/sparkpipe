# Iteration 083 CUDA Critical-Path Audit

This pass keeps SparkPipe at the firmware boundary. The changes remain inside the GLM 5.2 decode-stage firmware module.

## Fixed or tightened

### MoE grouped expert execution

The previous tree had a driver-owned grouped-MoE plan hook, but no firmware-provided grouped execution path. Iteration 083 adds a concrete persistent NVFP4 top-k grouped MoE launcher:

```text
SparkGlm52ResidentDecodeStageLaunchPersistentGroupedNvfp4Moe
```

The launcher performs:

```text
router logits/top-k
bound expert slot resolution
route count by resident expert
route scatter into expert-major order
BF16 hidden -> NVFP4 route quantization
persistent grouped gate/up workers
fused SiLU + NVFP4 quantization
persistent grouped down workers
weighted route combine
```

The persistent workers consume compact expert/route/row work items from a device queue. This avoids launching a huge `intermediate × bound_expert × max_route_tiles` grid where most experts are empty.

### Fast router path

The old routed-MoE path still computed router scores with a scalar per-expert dot product. Iteration 083 adds a router-logits linear plan:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS
```

When a SOTA profile binds that plan and provides `pipeline_slot->moe_router_logits`, routing becomes:

```text
prebound Tensor Core / cuBLASLt router projection
        ↓
small top-k over 256 router logits
```

instead of:

```text
256 scalar per-expert hidden dot products inside the top-k kernel
```

The strict flag is:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER
```

### Route-slot cache policy

Routed NVFP4 top-k no longer requires `moe_nvfp4_bound_expert_count == top_k`. That was a bring-up assumption and a performance/placement footgun. Production may bind a larger resident expert set and use `expert_id -> bound_slot` mapping plus per-route slot cache.

Spark-side measurement corrected one important assumption: building that
per-route slot cache is not free and should not run just because the buffer is
available. The cache is now opt-in through
`SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_NVFP4_ROUTE_SLOT_CACHE`.
Ordinary routed NVFP4 gates keep the previous in-kernel lookup path; SOTA
grouped-MoE profiles can still require the cache explicitly.

Measured result after that correction is recorded in:

```text
docs/GLM52_CUDA_PERFORMANCE_SPARK1_ITER083.md
```

The short version: persistent/grouped scaffolding compiles and passes
correctness, but scaffolding alone is not the SOTA win. It must be paired with
tuned SM121 FP4/Tensor Core expert math before it should become a production
fast path.

### SOTA grouped MoE contract

The grouped-MoE plan ABI now records:

```text
plan kind
capability flags
maximum active sequence count
maximum route count
route tile count
persistent worker block count
compact route grouping workspaces
persistent work-item queue
validated maximum latency
```

A persistent NVFP4 SOTA plan must provide:

```text
grouped by expert
persistent workers
NVFP4 top-k
route-slot cache
fused SiLU/quant
zero host staging
zero device memcpy
validated latency
```

## Still target-hardware work

This release is code-only. The target CUDA pass should still replace local decode/dequant loops with SM121-tuned tensor-core or custom FP4 kernels where appropriate, then validate with Nsight and end-to-end stage latency. The SparkPipe layer should not change for those optimizations.
