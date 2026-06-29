# GLM 5.2 SM121 CUDA SOTA Fixed Packet V6

This packet is CUDA-only. It is intended for the GLM 5.2 SparkPipe firmware driver, not for the SparkPipe runtime layer.

The production path is hard-coded for GLM 5.2 geometry:

```text
hidden = 6144
layers = 78
first dense layers = 3
MoE layers = 75
experts = 256
top_k = 8
MoE intermediate = 2048
NVFP4 group = 16
MLA latent = 512
RoPE = 64
heads = 64
selected sparse tokens = 2048
```

## What is fixed versus the previous packets

The prior packets still left the real fast math mostly as plan hooks. This packet adds production-oriented implementations and strict launch wrappers for the main critical paths:

```text
spark_glm52_sota_production_plan_sm121.cuh/.cu
    fail-closed production contract for GLM 5.2 SOTA profiles

spark_glm52_sota_cutlass_nvfp4_grouped_gemm_sm121.cu
    SM121 NVFP4 grouped-GEMM launch wrapper for CUTLASS 79d-family kernels

spark_glm52_sota_grouped_nvfp4_moe_production_sm121.cu
    token quant once, expert-major grouping, gate/up grouped GEMM,
    fused SiLU+requant, down grouped GEMM, weighted combine

spark_glm52_sota_router_topk_fused_sm121.cu
    prebound BF16 cuBLASLt router projection plus fused bias/sigmoid/top-8

spark_glm52_sota_flash_mla_sm121.cu
    tiled online sparse MLA two-pass kernel with stable workspace contract
```

The existing files from V3 remain present for RMSNorm/residual, RoPE+KV write, restricted logits, MTP MXFP4 verify, graph replay, and earlier bring-up paths. For production, use the V6 production plan and reject the old local debug fallbacks.

## Production rule

A GLM 5.2 production/SOTA package must call:

```text
SparkGlm52SotaValidateProductionDecodePlanSm121
```

and must reject the package if any required capability flag is missing. This prevents scalar, WMMA-dequant, serial attention, first-request graph-capture, or host-staging paths from being called SOTA.

## Integration notes for Codex

1. Bind `SparkGlm52SotaNvfp4GroupedGemmPlanSm121::cutlass_or_cublas_state` to a cached CUTLASS SM120/SM121 grouped NVFP4 argument object constructed at firmware initialization.
2. Store GLM expert gate/up weights in a concatenated expert-major NVFP4 layout for the gate/up grouped GEMM: `[expert][4096][6144]` logical weight rows.
3. Store down weights in expert-major NVFP4 layout: `[expert][6144][2048]` logical weight rows.
4. Use `gate_weight_nvfp4` as the concatenated gate/up view for the production MoE launch.
5. Keep `up_weight_nvfp4` only for debug compatibility or delete it from the production path.
6. The production driver must not call the older local scalar fallback functions.

## Remaining target-debug work

This code is written to be within target-debug distance, but this environment cannot run `nvcc`, device-link CUTLASS, or profile SM121. Codex must compile on CUDA 13 / SM121, fix any exact CUTLASS API details, run numerical gates, profile with Nsight, and only publish the artifact if full submission-to-completion latency passes.
