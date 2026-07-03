# GLM-5.2 PP13 Built-In Fused MoE Path

Date: 2026-07-02

This pass connects the PP13 fused MoE optimization to the required SM121 execution path instead of leaving it as an external callback contract.

## Production path

For B12x-routed GLM-5.2 layers, the resident stage now does:

```text
post-attention normalized hidden
    -> prebound production-fast router projection
    -> router logits
    -> B12x backend-owned top-k / dispatch preparation
    -> generated B12x expert kernel
    -> residual combine
```

The resident layer no longer launches a separate router top-k kernel before calling the B12x backend. The backend receives router logits, score bias, normalization mode, and routed scaling factor through the B12x launch ABI and owns top-k preparation immediately before expert dispatch.

## Micro-bucket fusion

For B12x micro buckets, the backend combines:

```text
router-logit top-k selection
compact top-k id generation
active expert list construction
```

into one backend preparation kernel. This removes the old resident top-k kernel plus the backend micro compact-topk kernel boundary for the micro path.

## Static / dynamic buckets

For larger generated B12x buckets, the backend performs router top-k preparation internally and passes the prepared top-k buffers to the generated expert launch. This keeps top-k and dispatch preparation inside the same backend launch contract and avoids contaminating the resident stage with backend-specific dispatch preparation.

## No placeholder fused-router kernel

The prior path referenced a future fused BF16 router/top-k kernel before falling back to the prebound router projection. That dependency is removed. Router logits are produced by the already-validated production-fast linear plan, and the B12x backend consumes those logits directly.

## ABI change

`SparkGlm52Sm121FlashInferB12xMoeArguments` is now ABI version 3 and adds:

```c
argument_flags
router_logits_f32
router_score_bias_f32
router_norm_topk_prob
router_routed_scaling_factor
```

`SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS` selects the backend-owned router-topK path. Without this flag, the backend keeps supporting precomputed top-k buffers for non-B12x or external integration paths.

## Validation status

Local C/Python tests pass. The B12x adapter source also compiles under the C compiler. CUDA hardware validation still needs Spark2 because this container does not have `nvcc`.
