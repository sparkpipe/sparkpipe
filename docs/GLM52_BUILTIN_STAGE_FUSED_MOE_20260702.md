# GLM-5.2 PP13 Built-In Fused MoE Path

Date: 2026-07-02

Updated for production NVFP4 dispatch on 2026-07-16. Micro and dynamic buckets
were removed. Current production requires an exact static AOT bucket and fails
closed when one is absent.

This pass connects the PP13 fused MoE optimization to the required SM121 execution path instead of leaving it as an external callback contract.

## Production path

For B12x-routed GLM-5.2 layers, the resident stage now does:

```text
post-attention normalized hidden
    -> prebound production-fast router projection
    -> router logits
    -> 256-thread resident top-k reduction per execution row
    -> generated exact-static B12x expert kernel
    -> deterministic parallel route finalizer
    -> residual combine
```

The resident stage owns router projection and top-k selection. The B12x backend
accepts precomputed top-k rows only; setting the router-logits argument flag is
rejected. This keeps one shared, parallel top-k implementation for NVFP4,
W8LUT, and the other production expert paths.

## Exact-static buckets

Every live execution-row count must have a generated static bucket with exact
capacity and geometry. The runtime does not select micro or dynamic kernels,
split a request across buckets, or substitute a smaller/larger bucket.

## Router and workspace contract

Router logits are produced by the validated production-fast linear plan. Top-k
uses a block-wide reduction over all 256 experts for each execution row. One
generated AOT workspace sized for the largest exact bucket is shared by the
model-ordered local routed layers; layer weights remain resident and distinct.

## ABI change

`SparkGlm52Sm121FlashInferB12xMoeArguments` is now ABI version 3 and adds:

```c
argument_flags
router_logits_f32
router_score_bias_f32
router_norm_topk_prob
router_routed_scaling_factor
```

The production backend requires precomputed `topk_ids_i32` and
`topk_weights_fp32`. `SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ARGUMENT_FLAG_ROUTER_LOGITS`
is retained in the public ABI for compatibility but is rejected by this
production implementation.

## Validation status

The host readiness and source-contract suites cover exact-bucket dispatch,
fallback rejection, parallel top-k/route preparation, generated-manifest
identity, and deterministic finalization. CUDA build validation is performed
from an isolated SM121 Spark checkout before hardware qualification.
