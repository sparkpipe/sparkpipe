# GLM 5.2 CUDA performance baseline, iteration 083

This records the Spark-side measurements for the iteration 083 persistent
grouped-MoE CUDA handoff.

```text
repo:        sparkpipe/sparkpipe
branch:      codex/iter083-persistent-grouped-moe
spark host:  spark1
checkout:    /home/spark1/src/sparkpipe-551c3ea
model dir:   /home/spark1/models/hf/nvidia/GLM-5.2-NVFP4
nvcc:        /usr/local/cuda-13.0/bin/nvcc
target:      sm_121
date:        2026-06-28
```

## Compile result

The first target build failed at commit `e1af188` because the persistent
grouped-MoE path called the top-k SiLU/quant kernel with the older argument
order. That was fixed in `c4cedd4`.

The decode-stage CUDA archive then compiled and linked on spark1. Current
warnings:

```text
SparkGlm52ResidentDecodeStageGroupedMoeNvfp4GateUpKernel declared but unused
SparkGlm52ResidentDecodeStageGroupedMoeNvfp4DownKernel declared but unused
SparkGlm52ResidentDecodeStageMtpDraftLogitsKernel declared but unused
```

The first two are stale non-persistent grouped kernels after the persistent
worker path became the active grouped implementation. The MTP warning is a
separate unreferenced draft-logits kernel.

## Initial measured result

At `c4cedd4`, correctness passed but the main MoE paths regressed versus the
iteration 080 baseline:

| Gate target | Iteration 080 direct | Iteration 083 direct | Iteration 080 orchestrated | Iteration 083 orchestrated |
| --- | ---: | ---: | ---: | ---: |
| `package_layer3_router_bf16` | 3909.536 us | 4370.016 us | 4306.304 us | 4816.032 us |
| `package_layer3_shared_expert_bf16` | 3822.432 us | 4003.296 us | 4001.376 us | 4459.648 us |
| `package_layer3_routed_expert_nvfp4` | 4345.280 us | 5010.304 us | 4485.984 us | 5221.248 us |
| `package_layer3_routed_expert_nvfp4_topk` | 7723.360 us | 7820.000 us | 7557.408 us | 7752.544 us |
| `package_layer0_full_reference_bf16` | 4086.667 us avg | 4679.285 us avg | 5285.440 us | 5901.824 us |
| `package_dense_chain_bf16` | 58469.471 us total | 56354.816 us total | 63849.632 us total | 63462.081 us total |

Sparse-MLA after the iteration 083 changes passed:

```text
average_us=536.400
maximum_us=719.069
```

## Corrected negative result

The first bad assumption was that resolving the per-route
`expert_id -> bound_slot` cache is always beneficial when the buffer exists.
For ordinary correctness/performance gates, that adds an extra kernel launch and
does not help unless the profile explicitly wants the route-slot cache.

Commit `f84ce02` gates the route-slot cache behind:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_NVFP4_ROUTE_SLOT_CACHE
```

and also checks the router launch status before continuing when the cache is not
used.

Retest at `f84ce02`:

| Gate target | Direct | Orchestrated | Reading |
| --- | ---: | ---: | --- |
| `package_layer3_routed_expert_nvfp4_topk` | 7699.616 us | 8119.168 us | direct recovered and slightly beat iteration 080; orchestrated result was noisy/worse |
| `package_layer3_routed_expert_nvfp4` | 4735.616 us | 5298.624 us | improved versus `c4cedd4`, still slower than iteration 080 |

## Current interpretation

Iteration 083 is a useful firmware-shape pass, not yet a performance win.

What it proves:

```text
persistent grouped-MoE ABI compiles on SM121
the target Spark package gates pass numerically
the persistent path can be carried behind explicit fast-path requirements
route-slot caching must be opt-in, not default
```

What it does not prove:

```text
SOTA grouped-MoE performance
faster default top-1 routed NVFP4
faster orchestrated top-8 routed NVFP4
CUDA graph replay benefit
FP4/Tensor Core expert math benefit
```

The next performance hypothesis is narrower:

```text
The route grouping and persistent work queue are only useful after the inner
gate/up/down dot-product loops are replaced by tuned SM121 FP4/Tensor Core
expert kernels. Queueing/scaffolding alone adds overhead and should not be the
default path.
```

Do not merge this branch as a performance win. Merge only if the intent is to
land the opt-in grouped-MoE ABI and measured negative-result record, with the
production profile still refusing to publish until validated latency improves.
