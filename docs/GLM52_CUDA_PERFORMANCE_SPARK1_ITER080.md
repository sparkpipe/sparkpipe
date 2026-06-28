# GLM 5.2 CUDA performance baseline, iteration 080

This is the first measured Spark-side baseline after merging the iteration 080
SOTA CUDA handoff with the newer routed NVFP4 expert gates.

```text
repo:        sparkpipe/sparkpipe
branch:      codex/iter080-sota-cuda-handoff
commit:      34427b3
spark host:  spark1
checkout:    /home/spark1/src/sparkpipe-551c3ea
model dir:   /home/spark1/models/hf/nvidia/GLM-5.2-NVFP4
nvcc:        /usr/local/cuda-13.0/bin/nvcc
target:      sm_121
date:        2026-06-28
```

The measurements use the official GLM-5.2-NVFP4 artifact tensors. Runtime
activation quantization is used where required by the resident CUDA path; the
expert weights are not re-quantized from a different checkpoint at startup.

## Build gate

```text
make -C modules/glm52_resident_decode_stage clean archive
```

Result:

```text
passed
```

Remaining warnings:

```text
restricted_grid declared but unused
SparkGlm52ResidentDecodeStageMtpDraftLogitsKernel declared but not referenced
```

## Module timings

All timings are CUDA event or package-reported elapsed microseconds from the
module validators on spark1. Direct backend means the module-local CUDA backend
path. Orchestrator means the generated driver and SparkPipe route/admission
boundary submitted the same package.

| Gate target | Direct backend | Orchestrator | Notes |
| --- | ---: | ---: | --- |
| `package_layer3_router_bf16` | 3909.536 us | 4306.304 us | top-k router gate, first expert `233`, first weight about `0.31617311` |
| `package_layer3_shared_expert_bf16` | 3822.432 us | 4001.376 us | BF16 shared expert, max error `0.00000000` |
| `package_layer3_routed_expert_nvfp4` | 4345.280 us | 4485.984 us | top-1 routed official NVFP4 expert, `21233664` bytes loaded |
| `package_layer3_routed_expert_nvfp4_topk` | 7723.360 us | 7557.408 us | top-8 routed official NVFP4 experts, `169869312` bytes loaded |
| `package_layer0_full_reference_bf16` | 4086.667 us avg, 4262.720 us max | 5285.440 us | dense layer-0 full reference, max error `0.00195312` |
| `package_dense_chain_bf16` | 58469.471 us total, 5191.200 us max | 63849.632 us total, 5905.088 us max | three dense layers, twelve submissions |

## Correctness evidence

The top-8 routed NVFP4 gate selected the same expert set from the official
router as the CPU reference:

```text
233,41,166,174,186,37,117,223
```

The GPU route order can differ for equal or near-equal candidates while the
weighted expert sum remains route-order invariant. The validator now checks
membership of the selected expert set while still failing missing experts and
nonzero/output mismatches.

Layer-0 full reference now passes:

```text
layer0_reference_full=1
layer0_reference_full_max_error=0.00195312
real_lm_head_max_logit_error=0.00000000
```

## Fixes forced by measurement

The first Spark build found CUDA compile issues that were not visible on the Mac
without `nvcc`:

```text
missing NVFP4 group-size constant
mutable router output pointers were declared const
CUDA FP8 intrinsic macros were not available through the active CUDA 13 include path
one malformed KV cache-latent expression from the handoff merge
```

The first layer-0 full reference run then exposed a second malformed raw-KV
split expression. Both expression bugs were in the resident CUDA body, not in
SparkPipe core. The measured package gates above are after those fixes.

## Current performance reading

This is still correctness-first firmware. The largest measured GLM 5.2 hot-path
cost in this baseline is routed local MoE:

```text
top-8 routed NVFP4 expert path: 7.7 ms for one token on one Spark stage
```

The next SOTA pass should therefore prioritize the grouped MoE launch boundary:

```text
prepacked official NVFP4 expert tiles
driver-owned route workspace
fused route gather, gate/up, activation quantize, down, and combine
no per-token scalar expert scans
CUDA graph replay after resident plan binding
```

The dense chain is slower in total wall time because it intentionally runs
multiple layer submissions. For single-module optimization, the routed top-8
expert path is the worse measured module and should be the next CUDA target.
