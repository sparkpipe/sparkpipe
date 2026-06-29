# GLM 5.2 CUDA V6 production packet, spark1 evidence

This records the Spark-side integration result for the GLM 5.2 production CUDA
fixed V6 packet.

```text
repo:        sparkpipe/sparkpipe
branch:      codex/glm52-sota-production-v6
commit:      b51457a
spark host:  spark1
checkout:    /home/spark1/src/sparkpipe-551c3ea
model dir:   /home/spark1/models/hf/nvidia/GLM-5.2-NVFP4
nvcc:        /usr/local/cuda-13.0/bin/nvcc
cuda:        13.0.88
target:      sm_121
date:        2026-06-29
```

## What compiled

The V6 packet is integrated under:

```text
modules/glm52_resident_decode_stage/source/sota_v6
```

and is included in:

```text
modules/glm52_resident_decode_stage/Makefile
```

Target compile command:

```sh
NVCC=/usr/local/cuda-13.0/bin/nvcc \
make -C modules/glm52_resident_decode_stage clean archive
```

Result:

```text
compiled: yes
archive: build/modules/glm52_resident_decode_stage/libglm52_resident_decode_stage.a
```

The first Spark compile failed because `CUDART_INF_F` was unavailable from the
included CUDA headers. Commit `b51457a` fixed that once in the shared V6 common
header with `<math_constants.h>` plus a fallback definition.

Current warnings:

```text
SparkGlm52ResidentDecodeStageGroupedMoeNvfp4GateUpKernel declared but unused
SparkGlm52ResidentDecodeStageGroupedMoeNvfp4DownKernel declared but unused
SparkGlm52ResidentDecodeStageMtpDraftLogitsKernel declared but unused
SparkGlm52SotaValidateCutlassGroupedPlan declared but unused when CUTLASS is disabled
```

## CUTLASS status

CUTLASS was not found under `/usr/local`, `/opt`, or `/home/spark1` during this
pass. The Makefile therefore leaves:

```text
SOTA_V6_ENABLE_CUTLASS=0
```

With that setting the V6 grouped-GEMM wrapper compiles, but
`SparkGlm52SotaLaunchCutlassNvfp4GroupedGemmSm121` returns
`cudaErrorNotSupported`. This is the desired fail-closed behavior. It is not a
SOTA performance path yet.

## Measurements

These are package-gate measurements on the current active firmware path with
the V6 archive linked. They should not be reported as V6 production-path speedups
until the driver binds the V6 production plan, CUTLASS grouped NVFP4 GEMM state,
and the production launch functions.

| Target | Direct validation | Packaged/orchestrated validation | Notes |
| --- | ---: | ---: | --- |
| `package_layer3_router_bf16` | 4514.784 us | 4431.456 us | Router top-k reference passed; top-8 experts `233,41,166,174,186,37,117,223`. |
| `package_layer3_routed_expert_nvfp4` | 4746.624 us | 5162.048 us | Top-1 diagnostic official NVFP4 expert path; selected expert `233`. |
| `package_layer3_routed_expert_nvfp4_topk` | 7682.080 us | 8863.040 us | Top-8 diagnostic official NVFP4 expert path; route count `8`. |
| `glm52_resident_sparse_mla_firmware_package` | 291.240 us avg | 414.881 us max | Separate resident sparse-MLA package gate. |

Published module artifacts from this pass:

```text
router_bf16:              e11730c02776c23c2557b87e1ba1b23590a682bd9c810ee19c55211145ac77a7
routed_expert_nvfp4:      630d61f79fbf2795b9f95f6edeca74b1f4307438cd0b6fd9821622c590eb144f
routed_expert_nvfp4_topk: a5ccac7b310b7f5cbd81b0da7be40dfec52a526c75f5846e37368848adffa018
sparse_mla:               fb58a682cec2c4201a8ab2685e8c22ecb7e3f477d0648c5af9847cddfa970d01
```

## Interpretation

This pass proves:

```text
V6 CUDA source compiles on CUDA 13 / SM121
the V6 source set can be carried inside the GLM firmware archive
existing package gates still pass against the live GLM 5.2 NVFP4 artifact
the V6 CUTLASS path fails closed when CUTLASS is not available or not enabled
```

This pass does not prove:

```text
SOTA grouped-MoE runtime
V6 production MoE path is active
V6 FlashMLA path is active
V6 graph replay path is active
end-to-end GLM 5.2 inference
```

The next real performance milestone is to install or vendor a pinned CUTLASS
version with SM120/SM121 NVFP4 grouped-GEMM support, build with
`SOTA_V6_ENABLE_CUTLASS=1`, bind the V6 production decode plan during resident
setup, and rerun these same package gates with counters proving the old
diagnostic MoE path was not called.
