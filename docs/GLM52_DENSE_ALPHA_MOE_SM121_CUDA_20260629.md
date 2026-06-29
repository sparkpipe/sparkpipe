# GLM 5.2 dense-alpha MoE SM121 CUDA handoff

This handoff implements the next CUDA experiment requested by `GLM52_MOE_SOTA_RESEARCH_20260629.md`: stop iterating on scalar tiny-N expert dots and measure a tensor-core dense-alpha MoE regime for medium token counts.

## Implemented path

```text
hidden BF16
    -> hidden NVFP4 quantized once per token
    -> SM1xx B-scale repack for broadcast gate/up GEMM
    -> CUTLASS SM121 grouped NVFP4 gate/up, B broadcast across experts
    -> fused SiLU(gate) * up + NVFP4 requant into expert-token workspace
    -> CUTLASS SM121 grouped NVFP4 down GEMM
    -> selected top-k weighted combine from dense expert-token outputs
```

The gate/up GEMM uses expert-resident A/SFA and a single broadcast hidden B/SFB. The down GEMM uses expert-token intermediate B/SFB. This avoids physically copying hidden activations for every expert while still measuring the dense all-local-expert tensor-core strategy.

## Strict recipe behavior

Dense-alpha mode is selected only when the MoE plan advertises:

```text
SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_MOE
SPARK_GLM52_SOTA_FAST_CAP_DENSE_ALPHA_STRICT
SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_B_BROADCAST
SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_GATE_UP
SPARK_GLM52_SOTA_FAST_CAP_CUTLASS_NVFP4_DOWN
SPARK_GLM52_SOTA_FAST_CAP_FUSED_SILU_REQUANT
SPARK_GLM52_SOTA_FAST_CAP_WEIGHTED_COMBINE
SPARK_GLM52_SOTA_FAST_CAP_NO_HOST_STAGING
SPARK_GLM52_SOTA_FAST_CAP_NO_DEVICE_MEMCPY
SPARK_GLM52_SOTA_FAST_CAP_SM121_NVFP4_SCALE_LAYOUT
SPARK_GLM52_SOTA_FAST_CAP_FIXED_GLM52_SHAPES
```

If any dense-alpha resource, shape, broadcast flag, CUTLASS state, workspace, output buffer, or exact active-token capacity match is missing, validation returns `cudaErrorInvalidValue` before launch. There is no automatic fallback inside the dense-alpha recipe.

## Benchmark command

Run from the module directory on `spark1`:

```sh
cd <sparkpipe checkout>/modules/glm52_resident_decode_stage
export PATH=/usr/local/cuda-13.0/bin:$PATH
make benchmark_dense_alpha_moe \
    CUDA_ARCH=sm_121a \
    SOTA_V6_ENABLE_CUTLASS=1 \
    SOTA_V6_CUTLASS_INCLUDE=/home/spark1/src/upstreams/cutlass/include \
    SOTA_V6_CUTLASS_UTIL_INCLUDE=/home/spark1/src/upstreams/cutlass/tools/util/include \
    DENSE_ALPHA_MOE_BENCH_ARGS="--tokens 96 --groups 256 --capacity 96 --warmup 3 --iterations 10 --workspace-mb 1024"
```

Suggested qualification sweep:

```sh
make benchmark_dense_alpha_moe CUDA_ARCH=sm_121a SOTA_V6_ENABLE_CUTLASS=1 SOTA_V6_CUTLASS_INCLUDE=/home/spark1/src/upstreams/cutlass/include SOTA_V6_CUTLASS_UTIL_INCLUDE=/home/spark1/src/upstreams/cutlass/tools/util/include DENSE_ALPHA_MOE_BENCH_ARGS="--tokens 64  --groups 256 --capacity 64  --warmup 3 --iterations 10 --workspace-mb 1024"
make benchmark_dense_alpha_moe CUDA_ARCH=sm_121a SOTA_V6_ENABLE_CUTLASS=1 SOTA_V6_CUTLASS_INCLUDE=/home/spark1/src/upstreams/cutlass/include SOTA_V6_CUTLASS_UTIL_INCLUDE=/home/spark1/src/upstreams/cutlass/tools/util/include DENSE_ALPHA_MOE_BENCH_ARGS="--tokens 96  --groups 256 --capacity 96  --warmup 3 --iterations 10 --workspace-mb 1024"
make benchmark_dense_alpha_moe CUDA_ARCH=sm_121a SOTA_V6_ENABLE_CUTLASS=1 SOTA_V6_CUTLASS_INCLUDE=/home/spark1/src/upstreams/cutlass/include SOTA_V6_CUTLASS_UTIL_INCLUDE=/home/spark1/src/upstreams/cutlass/tools/util/include DENSE_ALPHA_MOE_BENCH_ARGS="--tokens 128 --groups 256 --capacity 128 --warmup 3 --iterations 10 --workspace-mb 1024"
```

Compare against the existing fixed-group sparse CUTLASS baseline. If dense-alpha wins in the B64/B96/B128 regime, bind it as the exact GLM52 SM121 MoE recipe for that descriptor regime. If it loses, the next CUDA step is a compact active-expert tensor-core grouped path, not another scalar kernel.

## Spark1 measurement

Measured on `spark1` from a temporary checkout based on `main` commit
`8d2a7d3`, with this overlay applied:

```text
PATH=/usr/local/cuda-13.0/bin:$PATH
CUDA_ARCH=sm_121a
SOTA_V6_ENABLE_CUTLASS=1
SOTA_V6_CUTLASS_INCLUDE=/home/spark1/src/upstreams/cutlass/include
SOTA_V6_CUTLASS_UTIL_INCLUDE=/home/spark1/src/upstreams/cutlass/tools/util/include
```

Dense-alpha results:

```text
B64:  avg_us=29989.848  dense_estimated_tflops=41.246  expert_payload_scale_gbps=181.255  overflow=0
B96:  avg_us=34499.535  dense_estimated_tflops=53.781  expert_payload_scale_gbps=157.562  overflow=0
B128: avg_us=35979.039  dense_estimated_tflops=68.760  expert_payload_scale_gbps=151.083  overflow=0
```

Comparable fixed sparse CUTLASS uniform-route results:

```text
B64:  avg_us=25700.727  route_capacity_per_expert=2  overflow=0
B96:  avg_us=25664.520  route_capacity_per_expert=3  overflow=0
B128: avg_us=25604.879  route_capacity_per_expert=4  overflow=0
```

Decision:

```text
dense-alpha compiles and runs, but is rejected as the production GB10 path.
```

The hypothesis from the NVIDIA B200 DenseGEMM note does not transfer cleanly to
GB10 for this uniform 256-expert shape. The all-expert resident-weight traffic
dominates. Broadcast B/SFB removes replicated hidden activation storage, but it
does not remove the cost of streaming all expert weights and scale factors.

The next CUDA module should be the compact active-expert tensor-core grouped MoE
path:

```text
real router histogram
    -> compact active expert list
    -> fixed resident expert-major workspaces
    -> SM121 tensor-core grouped gate/up and down
    -> fused SiLU/requant/combine
    -> CUDA graph profiles for B1/B8/B32/B64/B96/B128
```

Do not return to scalar tiny-N workers. The failed dense-alpha result corrects
the model toward active-expert compaction plus real tensor-core math.
