# Advisor: AMD port boundary table (user-provided)

## Shared model-specific code vs hardware-target code
| Shared | Hardware-target |
| --- | --- |
| Model geometry and layer order | CUDA/HIP allocation, streams and events |
| Batch buckets and lane lifecycle | CUDA Graph/HIP Graph creation and replay |
| Expert queue admission and sealing policy | Device-resident queue representation |
| Logical routing and grouped-expert contract | Count/scan/scatter kernels |
| KV and prefix-cache policy | Physical KV layout and attention kernels |
| TP/PP operation ordering | NCCL, RCCL or custom transport implementation |
| Speculation and sampling semantics | Device kernels and target-specific fusion |
| Logical weight tensor inventory | CUDA- and AMD-specific packed layouts |
| Numerical oracle and tolerances | Per-target performance qualification |

## Expert aggregation split
Common core decides: which submissions may combine, queue keys/generations, max wait, batch-bucket selection, when a layer batch seals, cancellation/fairness. Backend receives an opaque sealed-route batch and implements counting/grouping/grouped GEMM.

## No hot-path backend vtable
Target chosen at package compile time. cuda.sm121 links CUDA symbols; rocm.gfx950 links ROCm symbols; both link the same host/control source. Direct calls, no backend branches. Opaque handles resolve via static linking.

## Interface granularity
NOT per-launch-function (61 launch fns would freeze CUDA fusion boundaries). Semantic execution-island level: attention block, routed-MoE block, shared expert, head, cache transition - each backend may fuse differently.

## AMD target
MI350P (gfx950, CDNA4): 64-lane wavefronts, native MXFP4, different LDS/register/tiling. HIP graphs, RCCL NCCL-aligned. Scope: separate homogeneous CUDA and AMD deployments (no mixed-vendor collective).

## Sequence
1. Neutral runtime primitives for DSV4 only. 2. Split DSV4 portable-core/device-contract/cuda-impl. 3. Byte-identical + no CUDA perf regression. 4. gfx950 runtime backend. 5. One complete DSV4 layer. 6. Attention, grouped MoE, TP4. 7. Generalize seams for qwen/glm/k3.
