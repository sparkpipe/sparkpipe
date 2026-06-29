# Expected Performance Versus Mature SOTA

These are code-shape expectations, not measured numbers.

| Path | V6 expected vs mature SOTA after target compile fixes | Notes |
|---|---:|---|
| Token hidden BF16 -> NVFP4 once | 80-95% | Memory-bound and now paid once per token, not per route. |
| Router projection + top-8 | 75-95% | cuBLASLt projection plus fused top-k. A custom persistent GEMV may win at B=1. |
| Grouped NVFP4 gate/up | 85-98% | If backed by CUTLASS SM121 NVFP4 grouped GEMM; otherwise reject. |
| Fused SiLU/mul + NVFP4 requant | 80-95% | Separate kernel; can be epilogue-fused later if measured worthwhile. |
| Grouped NVFP4 down | 85-98% | If backed by CUTLASS SM121 NVFP4 grouped GEMM; otherwise reject. |
| Weighted combine | 75-90% | Materializes down route output; acceptable unless profiling shows it dominates. |
| Sparse MLA | 75-90% | Two-pass tiled online path; FlashMLA/CuTe can still improve it. |
| RoPE + KV write | 85-98% | Already fused and final-layout oriented. |
| Large BF16 projections | 80-98% | Only when cublasLt/CUTLASS fixed-shape plans are bound. |
| Restricted logits | 70-95% | Strong for small K; larger K should use linear plan. |
| MTP MXFP4 | 60-90% | Correct format separation; final SOTA should fuse draft GEMM+argmax+verify. |
| CUDA graph replay | 90-99% | Requires pre-instantiated graphs, stable addresses, no first-live-request capture. |

Hard production rejection conditions:

```text
no CUTLASS/CuTe/cuBLASLt fast plan
scalar/WMMA-dequant fallback reachable
first live request performs graph capture
success path calls cudaStreamSynchronize
SparkPipe runtime sees GLM/KV/MoE/logits internals
```
