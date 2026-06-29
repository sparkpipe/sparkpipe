# FlashInfer provenance for SparkPipe

Vendored from the user-provided archive `flashinfer-main.zip`.

Primary code path extracted for GLM 5.2 SM121 MoE work:

```text
flashinfer/fused_moe/cute_dsl/b12x_moe.py
flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py
flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py
flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py
flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py
flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_direct_micro_kernel.py
flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py
```

The B12x path is CuTe DSL source. It is not a standalone `.cu` kernel that can be linked into SparkPipe as-is. SparkPipe therefore keeps it as vendored source, supplies a prequalification tool, and requires the final production module to provide explicit link symbols. Missing required symbols are a build/link error.
