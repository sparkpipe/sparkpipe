# Vendored third-party source

This directory intentionally contains source that SparkPipe needs for the exact GLM 5.2 SM121 recipe work. It is not fallback code and it is not a replacement for qualification.

## `flashinfer/`

Vendored from the uploaded `flashinfer-main.zip`. The relevant production target is FlashInfer's SM120/SM121 B12x fused MoE implementation under:

```text
third_party/flashinfer/flashinfer/fused_moe/cute_dsl/b12x_moe.py
third_party/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/
```

FlashInfer is licensed under Apache License 2.0. Its upstream `LICENSE`, `NOTICE`, and dependency license files are preserved in the vendored tree.

## `deepspec/`

Vendored from the uploaded `DeepSpec-main.zip` so DSpark/DFlash modeling and configuration code is available in-repo. DeepSpec is licensed under MIT. Its upstream `LICENSE` and `NOTICE` are preserved in the vendored tree.

## Policy

SparkPipe production code must link or restore the exact required CUDA module. Vendored source here is source material for the required module and one-time qualification tools. Runtime backend substitution is forbidden.
