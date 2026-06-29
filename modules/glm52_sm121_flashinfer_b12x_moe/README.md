# GLM52 SM121 FlashInfer B12x fused MoE adapter

This module is a strict adapter boundary for the required FlashInfer B12x fused MoE primitive.

It does not contain a fallback implementation. It forwards to these required backend symbols:

```text
SparkFlashInferB12xCompiledMoeCreate
SparkFlashInferB12xCompiledMoeLaunch
SparkFlashInferB12xCompiledMoeDestroy
```

If the compiled B12x backend library is not linked, the final model-driver link fails.

The vendored upstream source is under:

```text
third_party/flashinfer/
```

The exact SparkPipe primitive ABI is:

```text
include/sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h
```

The required contract is GLM 5.2, SM121A, NVFP4 weights/activations, BF16 output, 256 experts, top-k 8, hidden 6144, intermediate 2048, FlashInfer static weight views, FlashInfer static scale storage, and up/gate fused W1 order.
