# Status

SparkPipe is a compile-time firmware packager and runtime dispatcher. It is not a CUDA fallback framework.

Current GLM 5.2 SM121 policy:

```text
model description JSON
    -> exact required module IDs
    -> offline package link
    -> required CUDA symbols must resolve
    -> target validation runs once for the artifact contract
    -> serving restores the validated package and submits direct calls
```

The GLM 5.2 resident decode stage now depends on an externally supplied required CUDA implementation:

```text
spark.glm52.sm121.required_decode_stage.b12x_fused.v1
```

The required module must provide these symbols:

```text
SparkGlm52Sm121RequiredDecodeStageInitialize
SparkGlm52Sm121RequiredDecodeStageLaunch
SparkGlm52Sm121RequiredDecodeStageQuiesce
```

If the required CUDA module is absent, package linking or validation linking fails. There is no internal slow replacement path.
