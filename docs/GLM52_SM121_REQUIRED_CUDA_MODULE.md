# GLM 5.2 SM121 required CUDA module

SparkPipe owns the model-driver ABI and package link. The required CUDA module owns the actual high-performance GLM 5.2 SM121 implementation.

Required module identity:

```text
spark.glm52.sm121.required_decode_stage.b12x_fused.v1
```

Required symbols:

```c
SparkStatus SparkGlm52Sm121RequiredDecodeStageInitialize(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);

SparkStatus SparkGlm52Sm121RequiredDecodeStageLaunch(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    void *cuda_stream);

void SparkGlm52Sm121RequiredDecodeStageQuiesce(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);
```

`Initialize` performs the one-time restore work: resident state binding, converted weight/scale layout validation, CUDA handle creation, tactic selection restore, and process-local graph preparation. `Launch` submits the already-qualified decode recipe. `Quiesce` releases or drains process-local CUDA resources.

The selected implementation is linked explicitly and must satisfy the required
SM121/B12x contract. There is no runtime backend selection.
