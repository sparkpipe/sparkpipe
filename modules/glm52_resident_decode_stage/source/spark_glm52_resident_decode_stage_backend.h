#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_BACKEND_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_BACKEND_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SparkGlm52ResidentDecodeStageBackendCompletionFunction)(
    void *completion_context);

typedef struct SparkGlm52ResidentDecodeStageBackendCompletion
{
    SparkGlm52ResidentDecodeStageBackendCompletionFunction function;
    void *context;
} SparkGlm52ResidentDecodeStageBackendCompletion;

SparkStatus SparkGlm52ResidentDecodeStageBackendSubmit(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion);

void SparkGlm52ResidentDecodeStageBackendQuiesce(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context);

#ifdef __cplusplus
}
#endif

#endif
