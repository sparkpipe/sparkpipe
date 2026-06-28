#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_SPARSE_MLA_BACKEND_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_SPARSE_MLA_BACKEND_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_sparse_mla_firmware.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SparkGlm52ResidentSparseMlaBackendCompletionFunction)(
    void *completion_context);

typedef struct SparkGlm52ResidentSparseMlaBackendCompletion
{
    SparkGlm52ResidentSparseMlaBackendCompletionFunction function;
    void *context;
} SparkGlm52ResidentSparseMlaBackendCompletion;

SparkStatus SparkGlm52ResidentSparseMlaBackendSubmit(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentSparseMlaBackendCompletion *completion);

void SparkGlm52ResidentSparseMlaBackendQuiesce(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context);

#ifdef __cplusplus
}
#endif

#endif
