#include "spark_glm52_resident_sparse_mla_backend.h"

#include <stdbool.h>
#include <stdint.h>

#include "glm52_resident_sparse_mla_fake_backend.h"

SparkStatus SparkGlm52ResidentSparseMlaBackendSubmit(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentSparseMlaBackendCompletion *completion)
{
    SparkGlm52ResidentSparseMlaFakeStream *fake_stream;

    if (node_context == 0 || completion == 0 || completion->function == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    fake_stream =
        (SparkGlm52ResidentSparseMlaFakeStream *)
            node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
    if (fake_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentSparseMlaFakeStreamHasPending(fake_stream))
    {
        return SPARK_STATUS_BUSY;
    }

    fake_stream->submit_count += 1u;
    fake_stream->last_pipeline_slot = pipeline_slot_index;
    fake_stream->last_active_sequence_count = active_sequence_count;
    fake_stream->pending_completion_function = completion->function;
    fake_stream->pending_completion_context = completion->context;
    if (!fake_stream->defer_completion)
    {
        SparkGlm52ResidentSparseMlaFakeStreamComplete(fake_stream);
    }
    return SPARK_STATUS_OK;
}

void SparkGlm52ResidentSparseMlaBackendQuiesce(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context)
{
    uint32_t pipeline_slot_index;

    if (node_context == 0)
    {
        return;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        SparkGlm52ResidentSparseMlaFakeStream *fake_stream;

        fake_stream =
            (SparkGlm52ResidentSparseMlaFakeStream *)
                node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
        if (fake_stream != 0 &&
            SparkGlm52ResidentSparseMlaFakeStreamHasPending(fake_stream))
        {
            SparkGlm52ResidentSparseMlaFakeStreamComplete(fake_stream);
        }
    }
}
