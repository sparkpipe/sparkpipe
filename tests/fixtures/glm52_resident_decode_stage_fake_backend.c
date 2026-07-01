#include "spark_glm52_resident_decode_stage_backend.h"

#include <stdbool.h>
#include <stdint.h>

#include "glm52_resident_decode_stage_fake_backend.h"

SparkStatus SparkGlm52ResidentDecodeStageBackendSubmit(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream;

    if (node_context == 0 || completion == 0 || completion->function == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    fake_stream =
        (SparkGlm52ResidentDecodeStageFakeStream *)
            node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
    if (fake_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageFakeStreamHasPending(fake_stream))
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
        SparkGlm52ResidentDecodeStageFakeStreamComplete(fake_stream);
    }
    return SPARK_STATUS_OK;
}


SparkStatus SparkGlm52ResidentDecodeStageBackendSubmitStageSlice(
    const SparkGlm52ResidentDecodeStageStageSlicePlan *stage_slice_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t final_token_stage,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream;
    uint32_t layer_index;

    if (layer_node_contexts == 0 ||
        layer_count == 0u ||
        layer_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        completion == 0 ||
        completion->function == 0 ||
        final_token_stage > 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    first_node_context = layer_node_contexts[0];
    if (first_node_context == 0 ||
        pipeline_slot_index >= first_node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > first_node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    fake_stream =
        (SparkGlm52ResidentDecodeStageFakeStream *)
            first_node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
    if (fake_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageFakeStreamHasPending(fake_stream))
    {
        return SPARK_STATUS_BUSY;
    }
    for (layer_index = 0u; layer_index < layer_count; ++layer_index)
    {
        if (layer_node_contexts[layer_index] == 0 ||
            pipeline_slot_index >=
                layer_node_contexts[layer_index]->pipeline_slot_count ||
            layer_node_contexts[layer_index]->pipeline_slots[
                pipeline_slot_index].cuda_stream != fake_stream ||
            active_sequence_count >
                layer_node_contexts[layer_index]->max_active_sequence_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    fake_stream->submit_count += 1u;
    fake_stream->last_pipeline_slot = pipeline_slot_index;
    fake_stream->last_active_sequence_count = active_sequence_count;
    fake_stream->last_stage_slice_layer_count = layer_count;
    fake_stream->last_stage_slice_final_token_stage = final_token_stage;
    fake_stream->last_stage_slice_plan = stage_slice_plan;
    fake_stream->pending_completion_function = completion->function;
    fake_stream->pending_completion_context = completion->context;
    if (!fake_stream->defer_completion)
    {
        SparkGlm52ResidentDecodeStageFakeStreamComplete(fake_stream);
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageBackendSubmitBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream;

    if (node_context == 0 || completion == 0 || completion->function == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        prompt_token_count == 0u ||
        node_context->bulk_prefill_plan == 0 ||
        prompt_token_count >
            node_context->bulk_prefill_plan->maximum_prompt_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    fake_stream =
        (SparkGlm52ResidentDecodeStageFakeStream *)
            node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
    if (fake_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageFakeStreamHasPending(fake_stream))
    {
        return SPARK_STATUS_BUSY;
    }

    fake_stream->submit_count += 1u;
    fake_stream->last_pipeline_slot = pipeline_slot_index;
    fake_stream->last_active_sequence_count = active_sequence_count;
    fake_stream->last_bulk_prefill_prompt_token_count = prompt_token_count;
    fake_stream->pending_completion_function = completion->function;
    fake_stream->pending_completion_context = completion->context;
    if (!fake_stream->defer_completion)
    {
        SparkGlm52ResidentDecodeStageFakeStreamComplete(fake_stream);
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageBackendSubmitStageSliceBulkPrefill(
    const SparkGlm52ResidentDecodeStageNodeContext *const *layer_node_contexts,
    uint32_t layer_count,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream;
    uint32_t layer_index;

    if (layer_node_contexts == 0 ||
        layer_count == 0u ||
        layer_count >
            SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT ||
        completion == 0 ||
        completion->function == 0 ||
        prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    first_node_context = layer_node_contexts[0];
    if (first_node_context == 0 ||
        pipeline_slot_index >= first_node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > first_node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    fake_stream =
        (SparkGlm52ResidentDecodeStageFakeStream *)
            first_node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
    if (fake_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52ResidentDecodeStageFakeStreamHasPending(fake_stream))
    {
        return SPARK_STATUS_BUSY;
    }
    for (layer_index = 0u; layer_index < layer_count; ++layer_index)
    {
        if (layer_node_contexts[layer_index] == 0 ||
            pipeline_slot_index >=
                layer_node_contexts[layer_index]->pipeline_slot_count ||
            layer_node_contexts[layer_index]->pipeline_slots[
                pipeline_slot_index].cuda_stream != fake_stream ||
            active_sequence_count >
                layer_node_contexts[layer_index]->max_active_sequence_count ||
            layer_node_contexts[layer_index]->bulk_prefill_plan == 0 ||
            prompt_token_count >
                layer_node_contexts[layer_index]->bulk_prefill_plan->
                    maximum_prompt_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    fake_stream->submit_count += 1u;
    fake_stream->last_pipeline_slot = pipeline_slot_index;
    fake_stream->last_active_sequence_count = active_sequence_count;
    fake_stream->last_stage_slice_layer_count = layer_count;
    fake_stream->last_bulk_prefill_layer_count = layer_count;
    fake_stream->last_bulk_prefill_prompt_token_count = prompt_token_count;
    fake_stream->pending_completion_function = completion->function;
    fake_stream->pending_completion_context = completion->context;
    if (!fake_stream->defer_completion)
    {
        SparkGlm52ResidentDecodeStageFakeStreamComplete(fake_stream);
    }
    return SPARK_STATUS_OK;
}

void SparkGlm52ResidentDecodeStageBackendQuiesce(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
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
        SparkGlm52ResidentDecodeStageFakeStream *fake_stream;

        fake_stream =
            (SparkGlm52ResidentDecodeStageFakeStream *)
                node_context->pipeline_slots[pipeline_slot_index].cuda_stream;
        if (fake_stream != 0 &&
            SparkGlm52ResidentDecodeStageFakeStreamHasPending(fake_stream))
        {
            SparkGlm52ResidentDecodeStageFakeStreamComplete(fake_stream);
        }
    }
}
