#include "spark_glm52_resident_decode_stage_backend.h"

#include <stdbool.h>
#include <stdint.h>

#include "glm52_resident_decode_stage_fake_backend.h"

static SparkStatus SparkGlm52ResidentDecodeStageFakeValidateRuntimeKvBlockTable(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t active_sequence_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table)
{
    if (runtime_kv_block_table == 0)
    {
        return SPARK_STATUS_OK;
    }
    if (runtime_kv_block_table->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        runtime_kv_block_table->descriptor_bytes !=
            SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES ||
        runtime_kv_block_table->block_token_count == 0u ||
        runtime_kv_block_table->lane_count < active_sequence_count ||
        runtime_kv_block_table->lane_count > node_context->max_active_sequence_count ||
        runtime_kv_block_table->lane_stride != node_context->max_blocks_per_sequence ||
        runtime_kv_block_table->lane_capacity != node_context->max_blocks_per_sequence ||
        runtime_kv_block_table->physical_block_indices == 0 ||
        runtime_kv_block_table->lane_physical_block_counts == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52ResidentDecodeStageFakeRecordRuntimeKvBlockTable(
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table)
{
    if (runtime_kv_block_table == 0)
    {
        fake_stream->last_runtime_kv_block_table = 0;
        fake_stream->last_runtime_kv_physical_block_indices = 0;
        fake_stream->last_runtime_kv_block_token_count = 0u;
        fake_stream->last_runtime_kv_lane_count = 0u;
        return;
    }
    fake_stream->last_runtime_kv_block_table = runtime_kv_block_table;
    fake_stream->last_runtime_kv_physical_block_indices =
        runtime_kv_block_table->physical_block_indices;
    fake_stream->last_runtime_kv_block_token_count =
        runtime_kv_block_table->block_token_count;
    fake_stream->last_runtime_kv_lane_count = runtime_kv_block_table->lane_count;
}

static SparkStatus SparkGlm52ResidentDecodeStageFakeCopyFinalTokens(
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t final_token_stage,
    uint32_t active_sequence_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    uint32_t token_index;
    uint32_t token_count;

    if (final_token_stage == 0u)
        return SPARK_STATUS_OK;
    if (pipeline_slot == 0 || active_sequence_count == 0u ||
        completion == 0 || pipeline_slot->restricted_selected_token_ids == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    token_count = completion->requested_token_count;
    if (token_count == 0u ||
        token_count > SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY ||
        token_count >
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u))
        return SPARK_STATUS_INVALID_ARGUMENT;
    completion->token_ids[0u] = pipeline_slot->restricted_selected_token_ids[0u];
    if (token_count > 1u)
    {
        if (pipeline_slot->mtp_committed_token_ids == 0)
            return SPARK_STATUS_INVALID_ARGUMENT;
        for (token_index = 1u; token_index < token_count; ++token_index)
            completion->token_ids[token_index] =
                pipeline_slot->mtp_committed_token_ids[token_index - 1u];
    }
    completion->token_count = token_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ResidentDecodeStageBackendSubmit(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream;

    if (node_context == 0 || completion == 0 || completion->function == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        SparkGlm52ResidentDecodeStageFakeValidateRuntimeKvBlockTable(
            node_context,
            active_sequence_count,
            runtime_kv_block_table) != SPARK_STATUS_OK)
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
    SparkGlm52ResidentDecodeStageFakeRecordRuntimeKvBlockTable(
        fake_stream,
        runtime_kv_block_table);
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
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStageFrameContext *frame_context,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    const SparkGlm52ResidentDecodeStageNodeContext *first_node_context;
    const SparkGlm52ResidentDecodeStageNodeContext *completion_node_context;
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
    (void)frame_context;
    if (first_node_context == 0 ||
        first_node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= first_node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > first_node_context->max_active_sequence_count ||
        SparkGlm52ResidentDecodeStageFakeValidateRuntimeKvBlockTable(
            first_node_context,
            active_sequence_count,
            runtime_kv_block_table) != SPARK_STATUS_OK)
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
            layer_node_contexts[layer_index]->pipeline_slots == 0 ||
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
    completion_node_context = final_token_stage != 0u
        ? layer_node_contexts[layer_count - 1u]
        : first_node_context;
    if (SparkGlm52ResidentDecodeStageFakeCopyFinalTokens(
            &completion_node_context->pipeline_slots[pipeline_slot_index],
            final_token_stage,
            active_sequence_count,
            completion) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    fake_stream->submit_count += 1u;
    fake_stream->last_pipeline_slot = pipeline_slot_index;
    fake_stream->last_active_sequence_count = active_sequence_count;
    fake_stream->last_stage_slice_layer_count = layer_count;
    fake_stream->last_stage_slice_final_token_stage = final_token_stage;
    fake_stream->last_stage_slice_plan = stage_slice_plan;
    fake_stream->last_dspark_hidden_tap_frame_context_active =
        frame_context != 0 &&
        (frame_context->flags &
            SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_HIDDEN_TAPS) != 0u
            ? 1u
            : 0u;
    fake_stream->last_dspark_hidden_tap_frame_context = frame_context;
    SparkGlm52ResidentDecodeStageFakeRecordRuntimeKvBlockTable(
        fake_stream,
        runtime_kv_block_table);
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
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    SparkGlm52ResidentDecodeStageFakeStream *fake_stream;

    if (node_context == 0 || completion == 0 || completion->function == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count ||
        prompt_token_count == 0u ||
        SparkGlm52ResidentDecodeStageFakeValidateRuntimeKvBlockTable(
            node_context,
            active_sequence_count,
            runtime_kv_block_table) != SPARK_STATUS_OK ||
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
    fake_stream->last_bulk_prefill_prompt_token_offset = prompt_token_offset;
    fake_stream->last_bulk_prefill_prompt_token_count = prompt_token_count;
    fake_stream->last_prefill_frame_view = prefill_frame_view;
    SparkGlm52ResidentDecodeStageFakeRecordRuntimeKvBlockTable(
        fake_stream,
        runtime_kv_block_table);
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
    uint32_t prompt_token_offset,
    uint32_t prompt_token_count,
    const SparkGlm52KvBlockTableView *runtime_kv_block_table,
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_frame_view,
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
        active_sequence_count > first_node_context->max_active_sequence_count ||
        SparkGlm52ResidentDecodeStageFakeValidateRuntimeKvBlockTable(
            first_node_context,
            active_sequence_count,
            runtime_kv_block_table) != SPARK_STATUS_OK)
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
    fake_stream->last_bulk_prefill_prompt_token_offset = prompt_token_offset;
    fake_stream->last_bulk_prefill_prompt_token_count = prompt_token_count;
    fake_stream->last_prefill_frame_view = prefill_frame_view;
    SparkGlm52ResidentDecodeStageFakeRecordRuntimeKvBlockTable(
        fake_stream,
        runtime_kv_block_table);
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
