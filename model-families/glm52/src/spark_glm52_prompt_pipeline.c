#include "sparkpipe/spark_glm52_prompt_pipeline.h"

#include <string.h>

void SparkGlm52PromptPipelineInitializeRunStats(
    SparkGlm52PromptPipelineRunStats *stats)
{
    if (stats == 0)
    {
        return;
    }
    memset(stats, 0, sizeof(*stats));
    stats->abi_version = SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION;
    stats->descriptor_bytes =
        SPARK_GLM52_PROMPT_PIPELINE_RUN_STATS_DESCRIPTOR_BYTES;
}

static uint32_t SparkGlm52PromptPipelineMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left >= right ? left : right;
}

static SparkStatus SparkGlm52PromptPipelineValidateConfiguration(
    const SparkGlm52PromptPipelineConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_PROMPT_PIPELINE_CONFIGURATION_DESCRIPTOR_BYTES ||
        (configuration->run_flags &
            ~SPARK_GLM52_PROMPT_PIPELINE_RUN_KNOWN_FLAGS) != 0u ||
        configuration->request_api == 0 ||
        configuration->host_prefill_token_ids == 0 ||
        configuration->host_prefill_token_stride == 0u ||
        configuration->host_prefill_lane_capacity == 0u ||
        configuration->host_physical_block_indices == 0 ||
        configuration->kv_block_lane_stride == 0u ||
        configuration->kv_block_lane_capacity == 0u ||
        configuration->kv_block_lane_stride < configuration->kv_block_lane_capacity ||
        configuration->lane_physical_block_counts == 0 ||
        configuration->lane_count_capacity == 0u ||
        configuration->prefill_function == 0 ||
        configuration->decode_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52PromptPipelineInvokePrefill(
    const SparkGlm52PromptPipelineConfiguration *configuration,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t step_index,
    SparkGlm52PromptPipelineRunStats *stats)
{
    SparkGlm52RequestApiPrefillDispatchView prefill_view;
    SparkGlm52KvBlockTableView block_table_view;
    SparkGlm52PromptPipelinePrefillDispatch prefill_dispatch;
    SparkStatus status;

    status = SparkGlm52RequestApiDescribePrefillDispatch(
        dispatch,
        &prefill_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefill_view.lane_count > configuration->host_prefill_lane_capacity ||
        prefill_view.lane_count > configuration->lane_count_capacity ||
        prefill_view.prompt_token_stride > configuration->host_prefill_token_stride)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SparkGlm52RequestApiCopyPrefillDispatchTokenIds(
        dispatch,
        configuration->host_prefill_token_ids,
        configuration->host_prefill_token_stride,
        configuration->host_prefill_lane_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52RequestApiBuildDispatchKvBlockTableView(
        configuration->request_api,
        dispatch,
        configuration->host_physical_block_indices,
        configuration->execution_physical_block_indices,
        configuration->kv_block_lane_stride,
        configuration->kv_block_lane_capacity,
        configuration->lane_physical_block_counts,
        configuration->lane_count_capacity,
        &block_table_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&prefill_dispatch, 0, sizeof(prefill_dispatch));
    prefill_dispatch.abi_version = SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION;
    prefill_dispatch.descriptor_bytes =
        SPARK_GLM52_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES;
    prefill_dispatch.step_index = step_index;
    prefill_dispatch.dispatch_kind = dispatch->kind;
    prefill_dispatch.active_sequence_count = prefill_view.active_sequence_count;
    prefill_dispatch.lane_count = prefill_view.lane_count;
    prefill_dispatch.prompt_token_offset = prefill_view.prompt_token_offset;
    prefill_dispatch.prompt_token_count = prefill_view.prompt_token_count;
    prefill_dispatch.prompt_token_stride = prefill_view.prompt_token_stride;
    prefill_dispatch.host_token_stride = configuration->host_prefill_token_stride;
    prefill_dispatch.request_dispatch = dispatch;
    prefill_dispatch.prefill_view = &prefill_view;
    prefill_dispatch.host_token_ids = configuration->host_prefill_token_ids;
    prefill_dispatch.kv_block_table_view = &block_table_view;

    status = configuration->prefill_function(
        configuration->callback_context,
        &prefill_dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    stats->prefill_dispatch_count += 1u;
    stats->prefill_token_count += prefill_view.prompt_token_count;
    stats->maximum_prefill_token_count = SparkGlm52PromptPipelineMaximumU32(
        stats->maximum_prefill_token_count,
        prefill_view.prompt_token_count);
    stats->maximum_prefill_lane_count = SparkGlm52PromptPipelineMaximumU32(
        stats->maximum_prefill_lane_count,
        prefill_view.lane_count);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52PromptPipelineInvokeDecode(
    const SparkGlm52PromptPipelineConfiguration *configuration,
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52PromptPipelineRunStats *stats)
{
    SparkStatus status;

    status = configuration->decode_function(
        configuration->callback_context,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (dispatch->kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        stats->speculative_verify_dispatch_count += 1u;
    }
    else
    {
        stats->decode_dispatch_count += 1u;
    }
    stats->reached_decode_dispatch = 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52PromptPipelineRun(
    const SparkGlm52PromptPipelineConfiguration *configuration,
    SparkGlm52PromptPipelineRunStats *stats)
{
    SparkGlm52PromptPipelineRunStats local_stats;
    uint32_t step_index;
    uint32_t max_dispatch_steps;
    SparkStatus status;

    status = SparkGlm52PromptPipelineValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52PromptPipelineInitializeRunStats(&local_stats);
    max_dispatch_steps = configuration->max_dispatch_steps != 0u ?
        configuration->max_dispatch_steps :
        SPARK_GLM52_PROMPT_PIPELINE_DEFAULT_MAX_DISPATCH_STEPS;

    for (step_index = 0u; step_index < max_dispatch_steps; ++step_index)
    {
        SparkGlm52RequestApiDispatch dispatch;

        status = SparkGlm52RequestApiScheduleNext(
            configuration->request_api,
            &dispatch);
        if (status != SPARK_STATUS_OK)
        {
            if (stats != 0)
            {
                *stats = local_stats;
            }
            return status;
        }
        if (dispatch.accepted == 0u ||
            dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_NONE)
        {
            if (stats != 0)
            {
                *stats = local_stats;
            }
            return SPARK_STATUS_BUSY;
        }

        local_stats.completed_dispatch_count += 1u;
        local_stats.last_dispatch_kind = dispatch.kind;

        if (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
            dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
        {
            status = SparkGlm52PromptPipelineInvokePrefill(
                configuration,
                &dispatch,
                step_index,
                &local_stats);
        }
        else if (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
            dispatch.kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
        {
            status = SparkGlm52PromptPipelineInvokeDecode(
                configuration,
                &dispatch,
                &local_stats);
        }
        else
        {
            status = SPARK_STATUS_INVALID_ARGUMENT;
        }

        if (status == SPARK_STATUS_OK)
        {
            status = SparkGlm52RequestApiCompleteDispatch(
                configuration->request_api,
                &dispatch);
        }
        else
        {
            (void)SparkGlm52RequestApiCancelDispatch(
                configuration->request_api,
                &dispatch);
        }
        if (status != SPARK_STATUS_OK)
        {
            if (stats != 0)
            {
                *stats = local_stats;
            }
            return status;
        }

        if (local_stats.reached_decode_dispatch != 0u &&
            (configuration->run_flags &
                SPARK_GLM52_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH) != 0u)
        {
            if (stats != 0)
            {
                *stats = local_stats;
            }
            return SPARK_STATUS_OK;
        }
    }

    if (stats != 0)
    {
        *stats = local_stats;
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}
