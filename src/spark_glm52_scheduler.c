#include "sparkpipe/spark_glm52_scheduler.h"

#include <string.h>

static uint32_t SparkGlm52SchedulerNormalizeQuantizationMode(
    uint32_t quantization_mode)
{
    if (quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO)
    {
        return SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    }
    return quantization_mode;
}

static uint32_t SparkGlm52SchedulerQuantizationModeIsSupported(
    uint32_t quantization_mode)
{
    return quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO ||
        quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT ||
        quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
}

static uint32_t SparkGlm52SchedulerMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52SchedulerMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left > right ? left : right;
}

static uint32_t SparkGlm52SchedulerRequestIsDecode(
    const SparkGlm52SchedulerRequest *request);

static uint32_t SparkGlm52SchedulerRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple)
{
    if (multiple == 0u)
    {
        return value;
    }
    return value - (value % multiple);
}

static uint32_t SparkGlm52SchedulerCeilDivideU32(
    uint32_t numerator,
    uint32_t denominator)
{
    if (denominator == 0u)
    {
        return 0u;
    }
    return (numerator + denominator - 1u) / denominator;
}

static uint64_t SparkGlm52SchedulerStageCostNs(
    const uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t final_stage_extra_cost_ns,
    const SparkGlm52StagePlanStage *stage)
{
    uint64_t stage_cost_ns;
    uint32_t layer_index;
    uint32_t layer_end;

    stage_cost_ns = 0u;
    layer_end = stage->first_layer_index + stage->layer_count;
    for (layer_index = stage->first_layer_index;
         layer_index < layer_end;
         ++layer_index)
    {
        stage_cost_ns += layer_cost_ns[layer_index];
    }
    if ((stage->flags & SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
    {
        stage_cost_ns += final_stage_extra_cost_ns;
    }
    return stage_cost_ns;
}

static uint32_t SparkGlm52SchedulerNormalizeQueueDepthPerSpark(
    uint32_t queue_depth_per_spark)
{
    if (queue_depth_per_spark == 0u)
    {
        return SPARK_GLM52_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK;
    }
    return queue_depth_per_spark;
}

static uint32_t SparkGlm52SchedulerNormalizeMeasuredProfileId(
    uint32_t measured_profile_id)
{
    if (measured_profile_id == 0u)
    {
        return SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    }
    return measured_profile_id;
}

static uint32_t SparkGlm52SchedulerNormalizeConfigurationFlags(
    uint32_t configuration_flags)
{
    if (configuration_flags == 0u)
    {
        return SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    }
    return configuration_flags;
}

static uint32_t SparkGlm52SchedulerNormalizePrefillBlockTokens(
    uint32_t prefix_cache_block_tokens)
{
    if (prefix_cache_block_tokens == 0u)
    {
        return SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    }
    return prefix_cache_block_tokens;
}

static uint32_t SparkGlm52SchedulerNormalizeMaxPrefillTokensPerStep(
    uint32_t max_prefill_tokens_per_step,
    uint32_t prefix_cache_block_tokens)
{
    uint32_t normalized_token_count;

    if (max_prefill_tokens_per_step == 0u)
    {
        normalized_token_count =
            SPARK_GLM52_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP;
    }
    else
    {
        normalized_token_count = max_prefill_tokens_per_step;
    }
    if (normalized_token_count < prefix_cache_block_tokens)
    {
        normalized_token_count = prefix_cache_block_tokens;
    }
    return normalized_token_count;
}

static uint32_t SparkGlm52SchedulerConfigurationFlagsAreValid(
    uint32_t configuration_flags)
{
    return (configuration_flags &
        ~SPARK_GLM52_SCHEDULER_CONFIGURATION_KNOWN_FLAGS) == 0u;
}

static uint32_t SparkGlm52SchedulerPromptCacheIsEnabled(
    const SparkGlm52Scheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE) != 0u;
}

static uint32_t SparkGlm52SchedulerChunkedPrefillIsEnabled(
    const SparkGlm52Scheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL) != 0u;
}

static uint32_t SparkGlm52SchedulerCudaGraphPaddingIsEnabled(
    const SparkGlm52Scheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING) != 0u;
}

static uint32_t SparkGlm52SchedulerMeasuredDecodeBucketSelectionIsEnabled(
    const SparkGlm52Scheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION) != 0u;
}

static SparkStatus SparkGlm52SchedulerBuildMeasuredPlanAndCosts(
    const SparkGlm52Scheduler *scheduler,
    uint32_t batch_bucket,
    SparkGlm52StagePlan *stage_plan,
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    SparkStatus status;

    if (scheduler == 0 || stage_plan == 0 || layer_cost_ns == 0 ||
        final_stage_extra_cost_ns_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52StagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        stage_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        layer_cost_ns,
        final_stage_extra_cost_ns_out);
}

static uint64_t SparkGlm52SchedulerPlanCriticalPathNs(
    const SparkGlm52StagePlan *stage_plan,
    const uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t final_stage_extra_cost_ns,
    uint32_t prefill_block_count)
{
    uint64_t critical_path_ns;
    uint32_t stage_index;

    if (prefill_block_count == 0u)
    {
        prefill_block_count = 1u;
    }

    critical_path_ns = 0u;
    for (stage_index = 0u;
         stage_index < stage_plan->stage_count;
         ++stage_index)
    {
        uint64_t stage_service_time_ns;

        stage_service_time_ns = SparkGlm52SchedulerStageCostNs(
            layer_cost_ns,
            final_stage_extra_cost_ns,
            &stage_plan->stages[stage_index]) * (uint64_t)prefill_block_count;
        if (stage_service_time_ns > critical_path_ns)
        {
            critical_path_ns = stage_service_time_ns;
        }
    }
    return critical_path_ns;
}

static SparkStatus SparkGlm52SchedulerSelectDecodeBatchBucket(
    const SparkGlm52Scheduler *scheduler,
    uint32_t active_sequence_count,
    uint32_t *batch_bucket_out,
    uint32_t *minimal_batch_bucket_out)
{
    static const uint32_t candidate_buckets[] = {
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16,
        SPARK_GLM52_STAGE_PLAN_BUCKET_B32,
        SPARK_GLM52_STAGE_PLAN_BUCKET_B64
    };
    uint64_t best_critical_path_ns;
    uint32_t best_bucket;
    uint32_t minimal_bucket;
    uint32_t candidate_index;
    SparkStatus status;

    if (batch_bucket_out == 0 || minimal_batch_bucket_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52StagePlanSelectBatchBucket(
        active_sequence_count,
        &minimal_bucket);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    *minimal_batch_bucket_out = minimal_bucket;
    if (!SparkGlm52SchedulerMeasuredDecodeBucketSelectionIsEnabled(scheduler) ||
        active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B16)
    {
        *batch_bucket_out = minimal_bucket;
        return SPARK_STATUS_OK;
    }

    best_bucket = minimal_bucket;
    best_critical_path_ns = UINT64_MAX;
    for (candidate_index = 0u;
         candidate_index < (uint32_t)(sizeof(candidate_buckets) / sizeof(candidate_buckets[0]));
         ++candidate_index)
    {
        SparkGlm52StagePlan candidate_stage_plan;
        uint64_t candidate_layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT];
        uint64_t candidate_final_stage_extra_cost_ns;
        uint64_t candidate_critical_path_ns;
        uint32_t candidate_bucket;

        candidate_bucket = candidate_buckets[candidate_index];
        if (candidate_bucket < active_sequence_count)
        {
            continue;
        }

        status = SparkGlm52SchedulerBuildMeasuredPlanAndCosts(
            scheduler,
            candidate_bucket,
            &candidate_stage_plan,
            candidate_layer_cost_ns,
            &candidate_final_stage_extra_cost_ns);
        if (status != SPARK_STATUS_OK)
        {
            continue;
        }

        candidate_critical_path_ns = SparkGlm52SchedulerPlanCriticalPathNs(
            &candidate_stage_plan,
            candidate_layer_cost_ns,
            candidate_final_stage_extra_cost_ns,
            1u);
        if (candidate_critical_path_ns < best_critical_path_ns ||
            (candidate_critical_path_ns == best_critical_path_ns &&
             candidate_bucket < best_bucket))
        {
            best_bucket = candidate_bucket;
            best_critical_path_ns = candidate_critical_path_ns;
        }
    }

    *batch_bucket_out = best_bucket;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52SchedulerSelectRequestBatchBucket(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    uint32_t *batch_bucket_out,
    uint32_t *minimal_batch_bucket_out)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52SchedulerRequestIsDecode(request))
    {
        return SparkGlm52SchedulerSelectDecodeBatchBucket(
            scheduler,
            request->active_sequence_count,
            batch_bucket_out,
            minimal_batch_bucket_out);
    }
    if (minimal_batch_bucket_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52StagePlanSelectBatchBucket(
        request->active_sequence_count,
        batch_bucket_out) == SPARK_STATUS_OK
            ? ((*minimal_batch_bucket_out = *batch_bucket_out), SPARK_STATUS_OK)
            : SPARK_STATUS_CAPACITY_EXCEEDED;
}

static uint32_t SparkGlm52SchedulerPrefillDecodeInterleaveIsEnabled(
    const SparkGlm52Scheduler *scheduler)
{
    return (scheduler->configuration_flags &
        SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE) != 0u;
}

static uint32_t SparkGlm52SchedulerStageHasCapacity(
    const SparkGlm52Scheduler *scheduler,
    uint32_t spark_index,
    uint32_t request_is_prefill)
{
    uint32_t reserved_decode_slots;

    if (spark_index >= scheduler->spark_count ||
        scheduler->spark_inflight_counts[spark_index] >=
            scheduler->queue_depth_per_spark)
    {
        return 0u;
    }

    reserved_decode_slots = SparkGlm52SchedulerPrefillDecodeInterleaveIsEnabled(
        scheduler) && request_is_prefill != 0u &&
        scheduler->queue_depth_per_spark > 1u
        ? 1u
        : 0u;
    if (reserved_decode_slots != 0u &&
        scheduler->spark_inflight_counts[spark_index] >=
            scheduler->queue_depth_per_spark - reserved_decode_slots)
    {
        return 0u;
    }
    return 1u;
}

static uint32_t SparkGlm52SchedulerDecodeBypassIsActive(
    const SparkGlm52Scheduler *scheduler)
{
    uint32_t spark_index;

    if (!SparkGlm52SchedulerPrefillDecodeInterleaveIsEnabled(scheduler))
    {
        return 0u;
    }
    for (spark_index = 0u; spark_index < scheduler->spark_count; ++spark_index)
    {
        if (scheduler->spark_inflight_counts[spark_index] != 0u)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkGlm52SchedulerRequestIsPrefill(
    const SparkGlm52SchedulerRequest *request)
{
    return (request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL) != 0u;
}

static uint32_t SparkGlm52SchedulerRequestIsDecode(
    const SparkGlm52SchedulerRequest *request)
{
    return (request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE) != 0u;
}

static SparkStatus SparkGlm52SchedulerLookupCachedPrefixTokenCount(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    uint32_t *cached_prefix_token_count_out)
{
    SparkGlm52PrefixCacheLookup lookup;
    SparkStatus status;

    if (cached_prefix_token_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *cached_prefix_token_count_out = 0u;
    if (!SparkGlm52SchedulerRequestIsPrefill(request) ||
        !SparkGlm52SchedulerPromptCacheIsEnabled(scheduler))
    {
        return SPARK_STATUS_OK;
    }
    status = SparkGlm52PrefixCacheProbePrompt(
        scheduler->prefix_cache,
        request->sequence_id,
        request->prompt_token_ids,
        request->prompt_token_count,
        &lookup);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    *cached_prefix_token_count_out = lookup.matched_token_count;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52SchedulerEffectiveComputedPromptTokenCount(
    const SparkGlm52SchedulerRequest *request,
    uint32_t cached_prefix_token_count)
{
    return SparkGlm52SchedulerMaximumU32(
        request->computed_prompt_token_count,
        cached_prefix_token_count);
}

static uint32_t SparkGlm52SchedulerRequestMaxPrefillTokensPerStep(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request)
{
    uint32_t max_prefill_tokens_per_step;

    max_prefill_tokens_per_step = request->max_scheduled_prompt_token_count;
    if (max_prefill_tokens_per_step == 0u)
    {
        max_prefill_tokens_per_step = scheduler->max_prefill_tokens_per_step;
    }
    if (max_prefill_tokens_per_step < scheduler->prefix_cache_block_tokens)
    {
        max_prefill_tokens_per_step = scheduler->prefix_cache_block_tokens;
    }
    return max_prefill_tokens_per_step;
}

static uint32_t SparkGlm52SchedulerScheduledPrefillTokenCount(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    uint32_t computed_prompt_token_count)
{
    uint32_t remaining_prompt_token_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t scheduled_prompt_token_count;

    remaining_prompt_token_count =
        request->prompt_token_count - computed_prompt_token_count;
    if (!SparkGlm52SchedulerChunkedPrefillIsEnabled(scheduler))
    {
        return remaining_prompt_token_count;
    }

    max_prefill_tokens_per_step =
        SparkGlm52SchedulerRequestMaxPrefillTokensPerStep(scheduler, request);
    if (remaining_prompt_token_count <= max_prefill_tokens_per_step)
    {
        return remaining_prompt_token_count;
    }

    scheduled_prompt_token_count = SparkGlm52SchedulerRoundDownToMultiple(
        max_prefill_tokens_per_step,
        scheduler->prefix_cache_block_tokens);
    if (scheduled_prompt_token_count == 0u)
    {
        scheduled_prompt_token_count = SparkGlm52SchedulerMinimumU32(
            remaining_prompt_token_count,
            scheduler->prefix_cache_block_tokens);
    }
    return scheduled_prompt_token_count;
}

static uint32_t SparkGlm52SchedulerPrefillBlockCount(
    const SparkGlm52Scheduler *scheduler,
    uint32_t prompt_token_count)
{
    if (prompt_token_count == 0u)
    {
        return 1u;
    }
    return SparkGlm52SchedulerCeilDivideU32(
        prompt_token_count,
        scheduler->prefix_cache_block_tokens);
}

static uint32_t SparkGlm52SchedulerBuildDecisionFlags(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    uint32_t batch_bucket,
    uint32_t cached_prefix_token_count,
    uint32_t scheduled_prompt_token_count,
    uint32_t remaining_prompt_token_count_after_step,
    uint32_t decode_bypass_active)
{
    uint32_t decision_flags;

    decision_flags = 0u;
    if (SparkGlm52SchedulerRequestIsDecode(request))
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_STEP;
        if (decode_bypass_active != 0u)
        {
            decision_flags |=
                SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL;
        }
    }
    if (SparkGlm52SchedulerRequestIsPrefill(request))
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP;
        if (SparkGlm52SchedulerPrefillDecodeInterleaveIsEnabled(scheduler) &&
            scheduler->queue_depth_per_spark > 1u)
        {
            decision_flags |=
                SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT;
        }
        if (remaining_prompt_token_count_after_step == 0u)
        {
            decision_flags |=
                SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK;
        }
        if (scheduled_prompt_token_count != 0u &&
            remaining_prompt_token_count_after_step != 0u)
        {
            decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK;
        }
        if (cached_prefix_token_count != 0u)
        {
            decision_flags |=
                SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED;
        }
    }
    if (SparkGlm52SchedulerCudaGraphPaddingIsEnabled(scheduler) &&
        request->active_sequence_count < batch_bucket)
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING;
    }
    return decision_flags;
}

static uint32_t SparkGlm52SchedulerBuildDispatchFlags(
    uint32_t decision_flags)
{
    uint32_t dispatch_flags;

    dispatch_flags = 0u;
    if ((decision_flags & SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_STEP) != 0u)
    {
        dispatch_flags |= SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE;
    }
    if ((decision_flags & SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP) != 0u)
    {
        dispatch_flags |= SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL;
    }
    if ((decision_flags & SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u)
    {
        dispatch_flags |= SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_CHUNK;
    }
    if ((decision_flags &
         SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK) != 0u)
    {
        dispatch_flags |=
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK;
    }
    if ((decision_flags &
         SPARK_GLM52_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING) != 0u)
    {
        dispatch_flags |=
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING;
    }
    if ((decision_flags &
         SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u)
    {
        dispatch_flags |=
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT;
    }
    if ((decision_flags &
         SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL) != 0u)
    {
        dispatch_flags |=
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL;
    }
    if ((decision_flags &
         SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u)
    {
        dispatch_flags |=
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET;
    }
    return dispatch_flags;
}

static SparkStatus SparkGlm52SchedulerReject(
    SparkGlm52Scheduler *scheduler,
    SparkGlm52SchedulerDecision *decision,
    SparkStatus rejected_status)
{
    if (decision != 0)
    {
        decision->accepted = 0u;
        decision->rejected_status = rejected_status;
    }
    if (scheduler != 0)
    {
        scheduler->rejected_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52SchedulerValidateConfiguration(
    const SparkGlm52SchedulerConfiguration *configuration)
{
    uint32_t configuration_flags;
    uint32_t prefix_cache_block_tokens;

    if (configuration == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    configuration_flags = SparkGlm52SchedulerNormalizeConfigurationFlags(
        configuration->configuration_flags);
    prefix_cache_block_tokens = SparkGlm52SchedulerNormalizePrefillBlockTokens(
        configuration->prefix_cache_block_tokens);
    if (configuration->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->spark_count != SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT ||
        configuration->reserved != 0u ||
        !SparkGlm52SchedulerQuantizationModeIsSupported(
            configuration->quantization_mode) ||
        !SparkGlm52SchedulerConfigurationFlagsAreValid(configuration_flags) ||
        prefix_cache_block_tokens == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration_flags &
            SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE) != 0u)
    {
        if (configuration->prefix_cache == 0 ||
            configuration->prefix_cache->abi_version !=
                SPARK_GLM52_PREFIX_CACHE_ABI_VERSION ||
            configuration->prefix_cache->descriptor_bytes !=
                SPARK_GLM52_PREFIX_CACHE_DESCRIPTOR_BYTES ||
            configuration->prefix_cache->block_token_count !=
                prefix_cache_block_tokens ||
            configuration->prefix_cache->entries == 0 ||
            configuration->prefix_cache->sequence_bindings == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if ((configuration_flags &
                SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED) != 0u &&
            configuration->prefix_cache->kv_cache_arena == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52SchedulerValidateRequest(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request)
{
    uint32_t is_decode;
    uint32_t is_prefill;

    if (scheduler == 0 || request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    is_decode = SparkGlm52SchedulerRequestIsDecode(request);
    is_prefill = SparkGlm52SchedulerRequestIsPrefill(request);
    if (request->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES ||
        request->reserved != 0u ||
        (request->flags & ~SPARK_GLM52_SCHEDULER_REQUEST_KNOWN_FLAGS) != 0u ||
        request->active_sequence_count == 0u ||
        is_decode == is_prefill ||
        (is_decode && request->prompt_token_count != 0u) ||
        (is_decode && request->computed_prompt_token_count != 0u) ||
        (is_decode && request->cached_prefix_token_count != 0u) ||
        (is_decode && request->max_scheduled_prompt_token_count != 0u) ||
        (is_decode && request->sequence_id != 0u) ||
        (is_decode && request->prompt_token_ids != 0) ||
        (is_prefill && request->prompt_token_count == 0u) ||
        (is_prefill && request->cached_prefix_token_count != 0u) ||
        (is_prefill &&
         SparkGlm52SchedulerPromptCacheIsEnabled(scheduler) &&
         (request->sequence_id == 0u || request->prompt_token_ids == 0)) ||
        (is_prefill &&
         SparkGlm52SchedulerPromptCacheIsEnabled(scheduler) &&
         request->computed_prompt_token_count != 0u) ||
        (is_prefill &&
         request->computed_prompt_token_count >= request->prompt_token_count))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerInitialize(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerConfiguration *configuration)
{
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
    uint32_t prefix_cache_block_tokens;
    uint32_t max_prefill_tokens_per_step;
    uint32_t configuration_flags;
    SparkStatus status;

    if (scheduler == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52SchedulerValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    queue_depth_per_spark = SparkGlm52SchedulerNormalizeQueueDepthPerSpark(
        configuration->queue_depth_per_spark);
    measured_profile_id = SparkGlm52SchedulerNormalizeMeasuredProfileId(
        configuration->measured_profile_id);
    quantization_mode = SparkGlm52SchedulerNormalizeQuantizationMode(
        configuration->quantization_mode);
    configuration_flags = SparkGlm52SchedulerNormalizeConfigurationFlags(
        configuration->configuration_flags);
    prefix_cache_block_tokens = SparkGlm52SchedulerNormalizePrefillBlockTokens(
        configuration->prefix_cache_block_tokens);
    max_prefill_tokens_per_step =
        SparkGlm52SchedulerNormalizeMaxPrefillTokensPerStep(
            configuration->max_prefill_tokens_per_step,
            prefix_cache_block_tokens);

    memset(scheduler, 0, sizeof(*scheduler));
    scheduler->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler->descriptor_bytes = SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES;
    scheduler->spark_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    scheduler->queue_depth_per_spark = queue_depth_per_spark;
    scheduler->measured_profile_id = measured_profile_id;
    scheduler->quantization_mode = quantization_mode;
    scheduler->max_prefill_tokens_per_step = max_prefill_tokens_per_step;
    scheduler->prefix_cache_block_tokens = prefix_cache_block_tokens;
    scheduler->configuration_flags = configuration_flags;
    scheduler->prefix_cache = configuration->prefix_cache;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerAdmit(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerRequest *request,
    SparkGlm52SchedulerDecision *decision)
{
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    uint64_t stage_cost_ns;
    uint64_t stage_service_time_ns;
    uint32_t batch_bucket;
    uint32_t minimal_batch_bucket;
    uint32_t measured_decode_bucket_selected;
    uint32_t stage_index;
    uint32_t prefill_block_count;
    uint32_t cached_prefix_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t scheduled_prompt_token_count;
    uint32_t remaining_prompt_token_count_after_step;
    uint32_t graph_sequence_padding_count;
    uint32_t decision_flags;
    uint32_t decode_bypass_active;
    uint32_t dispatch_flags;
    SparkGlm52PrefixCacheReservation prefix_cache_reservation;
    SparkGlm52PrefixCachePromptHash prefix_cache_parent_hash;
    SparkStatus status;

    if (scheduler == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52SchedulerValidateRequest(scheduler, request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(decision, 0, sizeof(*decision));
    decision->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    decision->descriptor_bytes = SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES;
    decision->quantization_mode = scheduler->quantization_mode;
    decision->spark_count = scheduler->spark_count;

    minimal_batch_bucket = 0u;
    status = SparkGlm52SchedulerSelectRequestBatchBucket(
        scheduler,
        request,
        &batch_bucket,
        &minimal_batch_bucket);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52SchedulerReject(scheduler, decision, status);
    }
    measured_decode_bucket_selected = SparkGlm52SchedulerRequestIsDecode(request) &&
        batch_bucket != minimal_batch_bucket;

    status = SparkGlm52SchedulerBuildMeasuredPlanAndCosts(
        scheduler,
        batch_bucket,
        &decision->stage_plan,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52SchedulerReject(scheduler, decision, status);
    }

    for (stage_index = 0u;
         stage_index < decision->stage_plan.stage_count;
         ++stage_index)
    {
        if (!SparkGlm52SchedulerStageHasCapacity(
                scheduler,
                stage_index,
                SparkGlm52SchedulerRequestIsPrefill(request)))
        {
            return SparkGlm52SchedulerReject(
                scheduler,
                decision,
                SPARK_STATUS_BUSY);
        }
    }

    cached_prefix_token_count = 0u;
    computed_prompt_token_count = 0u;
    scheduled_prompt_token_count = 0u;
    remaining_prompt_token_count_after_step = 0u;
    prefill_block_count = 1u;
    if (SparkGlm52SchedulerRequestIsPrefill(request))
    {
        status = SparkGlm52SchedulerLookupCachedPrefixTokenCount(
            scheduler,
            request,
            &cached_prefix_token_count);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52SchedulerReject(scheduler, decision, status);
        }
        computed_prompt_token_count =
            SparkGlm52SchedulerEffectiveComputedPromptTokenCount(
                request,
                cached_prefix_token_count);
        scheduled_prompt_token_count =
            SparkGlm52SchedulerScheduledPrefillTokenCount(
                scheduler,
                request,
                computed_prompt_token_count);
        remaining_prompt_token_count_after_step =
            request->prompt_token_count - computed_prompt_token_count -
            scheduled_prompt_token_count;
        prefill_block_count = SparkGlm52SchedulerPrefillBlockCount(
            scheduler,
            scheduled_prompt_token_count);
        if (SparkGlm52SchedulerPromptCacheIsEnabled(scheduler))
        {
            memset(&prefix_cache_reservation, 0, sizeof(prefix_cache_reservation));
            prefix_cache_reservation.abi_version =
                SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
            prefix_cache_reservation.descriptor_bytes =
                SPARK_GLM52_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
            prefix_cache_reservation.physical_block_indices =
                decision->kv_physical_block_indices;
            prefix_cache_reservation.physical_block_capacity =
                SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
            status = SparkGlm52PrefixCacheReservePrompt(
                scheduler->prefix_cache,
                request->sequence_id,
                request->prompt_token_ids,
                computed_prompt_token_count + scheduled_prompt_token_count,
                &prefix_cache_reservation);
            if (status != SPARK_STATUS_OK)
            {
                return SparkGlm52SchedulerReject(scheduler, decision, status);
            }
        }
    }

    graph_sequence_padding_count = batch_bucket - request->active_sequence_count;
    decode_bypass_active = SparkGlm52SchedulerRequestIsDecode(request)
        ? SparkGlm52SchedulerDecodeBypassIsActive(scheduler)
        : 0u;
    decision_flags = SparkGlm52SchedulerBuildDecisionFlags(
        scheduler,
        request,
        batch_bucket,
        cached_prefix_token_count,
        scheduled_prompt_token_count,
        remaining_prompt_token_count_after_step,
        decode_bypass_active);
    if (measured_decode_bucket_selected != 0u)
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET;
    }
    dispatch_flags = SparkGlm52SchedulerBuildDispatchFlags(decision_flags);

    decision->accepted = 1u;
    decision->batch_bucket = batch_bucket;
    decision->stage_count = decision->stage_plan.stage_count;
    decision->rejected_status = SPARK_STATUS_OK;
    decision->decision_flags = decision_flags;
    decision->active_sequence_count = request->active_sequence_count;
    decision->graph_sequence_capacity = batch_bucket;
    decision->graph_sequence_padding_count = graph_sequence_padding_count;
    decision->prompt_token_count = request->prompt_token_count;
    decision->computed_prompt_token_count = computed_prompt_token_count;
    decision->cached_prefix_token_count = cached_prefix_token_count;
    decision->prefix_cache_block_count = cached_prefix_token_count /
        scheduler->prefix_cache_block_tokens;
    decision->scheduled_prompt_token_offset = computed_prompt_token_count;
    decision->scheduled_prompt_token_count = scheduled_prompt_token_count;
    decision->remaining_prompt_token_count_after_step =
        remaining_prompt_token_count_after_step;
    decision->prefill_block_count = prefill_block_count;
    decision->cache_commit_token_count_after_step =
        computed_prompt_token_count + scheduled_prompt_token_count;
    if (SparkGlm52SchedulerRequestIsPrefill(request) &&
        SparkGlm52SchedulerPromptCacheIsEnabled(scheduler))
    {
        decision->kv_block_token_count = scheduler->prefix_cache_block_tokens;
        decision->kv_physical_block_count =
            prefix_cache_reservation.physical_block_count;
        decision->kv_cached_physical_block_count =
            prefix_cache_reservation.cached_physical_block_count;
        decision->kv_pending_physical_block_count =
            prefix_cache_reservation.pending_physical_block_count;
        decision->kv_block_table_token_count =
            prefix_cache_reservation.reserved_token_count;
        decision->prefix_cache_reservation_epoch =
            prefix_cache_reservation.reservation_epoch;
        decision->prefix_cache_result_hash =
            prefix_cache_reservation.last_block_hash;
        if (computed_prompt_token_count == 0u)
        {
            decision->prefix_cache_parent_hash =
                SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
        }
        else
        {
            status = SparkGlm52PrefixCacheHashPromptTokens(
                scheduler->prefix_cache_block_tokens,
                SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH,
                request->prompt_token_ids,
                computed_prompt_token_count,
                &prefix_cache_parent_hash);
            if (status != SPARK_STATUS_OK)
            {
                SparkGlm52PrefixCacheCancelReservation(
                    scheduler->prefix_cache,
                    request->sequence_id,
                    prefix_cache_reservation.reservation_epoch);
                return SparkGlm52SchedulerReject(scheduler, decision, status);
            }
            decision->prefix_cache_parent_hash =
                prefix_cache_parent_hash.prompt_hash;
        }
    }
    decision->sequence_id = request->sequence_id;
    decision->prompt_token_ids = request->prompt_token_ids;
    if (SparkGlm52SchedulerRequestIsPrefill(request))
    {
        decision->total_scheduled_token_count =
            (uint64_t)request->active_sequence_count *
            (uint64_t)scheduled_prompt_token_count;
    }
    else
    {
        decision->total_scheduled_token_count =
            (uint64_t)request->active_sequence_count;
    }
    decision->estimated_critical_path_ns = 0u;

    for (stage_index = 0u;
         stage_index < decision->stage_plan.stage_count;
         ++stage_index)
    {
        stage_cost_ns = SparkGlm52SchedulerStageCostNs(
            layer_cost_ns,
            final_stage_extra_cost_ns,
            &decision->stage_plan.stages[stage_index]);
        stage_service_time_ns = stage_cost_ns * (uint64_t)prefill_block_count;
        decision->dispatch_stages[stage_index].spark_index = stage_index;
        decision->dispatch_stages[stage_index].batch_bucket = batch_bucket;
        decision->dispatch_stages[stage_index].first_layer_index =
            decision->stage_plan.stages[stage_index].first_layer_index;
        decision->dispatch_stages[stage_index].layer_count =
            decision->stage_plan.stages[stage_index].layer_count;
        decision->dispatch_stages[stage_index].stage_flags =
            decision->stage_plan.stages[stage_index].flags;
        decision->dispatch_stages[stage_index].dispatch_flags = dispatch_flags;
        decision->dispatch_stages[stage_index].active_sequence_count =
            request->active_sequence_count;
        decision->dispatch_stages[stage_index].graph_sequence_capacity =
            batch_bucket;
        decision->dispatch_stages[stage_index].graph_sequence_padding_count =
            graph_sequence_padding_count;
        decision->dispatch_stages[stage_index].scheduled_prompt_token_offset =
            computed_prompt_token_count;
        decision->dispatch_stages[stage_index].scheduled_prompt_token_count =
            scheduled_prompt_token_count;
        decision->dispatch_stages[stage_index].cached_prefix_token_count =
            cached_prefix_token_count;
        decision->dispatch_stages[stage_index].estimated_service_time_ns =
            stage_service_time_ns;
        if (stage_service_time_ns > decision->estimated_critical_path_ns)
        {
            decision->estimated_critical_path_ns = stage_service_time_ns;
        }
        scheduler->spark_inflight_counts[stage_index] += 1u;
    }

    scheduler->admitted_count += 1u;
    if (SparkGlm52SchedulerRequestIsPrefill(request))
    {
        scheduler->scheduled_prefill_token_count +=
            decision->total_scheduled_token_count;
        scheduler->prefix_cache_hit_token_count +=
            (uint64_t)request->active_sequence_count *
            (uint64_t)cached_prefix_token_count;
        if ((decision_flags & SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u)
        {
            scheduler->chunked_prefill_count += 1u;
        }
        if ((decision_flags &
             SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u)
        {
            scheduler->interleaved_prefill_admission_count += 1u;
        }
        if (SparkGlm52SchedulerPromptCacheIsEnabled(scheduler))
        {
            scheduler->kv_block_reservation_count +=
                decision->kv_pending_physical_block_count;
            scheduler->kv_block_reservation_token_count +=
                decision->kv_block_table_token_count;
        }
    }
    else
    {
        scheduler->scheduled_decode_token_count +=
            decision->total_scheduled_token_count;
        if ((decision_flags &
             SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL) != 0u)
        {
            scheduler->decode_bypass_admission_count += 1u;
        }
        if ((decision_flags &
             SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u)
        {
            scheduler->measured_decode_bucket_selection_count += 1u;
            scheduler->measured_decode_bucket_padding_token_count +=
                decision->graph_sequence_padding_count;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerComplete(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision)
{
    uint32_t stage_index;
    uint32_t spark_index;

    if (scheduler == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES ||
        decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        decision->descriptor_bytes != SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES ||
        decision->accepted == 0u ||
        decision->stage_count == 0u ||
        decision->stage_count > scheduler->spark_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        spark_index = decision->dispatch_stages[stage_index].spark_index;
        if (spark_index >= scheduler->spark_count ||
            scheduler->spark_inflight_counts[spark_index] == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        spark_index = decision->dispatch_stages[stage_index].spark_index;
        scheduler->spark_inflight_counts[spark_index] -= 1u;
    }
    if ((decision->decision_flags &
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP) != 0u &&
        SparkGlm52SchedulerPromptCacheIsEnabled(scheduler) &&
        decision->cache_commit_token_count_after_step != 0u)
    {
        SparkStatus status;

        status = SparkGlm52PrefixCacheCommitReservation(
            scheduler->prefix_cache,
            decision->sequence_id,
            decision->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    scheduler->completed_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52SchedulerValidateAcceptedDecision(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision)
{
    if (scheduler == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES ||
        decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        decision->descriptor_bytes != SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES ||
        decision->accepted == 0u ||
        decision->stage_count == 0u ||
        decision->stage_count > scheduler->spark_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52SchedulerReleaseDecisionInflight(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision)
{
    uint32_t stage_index;

    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        uint32_t spark_index;

        spark_index = decision->dispatch_stages[stage_index].spark_index;
        if (spark_index >= scheduler->spark_count ||
            scheduler->spark_inflight_counts[spark_index] == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    for (stage_index = 0u; stage_index < decision->stage_count; ++stage_index)
    {
        uint32_t spark_index;

        spark_index = decision->dispatch_stages[stage_index].spark_index;
        scheduler->spark_inflight_counts[spark_index] -= 1u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerCancel(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision)
{
    SparkStatus status;

    status = SparkGlm52SchedulerValidateAcceptedDecision(scheduler, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52SchedulerReleaseDecisionInflight(scheduler, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((decision->decision_flags &
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP) != 0u &&
        SparkGlm52SchedulerPromptCacheIsEnabled(scheduler) &&
        decision->prefix_cache_reservation_epoch != 0u)
    {
        status = SparkGlm52PrefixCacheCancelReservation(
            scheduler->prefix_cache,
            decision->sequence_id,
            decision->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        scheduler->kv_block_cancel_count += 1u;
    }
    return SPARK_STATUS_OK;
}


static void SparkGlm52SchedulerRejectPrefillBatch(
    SparkGlm52SchedulerPrefillBatchDecision *batch_decision,
    const SparkGlm52SchedulerPrefillBatchRequest *batch_request,
    SparkStatus rejected_status)
{
    if (batch_decision == 0)
    {
        return;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES;
    batch_decision->accepted = 0u;
    batch_decision->rejected_status = rejected_status;
    if (batch_request != 0)
    {
        batch_decision->source_request_count = batch_request->request_count;
    }
}

static SparkStatus SparkGlm52SchedulerValidatePrefillBatchRequest(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchRequest *batch_request)
{
    uint32_t request_index;

    if (scheduler == 0 || batch_request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (batch_request->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        batch_request->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES ||
        batch_request->reserved != 0u ||
        batch_request->request_count == 0u ||
        batch_request->request_count > SPARK_GLM52_SCHEDULER_MAX_BATCH_REQUEST_COUNT ||
        batch_request->requests == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (request_index = 0u;
         request_index < batch_request->request_count;
         ++request_index)
    {
        const SparkGlm52SchedulerRequest *request;
        SparkStatus status;

        request = &batch_request->requests[request_index];
        status = SparkGlm52SchedulerValidateRequest(scheduler, request);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkGlm52SchedulerRequestIsPrefill(request) ||
            request->active_sequence_count != 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52SchedulerInitializePrefillBatchLane(
    SparkGlm52SchedulerPrefillBatchLane *lane,
    const SparkGlm52SchedulerRequest *request,
    uint32_t request_index,
    uint32_t active_sequence_offset,
    uint32_t cached_prefix_token_count,
    uint32_t computed_prompt_token_count,
    uint32_t scheduled_prompt_token_count,
    uint32_t prefix_cache_block_tokens,
    const SparkGlm52PrefixCacheReservation *prefix_cache_reservation,
    uint64_t prefix_cache_parent_hash)
{
    memset(lane, 0, sizeof(*lane));
    lane->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    lane->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_PREFILL_BATCH_LANE_DESCRIPTOR_BYTES;
    lane->request_index = request_index;
    lane->active_sequence_offset = active_sequence_offset;
    lane->active_sequence_count = request->active_sequence_count;
    lane->prompt_token_count = request->prompt_token_count;
    lane->computed_prompt_token_count = computed_prompt_token_count;
    lane->cached_prefix_token_count = cached_prefix_token_count;
    lane->scheduled_prompt_token_offset = computed_prompt_token_count;
    lane->scheduled_prompt_token_count = scheduled_prompt_token_count;
    lane->remaining_prompt_token_count_after_step =
        request->prompt_token_count - computed_prompt_token_count -
        scheduled_prompt_token_count;
    lane->cache_commit_token_count_after_step =
        computed_prompt_token_count + scheduled_prompt_token_count;
    lane->prefix_cache_block_count = cached_prefix_token_count /
        prefix_cache_block_tokens;
    lane->sequence_id = request->sequence_id;
    lane->prompt_token_ids = request->prompt_token_ids;
    lane->prefix_cache_parent_hash = prefix_cache_parent_hash;
    if (prefix_cache_reservation != 0)
    {
        lane->kv_block_token_count = prefix_cache_block_tokens;
        lane->kv_physical_block_count =
            prefix_cache_reservation->physical_block_count;
        lane->kv_cached_physical_block_count =
            prefix_cache_reservation->cached_physical_block_count;
        lane->kv_pending_physical_block_count =
            prefix_cache_reservation->pending_physical_block_count;
        lane->kv_block_table_token_count =
            prefix_cache_reservation->reserved_token_count;
        lane->prefix_cache_reservation_epoch =
            prefix_cache_reservation->reservation_epoch;
        lane->prefix_cache_result_hash =
            prefix_cache_reservation->last_block_hash;
    }
}

static SparkStatus SparkGlm52SchedulerCancelAcceptedPrefillBatchReservations(
    SparkGlm52Scheduler *scheduler,
    SparkGlm52SchedulerPrefillBatchDecision *batch_decision)
{
    uint32_t lane_index;

    if (!SparkGlm52SchedulerPromptCacheIsEnabled(scheduler))
    {
        return SPARK_STATUS_OK;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        SparkGlm52SchedulerPrefillBatchLane *lane;
        SparkStatus status;

        lane = &batch_decision->lanes[lane_index];
        if (lane->prefix_cache_reservation_epoch == 0u)
        {
            continue;
        }
        status = SparkGlm52PrefixCacheCancelReservation(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerAdmitPrefillBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchRequest *batch_request,
    SparkGlm52SchedulerPrefillBatchDecision *batch_decision)
{
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    uint64_t stage_cost_ns;
    uint64_t stage_service_time_ns;
    uint32_t batch_bucket;
    uint32_t request_index;
    uint32_t stage_index;
    uint32_t active_sequence_count;
    uint32_t packed_request_count;
    uint32_t graph_sequence_padding_count;
    uint32_t maximum_scheduled_prompt_token_count;
    uint32_t maximum_prefill_block_count;
    uint32_t decision_flags;
    uint32_t dispatch_flags;
    uint32_t any_remaining_prompt_tokens;
    uint32_t total_cached_prefix_token_count;
    uint32_t total_pending_physical_block_count;
    uint64_t total_scheduled_token_count;
    SparkGlm52StagePlan stage_plan;
    SparkStatus status;

    if (batch_decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES;

    if (scheduler == 0 ||
        scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES)
    {
        SparkGlm52SchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52SchedulerValidatePrefillBatchRequest(
        scheduler,
        batch_request);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52SchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return status;
    }

    active_sequence_count = 0u;
    packed_request_count = 0u;
    for (request_index = 0u;
         request_index < batch_request->request_count &&
             packed_request_count < SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT;
         ++request_index)
    {
        const SparkGlm52SchedulerRequest *request;

        request = &batch_request->requests[request_index];
        if (active_sequence_count + request->active_sequence_count >
            SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT)
        {
            break;
        }
        active_sequence_count += request->active_sequence_count;
        packed_request_count += 1u;
    }
    if (packed_request_count == 0u || active_sequence_count == 0u)
    {
        SparkGlm52SchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_CAPACITY_EXCEEDED);
        return SPARK_STATUS_OK;
    }

    status = SparkGlm52StagePlanSelectBatchBucket(
        active_sequence_count,
        &batch_bucket);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52SchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return SPARK_STATUS_OK;
    }

    status = SparkGlm52StagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        &batch_decision->stage_decision.stage_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52SchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return SPARK_STATUS_OK;
    }

    status = SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52SchedulerRejectPrefillBatch(
            batch_decision,
            batch_request,
            status);
        return SPARK_STATUS_OK;
    }

    for (stage_index = 0u;
         stage_index < batch_decision->stage_decision.stage_plan.stage_count;
         ++stage_index)
    {
        if (!SparkGlm52SchedulerStageHasCapacity(
                scheduler,
                stage_index,
                1u))
        {
            SparkGlm52SchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                SPARK_STATUS_BUSY);
            return SPARK_STATUS_OK;
        }
    }

    batch_decision->source_request_count = batch_request->request_count;
    batch_decision->packed_request_count = 0u;
    batch_decision->active_sequence_count = active_sequence_count;
    maximum_scheduled_prompt_token_count = 0u;
    maximum_prefill_block_count = 1u;
    any_remaining_prompt_tokens = 0u;
    total_cached_prefix_token_count = 0u;
    total_pending_physical_block_count = 0u;
    total_scheduled_token_count = 0u;

    for (request_index = 0u;
         request_index < packed_request_count;
         ++request_index)
    {
        const SparkGlm52SchedulerRequest *request;
        SparkGlm52PrefixCacheReservation prefix_cache_reservation;
        SparkGlm52PrefixCachePromptHash prefix_cache_parent_hash;
        uint32_t cached_prefix_token_count;
        uint32_t computed_prompt_token_count;
        uint32_t scheduled_prompt_token_count;
        uint32_t prefill_block_count;
        uint64_t parent_hash;

        request = &batch_request->requests[request_index];
        status = SparkGlm52SchedulerLookupCachedPrefixTokenCount(
            scheduler,
            request,
            &cached_prefix_token_count);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52SchedulerCancelAcceptedPrefillBatchReservations(
                scheduler,
                batch_decision);
            SparkGlm52SchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                status);
            return status;
        }
        computed_prompt_token_count =
            SparkGlm52SchedulerEffectiveComputedPromptTokenCount(
                request,
                cached_prefix_token_count);
        scheduled_prompt_token_count =
            SparkGlm52SchedulerScheduledPrefillTokenCount(
                scheduler,
                request,
                computed_prompt_token_count);
        if (scheduled_prompt_token_count == 0u)
        {
            SparkGlm52SchedulerCancelAcceptedPrefillBatchReservations(
                scheduler,
                batch_decision);
            SparkGlm52SchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                SPARK_STATUS_INVALID_ARGUMENT);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        memset(&prefix_cache_reservation, 0, sizeof(prefix_cache_reservation));
        prefix_cache_reservation.abi_version =
            SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
        prefix_cache_reservation.descriptor_bytes =
            SPARK_GLM52_PREFIX_CACHE_RESERVATION_DESCRIPTOR_BYTES;
        status = SparkGlm52PrefixCacheReservePrompt(
            scheduler->prefix_cache,
            request->sequence_id,
            request->prompt_token_ids,
            computed_prompt_token_count + scheduled_prompt_token_count,
            &prefix_cache_reservation);
        if (status != SPARK_STATUS_OK)
        {
            SparkGlm52SchedulerCancelAcceptedPrefillBatchReservations(
                scheduler,
                batch_decision);
            SparkGlm52SchedulerRejectPrefillBatch(
                batch_decision,
                batch_request,
                status);
            return status;
        }

        if (computed_prompt_token_count == 0u)
        {
            parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
        }
        else
        {
            status = SparkGlm52PrefixCacheHashPromptTokens(
                scheduler->prefix_cache_block_tokens,
                SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH,
                request->prompt_token_ids,
                computed_prompt_token_count,
                &prefix_cache_parent_hash);
            if (status != SPARK_STATUS_OK)
            {
                SparkGlm52PrefixCacheCancelReservation(
                    scheduler->prefix_cache,
                    request->sequence_id,
                    prefix_cache_reservation.reservation_epoch);
                SparkGlm52SchedulerCancelAcceptedPrefillBatchReservations(
                    scheduler,
                    batch_decision);
                SparkGlm52SchedulerRejectPrefillBatch(
                    batch_decision,
                    batch_request,
                    status);
                return status;
            }
            parent_hash = prefix_cache_parent_hash.prompt_hash;
        }

        SparkGlm52SchedulerInitializePrefillBatchLane(
            &batch_decision->lanes[request_index],
            request,
            request_index,
            request_index,
            cached_prefix_token_count,
            computed_prompt_token_count,
            scheduled_prompt_token_count,
            scheduler->prefix_cache_block_tokens,
            &prefix_cache_reservation,
            parent_hash);
        batch_decision->packed_request_count += 1u;
        total_cached_prefix_token_count += cached_prefix_token_count;
        total_pending_physical_block_count +=
            prefix_cache_reservation.pending_physical_block_count;
        total_scheduled_token_count += scheduled_prompt_token_count;
        if (scheduled_prompt_token_count > maximum_scheduled_prompt_token_count)
        {
            maximum_scheduled_prompt_token_count = scheduled_prompt_token_count;
        }
        prefill_block_count = SparkGlm52SchedulerPrefillBlockCount(
            scheduler,
            scheduled_prompt_token_count);
        if (prefill_block_count > maximum_prefill_block_count)
        {
            maximum_prefill_block_count = prefill_block_count;
        }
        if (batch_decision->lanes[request_index].remaining_prompt_token_count_after_step != 0u)
        {
            any_remaining_prompt_tokens = 1u;
        }
    }

    graph_sequence_padding_count = batch_bucket - active_sequence_count;
    decision_flags = SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP |
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK;
    if (any_remaining_prompt_tokens != 0u)
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK;
    }
    else
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK;
    }
    if (total_cached_prefix_token_count != 0u)
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED;
    }
    if (graph_sequence_padding_count != 0u &&
        SparkGlm52SchedulerCudaGraphPaddingIsEnabled(scheduler))
    {
        decision_flags |= SPARK_GLM52_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING;
    }
    if (SparkGlm52SchedulerPrefillDecodeInterleaveIsEnabled(scheduler) &&
        scheduler->queue_depth_per_spark > 1u)
    {
        decision_flags |=
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT;
    }
    dispatch_flags = SparkGlm52SchedulerBuildDispatchFlags(decision_flags) |
        SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK;

    batch_decision->accepted = 1u;
    batch_decision->rejected_status = SPARK_STATUS_OK;
    batch_decision->batch_bucket = batch_bucket;
    batch_decision->graph_sequence_capacity = batch_bucket;
    batch_decision->graph_sequence_padding_count = graph_sequence_padding_count;
    batch_decision->decision_flags = decision_flags;
    batch_decision->maximum_scheduled_prompt_token_count =
        maximum_scheduled_prompt_token_count;
    batch_decision->total_scheduled_token_count = total_scheduled_token_count;

    stage_plan = batch_decision->stage_decision.stage_plan;
    memset(&batch_decision->stage_decision, 0, sizeof(batch_decision->stage_decision));
    batch_decision->stage_decision.stage_plan = stage_plan;
    batch_decision->stage_decision.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_decision->stage_decision.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES;
    batch_decision->stage_decision.accepted = 1u;
    batch_decision->stage_decision.batch_bucket = batch_bucket;
    batch_decision->stage_decision.quantization_mode = scheduler->quantization_mode;
    batch_decision->stage_decision.spark_count = scheduler->spark_count;
    batch_decision->stage_decision.stage_count =
        batch_decision->stage_decision.stage_plan.stage_count;
    batch_decision->stage_decision.rejected_status = SPARK_STATUS_OK;
    batch_decision->stage_decision.decision_flags = decision_flags;
    batch_decision->stage_decision.active_sequence_count = active_sequence_count;
    batch_decision->stage_decision.graph_sequence_capacity = batch_bucket;
    batch_decision->stage_decision.graph_sequence_padding_count =
        graph_sequence_padding_count;
    batch_decision->stage_decision.scheduled_prompt_token_count =
        maximum_scheduled_prompt_token_count;
    batch_decision->stage_decision.prefill_block_count =
        maximum_prefill_block_count;
    batch_decision->stage_decision.total_scheduled_token_count =
        total_scheduled_token_count;

    for (stage_index = 0u;
         stage_index < batch_decision->stage_decision.stage_count;
         ++stage_index)
    {
        stage_cost_ns = SparkGlm52SchedulerStageCostNs(
            layer_cost_ns,
            final_stage_extra_cost_ns,
            &batch_decision->stage_decision.stage_plan.stages[stage_index]);
        stage_service_time_ns = stage_cost_ns *
            (uint64_t)maximum_prefill_block_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].spark_index =
            stage_index;
        batch_decision->stage_decision.dispatch_stages[stage_index].batch_bucket =
            batch_bucket;
        batch_decision->stage_decision.dispatch_stages[stage_index].first_layer_index =
            batch_decision->stage_decision.stage_plan.stages[stage_index].first_layer_index;
        batch_decision->stage_decision.dispatch_stages[stage_index].layer_count =
            batch_decision->stage_decision.stage_plan.stages[stage_index].layer_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].stage_flags =
            batch_decision->stage_decision.stage_plan.stages[stage_index].flags;
        batch_decision->stage_decision.dispatch_stages[stage_index].dispatch_flags =
            dispatch_flags;
        batch_decision->stage_decision.dispatch_stages[stage_index].active_sequence_count =
            active_sequence_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].graph_sequence_capacity =
            batch_bucket;
        batch_decision->stage_decision.dispatch_stages[stage_index].graph_sequence_padding_count =
            graph_sequence_padding_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].scheduled_prompt_token_count =
            maximum_scheduled_prompt_token_count;
        batch_decision->stage_decision.dispatch_stages[stage_index].estimated_service_time_ns =
            stage_service_time_ns;
        if (stage_service_time_ns > batch_decision->estimated_critical_path_ns)
        {
            batch_decision->estimated_critical_path_ns = stage_service_time_ns;
        }
        scheduler->spark_inflight_counts[stage_index] += 1u;
    }
    batch_decision->stage_decision.estimated_critical_path_ns =
        batch_decision->estimated_critical_path_ns;

    scheduler->admitted_count += 1u;
    scheduler->scheduled_prefill_token_count += total_scheduled_token_count;
    scheduler->prefix_cache_hit_token_count += total_cached_prefix_token_count;
    if ((decision_flags & SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u)
    {
        scheduler->chunked_prefill_count += 1u;
    }
    if ((decision_flags &
         SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u)
    {
        scheduler->interleaved_prefill_admission_count += 1u;
    }
    scheduler->kv_block_reservation_count += total_pending_physical_block_count;
    scheduler->kv_block_reservation_token_count += total_scheduled_token_count;
    scheduler->adaptive_prefill_pack_admission_count += 1u;
    scheduler->adaptive_prefill_pack_request_count +=
        batch_decision->packed_request_count;
    scheduler->adaptive_prefill_pack_padding_token_count +=
        graph_sequence_padding_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52SchedulerValidateAcceptedPrefillBatchDecision(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision)
{
    if (scheduler == 0 || batch_decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES ||
        batch_decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        batch_decision->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES ||
        batch_decision->accepted == 0u ||
        batch_decision->packed_request_count == 0u ||
        batch_decision->packed_request_count >
            SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT ||
        batch_decision->stage_decision.accepted == 0u ||
        (batch_decision->decision_flags &
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerCompletePrefillBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkGlm52SchedulerValidateAcceptedPrefillBatchDecision(
        scheduler,
        batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52SchedulerReleaseDecisionInflight(
        scheduler,
        &batch_decision->stage_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        const SparkGlm52SchedulerPrefillBatchLane *lane;

        lane = &batch_decision->lanes[lane_index];
        if (lane->prefix_cache_reservation_epoch == 0u)
        {
            continue;
        }
        status = SparkGlm52PrefixCacheCommitReservation(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    scheduler->completed_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerCancelPrefillBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkGlm52SchedulerValidateAcceptedPrefillBatchDecision(
        scheduler,
        batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52SchedulerReleaseDecisionInflight(
        scheduler,
        &batch_decision->stage_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        const SparkGlm52SchedulerPrefillBatchLane *lane;

        lane = &batch_decision->lanes[lane_index];
        if (lane->prefix_cache_reservation_epoch == 0u)
        {
            continue;
        }
        status = SparkGlm52PrefixCacheCancelReservation(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->prefix_cache_reservation_epoch);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        scheduler->kv_block_cancel_count += 1u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerBuildPrefillBatchKvBlockTables(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerPrefillBatchDecision *batch_decision,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkGlm52SchedulerValidateAcceptedPrefillBatchDecision(
        scheduler,
        batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkGlm52SchedulerPromptCacheIsEnabled(scheduler) ||
        physical_block_indices == 0 ||
        lane_physical_block_counts == 0 ||
        lane_count_capacity < batch_decision->packed_request_count ||
        lane_capacity == 0u ||
        lane_stride < lane_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u; lane_index < lane_count_capacity; ++lane_index)
    {
        lane_physical_block_counts[lane_index] = 0u;
    }
    for (lane_index = 0u;
         lane_index < batch_decision->packed_request_count;
         ++lane_index)
    {
        const SparkGlm52SchedulerPrefillBatchLane *lane;

        lane = &batch_decision->lanes[lane_index];
        if (lane->sequence_id == 0u || lane->kv_block_table_token_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        status = SparkGlm52PrefixCacheBuildPhysicalBlockTable(
            scheduler->prefix_cache,
            lane->sequence_id,
            lane->kv_block_table_token_count,
            &physical_block_indices[(uint64_t)lane_index * lane_stride],
            lane_capacity,
            &lane_physical_block_counts[lane_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerBuildKvBlockTable(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerDecision *decision,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *physical_block_count_out)
{
    SparkStatus status;

    status = SparkGlm52SchedulerValidateAcceptedDecision(scheduler, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkGlm52SchedulerPromptCacheIsEnabled(scheduler) ||
        (decision->decision_flags &
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_STEP) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        scheduler->prefix_cache,
        decision->sequence_id,
        decision->kv_block_table_token_count,
        physical_block_indices,
        physical_block_capacity,
        physical_block_count_out);
}

SparkStatus SparkGlm52SchedulerReleaseSequence(
    SparkGlm52Scheduler *scheduler,
    uint64_t sequence_id)
{
    if (scheduler == 0 || sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52SchedulerPromptCacheIsEnabled(scheduler))
    {
        return SPARK_STATUS_OK;
    }
    return SparkGlm52PrefixCacheReleaseSequence(
        scheduler->prefix_cache,
        sequence_id);
}

static void SparkGlm52SchedulerInitializePackedRequest(
    SparkGlm52SchedulerPackedRequest *packed_request,
    uint32_t request_index,
    uint32_t active_sequence_offset,
    const SparkGlm52SchedulerRequest *request)
{
    memset(packed_request, 0, sizeof(*packed_request));
    packed_request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    packed_request->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_PACKED_REQUEST_DESCRIPTOR_BYTES;
    packed_request->request_index = request_index;
    packed_request->active_sequence_offset = active_sequence_offset;
    packed_request->active_sequence_count = request->active_sequence_count;
    packed_request->request_flags = request->flags;
    packed_request->scheduled_token_count = request->active_sequence_count;
    packed_request->total_scheduled_token_count = request->active_sequence_count;
}

static SparkStatus SparkGlm52SchedulerValidateDecodeBatchRequest(
    const SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchRequest *batch_request)
{
    uint32_t request_index;

    if (scheduler == 0 || batch_request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (batch_request->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        batch_request->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES ||
        batch_request->reserved != 0u ||
        batch_request->request_count == 0u ||
        batch_request->request_count > SPARK_GLM52_SCHEDULER_MAX_BATCH_REQUEST_COUNT ||
        batch_request->requests == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (request_index = 0u;
         request_index < batch_request->request_count;
         ++request_index)
    {
        SparkStatus status;

        status = SparkGlm52SchedulerValidateRequest(
            scheduler,
            &batch_request->requests[request_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (!SparkGlm52SchedulerRequestIsDecode(
                &batch_request->requests[request_index]))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52SchedulerRejectDecodeBatch(
    SparkGlm52SchedulerBatchDecision *batch_decision,
    const SparkGlm52SchedulerBatchRequest *batch_request,
    SparkStatus rejected_status)
{
    if (batch_decision == 0)
    {
        return;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES;
    batch_decision->accepted = 0u;
    batch_decision->rejected_status = rejected_status;
    if (batch_request != 0)
    {
        batch_decision->source_request_count = batch_request->request_count;
    }
}

SparkStatus SparkGlm52SchedulerAdmitDecodeBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchRequest *batch_request,
    SparkGlm52SchedulerBatchDecision *batch_decision)
{
    SparkGlm52SchedulerRequest aggregate_request;
    SparkStatus status;
    uint32_t request_index;
    uint32_t active_sequence_offset;
    uint32_t packed_request_count;
    uint32_t active_sequence_count;

    if (batch_decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(batch_decision, 0, sizeof(*batch_decision));
    batch_decision->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_decision->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES;

    if (scheduler == 0 ||
        scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES)
    {
        SparkGlm52SchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_INVALID_ARGUMENT);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52SchedulerValidateDecodeBatchRequest(
        scheduler,
        batch_request);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52SchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            status);
        return status;
    }

    active_sequence_offset = 0u;
    packed_request_count = 0u;
    active_sequence_count = 0u;
    for (request_index = 0u;
         request_index < batch_request->request_count &&
             packed_request_count < SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT;
         ++request_index)
    {
        const SparkGlm52SchedulerRequest *request;

        request = &batch_request->requests[request_index];
        if (request->active_sequence_count >
            SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT)
        {
            SparkGlm52SchedulerRejectDecodeBatch(
                batch_decision,
                batch_request,
                SPARK_STATUS_CAPACITY_EXCEEDED);
            return SPARK_STATUS_OK;
        }
        if (active_sequence_count + request->active_sequence_count >
            SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT)
        {
            break;
        }
        SparkGlm52SchedulerInitializePackedRequest(
            &batch_decision->packed_requests[packed_request_count],
            request_index,
            active_sequence_offset,
            request);
        active_sequence_offset += request->active_sequence_count;
        active_sequence_count += request->active_sequence_count;
        packed_request_count += 1u;
    }
    if (packed_request_count == 0u || active_sequence_count == 0u)
    {
        SparkGlm52SchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            SPARK_STATUS_CAPACITY_EXCEEDED);
        return SPARK_STATUS_OK;
    }

    memset(&aggregate_request, 0, sizeof(aggregate_request));
    aggregate_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    aggregate_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    aggregate_request.active_sequence_count = active_sequence_count;
    aggregate_request.flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE;

    status = SparkGlm52SchedulerAdmit(
        scheduler,
        &aggregate_request,
        &batch_decision->stage_decision);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52SchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            status);
        return status;
    }
    if (batch_decision->stage_decision.accepted == 0u)
    {
        SparkStatus rejected_status;

        rejected_status = batch_decision->stage_decision.rejected_status;
        SparkGlm52SchedulerRejectDecodeBatch(
            batch_decision,
            batch_request,
            rejected_status);
        return SPARK_STATUS_OK;
    }

    batch_decision->accepted = 1u;
    batch_decision->rejected_status = SPARK_STATUS_OK;
    batch_decision->source_request_count = batch_request->request_count;
    batch_decision->packed_request_count = packed_request_count;
    batch_decision->batch_bucket = batch_decision->stage_decision.batch_bucket;
    batch_decision->active_sequence_count = active_sequence_count;
    batch_decision->graph_sequence_capacity =
        batch_decision->stage_decision.graph_sequence_capacity;
    batch_decision->graph_sequence_padding_count =
        batch_decision->stage_decision.graph_sequence_padding_count;
    batch_decision->decision_flags =
        batch_decision->stage_decision.decision_flags |
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK;
    batch_decision->total_scheduled_token_count =
        batch_decision->stage_decision.total_scheduled_token_count;
    batch_decision->estimated_critical_path_ns =
        batch_decision->stage_decision.estimated_critical_path_ns;

    batch_decision->stage_decision.decision_flags |=
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK;
    for (request_index = 0u;
         request_index < batch_decision->stage_decision.stage_count;
         ++request_index)
    {
        batch_decision->stage_decision.dispatch_stages[request_index].dispatch_flags |=
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK;
    }

    scheduler->adaptive_decode_pack_admission_count += 1u;
    scheduler->adaptive_decode_pack_request_count += packed_request_count;
    scheduler->adaptive_decode_pack_padding_token_count +=
        batch_decision->graph_sequence_padding_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52SchedulerCompleteDecodeBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchDecision *batch_decision)
{
    if (batch_decision == 0 ||
        batch_decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        batch_decision->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES ||
        batch_decision->accepted == 0u ||
        batch_decision->packed_request_count == 0u ||
        (batch_decision->decision_flags &
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52SchedulerComplete(
        scheduler,
        &batch_decision->stage_decision);
}

SparkStatus SparkGlm52SchedulerCancelDecodeBatch(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerBatchDecision *batch_decision)
{
    if (batch_decision == 0 ||
        batch_decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        batch_decision->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES ||
        batch_decision->accepted == 0u ||
        batch_decision->packed_request_count == 0u ||
        (batch_decision->decision_flags &
            SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52SchedulerReleaseDecisionInflight(
        scheduler,
        &batch_decision->stage_decision);
}
