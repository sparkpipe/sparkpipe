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

static uint32_t SparkGlm52SchedulerPrefillBlockCount(
    uint32_t prompt_token_count)
{
    if (prompt_token_count == 0u)
    {
        return 1u;
    }
    return (prompt_token_count +
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS - 1u) /
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
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

SparkStatus SparkGlm52SchedulerInitialize(
    SparkGlm52Scheduler *scheduler,
    const SparkGlm52SchedulerConfiguration *configuration)
{
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;

    if (scheduler == 0 || configuration == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->spark_count != SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT ||
        !SparkGlm52SchedulerQuantizationModeIsSupported(
            configuration->quantization_mode))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    queue_depth_per_spark = configuration->queue_depth_per_spark;
    if (queue_depth_per_spark == 0u)
    {
        queue_depth_per_spark =
            SPARK_GLM52_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK;
    }
    measured_profile_id = configuration->measured_profile_id;
    if (measured_profile_id == 0u)
    {
        measured_profile_id = SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    }
    quantization_mode = SparkGlm52SchedulerNormalizeQuantizationMode(
        configuration->quantization_mode);

    memset(scheduler, 0, sizeof(*scheduler));
    scheduler->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler->descriptor_bytes = SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES;
    scheduler->spark_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    scheduler->queue_depth_per_spark = queue_depth_per_spark;
    scheduler->measured_profile_id = measured_profile_id;
    scheduler->quantization_mode = quantization_mode;
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
    uint32_t stage_index;
    uint32_t prefill_block_count;
    SparkStatus status;

    if (scheduler == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES ||
        request->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES ||
        request->reserved != 0u ||
        (request->flags & ~SPARK_GLM52_SCHEDULER_REQUEST_KNOWN_FLAGS) != 0u ||
        request->active_sequence_count == 0u ||
        ((request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE) != 0u &&
         (request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL) != 0u) ||
        ((request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL) != 0u &&
         request->prompt_token_count == 0u) ||
        ((request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE) != 0u &&
         request->prompt_token_count != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(decision, 0, sizeof(*decision));
    decision->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    decision->descriptor_bytes = SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES;
    decision->quantization_mode = scheduler->quantization_mode;
    decision->spark_count = scheduler->spark_count;

    status = SparkGlm52StagePlanSelectBatchBucket(
        request->active_sequence_count,
        &batch_bucket);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52SchedulerReject(scheduler, decision, status);
    }

    status = SparkGlm52StagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
        &decision->stage_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52SchedulerReject(scheduler, decision, status);
    }

    status = SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
        scheduler->measured_profile_id,
        batch_bucket,
        scheduler->quantization_mode,
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
        if (scheduler->spark_inflight_counts[stage_index] >=
            scheduler->queue_depth_per_spark)
        {
            return SparkGlm52SchedulerReject(
                scheduler,
                decision,
                SPARK_STATUS_BUSY);
        }
    }

    prefill_block_count =
        ((request->flags & SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL) != 0u)
        ? SparkGlm52SchedulerPrefillBlockCount(request->prompt_token_count)
        : 1u;
    decision->accepted = 1u;
    decision->batch_bucket = batch_bucket;
    decision->stage_count = decision->stage_plan.stage_count;
    decision->rejected_status = SPARK_STATUS_OK;
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
        decision->dispatch_stages[stage_index].estimated_service_time_ns =
            stage_service_time_ns;
        if (stage_service_time_ns > decision->estimated_critical_path_ns)
        {
            decision->estimated_critical_path_ns = stage_service_time_ns;
        }
        scheduler->spark_inflight_counts[stage_index] += 1u;
    }
    scheduler->admitted_count += 1u;
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
    scheduler->completed_count += 1u;
    return SPARK_STATUS_OK;
}
