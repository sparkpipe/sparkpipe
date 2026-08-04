#include "sparkpipe/spark_stage_plan.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#define SPARK_STAGE_PLAN_UNREACHABLE_COST (UINT64_MAX / 4u)

static SparkStatus SparkStagePlanNormalizeQuantizationMode(
    uint32_t quantization_mode,
    uint32_t *normalized_quantization_mode_out)
{
    if (normalized_quantization_mode_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_AUTO)
    {
        *normalized_quantization_mode_out =
            SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
        return SPARK_STATUS_OK;
    }
    if (quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT ||
        quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
    {
        *normalized_quantization_mode_out = quantization_mode;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkStagePlanLayerRangeIsValid(
    const SparkStagePlanGeometry *geometry,
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    uint32_t range_end;
    uint32_t routed_layer_count;

    if (layer_count == 0u || first_layer_index >= geometry->layer_count)
    {
        return 0u;
    }
    if (layer_count > geometry->layer_count - first_layer_index)
    {
        return 0u;
    }
    range_end = first_layer_index + layer_count;
    // The dense prefix stays whole in stage zero - a rule that only
    // means something when a routed region exists. A fully dense model
    // (first_routed == layer_count) may cut anywhere.
    if (geometry->first_routed_layer < geometry->layer_count)
    {
        if (first_layer_index != 0u &&
            first_layer_index < geometry->first_routed_layer)
        {
            return 0u;
        }
        if (range_end < geometry->first_routed_layer)
        {
            return 0u;
        }
    }
    routed_layer_count = SparkRoutedLayerCountForRange(
first_layer_index,
layer_count,
geometry->first_routed_layer,
geometry->layer_count);
    return routed_layer_count <=
        SPARK_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE;
}

static uint64_t SparkStagePlanMaximumU64(
    uint64_t left,
    uint64_t right)
{
    return left > right ? left : right;
}

static uint64_t SparkStagePlanSegmentCost(
    const uint64_t *prefix_cost_ns,
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    return prefix_cost_ns[first_layer_index + layer_count] -
        prefix_cost_ns[first_layer_index];
}

static void SparkStagePlanReset(
    SparkStagePlan *stage_plan)
{
    memset(stage_plan, 0, sizeof(*stage_plan));
    stage_plan->abi_version = SPARK_STAGE_PLAN_ABI_VERSION;
    stage_plan->descriptor_bytes = SPARK_STAGE_PLAN_DESCRIPTOR_BYTES;
}

static void SparkStagePlanAssignStageFlags(
    const SparkStagePlanGeometry *geometry,
    SparkStagePlan *stage_plan)
{
    uint32_t stage_index;
    SparkStagePlanStage *stage;

    for (stage_index = 0u;
         stage_index < stage_plan->stage_count;
         ++stage_index)
    {
        stage = &stage_plan->stages[stage_index];
        stage->flags = SPARK_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN;
        if (stage_index + 1u < stage_plan->stage_count)
        {
            stage->flags |= SPARK_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN;
        }
        else
        {
            stage->flags |= SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN;
        }
        if (stage->first_layer_index == 0u &&
            stage->layer_count >= geometry->first_routed_layer)
        {
            stage->flags |= SPARK_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX;
        }
    }
}

SparkStatus SparkStagePlanBuildFromLayerCounts(
    const SparkStagePlanGeometry *geometry,
    const uint32_t *layer_counts,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t first_layer_index;
    uint32_t stage_index;

    if (layer_counts == 0 || stage_plan == 0 || stage_count == 0u ||
        stage_count > SPARK_STAGE_PLAN_MAX_STAGE_COUNT)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "invalid table-driven stage-plan input");
    }
    SparkStagePlanReset(stage_plan);
    stage_plan->stage_count = stage_count;
    first_layer_index = 0u;
    for (stage_index = 0u; stage_index < stage_count; ++stage_index)
    {
        if (layer_counts[stage_index] == 0u ||
            layer_counts[stage_index] >
                geometry->layer_count - first_layer_index)
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "table-driven stage layer count is invalid");
        }
        stage_plan->stages[stage_index].first_layer_index = first_layer_index;
        stage_plan->stages[stage_index].layer_count = layer_counts[stage_index];
        first_layer_index += layer_counts[stage_index];
    }
    SparkStagePlanAssignStageFlags(geometry, stage_plan);
    return SparkStagePlanValidate(
        geometry,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanValidate(
    const SparkStagePlanGeometry *geometry,
    const SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    const SparkStagePlanStage *stage;
    uint32_t expected_first_layer_index;
    uint32_t final_stage_count;
    uint32_t range_end;
    uint32_t stage_index;

    if (stage_plan == 0)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan is null");
    }
    if (stage_plan->abi_version != SPARK_STAGE_PLAN_ABI_VERSION ||
        stage_plan->descriptor_bytes != SPARK_STAGE_PLAN_DESCRIPTOR_BYTES)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_ABI_MISMATCH,
            "stage plan ABI mismatch");
    }
    if (stage_plan->stage_count == 0u ||
        stage_plan->stage_count > SPARK_STAGE_PLAN_MAX_STAGE_COUNT)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage count is outside supported range");
    }

    expected_first_layer_index = 0u;
    final_stage_count = 0u;
    for (stage_index = 0u;
         stage_index < stage_plan->stage_count;
         ++stage_index)
    {
        stage = &stage_plan->stages[stage_index];
        if ((stage->flags & ~SPARK_STAGE_PLAN_STAGE_KNOWN_FLAGS) != 0u)
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "stage contains unknown flags");
        }
        if (stage->first_layer_index != expected_first_layer_index)
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "stage layers are not contiguous");
        }
        if (!SparkStagePlanLayerRangeIsValid(
        geometry,
                stage->first_layer_index,
                stage->layer_count))
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "stage layer range violates model geometry cut rules");
        }
        if ((stage->flags & SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
        {
            ++final_stage_count;
            if (stage_index + 1u != stage_plan->stage_count)
            {
                return SparkReportError(
                    error_buffer,
                    error_buffer_bytes,
                    SPARK_STATUS_INVALID_ARGUMENT,
                    "final-token stage is not the last stage");
            }
        }
        else if ((stage->flags & SPARK_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN) == 0u)
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "non-final stage must emit hidden state");
        }
        range_end = stage->first_layer_index + stage->layer_count;
        expected_first_layer_index = range_end;
    }

    if (expected_first_layer_index != geometry->layer_count)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan does not cover all model layers");
    }
    if (final_stage_count != 1u)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan must contain exactly one final-token stage");
    }
    return SparkReportError(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}

SparkStatus SparkStagePlanBuildBalancedWithFinalCost(
    const SparkStagePlanGeometry *geometry,
    const uint64_t *layer_cost_ns,
    uint64_t final_stage_extra_cost_ns,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t prefix_cost_ns[geometry->layer_count + 1u];
    uint64_t best_cost[SPARK_STAGE_PLAN_MAX_STAGE_COUNT + 1u]
        [geometry->layer_count + 1u];
    uint32_t best_split[SPARK_STAGE_PLAN_MAX_STAGE_COUNT + 1u]
        [geometry->layer_count + 1u];
    uint32_t current_layer_index;
    uint32_t layer_index;
    uint32_t split_layer_index;
    uint32_t stage_index;
    uint64_t candidate_cost;
    uint64_t segment_cost;

    if (layer_cost_ns == 0 || stage_plan == 0 || stage_count == 0u ||
        stage_count > SPARK_STAGE_PLAN_MAX_STAGE_COUNT)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "invalid balanced stage-plan input");
    }

    prefix_cost_ns[0] = 0u;
    for (layer_index = 0u;
         layer_index < geometry->layer_count;
         ++layer_index)
    {
        if (layer_cost_ns[layer_index] == 0u)
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "layer cost cannot be zero");
        }
        prefix_cost_ns[layer_index + 1u] =
            prefix_cost_ns[layer_index] + layer_cost_ns[layer_index];
        if (prefix_cost_ns[layer_index + 1u] < prefix_cost_ns[layer_index])
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_CAPACITY_EXCEEDED,
                "layer cost prefix overflow");
        }
    }

    for (stage_index = 0u;
         stage_index <= SPARK_STAGE_PLAN_MAX_STAGE_COUNT;
         ++stage_index)
    {
        for (layer_index = 0u;
             layer_index <= geometry->layer_count;
             ++layer_index)
        {
            best_cost[stage_index][layer_index] =
                SPARK_STAGE_PLAN_UNREACHABLE_COST;
            best_split[stage_index][layer_index] = UINT32_MAX;
        }
    }
    best_cost[0u][0u] = 0u;

    for (stage_index = 1u; stage_index <= stage_count; ++stage_index)
    {
        for (layer_index = 1u;
             layer_index <= geometry->layer_count;
             ++layer_index)
        {
            for (split_layer_index = 0u;
                 split_layer_index < layer_index;
                 ++split_layer_index)
            {
                if (best_cost[stage_index - 1u][split_layer_index] ==
                    SPARK_STAGE_PLAN_UNREACHABLE_COST)
                {
                    continue;
                }
                if (!SparkStagePlanLayerRangeIsValid(
        geometry,
                        split_layer_index,
                        layer_index - split_layer_index))
                {
                    continue;
                }
                segment_cost = SparkStagePlanSegmentCost(
                    prefix_cost_ns,
                    split_layer_index,
                    layer_index - split_layer_index);
                if (stage_index == stage_count)
                {
                    if (segment_cost > UINT64_MAX - final_stage_extra_cost_ns)
                    {
                        continue;
                    }
                    segment_cost += final_stage_extra_cost_ns;
                }
                candidate_cost = SparkStagePlanMaximumU64(
                    best_cost[stage_index - 1u][split_layer_index],
                    segment_cost);
                if (candidate_cost < best_cost[stage_index][layer_index])
                {
                    best_cost[stage_index][layer_index] = candidate_cost;
                    best_split[stage_index][layer_index] = split_layer_index;
                }
            }
        }
    }

    if (best_cost[stage_count][geometry->layer_count] ==
        SPARK_STAGE_PLAN_UNREACHABLE_COST)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_CAPACITY_EXCEEDED,
            "stage count cannot satisfy the cut rules");
    }

    SparkStagePlanReset(stage_plan);
    stage_plan->stage_count = stage_count;
    current_layer_index = geometry->layer_count;
    for (stage_index = stage_count; stage_index > 0u; --stage_index)
    {
        split_layer_index = best_split[stage_index][current_layer_index];
        if (split_layer_index == UINT32_MAX)
        {
            return SparkReportError(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INTERNAL_ERROR,
                "stage-plan backtrack failed");
        }
        stage_plan->stages[stage_index - 1u].first_layer_index =
            split_layer_index;
        stage_plan->stages[stage_index - 1u].layer_count =
            current_layer_index - split_layer_index;
        current_layer_index = split_layer_index;
    }
    SparkStagePlanAssignStageFlags(geometry, stage_plan);
    return SparkStagePlanValidate(
        geometry,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanBuildBalanced(
    const SparkStagePlanGeometry *geometry,
    const uint64_t *layer_cost_ns,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkStagePlanBuildBalancedWithFinalCost(
        geometry,
        layer_cost_ns,
        0u,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

static void SparkStagePlanStoreUniformSegmentCost(
    uint64_t segment_cost_ns,
    uint32_t first_layer_index,
    uint32_t layer_count,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT])
{
    uint64_t base_layer_cost_ns;
    uint64_t remainder_ns;
    uint32_t layer_offset;

    base_layer_cost_ns = segment_cost_ns / (uint64_t)layer_count;
    remainder_ns = segment_cost_ns % (uint64_t)layer_count;
    for (layer_offset = 0u; layer_offset < layer_count; ++layer_offset)
    {
        layer_cost_ns[first_layer_index + layer_offset] =
            base_layer_cost_ns + ((uint64_t)layer_offset < remainder_ns ? 1u : 0u);
    }
}

static SparkStatus SparkStagePlanLoadMeasuredB64CostProfile(
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    static const uint32_t first_layer_index[13] = {
        0u, 6u, 12u, 18u, 24u, 30u, 36u, 42u, 48u, 54u, 60u, 66u, 72u
    };
    static const uint32_t layer_count[13] = {
        6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u
    };
    static const uint64_t stage_cost_ns[13] = {
        50660288u,
        45685889u,
        45232480u,
        45782816u,
        44223711u,
        45062784u,
        45439968u,
        45055391u,
        45370304u,
        46190688u,
        44552225u,
        45824320u,
        46449792u
    };
    uint32_t stage_index;

    for (stage_index = 0u; stage_index < 13u; ++stage_index)
    {
        SparkStagePlanStoreUniformSegmentCost(
            stage_cost_ns[stage_index],
            first_layer_index[stage_index],
            layer_count[stage_index],
            layer_cost_ns);
    }
    *final_stage_extra_cost_ns_out = 0u;
    return SPARK_STATUS_OK;
}

static void SparkStagePlanScaleCostProfile(
    const SparkStagePlanGeometry *geometry,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns,
    uint32_t numerator,
    uint32_t denominator)
{
    uint32_t layer_index;

    for (layer_index = 0u;
         layer_index < geometry->layer_count;
         ++layer_index)
    {
        layer_cost_ns[layer_index] =
            ((layer_cost_ns[layer_index] * (uint64_t)numerator) +
             (uint64_t)denominator - 1u) / (uint64_t)denominator;
    }
    *final_stage_extra_cost_ns =
        ((*final_stage_extra_cost_ns * (uint64_t)numerator) +
         (uint64_t)denominator - 1u) / (uint64_t)denominator;
}

static SparkStatus SparkStagePlanLoadMeasuredB128CostProfile(
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    static const uint64_t stage_cost_ns[13] = {
        177756500u,
        188399000u,
        193653016u,
        197361562u,
        195107453u,
        195583047u,
        192805688u,
        194702031u,
        195701188u,
        193026547u,
        196165188u,
        190992359u,
        194542891u
    };
    uint32_t stage_index;

    for (stage_index = 0u; stage_index < 13u; ++stage_index)
    {
        SparkStagePlanStoreUniformSegmentCost(
            stage_cost_ns[stage_index],
            stage_index * 6u,
            6u,
            layer_cost_ns);
    }
    *final_stage_extra_cost_ns_out = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkStagePlanLoadEstimatedLargeBatchCostProfile(
    const SparkStagePlanGeometry *geometry,
    uint32_t batch_bucket,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    SparkStatus status;

    if (batch_bucket < SPARK_STAGE_PLAN_BUCKET_B128 ||
        SparkStagePlanBatchBucketIsSupported(batch_bucket) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkStagePlanLoadMeasuredB128CostProfile(
        layer_cost_ns,
        final_stage_extra_cost_ns_out);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkStagePlanScaleCostProfile(
        geometry,
        layer_cost_ns,
        final_stage_extra_cost_ns_out,
        batch_bucket,
        SPARK_STAGE_PLAN_BUCKET_B128);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkStagePlanLoadMeasuredB32CostProfile(
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    static const uint64_t stage_cost_ns[13] = {
        39691000u,
        35142000u,
        33496000u,
        36755000u,
        38837000u,
        41760000u,
        46160000u,
        56283000u,
        60862000u,
        69126000u,
        72429000u,
        81610000u,
        73314000u
    };
    uint32_t stage_index;

    for (stage_index = 0u; stage_index < 13u; ++stage_index)
    {
        SparkStagePlanStoreUniformSegmentCost(
            stage_cost_ns[stage_index],
            stage_index * 6u,
            6u,
            layer_cost_ns);
    }
    *final_stage_extra_cost_ns_out = 39342000u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkStagePlanBuildMeasuredB64RingExact(
    const SparkStagePlanGeometry *geometry,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    static const uint32_t first_layer_index[13] = {
        0u, 6u, 12u, 18u, 24u, 30u, 36u, 42u, 48u, 54u, 60u, 66u, 72u
    };
    uint32_t stage_index;

    if (stage_plan == 0)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan is null");
    }
    SparkStagePlanReset(stage_plan);
    stage_plan->stage_count = SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT;
    for (stage_index = 0u;
         stage_index < SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT;
         ++stage_index)
    {
        stage_plan->stages[stage_index].first_layer_index =
            first_layer_index[stage_index];
        stage_plan->stages[stage_index].layer_count = 6u;
    }
    SparkStagePlanAssignStageFlags(geometry, stage_plan);
    return SparkStagePlanValidate(
        geometry,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanLoadUniformCostProfile(
    const SparkStagePlanGeometry *geometry,
    uint64_t layer_cost_ns_estimate,
    uint64_t final_stage_extra_cost_ns_estimate,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    uint32_t layer_index;

    if (geometry == 0 || layer_cost_ns == 0 ||
        final_stage_extra_cost_ns_out == 0 ||
        layer_cost_ns_estimate == 0u ||
        geometry->layer_count == 0u ||
        geometry->layer_count > SPARK_STAGE_PLAN_MAX_LAYER_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (layer_index = 0u;
         layer_index < geometry->layer_count;
         ++layer_index)
    {
        layer_cost_ns[layer_index] = layer_cost_ns_estimate;
    }
    *final_stage_extra_cost_ns_out = final_stage_extra_cost_ns_estimate;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStagePlanLoadMeasuredCostProfileForQuantization(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    uint32_t normalized_quantization_mode;
    SparkStatus status;

    if (layer_cost_ns == 0 || final_stage_extra_cost_ns_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkStagePlanNormalizeQuantizationMode(
        quantization_mode,
        &normalized_quantization_mode);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (measured_profile_id != SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    switch (normalized_quantization_mode)
    {
        case SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT:
        case SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT:
            break;
        default:
            return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (batch_bucket == SPARK_STAGE_PLAN_BUCKET_B64)
    {
        return SparkStagePlanLoadMeasuredB64CostProfile(
            layer_cost_ns,
            final_stage_extra_cost_ns_out);
    }
    if (batch_bucket == SPARK_STAGE_PLAN_BUCKET_B128)
    {
        return SparkStagePlanLoadMeasuredB128CostProfile(
            layer_cost_ns,
            final_stage_extra_cost_ns_out);
    }
    if (batch_bucket > SPARK_STAGE_PLAN_BUCKET_B128 &&
        SparkStagePlanBatchBucketIsSupported(batch_bucket) != 0u)
    {
        return SparkStagePlanLoadEstimatedLargeBatchCostProfile(
        geometry,
            batch_bucket,
            layer_cost_ns,
            final_stage_extra_cost_ns_out);
    }
    if (batch_bucket == SPARK_STAGE_PLAN_BUCKET_B32 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B16)
    {
        return SparkStagePlanLoadMeasuredB32CostProfile(
            layer_cost_ns,
            final_stage_extra_cost_ns_out);
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkStagePlanLoadMeasuredCostProfile(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    return SparkStagePlanLoadMeasuredCostProfileForQuantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        SPARK_STAGE_PLAN_QUANTIZATION_AUTO,
        layer_cost_ns,
        final_stage_extra_cost_ns_out);
}

SparkStatus SparkStagePlanBuildMeasuredBalancedForQuantization(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    SparkStatus status;

    status = SparkStagePlanLoadMeasuredCostProfileForQuantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        quantization_mode,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            status,
            "measured stage-plan profile is unavailable");
    }
    return SparkStagePlanBuildBalancedWithFinalCost(
        geometry,
        layer_cost_ns,
        final_stage_extra_cost_ns,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanBuildMeasuredBalanced(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkStagePlanBuildMeasuredBalancedForQuantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        SPARK_STAGE_PLAN_QUANTIZATION_AUTO,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint32_t normalized_quantization_mode;
    SparkStatus status;

    status = SparkStagePlanNormalizeQuantizationMode(
        quantization_mode,
        &normalized_quantization_mode);
    if (status != SPARK_STATUS_OK)
    {
        return SparkReportError(
            error_buffer,
            error_buffer_bytes,
            status,
            "invalid measured stage-plan quantization mode");
    }
    if (measured_profile_id == SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701 &&
        batch_bucket >= SPARK_STAGE_PLAN_BUCKET_B64 &&
        SparkStagePlanBatchBucketIsSupported(batch_bucket) != 0u &&
        (normalized_quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT ||
         normalized_quantization_mode == SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT))
    {
        return SparkStagePlanBuildMeasuredB64RingExact(
        geometry,
            stage_plan,
            error_buffer,
            error_buffer_bytes);
    }
    return SparkStagePlanBuildMeasuredBalancedForQuantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        normalized_quantization_mode,
        SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanBuildCurrentSparkMeasuredBalanced(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        SPARK_STAGE_PLAN_QUANTIZATION_AUTO,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanBuildUniform(
    const SparkStagePlanGeometry *geometry,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT];
    uint32_t layer_index;

    for (layer_index = 0u;
         layer_index < geometry->layer_count;
         ++layer_index)
    {
        layer_cost_ns[layer_index] = 1u;
    }
    return SparkStagePlanBuildBalanced(
        geometry,
        layer_cost_ns,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkStagePlanSelectBatchBucket(
    uint32_t active_sequence_count,
    uint32_t *bucket_out)
{
    if (bucket_out == 0 || active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *bucket_out = SparkStagePlanSelectBatchBucketValue(
        active_sequence_count);
    if (*bucket_out == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkStagePlanExecutionChunkShape(
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t execution_row_capacity,
    uint32_t *maximum_sequences_per_chunk_out,
    uint32_t *chunk_count_out)
{
    uint32_t maximum_sequences_per_chunk;
    uint32_t chunk_count;

    if (logical_sequence_count == 0u || rows_per_sequence == 0u ||
        execution_row_capacity == 0u ||
        maximum_sequences_per_chunk_out == 0 || chunk_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (execution_row_capacity > SPARK_STAGE_PLAN_MAX_BATCH_BUCKET)
    {
        execution_row_capacity = SPARK_STAGE_PLAN_MAX_BATCH_BUCKET;
    }
    maximum_sequences_per_chunk = execution_row_capacity / rows_per_sequence;
    if (maximum_sequences_per_chunk == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (maximum_sequences_per_chunk > logical_sequence_count)
    {
        maximum_sequences_per_chunk = logical_sequence_count;
    }
    chunk_count = logical_sequence_count / maximum_sequences_per_chunk;
    if (logical_sequence_count % maximum_sequences_per_chunk != 0u)
    {
        chunk_count += 1u;
    }
    *maximum_sequences_per_chunk_out = maximum_sequences_per_chunk;
    *chunk_count_out = chunk_count;
    return SPARK_STATUS_OK;
}
