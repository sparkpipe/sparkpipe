#include "sparkpipe/spark_glm52_stage_plan.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#define SPARK_GLM52_STAGE_PLAN_UNREACHABLE_COST (UINT64_MAX / 4u)

static SparkStatus SparkGlm52StagePlanReport(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    SparkStatus status,
    const char *message)
{
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        if (message == 0)
        {
            error_buffer[0] = '\0';
        }
        else
        {
            (void)snprintf(error_buffer, error_buffer_bytes, "%s", message);
        }
    }
    return status;
}

static uint32_t SparkGlm52StagePlanMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52StagePlanMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left > right ? left : right;
}

static SparkStatus SparkGlm52StagePlanNormalizeQuantizationMode(
    uint32_t quantization_mode,
    uint32_t *normalized_quantization_mode_out)
{
    if (normalized_quantization_mode_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO)
    {
        *normalized_quantization_mode_out =
            SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
        return SPARK_STATUS_OK;
    }
    if (quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT ||
        quantization_mode == SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
    {
        *normalized_quantization_mode_out = quantization_mode;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkGlm52StagePlanRoutedLayerCountForRange(
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    uint32_t range_end;
    uint32_t routed_begin;
    uint32_t routed_end;

    range_end = first_layer_index + layer_count;
    routed_begin = SparkGlm52StagePlanMaximumU32(
        first_layer_index,
        SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER);
    routed_end = SparkGlm52StagePlanMinimumU32(
        range_end,
        SPARK_GLM52_STAGE_PLAN_LAYER_COUNT);
    if (routed_end <= routed_begin)
    {
        return 0u;
    }
    return routed_end - routed_begin;
}

static uint32_t SparkGlm52StagePlanLayerRangeIsValid(
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    uint32_t range_end;
    uint32_t routed_layer_count;

    if (layer_count == 0u || first_layer_index >= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT)
    {
        return 0u;
    }
    if (layer_count > SPARK_GLM52_STAGE_PLAN_LAYER_COUNT - first_layer_index)
    {
        return 0u;
    }
    range_end = first_layer_index + layer_count;
    if (first_layer_index != 0u &&
        first_layer_index < SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
    {
        return 0u;
    }
    if (range_end < SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER &&
        range_end != SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
    {
        return 0u;
    }
    routed_layer_count = SparkGlm52StagePlanRoutedLayerCountForRange(
        first_layer_index,
        layer_count);
    return routed_layer_count <=
        SPARK_GLM52_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE;
}

static uint64_t SparkGlm52StagePlanMaximumU64(
    uint64_t left,
    uint64_t right)
{
    return left > right ? left : right;
}

static uint64_t SparkGlm52StagePlanSegmentCost(
    const uint64_t *prefix_cost_ns,
    uint32_t first_layer_index,
    uint32_t layer_count)
{
    return prefix_cost_ns[first_layer_index + layer_count] -
        prefix_cost_ns[first_layer_index];
}

static void SparkGlm52StagePlanReset(
    SparkGlm52StagePlan *stage_plan)
{
    memset(stage_plan, 0, sizeof(*stage_plan));
    stage_plan->abi_version = SPARK_GLM52_STAGE_PLAN_ABI_VERSION;
    stage_plan->descriptor_bytes = SPARK_GLM52_STAGE_PLAN_DESCRIPTOR_BYTES;
}

static void SparkGlm52StagePlanAssignStageFlags(
    SparkGlm52StagePlan *stage_plan)
{
    uint32_t stage_index;
    SparkGlm52StagePlanStage *stage;

    for (stage_index = 0u;
         stage_index < stage_plan->stage_count;
         ++stage_index)
    {
        stage = &stage_plan->stages[stage_index];
        stage->flags = SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN;
        if (stage_index + 1u < stage_plan->stage_count)
        {
            stage->flags |= SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN;
        }
        else
        {
            stage->flags |= SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN;
        }
        if (stage->first_layer_index == 0u &&
            stage->layer_count >= SPARK_GLM52_STAGE_PLAN_FIRST_ROUTED_LAYER)
        {
            stage->flags |= SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX;
        }
    }
}

SparkStatus SparkGlm52StagePlanValidate(
    const SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    const SparkGlm52StagePlanStage *stage;
    uint32_t expected_first_layer_index;
    uint32_t final_stage_count;
    uint32_t range_end;
    uint32_t stage_index;

    if (stage_plan == 0)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan is null");
    }
    if (stage_plan->abi_version != SPARK_GLM52_STAGE_PLAN_ABI_VERSION ||
        stage_plan->descriptor_bytes != SPARK_GLM52_STAGE_PLAN_DESCRIPTOR_BYTES)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_ABI_MISMATCH,
            "stage plan ABI mismatch");
    }
    if (stage_plan->stage_count == 0u ||
        stage_plan->stage_count > SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT)
    {
        return SparkGlm52StagePlanReport(
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
        if ((stage->flags & ~SPARK_GLM52_STAGE_PLAN_STAGE_KNOWN_FLAGS) != 0u)
        {
            return SparkGlm52StagePlanReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "stage contains unknown flags");
        }
        if (stage->first_layer_index != expected_first_layer_index)
        {
            return SparkGlm52StagePlanReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "stage layers are not contiguous");
        }
        if (!SparkGlm52StagePlanLayerRangeIsValid(
                stage->first_layer_index,
                stage->layer_count))
        {
            return SparkGlm52StagePlanReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "stage layer range violates GLM-5.2 cut rules");
        }
        if ((stage->flags & SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN) != 0u)
        {
            ++final_stage_count;
            if (stage_index + 1u != stage_plan->stage_count)
            {
                return SparkGlm52StagePlanReport(
                    error_buffer,
                    error_buffer_bytes,
                    SPARK_STATUS_INVALID_ARGUMENT,
                    "final-token stage is not the last stage");
            }
        }
        else if ((stage->flags & SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN) == 0u)
        {
            return SparkGlm52StagePlanReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "non-final stage must emit hidden state");
        }
        range_end = stage->first_layer_index + stage->layer_count;
        expected_first_layer_index = range_end;
    }

    if (expected_first_layer_index != SPARK_GLM52_STAGE_PLAN_LAYER_COUNT)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan does not cover all GLM-5.2 layers");
    }
    if (final_stage_count != 1u)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "stage plan must contain exactly one final-token stage");
    }
    return SparkGlm52StagePlanReport(
        error_buffer,
        error_buffer_bytes,
        SPARK_STATUS_OK,
        "");
}

SparkStatus SparkGlm52StagePlanBuildBalancedWithFinalCost(
    const uint64_t *layer_cost_ns,
    uint64_t final_stage_extra_cost_ns,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t prefix_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT + 1u];
    uint64_t best_cost[SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT + 1u]
        [SPARK_GLM52_STAGE_PLAN_LAYER_COUNT + 1u];
    uint32_t best_split[SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT + 1u]
        [SPARK_GLM52_STAGE_PLAN_LAYER_COUNT + 1u];
    uint32_t current_layer_index;
    uint32_t layer_index;
    uint32_t split_layer_index;
    uint32_t stage_index;
    uint64_t candidate_cost;
    uint64_t segment_cost;

    if (layer_cost_ns == 0 || stage_plan == 0 || stage_count == 0u ||
        stage_count > SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "invalid balanced stage-plan input");
    }

    prefix_cost_ns[0] = 0u;
    for (layer_index = 0u;
         layer_index < SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
         ++layer_index)
    {
        if (layer_cost_ns[layer_index] == 0u)
        {
            return SparkGlm52StagePlanReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "layer cost cannot be zero");
        }
        prefix_cost_ns[layer_index + 1u] =
            prefix_cost_ns[layer_index] + layer_cost_ns[layer_index];
        if (prefix_cost_ns[layer_index + 1u] < prefix_cost_ns[layer_index])
        {
            return SparkGlm52StagePlanReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_CAPACITY_EXCEEDED,
                "layer cost prefix overflow");
        }
    }

    for (stage_index = 0u;
         stage_index <= SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT;
         ++stage_index)
    {
        for (layer_index = 0u;
             layer_index <= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
             ++layer_index)
        {
            best_cost[stage_index][layer_index] =
                SPARK_GLM52_STAGE_PLAN_UNREACHABLE_COST;
            best_split[stage_index][layer_index] = UINT32_MAX;
        }
    }
    best_cost[0u][0u] = 0u;

    for (stage_index = 1u; stage_index <= stage_count; ++stage_index)
    {
        for (layer_index = 1u;
             layer_index <= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
             ++layer_index)
        {
            for (split_layer_index = 0u;
                 split_layer_index < layer_index;
                 ++split_layer_index)
            {
                if (best_cost[stage_index - 1u][split_layer_index] ==
                    SPARK_GLM52_STAGE_PLAN_UNREACHABLE_COST)
                {
                    continue;
                }
                if (!SparkGlm52StagePlanLayerRangeIsValid(
                        split_layer_index,
                        layer_index - split_layer_index))
                {
                    continue;
                }
                segment_cost = SparkGlm52StagePlanSegmentCost(
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
                candidate_cost = SparkGlm52StagePlanMaximumU64(
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

    if (best_cost[stage_count][SPARK_GLM52_STAGE_PLAN_LAYER_COUNT] ==
        SPARK_GLM52_STAGE_PLAN_UNREACHABLE_COST)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_CAPACITY_EXCEEDED,
            "stage count cannot satisfy GLM-5.2 cut rules");
    }

    SparkGlm52StagePlanReset(stage_plan);
    stage_plan->stage_count = stage_count;
    current_layer_index = SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
    for (stage_index = stage_count; stage_index > 0u; --stage_index)
    {
        split_layer_index = best_split[stage_index][current_layer_index];
        if (split_layer_index == UINT32_MAX)
        {
            return SparkGlm52StagePlanReport(
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
    SparkGlm52StagePlanAssignStageFlags(stage_plan);
    return SparkGlm52StagePlanValidate(
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52StagePlanBuildBalanced(
    const uint64_t *layer_cost_ns,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkGlm52StagePlanBuildBalancedWithFinalCost(
        layer_cost_ns,
        0u,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

static void SparkGlm52StagePlanStoreUniformSegmentCost(
    uint64_t segment_cost_ns,
    uint32_t first_layer_index,
    uint32_t layer_count,
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT])
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

static SparkStatus SparkGlm52StagePlanLoadMeasuredB64CostProfile(
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    static const uint32_t first_layer_index[13] = {
        0u, 3u, 10u, 17u, 24u, 30u, 36u, 42u, 48u, 54u, 60u, 66u, 72u
    };
    static const uint32_t layer_count[13] = {
        3u, 7u, 7u, 7u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u
    };
    static const uint64_t stage_cost_ns[13] = {
        109282561u,
        53329600u,
        53304160u,
        53497633u,
        45740192u,
        46787521u,
        43794208u,
        45640128u,
        46685888u,
        44448416u,
        45181631u,
        45586912u,
        45202016u
    };
    uint32_t stage_index;

    for (stage_index = 0u; stage_index < 13u; ++stage_index)
    {
        SparkGlm52StagePlanStoreUniformSegmentCost(
            stage_cost_ns[stage_index],
            first_layer_index[stage_index],
            layer_count[stage_index],
            layer_cost_ns);
    }
    *final_stage_extra_cost_ns_out = 16000000u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52StagePlanLoadMeasuredB32CostProfile(
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
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
        SparkGlm52StagePlanStoreUniformSegmentCost(
            stage_cost_ns[stage_index],
            stage_index * 6u,
            6u,
            layer_cost_ns);
    }
    *final_stage_extra_cost_ns_out = 39342000u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    uint32_t normalized_quantization_mode;
    SparkStatus status;

    if (layer_cost_ns == 0 || final_stage_extra_cost_ns_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52StagePlanNormalizeQuantizationMode(
        quantization_mode,
        &normalized_quantization_mode);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (measured_profile_id != SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    switch (normalized_quantization_mode)
    {
        case SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT:
        case SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT:
            break;
        default:
            return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64)
    {
        return SparkGlm52StagePlanLoadMeasuredB64CostProfile(
            layer_cost_ns,
            final_stage_extra_cost_ns_out);
    }
    if (batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B32 ||
        batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16)
    {
        return SparkGlm52StagePlanLoadMeasuredB32CostProfile(
            layer_cost_ns,
            final_stage_extra_cost_ns_out);
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkGlm52StagePlanLoadMeasuredCostProfile(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out)
{
    return SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
        measured_profile_id,
        batch_bucket,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO,
        layer_cost_ns,
        final_stage_extra_cost_ns_out);
}

SparkStatus SparkGlm52StagePlanBuildMeasuredBalancedForQuantization(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT];
    uint64_t final_stage_extra_cost_ns;
    SparkStatus status;

    status = SparkGlm52StagePlanLoadMeasuredCostProfileForQuantization(
        measured_profile_id,
        batch_bucket,
        quantization_mode,
        layer_cost_ns,
        &final_stage_extra_cost_ns);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52StagePlanReport(
            error_buffer,
            error_buffer_bytes,
            status,
            "measured stage-plan profile is unavailable");
    }
    return SparkGlm52StagePlanBuildBalancedWithFinalCost(
        layer_cost_ns,
        final_stage_extra_cost_ns,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52StagePlanBuildMeasuredBalanced(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkGlm52StagePlanBuildMeasuredBalancedForQuantization(
        measured_profile_id,
        batch_bucket,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52StagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkGlm52StagePlanBuildMeasuredBalancedForQuantization(
        measured_profile_id,
        batch_bucket,
        quantization_mode,
        SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52StagePlanBuildCurrentSparkMeasuredBalanced(
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    return SparkGlm52StagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
        measured_profile_id,
        batch_bucket,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52StagePlanBuildUniform(
    uint32_t stage_count,
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    uint64_t layer_cost_ns[SPARK_GLM52_STAGE_PLAN_LAYER_COUNT];
    uint32_t layer_index;

    for (layer_index = 0u;
         layer_index < SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
         ++layer_index)
    {
        layer_cost_ns[layer_index] = 1u;
    }
    return SparkGlm52StagePlanBuildBalanced(
        layer_cost_ns,
        stage_count,
        stage_plan,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52StagePlanSelectBatchBucket(
    uint32_t active_sequence_count,
    uint32_t *bucket_out)
{
    if (bucket_out == 0 || active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B16)
    {
        *bucket_out = SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
        return SPARK_STATUS_OK;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B32)
    {
        *bucket_out = SPARK_GLM52_STAGE_PLAN_BUCKET_B32;
        return SPARK_STATUS_OK;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B64)
    {
        *bucket_out = SPARK_GLM52_STAGE_PLAN_BUCKET_B64;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}
