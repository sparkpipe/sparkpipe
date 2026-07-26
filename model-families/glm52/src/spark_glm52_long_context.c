#include "sparkpipe/spark_glm52_long_context.h"

#include <string.h>

static uint32_t SparkGlm52LongContextMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52LongContextCeilDivideU32(
    uint32_t numerator,
    uint32_t denominator)
{
    if (denominator == 0u)
    {
        return 0u;
    }
    return (numerator + denominator - 1u) / denominator;
}

static uint32_t SparkGlm52LongContextNormalizeFlags(
    uint32_t policy_flags)
{
    if (policy_flags == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_POLICY_DEFAULT_FLAGS;
    }
    return policy_flags;
}

static uint32_t SparkGlm52LongContextNormalizeSelectedTokenCapacity(
    uint32_t selected_token_capacity)
{
    if (selected_token_capacity == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY;
    }
    return selected_token_capacity;
}

static uint32_t SparkGlm52LongContextNormalizeBlockTokenCount(
    uint32_t block_token_count)
{
    if (block_token_count == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_BLOCK_TOKEN_COUNT;
    }
    return block_token_count;
}

static uint32_t SparkGlm52LongContextNormalizeMaxContextTokens(
    uint32_t max_context_tokens)
{
    if (max_context_tokens == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_MAX_CONTEXT_TOKENS;
    }
    return max_context_tokens;
}

static uint32_t SparkGlm52LongContextNormalizeRecentTokenCount(
    uint32_t recent_token_count)
{
    if (recent_token_count == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_RECENT_TOKEN_COUNT;
    }
    return recent_token_count;
}

static uint32_t SparkGlm52LongContextNormalizeSinkTokenCount(
    uint32_t sink_token_count)
{
    if (sink_token_count == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_SINK_TOKEN_COUNT;
    }
    return sink_token_count;
}

static uint32_t SparkGlm52LongContextNormalizeStrideSampleTokenCount(
    uint32_t stride_sample_token_count)
{
    if (stride_sample_token_count == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT;
    }
    return stride_sample_token_count;
}

static uint32_t SparkGlm52LongContextNormalizeMaximumDecodeScanTokenCount(
    uint32_t maximum_decode_scan_token_count,
    uint32_t selected_token_capacity)
{
    if (maximum_decode_scan_token_count == 0u)
    {
        return selected_token_capacity;
    }
    return maximum_decode_scan_token_count;
}

static uint32_t SparkGlm52LongContextNormalizePrefillChunkTokenCount(
    uint32_t prefill_chunk_token_count)
{
    if (prefill_chunk_token_count == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY;
    }
    return prefill_chunk_token_count;
}

static uint32_t SparkGlm52LongContextNormalizeLongContextThreshold(
    uint32_t long_context_threshold_token_count)
{
    if (long_context_threshold_token_count == 0u)
    {
        return SPARK_GLM52_LONG_CONTEXT_DEFAULT_LONG_CONTEXT_THRESHOLD;
    }
    return long_context_threshold_token_count;
}

static void SparkGlm52LongContextNormalizePolicyCopy(
    const SparkGlm52LongContextPolicy *input_policy,
    SparkGlm52LongContextPolicy *normalized_policy)
{
    *normalized_policy = *input_policy;
    normalized_policy->policy_flags = SparkGlm52LongContextNormalizeFlags(
        normalized_policy->policy_flags);
    if (normalized_policy->policy_mode == 0u)
    {
        normalized_policy->policy_mode =
            SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_BOUNDED_WINDOW;
    }
    normalized_policy->max_context_tokens =
        SparkGlm52LongContextNormalizeMaxContextTokens(
            normalized_policy->max_context_tokens);
    normalized_policy->selected_token_capacity =
        SparkGlm52LongContextNormalizeSelectedTokenCapacity(
            normalized_policy->selected_token_capacity);
    normalized_policy->block_token_count =
        SparkGlm52LongContextNormalizeBlockTokenCount(
            normalized_policy->block_token_count);
    normalized_policy->recent_token_count =
        SparkGlm52LongContextNormalizeRecentTokenCount(
            normalized_policy->recent_token_count);
    normalized_policy->sink_token_count =
        SparkGlm52LongContextNormalizeSinkTokenCount(
            normalized_policy->sink_token_count);
    normalized_policy->stride_sample_token_count =
        SparkGlm52LongContextNormalizeStrideSampleTokenCount(
            normalized_policy->stride_sample_token_count);
    normalized_policy->maximum_decode_scan_token_count =
        SparkGlm52LongContextNormalizeMaximumDecodeScanTokenCount(
            normalized_policy->maximum_decode_scan_token_count,
            normalized_policy->selected_token_capacity);
    normalized_policy->prefill_chunk_token_count =
        SparkGlm52LongContextNormalizePrefillChunkTokenCount(
            normalized_policy->prefill_chunk_token_count);
    normalized_policy->long_context_threshold_token_count =
        SparkGlm52LongContextNormalizeLongContextThreshold(
            normalized_policy->long_context_threshold_token_count);
}

void SparkGlm52LongContextInitializeDefaultPolicy(
    SparkGlm52LongContextPolicy *policy)
{
    if (policy == 0)
    {
        return;
    }
    memset(policy, 0, sizeof(*policy));
    policy->abi_version = SPARK_GLM52_LONG_CONTEXT_ABI_VERSION;
    policy->descriptor_bytes = SPARK_GLM52_LONG_CONTEXT_POLICY_DESCRIPTOR_BYTES;
    policy->policy_mode = SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_BOUNDED_WINDOW;
    policy->policy_flags = SPARK_GLM52_LONG_CONTEXT_POLICY_DEFAULT_FLAGS;
    policy->max_context_tokens = SPARK_GLM52_LONG_CONTEXT_DEFAULT_MAX_CONTEXT_TOKENS;
    policy->selected_token_capacity =
        SPARK_GLM52_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY;
    policy->block_token_count = SPARK_GLM52_LONG_CONTEXT_DEFAULT_BLOCK_TOKEN_COUNT;
    policy->recent_token_count = SPARK_GLM52_LONG_CONTEXT_DEFAULT_RECENT_TOKEN_COUNT;
    policy->sink_token_count = SPARK_GLM52_LONG_CONTEXT_DEFAULT_SINK_TOKEN_COUNT;
    policy->stride_sample_token_count =
        SPARK_GLM52_LONG_CONTEXT_DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT;
    policy->maximum_decode_scan_token_count =
        SPARK_GLM52_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY;
    policy->prefill_chunk_token_count =
        SPARK_GLM52_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY;
    policy->long_context_threshold_token_count =
        SPARK_GLM52_LONG_CONTEXT_DEFAULT_LONG_CONTEXT_THRESHOLD;
}

SparkStatus SparkGlm52LongContextValidatePolicy(
    const SparkGlm52LongContextPolicy *policy)
{
    SparkGlm52LongContextPolicy normalized_policy;

    if (policy == 0 ||
        policy->abi_version != SPARK_GLM52_LONG_CONTEXT_ABI_VERSION ||
        policy->descriptor_bytes != SPARK_GLM52_LONG_CONTEXT_POLICY_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkGlm52LongContextNormalizePolicyCopy(policy, &normalized_policy);
    if ((normalized_policy.policy_flags &
            ~SPARK_GLM52_LONG_CONTEXT_POLICY_KNOWN_FLAGS) != 0u ||
        normalized_policy.max_context_tokens == 0u ||
        normalized_policy.selected_token_capacity == 0u ||
        normalized_policy.selected_token_capacity >
            SPARK_GLM52_LONG_CONTEXT_MAX_SELECTED_TOKEN_CAPACITY ||
        normalized_policy.block_token_count == 0u ||
        normalized_policy.maximum_decode_scan_token_count == 0u ||
        normalized_policy.maximum_decode_scan_token_count >
            normalized_policy.selected_token_capacity ||
        normalized_policy.prefill_chunk_token_count == 0u ||
        (normalized_policy.policy_mode !=
            SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_BOUNDED_WINDOW &&
         normalized_policy.policy_mode !=
            SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_REQUIRE_BOUNDED_DECODE) != 0u &&
        normalized_policy.policy_mode ==
            SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN &&
        (normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_INCLUDE_SINK_TOKENS) == 0u &&
        (normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS) == 0u &&
        (normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_INCLUDE_RECENT_TOKENS) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52LongContextInitializeDecodePlan(
    SparkGlm52LongContextDecodePlan *decode_plan,
    const SparkGlm52LongContextPolicy *policy,
    uint32_t context_token_count,
    uint32_t selected_token_capacity)
{
    memset(decode_plan, 0, sizeof(*decode_plan));
    decode_plan->abi_version = SPARK_GLM52_LONG_CONTEXT_ABI_VERSION;
    decode_plan->descriptor_bytes =
        SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_DESCRIPTOR_BYTES;
    decode_plan->context_token_count = context_token_count;
    decode_plan->selected_token_capacity = selected_token_capacity;
    decode_plan->kv_block_token_count = policy->block_token_count;
    decode_plan->kv_block_count_for_context = SparkGlm52LongContextCeilDivideU32(
        context_token_count,
        policy->block_token_count);
    decode_plan->maximum_decode_scan_token_count =
        policy->maximum_decode_scan_token_count;
    if (context_token_count >= policy->long_context_threshold_token_count)
    {
        decode_plan->flags |= SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_LONG_CONTEXT;
    }
}

static uint32_t SparkGlm52LongContextAppendUniqueToken(
    uint32_t *selected_token_indices,
    uint32_t selected_token_capacity,
    uint32_t *selected_token_count,
    uint32_t token_index)
{
    uint32_t existing_index;

    if (selected_token_indices == 0 || selected_token_count == 0 ||
        *selected_token_count >= selected_token_capacity ||
        token_index == SPARK_GLM52_LONG_CONTEXT_INVALID_TOKEN_ID)
    {
        return 0u;
    }
    for (existing_index = 0u;
         existing_index < *selected_token_count;
         ++existing_index)
    {
        if (selected_token_indices[existing_index] == token_index)
        {
            return 0u;
        }
    }
    selected_token_indices[*selected_token_count] = token_index;
    *selected_token_count += 1u;
    return 1u;
}

static uint32_t SparkGlm52LongContextCountUniqueBlocks(
    const uint32_t *selected_token_indices,
    uint32_t selected_token_count,
    uint32_t block_token_count)
{
    uint32_t block_count;
    uint32_t selected_index;

    block_count = 0u;
    for (selected_index = 0u;
         selected_index < selected_token_count;
         ++selected_index)
    {
        uint32_t token_index;
        uint32_t block_index;
        uint32_t prior_index;
        uint32_t already_counted;

        token_index = selected_token_indices[selected_index];
        if (token_index == SPARK_GLM52_LONG_CONTEXT_INVALID_TOKEN_ID)
        {
            continue;
        }
        block_index = token_index / block_token_count;
        already_counted = 0u;
        for (prior_index = 0u;
             prior_index < selected_index;
             ++prior_index)
        {
            if (selected_token_indices[prior_index] !=
                    SPARK_GLM52_LONG_CONTEXT_INVALID_TOKEN_ID &&
                selected_token_indices[prior_index] / block_token_count == block_index)
            {
                already_counted = 1u;
                break;
            }
        }
        if (already_counted == 0u)
        {
            block_count += 1u;
        }
    }
    return block_count;
}

static void SparkGlm52LongContextPadSelection(
    uint32_t *selected_token_indices,
    uint32_t selected_token_count,
    uint32_t selected_token_capacity)
{
    while (selected_token_count < selected_token_capacity)
    {
        selected_token_indices[selected_token_count] =
            SPARK_GLM52_LONG_CONTEXT_INVALID_TOKEN_ID;
        selected_token_count += 1u;
    }
}

static SparkStatus SparkGlm52LongContextBuildFullSelection(
    const SparkGlm52LongContextPolicy *policy,
    uint32_t context_token_count,
    uint32_t *selected_token_indices,
    uint32_t selected_token_capacity,
    SparkGlm52LongContextDecodePlan *decode_plan)
{
    uint32_t token_index;

    if (context_token_count > selected_token_capacity ||
        context_token_count > policy->maximum_decode_scan_token_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (token_index = 0u; token_index < context_token_count; ++token_index)
    {
        selected_token_indices[token_index] = token_index;
    }
    SparkGlm52LongContextPadSelection(
        selected_token_indices,
        context_token_count,
        selected_token_capacity);
    decode_plan->selected_token_count = context_token_count;
    decode_plan->padded_token_count = selected_token_capacity - context_token_count;
    decode_plan->selected_block_count = SparkGlm52LongContextCeilDivideU32(
        context_token_count,
        policy->block_token_count);
    decode_plan->estimated_attention_token_reads = context_token_count;
    if (context_token_count < selected_token_capacity)
    {
        decode_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_SELECTION_PADDED;
    }
    if (policy->policy_mode ==
        SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN)
    {
        decode_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_FULL_CONTEXT_SCAN;
    }
    else
    {
        decode_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_BOUNDED_SELECTION;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52LongContextBuildBoundedSelection(
    const SparkGlm52LongContextPolicy *policy,
    uint32_t context_token_count,
    uint32_t *selected_token_indices,
    uint32_t selected_token_capacity,
    SparkGlm52LongContextDecodePlan *decode_plan)
{
    uint32_t selected_token_count;
    uint32_t sink_count;
    uint32_t recent_count;
    uint32_t recent_begin;
    uint32_t middle_begin;
    uint32_t middle_end;
    uint32_t middle_token_count;
    uint32_t stride_count;
    uint32_t token_index;

    selected_token_count = 0u;
    sink_count = 0u;
    recent_count = 0u;
    stride_count = 0u;
    if ((policy->policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_INCLUDE_SINK_TOKENS) != 0u)
    {
        sink_count = SparkGlm52LongContextMinimumU32(
            policy->sink_token_count,
            context_token_count);
        sink_count = SparkGlm52LongContextMinimumU32(
            sink_count,
            selected_token_capacity);
        for (token_index = 0u;
             token_index < sink_count;
             ++token_index)
        {
            SparkGlm52LongContextAppendUniqueToken(
                selected_token_indices,
                selected_token_capacity,
                &selected_token_count,
                token_index);
        }
    }

    if ((policy->policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_INCLUDE_RECENT_TOKENS) != 0u &&
        selected_token_count < selected_token_capacity)
    {
        recent_count = SparkGlm52LongContextMinimumU32(
            policy->recent_token_count,
            selected_token_capacity - selected_token_count);
        recent_count = SparkGlm52LongContextMinimumU32(
            recent_count,
            context_token_count);
    }
    recent_begin = context_token_count > recent_count
        ? context_token_count - recent_count
        : 0u;
    if (recent_begin < sink_count)
    {
        recent_begin = sink_count;
    }

    middle_begin = sink_count;
    middle_end = recent_begin;
    middle_token_count = middle_end > middle_begin
        ? middle_end - middle_begin
        : 0u;
    if ((policy->policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS) != 0u &&
        middle_token_count != 0u &&
        selected_token_count < selected_token_capacity)
    {
        stride_count = SparkGlm52LongContextMinimumU32(
            policy->stride_sample_token_count,
            selected_token_capacity - selected_token_count);
        stride_count = SparkGlm52LongContextMinimumU32(
            stride_count,
            middle_token_count);
        for (token_index = 0u;
             token_index < stride_count;
             ++token_index)
        {
            uint64_t numerator;
            uint32_t selected_middle_token;

            numerator = ((uint64_t)token_index + 1u) *
                (uint64_t)middle_token_count;
            selected_middle_token = middle_begin +
                (uint32_t)(numerator / ((uint64_t)stride_count + 1u));
            if (selected_middle_token >= middle_end)
            {
                selected_middle_token = middle_end - 1u;
            }
            SparkGlm52LongContextAppendUniqueToken(
                selected_token_indices,
                selected_token_capacity,
                &selected_token_count,
                selected_middle_token);
        }
    }

    for (token_index = recent_begin;
         token_index < context_token_count && selected_token_count < selected_token_capacity;
         ++token_index)
    {
        SparkGlm52LongContextAppendUniqueToken(
            selected_token_indices,
            selected_token_capacity,
            &selected_token_count,
            token_index);
    }

    SparkGlm52LongContextPadSelection(
        selected_token_indices,
        selected_token_count,
        selected_token_capacity);
    decode_plan->flags |= SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_BOUNDED_SELECTION;
    decode_plan->selected_token_count = selected_token_count;
    decode_plan->padded_token_count = selected_token_capacity - selected_token_count;
    decode_plan->selected_block_count = SparkGlm52LongContextCountUniqueBlocks(
        selected_token_indices,
        selected_token_count,
        policy->block_token_count);
    decode_plan->first_recent_token_index = recent_begin;
    decode_plan->sink_token_count = sink_count;
    decode_plan->stride_sample_token_count = stride_count;
    decode_plan->recent_token_count = context_token_count - recent_begin;
    decode_plan->estimated_attention_token_reads = selected_token_count;
    decode_plan->avoided_full_scan_token_reads =
        context_token_count > selected_token_count
        ? (uint64_t)(context_token_count - selected_token_count)
        : 0u;
    if (selected_token_count < selected_token_capacity)
    {
        decode_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_SELECTION_PADDED;
    }
    if (context_token_count > selected_token_count)
    {
        decode_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_DECODE_PLAN_FLAG_CONTEXT_TRUNCATED;
    }
}

SparkStatus SparkGlm52LongContextBuildPrefillPlan(
    const SparkGlm52LongContextPolicy *policy,
    uint32_t prompt_token_count,
    uint32_t max_prefill_tokens_per_step,
    SparkGlm52LongContextPrefillPlan *prefill_plan)
{
    SparkGlm52LongContextPolicy normalized_policy;
    SparkStatus status;

    if (policy == 0 || prefill_plan == 0 || prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52LongContextValidatePolicy(policy);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkGlm52LongContextNormalizePolicyCopy(policy, &normalized_policy);
    if (prompt_token_count > normalized_policy.max_context_tokens &&
        (normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW) != 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (max_prefill_tokens_per_step == 0u)
    {
        max_prefill_tokens_per_step = normalized_policy.prefill_chunk_token_count;
    }
    memset(prefill_plan, 0, sizeof(*prefill_plan));
    prefill_plan->abi_version = SPARK_GLM52_LONG_CONTEXT_ABI_VERSION;
    prefill_plan->descriptor_bytes =
        SPARK_GLM52_LONG_CONTEXT_PREFILL_PLAN_DESCRIPTOR_BYTES;
    prefill_plan->prompt_token_count = prompt_token_count;
    prefill_plan->max_prefill_tokens_per_step = max_prefill_tokens_per_step;
    prefill_plan->prefill_chunk_count = SparkGlm52LongContextCeilDivideU32(
        prompt_token_count,
        max_prefill_tokens_per_step);
    prefill_plan->final_chunk_token_count = prompt_token_count %
        max_prefill_tokens_per_step;
    if (prefill_plan->final_chunk_token_count == 0u)
    {
        prefill_plan->final_chunk_token_count = max_prefill_tokens_per_step;
    }
    prefill_plan->kv_block_token_count = normalized_policy.block_token_count;
    prefill_plan->kv_block_count_for_prompt = SparkGlm52LongContextCeilDivideU32(
        prompt_token_count,
        normalized_policy.block_token_count);
    prefill_plan->total_prompt_token_visits = (uint64_t)prompt_token_count;
    if (prefill_plan->prefill_chunk_count > 1u)
    {
        prefill_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_PREFILL_PLAN_FLAG_CHUNKED_PREFILL;
    }
    if (prompt_token_count >= normalized_policy.long_context_threshold_token_count)
    {
        prefill_plan->flags |=
            SPARK_GLM52_LONG_CONTEXT_PREFILL_PLAN_FLAG_LONG_CONTEXT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52LongContextBuildDecodeSelection(
    const SparkGlm52LongContextPolicy *policy,
    uint32_t context_token_count,
    uint32_t *selected_token_indices,
    uint32_t selected_token_capacity,
    SparkGlm52LongContextDecodePlan *decode_plan)
{
    SparkGlm52LongContextPolicy normalized_policy;
    SparkStatus status;

    if (policy == 0 || selected_token_indices == 0 || decode_plan == 0 ||
        context_token_count == 0u || selected_token_capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52LongContextValidatePolicy(policy);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkGlm52LongContextNormalizePolicyCopy(policy, &normalized_policy);
    if (selected_token_capacity > normalized_policy.selected_token_capacity)
    {
        selected_token_capacity = normalized_policy.selected_token_capacity;
    }
    if (context_token_count > normalized_policy.max_context_tokens &&
        (normalized_policy.policy_flags &
            SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW) != 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    SparkGlm52LongContextInitializeDecodePlan(
        decode_plan,
        &normalized_policy,
        context_token_count,
        selected_token_capacity);

    if (normalized_policy.policy_mode ==
        SPARK_GLM52_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN)
    {
        if ((normalized_policy.policy_flags &
                SPARK_GLM52_LONG_CONTEXT_POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN) == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SparkGlm52LongContextBuildFullSelection(
            &normalized_policy,
            context_token_count,
            selected_token_indices,
            selected_token_capacity,
            decode_plan);
    }

    if (context_token_count <= selected_token_capacity &&
        context_token_count <= normalized_policy.maximum_decode_scan_token_count)
    {
        return SparkGlm52LongContextBuildFullSelection(
            &normalized_policy,
            context_token_count,
            selected_token_indices,
            selected_token_capacity,
            decode_plan);
    }

    SparkGlm52LongContextBuildBoundedSelection(
        &normalized_policy,
        context_token_count,
        selected_token_indices,
        selected_token_capacity,
        decode_plan);
    if (decode_plan->selected_token_count >
        normalized_policy.maximum_decode_scan_token_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52LongContextBuildDecodeSelectionForLaneBatch(
    const SparkGlm52LongContextPolicy *policy,
    const uint32_t *context_token_counts,
    uint32_t lane_count,
    uint32_t *selected_token_indices,
    uint32_t selected_token_stride,
    uint32_t selected_token_capacity,
    SparkGlm52LongContextDecodePlan *decode_plans,
    uint32_t decode_plan_capacity)
{
    uint32_t lane_index;
    SparkStatus status;

    if (policy == 0 || context_token_counts == 0 || lane_count == 0u ||
        selected_token_indices == 0 || selected_token_stride == 0u ||
        selected_token_capacity == 0u || decode_plans == 0 ||
        decode_plan_capacity < lane_count ||
        selected_token_stride < selected_token_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (lane_index = 0u;
         lane_index < lane_count;
         ++lane_index)
    {
        status = SparkGlm52LongContextBuildDecodeSelection(
            policy,
            context_token_counts[lane_index],
            &selected_token_indices[(uint64_t)lane_index * selected_token_stride],
            selected_token_capacity,
            &decode_plans[lane_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}
