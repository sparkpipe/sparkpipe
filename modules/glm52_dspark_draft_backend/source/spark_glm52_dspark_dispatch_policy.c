#include "sparkpipe/spark_glm52_dspark.h"

#include <string.h>

static const uint32_t SparkGlm52DsparkDefaultAuxLayerIds[
    SPARK_DSPARK_AUX_LAYER_COUNT] =
    SPARK_DSPARK_AUX_LAYER_IDS_INITIALIZER;

SparkStatus SparkGlm52DsparkBuildDefaultModelContract(
    SparkGlm52DsparkModelContract *model_contract)
{
    uint32_t layer_index;

    if (model_contract == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(model_contract, 0, sizeof(*model_contract));
    model_contract->abi_version = SPARK_DSPARK_ABI_VERSION;
    model_contract->descriptor_bytes =
        SPARK_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES;
    model_contract->verifier_hidden_dtype =
        SPARK_DSPARK_VERIFIER_HIDDEN_DTYPE_BF16;
    model_contract->draft_dtype = SPARK_DSPARK_DRAFT_DTYPE_BF16;
    model_contract->draft_layer_count = SPARK_DSPARK_DRAFT_LAYER_COUNT;
    model_contract->block_size = SPARK_DSPARK_BLOCK_SIZE;
    model_contract->hidden_dimension = SPARK_DSPARK_HIDDEN_DIMENSION;
    model_contract->intermediate_dimension =
        SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION;
    model_contract->attention_head_count =
        SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT;
    model_contract->kv_head_count = SPARK_DSPARK_DRAFT_KV_HEAD_COUNT;
    model_contract->head_dimension = SPARK_DSPARK_DRAFT_HEAD_DIMENSION;
    model_contract->vocab_size = SPARK_DSPARK_FULL_VOCAB_SIZE;
    model_contract->draft_vocab_size = SPARK_DSPARK_FULL_VOCAB_SIZE;
    model_contract->markov_rank = SPARK_DSPARK_MARKOV_RANK;
    model_contract->max_anchors = SPARK_DSPARK_MAX_ANCHORS;
    model_contract->maximum_speculative_token_count =
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    model_contract->verifier_accept_k = 1u;
    model_contract->aux_layer_count = SPARK_DSPARK_AUX_LAYER_COUNT;
    model_contract->enable_confidence_head = 1u;
    model_contract->confidence_head_with_markov = 1u;
    for (layer_index = 0u;
         layer_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++layer_index)
    {
        model_contract->aux_layer_ids[layer_index] =
            SparkGlm52DsparkDefaultAuxLayerIds[layer_index];
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkValidateModelContract(
    const SparkGlm52DsparkModelContract *model_contract)
{
    uint32_t layer_index;

    if (model_contract == 0 ||
        model_contract->abi_version != SPARK_DSPARK_ABI_VERSION ||
        model_contract->descriptor_bytes !=
            SPARK_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES ||
        model_contract->verifier_hidden_dtype !=
            SPARK_DSPARK_VERIFIER_HIDDEN_DTYPE_BF16 ||
        model_contract->draft_dtype != SPARK_DSPARK_DRAFT_DTYPE_BF16 ||
        model_contract->draft_layer_count !=
            SPARK_DSPARK_DRAFT_LAYER_COUNT ||
        model_contract->block_size != SPARK_DSPARK_BLOCK_SIZE ||
        model_contract->hidden_dimension !=
            SPARK_DSPARK_HIDDEN_DIMENSION ||
        model_contract->intermediate_dimension !=
            SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION ||
        model_contract->attention_head_count !=
            SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT ||
        model_contract->kv_head_count !=
            SPARK_DSPARK_DRAFT_KV_HEAD_COUNT ||
        model_contract->head_dimension !=
            SPARK_DSPARK_DRAFT_HEAD_DIMENSION ||
        model_contract->vocab_size != SPARK_DSPARK_FULL_VOCAB_SIZE ||
        model_contract->draft_vocab_size !=
            SPARK_DSPARK_FULL_VOCAB_SIZE ||
        model_contract->markov_rank != SPARK_DSPARK_MARKOV_RANK ||
        model_contract->max_anchors != SPARK_DSPARK_MAX_ANCHORS ||
        model_contract->maximum_speculative_token_count !=
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        model_contract->verifier_accept_k != 1u ||
        model_contract->aux_layer_count != SPARK_DSPARK_AUX_LAYER_COUNT ||
        model_contract->enable_confidence_head != 1u ||
        model_contract->confidence_head_with_markov != 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (layer_index = 0u;
         layer_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++layer_index)
    {
        if (model_contract->aux_layer_ids[layer_index] !=
            SparkGlm52DsparkDefaultAuxLayerIds[layer_index])
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52DsparkNormalizePolicyFlags(
    uint32_t policy_flags)
{
    if (policy_flags == 0u)
    {
        return SPARK_DSPARK_POLICY_DEFAULT_FLAGS;
    }
    return policy_flags;
}

static uint32_t SparkGlm52DsparkNormalizeSpeculativeTokenCount(
    uint32_t token_count)
{
    if (token_count == 0u)
    {
        return SPARK_DSPARK_DEFAULT_MAX_VERIFY_TOKENS;
    }
    if (token_count > SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
    {
        return SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    }
    return token_count;
}

static uint32_t SparkGlm52DsparkNormalizeConfidenceMilli(
    uint32_t confidence_milli,
    uint32_t default_confidence_milli)
{
    if (confidence_milli == 0u)
    {
        return default_confidence_milli;
    }
    if (confidence_milli > SPARK_DSPARK_CONFIDENCE_MILLI_ONE)
    {
        return SPARK_DSPARK_CONFIDENCE_MILLI_ONE;
    }
    return confidence_milli;
}

SparkStatus SparkGlm52DsparkValidate(
    const SparkGlm52DsparkSpeculator *speculator)
{
    if (speculator == 0 ||
        speculator->abi_version != SPARK_DSPARK_ABI_VERSION ||
        speculator->descriptor_bytes != SPARK_DSPARK_DESCRIPTOR_BYTES ||
        speculator->sequence_state_count == 0u ||
        speculator->sequence_states == 0 ||
        speculator->draft_function == 0 ||
        speculator->default_speculative_token_count == 0u ||
        speculator->default_speculative_token_count >
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        (speculator->policy_flags & ~SPARK_DSPARK_POLICY_KNOWN_FLAGS) != 0u ||
        SparkGlm52DsparkValidateModelContract(
            &speculator->model_contract) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkValidateConfiguration(
    const SparkGlm52DsparkSpeculatorConfiguration *configuration)
{
    uint32_t policy_flags;

    if (configuration == 0 ||
        configuration->abi_version != SPARK_DSPARK_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->sequence_state_count == 0u ||
        configuration->sequence_states == 0 ||
        configuration->draft_function == 0 ||
        configuration->model_contract == 0 ||
        configuration->reserved != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    policy_flags = SparkGlm52DsparkNormalizePolicyFlags(
        configuration->policy_flags);
    if ((policy_flags & ~SPARK_DSPARK_POLICY_KNOWN_FLAGS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52DsparkValidateModelContract(
            configuration->model_contract) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52DsparkInitializeSequenceState(
    SparkGlm52DsparkSequenceState *sequence_state)
{
    memset(sequence_state, 0, sizeof(*sequence_state));
    sequence_state->abi_version = SPARK_DSPARK_ABI_VERSION;
    sequence_state->descriptor_bytes =
        SPARK_DSPARK_SEQUENCE_STATE_DESCRIPTOR_BYTES;
}

static SparkGlm52DsparkSequenceState *SparkGlm52DsparkFindSequenceState(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id)
{
    uint32_t state_index;

    if (sequence_id == 0u)
    {
        return 0;
    }
    for (state_index = 0u;
         state_index < speculator->sequence_state_count;
         ++state_index)
    {
        SparkGlm52DsparkSequenceState *sequence_state;

        sequence_state = &speculator->sequence_states[state_index];
        if ((sequence_state->flags &
                SPARK_DSPARK_SEQUENCE_STATE_FLAG_VALID) != 0u &&
            sequence_state->sequence_id == sequence_id)
        {
            return sequence_state;
        }
    }
    return 0;
}

static SparkGlm52DsparkSequenceState *SparkGlm52DsparkFindFreeSequenceState(
    SparkGlm52DsparkSpeculator *speculator)
{
    uint32_t state_index;

    for (state_index = 0u;
         state_index < speculator->sequence_state_count;
         ++state_index)
    {
        SparkGlm52DsparkSequenceState *sequence_state;

        sequence_state = &speculator->sequence_states[state_index];
        if ((sequence_state->flags &
                SPARK_DSPARK_SEQUENCE_STATE_FLAG_VALID) == 0u)
        {
            return sequence_state;
        }
    }
    return 0;
}

static SparkGlm52DsparkSequenceState *SparkGlm52DsparkAcquireSequenceState(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t request_id,
    uint64_t sequence_id)
{
    SparkGlm52DsparkSequenceState *sequence_state;

    sequence_state = SparkGlm52DsparkFindSequenceState(
        speculator,
        sequence_id);
    if (sequence_state != 0)
    {
        sequence_state->request_id = request_id;
        return sequence_state;
    }

    sequence_state = SparkGlm52DsparkFindFreeSequenceState(speculator);
    if (sequence_state == 0)
    {
        return 0;
    }
    SparkGlm52DsparkInitializeSequenceState(sequence_state);
    sequence_state->flags = SPARK_DSPARK_SEQUENCE_STATE_FLAG_VALID;
    sequence_state->request_id = request_id;
    sequence_state->sequence_id = sequence_id;
    return sequence_state;
}

SparkStatus SparkGlm52DsparkBuildDefaultHiddenTapPlan(
    SparkGlm52DsparkHiddenTapPlan *tap_plan)
{
    uint32_t tap_index;

    if (tap_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(tap_plan, 0, sizeof(*tap_plan));
    tap_plan->abi_version = SPARK_DSPARK_ABI_VERSION;
    tap_plan->descriptor_bytes = SPARK_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES;
    tap_plan->aux_layer_count = SPARK_DSPARK_AUX_LAYER_COUNT;
    tap_plan->hidden_dimension = SPARK_DSPARK_HIDDEN_DIMENSION;
    tap_plan->pp_stage_count = SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT;
    tap_plan->pp_stage_layer_count = SPARK_GLM52_MODEL_DSPARK_PP_STAGE_LAYER_COUNT;

    for (tap_index = 0u;
         tap_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        uint32_t target_layer_index;
        SparkGlm52DsparkTapStage *tap_stage;

        target_layer_index = SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_INDEX(
            SparkGlm52DsparkDefaultAuxLayerIds[tap_index]);
        tap_stage = &tap_plan->tap_stages[tap_index];
        tap_stage->target_layer_index = target_layer_index;
        tap_stage->stage_index = target_layer_index / 6u;
        tap_stage->stage_first_layer_index = tap_stage->stage_index * 6u;
        tap_stage->stage_layer_count = 6u;
        tap_stage->layer_offset_in_stage =
            target_layer_index - tap_stage->stage_first_layer_index;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkValidateHiddenTapPlan(
    const SparkGlm52DsparkHiddenTapPlan *tap_plan)
{
    uint32_t tap_index;

    if (tap_plan == 0 ||
        tap_plan->abi_version != SPARK_DSPARK_ABI_VERSION ||
        tap_plan->descriptor_bytes !=
            SPARK_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES ||
        tap_plan->aux_layer_count != SPARK_DSPARK_AUX_LAYER_COUNT ||
        tap_plan->hidden_dimension != SPARK_DSPARK_HIDDEN_DIMENSION ||
        tap_plan->pp_stage_count != 13u ||
        tap_plan->pp_stage_layer_count != 6u ||
        tap_plan->reserved0 != 0u ||
        tap_plan->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (tap_index = 0u;
         tap_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        const SparkGlm52DsparkTapStage *tap_stage;
        uint32_t target_layer_index;

        target_layer_index = SPARK_GLM52_MODEL_DSPARK_AUX_CAPTURE_LAYER_INDEX(
            SparkGlm52DsparkDefaultAuxLayerIds[tap_index]);
        tap_stage = &tap_plan->tap_stages[tap_index];
        if (tap_stage->target_layer_index != target_layer_index ||
            tap_stage->stage_index != target_layer_index / 6u ||
            tap_stage->stage_first_layer_index !=
                (target_layer_index / 6u) * 6u ||
            tap_stage->stage_layer_count != 6u ||
            tap_stage->layer_offset_in_stage !=
                target_layer_index - tap_stage->stage_first_layer_index ||
            tap_stage->reserved != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkInitialize(
    SparkGlm52DsparkSpeculator *speculator,
    const SparkGlm52DsparkSpeculatorConfiguration *configuration)
{
    uint32_t state_index;
    SparkStatus status;

    if (speculator == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52DsparkValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(speculator, 0, sizeof(*speculator));
    speculator->abi_version = SPARK_DSPARK_ABI_VERSION;
    speculator->descriptor_bytes = SPARK_DSPARK_DESCRIPTOR_BYTES;
    speculator->policy_flags = SparkGlm52DsparkNormalizePolicyFlags(
        configuration->policy_flags);
    speculator->sequence_state_count = configuration->sequence_state_count;
    speculator->default_speculative_token_count =
        SparkGlm52DsparkNormalizeSpeculativeTokenCount(
            configuration->default_speculative_token_count);
    speculator->minimum_confidence_milli =
        SparkGlm52DsparkNormalizeConfidenceMilli(
            configuration->minimum_confidence_milli,
            SPARK_DSPARK_DEFAULT_MIN_CONFIDENCE_MILLI);
    speculator->realtime_minimum_confidence_milli =
        SparkGlm52DsparkNormalizeConfidenceMilli(
            configuration->realtime_minimum_confidence_milli,
            SPARK_DSPARK_DEFAULT_REALTIME_MIN_CONFIDENCE_MILLI);
    speculator->next_tap_generation = 1u;
    speculator->next_draft_generation = 1u;
    speculator->model_contract = *configuration->model_contract;
    speculator->sequence_states = configuration->sequence_states;
    speculator->draft_function = configuration->draft_function;
    speculator->draft_context = configuration->draft_context;

    for (state_index = 0u;
         state_index < speculator->sequence_state_count;
         ++state_index)
    {
        SparkGlm52DsparkInitializeSequenceState(
            &speculator->sequence_states[state_index]);
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkMarkVerifierTapsReady(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint64_t *tap_generation_out)
{
    SparkGlm52DsparkSequenceState *sequence_state;
    SparkStatus status;

    status = SparkGlm52DsparkValidate(speculator);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (sequence_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    sequence_state = SparkGlm52DsparkAcquireSequenceState(
        speculator,
        request_id,
        sequence_id);
    if (sequence_state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    sequence_state->flags &=
        ~(SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY |
          SPARK_DSPARK_SEQUENCE_STATE_FLAG_VERIFY_INFLIGHT);
    sequence_state->flags |=
        SPARK_DSPARK_SEQUENCE_STATE_FLAG_TAPS_READY;
    sequence_state->sequence_position = sequence_position;
    sequence_state->tap_generation = speculator->next_tap_generation;
    speculator->next_tap_generation += 1u;
    if (speculator->next_tap_generation == 0u)
    {
        speculator->next_tap_generation = 1u;
    }
    sequence_state->draft_token_count = 0u;
    speculator->tap_ready_count += 1u;

    if (tap_generation_out != 0)
    {
        *tap_generation_out = sequence_state->tap_generation;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkValidateDraftRequest(
    const SparkGlm52DsparkDraftRequest *request)
{
    if (request == 0 ||
        request->abi_version != SPARK_DSPARK_ABI_VERSION ||
        request->descriptor_bytes != SPARK_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES ||
        request->requested_token_count == 0u ||
        request->requested_token_count >
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        request->sequence_id == 0u ||
        request->tap_generation == 0u ||
        request->reserved != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkValidateDraftResult(
    const SparkGlm52DsparkDraftResult *result,
    uint32_t requested_token_count)
{
    uint32_t token_index;

    if (result == 0 ||
        result->abi_version != SPARK_DSPARK_ABI_VERSION ||
        result->descriptor_bytes != SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES ||
        (result->flags & ~SPARK_DSPARK_DRAFT_RESULT_KNOWN_FLAGS) != 0u ||
        result->token_count == 0u ||
        result->token_count > requested_token_count ||
        result->token_count > SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (token_index = 0u;
         token_index < result->token_count;
         ++token_index)
    {
        if (result->token_ids[token_index] >= SPARK_DSPARK_FULL_VOCAB_SIZE ||
            result->confidence_milli[token_index] >
                SPARK_DSPARK_CONFIDENCE_MILLI_ONE)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52DsparkConfidenceThresholdForRequest(
    const SparkGlm52DsparkSpeculator *speculator,
    const SparkGlm52DsparkDraftRequest *request)
{
    if (request->priority >= SPARK_DSPARK_POLICY_REALTIME_PRIORITY_THRESHOLD)
    {
        return speculator->realtime_minimum_confidence_milli;
    }
    return speculator->minimum_confidence_milli;
}

static uint32_t SparkGlm52DsparkAcceptedDraftLengthByConfidence(
    const SparkGlm52DsparkSpeculator *speculator,
    const SparkGlm52DsparkDraftRequest *request,
    const SparkGlm52DsparkDraftResult *result)
{
    uint32_t token_index;
    uint32_t confidence_threshold_milli;

    confidence_threshold_milli = SparkGlm52DsparkConfidenceThresholdForRequest(
        speculator,
        request);
    for (token_index = 0u;
         token_index < result->token_count;
         ++token_index)
    {
        if (result->confidence_milli[token_index] < confidence_threshold_milli)
        {
            return token_index;
        }
    }
    return result->token_count;
}

SparkStatus SparkGlm52DsparkEnsureDraft(
    SparkGlm52DsparkSpeculator *speculator,
    const SparkGlm52DsparkDraftRequest *request)
{
    SparkGlm52DsparkSequenceState *sequence_state;
    SparkGlm52DsparkDraftResult result;
    uint32_t accepted_by_confidence;
    uint32_t token_index;
    SparkStatus status;

    status = SparkGlm52DsparkValidate(speculator);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52DsparkValidateDraftRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    sequence_state = SparkGlm52DsparkFindSequenceState(
        speculator,
        request->sequence_id);
    if (sequence_state == 0 ||
        (sequence_state->flags &
            SPARK_DSPARK_SEQUENCE_STATE_FLAG_TAPS_READY) == 0u ||
        sequence_state->tap_generation != request->tap_generation)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if ((sequence_state->flags &
            SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY) != 0u)
    {
        return SPARK_STATUS_OK;
    }

    memset(&result, 0, sizeof(result));
    result.abi_version = SPARK_DSPARK_ABI_VERSION;
    result.descriptor_bytes = SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;

    speculator->draft_request_count += 1u;
    status = speculator->draft_function(
        speculator->draft_context,
        request,
        &result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52DsparkValidateDraftResult(
        &result,
        request->requested_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    accepted_by_confidence = SparkGlm52DsparkAcceptedDraftLengthByConfidence(
        speculator,
        request,
        &result);
    if (accepted_by_confidence == 0u)
    {
        speculator->draft_rejected_by_confidence_count += 1u;
        return SPARK_STATUS_NOT_FOUND;
    }
    if (accepted_by_confidence < result.token_count)
    {
        result.flags |= SPARK_DSPARK_DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED;
    }

    sequence_state->flags |= SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY;
    sequence_state->draft_generation = speculator->next_draft_generation;
    speculator->next_draft_generation += 1u;
    if (speculator->next_draft_generation == 0u)
    {
        speculator->next_draft_generation = 1u;
    }
    sequence_state->draft_token_count = accepted_by_confidence;
    for (token_index = 0u;
         token_index < SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
         ++token_index)
    {
        if (token_index < accepted_by_confidence)
        {
            sequence_state->draft_token_ids[token_index] =
                result.token_ids[token_index];
            sequence_state->draft_confidence_milli[token_index] =
                result.confidence_milli[token_index];
        }
        else
        {
            sequence_state->draft_token_ids[token_index] = 0u;
            sequence_state->draft_confidence_milli[token_index] = 0u;
        }
    }
    speculator->draft_success_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkGetDraft(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id,
    SparkGlm52DsparkDraftResult *draft_result)
{
    SparkGlm52DsparkSequenceState *sequence_state;
    uint32_t token_index;
    SparkStatus status;

    status = SparkGlm52DsparkValidate(speculator);
    if (status != SPARK_STATUS_OK || draft_result == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    sequence_state = SparkGlm52DsparkFindSequenceState(
        speculator,
        sequence_id);
    if (sequence_state == 0 ||
        (sequence_state->flags &
            SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY) == 0u ||
        sequence_state->draft_token_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(draft_result, 0, sizeof(*draft_result));
    draft_result->abi_version = SPARK_DSPARK_ABI_VERSION;
    draft_result->descriptor_bytes =
        SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
    draft_result->token_count = sequence_state->draft_token_count;
    for (token_index = 0u;
         token_index < sequence_state->draft_token_count;
         ++token_index)
    {
        draft_result->token_ids[token_index] =
            sequence_state->draft_token_ids[token_index];
        draft_result->confidence_milli[token_index] =
            sequence_state->draft_confidence_milli[token_index];
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52DsparkValidateVerifyResult(
    const SparkGlm52DsparkVerifyResult *verify_result)
{
    if (verify_result == 0 ||
        verify_result->abi_version != SPARK_DSPARK_ABI_VERSION ||
        verify_result->descriptor_bytes !=
            SPARK_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES ||
        (verify_result->flags & ~SPARK_DSPARK_VERIFY_RESULT_KNOWN_FLAGS) != 0u ||
        verify_result->proposed_token_count == 0u ||
        verify_result->proposed_token_count >
            SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        verify_result->accepted_draft_token_count >
            verify_result->proposed_token_count ||
        verify_result->committed_token_count >
            verify_result->proposed_token_count + 1u ||
        verify_result->reserved != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((verify_result->flags &
            SPARK_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL) != 0u &&
        verify_result->accepted_draft_token_count !=
            verify_result->proposed_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((verify_result->flags &
            SPARK_DSPARK_VERIFY_RESULT_FLAG_REJECTED) != 0u &&
        verify_result->accepted_draft_token_count >=
            verify_result->proposed_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkCompleteVerify(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id,
    const SparkGlm52DsparkVerifyResult *verify_result)
{
    SparkGlm52DsparkSequenceState *sequence_state;
    SparkStatus status;

    status = SparkGlm52DsparkValidate(speculator);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52DsparkValidateVerifyResult(verify_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    sequence_state = SparkGlm52DsparkFindSequenceState(
        speculator,
        sequence_id);
    if (sequence_state == 0 ||
        (sequence_state->flags &
            SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY) == 0u ||
        sequence_state->draft_token_count != verify_result->proposed_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    sequence_state->accepted_draft_token_count +=
        verify_result->accepted_draft_token_count;
    sequence_state->committed_token_count += verify_result->committed_token_count;
    if (verify_result->accepted_draft_token_count <
        verify_result->proposed_token_count)
    {
        sequence_state->rejected_token_count +=
            verify_result->proposed_token_count -
            verify_result->accepted_draft_token_count;
        speculator->rejected_token_count +=
            verify_result->proposed_token_count -
            verify_result->accepted_draft_token_count;
    }
    speculator->accepted_draft_token_count +=
        verify_result->accepted_draft_token_count;
    speculator->committed_token_count += verify_result->committed_token_count;
    speculator->verify_dispatch_count += 1u;

    sequence_state->flags &=
        ~(SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY |
          SPARK_DSPARK_SEQUENCE_STATE_FLAG_VERIFY_INFLIGHT);
    sequence_state->draft_token_count = 0u;
    return SPARK_STATUS_OK;
}


SparkStatus SparkGlm52DsparkResolveVerifierTokens(
    const uint32_t *draft_token_ids,
    uint32_t draft_token_count,
    const uint32_t *verifier_token_ids,
    uint32_t verifier_token_count,
    SparkGlm52DsparkVerifyResult *verify_result)
{
    uint32_t token_index;
    uint32_t accepted_token_count;

    if (draft_token_ids == 0 || verifier_token_ids == 0 || verify_result == 0 ||
        draft_token_count == 0u ||
        draft_token_count > SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        verifier_token_count < draft_token_count ||
        verifier_token_count > draft_token_count + 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (token_index = 0u;
         token_index < draft_token_count;
         ++token_index)
    {
        if (draft_token_ids[token_index] >= SPARK_DSPARK_FULL_VOCAB_SIZE ||
            verifier_token_ids[token_index] >= SPARK_DSPARK_FULL_VOCAB_SIZE)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if (verifier_token_count > draft_token_count &&
        verifier_token_ids[draft_token_count] >= SPARK_DSPARK_FULL_VOCAB_SIZE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    accepted_token_count = draft_token_count;
    for (token_index = 0u;
         token_index < draft_token_count;
         ++token_index)
    {
        if (draft_token_ids[token_index] != verifier_token_ids[token_index])
        {
            accepted_token_count = token_index;
            break;
        }
    }

    memset(verify_result, 0, sizeof(*verify_result));
    verify_result->abi_version = SPARK_DSPARK_ABI_VERSION;
    verify_result->descriptor_bytes =
        SPARK_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES;
    verify_result->proposed_token_count = draft_token_count;
    verify_result->accepted_draft_token_count = accepted_token_count;
    if (accepted_token_count == draft_token_count)
    {
        verify_result->flags = SPARK_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL;
        verify_result->committed_token_count = verifier_token_count;
        if (verifier_token_count > draft_token_count)
        {
            verify_result->fallback_token_id = verifier_token_ids[draft_token_count];
        }
        return SPARK_STATUS_OK;
    }

    verify_result->flags = SPARK_DSPARK_VERIFY_RESULT_FLAG_REJECTED;
    verify_result->committed_token_count = accepted_token_count + 1u;
    verify_result->fallback_token_id = verifier_token_ids[accepted_token_count];
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52DsparkCancelSequence(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id)
{
    SparkGlm52DsparkSequenceState *sequence_state;
    SparkStatus status;

    status = SparkGlm52DsparkValidate(speculator);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    sequence_state = SparkGlm52DsparkFindSequenceState(
        speculator,
        sequence_id);
    if (sequence_state == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    SparkGlm52DsparkInitializeSequenceState(sequence_state);
    return SPARK_STATUS_OK;
}
