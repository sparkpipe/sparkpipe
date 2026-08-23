#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_dspark.h"

#define SPARK_TEST_DSPARK_SEQUENCE_STATE_COUNT 4u

typedef struct SparkTestDsparkBackend
{
    uint32_t call_count;
    uint32_t token_base;
    uint32_t confidence_milli[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint64_t last_sequence_id;
    uint64_t last_tap_generation;
} SparkTestDsparkBackend;

static SparkStatus SparkTestDsparkDraftBackend(
    void *context,
    const SparkGlm52DsparkDraftRequest *request,
    SparkGlm52DsparkDraftResult *result)
{
    SparkTestDsparkBackend *backend;
    uint32_t token_index;

    backend = (SparkTestDsparkBackend *)context;
    assert(backend != 0);
    assert(request != 0);
    assert(result != 0);
    assert(request->requested_token_count != 0u);
    assert(request->requested_token_count <=
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);

    backend->call_count += 1u;
    backend->last_sequence_id = request->sequence_id;
    backend->last_tap_generation = request->tap_generation;

    result->abi_version = SPARK_DSPARK_ABI_VERSION;
    result->descriptor_bytes = SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
    result->token_count = request->requested_token_count;
    for (token_index = 0u;
         token_index < request->requested_token_count;
         ++token_index)
    {
        result->token_ids[token_index] = backend->token_base + token_index;
        result->confidence_milli[token_index] =
            backend->confidence_milli[token_index] == 0u ?
                900u : backend->confidence_milli[token_index];
    }
    return SPARK_STATUS_OK;
}

static void SparkTestDsparkInitializesSpeculator(
    SparkGlm52DsparkSpeculator *speculator,
    SparkGlm52DsparkSequenceState *sequence_states,
    SparkTestDsparkBackend *backend)
{
    SparkGlm52DsparkSpeculatorConfiguration configuration;
    SparkGlm52DsparkModelContract model_contract;

    memset(&configuration, 0, sizeof(configuration));
    assert(SparkGlm52DsparkBuildDefaultModelContract(
        &model_contract) == SPARK_STATUS_OK);
    configuration.abi_version = SPARK_DSPARK_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.sequence_state_count = SPARK_TEST_DSPARK_SEQUENCE_STATE_COUNT;
    configuration.default_speculative_token_count = 7u;
    configuration.minimum_confidence_milli = 350u;
    configuration.realtime_minimum_confidence_milli = 250u;
    configuration.sequence_states = sequence_states;
    configuration.draft_function = SparkTestDsparkDraftBackend;
    configuration.draft_context = backend;
    configuration.model_contract = &model_contract;
    assert(SparkGlm52DsparkInitialize(
        speculator,
        &configuration) == SPARK_STATUS_OK);
}

static void SparkTestDsparkModelContractRejectsNonBf16VerifierTaps(void)
{
    SparkGlm52DsparkModelContract model_contract;

    assert(SparkGlm52DsparkBuildDefaultModelContract(
        &model_contract) == SPARK_STATUS_OK);
    assert(SparkGlm52DsparkValidateModelContract(
        &model_contract) == SPARK_STATUS_OK);
    model_contract.verifier_hidden_dtype = 0u;
    assert(SparkGlm52DsparkValidateModelContract(
        &model_contract) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestDsparkDefaultTapPlanMatchesGlm52Ring(void)
{
    SparkGlm52DsparkHiddenTapPlan tap_plan;
    static const uint32_t expected_layers[SPARK_DSPARK_AUX_LAYER_COUNT] =
        { 7u, 22u, 38u, 54u, 69u };
    /* PP7 split [12,11x6]: taps land in stages {0,1,3,4,6}. */
    static const uint32_t expected_stages[SPARK_DSPARK_AUX_LAYER_COUNT] =
        { 0u, 1u, 3u, 4u, 6u };
    static const uint32_t expected_firsts[SPARK_DSPARK_AUX_LAYER_COUNT] =
        { 0u, 12u, 34u, 45u, 67u };
    static const uint32_t expected_counts[SPARK_DSPARK_AUX_LAYER_COUNT] =
        { 12u, 11u, 11u, 11u, 11u };
    uint32_t tap_index;

    assert(SparkGlm52DsparkBuildDefaultHiddenTapPlan(
        &tap_plan) == SPARK_STATUS_OK);
    assert(SparkGlm52DsparkValidateHiddenTapPlan(
        &tap_plan) == SPARK_STATUS_OK);
    assert(tap_plan.aux_layer_count == SPARK_DSPARK_AUX_LAYER_COUNT);
    assert(tap_plan.pp_stage_count ==
        SPARK_GLM52_MODEL_DSPARK_PP_STAGE_COUNT);
    assert(tap_plan.pp_stage_max_layer_count ==
        SPARK_GLM52_MODEL_DSPARK_PP_STAGE_MAX_LAYER_COUNT);

    for (tap_index = 0u;
         tap_index < SPARK_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        const SparkGlm52DsparkTapStage *tap_stage;

        tap_stage = &tap_plan.tap_stages[tap_index];
        assert(tap_stage->target_layer_index == expected_layers[tap_index]);
        assert(tap_stage->stage_index == expected_stages[tap_index]);
        assert(tap_stage->stage_first_layer_index ==
            expected_firsts[tap_index]);
        assert(tap_stage->stage_layer_count == expected_counts[tap_index]);
        assert(tap_stage->layer_offset_in_stage ==
            expected_layers[tap_index] - tap_stage->stage_first_layer_index);
    }

    tap_plan.tap_stages[2u].target_layer_index = 39u;
    assert(SparkGlm52DsparkValidateHiddenTapPlan(
        &tap_plan) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestDsparkDraftLifecycleWithConfidenceTruncation(void)
{
    SparkGlm52DsparkSpeculator speculator;
    SparkGlm52DsparkSequenceState sequence_states[
        SPARK_TEST_DSPARK_SEQUENCE_STATE_COUNT];
    SparkTestDsparkBackend backend;
    SparkGlm52DsparkDraftRequest draft_request;
    SparkGlm52DsparkDraftResult draft_result;
    SparkGlm52DsparkVerifyResult verify_result;
    uint64_t tap_generation;

    memset(&backend, 0, sizeof(backend));
    backend.token_base = 120000u;
    backend.confidence_milli[0u] = 900u;
    backend.confidence_milli[1u] = 800u;
    backend.confidence_milli[2u] = 200u;
    SparkTestDsparkInitializesSpeculator(
        &speculator,
        sequence_states,
        &backend);

    assert(SparkGlm52DsparkMarkVerifierTapsReady(
        &speculator,
        100u,
        1000u,
        64u,
        &tap_generation) == SPARK_STATUS_OK);
    assert(tap_generation != 0u);

    memset(&draft_request, 0, sizeof(draft_request));
    draft_request.abi_version = SPARK_DSPARK_ABI_VERSION;
    draft_request.descriptor_bytes =
        SPARK_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
    draft_request.requested_token_count = 7u;
    draft_request.priority = 100u;
    draft_request.request_id = 100u;
    draft_request.sequence_id = 1000u;
    draft_request.sequence_position = 64u;
    draft_request.tap_generation = tap_generation;
    assert(SparkGlm52DsparkEnsureDraft(
        &speculator,
        &draft_request) == SPARK_STATUS_OK);
    assert(backend.call_count == 1u);

    assert(SparkGlm52DsparkGetDraft(
        &speculator,
        1000u,
        &draft_result) == SPARK_STATUS_OK);
    assert(draft_result.token_count == 2u);
    assert(draft_result.token_ids[0u] == 120000u);
    assert(draft_result.token_ids[1u] == 120001u);
    assert(draft_result.confidence_milli[1u] == 800u);

    memset(&verify_result, 0, sizeof(verify_result));
    verify_result.abi_version = SPARK_DSPARK_ABI_VERSION;
    verify_result.descriptor_bytes =
        SPARK_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES;
    verify_result.flags = SPARK_DSPARK_VERIFY_RESULT_FLAG_REJECTED;
    verify_result.proposed_token_count = 2u;
    verify_result.accepted_draft_token_count = 1u;
    verify_result.committed_token_count = 2u;
    verify_result.fallback_token_id = 130000u;
    assert(SparkGlm52DsparkCompleteVerify(
        &speculator,
        1000u,
        &verify_result) == SPARK_STATUS_OK);
    assert(speculator.accepted_draft_token_count == 1u);
    assert(speculator.committed_token_count == 2u);
    assert(speculator.rejected_token_count == 1u);
    assert(SparkGlm52DsparkGetDraft(
        &speculator,
        1000u,
        &draft_result) == SPARK_STATUS_NOT_FOUND);
}


static void SparkTestDsparkResolvesVerifierTokens(void)
{
    SparkGlm52DsparkVerifyResult verify_result;
    const uint32_t draft_tokens[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT] = {
        100u, 101u, 102u, 103u, 104u, 105u, 106u
    };
    const uint32_t rejected_verifier_tokens[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT] = {
        100u, 101u, 777u, 778u, 779u, 780u, 781u
    };
    const uint32_t accepted_with_bonus_tokens[
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u] = {
        100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u
    };

    assert(SparkGlm52DsparkResolveVerifierTokens(
        draft_tokens,
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT,
        rejected_verifier_tokens,
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT,
        &verify_result) == SPARK_STATUS_OK);
    assert((verify_result.flags &
        SPARK_DSPARK_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    assert(verify_result.proposed_token_count ==
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);
    assert(verify_result.accepted_draft_token_count == 2u);
    assert(verify_result.committed_token_count == 3u);
    assert(verify_result.fallback_token_id == 777u);

    assert(SparkGlm52DsparkResolveVerifierTokens(
        draft_tokens,
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT,
        accepted_with_bonus_tokens,
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u,
        &verify_result) == SPARK_STATUS_OK);
    assert((verify_result.flags &
        SPARK_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL) != 0u);
    assert(verify_result.accepted_draft_token_count ==
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);
    assert(verify_result.committed_token_count ==
        SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u);
    assert(verify_result.fallback_token_id == 107u);
}

int main(void)
{
    SparkTestDsparkModelContractRejectsNonBf16VerifierTaps();
    SparkTestDsparkDefaultTapPlanMatchesGlm52Ring();
    SparkTestDsparkDraftLifecycleWithConfidenceTruncation();
    SparkTestDsparkResolvesVerifierTokens();
    return 0;
}
