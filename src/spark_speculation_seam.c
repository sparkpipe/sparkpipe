#include "sparkpipe/spark_speculation_seam.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_draft_bridge.h"

#define SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID UINT64_MAX
#define SPARK_SPECULATION_SEAM_BF16_BYTES 2u
#define SPARK_SPECULATION_SEAM_CONTROL_VALUE_MAX UINT64_C(0xFFFFFFFF)
#define SPARK_SPECULATION_SEAM_NO_SELECTION UINT32_MAX

typedef struct SparkSpeculationSeamLane
{
    const uint32_t *stash_committed_token_ids;
    const void *stash_tap_rows;
    uint64_t stash_sequence_id;
    uint64_t remote_generation;
    uint32_t stash_committed_token_count;
    uint32_t stash_tap_row_count;
} SparkSpeculationSeamLane;

struct SparkSpeculationSeam
{
    SparkSpeculationSpeculator *speculator;
    SparkSpeculationSequenceState *sequence_states;
    SparkSpeculationSeamLane *lanes;
    SparkDraftBridgeNode *nodes;
    SparkDraftBridge *bridge;
    uint32_t available_source_mask;
    uint32_t enabled_source_mask;
    uint32_t lane_count;
    uint32_t max_committed_token_count;
    uint32_t max_tap_row_count;
    uint32_t draft_time_budget_ms;
    uint32_t draft_max_depth;
    uint32_t draft_max_node_count;
    uint32_t tap_row_bytes;
};

SparkStatus SparkSpeculationSeamParseControl(
    const char *control_value,
    uint32_t available_source_mask,
    uint32_t *enabled_source_mask_out)
{
    unsigned long long parsed;
    char *parse_end;

    if (enabled_source_mask_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (control_value == 0 ||
        strcmp(control_value, "1") == 0)
    {
        *enabled_source_mask_out = available_source_mask;
        return SPARK_STATUS_OK;
    }
    if (control_value[0] == '\0')
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    errno = 0;
    parse_end = 0;
    parsed = strtoull(control_value, &parse_end, 0);
    if (errno != 0 ||
        parse_end == control_value ||
        *parse_end != '\0' ||
        parsed > SPARK_SPECULATION_SEAM_CONTROL_VALUE_MAX)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    if (((uint32_t)parsed & ~available_source_mask) != 0u)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    *enabled_source_mask_out = (uint32_t)parsed;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSpeculationSeamValidateConfiguration(
    const SparkSpeculationSeamConfiguration *configuration,
    uint32_t *enabled_source_mask_out,
    uint32_t *tap_row_bytes_out)
{
    uint32_t enabled_source_mask;
    uint32_t remote_source_mask;
    uint32_t tap_row_bytes;
    SparkStatus status;

    if (configuration->abi_version != SPARK_SPECULATION_SEAM_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_SPECULATION_SEAM_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    if ((configuration->available_source_mask &
            ~SPARK_SPECULATION_SEAM_KNOWN_SOURCES) != 0u)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    if (configuration->lane_count == 0u ||
        configuration->max_committed_token_count == 0u)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    if (SparkSpeculationPolicyValidateModelContract(
            &configuration->model_contract) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    status = SparkSpeculationSeamParseControl(
        configuration->control_value,
        configuration->available_source_mask,
        &enabled_source_mask);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    enabled_source_mask &= configuration->available_source_mask;
    remote_source_mask =
        enabled_source_mask & SPARK_SPECULATION_SEAM_REMOTE_SOURCES;
    tap_row_bytes = 0u;
    if (remote_source_mask != 0u)
    {
        if (configuration->bridge_host == 0 ||
            configuration->bridge_host[0] == '\0' ||
            configuration->bridge_port == 0u ||
            configuration->bridge_port > SPARK_DRAFT_BRIDGE_MAX_PORT ||
            configuration->target_model[0] == '\0' ||
            memchr(
                configuration->target_model,
                '\0',
                SPARK_SPECULATION_SEAM_TARGET_MODEL_BYTES) == 0 ||
            configuration->connect_timeout_ms == 0u ||
            configuration->io_timeout_ms == 0u ||
            configuration->draft_max_depth == 0u ||
            configuration->draft_max_depth > SPARK_DRAFT_BRIDGE_MAX_TREE_DEPTH ||
            configuration->draft_max_node_count == 0u ||
            configuration->draft_max_node_count >
                SPARK_DRAFT_BRIDGE_MAX_TREE_NODES ||
            configuration->draft_time_budget_ms >
                SPARK_DRAFT_BRIDGE_MAX_TIME_BUDGET_MS)
        {
            return SPARK_STATUS_SCHEMA_ERROR;
        }
        if ((enabled_source_mask & SPARK_SPECULATION_SEAM_SOURCE_DFLASH2) !=
            0u)
        {
            if (configuration->max_tap_row_count == 0u ||
                configuration->model_contract.aux_layer_count == 0u)
            {
                return SPARK_STATUS_SCHEMA_ERROR;
            }
            tap_row_bytes =
                configuration->model_contract.aux_layer_count *
                configuration->model_contract.hidden_dimension *
                SPARK_SPECULATION_SEAM_BF16_BYTES;
        }
        else
        {
            if (configuration->max_tap_row_count != 0u)
            {
                return SPARK_STATUS_SCHEMA_ERROR;
            }
            tap_row_bytes = 0u;
        }
    }
    *enabled_source_mask_out = enabled_source_mask;
    *tap_row_bytes_out = tap_row_bytes;
    return SPARK_STATUS_OK;
}

static uint32_t SparkSpeculationSeamConfidenceMilliFromScore(
    float score)
{
    if (!(score > 0.0f))
    {
        return 0u;
    }
    if (score >= 1.0f)
    {
        return SPARK_SPECULATION_CONFIDENCE_MILLI_ONE;
    }
    return (uint32_t)(score * (float)SPARK_SPECULATION_CONFIDENCE_MILLI_ONE);
}

static SparkStatus SparkSpeculationSeamExtractChain(
    const SparkDraftBridgeNode *nodes,
    uint32_t node_count,
    uint32_t requested_token_count,
    SparkSpeculationPolicyDraftResult *result)
{
    uint32_t chain_node_indices[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t node_index;
    uint32_t best_index;
    uint32_t token_count;
    uint32_t walk_index;
    uint32_t chain_index;

    if (node_count == 0u ||
        nodes[0].parent_index != SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX ||
        nodes[0].depth != 0u)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    for (node_index = 1u; node_index < node_count; ++node_index)
    {
        if (nodes[node_index].parent_index >= node_index ||
            nodes[node_index].depth !=
                nodes[nodes[node_index].parent_index].depth + 1u)
        {
            return SPARK_STATUS_VALIDATION_FAILED;
        }
    }
    best_index = SPARK_SPECULATION_SEAM_NO_SELECTION;
    for (node_index = 1u; node_index < node_count; ++node_index)
    {
        if (best_index == SPARK_SPECULATION_SEAM_NO_SELECTION ||
            nodes[node_index].depth > nodes[best_index].depth ||
            (nodes[node_index].depth == nodes[best_index].depth &&
                nodes[node_index].score > nodes[best_index].score))
        {
            best_index = node_index;
        }
    }
    if (best_index == SPARK_SPECULATION_SEAM_NO_SELECTION)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    token_count = nodes[best_index].depth;
    if (token_count > requested_token_count)
    {
        token_count = requested_token_count;
    }
    if (token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT)
    {
        token_count = SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT;
    }
    walk_index = best_index;
    for (chain_index = 0u; chain_index < token_count; ++chain_index)
    {
        chain_node_indices[chain_index] = walk_index;
        walk_index = nodes[walk_index].parent_index;
    }
    result->token_count = token_count;
    for (chain_index = 0u; chain_index < token_count; ++chain_index)
    {
        node_index = chain_node_indices[token_count - 1u - chain_index];
        result->token_ids[chain_index] = nodes[node_index].token_id;
        result->confidence_milli[chain_index] =
            SparkSpeculationSeamConfidenceMilliFromScore(
                nodes[node_index].score);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkSpeculationSeamDraftCallback(
    void *context,
    const SparkSpeculationPolicyDraftRequest *request,
    SparkSpeculationPolicyDraftResult *result)
{
    SparkSpeculationSeam *seam;
    SparkSpeculationSeamLane *lane;
    SparkDraftBridgeProposalInfo proposal_info;
    uint32_t wire_source_mask;
    uint32_t lane_index;
    uint32_t node_count;
    uint32_t depth;
    SparkStatus status;

    seam = (SparkSpeculationSeam *)context;
    lane = 0;
    for (lane_index = 0u; lane_index < seam->lane_count; ++lane_index)
    {
        if (seam->lanes[lane_index].stash_sequence_id == request->sequence_id)
        {
            lane = &seam->lanes[lane_index];
            break;
        }
    }
    if (lane == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    wire_source_mask =
        seam->enabled_source_mask & SPARK_SPECULATION_SEAM_REMOTE_SOURCES;
    if (lane->stash_tap_row_count == 0u)
    {
        wire_source_mask &= ~SPARK_SPECULATION_SEAM_REMOTE_TAP_SOURCES;
    }
    if (wire_source_mask == 0u || seam->bridge == 0)
    {
        return SPARK_STATUS_UNSUPPORTED;
    }
    depth = request->requested_token_count;
    if (depth > SPARK_DRAFT_BRIDGE_MAX_DEPTH)
    {
        depth = SPARK_DRAFT_BRIDGE_MAX_DEPTH;
    }
    node_count = 0u;
    status = SparkDraftBridgeProposeTree(
        seam->bridge,
        wire_source_mask,
        request->sequence_id,
        lane->remote_generation,
        (uint64_t)(lane->stash_committed_token_count - 1u),
        lane->stash_committed_token_ids[lane->stash_committed_token_count - 1u],
        lane->stash_committed_token_ids,
        lane->stash_committed_token_count,
        lane->stash_tap_rows,
        lane->stash_tap_row_count,
        depth,
        seam->draft_time_budget_ms,
        seam->draft_max_depth,
        seam->nodes,
        seam->draft_max_node_count,
        &node_count,
        &proposal_info);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkSpeculationSeamExtractChain(
        seam->nodes,
        node_count,
        request->requested_token_count,
        result);
}

SparkStatus SparkSpeculationSeamInitialize(
    const SparkSpeculationSeamConfiguration *configuration,
    SparkSpeculationSeam **seam_out)
{
    SparkSpeculationConfiguration policy_configuration;
    SparkDraftBridgeConfig bridge_configuration;
    SparkSpeculationSeam *seam;
    uint32_t enabled_source_mask;
    uint32_t tap_row_bytes;
    uint32_t lane_index;
    SparkStatus status;

    if (configuration == 0 || seam_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *seam_out = 0;
    status = SparkSpeculationSeamValidateConfiguration(
        configuration,
        &enabled_source_mask,
        &tap_row_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    seam = (SparkSpeculationSeam *)calloc(1u, sizeof(*seam));
    if (seam == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    seam->available_source_mask = configuration->available_source_mask;
    seam->enabled_source_mask = enabled_source_mask;
    seam->lane_count = configuration->lane_count;
    seam->max_committed_token_count = configuration->max_committed_token_count;
    seam->max_tap_row_count = configuration->max_tap_row_count;
    seam->draft_time_budget_ms = configuration->draft_time_budget_ms;
    seam->draft_max_depth = configuration->draft_max_depth;
    seam->draft_max_node_count = configuration->draft_max_node_count;
    seam->tap_row_bytes = tap_row_bytes;
    seam->speculator = (SparkSpeculationSpeculator *)calloc(
        1u,
        sizeof(*seam->speculator));
    seam->sequence_states = (SparkSpeculationSequenceState *)calloc(
        configuration->lane_count,
        sizeof(*seam->sequence_states));
    seam->lanes = (SparkSpeculationSeamLane *)calloc(
        configuration->lane_count,
        sizeof(*seam->lanes));
    if (seam->speculator == 0 ||
        seam->sequence_states == 0 ||
        seam->lanes == 0)
    {
        SparkSpeculationSeamDestroy(seam);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((enabled_source_mask & SPARK_SPECULATION_SEAM_REMOTE_SOURCES) != 0u)
    {
        seam->nodes = (SparkDraftBridgeNode *)calloc(
            configuration->draft_max_node_count,
            sizeof(*seam->nodes));
        if (seam->nodes == 0)
        {
            SparkSpeculationSeamDestroy(seam);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }
    memset(&policy_configuration, 0, sizeof(policy_configuration));
    policy_configuration.abi_version = SPARK_SPECULATION_ABI_VERSION;
    policy_configuration.descriptor_bytes =
        SPARK_SPECULATION_CONFIGURATION_DESCRIPTOR_BYTES;
    policy_configuration.policy_flags = 0u;
    policy_configuration.sequence_state_count = configuration->lane_count;
    policy_configuration.default_speculative_token_count =
        configuration->default_speculative_token_count;
    policy_configuration.minimum_confidence_milli = 1u;
    policy_configuration.realtime_minimum_confidence_milli = 1u;
    policy_configuration.sequence_states = seam->sequence_states;
    policy_configuration.draft_function = SparkSpeculationSeamDraftCallback;
    policy_configuration.draft_context = seam;
    policy_configuration.model_contract = &configuration->model_contract;
    status = SparkSpeculationPolicyInitialize(
        seam->speculator,
        &policy_configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkSpeculationSeamDestroy(seam);
        return status;
    }
    for (lane_index = 0u; lane_index < configuration->lane_count; ++lane_index)
    {
        seam->sequence_states[lane_index].sequence_id =
            SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID;
        seam->lanes[lane_index].stash_sequence_id =
            SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID;
    }
    if ((enabled_source_mask & SPARK_SPECULATION_SEAM_REMOTE_SOURCES) != 0u)
    {
        memset(&bridge_configuration, 0, sizeof(bridge_configuration));
        bridge_configuration.abi_version = SPARK_DRAFT_BRIDGE_ABI_VERSION;
        bridge_configuration.descriptor_bytes =
            SPARK_DRAFT_BRIDGE_CONFIG_BYTES;
        bridge_configuration.host = configuration->bridge_host;
        bridge_configuration.port = configuration->bridge_port;
        memcpy(
            bridge_configuration.target_model,
            configuration->target_model,
            SPARK_SPECULATION_SEAM_TARGET_MODEL_BYTES);
        bridge_configuration.tap_row_bytes = tap_row_bytes;
        bridge_configuration.max_committed_tokens =
            configuration->max_committed_token_count;
        bridge_configuration.max_tap_rows = configuration->max_tap_row_count;
        bridge_configuration.max_nodes = configuration->draft_max_node_count;
        bridge_configuration.connect_timeout_ms =
            configuration->connect_timeout_ms;
        bridge_configuration.io_timeout_ms = configuration->io_timeout_ms;
        status = SparkDraftBridgeInitialize(
            &bridge_configuration,
            &seam->bridge);
        if (status != SPARK_STATUS_OK)
        {
            SparkSpeculationSeamDestroy(seam);
            return status;
        }
    }
    *seam_out = seam;
    return SPARK_STATUS_OK;
}

void SparkSpeculationSeamDestroy(
    SparkSpeculationSeam *seam)
{
    if (seam == 0)
    {
        return;
    }
    if (seam->bridge != 0)
    {
        SparkDraftBridgeDestroy(seam->bridge);
    }
    free(seam->nodes);
    free(seam->lanes);
    free(seam->sequence_states);
    free(seam->speculator);
    free(seam);
}

uint32_t SparkSpeculationSeamEnabledSources(
    const SparkSpeculationSeam *seam)
{
    if (seam == 0)
    {
        return 0u;
    }
    return seam->enabled_source_mask;
}

SparkSpeculationSpeculator *SparkSpeculationSeamSpeculator(
    SparkSpeculationSeam *seam)
{
    if (seam == 0)
    {
        return 0;
    }
    return seam->speculator;
}

static uint32_t SparkSpeculationSeamFindLane(
    const SparkSpeculationSeam *seam,
    uint64_t sequence_id)
{
    uint32_t lane_index;

    if (sequence_id == SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID)
    {
        return SPARK_SPECULATION_SEAM_NO_SELECTION;
    }
    for (lane_index = 0u; lane_index < seam->lane_count; ++lane_index)
    {
        if (seam->sequence_states[lane_index].sequence_id == sequence_id)
        {
            return lane_index;
        }
    }
    return SPARK_SPECULATION_SEAM_NO_SELECTION;
}

SparkStatus SparkSpeculationSeamDraftRemoteChain(
    SparkSpeculationSeam *seam,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t position,
    uint32_t anchor_token_id,
    const uint32_t *committed_token_ids,
    uint32_t committed_token_count,
    const void *tap_rows,
    uint32_t tap_row_count,
    uint32_t requested_token_count,
    uint32_t *draft_token_ids_out,
    uint32_t draft_token_id_capacity,
    uint32_t *draft_token_count_out)
{
    SparkSpeculationPolicyDraftRequest draft_request;
    SparkSpeculationPolicyDraftResult draft_result;
    SparkSpeculationSequenceState *sequence_state;
    SparkSpeculationSeamLane *lane;
    uint64_t tap_generation;
    uint32_t lane_index;
    uint32_t lane_claimed;
    SparkStatus status;

    if (seam == 0 ||
        committed_token_ids == 0 ||
        draft_token_ids_out == 0 ||
        draft_token_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (sequence_id == 0u ||
        sequence_id == SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID ||
        committed_token_count == 0u ||
        (uint64_t)committed_token_count != position + 1u ||
        committed_token_ids[position] != anchor_token_id ||
        requested_token_count == 0u ||
        requested_token_count >
            seam->speculator->model_contract.maximum_speculative_token_count ||
        draft_token_id_capacity < requested_token_count ||
        committed_token_count > seam->max_committed_token_count ||
        tap_row_count > seam->max_tap_row_count ||
        (tap_row_count != 0u && tap_rows == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((seam->enabled_source_mask &
            SPARK_SPECULATION_SEAM_REMOTE_SOURCES) == 0u)
    {
        return SPARK_STATUS_UNSUPPORTED;
    }

    lane_index = SparkSpeculationSeamFindLane(seam, sequence_id);
    lane_claimed = 0u;
    if (lane_index == SPARK_SPECULATION_SEAM_NO_SELECTION)
    {
        for (lane_index = 0u; lane_index < seam->lane_count; ++lane_index)
        {
            if (seam->sequence_states[lane_index].sequence_id ==
                SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID)
            {
                break;
            }
        }
        if (lane_index == seam->lane_count)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        seam->sequence_states[lane_index].sequence_id = sequence_id;
        lane_claimed = 1u;
    }
    sequence_state = &seam->sequence_states[lane_index];
    if ((sequence_state->flags &
            SPARK_SPECULATION_SEQUENCE_STATE_FLAG_TAPS_READY) == 0u)
    {
        status = SparkSpeculationPolicyMarkVerifierTapsReady(
            seam->speculator,
            request_id,
            sequence_id,
            position,
            &tap_generation);
        if (status != SPARK_STATUS_OK)
        {
            if (lane_claimed != 0u)
            {
                sequence_state->sequence_id =
                    SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID;
            }
            return status;
        }
    }
    else
    {
        tap_generation = sequence_state->tap_generation;
    }

    lane = &seam->lanes[lane_index];
    lane->stash_sequence_id = sequence_id;
    lane->stash_committed_token_ids = committed_token_ids;
    lane->stash_committed_token_count = committed_token_count;
    lane->stash_tap_rows = tap_rows;
    lane->stash_tap_row_count = tap_row_count;

    memset(&draft_request, 0, sizeof(draft_request));
    draft_request.abi_version = SPARK_SPECULATION_ABI_VERSION;
    draft_request.descriptor_bytes =
        SPARK_SPECULATION_DRAFT_REQUEST_DESCRIPTOR_BYTES;
    draft_request.requested_token_count = requested_token_count;
    draft_request.active_sequence_index = lane_index;
    draft_request.request_id = request_id;
    draft_request.sequence_id = sequence_id;
    draft_request.sequence_position = position;
    draft_request.tap_generation = tap_generation;
    status = SparkSpeculationPolicyEnsureDraft(
        seam->speculator,
        &draft_request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSpeculationPolicyGetDraft(
        seam->speculator,
        sequence_id,
        &draft_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (draft_result.token_count > draft_token_id_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(
        draft_token_ids_out,
        draft_result.token_ids,
        (size_t)draft_result.token_count * sizeof(uint32_t));
    *draft_token_count_out = draft_result.token_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkSpeculationSeamAcceptChain(
    SparkSpeculationSeam *seam,
    uint64_t sequence_id,
    const uint32_t *verifier_token_ids,
    uint32_t verifier_token_count,
    SparkSpeculationPolicyVerifyResult *verify_result_out)
{
    SparkSpeculationPolicyDraftResult draft_result;
    SparkSpeculationPolicyVerifyResult verify_result;
    uint32_t lane_index;
    SparkStatus status;

    if (seam == 0 ||
        verifier_token_ids == 0 ||
        verifier_token_count == 0u ||
        verify_result_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    lane_index = SparkSpeculationSeamFindLane(seam, sequence_id);
    if (lane_index == SPARK_SPECULATION_SEAM_NO_SELECTION)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkSpeculationPolicyGetDraft(
        seam->speculator,
        sequence_id,
        &draft_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSpeculationPolicyResolveVerifierTokens(
        draft_result.token_ids,
        draft_result.token_count,
        verifier_token_ids,
        verifier_token_count,
        seam->speculator->model_contract.vocab_size,
        &verify_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkSpeculationPolicyCompleteVerify(
        seam->speculator,
        sequence_id,
        &verify_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (verify_result.accepted_draft_token_count <
        verify_result.proposed_token_count)
    {
        seam->lanes[lane_index].remote_generation += 1u;
    }
    *verify_result_out = verify_result;
    return SPARK_STATUS_OK;
}

SparkStatus SparkSpeculationSeamCancelSequence(
    SparkSpeculationSeam *seam,
    uint64_t sequence_id)
{
    uint32_t lane_index;
    SparkStatus status;

    if (seam == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    lane_index = SparkSpeculationSeamFindLane(seam, sequence_id);
    if (lane_index == SPARK_SPECULATION_SEAM_NO_SELECTION)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkSpeculationPolicyCancelSequence(
        seam->speculator,
        sequence_id);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    seam->lanes[lane_index].remote_generation += 1u;
    seam->lanes[lane_index].stash_sequence_id =
        SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID;
    seam->sequence_states[lane_index].sequence_id =
        SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID;
    return SPARK_STATUS_OK;
}

SparkStatus SparkSpeculationSeamStageLocalDraft(
    SparkSpeculationSeam *seam,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t position,
    const uint32_t *draft_token_ids,
    uint32_t draft_token_count)
{
    SparkSpeculationSpeculator *speculator;
    SparkSpeculationSequenceState *sequence_state;
    uint32_t lane_index;
    uint32_t token_index;

    if (seam == 0 || draft_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    speculator = seam->speculator;
    if (sequence_id == 0u ||
        sequence_id == SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID ||
        draft_token_count == 0u ||
        draft_token_count >
            speculator->model_contract.maximum_speculative_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (token_index = 0u; token_index < draft_token_count; ++token_index)
    {
        if (draft_token_ids[token_index] >=
            speculator->model_contract.vocab_size)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    lane_index = SparkSpeculationSeamFindLane(seam, sequence_id);
    if (lane_index == SPARK_SPECULATION_SEAM_NO_SELECTION)
    {
        for (lane_index = 0u; lane_index < seam->lane_count; ++lane_index)
        {
            if (seam->sequence_states[lane_index].sequence_id ==
                SPARK_SPECULATION_SEAM_EMPTY_SEQUENCE_ID)
            {
                break;
            }
        }
        if (lane_index == seam->lane_count)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }
    sequence_state = &seam->sequence_states[lane_index];
    if ((sequence_state->flags &
            SPARK_SPECULATION_SEQUENCE_STATE_FLAG_DRAFT_READY) != 0u)
    {
        return SPARK_STATUS_BUSY;
    }

    sequence_state->flags =
        SPARK_SPECULATION_SEQUENCE_STATE_FLAG_VALID |
        SPARK_SPECULATION_SEQUENCE_STATE_FLAG_DRAFT_READY;
    sequence_state->draft_token_count = draft_token_count;
    sequence_state->request_id = request_id;
    sequence_state->sequence_id = sequence_id;
    sequence_state->sequence_position = position;
    sequence_state->draft_generation = speculator->next_draft_generation;
    speculator->next_draft_generation += 1u;
    if (speculator->next_draft_generation == 0u)
    {
        speculator->next_draft_generation = 1u;
    }
    for (token_index = 0u;
         token_index < SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT;
         ++token_index)
    {
        if (token_index < draft_token_count)
        {
            sequence_state->draft_token_ids[token_index] =
                draft_token_ids[token_index];
        }
        else
        {
            sequence_state->draft_token_ids[token_index] = 0u;
        }
        sequence_state->draft_confidence_milli[token_index] = 0u;
    }
    return SPARK_STATUS_OK;
}
