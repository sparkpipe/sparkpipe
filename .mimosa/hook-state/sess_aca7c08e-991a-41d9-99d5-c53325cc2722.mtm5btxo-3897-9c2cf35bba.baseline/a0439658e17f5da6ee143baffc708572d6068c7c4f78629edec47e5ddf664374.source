#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SPECULATION_ABI_VERSION 3u
#define SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT 32u
#define SPARK_SPECULATION_MAX_AUX_LAYER_COUNT 8u
#define SPARK_SPECULATION_PARENT_INDEX_ROOT 0xffffffffu
#define SPARK_SPECULATION_INVALID_LAYER_INDEX 0xffffffffu
#define SPARK_SPECULATION_CONFIDENCE_MILLI_ONE 1000u
#define SPARK_SPECULATION_DEFAULT_MIN_CONFIDENCE_MILLI 350u
#define SPARK_SPECULATION_DEFAULT_REALTIME_MIN_CONFIDENCE_MILLI 250u
#define SPARK_SPECULATION_POLICY_REALTIME_PRIORITY_THRESHOLD 4000000000u
#define SPARK_SPECULATION_VERIFIER_HIDDEN_DTYPE_BF16 1u
#define SPARK_SPECULATION_DRAFT_DTYPE_BF16 1u

#define SPARK_SPECULATION_POLICY_FLAG_ENABLE_REALTIME 0x00000001u
#define SPARK_SPECULATION_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE 0x00000002u
#define SPARK_SPECULATION_POLICY_FLAG_REQUIRE_CONFIDENCE_HEAD 0x00000004u
#define SPARK_SPECULATION_POLICY_FLAG_REQUIRE_MARKOV_HEAD 0x00000008u
#define SPARK_SPECULATION_POLICY_DEFAULT_FLAGS \
    (SPARK_SPECULATION_POLICY_FLAG_ENABLE_REALTIME | \
     SPARK_SPECULATION_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE | \
     SPARK_SPECULATION_POLICY_FLAG_REQUIRE_CONFIDENCE_HEAD | \
     SPARK_SPECULATION_POLICY_FLAG_REQUIRE_MARKOV_HEAD)
#define SPARK_SPECULATION_POLICY_KNOWN_FLAGS \
    SPARK_SPECULATION_POLICY_DEFAULT_FLAGS

#define SPARK_SPECULATION_SEQUENCE_STATE_FLAG_VALID 0x00000001u
#define SPARK_SPECULATION_SEQUENCE_STATE_FLAG_TAPS_READY 0x00000002u
#define SPARK_SPECULATION_SEQUENCE_STATE_FLAG_DRAFT_READY 0x00000004u
#define SPARK_SPECULATION_SEQUENCE_STATE_FLAG_VERIFY_INFLIGHT 0x00000008u

#define SPARK_SPECULATION_DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED 0x00000001u
#define SPARK_SPECULATION_DRAFT_RESULT_KNOWN_FLAGS \
    SPARK_SPECULATION_DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED

#define SPARK_SPECULATION_VERIFY_RESULT_FLAG_ACCEPTED_ALL UINT32_C(0x00000001)
#define SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED UINT32_C(0x00000002)
#define SPARK_SPECULATION_VERIFY_RESULT_KNOWN_FLAGS \
    (SPARK_SPECULATION_VERIFY_RESULT_FLAG_ACCEPTED_ALL | \
     SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED)

#define SPARK_SPECULATION_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationConfiguration))
#define SPARK_SPECULATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationSpeculator))
#define SPARK_SPECULATION_SEQUENCE_STATE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationSequenceState))
#define SPARK_SPECULATION_MODEL_CONTRACT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationModelContract))
#define SPARK_SPECULATION_DRAFT_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationPolicyDraftRequest))
#define SPARK_SPECULATION_DRAFT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationPolicyDraftResult))
#define SPARK_SPECULATION_VERIFY_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationPolicyVerifyResult))

typedef struct SparkSpeculationModelContract
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t verifier_hidden_dtype;
    uint32_t draft_dtype;
    uint32_t draft_layer_count;
    uint32_t block_size;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t attention_head_count;
    uint32_t kv_head_count;
    uint32_t head_dimension;
    uint32_t vocab_size;
    uint32_t draft_vocab_size;
    uint32_t markov_rank;
    uint32_t max_anchors;
    uint32_t maximum_speculative_token_count;
    uint32_t verifier_accept_k;
    uint32_t aux_layer_count;
    uint32_t enable_confidence_head;
    uint32_t confidence_head_with_markov;
    uint32_t aux_layer_ids[SPARK_SPECULATION_MAX_AUX_LAYER_COUNT];
} SparkSpeculationModelContract;

typedef struct SparkSpeculationPolicyDraftRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t requested_token_count;
    uint32_t active_sequence_index;
    uint32_t priority;
    uint32_t reserved;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t tap_generation;
} SparkSpeculationPolicyDraftRequest;

typedef struct SparkSpeculationPolicyDraftResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t token_count;
    uint32_t confidence_milli[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t token_ids[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkSpeculationPolicyDraftResult;

typedef struct SparkSpeculationPolicyVerifyResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t proposed_token_count;
    uint32_t accepted_draft_token_count;
    uint32_t committed_token_count;
    uint32_t fallback_token_id;
    uint32_t reserved;
} SparkSpeculationPolicyVerifyResult;

typedef SparkStatus (*SparkSpeculationDraftFunction)(
    void *context,
    const SparkSpeculationPolicyDraftRequest *request,
    SparkSpeculationPolicyDraftResult *result);

typedef struct SparkSpeculationSequenceState
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t draft_token_count;
    uint32_t accepted_draft_token_count;
    uint32_t committed_token_count;
    uint32_t rejected_token_count;
    uint32_t reserved;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t tap_generation;
    uint64_t draft_generation;
    uint32_t draft_token_ids[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t draft_confidence_milli[SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkSpeculationSequenceState;

typedef struct SparkSpeculationConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t policy_flags;
    uint32_t sequence_state_count;
    uint32_t default_speculative_token_count;
    uint32_t minimum_confidence_milli;
    uint32_t realtime_minimum_confidence_milli;
    uint32_t reserved;
    SparkSpeculationSequenceState *sequence_states;
    SparkSpeculationDraftFunction draft_function;
    void *draft_context;
    const SparkSpeculationModelContract *model_contract;
} SparkSpeculationConfiguration;

typedef struct SparkSpeculationSpeculator
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t policy_flags;
    uint32_t sequence_state_count;
    uint32_t default_speculative_token_count;
    uint32_t minimum_confidence_milli;
    uint32_t realtime_minimum_confidence_milli;
    uint32_t reserved;
    uint64_t next_tap_generation;
    uint64_t next_draft_generation;
    uint64_t tap_ready_count;
    uint64_t draft_request_count;
    uint64_t draft_success_count;
    uint64_t draft_rejected_by_confidence_count;
    uint64_t verify_dispatch_count;
    uint64_t accepted_draft_token_count;
    uint64_t committed_token_count;
    uint64_t rejected_token_count;
    SparkSpeculationModelContract model_contract;
    SparkSpeculationSequenceState *sequence_states;
    SparkSpeculationDraftFunction draft_function;
    void *draft_context;
} SparkSpeculationSpeculator;

SparkStatus SparkSpeculationPolicyValidateModelContract(
    const SparkSpeculationModelContract *model_contract);

SparkStatus SparkSpeculationPolicyInitialize(
    SparkSpeculationSpeculator *speculator,
    const SparkSpeculationConfiguration *configuration);

SparkStatus SparkSpeculationPolicyValidate(
    const SparkSpeculationSpeculator *speculator);

SparkStatus SparkSpeculationPolicyMarkVerifierTapsReady(
    SparkSpeculationSpeculator *speculator,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint64_t *tap_generation_out);

SparkStatus SparkSpeculationPolicyEnsureDraft(
    SparkSpeculationSpeculator *speculator,
    const SparkSpeculationPolicyDraftRequest *request);

SparkStatus SparkSpeculationPolicyGetDraft(
    SparkSpeculationSpeculator *speculator,
    uint64_t sequence_id,
    SparkSpeculationPolicyDraftResult *draft_result);

SparkStatus SparkSpeculationPolicyCompleteVerify(
    SparkSpeculationSpeculator *speculator,
    uint64_t sequence_id,
    const SparkSpeculationPolicyVerifyResult *verify_result);

SparkStatus SparkSpeculationPolicyResolveVerifierTree(
    const uint32_t *draft_token_ids,
    const uint32_t *draft_parent_indices,
    uint32_t draft_token_count,
    const uint32_t *verifier_token_ids,
    uint32_t verifier_token_count,
    uint32_t vocab_size,
    SparkSpeculationPolicyVerifyResult *verify_result);

SparkStatus SparkSpeculationPolicyResolveVerifierTokens(
    const uint32_t *draft_token_ids,
    uint32_t draft_token_count,
    const uint32_t *verifier_token_ids,
    uint32_t verifier_token_count,
    uint32_t vocab_size,
    SparkSpeculationPolicyVerifyResult *verify_result);

SparkStatus SparkSpeculationPolicyCancelSequence(
    SparkSpeculationSpeculator *speculator,
    uint64_t sequence_id);

#ifdef __cplusplus
}
#endif
