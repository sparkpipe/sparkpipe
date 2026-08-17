#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This header is the GLM-5.2 DSpark ABI surface. The drafter constants are
 * neutral now: selecting the GLM52 target resolves SPARK_DSPARK_* from
 * spark_dspark_drafter.h, and the legacy SPARK_GLM52_DSPARK_* names remain
 * as compatibility aliases for this module and its consumers. */
#define SPARK_DSPARK_TARGET_GLM52 1
#include "sparkpipe/spark_dspark_drafter.h"

#define SPARK_GLM52_DSPARK_ABI_VERSION SPARK_DSPARK_ABI_VERSION
#define SPARK_GLM52_DSPARK_AUX_LAYER_COUNT SPARK_DSPARK_AUX_LAYER_COUNT
#define SPARK_GLM52_DSPARK_AUX_LAYER_IDS_INITIALIZER SPARK_DSPARK_AUX_LAYER_IDS_INITIALIZER
#define SPARK_GLM52_DSPARK_DRAFT_LAYER_COUNT SPARK_DSPARK_DRAFT_LAYER_COUNT
#define SPARK_GLM52_DSPARK_BLOCK_SIZE SPARK_DSPARK_BLOCK_SIZE
#define SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
#define SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE SPARK_DSPARK_FULL_VOCAB_SIZE
#define SPARK_GLM52_DSPARK_HIDDEN_DIMENSION SPARK_DSPARK_HIDDEN_DIMENSION
#define SPARK_GLM52_DSPARK_MARKOV_RANK SPARK_DSPARK_MARKOV_RANK
#define SPARK_GLM52_DSPARK_DRAFT_INTERMEDIATE_DIMENSION SPARK_DSPARK_DRAFT_INTERMEDIATE_DIMENSION
#define SPARK_GLM52_DSPARK_DRAFT_ATTENTION_HEAD_COUNT SPARK_DSPARK_DRAFT_ATTENTION_HEAD_COUNT
#define SPARK_GLM52_DSPARK_DRAFT_KV_HEAD_COUNT SPARK_DSPARK_DRAFT_KV_HEAD_COUNT
#define SPARK_GLM52_DSPARK_DRAFT_HEAD_DIMENSION SPARK_DSPARK_DRAFT_HEAD_DIMENSION
#define SPARK_GLM52_DSPARK_MAX_ANCHORS SPARK_DSPARK_MAX_ANCHORS
#define SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE SPARK_DSPARK_CONFIDENCE_MILLI_ONE
#define SPARK_GLM52_DSPARK_DEFAULT_MIN_CONFIDENCE_MILLI SPARK_DSPARK_DEFAULT_MIN_CONFIDENCE_MILLI
#define SPARK_GLM52_DSPARK_DEFAULT_REALTIME_MIN_CONFIDENCE_MILLI SPARK_DSPARK_DEFAULT_REALTIME_MIN_CONFIDENCE_MILLI
#define SPARK_GLM52_DSPARK_DEFAULT_MAX_VERIFY_TOKENS SPARK_DSPARK_DEFAULT_MAX_VERIFY_TOKENS
#define SPARK_GLM52_DSPARK_INVALID_LAYER_INDEX SPARK_DSPARK_INVALID_LAYER_INDEX
#define SPARK_GLM52_DSPARK_VERIFIER_HIDDEN_DTYPE_BF16 SPARK_DSPARK_VERIFIER_HIDDEN_DTYPE_BF16
#define SPARK_GLM52_DSPARK_DRAFT_DTYPE_BF16 SPARK_DSPARK_DRAFT_DTYPE_BF16
#define SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_REALTIME SPARK_DSPARK_POLICY_FLAG_ENABLE_REALTIME
#define SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE SPARK_DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE
#define SPARK_GLM52_DSPARK_POLICY_FLAG_REQUIRE_CONFIDENCE_HEAD SPARK_DSPARK_POLICY_FLAG_REQUIRE_CONFIDENCE_HEAD
#define SPARK_GLM52_DSPARK_POLICY_FLAG_REQUIRE_MARKOV_HEAD SPARK_DSPARK_POLICY_FLAG_REQUIRE_MARKOV_HEAD
#define SPARK_GLM52_DSPARK_POLICY_DEFAULT_FLAGS SPARK_DSPARK_POLICY_DEFAULT_FLAGS
#define SPARK_GLM52_DSPARK_POLICY_KNOWN_FLAGS SPARK_DSPARK_POLICY_KNOWN_FLAGS
#define SPARK_GLM52_DSPARK_SEQUENCE_STATE_FLAG_VALID SPARK_DSPARK_SEQUENCE_STATE_FLAG_VALID
#define SPARK_GLM52_DSPARK_SEQUENCE_STATE_FLAG_TAPS_READY SPARK_DSPARK_SEQUENCE_STATE_FLAG_TAPS_READY
#define SPARK_GLM52_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY SPARK_DSPARK_SEQUENCE_STATE_FLAG_DRAFT_READY
#define SPARK_GLM52_DSPARK_SEQUENCE_STATE_FLAG_VERIFY_INFLIGHT SPARK_DSPARK_SEQUENCE_STATE_FLAG_VERIFY_INFLIGHT
#define SPARK_GLM52_DSPARK_DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED SPARK_DSPARK_DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED
#define SPARK_GLM52_DSPARK_DRAFT_RESULT_KNOWN_FLAGS SPARK_DSPARK_DRAFT_RESULT_KNOWN_FLAGS
#define SPARK_GLM52_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL SPARK_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL
#define SPARK_GLM52_DSPARK_VERIFY_RESULT_FLAG_REJECTED SPARK_DSPARK_VERIFY_RESULT_FLAG_REJECTED
#define SPARK_GLM52_DSPARK_VERIFY_RESULT_KNOWN_FLAGS SPARK_DSPARK_VERIFY_RESULT_KNOWN_FLAGS

/* Descriptor bytes stay here: the struct layouts below are the GLM52 ABI
 * surface until the structs themselves are neutralized. */
#define SPARK_GLM52_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkSpeculatorConfiguration))
#define SPARK_GLM52_DSPARK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkSpeculator))
#define SPARK_GLM52_DSPARK_SEQUENCE_STATE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkSequenceState))
#define SPARK_GLM52_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkHiddenTapPlan))
#define SPARK_GLM52_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkModelContract))
#define SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkDraftRequest))
#define SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkDraftResult))
#define SPARK_GLM52_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52DsparkVerifyResult))

#define SPARK_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES
#define SPARK_DSPARK_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_DESCRIPTOR_BYTES
#define SPARK_DSPARK_SEQUENCE_STATE_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_SEQUENCE_STATE_DESCRIPTOR_BYTES
#define SPARK_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_HIDDEN_TAP_PLAN_DESCRIPTOR_BYTES
#define SPARK_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES
#define SPARK_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES
#define SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES
#define SPARK_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES SPARK_GLM52_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES

typedef struct SparkGlm52DsparkTapStage
{
    uint32_t target_layer_index;
    uint32_t stage_index;
    uint32_t stage_first_layer_index;
    uint32_t stage_layer_count;
    uint32_t layer_offset_in_stage;
    uint32_t reserved;
} SparkGlm52DsparkTapStage;

typedef struct SparkGlm52DsparkHiddenTapPlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t aux_layer_count;
    uint32_t hidden_dimension;
    uint32_t pp_stage_count;
    uint32_t pp_stage_layer_count;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkGlm52DsparkTapStage tap_stages[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT];
} SparkGlm52DsparkHiddenTapPlan;

typedef struct SparkGlm52DsparkModelContract
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
    uint32_t aux_layer_ids[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT];
} SparkGlm52DsparkModelContract;

typedef struct SparkGlm52DsparkDraftRequest
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
} SparkGlm52DsparkDraftRequest;

typedef struct SparkGlm52DsparkDraftResult
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t token_count;
	uint32_t confidence_milli[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
	uint32_t token_ids[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkGlm52DsparkDraftResult;

typedef struct SparkGlm52DsparkVerifyResult
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t proposed_token_count;
	uint32_t accepted_draft_token_count;
	uint32_t committed_token_count;
	uint32_t fallback_token_id;
	uint32_t reserved;
} SparkGlm52DsparkVerifyResult;

typedef SparkStatus (*SparkGlm52DsparkDraftFunction)(
    void *context,
    const SparkGlm52DsparkDraftRequest *request,
    SparkGlm52DsparkDraftResult *result);

typedef struct SparkGlm52DsparkSequenceState
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
    uint32_t draft_token_ids[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t draft_confidence_milli[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkGlm52DsparkSequenceState;

typedef struct SparkGlm52DsparkSpeculatorConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t policy_flags;
    uint32_t sequence_state_count;
    uint32_t default_speculative_token_count;
    uint32_t minimum_confidence_milli;
    uint32_t realtime_minimum_confidence_milli;
    uint32_t reserved;
    SparkGlm52DsparkSequenceState *sequence_states;
    SparkGlm52DsparkDraftFunction draft_function;
    void *draft_context;
    const SparkGlm52DsparkModelContract *model_contract;
} SparkGlm52DsparkSpeculatorConfiguration;

typedef struct SparkGlm52DsparkSpeculator
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
    SparkGlm52DsparkModelContract model_contract;
    SparkGlm52DsparkSequenceState *sequence_states;
    SparkGlm52DsparkDraftFunction draft_function;
    void *draft_context;
} SparkGlm52DsparkSpeculator;

SparkStatus SparkGlm52DsparkBuildDefaultHiddenTapPlan(
    SparkGlm52DsparkHiddenTapPlan *tap_plan);

SparkStatus SparkGlm52DsparkValidateHiddenTapPlan(
    const SparkGlm52DsparkHiddenTapPlan *tap_plan);

SparkStatus SparkGlm52DsparkBuildDefaultModelContract(
    SparkGlm52DsparkModelContract *model_contract);

SparkStatus SparkGlm52DsparkValidateModelContract(
    const SparkGlm52DsparkModelContract *model_contract);

SparkStatus SparkGlm52DsparkInitialize(
    SparkGlm52DsparkSpeculator *speculator,
    const SparkGlm52DsparkSpeculatorConfiguration *configuration);

SparkStatus SparkGlm52DsparkValidate(
    const SparkGlm52DsparkSpeculator *speculator);

SparkStatus SparkGlm52DsparkMarkVerifierTapsReady(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t request_id,
    uint64_t sequence_id,
    uint64_t sequence_position,
    uint64_t *tap_generation_out);

SparkStatus SparkGlm52DsparkEnsureDraft(
    SparkGlm52DsparkSpeculator *speculator,
    const SparkGlm52DsparkDraftRequest *request);

SparkStatus SparkGlm52DsparkGetDraft(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id,
    SparkGlm52DsparkDraftResult *draft_result);

SparkStatus SparkGlm52DsparkCompleteVerify(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id,
    const SparkGlm52DsparkVerifyResult *verify_result);

SparkStatus SparkGlm52DsparkResolveVerifierTokens(
    const uint32_t *draft_token_ids,
    uint32_t draft_token_count,
    const uint32_t *verifier_token_ids,
    uint32_t verifier_token_count,
    SparkGlm52DsparkVerifyResult *verify_result);

SparkStatus SparkGlm52DsparkCancelSequence(
    SparkGlm52DsparkSpeculator *speculator,
    uint64_t sequence_id);

#ifdef __cplusplus
}
#endif
