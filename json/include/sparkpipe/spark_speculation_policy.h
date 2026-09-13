#pragma once

/* Model-neutral speculation dispatch policy (speculation audit step 3).
 *
 * This header OWNS the dispatch-policy descriptor struct layouts under neutral
 * names and the neutral-core function prototypes. The struct layouts are
 * byte-identical to the first adopting ABI's structs (same fields, same
 * order, same array sizes); only the type names move. Per-model shapes
 * (array sizes, tap sites) resolve through spark_dspark_drafter.h, which the
 * package selects with exactly one SPARK_DSPARK_TARGET_* define.
 *
 * The first adopting model's ABI header keeps typedef aliases (see the alias
 * block at the bottom) so existing consumers compile unchanged. */

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_dspark_drafter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Descriptor-byte macros, now over the neutral layouts. These are the
 * SPARK_DSPARK_* names the policy source already uses; the adopting model's
 * ABI header keeps its legacy *_DESCRIPTOR_BYTES as aliases to these. The
 * hidden tap plan stays model-specific and keeps its macro in the family
 * header. */
#define SPARK_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationConfiguration))
#define SPARK_DSPARK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationSpeculator))
#define SPARK_DSPARK_SEQUENCE_STATE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationSequenceState))
#define SPARK_DSPARK_MODEL_CONTRACT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationModelContract))
#define SPARK_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationDraftRequest))
#define SPARK_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationDraftResult))
#define SPARK_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSpeculationVerifyResult))

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
    uint32_t aux_layer_ids[SPARK_DSPARK_AUX_LAYER_COUNT];
} SparkSpeculationModelContract;

typedef struct SparkSpeculationDraftRequest
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
} SparkSpeculationDraftRequest;

typedef struct SparkSpeculationDraftResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t token_count;
    uint32_t confidence_milli[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t token_ids[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkSpeculationDraftResult;

typedef struct SparkSpeculationVerifyResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t proposed_token_count;
    uint32_t accepted_draft_token_count;
    uint32_t committed_token_count;
    uint32_t fallback_token_id;
    uint32_t reserved;
} SparkSpeculationVerifyResult;

typedef SparkStatus (*SparkSpeculationDraftFunction)(
    void *context,
    const SparkSpeculationDraftRequest *request,
    SparkSpeculationDraftResult *result);

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
    uint32_t draft_token_ids[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t draft_confidence_milli[SPARK_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
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

/* Neutral-core function prototypes (the 10 public entry points; the 13
 * internal helpers stay static in spark_speculation_policy.c). */
SparkStatus SparkSpeculationPolicyBuildDefaultModelContract(
    SparkSpeculationModelContract *model_contract);

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
    const SparkSpeculationDraftRequest *request);

SparkStatus SparkSpeculationPolicyGetDraft(
    SparkSpeculationSpeculator *speculator,
    uint64_t sequence_id,
    SparkSpeculationDraftResult *draft_result);

SparkStatus SparkSpeculationPolicyCompleteVerify(
    SparkSpeculationSpeculator *speculator,
    uint64_t sequence_id,
    const SparkSpeculationVerifyResult *verify_result);

SparkStatus SparkSpeculationPolicyResolveVerifierTokens(
    const uint32_t *draft_token_ids,
    uint32_t draft_token_count,
    const uint32_t *verifier_token_ids,
    uint32_t verifier_token_count,
    SparkSpeculationVerifyResult *verify_result);

SparkStatus SparkSpeculationPolicyCancelSequence(
    SparkSpeculationSpeculator *speculator,
    uint64_t sequence_id);

#ifdef __cplusplus
}
#endif
