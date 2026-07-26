#ifndef SPARKPIPE_SPARK_GLM52_SERVING_ENGINE_H
#define SPARKPIPE_SPARK_GLM52_SERVING_ENGINE_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_prompt_pipeline.h"
#include "sparkpipe/spark_glm52_request_api.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SERVING_ENGINE_ABI_VERSION 4u
#define SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingEngineConfiguration))
#define SPARK_GLM52_SERVING_ENGINE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingEngine))
#define SPARK_GLM52_SERVING_SUBMIT_TEXT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingSubmitTextRequest))
#define SPARK_GLM52_SERVING_SUBMIT_TOKENS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingSubmitTokenIdsRequest))
#define SPARK_GLM52_SERVING_SUBMIT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingSubmitResult))
#define SPARK_GLM52_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingDecodeDispatch))
#define SPARK_GLM52_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingDecodeResult))
#define SPARK_GLM52_SERVING_REQUEST_RECORD_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingRequestRecord))
#define SPARK_GLM52_SERVING_EVENT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingEvent))
#define SPARK_GLM52_SERVING_STATS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ServingStats))

#define SPARK_GLM52_SERVING_MAX_STOP_TOKEN_IDS 8u
#define SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE \
    (SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u)
#define SPARK_GLM52_SERVING_DEFAULT_OUTPUT_TOKEN_BUDGET 1024u
#define SPARK_GLM52_SERVING_DEFAULT_MAX_CONTEXT_TOKENS \
    SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS
#define SPARK_GLM52_SERVING_DEFAULT_MAX_PUMP_STEPS 256u
#define SPARK_GLM52_SERVING_DEFAULT_REQUEST_ID_BASE 1000000000ull

#define SPARK_GLM52_SERVING_ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT \
    0x00000001u
#define SPARK_GLM52_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS \
    0x00000002u
#define SPARK_GLM52_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT \
    0x00000004u
#define SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE \
    0x00000008u
#define SPARK_GLM52_SERVING_ENGINE_DEFAULT_FLAGS \
    (SPARK_GLM52_SERVING_ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT | \
     SPARK_GLM52_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS | \
     SPARK_GLM52_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT)
#define SPARK_GLM52_SERVING_ENGINE_KNOWN_FLAGS \
    (SPARK_GLM52_SERVING_ENGINE_DEFAULT_FLAGS | \
     SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE)

#define SPARK_GLM52_SERVING_SUBMIT_FLAG_REALTIME \
    SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME
#define SPARK_GLM52_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION \
    SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION
#define SPARK_GLM52_SERVING_SUBMIT_KNOWN_FLAGS \
    (SPARK_GLM52_SERVING_SUBMIT_FLAG_REALTIME | \
     SPARK_GLM52_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION)

#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS \
    0x00000001u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV \
    0x00000002u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV \
    0x00000004u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS \
    0x00000008u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES \
    0x00000010u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED \
    0x00000020u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED \
    0x00000040u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY \
    0x00000080u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE \
    0x00000100u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION \
    0x00000200u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE \
    0x00000400u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE \
    0x00000800u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY \
    0x80000000u
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS \
    (SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE)
#define SPARK_GLM52_SERVING_RUNTIME_CONTRACT_CURRENT_IMPLEMENTED_FLAGS \
    (SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS | \
     SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES)

#define SPARK_GLM52_SERVING_EVENT_KIND_NONE 0u
#define SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_ACCEPTED 1u
#define SPARK_GLM52_SERVING_EVENT_KIND_PREFILL_PROGRESS 2u
#define SPARK_GLM52_SERVING_EVENT_KIND_TOKEN 3u
#define SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_COMPLETED 4u
#define SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_CANCELLED 5u
#define SPARK_GLM52_SERVING_EVENT_KIND_ERROR 6u
#define SPARK_GLM52_SERVING_EVENT_KIND_BACKPRESSURE 7u

#define SPARK_GLM52_SERVING_DECODE_RESULT_FLAG_FINISH_REQUEST 0x00000001u
#define SPARK_GLM52_SERVING_DECODE_RESULT_FLAG_TOKEN_STREAM_SUPPRESSED 0x00000002u

#define SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE 0u
#define SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_SUBMITTED 1u
#define SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_COMPLETED 2u
#define SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_CANCELLED 3u

#define SPARK_GLM52_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH 0x00000001u
#define SPARK_GLM52_SERVING_PUMP_KNOWN_FLAGS \
    SPARK_GLM52_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH
#define SPARK_GLM52_SERVING_RECORD_HASH_SLOTS 4096u
#define SPARK_GLM52_SERVING_NO_RECORD_SLOT UINT32_MAX

typedef uint64_t SparkGlm52ServingRequestHandle;

typedef struct SparkGlm52ServingRequestRecord
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    uint32_t prompt_token_count;
    uint32_t token_capacity;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t streamed_decode_token_count;
    uint32_t reserved0;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkGlm52ServingRequestHandle request_handle;
    uint32_t handle_hash_next;
    uint32_t free_record_next;
    uint32_t *token_ids;
} SparkGlm52ServingRequestRecord;

typedef struct SparkGlm52ServingEvent
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t flags;
    uint32_t status;
    uint32_t token_id;
    uint32_t token_index;
    uint32_t token_count;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t dispatch_kind;
    uint32_t dispatch_flags;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkGlm52ServingRequestHandle request_handle;
} SparkGlm52ServingEvent;

typedef struct SparkGlm52ServingSubmitTokenIdsRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t token_count;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint64_t request_id;
    uint64_t sequence_id;
    const uint32_t *token_ids;
} SparkGlm52ServingSubmitTokenIdsRequest;

typedef struct SparkGlm52ServingSubmitTextRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    uint64_t request_id;
    uint64_t sequence_id;
    const char *text;
    uint32_t text_bytes;
} SparkGlm52ServingSubmitTextRequest;

typedef struct SparkGlm52ServingSubmitResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t prompt_token_count;
    uint32_t required_token_capacity;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkGlm52ServingRequestHandle request_handle;
} SparkGlm52ServingSubmitResult;

typedef struct SparkGlm52ServingDecodeDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t dispatch_kind;
    uint32_t request_count;
    uint32_t active_sequence_count;
    uint32_t speculative_token_index;
    uint32_t reserved1;
    uint32_t reserved2;
    const SparkGlm52RequestApiDispatch *request_dispatch;
    const SparkGlm52KvBlockTableView *kv_block_table_view;
    const SparkGlm52RequestApiDecodeDispatchView *decode_view;
    uint32_t input_token_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t speculative_token_count;
    uint32_t speculative_draft_token_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkGlm52ServingDecodeDispatch;

typedef struct SparkGlm52ServingDecodeResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_count;
    uint32_t token_stride;
    uint32_t token_counts[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t lane_flags[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t token_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
        SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE];
    uint32_t draft_token_counts[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t draft_token_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
        SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT];
} SparkGlm52ServingDecodeResult;

typedef SparkStatus (*SparkGlm52ServingPrefillFunction)(
    void *context,
    const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch);

typedef SparkStatus (*SparkGlm52ServingDecodeFunction)(
    void *context,
    const SparkGlm52ServingDecodeDispatch *decode_dispatch,
    SparkGlm52ServingDecodeResult *decode_result);
typedef SparkStatus (*SparkGlm52ServingReleaseSequenceFunction)(
    void *context,
    uint64_t request_id,
    uint64_t sequence_id,
    uint32_t token_count);

typedef struct SparkGlm52ServingStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t live_request_count;
    uint32_t queued_request_count;
    uint32_t completed_request_count;
    uint32_t cancelled_request_count;
    uint32_t event_count;
    uint32_t event_capacity;
    uint32_t dropped_event_count;
    uint32_t last_status;
    uint32_t maximum_prefill_active_sequence_count;
    uint32_t maximum_prefill_lane_count;
    uint32_t maximum_decode_active_sequence_count;
    uint32_t maximum_decode_lane_count;
    uint64_t submitted_request_count;
    uint64_t accepted_request_count;
    uint64_t prefill_dispatch_count;
    uint64_t prefill_batch_dispatch_count;
    uint64_t prefill_token_count;
    uint64_t decode_dispatch_count;
    uint64_t decoded_token_count;
    uint64_t mtp_draft_token_count;
    uint64_t mtp_verify_dispatch_count;
    uint64_t mtp_draft_ready_count;
    uint64_t mtp_accepted_draft_token_count;
    uint64_t mtp_committed_token_count;
    uint64_t mtp_rejected_token_count;
    uint64_t completed_stream_count;
    uint64_t jit_prefetch_dispatch_count;
    uint64_t jit_prefetch_block_count;
    uint64_t async_jit_prefetch_start_count;
    uint64_t async_jit_prefetch_poll_count;
    uint64_t async_jit_prefetch_completion_count;
    uint64_t prefix_family_dispatch_count;
    uint64_t prefix_family_member_count;
    uint64_t prefix_family_saved_prompt_token_count;
} SparkGlm52ServingStats;

typedef struct SparkGlm52ServingEngineConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t runtime_contract_flags;
    uint32_t default_thinking_token_budget;
    uint32_t default_output_token_budget;
    uint32_t default_max_prefill_tokens_per_step;
    uint32_t max_context_tokens;
    uint64_t request_id_base;
    SparkGlm52RequestApi *request_api;
    const SparkTokenizer *tokenizer;
    SparkGlm52ServingRequestRecord *request_records;
    uint32_t request_record_capacity;
    uint32_t *request_token_storage;
    uint32_t request_token_stride;
    SparkGlm52ServingEvent *event_ring;
    uint32_t event_ring_capacity;
    uint32_t *host_prefill_token_ids;
    uint32_t host_prefill_token_stride;
    uint32_t host_prefill_lane_capacity;
    uint32_t *host_physical_block_indices;
    const uint32_t *execution_physical_block_indices;
    uint32_t kv_block_lane_stride;
    uint32_t kv_block_lane_capacity;
    uint32_t *lane_physical_block_counts;
    uint32_t lane_count_capacity;
    SparkGlm52ServingPrefillFunction prefill_function;
    SparkGlm52ServingDecodeFunction decode_function;
    SparkGlm52ServingReleaseSequenceFunction release_sequence_function;
    void *callback_context;
    const uint32_t *stop_token_ids;
    uint32_t stop_token_id_count;
    uint32_t reserved0;
} SparkGlm52ServingEngineConfiguration;

typedef struct SparkGlm52ServingEngine
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t runtime_contract_flags;
    uint32_t default_thinking_token_budget;
    uint32_t default_output_token_budget;
    uint32_t default_max_prefill_tokens_per_step;
    uint32_t max_context_tokens;
    uint64_t next_generated_request_id;
    SparkGlm52RequestApi *request_api;
    const SparkTokenizer *tokenizer;
    SparkTokenizerWorkspace tokenizer_workspace;
    SparkGlm52ServingRequestRecord *request_records;
    uint32_t request_record_capacity;
    uint32_t free_record_head;
    uint32_t *request_token_storage;
    uint32_t request_token_stride;
    SparkGlm52ServingEvent *event_ring;
    uint32_t event_ring_capacity;
    uint32_t event_read_index;
    uint32_t event_write_index;
    uint32_t event_count;
    uint32_t dropped_event_count;
    uint32_t *host_prefill_token_ids;
    uint32_t host_prefill_token_stride;
    uint32_t host_prefill_lane_capacity;
    uint32_t *host_physical_block_indices;
    const uint32_t *execution_physical_block_indices;
    uint32_t kv_block_lane_stride;
    uint32_t kv_block_lane_capacity;
    uint32_t *lane_physical_block_counts;
    uint32_t lane_count_capacity;
    SparkGlm52ServingPrefillFunction prefill_function;
    SparkGlm52ServingDecodeFunction decode_function;
    SparkGlm52ServingReleaseSequenceFunction release_sequence_function;
    void *callback_context;
    uint32_t stop_token_ids[SPARK_GLM52_SERVING_MAX_STOP_TOKEN_IDS];
    uint32_t stop_token_id_count;
    uint32_t request_handle_hash_heads[
        SPARK_GLM52_SERVING_RECORD_HASH_SLOTS];
    SparkGlm52ServingStats stats;
} SparkGlm52ServingEngine;

void SparkGlm52ServingInitializeSubmitTextRequest(
    SparkGlm52ServingSubmitTextRequest *request);

void SparkGlm52ServingInitializeSubmitTokenIdsRequest(
    SparkGlm52ServingSubmitTokenIdsRequest *request);

void SparkGlm52ServingInitializeDecodeResult(
    SparkGlm52ServingDecodeResult *decode_result,
    uint32_t lane_count,
    uint32_t token_stride);

SparkStatus SparkGlm52ServingEngineInitialize(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingEngineConfiguration *configuration);

// Releases resources the engine owns internally. The engine's request records,
// token storage, and other buffers remain caller-owned and are not touched; this
// frees only the persistent tokenizer workspace the engine allocates to keep the
// piece cache warm across requests. Safe to call on a zero-initialized engine and
// safe to call more than once.
void SparkGlm52ServingEngineDestroy(
    SparkGlm52ServingEngine *engine);

SparkStatus SparkGlm52ServingEngineSubmitTokenIds(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingSubmitTokenIdsRequest *request,
    SparkGlm52ServingSubmitResult *result);

SparkStatus SparkGlm52ServingEngineSubmitText(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingSubmitTextRequest *request,
    SparkGlm52ServingSubmitResult *result);

SparkStatus SparkGlm52ServingEnginePump(
    SparkGlm52ServingEngine *engine,
    uint32_t pump_flags,
    uint32_t max_dispatch_steps,
    SparkGlm52ServingStats *stats);

SparkStatus SparkGlm52ServingEngineCompletePrefillDispatch(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch);
SparkStatus SparkGlm52ServingEngineCompleteDecodeDispatch(
    SparkGlm52ServingEngine *engine,
    SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52ServingDecodeResult *decode_result);

SparkStatus SparkGlm52ServingEnginePopEvent(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingEvent *event_out);

SparkStatus SparkGlm52ServingEngineGetStats(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingStats *stats_out);

SparkStatus SparkGlm52ServingEngineCancelRequest(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestHandle request_handle);

SparkStatus SparkGlm52ServingEngineFailRequestByRequestId(
    SparkGlm52ServingEngine *engine,
    uint64_t request_id,
    SparkStatus failure_status);

SparkStatus SparkGlm52ServingEngineReleaseCompletedRequest(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestHandle request_handle);

#ifdef __cplusplus
}
#endif

#endif
