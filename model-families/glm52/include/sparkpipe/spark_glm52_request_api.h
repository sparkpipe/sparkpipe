#ifndef SPARKPIPE_SPARK_GLM52_REQUEST_API_H
#define SPARKPIPE_SPARK_GLM52_REQUEST_API_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "sparkpipe/spark_glm52_scheduler.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_REQUEST_API_ABI_VERSION 7u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiConfiguration))
#define SPARK_GLM52_REQUEST_API_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApi))
#define SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiSubmitRequest))
#define SPARK_GLM52_REQUEST_API_SLOT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiSlot))
#define SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiDispatch))
#define SPARK_GLM52_REQUEST_API_CACHE_STATE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiCacheState))
#define SPARK_GLM52_REQUEST_API_PREFILL_DISPATCH_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiPrefillDispatchView))
#define SPARK_GLM52_REQUEST_API_DECODE_DISPATCH_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RequestApiDecodeDispatchView))

#define SPARK_GLM52_REQUEST_API_INVALID_HANDLE 0ull
#define SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY 1000000u
#define SPARK_GLM52_REQUEST_API_REALTIME_PRIORITY 4000000000u
#define SPARK_GLM52_REQUEST_API_DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT 64u
#define SPARK_GLM52_REQUEST_API_DEFAULT_PREFETCH_LANE_COUNT \
    SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT
#define SPARK_GLM52_REQUEST_API_DEFAULT_DECODE_BATCH_TARGET \
    SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT
#define SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT \
    SPARK_GLM52_SCHEDULER_MAX_PACKED_REQUEST_COUNT
#define SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT \
    SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY

#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH 0x00000001u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING 0x00000002u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING 0x00000004u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING 0x00000008u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH 0x00000010u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION 0x00000020u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE 0x00000040u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION 0x00000800u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS \
    (SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION)
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT 0x00000080u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE 0x00000200u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING 0x00000100u
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_KNOWN_FLAGS \
    (SPARK_GLM52_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION)

#define SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME 0x00000001u
#define SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION 0x00000002u
#define SPARK_GLM52_REQUEST_API_REQUEST_FLAG_KNOWN_FLAGS \
    (SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME | \
     SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION)

#define SPARK_GLM52_REQUEST_API_STATE_FREE 0u
#define SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL 1u
#define SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL 2u
#define SPARK_GLM52_REQUEST_API_STATE_READY_DECODE 3u
#define SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE 4u
#define SPARK_GLM52_REQUEST_API_STATE_COMPLETED 5u
#define SPARK_GLM52_REQUEST_API_STATE_CANCELLED 6u
#define SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT 7u
#define SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY 8u
#define SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY 9u

#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_NONE 0u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL 1u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH 2u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH 3u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH 4u

#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV 0x00000001u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE 0x00000002u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT 0x00000004u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING 0x00000008u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFILL_BATCH 0x00000010u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_FAMILY_SELECTED 0x00000020u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE 0x00000040u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY 0x00000080u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_CONFIDENCE_TRUNCATED 0x00000100u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT 0x00000200u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY 0x00000400u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY 0x00000800u
#define SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT \
    SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT
#define SPARK_GLM52_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT \
    SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT
#define SPARK_GLM52_REQUEST_API_MTP_COMMIT_EMA_DIVISOR 4
#define SPARK_GLM52_REQUEST_API_MTP_COMMIT_EMA_INITIAL_MILLI 2900u
#define SPARK_GLM52_REQUEST_API_MTP_SUPPRESS_THRESHOLD_MILLI 1150u
#define SPARK_GLM52_REQUEST_API_MTP_REPROBE_INTERVAL 16u
#define SPARK_GLM52_REQUEST_API_PREFILL_INFLIGHT_WAVE_LIMIT 12u

#define SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY 8u
#define SPARK_GLM52_REQUEST_API_SLOT_HASH_SLOTS 4096u
#define SPARK_GLM52_REQUEST_API_NO_SLOT UINT32_MAX

typedef uint64_t SparkGlm52RequestApiHandle;

typedef SparkStatus (*SparkGlm52RequestApiKvPrefetchFunction)(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

typedef SparkStatus (*SparkGlm52RequestApiKvPrefetchStartFunction)(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

typedef SparkStatus (*SparkGlm52RequestApiKvPrefetchPollFunction)(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

typedef struct SparkGlm52RequestApiSubmitRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t prompt_token_count;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint64_t request_id;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
} SparkGlm52RequestApiSubmitRequest;

typedef struct SparkGlm52RequestApiSlot
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    uint32_t priority;
    uint32_t prompt_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t dispatched_prompt_token_count;
    uint32_t inflight_prefill_dispatch_count;
    uint32_t scheduled_prefill_step_count;
    uint32_t completed_prefill_step_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t remaining_thinking_token_budget;
    uint32_t remaining_output_token_budget;
    uint32_t scheduled_decode_token_count;
    uint32_t completed_decode_token_count;
    uint32_t last_committed_prefix_token_count;
    uint32_t prefix_scan_hashed_token_count;
    uint64_t prefix_scan_hash;
    uint64_t last_committed_prefix_hash;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t committed_prefix_hash;
    uint64_t handle;
    uint64_t submission_order;
    uint32_t handle_hash_next;
    uint32_t free_slot_next;
    const uint32_t *prompt_token_ids;
    uint32_t mtp_draft_token_count;
    uint32_t mtp_next_draft_token_budget;
    uint32_t mtp_commit_ema_milli;
    uint32_t mtp_probe_countdown;
    uint64_t mtp_resolution_base_position;
    uint32_t mtp_resolution_proposed_token_count;
    uint32_t mtp_resolution_accepted_token_count;
    uint32_t mtp_resolution_committed_token_count;
    uint32_t mtp_resolution_path_id;
    uint32_t mtp_draft_token_ids[SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT];
} SparkGlm52RequestApiSlot;

typedef struct SparkGlm52RequestApiPendingPrefetch
{
    uint32_t active;
    uint32_t poll_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t prefetch_id;
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
} SparkGlm52RequestApiPendingPrefetch;

typedef struct SparkGlm52RequestApiCacheState
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t computed_prompt_token_count;
    uint32_t last_committed_prefix_token_count;
    uint32_t physical_block_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t last_committed_prefix_hash;
} SparkGlm52RequestApiCacheState;

typedef struct SparkGlm52RequestApiConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t configuration_flags;
    uint32_t request_capacity;
    uint32_t prefetch_lookahead_request_count;
    uint32_t prefetch_lane_count;
    uint32_t decode_batch_target;
    uint32_t max_resident_kv_block_count;
    uint32_t decode_execution_row_capacity;
    SparkGlm52Scheduler *scheduler;
    SparkGlm52RequestApiSlot *request_slots;
    SparkGlm52RequestApiKvPrefetchFunction kv_prefetch_function;
    void *kv_prefetch_context;
    SparkGlm52RequestApiKvPrefetchStartFunction kv_prefetch_start_function;
    SparkGlm52RequestApiKvPrefetchPollFunction kv_prefetch_poll_function;
    SparkGlm52DsparkSpeculator *dspark_speculator;
} SparkGlm52RequestApiConfiguration;

typedef struct SparkGlm52RequestApiDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t kind;
    uint32_t flags;
    uint32_t request_count;
    uint32_t highest_priority;
    uint32_t shared_prefix_token_count;
    uint32_t shared_prefix_block_count;
    uint64_t prefix_cache_parent_hash;
    uint64_t prefix_cache_result_hash;
    SparkGlm52RequestApiHandle request_handles[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t request_slot_indices[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint64_t request_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint64_t sequence_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint64_t committed_prefix_hash;
    SparkGlm52SchedulerDecision prefill_decision;
    SparkGlm52SchedulerPrefillBatchDecision prefill_batch_decision;
    SparkGlm52SchedulerBatchDecision decode_batch_decision;
    SparkGlm52KvCachePrefetchPlan kv_prefetch_plan;
    uint32_t speculative_token_count;
    uint32_t speculative_verifier_token_count;
    uint32_t speculative_max_committed_token_count;
    uint32_t speculative_committed_token_counts[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t speculative_accepted_token_counts[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t speculative_fallback_token_ids[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t speculative_resolution_path_ids[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t speculative_draft_token_ids[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t speculative_confidence_milli[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t mtp_draft_token_budget;
    uint32_t decode_committed_token_counts[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
} SparkGlm52RequestApiDispatch;


typedef struct SparkGlm52RequestApiPrefillDispatchLaneView
{
    uint32_t request_index;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t request_slot_index;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkGlm52RequestApiHandle request_handle;
    const uint32_t *prompt_token_ids;
} SparkGlm52RequestApiPrefillDispatchLaneView;

typedef struct SparkGlm52RequestApiPrefillDispatchView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t active_sequence_count;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t prompt_token_stride;
    uint32_t lane_count;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkGlm52RequestApiPrefillDispatchLaneView lanes[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
} SparkGlm52RequestApiPrefillDispatchView;

typedef struct SparkGlm52RequestApiDecodeDispatchLaneView
{
    uint32_t request_index;
    uint32_t sequence_position;
    uint32_t context_token_count;
    uint32_t request_slot_index;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkGlm52RequestApiHandle request_handle;
    uint64_t mtp_resolution_base_position;
    uint32_t mtp_resolution_proposed_token_count;
    uint32_t mtp_resolution_accepted_token_count;
    uint32_t mtp_resolution_committed_token_count;
    uint32_t mtp_resolution_path_id;
} SparkGlm52RequestApiDecodeDispatchLaneView;

typedef struct SparkGlm52RequestApiDecodeDispatchView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t active_sequence_count;
    uint32_t lane_count;
    uint32_t speculative_token_count;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkGlm52RequestApiDecodeDispatchLaneView lanes[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
} SparkGlm52RequestApiDecodeDispatchView;

typedef struct SparkGlm52RequestApi
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t configuration_flags;
    uint32_t request_capacity;
    uint32_t queued_request_count;
    uint32_t running_request_count;
    uint32_t completed_request_count;
    uint32_t cancelled_request_count;
    uint32_t prefetch_lookahead_request_count;
    uint32_t prefetch_lane_count;
    uint32_t decode_batch_target;
    uint32_t max_resident_kv_block_count;
    uint32_t decode_execution_row_capacity;
    uint32_t free_slot_head;
    uint64_t next_handle;
    uint64_t next_sequence_id;
    uint64_t submission_counter;
    uint64_t submitted_request_count;
    uint64_t scheduled_prefill_dispatch_count;
    uint64_t stale_prefill_completion_count;
    uint64_t scheduled_decode_dispatch_count;
    uint64_t jit_prefetch_dispatch_count;
    uint64_t jit_prefetch_block_count;
    uint64_t jit_residency_eviction_count;
    uint64_t jit_residency_protected_block_count;
    uint64_t async_jit_prefetch_start_count;
    uint64_t async_jit_prefetch_poll_count;
    uint64_t async_jit_prefetch_completion_count;
    uint64_t next_prefetch_id;
    uint64_t lookahead_protection_sweep_count;
    uint64_t lookahead_protected_block_count;
    uint64_t prefix_family_dispatch_count;
    uint64_t prefix_family_saved_prompt_token_count;
    uint64_t prefix_family_member_count;
    SparkGlm52Scheduler *scheduler;
    SparkGlm52RequestApiSlot *request_slots;
    SparkGlm52RequestApiKvPrefetchFunction kv_prefetch_function;
    void *kv_prefetch_context;
    SparkGlm52RequestApiKvPrefetchStartFunction kv_prefetch_start_function;
    SparkGlm52RequestApiKvPrefetchPollFunction kv_prefetch_poll_function;
    SparkGlm52DsparkSpeculator *dspark_speculator;
    uint64_t dspark_tap_capture_dispatch_count;
    uint64_t dspark_draft_ready_count;
    uint64_t dspark_verify_dispatch_count;
    uint64_t dspark_accepted_draft_token_count;
    uint64_t dspark_committed_token_count;
    uint64_t dspark_rejected_token_count;
    uint64_t mtp_draft_ready_count;
    uint64_t mtp_verify_dispatch_count;
    uint64_t mtp_accepted_draft_token_count;
    uint64_t mtp_committed_token_count;
    uint64_t mtp_rejected_token_count;
    uint32_t slot_handle_hash_heads[
        SPARK_GLM52_REQUEST_API_SLOT_HASH_SLOTS];
    SparkGlm52RequestApiPendingPrefetch pending_prefetches[
        SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY];
} SparkGlm52RequestApi;

SparkStatus SparkGlm52RequestApiConfigurationUseAsyncKvCachePrefetchBackend(
    SparkGlm52RequestApiConfiguration *configuration,
    SparkGlm52KvCacheAsyncPrefetchBackend *backend);

SparkStatus SparkGlm52RequestApiInitialize(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiConfiguration *configuration);

uint32_t SparkGlm52RequestApiCurrentPipelineBatchWidth(
    const SparkGlm52RequestApi *api);

SparkStatus SparkGlm52RequestApiSubmit(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSubmitRequest *request,
    SparkGlm52RequestApiHandle *handle_out);

SparkStatus SparkGlm52RequestApiBuildJitKvPrefetchPlan(
    SparkGlm52RequestApi *api,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52RequestApiDispatchJitKvPrefetch(
    SparkGlm52RequestApi *api,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52RequestApiScheduleNext(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiDispatch *dispatch);

SparkStatus SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiDispatch *dispatch,
    const uint32_t *verifier_token_ids,
    uint32_t lane_stride,
    uint32_t verifier_token_count);

SparkStatus SparkGlm52RequestApiArmMtpVerifyDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *completed_decode_dispatch,
    const uint32_t *draft_token_ids,
    uint32_t lane_stride,
    uint32_t draft_token_count);

SparkStatus SparkGlm52RequestApiCompleteDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch);

SparkStatus SparkGlm52RequestApiDescribePrefillDispatch(
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52RequestApiPrefillDispatchView *prefill_view);

SparkStatus SparkGlm52RequestApiCopyPrefillDispatchTokenIds(
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *destination_token_ids,
    uint32_t destination_token_stride,
    uint32_t destination_lane_capacity);

SparkStatus SparkGlm52RequestApiDescribeDecodeDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52RequestApiDecodeDispatchView *decode_view);

SparkStatus SparkGlm52RequestApiBuildDispatchKvBlockTables(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity);

SparkStatus SparkGlm52RequestApiBuildDispatchKvBlockTableView(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *host_physical_block_indices,
    const uint32_t *execution_physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity,
    SparkGlm52KvBlockTableView *block_table_view);

SparkStatus SparkGlm52RequestApiCancelDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch);

SparkStatus SparkGlm52RequestApiRetryDecodeDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch);

SparkStatus SparkGlm52RequestApiGetRequestCacheState(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle,
    SparkGlm52RequestApiCacheState *cache_state);


SparkStatus SparkGlm52RequestApiFinishRequestGeneration(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle);

SparkStatus SparkGlm52RequestApiCancelRequest(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle);

SparkStatus SparkGlm52RequestApiReleaseCompletedRequest(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle);

struct SparkGlm52RowAllocatorSlotInput;

// Wave-level draft-budget assignment: divides firing_row_cap between real rows
// and speculative draft rows across all decode-eligible slots by marginal
// expected commit, using each slot's commit EMA (see spark_glm52_row_allocator.h
// for the policy). Overwrites mtp_next_draft_token_budget on every eligible
// slot, so once a caller adopts this it must be called before every wave: the
// allocator becomes the budget authority and the per-slot suppress and reprobe
// machinery no longer needs to self-restore. Suppressed slots (zero budget with
// a live reprobe countdown) keep their countdown and receive no draft rows.
// scratch_inputs and scratch_budgets are caller-owned arrays of at least
// request_capacity entries; nothing is allocated. Returns the total rows
// assigned including one base row per eligible slot, clamped to the cap.
uint32_t SparkGlm52RequestApiAssignDraftBudgets(
    SparkGlm52RequestApi *api,
    uint32_t firing_row_cap,
    struct SparkGlm52RowAllocatorSlotInput *scratch_inputs,
    uint32_t *scratch_budgets);

#ifdef __cplusplus
}
#endif

#endif
