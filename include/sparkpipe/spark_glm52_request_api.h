#ifndef SPARKPIPE_SPARK_GLM52_REQUEST_API_H
#define SPARKPIPE_SPARK_GLM52_REQUEST_API_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_scheduler.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_REQUEST_API_ABI_VERSION 1u
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
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS \
    (SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION)
#define SPARK_GLM52_REQUEST_API_CONFIGURATION_KNOWN_FLAGS \
    (SPARK_GLM52_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS | \
     SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH)

#define SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME 0x00000001u
#define SPARK_GLM52_REQUEST_API_REQUEST_FLAG_KNOWN_FLAGS \
    SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME

#define SPARK_GLM52_REQUEST_API_STATE_FREE 0u
#define SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL 1u
#define SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL 2u
#define SPARK_GLM52_REQUEST_API_STATE_READY_DECODE 3u
#define SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE 4u
#define SPARK_GLM52_REQUEST_API_STATE_COMPLETED 5u
#define SPARK_GLM52_REQUEST_API_STATE_CANCELLED 6u
#define SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT 7u

#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_NONE 0u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL 1u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH 2u
#define SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH 3u

#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV 0x00000001u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE 0x00000002u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT 0x00000004u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING 0x00000008u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFILL_BATCH 0x00000010u
#define SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_FAMILY_SELECTED 0x00000020u

#define SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY 8u

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
    uint32_t reserved0;
    uint64_t last_committed_prefix_hash;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t committed_prefix_hash;
    uint64_t handle;
    uint64_t submission_order;
    const uint32_t *prompt_token_ids;
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
    uint32_t reserved;
    SparkGlm52Scheduler *scheduler;
    SparkGlm52RequestApiSlot *request_slots;
    SparkGlm52RequestApiKvPrefetchFunction kv_prefetch_function;
    void *kv_prefetch_context;
    SparkGlm52RequestApiKvPrefetchStartFunction kv_prefetch_start_function;
    SparkGlm52RequestApiKvPrefetchPollFunction kv_prefetch_poll_function;
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
    uint64_t request_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint64_t sequence_ids[SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint64_t committed_prefix_hash;
    SparkGlm52SchedulerDecision prefill_decision;
    SparkGlm52SchedulerPrefillBatchDecision prefill_batch_decision;
    SparkGlm52SchedulerBatchDecision decode_batch_decision;
    SparkGlm52KvCachePrefetchPlan kv_prefetch_plan;
} SparkGlm52RequestApiDispatch;

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
    uint32_t reserved;
    uint64_t next_handle;
    uint64_t next_sequence_id;
    uint64_t submission_counter;
    uint64_t submitted_request_count;
    uint64_t scheduled_prefill_dispatch_count;
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
    SparkGlm52RequestApiPendingPrefetch pending_prefetches[
        SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY];
} SparkGlm52RequestApi;

SparkStatus SparkGlm52RequestApiConfigurationUseAsyncKvCachePrefetchBackend(
    SparkGlm52RequestApiConfiguration *configuration,
    SparkGlm52KvCacheAsyncPrefetchBackend *backend);

SparkStatus SparkGlm52RequestApiInitialize(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiConfiguration *configuration);

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

SparkStatus SparkGlm52RequestApiCompleteDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch);

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

SparkStatus SparkGlm52RequestApiGetRequestCacheState(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle,
    SparkGlm52RequestApiCacheState *cache_state);

SparkStatus SparkGlm52RequestApiCancelRequest(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle);

SparkStatus SparkGlm52RequestApiReleaseCompletedRequest(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle);

#ifdef __cplusplus
}
#endif

#endif
