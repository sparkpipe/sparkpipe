#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_request_api.h"

#define SPARK_TEST_REQUEST_SLOT_COUNT 32u
#define SPARK_TEST_PREFIX_ENTRY_COUNT 128u
#define SPARK_TEST_PREFIX_BINDING_COUNT 512u
#define SPARK_TEST_KV_BLOCK_COUNT 128u

typedef struct SparkTestPrefetchCapture
{
    SparkStatus return_status;
    uint32_t busy_call_budget;
    uint32_t call_count;
    uint32_t last_lane_count;
    uint32_t last_prefetch_block_count;
    uint32_t last_lane_block_counts[SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT];
    uint32_t last_physical_block_indices[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint32_t last_first_token_indices[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint32_t last_token_counts[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint64_t last_parent_hashes[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint64_t last_block_hashes[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint64_t last_content_hashes[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    SparkStatus async_start_status;
    SparkStatus async_poll_status;
    uint32_t async_start_count;
    uint32_t async_poll_count;
    uint32_t async_poll_busy_budget;
    uint32_t async_pending;
    uint64_t async_prefetch_id;
    SparkGlm52KvCachePrefetchPlan async_prefetch_plan;
} SparkTestPrefetchCapture;

typedef struct SparkTestRequestApiFixture
{
    SparkGlm52KvCacheArena kv_arena;
    SparkGlm52KvCacheBlock kv_blocks[SPARK_TEST_KV_BLOCK_COUNT];
    SparkGlm52PrefixCache prefix_cache;
    SparkGlm52PrefixCacheEntry prefix_entries[SPARK_TEST_PREFIX_ENTRY_COUNT];
    SparkGlm52PrefixCacheSequenceBinding prefix_bindings[
        SPARK_TEST_PREFIX_BINDING_COUNT];
    SparkGlm52Scheduler scheduler;
    SparkGlm52RequestApiSlot request_slots[SPARK_TEST_REQUEST_SLOT_COUNT];
    SparkGlm52RequestApi api;
    SparkTestPrefetchCapture prefetch_capture;
    unsigned char key_destination[SPARK_TEST_KV_BLOCK_COUNT][64u];
    unsigned char value_destination[SPARK_TEST_KV_BLOCK_COUNT][64u];
    unsigned char key_source[SPARK_TEST_KV_BLOCK_COUNT][64u];
    unsigned char value_source[SPARK_TEST_KV_BLOCK_COUNT][64u];
    SparkGlm52KvCacheAsyncPrefetchBackend async_prefetch_backend;
} SparkTestRequestApiFixture;

static void SparkTestFillTokenIds(
    uint32_t *token_ids,
    uint32_t token_count,
    uint32_t first_token_id)
{
    uint32_t token_index;

    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        token_ids[token_index] = first_token_id + token_index;
    }
}

static SparkStatus SparkTestCaptureKvPrefetch(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkTestPrefetchCapture *capture;
    uint32_t block_index;
    uint32_t lane_index;

    capture = (SparkTestPrefetchCapture *)context;
    assert(capture != 0);
    assert(prefetch_plan != 0);
    assert(prefetch_plan->abi_version == SPARK_GLM52_KV_CACHE_ABI_VERSION);
    assert(prefetch_plan->descriptor_bytes ==
        SPARK_GLM52_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES);
    assert(prefetch_plan->prefetch_block_count <=
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY);

    capture->call_count += 1u;
    capture->last_lane_count = prefetch_plan->lane_count;
    capture->last_prefetch_block_count = prefetch_plan->prefetch_block_count;
    for (lane_index = 0u;
         lane_index < SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT;
         ++lane_index)
    {
        capture->last_lane_block_counts[lane_index] =
            prefetch_plan->lane_block_counts[lane_index];
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        assert(prefetch_plan->blocks[block_index].lane_index <
            prefetch_plan->lane_count);
        assert(prefetch_plan->blocks[block_index].key_device_address != 0u);
        assert(prefetch_plan->blocks[block_index].value_device_address != 0u);
        capture->last_physical_block_indices[block_index] =
            prefetch_plan->blocks[block_index].physical_block_index;
        capture->last_first_token_indices[block_index] =
            prefetch_plan->blocks[block_index].first_token_index;
        capture->last_token_counts[block_index] =
            prefetch_plan->blocks[block_index].token_count;
        capture->last_parent_hashes[block_index] =
            prefetch_plan->blocks[block_index].parent_hash;
        capture->last_block_hashes[block_index] =
            prefetch_plan->blocks[block_index].block_hash;
        capture->last_content_hashes[block_index] =
            prefetch_plan->blocks[block_index].content_hash;
    }
    if (capture->busy_call_budget != 0u)
    {
        capture->busy_call_budget -= 1u;
        return SPARK_STATUS_BUSY;
    }
    return capture->return_status;
}


static SparkStatus SparkTestStartAsyncKvPrefetch(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkTestPrefetchCapture *capture;

    capture = (SparkTestPrefetchCapture *)context;
    assert(capture != 0);
    assert(prefetch_plan != 0);
    assert(prefetch_id != 0u);
    assert(capture->async_pending == 0u);

    capture->async_start_count += 1u;
    capture->async_pending = 1u;
    capture->async_prefetch_id = prefetch_id;
    capture->async_prefetch_plan = *prefetch_plan;
    return capture->async_start_status;
}

static SparkStatus SparkTestPollAsyncKvPrefetch(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkTestPrefetchCapture *capture;

    capture = (SparkTestPrefetchCapture *)context;
    assert(capture != 0);
    assert(prefetch_plan != 0);
    assert(capture->async_pending != 0u);
    assert(capture->async_prefetch_id == prefetch_id);
    assert(capture->async_prefetch_plan.prefetch_block_count ==
        prefetch_plan->prefetch_block_count);

    capture->async_poll_count += 1u;
    if (capture->async_poll_busy_budget != 0u)
    {
        capture->async_poll_busy_budget -= 1u;
        return SPARK_STATUS_BUSY;
    }
    if (capture->async_poll_status != SPARK_STATUS_OK)
    {
        capture->async_pending = 0u;
        return capture->async_poll_status;
    }

    capture->async_pending = 0u;
    return SPARK_STATUS_OK;
}

static void SparkTestEnableAsyncPrefetch(
    SparkTestRequestApiFixture *fixture)
{
    fixture->api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH;
    fixture->api.kv_prefetch_function = 0;
    fixture->api.kv_prefetch_start_function = SparkTestStartAsyncKvPrefetch;
    fixture->api.kv_prefetch_poll_function = SparkTestPollAsyncKvPrefetch;
    fixture->api.next_prefetch_id = 1u;
    fixture->prefetch_capture.async_start_status = SPARK_STATUS_OK;
    fixture->prefetch_capture.async_poll_status = SPARK_STATUS_OK;
}

static void SparkTestInitializeFixture(
    SparkTestRequestApiFixture *fixture)
{
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration prefix_configuration;
    SparkGlm52SchedulerConfiguration scheduler_configuration;
    SparkGlm52RequestApiConfiguration api_configuration;

    memset(fixture, 0, sizeof(*fixture));
    fixture->prefetch_capture.return_status = SPARK_STATUS_OK;

    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count = SPARK_TEST_KV_BLOCK_COUNT;
    kv_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 78u;
    kv_configuration.kv_head_count = 8u;
    kv_configuration.head_dim = 128u;
    kv_configuration.bytes_per_scalar = 2u;
    kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    kv_configuration.blocks = fixture->kv_blocks;
    assert(SparkGlm52KvCacheArenaInitialize(
        &fixture->kv_arena,
        &kv_configuration) == SPARK_STATUS_OK);

    memset(&prefix_configuration, 0, sizeof(prefix_configuration));
    prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    prefix_configuration.descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    prefix_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    prefix_configuration.entry_count = SPARK_TEST_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count = SPARK_TEST_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count = SPARK_TEST_PREFIX_BINDING_COUNT;
    prefix_configuration.entries = fixture->prefix_entries;
    prefix_configuration.sequence_bindings = fixture->prefix_bindings;
    prefix_configuration.kv_cache_arena = &fixture->kv_arena;
    assert(SparkGlm52PrefixCacheInitialize(
        &fixture->prefix_cache,
        &prefix_configuration) == SPARK_STATUS_OK);

    memset(&scheduler_configuration, 0, sizeof(scheduler_configuration));
    scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_configuration.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    scheduler_configuration.spark_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    scheduler_configuration.queue_depth_per_spark = 2u;
    scheduler_configuration.measured_profile_id =
        SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    scheduler_configuration.quantization_mode =
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    scheduler_configuration.prefix_cache = &fixture->prefix_cache;
    assert(SparkGlm52SchedulerInitialize(
        &fixture->scheduler,
        &scheduler_configuration) == SPARK_STATUS_OK);

    memset(&api_configuration, 0, sizeof(api_configuration));
    api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api_configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    api_configuration.request_capacity = SPARK_TEST_REQUEST_SLOT_COUNT;
    api_configuration.prefetch_lane_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    api_configuration.decode_batch_target =
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    api_configuration.scheduler = &fixture->scheduler;
    api_configuration.request_slots = fixture->request_slots;
    api_configuration.kv_prefetch_function = SparkTestCaptureKvPrefetch;
    api_configuration.kv_prefetch_context = &fixture->prefetch_capture;
    assert(SparkGlm52RequestApiInitialize(
        &fixture->api,
        &api_configuration) == SPARK_STATUS_OK);
}


static void SparkTestFillMemoryKvSource(
    SparkTestRequestApiFixture *fixture)
{
    uint32_t block_index;
    uint32_t byte_index;

    for (block_index = 0u; block_index < SPARK_TEST_KV_BLOCK_COUNT; ++block_index)
    {
        for (byte_index = 0u; byte_index < 64u; ++byte_index)
        {
            fixture->key_source[block_index][byte_index] =
                (unsigned char)(0x31u + block_index + byte_index);
            fixture->value_source[block_index][byte_index] =
                (unsigned char)(0xa7u + block_index + byte_index);
        }
    }
}

static void SparkTestInitializeFixtureWithAsyncMemoryBackend(
    SparkTestRequestApiFixture *fixture)
{
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration prefix_configuration;
    SparkGlm52SchedulerConfiguration scheduler_configuration;
    SparkGlm52RequestApiConfiguration api_configuration;
    SparkGlm52KvCacheAsyncPrefetchBackendConfiguration backend_configuration;

    memset(fixture, 0, sizeof(*fixture));
    SparkTestFillMemoryKvSource(fixture);

    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count = SPARK_TEST_KV_BLOCK_COUNT;
    kv_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 1u;
    kv_configuration.kv_head_count = 1u;
    kv_configuration.head_dim = 32u;
    kv_configuration.bytes_per_scalar = 2u;
    kv_configuration.key_block_stride_bytes = 64u;
    kv_configuration.value_block_stride_bytes = 64u;
    kv_configuration.key_device_base = fixture->key_destination;
    kv_configuration.value_device_base = fixture->value_destination;
    kv_configuration.blocks = fixture->kv_blocks;
    assert(SparkGlm52KvCacheArenaInitialize(
        &fixture->kv_arena,
        &kv_configuration) == SPARK_STATUS_OK);

    memset(&prefix_configuration, 0, sizeof(prefix_configuration));
    prefix_configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    prefix_configuration.descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    prefix_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    prefix_configuration.entry_count = SPARK_TEST_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count = SPARK_TEST_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count = SPARK_TEST_PREFIX_BINDING_COUNT;
    prefix_configuration.entries = fixture->prefix_entries;
    prefix_configuration.sequence_bindings = fixture->prefix_bindings;
    prefix_configuration.kv_cache_arena = &fixture->kv_arena;
    assert(SparkGlm52PrefixCacheInitialize(
        &fixture->prefix_cache,
        &prefix_configuration) == SPARK_STATUS_OK);

    memset(&scheduler_configuration, 0, sizeof(scheduler_configuration));
    scheduler_configuration.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_configuration.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    scheduler_configuration.spark_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    scheduler_configuration.queue_depth_per_spark = 2u;
    scheduler_configuration.measured_profile_id =
        SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    scheduler_configuration.quantization_mode =
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
    scheduler_configuration.prefix_cache = &fixture->prefix_cache;
    assert(SparkGlm52SchedulerInitialize(
        &fixture->scheduler,
        &scheduler_configuration) == SPARK_STATUS_OK);

    memset(&backend_configuration, 0, sizeof(backend_configuration));
    backend_configuration.abi_version =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    backend_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
    backend_configuration.flags =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE |
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS;
    backend_configuration.lane_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    backend_configuration.max_inflight_prefetch_count = 2u;
    backend_configuration.physical_block_count = SPARK_TEST_KV_BLOCK_COUNT;
    backend_configuration.blocks_per_poll = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    backend_configuration.key_source_stride_bytes = 64u;
    backend_configuration.value_source_stride_bytes = 64u;
    backend_configuration.key_transfer_bytes = 64u;
    backend_configuration.value_transfer_bytes = 64u;
    backend_configuration.key_source_base = fixture->key_source;
    backend_configuration.value_source_base = fixture->value_source;
    assert(SparkGlm52KvCacheAsyncPrefetchBackendInitialize(
        &fixture->async_prefetch_backend,
        &backend_configuration) == SPARK_STATUS_OK);

    memset(&api_configuration, 0, sizeof(api_configuration));
    api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api_configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    api_configuration.request_capacity = SPARK_TEST_REQUEST_SLOT_COUNT;
    api_configuration.prefetch_lane_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    api_configuration.decode_batch_target =
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    api_configuration.scheduler = &fixture->scheduler;
    api_configuration.request_slots = fixture->request_slots;
    assert(SparkGlm52RequestApiConfigurationUseAsyncKvCachePrefetchBackend(
        &api_configuration,
        &fixture->async_prefetch_backend) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiInitialize(
        &fixture->api,
        &api_configuration) == SPARK_STATUS_OK);
}

static void SparkTestInitializeSubmitRequest(
    SparkGlm52RequestApiSubmitRequest *request,
    uint64_t request_id,
    uint64_t sequence_id,
    uint32_t priority,
    const uint32_t *prompt_token_ids,
    uint32_t prompt_token_count,
    uint32_t output_token_budget)
{
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES;
    request->priority = priority;
    request->prompt_token_count = prompt_token_count;
    request->output_token_budget = output_token_budget;
    request->request_id = request_id;
    request->sequence_id = sequence_id;
    request->prompt_token_ids = prompt_token_ids;
}

static void SparkTestRequestApiWarmsPrefixCacheAndReleases(
    SparkTestRequestApiFixture *fixture,
    const uint32_t *prompt_token_ids,
    uint32_t prompt_token_count)
{
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;

    SparkTestInitializeSubmitRequest(
        &request,
        1u,
        1001u,
        10u,
        prompt_token_ids,
        prompt_token_count,
        0u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture->api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture->api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture->api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture->api,
        handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiJitPrefetchesCachedPrefixForPriorityRequest(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle low_priority_handle;
    SparkGlm52RequestApiHandle realtime_handle;
    uint32_t shared_prompt[32u];
    uint32_t other_prompt[32u];
    uint64_t expected_parent_hash;
    uint64_t expected_block_hash;
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t cold_physical_block_index;

    SparkTestFillTokenIds(shared_prompt, 32u, 10000u);
    SparkTestFillTokenIds(other_prompt, 32u, 20000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        shared_prompt,
        32u);

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        shared_prompt,
        32u,
        physical_block_indices,
        4u,
        &matched_token_count,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(physical_block_count == 1u);
    cold_physical_block_index = physical_block_indices[0];
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_physical_block_index) == SPARK_STATUS_OK);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);

    SparkTestInitializeSubmitRequest(
        &request,
        2u,
        2002u,
        10u,
        other_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &low_priority_handle) == SPARK_STATUS_OK);
    (void)low_priority_handle;

    SparkTestInitializeSubmitRequest(
        &request,
        3u,
        3003u,
        0u,
        shared_prompt,
        32u,
        1u);
    request.flags = SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &realtime_handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0] == realtime_handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV) != 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE) != 0u);
    assert(dispatch.kv_prefetch_plan.lane_count ==
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT);
    assert(dispatch.kv_prefetch_plan.prefetch_block_count == 1u);
    expected_parent_hash = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    expected_block_hash = SparkGlm52PrefixCacheHashBlock(
        shared_prompt,
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS,
        expected_parent_hash);
    assert(dispatch.kv_prefetch_plan.blocks[0].physical_block_index ==
        cold_physical_block_index);
    assert(dispatch.kv_prefetch_plan.blocks[0].first_token_index == 0u);
    assert(dispatch.kv_prefetch_plan.blocks[0].token_count ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(dispatch.kv_prefetch_plan.blocks[0].parent_hash ==
        expected_parent_hash);
    assert(dispatch.kv_prefetch_plan.blocks[0].block_hash ==
        expected_block_hash);
    assert(dispatch.kv_prefetch_plan.blocks[0].content_hash != 0u);
    assert(fixture.prefetch_capture.call_count == 1u);
    assert(fixture.prefetch_capture.last_lane_count ==
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT);
    assert(fixture.prefetch_capture.last_prefetch_block_count == 1u);
    assert(fixture.prefetch_capture.last_physical_block_indices[0] ==
        cold_physical_block_index);
    assert(fixture.prefetch_capture.last_first_token_indices[0] == 0u);
    assert(fixture.prefetch_capture.last_token_counts[0] ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(fixture.prefetch_capture.last_parent_hashes[0] ==
        expected_parent_hash);
    assert(fixture.prefetch_capture.last_block_hashes[0] ==
        expected_block_hash);
    assert(fixture.prefetch_capture.last_content_hashes[0] ==
        dispatch.kv_prefetch_plan.blocks[0].content_hash);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert(dispatch.prefill_decision.cached_prefix_token_count ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiBatchesReadyDecodeRequestsAndConsumesBudgets(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch first_dispatch;
    SparkGlm52RequestApiDispatch second_dispatch;
    SparkGlm52RequestApiHandle first_handle;
    SparkGlm52RequestApiHandle second_handle;
    uint32_t first_prompt[16u];
    uint32_t second_prompt[16u];

    SparkTestFillTokenIds(first_prompt, 16u, 30000u);
    SparkTestFillTokenIds(second_prompt, 16u, 40000u);
    SparkTestInitializeFixture(&fixture);

    SparkTestInitializeSubmitRequest(
        &request,
        11u,
        5011u,
        10u,
        first_prompt,
        16u,
        2u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &first_dispatch) == SPARK_STATUS_OK);
    assert(first_dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &first_dispatch) == SPARK_STATUS_OK);

    SparkTestInitializeSubmitRequest(
        &request,
        12u,
        5012u,
        20u,
        second_prompt,
        16u,
        2u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &second_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &second_dispatch) == SPARK_STATUS_OK);
    assert(second_dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(second_dispatch.request_handles[0] == second_handle);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &second_dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &first_dispatch) == SPARK_STATUS_OK);
    assert(first_dispatch.accepted == 1u);
    assert(first_dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(first_dispatch.request_count == 2u);
    assert(first_dispatch.request_handles[0] == second_handle);
    assert(first_dispatch.request_handles[1] == first_handle);
    assert(first_dispatch.decode_batch_decision.active_sequence_count == 2u);
    assert((first_dispatch.decode_batch_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) != 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &first_dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &second_dispatch) == SPARK_STATUS_OK);
    assert(second_dispatch.accepted == 1u);
    assert(second_dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(second_dispatch.request_count == 2u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &second_dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        second_handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiCohortsSamePromptRequestsAndSharesBlocks(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handles[4u];
    uint64_t sequence_ids[4u];
    uint32_t prompt[32u];
    uint32_t leader_block_table[4u];
    uint32_t block_table[4u];
    uint32_t leader_block_count;
    uint32_t block_count;
    uint32_t request_index;

    SparkTestFillTokenIds(prompt, 32u, 50000u);
    SparkTestInitializeFixture(&fixture);
    for (request_index = 0u; request_index < 4u; ++request_index)
    {
        sequence_ids[request_index] = 7000u + request_index;
        SparkTestInitializeSubmitRequest(
            &request,
            100u + request_index,
            sequence_ids[request_index],
            40u,
            prompt,
            32u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 4u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT) != 0u);
    assert(dispatch.prefill_decision.total_scheduled_token_count == 32u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        &fixture.prefix_cache,
        sequence_ids[0],
        32u,
        leader_block_table,
        4u,
        &leader_block_count) == SPARK_STATUS_OK);
    assert(leader_block_count == 2u);
    for (request_index = 1u; request_index < 4u; ++request_index)
    {
        assert(SparkGlm52PrefixCacheBuildPhysicalBlockTable(
            &fixture.prefix_cache,
            sequence_ids[request_index],
            32u,
            block_table,
            4u,
            &block_count) == SPARK_STATUS_OK);
        assert(block_count == leader_block_count);
        assert(block_table[0] == leader_block_table[0]);
        assert(block_table[1] == leader_block_table[1]);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 4u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 4u; ++request_index)
    {
        assert(SparkGlm52RequestApiReleaseCompletedRequest(
            &fixture.api,
            handles[request_index]) == SPARK_STATUS_OK);
    }
}

static void SparkTestBuildSharedPrefixPrompt(
    uint32_t *prompt,
    const uint32_t *shared_prefix,
    uint32_t shared_prefix_token_count,
    uint32_t suffix_first_token_id,
    uint32_t suffix_token_count)
{
    uint32_t token_index;

    for (token_index = 0u;
         token_index < shared_prefix_token_count;
         ++token_index)
    {
        prompt[token_index] = shared_prefix[token_index];
    }
    for (token_index = 0u; token_index < suffix_token_count; ++token_index)
    {
        prompt[shared_prefix_token_count + token_index] =
            suffix_first_token_id + token_index;
    }
}

static void SparkTestRequestApiCohortsArbitrarySharedPrefixWithSuffixes(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiCacheState cache_state;
    SparkGlm52PrefixCachePromptHash shared_prefix_hash;
    SparkGlm52PrefixCachePromptHash full_prompt_hash;
    SparkGlm52PrefixCachePromptHash suffix_hash;
    SparkGlm52RequestApiHandle handles[7u];
    uint64_t sequence_ids[7u];
    uint32_t shared_prefix[32u];
    uint32_t prompts[7u][48u];
    uint32_t leader_block_table[4u];
    uint32_t follower_block_table[4u];
    uint32_t batch_block_tables[7u][4u];
    uint32_t execution_block_tables[7u][4u];
    uint32_t batch_block_counts[7u];
    SparkGlm52KvBlockTableView block_table_view;
    uint32_t leader_block_count;
    uint32_t follower_block_count;
    uint32_t request_index;

    SparkTestFillTokenIds(shared_prefix, 32u, 70000u);
    SparkTestInitializeFixture(&fixture);
    for (request_index = 0u; request_index < 7u; ++request_index)
    {
        SparkTestBuildSharedPrefixPrompt(
            prompts[request_index],
            shared_prefix,
            32u,
            80000u + request_index * 100u,
            16u);
        sequence_ids[request_index] = 9000u + request_index;
        SparkTestInitializeSubmitRequest(
            &request,
            500u + request_index,
            sequence_ids[request_index],
            60u,
            prompts[request_index],
            48u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 7u);
    assert(dispatch.shared_prefix_token_count == 32u);
    assert(dispatch.shared_prefix_block_count == 2u);
    assert(dispatch.prefill_decision.prompt_token_count == 48u);
    assert(dispatch.prefill_decision.scheduled_prompt_token_count == 32u);
    assert(dispatch.prefill_decision.total_scheduled_token_count == 32u);
    assert(SparkGlm52PrefixCacheHashPromptTokens(
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS,
        SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH,
        prompts[0],
        32u,
        &shared_prefix_hash) == SPARK_STATUS_OK);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT) != 0u);
    assert(dispatch.prefix_cache_parent_hash ==
        SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH);
    assert(dispatch.prefix_cache_result_hash == shared_prefix_hash.prompt_hash);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        &fixture.prefix_cache,
        sequence_ids[0],
        32u,
        leader_block_table,
        4u,
        &leader_block_count) == SPARK_STATUS_OK);
    assert(leader_block_count == 2u);
    for (request_index = 0u; request_index < 7u; ++request_index)
    {
        assert(SparkGlm52RequestApiGetRequestCacheState(
            &fixture.api,
            handles[request_index],
            &cache_state) == SPARK_STATUS_OK);
        assert(cache_state.computed_prompt_token_count == 32u);
        assert(cache_state.last_committed_prefix_token_count == 32u);
        assert(cache_state.last_committed_prefix_hash ==
            dispatch.prefix_cache_result_hash);
        assert(cache_state.state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);
        assert(SparkGlm52PrefixCacheBuildPhysicalBlockTable(
            &fixture.prefix_cache,
            sequence_ids[request_index],
            32u,
            follower_block_table,
            4u,
            &follower_block_count) == SPARK_STATUS_OK);
        assert(follower_block_count == leader_block_count);
        assert(follower_block_table[0] == leader_block_table[0]);
        assert(follower_block_table[1] == leader_block_table[1]);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFILL_BATCH) != 0u);
    assert(dispatch.request_count == 7u);
    assert(dispatch.prefill_batch_decision.accepted == 1u);
    assert(dispatch.prefill_batch_decision.active_sequence_count == 7u);
    assert(dispatch.prefill_batch_decision.batch_bucket ==
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(dispatch.prefill_batch_decision.maximum_scheduled_prompt_token_count ==
        16u);
    assert(dispatch.prefill_batch_decision.total_scheduled_token_count ==
        7u * 16u);
    assert((dispatch.prefill_batch_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK) != 0u);
    for (request_index = 0u; request_index < 7u; ++request_index)
    {
        const SparkGlm52SchedulerPrefillBatchLane *lane;

        lane = &dispatch.prefill_batch_decision.lanes[request_index];
        assert(dispatch.request_handles[request_index] == handles[request_index]);
        assert(lane->sequence_id == sequence_ids[request_index]);
        assert(lane->prompt_token_count == 48u);
        assert(lane->computed_prompt_token_count == 32u);
        assert(lane->cached_prefix_token_count == 32u);
        assert(lane->scheduled_prompt_token_offset == 32u);
        assert(lane->scheduled_prompt_token_count == 16u);
        assert(lane->cache_commit_token_count_after_step == 48u);
        assert(lane->kv_block_token_count ==
            SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
        assert(lane->kv_block_table_token_count == 48u);
        assert(SparkGlm52PrefixCacheHashPromptTokens(
            SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS,
            SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH,
            prompts[request_index],
            48u,
            &full_prompt_hash) == SPARK_STATUS_OK);
        assert(SparkGlm52PrefixCacheHashPromptTokens(
            SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS,
            shared_prefix_hash.prompt_hash,
            &prompts[request_index][32u],
            16u,
            &suffix_hash) == SPARK_STATUS_OK);
        assert(lane->prefix_cache_parent_hash == shared_prefix_hash.prompt_hash);
        assert(lane->prefix_cache_result_hash == full_prompt_hash.prompt_hash);
        assert(lane->prefix_cache_result_hash == suffix_hash.prompt_hash);
    }
    memset(batch_block_tables, 0, sizeof(batch_block_tables));
    memset(batch_block_counts, 0, sizeof(batch_block_counts));
    assert(SparkGlm52RequestApiBuildDispatchKvBlockTables(
        &fixture.api,
        &dispatch,
        &batch_block_tables[0u][0u],
        4u,
        4u,
        batch_block_counts,
        7u) == SPARK_STATUS_OK);
    memcpy(execution_block_tables, batch_block_tables, sizeof(batch_block_tables));
    memset(&block_table_view, 0, sizeof(block_table_view));
    assert(SparkGlm52RequestApiBuildDispatchKvBlockTableView(
        &fixture.api,
        &dispatch,
        &batch_block_tables[0u][0u],
        &execution_block_tables[0u][0u],
        4u,
        4u,
        batch_block_counts,
        7u,
        &block_table_view) == SPARK_STATUS_OK);
    assert(block_table_view.abi_version == SPARK_GLM52_KV_CACHE_ABI_VERSION);
    assert(block_table_view.descriptor_bytes ==
        SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES);
    assert(block_table_view.block_token_count ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(block_table_view.lane_count == 7u);
    assert(block_table_view.lane_stride == 4u);
    assert(block_table_view.lane_capacity == 4u);
    assert(block_table_view.physical_block_indices ==
        &execution_block_tables[0u][0u]);
    assert(block_table_view.host_physical_block_indices ==
        &batch_block_tables[0u][0u]);
    assert(block_table_view.lane_physical_block_counts == batch_block_counts);
    for (request_index = 0u; request_index < 7u; ++request_index)
    {
        assert(batch_block_counts[request_index] == 3u);
        assert(batch_block_tables[request_index][0u] == leader_block_table[0u]);
        assert(batch_block_tables[request_index][1u] == leader_block_table[1u]);
        assert(batch_block_tables[request_index][2u] != leader_block_table[0u]);
        assert(batch_block_tables[request_index][2u] != leader_block_table[1u]);
    }
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 7u; ++request_index)
    {
        assert(SparkGlm52RequestApiGetRequestCacheState(
            &fixture.api,
            handles[request_index],
            &cache_state) == SPARK_STATUS_OK);
        assert(cache_state.computed_prompt_token_count == 48u);
        assert(cache_state.last_committed_prefix_token_count == 48u);
        assert(cache_state.state == SPARK_GLM52_REQUEST_API_STATE_READY_DECODE);
    }
}



static void SparkTestRequestApiChoosesWiderSharedPrefixFamily(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiCacheState cache_state;
    SparkGlm52PrefixCachePromptHash shared_prefix_hash;
    SparkGlm52RequestApiHandle handles[10u];
    uint32_t shared_prefix[32u];
    uint32_t close_extra[16u];
    uint32_t prompts[10u][64u];
    uint32_t request_index;

    SparkTestFillTokenIds(shared_prefix, 32u, 122000u);
    SparkTestFillTokenIds(close_extra, 16u, 123000u);
    SparkTestInitializeFixture(&fixture);

    SparkTestBuildSharedPrefixPrompt(
        prompts[0u],
        shared_prefix,
        32u,
        124000u,
        32u);
    memcpy(&prompts[0u][32u], close_extra, sizeof(close_extra));
    SparkTestFillTokenIds(&prompts[0u][48u], 16u, 125000u);

    SparkTestBuildSharedPrefixPrompt(
        prompts[1u],
        shared_prefix,
        32u,
        126000u,
        32u);
    memcpy(&prompts[1u][32u], close_extra, sizeof(close_extra));
    SparkTestFillTokenIds(&prompts[1u][48u], 16u, 127000u);

    for (request_index = 2u; request_index < 10u; ++request_index)
    {
        SparkTestBuildSharedPrefixPrompt(
            prompts[request_index],
            shared_prefix,
            32u,
            128000u + request_index * 100u,
            32u);
    }

    for (request_index = 0u; request_index < 10u; ++request_index)
    {
        SparkTestInitializeSubmitRequest(
            &request,
            1220u + request_index,
            12200u + request_index,
            70u,
            prompts[request_index],
            64u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 10u);
    assert(dispatch.shared_prefix_token_count == 32u);
    assert(dispatch.shared_prefix_block_count == 2u);
    assert(dispatch.prefill_decision.prompt_token_count == 64u);
    assert(dispatch.prefill_decision.scheduled_prompt_token_count == 32u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT) != 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_FAMILY_SELECTED) != 0u);
    assert(SparkGlm52PrefixCacheHashPromptTokens(
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS,
        SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH,
        prompts[0u],
        32u,
        &shared_prefix_hash) == SPARK_STATUS_OK);
    assert(dispatch.prefix_cache_result_hash == shared_prefix_hash.prompt_hash);
    assert(fixture.api.prefix_family_dispatch_count == 1u);
    assert(fixture.api.prefix_family_member_count == 10u);
    assert(fixture.api.prefix_family_saved_prompt_token_count == 32u * 9u);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 10u; ++request_index)
    {
        assert(SparkGlm52RequestApiGetRequestCacheState(
            &fixture.api,
            handles[request_index],
            &cache_state) == SPARK_STATUS_OK);
        assert(cache_state.computed_prompt_token_count == 32u);
        assert(cache_state.state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);
    }
}

static void SparkTestRequestApiBatchesVariableOneBlockSuffixes(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiCacheState cache_state;
    SparkGlm52RequestApiHandle handles[5u];
    uint64_t sequence_ids[5u];
    uint32_t shared_prefix[32u];
    uint32_t prompts[5u][48u];
    uint32_t suffix_token_counts[5u] = { 1u, 5u, 16u, 9u, 12u };
    uint32_t scheduled_token_count_seen[17u];
    uint32_t batch_block_tables[5u][4u];
    uint32_t batch_block_counts[5u];
    uint32_t request_index;
    uint32_t scheduled_token_total;

    SparkTestFillTokenIds(shared_prefix, 32u, 94000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        shared_prefix,
        32u);

    for (request_index = 0u; request_index < 5u; ++request_index)
    {
        SparkTestBuildSharedPrefixPrompt(
            prompts[request_index],
            shared_prefix,
            32u,
            95000u + request_index * 100u,
            suffix_token_counts[request_index]);
        sequence_ids[request_index] = 9950u + request_index;
        SparkTestInitializeSubmitRequest(
            &request,
            950u + request_index,
            sequence_ids[request_index],
            75u,
            prompts[request_index],
            32u + suffix_token_counts[request_index],
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    memset(&dispatch, 0, sizeof(dispatch));
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(dispatch.request_count == 5u);
    assert(dispatch.prefill_batch_decision.accepted == 1u);
    assert(dispatch.prefill_batch_decision.active_sequence_count == 5u);
    assert(dispatch.prefill_batch_decision.batch_bucket ==
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(dispatch.prefill_batch_decision.maximum_scheduled_prompt_token_count ==
        16u);
    assert(dispatch.prefill_batch_decision.graph_sequence_padding_count == 11u);

    memset(scheduled_token_count_seen, 0, sizeof(scheduled_token_count_seen));
    scheduled_token_total = 0u;
    for (request_index = 0u;
         request_index < dispatch.prefill_batch_decision.packed_request_count;
         ++request_index)
    {
        const SparkGlm52SchedulerPrefillBatchLane *lane;

        lane = &dispatch.prefill_batch_decision.lanes[request_index];
        assert(lane->cached_prefix_token_count == 32u);
        assert(lane->computed_prompt_token_count == 32u);
        assert(lane->scheduled_prompt_token_offset == 32u);
        assert(lane->scheduled_prompt_token_count > 0u);
        assert(lane->scheduled_prompt_token_count <= 16u);
        assert(lane->remaining_prompt_token_count_after_step == 0u);
        assert(lane->kv_block_table_token_count == lane->prompt_token_count);
        scheduled_token_count_seen[lane->scheduled_prompt_token_count] += 1u;
        scheduled_token_total += lane->scheduled_prompt_token_count;
    }
    assert(scheduled_token_count_seen[1u] == 1u);
    assert(scheduled_token_count_seen[5u] == 1u);
    assert(scheduled_token_count_seen[9u] == 1u);
    assert(scheduled_token_count_seen[12u] == 1u);
    assert(scheduled_token_count_seen[16u] == 1u);
    assert(dispatch.prefill_batch_decision.total_scheduled_token_count ==
        scheduled_token_total);
    assert(scheduled_token_total == 43u);

    memset(batch_block_tables, 0, sizeof(batch_block_tables));
    memset(batch_block_counts, 0, sizeof(batch_block_counts));
    assert(SparkGlm52RequestApiBuildDispatchKvBlockTables(
        &fixture.api,
        &dispatch,
        &batch_block_tables[0u][0u],
        4u,
        4u,
        batch_block_counts,
        5u) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 5u; ++request_index)
    {
        assert(batch_block_counts[request_index] == 3u);
    }

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 5u; ++request_index)
    {
        assert(SparkGlm52RequestApiGetRequestCacheState(
            &fixture.api,
            handles[request_index],
            &cache_state) == SPARK_STATUS_OK);
        assert(cache_state.computed_prompt_token_count ==
            32u + suffix_token_counts[request_index]);
        assert(cache_state.last_committed_prefix_token_count ==
            32u + suffix_token_counts[request_index]);
        assert(cache_state.state == SPARK_GLM52_REQUEST_API_STATE_READY_DECODE);
    }
}


static void SparkTestRequestApiOpportunisticLookaheadDoesNotBlockReadyPriorityPrefill(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle low_priority_handle;
    SparkGlm52RequestApiHandle high_priority_handle;
    uint32_t high_priority_prompt[32u];
    uint32_t low_priority_prompt[32u];
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t low_priority_cold_block_index;

    SparkTestFillTokenIds(high_priority_prompt, 32u, 91000u);
    SparkTestFillTokenIds(low_priority_prompt, 32u, 92000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        high_priority_prompt,
        32u);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        low_priority_prompt,
        32u);

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        low_priority_prompt,
        32u,
        physical_block_indices,
        4u,
        &matched_token_count,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(physical_block_count == 1u);
    low_priority_cold_block_index = physical_block_indices[0];
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        low_priority_cold_block_index) == SPARK_STATUS_OK);
    fixture.prefetch_capture.return_status = SPARK_STATUS_BUSY;
    fixture.prefetch_capture.call_count = 0u;

    SparkTestInitializeSubmitRequest(
        &request,
        910u,
        9910u,
        10u,
        low_priority_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &low_priority_handle) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,
        911u,
        9911u,
        100u,
        high_priority_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &high_priority_handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0] == high_priority_handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING) == 0u);
    assert(fixture.prefetch_capture.call_count == 1u);
    assert(fixture.prefetch_capture.last_prefetch_block_count == 1u);
    assert(fixture.prefetch_capture.last_physical_block_indices[0] ==
        low_priority_cold_block_index);
    assert((fixture.kv_blocks[low_priority_cold_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert(fixture.api.request_slots[0].state ==
        SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);

    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        low_priority_handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiPrefetchesLiveNonresidentDecodeBlocks(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[32u];
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t cold_physical_block_index;

    SparkTestFillTokenIds(prompt, 32u, 93000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestInitializeSubmitRequest(
        &request,
        930u,
        9930u,
        70u,
        prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        &fixture.prefix_cache,
        9930u,
        32u,
        physical_block_indices,
        4u,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 2u);
    cold_physical_block_index = physical_block_indices[0];
    assert(fixture.kv_blocks[cold_physical_block_index].reference_count != 0u);
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_physical_block_index) == SPARK_STATUS_OK);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    fixture.prefetch_capture.call_count = 0u;
    fixture.prefetch_capture.return_status = SPARK_STATUS_OK;

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0] == handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV) != 0u);
    assert(fixture.prefetch_capture.call_count == 1u);
    assert(fixture.prefetch_capture.last_prefetch_block_count == 1u);
    assert(fixture.prefetch_capture.last_physical_block_indices[0] ==
        cold_physical_block_index);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}


static void SparkTestRequestApiBatchesDecodeAfterBatchCriticalPrefetch(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle first_handle;
    SparkGlm52RequestApiHandle second_handle;
    uint32_t first_prompt[32u];
    uint32_t second_prompt[32u];
    uint32_t second_block_table[4u];
    uint32_t second_block_count;
    uint32_t cold_second_block_index;
    uint32_t slot_index;

    SparkTestFillTokenIds(first_prompt, 32u, 94000u);
    SparkTestFillTokenIds(second_prompt, 32u, 95000u);
    SparkTestInitializeFixture(&fixture);

    SparkTestInitializeSubmitRequest(
        &request,
        940u,
        9940u,
        50u,
        first_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    SparkTestInitializeSubmitRequest(
        &request,
        941u,
        9941u,
        60u,
        second_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &second_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0u] == second_handle);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    for (slot_index = 0u; slot_index < fixture.api.request_capacity; ++slot_index)
    {
        if (fixture.api.request_slots[slot_index].handle == first_handle ||
            fixture.api.request_slots[slot_index].handle == second_handle)
        {
            fixture.api.request_slots[slot_index].priority = 50u;
        }
    }

    assert(SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        &fixture.prefix_cache,
        9941u,
        32u,
        second_block_table,
        4u,
        &second_block_count) == SPARK_STATUS_OK);
    assert(second_block_count == 2u);
    cold_second_block_index = second_block_table[0u];
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_second_block_index) == SPARK_STATUS_OK);
    fixture.prefetch_capture.call_count = 0u;
    fixture.prefetch_capture.busy_call_budget = 1u;
    fixture.prefetch_capture.return_status = SPARK_STATUS_OK;

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 2u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV) != 0u);
    assert(fixture.prefetch_capture.call_count == 2u);
    assert(dispatch.kv_prefetch_plan.prefetch_block_count == 1u);
    assert(dispatch.kv_prefetch_plan.blocks[0u].physical_block_index ==
        cold_second_block_index);
    assert((fixture.kv_blocks[cold_second_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        second_handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiBatchesPrefillAfterBatchCriticalPrefetch(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle first_handle;
    SparkGlm52RequestApiHandle second_handle;
    uint32_t first_prefix[32u];
    uint32_t second_prefix[32u];
    uint32_t first_prompt[48u];
    uint32_t second_prompt[48u];
    uint32_t second_prefix_block_table[4u];
    uint32_t matched_token_count;
    uint32_t second_prefix_block_count;
    uint32_t cold_second_prefix_block_index;

    SparkTestFillTokenIds(first_prefix, 32u, 96000u);
    SparkTestFillTokenIds(second_prefix, 32u, 97000u);
    SparkTestBuildSharedPrefixPrompt(
        first_prompt,
        first_prefix,
        32u,
        98000u,
        16u);
    SparkTestBuildSharedPrefixPrompt(
        second_prompt,
        second_prefix,
        32u,
        99000u,
        16u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        first_prefix,
        32u);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        second_prefix,
        32u);

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        second_prompt,
        48u,
        second_prefix_block_table,
        4u,
        &matched_token_count,
        &second_prefix_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == 32u);
    assert(second_prefix_block_count == 2u);
    cold_second_prefix_block_index = second_prefix_block_table[0u];
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_second_prefix_block_index) == SPARK_STATUS_OK);
    fixture.prefetch_capture.call_count = 0u;
    fixture.prefetch_capture.busy_call_budget = 1u;
    fixture.prefetch_capture.return_status = SPARK_STATUS_OK;

    SparkTestInitializeSubmitRequest(
        &request,
        960u,
        9960u,
        50u,
        first_prompt,
        48u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &first_handle) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,
        961u,
        9961u,
        50u,
        second_prompt,
        48u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &second_handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(dispatch.request_count == 2u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV) != 0u);
    assert(fixture.prefetch_capture.call_count == 2u);
    assert(dispatch.kv_prefetch_plan.prefetch_block_count == 1u);
    assert(dispatch.kv_prefetch_plan.blocks[0u].physical_block_index ==
        cold_second_prefix_block_index);
    assert(dispatch.prefill_batch_decision.active_sequence_count == 2u);
    assert(dispatch.prefill_batch_decision.maximum_scheduled_prompt_token_count ==
        16u);
    assert((fixture.kv_blocks[cold_second_prefix_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        second_handle) == SPARK_STATUS_OK);
}


static void SparkTestRequestApiAdaptivePrefillChoosesFullResidentBucketOverOlderSingleton(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle singleton_handle;
    SparkGlm52RequestApiHandle batch_handles[16u];
    uint32_t singleton_prompt[48u];
    uint32_t batch_prompts[16u][16u];
    uint32_t request_index;

    SparkTestFillTokenIds(singleton_prompt, 48u, 130000u);
    for (request_index = 0u; request_index < 16u; ++request_index)
    {
        SparkTestFillTokenIds(
            batch_prompts[request_index],
            16u,
            140000u + request_index * 1000u);
    }
    SparkTestInitializeFixture(&fixture);

    SparkTestInitializeSubmitRequest(
        &request,
        1300u,
        11300u,
        10u,
        singleton_prompt,
        48u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &singleton_handle) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 16u; ++request_index)
    {
        SparkTestInitializeSubmitRequest(
            &request,
            1400u + request_index,
            11400u + request_index,
            10u,
            batch_prompts[request_index],
            16u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &batch_handles[request_index]) == SPARK_STATUS_OK);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(dispatch.request_count == 16u);
    assert(dispatch.prefill_batch_decision.active_sequence_count == 16u);
    assert(dispatch.prefill_batch_decision.batch_bucket ==
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(dispatch.prefill_batch_decision.graph_sequence_padding_count == 0u);
    assert(dispatch.prefill_batch_decision.maximum_scheduled_prompt_token_count == 16u);
    assert(dispatch.prefill_batch_decision.total_scheduled_token_count == 256u);
    for (request_index = 0u; request_index < dispatch.request_count; ++request_index)
    {
        assert(dispatch.request_handles[request_index] != singleton_handle);
    }
    assert(fixture.api.request_slots[0].state ==
        SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);
}

static void SparkTestRequestApiRealtimePrefillBypassesFullBulkBatch(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle batch_handles[16u];
    SparkGlm52RequestApiHandle realtime_handle;
    uint32_t batch_prompts[16u][16u];
    uint32_t realtime_prompt[48u];
    uint32_t request_index;

    for (request_index = 0u; request_index < 16u; ++request_index)
    {
        SparkTestFillTokenIds(
            batch_prompts[request_index],
            16u,
            150000u + request_index * 1000u);
    }
    SparkTestFillTokenIds(realtime_prompt, 48u, 170000u);
    SparkTestInitializeFixture(&fixture);

    for (request_index = 0u; request_index < 16u; ++request_index)
    {
        SparkTestInitializeSubmitRequest(
            &request,
            1500u + request_index,
            11500u + request_index,
            10u,
            batch_prompts[request_index],
            16u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &batch_handles[request_index]) == SPARK_STATUS_OK);
    }

    SparkTestInitializeSubmitRequest(
        &request,
        1700u,
        11700u,
        0u,
        realtime_prompt,
        48u,
        1u);
    request.flags = SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &realtime_handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0] == realtime_handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE) != 0u);
    assert(fixture.api.request_slots[16u].state ==
        SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL);
    (void)batch_handles;
}



static void SparkTestRequestApiDecodeBatchUsesMeasuredB64ForSeventeenReadyRequests(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiCacheState cache_state;
    SparkGlm52RequestApiHandle handles[17u];
    uint32_t prompts[17u][16u];
    uint32_t request_index;
    uint32_t ready_count;
    uint32_t iteration_count;

    SparkTestInitializeFixture(&fixture);
    for (request_index = 0u; request_index < 17u; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],
            16u,
            200000u + request_index * 1000u);
        SparkTestInitializeSubmitRequest(
            &request,
            2000u + request_index,
            12000u + request_index,
            10u,
            prompts[request_index],
            16u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    ready_count = 0u;
    for (iteration_count = 0u; iteration_count < 8u && ready_count < 17u; ++iteration_count)
    {
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
        assert(dispatch.accepted == 1u);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
            dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);

        ready_count = 0u;
        for (request_index = 0u; request_index < 17u; ++request_index)
        {
            assert(SparkGlm52RequestApiGetRequestCacheState(
                &fixture.api,
                handles[request_index],
                &cache_state) == SPARK_STATUS_OK);
            if (cache_state.state == SPARK_GLM52_REQUEST_API_STATE_READY_DECODE)
            {
                ready_count += 1u;
            }
        }
    }
    assert(ready_count == 17u);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 17u);
    assert(dispatch.decode_batch_decision.active_sequence_count == 17u);
    assert(dispatch.decode_batch_decision.batch_bucket ==
        SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(dispatch.decode_batch_decision.graph_sequence_padding_count == 47u);
    assert((dispatch.decode_batch_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u);
    assert((dispatch.decode_batch_decision.stage_decision.dispatch_stages[0u].dispatch_flags &
        SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET) != 0u);
    assert(fixture.scheduler.measured_decode_bucket_selection_count == 1u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiAsyncJitPrefetchHoldsDispatchUntilResident(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[32u];
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t cold_physical_block_index;

    SparkTestFillTokenIds(prompt, 32u, 180000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        prompt,
        32u);
    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        prompt,
        32u,
        physical_block_indices,
        4u,
        &matched_token_count,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == 16u || matched_token_count == 32u);
    assert(physical_block_count != 0u);
    cold_physical_block_index = physical_block_indices[0u];
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_physical_block_index) == SPARK_STATUS_OK);
    SparkTestEnableAsyncPrefetch(&fixture);
    fixture.prefetch_capture.async_poll_busy_budget = 1u;

    SparkTestInitializeSubmitRequest(
        &request,
        1800u,
        11800u,
        80u,
        prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_BUSY);
    assert(dispatch.accepted == 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING) != 0u);
    assert(fixture.prefetch_capture.async_start_count == 1u);
    assert(fixture.prefetch_capture.async_poll_count == 1u);
    assert(fixture.prefetch_capture.async_pending != 0u);
    assert(fixture.api.async_jit_prefetch_start_count == 1u);
    assert(fixture.api.async_jit_prefetch_completion_count == 0u);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert(fixture.api.request_slots[0u].state ==
        SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0u] == handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV) != 0u);
    assert(fixture.prefetch_capture.async_start_count == 1u);
    assert(fixture.prefetch_capture.async_poll_count == 2u);
    assert(fixture.prefetch_capture.async_pending == 0u);
    assert(fixture.api.async_jit_prefetch_completion_count == 1u);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiDoesNotGreenlightMissingJitKv(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[32u];
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t cold_physical_block_index;

    SparkTestFillTokenIds(prompt, 32u, 90000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        prompt,
        32u);
    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        prompt,
        32u,
        physical_block_indices,
        4u,
        &matched_token_count,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 1u);
    cold_physical_block_index = physical_block_indices[0];
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_physical_block_index) == SPARK_STATUS_OK);
    fixture.prefetch_capture.return_status = SPARK_STATUS_BUSY;

    SparkTestInitializeSubmitRequest(
        &request,
        900u,
        9900u,
        70u,
        prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_BUSY);
    assert(dispatch.accepted == 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING) != 0u);
    assert(fixture.api.request_slots[0].state ==
        SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    (void)handle;
}



static void SparkTestRequestApiTrimsResidentKvWithoutEvictingNearFuturePrefix(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t cold_prompt[32u];
    uint32_t hot_prompt[32u];
    uint32_t cold_blocks[4u];
    uint32_t hot_blocks[4u];
    uint32_t matched_token_count;
    uint32_t cold_block_count;
    uint32_t hot_block_count;

    SparkTestFillTokenIds(cold_prompt, 32u, 131000u);
    SparkTestFillTokenIds(hot_prompt, 32u, 132000u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.max_resident_kv_block_count = 2u;

    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        cold_prompt,
        32u);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        hot_prompt,
        32u);
    assert(fixture.kv_arena.resident_block_count == 4u);

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        cold_prompt,
        32u,
        cold_blocks,
        4u,
        &matched_token_count,
        &cold_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(cold_block_count == 1u);
    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        hot_prompt,
        32u,
        hot_blocks,
        4u,
        &matched_token_count,
        &hot_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(hot_block_count == 1u);

    SparkTestInitializeSubmitRequest(
        &request,
        1310u,
        11310u,
        90u,
        hot_prompt,
        32u,
        0u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0u] == handle);
    assert(dispatch.prefill_decision.cached_prefix_token_count ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(fixture.api.jit_residency_eviction_count == 2u);
    assert(fixture.kv_arena.resident_block_count == 2u);
    assert((fixture.kv_blocks[hot_blocks[0u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((fixture.kv_blocks[cold_blocks[0u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}


static void SparkTestRequestApiEvictsColdResidentBlocksButKeepsLookaheadHotset(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t cold_prompt[32u];
    uint32_t hot_prompt[32u];
    uint32_t cold_block_table[4u];
    uint32_t hot_block_table[4u];
    uint32_t matched_token_count;
    uint32_t cold_block_count;
    uint32_t hot_block_count;
    uint32_t cold_reusable_block;
    uint32_t hot_reusable_block;

    SparkTestFillTokenIds(cold_prompt, 32u, 125000u);
    SparkTestFillTokenIds(hot_prompt, 32u, 126000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        cold_prompt,
        32u);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        hot_prompt,
        32u);

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        cold_prompt,
        32u,
        cold_block_table,
        4u,
        &matched_token_count,
        &cold_block_count) == SPARK_STATUS_OK);
    assert(cold_block_count == 1u);
    cold_reusable_block = cold_block_table[0u];
    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        hot_prompt,
        32u,
        hot_block_table,
        4u,
        &matched_token_count,
        &hot_block_count) == SPARK_STATUS_OK);
    assert(hot_block_count == 1u);
    hot_reusable_block = hot_block_table[0u];
    assert(cold_reusable_block != hot_reusable_block);
    assert((fixture.kv_blocks[cold_reusable_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((fixture.kv_blocks[hot_reusable_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);

    fixture.api.max_resident_kv_block_count = 1u;
    SparkTestInitializeSubmitRequest(
        &request,
        1250u,
        11250u,
        90u,
        hot_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0u] == handle);
    assert(fixture.api.jit_residency_eviction_count != 0u);
    assert(fixture.api.jit_residency_protected_block_count != 0u);
    assert((fixture.kv_blocks[hot_reusable_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((fixture.kv_blocks[cold_reusable_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert(fixture.kv_arena.resident_block_count <=
        fixture.api.max_resident_kv_block_count);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}


static void SparkTestRequestApiReuseScoredEvictionKeepsSharedPrefixFamily(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle first_hot_handle;
    SparkGlm52RequestApiHandle second_hot_handle;
    SparkGlm52RequestApiHandle cold_handle;
    uint32_t hot_prompt[32u];
    uint32_t cold_prompt[32u];
    uint32_t hot_block_table[4u];
    uint32_t cold_block_table[4u];
    uint32_t matched_token_count;
    uint32_t hot_block_count;
    uint32_t cold_block_count;
    uint32_t hot_reusable_block;
    uint32_t cold_reusable_block;

    SparkTestFillTokenIds(hot_prompt, 32u, 141000u);
    SparkTestFillTokenIds(cold_prompt, 32u, 142000u);
    SparkTestInitializeFixture(&fixture);

    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        hot_prompt,
        32u);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        cold_prompt,
        32u);
    assert(fixture.kv_arena.resident_block_count == 4u);
    fixture.api.max_resident_kv_block_count = 1u;

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        hot_prompt,
        32u,
        hot_block_table,
        4u,
        &matched_token_count,
        &hot_block_count) == SPARK_STATUS_OK);
    assert(hot_block_count == 1u);
    hot_reusable_block = hot_block_table[0u];
    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        cold_prompt,
        32u,
        cold_block_table,
        4u,
        &matched_token_count,
        &cold_block_count) == SPARK_STATUS_OK);
    assert(cold_block_count == 1u);
    cold_reusable_block = cold_block_table[0u];
    assert(hot_reusable_block != cold_reusable_block);

    SparkTestInitializeSubmitRequest(
        &request,
        1410u,
        11410u,
        50u,
        hot_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &first_hot_handle) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,
        1411u,
        11411u,
        50u,
        hot_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &second_hot_handle) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,
        1420u,
        11420u,
        50u,
        cold_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &cold_handle) == SPARK_STATUS_OK);

    {
        SparkStatus schedule_status;

        schedule_status = SparkGlm52RequestApiScheduleNext(
            &fixture.api,
            &dispatch);
        assert(schedule_status == SPARK_STATUS_OK ||
            schedule_status == SPARK_STATUS_BUSY);
    }
    assert(fixture.prefix_cache.reuse_scored_resident_eviction_count >= 3u);
    assert(fixture.prefix_cache.reuse_scored_lookahead_eviction_count != 0u);
    assert((fixture.kv_blocks[hot_reusable_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((fixture.kv_blocks[cold_reusable_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert(fixture.kv_arena.resident_block_count == 1u);

    if (dispatch.accepted != 0u)
    {
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
    }
    (void)first_hot_handle;
    (void)second_hot_handle;
    (void)cold_handle;
}


static void SparkTestRequestApiUsesBuiltInAsyncMemoryPrefetchBackend(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[32u];
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t cold_physical_block_index;

    SparkTestFillTokenIds(prompt, 32u, 131000u);
    SparkTestInitializeFixtureWithAsyncMemoryBackend(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        prompt,
        32u);

    assert(SparkGlm52PrefixCacheProbePhysicalBlockTable(
        &fixture.prefix_cache,
        prompt,
        32u,
        physical_block_indices,
        4u,
        &matched_token_count,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(matched_token_count == SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(physical_block_count == 1u);
    cold_physical_block_index = physical_block_indices[0u];
    memset(fixture.key_destination[cold_physical_block_index], 0, 64u);
    memset(fixture.value_destination[cold_physical_block_index], 0, 64u);
    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &fixture.kv_arena,
        cold_physical_block_index) == SPARK_STATUS_OK);

    SparkTestInitializeSubmitRequest(
        &request,
        1310u,
        11310u,
        900u,
        prompt,
        32u,
        1u);
    request.flags = SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.request_handles[0u] == handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV) != 0u);
    assert(fixture.async_prefetch_backend.started_prefetch_count == 1u);
    assert(fixture.async_prefetch_backend.completed_prefetch_count == 1u);
    assert(fixture.async_prefetch_backend.copied_key_block_count == 1u);
    assert(fixture.async_prefetch_backend.copied_value_block_count == 1u);
    assert(memcmp(fixture.key_destination[cold_physical_block_index],
        fixture.key_source[cold_physical_block_index],
        64u) == 0);
    assert(memcmp(fixture.value_destination[cold_physical_block_index],
        fixture.value_source[cold_physical_block_index],
        64u) == 0);
    assert((fixture.kv_blocks[cold_physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiRefreshesQueueAwarePrefixProtection(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[32u];
    uint64_t baseline_protection_sweep_count;
    uint64_t baseline_protected_block_count;

    SparkTestFillTokenIds(prompt, 32u, 121000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestRequestApiWarmsPrefixCacheAndReleases(
        &fixture,
        prompt,
        32u);
    baseline_protection_sweep_count =
        fixture.api.lookahead_protection_sweep_count;
    baseline_protected_block_count =
        fixture.api.lookahead_protected_block_count;

    SparkTestInitializeSubmitRequest(
        &request,
        1210u,
        11210u,
        80u,
        prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0u] == handle);
    assert(fixture.api.lookahead_protection_sweep_count ==
        baseline_protection_sweep_count + 1u);
    assert(fixture.api.lookahead_protected_block_count >
        baseline_protected_block_count);
    assert(fixture.prefix_cache.lookahead_protection_epoch != 0u);
    assert(fixture.prefix_cache.lookahead_protected_block_count != 0u);

    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}

int main(void)
{
    SparkTestRequestApiJitPrefetchesCachedPrefixForPriorityRequest();
    SparkTestRequestApiBatchesReadyDecodeRequestsAndConsumesBudgets();
    SparkTestRequestApiCohortsSamePromptRequestsAndSharesBlocks();
    SparkTestRequestApiCohortsArbitrarySharedPrefixWithSuffixes();
    SparkTestRequestApiChoosesWiderSharedPrefixFamily();
    SparkTestRequestApiBatchesVariableOneBlockSuffixes();
    SparkTestRequestApiAdaptivePrefillChoosesFullResidentBucketOverOlderSingleton();
    SparkTestRequestApiRealtimePrefillBypassesFullBulkBatch();
    SparkTestRequestApiDecodeBatchUsesMeasuredB64ForSeventeenReadyRequests();
    SparkTestRequestApiOpportunisticLookaheadDoesNotBlockReadyPriorityPrefill();
    SparkTestRequestApiPrefetchesLiveNonresidentDecodeBlocks();
    SparkTestRequestApiBatchesDecodeAfterBatchCriticalPrefetch();
    SparkTestRequestApiBatchesPrefillAfterBatchCriticalPrefetch();
    SparkTestRequestApiAsyncJitPrefetchHoldsDispatchUntilResident();
    SparkTestRequestApiDoesNotGreenlightMissingJitKv();
    SparkTestRequestApiTrimsResidentKvWithoutEvictingNearFuturePrefix();
    SparkTestRequestApiEvictsColdResidentBlocksButKeepsLookaheadHotset();
    SparkTestRequestApiReuseScoredEvictionKeepsSharedPrefixFamily();
    SparkTestRequestApiUsesBuiltInAsyncMemoryPrefetchBackend();
    SparkTestRequestApiRefreshesQueueAwarePrefixProtection();
    return 0;
}
