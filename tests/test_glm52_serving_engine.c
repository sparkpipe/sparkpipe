#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_serving_engine.h"

#define SPARK_TEST_SERVING_REQUEST_SLOT_COUNT 4u
#define SPARK_TEST_SERVING_REQUEST_RECORD_COUNT 4u
#define SPARK_TEST_SERVING_REQUEST_TOKEN_STRIDE 128u
#define SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE 64u
#define SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT 97u
#define SPARK_TEST_SERVING_KV_BLOCK_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_TEST_SERVING_PREFIX_ENTRY_COUNT \
    SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY
#define SPARK_TEST_SERVING_PREFIX_BINDING_COUNT \
    (SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY + 8u)
#define SPARK_TEST_SERVING_EVENT_CAPACITY 16384u
#define SPARK_TEST_SERVING_LANE_CAPACITY \
    SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT

typedef struct SparkTestServingFixture
{
    SparkGlm52KvCacheArena kv_arena;
    SparkGlm52KvCacheBlock kv_blocks[SPARK_TEST_SERVING_KV_BLOCK_COUNT];
    SparkGlm52PrefixCache prefix_cache;
    SparkGlm52PrefixCacheEntry prefix_entries[
        SPARK_TEST_SERVING_PREFIX_ENTRY_COUNT];
    SparkGlm52PrefixCacheSequenceBinding prefix_bindings[
        SPARK_TEST_SERVING_PREFIX_BINDING_COUNT];
    SparkGlm52Scheduler scheduler;
    SparkGlm52RequestApiSlot request_slots[
        SPARK_TEST_SERVING_REQUEST_SLOT_COUNT];
    SparkGlm52RequestApi request_api;
    SparkGlm52ServingEngine serving_engine;
    SparkGlm52ServingRequestRecord request_records[
        SPARK_TEST_SERVING_REQUEST_RECORD_COUNT];
    uint32_t request_token_storage[
        SPARK_TEST_SERVING_REQUEST_RECORD_COUNT *
        SPARK_TEST_SERVING_REQUEST_TOKEN_STRIDE];
    SparkGlm52ServingEvent event_ring[SPARK_TEST_SERVING_EVENT_CAPACITY];
    uint32_t host_prefill_token_ids[
        SPARK_TEST_SERVING_LANE_CAPACITY *
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE];
    uint32_t physical_block_indices[
        SPARK_TEST_SERVING_LANE_CAPACITY *
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    uint32_t lane_physical_block_counts[SPARK_TEST_SERVING_LANE_CAPACITY];
    uint32_t prompt_tokens[SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT];
    uint32_t expected_prompt_tokens[SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT];
} SparkTestServingFixture;

typedef struct SparkTestServingCallbackContext
{
    const uint32_t *expected_prompt_tokens;
    const uint32_t *alternate_expected_prompt_tokens;
    uint64_t alternate_request_id;
    uint32_t prefill_callback_count;
    uint32_t decode_callback_count;
    uint32_t largest_prefill_lane_count;
    uint32_t saw_decode_kv_table;
} SparkTestServingCallbackContext;

static SparkTestServingFixture Fixture;
static SparkTestServingCallbackContext CallbackContext;

static void SparkTestServingFillTokenIds(
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

static SparkStatus SparkTestServingKvPrefetch(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    (void)context;
    assert(prefetch_plan != 0);
    assert(prefetch_plan->abi_version == SPARK_GLM52_KV_CACHE_ABI_VERSION);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServingPrefill(
    void *context,
    const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
    SparkTestServingCallbackContext *callback_context;
    uint32_t lane_index;

    callback_context = (SparkTestServingCallbackContext *)context;
    assert(callback_context != 0);
    assert(prefill_dispatch != 0);
    assert(prefill_dispatch->host_token_ids != 0);
    assert(prefill_dispatch->prefill_view != 0);
    assert(prefill_dispatch->kv_block_table_view != 0);
    assert(prefill_dispatch->kv_block_table_view->host_physical_block_indices != 0);
    assert(prefill_dispatch->kv_block_table_view->lane_count ==
        prefill_dispatch->lane_count);
    assert(prefill_dispatch->lane_count != 0u);
    assert(prefill_dispatch->prompt_token_count != 0u);
    assert(prefill_dispatch->prompt_token_count <=
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE);

    if (prefill_dispatch->lane_count > callback_context->largest_prefill_lane_count)
    {
        callback_context->largest_prefill_lane_count = prefill_dispatch->lane_count;
    }

    for (lane_index = 0u;
         lane_index < prefill_dispatch->lane_count;
         ++lane_index)
    {
        const SparkGlm52RequestApiPrefillDispatchLaneView *lane;
        const uint32_t *expected_tokens;
        const uint32_t *lane_tokens;
        uint32_t token_index;

        lane = &prefill_dispatch->prefill_view->lanes[lane_index];
        expected_tokens = callback_context->expected_prompt_tokens;
        if (callback_context->alternate_expected_prompt_tokens != 0 &&
            lane->request_id == callback_context->alternate_request_id)
        {
            expected_tokens = callback_context->alternate_expected_prompt_tokens;
        }
        lane_tokens = &prefill_dispatch->host_token_ids[
            (uint64_t)lane_index * prefill_dispatch->host_token_stride];
        for (token_index = 0u;
             token_index < lane->prompt_token_count;
             ++token_index)
        {
            assert(lane_tokens[token_index] ==
                expected_tokens[lane->prompt_token_offset + token_index]);
        }
    }

    callback_context->prefill_callback_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServingDecode(
    void *context,
    const SparkGlm52ServingDecodeDispatch *decode_dispatch,
    SparkGlm52ServingDecodeResult *decode_result)
{
    SparkTestServingCallbackContext *callback_context;
    uint32_t lane_index;

    callback_context = (SparkTestServingCallbackContext *)context;
    assert(callback_context != 0);
    assert(decode_dispatch != 0);
    assert(decode_dispatch->request_dispatch != 0);
    assert(decode_dispatch->kv_block_table_view != 0);
    assert(decode_dispatch->decode_view != 0);
    assert(decode_dispatch->kv_block_table_view->lane_count ==
        decode_dispatch->request_count);
    assert(decode_dispatch->decode_view->lane_count ==
        decode_dispatch->request_count);
    assert(decode_result != 0);

    SparkGlm52ServingInitializeDecodeResult(
        decode_result,
        decode_dispatch->request_count,
        SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE);
    for (lane_index = 0u;
         lane_index < decode_dispatch->request_count;
         ++lane_index)
    {
        if (callback_context->decode_callback_count == 0u)
        {
            assert(decode_dispatch->input_token_ids[lane_index] ==
                callback_context->expected_prompt_tokens[
                    SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT - 1u]);
            assert(decode_dispatch->decode_view->lanes[lane_index].
                sequence_position == SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT);
        }
        else
        {
            assert(decode_dispatch->input_token_ids[lane_index] ==
                90000u + callback_context->decode_callback_count - 1u +
                lane_index);
            assert(decode_dispatch->decode_view->lanes[lane_index].
                sequence_position ==
                SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT +
                callback_context->decode_callback_count);
        }
        decode_result->token_counts[lane_index] = 1u;
        decode_result->token_ids[lane_index][0u] =
            90000u + callback_context->decode_callback_count + lane_index;
    }
    callback_context->decode_callback_count += 1u;
    callback_context->saw_decode_kv_table = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestServingMtpDecode(
    void *context,
    const SparkGlm52ServingDecodeDispatch *decode_dispatch,
    SparkGlm52ServingDecodeResult *decode_result)
{
    SparkTestServingCallbackContext *callback_context;
    uint32_t lane_index;
    uint32_t expected_budget;

    callback_context = (SparkTestServingCallbackContext *)context;
    assert(callback_context != 0);
    assert(decode_dispatch != 0);
    assert(decode_dispatch->request_dispatch != 0);
    assert((decode_dispatch->request_dispatch->flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u);
    expected_budget = callback_context->decode_callback_count == 0u ? 2u : 1u;
    assert(decode_dispatch->request_dispatch->mtp_draft_token_budget ==
        expected_budget);
    SparkGlm52ServingInitializeDecodeResult(
        decode_result,
        decode_dispatch->request_count,
        SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE);
    for (lane_index = 0u;
         lane_index < decode_dispatch->request_count;
         ++lane_index)
    {
        decode_result->token_counts[lane_index] = 2u;
        decode_result->token_ids[lane_index][0u] =
            90000u + (callback_context->decode_callback_count * 10u);
        decode_result->token_ids[lane_index][1u] =
            90001u + (callback_context->decode_callback_count * 10u);
    }
    callback_context->decode_callback_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkTestServingInitializeFixture(
    SparkTestServingFixture *fixture,
    SparkTestServingCallbackContext *callback_context)
{
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration prefix_configuration;
    SparkGlm52SchedulerConfiguration scheduler_configuration;
    SparkGlm52RequestApiConfiguration request_api_configuration;
    SparkGlm52ServingEngineConfiguration serving_configuration;

    memset(fixture, 0, sizeof(*fixture));
    memset(callback_context, 0, sizeof(*callback_context));
    SparkTestServingFillTokenIds(
        fixture->prompt_tokens,
        SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT,
        81000u);
    memcpy(
        fixture->expected_prompt_tokens,
        fixture->prompt_tokens,
        sizeof(fixture->expected_prompt_tokens));
    callback_context->expected_prompt_tokens = fixture->expected_prompt_tokens;

    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count = SPARK_TEST_SERVING_KV_BLOCK_COUNT;
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
    prefix_configuration.entry_count = SPARK_TEST_SERVING_PREFIX_ENTRY_COUNT;
    prefix_configuration.physical_block_count = SPARK_TEST_SERVING_KV_BLOCK_COUNT;
    prefix_configuration.sequence_binding_count =
        SPARK_TEST_SERVING_PREFIX_BINDING_COUNT;
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
    scheduler_configuration.max_prefill_tokens_per_step =
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE;
    scheduler_configuration.prefix_cache = &fixture->prefix_cache;
    assert(SparkGlm52SchedulerInitialize(
        &fixture->scheduler,
        &scheduler_configuration) == SPARK_STATUS_OK);

    memset(&request_api_configuration, 0, sizeof(request_api_configuration));
    request_api_configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    request_api_configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    request_api_configuration.request_capacity = SPARK_TEST_SERVING_REQUEST_SLOT_COUNT;
    request_api_configuration.prefetch_lane_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    request_api_configuration.decode_batch_target = 2u;
    request_api_configuration.scheduler = &fixture->scheduler;
    request_api_configuration.request_slots = fixture->request_slots;
    request_api_configuration.kv_prefetch_function = SparkTestServingKvPrefetch;
    assert(SparkGlm52RequestApiInitialize(
        &fixture->request_api,
        &request_api_configuration) == SPARK_STATUS_OK);

    memset(&serving_configuration, 0, sizeof(serving_configuration));
    serving_configuration.abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    serving_configuration.descriptor_bytes =
        SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES;
    serving_configuration.runtime_contract_flags =
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS |
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED |
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY;
    serving_configuration.default_output_token_budget = 2u;
    serving_configuration.default_max_prefill_tokens_per_step =
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE;
    serving_configuration.max_context_tokens = 256u;
    serving_configuration.request_api = &fixture->request_api;
    serving_configuration.request_records = fixture->request_records;
    serving_configuration.request_record_capacity =
        SPARK_TEST_SERVING_REQUEST_RECORD_COUNT;
    serving_configuration.request_token_storage = fixture->request_token_storage;
    serving_configuration.request_token_stride =
        SPARK_TEST_SERVING_REQUEST_TOKEN_STRIDE;
    serving_configuration.event_ring = fixture->event_ring;
    serving_configuration.event_ring_capacity = SPARK_TEST_SERVING_EVENT_CAPACITY;
    serving_configuration.host_prefill_token_ids = fixture->host_prefill_token_ids;
    serving_configuration.host_prefill_token_stride =
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE;
    serving_configuration.host_prefill_lane_capacity =
        SPARK_TEST_SERVING_LANE_CAPACITY;
    serving_configuration.host_physical_block_indices =
        fixture->physical_block_indices;
    serving_configuration.kv_block_lane_stride =
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    serving_configuration.kv_block_lane_capacity =
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    serving_configuration.lane_physical_block_counts =
        fixture->lane_physical_block_counts;
    serving_configuration.lane_count_capacity = SPARK_TEST_SERVING_LANE_CAPACITY;
    serving_configuration.prefill_function = SparkTestServingPrefill;
    serving_configuration.decode_function = SparkTestServingDecode;
    serving_configuration.callback_context = callback_context;
    assert(SparkGlm52ServingEngineInitialize(
        &fixture->serving_engine,
        &serving_configuration) == SPARK_STATUS_OK);
}

static void SparkTestServingRejectsTailWindowRuntimeContract(void)
{
    SparkGlm52ServingEngineConfiguration serving_configuration;
    SparkGlm52ServingEngine serving_engine;

    SparkTestServingInitializeFixture(&Fixture, &CallbackContext);
    memset(&serving_configuration, 0, sizeof(serving_configuration));
    serving_configuration.abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    serving_configuration.descriptor_bytes =
        SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES;
    serving_configuration.runtime_contract_flags =
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS |
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY;
    serving_configuration.request_api = &Fixture.request_api;
    serving_configuration.request_records = Fixture.request_records;
    serving_configuration.request_record_capacity =
        SPARK_TEST_SERVING_REQUEST_RECORD_COUNT;
    serving_configuration.request_token_storage = Fixture.request_token_storage;
    serving_configuration.request_token_stride =
        SPARK_TEST_SERVING_REQUEST_TOKEN_STRIDE;
    serving_configuration.event_ring = Fixture.event_ring;
    serving_configuration.event_ring_capacity = SPARK_TEST_SERVING_EVENT_CAPACITY;
    serving_configuration.host_prefill_token_ids = Fixture.host_prefill_token_ids;
    serving_configuration.host_prefill_token_stride =
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE;
    serving_configuration.host_prefill_lane_capacity =
        SPARK_TEST_SERVING_LANE_CAPACITY;
    serving_configuration.host_physical_block_indices =
        Fixture.physical_block_indices;
    serving_configuration.kv_block_lane_stride =
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    serving_configuration.kv_block_lane_capacity =
        SPARK_GLM52_SCHEDULER_KV_BLOCK_TABLE_CAPACITY;
    serving_configuration.lane_physical_block_counts =
        Fixture.lane_physical_block_counts;
    serving_configuration.lane_count_capacity = SPARK_TEST_SERVING_LANE_CAPACITY;
    serving_configuration.prefill_function = SparkTestServingPrefill;
    serving_configuration.decode_function = SparkTestServingDecode;
    serving_configuration.callback_context = &CallbackContext;
    assert(SparkGlm52ServingEngineInitialize(
        &serving_engine,
        &serving_configuration) == SPARK_STATUS_INVALID_ARGUMENT);

    serving_configuration.runtime_contract_flags =
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
    Fixture.request_api.configuration_flags &=
        ~SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING;
    assert(SparkGlm52ServingEngineInitialize(
        &serving_engine,
        &serving_configuration) == SPARK_STATUS_INVALID_ARGUMENT);

    Fixture.request_api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING;
    serving_configuration.runtime_contract_flags &=
        ~SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE;
    assert(SparkGlm52ServingEngineInitialize(
        &serving_engine,
        &serving_configuration) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestServingFireAndForgetPumpRunsFullPromptToDecode(void)
{
    SparkGlm52ServingSubmitTokenIdsRequest submit_request;
    SparkGlm52ServingSubmitResult submit_result;
    SparkGlm52ServingStats stats;
    SparkGlm52ServingEvent event;
    uint32_t accepted_event_count;
    uint32_t prefill_event_count;
    uint32_t token_event_count;
    uint32_t completion_event_count;
    uint32_t token_index;
    SparkStatus status;

    SparkTestServingInitializeFixture(&Fixture, &CallbackContext);
    SparkGlm52ServingInitializeSubmitTokenIdsRequest(&submit_request);
    submit_request.token_count = SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT;
    submit_request.token_ids = Fixture.prompt_tokens;
    submit_request.request_id = 9001u;
    submit_request.sequence_id = 19001u;
    assert(SparkGlm52ServingEngineSubmitTokenIds(
        &Fixture.serving_engine,
        &submit_request,
        &submit_result) == SPARK_STATUS_OK);
    assert(submit_result.request_handle != 0u);
    assert(submit_result.output_token_budget == 2u);

    for (token_index = 0u;
         token_index < SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT;
         ++token_index)
    {
        Fixture.prompt_tokens[token_index] = 1u;
    }

    status = SparkGlm52ServingEnginePump(
        &Fixture.serving_engine,
        0u,
        16u,
        &stats);
    assert(status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_OK);
    assert(CallbackContext.prefill_callback_count == 2u);
    assert(CallbackContext.decode_callback_count == 2u);
    assert(CallbackContext.saw_decode_kv_table == 1u);
    assert(stats.prefill_dispatch_count == 2u);
    assert(stats.prefill_token_count == SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT);
    assert(stats.decode_dispatch_count == 2u);
    assert(stats.decoded_token_count == 2u);

    accepted_event_count = 0u;
    prefill_event_count = 0u;
    token_event_count = 0u;
    completion_event_count = 0u;
    while (SparkGlm52ServingEnginePopEvent(
            &Fixture.serving_engine,
            &event) == SPARK_STATUS_OK)
    {
        assert(event.request_id == 9001u);
        assert(event.sequence_id == 19001u);
        if (event.kind == SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_ACCEPTED)
        {
            accepted_event_count += 1u;
        }
        else if (event.kind == SPARK_GLM52_SERVING_EVENT_KIND_PREFILL_PROGRESS)
        {
            prefill_event_count += 1u;
        }
        else if (event.kind == SPARK_GLM52_SERVING_EVENT_KIND_TOKEN)
        {
            token_event_count += 1u;
            assert(event.token_id == 90000u || event.token_id == 90001u);
        }
        else if (event.kind == SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_COMPLETED)
        {
            completion_event_count += 1u;
        }
    }
    assert(accepted_event_count == 1u);
    assert(prefill_event_count == 2u);
    assert(token_event_count == 2u);
    assert(completion_event_count == 1u);
}

static void SparkTestServingPrefillBatchingIsInternal(void)
{
    SparkGlm52ServingSubmitTokenIdsRequest submit_request;
    SparkGlm52ServingSubmitResult submit_result;
    uint32_t second_prompt[SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE];
    SparkStatus status;

    SparkTestServingInitializeFixture(&Fixture, &CallbackContext);
    SparkGlm52ServingInitializeSubmitTokenIdsRequest(&submit_request);
    submit_request.token_count = SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE;
    submit_request.token_ids = Fixture.prompt_tokens;
    submit_request.output_token_budget = 1u;
    submit_request.request_id = 9101u;
    submit_request.sequence_id = 19101u;
    assert(SparkGlm52ServingEngineSubmitTokenIds(
        &Fixture.serving_engine,
        &submit_request,
        &submit_result) == SPARK_STATUS_OK);

    SparkTestServingFillTokenIds(
        second_prompt,
        SPARK_TEST_SERVING_PREFILL_TOKEN_STRIDE,
        82000u);
    CallbackContext.alternate_expected_prompt_tokens = second_prompt;
    CallbackContext.alternate_request_id = 9102u;
    submit_request.token_ids = second_prompt;
    submit_request.request_id = 9102u;
    submit_request.sequence_id = 19102u;
    assert(SparkGlm52ServingEngineSubmitTokenIds(
        &Fixture.serving_engine,
        &submit_request,
        &submit_result) == SPARK_STATUS_OK);

    status = SparkGlm52ServingEnginePump(
        &Fixture.serving_engine,
        SPARK_GLM52_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH,
        1u,
        0);
    assert(status == SPARK_STATUS_OK);
    assert(CallbackContext.prefill_callback_count == 1u);
    assert(CallbackContext.largest_prefill_lane_count == 2u);
}

static void SparkTestServingMtpCommitStreamsMultiTokenLanes(void)
{
    SparkGlm52ServingSubmitTokenIdsRequest submit_request;
    SparkGlm52ServingSubmitResult submit_result;
    SparkGlm52ServingStats stats;
    SparkGlm52ServingEvent event;
    uint32_t token_event_count;
    uint32_t completion_event_count;
    SparkStatus status;

    SparkTestServingInitializeFixture(&Fixture, &CallbackContext);
    Fixture.request_api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT;
    Fixture.serving_engine.decode_function = SparkTestServingMtpDecode;
    SparkGlm52ServingInitializeSubmitTokenIdsRequest(&submit_request);
    submit_request.token_count = SPARK_TEST_SERVING_PROMPT_TOKEN_COUNT;
    submit_request.token_ids = Fixture.prompt_tokens;
    submit_request.output_token_budget = 4u;
    submit_request.request_id = 9301u;
    submit_request.sequence_id = 19301u;
    assert(SparkGlm52ServingEngineSubmitTokenIds(
        &Fixture.serving_engine,
        &submit_request,
        &submit_result) == SPARK_STATUS_OK);
    assert(submit_result.output_token_budget == 4u);

    status = SparkGlm52ServingEnginePump(
        &Fixture.serving_engine,
        0u,
        16u,
        &stats);
    assert(status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_OK);
    assert(CallbackContext.decode_callback_count == 2u);
    assert(stats.decode_dispatch_count == 2u);
    assert(stats.decoded_token_count == 4u);

    token_event_count = 0u;
    completion_event_count = 0u;
    while (SparkGlm52ServingEnginePopEvent(
            &Fixture.serving_engine,
            &event) == SPARK_STATUS_OK)
    {
        if (event.kind == SPARK_GLM52_SERVING_EVENT_KIND_TOKEN)
        {
            token_event_count += 1u;
            assert(event.token_id == 90000u || event.token_id == 90001u ||
                event.token_id == 90010u || event.token_id == 90011u);
        }
        else if (event.kind == SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_COMPLETED)
        {
            completion_event_count += 1u;
        }
    }
    assert(token_event_count == 4u);
    assert(completion_event_count == 1u);
}

int main(void)
{
    SparkTestServingRejectsTailWindowRuntimeContract();
    SparkTestServingMtpCommitStreamsMultiTokenLanes();
    SparkTestServingFireAndForgetPumpRunsFullPromptToDecode();
    SparkTestServingPrefillBatchingIsInternal();
    return 0;
}
