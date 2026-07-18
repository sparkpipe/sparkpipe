#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "sparkpipe/spark_glm52_text_prompt.h"
#include "sparkpipe/spark_glm52_request_api.h"
#include "sparkpipe/spark_tokenizer.h"

#define SPARK_TEST_REQUEST_SLOT_COUNT 32u
#define SPARK_TEST_PREFIX_ENTRY_COUNT 128u
#define SPARK_TEST_PREFIX_BINDING_COUNT 512u
#define SPARK_TEST_KV_BLOCK_COUNT 128u
#define SPARK_TEST_KV_BLOCK_PAYLOAD_BYTES 64u
#define SPARK_TEST_DSPARK_SEQUENCE_STATE_COUNT SPARK_TEST_REQUEST_SLOT_COUNT
#define SPARK_TEST_TEXT_TOKEN_A 1u
#define SPARK_TEST_TEXT_TOKEN_B 2u
#define SPARK_TEST_TEXT_TOKEN_C 3u
#define SPARK_TEST_TEXT_TOKEN_AB 4u
#define SPARK_TEST_TEXT_TOKEN_ABC 5u
#define SPARK_TEST_TEXT_TOKEN_UNKNOWN 6u

typedef struct SparkTestDsparkDraftCapture
{
    uint32_t call_count;
    uint32_t requested_token_count;
    uint32_t priority;
    uint32_t next_token_id;
    uint32_t confidence_milli[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t tap_generation;
} SparkTestDsparkDraftCapture;

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
    SparkGlm52DsparkSpeculator dspark_speculator;
    SparkGlm52DsparkSequenceState dspark_sequence_states[
        SPARK_TEST_DSPARK_SEQUENCE_STATE_COUNT];
    SparkTestDsparkDraftCapture dspark_capture;
    SparkGlm52DsparkModelContract dspark_model_contract;
    SparkTestPrefetchCapture prefetch_capture;
    uint8_t key_destination[SPARK_TEST_KV_BLOCK_COUNT][
        SPARK_TEST_KV_BLOCK_PAYLOAD_BYTES];
    uint8_t value_destination[SPARK_TEST_KV_BLOCK_COUNT][
        SPARK_TEST_KV_BLOCK_PAYLOAD_BYTES];
    uint8_t key_source[SPARK_TEST_KV_BLOCK_COUNT][
        SPARK_TEST_KV_BLOCK_PAYLOAD_BYTES];
    uint8_t value_source[SPARK_TEST_KV_BLOCK_COUNT][
        SPARK_TEST_KV_BLOCK_PAYLOAD_BYTES];
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


static SparkStatus SparkTestDsparkDraft(
    void *context,
    const SparkGlm52DsparkDraftRequest *request,
    SparkGlm52DsparkDraftResult *result)
{
    SparkTestDsparkDraftCapture *capture;
    uint32_t token_index;

    capture = (SparkTestDsparkDraftCapture *)context;
    assert(capture != 0);
    assert(request != 0);
    assert(result != 0);
    assert(request->requested_token_count != 0u);
    assert(request->requested_token_count <=
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);
    assert(request->sequence_id != 0u);
    assert(request->tap_generation != 0u);

    capture->call_count += 1u;
    capture->requested_token_count = request->requested_token_count;
    capture->priority = request->priority;
    capture->request_id = request->request_id;
    capture->sequence_id = request->sequence_id;
    capture->sequence_position = request->sequence_position;
    capture->tap_generation = request->tap_generation;

    result->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    result->descriptor_bytes = SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
    result->token_count = request->requested_token_count;
    for (token_index = 0u;
         token_index < request->requested_token_count;
         ++token_index)
    {
        uint32_t confidence_milli;

        confidence_milli = capture->confidence_milli[token_index];
        if (confidence_milli == 0u)
        {
            confidence_milli = 800u;
        }
        result->token_ids[token_index] = capture->next_token_id + token_index;
        result->confidence_milli[token_index] = confidence_milli;
    }
    return SPARK_STATUS_OK;
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
    kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
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
    scheduler_configuration.configuration_flags =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    scheduler_configuration.prefix_cache_block_tokens =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
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
    kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
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
    scheduler_configuration.configuration_flags =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    scheduler_configuration.prefix_cache_block_tokens =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
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


static void SparkTestEnableDsparkSpeculation(
    SparkTestRequestApiFixture *fixture)
{
    SparkGlm52DsparkSpeculatorConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    assert(SparkGlm52DsparkBuildDefaultModelContract(
        &fixture->dspark_model_contract) == SPARK_STATUS_OK);
    configuration.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.sequence_state_count = SPARK_TEST_DSPARK_SEQUENCE_STATE_COUNT;
    configuration.default_speculative_token_count =
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    configuration.minimum_confidence_milli = 350u;
    configuration.realtime_minimum_confidence_milli = 250u;
    configuration.sequence_states = fixture->dspark_sequence_states;
    configuration.draft_function = SparkTestDsparkDraft;
    configuration.draft_context = &fixture->dspark_capture;
    configuration.model_contract = &fixture->dspark_model_contract;
    assert(SparkGlm52DsparkInitialize(
        &fixture->dspark_speculator,
        &configuration) == SPARK_STATUS_OK);

    fixture->dspark_capture.next_token_id = 140000u;
    fixture->api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE;
    fixture->api.dspark_speculator = &fixture->dspark_speculator;
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

static const char *SparkTestRequestApiTokenizerJsonPath(void)
{
    return "build/test_glm52_request_api_tokenizer.json";
}

static void SparkTestRequestApiWriteTokenizerJson(void)
{
    FILE *file;

    file = fopen(SparkTestRequestApiTokenizerJsonPath(), "wb");
    assert(file != 0);
    fprintf(file,
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"unk_token\": \"<unk>\",\n"
        "    \"byte_fallback\": false,\n"
        "    \"vocab\": {\n"
        "      \"a\": %u,\n"
        "      \"b\": %u,\n"
        "      \"c\": %u,\n"
        "      \"ab\": %u,\n"
        "      \"abc\": %u,\n"
        "      \"<unk>\": %u\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"a b\",\n"
        "      \"ab c\"\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"ByteLevel\",\n"
        "    \"add_prefix_space\": false\n"
        "  },\n"
        "  \"added_tokens\": []\n"
        "}\n",
        SPARK_TEST_TEXT_TOKEN_A,
        SPARK_TEST_TEXT_TOKEN_B,
        SPARK_TEST_TEXT_TOKEN_C,
        SPARK_TEST_TEXT_TOKEN_AB,
        SPARK_TEST_TEXT_TOKEN_ABC,
        SPARK_TEST_TEXT_TOKEN_UNKNOWN);
    assert(fclose(file) == 0);
}

static void SparkTestRequestApiLoadTokenizer(SparkTokenizer *tokenizer)
{
    SparkTokenizerHuggingFaceJsonConfiguration configuration;

    SparkTokenizerReset(tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = SparkTestRequestApiTokenizerJsonPath();
    assert(SparkTokenizerLoadHuggingFaceJson(tokenizer, &configuration) ==
        SPARK_STATUS_OK);
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
    uint32_t decode_block_tables[2u][4u];
    uint32_t decode_block_counts[2u];

    SparkTestFillTokenIds(first_prompt, 16u, 30000u);
    SparkTestFillTokenIds(second_prompt, 16u, 40000u);
    SparkTestInitializeFixture(&fixture);

    SparkTestInitializeSubmitRequest(
        &request,
        11u,
        5011u,
        20u,
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
    assert(first_dispatch.request_handles[0] == first_handle);
    assert(first_dispatch.request_handles[1] == second_handle);
    assert(first_dispatch.decode_batch_decision.active_sequence_count == 2u);
    assert((first_dispatch.decode_batch_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) != 0u);
    memset(decode_block_tables, 0, sizeof(decode_block_tables));
    memset(decode_block_counts, 0, sizeof(decode_block_counts));
    assert(SparkGlm52RequestApiBuildDispatchKvBlockTables(
        &fixture.api,
        &first_dispatch,
        &decode_block_tables[0u][0u],
        4u,
        4u,
        decode_block_counts,
        2u) == SPARK_STATUS_OK);
    assert(decode_block_counts[0u] == 2u);
    assert(decode_block_counts[1u] == 2u);
    assert(decode_block_tables[0u][0u] != decode_block_tables[0u][1u]);
    assert(decode_block_tables[1u][0u] != decode_block_tables[1u][1u]);
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

static void SparkTestRequestApiFillsDecodeBatchBeforeEqualPriorityDecode(void)
{
	SparkTestRequestApiFixture fixture;
	SparkGlm52RequestApiSubmitRequest request;
	SparkGlm52RequestApiDispatch dispatch;
	SparkGlm52RequestApiHandle handle;
	uint32_t prompts[4u][16u];
	uint32_t request_index;

	SparkTestInitializeFixture(&fixture);
	fixture.api.configuration_flags &=
		~SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING;
	fixture.api.decode_batch_target = 4u;
	for (request_index = 0u; request_index < 4u; ++request_index)
	{
		SparkTestFillTokenIds(
			prompts[request_index],16u,50000u + (request_index * 100u));
		SparkTestInitializeSubmitRequest(
			&request,
			100u + request_index,
			6000u + request_index,
			100u,
			prompts[request_index],
			16u,
			2u);
		assert(SparkGlm52RequestApiSubmit(
			&fixture.api,&request,&handle) == SPARK_STATUS_OK);
	}
	for (request_index = 0u; request_index < 4u; ++request_index)
	{
		assert(SparkGlm52RequestApiScheduleNext(
			&fixture.api,&dispatch) == SPARK_STATUS_OK);
		assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
		assert(SparkGlm52RequestApiCompleteDispatch(
			&fixture.api,&dispatch) == SPARK_STATUS_OK);
	}
	assert(SparkGlm52RequestApiScheduleNext(
		&fixture.api,&dispatch) == SPARK_STATUS_OK);
	assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
	assert(dispatch.request_count == 4u);
	assert(dispatch.decode_batch_decision.active_sequence_count == 4u);
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

static void SparkTestRequestApiWidePrefixFamilyCannotBeatHigherPriority(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle low_handles[4u];
    SparkGlm52RequestApiHandle high_handle;
    uint32_t low_prompt[32u];
    uint32_t high_prompt[32u];
    uint32_t request_index;

    SparkTestFillTokenIds(low_prompt, 32u, 61000u);
    memcpy(high_prompt, low_prompt, sizeof(high_prompt));
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
    for (request_index = 0u; request_index < 4u; ++request_index)
    {
        SparkTestInitializeSubmitRequest(
            &request,
            200u + request_index,
            7200u + request_index,
            10u,
            low_prompt,
            32u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &low_handles[request_index]) == SPARK_STATUS_OK);
    }
    SparkTestInitializeSubmitRequest(
        &request,
        300u,
        7300u,
        100u,
        high_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &high_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0u] == high_handle);
    assert(dispatch.shared_prefix_token_count == 0u);
    for (request_index = 0u; request_index < 4u; ++request_index)
    {
        assert(fixture.api.request_slots[request_index].state ==
            SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL);
    }
}

static void SparkTestRequestApiLateHighPriorityUsesNextAvailablePipelineSlot(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch low_dispatch;
    SparkGlm52RequestApiDispatch high_dispatch;
    SparkGlm52RequestApiHandle low_handle;
    SparkGlm52RequestApiHandle high_handle;
    uint32_t low_prompt[16u];
    uint32_t high_prompt[16u];

    SparkTestFillTokenIds(low_prompt, 16u, 63000u);
    SparkTestFillTokenIds(high_prompt, 16u, 64000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestInitializeSubmitRequest(
        &request,
        400u,
        7400u,
        10u,
        low_prompt,
        16u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &low_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &low_dispatch) == SPARK_STATUS_OK);
    assert(low_dispatch.request_handles[0u] == low_handle);
    SparkTestInitializeSubmitRequest(
        &request,
        401u,
        7401u,
        100u,
        high_prompt,
        16u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &high_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &high_dispatch) == SPARK_STATUS_OK);
    assert(high_dispatch.accepted == 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &low_dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &high_dispatch) == SPARK_STATUS_OK);
    assert(high_dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
        high_dispatch.kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(high_dispatch.request_count == 1u);
    assert(high_dispatch.request_handles[0u] == high_handle);
    assert(fixture.api.request_slots[0u].state ==
        SPARK_GLM52_REQUEST_API_STATE_READY_DECODE);
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
    assert(block_table_view.host_lane_physical_block_counts == batch_block_counts);
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

static void SparkTestRequestApiAsyncJitPrefetchOverlapsResidentWork(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle high_priority_handle;
    SparkGlm52RequestApiHandle low_priority_handle;
    uint32_t prompt[32u];
    uint32_t low_priority_prompt[32u];
    uint32_t matched_token_count;
    uint32_t physical_block_count;
    uint32_t physical_block_indices[4u];
    uint32_t cold_physical_block_index;

    SparkTestFillTokenIds(prompt, 32u, 180000u);
    SparkTestFillTokenIds(low_priority_prompt, 32u, 181000u);
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
        &high_priority_handle) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,
        1801u,
        11801u,
        10u,
        low_priority_prompt,
        32u,
        1u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &low_priority_handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.request_handles[0u] == low_priority_handle);
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
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_handles[0u] == high_priority_handle);
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
        high_priority_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        low_priority_handle) == SPARK_STATUS_OK);
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
    memset(
        fixture.key_destination[cold_physical_block_index],
        0,
        sizeof(fixture.key_destination[cold_physical_block_index]));
    memset(
        fixture.value_destination[cold_physical_block_index],
        0,
        sizeof(fixture.value_destination[cold_physical_block_index]));
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
        sizeof(fixture.key_destination[cold_physical_block_index])) == 0);
    assert(memcmp(fixture.value_destination[cold_physical_block_index],
        fixture.value_source[cold_physical_block_index],
        sizeof(fixture.value_destination[cold_physical_block_index])) == 0);
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


static void SparkTestRequestApiDsparkCapturesTapsAndRunsSpeculativeVerify(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];

    SparkTestFillTokenIds(prompt, 16u, 150000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestEnableDsparkSpeculation(&fixture);

    SparkTestInitializeSubmitRequest(
        &request,
        1500u,
        11500u,
        SPARK_GLM52_REQUEST_API_REALTIME_PRIORITY,
        prompt,
        16u,
        10u);
    request.flags = SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0u] == handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(fixture.dspark_capture.call_count == 1u);
    assert(fixture.dspark_capture.requested_token_count ==
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);
    assert(fixture.dspark_capture.sequence_id == 11500u);
    assert(fixture.dspark_capture.sequence_position == 17u);
    assert(fixture.api.dspark_tap_capture_dispatch_count == 2u);
    assert(fixture.api.dspark_draft_ready_count == 1u);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0u] == handle);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u);
    assert(dispatch.speculative_token_count ==
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);
    assert(dispatch.speculative_max_committed_token_count ==
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u);
    assert(dispatch.speculative_draft_token_ids[0u][0u] == 140000u);
    assert(dispatch.speculative_draft_token_ids[0u][6u] == 140006u);
    assert(dispatch.speculative_confidence_milli[0u][0u] == 800u);

    {
        uint32_t verifier_tokens[
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u] = {
            140000u, 140001u, 140002u, 141111u,
            141112u, 141113u, 141114u, 141115u
        };
        assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
            &fixture.api,
            &dispatch,
            verifier_tokens,
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u,
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u) ==
            SPARK_STATUS_OK);
    }
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(fixture.api.dspark_verify_dispatch_count == 1u);
    assert(fixture.api.dspark_accepted_draft_token_count == 3u);
    assert(fixture.api.dspark_committed_token_count == 4u);
    assert(fixture.api.dspark_rejected_token_count == 4u);
    assert(fixture.api.completed_request_count == 0u);
    assert(fixture.dspark_capture.call_count == 2u);
    assert(fixture.dspark_capture.requested_token_count == 5u);
    assert(fixture.dspark_capture.sequence_position == 21u);

    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiDsparkBatchesEqualLengthDrafts(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handles[3u];
    uint32_t prompts[3u][16u];
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    SparkTestEnableDsparkSpeculation(&fixture);

    for (request_index = 0u; request_index < 3u; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],
            16u,
            151000u + request_index * 100u);
        SparkTestInitializeSubmitRequest(
            &request,
            1510u + request_index,
            11510u + request_index,
            SPARK_GLM52_REQUEST_API_REALTIME_PRIORITY - request_index,
            prompts[request_index],
            16u,
            8u);
        request.flags = SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME;
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    request_index = 0u;
    while (request_index < 3u)
    {
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
            dispatch.kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
        request_index += dispatch.request_count;
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 3u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(fixture.dspark_capture.call_count == 3u);
    assert(fixture.api.dspark_draft_ready_count == 3u);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.request_count == 3u);
    assert(dispatch.speculative_token_count == 7u);
    {
        uint32_t verifier_tokens[3u][
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u];
        uint32_t token_index;

        for (request_index = 0u; request_index < 3u; ++request_index)
        {
            for (token_index = 0u;
                 token_index <
                    SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT - 1u;
                 ++token_index)
            {
                verifier_tokens[request_index][token_index] =
                    dispatch.speculative_draft_token_ids[request_index][token_index];
            }
            verifier_tokens[request_index][
                SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT - 1u] =
                148000u + request_index;
            verifier_tokens[request_index][
                SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT] =
                149000u + request_index;
        }
        assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
            &fixture.api,
            &dispatch,
            &verifier_tokens[0u][0u],
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u,
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT + 1u) ==
            SPARK_STATUS_OK);
    }
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(fixture.api.completed_request_count == 3u);
    assert(fixture.api.dspark_accepted_draft_token_count == 18u);
    assert(fixture.api.dspark_committed_token_count == 21u);
}

static void SparkTestRequestApiDescribesAndCopiesFullPrefillTokenWindows(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiPrefillDispatchView prefill_view;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[97u];
    uint32_t copied_tokens[64u];
    uint32_t token_index;

    SparkTestFillTokenIds(prompt, 97u, 160000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestInitializeSubmitRequest(
        &request,
        1600u,
        11600u,
        SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,
        97u,
        1u);
    request.max_prefill_tokens_per_step = 64u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiDescribePrefillDispatch(
        &dispatch,
        &prefill_view) == SPARK_STATUS_OK);
    assert(prefill_view.abi_version == SPARK_GLM52_REQUEST_API_ABI_VERSION);
    assert(prefill_view.lane_count == 1u);
    assert(prefill_view.active_sequence_count == 1u);
    assert(prefill_view.prompt_token_offset == 0u);
    assert(prefill_view.prompt_token_count == 64u);
    assert(prefill_view.prompt_token_stride == 64u);
    assert(prefill_view.lanes[0u].prompt_token_ids == prompt);
    memset(copied_tokens, 0xa5, sizeof(copied_tokens));
    assert(SparkGlm52RequestApiCopyPrefillDispatchTokenIds(
        &dispatch,
        copied_tokens,
        64u,
        1u) == SPARK_STATUS_OK);
    for (token_index = 0u; token_index < 64u; ++token_index)
    {
        assert(copied_tokens[token_index] == prompt[token_index]);
    }
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiDescribePrefillDispatch(
        &dispatch,
        &prefill_view) == SPARK_STATUS_OK);
    assert(prefill_view.lane_count == 1u);
    assert(prefill_view.prompt_token_offset == 64u);
    assert(prefill_view.prompt_token_count == 33u);
    assert(prefill_view.prompt_token_stride == 33u);
    memset(copied_tokens, 0xa5, sizeof(copied_tokens));
    assert(SparkGlm52RequestApiCopyPrefillDispatchTokenIds(
        &dispatch,
        copied_tokens,
        64u,
        1u) == SPARK_STATUS_OK);
    for (token_index = 0u; token_index < 33u; ++token_index)
    {
        assert(copied_tokens[token_index] ==
            prompt[64u + token_index]);
    }
    for (token_index = 33u; token_index < 64u; ++token_index)
    {
        assert(copied_tokens[token_index] == 0u);
    }
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(SparkGlm52RequestApiDescribePrefillDispatch(
        &dispatch,
        &prefill_view) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiDsparkDisabledPerRequestFallsBackToDecode(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];

    SparkTestFillTokenIds(prompt, 16u, 152000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestEnableDsparkSpeculation(&fixture);

    SparkTestInitializeSubmitRequest(
        &request,
        1520u,
        11520u,
        SPARK_GLM52_REQUEST_API_REALTIME_PRIORITY,
        prompt,
        16u,
        2u);
    request.flags =
        SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME |
        SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) == 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(fixture.dspark_capture.call_count == 0u);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}

static void SparkTestFillMtpTreeCandidates(
    uint32_t *candidate_token_ids,
    uint32_t token_seed)
{
    uint32_t token_index;
    for (token_index = 0u;
         token_index < SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
         ++token_index)
    {
        candidate_token_ids[token_index] = token_seed + token_index;
    }
}

static void SparkTestFillMtpTreeVerifier(
    const uint32_t *candidate_token_ids,
    uint32_t path_id,
    uint32_t token_seed,
    uint32_t *verifier_token_ids)
{
    uint32_t row_index;
    for (row_index = 0u;
         row_index < SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
         ++row_index)
    {
        verifier_token_ids[row_index] = token_seed + 100u + row_index;
    }
    if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE)
        return;
    verifier_token_ids[SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW] =
        candidate_token_ids[SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX];
    if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1)
        return;
    if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE)
    {
        verifier_token_ids[SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] =
            candidate_token_ids[
                SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX];
        return;
    }
    verifier_token_ids[SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW] =
        candidate_token_ids[SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX];
    if (path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_PRIMARY)
        return;
    verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW] =
        candidate_token_ids[
            path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY
                ? SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX
                : SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX];
}

static void SparkTestMtpTreeResolvesEveryPath(void)
{
    SparkGlm52MtpTreeResolution resolution;
    static const uint32_t expected_candidate_indices[
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT] = {
        SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX,
        SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX,
        SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX,
        SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX,
        SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX,
        SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX
    };
    static const uint32_t expected_parent_rows[
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT] = {
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_INPUT_ROW,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH1_ROW,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW
    };
    static const uint32_t expected_base_offsets[
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT] = {0u,0u,1u,1u,2u,2u};
    uint32_t candidate_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];
    uint32_t verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];
    uint32_t path_id;

    SparkTestFillMtpTreeCandidates(candidate_token_ids,90000u);
    for (path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
         path_id < SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT;
         ++path_id)
    {
        SparkTestFillMtpTreeVerifier(
            candidate_token_ids,path_id,91000u,verifier_token_ids);
        assert(SparkGlm52MtpTreeResolve(
            candidate_token_ids,verifier_token_ids,&resolution) ==
            SPARK_STATUS_OK);
        assert(resolution.path_id == path_id);
        assert(resolution.accepted_token_count ==
            SparkGlm52MtpTreeAcceptedTokenCount(path_id));
        assert(resolution.committed_token_count ==
            resolution.accepted_token_count + 1u);
        assert(resolution.fallback_row_index ==
            SparkGlm52MtpTreeFallbackRowIndex(path_id));
        assert(SparkGlm52MtpTreeTailCandidateIndex(path_id) ==
            expected_candidate_indices[path_id]);
        assert(SparkGlm52MtpTreeTailParentRowIndex(path_id) ==
            expected_parent_rows[path_id]);
        assert(SparkGlm52MtpTreeTailBasePositionOffset(path_id) ==
            expected_base_offsets[path_id]);
    }
}

static void SparkTestMtpTreeUsesCompactAlternateStorage(void)
{
    assert(SPARK_GLM52_MODEL_MTP_TREE_BRANCH_ROW_COUNT == 4u);
    assert(SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT == 2u);
    assert(SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT == 2u);
    assert(SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_ALTERNATE_INDEX == 0u);
    assert(SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_ALTERNATE_INDEX == 1u);
    assert(SparkGlm52MtpTreeVerifierPositionOffset(
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_PRIMARY_ROW) == 2u);
    assert(SparkGlm52MtpTreeVerifierPositionOffset(
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW) == 2u);
    assert(SparkGlm52MtpTreeVerifierPositionOffset(
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_PRIMARY_ROW) == 3u);
    assert(SparkGlm52MtpTreeVerifierPositionOffset(
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW) == 3u);
}

static void SparkTestRequestApiMtpDraftRequiresSpeculativeVerify(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiCacheState cache_state;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];
    uint32_t draft_token_ids[SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT];
    uint32_t verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];
    uint32_t verify_block_table[4u];
    uint32_t verify_block_count[1u];
    uint32_t token_index;

    SparkTestFillTokenIds(prompt, 16u, 153000u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    SparkTestInitializeSubmitRequest(
        &request,
        1530u,
        11530u,
        SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,
        16u,
        5u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u);
    assert(dispatch.mtp_draft_token_budget ==
        SPARK_GLM52_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT);

    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,91000u);
    SparkTestFillMtpTreeVerifier(
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE,
        92000u,
        verifier_token_ids);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,
        &dispatch,
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    assert(fixture.api.mtp_draft_ready_count == 1u);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u);
    assert(dispatch.speculative_token_count ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(dispatch.speculative_verifier_token_count ==
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT);
    assert(dispatch.mtp_draft_token_budget ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    memset(verify_block_table, 0, sizeof(verify_block_table));
    memset(verify_block_count, 0, sizeof(verify_block_count));
    assert(SparkGlm52RequestApiBuildDispatchKvBlockTables(
        &fixture.api,
        &dispatch,
        verify_block_table,
        4u,
        4u,
        verify_block_count,
        1u) == SPARK_STATUS_OK);
    assert(verify_block_count[0u] == 2u);
    assert(verify_block_table[0u] != verify_block_table[1u]);
    for (token_index = 0u;
         token_index < dispatch.speculative_token_count;
         ++token_index)
    {
        assert(dispatch.speculative_draft_token_ids[0u][token_index] ==
            draft_token_ids[token_index]);
    }
    assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        &fixture.api,
        &dispatch,
        verifier_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) ==
        SPARK_STATUS_OK);
    assert(dispatch.speculative_accepted_token_counts[0u] == 3u);
    assert(dispatch.speculative_committed_token_counts[0u] == 4u);
    assert(dispatch.speculative_fallback_token_ids[0u] ==
        verifier_token_ids[
            SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW]);
    assert(dispatch.speculative_resolution_path_ids[0u] ==
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiGetRequestCacheState(
        &fixture.api,
        handle,
        &cache_state) == SPARK_STATUS_OK);
    assert(cache_state.state == SPARK_GLM52_REQUEST_API_STATE_COMPLETED);
    assert(fixture.api.request_slots[0u].completed_decode_token_count ==
        5u);
    assert(fixture.api.request_slots[0u].mtp_draft_token_count == 0u);
    assert(fixture.api.mtp_verify_dispatch_count == 1u);
    assert(fixture.api.mtp_accepted_draft_token_count ==
        3u);
    assert(fixture.api.mtp_committed_token_count ==
        4u);
}

static void SparkTestRequestApiMtpVerifyCapturesDsparkBatchTap(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];
    uint32_t draft_token_ids[SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];
    uint32_t verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];

    SparkTestFillTokenIds(prompt,16u,153200u);
    SparkTestInitializeFixture(&fixture);
    SparkTestEnableDsparkSpeculation(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    fixture.dspark_speculator.policy_flags = 0u;
    SparkTestInitializeSubmitRequest(
        &request,
        1532u,
        11532u,
        SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,
        16u,
        20u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) == 0u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u);
    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,93000u);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,
        &dispatch,
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    fixture.dspark_speculator.policy_flags =
        SPARK_GLM52_DSPARK_POLICY_DEFAULT_FLAGS;
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.request_count == 1u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) == 0u);
    SparkTestFillMtpTreeVerifier(
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH1,
        94000u,
        verifier_token_ids);
    assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        &fixture.api,
        &dispatch,
        verifier_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(fixture.dspark_capture.call_count == 1u);
    assert(fixture.api.dspark_draft_ready_count == 1u);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiUsesHigherYieldMtpBeforeEqualPriorityDecode(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle first_handle;
    SparkGlm52RequestApiHandle second_handle;
    uint32_t first_prompt[16u];
    uint32_t second_prompt[16u];
    uint32_t draft_token_ids[SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];

    SparkTestFillTokenIds(first_prompt,16u,154000u);
    SparkTestFillTokenIds(second_prompt,16u,155000u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    fixture.api.decode_batch_target = 1u;
    SparkTestInitializeSubmitRequest(
        &request,1540u,11540u,10u,first_prompt,16u,10u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&first_handle) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,1550u,11550u,10u,second_prompt,16u,2u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&second_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
        dispatch.kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(dispatch.request_handles[0u] == first_handle);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_handles[0u] == first_handle);
    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,93000u);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,&dispatch,draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    fixture.api.request_slots[1u].priority = 11u;
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
        dispatch.kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(dispatch.request_handles[0u] == second_handle);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    fixture.api.request_slots[1u].priority = 10u;
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.request_handles[0u] == first_handle);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,second_handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiMtpDraftBudgetRemainsTransactional(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];
    uint32_t draft_token_ids[SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];
    uint32_t verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];

    SparkTestFillTokenIds(prompt,16u,153500u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    SparkTestInitializeSubmitRequest(
        &request,1535u,11535u,SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,16u,20u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.mtp_draft_token_budget ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,93000u);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,&dispatch,draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.mtp_draft_token_budget ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(SparkGlm52RequestApiRetryDecodeDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(fixture.api.request_slots[0u].state ==
        SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY);
    assert(fixture.api.request_slots[0u].mtp_draft_token_count ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    SparkTestFillMtpTreeVerifier(
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE,
        94000u,
        verifier_token_ids);
    assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        &fixture.api,&dispatch,verifier_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) == SPARK_STATUS_OK);
    assert(dispatch.speculative_accepted_token_counts[0u] == 2u);
    assert(dispatch.speculative_committed_token_counts[0u] == 3u);
    assert(dispatch.speculative_resolution_path_ids[0u] ==
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,95000u);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,&dispatch,draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.request_handles[0u] == handle);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiPrefillWaveSpansMultipleKvBlocks(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiPrefillDispatchView prefill_view;
    SparkGlm52RequestApiHandle handle;
    static uint32_t prompt[300u];

    SparkTestFillTokenIds(prompt,300u,120000u);
    SparkTestInitializeFixture(&fixture);
    SparkTestInitializeSubmitRequest(
        &request,1570u,11570u,SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,300u,8u);
    request.thinking_token_budget = 0u;
    request.max_prefill_tokens_per_step = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiDescribePrefillDispatch(
        &dispatch,&prefill_view) == SPARK_STATUS_OK);
    assert(prefill_view.prompt_token_offset == 0u);
    assert(prefill_view.prompt_token_count ==
        SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiDescribePrefillDispatch(
        &dispatch,&prefill_view) == SPARK_STATUS_OK);
    assert(prefill_view.prompt_token_offset ==
        SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH);
    assert(prefill_view.prompt_token_count ==
        300u - SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiMtpAdaptiveFloorSuppressesAndRecovers(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiSlot *slot;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];
    uint32_t draft_token_ids[SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];
    uint32_t verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];
    uint32_t cycle_index,plain_index,failing_cycle_count;
    SparkStatus status;

    SparkTestFillTokenIds(prompt,16u,155000u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    SparkTestInitializeSubmitRequest(
        &request,1560u,11560u,SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,16u,128u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&handle) == SPARK_STATUS_OK);
    slot = &fixture.request_slots[0u];
    assert(slot->mtp_commit_ema_milli ==
        SPARK_GLM52_REQUEST_API_MTP_COMMIT_EMA_INITIAL_MILLI);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    failing_cycle_count = 0u;
    for (cycle_index = 0u; cycle_index < 32u; ++cycle_index)
    {
        SparkTestFillMtpTreeCandidates(draft_token_ids,96000u);
        SparkTestFillMtpTreeVerifier(
            draft_token_ids,
            SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE,
            97000u,
            verifier_token_ids);
        status = SparkGlm52RequestApiArmMtpVerifyDispatch(
            &fixture.api,&dispatch,draft_token_ids,
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
        if (status == SPARK_STATUS_NOT_FOUND)
            break;
        assert(status == SPARK_STATUS_OK);
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
        assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
            &fixture.api,&dispatch,verifier_token_ids,
            SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
            SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) == SPARK_STATUS_OK);
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
        failing_cycle_count += 1u;
    }
    assert(failing_cycle_count >= 4u && failing_cycle_count < 32u);
    assert(slot->mtp_commit_ema_milli <
        SPARK_GLM52_REQUEST_API_MTP_SUPPRESS_THRESHOLD_MILLI);
    assert(slot->mtp_next_draft_token_budget == 0u);
    assert(slot->mtp_probe_countdown ==
        SPARK_GLM52_REQUEST_API_MTP_REPROBE_INTERVAL);
    for (plain_index = 0u;
         plain_index < SPARK_GLM52_REQUEST_API_MTP_REPROBE_INTERVAL;
         ++plain_index)
    {
        assert(slot->mtp_next_draft_token_budget == 0u);
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
        assert(dispatch.mtp_draft_token_budget == 0u);
        dispatch.decode_committed_token_counts[0u] = 1u;
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
    }
    assert(slot->mtp_probe_countdown == 0u);
    assert(slot->mtp_next_draft_token_budget ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.mtp_draft_token_budget ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,98000u);
    SparkTestFillMtpTreeVerifier(
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY,
        99000u,
        verifier_token_ids);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,&dispatch,draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        &fixture.api,&dispatch,verifier_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) == SPARK_STATUS_OK);
    assert(dispatch.speculative_committed_token_counts[0u] ==
        SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(slot->mtp_commit_ema_milli >=
        SPARK_GLM52_REQUEST_API_MTP_SUPPRESS_THRESHOLD_MILLI);
    assert(slot->mtp_next_draft_token_budget ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiMtpRejectedDraftStaysOutsideNextContext(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiDecodeDispatchView decode_view;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];
    uint32_t draft_token_ids[SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];
    uint32_t verifier_token_ids[
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];

    SparkTestFillTokenIds(prompt,16u,155000u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    SparkTestInitializeSubmitRequest(
        &request,1550u,11550u,SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,16u,20u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    dispatch.decode_committed_token_counts[0u] = 1u;
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    SparkTestFillMtpTreeCandidates(draft_token_ids,96000u);
    SparkTestFillMtpTreeVerifier(
        draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE,
        97000u,
        verifier_token_ids);
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,&dispatch,draft_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        &fixture.api,&dispatch,verifier_token_ids,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) == SPARK_STATUS_OK);
    assert(dispatch.speculative_accepted_token_counts[0u] == 0u);
    assert(dispatch.speculative_committed_token_counts[0u] == 1u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiDescribeDecodeDispatch(
        &fixture.api,&dispatch,&decode_view) == SPARK_STATUS_OK);
    assert(decode_view.lanes[0u].sequence_position == 17u);
    assert(decode_view.lanes[0u].context_token_count == 18u);
    assert(decode_view.lanes[0u].mtp_resolution_base_position == 16u);
    assert(decode_view.lanes[0u].mtp_resolution_proposed_token_count ==
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT);
    assert(decode_view.lanes[0u].mtp_resolution_accepted_token_count == 0u);
    assert(decode_view.lanes[0u].mtp_resolution_committed_token_count == 1u);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiMtpBudgetLeavesVerifierFallbackHeadroom(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[16u];

    SparkTestFillTokenIds(prompt,16u,154000u);
    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    SparkTestInitializeSubmitRequest(
        &request,
        1540u,
        11540u,
        SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,
        16u,
        3u);
    request.thinking_token_budget = 0u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u);
    assert(dispatch.mtp_draft_token_budget == 0u);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,handle) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiMtpVerifyCapsPackedExecutionRows(void)
{
    enum {
        request_count = 2u,
        execution_row_budget = 7u,
        decode_batch_target = 2u,
        utility_sample_count = SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
    };
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handles[request_count];
    uint32_t prompts[request_count][16u];
    uint32_t draft_token_ids[request_count][
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT];
    uint32_t verifier_token_ids[request_count][
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT];
    uint32_t completed_prefill_count;
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    fixture.api.decode_batch_target = decode_batch_target;
    fixture.api.decode_execution_row_capacity = execution_row_budget;
    fixture.api.mtp_accepted_draft_token_count =
        3u * utility_sample_count;
    fixture.api.mtp_rejected_token_count =
        2u * utility_sample_count;
    fixture.api.mtp_committed_token_count =
        4u * utility_sample_count;
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],
            16u,
            154000u + request_index * 100u);
        SparkTestInitializeSubmitRequest(
            &request,
            1540u + request_index,
            11540u + request_index,
            SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
            prompts[request_index],
            16u,
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT + 2u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
    }

    completed_prefill_count = 0u;
    while (completed_prefill_count < request_count)
    {
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
            dispatch.kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
        completed_prefill_count += dispatch.request_count;
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        fixture.api.request_slots[request_index].mtp_next_draft_token_budget =
            SPARK_GLM52_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT;
    }

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == request_count);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u);
    for (request_index = 0u; request_index < dispatch.request_count;
         ++request_index)
    {
        dispatch.decode_committed_token_counts[request_index] = 1u;
    }
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        SparkTestFillMtpTreeCandidates(
            draft_token_ids[request_index],
            92000u + request_index * 100u);
        SparkTestFillMtpTreeVerifier(
            draft_token_ids[request_index],
            SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_PRIMARY,
            93000u + request_index * 100u,
            verifier_token_ids[request_index]);
    }
    assert(SparkGlm52RequestApiArmMtpVerifyDispatch(
        &fixture.api,
        &dispatch,
        &draft_token_ids[0u][0u],
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) ==
        SPARK_STATUS_OK);

    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH);
    assert(dispatch.request_count == execution_row_budget /
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT);
    assert(dispatch.request_count *
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT <=
        execution_row_budget);
    assert(SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        &fixture.api,
        &dispatch,
        &verifier_token_ids[0u][0u],
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);

    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        SparkStatus cancel_status;
        cancel_status = SparkGlm52RequestApiCancelRequest(
            &fixture.api,
            handles[request_index]);
        assert(cancel_status == SPARK_STATUS_OK ||
            cancel_status == SPARK_STATUS_NOT_FOUND);
    }
}

static void SparkTestRequestApiSkipsUnprofitableWideMtp(void)
{
    enum { request_count = 16u };
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handles[request_count];
    uint32_t prompts[request_count][16u];
    uint32_t completed_prefill_count;
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT;
    fixture.api.decode_batch_target = request_count;
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],16u,180000u + request_index * 100u);
        SparkTestInitializeSubmitRequest(
            &request,
            1800u + request_index,
            11800u + request_index,
            SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
            prompts[request_index],
            16u,
            8u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,&request,&handles[request_index]) == SPARK_STATUS_OK);
    }
    completed_prefill_count = 0u;
    while (completed_prefill_count < request_count)
    {
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
            dispatch.kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
        completed_prefill_count += dispatch.request_count;
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
    }
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == request_count);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u);
    assert(SparkGlm52RequestApiCancelDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiSubmitsCTextPromptToPrefillSchedule(void)
{
    SparkTestRequestApiFixture fixture;
    SparkTokenizer tokenizer;
    SparkGlm52TextPromptSubmitRequest request;
    SparkGlm52TextPromptSubmitResult result;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiPrefillDispatchView prefill_view;
    uint32_t prompt_token_storage[8u];

    SparkTestRequestApiWriteTokenizerJson();
    SparkTestRequestApiLoadTokenizer(&tokenizer);
    SparkTestInitializeFixture(&fixture);
    SparkGlm52TextPromptGetDefaultSubmitRequest(&request);
    request.priority = SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY;
    request.output_token_budget = 2u;
    request.request_id = 1540u;
    request.sequence_id = 11540u;
    request.prompt_text = "abc";
    request.prompt_text_bytes = 3u;
    request.prompt_token_storage = prompt_token_storage;
    request.prompt_token_storage_capacity = 8u;
    memset(prompt_token_storage, 0, sizeof(prompt_token_storage));
    assert(SparkGlm52RequestApiSubmitTextPrompt(
        &fixture.api,
        &tokenizer,
        &request,
        &result) == SPARK_STATUS_OK);
    assert(result.prompt_token_count == 1u);
    assert(result.required_prompt_token_count == 1u);
    assert(result.request_handle != SPARK_GLM52_REQUEST_API_INVALID_HANDLE);
    assert(prompt_token_storage[0u] == SPARK_TEST_TEXT_TOKEN_ABC);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
    assert(dispatch.request_count == 1u);
    assert(dispatch.request_handles[0u] == result.request_handle);
    assert(SparkGlm52RequestApiDescribePrefillDispatch(
        &dispatch,
        &prefill_view) == SPARK_STATUS_OK);
    assert(prefill_view.prompt_token_count == 1u);
    assert(prefill_view.lanes[0u].prompt_token_ids == prompt_token_storage);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    SparkTokenizerDestroy(&tokenizer);
}

static void SparkTestRequestApiAdmitsThirteenThousandWithFreeList(void)
{
    enum { request_capacity = 13000u };
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiConfiguration configuration;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiSlot *slots;
    SparkGlm52RequestApiHandle first_handle;
    SparkGlm52RequestApiHandle handle;
    uint32_t prompt[1u] = {17u};
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    slots = (SparkGlm52RequestApiSlot *)calloc(
        request_capacity,
        sizeof(slots[0u]));
    assert(slots != 0);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.request_capacity = request_capacity;
    configuration.prefetch_lane_count =
        SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    configuration.decode_batch_target =
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    configuration.scheduler = &fixture.scheduler;
    configuration.request_slots = slots;
    configuration.kv_prefetch_function = SparkTestCaptureKvPrefetch;
    configuration.kv_prefetch_context = &fixture.prefetch_capture;
    assert(SparkGlm52RequestApiInitialize(
        &fixture.api,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializeSubmitRequest(
        &request,
        1u,
        1u,
        SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY,
        prompt,
        1u,
        1u);
    first_handle = SPARK_GLM52_REQUEST_API_INVALID_HANDLE;
    for (request_index = 0u; request_index < request_capacity; ++request_index)
    {
        request.request_id = 500000u + request_index;
        request.sequence_id = 600000u + request_index;
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handle) == SPARK_STATUS_OK);
        if (request_index == 0u)
        {
            first_handle = handle;
        }
    }
    assert(fixture.api.queued_request_count == request_capacity);
    assert(fixture.api.free_slot_head == SPARK_GLM52_REQUEST_API_NO_SLOT);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_CAPACITY_EXCEEDED);

    assert(SparkGlm52RequestApiCancelRequest(
        &fixture.api,
        first_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiReleaseCompletedRequest(
        &fixture.api,
        first_handle) == SPARK_STATUS_OK);
    request.request_id += 1u;
    request.sequence_id += 1u;
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,
        &request,
        &handle) == SPARK_STATUS_OK);
    assert(fixture.api.free_slot_head == SPARK_GLM52_REQUEST_API_NO_SLOT);
    free(slots);
}

static void SparkTestRequestApiDecodeBatchBackfillsByDescendingPriority(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handles[6u];
    uint32_t prompts[6u][16u];
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    for (request_index = 0u; request_index < 6u; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],
            16u,
            60000u + (request_index * 1000u));
        SparkTestInitializeSubmitRequest(
            &request,
            700u + request_index,
            8700u + request_index,
            10u + (request_index * 10u),
            prompts[request_index],
            16u,
            2u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,
            &request,
            &handles[request_index]) == SPARK_STATUS_OK);
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
        assert(dispatch.request_handles[0] == handles[request_index]);
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,
            &dispatch) == SPARK_STATUS_OK);
    }

    fixture.api.decode_batch_target = 4u;
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
    assert(dispatch.accepted == 1u);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 4u);
    assert(dispatch.request_handles[0] == handles[5]);
    assert(dispatch.request_handles[1] == handles[4]);
    assert(dispatch.request_handles[2] == handles[3]);
    assert(dispatch.request_handles[3] == handles[2]);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,
        &dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiAdaptiveWidthBackfillsWithoutGrowingPriorityClass(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle low_handle;
    SparkGlm52RequestApiHandle high_handle;
    uint32_t prompts[2u][16u];
    uint32_t slot_index;

    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
    SparkTestFillTokenIds(prompts[0u],16u,66000u);
    SparkTestFillTokenIds(prompts[1u],16u,67000u);
    SparkTestInitializeSubmitRequest(
        &request,720u,8720u,10u,prompts[0u],16u,2u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&low_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    SparkTestInitializeSubmitRequest(
        &request,721u,8721u,100u,prompts[1u],16u,2u);
    assert(SparkGlm52RequestApiSubmit(
        &fixture.api,&request,&high_handle) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    for (slot_index = 2u; slot_index < 15u; ++slot_index)
    {
        fixture.api.request_slots[slot_index].state =
            SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE;
        fixture.api.request_slots[slot_index].priority = 100u;
    }
    fixture.api.running_request_count = 13u;
    assert(SparkGlm52RequestApiCurrentPipelineBatchWidth(&fixture.api) == 2u);
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.request_count == 2u);
    assert(dispatch.request_handles[0u] == high_handle);
    assert(dispatch.request_handles[1u] == low_handle);
    assert(dispatch.highest_priority == 100u);
}

static void SparkTestRequestApiCapsDecodeBatchByActiveKvBlocks(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52RequestApiHandle handles[4u];
    uint32_t prompts[4u][16u];
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    for (request_index = 0u; request_index < 4u; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],16u,70000u + (request_index * 1000u));
        SparkTestInitializeSubmitRequest(
            &request,
            800u + request_index,
            8800u + request_index,
            10u,
            prompts[request_index],
            16u,
            2u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,&request,&handles[request_index]) == SPARK_STATUS_OK);
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
    }

    fixture.api.decode_batch_target = 4u;
    fixture.api.max_resident_kv_block_count = 4u;
    fixture.api.configuration_flags &=
        ~SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH;
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert(dispatch.request_count == 2u);
    assert(dispatch.request_handles[0u] == handles[0u]);
    assert(dispatch.request_handles[1u] == handles[1u]);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiHoldsPrefillAtGlobalResidentKvLimit(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch prefill_dispatch;
    SparkGlm52RequestApiDispatch decode_dispatches[2u];
    SparkGlm52RequestApiHandle handles[6u];
    uint32_t prompts[6u][4u];
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    fixture.scheduler.queue_depth_per_spark = 8u;
    fixture.api.configuration_flags &=
        ~(SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH |
          SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING);
    fixture.api.decode_batch_target = 2u;
    fixture.api.max_resident_kv_block_count = 4u;
    for (request_index = 0u; request_index < 6u; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],4u,90000u + (request_index * 1000u));
        SparkTestInitializeSubmitRequest(
            &request,
            1000u + request_index,
            9000u + request_index,
            10u,
            prompts[request_index],
            4u,
            1u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,&request,&handles[request_index]) == SPARK_STATUS_OK);
    }
    for (request_index = 0u; request_index < 2u; ++request_index)
    {
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&prefill_dispatch) == SPARK_STATUS_OK);
        assert(prefill_dispatch.kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
        assert(prefill_dispatch.request_count == 2u);
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,&prefill_dispatch) == SPARK_STATUS_OK);
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&decode_dispatches[request_index]) == SPARK_STATUS_OK);
        assert(decode_dispatches[request_index].kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
        assert(decode_dispatches[request_index].request_count == 2u);
    }
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&prefill_dispatch) == SPARK_STATUS_BUSY);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&decode_dispatches[0u]) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 2u; ++request_index)
    {
        assert(SparkGlm52RequestApiReleaseCompletedRequest(
            &fixture.api,
            decode_dispatches[0u].request_handles[request_index]) ==
            SPARK_STATUS_OK);
    }
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&prefill_dispatch) == SPARK_STATUS_OK);
    assert(prefill_dispatch.kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH);
    assert(prefill_dispatch.request_count == 2u);
}

static void SparkTestRequestApiReservesMtpDraftKvBlocks(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiSubmitRequest request;
    SparkGlm52RequestApiDispatch dispatch;
    SparkGlm52KvBlockTableView block_table_view;
    SparkGlm52RequestApiHandle handles[4u];
    uint32_t physical_block_indices[8u];
    uint32_t lane_block_counts[3u];
    uint32_t prompts[4u][26u];
    uint32_t request_index;

    SparkTestInitializeFixture(&fixture);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    for (request_index = 0u; request_index < 4u; ++request_index)
    {
        SparkTestFillTokenIds(
            prompts[request_index],26u,80000u + (request_index * 1000u));
        SparkTestInitializeSubmitRequest(
            &request,
            900u + request_index,
            8900u + request_index,
            10u,
            prompts[request_index],
            26u,
            8u);
        assert(SparkGlm52RequestApiSubmit(
            &fixture.api,&request,&handles[request_index]) == SPARK_STATUS_OK);
        assert(SparkGlm52RequestApiScheduleNext(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
        assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL);
        assert(SparkGlm52RequestApiCompleteDispatch(
            &fixture.api,&dispatch) == SPARK_STATUS_OK);
    }

    fixture.api.decode_batch_target = 4u;
    fixture.api.max_resident_kv_block_count = 6u;
    fixture.api.configuration_flags &=
        ~SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH;
    assert(SparkGlm52RequestApiScheduleNext(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
    assert(dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH);
    assert((dispatch.flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u);
    assert(dispatch.mtp_draft_token_budget ==
        SPARK_GLM52_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT);
    assert(dispatch.request_count == 3u);
    assert(SparkGlm52RequestApiBuildDispatchKvBlockTableView(
        &fixture.api,
        &dispatch,
        physical_block_indices,
        0,
        4u,
        4u,
        lane_block_counts,
        3u,
        &block_table_view) == SPARK_STATUS_OK);
    assert(lane_block_counts[0u] == 2u);
    assert(lane_block_counts[1u] == 2u);
    assert(lane_block_counts[2u] == 2u);
    assert(SparkGlm52RequestApiCompleteDispatch(
        &fixture.api,&dispatch) == SPARK_STATUS_OK);
}

static void SparkTestRequestApiUsesAdaptivePipelineBatchWidth(void)
{
    SparkTestRequestApiFixture fixture;
    uint32_t slot_index;

    SparkTestInitializeFixture(&fixture);
    assert(SparkGlm52RequestApiCurrentPipelineBatchWidth(&fixture.api) ==
        fixture.api.decode_batch_target);
    fixture.api.configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
    for (slot_index = 0u; slot_index < 13u; ++slot_index)
    {
        fixture.api.request_slots[slot_index].state =
            SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
        fixture.api.request_slots[slot_index].priority = 10u;
    }
    assert(SparkGlm52RequestApiCurrentPipelineBatchWidth(&fixture.api) == 1u);
    for (slot_index = 0u; slot_index < 13u; ++slot_index)
    {
        fixture.api.request_slots[slot_index].state =
            SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE;
    }
    fixture.api.request_slots[13u].state =
        SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
    fixture.api.request_slots[13u].priority = 10u;
    assert(SparkGlm52RequestApiCurrentPipelineBatchWidth(&fixture.api) == 2u);
    fixture.api.request_slots[14u].state =
        SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
    fixture.api.request_slots[14u].priority = 100u;
    assert(SparkGlm52RequestApiCurrentPipelineBatchWidth(&fixture.api) == 1u);
}

static void SparkTestRequestApiAcceptsMtpForceEnableConfiguration(void)
{
    SparkTestRequestApiFixture fixture;
    SparkGlm52RequestApiConfiguration configuration;
    SparkGlm52RequestApi api;

    SparkTestInitializeFixture(&fixture);
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.configuration_flags =
        SPARK_GLM52_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    configuration.request_capacity = SPARK_TEST_REQUEST_SLOT_COUNT;
    configuration.prefetch_lane_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    configuration.decode_batch_target =
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    configuration.scheduler = &fixture.scheduler;
    configuration.request_slots = fixture.request_slots;
    configuration.kv_prefetch_function = SparkTestCaptureKvPrefetch;
    configuration.kv_prefetch_context = &fixture.prefetch_capture;
    assert(SparkGlm52RequestApiInitialize(&api,&configuration) == SPARK_STATUS_OK);
    assert((api.configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE) != 0u);
}

int main(void)
{
    SparkTestRequestApiJitPrefetchesCachedPrefixForPriorityRequest();
    SparkTestMtpTreeResolvesEveryPath();
    SparkTestMtpTreeUsesCompactAlternateStorage();
    SparkTestRequestApiMtpDraftRequiresSpeculativeVerify();
    SparkTestRequestApiMtpVerifyCapturesDsparkBatchTap();
    SparkTestRequestApiUsesHigherYieldMtpBeforeEqualPriorityDecode();
    SparkTestRequestApiMtpDraftBudgetRemainsTransactional();
    SparkTestRequestApiPrefillWaveSpansMultipleKvBlocks();
    SparkTestRequestApiMtpAdaptiveFloorSuppressesAndRecovers();
    SparkTestRequestApiMtpRejectedDraftStaysOutsideNextContext();
    SparkTestRequestApiMtpBudgetLeavesVerifierFallbackHeadroom();
    SparkTestRequestApiMtpVerifyCapsPackedExecutionRows();
    SparkTestRequestApiSkipsUnprofitableWideMtp();
    SparkTestRequestApiBatchesReadyDecodeRequestsAndConsumesBudgets();
	SparkTestRequestApiFillsDecodeBatchBeforeEqualPriorityDecode();
    SparkTestRequestApiHoldsPrefillAtGlobalResidentKvLimit();
    SparkTestRequestApiCohortsSamePromptRequestsAndSharesBlocks();
    SparkTestRequestApiWidePrefixFamilyCannotBeatHigherPriority();
    SparkTestRequestApiLateHighPriorityUsesNextAvailablePipelineSlot();
    SparkTestRequestApiCohortsArbitrarySharedPrefixWithSuffixes();
    SparkTestRequestApiChoosesWiderSharedPrefixFamily();
    SparkTestRequestApiBatchesVariableOneBlockSuffixes();
    SparkTestRequestApiAdaptivePrefillChoosesFullResidentBucketOverOlderSingleton();
    SparkTestRequestApiRealtimePrefillBypassesFullBulkBatch();
    SparkTestRequestApiDecodeBatchUsesMeasuredB64ForSeventeenReadyRequests();
    SparkTestRequestApiDecodeBatchBackfillsByDescendingPriority();
    SparkTestRequestApiAdaptiveWidthBackfillsWithoutGrowingPriorityClass();
    SparkTestRequestApiCapsDecodeBatchByActiveKvBlocks();
    SparkTestRequestApiReservesMtpDraftKvBlocks();
    SparkTestRequestApiOpportunisticLookaheadDoesNotBlockReadyPriorityPrefill();
    SparkTestRequestApiPrefetchesLiveNonresidentDecodeBlocks();
    SparkTestRequestApiBatchesDecodeAfterBatchCriticalPrefetch();
    SparkTestRequestApiBatchesPrefillAfterBatchCriticalPrefetch();
    SparkTestRequestApiAsyncJitPrefetchOverlapsResidentWork();
    SparkTestRequestApiDoesNotGreenlightMissingJitKv();
    SparkTestRequestApiTrimsResidentKvWithoutEvictingNearFuturePrefix();
    SparkTestRequestApiEvictsColdResidentBlocksButKeepsLookaheadHotset();
    SparkTestRequestApiReuseScoredEvictionKeepsSharedPrefixFamily();
    SparkTestRequestApiUsesBuiltInAsyncMemoryPrefetchBackend();
    SparkTestRequestApiRefreshesQueueAwarePrefixProtection();
    SparkTestRequestApiDsparkCapturesTapsAndRunsSpeculativeVerify();
    SparkTestRequestApiDsparkBatchesEqualLengthDrafts();
    SparkTestRequestApiDsparkDisabledPerRequestFallsBackToDecode();
    SparkTestRequestApiDescribesAndCopiesFullPrefillTokenWindows();
    SparkTestRequestApiSubmitsCTextPromptToPrefillSchedule();
    SparkTestRequestApiAdmitsThirteenThousandWithFreeList();
    SparkTestRequestApiUsesAdaptivePipelineBatchWidth();
    SparkTestRequestApiAcceptsMtpForceEnableConfiguration();
    return 0;
}
