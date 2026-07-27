#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_glm52_scheduler.h"

static void SparkTestFillTokenIds(
    uint32_t *token_ids,
    uint32_t token_count,
    uint32_t base_token_id)
{
    uint32_t token_index;

    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        token_ids[token_index] = base_token_id + token_index;
    }
}

static void SparkTestBuildSharedPrefixPrompt(
    uint32_t *prompt_token_ids,
    const uint32_t *shared_prefix_token_ids,
    uint32_t shared_prefix_token_count,
    uint32_t suffix_base_token_id,
    uint32_t suffix_token_count)
{
    uint32_t token_index;

    for (token_index = 0u; token_index < shared_prefix_token_count; ++token_index)
    {
        prompt_token_ids[token_index] = shared_prefix_token_ids[token_index];
    }
    for (token_index = 0u; token_index < suffix_token_count; ++token_index)
    {
        prompt_token_ids[shared_prefix_token_count + token_index] =
            suffix_base_token_id + token_index;
    }
}

static void SparkTestInitializePrefixCache(
    SparkGlm52PrefixCache *cache,
    SparkGlm52PrefixCacheEntry *entries,
    SparkGlm52PrefixCacheSequenceBinding *bindings,
    uint32_t entry_count,
    uint32_t binding_count)
{
    static SparkGlm52KvCacheArena arena;
    static SparkGlm52KvCacheBlock blocks[512u];
    SparkGlm52KvCacheConfiguration kv_configuration;
    SparkGlm52PrefixCacheConfiguration configuration;

    assert(entry_count <= 512u);
    memset(&kv_configuration, 0, sizeof(kv_configuration));
    kv_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    kv_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    kv_configuration.physical_block_count = entry_count;
    kv_configuration.block_token_count =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    kv_configuration.layer_count = 78u;
    kv_configuration.kv_head_count = 8u;
    kv_configuration.head_dim = 128u;
    kv_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
    kv_configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    kv_configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    kv_configuration.blocks = blocks;
    assert(SparkGlm52KvCacheArenaInitialize(&arena, &kv_configuration) ==
        SPARK_STATUS_OK);

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_PREFIX_CACHE_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.block_token_count = SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    configuration.entry_count = entry_count;
    configuration.physical_block_count = entry_count;
    configuration.sequence_binding_count = binding_count;
    configuration.entries = entries;
    configuration.sequence_bindings = bindings;
    configuration.kv_cache_arena = &arena;
    assert(SparkGlm52PrefixCacheInitialize(cache, &configuration) ==
        SPARK_STATUS_OK);
}

static void SparkTestInitializeSchedulerConfiguration(
    SparkGlm52SchedulerConfiguration *configuration,
    uint32_t quantization_mode,
    SparkGlm52PrefixCache *prefix_cache)
{
    memset(configuration, 0, sizeof(*configuration));
    configuration->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    configuration->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration->spark_count = SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT;
    configuration->queue_depth_per_spark = 1u;
    configuration->measured_profile_id =
        SPARK_GLM52_STAGE_PLAN_MEASURED_PROFILE_20260701;
    configuration->quantization_mode = quantization_mode;
    configuration->configuration_flags =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS;
    configuration->prefix_cache_block_tokens =
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS;
    configuration->prefix_cache = prefix_cache;
}

static void SparkTestInitializeDecodeRequest(
    SparkGlm52SchedulerRequest *request,
    uint32_t active_sequence_count)
{
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    request->active_sequence_count = active_sequence_count;
    request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE;
}

static void SparkTestInitializePrefillRequest(
    SparkGlm52SchedulerRequest *request,
    uint32_t active_sequence_count,
    uint32_t prompt_token_count,
    uint64_t sequence_id,
    const uint32_t *prompt_token_ids)
{
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    request->active_sequence_count = active_sequence_count;
    request->prompt_token_count = prompt_token_count;
    request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL;
    request->sequence_id = sequence_id;
    request->prompt_token_ids = prompt_token_ids;
}

static void SparkTestGlm52SchedulerAdmitsCurrentSparkRingDecode(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    uint32_t stage_index;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    assert(scheduler.configuration_flags ==
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS);
    assert(scheduler.max_prefill_tokens_per_step ==
        SPARK_GLM52_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP);
    assert(scheduler.prefix_cache_block_tokens ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);

    SparkTestInitializeDecodeRequest(&request, 64u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(decision.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT);
    assert(decision.stage_count == SPARK_GLM52_SCHEDULER_MAX_SPARK_COUNT);
    assert(decision.estimated_critical_path_ns != 0u);
    assert(decision.decision_flags ==
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_STEP);
    assert(decision.total_scheduled_token_count == 64u);
    assert(decision.graph_sequence_padding_count == 0u);
    assert(scheduler.scheduled_decode_token_count == 64u);
    assert(decision.stage_plan.stages[0].first_layer_index == 0u);
    assert(decision.stage_plan.stages[0].layer_count == 6u);
    for (stage_index = 0u; stage_index < decision.stage_count; ++stage_index)
    {
        assert(decision.dispatch_stages[stage_index].spark_index == stage_index);
        assert(decision.dispatch_stages[stage_index].dispatch_flags ==
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE);
        assert(decision.dispatch_stages[stage_index].estimated_service_time_ns !=
            0u);
        assert(scheduler.spark_inflight_counts[stage_index] == 1u);
    }

    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);
    assert(decision.rejected_status == SPARK_STATUS_BUSY);
    assert(scheduler.rejected_count == 1u);

    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeDecodeRequest(&request, 64u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);

    SparkTestInitializeDecodeRequest(&request, 64u);
    configuration.queue_depth_per_spark = 2u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    for (stage_index = 0u; stage_index < decision.stage_count; ++stage_index)
    {
        assert(scheduler.spark_inflight_counts[stage_index] == 0u);
    }
}

static void SparkTestGlm52SchedulerSupportsFp8AndPrefill(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decode_decision;
    SparkGlm52SchedulerDecision prefill_decision;
    uint32_t tokens[32u];

    SparkTestFillTokenIds(tokens, 32u, 3000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &cache);
    configuration.queue_depth_per_spark = 2u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializeDecodeRequest(&request, 16u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decode_decision) == SPARK_STATUS_OK);
    assert(decode_decision.accepted == 1u);
    assert(decode_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(decode_decision.graph_sequence_padding_count == 0u);
    assert(decode_decision.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decode_decision) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(&request, 33u, 32u, 11u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &prefill_decision) == SPARK_STATUS_OK);
    assert(prefill_decision.accepted == 1u);
    assert(prefill_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(prefill_decision.graph_sequence_padding_count == 31u);
    assert((prefill_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING) != 0u);
    assert((prefill_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK) != 0u);
    assert((prefill_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u);
    assert(prefill_decision.scheduled_prompt_token_count == 32u);
    assert(prefill_decision.prefill_block_count == 2u);
    assert(prefill_decision.cache_commit_token_count_after_step == 32u);
    assert(prefill_decision.total_scheduled_token_count == 1056u);
    assert(prefill_decision.estimated_critical_path_ns >=
        decode_decision.estimated_critical_path_ns);

    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &prefill_decision) == SPARK_STATUS_OK);
    assert(cache.inserted_block_count == 2u);
    assert(SparkGlm52SchedulerReleaseSequence(&scheduler, 11u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerSupportsW8lut(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    SparkTestInitializeDecodeRequest(&request, 16u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerUsesVllmStyleChunkedPrefill(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    uint32_t tokens[1024u];
    uint64_t chunk_critical_path_ns;

    SparkTestFillTokenIds(tokens, 1024u, 4000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.max_prefill_tokens_per_step = 128u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(&request, 64u, 1024u, 21u, tokens);
    request.max_scheduled_prompt_token_count = 256u;
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.scheduled_prompt_token_offset == 0u);
    assert(decision.scheduled_prompt_token_count == 128u);
    assert(decision.remaining_prompt_token_count_after_step == 896u);
    assert(decision.prefill_block_count == 8u);
    assert(decision.total_scheduled_token_count == 8192u);
    assert((decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u);
    assert((decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK) == 0u);
    assert((decision.dispatch_stages[0].dispatch_flags &
        SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_CHUNK) != 0u);
    assert(scheduler.chunked_prefill_count == 1u);
    assert(scheduler.scheduled_prefill_token_count == 8192u);
    chunk_critical_path_ns = decision.estimated_critical_path_ns;

    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    assert(cache.inserted_block_count == 8u);
    SparkTestInitializePrefillRequest(&request, 64u, 1024u, 21u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.cached_prefix_token_count == 128u);
    assert(decision.scheduled_prompt_token_offset == 128u);
    assert(decision.scheduled_prompt_token_count == 128u);
    assert(decision.remaining_prompt_token_count_after_step == 768u);
    assert((decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK) != 0u);
    assert((decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK) == 0u);
    assert(decision.estimated_critical_path_ns == chunk_critical_path_ns);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerReleaseSequence(&scheduler, 21u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerUsesIntegratedPrefixCacheAdmission(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    uint32_t tokens[1024u];
    uint32_t short_tokens[100u];

    SparkTestFillTokenIds(tokens, 1024u, 5000u);
    SparkTestFillTokenIds(short_tokens, 100u, 7000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    assert(SparkGlm52PrefixCacheCommitPrompt(
        &cache,
        77u,
        tokens,
        768u,
        0) == SPARK_STATUS_OK);
    assert(SparkGlm52PrefixCacheCommitPrompt(
        &cache,
        78u,
        short_tokens,
        100u,
        0) == SPARK_STATUS_OK);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.max_prefill_tokens_per_step = 256u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(&request, 32u, 1024u, 88u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.cached_prefix_token_count == 768u);
    assert(decision.prefix_cache_block_count == 48u);
    assert(decision.scheduled_prompt_token_offset == 768u);
    assert(decision.scheduled_prompt_token_count == 256u);
    assert(decision.remaining_prompt_token_count_after_step == 0u);
    assert(decision.prefill_block_count == 16u);
    assert((decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED) != 0u);
    assert((decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK) != 0u);
    assert(scheduler.prefix_cache_hit_token_count == 24576u);
    assert(scheduler.scheduled_prefill_token_count == 8192u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerReleaseSequence(&scheduler, 88u) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(&request, 1u, 100u, 89u, short_tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.cached_prefix_token_count == 96u);
    assert(decision.prefix_cache_block_count == 6u);
    assert(decision.scheduled_prompt_token_offset == 96u);
    assert(decision.scheduled_prompt_token_count == 4u);
    assert(decision.remaining_prompt_token_count_after_step == 0u);
    assert(decision.prefill_block_count == 1u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerReleaseSequence(&scheduler, 89u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerDisablesCrossSequencePrefixReuse(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    uint32_t tokens[64u];
    uint32_t first_blocks[2u];

    SparkTestFillTokenIds(tokens,64u,7500u);
    SparkTestInitializePrefixCache(&cache,entries,bindings,128u,512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &cache);
    configuration.max_prefill_tokens_per_step = 32u;
    configuration.configuration_flags &=
        ~SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,&configuration) == SPARK_STATUS_OK);
    SparkTestInitializePrefillRequest(&request,1u,64u,601u,tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,&request,&decision) == SPARK_STATUS_OK);
    assert(decision.scheduled_prompt_token_offset == 0u);
    first_blocks[0u] = decision.kv_physical_block_indices[0u];
    first_blocks[1u] = decision.kv_physical_block_indices[1u];
    assert(SparkGlm52SchedulerComplete(
        &scheduler,&decision) == SPARK_STATUS_OK);
    SparkTestInitializePrefillRequest(&request,1u,64u,601u,tokens);
    request.computed_prompt_token_count = 32u;
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,&request,&decision) == SPARK_STATUS_OK);
    assert(decision.cached_prefix_token_count == 0u);
    assert(decision.scheduled_prompt_token_offset == 32u);
    assert(decision.kv_physical_block_indices[0u] == first_blocks[0u]);
    assert(decision.kv_physical_block_indices[1u] == first_blocks[1u]);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,&decision) == SPARK_STATUS_OK);
    SparkTestInitializePrefillRequest(&request,1u,64u,602u,tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,&request,&decision) == SPARK_STATUS_OK);
    assert(decision.cached_prefix_token_count == 0u);
    assert(decision.scheduled_prompt_token_offset == 0u);
    assert(decision.kv_physical_block_indices[0u] != first_blocks[0u]);
    assert(SparkGlm52SchedulerCancel(
        &scheduler,&decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerReleaseSequence(
        &scheduler,601u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerInterleavesPrefillAndDecode(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision prefill_decision;
    SparkGlm52SchedulerDecision rejected_prefill_decision;
    SparkGlm52SchedulerDecision decode_decision;
    uint32_t tokens[1024u];

    SparkTestFillTokenIds(tokens, 1024u, 9000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.queue_depth_per_spark = 2u;
    configuration.max_prefill_tokens_per_step = 256u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(&request, 64u, 1024u, 31u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &prefill_decision) == SPARK_STATUS_OK);
    assert(prefill_decision.accepted == 1u);
    assert((prefill_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT) != 0u);
    assert(scheduler.interleaved_prefill_admission_count == 1u);

    SparkTestInitializePrefillRequest(&request, 64u, 1024u, 32u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &rejected_prefill_decision) == SPARK_STATUS_OK);
    assert(rejected_prefill_decision.accepted == 0u);
    assert(rejected_prefill_decision.rejected_status == SPARK_STATUS_BUSY);

    SparkTestInitializeDecodeRequest(&request, 64u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decode_decision) == SPARK_STATUS_OK);
    assert(decode_decision.accepted == 1u);
    assert((decode_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL) != 0u);
    assert((decode_decision.dispatch_stages[0].dispatch_flags &
        SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL) != 0u);
    assert(scheduler.decode_bypass_admission_count == 1u);

    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decode_decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &prefill_decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerReleaseSequence(&scheduler, 31u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerFillsCurrentSparkPipeline(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    SparkGlm52SchedulerDecision prefill_decision;
    uint32_t tokens[16u];
    uint32_t cohort_index;

    SparkTestFillTokenIds(tokens, 16u, 9500u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &cache);
    configuration.queue_depth_per_spark =
        SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT + 1u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    for (cohort_index = 0u;
         cohort_index + 1u < SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT;
         ++cohort_index)
    {
        SparkTestInitializeDecodeRequest(&request, 64u);
        assert(SparkGlm52SchedulerAdmit(
            &scheduler,
            &request,
            &decision) == SPARK_STATUS_OK);
        assert(decision.accepted == 1u);
    }

    SparkTestInitializePrefillRequest(&request, 1u, 16u, 51u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &prefill_decision) == SPARK_STATUS_OK);
    assert(prefill_decision.accepted == 1u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &prefill_decision) == SPARK_STATUS_OK);

    SparkTestInitializeDecodeRequest(&request, 64u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(scheduler.spark_inflight_counts[0] ==
        SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT);

    SparkTestInitializePrefillRequest(&request, 1u, 16u, 52u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);
    assert(decision.rejected_status == SPARK_STATUS_BUSY);
    assert(SparkGlm52SchedulerReleaseSequence(&scheduler, 51u) == SPARK_STATUS_OK);
}


static void SparkTestGlm52SchedulerPacksDecodeRequestsIntoSingleGraphDecision(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest requests[8u];
    SparkGlm52SchedulerBatchRequest batch_request;
    SparkGlm52SchedulerBatchDecision batch_decision;
    uint32_t request_index;
    uint32_t stage_index;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.queue_depth_per_spark = 1u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    for (request_index = 0u; request_index < 8u; ++request_index)
    {
        SparkTestInitializeDecodeRequest(&requests[request_index], 1u);
    }
    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = 8u;
    batch_request.requests = requests;

    assert(SparkGlm52SchedulerAdmitDecodeBatch(
        &scheduler,
        &batch_request,
        &batch_decision) == SPARK_STATUS_OK);
    assert(batch_decision.accepted == 1u);
    assert(batch_decision.source_request_count == 8u);
    assert(batch_decision.packed_request_count == 8u);
    assert(batch_decision.active_sequence_count == 8u);
    assert(batch_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(batch_decision.graph_sequence_capacity ==
        SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
    assert(batch_decision.graph_sequence_padding_count == 8u);
    assert(batch_decision.total_scheduled_token_count == 8u);
    assert((batch_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) != 0u);
    assert((batch_decision.stage_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_DECODE_STEP) != 0u);
    assert((batch_decision.stage_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK) != 0u);
    assert(scheduler.admitted_count == 1u);
    assert(scheduler.scheduled_decode_token_count == 8u);
    assert(scheduler.adaptive_decode_pack_admission_count == 1u);
    assert(scheduler.adaptive_decode_pack_request_count == 8u);
    assert(scheduler.adaptive_decode_pack_padding_token_count == 8u);

    for (request_index = 0u; request_index < 8u; ++request_index)
    {
        assert(batch_decision.packed_requests[request_index].request_index ==
            request_index);
        assert(batch_decision.packed_requests[request_index].active_sequence_offset ==
            request_index);
        assert(batch_decision.packed_requests[request_index].active_sequence_count ==
            1u);
        assert(batch_decision.packed_requests[request_index].scheduled_token_count ==
            1u);
    }
    for (stage_index = 0u;
         stage_index < batch_decision.stage_decision.stage_count;
         ++stage_index)
    {
        assert((batch_decision.stage_decision.dispatch_stages[stage_index].dispatch_flags &
            SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK) != 0u);
        assert(scheduler.spark_inflight_counts[stage_index] == 1u);
    }

    assert(SparkGlm52SchedulerCompleteDecodeBatch(
        &scheduler,
        &batch_decision) == SPARK_STATUS_OK);
    for (stage_index = 0u;
         stage_index < batch_decision.stage_decision.stage_count;
         ++stage_index)
    {
        assert(scheduler.spark_inflight_counts[stage_index] == 0u);
    }
    assert(scheduler.completed_count == 1u);
}

static void SparkTestGlm52SchedulerDecodeBatchFillsMaxBucketFromOversubscribedQueue(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest requests[1040u];
    SparkGlm52SchedulerBatchRequest batch_request;
    SparkGlm52SchedulerBatchDecision batch_decision;
    uint32_t request_index;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    for (request_index = 0u; request_index < 1040u; ++request_index)
    {
        SparkTestInitializeDecodeRequest(&requests[request_index], 1u);
    }
    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = 1040u;
    batch_request.requests = requests;

    assert(SparkGlm52SchedulerAdmitDecodeBatch(
        &scheduler,
        &batch_request,
        &batch_decision) == SPARK_STATUS_OK);
    assert(batch_decision.accepted == 1u);
    assert(batch_decision.source_request_count == 1040u);
    assert(batch_decision.packed_request_count ==
        SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET);
    assert(batch_decision.active_sequence_count ==
        SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET);
    assert(batch_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET);
    assert(batch_decision.graph_sequence_padding_count == 0u);
    assert(batch_decision.packed_requests[
        SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET - 1u].request_index ==
            SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET - 1u);
    assert(batch_decision.packed_requests[
        SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET - 1u].active_sequence_offset ==
            SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET - 1u);
    assert(batch_decision.stage_decision.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    assert(SparkGlm52SchedulerCompleteDecodeBatch(
        &scheduler,
        &batch_decision) == SPARK_STATUS_OK);
}


static void SparkTestGlm52SchedulerUsesMeasuredDecodeBucketForMidSizedBatch(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest measured_requests[17u];
    SparkGlm52SchedulerRequest legacy_requests[17u];
    SparkGlm52SchedulerBatchRequest batch_request;
    SparkGlm52SchedulerBatchDecision measured_decision;
    SparkGlm52SchedulerBatchDecision legacy_decision;
    uint32_t request_index;
    uint64_t measured_critical_path_ns;

    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.queue_depth_per_spark = 2u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    for (request_index = 0u; request_index < 17u; ++request_index)
    {
        SparkTestInitializeDecodeRequest(&measured_requests[request_index], 1u);
    }
    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = 17u;
    batch_request.requests = measured_requests;

    assert(SparkGlm52SchedulerAdmitDecodeBatch(
        &scheduler,
        &batch_request,
        &measured_decision) == SPARK_STATUS_OK);
    assert(measured_decision.accepted == 1u);
    assert(measured_decision.active_sequence_count == 17u);
    assert(measured_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(measured_decision.graph_sequence_capacity ==
        SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
    assert(measured_decision.graph_sequence_padding_count == 47u);
    assert((measured_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u);
    assert((measured_decision.stage_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) != 0u);
    assert((measured_decision.stage_decision.dispatch_stages[0u].dispatch_flags &
        SPARK_GLM52_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET) != 0u);
    assert(scheduler.measured_decode_bucket_selection_count == 1u);
    assert(scheduler.measured_decode_bucket_padding_token_count == 47u);
    measured_critical_path_ns = measured_decision.estimated_critical_path_ns;
    assert(SparkGlm52SchedulerCompleteDecodeBatch(
        &scheduler,
        &measured_decision) == SPARK_STATUS_OK);

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.queue_depth_per_spark = 2u;
    configuration.configuration_flags =
        SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS &
        ~SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 17u; ++request_index)
    {
        SparkTestInitializeDecodeRequest(&legacy_requests[request_index], 1u);
    }
    batch_request.requests = legacy_requests;
    assert(SparkGlm52SchedulerAdmitDecodeBatch(
        &scheduler,
        &batch_request,
        &legacy_decision) == SPARK_STATUS_OK);
    assert(legacy_decision.accepted == 1u);
    assert(legacy_decision.active_sequence_count == 17u);
    assert(legacy_decision.batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B32);
    assert(legacy_decision.graph_sequence_padding_count == 15u);
    assert((legacy_decision.decision_flags &
        SPARK_GLM52_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET) == 0u);
    assert(measured_critical_path_ns < legacy_decision.estimated_critical_path_ns);
    assert(SparkGlm52SchedulerCompleteDecodeBatch(
        &scheduler,
        &legacy_decision) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerRejectsPrefillInDecodeBatch(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest requests[2u];
    SparkGlm52SchedulerBatchRequest batch_request;
    SparkGlm52SchedulerBatchDecision batch_decision;
    uint32_t tokens[16u];

    SparkTestFillTokenIds(tokens, 16u, 12000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializeDecodeRequest(&requests[0u], 1u);
    SparkTestInitializePrefillRequest(&requests[1u], 1u, 16u, 99u, tokens);
    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = 2u;
    batch_request.requests = requests;

    assert(SparkGlm52SchedulerAdmitDecodeBatch(
        &scheduler,
        &batch_request,
        &batch_decision) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(batch_decision.accepted == 0u);
    assert(batch_decision.rejected_status == SPARK_STATUS_INVALID_ARGUMENT);
    assert(scheduler.admitted_count == 0u);
}


static void SparkTestGlm52SchedulerExposesKvBlockTableAndCancelsReservation(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    SparkGlm52PrefixCacheLookup lookup;
    uint32_t tokens[48u];
    uint32_t physical_blocks[8u];
    uint32_t physical_block_count;

    SparkTestFillTokenIds(tokens, 48u, 13000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.max_prefill_tokens_per_step = 48u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(&request, 16u, 48u, 501u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(decision.kv_block_token_count ==
        SPARK_GLM52_SCHEDULER_PREFILL_BLOCK_TOKENS);
    assert(decision.kv_physical_block_count == 3u);
    assert(decision.kv_pending_physical_block_count == 3u);
    assert(decision.kv_cached_physical_block_count == 0u);
    assert(decision.prefix_cache_reservation_epoch != 0u);

    assert(SparkGlm52SchedulerBuildKvBlockTable(
        &scheduler,
        &decision,
        physical_blocks,
        8u,
        &physical_block_count) == SPARK_STATUS_OK);
    assert(physical_block_count == 3u);
    assert(physical_blocks[0u] == decision.kv_physical_block_indices[0u]);
    assert(physical_blocks[1u] == decision.kv_physical_block_indices[1u]);
    assert(physical_blocks[2u] == decision.kv_physical_block_indices[2u]);

    assert(SparkGlm52PrefixCacheProbePrompt(
        &cache,
        502u,
        tokens,
        48u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 0u);

    assert(SparkGlm52SchedulerCancel(&scheduler, &decision) == SPARK_STATUS_OK);
    assert(scheduler.kv_block_cancel_count == 1u);
    assert(SparkGlm52PrefixCacheProbePrompt(
        &cache,
        503u,
        tokens,
        48u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 0u);

    SparkTestInitializePrefillRequest(&request, 16u, 48u, 504u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &decision) == SPARK_STATUS_OK);
    assert(SparkGlm52PrefixCacheProbePrompt(
        &cache,
        505u,
        tokens,
        48u,
        &lookup) == SPARK_STATUS_OK);
    assert(lookup.matched_token_count == 32u);
    assert(SparkGlm52SchedulerReleaseSequence(
        &scheduler,
        504u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerBuildsBatchedPrefillKvTables(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest warm_request;
    SparkGlm52SchedulerRequest batch_requests[3u];
    SparkGlm52SchedulerDecision warm_decision;
    SparkGlm52SchedulerPrefillBatchRequest batch_request;
    SparkGlm52SchedulerPrefillBatchDecision batch_decision;
    uint32_t shared_prefix[32u];
    uint32_t prompts[3u][48u];
    uint32_t shared_physical_blocks[4u];
    uint32_t batch_physical_blocks[3u][4u];
    uint32_t batch_physical_block_counts[3u];
    uint32_t shared_physical_block_count;
    uint32_t request_index;

    SparkTestFillTokenIds(shared_prefix, 32u, 15000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 128u, 512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &cache);
    configuration.max_prefill_tokens_per_step = 48u;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);

    SparkTestInitializePrefillRequest(
        &warm_request,
        1u,
        32u,
        601u,
        shared_prefix);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &warm_request,
        &warm_decision) == SPARK_STATUS_OK);
    assert(warm_decision.accepted == 1u);
    assert(SparkGlm52SchedulerComplete(
        &scheduler,
        &warm_decision) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerBuildKvBlockTable(
        &scheduler,
        &warm_decision,
        shared_physical_blocks,
        4u,
        &shared_physical_block_count) == SPARK_STATUS_OK);
    assert(shared_physical_block_count == 2u);

    for (request_index = 0u; request_index < 3u; ++request_index)
    {
        SparkTestBuildSharedPrefixPrompt(
            prompts[request_index],
            shared_prefix,
            32u,
            16000u + request_index * 100u,
            16u);
        SparkTestInitializePrefillRequest(
            &batch_requests[request_index],
            1u,
            48u,
            701u + request_index,
            prompts[request_index]);
    }

    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = 3u;
    batch_request.requests = batch_requests;
    assert(SparkGlm52SchedulerAdmitPrefillBatch(
        &scheduler,
        &batch_request,
        &batch_decision) == SPARK_STATUS_OK);
    assert(batch_decision.accepted == 1u);
    assert(batch_decision.active_sequence_count == 3u);
    assert(batch_decision.maximum_scheduled_prompt_token_count == 16u);
    assert(batch_decision.total_scheduled_token_count == 3u * 16u);
    for (request_index = 0u; request_index < 3u; ++request_index)
    {
        assert(batch_decision.lanes[request_index].cached_prefix_token_count == 32u);
        assert(batch_decision.lanes[request_index].scheduled_prompt_token_count == 16u);
        assert(batch_decision.lanes[request_index].kv_block_table_token_count == 48u);
    }

    memset(batch_physical_blocks, 0, sizeof(batch_physical_blocks));
    memset(batch_physical_block_counts, 0, sizeof(batch_physical_block_counts));
    assert(SparkGlm52SchedulerBuildPrefillBatchKvBlockTables(
        &scheduler,
        &batch_decision,
        &batch_physical_blocks[0u][0u],
        4u,
        4u,
        batch_physical_block_counts,
        3u) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 3u; ++request_index)
    {
        assert(batch_physical_block_counts[request_index] == 3u);
        assert(batch_physical_blocks[request_index][0u] == shared_physical_blocks[0u]);
        assert(batch_physical_blocks[request_index][1u] == shared_physical_blocks[1u]);
        assert(batch_physical_blocks[request_index][2u] != shared_physical_blocks[0u]);
        assert(batch_physical_blocks[request_index][2u] != shared_physical_blocks[1u]);
    }
    assert(SparkGlm52SchedulerCompletePrefillBatch(
        &scheduler,
        &batch_decision) == SPARK_STATUS_OK);
    for (request_index = 0u; request_index < 3u; ++request_index)
    {
        assert(SparkGlm52SchedulerReleaseSequence(
            &scheduler,
            701u + request_index) == SPARK_STATUS_OK);
    }
    assert(SparkGlm52SchedulerReleaseSequence(
        &scheduler,
        601u) == SPARK_STATUS_OK);
}

static void SparkTestGlm52SchedulerRejectsInvalidInputs(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[32u];
    SparkGlm52PrefixCacheSequenceBinding bindings[64u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    SparkGlm52SchedulerRequest request;
    SparkGlm52SchedulerDecision decision;
    uint32_t tokens[16u];

    SparkTestFillTokenIds(tokens, 16u, 11000u);
    SparkTestInitializePrefixCache(&cache, entries, bindings, 32u, 64u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        99u,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO,
        &cache);
    configuration.configuration_flags = 0xffffffffu;
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO,
        0);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_AUTO,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,
        &configuration) == SPARK_STATUS_OK);
    assert(scheduler.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT);

    SparkTestInitializeDecodeRequest(
        &request,
        SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET + 1u);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 0u);
    assert(decision.rejected_status == SPARK_STATUS_CAPACITY_EXCEEDED);

    SparkTestInitializePrefillRequest(&request, 1u, 16u, 41u, tokens);
    request.computed_prompt_token_count = 16u;
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeDecodeRequest(&request, 1u);
    request.cached_prefix_token_count = 16u;
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializePrefillRequest(&request, 1u, 16u, 42u, tokens);
    request.cached_prefix_token_count = 16u;
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializePrefillRequest(&request, 1u, 16u, 0u, tokens);
    assert(SparkGlm52SchedulerAdmit(
        &scheduler,
        &request,
        &decision) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52SchedulerSelectsPipelineBatchWidth(void)
{
    SparkGlm52Scheduler scheduler;

    memset(&scheduler, 0, sizeof(scheduler));
    scheduler.spark_count = SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT;
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 0u, 256u) == 0u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 1u, 256u) == 1u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 4u, 256u) == 1u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 13u, 256u) == 1u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 14u, 256u) == 2u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 92u, 256u) == 8u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 184u, 256u) == 15u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 256u, 256u) == 20u);
    assert(SparkGlm52SchedulerSelectPipelineBatchWidth(
        &scheduler, 13312u, 256u) == 256u);
}

static void SparkTestGlm52SchedulerEstimatesExpandedDecodeWork(void)
{
    SparkGlm52PrefixCache cache;
    SparkGlm52PrefixCacheEntry entries[128u];
    SparkGlm52PrefixCacheSequenceBinding bindings[512u];
    SparkGlm52SchedulerConfiguration configuration;
    SparkGlm52Scheduler scheduler;
    uint64_t mtp_b1_work_ns;
    uint64_t mtp_b16_work_ns;
    uint64_t mtp_b1024_work_ns;
    uint64_t plain_b1_work_ns;
    uint64_t plain_b16_work_ns;
    uint64_t plain_b1024_work_ns;

    SparkTestInitializePrefixCache(&cache,entries,bindings,128u,512u);
    SparkTestInitializeSchedulerConfiguration(
        &configuration,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &cache);
    assert(SparkGlm52SchedulerInitialize(
        &scheduler,&configuration) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,1u,1u,112u,&plain_b1_work_ns) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,1u,6u,112u,&mtp_b1_work_ns) == SPARK_STATUS_OK);
    assert(plain_b1_work_ns == mtp_b1_work_ns);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,16u,1u,112u,&plain_b16_work_ns) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,16u,6u,112u,&mtp_b16_work_ns) == SPARK_STATUS_OK);
    assert(mtp_b16_work_ns > plain_b16_work_ns);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,1024u,1u,7168u,&plain_b1024_work_ns) == SPARK_STATUS_OK);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,1024u,6u,7168u,&mtp_b1024_work_ns) == SPARK_STATUS_OK);
    assert(mtp_b1024_work_ns > plain_b1024_work_ns);
    assert(SparkGlm52SchedulerEstimateDecodeWorkNs(
        &scheduler,1u,0u,112u,&mtp_b1_work_ns) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestGlm52SchedulerAdmitsCurrentSparkRingDecode();
    SparkTestGlm52SchedulerSupportsFp8AndPrefill();
    SparkTestGlm52SchedulerSupportsW8lut();
    SparkTestGlm52SchedulerUsesVllmStyleChunkedPrefill();
    SparkTestGlm52SchedulerUsesIntegratedPrefixCacheAdmission();
    SparkTestGlm52SchedulerDisablesCrossSequencePrefixReuse();
    SparkTestGlm52SchedulerInterleavesPrefillAndDecode();
    SparkTestGlm52SchedulerFillsCurrentSparkPipeline();
    SparkTestGlm52SchedulerPacksDecodeRequestsIntoSingleGraphDecision();
    SparkTestGlm52SchedulerDecodeBatchFillsMaxBucketFromOversubscribedQueue();
    SparkTestGlm52SchedulerUsesMeasuredDecodeBucketForMidSizedBatch();
    SparkTestGlm52SchedulerRejectsPrefillInDecodeBatch();
    SparkTestGlm52SchedulerExposesKvBlockTableAndCancelsReservation();
    SparkTestGlm52SchedulerBuildsBatchedPrefillKvTables();
    SparkTestGlm52SchedulerRejectsInvalidInputs();
    SparkTestGlm52SchedulerSelectsPipelineBatchWidth();
    SparkTestGlm52SchedulerEstimatesExpandedDecodeWork();
    return 0;
}
