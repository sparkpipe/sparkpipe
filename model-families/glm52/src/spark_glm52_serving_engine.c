#include "sparkpipe/spark_glm52_serving_engine.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static uint32_t SparkGlm52ServingNormalizeFlags(
    uint32_t flags)
{
    if (flags == 0u)
    {
        return SPARK_GLM52_SERVING_ENGINE_DEFAULT_FLAGS;
    }
    return flags;
}

static uint32_t SparkGlm52ServingNormalizeOutputTokenBudget(
    uint32_t output_token_budget)
{
    if (output_token_budget == 0u)
    {
        return SPARK_GLM52_SERVING_DEFAULT_OUTPUT_TOKEN_BUDGET;
    }
    return output_token_budget;
}

static uint32_t SparkGlm52ServingNormalizeMaxContextTokens(
    uint32_t max_context_tokens)
{
    if (max_context_tokens == 0u)
    {
        return SPARK_GLM52_SERVING_DEFAULT_MAX_CONTEXT_TOKENS;
    }
    return max_context_tokens;
}

static uint64_t SparkGlm52ServingNormalizeRequestIdBase(
    uint64_t request_id_base)
{
    if (request_id_base == 0u)
    {
        return SPARK_GLM52_SERVING_DEFAULT_REQUEST_ID_BASE;
    }
    return request_id_base;
}

static uint32_t SparkGlm52ServingEventRingSafetyMargin(void)
{
    return (SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT *
            SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE) +
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT + 8u;
}

static uint32_t SparkGlm52ServingEventRingFreeCount(
    const SparkGlm52ServingEngine *engine)
{
    if (engine == 0 || engine->event_ring_capacity < engine->event_count)
    {
        return 0u;
    }
    return engine->event_ring_capacity - engine->event_count;
}

static uint32_t SparkGlm52ServingRuntimeContractIsProduction(
    uint32_t runtime_contract_flags)
{
    if ((runtime_contract_flags &
            SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY) != 0u)
    {
        return 0u;
    }
    return (runtime_contract_flags &
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS) ==
        SPARK_GLM52_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
}

static uint32_t SparkGlm52ServingRuntimeContractMatchesRequestApi(
    const SparkGlm52RequestApi *request_api,
    uint32_t runtime_contract_flags)
{
    uint32_t configuration_flags;

    if (request_api == 0)
    {
        return 0u;
    }
    configuration_flags = request_api->configuration_flags;
    if ((runtime_contract_flags &
            SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED) != 0u)
    {
        if ((configuration_flags &
                SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING) == 0u ||
            (configuration_flags &
                SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING) == 0u)
        {
            return 0u;
        }
    }
    if ((runtime_contract_flags &
            SPARK_GLM52_SERVING_RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED) != 0u)
    {
        if ((configuration_flags &
                SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) == 0u)
        {
            return 0u;
        }
    }
    return 1u;
}

static SparkStatus SparkGlm52ServingValidateConfiguration(
    const SparkGlm52ServingEngineConfiguration *configuration)
{
    uint32_t flags;

    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    flags = SparkGlm52ServingNormalizeFlags(configuration->flags);
    if ((flags & ~SPARK_GLM52_SERVING_ENGINE_KNOWN_FLAGS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((flags &
            SPARK_GLM52_SERVING_ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT) != 0u &&
        !SparkGlm52ServingRuntimeContractIsProduction(
            configuration->runtime_contract_flags))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (configuration->request_api == 0 ||
        configuration->request_api->abi_version !=
            SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        configuration->request_api->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DESCRIPTOR_BYTES ||
        configuration->request_records == 0 ||
        configuration->request_record_capacity == 0u ||
        configuration->event_ring == 0 ||
        configuration->event_ring_capacity <= SparkGlm52ServingEventRingSafetyMargin() ||
        configuration->host_prefill_token_ids == 0 ||
        configuration->host_prefill_token_stride == 0u ||
        configuration->host_prefill_lane_capacity == 0u ||
        configuration->host_physical_block_indices == 0 ||
        configuration->kv_block_lane_stride == 0u ||
        configuration->kv_block_lane_capacity == 0u ||
        configuration->kv_block_lane_stride < configuration->kv_block_lane_capacity ||
        configuration->lane_physical_block_counts == 0 ||
        configuration->lane_count_capacity == 0u ||
        configuration->prefill_function == 0 ||
        configuration->decode_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((flags &
            SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0u)
    {
        if (configuration->request_token_storage != 0 ||
            configuration->request_token_stride != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (configuration->request_token_storage == 0 ||
        configuration->request_token_stride == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52ServingRuntimeContractMatchesRequestApi(
            configuration->request_api,
            configuration->runtime_contract_flags))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->stop_token_id_count > SPARK_GLM52_SERVING_MAX_STOP_TOKEN_IDS ||
        (configuration->stop_token_id_count != 0u &&
            configuration->stop_token_ids == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52ServingInitializeRequestRecord(
    SparkGlm52ServingRequestRecord *record,
    uint32_t *token_storage,
    uint32_t token_capacity)
{
    memset(record, 0, sizeof(*record));
    record->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    record->descriptor_bytes = SPARK_GLM52_SERVING_REQUEST_RECORD_DESCRIPTOR_BYTES;
    record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE;
    record->handle_hash_next = SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    record->free_record_next = SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    record->token_ids = token_storage;
    record->token_capacity = token_capacity;
}

static uint32_t SparkGlm52ServingHashHandle(
    SparkGlm52ServingRequestHandle request_handle)
{
    uint64_t hash;

    hash = request_handle;
    hash ^= (hash >> 33u);
    hash *= 0xff51afd7ed558ccdull;
    hash ^= (hash >> 33u);
    return (uint32_t)(hash % SPARK_GLM52_SERVING_RECORD_HASH_SLOTS);
}

static uint32_t SparkGlm52ServingRecordIndex(
    const SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingRequestRecord *record)
{
    uint64_t byte_offset;
    uint64_t record_index;

    if (engine == 0 || record == 0 || engine->request_records == 0 ||
        record < engine->request_records ||
        record >= &engine->request_records[engine->request_record_capacity])
    {
        return SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    }
    byte_offset =
        (uint64_t)((uintptr_t)record - (uintptr_t)engine->request_records);
    record_index = byte_offset / (uint64_t)sizeof(*record);
    if (record_index >= engine->request_record_capacity)
    {
        return SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    }
    return (uint32_t)record_index;
}

static void SparkGlm52ServingInsertRecordHash(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestRecord *record)
{
    uint32_t record_index;
    uint32_t hash_slot;

    record_index = SparkGlm52ServingRecordIndex(engine, record);
    if (record_index == SPARK_GLM52_SERVING_NO_RECORD_SLOT ||
        record->request_handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return;
    }
    hash_slot = SparkGlm52ServingHashHandle(record->request_handle);
    record->handle_hash_next = engine->request_handle_hash_heads[hash_slot];
    engine->request_handle_hash_heads[hash_slot] = record_index;
}

static void SparkGlm52ServingRemoveRecordHash(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestRecord *record)
{
    uint32_t record_index;
    uint32_t hash_slot;
    uint32_t current_index;
    uint32_t previous_index;

    record_index = SparkGlm52ServingRecordIndex(engine, record);
    if (record_index == SPARK_GLM52_SERVING_NO_RECORD_SLOT ||
        record->request_handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return;
    }
    hash_slot = SparkGlm52ServingHashHandle(record->request_handle);
    current_index = engine->request_handle_hash_heads[hash_slot];
    previous_index = SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    while (current_index != SPARK_GLM52_SERVING_NO_RECORD_SLOT)
    {
        if (current_index == record_index)
        {
            if (previous_index == SPARK_GLM52_SERVING_NO_RECORD_SLOT)
            {
                engine->request_handle_hash_heads[hash_slot] =
                    engine->request_records[current_index].handle_hash_next;
            }
            else
            {
                engine->request_records[previous_index].handle_hash_next =
                    engine->request_records[current_index].handle_hash_next;
            }
            engine->request_records[current_index].handle_hash_next =
                SPARK_GLM52_SERVING_NO_RECORD_SLOT;
            return;
        }
        previous_index = current_index;
        current_index = engine->request_records[current_index].handle_hash_next;
    }
}

static void SparkGlm52ServingInitializeEvent(
    SparkGlm52ServingEvent *event)
{
    memset(event, 0, sizeof(*event));
    event->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    event->descriptor_bytes = SPARK_GLM52_SERVING_EVENT_DESCRIPTOR_BYTES;
}

static void SparkGlm52ServingRefreshStats(
    SparkGlm52ServingEngine *engine)
{
    SparkGlm52ServingStats *stats;
    uint32_t record_index;
    uint32_t live_request_count;

    stats = &engine->stats;
    stats->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    stats->descriptor_bytes = SPARK_GLM52_SERVING_STATS_DESCRIPTOR_BYTES;
    live_request_count = 0u;
    for (record_index = 0u;
         record_index < engine->request_record_capacity;
         ++record_index)
    {
        if (engine->request_records[record_index].state !=
            SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE)
        {
            live_request_count += 1u;
        }
    }
    stats->live_request_count = live_request_count;
    stats->queued_request_count = engine->request_api->queued_request_count;
    stats->completed_request_count = engine->request_api->completed_request_count;
    stats->cancelled_request_count = engine->request_api->cancelled_request_count;
    stats->event_count = engine->event_count;
    stats->event_capacity = engine->event_ring_capacity;
    stats->dropped_event_count = engine->dropped_event_count;
    stats->jit_prefetch_dispatch_count =
        engine->request_api->jit_prefetch_dispatch_count;
    stats->jit_prefetch_block_count =
        engine->request_api->jit_prefetch_block_count;
    stats->async_jit_prefetch_start_count =
        engine->request_api->async_jit_prefetch_start_count;
    stats->async_jit_prefetch_poll_count =
        engine->request_api->async_jit_prefetch_poll_count;
    stats->async_jit_prefetch_completion_count =
        engine->request_api->async_jit_prefetch_completion_count;
    stats->prefix_family_dispatch_count =
        engine->request_api->prefix_family_dispatch_count;
    stats->prefix_family_member_count =
        engine->request_api->prefix_family_member_count;
    stats->prefix_family_saved_prompt_token_count =
        engine->request_api->prefix_family_saved_prompt_token_count;
    stats->mtp_draft_ready_count =
        engine->request_api->mtp_draft_ready_count;
    stats->mtp_accepted_draft_token_count =
        engine->request_api->mtp_accepted_draft_token_count;
    stats->mtp_committed_token_count =
        engine->request_api->mtp_committed_token_count;
    stats->mtp_rejected_token_count =
        engine->request_api->mtp_rejected_token_count;
}

void SparkGlm52ServingInitializeSubmitTextRequest(
    SparkGlm52ServingSubmitTextRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SERVING_SUBMIT_TEXT_DESCRIPTOR_BYTES;
}

void SparkGlm52ServingInitializeSubmitTokenIdsRequest(
    SparkGlm52ServingSubmitTokenIdsRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    request->descriptor_bytes = SPARK_GLM52_SERVING_SUBMIT_TOKENS_DESCRIPTOR_BYTES;
}

void SparkGlm52ServingInitializeDecodeResult(
    SparkGlm52ServingDecodeResult *decode_result,
    uint32_t lane_count,
    uint32_t token_stride)
{
    if (decode_result == 0)
    {
        return;
    }
    memset(decode_result, 0, sizeof(*decode_result));
    decode_result->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    decode_result->descriptor_bytes = SPARK_GLM52_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES;
    decode_result->lane_count = lane_count;
    decode_result->token_stride = token_stride;
}

void SparkGlm52ServingEngineDestroy(
    SparkGlm52ServingEngine *engine)
{
    if (engine == 0)
    {
        return;
    }
    SparkTokenizerWorkspaceDestroy(&engine->tokenizer_workspace);
}

SparkStatus SparkGlm52ServingEngineInitialize(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingEngineConfiguration *configuration)
{
    uint32_t record_index;
    uint32_t stop_index;
    uint32_t hash_index;
    SparkStatus status;

    if (engine == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ServingValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(engine, 0, sizeof(*engine));
    engine->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    engine->descriptor_bytes = SPARK_GLM52_SERVING_ENGINE_DESCRIPTOR_BYTES;
    engine->flags = SparkGlm52ServingNormalizeFlags(configuration->flags);
    engine->runtime_contract_flags = configuration->runtime_contract_flags;
    engine->default_thinking_token_budget =
        configuration->default_thinking_token_budget;
    engine->default_output_token_budget = SparkGlm52ServingNormalizeOutputTokenBudget(
        configuration->default_output_token_budget);
    engine->default_max_prefill_tokens_per_step =
        configuration->default_max_prefill_tokens_per_step;
    engine->max_context_tokens = SparkGlm52ServingNormalizeMaxContextTokens(
        configuration->max_context_tokens);
    engine->next_generated_request_id = SparkGlm52ServingNormalizeRequestIdBase(
        configuration->request_id_base);
    engine->request_api = configuration->request_api;
    engine->tokenizer = configuration->tokenizer;
    SparkTokenizerWorkspaceReset(&engine->tokenizer_workspace);
    engine->request_records = configuration->request_records;
    engine->request_record_capacity = configuration->request_record_capacity;
    engine->request_token_storage = configuration->request_token_storage;
    engine->request_token_stride = configuration->request_token_stride;
    engine->event_ring = configuration->event_ring;
    engine->event_ring_capacity = configuration->event_ring_capacity;
    engine->host_prefill_token_ids = configuration->host_prefill_token_ids;
    engine->host_prefill_token_stride = configuration->host_prefill_token_stride;
    engine->host_prefill_lane_capacity = configuration->host_prefill_lane_capacity;
    engine->host_physical_block_indices = configuration->host_physical_block_indices;
    engine->execution_physical_block_indices =
        configuration->execution_physical_block_indices;
    engine->kv_block_lane_stride = configuration->kv_block_lane_stride;
    engine->kv_block_lane_capacity = configuration->kv_block_lane_capacity;
    engine->lane_physical_block_counts =
        configuration->lane_physical_block_counts;
    engine->lane_count_capacity = configuration->lane_count_capacity;
    engine->prefill_function = configuration->prefill_function;
    engine->decode_function = configuration->decode_function;
    engine->release_sequence_function =
        configuration->release_sequence_function;
    engine->callback_context = configuration->callback_context;
    engine->stop_token_id_count = configuration->stop_token_id_count;
    for (stop_index = 0u;
         stop_index < configuration->stop_token_id_count;
         ++stop_index)
    {
        engine->stop_token_ids[stop_index] = configuration->stop_token_ids[stop_index];
    }

    for (hash_index = 0u;
         hash_index < SPARK_GLM52_SERVING_RECORD_HASH_SLOTS;
         ++hash_index)
    {
        engine->request_handle_hash_heads[hash_index] =
            SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    }
    for (record_index = 0u;
         record_index < engine->request_record_capacity;
         ++record_index)
    {
        uint32_t *token_storage;
        uint32_t token_capacity;

        if ((engine->flags &
                SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0u)
        {
            token_storage = 0;
            token_capacity = 0u;
        }
        else
        {
            token_storage = &engine->request_token_storage[
                (uint64_t)record_index * engine->request_token_stride];
            token_capacity = engine->request_token_stride;
        }
        SparkGlm52ServingInitializeRequestRecord(
            &engine->request_records[record_index],
            token_storage,
            token_capacity);
        engine->request_records[record_index].free_record_next =
            record_index + 1u < engine->request_record_capacity
                ? record_index + 1u
                : SPARK_GLM52_SERVING_NO_RECORD_SLOT;
    }
    engine->free_record_head = 0u;
    for (record_index = 0u;
         record_index < engine->event_ring_capacity;
         ++record_index)
    {
        SparkGlm52ServingInitializeEvent(&engine->event_ring[record_index]);
    }
    SparkGlm52ServingRefreshStats(engine);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingValidateEngine(
    SparkGlm52ServingEngine *engine)
{
    if (engine == 0 ||
        engine->abi_version != SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
        engine->descriptor_bytes != SPARK_GLM52_SERVING_ENGINE_DESCRIPTOR_BYTES ||
        engine->request_api == 0 ||
        engine->request_records == 0 ||
        (((engine->flags &
                SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) == 0u) &&
            engine->request_token_storage == 0) ||
        engine->event_ring == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkGlm52ServingRequestRecord *SparkGlm52ServingFindFreeRecord(
    SparkGlm52ServingEngine *engine)
{
    if (engine == 0 ||
        engine->free_record_head == SPARK_GLM52_SERVING_NO_RECORD_SLOT ||
        engine->free_record_head >= engine->request_record_capacity)
    {
        return 0;
    }
    if (engine->request_records[engine->free_record_head].state !=
        SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE)
    {
        return 0;
    }
    return &engine->request_records[engine->free_record_head];
}

static SparkStatus SparkGlm52ServingEnsureRecordTokenCapacity(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestRecord *record,
    uint32_t required_token_capacity)
{
    uint32_t grown_capacity;
    uint32_t *grown_token_ids;

    if (engine == 0 || record == 0 || required_token_capacity == 0u ||
        required_token_capacity > engine->max_context_tokens)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (required_token_capacity <= record->token_capacity)
    {
        return SPARK_STATUS_OK;
    }
    if ((engine->flags &
            SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    grown_capacity = record->token_capacity != 0u
        ? record->token_capacity
        : 64u;
    while (grown_capacity < required_token_capacity &&
        grown_capacity <= engine->max_context_tokens / 2u)
    {
        grown_capacity *= 2u;
    }
    if (grown_capacity < required_token_capacity)
    {
        grown_capacity = required_token_capacity;
    }
    if (grown_capacity > engine->max_context_tokens)
    {
        grown_capacity = engine->max_context_tokens;
    }
    grown_token_ids = (uint32_t *)realloc(
        record->token_ids,
        (size_t)grown_capacity * sizeof(record->token_ids[0u]));
    if (grown_token_ids == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    record->token_ids = grown_token_ids;
    record->token_capacity = grown_capacity;
    return SPARK_STATUS_OK;
}

static SparkGlm52ServingRequestRecord *SparkGlm52ServingFindRecordByHandle(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestHandle request_handle)
{
    uint32_t hash_slot;
    uint32_t record_index;

    if (request_handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return 0;
    }
    hash_slot = SparkGlm52ServingHashHandle(request_handle);
    record_index = engine->request_handle_hash_heads[hash_slot];
    while (record_index != SPARK_GLM52_SERVING_NO_RECORD_SLOT &&
           record_index < engine->request_record_capacity)
    {
        SparkGlm52ServingRequestRecord *record;

        record = &engine->request_records[record_index];
        if (record->state != SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE &&
            record->request_handle == request_handle)
        {
            return record;
        }
        record_index = record->handle_hash_next;
    }
    return 0;
}

static SparkGlm52ServingRequestRecord *SparkGlm52ServingFindRecordByRequestId(
    SparkGlm52ServingEngine *engine,
    uint64_t request_id)
{
    SparkGlm52ServingRequestRecord *record;
    uint32_t record_index;

    for (record_index = 0u;
         record_index < engine->request_record_capacity;
         ++record_index)
    {
        record = &engine->request_records[record_index];
        if (record->state != SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE &&
            record->request_id == request_id)
        {
            return record;
        }
    }
    return 0;
}

static uint32_t SparkGlm52ServingTokenIsStopToken(
    const SparkGlm52ServingEngine *engine,
    uint32_t token_id)
{
    uint32_t stop_index;

    for (stop_index = 0u;
         stop_index < engine->stop_token_id_count;
         ++stop_index)
    {
        if (engine->stop_token_ids[stop_index] == token_id)
        {
            return 1u;
        }
    }
    return 0u;
}

static SparkStatus SparkGlm52ServingPushEvent(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingEvent *event)
{
    SparkGlm52ServingEvent *destination;

    if (engine == 0 || event == 0 || engine->event_ring_capacity == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (engine->event_count >= engine->event_ring_capacity)
    {
        engine->dropped_event_count += 1u;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    destination = &engine->event_ring[engine->event_write_index];
    *destination = *event;
    engine->event_write_index += 1u;
    if (engine->event_write_index == engine->event_ring_capacity)
    {
        engine->event_write_index = 0u;
    }
    engine->event_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingPushSimpleEvent(
    SparkGlm52ServingEngine *engine,
    uint32_t event_kind,
    uint32_t status,
    const SparkGlm52ServingRequestRecord *record)
{
    SparkGlm52ServingEvent event;

    SparkGlm52ServingInitializeEvent(&event);
    event.kind = event_kind;
    event.status = status;
    if (record != 0)
    {
        event.request_id = record->request_id;
        event.sequence_id = record->sequence_id;
        event.request_handle = record->request_handle;
        event.prompt_token_count = record->prompt_token_count;
    }
    return SparkGlm52ServingPushEvent(engine, &event);
}

static void SparkGlm52ServingFailDispatchRequests(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkStatus failure_status)
{
    uint32_t lane_index;

    if (engine == 0 || dispatch == 0)
    {
        return;
    }
    for (lane_index = 0u;
         lane_index < dispatch->request_count &&
             lane_index < SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
         ++lane_index)
    {
        SparkGlm52ServingRequestRecord *record;

        record = SparkGlm52ServingFindRecordByHandle(
            engine,
            dispatch->request_handles[lane_index]);
        if (record == 0 ||
            record->state == SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_COMPLETED ||
            record->state == SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_CANCELLED)
        {
            continue;
        }
        record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_CANCELLED;
        (void)SparkGlm52ServingPushSimpleEvent(
            engine,
            SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_CANCELLED,
            (uint32_t)failure_status,
            record);
    }
}

static SparkStatus SparkGlm52ServingApplyContextBudget(
    SparkGlm52ServingEngine *engine,
    uint32_t prompt_token_count,
    uint32_t requested_thinking_token_budget,
    uint32_t requested_output_token_budget,
    uint32_t *thinking_token_budget_out,
    uint32_t *output_token_budget_out)
{
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t available_generation_tokens;

    if (thinking_token_budget_out == 0 || output_token_budget_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (prompt_token_count == 0u || prompt_token_count > engine->max_context_tokens)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    thinking_token_budget = requested_thinking_token_budget != 0u
        ? requested_thinking_token_budget
        : engine->default_thinking_token_budget;
    output_token_budget = requested_output_token_budget != 0u
        ? requested_output_token_budget
        : engine->default_output_token_budget;
    if ((engine->flags &
            SPARK_GLM52_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT) == 0u)
    {
        if ((uint64_t)prompt_token_count + thinking_token_budget +
            output_token_budget > engine->max_context_tokens)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        *thinking_token_budget_out = thinking_token_budget;
        *output_token_budget_out = output_token_budget;
        return SPARK_STATUS_OK;
    }

    available_generation_tokens = engine->max_context_tokens - prompt_token_count;
    if (thinking_token_budget > available_generation_tokens)
    {
        thinking_token_budget = available_generation_tokens;
    }
    available_generation_tokens -= thinking_token_budget;
    if (output_token_budget > available_generation_tokens)
    {
        output_token_budget = available_generation_tokens;
    }
    if (thinking_token_budget == 0u && output_token_budget == 0u)
    {
        output_token_budget = 1u;
        if ((uint64_t)prompt_token_count + output_token_budget >
            engine->max_context_tokens)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }

    *thinking_token_budget_out = thinking_token_budget;
    *output_token_budget_out = output_token_budget;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingResolveSubmittedSequence(
    SparkGlm52RequestApi *request_api,
    SparkGlm52RequestApiHandle request_handle,
    uint64_t request_id,
    uint64_t *sequence_id_out)
{
    SparkGlm52RequestApiCacheState cache_state;
    SparkStatus status;

    if (request_api == 0 || request_handle == 0u || request_id == 0u ||
        sequence_id_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52RequestApiGetRequestCacheState(
        request_api,
        request_handle,
        &cache_state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (cache_state.request_id != request_id || cache_state.sequence_id == 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    *sequence_id_out = cache_state.sequence_id;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingSubmitPreparedRecord(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestRecord *record,
    uint32_t request_flags,
    uint32_t priority,
    uint32_t thinking_token_budget,
    uint32_t output_token_budget,
    uint32_t max_prefill_tokens_per_step,
    uint64_t request_id,
    uint64_t sequence_id,
    SparkGlm52ServingSubmitResult *result)
{
    SparkGlm52RequestApiSubmitRequest api_request;
    SparkGlm52RequestApiHandle request_handle;
    uint32_t record_index;
    SparkStatus status;

    memset(&api_request, 0, sizeof(api_request));
    api_request.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api_request.descriptor_bytes = SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES;
    api_request.flags = request_flags;
    api_request.priority = priority;
    api_request.prompt_token_count = record->prompt_token_count;
    api_request.thinking_token_budget = thinking_token_budget;
    api_request.output_token_budget = output_token_budget;
    api_request.max_prefill_tokens_per_step = max_prefill_tokens_per_step != 0u
        ? max_prefill_tokens_per_step
        : engine->default_max_prefill_tokens_per_step;
    api_request.request_id = request_id;
    api_request.sequence_id = sequence_id;
    api_request.prompt_token_ids = record->token_ids;

    status = SparkGlm52RequestApiSubmit(
        engine->request_api,
        &api_request,
        &request_handle);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ServingResolveSubmittedSequence(
        engine->request_api,
        request_handle,
        request_id,
        &sequence_id);
    if (status != SPARK_STATUS_OK)
    {
        (void)SparkGlm52RequestApiCancelRequest(
            engine->request_api,
            request_handle);
        (void)SparkGlm52RequestApiReleaseCompletedRequest(
            engine->request_api,
            request_handle);
        return status;
    }

    record_index = SparkGlm52ServingRecordIndex(engine, record);
    if (record_index == SPARK_GLM52_SERVING_NO_RECORD_SLOT ||
        engine->free_record_head != record_index ||
        record->state != SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_FREE)
    {
        (void)SparkGlm52RequestApiCancelRequest(
            engine->request_api,
            request_handle);
        (void)SparkGlm52RequestApiReleaseCompletedRequest(
            engine->request_api,
            request_handle);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    engine->free_record_head = record->free_record_next;
    record->free_record_next = SPARK_GLM52_SERVING_NO_RECORD_SLOT;

    record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_SUBMITTED;
    record->flags = request_flags;
    record->thinking_token_budget = thinking_token_budget;
    record->output_token_budget = output_token_budget;
    record->request_id = request_id;
    record->sequence_id = sequence_id;
    record->request_handle = request_handle;
    SparkGlm52ServingInsertRecordHash(engine, record);
    engine->stats.submitted_request_count += 1u;
    engine->stats.accepted_request_count += 1u;

    if (result != 0)
    {
        memset(result, 0, sizeof(*result));
        result->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
        result->descriptor_bytes = SPARK_GLM52_SERVING_SUBMIT_RESULT_DESCRIPTOR_BYTES;
        result->prompt_token_count = record->prompt_token_count;
        result->required_token_capacity = record->prompt_token_count;
        result->thinking_token_budget = thinking_token_budget;
        result->output_token_budget = output_token_budget;
        result->request_id = request_id;
        result->sequence_id = sequence_id;
        result->request_handle = request_handle;
    }

    status = SparkGlm52ServingPushSimpleEvent(
        engine,
        SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_ACCEPTED,
        SPARK_STATUS_OK,
        record);
    SparkGlm52ServingRefreshStats(engine);
    return status;
}

SparkStatus SparkGlm52ServingEngineSubmitTokenIds(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingSubmitTokenIdsRequest *request,
    SparkGlm52ServingSubmitResult *result)
{
    SparkGlm52ServingRequestRecord *record;
    uint32_t token_index;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint64_t required_token_capacity;
    uint64_t request_id;
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (request == 0 ||
        request->abi_version != SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_SERVING_SUBMIT_TOKENS_DESCRIPTOR_BYTES ||
        (request->flags & ~SPARK_GLM52_SERVING_SUBMIT_KNOWN_FLAGS) != 0u ||
        request->token_count == 0u ||
        request->token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    record = SparkGlm52ServingFindFreeRecord(engine);
    if (record == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkGlm52ServingApplyContextBudget(
        engine,
        request->token_count,
        request->thinking_token_budget,
        request->output_token_budget,
        &thinking_token_budget,
        &output_token_budget);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    required_token_capacity = (uint64_t)request->token_count +
        thinking_token_budget + output_token_budget;
    if (required_token_capacity > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkGlm52ServingEnsureRecordTokenCapacity(
        engine,
        record,
        (uint32_t)required_token_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    for (token_index = 0u; token_index < request->token_count; ++token_index)
    {
        record->token_ids[token_index] = request->token_ids[token_index];
    }
    record->prompt_token_count = request->token_count;
    request_id = request->request_id != 0u
        ? request->request_id
        : engine->next_generated_request_id++;
    return SparkGlm52ServingSubmitPreparedRecord(
        engine,
        record,
        request->flags,
        request->priority,
        thinking_token_budget,
        output_token_budget,
        request->max_prefill_tokens_per_step,
        request_id,
        request->sequence_id,
        result);
}

SparkStatus SparkGlm52ServingEngineSubmitText(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingSubmitTextRequest *request,
    SparkGlm52ServingSubmitResult *result)
{
    SparkGlm52ServingRequestRecord *record;
    SparkTokenizerEncoding encoding;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t required_prompt_token_capacity;
    uint64_t required_token_capacity;
    uint32_t text_bytes;
    uint64_t request_id;
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (request == 0 ||
        request->abi_version != SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_SERVING_SUBMIT_TEXT_DESCRIPTOR_BYTES ||
        (request->flags & ~SPARK_GLM52_SERVING_SUBMIT_KNOWN_FLAGS) != 0u ||
        request->text == 0 ||
        engine->tokenizer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (request->text_bytes == 0u)
    {
        size_t text_length;

        text_length = strlen(request->text);
        if (text_length == 0u || text_length > UINT32_MAX)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        text_bytes = (uint32_t)text_length;
    }
    else
    {
        text_bytes = request->text_bytes;
    }

    record = SparkGlm52ServingFindFreeRecord(engine);
    if (record == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    memset(&encoding, 0, sizeof(encoding));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_ids = record->token_ids;
    encoding.token_capacity = record->token_capacity;
    status = SparkTokenizerEncodeUtf8WithWorkspace(
        engine->tokenizer,
        request->text,
        text_bytes,
        request->tokenizer_encode_flags,
        &engine->tokenizer_workspace,
        &encoding);
    if (status == SPARK_STATUS_CAPACITY_EXCEEDED &&
        (engine->flags &
            SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0u)
    {
        required_prompt_token_capacity =
            encoding.token_count + encoding.overflow_token_count;
        status = SparkGlm52ServingEnsureRecordTokenCapacity(
            engine,
            record,
            required_prompt_token_capacity);
        if (status == SPARK_STATUS_OK)
        {
            SparkTokenizerEncodingReset(&encoding);
            encoding.token_ids = record->token_ids;
            encoding.token_capacity = record->token_capacity;
            status = SparkTokenizerEncodeUtf8WithWorkspace(
                engine->tokenizer,
                request->text,
                text_bytes,
                request->tokenizer_encode_flags,
                &engine->tokenizer_workspace,
                &encoding);
        }
    }
    if (status != SPARK_STATUS_OK)
    {
        if (result != 0)
        {
            memset(result, 0, sizeof(*result));
            result->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
            result->descriptor_bytes = SPARK_GLM52_SERVING_SUBMIT_RESULT_DESCRIPTOR_BYTES;
            result->prompt_token_count = encoding.token_count;
            result->required_token_capacity = encoding.token_count +
                encoding.overflow_token_count;
        }
        return status;
    }

    status = SparkGlm52ServingApplyContextBudget(
        engine,
        encoding.token_count,
        request->thinking_token_budget,
        request->output_token_budget,
        &thinking_token_budget,
        &output_token_budget);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    required_token_capacity = (uint64_t)encoding.token_count +
        thinking_token_budget + output_token_budget;
    if (required_token_capacity > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    status = SparkGlm52ServingEnsureRecordTokenCapacity(
        engine,
        record,
        (uint32_t)required_token_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    record->prompt_token_count = encoding.token_count;
    request_id = request->request_id != 0u
        ? request->request_id
        : engine->next_generated_request_id++;
    return SparkGlm52ServingSubmitPreparedRecord(
        engine,
        record,
        request->flags,
        request->priority,
        thinking_token_budget,
        output_token_budget,
        request->max_prefill_tokens_per_step,
        request_id,
        request->sequence_id,
        result);
}

static SparkStatus SparkGlm52ServingBuildPrefillDispatch(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t step_index,
    SparkGlm52RequestApiPrefillDispatchView *prefill_view,
    SparkGlm52KvBlockTableView *block_table_view,
    SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
    SparkStatus status;

    status = SparkGlm52RequestApiDescribePrefillDispatch(
        dispatch,
        prefill_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefill_view->lane_count > engine->host_prefill_lane_capacity ||
        prefill_view->lane_count > engine->lane_count_capacity ||
        prefill_view->prompt_token_stride > engine->host_prefill_token_stride)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SparkGlm52RequestApiCopyPrefillDispatchTokenIds(
        dispatch,
        engine->host_prefill_token_ids,
        engine->host_prefill_token_stride,
        engine->host_prefill_lane_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52RequestApiBuildDispatchKvBlockTableView(
        engine->request_api,
        dispatch,
        engine->host_physical_block_indices,
        engine->execution_physical_block_indices,
        engine->kv_block_lane_stride,
        engine->kv_block_lane_capacity,
        engine->lane_physical_block_counts,
        engine->lane_count_capacity,
        block_table_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(prefill_dispatch, 0, sizeof(*prefill_dispatch));
    prefill_dispatch->abi_version = SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION;
    prefill_dispatch->descriptor_bytes =
        SPARK_GLM52_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES;
    prefill_dispatch->step_index = step_index;
    prefill_dispatch->dispatch_kind = dispatch->kind;
    prefill_dispatch->active_sequence_count = prefill_view->active_sequence_count;
    prefill_dispatch->lane_count = prefill_view->lane_count;
    prefill_dispatch->prompt_token_offset = prefill_view->prompt_token_offset;
    prefill_dispatch->prompt_token_count = prefill_view->prompt_token_count;
    prefill_dispatch->prompt_token_stride = prefill_view->prompt_token_stride;
    prefill_dispatch->host_token_stride = engine->host_prefill_token_stride;
    prefill_dispatch->request_dispatch = dispatch;
    prefill_dispatch->prefill_view = prefill_view;
    prefill_dispatch->host_token_ids = engine->host_prefill_token_ids;
    prefill_dispatch->kv_block_table_view = block_table_view;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingPublishPrefillEvents(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
    uint32_t lane_index;

    for (lane_index = 0u;
         lane_index < prefill_dispatch->prefill_view->lane_count;
         ++lane_index)
    {
        SparkGlm52ServingEvent event;
        const SparkGlm52RequestApiPrefillDispatchLaneView *lane;

        lane = &prefill_dispatch->prefill_view->lanes[lane_index];
        SparkGlm52ServingInitializeEvent(&event);
        event.kind = SPARK_GLM52_SERVING_EVENT_KIND_PREFILL_PROGRESS;
        event.status = SPARK_STATUS_OK;
        event.prompt_token_offset = lane->prompt_token_offset;
        event.prompt_token_count = lane->prompt_token_count;
        event.dispatch_kind = prefill_dispatch->dispatch_kind;
        event.dispatch_flags = prefill_dispatch->request_dispatch->flags;
        event.request_id = lane->request_id;
        event.sequence_id = lane->sequence_id;
        event.request_handle = lane->request_handle;
        if (SparkGlm52ServingPushEvent(engine, &event) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingInvokePrefill(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t step_index)
{
    SparkGlm52RequestApiPrefillDispatchView prefill_view;
    SparkGlm52KvBlockTableView block_table_view;
    SparkGlm52PromptPipelinePrefillDispatch prefill_dispatch;
    SparkStatus status;

    status = SparkGlm52ServingBuildPrefillDispatch(
        engine,
        dispatch,
        step_index,
        &prefill_view,
        &block_table_view,
        &prefill_dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = engine->prefill_function(
        engine->callback_context,
        &prefill_dispatch);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_PENDING)
    {
        return status;
    }

    if (SparkGlm52ServingPublishPrefillEvents(
            engine,
            &prefill_dispatch) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    engine->stats.prefill_dispatch_count += 1u;
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        engine->stats.prefill_batch_dispatch_count += 1u;
    }
    engine->stats.prefill_token_count +=
        (uint64_t)prefill_view.prompt_token_count *
        (uint64_t)prefill_view.lane_count;
    if (prefill_view.active_sequence_count >
        engine->stats.maximum_prefill_active_sequence_count)
    {
        engine->stats.maximum_prefill_active_sequence_count =
            prefill_view.active_sequence_count;
    }
    if (prefill_view.lane_count > engine->stats.maximum_prefill_lane_count)
    {
        engine->stats.maximum_prefill_lane_count = prefill_view.lane_count;
    }
    return status;
}

static SparkStatus SparkGlm52ServingBuildDecodeDispatch(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52KvBlockTableView *block_table_view,
    SparkGlm52RequestApiDecodeDispatchView *decode_view,
    SparkGlm52ServingDecodeDispatch *decode_dispatch)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkGlm52RequestApiBuildDispatchKvBlockTableView(
        engine->request_api,
        dispatch,
        engine->host_physical_block_indices,
        engine->execution_physical_block_indices,
        engine->kv_block_lane_stride,
        engine->kv_block_lane_capacity,
        engine->lane_physical_block_counts,
        engine->lane_count_capacity,
        block_table_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52RequestApiDescribeDecodeDispatch(
        engine->request_api,
        dispatch,
        decode_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(decode_dispatch, 0, sizeof(*decode_dispatch));
    decode_dispatch->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
    decode_dispatch->descriptor_bytes =
        SPARK_GLM52_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES;
    decode_dispatch->dispatch_kind = dispatch->kind;
    decode_dispatch->request_count = dispatch->request_count;
    decode_dispatch->active_sequence_count = block_table_view->lane_count;
    decode_dispatch->request_dispatch = dispatch;
    decode_dispatch->kv_block_table_view = block_table_view;
    decode_dispatch->decode_view = decode_view;
    for (lane_index = 0u;
         lane_index < decode_view->lane_count;
         ++lane_index)
    {
        SparkGlm52ServingRequestRecord *record;
        uint32_t input_token_index;

        record = SparkGlm52ServingFindRecordByHandle(
            engine,
            dispatch->request_handles[lane_index]);
        if (record == 0 ||
            record->prompt_token_count == 0u ||
            record->prompt_token_count + record->streamed_decode_token_count >
                record->token_capacity)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        input_token_index =
            record->prompt_token_count + record->streamed_decode_token_count - 1u;
        decode_dispatch->input_token_ids[lane_index] =
            record->token_ids[input_token_index];
        if (dispatch->kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
        {
            uint32_t draft_index;

            decode_dispatch->speculative_token_count =
                dispatch->speculative_token_count;
            for (draft_index = 0u;
                 draft_index < dispatch->speculative_token_count;
                 ++draft_index)
            {
                decode_dispatch->speculative_draft_token_ids[lane_index][draft_index] =
                    dispatch->speculative_draft_token_ids[lane_index][draft_index];
            }
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingValidateDecodeResult(
    const SparkGlm52RequestApiDispatch *dispatch,
    const SparkGlm52ServingDecodeResult *decode_result)
{
    uint32_t lane_index;
    uint32_t maximum_token_count;

    if (dispatch == 0 || decode_result == 0 ||
        decode_result->abi_version != SPARK_GLM52_SERVING_ENGINE_ABI_VERSION ||
        decode_result->descriptor_bytes !=
            SPARK_GLM52_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES ||
        decode_result->lane_count != dispatch->request_count ||
        decode_result->lane_count == 0u ||
        decode_result->lane_count > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
        decode_result->token_stride == 0u ||
        decode_result->token_stride > SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    maximum_token_count = 1u;
    if (dispatch->kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        maximum_token_count = dispatch->speculative_verifier_token_count;
    }
    else if ((dispatch->flags &
        SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u)
    {
        maximum_token_count = dispatch->mtp_draft_token_budget + 1u;
    }
    if (maximum_token_count == 0u ||
        maximum_token_count > SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        if (decode_result->token_counts[lane_index] == 0u ||
            decode_result->token_counts[lane_index] > maximum_token_count ||
            decode_result->token_counts[lane_index] > decode_result->token_stride)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (decode_result->draft_token_counts[lane_index] >
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (decode_result->draft_token_counts[lane_index] != 0u &&
            (dispatch->kind !=
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH ||
             (dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) == 0u) &&
            (dispatch->kind !=
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
             (dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (dispatch->kind !=
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
            (dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u &&
            decode_result->token_counts[lane_index] != 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingResolveMtpDecode(
    SparkGlm52RequestApiDispatch *dispatch,
    const SparkGlm52ServingDecodeResult *decode_result)
{
    uint32_t lane_index;

    if (dispatch->kind != SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
        (dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    for (lane_index = 0u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        if (decode_result->token_counts[lane_index] == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        dispatch->decode_committed_token_counts[lane_index] = 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingCaptureMtpDraftTokens(
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52ServingDecodeResult *decode_result,
    uint32_t *draft_token_ids,
    uint32_t draft_lane_stride,
    uint32_t *draft_token_count_out)
{
    uint32_t lane_index;
    uint32_t draft_token_count;

    if (draft_token_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *draft_token_count_out = 0u;
    if (decode_result != 0 && decode_result->lane_count != 0u &&
        decode_result->draft_token_counts[0u] != 0u)
    {
        draft_token_count = decode_result->draft_token_counts[0u];
        if (((dispatch->kind !=
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH ||
              (dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) == 0u) &&
             (dispatch->kind !=
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
              (dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u)) ||
            draft_token_ids == 0 || draft_lane_stride < draft_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (lane_index = 0u;
             lane_index < decode_result->lane_count;
             ++lane_index)
        {
            uint32_t draft_index;

            if (decode_result->draft_token_counts[lane_index] !=
                draft_token_count)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            for (draft_index = 0u;
                 draft_index < draft_token_count;
                 ++draft_index)
            {
                draft_token_ids[(uint64_t)lane_index * draft_lane_stride +
                    draft_index] =
                    decode_result->draft_token_ids[lane_index][draft_index];
            }
        }
        *draft_token_count_out = draft_token_count;
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind != SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
        (dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (decode_result == 0 || draft_token_ids == 0 ||
        draft_lane_stride < dispatch->mtp_draft_token_budget)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    draft_token_count = 0u;
    for (lane_index = 0u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        uint32_t lane_draft_count;
        uint32_t draft_index;

        if (decode_result->token_counts[lane_index] == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        lane_draft_count = decode_result->token_counts[lane_index] - 1u;
        if (lane_draft_count > dispatch->mtp_draft_token_budget)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (lane_index == 0u)
        {
            draft_token_count = lane_draft_count;
        }
        else if (lane_draft_count != draft_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (draft_index = 0u;
             draft_index < lane_draft_count;
             ++draft_index)
        {
            draft_token_ids[(uint64_t)lane_index * draft_lane_stride + draft_index] =
                decode_result->token_ids[lane_index][draft_index + 1u];
        }
        decode_result->token_counts[lane_index] = 1u;
    }
    *draft_token_count_out = draft_token_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingClampSpeculativeVerifyDecodeResult(
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52ServingDecodeResult *decode_result)
{
    uint32_t lane_index;

    if (dispatch->kind !=
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return SPARK_STATUS_OK;
    }
    for (lane_index = 0u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        uint32_t accepted_token_count;
        uint32_t committed_token_count;

        accepted_token_count =
            dispatch->speculative_accepted_token_counts[lane_index];
        committed_token_count =
            dispatch->speculative_committed_token_counts[lane_index];
        if (committed_token_count == 0u ||
            committed_token_count != accepted_token_count + 1u ||
            committed_token_count > decode_result->token_counts[lane_index])
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        decode_result->token_ids[lane_index][accepted_token_count] =
            dispatch->speculative_fallback_token_ids[lane_index];
        decode_result->token_counts[lane_index] = committed_token_count;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingResolveSpeculativeDecode(
    SparkGlm52ServingEngine *engine,
    SparkGlm52RequestApiDispatch *dispatch,
    const SparkGlm52ServingDecodeResult *decode_result)
{
    uint32_t verifier_token_ids[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT *
        SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE];
    uint32_t lane_index;
    uint32_t token_index;
    uint32_t verifier_token_count;

    if (dispatch->kind !=
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return SPARK_STATUS_OK;
    }

    verifier_token_count = decode_result->token_counts[0u];
    for (lane_index = 1u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        if (decode_result->token_counts[lane_index] != verifier_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    for (lane_index = 0u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        for (token_index = 0u;
             token_index < verifier_token_count;
             ++token_index)
        {
            verifier_token_ids[(uint64_t)lane_index *
                SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE + token_index] =
                decode_result->token_ids[lane_index][token_index];
        }
    }

    return SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
        engine->request_api,
        dispatch,
        verifier_token_ids,
        SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE,
        verifier_token_count);
}

static SparkStatus SparkGlm52ServingPublishDecodeEvents(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch,
    const SparkGlm52ServingDecodeResult *decode_result,
    SparkGlm52ServingRequestHandle *finish_handles,
    uint32_t *finish_handle_count)
{
    uint32_t lane_index;

    *finish_handle_count = 0u;
    for (lane_index = 0u;
         lane_index < decode_result->lane_count;
         ++lane_index)
    {
        uint32_t lane_token_count;
        uint32_t token_index;
        uint32_t lane_finish;
        SparkGlm52ServingRequestRecord *record;

        record = SparkGlm52ServingFindRecordByHandle(
            engine,
            dispatch->request_handles[lane_index]);
        if (record == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        lane_finish = (decode_result->lane_flags[lane_index] &
            SPARK_GLM52_SERVING_DECODE_RESULT_FLAG_FINISH_REQUEST) != 0u;
        lane_token_count = decode_result->token_counts[lane_index];
        for (token_index = 0u; token_index < lane_token_count; ++token_index)
        {
            if (SparkGlm52ServingTokenIsStopToken(
                    engine,
                    decode_result->token_ids[lane_index][token_index]))
            {
                lane_token_count = token_index + 1u;
                lane_finish = 1u;
                break;
            }
        }
        for (token_index = 0u;
             token_index < lane_token_count;
             ++token_index)
        {
            uint32_t token_id;
            SparkGlm52ServingEvent event;

            token_id = decode_result->token_ids[lane_index][token_index];
            if (record->prompt_token_count + record->streamed_decode_token_count +
                    token_index >= record->token_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            record->token_ids[
                record->prompt_token_count +
                record->streamed_decode_token_count +
                token_index] = token_id;
            if ((decode_result->lane_flags[lane_index] &
                    SPARK_GLM52_SERVING_DECODE_RESULT_FLAG_TOKEN_STREAM_SUPPRESSED) != 0u)
            {
                continue;
            }
            SparkGlm52ServingInitializeEvent(&event);
            event.kind = SPARK_GLM52_SERVING_EVENT_KIND_TOKEN;
            event.status = SPARK_STATUS_OK;
            event.token_id = token_id;
            event.token_index = record->prompt_token_count +
                record->streamed_decode_token_count + token_index;
            event.token_count = lane_token_count;
            event.dispatch_kind = dispatch->kind;
            event.dispatch_flags = dispatch->flags;
            event.request_id = record->request_id;
            event.sequence_id = record->sequence_id;
            event.request_handle = record->request_handle;
            if (SparkGlm52ServingPushEvent(engine, &event) != SPARK_STATUS_OK)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            engine->stats.decoded_token_count += 1u;
        }
        record->streamed_decode_token_count += lane_token_count;
        if (lane_finish != 0u &&
            *finish_handle_count < SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
        {
            finish_handles[*finish_handle_count] =
                dispatch->request_handles[lane_index];
            *finish_handle_count += 1u;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingCompleteFinishedHandles(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52ServingRequestHandle *finish_handles,
    uint32_t finish_handle_count)
{
    uint32_t finish_index;

    for (finish_index = 0u;
         finish_index < finish_handle_count;
         ++finish_index)
    {
        SparkGlm52ServingRequestHandle request_handle;
        SparkGlm52ServingRequestRecord *record;
        SparkStatus status;

        request_handle = finish_handles[finish_index];
        record = SparkGlm52ServingFindRecordByHandle(engine, request_handle);
        if (record == 0)
        {
            continue;
        }
        status = SparkGlm52RequestApiFinishRequestGeneration(
            engine->request_api,
            request_handle);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
        record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_COMPLETED;
        status = SparkGlm52ServingPushSimpleEvent(
            engine,
            SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_COMPLETED,
            SPARK_STATUS_OK,
            record);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        engine->stats.completed_stream_count += 1u;
        if ((engine->flags &
                SPARK_GLM52_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS) != 0u)
        {
            (void)SparkGlm52ServingEngineReleaseCompletedRequest(
                engine,
                request_handle);
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ServingCompleteBudgetFinishedRequests(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t request_index;

    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiCacheState cache_state;
        SparkGlm52ServingRequestRecord *record;
        SparkStatus status;

        record = SparkGlm52ServingFindRecordByHandle(
            engine,
            dispatch->request_handles[request_index]);
        if (record == 0)
        {
            continue;
        }
        status = SparkGlm52RequestApiGetRequestCacheState(
            engine->request_api,
            record->request_handle,
            &cache_state);
        if (status == SPARK_STATUS_NOT_FOUND)
        {
            continue;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (cache_state.state == SPARK_GLM52_REQUEST_API_STATE_COMPLETED)
        {
            record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_COMPLETED;
            status = SparkGlm52ServingPushSimpleEvent(
                engine,
                SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_COMPLETED,
                SPARK_STATUS_OK,
                record);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            engine->stats.completed_stream_count += 1u;
            if ((engine->flags &
                    SPARK_GLM52_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS) != 0u)
            {
                (void)SparkGlm52ServingEngineReleaseCompletedRequest(
                    engine,
                    record->request_handle);
            }
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServingEngineCompleteDecodeDispatch(
    SparkGlm52ServingEngine *engine,
    SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52ServingDecodeResult *decode_result)
{
    SparkGlm52ServingRequestHandle finish_handles[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t mtp_draft_token_ids[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT *
        SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT];
    uint32_t mtp_draft_token_count;
    uint32_t finish_handle_count;
    SparkStatus status;

    if (engine == 0 || dispatch == 0 || decode_result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ServingValidateDecodeResult(dispatch, decode_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ServingResolveSpeculativeDecode(
        engine,
        dispatch,
        decode_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ServingClampSpeculativeVerifyDecodeResult(
        dispatch,
        decode_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    mtp_draft_token_count = 0u;
    status = SparkGlm52ServingCaptureMtpDraftTokens(
        dispatch,
        decode_result,
        mtp_draft_token_ids,
        SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT,
        &mtp_draft_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ServingResolveMtpDecode(dispatch, decode_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52ServingPublishDecodeEvents(
        engine,
        dispatch,
        decode_result,
        finish_handles,
        &finish_handle_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    engine->stats.decode_dispatch_count += 1u;
    if (decode_result->lane_count >
        engine->stats.maximum_decode_active_sequence_count)
    {
        engine->stats.maximum_decode_active_sequence_count =
            decode_result->lane_count;
    }
    if (decode_result->lane_count > engine->stats.maximum_decode_lane_count)
    {
        engine->stats.maximum_decode_lane_count = decode_result->lane_count;
    }
    status = SparkGlm52RequestApiCompleteDispatch(engine->request_api, dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (mtp_draft_token_count != 0u)
    {
        status = SparkGlm52RequestApiArmMtpVerifyDispatch(
            engine->request_api,
            dispatch,
            mtp_draft_token_ids,
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT,
            mtp_draft_token_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
                return status;
        }
        if (status == SPARK_STATUS_OK)
        {
            engine->stats.mtp_draft_token_count +=
                (uint64_t)mtp_draft_token_count *
                (uint64_t)dispatch->request_count;
        }
    }
    if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
    {
        engine->stats.mtp_verify_dispatch_count += 1u;
    }
    status = SparkGlm52ServingCompleteFinishedHandles(
        engine,
        finish_handles,
        finish_handle_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52ServingCompleteBudgetFinishedRequests(engine, dispatch);
}

static SparkStatus SparkGlm52ServingInvokeDecode(
    SparkGlm52ServingEngine *engine,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52KvBlockTableView block_table_view;
    SparkGlm52RequestApiDecodeDispatchView decode_view;
    SparkGlm52ServingDecodeDispatch decode_dispatch;
    SparkGlm52ServingDecodeResult decode_result;
    SparkStatus status;

    status = SparkGlm52ServingBuildDecodeDispatch(
        engine,
        dispatch,
        &block_table_view,
        &decode_view,
        &decode_dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkGlm52ServingInitializeDecodeResult(
        &decode_result,
        dispatch->request_count,
        SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE);
    status = engine->decode_function(
        engine->callback_context,
        &decode_dispatch,
        &decode_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SparkGlm52ServingEngineCompleteDecodeDispatch(
        engine,
        dispatch,
        &decode_result);
}

static SparkStatus SparkGlm52ServingCompletePrefillDispatch(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    return SparkGlm52RequestApiCompleteDispatch(engine->request_api, dispatch);
}

SparkStatus SparkGlm52ServingEngineCompletePrefillDispatch(
    SparkGlm52ServingEngine *engine,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    if (engine == 0 || dispatch == 0 ||
        (dispatch->kind != SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL &&
         dispatch->kind != SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52ServingCompletePrefillDispatch(engine, dispatch);
}

SparkStatus SparkGlm52ServingEnginePump(
    SparkGlm52ServingEngine *engine,
    uint32_t pump_flags,
    uint32_t max_dispatch_steps,
    SparkGlm52ServingStats *stats)
{
    uint32_t accepted_pending_dispatch_count;
    uint32_t step_index;
    uint32_t step_limit;
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((pump_flags & ~SPARK_GLM52_SERVING_PUMP_KNOWN_FLAGS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    step_limit = max_dispatch_steps != 0u
        ? max_dispatch_steps
        : SPARK_GLM52_SERVING_DEFAULT_MAX_PUMP_STEPS;
    accepted_pending_dispatch_count = 0u;
    for (step_index = 0u; step_index < step_limit; ++step_index)
    {
        SparkGlm52RequestApiDispatch dispatch;
        uint32_t dispatch_was_completed_by_decode_path;
        uint32_t dispatch_was_retried;

        if (SparkGlm52ServingEventRingFreeCount(engine) <
            SparkGlm52ServingEventRingSafetyMargin())
        {
            status = SPARK_STATUS_BUSY;
            engine->stats.last_status = status;
            SparkGlm52ServingRefreshStats(engine);
            if (stats != 0)
            {
                *stats = engine->stats;
            }
            return status;
        }

        status = SparkGlm52RequestApiScheduleNext(
            engine->request_api,
            &dispatch);
        if (status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_BUSY)
        {
            if (accepted_pending_dispatch_count != 0u)
                status = SPARK_STATUS_PENDING;
            engine->stats.last_status = status;
            SparkGlm52ServingRefreshStats(engine);
            if (stats != 0)
            {
                *stats = engine->stats;
            }
            return status;
        }
        if (status != SPARK_STATUS_OK)
        {
            engine->stats.last_status = status;
            SparkGlm52ServingRefreshStats(engine);
            if (stats != 0)
            {
                *stats = engine->stats;
            }
            return status;
        }
        if (dispatch.accepted == 0u ||
            dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_NONE)
        {
            engine->stats.last_status = SPARK_STATUS_BUSY;
            SparkGlm52ServingRefreshStats(engine);
            if (stats != 0)
            {
                *stats = engine->stats;
            }
            return SPARK_STATUS_BUSY;
        }

        dispatch_was_completed_by_decode_path = 0u;
        dispatch_was_retried = 0u;
        if (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
            dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
        {
            status = SparkGlm52ServingInvokePrefill(
                engine,
                &dispatch,
                step_index);
            if (status == SPARK_STATUS_OK)
            {
                status = SparkGlm52ServingCompletePrefillDispatch(
                    engine,
                    &dispatch);
                if (status != SPARK_STATUS_OK)
                {
                    fprintf(stderr,"serving_prefill_complete_failed status=%u request=%llu\n",(uint32_t)status,(unsigned long long)dispatch.request_ids[0u]);
                }
            }
            else if (status != SPARK_STATUS_PENDING)
            {
                fprintf(stderr,"serving_prefill_invoke_failed status=%u request=%llu\n",(uint32_t)status,(unsigned long long)dispatch.request_ids[0u]);
            }
        }
        else if (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
            dispatch.kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
        {
            status = SparkGlm52ServingInvokeDecode(engine, &dispatch);
            if (status == SPARK_STATUS_OK)
            {
                dispatch_was_completed_by_decode_path = 1u;
            }
            else if (status == SPARK_STATUS_BUSY)
            {
                status = SparkGlm52RequestApiRetryDecodeDispatch(
                    engine->request_api,
                    &dispatch);
                if (status == SPARK_STATUS_OK)
                {
                    dispatch_was_retried = 1u;
                    status = SPARK_STATUS_BUSY;
                }
            }
            else if (status != SPARK_STATUS_PENDING)
            {
                fprintf(stderr,"serving_decode_invoke_failed status=%u request=%llu\n",(uint32_t)status,(unsigned long long)dispatch.request_ids[0u]);
            }
        }
        else
        {
            status = SPARK_STATUS_INVALID_ARGUMENT;
        }

        if (status != SPARK_STATUS_OK)
        {
            if (status == SPARK_STATUS_PENDING &&
                (dispatch.kind ==
                    SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
                 dispatch.kind ==
                    SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH ||
                 dispatch.kind ==
                    SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
                 dispatch.kind ==
                    SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH))
            {
                accepted_pending_dispatch_count += 1u;
                if ((pump_flags &
                        SPARK_GLM52_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH) != 0u)
                {
                    engine->stats.last_status = SPARK_STATUS_PENDING;
                    SparkGlm52ServingRefreshStats(engine);
                    if (stats != 0)
                    {
                        *stats = engine->stats;
                    }
                    return SPARK_STATUS_PENDING;
                }
                continue;
            }
            else if (status == SPARK_STATUS_BUSY &&
                (dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL ||
                 dispatch.kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH ||
                 dispatch_was_retried != 0u))
            {
                engine->stats.last_status = status;
                SparkGlm52ServingRefreshStats(engine);
                if (stats != 0)
                {
                    *stats = engine->stats;
                }
                return status;
            }
            if (dispatch_was_completed_by_decode_path == 0u &&
                dispatch_was_retried == 0u)
            {
                (void)SparkGlm52RequestApiCancelDispatch(
                    engine->request_api,
                    &dispatch);
                SparkGlm52ServingFailDispatchRequests(
                    engine,
                    &dispatch,
                    status);
            }
            engine->stats.last_status = status;
            SparkGlm52ServingRefreshStats(engine);
            if (stats != 0)
            {
                *stats = engine->stats;
            }
            return status;
        }

        if ((pump_flags & SPARK_GLM52_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH) != 0u)
        {
            engine->stats.last_status = SPARK_STATUS_OK;
            SparkGlm52ServingRefreshStats(engine);
            if (stats != 0)
            {
                *stats = engine->stats;
            }
            return SPARK_STATUS_OK;
        }
    }

    status = accepted_pending_dispatch_count != 0u
        ? SPARK_STATUS_PENDING
        : SPARK_STATUS_OK;
    engine->stats.last_status = status;
    SparkGlm52ServingRefreshStats(engine);
    if (stats != 0)
    {
        *stats = engine->stats;
    }
    return status;
}

SparkStatus SparkGlm52ServingEnginePopEvent(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingEvent *event_out)
{
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK || event_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (engine->event_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    *event_out = engine->event_ring[engine->event_read_index];
    SparkGlm52ServingInitializeEvent(&engine->event_ring[engine->event_read_index]);
    engine->event_read_index += 1u;
    if (engine->event_read_index == engine->event_ring_capacity)
    {
        engine->event_read_index = 0u;
    }
    engine->event_count -= 1u;
    SparkGlm52ServingRefreshStats(engine);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServingEngineGetStats(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingStats *stats_out)
{
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK || stats_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    SparkGlm52ServingRefreshStats(engine);
    *stats_out = engine->stats;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ServingEngineCancelRequest(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestHandle request_handle)
{
    SparkGlm52ServingRequestRecord *record;
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    record = SparkGlm52ServingFindRecordByHandle(engine, request_handle);
    if (record == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkGlm52RequestApiCancelRequest(engine->request_api, request_handle);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_CANCELLED;
    status = SparkGlm52ServingPushSimpleEvent(
        engine,
        SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_CANCELLED,
        SPARK_STATUS_OK,
        record);
    SparkGlm52ServingRefreshStats(engine);
    return status;
}

SparkStatus SparkGlm52ServingEngineFailRequestByRequestId(
    SparkGlm52ServingEngine *engine,
    uint64_t request_id,
    SparkStatus failure_status)
{
    SparkGlm52ServingRequestRecord *record;
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (request_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    record = SparkGlm52ServingFindRecordByRequestId(engine,request_id);
    if (record == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (record->state == SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_COMPLETED ||
        record->state == SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_CANCELLED)
    {
        return SPARK_STATUS_OK;
    }
    (void)SparkGlm52RequestApiCancelRequest(
        engine->request_api,
        record->request_handle);
    record->state = SPARK_GLM52_SERVING_REQUEST_RECORD_STATE_CANCELLED;
    status = SparkGlm52ServingPushSimpleEvent(
        engine,
        SPARK_GLM52_SERVING_EVENT_KIND_REQUEST_CANCELLED,
        (uint32_t)failure_status,
        record);
    SparkGlm52ServingRefreshStats(engine);
    return status;
}

SparkStatus SparkGlm52ServingEngineReleaseCompletedRequest(
    SparkGlm52ServingEngine *engine,
    SparkGlm52ServingRequestHandle request_handle)
{
    SparkGlm52ServingRequestRecord *record;
    uint32_t record_index;
    uint32_t *token_ids;
    uint32_t token_capacity;
    SparkStatus status;

    status = SparkGlm52ServingValidateEngine(engine);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    record = SparkGlm52ServingFindRecordByHandle(engine, request_handle);
    if (record == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (engine->release_sequence_function != 0)
    {
        uint64_t token_count;
        token_count = (uint64_t)record->prompt_token_count +
            record->streamed_decode_token_count;
        if (token_count == 0u || token_count > UINT32_MAX)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = engine->release_sequence_function(
            engine->callback_context,record->request_id,record->sequence_id,
            (uint32_t)token_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    status = SparkGlm52RequestApiReleaseCompletedRequest(
        engine->request_api,
        request_handle);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
    {
        return status;
    }
    token_ids = record->token_ids;
    token_capacity = record->token_capacity;
    record_index = SparkGlm52ServingRecordIndex(engine, record);
    if (record_index == SPARK_GLM52_SERVING_NO_RECORD_SLOT)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if ((engine->flags &
            SPARK_GLM52_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0u)
    {
        free(token_ids);
        token_ids = 0;
        token_capacity = 0u;
    }
    SparkGlm52ServingRemoveRecordHash(engine, record);
    SparkGlm52ServingInitializeRequestRecord(
        record,
        token_ids,
        token_capacity);
    record->free_record_next = engine->free_record_head;
    engine->free_record_head = record_index;
    SparkGlm52ServingRefreshStats(engine);
    return SPARK_STATUS_OK;
}
