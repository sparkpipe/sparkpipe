#include "sparkpipe/spark_glm52_request_api.h"

#include <string.h>

static uint32_t SparkGlm52RequestApiNormalizeConfigurationFlags(
    uint32_t configuration_flags)
{
    if (configuration_flags == 0u)
    {
        return SPARK_GLM52_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS;
    }
    return configuration_flags;
}

static uint32_t SparkGlm52RequestApiNormalizePrefetchLookaheadRequestCount(
    uint32_t prefetch_lookahead_request_count,
    uint32_t request_capacity)
{
    if (prefetch_lookahead_request_count == 0u)
    {
        prefetch_lookahead_request_count =
            SPARK_GLM52_REQUEST_API_DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT;
    }
    if (prefetch_lookahead_request_count > request_capacity)
    {
        return request_capacity;
    }
    return prefetch_lookahead_request_count;
}

static uint32_t SparkGlm52RequestApiNormalizePrefetchLaneCount(
    uint32_t prefetch_lane_count)
{
    if (prefetch_lane_count == 0u)
    {
        return SPARK_GLM52_REQUEST_API_DEFAULT_PREFETCH_LANE_COUNT;
    }
    return prefetch_lane_count;
}

static uint32_t SparkGlm52RequestApiNormalizeDecodeBatchTarget(
    uint32_t decode_batch_target)
{
    if (decode_batch_target == 0u)
    {
        return SPARK_GLM52_REQUEST_API_DEFAULT_DECODE_BATCH_TARGET;
    }
    if (decode_batch_target > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        return SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    return decode_batch_target;
}

static uint32_t SparkGlm52RequestApiNormalizeMaxResidentKvBlockCount(
    const SparkGlm52RequestApiConfiguration *configuration)
{
    uint32_t physical_block_count;

    if (configuration == 0 ||
        configuration->scheduler == 0 ||
        configuration->scheduler->prefix_cache == 0 ||
        configuration->scheduler->prefix_cache->kv_cache_arena == 0)
    {
        return 0u;
    }

    physical_block_count =
        configuration->scheduler->prefix_cache->kv_cache_arena->physical_block_count;
    if (configuration->max_resident_kv_block_count == 0u ||
        configuration->max_resident_kv_block_count >= physical_block_count)
    {
        return 0u;
    }
    return configuration->max_resident_kv_block_count;
}

static uint32_t SparkGlm52RequestApiNormalizePriority(
    const SparkGlm52RequestApiSubmitRequest *request)
{
    if ((request->flags & SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME) != 0u)
    {
        return SPARK_GLM52_REQUEST_API_REALTIME_PRIORITY;
    }
    if (request->priority == 0u)
    {
        return SPARK_GLM52_REQUEST_API_DEFAULT_PRIORITY;
    }
    return request->priority;
}

static uint32_t SparkGlm52RequestApiConfigurationFlagsAreValid(
    uint32_t configuration_flags)
{
    return (configuration_flags &
        ~SPARK_GLM52_REQUEST_API_CONFIGURATION_KNOWN_FLAGS) == 0u;
}

static uint32_t SparkGlm52RequestApiJitPrefetchIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) != 0u;
}

static uint32_t SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH) != 0u;
}

static uint32_t SparkGlm52RequestApiQueueAwarePrefixCacheEvictionIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION) != 0u;
}

static uint32_t SparkGlm52RequestApiDecodeBatchingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING) != 0u;
}

static uint32_t SparkGlm52RequestApiPrefixCohortingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING) != 0u;
}

static uint32_t SparkGlm52RequestApiPrefillBatchingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING) != 0u;
}

static SparkStatus SparkGlm52RequestApiValidate(
    const SparkGlm52RequestApi *api)
{
    if (api == 0 ||
        api->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        api->descriptor_bytes != SPARK_GLM52_REQUEST_API_DESCRIPTOR_BYTES ||
        api->request_capacity == 0u ||
        api->scheduler == 0 ||
        api->request_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiValidateScheduler(
    SparkGlm52Scheduler *scheduler)
{
    if (scheduler == 0 ||
        scheduler->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
        scheduler->descriptor_bytes != SPARK_GLM52_SCHEDULER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiValidateConfiguration(
    const SparkGlm52RequestApiConfiguration *configuration)
{
    uint32_t configuration_flags;
    uint32_t prefetch_lane_count;
    SparkStatus status;

    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->request_capacity == 0u ||
        configuration->request_slots == 0 ||
        configuration->reserved != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    configuration_flags = SparkGlm52RequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    prefetch_lane_count = SparkGlm52RequestApiNormalizePrefetchLaneCount(
        configuration->prefetch_lane_count);
    if (!SparkGlm52RequestApiConfigurationFlagsAreValid(configuration_flags) ||
        prefetch_lane_count == 0u ||
        prefetch_lane_count > SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52RequestApiValidateScheduler(configuration->scheduler);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if ((configuration_flags &
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH) != 0u &&
        (configuration_flags &
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration_flags &
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION) != 0u &&
        configuration->scheduler->prefix_cache == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration_flags &
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH) != 0u)
    {
        if (configuration->scheduler->prefix_cache == 0 ||
            configuration->scheduler->prefix_cache->kv_cache_arena == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if ((configuration_flags &
                SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH) != 0u)
        {
            if (configuration->kv_prefetch_start_function == 0 ||
                configuration->kv_prefetch_poll_function == 0)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else if (configuration->kv_prefetch_function == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52RequestApiInitializeSlot(
    SparkGlm52RequestApiSlot *slot)
{
    memset(slot, 0, sizeof(*slot));
    slot->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    slot->descriptor_bytes = SPARK_GLM52_REQUEST_API_SLOT_DESCRIPTOR_BYTES;
    slot->state = SPARK_GLM52_REQUEST_API_STATE_FREE;
}


SparkStatus SparkGlm52RequestApiConfigurationUseAsyncKvCachePrefetchBackend(
    SparkGlm52RequestApiConfiguration *configuration,
    SparkGlm52KvCacheAsyncPrefetchBackend *backend)
{
    uint32_t configuration_flags;

    if (configuration == 0 || backend == 0 ||
        backend->abi_version != SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        backend->descriptor_bytes != SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    configuration_flags = SparkGlm52RequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    configuration_flags |=
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH |
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH;
    configuration->configuration_flags = configuration_flags;
    configuration->kv_prefetch_context = backend;
    configuration->kv_prefetch_function =
        SparkGlm52KvCacheAsyncPrefetchBackendSubmitSynchronous;
    configuration->kv_prefetch_start_function =
        SparkGlm52KvCacheAsyncPrefetchBackendStart;
    configuration->kv_prefetch_poll_function =
        SparkGlm52KvCacheAsyncPrefetchBackendPoll;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiInitialize(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiConfiguration *configuration)
{
    uint32_t slot_index;
    uint32_t configuration_flags;
    SparkStatus status;

    if (api == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52RequestApiValidateConfiguration(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    configuration_flags = SparkGlm52RequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    memset(api, 0, sizeof(*api));
    api->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    api->descriptor_bytes = SPARK_GLM52_REQUEST_API_DESCRIPTOR_BYTES;
    api->configuration_flags = configuration_flags;
    api->request_capacity = configuration->request_capacity;
    api->prefetch_lookahead_request_count =
        SparkGlm52RequestApiNormalizePrefetchLookaheadRequestCount(
            configuration->prefetch_lookahead_request_count,
            configuration->request_capacity);
    api->prefetch_lane_count = SparkGlm52RequestApiNormalizePrefetchLaneCount(
        configuration->prefetch_lane_count);
    api->decode_batch_target = SparkGlm52RequestApiNormalizeDecodeBatchTarget(
        configuration->decode_batch_target);
    api->max_resident_kv_block_count =
        SparkGlm52RequestApiNormalizeMaxResidentKvBlockCount(configuration);
    api->next_handle = 1u;
    api->next_sequence_id = 1u;
    api->next_prefetch_id = 1u;
    api->scheduler = configuration->scheduler;
    api->request_slots = configuration->request_slots;
    api->kv_prefetch_function = configuration->kv_prefetch_function;
    api->kv_prefetch_context = configuration->kv_prefetch_context;
    api->kv_prefetch_start_function =
        configuration->kv_prefetch_start_function;
    api->kv_prefetch_poll_function =
        configuration->kv_prefetch_poll_function;

    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiInitializeSlot(&api->request_slots[slot_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindFreeSlot(
    SparkGlm52RequestApi *api)
{
    uint32_t slot_index;

    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        if (api->request_slots[slot_index].state ==
            SPARK_GLM52_REQUEST_API_STATE_FREE)
        {
            return &api->request_slots[slot_index];
        }
    }
    return 0;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindSlotByHandle(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle)
{
    uint32_t slot_index;

    if (handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return 0;
    }
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (slot->state != SPARK_GLM52_REQUEST_API_STATE_FREE &&
            slot->handle == handle)
        {
            return slot;
        }
    }
    return 0;
}

static SparkStatus SparkGlm52RequestApiValidateSubmitRequest(
    const SparkGlm52RequestApiSubmitRequest *request)
{
    if (request == 0 ||
        request->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        request->descriptor_bytes != SPARK_GLM52_REQUEST_API_SUBMIT_DESCRIPTOR_BYTES ||
        (request->flags & ~SPARK_GLM52_REQUEST_API_REQUEST_FLAG_KNOWN_FLAGS) != 0u ||
        request->prompt_token_count == 0u ||
        request->prompt_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiSubmit(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSubmitRequest *request,
    SparkGlm52RequestApiHandle *handle_out)
{
    SparkGlm52RequestApiSlot *slot;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || handle_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    status = SparkGlm52RequestApiValidateSubmitRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkGlm52RequestApiFindFreeSlot(api);
    if (slot == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    SparkGlm52RequestApiInitializeSlot(slot);
    slot->state = SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
    slot->flags = request->flags;
    slot->priority = SparkGlm52RequestApiNormalizePriority(request);
    slot->prompt_token_count = request->prompt_token_count;
    slot->thinking_token_budget = request->thinking_token_budget;
    slot->output_token_budget = request->output_token_budget;
    slot->remaining_thinking_token_budget = request->thinking_token_budget;
    slot->remaining_output_token_budget = request->output_token_budget;
    slot->max_prefill_tokens_per_step = request->max_prefill_tokens_per_step;
    slot->request_id = request->request_id;
    if (request->sequence_id != 0u)
    {
        slot->sequence_id = request->sequence_id;
    }
    else
    {
        slot->sequence_id = api->next_sequence_id;
        api->next_sequence_id += 1u;
    }
    slot->handle = api->next_handle;
    api->next_handle += 1u;
    slot->submission_order = api->submission_counter;
    api->submission_counter += 1u;
    slot->prompt_token_ids = request->prompt_token_ids;

    api->queued_request_count += 1u;
    api->submitted_request_count += 1u;
    *handle_out = slot->handle;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiSlotIsSchedulablePrefill(
    const SparkGlm52RequestApiSlot *slot)
{
    return slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
}

static uint32_t SparkGlm52RequestApiSlotIsSchedulableDecode(
    const SparkGlm52RequestApiSlot *slot)
{
    return slot->state == SPARK_GLM52_REQUEST_API_STATE_READY_DECODE &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u);
}

static uint32_t SparkGlm52RequestApiMinimumU32(
    uint32_t left,
    uint32_t right);

static uint32_t SparkGlm52RequestApiRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple);

static uint32_t SparkGlm52RequestApiPrefillCachedBlocksAreResident(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint32_t prompt_token_count);

static uint32_t SparkGlm52RequestApiNextPrefillStepTokenCount(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint32_t *computed_prompt_token_count_out);

static uint32_t SparkGlm52RequestApiPrefillBlockCountForScheduledTokens(
    const SparkGlm52RequestApi *api,
    uint32_t scheduled_prompt_token_count);

static uint32_t SparkGlm52RequestApiSlotIsCompatiblePrefillBatchMember(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *leader_slot,
    const SparkGlm52RequestApiSlot *candidate_slot,
    uint32_t leader_prefill_block_count,
    uint32_t require_resident_cached_blocks,
    uint32_t *candidate_scheduled_prompt_token_count_out);

static uint32_t SparkGlm52RequestApiDecodeBlocksAreResident(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot);

static uint32_t SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
    const SparkGlm52RequestApiSlot *candidate,
    const SparkGlm52RequestApiSlot *current)
{
    if (current == 0)
    {
        return 1u;
    }
    if (candidate->priority != current->priority)
    {
        return candidate->priority > current->priority;
    }
    return candidate->submission_order < current->submission_order;
}

typedef struct SparkGlm52RequestApiPrefillBatchShape
{
    SparkGlm52RequestApiSlot *slot;
    uint32_t scheduled_prompt_token_count;
    uint32_t compatible_request_count;
    uint32_t bucket_capacity;
    uint32_t graph_padding_count;
    uint32_t resident_cached_blocks;
    uint32_t realtime_priority;
    uint32_t prefix_family_shared_token_count;
    uint32_t prefix_family_request_count;
    uint64_t prefix_family_saved_prompt_token_count;
} SparkGlm52RequestApiPrefillBatchShape;

#define SPARK_GLM52_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY 256u

typedef struct SparkGlm52RequestApiPrefixFamilyGroup
{
    uint32_t valid;
    uint32_t shared_prefix_token_count;
    uint32_t request_count;
    uint32_t capped_request_count;
    uint32_t highest_priority;
    uint32_t realtime_priority;
    uint64_t prefix_hash;
    SparkGlm52RequestApiSlot *leader_slot;
    uint64_t earliest_submission_order;
} SparkGlm52RequestApiPrefixFamilyGroup;

typedef struct SparkGlm52RequestApiPrefixFamilyChoice
{
    SparkGlm52RequestApiSlot *leader_slot;
    uint32_t shared_prefix_token_count;
    uint32_t request_count;
    uint32_t realtime_priority;
    uint32_t highest_priority;
    uint64_t saved_prompt_token_count;
} SparkGlm52RequestApiPrefixFamilyChoice;

static uint32_t SparkGlm52RequestApiBatchBucketCapacityForSequenceCount(
    uint32_t active_sequence_count)
{
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B16)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B32)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B32;
    }
    if (active_sequence_count <= SPARK_GLM52_STAGE_PLAN_BUCKET_B64)
    {
        return SPARK_GLM52_STAGE_PLAN_BUCKET_B64;
    }
    return 0u;
}

static uint32_t SparkGlm52RequestApiSlotHasRealtimePriority(
    const SparkGlm52RequestApiSlot *slot)
{
    return slot != 0 &&
        ((slot->flags & SPARK_GLM52_REQUEST_API_REQUEST_FLAG_REALTIME) != 0u ||
         slot->priority >= SPARK_GLM52_REQUEST_API_REALTIME_PRIORITY);
}

static uint64_t SparkGlm52RequestApiPrefixFamilySavedTokenCount(
    uint32_t shared_prefix_token_count,
    uint32_t request_count)
{
    if (shared_prefix_token_count == 0u || request_count < 2u)
    {
        return 0u;
    }
    return (uint64_t)shared_prefix_token_count *
        (uint64_t)(request_count - 1u);
}

static uint32_t SparkGlm52RequestApiPrefixFamilyLeaderIsBetter(
    const SparkGlm52RequestApiSlot *candidate,
    const SparkGlm52RequestApiSlot *current)
{
    if (current == 0)
    {
        return 1u;
    }
    if (candidate->priority != current->priority)
    {
        return candidate->priority > current->priority;
    }
    return candidate->submission_order < current->submission_order;
}

static void SparkGlm52RequestApiInitializePrefixFamilyChoice(
    SparkGlm52RequestApiPrefixFamilyChoice *choice)
{
    memset(choice, 0, sizeof(*choice));
}

static uint32_t SparkGlm52RequestApiPrefixFamilyGroupIsBetter(
    const SparkGlm52RequestApiPrefixFamilyGroup *candidate,
    const SparkGlm52RequestApiPrefixFamilyGroup *current)
{
    uint64_t candidate_saved_token_count;
    uint64_t current_saved_token_count;

    if (candidate == 0 || candidate->valid == 0u ||
        candidate->capped_request_count < 2u)
    {
        return 0u;
    }
    if (current == 0 || current->valid == 0u ||
        current->capped_request_count < 2u)
    {
        return 1u;
    }

    if (candidate->realtime_priority != current->realtime_priority)
    {
        return candidate->realtime_priority > current->realtime_priority;
    }

    candidate_saved_token_count = SparkGlm52RequestApiPrefixFamilySavedTokenCount(
        candidate->shared_prefix_token_count,
        candidate->capped_request_count);
    current_saved_token_count = SparkGlm52RequestApiPrefixFamilySavedTokenCount(
        current->shared_prefix_token_count,
        current->capped_request_count);
    if (candidate_saved_token_count != current_saved_token_count)
    {
        return candidate_saved_token_count > current_saved_token_count;
    }
    if (candidate->capped_request_count != current->capped_request_count)
    {
        return candidate->capped_request_count > current->capped_request_count;
    }
    if (candidate->shared_prefix_token_count != current->shared_prefix_token_count)
    {
        return candidate->shared_prefix_token_count >
            current->shared_prefix_token_count;
    }
    if (candidate->highest_priority != current->highest_priority)
    {
        return candidate->highest_priority > current->highest_priority;
    }
    return candidate->earliest_submission_order <
        current->earliest_submission_order;
}

static SparkGlm52RequestApiPrefixFamilyGroup *
SparkGlm52RequestApiFindPrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *groups,
    uint32_t group_count,
    uint64_t prefix_hash,
    uint32_t shared_prefix_token_count)
{
    uint32_t group_index;

    for (group_index = 0u; group_index < group_count; ++group_index)
    {
        if (groups[group_index].valid != 0u &&
            groups[group_index].prefix_hash == prefix_hash &&
            groups[group_index].shared_prefix_token_count ==
                shared_prefix_token_count)
        {
            return &groups[group_index];
        }
    }
    return 0;
}

static SparkGlm52RequestApiPrefixFamilyGroup *
SparkGlm52RequestApiAcquirePrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *groups,
    uint32_t *group_count,
    uint64_t prefix_hash,
    uint32_t shared_prefix_token_count)
{
    SparkGlm52RequestApiPrefixFamilyGroup *group;

    group = SparkGlm52RequestApiFindPrefixFamilyGroup(
        groups,
        *group_count,
        prefix_hash,
        shared_prefix_token_count);
    if (group != 0)
    {
        return group;
    }
    if (*group_count >= SPARK_GLM52_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY)
    {
        return 0;
    }

    group = &groups[*group_count];
    memset(group, 0, sizeof(*group));
    group->valid = 1u;
    group->prefix_hash = prefix_hash;
    group->shared_prefix_token_count = shared_prefix_token_count;
    group->earliest_submission_order = UINT64_MAX;
    *group_count += 1u;
    return group;
}

static void SparkGlm52RequestApiAddSlotToPrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *group,
    SparkGlm52RequestApiSlot *slot)
{
    if (group == 0 || slot == 0)
    {
        return;
    }

    group->request_count += 1u;
    if (group->capped_request_count <
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        group->capped_request_count += 1u;
    }
    if (slot->priority > group->highest_priority)
    {
        group->highest_priority = slot->priority;
    }
    if (SparkGlm52RequestApiSlotHasRealtimePriority(slot))
    {
        group->realtime_priority = 1u;
    }
    if (slot->submission_order < group->earliest_submission_order)
    {
        group->earliest_submission_order = slot->submission_order;
    }
    if (SparkGlm52RequestApiPrefixFamilyLeaderIsBetter(
            slot,
            group->leader_slot))
    {
        group->leader_slot = slot;
    }
}

static uint32_t SparkGlm52RequestApiBuildBestPrefixFamilyChoice(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiPrefixFamilyChoice *choice)
{
    SparkGlm52RequestApiPrefixFamilyGroup groups[
        SPARK_GLM52_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY];
    SparkGlm52RequestApiPrefixFamilyGroup *best_group;
    uint32_t group_count;
    uint32_t slot_index;
    uint32_t block_token_count;

    SparkGlm52RequestApiInitializePrefixFamilyChoice(choice);
    if (api == 0 ||
        !SparkGlm52RequestApiPrefixCohortingIsEnabled(api) ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        api->scheduler->prefix_cache->block_token_count == 0u)
    {
        return 0u;
    }

    memset(groups, 0, sizeof(groups));
    group_count = 0u;
    best_group = 0;
    block_token_count = api->scheduler->prefix_cache->block_token_count;

    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t reusable_prefix_token_count;
        uint32_t scheduled_prompt_token_count;
        uint32_t maximum_family_prefix_token_count;
        uint32_t prefix_token_count;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiSlotIsSchedulablePrefill(slot) ||
            slot->prompt_token_ids == 0)
        {
            continue;
        }

        scheduled_prompt_token_count = SparkGlm52RequestApiNextPrefillStepTokenCount(
            api,
            slot,
            &reusable_prefix_token_count);
        if (scheduled_prompt_token_count == 0u)
        {
            continue;
        }

        maximum_family_prefix_token_count = reusable_prefix_token_count +
            scheduled_prompt_token_count;
        maximum_family_prefix_token_count = SparkGlm52RequestApiRoundDownToMultiple(
            SparkGlm52RequestApiMinimumU32(
                maximum_family_prefix_token_count,
                slot->prompt_token_count),
            block_token_count);
        prefix_token_count = SparkGlm52RequestApiRoundDownToMultiple(
            reusable_prefix_token_count + block_token_count,
            block_token_count);
        if (prefix_token_count <= reusable_prefix_token_count)
        {
            prefix_token_count += block_token_count;
        }

        while (prefix_token_count <= maximum_family_prefix_token_count)
        {
            SparkGlm52PrefixCachePromptHash prefix_hash;
            SparkGlm52RequestApiPrefixFamilyGroup *group;

            if (SparkGlm52PrefixCacheHashPromptTokens(
                    block_token_count,
                    SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH,
                    slot->prompt_token_ids,
                    prefix_token_count,
                    &prefix_hash) != SPARK_STATUS_OK)
            {
                break;
            }
            group = SparkGlm52RequestApiAcquirePrefixFamilyGroup(
                groups,
                &group_count,
                prefix_hash.prompt_hash,
                prefix_token_count);
            SparkGlm52RequestApiAddSlotToPrefixFamilyGroup(group, slot);
            prefix_token_count += block_token_count;
        }
    }

    for (slot_index = 0u; slot_index < group_count; ++slot_index)
    {
        if (SparkGlm52RequestApiPrefixFamilyGroupIsBetter(
                &groups[slot_index],
                best_group))
        {
            best_group = &groups[slot_index];
        }
    }

    if (best_group == 0 || best_group->capped_request_count < 2u)
    {
        return 0u;
    }

    choice->leader_slot = best_group->leader_slot;
    choice->shared_prefix_token_count = best_group->shared_prefix_token_count;
    choice->request_count = best_group->capped_request_count;
    choice->realtime_priority = best_group->realtime_priority;
    choice->highest_priority = best_group->highest_priority;
    choice->saved_prompt_token_count =
        SparkGlm52RequestApiPrefixFamilySavedTokenCount(
            best_group->shared_prefix_token_count,
            best_group->capped_request_count);
    return choice->leader_slot != 0 && choice->saved_prompt_token_count != 0u;
}

static uint32_t SparkGlm52RequestApiPrefixFamilyChoiceBeatsPrefillSlot(
    const SparkGlm52RequestApiPrefixFamilyChoice *choice,
    const SparkGlm52RequestApiSlot *slot)
{
    if (choice == 0 || choice->leader_slot == 0 ||
        choice->saved_prompt_token_count == 0u)
    {
        return 0u;
    }
    if (slot == 0)
    {
        return 1u;
    }
    if (choice->realtime_priority !=
        SparkGlm52RequestApiSlotHasRealtimePriority(slot))
    {
        return choice->realtime_priority != 0u;
    }
    if (choice->leader_slot->priority != slot->priority)
    {
        return choice->leader_slot->priority > slot->priority;
    }
    return choice->saved_prompt_token_count != 0u;
}

static uint32_t SparkGlm52RequestApiEvaluatePrefillBatchShape(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    SparkGlm52RequestApiPrefillBatchShape *shape)
{
    uint32_t slot_index;
    uint32_t batch_target;
    uint32_t leader_prefill_block_count;

    memset(shape, 0, sizeof(*shape));
    if (!SparkGlm52RequestApiSlotIsSchedulablePrefill(slot))
    {
        return 0u;
    }

    shape->slot = slot;
    shape->scheduled_prompt_token_count =
        SparkGlm52RequestApiNextPrefillStepTokenCount(api, slot, 0);
    if (shape->scheduled_prompt_token_count == 0u)
    {
        return 0u;
    }

    shape->resident_cached_blocks = SparkGlm52RequestApiPrefillCachedBlocksAreResident(
        api,
        slot,
        slot->prompt_token_count);
    leader_prefill_block_count =
        SparkGlm52RequestApiPrefillBlockCountForScheduledTokens(
            api,
            shape->scheduled_prompt_token_count);
    if (leader_prefill_block_count == 0u)
    {
        return 0u;
    }
    shape->realtime_priority = SparkGlm52RequestApiSlotHasRealtimePriority(slot);
    shape->compatible_request_count = 1u;

    if (SparkGlm52RequestApiPrefillBatchingIsEnabled(api))
    {
        batch_target = api->decode_batch_target;
        if (batch_target > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
        {
            batch_target = SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
        }
        for (slot_index = 0u;
             slot_index < api->request_capacity &&
                 shape->compatible_request_count < batch_target;
             ++slot_index)
        {
            SparkGlm52RequestApiSlot *candidate;

            candidate = &api->request_slots[slot_index];
            if (SparkGlm52RequestApiSlotIsCompatiblePrefillBatchMember(
                    api,
                    slot,
                    candidate,
                    leader_prefill_block_count,
                    1u,
                    0))
            {
                shape->compatible_request_count += 1u;
            }
        }
    }

    shape->bucket_capacity = SparkGlm52RequestApiBatchBucketCapacityForSequenceCount(
        shape->compatible_request_count);
    if (shape->bucket_capacity == 0u)
    {
        return 0u;
    }
    shape->graph_padding_count =
        shape->bucket_capacity - shape->compatible_request_count;
    return 1u;
}

static uint32_t SparkGlm52RequestApiPrefillShapeIsBetter(
    const SparkGlm52RequestApiPrefillBatchShape *candidate,
    const SparkGlm52RequestApiPrefillBatchShape *current)
{
    if (current->slot == 0)
    {
        return 1u;
    }

    if (candidate->realtime_priority != current->realtime_priority)
    {
        return candidate->realtime_priority > current->realtime_priority;
    }

    if (candidate->realtime_priority != 0u)
    {
        return SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
            candidate->slot,
            current->slot);
    }

    if (candidate->resident_cached_blocks != current->resident_cached_blocks)
    {
        return candidate->resident_cached_blocks > current->resident_cached_blocks;
    }

    if (candidate->compatible_request_count != current->compatible_request_count)
    {
        return candidate->compatible_request_count > current->compatible_request_count;
    }

    if (candidate->graph_padding_count != current->graph_padding_count)
    {
        return candidate->graph_padding_count < current->graph_padding_count;
    }

    if (candidate->scheduled_prompt_token_count !=
        current->scheduled_prompt_token_count)
    {
        return candidate->scheduled_prompt_token_count >
            current->scheduled_prompt_token_count;
    }

    return SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
        candidate->slot,
        current->slot);
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindBestPrefillSlot(
    SparkGlm52RequestApi *api)
{
    SparkGlm52RequestApiPrefillBatchShape best_shape;
    uint32_t slot_index;

    memset(&best_shape, 0, sizeof(best_shape));
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiPrefillBatchShape candidate_shape;
        SparkGlm52RequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiEvaluatePrefillBatchShape(
                api,
                slot,
                &candidate_shape))
        {
            continue;
        }
        if (SparkGlm52RequestApiPrefillShapeIsBetter(
                &candidate_shape,
                &best_shape))
        {
            best_shape = candidate_shape;
        }
    }
    return best_shape.slot;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindBestDecodeSlot(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle *excluded_handles,
    uint32_t excluded_handle_count,
    uint32_t require_resident_kv)
{
    SparkGlm52RequestApiSlot *best_slot;
    uint32_t slot_index;

    best_slot = 0;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t excluded_index;
        uint32_t is_excluded;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiSlotIsSchedulableDecode(slot) ||
            (require_resident_kv != 0u &&
             !SparkGlm52RequestApiDecodeBlocksAreResident(api, slot)))
        {
            continue;
        }
        is_excluded = 0u;
        for (excluded_index = 0u;
             excluded_index < excluded_handle_count;
             ++excluded_index)
        {
            if (excluded_handles[excluded_index] == slot->handle)
            {
                is_excluded = 1u;
                break;
            }
        }
        if (is_excluded != 0u)
        {
            continue;
        }
        if (SparkGlm52RequestApiSlotHasHigherSchedulingPriority(slot, best_slot))
        {
            best_slot = slot;
        }
    }
    return best_slot;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindBestDecodeBatchMember(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *leader_slot,
    SparkGlm52RequestApiHandle *excluded_handles,
    uint32_t excluded_handle_count,
    uint32_t require_resident_kv)
{
    SparkGlm52RequestApiSlot *best_slot;
    uint32_t slot_index;

    best_slot = 0;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t excluded_index;
        uint32_t is_excluded;
        uint32_t slot_blocks_are_resident;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiSlotIsSchedulableDecode(slot))
        {
            continue;
        }
        slot_blocks_are_resident = SparkGlm52RequestApiDecodeBlocksAreResident(
            api,
            slot);
        if ((require_resident_kv != 0u || slot->priority < leader_slot->priority) &&
            slot_blocks_are_resident == 0u)
        {
            continue;
        }
        is_excluded = 0u;
        for (excluded_index = 0u;
             excluded_index < excluded_handle_count;
             ++excluded_index)
        {
            if (excluded_handles[excluded_index] == slot->handle)
            {
                is_excluded = 1u;
                break;
            }
        }
        if (is_excluded != 0u)
        {
            continue;
        }
        if (SparkGlm52RequestApiSlotHasHigherSchedulingPriority(slot, best_slot))
        {
            best_slot = slot;
        }
    }
    return best_slot;
}

static uint32_t SparkGlm52RequestApiMinimumU32(
    uint32_t left,
    uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52RequestApiMaximumU32(
    uint32_t left,
    uint32_t right)
{
    return left > right ? left : right;
}

static uint32_t SparkGlm52RequestApiRoundDownToMultiple(
    uint32_t value,
    uint32_t multiple)
{
    if (multiple == 0u)
    {
        return 0u;
    }
    return value - (value % multiple);
}

static uint32_t SparkGlm52RequestApiCountCommonPrefixTokens(
    const SparkGlm52RequestApiSlot *left,
    const SparkGlm52RequestApiSlot *right)
{
    uint32_t shared_token_count;
    uint32_t token_index;

    if (left == 0 || right == 0 ||
        left->prompt_token_ids == 0 || right->prompt_token_ids == 0)
    {
        return 0u;
    }
    shared_token_count = SparkGlm52RequestApiMinimumU32(
        left->prompt_token_count,
        right->prompt_token_count);
    for (token_index = 0u;
         token_index < shared_token_count;
         ++token_index)
    {
        if (left->prompt_token_ids[token_index] !=
            right->prompt_token_ids[token_index])
        {
            return token_index;
        }
    }
    return shared_token_count;
}

static uint32_t SparkGlm52RequestApiSharedCachePrefixTokenCount(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *left,
    const SparkGlm52RequestApiSlot *right)
{
    uint32_t block_token_count;
    uint32_t common_prefix_token_count;

    if (api == 0 || api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0)
    {
        return 0u;
    }
    block_token_count = api->scheduler->prefix_cache->block_token_count;
    common_prefix_token_count = SparkGlm52RequestApiCountCommonPrefixTokens(
        left,
        right);
    common_prefix_token_count = SparkGlm52RequestApiRoundDownToMultiple(
        common_prefix_token_count,
        block_token_count);
    if (common_prefix_token_count <= SparkGlm52RequestApiMaximumU32(
            left->computed_prompt_token_count,
            right->computed_prompt_token_count))
    {
        return 0u;
    }
    return common_prefix_token_count;
}

static uint32_t SparkGlm52RequestApiPrefillCachedBlocksAreResident(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint32_t prompt_token_count)
{
    uint32_t matched_token_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api) ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        prompt_token_count == 0u)
    {
        return 1u;
    }

    status = SparkGlm52PrefixCacheProbeReusablePrefixResidency(
        api->scheduler->prefix_cache,
        slot->prompt_token_ids,
        prompt_token_count,
        &matched_token_count,
        &resident_block_count,
        &nonresident_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    (void)matched_token_count;
    (void)resident_block_count;
    return nonresident_block_count == 0u;
}

static uint32_t SparkGlm52RequestApiDecodeBlocksAreResident(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    uint32_t physical_block_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api) ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        slot->computed_prompt_token_count == 0u)
    {
        return 1u;
    }

    status = SparkGlm52PrefixCacheProbeSequenceResidency(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        slot->computed_prompt_token_count,
        &physical_block_count,
        &resident_block_count,
        &nonresident_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    (void)physical_block_count;
    (void)resident_block_count;
    return nonresident_block_count == 0u;
}

static uint32_t SparkGlm52RequestApiOlderLowerPrioritySchedulableSlotExists(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *chosen_slot)
{
    uint32_t slot_index;

    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if ((SparkGlm52RequestApiSlotIsSchedulablePrefill(slot) ||
             SparkGlm52RequestApiSlotIsSchedulableDecode(slot)) &&
            slot->submission_order < chosen_slot->submission_order &&
            chosen_slot->priority > slot->priority)
        {
            return 1u;
        }
    }
    return 0u;
}

static void SparkGlm52RequestApiCollectPhysicalBlockIndex(
    uint32_t *physical_block_indices,
    uint32_t *physical_block_count,
    uint32_t physical_block_index)
{
    uint32_t block_index;

    for (block_index = 0u; block_index < *physical_block_count; ++block_index)
    {
        if (physical_block_indices[block_index] == physical_block_index)
        {
            return;
        }
    }
    if (*physical_block_count <
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT)
    {
        physical_block_indices[*physical_block_count] = physical_block_index;
        *physical_block_count += 1u;
    }
}

static SparkStatus SparkGlm52RequestApiCollectPrefillSlotBlocks(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t *physical_block_indices,
    uint32_t *physical_block_count)
{
    uint32_t matched_token_count;
    uint32_t slot_physical_block_count;
    uint32_t slot_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheProbePhysicalBlockTable(
        api->scheduler->prefix_cache,
        slot->prompt_token_ids,
        slot->prompt_token_count,
        slot_physical_block_indices,
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &matched_token_count,
        &slot_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    (void)matched_token_count;
    for (block_index = 0u;
         block_index < slot_physical_block_count &&
             *physical_block_count <
                 SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkGlm52RequestApiCollectPhysicalBlockIndex(
            physical_block_indices,
            physical_block_count,
            slot_physical_block_indices[block_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiCollectDecodeSlotBlocks(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t *physical_block_indices,
    uint32_t *physical_block_count)
{
    uint32_t slot_physical_block_count;
    uint32_t slot_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        slot->computed_prompt_token_count,
        slot_physical_block_indices,
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &slot_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (block_index = 0u;
         block_index < slot_physical_block_count &&
             *physical_block_count <
                 SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkGlm52RequestApiCollectPhysicalBlockIndex(
            physical_block_indices,
            physical_block_count,
            slot_physical_block_indices[block_index]);
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52RequestApiCollectPrefetchSourceBlock(
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t *source_block_count,
    const SparkGlm52KvCachePrefetchSourceBlock *source_block)
{
    uint32_t block_index;

    for (block_index = 0u; block_index < *source_block_count; ++block_index)
    {
        if (source_blocks[block_index].physical_block_index ==
                source_block->physical_block_index &&
            source_blocks[block_index].block_hash == source_block->block_hash &&
            source_blocks[block_index].content_hash == source_block->content_hash)
        {
            return;
        }
    }
    if (*source_block_count <
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT)
    {
        source_blocks[*source_block_count] = *source_block;
        *source_block_count += 1u;
    }
}

static SparkStatus SparkGlm52RequestApiCollectPrefillSlotPrefetchSources(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t *source_block_count)
{
    SparkGlm52KvCachePrefetchSourceBlock slot_source_blocks[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t matched_token_count;
    uint32_t slot_source_block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheProbeReusablePrefixPrefetchSources(
        api->scheduler->prefix_cache,
        slot->prompt_token_ids,
        slot->prompt_token_count,
        slot_source_blocks,
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &matched_token_count,
        &slot_source_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    (void)matched_token_count;
    for (block_index = 0u;
         block_index < slot_source_block_count &&
             *source_block_count <
                 SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkGlm52RequestApiCollectPrefetchSourceBlock(
            source_blocks,
            source_block_count,
            &slot_source_blocks[block_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiCollectDecodeSlotPrefetchSources(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t *source_block_count)
{
    SparkGlm52KvCachePrefetchSourceBlock slot_source_blocks[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t slot_source_block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52PrefixCacheBuildSequencePrefetchSources(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        slot->computed_prompt_token_count,
        slot_source_blocks,
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT,
        &slot_source_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (block_index = 0u;
         block_index < slot_source_block_count &&
             *source_block_count <
                 SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkGlm52RequestApiCollectPrefetchSourceBlock(
            source_blocks,
            source_block_count,
            &slot_source_blocks[block_index]);
    }
    return SPARK_STATUS_OK;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindBestPrefetchLookaheadSlot(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle *selected_handles,
    uint32_t selected_handle_count)
{
    SparkGlm52RequestApiSlot *best_slot;
    uint32_t slot_index;

    best_slot = 0;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t selected_index;
        uint32_t is_selected;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiSlotIsSchedulablePrefill(slot) &&
            !SparkGlm52RequestApiSlotIsSchedulableDecode(slot))
        {
            continue;
        }
        is_selected = 0u;
        for (selected_index = 0u;
             selected_index < selected_handle_count;
             ++selected_index)
        {
            if (selected_handles[selected_index] == slot->handle)
            {
                is_selected = 1u;
                break;
            }
        }
        if (is_selected != 0u)
        {
            continue;
        }
        if (SparkGlm52RequestApiSlotHasHigherSchedulingPriority(slot, best_slot))
        {
            best_slot = slot;
        }
    }
    return best_slot;
}

static SparkStatus SparkGlm52RequestApiRefreshLookaheadPrefixProtections(
    SparkGlm52RequestApi *api)
{
    SparkGlm52RequestApiHandle selected_handles[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t selected_handle_count;
    uint32_t lookahead_index;
    uint32_t total_protected_block_count;
    SparkStatus status;

    if (!SparkGlm52RequestApiQueueAwarePrefixCacheEvictionIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }
    if (api->scheduler == 0 || api->scheduler->prefix_cache == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52PrefixCacheResetLookaheadProtection(
        api->scheduler->prefix_cache);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    selected_handle_count = 0u;
    total_protected_block_count = 0u;
    for (lookahead_index = 0u;
         lookahead_index < api->prefetch_lookahead_request_count &&
             selected_handle_count <
                SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++lookahead_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t protected_token_count;
        uint32_t protected_block_count;

        slot = SparkGlm52RequestApiFindBestPrefetchLookaheadSlot(
            api,
            selected_handles,
            selected_handle_count);
        if (slot == 0)
        {
            break;
        }
        selected_handles[selected_handle_count] = slot->handle;
        selected_handle_count += 1u;

        if (slot->prompt_token_ids == 0 || slot->prompt_token_count == 0u)
        {
            continue;
        }
        status = SparkGlm52PrefixCacheProtectPromptLookahead(
            api->scheduler->prefix_cache,
            slot->prompt_token_ids,
            slot->prompt_token_count,
            slot->priority,
            &protected_token_count,
            &protected_block_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        total_protected_block_count += protected_block_count;
    }

    api->lookahead_protection_sweep_count += 1u;
    api->lookahead_protected_block_count += total_protected_block_count;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiJitResidencyPolicyIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return SparkGlm52RequestApiJitPrefetchIsEnabled(api) &&
        api->max_resident_kv_block_count != 0u &&
        api->scheduler != 0 &&
        api->scheduler->prefix_cache != 0 &&
        api->scheduler->prefix_cache->kv_cache_arena != 0;
}

static void SparkGlm52RequestApiCollectProtectedPrefetchPlanBlocks(
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    uint32_t *protected_physical_block_indices,
    uint32_t *protected_physical_block_count)
{
    uint32_t block_index;

    if (prefetch_plan == 0)
    {
        return;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count &&
             *protected_physical_block_count <
                SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++block_index)
    {
        SparkGlm52RequestApiCollectPhysicalBlockIndex(
            protected_physical_block_indices,
            protected_physical_block_count,
            prefetch_plan->blocks[block_index].physical_block_index);
    }
}

static SparkStatus SparkGlm52RequestApiCollectProtectedSlotBlocks(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t *protected_physical_block_indices,
    uint32_t *protected_physical_block_count)
{
    if (slot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52RequestApiSlotIsSchedulablePrefill(slot))
    {
        return SparkGlm52RequestApiCollectPrefillSlotBlocks(
            api,
            slot,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    if ((SparkGlm52RequestApiSlotIsSchedulableDecode(slot) ||
         slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE ||
         slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL ||
         slot->state == SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT) &&
        slot->computed_prompt_token_count != 0u)
    {
        return SparkGlm52RequestApiCollectDecodeSlotBlocks(
            api,
            slot,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiCollectRunningProtectedBlocks(
    SparkGlm52RequestApi *api,
    uint32_t *protected_physical_block_indices,
    uint32_t *protected_physical_block_count)
{
    uint32_t slot_index;

    for (slot_index = 0u;
         slot_index < api->request_capacity &&
             *protected_physical_block_count <
                SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        SparkStatus status;

        slot = &api->request_slots[slot_index];
        if (slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL &&
            slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE &&
            slot->state != SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT)
        {
            continue;
        }
        status = SparkGlm52RequestApiCollectProtectedSlotBlocks(
            api,
            slot,
            protected_physical_block_indices,
            protected_physical_block_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiApplyJitKvResidencyPolicy(
    SparkGlm52RequestApi *api,
    const SparkGlm52KvCachePrefetchPlan *protected_prefetch_plan,
    const uint32_t *additional_protected_physical_block_indices,
    uint32_t additional_protected_physical_block_count)
{
    uint32_t protected_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t protected_physical_block_count;
    uint32_t pending_index;
    uint32_t evicted_block_count;
    SparkStatus status;

    if (!SparkGlm52RequestApiJitResidencyPolicyIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    if (additional_protected_physical_block_count != 0u &&
        additional_protected_physical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    protected_physical_block_count = 0u;
    SparkGlm52RequestApiCollectProtectedPrefetchPlanBlocks(
        protected_prefetch_plan,
        protected_physical_block_indices,
        &protected_physical_block_count);
    for (pending_index = 0u;
         pending_index < additional_protected_physical_block_count;
         ++pending_index)
    {
        SparkGlm52RequestApiCollectPhysicalBlockIndex(
            protected_physical_block_indices,
            &protected_physical_block_count,
            additional_protected_physical_block_indices[pending_index]);
    }
    for (pending_index = 0u;
         pending_index < SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY;
         ++pending_index)
    {
        if (api->pending_prefetches[pending_index].active == 0u)
        {
            continue;
        }
        SparkGlm52RequestApiCollectProtectedPrefetchPlanBlocks(
            &api->pending_prefetches[pending_index].prefetch_plan,
            protected_physical_block_indices,
            &protected_physical_block_count);
    }

    status = SparkGlm52RequestApiCollectRunningProtectedBlocks(
        api,
        protected_physical_block_indices,
        &protected_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    evicted_block_count = 0u;
    status = SparkGlm52PrefixCacheTrimResidentBlocksByReuseScore(
        api->scheduler->prefix_cache,
        api->max_resident_kv_block_count,
        protected_physical_block_indices,
        protected_physical_block_count,
        &evicted_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    api->jit_residency_eviction_count += evicted_block_count;
    api->jit_residency_protected_block_count +=
        protected_physical_block_count +
        api->scheduler->prefix_cache->lookahead_protected_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiBuildJitKvPrefetchPlan(
    SparkGlm52RequestApi *api,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52RequestApiHandle selected_handles[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    SparkGlm52KvCachePrefetchSourceBlock source_blocks[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t selected_handle_count;
    uint32_t source_block_count;
    uint32_t lookahead_index;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || prefetch_plan == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return SparkGlm52KvCacheArenaBuildPrefetchPlan(
            api->scheduler->prefix_cache->kv_cache_arena,
            0,
            0u,
            api->prefetch_lane_count,
            prefetch_plan);
    }

    selected_handle_count = 0u;
    source_block_count = 0u;
    for (lookahead_index = 0u;
         lookahead_index < api->prefetch_lookahead_request_count &&
             selected_handle_count <
                SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++lookahead_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = SparkGlm52RequestApiFindBestPrefetchLookaheadSlot(
            api,
            selected_handles,
            selected_handle_count);
        if (slot == 0)
        {
            break;
        }
        selected_handles[selected_handle_count] = slot->handle;
        selected_handle_count += 1u;
        if (SparkGlm52RequestApiSlotIsSchedulablePrefill(slot))
        {
            status = SparkGlm52RequestApiCollectPrefillSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        else
        {
            status = SparkGlm52RequestApiCollectDecodeSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (source_block_count >=
            SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT)
        {
            break;
        }
    }

    return SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        source_block_count != 0u ? source_blocks : 0,
        source_block_count,
        api->prefetch_lane_count,
        prefetch_plan);
}

static uint32_t SparkGlm52RequestApiPrefetchBlockIsResident(
    const SparkGlm52RequestApi *api,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block)
{
    const SparkGlm52KvCacheArena *arena;
    const SparkGlm52KvCacheBlock *block;

    if (api == 0 ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        api->scheduler->prefix_cache->kv_cache_arena == 0 ||
        prefetch_block == 0)
    {
        return 0u;
    }

    arena = api->scheduler->prefix_cache->kv_cache_arena;
    if (prefetch_block->physical_block_index >= arena->physical_block_count)
    {
        return 0u;
    }

    block = &arena->blocks[prefetch_block->physical_block_index];
    return (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u &&
        (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u &&
        block->generation == prefetch_block->generation;
}

static uint32_t SparkGlm52RequestApiPrefetchPlanIsResident(
    const SparkGlm52RequestApi *api,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;

    if (prefetch_plan == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        if (!SparkGlm52RequestApiPrefetchBlockIsResident(
                api,
                &prefetch_plan->blocks[block_index]))
        {
            return 0u;
        }
    }
    return 1u;
}

static uint32_t SparkGlm52RequestApiPendingPrefetchContainsBlock(
    const SparkGlm52RequestApiPendingPrefetch *pending_prefetch,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block)
{
    uint32_t block_index;

    if (pending_prefetch == 0 ||
        pending_prefetch->active == 0u ||
        prefetch_block == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < pending_prefetch->prefetch_plan.prefetch_block_count;
         ++block_index)
    {
        const SparkGlm52KvCachePrefetchBlock *pending_block;

        pending_block = &pending_prefetch->prefetch_plan.blocks[block_index];
        if (pending_block->physical_block_index ==
                prefetch_block->physical_block_index &&
            pending_block->generation == prefetch_block->generation)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkGlm52RequestApiPendingPrefetchesCoverPlan(
    const SparkGlm52RequestApi *api,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;

    if (api == 0 || prefetch_plan == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        uint32_t pending_index;
        uint32_t found_pending_block;

        if (SparkGlm52RequestApiPrefetchBlockIsResident(
                api,
                &prefetch_plan->blocks[block_index]))
        {
            continue;
        }

        found_pending_block = 0u;
        for (pending_index = 0u;
             pending_index < SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY;
             ++pending_index)
        {
            if (SparkGlm52RequestApiPendingPrefetchContainsBlock(
                    &api->pending_prefetches[pending_index],
                    &prefetch_plan->blocks[block_index]))
            {
                found_pending_block = 1u;
                break;
            }
        }
        if (found_pending_block == 0u)
        {
            return 0u;
        }
    }
    return 1u;
}

static SparkGlm52RequestApiPendingPrefetch *
SparkGlm52RequestApiFindFreePendingPrefetch(
    SparkGlm52RequestApi *api)
{
    uint32_t pending_index;

    for (pending_index = 0u;
         pending_index < SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY;
         ++pending_index)
    {
        if (api->pending_prefetches[pending_index].active == 0u)
        {
            return &api->pending_prefetches[pending_index];
        }
    }
    return 0;
}

static void SparkGlm52RequestApiClearPendingPrefetch(
    SparkGlm52RequestApiPendingPrefetch *pending_prefetch)
{
    if (pending_prefetch == 0)
    {
        return;
    }
    memset(pending_prefetch, 0, sizeof(*pending_prefetch));
}

static SparkStatus SparkGlm52RequestApiPollOnePendingPrefetch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiPendingPrefetch *pending_prefetch)
{
    SparkStatus status;

    if (api == 0 || pending_prefetch == 0 || pending_prefetch->active == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = api->kv_prefetch_poll_function(
        api->kv_prefetch_context,
        pending_prefetch->prefetch_id,
        &pending_prefetch->prefetch_plan);
    pending_prefetch->poll_count += 1u;
    api->async_jit_prefetch_poll_count += 1u;
    if (status == SPARK_STATUS_BUSY)
    {
        return SPARK_STATUS_BUSY;
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RequestApiClearPendingPrefetch(pending_prefetch);
        return status;
    }

    status = SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
        api->scheduler->prefix_cache->kv_cache_arena,
        &pending_prefetch->prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RequestApiClearPendingPrefetch(pending_prefetch);
        return status;
    }
    status = SparkGlm52RequestApiApplyJitKvResidencyPolicy(
        api,
        &pending_prefetch->prefetch_plan,
        0,
        0u);
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52RequestApiClearPendingPrefetch(pending_prefetch);
        return status;
    }

    api->jit_prefetch_dispatch_count += 1u;
    api->jit_prefetch_block_count +=
        pending_prefetch->prefetch_plan.prefetch_block_count;
    api->async_jit_prefetch_completion_count += 1u;
    SparkGlm52RequestApiClearPendingPrefetch(pending_prefetch);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiPollPendingJitKvPrefetches(
    SparkGlm52RequestApi *api)
{
    uint32_t pending_index;

    for (pending_index = 0u;
         pending_index < SPARK_GLM52_REQUEST_API_PENDING_PREFETCH_CAPACITY;
         ++pending_index)
    {
        SparkStatus status;

        if (api->pending_prefetches[pending_index].active == 0u)
        {
            continue;
        }
        status = SparkGlm52RequestApiPollOnePendingPrefetch(
            api,
            &api->pending_prefetches[pending_index]);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiStartAsyncJitKvPrefetch(
    SparkGlm52RequestApi *api,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52RequestApiPendingPrefetch *pending_prefetch;
    SparkStatus status;
    uint64_t prefetch_id;

    pending_prefetch = SparkGlm52RequestApiFindFreePendingPrefetch(api);
    if (pending_prefetch == 0)
    {
        return SPARK_STATUS_BUSY;
    }

    prefetch_id = api->next_prefetch_id;
    api->next_prefetch_id += 1u;
    if (api->next_prefetch_id == 0u)
    {
        api->next_prefetch_id = 1u;
    }

    status = api->kv_prefetch_start_function(
        api->kv_prefetch_context,
        prefetch_id,
        prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(pending_prefetch, 0, sizeof(*pending_prefetch));
    pending_prefetch->active = 1u;
    pending_prefetch->prefetch_id = prefetch_id;
    pending_prefetch->prefetch_plan = *prefetch_plan;
    api->async_jit_prefetch_start_count += 1u;

    status = SparkGlm52RequestApiPollOnePendingPrefetch(
        api,
        pending_prefetch);
    if (status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_OK;
    }
    if (status == SPARK_STATUS_BUSY)
    {
        return SPARK_STATUS_BUSY;
    }
    return status;
}

static SparkStatus SparkGlm52RequestApiDispatchJitKvPrefetchWithProtectedBlocks(
    SparkGlm52RequestApi *api,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *additional_protected_physical_block_indices,
    uint32_t additional_protected_physical_block_count)
{
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || prefetch_plan == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }
    if (SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(api))
    {
        status = SparkGlm52RequestApiPollPendingJitKvPrefetches(api);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (prefetch_plan->prefetch_block_count == 0u)
        {
            return SparkGlm52RequestApiApplyJitKvResidencyPolicy(
                api,
                prefetch_plan,
                additional_protected_physical_block_indices,
                additional_protected_physical_block_count);
        }
        if (SparkGlm52RequestApiPrefetchPlanIsResident(api, prefetch_plan))
        {
            return SparkGlm52RequestApiApplyJitKvResidencyPolicy(
                api,
                prefetch_plan,
                additional_protected_physical_block_indices,
                additional_protected_physical_block_count);
        }
        if (SparkGlm52RequestApiPendingPrefetchesCoverPlan(
                api,
                prefetch_plan))
        {
            return SPARK_STATUS_BUSY;
        }
        return SparkGlm52RequestApiStartAsyncJitKvPrefetch(
            api,
            prefetch_plan);
    }

    if (prefetch_plan->prefetch_block_count == 0u)
    {
        return SparkGlm52RequestApiApplyJitKvResidencyPolicy(
            api,
            prefetch_plan,
            additional_protected_physical_block_indices,
            additional_protected_physical_block_count);
    }
    if (api->kv_prefetch_function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = api->kv_prefetch_function(
        api->kv_prefetch_context,
        prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52KvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        prefetch_plan,
        additional_protected_physical_block_indices,
        additional_protected_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52RequestApiApplyJitKvResidencyPolicy(
        api,
        prefetch_plan,
        additional_protected_physical_block_indices,
        additional_protected_physical_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    api->jit_prefetch_dispatch_count += 1u;
    api->jit_prefetch_block_count += prefetch_plan->prefetch_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiDispatchJitKvPrefetch(
    SparkGlm52RequestApi *api,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    return SparkGlm52RequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        prefetch_plan,
        0,
        0u);
}


static SparkStatus SparkGlm52RequestApiBuildSlotJitKvPrefetchPlan(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52KvCachePrefetchSourceBlock source_blocks[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t source_block_count;
    SparkStatus status;

    if (prefetch_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        memset(prefetch_plan, 0, sizeof(*prefetch_plan));
        return SPARK_STATUS_OK;
    }
    if (slot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    source_block_count = 0u;
    if (SparkGlm52RequestApiSlotIsSchedulablePrefill(slot))
    {
        status = SparkGlm52RequestApiCollectPrefillSlotPrefetchSources(
            api,
            slot,
            source_blocks,
            &source_block_count);
    }
    else if (SparkGlm52RequestApiSlotIsSchedulableDecode(slot))
    {
        status = SparkGlm52RequestApiCollectDecodeSlotPrefetchSources(
            api,
            slot,
            source_blocks,
            &source_block_count);
    }
    else
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    return SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        source_block_count != 0u ? source_blocks : 0,
        source_block_count,
        api->prefetch_lane_count,
        prefetch_plan);
}

static SparkStatus SparkGlm52RequestApiRunDispatchCriticalJitKvPrefetch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t critical_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t critical_physical_block_count;
    SparkStatus status;

    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    critical_physical_block_count = 0u;
    status = SparkGlm52RequestApiCollectProtectedSlotBlocks(
        api,
        slot,
        critical_physical_block_indices,
        &critical_physical_block_count);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
    {
        return status;
    }

    status = SparkGlm52RequestApiBuildSlotJitKvPrefetchPlan(
        api,
        slot,
        &dispatch->kv_prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52RequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        &dispatch->kv_prefetch_plan,
        critical_physical_block_indices,
        critical_physical_block_count);
    if (status == SPARK_STATUS_BUSY)
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        return SPARK_STATUS_BUSY;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->kv_prefetch_plan.prefetch_block_count != 0u)
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiBuildSlotArrayJitKvPrefetchPlan(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot **slots,
    uint32_t slot_count,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52KvCachePrefetchSourceBlock source_blocks[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t source_block_count;
    uint32_t slot_index;
    SparkStatus status;

    if (prefetch_plan == 0 || slots == 0 || slot_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        memset(prefetch_plan, 0, sizeof(*prefetch_plan));
        return SPARK_STATUS_OK;
    }

    source_block_count = 0u;
    for (slot_index = 0u;
         slot_index < slot_count &&
             source_block_count <
                SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT;
         ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = slots[slot_index];
        if (slot == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (SparkGlm52RequestApiSlotIsSchedulablePrefill(slot))
        {
            status = SparkGlm52RequestApiCollectPrefillSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        else if (SparkGlm52RequestApiSlotIsSchedulableDecode(slot))
        {
            status = SparkGlm52RequestApiCollectDecodeSlotPrefetchSources(
                api,
                slot,
                source_blocks,
                &source_block_count);
        }
        else
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    return SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        api->scheduler->prefix_cache->kv_cache_arena,
        source_block_count != 0u ? source_blocks : 0,
        source_block_count,
        api->prefetch_lane_count,
        prefetch_plan);
}

static SparkStatus SparkGlm52RequestApiRunSlotArrayCriticalJitKvPrefetch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot **slots,
    uint32_t slot_count,
    SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t critical_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t critical_physical_block_count;
    uint32_t slot_index;
    SparkStatus status;

    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    critical_physical_block_count = 0u;
    for (slot_index = 0u; slot_index < slot_count; ++slot_index)
    {
        status = SparkGlm52RequestApiCollectProtectedSlotBlocks(
            api,
            slots[slot_index],
            critical_physical_block_indices,
            &critical_physical_block_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }

    status = SparkGlm52RequestApiBuildSlotArrayJitKvPrefetchPlan(
        api,
        slots,
        slot_count,
        &dispatch->kv_prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52RequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        &dispatch->kv_prefetch_plan,
        critical_physical_block_indices,
        critical_physical_block_count);
    if (status == SPARK_STATUS_BUSY)
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        return SPARK_STATUS_BUSY;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->kv_prefetch_plan.prefetch_block_count != 0u)
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV;
    }
    return SPARK_STATUS_OK;
}


static uint32_t SparkGlm52RequestApiPrefetchPlanFitsResidentLimit(
    const SparkGlm52RequestApi *api,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count)
{
    uint32_t combined_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t combined_physical_block_count;
    uint32_t block_index;

    if (api == 0 || api->max_resident_kv_block_count == 0u)
    {
        return 1u;
    }
    combined_physical_block_count = 0u;
    for (block_index = 0u;
         block_index < protected_physical_block_count;
         ++block_index)
    {
        SparkGlm52RequestApiCollectPhysicalBlockIndex(
            combined_physical_block_indices,
            &combined_physical_block_count,
            protected_physical_block_indices[block_index]);
    }
    if (prefetch_plan != 0)
    {
        for (block_index = 0u;
             block_index < prefetch_plan->prefetch_block_count;
             ++block_index)
        {
            SparkGlm52RequestApiCollectPhysicalBlockIndex(
                combined_physical_block_indices,
                &combined_physical_block_count,
                prefetch_plan->blocks[block_index].physical_block_index);
        }
    }
    return combined_physical_block_count <= api->max_resident_kv_block_count;
}

static SparkStatus SparkGlm52RequestApiRunOpportunisticJitKvPrefetch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *protected_slot)
{
    uint32_t protected_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t protected_physical_block_count;
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    SparkStatus status;

    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return SPARK_STATUS_OK;
    }

    protected_physical_block_count = 0u;
    if (protected_slot != 0)
    {
        status = SparkGlm52RequestApiCollectProtectedSlotBlocks(
            api,
            protected_slot,
            protected_physical_block_indices,
            &protected_physical_block_count);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }

    status = SparkGlm52RequestApiBuildJitKvPrefetchPlan(
        api,
        &prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkGlm52RequestApiPrefetchPlanFitsResidentLimit(
            api,
            &prefetch_plan,
            protected_physical_block_indices,
            protected_physical_block_count))
    {
        return SparkGlm52RequestApiApplyJitKvResidencyPolicy(
            api,
            0,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    status = SparkGlm52RequestApiDispatchJitKvPrefetchWithProtectedBlocks(
        api,
        &prefetch_plan,
        protected_physical_block_indices,
        protected_physical_block_count);
    if (status == SPARK_STATUS_BUSY || status == SPARK_STATUS_CAPACITY_EXCEEDED)
    {
        return SparkGlm52RequestApiApplyJitKvResidencyPolicy(
            api,
            0,
            protected_physical_block_indices,
            protected_physical_block_count);
    }
    return status;
}

static void SparkGlm52RequestApiInitializeDispatch(
    SparkGlm52RequestApiDispatch *dispatch)
{
    memset(dispatch, 0, sizeof(*dispatch));
    dispatch->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    dispatch->descriptor_bytes = SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES;
}

static uint32_t SparkGlm52RequestApiProbeReusablePrefixTokenCount(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    SparkGlm52PrefixCacheLookup lookup;
    SparkStatus status;

    if (api == 0 ||
        api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        slot->prompt_token_ids == 0 ||
        slot->prompt_token_count == 0u)
    {
        return 0u;
    }

    status = SparkGlm52PrefixCacheProbePrompt(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        slot->prompt_token_ids,
        slot->prompt_token_count,
        &lookup);
    if (status != SPARK_STATUS_OK)
    {
        return slot->computed_prompt_token_count;
    }
    return SparkGlm52RequestApiMaximumU32(
        slot->computed_prompt_token_count,
        lookup.matched_token_count);
}


static uint32_t SparkGlm52RequestApiSchedulerMaxPrefillTokensPerStep(
    const SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    uint32_t max_prefill_tokens_per_step;

    max_prefill_tokens_per_step = slot->max_prefill_tokens_per_step;
    if (max_prefill_tokens_per_step == 0u)
    {
        max_prefill_tokens_per_step = api->scheduler->max_prefill_tokens_per_step;
    }
    if (max_prefill_tokens_per_step < api->scheduler->prefix_cache_block_tokens)
    {
        max_prefill_tokens_per_step = api->scheduler->prefix_cache_block_tokens;
    }
    return max_prefill_tokens_per_step;
}

static uint32_t SparkGlm52RequestApiConfigurationHasChunkedPrefill(
    const SparkGlm52RequestApi *api)
{
    return (api->scheduler->configuration_flags &
        SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL) != 0u;
}

static uint32_t SparkGlm52RequestApiRoundDownSchedulerBlock(
    const SparkGlm52RequestApi *api,
    uint32_t token_count)
{
    uint32_t block_token_count;

    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return token_count;
    }
    return token_count - (token_count % block_token_count);
}

static uint32_t SparkGlm52RequestApiNextPrefillStepTokenCount(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint32_t *computed_prompt_token_count_out)
{
    uint32_t cached_prefix_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t remaining_prompt_token_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t scheduled_prompt_token_count;

    cached_prefix_token_count = SparkGlm52RequestApiProbeReusablePrefixTokenCount(
        api,
        slot);
    computed_prompt_token_count = SparkGlm52RequestApiMaximumU32(
        slot->computed_prompt_token_count,
        cached_prefix_token_count);
    if (computed_prompt_token_count_out != 0)
    {
        *computed_prompt_token_count_out = computed_prompt_token_count;
    }
    if (computed_prompt_token_count >= slot->prompt_token_count)
    {
        return 0u;
    }
    remaining_prompt_token_count =
        slot->prompt_token_count - computed_prompt_token_count;
    if (!SparkGlm52RequestApiConfigurationHasChunkedPrefill(api))
    {
        return remaining_prompt_token_count;
    }
    max_prefill_tokens_per_step = SparkGlm52RequestApiSchedulerMaxPrefillTokensPerStep(
        api,
        slot);
    if (remaining_prompt_token_count <= max_prefill_tokens_per_step)
    {
        return remaining_prompt_token_count;
    }
    scheduled_prompt_token_count = SparkGlm52RequestApiRoundDownSchedulerBlock(
        api,
        max_prefill_tokens_per_step);
    if (scheduled_prompt_token_count == 0u)
    {
        scheduled_prompt_token_count = SparkGlm52RequestApiMinimumU32(
            remaining_prompt_token_count,
            api->scheduler->prefix_cache_block_tokens);
    }
    return scheduled_prompt_token_count;
}

static void SparkGlm52RequestApiFillPrefillSchedulerRequest(
    const SparkGlm52RequestApiSlot *slot,
    uint32_t prompt_token_count,
    uint32_t max_scheduled_prompt_token_count,
    SparkGlm52SchedulerRequest *scheduler_request)
{
    memset(scheduler_request, 0, sizeof(*scheduler_request));
    scheduler_request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_request->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    scheduler_request->active_sequence_count = 1u;
    scheduler_request->prompt_token_count = prompt_token_count;
    scheduler_request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_PREFILL;
    scheduler_request->max_scheduled_prompt_token_count =
        max_scheduled_prompt_token_count != 0u
            ? max_scheduled_prompt_token_count
            : slot->max_prefill_tokens_per_step;
    scheduler_request->sequence_id = slot->sequence_id;
    scheduler_request->prompt_token_ids = slot->prompt_token_ids;
}

static uint32_t SparkGlm52RequestApiFindBestSharedPrefixTokenCount(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *leader_slot)
{
    uint32_t slot_index;
    uint32_t best_shared_prefix_token_count;

    if (!SparkGlm52RequestApiPrefixCohortingIsEnabled(api))
    {
        return 0u;
    }

    best_shared_prefix_token_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *candidate;
        uint32_t candidate_shared_prefix_token_count;

        candidate = &api->request_slots[slot_index];
        if (candidate == leader_slot ||
            candidate->state != SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
        {
            continue;
        }
        candidate_shared_prefix_token_count =
            SparkGlm52RequestApiSharedCachePrefixTokenCount(
                api,
                leader_slot,
                candidate);
        if (candidate_shared_prefix_token_count >
            best_shared_prefix_token_count)
        {
            best_shared_prefix_token_count = candidate_shared_prefix_token_count;
        }
    }
    return best_shared_prefix_token_count;
}

static SparkStatus SparkGlm52RequestApiSchedulePrefill(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t selected_shared_prefix_token_count,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52SchedulerRequest scheduler_request;
    uint32_t slot_index;
    uint32_t scheduler_prompt_token_count;
    uint32_t scheduler_step_token_limit;
    uint32_t shared_prefix_token_count;
    uint32_t reusable_prefix_token_count;
    uint32_t committed_prefix_token_count;
    SparkStatus status;

    shared_prefix_token_count = selected_shared_prefix_token_count;
    if (shared_prefix_token_count == 0u)
    {
        shared_prefix_token_count = SparkGlm52RequestApiFindBestSharedPrefixTokenCount(
            api,
            slot);
    }
    reusable_prefix_token_count = SparkGlm52RequestApiProbeReusablePrefixTokenCount(
        api,
        slot);
    scheduler_prompt_token_count = slot->prompt_token_count;
    scheduler_step_token_limit = 0u;
    if (shared_prefix_token_count > reusable_prefix_token_count)
    {
        scheduler_step_token_limit =
            shared_prefix_token_count - reusable_prefix_token_count;
    }

    if (!SparkGlm52RequestApiPrefillCachedBlocksAreResident(
            api,
            slot,
            scheduler_prompt_token_count))
    {
        return SPARK_STATUS_BUSY;
    }

    SparkGlm52RequestApiFillPrefillSchedulerRequest(
        slot,
        scheduler_prompt_token_count,
        scheduler_step_token_limit,
        &scheduler_request);
    status = SparkGlm52SchedulerAdmit(
        api->scheduler,
        &scheduler_request,
        &dispatch->prefill_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->prefill_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    committed_prefix_token_count =
        dispatch->prefill_decision.cache_commit_token_count_after_step;
    if (committed_prefix_token_count > slot->prompt_token_count)
    {
        committed_prefix_token_count = slot->prompt_token_count;
    }

    slot->state = SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL;
    slot->scheduled_prefill_step_count += 1u;
    api->queued_request_count -= 1u;
    api->running_request_count += 1u;
    api->scheduled_prefill_dispatch_count += 1u;

    dispatch->accepted = 1u;
    dispatch->kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL;
    dispatch->request_count = 1u;
    dispatch->highest_priority = slot->priority;
    dispatch->shared_prefix_token_count =
        shared_prefix_token_count > committed_prefix_token_count
            ? committed_prefix_token_count
            : shared_prefix_token_count;
    if (api->scheduler != 0 && api->scheduler->prefix_cache_block_tokens != 0u)
    {
        dispatch->shared_prefix_block_count = dispatch->shared_prefix_token_count /
            api->scheduler->prefix_cache_block_tokens;
    }
    dispatch->prefix_cache_parent_hash =
        dispatch->prefill_decision.prefix_cache_parent_hash;
    dispatch->prefix_cache_result_hash =
        dispatch->prefill_decision.prefix_cache_result_hash;
    dispatch->request_handles[0] = slot->handle;
    dispatch->request_ids[0] = slot->request_id;
    dispatch->sequence_ids[0] = slot->sequence_id;

    if (dispatch->shared_prefix_token_count == 0u)
    {
        return SPARK_STATUS_OK;
    }

    for (slot_index = 0u;
         slot_index < api->request_capacity &&
             dispatch->request_count <
                SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
         ++slot_index)
    {
        SparkGlm52RequestApiSlot *candidate;

        candidate = &api->request_slots[slot_index];
        if (candidate == slot ||
            candidate->state != SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL ||
            SparkGlm52RequestApiSharedCachePrefixTokenCount(
                api,
                slot,
                candidate) < dispatch->shared_prefix_token_count)
        {
            continue;
        }
        candidate->state = SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT;
        candidate->scheduled_prefill_step_count += 1u;
        api->queued_request_count -= 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[dispatch->request_count] = candidate->handle;
        dispatch->request_ids[dispatch->request_count] = candidate->request_id;
        dispatch->sequence_ids[dispatch->request_count] = candidate->sequence_id;
        dispatch->request_count += 1u;
    }
    if (dispatch->request_count > 1u)
    {
        dispatch->flags |= SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT;
        if (selected_shared_prefix_token_count != 0u)
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFIX_FAMILY_SELECTED;
        }
        api->prefix_family_dispatch_count += 1u;
        api->prefix_family_member_count += dispatch->request_count;
        api->prefix_family_saved_prompt_token_count +=
            SparkGlm52RequestApiPrefixFamilySavedTokenCount(
                dispatch->shared_prefix_token_count,
                dispatch->request_count);
    }
    return SPARK_STATUS_OK;
}


static uint32_t SparkGlm52RequestApiPrefillBlockCountForScheduledTokens(
    const SparkGlm52RequestApi *api,
    uint32_t scheduled_prompt_token_count)
{
    uint32_t block_token_count;

    if (api == 0 ||
        api->scheduler == 0 ||
        scheduled_prompt_token_count == 0u)
    {
        return 0u;
    }

    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return 1u;
    }
    return (scheduled_prompt_token_count + block_token_count - 1u) /
        block_token_count;
}

static uint32_t SparkGlm52RequestApiSlotIsCompatiblePrefillBatchMember(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *leader_slot,
    const SparkGlm52RequestApiSlot *candidate_slot,
    uint32_t leader_prefill_block_count,
    uint32_t require_resident_cached_blocks,
    uint32_t *candidate_scheduled_prompt_token_count_out)
{
    uint32_t candidate_scheduled_prompt_token_count;
    uint32_t candidate_prefill_block_count;

    if (candidate_scheduled_prompt_token_count_out != 0)
    {
        *candidate_scheduled_prompt_token_count_out = 0u;
    }
    if (candidate_slot == leader_slot ||
        !SparkGlm52RequestApiSlotIsSchedulablePrefill(candidate_slot))
    {
        return 0u;
    }
    if ((require_resident_cached_blocks != 0u ||
         candidate_slot->priority < leader_slot->priority) &&
        !SparkGlm52RequestApiPrefillCachedBlocksAreResident(
            api,
            candidate_slot,
            candidate_slot->prompt_token_count))
    {
        return 0u;
    }

    candidate_scheduled_prompt_token_count =
        SparkGlm52RequestApiNextPrefillStepTokenCount(
            api,
            candidate_slot,
            0);
    candidate_prefill_block_count =
        SparkGlm52RequestApiPrefillBlockCountForScheduledTokens(
            api,
            candidate_scheduled_prompt_token_count);
    if (candidate_prefill_block_count == 0u ||
        candidate_prefill_block_count != leader_prefill_block_count)
    {
        return 0u;
    }
    if (candidate_scheduled_prompt_token_count_out != 0)
    {
        *candidate_scheduled_prompt_token_count_out =
            candidate_scheduled_prompt_token_count;
    }
    return 1u;
}

static uint32_t SparkGlm52RequestApiPrefillBatchCandidateIsBetter(
    const SparkGlm52RequestApiSlot *candidate_slot,
    uint32_t candidate_scheduled_prompt_token_count,
    const SparkGlm52RequestApiSlot *best_slot,
    uint32_t best_scheduled_prompt_token_count)
{
    if (best_slot == 0)
    {
        return 1u;
    }
    if (candidate_slot->priority != best_slot->priority)
    {
        return candidate_slot->priority > best_slot->priority;
    }
    if (candidate_scheduled_prompt_token_count !=
        best_scheduled_prompt_token_count)
    {
        return candidate_scheduled_prompt_token_count >
            best_scheduled_prompt_token_count;
    }
    return candidate_slot->submission_order < best_slot->submission_order;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindBestPrefillBatchMember(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *leader_slot,
    uint32_t leader_prefill_block_count,
    SparkGlm52RequestApiHandle *selected_handles,
    uint32_t selected_handle_count,
    uint32_t require_resident_cached_blocks,
    uint32_t *selected_scheduled_prompt_token_count_out)
{
    SparkGlm52RequestApiSlot *best_slot;
    uint32_t best_scheduled_prompt_token_count;
    uint32_t slot_index;

    if (selected_scheduled_prompt_token_count_out != 0)
    {
        *selected_scheduled_prompt_token_count_out = 0u;
    }
    best_slot = 0;
    best_scheduled_prompt_token_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t selected_index;
        uint32_t is_selected;
        uint32_t candidate_scheduled_prompt_token_count;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiSlotIsCompatiblePrefillBatchMember(
                api,
                leader_slot,
                slot,
                leader_prefill_block_count,
                require_resident_cached_blocks,
                &candidate_scheduled_prompt_token_count))
        {
            continue;
        }
        is_selected = 0u;
        for (selected_index = 0u;
             selected_index < selected_handle_count;
             ++selected_index)
        {
            if (selected_handles[selected_index] == slot->handle)
            {
                is_selected = 1u;
                break;
            }
        }
        if (is_selected != 0u)
        {
            continue;
        }
        if (SparkGlm52RequestApiPrefillBatchCandidateIsBetter(
                slot,
                candidate_scheduled_prompt_token_count,
                best_slot,
                best_scheduled_prompt_token_count))
        {
            best_slot = slot;
            best_scheduled_prompt_token_count =
                candidate_scheduled_prompt_token_count;
        }
    }
    if (selected_scheduled_prompt_token_count_out != 0)
    {
        *selected_scheduled_prompt_token_count_out =
            best_scheduled_prompt_token_count;
    }
    return best_slot;
}

static SparkStatus SparkGlm52RequestApiSchedulePrefillBatch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *first_slot,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52SchedulerRequest scheduler_requests[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52SchedulerPrefillBatchRequest batch_request;
    SparkGlm52RequestApiHandle selected_handles[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52RequestApiSlot *selected_slots[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52RequestApiSlot *slot;
    uint32_t request_count;
    uint32_t request_index;
    uint32_t batch_target;
    uint32_t leader_scheduled_prompt_token_count;
    uint32_t leader_prefill_block_count;
    uint32_t require_resident_batch_members;
    uint32_t selected_scheduled_prompt_token_counts[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkStatus status;

    if (!SparkGlm52RequestApiPrefillBatchingIsEnabled(api))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (!SparkGlm52RequestApiPrefillCachedBlocksAreResident(
            api,
            first_slot,
            first_slot->prompt_token_count))
    {
        return SPARK_STATUS_BUSY;
    }
    leader_scheduled_prompt_token_count =
        SparkGlm52RequestApiNextPrefillStepTokenCount(
            api,
            first_slot,
            0);
    leader_prefill_block_count =
        SparkGlm52RequestApiPrefillBlockCountForScheduledTokens(
            api,
            leader_scheduled_prompt_token_count);
    if (leader_prefill_block_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    batch_target = api->decode_batch_target;
    if (batch_target > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        batch_target = SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    require_resident_batch_members =
        SparkGlm52RequestApiSlotHasRealtimePriority(first_slot);
    request_count = 0u;
    slot = first_slot;
    while (slot != 0 && request_count < batch_target)
    {
        if (request_count == 0u)
        {
            selected_scheduled_prompt_token_counts[request_count] =
                leader_scheduled_prompt_token_count;
        }
        selected_slots[request_count] = slot;
        selected_handles[request_count] = slot->handle;
        SparkGlm52RequestApiFillPrefillSchedulerRequest(
            slot,
            slot->prompt_token_count,
            selected_scheduled_prompt_token_counts[request_count],
            &scheduler_requests[request_count]);
        request_count += 1u;
        slot = SparkGlm52RequestApiFindBestPrefillBatchMember(
            api,
            first_slot,
            leader_prefill_block_count,
            selected_handles,
            request_count,
            require_resident_batch_members,
            request_count < batch_target
                ? &selected_scheduled_prompt_token_counts[request_count]
                : 0);
    }
    if (request_count < 2u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    status = SparkGlm52RequestApiRunSlotArrayCriticalJitKvPrefetch(
        api,
        selected_slots,
        request_count,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        if (!SparkGlm52RequestApiPrefillCachedBlocksAreResident(
                api,
                selected_slots[request_index],
                selected_slots[request_index]->prompt_token_count))
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return SPARK_STATUS_BUSY;
        }
    }

    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = request_count;
    batch_request.requests = scheduler_requests;
    status = SparkGlm52SchedulerAdmitPrefillBatch(
        api->scheduler,
        &batch_request,
        &dispatch->prefill_batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->prefill_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    dispatch->accepted = 1u;
    dispatch->kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH;
    dispatch->flags |= SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PREFILL_BATCH;
    dispatch->request_count =
        dispatch->prefill_batch_decision.packed_request_count;
    dispatch->highest_priority = first_slot->priority;
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *selected_slot;

        selected_slot = selected_slots[request_index];
        selected_slot->state = SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL;
        selected_slot->scheduled_prefill_step_count += 1u;
        api->queued_request_count -= 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_ids[request_index] = selected_slot->request_id;
        dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
    }
    api->scheduled_prefill_dispatch_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlm52RequestApiFillDecodeSchedulerRequest(
    SparkGlm52SchedulerRequest *scheduler_request)
{
    memset(scheduler_request, 0, sizeof(*scheduler_request));
    scheduler_request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_request->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    scheduler_request->active_sequence_count = 1u;
    scheduler_request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE;
}

static SparkStatus SparkGlm52RequestApiScheduleDecodeBatch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *first_slot,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52SchedulerRequest scheduler_requests[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52SchedulerBatchRequest batch_request;
    SparkGlm52RequestApiHandle selected_handles[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52RequestApiSlot *selected_slots[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52RequestApiSlot *slot;
    uint32_t request_count;
    uint32_t request_index;
    uint32_t batch_target;
    uint32_t require_resident_batch_members;
    SparkStatus status;

    batch_target = SparkGlm52RequestApiDecodeBatchingIsEnabled(api)
        ? api->decode_batch_target
        : 1u;
    require_resident_batch_members =
        SparkGlm52RequestApiSlotHasRealtimePriority(first_slot) ||
        !SparkGlm52RequestApiJitPrefetchIsEnabled(api);
    request_count = 0u;
    slot = first_slot;
    while (slot != 0 && request_count < batch_target)
    {
        selected_slots[request_count] = slot;
        selected_handles[request_count] = slot->handle;
        SparkGlm52RequestApiFillDecodeSchedulerRequest(
            &scheduler_requests[request_count]);
        request_count += 1u;
        slot = SparkGlm52RequestApiFindBestDecodeBatchMember(
            api,
            first_slot,
            selected_handles,
            request_count,
            require_resident_batch_members);
    }
    if (request_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    status = SparkGlm52RequestApiRunSlotArrayCriticalJitKvPrefetch(
        api,
        selected_slots,
        request_count,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        if (!SparkGlm52RequestApiDecodeBlocksAreResident(
                api,
                selected_slots[request_index]))
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return SPARK_STATUS_BUSY;
        }
    }

    memset(&batch_request, 0, sizeof(batch_request));
    batch_request.abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    batch_request.descriptor_bytes =
        SPARK_GLM52_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES;
    batch_request.request_count = request_count;
    batch_request.requests = scheduler_requests;
    status = SparkGlm52SchedulerAdmitDecodeBatch(
        api->scheduler,
        &batch_request,
        &dispatch->decode_batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->decode_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    dispatch->accepted = 1u;
    dispatch->kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
    dispatch->request_count =
        dispatch->decode_batch_decision.packed_request_count;
    dispatch->highest_priority = first_slot->priority;
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *selected_slot;

        selected_slot = selected_slots[request_index];
        selected_slot->state = SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE;
        selected_slot->scheduled_decode_token_count += 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_ids[request_index] = selected_slot->request_id;
        dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
    }
    api->scheduled_decode_dispatch_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiScheduleNext(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52RequestApiSlot *prefill_slot;
    SparkGlm52RequestApiSlot *decode_slot;
    SparkGlm52RequestApiSlot *chosen_slot;
    SparkGlm52RequestApiPrefixFamilyChoice prefix_family_choice;
    uint32_t chosen_is_prefill;
    uint32_t selected_shared_prefix_token_count;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    SparkGlm52RequestApiInitializeDispatch(dispatch);
    if (SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(api))
    {
        uint64_t completed_prefetch_count_before_poll;

        completed_prefetch_count_before_poll =
            api->async_jit_prefetch_completion_count;
        status = SparkGlm52RequestApiPollPendingJitKvPrefetches(api);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (api->async_jit_prefetch_completion_count >
            completed_prefetch_count_before_poll)
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV;
        }
    }
    status = SparkGlm52RequestApiRefreshLookaheadPrefixProtections(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    selected_shared_prefix_token_count = 0u;
    prefill_slot = SparkGlm52RequestApiFindBestPrefillSlot(api);
    if (SparkGlm52RequestApiBuildBestPrefixFamilyChoice(
            api,
            &prefix_family_choice) &&
        SparkGlm52RequestApiPrefixFamilyChoiceBeatsPrefillSlot(
            &prefix_family_choice,
            prefill_slot))
    {
        prefill_slot = prefix_family_choice.leader_slot;
        selected_shared_prefix_token_count =
            prefix_family_choice.shared_prefix_token_count;
    }
    decode_slot = SparkGlm52RequestApiFindBestDecodeSlot(
        api,
        0,
        0u,
        SparkGlm52RequestApiJitPrefetchIsEnabled(api) ? 0u : 1u);
    if (prefill_slot == 0 && decode_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    chosen_is_prefill = 0u;
    chosen_slot = decode_slot;
    if (prefill_slot != 0 &&
        (decode_slot == 0 ||
         SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
            prefill_slot,
            decode_slot)))
    {
        chosen_is_prefill = 1u;
        chosen_slot = prefill_slot;
    }

    status = SparkGlm52RequestApiRunDispatchCriticalJitKvPrefetch(
        api,
        chosen_slot,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52RequestApiRunOpportunisticJitKvPrefetch(
        api,
        chosen_slot);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (chosen_is_prefill != 0u)
    {
        if ((decode_slot != 0 &&
             SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
                prefill_slot,
                decode_slot)) ||
            SparkGlm52RequestApiOlderLowerPrioritySchedulableSlotExists(
                api,
                prefill_slot))
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
        }
        if ((selected_shared_prefix_token_count != 0u
                ? selected_shared_prefix_token_count
                : SparkGlm52RequestApiFindBestSharedPrefixTokenCount(
                    api,
                    prefill_slot)) <=
            SparkGlm52RequestApiProbeReusablePrefixTokenCount(
                api,
                prefill_slot))
        {
            status = SparkGlm52RequestApiSchedulePrefillBatch(
                api,
                prefill_slot,
                dispatch);
            if (status != SPARK_STATUS_NOT_FOUND)
            {
                return status;
            }
        }
        return SparkGlm52RequestApiSchedulePrefill(
            api,
            prefill_slot,
            selected_shared_prefix_token_count,
            dispatch);
    }

    if (!SparkGlm52RequestApiDecodeBlocksAreResident(api, decode_slot))
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        return SPARK_STATUS_BUSY;
    }
    if (SparkGlm52RequestApiOlderLowerPrioritySchedulableSlotExists(
            api,
            decode_slot))
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
    }
    return SparkGlm52RequestApiScheduleDecodeBatch(api, decode_slot, dispatch);
}

static void SparkGlm52RequestApiFinishSlotAfterPrefill(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    const SparkGlm52SchedulerDecision *decision)
{
    uint32_t committed_prompt_token_count;

    committed_prompt_token_count = decision->cache_commit_token_count_after_step;
    if (committed_prompt_token_count > slot->prompt_token_count)
    {
        committed_prompt_token_count = slot->prompt_token_count;
    }
    if (committed_prompt_token_count > slot->computed_prompt_token_count)
    {
        slot->computed_prompt_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_hash = decision->prefix_cache_result_hash;
    }

    slot->completed_prefill_step_count += 1u;
    api->running_request_count -= 1u;
    if (slot->computed_prompt_token_count < slot->prompt_token_count)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
        api->queued_request_count += 1u;
        return;
    }

    slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
    }
}


static void SparkGlm52RequestApiFinishSlotAfterPrefillBatchLane(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    const SparkGlm52SchedulerPrefillBatchLane *lane)
{
    uint32_t committed_prompt_token_count;

    committed_prompt_token_count = lane->cache_commit_token_count_after_step;
    if (committed_prompt_token_count > slot->prompt_token_count)
    {
        committed_prompt_token_count = slot->prompt_token_count;
    }
    if (committed_prompt_token_count > slot->computed_prompt_token_count)
    {
        slot->computed_prompt_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_token_count = committed_prompt_token_count;
        slot->last_committed_prefix_hash = lane->prefix_cache_result_hash;
    }

    slot->completed_prefill_step_count += 1u;
    api->running_request_count -= 1u;
    if (slot->computed_prompt_token_count < slot->prompt_token_count)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
        api->queued_request_count += 1u;
        return;
    }

    slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
    }
}

static void SparkGlm52RequestApiFinishSlotAfterDecode(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot)
{
    if (slot->remaining_thinking_token_budget != 0u)
    {
        slot->remaining_thinking_token_budget -= 1u;
    }
    else if (slot->remaining_output_token_budget != 0u)
    {
        slot->remaining_output_token_budget -= 1u;
    }
    slot->completed_decode_token_count += 1u;
    api->running_request_count -= 1u;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
    }
    else
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    }
}

SparkStatus SparkGlm52RequestApiCompleteDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0 ||
        dispatch->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u || dispatch->request_count == 0u)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        status = SparkGlm52SchedulerComplete(
            api->scheduler,
            &dispatch->prefill_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                (request_index == 0u &&
                 slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL) ||
                (request_index != 0u &&
                 slot->state !=
                    SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            if (request_index != 0u &&
                dispatch->prefill_decision.cache_commit_token_count_after_step != 0u)
            {
                status = SparkGlm52PrefixCacheBindCommittedPrefixFromSequence(
                    api->scheduler->prefix_cache,
                    dispatch->sequence_ids[0],
                    slot->sequence_id,
                    dispatch->prefill_decision.cache_commit_token_count_after_step);
                if (status != SPARK_STATUS_OK)
                {
                    return status;
                }
            }
            SparkGlm52RequestApiFinishSlotAfterPrefill(
                api,
                slot,
                &dispatch->prefill_decision);
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        status = SparkGlm52SchedulerCompletePrefillBatch(
            api->scheduler,
            &dispatch->prefill_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            SparkGlm52RequestApiFinishSlotAfterPrefillBatchLane(
                api,
                slot,
                &dispatch->prefill_batch_decision.lanes[request_index]);
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        status = SparkGlm52SchedulerCompleteDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 || slot->state !=
                SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            SparkGlm52RequestApiFinishSlotAfterDecode(api, slot);
        }
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkGlm52RequestApiDispatchLaneCount(
    const SparkGlm52RequestApiDispatch *dispatch)
{
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        return dispatch->prefill_decision.active_sequence_count;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        return dispatch->prefill_batch_decision.active_sequence_count;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        return dispatch->decode_batch_decision.active_sequence_count;
    }
    return 0u;
}

SparkStatus SparkGlm52RequestApiBuildDispatchKvBlockTables(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity)
{
    uint32_t lane_index;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0 ||
        dispatch->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        physical_block_indices == 0 ||
        lane_physical_block_counts == 0 ||
        lane_capacity == 0u ||
        lane_stride < lane_capacity)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        if (lane_count_capacity < 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (lane_index = 0u; lane_index < lane_count_capacity; ++lane_index)
        {
            lane_physical_block_counts[lane_index] = 0u;
        }
        return SparkGlm52SchedulerBuildKvBlockTable(
            api->scheduler,
            &dispatch->prefill_decision,
            physical_block_indices,
            lane_capacity,
            &lane_physical_block_counts[0]);
    }

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        return SparkGlm52SchedulerBuildPrefillBatchKvBlockTables(
            api->scheduler,
            &dispatch->prefill_batch_decision,
            physical_block_indices,
            lane_stride,
            lane_capacity,
            lane_physical_block_counts,
            lane_count_capacity);
    }

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        if (lane_count_capacity < dispatch->request_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (lane_index = 0u; lane_index < lane_count_capacity; ++lane_index)
        {
            lane_physical_block_counts[lane_index] = 0u;
        }
        for (lane_index = 0u; lane_index < dispatch->request_count; ++lane_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[lane_index]);
            if (slot == 0 ||
                slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE ||
                slot->computed_prompt_token_count == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkGlm52PrefixCacheBuildPhysicalBlockTable(
                api->scheduler->prefix_cache,
                slot->sequence_id,
                slot->computed_prompt_token_count,
                &physical_block_indices[(uint64_t)lane_index * lane_stride],
                lane_capacity,
                &lane_physical_block_counts[lane_index]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkGlm52RequestApiBuildDispatchKvBlockTableView(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *host_physical_block_indices,
    const uint32_t *execution_physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity,
    SparkGlm52KvBlockTableView *block_table_view)
{
    uint32_t lane_count;
    SparkStatus status;

    if (block_table_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(block_table_view, 0, sizeof(*block_table_view));

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (dispatch == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    lane_count = SparkGlm52RequestApiDispatchLaneCount(dispatch);
    if (dispatch->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        lane_count == 0u ||
        lane_count > lane_count_capacity ||
        host_physical_block_indices == 0 ||
        lane_physical_block_counts == 0 ||
        lane_capacity == 0u ||
        lane_stride < lane_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52RequestApiBuildDispatchKvBlockTables(
        api,
        dispatch,
        host_physical_block_indices,
        lane_stride,
        lane_capacity,
        lane_physical_block_counts,
        lane_count_capacity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    block_table_view->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    block_table_view->descriptor_bytes =
        SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
    block_table_view->block_token_count = api->scheduler->prefix_cache_block_tokens;
    block_table_view->lane_count = lane_count;
    block_table_view->lane_stride = lane_stride;
    block_table_view->lane_capacity = lane_capacity;
    block_table_view->physical_block_indices = execution_physical_block_indices != 0 ?
        execution_physical_block_indices : host_physical_block_indices;
    block_table_view->lane_physical_block_counts = lane_physical_block_counts;
    block_table_view->host_physical_block_indices = host_physical_block_indices;

    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiReleaseSlotSequence(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot)
{
    SparkStatus status;

    if (slot->sequence_id == 0u)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkGlm52SchedulerReleaseSequence(api->scheduler, slot->sequence_id);
    if (status == SPARK_STATUS_OK)
    {
        slot->sequence_id = 0u;
    }
    return status;
}

SparkStatus SparkGlm52RequestApiCancelDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || dispatch == 0 || dispatch->accepted == 0u)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        status = SparkGlm52SchedulerCancel(
            api->scheduler,
            &dispatch->prefill_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                (request_index == 0u &&
                 slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL) ||
                (request_index != 0u &&
                 slot->state !=
                    SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_GLM52_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkGlm52RequestApiReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        status = SparkGlm52SchedulerCancelPrefillBatch(
            api->scheduler,
            &dispatch->prefill_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_GLM52_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkGlm52RequestApiReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH)
    {
        status = SparkGlm52SchedulerCancelDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 || slot->state !=
                SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            api->running_request_count -= 1u;
            slot->state = SPARK_GLM52_REQUEST_API_STATE_CANCELLED;
            api->cancelled_request_count += 1u;
            status = SparkGlm52RequestApiReleaseSlotSequence(api, slot);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkGlm52RequestApiGetRequestCacheState(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle,
    SparkGlm52RequestApiCacheState *cache_state)
{
    SparkGlm52RequestApiSlot *slot;
    uint32_t physical_block_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK || cache_state == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }
    slot = SparkGlm52RequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(cache_state, 0, sizeof(*cache_state));
    cache_state->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    cache_state->descriptor_bytes =
        SPARK_GLM52_REQUEST_API_CACHE_STATE_DESCRIPTOR_BYTES;
    cache_state->state = slot->state;
    cache_state->computed_prompt_token_count =
        slot->computed_prompt_token_count;
    cache_state->last_committed_prefix_token_count =
        slot->last_committed_prefix_token_count;
    cache_state->request_id = slot->request_id;
    cache_state->sequence_id = slot->sequence_id;
    cache_state->last_committed_prefix_hash =
        slot->last_committed_prefix_hash;

    if (slot->computed_prompt_token_count != 0u &&
        api->scheduler != 0 &&
        api->scheduler->prefix_cache != 0)
    {
        status = SparkGlm52PrefixCacheProbeSequenceResidency(
            api->scheduler->prefix_cache,
            slot->sequence_id,
            slot->computed_prompt_token_count,
            &physical_block_count,
            &resident_block_count,
            &nonresident_block_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        (void)resident_block_count;
        (void)nonresident_block_count;
        cache_state->physical_block_count = physical_block_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiCancelRequest(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle)
{
    SparkGlm52RequestApiSlot *slot;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkGlm52RequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL ||
        slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE)
    {
        return SPARK_STATUS_BUSY;
    }
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        api->queued_request_count -= 1u;
    }
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_COMPLETED)
    {
        api->completed_request_count -= 1u;
    }
    slot->state = SPARK_GLM52_REQUEST_API_STATE_CANCELLED;
    api->cancelled_request_count += 1u;
    return SparkGlm52RequestApiReleaseSlotSequence(api, slot);
}

SparkStatus SparkGlm52RequestApiReleaseCompletedRequest(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle)
{
    SparkGlm52RequestApiSlot *slot;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    slot = SparkGlm52RequestApiFindSlotByHandle(api, handle);
    if (slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (slot->state != SPARK_GLM52_REQUEST_API_STATE_COMPLETED &&
        slot->state != SPARK_GLM52_REQUEST_API_STATE_CANCELLED)
    {
        return SPARK_STATUS_BUSY;
    }
    status = SparkGlm52RequestApiReleaseSlotSequence(api, slot);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkGlm52RequestApiInitializeSlot(slot);
    return SPARK_STATUS_OK;
}
