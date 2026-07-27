// The request surface: session and slot lifecycle, stop conditions, validation,
// and the speculative dispatch policy.
//
// 7,178 lines carrying 34 model constants, which sounds model-specific and is
// not. Measured by concern:
//
//     4,176 lines   6 model constants   session and request plumbing
//     1,603         5                   slot lifecycle
//     1,040        20                   SPECULATIVE DISPATCH POLICY
//       309         2                   stop conditions
//       198         0                   validation
//       137         0                   cost model
//
// Twenty of the thirty-four are in the speculative dispatch policy, and that is
// where they belong: whether drafting beats plain decode depends on the model's
// draft token count, its layer count, and how fast its first layers are. This
// file's own comments record the measurement - plain 3.89 tok/s against MTP 3.47
// at B1, so MTP was LOSING - and note that without hysteresis the scheduler
// oscillates around the break-even point.
//
// WHICH RANK DRAFTS IS A MODEL PROPERTY, NOT A CONSTANT. The policy here assumes
// the last rank, which is where the logits are. For GLM 5.2 that is the wrong
// choice: its first three layers are dense and fast, so the rank holding them
// has slack the last rank does not, and drafting there costs less. A model whose
// early layers are not cheap would want the opposite. That belongs in the model
// description as a list of drafting ranks, and it is not there yet.
#include "sparkpipe/spark_glm52_request_api.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "sparkpipe/spark_glm52_row_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_GLM52_REQUEST_API_MTP_UTILITY_SCALE 1000ull
#define SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_SAMPLE_COUNT \
    SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_COMMITTED_TOKEN_COUNT \
    (SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT / 2u)
/* An MTP cycle pays for the draft-chain dispatches in addition to the
 * verify dispatch. The measured-plan cost model only sees batched
 * weight streaming, so it undercounts the serialized draft work on the
 * final rank. Calibrated against the 2026-07-17 B1 A/B (plain 3.89
 * tok/s vs MTP 3.47 tok/s): re-measure per release and adjust. */
#define SPARK_GLM52_REQUEST_API_MTP_DRAFT_CHAIN_WORK_MULTIPLIER 2u
/* MTP must beat plain decode by this margin before new drafts are
 * budgeted; without hysteresis the scheduler oscillates between
 * drafting and plain decode around the break-even point. */
#define SPARK_GLM52_REQUEST_API_MTP_UTILITY_MARGIN_SCALE 1250ull

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

static uint32_t SparkGlm52RequestApiNormalizeDecodeExecutionRowCapacity(
    const SparkGlm52RequestApiConfiguration *configuration)
{
    uint32_t decode_batch_target;
    uint32_t maximum_speculative_token_count;

    if (configuration == 0)
    {
        return 0u;
    }
    if (configuration->decode_execution_row_capacity != 0u)
    {
        return configuration->decode_execution_row_capacity;
    }
    decode_batch_target = SparkGlm52RequestApiNormalizeDecodeBatchTarget(
        configuration->decode_batch_target);
    maximum_speculative_token_count =
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT >
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT
        ? SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
        : SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT;
    return decode_batch_target * (maximum_speculative_token_count + 1u);
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

static uint32_t SparkGlm52RequestApiMtpCommitIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT) != 0u;
}

static uint32_t SparkGlm52RequestApiDecodeBatchingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING) != 0u;
}

static uint32_t SparkGlm52RequestApiAdaptivePipelineBatchingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING) != 0u;
}

static uint32_t SparkGlm52RequestApiPrefixCohortingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING) != 0u;
}

static uint32_t SparkGlm52RequestApiCrossSequencePrefixReuseIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return api != 0 && api->scheduler != 0 &&
        (api->scheduler->configuration_flags &
            SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE) != 0u;
}

static uint32_t SparkGlm52RequestApiPrefillBatchingIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING) != 0u;
}

static uint32_t SparkGlm52RequestApiDsparkSpeculationIsEnabled(
    const SparkGlm52RequestApi *api)
{
    return (api->configuration_flags &
        SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE) != 0u &&
        api->dspark_speculator != 0;
}


static SparkStatus SparkGlm52RequestApiValidate(
    const SparkGlm52RequestApi *api)
{
    if (api == 0 ||
        api->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        api->descriptor_bytes != SPARK_GLM52_REQUEST_API_DESCRIPTOR_BYTES ||
        api->request_capacity == 0u ||
        api->decode_batch_target == 0u ||
        api->decode_execution_row_capacity < api->decode_batch_target ||
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
    uint32_t decode_batch_target;
    uint32_t decode_execution_row_capacity;
    uint32_t prefetch_lane_count;
    SparkStatus status;

    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->request_capacity == 0u ||
        configuration->request_slots == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    configuration_flags = SparkGlm52RequestApiNormalizeConfigurationFlags(
        configuration->configuration_flags);
    prefetch_lane_count = SparkGlm52RequestApiNormalizePrefetchLaneCount(
        configuration->prefetch_lane_count);
    decode_batch_target = SparkGlm52RequestApiNormalizeDecodeBatchTarget(
        configuration->decode_batch_target);
    decode_execution_row_capacity =
        SparkGlm52RequestApiNormalizeDecodeExecutionRowCapacity(configuration);
    if (!SparkGlm52RequestApiConfigurationFlagsAreValid(configuration_flags) ||
        prefetch_lane_count == 0u ||
        prefetch_lane_count > SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        decode_execution_row_capacity < decode_batch_target)
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
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE) != 0u &&
        (configuration->dspark_speculator == 0 ||
         SparkGlm52DsparkValidate(configuration->dspark_speculator) !=
            SPARK_STATUS_OK))
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
    slot->handle_hash_next = SPARK_GLM52_REQUEST_API_NO_SLOT;
    slot->free_slot_next = SPARK_GLM52_REQUEST_API_NO_SLOT;
    slot->mtp_commit_ema_milli =
        SPARK_GLM52_REQUEST_API_MTP_COMMIT_EMA_INITIAL_MILLI;
}

static uint32_t SparkGlm52RequestApiHashHandle(
    SparkGlm52RequestApiHandle handle)
{
    uint64_t hash;

    hash = handle;
    hash ^= (hash >> 33u);
    hash *= 0xff51afd7ed558ccdull;
    hash ^= (hash >> 33u);
    return (uint32_t)(hash % SPARK_GLM52_REQUEST_API_SLOT_HASH_SLOTS);
}

static uint32_t SparkGlm52RequestApiSlotIndex(
    const SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    uint64_t byte_offset;
    uint64_t slot_index;

    if (api == 0 || slot == 0 || api->request_slots == 0 ||
        slot < api->request_slots ||
        slot >= &api->request_slots[api->request_capacity])
    {
        return SPARK_GLM52_REQUEST_API_NO_SLOT;
    }
    byte_offset = (uint64_t)((uintptr_t)slot - (uintptr_t)api->request_slots);
    slot_index = byte_offset / (uint64_t)sizeof(*slot);
    if (slot_index >= api->request_capacity)
    {
        return SPARK_GLM52_REQUEST_API_NO_SLOT;
    }
    return (uint32_t)slot_index;
}

static void SparkGlm52RequestApiInsertSlotHash(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot)
{
    uint32_t slot_index;
    uint32_t hash_slot;

    slot_index = SparkGlm52RequestApiSlotIndex(api, slot);
    if (slot_index == SPARK_GLM52_REQUEST_API_NO_SLOT ||
        slot->handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return;
    }
    hash_slot = SparkGlm52RequestApiHashHandle(slot->handle);
    slot->handle_hash_next = api->slot_handle_hash_heads[hash_slot];
    api->slot_handle_hash_heads[hash_slot] = slot_index;
}

static void SparkGlm52RequestApiRemoveSlotHash(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot)
{
    uint32_t slot_index;
    uint32_t hash_slot;
    uint32_t current_slot;
    uint32_t previous_slot;

    slot_index = SparkGlm52RequestApiSlotIndex(api, slot);
    if (slot_index == SPARK_GLM52_REQUEST_API_NO_SLOT ||
        slot->handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return;
    }
    hash_slot = SparkGlm52RequestApiHashHandle(slot->handle);
    current_slot = api->slot_handle_hash_heads[hash_slot];
    previous_slot = SPARK_GLM52_REQUEST_API_NO_SLOT;
    while (current_slot != SPARK_GLM52_REQUEST_API_NO_SLOT)
    {
        if (current_slot == slot_index)
        {
            if (previous_slot == SPARK_GLM52_REQUEST_API_NO_SLOT)
            {
                api->slot_handle_hash_heads[hash_slot] =
                    api->request_slots[current_slot].handle_hash_next;
            }
            else
            {
                api->request_slots[previous_slot].handle_hash_next =
                    api->request_slots[current_slot].handle_hash_next;
            }
            api->request_slots[current_slot].handle_hash_next =
                SPARK_GLM52_REQUEST_API_NO_SLOT;
            return;
        }
        previous_slot = current_slot;
        current_slot = api->request_slots[current_slot].handle_hash_next;
    }
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
    uint32_t hash_index;
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
    api->decode_execution_row_capacity =
        SparkGlm52RequestApiNormalizeDecodeExecutionRowCapacity(configuration);
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
    api->dspark_speculator = configuration->dspark_speculator;

    for (hash_index = 0u;
         hash_index < SPARK_GLM52_REQUEST_API_SLOT_HASH_SLOTS;
         ++hash_index)
    {
        api->slot_handle_hash_heads[hash_index] =
            SPARK_GLM52_REQUEST_API_NO_SLOT;
    }
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiInitializeSlot(&api->request_slots[slot_index]);
        api->request_slots[slot_index].free_slot_next =
            slot_index + 1u < api->request_capacity
                ? slot_index + 1u
                : SPARK_GLM52_REQUEST_API_NO_SLOT;
    }
    api->free_slot_head = 0u;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiSlotIsReadyForDispatch(
    const SparkGlm52RequestApiSlot *slot)
{
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        return 1u;
    }
    if ((slot->state == SPARK_GLM52_REQUEST_API_STATE_READY_DECODE ||
         slot->state == SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u))
    {
        return 1u;
    }
    return 0u;
}

static uint32_t SparkGlm52RequestApiSlotIsActive(
    const SparkGlm52RequestApiSlot *slot)
{
    return slot->state != SPARK_GLM52_REQUEST_API_STATE_FREE &&
        slot->state != SPARK_GLM52_REQUEST_API_STATE_COMPLETED &&
        slot->state != SPARK_GLM52_REQUEST_API_STATE_CANCELLED;
}

uint32_t SparkGlm52RequestApiCurrentPipelineBatchWidth(
    const SparkGlm52RequestApi *api)
{
    uint32_t highest_ready_priority;
    uint32_t ready_request_count;
    uint32_t slot_index;

    if (api == 0 || api->scheduler == 0 || api->decode_batch_target == 0u)
    {
        return 0u;
    }
    if (!SparkGlm52RequestApiAdaptivePipelineBatchingIsEnabled(api))
    {
        return api->decode_batch_target;
    }
    highest_ready_priority = 0u;
    ready_request_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        const SparkGlm52RequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (!SparkGlm52RequestApiSlotIsReadyForDispatch(slot))
        {
            continue;
        }
        if (slot->priority > highest_ready_priority)
        {
            highest_ready_priority = slot->priority;
        }
    }
    if (highest_ready_priority == 0u)
    {
        return 0u;
    }
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        const SparkGlm52RequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (SparkGlm52RequestApiSlotIsActive(slot) &&
            slot->priority == highest_ready_priority)
        {
            ready_request_count += 1u;
        }
    }
    return SparkGlm52SchedulerSelectPipelineBatchWidth(
        api->scheduler,
        ready_request_count,
        api->decode_batch_target);
}

static uint32_t SparkGlm52RequestApiSlotHasRealtimePriority(
    const SparkGlm52RequestApiSlot *slot);

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindFreeSlot(
    SparkGlm52RequestApi *api)
{
    uint32_t slot_index;
    SparkGlm52RequestApiSlot *slot;

    if (api == 0 || api->free_slot_head == SPARK_GLM52_REQUEST_API_NO_SLOT)
    {
        return 0;
    }
    slot_index = api->free_slot_head;
    if (slot_index >= api->request_capacity)
    {
        return 0;
    }
    slot = &api->request_slots[slot_index];
    if (slot->state != SPARK_GLM52_REQUEST_API_STATE_FREE)
    {
        return 0;
    }
    api->free_slot_head = slot->free_slot_next;
    slot->free_slot_next = SPARK_GLM52_REQUEST_API_NO_SLOT;
    return slot;
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindSlotByHandle(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiHandle handle)
{
    uint32_t hash_slot;
    uint32_t slot_index;

    if (handle == SPARK_GLM52_REQUEST_API_INVALID_HANDLE)
    {
        return 0;
    }
    hash_slot = SparkGlm52RequestApiHashHandle(handle);
    slot_index = api->slot_handle_hash_heads[hash_slot];
    while (slot_index != SPARK_GLM52_REQUEST_API_NO_SLOT &&
           slot_index < api->request_capacity)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = &api->request_slots[slot_index];
        if (slot->state != SPARK_GLM52_REQUEST_API_STATE_FREE &&
            slot->handle == handle)
        {
            return slot;
        }
        slot_index = slot->handle_hash_next;
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
    slot->mtp_next_draft_token_budget =
        SPARK_GLM52_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT;
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
    SparkGlm52RequestApiInsertSlotHash(api, slot);

    api->queued_request_count += 1u;
    api->submitted_request_count += 1u;
    *handle_out = slot->handle;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiSlotIsSchedulablePrefill(
    const SparkGlm52RequestApiSlot *slot)
{
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
        return 1u;
    return slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL &&
        slot->dispatched_prompt_token_count < slot->prompt_token_count &&
        slot->inflight_prefill_dispatch_count <
            SPARK_GLM52_REQUEST_API_PREFILL_INFLIGHT_WAVE_LIMIT;
}

static uint32_t SparkGlm52RequestApiSlotIsSchedulableDecode(
    const SparkGlm52RequestApiSlot *slot)
{
    return (slot->state == SPARK_GLM52_REQUEST_API_STATE_READY_DECODE ||
            (slot->state ==
                SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
             slot->mtp_draft_token_count != 0u)) &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u);
}

static uint32_t SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify(
    const SparkGlm52RequestApiSlot *slot)
{
    return slot->state ==
        SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
        (slot->remaining_thinking_token_budget != 0u ||
         slot->remaining_output_token_budget != 0u);
}

static uint32_t SparkGlm52RequestApiSlotCanUseDspark(
    const SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    if (!SparkGlm52RequestApiDsparkSpeculationIsEnabled(api) || slot == 0)
    {
        return 0u;
    }
    if ((slot->flags &
            SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION) != 0u)
    {
        return 0u;
    }
    if ((api->dspark_speculator->policy_flags &
            SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_REALTIME) != 0u &&
        SparkGlm52RequestApiSlotHasRealtimePriority(slot))
    {
        return 1u;
    }
    if ((api->dspark_speculator->policy_flags &
            SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE) != 0u)
    {
        return 1u;
    }
    return 0u;
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

static uint32_t SparkGlm52RequestApiSlotsHaveSameSchedulingPriority(
    const SparkGlm52RequestApiSlot *left,
    const SparkGlm52RequestApiSlot *right)
{
    return left != 0 && right != 0 && left->priority == right->priority;
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
    return SparkGlm52StagePlanSelectBatchBucketValue(active_sequence_count);
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

    if (candidate->highest_priority != current->highest_priority)
    {
        return candidate->highest_priority > current->highest_priority;
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
    return candidate->earliest_submission_order <
        current->earliest_submission_order;
}

static SparkGlm52RequestApiPrefixFamilyGroup *
SparkGlm52RequestApiFindPrefixFamilyGroup(
    SparkGlm52RequestApiPrefixFamilyGroup *groups,
    uint32_t group_count,
    uint64_t prefix_hash,
    uint32_t shared_prefix_token_count,
    uint32_t priority)
{
    uint32_t group_index;

    for (group_index = 0u; group_index < group_count; ++group_index)
    {
        if (groups[group_index].valid != 0u &&
            groups[group_index].prefix_hash == prefix_hash &&
            groups[group_index].shared_prefix_token_count ==
                shared_prefix_token_count &&
            groups[group_index].highest_priority == priority)
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
    uint32_t shared_prefix_token_count,
    uint32_t priority)
{
    SparkGlm52RequestApiPrefixFamilyGroup *group;

    group = SparkGlm52RequestApiFindPrefixFamilyGroup(
        groups,
        *group_count,
        prefix_hash,
        shared_prefix_token_count,
        priority);
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
    group->highest_priority = priority;
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

static SparkStatus SparkGlm52RequestApiExtendPrefixScanHash(
    SparkGlm52RequestApiSlot *slot,
    uint32_t block_token_count,
    uint32_t needed_token_count,
    uint64_t *prefix_hash_out)
{
    SparkGlm52PrefixCachePromptHash block_hash;
    uint32_t hashed_token_count;
    uint64_t hash_value;
    SparkStatus status;

    hashed_token_count = slot->prefix_scan_hashed_token_count;
    hash_value = slot->prefix_scan_hash;
    if (hashed_token_count == 0u ||
        hashed_token_count > needed_token_count ||
        (hashed_token_count % block_token_count) != 0u)
    {
        hashed_token_count = 0u;
        hash_value = SPARK_GLM52_PREFIX_CACHE_EMPTY_PARENT_HASH;
    }
    while (hashed_token_count < needed_token_count)
    {
        status = SparkGlm52PrefixCacheHashPromptTokens(
            block_token_count,
            hash_value,
            &slot->prompt_token_ids[hashed_token_count],
            block_token_count,
            &block_hash);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        hash_value = block_hash.prompt_hash;
        hashed_token_count += block_token_count;
    }
    slot->prefix_scan_hashed_token_count = hashed_token_count;
    slot->prefix_scan_hash = hash_value;
    *prefix_hash_out = hash_value;
    return SPARK_STATUS_OK;
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
            uint64_t prefix_hash_value;
            SparkGlm52RequestApiPrefixFamilyGroup *group;

            if (SparkGlm52RequestApiExtendPrefixScanHash(
                    slot,
                    block_token_count,
                    prefix_token_count,
                    &prefix_hash_value) != SPARK_STATUS_OK)
            {
                break;
            }
            group = SparkGlm52RequestApiAcquirePrefixFamilyGroup(
                groups,
                &group_count,
                prefix_hash_value,
                prefix_token_count,
                slot->priority);
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
    if (choice->leader_slot->priority != slot->priority)
    {
        return choice->leader_slot->priority > slot->priority;
    }
    if (choice->realtime_priority !=
        SparkGlm52RequestApiSlotHasRealtimePriority(slot))
    {
        return choice->realtime_priority != 0u;
    }
    return choice->saved_prompt_token_count != 0u;
}

static uint32_t SparkGlm52RequestApiEvaluatePrefillBatchShape(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t require_resident_cached_blocks,
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
    if (require_resident_cached_blocks != 0u &&
        !SparkGlm52RequestApiPrefillCachedBlocksAreResident(
            api,
            slot,
            slot->prompt_token_count))
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
        batch_target = SparkGlm52RequestApiCurrentPipelineBatchWidth(api);
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

    if (candidate->slot->priority != current->slot->priority)
    {
        return candidate->slot->priority > current->slot->priority;
    }
    if (candidate->realtime_priority != current->realtime_priority)
    {
        return candidate->realtime_priority > current->realtime_priority;
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
    SparkGlm52RequestApi *api,
    uint32_t require_resident_cached_blocks)
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
                require_resident_cached_blocks,
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

// The decode and speculative-verify searches differed in one predicate and
// nothing else, so the predicate is the parameter.
static SparkGlm52RequestApiSlot *SparkGlm52RequestApiFindBestSchedulableSlot(
    SparkGlm52RequestApi *api,
    uint32_t (*is_schedulable)(const SparkGlm52RequestApiSlot *),
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
        if (!is_schedulable(slot) ||
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

static void SparkGlm52RequestApiInsertBatchMemberByPriority(
    SparkGlm52RequestApiSlot **selected_slots,
    uint32_t *selected_count,
    uint32_t selected_capacity,
    SparkGlm52RequestApiSlot *slot)
{
    uint32_t insert_index;
    uint32_t shift_index;

    if (*selected_count >= selected_capacity)
    {
        if (!SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
                slot,
                selected_slots[selected_capacity - 1u]))
        {
            return;
        }
        *selected_count = selected_capacity - 1u;
    }
    insert_index = *selected_count;
    while (insert_index > 1u &&
           SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
               slot,
               selected_slots[insert_index - 1u]))
    {
        insert_index -= 1u;
    }
    for (shift_index = *selected_count;
         shift_index > insert_index;
         --shift_index)
    {
        selected_slots[shift_index] = selected_slots[shift_index - 1u];
    }
    selected_slots[insert_index] = slot;
    *selected_count += 1u;
}

static uint32_t SparkGlm52RequestApiCollectDecodeBatchMembers(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *leader_slot,
    uint32_t require_resident_kv,
    SparkGlm52RequestApiSlot **selected_slots,
    uint32_t selected_capacity)
{
    uint32_t selected_count;
    uint32_t slot_index;

    selected_slots[0] = leader_slot;
    selected_count = 1u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;

		slot = &api->request_slots[slot_index];
		if (slot == leader_slot ||
			!SparkGlm52RequestApiSlotIsSchedulableDecode(slot) ||
			SparkGlm52RequestApiSlotCanUseDspark(api,slot) !=
				SparkGlm52RequestApiSlotCanUseDspark(api,leader_slot))
		{
			continue;
		}
        if ((require_resident_kv != 0u ||
             slot->priority < leader_slot->priority) &&
            !SparkGlm52RequestApiDecodeBlocksAreResident(api, slot))
        {
            continue;
        }
        SparkGlm52RequestApiInsertBatchMemberByPriority(
            selected_slots,
            &selected_count,
            selected_capacity,
            slot);
    }
    return selected_count;
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

    if (SparkGlm52RequestApiCrossSequencePrefixReuseIsEnabled(api) == 0u ||
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

static SparkStatus SparkGlm52RequestApiPendingSpeculativeTokenCount(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint32_t *speculative_token_count_out)
{
    SparkGlm52DsparkDraftResult draft_result;
    SparkStatus status;

    if (api == 0 || slot == 0 || speculative_token_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *speculative_token_count_out = 0u;
    if (!SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify(slot) &&
        slot->state !=
            SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
    {
        return SPARK_STATUS_OK;
    }
    if (slot->mtp_draft_token_count != 0u)
    {
        if (slot->mtp_draft_token_count >
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        *speculative_token_count_out = slot->mtp_draft_token_count;
        return SPARK_STATUS_OK;
    }
    if (!SparkGlm52RequestApiDsparkSpeculationIsEnabled(api))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkGlm52DsparkGetDraft(
        api->dspark_speculator,
        slot->sequence_id,
        &draft_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (draft_result.token_count == 0u ||
        draft_result.token_count >
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *speculative_token_count_out = draft_result.token_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiRequiredDecodeKvTokenCount(
    const SparkGlm52RequestApiSlot *slot,
    uint32_t speculative_token_count,
    uint32_t *required_token_count_out)
{
    uint64_t required_token_count;

    if (slot == 0 || required_token_count_out == 0 ||
        slot->computed_prompt_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    required_token_count =
        (uint64_t)slot->computed_prompt_token_count +
        (uint64_t)slot->completed_decode_token_count + 1u +
        (uint64_t)speculative_token_count;
    if (required_token_count > UINT32_MAX ||
        required_token_count > SPARK_GLM52_SCHEDULER_MAX_CONTEXT_TOKENS)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *required_token_count_out = (uint32_t)required_token_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiApplyActiveKvBlockBudget(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot **selected_slots,
    uint32_t *selected_count,
    uint32_t additional_token_count)
{
    uint64_t selected_block_count;
    uint32_t block_token_count;
    uint32_t input_index;
    uint32_t output_count;

    if (api == 0 || selected_slots == 0 || selected_count == 0 ||
        *selected_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (api->max_resident_kv_block_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (api->scheduler == 0 || api->scheduler->prefix_cache == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    selected_block_count = 0u;
    output_count = 0u;
    for (input_index = 0u; input_index < *selected_count; ++input_index)
    {
        uint64_t required_block_count;
        uint32_t required_token_count;
        SparkStatus status;

        status = SparkGlm52RequestApiRequiredDecodeKvTokenCount(
            selected_slots[input_index],
            additional_token_count,
            &required_token_count);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        required_block_count =
            ((uint64_t)required_token_count + block_token_count - 1u) /
            block_token_count;
        if (required_block_count > api->max_resident_kv_block_count)
        {
            if (input_index == 0u)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            continue;
        }
        if (selected_block_count >
            api->max_resident_kv_block_count - required_block_count)
        {
            continue;
        }
        selected_slots[output_count++] = selected_slots[input_index];
        selected_block_count += required_block_count;
    }
    if (output_count == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *selected_count = output_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiEnsureDecodeSlotKvCapacity(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t speculative_token_count,
    uint32_t *required_token_count_out)
{
    uint32_t required_token_count;
    SparkStatus status;

    if (api == 0 || api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 || slot == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52RequestApiRequiredDecodeKvTokenCount(
        slot,
        speculative_token_count,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52PrefixCacheEnsureSequenceTokenCapacity(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count);
    if (status == SPARK_STATUS_OK && required_token_count_out != 0)
    {
        *required_token_count_out = required_token_count;
    }
    return status;
}

static SparkStatus SparkGlm52RequestApiEnsurePendingDecodeSlotKvCapacity(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t *required_token_count_out)
{
    uint32_t speculative_token_count;
    SparkStatus status;

    status = SparkGlm52RequestApiPendingSpeculativeTokenCount(
        api,
        slot,
        &speculative_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52RequestApiEnsureDecodeSlotKvCapacity(
        api,
        slot,
        speculative_token_count,
        required_token_count_out);
}

static uint32_t SparkGlm52RequestApiDecodeBlocksAreResident(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    uint32_t required_token_count;
    uint32_t physical_block_count;
    uint32_t resident_block_count;
    uint32_t nonresident_block_count;
    SparkStatus status;

    if (api->scheduler == 0 ||
        api->scheduler->prefix_cache == 0 ||
        slot == 0 ||
        slot->computed_prompt_token_count == 0u)
    {
        return 1u;
    }

    status = SparkGlm52RequestApiEnsurePendingDecodeSlotKvCapacity(
        api,
        (SparkGlm52RequestApiSlot *)slot,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    if (!SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return 1u;
    }

    status = SparkGlm52PrefixCacheProbeSequenceResidency(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count,
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
    uint32_t required_token_count;
    uint32_t slot_physical_block_count;
    uint32_t slot_physical_block_indices[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52RequestApiEnsurePendingDecodeSlotKvCapacity(
        api,
        slot,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52PrefixCacheBuildPhysicalBlockTable(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count,
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
    uint32_t required_token_count;
    SparkGlm52KvCachePrefetchSourceBlock slot_source_blocks[
        SPARK_GLM52_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT];
    uint32_t slot_source_block_count;
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52RequestApiEnsurePendingDecodeSlotKvCapacity(
        api,
        slot,
        &required_token_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52PrefixCacheBuildSequencePrefetchSources(
        api->scheduler->prefix_cache,
        slot->sequence_id,
        required_token_count,
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
            !SparkGlm52RequestApiSlotIsSchedulableDecode(slot) &&
            !SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify(slot))
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
         SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify(slot) ||
         slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE ||
         slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY ||
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

    if (SparkGlm52RequestApiCrossSequencePrefixReuseIsEnabled(api) == 0u ||
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
        if (prefetch_plan->prefetch_block_count == 0u ||
            SparkGlm52RequestApiPrefetchPlanIsResident(api, prefetch_plan))
        {
            return SparkGlm52RequestApiApplyJitKvResidencyPolicy(
                api,
                prefetch_plan,
                additional_protected_physical_block_indices,
                additional_protected_physical_block_count);
        }
        status = SparkGlm52RequestApiPollPendingJitKvPrefetches(api);
        if (status != SPARK_STATUS_OK)
        {
            return status;
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
        else if (SparkGlm52RequestApiSlotIsSchedulableDecode(slot) ||
                 SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify(slot))
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
    if (max_prefill_tokens_per_step == 0u ||
        max_prefill_tokens_per_step > api->scheduler->max_prefill_tokens_per_step)
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
    const SparkGlm52RequestApi *api,
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
    scheduler_request->computed_prompt_token_count =
        SparkGlm52RequestApiCrossSequencePrefixReuseIsEnabled(api) != 0u
            ? 0u : SparkGlm52RequestApiMaximumU32(
                slot->computed_prompt_token_count,
                slot->dispatched_prompt_token_count);
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
            candidate->state != SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL ||
            !SparkGlm52RequestApiSlotsHaveSameSchedulingPriority(
                candidate,
                leader_slot))
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

static uint32_t SparkGlm52RequestApiSlotResidentKvBlockCount(
    const SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot)
{
    uint64_t token_count;
    uint32_t block_token_count;

    if (api == 0 || api->scheduler == 0 || slot == 0 ||
        slot->state == SPARK_GLM52_REQUEST_API_STATE_FREE ||
        slot->state == SPARK_GLM52_REQUEST_API_STATE_COMPLETED ||
        slot->state == SPARK_GLM52_REQUEST_API_STATE_CANCELLED ||
        (slot->computed_prompt_token_count == 0u &&
         (slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL ||
          slot->state == SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT)))
    {
        return 0u;
    }
    block_token_count = api->scheduler->prefix_cache_block_tokens;
    if (block_token_count == 0u)
    {
        return 0u;
    }
    token_count = slot->prompt_token_count;
    token_count += slot->scheduled_decode_token_count;
    if (slot->state ==
        SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
    {
        token_count += 1u;
    }
    return token_count > UINT32_MAX
        ? UINT32_MAX
        : (uint32_t)((token_count + block_token_count - 1u) /
            block_token_count);
}

static uint64_t SparkGlm52RequestApiResidentKvBlockCount(
    const SparkGlm52RequestApi *api)
{
    uint64_t block_count;
    uint32_t slot_index;

    block_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        block_count += SparkGlm52RequestApiSlotResidentKvBlockCount(
            api,
            &api->request_slots[slot_index]);
    }
    return block_count;
}

static uint32_t SparkGlm52RequestApiReservePrefillResidentKvBlocks(
    const SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint64_t *reserved_block_count)
{
    uint64_t additional_block_count;
    uint32_t current_block_count;
    uint32_t required_block_count;

    if (api->max_resident_kv_block_count == 0u ||
        SparkGlm52RequestApiJitPrefetchIsEnabled(api))
    {
        return 1u;
    }
    current_block_count = SparkGlm52RequestApiSlotResidentKvBlockCount(
        api,
        slot);
    required_block_count = SparkGlm52RequestApiPrefillBlockCountForScheduledTokens(
        api,
        slot->prompt_token_count);
    additional_block_count = required_block_count > current_block_count
        ? required_block_count - current_block_count
        : 0u;
    if (*reserved_block_count > api->max_resident_kv_block_count ||
        additional_block_count >
            api->max_resident_kv_block_count - *reserved_block_count)
    {
        return 0u;
    }
    *reserved_block_count += additional_block_count;
    return 1u;
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
    uint64_t reserved_block_count;
    SparkStatus status;

    reserved_block_count = SparkGlm52RequestApiResidentKvBlockCount(api);
    if (!SparkGlm52RequestApiReservePrefillResidentKvBlocks(
            api,
            slot,
            &reserved_block_count))
    {
        return SPARK_STATUS_BUSY;
    }

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
        api,
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

    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL;
        api->queued_request_count -= 1u;
    }
    if (slot->inflight_prefill_dispatch_count == 0u &&
        slot->dispatched_prompt_token_count <
            slot->computed_prompt_token_count)
        slot->dispatched_prompt_token_count =
            slot->computed_prompt_token_count;
    if (committed_prefix_token_count > slot->dispatched_prompt_token_count)
        slot->dispatched_prompt_token_count = committed_prefix_token_count;
    slot->inflight_prefill_dispatch_count += 1u;
    slot->scheduled_prefill_step_count += 1u;
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
    dispatch->request_slot_indices[0] = SparkGlm52RequestApiSlotIndex(api, slot);
    dispatch->request_ids[0] = slot->request_id;
    dispatch->sequence_ids[0] = slot->sequence_id;
    if (SparkGlm52RequestApiSlotCanUseDspark(api, slot))
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
        api->dspark_tap_capture_dispatch_count += 1u;
    }

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
			SparkGlm52RequestApiSlotCanUseDspark(api,candidate) !=
				SparkGlm52RequestApiSlotCanUseDspark(api,slot) ||
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
        dispatch->request_slot_indices[dispatch->request_count] =
            SparkGlm52RequestApiSlotIndex(api, candidate);
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
		!SparkGlm52RequestApiSlotIsSchedulablePrefill(candidate_slot) ||
		SparkGlm52RequestApiSlotCanUseDspark(api,candidate_slot) !=
			SparkGlm52RequestApiSlotCanUseDspark(api,leader_slot))
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
    uint64_t reserved_block_count;
    uint32_t selected_scheduled_prompt_token_counts[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkStatus status;

    if (!SparkGlm52RequestApiPrefillBatchingIsEnabled(api))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (first_slot->state != SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
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

    batch_target = SparkGlm52RequestApiCurrentPipelineBatchWidth(api);
    if (batch_target > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        batch_target = SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    require_resident_batch_members =
        SparkGlm52RequestApiSlotHasRealtimePriority(first_slot) ||
        SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(api);
    reserved_block_count = SparkGlm52RequestApiResidentKvBlockCount(api);
    request_count = 0u;
    slot = first_slot;
    while (slot != 0 && request_count < batch_target)
    {
        if (!SparkGlm52RequestApiReservePrefillResidentKvBlocks(
                api,
                slot,
                &reserved_block_count))
        {
            if (request_count == 0u)
            {
                return SPARK_STATUS_BUSY;
            }
            break;
        }
        if (request_count == 0u)
        {
            selected_scheduled_prompt_token_counts[request_count] =
                leader_scheduled_prompt_token_count;
        }
        selected_slots[request_count] = slot;
        selected_handles[request_count] = slot->handle;
        SparkGlm52RequestApiFillPrefillSchedulerRequest(
            api,
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
        if (selected_slot->inflight_prefill_dispatch_count == 0u &&
            selected_slot->dispatched_prompt_token_count <
                selected_slot->computed_prompt_token_count)
            selected_slot->dispatched_prompt_token_count =
                selected_slot->computed_prompt_token_count;
        {
            uint32_t lane_committed;

            lane_committed = dispatch->prefill_batch_decision.lanes[
                request_index].cache_commit_token_count_after_step;
            if (lane_committed > selected_slot->prompt_token_count)
                lane_committed = selected_slot->prompt_token_count;
            if (lane_committed > selected_slot->dispatched_prompt_token_count)
                selected_slot->dispatched_prompt_token_count = lane_committed;
        }
        selected_slot->inflight_prefill_dispatch_count += 1u;
        api->queued_request_count -= 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_slot_indices[request_index] =
            SparkGlm52RequestApiSlotIndex(api, selected_slot);
        dispatch->request_ids[request_index] = selected_slot->request_id;
		dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
	}
	if (SparkGlm52RequestApiSlotCanUseDspark(api,first_slot))
	{
		dispatch->flags |=
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
		api->dspark_tap_capture_dispatch_count += 1u;
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


static SparkStatus SparkGlm52RequestApiGetSlotDsparkDraft(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    SparkGlm52DsparkDraftResult *draft_result)
{
    if (!SparkGlm52RequestApiDsparkSpeculationIsEnabled(api) ||
        slot == 0 || draft_result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52DsparkGetDraft(
        api->dspark_speculator,
        slot->sequence_id,
        draft_result);
}

#define SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_DSPARK 1u
#define SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP 2u

static SparkStatus SparkGlm52RequestApiGetSlotMtpDraft(
    const SparkGlm52RequestApiSlot *slot,
    SparkGlm52DsparkDraftResult *draft_result)
{
    uint32_t token_index;

    if (slot == 0 || draft_result == 0 ||
        slot->mtp_draft_token_count == 0u ||
        slot->mtp_draft_token_count >
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(draft_result, 0, sizeof(*draft_result));
    draft_result->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    draft_result->descriptor_bytes =
        SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES;
    draft_result->token_count = slot->mtp_draft_token_count;
    for (token_index = 0u;
         token_index < slot->mtp_draft_token_count;
         ++token_index)
    {
        draft_result->token_ids[token_index] =
            slot->mtp_draft_token_ids[token_index];
        draft_result->confidence_milli[token_index] =
            SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiGetSlotSpeculativeDraft(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiSlot *slot,
    uint32_t preferred_source,
    SparkGlm52DsparkDraftResult *draft_result,
    uint32_t *source_out)
{
    SparkStatus status;

    if (draft_result == 0 || source_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *source_out = 0u;

    if (preferred_source != SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_DSPARK)
    {
        status = SparkGlm52RequestApiGetSlotMtpDraft(slot, draft_result);
        if (status == SPARK_STATUS_OK)
        {
            *source_out = SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP;
            return SPARK_STATUS_OK;
        }
        if (preferred_source == SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP)
        {
            return status;
        }
        status = SparkGlm52RequestApiGetSlotDsparkDraft(api, slot, draft_result);
        if (status == SPARK_STATUS_OK)
        {
            *source_out = SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_DSPARK;
            return SPARK_STATUS_OK;
        }
        return status;
    }

    status = SparkGlm52RequestApiGetSlotDsparkDraft(api, slot, draft_result);
    if (status == SPARK_STATUS_OK)
    {
        *source_out = SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_DSPARK;
        return SPARK_STATUS_OK;
    }
    status = SparkGlm52RequestApiGetSlotMtpDraft(slot, draft_result);
    if (status == SPARK_STATUS_OK)
    {
        *source_out = SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP;
        return SPARK_STATUS_OK;
    }
    return status;
}

static uint32_t SparkGlm52RequestApiCollectSpeculativeVerifyBatchMembers(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *leader_slot,
    uint32_t leader_token_count,
    uint32_t leader_source,
    uint32_t require_resident_kv,
    SparkGlm52RequestApiSlot **selected_slots,
    uint32_t selected_capacity)
{
    uint32_t selected_count;
    uint32_t slot_index;

    selected_slots[0] = leader_slot;
    selected_count = 1u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot;
        SparkGlm52DsparkDraftResult draft_result;
        uint32_t draft_source;

		slot = &api->request_slots[slot_index];
		if (slot == leader_slot ||
			!SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify(slot) ||
			SparkGlm52RequestApiSlotCanUseDspark(api,slot) !=
				SparkGlm52RequestApiSlotCanUseDspark(api,leader_slot) ||
			((require_resident_kv != 0u ||
              slot->priority < leader_slot->priority) &&
             !SparkGlm52RequestApiDecodeBlocksAreResident(api, slot)))
        {
            continue;
        }
        if (SparkGlm52RequestApiGetSlotSpeculativeDraft(
                api,
                slot,
                leader_source,
                &draft_result,
                &draft_source) != SPARK_STATUS_OK ||
            draft_source != leader_source ||
            draft_result.token_count != leader_token_count)
        {
            continue;
        }
        SparkGlm52RequestApiInsertBatchMemberByPriority(
            selected_slots,
            &selected_count,
            selected_capacity,
            slot);
    }
    return selected_count;
}

static void SparkGlm52RequestApiFillSpeculativeVerifySchedulerRequest(
    SparkGlm52SchedulerRequest *scheduler_request)
{
    memset(scheduler_request, 0, sizeof(*scheduler_request));
    scheduler_request->abi_version = SPARK_GLM52_SCHEDULER_ABI_VERSION;
    scheduler_request->descriptor_bytes =
        SPARK_GLM52_SCHEDULER_REQUEST_DESCRIPTOR_BYTES;
    scheduler_request->active_sequence_count = 1u;
    scheduler_request->flags = SPARK_GLM52_SCHEDULER_REQUEST_FLAG_DECODE;
}

static SparkStatus SparkGlm52RequestApiAdmitDecodeBatchMembers(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot **selected_slots,
    SparkGlm52SchedulerRequest *scheduler_requests,
    uint32_t *request_count_io,
    uint32_t context_extension,
    void (*fill_scheduler_request)(SparkGlm52SchedulerRequest *),
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52SchedulerBatchRequest batch_request;
    uint32_t request_count;
    uint32_t request_index;
    SparkStatus status;

    request_count = *request_count_io;
    status = SparkGlm52RequestApiApplyActiveKvBlockBudget(
        api,
        selected_slots,
        &request_count,
        context_extension);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        fill_scheduler_request(
            &scheduler_requests[request_index]);
    }
    if (request_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        status = SparkGlm52RequestApiEnsureDecodeSlotKvCapacity(
            api,
            selected_slots[request_index],
            context_extension,
            0);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
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
    *request_count_io = request_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiScheduleSpeculativeVerifyBatch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *first_slot,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52SchedulerRequest scheduler_requests[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52RequestApiSlot *selected_slots[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52DsparkDraftResult leader_draft;
    uint32_t leader_source;
    uint32_t batch_target;
    uint32_t request_count;
    uint32_t request_index;
    uint32_t token_index;
    uint32_t require_resident_batch_members;
    uint32_t speculative_context_extension;
    SparkStatus status;

    if (first_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkGlm52RequestApiGetSlotSpeculativeDraft(
        api,
        first_slot,
        (api->configuration_flags &
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION) != 0u &&
            SparkGlm52RequestApiDsparkSpeculationIsEnabled(api)
            ? SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_DSPARK
            : 0u,
        &leader_draft,
        &leader_source);
    if (status != SPARK_STATUS_OK || leader_draft.token_count == 0u)
    {
        return status == SPARK_STATUS_NOT_FOUND ? SPARK_STATUS_NOT_FOUND : status;
    }

    batch_target = SparkGlm52RequestApiDecodeBatchingIsEnabled(api)
        ? SparkGlm52RequestApiCurrentPipelineBatchWidth(api)
        : 1u;
    if (batch_target > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        batch_target = SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT;
    }
    {
        uint32_t row_limited_batch_target;
        uint32_t verifier_row_count;
        verifier_row_count =
            leader_source == SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP
            ? SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT
            : leader_draft.token_count + 1u;
        row_limited_batch_target = api->decode_execution_row_capacity /
            verifier_row_count;
        if (row_limited_batch_target == 0u)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        if (batch_target > row_limited_batch_target)
        {
            batch_target = row_limited_batch_target;
        }
    }
    require_resident_batch_members =
        SparkGlm52RequestApiSlotHasRealtimePriority(first_slot) ||
        !SparkGlm52RequestApiJitPrefetchIsEnabled(api) ||
        SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(api);
    request_count = SparkGlm52RequestApiCollectSpeculativeVerifyBatchMembers(
        api,
        first_slot,
        leader_draft.token_count,
        leader_source,
        require_resident_batch_members,
        selected_slots,
        batch_target);
    speculative_context_extension =
        leader_source == SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP
            ? SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION
            : leader_draft.token_count;
    status = SparkGlm52RequestApiAdmitDecodeBatchMembers(
        api,
        selected_slots,
        scheduler_requests,
        &request_count,
        speculative_context_extension,
        SparkGlm52RequestApiFillSpeculativeVerifySchedulerRequest,
        dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch->decode_batch_decision.accepted == 0u)
    {
        return SPARK_STATUS_OK;
    }

    dispatch->accepted = 1u;
    dispatch->kind =
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    if (leader_source == SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP)
    {
        if (leader_draft.token_count !=
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
        {
            return SPARK_STATUS_MODULE_NOT_VALIDATED;
        }
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY;
        dispatch->mtp_draft_token_budget =
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
        dispatch->speculative_verifier_token_count =
            SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
        dispatch->speculative_max_committed_token_count =
            SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT;
    }
    else
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY |
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
        dispatch->speculative_verifier_token_count =
            leader_draft.token_count + 1u;
        dispatch->speculative_max_committed_token_count =
            leader_draft.token_count + 1u;
    }
    if (leader_source == SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP &&
        SparkGlm52RequestApiSlotCanUseDspark(api,first_slot))
    {
        dispatch->flags |=
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
    }
    if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
        api->dspark_tap_capture_dispatch_count += 1u;
    dispatch->request_count =
        dispatch->decode_batch_decision.packed_request_count;
    dispatch->highest_priority = first_slot->priority;
    dispatch->speculative_token_count = leader_draft.token_count;

    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *selected_slot;
        SparkGlm52DsparkDraftResult draft_result;
        uint32_t draft_source;

        selected_slot = selected_slots[request_index];
        status = SparkGlm52RequestApiGetSlotSpeculativeDraft(
            api,
            selected_slot,
            leader_source,
            &draft_result,
            &draft_source);
        if (status != SPARK_STATUS_OK ||
            draft_source != leader_source ||
            draft_result.token_count != leader_draft.token_count)
        {
            return status == SPARK_STATUS_OK ?
                SPARK_STATUS_INVALID_ARGUMENT : status;
        }
        selected_slot->state =
            SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY;
        selected_slot->scheduled_decode_token_count +=
            draft_result.token_count;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_slot_indices[request_index] =
            SparkGlm52RequestApiSlotIndex(api, selected_slot);
        dispatch->request_ids[request_index] = selected_slot->request_id;
        dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
        for (token_index = 0u;
             token_index < draft_result.token_count;
             ++token_index)
        {
            dispatch->speculative_draft_token_ids[request_index][token_index] =
                draft_result.token_ids[token_index];
            dispatch->speculative_confidence_milli[request_index][token_index] =
                draft_result.confidence_milli[token_index];
        }
    }
    api->scheduled_decode_dispatch_count += 1u;
    if (leader_source == SPARK_GLM52_REQUEST_API_SPECULATIVE_SOURCE_MTP)
    {
        api->mtp_verify_dispatch_count += 1u;
    }
    else
    {
        api->dspark_verify_dispatch_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiSlotRemainingDecodeBudget(
    const SparkGlm52RequestApiSlot *slot)
{
    if (slot == 0)
    {
        return 0u;
    }
    return slot->remaining_thinking_token_budget +
        slot->remaining_output_token_budget;
}

static uint64_t SparkGlm52RequestApiMtpResolvedRequestCount(
    const SparkGlm52RequestApi *api,
    uint64_t *committed_token_count_out)
{
    uint64_t proposed_token_count;

    proposed_token_count =
        api->mtp_accepted_draft_token_count +
        api->mtp_rejected_token_count;
    *committed_token_count_out = api->mtp_committed_token_count;
    /* A partially-resolved in-flight cycle leaves a non-zero remainder;
     * floor the completed cycle count instead of discarding every
     * sample, which previously zeroed the utility estimate. */
    return proposed_token_count /
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
}

static uint64_t SparkGlm52RequestApiMtpExpectedCommittedTokensScaled(
    const SparkGlm52RequestApi *api)
{
    uint64_t committed_token_count;
    uint64_t resolved_request_count;
    uint64_t weighted_committed_token_count;
    uint64_t weighted_request_count;

    resolved_request_count = SparkGlm52RequestApiMtpResolvedRequestCount(
        api,
        &committed_token_count);
    weighted_committed_token_count =
        committed_token_count +
        ((uint64_t)SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_SAMPLE_COUNT *
         SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_COMMITTED_TOKEN_COUNT);
    weighted_request_count =
        resolved_request_count +
        SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_SAMPLE_COUNT;
    return weighted_committed_token_count *
        SPARK_GLM52_REQUEST_API_MTP_UTILITY_SCALE /
        weighted_request_count;
}

static uint32_t SparkGlm52RequestApiMtpOutranksPlainDecode(
    const SparkGlm52RequestApi *api,
    uint32_t plain_request_count,
    uint32_t mtp_request_count)
{
    uint64_t mtp_expected_tokens_scaled;
    uint64_t mtp_work_ns;
    uint64_t plain_work_ns;
    SparkStatus status;

    if (api == 0 || api->scheduler == 0 ||
        plain_request_count == 0u || mtp_request_count == 0u)
    {
        return 0u;
    }
    if ((api->configuration_flags &
            SPARK_GLM52_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE) != 0u)
    {
        return 1u;
    }
    status = SparkGlm52SchedulerEstimateDecodeWorkNs(
        api->scheduler,
        plain_request_count,
        1u,
        api->decode_execution_row_capacity,
        &plain_work_ns);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    status = SparkGlm52SchedulerEstimateDecodeWorkNs(
        api->scheduler,
        mtp_request_count,
        SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
        api->decode_execution_row_capacity,
        &mtp_work_ns);
    if (status != SPARK_STATUS_OK)
    {
        return 0u;
    }
    /* Compare plain decode against the full MTP cycle: the verify batch
     * plus the draft-chain dispatches that produced the candidates.
     * The measured-plan model cannot see the serialized draft work, so
     * scale the plain estimate by the calibrated chain multiplier. */
    if (plain_work_ns > UINT64_MAX /
            SPARK_GLM52_REQUEST_API_MTP_DRAFT_CHAIN_WORK_MULTIPLIER)
    {
        return 0u;
    }
    {
        uint64_t draft_chain_work_ns;
        uint64_t mtp_cycle_work_ns;

        draft_chain_work_ns =
            plain_work_ns *
            SPARK_GLM52_REQUEST_API_MTP_DRAFT_CHAIN_WORK_MULTIPLIER;
        if (mtp_work_ns > UINT64_MAX - draft_chain_work_ns)
        {
            return 0u;
        }
        mtp_cycle_work_ns = mtp_work_ns + draft_chain_work_ns;
        mtp_expected_tokens_scaled =
            SparkGlm52RequestApiMtpExpectedCommittedTokensScaled(api);
        return mtp_expected_tokens_scaled * mtp_request_count * plain_work_ns >
            SPARK_GLM52_REQUEST_API_MTP_UTILITY_MARGIN_SCALE *
            plain_request_count * mtp_cycle_work_ns;
    }
}

static uint32_t SparkGlm52RequestApiDecodeBatchMtpBudget(
    const SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *const *selected_slots,
    uint32_t request_count)
{
    uint32_t budget;
    uint32_t request_index;

    if (!SparkGlm52RequestApiMtpCommitIsEnabled(api) ||
        selected_slots == 0 || request_count == 0u ||
        !SparkGlm52RequestApiMtpOutranksPlainDecode(
            api,
            request_count,
            request_count))
    {
        return 0u;
    }
    budget = SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT;
    for (request_index = 0u; request_index < request_count; ++request_index)
    {
        const SparkGlm52RequestApiSlot *slot;
        uint32_t lane_budget;

        slot = selected_slots[request_index];
        if (slot == 0 ||
            (slot->flags &
                SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION) != 0u ||
            SparkGlm52RequestApiSlotRemainingDecodeBudget(slot) <
                SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT + 1u)
        {
            return 0u;
        }
        lane_budget = SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
        if (slot->mtp_next_draft_token_budget != lane_budget)
            return 0u;
        if (lane_budget < budget)
        {
            budget = lane_budget;
        }
    }
    return budget;
}

static void SparkGlm52RequestApiDiscardMtpDraft(
    SparkGlm52RequestApiSlot *slot)
{
    if (slot->state !=
            SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY ||
        slot->mtp_draft_token_count == 0u)
    {
        return;
    }
    memset(slot->mtp_draft_token_ids,0,sizeof(slot->mtp_draft_token_ids));
    slot->mtp_draft_token_count = 0u;
    slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
}

static SparkStatus SparkGlm52RequestApiScheduleDecodeBatch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *first_slot,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52SchedulerRequest scheduler_requests[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    SparkGlm52RequestApiSlot *selected_slots[
        SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t request_count;
    uint32_t request_index;
    uint32_t batch_target;
    uint32_t require_resident_batch_members;
    uint32_t batch_disables_speculation;
    uint32_t mtp_draft_token_budget;
    SparkStatus status;

    batch_disables_speculation = 0u;
    batch_target = SparkGlm52RequestApiDecodeBatchingIsEnabled(api)
        ? SparkGlm52RequestApiCurrentPipelineBatchWidth(api)
        : 1u;
    require_resident_batch_members =
        SparkGlm52RequestApiSlotHasRealtimePriority(first_slot) ||
        !SparkGlm52RequestApiJitPrefetchIsEnabled(api) ||
        SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(api);
    request_count = SparkGlm52RequestApiCollectDecodeBatchMembers(
        api,
        first_slot,
        require_resident_batch_members,
        selected_slots,
        batch_target);
    mtp_draft_token_budget = SparkGlm52RequestApiDecodeBatchMtpBudget(
        api,selected_slots,request_count);
    status = SparkGlm52RequestApiAdmitDecodeBatchMembers(
        api,
        selected_slots,
        scheduler_requests,
        &request_count,
        mtp_draft_token_budget,
        SparkGlm52RequestApiFillDecodeSchedulerRequest,
        dispatch);
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
        SparkGlm52RequestApiDiscardMtpDraft(selected_slot);
        selected_slot->state = SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE;
        selected_slot->scheduled_decode_token_count += 1u;
        api->running_request_count += 1u;
        dispatch->request_handles[request_index] = selected_slot->handle;
        dispatch->request_slot_indices[request_index] =
            SparkGlm52RequestApiSlotIndex(api, selected_slot);
        dispatch->request_ids[request_index] = selected_slot->request_id;
        dispatch->sequence_ids[request_index] = selected_slot->sequence_id;
        dispatch->decode_committed_token_counts[request_index] = 1u;
		if ((selected_slot->flags &
				SPARK_GLM52_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION) != 0u)
		{
			batch_disables_speculation = 1u;
		}
	}
	if (SparkGlm52RequestApiSlotCanUseDspark(api,first_slot))
		dispatch->flags |=
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
    if (mtp_draft_token_budget != 0u &&
        batch_disables_speculation == 0u)
    {
        dispatch->flags |= SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT;
        dispatch->mtp_draft_token_budget = mtp_draft_token_budget;
    }
    if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
    {
        api->dspark_tap_capture_dispatch_count += 1u;
    }
    api->scheduled_decode_dispatch_count += 1u;
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RequestApiMtpVerifyOutranksDecode(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *decode_slot,
    SparkGlm52RequestApiSlot *speculative_verify_slot)
{
    (void)api;
    (void)decode_slot;
    /* A pending draft is sunk cost: the draft-chain dispatches are
     * already paid, so verifying commits ~E tokens for the price of one
     * verify batch while discarding the draft to run plain decode pays
     * the same batch price for one token and throws the draft work
     * away. Draft utility is therefore gated only where new drafts are
     * budgeted (SparkGlm52RequestApiDecodeBatchMtpBudget), never here —
     * gating here oscillated between drafting and discarding. */
    return speculative_verify_slot != 0 &&
        speculative_verify_slot->state ==
            SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
}

static uint32_t SparkGlm52RequestApiShouldFillDecodeBatch(
	const SparkGlm52RequestApi *api,
	const SparkGlm52RequestApiSlot *prefill_slot,
	const SparkGlm52RequestApiSlot *decode_slot)
{
	const SparkGlm52RequestApiSlot *slot;
	uint32_t batch_target;
	uint32_t ready_decode_count;
	uint32_t slot_index;

	if (!SparkGlm52RequestApiDecodeBatchingIsEnabled(api) ||
		prefill_slot == 0 || decode_slot == 0 ||
		prefill_slot->priority < decode_slot->priority)
		return 0u;
	batch_target = SparkGlm52RequestApiCurrentPipelineBatchWidth(api);
	ready_decode_count = 0u;
	for (slot_index = 0u;
		 slot_index < api->request_capacity &&
			ready_decode_count < batch_target;
		 ++slot_index)
	{
		slot = &api->request_slots[slot_index];
		if (SparkGlm52RequestApiSlotIsSchedulableDecode(slot) &&
			SparkGlm52RequestApiSlotsHaveSameSchedulingPriority(
				slot,
				decode_slot))
			ready_decode_count += 1u;
	}
	return ready_decode_count < batch_target;
}

static uint32_t SparkGlm52RequestApiPrefillHasResidentKvHeadroom(
	const SparkGlm52RequestApi *api,
	const SparkGlm52RequestApiSlot *prefill_slot)
{
	uint64_t reserved_block_count;

	reserved_block_count = SparkGlm52RequestApiResidentKvBlockCount(api);
	return SparkGlm52RequestApiReservePrefillResidentKvBlocks(
		api,prefill_slot,&reserved_block_count);
}

static SparkGlm52RequestApiSlot *SparkGlm52RequestApiChooseReadySlot(
	SparkGlm52RequestApi *api,
	SparkGlm52RequestApiSlot *prefill_slot,
	SparkGlm52RequestApiSlot *decode_slot,
	SparkGlm52RequestApiSlot *speculative_verify_slot,
	uint32_t *chosen_is_prefill)
{
	SparkGlm52RequestApiSlot *chosen_slot;

	SparkGlm52SchedulerSetPrefillDemand(
		api->scheduler,
		prefill_slot != 0 ? 1u : 0u);
	*chosen_is_prefill = 0u;
	chosen_slot = decode_slot;
	if (prefill_slot != 0 &&
		(chosen_slot == 0 ||
		 SparkGlm52RequestApiSlotHasHigherSchedulingPriority(
			prefill_slot,
			chosen_slot) ||
		 (chosen_slot == decode_slot &&
		  ((SparkGlm52RequestApiSlotsHaveSameSchedulingPriority(
			prefill_slot,decode_slot) &&
			SparkGlm52RequestApiPrefillHasResidentKvHeadroom(
				api,prefill_slot) != 0u) ||
		   SparkGlm52RequestApiShouldFillDecodeBatch(
			api,prefill_slot,decode_slot) != 0u))))
	{
		*chosen_is_prefill = 1u;
		chosen_slot = prefill_slot;
	}
	if (speculative_verify_slot != 0 &&
		(chosen_slot == 0 ||
		 speculative_verify_slot->priority > chosen_slot->priority ||
		 (*chosen_is_prefill != 0u &&
		  SparkGlm52RequestApiSlotsHaveSameSchedulingPriority(
			speculative_verify_slot,chosen_slot)) ||
		 (chosen_slot == decode_slot &&
		  SparkGlm52RequestApiSlotsHaveSameSchedulingPriority(
			speculative_verify_slot,decode_slot) &&
		  SparkGlm52RequestApiMtpVerifyOutranksDecode(
			api,decode_slot,speculative_verify_slot))))
	{
		*chosen_is_prefill = 0u;
		chosen_slot = speculative_verify_slot;
	}
	return chosen_slot;
}

SparkStatus SparkGlm52RequestApiScheduleNext(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiDispatch *dispatch)
{
    SparkGlm52RequestApiSlot *prefill_slot;
    SparkGlm52RequestApiSlot *decode_slot;
    SparkGlm52RequestApiSlot *speculative_verify_slot;
    SparkGlm52RequestApiSlot *chosen_slot;
    SparkGlm52RequestApiPrefixFamilyChoice prefix_family_choice;
    uint32_t chosen_is_prefill;
    uint32_t overlaps_pending_prefetch;
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
    overlaps_pending_prefetch = 0u;
    prefill_slot = SparkGlm52RequestApiFindBestPrefillSlot(api, 0u);
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
    speculative_verify_slot = SparkGlm52RequestApiFindBestSchedulableSlot(
        api,
        SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify,
        0,
        0u,
        SparkGlm52RequestApiJitPrefetchIsEnabled(api) ? 0u : 1u);
    decode_slot = SparkGlm52RequestApiFindBestSchedulableSlot(
        api,
        SparkGlm52RequestApiSlotIsSchedulableDecode,
        0,
        0u,
        SparkGlm52RequestApiJitPrefetchIsEnabled(api) ? 0u : 1u);
    if (prefill_slot == 0 && decode_slot == 0 && speculative_verify_slot == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    chosen_slot = SparkGlm52RequestApiChooseReadySlot(
        api,
        prefill_slot,
        decode_slot,
        speculative_verify_slot,
        &chosen_is_prefill);

    status = SparkGlm52RequestApiRunSlotArrayCriticalJitKvPrefetch(
        api,
        &chosen_slot,
        1u,
        dispatch);
    if (status == SPARK_STATUS_BUSY &&
        SparkGlm52RequestApiAsyncJitPrefetchIsEnabled(api))
    {
        uint32_t pending_dispatch_flags;

        pending_dispatch_flags = dispatch->flags |
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
        overlaps_pending_prefetch = 1u;
        selected_shared_prefix_token_count = 0u;
        prefill_slot = SparkGlm52RequestApiFindBestPrefillSlot(api, 1u);
        speculative_verify_slot =
            SparkGlm52RequestApiFindBestSchedulableSlot(
        api,
        SparkGlm52RequestApiSlotIsSchedulableSpeculativeVerify, 0, 0u, 1u);
        decode_slot = SparkGlm52RequestApiFindBestSchedulableSlot(api, SparkGlm52RequestApiSlotIsSchedulableDecode, 0, 0u, 1u);
        chosen_slot = SparkGlm52RequestApiChooseReadySlot(
            api,
            prefill_slot,
            decode_slot,
            speculative_verify_slot,
            &chosen_is_prefill);
        if (chosen_slot == 0)
        {
            return SPARK_STATUS_BUSY;
        }
        SparkGlm52RequestApiInitializeDispatch(dispatch);
        dispatch->flags = pending_dispatch_flags;
        status = SparkGlm52RequestApiRunSlotArrayCriticalJitKvPrefetch(
            api,
            &chosen_slot,
            1u,
            dispatch);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (overlaps_pending_prefetch == 0u)
    {
        status = SparkGlm52RequestApiRunOpportunisticJitKvPrefetch(
            api,
            chosen_slot);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
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
            if (status == SPARK_STATUS_OK && dispatch->accepted != 0u)
            {
                return SPARK_STATUS_OK;
            }
            if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY &&
                status != SPARK_STATUS_NOT_FOUND)
            {
                return status;
            }
        }
        status = SparkGlm52RequestApiSchedulePrefill(
            api,
            prefill_slot,
            selected_shared_prefix_token_count,
            dispatch);
        if (status == SPARK_STATUS_OK && dispatch->accepted != 0u)
        {
            return SPARK_STATUS_OK;
        }
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
        {
            return status;
        }
        if (decode_slot == 0 && speculative_verify_slot == 0)
        {
            return status;
        }
        chosen_slot = decode_slot != 0 ? decode_slot : speculative_verify_slot;
        chosen_is_prefill = 0u;
        {
            uint32_t saved_dispatch_flags;

            saved_dispatch_flags = dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            SparkGlm52RequestApiInitializeDispatch(dispatch);
            dispatch->flags = saved_dispatch_flags;
        }
    }

    if (chosen_slot == speculative_verify_slot && speculative_verify_slot != 0)
    {
        if (!SparkGlm52RequestApiDecodeBlocksAreResident(
                api,
                speculative_verify_slot))
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return SPARK_STATUS_BUSY;
        }
        if (SparkGlm52RequestApiOlderLowerPrioritySchedulableSlotExists(
                api,
                speculative_verify_slot))
        {
            dispatch->flags |=
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
        }
        return SparkGlm52RequestApiScheduleSpeculativeVerifyBatch(
            api,
            speculative_verify_slot,
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

    if (slot->inflight_prefill_dispatch_count != 0u)
        slot->inflight_prefill_dispatch_count -= 1u;
    api->running_request_count -= 1u;
    slot->completed_prefill_step_count += 1u;
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
    if (slot->computed_prompt_token_count < slot->prompt_token_count)
    {
        if (slot->inflight_prefill_dispatch_count != 0u)
            return;
        if (slot->dispatched_prompt_token_count >
            slot->computed_prompt_token_count)
            slot->dispatched_prompt_token_count =
                slot->computed_prompt_token_count;
        slot->state = SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL;
        api->queued_request_count += 1u;
        return;
    }
    if (slot->inflight_prefill_dispatch_count != 0u)
        return;

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

    if (slot->inflight_prefill_dispatch_count != 0u)
        slot->inflight_prefill_dispatch_count -= 1u;
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
        if (slot->dispatched_prompt_token_count >
            slot->computed_prompt_token_count)
            slot->dispatched_prompt_token_count =
                slot->computed_prompt_token_count;
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

static void SparkGlm52RequestApiConsumeDecodeBudget(
    SparkGlm52RequestApiSlot *slot,
    uint32_t committed_token_count)
{
    uint32_t consumed_token_count;

    consumed_token_count = 0u;
    while (consumed_token_count < committed_token_count &&
           slot->remaining_thinking_token_budget != 0u)
    {
        slot->remaining_thinking_token_budget -= 1u;
        consumed_token_count += 1u;
    }
    while (consumed_token_count < committed_token_count &&
           slot->remaining_output_token_budget != 0u)
    {
        slot->remaining_output_token_budget -= 1u;
        consumed_token_count += 1u;
    }
}

static SparkStatus SparkGlm52RequestApiPrepareDsparkDraftForSlot(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot)
{
    SparkGlm52DsparkDraftRequest draft_request;
    uint64_t tap_generation;
    uint32_t requested_token_count;
    SparkStatus status;

    if (!SparkGlm52RequestApiSlotCanUseDspark(api, slot) ||
        SparkGlm52RequestApiSlotRemainingDecodeBudget(slot) == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = SparkGlm52DsparkMarkVerifierTapsReady(
        api->dspark_speculator,
        slot->request_id,
        slot->sequence_id,
        (uint64_t)slot->computed_prompt_token_count +
            (uint64_t)slot->completed_decode_token_count,
        &tap_generation);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    requested_token_count = api->dspark_speculator->default_speculative_token_count;
    if (requested_token_count + 1u > SparkGlm52RequestApiSlotRemainingDecodeBudget(slot))
    {
        requested_token_count =
            SparkGlm52RequestApiSlotRemainingDecodeBudget(slot) - 1u;
    }
    if (requested_token_count == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(&draft_request, 0, sizeof(draft_request));
    draft_request.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    draft_request.descriptor_bytes =
        SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
    draft_request.requested_token_count = requested_token_count;
    draft_request.priority = slot->priority;
    draft_request.request_id = slot->request_id;
    draft_request.sequence_id = slot->sequence_id;
    draft_request.sequence_position =
        (uint64_t)slot->computed_prompt_token_count +
        (uint64_t)slot->completed_decode_token_count;
    draft_request.tap_generation = tap_generation;

    status = SparkGlm52DsparkEnsureDraft(
        api->dspark_speculator,
        &draft_request);
    if (status == SPARK_STATUS_OK)
    {
        slot->state =
            SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
        api->dspark_draft_ready_count += 1u;
    }
    return status;
}

static void SparkGlm52RequestApiFinishSlotAfterDecode(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t committed_token_count)
{
    SparkStatus status;

    slot->mtp_resolution_base_position = 0u;
    slot->mtp_resolution_proposed_token_count = 0u;
    slot->mtp_resolution_accepted_token_count = 0u;
    slot->mtp_resolution_committed_token_count = 0u;
    slot->mtp_resolution_path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
    if (slot->mtp_next_draft_token_budget == 0u &&
        slot->mtp_probe_countdown != 0u)
    {
        if (slot->mtp_probe_countdown > committed_token_count)
        {
            slot->mtp_probe_countdown -= committed_token_count;
        }
        else
        {
            slot->mtp_probe_countdown = 0u;
            slot->mtp_next_draft_token_budget =
                SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
        }
    }
    SparkGlm52RequestApiConsumeDecodeBudget(slot, committed_token_count);
    slot->completed_decode_token_count += committed_token_count;
    api->running_request_count -= 1u;
    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
        return;
    }

    slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    status = SparkGlm52RequestApiPrepareDsparkDraftForSlot(api, slot);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    }
}

static SparkStatus SparkGlm52RequestApiFinishSlotAfterSpeculativeVerify(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiSlot *slot,
    uint32_t proposed_token_count,
    uint32_t accepted_draft_token_count,
    uint32_t committed_token_count,
    uint32_t fallback_token_id,
    uint32_t resolution_path_id,
    uint32_t mtp_verify)
{
    SparkGlm52DsparkVerifyResult verify_result;
    uint64_t resolution_base_position;
    SparkStatus status;

    if (proposed_token_count == 0u ||
        proposed_token_count > SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        accepted_draft_token_count > proposed_token_count ||
        committed_token_count == 0u ||
        committed_token_count > proposed_token_count + 1u ||
        committed_token_count > SparkGlm52RequestApiSlotRemainingDecodeBudget(slot))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    resolution_base_position = 0u;
    if (mtp_verify != 0u)
    {
        if (proposed_token_count !=
                SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
            accepted_draft_token_count >
                SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION ||
            committed_token_count != accepted_draft_token_count + 1u ||
            committed_token_count >
                SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT ||
            resolution_path_id >=
                SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_COUNT ||
            SparkGlm52MtpTreeAcceptedTokenCount(
                resolution_path_id) != accepted_draft_token_count ||
            slot->computed_prompt_token_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        resolution_base_position =
            (uint64_t)slot->computed_prompt_token_count +
            (uint64_t)slot->completed_decode_token_count - 1u;
    }

    memset(&verify_result, 0, sizeof(verify_result));
    verify_result.abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    verify_result.descriptor_bytes =
        SPARK_GLM52_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES;
    verify_result.proposed_token_count = proposed_token_count;
    verify_result.accepted_draft_token_count = accepted_draft_token_count;
    verify_result.committed_token_count = committed_token_count;
    verify_result.fallback_token_id = fallback_token_id;
    if (accepted_draft_token_count == proposed_token_count)
    {
        verify_result.flags |= SPARK_GLM52_DSPARK_VERIFY_RESULT_FLAG_ACCEPTED_ALL;
    }
    else
    {
        verify_result.flags |= SPARK_GLM52_DSPARK_VERIFY_RESULT_FLAG_REJECTED;
    }

    if (mtp_verify != 0u)
    {
        int32_t ema_delta;
        if (slot->mtp_draft_token_count != proposed_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        ema_delta = (int32_t)(committed_token_count * 1000u) -
            (int32_t)slot->mtp_commit_ema_milli;
        slot->mtp_commit_ema_milli = (uint32_t)(
            (int32_t)slot->mtp_commit_ema_milli +
            ema_delta / SPARK_GLM52_REQUEST_API_MTP_COMMIT_EMA_DIVISOR);
        if (slot->mtp_commit_ema_milli <
            SPARK_GLM52_REQUEST_API_MTP_SUPPRESS_THRESHOLD_MILLI)
        {
            slot->mtp_next_draft_token_budget = 0u;
            slot->mtp_probe_countdown =
                SPARK_GLM52_REQUEST_API_MTP_REPROBE_INTERVAL;
        }
        else
        {
            slot->mtp_next_draft_token_budget =
                SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
            slot->mtp_probe_countdown = 0u;
        }
        slot->mtp_resolution_base_position = resolution_base_position;
        slot->mtp_resolution_proposed_token_count = proposed_token_count;
        slot->mtp_resolution_accepted_token_count = accepted_draft_token_count;
        slot->mtp_resolution_committed_token_count = committed_token_count;
        slot->mtp_resolution_path_id = resolution_path_id;
        memset(slot->mtp_draft_token_ids, 0, sizeof(slot->mtp_draft_token_ids));
        slot->mtp_draft_token_count = 0u;
        api->mtp_accepted_draft_token_count += accepted_draft_token_count;
        api->mtp_committed_token_count += committed_token_count;
        if (accepted_draft_token_count < proposed_token_count)
        {
            api->mtp_rejected_token_count +=
                proposed_token_count - accepted_draft_token_count;
        }
    }
    else
    {
        slot->mtp_resolution_base_position = 0u;
        slot->mtp_resolution_proposed_token_count = 0u;
        slot->mtp_resolution_accepted_token_count = 0u;
        slot->mtp_resolution_committed_token_count = 0u;
        if (resolution_path_id !=
            SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        slot->mtp_resolution_path_id =
            SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
        if (slot->mtp_draft_token_count != 0u)
        {
            memset(slot->mtp_draft_token_ids,0,sizeof(slot->mtp_draft_token_ids));
            slot->mtp_draft_token_count = 0u;
            if (api->mtp_draft_ready_count != 0u)
                api->mtp_draft_ready_count -= 1u;
        }
        status = SparkGlm52DsparkCompleteVerify(
            api->dspark_speculator,
            slot->sequence_id,
            &verify_result);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        api->dspark_accepted_draft_token_count += accepted_draft_token_count;
        api->dspark_committed_token_count += committed_token_count;
        if (accepted_draft_token_count < proposed_token_count)
        {
            api->dspark_rejected_token_count +=
                proposed_token_count - accepted_draft_token_count;
        }
    }

    SparkGlm52RequestApiConsumeDecodeBudget(slot, committed_token_count);
    slot->completed_decode_token_count += committed_token_count;
    api->running_request_count -= 1u;

    if (slot->remaining_thinking_token_budget == 0u &&
        slot->remaining_output_token_budget == 0u)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_COMPLETED;
        api->completed_request_count += 1u;
        return SPARK_STATUS_OK;
    }

    slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    status = SparkGlm52RequestApiPrepareDsparkDraftForSlot(api, slot);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
    {
        slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_DECODE;
    }
    return SPARK_STATUS_OK;
}


SparkStatus SparkGlm52RequestApiArmMtpVerifyDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *completed_decode_dispatch,
    const uint32_t *draft_token_ids,
    uint32_t lane_stride,
    uint32_t draft_token_count)
{
	uint32_t arm_draft_token_count;
	uint32_t dispatch_is_mtp_producer;
	uint32_t dispatch_is_mtp_verify;
    uint32_t request_index;
    uint32_t token_index;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    dispatch_is_mtp_producer = completed_decode_dispatch != 0 &&
        completed_decode_dispatch->kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
        (completed_decode_dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u;
    dispatch_is_mtp_verify = completed_decode_dispatch != 0 &&
        completed_decode_dispatch->kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
        (completed_decode_dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
    if (completed_decode_dispatch == 0 ||
        completed_decode_dispatch->abi_version !=
            SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        completed_decode_dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        completed_decode_dispatch->accepted == 0u ||
        (dispatch_is_mtp_producer == 0u && dispatch_is_mtp_verify == 0u) ||
        completed_decode_dispatch->request_count == 0u ||
        completed_decode_dispatch->request_count >
            SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
        draft_token_ids == 0 ||
        draft_token_count == 0u ||
        draft_token_count >
            SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT ||
        (dispatch_is_mtp_producer != 0u &&
         draft_token_count > completed_decode_dispatch->mtp_draft_token_budget) ||
        lane_stride < draft_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (draft_token_count != SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
        (dispatch_is_mtp_producer != 0u &&
         completed_decode_dispatch->mtp_draft_token_budget !=
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) ||
        (dispatch_is_mtp_verify != 0u &&
         ((completed_decode_dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) == 0u ||
          completed_decode_dispatch->mtp_draft_token_budget !=
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)))
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    arm_draft_token_count = SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
    for (request_index = 0u;
         request_index < completed_decode_dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = SparkGlm52RequestApiFindSlotByHandle(
            api,
            completed_decode_dispatch->request_handles[request_index]);
        if (slot == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        if (slot->state == SPARK_GLM52_REQUEST_API_STATE_COMPLETED ||
            SparkGlm52RequestApiSlotRemainingDecodeBudget(slot) <
                SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT)
            return SPARK_STATUS_NOT_FOUND;
        if ((slot->state != SPARK_GLM52_REQUEST_API_STATE_READY_DECODE &&
             slot->state !=
                SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) ||
            slot->mtp_draft_token_count != 0u)
            return SPARK_STATUS_INVALID_ARGUMENT;
        if (slot->mtp_next_draft_token_budget == 0u)
            return SPARK_STATUS_NOT_FOUND;
        if (slot->mtp_next_draft_token_budget !=
            SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
            return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }
    if (arm_draft_token_count == 0u)
        return SPARK_STATUS_NOT_FOUND;
    for (request_index = 0u;
         request_index < completed_decode_dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = SparkGlm52RequestApiFindSlotByHandle(
            api,
            completed_decode_dispatch->request_handles[request_index]);
        for (token_index = 0u;
             token_index < arm_draft_token_count;
             ++token_index)
        {
            slot->mtp_draft_token_ids[token_index] =
                draft_token_ids[(uint64_t)request_index * lane_stride +
                    token_index];
        }
        for (; token_index < SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT;
             ++token_index)
        {
            slot->mtp_draft_token_ids[token_index] = 0u;
        }
        slot->mtp_draft_token_count = arm_draft_token_count;
        if (slot->state ==
                SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
            api->dspark_draft_ready_count != 0u)
            api->dspark_draft_ready_count -= 1u;
        slot->state = SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
        api->mtp_draft_ready_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiResolveMtpTreeVerifierTokens(
    const uint32_t *candidate_token_ids,
    const uint32_t *verifier_token_ids,
    SparkGlm52DsparkVerifyResult *verify_result,
    uint32_t *resolution_path_id_out)
{
    SparkGlm52MtpTreeResolution resolution;
    SparkStatus status;
    if (candidate_token_ids == 0 || verifier_token_ids == 0 ||
        verify_result == 0 || resolution_path_id_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkGlm52MtpTreeResolve(
        candidate_token_ids,verifier_token_ids,&resolution);
    if (status != SPARK_STATUS_OK)
        return status;
    memset(verify_result,0,sizeof(*verify_result));
    verify_result->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
    verify_result->descriptor_bytes =
        SPARK_GLM52_DSPARK_VERIFY_RESULT_DESCRIPTOR_BYTES;
    verify_result->proposed_token_count =
        SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
    verify_result->accepted_draft_token_count =
        resolution.accepted_token_count;
    verify_result->committed_token_count = resolution.committed_token_count;
    verify_result->flags = SPARK_GLM52_DSPARK_VERIFY_RESULT_FLAG_REJECTED;
    verify_result->fallback_token_id =
        verifier_token_ids[resolution.fallback_row_index];
    *resolution_path_id_out = resolution.path_id;
    return SPARK_STATUS_OK;
}

static const char *SparkGlm52RequestApiSpeculativeTraceSource(
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *trace_confidence)
{
    if (dispatch == 0 || trace_confidence == 0)
    {
        return 0;
    }
    if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
    {
        if (getenv("SPARKPIPE_DSPARK_TRACE") == 0)
        {
            return 0;
        }
        *trace_confidence = 1u;
        return "dspark";
    }
    if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
    {
        if (getenv("SPARKPIPE_RING_TRACE") == 0)
        {
            return 0;
        }
        *trace_confidence = 0u;
        return "mtp";
    }
    return 0;
}

static void SparkGlm52RequestApiTraceTokenIds(
    const char *label,
    const uint32_t *token_ids,
    uint32_t token_count)
{
    uint32_t token_index;

    fprintf(stderr," %s=",label);
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        fprintf(stderr,"%s%u",token_index == 0u ? "" : ",",
            token_ids[token_index]);
    }
}

static void SparkGlm52RequestApiTraceSpeculativeVerify(
    const SparkGlm52RequestApiDispatch *dispatch,
    const uint32_t *verifier_token_ids,
    uint32_t lane_stride,
    uint32_t verifier_token_count,
    uint32_t request_index,
    const SparkGlm52DsparkVerifyResult *verify_result)
{
    const char *source;
    uint32_t trace_confidence;

    if (verifier_token_ids == 0 || verify_result == 0)
    {
        return;
    }
    source = SparkGlm52RequestApiSpeculativeTraceSource(
        dispatch,&trace_confidence);
    if (source == 0)
    {
        return;
    }
    fprintf(stderr,
        "%s_trace verify request=%llu sequence=%llu proposed=%u accepted=%u committed=%u fallback=%u",
        source,
        (unsigned long long)dispatch->request_ids[request_index],
        (unsigned long long)dispatch->sequence_ids[request_index],
        dispatch->speculative_token_count,
        verify_result->accepted_draft_token_count,
        verify_result->committed_token_count,
        verify_result->fallback_token_id);
    SparkGlm52RequestApiTraceTokenIds(
        "draft_ids",
        dispatch->speculative_draft_token_ids[request_index],
        dispatch->speculative_token_count);
    if (trace_confidence != 0u)
        SparkGlm52RequestApiTraceTokenIds(
            "confidence_milli",
            dispatch->speculative_confidence_milli[request_index],
            dispatch->speculative_token_count);
    SparkGlm52RequestApiTraceTokenIds(
        "verifier_ids",
        &verifier_token_ids[(uint64_t)request_index * lane_stride],
        verifier_token_count);
    fprintf(stderr,"\n");
}

SparkStatus SparkGlm52RequestApiResolveSpeculativeVerifyDispatch(
    SparkGlm52RequestApi *api,
    SparkGlm52RequestApiDispatch *dispatch,
    const uint32_t *verifier_token_ids,
    uint32_t lane_stride,
    uint32_t verifier_token_count)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (dispatch == 0 ||
        dispatch->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        dispatch->kind !=
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH ||
        dispatch->request_count == 0u ||
        dispatch->request_count >
            SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
        dispatch->speculative_token_count == 0u ||
        dispatch->speculative_token_count >
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT ||
        verifier_token_ids == 0 ||
        dispatch->speculative_verifier_token_count == 0u ||
        verifier_token_count != dispatch->speculative_verifier_token_count ||
        lane_stride < verifier_token_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52DsparkVerifyResult verify_result;
        uint32_t resolution_path_id;

        resolution_path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
        if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u)
        {
            if ((dispatch->flags &
                    SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) == 0u ||
                dispatch->speculative_token_count !=
                    SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
                dispatch->speculative_verifier_token_count !=
                    SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT ||
                dispatch->speculative_max_committed_token_count !=
                    SPARK_GLM52_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkGlm52RequestApiResolveMtpTreeVerifierTokens(
                dispatch->speculative_draft_token_ids[request_index],
                &verifier_token_ids[(uint64_t)request_index * lane_stride],
                &verify_result,
                &resolution_path_id);
        }
        else
        {
            status = SparkGlm52DsparkResolveVerifierTokens(
                dispatch->speculative_draft_token_ids[request_index],
                dispatch->speculative_token_count,
                &verifier_token_ids[(uint64_t)request_index * lane_stride],
                verifier_token_count,
                &verify_result);
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        SparkGlm52RequestApiTraceSpeculativeVerify(
            dispatch,
            verifier_token_ids,
            lane_stride,
            verifier_token_count,
            request_index,
            &verify_result);

        dispatch->speculative_accepted_token_counts[request_index] =
            verify_result.accepted_draft_token_count;
        dispatch->speculative_committed_token_counts[request_index] =
            verify_result.committed_token_count;
        dispatch->speculative_fallback_token_ids[request_index] =
            verify_result.fallback_token_id;
        dispatch->speculative_resolution_path_ids[request_index] =
            resolution_path_id;
    }
    return SPARK_STATUS_OK;
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
            if (slot != 0 && request_index == 0u &&
                slot->state == SPARK_GLM52_REQUEST_API_STATE_CANCELLED)
            {
                api->stale_prefill_completion_count += 1u;
                continue;
            }
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
            uint32_t committed_token_count;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 || slot->state !=
                SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            committed_token_count = 1u;
            if ((dispatch->flags &
                    SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u)
            {
                committed_token_count =
                    dispatch->decode_committed_token_counts[request_index];
                if (committed_token_count != 1u ||
                    committed_token_count >
                        SparkGlm52RequestApiSlotRemainingDecodeBudget(slot))
                {
                    return SPARK_STATUS_INVALID_ARGUMENT;
                }
            }
            SparkGlm52RequestApiFinishSlotAfterDecode(
                api,
                slot,
                committed_token_count);
        }
        return SPARK_STATUS_OK;
    }
    if (dispatch->kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        status = SparkGlm52SchedulerCompleteDecodeBatch(
            api->scheduler,
            &dispatch->decode_batch_decision);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (dispatch->speculative_token_count == 0u ||
            dispatch->speculative_token_count >
                SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (request_index = 0u;
             request_index < dispatch->request_count;
             ++request_index)
        {
            SparkGlm52RequestApiSlot *slot;
            uint32_t committed_token_count;
            uint32_t accepted_token_count;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[request_index]);
            if (slot == 0 ||
                slot->state !=
                    SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            committed_token_count =
                dispatch->speculative_committed_token_counts[request_index];
            accepted_token_count =
                dispatch->speculative_accepted_token_counts[request_index];
            status = SparkGlm52RequestApiFinishSlotAfterSpeculativeVerify(
                api,
                slot,
                dispatch->speculative_token_count,
                accepted_token_count,
                committed_token_count,
                dispatch->speculative_fallback_token_ids[request_index],
                dispatch->speculative_resolution_path_ids[request_index],
                (dispatch->flags &
                    SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY)
                        != 0u ? 1u : 0u);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
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
    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
        dispatch->kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return dispatch->decode_batch_decision.active_sequence_count;
    }
    return 0u;
}


SparkStatus SparkGlm52RequestApiDescribePrefillDispatch(
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52RequestApiPrefillDispatchView *prefill_view)
{
    uint32_t lane_index;
    uint32_t lane_count;
    uint32_t prompt_token_stride;

    if (prefill_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(prefill_view, 0, sizeof(*prefill_view));

    if (dispatch == 0 ||
        dispatch->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
    {
        const SparkGlm52SchedulerDecision *decision;

        decision = &dispatch->prefill_decision;
        if (decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
            decision->descriptor_bytes !=
                SPARK_GLM52_SCHEDULER_DECISION_DESCRIPTOR_BYTES ||
            decision->accepted == 0u ||
            decision->active_sequence_count == 0u ||
            decision->active_sequence_count > 1u ||
            decision->scheduled_prompt_token_count == 0u ||
            decision->prompt_token_ids == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        prefill_view->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
        prefill_view->descriptor_bytes =
            SPARK_GLM52_REQUEST_API_PREFILL_DISPATCH_VIEW_DESCRIPTOR_BYTES;
        prefill_view->kind = dispatch->kind;
        prefill_view->active_sequence_count = decision->active_sequence_count;
        prefill_view->lane_count = 1u;
        prefill_view->prompt_token_offset =
            decision->scheduled_prompt_token_offset;
        prefill_view->prompt_token_count =
            decision->scheduled_prompt_token_count;
        prefill_view->prompt_token_stride =
            decision->scheduled_prompt_token_count;
        prefill_view->lanes[0u].request_index = 0u;
        prefill_view->lanes[0u].prompt_token_offset =
            decision->scheduled_prompt_token_offset;
        prefill_view->lanes[0u].prompt_token_count =
            decision->scheduled_prompt_token_count;
        prefill_view->lanes[0u].request_slot_index =
            dispatch->request_slot_indices[0u];
        prefill_view->lanes[0u].request_id = dispatch->request_ids[0u];
        prefill_view->lanes[0u].sequence_id = dispatch->sequence_ids[0u];
        prefill_view->lanes[0u].request_handle = dispatch->request_handles[0u];
        prefill_view->lanes[0u].prompt_token_ids = decision->prompt_token_ids;
        return SPARK_STATUS_OK;
    }

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
    {
        const SparkGlm52SchedulerPrefillBatchDecision *batch_decision;

        batch_decision = &dispatch->prefill_batch_decision;
        lane_count = batch_decision->active_sequence_count;
        prompt_token_stride = batch_decision->maximum_scheduled_prompt_token_count;
        if (batch_decision->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
            batch_decision->descriptor_bytes !=
                SPARK_GLM52_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES ||
            batch_decision->accepted == 0u ||
            lane_count == 0u ||
            lane_count > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT ||
            prompt_token_stride == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        prefill_view->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
        prefill_view->descriptor_bytes =
            SPARK_GLM52_REQUEST_API_PREFILL_DISPATCH_VIEW_DESCRIPTOR_BYTES;
        prefill_view->kind = dispatch->kind;
        prefill_view->active_sequence_count = lane_count;
        prefill_view->lane_count = lane_count;
        prefill_view->prompt_token_stride = prompt_token_stride;
        for (lane_index = 0u; lane_index < lane_count; ++lane_index)
        {
            const SparkGlm52SchedulerPrefillBatchLane *lane;

            lane = &batch_decision->lanes[lane_index];
            if (lane->abi_version != SPARK_GLM52_SCHEDULER_ABI_VERSION ||
                lane->descriptor_bytes !=
                    SPARK_GLM52_SCHEDULER_PREFILL_BATCH_LANE_DESCRIPTOR_BYTES ||
                lane->active_sequence_count == 0u ||
                lane->scheduled_prompt_token_count == 0u ||
                lane->scheduled_prompt_token_count > prompt_token_stride ||
                lane->prompt_token_ids == 0)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            if (lane_index == 0u)
            {
                prefill_view->prompt_token_offset =
                    lane->scheduled_prompt_token_offset;
            }
            else if (prefill_view->prompt_token_offset !=
                lane->scheduled_prompt_token_offset)
            {
                prefill_view->prompt_token_offset = 0u;
            }
            prefill_view->prompt_token_count = SparkGlm52RequestApiMaximumU32(
                prefill_view->prompt_token_count,
                lane->scheduled_prompt_token_count);
            prefill_view->lanes[lane_index].request_index = lane->request_index;
            prefill_view->lanes[lane_index].prompt_token_offset =
                lane->scheduled_prompt_token_offset;
            prefill_view->lanes[lane_index].prompt_token_count =
                lane->scheduled_prompt_token_count;
            prefill_view->lanes[lane_index].request_slot_index =
                dispatch->request_slot_indices[lane_index];
            prefill_view->lanes[lane_index].request_id =
                dispatch->request_ids[lane_index];
            prefill_view->lanes[lane_index].sequence_id =
                dispatch->sequence_ids[lane_index];
            prefill_view->lanes[lane_index].request_handle =
                dispatch->request_handles[lane_index];
            prefill_view->lanes[lane_index].prompt_token_ids =
                lane->prompt_token_ids;
        }
        return SPARK_STATUS_OK;
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkGlm52RequestApiCopyPrefillDispatchTokenIds(
    const SparkGlm52RequestApiDispatch *dispatch,
    uint32_t *destination_token_ids,
    uint32_t destination_token_stride,
    uint32_t destination_lane_capacity)
{
    SparkGlm52RequestApiPrefillDispatchView prefill_view;
    uint32_t lane_index;
    SparkStatus status;

    if (destination_token_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52RequestApiDescribePrefillDispatch(
        dispatch,
        &prefill_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (destination_lane_capacity < prefill_view.lane_count ||
        destination_token_stride < prefill_view.prompt_token_stride)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (lane_index = 0u; lane_index < prefill_view.lane_count; ++lane_index)
    {
        const SparkGlm52RequestApiPrefillDispatchLaneView *lane;
        uint32_t token_index;
        uint32_t *destination_lane;

        lane = &prefill_view.lanes[lane_index];
        destination_lane =
            &destination_token_ids[(uint64_t)lane_index * destination_token_stride];
        for (token_index = 0u;
             token_index < lane->prompt_token_count;
             ++token_index)
        {
            destination_lane[token_index] =
                lane->prompt_token_ids[lane->prompt_token_offset + token_index];
        }
        for (token_index = lane->prompt_token_count;
             token_index < destination_token_stride;
             ++token_index)
        {
            destination_lane[token_index] = 0u;
        }
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RequestApiDescribeDecodeDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch,
    SparkGlm52RequestApiDecodeDispatchView *decode_view)
{
    uint32_t lane_index;
    SparkStatus status;

    if (decode_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(decode_view, 0, sizeof(*decode_view));

    status = SparkGlm52RequestApiValidate(api);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (dispatch == 0 ||
        dispatch->abi_version != SPARK_GLM52_REQUEST_API_ABI_VERSION ||
        dispatch->descriptor_bytes !=
            SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES ||
        dispatch->accepted == 0u ||
        (dispatch->kind != SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         dispatch->kind !=
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH) ||
        dispatch->request_count == 0u ||
        dispatch->request_count > SPARK_GLM52_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    decode_view->abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
    decode_view->descriptor_bytes =
        SPARK_GLM52_REQUEST_API_DECODE_DISPATCH_VIEW_DESCRIPTOR_BYTES;
    decode_view->kind = dispatch->kind;
    decode_view->active_sequence_count = dispatch->request_count;
    decode_view->lane_count = dispatch->request_count;
    decode_view->speculative_token_count = dispatch->speculative_token_count;
    for (lane_index = 0u;
         lane_index < dispatch->request_count;
         ++lane_index)
    {
        SparkGlm52RequestApiSlot *slot;
        SparkGlm52RequestApiDecodeDispatchLaneView *lane;
        uint64_t sequence_position;

        slot = SparkGlm52RequestApiFindSlotByHandle(
            api,
            dispatch->request_handles[lane_index]);
        if (slot == 0 ||
            dispatch->request_slot_indices[lane_index] !=
                SparkGlm52RequestApiSlotIndex(api, slot) ||
            slot->computed_prompt_token_count == 0u ||
            (slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE &&
             slot->state !=
                SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        sequence_position =
            (uint64_t)slot->computed_prompt_token_count +
            (uint64_t)slot->completed_decode_token_count - 1u;
        if (sequence_position > UINT32_MAX - 1u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }

        lane = &decode_view->lanes[lane_index];
        lane->request_index = lane_index;
        lane->sequence_position = (uint32_t)sequence_position;
        lane->context_token_count = (uint32_t)(sequence_position + 1u);
        lane->request_slot_index = SparkGlm52RequestApiSlotIndex(api, slot);
        if (lane->request_slot_index == SPARK_GLM52_REQUEST_API_NO_SLOT)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        lane->request_id = slot->request_id;
        lane->sequence_id = slot->sequence_id;
        lane->request_handle = slot->handle;
        if (slot->mtp_resolution_proposed_token_count == 0u)
        {
            if (slot->mtp_resolution_base_position != 0u ||
                slot->mtp_resolution_accepted_token_count != 0u ||
                slot->mtp_resolution_committed_token_count != 0u ||
                slot->mtp_resolution_path_id !=
                    SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
        }
        else if (slot->mtp_resolution_proposed_token_count >
                SPARK_GLM52_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT ||
            slot->mtp_resolution_committed_token_count !=
                slot->mtp_resolution_accepted_token_count + 1u ||
            slot->mtp_resolution_base_position >
                UINT64_MAX - slot->mtp_resolution_committed_token_count ||
            slot->mtp_resolution_base_position +
                slot->mtp_resolution_committed_token_count != sequence_position ||
            SparkGlm52MtpTreeResolutionIsValid(
                slot->mtp_resolution_proposed_token_count,
                slot->mtp_resolution_accepted_token_count,
                slot->mtp_resolution_path_id) == 0u)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        lane->mtp_resolution_base_position =
            slot->mtp_resolution_base_position;
        lane->mtp_resolution_proposed_token_count =
            slot->mtp_resolution_proposed_token_count;
        lane->mtp_resolution_accepted_token_count =
            slot->mtp_resolution_accepted_token_count;
        lane->mtp_resolution_committed_token_count =
            slot->mtp_resolution_committed_token_count;
        lane->mtp_resolution_path_id = slot->mtp_resolution_path_id;
    }

    return SPARK_STATUS_OK;
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

    if (dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
        dispatch->kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
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
            uint32_t required_token_count;

            slot = SparkGlm52RequestApiFindSlotByHandle(
                api,
                dispatch->request_handles[lane_index]);
            if (slot == 0 ||
                (slot->state != SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE &&
                 slot->state !=
                    SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY) ||
                slot->computed_prompt_token_count == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            status = SparkGlm52RequestApiEnsureDecodeSlotKvCapacity(
                api,
                slot,
                dispatch->kind ==
                    SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
                        ? (dispatch->flags &
                            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u
                            ? SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION
                            : dispatch->speculative_token_count
                        : (dispatch->flags &
                            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u
                            ? dispatch->mtp_draft_token_budget : 0u,
                &required_token_count);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            status = SparkGlm52PrefixCacheBuildPhysicalBlockTable(
                api->scheduler->prefix_cache,
                slot->sequence_id,
                required_token_count,
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
    block_table_view->host_lane_physical_block_counts = lane_physical_block_counts;

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
    if (SparkGlm52RequestApiDsparkSpeculationIsEnabled(api))
    {
        status = SparkGlm52DsparkCancelSequence(
            api->dspark_speculator,
            slot->sequence_id);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }
    status = SparkGlm52SchedulerReleaseSequence(api->scheduler, slot->sequence_id);
    if (status == SPARK_STATUS_OK)
    {
        slot->sequence_id = 0u;
    }
    return status;
}

static uint32_t SparkGlm52RequestApiRetryDecodeTokenCount(
    const SparkGlm52RequestApiDispatch *dispatch)
{
    if (dispatch != 0 &&
        dispatch->kind ==
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return dispatch->speculative_token_count;
    }
    return 1u;
}

static SparkStatus SparkGlm52RequestApiValidateRetryDecodeCounters(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t retry_token_count;

    if (api == 0 || dispatch == 0 || dispatch->accepted == 0u ||
        (dispatch->kind !=
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
         dispatch->kind !=
            SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH) ||
        dispatch->request_count == 0u ||
        api->running_request_count < dispatch->request_count ||
        api->scheduled_decode_dispatch_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    retry_token_count = SparkGlm52RequestApiRetryDecodeTokenCount(dispatch);
    if (retry_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (dispatch->kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        if ((dispatch->flags &
                SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
        {
            if (api->mtp_verify_dispatch_count == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else if (api->dspark_verify_dispatch_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiValidateRetryDecodeSlots(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t request_index;
    uint32_t retry_token_count;

    retry_token_count = SparkGlm52RequestApiRetryDecodeTokenCount(dispatch);
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *slot;
        uint32_t expected_state;

        slot = SparkGlm52RequestApiFindSlotByHandle(
            api,
            dispatch->request_handles[request_index]);
        expected_state =
            dispatch->kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH
            ? SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE
            : SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY;
        if (slot == 0 || slot->state != expected_state ||
            slot->scheduled_decode_token_count < retry_token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RequestApiValidateRetryDecodeDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    SparkStatus status;

    status = SparkGlm52RequestApiValidateRetryDecodeCounters(api,dispatch);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52RequestApiValidateRetryDecodeSlots(api,dispatch);
}

static void SparkGlm52RequestApiRestoreRetriedDecodeSlots(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    uint32_t request_index;
    uint32_t retry_token_count;

    retry_token_count = SparkGlm52RequestApiRetryDecodeTokenCount(dispatch);
    for (request_index = 0u;
         request_index < dispatch->request_count;
         ++request_index)
    {
        SparkGlm52RequestApiSlot *slot;

        slot = SparkGlm52RequestApiFindSlotByHandle(
            api,
            dispatch->request_handles[request_index]);
        slot->scheduled_decode_token_count -= retry_token_count;
        slot->state =
            dispatch->kind ==
                SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH
            ? SPARK_GLM52_REQUEST_API_STATE_READY_DECODE
            : SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY;
    }
}

static void SparkGlm52RequestApiRestoreRetriedDecodeCounters(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    api->running_request_count -= dispatch->request_count;
    api->scheduled_decode_dispatch_count -= 1u;
    if (dispatch->kind !=
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
    {
        return;
    }
    if ((dispatch->flags &
            SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
    {
        api->mtp_verify_dispatch_count -= 1u;
    }
    else
    {
        api->dspark_verify_dispatch_count -= 1u;
    }
}

SparkStatus SparkGlm52RequestApiRetryDecodeDispatch(
    SparkGlm52RequestApi *api,
    const SparkGlm52RequestApiDispatch *dispatch)
{
    SparkStatus status;

    status = SparkGlm52RequestApiValidate(api);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52RequestApiValidateRetryDecodeDispatch(
            api,
            dispatch);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52SchedulerCancelDecodeBatch(
        api->scheduler,
        &dispatch->decode_batch_decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkGlm52RequestApiRestoreRetriedDecodeSlots(api,dispatch);
    SparkGlm52RequestApiRestoreRetriedDecodeCounters(api,dispatch);
    return SPARK_STATUS_OK;
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
    if (dispatch->kind ==
        SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
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
            if (slot == 0 ||
                slot->state !=
                    SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            if (SparkGlm52RequestApiDsparkSpeculationIsEnabled(api))
            {
                (void)SparkGlm52DsparkCancelSequence(
                    api->dspark_speculator,
                    slot->sequence_id);
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

SparkStatus SparkGlm52RequestApiFinishRequestGeneration(
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

    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_COMPLETED)
    {
        return SPARK_STATUS_OK;
    }
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_CANCELLED)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_PREFILL ||
        slot->state == SPARK_GLM52_REQUEST_API_STATE_RUNNING_DECODE ||
        slot->state ==
            SPARK_GLM52_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY ||
        slot->state == SPARK_GLM52_REQUEST_API_STATE_WAITING_PREFIX_COHORT)
    {
        return SPARK_STATUS_BUSY;
    }

    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_QUEUED_PREFILL)
    {
        if (api->queued_request_count == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        api->queued_request_count -= 1u;
    }
    else if (slot->state != SPARK_GLM52_REQUEST_API_STATE_READY_DECODE &&
             slot->state !=
                SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY &&
        SparkGlm52RequestApiDsparkSpeculationIsEnabled(api))
    {
        status = SparkGlm52DsparkCancelSequence(
            api->dspark_speculator,
            slot->sequence_id);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
        {
            return status;
        }
    }

    slot->remaining_thinking_token_budget = 0u;
    slot->remaining_output_token_budget = 0u;
    slot->state = SPARK_GLM52_REQUEST_API_STATE_COMPLETED;
    api->completed_request_count += 1u;
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
    if (slot->state == SPARK_GLM52_REQUEST_API_STATE_CANCELLED)
    {
        return SPARK_STATUS_OK;
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
    SparkGlm52RequestApiRemoveSlotHash(api, slot);
    {
        uint32_t released_slot_index;

        released_slot_index = SparkGlm52RequestApiSlotIndex(api, slot);
        if (released_slot_index == SPARK_GLM52_REQUEST_API_NO_SLOT)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        SparkGlm52RequestApiInitializeSlot(slot);
        slot->free_slot_next = api->free_slot_head;
        api->free_slot_head = released_slot_index;
    }
    return SPARK_STATUS_OK;
}

uint32_t SparkGlm52RequestApiAssignDraftBudgets(
    SparkGlm52RequestApi *api,
    uint32_t firing_row_cap,
    struct SparkGlm52RowAllocatorSlotInput *scratch_inputs,
    uint32_t *scratch_budgets)
{
    uint32_t slot_index,eligible_count,total,apply_index;
    if (api == 0 || scratch_inputs == 0 || scratch_budgets == 0 || firing_row_cap == 0u)
        return 0u;
    eligible_count = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot = &api->request_slots[slot_index];
        if ((slot->state != SPARK_GLM52_REQUEST_API_STATE_READY_DECODE &&
             slot->state != SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) ||
            (slot->remaining_thinking_token_budget == 0u &&
             slot->remaining_output_token_budget == 0u))
            continue;
        scratch_inputs[eligible_count].commit_ema_milli = slot->mtp_commit_ema_milli;
        scratch_inputs[eligible_count].maximum_draft_depth =
            (slot->mtp_next_draft_token_budget == 0u && slot->mtp_probe_countdown != 0u) ?
            0u : SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
        scratch_inputs[eligible_count].probe = 0u;
        eligible_count += 1u;
    }
    total = SparkGlm52RowAllocatorAssign(scratch_inputs, eligible_count, firing_row_cap, 1000u, scratch_budgets);
    apply_index = 0u;
    for (slot_index = 0u; slot_index < api->request_capacity && apply_index < eligible_count; ++slot_index)
    {
        SparkGlm52RequestApiSlot *slot = &api->request_slots[slot_index];
        if ((slot->state != SPARK_GLM52_REQUEST_API_STATE_READY_DECODE &&
             slot->state != SPARK_GLM52_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY) ||
            (slot->remaining_thinking_token_budget == 0u &&
             slot->remaining_output_token_budget == 0u))
            continue;
        slot->mtp_next_draft_token_budget = scratch_budgets[apply_index];
        apply_index += 1u;
    }
    return total;
}
