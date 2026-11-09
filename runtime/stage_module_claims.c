#include "sparkpipe/spark_stage_module_claims.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_model_driver.h"

SparkStatus SparkStageModuleSlotClaim(
    atomic_uint *slot_states,
    uint32_t slot_count,
    uint32_t *slot_index)
{
    uint32_t index;

    if (slot_states == 0 || slot_index == 0 || slot_count == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    *slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    for (index = 0u; index < slot_count; index++)
    {
        unsigned int expected = SPARK_STAGE_MODULE_SLOT_FREE;
        if (atomic_compare_exchange_strong_explicit(
                &slot_states[index],
                &expected,
                SPARK_STAGE_MODULE_SLOT_CLAIMED,
                memory_order_acquire,
                memory_order_relaxed))
        {
            *slot_index = index;
            return SPARK_STATUS_OK;
        }
    }
    SPARK_FAIL(SPARK_STATUS_BUSY);
}

SparkStatus SparkStageModuleIndexSetClaim(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count)
{
    uint32_t claimed_count;

    if (index_states == 0 || indices == 0 || index_capacity == 0u ||
        index_count == 0u || index_count > index_capacity)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }

    for (claimed_count = 0u; claimed_count < index_count; claimed_count++)
    {
        unsigned int expected_state;
        uint32_t index;
        uint32_t previous_index;
        SparkStatus status;

        index = indices[claimed_count];
        if (index >= index_capacity)
        {
            SparkStageModuleIndexSetRelease(
                index_states,
                index_capacity,
                indices,
                claimed_count);
            SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
        }
        expected_state = SPARK_STAGE_MODULE_SLOT_FREE;
        if (!atomic_compare_exchange_strong_explicit(
                &index_states[index],
                &expected_state,
                claimed_count + 1u,
                memory_order_acquire,
                memory_order_relaxed))
        {
            status = SPARK_STATUS_BUSY;
            for (previous_index = 0u; previous_index < claimed_count; previous_index++)
            {
                if (indices[previous_index] == index)
                {
                    status = SPARK_STATUS_INVALID_ARGUMENT;
                    break;
                }
            }
            SparkStageModuleIndexSetRelease(
                index_states,
                index_capacity,
                indices,
                claimed_count);
            SPARK_RETURN(status);
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleIndexClaimOrdinal(
    const atomic_uint *index_states,
    uint32_t index_capacity,
    uint32_t index,
    uint32_t *ordinal_out)
{
    unsigned int claim;

    if (index_states == 0 || ordinal_out == 0 || index >= index_capacity)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    *ordinal_out = UINT32_MAX;
    claim = atomic_load_explicit(&index_states[index], memory_order_acquire);
    if (claim == SPARK_STAGE_MODULE_SLOT_FREE)
    {
        SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
    }
    if (claim > index_capacity)
    {
        SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
    }
    *ordinal_out = claim - 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkStageModuleIndexSetClaimAndPrepare(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count,
    SparkStageModuleClaimedIndexPrepareFunction prepare_function,
    void *prepare_context)
{
    SparkStatus status;

    if (prepare_function == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    status = SparkStageModuleIndexSetClaim(
        index_states,
        index_capacity,
        indices,
        index_count);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    status = prepare_function(prepare_context);
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleIndexSetRelease(
            index_states,
            index_capacity,
            indices,
            index_count);
    }
    SPARK_RETURN(status);
}

void SparkStageModuleIndexSetRelease(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count)
{
    uint32_t index_offset;

    if (index_states == 0 || indices == 0)
    {
        return;
    }
    for (index_offset = 0u; index_offset < index_count; index_offset++)
    {
        uint32_t index;

        index = indices[index_offset];
        if (index < index_capacity)
        {
            atomic_store_explicit(
                &index_states[index],
                SPARK_STAGE_MODULE_SLOT_FREE,
                memory_order_release);
        }
    }
}

void SparkStageModuleAtomicStateArrayInitialize(
    atomic_uint *states,
    uint32_t state_count)
{
    uint32_t state_index;

    if (states == 0)
    {
        return;
    }
    for (state_index = 0u; state_index < state_count; state_index++)
    {
        atomic_init(&states[state_index], SPARK_STAGE_MODULE_SLOT_FREE);
    }
}

uint32_t SparkStageModuleSlotAvailableCount(
    const atomic_uint *slot_states,
    uint32_t slot_count)
{
    uint32_t available_count;
    uint32_t index;

    if (slot_states == 0)
    {
        return 0u;
    }

    available_count = 0u;
    for (index = 0u; index < slot_count; index++)
    {
        if (atomic_load_explicit(&slot_states[index], memory_order_relaxed) ==
            SPARK_STAGE_MODULE_SLOT_FREE)
        {
            available_count++;
        }
    }
    return available_count;
}

uint32_t SparkStageModuleSlotCountFree(
    const atomic_uint *slot_states,
    uint32_t slot_count)
{
    return SparkStageModuleSlotAvailableCount(slot_states, slot_count);
}

void SparkStageModuleSlotRelease(
    atomic_uint *slot_states,
    uint32_t slot_index)
{
    if (slot_states == 0 || slot_index == SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT)
    {
        return;
    }
    atomic_store_explicit(
        &slot_states[slot_index],
        SPARK_STAGE_MODULE_SLOT_FREE,
        memory_order_release);
}
