#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_stage_module_common.h"

typedef struct SparkTestClaimContext
{
    atomic_uint *states;
    const uint32_t *indices;
    uint32_t index_capacity;
    uint32_t index_count;
    uint32_t call_count;
    SparkStatus result;
} SparkTestClaimContext;

typedef struct SparkTestCompletionContext
{
    atomic_uint *lane_states;
    atomic_uint *slot_states;
    const uint32_t *lane_indices;
    uint32_t lane_count;
    uint32_t slot_index;
    uint32_t call_count;
} SparkTestCompletionContext;

static SparkStatus SparkTestPrepareClaimedIndices(void *prepare_context)
{
    SparkTestClaimContext *context;
    uint32_t index,ordinal;

    context = (SparkTestClaimContext *)prepare_context;
    context->call_count++;
    for (index = 0u; index < context->index_count; index++)
    {
        assert(atomic_load_explicit(
                   &context->states[context->indices[index]],
                   memory_order_acquire) == index + 1u);
        assert(SparkStageModuleIndexClaimOrdinal(
                   context->states,
                   context->index_capacity,
                   context->indices[index],
                   &ordinal) == SPARK_STATUS_OK);
        assert(ordinal == index);
    }
    return context->result;
}

static void SparkTestCompleteWhileClaimsAreHeld(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkTestCompletionContext *context;
    uint32_t lane,ordinal,slot_index;

    context = (SparkTestCompletionContext *)completion_context;
    assert(completion != 0);
    assert(completion->request_id == UINT64_C(176000));
    for (lane = 0u; lane < context->lane_count; lane++)
    {
        assert(SparkStageModuleIndexClaimOrdinal(
                   context->lane_states,
                   context->lane_count,
                   context->lane_indices[lane],
                   &ordinal) == SPARK_STATUS_OK);
        assert(ordinal == lane);
    }
    assert(atomic_load_explicit(
               &context->slot_states[context->slot_index],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_CLAIMED);
    assert(SparkStageModuleIndexSetClaim(
               context->lane_states,
               context->lane_count,
               context->lane_indices,
               context->lane_count) == SPARK_STATUS_BUSY);
    assert(SparkStageModuleSlotClaim(
               context->slot_states,
               1u,
               &slot_index) == SPARK_STATUS_BUSY);
    context->call_count++;
}

static void SparkTestFailedIndexSetClaimPreservesForeignOwnership(void)
{
    atomic_uint lane_states[3];
    const uint32_t first_owner_lanes[] = {1u};
    const uint32_t second_owner_lanes[] = {0u, 1u};

    SparkStageModuleAtomicStateArrayInitialize(lane_states, 3u);
    assert(SparkStageModuleIndexSetClaim(
               lane_states,
               3u,
               first_owner_lanes,
               1u) == SPARK_STATUS_OK);
    assert(SparkStageModuleIndexSetClaim(
               lane_states,
               3u,
               second_owner_lanes,
               2u) == SPARK_STATUS_BUSY);
    assert(atomic_load_explicit(
               &lane_states[0],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(atomic_load_explicit(
               &lane_states[1],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_CLAIMED);
    SparkStageModuleIndexSetRelease(
        lane_states,
        3u,
        first_owner_lanes,
        1u);
    assert(atomic_load_explicit(
               &lane_states[1],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
}

static void SparkTestDuplicateIndexSetIsRejectedWithoutLeakingClaims(void)
{
    atomic_uint lane_states[2];
    const uint32_t duplicate_lanes[] = {0u, 0u};

    SparkStageModuleAtomicStateArrayInitialize(lane_states, 2u);
    assert(SparkStageModuleIndexSetClaim(
               lane_states,
               2u,
               duplicate_lanes,
               2u) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(atomic_load_explicit(
               &lane_states[0],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(SparkStageModuleSlotAvailableCount(lane_states, 2u) == 2u);
}

static void SparkTestSlotClaimAndRelease(void)
{
    atomic_uint slot_states[2];
    uint32_t slot_index;

    SparkStageModuleAtomicStateArrayInitialize(slot_states, 2u);
    assert(SparkStageModuleSlotClaim(
               slot_states,
               2u,
               &slot_index) == SPARK_STATUS_OK);
    assert(slot_index < 2u);
    assert(SparkStageModuleSlotAvailableCount(slot_states, 2u) == 1u);
    SparkStageModuleSlotRelease(slot_states, slot_index);
    assert(SparkStageModuleSlotAvailableCount(slot_states, 2u) == 2u);
}

static void SparkTestClaimedPrepareOwnsAndUnwindsIndices(void)
{
    atomic_uint lane_states[3];
    const uint32_t lane_indices[] = {0u, 2u};
    SparkTestClaimContext context;

    SparkStageModuleAtomicStateArrayInitialize(lane_states, 3u);
    context.states = lane_states;
    context.indices = lane_indices;
    context.index_capacity = 3u;
    context.index_count = 2u;
    context.call_count = 0u;
    context.result = SPARK_STATUS_SCHEMA_ERROR;
    assert(SparkStageModuleIndexSetClaimAndPrepare(
               lane_states,
               3u,
               lane_indices,
               2u,
               SparkTestPrepareClaimedIndices,
               &context) == SPARK_STATUS_SCHEMA_ERROR);
    assert(context.call_count == 1u);
    assert(atomic_load_explicit(
               &lane_states[0],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(atomic_load_explicit(
               &lane_states[2],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(SparkStageModuleSlotAvailableCount(lane_states, 3u) == 3u);
    context.result = SPARK_STATUS_OK;
    assert(SparkStageModuleIndexSetClaimAndPrepare(
               lane_states,
               3u,
               lane_indices,
               2u,
               SparkTestPrepareClaimedIndices,
               &context) == SPARK_STATUS_OK);
    assert(context.call_count == 2u);
    assert(SparkStageModuleSlotAvailableCount(lane_states, 3u) == 1u);
    SparkStageModuleIndexSetRelease(lane_states, 3u, lane_indices, 2u);
    assert(SparkStageModuleSlotAvailableCount(lane_states, 3u) == 3u);
}

static void SparkTestCompletionHoldsClaimsThroughCallback(void)
{
    atomic_uint lane_states[2],slot_states[1];
    const uint32_t lane_indices[] = {0u, 1u};
    SparkModelDriverCompletion completion = {0};
    SparkTestCompletionContext context;
    uint32_t slot_index;

    SparkStageModuleAtomicStateArrayInitialize(lane_states, 2u);
    SparkStageModuleAtomicStateArrayInitialize(slot_states, 1u);
    assert(SparkStageModuleIndexSetClaim(
               lane_states,
               2u,
               lane_indices,
               2u) == SPARK_STATUS_OK);
    assert(SparkStageModuleSlotClaim(
               slot_states,
               1u,
               &slot_index) == SPARK_STATUS_OK);
    context.lane_states = lane_states;
    context.slot_states = slot_states;
    context.lane_indices = lane_indices;
    context.lane_count = 2u;
    context.slot_index = slot_index;
    context.call_count = 0u;
    completion.request_id = UINT64_C(176000);
    SparkStageModuleCompleteAndReleaseClaims(
        SparkTestCompleteWhileClaimsAreHeld,
        &context,
        &completion,
        lane_states,
        2u,
        lane_indices,
        2u,
        slot_states,
        slot_index);
    assert(context.call_count == 1u);
    assert(atomic_load_explicit(
               &lane_states[0],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(atomic_load_explicit(
               &lane_states[1],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(atomic_load_explicit(
               &slot_states[0],
               memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE);
    assert(SparkStageModuleSlotAvailableCount(lane_states, 2u) == 2u);
    assert(SparkStageModuleSlotAvailableCount(slot_states, 1u) == 1u);
}

static void SparkTestAllocationLedgerAccountsAndReleases(void)
{
    SparkStageModuleLedger ledger = {0};
    void *allocation;

    ledger.module_tag = "stage_common_test";
    allocation = 0;
    assert(SparkStageModuleDeviceAllocateZeroed(
               &ledger,
               4096u,
               &allocation) == SPARK_STATUS_OK);
    assert(allocation != 0);
    assert(ledger.device_allocation_count == 1u);
    assert(ledger.device_bytes_resident == 4096u);
    SparkStageModuleLedgerRelease(&ledger);
    assert(ledger.device_allocation_count == 0u);
    assert(ledger.device_bytes_resident == 0u);
}

static void SparkTestCudaForkOwnsReusableResources(void)
{
    SparkStageModuleCudaFork fork = {0};
    assert(SparkStageModuleCudaForkInitialize(
               "stage_common_test", &fork) == SPARK_STATUS_OK);
    assert(fork.auxiliary_streams[0] != 0);
    assert(fork.auxiliary_streams[1] != 0);
    assert(fork.fork_event != 0);
    assert(fork.milestone_event != 0);
    assert(fork.join_events[0] != 0);
    assert(fork.join_events[1] != 0);
    assert(SparkStageModuleCudaForkInitialize(
               "stage_common_test", &fork) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkStageModuleCudaForkDestroy(&fork);
    assert(fork.auxiliary_streams[0] == 0);
    assert(fork.auxiliary_streams[1] == 0);
    assert(fork.fork_event == 0);
    assert(fork.milestone_event == 0);
    assert(fork.join_events[0] == 0);
    assert(fork.join_events[1] == 0);
}

int main(void)
{
    SparkTestFailedIndexSetClaimPreservesForeignOwnership();
    SparkTestDuplicateIndexSetIsRejectedWithoutLeakingClaims();
    SparkTestSlotClaimAndRelease();
    SparkTestClaimedPrepareOwnsAndUnwindsIndices();
    SparkTestCompletionHoldsClaimsThroughCallback();
    SparkTestAllocationLedgerAccountsAndReleases();
    SparkTestCudaForkOwnsReusableResources();
    return 0;
}
