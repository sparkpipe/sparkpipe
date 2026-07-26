#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_stage_module_common.h"

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

int main(void)
{
    SparkTestFailedIndexSetClaimPreservesForeignOwnership();
    SparkTestDuplicateIndexSetIsRejectedWithoutLeakingClaims();
    SparkTestSlotClaimAndRelease();
    SparkTestAllocationLedgerAccountsAndReleases();
    return 0;
}
