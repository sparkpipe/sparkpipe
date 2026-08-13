#pragma once

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#define SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS 4096u
#define SPARK_STAGE_MODULE_STAGING_CHUNK_BYTES (64ull * 1024ull * 1024ull)
#define SPARK_STAGE_MODULE_SLOT_FREE 0u
#define SPARK_STAGE_MODULE_SLOT_CLAIMED 1u
#define SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS 30000000000ull
#define SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES 2u

typedef struct SparkStageModuleLedger
{
    const char *module_tag;
    void *device_allocations[SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS];
    uint64_t device_allocation_bytes[SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS];
    uint32_t device_allocation_count;
    uint64_t device_bytes_resident;
} SparkStageModuleLedger;

typedef struct SparkStageModuleCudaFork
{
    cudaStream_t auxiliary_streams[SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES];
    cudaEvent_t fork_event;
    cudaEvent_t milestone_event;
    cudaEvent_t join_events[SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES];
} SparkStageModuleCudaFork;

typedef SparkStatus (*SparkStageModuleClaimedIndexPrepareFunction)(
    void *prepare_context);

SparkStatus SparkStageModuleCudaStatus(
    const char *module_tag,
    cudaError_t error,
    const char *site);
SparkStatus SparkStageModuleCudaForkInitialize(
    const char *module_tag,
    SparkStageModuleCudaFork *fork);
cudaError_t SparkStageModuleCudaForkBegin(
    SparkStageModuleCudaFork *fork,
    cudaStream_t primary_stream,
    uint32_t branch_count);
cudaError_t SparkStageModuleCudaForkJoin(
    SparkStageModuleCudaFork *fork,
    cudaStream_t primary_stream,
    uint32_t branch_count);
void SparkStageModuleCudaForkDestroy(SparkStageModuleCudaFork *fork);
SparkStatus SparkStageModuleEnvironmentText(
    const char *module_tag,
    const char *name,
    const char **value);
SparkStatus SparkStageModuleEnvironmentUnsigned(
    const char *module_tag,
    const char *name,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t *value);
SparkStatus SparkStageModuleEnvironmentUnsigned64(
    const char *module_tag,
    const char *name,
    uint64_t minimum,
    uint64_t maximum,
    uint64_t *value);
SparkStatus SparkStageModuleDeviceAllocate(
    SparkStageModuleLedger *ledger,
    uint64_t bytes,
    void **pointer);
SparkStatus SparkStageModuleDeviceAllocateZeroed(
    SparkStageModuleLedger *ledger,
    uint64_t bytes,
    void **pointer);
void SparkStageModuleLedgerRollback(
    SparkStageModuleLedger *ledger,
    uint32_t allocation_count);
void SparkStageModuleLedgerRelease(SparkStageModuleLedger *ledger);
SparkStatus SparkStageModulePackRead(
    const char *module_tag,
    FILE *file,
    uint64_t offset,
    void *destination,
    uint64_t bytes);
SparkStatus SparkStageModuleLoadDeviceRegion(
    SparkStageModuleLedger *ledger,
    FILE *file,
    uint64_t offset,
    uint64_t bytes,
    void **pointer);
void SparkStageModuleAdmissionDecisionInitialize(
    SparkModelDriverAdmissionDecision *decision,
    uint32_t available_dispatch_slot_count);
void SparkStageModuleAdmissionDecisionAccept(
    SparkModelDriverAdmissionDecision *decision);
void SparkStageModuleAdmissionDecisionReject(
    SparkModelDriverAdmissionDecision *decision,
    SparkModelDriverAdmissionRejection rejection_reason);
void SparkStageModuleRuntimeSnapshotInitialize(
    SparkModelDriverRuntimeSnapshot *snapshot,
    uint32_t program_id,
    const atomic_uint *slot_states,
    uint32_t slot_count);
SparkStatus SparkStageModuleSlotClaim(
    atomic_uint *slot_states,
    uint32_t slot_count,
    uint32_t *slot_index);
SparkStatus SparkStageModuleIndexSetClaim(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count);
SparkStatus SparkStageModuleIndexClaimOrdinal(
    const atomic_uint *index_states,
    uint32_t index_capacity,
    uint32_t index,
    uint32_t *ordinal_out);
/* Prepare runs after every index is claimed; failure releases the full set. */
SparkStatus SparkStageModuleIndexSetClaimAndPrepare(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count,
    SparkStageModuleClaimedIndexPrepareFunction prepare_function,
    void *prepare_context);
void SparkStageModuleIndexSetRelease(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count);
void SparkStageModuleAtomicStateArrayInitialize(
    atomic_uint *states,
    uint32_t state_count);
uint32_t SparkStageModuleSlotAvailableCount(
    const atomic_uint *slot_states,
    uint32_t slot_count);
uint32_t SparkStageModuleSlotCountFree(
    const atomic_uint *slot_states,
    uint32_t slot_count);
SparkStatus SparkStageModuleWaitForSlots(
    const char *module_tag,
    const atomic_uint *slot_states,
    uint32_t slot_count,
    uint64_t timeout_nanoseconds);
void SparkStageModuleSlotRelease(
    atomic_uint *slot_states,
    uint32_t slot_index);
/* The callback runs under both claims; the dispatch slot is released last. */
void SparkStageModuleCompleteAndReleaseClaims(
    SparkModelDriverCompletionFunction completion_function,
    void *completion_context,
    const SparkModelDriverCompletion *completion,
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count,
    atomic_uint *slot_states,
    uint32_t slot_index);
