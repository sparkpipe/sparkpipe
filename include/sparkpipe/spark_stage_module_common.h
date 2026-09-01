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
#define SPARK_STAGE_MODULE_CUDA_READ_AHEAD_IDLE 0u
#define SPARK_STAGE_MODULE_CUDA_READ_AHEAD_BUILDING 1u
#define SPARK_STAGE_MODULE_CUDA_READ_AHEAD_ARMED 2u

typedef struct SparkStageModuleLedger
{
    const char *module_tag;
    void *device_allocations[SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS];
    uint64_t device_allocation_bytes[SPARK_STAGE_MODULE_MAX_DEVICE_ALLOCATIONS];
    uint32_t device_allocation_count;
    uint64_t device_bytes_resident;
    /* The weightd pack arena (structural residency): opaque
     * SparkStageModulePackArena, lazily attached on the first device
     * region load when SPARK_WEIGHTD_* names a live daemon. Regions
     * served from the arena are slices of the consumer-side VMM map -
     * never entered into device_allocations, released by unmap at
     * LedgerRelease while the daemon keeps the arena warm for the next
     * code-only attach. Null when unset or after a clean fallback. */
    void *pack_arena;
} SparkStageModuleLedger;

typedef struct SparkStageModuleCudaFork
{
    cudaStream_t auxiliary_streams[SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES];
    cudaEvent_t fork_event;
    cudaEvent_t milestone_event;
    cudaEvent_t join_events[SPARK_STAGE_MODULE_CUDA_FORK_MAX_BRANCHES];
} SparkStageModuleCudaFork;

typedef struct SparkStageModuleCudaReadAhead
{
    cudaStream_t stream;
    cudaEvent_t source_ready_event;
    cudaEvent_t completion_event;
    uint32_t *sink_u32;
    uint32_t sink_word_capacity;
    atomic_uint state;
} SparkStageModuleCudaReadAhead;

typedef struct SparkStageModuleLoadPipeline SparkStageModuleLoadPipeline;

typedef cudaError_t (*SparkStageModuleCudaReadAheadLaunchFunction)(
    cudaStream_t stream,
    uint32_t *sink_u32,
    uint32_t sink_word_capacity,
    void *launch_context);

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
SparkStatus SparkStageModuleCudaReadAheadInitialize(
    const char *module_tag,
    SparkStageModuleLedger *ledger,
    SparkStageModuleCudaReadAhead *read_ahead,
    uint32_t sink_word_capacity);
SparkStatus SparkStageModuleCudaReadAheadArm(
    const char *module_tag,
    SparkStageModuleCudaReadAhead *read_ahead,
    cudaStream_t primary_stream,
    SparkStageModuleCudaReadAheadLaunchFunction launch_function,
    void *launch_context);
SparkStatus SparkStageModuleCudaReadAheadJoin(
    const char *module_tag,
    SparkStageModuleCudaReadAhead *read_ahead,
    cudaStream_t primary_stream);
void SparkStageModuleCudaReadAheadDestroy(
    SparkStageModuleCudaReadAhead *read_ahead);
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
/* The optional counterpart: an absent/empty variable yields the named
 * fallback without a config_missing line; a present variable is parsed and
 * range-checked with the same fail-loud diagnostics as the required read. */
SparkStatus SparkStageModuleEnvironmentUnsignedOrDefault(
    const char *module_tag,
    const char *name,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t fallback,
    uint32_t *value);
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
/* Pipelined pack loading (docs/WEIGHTD_DESIGN.md L1): a worker thread
 * reads pack regions into a two-slot ring of pinned host staging while
 * the calling thread issues the H2D copies onto one internal stream.
 * Regions may be enqueued back to back (a whole pack directory), and the
 * copies are issued strictly in enqueue order, so the device bytes and
 * their arrival order match the synchronous loader bit for bit - only
 * the host read and the device copy now overlap. Every pointer returned
 * by Region becomes safe to use only after Finish reports OK; Finish
 * issues the remaining copies and blocks until the stream drains. */
SparkStatus SparkStageModuleLoadPipelineRequested(void);
SparkStatus SparkStageModuleLoadPipelineCreate(
    const char *module_tag,
    FILE *file,
    SparkStageModuleLoadPipeline **pipeline);
SparkStatus SparkStageModuleLoadPipelineRegion(
    SparkStageModuleLoadPipeline *pipeline,
    SparkStageModuleLedger *ledger,
    uint64_t offset,
    uint64_t bytes,
    void **pointer);
SparkStatus SparkStageModuleLoadPipelineFinish(
    SparkStageModuleLoadPipeline *pipeline);
void SparkStageModuleLoadPipelineDestroy(
    SparkStageModuleLoadPipeline *pipeline);
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
