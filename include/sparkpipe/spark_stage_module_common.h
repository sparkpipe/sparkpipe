#pragma once

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"

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

#define SPARK_STAGE_MODULE_STAGE_TIMING_SAMPLE_CAPACITY 96u
#define SPARK_STAGE_MODULE_STAGE_TIMING_STAGE_CAPACITY 32u

typedef struct SparkStageModuleStageTimingSample
{
    cudaEvent_t start;
    cudaEvent_t stop;
    uint32_t stage;
} SparkStageModuleStageTimingSample;

typedef struct SparkStageModuleStageTiming
{
    const char *record_tag;
    const char *const *stage_names;
    uint32_t stage_count;
    uint32_t enabled;
    uint32_t events_ready;
    uint32_t sample_count;
    SparkStageModuleStageTimingSample samples[SPARK_STAGE_MODULE_STAGE_TIMING_SAMPLE_CAPACITY];
    uint64_t frame_microseconds[SPARK_STAGE_MODULE_STAGE_TIMING_STAGE_CAPACITY];
    uint32_t frame_calls[SPARK_STAGE_MODULE_STAGE_TIMING_STAGE_CAPACITY];
    uint64_t total_microseconds[SPARK_STAGE_MODULE_STAGE_TIMING_STAGE_CAPACITY];
    uint64_t total_calls[SPARK_STAGE_MODULE_STAGE_TIMING_STAGE_CAPACITY];
    uint64_t frame_index;
} SparkStageModuleStageTiming;

SparkStatus SparkStageModuleStageTimingEnable(
    SparkStageModuleStageTiming *timing,
    const char *record_tag,
    const char *const *stage_names,
    uint32_t stage_count);
SparkStatus SparkStageModuleStageTimingFrameBegin(
    SparkStageModuleStageTiming *timing);
void SparkStageModuleStageTimingBegin(
    SparkStageModuleStageTiming *timing,
    cudaStream_t stream,
    uint32_t stage);
void SparkStageModuleStageTimingEnd(
    SparkStageModuleStageTiming *timing,
    cudaStream_t stream);
void SparkStageModuleStageTimingFold(
    SparkStageModuleStageTiming *timing);
void SparkStageModuleStageTimingShutdown(
    SparkStageModuleStageTiming *timing);

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
SparkStatus SparkStageModuleEnvironmentUnsigned64OrDefault(
    const char *module_tag,
    const char *name,
    uint64_t minimum,
    uint64_t maximum,
    uint64_t fallback,
    uint64_t *value);

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

typedef struct SparkStageModuleClaimedLaneContext
{
	const atomic_uint *index_states;
	uint32_t index_capacity;
} SparkStageModuleClaimedLaneContext;

static inline SparkStatus SparkStageModuleClaimedLaneOrdinal(
	void *context,
	uint32_t lane_id,
	uint32_t *ordinal_out)
{
	const SparkStageModuleClaimedLaneContext *lanes;
	lanes = (const SparkStageModuleClaimedLaneContext *)context;
	if ( lanes == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleIndexClaimOrdinal(lanes->index_states,lanes->index_capacity,lane_id,ordinal_out));
}

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
uint64_t SparkStageModuleFingerprint(
    const void *bytes,
    uint64_t count,
    uint64_t basis);
SparkStatus SparkStageModulePackFileSize(
    FILE *file,
    uint64_t *bytes);
void SparkStageModuleTpCompletionFlag(
    void *context,
    const SparkTpDeviceCollectiveCompletion *completion);
void SparkStageModuleAdmissionCost(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);
#define SPARK_STAGE_MODULE_TP_CHAIN_COMPLETION(function_name, chain_type, \
    advance_function) \
    static void function_name(void *context, \
        const SparkTpDeviceCollectiveCompletion *completion) \
    { \
        chain_type *chain; \
        chain = (chain_type *)context; \
        if ( chain == 0 || chain->active == 0u || completion == 0 ) \
            return; \
        advance_function(chain,completion->status); \
    }
