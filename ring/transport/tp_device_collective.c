#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define MESH_MAX_PEERS 15
#define MESH_SLOT_BYTES 1048576
#define MESH_SCRATCH_OFFSET 0
#define MESH_RECV_BASE MESH_SLOT_BYTES
#define MESH_SEQ_OFFSET(slot) (MESH_RECV_BASE + (slot) * MESH_SLOT_BYTES + MESH_SLOT_BYTES - 8)

extern SparkStatus SparkWeightdClientConnect(const char *path,
    void *client, uint64_t reserved);
extern SparkStatus SparkWeightdClientDisconnect(void *client);
extern SparkStatus SparkWeightdClientMeshWrite(void *client,
    uint32_t peer_rank, uint64_t source_offset, uint64_t remote_offset,
    uint32_t length, uint64_t timeout_nanoseconds);
extern SparkStatus SparkWeightdClientMeshBroadcast(void *client,
    uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset,
    uint32_t length, uint64_t timeout_nanoseconds);

__attribute__((weak)) SparkStatus SparkWeightdClientConnect(const char *path,
    void *client, uint64_t reserved)
{
    (void)path;(void)client;(void)reserved;
    return SPARK_STATUS_UNSUPPORTED;
}
__attribute__((weak)) SparkStatus SparkWeightdClientDisconnect(void *client)
{
    (void)client;
    return SPARK_STATUS_UNSUPPORTED;
}
__attribute__((weak)) SparkStatus SparkWeightdClientMeshWrite(void *client,
    uint32_t peer_rank, uint64_t source_offset, uint64_t remote_offset,
    uint32_t length, uint64_t timeout_nanoseconds)
{
    (void)client;(void)peer_rank;(void)source_offset;
    (void)remote_offset;(void)length;(void)timeout_nanoseconds;
    return SPARK_STATUS_UNSUPPORTED;
}
__attribute__((weak)) SparkStatus SparkWeightdClientMeshBroadcast(void *client,
    uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset,
    uint32_t length, uint64_t timeout_nanoseconds)
{
    (void)client;(void)peer_mask;(void)source_offset;
    (void)remote_offset;(void)length;(void)timeout_nanoseconds;
    return SPARK_STATUS_UNSUPPORTED;
}
extern uint32_t SparkWeightdMeshReady(void);

typedef struct SparkTpDeviceCollectiveImplementation
{
    char client[4096];
    void *mesh_buffer;
    uint32_t tp_rank;
    uint32_t tp_degree;
    uint32_t local_hidden_dimension;
    uint32_t max_active_sequence_count;
    uint64_t next_ordinal;
    SparkTpDeviceCollectiveCompletionFunction completion_function;
} SparkTpDeviceCollectiveImplementation;

SparkStatus SparkTpDeviceCollectiveProbeMemoryMode(
    uint32_t backend_kind,
    const char *backend_module_path,
    uint32_t *memory_mode_out)
{
    (void)backend_kind;(void)backend_module_path;
    if ( memory_mode_out == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *memory_mode_out = SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST;
    return SPARK_STATUS_OK;
}

void SparkTpDeviceCollectiveDumpOperations(
    const SparkTpDeviceCollective *collective)
{
    (void)collective;
}

SparkStatus SparkTpDeviceCollectiveCreditStepCount(
    uint32_t backend_kind,
    uint32_t tp_degree,
    uint32_t *step_count_out)
{
    (void)backend_kind;(void)tp_degree;
    if ( step_count_out == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *step_count_out = 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveCreditBindingRouteCount(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t *route_count_out)
{
    if ( config == 0 || route_count_out == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *route_count_out = 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveApplyTopology(
    const SparkTpDeviceCollectiveTopology *topology,
    SparkTpDeviceCollectiveConfig *config)
{
    if ( topology == 0 || config == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    config->tp_degree = topology->rank_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveSliceTopology(
    const SparkTpDeviceCollectiveTopology *source,
    uint32_t first_rank,
    uint32_t rank_count,
    SparkTpDeviceCollectiveTopology *destination)
{
    (void)first_rank;
    if ( source == 0 || destination == 0 || rank_count == 0u )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *destination = *source;
    destination->rank_count = rank_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    const char *socket;
    char path[128];

    if ( config == 0 || collective_out == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(collective_out,0,sizeof(*collective_out));
    collective_out->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    collective_out->backend_kind = config->backend_kind;
    collective_out->tp_degree = config->tp_degree;
    collective_out->tp_rank = config->tp_rank;
    collective_out->local_hidden_dimension = config->local_hidden_dimension;
    collective_out->max_active_sequence_count =
        config->max_active_sequence_count;
    collective_out->credit_count = 1u;
    collective_out->operation_timeout_milli = config->operation_timeout_milli;
    collective_out->memory_mode =
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST;
    implementation = calloc(1u,sizeof(*implementation));
    if ( implementation == 0 )
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    implementation->tp_rank = config->tp_rank;
    implementation->tp_degree = config->tp_degree;
    implementation->local_hidden_dimension = config->local_hidden_dimension;
    implementation->max_active_sequence_count =
        config->max_active_sequence_count;
    socket = getenv("SPARK_WEIGHTD_SOCKET");
    snprintf(path,sizeof(path),"%s",
        socket != 0 ? socket : "/tmp/spark_weightd.sock");
    if ( SparkWeightdClientConnect(path,implementation->client,0) !=
            SPARK_STATUS_OK )
    {
        free(implementation);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    collective_out->implementation = implementation;
    return SPARK_STATUS_OK;
}

static void SparkTpDeviceCollectiveInvokeCompletion(
    const SparkTpDeviceCollectiveSubmission *submission,
    uint64_t ordinal,
    SparkStatus status)
{
    SparkTpDeviceCollectiveCompletion completion;
    if ( submission->completion_function == 0 )
        return;
    memset(&completion,0,sizeof(completion));
    completion.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    completion.descriptor_bytes = sizeof(completion);
    completion.status = status;
    completion.slot_index = submission->slot_index;
    completion.ordinal = ordinal;
    completion.credit_index = 0u;
    completion.generation = ordinal + 1u;
    submission->completion_function(submission->completion_context,
        &completion);
}

static SparkStatus SparkTpDeviceCollectiveSubmitInternal(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t bytes;
    uint64_t ordinal;

    (void)operation_kind;
    if ( collective == 0 || collective->implementation == 0 ||
         submission == 0 || submission->local_device == 0 ||
         submission->full_device == 0 || submission->cuda_stream == 0 ||
         submission->completion_function == 0 ||
         submission->active_sequence_count == 0u ||
         submission->ordinal == UINT64_MAX )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    bytes = (uint64_t)submission->active_sequence_count *
        collective->local_hidden_dimension * 2u;
    if ( bytes > MESH_SLOT_BYTES )
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    ordinal = submission->ordinal;
    memcpy(implementation->mesh_buffer,submission->local_device,
        (size_t)bytes);
    {
        uint64_t *seq_slot = (uint64_t *)((uint8_t *)
            implementation->mesh_buffer + MESH_SLOT_BYTES - 8);
        *seq_slot = ordinal + 1u;
    }
    {
        uint32_t peer;
        uint32_t peer_rank;
        uint64_t remote_base;
        for (peer = 0u; peer < collective->tp_degree - 1u; peer++)
        {
            peer_rank = peer < collective->tp_rank ? peer : peer + 1u;
            remote_base = (uint64_t)(collective->tp_rank < peer_rank ?
                collective->tp_rank : collective->tp_rank - 1u) *
                (uint64_t)bytes;
            SparkStatus ws = SparkWeightdClientMeshWrite(
                implementation->client,
                peer_rank,
                MESH_SCRATCH_OFFSET,
                remote_base,
                (uint32_t)bytes,
                (uint64_t)collective->operation_timeout_milli * 1000000ull);
            if ( ws != SPARK_STATUS_OK )
                SPARK_RETURN(ws);
        }
    }
    {
        volatile uint64_t *seq;
        uint32_t peer;
        for (peer = 0u; peer < collective->tp_degree - 1u; peer++)
        {
            seq = (volatile uint64_t *)((uint8_t *)
                implementation->mesh_buffer +
                (uint64_t)(peer + 1u) * (uint64_t)bytes);
            while ( *seq < ordinal + 1u )
            {
                struct timespec pause = {0,100000};
                nanosleep(&pause,0);
            }
        }
    }
    {
        uint16_t *local = (uint16_t *)implementation->mesh_buffer;
        uint16_t *result = (uint16_t *)submission->full_device;
        uint32_t peer;
        uint32_t i;
        uint32_t count = (uint32_t)(bytes / 2u);
        memcpy(result,local,(size_t)bytes);
        for (peer = 0u; peer < collective->tp_degree - 1u; peer++)
        {
            uint16_t *peer_data = (uint16_t *)((uint8_t *)
                implementation->mesh_buffer +
                (uint64_t)(peer + 1u) * (uint64_t)bytes);
            for (i = 0u; i < count; i++)
            {
                int32_t a = (int32_t)(int16_t)result[i];
                int32_t b = (int32_t)(int16_t)peer_data[i];
                int32_t sum = a + b;
                result[i] = (uint16_t)((sum > 32767 ? 32767 :
                    sum < -32768 ? -32768 : sum) & 0xFFFF);
            }
        }
    }
    SparkTpDeviceCollectiveInvokeCompletion(submission,
        ordinal,SPARK_STATUS_OK);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveSubmitBf16(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission)
{
    return SparkTpDeviceCollectiveSubmitInternal(collective,submission,
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
}

SparkStatus SparkTpDeviceCollectiveSubmitU64Max(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission)
{
    return SparkTpDeviceCollectiveSubmitInternal(collective,submission,
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
}

SparkStatus SparkTpDeviceCollectiveEnqueue(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    return SparkTpDeviceCollectiveSubmitInternal(collective,submission,
        operation_kind);
}

SparkStatus SparkTpDeviceCollectiveWaitAllRoutes(
    SparkTpDeviceCollective *collective,
    uint32_t timeout_milli)
{
    (void)timeout_milli;
    if ( collective == 0 || collective->implementation == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveOpWaitHandles(
    SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    void **flag_device,
    uint64_t *wait_value)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    if ( collective == 0 || collective->implementation == 0 ||
         flag_device == 0 || wait_value == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    *flag_device = (uint8_t *)implementation->mesh_buffer +
        MESH_SCRATCH_OFFSET;
    *wait_value = ordinal + 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveRequestFailure(
    SparkTpDeviceCollective *collective,
    SparkStatus failure_status)
{
    (void)collective;(void)failure_status;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveRequestOperationFailure(
    SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    SparkStatus failure_status)
{
    (void)collective;(void)ordinal;(void)failure_status;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveOperationPhase(
    const SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    uint32_t *phase_out,
    uint32_t *failure_requested_out)
{
    (void)ordinal;
    if ( collective == 0 || phase_out == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    *phase_out = 3u;
    if ( failure_requested_out != 0 )
        *failure_requested_out = 0u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveExchangeBf16(
    SparkTpDeviceCollective *collective,
    const void *send_device,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    (void)step_index;
    if ( collective == 0 || collective->implementation == 0 ||
         send_device == 0 || receive_device == 0 || cuda_stream == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    memcpy(implementation->mesh_buffer,send_device,
        (size_t)active_sequence_count * hidden_dimension * 2u);
    return SparkWeightdClientMeshBroadcast(implementation->client,
        0x7FFFu,MESH_SCRATCH_OFFSET,MESH_SCRATCH_OFFSET,
        active_sequence_count * hidden_dimension * 2u,
        (uint64_t)collective->operation_timeout_milli * 1000000ull);
}

SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(
    SparkTpDeviceCollective *collective,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    (void)active_sequence_count;(void)hidden_dimension;
    (void)step_index;(void)cuda_stream;
    if ( collective == 0 || collective->implementation == 0 ||
         receive_device == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    implementation->mesh_buffer = receive_device;
    return SPARK_STATUS_OK;
}

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    if ( collective == 0 || collective->implementation == 0 )
        return;
    implementation = collective->implementation;
    (void)SparkWeightdClientDisconnect(implementation->client);
    free(implementation);
    collective->implementation = 0;
}
