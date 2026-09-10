#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct SparkTpDeviceCollectiveImplementation
{
    SparkWeightdClient *client;
    uint8_t *mesh_buffer;
    uint64_t band_base;
    uint64_t slot_bytes;
    uint64_t round_timeout_ns;
    uint64_t staging_ordinal;
    uint64_t staging_bytes;
    uint32_t registered;
    SparkTpDeviceCollectiveCombineBf16Function combine;
    void *combine_context;
    uint32_t tp_rank;
    uint32_t tp_degree;
    uint32_t local_hidden_dimension;
} SparkTpDeviceCollectiveImplementation;

static uint64_t SparkTpDeviceCollectiveTimeNs(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

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

static SparkStatus SparkTpDeviceCollectiveRunRound(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    uint64_t bytes;
    uint64_t ordinal;
    uint64_t deadline;
    uint64_t slot_bytes;
    uint64_t band_index;
    uint8_t *slot;
    uint64_t *entry;
    uint32_t peer;

    bytes = (uint64_t)submission->active_sequence_count *
        implementation->local_hidden_dimension * 2u;
    if ( bytes + 16u > implementation->slot_bytes )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    if ( implementation->registered == 0u ||
         operation_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 ||
         implementation->combine == 0 )
        return SPARK_STATUS_UNSUPPORTED;
    ordinal = submission->ordinal;
    deadline = SparkTpDeviceCollectiveTimeNs() +
        implementation->round_timeout_ns;
    slot_bytes = implementation->slot_bytes;
    band_index = implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES *
         SPARK_WEIGHTD_MESH_SLOTS_PER_BAND);
    slot = implementation->mesh_buffer + implementation->band_base +
        (uint64_t)implementation->tp_rank * slot_bytes;
    entry = (uint64_t *)(implementation->mesh_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
            implementation->tp_rank));
    implementation->staging_ordinal = ordinal + 1u;
    implementation->staging_bytes = bytes;
    __sync_synchronize();
    if ( cudaMemcpyAsync(slot,submission->local_device,(size_t)bytes,
            cudaMemcpyDeviceToHost,submission->cuda_stream) != 0 )
        return SPARK_STATUS_IO_ERROR;
    if ( cudaMemcpyAsync(&entry[1],&implementation->staging_bytes,8u,
            cudaMemcpyHostToHost,submission->cuda_stream) != 0 )
        return SPARK_STATUS_IO_ERROR;
    if ( cudaMemcpyAsync(&entry[0],&implementation->staging_ordinal,8u,
            cudaMemcpyHostToHost,submission->cuda_stream) != 0 )
        return SPARK_STATUS_IO_ERROR;
    for ( peer = 0u; peer < implementation->tp_degree - 1u; peer++ )
    {
        uint32_t peer_rank =
            peer < implementation->tp_rank ? peer : peer + 1u;
        volatile uint64_t *sequence = (volatile uint64_t *)
            (implementation->mesh_buffer + implementation->band_base +
            (uint64_t)peer_rank * slot_bytes + slot_bytes - 8u);
        while ( *sequence < ordinal + 1u )
        {
            struct timespec pause = {0,1000};
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
            {
                fprintf(stderr,
                    "MESH-SPIN-TIMEOUT rank=%u peer=%u want=%llu got=%llu bytes=%llu\n",
                    implementation->tp_rank,peer_rank,
                    (unsigned long long)(ordinal + 1u),
                    (unsigned long long)*sequence,
                    (unsigned long long)bytes);
                return SPARK_STATUS_BUSY;
            }
            nanosleep(&pause,0);
        }
    }
    if ( cudaMemcpyAsync(submission->full_device,slot,(size_t)bytes,
            cudaMemcpyHostToDevice,submission->cuda_stream) != 0 )
        return SPARK_STATUS_IO_ERROR;
    for ( peer = 0u; peer < implementation->tp_degree - 1u; peer++ )
    {
        uint8_t *source = implementation->mesh_buffer +
            implementation->band_base +
            (uint64_t)(peer < implementation->tp_rank ?
                peer : peer + 1u) * slot_bytes;
        if ( implementation->combine(implementation->combine_context,
                submission->full_device,source,
                submission->active_sequence_count,
                implementation->local_hidden_dimension,
                submission->cuda_stream) != SPARK_STATUS_OK )
            return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    const char *socket;

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
    socket = getenv("SPARK_WEIGHTD_SOCKET");
    if ( socket == 0 || socket[0] == '\0' )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    implementation = calloc(1u,sizeof(*implementation));
    if ( implementation == 0 )
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    implementation->tp_rank = config->tp_rank;
    implementation->tp_degree = config->tp_degree;
    implementation->local_hidden_dimension = config->local_hidden_dimension;
    implementation->band_base = (uint64_t)(config->collective_identifier &
        (SPARK_WEIGHTD_MESH_BANDS - 1u)) *
        SPARK_WEIGHTD_MESH_SLOT_BYTES *
        SPARK_WEIGHTD_MESH_SLOTS_PER_BAND;
    implementation->slot_bytes = SPARK_WEIGHTD_MESH_SLOT_BYTES;
    implementation->combine = config->combine_bf16_function;
    implementation->combine_context = config->combine_context;
    implementation->round_timeout_ns =
        (uint64_t)config->operation_timeout_milli * 1000000ull;
    if ( SparkWeightdClientConnect(socket,&implementation->client,0) !=
            SPARK_STATUS_OK )
    {
        free(implementation);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    collective_out->implementation = implementation;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveSubmitInternal(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    SparkTpDeviceCollectiveSubmission round;
    SparkStatus status;

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
    round = *submission;
    status = SparkTpDeviceCollectiveRunRound(implementation,&round,
        operation_kind);
    SparkTpDeviceCollectiveInvokeCompletion(&round,round.ordinal,status);
    return status;
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
    if ( implementation->registered == 0u )
    {
        if ( cudaHostRegister(receive_device,
                SPARK_WEIGHTD_MESH_REGION_BYTES,0u) != 0 )
            SPARK_FAIL(SPARK_STATUS_IO_ERROR);
        implementation->registered = 1u;
    }
    implementation->mesh_buffer = receive_device;
    return SPARK_STATUS_OK;
}

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;

    if ( collective == 0 || collective->implementation == 0 )
        return;
    implementation = collective->implementation;
    SparkWeightdClientClose(implementation->client);
    free(implementation);
    collective->implementation = 0;
}
