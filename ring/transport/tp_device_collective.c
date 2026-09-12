#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define SPARK_TP_CUDA_MEMCPY_HOST_TO_HOST 0
#define SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE 1
#define SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST 2
extern int cudaGetLastError(void);
extern const char *cudaGetErrorString(int error);
extern int cudaMemcpyAsync(void *destination,const void *source,
    size_t bytes,int kind,void *stream);
extern int cudaHostRegister(void *address,size_t bytes,unsigned int flags);
extern int cudaMemcpy(void *destination,const void *source,
    size_t bytes,int kind);
extern int cudaMalloc(void **address,size_t bytes);
extern int SparkGlm5NextLaunchMeshPublish(void *stream,
    volatile void *entry,void *seq_cell,void *round_seq,uint64_t bytes,
    uint64_t slot_index);
extern int SparkGlm5NextLaunchMeshWait(void *stream,
    volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint64_t ring,uint32_t rank,uint32_t degree,
    void *error_word,unsigned long long deadline_ns);

#define SPARK_TP_DEVICE_COLLECTIVE_STAGING_SETS \
    (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK * 16u)
typedef struct SparkTpDeviceCollectiveStagingSet
{
    uint64_t seq;
    uint64_t bytes;
    uint64_t slot;
} SparkTpDeviceCollectiveStagingSet;

typedef struct SparkTpDeviceCollectiveCompletionNode
{
    struct SparkTpDeviceCollectiveCompletionNode *next;
    struct SparkTpDeviceCollectiveImplementation *implementation;
    SparkTpDeviceCollectiveSubmission submission;
    uint64_t ordinal;
    SparkStatus status;
} SparkTpDeviceCollectiveCompletionNode;

typedef struct SparkTpDeviceCollectiveImplementation
{
    SparkWeightdClient *client;
    uint8_t *mesh_buffer;
    uint64_t band_base;
    uint64_t slot_bytes;
    uint64_t round_timeout_ns;
    uint64_t round_seq;
    uint64_t base_seen;
    uint64_t round_wave_limit;
    uint64_t chain_key;
    uint64_t chain_epoch;
    uint64_t round_index;
    uint64_t cancel_epoch;
    SparkTpDeviceCollectiveStagingSet
        staging[SPARK_TP_DEVICE_COLLECTIVE_STAGING_SETS];
    void *seq_cell;
    void *round_seq_device;
    void *error_word;
    uint32_t capture_armed;
    uint32_t round_rebased;
    uint64_t cancel_seen;
    uint32_t round_deadline_ms;
    pthread_mutex_t completion_lock;
    pthread_cond_t completion_wake;
    SparkTpDeviceCollectiveCompletionNode *completion_head;
    SparkTpDeviceCollectiveCompletionNode *completion_tail;
    SparkTpDeviceCollectiveCombineBf16Function combine_bf16;
    SparkTpDeviceCollectiveCombineU64MaxFunction combine_u64_max;
    void *combine_context;
    uint32_t tp_rank;
    uint32_t tp_degree;
    uint32_t local_hidden_dimension;
} SparkTpDeviceCollectiveImplementation;

static uint64_t SparkTpDeviceCollectiveTimeNs(void)
{
    struct timespec now;
    if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
        return(0ull);
    return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static void *SparkTpDeviceCollectiveRegisteredRegion;

static void SparkTpDeviceCollectiveInvokeCompletion(
    const SparkTpDeviceCollectiveSubmission *submission,
    uint64_t ordinal,
    SparkStatus status);

static void SparkTpDeviceCollectiveQueueCompletion(
    SparkTpDeviceCollectiveImplementation *implementation,
    const SparkTpDeviceCollectiveSubmission *submission,
    uint64_t ordinal,
    SparkStatus status)
{
    SparkTpDeviceCollectiveCompletionNode *node = calloc(1u,sizeof(*node));
    if ( node == 0 )
        return;
    node->implementation = implementation;
    node->submission = *submission;
    node->ordinal = ordinal;
    node->status = status;
    pthread_mutex_lock(&implementation->completion_lock);
    if ( implementation->completion_tail == 0 )
        implementation->completion_head = node;
    else
        implementation->completion_tail->next = node;
    implementation->completion_tail = node;
    pthread_cond_signal(&implementation->completion_wake);
    pthread_mutex_unlock(&implementation->completion_lock);
}

static void *SparkTpDeviceCollectiveCompletionThread(void *argument)
{
    SparkTpDeviceCollectiveImplementation *implementation =
        (SparkTpDeviceCollectiveImplementation *)argument;
    for (;;)
    {
        SparkTpDeviceCollectiveCompletionNode *node;
        pthread_mutex_lock(&implementation->completion_lock);
        while (implementation->completion_head == 0)
            pthread_cond_wait(&implementation->completion_wake,
                &implementation->completion_lock);
        node = implementation->completion_head;
        implementation->completion_head = node->next;
        if (implementation->completion_head == 0)
            implementation->completion_tail = 0;
        pthread_mutex_unlock(&implementation->completion_lock);
        SparkTpDeviceCollectiveInvokeCompletion(&node->submission,
            node->ordinal,node->status);
        free(node);
    }
    return 0;
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

#define SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS 10u
#define SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE \
    (1ull << SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS)
#define SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS (5ull * 1000000000ull)

static uint64_t SparkTpDeviceCollectiveBaseCellOffset(uint32_t band_index)
{
    return SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (100u + 2u * band_index) * 24u;
}

static uint64_t SparkTpDeviceCollectiveCancelCellOffset(uint32_t band_index)
{
    return SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (101u + 2u * band_index) * 24u;
}

static SparkStatus SparkTpDeviceCollectiveRebase(
    SparkTpDeviceCollectiveImplementation *implementation,
    uint32_t use_broadcast)
{
    uint32_t band_index = (uint32_t)(implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES *
         SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
    uint64_t base_offset = SparkTpDeviceCollectiveBaseCellOffset(band_index);
    volatile uint64_t *base_cell = (volatile uint64_t *)
        (implementation->mesh_buffer + base_offset);
    volatile uint64_t *cancel_cell = (volatile uint64_t *)
        (implementation->mesh_buffer +
        SparkTpDeviceCollectiveCancelCellOffset((uint32_t)(
            implementation->band_base / (SPARK_WEIGHTD_MESH_SLOT_BYTES *
                SPARK_WEIGHTD_MESH_SLOTS_PER_BAND))));
    uint64_t deadline = SparkTpDeviceCollectiveTimeNs() +
        implementation->round_timeout_ns;
    implementation->cancel_seen = *cancel_cell;
    if ( implementation->tp_rank == 0u )
    {
        uint32_t peer_rank;
        uint64_t high = implementation->round_seq;
        uint64_t base;
        if ( *base_cell > high )
            high = *base_cell;
        for ( peer_rank = 0u;
              peer_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND;
              peer_rank++ )
        {
            volatile uint64_t *peer_entry = (volatile uint64_t *)
                (implementation->mesh_buffer +
                SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
                    peer_rank));
            if ( peer_entry[0] > high )
                high = peer_entry[0];
        }
        base = ((high / SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE) + 1ull) *
            SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
        *base_cell = base;
        if ( use_broadcast != 0u )
        {
            __sync_synchronize();
            (void)SparkWeightdClientMeshBroadcast(
                implementation->client,
                0xffffu & ~(1u << implementation->tp_rank),
                base_offset,base_offset,8u,0ull,0ull,
                implementation->round_timeout_ns);
        }
        implementation->base_seen = base;
    }
    else
    {
        uint64_t marker = implementation->base_seen;
        while ( *base_cell == marker )
        {
            struct timespec pause = {0,1000};
            if ( *cancel_cell != implementation->cancel_seen )
                return SPARK_STATUS_BUSY;
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
                return SPARK_STATUS_BUSY;
            nanosleep(&pause,0);
        }
        implementation->base_seen = *base_cell;
    }
    implementation->round_seq = implementation->base_seen - 1ull;
    implementation->round_wave_limit = implementation->base_seen +
        SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
    implementation->round_rebased = 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveChainKey(
    SparkTpDeviceCollective *collective,uint64_t request_id)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint32_t band_index;
    volatile uint64_t *base_cell;
    uint64_t epoch;
    SparkStatus status;
    if ( collective == 0 || collective->implementation == 0 ||
         request_id > SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 ||
         implementation->capture_armed != 0u )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    band_index = (uint32_t)(implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES *
         SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
    base_cell = (volatile uint64_t *)
        (implementation->mesh_buffer +
        SparkTpDeviceCollectiveBaseCellOffset(band_index));
    if ( implementation->round_rebased == 0u || *base_cell == 0ull )
    {
        status = SparkTpDeviceCollectiveRebase(implementation,1u);
        if ( status != SPARK_STATUS_OK )
            return status;
    }
    epoch = *base_cell >> SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS;
    if ( epoch == 0ull ||
         epoch > SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK )
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    implementation->chain_epoch = epoch;
    implementation->chain_key =
        (epoch << SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS) | request_id;
    implementation->round_index = 0ull;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveRunRound(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    uint64_t bytes;
    uint64_t ordinal;
    uint64_t round_seq;
    uint64_t deadline;
    uint64_t slot_bytes;
    uint64_t slot_index;
    uint64_t slot_base;
    uint8_t *slot;
    volatile uint64_t *entry;
    SparkTpDeviceCollectiveStagingSet *staging;
    uint32_t band_index;
    uint32_t peer;

    if ( operation_kind ==
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
        bytes = (uint64_t)submission->active_sequence_count * 8u;
    else
        bytes = (uint64_t)submission->active_sequence_count *
            implementation->local_hidden_dimension * 2u;
    if ( bytes + 16u > implementation->slot_bytes )
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    if ( SparkTpDeviceCollectiveRegisteredRegion == 0 ||
         (operation_kind ==
                SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 &&
            implementation->combine_bf16 == 0) ||
         (operation_kind ==
                SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 &&
            implementation->combine_u64_max == 0) )
        return SPARK_STATUS_UNSUPPORTED;
    if ( implementation->chain_key != 0ull )
    {
        if ( implementation->round_index >=
                (1ull << SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ROUND_BITS) )
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        round_seq = (implementation->chain_key <<
            SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ROUND_BITS) |
            implementation->round_index;
        implementation->round_index++;
    }
    else
    {
        if ( implementation->round_rebased == 0u )
        {
            SparkStatus rebase = SparkTpDeviceCollectiveRebase(
                implementation,1u);
            if ( rebase != SPARK_STATUS_OK )
                return rebase;
        }
        if ( implementation->round_seq + 1ull >=
                implementation->round_wave_limit )
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        round_seq = ++implementation->round_seq;
    }
    ordinal = submission->ordinal;
    slot_bytes = implementation->slot_bytes;
    if ( implementation->capture_armed != 0u )
    {
        band_index = (uint32_t)(implementation->band_base /
            (SPARK_WEIGHTD_MESH_SLOT_BYTES *
             SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
        slot_index = (uint64_t)implementation->tp_rank *
            SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
            (round_seq & (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
        if ( cudaMemcpyAsync(implementation->mesh_buffer +
                implementation->band_base + slot_index * slot_bytes,
                submission->local_device,(size_t)bytes,
                SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST,
                submission->cuda_stream) != 0 )
            return SPARK_STATUS_IO_ERROR;
        if ( SparkGlm5NextLaunchMeshPublish(submission->cuda_stream,
                implementation->mesh_buffer +
                SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
                    implementation->tp_rank),
                implementation->seq_cell,implementation->round_seq_device,
                bytes,slot_index) != 0 )
            return SPARK_STATUS_IO_ERROR;
        if ( SparkGlm5NextLaunchMeshWait(submission->cuda_stream,
                implementation->mesh_buffer + implementation->band_base,
                implementation->slot_bytes,
                implementation->round_seq_device,
                SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                round_seq & (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u),
                implementation->tp_rank,implementation->tp_degree,
                implementation->error_word,
                (unsigned long long)implementation->round_timeout_ns) != 0 )
            return SPARK_STATUS_IO_ERROR;
        goto combine;
    }
    deadline = SparkTpDeviceCollectiveTimeNs() +
        (implementation->round_timeout_ns <
            SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS
            ? implementation->round_timeout_ns
            : SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS);
    slot_index = (uint64_t)implementation->tp_rank *
        SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
        (round_seq & (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    slot_base = implementation->band_base + slot_index * slot_bytes;
    slot = implementation->mesh_buffer + slot_base;
    band_index = (uint32_t)(implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
    entry = (volatile uint64_t *)(implementation->mesh_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
            implementation->tp_rank));
    staging = &implementation->staging[implementation->round_index &
        (uint64_t)(SPARK_TP_DEVICE_COLLECTIVE_STAGING_SETS - 1u)];
    staging->slot = slot_index;
    staging->bytes = bytes;
    staging->seq = round_seq;
    __sync_synchronize();
    if ( cudaMemcpyAsync(slot,submission->local_device,(size_t)bytes,
            SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST,submission->cuda_stream) != 0 )
        goto publish_fail;
    if ( cudaMemcpyAsync(slot + bytes,&staging->seq,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_HOST,
            submission->cuda_stream) != 0 )
        goto publish_fail;
    if ( cudaMemcpyAsync((void *)&entry[2],&staging->slot,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_HOST,
            submission->cuda_stream) != 0 )
        goto publish_fail;
    if ( cudaMemcpyAsync((void *)&entry[1],&staging->bytes,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_HOST,
            submission->cuda_stream) != 0 )
        goto publish_fail;
    if ( cudaMemcpyAsync((void *)&entry[0],&staging->seq,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_HOST,
            submission->cuda_stream) != 0 )
        goto publish_fail;
    for ( peer = 0u; peer < implementation->tp_degree - 1u; peer++ )
    {
        uint32_t peer_rank =
            peer < implementation->tp_rank ? peer : peer + 1u;
        volatile uint64_t *end_word = (volatile uint64_t *)
            (implementation->mesh_buffer + implementation->band_base +
            ((uint64_t)peer_rank *
                SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                (round_seq &
                    (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u))) *
                slot_bytes +
            bytes);
        uint32_t exact = implementation->chain_key != 0ull;
        while ( exact != 0u ? *end_word != round_seq :
            *end_word < round_seq )
        {
            struct timespec pause = {0,1000};
            volatile uint64_t *cancel_cell = (volatile uint64_t *)
                (implementation->mesh_buffer +
                SparkTpDeviceCollectiveCancelCellOffset(band_index));
            if ( *cancel_cell != implementation->cancel_seen )
            {
                implementation->cancel_seen = *cancel_cell;
                fprintf(stderr,
                    "MESH-CANCEL-ABORT rank=%u peer=%u want=%llu got=%llu mi=%llu pub seq=%llu bytes=%llu slot=%llu\n",
                    implementation->tp_rank,peer_rank,
                    (unsigned long long)round_seq,
                    (unsigned long long)*end_word,
                    (unsigned long long)implementation->round_index,
                    (unsigned long long)staging->seq,
                    (unsigned long long)staging->bytes,
                    (unsigned long long)staging->slot);
                return SPARK_STATUS_BUSY;
            }
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
            {
                fprintf(stderr,
                    "MESH-SPIN-TIMEOUT rank=%u peer=%u want=%llu got=%llu bytes=%llu mi=%llu pub seq=%llu bytes=%llu slot=%llu\n",
                    implementation->tp_rank,peer_rank,
                    (unsigned long long)round_seq,
                    (unsigned long long)*end_word,
                    (unsigned long long)bytes,
                    (unsigned long long)implementation->round_index,
                    (unsigned long long)staging->seq,
                    (unsigned long long)staging->bytes,
                    (unsigned long long)staging->slot);
                return SPARK_STATUS_BUSY;
            }
            nanosleep(&pause,0);
        }
    }
combine:
    for ( peer = 0u; peer < implementation->tp_degree - 1u; peer++ )
    {
        uint8_t *source = implementation->mesh_buffer +
            implementation->band_base +
            ((uint64_t)(peer < implementation->tp_rank ?
                peer : peer + 1u) *
                SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                (round_seq &
                    (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u))) *
                slot_bytes;
        SparkStatus status;
        if ( operation_kind ==
                SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
            status = implementation->combine_u64_max(
                implementation->combine_context,
                (uint64_t *)submission->full_device,
                (const uint64_t *)source,
                submission->active_sequence_count,
                submission->cuda_stream);
        else
            status = implementation->combine_bf16(
                implementation->combine_context,
                submission->full_device,source,
                submission->active_sequence_count,
                implementation->local_hidden_dimension,
                submission->cuda_stream);
        if ( status != SPARK_STATUS_OK )
            return status;
    }
    if ( implementation->capture_armed == 0u )
        SparkTpDeviceCollectiveQueueCompletion(implementation,submission,
            ordinal,SPARK_STATUS_OK);
    return(SPARK_STATUS_OK);
publish_fail:
    if ( implementation->chain_key != 0ull )
        implementation->round_index--;
    else
        implementation->round_seq--;
    return SPARK_STATUS_IO_ERROR;
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
    implementation->round_timeout_ns =
        (uint64_t)config->operation_timeout_milli * 1000000ull;
    implementation->combine_bf16 = config->combine_bf16_function;
    implementation->combine_u64_max = config->combine_u64_max_function;
    implementation->combine_context = config->combine_context;
    if ( SparkWeightdClientConnect(socket,&implementation->client,0) !=
            SPARK_STATUS_OK )
    {
        free(implementation);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    if ( pthread_mutex_init(&implementation->completion_lock,0) != 0 ||
         pthread_cond_init(&implementation->completion_wake,0) != 0 )
    {
        (void)SparkWeightdClientClose(implementation->client);
        free(implementation);
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    }
    {
        pthread_t completion_thread;
        if ( pthread_create(&completion_thread,0,
                SparkTpDeviceCollectiveCompletionThread,
                implementation) != 0 )
        {
            (void)SparkWeightdClientClose(implementation->client);
            free(implementation);
            SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
        }
        pthread_detach(completion_thread);
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
         submission->active_sequence_count == 0u ||
         submission->ordinal == UINT64_MAX )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    if ( implementation->capture_armed == 0u &&
         submission->completion_function == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    round = *submission;
    status = SparkTpDeviceCollectiveRunRound(implementation,&round,
        operation_kind);
    if ( status != SPARK_STATUS_OK )
        SparkTpDeviceCollectiveQueueCompletion(implementation,&round,
            round.ordinal,status);
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
    if ( SparkTpDeviceCollectiveRegisteredRegion != receive_device )
    {
        uint64_t lane_bytes = 2ull *
            SPARK_WEIGHTD_MESH_SLOTS_PER_BAND *
            SPARK_WEIGHTD_MESH_SLOT_BYTES;
        if ( SparkTpDeviceCollectiveRegisteredRegion != 0 ||
             cudaHostRegister(receive_device +
                    implementation->band_base -
                    (implementation->band_base % lane_bytes),
                lane_bytes,0u) != 0 )
        {
            int first = cudaGetLastError();
            int retry = cudaHostRegister(receive_device +
                    implementation->band_base -
                    (implementation->band_base % lane_bytes),
                lane_bytes,0u);
            if ( SparkTpDeviceCollectiveRegisteredRegion != 0 || retry != 0 )
            {
                FILE *maps = fopen("/proc/self/maps","r");
                char line[256];
                fprintf(stderr,"MESH-REGISTER-FAIL ptr=%p band=%llu first=%d retry=%d(%s)\n",
                    receive_device,
                    (unsigned long long)implementation->band_base,first,retry,
                    cudaGetErrorString(retry != 0 ? retry : first));
                if ( maps != 0 )
                {
                    while ( fgets(line,sizeof(line),maps) != 0 )
                        if ( strstr(line,"spark-mesh") != 0 )
                            fputs(line,stderr);
                    fclose(maps);
                }
                SPARK_FAIL(SPARK_STATUS_IO_ERROR);
            }
        }
        SparkTpDeviceCollectiveRegisteredRegion = receive_device;
    }
    implementation->mesh_buffer = receive_device;
    {
        uint32_t band_index = (uint32_t)(implementation->band_base /
            (SPARK_WEIGHTD_MESH_SLOT_BYTES *
             SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
        volatile uint64_t *entry = (volatile uint64_t *)
            (implementation->mesh_buffer +
            SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
                implementation->tp_rank));
        uint32_t peer_rank;
        for ( peer_rank = 0u;
              peer_rank < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer_rank++ )
        {
            volatile uint64_t *peer_entry = (volatile uint64_t *)
                (implementation->mesh_buffer +
                SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,peer_rank));
            if ( peer_entry[0] > implementation->round_seq )
                implementation->round_seq = peer_entry[0];
        }
        __sync_synchronize();
        entry[2] = 0ull;
        entry[1] = 0ull;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveArmCapture(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t zero = 0u;
    if ( collective == 0 || collective->implementation == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    if ( implementation->seq_cell == 0 )
    {
        if ( cudaMalloc(&implementation->seq_cell,8u) != 0 ||
             cudaMalloc(&implementation->round_seq_device,8u) != 0 ||
             cudaMalloc(&implementation->error_word,8u) != 0 )
        {
            implementation->seq_cell = 0;
            implementation->round_seq_device = 0;
            implementation->error_word = 0;
            SPARK_FAIL(SPARK_STATUS_IO_ERROR);
        }
    }
    if ( cudaMemcpy(implementation->seq_cell,&implementation->round_seq,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 ||
         cudaMemcpy(implementation->round_seq_device,&zero,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 ||
         cudaMemcpy(implementation->error_word,&zero,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 )
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    implementation->capture_armed = 1u;
    return(SPARK_STATUS_OK);
}

SparkStatus SparkTpDeviceCollectiveDisarmCapture(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t cell = 0u;
    if ( collective == 0 || collective->implementation == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    implementation->capture_armed = 0u;
    if ( implementation->seq_cell != 0 &&
         cudaMemcpy(&cell,implementation->seq_cell,sizeof(uint64_t),
             SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST) == 0 && cell != 0ull )
        implementation->round_seq = cell;
    return(SPARK_STATUS_OK);
}

uint64_t SparkTpDeviceCollectiveRoundIndex(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    if ( collective == 0 || collective->implementation == 0 )
        return(0ull);
    implementation = collective->implementation;
    return(implementation->round_index);
}

void SparkTpDeviceCollectiveBroadcastCancel(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    volatile uint64_t *cancel_cell;
    if ( collective == 0 || collective->implementation == 0 )
        return;
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 )
        return;
    cancel_cell = (volatile uint64_t *)(implementation->mesh_buffer +
        SparkTpDeviceCollectiveCancelCellOffset((uint32_t)(
            implementation->band_base / (SPARK_WEIGHTD_MESH_SLOT_BYTES *
                SPARK_WEIGHTD_MESH_SLOTS_PER_BAND))));
    implementation->cancel_epoch++;
    *cancel_cell = implementation->cancel_epoch |
        ((uint64_t)implementation->tp_rank << 56);
    implementation->cancel_seen = *cancel_cell;
    __sync_synchronize();
    {
        uint64_t cancel_offset = SparkTpDeviceCollectiveCancelCellOffset(
            (uint32_t)(implementation->band_base /
                (SPARK_WEIGHTD_MESH_SLOT_BYTES *
                 SPARK_WEIGHTD_MESH_SLOTS_PER_BAND)));
        (void)SparkWeightdClientMeshBroadcast(implementation->client,
            0xffffu & ~(1u << implementation->tp_rank),
            cancel_offset,cancel_offset,
            8u,0ull,0ull,implementation->round_timeout_ns);
    }
}

uint64_t SparkTpDeviceCollectiveGraphError(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t error = 0u;
    if ( collective == 0 || collective->implementation == 0 )
        return(0ull);
    implementation = collective->implementation;
    if ( implementation->error_word == 0 )
        return(0ull);
    if ( cudaMemcpy(&error,implementation->error_word,sizeof(uint64_t),
            SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST) != 0 )
        return(1ull);
    return(error);
}

void SparkTpDeviceCollectiveClearGraphError(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t zero = 0u;
    if ( collective == 0 || collective->implementation == 0 )
        return;
    implementation = collective->implementation;
    if ( implementation->error_word != 0 )
        (void)cudaMemcpy(implementation->error_word,&zero,sizeof(uint64_t),
            SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE);
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
