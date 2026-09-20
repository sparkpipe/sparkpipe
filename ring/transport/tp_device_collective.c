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
extern int cudaMemsetAsync(void *destination,int value,size_t bytes,
    void *stream);
extern int cudaMalloc(void **pointer,size_t bytes);
extern int cudaFree(void *pointer);
extern int cudaMemcpyAsync(void *destination,const void *source,
    size_t bytes,int kind,void *stream);
extern int cudaHostRegister(void *address,size_t bytes,unsigned int flags);
extern int cudaMemcpy(void *destination,const void *source,
    size_t bytes,int kind);
extern int cudaMalloc(void **address,size_t bytes);
extern int cudaHostAlloc(void **address,size_t bytes,unsigned int flags);
extern int cudaStreamSynchronize(void *stream);
extern int SparkGlm5NextLaunchMeshCopyDown(void *stream,
    volatile void *destination,const void *source,uint64_t bytes);
extern int SparkGlm5NextLaunchMeshPublish(void *stream,
    volatile void *entry,void *seq_cell,const void *epoch_cell,
    void *round_seq,uint64_t bytes,
    uint64_t slot_index,volatile void *slot_tail);
extern int SparkGlm5NextLaunchMeshGuard(void *stream,
    volatile void *error_word,void *output);
extern int SparkGlm5NextLaunchMeshWait(void *stream,
    volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint64_t ring,uint32_t rank,uint32_t degree,
    void *error_word,unsigned long long deadline_ns,void *diag_word,
    volatile void *cancel_cell,const void *cancel_expected);

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
    uint64_t consumed_cell;
    uint64_t round_index;
    uint64_t cancel_epoch;
    uint64_t publish_ack_prev;
    SparkTpDeviceCollectiveStagingSet
        staging[SPARK_TP_DEVICE_COLLECTIVE_STAGING_SETS];
    void *seq_cell;
    void *epoch_cell;
    void *round_seq_device;
    void *error_word;
    void *diag_word;
    void *cancel_expected;
    volatile uint64_t *published_host_cell;
    uint32_t capture_armed;
    uint32_t round_rebased;
    uint64_t cancel_seen;
    uint32_t round_deadline_ms;
    pthread_mutex_t completion_lock;
    pthread_cond_t completion_wake;
    pthread_t completion_thread;
    uint32_t completion_thread_live;
    uint32_t completion_stop;
    SparkTpDeviceCollectiveCompletionNode *completion_head;
    SparkTpDeviceCollectiveCompletionNode *completion_tail;
    SparkTpDeviceCollectiveCombineBf16Function combine_bf16;
    SparkTpDeviceCollectiveCombineU64MaxFunction combine_u64_max;
    SparkTpDeviceCollectiveCombineF32SeedFunction combine_f32_seed;
    SparkTpDeviceCollectiveCombineF32AddFunction combine_f32_add;
    SparkTpDeviceCollectiveRoundF32Function round_f32;
    SparkTpDeviceCollectiveCombineFusedBf16Function combine_fused_bf16;
    float *f32_scratch;
    uint64_t f32_scratch_bytes;
    void *combine_context;
    uint32_t tp_rank;
    uint32_t tp_degree;
    uint32_t local_hidden_dimension;
    uint64_t round_ns_total;
    uint64_t round_count;
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
        while (implementation->completion_head == 0 &&
               implementation->completion_stop == 0u)
            pthread_cond_wait(&implementation->completion_wake,
                &implementation->completion_lock);
        node = implementation->completion_head;
        if ( node == 0 )
        {
            pthread_mutex_unlock(&implementation->completion_lock);
            break;
        }
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
#define SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS (30ull * 1000000000ull)

static uint64_t SparkTpDeviceCollectiveBaseCellOffset(uint32_t band_index)
{
    return SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (SPARK_WEIGHTD_MESH_DOORBELL_CELL_BASE + 2u * band_index) *
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES;
}

static uint64_t SparkTpDeviceCollectiveCancelCellOffset(uint32_t band_index)
{
    return SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
        (SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL + 2u * band_index) *
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES;
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
        uint64_t high = implementation->round_seq;
        uint64_t base;
        if ( (*base_cell >> SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS) >
                SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK ||
            (implementation->round_seq >>
                SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS) >
                SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK )
        {
            fprintf(stderr,
                "CKEY-RESET rank=0 poisoned epoch cell=%llu round_seq=%llu\n",
                (unsigned long long)*base_cell,
                (unsigned long long)implementation->round_seq);
            *base_cell = 0ull;
            implementation->round_seq = 0ull;
            implementation->base_seen = 0ull;
            high = 0ull;
        }
        if ( *base_cell > high )
            high = *base_cell;
        base = ((high / SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE) + 1ull) *
            SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
        fprintf(stderr,"CKEY-REBASE rank=%u band=%u base=%llu high=%llu round_seq=%llu cell_before=%llu\n",
            implementation->tp_rank,band_index,
            (unsigned long long)base,(unsigned long long)high,
            (unsigned long long)implementation->round_seq,
            (unsigned long long)*base_cell);
        *base_cell = base;
        if ( use_broadcast != 0u )
        {
            __sync_synchronize();
            (void)SparkWeightdClientMeshBroadcast(
                implementation->client,
                ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
                    ~(1u << implementation->tp_rank),
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
            if ( *cancel_cell != implementation->cancel_seen )
            {
                fprintf(stderr,
                    "REBASE-CANCEL-QUIET rank=%u cell=%llu marker=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)*base_cell,
                    (unsigned long long)marker);
                return SPARK_STATUS_BUSY;
            }
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
            {
                fprintf(stderr,
                    "REBASE-TIMEOUT-QUIET rank=%u cell=%llu marker=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)*base_cell,
                    (unsigned long long)marker);
                return SPARK_STATUS_BUSY;
            }
        }
        implementation->base_seen = *base_cell;
    }
    implementation->round_seq = implementation->base_seen - 1ull;
    implementation->round_wave_limit = implementation->base_seen +
        SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
    implementation->round_rebased = 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveChainRetire(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint32_t band_index;
    volatile uint64_t *base_cell;
    if ( collective == 0 || collective->implementation == 0 )
        return(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->mesh_buffer == 0 || implementation->tp_rank != 0u )
        return(SPARK_STATUS_OK);
    band_index = (uint32_t)(implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES *
         SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
    base_cell = (volatile uint64_t *)(implementation->mesh_buffer +
        SparkTpDeviceCollectiveBaseCellOffset(band_index));
    *base_cell = *base_cell +
        SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
    __sync_synchronize();
    (void)SparkWeightdClientMeshBroadcast(
        implementation->client,
        ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
            ~(1u << implementation->tp_rank),
        SparkTpDeviceCollectiveBaseCellOffset(band_index),
        SparkTpDeviceCollectiveBaseCellOffset(band_index),8u,0ull,0ull,
        implementation->round_timeout_ns);
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkTpDeviceCollectiveEnsureCells(
    SparkTpDeviceCollectiveImplementation *implementation);

SparkStatus SparkTpDeviceCollectiveChainKey(
    SparkTpDeviceCollective *collective,uint64_t request_id)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint32_t band_index;
    volatile uint64_t *base_cell;
    uint64_t epoch;
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
    if ( implementation->tp_rank == 0u )
    {
        uint64_t cell_epoch = *base_cell >>
            SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS;
        implementation->cancel_seen = *(volatile uint64_t *)
            (implementation->mesh_buffer +
            SparkTpDeviceCollectiveCancelCellOffset(band_index));
        if ( cell_epoch > SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK )
            cell_epoch = 0ull;
        if ( implementation->chain_epoch > cell_epoch )
            cell_epoch = implementation->chain_epoch;
        epoch = cell_epoch + 1ull;
        implementation->base_seen = epoch <<
            SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS;
        implementation->round_seq = implementation->base_seen - 1ull;
        implementation->round_wave_limit = implementation->base_seen +
            SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
        implementation->round_rebased = 1u;
        *base_cell = (epoch << SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS) |
            request_id;
        fprintf(stderr,
            "CKEY-WRITE rank=0 epoch=%llu cell=%llu req=%llu\n",
            (unsigned long long)epoch,
            (unsigned long long)*base_cell,
            (unsigned long long)request_id);
        __sync_synchronize();
        {
            SparkStatus bcast = SparkWeightdClientMeshBroadcast(
                implementation->client,
                ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
                    ~(1u << implementation->tp_rank),
                SparkTpDeviceCollectiveBaseCellOffset(band_index),
                SparkTpDeviceCollectiveBaseCellOffset(band_index),8u,0ull,0ull,
                implementation->round_timeout_ns);
            if ( bcast != SPARK_STATUS_OK )
                fprintf(stderr,"CKEY-BCAST-FAIL rank=0 epoch=%llu status=%d\n",
                    (unsigned long long)epoch,(int)bcast);
        }
    }
    else
    {
        volatile uint64_t *cancel_cell = (volatile uint64_t *)
            (implementation->mesh_buffer +
            SparkTpDeviceCollectiveCancelCellOffset(band_index));
        uint64_t cell = *base_cell;
        uint64_t cell_epoch = cell >> SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS;
        uint64_t deadline = SparkTpDeviceCollectiveTimeNs() +
            (implementation->round_timeout_ns <
                SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS ?
                implementation->round_timeout_ns :
                SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS);
        implementation->cancel_seen = *cancel_cell;
        while ( cell_epoch == 0ull ||
                cell_epoch > SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK ||
                cell == implementation->consumed_cell ||
                (cell & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK) !=
                    request_id )
        {
            if ( *cancel_cell != implementation->cancel_seen )
            {
                fprintf(stderr,
                    "CKEY-CANCEL rank=%u cell=%llu req=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)*base_cell,
                    (unsigned long long)request_id);
                return SPARK_STATUS_BUSY;
            }
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
            {
                fprintf(stderr,
                    "CKEY-CELL-TIMEOUT rank=%u cell=%llu req=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)*base_cell,
                    (unsigned long long)request_id);
                return SPARK_STATUS_BUSY;
            }
            ;
            cell = *base_cell;
            cell_epoch = cell >> SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS;
        }
        epoch = cell_epoch;
        implementation->consumed_cell = cell;
        if ( epoch != implementation->chain_epoch ||
             implementation->round_rebased == 0u )
        {
            fprintf(stderr,
                "CKEY-CELL-ADOPT rank=%u epoch=%llu had=%llu req=%llu\n",
                implementation->tp_rank,
                (unsigned long long)epoch,
                (unsigned long long)implementation->chain_epoch,
                (unsigned long long)request_id);
            implementation->base_seen = epoch <<
                SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE_BITS;
            implementation->round_seq = implementation->base_seen - 1ull;
            implementation->round_wave_limit = implementation->base_seen +
                SPARK_TP_DEVICE_COLLECTIVE_WAVE_STRIDE;
            implementation->round_rebased = 1u;
        }
    }
    if ( epoch == 0ull ||
         epoch > SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK )
    {
        fprintf(stderr,
            "CKEY-BAD-CELL rank=%u cell=%llu rebased=%u band=%u base_seen=%llu round_seq=%llu limit=%llu chain_key=%llu req=%llu\n",
            implementation->tp_rank,
            (unsigned long long)*base_cell,
            implementation->round_rebased,
            band_index,
            (unsigned long long)implementation->base_seen,
            (unsigned long long)implementation->round_seq,
            (unsigned long long)implementation->round_wave_limit,
            (unsigned long long)implementation->chain_key,
            (unsigned long long)request_id);
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    }
    if ( epoch != implementation->chain_epoch )
    {
        fprintf(stderr,
            "CKEY-ADOPT rank=%u epoch=%llu had=%llu\n",
            implementation->tp_rank,
            (unsigned long long)epoch,
            (unsigned long long)implementation->chain_epoch);
        implementation->chain_epoch = epoch;
        implementation->round_index = 0ull;
        if ( SparkTpDeviceCollectiveEnsureCells(implementation) ==
                SPARK_STATUS_OK )
        {
            uint64_t zero = 0ull;
            if ( cudaMemcpy(implementation->epoch_cell,&epoch,
                    sizeof(uint64_t),
                    SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 ||
                 cudaMemcpy(implementation->seq_cell,&zero,
                    sizeof(uint64_t),
                    SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 )
                SPARK_FAIL(SPARK_STATUS_IO_ERROR);
        }
        else
            SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    implementation->chain_key =
        (epoch << SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_BITS) | request_id;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveEnsureCells(
    SparkTpDeviceCollectiveImplementation *implementation);

static SparkStatus SparkTpDeviceCollectiveRunRound(
    SparkTpDeviceCollectiveImplementation *implementation,
    SparkTpDeviceCollectiveSubmission *submission,
    uint32_t operation_kind)
{
    uint64_t bytes;
    uint64_t ordinal;
    uint64_t deadline;
    uint64_t slot_bytes;
    uint64_t slot_index;
    uint64_t slot_base;
    uint64_t parity;
    uint64_t published = 0ull;
    uint8_t *slot;
    volatile uint64_t *entry;
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
        implementation->round_seq++;
    }
    {
        SparkStatus ensure = SparkTpDeviceCollectiveEnsureCells(implementation);
        if ( ensure != SPARK_STATUS_OK )
            return ensure;
    }
    ordinal = submission->ordinal;
    slot_bytes = implementation->slot_bytes;
    parity = ordinal &
        (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
    if ( implementation->capture_armed != 0u )
    {
        band_index = (uint32_t)(implementation->band_base /
            (SPARK_WEIGHTD_MESH_SLOT_BYTES *
             SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
        slot_index = (uint64_t)implementation->tp_rank *
            SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + parity;
        if ( SparkGlm5NextLaunchMeshCopyDown(submission->cuda_stream,
                implementation->mesh_buffer +
                implementation->band_base + slot_index * slot_bytes,
                submission->local_device,bytes) != 0 )
            return SPARK_STATUS_IO_ERROR;
        if ( SparkGlm5NextLaunchMeshPublish(submission->cuda_stream,
                implementation->mesh_buffer +
                SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
                    implementation->tp_rank),
                implementation->seq_cell,implementation->epoch_cell,
                implementation->round_seq_device,
                bytes,slot_index,
                implementation->mesh_buffer +
                implementation->band_base + slot_index * slot_bytes +
                slot_bytes - 8u) != 0 )
            return SPARK_STATUS_IO_ERROR;
        if ( SparkGlm5NextLaunchMeshWait(submission->cuda_stream,
                implementation->mesh_buffer + implementation->band_base,
                implementation->slot_bytes,
                implementation->round_seq_device,
                SPARK_WEIGHTD_MESH_SLOTS_PER_RANK,
                parity,
                implementation->tp_rank,implementation->tp_degree,
                implementation->error_word,
                (unsigned long long)(implementation->round_timeout_ns <
                    SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS ?
                    implementation->round_timeout_ns :
                    SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS),
                implementation->diag_word,
                (volatile uint64_t *)(implementation->mesh_buffer +
                    SparkTpDeviceCollectiveCancelCellOffset(
                        (uint32_t)(implementation->band_base /
                            (SPARK_WEIGHTD_MESH_SLOT_BYTES *
                             SPARK_WEIGHTD_MESH_SLOTS_PER_BAND)))),
                implementation->cancel_expected) != 0 )
            return SPARK_STATUS_IO_ERROR;
        goto combine;
    }
    deadline = SparkTpDeviceCollectiveTimeNs() +
        (implementation->round_timeout_ns <
            SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS
            ? implementation->round_timeout_ns
            : SPARK_TP_DEVICE_COLLECTIVE_ROUND_SPIN_TIMEOUT_NS);
    slot_index = (uint64_t)implementation->tp_rank *
        SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + parity;
    slot_base = implementation->band_base + slot_index * slot_bytes;
    slot = implementation->mesh_buffer + slot_base;
    band_index = (uint32_t)(implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
    entry = (volatile uint64_t *)(implementation->mesh_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
            implementation->tp_rank));
    if ( bytes == 0ull || slot_index >=
            (uint64_t)SPARK_WEIGHTD_MESH_SLOTS_PER_BAND )
    {
        fprintf(stderr,"MESH-STAGING-BAD rank=%u bytes=%llu slot=%llu\n",
            implementation->tp_rank,
            (unsigned long long)bytes,
            (unsigned long long)slot_index);
        return SPARK_STATUS_BUSY;
    }
    if ( implementation->publish_ack_prev != 0u )
    {
        volatile uint32_t *shipped_cell = (volatile uint32_t *)
            (implementation->mesh_buffer +
            SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band_index,
                implementation->tp_rank));
        while ( *shipped_cell != (uint32_t)implementation->publish_ack_prev )
        {
            volatile uint64_t *cancel_cell = (volatile uint64_t *)
                (implementation->mesh_buffer +
                SparkTpDeviceCollectiveCancelCellOffset(band_index));
            if ( *cancel_cell != implementation->cancel_seen )
            {
                implementation->cancel_seen = *cancel_cell;
                fprintf(stderr,
                    "MESH-SHIP-CANCEL rank=%u prev=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)implementation->publish_ack_prev);
                return SPARK_STATUS_BUSY;
            }
            if ( SparkWeightdClientAlive(implementation->client) == 0u )
            {
                fprintf(stderr,
                    "WEIGHTD-DEAD rank=%u awaiting ship ack — failing fast\n",
                    implementation->tp_rank);
                return SPARK_STATUS_IO_ERROR;
            }
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
            {
                fprintf(stderr,
                    "MESH-SHIP-TIMEOUT rank=%u prev=%llu cell=%u slot=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)implementation->publish_ack_prev,
                    (unsigned)*shipped_cell,
                    (unsigned long long)slot_index);
                return SPARK_STATUS_BUSY;
            }
        }
    }
    if ( SparkGlm5NextLaunchMeshCopyDown(submission->cuda_stream,
            slot,submission->local_device,bytes) != 0 )
    {
        fprintf(stderr,"MESH-COPYDOWN-FAIL rank=%u bytes=%llu slot=%llu cuda=%s\n",
            implementation->tp_rank,(unsigned long long)bytes,
            (unsigned long long)slot_index,
            cudaGetErrorString(cudaGetLastError()));
        return SPARK_STATUS_IO_ERROR;
    }
    if ( SparkGlm5NextLaunchMeshPublish(submission->cuda_stream,
            (volatile void *)entry,implementation->seq_cell,
            implementation->epoch_cell,
            implementation->round_seq_device,bytes,slot_index,
            slot + slot_bytes - 8u) != 0 )
    {
        fprintf(stderr,"MESH-PUBLISH-FAIL rank=%u bytes=%llu slot=%llu cuda=%s\n",
            implementation->tp_rank,(unsigned long long)bytes,
            (unsigned long long)slot_index,
            cudaGetErrorString(cudaGetLastError()));
        return SPARK_STATUS_IO_ERROR;
    }
    if ( implementation->published_host_cell == 0 &&
         cudaHostAlloc((void **)&implementation->published_host_cell,
             sizeof(uint64_t),0u) != 0 )
        implementation->published_host_cell = 0;
    if ( implementation->published_host_cell == 0 )
        return SPARK_STATUS_IO_ERROR;
    if ( cudaMemcpyAsync((void *)implementation->published_host_cell,
            implementation->round_seq_device,sizeof(uint64_t),
            SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST,submission->cuda_stream) != 0 ||
         cudaStreamSynchronize(submission->cuda_stream) != 0 )
    {
        fprintf(stderr,"MESH-READBACK-FAIL rank=%u bytes=%llu slot=%llu cuda=%s\n",
            implementation->tp_rank,(unsigned long long)bytes,
            (unsigned long long)slot_index,
            cudaGetErrorString(cudaGetLastError()));
        return SPARK_STATUS_IO_ERROR;
    }
    published = *implementation->published_host_cell;
    implementation->publish_ack_prev = published;
    {
        uint32_t peers_remaining = implementation->tp_degree - 1u;
        uint32_t peer_passed[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE] = {0u};
        while ( peers_remaining != 0u )
        {
            volatile uint64_t *cancel_cell = (volatile uint64_t *)
                (implementation->mesh_buffer +
                SparkTpDeviceCollectiveCancelCellOffset(band_index));
            if ( *cancel_cell != implementation->cancel_seen )
            {
                implementation->cancel_seen = *cancel_cell;
                fprintf(stderr,
                    "MESH-CANCEL-ABORT rank=%u pub seq=%llu bytes=%llu slot=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)published,
                    (unsigned long long)bytes,
                    (unsigned long long)slot_index);
                return SPARK_STATUS_BUSY;
            }
            if ( SparkWeightdClientAlive(implementation->client) == 0u )
            {
                fprintf(stderr,
                    "WEIGHTD-DEAD rank=%u mid-wait — failing fast\n",
                    implementation->tp_rank);
                return SPARK_STATUS_IO_ERROR;
            }
            if ( SparkTpDeviceCollectiveTimeNs() >= deadline )
            {
                char missing[128];
                uint32_t missing_len = 0u;
                missing[0] = 0;
                for ( peer = 0u; peer < implementation->tp_degree - 1u; peer++ )
                {
                    if ( peer_passed[peer] == 0u )
                    {
                        uint32_t peer_rank =
                            peer < implementation->tp_rank ? peer : peer + 1u;
                        int wrote = snprintf(missing + missing_len,
                            sizeof(missing) - missing_len,"%u ",peer_rank);
                        if ( wrote > 0 )
                            missing_len += (uint32_t)wrote < sizeof(missing) - missing_len ?
                                (uint32_t)wrote : sizeof(missing) - missing_len - 1u;
                    }
                }
                fprintf(stderr,
                    "MESH-SPIN-TIMEOUT rank=%u pub=%llu remaining=%u missing=%s bytes=%llu slot=%llu\n",
                    implementation->tp_rank,
                    (unsigned long long)published,
                    (unsigned)peers_remaining,
                    missing,
                    (unsigned long long)bytes,
                    (unsigned long long)slot_index);
                return SPARK_STATUS_BUSY;
            }
            for ( peer = 0u; peer < implementation->tp_degree - 1u; peer++ )
            {
                uint32_t peer_rank =
                    peer < implementation->tp_rank ? peer : peer + 1u;
                volatile uint64_t *end_word = (volatile uint64_t *)
                    (implementation->mesh_buffer + implementation->band_base +
                    ((uint64_t)peer_rank *
                        SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + parity) *
                        slot_bytes +
                    slot_bytes - 8u);
                if ( peer_passed[peer] == 0u && *end_word >= published )
                {
                    peer_passed[peer] = 1u;
                    peers_remaining--;
                }
            }
            ;
        }
    }
combine:
    if ( operation_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 &&
         implementation->combine_fused_bf16 != 0 )
    {
        const void *source_devices[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
        SparkStatus fused_status;
        for ( peer = 0u;
              peer < implementation->tp_degree;
              peer++ )
            source_devices[peer] = implementation->mesh_buffer +
                implementation->band_base +
                ((uint64_t)peer *
                    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                    (parity)) *
                    slot_bytes;
        fused_status = implementation->combine_fused_bf16(
            implementation->combine_context,
            submission->full_device,source_devices,
            implementation->tp_degree,
            submission->active_sequence_count,
            implementation->local_hidden_dimension,
            submission->cuda_stream);
        if ( fused_status != SPARK_STATUS_OK )
            return fused_status;
        goto combine_done;
    }
    if ( operation_kind !=
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 &&
         implementation->combine_f32_seed != 0 &&
         implementation->combine_f32_add != 0 &&
         implementation->round_f32 != 0 )
    {
        uint64_t f32_bytes = ((uint64_t)bytes) * 2ull;
        if ( implementation->f32_scratch == 0 ||
             implementation->f32_scratch_bytes < f32_bytes )
        {
            if ( implementation->f32_scratch != 0 )
                (void)cudaFree(implementation->f32_scratch);
            if ( cudaMalloc((void **)&implementation->f32_scratch,
                     (size_t)f32_bytes) != 0 )
            {
                implementation->f32_scratch = 0;
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            implementation->f32_scratch_bytes = f32_bytes;
        }
        {
            uint8_t *slot_zero = implementation->mesh_buffer +
                implementation->band_base +
                ((uint64_t)0u *
                    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                    (parity)) *
                    slot_bytes;
            uint8_t *slot_one = implementation->mesh_buffer +
                implementation->band_base +
                ((uint64_t)1u *
                    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                    (parity)) *
                    slot_bytes;
            SparkStatus f32_status;
            f32_status = implementation->combine_f32_seed(
                implementation->combine_context,
                implementation->f32_scratch,slot_zero,slot_one,
                (uint32_t)(bytes / 2u),submission->cuda_stream);
            for ( peer = 1u;
                  peer < implementation->tp_degree && f32_status == SPARK_STATUS_OK;
                  peer++ )
            {
                uint8_t *source = implementation->mesh_buffer +
                    implementation->band_base +
                    ((uint64_t)peer *
                        SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                        (parity)) *
                        slot_bytes;
                f32_status = implementation->combine_f32_add(
                    implementation->combine_context,
                    implementation->f32_scratch,source,
                    (uint32_t)(bytes / 2u),submission->cuda_stream);
            }
            if ( f32_status == SPARK_STATUS_OK )
                f32_status = implementation->round_f32(
                    implementation->combine_context,
                    submission->full_device,implementation->f32_scratch,
                    (uint32_t)(bytes / 2u),submission->cuda_stream);
            if ( f32_status != SPARK_STATUS_OK )
                return f32_status;
            goto combine_done;
        }
    }
    if ( cudaMemsetAsync(submission->full_device,0,(size_t)bytes,
             submission->cuda_stream) != 0 )
        return SPARK_STATUS_IO_ERROR;
    for ( peer = 0u; peer < implementation->tp_degree; peer++ )
    {
        uint8_t *source = implementation->mesh_buffer +
            implementation->band_base +
            ((uint64_t)peer *
                SPARK_WEIGHTD_MESH_SLOTS_PER_RANK +
                (parity)) *
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
combine_done:
    if ( operation_kind ==
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 &&
         implementation->error_word != 0 &&
         SparkGlm5NextLaunchMeshGuard(submission->cuda_stream,
             implementation->error_word,submission->full_device) != 0 )
        return(SPARK_STATUS_IO_ERROR);
    if ( implementation->capture_armed == 0u )
        SparkTpDeviceCollectiveQueueCompletion(implementation,submission,
            ordinal,SPARK_STATUS_OK);
    return(SPARK_STATUS_OK);
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
    implementation->combine_f32_seed = config->combine_f32_seed_function;
    implementation->combine_f32_add = config->combine_f32_add_function;
    implementation->round_f32 = config->round_f32_function;
    implementation->combine_fused_bf16 = config->combine_fused_bf16_function;
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
        if ( pthread_create(&implementation->completion_thread,0,
                SparkTpDeviceCollectiveCompletionThread,
                implementation) != 0 )
        {
            (void)SparkWeightdClientClose(implementation->client);
            free(implementation);
            SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
        }
        implementation->completion_thread_live = 1u;
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
    {
        uint64_t round_start_ns = SparkTpDeviceCollectiveTimeNs();
        status = SparkTpDeviceCollectiveRunRound(implementation,&round,
            operation_kind);
        implementation->round_ns_total += SparkTpDeviceCollectiveTimeNs() - round_start_ns;
        implementation->round_count++;
    }
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
        if ( SparkTpDeviceCollectiveRegisteredRegion == 0 &&
             cudaHostRegister(receive_device,
                (size_t)SPARK_WEIGHTD_MESH_REGION_BYTES,0u) != 0 )
        {
            fprintf(stderr,"MESH-REGISTER-FALLBACK ptr=%p region_bytes=%llu (%s); continuing with pageable mesh memory\n",
                receive_device,
                (unsigned long long)SPARK_WEIGHTD_MESH_REGION_BYTES,
                cudaGetErrorString(cudaGetLastError()));
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

static SparkStatus SparkTpDeviceCollectiveEnsureCells(
    SparkTpDeviceCollectiveImplementation *implementation)
{
    uint64_t seed;
    uint64_t zero_epoch = 0ull;
    if ( implementation->seq_cell != 0 )
        return SPARK_STATUS_OK;
    if ( cudaMalloc(&implementation->seq_cell,8u) != 0 ||
         cudaMalloc(&implementation->epoch_cell,8u) != 0 ||
         cudaMalloc(&implementation->round_seq_device,8u) != 0 ||
         cudaMalloc(&implementation->error_word,8u) != 0 ||
         cudaMalloc(&implementation->diag_word,8u) != 0 ||
         cudaMalloc(&implementation->cancel_expected,8u) != 0 )
    {
        implementation->seq_cell = 0;
        implementation->epoch_cell = 0;
        implementation->round_seq_device = 0;
        implementation->error_word = 0;
        implementation->diag_word = 0;
        implementation->cancel_expected = 0;
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    seed = implementation->round_seq;
    if ( cudaMemcpy(implementation->seq_cell,&seed,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 ||
         cudaMemcpy(implementation->epoch_cell,&zero_epoch,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 ||
         cudaHostAlloc((void **)&implementation->published_host_cell,
             sizeof(uint64_t),0u) != 0 )
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
    {
        SparkStatus ensure = SparkTpDeviceCollectiveEnsureCells(implementation);
        if ( ensure != SPARK_STATUS_OK )
            return ensure;
    }
    if ( cudaMemcpy(implementation->error_word,&zero,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 ||
         cudaMemcpy(implementation->diag_word,&zero,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE) != 0 )
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    if ( implementation->f32_scratch == 0 &&
         implementation->slot_bytes != 0u )
    {
        uint64_t pre_bytes = 2ull * implementation->slot_bytes;
        if ( cudaMalloc((void **)&implementation->f32_scratch,
                 (size_t)pre_bytes) == 0 )
            implementation->f32_scratch_bytes = pre_bytes;
        else
            implementation->f32_scratch = 0;
    }
    implementation->capture_armed = 1u;
    return(SPARK_STATUS_OK);
}


SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(
    SparkTpDeviceCollective *collective,void *stream)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    if ( collective == 0 || collective->implementation == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    implementation = collective->implementation;
    if ( implementation->cancel_expected == 0 )
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    if ( cudaMemcpyAsync(implementation->cancel_expected,
            &implementation->cancel_seen,sizeof(uint64_t),
            SPARK_TP_CUDA_MEMCPY_HOST_TO_DEVICE,stream) != 0 )
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
            ((1u << SPARK_WEIGHTD_MESH_RANKS_PER_BAND) - 1u) &
                ~(1u << implementation->tp_rank),
            cancel_offset,cancel_offset,
            8u,0ull,0ull,implementation->round_timeout_ns);
    }
}

uint64_t SparkTpDeviceCollectiveGraphProgress(
    SparkTpDeviceCollective *collective,
    uint64_t *cell_out)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t round = 0ull;
    if ( cell_out != 0 )
        *cell_out = 0ull;
    if ( collective == 0 || collective->implementation == 0 )
        return(0ull);
    implementation = collective->implementation;
    if ( implementation->round_seq_device == 0 ||
         implementation->seq_cell == 0 )
        return(0ull);
    if ( cudaMemcpy(&round,implementation->round_seq_device,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST) != 0 )
        return(0ull);
    if ( cell_out != 0 &&
         cudaMemcpy(cell_out,implementation->seq_cell,sizeof(uint64_t),
             SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST) != 0 )
        *cell_out = 0ull;
    return(round);
}

uint64_t SparkTpDeviceCollectiveGraphStuckDump(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    volatile uint64_t *entry;
    uint64_t sequence = 0ull;
    uint64_t parity;
    uint32_t peer,band_index;
    if ( collective == 0 || collective->implementation == 0 )
        return(0ull);
    implementation = collective->implementation;
    if ( implementation->round_seq_device == 0 ||
         implementation->mesh_buffer == 0 )
        return(0ull);
    if ( cudaMemcpy(&sequence,implementation->round_seq_device,
            sizeof(uint64_t),SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST) != 0 )
        return(0ull);
    parity = sequence &
        (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
    band_index = (uint32_t)(implementation->band_base /
        (SPARK_WEIGHTD_MESH_SLOT_BYTES *
         SPARK_WEIGHTD_MESH_SLOTS_PER_BAND));
    entry = (volatile uint64_t *)(implementation->mesh_buffer +
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band_index,
            implementation->tp_rank));
    fprintf(stderr,
        "STUCK-TAILS rank=%u seq=%llu parity=%llu doorbell=(%llu,%llu,%llu) tails:",
        implementation->tp_rank,
        (unsigned long long)sequence,
        (unsigned long long)parity,
        (unsigned long long)entry[0],(unsigned long long)entry[1],
        (unsigned long long)entry[2]);
    for ( peer = 0u;
          peer < 16u && peer < implementation->tp_degree; peer++ )
    {
        volatile uint64_t *end_word;
        if ( peer == implementation->tp_rank )
        {
            fprintf(stderr," self");
            continue;
        }
        end_word = (volatile uint64_t *)(implementation->mesh_buffer +
            implementation->band_base +
            ((uint64_t)peer *
                SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + parity) *
            SPARK_WEIGHTD_MESH_SLOT_BYTES +
            SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u);
        fprintf(stderr," %u:%llu",peer,(unsigned long long)*end_word);
    }
    fprintf(stderr,"\n");
    return(sequence);
}

uint64_t SparkTpDeviceCollectiveGraphDiag(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    uint64_t diag = 0u;
    if ( collective == 0 || collective->implementation == 0 )
        return(0ull);
    implementation = collective->implementation;
    if ( implementation->diag_word == 0 )
        return(0ull);
    if ( cudaMemcpy(&diag,implementation->diag_word,sizeof(uint64_t),
            SPARK_TP_CUDA_MEMCPY_DEVICE_TO_HOST) != 0 )
        return(0ull);
    return(diag);
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
    if ( implementation->completion_thread_live != 0u )
    {
        pthread_mutex_lock(&implementation->completion_lock);
        implementation->completion_stop = 1u;
        pthread_cond_signal(&implementation->completion_wake);
        pthread_mutex_unlock(&implementation->completion_lock);
        pthread_join(implementation->completion_thread,0);
    }
    SparkWeightdClientClose(implementation->client);
    free(implementation);
    collective->implementation = 0;
}

void SparkTpDeviceCollectiveRoundStats(
    SparkTpDeviceCollective *collective,
    uint64_t *count_out,
    uint64_t *total_ns_out,
    uint32_t reset)
{
    SparkTpDeviceCollectiveImplementation *implementation;
    if ( collective == 0 || collective->implementation == 0 )
        return;
    implementation = collective->implementation;
    if ( count_out != 0 )
        *count_out = implementation->round_count;
    if ( total_ns_out != 0 )
        *total_ns_out = implementation->round_ns_total;
    if ( reset != 0u )
    {
        implementation->round_count = 0u;
        implementation->round_ns_total = 0u;
    }
}
