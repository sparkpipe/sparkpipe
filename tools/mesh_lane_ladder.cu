#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tp_mesh_register.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define LADDER_HIDDEN 4096u
#define LADDER_MAX_TIMED 8192u
#define LADDER_SHM_BYTES SPARK_WEIGHTD_MESH_REGION_BYTES
#define LADDER_WAIT_NS UINT64_C(60000000000)
#define LADDER_CHUNK UINT64_C(2097152)
#define LADDER_PACK_BYTES (4u * LADDER_CHUNK)

struct LadderCompletion
{
    pthread_mutex_t lock;
    pthread_cond_t wake;
    uint32_t done;
    uint32_t status;
};

struct LadderTimed
{
    double values[LADDER_MAX_TIMED];
    uint32_t count;
};

static uint64_t ladder_now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static int ladder_compare_double(const void *left,const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static double ladder_percentile(const struct LadderTimed *timed,double fraction)
{
    double sorted[LADDER_MAX_TIMED];
    uint32_t index;
    if (timed->count == 0u)
        return 0.0;
    memcpy(sorted,timed->values,sizeof(double) * timed->count);
    qsort(sorted,timed->count,sizeof(double),ladder_compare_double);
    index = (uint32_t)(fraction * (double)(timed->count - 1u));
    return sorted[index];
}

static void ladder_completion_mark(void *context,
    const SparkTpDeviceCollectiveCompletion *completion)
{
    struct LadderCompletion *state = (struct LadderCompletion *)context;
    pthread_mutex_lock(&state->lock);
    state->done = 1u;
    state->status = (uint32_t)completion->status;
    pthread_cond_signal(&state->wake);
    pthread_mutex_unlock(&state->lock);
}

static __global__ void ladder_fill_bf16(__nv_bfloat16 *destination,
    float value,uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        destination[index] = __float2bfloat16(value);
}

static int ladder_acquire_lane(const char *socket_path,
    SparkWeightdClient **client_out,uint32_t *lane_out)
{
    SparkWeightdClient *client = 0;
    uint32_t lane = 0;
    SparkStatus status;
    if (SparkWeightdClientConnect(socket_path,&client,0) != SPARK_STATUS_OK)
    {
        fprintf(stderr,"LADDER lane connect failed socket=%s\n",
            socket_path);
        return 2;
    }
    status = SparkWeightdClientLaneAcquire(client,&lane,LADDER_WAIT_NS);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"LADDER lane_acquire=FAILED status=%s\n",
            SparkStatusToString(status));
        (void)SparkWeightdClientClose(client);
        return 3;
    }
    *client_out = client;
    *lane_out = lane;
    return 0;
}

static int ladder_hold_lane(const char *socket_path,uint32_t ephemeral)
{
    SparkWeightdClient *client = 0;
    uint32_t lane = 0;
    int result = ladder_acquire_lane(socket_path,&client,&lane);
    if (result != 0)
        return result;
    printf("LADDER-INIT lane=%u hold=%u\n",lane,ephemeral == 0u);
    fflush(stdout);
    if (ephemeral != 0u)
    {
        (void)SparkWeightdClientClose(client);
        return 0;
    }
    for (;;)
    {
        struct timespec pause = {0,100000000};
        nanosleep(&pause,0);
    }
    return 0;
}

static int ladder_mesh_run(const char *socket_path,uint32_t rank,
    uint32_t degree,uint32_t iters,uint32_t rows,uint32_t lane,
    const char *shm_path)
{
    SparkTpDeviceCollectiveConfig config;
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission submission;
    struct LadderCompletion completion;
    struct LadderTimed timed;
    uint16_t *verify_host = 0;
    __nv_bfloat16 *local_device = 0;
    __nv_bfloat16 *full_device = 0;
    void *shm_base = MAP_FAILED;
    double latency_us;
    uint64_t started_ns;
    uint64_t waited_ns;
    uint64_t expected_bits;
    uint64_t wait_deadline_ns;
    uint32_t elements;
    uint32_t payload_bytes;
    uint32_t ordinal;
    uint32_t bad_rounds = 0u;
    uint32_t bad_elements;
    uint32_t index;
    int shm_fd;
    int acquired;
    cudaStream_t stream = 0;
    cudaError_t error;
    SparkStatus status;

    memset(&timed,0,sizeof(timed));
    elements = rows * LADDER_HIDDEN;
    payload_bytes = elements * 2u;
    shm_fd = open(shm_path,O_RDWR | O_CREAT,0600);
    if (shm_fd < 0)
    {
        fprintf(stderr,"LADDER shm open failed errno=%d\n",errno);
        return 2;
    }
    if (ftruncate(shm_fd,(off_t)LADDER_SHM_BYTES) != 0 ||
        (shm_base = mmap(0,LADDER_SHM_BYTES,PROT_READ | PROT_WRITE,
            MAP_SHARED,shm_fd,0)) == MAP_FAILED)
    {
        fprintf(stderr,"LADDER shm map failed errno=%d\n",errno);
        (void)close(shm_fd);
        return 2;
    }
    (void)close(shm_fd);
    error = cudaFree(0);
    if (error != cudaSuccess)
    {
        fprintf(stderr,"LADDER cuda init failed\n");
        return 2;
    }
    if (cudaMalloc((void **)&local_device,payload_bytes) != cudaSuccess ||
        cudaMalloc((void **)&full_device,payload_bytes) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess)
    {
        fprintf(stderr,"LADDER cuda alloc failed\n");
        return 2;
    }
    verify_host = (uint16_t *)malloc(payload_bytes);
    if (verify_host == 0)
        return 2;
    memset(&config,0,sizeof(config));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
    config.tp_degree = degree;
    config.tp_rank = rank;
    config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    config.credit_count = 1u;
    config.local_hidden_dimension = LADDER_HIDDEN;
    config.max_active_sequence_count = rows;
    config.connect_timeout_milli = 60000u;
    config.operation_timeout_milli = 60000u;
    config.collective_identifier = 2u * (uint64_t)lane;
    SparkTpMeshRegisterCommonCombines(&config);
    status = SparkTpDeviceCollectiveCreate(&config,&collective);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"LADDER create failed status=%s\n",
            SparkStatusToString(status));
        return 2;
    }
    status = SparkTpDeviceCollectivePrepareReceiveBf16(&collective,shm_base,
        0u,0u,0u,0);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"LADDER prepare failed status=%s\n",
            SparkStatusToString(status));
        return 2;
    }
    pthread_mutex_init(&completion.lock,0);
    pthread_cond_init(&completion.wake,0);
    printf("LADDER-INIT lane=%u hold=0 rank=%u\n",lane,rank);
    fflush(stdout);
    {
        float expected_sum = (float)degree * ((float)degree + 1.0f) / 2.0f;
        __nv_bfloat16 expected = __float2bfloat16(expected_sum);
        uint16_t expected_u16;
        memcpy(&expected_u16,&expected,sizeof(expected_u16));
        expected_bits = expected_u16;
    }
    for (ordinal = 0u; ordinal < iters; ordinal++)
    {
        ladder_fill_bf16<<<(elements + 255u) / 256u,256u,0,stream>>>(
            local_device,(float)(rank + 1u),elements);
        pthread_mutex_lock(&completion.lock);
        completion.done = 0u;
        completion.status = (uint32_t)SPARK_STATUS_OK;
        pthread_mutex_unlock(&completion.lock);
        memset(&submission,0,sizeof(submission));
        submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        submission.descriptor_bytes = sizeof(submission);
        submission.slot_index = 0u;
        submission.active_sequence_count = rows;
        submission.logical_sequence_count = rows;
        submission.ordinal = ordinal;
        submission.local_device = local_device;
        submission.full_device = full_device;
        submission.cuda_stream = stream;
        submission.completion_function = ladder_completion_mark;
        submission.completion_context = &completion;
        started_ns = ladder_now_ns();
        status = SparkTpDeviceCollectiveSubmitBf16(&collective,&submission);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"LADDER submit failed ordinal=%u status=%s\n",
                ordinal,SparkStatusToString(status));
            bad_rounds++;
            break;
        }
        acquired = 0;
        wait_deadline_ns = ladder_now_ns() + LADDER_WAIT_NS;
        pthread_mutex_lock(&completion.lock);
        while (completion.done == 0u)
        {
            struct timespec pause;
            uint64_t remaining_ns;
            waited_ns = ladder_now_ns();
            if (waited_ns >= wait_deadline_ns)
                break;
            remaining_ns = wait_deadline_ns - waited_ns;
            clock_gettime(CLOCK_REALTIME,&pause);
            pause.tv_sec += (time_t)(remaining_ns / 1000000000ull);
            pause.tv_nsec += (long)(remaining_ns % 1000000000ull);
            if (pause.tv_nsec >= 1000000000L)
            {
                pause.tv_sec += 1;
                pause.tv_nsec -= 1000000000L;
            }
            pthread_cond_timedwait(&completion.wake,&completion.lock,&pause);
        }
        acquired = completion.done;
        status = (SparkStatus)completion.status;
        pthread_mutex_unlock(&completion.lock);
        waited_ns = ladder_now_ns() - started_ns;
        latency_us = (double)waited_ns / 1000.0;
        bad_elements = 0u;
        if (acquired == 0u || status != SPARK_STATUS_OK)
        {
            fprintf(stderr,
                "LADDER round-failed rank=%u ordinal=%u done=%u status=%s\n",
                rank,ordinal,acquired,SparkStatusToString(status));
            bad_rounds++;
            continue;
        }
        if (ordinal + 1u == iters || (ordinal & 7u) == 0u || ordinal < 4u)
        {
            if (cudaMemcpy(verify_host,full_device,payload_bytes,
                    cudaMemcpyDeviceToHost) != cudaSuccess)
            {
                fprintf(stderr,"LADDER verify copy failed rank=%u\n",rank);
                bad_rounds++;
                continue;
            }
            for (index = 0u; index < elements; index++)
            {
                uint64_t bits = verify_host[index];
                if (bits != expected_bits)
                    bad_elements++;
            }
            if (bad_elements != 0u)
            {
                fprintf(stderr,
                    "LADDER checksum-failed rank=%u ordinal=%u bad=%u expected_bits=%llx\n",
                    rank,ordinal,bad_elements,
                    (unsigned long long)expected_bits);
                bad_rounds++;
            }
        }
        if (timed.count < LADDER_MAX_TIMED)
            timed.values[timed.count++] = latency_us;
        printf(
            "ROUND rank=%u lane=%u ordinal=%u latency_us=%.1f status=ok\n",
            rank,lane,ordinal,latency_us);
    }
    printf(
        "SUMMARY rank=%u lane=%u rounds=%u ok=%u bad=%u p50_us=%.1f p99_us=%.1f max_us=%.1f\n",
        rank,lane,timed.count,timed.count - bad_rounds,bad_rounds,
        ladder_percentile(&timed,0.50),ladder_percentile(&timed,0.99),
        ladder_percentile(&timed,1.0));
    fflush(stdout);
    SparkTpDeviceCollectiveDestroy(&collective);
    (void)munmap(shm_base,LADDER_SHM_BYTES);
    (void)cudaFree(local_device);
    (void)cudaFree(full_device);
    (void)cudaStreamDestroy(stream);
    free(verify_host);
    return bad_rounds != 0u ? 1 : 0;
}

static int ladder_evict_matrix(const char *socket_path,uint32_t lanes,
    const char *pack_path)
{
    SparkWeightdClient *clients[SPARK_WEIGHTD_MESH_MAX_LANES];
    SparkWeightdLazyAttachResult attachments[SPARK_WEIGHTD_MESH_MAX_LANES];
    SparkWeightdWorkingSetResult leases[SPARK_WEIGHTD_MESH_MAX_LANES];
    SparkWeightdWorkingSetResult release_result;
    SparkWeightdLazyAttachRequest attach_request;
    SparkWeightdExpertKey keys[2] = {{0u,0u},{0u,1u}};
    uint32_t released;
    uint32_t i,j;
    uint32_t failures = 0u;
    SparkStatus status;
    char sha[SPARK_SHA256_HEX_BYTES];

    if (lanes < 2u || lanes > SPARK_WEIGHTD_MESH_MAX_LANES)
        return 2;
    if (SparkSha256File(pack_path,sha) != SPARK_STATUS_OK)
    {
        fprintf(stderr,"EVICT pack sha failed pack=%s\n",pack_path);
        return 2;
    }
    for (i = 0u; i < lanes; i++)
    {
        clients[i] = 0;
        memset(&attachments[i],0,sizeof(attachments[i]));
        memset(&leases[i],0,sizeof(leases[i]));
        if (SparkWeightdClientConnect(socket_path,&clients[i],0) !=
            SPARK_STATUS_OK)
        {
            fprintf(stderr,"EVICT connect failed lane-index=%u\n",i);
            return 2;
        }
        status = SparkWeightdClientLaneAcquire(clients[i],&released,
            LADDER_WAIT_NS);
        if (status != SPARK_STATUS_OK || released != i)
        {
            fprintf(stderr,
                "EVICT lane-assignment mismatch index=%u got=%u status=%s\n",
                i,released,SparkStatusToString(status));
            return 2;
        }
        printf("EVICT-LANE index=%u lane=%u\n",i,released);
        memset(&attach_request,0,sizeof(attach_request));
        attach_request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
        attach_request.identity.arena_bytes = LADDER_PACK_BYTES;
        memcpy(attach_request.identity.model,"mesh-lane-evict",16u);
        attach_request.expert_pool_bytes = 3u * LADDER_CHUNK;
        if (snprintf(attach_request.pack_path,
                sizeof(attach_request.pack_path),"%s",pack_path) >=
            (int)sizeof(attach_request.pack_path))
            return 2;
        memcpy(attach_request.identity.pack_sha256,sha,SPARK_SHA256_HEX_BYTES);
        status = SparkWeightdClientAttachLazy(clients[i],&attach_request,
            &attachments[i],LADDER_WAIT_NS);
        if (status != SPARK_STATUS_OK ||
            attachments[i].status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"EVICT attach failed index=%u status=%s\n",i,
                SparkStatusToString(status != SPARK_STATUS_OK ? status :
                attachments[i].status));
            return 2;
        }
        status = SparkWeightdClientAcquire(clients[i],
            attachments[i].arena_generation,keys,2u,&leases[i],
            LADDER_WAIT_NS);
        if (status != SPARK_STATUS_OK || leases[i].lease_identifier == 0u)
        {
            fprintf(stderr,"EVICT lease failed index=%u status=%s\n",i,
                SparkStatusToString(status));
            return 2;
        }
        printf("EVICT-LEASE index=%u lane=%u lease=%llu\n",i,i,
            (unsigned long long)leases[i].lease_identifier);
    }
    for (i = 0u; i < lanes; i++)
    {
        for (j = 0u; j < lanes; j++)
        {
            uint32_t expect_ok = i < j;
            if (i == j)
                continue;
            status = SparkWeightdClientEvict(clients[i],j,&released,
                LADDER_WAIT_NS);
            {
                uint32_t pass = expect_ok ?
                    (status == SPARK_STATUS_OK) :
                    (status == SPARK_STATUS_EVICT_DENIED);
                printf(
                    "EVICT-TEST requester=%u target=%u status=%s released=%u expected=%s %s\n",
                    i,j,SparkStatusToString(status),released,
                    expect_ok ? "ok" : "evict_denied",
                    pass ? "pass" : "FAIL");
                if (pass == 0u)
                    failures++;
            }
        }
    }
    memset(&release_result,0,sizeof(release_result));
    status = SparkWeightdClientRelease(clients[0],
        attachments[0].arena_generation,leases[0].lease_identifier,
        &release_result,LADDER_WAIT_NS);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "EVICT lane0 lease vanished without eviction status=%s\n",
            SparkStatusToString(status));
        failures++;
    }
    if (lanes > 1u)
    {
        SparkWeightdWorkingSetResult reacquired;
        memset(&reacquired,0,sizeof(reacquired));
        status = SparkWeightdClientAcquire(clients[1],
            attachments[1].arena_generation,keys,2u,&reacquired,
            LADDER_WAIT_NS);
        if (status != SPARK_STATUS_OK || reacquired.lease_identifier == 0u)
        {
            fprintf(stderr,
                "EVICT lane1 re-acquire after eviction failed status=%s\n",
                SparkStatusToString(status));
            failures++;
        }
        else
        {
            memset(&release_result,0,sizeof(release_result));
            (void)SparkWeightdClientRelease(clients[1],
                attachments[1].arena_generation,
                reacquired.lease_identifier,&release_result,LADDER_WAIT_NS);
            printf("EVICT-REACQUIRE lane=1 lease=%llu pass\n",
                (unsigned long long)reacquired.lease_identifier);
        }
    }
    printf("EVICT-MATRIX %s lanes=%u failures=%u\n",
        failures == 0u ? "PASS" : "FAIL",lanes,failures);
    fflush(stdout);
    for (i = 0u; i < lanes; i++)
        if (clients[i] != 0)
            (void)SparkWeightdClientClose(clients[i]);
    return failures == 0u ? 0 : 1;
}

int main(int argument_count,char **arguments)
{
    const char *socket_path;
    if (argument_count < 3)
    {
        fprintf(stderr,
            "usage: mesh_lane_ladder SOCKET hold\n"
            "       mesh_lane_ladder SOCKET mesh RANK DEGREE ITERS ROWS LANE SHM\n"
            "       mesh_lane_ladder SOCKET evict LANES PACK\n");
        return 2;
    }
    socket_path = arguments[1];
    if (strcmp(arguments[2],"hold") == 0 && argument_count == 3)
        return ladder_hold_lane(socket_path,0u);
    if (strcmp(arguments[2],"ephemeral") == 0 && argument_count == 3)
        return ladder_hold_lane(socket_path,1u);
    if (strcmp(arguments[2],"mesh") == 0 && argument_count == 9)
    {
        return ladder_mesh_run(socket_path,
            (uint32_t)strtoul(arguments[3],0,10),
            (uint32_t)strtoul(arguments[4],0,10),
            (uint32_t)strtoul(arguments[5],0,10),
            (uint32_t)strtoul(arguments[6],0,10),
            (uint32_t)strtoul(arguments[7],0,10),
            arguments[8]);
    }
    if (strcmp(arguments[2],"evict") == 0 && argument_count == 5)
    {
        return ladder_evict_matrix(socket_path,
            (uint32_t)strtoul(arguments[3],0,10),arguments[4]);
    }
    fprintf(stderr,"usage: mesh_lane_ladder SOCKET hold | SOCKET mesh RANK DEGREE ITERS ROWS LANE SHM | SOCKET evict LANES PACK\n");
    return 2;
}
