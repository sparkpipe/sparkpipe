#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_tp_mesh_register.h"
#include <poll.h>
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
    double position;
    if (timed->count == 0u)
        return 0.0;
    memcpy(sorted,timed->values,sizeof(double) * timed->count);
    qsort(sorted,timed->count,sizeof(double),ladder_compare_double);
    position = fraction * (double)(timed->count - 1u);
    index = (uint32_t)position;
    return index + 1u == timed->count ? sorted[index] :
        sorted[index] + (position - index) * (sorted[index + 1u] - sorted[index]);
}

static void ladder_completion_mark(void *context,
    const SparkTpDeviceCollectiveCompletion *completion)
{
    struct LadderCompletion *state = (struct LadderCompletion *)context;
    pthread_mutex_lock(&state->lock);
    state->done++;
    state->status = (uint32_t)completion->status;
    pthread_cond_signal(&state->wake);
    pthread_mutex_unlock(&state->lock);
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
    status = SparkWeightdClientLaneAcquire(client,SPARK_WEIGHTD_LANE_NONE,0,&lane,LADDER_WAIT_NS);
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

static void ladder_require(int valid,const char *phase)
{
    if (valid != 0) return;
    fprintf(stderr,"LADDER-FAIL phase=%s\n",phase);
    exit(1);
}

static void ladder_status(SparkStatus status,const char *phase)
{
    if (status == SPARK_STATUS_OK) return;
    fprintf(stderr,"LADDER-FAIL phase=%s status=%s\n",phase,SparkStatusToString(status));
    exit(1);
}

static void ladder_wait_stream(cudaStream_t stream)
{
    uint64_t deadline = ladder_now_ns() + LADDER_WAIT_NS;
    for (;;)
    {
        cudaError_t status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        ladder_require(status == cudaErrorNotReady && ladder_now_ns() < deadline,"stream-terminal");
        usleep(100u);
    }
}

static uint32_t ladder_number(const char *text,uint32_t minimum,uint32_t maximum)
{
    uint64_t value = 0u;
    ladder_require(text != 0 && *text != 0,"empty-number");
    for (const char *p = text; *p != 0; p++)
    {
        ladder_require(*p >= '0' && *p <= '9',"invalid-number");
        value = value * 10u + (uint32_t)(*p - '0');
        ladder_require(value <= maximum,"number-range");
    }
    ladder_require(value >= minimum,"number-range");
    return (uint32_t)value;
}

static void ladder_barrier(char expected)
{
    struct pollfd input = {STDIN_FILENO,POLLIN,0};
    char actual = 0;
    ladder_require(poll(&input,1u,60000) == 1 &&
        (input.revents & POLLIN) != 0 && read(STDIN_FILENO,&actual,1u) == 1 &&
        actual == expected,"coordinator-barrier");
}

static uint16_t ladder_bf16(float value)
{
    uint32_t bits;
    memcpy(&bits,&value,sizeof(bits));
    return (uint16_t)(bits >> 16u);
}

static float ladder_value(uint32_t lane,uint32_t rank,uint32_t ordinal,uint32_t index)
{
    if (index == 0u)
        return rank == 0u ? 256.0f : rank == 2u ? -256.0f :
            (rank == 1u || rank == 3u) ? 1.0f : 0.0f;
    return (float)((int32_t)((lane * 19u + rank * 13u + ordinal * 23u + index * 7u) % 127u) - 63);
}

static uint64_t ladder_u64(uint32_t lane,uint32_t rank,uint32_t degree,uint32_t ordinal,uint32_t index)
{
    return (rank == index % degree ? UINT64_C(0x8000000000000000) : 0u) |
        ((uint64_t)(lane + 1u) << 52u) | ((uint64_t)(ordinal + 1u) << 32u) |
        ((uint64_t)(rank + 1u) << 16u) | index;
}

static int ladder_mesh_run(const char *socket_path,uint32_t rank,
    uint32_t degree,uint32_t iters,uint32_t rows,uint32_t lane,
    const char *pack_path,const char *rank_map)
{
    SparkTpDeviceCollectiveConfig config = {};
    SparkTpDeviceCollective collective = {};
    SparkTpDeviceCollectiveSubmission submission = {};
    SparkWeightdMeshTopology topology = {};
    SparkWeightdLazyAttachRequest request = {};
    SparkWeightdLazyAttachResult attached = {};
    SparkWeightdDetachResult detached = {};
    SparkWeightdClient *owner = 0;
    LadderCompletion completion = {};
    LadderTimed timed = {};
    cudaGraph_t graphs[3] = {};
    cudaGraphExec_t executables[3] = {};
    cudaStream_t stream = 0;
    uint8_t *local = 0,*output = 0;
    uint32_t acquired_lane = SPARK_WEIGHTD_LANE_NONE;
    uint32_t elements = rows * LADDER_HIDDEN;
    size_t capacity = (size_t)elements * degree * sizeof(uint16_t);
    uint8_t *input = (uint8_t *)malloc(capacity);
    uint8_t *actual = (uint8_t *)malloc(capacity + 256u);
    uint8_t *input_after = (uint8_t *)malloc(capacity);
    const uint32_t operations[3] = {1u,2u,0u};
    uint32_t completed = 0u;
    uint32_t callback_count = 0u;
    alarm(180u);
    ladder_require(input != 0 && actual != 0 && input_after != 0,"host-allocation");
    ladder_require(setenv("SPARK_WEIGHTD_SOCKET",socket_path,1) == 0 &&
        setenv("SPARK_TP_MESH_RANKS",rank_map,1) == 0,"environment");
    ladder_require(getenv("SPARK_TP_WAIT_MODE") != 0 &&
        strcmp(getenv("SPARK_TP_WAIT_MODE"),"hardware") == 0,"hardware-mode-required");
    ladder_status(SparkTpDeviceCollectiveMeshTopology(rank,degree,&topology),"topology");
    ladder_require(cudaFree(0) == cudaSuccess,"cuda-init");
    ladder_status(SparkWeightdClientConnect(socket_path,&owner,0),"connect");
    ladder_status(SparkWeightdClientLaneAcquire(owner,lane,&topology,&acquired_lane,
        LADDER_WAIT_NS),"lane-acquire");
    ladder_require(acquired_lane == lane,"exact-lane");
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = LADDER_PACK_BYTES;
    memcpy(request.identity.model,"mesh-lane-evict",16u);
    request.expert_pool_bytes = 3u * LADDER_CHUNK;
    ladder_require(snprintf(request.pack_path,sizeof(request.pack_path),"%s",pack_path) <
        (int)sizeof(request.pack_path),"pack-path");
    ladder_status(SparkSha256File(pack_path,request.identity.pack_sha256),"pack-sha");
    ladder_status(SparkWeightdClientAttachLazy(owner,&request,&attached,LADDER_WAIT_NS),"attach");
    ladder_status(attached.status,"attach-result");
    ladder_require(attached.mesh_ready != 0u && attached.mesh_mapping != 0 &&
        attached.mesh_send_buffer_bytes == SPARK_WEIGHTD_MESH_REGION_BYTES,"real-mesh-mapping");
    if (attached.pool_fd >= 0) ladder_require(close(attached.pool_fd) == 0,"pool-fd-close");
    ladder_require(cudaMalloc((void **)&local,capacity) == cudaSuccess &&
        cudaMalloc((void **)&output,capacity + 256u) == cudaSuccess &&
        cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess,"cuda-allocation");
    ladder_require(pthread_mutex_init(&completion.lock,0) == 0,"completion-lock");
    pthread_condattr_t attr;
    ladder_require(pthread_condattr_init(&attr) == 0 &&
        pthread_condattr_setclock(&attr,CLOCK_MONOTONIC) == 0 &&
        pthread_cond_init(&completion.wake,&attr) == 0 &&
        pthread_condattr_destroy(&attr) == 0,"completion-condition");
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
    config.mesh_lane_client = owner;
    config.mesh_band_index = 0u;
    config.collective_identifier = 2u * (uint64_t)lane;
    SparkTpMeshRegisterCommonCombines(&config);
    ladder_status(SparkTpDeviceCollectiveCreate(&config,&collective),"collective-create");
    ladder_status(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,
        attached.mesh_mapping,0u,0u,0u,0),"collective-prepare");
    printf("LADDER-READY lane=%u rank=%u physical=%u degree=%u rows=%u map=%s transport=rdma\n",
        lane,rank,topology.physical_ranks[rank],degree,rows,rank_map);
    fflush(stdout);
    ladder_barrier('G');
    ladder_status(SparkTpDeviceCollectiveChainKey(&collective,1u),"chain-key");
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = rows;
    submission.logical_sequence_count = rows;
    submission.local_device = local;
    submission.full_device = output + 128u;
    submission.cuda_stream = stream;
    submission.completion_context = &completion;
    for (uint32_t ordinal = 0u; ordinal <= iters; ordinal++)
    {
        for (uint32_t op_index = 0u; op_index < 3u; op_index++)
        {
            uint32_t operation = operations[op_index];
            size_t local_bytes = operation == 2u ? (size_t)rows * 8u : (size_t)elements * 2u;
            size_t result_bytes = operation == 0u ? local_bytes * degree : local_bytes;
            if (operation == 2u)
                for (uint32_t i = 0u; i < rows; i++)
                    ((uint64_t *)input)[i] = ladder_u64(lane,rank,degree,ordinal,i);
            else
                for (uint32_t i = 0u; i < elements; i++)
                    ((uint16_t *)input)[i] = ladder_bf16(ladder_value(lane,rank,ordinal,i));
            ladder_require(cudaMemcpyAsync(local,input,local_bytes,cudaMemcpyHostToDevice,stream) == cudaSuccess &&
                cudaMemsetAsync(output,0xa5,capacity + 256u,stream) == cudaSuccess,"round-input");
            uint64_t started = ladder_now_ns();
            if (ordinal == 0u)
            {
                submission.ordinal = op_index;
                submission.completion_function = ladder_completion_mark;
                ladder_status(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation),"enqueue");
                struct timespec deadline;
                ladder_require(clock_gettime(CLOCK_MONOTONIC,&deadline) == 0,"completion-clock");
                deadline.tv_sec += 60;
                pthread_mutex_lock(&completion.lock);
                while (completion.done == callback_count)
                    if (pthread_cond_timedwait(&completion.wake,&completion.lock,&deadline) != 0) break;
                ladder_require(completion.done == callback_count + 1u,"exact-completion-count");
                ladder_status((SparkStatus)completion.status,"completion-status");
                callback_count = completion.done;
                pthread_mutex_unlock(&completion.lock);
            }
            else
            {
                ladder_status(SparkTpDeviceCollectiveGraphCancelSeed(&collective,stream),"graph-cancel-seed");
                ladder_status(SparkTpDeviceCollectiveGraphPreLaunch(&collective,stream),"graph-prelaunch");
                ladder_require(cudaGraphLaunch(executables[op_index],stream) == cudaSuccess,"graph-launch");
            }
            ladder_wait_stream(stream);
            ladder_require(SparkTpDeviceCollectiveGraphError(&collective) == 0u,"collective-error");
            double elapsed = (double)(ladder_now_ns() - started) / 1000.0;
            ladder_require(cudaMemcpy(actual,output,capacity + 256u,cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(input_after,local,local_bytes,cudaMemcpyDeviceToHost) == cudaSuccess,"readback");
            ladder_require(memcmp(input,input_after,local_bytes) == 0,"input-preserved");
            for (size_t i = 0u; i < capacity + 256u; i++)
                if (i < 128u || i >= 128u + result_bytes)
                    ladder_require(actual[i] == 0xa5u,"output-sentinel");
            for (uint32_t i = 0u; i < (operation == 2u ? rows : elements * (operation == 0u ? degree : 1u)); i++)
            {
                if (operation == 2u)
                {
                    uint64_t expected = 0u;
                    for (uint32_t peer = 0u; peer < degree; peer++)
                    {
                        uint64_t value = ladder_u64(lane,peer,degree,ordinal,i);
                        if (value > expected) expected = value;
                    }
                    ladder_require(((uint64_t *)(actual + 128u))[i] == expected,"unsigned-max");
                }
                else
                {
                    float expected = 0.0f;
                    if (operation == 0u) expected = ladder_value(lane,i / elements,ordinal,i % elements);
                    else for (uint32_t peer = 0u; peer < degree; peer++)
                        expected += ladder_value(lane,peer,ordinal,i);
                    ladder_require(((uint16_t *)(actual + 128u))[i] == ladder_bf16(expected),
                        operation == 0u ? "gather-logical-order" : "sum-fp32-final-bf16");
                }
            }
            if (timed.count < LADDER_MAX_TIMED) timed.values[timed.count++] = elapsed;
            completed++;
            printf("ROUND rank=%u lane=%u ordinal=%u operation=%u graph=%u latency_us=%.1f status=ok\n",
                rank,lane,ordinal,operation,ordinal != 0u,elapsed);
            fflush(stdout);
        }
        if (ordinal == 0u)
        {
            submission.completion_function = 0;
            for (uint32_t op_index = 0u; op_index < 3u; op_index++)
            {
                ladder_status(SparkTpDeviceCollectiveArmCapture(&collective),"capture-arm");
                ladder_require(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) == cudaSuccess,"capture-begin");
                ladder_status(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operations[op_index]),"capture-enqueue");
                ladder_require(cudaStreamEndCapture(stream,&graphs[op_index]) == cudaSuccess && graphs[op_index] != 0,"capture-end");
                ladder_status(SparkTpDeviceCollectiveDisarmCapture(&collective),"capture-disarm");
                ladder_require(cudaGraphInstantiate(&executables[op_index],graphs[op_index],0) == cudaSuccess,"graph-instantiate");
            }
        }
    }
    ladder_status(SparkTpDeviceCollectiveEndChain(&collective,stream),"chain-end");
    printf("LADDER-DONE lane=%u rank=%u rounds=%u callbacks=%u\n",lane,rank,completed,callback_count);
    fflush(stdout);
    ladder_barrier('R');
    for (uint32_t i = 0u; i < 3u; i++)
        ladder_require(cudaGraphExecDestroy(executables[i]) == cudaSuccess &&
            cudaGraphDestroy(graphs[i]) == cudaSuccess,"graph-destroy");
    SparkTpDeviceCollectiveDestroy(&collective);
    ladder_require(collective.implementation == 0,"collective-destroy");
    ladder_require(cudaFree(local) == cudaSuccess && cudaFree(output) == cudaSuccess &&
        cudaStreamDestroy(stream) == cudaSuccess,"cuda-destroy");
    ladder_require(munmap(attached.mesh_mapping,attached.mesh_send_buffer_bytes) == 0,"mesh-unmap");
    ladder_status(SparkWeightdClientDetach(owner,attached.arena_generation,&detached,LADDER_WAIT_NS),"detach");
    ladder_status(detached.status,"detach-result");
    SparkWeightdClientClose(owner);
    ladder_require(pthread_cond_destroy(&completion.wake) == 0 &&
        pthread_mutex_destroy(&completion.lock) == 0,"completion-destroy");
    free(input); free(actual); free(input_after);
    printf("SUMMARY rank=%u lane=%u rounds=%u ok=%u bad=0 callbacks=%u p50_us=%.1f p99_us=%.1f max_us=%.1f transport=rdma\n",
        rank,lane,completed,completed,callback_count,
        ladder_percentile(&timed,0.50),ladder_percentile(&timed,0.99),ladder_percentile(&timed,1.0));
    return 0;
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
        status = SparkWeightdClientLaneAcquire(clients[i],SPARK_WEIGHTD_LANE_NONE,0,&released,
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
            "       mesh_lane_ladder SOCKET mesh RANK DEGREE ITERS ROWS LANE FIXTURE_PACK RANK_MAP\n"
            "       mesh_lane_ladder SOCKET evict LANES PACK\n");
        return 2;
    }
    socket_path = arguments[1];
    if (strcmp(arguments[2],"hold") == 0 && argument_count == 3)
        return ladder_hold_lane(socket_path,0u);
    if (strcmp(arguments[2],"ephemeral") == 0 && argument_count == 3)
        return ladder_hold_lane(socket_path,1u);
    if (strcmp(arguments[2],"mesh") == 0 && argument_count == 10)
    {
        uint32_t degree = ladder_number(arguments[4],2u,16u);
        return ladder_mesh_run(socket_path,
            ladder_number(arguments[3],0u,degree - 1u),degree,
            ladder_number(arguments[5],1u,1024u),
            ladder_number(arguments[6],1u,512u),
            ladder_number(arguments[7],0u,7u),arguments[8],arguments[9]);
    }
    if (strcmp(arguments[2],"evict") == 0 && argument_count == 5)
    {
        return ladder_evict_matrix(socket_path,
            ladder_number(arguments[3],2u,8u),arguments[4]);
    }
    fprintf(stderr,"usage: mesh_lane_ladder SOCKET hold | SOCKET mesh RANK DEGREE ITERS ROWS LANE FIXTURE_PACK RANK_MAP | SOCKET evict LANES PACK\n");
    return 2;
}
