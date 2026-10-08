#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include <poll.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

extern "C" SparkStatus SparkStageModuleCudaStatus(const char *module_tag, cudaError_t error, const char *site);

#define SPARK_FAMILY_CAMEL MeshBench
#define SPARK_FAMILY_UPPER MESH_BENCH
#define SPARK_FAMILY_LOWER mesh_bench
#include "sparkpipe/family/spark_family.h"
#define SPARK_MESH_BENCH_MODULE_TAG "mesh_round_bench"
#include "sparkpipe/family/module/spark_module_combine.h"

#define BENCH_WAIT_NS UINT64_C(60000000000)
#define BENCH_MAX_REPS 256u

static uint64_t bench_now_ns(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void bench_require(int valid,const char *phase)
{
    if (valid != 0) return;
    fprintf(stderr,"BENCH-FAIL phase=%s\n",phase);
    exit(1);
}

static void bench_status(SparkStatus status,const char *phase)
{
    if (status == SPARK_STATUS_OK) return;
    fprintf(stderr,"BENCH-FAIL phase=%s status=%s\n",phase,SparkStatusToString(status));
    exit(1);
}

static void bench_start_at(uint64_t start_ms)
{
    struct timespec now;
    for (;;)
    {
        clock_gettime(CLOCK_REALTIME,&now);
        if ((uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u >= start_ms) return;
        usleep(200u);
    }
}

static void bench_wait_stream(cudaStream_t stream)
{
    uint64_t deadline = bench_now_ns() + BENCH_WAIT_NS;
    for (;;)
    {
        cudaError_t status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        bench_require(status == cudaErrorNotReady && bench_now_ns() < deadline,"stream-terminal");
    }
}

static uint16_t bench_bf16(float value)
{
    uint32_t bits;
    memcpy(&bits,&value,sizeof(bits));
    return (uint16_t)(bits >> 16u);
}

static float bench_value(uint32_t rank,uint32_t index)
{
    return (float)((int32_t)((rank * 13u + index * 7u) % 17u) - 8);
}

static int bench_compare(const void *a,const void *b)
{
    double x = *(const double *)a,y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int count,char **arguments)
{
    if (count != 12)
    {
        fprintf(stderr,"usage: mesh_round_bench SOCKET RANK DEGREE LANE RANK_MAP OPERATION ROWS ROW_ELEMENTS ROUNDS REPS START_UNIX_MS\n");
        return 2;
    }
    const uint64_t start_ms = strtoull(arguments[11],0,10);
    const char *socket_path = arguments[1];
    const uint32_t rank = (uint32_t)atoi(arguments[2]),degree = (uint32_t)atoi(arguments[3]),lane = (uint32_t)atoi(arguments[4]);
    const char *rank_map = arguments[5];
    const uint32_t operation = (uint32_t)atoi(arguments[6]),rows = (uint32_t)atoi(arguments[7]);
    const uint32_t row_elements = (uint32_t)atoi(arguments[8]),rounds = (uint32_t)atoi(arguments[9]),reps = (uint32_t)atoi(arguments[10]);
    SparkTpDeviceCollectiveConfig config = {};
    SparkTpDeviceCollective collective = {};
    SparkTpDeviceCollectiveSubmission submission = {};
    SparkWeightdMeshTopology topology = {};
    SparkWeightdClient *owner = 0;
    cudaGraph_t graph = 0;
    cudaGraphExec_t executable = 0;
    cudaStream_t stream = 0;
    uint16_t *local = 0,*output = 0;
    uint32_t acquired = SPARK_WEIGHTD_LANE_NONE;
    double timed[BENCH_MAX_REPS];
    bench_require(degree >= 2u && degree <= 16u && rank < degree && rows >= 1u && row_elements >= 4u &&
        rounds >= 1u && reps >= 1u && reps <= BENCH_MAX_REPS && (operation == 0u || operation == 1u),"arguments");
    const size_t local_elements = (size_t)rows * row_elements;
    const size_t output_elements = operation == 0u ? local_elements * degree : local_elements;
    bench_require(setenv("SPARK_WEIGHTD_SOCKET",socket_path,1) == 0 && setenv("SPARK_TP_MESH_RANKS",rank_map,1) == 0,"environment");
    bench_status(SparkTpDeviceCollectiveMeshTopology(rank,degree,&topology),"topology");
    bench_require(cudaFree(0) == cudaSuccess,"cuda-init");
    bench_status(SparkWeightdClientConnect(socket_path,&owner,0),"connect");
    bench_status(SparkWeightdClientLaneAcquire(owner,lane,&topology,&acquired,BENCH_WAIT_NS),"lane-acquire");
    bench_require(acquired == lane,"exact-lane");
    bench_require(cudaMalloc((void **)&local,local_elements * 2u) == cudaSuccess &&
        cudaMalloc((void **)&output,output_elements * 2u + 256u) == cudaSuccess &&
        cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess,"cuda-allocation");
    {
        uint16_t *host = (uint16_t *)malloc(local_elements * 2u);
        for (size_t i = 0u; i < local_elements; i++)
            host[i] = bench_bf16(bench_value(rank,(uint32_t)i));
        bench_require(cudaMemcpy(local,host,local_elements * 2u,cudaMemcpyHostToDevice) == cudaSuccess,"input");
        free(host);
    }
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
    config.tp_degree = degree;
    config.tp_rank = rank;
    config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    config.credit_count = 1u;
    config.local_hidden_dimension = row_elements;
    config.max_active_sequence_count = rows;
    config.connect_timeout_milli = 60000u;
    config.operation_timeout_milli = 30000u;
    config.wait_mode = SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE;
    config.mesh_lane_client = owner;
    config.mesh_band_index = 0u;
    config.collective_identifier = 2u * (uint64_t)lane;
    SparkMeshBenchModuleRegisterCombines(&config);
    bench_status(SparkTpDeviceCollectiveCreate(&config,&collective),"collective-create");
    void *mapping = 0;
    bench_status(SparkWeightdClientMeshMap(owner,&mapping,BENCH_WAIT_NS),"mesh-map");
    bench_status(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,mapping,0u,0u,0u,0),"mesh-prepare");
    printf("BENCH-READY rank=%u lane=%u operation=%u rows=%u row_elements=%u rounds=%u\n",rank,lane,operation,rows,row_elements,rounds);
    fflush(stdout);
    bench_start_at(start_ms);
    bench_status(SparkTpDeviceCollectiveChainKey(&collective,1u),"chain-key");
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = rows;
    submission.logical_sequence_count = rows;
    submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
    submission.local_device = local;
    submission.full_device = output + 64u;
    submission.cuda_stream = stream;
    submission.row_elements = operation == 0u ? row_elements : 0u;
    uint64_t ordinal = 0u;
    for (uint32_t warm = 0u; warm < 4u; warm++)
    {
        submission.ordinal = ordinal++;
        bench_status(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation),"eager-enqueue");
    }
    bench_wait_stream(stream);
    bench_status(SparkTpDeviceCollectiveVerifyDeferred(&collective,stream),"eager-verify");
    bench_status(SparkTpDeviceCollectiveArmCapture(&collective),"capture-arm");
    bench_require(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) == cudaSuccess,"capture-begin");
    for (uint32_t r = 0u; r < rounds; r++)
    {
        submission.ordinal = ordinal++;
        bench_status(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation),"capture-enqueue");
    }
    bench_require(cudaStreamEndCapture(stream,&graph) == cudaSuccess && graph != 0,"capture-end");
    bench_status(SparkTpDeviceCollectiveDisarmCapture(&collective),"capture-disarm");
    bench_require(cudaGraphInstantiate(&executable,graph,0) == cudaSuccess,"graph-instantiate");
    SparkTpDeviceCollectiveHardwareTiming before = {},after = {};
    bench_status(SparkTpDeviceCollectiveHardwareStats(&collective,&before),"stats-before");
    for (uint32_t rep = 0u; rep < reps; rep++)
    {
        bench_status(SparkTpDeviceCollectiveGraphCancelSeed(&collective,stream),"graph-cancel-seed");
        bench_status(SparkTpDeviceCollectiveGraphPreLaunch(&collective,stream),"graph-prelaunch");
        uint64_t started = bench_now_ns();
        bench_require(cudaGraphLaunch(executable,stream) == cudaSuccess,"graph-launch");
        bench_wait_stream(stream);
        timed[rep] = (double)(bench_now_ns() - started) / 1000.0 / rounds;
        bench_require(SparkTpDeviceCollectiveGraphError(&collective) == 0u,"collective-error");
    }
    bench_status(SparkTpDeviceCollectiveHardwareStats(&collective,&after),"stats-after");
    {
        uint16_t *host = (uint16_t *)malloc(output_elements * 2u);
        uint32_t bad = 0u;
        bench_require(cudaMemcpy(host,output + 64u,output_elements * 2u,cudaMemcpyDeviceToHost) == cudaSuccess,"readback");
        for (size_t i = 0u; i < output_elements; i++)
        {
            float expected = 0.0f;
            if (operation == 0u)
            {
                const size_t per = local_elements;
                expected = bench_value((uint32_t)(i / per),(uint32_t)(i % per));
            }
            else
                for (uint32_t peer = 0u; peer < degree; peer++)
                    expected += bench_value(peer,(uint32_t)i);
            bad += host[i] != bench_bf16(expected) ? 1u : 0u;
        }
        free(host);
        bench_require(bad == 0u,"result-mismatch");
    }
    qsort(timed,reps,sizeof(double),bench_compare);
    const double per = 1000.0 * rounds * reps;
    printf("BENCH-RESULT rank=%u operation=%u rows=%u row_elements=%u rounds=%u reps=%u per_round_us p50=%.2f min=%.2f max=%.2f source_wait=%.2f peer_wait=%.2f copy=%.2f combine=%.2f first_arrival=%.2f\n",
        rank,operation,rows,row_elements,rounds,reps,timed[reps / 2u],timed[0],timed[reps - 1u],
        (after.source_wait_ns - before.source_wait_ns) / per,(after.peer_wait_ns - before.peer_wait_ns) / per,
        (after.copy_ns - before.copy_ns) / per,(after.combine_ns - before.combine_ns) / per,
        (after.first_arrival_ns - before.first_arrival_ns) / per);
    printf("BENCH-ARRIVAL rank=%u",rank);
    for (uint32_t peer = 0u; peer < degree; peer++)
        printf(" %.2f",(after.peer_arrival_ns[peer] - before.peer_arrival_ns[peer]) / per);
    printf("\n");
    fflush(stdout);
    sleep(5u);
    bench_status(SparkTpDeviceCollectiveEndChain(&collective,stream),"chain-end");
    bench_require(cudaGraphExecDestroy(executable) == cudaSuccess && cudaGraphDestroy(graph) == cudaSuccess,"graph-destroy");
    SparkTpDeviceCollectiveDestroy(&collective);
    cudaFree(local);
    cudaFree(output);
    cudaStreamDestroy(stream);
    SparkWeightdClientClose(owner);
    return 0;
}
