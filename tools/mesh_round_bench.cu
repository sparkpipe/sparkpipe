#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static SparkStatus SparkStageModuleCudaStatus(const char *module_tag, SparkGpuError error, const char *site)
{
	if ( error == SPARK_GPU_SUCCESS )
		return SPARK_STATUS_OK;
	fprintf(stderr, "%s gpu_error site=%s error=%s\n", module_tag, site, SparkGpuGetErrorString(error));
	return error == SPARK_GPU_ERROR_MEMORY_ALLOCATION ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_INTERNAL_ERROR;
}

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

static void bench_wait_stream(SparkGpuStream stream)
{
    uint64_t deadline = bench_now_ns() + BENCH_WAIT_NS;
    for (;;)
    {
        SparkGpuError status = SparkGpuStreamQuery(stream);
        if (status == SPARK_GPU_SUCCESS) return;
        bench_require(status == SPARK_GPU_ERROR_NOT_READY && bench_now_ns() < deadline,"stream-terminal");
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

__global__ void BenchWorkKernel(const uint4 *work,uint64_t quads,uint32_t *sink)
{
    const uint64_t share = quads / gridDim.x,first = (uint64_t)blockIdx.x * share;
    uint32_t sum = 0u;
    for (uint64_t i = first + threadIdx.x; i < first + share; i += blockDim.x)
        sum += work[i].x;
    if (sum == UINT32_MAX) atomicAdd(sink,1u);
}

static int bench_compare(const void *a,const void *b)
{
    double x = *(const double *)a,y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int count,char **arguments)
{
    if (count < 12 || count > 14)
    {
        fprintf(stderr,"usage: mesh_round_bench SOCKET RANK DEGREE LANE RANK_MAP OPERATION ROWS ROW_ELEMENTS ROUNDS REPS START_UNIX_MS [WORK_KIB [PREFETCH]]\n");
        return 2;
    }
    const uint64_t work_words = count >= 13 ? strtoull(arguments[12],0,10) * 256u : 0u;
    const uint32_t prefetch = count == 14 && atoi(arguments[13]) != 0 ? 1u : 0u;
    LmWeightPrefetchPlan plan = {};
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
    SparkGpuGraph graph = 0;
    SparkGpuGraphExec executable = 0;
    SparkGpuStream stream = 0;
    uint16_t *local = 0,*output = 0;
    uint32_t *work = 0,*sink = 0;
    uint32_t acquired = SPARK_WEIGHTD_LANE_NONE;
    double timed[BENCH_MAX_REPS];
    bench_require(degree >= 2u && degree <= 16u && rank < degree && rows >= 1u && row_elements >= 4u &&
        rounds >= 1u && reps >= 1u && reps <= BENCH_MAX_REPS &&
        (operation == 0u || operation == 1u || operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ||
         operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16 ||
         operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_REDUCE_SCATTER_SUM_BF16 ||
         operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_ALL_REDUCE_SUM_BF16),"arguments");
    const uint32_t exchange = operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? 1u : 0u;
    const size_t block_elements = (size_t)rows * row_elements;
    const size_t local_elements = exchange != 0u ? block_elements * degree : block_elements;
    const size_t output_elements = operation == 0u ? local_elements * degree : exchange != 0u ? local_elements :
        operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16 || operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_REDUCE_SCATTER_SUM_BF16 ?
            local_elements / degree : local_elements;
    {
        const char *cursor = rank_map;
        for (uint32_t i = 0u; i < degree; i++)
        {
            char *end = 0;
            const unsigned long physical = strtoul(cursor,&end,10);
            bench_require(end != cursor && physical < 16u && (i + 1u == degree ? *end == '\0' : *end == ','),"rank-map");
            config.mesh_ranks[i] = (uint32_t)physical;
            cursor = end + 1;
        }
    }
    bench_status(SparkTpDeviceCollectiveMeshTopology(rank,degree,config.mesh_ranks,&topology),"topology");
    bench_require(SparkGpuSetDevice(0) == SPARK_GPU_SUCCESS,"device-init");
    bench_status(SparkWeightdClientConnect(socket_path,&owner,0),"connect");
    bench_status(SparkWeightdClientLaneAcquire(owner,lane,&topology,&acquired,BENCH_WAIT_NS),"lane-acquire");
    bench_require(acquired == lane,"exact-lane");
    bench_require(SparkGpuMalloc((void **)&local,local_elements * 2u) == SPARK_GPU_SUCCESS &&
        SparkGpuMalloc((void **)&output,output_elements * 2u + 256u) == SPARK_GPU_SUCCESS &&
        SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS,"device-allocation");
    bench_require(work_words == 0u || (SparkGpuMalloc((void **)&work,(size_t)rounds * work_words * sizeof(*work)) == SPARK_GPU_SUCCESS && SparkGpuMalloc((void **)&sink,sizeof(*sink)) == SPARK_GPU_SUCCESS &&
        SparkGpuMemset(work,1,(size_t)rounds * work_words * sizeof(*work)) == SPARK_GPU_SUCCESS && SparkGpuMemset(sink,0,sizeof(*sink)) == SPARK_GPU_SUCCESS),"work-allocation");
    {
        uint16_t *host = (uint16_t *)malloc(local_elements * 2u);
        for (size_t i = 0u; i < local_elements; i++)
            host[i] = bench_bf16(bench_value(rank,(uint32_t)i));
        bench_require(SparkGpuMemcpy(local,host,local_elements * 2u,SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS,"input");
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
    config.weightd_socket_path = socket_path;
    config.mesh_lane_client = owner;
    config.mesh_lane = lane;
    config.mesh_band_index = 0u;
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
    if (prefetch != 0u)
    {
        plan.host = (LmWeightPrefetchGroup *)calloc(LM_WEIGHT_PREFETCH_GROUPS,sizeof(LmWeightPrefetchGroup));
        bench_require(plan.host != 0 && SparkGpuMalloc((void **)&plan.device,(size_t)LM_WEIGHT_PREFETCH_GROUPS * sizeof(LmWeightPrefetchGroup)) == SPARK_GPU_SUCCESS,"prefetch-plan");
        plan.capacity = LM_WEIGHT_PREFETCH_GROUPS;
        LmWeightPrefetchArm(&plan,stream);
    }
    bench_require(SparkGpuStreamBeginCapture(stream,SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL) == SPARK_GPU_SUCCESS,"capture-begin");
    for (uint32_t r = 0u; r < rounds; r++)
    {
        if (work_words != 0u)
        {
            LmWeightPrefetchNote(stream,work + (uint64_t)r * work_words,work_words * sizeof(*work));
            LM_LAUNCH((BenchWorkKernel),96u,256u,0,stream,(const uint4 *)(work + (uint64_t)r * work_words),work_words / 4u,sink);
        }
        submission.ordinal = ordinal++;
        bench_status(SparkTpDeviceCollectiveEnqueue(&collective,&submission,operation),"capture-enqueue");
    }
    bench_require(SparkGpuStreamEndCapture(stream,&graph) == SPARK_GPU_SUCCESS && graph != 0,"capture-end");
    bench_require(LmWeightPrefetchDisarm(&plan,1u) == SPARK_GPU_SUCCESS && plan.dropped == 0u,"prefetch-publish");
    bench_status(SparkTpDeviceCollectiveDisarmCapture(&collective),"capture-disarm");
    bench_require(SparkGpuGraphInstantiate(&executable,graph,0) == SPARK_GPU_SUCCESS,"graph-instantiate");
    SparkTpDeviceCollectiveHardwareTiming before = {},after = {};
    bench_status(SparkTpDeviceCollectiveHardwareStats(&collective,&before),"stats-before");
    for (uint32_t rep = 0u; rep < reps; rep++)
    {
        bench_status(SparkTpDeviceCollectiveGraphCancelSeed(&collective,stream),"graph-cancel-seed");
        bench_status(SparkTpDeviceCollectiveGraphPreLaunch(&collective,stream),"graph-prelaunch");
        uint64_t started = bench_now_ns();
        bench_require(SparkGpuGraphLaunch(executable,stream) == SPARK_GPU_SUCCESS,"graph-launch");
        bench_wait_stream(stream);
        timed[rep] = (double)(bench_now_ns() - started) / 1000.0 / rounds;
        bench_require(SparkTpDeviceCollectiveGraphError(&collective) == 0u,"collective-error");
    }
    bench_status(SparkTpDeviceCollectiveHardwareStats(&collective,&after),"stats-after");
    {
        uint16_t *host = (uint16_t *)malloc(output_elements * 2u);
        uint64_t digest = UINT64_C(1469598103934665603);
        uint32_t bad = 0u;
        bench_require(SparkGpuMemcpy(host,output + 64u,output_elements * 2u,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"readback");
        for (size_t i = 0u; i < output_elements; i++)
        {
            float expected = 0.0f;
            if (operation == 0u)
            {
                const size_t per = local_elements;
                expected = bench_value((uint32_t)(i / per),(uint32_t)(i % per));
            }
            else if (exchange != 0u)
                expected = bench_value((uint32_t)(i / block_elements),(uint32_t)(rank * block_elements + i % block_elements));
            else
                for (uint32_t peer = 0u; peer < degree; peer++)
                    expected += bench_value(peer,(uint32_t)(operation == 1u || operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_ALL_REDUCE_SUM_BF16 ?
                        i : rank * output_elements + i));
            bad += host[i] != bench_bf16(expected) ? 1u : 0u;
            digest = (digest ^ host[i]) * UINT64_C(1099511628211);
        }
        free(host);
        printf("BENCH-OUTPUT rank=%u elements=%llu digest=%016llx mismatches=%u\n",rank,(unsigned long long)output_elements,(unsigned long long)digest,bad);
        bench_require(bad == 0u,"result-mismatch");
    }
    qsort(timed,reps,sizeof(double),bench_compare);
    const double per = 1000.0 * rounds * reps;
    printf("BENCH-RESULT rank=%u operation=%u rows=%u row_elements=%u rounds=%u reps=%u work_kib=%llu prefetch_groups=%u per_round_us p50=%.2f min=%.2f max=%.2f source_wait=%.2f peer_wait=%.2f copy=%.2f combine=%.2f first_arrival=%.2f\n",
        rank,operation,rows,row_elements,rounds,reps,(unsigned long long)(work_words / 256u),plan.groups,timed[reps / 2u],timed[0],timed[reps - 1u],
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
    bench_require(SparkGpuGraphExecDestroy(executable) == SPARK_GPU_SUCCESS && SparkGpuGraphDestroy(graph) == SPARK_GPU_SUCCESS,"graph-destroy");
    SparkTpDeviceCollectiveDestroy(&collective);
    (void)SparkGpuFree(local);
    (void)SparkGpuFree(output);
    if (work != 0) (void)SparkGpuFree(work);
    if (sink != 0) (void)SparkGpuFree(sink);
    if (plan.device != 0) (void)SparkGpuFree(plan.device);
    free(plan.host);
    (void)SparkGpuStreamDestroy(stream);
    SparkWeightdClientClose(owner);
    return 0;
}
