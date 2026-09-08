#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define BENCH_HIDDEN 4096u
#define BENCH_PORT_BASE_DEFAULT 57340u
static uint32_t BENCH_PORT_BASE = BENCH_PORT_BASE_DEFAULT;
#define BENCH_IDENTIFIER 0x444f4f52424f4f01ull
static uint32_t BENCH_CREDITS = 64u;
static uint32_t BENCH_D2A_MAX = 0u;
static uint32_t BENCH_SKEW_US = 0u;
static uint32_t BENCH_ORDINAL_JUMP = 0u;
#define BENCH_ORDINAL_JUMP_AFTER 100u
#define BENCH_ORDINAL_JUMP_DELTA (1ull << 20)
#define BENCH_MAX_TIMED_ITERS 4096u
static double bench_latency_us[BENCH_MAX_TIMED_ITERS];

static uint64_t bench_wire_ordinal(uint64_t ordinal)
{
    return BENCH_ORDINAL_JUMP != 0u && ordinal >= BENCH_ORDINAL_JUMP_AFTER ?
        ordinal + BENCH_ORDINAL_JUMP_DELTA : ordinal;
}

static int bench_compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static __global__ void bench_fill_bf16_kernel(
    __nv_bfloat16 *destination, float value, uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        destination[index] = __float2bfloat16(value);
}

static __global__ void bench_add_bf16_kernel(
    __nv_bfloat16 *destination, const __nv_bfloat16 *source, uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        destination[index] = __float2bfloat16(
            __bfloat162float(destination[index]) + __bfloat162float(source[index]));
}

static __global__ void bench_u64_max_kernel(
    uint64_t *destination, const uint64_t *source, uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements && source[index] > destination[index])
        destination[index] = source[index];
}

static SparkStatus bench_combine_bf16(
    void *context, void *destination, const void *source,
    uint32_t rows, uint32_t hidden, void *stream)
{
    uint32_t elements = rows * hidden;
    (void)context;
    bench_add_bf16_kernel<<<(elements + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16 *)destination, (const __nv_bfloat16 *)source, elements);
    return cudaGetLastError() == cudaSuccess ?
        SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus bench_combine_u64_max(
    void *context, uint64_t *destination, const uint64_t *source,
    uint32_t elements, void *stream)
{
    (void)context;
    bench_u64_max_kernel<<<(elements + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        destination, source, elements);
    return cudaGetLastError() == cudaSuccess ?
        SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

struct BenchFoldDevices
{
    const __nv_bfloat16 *devices[16];
};

static __global__ void bench_fold_all_kernel(
    __nv_bfloat16 *destination, const __nv_bfloat16 *const *rank_devices,
    uint32_t tp_rank, uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    float total;
    uint32_t rank_index;
    if (index >= elements)
        return;
    total = __bfloat162float(destination[index]);
    for (rank_index = 0u; rank_index < 16u; rank_index++)
    {
        if (rank_index != tp_rank && rank_devices[rank_index] != 0)
            total += __bfloat162float(rank_devices[rank_index][index]);
    }
    destination[index] = __float2bfloat16(total);
}

static BenchFoldDevices *bench_fold_context = 0;

static SparkStatus bench_combine_all_bf16(
    void *context, void *destination, const void *const *rank_devices,
    uint32_t tp_rank, uint32_t rows, uint32_t hidden, void *stream)
{
    BenchFoldDevices *devices = (BenchFoldDevices *)context;
    uint32_t elements = rows * hidden;
    uint32_t index;
    for (index = 0u; index < 16u; index++)
        devices->devices[index] =
            (const __nv_bfloat16 *)rank_devices[index];
    bench_fold_all_kernel<<<(elements + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16 *)destination, devices->devices, tp_rank, elements);
    return cudaGetLastError() == cudaSuccess ?
        SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static volatile int64_t bench_completions;
static volatile uint32_t bench_last_status;

static void bench_completion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
    (void)context;
    if (completion != 0 && completion->status != SPARK_STATUS_OK)
        (void)__sync_val_compare_and_swap(&bench_last_status,0u,(uint32_t)completion->status);
    __sync_fetch_and_add(&bench_completions, 1);
}

static uint64_t bench_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    SparkTpDeviceCollectiveTopology topology;
    SparkTpDeviceCollectiveConfig config;
    SparkTpDeviceCollectiveCreditBinding bindings[2048];
    SparkTpDeviceCollective collective;
    char rail_direct[16][24];
    char rail_switch[16][24];
    const char *transport_path;
    void *host_send = 0;
    void *host_receive = 0;
    void *mapped_send = 0;
    void *mapped_receive = 0;
    __nv_bfloat16 *payload[64];
    uint16_t *verify_host = 0;
    cudaStream_t stream;
    SparkTpDeviceCollectiveSubmission submission;
    uint64_t started_ns;
    uint64_t ordinal;
    uint64_t completed;
    double elapsed_us;
    uint32_t rank;
    uint32_t degree;
    uint32_t iters;
    uint32_t rows;
    uint32_t mode;
    uint32_t memory_mode;
    uint32_t route_count;
    uint32_t credit_bytes;
    uint32_t total_bytes;
    uint32_t binding_count;
    uint32_t offset;
    uint32_t route;
    uint32_t credit;
    uint32_t index;
    uint32_t bad;
    uint32_t step;
    SparkStatus status;
    int retry;

    {
        const char *credits_env = getenv("BENCH_CREDITS");
        if (credits_env != 0 && credits_env[0] >= '0' && credits_env[0] <= '9')
            BENCH_CREDITS = (uint32_t)strtoul(credits_env,0,10);
        if (BENCH_CREDITS == 0u || BENCH_CREDITS > 64u)
        {
            fprintf(stderr,"BENCH_CREDITS must be between 1 and 64\n");
            return 2;
        }
        const char *d2a_env = getenv("BENCH_D2A_MAX_BYTES");
        if (d2a_env != 0 && d2a_env[0] >= '0' && d2a_env[0] <= '9')
            BENCH_D2A_MAX = (uint32_t)strtoul(d2a_env,0,10);
        const char *skew_env = getenv("BENCH_SKEW_US");
        if (skew_env != 0 && skew_env[0] >= '0' && skew_env[0] <= '9')
            BENCH_SKEW_US = (uint32_t)strtoul(skew_env,0,10);
        const char *jump_env = getenv("BENCH_ORDINAL_BASE_JUMP");
        if (jump_env != 0 && jump_env[0] == '1')
            BENCH_ORDINAL_JUMP = 1u;
        const char *port_env = getenv("BENCH_PORT_BASE");
        if (port_env != 0 && port_env[0] >= '0' && port_env[0] <= '9')
        {
            uint32_t candidate = (uint32_t)strtoul(port_env,0,10);
            if (candidate >= 20000u && candidate <= 64511u)
                BENCH_PORT_BASE = candidate;
        }
    }
    printf("BENCH-BUILD %s %s credits=%u d2a_max=%u\n", __DATE__, __TIME__,
        BENCH_CREDITS, BENCH_D2A_MAX);
    if (argc < 6)
    {
        printf("usage: rank degree iters rows mode(0 async 1 sync) [transport]\n");
        return 2;
    }
    rank = (uint32_t)strtoul(argv[1], 0, 10);
    degree = (uint32_t)strtoul(argv[2], 0, 10);
    iters = (uint32_t)strtoul(argv[3], 0, 10);
    rows = (uint32_t)strtoul(argv[4], 0, 10);
    mode = (uint32_t)strtoul(argv[5], 0, 10);
    transport_path = argc > 6 ? argv[6] :
        "/home/spark0/sparkdata/glm5_next.tp16/lib/hidden_transport.so";
    if ((degree != 4u && degree != 16u) || rank >= degree || iters == 0u || iters > BENCH_MAX_TIMED_ITERS || rows == 0u || rows > 1024u || mode > 1u)
    {
        printf("doorbell bench requires degree 4 or 16\n");
        return 2;
    }

    for (index = 0u; index < degree; index++)
    {
        snprintf(rail_direct[index], sizeof(rail_direct[0]), "10.10.200.%u", index);
        snprintf(rail_switch[index], sizeof(rail_switch[0]), "10.10.100.%u", 10u + index);
    }

    memset(&topology, 0, sizeof(topology));
    topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
    topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
    topology.rank_count = degree;
    {
        const char *algo = getenv("BENCH_ALGO");
        topology.algorithm_mask =
            SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE |
            SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING |
            (BENCH_D2A_MAX != 0u ?
                SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL : 0u);
        topology.direct_all_to_all_max_payload_bytes = BENCH_D2A_MAX;
        topology.split_ring_min_payload_bytes = 0u;
        (void)algo;
    }
    topology.rail_count = 2u;
    {
        topology.step_rail_indices[0] = 0u;
        for (step = 1u; step < SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS; step++)
            topology.step_rail_indices[step] = 1u;
    }
    for (index = 0u; index < degree; index++)
    {
        uint32_t peer;
        for (peer = 0u; peer < degree; peer++)
            if (peer != index)
                topology.session_ports[index][peer] =
                    (uint16_t)(BENCH_PORT_BASE + index * 16u + peer);
    }
    for (index = 0u; index < degree; index++)
    {
        snprintf(topology.rank_hosts[index], SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,
            "%s", rail_switch[index]);
        {
            const char *rail0_env = getenv("BENCH_RAIL0");
            snprintf(topology.rail_rank_hosts[0][index],
                SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES, "%s",
                rail0_env != 0 && rail0_env[0] == 'd' ?
                    rail_direct[index] : rail_switch[index]);
        }
        snprintf(topology.rail_rank_hosts[1][index],
            SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES, "%s", rail_switch[index]);
    }

    memset(&config, 0, sizeof(config));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
    config.tp_degree = degree;
    config.tp_rank = rank;
    config.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    config.credit_count = BENCH_CREDITS;
    config.local_hidden_dimension = BENCH_HIDDEN;
    config.max_active_sequence_count = rows;
    config.connect_timeout_milli = 60000u;
    config.operation_timeout_milli = 5000u;
    config.control_port_base = BENCH_PORT_BASE;
    config.collective_identifier = BENCH_IDENTIFIER;
    config.backend_module_path = transport_path;
    config.local_host = topology.rank_hosts[rank];
    status = SparkTpDeviceCollectiveApplyTopology(&topology, &config);
    if (status != SPARK_STATUS_OK)
    {
        printf("apply_topology -> %u\n", (unsigned)status);
        return 1;
    }
    config.registration_cuda_stream = 0;
    config.combine_bf16_function = bench_combine_bf16;
    config.combine_u64_max_function = bench_combine_u64_max;
    if (cudaMallocManaged(&bench_fold_context, sizeof(BenchFoldDevices)) != cudaSuccess)
    {
        printf("fold context alloc failed\n");
        return 1;
    }
    config.combine_tp4_bf16_function = bench_combine_all_bf16;
    config.combine_context = bench_fold_context;

    status = SparkTpDeviceCollectiveProbeMemoryMode(config.backend_kind,
        config.backend_module_path, &memory_mode);
    if (status != SPARK_STATUS_OK)
    {
        printf("probe_memory_mode -> %u\n", (unsigned)status);
        return 1;
    }
    status = SparkTpDeviceCollectiveCreditBindingRouteCount(&config, &route_count);
    if (status != SPARK_STATUS_OK)
    {
        printf("route_count -> %u\n", (unsigned)status);
        return 1;
    }
    printf("doorbell rank=%u memory_mode=%u routes=%u connect_ms=%u d2a_max=%u\n", rank, memory_mode, route_count, config.connect_timeout_milli, BENCH_D2A_MAX);

    credit_bytes = rows * BENCH_HIDDEN * 2u;
    total_bytes = route_count * BENCH_CREDITS * (credit_bytes + 8u);
    if (cudaHostAlloc(&host_send, total_bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc(&host_receive, total_bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
    {
        printf("host alloc failed\n");
        return 1;
    }
    if (cudaHostGetDevicePointer(&mapped_send, host_send, 0u) != cudaSuccess ||
        cudaHostGetDevicePointer(&mapped_receive, host_receive, 0u) != cudaSuccess)
    {
        printf("host map failed\n");
        return 1;
    }
    binding_count = 0u;
    offset = 0u;
    {
        uint32_t d2a_routes = BENCH_D2A_MAX != 0u ? degree - 1u : 0u;
        uint32_t tree_routes = route_count - d2a_routes;
        for (route = 0u; route < route_count; route++)
        {
            for (credit = 0u; credit < BENCH_CREDITS; credit++)
            {
                bindings[binding_count].step_index =
                    route < tree_routes ? route : route - tree_routes;
                bindings[binding_count].credit_index = credit;
                bindings[binding_count].send_device = (uint8_t *)mapped_send + offset;
                bindings[binding_count].receive_device = (uint8_t *)mapped_receive + offset;
                bindings[binding_count].send_transport = (uint8_t *)host_send + offset;
                bindings[binding_count].receive_transport = (uint8_t *)host_receive + offset;
                bindings[binding_count].flags =
                    SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS |
                    SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS |
                    (route < tree_routes ? 0u :
                        SPARK_TP_DEVICE_COLLECTIVE_BINDING_DIRECT_ALL_TO_ALL);
                bindings[binding_count].reserved0 = 0u;
                binding_count++;
                offset += credit_bytes + 8u;
            }
        }
    }
    config.credit_bindings = bindings;
    config.credit_binding_count = binding_count;

    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        printf("stream create failed\n");
        return 1;
    }
    config.registration_cuda_stream = stream;

    status = SparkTpDeviceCollectiveCreate(&config, &collective);
    if (status != SPARK_STATUS_OK)
    {
        printf("create -> %u\n", (unsigned)status);
        return 1;
    }
    printf("doorbell rank=%u collective_ready steps=%u\n", rank, collective.step_count);

    {
        uint32_t buffer_index;
        for (buffer_index = 0u; buffer_index < 64u; buffer_index++)
        {
            payload[buffer_index] = 0;
            void *host_alias = 0;
            void *device_alias = 0;
            if (cudaHostAlloc(&host_alias, (size_t)rows * BENCH_HIDDEN * 2u,
                    cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer(&device_alias, host_alias, 0u) != cudaSuccess)
            {
                printf("payload mapped alloc failed\n");
                return 1;
            }
            payload[buffer_index] = (__nv_bfloat16 *)device_alias;
        }
    }
    verify_host = (uint16_t *)malloc((size_t)rows * BENCH_HIDDEN * 2u);
    if (verify_host == 0)
        return 1;

    for (ordinal = 0u; ordinal < 68u; ordinal++)
    {
        uint64_t wait_started = bench_now_ns();
        bench_fill_bf16_kernel<<<(rows * BENCH_HIDDEN + 255u) / 256u, 256u, 0, stream>>>(
            payload[ordinal % 64u], (float)(rank + 1u + ((ordinal / 64u) % 4u)), rows * BENCH_HIDDEN);
        memset(&submission, 0, sizeof(submission));
        submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        submission.descriptor_bytes = sizeof(submission);
        submission.slot_index = (uint32_t)(ordinal % BENCH_CREDITS);
        submission.active_sequence_count = rows;
        submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
        submission.ordinal = bench_wire_ordinal(ordinal);
        submission.local_device = payload[ordinal % 64u];
        submission.full_device = payload[ordinal % 64u];
        submission.cuda_stream = stream;
        submission.completion_function = bench_completion;
        submission.completion_context = 0;
        {
            uint64_t retry_started = bench_now_ns();
            for (;;)
            {
                status = SparkTpDeviceCollectiveSubmitBf16(&collective, &submission);
                if (status == SPARK_STATUS_OK)
                    break;
                if (status != SPARK_STATUS_BUSY ||
                    bench_now_ns() - retry_started > 5000000000ull)
                {
                    printf("warmup submit %llu -> %u t=%.1fs\n", (unsigned long long)ordinal, (unsigned)status,(bench_now_ns()-retry_started)/1e9);
                    SparkTpDeviceCollectiveDumpOperations(&collective);
                    return 1;
                }
                usleep(50u);
            }
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            printf("warmup sync failed\n");
            return 1;
        }
        while (__sync_fetch_and_add(&bench_completions, 0) <=
            (int64_t)ordinal)
        {
            if (bench_now_ns() - wait_started > 5000000000ull)
            {
                printf("warmup completion timeout at %llu\n", (unsigned long long)ordinal);
                SparkTpDeviceCollectiveDumpOperations(&collective);
                return 1;
            }
            usleep(1000u);
        }
        if (__sync_fetch_and_add(&bench_last_status,0u) != SPARK_STATUS_OK)
        {
            fprintf(stderr,"warmup completion failed ordinal=%llu status=%u\n",(unsigned long long)ordinal,bench_last_status);
            SparkTpDeviceCollectiveDumpOperations(&collective);
            return 1;
        }
    }

    started_ns = bench_now_ns();
    for (ordinal = 68u; ordinal < 68u + iters; ordinal++)
    {
        bench_fill_bf16_kernel<<<(rows * BENCH_HIDDEN + 255u) / 256u, 256u, 0, stream>>>(
            payload[ordinal % 64u], (float)(rank + 1u + ((ordinal / 64u) % 4u)), rows * BENCH_HIDDEN);
        memset(&submission, 0, sizeof(submission));
        submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        submission.descriptor_bytes = sizeof(submission);
        submission.slot_index = (uint32_t)(ordinal % BENCH_CREDITS);
        submission.active_sequence_count = rows;
        submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
        submission.ordinal = bench_wire_ordinal(ordinal);
        submission.local_device = payload[ordinal % 64u];
        submission.full_device = payload[ordinal % 64u];
        submission.cuda_stream = stream;
        submission.completion_function = bench_completion;
        submission.completion_context = 0;
        retry = 0;
        {
            uint64_t op_started = bench_now_ns();
            for (;;)
            {
                status = SparkTpDeviceCollectiveSubmitBf16(&collective, &submission);
                if (status == SPARK_STATUS_OK)
                    break;
                if (status != SPARK_STATUS_BUSY || ++retry > 100)
                {
                    printf("submit %llu -> %u\n", (unsigned long long)ordinal, (unsigned)status);
                    SparkTpDeviceCollectiveDumpOperations(&collective);
                    return 1;
                }
                usleep(50u);
            }
            if (mode != 0u)
            {
                if (cudaStreamSynchronize(stream) != cudaSuccess)
                {
                    printf("sync failed at %llu\n", (unsigned long long)ordinal);
                    return 1;
                }
                while (__sync_fetch_and_add(&bench_completions, 0) <=
                    (int64_t)(ordinal - 20u + 20u))
                {
                    if (bench_now_ns() - started_ns > 60000000000ull)
                    {
                        printf("completion timeout at %llu\n", (unsigned long long)ordinal);
                        return 1;
                    }
                    usleep(20u);
                }
                if (ordinal - 68u < BENCH_MAX_TIMED_ITERS)
                    bench_latency_us[ordinal - 68u] =
                        (double)(bench_now_ns() - op_started) / 1000.0;
            }
        }
        if (BENCH_SKEW_US != 0u && rank == 8u)
            usleep(BENCH_SKEW_US);
    }
    if (mode == 0u)
    {
        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            printf("final sync failed\n");
            return 1;
        }
        while (__sync_fetch_and_add(&bench_completions, 0) < (int64_t)(68u + iters))
        {
            usleep(1000u);
            if (bench_now_ns() - started_ns > 120000000000ull)
            {
                printf("drain timeout %lld/%u\n",
                    (long long)__sync_fetch_and_add(&bench_completions, 0), iters);
                SparkTpDeviceCollectiveDumpOperations(&collective);
                return 1;
            }
        }
    }
    elapsed_us = (double)(bench_now_ns() - started_ns) / 1000.0;

    bad = 0u;
    for (uint32_t buffer = 0u; buffer < 64u; buffer++)
    {
        uint64_t last = 67u + iters;
        last -= (last + 64u - buffer) % 64u;
        float expected = (float)(degree * (degree + 1u) / 2u + degree * ((last / 64u) % 4u));
        uint32_t expected_bits;
        memcpy(&expected_bits,&expected,sizeof(expected_bits));
        if (cudaMemcpy(verify_host,payload[buffer],(size_t)rows * BENCH_HIDDEN * 2u,
                cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            printf("verify copy failed buffer=%u\n",buffer);
            return 1;
        }
        for (index = 0u; index < rows * BENCH_HIDDEN; index++)
            if ((uint32_t)verify_host[index] != (expected_bits >> 16u))
            {
                bad++;
                if (bad < 4u)
                    printf("buffer%u el%u=%u expected=%u ",buffer,index,
                        (unsigned)verify_host[index],expected_bits >> 16u);
            }
    }
    printf("doorbell rank=%u rows=%u mode=%u iters=%u per_op_us=%.1f completions=%lld status=%u %s\n",
        rank, rows, mode, iters, elapsed_us / (double)iters,
        (long long)__sync_fetch_and_add(&bench_completions, 0),
        bench_last_status, bad != 0u ? "CORRUPT" : "OK");
    if (mode != 0u && iters <= BENCH_MAX_TIMED_ITERS)
    {
        qsort(bench_latency_us, iters, sizeof(bench_latency_us[0]), bench_compare_double);
        printf("latency rank=%u p50=%.1f p95=%.1f p99=%.1f min=%.1f max=%.1f us\n",
            rank,
            bench_latency_us[iters / 2u],
            bench_latency_us[(iters * 95u) / 100u],
            bench_latency_us[(iters * 99u) / 100u],
            bench_latency_us[0],
            bench_latency_us[iters - 1u]);
    }
    return bad != 0u || bench_last_status != SPARK_STATUS_OK ? 3 : 0;
}
