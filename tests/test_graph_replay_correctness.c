#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"

#define TEST_DEGREE 4
#define TEST_ROWS 1
#define TEST_HIDDEN 512
#define TEST_ROUNDS 8
#define TEST_PAYLOAD_BYTES (TEST_ROWS * TEST_HIDDEN * 2)

static int test_failures;

#define CHECK(cond, name) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, name); \
        test_failures++; \
    } \
} while (0)

static void fill_pattern(uint16_t *data, uint32_t count, uint32_t seed)
{
    uint32_t i;
    for (i = 0; i < count; i++)
        data[i] = (uint16_t)(seed * 31 + i * 7 + (i >> 3));
}

static int compare_bf16_sums(const uint16_t *a, const uint16_t *b,
    uint32_t count, const char *label)
{
    uint32_t i;
    int max_diff = 0;
    for (i = 0; i < count; i++) {
        int32_t va = (int16_t)a[i];
        int32_t vb = (int16_t)b[i];
        int diff = va > vb ? va - vb : vb - va;
        if (diff > max_diff)
            max_diff = diff;
    }
    if (max_diff > 2) {
        fprintf(stderr, "FAIL %s: max bf16 diff %d at count=%u\n",
            label, max_diff, count);
        test_failures++;
        return 0;
    }
    return 1;
}

static SparkStatus completion_status;
static uint32_t completion_count;

static void test_completion(void *context, SparkStatus status)
{
    (void)context;
    completion_status = status;
    completion_count++;
}

static int test_graph_capture_and_replay(
    SparkTpDeviceCollective *collective,
    void *device_scratch,
    void *host_reference,
    uint32_t rank,
    const char *socket_path)
{
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    uint16_t *host_result;
    uint16_t *host_eager;
    uint32_t round;
    int status;

    cudaStreamCreate(&stream);
    host_result = calloc(TEST_PAYLOAD_BYTES, 1);
    host_eager = calloc(TEST_PAYLOAD_BYTES, 1);

    fill_pattern((uint16_t *)host_reference,
        TEST_PAYLOAD_BYTES / 2, rank + 1);
    cudaMemcpy(device_scratch, host_reference, TEST_PAYLOAD_BYTES,
        cudaMemcpyHostToDevice);

    for (round = 0; round < TEST_ROUNDS; round++) {
        SparkTpDeviceCollectiveSubmission sub;
        memset(&sub, 0, sizeof(sub));
        sub.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        sub.descriptor_bytes = sizeof(sub);
        sub.slot_index = 0;
        sub.active_sequence_count = TEST_ROWS;
        sub.ordinal = round;
        sub.local_device = device_scratch;
        sub.full_device = device_scratch;
        sub.cuda_stream = stream;
        sub.completion_function = test_completion;
        sub.completion_context = 0;

        status = SparkTpDeviceCollectiveEnqueue(collective, &sub,
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
        CHECK(status == SPARK_STATUS_OK, "eager enqueue");
        if (status != SPARK_STATUS_OK)
            goto cleanup;

        cudaStreamSynchronize(stream);
        cudaMemcpy(host_eager, device_scratch, TEST_PAYLOAD_BYTES,
            cudaMemcpyDeviceToHost);
    }

    cudaStreamDestroy(stream);
    cudaStreamCreate(&stream);

    if (SparkTpDeviceCollectiveArmCapture(collective) != SPARK_STATUS_OK) {
        fprintf(stderr, "FAIL: arm capture rank=%u\n", rank);
        test_failures++;
        goto cleanup;
    }

    for (round = 0; round < TEST_ROUNDS; round++) {
        SparkTpDeviceCollectiveSubmission sub;
        memset(&sub, 0, sizeof(sub));
        sub.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        sub.descriptor_bytes = sizeof(sub);
        sub.slot_index = 0;
        sub.active_sequence_count = TEST_ROWS;
        sub.ordinal = round;
        sub.local_device = device_scratch;
        sub.full_device = device_scratch;
        sub.cuda_stream = stream;
        sub.completion_function = 0;
        sub.completion_context = 0;

        status = SparkTpDeviceCollectiveEnqueue(collective, &sub,
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
        CHECK(status == SPARK_STATUS_OK, "armed enqueue");
    }

    if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) {
        fprintf(stderr, "FAIL: end capture rank=%u: %s\n", rank,
            cudaGetErrorString(cudaGetLastError()));
        test_failures++;
        SparkTpDeviceCollectiveDisarmCapture(collective);
        goto cleanup;
    }

    if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        fprintf(stderr, "FAIL: instantiate rank=%u: %s\n", rank,
            cudaGetErrorString(cudaGetLastError()));
        test_failures++;
        goto cleanup;
    }

    SparkTpDeviceCollectiveDisarmCapture(collective);

    uint32_t replay;
    for (replay = 0; replay < 3; replay++) {
        cudaError_t err;
        uint64_t watchdog_stop;
        struct timespec now;
        cudaEvent_t start_ev, stop_ev;
        float elapsed_ms;

        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);
        cudaEventRecord(start_ev, stream);

        if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
            fprintf(stderr, "FAIL: graph launch rank=%u replay=%u: %s\n",
                rank, replay, cudaGetErrorString(cudaGetLastError()));
            test_failures++;
            goto cleanup;
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
        watchdog_stop = (uint64_t)now.tv_sec * 1000000000ull +
            (uint64_t)now.tv_nsec + 15000000000ull;
        for (;;) {
            err = cudaStreamQuery(stream);
            if (err == cudaSuccess)
                break;
            if (err != cudaErrorNotReady) {
                fprintf(stderr, "FAIL: replay error rank=%u replay=%u: %s\n",
                    rank, replay, cudaGetErrorString(err));
                test_failures++;
                goto cleanup;
            }
            clock_gettime(CLOCK_MONOTONIC, &now);
            if ((uint64_t)now.tv_sec * 1000000000ull +
                (uint64_t)now.tv_nsec >= watchdog_stop) {
                fprintf(stderr, "FAIL: replay hung rank=%u replay=%u (15s)\n",
                    rank, replay);
                test_failures++;
                goto cleanup;
            }
            usleep(500);
        }

        cudaEventRecord(stop_ev, stream);
        cudaEventSynchronize(stop_ev);
        cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);

        if (replay > 0)
            fprintf(stderr,
                "GRAPH-REPLAY rank=%u replay=%u elapsed_ms=%.3f\n",
                rank, replay, elapsed_ms);

        uint64_t graph_error = SparkTpDeviceCollectiveGraphError(collective);
        SparkTpDeviceCollectiveClearGraphError(collective);
        CHECK(graph_error == 0, "graph error word clear after replay");

        cudaMemcpy(host_result, device_scratch, TEST_PAYLOAD_BYTES,
            cudaMemcpyDeviceToHost);
        compare_bf16_sums(host_result, host_eager,
            TEST_PAYLOAD_BYTES / 2, "graph vs eager");
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);

cleanup:
    free(host_result);
    free(host_eager);
    cudaStreamDestroy(stream);
    return test_failures;
}

int main(int argc, char **argv)
{
    const char *socket_path;
    uint32_t rank;
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfiguration config;
    void *device_scratch;
    uint16_t *host_reference;
    int status;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <socket_path> <rank>\n", argv[0]);
        return 2;
    }
    socket_path = argv[1];
    rank = (uint32_t)strtoul(argv[2], 0, 10);

    memset(&config, 0, sizeof(config));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.descriptor_bytes = sizeof(config);
    config.tp_rank = rank;
    config.tp_degree = TEST_DEGREE;
    config.local_hidden_dimension = TEST_HIDDEN;
    config.collective_identifier = 0;
    config.connect_timeout_milli = 30000;
    config.operation_timeout_milli = 10000;
    config.max_active_sequence_count = 1;

    status = SparkTpDeviceCollectiveCreate(&config, &collective);
    CHECK(status == SPARK_STATUS_OK, "collective create");
    if (status != SPARK_STATUS_OK)
        return 1;

    cudaMalloc(&device_scratch, TEST_PAYLOAD_BYTES);
    host_reference = calloc(TEST_PAYLOAD_BYTES, 1);

    fill_pattern(host_reference, TEST_PAYLOAD_BYTES / 2, rank + 100);
    cudaMemcpy(device_scratch, host_reference, TEST_PAYLOAD_BYTES,
        cudaMemcpyHostToDevice);

    SparkTpDeviceCollectivePrepareReceiveBf16(&collective,
        device_scratch, TEST_ROWS, TEST_HIDDEN, 0, 0);

    test_failures = test_graph_capture_and_replay(
        &collective, device_scratch, host_reference, rank, socket_path);

    SparkTpDeviceCollectiveDestroy(&collective);
    cudaFree(device_scratch);
    free(host_reference);

    fprintf(stderr, "%s: %d failures\n", argv[0], test_failures);
    return test_failures ? 1 : 0;
}
