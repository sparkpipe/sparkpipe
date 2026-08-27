/* Fault-injection coverage for the Qwen 27B TP credit-buffer ownership
 * (audit P1: mapped-host initialization conflated allocation ownership).
 * Every scenario drives SparkQwen38_27bTpAllocateCreditMemory to a terminal
 * state and then asserts, via the stub's allocation ledger, that
 * SparkQwen38_27bTpDestroy releases every live allocation - including the
 * partial states a failed second cudaHostAlloc or a failed
 * cudaHostGetDevicePointer used to leak. */
#include <stdio.h>
#include <string.h>

#include "cuda_runtime_api.h"
#include "sparkpipe/spark_status.h"
#include "spark_qwen38_27b_tp.h"

/* GPU accumulate kernels referenced by tp.c's combine callbacks. They are
 * never executed by this test (no collective runs); definitions exist only
 * to satisfy the host link. */
cudaError_t SparkQwen38_27bLaunchAccumAdd(cudaStream_t stream, void *destination, const void *source, uint32_t active_sequence_count, uint32_t hidden_dimension)
{
    (void)stream; (void)destination; (void)source;
    (void)active_sequence_count; (void)hidden_dimension;
    return cudaErrorMemoryAllocation;
}
cudaError_t SparkQwen38_27bLaunchAccumAddRelay(cudaStream_t stream, void *destination, const void *source, void *relay, uint32_t active_sequence_count, uint32_t hidden_dimension)
{
    (void)stream; (void)destination; (void)source; (void)relay;
    (void)active_sequence_count; (void)hidden_dimension;
    return cudaErrorMemoryAllocation;
}
cudaError_t SparkQwen38_27bLaunchAccumAddTp4(cudaStream_t stream, void *destination, const void *const rank_devices[4], uint32_t tp_rank, uint32_t active_sequence_count, uint32_t hidden_dimension)
{
    (void)stream; (void)destination; (void)rank_devices; (void)tp_rank;
    (void)active_sequence_count; (void)hidden_dimension;
    return cudaErrorMemoryAllocation;
}
cudaError_t SparkQwen38_27bLaunchAccumU64Max(cudaStream_t stream, void *destination, const void *source, uint32_t active_sequence_count)
{
    (void)stream; (void)destination; (void)source;
    (void)active_sequence_count;
    return cudaErrorMemoryAllocation;
}

static int failures = 0;

static void check(int condition, const char *what)
{
    if (condition)
        return;
    fprintf(stderr, "FAIL %s\n", what);
    failures++;
}

static void run_scenario(const char *name, uint32_t mapped_host,
    void (*inject)(void), SparkStatus expect_status, int expect_device_owned)
{
    SparkQwen38_27bTpState tp;
    SparkStatus status;
    memset(&tp, 0, sizeof(tp));
    spark_stub_cuda_reset_faults();
    if (inject != 0)
        inject();
    status = SparkQwen38_27bTpAllocateCreditMemory(&tp, 655360u, mapped_host);
    check(status == expect_status, name);
    if (status == SPARK_STATUS_OK)
        check(tp.credit_device_allocated == (uint32_t)expect_device_owned, name);
    SparkQwen38_27bTpDestroy(&tp);
    check(spark_stub_cuda_outstanding_allocs() == 0u, name);
}

static void fail_first_malloc(void)  { spark_stub_cuda_fail_alloc_call(1); }
static void fail_second_malloc(void) { spark_stub_cuda_fail_alloc_call(2); }
static void fail_first_host(void)    { spark_stub_cuda_fail_alloc_call(3); }
static void fail_second_host(void)   { spark_stub_cuda_fail_alloc_call(4); }
static void fail_first_map(void)     { spark_stub_cuda_fail_host_map_call(1); }
static void fail_second_map(void)    { spark_stub_cuda_fail_host_map_call(2); }

int main(void)
{
    /* device mode: two cudaMallocs owned by the device fields */
    run_scenario("device_happy", 0u, 0, SPARK_STATUS_OK, 1);
    /* mapped-host mode: allocs + maps succeed; device originals are freed
     * and the device fields alias the host buffers (flag cleared) */
    run_scenario("mapped_happy", 1u, 0, SPARK_STATUS_OK, 0);
    /* first cudaMalloc fails: nothing survives */
    run_scenario("fail_first_malloc", 1u, fail_first_malloc,
        SPARK_STATUS_CAPACITY_EXCEEDED, 0);
    /* second cudaMalloc fails: the first must not leak */
    run_scenario("fail_second_malloc", 1u, fail_second_malloc,
        SPARK_STATUS_CAPACITY_EXCEEDED, 0);
    /* first cudaHostAlloc fails: both device allocations must be freed */
    run_scenario("fail_first_host_alloc", 1u, fail_first_host,
        SPARK_STATUS_CAPACITY_EXCEEDED, 0);
    /* the audited defect: second cudaHostAlloc failed while the first
     * stayed populated; nothing may leak now */
    run_scenario("fail_second_host_alloc", 1u, fail_second_host,
        SPARK_STATUS_CAPACITY_EXCEEDED, 0);
    /* mapping failures are fatal and must not leak the host+device set */
    run_scenario("fail_first_host_map", 1u, fail_first_map,
        SPARK_STATUS_CAPACITY_EXCEEDED, 0);
    run_scenario("fail_second_host_map", 1u, fail_second_map,
        SPARK_STATUS_CAPACITY_EXCEEDED, 0);

    if (failures != 0)
    {
        fprintf(stderr, "FAIL qwen38_27b tp fault injection (%d)\n", failures);
        return 1;
    }
    printf("PASS qwen38_27b tp fault injection: 8 scenarios, zero outstanding "
        "allocations after every destroy\n");
    return 0;
}
