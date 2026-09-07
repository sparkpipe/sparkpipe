#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <atomic>
#include "sparkpipe/spark_tp_device_collective.h"

typedef struct SlotBenchRank
{
    SparkTpDeviceCollective collective;
    void *local_device;
    void *full_device;
    uint64_t ordinal;
    std::atomic<unsigned> done;
} SlotBenchRank;

static uint64_t NowNs(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t) != 0)
        return 0u;
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

__global__ void SlotBenchProducerKernel(
    unsigned long long nanoseconds,
    unsigned long long *sink)
{
    unsigned long long start = clock64();
    unsigned long long spins = nanoseconds;
    unsigned long long accumulator = (unsigned long long)threadIdx.x;
    while ((unsigned long long)(clock64() - start) < spins)
        accumulator += 1u;
    if (sink != 0 && accumulator == UINT64_MAX)
        *sink = accumulator;
}

__global__ void SlotBenchFoldKernel(
    unsigned short *destination,
    const unsigned short *source,
    uint32_t elements)
{
    extern __shared__ unsigned short scratch[];
    uint32_t index;
    for (index = threadIdx.x; index < elements; index += blockDim.x)
        scratch[index] = destination[index];
    __syncthreads();
    for (index = threadIdx.x; index < elements; index += blockDim.x)
        destination[index] =
            (unsigned short)(((unsigned)scratch[index] +
                (unsigned)source[index]) >> 1u);
}

__global__ void SlotBenchWaitKernel(
    volatile unsigned long long *flag,
    unsigned long long value)
{
    while (*flag < value)
        __nanosleep(100u);
}

static SparkStatus SlotBenchCombineBf16(
    void *context,
    void *destination_device,
    const void *source_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream)
{
    uint32_t row;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    for (row = 0u; row < active_sequence_count; ++row)
        SlotBenchFoldKernel<<<1,256,(size_t)hidden_dimension * 2u,stream>>>(
            (unsigned short *)destination_device +
                (size_t)row * hidden_dimension,
            (const unsigned short *)source_device +
                (size_t)row * hidden_dimension,
            hidden_dimension);
    (void)context;
    return cudaPeekAtLastError() == cudaSuccess ?
        SPARK_STATUS_OK : SPARK_STATUS_DRIVER_LOAD_ERROR;
}

static void SlotBenchCompletion(
    void *context,
    const SparkTpDeviceCollectiveCompletion *completion)
{
    SlotBenchRank *rank = (SlotBenchRank *)context;
    if (completion != 0 && completion->status == SPARK_STATUS_OK)
        rank->done.store(1u,std::memory_order_release);
    else if (completion != 0)
        rank->done.store(2u,std::memory_order_release);
}

int main(int argc,char **argv)
{
    uint32_t tp_degree = argc > 1u ? (uint32_t)strtoul(argv[1],0,10) : 16u;
    uint32_t ops = argc > 2u ? (uint32_t)strtoul(argv[2],0,10) : 180u;
    double producer_us = argc > 3u ? strtod(argv[3],0) : 0.0;
    uint32_t hidden = 4096u;
    uint32_t rows = 1u;
    const char *dso = getenv("SLOT_BENCH_DSO");
    if (dso == 0)
        dso = "lib/hidden_transport.so";
    SlotBenchRank ranks[16u];
    pthread_barrier_t barrier;
    uint32_t rank_index;
    uint32_t op_index;
    uint64_t *submit_ns;
    uint64_t *done_ns;
    cudaStream_t compute_stream;
    double clock_mhz = 1000.0;

    if (tp_degree > 16u)
    {
        fprintf(stderr,"tp degree capped at 16\n");
        return 1;
    }
    memset(ranks,0,sizeof(ranks));
    submit_ns = (uint64_t *)calloc(ops,sizeof(uint64_t));
    done_ns = (uint64_t *)calloc(ops,sizeof(uint64_t));
    if (submit_ns == 0 || done_ns == 0)
        return 1;
    cudaStreamCreate(&compute_stream);
    pthread_barrier_init(&barrier,0,(unsigned)tp_degree);
    for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
    {
        SparkTpDeviceCollectiveConfig config;
        SparkStatus status;
        memset(&config,0,sizeof(config));
        config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        config.backend_kind =
            SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
        config.tp_degree = tp_degree;
        config.tp_rank = rank_index;
        config.operation_kind =
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
        config.credit_count = 8u;
        config.local_hidden_dimension = hidden;
        config.max_active_sequence_count = rows;
        config.connect_timeout_milli = 60000u;
        config.operation_timeout_milli = 30000u;
        config.control_port_base = 21000u + rank_index;
        config.backend_module_path = dso;
        config.local_host = "bench";
        config.registration_cuda_stream = compute_stream;
        config.combine_bf16_function = SlotBenchCombineBf16;
        config.combine_context = 0;
        cudaMalloc(&ranks[rank_index].local_device,
            (size_t)hidden * rows * 2u + 64u);
        cudaMalloc(&ranks[rank_index].full_device,
            (size_t)hidden * rows * 2u + 64u);
        cudaMemset(ranks[rank_index].full_device,0,
            (size_t)hidden * rows * 2u);
        status = SparkTpDeviceCollectiveCreate(&config,
            &ranks[rank_index].collective);
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr,"rank %u create failed %u\n",rank_index,
                (uint32_t)status);
            return 1;
        }
        ranks[rank_index].ordinal = 0u;
        ranks[rank_index].done.store(0u,std::memory_order_relaxed);
    }
    fprintf(stderr,"mesh up: %u ranks, %u ops, producer %.1f us\n",
        tp_degree,ops,producer_us);
    for (op_index = 0u; op_index < ops; ++op_index)
    {
        uint32_t slowest = 0u;
        pthread_barrier_wait(&barrier);
        for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
        {
            SparkTpDeviceCollectiveSubmission submission;
            void *flag_device;
            uint64_t wait_value;
            memset(&submission,0,sizeof(submission));
            submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
            submission.descriptor_bytes = sizeof(submission);
            submission.slot_index = 0u;
            submission.active_sequence_count = rows;
            submission.ordinal = ranks[rank_index].ordinal;
            submission.local_device = ranks[rank_index].local_device;
            submission.full_device = ranks[rank_index].full_device;
            submission.cuda_stream = compute_stream;
            submission.completion_function = SlotBenchCompletion;
            submission.completion_context = &ranks[rank_index];
            if (producer_us > 0.0)
                SlotBenchProducerKernel<<<1,32,0,compute_stream>>>(
                    (unsigned long long)(producer_us * clock_mhz),0);
            ranks[rank_index].done.store(0u,std::memory_order_release);
            if (SparkTpDeviceCollectiveSubmitBf16(
                    &ranks[rank_index].collective,&submission) !=
                    SPARK_STATUS_OK)
            {
                fprintf(stderr,"rank %u submit failed op %u\n",rank_index,
                    op_index);
                return 1;
            }
            if (SparkTpDeviceCollectiveOpWaitHandles(
                    &ranks[rank_index].collective,
                    ranks[rank_index].ordinal,&flag_device,
                    &wait_value) == SPARK_STATUS_OK)
                SlotBenchWaitKernel<<<1,1,0,compute_stream>>>(
                    (volatile unsigned long long *)flag_device,
                    (unsigned long long)wait_value);
            if (rank_index == 0u)
                submit_ns[op_index] = NowNs();
            ranks[rank_index].ordinal += 1u;
        }
        for (rank_index = 0u; rank_index < tp_degree; ++rank_index)
        {
            uint64_t waited = 0u;
            while (ranks[rank_index].done.load(std::memory_order_acquire) ==
                    0u)
            {
                if (++waited > 1200000000000ull)
                    break;
            }
            if (ranks[rank_index].done.load(std::memory_order_acquire) !=
                    1u)
            {
                fprintf(stderr,"rank %u op %u failed/timed out\n",
                    rank_index,op_index);
                return 1;
            }
        }
        cudaStreamSynchronize(compute_stream);
        done_ns[op_index] = NowNs();
        (void)slowest;
    }
    {
        double total = (double)(done_ns[ops - 1u] - submit_ns[0u]) /
            1000000000.0;
        double per_op_ms = total / (double)ops * 1000.0;
        uint32_t warm = ops / 10u;
        double warm_total = (double)(done_ns[ops - 1u] - submit_ns[warm]) /
            1000000000.0;
        double warm_per_op_ms = warm_total / (double)(ops - warm) * 1000.0;
        double token_s = warm_per_op_ms * 91u / 1000.0;
        printf("SLOT-BENCH ranks=%u ops=%u producer_us=%.1f total_s=%.3f "
            "per_op_ms=%.3f warm_per_op_ms=%.3f token_ms=%.1f tok_s=%.3f\n",
            tp_degree,ops,producer_us,total,per_op_ms,warm_per_op_ms,
            token_s * 1000.0,token_s > 0.0 ? 1.0 / token_s : 0.0);
    }
    return 0;
}
