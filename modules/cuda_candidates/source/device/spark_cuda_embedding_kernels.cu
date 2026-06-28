#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_embedding_kernels.h"

#define SPARK_CUDA_EMBEDDING_THREADS 256u

static uint64_t SparkCudaEmbeddingOutputValueCountHost(const SparkCudaEmbeddingRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->hidden_size;
}

static uint64_t SparkCudaEmbeddingTableValueCountHost(const SparkCudaEmbeddingRequest *request)
{
    return (uint64_t)request->vocab_size * (uint64_t)request->hidden_size;
}

static uint32_t SparkCudaEmbeddingBlockCount(uint64_t value_count)
{
    uint64_t block_count;

    block_count = (value_count + (uint64_t)SPARK_CUDA_EMBEDDING_THREADS - 1u) / (uint64_t)SPARK_CUDA_EMBEDDING_THREADS;
    if (block_count == 0u)
    {
        return 1u;
    }
    if (block_count > 65535u)
    {
        return 65535u;
    }
    return (uint32_t)block_count;
}

static void SparkCudaEmbeddingFillReportShape(const SparkCudaEmbeddingRequest *request, SparkCudaEmbeddingReport *report)
{
    report->output_value_count = SparkCudaEmbeddingOutputValueCountHost(request);
    report->table_value_count = SparkCudaEmbeddingTableValueCountHost(request);
    report->error_counter_count = SPARKPIPE_CUDA_EMBEDDING_ERROR_COUNTERS;
}

static __global__ void SparkCudaEmbeddingClearKernel(SparkCudaEmbeddingRequest request, uint32_t *error_counters)
{
    if (blockIdx.x != 0u || threadIdx.x >= SPARKPIPE_CUDA_EMBEDDING_ERROR_COUNTERS)
    {
        return;
    }
    error_counters[threadIdx.x] = 0u;
    if (threadIdx.x == 0u && request.sentinel != SPARKPIPE_CUDA_EMBEDDING_SENTINEL)
    {
        error_counters[SPARK_CUDA_EMBEDDING_ERROR_SENTINEL] = 1u;
    }
}

static __global__ void SparkCudaEmbeddingGatherBf16Kernel(SparkCudaEmbeddingRequest request, const uint32_t *token_ids, const uint16_t *embedding_table, uint16_t *output, uint32_t *error_counters)
{
    uint64_t value_index;
    uint64_t output_value_count;
    uint64_t token_index;
    uint32_t hidden_index;
    uint32_t token_id;

    output_value_count = (uint64_t)request.token_count * (uint64_t)request.hidden_size;
    value_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (value_index < output_value_count)
    {
        token_index = value_index / (uint64_t)request.hidden_size;
        hidden_index = (uint32_t)(value_index - (token_index * (uint64_t)request.hidden_size));
        token_id = token_ids[token_index];
        if (token_id < request.vocab_size)
        {
            output[value_index] = embedding_table[((uint64_t)token_id * (uint64_t)request.hidden_size) + hidden_index];
        }
        else
        {
            output[value_index] = 0u;
            if (hidden_index == 0u)
            {
                atomicAdd(&error_counters[SPARK_CUDA_EMBEDDING_ERROR_INVALID_TOKEN], 1u);
            }
        }
        value_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

extern "C" SparkStatus SparkRunCudaEmbeddingGatherBf16(const SparkCudaEmbeddingRequest *request, const uint32_t *device_token_ids, const void *device_embedding_table_bf16, void *device_output_bf16, uint32_t *device_error_counters, SparkCudaEmbeddingReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t output_value_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaEmbeddingRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_token_ids == 0 || device_embedding_table_bf16 == 0 || device_output_bf16 == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaEmbeddingFillReportShape(request, report);
    SparkCudaEmbeddingClearKernel<<<1u, SPARKPIPE_CUDA_EMBEDDING_ERROR_COUNTERS>>>(*request, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    output_value_count = report->output_value_count;
    SparkCudaEmbeddingGatherBf16Kernel<<<SparkCudaEmbeddingBlockCount(output_value_count), SPARK_CUDA_EMBEDDING_THREADS>>>(*request, device_token_ids, (const uint16_t *)device_embedding_table_bf16, (uint16_t *)device_output_bf16, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 1u;
    report->gather_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}
