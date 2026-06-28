#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_sparse_map_kernels.h"

#define SPARK_CUDA_SPARSE_MAP_THREADS 256u

static uint64_t SparkCudaSparseMapValueCountHost(const SparkCudaSparseMapRequest *request)
{
    return (uint64_t)request->query_count * (uint64_t)request->sparse_top_k;
}

static uint32_t SparkCudaSparseMapBlockCount(uint64_t value_count)
{
    uint64_t block_count;

    block_count = (value_count + (uint64_t)SPARK_CUDA_SPARSE_MAP_THREADS - 1u) / (uint64_t)SPARK_CUDA_SPARSE_MAP_THREADS;
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

static void SparkCudaSparseMapFillReportShape(const SparkCudaSparseMapRequest *request, SparkCudaSparseMapReport *report)
{
    report->map_value_count = SparkCudaSparseMapValueCountHost(request);
    report->error_counter_count = SPARKPIPE_CUDA_SPARSE_MAP_ERROR_COUNTERS;
}

static __global__ void SparkCudaSparseMapClearKernel(SparkCudaSparseMapRequest request, uint32_t *error_counters)
{
    if (blockIdx.x != 0u || threadIdx.x >= SPARKPIPE_CUDA_SPARSE_MAP_ERROR_COUNTERS)
    {
        return;
    }
    error_counters[threadIdx.x] = 0u;
    if (threadIdx.x == 0u && request.sentinel != SPARKPIPE_CUDA_SPARSE_MAP_SENTINEL)
    {
        error_counters[SPARK_CUDA_SPARSE_MAP_ERROR_SENTINEL] = 1u;
    }
}

static __global__ void SparkCudaSparseMapKernel(SparkCudaSparseMapRequest request, const uint32_t *logical_block_ids, const uint32_t *physical_block_map, uint32_t *physical_block_ids, uint32_t *error_counters)
{
    uint64_t map_index;
    uint64_t map_count;
    uint32_t logical_block_id;
    uint32_t physical_block_id;

    map_count = (uint64_t)request.query_count * (uint64_t)request.sparse_top_k;
    map_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (map_index < map_count)
    {
        logical_block_id = logical_block_ids[map_index];
        if (logical_block_id >= request.logical_block_count)
        {
            physical_block_ids[map_index] = SPARKPIPE_CUDA_SPARSE_MAP_INVALID;
            atomicAdd(&error_counters[SPARK_CUDA_SPARSE_MAP_ERROR_INVALID_LOGICAL], 1u);
        }
        else
        {
            physical_block_id = physical_block_map[logical_block_id];
            if (physical_block_id >= request.physical_block_count)
            {
                physical_block_ids[map_index] = SPARKPIPE_CUDA_SPARSE_MAP_INVALID;
                atomicAdd(&error_counters[SPARK_CUDA_SPARSE_MAP_ERROR_INVALID_PHYSICAL], 1u);
            }
            else
            {
                physical_block_ids[map_index] = physical_block_id;
            }
        }
        map_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

extern "C" SparkStatus SparkRunCudaSparsePhysicalBlockMap(const SparkCudaSparseMapRequest *request, const uint32_t *device_logical_block_ids, const uint32_t *device_physical_block_map, uint32_t *device_physical_block_ids, uint32_t *device_error_counters, SparkCudaSparseMapReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t map_value_count;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaSparseMapRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_logical_block_ids == 0 || device_physical_block_map == 0 || device_physical_block_ids == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaSparseMapFillReportShape(request, report);
    SparkCudaSparseMapClearKernel<<<1u, SPARKPIPE_CUDA_SPARSE_MAP_ERROR_COUNTERS>>>(*request, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    map_value_count = report->map_value_count;
    SparkCudaSparseMapKernel<<<SparkCudaSparseMapBlockCount(map_value_count), SPARK_CUDA_SPARSE_MAP_THREADS>>>(*request, device_logical_block_ids, device_physical_block_map, device_physical_block_ids, device_error_counters);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->clear_kernel_count = 1u;
    report->map_kernel_count = 1u;
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}
