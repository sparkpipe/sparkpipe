#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_SPARSE_MAP_SENTINEL 0x535053504D415032ull
#define SPARKPIPE_CUDA_SPARSE_MAP_ERROR_COUNTERS 3u
#define SPARKPIPE_CUDA_SPARSE_MAP_INVALID UINT32_MAX

typedef enum SparkCudaSparseMapErrorCounter
{
    SPARK_CUDA_SPARSE_MAP_ERROR_INVALID_LOGICAL = 0,
    SPARK_CUDA_SPARSE_MAP_ERROR_INVALID_PHYSICAL = 1,
    SPARK_CUDA_SPARSE_MAP_ERROR_SENTINEL = 2
} SparkCudaSparseMapErrorCounter;

typedef struct SparkCudaSparseMapRequest
{
    uint32_t query_count;
    uint32_t sparse_top_k;
    uint32_t logical_block_count;
    uint32_t physical_block_count;
    uint64_t sentinel;
} SparkCudaSparseMapRequest;

typedef struct SparkCudaSparseMapReport
{
    uint64_t map_value_count;
    uint32_t clear_kernel_count;
    uint32_t map_kernel_count;
    uint32_t hot_path_allocation_count;
    uint32_t error_counter_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
} SparkCudaSparseMapReport;

SparkStatus SparkValidateCudaSparseMapRequest(const SparkCudaSparseMapRequest *request);
SparkStatus SparkRunCudaSparsePhysicalBlockMap(const SparkCudaSparseMapRequest *request, const uint32_t *device_logical_block_ids, const uint32_t *device_physical_block_map, uint32_t *device_physical_block_ids, uint32_t *device_error_counters, SparkCudaSparseMapReport *report);

#ifdef __cplusplus
}
#endif
