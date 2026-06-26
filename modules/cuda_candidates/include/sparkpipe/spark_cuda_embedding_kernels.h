#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_EMBEDDING_SENTINEL 0x5350454D424B3031ull
#define SPARKPIPE_CUDA_EMBEDDING_ERROR_COUNTERS 2u

typedef enum SparkCudaEmbeddingErrorCounter
{
    SPARK_CUDA_EMBEDDING_ERROR_INVALID_TOKEN = 0,
    SPARK_CUDA_EMBEDDING_ERROR_SENTINEL = 1
} SparkCudaEmbeddingErrorCounter;

typedef struct SparkCudaEmbeddingRequest
{
    uint32_t token_count;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t reserved;
    uint64_t sentinel;
} SparkCudaEmbeddingRequest;

typedef struct SparkCudaEmbeddingReport
{
    uint64_t output_value_count;
    uint64_t table_value_count;
    uint32_t clear_kernel_count;
    uint32_t gather_kernel_count;
    uint32_t hot_path_allocation_count;
    uint32_t error_counter_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
} SparkCudaEmbeddingReport;

SparkStatus SparkValidateCudaEmbeddingRequest(const SparkCudaEmbeddingRequest *request);
SparkStatus SparkRunCudaEmbeddingGatherBf16(const SparkCudaEmbeddingRequest *request, const uint32_t *device_token_ids, const void *device_embedding_table_bf16, void *device_output_bf16, uint32_t *device_error_counters, SparkCudaEmbeddingReport *report);

#ifdef __cplusplus
}
#endif
