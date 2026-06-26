#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_SPARSE_INDEX_SENTINEL 0x5350535049445831ull
#define SPARKPIPE_CUDA_SPARSE_INDEX_MAX_TOP_K 64u
#define SPARKPIPE_CUDA_SPARSE_INDEX_INVALID UINT32_MAX

typedef struct SparkCudaSparseIndexRequest
{
	uint32_t row_count;
	uint32_t candidate_count;
	uint32_t top_k;
	uint32_t score_stride;
	uint32_t reserved;
	uint64_t sentinel;
} SparkCudaSparseIndexRequest;

typedef struct SparkCudaSparseIndexReport
{
	uint64_t score_count;
	uint64_t topk_value_count;
	uint32_t indexer_kernel_count;
	uint32_t hot_path_allocation_count;
	uint32_t sentinel_violation_count;
	uint32_t unsupported_shape_count;
} SparkCudaSparseIndexReport;

SparkStatus SparkValidateCudaSparseIndexRequest(const SparkCudaSparseIndexRequest *request);
SparkStatus SparkRunCudaSparseIndexTopK(const SparkCudaSparseIndexRequest *request, const float *device_scores, float *device_topk_scores, uint32_t *device_topk_indices, SparkCudaSparseIndexReport *report);

#ifdef __cplusplus
}
#endif
