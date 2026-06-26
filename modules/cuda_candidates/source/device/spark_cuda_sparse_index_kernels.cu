#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_sparse_index_kernels.h"

#define SPARK_CUDA_SPARSE_INDEX_THREADS 256u

static uint64_t SparkCudaSparseIndexScoreCountHost(const SparkCudaSparseIndexRequest *request)
{
	return (uint64_t)request->row_count * (uint64_t)request->candidate_count;
}

static uint64_t SparkCudaSparseIndexTopKCountHost(const SparkCudaSparseIndexRequest *request)
{
	return (uint64_t)request->row_count * (uint64_t)request->top_k;
}

static void SparkCudaSparseIndexFillReportShape(const SparkCudaSparseIndexRequest *request, SparkCudaSparseIndexReport *report)
{
	report->score_count = SparkCudaSparseIndexScoreCountHost(request);
	report->topk_value_count = SparkCudaSparseIndexTopKCountHost(request);
}

static __device__ uint32_t SparkCudaSparseIndexBetter(float next_score, uint32_t next_id, float best_score, uint32_t best_id)
{
	if (next_score > best_score)
		return 1u;
	if (next_score == best_score && next_id < best_id)
		return 1u;
	return 0u;
}

static __device__ uint32_t SparkCudaSparseIndexAlreadySelected(const uint32_t *selected_indices, uint32_t selected_count, uint32_t candidate_index)
{
	uint32_t selected_index;

	for (selected_index = 0u; selected_index < selected_count; ++selected_index)
	{
		if (selected_indices[selected_index] == candidate_index)
			return 1u;
	}
	return 0u;
}

static __device__ void SparkCudaSparseIndexBlockBest(float local_score, uint32_t local_id, float *best_scores, uint32_t *best_ids)
{
	uint32_t stride;

	best_scores[threadIdx.x] = local_score;
	best_ids[threadIdx.x] = local_id;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
		{
			if (SparkCudaSparseIndexBetter(best_scores[threadIdx.x + stride], best_ids[threadIdx.x + stride], best_scores[threadIdx.x], best_ids[threadIdx.x]) != 0u)
			{
				best_scores[threadIdx.x] = best_scores[threadIdx.x + stride];
				best_ids[threadIdx.x] = best_ids[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}
}

static __global__ void SparkCudaSparseIndexTopKKernel(SparkCudaSparseIndexRequest request, const float *scores, float *topk_scores, uint32_t *topk_indices)
{
	__shared__ float best_scores[SPARK_CUDA_SPARSE_INDEX_THREADS];
	__shared__ uint32_t best_ids[SPARK_CUDA_SPARSE_INDEX_THREADS];
	__shared__ uint32_t selected_indices[SPARKPIPE_CUDA_SPARSE_INDEX_MAX_TOP_K];
	uint32_t selected_index;
	uint32_t candidate_index;
	uint32_t row_index;
	uint32_t local_id;
	uint64_t row_offset;
	uint64_t output_offset;
	float local_score;
	float candidate_score;

	row_index = blockIdx.x;
	row_offset = (uint64_t)row_index * (uint64_t)request.score_stride;
	if (threadIdx.x < SPARKPIPE_CUDA_SPARSE_INDEX_MAX_TOP_K)
		selected_indices[threadIdx.x] = SPARKPIPE_CUDA_SPARSE_INDEX_INVALID;
	__syncthreads();
	for (selected_index = 0u; selected_index < request.top_k; ++selected_index)
	{
		local_score = -FLT_MAX;
		local_id = SPARKPIPE_CUDA_SPARSE_INDEX_INVALID;
		for (candidate_index = threadIdx.x; candidate_index < request.candidate_count; candidate_index += blockDim.x)
		{
			if (SparkCudaSparseIndexAlreadySelected(selected_indices, selected_index, candidate_index) != 0u)
				continue;
			candidate_score = scores[row_offset + candidate_index];
			if (SparkCudaSparseIndexBetter(candidate_score, candidate_index, local_score, local_id) != 0u)
			{
				local_score = candidate_score;
				local_id = candidate_index;
			}
		}
		SparkCudaSparseIndexBlockBest(local_score, local_id, best_scores, best_ids);
		if (threadIdx.x == 0u)
		{
			selected_indices[selected_index] = best_ids[0];
			output_offset = ((uint64_t)row_index * (uint64_t)request.top_k) + (uint64_t)selected_index;
			topk_scores[output_offset] = best_scores[0];
			topk_indices[output_offset] = best_ids[0];
		}
		__syncthreads();
	}
}

extern "C" SparkStatus SparkRunCudaSparseIndexTopK(const SparkCudaSparseIndexRequest *request, const float *device_scores, float *device_topk_scores, uint32_t *device_topk_indices, SparkCudaSparseIndexReport *report)
{
	cudaError_t cuda_status;
	SparkStatus status;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	status = SparkValidateCudaSparseIndexRequest(request);
	if (status != SPARK_STATUS_OK)
		return status;
	if (device_scores == 0 || device_topk_scores == 0 || device_topk_indices == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkCudaSparseIndexFillReportShape(request, report);
	SparkCudaSparseIndexTopKKernel<<<request->row_count, SPARK_CUDA_SPARSE_INDEX_THREADS>>>(*request, device_scores, device_topk_scores, device_topk_indices);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SPARK_STATUS_INTERNAL_ERROR;
	report->indexer_kernel_count = 1u;
	report->hot_path_allocation_count = 0u;
	return SPARK_STATUS_OK;
}
