#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/kv_shard.cuh"

#define LM_INDEX_SHARD_CANDIDATE_BYTES 8u
#define LM_INDEX_SHARD_NO_POSITION 0xffffffffu

#ifdef __CUDACC__
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmIndexShardCandidatePackKernel(SparkKvShard shard, const float *__restrict__ local_scores, uint32_t score_stride, const uint32_t *__restrict__ local_selected, uint32_t keep, uint2 *__restrict__ candidates)
{
	uint32_t row = blockIdx.x, entry, local;
	float score;
	for (entry = threadIdx.x; entry < keep; entry += THREADS)
	{
		local = local_selected[(uint64_t)row * keep + entry];
		score = local < score_stride ? local_scores[(uint64_t)row * score_stride + local] : -INFINITY;
		candidates[(uint64_t)row * keep + entry] = make_uint2(__float_as_uint(score), local < score_stride && score > -INFINITY ? SparkKvShardLocalPosition(shard, local) : LM_INDEX_SHARD_NO_POSITION);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmIndexShardCandidateScatterKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep, uint32_t score_stride, float *__restrict__ scores)
{
	uint32_t row = blockIdx.x, index, source;
	uint2 candidate;
	for (index = threadIdx.x; index < score_stride; index += THREADS)
		scores[(uint64_t)row * score_stride + index] = -INFINITY;
	__syncthreads();
	for (index = threadIdx.x; index < degree * keep; index += THREADS)
	{
		source = index / keep;
		candidate = gathered[(uint64_t)source * rank_stride_candidates + (uint64_t)row * keep + index % keep];
		if (candidate.y < score_stride)
			scores[(uint64_t)row * score_stride + candidate.y] = __uint_as_float(candidate.x);
	}
}

template<uint32_t THREADS>
static inline cudaError_t LmIndexShardCandidatePackLaunch(SparkKvShard shard, const float *local_scores, uint32_t score_stride, const uint32_t *local_selected, uint32_t keep, uint32_t rows, uint2 *candidates, cudaStream_t stream)
{
	if (local_scores == 0 || local_selected == 0 || candidates == 0 || rows == 0u || keep == 0u || score_stride == 0u || shard.degree == 0u || shard.grain == 0u)
		return cudaErrorInvalidValue;
	LM_LAUNCH((LmIndexShardCandidatePackKernel<THREADS>), rows, THREADS, 0, stream, shard, local_scores, score_stride, local_selected, keep, candidates);
	return cudaPeekAtLastError();
}

template<uint32_t THREADS>
static inline cudaError_t LmIndexShardCandidateScatterLaunch(const uint2 *gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep, uint32_t rows, uint32_t score_stride, float *scores, cudaStream_t stream)
{
	if (gathered == 0 || scores == 0 || rows == 0u || keep == 0u || score_stride == 0u || degree == 0u || degree > SPARK_KV_SHARD_MAX_DEGREE || rank_stride_candidates < (uint64_t)rows * keep)
		return cudaErrorInvalidValue;
	LM_LAUNCH((LmIndexShardCandidateScatterKernel<THREADS>), rows, THREADS, 0, stream, gathered, rank_stride_candidates, degree, keep, score_stride, scores);
	return cudaPeekAtLastError();
}
#endif
