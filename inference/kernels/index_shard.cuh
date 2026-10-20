#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/topk.cuh"

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

template<uint32_t THREADS, uint32_t CAPACITY>
static __device__ __forceinline__ void LmIndexShardSortPositions(uint32_t *positions)
{
	uint32_t size,stride,index,other,left,right;
	for (size = 2u; size <= CAPACITY; size <<= 1u)
		for (stride = size >> 1u; stride != 0u; stride >>= 1u)
		{
			for (index = threadIdx.x; index < CAPACITY; index += THREADS)
			{
				other = index ^ stride;
				if ( other <= index )
					continue;
				left = positions[index];
				right = positions[other];
				if ( (left > right) == ((index & size) == 0u) )
				{
					positions[index] = right;
					positions[other] = left;
				}
			}
			__syncthreads();
		}
}

static __device__ __forceinline__ uint64_t LmIndexShardCandidateRank(uint2 candidate)
{
	return ((uint64_t)LmTopkKey(__uint_as_float(candidate.x)) << 32u) | (uint64_t)(0xffffffffu - candidate.y);
}

template<uint32_t THREADS, uint32_t CAPACITY>
__global__ __launch_bounds__(THREADS, 1)
void LmIndexShardCandidateTopkKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep, uint32_t positions, uint32_t k, uint32_t fill, uint32_t *__restrict__ selected)
{
	__shared__ uint32_t histogram[256];
	__shared__ uint32_t chosen[CAPACITY];
	__shared__ uint64_t prefix,mask;
	__shared__ uint32_t need,done,count;
	static_assert((CAPACITY & (CAPACITY - 1u)) == 0u, "the position sort needs a power-of-two capacity");
	const uint32_t row = blockIdx.x,total = degree * keep;
	uint32_t index,pass,shift,bucket,running,cursor,next,value;
	uint2 candidate;
	uint64_t rank;
	if ( threadIdx.x == 0u )
	{
		prefix = 0u;
		mask = 0u;
		need = k;
		done = 0u;
		count = 0u;
	}
	__syncthreads();
	for (pass = 0u; pass < 8u && done == 0u; pass++)
	{
		shift = 56u - 8u * pass;
		for (index = threadIdx.x; index < 256u; index += THREADS)
			histogram[index] = 0u;
		__syncthreads();
		for (index = threadIdx.x; index < total; index += THREADS)
		{
			candidate = gathered[(uint64_t)(index / keep) * rank_stride_candidates + (uint64_t)row * keep + index % keep];
			rank = LmIndexShardCandidateRank(candidate);
			if ( candidate.y < positions && (rank & mask) == prefix )
				atomicAdd(&histogram[(uint32_t)(rank >> shift) & 255u],1u);
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			running = 0u;
			for (bucket = 0u; bucket < 256u; bucket++)
				running += histogram[bucket];
			if ( pass == 0u && running <= need )
				done = 1u;
			else
			{
				running = 0u;
				for (bucket = 255u; bucket > 0u && running + histogram[bucket] < need; --bucket)
					running += histogram[bucket];
				prefix |= (uint64_t)bucket << shift;
				mask |= (uint64_t)255u << shift;
				done = histogram[bucket] == need - running ? 1u : 0u;
				need -= running;
			}
		}
		__syncthreads();
	}
	for (index = threadIdx.x; index < total; index += THREADS)
	{
		candidate = gathered[(uint64_t)(index / keep) * rank_stride_candidates + (uint64_t)row * keep + index % keep];
		if ( candidate.y < positions && LmIndexShardCandidateRank(candidate) >= prefix )
		{
			cursor = atomicAdd(&count,1u);
			if ( cursor < CAPACITY )
				chosen[cursor] = candidate.y;
		}
	}
	__syncthreads();
	for (index = count + threadIdx.x; index < CAPACITY; index += THREADS)
		chosen[index] = LM_INDEX_SHARD_NO_POSITION;
	__syncthreads();
	LmIndexShardSortPositions<THREADS,CAPACITY>(chosen);
	if ( fill != 0u && count < k )
	{
		if ( threadIdx.x == 0u )
			for (cursor = 0u, next = count, value = 0u; next < k && value < positions; value++)
			{
				if ( cursor < count && chosen[cursor] == value )
					cursor++;
				else
					chosen[next++] = value;
			}
		__syncthreads();
		LmIndexShardSortPositions<THREADS,CAPACITY>(chosen);
	}
	for (index = threadIdx.x; index < k; index += THREADS)
		selected[(uint64_t)row * k + index] = chosen[index];
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
template<uint32_t THREADS, uint32_t CAPACITY>
static inline cudaError_t LmIndexShardCandidateTopkLaunch(const uint2 *gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep, uint32_t rows, uint32_t positions, uint32_t k, uint32_t fill, uint32_t *selected, cudaStream_t stream)
{
	if (gathered == 0 || selected == 0 || rows == 0u || keep == 0u || k == 0u || k > CAPACITY || positions == 0u || degree == 0u || degree > SPARK_KV_SHARD_MAX_DEGREE || rank_stride_candidates < (uint64_t)rows * keep)
		return cudaErrorInvalidValue;
	LM_LAUNCH((LmIndexShardCandidateTopkKernel<THREADS,CAPACITY>), rows, THREADS, 0, stream, gathered, rank_stride_candidates, degree, keep, positions, k, fill, selected);
	return cudaPeekAtLastError();
}
#endif
