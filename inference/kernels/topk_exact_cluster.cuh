#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "runtime/launch.h"
#include <cooperative_groups.h>

#define LM_TOPK_EXACT_CLUSTER_BLOCKS 8u
#define LM_TOPK_EXACT_CLUSTER_THREADS 512u
#define LM_TOPK_EXACT_CLUSTER_KEYS 8192u

template<uint32_t THREADS, uint32_t BLOCKS>
static __device__ __forceinline__ uint32_t LmTopkExactClusterPass(cooperative_groups::cluster_group &cluster, const uint32_t *keys, uint32_t length, uint32_t shift,
	uint32_t *histogram, uint32_t (*counts)[LM_TOPK_BUCKETS], uint32_t *scan, uint32_t *state, uint32_t *chosen)
{
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES,warp = threadIdx.x / LM_TOPK_WARP_LANES;
	uint32_t index,rank,key,bucket = 0u,total = 0u,inclusive,other,value;
	for (index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS)
		histogram[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < length; index += THREADS)
	{
		key = keys[index];
		if ( (key & state[LM_TOPK_EXACT_MASK]) == state[LM_TOPK_EXACT_PREFIX] )
			atomicAdd(&histogram[(key >> shift) & (LM_TOPK_BUCKETS - 1u)],1u);
	}
	LmClusterBarrier();
	if ( threadIdx.x < LM_TOPK_BUCKETS )
	{
		bucket = LM_TOPK_BUCKETS - 1u - threadIdx.x;
		for (rank = 0u; rank < BLOCKS; rank++)
		{
			value = cluster.map_shared_rank(histogram,rank)[bucket];
			counts[rank][bucket] = value;
			total += value;
		}
	}
	inclusive = total;
	for (other = 1u; other < LM_TOPK_WARP_LANES; other <<= 1u)
	{
		value = __shfl_up_sync(0xffffffffu,inclusive,other);
		if ( lane >= other )
			inclusive += value;
	}
	if ( lane == LM_TOPK_WARP_LANES - 1u )
		scan[warp] = inclusive;
	__syncthreads();
	if ( threadIdx.x < LM_TOPK_BUCKETS )
	{
		for (other = 0u; other < warp; other++)
			inclusive += scan[other];
		if ( inclusive >= state[LM_TOPK_EXACT_NEED] && inclusive - total < state[LM_TOPK_EXACT_NEED] )
		{
			chosen[0] = bucket;
			chosen[1] = inclusive - total;
		}
	}
	__syncthreads();
	return(chosen[0]);
}

template<uint32_t THREADS, uint32_t BLOCKS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkExactClusterKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ out_indices)
{
	__shared__ uint32_t keys[LM_TOPK_EXACT_CLUSTER_KEYS];
	__shared__ uint32_t histogram[2][LM_TOPK_BUCKETS];
	__shared__ uint32_t counts[BLOCKS][LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t chosen[2],greater[BLOCKS],equal[BLOCKS];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = cluster.block_rank(),row = blockIdx.x / BLOCKS;
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES,warp = threadIdx.x / LM_TOPK_WARP_LANES;
	const uint32_t span = (n + BLOCKS - 1u) / BLOCKS,first = part * span;
	const uint32_t length = first < n ? (n - first < span ? n - first : span) : 0u;
	const float *row_scores = scores + (uint64_t)row * n + first;
	uint32_t index,pass,bucket,shift,sum,greater_before = 0u,equal_before = 0u;
	static_assert(THREADS >= LM_TOPK_BUCKETS && THREADS / LM_TOPK_WARP_LANES >= BLOCKS, "one bucket per thread and one warp per cluster rank");
	for (index = threadIdx.x; index < length; index += THREADS)
		keys[index] = LmTopkKey(row_scores[index]);
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = k;
	}
	if ( threadIdx.x < BLOCKS )
	{
		greater[threadIdx.x] = 0u;
		equal[threadIdx.x] = 0u;
	}
	for (pass = 0u; pass < LM_TOPK_EXACT_PASSES; ++pass)
	{
		shift = 32u - LM_TOPK_RADIX_BITS * (pass + 1u);
		bucket = LmTopkExactClusterPass<THREADS,BLOCKS>(cluster,keys,length,shift,histogram[pass & 1u],counts,scan,state,chosen);
		if ( warp < BLOCKS )
		{
			sum = 0u;
			for (index = bucket + 1u + lane; index < LM_TOPK_BUCKETS; index += LM_TOPK_WARP_LANES)
				sum += counts[warp][index];
			for (index = LM_TOPK_WARP_LANES / 2u; index > 0u; index >>= 1u)
				sum += __shfl_xor_sync(0xffffffffu,sum,index);
			if ( lane == 0u )
			{
				greater[warp] += sum;
				if ( pass + 1u == LM_TOPK_EXACT_PASSES )
					equal[warp] = counts[warp][bucket];
			}
		}
		if ( threadIdx.x == 0u )
		{
			state[LM_TOPK_EXACT_PREFIX] |= bucket << shift;
			state[LM_TOPK_EXACT_MASK] |= (LM_TOPK_BUCKETS - 1u) << shift;
			state[LM_TOPK_EXACT_NEED] -= chosen[1];
		}
		__syncthreads();
	}
	LmClusterArrive();
	if ( threadIdx.x == 0u )
	{
		for (index = 0u; index < part; index++)
		{
			greater_before += greater[index];
			equal_before += equal[index];
		}
		state[LM_TOPK_EXACT_EMITTED] = greater_before + (equal_before < state[LM_TOPK_EXACT_NEED] ? equal_before : state[LM_TOPK_EXACT_NEED]);
		state[LM_TOPK_EXACT_SEEN] = equal_before;
	}
	__syncthreads();
	LmTopkExactEmitKeys<THREADS>(LmTopkSharedKeys{keys},length,first,state[LM_TOPK_EXACT_PREFIX],out_indices + (uint64_t)row * k,scan,state);
	LmClusterWait();
}

static inline uint32_t LmTopkExactClusterFits(uint32_t rows, uint32_t n, uint32_t k)
{
	return(rows != 0u && rows <= LM_TOPK_EXACT_CHUNKED_ROWS && k != 0u && n > k && n <= LM_TOPK_EXACT_CLUSTER_BLOCKS * LM_TOPK_EXACT_CLUSTER_KEYS ? 1u : 0u);
}

template<uint32_t THREADS>
static inline cudaError_t LmTopkExactRowsLaunch(const float *scores, uint32_t rows, uint32_t n, uint32_t k, uint32_t *out_indices, cudaStream_t stream)
{
	if ( LmTopkExactClusterFits(rows,n,k) != 0u )
		return(LmLaunchClusterKernel((LmTopkExactClusterKernel<LM_TOPK_EXACT_CLUSTER_THREADS,LM_TOPK_EXACT_CLUSTER_BLOCKS>),
			dim3(rows * LM_TOPK_EXACT_CLUSTER_BLOCKS),dim3(LM_TOPK_EXACT_CLUSTER_THREADS),0,stream,LM_TOPK_EXACT_CLUSTER_BLOCKS,0u,scores,n,k,out_indices));
	LM_LAUNCH((LmTopkExactKernel<THREADS>),rows,THREADS,0,stream,scores,n,k,out_indices);
	return(cudaPeekAtLastError());
}
