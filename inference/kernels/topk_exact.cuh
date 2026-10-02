#pragma once

#include "inference/kernels/topk.cuh"

#define LM_TOPK_RADIX_BITS 8u
#define LM_TOPK_BUCKETS (1u << LM_TOPK_RADIX_BITS)
#define LM_TOPK_EXACT_PASSES (32u / LM_TOPK_RADIX_BITS)
#define LM_TOPK_EXACT_PREFIX 0u
#define LM_TOPK_EXACT_MASK 1u
#define LM_TOPK_EXACT_NEED 2u
#define LM_TOPK_EXACT_EMITTED 3u
#define LM_TOPK_EXACT_SEEN 4u
#define LM_TOPK_EXACT_STATE 5u
#define LM_TOPK_WARP_LANES 32u

template<uint32_t THREADS>
static __device__ __forceinline__ void LmTopkExactPass(const float *scores, uint32_t n, uint32_t shift, uint32_t *histogram, uint32_t *state)
{
	uint32_t index,key,running,bucket;
	for (index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS)
		histogram[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < n; index += THREADS)
	{
		key = LmTopkKey(scores[index]);
		if ( (key & state[LM_TOPK_EXACT_MASK]) == state[LM_TOPK_EXACT_PREFIX] )
			atomicAdd(&histogram[(key >> shift) & (LM_TOPK_BUCKETS - 1u)],1u);
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		running = 0u;
		for (bucket = LM_TOPK_BUCKETS - 1u; bucket > 0u && running + histogram[bucket] < state[LM_TOPK_EXACT_NEED]; --bucket)
			running += histogram[bucket];
		state[LM_TOPK_EXACT_PREFIX] |= bucket << shift;
		state[LM_TOPK_EXACT_MASK] |= (LM_TOPK_BUCKETS - 1u) << shift;
		state[LM_TOPK_EXACT_NEED] -= running;
	}
	__syncthreads();
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmTopkExactEmit(const float *scores, uint32_t n, uint32_t threshold, uint32_t *out, uint32_t *scan, uint32_t *state)
{
	const uint32_t warps = (THREADS + LM_TOPK_WARP_LANES - 1u) / LM_TOPK_WARP_LANES;
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES,warp = threadIdx.x / LM_TOPK_WARP_LANES;
	const uint32_t below = lane != 0u ? 0xffffffffu >> (LM_TOPK_WARP_LANES - lane) : 0u;
	uint32_t start,index,key,greater,equal,greater_mask,equal_mask,greater_before,equal_before,greater_total,equal_total,remaining,other;
	static_assert(THREADS <= LM_TOPK_WARP_LANES * LM_TOPK_WARP_LANES, "one scan slot pair per warp");
	for (start = 0u; start < n; start += THREADS)
	{
		index = start + threadIdx.x;
		key = index < n ? LmTopkKey(scores[index]) : 0u;
		greater = index < n && key > threshold ? 1u : 0u;
		equal = index < n && key == threshold ? 1u : 0u;
		greater_mask = __ballot_sync(0xffffffffu,greater != 0u);
		equal_mask = __ballot_sync(0xffffffffu,equal != 0u);
		if ( lane == 0u )
		{
			scan[warp] = (uint32_t)__popc(greater_mask);
			scan[warps + warp] = (uint32_t)__popc(equal_mask);
		}
		__syncthreads();
		greater_before = (uint32_t)__popc(greater_mask & below);
		equal_before = (uint32_t)__popc(equal_mask & below);
		greater_total = 0u;
		equal_total = 0u;
		for (other = 0u; other < warps; other++)
		{
			if ( other < warp )
			{
				greater_before += scan[other];
				equal_before += scan[warps + other];
			}
			greater_total += scan[other];
			equal_total += scan[warps + other];
		}
		remaining = state[LM_TOPK_EXACT_NEED] > state[LM_TOPK_EXACT_SEEN] ? state[LM_TOPK_EXACT_NEED] - state[LM_TOPK_EXACT_SEEN] : 0u;
		if ( greater != 0u || (equal != 0u && equal_before < remaining) )
			out[state[LM_TOPK_EXACT_EMITTED] + greater_before + (equal_before < remaining ? equal_before : remaining)] = index;
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			state[LM_TOPK_EXACT_EMITTED] += greater_total + (equal_total < remaining ? equal_total : remaining);
			state[LM_TOPK_EXACT_SEEN] += equal_total;
		}
		__syncthreads();
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkExactKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ out_indices)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	const float *row_scores = scores + (uint64_t)blockIdx.x * n;
	uint32_t *out = out_indices + (uint64_t)blockIdx.x * k;
	uint32_t index,pass;
	static_assert(THREADS < 65536u, "the packed scan counts one tile in 16 bits");
	if ( n <= k )
	{
		for (index = threadIdx.x; index < k; index += THREADS)
			out[index] = index < n ? index : 0xffffffffu;
		return;
	}
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = k;
		state[LM_TOPK_EXACT_EMITTED] = 0u;
		state[LM_TOPK_EXACT_SEEN] = 0u;
	}
	__syncthreads();
	for (pass = 0u; pass < LM_TOPK_EXACT_PASSES; ++pass)
		LmTopkExactPass<THREADS>(row_scores,n,32u - LM_TOPK_RADIX_BITS * (pass + 1u),histogram,state);
	LmTopkExactEmit<THREADS>(row_scores,n,state[LM_TOPK_EXACT_PREFIX],out,scan,state);
}
