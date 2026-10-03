#pragma once

#include "inference/kernels/topk.cuh"
#include "inference/kernels/topk_exact_plan.h"

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
static __device__ __forceinline__ void LmTopkExactSelect(const float *row_scores, uint32_t n, uint32_t k, uint32_t *out, uint32_t *histogram, uint32_t *scan, uint32_t *state)
{
	uint32_t index,pass;
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

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkExactKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ out_indices)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	LmTopkExactSelect<THREADS>(scores + (uint64_t)blockIdx.x * n,n,k,out_indices + (uint64_t)blockIdx.x * k,histogram,scan,state);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkExactChunkKernel(const float *__restrict__ values, const uint32_t *__restrict__ positions, uint32_t n, uint32_t k, uint32_t chunk, float *__restrict__ out_values, uint32_t *__restrict__ out_positions)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	const uint32_t first = blockIdx.x * chunk,chunks = gridDim.x;
	const uint32_t length = n - first < chunk ? n - first : chunk;
	const float *row_values = values + (uint64_t)blockIdx.y * n;
	const uint32_t *row_positions = positions != 0 ? positions + (uint64_t)blockIdx.y * n : 0;
	const uint64_t base = ((uint64_t)blockIdx.y * chunks + blockIdx.x) * k;
	uint32_t index,local;
	LmTopkExactSelect<THREADS>(row_values + first,length,k,out_positions + base,histogram,scan,state);
	__syncthreads();
	for (index = threadIdx.x; index < k; index += THREADS)
	{
		local = out_positions[base + index];
		out_values[base + index] = local != 0xffffffffu ? row_values[first + local] : -INFINITY;
		out_positions[base + index] = local == 0xffffffffu ? 0xffffffffu : row_positions != 0 ? row_positions[first + local] : first + local;
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkExactGatherKernel(const float *__restrict__ values, const uint32_t *__restrict__ positions, uint32_t n, uint32_t k, uint32_t *__restrict__ out_indices)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	uint32_t *out = out_indices + (uint64_t)blockIdx.x * k;
	const uint32_t *row_positions = positions + (uint64_t)blockIdx.x * n;
	uint32_t index;
	LmTopkExactSelect<THREADS>(values + (uint64_t)blockIdx.x * n,n,k,out,histogram,scan,state);
	__syncthreads();
	for (index = threadIdx.x; index < k; index += THREADS)
		out[index] = out[index] != 0xffffffffu ? row_positions[out[index]] : 0xffffffffu;
}

template<uint32_t THREADS>
static inline cudaError_t LmTopkExactLaunch(const float *scores, uint32_t rows, uint32_t n, uint32_t k, uint32_t chunk, uint32_t chunked_rows, float *scratch_values, uint32_t *scratch_positions, uint64_t scratch_entries, uint32_t *out_indices, cudaStream_t stream)
{
	const float *values = scores;
	const uint32_t *positions = 0;
	uint32_t length = n,half = 0u;
	uint64_t level;
	if ( rows > chunked_rows || LmTopkExactCandidateEntries(n,k,chunk) == 0u )
	{
		LM_LAUNCH((LmTopkExactKernel<THREADS>),rows,THREADS,0,stream,scores,n,k,out_indices);
		return(cudaPeekAtLastError());
	}
	if ( scratch_values == 0 || scratch_positions == 0 || (uint64_t)rows * LmTopkExactCandidateEntries(n,k,chunk) * 2u > scratch_entries )
		return(cudaErrorInvalidValue);
	while ( (level = LmTopkExactCandidateEntries(length,k,chunk)) != 0u )
	{
		float *next_values = scratch_values + (uint64_t)half * (scratch_entries / 2u);
		uint32_t *next_positions = scratch_positions + (uint64_t)half * (scratch_entries / 2u);
		LM_LAUNCH((LmTopkExactChunkKernel<THREADS>),dim3((length + chunk - 1u) / chunk,rows),THREADS,0,stream,values,positions,length,k,chunk,next_values,next_positions);
		values = next_values;
		positions = next_positions;
		length = (uint32_t)level;
		half ^= 1u;
	}
	LM_LAUNCH((LmTopkExactGatherKernel<THREADS>),rows,THREADS,0,stream,values,positions,length,k,out_indices);
	return(cudaPeekAtLastError());
}
