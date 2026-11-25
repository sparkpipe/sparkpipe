#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "runtime/launch.h"
#include <cooperative_groups.h>
#include "sparkpipe/spark_gpu.h"

#define LM_TOPK_EXACT_CLUSTER_BLOCKS 8u
#define LM_TOPK_EXACT_CLUSTER_THREADS 512u
#define LM_TOPK_EXACT_CLUSTER_KEYS 8192u

template<uint32_t THREADS, uint32_t BLOCKS>
static __device__ __forceinline__ uint32_t LmTopkExactClusterSelectBucket(cooperative_groups::cluster_group &cluster,
	uint32_t *histogram, uint32_t (*counts)[LM_TOPK_BUCKETS], uint32_t *scan, uint32_t need, uint32_t *chosen)
{
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES,warp = threadIdx.x / LM_TOPK_WARP_LANES;
	uint32_t rank,bucket = 0u,total = 0u,inclusive,other,value;
	LmClusterBarrier();
	if ( threadIdx.x < LM_TOPK_BUCKETS )
	{
		bucket = LM_TOPK_BUCKETS - 1u - threadIdx.x;
		#pragma unroll
		for (rank = 0u; rank < BLOCKS; rank++)
		{
			value = cluster.map_shared_rank(histogram,rank)[bucket];
			counts[rank][bucket] = value;
			total += value;
		}
	}
	inclusive = total;
	#pragma unroll
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
		#pragma unroll
		for (other = 0u; other < warp; other++)
			inclusive += scan[other];
		if ( inclusive >= need && inclusive - total < need )
		{
			chosen[0] = bucket;
			chosen[1] = inclusive - total;
		}
	}
	__syncthreads();
	return(chosen[0]);
}

template<uint32_t THREADS, uint32_t BLOCKS>
static __device__ __forceinline__ uint32_t LmTopkExactClusterPass(cooperative_groups::cluster_group &cluster, const uint32_t *keys, uint32_t length, uint32_t shift,
	uint32_t prefix, uint32_t mask, uint32_t need, uint32_t *histogram, uint32_t (*counts)[LM_TOPK_BUCKETS], uint32_t *scan, uint32_t *chosen)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t vecs = length >> 2u;
	uint32_t index;
	for (index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS)
		histogram[index] = 0u;
	__syncthreads();
	const uint4 *keys4 = (const uint4 *)keys;
	for (uint32_t base = 0u; base < vecs; base += THREADS)
	{
		const uint32_t v = base + threadIdx.x;
		uint32_t b0 = 0xffffffffu, b1 = 0xffffffffu, b2 = 0xffffffffu, b3 = 0xffffffffu;
		if ( v < vecs )
		{
			const uint4 k4 = keys4[v];
			if ( (k4.x & mask) == prefix ) b0 = (k4.x >> shift) & (LM_TOPK_BUCKETS - 1u);
			if ( (k4.y & mask) == prefix ) b1 = (k4.y >> shift) & (LM_TOPK_BUCKETS - 1u);
			if ( (k4.z & mask) == prefix ) b2 = (k4.z >> shift) & (LM_TOPK_BUCKETS - 1u);
			if ( (k4.w & mask) == prefix ) b3 = (k4.w >> shift) & (LM_TOPK_BUCKETS - 1u);
		}
		LmTopkHistWarpAdd(histogram, b0, lane);
		LmTopkHistWarpAdd(histogram, b1, lane);
		LmTopkHistWarpAdd(histogram, b2, lane);
		LmTopkHistWarpAdd(histogram, b3, lane);
	}
	for (index = (vecs << 2u) + threadIdx.x; index < length; index += THREADS)
	{
		const uint32_t key = keys[index];
		if ( (key & mask) == prefix )
			atomicAdd(&histogram[(key >> shift) & (LM_TOPK_BUCKETS - 1u)], 1u);
	}
	return(LmTopkExactClusterSelectBucket<THREADS,BLOCKS>(cluster,histogram,counts,scan,need,chosen));
}

template<uint32_t THREADS, uint32_t BLOCKS>
__global__ __launch_bounds__(THREADS, 2)
void LmTopkExactClusterKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ out_indices, const uint32_t *__restrict__ sequence_of_row = 0, const uint32_t *__restrict__ context_length = 0, const uint32_t *__restrict__ row_position = 0, uint32_t dense_limit = 0u, uint32_t pool_stride = 1u)
{
	__shared__ __align__(16) uint32_t keys[LM_TOPK_EXACT_CLUSTER_KEYS];
	__shared__ uint32_t histogram[2][LM_TOPK_BUCKETS];
	__shared__ uint32_t counts[BLOCKS][LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t chosen[2],greater[BLOCKS],equal[BLOCKS];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = cluster.block_rank(),row = blockIdx.x / BLOCKS;
	uint32_t skip = 0u;
	LmDependentWait();
	const uint32_t eff_n = LmTopkExactRowActive(row,n,sequence_of_row,context_length,row_position,dense_limit,pool_stride,&skip);
	if ( skip != 0u )
	{
		LmDependentRelease();
		return;
	}
	if ( eff_n <= k )
	{
		if ( part == 0u )
			for (uint32_t i = threadIdx.x; i < k; i += THREADS)
				out_indices[(uint64_t)row * k + i] = i < eff_n ? i : 0xffffffffu;
		LmDependentRelease();
		return;
	}
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES,warp = threadIdx.x / LM_TOPK_WARP_LANES;
	const uint32_t span = (eff_n + BLOCKS - 1u) / BLOCKS,first = part * span;
	const uint32_t length = first < eff_n ? (eff_n - first < span ? eff_n - first : span) : 0u;
	const float *row_scores = scores + (uint64_t)row * n + first;
	uint32_t index,pass,bucket,shift,sum,greater_before = 0u,equal_before = 0u;
	uint32_t prefix = 0u,mask = 0u,need = k;
	static_assert(THREADS >= LM_TOPK_BUCKETS && THREADS / LM_TOPK_WARP_LANES >= BLOCKS, "one bucket per thread and one warp per cluster rank");
	for (index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS)
		histogram[0][index] = 0u;
	if ( threadIdx.x < BLOCKS )
	{
		greater[threadIdx.x] = 0u;
		equal[threadIdx.x] = 0u;
	}
	__syncthreads();
	if ( (((uintptr_t)row_scores & 15u) == 0u) && length >= 4u )
	{
		const uint32_t vecs = length >> 2u;
		const float4 *scores4 = (const float4 *)row_scores;
		uint4 *keys4 = (uint4 *)keys;
		for (uint32_t base = 0u; base < vecs; base += THREADS)
		{
			const uint32_t v = base + threadIdx.x;
			uint32_t b0 = 0xffffffffu, b1 = 0xffffffffu, b2 = 0xffffffffu, b3 = 0xffffffffu;
			if ( v < vecs )
			{
				const float4 f = __ldg(scores4 + v);
				const uint4 k4 = make_uint4(LmTopkKey(f.x), LmTopkKey(f.y), LmTopkKey(f.z), LmTopkKey(f.w));
				keys4[v] = k4;
				b0 = k4.x >> 24u;
				b1 = k4.y >> 24u;
				b2 = k4.z >> 24u;
				b3 = k4.w >> 24u;
			}
			LmTopkHistWarpAdd(histogram[0], b0, lane);
			LmTopkHistWarpAdd(histogram[0], b1, lane);
			LmTopkHistWarpAdd(histogram[0], b2, lane);
			LmTopkHistWarpAdd(histogram[0], b3, lane);
		}
		for (index = (vecs << 2u) + threadIdx.x; index < length; index += THREADS)
		{
			const uint32_t kv = LmTopkKey(row_scores[index]);
			keys[index] = kv;
			atomicAdd(&histogram[0][kv >> 24u], 1u);
		}
	}
	else
	{
		for (uint32_t base = 0u; base < length; base += THREADS)
		{
			index = base + threadIdx.x;
			uint32_t b = 0xffffffffu;
			if ( index < length )
			{
				const uint32_t kv = LmTopkKey(row_scores[index]);
				keys[index] = kv;
				b = kv >> 24u;
			}
			LmTopkHistWarpAdd(histogram[0], b, lane);
		}
	}
	#pragma unroll 1
	for (pass = 0u; pass < LM_TOPK_EXACT_PASSES; ++pass)
	{
		shift = 32u - LM_TOPK_RADIX_BITS * (pass + 1u);
		if ( pass == 0u )
			bucket = LmTopkExactClusterSelectBucket<THREADS,BLOCKS>(cluster,histogram[0],counts,scan,need,chosen);
		else
			bucket = LmTopkExactClusterPass<THREADS,BLOCKS>(cluster,keys,length,shift,prefix,mask,need,histogram[pass & 1u],counts,scan,chosen);
		if ( warp < BLOCKS )
		{
			sum = 0u;
			for (index = bucket + 1u + lane; index < LM_TOPK_BUCKETS; index += LM_TOPK_WARP_LANES)
				sum += counts[warp][index];
			#pragma unroll
			for (index = LM_TOPK_WARP_LANES / 2u; index > 0u; index >>= 1u)
				sum += __shfl_xor_sync(0xffffffffu,sum,index);
			if ( lane == 0u )
			{
				greater[warp] += sum;
				if ( pass + 1u == LM_TOPK_EXACT_PASSES )
					equal[warp] = counts[warp][bucket];
			}
		}
		prefix |= bucket << shift;
		mask |= (LM_TOPK_BUCKETS - 1u) << shift;
		need -= chosen[1];
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
		state[LM_TOPK_EXACT_PREFIX] = prefix;
		state[LM_TOPK_EXACT_MASK] = mask;
		state[LM_TOPK_EXACT_NEED] = need;
		state[LM_TOPK_EXACT_EMITTED] = greater_before + (equal_before < need ? equal_before : need);
		state[LM_TOPK_EXACT_SEEN] = equal_before;
	}
	__syncthreads();
	LmTopkExactEmitKeys<THREADS>(LmTopkSharedKeys{keys},length,first,prefix,out_indices + (uint64_t)row * k,scan,state);
	LmClusterWait();
	LmDependentRelease();
}

static inline uint32_t LmTopkExactClusterFits(uint32_t rows, uint32_t n, uint32_t k)
{
	return(rows != 0u && rows <= LM_TOPK_EXACT_CHUNKED_ROWS && k != 0u && n > k && n <= LM_TOPK_EXACT_CLUSTER_BLOCKS * LM_TOPK_EXACT_CLUSTER_KEYS ? 1u : 0u);
}

template<uint32_t THREADS>
static inline SparkGpuError LmTopkExactRowsLaunch(const float *scores, uint32_t rows, uint32_t n, uint32_t k, uint32_t *out_indices, SparkGpuStream stream, const uint32_t *sequence_of_row = 0, const uint32_t *context_length = 0, const uint32_t *row_position = 0, uint32_t dense_limit = 0u, uint32_t pool_stride = 1u)
{
	if ( LmTopkExactClusterFits(rows,n,k) != 0u )
		return(LmLaunchClusterKernel((LmTopkExactClusterKernel<LM_TOPK_EXACT_CLUSTER_THREADS,LM_TOPK_EXACT_CLUSTER_BLOCKS>),
			dim3(rows * LM_TOPK_EXACT_CLUSTER_BLOCKS),dim3(LM_TOPK_EXACT_CLUSTER_THREADS),0,stream,LM_TOPK_EXACT_CLUSTER_BLOCKS,1u,scores,n,k,out_indices,sequence_of_row,context_length,row_position,dense_limit,pool_stride));
	if ( n > LM_TOPK_EXACT_CACHED_ROW && n > k && k <= LM_TOPK_WIDE_MAX_K )
		LM_LAUNCH_DEPENDENT((LmTopkExactWideKernel),rows,LM_TOPK_WIDE_THREADS,0,stream,scores,n,k,out_indices,sequence_of_row,context_length,row_position,dense_limit,pool_stride);
	else
		LM_LAUNCH_DEPENDENT((LmTopkExactKernel<THREADS>),rows,THREADS,0,stream,scores,n,k,out_indices,sequence_of_row,context_length,row_position,dense_limit,pool_stride);
	return(SparkGpuPeekAtLastError());
}
