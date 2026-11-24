#pragma once

#include <cooperative_groups.h>
#include <stdint.h>

#include "runtime/launch.h"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/norm.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_HC_SITE_STREAMS 4u
#define LM_HC_SITE_MIX ((2u + LM_HC_SITE_STREAMS) * LM_HC_SITE_STREAMS)
#define LM_HC_SITE_CLUSTER 8u
#define LM_HC_SITE_THREADS 256u
#define LM_HC_SITE_ROWS 8u

template<uint32_t HIDDEN>
struct LmHcSiteShape
{
	static constexpr uint32_t kHidden = HIDDEN;
	static constexpr uint32_t kFlat = LM_HC_SITE_STREAMS * HIDDEN;
	static constexpr uint32_t kSlice = kFlat / LM_HC_SITE_CLUSTER;
	static constexpr uint32_t kPreSlice = HIDDEN / LM_HC_SITE_CLUSTER;
	static constexpr uint32_t kVectors = kSlice / 4u / LM_WARP_LANES;
	static_assert(kSlice % (4u * LM_WARP_LANES) == 0u && HIDDEN % LM_HC_SITE_CLUSTER == 0u, "HC slices are whole float4 warp spans");
};

template<uint32_t HIDDEN, uint32_t ROWS>
struct __align__(16) LmHcSiteShared
{
	uint16_t staged[ROWS][LmHcSiteShape<HIDDEN>::kSlice];
	float partial[ROWS][LM_HC_SITE_CLUSTER][LM_HC_SITE_MIX + 1u];
	float mixes[ROWS][LM_HC_SITE_MIX + 1u];
	float reduction[LM_HC_SITE_THREADS / LM_WARP_LANES];
	float pre[ROWS][LM_HC_SITE_STREAMS];
	float post[ROWS][LM_HC_SITE_STREAMS];
};

static __device__ __forceinline__ float LmHcRowSum(float value, uint32_t lane)
{
	const uint32_t row = lane & 12u;
	float total = 0.0f;
#pragma unroll
	for ( uint32_t j = 0u; j < LM_HC_SITE_STREAMS; ++j )
		total += __shfl_sync(0xffffffffu, value, row + j);
	return total;
}

static __device__ __forceinline__ float LmHcColumnSum(float value, uint32_t lane)
{
	const uint32_t column = lane & 3u;
	float total = 0.0f;
#pragma unroll
	for ( uint32_t i = 0u; i < LM_HC_SITE_STREAMS; ++i )
		total += __shfl_sync(0xffffffffu, value, column + i * LM_HC_SITE_STREAMS);
	return total;
}

static __device__ __forceinline__ float LmHcRowMax(float value, uint32_t lane)
{
	const uint32_t row = lane & 12u;
	float maximum = -3.0e38f;
#pragma unroll
	for ( uint32_t j = 0u; j < LM_HC_SITE_STREAMS; ++j )
		maximum = fmaxf(maximum, __shfl_sync(0xffffffffu, value, row + j));
	return maximum;
}

template<uint32_t HIDDEN, uint32_t ROWS>
static __device__ float LmHcStage(LmHcSiteShared<HIDDEN, ROWS> *shared, uint32_t row, const uint16_t *slice)
{
	float total = 0.0f;
	for ( uint32_t element = threadIdx.x; element < LmHcSiteShape<HIDDEN>::kSlice; element += LM_HC_SITE_THREADS )
	{
		const uint16_t raw = slice[element];
		const float value = LmBf16ToFloat(raw);
		shared->staged[row][element] = raw;
		total += value * value;
	}
	return LmBlockSum<LM_HC_SITE_THREADS>(total, shared->reduction);
}

template<uint32_t HIDDEN, uint32_t ROWS>
static __device__ void LmHcDot(LmHcSiteShared<HIDDEN, ROWS> *shared, const float *fn, uint32_t offset, uint32_t slice, uint32_t live)
{
	using S = LmHcSiteShape<HIDDEN>;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	float4 weight[S::kVectors], value;
	uint2 packed;
	for ( uint32_t mix = warp; mix < LM_HC_SITE_MIX; mix += LM_HC_SITE_THREADS / LM_WARP_LANES )
	{
#pragma unroll
		for ( uint32_t vector = 0u; vector < S::kVectors; vector++ )
			weight[vector] = __ldcs((const float4 *)(fn + (uint64_t)mix * S::kFlat + offset) + lane + vector * LM_WARP_LANES);
		for ( uint32_t row = 0u; row < live; ++row )
		{
			float accumulator = 0.0f;
#pragma unroll
			for ( uint32_t vector = 0u; vector < S::kVectors; vector++ )
			{
				packed = ((const uint2 *)shared->staged[row])[lane + vector * LM_WARP_LANES];
				value.x = __uint_as_float(packed.x << 16u);
				value.y = __uint_as_float(packed.x & 0xffff0000u);
				value.z = __uint_as_float(packed.y << 16u);
				value.w = __uint_as_float(packed.y & 0xffff0000u);
				accumulator = fmaf(weight[vector].x, value.x, accumulator);
				accumulator = fmaf(weight[vector].y, value.y, accumulator);
				accumulator = fmaf(weight[vector].z, value.z, accumulator);
				accumulator = fmaf(weight[vector].w, value.w, accumulator);
			}
			for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
				accumulator += __shfl_xor_sync(0xffffffffu, accumulator, step);
			if ( lane == 0u )
				shared->partial[row][slice][mix] = accumulator;
		}
	}
}

template<uint32_t HIDDEN, uint32_t ROWS>
static __device__ void LmHcScatter(cooperative_groups::cluster_group &cluster, LmHcSiteShared<HIDDEN, ROWS> *shared, uint32_t slice, uint32_t live)
{
	for ( uint32_t index = threadIdx.x; index < live * LM_HC_SITE_CLUSTER * (LM_HC_SITE_MIX + 1u); index += LM_HC_SITE_THREADS )
	{
		const uint32_t row = index / (LM_HC_SITE_CLUSTER * (LM_HC_SITE_MIX + 1u)), rank = index / (LM_HC_SITE_MIX + 1u) % LM_HC_SITE_CLUSTER;
		const uint32_t mix = index % (LM_HC_SITE_MIX + 1u);
		if ( rank != slice )
			cluster.map_shared_rank(shared->partial[row][slice], rank)[mix] = shared->partial[row][slice][mix];
	}
}

template<uint32_t HIDDEN, class P, uint32_t ROWS>
static __device__ void LmHcFinish(LmHcSiteShared<HIDDEN, ROWS> *shared, const float *scale3, const float *base, float *mixes, float *pre, float *post,
	uint32_t slice, uint32_t live)
{
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	for ( uint32_t index = threadIdx.x; index < live * (LM_HC_SITE_MIX + 1u); index += LM_HC_SITE_THREADS )
	{
		const uint32_t row = index / (LM_HC_SITE_MIX + 1u), mix = index % (LM_HC_SITE_MIX + 1u);
		float total = 0.0f;
		for ( uint32_t rank = 0u; rank < LM_HC_SITE_CLUSTER; rank++ )
			total += shared->partial[row][rank][mix];
		shared->mixes[row][mix] = total;
	}
	__syncthreads();
	if ( warp < live )
	{
		const float inverse = SparkMathRmsScale(shared->mixes[warp][LM_HC_SITE_MIX], (float)LmHcSiteShape<HIDDEN>::kFlat, P::kNormEpsilon);
		if ( lane < LM_HC_SITE_MIX )
		{
			const float value = shared->mixes[warp][lane] * inverse;
			shared->mixes[warp][lane] = value;
			if ( slice == 0u )
				mixes[(uint64_t)warp * LM_HC_SITE_MIX + lane] = value;
		}
		__syncwarp();
		P::Gates(shared->mixes[warp], scale3, base, shared->pre[warp], slice == 0u ? post + (uint64_t)warp * LM_HC_SITE_STREAMS : shared->post[warp]);
		__syncwarp();
		if ( slice == 0u && lane < LM_HC_SITE_STREAMS )
			pre[(uint64_t)warp * LM_HC_SITE_STREAMS + lane] = shared->pre[warp][lane];
	}
	__syncthreads();
}

template<uint32_t HIDDEN>
static __device__ void LmHcCollapse(const float *pre, const uint16_t *streams, uint16_t *collapsed, uint16_t *snapshot, uint32_t first, uint32_t thread,
	uint32_t threads)
{
	for ( uint32_t element = first + thread; element < first + LmHcSiteShape<HIDDEN>::kPreSlice; element += threads )
	{
		float value = 0.0f;
#pragma unroll
		for ( uint32_t stream = 0u; stream < LM_HC_SITE_STREAMS; ++stream )
		{
			const uint64_t index = (uint64_t)stream * HIDDEN + element;
			const uint16_t raw = streams[index];
			snapshot[index] = raw;
			value += pre[stream] * LmBf16ToFloat(raw);
		}
		collapsed[element] = LmFloatToBf16(value);
	}
}

template<uint32_t HIDDEN, class P, uint32_t ROWS>
__global__ void __cluster_dims__(LM_HC_SITE_CLUSTER, 1, 1) __launch_bounds__(LM_HC_SITE_THREADS) LmHcSiteKernel(const uint16_t *__restrict__ streams_bf16,
	const float *__restrict__ fn_f32, const float *__restrict__ scale3_f32, const float *__restrict__ base_f32, float *__restrict__ mixes_f32,
	float *__restrict__ pre_f32, float *__restrict__ post_f32, float *__restrict__ comb_f32, uint16_t *__restrict__ collapsed_bf16,
	uint16_t *__restrict__ snapshot_bf16, uint32_t rows)
{
	using S = LmHcSiteShape<HIDDEN>;
	__shared__ LmHcSiteShared<HIDDEN, ROWS> shared;
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t slice = cluster.block_rank(), first = blockIdx.x / LM_HC_SITE_CLUSTER * ROWS, warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t live = rows - first < ROWS ? rows - first : ROWS;
	const uint16_t *streams = streams_bf16 + (uint64_t)first * S::kFlat;
	float total;
	asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
	if ( threadIdx.x % LM_WARP_LANES == 0u )
		for ( uint32_t mix = threadIdx.x / LM_WARP_LANES; mix < LM_HC_SITE_MIX; mix += LM_HC_SITE_THREADS / LM_WARP_LANES )
			LmPrefetchL2(fn_f32 + (uint64_t)mix * S::kFlat + slice * S::kSlice, S::kSlice * sizeof(float));
	LmDependentWait();
	for ( uint32_t row = 0u; row < live; ++row )
	{
		total = LmHcStage(&shared, row, streams + (uint64_t)row * S::kFlat + slice * S::kSlice);
		if ( threadIdx.x == 0u )
			shared.partial[row][slice][LM_HC_SITE_MIX] = total;
	}
	LmHcDot(&shared, fn_f32, slice * S::kSlice, slice, live);
	__syncthreads();
	asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
	LmHcScatter(cluster, &shared, slice, live);
	cluster.sync();
	LmHcFinish<HIDDEN, P, ROWS>(&shared, scale3_f32, base_f32, mixes_f32 + (uint64_t)first * LM_HC_SITE_MIX, pre_f32 + (uint64_t)first * LM_HC_SITE_STREAMS,
		post_f32 + (uint64_t)first * LM_HC_SITE_STREAMS, slice, live);
	if ( slice == 0u && warp < live )
		P::Comb(shared.mixes[warp], scale3_f32, base_f32, comb_f32 + (uint64_t)(first + warp) * LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS);
	for ( uint32_t row = 0u; row < live; ++row )
		LmHcCollapse<HIDDEN>(shared.pre[row], streams + (uint64_t)row * S::kFlat, collapsed_bf16 + (uint64_t)(first + row) * HIDDEN,
			snapshot_bf16 + (uint64_t)(first + row) * S::kFlat, slice * S::kPreSlice, threadIdx.x, LM_HC_SITE_THREADS);
}

template<uint32_t HIDDEN, uint32_t MIX_GROUPS>
__global__ __launch_bounds__(LM_HC_SITE_THREADS) void LmHcSiteB1DotKernel(const uint16_t *__restrict__ streams_bf16, const float *__restrict__ fn_f32, float *__restrict__ partials)
{
	using S = LmHcSiteShape<HIDDEN>;
	constexpr uint32_t MIXES_PER_GROUP = LM_HC_SITE_MIX / MIX_GROUPS;
	static_assert(LM_HC_SITE_MIX % MIX_GROUPS == 0u && MIXES_PER_GROUP <= LM_HC_SITE_THREADS / LM_WARP_LANES, "HC B1 mix groups divide mixes evenly");
	__shared__ __align__(16) uint16_t staged[S::kSlice];
	__shared__ float reduction[LM_HC_SITE_THREADS / LM_WARP_LANES];
	const uint32_t slice = blockIdx.x, mg = blockIdx.y, warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t off = slice * S::kSlice;
	const uint16_t *src = streams_bf16 + off;
	if ( lane == 0u && warp < MIXES_PER_GROUP )
		LmPrefetchL2(fn_f32 + (uint64_t)(mg * MIXES_PER_GROUP + warp) * S::kFlat + off, S::kSlice * sizeof(float));
	LmDependentWait();
	if ( mg == 0u )
	{
		float total = 0.0f;
		for ( uint32_t element = threadIdx.x; element < S::kSlice; element += LM_HC_SITE_THREADS )
		{
			const uint16_t raw = src[element];
			const float value = LmBf16ToFloat(raw);
			staged[element] = raw;
			total += value * value;
		}
		total = LmBlockSum<LM_HC_SITE_THREADS>(total, reduction);
		if ( threadIdx.x == 0u )
			partials[slice * (LM_HC_SITE_MIX + 1u) + LM_HC_SITE_MIX] = total;
	}
	else
	{
		for ( uint32_t v = threadIdx.x; v < S::kSlice / 8u; v += LM_HC_SITE_THREADS )
			((uint4 *)staged)[v] = __ldg((const uint4 *)src + v);
		__syncthreads();
	}
	if ( warp < MIXES_PER_GROUP )
	{
		const uint32_t mix = mg * MIXES_PER_GROUP + warp;
		float4 weight[S::kVectors];
#pragma unroll
		for ( uint32_t vector = 0u; vector < S::kVectors; ++vector )
			weight[vector] = __ldcs((const float4 *)(fn_f32 + (uint64_t)mix * S::kFlat + off) + lane + vector * LM_WARP_LANES);
		float accumulator = 0.0f;
#pragma unroll
		for ( uint32_t vector = 0u; vector < S::kVectors; ++vector )
		{
			const uint2 packed = ((const uint2 *)staged)[lane + vector * LM_WARP_LANES];
			accumulator = fmaf(weight[vector].x, __uint_as_float(packed.x << 16u), accumulator);
			accumulator = fmaf(weight[vector].y, __uint_as_float(packed.x & 0xffff0000u), accumulator);
			accumulator = fmaf(weight[vector].z, __uint_as_float(packed.y << 16u), accumulator);
			accumulator = fmaf(weight[vector].w, __uint_as_float(packed.y & 0xffff0000u), accumulator);
		}
		for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
			accumulator += __shfl_xor_sync(0xffffffffu, accumulator, step);
		if ( lane == 0u )
			partials[slice * (LM_HC_SITE_MIX + 1u) + mix] = accumulator;
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN, class P, bool FUSE_NORM>
__global__ __launch_bounds__(LM_HC_SITE_THREADS) void LmHcSiteB1FinishKernel(const float *__restrict__ partials, const uint16_t *__restrict__ streams_bf16,
	const float *__restrict__ scale3_f32, const float *__restrict__ base_f32, float *__restrict__ mixes_f32, float *__restrict__ pre_f32,
	float *__restrict__ post_f32, float *__restrict__ comb_f32, uint16_t *__restrict__ collapsed_bf16, uint16_t *__restrict__ snapshot_bf16,
	const uint16_t *__restrict__ norm_weight, uint16_t *__restrict__ normed_bf16)
{
	constexpr uint32_t SLOTS = HIDDEN / (LM_HC_SITE_THREADS * 8u);
	static_assert(HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u, "HC B1 finish covers whole 16-byte thread spans");
	__shared__ float mixes[LM_HC_SITE_MIX + 1u];
	__shared__ float pre[LM_HC_SITE_STREAMS];
	__shared__ float reduction[LM_HC_SITE_THREADS / LM_WARP_LANES];
	__shared__ __align__(16) float sq_buf[FUSE_NORM ? HIDDEN : 1u];
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	LmDependentWait();
	if ( threadIdx.x < LM_HC_SITE_MIX + 1u )
	{
		float total = 0.0f;
		for ( uint32_t rank = 0u; rank < LM_HC_SITE_CLUSTER; ++rank )
			total += partials[rank * (LM_HC_SITE_MIX + 1u) + threadIdx.x];
		mixes[threadIdx.x] = total;
	}
	__syncthreads();
	if ( warp == 0u )
	{
		const float inverse = SparkMathRmsScale(mixes[LM_HC_SITE_MIX], (float)LmHcSiteShape<HIDDEN>::kFlat, P::kNormEpsilon);
		if ( lane < LM_HC_SITE_MIX )
		{
			const float value = mixes[lane] * inverse;
			mixes[lane] = value;
			mixes_f32[lane] = value;
		}
		__syncwarp();
		P::Gates(mixes, scale3_f32, base_f32, pre, post_f32);
		__syncwarp();
		if ( lane < LM_HC_SITE_STREAMS )
			pre_f32[lane] = pre[lane];
	}
	__syncthreads();
	const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
	uint4 c_staged[SLOTS];
#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t off = (slot * LM_HC_SITE_THREADS + threadIdx.x) * 8u;
		const uint4 v0 = *(const uint4 *)(streams_bf16 + 0u * HIDDEN + off);
		const uint4 v1 = *(const uint4 *)(streams_bf16 + 1u * HIDDEN + off);
		const uint4 v2 = *(const uint4 *)(streams_bf16 + 2u * HIDDEN + off);
		const uint4 v3 = *(const uint4 *)(streams_bf16 + 3u * HIDDEN + off);
		*(uint4 *)(snapshot_bf16 + 0u * HIDDEN + off) = v0;
		*(uint4 *)(snapshot_bf16 + 1u * HIDDEN + off) = v1;
		*(uint4 *)(snapshot_bf16 + 2u * HIDDEN + off) = v2;
		*(uint4 *)(snapshot_bf16 + 3u * HIDDEN + off) = v3;
		float f0[8], f1[8], f2[8], f3[8], out[8];
		LmLoadBf16x8AsFloat(v0, f0);
		LmLoadBf16x8AsFloat(v1, f1);
		LmLoadBf16x8AsFloat(v2, f2);
		LmLoadBf16x8AsFloat(v3, f3);
#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = (((0.0f + p0 * f0[k]) + p1 * f1[k]) + p2 * f2[k]) + p3 * f3[k];
		const uint4 packed = LmPackUint4FloatToBf16(out);
		*(uint4 *)(collapsed_bf16 + off) = packed;
		if constexpr ( FUSE_NORM )
		{
			c_staged[slot] = packed;
			LmLoadBf16x8AsFloat(packed, out);
			float sq[8];
#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				sq[k] = out[k] * out[k];
			LmStoreFloat8(sq_buf + off, sq);
		}
	}
	if constexpr ( FUSE_NORM )
	{
		__syncthreads();
		float total_sq = 0.0f;
#pragma unroll
		for ( uint32_t slot = 0u; slot < HIDDEN / LM_HC_SITE_THREADS; ++slot )
			total_sq += sq_buf[threadIdx.x + slot * LM_HC_SITE_THREADS];
		const float rscale = SparkMathRmsScale(LmBlockSumOnce<LM_HC_SITE_THREADS>(total_sq, reduction), (float)HIDDEN, P::kNormEpsilon);
#pragma unroll
		for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
		{
			const uint32_t off = (slot * LM_HC_SITE_THREADS + threadIdx.x) * 8u;
			float vals[8], w[8], out[8];
			LmLoadBf16x8AsFloat(c_staged[slot], vals);
			LmLoadWeight8(norm_weight, off, w);
#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathNormGainRounded(vals[k] * rscale, w[k]);
			*(uint4 *)(normed_bf16 + off) = LmPackUint4FloatToBf16(out);
		}
	}
	LmDependentRelease();
	if ( warp == 0u )
		P::Comb(mixes, scale3_f32, base_f32, comb_f32);
}

template<uint32_t HIDDEN, class P, uint32_t ROWS>
static SparkGpuError LmHcSiteLaunch(const uint16_t *streams, const float *fn, const float *scale, const float *base, float *mixes, float *pre, float *post,
	float *comb, uint16_t *collapsed, uint16_t *snapshot, uint32_t rows, SparkGpuStream stream)
{
	return LM_LAUNCH_DEPENDENT((LmHcSiteKernel<HIDDEN, P, ROWS>), (rows + ROWS - 1u) / ROWS * LM_HC_SITE_CLUSTER, LM_HC_SITE_THREADS, 0, stream, streams,
		fn, scale, base, mixes, pre, post, comb, collapsed, snapshot, rows);
}

template<uint32_t HIDDEN, class P>
static SparkGpuError LmHcSite(const uint16_t *streams, const float *fn, const float *scale, const float *base, float *mixes, float *pre, float *post,
	float *comb, uint16_t *collapsed, uint16_t *snapshot, uint32_t rows, SparkGpuStream stream)
{
	if constexpr ( HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u && LM_HC_SITE_MIX % 6u == 0u )
		if ( rows == 1u && snapshot != streams && LmIsAligned16(streams) && LmIsAligned16(collapsed) && LmIsAligned16(snapshot) )
		{
			LM_LAUNCH_DEPENDENT((LmHcSiteB1DotKernel<HIDDEN, 6u>), dim3(LM_HC_SITE_CLUSTER, 6u), LM_HC_SITE_THREADS, 0, stream, streams, fn, (float *)snapshot);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				return SparkGpuPeekAtLastError();
			return LM_LAUNCH_DEPENDENT((LmHcSiteB1FinishKernel<HIDDEN, P, false>), 1u, LM_HC_SITE_THREADS, 0, stream, (const float *)snapshot, streams, scale,
				base, mixes, pre, post, comb, collapsed, snapshot, (const uint16_t *)0, (uint16_t *)0);
		}
	return (rows >= LM_HC_SITE_ROWS ? LmHcSiteLaunch<HIDDEN, P, LM_HC_SITE_ROWS> : LmHcSiteLaunch<HIDDEN, P, 1u>)(streams, fn, scale, base, mixes, pre, post,
		comb, collapsed, snapshot, rows, stream);
}

template<uint32_t HIDDEN, class P>
static SparkGpuError LmHcSiteNorm(const uint16_t *streams, const float *fn, const float *scale, const float *base, float *mixes, float *pre, float *post,
	float *comb, uint16_t *collapsed, uint16_t *snapshot, const uint16_t *norm_weight, uint16_t *normed, uint32_t rows, SparkGpuStream stream)
{
	if constexpr ( HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u && LM_HC_SITE_MIX % 6u == 0u )
		if ( rows == 1u && snapshot != streams && norm_weight != 0 && normed != 0 && LmIsAligned16(streams) && LmIsAligned16(collapsed) &&
			LmIsAligned16(snapshot) && LmIsAligned16(norm_weight) && LmIsAligned16(normed) )
		{
			LM_LAUNCH_DEPENDENT((LmHcSiteB1DotKernel<HIDDEN, 6u>), dim3(LM_HC_SITE_CLUSTER, 6u), LM_HC_SITE_THREADS, 0, stream, streams, fn, (float *)snapshot);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				return SparkGpuPeekAtLastError();
			return LM_LAUNCH_DEPENDENT((LmHcSiteB1FinishKernel<HIDDEN, P, true>), 1u, LM_HC_SITE_THREADS, 0, stream, (const float *)snapshot, streams, scale,
				base, mixes, pre, post, comb, collapsed, snapshot, norm_weight, normed);
		}
	const SparkGpuError err = LmHcSite<HIDDEN, P>(streams, fn, scale, base, mixes, pre, post, comb, collapsed, snapshot, rows, stream);
	if ( err != SPARK_GPU_SUCCESS )
		return err;
	LM_LAUNCH((LmBf16RmsNormKernel<LM_HC_SITE_THREADS>), rows, LM_HC_SITE_THREADS, (HIDDEN + 8u) * sizeof(float), stream, collapsed, norm_weight, normed,
		HIDDEN, HIDDEN, P::kNormEpsilon);
	return SparkGpuPeekAtLastError();
}

template<uint32_t HIDDEN>
__global__ void LmHcExpandStreamsKernel(const uint16_t *__restrict__ hidden, uint16_t *__restrict__ streams, uint32_t rows)
{
	const uint32_t row = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows || element >= HIDDEN )
		return;
	const uint16_t value = hidden[(uint64_t)row * HIDDEN + element];
	for ( uint32_t stream = 0u; stream < LM_HC_SITE_STREAMS; ++stream )
		streams[((uint64_t)row * LM_HC_SITE_STREAMS + stream) * HIDDEN + element] = value;
}

template<uint32_t HIDDEN>
__global__ void LmHcStreamsMeanKernel(const uint16_t *__restrict__ streams_bf16, uint16_t *__restrict__ reduced_bf16, uint32_t row_count)
{
	const uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	for ( uint32_t element = threadIdx.x; element < HIDDEN; element += blockDim.x )
	{
		float value = 0.0f;
		for ( uint32_t stream = 0u; stream < LM_HC_SITE_STREAMS; ++stream )
			value += LmBf16ToFloat(streams_bf16[((uint64_t)row * LM_HC_SITE_STREAMS + stream) * HIDDEN + element]);
		reduced_bf16[(uint64_t)row * HIDDEN + element] = LmFloatToBf16(value / (float)LM_HC_SITE_STREAMS);
	}
}
