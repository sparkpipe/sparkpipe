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

template<uint32_t HIDDEN, uint32_t MIX_GROUPS, bool FUSE_POST>
__global__ __launch_bounds__(LM_HC_SITE_THREADS) void LmHcSiteMultiDotKernel(const uint16_t *__restrict__ streams_in,
	const uint16_t *__restrict__ post_out, const float *__restrict__ prev_post, const float *__restrict__ prev_comb,
	uint16_t *__restrict__ streams_out, const float *__restrict__ fn_f32, float *__restrict__ partials, uint32_t rows)
{
	using S = LmHcSiteShape<HIDDEN>;
	constexpr uint32_t MIXES_PER_GROUP = LM_HC_SITE_MIX / MIX_GROUPS;
	constexpr uint32_t SLICE_SLOTS = S::kSlice / (LM_HC_SITE_THREADS * 8u);
	static_assert(LM_HC_SITE_MIX % MIX_GROUPS == 0u && MIXES_PER_GROUP <= LM_HC_SITE_THREADS / LM_WARP_LANES, "HC multi-dot mix groups divide mixes evenly");
	static_assert(S::kSlice % (LM_HC_SITE_THREADS * 8u) == 0u, "HC slice covers whole 16-byte thread spans");
	__shared__ __align__(16) uint16_t staged[S::kSlice];
	__shared__ float reduction[LM_HC_SITE_THREADS / LM_WARP_LANES];
	const uint32_t slice = blockIdx.x, mg = blockIdx.y, tile = blockIdx.z, tiles = gridDim.z;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t off = slice * S::kSlice;
	const uint32_t out_stream = slice >> 1u, col_base = (slice & 1u) * S::kSlice;
	const uint32_t row_begin = (rows * tile) / tiles, row_end = (rows * (tile + 1u)) / tiles;
	float4 weight[S::kVectors];
	if ( lane == 0u && warp < MIXES_PER_GROUP )
		LmPrefetchL2(fn_f32 + (uint64_t)(mg * MIXES_PER_GROUP + warp) * S::kFlat + off, S::kSlice * sizeof(float));
	if ( warp < MIXES_PER_GROUP )
	{
		const uint32_t mix = mg * MIXES_PER_GROUP + warp;
#pragma unroll
		for ( uint32_t vector = 0u; vector < S::kVectors; ++vector )
			weight[vector] = __ldcs((const float4 *)(fn_f32 + (uint64_t)mix * S::kFlat + off) + lane + vector * LM_WARP_LANES);
	}
	LmDependentWait();
	for ( uint32_t row = row_begin; row < row_end; ++row )
	{
		const uint16_t *src_row = streams_in + (uint64_t)row * S::kFlat;
		if constexpr ( FUSE_POST )
		{
			const float p_s = LmBf16ToFloat(LmFloatToBf16(prev_post[(uint64_t)row * LM_HC_SITE_STREAMS + out_stream]));
			const float c0 = LmBf16ToFloat(LmFloatToBf16(prev_comb[(uint64_t)row * 16u + 0u * LM_HC_SITE_STREAMS + out_stream]));
			const float c1 = LmBf16ToFloat(LmFloatToBf16(prev_comb[(uint64_t)row * 16u + 1u * LM_HC_SITE_STREAMS + out_stream]));
			const float c2 = LmBf16ToFloat(LmFloatToBf16(prev_comb[(uint64_t)row * 16u + 2u * LM_HC_SITE_STREAMS + out_stream]));
			const float c3 = LmBf16ToFloat(LmFloatToBf16(prev_comb[(uint64_t)row * 16u + 3u * LM_HC_SITE_STREAMS + out_stream]));
			const uint16_t *o_row = post_out + (uint64_t)row * HIDDEN;
			uint16_t *dst_slice = streams_out + (uint64_t)row * S::kFlat + off;
#pragma unroll
			for ( uint32_t slot = 0u; slot < SLICE_SLOTS; ++slot )
			{
				const uint32_t local = (slot * LM_HC_SITE_THREADS + threadIdx.x) * 8u;
				const uint32_t col = col_base + local;
				float fo[8], r0[8], r1[8], r2[8], r3[8], res[8];
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(o_row + col)), fo);
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(src_row + 0u * HIDDEN + col)), r0);
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(src_row + 1u * HIDDEN + col)), r1);
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(src_row + 2u * HIDDEN + col)), r2);
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(src_row + 3u * HIDDEN + col)), r3);
#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
				{
					float v = fmaf(c0, r0[k], 0.0f);
					v = fmaf(c1, r1[k], v);
					v = fmaf(c2, r2[k], v);
					v = fmaf(c3, r3[k], v);
					v = LmBf16ToFloat(LmFloatToBf16(v));
					const float contrib = LmBf16ToFloat(LmFloatToBf16(p_s * fo[k]));
					res[k] = contrib + v;
				}
				const uint4 packed = LmPackUint4FloatToBf16(res);
				*(uint4 *)(staged + local) = packed;
				if ( mg == 0u )
					*(uint4 *)(dst_slice + local) = packed;
			}
		}
		else
		{
			const uint16_t *src = src_row + off;
#pragma unroll
			for ( uint32_t slot = 0u; slot < SLICE_SLOTS; ++slot )
			{
				const uint32_t v = slot * LM_HC_SITE_THREADS + threadIdx.x;
				((uint4 *)staged)[v] = __ldg((const uint4 *)src + v);
			}
		}
		__syncthreads();
		float *row_partials = partials + ((uint64_t)row * LM_HC_SITE_CLUSTER + slice) * (LM_HC_SITE_MIX + 1u);
		if ( mg == 0u )
		{
			float total = 0.0f;
			for ( uint32_t element = threadIdx.x; element < S::kSlice; element += LM_HC_SITE_THREADS )
			{
				const float value = LmBf16ToFloat(staged[element]);
				total += value * value;
			}
			total = LmBlockSum<LM_HC_SITE_THREADS>(total, reduction);
			if ( threadIdx.x == 0u )
				row_partials[LM_HC_SITE_MIX] = total;
		}
		if ( warp < MIXES_PER_GROUP )
		{
			const uint32_t mix = mg * MIXES_PER_GROUP + warp;
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
				row_partials[mix] = accumulator;
		}
		if ( row + 1u < row_end )
			__syncthreads();
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN, class P, bool FUSE_NORM, bool WRITE_SNAPSHOT, bool WRITE_COLLAPSED>
__global__ __launch_bounds__(LM_HC_SITE_THREADS) void LmHcSiteMultiFinishKernel(const float *__restrict__ partials,
	const uint16_t *__restrict__ streams_bf16, const float *__restrict__ scale3_f32, const float *__restrict__ base_f32,
	float *__restrict__ mixes_f32, float *__restrict__ pre_f32, float *__restrict__ post_f32, float *__restrict__ comb_f32,
	uint16_t *__restrict__ collapsed_bf16, uint16_t *__restrict__ snapshot_bf16, const uint16_t *__restrict__ norm_weight,
	uint16_t *__restrict__ normed_bf16)
{
	using S = LmHcSiteShape<HIDDEN>;
	constexpr uint32_t SLOTS = HIDDEN / (LM_HC_SITE_THREADS * 8u);
	static_assert(HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u, "HC finish covers whole 16-byte thread spans");
	__shared__ float mixes[LM_HC_SITE_MIX + 1u];
	__shared__ float pre[LM_HC_SITE_STREAMS];
	__shared__ float reduction[LM_HC_SITE_THREADS / LM_WARP_LANES];
	__shared__ __align__(16) float sq_buf[FUSE_NORM ? HIDDEN : 1u];
	const uint32_t row = blockIdx.x, warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const float *row_partials = partials + (uint64_t)row * LM_HC_SITE_CLUSTER * (LM_HC_SITE_MIX + 1u);
	const uint16_t *row_streams = streams_bf16 + (uint64_t)row * S::kFlat;
	LmDependentWait();
	if ( threadIdx.x < LM_HC_SITE_MIX + 1u )
	{
		float total = 0.0f;
#pragma unroll
		for ( uint32_t rank = 0u; rank < LM_HC_SITE_CLUSTER; ++rank )
			total += row_partials[rank * (LM_HC_SITE_MIX + 1u) + threadIdx.x];
		mixes[threadIdx.x] = total;
	}
	__syncthreads();
	if ( warp == 0u )
	{
		const float inverse = SparkMathRmsScale(mixes[LM_HC_SITE_MIX], (float)S::kFlat, P::kNormEpsilon);
		if ( lane < LM_HC_SITE_MIX )
		{
			const float value = mixes[lane] * inverse;
			mixes[lane] = value;
			if ( mixes_f32 != 0 )
				mixes_f32[(uint64_t)row * LM_HC_SITE_MIX + lane] = value;
		}
		__syncwarp();
		P::Gates(mixes, scale3_f32, base_f32, pre, post_f32 + (uint64_t)row * LM_HC_SITE_STREAMS);
		__syncwarp();
		if ( pre_f32 != 0 && lane < LM_HC_SITE_STREAMS )
			pre_f32[(uint64_t)row * LM_HC_SITE_STREAMS + lane] = pre[lane];
	}
	__syncthreads();
	const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
	uint4 c_staged[SLOTS];
#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t off = (slot * LM_HC_SITE_THREADS + threadIdx.x) * 8u;
		const uint4 v0 = *(const uint4 *)(row_streams + 0u * HIDDEN + off);
		const uint4 v1 = *(const uint4 *)(row_streams + 1u * HIDDEN + off);
		const uint4 v2 = *(const uint4 *)(row_streams + 2u * HIDDEN + off);
		const uint4 v3 = *(const uint4 *)(row_streams + 3u * HIDDEN + off);
		if constexpr ( WRITE_SNAPSHOT )
		{
			uint16_t *row_snap = snapshot_bf16 + (uint64_t)row * S::kFlat;
			*(uint4 *)(row_snap + 0u * HIDDEN + off) = v0;
			*(uint4 *)(row_snap + 1u * HIDDEN + off) = v1;
			*(uint4 *)(row_snap + 2u * HIDDEN + off) = v2;
			*(uint4 *)(row_snap + 3u * HIDDEN + off) = v3;
		}
		float f0[8], f1[8], f2[8], f3[8], out[8];
		LmLoadBf16x8AsFloat(v0, f0);
		LmLoadBf16x8AsFloat(v1, f1);
		LmLoadBf16x8AsFloat(v2, f2);
		LmLoadBf16x8AsFloat(v3, f3);
#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = (((0.0f + p0 * f0[k]) + p1 * f1[k]) + p2 * f2[k]) + p3 * f3[k];
		const uint4 packed = LmPackUint4FloatToBf16(out);
		if constexpr ( WRITE_COLLAPSED )
			*(uint4 *)(collapsed_bf16 + (uint64_t)row * HIDDEN + off) = packed;
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
		uint16_t *row_normed = normed_bf16 + (uint64_t)row * HIDDEN;
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
			*(uint4 *)(row_normed + off) = LmPackUint4FloatToBf16(out);
		}
	}
	LmDependentRelease();
	if ( warp == 0u )
		P::Comb(mixes, scale3_f32, base_f32, comb_f32 + (uint64_t)row * LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS);
}

template<uint32_t HIDDEN>
__global__ __launch_bounds__(LM_HC_SITE_THREADS) void LmHcPostBf16VecKernel(const uint16_t *__restrict__ out_bf16,
	const uint16_t *__restrict__ src_streams, const float *__restrict__ post_f32, const float *__restrict__ comb_f32,
	uint16_t *__restrict__ dst_streams)
{
	using S = LmHcSiteShape<HIDDEN>;
	constexpr uint32_t SLOTS = HIDDEN / (LM_HC_SITE_THREADS * 8u);
	static_assert(HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u, "vectorized HC post covers whole 16-byte spans");
	const uint32_t row = blockIdx.x;
	float p[LM_HC_SITE_STREAMS], c[LM_HC_SITE_STREAMS][LM_HC_SITE_STREAMS];
	LmDependentWait();
#pragma unroll
	for ( uint32_t s = 0u; s < LM_HC_SITE_STREAMS; ++s )
		p[s] = LmBf16ToFloat(LmFloatToBf16(post_f32[(uint64_t)row * LM_HC_SITE_STREAMS + s]));
#pragma unroll
	for ( uint32_t src = 0u; src < LM_HC_SITE_STREAMS; ++src )
#pragma unroll
		for ( uint32_t s = 0u; s < LM_HC_SITE_STREAMS; ++s )
			c[src][s] = LmBf16ToFloat(LmFloatToBf16(comb_f32[(uint64_t)row * 16u + src * LM_HC_SITE_STREAMS + s]));
	const uint16_t *o_row = out_bf16 + (uint64_t)row * HIDDEN;
	const uint16_t *s_row = src_streams + (uint64_t)row * S::kFlat;
	uint16_t *d_row = dst_streams + (uint64_t)row * S::kFlat;
#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t off = (slot * LM_HC_SITE_THREADS + threadIdx.x) * 8u;
		float fo[8], r[LM_HC_SITE_STREAMS][8];
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(o_row + off)), fo);
#pragma unroll
		for ( uint32_t src = 0u; src < LM_HC_SITE_STREAMS; ++src )
			LmLoadBf16x8AsFloat(*(const uint4 *)(s_row + (uint64_t)src * HIDDEN + off), r[src]);
#pragma unroll
		for ( uint32_t s = 0u; s < LM_HC_SITE_STREAMS; ++s )
		{
			float out[8];
#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				float v = fmaf(c[0][s], r[0][k], 0.0f);
				v = fmaf(c[1][s], r[1][k], v);
				v = fmaf(c[2][s], r[2][k], v);
				v = fmaf(c[3][s], r[3][k], v);
				v = LmBf16ToFloat(LmFloatToBf16(v));
				out[k] = LmBf16ToFloat(LmFloatToBf16(p[s] * fo[k])) + v;
			}
			*(uint4 *)(d_row + (uint64_t)s * HIDDEN + off) = LmPackUint4FloatToBf16(out);
		}
	}
	LmDependentRelease();
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
			LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<HIDDEN, 6u, false>), dim3(LM_HC_SITE_CLUSTER, 6u, 1u), LM_HC_SITE_THREADS, 0, stream, streams,
				(const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, fn, (float *)snapshot, 1u);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				return SparkGpuPeekAtLastError();
			return LM_LAUNCH_DEPENDENT((LmHcSiteMultiFinishKernel<HIDDEN, P, false, true, true>), 1u, LM_HC_SITE_THREADS, 0, stream,
				(const float *)snapshot, streams, scale, base, mixes, pre, post, comb, collapsed, snapshot, (const uint16_t *)0, (uint16_t *)0);
		}
	return (rows >= LM_HC_SITE_ROWS ? LmHcSiteLaunch<HIDDEN, P, LM_HC_SITE_ROWS> : LmHcSiteLaunch<HIDDEN, P, 1u>)(streams, fn, scale, base, mixes, pre, post,
		comb, collapsed, snapshot, rows, stream);
}

template<uint32_t HIDDEN, class P>
static SparkGpuError LmHcSitePostNorm(const uint16_t *streams_in, const uint16_t *post_out, const float *prev_post, const float *prev_comb,
	uint16_t *streams_out, const float *fn, const float *scale, const float *base, float *partials, float *post, float *comb,
	const uint16_t *norm_weight, uint16_t *normed, uint32_t rows, SparkGpuStream stream)
{
	if constexpr ( HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u && LM_HC_SITE_MIX % 6u == 0u )
		if ( LmIsAligned16(streams_in) && LmIsAligned16(partials) && LmIsAligned16(norm_weight) && LmIsAligned16(normed) &&
			(post_out == 0 || (streams_out != streams_in && LmIsAligned16(post_out) && LmIsAligned16(streams_out))) )
		{
			const uint32_t row_tiles = rows >= 512u ? 8u : (rows >= 128u ? 4u : (rows >= 32u ? 2u : 1u));
			const uint16_t *active = post_out != 0 ? streams_out : streams_in;
			if ( post_out != 0 && rows < 32u )
			{
				if ( rows <= 4u )
					LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<HIDDEN, 6u, true>), dim3(LM_HC_SITE_CLUSTER, 6u, 1u), LM_HC_SITE_THREADS, 0, stream,
						streams_in, post_out, prev_post, prev_comb, streams_out, fn, partials, rows);
				else
					LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<HIDDEN, 3u, true>), dim3(LM_HC_SITE_CLUSTER, 3u, 1u), LM_HC_SITE_THREADS, 0, stream,
						streams_in, post_out, prev_post, prev_comb, streams_out, fn, partials, rows);
			}
			else
			{
				if ( post_out != 0 )
				{
					LM_LAUNCH_DEPENDENT((LmHcPostBf16VecKernel<HIDDEN>), rows, LM_HC_SITE_THREADS, 0, stream, post_out, streams_in, prev_post, prev_comb, streams_out);
					if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
						return SparkGpuPeekAtLastError();
				}
				if ( rows <= 4u )
					LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<HIDDEN, 6u, false>), dim3(LM_HC_SITE_CLUSTER, 6u, 1u), LM_HC_SITE_THREADS, 0, stream,
						active, (const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, fn, partials, rows);
				else
					LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<HIDDEN, 3u, false>), dim3(LM_HC_SITE_CLUSTER, 3u, row_tiles), LM_HC_SITE_THREADS, 0, stream,
						active, (const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, fn, partials, rows);
			}
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				return SparkGpuPeekAtLastError();
			return LM_LAUNCH_DEPENDENT((LmHcSiteMultiFinishKernel<HIDDEN, P, true, false, false>), rows, LM_HC_SITE_THREADS, 0, stream,
				partials, active, scale, base, (float *)0, (float *)0, post, comb, (uint16_t *)0, (uint16_t *)0, norm_weight, normed);
		}
	return SPARK_GPU_ERROR_INVALID_VALUE;
}

template<uint32_t HIDDEN, class P>
static SparkGpuError LmHcSiteNorm(const uint16_t *streams, const float *fn, const float *scale, const float *base, float *mixes, float *pre, float *post,
	float *comb, uint16_t *collapsed, uint16_t *snapshot, const uint16_t *norm_weight, uint16_t *normed, uint32_t rows, SparkGpuStream stream)
{
	if constexpr ( HIDDEN % (LM_HC_SITE_THREADS * 8u) == 0u && LM_HC_SITE_MIX % 6u == 0u )
		if ( rows == 1u && snapshot != streams && norm_weight != 0 && normed != 0 && LmIsAligned16(streams) && LmIsAligned16(collapsed) &&
			LmIsAligned16(snapshot) && LmIsAligned16(norm_weight) && LmIsAligned16(normed) )
		{
			LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<HIDDEN, 6u, false>), dim3(LM_HC_SITE_CLUSTER, 6u, 1u), LM_HC_SITE_THREADS, 0, stream, streams,
				(const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, fn, (float *)snapshot, 1u);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				return SparkGpuPeekAtLastError();
			return LM_LAUNCH_DEPENDENT((LmHcSiteMultiFinishKernel<HIDDEN, P, true, true, true>), 1u, LM_HC_SITE_THREADS, 0, stream,
				(const float *)snapshot, streams, scale, base, mixes, pre, post, comb, collapsed, snapshot, norm_weight, normed);
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
