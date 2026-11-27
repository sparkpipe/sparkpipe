#pragma once

#include <atomic>
#include <cstddef>
#include <stdint.h>
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_prefetch.cuh"
#include "model-families/k3/source/config.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"
#include "skl2_k3_embedded.h"

struct K3Skl2BundleTag {};

template<typename... Args>
static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Args... args)
{
	return LmLaunchEmbedded<K3Skl2BundleTag>(kK3Skl2Cubin, sizeof(kK3Skl2Cubin), (const char *)kK3Skl2Manifest,
		sizeof(kK3Skl2Manifest) - 1u, kK3Skl2Names, (uint32_t)K3SKL2_COUNT, id, grid, block, shared, stream, args...);
}

#define K3_ATTNRES_PERSISTENT_BLOCKS 96u

static __device__ __forceinline__ void K3CpAsync16L256(uint32_t shared, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" :: "r"(shared), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void K3CpAsync16L256Ptr(void *destination, const void *source, uint32_t bytes)
{
	K3CpAsync16L256((uint32_t)__cvta_generic_to_shared(destination), source, bytes);
}

static __device__ __forceinline__ void K3DecodeCellFrags(const uint32_t current[4], const uint32_t scale_bits[4], uint32_t b[4][4])
{
	#pragma unroll
	for ( uint32_t f = 0u; f < 4u; ++f )
	{
		const uint32_t code = scale_bits[f] & 255u;
		if ( code >= 2u && code <= 142u )
		{
			const uint32_t s16 = (code + 112u) << 7u;
			const uint32_t s2 = s16 | (s16 << 16u);
			#pragma unroll
			for ( uint32_t j = 0u; j < 4u; ++j )
				b[f][j] = LmCellMmaE2m1Bf16Fused(current[f], j, s2);
		}
		else
		{
			const float scale = LmUe8m0ToFloat((uint8_t)code);
			#pragma unroll
			for ( uint32_t j = 0u; j < 4u; ++j )
			{
				const float2 pair = LmSkinnyE2m1Pair(current[f], j);
				b[f][j] = LmCellMmaPackExact(pair.x * scale, pair.y * scale);
			}
		}
	}
}

template<uint32_t CHUNKS, uint32_t STAGE_ELEMENTS>
static __device__ __forceinline__ void K3CellMmaW1SituRows(const uint8_t *gate_base, const uint8_t *up_base, uint64_t tile_stride,
	const uint16_t *activation, const uint32_t *route_source_token, uint16_t *output_bf16,
	uint32_t k_tiles, uint32_t input_dim, uint32_t channels, uint32_t g_cell_base,
	uint32_t first, uint32_t end, uint16_t *stage, uint8_t *weights)
{
	constexpr uint32_t COPIES = (CHUNKS * 16u + LM_CELL_MMA_STAGE_ROW_STEP - 1u) / LM_CELL_MMA_STAGE_ROW_STEP;
	constexpr uint32_t HALF_VECS = 4u * LM_CELL_MMA_CELL_VECTORS;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t g = lane / 4u, c = lane % 4u;
	const uint32_t vector = threadIdx.x % LM_CELL_MMA_STAGE_VECTORS, row = threadIdx.x / LM_CELL_MMA_STAGE_VECTORS;
	float acc[CHUNKS][4][4];
	const uint16_t *source[COPIES];
	#pragma unroll
	for ( uint32_t m = 0u; m < CHUNKS; ++m )
		#pragma unroll
		for ( uint32_t f = 0u; f < 4u; ++f )
			acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
	#pragma unroll
	for ( uint32_t m = 0u; m < COPIES; ++m )
	{
		const uint32_t local = row + m * LM_CELL_MMA_STAGE_ROW_STEP;
		const uint32_t packed = first + local;
		source[m] = packed < end && local < CHUNKS * 16u ?
			activation + (uint64_t)route_source_token[packed] * input_dim + vector * 8u : (const uint16_t *)0;
	}
	const uint8_t *g_ptr = gate_base, *u_ptr = up_base;
	auto load_a = [&](uint32_t s_slot)
	{
		uint16_t *s_buf = stage + s_slot * STAGE_ELEMENTS;
		#pragma unroll
		for ( uint32_t m = 0u; m < COPIES; ++m )
		{
			const uint32_t local = row + m * LM_CELL_MMA_STAGE_ROW_STEP;
			if ( local < CHUNKS * 16u )
			{
				LmCellMmaCopy(s_buf + local * LM_CELL_MMA_STAGE_PITCH + vector * 8u,
					source[m] != 0 ? source[m] : activation, source[m] != 0 ? 16u : 0u);
				if ( source[m] != 0 )
					source[m] += LM_CELL_MMA_STAGE_K;
			}
		}
		LmCellMmaCommit();
	};
	uint4 w_reg0, w_reg1;
	auto pref_w = [&]()
	{
		w_reg0 = __ldcs((const uint4 *)(threadIdx.x < HALF_VECS ? g_ptr + threadIdx.x * 16u : u_ptr + (threadIdx.x - HALF_VECS) * 16u));
		if ( threadIdx.x + LM_CELL_MMA_THREADS < 2u * HALF_VECS )
			w_reg1 = __ldcs((const uint4 *)(u_ptr + (threadIdx.x + LM_CELL_MMA_THREADS - HALF_VECS) * 16u));
		g_ptr += tile_stride;
		u_ptr += tile_stride;
	};
	auto store_w = [&](uint32_t s_slot)
	{
		uint8_t *w_buf = weights + s_slot * LM_CELL_MMA_WEIGHT_TILE_BYTES;
		*(uint4 *)(w_buf + threadIdx.x * 16u) = w_reg0;
		if ( threadIdx.x + LM_CELL_MMA_THREADS < 2u * HALF_VECS )
			*(uint4 *)(w_buf + (threadIdx.x + LM_CELL_MMA_THREADS) * 16u) = w_reg1;
	};
	if ( k_tiles != 0u )
	{
		load_a(0u);
		pref_w();
		store_w(0u);
	}
	uint32_t sc = 0u, sp = 1u;
	for ( uint32_t t = 0u; t < k_tiles; ++t )
	{
		const uint16_t *buffer = stage + sc * STAGE_ELEMENTS;
		const uint8_t *w_slot = weights + sc * LM_CELL_MMA_WEIGHT_TILE_BYTES;
		LmCellMmaWait<0u>();
		__syncthreads();
		if ( t + 1u < k_tiles )
		{
			load_a(sp);
			pref_w();
		}
		uint32_t current[4], scale_bits[4], b[4][4];
		const uint8_t *g_cell = w_slot + warp * LM_CELL_MMA_CELL_BYTES;
		const uint8_t *u_cell = w_slot + (4u + warp) * LM_CELL_MMA_CELL_BYTES;
		#pragma unroll
		for ( uint32_t h = 0u; h < 2u; ++h )
		{
			current[h] = ((const uint32_t *)(g_cell + h * 8u * LM_SKINNY_CHUNK_BYTES))[lane];
			scale_bits[h] = g_cell[LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + h * 8u + g];
			current[2u + h] = ((const uint32_t *)(u_cell + h * 8u * LM_SKINNY_CHUNK_BYTES))[lane];
			scale_bits[2u + h] = u_cell[LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + h * 8u + g];
		}
		K3DecodeCellFrags(current, scale_bits, b);
		uint4 low = *(const uint4 *)(buffer + g * LM_CELL_MMA_STAGE_PITCH + c * 8u);
		uint4 high = *(const uint4 *)(buffer + (g + 8u) * LM_CELL_MMA_STAGE_PITCH + c * 8u);
		#pragma unroll
		for ( uint32_t m = 0u; m < CHUNKS; ++m )
		{
			uint4 next_low, next_high;
			if ( m + 1u < CHUNKS )
			{
				const uint16_t *next_row = buffer + ((m + 1u) * 16u + g) * LM_CELL_MMA_STAGE_PITCH + c * 8u;
				next_low = *(const uint4 *)next_row;
				next_high = *(const uint4 *)(next_row + 8u * LM_CELL_MMA_STAGE_PITCH);
			}
			const uint32_t a0[4] = {low.x, high.x, low.y, high.y};
			const uint32_t a1[4] = {low.z, high.z, low.w, high.w};
			#pragma unroll
			for ( uint32_t f = 0u; f < 4u; ++f )
			{
				const uint32_t b0[2] = {b[f][0], b[f][1]};
				const uint32_t b1[2] = {b[f][2], b[f][3]};
				LmMmaBf16(acc[m][f], a0, b0);
				LmMmaBf16(acc[m][f], a1, b1);
			}
			if ( m + 1u < CHUNKS ) { low = next_low; high = next_high; }
		}
		if ( t + 1u < k_tiles )
			store_w(sp);
		sc ^= 1u;
		sp ^= 1u;
	}
	LmCellMmaWait<0u>();
	__syncthreads();
	#pragma unroll
	for ( uint32_t m = 0u; m < CHUNKS; ++m )
	{
		#pragma unroll
		for ( uint32_t h = 0u; h < 2u; ++h )
		{
			const uint32_t packed = first + m * 16u + g + h * 8u;
			if ( packed >= end )
				continue;
			#pragma unroll
			for ( uint32_t half = 0u; half < 2u; ++half )
			{
				const uint32_t g_bf = LmPackBf16Pair(acc[m][half][h * 2u], acc[m][half][h * 2u + 1u]);
				const uint32_t u_bf = LmPackBf16Pair(acc[m][2u + half][h * 2u], acc[m][2u + half][h * 2u + 1u]);
				const float out0 = SparkMathSitu(LmBf16ToFloat((uint16_t)(g_bf & 0xffffu)),
					LmBf16ToFloat((uint16_t)(u_bf & 0xffffu)), K3_SITU_BETA, K3_SITU_LINEAR_BETA);
				const float out1 = SparkMathSitu(LmBf16ToFloat((uint16_t)(g_bf >> 16u)),
					LmBf16ToFloat((uint16_t)(u_bf >> 16u)), K3_SITU_BETA, K3_SITU_LINEAR_BETA);
				*(uint32_t *)(output_bf16 + (uint64_t)packed * channels + (g_cell_base + warp) * LM_SKINNY_CELL_NEURONS + half * 8u + c * 2u) =
					LmPackBf16Pair(out0, out1);
			}
		}
	}
}

template<uint32_t MAX_CHUNKS>
__global__ __launch_bounds__(LM_CELL_MMA_THREADS)
static void K3CellMmaW1SituKernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ active_experts,
	uint64_t group_bytes, uint32_t k_tiles, uint32_t half_cells, uint32_t input_dim, uint32_t channels)
{
	constexpr uint32_t MAX_ROWS = MAX_CHUNKS * 16u;
	constexpr uint32_t STAGE_ELEMENTS = MAX_ROWS * LM_CELL_MMA_STAGE_PITCH;
	__shared__ __align__(16) uint16_t stage[2u * STAGE_ELEMENTS];
	__shared__ __align__(16) uint8_t weights[2u * LM_CELL_MMA_WEIGHT_TILE_BYTES];
	LmDependentWait();
	const uint32_t col_blocks = half_cells / 4u;
	const uint32_t active_idx = blockIdx.x / col_blocks;
	const uint32_t n_block = blockIdx.x % col_blocks;
	if ( active_idx >= active_experts[0u] )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t group = active_experts[1u + active_idx];
	const uint32_t start = group_row_offset[group], end = group_row_offset[group + 1u];
	if ( start >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t g_cell_base = n_block * 4u, u_cell_base = half_cells + g_cell_base;
	const uint64_t tile_stride = (uint64_t)(half_cells * 2u) * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *group_ptr = weight + (uint64_t)group * group_bytes;
	const uint8_t *gate_base = group_ptr + (uint64_t)g_cell_base * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *up_base = group_ptr + (uint64_t)u_cell_base * LM_CELL_MMA_CELL_BYTES;
	for ( uint32_t first = start; first < end; first += MAX_ROWS )
	{
		const uint32_t last = end - first < MAX_ROWS ? end : first + MAX_ROWS;
		if constexpr ( MAX_CHUNKS >= 8u )
		{
			if ( last - first > 6u * 16u )
			{
				K3CellMmaW1SituRows<8u, STAGE_ELEMENTS>(gate_base, up_base, tile_stride, activation, route_source_token,
					output_bf16, k_tiles, input_dim, channels, g_cell_base, first, last, stage, weights);
				continue;
			}
		}
		if constexpr ( MAX_CHUNKS >= 6u )
		{
			if ( last - first > 4u * 16u )
			{
				K3CellMmaW1SituRows<6u, STAGE_ELEMENTS>(gate_base, up_base, tile_stride, activation, route_source_token,
					output_bf16, k_tiles, input_dim, channels, g_cell_base, first, last, stage, weights);
				continue;
			}
		}
		if constexpr ( MAX_CHUNKS >= 4u )
		{
			if ( last - first > 3u * 16u )
			{
				K3CellMmaW1SituRows<4u, STAGE_ELEMENTS>(gate_base, up_base, tile_stride, activation, route_source_token,
					output_bf16, k_tiles, input_dim, channels, g_cell_base, first, last, stage, weights);
				continue;
			}
			if ( last - first > 2u * 16u )
			{
				K3CellMmaW1SituRows<3u, STAGE_ELEMENTS>(gate_base, up_base, tile_stride, activation, route_source_token,
					output_bf16, k_tiles, input_dim, channels, g_cell_base, first, last, stage, weights);
				continue;
			}
		}
		if ( last - first > 16u )
			K3CellMmaW1SituRows<2u, STAGE_ELEMENTS>(gate_base, up_base, tile_stride, activation, route_source_token,
				output_bf16, k_tiles, input_dim, channels, g_cell_base, first, last, stage, weights);
		else
			K3CellMmaW1SituRows<1u, STAGE_ELEMENTS>(gate_base, up_base, tile_stride, activation, route_source_token,
				output_bf16, k_tiles, input_dim, channels, g_cell_base, first, last, stage, weights);
	}
	LmDependentRelease();
}

template<uint32_t CHUNKS, uint32_t STAGE_ELEMENTS, uint32_t STAGES, bool STREAM_OUT = false>
static __device__ __forceinline__ void K3CellMmaW2SliceRows(const uint8_t *cells_base, uint64_t tile_stride,
	const uint16_t *activation, uint16_t *output_bf16, uint32_t k_tiles, uint32_t input_dim,
	uint32_t slice_cols, uint32_t out_cell_local, uint32_t first, uint32_t end,
	uint16_t *stage, uint8_t *weights)
{
	constexpr uint32_t COPIES = (CHUNKS * 16u + LM_CELL_MMA_STAGE_ROW_STEP - 1u) / LM_CELL_MMA_STAGE_ROW_STEP;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t g = lane / 4u, c = lane % 4u;
	const uint32_t vector = threadIdx.x % LM_CELL_MMA_STAGE_VECTORS, row = threadIdx.x / LM_CELL_MMA_STAGE_VECTORS;
	const uint32_t warp_offset = warp * LM_CELL_MMA_CELLS * LM_CELL_MMA_CELL_BYTES;
	float acc[CHUNKS][4][4];
	const uint16_t *source[COPIES];
	#pragma unroll
	for ( uint32_t m = 0u; m < CHUNKS; ++m )
		#pragma unroll
		for ( uint32_t f = 0u; f < 4u; ++f )
			acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
	#pragma unroll
	for ( uint32_t m = 0u; m < COPIES; ++m )
	{
		const uint32_t local = row + m * LM_CELL_MMA_STAGE_ROW_STEP;
		const uint32_t packed = first + local;
		source[m] = packed < end && local < CHUNKS * 16u ?
			activation + (uint64_t)packed * input_dim + vector * 8u : (const uint16_t *)0;
	}
	const uint8_t *c_ptr = cells_base;
	if constexpr ( STAGES == 2u )
	{
		auto load_a = [&](uint32_t s_slot)
		{
			uint16_t *s_buf = stage + s_slot * STAGE_ELEMENTS;
			#pragma unroll
			for ( uint32_t m = 0u; m < COPIES; ++m )
			{
				const uint32_t local = row + m * LM_CELL_MMA_STAGE_ROW_STEP;
				if ( local < CHUNKS * 16u )
				{
					LmCellMmaCopy(s_buf + local * LM_CELL_MMA_STAGE_PITCH + vector * 8u,
						source[m] != 0 ? source[m] : activation, source[m] != 0 ? 16u : 0u);
					if ( source[m] != 0 )
						source[m] += LM_CELL_MMA_STAGE_K;
				}
			}
			LmCellMmaCommit();
		};
		uint4 w_reg0, w_reg1;
		auto pref_w = [&]()
		{
			w_reg0 = __ldcs((const uint4 *)(c_ptr + threadIdx.x * 16u));
			if ( threadIdx.x + LM_CELL_MMA_THREADS < LM_CELL_MMA_WEIGHT_TILE_VECTORS )
				w_reg1 = __ldcs((const uint4 *)(c_ptr + (threadIdx.x + LM_CELL_MMA_THREADS) * 16u));
			c_ptr += tile_stride;
		};
		auto store_w = [&](uint32_t s_slot)
		{
			uint8_t *w_buf = weights + s_slot * LM_CELL_MMA_WEIGHT_TILE_BYTES;
			*(uint4 *)(w_buf + threadIdx.x * 16u) = w_reg0;
			if ( threadIdx.x + LM_CELL_MMA_THREADS < LM_CELL_MMA_WEIGHT_TILE_VECTORS )
				*(uint4 *)(w_buf + (threadIdx.x + LM_CELL_MMA_THREADS) * 16u) = w_reg1;
		};
		if ( k_tiles != 0u )
		{
			load_a(0u);
			pref_w();
			store_w(0u);
		}
		uint32_t sc = 0u, sp = 1u;
		for ( uint32_t t = 0u; t < k_tiles; ++t )
		{
			const uint16_t *buffer = stage + sc * STAGE_ELEMENTS;
			LmCellMmaWait<0u>();
			__syncthreads();
			if ( t + 1u < k_tiles )
			{
				load_a(sp);
				pref_w();
			}
			uint32_t current[4], scale_bits[4], b[4][4];
			LmCellMmaLoad(weights + sc * LM_CELL_MMA_WEIGHT_TILE_BYTES + warp_offset, lane, g, current, scale_bits);
			K3DecodeCellFrags(current, scale_bits, b);
			uint4 low = *(const uint4 *)(buffer + g * LM_CELL_MMA_STAGE_PITCH + c * 8u);
			uint4 high = *(const uint4 *)(buffer + (g + 8u) * LM_CELL_MMA_STAGE_PITCH + c * 8u);
			#pragma unroll
			for ( uint32_t m = 0u; m < CHUNKS; ++m )
			{
				uint4 next_low, next_high;
				if ( m + 1u < CHUNKS )
				{
					const uint16_t *next_row = buffer + ((m + 1u) * 16u + g) * LM_CELL_MMA_STAGE_PITCH + c * 8u;
					next_low = *(const uint4 *)next_row;
					next_high = *(const uint4 *)(next_row + 8u * LM_CELL_MMA_STAGE_PITCH);
				}
				const uint32_t a0[4] = {low.x, high.x, low.y, high.y};
				const uint32_t a1[4] = {low.z, high.z, low.w, high.w};
				#pragma unroll
				for ( uint32_t f = 0u; f < 4u; ++f )
				{
					const uint32_t b0[2] = {b[f][0], b[f][1]};
					const uint32_t b1[2] = {b[f][2], b[f][3]};
					LmMmaBf16(acc[m][f], a0, b0);
					LmMmaBf16(acc[m][f], a1, b1);
				}
				if ( m + 1u < CHUNKS ) { low = next_low; high = next_high; }
			}
			if ( t + 1u < k_tiles )
				store_w(sp);
			sc ^= 1u;
			sp ^= 1u;
		}
		LmCellMmaWait<0u>();
		__syncthreads();
	}
	else
	{
	auto load_stage = [&](uint32_t s_slot)
	{
		uint16_t *s_buf = stage + s_slot * STAGE_ELEMENTS;
		uint8_t *w_buf = weights + s_slot * LM_CELL_MMA_WEIGHT_TILE_BYTES;
		#pragma unroll
		for ( uint32_t m = 0u; m < COPIES; ++m )
		{
			const uint32_t local = row + m * LM_CELL_MMA_STAGE_ROW_STEP;
			if ( local < CHUNKS * 16u )
			{
				LmCellMmaCopy(s_buf + local * LM_CELL_MMA_STAGE_PITCH + vector * 8u,
					source[m] != 0 ? source[m] : activation, source[m] != 0 ? 16u : 0u);
				if ( source[m] != 0 )
					source[m] += LM_CELL_MMA_STAGE_K;
			}
		}
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < LM_CELL_MMA_WEIGHT_TILE_VECTORS; idx += LM_CELL_MMA_THREADS )
			K3CpAsync16L256Ptr(w_buf + idx * 16u, c_ptr + idx * 16u, 16u);
		c_ptr += tile_stride;
	};
	#pragma unroll
	for ( uint32_t t = 0u; t + 1u < STAGES; ++t )
	{
		if ( t < k_tiles )
			load_stage(t);
		LmCellMmaCommit();
	}
	for ( uint32_t t = 0u; t < k_tiles; ++t )
	{
		const uint32_t slot = t % STAGES, ahead = t + STAGES - 1u;
		const uint16_t *buffer = stage + slot * STAGE_ELEMENTS;
		LmCellMmaWait<STAGES - 2u>();
		__syncthreads();
		if ( ahead < k_tiles )
			load_stage(ahead % STAGES);
		LmCellMmaCommit();
		uint32_t current[4], scale_bits[4], b[4][4];
		LmCellMmaLoad(weights + slot * LM_CELL_MMA_WEIGHT_TILE_BYTES + warp_offset, lane, g, current, scale_bits);
		K3DecodeCellFrags(current, scale_bits, b);
		uint4 low = *(const uint4 *)(buffer + g * LM_CELL_MMA_STAGE_PITCH + c * 8u);
		uint4 high = *(const uint4 *)(buffer + (g + 8u) * LM_CELL_MMA_STAGE_PITCH + c * 8u);
		#pragma unroll
		for ( uint32_t m = 0u; m < CHUNKS; ++m )
		{
			uint4 next_low, next_high;
			if ( m + 1u < CHUNKS )
			{
				const uint16_t *next_row = buffer + ((m + 1u) * 16u + g) * LM_CELL_MMA_STAGE_PITCH + c * 8u;
				next_low = *(const uint4 *)next_row;
				next_high = *(const uint4 *)(next_row + 8u * LM_CELL_MMA_STAGE_PITCH);
			}
			const uint32_t a0[4] = {low.x, high.x, low.y, high.y};
			const uint32_t a1[4] = {low.z, high.z, low.w, high.w};
			#pragma unroll
			for ( uint32_t f = 0u; f < 4u; ++f )
			{
				const uint32_t b0[2] = {b[f][0], b[f][1]};
				const uint32_t b1[2] = {b[f][2], b[f][3]};
				LmMmaBf16(acc[m][f], a0, b0);
				LmMmaBf16(acc[m][f], a1, b1);
			}
			if ( m + 1u < CHUNKS ) { low = next_low; high = next_high; }
		}
	}
	LmCellMmaWait<0u>();
	__syncthreads();
	}
	#pragma unroll
	for ( uint32_t m = 0u; m < CHUNKS; ++m )
	{
		#pragma unroll
		for ( uint32_t h = 0u; h < 2u; ++h )
		{
			const uint32_t packed = first + m * 16u + g + h * 8u;
			if ( packed >= end )
				continue;
			#pragma unroll
			for ( uint32_t f = 0u; f < 4u; ++f )
			{
				uint32_t *dst = (uint32_t *)(output_bf16 + (uint64_t)packed * slice_cols + (out_cell_local + f / 2u) * LM_SKINNY_CELL_NEURONS + (f % 2u) * 8u + c * 2u);
				const uint32_t pair = LmPackBf16Pair(acc[m][f][h * 2u], acc[m][f][h * 2u + 1u]);
				if constexpr ( STREAM_OUT )
					__stcs(dst, pair);
				else
					*dst = pair;
			}
		}
	}
}

template<uint32_t MAX_CHUNKS, uint32_t STAGES, bool STREAM_OUT = false>
__global__ __launch_bounds__(LM_CELL_MMA_THREADS)
static void K3CellMmaW2SliceKernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ active_experts, uint64_t group_bytes, uint32_t k_tiles,
	uint32_t total_cells, uint32_t cell_begin, uint32_t slice_cells, uint32_t input_dim)
{
	constexpr uint32_t MAX_ROWS = MAX_CHUNKS * 16u;
	constexpr uint32_t STAGE_ELEMENTS = MAX_ROWS * LM_CELL_MMA_STAGE_PITCH;
	__shared__ __align__(16) uint16_t stage[STAGES * STAGE_ELEMENTS];
	__shared__ __align__(16) uint8_t weights[STAGES * LM_CELL_MMA_WEIGHT_TILE_BYTES];
	LmDependentWait();
	const uint32_t col_blocks = slice_cells / LM_CELL_MMA_BLOCK_CELLS;
	const uint32_t active_idx = blockIdx.x / col_blocks;
	const uint32_t n_block = blockIdx.x % col_blocks;
	if ( active_idx >= active_experts[0u] )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t group = active_experts[1u + active_idx];
	const uint32_t start = group_row_offset[group], end = group_row_offset[group + 1u];
	if ( start >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t local_block_cell = n_block * LM_CELL_MMA_BLOCK_CELLS;
	const uint32_t out_cell_local = local_block_cell + warp * LM_CELL_MMA_CELLS;
	const uint32_t slice_cols = slice_cells * LM_SKINNY_CELL_NEURONS;
	const uint64_t tile_stride = (uint64_t)total_cells * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *cells_base = weight + (uint64_t)group * group_bytes + (uint64_t)(cell_begin + local_block_cell) * LM_CELL_MMA_CELL_BYTES;
	for ( uint32_t first = start; first < end; first += MAX_ROWS )
	{
		const uint32_t last = end - first < MAX_ROWS ? end : first + MAX_ROWS;
		if constexpr ( MAX_CHUNKS >= 8u )
		{
			if ( last - first > 6u * 16u )
			{
				K3CellMmaW2SliceRows<8u, STAGE_ELEMENTS, STAGES, STREAM_OUT>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
		}
		if constexpr ( MAX_CHUNKS >= 6u )
		{
			if ( last - first > 4u * 16u )
			{
				K3CellMmaW2SliceRows<6u, STAGE_ELEMENTS, STAGES, STREAM_OUT>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
		}
		if constexpr ( MAX_CHUNKS >= 4u )
		{
			if ( last - first > 3u * 16u )
			{
				K3CellMmaW2SliceRows<4u, STAGE_ELEMENTS, STAGES, STREAM_OUT>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
			if ( last - first > 2u * 16u )
			{
				K3CellMmaW2SliceRows<3u, STAGE_ELEMENTS, STAGES, STREAM_OUT>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
		}
		if ( last - first > 16u )
			K3CellMmaW2SliceRows<2u, STAGE_ELEMENTS, STAGES, STREAM_OUT>(cells_base, tile_stride, activation, output_bf16,
				k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
		else
			K3CellMmaW2SliceRows<1u, STAGE_ELEMENTS, STAGES, STREAM_OUT>(cells_base, tile_stride, activation, output_bf16,
				k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
	}
	LmDependentRelease();
}

static int32_t K3MoeCellsMmaFastLaunch(const void *w1_weight, const void *w2_weight,
	const uint16_t *latent_full_bf16, uint16_t *gate_up_bf16, uint16_t *intermediate_bf16, uint16_t *latent_bf16,
	const uint32_t *group_row_offset, const uint32_t *route_source_token, const uint32_t *route_packed_row,
	const float *route_weight, const uint32_t *active_experts, uint32_t rows, uint32_t packed_rows,
	uint32_t channels, SparkGpuStream stream)
{
	const uint32_t w1_k_tiles = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_TILE_K;
	const uint32_t half_cells = channels / LM_SKINNY_CELL_NEURONS;
	const uint32_t w1_col_blocks = half_cells / 4u;
	const uint32_t max_active = packed_rows < K3_EXPERTS ? packed_rows : K3_EXPERTS;
	const uint64_t w1_group_bytes = (uint64_t)w1_k_tiles * (half_cells * 2u) * LM_CELL_MMA_CELL_BYTES;
	if ( packed_rows <= 1024u )
		LM_LAUNCH_DEPENDENT((K3CellMmaW1SituKernel<2u>), max_active * w1_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w1_weight, latent_full_bf16, intermediate_bf16, group_row_offset, route_source_token,
			active_experts, w1_group_bytes, w1_k_tiles, half_cells, K3_ROUTED_EXPERT_HIDDEN, channels);
	else
	{
		static std::atomic<bool> w1_carveout_set(false);
		if ( !w1_carveout_set.load(std::memory_order_acquire) )
		{
			(void)SparkGpuFuncSetAttribute((const void *)K3CellMmaW1SituKernel<4u>,
				SPARK_GPU_FUNCTION_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
			w1_carveout_set.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((K3CellMmaW1SituKernel<4u>), max_active * w1_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w1_weight, latent_full_bf16, intermediate_bf16, group_row_offset, route_source_token,
			active_experts, w1_group_bytes, w1_k_tiles, half_cells, K3_ROUTED_EXPERT_HIDDEN, channels);
	}
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w2_k_tiles = channels / LM_SKINNY_CELL_TILE_K;
	const uint32_t w2_total_cells = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_NEURONS;
	const uint64_t w2_group_bytes = (uint64_t)w2_k_tiles * w2_total_cells * LM_CELL_MMA_CELL_BYTES;
	const uint64_t full_out_bytes = (uint64_t)packed_rows * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t);
	uint32_t slice_cells = w2_total_cells;
	if ( full_out_bytes > 16u * 1024u * 1024u )
	{
		const uint64_t target_bytes = 10u * 1024u * 1024u;
		uint32_t raw_cells = (uint32_t)(target_bytes / ((uint64_t)packed_rows * LM_SKINNY_CELL_NEURONS * sizeof(uint16_t)));
		raw_cells = (raw_cells / LM_CELL_MMA_BLOCK_CELLS) * LM_CELL_MMA_BLOCK_CELLS;
		if ( raw_cells < LM_CELL_MMA_BLOCK_CELLS )
			raw_cells = LM_CELL_MMA_BLOCK_CELLS;
		if ( raw_cells > w2_total_cells )
			raw_cells = w2_total_cells;
		slice_cells = raw_cells;
	}
	if ( slice_cells == w2_total_cells )
	{
		const uint32_t w2_col_blocks = w2_total_cells / LM_CELL_MMA_BLOCK_CELLS;
		if ( packed_rows <= 2048u )
			LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<2u, 2u, false>), max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
				(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
				w2_group_bytes, w2_k_tiles, w2_total_cells, 0u, w2_total_cells, channels);
		else
			LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<4u, 2u, true>), max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
				(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
				w2_group_bytes, w2_k_tiles, w2_total_cells, 0u, w2_total_cells, channels);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_moe_finalize_token224_224, rows, 224u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows, (uint32_t)K3_ROUTED_EXPERT_HIDDEN);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	for ( uint32_t cell_begin = 0u; cell_begin < w2_total_cells; cell_begin += slice_cells )
	{
		const uint32_t cur_cells = w2_total_cells - cell_begin < slice_cells ? w2_total_cells - cell_begin : slice_cells;
		const uint32_t cur_col_blocks = cur_cells / LM_CELL_MMA_BLOCK_CELLS;
		const uint32_t cur_cols = cur_cells * LM_SKINNY_CELL_NEURONS;
		LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<4u, 2u, false>), max_active * cur_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
			w2_group_bytes, w2_k_tiles, w2_total_cells, cell_begin, cur_cells, channels);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		if ( K3Skl2Launch(K3SKL2_k3_moe_finalize_warp_slice_128, (rows + 3u) / 4u, 128u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows,
			cell_begin * LM_SKINNY_CELL_NEURONS, cur_cols, (uint32_t)K3_ROUTED_EXPERT_HIDDEN) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	return LM_LAUNCH_OK;
}

template<uint32_t ROWS>
static __device__ __forceinline__ void K3CellSkinnyW1SituRows(const uint8_t *base, uint64_t tile_stride,
	const uint16_t *activation, const uint32_t *route_source_token, uint16_t *output_bf16,
	uint32_t k_tiles, uint32_t input_dim, uint32_t channels, uint32_t hcell,
	uint32_t warp, uint32_t lane, uint32_t sub, uint32_t is_up, uint32_t first, uint32_t more, float *partial)
{
	constexpr uint32_t WARPS = 4u;
	const uint32_t t0 = k_tiles * warp / WARPS, t1 = k_tiles * (warp + 1u) / WARPS;
	float acc[ROWS];
	const uint16_t *act[ROWS];
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
	{
		acc[r] = 0.0f;
		act[r] = activation + (uint64_t)__ldg(route_source_token + first + r) * input_dim;
	}
	uint4 w0, w1;
	uint8_t s0 = 0u, s1 = 0u;
	if ( t0 < t1 )
	{
		const uint8_t *tile = base + (uint64_t)t0 * tile_stride;
		w0 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
		s0 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
	}
	if ( t0 + 1u < t1 )
	{
		const uint8_t *tile = base + (uint64_t)(t0 + 1u) * tile_stride;
		w1 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
		s1 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
	}
	for ( uint32_t t = t0; t < t1; ++t )
	{
		uint4 w2;
		uint8_t s2 = 0u;
		if ( t + 2u < t1 )
		{
			const uint8_t *tile = base + (uint64_t)(t + 2u) * tile_stride;
			w2 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
			s2 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		LmSkinnyCellTile<ROWS>(w0, s0, act, t * LM_SKINNY_CELL_TILE_K, acc);
		w0 = w1;
		s0 = s1;
		w1 = w2;
		s1 = s2;
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
		partial[(warp * ROWS + r) * 32u + lane] = acc[r];
	__syncthreads();
	if ( warp == 0u )
	{
		#pragma unroll
		for ( uint32_t r = 0u; r < ROWS; ++r )
		{
			float total = partial[r * 32u + lane];
			#pragma unroll
			for ( uint32_t w = 1u; w < WARPS; ++w )
				total += partial[(w * ROWS + r) * 32u + lane];
			const uint16_t bf = LmFloatToBf16(total);
			const uint32_t other = __shfl_xor_sync(0xffffffffu, (uint32_t)bf, 16u);
			if ( is_up == 0u )
			{
				const float out = SparkMathSitu(LmBf16ToFloat(bf), LmBf16ToFloat((uint16_t)other), K3_SITU_BETA, K3_SITU_LINEAR_BETA);
				output_bf16[(uint64_t)(first + r) * channels + hcell * LM_SKINNY_CELL_NEURONS + sub] = LmFloatToBf16(out);
			}
		}
	}
	if ( more != 0u )
		__syncthreads();
}

template<uint32_t MAX_BATCH>
__global__ __launch_bounds__(128u, 4)
static void K3CellSkinnyW1SituKernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ active_experts,
	uint64_t group_bytes, uint32_t k_tiles, uint32_t half_cells, uint32_t input_dim, uint32_t channels)
{
	__shared__ float partial[4u * MAX_BATCH * 32u];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t sub = lane & 15u, is_up = lane >> 4u;
	const uint32_t active_idx = blockIdx.x / half_cells, hcell = blockIdx.x % half_cells;
	LmDependentWait();
	if ( active_idx >= __ldg(active_experts) )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t group = __ldg(active_experts + 1u + active_idx);
	const uint32_t start = __ldg(group_row_offset + group), end = __ldg(group_row_offset + group + 1u);
	if ( start >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t cell = is_up * half_cells + hcell;
	const uint64_t tile_stride = (uint64_t)(half_cells * 2u) * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *base = weight + (uint64_t)group * group_bytes + (uint64_t)cell * LM_CELL_MMA_CELL_BYTES;
	for ( uint32_t first = start; first < end; )
	{
		const uint32_t count = end - first;
		if constexpr ( MAX_BATCH >= 4u )
		{
			if ( count >= 4u )
			{
				K3CellSkinnyW1SituRows<4u>(base, tile_stride, activation, route_source_token, output_bf16,
					k_tiles, input_dim, channels, hcell, warp, lane, sub, is_up, first, count > 4u ? 1u : 0u, partial);
				first += 4u;
				continue;
			}
			if ( count == 3u )
			{
				K3CellSkinnyW1SituRows<3u>(base, tile_stride, activation, route_source_token, output_bf16,
					k_tiles, input_dim, channels, hcell, warp, lane, sub, is_up, first, 0u, partial);
				first += 3u;
				continue;
			}
			if ( count == 2u )
			{
				K3CellSkinnyW1SituRows<2u>(base, tile_stride, activation, route_source_token, output_bf16,
					k_tiles, input_dim, channels, hcell, warp, lane, sub, is_up, first, 0u, partial);
				first += 2u;
				continue;
			}
		}
		K3CellSkinnyW1SituRows<1u>(base, tile_stride, activation, route_source_token, output_bf16,
			k_tiles, input_dim, channels, hcell, warp, lane, sub, is_up, first, count > 1u ? 1u : 0u, partial);
		first += 1u;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void K3CellSkinnyW2FinalizeRow1Kernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row,
	const float *__restrict__ route_weight, uint16_t *__restrict__ output_bf16,
	uint64_t group_bytes, uint32_t k_tiles, uint32_t total_cells, uint32_t input_dim)
{
	static_assert(K3_TOP_K == 16u, "four warps of two half-warps cover sixteen routes in two passes");
	__shared__ uint16_t s_out[K3_TOP_K][LM_SKINNY_CELL_NEURONS];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t sub = lane & 15u, half = lane >> 4u;
	const uint32_t cell = blockIdx.x;
	const uint64_t tile_stride = (uint64_t)total_cells * LM_CELL_MMA_CELL_BYTES;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t pass = 0u; pass < 2u; ++pass )
	{
		const uint32_t route = pass * 8u + warp * 2u + half;
		const uint32_t group = __ldg(route_expert + route);
		const uint32_t packed = __ldg(route_packed_row + route);
		const uint16_t *act[1] = {activation + (uint64_t)packed * input_dim};
		const uint8_t *base = weight + (uint64_t)group * group_bytes + (uint64_t)cell * LM_CELL_MMA_CELL_BYTES;
		float acc[1] = {0.0f};
		uint4 w0, w1;
		uint8_t s0 = 0u, s1 = 0u;
		if ( k_tiles > 0u )
		{
			w0 = __ldcs((const uint4 *)(base + sub * LM_SKINNY_CHUNK_BYTES));
			s0 = __ldcs(base + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		if ( k_tiles > 1u )
		{
			const uint8_t *tile = base + tile_stride;
			w1 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
			s1 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		for ( uint32_t t = 0u; t < k_tiles; ++t )
		{
			uint4 w2;
			uint8_t s2 = 0u;
			if ( t + 2u < k_tiles )
			{
				const uint8_t *tile = base + (uint64_t)(t + 2u) * tile_stride;
				w2 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
				s2 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
			}
			LmSkinnyCellTile<1u>(w0, s0, act, t * LM_SKINNY_CELL_TILE_K, acc);
			w0 = w1;
			s0 = s1;
			w1 = w2;
			s1 = s2;
		}
		s_out[route][sub] = LmFloatToBf16(acc[0]);
	}
	__syncthreads();
	if ( warp == 0u && half == 0u )
	{
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t route = 0u; route < K3_TOP_K; ++route )
			total = fmaf(__ldg(route_weight + route), LmBf16ToFloat(s_out[route][sub]), total);
		output_bf16[cell * LM_SKINNY_CELL_NEURONS + sub] = LmFloatToBf16(total);
	}
	LmDependentRelease();
}

template<uint32_t ROWS>
static __device__ __forceinline__ void K3CellSkinnyW2Rows(const uint8_t *base, uint64_t tile_stride,
	const uint16_t *activation, uint16_t *output_bf16, uint32_t k_tiles, uint32_t input_dim,
	uint32_t output_dim, uint32_t cell, uint32_t sub, uint32_t first)
{
	float acc[ROWS];
	const uint16_t *act[ROWS];
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
	{
		acc[r] = 0.0f;
		act[r] = activation + (uint64_t)(first + r) * input_dim;
	}
	uint4 w0, w1;
	uint8_t s0 = 0u, s1 = 0u;
	if ( k_tiles > 0u )
	{
		w0 = __ldcs((const uint4 *)(base + sub * LM_SKINNY_CHUNK_BYTES));
		s0 = __ldcs(base + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
	}
	if ( k_tiles > 1u )
	{
		const uint8_t *tile = base + tile_stride;
		w1 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
		s1 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
	}
	for ( uint32_t t = 0u; t < k_tiles; ++t )
	{
		uint4 w2;
		uint8_t s2 = 0u;
		if ( t + 2u < k_tiles )
		{
			const uint8_t *tile = base + (uint64_t)(t + 2u) * tile_stride;
			w2 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
			s2 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		LmSkinnyCellTile<ROWS>(w0, s0, act, t * LM_SKINNY_CELL_TILE_K, acc);
		w0 = w1;
		s0 = s1;
		w1 = w2;
		s1 = s2;
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS; ++r )
		output_bf16[(uint64_t)(first + r) * output_dim + cell * LM_SKINNY_CELL_NEURONS + sub] = LmFloatToBf16(acc[r]);
}

template<uint32_t MAX_BATCH>
__global__ __launch_bounds__(128u, 4)
static void K3CellSkinnyW2Kernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ active_experts, uint64_t group_bytes, uint32_t k_tiles,
	uint32_t total_cells, uint32_t input_dim, uint32_t output_dim)
{
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t sub = lane & 15u, half = lane >> 4u;
	const uint32_t col_blocks = total_cells / 8u;
	const uint32_t active_idx = blockIdx.x / col_blocks, n_block = blockIdx.x % col_blocks;
	LmDependentWait();
	if ( active_idx >= __ldg(active_experts) )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t group = __ldg(active_experts + 1u + active_idx);
	const uint32_t start = __ldg(group_row_offset + group), end = __ldg(group_row_offset + group + 1u);
	if ( start >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t cell = n_block * 8u + warp * 2u + half;
	const uint64_t tile_stride = (uint64_t)total_cells * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *base = weight + (uint64_t)group * group_bytes + (uint64_t)cell * LM_CELL_MMA_CELL_BYTES;
	for ( uint32_t first = start; first < end; )
	{
		const uint32_t count = end - first;
		if constexpr ( MAX_BATCH >= 4u )
		{
			if ( count >= 4u )
			{
				K3CellSkinnyW2Rows<4u>(base, tile_stride, activation, output_bf16, k_tiles, input_dim, output_dim, cell, sub, first);
				first += 4u;
				continue;
			}
			if ( count == 3u )
			{
				K3CellSkinnyW2Rows<3u>(base, tile_stride, activation, output_bf16, k_tiles, input_dim, output_dim, cell, sub, first);
				first += 3u;
				continue;
			}
			if ( count == 2u )
			{
				K3CellSkinnyW2Rows<2u>(base, tile_stride, activation, output_bf16, k_tiles, input_dim, output_dim, cell, sub, first);
				first += 2u;
				continue;
			}
		}
		K3CellSkinnyW2Rows<1u>(base, tile_stride, activation, output_bf16, k_tiles, input_dim, output_dim, cell, sub, first);
		first += 1u;
	}
	LmDependentRelease();
}

static int32_t K3MoeCellsSkinnyFastLaunch(const void *w1_weight, const void *w2_weight,
	const uint16_t *latent_full_bf16, uint16_t *gate_up_bf16, uint16_t *intermediate_bf16, uint16_t *latent_bf16,
	const uint32_t *group_row_offset, const uint32_t *route_expert, const uint32_t *route_source_token,
	const uint32_t *route_packed_row, const float *route_weight, const uint32_t *active_experts,
	uint32_t rows, uint32_t packed_rows, uint32_t channels, SparkGpuStream stream)
{
	const uint32_t w1_k_tiles = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_TILE_K;
	const uint32_t half_cells = channels / LM_SKINNY_CELL_NEURONS;
	const uint32_t max_active = packed_rows < K3_EXPERTS ? packed_rows : K3_EXPERTS;
	const uint64_t w1_group_bytes = (uint64_t)w1_k_tiles * (half_cells * 2u) * LM_CELL_MMA_CELL_BYTES;
	if ( rows == 1u )
		LM_LAUNCH_DEPENDENT((K3CellSkinnyW1SituKernel<1u>), max_active * half_cells, 128u, 0u, stream,
			(const uint8_t *)w1_weight, latent_full_bf16, intermediate_bf16, group_row_offset, route_source_token,
			active_experts, w1_group_bytes, w1_k_tiles, half_cells, K3_ROUTED_EXPERT_HIDDEN, channels);
	else
		LM_LAUNCH_DEPENDENT((K3CellSkinnyW1SituKernel<4u>), max_active * half_cells, 128u, 0u, stream,
			(const uint8_t *)w1_weight, latent_full_bf16, intermediate_bf16, group_row_offset, route_source_token,
			active_experts, w1_group_bytes, w1_k_tiles, half_cells, K3_ROUTED_EXPERT_HIDDEN, channels);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w2_k_tiles = channels / LM_SKINNY_CELL_TILE_K;
	const uint32_t w2_total_cells = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_NEURONS;
	const uint64_t w2_group_bytes = (uint64_t)w2_k_tiles * w2_total_cells * LM_CELL_MMA_CELL_BYTES;
	if ( rows == 1u )
	{
		LM_LAUNCH_DEPENDENT((K3CellSkinnyW2FinalizeRow1Kernel), w2_total_cells, 128u, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, route_expert, route_packed_row, route_weight,
			latent_bf16, w2_group_bytes, w2_k_tiles, w2_total_cells, channels);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	const uint32_t w2_col_blocks = w2_total_cells / 8u;
	LM_LAUNCH_DEPENDENT((K3CellSkinnyW2Kernel<4u>), max_active * w2_col_blocks, 128u, 0u, stream,
		(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
		w2_group_bytes, w2_k_tiles, w2_total_cells, channels, K3_ROUTED_EXPERT_HIDDEN);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_moe_finalize_token224_224, rows, 224u, 0u, stream,
		gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows, (uint32_t)K3_ROUTED_EXPERT_HIDDEN);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

struct K3SplitKTarget
{
	const uint8_t *weight;
	uint16_t *out_bf16;
	float *out_f32;
	uint32_t n;
	uint32_t stride;
	uint32_t col_offset;
	uint32_t first_tile;
};

struct K3SplitKArgs
{
	K3SplitKTarget t[2];
	uint32_t count;
	uint32_t total_tiles;
	uint32_t total_cols;
	uint32_t k_tiles;
	uint32_t split_k;
	uint32_t rows;
	uint32_t input_dim;
	const uint16_t *activation;
	float *scratch;
};

static_assert(sizeof(K3SplitKTarget) == 40u && sizeof(K3SplitKArgs) == 128u &&
	offsetof(K3SplitKArgs, count) == 80u && offsetof(K3SplitKArgs, total_cols) == 88u &&
	offsetof(K3SplitKArgs, k_tiles) == 92u && offsetof(K3SplitKArgs, split_k) == 96u &&
	offsetof(K3SplitKArgs, rows) == 100u && offsetof(K3SplitKArgs, input_dim) == 104u &&
	offsetof(K3SplitKArgs, activation) == 112u && offsetof(K3SplitKArgs, scratch) == 120u,
	"K3SplitKArgs layout matches skl2 blob[128, 8]");

static int32_t K3SplitKMmaLaunch(const K3SplitKTarget *targets, uint32_t count, const uint16_t *activation,
	float *scratch, uint32_t rows, uint32_t input_dim, uint32_t split_k, uint16_t *situ_out, SparkGpuStream stream)
{
	K3SplitKArgs args = {};
	args.count = count;
	args.k_tiles = input_dim / 64u;
	args.split_k = split_k;
	args.rows = rows;
	args.input_dim = input_dim;
	args.activation = activation;
	args.scratch = scratch;
	uint32_t tiles = 0u;
	for ( uint32_t p = 0u; p < count; ++p )
	{
		args.t[p] = targets[p];
		args.t[p].first_tile = tiles;
		tiles += (targets[p].n + 31u) / 32u;
		LmWeightPrefetchNote(stream, targets[p].weight, (uint64_t)targets[p].n * input_dim * sizeof(uint16_t));
	}
	args.total_tiles = tiles;
	args.total_cols = tiles * 32u;
	const dim3 grid(tiles, split_k);
	if ( K3Skl2Launch(rows <= 32u ? K3SKL2_k3_splitk_mma_32x32 : K3SKL2_k3_splitk_mma_64x32,
		grid, 128u, 0u, stream, args) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	SparkGpuError err;
	if ( situ_out != 0 )
	{
		const uint32_t ch = targets[0].n / 2u;
		err = K3Skl2Launch(K3SKL2_k3_splitk_reduce_situ_256, dim3(((ch / 2u) + 255u) / 256u, rows), 256u, 0u, stream,
			scratch, situ_out, rows, ch, split_k, (float)K3_SITU_BETA, (float)K3_SITU_LINEAR_BETA);
	}
	else
		err = K3Skl2Launch(K3SKL2_k3_splitk_reduce_256, dim3(((args.total_cols / 2u) + 255u) / 256u, rows), 256u, 0u, stream, args);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

struct K3KdaProjMmaArgs
{
	const uint8_t *w_qkvb;
	const uint8_t *w_gate;
	const uint8_t *w_decay;
	const uint16_t *activation;
	uint16_t *q;
	uint16_t *k;
	uint16_t *v;
	uint16_t *beta;
	uint16_t *gate;
	uint16_t *latent;
	uint16_t *rq;
	uint16_t *rk;
	uint16_t *rv;
	uint32_t rows;
	uint32_t qkvb_n;
	uint32_t gate_n;
	uint32_t decay_n;
	uint32_t rank_qk;
	uint32_t rank_v;
	uint32_t rank_heads;
	uint32_t tiles_qkvb;
	uint32_t tiles_gate;
};

static_assert(sizeof(K3KdaProjMmaArgs) == 144u &&
	offsetof(K3KdaProjMmaArgs, activation) == 24u && offsetof(K3KdaProjMmaArgs, q) == 32u &&
	offsetof(K3KdaProjMmaArgs, rq) == 80u && offsetof(K3KdaProjMmaArgs, rows) == 104u &&
	offsetof(K3KdaProjMmaArgs, tiles_qkvb) == 132u && offsetof(K3KdaProjMmaArgs, tiles_gate) == 136u,
	"K3KdaProjMmaArgs layout matches skl2 blob[144, 8]");

static int32_t K3WideMmaLaunch(const void *weight, const uint16_t *activation, uint16_t *output_bf16,
	uint32_t rows, uint32_t input_dim, uint32_t output_dim, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)output_dim * input_dim * sizeof(uint16_t));
	const uint32_t tiles = (output_dim + 31u) / 32u;
	const SparkGpuError err = K3Skl2Launch(rows <= 32u ? K3SKL2_k3_wide_mma_32x32 : K3SKL2_k3_wide_mma_64x32,
		tiles, 128u, 0u, stream, (const uint8_t *)weight, activation, output_bf16, rows, input_dim, output_dim);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3IndexQMmaWideSkipLaunch(const void *weight, const uint16_t *activation, uint16_t *output,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *positions,
	uint32_t rows, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)K3_INDEX_QUERY * K3_Q_LORA_RANK * sizeof(uint16_t));
	const bool wide = rows >= 128u;
	const uint32_t shared = wide ? (3u * (128u + 128u) * 128u) : (3u * (64u + 128u) * 128u);
	const uint32_t tiles = (wide ? ((rows + 127u) / 128u) : ((rows + 63u) / 64u)) * (K3_INDEX_QUERY / 128u);
	const SparkGpuError err = K3Skl2Launch(wide ? K3SKL2_k3_index_q_mma_wide_skip_128x128 : K3SKL2_k3_index_q_mma_wide_skip_64x128,
		tiles, LM_STREAM_THREADS, shared, stream, (const uint8_t *)weight, activation, output,
		sequence_of_row, context_length, positions, rows,
		(uint32_t)K3_Q_LORA_RANK, (uint32_t)K3_INDEX_QUERY, (uint32_t)K3_INDEX_TOPK);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3KdaProjPrefillMmaLaunch(const K3KdaProjMmaArgs &args, SparkGpuStream stream)
{
	constexpr uint32_t SHARED = 3u * (128u + 128u) * 128u;
	LmWeightPrefetchNote(stream, args.w_qkvb, (uint64_t)args.qkvb_n * K3_HIDDEN * sizeof(uint16_t));
	LmWeightPrefetchNote(stream, args.w_gate, (uint64_t)args.gate_n * K3_HIDDEN * sizeof(uint16_t));
	if ( args.decay_n != 0u )
		LmWeightPrefetchNote(stream, args.w_decay, (uint64_t)args.decay_n * K3_HIDDEN * sizeof(uint16_t));
	const uint32_t total_n = args.tiles_qkvb + args.tiles_gate + (args.decay_n + 127u) / 128u;
	const uint32_t tiles = ((args.rows + 127u) / 128u) * total_n;
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_kda_proj_prefill_mma_128x128,
		tiles, LM_STREAM_THREADS, SHARED, stream, args, (uint32_t)K3_HIDDEN);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3MoeDownRoutePrefillMmaLaunch(const void *w_down, const void *w_router, const uint16_t *activation,
	uint16_t *latent_bf16, float *router_f32, uint32_t rows, uint32_t down_n, uint32_t router_n, SparkGpuStream stream)
{
	constexpr uint32_t SHARED = 3u * (64u + 64u) * 128u;
	LmWeightPrefetchNote(stream, w_down, (uint64_t)down_n * K3_HIDDEN * sizeof(uint16_t));
	LmWeightPrefetchNote(stream, w_router, (uint64_t)router_n * K3_HIDDEN * sizeof(uint16_t));
	const uint32_t total_n = (down_n + 63u) / 64u + (router_n + 63u) / 64u;
	const uint32_t tiles = ((rows + 63u) / 64u) * total_n;
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_moe_down_route_prefill_mma_64x64,
		tiles, LM_STREAM_THREADS, SHARED, stream, (const uint8_t *)w_down, (const uint8_t *)w_router,
		activation, latent_bf16, router_f32, rows, (uint32_t)K3_HIDDEN, down_n, router_n);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3SharedW1SituPrefillMmaLaunch(const void *weight, const uint16_t *activation, uint16_t *output_bf16,
	uint32_t rows, uint32_t channels, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)channels * 2u * K3_HIDDEN * sizeof(uint16_t));
	const uint32_t n_tiles = (channels + 63u) / 64u;
	const bool wide = rows >= 1024u;
	const uint32_t shared = wide ? (3u * (128u + 128u) * 128u) : (3u * (64u + 128u) * 128u);
	const uint32_t tiles = (wide ? ((rows + 127u) / 128u) : ((rows + 63u) / 64u)) * n_tiles;
	const SparkGpuError err = K3Skl2Launch(wide ? K3SKL2_k3_shared_w1_situ_prefill_mma_128x128 : K3SKL2_k3_shared_w1_situ_prefill_mma_64x128,
		tiles, LM_STREAM_THREADS, shared, stream, (const uint8_t *)weight, activation, output_bf16,
		rows, (uint32_t)K3_HIDDEN, channels, (float)K3_SITU_BETA, (float)K3_SITU_LINEAR_BETA);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3RoutedUpAddPrefillMmaLaunch(const void *weight, const uint16_t *activation,
	const uint16_t *addend_bf16, uint16_t *output_bf16, uint32_t rows, uint32_t input_dim, uint32_t output_dim,
	SparkGpuStream stream)
{
	constexpr uint32_t SHARED = 3u * (128u + 128u) * 64u;
	LmWeightPrefetchNote(stream, weight, (uint64_t)output_dim * input_dim * sizeof(uint16_t));
	const uint32_t tiles = ((rows + 127u) / 128u) * ((output_dim + 127u) / 128u);
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_routed_up_add_prefill_mma_128x128,
		tiles, LM_STREAM_THREADS, SHARED, stream, (const uint8_t *)weight, activation, addend_bf16,
		output_bf16, rows, input_dim, output_dim);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
