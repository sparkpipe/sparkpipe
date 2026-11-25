#pragma once

#include <stdint.h>
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_prefetch.cuh"
#include "model-families/k3/source/config.h"

#define K3_ATTNRES_PERSISTENT_BLOCKS 96u

__global__ __launch_bounds__(256u, 1)
static void K3TopkRouteBuildRow1Kernel(const float *__restrict__ router_logits, const float *__restrict__ router_bias,
	uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_fault,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t *__restrict__ active_experts)
{
	__shared__ uint32_t sel[K3_TOP_K];
	LmDependentWait();
	LmTopkWarpsRow<K3_TOP_K, true, LM_TOPK_SCORE_SIGMOID>(router_logits, K3_EXPERTS, route_expert, route_weight,
		router_bias, (const uint16_t *)0, K3_ROUTED_SCALE, route_fault, 0u);
	__syncthreads();
	if ( threadIdx.x < K3_TOP_K )
		sel[threadIdx.x] = route_expert[threadIdx.x];
	__syncthreads();
	if ( threadIdx.x < K3_TOP_K )
	{
		const uint32_t mine = sel[threadIdx.x];
		uint32_t packed = 0u;
		#pragma unroll
		for ( uint32_t j = 0u; j < K3_TOP_K; ++j )
			if ( sel[j] < mine || (sel[j] == mine && j < threadIdx.x) )
				packed++;
		route_packed_row[threadIdx.x] = packed;
		route_source_token[packed] = 0u;
		if ( active_experts != 0 )
			active_experts[1u + packed] = mine;
	}
	if ( threadIdx.x == 0u && active_experts != 0 )
		active_experts[0u] = K3_TOP_K;
	for ( uint32_t e = threadIdx.x; e <= K3_EXPERTS; e += 256u )
	{
		uint32_t before = 0u;
		#pragma unroll
		for ( uint32_t j = 0u; j < K3_TOP_K; ++j )
			if ( sel[j] < e )
				before++;
		group_row_offset[e] = before;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 1)
static void K3RouteBuildFastKernel(const uint32_t *__restrict__ route_expert, uint32_t routes,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t *__restrict__ active_experts)
{
	__shared__ uint16_t w_hist[8][K3_EXPERTS];
	__shared__ uint32_t warp_tot[8], act_tot[8], chunk_base, act_base;
	const uint32_t tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
	LmDependentWait();
	for ( uint32_t i = tid; i < 8u * K3_EXPERTS; i += 256u )
		((uint16_t *)w_hist)[i] = 0u;
	if ( tid == 0u )
	{
		chunk_base = 0u;
		act_base = 0u;
	}
	__syncthreads();
	const uint32_t slice = (routes + 7u) >> 3u;
	const uint32_t w_beg = warp * slice;
	const uint32_t w_end = w_beg + slice < routes ? w_beg + slice : routes;
	for ( uint32_t p = w_beg; p < w_end; p += 32u )
	{
		const uint32_t i = p + lane;
		const bool active = i < w_end;
		const uint32_t e = active ? __ldg(route_expert + i) : 0xffffffffu;
		const uint32_t amask = __ballot_sync(0xffffffffu, active);
		if ( active )
		{
			const uint32_t m = __match_any_sync(amask, e);
			if ( lane == (uint32_t)(__ffs(m) - 1) )
				w_hist[warp][e] = (uint16_t)(w_hist[warp][e] + (uint16_t)__popc(m));
		}
	}
	__syncthreads();
	for ( uint32_t base_e = 0u; base_e < K3_EXPERTS; base_e += 256u )
	{
		const uint32_t e = base_e + tid;
		const bool valid = e < K3_EXPERTS;
		uint32_t c = 0u;
		uint16_t w_off[8];
		if ( valid )
		{
			#pragma unroll
			for ( uint32_t w = 0u; w < 8u; ++w )
			{
				w_off[w] = (uint16_t)c;
				c += w_hist[w][e];
			}
		}
		uint32_t scan = c;
		#pragma unroll
		for ( uint32_t step = 1u; step < 32u; step <<= 1u )
		{
			const uint32_t up = __shfl_up_sync(0xffffffffu, scan, step);
			if ( lane >= step )
				scan += up;
		}
		const uint32_t is_act = valid && c > 0u ? 1u : 0u;
		const uint32_t act_mask = __ballot_sync(0xffffffffu, is_act != 0u);
		const uint32_t act_rank = (uint32_t)__popc(act_mask & ((1u << lane) - 1u));
		if ( lane == 31u )
		{
			warp_tot[warp] = scan;
			act_tot[warp] = (uint32_t)__popc(act_mask);
		}
		__syncthreads();
		uint32_t w_pre = 0u, a_pre = 0u, c_add = 0u, a_add = 0u;
		#pragma unroll
		for ( uint32_t w = 0u; w < 8u; ++w )
		{
			if ( w < warp ) { w_pre += warp_tot[w]; a_pre += act_tot[w]; }
			c_add += warp_tot[w];
			a_add += act_tot[w];
		}
		const uint32_t ex_off = chunk_base + w_pre + (scan - c);
		if ( valid )
		{
			group_row_offset[e] = ex_off;
			#pragma unroll
			for ( uint32_t w = 0u; w < 8u; ++w )
				w_hist[w][e] = (uint16_t)(ex_off + w_off[w]);
			if ( is_act != 0u && active_experts != 0 )
				active_experts[1u + act_base + a_pre + act_rank] = e;
		}
		__syncthreads();
		if ( tid == 0u )
		{
			chunk_base += c_add;
			act_base += a_add;
		}
		__syncthreads();
	}
	if ( tid == 0u )
	{
		group_row_offset[K3_EXPERTS] = chunk_base;
		if ( active_experts != 0 )
			active_experts[0u] = act_base;
	}
	__syncthreads();
	for ( uint32_t p = w_beg; p < w_end; p += 32u )
	{
		const uint32_t i = p + lane;
		const bool active = i < w_end;
		const uint32_t e = active ? __ldg(route_expert + i) : 0xffffffffu;
		const uint32_t amask = __ballot_sync(0xffffffffu, active);
		if ( active )
		{
			const uint32_t m = __match_any_sync(amask, e);
			const uint32_t leader = (uint32_t)(__ffs(m) - 1);
			const uint32_t rnk = (uint32_t)__popc(m & ((1u << lane) - 1u));
			const uint32_t base = w_hist[warp][e];
			const uint32_t packed = base + rnk;
			route_packed_row[i] = packed;
			route_source_token[packed] = i / K3_TOP_K;
			if ( lane == leader )
				w_hist[warp][e] = (uint16_t)(base + (uint32_t)__popc(m));
		}
	}
	LmDependentRelease();
}

static __device__ __forceinline__ float K3AttnResAddAndScore(uint16_t *partial_row, const uint16_t *addend_row,
	const uint16_t *score_weight_bf16, float epsilon, float *reduction)
{
	uint4 staged[4];
	float square = 0.0f, dot = 0.0f;
	#pragma unroll
	for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
	{
		const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
		if ( index < K3_HIDDEN )
		{
			float sum[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(partial_row + index), sum);
			LmAddBf16x8(sum, *(const uint4 *)(addend_row + index));
			staged[chunk] = LmPackUint4FloatToBf16(sum);
			*(uint4 *)(partial_row + index) = staged[chunk];
			square = LmSumSqBf16x8(staged[chunk], square);
		}
	}
	const float inverse = SparkMathRmsScale(LmBlockSum<256u>(square, reduction), (float)K3_HIDDEN, epsilon);
	#pragma unroll
	for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
	{
		const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
		if ( index < K3_HIDDEN )
		{
			float v[8], w[8];
			LmLoadBf16x8AsFloat(staged[chunk], v);
			LmLoadBf16x8AsFloat(*(const uint4 *)(score_weight_bf16 + index), w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				dot += v[k] * inverse * w[k];
		}
	}
	return LmBlockSum<256u>(dot, reduction);
}

__global__ __launch_bounds__(256u, 2)
static void K3AttnResScoreAddKernel(const uint16_t *__restrict__ bank_bf16, uint16_t *__restrict__ partial_bf16,
	const uint16_t *__restrict__ addend_bf16, const uint16_t *__restrict__ score_weight_bf16,
	float *__restrict__ score_f32, uint32_t sources, uint32_t rows, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<256u>()];
	const uint32_t row = blockIdx.x, source = blockIdx.y;
	LmDependentWait();
	float dot;
	if ( source + 1u == sources && addend_bf16 != 0 )
		dot = K3AttnResAddAndScore(partial_bf16 + (uint64_t)row * K3_HIDDEN,
			addend_bf16 + (uint64_t)row * K3_HIDDEN, score_weight_bf16, epsilon, reduction);
	else
	{
		const uint16_t *values = LmAttnResSourceRow(bank_bf16, partial_bf16, source, sources, rows, row, K3_HIDDEN);
		dot = LmAttnResScoreValue<256u>(values, score_weight_bf16, K3_HIDDEN, epsilon, reduction, true);
	}
	if ( threadIdx.x == 0u )
		score_f32[(uint64_t)row * K3_ATTNRES_MAX_SOURCES + source] = dot;
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 2)
static void K3AttnResMixNormKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16,
	const float *__restrict__ score_f32, const uint16_t *__restrict__ norm_weight_bf16,
	uint16_t *__restrict__ hidden_bf16, uint16_t *__restrict__ normed_bf16,
	uint32_t sources, uint32_t rows, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<256u>()], weight[K3_ATTNRES_MAX_SOURCES];
	__shared__ __align__(16) uint16_t s_row[K3_HIDDEN];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	LmWarpSoftmax<256u>(score_f32 + (uint64_t)row * K3_ATTNRES_MAX_SOURCES, weight, sources);
	uint16_t *h_row = hidden_bf16 + (uint64_t)row * K3_HIDDEN;
	for ( uint32_t index = threadIdx.x * 8u; index < K3_HIDDEN; index += 256u * 8u )
	{
		const uint4 mixed = LmAttnResMix8(bank_bf16, partial_bf16, weight, sources, rows, row, K3_HIDDEN, index);
		*(uint4 *)(h_row + index) = mixed;
		if ( norm_weight_bf16 != 0 )
			*(uint4 *)(s_row + index) = mixed;
	}
	if ( norm_weight_bf16 != 0 )
	{
		__syncthreads();
		float total = 0.0f;
		for ( uint32_t idx = threadIdx.x; idx < K3_HIDDEN; idx += 256u )
		{
			const float v = LmBf16ToFloat(s_row[idx]);
			total += v * v;
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<256u>(total, reduction), (float)K3_HIDDEN, epsilon);
		uint16_t *n_row = normed_bf16 + (uint64_t)row * K3_HIDDEN;
		for ( uint32_t index = threadIdx.x * 8u; index < K3_HIDDEN; index += 256u * 8u )
		{
			float vals[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(s_row + index), vals);
			LmRmsStore8(vals, norm_weight_bf16, index, scale, n_row + index, LmRmsGainOp<true>{});
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 2)
static void K3PartialAddAttnResNormKernel(const uint16_t *__restrict__ bank_bf16, uint16_t *__restrict__ partial_bf16,
	const uint16_t *__restrict__ addend_bf16, const uint16_t *__restrict__ score_weight_bf16,
	const uint16_t *__restrict__ norm_weight_bf16, uint16_t *__restrict__ hidden_bf16,
	uint16_t *__restrict__ normed_bf16, uint32_t sources, uint32_t rows, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<256u>()], score[K3_ATTNRES_MAX_SOURCES], weight[K3_ATTNRES_MAX_SOURCES];
	__shared__ __align__(16) uint16_t s_row[K3_HIDDEN];
	LmDependentWait();
	for ( uint32_t row = blockIdx.x; row < rows; row += gridDim.x )
	{
		for ( uint32_t s = 0u; s < sources; ++s )
		{
			float dot;
			if ( s + 1u == sources && addend_bf16 != 0 )
				dot = K3AttnResAddAndScore(partial_bf16 + (uint64_t)row * K3_HIDDEN,
					addend_bf16 + (uint64_t)row * K3_HIDDEN, score_weight_bf16, epsilon, reduction);
			else
				dot = LmAttnResScoreValue<256u>(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, K3_HIDDEN),
					score_weight_bf16, K3_HIDDEN, epsilon, reduction, true);
			if ( threadIdx.x == 0u )
				score[s] = dot;
			__syncthreads();
		}
		LmWarpSoftmax<256u>(score, weight, sources);
		uint16_t *h_row = hidden_bf16 + (uint64_t)row * K3_HIDDEN;
		for ( uint32_t index = threadIdx.x * 8u; index < K3_HIDDEN; index += 256u * 8u )
		{
			const uint4 mixed = LmAttnResMix8(bank_bf16, partial_bf16, weight, sources, rows, row, K3_HIDDEN, index);
			*(uint4 *)(h_row + index) = mixed;
			if ( norm_weight_bf16 != 0 )
				*(uint4 *)(s_row + index) = mixed;
		}
		if ( norm_weight_bf16 != 0 )
		{
			__syncthreads();
			float total = 0.0f;
			for ( uint32_t idx = threadIdx.x; idx < K3_HIDDEN; idx += 256u )
			{
				const float v = LmBf16ToFloat(s_row[idx]);
				total += v * v;
			}
			const float scale = SparkMathRmsScale(LmBlockSumOnce<256u>(total, reduction), (float)K3_HIDDEN, epsilon);
			uint16_t *n_row = normed_bf16 + (uint64_t)row * K3_HIDDEN;
			for ( uint32_t index = threadIdx.x * 8u; index < K3_HIDDEN; index += 256u * 8u )
			{
				float vals[8];
				LmLoadBf16x8AsFloat(*(const uint4 *)(s_row + index), vals);
				LmRmsStore8(vals, norm_weight_bf16, index, scale, n_row + index, LmRmsGainOp<true>{});
			}
		}
		if ( row + gridDim.x < rows )
			__syncthreads();
	}
	LmDependentRelease();
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
		for ( uint32_t idx = threadIdx.x; idx < 2u * HALF_VECS; idx += LM_CELL_MMA_THREADS )
		{
			const uint8_t *src = idx < HALF_VECS ? g_ptr + idx * 16u : u_ptr + (idx - HALF_VECS) * 16u;
			LmCellMmaCopy(w_buf + idx * 16u, src, 16u);
		}
		g_ptr += tile_stride;
		u_ptr += tile_stride;
	};
	#pragma unroll
	for ( uint32_t t = 0u; t + 1u < LM_CELL_MMA_STAGES; ++t )
	{
		if ( t < k_tiles )
			load_stage(t);
		LmCellMmaCommit();
	}
	for ( uint32_t t = 0u; t < k_tiles; ++t )
	{
		const uint32_t slot = t % LM_CELL_MMA_STAGES, ahead = t + LM_CELL_MMA_STAGES - 1u;
		const uint16_t *buffer = stage + slot * STAGE_ELEMENTS;
		const uint8_t *w_slot = weights + slot * LM_CELL_MMA_WEIGHT_TILE_BYTES;
		LmCellMmaWait<LM_CELL_MMA_STAGES - 2u>();
		__syncthreads();
		if ( ahead < k_tiles )
			load_stage(ahead % LM_CELL_MMA_STAGES);
		LmCellMmaCommit();
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
		#pragma unroll
		for ( uint32_t m = 0u; m < CHUNKS; ++m )
		{
			const uint16_t *a_row = buffer + (m * 16u + g) * LM_CELL_MMA_STAGE_PITCH + c * 8u;
			const uint4 low = *(const uint4 *)a_row;
			const uint4 high = *(const uint4 *)(a_row + 8u * LM_CELL_MMA_STAGE_PITCH);
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
		}
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
	__shared__ __align__(16) uint16_t stage[LM_CELL_MMA_STAGES * STAGE_ELEMENTS];
	__shared__ __align__(16) uint8_t weights[LM_CELL_MMA_STAGES * LM_CELL_MMA_WEIGHT_TILE_BYTES];
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

template<uint32_t CHUNKS, uint32_t STAGE_ELEMENTS, uint32_t STAGES>
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
			LmCellMmaCopy(w_buf + idx * 16u, c_ptr + idx * 16u, 16u);
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
		#pragma unroll
		for ( uint32_t m = 0u; m < CHUNKS; ++m )
		{
			const uint16_t *a_row = buffer + (m * 16u + g) * LM_CELL_MMA_STAGE_PITCH + c * 8u;
			const uint4 low = *(const uint4 *)a_row;
			const uint4 high = *(const uint4 *)(a_row + 8u * LM_CELL_MMA_STAGE_PITCH);
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
		}
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
			for ( uint32_t f = 0u; f < 4u; ++f )
				*(uint32_t *)(output_bf16 + (uint64_t)packed * slice_cols + (out_cell_local + f / 2u) * LM_SKINNY_CELL_NEURONS + (f % 2u) * 8u + c * 2u) =
					LmPackBf16Pair(acc[m][f][h * 2u], acc[m][f][h * 2u + 1u]);
		}
	}
}

template<uint32_t MAX_CHUNKS, uint32_t STAGES>
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
				K3CellMmaW2SliceRows<8u, STAGE_ELEMENTS, STAGES>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
		}
		if constexpr ( MAX_CHUNKS >= 6u )
		{
			if ( last - first > 4u * 16u )
			{
				K3CellMmaW2SliceRows<6u, STAGE_ELEMENTS, STAGES>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
		}
		if constexpr ( MAX_CHUNKS >= 4u )
		{
			if ( last - first > 3u * 16u )
			{
				K3CellMmaW2SliceRows<4u, STAGE_ELEMENTS, STAGES>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
			if ( last - first > 2u * 16u )
			{
				K3CellMmaW2SliceRows<3u, STAGE_ELEMENTS, STAGES>(cells_base, tile_stride, activation, output_bf16,
					k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
				continue;
			}
		}
		if ( last - first > 16u )
			K3CellMmaW2SliceRows<2u, STAGE_ELEMENTS, STAGES>(cells_base, tile_stride, activation, output_bf16,
				k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
		else
			K3CellMmaW2SliceRows<1u, STAGE_ELEMENTS, STAGES>(cells_base, tile_stride, activation, output_bf16,
				k_tiles, input_dim, slice_cols, out_cell_local, first, last, stage, weights);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void K3MoeFinalizeWarpSliceKernel(const uint16_t *__restrict__ packed_slice_bf16,
	const uint32_t *__restrict__ route_packed_row, const float *__restrict__ route_weight,
	uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t col_offset, uint32_t slice_cols, uint32_t total_cols)
{
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t token = blockIdx.x * 4u + warp;
	if ( token >= tokens )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t my_pr = lane < K3_TOP_K ? __ldg(route_packed_row + (uint64_t)token * K3_TOP_K + lane) : 0u;
	const float my_rw = lane < K3_TOP_K ? __ldg(route_weight + (uint64_t)token * K3_TOP_K + lane) : 0.0f;
	for ( uint32_t elem = lane * 8u; elem < slice_cols; elem += 32u * 8u )
	{
		float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		#pragma unroll
		for ( uint32_t route = 0u; route < K3_TOP_K; ++route )
		{
			const uint32_t pr = __shfl_sync(0xffffffffu, my_pr, route);
			const float rw = __shfl_sync(0xffffffffu, my_rw, route);
			LmFmaBf16x8(total, *(const uint4 *)(packed_slice_bf16 + (uint64_t)pr * slice_cols + elem), rw);
		}
		*(uint4 *)(output_bf16 + (uint64_t)token * total_cols + col_offset + elem) = LmPackUint4FloatToBf16(total);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void K3MoeFinalizeBlockKernel(const uint16_t *__restrict__ packed_bf16,
	const uint32_t *__restrict__ route_packed_row, const float *__restrict__ route_weight,
	uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t total_cols)
{
	__shared__ uint32_t s_pr[K3_TOP_K];
	__shared__ float s_rw[K3_TOP_K];
	LmDependentWait();
	const uint32_t elem = (blockIdx.x * 128u + threadIdx.x) * 8u;
	const uint32_t token = blockIdx.y;
	if ( token >= tokens )
	{
		LmDependentRelease();
		return;
	}
	if ( threadIdx.x < K3_TOP_K )
	{
		const uint64_t slot = (uint64_t)token * K3_TOP_K + threadIdx.x;
		s_pr[threadIdx.x] = route_packed_row[slot];
		s_rw[threadIdx.x] = route_weight[slot];
	}
	__syncthreads();
	if ( elem < total_cols )
	{
		float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		#pragma unroll
		for ( uint32_t route = 0u; route < K3_TOP_K; ++route )
			LmFmaBf16x8(total, *(const uint4 *)(packed_bf16 + (uint64_t)s_pr[route] * total_cols + elem), s_rw[route]);
		*(uint4 *)(output_bf16 + (uint64_t)token * total_cols + elem) = LmPackUint4FloatToBf16(total);
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
		static bool w1_carveout_set = false;
		if ( !w1_carveout_set )
		{
			(void)cudaFuncSetAttribute((const void *)K3CellMmaW1SituKernel<8u>,
				cudaFuncAttributePreferredSharedMemoryCarveout, 100);
			w1_carveout_set = true;
		}
		LM_LAUNCH_DEPENDENT((K3CellMmaW1SituKernel<8u>), max_active * w1_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
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
	if ( full_out_bytes > 16u * 1024u * 1024u && packed_rows <= 16384u )
	{
		const uint64_t target_bytes = 10u * 1024u * 1024u;
		uint32_t raw_cells = (uint32_t)(target_bytes / ((uint64_t)packed_rows * LM_SKINNY_CELL_NEURONS * sizeof(uint16_t)));
		raw_cells = (raw_cells / LM_CELL_MMA_BLOCK_CELLS) * LM_CELL_MMA_BLOCK_CELLS;
		if ( raw_cells < 2u * LM_CELL_MMA_BLOCK_CELLS )
			raw_cells = 2u * LM_CELL_MMA_BLOCK_CELLS;
		if ( raw_cells > w2_total_cells )
			raw_cells = w2_total_cells;
		slice_cells = raw_cells;
	}
	if ( slice_cells == w2_total_cells )
	{
		const uint32_t w2_col_blocks = w2_total_cells / LM_CELL_MMA_BLOCK_CELLS;
		if ( packed_rows <= 2048u )
			LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<2u, 4u>), max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
				(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
				w2_group_bytes, w2_k_tiles, w2_total_cells, 0u, w2_total_cells, channels);
		else
			LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<6u, 3u>), max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
				(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
				w2_group_bytes, w2_k_tiles, w2_total_cells, 0u, w2_total_cells, channels);
		const uint32_t vec_blocks = ((K3_ROUTED_EXPERT_HIDDEN / 8u) + 127u) / 128u;
		LM_LAUNCH_DEPENDENT((K3MoeFinalizeBlockKernel), dim3(vec_blocks, rows), 128u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows, K3_ROUTED_EXPERT_HIDDEN);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	for ( uint32_t cell_begin = 0u; cell_begin < w2_total_cells; cell_begin += slice_cells )
	{
		const uint32_t cur_cells = w2_total_cells - cell_begin < slice_cells ? w2_total_cells - cell_begin : slice_cells;
		const uint32_t cur_col_blocks = cur_cells / LM_CELL_MMA_BLOCK_CELLS;
		const uint32_t cur_cols = cur_cells * LM_SKINNY_CELL_NEURONS;
		LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<2u, 4u>), max_active * cur_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
			w2_group_bytes, w2_k_tiles, w2_total_cells, cell_begin, cur_cells, channels);
		LM_LAUNCH_DEPENDENT((K3MoeFinalizeWarpSliceKernel), (rows + 3u) / 4u, 128u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows,
			cell_begin * LM_SKINNY_CELL_NEURONS, cur_cols, K3_ROUTED_EXPERT_HIDDEN);
	}
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
