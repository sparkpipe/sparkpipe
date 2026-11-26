#pragma once

#include <stdint.h>
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
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

static __device__ __forceinline__ float K3AttnResScoreSinglePass(const uint16_t *values,
	const uint16_t *score_weight_bf16, float epsilon, float *reduction, uint16_t *cache_smem = 0)
{
	uint4 staged[4];
	float square = 0.0f, dot = 0.0f;
	#pragma unroll
	for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
	{
		const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
		if ( index < K3_HIDDEN )
		{
			staged[chunk] = __ldg((const uint4 *)(values + index));
			if ( cache_smem != 0 )
				*(uint4 *)(cache_smem + index) = staged[chunk];
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
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(score_weight_bf16 + index)), w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				dot += v[k] * inverse * w[k];
		}
	}
	return LmBlockSum<256u>(dot, reduction);
}

static __device__ __forceinline__ float K3AttnResAddAndScore(uint16_t *partial_row, const uint16_t *addend_row,
	const uint16_t *score_weight_bf16, float epsilon, float *reduction, uint16_t *cache_smem = 0, bool stream_partial = false)
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
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(partial_row + index)), sum);
			LmAddBf16x8(sum, __ldcs((const uint4 *)(addend_row + index)));
			staged[chunk] = LmPackUint4FloatToBf16(sum);
			if ( stream_partial )
				__stcs((uint4 *)(partial_row + index), staged[chunk]);
			else
				*(uint4 *)(partial_row + index) = staged[chunk];
			if ( cache_smem != 0 )
				*(uint4 *)(cache_smem + index) = staged[chunk];
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
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(score_weight_bf16 + index)), w);
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
		dot = K3AttnResScoreSinglePass(values, score_weight_bf16, epsilon, reduction);
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
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	LmWarpSoftmax<256u>(score_f32 + (uint64_t)row * K3_ATTNRES_MAX_SOURCES, weight, sources);
	if ( norm_weight_bf16 != 0 )
	{
		uint4 mixed_staged[4];
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
		{
			const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
			if ( index < K3_HIDDEN )
			{
				mixed_staged[chunk] = LmAttnResMix8(bank_bf16, partial_bf16, weight, sources, rows, row, K3_HIDDEN, index);
				total = LmSumSqBf16x8(mixed_staged[chunk], total);
			}
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<256u>(total, reduction), (float)K3_HIDDEN, epsilon);
		uint16_t *n_row = normed_bf16 + (uint64_t)row * K3_HIDDEN;
		#pragma unroll
		for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
		{
			const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
			if ( index < K3_HIDDEN )
			{
				float vals[8];
				LmLoadBf16x8AsFloat(mixed_staged[chunk], vals);
				LmRmsStore8(vals, norm_weight_bf16, index, scale, n_row + index, LmRmsGainOp<true>{});
			}
		}
	}
	else
	{
		uint16_t *h_row = hidden_bf16 + (uint64_t)row * K3_HIDDEN;
		for ( uint32_t index = threadIdx.x * 8u; index < K3_HIDDEN; index += 256u * 8u )
			*(uint4 *)(h_row + index) = LmAttnResMix8(bank_bf16, partial_bf16, weight, sources, rows, row, K3_HIDDEN, index);
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
	const bool stream_out = rows >= 128u;
	for ( uint32_t row = blockIdx.x; row < rows; row += gridDim.x )
	{
		for ( uint32_t s = 0u; s < sources; ++s )
		{
			const bool is_last = s + 1u == sources;
			float dot;
			if ( is_last && addend_bf16 != 0 )
				dot = K3AttnResAddAndScore(partial_bf16 + (uint64_t)row * K3_HIDDEN,
					addend_bf16 + (uint64_t)row * K3_HIDDEN, score_weight_bf16, epsilon, reduction, s_row, stream_out);
			else
				dot = K3AttnResScoreSinglePass(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, K3_HIDDEN),
					score_weight_bf16, epsilon, reduction, is_last ? s_row : (uint16_t *)0);
			if ( threadIdx.x == 0u )
				score[s] = dot;
			__syncthreads();
		}
		LmWarpSoftmax<256u>(score, weight, sources);
		if ( norm_weight_bf16 != 0 )
		{
			uint4 mixed_staged[4];
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
			{
				const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
				if ( index < K3_HIDDEN )
				{
					float total_mix[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
					for ( uint32_t s = 0u; s + 1u < sources; ++s )
						LmFmaBf16x8(total_mix, __ldg((const uint4 *)(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, K3_HIDDEN) + index)), weight[s]);
					LmFmaBf16x8(total_mix, *(const uint4 *)(s_row + index), weight[sources - 1u]);
					mixed_staged[chunk] = LmPackUint4FloatToBf16(total_mix);
					total = LmSumSqBf16x8(mixed_staged[chunk], total);
				}
			}
			const float scale = SparkMathRmsScale(LmBlockSumOnce<256u>(total, reduction), (float)K3_HIDDEN, epsilon);
			uint16_t *n_row = normed_bf16 + (uint64_t)row * K3_HIDDEN;
			#pragma unroll
			for ( uint32_t chunk = 0u; chunk < 4u; ++chunk )
			{
				const uint32_t index = (chunk * 256u + threadIdx.x) * 8u;
				if ( index < K3_HIDDEN )
				{
					float vals[8];
					LmLoadBf16x8AsFloat(mixed_staged[chunk], vals);
					LmRmsStore8(vals, norm_weight_bf16, index, scale, n_row + index, LmRmsGainOp<true>{});
				}
			}
		}
		else
		{
			uint16_t *h_row = hidden_bf16 + (uint64_t)row * K3_HIDDEN;
			for ( uint32_t index = threadIdx.x * 8u; index < K3_HIDDEN; index += 256u * 8u )
			{
				float total_mix[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
				for ( uint32_t s = 0u; s + 1u < sources; ++s )
					LmFmaBf16x8(total_mix, __ldg((const uint4 *)(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, K3_HIDDEN) + index)), weight[s]);
				LmFmaBf16x8(total_mix, *(const uint4 *)(s_row + index), weight[sources - 1u]);
				*(uint4 *)(h_row + index) = LmPackUint4FloatToBf16(total_mix);
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
			(void)SparkGpuFuncSetAttribute((const void *)K3CellMmaW1SituKernel<8u>,
				SPARK_GPU_FUNCTION_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
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
			LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<2u, 4u, false>), max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
				(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
				w2_group_bytes, w2_k_tiles, w2_total_cells, 0u, w2_total_cells, channels);
		else
			LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<6u, 3u, true>), max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
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
		LM_LAUNCH_DEPENDENT((K3CellMmaW2SliceKernel<4u, 4u, false>), max_active * cur_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
			w2_group_bytes, w2_k_tiles, w2_total_cells, cell_begin, cur_cells, channels);
		LM_LAUNCH_DEPENDENT((K3MoeFinalizeWarpSliceKernel), (rows + 3u) / 4u, 128u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows,
			cell_begin * LM_SKINNY_CELL_NEURONS, cur_cols, K3_ROUTED_EXPERT_HIDDEN);
	}
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
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
		if constexpr ( MAX_BATCH >= 8u )
		{
			if ( count >= 8u )
			{
				K3CellSkinnyW1SituRows<8u>(base, tile_stride, activation, route_source_token, output_bf16,
					k_tiles, input_dim, channels, hcell, warp, lane, sub, is_up, first, count > 8u ? 1u : 0u, partial);
				first += 8u;
				continue;
			}
		}
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
	__shared__ uint16_t s_out[K3_TOP_K][32u];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t sub = lane & 15u, half = lane >> 4u;
	const uint32_t cell = blockIdx.x * 2u + half;
	const uint64_t tile_stride = (uint64_t)total_cells * LM_CELL_MMA_CELL_BYTES;
	LmDependentWait();
	for ( uint32_t route = warp; route < K3_TOP_K; route += 4u )
	{
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
		s_out[route][lane] = LmFloatToBf16(acc[0]);
	}
	__syncthreads();
	if ( warp == 0u )
	{
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t route = 0u; route < K3_TOP_K; ++route )
			total = fmaf(__ldg(route_weight + route), LmBf16ToFloat(s_out[route][lane]), total);
		output_bf16[blockIdx.x * 32u + lane] = LmFloatToBf16(total);
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
		if constexpr ( MAX_BATCH >= 8u )
		{
			if ( count >= 8u )
			{
				K3CellSkinnyW2Rows<8u>(base, tile_stride, activation, output_bf16, k_tiles, input_dim, output_dim, cell, sub, first);
				first += 8u;
				continue;
			}
		}
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
		LM_LAUNCH_DEPENDENT((K3CellSkinnyW2FinalizeRow1Kernel), w2_total_cells / 2u, 128u, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, route_expert, route_packed_row, route_weight,
			latent_bf16, w2_group_bytes, w2_k_tiles, w2_total_cells, channels);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	const uint32_t w2_col_blocks = w2_total_cells / 8u;
	LM_LAUNCH_DEPENDENT((K3CellSkinnyW2Kernel<4u>), max_active * w2_col_blocks, 128u, 0u, stream,
		(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
		w2_group_bytes, w2_k_tiles, w2_total_cells, channels, K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t vec_blocks = ((K3_ROUTED_EXPERT_HIDDEN / 8u) + 127u) / 128u;
	LM_LAUNCH_DEPENDENT((K3MoeFinalizeBlockKernel), dim3(vec_blocks, rows), 128u, 0u, stream,
		gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows, K3_ROUTED_EXPERT_HIDDEN);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

__global__ __launch_bounds__(128u, 4)
static void K3IndexHeadSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t rows)
{
	constexpr uint32_t CHUNKS = K3_HIDDEN / 8u;
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	uint32_t bound = context_length[sequence_of_row[row]];
	const uint32_t pos = positions[row];
	if ( pos + 1u < bound )
		bound = pos + 1u;
	if ( bound <= K3_INDEX_TOPK )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = blockIdx.x * 4u + warp;
	if ( head < K3_INDEX_HEADS )
	{
		const uint4 *w_row = (const uint4 *)(weight + (uint64_t)head * K3_HIDDEN);
		const uint4 *a_row = (const uint4 *)(activation + (uint64_t)row * K3_HIDDEN);
		float acc = 0.0f;
		for ( uint32_t c = lane; c < CHUNKS; c += 32u )
		{
			float wf[8], af[8];
			LmSkinnyUnpackBf16x8(__ldcs(w_row + c), wf);
			LmSkinnyUnpackBf16x8(__ldg(a_row + c), af);
			acc = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf, af), acc);
		}
		#pragma unroll
		for ( uint32_t off = 16u; off > 0u; off >>= 1u )
			acc += __shfl_xor_sync(0xffffffffu, acc, off);
		if ( lane == 0u )
			output[(uint64_t)row * K3_INDEX_HEADS + head] = LmFloatToBf16(acc);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void K3IndexQExactSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t rows)
{
	constexpr uint32_t K = K3_Q_LORA_RANK, N = K3_INDEX_QUERY, CHUNKS = K / 8u, RG = 8u;
	__shared__ uint32_t s_rows[RG], s_cnt;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t col0 = (blockIdx.x * 4u + warp) * 2u;
	LmDependentWait();
	if ( col0 < N )
	{
		const uint4 *w0 = (const uint4 *)(weight + (uint64_t)col0 * K);
		const uint4 *w1 = w0 + CHUNKS;
		for ( uint32_t r_base = 0u; r_base < rows; r_base += RG )
		{
			if ( threadIdx.x == 0u )
			{
				uint32_t cnt = 0u;
				const uint32_t lim = rows - r_base < RG ? rows - r_base : RG;
				for ( uint32_t r = 0u; r < lim; ++r )
				{
					const uint32_t row = r_base + r;
					uint32_t bound = context_length[sequence_of_row[row]];
					const uint32_t pos = positions[row];
					if ( pos + 1u < bound )
						bound = pos + 1u;
					if ( bound > K3_INDEX_TOPK )
						s_rows[cnt++] = row;
				}
				s_cnt = cnt;
			}
			__syncthreads();
			const uint32_t cnt = s_cnt;
			if ( cnt == 1u )
			{
				const uint4 *a0 = (const uint4 *)(activation + (uint64_t)s_rows[0] * K);
				float acc0 = 0.0f, acc1 = 0.0f;
				for ( uint32_t c = lane; c < CHUNKS; c += 32u )
				{
					float wf0[8], wf1[8], af[8];
					LmSkinnyUnpackBf16x8(__ldcs(w0 + c), wf0);
					LmSkinnyUnpackBf16x8(__ldcs(w1 + c), wf1);
					LmSkinnyUnpackBf16x8(__ldg(a0 + c), af);
					acc0 = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf0, af), acc0);
					acc1 = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf1, af), acc1);
				}
				#pragma unroll
				for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				{
					acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
					acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
				}
				if ( lane == 0u )
					*(uint32_t *)(output + (uint64_t)s_rows[0] * N + col0) = LmPackBf16Pair(acc0, acc1);
			}
			else if ( cnt > 1u )
			{
				const uint4 *act_ptr[RG];
				float acc0[RG], acc1[RG];
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					act_ptr[r] = (const uint4 *)(activation + (uint64_t)(r < cnt ? s_rows[r] : s_rows[0]) * K);
					acc0[r] = 0.0f;
					acc1[r] = 0.0f;
				}
				for ( uint32_t c = lane; c < CHUNKS; c += 32u )
				{
					float wf0[8], wf1[8];
					LmSkinnyUnpackBf16x8(__ldcs(w0 + c), wf0);
					LmSkinnyUnpackBf16x8(__ldcs(w1 + c), wf1);
					#pragma unroll
					for ( uint32_t r = 0u; r < RG; ++r )
					{
						if ( r < cnt )
						{
							float af[8];
							LmSkinnyUnpackBf16x8(__ldg(act_ptr[r] + c), af);
							acc0[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf0, af), acc0[r]);
							acc1[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf1, af), acc1[r]);
						}
					}
				}
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < cnt )
					{
						#pragma unroll
						for ( uint32_t off = 16u; off > 0u; off >>= 1u )
						{
							acc0[r] += __shfl_xor_sync(0xffffffffu, acc0[r], off);
							acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], off);
						}
						if ( lane == 0u )
							*(uint32_t *)(output + (uint64_t)s_rows[r] * N + col0) = LmPackBf16Pair(acc0[r], acc1[r]);
					}
				}
			}
			if ( r_base + RG < rows )
				__syncthreads();
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(LM_STREAM_THREADS, 2)
static void K3IndexQMmaSkipKernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t rows)
{
	using Weight = LmStreamWeight<LmBf16Format>;
	constexpr uint32_t TILE_M = 16u, TILE_N = 64u, TILE_K = 64u, STAGES = 2u;
	constexpr uint32_t K = K3_Q_LORA_RANK, N = K3_INDEX_QUERY, K_TILES = K / TILE_K;
	constexpr uint32_t ROW_BYTES = TILE_K * 2u, ROW_CHUNKS = ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, B_CHUNKS = TILE_N * ROW_CHUNKS;
	constexpr uint32_t B_PER_THREAD = B_CHUNKS / LM_STREAM_THREADS;
	static_assert(K % TILE_K == 0u && N % TILE_N == 0u && A_CHUNKS <= LM_STREAM_THREADS && B_CHUNKS % LM_STREAM_THREADS == 0u,
		"K3 index_q geometry divides 16x64 skip MMA tile cleanly");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	__shared__ uint32_t s_any;
	const uint32_t row_base = blockIdx.y * TILE_M;
	const uint32_t row_limit = row_base + TILE_M < rows ? row_base + TILE_M : rows;
	LmDependentWait();
	if ( threadIdx.x < 32u )
	{
		const uint32_t r = row_base + threadIdx.x;
		uint32_t sparse = 0u;
		if ( r < row_limit )
		{
			uint32_t bound = context_length[sequence_of_row[r]];
			const uint32_t pos = positions[r];
			if ( pos + 1u < bound )
				bound = pos + 1u;
			sparse = bound > K3_INDEX_TOPK ? 1u : 0u;
		}
		const uint32_t mask = __ballot_sync(0xffffffffu, sparse != 0u);
		if ( threadIdx.x == 0u )
			s_any = mask;
	}
	__syncthreads();
	if ( s_any == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t neuron_base = blockIdx.x * TILE_N;
	const uint64_t weight_row_bytes = (uint64_t)K * 2u;
	const uint32_t b_row0 = threadIdx.x / ROW_CHUNKS, b_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t b_row_step = LM_STREAM_THREADS / ROW_CHUNKS;
	const uint64_t b_ptr_step = (uint64_t)b_row_step * weight_row_bytes;
	const uint32_t a_row = threadIdx.x / ROW_CHUNKS, a_col = threadIdx.x % ROW_CHUNKS;
	const bool a_valid = threadIdx.x < A_CHUNKS && row_base + a_row < row_limit;
	const uint16_t *a_src = activation + (uint64_t)(a_valid ? row_base + a_row : 0u) * K + a_col * 8u;
	const uint8_t *b_base = weight + (uint64_t)(neuron_base + b_row0) * weight_row_bytes + b_col0 * 16u;
	const uint32_t a_swiz = a_row * ROW_BYTES + LmStreamSwizzle8(a_row, a_col) * 16u;
	const uint32_t ldm_a_row = lane & 15u, ldm_a_col = (lane >> 4u) & 1u;
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		if ( threadIdx.x < A_CHUNKS )
			LmStreamCopy(stage + a_swiz, a_src, a_valid ? 16u : 0u);
		a_src += TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		{
			const uint32_t row = b_row0 + q * b_row_step;
			LmStreamCopy(stage + A_BYTES + Weight::template Offset<ROW_BYTES>(row, b_col0), b_base + q * b_ptr_step, 16u);
		}
		b_base += ROW_BYTES;
	};
	float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	issue_k(sp_off);
	LmStreamCommit();
	kp++;
	sp_off ^= STAGE_BYTES;
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		LmStreamWait<0u>();
		__syncthreads();
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			kp++;
			sp_off ^= STAGE_BYTES;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off;
		const uint32_t stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			uint32_t b[4], a0[4], a1[4];
			Weight::template Fragment<ROW_BYTES>(stage_b, warp * 8u, step, lane, b);
			LmStreamLdMatrix4(a0, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
			LmStreamLdMatrix4(a1, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
			const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
			LmMmaBf16(total, a0, b0);
			LmMmaBf16(total, a1, b1);
		}
		sc_off ^= STAGE_BYTES;
		if ( kt + 1u < K_TILES )
			__syncthreads();
	}
	const uint32_t col = neuron_base + warp * 8u + quad * 2u;
	if ( row_base + quad_row < row_limit )
		*(uint32_t *)(output + (uint64_t)(row_base + quad_row) * N + col) = LmPackBf16Pair(total[0], total[1]);
	if ( row_base + quad_row + 8u < row_limit )
		*(uint32_t *)(output + (uint64_t)(row_base + quad_row + 8u) * N + col) = LmPackBf16Pair(total[2], total[3]);
	LmDependentRelease();
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

template<uint32_t TILE_M>
__global__ __launch_bounds__(128u, TILE_M <= 32u ? 4 : 3)
static void K3SplitKMmaKernel(const __grid_constant__ K3SplitKArgs args)
{
	using Weight = LmStreamWeight<LmBf16Format>;
	constexpr uint32_t TILE_N = 32u, TILE_K = 64u, STAGES = 2u, THREADS = 128u;
	constexpr uint32_t ROW_BYTES = TILE_K * 2u, ROW_CHUNKS = ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, B_CHUNKS = TILE_N * ROW_CHUNKS;
	constexpr uint32_t A_PER_THREAD = A_CHUNKS / THREADS, B_PER_THREAD = B_CHUNKS / THREADS;
	constexpr uint32_t WM = TILE_M / 16u, WN = 4u / WM, N_PER_WARP = TILE_N / (WN * 8u);
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	LmDependentWait();
	const uint32_t tile_idx = blockIdx.x, sk = blockIdx.y;
	uint32_t part = 0u;
	if ( args.count > 1u && tile_idx >= args.t[1].first_tile )
		part = 1u;
	const K3SplitKTarget &tgt = args.t[part];
	const uint32_t neuron_base = (tile_idx - tgt.first_tile) * TILE_N;
	const uint32_t k_beg = args.k_tiles * sk / args.split_k;
	const uint32_t k_end = args.k_tiles * (sk + 1u) / args.split_k;
	const uint32_t my_k = k_end - k_beg;
	if ( my_k == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t warp_m = warp / WN, warp_n = warp % WN;
	const uint32_t quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t ldm_a_row = warp_m * 16u + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	const uint64_t weight_row_bytes = (uint64_t)args.input_dim * 2u;
	const uint32_t a_row0 = threadIdx.x / ROW_CHUNKS, a_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t a_row_step = THREADS / ROW_CHUNKS;
	const uint32_t b_row0 = threadIdx.x / ROW_CHUNKS, b_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t b_row_step = THREADS / ROW_CHUNKS;
	const uint16_t *a_base = args.activation + (uint64_t)k_beg * TILE_K + a_col0 * 8u;
	const uint8_t *b_base = tgt.weight + (uint64_t)k_beg * ROW_BYTES + b_col0 * 16u;
	auto issue_k = [&](uint32_t st_off, uint32_t kt_rel)
	{
		const uint32_t stage = shared_base + st_off;
		const uint32_t k_off_elem = kt_rel * TILE_K;
		const uint64_t k_off_byte = (uint64_t)kt_rel * ROW_BYTES;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
		{
			const uint32_t r = a_row0 + q * a_row_step;
			const bool ok = r < args.rows;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u,
				ok ? (a_base + (uint64_t)r * args.input_dim + k_off_elem) : args.activation, ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		{
			const uint32_t nr = b_row0 + q * b_row_step;
			const bool ok = neuron_base + nr < tgt.n;
			LmStreamCopy(stage + A_BYTES + Weight::template Offset<ROW_BYTES>(nr, b_col0),
				ok ? (b_base + (uint64_t)(neuron_base + nr) * weight_row_bytes + k_off_byte) : tgt.weight, ok ? 16u : 0u);
		}
	};
	float acc[N_PER_WARP][4];
	#pragma unroll
	for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
		acc[nf][0] = acc[nf][1] = acc[nf][2] = acc[nf][3] = 0.0f;
	uint32_t sp_off = 0u, sc_off = 0u;
	issue_k(sp_off, 0u);
	LmStreamCommit();
	sp_off ^= STAGE_BYTES;
	for ( uint32_t kt = 0u; kt < my_k; ++kt )
	{
		LmStreamWait<0u>();
		__syncthreads();
		if ( kt + 1u < my_k )
		{
			issue_k(sp_off, kt + 1u);
			sp_off ^= STAGE_BYTES;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off;
		const uint32_t stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			uint32_t a0[4], a1[4];
			LmStreamLdMatrix4(a0, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
			LmStreamLdMatrix4(a1, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
			{
				uint32_t b[4];
				Weight::template Fragment<ROW_BYTES>(stage_b, (warp_n * N_PER_WARP + nf) * 8u, step, lane, b);
				const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
				LmMmaBf16(acc[nf], a0, b0);
				LmMmaBf16(acc[nf], a1, b1);
			}
		}
		sc_off ^= STAGE_BYTES;
		if ( kt + 1u < my_k )
			__syncthreads();
	}
	const uint32_t r0 = warp_m * 16u + quad_row, r1 = r0 + 8u;
	const uint64_t sk_base = (uint64_t)sk * args.rows * args.total_cols;
	#pragma unroll
	for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
	{
		const uint32_t local_c = (warp_n * N_PER_WARP + nf) * 8u + quad * 2u;
		if ( neuron_base + local_c < tgt.n )
		{
			const uint32_t sc_col = tile_idx * TILE_N + local_c;
			if ( r0 < args.rows )
				*(float2 *)(args.scratch + sk_base + (uint64_t)r0 * args.total_cols + sc_col) = make_float2(acc[nf][0], acc[nf][1]);
			if ( r1 < args.rows )
				*(float2 *)(args.scratch + sk_base + (uint64_t)r1 * args.total_cols + sc_col) = make_float2(acc[nf][2], acc[nf][3]);
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 4)
static void K3SplitKReduceKernel(const __grid_constant__ K3SplitKArgs args)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y;
	const uint32_t col = (blockIdx.x * 256u + threadIdx.x) * 2u;
	if ( row < args.rows && col < args.total_cols )
	{
		const uint32_t tile_idx = col >> 5u;
		uint32_t part = 0u;
		if ( args.count > 1u && tile_idx >= args.t[1].first_tile )
			part = 1u;
		const K3SplitKTarget &tgt = args.t[part];
		const uint32_t local_c = col - (tgt.first_tile << 5u);
		if ( local_c < tgt.n )
		{
			float s0 = 0.0f, s1 = 0.0f;
			const uint64_t sk_stride = (uint64_t)args.rows * args.total_cols;
			const float *src = args.scratch + (uint64_t)row * args.total_cols + col;
			for ( uint32_t sk = 0u; sk < args.split_k; ++sk )
			{
				const float2 v = *(const float2 *)(src + (uint64_t)sk * sk_stride);
				s0 += v.x;
				s1 += v.y;
			}
			const uint64_t oidx = (uint64_t)row * tgt.stride + tgt.col_offset + local_c;
			if ( tgt.out_f32 != 0 )
				*(float2 *)(tgt.out_f32 + oidx) = make_float2(s0, s1);
			else
				*(uint32_t *)(tgt.out_bf16 + oidx) = LmPackBf16Pair(s0, s1);
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 4)
static void K3SplitKReduceSituKernel(const float *__restrict__ scratch, uint16_t *__restrict__ output_bf16,
	uint32_t rows, uint32_t channels, uint32_t split_k)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y;
	const uint32_t col = (blockIdx.x * 256u + threadIdx.x) * 2u;
	if ( row < rows && col < channels )
	{
		const uint32_t total_cols = channels * 2u;
		const uint64_t sk_stride = (uint64_t)rows * total_cols;
		const float *g_src = scratch + (uint64_t)row * total_cols + col;
		const float *u_src = g_src + channels;
		float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
		for ( uint32_t sk = 0u; sk < split_k; ++sk )
		{
			const float2 gv = *(const float2 *)(g_src + (uint64_t)sk * sk_stride);
			const float2 uv = *(const float2 *)(u_src + (uint64_t)sk * sk_stride);
			g0 += gv.x;
			g1 += gv.y;
			u0 += uv.x;
			u1 += uv.y;
		}
		const uint32_t gp = LmPackBf16Pair(g0, g1), up = LmPackBf16Pair(u0, u1);
		const float o0 = SparkMathSitu(LmBf16ToFloat((uint16_t)(gp & 0xffffu)), LmBf16ToFloat((uint16_t)(up & 0xffffu)), K3_SITU_BETA, K3_SITU_LINEAR_BETA);
		const float o1 = SparkMathSitu(LmBf16ToFloat((uint16_t)(gp >> 16u)), LmBf16ToFloat((uint16_t)(up >> 16u)), K3_SITU_BETA, K3_SITU_LINEAR_BETA);
		*(uint32_t *)(output_bf16 + (uint64_t)row * channels + col) = LmPackBf16Pair(o0, o1);
	}
	LmDependentRelease();
}

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
	if ( rows <= 32u )
		LM_LAUNCH_DEPENDENT((K3SplitKMmaKernel<32u>), grid, 128u, 0u, stream, args);
	else
		LM_LAUNCH_DEPENDENT((K3SplitKMmaKernel<64u>), grid, 128u, 0u, stream, args);
	if ( situ_out != 0 )
	{
		const uint32_t ch = targets[0].n / 2u;
		LM_LAUNCH_DEPENDENT((K3SplitKReduceSituKernel), dim3(((ch / 2u) + 255u) / 256u, rows), 256u, 0u, stream,
			scratch, situ_out, rows, ch, split_k);
	}
	else
		LM_LAUNCH_DEPENDENT((K3SplitKReduceKernel), dim3(((args.total_cols / 2u) + 255u) / 256u, rows), 256u, 0u, stream, args);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
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

static __device__ __forceinline__ void K3KdaProjStorePair(const K3KdaProjMmaArgs &args, uint32_t part, uint32_t row, uint32_t col, uint32_t pair)
{
	if ( part == 0u )
	{
		if ( col < args.rank_qk )
		{
			const uint64_t idx = (uint64_t)row * args.rank_qk + col;
			*(uint32_t *)(args.q + idx) = pair;
			if ( args.rq != 0 )
				*(uint32_t *)(args.rq + idx) = pair;
		}
		else if ( col < 2u * args.rank_qk )
		{
			const uint64_t idx = (uint64_t)row * args.rank_qk + (col - args.rank_qk);
			*(uint32_t *)(args.k + idx) = pair;
			if ( args.rk != 0 )
				*(uint32_t *)(args.rk + idx) = pair;
		}
		else if ( col < 2u * args.rank_qk + args.rank_v )
		{
			const uint64_t idx = (uint64_t)row * args.rank_v + (col - 2u * args.rank_qk);
			*(uint32_t *)(args.v + idx) = pair;
			if ( args.rv != 0 )
				*(uint32_t *)(args.rv + idx) = pair;
		}
		else if ( col < args.qkvb_n )
			*(uint32_t *)(args.beta + (uint64_t)row * args.rank_heads + (col - 2u * args.rank_qk - args.rank_v)) = pair;
	}
	else if ( part == 1u )
	{
		if ( col < args.gate_n )
			*(uint32_t *)(args.gate + (uint64_t)row * args.gate_n + col) = pair;
	}
	else if ( col < args.decay_n )
		*(uint32_t *)(args.latent + (uint64_t)row * args.decay_n + col) = pair;
}

template<uint32_t TILE_M>
__global__ __launch_bounds__(128u, TILE_M <= 32u ? 4 : 3)
static void K3WideMmaKernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
	using Weight = LmStreamWeight<LmBf16Format>;
	constexpr uint32_t TILE_N = 32u, TILE_K = 64u, STAGES = 2u, THREADS = 128u;
	constexpr uint32_t ROW_BYTES = TILE_K * 2u, ROW_CHUNKS = ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, B_CHUNKS = TILE_N * ROW_CHUNKS;
	constexpr uint32_t A_PER_THREAD = A_CHUNKS / THREADS, B_PER_THREAD = B_CHUNKS / THREADS;
	constexpr uint32_t WM = TILE_M / 16u, WN = 4u / WM, N_PER_WARP = TILE_N / (WN * 8u);
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	LmDependentWait();
	const uint32_t neuron_base = blockIdx.x * TILE_N;
	const uint32_t k_tiles = (input_dim + TILE_K - 1u) / TILE_K;
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t warp_m = warp / WN, warp_n = warp % WN;
	const uint32_t quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t ldm_a_row = warp_m * 16u + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	const uint64_t weight_row_bytes = (uint64_t)input_dim * 2u;
	const uint32_t a_row0 = threadIdx.x / ROW_CHUNKS, a_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t a_row_step = THREADS / ROW_CHUNKS;
	const uint32_t b_row0 = threadIdx.x / ROW_CHUNKS, b_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t b_row_step = THREADS / ROW_CHUNKS;
	auto issue_k = [&](uint32_t st_off, uint32_t kt)
	{
		const uint32_t stage = shared_base + st_off;
		const uint32_t k_elem = kt * TILE_K + a_col0 * 8u;
		const uint64_t k_byte = (uint64_t)kt * ROW_BYTES + b_col0 * 16u;
		const bool k_ok_a = k_elem < input_dim;
		const bool k_ok_b = k_byte < weight_row_bytes;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
		{
			const uint32_t r = a_row0 + q * a_row_step;
			const bool ok = k_ok_a && r < rows;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u,
				ok ? (activation + (uint64_t)r * input_dim + k_elem) : activation, ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		{
			const uint32_t nr = b_row0 + q * b_row_step;
			const bool ok = k_ok_b && neuron_base + nr < output_dim;
			LmStreamCopy(stage + A_BYTES + Weight::template Offset<ROW_BYTES>(nr, b_col0),
				ok ? (weight + (uint64_t)(neuron_base + nr) * weight_row_bytes + k_byte) : weight, ok ? 16u : 0u);
		}
	};
	float acc[N_PER_WARP][4];
	#pragma unroll
	for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
		acc[nf][0] = acc[nf][1] = acc[nf][2] = acc[nf][3] = 0.0f;
	uint32_t sp_off = 0u, sc_off = 0u;
	issue_k(sp_off, 0u);
	LmStreamCommit();
	sp_off ^= STAGE_BYTES;
	for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
	{
		LmStreamWait<0u>();
		__syncthreads();
		if ( kt + 1u < k_tiles )
		{
			issue_k(sp_off, kt + 1u);
			sp_off ^= STAGE_BYTES;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off;
		const uint32_t stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			uint32_t a0[4], a1[4];
			LmStreamLdMatrix4(a0, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
			LmStreamLdMatrix4(a1, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
			{
				uint32_t b[4];
				Weight::template Fragment<ROW_BYTES>(stage_b, (warp_n * N_PER_WARP + nf) * 8u, step, lane, b);
				const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
				LmMmaBf16(acc[nf], a0, b0);
				LmMmaBf16(acc[nf], a1, b1);
			}
		}
		sc_off ^= STAGE_BYTES;
		if ( kt + 1u < k_tiles )
			__syncthreads();
	}
	const uint32_t r0 = warp_m * 16u + quad_row, r1 = r0 + 8u;
	#pragma unroll
	for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
	{
		const uint32_t col = neuron_base + (warp_n * N_PER_WARP + nf) * 8u + quad * 2u;
		if ( col < output_dim )
		{
			if ( r0 < rows )
				*(uint32_t *)(output_bf16 + (uint64_t)r0 * output_dim + col) = LmPackBf16Pair(acc[nf][0], acc[nf][1]);
			if ( r1 < rows )
				*(uint32_t *)(output_bf16 + (uint64_t)r1 * output_dim + col) = LmPackBf16Pair(acc[nf][2], acc[nf][3]);
		}
	}
	LmDependentRelease();
}

static int32_t K3WideMmaLaunch(const void *weight, const uint16_t *activation, uint16_t *output_bf16,
	uint32_t rows, uint32_t input_dim, uint32_t output_dim, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)output_dim * input_dim * sizeof(uint16_t));
	const uint32_t tiles = (output_dim + 31u) / 32u;
	if ( rows <= 32u )
		LM_LAUNCH_DEPENDENT((K3WideMmaKernel<32u>), tiles, 128u, 0u, stream,
			(const uint8_t *)weight, activation, output_bf16, rows, input_dim, output_dim);
	else
		LM_LAUNCH_DEPENDENT((K3WideMmaKernel<64u>), tiles, 128u, 0u, stream,
			(const uint8_t *)weight, activation, output_bf16, rows, input_dim, output_dim);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t TILE_M>
__global__ __launch_bounds__(128u, TILE_M <= 32u ? 4 : 3)
static void K3KdaProjMmaKernel(const __grid_constant__ K3KdaProjMmaArgs args)
{
	using Weight = LmStreamWeight<LmBf16Format>;
	constexpr uint32_t TILE_N = 32u, TILE_K = 64u, STAGES = 2u, THREADS = 128u;
	constexpr uint32_t K = K3_HIDDEN, K_TILES = K / TILE_K;
	constexpr uint32_t ROW_BYTES = TILE_K * 2u, ROW_CHUNKS = ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, B_CHUNKS = TILE_N * ROW_CHUNKS;
	constexpr uint32_t A_PER_THREAD = A_CHUNKS / THREADS, B_PER_THREAD = B_CHUNKS / THREADS;
	constexpr uint32_t WM = TILE_M / 16u, WN = 4u / WM, N_PER_WARP = TILE_N / (WN * 8u);
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	LmDependentWait();
	const uint32_t tile_idx = blockIdx.x;
	uint32_t part, local_tile, part_n;
	const uint8_t *w_ptr;
	if ( tile_idx < args.tiles_qkvb )
	{
		part = 0u;
		local_tile = tile_idx;
		part_n = args.qkvb_n;
		w_ptr = args.w_qkvb;
	}
	else if ( tile_idx < args.tiles_qkvb + args.tiles_gate )
	{
		part = 1u;
		local_tile = tile_idx - args.tiles_qkvb;
		part_n = args.gate_n;
		w_ptr = args.w_gate;
	}
	else
	{
		part = 2u;
		local_tile = tile_idx - args.tiles_qkvb - args.tiles_gate;
		part_n = args.decay_n;
		w_ptr = args.w_decay;
	}
	const uint32_t neuron_base = local_tile * TILE_N;
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t warp_m = warp / WN, warp_n = warp % WN;
	const uint32_t quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t ldm_a_row = warp_m * 16u + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	const uint64_t weight_row_bytes = (uint64_t)K * 2u;
	const uint32_t a_row0 = threadIdx.x / ROW_CHUNKS, a_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t a_row_step = THREADS / ROW_CHUNKS;
	const uint32_t b_row0 = threadIdx.x / ROW_CHUNKS, b_col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t b_row_step = THREADS / ROW_CHUNKS;
	auto issue_k = [&](uint32_t st_off, uint32_t kt)
	{
		const uint32_t stage = shared_base + st_off;
		const uint32_t k_off_elem = kt * TILE_K;
		const uint64_t k_off_byte = (uint64_t)kt * ROW_BYTES;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
		{
			const uint32_t r = a_row0 + q * a_row_step;
			const bool ok = r < args.rows;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, a_col0) * 16u,
				ok ? (args.activation + (uint64_t)r * K + k_off_elem + a_col0 * 8u) : args.activation, ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		{
			const uint32_t nr = b_row0 + q * b_row_step;
			const bool ok = neuron_base + nr < part_n;
			LmStreamCopy(stage + A_BYTES + Weight::template Offset<ROW_BYTES>(nr, b_col0),
				ok ? (w_ptr + (uint64_t)(neuron_base + nr) * weight_row_bytes + k_off_byte + b_col0 * 16u) : w_ptr, ok ? 16u : 0u);
		}
	};
	float acc[N_PER_WARP][4];
	#pragma unroll
	for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
		acc[nf][0] = acc[nf][1] = acc[nf][2] = acc[nf][3] = 0.0f;
	uint32_t sp_off = 0u, sc_off = 0u;
	issue_k(sp_off, 0u);
	LmStreamCommit();
	sp_off ^= STAGE_BYTES;
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		LmStreamWait<0u>();
		__syncthreads();
		if ( kt + 1u < K_TILES )
		{
			issue_k(sp_off, kt + 1u);
			sp_off ^= STAGE_BYTES;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off;
		const uint32_t stage_b = stage_a + A_BYTES;
		#pragma unroll
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			uint32_t a0[4], a1[4];
			LmStreamLdMatrix4(a0, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
			LmStreamLdMatrix4(a1, stage_a + ldm_a_row * ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
			{
				uint32_t b[4];
				Weight::template Fragment<ROW_BYTES>(stage_b, (warp_n * N_PER_WARP + nf) * 8u, step, lane, b);
				const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
				LmMmaBf16(acc[nf], a0, b0);
				LmMmaBf16(acc[nf], a1, b1);
			}
		}
		sc_off ^= STAGE_BYTES;
		if ( kt + 1u < K_TILES )
			__syncthreads();
	}
	const uint32_t r0 = warp_m * 16u + quad_row, r1 = r0 + 8u;
	#pragma unroll
	for ( uint32_t nf = 0u; nf < N_PER_WARP; ++nf )
	{
		const uint32_t col = neuron_base + (warp_n * N_PER_WARP + nf) * 8u + quad * 2u;
		if ( r0 < args.rows )
			K3KdaProjStorePair(args, part, r0, col, LmPackBf16Pair(acc[nf][0], acc[nf][1]));
		if ( r1 < args.rows )
			K3KdaProjStorePair(args, part, r1, col, LmPackBf16Pair(acc[nf][2], acc[nf][3]));
	}
	LmDependentRelease();
}

template<uint32_t RG>
__global__ __launch_bounds__(256u, 2)
static void K3SharedW1SituSkinnyKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t channels)
{
	constexpr uint32_t SW = 8u, CHUNKS = K3_HIDDEN / 8u;
	__shared__ float partial[SW][2u][RG];
	const uint32_t c0 = blockIdx.x, base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u || c0 >= channels )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t first = CHUNKS * warp / SW, last = CHUNKS * (warp + 1u) / SW;
	const uint4 *wg = (const uint4 *)(weight + (uint64_t)c0 * K3_HIDDEN);
	const uint4 *wu = (const uint4 *)(weight + (uint64_t)(channels + c0) * K3_HIDDEN);
	float acc_g[RG], acc_u[RG];
	#pragma unroll
	for ( uint32_t r = 0u; r < RG; ++r )
		acc_g[r] = acc_u[r] = 0.0f;
	for ( uint32_t c = first + sub; c < last; c += 32u )
	{
		float wgf[8], wuf[8];
		LmSkinnyUnpackBf16x8(__ldcs(wg + c), wgf);
		LmSkinnyUnpackBf16x8(__ldcs(wu + c), wuf);
		const uint16_t *act_base = activation + (uint64_t)base_row * K3_HIDDEN + c * 8u;
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
		{
			if ( r < count )
			{
				float af[8];
				LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act_base + (uint64_t)r * K3_HIDDEN)), af);
				acc_g[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wgf, af), acc_g[r]);
				acc_u[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wuf, af), acc_u[r]);
			}
		}
	}
	#pragma unroll
	for ( uint32_t off = 16u; off > 0u; off >>= 1u )
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
		{
			acc_g[r] += __shfl_xor_sync(0xffffffffu, acc_g[r], off);
			acc_u[r] += __shfl_xor_sync(0xffffffffu, acc_u[r], off);
		}
	if ( sub == 0u )
	{
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
		{
			partial[warp][0][r] = acc_g[r];
			partial[warp][1][r] = acc_u[r];
		}
	}
	__syncthreads();
	if ( threadIdx.x < count )
	{
		const uint32_t r = threadIdx.x;
		float g = 0.0f, u = 0.0f;
		#pragma unroll
		for ( uint32_t w = 0u; w < SW; ++w )
		{
			g += partial[w][0][r];
			u += partial[w][1][r];
		}
		const float out = SparkMathSitu(LmBf16ToFloat(LmFloatToBf16(g)), LmBf16ToFloat(LmFloatToBf16(u)), K3_SITU_BETA, K3_SITU_LINEAR_BETA);
		output_bf16[(uint64_t)(base_row + r) * channels + c0] = LmFloatToBf16(out);
	}
	LmDependentRelease();
}

template<uint32_t RG>
__global__ __launch_bounds__(256u, 2)
static void K3KdaNormGateOutKernel(const uint16_t *__restrict__ mixed, const uint16_t *__restrict__ gate,
	const float *__restrict__ norm_weight, const uint16_t *__restrict__ out_weight,
	uint16_t *__restrict__ output, uint32_t rows, float epsilon)
{
	constexpr uint32_t HEADS = 6u, HD = K3_KDA_VALUE_DIM, K = HEADS * HD, CHUNKS = K / 8u;
	__shared__ __align__(16) uint16_t s_in[RG][K];
	__shared__ float s_wsum[8u];
	const uint32_t base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t half_id = threadIdx.x >> 7u, within = threadIdx.x & 127u;
	const uint32_t warp_in_half = within >> 5u, lane = threadIdx.x & 31u;
	const float nw = __ldg(norm_weight + within);
	for ( uint32_t r = 0u; r < count; ++r )
	{
		const uint32_t row = base_row + r;
		#pragma unroll
		for ( uint32_t h0 = 0u; h0 < HEADS; h0 += 2u )
		{
			const uint32_t h = h0 + half_id;
			const uint64_t idx = ((uint64_t)row * HEADS + h) * HD + within;
			const float v = LmBf16ToFloat(__ldg(mixed + idx));
			float sq = v * v;
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				sq += __shfl_xor_sync(0xffffffffu, sq, off);
			if ( lane == 0u )
				s_wsum[half_id * 4u + warp_in_half] = sq;
			__syncthreads();
			const float sum = s_wsum[half_id * 4u] + s_wsum[half_id * 4u + 1u] + s_wsum[half_id * 4u + 2u] + s_wsum[half_id * 4u + 3u];
			const float scale = SparkMathRmsScale(sum, (float)HD, epsilon);
			const float normed = LmBf16ToFloat(LmFloatToBf16(v * scale * nw));
			const float g = LmBf16ToFloat(__ldg(gate + idx));
			s_in[r][h * HD + within] = LmFloatToBf16(normed * SparkMathSigmoidFast(g));
			if ( h0 + 2u < HEADS || r + 1u < count )
				__syncthreads();
		}
	}
	__syncthreads();
	const uint32_t warp = threadIdx.x >> 5u;
	#pragma unroll
	for ( uint32_t pass = 0u; pass < 4u; ++pass )
	{
		const uint32_t col = (blockIdx.x * 4u + pass) * 8u + warp;
		if ( col < K3_HIDDEN )
		{
			const uint4 *w0 = (const uint4 *)(out_weight + (uint64_t)col * K);
			float acc[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[r] = 0.0f;
			#pragma unroll
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				float wf[8];
				LmSkinnyUnpackBf16x8(__ldcs(w0 + c), wf);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < count )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(((const uint4 *)s_in[r])[c], af);
						acc[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf, af), acc[r]);
					}
				}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				if ( r < count )
				{
					#pragma unroll
					for ( uint32_t off = 16u; off > 0u; off >>= 1u )
						acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
					if ( lane == 0u )
						output[(uint64_t)(base_row + r) * K3_HIDDEN + col] = LmFloatToBf16(acc[r]);
				}
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t RG>
__global__ __launch_bounds__(256u, 2)
static void K3MlaGateOutKernel(const uint16_t *__restrict__ value, const uint16_t *__restrict__ gate,
	const uint16_t *__restrict__ out_weight, uint16_t *__restrict__ output, uint32_t rows)
{
	constexpr uint32_t K = 6u * K3_V_HEAD_DIM, CHUNKS = K / 8u;
	__shared__ __align__(16) uint16_t s_in[RG][K];
	const uint32_t base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	for ( uint32_t r = 0u; r < count; ++r )
	{
		const uint64_t rbase = (uint64_t)(base_row + r) * K;
		for ( uint32_t i = threadIdx.x; i < K; i += 256u )
			s_in[r][i] = LmFloatToBf16(LmBf16ToFloat(__ldg(value + rbase + i)) * SparkMathSigmoidFast(LmBf16ToFloat(__ldg(gate + rbase + i))));
	}
	__syncthreads();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	#pragma unroll
	for ( uint32_t pass = 0u; pass < 4u; ++pass )
	{
		const uint32_t col = (blockIdx.x * 4u + pass) * 8u + warp;
		if ( col < K3_HIDDEN )
		{
			const uint4 *w0 = (const uint4 *)(out_weight + (uint64_t)col * K);
			float acc[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[r] = 0.0f;
			#pragma unroll
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				float wf[8];
				LmSkinnyUnpackBf16x8(__ldcs(w0 + c), wf);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < count )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(((const uint4 *)s_in[r])[c], af);
						acc[r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf, af), acc[r]);
					}
				}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				if ( r < count )
				{
					#pragma unroll
					for ( uint32_t off = 16u; off > 0u; off >>= 1u )
						acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
					if ( lane == 0u )
						output[(uint64_t)(base_row + r) * K3_HIDDEN + col] = LmFloatToBf16(acc[r]);
				}
			}
		}
	}
	LmDependentRelease();
}
