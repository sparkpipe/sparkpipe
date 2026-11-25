#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_mimo26_family.h"
#include "sparkpipe/spark_gpu.h"

#define MM_THREADS 256u
#define MM_FP8_GROUP 128u

template<class G>
struct MimoShape : MimoLayout<G>
{
	using SlidingKv = LmKvGeometry<G::kSlidingKvHeads * (G::kHeadDim + G::kValueDim) * 2u, G::kPageSlots, true>;
	using FullKv = LmKvGeometry<G::kFullKvHeads * (G::kHeadDim + G::kValueDim) * 2u, G::kPageSlots, true>;
	static_assert(MimoLayout<G>::kMaxFused / G::kTpDegree <= G::kHidden, "a rank's fused rows fit one collective row");
	static_assert(G::kHidden % MM_FP8_GROUP == 0u && MimoLayout<G>::kDenseRows % MM_FP8_GROUP == 0u, "FP8 inputs are whole 128-wide scale groups");
	static_assert(G::kHidden % 32u == 0u && G::kMoeInter % 32u == 0u, "MXFP4 inputs are whole 32-wide scale groups");
	static_assert(LM_SKINNY_ROWS_WIDE * G::kTopK <= LM_SKINNY_GROUPED_MAX_MEAN_ROWS * G::kLocalExperts,
		"a 16-row verify chunk keeps its routes within the grouped skinny expert kernel");
};

typedef struct MimoScaled
{
	const uint8_t *weight;
	const void *scale;
} MimoScaled;

typedef struct MimoLayer
{
	const uint16_t *input_norm;
	const uint16_t *post_attn_norm;
	MimoScaled qkv;
	const uint16_t *out;
	const float *sinks;
	MimoScaled ffn_gate_up;
	MimoScaled ffn_down;
	const uint16_t *router;
	const float *router_bias;
	MimoScaled moe_w1;
	MimoScaled moe_w3;
	MimoScaled moe_w2;
	uint32_t full;
	uint32_t kv_index;
} MimoLayer;

template<class G>
struct MimoState : CudaStagePackState
{
	uint32_t expert_first;
	uint64_t record_stride;
	MimoLayer layers[G::kLayers];
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *query;
	float *send;
	float *received;
	uint16_t *merged;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *sliding_inv_freq;
	float *full_inv_freq;
	float *router_logits;
	uint32_t *route_expert;
	uint32_t *route_global;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	uint16_t *expert_gate;
	uint16_t *expert_up;
	uint16_t *expert_act;
	uint16_t *expert_out;
	LmKvShardView sliding_views[MimoShape<G>::kSlidingLayers];
	LmKvShardView full_views[G::kFullLayers];
};

template<uint32_t LOCAL_V>
static __device__ __forceinline__ uint4 MimoLoadRankMajor8Const(const uint4 *__restrict__ gathered4, uint32_t rows, uint32_t row, uint32_t logical_v)
{
	const uint32_t rank = logical_v / LOCAL_V;
	const uint32_t elem = logical_v % LOCAL_V;
	return __ldg(gathered4 + (((uint64_t)rank * rows) + row) * LOCAL_V + elem);
}

template<uint32_t THREADS, uint32_t WIDTH, bool HAS_ADD>
__global__ __launch_bounds__(THREADS, 4)
static void MimoAddRmsNormKernel(uint16_t *__restrict__ hidden_bf16, const uint16_t *__restrict__ partial_bf16,
	const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, float epsilon)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= THREADS * LM_RMS_NORM_STAGED, "hidden width fits in staged vectorized waves");
	constexpr uint32_t VEC_SLOTS = (WIDTH + THREADS * 8u - 1u) / (THREADS * 8u);
	__shared__ float reduction[LmBlockWarps<THREADS>()], sq_wave[THREADS * 8u];
	const uint64_t base = (uint64_t)blockIdx.x * WIDTH;
	uint16_t *hid_row = hidden_bf16 + base;
	const uint16_t *part_row = HAS_ADD ? partial_bf16 + base : (const uint16_t *)0;
	uint16_t *out_row = output_bf16 + base;
	uint4 staged[VEC_SLOTS];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t wave_base = slot * THREADS * 8u;
		const uint32_t offset = wave_base + threadIdx.x * 8u;
		if ( offset < WIDTH )
		{
			float vals[8], sq[8];
			uint4 v = *(const uint4 *)(hid_row + offset);
			LmLoadBf16x8AsFloat(v, vals);
			if constexpr ( HAS_ADD )
			{
				float p[8];
				LmLoadBf16x8AsFloat(*(const uint4 *)(part_row + offset), p);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					vals[k] += p[k];
				v = LmPackUint4FloatToBf16(vals);
				*(uint4 *)(hid_row + offset) = v;
				LmLoadBf16x8AsFloat(v, vals);
			}
			staged[slot] = v;
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				sq[k] = vals[k] * vals[k];
			LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
		}
		else
			staged[slot] = make_uint4(0u, 0u, 0u, 0u);
		__syncthreads();
		#pragma unroll
		for ( uint32_t s = 0u; s < 8u; ++s )
			if ( wave_base + s * THREADS + threadIdx.x < WIDTH )
				total += sq_wave[s * THREADS + threadIdx.x];
		if ( slot + 1u < VEC_SLOTS )
			__syncthreads();
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)WIDTH, epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < WIDTH )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(staged[slot], x);
			LmLoadWeight8(weight_bf16, offset, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathNormGainRounded(x[k] * scale, w[k]);
			*(uint4 *)(out_row + offset) = LmPackUint4FloatToBf16(out);
		}
	}
	LmDependentRelease();
}

template<class G, bool FULL>
__global__ __launch_bounds__(MM_THREADS, 2)
static void MimoAttnPrepStoreKernel(const uint16_t *__restrict__ gathered, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row, const float *__restrict__ inv_freq,
	LmKvShardView kv_view, uint16_t *__restrict__ query_out, uint32_t rows)
{
	using S = MimoShape<G>;
	using Kv = typename std::conditional<FULL, typename S::FullKv, typename S::SlidingKv>::type;
	constexpr uint32_t KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
	constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
	constexpr uint32_t LOCAL_V = FUSED / (G::kTpDegree * 8u);
	constexpr uint32_t HD_V = G::kHeadDim >> 3u;
	constexpr uint32_t VD_V = G::kValueDim >> 3u;
	constexpr uint32_t ROPE_V = G::kRopeDim >> 3u;
	constexpr uint32_t HALF = G::kRopeDim / 2u;
	constexpr uint32_t Q_V = S::kQuery >> 3u;
	constexpr uint32_t K_V = (FULL ? S::kFullKey : S::kSlidingKey) >> 3u;
	constexpr uint32_t WARPS = MM_THREADS / LM_WARP_LANES;
	constexpr uint32_t TASKS = G::kHeads + 2u * KVH;
	static_assert(HD_V <= 32u && VD_V <= 32u && (G::kHeadDim & 7u) == 0u && (G::kValueDim & 7u) == 0u && (G::kRopeDim & 7u) == 0u,
		"one warp covers a full query, key or value head in 16-byte vectors");
	__shared__ __align__(16) uint16_t rope_smem[WARPS][G::kRopeDim];
	__shared__ float s_sin[HALF], s_cos[HALF];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	if ( row < rows )
	{
		const float pos = (float)positions[row];
		if ( threadIdx.x < HALF )
		{
			const float angle = pos * __ldg(inv_freq + threadIdx.x);
			s_sin[threadIdx.x] = SparkMathRopeSin(angle, 1.0f);
			s_cos[threadIdx.x] = SparkMathRopeCos(angle, 1.0f);
		}
		__syncthreads();
		const uint4 *gathered4 = (const uint4 *)gathered;
		const uint32_t store_pos = store_positions[row];
		const bool owns = SparkKvShardOwns(kv_view.shard, store_pos) != 0u;
		uint16_t *slot = 0;
		bool slot_resolved = false;
		const uint32_t step = gridDim.x * WARPS;
		for ( uint32_t task = blockIdx.x * WARPS + warp; task < TASKS; task += step )
		{
			if ( task < G::kHeads )
			{
				const uint32_t h = task;
				uint4 v = make_uint4(0u, 0u, 0u, 0u);
				if ( lane < HD_V )
				{
					v = MimoLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, h * HD_V + lane);
					if ( lane < ROPE_V )
						*(uint4 *)(rope_smem[warp] + (lane << 3u)) = v;
				}
				__syncwarp();
				for ( uint32_t pair = lane; pair < HALF; pair += 32u )
				{
					const float sv = s_sin[pair], cv = s_cos[pair];
					const float x0 = LmBf16ToFloat(rope_smem[warp][pair]);
					const float x1 = LmBf16ToFloat(rope_smem[warp][pair + HALF]);
					rope_smem[warp][pair] = LmFloatToBf16(SparkMathRopeRoundedTerm(x0, -x1, cv, sv));
					rope_smem[warp][pair + HALF] = LmFloatToBf16(SparkMathRopeRoundedTerm(x1, x0, cv, sv));
				}
				__syncwarp();
				if ( lane < HD_V )
				{
					if ( lane < ROPE_V )
						v = *(const uint4 *)(rope_smem[warp] + (lane << 3u));
					((uint4 *)query_out)[((uint64_t)row * G::kHeads + h) * HD_V + lane] = v;
				}
			}
			else if ( owns )
			{
				if ( !slot_resolved )
				{
					slot = (uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
					slot_resolved = true;
				}
				if ( slot != 0 )
				{
					if ( task < G::kHeads + KVH )
					{
						const uint32_t kh = task - G::kHeads;
						uint4 v = make_uint4(0u, 0u, 0u, 0u);
						if ( lane < HD_V )
						{
							v = MimoLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, Q_V + kh * HD_V + lane);
							if ( lane < ROPE_V )
								*(uint4 *)(rope_smem[warp] + (lane << 3u)) = v;
						}
						__syncwarp();
						for ( uint32_t pair = lane; pair < HALF; pair += 32u )
						{
							const float sv = s_sin[pair], cv = s_cos[pair];
							const float x0 = LmBf16ToFloat(rope_smem[warp][pair]);
							const float x1 = LmBf16ToFloat(rope_smem[warp][pair + HALF]);
							rope_smem[warp][pair] = LmFloatToBf16(SparkMathRopeRoundedTerm(x0, -x1, cv, sv));
							rope_smem[warp][pair + HALF] = LmFloatToBf16(SparkMathRopeRoundedTerm(x1, x0, cv, sv));
						}
						__syncwarp();
						if ( lane < HD_V )
						{
							if ( lane < ROPE_V )
								v = *(const uint4 *)(rope_smem[warp] + (lane << 3u));
							((uint4 *)slot)[kh * HD_V + lane] = v;
						}
					}
					else
					{
						const uint32_t vh = task - G::kHeads - KVH;
						if ( lane < VD_V )
						{
							const uint4 raw = MimoLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, Q_V + K_V + vh * VD_V + lane);
							float vf[8];
							LmLoadBf16x8AsFloat(raw, vf);
							#pragma unroll
							for ( uint32_t k = 0u; k < 8u; ++k )
								vf[k] *= G::kValueScale;
							((uint4 *)slot)[K_V + vh * VD_V + lane] = LmPackUint4FloatToBf16(vf);
						}
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__((G::kHeads / G::kTpDegree) * 32u, 4)
static void MimoGqaShardMergeKernel(const float *__restrict__ received, uint64_t source_stride, uint32_t rank, uint16_t *__restrict__ output_bf16,
	const float *__restrict__ sinks)
{
	constexpr uint32_t SOURCES = G::kTpDegree;
	constexpr uint32_t RANK_HEADS = G::kHeads / SOURCES;
	constexpr uint32_t VD = G::kValueDim;
	constexpr uint32_t WIDTH = RANK_HEADS * VD;
	constexpr uint32_t RECORD = WIDTH + 2u * (RANK_HEADS + 1u);
	static_assert(VD == 128u, "one warp covers one 128-wide head with 4 floats per lane");
	__shared__ float scales[RANK_HEADS][SOURCES];
	__shared__ float denom_shared[RANK_HEADS];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t origin = rank * RANK_HEADS;
	const float *row_base = received + (uint64_t)row * RECORD;
	uint16_t *out_row = output_bf16 + (uint64_t)row * WIDTH;
	LmDependentWait();
	{
		const float sink = sinks != 0 ? __ldg(sinks + origin + warp) : -INFINITY;
		float gm = -INFINITY;
		float den = LmWarpMergeScales(row_base + WIDTH + 2u * warp, source_stride, SOURCES, scales[warp], &gm, sink);
		if ( lane == 0u )
		{
			if ( sinks != 0 )
				den += __expf(sink - gm);
			denom_shared[warp] = fmaxf(den, 1.0e-20f);
		}
	}
	__syncthreads();
	{
		const uint32_t v2 = (warp * 64u) + lane * 2u;
		const float d = denom_shared[warp];
		float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
		#pragma unroll 4
		for ( uint32_t s = 0u; s < SOURCES; ++s )
		{
			const float w = scales[warp][s];
			const float2 *src2 = (const float2 *)(row_base + (uint64_t)s * source_stride);
			const float2 x0 = __ldg(src2 + v2);
			const float2 x1 = __ldg(src2 + v2 + 1u);
			m0 = fmaf(x0.x, w, m0);
			m1 = fmaf(x0.y, w, m1);
			m2 = fmaf(x1.x, w, m2);
			m3 = fmaf(x1.y, w, m3);
		}
		((uint2 *)out_row)[threadIdx.x] = make_uint2(LmPackBf16Pair(m0 / d, m1 / d), LmPackBf16Pair(m2 / d, m3 / d));
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void MimoDenseSiluMulKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width)
{
	const uint64_t base = (uint64_t)blockIdx.y * 2u * width;
	LmDependentWait();
	LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, LmStridedGateUpOp<true>{});
	LmDependentRelease();
}

template<uint32_t EXPERTS, uint32_t TOPK>
static __device__ __forceinline__ void MimoSigmoidTopkWarp(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t row,
	uint32_t expert_first, uint32_t local_experts, uint32_t *__restrict__ fault, uint32_t *__restrict__ out_exp, float *__restrict__ out_wt,
	uint32_t lane)
{
	constexpr uint32_t PER_LANE = (EXPERTS + 31u) / 32u;
	float sc[PER_LANE];
	unsigned long long key[PER_LANE];
	const uint64_t row_base = (uint64_t)row * EXPERTS;
	#pragma unroll
	for ( uint32_t j = 0u; j < PER_LANE; ++j )
	{
		const uint32_t e = lane + j * 32u;
		if ( e < EXPERTS )
		{
			const float s = SparkMathSigmoid(logits[row_base + e]);
			const float sel = s + (bias != 0 ? bias[e] : 0.0f);
			sc[j] = s;
			key[j] = ((unsigned long long)LmTopkRouteKey(sel + 0.0f) << 32u) | (0xffffffffu - e);
		}
		else
		{
			sc[j] = 0.0f;
			key[j] = 0ull;
		}
	}
	float total = 0.0f;
	#pragma unroll
	for ( uint32_t pick = 0u; pick < TOPK; ++pick )
	{
		unsigned long long best = 0ull;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			best = key[j] > best ? key[j] : best;
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		{
			const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
			best = other > best ? other : best;
		}
		const uint32_t e = 0xffffffffu - (uint32_t)best;
		const uint32_t owner_lane = e & 31u;
		float picked_s = 0.0f;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			if ( lane + j * 32u == e )
			{
				picked_s = sc[j];
				key[j] = 0ull;
			}
		picked_s = __shfl_sync(0xffffffffu, picked_s, (int)owner_lane);
		total += picked_s;
		if ( lane == 0u )
		{
			out_exp[pick] = e;
			out_wt[pick] = picked_s;
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
		}
	}
	if ( lane == 0u )
	{
		const float denom = total + 1e-20f;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			const uint32_t ge = out_exp[pick];
			if ( ge >= expert_first && ge < expert_first + local_experts )
			{
				out_exp[pick] = ge - expert_first;
				out_wt[pick] = out_wt[pick] / denom;
			}
			else
			{
				out_exp[pick] = local_experts;
				out_wt[pick] = 0.0f;
			}
		}
	}
}

template<class G>
__global__ __launch_bounds__(MM_THREADS, 1)
static void MimoSigmoidRouteSmallKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ packed_expert,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m,
	uint32_t n_up, uint32_t *__restrict__ tile_prefix_up, uint32_t n_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, routes = rows * G::kTopK;
	LmDependentWait();
	if ( warp < rows )
		MimoSigmoidTopkWarp<G::kExperts, G::kTopK>(logits, bias, warp, expert_first, G::kLocalExperts, fault, s_exp + warp * G::kTopK,
			s_wt + warp * G::kTopK, lane);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	LmRouteBuildSmallBody<MM_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
		tile_m, n_up, tile_prefix_up, n_down, tile_prefix_down);
	__syncthreads();
	if ( threadIdx.x < routes )
		packed_expert[route_packed_row[threadIdx.x]] = s_exp[threadIdx.x];
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(MM_THREADS, 2)
static void MimoSigmoidTopkLocalKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	constexpr uint32_t WARPS = MM_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x * WARPS + warp;
	LmDependentWait();
	if ( row < rows )
		MimoSigmoidTopkWarp<G::kExperts, G::kTopK>(logits, bias, row, expert_first, G::kLocalExperts, fault,
			route_expert + (uint64_t)row * G::kTopK, route_weight + (uint64_t)row * G::kTopK, lane);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
static void MimoRouteBuildPackedKernel(const uint32_t *__restrict__ route_expert, uint32_t routes, uint32_t top_k,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t *__restrict__ packed_expert, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	LmDependentWait();
	LmRouteBuildBlock<THREADS, EXPERTS>(route_expert, routes, top_k, group_row_offset, route_packed_row, route_source_token, tile_m,
		neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	__syncthreads();
	for ( uint32_t route = threadIdx.x; route < routes; route += THREADS )
		packed_expert[route_packed_row[route]] = route_expert[route];
	LmDependentRelease();
}

__global__ __launch_bounds__(MM_THREADS, 2)
static void MimoMoeSiluPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint16_t *__restrict__ output, uint32_t width)
{
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < group_row_offset[local_experts] )
	{
		const uint64_t row_base = (uint64_t)pr * width;
		LmBinaryBf16GridBody<MM_THREADS>(gate + row_base, up + row_base, output + row_base, width, LmStridedGateUpOp<true>{});
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(MM_THREADS, 2)
static void MimoMoeSmallW1W3SiluKernel(const uint8_t *__restrict__ w1, const uint8_t *__restrict__ s1, const uint8_t *__restrict__ w3,
	const uint8_t *__restrict__ s3, const uint16_t *__restrict__ activation, const uint32_t *__restrict__ packed_expert,
	const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output, uint32_t rows)
{
	constexpr uint32_t CHUNKS = G::kHidden / 32u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = MM_THREADS / LANES;
	constexpr uint32_t N = G::kMoeInter;
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < __ldg(group_row_offset + G::kLocalExperts) )
	{
		const uint32_t exp = __ldg(packed_expert + pr);
		const uint32_t src_row = rows > 1u ? __ldg(route_source_token + pr) : 0u;
		const uint16_t *act_row = activation + (uint64_t)src_row * G::kHidden;
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const bool reuse = rows > 1u;
			const uint64_t row_base = (uint64_t)exp * N + col0;
			const uint4 *w1_0 = (const uint4 *)w1 + row_base * CHUNKS;
			const uint4 *w3_0 = (const uint4 *)w3 + row_base * CHUNKS;
			const uint4 *w1_1 = has1 ? w1_0 + CHUNKS : w1_0;
			const uint4 *w3_1 = has1 ? w3_0 + CHUNKS : w3_0;
			const uint8_t *sc1_0 = s1 + row_base * CHUNKS;
			const uint8_t *sc3_0 = s3 + row_base * CHUNKS;
			const uint8_t *sc1_1 = has1 ? sc1_0 + CHUNKS : sc1_0;
			const uint8_t *sc3_1 = has1 ? sc3_0 + CHUNKS : sc3_0;
			float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const LmSkinnyQuad act = LmSkinnyFormat<LmMxfp4>::Load(act_row + c * 32u);
				const uint4 vw1_0 = reuse ? __ldg(w1_0 + c) : __ldcs(w1_0 + c);
				const uint4 vw3_0 = reuse ? __ldg(w3_0 + c) : __ldcs(w3_0 + c);
				const uint4 vw1_1 = reuse ? __ldg(w1_1 + c) : __ldcs(w1_1 + c);
				const uint4 vw3_1 = reuse ? __ldg(w3_1 + c) : __ldcs(w3_1 + c);
				const uint8_t rs1_0 = __ldg(sc1_0 + c);
				const uint8_t rs3_0 = __ldg(sc3_0 + c);
				const uint8_t rs1_1 = __ldg(sc1_1 + c);
				const uint8_t rs3_1 = __ldg(sc3_1 + c);
				g0 = fmaf(LmUe8m0ToFloat(rs1_0), LmSkinnyFormat<LmMxfp4>::Chunk(vw1_0, act), g0);
				u0 = fmaf(LmUe8m0ToFloat(rs3_0), LmSkinnyFormat<LmMxfp4>::Chunk(vw3_0, act), u0);
				g1 = fmaf(LmUe8m0ToFloat(rs1_1), LmSkinnyFormat<LmMxfp4>::Chunk(vw1_1, act), g1);
				u1 = fmaf(LmUe8m0ToFloat(rs3_1), LmSkinnyFormat<LmMxfp4>::Chunk(vw3_1, act), u1);
			}
			#pragma unroll
			for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
			{
				g0 += __shfl_down_sync(0xffffffffu, g0, offset);
				u0 += __shfl_down_sync(0xffffffffu, u0, offset);
				g1 += __shfl_down_sync(0xffffffffu, g1, offset);
				u1 += __shfl_down_sync(0xffffffffu, u1, offset);
			}
			if ( sub == 0u )
			{
				output[(uint64_t)pr * N + col0] = LmFloatToBf16(SparkMathRoundBf16(SparkMathSilu(SparkMathRoundBf16(g0))) * SparkMathRoundBf16(u0));
				if ( has1 )
					output[(uint64_t)pr * N + col0 + 1u] = LmFloatToBf16(SparkMathRoundBf16(SparkMathSilu(SparkMathRoundBf16(g1))) * SparkMathRoundBf16(u1));
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(MM_THREADS, 2)
static void MimoMoeSmallW2FinalizeKernel(const uint8_t *__restrict__ w2, const uint8_t *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight,
	uint16_t *__restrict__ output)
{
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t CHUNKS = K / 32u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = MM_THREADS / LANES;
	constexpr uint32_t N = G::kHidden;
	__shared__ uint32_t s_exp[G::kTopK], s_pr[G::kTopK], s_count;
	__shared__ float s_wt[G::kTopK];
	const uint32_t row = blockIdx.y;
	uint16_t *out_row = output + (uint64_t)row * N;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		const uint32_t rbase = row * G::kTopK;
		uint32_t cnt = 0u;
		for ( uint32_t k = 0u; k < G::kTopK; ++k )
		{
			const uint32_t e = route_expert[rbase + k];
			if ( e < G::kLocalExperts )
			{
				s_exp[cnt] = e;
				s_pr[cnt] = route_packed_row[rbase + k];
				s_wt[cnt] = route_weight[rbase + k];
				cnt++;
			}
		}
		s_count = cnt;
	}
	__syncthreads();
	const uint32_t col0 = (blockIdx.x * GROUPS + (threadIdx.x / LANES)) * 4u;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	if ( col0 < N )
	{
		const uint32_t count = s_count;
		const bool has1 = col0 + 1u < N, has2 = col0 + 2u < N, has3 = col0 + 3u < N;
		const bool reuse = gridDim.y > 1u;
		if ( count == 0u )
		{
			if ( sub == 0u )
			{
				out_row[col0] = 0u;
				if ( has1 ) out_row[col0 + 1u] = 0u;
				if ( has2 ) out_row[col0 + 2u] = 0u;
				if ( has3 ) out_row[col0 + 3u] = 0u;
			}
		}
		else
		{
			float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
			for ( uint32_t k = 0u; k < count; ++k )
			{
				const uint16_t *a_row = expert_act + (uint64_t)s_pr[k] * K;
				const uint64_t row_base = (uint64_t)s_exp[k] * N + col0;
				const uint4 *w0 = (const uint4 *)w2 + row_base * CHUNKS;
				const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
				const uint4 *w2p = has2 ? w0 + 2u * CHUNKS : w0;
				const uint4 *w3p = has3 ? w0 + 3u * CHUNKS : w0;
				const uint8_t *sc0 = s2 + row_base * CHUNKS;
				const uint8_t *sc1 = has1 ? sc0 + CHUNKS : sc0;
				const uint8_t *sc2 = has2 ? sc0 + 2u * CHUNKS : sc0;
				const uint8_t *sc3 = has3 ? sc0 + 3u * CHUNKS : sc0;
				float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const LmSkinnyQuad act = LmSkinnyFormat<LmMxfp4>::Load(a_row + c * 32u);
					const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
					const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
					acc0 = fmaf(LmUe8m0ToFloat(__ldg(sc0 + c)), LmSkinnyFormat<LmMxfp4>::Chunk(vw0, act), acc0);
					acc1 = fmaf(LmUe8m0ToFloat(__ldg(sc1 + c)), LmSkinnyFormat<LmMxfp4>::Chunk(vw1, act), acc1);
					const uint4 vw2 = reuse ? __ldg(w2p + c) : __ldcs(w2p + c);
					const uint4 vw3 = reuse ? __ldg(w3p + c) : __ldcs(w3p + c);
					acc2 = fmaf(LmUe8m0ToFloat(__ldg(sc2 + c)), LmSkinnyFormat<LmMxfp4>::Chunk(vw2, act), acc2);
					acc3 = fmaf(LmUe8m0ToFloat(__ldg(sc3 + c)), LmSkinnyFormat<LmMxfp4>::Chunk(vw3, act), acc3);
				}
				#pragma unroll
				for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
				{
					acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
					acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
					acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
					acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
				}
				if ( sub == 0u )
				{
					const float wt = s_wt[k];
					total0 = fmaf(SparkMathRoundBf16(acc0), wt, total0);
					total1 = fmaf(SparkMathRoundBf16(acc1), wt, total1);
					total2 = fmaf(SparkMathRoundBf16(acc2), wt, total2);
					total3 = fmaf(SparkMathRoundBf16(acc3), wt, total3);
				}
			}
			if ( sub == 0u )
			{
				out_row[col0] = LmFloatToBf16(total0);
				if ( has1 ) out_row[col0 + 1u] = LmFloatToBf16(total1);
				if ( has2 ) out_row[col0 + 2u] = LmFloatToBf16(total2);
				if ( has3 ) out_row[col0 + 3u] = LmFloatToBf16(total3);
			}
		}
	}
	LmDependentRelease();
}

static __device__ __forceinline__ uint32_t MimoE8m0ToBf16x2(uint8_t sc)
{
	const uint32_t s = (uint32_t)sc;
	if ( s >= 2u && s <= 142u )
	{
		const uint32_t b = (s + 112u) << 7u;
		return b | (b << 16u);
	}
	return 0u;
}

static __device__ __forceinline__ uint32_t MimoE2m1ByteFast(uint32_t shifted, uint32_t scale_bf16x2)
{
	uint32_t widened, bits, out;
	asm("{\n\t.reg .b8 narrow;\n\tcvt.u8.u32 narrow, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(widened) : "r"(shifted));
	bits = ((widened >> 3u) & 0x0fff0fffu) | (widened & 0x80008000u);
	asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(out) : "r"(bits), "r"(scale_bf16x2));
	return out;
}

static __device__ __forceinline__ void MimoE2m1Word4(uint32_t word, uint8_t sc, uint32_t scale_bf16x2, uint32_t lo[2], uint32_t hi[2])
{
	if ( scale_bf16x2 != 0u )
	{
		lo[0] = MimoE2m1ByteFast(word, scale_bf16x2);
		lo[1] = MimoE2m1ByteFast(word >> 8u, scale_bf16x2);
		hi[0] = MimoE2m1ByteFast(word >> 16u, scale_bf16x2);
		hi[1] = MimoE2m1ByteFast(word >> 24u, scale_bf16x2);
	}
	else
	{
		const float scale = LmUe8m0ToFloat(sc);
		const float2 p0 = LmSkinnyE2m1Pair(word, 0u);
		const float2 p1 = LmSkinnyE2m1Pair(word, 1u);
		const float2 p2 = LmSkinnyE2m1Pair(word, 2u);
		const float2 p3 = LmSkinnyE2m1Pair(word, 3u);
		lo[0] = LmPackBf16Pair(p0.x * scale, p0.y * scale);
		lo[1] = LmPackBf16Pair(p1.x * scale, p1.y * scale);
		hi[0] = LmPackBf16Pair(p2.x * scale, p2.y * scale);
		hi[1] = LmPackBf16Pair(p3.x * scale, p3.y * scale);
	}
}

static __device__ __forceinline__ void MimoCpAsync16(void *dst, const void *src, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(src), "r"(bytes) : "memory");
}

template<uint32_t MAX_M_TILES, uint32_t STAGES, uint32_t WARPS, bool ACTIVATION_PACKED, bool FUSED_GATE_UP>
static __host__ __device__ constexpr uint32_t MimoMxfp4SharedBytes(uint32_t groups)
{
	constexpr uint32_t TILE_M = MAX_M_TILES * 16u;
	constexpr uint32_t TILE_K = 128u;
	constexpr uint32_t N_PER_WARP = FUSED_GATE_UP ? 8u : 16u;
	constexpr uint32_t W_ROWS_TOTAL = (FUSED_GATE_UP ? 2u : 1u) * WARPS * N_PER_WARP;
	constexpr uint32_t A_BYTES = STAGES * TILE_M * (TILE_K + 8u) * 2u;
	constexpr uint32_t B_BYTES = STAGES * W_ROWS_TOTAL * (TILE_K / 2u);
	constexpr uint32_t SC_BYTES = 2u * W_ROWS_TOTAL * 16u;
	return A_BYTES + B_BYTES + SC_BYTES + (groups + 1u) * (uint32_t)sizeof(uint32_t);
}

template<uint32_t MAX_M_TILES, uint32_t STAGES, uint32_t WARPS, bool ACTIVATION_PACKED, bool FUSED_GATE_UP>
__global__ __launch_bounds__(WARPS * 32u, MAX_M_TILES <= 2u ? 2 : 1)
static void MimoMxfp4ExpertsKernel(const uint8_t *__restrict__ w1, const uint8_t *__restrict__ s1, const uint8_t *__restrict__ w3,
	const uint8_t *__restrict__ s3, const uint16_t *__restrict__ activation, uint16_t *__restrict__ output,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token, uint32_t groups, uint32_t input_dim,
	uint32_t output_dim)
{
	constexpr uint32_t THREADS = WARPS * 32u;
	constexpr uint32_t TILE_M = MAX_M_TILES * 16u;
	constexpr uint32_t TILE_K = 128u;
	constexpr uint32_t N_PER_WARP = FUSED_GATE_UP ? 8u : 16u;
	constexpr uint32_t TILE_N = WARPS * N_PER_WARP;
	constexpr uint32_t W_MATS = FUSED_GATE_UP ? 2u : 1u;
	constexpr uint32_t W_ROWS_TOTAL = W_MATS * TILE_N;
	constexpr uint32_t A_PITCH = TILE_K + 8u;
	constexpr uint32_t B_ROW_BYTES = TILE_K / 2u;
	constexpr uint32_t A_STAGE_BYTES = TILE_M * A_PITCH * 2u;
	constexpr uint32_t B_STAGE_BYTES = W_ROWS_TOTAL * B_ROW_BYTES;
	extern __shared__ __align__(128) uint8_t mimo_mx4_smem[];
	uint16_t (*s_a)[TILE_M][A_PITCH] = (uint16_t (*)[TILE_M][A_PITCH])mimo_mx4_smem;
	uint8_t (*s_b)[W_ROWS_TOTAL][B_ROW_BYTES] = (uint8_t (*)[W_ROWS_TOTAL][B_ROW_BYTES])(mimo_mx4_smem + STAGES * A_STAGE_BYTES);
	uint8_t (*s_sc)[W_ROWS_TOTAL][16u] = (uint8_t (*)[W_ROWS_TOTAL][16u])(mimo_mx4_smem + STAGES * A_STAGE_BYTES + STAGES * B_STAGE_BYTES);
	uint32_t *prefix = (uint32_t *)(mimo_mx4_smem + STAGES * A_STAGE_BYTES + STAGES * B_STAGE_BYTES + 2u * W_ROWS_TOTAL * 16u);
	const uint32_t n_blocks = (output_dim + TILE_N - 1u) / TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, g = lane >> 2u, c = lane & 3u;
	const uint32_t k_tiles = input_dim / TILE_K;
	const uint64_t payload_pitch = (uint64_t)(input_dim >> 1u);
	const uint64_t scale_pitch = (uint64_t)(input_dim >> 5u);
	const uint64_t scale_group_stride = (uint64_t)output_dim * scale_pitch;
	const uint32_t b_row0 = FUSED_GATE_UP ? (warp * 8u + g) : (warp * 16u + g);
	const uint32_t b_row1 = FUSED_GATE_UP ? (TILE_N + warp * 8u + g) : (warp * 16u + 8u + g);
	constexpr uint32_t A_VEC_THREADS = TILE_K / 8u;
	constexpr uint32_t A_ROW_STEP = THREADS / A_VEC_THREADS;
	constexpr uint32_t A_COPIES = (TILE_M + A_ROW_STEP - 1u) / A_ROW_STEP;
	const uint32_t a_vec = threadIdx.x % A_VEC_THREADS, a_row0 = threadIdx.x / A_VEC_THREADS;
	LmStreamGemmArguments targs;
	targs.group_row_offset = group_row_offset;
	targs.groups = groups;
	targs.rows = 0u;
	LmDependentWait();
	LmStreamTilePrefix(targs, prefix, groups, TILE_M, n_blocks);
	const uint32_t total_tiles = prefix[groups];
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		const LmStreamTile work = LmStreamTileAt(targs, prefix, groups, TILE_M, TILE_N, n_blocks, tile);
		const uint32_t group = work.group, first = work.row_base, row_end = work.row_limit, n_base = work.neuron_base;
		const uint32_t count = row_end - first < TILE_M ? row_end - first : TILE_M;
		const uint32_t m_tiles_live = (count + 15u) >> 4u;
		const uint16_t *a_ptrs[A_COPIES];
		#pragma unroll
		for ( uint32_t i = 0u; i < A_COPIES; ++i )
		{
			const uint32_t lr = a_row0 + i * A_ROW_STEP;
			if ( lr < count )
			{
				const uint32_t pr = first + lr;
				a_ptrs[i] = activation + (uint64_t)(ACTIVATION_PACKED ? pr : route_source_token[pr]) * input_dim + (uint64_t)a_vec * 8u;
			}
			else
				a_ptrs[i] = 0;
		}
		auto stage_tile = [&](uint32_t st, uint32_t kt) {
			#pragma unroll
			for ( uint32_t i = 0u; i < A_COPIES; ++i )
			{
				const uint32_t lr = a_row0 + i * A_ROW_STEP;
				if ( lr < TILE_M && a_ptrs[i] != 0 )
					MimoCpAsync16(&s_a[st][lr][a_vec * 8u], a_ptrs[i] + (uint64_t)kt * TILE_K, 16u);
			}
			constexpr uint32_t B_VECS_PER_ROW = B_ROW_BYTES / 16u;
			constexpr uint32_t B_TOTAL_VECS = W_ROWS_TOTAL * B_VECS_PER_ROW;
			#pragma unroll
			for ( uint32_t idx = threadIdx.x; idx < B_TOTAL_VECS; idx += THREADS )
			{
				const uint32_t wr = idx / B_VECS_PER_ROW, bv = idx % B_VECS_PER_ROW;
				const uint32_t mat = FUSED_GATE_UP ? (wr / TILE_N) : 0u;
				const uint32_t neuron = n_base + (FUSED_GATE_UP ? (wr % TILE_N) : wr);
				const bool ok = neuron < output_dim;
				const uint8_t *wbase = (mat == 0u ? w1 : w3);
				const uint8_t *src = ok ? wbase + ((uint64_t)group * output_dim + neuron) * payload_pitch + (uint64_t)kt * B_ROW_BYTES + (uint64_t)bv * 16u : wbase;
				MimoCpAsync16(&s_b[st][wr][bv * 16u], src, ok ? 16u : 0u);
			}
			if ( (kt & 3u) == 0u )
			{
				const uint32_t sc_off = kt * 4u, avail = sc_off < scale_pitch ? (uint32_t)(scale_pitch - sc_off) : 0u;
				const uint32_t sb_slot = (kt >> 2u) & 1u;
				#pragma unroll
				for ( uint32_t wr = threadIdx.x; wr < W_ROWS_TOTAL; wr += THREADS )
				{
					const uint32_t mat = FUSED_GATE_UP ? (wr / TILE_N) : 0u;
					const uint32_t neuron = n_base + (FUSED_GATE_UP ? (wr % TILE_N) : wr);
					const uint32_t bytes = (neuron < output_dim && avail != 0u) ? (avail >= 16u ? 16u : avail) : 0u;
					const uint8_t *sbase = (mat == 0u ? s1 : s3);
					const uint8_t *src = bytes != 0u ? sbase + (uint64_t)group * scale_group_stride + (uint64_t)neuron * scale_pitch + sc_off : sbase;
					MimoCpAsync16(&s_sc[sb_slot][wr][0], src, bytes);
				}
			}
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		};
		float acc0[MAX_M_TILES][4], acc1[MAX_M_TILES][4];
		#pragma unroll
		for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
			acc0[m][0] = acc0[m][1] = acc0[m][2] = acc0[m][3] = acc1[m][0] = acc1[m][1] = acc1[m][2] = acc1[m][3] = 0.0f;
		uint32_t kp = 0u, sp = 0u, sc = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < k_tiles )
			{
				stage_tile(sp, kp);
				kp++;
				sp = (sp + 1u == STAGES) ? 0u : (sp + 1u);
			}
		}
		for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
		{
			if constexpr ( STAGES == 3u )
			{
				if ( kt + 1u < k_tiles )
					asm volatile("cp.async.wait_group 1;\n" ::: "memory");
				else
					asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			}
			else
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			if ( kp < k_tiles )
			{
				stage_tile(sp, kp);
				kp++;
				sp = (sp + 1u == STAGES) ? 0u : (sp + 1u);
			}
			const uint32_t sb_slot = (kt >> 2u) & 1u, sb_sub = (kt & 3u) * 4u;
			const uint32_t sc0_all = *(const uint32_t *)&s_sc[sb_slot][b_row0][sb_sub];
			const uint32_t sc1_all = *(const uint32_t *)&s_sc[sb_slot][b_row1][sb_sub];
			#pragma unroll
			for ( uint32_t sg = 0u; sg < 4u; ++sg )
			{
				const uint8_t raw0 = (uint8_t)(sc0_all >> (sg * 8u));
				const uint8_t raw1 = (uint8_t)(sc1_all >> (sg * 8u));
				const uint32_t s0_2 = MimoE8m0ToBf16x2(raw0);
				const uint32_t s1_2 = MimoE8m0ToBf16x2(raw1);
				const uint32_t w0 = *(const uint32_t *)&s_b[sc][b_row0][sg * 16u + c * 4u];
				const uint32_t w1_w = *(const uint32_t *)&s_b[sc][b_row1][sg * 16u + c * 4u];
				uint32_t b0_lo[2], b0_hi[2], b1_lo[2], b1_hi[2];
				MimoE2m1Word4(w0, raw0, s0_2, b0_lo, b0_hi);
				MimoE2m1Word4(w1_w, raw1, s1_2, b1_lo, b1_hi);
				const uint32_t k_off = sg * 32u + c * 8u;
				#pragma unroll
				for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
				{
					if ( m < m_tiles_live )
					{
						const uint4 low = *(const uint4 *)&s_a[sc][m * 16u + g][k_off];
						const uint4 high = *(const uint4 *)&s_a[sc][m * 16u + g + 8u][k_off];
						const uint32_t a0[4] = { low.x, high.x, low.y, high.y };
						const uint32_t a1[4] = { low.z, high.z, low.w, high.w };
						LmMmaBf16(acc0[m], a0, b0_lo);
						LmMmaBf16(acc0[m], a1, b0_hi);
						LmMmaBf16(acc1[m], a0, b1_lo);
						LmMmaBf16(acc1[m], a1, b1_hi);
					}
				}
			}
			sc = (sc + 1u == STAGES) ? 0u : (sc + 1u);
		}
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		if constexpr ( FUSED_GATE_UP )
		{
			const uint32_t col = n_base + warp * 8u + c * 2u;
			if ( col < output_dim )
				#pragma unroll
				for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
					if ( m < m_tiles_live )
						#pragma unroll
						for ( uint32_t h = 0u; h < 2u; ++h )
						{
							const uint32_t pr = first + m * 16u + g + h * 8u;
							if ( pr < row_end )
							{
								const float v0 = SparkMathRoundBf16(SparkMathSilu(SparkMathRoundBf16(acc0[m][h * 2u]))) * SparkMathRoundBf16(acc1[m][h * 2u]);
								const float v1 = SparkMathRoundBf16(SparkMathSilu(SparkMathRoundBf16(acc0[m][h * 2u + 1u]))) * SparkMathRoundBf16(acc1[m][h * 2u + 1u]);
								*(uint32_t *)(output + (uint64_t)pr * output_dim + col) = LmPackBf16Pair(v0, v1);
							}
						}
		}
		else
		{
			const uint32_t col0 = n_base + warp * 16u + c * 2u, col1 = col0 + 8u;
			#pragma unroll
			for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
				if ( m < m_tiles_live )
					#pragma unroll
					for ( uint32_t h = 0u; h < 2u; ++h )
					{
						const uint32_t pr = first + m * 16u + g + h * 8u;
						if ( pr < row_end )
						{
							if ( col0 < output_dim )
								*(uint32_t *)(output + (uint64_t)pr * output_dim + col0) = LmPackBf16Pair(acc0[m][h * 2u], acc0[m][h * 2u + 1u]);
							if ( col1 < output_dim )
								*(uint32_t *)(output + (uint64_t)pr * output_dim + col1) = LmPackBf16Pair(acc1[m][h * 2u], acc1[m][h * 2u + 1u]);
						}
					}
		}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<class G>
struct MimoModel : CudaStage<MimoModel<G>>
{
	using S = MimoShape<G>;
	using State = MimoState<G>;
	using Layer = MimoLayer;
	using Schedule = MimoSchedule<G, MimoModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::SlidingKv, &State::sliding_views, 0u, S::kSlidingLayers>,
		CudaStageKvViews<typename S::FullKv, &State::full_views, 1u, G::kFullLayers>>;
	using LaneShared = CudaStageFields<&State::sliding_inv_freq, &State::full_inv_freq, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = true;
	static constexpr bool kFusedResidualNorm = true;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kEpsilon;
	static constexpr uint32_t kPackAlignment = MM_PACK_ALIGNMENT;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
	};

	static const char *Tag(void) { return G::Tag(); }
	static const char *ReportTag(void) { return G::ReportTag(); }
	static const char *WeightdModel(void) { return G::Name(); }
	static uint32_t RowExact(const SparkStageRunnerStep *step) { return step->commit == 0u ? 1u : 0u; }

	static int32_t Fp8(const MimoScaled &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return LmDenseRows<LmFp8>(weight.weight, LmScaleTensorBuild(weight.scale, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u,
			MM_FP8_GROUP), source, destination, (float *)0, rows, input_dimension, output_dimension, RowExact(step), step->multiprocessors, stream);
	}

	static int32_t Bf16(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			RowExact(step), step->multiprocessors, stream);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const MimoScaled &weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const LmScaleTensor scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(weight.scale, G::kLocalExperts, output_dimension,
			input_dimension);
		return down != 0u ? LmLocalExpertsDown<LmMxfp4>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream,
			RowExact(step)) : LmLocalExpertsUp<LmMxfp4>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream,
			RowExact(step));
	}

	static void Norm(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t rows, SparkGpuStream stream)
	{
		(void)LM_LAUNCH_DEPENDENT((MimoAddRmsNormKernel<MM_THREADS, G::kHidden, false>), rows, MM_THREADS, 0, stream,
			const_cast<uint16_t *>(input), (const uint16_t *)0, weight, output, G::kEpsilon);
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const MimoLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		using Kv = typename std::conditional<FULL, typename S::FullKv, typename S::SlidingKv>::type;
		constexpr uint32_t KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
		constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
		constexpr uint32_t WARPS = MM_THREADS / LM_WARP_LANES;
		constexpr uint32_t TASKS = G::kHeads + 2u * KVH;
		const uint32_t rows = step->rows, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kValueDim, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Fp8(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		const uint32_t prep_blocks = rows >= 128u ? 2u : (TASKS + WARPS - 1u) / WARPS;
		(void)LM_LAUNCH_DEPENDENT((MimoAttnPrepStoreKernel<G, FULL>), dim3(prep_blocks, rows), MM_THREADS, 0, stream,
			state->attn_gathered, step->positions, step->tree != 0u ? step->slots : step->positions, step->sequence_of_row,
			FULL ? state->full_inv_freq : state->sliding_inv_freq, *view, state->query, rows);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,G::kHeadDim,G::kValueDim>(*view, state->query, G::kHeads, step->sequence_of_row,
				step->context_length, step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u,
				FULL ? 0u : G::kWindow, step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
				step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if constexpr ( G::kTpDegree > 1u && (G::kHeads % G::kTpDegree) == 0u && G::kHeads / G::kTpDegree <= 8u && G::kValueDim == 128u )
		{
			(void)LM_LAUNCH_DEPENDENT((MimoGqaShardMergeKernel<G>), rows, (G::kHeads / G::kTpDegree) * 32u, 0, stream, state->received, stride,
				state->tp_rank, state->merged, FULL ? (const float *)0 : layer->sinks);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( LmGqaShardMergeLaunch<G::kValueDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream,
				FULL ? (const float *)0 : layer->sinks) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( Bf16(layer->out, state->merged, state->partial, (float *)0, rows, S::kRankWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	template<uint32_t MAX_M_TILES, uint32_t STAGES, bool ACTIVATION_PACKED, bool FUSED_GATE_UP>
	static int32_t Mxfp4Launch(const uint8_t *w1, const uint8_t *s1, const uint8_t *w3, const uint8_t *s3, const uint16_t *activation, uint16_t *output,
		const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t routes, uint32_t input_dim, uint32_t output_dim,
		uint32_t multiprocessors, SparkGpuStream stream)
	{
		static std::atomic<uint64_t> granted(0u);
		static std::atomic<int32_t> resident(0);
		constexpr uint32_t TILE_M = MAX_M_TILES * 16u;
		constexpr uint32_t TILE_N = FUSED_GATE_UP ? 64u : 128u;
		constexpr uint32_t shared = MimoMxfp4SharedBytes<MAX_M_TILES, STAGES, 8u, ACTIVATION_PACKED, FUSED_GATE_UP>(G::kLocalExperts);
		int device = 0, per_sm = 1;
		if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_STREAM_MAX_DEVICES )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
		{
			if ( SparkGpuFuncSetAttribute((const void *)MimoMxfp4ExpertsKernel<MAX_M_TILES, STAGES, 8u, ACTIVATION_PACKED, FUSED_GATE_UP>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,
					MimoMxfp4ExpertsKernel<MAX_M_TILES, STAGES, 8u, ACTIVATION_PACKED, FUSED_GATE_UP>, 256u, (int)shared) != SPARK_GPU_SUCCESS || per_sm < 1 )
				per_sm = 1;
			resident.store(per_sm);
			granted.fetch_or(UINT64_C(1) << (uint32_t)device);
		}
		const uint64_t n_tiles = (output_dim + TILE_N - 1u) / TILE_N;
		const uint64_t tile_bound = ((uint64_t)routes / TILE_M + G::kLocalExperts) * n_tiles;
		uint64_t blocks = (uint64_t)multiprocessors * (uint64_t)resident.load();
		if ( tile_bound < blocks )
			blocks = tile_bound;
		(void)LM_LAUNCH_DEPENDENT((MimoMxfp4ExpertsKernel<MAX_M_TILES, STAGES, 8u, ACTIVATION_PACKED, FUSED_GATE_UP>), (uint32_t)blocks, 256u, shared, stream,
			w1, s1, w3, s3, activation, output, group_row_offset, route_source_token, G::kLocalExperts, input_dim, output_dim);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static SparkStatus Dense(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const MimoLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows;
		if ( Fp8(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * S::kDenseRows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((MimoDenseSiluMulKernel<MM_THREADS>), dim3(LmVecBlocks<MM_THREADS>(S::kDenseRows), rows), MM_THREADS, 0, stream,
			state->gate_up, state->intermediate, S::kDenseRows);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Fp8(layer->ffn_down, state->intermediate, state->partial, rows, S::kDenseRows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus ExpertRows(State *state, const MimoLayer *layer, const SparkStageRunnerStep *step, uint32_t first, uint32_t rows, SparkGpuStream stream)
	{
		const uint32_t routes = rows * G::kTopK, exact = RowExact(step);
		const uint16_t *normed = state->normed + (uint64_t)first * G::kHidden;
		uint16_t *partial_out = state->partial + (uint64_t)first * G::kHidden;
		if ( Bf16(layer->router, normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( routes <= 32u )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((MimoSigmoidRouteSmallKernel<G>), 1u, MM_THREADS, 0, stream, state->router_logits, layer->router_bias, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			constexpr uint32_t WARPS = MM_THREADS / LM_WARP_LANES;
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((MimoSigmoidTopkLocalKernel<G>), (rows + WARPS - 1u) / WARPS, MM_THREADS, 0, stream, state->router_logits,
				layer->router_bias, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			(void)LM_LAUNCH_DEPENDENT((MimoRouteBuildPackedKernel<MM_THREADS, G::kLocalExperts + 1u>), 1u, MM_THREADS, 0, stream,
				state->route_expert, routes, G::kTopK, state->group_row_offset, state->route_packed_row, state->route_source_token,
				state->route_global, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( rows <= 8u )
		{
			constexpr uint32_t UP_CHUNKS = G::kHidden / 32u;
			constexpr uint32_t UP_LANES = UP_CHUNKS >= 32u ? 32u : (UP_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t UP_COLS = (MM_THREADS / UP_LANES) * 2u;
			const uint32_t up_blocks = (G::kMoeInter + UP_COLS - 1u) / UP_COLS;
			(void)LM_LAUNCH_DEPENDENT((MimoMoeSmallW1W3SiluKernel<G>), dim3(up_blocks, routes), MM_THREADS, 0, stream,
				layer->moe_w1.weight, (const uint8_t *)layer->moe_w1.scale, layer->moe_w3.weight, (const uint8_t *)layer->moe_w3.scale,
				normed, state->route_global, state->route_source_token, state->group_row_offset, state->expert_act, rows);
			constexpr uint32_t DN_CHUNKS = G::kMoeInter / 32u;
			constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t DN_COLS = (MM_THREADS / DN_LANES) * 4u;
			const uint32_t dn_blocks = (G::kHidden + DN_COLS - 1u) / DN_COLS;
			(void)LM_LAUNCH_DEPENDENT((MimoMoeSmallW2FinalizeKernel<G>), dim3(dn_blocks, rows), MM_THREADS, 0, stream,
				layer->moe_w2.weight, (const uint8_t *)layer->moe_w2.scale, state->expert_act, state->route_packed_row, state->route_expert,
				state->route_weight, partial_out);
			return CudaStageLaunched();
		}
		if constexpr ( (G::kHidden % 512u) == 0u && (G::kMoeInter % 512u) == 0u )
		{
			if ( exact == 0u && rows >= 32u )
			{
				const uint32_t mean_rows = (routes + G::kLocalExperts - 1u) / G::kLocalExperts;
				int32_t st_up, st_dn;
				if ( mean_rows > 32u )
				{
					st_up = Mxfp4Launch<4u, 3u, false, true>(layer->moe_w1.weight, (const uint8_t *)layer->moe_w1.scale, layer->moe_w3.weight,
						(const uint8_t *)layer->moe_w3.scale, normed, state->expert_act, state->group_row_offset, state->route_source_token, routes,
						G::kHidden, G::kMoeInter, step->multiprocessors, stream);
					st_dn = st_up == LM_LAUNCH_OK ? Mxfp4Launch<4u, 3u, true, false>(layer->moe_w2.weight, (const uint8_t *)layer->moe_w2.scale,
						(const uint8_t *)0, (const uint8_t *)0, state->expert_act, state->expert_out, state->group_row_offset, (const uint32_t *)0,
						routes, G::kMoeInter, G::kHidden, step->multiprocessors, stream) : st_up;
				}
				else
				{
					st_up = Mxfp4Launch<2u, 2u, false, true>(layer->moe_w1.weight, (const uint8_t *)layer->moe_w1.scale, layer->moe_w3.weight,
						(const uint8_t *)layer->moe_w3.scale, normed, state->expert_act, state->group_row_offset, state->route_source_token, routes,
						G::kHidden, G::kMoeInter, step->multiprocessors, stream);
					st_dn = st_up == LM_LAUNCH_OK ? Mxfp4Launch<2u, 2u, true, false>(layer->moe_w2.weight, (const uint8_t *)layer->moe_w2.scale,
						(const uint8_t *)0, (const uint8_t *)0, state->expert_act, state->expert_out, state->group_row_offset, (const uint32_t *)0,
						routes, G::kMoeInter, G::kHidden, step->multiprocessors, stream) : st_up;
				}
				if ( st_dn != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				LM_LAUNCH((LmMoeLocalWeightedFinalizeKernel), dim3(LmVecBlocks<MM_THREADS>(G::kHidden), rows), MM_THREADS, 0, stream, state->expert_out,
					state->route_packed_row, state->route_expert, state->route_weight, (const uint16_t *)0, partial_out, G::kTopK, G::kLocalExperts, G::kHidden);
				return CudaStageLaunched();
			}
		}
		LmLocalExpertRoutes expert_routes;
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = G::kTopK;
		expert_routes.local_experts = G::kLocalExperts;
		expert_routes.multiprocessors = step->multiprocessors;
		if ( ExpertProjection(expert_routes, layer->moe_w1, normed, state->expert_gate, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, normed, state->expert_up, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((MimoMoeSiluPackedKernel), dim3(LmVecBlocks<MM_THREADS>(G::kMoeInter), routes), MM_THREADS, 0, stream,
			state->expert_gate, state->expert_up, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmMoeLocalWeightedFinalizeKernel), dim3(LmVecBlocks<MM_THREADS>(G::kHidden), rows), MM_THREADS, 0, stream, state->expert_out, state->route_packed_row,
			state->route_expert, state->route_weight, (const uint16_t *)0, partial_out, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus Experts(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const SparkStageRunnerStep *step = frame.step;
		const uint32_t chunk = step->commit == 0u ? LM_SKINNY_ROWS_WIDE : step->rows;
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t first = 0u; first < step->rows && status == SPARK_STATUS_OK; first += chunk )
			status = ExpertRows(state, &entry, step, first, step->rows - first < chunk ? step->rows - first : chunk, frame.stream);
		return status;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->tree != 0u && (step->commit != 0u || step->sequences != 1u || step->slots == 0 || step->fork_positions == 0 || step->slot_offsets == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		return CudaStage<MimoModel>::Begin(state, step, stream, frame);
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		if ( state == 0 || step == 0 || kept == 0 || step->rows == 0u || step->rows > state->max_rows || step->sequences == 0u || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->tree == 0u )
			return SPARK_STATUS_OK;
		if ( state->kv_attached == 0u || step->commit != 0u || step->sequences != 1u || step->slots == 0 || step->sequence_of_row == 0 || step->sequence_row_indices == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < S::kSlidingLayers; ++index )
		{
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::SlidingKv,MM_THREADS>), step->rows, MM_THREADS, 0, stream,
				state->sliding_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, S::kSlidingKey + S::kSlidingValue);
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		}
		for ( uint32_t index = 0u; index < G::kFullLayers; ++index )
		{
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::FullKv,MM_THREADS>), step->rows, MM_THREADS, 0, stream,
				state->full_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, S::kFullKey + S::kFullValue);
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		}
		return CudaStageLaunched();
	}

	static SparkStatus Residual(State *state, uint16_t *hidden, uint32_t rows, SparkGpuStream stream)
	{
		LM_LAUNCH((LmAddElementsKernel), LmVecBlocks<MM_THREADS>((uint64_t)rows * G::kHidden), MM_THREADS, 0, stream, hidden, state->partial, (uint64_t)rows * G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const uint16_t *weight) { Norm(frame.step->hidden_bf16, weight, state->normed, frame.step->rows, frame.stream); return CudaStageLaunched(); }
	static SparkStatus AddRmsNorm(State *state, const Frame &frame, const uint16_t *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((MimoAddRmsNormKernel<MM_THREADS, G::kHidden, true>), frame.step->rows, MM_THREADS, 0, frame.stream,
			frame.step->hidden_bf16, state->partial, weight, state->normed, G::kEpsilon);
		return CudaStageLaunched();
	}
	static SparkStatus AllReduce(State *state, const Frame &frame) { return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial); }
	static SparkStatus AddResidual(State *state, const Frame &frame) { return Residual(state, frame.step->hidden_bf16, frame.step->rows, frame.stream); }

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kLocalExperts + 2u;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->attn_local, rows * (S::kMaxFused / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kQuery * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * S::kRankWidth * sizeof(uint16_t) },
			{ (void **)&state->gate_up, rows * 2u * S::kDenseRows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * S::kDenseRows * sizeof(uint16_t) },
			{ (void **)&state->sliding_inv_freq, lane == 0u ? (G::kRopeDim / 2u) * sizeof(float) : 0u },
			{ (void **)&state->full_inv_freq, lane == 0u ? (G::kRopeDim / 2u) * sizeof(float) : 0u },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->expert_gate, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t row_bytes = G::kHidden * sizeof(uint16_t);
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kValueDim, state->tp_degree) * sizeof(float) +
			row_bytes - 1u) / row_bytes) * row_bytes / sizeof(float);
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration) { return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 ? 1u : 0u; }

	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t pattern_full = 0u;
		(void)state;
		(void)configuration;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			pattern_full += G::FullLayer(index) ? 1u : 0u;
		if ( pattern_full != G::kFullLayers || !G::FullLayer(0u) )
		{
			fprintf(stderr, "%s: the layer pattern has %u full layers, the geometry declares %u\n", G::Tag(), pattern_full, G::kFullLayers);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u, full_layers = 0u;
		uint32_t ffn_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "full_layers", &full_layers },
			{ "vocab_rows", &vocab_rows }, { "ffn_rows", &ffn_rows }, { "sliding_window", &window }, { "experts", &experts }, { "local_experts", &local },
			{ "expert_first", &state->expert_first }, { "top_k", &top_k }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			full_layers != G::kFullLayers || vocab_rows * degree != G::kVocab || ffn_rows != S::kDenseRows || window != G::kWindow ||
			experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u/%u full window %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, full_layers, window, experts, local, state->expert_first, top_k, configuration->tp_degree,
				configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		const uint32_t vocab_rows = G::kVocab / state->tp_degree;
		geometry->tree_verify = 1u;
		if ( CudaStage<MimoModel>::HeadRanges(state, geometry, false, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_rows != vocab_rows || geometry->head_rows != vocab_rows ||
			geometry->embed_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the pack's %u vocab rows of %u\n",
				G::Tag(), geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes, vocab_rows, G::kHidden);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static uint64_t Fp8Bytes(uint32_t rows, uint32_t columns) { return (uint64_t)rows * columns; }
	static uint64_t Fp8ScaleBytes(uint32_t rows, uint32_t columns) { return (uint64_t)rows * ((columns + MM_FP8_GROUP - 1u) / MM_FP8_GROUP) * sizeof(float); }

	static void AddFp8(CudaStageTensors *tensors, const char *name, uint32_t rows, uint32_t columns, MimoScaled *field)
	{
		tensors->Add(name, "fp8_e4m3", Fp8Bytes(rows, columns), &field->weight, rows);
		tensors->AddScaled(name, ".scale", "f32", Fp8ScaleBytes(rows, columns), &field->scale, rows);
	}

	static void AddExperts(CudaStageTensors *tensors, const char *name, MimoScaled *field, uint32_t rows, uint32_t columns)
	{
		tensors->Add(name, "mxfp4_e2m1", SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1, (uint64_t)G::kLocalExperts * rows, columns), &field->weight,
			G::kLocalExperts);
		tensors->AddScaled(name, ".scale", "e8m0", SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1, G::kLocalExperts, rows, columns), &field->scale,
			G::kLocalExperts);
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		MimoLayer *layer = &state->layers[index];
		const uint32_t full = G::FullLayer(index) ? 1u : 0u;
		const uint32_t qkv_rows = (full != 0u ? S::kFullFused : S::kSlidingFused) / G::kTpDegree;
		tensors->Add("input_norm", "bf16", (uint64_t)G::kHidden * sizeof(uint16_t), &layer->input_norm);
		tensors->Add("post_attn_norm", "bf16", (uint64_t)G::kHidden * sizeof(uint16_t), &layer->post_attn_norm);
		AddFp8(tensors, "attn_qkv", qkv_rows, G::kHidden, &layer->qkv);
		tensors->Add("attn_out", "bf16", (uint64_t)G::kHidden * S::kRankWidth * sizeof(uint16_t), &layer->out, G::kHidden);
		if ( full == 0u )
			tensors->Add("sinks", "f32", (uint64_t)G::kHeads * sizeof(float), &layer->sinks, G::kHeads);
		else
		{
			char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
			SparkNamedPackEntry entry;
			snprintf(key, sizeof(key), "layers.%u.sinks", index);
			if ( SparkNamedPackEntryLoad(&state->pack, key, &entry) == SPARK_STATUS_OK )
			{
				fprintf(stderr, "%s: layer %u is a full-attention layer in the model but the pack gives it sinks\n", G::Tag(), index);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		if ( index < G::kDenseLayers )
		{
			AddFp8(tensors, "ffn_gate_up", 2u * S::kDenseRows, G::kHidden, &layer->ffn_gate_up);
			AddFp8(tensors, "ffn_down", G::kHidden, S::kDenseRows, &layer->ffn_down);
		}
		else
		{
			tensors->Add("router", "bf16", (uint64_t)G::kExperts * G::kHidden * sizeof(uint16_t), &layer->router, G::kExperts);
			tensors->Add("router_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->router_bias, G::kExperts);
			AddExperts(tensors, "moe_w1", &layer->moe_w1, G::kMoeInter, G::kHidden);
			AddExperts(tensors, "moe_w3", &layer->moe_w3, G::kMoeInter, G::kHidden);
			AddExperts(tensors, "moe_w2", &layer->moe_w2, G::kHidden, G::kMoeInter);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindRefusal(SparkStatus status) { SPARK_FAIL(status == SPARK_STATUS_TARGET_MISMATCH ? status : SPARK_STATUS_PARSE_ERROR); }

	static SparkStatus Arrange(State *state)
	{
		uint32_t sliding = 0u, full = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			MimoLayer *layer = &state->layers[index];
			layer->full = G::FullLayer(index) ? 1u : 0u;
			layer->kv_index = layer->full != 0u ? full++ : sliding++;
		}
		if ( full != G::kFullLayers || sliding != S::kSlidingLayers )
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[2][G::kRopeDim / 2u];
		for ( uint32_t pair = 0u; pair < G::kRopeDim / 2u; ++pair )
		{
			const float exponent = (float)(2u * pair) / (float)G::kRopeDim;
			inv_freq[0][pair] = 1.0f / (float)pow((double)G::kSlidingTheta, (double)exponent);
			inv_freq[1][pair] = 1.0f / (float)pow((double)G::kFullTheta, (double)exponent);
		}
		if ( SparkGpuMemcpy(state->sliding_inv_freq, inv_freq[0], sizeof(inv_freq[0]), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpy(state->full_inv_freq, inv_freq[1], sizeof(inv_freq[1]), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "expert_first", state->expert_first },
			{ "record_stride", state->record_stride }
		};
		return CudaStageReportOut(fields, report);
	}
};
