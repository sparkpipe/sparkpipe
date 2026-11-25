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
#include "inference/kernels/attn.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_gemma4_family.h"
#include "sparkpipe/spark_gpu.h"

#define GM_THREADS 256u

template<class G>
struct GemmaShape : GemmaLayout<G>
{
	using SlidingKvGeometry = LmKvHeads<16u, G::kSlidingKvHeads, G::kSlidingDim, G::kPageSlots>;
	using FullKvGeometry = LmKvHeads<16u, G::kFullKvHeads, G::kFullDim, G::kPageSlots>;
};

typedef struct GemmaLayer
{
	const uint16_t *qkv;
	const uint16_t *out;
	const uint16_t *gate_up;
	const uint16_t *down;
	const float *input_norm;
	const float *post_attn_norm;
	const float *pre_ff_norm;
	const float *post_ff_norm;
	const float *layer_scalar;
	const float *q_norm;
	const float *k_norm;
	const uint16_t *gate;
	const uint16_t *router_proj;
	const float *router_scale;
	const float *expert_scale;
	const uint16_t *moe_w1;
	const uint16_t *moe_w3;
	const uint16_t *moe_w2;
	const float *post_ff_norm_1;
	const float *pre_ff_norm_2;
	const float *post_ff_norm_2;
	uint32_t kv_index;
	const float *expert_bias;
	uint32_t ffn_rows;
	uint32_t gate_up_rows;
	uint32_t window;
	float swiglu_limit;
} GemmaLayer;

template<class G>
struct GemmaState : CudaStagePackState
{
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint32_t router_fused_done;
	uint64_t record_stride;
	GemmaLayer layers[G::kLayers];
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *attn_full;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	uint16_t *slot;
	float *send;
	float *received;
	uint16_t *merged;
	uint16_t *gate_logits;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *full_inv_freq;
	uint16_t *moe_normed;
	uint16_t *router_logits;
	float *router_probs;
	uint32_t *route_expert;
	uint32_t *route_global;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	float *packed_weight;
	uint16_t *expert_gate;
	uint16_t *expert_up;
	uint16_t *expert_act;
	uint16_t *expert_out;
	LmKvShardView sliding_views[GemmaShape<G>::kSlidingLayers];
	LmKvShardView full_views[GemmaShape<G>::kFullLayers];
	int16_t tap_output[G::kLayers];
	uint16_t *tap_slab;
	uint64_t tap_stride;
};

static __device__ __forceinline__ float GemmaWarpSum(float value)
{
	#pragma unroll
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu, value, offset);
	return __shfl_sync(0xffffffffu, value, 0u);
}

static __device__ __forceinline__ float GemmaHalfWarpSum(float value)
{
	#pragma unroll
	for ( uint32_t offset = 8u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu, value, offset);
	return value;
}

template<uint32_t LOCAL_V>
static __device__ __forceinline__ uint4 GemmaLoadRankMajor8Const(const uint4 *__restrict__ gathered4, uint32_t rows, uint32_t row, uint32_t logical_v)
{
	const uint32_t rank = logical_v / LOCAL_V;
	const uint32_t elem = logical_v % LOCAL_V;
	return __ldg(gathered4 + (((uint64_t)rank * rows) + row) * LOCAL_V + elem);
}

static __global__ void GemmaInitSlidingInvFreqKernel(float *__restrict__ dst, uint32_t half, uint32_t width, float theta)
{
	const uint32_t pair = threadIdx.x;
	if ( pair < half )
		dst[pair] = __powf(theta, -2.0f * (float)pair / (float)width);
}

template<uint32_t HD, uint32_t ROPE_W, uint32_t ROT_PAIRS, bool FULL>
static __device__ __forceinline__ void GemmaRopeInReg(uint4 *out_v, const float *__restrict__ inv_freq, float pos, float rope_scale, uint32_t lane)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t SLOTS = (HD_V + 31u) / 32u;
	constexpr uint32_t ROPE_V = ROPE_W >> 3u;
	constexpr uint32_t HALF_V = ROPE_V >> 1u;
	if constexpr ( SLOTS == 1u )
	{
		uint4 partner;
		partner.x = __shfl_xor_sync(0xffffffffu, out_v[0].x, HALF_V);
		partner.y = __shfl_xor_sync(0xffffffffu, out_v[0].y, HALF_V);
		partner.z = __shfl_xor_sync(0xffffffffu, out_v[0].z, HALF_V);
		partner.w = __shfl_xor_sync(0xffffffffu, out_v[0].w, HALF_V);
		if ( lane < ROPE_V )
		{
			const bool is_hi = lane >= HALF_V;
			const uint32_t pair_base = (lane & (HALF_V - 1u)) << 3u;
			float mine[8], other[8], res[8];
			LmLoadBf16x8AsFloat(out_v[0], mine);
			LmLoadBf16x8AsFloat(partner, other);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				float lo = is_hi ? other[k] : mine[k];
				float hi = is_hi ? mine[k] : other[k];
				const float angle = pos * __ldg(inv_freq + pair_base + k);
				LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&lo, &hi, angle, rope_scale);
				res[k] = is_hi ? hi : lo;
			}
			out_v[0] = LmPackUint4FloatToBf16(res);
		}
	}
	else
	{
		static_assert(SLOTS == 2u && ROPE_W == HD, "two-slot RoPE covers full 512-dim head");
		const uint32_t pair_base = lane << 3u;
		if ( ROT_PAIRS == 0u || pair_base < ROT_PAIRS )
		{
			float lo_v[8], hi_v[8];
			LmLoadBf16x8AsFloat(out_v[0], lo_v);
			LmLoadBf16x8AsFloat(out_v[1], hi_v);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				const uint32_t pair = pair_base + k;
				if ( ROT_PAIRS == 0u || pair < ROT_PAIRS )
				{
					const float angle = pos * __ldg(inv_freq + pair);
					LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&lo_v[k], &hi_v[k], angle, rope_scale);
				}
			}
			out_v[0] = LmPackUint4FloatToBf16(lo_v);
			out_v[1] = LmPackUint4FloatToBf16(hi_v);
		}
	}
}

template<uint32_t HD, bool WEIGHTED, bool ROUND_FIRST>
static __device__ __forceinline__ void GemmaHeadNormVec(const uint4 *in_v, const float *__restrict__ weight, uint4 *out_v, uint32_t lane,
	float epsilon, float head_multiply)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t SLOTS = (HD_V + 31u) / 32u;
	float total = 0.0f;
	if constexpr ( SLOTS == 1u )
	{
		const float sq = lane < HD_V ? LmSumSqBf16x8(in_v[0], 0.0f) : 0.0f;
		total = GemmaWarpSum(sq);
	}
	else
	{
		#pragma unroll
		for ( uint32_t s = 0u; s < SLOTS; ++s )
		{
			const uint32_t c = s * 32u + lane;
			const float sq = c < HD_V ? LmSumSqBf16x8(in_v[s], 0.0f) : 0.0f;
			total += GemmaWarpSum(sq);
		}
	}
	const float scale = SparkMathRmsScale(total, (float)HD, epsilon);
	#pragma unroll
	for ( uint32_t s = 0u; s < SLOTS; ++s )
	{
		const uint32_t c = s * 32u + lane;
		if ( c < HD_V )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(in_v[s], x);
			if constexpr ( WEIGHTED )
				LmLoadWeight8(weight, c << 3u, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				float v = x[k] * scale;
				if constexpr ( WEIGHTED )
					v = ROUND_FIRST ? SparkMathNormGainRounded(v, w[k]) : v * w[k];
				out[k] = SparkMathRoundBf16(v) * head_multiply;
			}
			out_v[s] = LmPackUint4FloatToBf16(out);
		}
	}
}

template<class G, bool FULL>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaAttnPrepStoreKernel(const uint16_t *__restrict__ gathered, const float *__restrict__ q_norm, const float *__restrict__ k_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row,
	const float *__restrict__ inv_freq, LmKvShardView kv_view, uint16_t *__restrict__ query_out, uint32_t rows)
{
	using S = GemmaShape<G>;
	using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
	constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim;
	constexpr uint32_t KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
	constexpr uint32_t HEADS = FULL ? G::kFullHeads : G::kSlidingHeads;
	constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ;
	constexpr uint32_t KVW = FULL ? S::kFullKv : S::kSlidingKv;
	constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
	constexpr uint32_t LOCAL_V = (FUSED / G::kTpDegree) >> 3u;
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t Q_V = QW >> 3u;
	constexpr uint32_t KV_V = KVW >> 3u;
	constexpr uint32_t SLOTS = (HD_V + 31u) / 32u;
	constexpr uint32_t ROPE_W = FULL ? S::kFullRopeWidth : HD;
	constexpr uint32_t ROT_PAIRS = FULL ? G::kFullRopePairs : 0u;
	constexpr bool DO_ROPE = !FULL || G::kFullRope;
	constexpr bool KEY_IS_VAL = FULL && G::kKeyIsValue;
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	constexpr uint32_t TASKS = HEADS + 2u * KVH;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( task < TASKS && row < rows )
	{
		const uint4 *gathered4 = (const uint4 *)gathered;
		if ( task < HEADS )
		{
			const uint32_t h = task;
			uint4 in_v[SLOTS], out_v[SLOTS];
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t c = s * 32u + lane;
				if ( c < HD_V )
					in_v[s] = GemmaLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, h * HD_V + c);
			}
			GemmaHeadNormVec<HD, G::kQkNormWeighted, G::kNormRoundFirst>(in_v, q_norm, out_v, lane, G::kNormEpsilon, G::kQueryMultiply);
			if constexpr ( DO_ROPE )
				GemmaRopeInReg<HD, ROPE_W, ROT_PAIRS, FULL>(out_v, inv_freq, (float)__ldg(positions + row), FULL ? G::kFullRopeScale : 1.0f, lane);
			uint4 *q4 = (uint4 *)query_out + ((uint64_t)row * HEADS + h) * HD_V;
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t c = s * 32u + lane;
				if ( c < HD_V )
					q4[c] = out_v[s];
			}
		}
		else
		{
			const uint32_t store_pos = __ldg(store_positions + row);
			if ( SparkKvShardOwns(kv_view.shard, store_pos) != 0u )
			{
				uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, __ldg(sequence_of_row + row), store_pos, row, LM_KV_ACCESS_WRITE);
				if ( slot != 0 )
				{
					if ( task < HEADS + KVH )
					{
						const uint32_t kh = task - HEADS;
						uint4 in_v[SLOTS], out_v[SLOTS];
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								in_v[s] = GemmaLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, Q_V + kh * HD_V + c);
						}
						GemmaHeadNormVec<HD, G::kQkNormWeighted, G::kNormRoundFirst>(in_v, k_norm, out_v, lane, G::kNormEpsilon, 1.0f);
						if constexpr ( DO_ROPE )
							GemmaRopeInReg<HD, ROPE_W, ROT_PAIRS, FULL>(out_v, inv_freq, (float)__ldg(positions + row), FULL ? G::kFullRopeScale : 1.0f, lane);
						uint4 *k4 = (uint4 *)slot + kh * HD_V;
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								k4[c] = out_v[s];
						}
					}
					else
					{
						const uint32_t vh = task - HEADS - KVH;
						const uint32_t src_base = (KEY_IS_VAL ? Q_V : Q_V + KV_V) + vh * HD_V;
						uint4 in_v[SLOTS], out_v[SLOTS];
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								in_v[s] = GemmaLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, src_base + c);
						}
						if constexpr ( G::kValueNorm )
							GemmaHeadNormVec<HD, false, false>(in_v, (const float *)0, out_v, lane, G::kNormEpsilon, 1.0f);
						uint4 *v4 = (uint4 *)slot + KV_V + vh * HD_V;
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								v4[c] = G::kValueNorm ? out_v[s] : in_v[s];
						}
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool FULL>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaAttnPrepStoreLocalKernel(const uint16_t *__restrict__ gathered, const float *__restrict__ q_norm, const float *__restrict__ k_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row,
	const float *__restrict__ inv_freq, LmKvShardView kv_view, uint16_t *__restrict__ query_out, uint16_t *__restrict__ kv_cur,
	uint32_t rows, uint32_t q_first, uint32_t rank_heads, uint32_t kv_first, uint32_t local_kvh)
{
	using S = GemmaShape<G>;
	using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
	constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim;
	constexpr uint32_t KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
	constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ;
	constexpr uint32_t KVW = FULL ? S::kFullKv : S::kSlidingKv;
	constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
	constexpr uint32_t LOCAL_V = (FUSED / G::kTpDegree) >> 3u;
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t Q_V = QW >> 3u;
	constexpr uint32_t KV_V = KVW >> 3u;
	constexpr uint32_t SLOTS = (HD_V + 31u) / 32u;
	constexpr uint32_t ROPE_W = FULL ? S::kFullRopeWidth : HD;
	constexpr uint32_t ROT_PAIRS = FULL ? G::kFullRopePairs : 0u;
	constexpr bool DO_ROPE = !FULL || G::kFullRope;
	constexpr bool KEY_IS_VAL = FULL && G::kKeyIsValue;
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	const uint32_t tasks = rank_heads + 2u * KVH;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( task < tasks && row < rows )
	{
		const uint4 *gathered4 = (const uint4 *)gathered;
		if ( task < rank_heads )
		{
			const uint32_t h = q_first + task;
			uint4 in_v[SLOTS], out_v[SLOTS];
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t c = s * 32u + lane;
				if ( c < HD_V )
					in_v[s] = GemmaLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, h * HD_V + c);
			}
			GemmaHeadNormVec<HD, G::kQkNormWeighted, G::kNormRoundFirst>(in_v, q_norm, out_v, lane, G::kNormEpsilon, G::kQueryMultiply);
			if constexpr ( DO_ROPE )
				GemmaRopeInReg<HD, ROPE_W, ROT_PAIRS, FULL>(out_v, inv_freq, (float)__ldg(positions + row), FULL ? G::kFullRopeScale : 1.0f, lane);
			uint4 *q4 = (uint4 *)query_out + ((uint64_t)row * rank_heads + task) * HD_V;
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t c = s * 32u + lane;
				if ( c < HD_V )
					q4[c] = out_v[s];
			}
		}
		else if ( task < rank_heads + KVH )
		{
			const uint32_t kh = task - rank_heads;
			const uint32_t store_pos = __ldg(store_positions + row);
			const bool owns = SparkKvShardOwns(kv_view.shard, store_pos) != 0u;
			const bool need = kh >= kv_first && kh < kv_first + local_kvh;
			if ( owns || need )
			{
				uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, __ldg(sequence_of_row + row), store_pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
				uint4 in_v[SLOTS], out_v[SLOTS];
				#pragma unroll
				for ( uint32_t s = 0u; s < SLOTS; ++s )
				{
					const uint32_t c = s * 32u + lane;
					if ( c < HD_V )
						in_v[s] = GemmaLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, Q_V + kh * HD_V + c);
				}
				GemmaHeadNormVec<HD, G::kQkNormWeighted, G::kNormRoundFirst>(in_v, k_norm, out_v, lane, G::kNormEpsilon, 1.0f);
				if constexpr ( DO_ROPE )
					GemmaRopeInReg<HD, ROPE_W, ROT_PAIRS, FULL>(out_v, inv_freq, (float)__ldg(positions + row), FULL ? G::kFullRopeScale : 1.0f, lane);
				if ( slot != 0 )
				{
					uint4 *k4 = (uint4 *)slot + kh * HD_V;
					#pragma unroll
					for ( uint32_t s = 0u; s < SLOTS; ++s )
					{
						const uint32_t c = s * 32u + lane;
						if ( c < HD_V )
							k4[c] = out_v[s];
					}
				}
				if ( need )
				{
					uint4 *ck4 = (uint4 *)kv_cur + (((uint64_t)(kh - kv_first) * rows + row) * 2u) * HD_V;
					#pragma unroll
					for ( uint32_t s = 0u; s < SLOTS; ++s )
					{
						const uint32_t c = s * 32u + lane;
						if ( c < HD_V )
							ck4[c] = out_v[s];
					}
				}
			}
		}
		else
		{
			const uint32_t vh = task - rank_heads - KVH;
			const uint32_t store_pos = __ldg(store_positions + row);
			const bool owns = SparkKvShardOwns(kv_view.shard, store_pos) != 0u;
			const bool need = vh >= kv_first && vh < kv_first + local_kvh;
			if ( owns || need )
			{
				uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, __ldg(sequence_of_row + row), store_pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
				const uint32_t src_base = (KEY_IS_VAL ? Q_V : Q_V + KV_V) + vh * HD_V;
				uint4 in_v[SLOTS], out_v[SLOTS];
				#pragma unroll
				for ( uint32_t s = 0u; s < SLOTS; ++s )
				{
					const uint32_t c = s * 32u + lane;
					if ( c < HD_V )
						in_v[s] = GemmaLoadRankMajor8Const<LOCAL_V>(gathered4, rows, row, src_base + c);
				}
				if constexpr ( G::kValueNorm )
					GemmaHeadNormVec<HD, false, false>(in_v, (const float *)0, out_v, lane, G::kNormEpsilon, 1.0f);
				if ( slot != 0 )
				{
					uint4 *v4 = (uint4 *)slot + KV_V + vh * HD_V;
					#pragma unroll
					for ( uint32_t s = 0u; s < SLOTS; ++s )
					{
						const uint32_t c = s * 32u + lane;
						if ( c < HD_V )
							v4[c] = G::kValueNorm ? out_v[s] : in_v[s];
					}
				}
				if ( need )
				{
					uint4 *cv4 = (uint4 *)kv_cur + (((uint64_t)(vh - kv_first) * rows + row) * 2u + 1u) * HD_V;
					#pragma unroll
					for ( uint32_t s = 0u; s < SLOTS; ++s )
					{
						const uint32_t c = s * 32u + lane;
						if ( c < HD_V )
							cv4[c] = G::kValueNorm ? out_v[s] : in_v[s];
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class Kv, uint32_t KVH, uint32_t HD>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaPriorKvDirectKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row, uint32_t win_begin, uint32_t prior_keys,
	uint32_t kv_first, uint32_t local_kvh, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t KV_V = KVH * HD_V;
	const uint32_t k_idx = blockIdx.x, lh = blockIdx.y;
	LmDependentWait();
	if ( k_idx < prior_keys && lh < local_kvh )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		const uint32_t pos = win_begin + k_idx;
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, seq, pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint32_t kh = kv_first + lh;
			const uint4 *ks = (const uint4 *)slot + kh * HD_V;
			const uint4 *vs = (const uint4 *)slot + KV_V + kh * HD_V;
			uint4 *kd = (uint4 *)kv_prior + (((uint64_t)lh * prior_keys + k_idx) * 2u) * HD_V;
			uint4 *vd = kd + HD_V;
			for ( uint32_t c = threadIdx.x; c < HD_V; c += GM_THREADS )
			{
				kd[c] = __ldg(ks + c);
				vd[c] = __ldg(vs + c);
			}
		}
	}
	LmDependentRelease();
}

template<class Kv, uint32_t KVH, uint32_t HD>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaPriorKvPackKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row, uint32_t win_begin, uint32_t old_ctx,
	uint32_t b0, uint32_t max_peer_keys, uint32_t rank_heads, uint32_t group, uint32_t local_kvh, uint64_t a2a_stride_bytes, uint8_t *__restrict__ send_bytes)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t KV_V = KVH * HD_V;
	const uint32_t idx = blockIdx.x, peer = blockIdx.y;
	const uint32_t degree = kv_view.shard.degree, grain = kv_view.shard.grain, my_rank = kv_view.shard.rank;
	const uint32_t first_b = b0 + ((my_rank + degree - (b0 % degree)) % degree);
	const uint32_t pos = (first_b + (idx / grain) * degree) * grain + (idx % grain);
	LmDependentWait();
	if ( idx < max_peer_keys && peer < degree && pos >= win_begin && pos < old_ctx )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, seq, pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint32_t peer_kv_first = (peer * rank_heads) / group;
			uint4 *peer_dst = (uint4 *)(send_bytes + (uint64_t)peer * a2a_stride_bytes);
			for ( uint32_t lh = 0u; lh < local_kvh; ++lh )
			{
				const uint32_t kh = peer_kv_first + lh;
				const uint4 *ks = (const uint4 *)slot + kh * HD_V;
				const uint4 *vs = (const uint4 *)slot + KV_V + kh * HD_V;
				uint4 *kd = peer_dst + (((uint64_t)lh * max_peer_keys + idx) * 2u) * HD_V;
				uint4 *vd = kd + HD_V;
				for ( uint32_t c = threadIdx.x; c < HD_V; c += GM_THREADS )
				{
					kd[c] = __ldg(ks + c);
					vd[c] = __ldg(vs + c);
				}
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t HD>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaPriorKvUnpackKernel(SparkKvShard shard, const uint8_t *__restrict__ recv_bytes, uint64_t a2a_stride_bytes, uint32_t win_begin,
	uint32_t old_ctx, uint32_t b0, uint32_t max_peer_keys, uint32_t prior_keys, uint32_t local_kvh, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t HD_V = HD >> 3u;
	const uint32_t idx = blockIdx.x, src_rank = blockIdx.y;
	const uint32_t degree = shard.degree, grain = shard.grain;
	const uint32_t first_b = b0 + ((src_rank + degree - (b0 % degree)) % degree);
	const uint32_t pos = (first_b + (idx / grain) * degree) * grain + (idx % grain);
	LmDependentWait();
	if ( idx < max_peer_keys && src_rank < degree && pos >= win_begin && pos < old_ctx )
	{
		const uint32_t k_idx = pos - win_begin;
		const uint4 *src_base = (const uint4 *)(recv_bytes + (uint64_t)src_rank * a2a_stride_bytes);
		for ( uint32_t lh = 0u; lh < local_kvh; ++lh )
		{
			const uint4 *s4 = src_base + (((uint64_t)lh * max_peer_keys + idx) * 2u) * HD_V;
			uint4 *d4 = (uint4 *)kv_prior + (((uint64_t)lh * prior_keys + k_idx) * 2u) * HD_V;
			for ( uint32_t c = threadIdx.x; c < 2u * HD_V; c += GM_THREADS )
				d4[c] = __ldg(s4 + c);
		}
	}
	LmDependentRelease();
}

template<uint32_t HD>
static __host__ __device__ constexpr uint32_t GemmaLocalPrefillKvTile(void)
{
	return HD <= 128u ? 64u : 32u;
}

template<uint32_t HD>
static __host__ __device__ constexpr uint32_t GemmaLocalPrefillSharedBytes(void)
{
	constexpr uint32_t V_WARPS = HD <= 128u ? 1u : (HD / 128u);
	constexpr uint32_t Q_WARPS = V_WARPS == 1u ? 4u : 2u;
	constexpr uint32_t KV_TILE = GemmaLocalPrefillKvTile<HD>();
	constexpr uint32_t ROW_BYTES = HD * 2u;
	constexpr uint32_t KV_BYTES = 2u * KV_TILE * ROW_BYTES;
	constexpr uint32_t SCORE_BYTES = V_WARPS > 1u ? Q_WARPS * V_WARPS * (KV_TILE / 2u) * 32u * 4u : 0u;
	return KV_BYTES + SCORE_BYTES;
}

template<uint32_t HD>
__global__ __launch_bounds__((HD <= 128u ? 128u : (HD <= 256u ? 128u : 256u)), HD <= 256u ? 2 : 1)
static void GemmaLocalPrefillAttnKernel(const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ kv_prior,
	const uint16_t *__restrict__ kv_cur, const uint32_t *__restrict__ positions, uint32_t rows, uint32_t rank_heads, uint32_t q_first,
	uint32_t group, uint32_t kv_first, uint32_t win_begin, uint32_t prior_keys, uint32_t window, float qk_scale, uint16_t *__restrict__ merged_out)
{
	constexpr uint32_t KV_TILE = GemmaLocalPrefillKvTile<HD>();
	constexpr uint32_t V_WARPS = HD <= 128u ? 1u : (HD / 128u);
	constexpr uint32_t Q_WARPS = V_WARPS == 1u ? 4u : 2u;
	constexpr uint32_t THREADS = Q_WARPS * V_WARPS * 32u;
	constexpr uint32_t Q_TILE = Q_WARPS * 16u;
	constexpr uint32_t PART = HD / V_WARPS;
	constexpr uint32_t K_SLICE_STEPS = PART / 16u;
	constexpr uint32_t VALUE_FRAGS = PART / 8u;
	constexpr uint32_t KEY_FRAGS = KV_TILE / 8u;
	constexpr uint32_t P_STEPS = KV_TILE / 16u;
	constexpr uint32_t CHUNKS = HD / 8u;
	constexpr uint32_t ROW_BYTES = HD * 2u;
	constexpr uint32_t TILE_BYTES = KV_TILE * ROW_BYTES;
	static_assert(HD % 64u == 0u && PART <= 128u && Q_TILE * ROW_BYTES <= 2u * TILE_BYTES, "valid local prefill MMA geometry");
	extern __shared__ __align__(128) uint8_t gm_prefill_smem[];
	uint8_t *k_smem = gm_prefill_smem;
	uint8_t *v_smem = gm_prefill_smem + TILE_BYTES;
	float *score_buf = (float *)(gm_prefill_smem + 2u * TILE_BYTES);
	const uint32_t k_base = (uint32_t)__cvta_generic_to_shared(k_smem);
	const uint32_t v_base = (uint32_t)__cvta_generic_to_shared(v_smem);
	const uint32_t local_h = blockIdx.x, q_block = blockIdx.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t q_warp = warp / V_WARPS, v_warp = warp % V_WARPS;
	const uint32_t row_first = q_block * Q_TILE;
	const uint32_t row_end = row_first + Q_TILE < rows ? row_first + Q_TILE : rows;
	const uint32_t lkvh = ((q_first + local_h) / group) - kv_first;
	const uint32_t total_kv = prior_keys + rows;
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	LmDependentWait();
	if ( row_first >= rows )
	{
		LmDependentRelease();
		return;
	}
	for ( uint32_t idx = threadIdx.x; idx < Q_TILE * CHUNKS; idx += THREADS )
	{
		const uint32_t r = idx / CHUNKS, c = idx % CHUNKS;
		const bool vr = row_first + r < row_end;
		const uint16_t *src = vr ? query_bf16 + ((uint64_t)(row_first + r) * rank_heads + local_h) * HD + c * 8u : query_bf16;
		LmPrefillCopy(k_base + r * ROW_BYTES + LmPrefillSwizzle(r, c) * 16u, src, vr ? 16u : 0u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	uint32_t q_frag[K_SLICE_STEPS][4];
	{
		const uint32_t a_row = q_warp * 16u + (matrix & 1u) * 8u + within;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < K_SLICE_STEPS; ++ks )
		{
			const uint32_t a_chunk = 2u * (v_warp * K_SLICE_STEPS + ks) + (matrix >> 1u);
			LmPrefillLoadX4(k_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row, a_chunk) * 16u, q_frag[ks]);
		}
	}
	__syncthreads();
	const uint32_t min_pos = __ldg(positions + row_first);
	const uint32_t max_pos = __ldg(positions + row_end - 1u);
	const uint32_t cta_start = (window != 0u && min_pos + 1u > window) ? (min_pos + 1u - window) : 0u;
	const uint32_t cta_end = max_pos + 1u;
	const uint32_t tile_first = cta_start > win_begin ? (cta_start - win_begin) / KV_TILE : 0u;
	const uint32_t tile_end = cta_end > win_begin ? (cta_end - win_begin + KV_TILE - 1u) / KV_TILE : 0u;
	const uint32_t row_lo = row_first + q_warp * 16u + (lane >> 2u);
	const uint32_t row_hi = row_lo + 8u;
	const bool valid_row[2] = { row_lo < row_end, row_hi < row_end };
	const uint32_t rpos[2] = { valid_row[0] ? __ldg(positions + row_lo) : 0u, valid_row[1] ? __ldg(positions + row_hi) : 0u };
	const uint32_t rstart[2] = {
		(window != 0u && rpos[0] + 1u > window) ? (rpos[0] + 1u - window) : 0u,
		(window != 0u && rpos[1] + 1u > window) ? (rpos[1] + 1u - window) : 0u
	};
	auto copy_kv = [&](uint32_t tile, uint32_t dst_base, uint32_t is_v)
	{
		for ( uint32_t idx = threadIdx.x; idx < KV_TILE * CHUNKS; idx += THREADS )
		{
			const uint32_t s = idx / CHUNKS, c = idx % CHUNKS;
			const uint32_t k_idx = tile * KV_TILE + s;
			const bool valid = k_idx < total_kv;
			const uint16_t *src = valid ?
				(k_idx < prior_keys ?
					kv_prior + (((uint64_t)lkvh * prior_keys + k_idx) * 2u + is_v) * HD + c * 8u :
					kv_cur + (((uint64_t)lkvh * rows + (k_idx - prior_keys)) * 2u + is_v) * HD + c * 8u) :
				query_bf16;
			LmPrefillCopy(dst_base + s * ROW_BYTES + LmPrefillSwizzle(s, c) * 16u, src, valid ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	float output[VALUE_FRAGS][4];
	#pragma unroll
	for ( uint32_t f = 0u; f < VALUE_FRAGS; ++f )
		output[f][0] = output[f][1] = output[f][2] = output[f][3] = 0.0f;
	float running_max[2] = { -INFINITY, -INFINITY };
	float running_sum[2] = { 0.0f, 0.0f };
	if ( tile_first < tile_end )
		copy_kv(tile_first, k_base, 0u);
	for ( uint32_t tile = tile_first; tile < tile_end; ++tile )
	{
		copy_kv(tile, v_base, 1u);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		__syncthreads();
		float score[KEY_FRAGS][4];
		#pragma unroll
		for ( uint32_t f = 0u; f < KEY_FRAGS; ++f )
			score[f][0] = score[f][1] = score[f][2] = score[f][3] = 0.0f;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < K_SLICE_STEPS; ++ks )
		{
			const uint32_t k_step = v_warp * K_SLICE_STEPS + ks;
			#pragma unroll
			for ( uint32_t f = 0u; f < KEY_FRAGS; f += 2u )
			{
				uint32_t b[4], first[2], second[2];
				const uint32_t b_row = (f + (matrix >> 1u)) * 8u + within;
				const uint32_t b_chunk = 2u * k_step + (matrix & 1u);
				LmPrefillLoadX4(k_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(score[f], q_frag[ks], first);
				LmMmaBf16(score[f + 1u], q_frag[ks], second);
			}
		}
		if constexpr ( V_WARPS > 1u )
		{
			float *my_score = score_buf + (q_warp * V_WARPS + v_warp) * 16u * 32u;
			#pragma unroll
			for ( uint32_t f = 0u; f < KEY_FRAGS; ++f )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					my_score[(f * 4u + e) * 32u + lane] = score[f][e];
		}
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( tile + 1u < tile_end )
			copy_kv(tile + 1u, k_base, 0u);
		if constexpr ( V_WARPS > 1u )
		{
			const float *tile_score = score_buf + (q_warp * V_WARPS) * 16u * 32u;
			#pragma unroll
			for ( uint32_t f = 0u; f < KEY_FRAGS; ++f )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
				{
					const uint32_t off = (f * 4u + e) * 32u + lane;
					float s = tile_score[off];
					#pragma unroll
					for ( uint32_t g = 1u; g < V_WARPS; ++g )
						s += tile_score[g * 16u * 32u + off];
					score[f][e] = s;
				}
		}
		const uint32_t tile_pos0 = win_begin + tile * KV_TILE;
		const uint32_t tile_pos1 = tile_pos0 + KV_TILE - 1u;
		const bool whole = __all_sync(0xffffffffu, valid_row[0] && valid_row[1] &&
			tile_pos0 >= rstart[0] && tile_pos1 <= rpos[0] && tile_pos0 >= rstart[1] && tile_pos1 <= rpos[1]);
		float peak[2] = { -INFINITY, -INFINITY };
		if ( whole )
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < KEY_FRAGS; ++f )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					peak[e >> 1u] = fmaxf(peak[e >> 1u], score[f][e]);
		}
		else
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < KEY_FRAGS; ++f )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
				{
					const uint32_t key_pos = tile_pos0 + f * 8u + 2u * (lane & 3u) + (e & 1u);
					const uint32_t side = e >> 1u;
					const bool vis = valid_row[side] && key_pos >= rstart[side] && key_pos <= rpos[side];
					score[f][e] = vis ? score[f][e] : -INFINITY;
					peak[side] = fmaxf(peak[side], score[f][e]);
				}
		}
		float rescale[2], shift[2];
		#pragma unroll
		for ( uint32_t side = 0u; side < 2u; ++side )
		{
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 1u));
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 2u));
			const float next_max = fmaxf(running_max[side], peak[side]);
			rescale[side] = running_max[side] == -INFINITY ? 0.0f : SparkMathExp2Fast((running_max[side] - next_max) * scale_log2);
			running_max[side] = next_max;
			shift[side] = next_max == -INFINITY ? 0.0f : next_max * scale_log2;
			running_sum[side] *= rescale[side];
		}
		if ( rescale[0] != 1.0f || rescale[1] != 1.0f )
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < VALUE_FRAGS; ++f )
			{
				output[f][0] *= rescale[0];
				output[f][1] *= rescale[0];
				output[f][2] *= rescale[1];
				output[f][3] *= rescale[1];
			}
		}
		#pragma unroll
		for ( uint32_t f = 0u; f < KEY_FRAGS; ++f )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
			{
				const uint32_t side = e >> 1u;
				const float wt = (running_max[side] == -INFINITY || score[f][e] == -INFINITY) ? 0.0f :
					SparkMathExp2Fast(fmaf(score[f][e], scale_log2, -shift[side]));
				running_sum[side] += wt;
				score[f][e] = wt;
			}
		#pragma unroll
		for ( uint32_t step = 0u; step < P_STEPS; ++step )
		{
			uint32_t weights[4];
			#pragma unroll
			for ( uint32_t i = 0u; i < 4u; ++i )
			{
				const float *pair = &score[2u * step + (i >> 1u)][2u * (i & 1u)];
				weights[i] = LmPackBf16Pair(pair[0], pair[1]);
			}
			#pragma unroll
			for ( uint32_t f = 0u; f < VALUE_FRAGS; f += 2u )
			{
				uint32_t b[4], first[2], second[2];
				const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
				const uint32_t b_chunk = v_warp * (PART / 8u) + f + (matrix >> 1u);
				LmPrefillLoadX4Trans(v_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(output[f], weights, first);
				LmMmaBf16(output[f + 1u], weights, second);
			}
		}
		if ( tile + 1u < tile_end )
			__syncthreads();
	}
	#pragma unroll
	for ( uint32_t side = 0u; side < 2u; ++side )
	{
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 2u);
	}
	const float inv_lo = valid_row[0] && running_sum[0] > 0.0f ? (1.0f / running_sum[0]) : 0.0f;
	const float inv_hi = valid_row[1] && running_sum[1] > 0.0f ? (1.0f / running_sum[1]) : 0.0f;
	#pragma unroll
	for ( uint32_t f = 0u; f < VALUE_FRAGS; ++f )
	{
		const uint32_t col = v_warp * PART + f * 8u + (lane & 3u) * 2u;
		if ( valid_row[0] )
			__stcs((uint32_t *)(merged_out + ((uint64_t)row_lo * rank_heads + local_h) * HD + col),
				LmPackBf16Pair(output[f][0] * inv_lo, output[f][1] * inv_lo));
		if ( valid_row[1] )
			__stcs((uint32_t *)(merged_out + ((uint64_t)row_hi * rank_heads + local_h) * HD + col),
				LmPackBf16Pair(output[f][2] * inv_hi, output[f][3] * inv_hi));
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaSmallNBf16ProjKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation, uint16_t *__restrict__ output,
	uint32_t rows, uint32_t output_dim)
{
	constexpr uint32_t VECS = HIDDEN >> 3u;
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	__shared__ __align__(16) uint4 s_x[VECS];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row < rows )
	{
		const uint4 *x4 = (const uint4 *)(activation + (uint64_t)row * HIDDEN);
		for ( uint32_t c = threadIdx.x; c < VECS; c += GM_THREADS )
			s_x[c] = x4[c];
		__syncthreads();
		for ( uint32_t n = warp; n < output_dim; n += WARPS )
		{
			const uint4 *w4 = (const uint4 *)(weight + (uint64_t)n * HIDDEN);
			float acc = 0.0f;
			for ( uint32_t c = lane; c < VECS; c += LM_WARP_LANES )
				acc = LmDotBf16x8(s_x[c], __ldg(&w4[c]), acc);
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				acc += __shfl_down_sync(0xffffffffu, acc, offset);
			if ( lane == 0u )
				output[(uint64_t)row * output_dim + n] = LmFloatToBf16(acc);
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmDependentWait();
	LmGateUpRowBody<THREADS>(gate_bf16 + base, output_bf16 + base, output_bf16 + base, dimension, LmSigmoidGateOp<true>{});
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaHeadGateBroadcastKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads,
	uint32_t head_dimension, uint32_t gate_stride)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	uint16_t *base = heads_bf16 + ((((uint64_t)row * heads) + head) * head_dimension);
	LmDependentWait();
	const float gate = SparkMathSoftplusGate(LmBf16ToFloat(gate_bf16[((uint64_t)row * gate_stride) + head]));
	for ( uint32_t index = threadIdx.x * 8u; index < head_dimension; index += THREADS * 8u )
	{
		float v[8];
		LmLoadBf16x8AsFloat(*(const uint4 *)(base + index), v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			v[k] *= gate;
		*(uint4 *)(base + index) = LmPackUint4FloatToBf16(v);
	}
	LmDependentRelease();
}

struct GemmaGeluTanhOp
{
	__device__ __forceinline__ float operator()(float g, float u) const
	{
		return SparkMathRoundBf16(SparkMathGeluTanh(g)) * u;
	}
};

template<uint32_t THREADS, bool ROUND_FIRST, bool HAS_ADD, uint32_t SLOTS = 4u>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaAddRmsNormKernel(const uint16_t *__restrict__ input_bf16, uint16_t *__restrict__ residual_bf16, const float *__restrict__ weight,
	uint16_t *__restrict__ output_bf16, uint32_t width, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *in_row = input_bf16 + base;
	uint16_t *res_row = HAS_ADD ? residual_bf16 + base : (uint16_t *)0;
	uint16_t *out_row = output_bf16 + base;
	uint4 staged[SLOTS];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			uint4 v = *(const uint4 *)(in_row + offset);
			if constexpr ( HAS_ADD )
			{
				float a[8], b[8];
				LmLoadBf16x8AsFloat(v, a);
				LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), b);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					a[k] += b[k];
				v = LmPackUint4FloatToBf16(a);
				*(uint4 *)(res_row + offset) = v;
			}
			staged[slot] = v;
			total = LmSumSqBf16x8(v, total);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(staged[slot], x);
			LmLoadWeight8(weight, offset, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale, w[k]) : x[k] * scale * w[k];
			*(uint4 *)(out_row + offset) = LmPackUint4FloatToBf16(out);
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, bool ROUND_FIRST, bool HAS_SCALAR, bool TRIPLE_OUT, uint32_t SLOTS = 4u>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaPostNormAddNextKernel(uint16_t *__restrict__ residual, const uint16_t *__restrict__ partial, const float *__restrict__ post_weight,
	const float *__restrict__ scalar, const float *__restrict__ next_weight, uint16_t *__restrict__ next_out,
	const float *__restrict__ router_scale, uint16_t *__restrict__ router_out, const float *__restrict__ moe_weight,
	uint16_t *__restrict__ moe_out, uint32_t width, float post_eps, float norm_eps, float router_root)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *part_row = partial + base;
	uint16_t *res_row = residual + base;
	uint4 staged[SLOTS];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			staged[slot] = *(const uint4 *)(part_row + offset);
			total = LmSumSqBf16x8(staged[slot], total);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, post_eps);
	const float mult = HAS_SCALAR ? __ldg(scalar) : 1.0f;
	float total2 = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float p[8], w[8], r[8], out[8];
			LmLoadBf16x8AsFloat(staged[slot], p);
			LmLoadWeight8(post_weight, offset, w);
			LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), r);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				const float added = SparkMathRoundBf16(r[k] + SparkMathRoundBf16(p[k] * scale * w[k]));
				out[k] = HAS_SCALAR ? added * mult : added;
			}
			const uint4 packed = LmPackUint4FloatToBf16(out);
			*(uint4 *)(res_row + offset) = packed;
			staged[slot] = packed;
			if ( next_out != 0 )
				total2 = LmSumSqBf16x8(packed, total2);
		}
	}
	if ( next_out != 0 )
	{
		__syncthreads();
		const float scale2 = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total2, reduction), (float)width, norm_eps);
		uint16_t *n_row = next_out + base;
		uint16_t *rt_row = TRIPLE_OUT ? router_out + base : (uint16_t *)0;
		uint16_t *m_row = TRIPLE_OUT ? moe_out + base : (uint16_t *)0;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
		{
			const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
			if ( offset < width )
			{
				float x[8], w[8], out[8];
				LmLoadBf16x8AsFloat(staged[slot], x);
				LmLoadWeight8(next_weight, offset, w);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale2, w[k]) : x[k] * scale2 * w[k];
				*(uint4 *)(n_row + offset) = LmPackUint4FloatToBf16(out);
				if constexpr ( TRIPLE_OUT )
				{
					float rs[8], mw[8], rt[8], mo[8];
					LmLoadWeight8(router_scale, offset, rs);
					LmLoadWeight8(moe_weight, offset, mw);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
					{
						rt[k] = SparkMathRoundBf16(SparkMathRoundBf16(x[k] * scale2) * rs[k]) * router_root;
						mo[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale2, mw[k]) : x[k] * scale2 * mw[k];
					}
					*(uint4 *)(rt_row + offset) = LmPackUint4FloatToBf16(rt);
					*(uint4 *)(m_row + offset) = LmPackUint4FloatToBf16(mo);
				}
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, bool ROUND_FIRST>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaMoeCombineNextKernel(uint16_t *__restrict__ residual, const uint16_t *__restrict__ dense, const uint16_t *__restrict__ moe,
	const float *__restrict__ dense_norm, const float *__restrict__ moe_norm, const float *__restrict__ post_norm, const float *__restrict__ scalar,
	const float *__restrict__ next_weight, uint16_t *__restrict__ next_out, uint32_t width, float post_eps, float norm_eps)
{
	__shared__ float reduction[2u * LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *dense_row = dense + base, *moe_row = moe + base;
	uint16_t *res_row = residual + base;
	uint4 d_staged[2], m_staged[2];
	float dense_total = 0.0f, moe_total = 0.0f, total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 2u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			d_staged[slot] = *(const uint4 *)(dense_row + offset);
			m_staged[slot] = *(const uint4 *)(moe_row + offset);
			dense_total = LmSumSqBf16x8(d_staged[slot], dense_total);
			moe_total = LmSumSqBf16x8(m_staged[slot], moe_total);
		}
	}
	LmBlockSum2Once<THREADS>(&dense_total, &moe_total, reduction);
	__syncthreads();
	const float dense_scale = SparkMathRmsScale(dense_total, (float)width, post_eps);
	const float moe_scale = SparkMathRmsScale(moe_total, (float)width, post_eps);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 2u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float dv[8], mv[8], dn[8], mn[8], comb[8];
			LmLoadBf16x8AsFloat(d_staged[slot], dv);
			LmLoadBf16x8AsFloat(m_staged[slot], mv);
			LmLoadWeight8(dense_norm, offset, dn);
			LmLoadWeight8(moe_norm, offset, mn);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				const float c = SparkMathRoundBf16(SparkMathRoundBf16(dv[k] * dense_scale * dn[k]) + SparkMathRoundBf16(mv[k] * moe_scale * mn[k]));
				comb[k] = c;
				total += c * c;
			}
			d_staged[slot] = LmPackUint4FloatToBf16(comb);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, post_eps);
	const float multiplier = __ldg(scalar);
	float total2 = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 2u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float comb[8], pn[8], rv[8], out[8];
			LmLoadBf16x8AsFloat(d_staged[slot], comb);
			LmLoadWeight8(post_norm, offset, pn);
			LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), rv);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathRoundBf16(rv[k] + SparkMathRoundBf16(comb[k] * scale * pn[k])) * multiplier;
			const uint4 packed = LmPackUint4FloatToBf16(out);
			*(uint4 *)(res_row + offset) = packed;
			d_staged[slot] = packed;
			if ( next_out != 0 )
				total2 = LmSumSqBf16x8(packed, total2);
		}
	}
	if ( next_out != 0 )
	{
		__syncthreads();
		const float scale2 = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total2, reduction), (float)width, norm_eps);
		uint16_t *n_row = next_out + base;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < 2u; ++slot )
		{
			const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
			if ( offset < width )
			{
				float x[8], w[8], out[8];
				LmLoadBf16x8AsFloat(d_staged[slot], x);
				LmLoadWeight8(next_weight, offset, w);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale2, w[k]) : x[k] * scale2 * w[k];
				*(uint4 *)(n_row + offset) = LmPackUint4FloatToBf16(out);
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaGeluTanhMulKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width)
{
	const uint64_t base = (uint64_t)blockIdx.y * 2u * width;
	LmDependentWait();
	LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, GemmaGeluTanhOp{});
	LmDependentRelease();
}

template<uint32_t THREADS, bool HAS_LIMIT>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaSiluMulKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.y * 2u * width;
	LmDependentWait();
	if constexpr ( HAS_LIMIT )
		LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, LmSiluGateLimitOp{limit});
	else
		LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, LmStridedGateUpOp<true>{});
	LmDependentRelease();
}

template<uint32_t EXPERTS, uint32_t TOPK, bool BF16_INPUT>
static __device__ __forceinline__ void GemmaSigmoidTopkWarp(const void *logits_ptr, const float *__restrict__ bias, uint32_t row,
	uint32_t expert_first, uint32_t local_experts, float scale, uint32_t *__restrict__ fault, uint32_t *__restrict__ out_exp,
	float *__restrict__ out_wt, uint32_t lane)
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
			const float lv = BF16_INPUT ? LmBf16ToFloat(((const uint16_t *)logits_ptr)[row_base + e]) : ((const float *)logits_ptr)[row_base + e];
			const float s = SparkMathSigmoid(lv);
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
	uint32_t chosen_e[TOPK];
	float chosen_s[TOPK];
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
		chosen_e[pick] = e;
		chosen_s[pick] = picked_s;
		total += picked_s;
		if ( lane == 0u )
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
	}
	if ( lane == 0u )
	{
		const float denom = total + 1e-20f;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			const uint32_t ge = chosen_e[pick];
			if ( ge >= expert_first && ge < expert_first + local_experts )
			{
				out_exp[pick] = ge - expert_first;
				out_wt[pick] = chosen_s[pick] / denom * scale;
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
__global__ __launch_bounds__(GM_THREADS, 1)
static void GemmaSigmoidRouteSmallKernel(const void *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ packed_expert,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t tile_m, uint32_t n_up, uint32_t *__restrict__ tile_prefix_up, uint32_t n_down,
	uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, routes = rows * G::kTopK;
	LmDependentWait();
	if ( warp < rows )
		GemmaSigmoidTopkWarp<G::kExperts, G::kTopK, !G::kRouterFloat>(logits, bias, warp, expert_first, G::kLocalExperts, G::kRoutedScale, fault,
			s_exp + warp * G::kTopK, s_wt + warp * G::kTopK, lane);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	LmRouteBuildSmallBody<GM_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
		tile_m, n_up, tile_prefix_up, n_down, tile_prefix_down);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		const uint32_t packed = route_packed_row[threadIdx.x];
		packed_weight[packed] = s_wt[threadIdx.x];
		packed_expert[packed] = s_exp[threadIdx.x];
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaSigmoidTopkLocalKernel(const void *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x * WARPS + warp;
	LmDependentWait();
	if ( row < rows )
		GemmaSigmoidTopkWarp<G::kExperts, G::kTopK, !G::kRouterFloat>(logits, bias, row, expert_first, G::kLocalExperts, G::kRoutedScale, fault,
			route_expert + (uint64_t)row * G::kTopK, route_weight + (uint64_t)row * G::kTopK, lane);
	LmDependentRelease();
}

template<class G>
static __device__ __forceinline__ void GemmaSoftmaxTopkWarp(const uint16_t *__restrict__ logits, const float *__restrict__ expert_scale, uint32_t row,
	uint32_t expert_first, uint32_t *__restrict__ fault, uint32_t *__restrict__ out_exp, float *__restrict__ out_wt, uint32_t lane)
{
	constexpr uint32_t EXPERTS = G::kExperts;
	constexpr uint32_t PER_LANE = (EXPERTS + 31u) / 32u;
	const uint64_t row_base = (uint64_t)row * EXPERTS;
	float vals[PER_LANE];
	float top = -INFINITY;
	#pragma unroll
	for ( uint32_t j = 0u; j < PER_LANE; ++j )
	{
		const uint32_t e = j * 32u + lane;
		vals[j] = e < EXPERTS ? LmBf16ToFloat(logits[row_base + e]) : -INFINITY;
		top = fmaxf(top, vals[j]);
	}
	#pragma unroll
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
	float wsum[PER_LANE];
	#pragma unroll
	for ( uint32_t j = 0u; j < PER_LANE; ++j )
	{
		const uint32_t e = j * 32u + lane;
		vals[j] = e < EXPERTS ? expf(vals[j] - top) : 0.0f;
		wsum[j] = GemmaWarpSum(vals[j]);
	}
	float sum = 0.0f;
	if constexpr ( PER_LANE == 1u )
		sum = wsum[0];
	else if constexpr ( PER_LANE == 2u )
		sum = wsum[0] + wsum[1];
	else if constexpr ( PER_LANE == 4u )
		sum = (wsum[0] + wsum[2]) + (wsum[1] + wsum[3]);
	else
	{
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			sum += wsum[j];
	}
	uint64_t key[PER_LANE];
	#pragma unroll
	for ( uint32_t j = 0u; j < PER_LANE; ++j )
	{
		const uint32_t e = j * 32u + lane;
		if ( e < EXPERTS )
		{
			vals[j] /= sum;
			key[j] = ((uint64_t)LmTopkRouteKey(vals[j]) << 32u) | (uint64_t)(0xffffffffu - e);
		}
		else
		{
			vals[j] = 0.0f;
			key[j] = 0ull;
		}
	}
	uint32_t chosen = 0u, high = 0u;
	float my_p = 0.0f;
	#pragma unroll
	for ( uint32_t round = 0u; round < G::kTopK; ++round )
	{
		uint64_t best = 0ull;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			best = key[j] > best ? key[j] : best;
		high = __reduce_max_sync(0xffffffffu, (uint32_t)(best >> 32u));
		const uint32_t low = __reduce_max_sync(0xffffffffu, (uint32_t)(best >> 32u) == high ? (uint32_t)best : 0u);
		const uint64_t winner = ((uint64_t)high << 32u) | (uint64_t)low;
		const uint32_t e = 0xffffffffu - low;
		float picked_p = 0.0f;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			if ( key[j] == winner )
			{
				picked_p = vals[j];
				key[j] = 0ull;
			}
		picked_p = __shfl_sync(0xffffffffu, picked_p, (int)(e & 31u));
		if ( lane == round )
		{
			chosen = e;
			my_p = picked_p;
		}
	}
	if ( lane == 0u )
		LmTopkRouteFault(fault, high);
	constexpr uint32_t WIDTH = G::kTopK <= 1u ? 1u : G::kTopK <= 2u ? 2u : G::kTopK <= 4u ? 4u : G::kTopK <= 8u ? 8u : G::kTopK <= 16u ? 16u : 32u;
	float tree = lane < G::kTopK ? my_p : 0.0f;
	#pragma unroll
	for ( uint32_t step = WIDTH / 2u; step > 0u; step >>= 1u )
	{
		const float other = __shfl_down_sync(0xffffffffu, tree, step);
		tree = lane < step ? tree + other : tree;
	}
	const float total = __shfl_sync(0xffffffffu, tree, 0) + 1e-20f;
	if ( lane < G::kTopK )
	{
		const float wt = (my_p / total) * __ldg(expert_scale + chosen);
		if ( chosen >= expert_first && chosen < expert_first + G::kLocalExperts )
		{
			out_exp[lane] = chosen - expert_first;
			out_wt[lane] = wt;
		}
		else
		{
			out_exp[lane] = G::kLocalExperts;
			out_wt[lane] = 0.0f;
		}
	}
}

template<class G>
__global__ __launch_bounds__(GM_THREADS, 1)
static void GemmaSoftmaxRouteSmallKernel(const uint16_t *__restrict__ logits, const float *__restrict__ expert_scale, uint32_t rows,
	uint32_t expert_first, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ packed_expert, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, float *__restrict__ packed_weight, uint32_t tile_m, uint32_t n_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t n_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	const uint32_t routes = rows * G::kTopK, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( warp < rows )
		GemmaSoftmaxTopkWarp<G>(logits, expert_scale, warp, expert_first, fault, s_exp + warp * G::kTopK, s_wt + warp * G::kTopK, lane);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	LmRouteBuildSmallBody<GM_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
		tile_m, n_up, tile_prefix_up, n_down, tile_prefix_down);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		const uint32_t packed = route_packed_row[threadIdx.x];
		packed_weight[packed] = s_wt[threadIdx.x];
		packed_expert[packed] = s_exp[threadIdx.x];
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaSoftmaxTopkLocalKernel(const uint16_t *__restrict__ logits, const float *__restrict__ expert_scale, uint32_t rows,
	uint32_t expert_first, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x * WARPS + warp;
	LmDependentWait();
	if ( row < rows )
		GemmaSoftmaxTopkWarp<G>(logits, expert_scale, row, expert_first, fault, route_expert + (uint64_t)row * G::kTopK,
			route_weight + (uint64_t)row * G::kTopK, lane);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
static void GemmaRouteBuildPackedKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint32_t routes,
	uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t *__restrict__ packed_expert, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	LmDependentWait();
	LmRouteBuildBlock<THREADS, EXPERTS>(route_expert, routes, top_k, group_row_offset, route_packed_row, route_source_token, tile_m,
		neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	__syncthreads();
	for ( uint32_t route = threadIdx.x; route < routes; route += THREADS )
	{
		const uint32_t packed = route_packed_row[route];
		packed_weight[packed] = route_weight[route];
		packed_expert[packed] = route_expert[route];
	}
	LmDependentRelease();
}

template<bool GELU>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeActPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < group_row_offset[local_experts] )
	{
		const uint64_t row_base = (uint64_t)pr * width;
		const uint4 *g4 = (const uint4 *)(gate + row_base);
		const uint4 *u4 = (const uint4 *)(up + row_base);
		uint4 *o4 = (uint4 *)(output + row_base);
		const float wt = weight[pr];
		const uint32_t chunks = width >> 3u;
		for ( uint32_t c = blockIdx.x * GM_THREADS + threadIdx.x; c < chunks; c += gridDim.x * GM_THREADS )
		{
			float gv[8], uv[8];
			LmLoadBf16x8AsFloat(g4[c], gv);
			LmLoadBf16x8AsFloat(u4[c], uv);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				if constexpr ( GELU )
					gv[k] = SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(gv[k])) * uv[k] * wt;
				else
					gv[k] = SparkMathGateLimitedGlu(gv[k], uv[k], limit) * wt;
			}
			o4[c] = LmPackUint4FloatToBf16(gv);
		}
	}
	LmDependentRelease();
}

template<bool HAS_SHARED>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeLocalFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK], active_count;
	const uint32_t row = blockIdx.y;
	const uint64_t row_base = (uint64_t)row * width;
	uint16_t *out_row = output + row_base;
	const uint16_t *shared_row = HAS_SHARED ? shared + row_base : (const uint16_t *)0;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t count = 0u;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				active_rows[count++] = route_packed_row[route];
		}
		active_count = count;
	}
	__syncthreads();
	const uint32_t count = active_count;
	const uint32_t chunks = width >> 3u;
	const uint4 *packed4 = (const uint4 *)packed;
	const uint4 *shared4 = HAS_SHARED ? (const uint4 *)shared_row : (const uint4 *)0;
	uint4 *out4 = (uint4 *)out_row;
	if ( count == 0u )
	{
		const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
		for ( uint32_t c = blockIdx.x * GM_THREADS + threadIdx.x; c < chunks; c += gridDim.x * GM_THREADS )
			out4[c] = HAS_SHARED ? shared4[c] : zero;
	}
	else
	{
		for ( uint32_t c = blockIdx.x * GM_THREADS + threadIdx.x; c < chunks; c += gridDim.x * GM_THREADS )
		{
			float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t k = 0u; k < count; ++k )
				LmAddBf16x8(acc, packed4[(uint64_t)active_rows[k] * chunks + c]);
			if constexpr ( HAS_SHARED )
				LmAddBf16x8(acc, shared4[c]);
			out4[c] = LmPackUint4FloatToBf16(acc);
		}
	}
	LmDependentRelease();
}

template<class G, bool GELU>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeB1W1W3ActKernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3, const uint16_t *__restrict__ activation,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ packed_expert, const uint32_t *__restrict__ group_row_offset,
	uint16_t *__restrict__ output, float limit)
{
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	constexpr uint32_t K_V = G::kHidden >> 3u;
	constexpr uint32_t N = G::kMoeInter;
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < __ldg(group_row_offset + G::kLocalExperts) )
	{
		const uint32_t exp = __ldg(packed_expert + pr);
		const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
		const uint32_t col0 = (blockIdx.x * WARPS + warp) * 2u;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const uint64_t w_off0 = ((uint64_t)exp * N + col0) * K_V;
			const uint4 *w1_0 = (const uint4 *)w1 + w_off0;
			const uint4 *w3_0 = (const uint4 *)w3 + w_off0;
			const uint4 *w1_1 = has1 ? w1_0 + K_V : w1_0;
			const uint4 *w3_1 = has1 ? w3_0 + K_V : w3_0;
			const uint4 *act_row = (const uint4 *)activation;
			if ( lane == 0u )
			{
				LmPrefetchL2(w1_0, K_V * 16u);
				LmPrefetchL2(w3_0, K_V * 16u);
				if ( has1 )
				{
					LmPrefetchL2(w1_1, K_V * 16u);
					LmPrefetchL2(w3_1, K_V * 16u);
				}
			}
			float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
			#pragma unroll 4
			for ( uint32_t c = lane; c < K_V; c += LM_WARP_LANES )
			{
				const uint4 va = __ldg(act_row + c);
				const uint4 vw1_0 = __ldcs(w1_0 + c);
				const uint4 vw3_0 = __ldcs(w3_0 + c);
				const uint4 vw1_1 = __ldcs(w1_1 + c);
				const uint4 vw3_1 = __ldcs(w3_1 + c);
				g0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1_0, va, 0.0f), g0);
				u0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw3_0, va, 0.0f), u0);
				g1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1_1, va, 0.0f), g1);
				u1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw3_1, va, 0.0f), u1);
			}
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			{
				g0 += __shfl_down_sync(0xffffffffu, g0, offset);
				u0 += __shfl_down_sync(0xffffffffu, u0, offset);
				g1 += __shfl_down_sync(0xffffffffu, g1, offset);
				u1 += __shfl_down_sync(0xffffffffu, u1, offset);
			}
			if ( lane == 0u )
			{
				const float wt = __ldg(packed_weight + pr);
				const float rg0 = SparkMathRoundBf16(g0), ru0 = SparkMathRoundBf16(u0);
				output[(uint64_t)pr * N + col0] = LmFloatToBf16((GELU ? SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(rg0)) * ru0 : SparkMathGateLimitedGlu(rg0, ru0, limit)) * wt);
				if ( has1 )
				{
					const float rg1 = SparkMathRoundBf16(g1), ru1 = SparkMathRoundBf16(u1);
					output[(uint64_t)pr * N + col0 + 1u] = LmFloatToBf16((GELU ? SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(rg1)) * ru1 : SparkMathGateLimitedGlu(rg1, ru1, limit)) * wt);
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool HAS_SHARED>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeB1W2FinalizeKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared,
	uint16_t *__restrict__ output)
{
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t K_V = K >> 3u;
	constexpr uint32_t N = G::kHidden;
	constexpr bool HALF_WARP = (K == 128u);
	constexpr uint32_t SUB_LANES = HALF_WARP ? 16u : 32u;
	constexpr uint32_t NPG = HALF_WARP ? 4u : 2u;
	constexpr uint32_t GROUPS = GM_THREADS / SUB_LANES;
	__shared__ uint32_t s_exp[G::kTopK], s_pr[G::kTopK], s_count;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t cnt = 0u;
		for ( uint32_t k = 0u; k < G::kTopK; ++k )
		{
			const uint32_t e = route_expert[k];
			if ( e < G::kLocalExperts )
			{
				s_exp[cnt] = e;
				s_pr[cnt] = route_packed_row[k];
				cnt++;
			}
		}
		s_count = cnt;
	}
	__syncthreads();
	const uint32_t col0 = (blockIdx.x * GROUPS + (threadIdx.x / SUB_LANES)) * NPG;
	const uint32_t sub = threadIdx.x & (SUB_LANES - 1u);
	if ( col0 < N )
	{
		const uint32_t count = s_count;
		if ( count == 0u )
		{
			if ( sub == 0u )
				#pragma unroll
				for ( uint32_t n = 0u; n < NPG; ++n )
					if ( col0 + n < N )
						output[col0 + n] = HAS_SHARED ? shared[col0 + n] : 0u;
		}
		else if constexpr ( HALF_WARP )
		{
			float total[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			const uint4 *w_cur = (const uint4 *)w2 + ((uint64_t)s_exp[0] * N + col0) * K_V + sub;
			uint4 va_cur = __ldg((const uint4 *)expert_act + (uint64_t)s_pr[0] * K_V + sub);
			uint4 w0_cur = __ldcs(w_cur);
			uint4 w1_cur = __ldcs(w_cur + K_V);
			uint4 w2_cur = __ldcs(w_cur + 2u * K_V);
			uint4 w3_cur = __ldcs(w_cur + 3u * K_V);
			for ( uint32_t k = 0u; k < count; ++k )
			{
				uint4 va_nxt, w0_nxt, w1_nxt, w2_nxt, w3_nxt;
				if ( k + 1u < count )
				{
					const uint4 *w_nxt = (const uint4 *)w2 + ((uint64_t)s_exp[k + 1u] * N + col0) * K_V + sub;
					va_nxt = __ldg((const uint4 *)expert_act + (uint64_t)s_pr[k + 1u] * K_V + sub);
					w0_nxt = __ldcs(w_nxt);
					w1_nxt = __ldcs(w_nxt + K_V);
					w2_nxt = __ldcs(w_nxt + 2u * K_V);
					w3_nxt = __ldcs(w_nxt + 3u * K_V);
				}
				const float d0 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w0_cur, va_cur, 0.0f));
				const float d1 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w1_cur, va_cur, 0.0f));
				const float d2 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w2_cur, va_cur, 0.0f));
				const float d3 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w3_cur, va_cur, 0.0f));
				if ( sub == 0u )
				{
					total[0] += SparkMathRoundBf16(d0);
					total[1] += SparkMathRoundBf16(d1);
					total[2] += SparkMathRoundBf16(d2);
					total[3] += SparkMathRoundBf16(d3);
				}
				if ( k + 1u < count )
				{
					va_cur = va_nxt;
					w0_cur = w0_nxt;
					w1_cur = w1_nxt;
					w2_cur = w2_nxt;
					w3_cur = w3_nxt;
				}
			}
			if ( sub == 0u )
			{
				#pragma unroll
				for ( uint32_t n = 0u; n < 4u; ++n )
				{
					if constexpr ( HAS_SHARED )
						total[n] += LmBf16ToFloat(shared[col0 + n]);
					output[col0 + n] = LmFloatToBf16(total[n]);
				}
			}
		}
		else
		{
			float total[2] = { 0.0f, 0.0f };
			for ( uint32_t k = 0u; k < count; ++k )
			{
				const uint4 *a_row = (const uint4 *)expert_act + (uint64_t)s_pr[k] * K_V;
				const uint4 *w0_row = (const uint4 *)w2 + ((uint64_t)s_exp[k] * N + col0) * K_V;
				const uint4 *w1_row = w0_row + K_V;
				float acc0 = 0.0f, acc1 = 0.0f;
				for ( uint32_t c = sub; c < K_V; c += LM_WARP_LANES )
				{
					const uint4 va = __ldg(a_row + c);
					acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w0_row + c), va, 0.0f), acc0);
					acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w1_row + c), va, 0.0f), acc1);
				}
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
					acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
				}
				if ( sub == 0u )
				{
					total[0] += SparkMathRoundBf16(acc0);
					total[1] += SparkMathRoundBf16(acc1);
				}
			}
			if ( sub == 0u )
			{
				#pragma unroll
				for ( uint32_t n = 0u; n < 2u; ++n )
				{
					if constexpr ( HAS_SHARED )
						total[n] += LmBf16ToFloat(shared[col0 + n]);
					output[col0 + n] = LmFloatToBf16(total[n]);
				}
			}
		}
	}
	LmDependentRelease();
}

static __device__ __forceinline__ void GemmaTilePrefixScan(const uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ prefix, uint32_t groups,
	uint32_t tile_m, uint32_t n_tiles)
{
	const uint32_t lane = threadIdx.x & 31u;
	if ( threadIdx.x < 32u )
	{
		const uint32_t per_lane = (groups + 31u) / 32u, first = lane * per_lane;
		uint32_t own = 0u;
		for ( uint32_t g = first; g < first + per_lane && g < groups; ++g )
		{
			const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
			own += (r + tile_m - 1u) / tile_m * n_tiles;
		}
		uint32_t running = own;
		#pragma unroll
		for ( uint32_t offset = 1u; offset < 32u; offset <<= 1u )
		{
			const uint32_t other = __shfl_up_sync(0xffffffffu, running, offset);
			if ( lane >= offset )
				running += other;
		}
		uint32_t off = running - own;
		for ( uint32_t g = first; g < first + per_lane && g < groups; ++g )
		{
			prefix[g] = off;
			const uint32_t r = group_row_offset[g + 1u] - group_row_offset[g];
			off += (r + tile_m - 1u) / tile_m * n_tiles;
		}
		if ( lane == 31u )
			prefix[groups] = running;
	}
	__syncthreads();
}

template<uint32_t BYTES, bool GROUPED>
struct GemmaMmaSmem
{
	__align__(128) uint8_t buf[BYTES];
	uint32_t prefix[257u];
};

template<uint32_t BYTES>
struct GemmaMmaSmem<BYTES, false>
{
	__align__(128) uint8_t buf[BYTES];
};

static __device__ __forceinline__ uint32_t GemmaGroupedTotalTiles(const uint32_t *__restrict__ group_row_offset, uint32_t groups, uint32_t tile_m,
	uint32_t n_tiles)
{
	uint32_t total = 0u;
	for ( uint32_t g = 0u; g < groups; ++g )
		total += (__ldg(group_row_offset + g + 1u) - __ldg(group_row_offset + g) + tile_m - 1u) / tile_m * n_tiles;
	return total;
}

static __device__ __forceinline__ void GemmaGroupedTileScan(const uint32_t *__restrict__ group_row_offset, uint32_t groups, uint32_t tile,
	uint32_t tile_m, uint32_t n_tiles, uint32_t &grp, uint32_t &row_base, uint32_t &row_limit, uint32_t &nt)
{
	uint32_t off = 0u;
	for ( uint32_t g = 0u; g < groups; ++g )
	{
		const uint32_t first = __ldg(group_row_offset + g), end = __ldg(group_row_offset + g + 1u);
		const uint32_t cnt = (end - first + tile_m - 1u) / tile_m * n_tiles;
		if ( tile < off + cnt )
		{
			grp = g;
			row_limit = end;
			const uint32_t within = tile - off;
			row_base = first + (within / n_tiles) * tile_m;
			nt = within % n_tiles;
			return;
		}
		off += cnt;
	}
	grp = 0u; row_base = 0u; row_limit = 0u; nt = 0u;
}

static __device__ __forceinline__ void GemmaGroupedTileLookup(const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ prefix,
	uint32_t groups, uint32_t tile, uint32_t tile_m, uint32_t n_tiles, uint32_t &grp, uint32_t &row_base, uint32_t &row_limit, uint32_t &nt)
{
	uint32_t lo = 0u, hi = groups;
	while ( lo + 1u < hi )
	{
		const uint32_t mid = (lo + hi) >> 1u;
		if ( prefix[mid] <= tile ) lo = mid;
		else hi = mid;
	}
	grp = lo;
	const uint32_t first = group_row_offset[grp];
	row_limit = group_row_offset[grp + 1u];
	const uint32_t within = tile - prefix[grp];
	row_base = first + (within / n_tiles) * tile_m;
	nt = within % n_tiles;
}

template<uint32_t ACT_MODE>
static __device__ __forceinline__ uint16_t GemmaMmaActValue(float g_acc, float u_acc, float wt, float limit)
{
	const float g = SparkMathRoundBf16(g_acc);
	const float u = SparkMathRoundBf16(u_acc);
	if constexpr ( ACT_MODE == 3u )
		return LmFloatToBf16(SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(g)) * u * wt);
	else
		return LmFloatToBf16(SparkMathGateLimitedGlu(g, u, limit) * wt);
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t THREADS, uint32_t WARPS_M, uint32_t STAGES, bool DUAL, bool GROUPED, bool LDCS_B>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaMmaKernel(const uint16_t *__restrict__ w0, const uint16_t *__restrict__ w1, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ out0, uint16_t *__restrict__ out1, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ source_row_map, uint32_t rows, uint32_t groups, uint32_t K, uint32_t n0, uint32_t n1)
{
	constexpr uint32_t KSTEPS = TILE_K / 16u, ROW_CHUNKS = TILE_K / 8u, ROW_BYTES = TILE_K * 2u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	constexpr bool USE_PREFIX = GROUPED && (STAGES * STAGE_BYTES <= 48000u);
	static_assert(NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS, "valid MMA warp tile");
	__shared__ GemmaMmaSmem<STAGES * STAGE_BYTES, USE_PREFIX> smem_s;
	uint8_t *const smem = smem_s.buf;
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t n0_tiles = (n0 + TILE_N - 1u) / TILE_N;
	const uint32_t n_tiles = DUAL ? (n0_tiles + (n1 + TILE_N - 1u) / TILE_N) : n0_tiles;
	LmDependentWait();
	uint32_t total_tiles;
	if constexpr ( USE_PREFIX )
	{
		GemmaTilePrefixScan(group_row_offset, smem_s.prefix, groups, TILE_M, n_tiles);
		total_tiles = smem_s.prefix[groups];
	}
	else if constexpr ( GROUPED )
		total_tiles = GemmaGroupedTotalTiles(group_row_offset, groups, TILE_M, n_tiles);
	else
		total_tiles = ((rows + TILE_M - 1u) / TILE_M) * n_tiles;
	auto swz = [](uint32_t r, uint32_t c) -> uint32_t { return (c & ~7u) | LmStreamSwizzle8(r, c & 7u); };
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp = 0u, row_base, row_limit, nt;
		if constexpr ( USE_PREFIX )
			GemmaGroupedTileLookup(group_row_offset, smem_s.prefix, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		else if constexpr ( GROUPED )
			GemmaGroupedTileScan(group_row_offset, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		else
		{
			const uint32_t m_tiles = (rows + TILE_M - 1u) / TILE_M;
			row_base = (tile % m_tiles) * TILE_M;
			row_limit = rows;
			nt = tile / m_tiles;
		}
		const bool second = DUAL && nt >= n0_tiles;
		const uint32_t col_base = (second ? nt - n0_tiles : nt) * TILE_N;
		const uint32_t cur_n = second ? n1 : n0;
		const uint16_t *w_ptr = second ? w1 : w0;
		uint16_t *o_ptr = second ? out1 : out0;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u, b_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = row0 + q * ROW_STEP;
			const uint32_t packed = row_base + r;
			const uint32_t valid = packed < row_limit ? 1u : 0u;
			const uint32_t src_r = valid != 0u ? (GROUPED && source_row_map != 0 ? source_row_map[packed] : packed) : 0u;
			a_src[q] = act + (uint64_t)src_r * K + col0 * 8u;
			a_mask |= valid << q;
		}
		const uint16_t *b_base = w_ptr + (((uint64_t)grp * cur_n) + col_base + row0) * K + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
			b_mask |= (col_base + row0 + q * ROW_STEP < cur_n ? 1u : 0u) << q;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				const uint32_t dst = stage + r * ROW_BYTES + swz(r, col0) * 16u;
				const uint32_t valid = (a_mask >> q) & 1u;
				LmStreamCopy(dst, valid != 0u ? a_src[q] : act, valid << 4u);
				a_src[q] += TILE_K;
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				const uint32_t off = A_BYTES + r * ROW_BYTES + swz(r, col0) * 16u;
				const uint32_t valid = (b_mask >> q) & 1u;
				const uint16_t *src = valid != 0u ? b_base + q * b_step : w_ptr;
				if constexpr ( LDCS_B )
					*(uint4 *)(smem + st_off + off) = valid != 0u ? __ldcs((const uint4 *)src) : make_uint4(0u, 0u, 0u, 0u);
				else
					LmStreamCopy(stage + off, src, valid << 4u);
			}
			b_base += TILE_K;
		};
		auto advance = [](uint32_t &off)
		{
			if constexpr ( STAGES == 2u )
				off ^= STAGE_BYTES;
			else
			{
				off += STAGE_BYTES;
				if ( off == STAGES * STAGE_BYTES )
					off = 0u;
			}
		};
		uint32_t a_reg[2][MF][4], b_reg[2][NF][2];
		float total[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				total[i][j][0] = total[i][j][1] = total[i][j][2] = total[i][j][3] = 0.0f;
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + swz(r, c) * 16u);
			}
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + swz(r, c) * 16u);
				b_reg[buf][jj][0] = tmp[0];
				b_reg[buf][jj][1] = tmp[1];
				b_reg[buf][jj + 1u][0] = tmp[2];
				b_reg[buf][jj + 1u][1] = tmp[3];
			}
		};
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < k_tiles )
				issue_k(sp_off);
			LmStreamCommit();
			kp++;
			advance(sp_off);
		}
		if constexpr ( STAGES >= 3u )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			load_frag(0u, 0u, shared_base + sc_off);
			for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
			{
				const uint32_t stage_s = shared_base + sc_off;
				#pragma unroll
				for ( uint32_t ks = 0u; ks < KSTEPS; ++ks )
				{
					const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
					if ( ks == 0u )
					{
						if ( kp < k_tiles )
						{
							issue_k(sp_off);
							kp++;
							advance(sp_off);
						}
						LmStreamCommit();
					}
					if ( ks + 1u < KSTEPS )
						load_frag(nxt, ks + 1u, stage_s);
					else if ( kt + 1u < k_tiles )
					{
						LmStreamWait<STAGES - 2u>();
						__syncthreads();
						advance(sc_off);
						load_frag(nxt, 0u, shared_base + sc_off);
					}
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
						#pragma unroll
						for ( uint32_t j = 0u; j < NF; ++j )
							LmMmaBf16(total[i][j], a_reg[cur][i], b_reg[cur][j]);
				}
			}
		}
		else
		{
			for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
			{
				LmStreamWait<0u>();
				__syncthreads();
				const uint32_t stage_s = shared_base + sc_off;
				if ( kp < k_tiles )
				{
					issue_k(sp_off);
					kp++;
					advance(sp_off);
				}
				LmStreamCommit();
				load_frag(0u, 0u, stage_s);
				#pragma unroll
				for ( uint32_t ks = 0u; ks < KSTEPS; ++ks )
				{
					const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
					if ( ks + 1u < KSTEPS )
						load_frag(nxt, ks + 1u, stage_s);
					#pragma unroll
					for ( uint32_t i = 0u; i < MF; ++i )
						#pragma unroll
						for ( uint32_t j = 0u; j < NF; ++j )
							LmMmaBf16(total[i][j], a_reg[cur][i], b_reg[cur][j]);
				}
				advance(sc_off);
			}
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit && c < cur_n )
						*(uint32_t *)(o_ptr + (uint64_t)r * cur_n + c) = (uint32_t)LmFloatToBf16(total[i][j][e]) |
							((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M, uint32_t ACT_MODE>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaMmaGateUpActKernel(const uint16_t *__restrict__ w_gate, const uint16_t *__restrict__ w_up, const uint16_t *__restrict__ act,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ source_row_map,
	uint16_t *__restrict__ out, uint32_t groups, uint32_t K, uint32_t N, float limit)
{
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u, STAGES = 2u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	constexpr bool USE_PREFIX = (STAGES * STAGE_BYTES <= 48000u);
	static_assert(NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS, "valid fused gate-up MMA tile");
	__shared__ GemmaMmaSmem<STAGES * STAGE_BYTES, USE_PREFIX> smem_s;
	uint8_t *const smem = smem_s.buf;
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K, n_tiles = (N + TILE_N - 1u) / TILE_N;
	LmDependentWait();
	uint32_t total_tiles;
	if constexpr ( USE_PREFIX )
	{
		GemmaTilePrefixScan(group_row_offset, smem_s.prefix, groups, TILE_M, n_tiles);
		total_tiles = smem_s.prefix[groups];
	}
	else
		total_tiles = GemmaGroupedTotalTiles(group_row_offset, groups, TILE_M, n_tiles);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		if constexpr ( USE_PREFIX )
			GemmaGroupedTileLookup(group_row_offset, smem_s.prefix, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		else
			GemmaGroupedTileScan(group_row_offset, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u, b_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t packed = row_base + row0 + q * ROW_STEP;
			const uint32_t valid = packed < row_limit ? 1u : 0u;
			const uint32_t src_r = valid != 0u ? source_row_map[packed] : 0u;
			a_src[q] = act + (uint64_t)src_r * K + col0 * 8u;
			a_mask |= valid << q;
		}
		const uint64_t g_off = (((uint64_t)grp * N) + col_base + row0) * K + col0 * 8u;
		const uint16_t *bg_base = w_gate + g_off, *bu_base = w_up + g_off;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
			b_mask |= (col_base + row0 + q * ROW_STEP < N ? 1u : 0u) << q;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				const uint32_t valid = (a_mask >> q) & 1u;
				LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, valid != 0u ? a_src[q] : act, valid << 4u);
				a_src[q] += TILE_K;
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u;
				const uint32_t valid = (b_mask >> q) & 1u;
				LmStreamCopy(stage + A_BYTES + sw, valid != 0u ? bg_base + q * b_step : w_gate, valid << 4u);
				LmStreamCopy(stage + A_BYTES + B_BYTES + sw, valid != 0u ? bu_base + q * b_step : w_up, valid << 4u);
			}
			bg_base += TILE_K;
			bu_base += TILE_K;
		};
		uint32_t a_reg[2][MF][4], bg_reg[2][NF][2], bu_reg[2][NF][2];
		float tg[MF][NF][4], tu[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				tg[i][j][0] = tg[i][j][1] = tg[i][j][2] = tg[i][j][3] = 0.0f;
				tu[i][j][0] = tu[i][j][1] = tu[i][j][2] = tu[i][j][3] = 0.0f;
			}
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			}
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				const uint32_t off = r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u;
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + off);
				bg_reg[buf][jj][0] = tmp[0]; bg_reg[buf][jj][1] = tmp[1];
				bg_reg[buf][jj + 1u][0] = tmp[2]; bg_reg[buf][jj + 1u][1] = tmp[3];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + B_BYTES + off);
				bu_reg[buf][jj][0] = tmp[0]; bu_reg[buf][jj][1] = tmp[1];
				bu_reg[buf][jj + 1u][0] = tmp[2]; bu_reg[buf][jj + 1u][1] = tmp[3];
			}
		};
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		if ( kp < k_tiles )
			issue_k(sp_off);
		LmStreamCommit();
		kp++;
		sp_off ^= STAGE_BYTES;
		for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
		{
			LmStreamWait<0u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + sc_off;
			if ( kp < k_tiles )
			{
				issue_k(sp_off);
				kp++;
				sp_off ^= STAGE_BYTES;
			}
			LmStreamCommit();
			load_frag(0u, 0u, stage_s);
			#pragma unroll
			for ( uint32_t ks = 0u; ks < KSTEPS; ++ks )
			{
				const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
				if ( ks + 1u < KSTEPS )
					load_frag(nxt, ks + 1u, stage_s);
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
					{
						LmMmaBf16(tg[i][j], a_reg[cur][i], bg_reg[cur][j]);
						LmMmaBf16(tu[i][j], a_reg[cur][i], bu_reg[cur][j]);
					}
			}
			sc_off ^= STAGE_BYTES;
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t c = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
					if ( r < row_limit && c < N )
					{
						const float wt = __ldg(packed_weight + r);
						const uint16_t lo = GemmaMmaActValue<ACT_MODE>(tg[i][j][e], tu[i][j][e], wt, limit);
						const uint16_t hi = GemmaMmaActValue<ACT_MODE>(tg[i][j][e + 1u], tu[i][j][e + 1u], wt, limit);
						*(uint32_t *)(out + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

static __device__ __forceinline__ float GemmaBf16Dot8Exact(const float *wf, const float *af)
{
	float a0 = fmaf(wf[0], af[0], 0.0f);
	float a1 = wf[1] * af[1];
	a0 = fmaf(wf[2], af[2], a0);
	a1 = fmaf(wf[3], af[3], a1);
	a0 = fmaf(wf[4], af[4], a0);
	a1 = fmaf(wf[5], af[5], a1);
	a0 = fmaf(wf[6], af[6], a0);
	a1 = fmaf(wf[7], af[7], a1);
	return a0 + a1;
}

template<bool DUAL>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaExactSplitDenseKernel(const uint16_t *__restrict__ w0, const uint16_t *__restrict__ w1, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ out0, float *__restrict__ out0_f32, uint16_t *__restrict__ out1, uint32_t rows, uint32_t K, uint32_t n0, uint32_t n1)
{
	constexpr uint32_t SPLIT_W = 8u, BN = 4u, RT = 8u;
	__shared__ float partial[SPLIT_W][BN][RT];
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t n0_tiles = (n0 + BN - 1u) / BN;
	const uint32_t nt = blockIdx.x, r0 = blockIdx.y * RT;
	const uint32_t count = rows - r0 < RT ? rows - r0 : RT;
	const bool second = DUAL && nt >= n0_tiles;
	const uint32_t col0 = (second ? nt - n0_tiles : nt) * BN;
	const uint32_t cur_n = second ? n1 : n0;
	const uint16_t *w_ptr = second ? w1 : w0;
	uint16_t *o_ptr = second ? out1 : out0;
	float *o_f32 = second ? (float *)0 : out0_f32;
	const uint32_t chunks = K >> 3u;
	const uint32_t first = chunks * warp / SPLIT_W, last = chunks * (warp + 1u) / SPLIT_W, seg = last - first;
	const uint4 *w_rows[BN];
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		w_rows[b] = (const uint4 *)(w_ptr + (uint64_t)(col0 + b < cur_n ? col0 + b : 0u) * K) + first;
	float acc[BN][RT];
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
			acc[b][r] = 0.0f;
	LmDependentWait();
	for ( uint32_t c = sub; c < seg; c += 32u )
	{
		float wf[BN][8];
		#pragma unroll
		for ( uint32_t b = 0u; b < BN; ++b )
			LmSkinnyUnpackBf16x8(__ldg(w_rows[b] + c), wf[b]);
		const uint32_t col_elem = (first + c) * 8u;
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
		{
			const uint32_t rr = r0 + (r < count ? r : 0u);
			float af[8];
			LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)rr * K + col_elem)), af);
			#pragma unroll
			for ( uint32_t b = 0u; b < BN; ++b )
				acc[b][r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf[b], af), acc[b][r]);
		}
	}
	#pragma unroll
	for ( uint32_t b = 0u; b < BN; ++b )
		#pragma unroll
		for ( uint32_t r = 0u; r < RT; ++r )
		{
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				acc[b][r] += __shfl_xor_sync(0xffffffffu, acc[b][r], step);
			if ( sub == 0u )
				partial[warp][b][r] = acc[b][r];
		}
	__syncthreads();
	if ( warp == 0u )
	{
		const uint32_t b = sub >> 3u, r = sub & 7u;
		if ( col0 + b < cur_n && r < count )
		{
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < SPLIT_W; ++w )
				total += partial[w][b][r];
			const uint64_t idx = (uint64_t)(r0 + r) * cur_n + (col0 + b);
			if ( o_f32 != 0 )
				o_f32[idx] = total;
			else
				o_ptr[idx] = LmFloatToBf16(total);
		}
	}
	LmDependentRelease();
}

template<class G, bool GELU>
__global__ __launch_bounds__(128u, 4)
static void GemmaMoeExactW1W3ActKernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3, const uint16_t *__restrict__ act,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token,
	uint16_t *__restrict__ out, float limit)
{
	constexpr uint32_t WARPS = 4u, RG = 8u;
	constexpr uint32_t K = G::kHidden, N = G::kMoeInter, CHUNKS = K >> 3u;
	constexpr uint32_t N_TILES = (N + WARPS - 1u) / WARPS;
	__shared__ uint32_t prefix[G::kLocalExperts + 1u];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	GemmaTilePrefixScan(group_row_offset, prefix, G::kLocalExperts, RG, N_TILES);
	const uint32_t total_tiles = prefix[G::kLocalExperts];
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, r0, row_limit, nt;
		GemmaGroupedTileLookup(group_row_offset, prefix, G::kLocalExperts, tile, RG, N_TILES, grp, r0, row_limit, nt);
		const uint32_t count = row_limit - r0 < RG ? row_limit - r0 : RG;
		const uint32_t col = nt * WARPS + warp;
		if ( col < N )
		{
			uint32_t src_r[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				src_r[r] = r < count ? __ldg(route_source_token + r0 + r) : 0u;
			const uint4 *w1_row = (const uint4 *)w1 + ((uint64_t)grp * N + col) * CHUNKS;
			const uint4 *w3_row = (const uint4 *)w3 + ((uint64_t)grp * N + col) * CHUNKS;
			float g_acc[RG], u_acc[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				g_acc[r] = 0.0f;
				u_acc[r] = 0.0f;
			}
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				float wg[8], wu[8];
				LmSkinnyUnpackBf16x8(__ldcs(w1_row + c), wg);
				LmSkinnyUnpackBf16x8(__ldcs(w3_row + c), wu);
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < count )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)src_r[r] * K) + c), af);
						g_acc[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wg, af), g_acc[r]);
						u_acc[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wu, af), u_acc[r]);
					}
				}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				if ( r < count )
				{
					#pragma unroll
					for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					{
						g_acc[r] += __shfl_xor_sync(0xffffffffu, g_acc[r], step);
						u_acc[r] += __shfl_xor_sync(0xffffffffu, u_acc[r], step);
					}
					if ( lane == 0u )
					{
						const float wt = __ldg(packed_weight + r0 + r);
						out[(uint64_t)(r0 + r) * N + col] = GemmaMmaActValue<GELU ? 3u : 4u>(g_acc[r], u_acc[r], wt, limit);
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(128u, 4)
static void GemmaMoeExactW2Kernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ act, const uint32_t *__restrict__ group_row_offset,
	uint16_t *__restrict__ out)
{
	constexpr uint32_t WARPS = 4u, RG = 8u, NPG = 2u;
	constexpr uint32_t K = G::kMoeInter, N = G::kHidden, CHUNKS = K >> 3u;
	constexpr uint32_t COLS_PER_BLK = WARPS * NPG;
	constexpr uint32_t N_TILES = (N + COLS_PER_BLK - 1u) / COLS_PER_BLK;
	__shared__ uint32_t prefix[G::kLocalExperts + 1u];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	GemmaTilePrefixScan(group_row_offset, prefix, G::kLocalExperts, RG, N_TILES);
	const uint32_t total_tiles = prefix[G::kLocalExperts];
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, r0, row_limit, nt;
		GemmaGroupedTileLookup(group_row_offset, prefix, G::kLocalExperts, tile, RG, N_TILES, grp, r0, row_limit, nt);
		const uint32_t count = row_limit - r0 < RG ? row_limit - r0 : RG;
		const uint32_t col0 = nt * COLS_PER_BLK + warp * NPG;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const uint4 *w0 = (const uint4 *)w2 + ((uint64_t)grp * N + col0) * CHUNKS;
			const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
			float acc0[RG], acc1[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
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
					if ( r < count )
					{
						float af[8];
						LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act + (uint64_t)(r0 + r) * K) + c), af);
						acc0[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf0, af), acc0[r]);
						acc1[r] = fmaf(1.0f, GemmaBf16Dot8Exact(wf1, af), acc1[r]);
					}
				}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				if ( r < count )
				{
					#pragma unroll
					for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					{
						acc0[r] += __shfl_xor_sync(0xffffffffu, acc0[r], step);
						acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], step);
					}
					if ( lane == 0u )
					{
						out[(uint64_t)(r0 + r) * N + col0] = LmFloatToBf16(acc0[r]);
						if ( has1 )
							out[(uint64_t)(r0 + r) * N + col0 + 1u] = LmFloatToBf16(acc1[r]);
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G>
struct GemmaModel : CudaStage<GemmaModel<G>>
{
	using S = GemmaShape<G>;
	using State = GemmaState<G>;
	using Layer = GemmaLayer;
	using Schedule = GemmaSchedule<G, GemmaModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::SlidingKvGeometry, &State::sliding_views, 0u, S::kSlidingLayers>,
		CudaStageKvViews<typename S::FullKvGeometry, &State::full_views, 1u, S::kFullLayers>>;
	using LaneShared = CudaStageFields<&State::full_inv_freq, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = true;
	static constexpr bool kTapPlan = true;
	static constexpr bool kFusedPostNorm = true;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kNormEpsilon;
	static constexpr uint32_t kPackAlignment = GEMMA_PACK_ALIGNMENT;

	static_assert(G::kHidden <= GM_THREADS * 32u && (G::kHidden & 7u) == 0u, "hidden width fits four vectorized register slots per thread");
	static_assert(!G::kMoe || G::kSigmoidRouter || S::kMaxFused >= G::kHidden, "attn_full holds router_input during FFN");

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
	};

	static const char *Tag(void) { return G::Tag(); }
	static const char *ReportTag(void) { return G::ReportTag(); }
	static const char *WeightdModel(void) { return G::Name(); }

	static bool PrefillUnified(const SparkStageRunnerStep *step)
	{
		return step->commit != 0u && step->last_rows_only != 0u && step->sequences == 1u && step->tree == 0u;
	}

	static int32_t DenseMma(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t K, uint32_t N,
		uint32_t sms, SparkGpuStream stream)
	{
		if ( rows > 128u && (N > 3072u || (N >= 640u && K > 512u)) )
			return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, (float *)0, rows, K, N, 0u, 0u, sms, stream);
		const bool k128 = (K & 127u) == 0u;
		if ( rows <= 16u && N >= 2048u && (N & 63u) == 0u )
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<16u, 64u, 64u, 128u, 1u, 3u, false, false, false>), ((rows + 15u) / 16u) * ((N + 63u) / 64u), 128u, 0, stream,
				weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
		else if ( rows <= 32u )
		{
			if ( k128 && N >= 1024u )
				(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<32u, 32u, 128u, 128u, 2u, 2u, false, false, true>), ((rows + 31u) / 32u) * ((N + 31u) / 32u), 128u, 0, stream,
					weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
			else
				(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<32u, 32u, 64u, 128u, 2u, 3u, false, false, false>), ((rows + 31u) / 32u) * ((N + 31u) / 32u), 128u, 0, stream,
					weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
		}
		else if ( rows <= 64u )
		{
			if ( k128 && N >= 1024u )
				(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 32u, 128u, 128u, 2u, 2u, false, false, true>), ((rows + 63u) / 64u) * ((N + 31u) / 32u), 128u, 0, stream,
					weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
			else
				(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 32u, 64u, 128u, 2u, 3u, false, false, false>), ((rows + 63u) / 64u) * ((N + 31u) / 32u), 128u, 0, stream,
					weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
		}
		else if ( rows <= 128u )
		{
			if ( N >= 1024u )
				(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<128u, 32u, 64u, 256u, 4u, 2u, false, false, true>), ((rows + 127u) / 128u) * ((N + 31u) / 32u), 256u, 0, stream,
					weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
			else
				(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 32u, 64u, 128u, 2u, 3u, false, false, false>), ((rows + 63u) / 64u) * ((N + 31u) / 32u), 128u, 0, stream,
					weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
		}
		else if ( N <= 256u )
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 32u, 64u, 128u, 2u, 3u, false, false, false>), ((rows + 63u) / 64u) * ((N + 31u) / 32u), 128u, 0, stream,
				weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 64u, 64u, 256u, 2u, 3u, false, false, false>), ((rows + 63u) / 64u) * ((N + 63u) / 64u), 256u, 0, stream,
				weight, (const uint16_t *)0, source, destination, (uint16_t *)0, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, N, 0u);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t DualDenseMma(const uint16_t *w0, const uint16_t *w1, const uint16_t *source, uint16_t *out0, uint16_t *out1, uint32_t rows,
		uint32_t K, uint32_t n0, uint32_t n1, SparkGpuStream stream)
	{
		if ( rows <= 128u )
		{
			const uint32_t nt = (n0 + 31u) / 32u + (n1 + 31u) / 32u;
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 32u, 64u, 128u, 2u, 3u, true, false, false>), ((rows + 63u) / 64u) * nt, 128u, 0, stream,
				w0, w1, source, out0, out1, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, n0, n1);
		}
		else if ( rows <= 1024u )
		{
			const uint32_t nt = (n0 + 63u) / 64u + (n1 + 63u) / 64u;
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 64u, 64u, 256u, 2u, 3u, true, false, false>), ((rows + 63u) / 64u) * nt, 256u, 0, stream,
				w0, w1, source, out0, out1, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, n0, n1);
		}
		else
		{
			const uint32_t nt = (n0 + 63u) / 64u + (n1 + 63u) / 64u;
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<128u, 64u, 64u, 256u, 4u, 2u, true, false, false>), ((rows + 127u) / 128u) * nt, 256u, 0, stream,
				w0, w1, source, out0, out1, (const uint32_t *)0, (const uint32_t *)0, rows, 1u, K, n0, n1);
		}
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<bool GELU>
	static int32_t MoeW1W3ActMma(const State *state, const GemmaLayer *layer, const uint16_t *act, uint32_t rows, uint32_t sms, SparkGpuStream stream)
	{
		constexpr uint32_t ACT_MODE = GELU ? 3u : 4u;
		const uint32_t local_mean = (rows * G::kTopK + G::kExperts - 1u) / G::kExperts;
		const uint32_t grid = sms * 2u;
		if ( local_mean <= 16u )
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaGateUpActKernel<16u, 64u, 128u, 1u, ACT_MODE>), grid, 128u, 0, stream,
				layer->moe_w1, layer->moe_w3, act, state->packed_weight, state->group_row_offset, state->route_source_token,
				state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter, layer->swiglu_limit);
		else if ( local_mean <= 32u )
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaGateUpActKernel<32u, 64u, 256u, 2u, ACT_MODE>), grid, 256u, 0, stream,
				layer->moe_w1, layer->moe_w3, act, state->packed_weight, state->group_row_offset, state->route_source_token,
				state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter, layer->swiglu_limit);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaGateUpActKernel<64u, 64u, 256u, 2u, ACT_MODE>), grid, 256u, 0, stream,
				layer->moe_w1, layer->moe_w3, act, state->packed_weight, state->group_row_offset, state->route_source_token,
				state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter, layer->swiglu_limit);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t MoeDownMma(const State *state, const GemmaLayer *layer, uint32_t rows, uint32_t sms, SparkGpuStream stream)
	{
		const uint32_t local_mean = (rows * G::kTopK + G::kExperts - 1u) / G::kExperts;
		if ( local_mean <= 16u )
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<16u, 64u, 64u, 128u, 1u, 3u, false, true, false>), sms * 3u, 128u, 0, stream,
				layer->moe_w2, (const uint16_t *)0, state->expert_act, state->expert_out, (uint16_t *)0, state->group_row_offset,
				(const uint32_t *)0, rows * G::kTopK, G::kLocalExperts, G::kMoeInter, G::kHidden, 0u);
		else if ( local_mean <= 32u )
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<32u, 64u, 64u, 256u, 2u, 3u, false, true, false>), sms * 2u, 256u, 0, stream,
				layer->moe_w2, (const uint16_t *)0, state->expert_act, state->expert_out, (uint16_t *)0, state->group_row_offset,
				(const uint32_t *)0, rows * G::kTopK, G::kLocalExperts, G::kMoeInter, G::kHidden, 0u);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaMmaKernel<64u, 128u, 64u, 256u, 2u, 2u, false, true, false>), sms * 2u, 256u, 0, stream,
				layer->moe_w2, (const uint16_t *)0, state->expert_act, state->expert_out, (uint16_t *)0, state->group_row_offset,
				(const uint32_t *)0, rows * G::kTopK, G::kLocalExperts, G::kMoeInter, G::kHidden, 0u);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t Project(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		if ( prefill_unified )
		{
			if ( destination_f32 == 0 && (input_dimension & 63u) == 0u && (output_dimension & 31u) == 0u )
				return DenseMma(weight, source, destination, rows, input_dimension, output_dimension, step->multiprocessors, stream);
			return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension,
				output_dimension, 0u, 0u, step->multiprocessors, stream);
		}
		if ( exact == 0u && destination_f32 == 0 && rows > LM_SKINNY_ROWS_WIDE &&
			(input_dimension & 63u) == 0u && (output_dimension & 31u) == 0u )
			return DenseMma(weight, source, destination, rows, input_dimension, output_dimension, step->multiprocessors, stream);
		if ( exact != 0u && rows > 8u && (input_dimension & 7u) == 0u &&
			output_dimension <= LM_SKINNY_SPLIT_MAX_OUTPUTS && input_dimension >= 2048u )
		{
			(void)LM_LAUNCH_DEPENDENT((GemmaExactSplitDenseKernel<false>), dim3((output_dimension + 3u) / 4u, (rows + 7u) / 8u), GM_THREADS, 0, stream,
				weight, (const uint16_t *)0, source, destination, destination_f32, (uint16_t *)0, rows, input_dimension, output_dimension, 0u);
			return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
		}
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			exact, step->multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return Project(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, float *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return Project(weight, source, (uint16_t *)0, destination, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const uint16_t *weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		if ( exact == 0u && (routes.rows > 1u || prefill_unified) )
		{
			const bool use_stream = prefill_unified || (down != 0u ? (routes.routes >= 8u) : (routes.rows >= 16u || routes.routes >= 128u));
			if ( use_stream )
			{
				const int32_t launch = LmStreamGemmGrouped<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, routes.group_row_offset,
					down != 0u ? (const uint32_t *)0 : routes.route_source_token, routes.local_experts, routes.routes, input_dimension, output_dimension,
					routes.multiprocessors, stream);
				if ( launch != LM_LAUNCH_ERR_SHAPE )
					return launch;
			}
		}
		return down != 0u ? LmLocalExpertsDown<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, input_dimension, output_dimension, stream, exact) :
			LmLocalExpertsUp<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, input_dimension, output_dimension, stream, exact);
	}

	static void ExpertRoutes(const State *state, const SparkStageRunnerStep *step, uint32_t rows, LmLocalExpertRoutes *routes)
	{
		routes->group_row_offset = state->group_row_offset;
		routes->route_source_token = state->route_source_token;
		routes->tile_prefix_up = state->tile_prefix_up;
		routes->tile_prefix_down = state->tile_prefix_down;
		routes->rows = rows;
		routes->routes = rows * G::kTopK;
		routes->top_k = G::kTopK;
		routes->local_experts = G::kLocalExperts;
		routes->multiprocessors = step->multiprocessors;
	}

	static SparkStatus BuildRoutesPacked(State *state, uint32_t rows, uint32_t routes, SparkGpuStream stream)
	{
		const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
		const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		(void)LM_LAUNCH_DEPENDENT((GemmaRouteBuildPackedKernel<GM_THREADS, G::kLocalExperts + 1u>), 1u, GM_THREADS, 0, stream,
			state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row, state->route_source_token,
			state->packed_weight, state->route_global, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
		return CudaStageLaunched();
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		const GemmaLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
		constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim, KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
		constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ;
		constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
		constexpr uint32_t HEADS = FULL ? G::kFullHeads : G::kSlidingHeads;
		constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
		constexpr uint32_t TASKS = HEADS + 2u * KVH;
		const uint32_t rows = step->rows, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint32_t width = QW / degree, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(HEADS, HD, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		bool gate_done = false;
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if constexpr ( G::kOutputGate || G::kHeadGate )
		{
			const uint32_t gate_out = G::kOutputGate ? width : (FULL ? S::kFullGateRows : S::kSlidingGateRows);
			const bool allow_multi = local_rows > LM_SKINNY_SPLIT_MAX_OUTPUTS || (local_rows + gate_out <= LM_SKINNY_SPLIT_MAX_OUTPUTS) ||
				(exact == 0u && rows > 1u);
			if ( (prefill_unified || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE)) && (G::kHidden & 63u) == 0u && (local_rows & 31u) == 0u && (gate_out & 31u) == 0u )
			{
				if ( DualDenseMma(layer->qkv, layer->gate, state->normed, state->attn_local, state->gate_logits, rows, G::kHidden, local_rows, gate_out, stream) == LM_LAUNCH_OK )
					gate_done = true;
			}
			else if ( !prefill_unified && exact != 0u && rows > 8u && local_rows <= LM_SKINNY_SPLIT_MAX_OUTPUTS && gate_out <= LM_SKINNY_SPLIT_MAX_OUTPUTS &&
				G::kHidden >= 2048u )
			{
				const uint32_t nt = (local_rows + 3u) / 4u + (gate_out + 3u) / 4u;
				(void)LM_LAUNCH_DEPENDENT((GemmaExactSplitDenseKernel<true>), dim3(nt, (rows + 7u) / 8u), GM_THREADS, 0, stream,
					layer->qkv, layer->gate, state->normed, state->attn_local, (float *)0, state->gate_logits, rows, G::kHidden, local_rows, gate_out);
				if ( CudaStageLaunched() == SPARK_STATUS_OK )
					gate_done = true;
			}
			else if ( !prefill_unified && allow_multi && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			{
				const uint32_t allow_split = local_rows + gate_out <= LM_SKINNY_SPLIT_MAX_OUTPUTS ? 1u : 0u;
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->qkv, state->attn_local, (float *)0, local_rows, local_rows, 0u, LmScaleTensorNone() },
					{ layer->gate, state->gate_logits, (float *)0, gate_out, gate_out, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->normed, rows, G::kHidden, stream, exact, allow_split) == LM_LAUNCH_OK )
					gate_done = true;
			}
		}
		if ( !gate_done && Linear(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		const uint32_t rank_heads = HEADS / degree;
		const uint32_t group = HEADS / KVH;
		const bool uniform_kvh = (degree == 1u) || ((HEADS % degree) == 0u && ((group % rank_heads) == 0u || (rank_heads % group) == 0u));
		const uint32_t q_first = state->tp_rank * rank_heads;
		const uint32_t kv_first = q_first / group;
		const uint32_t kv_last = (q_first + rank_heads - 1u) / group;
		const uint32_t local_kvh = kv_last - kv_first + 1u;
		const uint32_t total_ctx = step->gather_context;
		const uint32_t old_ctx = total_ctx > rows ? total_ctx - rows : 0u;
		const uint32_t window = FULL ? 0u : layer->window;
		const uint32_t raw_win_begin = (window != 0u && old_ctx + 1u > window) ? (old_ctx + 1u - window) : 0u;
		constexpr uint32_t KV_TILE_ALIGN = GemmaLocalPrefillKvTile<HD>();
		const uint32_t win_begin = raw_win_begin & ~(KV_TILE_ALIGN - 1u);
		const uint32_t prior_keys = old_ctx > win_begin ? old_ctx - win_begin : 0u;
		const uint32_t grain = view->shard.grain;
		const uint32_t b0 = win_begin / grain;
		const uint32_t b1 = (old_ctx + grain - 1u) / grain;
		const uint32_t max_peer_keys = prior_keys != 0u ? (((b1 - b0 + degree - 1u) / degree) * grain) : 0u;
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t peer_bytes = (uint64_t)max_peer_keys * local_kvh * (2u * HD) * sizeof(uint16_t);
		const uint32_t a2a_units = (degree > 1u && prior_keys != 0u) ? (uint32_t)((peer_bytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t a2a_stride_bytes = (uint64_t)a2a_units * unit_bytes;
		const uint64_t prior_bytes = (uint64_t)local_kvh * prior_keys * (2u * HD) * sizeof(uint16_t);
		const uint64_t buf_cap_bytes = (uint64_t)degree * state->record_stride * sizeof(float);
		const bool use_local_prefill = prefill_unified && uniform_kvh && total_ctx >= rows && (HD % 64u) == 0u &&
			(uint64_t)degree * a2a_stride_bytes <= buf_cap_bytes && prior_bytes <= buf_cap_bytes;
		if ( use_local_prefill )
		{
			const uint32_t local_tasks = rank_heads + 2u * KVH;
			uint16_t *kv_cur = state->slot;
			uint16_t *kv_prior = (uint16_t *)state->send;
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			(void)LM_LAUNCH_DEPENDENT((GemmaAttnPrepStoreLocalKernel<G, FULL>), dim3((local_tasks + WARPS - 1u) / WARPS, rows), GM_THREADS, 0, stream,
				state->attn_gathered, G::kQkNormWeighted ? layer->q_norm : (const float *)0, G::kQkNormWeighted ? layer->k_norm : (const float *)0,
				step->positions, step->positions, step->sequence_of_row, FULL ? state->full_inv_freq : state->full_inv_freq + (G::kFullDim / 2u),
				*view, state->query, kv_cur, rows, q_first, rank_heads, kv_first, local_kvh);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( prior_keys != 0u )
			{
				if ( degree > 1u )
				{
					(void)LM_LAUNCH_DEPENDENT((GemmaPriorKvPackKernel<Kv, KVH, HD>), dim3(max_peer_keys, degree), GM_THREADS, 0, stream,
						*view, step->sequence_of_row, win_begin, old_ctx, b0, max_peer_keys, rank_heads, group, local_kvh, a2a_stride_bytes,
						(uint8_t *)state->send);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, a2a_units, 0u, state->send, state->received);
					if ( status != SPARK_STATUS_OK )
						return status;
					(void)LM_LAUNCH_DEPENDENT((GemmaPriorKvUnpackKernel<HD>), dim3(max_peer_keys, degree), GM_THREADS, 0, stream,
						view->shard, (const uint8_t *)state->received, a2a_stride_bytes, win_begin, old_ctx, b0, max_peer_keys, prior_keys,
						local_kvh, kv_prior);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((GemmaPriorKvDirectKernel<Kv, KVH, HD>), dim3(prior_keys, local_kvh), GM_THREADS, 0, stream,
						*view, step->sequence_of_row, win_begin, prior_keys, kv_first, local_kvh, kv_prior);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			constexpr uint32_t V_WARPS = HD <= 128u ? 1u : (HD / 128u);
			constexpr uint32_t Q_WARPS = V_WARPS == 1u ? 4u : 2u;
			constexpr uint32_t ATTN_THREADS = Q_WARPS * V_WARPS * 32u;
			constexpr uint32_t Q_TILE = Q_WARPS * 16u;
			constexpr uint32_t shared_bytes = GemmaLocalPrefillSharedBytes<HD>();
			if constexpr ( shared_bytes > 48000u )
			{
				static std::atomic<uint64_t> lp_granted(0u);
				int dev = 0;
				if ( SparkGpuGetDevice(&dev) != SPARK_GPU_SUCCESS || dev < 0 || (uint32_t)dev >= LM_STREAM_MAX_DEVICES )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( (lp_granted.load() & (UINT64_C(1) << (uint32_t)dev)) == 0u )
				{
					if ( SparkGpuFuncSetAttribute((const void *)GemmaLocalPrefillAttnKernel<HD>,
							SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared_bytes) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					lp_granted.fetch_or(UINT64_C(1) << (uint32_t)dev);
				}
			}
			(void)LM_LAUNCH_DEPENDENT((GemmaLocalPrefillAttnKernel<HD>), dim3(rank_heads, (rows + Q_TILE - 1u) / Q_TILE), ATTN_THREADS, shared_bytes, stream,
				state->query, kv_prior, kv_cur, step->positions, rows, rank_heads, q_first, group, kv_first, win_begin, prior_keys, window,
				G::kAttentionScale, state->merged);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			(void)LM_LAUNCH_DEPENDENT((GemmaAttnPrepStoreKernel<G, FULL>), dim3((TASKS + WARPS - 1u) / WARPS, rows), GM_THREADS, 0, stream,
				state->attn_gathered, G::kQkNormWeighted ? layer->q_norm : (const float *)0, G::kQkNormWeighted ? layer->k_norm : (const float *)0,
				step->positions, step->tree != 0u ? step->slots : step->positions, step->sequence_of_row,
				FULL ? state->full_inv_freq : state->full_inv_freq + (G::kFullDim / 2u), *view, state->query, rows);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,HD,HD>(*view, state->query, HEADS, step->sequence_of_row, step->context_length,
					step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u, FULL ? 0u : layer->window,
					step->tree != 0u ? step->fork_positions : (const uint32_t *)0, step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmGqaShardMergeLaunch<HD>(state->received, stride, degree, HEADS, state->tp_rank, state->merged, rows, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( G::kOutputGate )
		{
			if ( !gate_done && Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, width, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((GemmaOutputGateKernel<GM_THREADS>), rows, GM_THREADS, 0, stream, state->merged, state->gate_logits, width);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( G::kHeadGate )
		{
			constexpr uint32_t GATE_ROWS = FULL ? S::kFullGateRows : S::kSlidingGateRows;
			if ( !gate_done )
			{
				if ( exact == 0u && GATE_ROWS <= 16u )
					(void)LM_LAUNCH_DEPENDENT((GemmaSmallNBf16ProjKernel<G::kHidden>), rows, GM_THREADS, 0, stream,
						layer->gate, state->normed, state->gate_logits, rows, GATE_ROWS);
				else if ( Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, GATE_ROWS, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			(void)LM_LAUNCH_DEPENDENT((GemmaHeadGateBroadcastKernel<128u>), dim3(rows, width / HD), 128u, 0, stream,
				state->merged, state->gate_logits, width / HD, HD, GATE_ROWS);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Linear(layer->out, state->merged, state->partial, rows, width, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mlp(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		const GemmaLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, ffn_rows = layer->ffn_rows, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		bool fused_router = false;
		if constexpr ( G::kMoe && G::kSigmoidRouter )
		{
			const uint32_t gu_out = 2u * ffn_rows;
			const bool allow_multi = gu_out > LM_SKINNY_SPLIT_MAX_OUTPUTS || (gu_out + G::kExperts <= LM_SKINNY_SPLIT_MAX_OUTPUTS) ||
				(exact == 0u && rows > 1u);
			if ( !prefill_unified && allow_multi && layer->router_proj != 0 && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			{
				const uint32_t allow_split = gu_out + G::kExperts <= LM_SKINNY_SPLIT_MAX_OUTPUTS ? 1u : 0u;
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->gate_up, state->gate_up, (float *)0, gu_out, gu_out, 0u, LmScaleTensorNone() },
					{ layer->router_proj, G::kRouterFloat ? (uint16_t *)0 : state->router_logits, G::kRouterFloat ? state->router_probs : (float *)0,
						G::kExperts, G::kExperts, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->normed, rows, G::kHidden, stream, exact, allow_split) == LM_LAUNCH_OK )
					fused_router = true;
			}
		}
		state->router_fused_done = fused_router ? 1u : 0u;
		if ( !fused_router && Linear(layer->gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * ffn_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kGeluMlp )
			(void)LM_LAUNCH_DEPENDENT((GemmaGeluTanhMulKernel<GM_THREADS>), dim3(LmVecBlocks<GM_THREADS>(ffn_rows), rows), GM_THREADS, 0, stream,
				state->gate_up, state->intermediate, ffn_rows);
		else if ( layer->swiglu_limit != INFINITY )
			(void)LM_LAUNCH_DEPENDENT((GemmaSiluMulKernel<GM_THREADS, true>), dim3(LmVecBlocks<GM_THREADS>(ffn_rows), rows), GM_THREADS, 0, stream,
				state->gate_up, state->intermediate, ffn_rows, layer->swiglu_limit);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaSiluMulKernel<GM_THREADS, false>), dim3(LmVecBlocks<GM_THREADS>(ffn_rows), rows), GM_THREADS, 0, stream,
				state->gate_up, state->intermediate, ffn_rows, INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->down, state->intermediate, state->partial, rows, ffn_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	template<bool GELU, bool HAS_SHARED>
	static SparkStatus RunExpertsBody(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream,
		const uint16_t *act_in, uint16_t *output)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		const bool prefill_unified = PrefillUnified(step);
		if ( !prefill_unified && rows == 1u )
		{
			constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
			const uint32_t w1_blocks = (G::kMoeInter + WARPS * 2u - 1u) / (WARPS * 2u);
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeB1W1W3ActKernel<G, GELU>), dim3(w1_blocks, routes), GM_THREADS, 0, stream,
				layer->moe_w1, layer->moe_w3, act_in, state->packed_weight, state->route_global, state->group_row_offset,
				state->expert_act, layer->swiglu_limit);
			constexpr uint32_t W2_SUB = G::kMoeInter == 128u ? 16u : 32u;
			constexpr uint32_t W2_NPG = G::kMoeInter == 128u ? 4u : 2u;
			constexpr uint32_t W2_COLS = (GM_THREADS / W2_SUB) * W2_NPG;
			const uint32_t w2_blocks = (G::kHidden + W2_COLS - 1u) / W2_COLS;
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeB1W2FinalizeKernel<G, HAS_SHARED>), w2_blocks, GM_THREADS, 0, stream,
				layer->moe_w2, state->expert_act, state->route_packed_row, state->route_expert, HAS_SHARED ? state->partial : (const uint16_t *)0, output);
			return CudaStageLaunched();
		}
		LmLocalExpertRoutes expert_routes;
		ExpertRoutes(state, step, rows, &expert_routes);
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool small_exact = !prefill_unified && rows <= 8u;
		if ( exact == 0u && !small_exact && (prefill_unified || rows >= 16u || routes >= 128u) && (G::kHidden & 63u) == 0u && (G::kMoeInter & 63u) == 0u && G::kLocalExperts <= 256u )
		{
			if ( MoeW1W3ActMma<GELU>(state, layer, act_in, rows, step->multiprocessors, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( G::kHidden >= 256u && G::kLocalExperts <= 256u )
		{
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeExactW1W3ActKernel<G, GELU>), step->multiprocessors * 4u, 128u, 0, stream,
				layer->moe_w1, layer->moe_w3, act_in, state->packed_weight, state->group_row_offset, state->route_source_token,
				state->expert_act, layer->swiglu_limit);
		}
		else
		{
			if ( ExpertProjection(expert_routes, layer->moe_w1, act_in, state->expert_gate, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK ||
				ExpertProjection(expert_routes, layer->moe_w3, act_in, state->expert_up, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeActPackedKernel<GELU>), dim3(LmVecBlocks<GM_THREADS>(G::kMoeInter), routes), GM_THREADS, 0, stream,
				state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter,
				layer->swiglu_limit);
		}
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( exact == 0u && !small_exact && (prefill_unified || routes >= 8u) && (G::kMoeInter & 63u) == 0u && (G::kHidden & 63u) == 0u && G::kLocalExperts <= 256u )
		{
			if ( MoeDownMma(state, layer, rows, step->multiprocessors, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( (exact != 0u || small_exact) && G::kMoeInter >= 128u && G::kLocalExperts <= 256u )
		{
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeExactW2Kernel<G>), step->multiprocessors * 4u, 128u, 0, stream,
				layer->moe_w2, state->expert_act, state->group_row_offset, state->expert_out);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((GemmaMoeLocalFinalizeKernel<HAS_SHARED>), dim3(LmVecBlocks<GM_THREADS>(G::kHidden), rows), GM_THREADS, 0, stream,
			state->expert_out, state->route_packed_row, state->route_expert, HAS_SHARED ? state->partial : (const uint16_t *)0, output,
			G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus ExpertRows(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		uint16_t *output = state->partial + (uint64_t)rows * G::kHidden;
		if ( Linear(layer->router_proj, state->attn_full, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( routes <= 32u )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((GemmaSoftmaxRouteSmallKernel<G>), 1u, GM_THREADS, 0, stream, state->router_logits, layer->expert_scale, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((GemmaSoftmaxTopkLocalKernel<G>), (rows + WARPS - 1u) / WARPS, GM_THREADS, 0, stream, state->router_logits,
				layer->expert_scale, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			if ( CudaStageLaunched() != SPARK_STATUS_OK || BuildRoutesPacked(state, rows, routes, stream) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return RunExpertsBody<true, false>(state, layer, step, stream, state->moe_normed, output);
	}

	static SparkStatus SigmoidExpertRows(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		const uint16_t *normed = state->normed;
		uint16_t *output = state->partial + (uint64_t)rows * G::kHidden;
		const bool already_projected = state->router_fused_done != 0u;
		state->router_fused_done = 0u;
		if ( !already_projected )
		{
			if constexpr ( G::kRouterFloat )
			{
				if ( Linear(layer->router_proj, normed, state->router_probs, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				if ( Linear(layer->router_proj, normed, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		const void *logits_ptr = G::kRouterFloat ? (const void *)state->router_probs : (const void *)state->router_logits;
		if ( routes <= 32u )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((GemmaSigmoidRouteSmallKernel<G>), 1u, GM_THREADS, 0, stream, logits_ptr, layer->expert_bias, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((GemmaSigmoidTopkLocalKernel<G>), (rows + WARPS - 1u) / WARPS, GM_THREADS, 0, stream, logits_ptr,
				layer->expert_bias, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			if ( CudaStageLaunched() != SPARK_STATUS_OK || BuildRoutesPacked(state, rows, routes, stream) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return RunExpertsBody<false, true>(state, layer, step, stream, normed, output);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const GemmaLayer &entry) { return ExpertRows(state, &entry, frame.step, frame.stream); }
	static SparkStatus SigmoidExperts(State *state, const Frame &frame, const GemmaLayer &entry) { return SigmoidExpertRows(state, &entry, frame.step, frame.stream); }
	static uint16_t *BranchRows(State *state, const Frame &frame, GemmaBranch branch) { return state->partial + (uint64_t)branch * frame.step->rows * G::kHidden; }

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->tree != 0u && (step->commit != 0u || step->sequences != 1u || step->slots == 0 || step->fork_positions == 0 || step->slot_offsets == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		state->router_fused_done = 0u;
		return CudaStage<GemmaModel>::Begin(state, step, stream, frame);
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
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::SlidingKvGeometry,GM_THREADS>), step->rows, GM_THREADS, 0, stream,
				state->sliding_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, 2u * S::kSlidingKv);
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		}
		for ( uint32_t index = 0u; index < S::kFullLayers; ++index )
		{
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::FullKvGeometry,GM_THREADS>), step->rows, GM_THREADS, 0, stream,
				state->full_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, 2u * S::kFullKv);
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		}
		return CudaStageLaunched();
	}

	static SparkStatus EmbedNorm(State *state, const Frame &frame)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		(void)state;
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(1u,frame.step->rows), GM_THREADS, 0, frame.stream,
			hidden, (const float *)0, hidden, frame.step->rows, 1u, G::kHidden, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus EmbedScale(State *state, const Frame &frame, float scale)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		(void)state;
		LM_LAUNCH((LmScaleElementsKernel<uint16_t>), LmVecBlocks<GM_THREADS>(width), GM_THREADS, 0, frame.stream, frame.step->hidden_bf16, width, scale);
		return SPARK_STATUS_OK;
	}

	static constexpr uint32_t kNormSlots = (G::kHidden + GM_THREADS * 8u - 1u) / (GM_THREADS * 8u);

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((GemmaAddRmsNormKernel<GM_THREADS, G::kNormRoundFirst, false, kNormSlots>), frame.step->rows, GM_THREADS, 0, frame.stream,
			frame.step->hidden_bf16, (uint16_t *)0, weight, state->normed, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, GemmaBranch branch, const float *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((GemmaAddRmsNormKernel<GM_THREADS, G::kNormRoundFirst, true, kNormSlots>), frame.step->rows, GM_THREADS, 0, frame.stream,
			BranchRows(state, frame, branch), frame.step->hidden_bf16, weight, state->normed, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus PostAttnNorm(State *state, const Frame &frame, const GemmaLayer &layer)
	{
		if constexpr ( G::kMoe && !G::kSigmoidRouter )
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, false, true, kNormSlots>), frame.step->rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_attn_norm, (const float *)0, layer.pre_ff_norm, state->normed,
				layer.router_scale, state->attn_full, layer.pre_ff_norm_2, state->moe_normed, G::kHidden, G::kPostEpsilon, G::kNormEpsilon,
				(float)(1.0 / sqrt((double)G::kHidden)));
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, false, false, kNormSlots>), frame.step->rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_attn_norm, (const float *)0, layer.pre_ff_norm, state->normed,
				(const float *)0, (uint16_t *)0, (const float *)0, (uint16_t *)0, G::kHidden, G::kPostEpsilon, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus PostFfnNorm(State *state, const Frame &frame, const GemmaLayer &layer, const float *next_norm)
	{
		const uint32_t rows = frame.step->rows;
		if constexpr ( G::kMoe )
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeCombineNextKernel<GM_THREADS, G::kNormRoundFirst>), rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, state->partial + (uint64_t)rows * G::kHidden, layer.post_ff_norm_1, layer.post_ff_norm_2,
				layer.post_ff_norm, layer.layer_scalar, next_norm, next_norm != 0 ? state->normed : (uint16_t *)0, G::kHidden, G::kPostEpsilon,
				G::kNormEpsilon);
		else if constexpr ( G::kLayerScalar )
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, true, false, kNormSlots>), rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_ff_norm, layer.layer_scalar, next_norm, next_norm != 0 ? state->normed : (uint16_t *)0,
				(const float *)0, (uint16_t *)0, (const float *)0, (uint16_t *)0, G::kHidden, G::kPostEpsilon, G::kNormEpsilon, 1.0f);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, false, false, kNormSlots>), rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_ff_norm, (const float *)0, next_norm, next_norm != 0 ? state->normed : (uint16_t *)0,
				(const float *)0, (uint16_t *)0, (const float *)0, (uint16_t *)0, G::kHidden, G::kPostEpsilon, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRows(State *state, const Frame &frame, GemmaBranch branch)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		LM_LAUNCH((LmAddElementsKernel), LmVecBlocks<GM_THREADS>(width), GM_THREADS, 0, frame.stream, frame.step->hidden_bf16, BranchRows(state, frame, branch), width);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame, GemmaBranch branch, uint32_t blocks)
	{
		uint16_t *rows = BranchRows(state, frame, branch);
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, blocks * frame.step->rows, 0u, rows, rows);
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		SparkStatus status = SPARK_STATUS_OK;
		if ( state->tap_slab != 0 && state->tap_output[index] >= 0 )
		{
			LM_LAUNCH((LmCopyRowsStridedKernel<GM_THREADS>), dim3(LmGridBlocks<GM_THREADS>(G::kHidden), frame.step->rows), GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->tap_slab + (uint64_t)frame.step->dispatch_row_offset * state->tap_stride +
				(uint64_t)state->tap_output[index] * G::kHidden, frame.step->rows, G::kHidden, state->tap_stride);
			status = CudaStageLaunched();
		}
		if ( status == SPARK_STATUS_OK )
			CudaStageLayerDone(state, index);
		return status;
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kMoe ? G::kLocalExperts + 2u : 0u;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, G::kMoe ? 2u * width : width },
			{ (void **)&state->attn_local, rows * (S::kMaxFused / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kMaxQ * sizeof(uint16_t) },
			{ (void **)&state->key, rows * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->slot, rows * 2u * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->gate_logits, G::kOutputGate ? rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) : G::kHeadGate ? rows * S::kMaxGateRows * sizeof(uint16_t) : 0u },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->full_inv_freq, lane == 0u ? (G::kFullDim / 2u + G::kSlidingDim / 2u) * sizeof(float) : 0u },
			{ (void **)&state->moe_normed, G::kMoe ? width : 0u },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(uint16_t) },
			{ (void **)&state->router_probs, rows * G::kExperts * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->packed_weight, routes * sizeof(float) },
			{ (void **)&state->expert_gate, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		(void)configuration;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			state->tap_output[index] = -1;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t full = LmGqaShardRecordFloats(G::kFullHeads, G::kFullDim, state->tp_degree);
		const uint64_t sliding = LmGqaShardRecordFloats(G::kSlidingHeads, G::kSlidingDim, state->tp_degree);
		const uint64_t row_bytes = G::kHidden * sizeof(uint16_t);
		state->record_stride = (((uint64_t)state->max_rows * (full > sliding ? full : sliding) * sizeof(float) + row_bytes - 1u) / row_bytes) * row_bytes /
			sizeof(float);
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration) { return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 ? 1u : 0u; }

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "vocab_rows", &vocab_rows },
			{ "ffn_rows", &state->ffn_rows }, { "sliding_window", &window }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || window != G::kWindow || state->ffn_rows == 0u || state->ffn_rows % 64u != 0u )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u window %u, the runner is tp %u rank %u\n", G::Tag(), degree, rank,
				hidden, layers, window, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( G::kMoe )
		{
			uint32_t experts = 0u, local = 0u, top_k = 0u;
			const CudaStagePackValue moe[] =
			{
				{ "experts", &experts }, { "local_experts", &local }, { "expert_first", &state->expert_first }, { "top_k", &top_k }
			};
			if ( CudaStagePackValues(&state->pack, moe) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK ||
				state->expert_first != (G::kLocalExperts == G::kExperts ? 0u : rank * G::kLocalExperts) )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = 1u;
		geometry->head_norm_round_first = G::kNormRoundFirst ? 1u : 0u;
		geometry->logit_softcap = G::kLogitSoftcap;
		geometry->logit_scale = G::kLogitScale;
		if constexpr ( G::kEmbedAliasCount != 0u )
		{
			geometry->embed_alias_first = G::kEmbedAliasFirst;
			geometry->embed_alias_count = G::kEmbedAliasCount;
			geometry->embed_alias_row = G::kEmbedAliasRow;
		}
		if ( CudaStage<GemmaModel>::HeadRanges(state, geometry, !G::kUntiedHead, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_bytes != (uint64_t)geometry->embed_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)geometry->head_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the stage geometry\n", G::Tag(),
				geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		GemmaLayer *layer = &state->layers[index];
		const bool full = S::Full(index);
		const uint64_t hidden = G::kHidden, norm = hidden * sizeof(float);
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		uint64_t offset, bytes;
		uint32_t gate_rows = 0u;
		snprintf(key, sizeof(key), "layers.%u.ffn_gate_up", index);
		layer->gate_up_rows = 0u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, "bf16", &offset, &bytes, &layer->gate_up_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		tensors->Add("attn_qkv", "bf16", (uint64_t)((full ? S::kFullFused : S::kSlidingFused) / state->tp_degree) * hidden * sizeof(uint16_t), &layer->qkv);
		tensors->Add("attn_out", "bf16", hidden * ((full ? S::kFullQ : S::kSlidingQ) / state->tp_degree) * sizeof(uint16_t), &layer->out);
		tensors->Add("ffn_gate_up", "bf16", (uint64_t)layer->gate_up_rows * hidden * sizeof(uint16_t), &layer->gate_up);
		tensors->Add("ffn_down", "bf16", hidden * (layer->gate_up_rows / 2u) * sizeof(uint16_t), &layer->down);
		tensors->Add("input_norm", "f32", norm, &layer->input_norm);
		if ( !G::kPreNorm )
			tensors->Add("post_attn_norm", "f32", norm, &layer->post_attn_norm);
		tensors->Add("pre_ff_norm", "f32", norm, &layer->pre_ff_norm);
		if ( !G::kPreNorm )
			tensors->Add("post_ff_norm", "f32", norm, &layer->post_ff_norm);
		if ( G::kLayerScalar )
			tensors->Add("layer_scalar", "f32", (uint64_t)sizeof(float), &layer->layer_scalar);
		if ( G::kQkNormWeighted )
		{
			tensors->Add("attn_q_norm", "f32", (uint64_t)(full ? G::kFullDim : G::kSlidingDim) * sizeof(float), &layer->q_norm);
			tensors->Add("attn_k_norm", "f32", (uint64_t)(full ? G::kFullDim : G::kSlidingDim) * sizeof(float), &layer->k_norm);
		}
		if ( G::kOutputGate )
			tensors->Add("attn_gate", "bf16", ((full ? S::kFullQ : S::kSlidingQ) / state->tp_degree) * hidden * sizeof(uint16_t), &layer->gate);
		if ( G::kHeadGate )
		{
			snprintf(key, sizeof(key), "layers.%u.attn_gate", index);
			if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, "bf16", &offset, &bytes, &gate_rows) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( gate_rows != (full ? S::kFullGateRows : S::kSlidingGateRows) )
			{
				fprintf(stderr, "%s: layer %u attn_gate has %u rows, the rank's padded head count is %u\n", G::Tag(), index, gate_rows,
					full ? S::kFullGateRows : S::kSlidingGateRows);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
			tensors->Add("attn_gate", "bf16", (uint64_t)gate_rows * hidden * sizeof(uint16_t), &layer->gate);
		}
		if ( (G::kSigmoidRouter && index >= G::kFirstMoeLayer) || (G::kMoe && !G::kSigmoidRouter) )
		{
			const uint64_t experts = (uint64_t)G::kLocalExperts * G::kMoeInter * hidden * sizeof(uint16_t);
			tensors->Add("router_proj", "bf16", (uint64_t)G::kExperts * hidden * sizeof(uint16_t), &layer->router_proj);
			if ( G::kSigmoidRouter )
				tensors->Add("expert_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->expert_bias);
			else
			{
				tensors->Add("router_scale", "f32", norm, &layer->router_scale);
				tensors->Add("expert_scale", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->expert_scale);
			}
			tensors->Add("moe_w1", "bf16", experts, &layer->moe_w1);
			tensors->Add("moe_w3", "bf16", experts, &layer->moe_w3);
			tensors->Add("moe_w2", "bf16", experts, &layer->moe_w2);
			if ( !G::kSigmoidRouter )
			{
				tensors->Add("post_ff_norm_1", "f32", norm, &layer->post_ff_norm_1);
				tensors->Add("pre_ff_norm_2", "f32", norm, &layer->pre_ff_norm_2);
				tensors->Add("post_ff_norm_2", "f32", norm, &layer->post_ff_norm_2);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerBound(State *state, uint32_t index)
	{
		GemmaLayer *layer = &state->layers[index];
		layer->ffn_rows = layer->gate_up_rows / 2u;
		if ( layer->gate_up_rows % 128u != 0u || layer->ffn_rows == 0u || layer->ffn_rows > state->ffn_rows )
		{
			fprintf(stderr, "%s: layer %u ffn_gate_up has %u rows; a rank holds gate|up in 64-row blocks of at most %u rows each\n", G::Tag(),
				index, layer->gate_up_rows, state->ffn_rows);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t sliding = 0u, full = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			GemmaLayer *layer = &state->layers[index];
			layer->kv_index = S::Full(index) ? full++ : sliding++;
			layer->window = S::Window(index);
			layer->swiglu_limit = S::SwigluLimit(index);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[G::kFullDim / 2u];
		for ( uint32_t pair = 0u; pair < G::kFullDim / 2u; ++pair )
		{
			if constexpr ( G::kFullRopePairs != 0u )
				inv_freq[pair] = pair < G::kFullRopePairs ? (float)(1.0 / pow((double)G::kFullTheta, 2.0 * pair / (double)G::kFullDim)) : 0.0f;
			else
				inv_freq[pair] = 0.0f;
		}
		if constexpr ( G::kYarnFactor > 1.0f )
		{
			const SparkRopePlanDomain yarn = { G::kFullTheta, G::kYarnFactor, G::kYarnOriginal, G::kYarnBetaFast, G::kYarnBetaSlow, G::kFullRopeScale,
				S::kFullRopeWidth };
			memset(inv_freq, 0, sizeof(inv_freq));
			SparkRopePlanBuildYarnInvFrequency(&yarn, inv_freq);
			if ( SparkRopePlanYarnTableIsValid(&yarn, inv_freq) == 0u )
			{
				fprintf(stderr, "%s: the YaRN inverse-frequency table is invalid\n", G::Tag());
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			}
		}
		LM_LAUNCH((GemmaInitSlidingInvFreqKernel), 1u, G::kSlidingDim / 2u, 0, (SparkGpuStream)0,
			state->full_inv_freq + (G::kFullDim / 2u), G::kSlidingDim / 2u, G::kSlidingDim, G::kSlidingTheta);
		if ( SparkGpuMemcpy(state->full_inv_freq, inv_freq, sizeof(inv_freq), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus TapPlan(void *model, const SparkStageRunnerTapPlan *plan)
	{
		State *state = (State *)model;
		int16_t outputs[G::kLayers];
		if ( state == 0 || plan == 0 || plan->entry_count > SPARK_STAGE_RUNNER_TAP_MAX_ENTRIES )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < G::kLayers; ++layer )
			outputs[layer] = -1;
		for ( uint32_t index = 0u; index < plan->entry_count; ++index )
		{
			if ( plan->entries[index].form == SPARK_STAGE_RUNNER_TAP_FORM_FINAL )
				continue;
			if ( plan->entries[index].form != SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT || plan->entries[index].layer >= G::kLayers )
				SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
			outputs[plan->entries[index].layer] = (int16_t)index;
		}
		memcpy(state->tap_output, outputs, sizeof(outputs));
		state->tap_slab = plan->capture != 0u ? plan->slab : 0;
		state->tap_stride = plan->row_stride_elements;
		fprintf(stderr, "%s: taps entries=%u capture=%u; a layer output is its residual after the feed-forward add\n", G::Tag(), plan->entry_count, plan->capture);
		return SPARK_STATUS_OK;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "ffn_rows", state->ffn_rows },
			{ "expert_first", state->expert_first }
		};
		return CudaStageReportOut(fields, report);
	}
};
