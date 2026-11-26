#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/project_chain.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/weight_prefetch.cuh"
#include "runtime/gemm.cuh"

#include "sparkpipe/spark_ling_family.h"
#include "sparkpipe/spark_gpu.h"

#define LG_THREADS 256u
#define LG_FP8_BLOCK 128u

template<class G>
struct LingShape : LingLayout<G>
{
	static constexpr uint32_t kRecord = LM_LATENT_SHARD_RECORD_FLOATS(G::kLatent);
	using Kv = LmKvGeometry<LingLayout<G>::kSlot * 2u, G::kPageSlots, true>;
	static_assert(LingLayout<G>::kMlaRankHeads * LingLayout<G>::kSlot <= G::kHidden, "a rank's packed queries fit one collective row");
	static_assert(!LingLayout<G>::kGqa || (G::kMlaHeads == 0u && G::kGqaHeads % G::kTpDegree == 0u && LingLayout<G>::kGqaFused % G::kTpDegree == 0u &&
		LingLayout<G>::kGqaRankFused <= G::kHidden && LingLayout<G>::kGqaRankWidth % (G::kGqaHeadDim != 0u ? G::kGqaHeadDim : 1u) == 0u),
		"a GQA rank owns whole query heads and its fused rows fit one collective row");
	static_assert(G::kKdaRank % 64u == 0u, "the low-rank decay and gate projections stream in 64-column steps");
	static_assert(G::kQLora % 64u == 0u, "the query LoRA projection aligns with streaming widths");
};

typedef struct LingWeight
{
	const void *weight;
	const float *scale;
} LingWeight;

typedef struct LingLayer
{
	const float *input_norm;
	const float *post_norm;
	LingWeight kda_qkv;
	LingWeight kda_f_a;
	LingWeight kda_f;
	LingWeight kda_g_a;
	LingWeight kda_g;
	const uint16_t *kda_beta;
	const uint16_t *kda_conv;
	const float *kda_a_log;
	const float *kda_dt_bias;
	const uint16_t *kda_norm;
	LingWeight kda_out;
	LingWeight mla_q;
	LingWeight mla_q_a;
	const float *mla_q_norm;
	LingWeight mla_q_b;
	LingWeight mla_kv_a;
	const float *mla_kv_norm;
	const uint16_t *mla_uk;
	const uint16_t *mla_uv;
	const uint16_t *mla_gate;
	LingWeight mla_out;
	const uint16_t *gqa_qkv;
	const uint16_t *gqa_gate;
	const uint16_t *gqa_out;
	LingWeight ffn_gate_up;
	LingWeight ffn_down;
	const uint16_t *router;
	const float *expert_bias;
	LingWeight moe_w1;
	LingWeight moe_w3;
	LingWeight moe_w2;
	const uint16_t *shared_gate_up;
	const uint16_t *shared_down;
	uint32_t recurrent_index;
	uint32_t kv_index;
} LingLayer;

template<class G>
struct LingState : CudaStagePackState
{
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint64_t partial_stride;
	uint64_t kda_state_bytes;
	uint64_t kda_window_bytes;
	LingLayer layers[G::kLayers];
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *qkv;
	uint16_t *kda_low_f;
	uint16_t *kda_low_g;
	uint16_t *decay_logit;
	uint16_t *kda_gate;
	uint16_t *beta;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	float *retention;
	float *write_gate;
	uint16_t *kda_mixed;
	uint16_t *replay_qkv;
	float *replay_retention;
	float *replay_write_gate;
	float *kda_snapshots;
	uint16_t *mla_q_resid;
	uint16_t *mla_q;
	uint16_t *mla_latent_query;
	uint16_t *mla_query;
	uint16_t *mla_kv;
	uint16_t *mla_gathered;
	float *attn_send;
	float *attn_received;
	uint16_t *mla_latent;
	uint16_t *mla_value;
	uint16_t *mla_gate;
	uint16_t *gqa_local;
	uint16_t *gqa_gathered;
	uint16_t *gqa_full;
	uint16_t *gqa_query;
	uint16_t *gqa_slot;
	uint16_t *gqa_gate;
	uint16_t *gqa_merged;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *router_logits;
	uint32_t *route_expert;
	uint32_t *route_global;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	uint32_t *active_experts;
	float *packed_weight;
	uint16_t *expert_gate;
	uint16_t *expert_up;
	uint16_t *expert_act;
	uint16_t *expert_out;
	uint16_t *shared_gate_up;
	uint16_t *shared_act;
	uint16_t *shared_out;
	uint8_t *kda_state;
	uint16_t *kda_window;
	uint32_t *gather_seq;
	uint32_t *gather_offset;
	uint32_t *gather_prior_ctx;
	uint32_t *gather_total_ctx;
	LmKvShardView kv_views[LingShape<G>::kAttnLayers];
};

template<bool ROUNDED, bool WEIGHTED>
struct LingSiluClampOp
{
	float weight, limit;
	__device__ __forceinline__ float operator()(float g, float u) const
	{
		const float activated = SparkMathClampedGlu(ROUNDED ? SparkMathRoundBf16(SparkMathSilu(g)) : SparkMathSilu(g), u, limit);
		return WEIGHTED ? activated * weight : activated;
	}
};

#include "ling_fast.cuh"

template<uint32_t WIDTH, bool ROUND_FIRST, bool HAS_ADD>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingRmsNormKernel(uint16_t *__restrict__ hidden, const uint16_t *__restrict__ partial, const float *__restrict__ weight,
	uint16_t *__restrict__ normed, float epsilon)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= LG_THREADS * LM_RMS_NORM_STAGED, "width fits in staged vectorized waves");
	constexpr uint32_t VEC_SLOTS = (WIDTH + LG_THREADS * 8u - 1u) / (LG_THREADS * 8u);
	__shared__ float sq_wave[LG_THREADS * 8u];
	__shared__ float reduction[LmBlockWarps<LG_THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * WIDTH;
	uint16_t *h_row = hidden + base;
	const uint16_t *p_row = HAS_ADD ? partial + base : (const uint16_t *)0;
	uint16_t *n_row = normed + base;
	uint4 staged[VEC_SLOTS];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t wave_base = slot * LG_THREADS * 8u;
		const uint32_t offset = wave_base + threadIdx.x * 8u;
		if ( offset < WIDTH )
		{
			float vals[8], sq[8];
			staged[slot] = LmRmsLoadAdd8<HAS_ADD>(h_row + offset, HAS_ADD ? p_row + offset : (const uint16_t *)0, vals);
			if constexpr ( HAS_ADD )
				*(uint4 *)(h_row + offset) = staged[slot];
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
			if ( wave_base + s * LG_THREADS + threadIdx.x < WIDTH )
				total += sq_wave[s * LG_THREADS + threadIdx.x];
		if ( slot + 1u < VEC_SLOTS )
			__syncthreads();
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<LG_THREADS>(total, reduction), (float)WIDTH, epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t offset = (slot * LG_THREADS + threadIdx.x) * 8u;
		if ( offset < WIDTH )
		{
			float vals[8];
			LmLoadBf16x8AsFloat(staged[slot], vals);
			LmRmsStore8(vals, weight, offset, scale, n_row + offset, LmRmsGainOp<ROUND_FIRST>{});
		}
	}
	LmDependentRelease();
}

template<uint32_t WIDTH>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingAddRowsKernel(uint16_t *__restrict__ hidden, const uint16_t *__restrict__ partial)
{
	static_assert((WIDTH & 7u) == 0u, "hidden width is whole 16-byte vectors");
	constexpr uint32_t VECS = WIDTH >> 3u;
	const uint64_t base = (uint64_t)blockIdx.y * VECS;
	uint4 *h4 = (uint4 *)hidden + base;
	const uint4 *p4 = (const uint4 *)partial + base;
	LmDependentWait();
	for ( uint32_t c = blockIdx.x * LG_THREADS + threadIdx.x; c < VECS; c += gridDim.x * LG_THREADS )
	{
		float a[8], b[8];
		LmLoadBf16x8AsFloat(h4[c], a);
		LmLoadBf16x8AsFloat(p4[c], b);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			a[k] += b[k];
		h4[c] = LmPackUint4FloatToBf16(a);
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN>
__global__ __launch_bounds__(128u, 2)
static void LingSmallNBf16ProjKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation, uint16_t *__restrict__ output,
	uint32_t rows, uint32_t output_dim)
{
	constexpr uint32_t VECS = HIDDEN >> 3u;
	constexpr uint32_t WARPS = 4u;
	__shared__ __align__(16) uint4 s_x[VECS];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row < rows )
	{
		const uint4 *x4 = (const uint4 *)(activation + (uint64_t)row * HIDDEN);
		for ( uint32_t c = threadIdx.x; c < VECS; c += 128u )
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

template<uint32_t WIDTH>
static __device__ __forceinline__ void LingKdaWriteSplit(uint16_t *__restrict__ query, uint16_t *__restrict__ key, uint16_t *__restrict__ value,
	uint32_t row, uint32_t channel, uint16_t out_bf16)
{
	if ( channel < WIDTH )
		query[(uint64_t)row * WIDTH + channel] = out_bf16;
	else if ( channel < 2u * WIDTH )
		key[(uint64_t)row * WIDTH + (channel - WIDTH)] = out_bf16;
	else
		value[(uint64_t)row * WIDTH + (channel - 2u * WIDTH)] = out_bf16;
}

template<uint32_t CONV, bool ROUNDED_SUM>
static __device__ __forceinline__ uint16_t LingKdaConvActivate(float total)
{
	if constexpr ( ROUNDED_SUM )
		total = LmBf16ToFloat(LmFloatToBf16(total));
	return LmFloatToBf16(SparkMathSwishFast(total));
}

template<bool DOUBLED_BETA>
static __device__ __forceinline__ float LingKdaBetaValue(uint16_t raw)
{
	if constexpr ( DOUBLED_BETA )
		return 2.0f * SparkMathRoundBf16(SparkMathSigmoidFast(LmBf16ToFloat(raw)));
	else
		return SparkMathSigmoidFast(LmBf16ToFloat(raw));
}

template<uint32_t HIDDEN>
static __device__ __forceinline__ float LingKdaBetaDot(const uint16_t *__restrict__ w_row, const uint16_t *__restrict__ x_row, uint32_t lane)
{
	constexpr uint32_t VECS = HIDDEN >> 3u;
	const uint4 *w4 = (const uint4 *)w_row;
	const uint4 *x4 = (const uint4 *)x_row;
	float acc = 0.0f;
	for ( uint32_t c = lane; c < VECS; c += LM_WARP_LANES )
		acc = LmDotBf16x8(__ldg(x4 + c), __ldg(w4 + c), acc);
	#pragma unroll
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		acc += __shfl_down_sync(0xffffffffu, acc, offset);
	return acc;
}

template<class G, bool REPLAY, bool FUSE_BETA = false>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingKdaPrepSerialKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint16_t *__restrict__ qkv, const uint16_t *__restrict__ conv_weight,
	const uint16_t *__restrict__ decay_logit, const float *__restrict__ dt_bias, const float *__restrict__ a_log,
	const uint16_t *__restrict__ beta, const uint16_t *__restrict__ beta_weight, const uint16_t *__restrict__ normed,
	uint16_t *__restrict__ query, uint16_t *__restrict__ key, uint16_t *__restrict__ value,
	float *__restrict__ retention, float *__restrict__ write_gate, uint16_t *__restrict__ replay_qkv,
	float *__restrict__ replay_retention, float *__restrict__ replay_write_gate,
	uint32_t sequences, uint32_t rows, uint32_t commit)
{
	using S = LingShape<G>;
	constexpr uint32_t CHANNELS = S::kKdaChannels, WIDTH = S::kKdaWidth, HEADS = S::kKdaRankHeads;
	constexpr uint32_t CONV_BLOCKS = (CHANNELS + LG_THREADS - 1u) / LG_THREADS;
	const uint32_t blk = blockIdx.y;
	LmDependentWait();
	if ( blk < CONV_BLOCKS )
	{
		const uint32_t sequence = blockIdx.x;
		const uint32_t channel = blk * LG_THREADS + threadIdx.x;
		if ( sequence < sequences && channel < CHANNELS )
		{
			const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
			const uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
			uint16_t *slot = window + ((uint64_t)state_index[sequence] * CHANNELS + channel) * G::kConv;
			uint16_t taps[G::kConv];
			float w[G::kConv];
			#pragma unroll
			for ( uint32_t tap = 0u; tap < G::kConv; ++tap )
			{
				taps[tap] = slot[tap];
				w[tap] = LmBf16ToFloat(conv_weight[(uint64_t)channel * G::kConv + tap]);
			}
			for ( uint32_t ordinal = begin; ordinal < end; ++ordinal )
			{
				const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
				const uint16_t in_val = qkv[(uint64_t)row * CHANNELS + channel];
				if constexpr ( REPLAY )
					replay_qkv[(uint64_t)row * CHANNELS + channel] = in_val;
				#pragma unroll
				for ( uint32_t tap = 0u; tap + 1u < G::kConv; ++tap )
					taps[tap] = taps[tap + 1u];
				taps[G::kConv - 1u] = in_val;
				float total = 0.0f;
				#pragma unroll
				for ( uint32_t tap = 0u; tap < G::kConv; ++tap )
					total += LmBf16ToFloat(taps[tap]) * w[tap];
				LingKdaWriteSplit<WIDTH>(query, key, value, row, channel, LingKdaConvActivate<G::kConv, G::kKdaConvRoundedSum>(total));
			}
			if ( commit != 0u )
			{
				#pragma unroll
				for ( uint32_t tap = 0u; tap < G::kConv; ++tap )
					slot[tap] = taps[tap];
			}
		}
	}
	else
	{
		const uint32_t db = blk - CONV_BLOCKS;
		const uint32_t col = db * LG_THREADS + threadIdx.x;
		const uint32_t row_per_seq = (rows + sequences - 1u) / sequences;
		const uint32_t r0 = blockIdx.x * row_per_seq;
		const uint32_t r1 = r0 + row_per_seq < rows ? r0 + row_per_seq : rows;
		if ( col < WIDTH )
		{
			const uint32_t head = col / G::kKdaDim;
			const float bias = dt_bias[col];
			const float a_scale = a_log[head];
			for ( uint32_t row = r0; row < r1; ++row )
			{
				const uint64_t idx = (uint64_t)row * WIDTH + col;
				const float ret = LmBoundedDecay(LmBf16ToFloat(decay_logit[idx]), bias, a_scale, G::kKdaLowerBound);
				retention[idx] = ret;
				if constexpr ( REPLAY )
					replay_retention[idx] = ret;
			}
		}
		if constexpr ( FUSE_BETA )
		{
			if ( db == 0u )
			{
				const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
				if ( warp < HEADS )
				{
					const uint16_t *w_row = beta_weight + (uint64_t)warp * G::kHidden;
					for ( uint32_t row = r0; row < r1; ++row )
					{
						const float acc = LingKdaBetaDot<G::kHidden>(w_row, normed + (uint64_t)row * G::kHidden, lane);
						if ( lane == 0u )
						{
							const uint64_t idx = (uint64_t)row * HEADS + warp;
							const float wg = LingKdaBetaValue<G::kKdaDoubledBeta>(LmFloatToBf16(acc));
							write_gate[idx] = wg;
							if constexpr ( REPLAY )
								replay_write_gate[idx] = wg;
						}
					}
				}
			}
		}
		else if ( db == 0u && threadIdx.x < HEADS )
		{
			const uint32_t head = threadIdx.x;
			for ( uint32_t row = r0; row < r1; ++row )
			{
				const uint64_t idx = (uint64_t)row * HEADS + head;
				const float wg = LingKdaBetaValue<G::kKdaDoubledBeta>(beta[idx]);
				write_gate[idx] = wg;
				if constexpr ( REPLAY )
					replay_write_gate[idx] = wg;
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool REPLAY>
__global__ __launch_bounds__(LM_CAUSAL_CONV_TILE_THREADS, 1)
static void LingKdaConvSplitTiledKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count,
	const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ query, uint16_t *__restrict__ key, uint16_t *__restrict__ value,
	uint16_t *__restrict__ replay_qkv, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices)
{
	using S = LingShape<G>;
	constexpr uint32_t CHANNELS = S::kKdaChannels, WIDTH = S::kKdaWidth;
	constexpr uint32_t SPAN = G::kConv, HALO = SPAN - 1u;
	constexpr uint32_t GROUPS = LM_CAUSAL_CONV_TILE_THREADS / LM_CAUSAL_CONV_TILE_CHANNELS;
	__shared__ uint16_t tile[HALO + LM_CAUSAL_CONV_TILE_ROWS][LM_CAUSAL_CONV_TILE_CHANNELS];
	const uint32_t sequence = blockIdx.x, lane = threadIdx.x % LM_CAUSAL_CONV_TILE_CHANNELS, group = threadIdx.x / LM_CAUSAL_CONV_TILE_CHANNELS;
	const uint32_t channel = blockIdx.y * LM_CAUSAL_CONV_TILE_CHANNELS + lane;
	const bool live = channel < CHANNELS;
	LmDependentWait();
	if ( sequence < sequences )
	{
		uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
		uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
		if ( sequence_row_count != 0 )
			end = begin + sequence_row_count[sequence];
		uint16_t *slot = window + ((uint64_t)state_index[sequence] * CHANNELS + channel) * SPAN;
		float taps_weight[SPAN];
		uint16_t carried[HALO];
		#pragma unroll
		for ( uint32_t tap = 0u; tap < SPAN; ++tap )
			taps_weight[tap] = live ? LmBf16ToFloat(weight[(uint64_t)channel * SPAN + tap]) : 0.0f;
		if ( group == 0u && live )
			#pragma unroll
			for ( uint32_t tap = 0u; tap < HALO; ++tap )
				tile[tap][lane] = slot[1u + tap];
		for ( uint32_t base = begin; base < end; base += LM_CAUSAL_CONV_TILE_ROWS )
		{
			const uint32_t count = end - base < LM_CAUSAL_CONV_TILE_ROWS ? end - base : LM_CAUSAL_CONV_TILE_ROWS;
			__syncthreads();
			for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
			{
				const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
				const uint16_t in_val = input_bf16[(uint64_t)row * CHANNELS + channel];
				tile[HALO + offset][lane] = in_val;
				if constexpr ( REPLAY )
					replay_qkv[(uint64_t)row * CHANNELS + channel] = in_val;
			}
			__syncthreads();
			for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
			{
				const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
				float total = 0.0f;
				#pragma unroll
				for ( uint32_t tap = 0u; tap < SPAN; ++tap )
					total += LmBf16ToFloat(tile[offset + tap][lane]) * taps_weight[tap];
				LingKdaWriteSplit<WIDTH>(query, key, value, row, channel, LingKdaConvActivate<SPAN, G::kKdaConvRoundedSum>(total));
			}
			__syncthreads();
			if ( group == 0u && live )
			{
				if ( commit != 0u && base + count == end )
					#pragma unroll
					for ( uint32_t tap = 0u; tap < SPAN; ++tap )
						slot[tap] = tile[count - 1u + tap][lane];
				#pragma unroll
				for ( uint32_t tap = 0u; tap < HALO; ++tap )
					carried[tap] = tile[count + tap][lane];
				#pragma unroll
				for ( uint32_t tap = 0u; tap < HALO; ++tap )
					tile[tap][lane] = carried[tap];
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingKdaConvSplitSerialKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count,
	const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ query, uint16_t *__restrict__ key, uint16_t *__restrict__ value,
	uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ parent_rows)
{
	using S = LingShape<G>;
	constexpr uint32_t CHANNELS = S::kKdaChannels, WIDTH = S::kKdaWidth, SPAN = G::kConv;
	const uint32_t sequence = blockIdx.x, channel = blockIdx.y * LG_THREADS + threadIdx.x;
	LmDependentWait();
	if ( sequence < sequences && channel < CHANNELS )
	{
		uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
		uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
		if ( sequence_row_count != 0 )
			end = begin + sequence_row_count[sequence];
		uint16_t *slot = window + ((uint64_t)state_index[sequence] * CHANNELS + channel) * SPAN;
		uint16_t taps[SPAN];
		float w[SPAN];
		#pragma unroll
		for ( uint32_t tap = 0u; tap < SPAN; ++tap )
		{
			taps[tap] = slot[tap];
			w[tap] = LmBf16ToFloat(weight[(uint64_t)channel * SPAN + tap]);
		}
		for ( uint32_t ordinal = begin; ordinal < end; ++ordinal )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
			if ( parent_rows != 0 )
			{
				uint32_t ancestor = row, missing = SPAN;
				#pragma unroll
				for ( uint32_t tap = 0u; tap < SPAN; ++tap )
				{
					if ( ancestor != 0xffffffffu )
					{
						taps[SPAN - 1u - tap] = input_bf16[(uint64_t)ancestor * CHANNELS + channel];
						ancestor = parent_rows[ancestor];
					}
					else
					{
						missing = missing == SPAN ? tap : missing;
						taps[SPAN - 1u - tap] = slot[SPAN - tap + missing - 1u];
					}
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t tap = 0u; tap + 1u < SPAN; ++tap )
					taps[tap] = taps[tap + 1u];
				taps[SPAN - 1u] = input_bf16[(uint64_t)row * CHANNELS + channel];
			}
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t tap = 0u; tap < SPAN; ++tap )
				total += LmBf16ToFloat(taps[tap]) * w[tap];
			LingKdaWriteSplit<WIDTH>(query, key, value, row, channel, LingKdaConvActivate<SPAN, G::kKdaConvRoundedSum>(total));
		}
		if ( commit != 0u && parent_rows == 0 )
		{
			#pragma unroll
			for ( uint32_t tap = 0u; tap < SPAN; ++tap )
				slot[tap] = taps[tap];
		}
	}
	LmDependentRelease();
}

template<class G, bool REPLAY, bool FUSE_BETA = false>
__global__ __launch_bounds__(FUSE_BETA ? LG_THREADS : 64u, 2)
static void LingKdaGatesKernel(const uint16_t *__restrict__ decay_logit, const float *__restrict__ dt_bias, const float *__restrict__ a_log,
	const uint16_t *__restrict__ beta, const uint16_t *__restrict__ beta_weight, const uint16_t *__restrict__ normed,
	float *__restrict__ retention, float *__restrict__ write_gate,
	float *__restrict__ replay_retention, float *__restrict__ replay_write_gate, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t WIDTH = S::kKdaWidth, HEADS = S::kKdaRankHeads, VECS = WIDTH >> 3u;
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row < rows )
	{
		const uint64_t base = (uint64_t)row * WIDTH;
		if constexpr ( !FUSE_BETA )
		{
			const uint4 *dl4 = (const uint4 *)(decay_logit + base);
			for ( uint32_t v = threadIdx.x; v < VECS; v += 64u )
			{
				const uint32_t col = v << 3u;
				const float a_scale = __ldg(a_log + (col / G::kKdaDim));
				float d[8], b[8], r[8];
				LmLoadBf16x8AsFloat(__ldg(dl4 + v), d);
				LmLoadFloat8(dt_bias + col, b);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					r[k] = LmBoundedDecay(d[k], b[k], a_scale, G::kKdaLowerBound);
				LmStoreFloat8(retention + base + col, r);
				if constexpr ( REPLAY )
					LmStoreFloat8(replay_retention + base + col, r);
			}
			for ( uint32_t h = threadIdx.x; h < HEADS; h += 64u )
			{
				const uint64_t bidx = (uint64_t)row * HEADS + h;
				const float wg = LingKdaBetaValue<G::kKdaDoubledBeta>(__ldg(beta + bidx));
				write_gate[bidx] = wg;
				if constexpr ( REPLAY )
					replay_write_gate[bidx] = wg;
			}
		}
		else
		{
			for ( uint32_t col = threadIdx.x; col < WIDTH; col += LG_THREADS )
			{
				const uint32_t head = col / G::kKdaDim;
				const float ret = LmBoundedDecay(LmBf16ToFloat(__ldg(decay_logit + base + col)), __ldg(dt_bias + col), __ldg(a_log + head), G::kKdaLowerBound);
				retention[base + col] = ret;
				if constexpr ( REPLAY )
					replay_retention[base + col] = ret;
			}
			constexpr uint32_t HVECS = G::kHidden >> 3u;
			__shared__ __align__(16) uint4 s_x[HVECS];
			const uint4 *x4 = (const uint4 *)(normed + (uint64_t)row * G::kHidden);
			for ( uint32_t c = threadIdx.x; c < HVECS; c += LG_THREADS )
				s_x[c] = __ldg(x4 + c);
			__syncthreads();
			const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
			if ( warp < HEADS )
			{
				const uint4 *w4 = (const uint4 *)(beta_weight + (uint64_t)warp * G::kHidden);
				float acc = 0.0f;
				for ( uint32_t c = lane; c < HVECS; c += LM_WARP_LANES )
					acc = LmDotBf16x8(s_x[c], __ldg(w4 + c), acc);
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
					acc += __shfl_down_sync(0xffffffffu, acc, offset);
				if ( lane == 0u )
				{
					const uint64_t bidx = (uint64_t)row * HEADS + warp;
					const float wg = LingKdaBetaValue<G::kKdaDoubledBeta>(LmFloatToBf16(acc));
					write_gate[bidx] = wg;
					if constexpr ( REPLAY )
						replay_write_gate[bidx] = wg;
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingKdaConvGatesPrefillKernel(const uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint16_t *__restrict__ qkv, const uint16_t *__restrict__ conv_weight, const uint16_t *__restrict__ decay_logit,
	const float *__restrict__ dt_bias, const float *__restrict__ a_log, const uint16_t *__restrict__ beta,
	uint16_t *__restrict__ query, uint16_t *__restrict__ key, uint16_t *__restrict__ value,
	float *__restrict__ retention, float *__restrict__ write_gate, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t CHANNELS = S::kKdaChannels, WIDTH = S::kKdaWidth, HEADS = S::kKdaRankHeads;
	constexpr uint32_t SPAN = G::kConv, HALO = SPAN - 1u, VECS = WIDTH >> 3u;
	constexpr uint32_t CONV_BLKS = (CHANNELS + LG_THREADS - 1u) / LG_THREADS;
	const uint32_t r0 = blockIdx.x * 32u;
	const uint32_t r1 = r0 + 32u < rows ? r0 + 32u : rows;
	LmDependentWait();
	if ( blockIdx.y < CONV_BLKS )
	{
		const uint32_t ch = blockIdx.y * LG_THREADS + threadIdx.x;
		if ( ch < CHANNELS && r0 < rows )
		{
			const uint16_t *slot = window + ((uint64_t)__ldg(state_index) * CHANNELS + ch) * SPAN;
			float w[SPAN];
			uint16_t taps[SPAN];
			#pragma unroll
			for ( uint32_t k = 0u; k < SPAN; ++k )
				w[k] = LmBf16ToFloat(__ldg(conv_weight + (uint64_t)ch * SPAN + k));
			#pragma unroll
			for ( uint32_t k = 0u; k < HALO; ++k )
				taps[k + 1u] = r0 + k >= HALO ? __ldg(qkv + (uint64_t)(r0 + k - HALO) * CHANNELS + ch) : __ldg(slot + 1u + r0 + k);
			for ( uint32_t r = r0; r < r1; ++r )
			{
				#pragma unroll
				for ( uint32_t k = 0u; k < HALO; ++k )
					taps[k] = taps[k + 1u];
				taps[HALO] = __ldg(qkv + (uint64_t)r * CHANNELS + ch);
				float total = 0.0f;
				#pragma unroll
				for ( uint32_t k = 0u; k < SPAN; ++k )
					total += LmBf16ToFloat(taps[k]) * w[k];
				LingKdaWriteSplit<WIDTH>(query, key, value, r, ch, LingKdaConvActivate<SPAN, G::kKdaConvRoundedSum>(total));
			}
		}
	}
	else
	{
		const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
		for ( uint32_t r = r0 + warp; r < r1; r += 8u )
		{
			const uint64_t base = (uint64_t)r * WIDTH;
			const uint4 *dl4 = (const uint4 *)(decay_logit + base);
			for ( uint32_t v = lane; v < VECS; v += 32u )
			{
				const uint32_t col = v << 3u;
				const float a_scale = __ldg(a_log + (col / G::kKdaDim));
				float d[8], b[8], out[8];
				LmLoadBf16x8AsFloat(__ldg(dl4 + v), d);
				LmLoadFloat8(dt_bias + col, b);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					out[k] = LmBoundedDecay(d[k], b[k], a_scale, G::kKdaLowerBound);
				LmStoreFloat8(retention + base + col, out);
			}
			if ( lane < HEADS )
			{
				const uint64_t bidx = (uint64_t)r * HEADS + lane;
				write_gate[bidx] = LingKdaBetaValue<G::kKdaDoubledBeta>(__ldg(beta + bidx));
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingKdaWindowCommitKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint16_t *__restrict__ qkv, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t CHANNELS = S::kKdaChannels, SPAN = G::kConv;
	const uint32_t ch = blockIdx.x * LG_THREADS + threadIdx.x;
	LmDependentWait();
	if ( ch < CHANNELS )
	{
		uint16_t *slot = window + ((uint64_t)__ldg(state_index) * CHANNELS + ch) * SPAN;
		#pragma unroll
		for ( uint32_t k = 0u; k < SPAN; ++k )
			slot[k] = __ldg(qkv + (uint64_t)(rows - SPAN + k) * CHANNELS + ch);
	}
	LmDependentRelease();
}

template<uint32_t HEAD_DIM, bool ROUND_ONCE>
__global__ __launch_bounds__(128u, 4)
static void LingKdaHeadNormGateWarpKernel(uint16_t *__restrict__ mixed, const uint16_t *__restrict__ gate, const uint16_t *__restrict__ weight,
	uint32_t total_heads, float epsilon)
{
	static_assert(HEAD_DIM == 128u, "KDA head dimension is 128");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t item = blockIdx.x * 4u + warp;
	LmDependentWait();
	if ( item < total_heads )
	{
		const uint64_t base = (uint64_t)item * HEAD_DIM;
		uint2 *m2 = (uint2 *)(mixed + base);
		const uint2 mv = __ldg((const uint2 *)(m2 + lane));
		const uint2 gv = __ldg((const uint2 *)(gate + base) + lane);
		const uint2 wv = __ldg((const uint2 *)weight + lane);
		const float v0 = __uint_as_float(mv.x << 16u), v1 = __uint_as_float(mv.x & 0xffff0000u);
		const float v2 = __uint_as_float(mv.y << 16u), v3 = __uint_as_float(mv.y & 0xffff0000u);
		float p0 = v0 * v0, p1 = v1 * v1, p2 = v2 * v2, p3 = v3 * v3;
		#pragma unroll
		for ( uint32_t off = 4u; off > 0u; off >>= 1u )
		{
			p0 += __shfl_down_sync(0xffffffffu, p0, off);
			p1 += __shfl_down_sync(0xffffffffu, p1, off);
			p2 += __shfl_down_sync(0xffffffffu, p2, off);
			p3 += __shfl_down_sync(0xffffffffu, p3, off);
		}
		float s = (p0 + p2) + (p1 + p3);
		s += __shfl_down_sync(0xffffffffu, s, 16u);
		s += __shfl_down_sync(0xffffffffu, s, 8u);
		const float scale = SparkMathRmsScale(__shfl_sync(0xffffffffu, s, 0u), (float)HEAD_DIM, epsilon);
		const float w0 = __uint_as_float(wv.x << 16u), w1 = __uint_as_float(wv.x & 0xffff0000u);
		const float w2 = __uint_as_float(wv.y << 16u), w3 = __uint_as_float(wv.y & 0xffff0000u);
		const float g0 = __uint_as_float(gv.x << 16u), g1 = __uint_as_float(gv.x & 0xffff0000u);
		const float g2 = __uint_as_float(gv.y << 16u), g3 = __uint_as_float(gv.y & 0xffff0000u);
		auto apply = [&](float v, float w, float g) -> float
		{
			return ROUND_ONCE ? SparkMathRmsNormGateOnce(v, scale, w, SparkMathSigmoidFast(g))
				: SparkMathRoundBf16(SparkMathRoundBf16(v * scale) * w) * SparkMathSigmoidFast(g);
		};
		uint2 out;
		out.x = LmPackBf16Pair(apply(v0, w0, g0), apply(v1, w1, g1));
		out.y = LmPackBf16Pair(apply(v2, w2, g2), apply(v3, w3, g3));
		m2[lane] = out;
	}
	LmDependentRelease();
}

template<uint32_t HEAD_DIM, bool ROUND_ONCE>
__global__ __launch_bounds__(128u, 2)
static void LingKdaHeadNormGateKernel(uint16_t *__restrict__ mixed, const uint16_t *__restrict__ gate, const uint16_t *__restrict__ weight,
	uint32_t heads, float epsilon)
{
	static_assert(HEAD_DIM == 128u, "KDA head dimension is 128");
	__shared__ float reduction[LmBlockWarps<128u>()];
	const uint32_t head = blockIdx.x, row = blockIdx.y;
	const uint64_t base = ((uint64_t)row * heads + head) * HEAD_DIM;
	LmDependentWait();
	const float v = LmBf16ToFloat(mixed[base + threadIdx.x]);
	const float scale = SparkMathRmsScale(LmBlockSumOnce<128u>(v * v, reduction), (float)HEAD_DIM, epsilon);
	const float w = LmBf16ToFloat(weight[threadIdx.x]);
	const float g = LmBf16ToFloat(gate[base + threadIdx.x]);
	mixed[base + threadIdx.x] = LmFloatToBf16(ROUND_ONCE ? SparkMathRmsNormGateOnce(v, scale, w, SparkMathSigmoidFast(g))
		: SparkMathRoundBf16(SparkMathRoundBf16(v * scale) * w) * SparkMathSigmoidFast(g));
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMlaPrepStoreKernel(const uint16_t *__restrict__ mla_q, const uint16_t *__restrict__ mla_kv, const float *__restrict__ kv_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row,
	LmKvShardView kv_view, uint16_t *__restrict__ mla_query)
{
	using S = LingShape<G>;
	constexpr uint32_t LAT = G::kLatent, HALF = G::kRope / 2u;
	constexpr uint32_t SLOTS = (LAT + LG_THREADS - 1u) / LG_THREADS;
	__shared__ float reduction[LmBlockWarps<LG_THREADS>()];
	const uint32_t task = blockIdx.x, row = blockIdx.y;
	LmDependentWait();
	if ( task < S::kMlaRankHeads )
	{
		if ( threadIdx.x < HALF )
		{
			const uint32_t pair = threadIdx.x;
			const uint64_t src_base = ((uint64_t)row * S::kMlaRankHeads + task) * S::kQkDim + G::kNope;
			const uint64_t dst_base = ((uint64_t)row * S::kMlaRankHeads + task) * S::kSlot + LAT;
			const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
			const uint32_t packed_in = *(const uint32_t *)(mla_q + src_base + 2u * pair);
			float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
			float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
			LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&x0, &x1, angle, 1.0f);
			*(uint32_t *)(mla_query + dst_base + 2u * pair) = LmPackBf16Pair(x0, x1);
		}
	}
	else
	{
		const uint32_t store_pos = store_positions[row];
		if ( SparkKvShardOwns(kv_view.shard, store_pos) != 0u )
		{
			uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
			if ( slot != 0 )
			{
				const uint16_t *src = mla_kv + (uint64_t)row * S::kSlot;
				float vals[SLOTS], total = 0.0f;
				#pragma unroll
				for ( uint32_t s = 0u; s < SLOTS; ++s )
				{
					const uint32_t idx = threadIdx.x + s * LG_THREADS;
					if ( idx < LAT )
					{
						const float v = LmBf16ToFloat(src[idx]);
						vals[s] = v;
						total += v * v;
					}
				}
				const float scale = SparkMathRmsScale(LmBlockSumOnce<LG_THREADS>(total, reduction), (float)LAT, G::kNormEpsilon);
				#pragma unroll
				for ( uint32_t s = 0u; s < SLOTS; ++s )
				{
					const uint32_t idx = threadIdx.x + s * LG_THREADS;
					if ( idx < LAT )
						slot[idx] = LmFloatToBf16(G::kNormRoundFirst ? SparkMathNormGainRounded(vals[s] * scale, kv_norm[idx])
							: vals[s] * scale * kv_norm[idx]);
				}
				if ( threadIdx.x < HALF )
				{
					const uint32_t pair = threadIdx.x;
					const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
					const uint32_t packed_in = *(const uint32_t *)(src + LAT + 2u * pair);
					float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
					float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
					LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&x0, &x1, angle, 1.0f);
					*(uint32_t *)(slot + LAT + 2u * pair) = LmPackBf16Pair(x0, x1);
				}
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(32u, 1)
static void LingGatherMetaKernel(const uint32_t *__restrict__ sequence_of_row, uint32_t old_ctx, uint32_t total_ctx,
	uint32_t *__restrict__ gather_seq, uint32_t *__restrict__ gather_offset, uint32_t *__restrict__ gather_prior_ctx,
	uint32_t *__restrict__ gather_total_ctx)
{
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		gather_seq[0] = seq;
		gather_offset[seq] = 0u;
		gather_prior_ctx[seq] = old_ctx;
		gather_total_ctx[seq] = total_ctx;
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMlaPrepStoreLocalKernel(const uint16_t *__restrict__ mla_q, const uint16_t *__restrict__ mla_kv, const float *__restrict__ kv_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view,
	uint16_t *__restrict__ mla_query, uint16_t *__restrict__ kv_all, uint32_t total_ctx)
{
	using S = LingShape<G>;
	constexpr uint32_t LAT = G::kLatent, HALF = G::kRope / 2u;
	constexpr uint32_t SLOTS = (LAT + LG_THREADS - 1u) / LG_THREADS;
	__shared__ float reduction[LmBlockWarps<LG_THREADS>()];
	const uint32_t task = blockIdx.x, row = blockIdx.y;
	LmDependentWait();
	if ( task < S::kMlaRankHeads )
	{
		if ( threadIdx.x < HALF )
		{
			const uint32_t pair = threadIdx.x;
			const uint64_t src_base = ((uint64_t)row * S::kMlaRankHeads + task) * S::kQkDim + G::kNope;
			const uint64_t dst_base = ((uint64_t)row * S::kMlaRankHeads + task) * S::kSlot + LAT;
			const float angle = (float)__ldg(positions + row) * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
			const uint32_t packed_in = *(const uint32_t *)(mla_q + src_base + 2u * pair);
			float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
			float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
			LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&x0, &x1, angle, 1.0f);
			*(uint32_t *)(mla_query + dst_base + 2u * pair) = LmPackBf16Pair(x0, x1);
		}
	}
	else
	{
		const uint32_t pos = __ldg(positions + row);
		const bool owns = SparkKvShardOwns(kv_view.shard, pos) != 0u;
		uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, __ldg(sequence_of_row + row), pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
		uint16_t *dst_all = pos < total_ctx ? kv_all + (uint64_t)pos * S::kSlot : (uint16_t *)0;
		const uint16_t *src = mla_kv + (uint64_t)row * S::kSlot;
		float vals[SLOTS], total = 0.0f;
		#pragma unroll
		for ( uint32_t s = 0u; s < SLOTS; ++s )
		{
			const uint32_t idx = threadIdx.x + s * LG_THREADS;
			if ( idx < LAT )
			{
				const float v = LmBf16ToFloat(src[idx]);
				vals[s] = v;
				total += v * v;
			}
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<LG_THREADS>(total, reduction), (float)LAT, G::kNormEpsilon);
		#pragma unroll
		for ( uint32_t s = 0u; s < SLOTS; ++s )
		{
			const uint32_t idx = threadIdx.x + s * LG_THREADS;
			if ( idx < LAT )
			{
				const uint16_t out_v = LmFloatToBf16(G::kNormRoundFirst ? SparkMathNormGainRounded(vals[s] * scale, kv_norm[idx])
					: vals[s] * scale * kv_norm[idx]);
				if ( slot != 0 )
					slot[idx] = out_v;
				if ( dst_all != 0 )
					dst_all[idx] = out_v;
			}
		}
		if ( threadIdx.x < HALF )
		{
			const uint32_t pair = threadIdx.x;
			const float angle = (float)pos * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
			const uint32_t packed_in = *(const uint32_t *)(src + LAT + 2u * pair);
			float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
			float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
			LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&x0, &x1, angle, 1.0f);
			const uint32_t packed_out = LmPackBf16Pair(x0, x1);
			if ( slot != 0 )
				*(uint32_t *)(slot + LAT + 2u * pair) = packed_out;
			if ( dst_all != 0 )
				*(uint32_t *)(dst_all + LAT + 2u * pair) = packed_out;
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMlaPrepStoreLocalFastKernel(const uint16_t *__restrict__ mla_q, const uint16_t *__restrict__ mla_kv, const float *__restrict__ kv_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view,
	uint16_t *__restrict__ mla_query, uint16_t *__restrict__ kv_all, uint32_t total_ctx)
{
	using S = LingShape<G>;
	constexpr uint32_t LAT = G::kLatent, HALF = G::kRope / 2u;
	constexpr uint32_t SLOTS = (LAT + LG_THREADS - 1u) / LG_THREADS;
	__shared__ float reduction[LmBlockWarps<LG_THREADS>()];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	const uint32_t pos = __ldg(positions + row);
	const bool owns = SparkKvShardOwns(kv_view.shard, pos) != 0u;
	uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, __ldg(sequence_of_row + row), pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
	uint16_t *dst_all = pos < total_ctx ? kv_all + (uint64_t)pos * S::kSlot : (uint16_t *)0;
	const uint16_t *src = mla_kv + (uint64_t)row * S::kSlot;
	float vals[SLOTS], total = 0.0f;
	#pragma unroll
	for ( uint32_t s = 0u; s < SLOTS; ++s )
	{
		const uint32_t idx = threadIdx.x + s * LG_THREADS;
		if ( idx < LAT )
		{
			const float v = LmBf16ToFloat(__ldg(src + idx));
			vals[s] = v;
			total += v * v;
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<LG_THREADS>(total, reduction), (float)LAT, G::kNormEpsilon);
	#pragma unroll
	for ( uint32_t s = 0u; s < SLOTS; ++s )
	{
		const uint32_t idx = threadIdx.x + s * LG_THREADS;
		if ( idx < LAT )
		{
			const uint16_t out_v = LmFloatToBf16(G::kNormRoundFirst ? SparkMathNormGainRounded(vals[s] * scale, __ldg(kv_norm + idx))
				: vals[s] * scale * __ldg(kv_norm + idx));
			if ( slot != 0 )
				slot[idx] = out_v;
			if ( dst_all != 0 )
				dst_all[idx] = out_v;
		}
	}
	if ( warp == 0u && lane < HALF )
	{
		const float angle = (float)pos * __powf(G::kRopeTheta, -2.0f * (float)lane / (float)G::kRope);
		const uint32_t packed_in = __ldg((const uint32_t *)(src + LAT + 2u * lane));
		float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
		float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
		LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&x0, &x1, angle, 1.0f);
		const uint32_t packed_out = LmPackBf16Pair(x0, x1);
		if ( slot != 0 )
			*(uint32_t *)(slot + LAT + 2u * lane) = packed_out;
		if ( dst_all != 0 )
			*(uint32_t *)(dst_all + LAT + 2u * lane) = packed_out;
	}
	else if ( warp >= 1u && warp <= S::kMlaRankHeads && lane < HALF )
	{
		const uint32_t h = warp - 1u;
		const uint64_t src_base = ((uint64_t)row * S::kMlaRankHeads + h) * S::kQkDim + G::kNope;
		const uint64_t dst_base = ((uint64_t)row * S::kMlaRankHeads + h) * S::kSlot + LAT;
		const float angle = (float)pos * __powf(G::kRopeTheta, -2.0f * (float)lane / (float)G::kRope);
		const uint32_t packed_in = __ldg((const uint32_t *)(mla_q + src_base + 2u * lane));
		float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
		float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
		LmRopePairRounded<LM_ROPE_ROUNDED_TERMS>(&x0, &x1, angle, 1.0f);
		*(uint32_t *)(mla_query + dst_base + 2u * lane) = LmPackBf16Pair(x0, x1);
	}
	LmDependentRelease();
}

template<class Kv, uint32_t SLOT>
__global__ __launch_bounds__(128u, 4)
static void LingMlaDirectPriorKvKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row, uint32_t old_ctx, uint16_t *__restrict__ kv_all)
{
	constexpr uint32_t SLOT_V = SLOT >> 3u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t pos = blockIdx.x * 4u + warp;
	LmDependentWait();
	if ( pos < old_ctx )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, seq, pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint4 *s4 = (const uint4 *)slot;
			uint4 *d4 = (uint4 *)(kv_all + (uint64_t)pos * SLOT);
			for ( uint32_t c = lane; c < SLOT_V; c += 32u )
				d4[c] = __ldg(s4 + c);
		}
	}
	LmDependentRelease();
}

template<uint32_t SLOT>
__global__ __launch_bounds__(128u, 4)
static void LingMlaUnpackPriorKvKernel(SparkKvShard shard, const uint8_t *__restrict__ gathered_bytes, uint64_t rank_stride,
	uint32_t prior_local_keys, uint32_t old_ctx, uint16_t *__restrict__ kv_all)
{
	constexpr uint32_t SLOT_V = SLOT >> 3u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t local = blockIdx.x * 4u + warp, src_rank = blockIdx.y;
	SparkKvShard src_shard = shard;
	src_shard.rank = src_rank;
	const uint32_t pos = SparkKvShardLocalPosition(src_shard, local);
	LmDependentWait();
	if ( local < prior_local_keys && src_rank < shard.degree && pos < old_ctx )
	{
		const uint4 *s4 = (const uint4 *)(gathered_bytes + (uint64_t)src_rank * rank_stride) + (uint64_t)local * SLOT_V;
		uint4 *d4 = (uint4 *)(kv_all + (uint64_t)pos * SLOT);
		for ( uint32_t c = lane; c < SLOT_V; c += 32u )
			d4[c] = __ldg(s4 + c);
	}
	LmDependentRelease();
}

template<uint32_t VALUE_DIM>
__global__ __launch_bounds__(128u, 4)
static void LingMlaHeadGateFastKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t total_heads)
{
	static_assert(VALUE_DIM == 128u, "fast MLA head gate requires VALUE_DIM 128");
	const uint32_t item = blockIdx.x * 8u + (threadIdx.x >> 4u);
	const uint32_t c = threadIdx.x & 15u;
	LmDependentWait();
	if ( item < total_heads )
	{
		uint4 *base = (uint4 *)(heads_bf16 + (uint64_t)item * VALUE_DIM);
		const float gate = SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(__ldg(gate_bf16 + item))));
		float v[8];
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(base + c)), v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			v[k] *= gate;
		base[c] = LmPackUint4FloatToBf16(v);
	}
	LmDependentRelease();
}

template<uint32_t VALUE_DIM>
__global__ __launch_bounds__(128u, 2)
static void LingMlaHeadGateKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads)
{
	static_assert((VALUE_DIM & 7u) == 0u, "MLA value dim is whole 16-byte vectors");
	constexpr uint32_t VECS = VALUE_DIM >> 3u;
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	uint4 *base = (uint4 *)(heads_bf16 + ((uint64_t)row * heads + head) * VALUE_DIM);
	LmDependentWait();
	const float gate = SparkMathRoundBf16(SparkMathSigmoid(LmBf16ToFloat(__ldg(gate_bf16 + (uint64_t)row * heads + head))));
	for ( uint32_t c = threadIdx.x; c < VECS; c += 128u )
	{
		float v[8];
		LmLoadBf16x8AsFloat(base[c], v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			v[k] *= gate;
		base[c] = LmPackUint4FloatToBf16(v);
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingGqaPrepStoreKernel(const uint16_t *__restrict__ gathered, const uint32_t *__restrict__ store_positions,
	const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view, uint16_t *__restrict__ query_out, uint32_t rows)
{
	using S = LingShape<G>;
	constexpr uint32_t LOCAL_V = S::kGqaRankFused >> 3u;
	constexpr uint32_t Q_V = S::kGqaWidth >> 3u;
	constexpr uint32_t SLOT_V = S::kSlot >> 3u;
	constexpr uint32_t TOTAL_V = Q_V + SLOT_V;
	const uint32_t row = blockIdx.y;
	const uint32_t v = blockIdx.x * LG_THREADS + threadIdx.x;
	LmDependentWait();
	if ( row < rows && v < TOTAL_V )
	{
		const uint4 *g4 = (const uint4 *)gathered;
		if ( v < Q_V )
		{
			const uint32_t rank = v / LOCAL_V, elem = v % LOCAL_V;
			const uint4 val = g4[(((uint64_t)rank * rows) + row) * LOCAL_V + elem];
			((uint4 *)query_out)[(uint64_t)row * Q_V + v] = val;
		}
		else
		{
			const uint32_t store_pos = store_positions[row];
			if ( SparkKvShardOwns(kv_view.shard, store_pos) != 0u )
			{
				uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
				if ( slot != 0 )
				{
					const uint32_t rank = v / LOCAL_V, elem = v % LOCAL_V;
					((uint4 *)slot)[v - Q_V] = g4[(((uint64_t)rank * rows) + row) * LOCAL_V + elem];
				}
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(LG_THREADS, 2)
static void LingOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmDependentWait();
	LmGateUpRowBody<LG_THREADS>(gate_bf16 + base, output_bf16 + base, output_bf16 + base, dimension, LmSigmoidGateOp<true>{});
	LmDependentRelease();
}

template<bool ROUNDED>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingSiluMulLimitKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.x * width * 2u;
	LmDependentWait();
	if constexpr ( ROUNDED )
		LmGateUpRowBody<LG_THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.x * width, width, LmSiluRoundedLimitOp{limit});
	else
		LmGateUpRowBody<LG_THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.x * width, width, LmSiluLimitOp{limit});
	LmDependentRelease();
}

template<bool ROUNDED, bool WEIGHTED>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingSiluClampPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < __ldg(group_row_offset + local_experts) )
	{
		const uint64_t base = (uint64_t)pr * width;
		LmBinaryBf16GridBody(gate + base, up + base, output + base, width, LingSiluClampOp<ROUNDED, WEIGHTED>{WEIGHTED ? weight[pr] : 1.0f, limit});
	}
	LmDependentRelease();
}

template<bool WEIGHTED>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMoeLocalFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, const uint16_t *__restrict__ shared,
	uint16_t *__restrict__ output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK], active_count;
	__shared__ float active_weights[WEIGHTED ? LM_MOE_LOCAL_MAX_TOPK : 1u];
	const uint32_t row = blockIdx.y;
	const uint64_t row_base = (uint64_t)row * width;
	LmDependentWait();
	if ( threadIdx.x < 32u )
	{
		const uint32_t lane = threadIdx.x;
		const uint32_t route = row * top_k + lane;
		const bool keep = lane < top_k && __ldg(route_expert + route) < local_experts;
		const uint32_t mask = __ballot_sync(0xffffffffu, keep);
		if ( keep )
		{
			const uint32_t pos = (uint32_t)__popc(mask & ((1u << lane) - 1u));
			active_rows[pos] = __ldg(route_packed_row + route);
			if constexpr ( WEIGHTED )
				active_weights[pos] = __ldg(route_weight + route);
		}
		if ( lane == 0u )
			active_count = (uint32_t)__popc(mask);
	}
	__syncthreads();
	const uint32_t count = active_count;
	const uint32_t chunks = width >> 3u;
	const uint4 *packed4 = (const uint4 *)packed;
	const uint4 *shared4 = (const uint4 *)(shared + row_base);
	uint4 *out4 = (uint4 *)(output + row_base);
	if ( count == 0u )
	{
		for ( uint32_t c = blockIdx.x * LG_THREADS + threadIdx.x; c < chunks; c += gridDim.x * LG_THREADS )
			out4[c] = __ldcs(shared4 + c);
	}
	else
	{
		for ( uint32_t c = blockIdx.x * LG_THREADS + threadIdx.x; c < chunks; c += gridDim.x * LG_THREADS )
		{
			float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t k = 0u; k < count; ++k )
			{
				const uint4 v = __ldcs(packed4 + (uint64_t)active_rows[k] * chunks + c);
				if constexpr ( WEIGHTED )
					LmFmaBf16x8(acc, v, active_weights[k]);
				else
					LmAddBf16x8(acc, v);
			}
			LmAddBf16x8(acc, __ldcs(shared4 + c));
			out4[c] = LmPackUint4FloatToBf16(acc);
		}
	}
	LmDependentRelease();
}

template<class G, class Format>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMoePackedW1W3SiluKernel(const void *__restrict__ w1, const float *__restrict__ s1, const void *__restrict__ w3,
	const float *__restrict__ s3, const uint16_t *__restrict__ activation, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ packed_expert, const uint32_t *__restrict__ route_source_token,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output, float limit)
{
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr bool SCALED = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t CHUNKS = G::kHidden / ELEMS;
	constexpr uint32_t SC_STRIDE = G::kHidden / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = LG_THREADS / LANES;
	constexpr uint32_t N = G::kMoeInter;
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < __ldg(group_row_offset + G::kLocalExperts) )
	{
		const uint32_t exp = __ldg(packed_expert + pr);
		const uint32_t src_row = __ldg(route_source_token + pr);
		const uint16_t *act_row = activation + (uint64_t)src_row * G::kHidden;
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const uint64_t row_base = (uint64_t)exp * N + col0;
			const uint4 *w1_0 = (const uint4 *)w1 + row_base * CHUNKS;
			const uint4 *w3_0 = (const uint4 *)w3 + row_base * CHUNKS;
			const uint4 *w1_1 = has1 ? w1_0 + CHUNKS : w1_0;
			const uint4 *w3_1 = has1 ? w3_0 + CHUNKS : w3_0;
			const uint64_t sc_base0 = row_base * SC_STRIDE;
			const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
			float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const typename LmSkinnyFormat<Format>::Activation act = LmSkinnyFormat<Format>::Load(act_row + c * ELEMS);
				const uint4 vw1_0 = __ldcs(w1_0 + c);
				const uint4 vw3_0 = __ldcs(w3_0 + c);
				const uint4 vw1_1 = __ldcs(w1_1 + c);
				const uint4 vw3_1 = __ldcs(w3_1 + c);
				const uint32_t sc_idx = c >> 3u;
				const float f1_0 = SCALED ? __ldg(s1 + sc_base0 + sc_idx) : 1.0f;
				const float f3_0 = SCALED ? __ldg(s3 + sc_base0 + sc_idx) : 1.0f;
				const float f1_1 = SCALED ? __ldg(s1 + sc_base1 + sc_idx) : 1.0f;
				const float f3_1 = SCALED ? __ldg(s3 + sc_base1 + sc_idx) : 1.0f;
				g0 = fmaf(f1_0, LmSkinnyFormat<Format>::Chunk(vw1_0, act), g0);
				u0 = fmaf(f3_0, LmSkinnyFormat<Format>::Chunk(vw3_0, act), u0);
				g1 = fmaf(f1_1, LmSkinnyFormat<Format>::Chunk(vw1_1, act), g1);
				u1 = fmaf(f3_1, LmSkinnyFormat<Format>::Chunk(vw3_1, act), u1);
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
				const LingSiluClampOp<G::kFfnSiluRounded, !G::kExpertWeightAfterDown> op{
					!G::kExpertWeightAfterDown ? __ldg(packed_weight + pr) : 1.0f, limit };
				const float v0 = op(SparkMathRoundBf16(g0), SparkMathRoundBf16(u0));
				if ( has1 )
				{
					const float v1 = op(SparkMathRoundBf16(g1), SparkMathRoundBf16(u1));
					*(uint32_t *)(output + (uint64_t)pr * N + col0) = LmPackBf16Pair(v0, v1);
				}
				else
					output[(uint64_t)pr * N + col0] = LmFloatToBf16(v0);
			}
		}
	}
	LmDependentRelease();
}

template<class G, class Format>
__global__ __launch_bounds__(LG_THREADS, 2)
static void LingMoeSmallW2FinalizeKernel(const void *__restrict__ w2, const float *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight,
	const uint16_t *__restrict__ shared_gate_up, const uint16_t *__restrict__ shared_down, uint16_t *__restrict__ output, float shared_limit)
{
	constexpr uint32_t ELEMS = LmSkinnyFormat<Format>::kElements;
	constexpr bool SCALED = LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32;
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t SC_STRIDE = K / LG_FP8_BLOCK;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = LG_THREADS / LANES;
	constexpr uint32_t N = G::kHidden;
	constexpr uint32_t SH_K = G::kSharedRows;
	constexpr uint32_t SH_CHUNKS = SH_K >> 3u;
	constexpr uint32_t SH_LANES = SH_CHUNKS >= 32u ? 32u : (SH_CHUNKS >= 16u ? 16u : 8u);
	__shared__ uint32_t s_exp[G::kTopK], s_pr[G::kTopK], s_count;
	__shared__ float s_wt[G::kExpertWeightAfterDown ? G::kTopK : 1u];
	__shared__ __align__(16) uint16_t s_sh_act[SH_K];
	const uint32_t row = blockIdx.y;
	const uint16_t *sh_gu_row = shared_gate_up + (uint64_t)row * 2u * SH_K;
	uint16_t *out_row = output + (uint64_t)row * N;
	LmDependentWait();
	for ( uint32_t i = threadIdx.x; i < SH_K; i += LG_THREADS )
	{
		const float g = LmBf16ToFloat(__ldg(sh_gu_row + i));
		const float u = LmBf16ToFloat(__ldg(sh_gu_row + SH_K + i));
		const float a = G::kFfnSiluRounded ? LmSiluRoundedLimitOp{shared_limit}(g, u) : LmSiluLimitOp{shared_limit}(g, u);
		s_sh_act[i] = LmFloatToBf16(a);
	}
	if ( threadIdx.x < 32u )
	{
		const uint32_t lane = threadIdx.x;
		const uint32_t route = row * G::kTopK + lane;
		const uint32_t e = lane < G::kTopK ? __ldg(route_expert + route) : G::kLocalExperts;
		const bool keep = lane < G::kTopK && e < G::kLocalExperts;
		const uint32_t mask = __ballot_sync(0xffffffffu, keep);
		if ( keep )
		{
			const uint32_t pos = (uint32_t)__popc(mask & ((1u << lane) - 1u));
			s_exp[pos] = e;
			s_pr[pos] = __ldg(route_packed_row + route);
			if constexpr ( G::kExpertWeightAfterDown )
				s_wt[pos] = __ldg(route_weight + route);
		}
		if ( lane == 0u )
			s_count = (uint32_t)__popc(mask);
	}
	__syncthreads();
	const uint32_t col0 = (blockIdx.x * GROUPS + (threadIdx.x / LANES)) * 2u;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	const bool valid_col = col0 < N, has1 = col0 + 1u < N;
	const uint32_t safe_c = valid_col ? col0 : 0u;
	const uint32_t count = s_count;
	const bool reuse = gridDim.y > 1u;
	float sh0 = 0.0f, sh1 = 0.0f;
	const uint4 *sh_w0 = (const uint4 *)shared_down + (uint64_t)safe_c * SH_CHUNKS;
	const uint4 *sh_w1 = has1 ? sh_w0 + SH_CHUNKS : sh_w0;
	const uint4 *sh_a = (const uint4 *)s_sh_act;
	if constexpr ( SH_LANES <= LANES )
	{
		if ( sub < SH_LANES )
		{
			#pragma unroll
			for ( uint32_t c = sub; c < SH_CHUNKS; c += SH_LANES )
			{
				const uint4 va = sh_a[c];
				const uint4 sw0 = reuse ? __ldg(sh_w0 + c) : __ldcs(sh_w0 + c);
				const uint4 sw1 = reuse ? __ldg(sh_w1 + c) : __ldcs(sh_w1 + c);
				sh0 = fmaf(1.0f, LmSkinnyBf16Chunk(sw0, va, 0.0f), sh0);
				sh1 = fmaf(1.0f, LmSkinnyBf16Chunk(sw1, va, 0.0f), sh1);
			}
		}
		#pragma unroll
		for ( uint32_t offset = SH_LANES >> 1u; offset > 0u; offset >>= 1u )
		{
			sh0 += __shfl_down_sync(0xffffffffu, sh0, offset);
			sh1 += __shfl_down_sync(0xffffffffu, sh1, offset);
		}
	}
	else
	{
		float p0_lo = 0.0f, p0_hi = 0.0f, p1_lo = 0.0f, p1_hi = 0.0f;
		#pragma unroll
		for ( uint32_t c = sub; c < SH_CHUNKS; c += 2u * LANES )
		{
			const uint4 va_lo = sh_a[c];
			const uint4 sw0_lo = reuse ? __ldg(sh_w0 + c) : __ldcs(sh_w0 + c);
			const uint4 sw1_lo = reuse ? __ldg(sh_w1 + c) : __ldcs(sh_w1 + c);
			p0_lo = fmaf(1.0f, LmSkinnyBf16Chunk(sw0_lo, va_lo, 0.0f), p0_lo);
			p1_lo = fmaf(1.0f, LmSkinnyBf16Chunk(sw1_lo, va_lo, 0.0f), p1_lo);
			if ( c + LANES < SH_CHUNKS )
			{
				const uint4 va_hi = sh_a[c + LANES];
				const uint4 sw0_hi = reuse ? __ldg(sh_w0 + c + LANES) : __ldcs(sh_w0 + c + LANES);
				const uint4 sw1_hi = reuse ? __ldg(sh_w1 + c + LANES) : __ldcs(sh_w1 + c + LANES);
				p0_hi = fmaf(1.0f, LmSkinnyBf16Chunk(sw0_hi, va_hi, 0.0f), p0_hi);
				p1_hi = fmaf(1.0f, LmSkinnyBf16Chunk(sw1_hi, va_hi, 0.0f), p1_hi);
			}
		}
		sh0 = p0_lo + p0_hi;
		sh1 = p1_lo + p1_hi;
		#pragma unroll
		for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
		{
			sh0 += __shfl_down_sync(0xffffffffu, sh0, offset);
			sh1 += __shfl_down_sync(0xffffffffu, sh1, offset);
		}
	}
	if ( count == 0u )
	{
		if ( sub == 0u && valid_col )
		{
			if ( has1 )
				*(uint32_t *)(out_row + col0) = LmPackBf16Pair(sh0, sh1);
			else
				out_row[col0] = LmFloatToBf16(sh0);
		}
	}
	else
	{
		float total0 = 0.0f, total1 = 0.0f;
		for ( uint32_t k = 0u; k < count; ++k )
		{
			const uint16_t *a_row = expert_act + (uint64_t)s_pr[k] * K;
			const uint64_t row_base = (uint64_t)s_exp[k] * N + safe_c;
			const uint4 *w0 = (const uint4 *)w2 + row_base * CHUNKS;
			const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
			const uint64_t sc_base0 = row_base * SC_STRIDE;
			const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
			float acc0 = 0.0f, acc1 = 0.0f;
			const bool route_reuse = (G::kLocalExperts < G::kExperts) && reuse;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const typename LmSkinnyFormat<Format>::Activation act = LmSkinnyFormat<Format>::Load(a_row + c * ELEMS);
				const uint4 vw0 = route_reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
				const uint4 vw1 = route_reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
				const uint32_t sc_idx = c >> 3u;
				const float f0 = SCALED ? __ldg(s2 + sc_base0 + sc_idx) : 1.0f;
				const float f1 = SCALED ? __ldg(s2 + sc_base1 + sc_idx) : 1.0f;
				acc0 = fmaf(f0, LmSkinnyFormat<Format>::Chunk(vw0, act), acc0);
				acc1 = fmaf(f1, LmSkinnyFormat<Format>::Chunk(vw1, act), acc1);
			}
			#pragma unroll
			for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
			{
				acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
				acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
			}
			if ( sub == 0u )
			{
				const float r0 = SparkMathRoundBf16(acc0), r1 = SparkMathRoundBf16(acc1);
				if constexpr ( G::kExpertWeightAfterDown )
				{
					const float wt = s_wt[k];
					total0 = fmaf(r0, wt, total0);
					total1 = fmaf(r1, wt, total1);
				}
				else
				{
					total0 += r0;
					total1 += r1;
				}
			}
		}
		if ( sub == 0u && valid_col )
		{
			total0 += SparkMathRoundBf16(sh0);
			if ( has1 )
			{
				total1 += SparkMathRoundBf16(sh1);
				*(uint32_t *)(out_row + col0) = LmPackBf16Pair(total0, total1);
			}
			else
				out_row[col0] = LmFloatToBf16(total0);
		}
	}
	LmDependentRelease();
}

template<class G>
struct LingModel : CudaStage<LingModel<G>>
{
	using S = LingShape<G>;
	using State = LingState<G>;
	using Layer = LingLayer;
	using Schedule = LingSchedule<G, LingModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::Kv, &State::kv_views, 0u, S::kAttnLayers>>;
	using LaneShared = CudaStageFields<&State::kda_state, &State::kda_window, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = G::kVerifyRows != 0u;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kNormEpsilon;
	static constexpr uint32_t kPackAlignment = LG_PACK_ALIGNMENT;

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

	static LmScaleTensor ScaleOf(const LingWeight &weight, uint32_t groups, uint32_t rows, uint32_t input_dimension)
	{
		return weight.scale != 0 ? LmScaleTensorBuild(weight.scale, LM_SCALE_ENCODING_F32, groups, rows, input_dimension, 1u, LG_FP8_BLOCK) : LmScaleTensorNone();
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		if ( exact == 0u && rows > LM_SKINNY_ROWS_WIDE )
			LmWeightPrefetchNote(stream, weight, (uint64_t)output_dimension * input_dimension * sizeof(uint16_t));
		if ( exact == 0u && (prefill_unified || rows > LM_SKINNY_ROWS_WIDE) && output_dimension <= 8u && destination_f32 == 0 && input_dimension == G::kHidden )
		{
			(void)LM_LAUNCH_DEPENDENT((LingSmallNBf16ProjKernel<G::kHidden>), rows, 128u, 0, stream, weight, source, destination, rows, output_dimension);
			return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
		}
		if ( prefill_unified )
			return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension,
				output_dimension, 0u, 0u, step->multiprocessors, stream);
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			exact, step->multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return Linear(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t Linear(const LingWeight &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		if ( weight.scale != 0 )
		{
			if ( exact == 0u && rows > LM_SKINNY_ROWS_WIDE )
				LmWeightPrefetchNote(stream, weight.weight, (uint64_t)output_dimension * input_dimension);
			if ( prefill_unified )
				return LmStreamGemmDense<LmFp8>(weight.weight, ScaleOf(weight, 1u, output_dimension, input_dimension), source, destination, (float *)0,
					rows, input_dimension, output_dimension, 0u, 0u, step->multiprocessors, stream);
			return LmDenseRows<LmFp8>(weight.weight, ScaleOf(weight, 1u, output_dimension, input_dimension), source, destination, (float *)0, rows,
				input_dimension, output_dimension, exact, step->multiprocessors, stream);
		}
		if ( exact == 0u && rows > LM_SKINNY_ROWS_WIDE )
			LmWeightPrefetchNote(stream, weight.weight, (uint64_t)output_dimension * input_dimension * sizeof(uint16_t));
		if ( prefill_unified )
			return LmStreamGemmDense<LmBf16Format>(weight.weight, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension,
				output_dimension, 0u, 0u, step->multiprocessors, stream);
		return LmDenseRows<LmBf16Format>(weight.weight, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension,
			output_dimension, exact, step->multiprocessors, stream);
	}

	static int32_t LaunchMultiDenseBf16(const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *source, uint32_t rows,
		uint32_t K, uint32_t sms, SparkGpuStream stream)
	{
		if ( count < 2u || count > 3u || (K & 63u) != 0u )
			return LM_LAUNCH_ERR_SHAPE;
		uint32_t n128 = 0u, n64 = 0u;
		for ( uint32_t i = 0u; i < count; ++i )
		{
			if ( targets[i].output_dimension <= 8u || (targets[i].output_dimension & 1u) != 0u )
				return LM_LAUNCH_ERR_SHAPE;
			n128 += (targets[i].output_dimension + 127u) / 128u;
			n64 += (targets[i].output_dimension + 63u) / 64u;
		}
		const bool use_128 = rows >= 256u && ((rows + 63u) / 64u) * n128 >= sms;
		const uint32_t tile_n = use_128 ? 128u : 64u;
		LingMultiDensePlan plan;
		memset(&plan, 0, sizeof(plan));
		plan.count = count;
		for ( uint32_t i = 0u; i < count; ++i )
		{
			plan.weight[i] = (const uint16_t *)targets[i].weight;
			plan.out_bf16[i] = targets[i].output_bf16;
			plan.out_f32[i] = targets[i].output_f32;
			plan.out_dim[i] = targets[i].output_dimension;
			plan.tile_prefix[i + 1u] = plan.tile_prefix[i] + (targets[i].output_dimension + tile_n - 1u) / tile_n;
		}
		uint32_t blocks = sms;
		if ( use_128 )
		{
			constexpr uint32_t sh = 4u * (64u + 128u) * 128u;
			if ( LingMoeMmaSetup<G, 51u>(LingMultiDenseMmaKernel<64u, 128u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
				return LM_LAUNCH_ERR_SHAPE;
			(void)LM_LAUNCH_DEPENDENT((LingMultiDenseMmaKernel<64u, 128u, 2u, 4u>), blocks, 256u, sh, stream, plan, source, rows, K);
		}
		else if ( rows <= 32u )
		{
			constexpr uint32_t sh = 4u * (32u + 64u) * 128u;
			if ( LingMoeMmaSetup<G, 52u>(LingMultiDenseMmaKernel<32u, 64u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
				return LM_LAUNCH_ERR_SHAPE;
			(void)LM_LAUNCH_DEPENDENT((LingMultiDenseMmaKernel<32u, 64u, 2u, 4u>), blocks, 256u, sh, stream, plan, source, rows, K);
		}
		else
		{
			constexpr uint32_t sh = 4u * (64u + 64u) * 128u;
			if ( LingMoeMmaSetup<G, 53u>(LingMultiDenseMmaKernel<64u, 64u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
				return LM_LAUNCH_ERR_SHAPE;
			(void)LM_LAUNCH_DEPENDENT((LingMultiDenseMmaKernel<64u, 64u, 2u, 4u>), blocks, 256u, sh, stream, plan, source, rows, K);
		}
		(void)n64;
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t LinearMulti(const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *source, uint32_t rows, uint32_t input_dimension,
		const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		if ( !prefill_unified && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			return LmSkinnyDenseMulti<LmBf16Format>(targets, count, source, rows, input_dimension, stream, exact);
		if ( prefill_unified && rows >= 32u && (input_dimension & 63u) == 0u )
		{
			LmSkinnyDenseTarget big[3];
			uint32_t small_idx[4];
			uint32_t big_count = 0u, small_count = 0u;
			bool can_fuse = true;
			for ( uint32_t i = 0u; i < count; ++i )
			{
				if ( targets[i].output_dimension <= 8u && targets[i].output_f32 == 0 && input_dimension == G::kHidden && small_count < 4u )
					small_idx[small_count++] = i;
				else if ( big_count < 3u )
					big[big_count++] = targets[i];
				else
					can_fuse = false;
			}
			if ( can_fuse && big_count >= 2u )
			{
				const int32_t st = LaunchMultiDenseBf16(big, big_count, source, rows, input_dimension, step->multiprocessors, stream);
				if ( st == LM_LAUNCH_OK )
				{
					for ( uint32_t s = 0u; s < small_count; ++s )
					{
						const LmSkinnyDenseTarget &t = targets[small_idx[s]];
						if ( Linear((const uint16_t *)t.weight, source, t.output_bf16, t.output_f32, rows, input_dimension,
								t.output_dimension, step, stream) != LM_LAUNCH_OK )
							SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
					}
					return LM_LAUNCH_OK;
				}
				if ( st != LM_LAUNCH_ERR_SHAPE )
					return st;
			}
		}
		for ( uint32_t index = 0u; index < count; ++index )
			if ( Linear((const uint16_t *)targets[index].weight, source, targets[index].output_bf16, targets[index].output_f32, rows, input_dimension,
					targets[index].output_dimension, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return LM_LAUNCH_OK;
	}

	static int32_t LinearMultiFp8(const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *source, uint32_t rows, uint32_t input_dimension,
		const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		if ( !prefill_unified && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			const int32_t status = LmSkinnyDenseMulti<LmFp8>(targets, count, source, rows, input_dimension, stream, exact);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		for ( uint32_t index = 0u; index < count; ++index )
		{
			if ( exact == 0u && rows > LM_SKINNY_ROWS_WIDE )
				LmWeightPrefetchNote(stream, targets[index].weight, (uint64_t)targets[index].output_dimension * input_dimension);
			const int32_t st = prefill_unified ?
				LmStreamGemmDense<LmFp8>(targets[index].weight, targets[index].scale, source, targets[index].output_bf16, targets[index].output_f32, rows,
					input_dimension, targets[index].output_dimension, 0u, 0u, step->multiprocessors, stream) :
				LmDenseRows<LmFp8>(targets[index].weight, targets[index].scale, source, targets[index].output_bf16, targets[index].output_f32, rows,
					input_dimension, targets[index].output_dimension, exact, step->multiprocessors, stream);
			if ( st != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		return LM_LAUNCH_OK;
	}

	template<class Format>
	static int32_t ExpertProjectionFormat(const LmLocalExpertRoutes &routes, const LingWeight &weight, const uint16_t *source, uint16_t *destination,
		uint32_t down, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream, bool force_stream = false)
	{
		const LmScaleTensor scale = ScaleOf(weight, G::kLocalExperts, output_dimension, input_dimension);
		if ( exact == 0u )
		{
			const uint32_t expected_local = (uint32_t)(((uint64_t)routes.routes * G::kLocalExperts + G::kExperts - 1u) / G::kExperts);
			const bool use_stream = force_stream || ((G::kLocalExperts == G::kExperts) ? (routes.rows >= 64u) : (routes.rows >= 32u));
			if ( use_stream )
			{
				const uint32_t stream_rows = expected_local > 0u ? expected_local : 1u;
				const int32_t launch = LmStreamGemmGrouped<Format>(weight.weight, scale, source, destination, routes.group_row_offset,
					down != 0u ? (const uint32_t *)0 : routes.route_source_token, routes.local_experts, stream_rows, input_dimension,
					output_dimension, routes.multiprocessors, stream);
				if ( launch != LM_LAUNCH_ERR_SHAPE )
					return launch;
			}
		}
		return down != 0u ? LmLocalExpertsDown<Format>(routes, weight.weight, scale, source, destination, input_dimension, output_dimension, stream, exact) :
			LmLocalExpertsUp<Format>(routes, weight.weight, scale, source, destination, input_dimension, output_dimension, stream, exact);
	}

	static int32_t ExpertUp(const LmLocalExpertRoutes &routes, const LingWeight &weight, const uint16_t *source, uint16_t *destination,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream, bool force_stream = false)
	{
		if ( weight.scale != 0 )
			return ExpertProjectionFormat<LmFp8>(routes, weight, source, destination, 0u, input_dimension, output_dimension, exact, stream, force_stream);
		return ExpertProjectionFormat<LmBf16Format>(routes, weight, source, destination, 0u, input_dimension, output_dimension, exact, stream, force_stream);
	}

	static int32_t ExpertDown(const LmLocalExpertRoutes &routes, const LingWeight &weight, const uint16_t *source, uint16_t *destination,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream, bool force_stream = false)
	{
		if ( weight.scale != 0 )
			return ExpertProjectionFormat<LmFp8>(routes, weight, source, destination, 1u, input_dimension, output_dimension, exact, stream, force_stream);
		return ExpertProjectionFormat<LmBf16Format>(routes, weight, source, destination, 1u, input_dimension, output_dimension, exact, stream, force_stream);
	}

	static uint16_t *ReplayQkv(State *state, const LingLayer *layer) { return state->replay_qkv + (uint64_t)layer->recurrent_index * G::kVerifyRows * S::kKdaChannels; }
	static float *ReplayRetention(State *state, const LingLayer *layer) { return state->replay_retention + (uint64_t)layer->recurrent_index * G::kVerifyRows * S::kKdaWidth; }
	static float *ReplayWriteGate(State *state, const LingLayer *layer) { return state->replay_write_gate + (uint64_t)layer->recurrent_index * G::kVerifyRows * S::kKdaRankHeads; }

	static void LaunchKdaGates(State *state, const LingLayer *layer, uint32_t rows, bool replay, bool beta_fused, SparkGpuStream stream)
	{
		if ( replay )
		{
			if ( beta_fused )
				(void)LM_LAUNCH_DEPENDENT((LingKdaGatesKernel<G, true, true>), rows, LG_THREADS, 0, stream, state->decay_logit, layer->kda_dt_bias,
					layer->kda_a_log, state->beta, layer->kda_beta, state->normed, state->retention, state->write_gate,
					ReplayRetention(state, layer), ReplayWriteGate(state, layer), rows);
			else
				(void)LM_LAUNCH_DEPENDENT((LingKdaGatesKernel<G, true, false>), rows, 64u, 0, stream, state->decay_logit, layer->kda_dt_bias,
					layer->kda_a_log, state->beta, layer->kda_beta, state->normed, state->retention, state->write_gate,
					ReplayRetention(state, layer), ReplayWriteGate(state, layer), rows);
		}
		else if ( beta_fused )
			(void)LM_LAUNCH_DEPENDENT((LingKdaGatesKernel<G, false, true>), rows, LG_THREADS, 0, stream, state->decay_logit, layer->kda_dt_bias,
				layer->kda_a_log, state->beta, layer->kda_beta, state->normed, state->retention, state->write_gate, (float *)0, (float *)0, rows);
		else
			(void)LM_LAUNCH_DEPENDENT((LingKdaGatesKernel<G, false, false>), rows, 64u, 0, stream, state->decay_logit, layer->kda_dt_bias,
				layer->kda_a_log, state->beta, layer->kda_beta, state->normed, state->retention, state->write_gate, (float *)0, (float *)0, rows);
	}

	static SparkStatus Kda(State *state, const Frame &frame, const LingLayer &entry)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sequences = step->sequences, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		bool beta_fused = false;
		uint8_t *state_pool = state->kda_state + (uint64_t)layer->recurrent_index * state->slots * state->kda_state_bytes;
		uint16_t *window = state->kda_window + (uint64_t)layer->recurrent_index * state->slots * (state->kda_window_bytes / sizeof(uint16_t));
		if constexpr ( G::kKdaRank != 0u )
		{
			const LmSkinnyDenseTarget targets[4] =
			{
				{ layer->kda_qkv.weight, state->qkv, (float *)0, S::kKdaChannels, 0u, 0u, LmScaleTensorNone() },
				{ layer->kda_beta, state->beta, (float *)0, S::kKdaRankHeads, 0u, 0u, LmScaleTensorNone() },
				{ layer->kda_f_a.weight, state->kda_low_f, (float *)0, G::kKdaRank, 0u, 0u, LmScaleTensorNone() },
				{ layer->kda_g_a.weight, state->kda_low_g, (float *)0, G::kKdaRank, 0u, 0u, LmScaleTensorNone() }
			};
			if ( LinearMulti(targets, 4u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( !prefill_unified && rows <= 8u && layer->kda_f.scale == 0 && layer->kda_g.scale == 0 )
			{
				if ( LingKdaLowRankPairLaunch<G::kKdaRank, S::kKdaWidth>(
						(const uint16_t *)layer->kda_f.weight, state->kda_low_f, state->decay_logit,
						(const uint16_t *)layer->kda_g.weight, state->kda_low_g, state->kda_gate,
						rows, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( Linear(layer->kda_f, state->kda_low_f, state->decay_logit, rows, G::kKdaRank, S::kKdaWidth, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_g, state->kda_low_g, state->kda_gate, rows, G::kKdaRank, S::kKdaWidth, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( layer->kda_qkv.scale != 0 )
		{
			const LmSkinnyDenseTarget targets[3] =
			{
				{ layer->kda_qkv.weight, state->qkv, (float *)0, S::kKdaChannels, S::kKdaChannels, 0u, ScaleOf(layer->kda_qkv, 1u, S::kKdaChannels, G::kHidden) },
				{ layer->kda_f.weight, state->decay_logit, (float *)0, S::kKdaWidth, S::kKdaWidth, 0u, ScaleOf(layer->kda_f, 1u, S::kKdaWidth, G::kHidden) },
				{ layer->kda_g.weight, state->kda_gate, (float *)0, S::kKdaWidth, S::kKdaWidth, 0u, ScaleOf(layer->kda_g, 1u, S::kKdaWidth, G::kHidden) }
			};
			if ( LinearMultiFp8(targets, 3u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if constexpr ( S::kKdaRankHeads <= 8u )
				beta_fused = true;
			else if ( Linear(layer->kda_beta, state->normed, state->beta, rows, G::kHidden, S::kKdaRankHeads, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			const LmSkinnyDenseTarget targets[4] =
			{
				{ layer->kda_qkv.weight, state->qkv, (float *)0, S::kKdaChannels, 0u, 0u, LmScaleTensorNone() },
				{ layer->kda_beta, state->beta, (float *)0, S::kKdaRankHeads, 0u, 0u, LmScaleTensorNone() },
				{ layer->kda_f.weight, state->decay_logit, (float *)0, S::kKdaWidth, 0u, 0u, LmScaleTensorNone() },
				{ layer->kda_g.weight, state->kda_gate, (float *)0, S::kKdaWidth, 0u, 0u, LmScaleTensorNone() }
			};
			if ( LinearMulti(targets, 4u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		const bool replay = G::kVerifyRows != 0u && step->commit == 0u;
		if ( step->commit == 0u && (G::kVerifyRows == 0u || rows > G::kVerifyRows) )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		if ( step->tree != 0u )
		{
			if ( replay && SparkGpuMemcpyAsync(ReplayQkv(state, layer), state->qkv, (uint64_t)rows * S::kKdaChannels * sizeof(uint16_t),
				SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((LingKdaConvSplitSerialKernel<G>), dim3(sequences, (S::kKdaChannels + LG_THREADS - 1u) / LG_THREADS),
				LG_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0,
				ReplayQkv(state, layer), layer->kda_conv, state->query, state->key, state->value, sequences, step->commit,
				step->sequence_row_indices, step->parent_rows);
			LaunchKdaGates(state, layer, rows, replay, beta_fused, stream);
		}
		else if ( rows <= LM_CAUSAL_CONV_SERIAL_ROWS * sequences )
		{
			constexpr uint32_t CONV_BLOCKS = (S::kKdaChannels + LG_THREADS - 1u) / LG_THREADS;
			constexpr uint32_t DECAY_BLOCKS = (S::kKdaWidth + LG_THREADS - 1u) / LG_THREADS;
			if ( replay )
			{
				if ( beta_fused )
					(void)LM_LAUNCH_DEPENDENT((LingKdaPrepSerialKernel<G, true, true>), dim3(sequences, CONV_BLOCKS + DECAY_BLOCKS), LG_THREADS, 0, stream,
						window, step->recurrent_index, step->sequence_row_begin, step->sequence_row_indices, state->qkv, layer->kda_conv,
						state->decay_logit, layer->kda_dt_bias, layer->kda_a_log, state->beta, layer->kda_beta, state->normed,
						state->query, state->key, state->value, state->retention, state->write_gate, ReplayQkv(state, layer),
						ReplayRetention(state, layer), ReplayWriteGate(state, layer), sequences, rows, step->commit);
				else
					(void)LM_LAUNCH_DEPENDENT((LingKdaPrepSerialKernel<G, true, false>), dim3(sequences, CONV_BLOCKS + DECAY_BLOCKS), LG_THREADS, 0, stream,
						window, step->recurrent_index, step->sequence_row_begin, step->sequence_row_indices, state->qkv, layer->kda_conv,
						state->decay_logit, layer->kda_dt_bias, layer->kda_a_log, state->beta, layer->kda_beta, state->normed,
						state->query, state->key, state->value, state->retention, state->write_gate, ReplayQkv(state, layer),
						ReplayRetention(state, layer), ReplayWriteGate(state, layer), sequences, rows, step->commit);
			}
			else if ( beta_fused )
				(void)LM_LAUNCH_DEPENDENT((LingKdaPrepSerialKernel<G, false, true>), dim3(sequences, CONV_BLOCKS + DECAY_BLOCKS), LG_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, step->sequence_row_indices, state->qkv, layer->kda_conv,
					state->decay_logit, layer->kda_dt_bias, layer->kda_a_log, state->beta, layer->kda_beta, state->normed,
					state->query, state->key, state->value, state->retention, state->write_gate, (uint16_t *)0, (float *)0, (float *)0,
					sequences, rows, step->commit);
			else
				(void)LM_LAUNCH_DEPENDENT((LingKdaPrepSerialKernel<G, false, false>), dim3(sequences, CONV_BLOCKS + DECAY_BLOCKS), LG_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, step->sequence_row_indices, state->qkv, layer->kda_conv,
					state->decay_logit, layer->kda_dt_bias, layer->kda_a_log, state->beta, layer->kda_beta, state->normed,
					state->query, state->key, state->value, state->retention, state->write_gate, (uint16_t *)0, (float *)0, (float *)0,
					sequences, rows, step->commit);
		}
		else if ( prefill_unified && !replay && !beta_fused && step->sequence_row_indices == 0 && rows >= 32u )
		{
			constexpr uint32_t CONV_BLKS = (S::kKdaChannels + LG_THREADS - 1u) / LG_THREADS;
			(void)LM_LAUNCH_DEPENDENT((LingKdaConvGatesPrefillKernel<G>), dim3((rows + 31u) / 32u, CONV_BLKS + 1u), LG_THREADS, 0, stream,
				window, step->recurrent_index, state->qkv, layer->kda_conv, state->decay_logit, layer->kda_dt_bias, layer->kda_a_log,
				state->beta, state->query, state->key, state->value, state->retention, state->write_gate, rows);
			(void)LM_LAUNCH_DEPENDENT((LingKdaWindowCommitKernel<G>), CONV_BLKS, LG_THREADS, 0, stream,
				window, step->recurrent_index, state->qkv, rows);
		}
		else
		{
			if ( replay )
				(void)LM_LAUNCH_DEPENDENT((LingKdaConvSplitTiledKernel<G, true>),
					dim3(sequences, (S::kKdaChannels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
					LM_CAUSAL_CONV_TILE_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0,
					state->qkv, layer->kda_conv, state->query, state->key, state->value, ReplayQkv(state, layer), sequences, step->commit,
					step->sequence_row_indices);
			else
				(void)LM_LAUNCH_DEPENDENT((LingKdaConvSplitTiledKernel<G, false>),
					dim3(sequences, (S::kKdaChannels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
					LM_CAUSAL_CONV_TILE_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0,
					state->qkv, layer->kda_conv, state->query, state->key, state->value, (uint16_t *)0, sequences, step->commit,
					step->sequence_row_indices);
			LaunchKdaGates(state, layer, rows, replay, beta_fused, stream);
		}
		const SparkGpuError delta_err = prefill_unified ?
			LmDeltaRuleChunkLaunch<LG_THREADS,G::kKdaDim,G::kKdaDim,float>(state_pool, (uint32_t)state->kda_state_bytes, step->recurrent_index,
				step->sequence_row_begin, (const uint32_t *)0, state->query, state->key, state->value, state->retention, state->write_gate, state->kda_mixed,
				S::kKdaRankHeads, 1u, sequences, step->commit, step->sequence_row_indices, stream) :
			LmDeltaRuleLaunch<LG_THREADS,G::kKdaDim,G::kKdaDim,float>(step->rows, state_pool, (uint32_t)state->kda_state_bytes, step->recurrent_index,
				step->sequence_row_begin, (const uint32_t *)0, state->query, state->key, state->value, state->retention, state->write_gate, state->kda_mixed,
				S::kKdaRankHeads, 1u, sequences, step->commit, step->sequence_row_indices, stream, step->tree != 0u ? step->state_loads : (const uint32_t *)0,
				step->tree != 0u ? step->state_saves : (const uint32_t *)0, step->tree != 0u ? state->kda_snapshots : (float *)0);
		if ( delta_err != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if constexpr ( S::kKdaWidth <= 1024u )
		{
			if ( !prefill_unified && rows <= 8u )
			{
				if ( layer->kda_out.scale != 0 )
				{
					if constexpr ( (S::kKdaWidth % LG_FP8_BLOCK) == 0u )
						return LingLaunchKdaNormGateOut<G, LmFp8>(state->kda_mixed, state->kda_gate, layer->kda_norm,
							layer->kda_out.weight, layer->kda_out.scale, state->partial, rows, stream);
				}
				else
					return LingLaunchKdaNormGateOut<G, LmBf16Format>(state->kda_mixed, state->kda_gate, layer->kda_norm,
						layer->kda_out.weight, (const float *)0, state->partial, rows, stream);
			}
		}
		if constexpr ( G::kKdaDim == 128u )
		{
			const uint32_t total_heads = rows * S::kKdaRankHeads;
			(void)LM_LAUNCH_DEPENDENT((LingKdaHeadNormGateWarpKernel<G::kKdaDim, G::kKdaNormRoundOnce>), (total_heads + 3u) / 4u, 128u, 0, stream,
				state->kda_mixed, state->kda_gate, layer->kda_norm, total_heads, G::kNormEpsilon);
		}
		else
			(void)LM_LAUNCH_DEPENDENT((LingKdaHeadNormGateKernel<G::kKdaDim, G::kKdaNormRoundOnce>), dim3(S::kKdaRankHeads, rows), 128u, 0, stream,
				state->kda_mixed, state->kda_gate, layer->kda_norm, S::kKdaRankHeads, G::kNormEpsilon);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->kda_out, state->kda_mixed, state->partial, rows, S::kKdaWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mla(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, degree = state->tp_degree, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		const uint32_t force_mma = prefill_unified ? 1u : 0u;
		const uint64_t query_stride = (uint64_t)rows * S::kMlaRankHeads * S::kSlot;
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t record_unit = unit_bytes / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * S::kMlaRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		const float scale = 1.0f / sqrtf((float)S::kQkDim);
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if constexpr ( G::kQLora != 0u )
		{
			if ( layer->mla_q_a.scale != 0 && layer->mla_kv_a.scale != 0 )
			{
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->mla_q_a.weight, state->mla_q_resid, (float *)0, G::kQLora, G::kQLora, 0u, ScaleOf(layer->mla_q_a, 1u, G::kQLora, G::kHidden) },
					{ layer->mla_kv_a.weight, state->mla_kv, (float *)0, S::kSlot, S::kSlot, 0u, ScaleOf(layer->mla_kv_a, 1u, S::kSlot, G::kHidden) }
				};
				if ( LinearMultiFp8(targets, 2u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK ||
					Linear(layer->mla_gate, state->normed, state->mla_gate, rows, G::kHidden, S::kMlaRankHeads, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( Linear(layer->mla_q_a, state->normed, state->mla_q_resid, rows, G::kHidden, G::kQLora, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->mla_kv_a, state->normed, state->mla_kv, rows, G::kHidden, S::kSlot, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->mla_gate, state->normed, state->mla_gate, rows, G::kHidden, S::kMlaRankHeads, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((LingRmsNormKernel<G::kQLora, G::kNormRoundFirst, false>), rows, LG_THREADS, 0, stream,
				state->mla_q_resid, (const uint16_t *)0, layer->mla_q_norm, state->mla_q_resid, G::kNormEpsilon);
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				Linear(layer->mla_q_b, state->mla_q_resid, state->mla_q, rows, G::kQLora, S::kMlaRankHeads * S::kQkDim, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( layer->mla_q.scale != 0 && layer->mla_kv_a.scale != 0 )
		{
			const LmSkinnyDenseTarget targets[2] =
			{
				{ layer->mla_q.weight, state->mla_q, (float *)0, S::kMlaRankHeads * S::kQkDim, S::kMlaRankHeads * S::kQkDim, 0u,
					ScaleOf(layer->mla_q, 1u, S::kMlaRankHeads * S::kQkDim, G::kHidden) },
				{ layer->mla_kv_a.weight, state->mla_kv, (float *)0, S::kSlot, S::kSlot, 0u, ScaleOf(layer->mla_kv_a, 1u, S::kSlot, G::kHidden) }
			};
			if ( LinearMultiFp8(targets, 2u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->mla_gate, state->normed, state->mla_gate, rows, G::kHidden, S::kMlaRankHeads, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( layer->mla_q.scale == 0 && layer->mla_kv_a.scale == 0 )
		{
			const LmSkinnyDenseTarget targets[3] =
			{
				{ layer->mla_q.weight, state->mla_q, (float *)0, S::kMlaRankHeads * S::kQkDim, 0u, 0u, LmScaleTensorNone() },
				{ layer->mla_kv_a.weight, state->mla_kv, (float *)0, S::kSlot, 0u, 0u, LmScaleTensorNone() },
				{ layer->mla_gate, state->mla_gate, (float *)0, S::kMlaRankHeads, 0u, 0u, LmScaleTensorNone() }
			};
			if ( LinearMulti(targets, 3u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Linear(layer->mla_q, state->normed, state->mla_q, rows, G::kHidden, S::kMlaRankHeads * S::kQkDim, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->mla_kv_a, state->normed, state->mla_kv, rows, G::kHidden, S::kSlot, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->mla_gate, state->normed, state->mla_gate, rows, G::kHidden, S::kMlaRankHeads, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( LmPerHeadProjectChainLaunch<LG_THREADS,G::kNope,G::kLatent,S::kQkDim,0u,S::kSlot>(state->mla_q, layer->mla_uk, state->mla_query, S::kMlaRankHeads,
				rows, stream, exact, force_mma) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint32_t total_ctx = step->gather_context;
		const uint32_t old_ctx = total_ctx > rows ? total_ctx - rows : 0u;
		const uint32_t prior_local_keys = old_ctx != 0u ? SparkKvShardGatherKeys(view->shard, old_ctx) : 0u;
		const uint32_t kv_pack_units = (degree > 1u && prior_local_keys != 0u) ?
			(uint32_t)(((uint64_t)prior_local_keys * S::Kv::kSlotBytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t kv_rank_stride = (uint64_t)kv_pack_units * unit_bytes;
		const uint64_t kv_all_bytes = (uint64_t)total_ctx * S::Kv::kSlotBytes;
		const uint64_t recv_cap_bytes = (uint64_t)degree * state->partial_stride * sizeof(float);
		const uint64_t gathered_cap_bytes = (uint64_t)degree * state->max_rows * S::kMlaRankHeads * S::kSlot * sizeof(uint16_t);
		const bool use_kv_gather = prefill_unified && total_ctx >= rows &&
			kv_all_bytes <= recv_cap_bytes &&
			kv_rank_stride <= recv_cap_bytes &&
			(uint64_t)degree * kv_rank_stride <= gathered_cap_bytes;
		if ( use_kv_gather )
		{
			uint16_t *kv_all = (uint16_t *)state->attn_received;
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			if constexpr ( S::kMlaRankHeads <= 7u && (G::kRope / 2u) <= 32u )
				(void)LM_LAUNCH_DEPENDENT((LingMlaPrepStoreLocalFastKernel<G>), rows, LG_THREADS, 0, stream,
					state->mla_q, state->mla_kv, layer->mla_kv_norm, step->positions, step->sequence_of_row, *view, state->mla_query, kv_all, total_ctx);
			else
				(void)LM_LAUNCH_DEPENDENT((LingMlaPrepStoreLocalKernel<G>), dim3(S::kMlaRankHeads + 1u, rows), LG_THREADS, 0, stream,
					state->mla_q, state->mla_kv, layer->mla_kv_norm, step->positions, step->sequence_of_row, *view, state->mla_query, kv_all, total_ctx);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			(void)LM_LAUNCH_DEPENDENT((LingGatherMetaKernel), 1u, 32u, 0, stream, step->sequence_of_row, old_ctx, total_ctx,
				state->gather_seq, state->gather_offset, state->gather_prior_ctx, state->gather_total_ctx);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( old_ctx != 0u )
			{
				if ( degree > 1u )
				{
					if ( LmKvShardGatherPackLaunch<typename S::Kv, LG_THREADS>(*view, state->gather_seq, state->gather_offset, state->gather_prior_ctx, 1u,
							prior_local_keys, (uint8_t *)state->attn_send, stream) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, kv_pack_units, 0u, state->attn_send, state->mla_gathered);
					if ( status != SPARK_STATUS_OK )
						return status;
					(void)LM_LAUNCH_DEPENDENT((LingMlaUnpackPriorKvKernel<S::kSlot>), dim3((prior_local_keys + 3u) / 4u, degree), 128u, 0, stream,
						view->shard, (const uint8_t *)state->mla_gathered, kv_rank_stride, prior_local_keys, old_ctx, kv_all);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((LingMlaDirectPriorKvKernel<typename S::Kv, S::kSlot>), (old_ctx + 3u) / 4u, 128u, 0, stream,
						*view, step->sequence_of_row, old_ctx, kv_all);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			SparkKvShard flat_shard = { 1u, 0u, 1u };
			LmKvShardGatherAllView all_view;
			if ( LmKvShardGatherAllViewInitialize<typename S::Kv>(&all_view, view->pages, flat_shard, (const uint8_t *)kv_all, kv_all_bytes,
					state->gather_offset, state->gather_total_ctx) != 0 )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			const uint64_t local_partial_stride = (uint64_t)rows * S::kMlaRankHeads * S::kRecord;
			LmWeightPrefetchNote(stream, layer->mla_uv, (uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent * sizeof(uint16_t));
			if ( LmLatentGatherPrefillLaunch<typename S::Kv, LmKvShardGatherAllView, G::kLatent, G::kRope>(all_view, state->mla_query, S::kMlaRankHeads,
					step->sequence_of_row, step->context_length, step->positions, scale, state->attn_send, rows, stream) != SPARK_GPU_SUCCESS ||
				LmLatentShardMergeLaunch<G::kLatent>(state->attn_send, local_partial_stride, 1u, S::kMlaRankHeads, state->mla_latent, rows, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			(void)LM_LAUNCH_DEPENDENT((LingMlaPrepStoreKernel<G>), dim3(S::kMlaRankHeads + 1u, rows), LG_THREADS, 0, stream,
				state->mla_q, state->mla_kv, layer->mla_kv_norm, step->positions, step->tree != 0u ? step->slots : step->positions,
				step->sequence_of_row, *view, state->mla_query);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kMlaRankHeads * S::kSlot, state->mla_query,
				state->mla_gathered);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( (rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS && step->commit != 0u &&
					(S::kMlaRankHeads * degree) % LM_LATENT_SHARD_QUERIES == 0u ?
					LmLatentShardPrefillLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->mla_gathered, query_stride, S::kMlaRankHeads,
						step->sequence_of_row, step->context_length, step->positions, scale, state->attn_send, partial_stride, rows, stream) :
					LmLatentShardAttendLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->mla_gathered, query_stride, S::kMlaRankHeads,
						step->sequence_of_row, step->context_length, step->positions, (const uint32_t *)0, 0u, 0u, scale, state->attn_send, partial_stride,
						rows, stream, step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
						step->tree != 0u ? step->slot_offsets : (const uint32_t *)0)) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->attn_send,
				state->attn_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			LmWeightPrefetchNote(stream, layer->mla_uv, (uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent * sizeof(uint16_t));
			if ( LmLatentShardMergeLaunch<G::kLatent>(state->attn_received, partial_stride, degree, S::kMlaRankHeads, state->mla_latent, rows, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( LmPerHeadProjectChainLaunch<LG_THREADS,G::kLatent,G::kValueDim>(state->mla_latent, layer->mla_uv, state->mla_value, S::kMlaRankHeads, rows,
				stream, exact, force_mma) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( S::kMlaRankHeads * G::kValueDim <= 1024u )
		{
			if ( !prefill_unified && rows <= 8u )
			{
				if ( layer->mla_out.scale != 0 )
				{
					if constexpr ( ((S::kMlaRankHeads * G::kValueDim) % LG_FP8_BLOCK) == 0u )
						return LingLaunchMlaGateOut<G, LmFp8>(state->mla_value, state->mla_gate, layer->mla_out.weight, layer->mla_out.scale,
							state->partial, rows, stream);
				}
				else
					return LingLaunchMlaGateOut<G, LmBf16Format>(state->mla_value, state->mla_gate, layer->mla_out.weight, (const float *)0,
						state->partial, rows, stream);
			}
		}
		if constexpr ( G::kValueDim == 128u )
		{
			const uint32_t total_heads = rows * S::kMlaRankHeads;
			(void)LM_LAUNCH_DEPENDENT((LingMlaHeadGateFastKernel<G::kValueDim>), (total_heads + 7u) / 8u, 128u, 0, stream,
				state->mla_value, state->mla_gate, total_heads);
		}
		else
			(void)LM_LAUNCH_DEPENDENT((LingMlaHeadGateKernel<G::kValueDim>), dim3(rows, S::kMlaRankHeads), 128u, 0, stream,
				state->mla_value, state->mla_gate, S::kMlaRankHeads);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->mla_out, state->mla_value, state->partial, rows, S::kMlaRankHeads * G::kValueDim, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Gqa(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, degree = state->tp_degree;
		const bool prefill_unified = PrefillUnified(step);
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t record_unit = unit_bytes / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * LmGqaShardRecordFloats(G::kGqaHeads, G::kGqaHeadDim, degree) + record_unit - 1u) /
			record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		const float scale = 1.0f / sqrtf((float)G::kGqaHeadDim);
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		const LmSkinnyDenseTarget targets[2] =
		{
			{ layer->gqa_qkv, state->gqa_local, (float *)0, S::kGqaRankFused, 0u, 0u, LmScaleTensorNone() },
			{ layer->gqa_gate, state->gqa_gate, (float *)0, S::kGqaRankWidth, 0u, 0u, LmScaleTensorNone() }
		};
		if ( LinearMulti(targets, 2u, state->normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kGqaRankFused, state->gqa_local, state->gqa_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		const uint32_t total_ctx = step->gather_context;
		const uint32_t old_ctx = total_ctx > rows ? total_ctx - rows : 0u;
		const uint32_t rank_heads = G::kGqaHeads / degree;
		const uint32_t group = G::kGqaHeads / G::kGqaKvHeads;
		const uint32_t q_first = state->tp_rank * rank_heads;
		const uint32_t kv_first = q_first / group;
		const uint32_t local_kvh = ((q_first + rank_heads - 1u) / group) - kv_first + 1u;
		const uint32_t max_peer_keys = (degree > 1u && old_ctx != 0u) ? SparkKvShardGatherKeys(view->shard, old_ctx) : 0u;
		const uint64_t kv_slot_bytes = (uint64_t)2u * G::kGqaHeadDim * sizeof(uint16_t);
		const uint32_t a2a_units = (degree > 1u && max_peer_keys != 0u) ?
			(uint32_t)(((uint64_t)local_kvh * max_peer_keys * kv_slot_bytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t a2a_stride_bytes = (uint64_t)a2a_units * unit_bytes;
		const uint64_t prior_bytes = (uint64_t)local_kvh * old_ctx * kv_slot_bytes;
		const uint64_t send_cap_bytes = (uint64_t)degree * state->partial_stride * sizeof(float);
		const bool use_local_gqa = prefill_unified && total_ctx >= rows && G::kGqaHeadDim == 128u &&
			(group % rank_heads == 0u || rank_heads % group == 0u) &&
			prior_bytes <= send_cap_bytes && (uint64_t)degree * a2a_stride_bytes <= send_cap_bytes;
		if constexpr ( G::kGqaHeadDim == 128u )
		{
			if ( use_local_gqa )
			{
				uint16_t *kv_prior = (uint16_t *)state->attn_send;
				const uint32_t local_total_v = rank_heads * (G::kGqaHeadDim >> 3u) + (S::kSlot >> 3u);
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
				(void)LM_LAUNCH_DEPENDENT((LingGqaPrepStoreLocalKernel<G>), dim3((local_total_v + LG_THREADS - 1u) / LG_THREADS, rows), LG_THREADS, 0, stream,
					state->gqa_gathered, step->positions, step->sequence_of_row, *view, state->gqa_query, state->gqa_slot, rows, q_first, rank_heads, kv_first, local_kvh);
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( old_ctx != 0u )
				{
					if ( degree > 1u )
					{
						(void)LM_LAUNCH_DEPENDENT((LingGqaPriorKvPackKernel<typename S::Kv, G::kGqaKvHeads, G::kGqaHeadDim>),
							dim3((max_peer_keys + 3u) / 4u, degree), 128u, 0, stream, *view, step->sequence_of_row, old_ctx, max_peer_keys,
							rank_heads, group, local_kvh, a2a_stride_bytes, (uint8_t *)state->attn_send);
						if ( CudaStageLaunched() != SPARK_STATUS_OK )
							SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
						status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, a2a_units, 0u, state->attn_send, state->attn_received);
						if ( status != SPARK_STATUS_OK )
							return status;
						(void)LM_LAUNCH_DEPENDENT((LingGqaPriorKvUnpackKernel<G::kGqaHeadDim>),
							dim3((max_peer_keys + 3u) / 4u, degree), 128u, 0, stream, view->shard, (const uint8_t *)state->attn_received,
							a2a_stride_bytes, old_ctx, max_peer_keys, local_kvh, kv_prior);
					}
					else
						(void)LM_LAUNCH_DEPENDENT((LingGqaPriorKvDirectKernel<typename S::Kv, G::kGqaKvHeads, G::kGqaHeadDim>),
							dim3((old_ctx + 3u) / 4u, local_kvh), 128u, 0, stream, *view, step->sequence_of_row, old_ctx, kv_first, local_kvh, kv_prior);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				const uint32_t num_chunks = (total_ctx + 511u) / 512u;
				const uint64_t chunk_bytes = (uint64_t)num_chunks * rows * rank_heads * (G::kGqaHeadDim + 2u) * sizeof(float);
				if ( num_chunks > 1u && chunk_bytes <= send_cap_bytes )
				{
					(void)LM_LAUNCH_DEPENDENT((LingGqaLocalPrefillAttnKernel<G::kGqaHeadDim, true>), dim3(rank_heads, (rows + 15u) / 16u, num_chunks), 64u, 0, stream,
						state->gqa_query, kv_prior, state->gqa_slot, step->positions, state->gqa_gate, rows, rank_heads, q_first, group, kv_first, old_ctx,
						scale, state->gqa_merged, state->attn_received);
					(void)LM_LAUNCH_DEPENDENT((LingGqaLocalPrefillMergeGateKernel<G::kGqaHeadDim>), dim3(rank_heads, rows), G::kGqaHeadDim, 0, stream,
						state->attn_received, step->positions, state->gqa_gate, rows, rank_heads, num_chunks, scale, state->gqa_merged);
				}
				else
					(void)LM_LAUNCH_DEPENDENT((LingGqaLocalPrefillAttnKernel<G::kGqaHeadDim, false>), dim3(rank_heads, (rows + 15u) / 16u), 64u, 0, stream,
						state->gqa_query, kv_prior, state->gqa_slot, step->positions, state->gqa_gate, rows, rank_heads, q_first, group, kv_first, old_ctx,
						scale, state->gqa_merged, (float *)0);
				if ( CudaStageLaunched() != SPARK_STATUS_OK ||
					Linear(layer->gqa_out, state->gqa_merged, state->partial, rows, S::kGqaRankWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				return SPARK_STATUS_OK;
			}
		}
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		{
			constexpr uint32_t TOTAL_V = (S::kGqaWidth + S::kSlot) >> 3u;
			(void)LM_LAUNCH_DEPENDENT((LingGqaPrepStoreKernel<G>), dim3((TOTAL_V + LG_THREADS - 1u) / LG_THREADS, rows), LG_THREADS, 0, stream,
				state->gqa_gathered, step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, *view, state->gqa_query, rows);
		}
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kGqaKvHeads,G::kGqaHeadDim,G::kGqaHeadDim>(*view, state->gqa_query, G::kGqaHeads,
				step->sequence_of_row, step->context_length, step->positions, scale, state->attn_send, partial_stride, rows,
				stream, (const uint32_t *)0, 0u, 1u, 0u, step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
				step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->attn_send,
			state->attn_received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kGqaHeadDim>(state->attn_received, partial_stride, degree, G::kGqaHeads, state->tp_rank, state->gqa_merged, rows,
			stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( S::kGqaRankWidth <= 1024u )
		{
			if ( !prefill_unified && rows <= 8u )
				return LingLaunchGqaGateOutBf16<G>(state->gqa_merged, state->gqa_gate, layer->gqa_out, state->partial, rows, stream);
		}
		(void)LM_LAUNCH_DEPENDENT((LingOutputGateKernel), rows, LG_THREADS, 0, stream, state->gqa_merged, state->gqa_gate, S::kGqaRankWidth);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->gqa_out, state->gqa_merged, state->partial, rows, S::kGqaRankWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Attention(State *state, const Frame &frame, const LingLayer &entry)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		if constexpr ( S::kGqa )
			return Gqa(state, layer, step, stream);
		else
			return Mla(state, layer, step, stream);
	}

	static SparkStatus Dense(State *state, const Frame &frame, const LingLayer &entry)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows;
		if ( Linear(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((LingSiluMulLimitKernel<G::kFfnSiluRounded>), rows, LG_THREADS, 0, stream, state->gate_up, state->intermediate,
			state->ffn_rows, INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->ffn_down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static void LaunchExactW1W3(State *state, const LingLayer *layer, const uint16_t *normed, uint32_t routes, float rlim, SparkGpuStream stream)
	{
		const uint32_t max_active = routes < G::kLocalExperts ? routes : G::kLocalExperts;
		constexpr uint32_t THREADS = G::kMoeInter <= 128u ? 128u : LG_THREADS;
		constexpr uint32_t BF16_CPG = G::kMoeInter <= 128u ? 1u : 2u;
		if ( layer->moe_w1.scale != 0 )
		{
			if constexpr ( (G::kHidden % LG_FP8_BLOCK) == 0u )
			{
				constexpr uint32_t CHUNKS = G::kHidden / 16u;
				constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t GROUPS = THREADS / LANES;
				if constexpr ( G::kMoeInter > 128u )
				{
					if ( routes <= 256u )
						(void)LM_LAUNCH_DEPENDENT((LingMoeExactW1W3SiluKernel<G, LmFp8, THREADS, 2u>),
							dim3((G::kMoeInter + 2u * GROUPS - 1u) / (2u * GROUPS), max_active), THREADS, 0, stream,
							layer->moe_w1.weight, layer->moe_w1.scale, layer->moe_w3.weight, layer->moe_w3.scale, normed,
							state->packed_weight, state->active_experts, state->group_row_offset, state->route_source_token,
							state->expert_act, rlim);
					else
						(void)LM_LAUNCH_DEPENDENT((LingMoeExactW1W3SiluKernel<G, LmFp8, THREADS, 1u>),
							dim3((G::kMoeInter + GROUPS - 1u) / GROUPS, max_active), THREADS, 0, stream,
							layer->moe_w1.weight, layer->moe_w1.scale, layer->moe_w3.weight, layer->moe_w3.scale, normed,
							state->packed_weight, state->active_experts, state->group_row_offset, state->route_source_token,
							state->expert_act, rlim);
				}
				else
					(void)LM_LAUNCH_DEPENDENT((LingMoeExactW1W3SiluKernel<G, LmFp8, THREADS, 1u>),
						dim3((G::kMoeInter + GROUPS - 1u) / GROUPS, max_active), THREADS, 0, stream,
						layer->moe_w1.weight, layer->moe_w1.scale, layer->moe_w3.weight, layer->moe_w3.scale, normed,
						state->packed_weight, state->active_experts, state->group_row_offset, state->route_source_token,
						state->expert_act, rlim);
			}
		}
		else
		{
			constexpr uint32_t CHUNKS = G::kHidden / 8u;
			constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t COLS = (THREADS / LANES) * BF16_CPG;
			(void)LM_LAUNCH_DEPENDENT((LingMoeExactW1W3SiluKernel<G, LmBf16Format, THREADS, BF16_CPG>),
				dim3((G::kMoeInter + COLS - 1u) / COLS, max_active), THREADS, 0, stream,
				layer->moe_w1.weight, (const float *)0, layer->moe_w3.weight, (const float *)0, normed,
				state->packed_weight, state->active_experts, state->group_row_offset, state->route_source_token,
				state->expert_act, rlim);
		}
	}

	static void LaunchExactW2(State *state, const LingLayer *layer, uint32_t routes, SparkGpuStream stream)
	{
		const uint32_t max_active = routes < G::kLocalExperts ? routes : G::kLocalExperts;
		constexpr uint32_t BF16_CPG = G::kMoeInter <= 128u ? 1u : 2u;
		if ( layer->moe_w2.scale != 0 )
		{
			if constexpr ( (G::kMoeInter % LG_FP8_BLOCK) == 0u )
			{
				constexpr uint32_t CHUNKS = G::kMoeInter / 16u;
				constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t GROUPS = LG_THREADS / LANES;
				if constexpr ( G::kMoeInter > 128u )
				{
					if ( routes <= 256u )
						(void)LM_LAUNCH_DEPENDENT((LingMoeExactW2Kernel<G, LmFp8, 2u>),
							dim3((G::kHidden + 2u * GROUPS - 1u) / (2u * GROUPS), max_active), LG_THREADS, 0, stream,
							layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->active_experts,
							state->group_row_offset, state->expert_out);
					else
						(void)LM_LAUNCH_DEPENDENT((LingMoeExactW2Kernel<G, LmFp8, 1u>),
							dim3((G::kHidden + GROUPS - 1u) / GROUPS, max_active), LG_THREADS, 0, stream,
							layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->active_experts,
							state->group_row_offset, state->expert_out);
				}
				else
					(void)LM_LAUNCH_DEPENDENT((LingMoeExactW2Kernel<G, LmFp8, 1u>),
						dim3((G::kHidden + GROUPS - 1u) / GROUPS, max_active), LG_THREADS, 0, stream,
						layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->active_experts,
						state->group_row_offset, state->expert_out);
			}
		}
		else
		{
			constexpr uint32_t CHUNKS = G::kMoeInter / 8u;
			constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t COLS = (LG_THREADS / LANES) * BF16_CPG;
			(void)LM_LAUNCH_DEPENDENT((LingMoeExactW2Kernel<G, LmBf16Format, BF16_CPG>),
				dim3((G::kHidden + COLS - 1u) / COLS, max_active), LG_THREADS, 0, stream,
				layer->moe_w2.weight, (const float *)0, state->expert_act, state->active_experts,
				state->group_row_offset, state->expert_out);
		}
	}

	static bool LaunchMmaW1W3(State *state, const LingLayer *layer, const uint16_t *normed, uint32_t expected_local, uint32_t sms, float rlim, SparkGpuStream stream)
	{
		const uint32_t mean = (expected_local + G::kLocalExperts - 1u) / G::kLocalExperts;
		const uint32_t tile_m_64 = mean >= 22u ? 64u : (mean >= 10u ? 32u : 16u);
		const uint32_t tile_m_128 = mean >= 50u ? 128u : tile_m_64;
		uint32_t blocks = sms;
		if ( layer->moe_w1.scale != 0 )
		{
			if constexpr ( (G::kHidden % 128u) == 0u && (G::kMoeInter % 64u) == 0u && G::kMoeInter <= 768u )
			{
				LM_LAUNCH_DEPENDENT((LingBuildTilePrefixKernel), 1u, 32u, 0, stream,
					state->group_row_offset, state->tile_prefix_up, G::kLocalExperts, tile_m_64, G::kMoeInter / 64u);
				if ( tile_m_64 == 64u )
				{
					constexpr uint32_t sh = 3u * (64u * 256u + 2u * 64u * 128u);
					if ( LingMoeMmaSetup<G, 1u>(LingMoeMmaW1W3SiluFp8Kernel<G, 64u, 4u, 3u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
						return false;
					LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluFp8Kernel<G, 64u, 4u, 3u>), blocks, 256u, sh, stream,
						(const uint8_t *)layer->moe_w1.weight, layer->moe_w1.scale, (const uint8_t *)layer->moe_w3.weight, layer->moe_w3.scale,
						normed, state->packed_weight, state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
				}
				else if ( tile_m_64 == 32u )
				{
					constexpr uint32_t sh = 4u * (32u * 256u + 2u * 64u * 128u);
					if ( LingMoeMmaSetup<G, 2u>(LingMoeMmaW1W3SiluFp8Kernel<G, 32u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
						return false;
					LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluFp8Kernel<G, 32u, 2u, 4u>), blocks, 256u, sh, stream,
						(const uint8_t *)layer->moe_w1.weight, layer->moe_w1.scale, (const uint8_t *)layer->moe_w3.weight, layer->moe_w3.scale,
						normed, state->packed_weight, state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
				}
				else
				{
					constexpr uint32_t sh = 4u * (16u * 256u + 2u * 64u * 128u);
					if ( LingMoeMmaSetup<G, 3u>(LingMoeMmaW1W3SiluFp8Kernel<G, 16u, 1u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
						return false;
					LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluFp8Kernel<G, 16u, 1u, 4u>), blocks, 256u, sh, stream,
						(const uint8_t *)layer->moe_w1.weight, layer->moe_w1.scale, (const uint8_t *)layer->moe_w3.weight, layer->moe_w3.scale,
						normed, state->packed_weight, state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
				}
				return true;
			}
			return false;
		}
		if constexpr ( G::kMoeInter == 80u && (G::kHidden % 64u) == 0u )
		{
			LM_LAUNCH_DEPENDENT((LingBuildTilePrefixKernel), 1u, 32u, 0, stream,
				state->group_row_offset, state->tile_prefix_up, G::kLocalExperts, tile_m_64, 1u);
			if ( tile_m_64 == 64u )
			{
				constexpr uint32_t sh = 2u * (64u * 128u + 2u * 80u * 128u);
				if ( LingMoeMmaSetup<G, 11u>(LingMoeMmaW1W3SiluN80Kernel<G, 64u, 2u, 2u>, 320u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluN80Kernel<G, 64u, 2u, 2u>), blocks, 320u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			else if ( tile_m_64 == 32u )
			{
				constexpr uint32_t sh = 2u * (32u * 128u + 2u * 80u * 128u);
				if ( LingMoeMmaSetup<G, 12u>(LingMoeMmaW1W3SiluN80Kernel<G, 32u, 1u, 2u>, 160u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluN80Kernel<G, 32u, 1u, 2u>), blocks, 160u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			else
			{
				constexpr uint32_t sh = 2u * (16u * 128u + 2u * 80u * 128u);
				if ( LingMoeMmaSetup<G, 13u>(LingMoeMmaW1W3SiluN80Kernel<G, 16u, 1u, 2u>, 160u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluN80Kernel<G, 16u, 1u, 2u>), blocks, 160u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			return true;
		}
		else if constexpr ( (G::kHidden % 64u) == 0u && (G::kMoeInter % 64u) == 0u )
		{
			LM_LAUNCH_DEPENDENT((LingBuildTilePrefixKernel), 1u, 32u, 0, stream,
				state->group_row_offset, state->tile_prefix_up, G::kLocalExperts, tile_m_128, G::kMoeInter / 64u);
			if ( tile_m_128 == 128u )
			{
				constexpr uint32_t sh = 3u * (128u * 128u + 2u * 64u * 128u);
				if ( LingMoeMmaSetup<G, 24u>(LingMoeMmaW1W3SiluBf16Kernel<G, 128u, 4u, 3u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluBf16Kernel<G, 128u, 4u, 3u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			else if ( tile_m_128 == 64u )
			{
				constexpr uint32_t sh = 4u * (64u * 128u + 2u * 64u * 128u);
				if ( LingMoeMmaSetup<G, 21u>(LingMoeMmaW1W3SiluBf16Kernel<G, 64u, 4u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluBf16Kernel<G, 64u, 4u, 4u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			else if ( tile_m_128 == 32u )
			{
				constexpr uint32_t sh = 4u * (32u * 128u + 2u * 64u * 128u);
				if ( LingMoeMmaSetup<G, 22u>(LingMoeMmaW1W3SiluBf16Kernel<G, 32u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluBf16Kernel<G, 32u, 2u, 4u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			else
			{
				constexpr uint32_t sh = 4u * (16u * 128u + 2u * 64u * 128u);
				if ( LingMoeMmaSetup<G, 23u>(LingMoeMmaW1W3SiluBf16Kernel<G, 16u, 1u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW1W3SiluBf16Kernel<G, 16u, 1u, 4u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w1.weight, (const uint16_t *)layer->moe_w3.weight, normed, state->packed_weight,
					state->group_row_offset, state->tile_prefix_up, state->route_source_token, state->expert_act, rlim);
			}
			return true;
		}
		return false;
	}

	static bool LaunchMmaW2Bf16(State *state, const LingLayer *layer, uint32_t expected_local, uint32_t sms, SparkGpuStream stream)
	{
		if constexpr ( (G::kMoeInter % 64u) == 0u && (G::kHidden % 128u) == 0u )
		{
			const uint32_t mean = (expected_local + G::kLocalExperts - 1u) / G::kLocalExperts;
			const uint32_t tile_m = mean >= 50u ? 128u : mean >= 22u ? 64u : mean >= 10u ? 32u : 16u;
			uint32_t blocks = sms;
			LM_LAUNCH_DEPENDENT((LingBuildTilePrefixKernel), 1u, 32u, 0, stream,
				state->group_row_offset, state->tile_prefix_down, G::kLocalExperts, tile_m, G::kHidden / 128u);
			if ( tile_m == 128u )
			{
				constexpr uint32_t sh = 3u * (128u + 128u) * 128u;
				if ( LingMoeMmaSetup<G, 44u>(LingMoeMmaW2Bf16Kernel<G, 128u, 128u, 4u, 3u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW2Bf16Kernel<G, 128u, 128u, 4u, 3u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down, state->expert_out);
			}
			else if ( tile_m == 64u )
			{
				constexpr uint32_t sh = 4u * (64u + 128u) * 128u;
				if ( LingMoeMmaSetup<G, 43u>(LingMoeMmaW2Bf16Kernel<G, 64u, 128u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW2Bf16Kernel<G, 64u, 128u, 2u, 4u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down, state->expert_out);
			}
			else if ( tile_m == 32u )
			{
				constexpr uint32_t sh = 4u * (32u + 128u) * 128u;
				if ( LingMoeMmaSetup<G, 42u>(LingMoeMmaW2Bf16Kernel<G, 32u, 128u, 2u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW2Bf16Kernel<G, 32u, 128u, 2u, 4u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down, state->expert_out);
			}
			else
			{
				constexpr uint32_t sh = 4u * (16u + 128u) * 128u;
				if ( LingMoeMmaSetup<G, 41u>(LingMoeMmaW2Bf16Kernel<G, 16u, 128u, 1u, 4u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
					return false;
				LM_LAUNCH_DEPENDENT((LingMoeMmaW2Bf16Kernel<G, 16u, 128u, 1u, 4u>), blocks, 256u, sh, stream,
					(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down, state->expert_out);
			}
			return true;
		}
		return false;
	}

	static bool LaunchSolarW2AndFinalize(State *state, const LingLayer *layer, uint32_t rows, uint32_t routes, uint32_t sms, SparkGpuStream stream)
	{
		if constexpr ( G::kMoeInter == 80u && (G::kHidden % 64u) == 0u )
		{
			const uint32_t mean = (routes + G::kLocalExperts - 1u) / G::kLocalExperts;
			const uint32_t tile_m = mean > 32u ? 64u : (mean > 16u ? 32u : 16u);
			const uint64_t full_bytes = (uint64_t)routes * G::kHidden * sizeof(uint16_t);
			uint32_t slice_cols = G::kHidden;
			if ( full_bytes > 16u * 1024u * 1024u )
			{
				const uint64_t target_bytes = 10u * 1024u * 1024u;
				uint32_t raw_cols = (uint32_t)(target_bytes / ((uint64_t)routes * sizeof(uint16_t)));
				raw_cols = (raw_cols / 256u) * 256u;
				if ( raw_cols < 256u )
					raw_cols = 256u;
				if ( raw_cols > G::kHidden )
					raw_cols = G::kHidden;
				slice_cols = raw_cols;
			}
			uint32_t blocks = sms;
			uint32_t prev_cur_cols = 0u;
			for ( uint32_t col_begin = 0u; col_begin < G::kHidden; col_begin += slice_cols )
			{
				const uint32_t cur_cols = G::kHidden - col_begin < slice_cols ? G::kHidden - col_begin : slice_cols;
				if ( cur_cols != prev_cur_cols )
				{
					LM_LAUNCH_DEPENDENT((LingBuildTilePrefixKernel), 1u, 32u, 0, stream,
						state->group_row_offset, state->tile_prefix_down, G::kLocalExperts, tile_m, cur_cols / 64u);
					prev_cur_cols = cur_cols;
				}
				if ( tile_m == 64u )
				{
					constexpr uint32_t sh = 2u * ((64u + 64u) * 160u);
					if ( LingMoeMmaSetup<G, 31u>(LingMoeMmaW2K80SliceKernel<G, 64u, 4u, 2u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
						return false;
					LM_LAUNCH_DEPENDENT((LingMoeMmaW2K80SliceKernel<G, 64u, 4u, 2u>), blocks, 256u, sh, stream,
						(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down,
						state->expert_out, col_begin, cur_cols);
				}
				else if ( tile_m == 32u )
				{
					constexpr uint32_t sh = 2u * ((32u + 64u) * 160u);
					if ( LingMoeMmaSetup<G, 32u>(LingMoeMmaW2K80SliceKernel<G, 32u, 2u, 2u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
						return false;
					LM_LAUNCH_DEPENDENT((LingMoeMmaW2K80SliceKernel<G, 32u, 2u, 2u>), blocks, 256u, sh, stream,
						(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down,
						state->expert_out, col_begin, cur_cols);
				}
				else
				{
					constexpr uint32_t sh = 2u * ((16u + 64u) * 160u);
					if ( LingMoeMmaSetup<G, 33u>(LingMoeMmaW2K80SliceKernel<G, 16u, 1u, 2u>, 256u, sh, sms, &blocks) != LM_LAUNCH_OK )
						return false;
					LM_LAUNCH_DEPENDENT((LingMoeMmaW2K80SliceKernel<G, 16u, 1u, 2u>), blocks, 256u, sh, stream,
						(const uint16_t *)layer->moe_w2.weight, state->expert_act, state->group_row_offset, state->tile_prefix_down,
						state->expert_out, col_begin, cur_cols);
				}
				(void)LM_LAUNCH_DEPENDENT((LingMoeFinalizeWarpSliceKernel<G::kExpertWeightAfterDown, G::kTopK>),
					(rows + 3u) / 4u, 128u, 0, stream, state->expert_out, state->route_packed_row, state->route_expert,
					state->route_weight, state->shared_out, state->partial, rows, G::kLocalExperts, col_begin, cur_cols, G::kHidden);
			}
			return true;
		}
		return false;
	}

	static SparkStatus Experts(State *state, const Frame &frame, const LingLayer &entry, uint32_t index)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, routes = rows * G::kTopK, exact = step->commit == 0u ? 1u : 0u;
		const uint16_t *normed = state->normed;
		const float routed_limit = G::RoutedLimit(index), shared_limit = G::SharedLimit(index);
		const float rlim = routed_limit > 0.0f ? routed_limit : INFINITY;
		const float slim = shared_limit > 0.0f ? shared_limit : INFINITY;
		const LmSkinnyDenseTarget targets[2] =
		{
			{ layer->router, (uint16_t *)0, state->router_logits, G::kExperts, 0u, 0u, LmScaleTensorNone() },
			{ layer->shared_gate_up, state->shared_gate_up, (float *)0, 2u * G::kSharedRows, 0u, 0u, LmScaleTensorNone() }
		};
		if ( LinearMulti(targets, 2u, normed, rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( rows == 1u )
		{
			(void)LM_LAUNCH_DEPENDENT((LingGroupRouteRow1FastKernel<G>), 1u, LG_THREADS, 0, stream, state->router_logits, layer->expert_bias,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, state->active_experts);
		}
		else if ( routes <= 32u )
		{
			(void)LM_LAUNCH_DEPENDENT((LingGroupRouteSmallFastKernel<G>), 1u, LG_THREADS, 0, stream, state->router_logits, layer->expert_bias, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, state->active_experts);
		}
		else
		{
			constexpr uint32_t WARPS = LG_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((LingGroupTopkFastKernel<G>), (rows + WARPS - 1u) / WARPS, LG_THREADS, 0, stream, state->router_logits,
				layer->expert_bias, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			(void)LM_LAUNCH_DEPENDENT((LingRouteBuildFastKernel<G::kLocalExperts>), 1u, LG_THREADS, 0, stream,
				state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row,
				state->route_source_token, state->packed_weight, state->route_global, state->active_experts);
		}
		const bool prefill_unified = PrefillUnified(step);
		constexpr uint32_t kFuseSharedRows = G::kLocalExperts < G::kExperts ? 8u : 4u;
		if ( !prefill_unified && rows <= kFuseSharedRows && (layer->moe_w1.scale == 0) == (layer->moe_w3.scale == 0) &&
			(layer->moe_w1.scale == 0) == (layer->moe_w2.scale == 0) )
		{
			if ( layer->moe_w1.scale != 0 )
			{
				if constexpr ( (G::kHidden % LG_FP8_BLOCK) == 0u && (G::kMoeInter % LG_FP8_BLOCK) == 0u )
				{
					constexpr uint32_t DN_CHUNKS = G::kMoeInter / 16u;
					constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
					constexpr uint32_t DN_GROUPS = LG_THREADS / DN_LANES;
					LaunchExactW1W3(state, layer, normed, routes, rlim, stream);
					(void)LM_LAUNCH_DEPENDENT((LingMoeSmallW2FinalizeKernel<G, LmFp8>),
						dim3((G::kHidden + 2u * DN_GROUPS - 1u) / (2u * DN_GROUPS), rows), LG_THREADS, 0, stream,
						layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->route_packed_row, state->route_expert,
						state->route_weight, state->shared_gate_up, layer->shared_down, state->partial, slim);
					return CudaStageLaunched();
				}
			}
			else
			{
				constexpr uint32_t DN_CHUNKS = G::kMoeInter / 8u;
				constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t DN_GROUPS = LG_THREADS / DN_LANES;
				if constexpr ( G::kLocalExperts == G::kExperts )
				{
					constexpr uint32_t UP_CHUNKS = G::kHidden / 8u;
					constexpr uint32_t UP_LANES = UP_CHUNKS >= 32u ? 32u : (UP_CHUNKS >= 16u ? 16u : 8u);
					constexpr uint32_t UP_GROUPS = LG_THREADS / UP_LANES;
					(void)LM_LAUNCH_DEPENDENT((LingMoePackedW1W3SiluKernel<G, LmBf16Format>),
						dim3((G::kMoeInter + 2u * UP_GROUPS - 1u) / (2u * UP_GROUPS), routes), LG_THREADS, 0, stream,
						layer->moe_w1.weight, (const float *)0, layer->moe_w3.weight, (const float *)0, normed, state->packed_weight,
						state->route_global, state->route_source_token, state->group_row_offset, state->expert_act, rlim);
				}
				else
					LaunchExactW1W3(state, layer, normed, routes, rlim, stream);
				(void)LM_LAUNCH_DEPENDENT((LingMoeSmallW2FinalizeKernel<G, LmBf16Format>),
					dim3((G::kHidden + 2u * DN_GROUPS - 1u) / (2u * DN_GROUPS), rows), LG_THREADS, 0, stream,
					layer->moe_w2.weight, (const float *)0, state->expert_act, state->route_packed_row, state->route_expert,
					state->route_weight, state->shared_gate_up, layer->shared_down, state->partial, slim);
				return CudaStageLaunched();
			}
		}
		if constexpr ( G::kSharedRows == 64u && (G::kHidden % 64u) == 0u )
		{
			if ( prefill_unified )
			{
				(void)LM_LAUNCH_DEPENDENT((LingSharedActDownMmaKernel<G>), dim3(G::kHidden / 64u, (rows + 63u) / 64u), 128u, 0, stream,
					state->shared_gate_up, layer->shared_down, state->shared_out, rows, slim);
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				(void)LM_LAUNCH_DEPENDENT((LingSiluMulLimitKernel<G::kFfnSiluRounded>), rows, LG_THREADS, 0, stream, state->shared_gate_up,
					state->shared_act, G::kSharedRows, slim);
				if ( CudaStageLaunched() != SPARK_STATUS_OK ||
					Linear(layer->shared_down, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else
		{
			(void)LM_LAUNCH_DEPENDENT((LingSiluMulLimitKernel<G::kFfnSiluRounded>), rows, LG_THREADS, 0, stream, state->shared_gate_up,
				state->shared_act, G::kSharedRows, slim);
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				Linear(layer->shared_down, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
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
		const uint32_t expected_local = (uint32_t)(((uint64_t)routes * G::kLocalExperts + G::kExperts - 1u) / G::kExperts);
		const bool use_exact_moe = !prefill_unified && (exact != 0u || (G::kLocalExperts == G::kExperts ? (rows <= 64u) : (expected_local <= G::kLocalExperts)));
		const bool use_solar_mma = exact == 0u && G::kMoeInter == 80u && (prefill_unified || rows >= 256u);
		bool fused_up_done = false;
		if ( (layer->moe_w1.scale == 0) == (layer->moe_w3.scale == 0) )
		{
			if ( !use_exact_moe && (G::kMoeInter != 80u || use_solar_mma) )
				fused_up_done = LaunchMmaW1W3(state, layer, normed, expected_local, step->multiprocessors, rlim, stream);
			if ( !fused_up_done && use_exact_moe &&
				(layer->moe_w1.scale == 0 || (G::kHidden % LG_FP8_BLOCK) == 0u) )
			{
				LaunchExactW1W3(state, layer, normed, routes, rlim, stream);
				fused_up_done = true;
			}
		}
		if ( !fused_up_done )
		{
			if ( ExpertUp(expert_routes, layer->moe_w1, normed, state->expert_gate, G::kHidden, G::kMoeInter, exact, stream, prefill_unified) != LM_LAUNCH_OK ||
				ExpertUp(expert_routes, layer->moe_w3, normed, state->expert_up, G::kHidden, G::kMoeInter, exact, stream, prefill_unified) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((LingSiluClampPackedKernel<G::kFfnSiluRounded, !G::kExpertWeightAfterDown>),
				dim3(LmVecBlocks<LG_THREADS>(G::kMoeInter), routes), LG_THREADS, 0, stream, state->expert_gate, state->expert_up,
				state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter, rlim);
		}
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( use_solar_mma && layer->moe_w2.scale == 0 &&
			LaunchSolarW2AndFinalize(state, layer, rows, routes, step->multiprocessors, stream) )
			return CudaStageLaunched();
		bool mma_w2_done = false;
		if ( !use_exact_moe && layer->moe_w2.scale == 0 )
			mma_w2_done = LaunchMmaW2Bf16(state, layer, expected_local, step->multiprocessors, stream);
		if ( mma_w2_done )
		{
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( use_exact_moe && (layer->moe_w2.scale == 0 || (G::kMoeInter % LG_FP8_BLOCK) == 0u) )
		{
			LaunchExactW2(state, layer, routes, stream);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( ExpertDown(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, G::kMoeInter, G::kHidden, exact, stream, prefill_unified) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( rows >= 32u && G::kTopK <= 32u )
			(void)LM_LAUNCH_DEPENDENT((LingMoeFinalizeWarpSliceKernel<G::kExpertWeightAfterDown, G::kTopK>),
				(rows + 3u) / 4u, 128u, 0, stream, state->expert_out, state->route_packed_row, state->route_expert,
				state->route_weight, state->shared_out, state->partial, rows, G::kLocalExperts, 0u, G::kHidden, G::kHidden);
		else
			(void)LM_LAUNCH_DEPENDENT((LingMoeLocalFinalizeKernel<G::kExpertWeightAfterDown>), dim3(LmVecBlocks<LG_THREADS>(G::kHidden), rows),
				LG_THREADS, 0, stream, state->expert_out, state->route_packed_row, state->route_expert, state->route_weight, state->shared_out,
				state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( step->tree != 0u && (G::kVerifyRows == 0u || step->commit != 0u || step->sequences != 1u || step->slots == 0 ||
			step->fork_positions == 0 || step->slot_offsets == 0 || step->parent_rows == 0 || step->state_loads == 0 ||
			step->state_saves == 0 || state->kda_snapshots == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		frame->step = step;
		frame->stream = (SparkGpuStream)stream;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((LingRmsNormKernel<G::kHidden, G::kNormRoundFirst, false>), frame.step->rows, LG_THREADS, 0, frame.stream,
			frame.step->hidden_bf16, (const uint16_t *)0, weight, state->normed, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, const float *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((LingRmsNormKernel<G::kHidden, G::kNormRoundFirst, true>), frame.step->rows, LG_THREADS, 0, frame.stream,
			frame.step->hidden_bf16, state->partial, weight, state->normed, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus AddRows(State *state, const Frame &frame)
	{
		(void)LM_LAUNCH_DEPENDENT((LingAddRowsKernel<G::kHidden>), dim3(LmVecBlocks<LG_THREADS>(G::kHidden), frame.step->rows), LG_THREADS, 0,
			frame.stream, frame.step->hidden_bf16, state->partial);
		return SPARK_STATUS_OK;
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		const uint32_t rows = step->rows, sequences = step->sequences;
		if ( G::kVerifyRows == 0u || kept == 0 || rows == 0u || rows > G::kVerifyRows || sequences == 0u || sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			const LingLayer *layer = &state->layers[index];
			if ( index % G::kPeriod == G::kPhase )
				continue;
			uint8_t *state_pool = state->kda_state + (uint64_t)layer->recurrent_index * state->slots * state->kda_state_bytes;
			uint16_t *window = state->kda_window + (uint64_t)layer->recurrent_index * state->slots * (state->kda_window_bytes / sizeof(uint16_t));
			if ( rows <= LM_CAUSAL_CONV_SERIAL_ROWS * sequences )
				(void)LM_LAUNCH_DEPENDENT((LingKdaConvSplitSerialKernel<G>), dim3(sequences, (S::kKdaChannels + LG_THREADS - 1u) / LG_THREADS),
					LG_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, kept, ReplayQkv(state, layer), layer->kda_conv,
					state->query, state->key, state->value, sequences, 1u, step->sequence_row_indices, (const uint32_t *)0);
			else
				(void)LM_LAUNCH_DEPENDENT((LingKdaConvSplitTiledKernel<G, false>),
					dim3(sequences, (S::kKdaChannels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
					LM_CAUSAL_CONV_TILE_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, kept, ReplayQkv(state, layer),
					layer->kda_conv, state->query, state->key, state->value, (uint16_t *)0, sequences, 1u, step->sequence_row_indices);
			LM_LAUNCH((LmDeltaRuleColumnKernel<LG_THREADS,G::kKdaDim,G::kKdaDim,float>), dim3(sequences,S::kKdaRankHeads,G::kKdaDim / LM_WARP_LANES),
				LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, (uint32_t)state->kda_state_bytes, step->recurrent_index, step->sequence_row_begin,
				kept, state->query, state->key, state->value, ReplayRetention(state, layer), ReplayWriteGate(state, layer), state->kda_mixed,
				S::kKdaRankHeads, 1u, sequences, 1u, step->sequence_row_indices, 0, 0, 0);
		}
		if ( step->tree != 0u )
		{
			if ( sequences != 1u || step->slots == 0 || step->sequence_of_row == 0 || step->sequence_row_indices == 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			for ( uint32_t index = 0u; index < G::kLayers; ++index )
			{
				const LingLayer *layer = &state->layers[index];
				if ( index % G::kPeriod != G::kPhase )
					continue;
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
				LM_LAUNCH((LmKvShardPathMoveKernel<typename S::Kv,LG_THREADS>), G::kVerifyRows, LG_THREADS, 0, stream, state->kv_views[layer->kv_index],
					step->sequence_of_row, step->sequence_row_indices, kept, step->slots, S::kSlot);
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			}
		}
		return CudaStageLaunched();
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, slots = lane == 0u ? state->slots : 0u, routes = rows * G::kTopK, groups = G::kLocalExperts + 2u;
		const uint64_t degree = state->tp_degree, verify_rows = lane == 0u ? G::kVerifyRows : 0u;
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t);
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->qkv, rows * S::kKdaChannels * sizeof(uint16_t) },
			{ (void **)&state->kda_low_f, rows * G::kKdaRank * sizeof(uint16_t) },
			{ (void **)&state->kda_low_g, rows * G::kKdaRank * sizeof(uint16_t) },
			{ (void **)&state->decay_logit, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->kda_gate, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->beta, rows * S::kKdaRankHeads * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->key, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->retention, rows * S::kKdaWidth * sizeof(float) },
			{ (void **)&state->write_gate, rows * S::kKdaRankHeads * sizeof(float) },
			{ (void **)&state->kda_mixed, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->replay_qkv, (uint64_t)S::kKdaLayers * verify_rows * S::kKdaChannels * sizeof(uint16_t) },
			{ (void **)&state->replay_retention, (uint64_t)S::kKdaLayers * verify_rows * S::kKdaWidth * sizeof(float) },
			{ (void **)&state->replay_write_gate, (uint64_t)S::kKdaLayers * verify_rows * S::kKdaRankHeads * sizeof(float) },
			{ (void **)&state->kda_snapshots, verify_rows != 0u ? LmDeltaRuleSnapshotBytes(S::kKdaRankHeads, G::kKdaDim, G::kKdaDim) : 0u },
			{ (void **)&state->mla_q_resid, rows * G::kQLora * sizeof(uint16_t) },
			{ (void **)&state->mla_q, rows * S::kMlaRankHeads * S::kQkDim * sizeof(uint16_t) },
			{ (void **)&state->mla_latent_query, rows * S::kMlaRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->mla_query, rows * S::kMlaRankHeads * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->mla_kv, rows * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->mla_gathered, (uint64_t)state->tp_degree * rows * S::kMlaRankHeads * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->attn_send, (uint64_t)state->tp_degree * state->partial_stride * sizeof(float) },
			{ (void **)&state->attn_received, (uint64_t)state->tp_degree * state->partial_stride * sizeof(float) },
			{ (void **)&state->gqa_local, rows * S::kGqaRankFused * sizeof(uint16_t) },
			{ (void **)&state->gqa_gathered, degree * rows * S::kGqaRankFused * sizeof(uint16_t) },
			{ (void **)&state->gqa_full, rows * S::kGqaFused * sizeof(uint16_t) },
			{ (void **)&state->gqa_query, rows * S::kGqaWidth * sizeof(uint16_t) },
			{ (void **)&state->gqa_slot, S::kGqa ? rows * S::kSlot * sizeof(uint16_t) : 0u },
			{ (void **)&state->gqa_gate, rows * S::kGqaRankWidth * sizeof(uint16_t) },
			{ (void **)&state->gqa_merged, rows * S::kGqaRankWidth * sizeof(uint16_t) },
			{ (void **)&state->mla_latent, rows * S::kMlaRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->mla_value, rows * S::kMlaRankHeads * G::kValueDim * sizeof(uint16_t) },
			{ (void **)&state->mla_gate, rows * S::kMlaRankHeads * sizeof(uint16_t) },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->active_experts, (G::kLocalExperts + 1u) * sizeof(uint32_t) },
			{ (void **)&state->packed_weight, routes * sizeof(float) },
			{ (void **)&state->expert_gate, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->shared_gate_up, rows * 2u * G::kSharedRows * sizeof(uint16_t) },
			{ (void **)&state->shared_act, rows * G::kSharedRows * sizeof(uint16_t) },
			{ (void **)&state->shared_out, width },
			{ (void **)&state->kda_state, (uint64_t)S::kKdaLayers * slots * state->kda_state_bytes },
			{ (void **)&state->kda_window, (uint64_t)S::kKdaLayers * slots * state->kda_window_bytes },
			{ (void **)&state->gather_seq, sizeof(uint32_t) },
			{ (void **)&state->gather_offset, (uint64_t)state->slots * sizeof(uint32_t) },
			{ (void **)&state->gather_prior_ctx, (uint64_t)state->slots * sizeof(uint32_t) },
			{ (void **)&state->gather_total_ctx, (uint64_t)state->slots * sizeof(uint32_t) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		state->kda_state_bytes = (uint64_t)S::kKdaRankHeads * G::kKdaDim * G::kKdaDim * sizeof(float);
		state->kda_window_bytes = (uint64_t)S::kKdaChannels * G::kConv * sizeof(uint16_t);
		state->partial_stride = ((uint64_t)state->max_rows * (S::kGqa ? LmGqaShardRecordFloats(G::kGqaHeads, G::kGqaHeadDim, state->tp_degree) :
			S::kMlaRankHeads * S::kRecord) + record_unit - 1u) / record_unit * record_unit;
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration)
	{
		return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 ||
			configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 ? 1u : 0u;
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "vocab_rows", &vocab_rows },
			{ "ffn_rows", &state->ffn_rows }, { "experts", &experts }, { "local_experts", &local }, { "expert_first", &state->expert_first },
			{ "top_k", &top_k }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || (G::kDenseLayers != 0u && (state->ffn_rows == 0u || state->ffn_rows % 64u != 0u)) || experts != G::kExperts ||
			local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != (G::kLocalExperts == G::kExperts ? 0u : rank * G::kLocalExperts) )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, experts, local, state->expert_first, top_k, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( S::kGqa && LmGqaShardGeometryValid(G::kGqaHeads, G::kGqaKvHeads, G::kGqaHeadDim, state->tp_degree) == 0u )
		{
			fprintf(stderr, "%s: %u query heads over %u KV heads do not shard over %u ranks\n", G::Tag(), G::kGqaHeads, G::kGqaKvHeads, state->tp_degree);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = G::kVerifyRows != 0u ? 1u : 0u;
		geometry->head_norm_round_first = G::kNormRoundFirst ? 1u : 0u;
		if ( CudaStage<LingModel>::HeadRanges(state, geometry, false, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_rows * state->tp_degree != G::kVocab || geometry->head_rows * state->tp_degree != G::kVocab ||
			geometry->embed_bytes != (uint64_t)geometry->embed_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)geometry->head_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the stage geometry\n", G::Tag(),
				geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	template<class T>
	static void Tensor(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t bytes) { tensors->Add(name, kind, bytes, field); }

	static SparkStatus AddWeight(State *state, CudaStageTensors *tensors, uint32_t index, const char *name, LingWeight *field, uint64_t rows,
		uint32_t columns)
	{
		char full_name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		SparkNamedPackEntry entry;
		snprintf(full_name, sizeof(full_name), "layers.%u.%s", index, name);
		if ( SparkNamedPackEntryLoad(&state->pack, full_name, &entry) == SPARK_STATUS_OK && strcmp(entry.kind, "fp8_e4m3") == 0 )
		{
			if ( columns % LG_FP8_BLOCK != 0u )
			{
				fprintf(stderr, "%s: layer %u %s has %u columns, not whole %u-column fp8 blocks\n", G::Tag(), index, name, columns, LG_FP8_BLOCK);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
			tensors->Add(name, "fp8_e4m3", rows * columns, &field->weight);
			tensors->AddScaled(name, ".scale", "f32", rows * (columns / LG_FP8_BLOCK) * sizeof(float), &field->scale);
			return SPARK_STATUS_OK;
		}
		field->scale = 0;
		tensors->Add(name, "bf16", rows * columns * sizeof(uint16_t), &field->weight);
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		LingLayer *layer = &state->layers[index];
		SparkStatus status = SPARK_STATUS_OK;
		Tensor(tensors, "input_norm", "f32", &layer->input_norm, (uint64_t)G::kHidden * sizeof(float));
		Tensor(tensors, "post_norm", "f32", &layer->post_norm, (uint64_t)G::kHidden * sizeof(float));
		if ( index % G::kPeriod == G::kPhase )
		{
			if constexpr ( S::kGqa )
			{
				Tensor(tensors, "gqa_qkv", "bf16", &layer->gqa_qkv, (uint64_t)S::kGqaRankFused * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "gqa_gate", "bf16", &layer->gqa_gate, (uint64_t)S::kGqaRankWidth * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "gqa_out", "bf16", &layer->gqa_out, (uint64_t)G::kHidden * S::kGqaRankWidth * sizeof(uint16_t));
			}
			else
			{
				if constexpr ( G::kQLora != 0u )
				{
					if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "mla_q_a", &layer->mla_q_a, G::kQLora, G::kHidden);
					Tensor(tensors, "mla_q_norm", "f32", &layer->mla_q_norm, (uint64_t)G::kQLora * sizeof(float));
					if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "mla_q_b", &layer->mla_q_b, (uint64_t)S::kMlaRankHeads * S::kQkDim, G::kQLora);
				}
				else if ( status == SPARK_STATUS_OK )
					status = AddWeight(state, tensors, index, "mla_q", &layer->mla_q, (uint64_t)S::kMlaRankHeads * S::kQkDim, G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "mla_kv_a", &layer->mla_kv_a, S::kSlot, G::kHidden);
				Tensor(tensors, "mla_kv_norm", "f32", &layer->mla_kv_norm, (uint64_t)G::kLatent * sizeof(float));
				Tensor(tensors, "mla_uk", "bf16", &layer->mla_uk, (uint64_t)S::kMlaRankHeads * G::kLatent * G::kNope * sizeof(uint16_t));
				Tensor(tensors, "mla_uv", "bf16", &layer->mla_uv, (uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent * sizeof(uint16_t));
				Tensor(tensors, "mla_gate", "bf16", &layer->mla_gate, (uint64_t)S::kMlaRankHeads * G::kHidden * sizeof(uint16_t));
				if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "mla_out", &layer->mla_out, G::kHidden, S::kMlaRankHeads * G::kValueDim);
			}
		}
		else
		{
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "kda_qkv", &layer->kda_qkv, S::kKdaChannels, G::kHidden);
			if ( G::kKdaRank != 0u && status == SPARK_STATUS_OK )
				status = AddWeight(state, tensors, index, "kda_f_a", &layer->kda_f_a, G::kKdaRank, G::kHidden);
			if ( status == SPARK_STATUS_OK )
				status = AddWeight(state, tensors, index, "kda_f", &layer->kda_f, S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
			if ( G::kKdaRank != 0u && status == SPARK_STATUS_OK )
				status = AddWeight(state, tensors, index, "kda_g_a", &layer->kda_g_a, G::kKdaRank, G::kHidden);
			if ( status == SPARK_STATUS_OK )
				status = AddWeight(state, tensors, index, "kda_g", &layer->kda_g, S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
			Tensor(tensors, "kda_beta", "bf16", &layer->kda_beta, (uint64_t)S::kKdaRankHeads * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "kda_conv", "bf16", &layer->kda_conv, (uint64_t)S::kKdaChannels * G::kConv * sizeof(uint16_t));
			Tensor(tensors, "kda_a_log", "f32", &layer->kda_a_log, (uint64_t)S::kKdaRankHeads * sizeof(float));
			Tensor(tensors, "kda_dt_bias", "f32", &layer->kda_dt_bias, (uint64_t)S::kKdaWidth * sizeof(float));
			Tensor(tensors, "kda_norm", "bf16", &layer->kda_norm, (uint64_t)G::kKdaDim * sizeof(uint16_t));
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "kda_out", &layer->kda_out, G::kHidden, S::kKdaWidth);
		}
		if ( index < G::kDenseLayers )
		{
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "ffn_gate_up", &layer->ffn_gate_up, (uint64_t)2u * state->ffn_rows, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "ffn_down", &layer->ffn_down, G::kHidden, state->ffn_rows);
		}
		else
		{
			Tensor(tensors, "router", "bf16", &layer->router, (uint64_t)G::kExperts * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "expert_bias", "f32", &layer->expert_bias, (uint64_t)G::kExperts * sizeof(float));
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "moe_w1", &layer->moe_w1, (uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "moe_w3", &layer->moe_w3, (uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddWeight(state, tensors, index, "moe_w2", &layer->moe_w2, (uint64_t)G::kLocalExperts * G::kHidden, G::kMoeInter);
			Tensor(tensors, "shared_gate_up", "bf16", &layer->shared_gate_up, (uint64_t)2u * G::kSharedRows * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "shared_down", "bf16", &layer->shared_down, (uint64_t)G::kSharedRows * G::kHidden * sizeof(uint16_t));
		}
		return status;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t attention = 0u, kda = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			if ( index % G::kPeriod == G::kPhase )
				state->layers[index].kv_index = attention++;
			else
				state->layers[index].recurrent_index = kda++;
		}
		return SPARK_STATUS_OK;
	}

	static uint32_t RecurrentParts(const State *state, CudaStageRecurrent *parts)
	{
		parts[0] = { state->kda_state, state->kda_state_bytes, S::kKdaLayers };
		parts[1] = { (uint8_t *)state->kda_window, state->kda_window_bytes, S::kKdaLayers };
		return 2u;
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
