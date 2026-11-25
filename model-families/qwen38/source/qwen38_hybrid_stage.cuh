#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/hyper_mix.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/ngram_hash.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_qwen38_family.h"
#include "sparkpipe/spark_gpu.h"

#define QH_THREADS 256u
#define QH_SCALE_K 128u

constexpr uint32_t QwenHybridGatherParts(uint32_t ranks, uint32_t width, uint32_t limit)
{
	uint32_t parts = ranks == 1u ? 1u : (width + limit - 1u) / limit;
	while ( width % parts != 0u )
		++parts;
	return parts;
}

template<class G>
struct QwenHybridShape : QwenHybridLayout<G>
{
	using Kv = LmKvHeads<16u, G::kKvHeads, G::kHeadDim, G::kPageSlots>;
	static constexpr uint32_t kGatherParts = QwenHybridGatherParts(G::kTpDegree, QwenHybridLayout<G>::kQkvWidth / G::kTpDegree, G::kHidden);
	static_assert(QwenHybridLayout<G>::kQkvWidth / G::kTpDegree % kGatherParts == 0u &&
		(G::kTpDegree == 1u || QwenHybridLayout<G>::kQkvWidth / G::kTpDegree / kGatherParts <= G::kHidden),
		"a rank's fused attention rows gather as whole collective rows of at most the hidden width");
	using IndexKv = LmKvGeometry<QwenHybridLayout<G>::kIndexer ? G::kIndexDim * 2u : 16u, G::kPageSlots / G::kIndexRatio, true>;
	static_assert(!G::kMoe || LM_SKINNY_ROWS_WIDE * G::kTopK <= LM_SKINNY_ROWS * G::kLocalExperts,
		"a 16-row verify chunk keeps its routes on the grouped skinny expert variants whose per-route sums match one-row decode");
};

typedef struct QwenHybridLayer
{
	const uint8_t *mix_weight, *out_weight, *ffn_gate_up, *ffn_down;
	const uint8_t *moe_w1, *moe_w1_scale, *moe_w3, *moe_w3_scale, *moe_w2, *moe_w2_scale;
	const float *mix_scale, *out_scale, *ffn_gate_up_scale, *ffn_down_scale;
	const float *gdn_a_log, *gdn_dt_bias, *q_norm, *k_norm, *input_norm, *post_norm;
	const float *hc_norm[2], *index_q_norm, *index_k_norm;
	const uint16_t *gdn_ba, *gdn_conv, *gdn_norm, *moe_router, *moe_shared_gate_up, *moe_shared_down, *moe_shared_gate;
	const uint16_t *hc_down[2], *hc_up[2], *hc_inject[2], *index_qk;
	uint32_t recurrent_index, kv_index;
} QwenHybridLayer;

template<class G>
struct QwenHybridState : CudaStagePackState
{
	const char *linear_kind;
	uint32_t linear_bytes, ffn_rows, expert_first, gdn_heads, gdn_qk, gdn_v, gdn_channels, attn_rows, attn_width;
	uint64_t gdn_state_bytes, gdn_window_bytes, record_stride, tap_stride, ple_window_bytes;
	QwenHybridLayer layers[G::kLayers];
	FILE *dump;
	int16_t tap_output[G::kLayers];
	uint16_t *dump_rows, *tap_slab, *normed, *partial, *gate_up, *intermediate;
	uint16_t *qkv, *z, *ba, *query, *key, *value, *query_heads, *key_heads, *mixed;
	uint16_t *attn_local, *attn_gathered, *attn_full, *attn_query, *attn_gate, *attn_key, *attn_value, *attn_slot, *merged, *merged_gate;
	uint16_t *router_logits, *expert_gate, *expert_up, *expert_act, *expert_out, *shared_gate_up, *shared_act, *shared_out, *shared_logit;
	uint16_t *gdn_window, *replay_qkv, *replay_ba, *replay_ple, *replay_index, *residual, *hc_normed, *hc_low, *hc_up;
	uint16_t *ple_embed, *ple_key, *ple_query, *ple_value, *ple_gated, *ple_conv_in, *ple_conv_out, *ple_window;
	uint16_t *index_qk, *index_query, *index_key, *index_block, *index_tail;
	float *retention, *write_gate, *send, *received, *router_probs, *route_weight, *packed_weight, *gdn_snapshots, *hc_inject;
	float *index_scores;
	uint32_t *dump_positions, *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *packed_expert, *tile_prefix_up, *tile_prefix_down;
	uint32_t *row_lane, *row_ordinal, *index_local_selected, *index_mask;
	uint8_t *gdn_state;
	const float *final_hc_norm, *ple_norm_key, *ple_norm_query, *ple_norm_conv, *ple_table_scale;
	const uint16_t *final_hc_down, *final_hc_up, *ple_key_weight, *ple_value_weight, *ple_conv_weight;
	const uint8_t *ple_table;
	const int64_t *ple_multipliers, *ple_vocab_sizes, *ple_offsets;
	int64_t *ple_ids;
	int32_t *ple_history;
	uint32_t ple_table_first, ple_table_rows, max_positions, index_local_stride, index_global_stride, index_mask_words, hc_pending_inject;
	uint2 *index_candidates, *index_gathered;
	LmKvShardView index_views[QwenHybridShape<G>::kAttnLayers], kv_views[QwenHybridShape<G>::kAttnLayers];
};

static __device__ __forceinline__ float QwenWarpSum(float value)
{
	#pragma unroll
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu, value, offset);
	return __shfl_sync(0xffffffffu, value, 0u);
}

template<class G>
static __device__ __forceinline__ uint4 QwenAttnLoadVec8(const uint4 *__restrict__ gathered4, uint32_t rows, uint32_t row, uint32_t attn_rows_v, uint32_t logical_v)
{
	using S = QwenHybridShape<G>;
	if constexpr ( G::kRankHeads )
	{
		constexpr uint32_t q2_v = (2u * S::kQWidth) >> 3u, kv_v = S::kKvWidth >> 3u;
		constexpr uint32_t q_rank_v = q2_v / G::kTpDegree, kv_rank_v = kv_v / G::kTpDegree;
		uint32_t rank, elem;
		if ( logical_v < q2_v )
		{
			rank = logical_v / q_rank_v;
			elem = logical_v % q_rank_v;
		}
		else if ( logical_v < q2_v + kv_v )
		{
			const uint32_t k_v = logical_v - q2_v;
			rank = k_v / kv_rank_v;
			elem = q_rank_v + (k_v % kv_rank_v);
		}
		else
		{
			const uint32_t v_v = logical_v - q2_v - kv_v;
			rank = v_v / kv_rank_v;
			elem = q_rank_v + kv_rank_v + (v_v % kv_rank_v);
		}
		return gathered4[(((uint64_t)rank * rows) + row) * attn_rows_v + elem];
	}
	else
	{
		const uint32_t rank = logical_v / attn_rows_v;
		const uint32_t elem = logical_v % attn_rows_v;
		return gathered4[(((uint64_t)rank * rows) + row) * attn_rows_v + elem];
	}
}

template<class G>
static __device__ __forceinline__ uint4 QwenAttnNormRopeHead(uint4 v, const float *__restrict__ norm_w, uint16_t *__restrict__ rope_buf, float position, uint32_t lane)
{
	constexpr uint32_t HALF = G::kRopeDim / 2u;
	float x[8], w[8], out[8];
	const float total = QwenWarpSum(LmSumSqBf16x8(v, 0.0f));
	const float scale = SparkMathRmsScale(total, (float)G::kHeadDim, QH_EPSILON);
	LmLoadBf16x8AsFloat(v, x);
	LmLoadWeight8(norm_w, lane << 3u, w);
	#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k )
		out[k] = SparkMathRoundBf16(x[k] * scale * w[k]);
	const uint4 normed = LmPackUint4FloatToBf16(out);
	if ( lane < (G::kRopeDim >> 3u) )
		*(uint4 *)(rope_buf + (lane << 3u)) = normed;
	__syncwarp();
	if ( lane < HALF )
	{
		const float angle = position * __powf(G::kRopeTheta, -2.0f * (float)lane / (float)G::kRopeDim);
		LmRopeRotate<LM_ROPE_HALF_SPLIT, G::kRopeRoundedTerms ? LM_ROPE_ROUNDED_TERMS : LM_ROPE_FP32>(rope_buf, 0u, lane, HALF, angle, 1.0f);
	}
	__syncwarp();
	return lane < (G::kRopeDim >> 3u) ? *(const uint4 *)(rope_buf + (lane << 3u)) : normed;
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenHybridAttnPrepKernel(const uint16_t *__restrict__ gathered, const float *__restrict__ q_norm, const float *__restrict__ k_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row,
	LmKvShardView kv_view, uint16_t *__restrict__ attn_query, uint16_t *__restrict__ merged_gate, uint32_t rows, uint32_t attn_rows,
	uint32_t tp_rank, uint32_t attn_width)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	constexpr uint32_t HD_V = G::kHeadDim >> 3u;
	constexpr uint32_t Q2_V = (2u * S::kQWidth) >> 3u;
	constexpr uint32_t KV_V = S::kKvWidth >> 3u;
	constexpr uint32_t TASKS = G::kHeads + 2u * G::kKvHeads;
	__shared__ __align__(16) uint16_t rope_smem[WARPS][G::kRopeDim];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	LmDependentRelease();
	if ( task >= TASKS || row >= rows )
		return;
	const uint4 *gathered4 = (const uint4 *)gathered;
	const uint32_t attn_rows_v = attn_rows >> 3u;
	if ( task < G::kHeads )
	{
		const uint32_t h = task;
		const uint32_t gate_begin_v = (tp_rank * attn_width) >> 3u;
		const uint32_t attn_width_v = attn_width >> 3u;
		const uint4 qv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, (h * 2u) * HD_V + lane);
		const uint4 q_out = QwenAttnNormRopeHead<G>(qv, q_norm, rope_smem[warp], (float)positions[row], lane);
		((uint4 *)attn_query)[((uint64_t)row * G::kHeads + h) * HD_V + lane] = q_out;
		const uint32_t gate_vec = h * HD_V + lane;
		if ( gate_vec >= gate_begin_v && gate_vec < gate_begin_v + attn_width_v )
			((uint4 *)merged_gate)[(uint64_t)row * attn_width_v + (gate_vec - gate_begin_v)] =
				QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, (h * 2u + 1u) * HD_V + lane);
		return;
	}
	const uint32_t store_pos = store_positions[row];
	if ( SparkKvShardOwns(kv_view.shard, store_pos) == 0u )
		return;
	uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
		return;
	if ( task < G::kHeads + G::kKvHeads )
	{
		const uint32_t kh = task - G::kHeads;
		const uint4 kv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, Q2_V + kh * HD_V + lane);
		const uint4 k_out = QwenAttnNormRopeHead<G>(kv, k_norm, rope_smem[warp], (float)positions[row], lane);
		((uint4 *)slot)[kh * HD_V + lane] = k_out;
	}
	else
	{
		const uint32_t vh = task - G::kHeads - G::kKvHeads;
		const uint4 vv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, Q2_V + KV_V + vh * HD_V + lane);
		((uint4 *)slot)[KV_V + vh * HD_V + lane] = vv;
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void QwenHybridOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(gate_bf16 + base, output_bf16 + base, output_bf16 + base, dimension, LmSigmoidGateOp<true>{});
}

template<uint32_t HIDDEN>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenSmallNBf16ProjKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation, uint16_t *__restrict__ output,
	uint32_t rows, uint32_t output_dim)
{
	constexpr uint32_t VECS = HIDDEN >> 3u;
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	__shared__ __align__(16) uint4 s_x[VECS];
	const uint32_t row = blockIdx.x;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows )
		return;
	const uint4 *x4 = (const uint4 *)(activation + (uint64_t)row * HIDDEN);
	for ( uint32_t c = threadIdx.x; c < VECS; c += QH_THREADS )
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

template<uint32_t KERNEL>
__global__ __launch_bounds__(LM_CAUSAL_CONV_TILE_THREADS, 1)
static void QwenGdnStridedConvTiledKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t input_stride, uint32_t sequences, uint32_t commit,
	const uint32_t *__restrict__ sequence_row_indices)
{
	constexpr uint32_t HALO = KERNEL - 1u;
	constexpr uint32_t GROUPS = LM_CAUSAL_CONV_TILE_THREADS / LM_CAUSAL_CONV_TILE_CHANNELS;
	__shared__ uint16_t tile[HALO + LM_CAUSAL_CONV_TILE_ROWS][LM_CAUSAL_CONV_TILE_CHANNELS];
	const uint32_t sequence = blockIdx.x, lane = threadIdx.x % LM_CAUSAL_CONV_TILE_CHANNELS, group = threadIdx.x / LM_CAUSAL_CONV_TILE_CHANNELS;
	const uint32_t channel = blockIdx.y * LM_CAUSAL_CONV_TILE_CHANNELS + lane;
	const bool live = channel < channels;
	LmDependentWait();
	LmDependentRelease();
	if ( sequence >= sequences )
		return;
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	const uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	uint16_t *slot = window + (uint64_t)state_index[sequence] * channels * KERNEL;
	float taps_weight[KERNEL];
	uint16_t carried[HALO];
	#pragma unroll
	for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
		taps_weight[tap] = live ? LmBf16ToFloat(weight[(channel * KERNEL) + tap]) : 0.0f;
	if ( group == 0u && live )
		#pragma unroll
		for ( uint32_t tap = 0u; tap < HALO; ++tap )
			tile[tap][lane] = slot[(channel * KERNEL) + 1u + tap];
	for ( uint32_t base = begin; base < end; base += LM_CAUSAL_CONV_TILE_ROWS )
	{
		const uint32_t count = end - base < LM_CAUSAL_CONV_TILE_ROWS ? end - base : LM_CAUSAL_CONV_TILE_ROWS;
		__syncthreads();
		for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
			tile[HALO + offset][lane] = input_bf16[(uint64_t)row * input_stride + channel];
		}
		__syncthreads();
		for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
				total += LmBf16ToFloat(tile[offset + tap][lane]) * taps_weight[tap];
			total = SparkMathSwishFast(LmBf16ToFloat(LmFloatToBf16(total)));
			output_bf16[(uint64_t)row * channels + channel] = LmFloatToBf16(total);
		}
		__syncthreads();
		if ( group == 0u && live )
		{
			if ( commit != 0u && base + count == end )
				#pragma unroll
				for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
					slot[(channel * KERNEL) + tap] = tile[count - 1u + tap][lane];
			#pragma unroll
			for ( uint32_t tap = 0u; tap < HALO; ++tap )
				carried[tap] = tile[count + tap][lane];
			#pragma unroll
			for ( uint32_t tap = 0u; tap < HALO; ++tap )
				tile[tap][lane] = carried[tap];
		}
	}
}

template<uint32_t THREADS, uint32_t KEY_DIM>
__global__ __launch_bounds__(THREADS, 2)
static void QwenGdnSplitExpandGateKernel(const uint16_t *__restrict__ qkv, const uint16_t *__restrict__ ba,
	const float *__restrict__ a_log, const float *__restrict__ dt_bias, uint16_t *__restrict__ query_heads,
	uint16_t *__restrict__ key_heads, uint16_t *__restrict__ value, float *__restrict__ retention, float *__restrict__ write_gate,
	uint32_t gdn_qk, uint32_t gdn_v, uint32_t heads, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows )
		return;
	const uint32_t qk_heads = gdn_qk / KEY_DIM;
	const uint32_t repeat = heads / qk_heads;
	const uint32_t total_ch = 2u * gdn_qk + gdn_v;
	const uint4 *src4 = (const uint4 *)(qkv + (uint64_t)row * total_ch);
	const uint32_t qk_v = gdn_qk >> 3u, v_v = gdn_v >> 3u, hd_v = KEY_DIM >> 3u;
	uint4 *qh4 = (uint4 *)(query_heads + (uint64_t)row * heads * KEY_DIM);
	uint4 *kh4 = (uint4 *)(key_heads + (uint64_t)row * heads * KEY_DIM);
	uint4 *val4 = (uint4 *)(value + (uint64_t)row * gdn_v);
	for ( uint32_t c = threadIdx.x; c < qk_v; c += THREADS )
	{
		const uint32_t head = c / hd_v, within = c % hd_v;
		const uint4 qv = src4[c];
		const uint4 kv = src4[qk_v + c];
		for ( uint32_t r = 0u; r < repeat; ++r )
		{
			const uint32_t dst_idx = (head * repeat + r) * hd_v + within;
			qh4[dst_idx] = qv;
			kh4[dst_idx] = kv;
		}
	}
	for ( uint32_t c = threadIdx.x; c < v_v; c += THREADS )
		val4[c] = src4[2u * qk_v + c];
	const uint32_t ret_v = (heads * KEY_DIM) >> 2u;
	const uint32_t key_v = KEY_DIM >> 2u;
	const uint64_t ba_base = (uint64_t)row * 2u * heads;
	float4 *ret4 = (float4 *)(retention + (uint64_t)row * heads * KEY_DIM);
	for ( uint32_t c = threadIdx.x; c < ret_v; c += THREADS )
	{
		const uint32_t h = c / key_v;
		const float shifted = LmBf16ToFloat(ba[ba_base + heads + h]) + dt_bias[h];
		const float factor = SparkMathGatedDeltaDecay(a_log[h], shifted);
		ret4[c] = make_float4(factor, factor, factor, factor);
		if ( (c % key_v) == 0u )
			write_gate[(uint64_t)row * heads + h] = SparkMathSigmoidFast(LmBf16ToFloat(ba[ba_base + h]));
	}
}

template<bool SIGMOID>
__global__ __launch_bounds__(QH_THREADS, 4)
static void QwenGdnHeadRmsNormGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16,
	const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads,
	uint32_t gate_row_stride, uint32_t gate_col_offset)
{
	static_assert(QH_VALUE_DIM == 128u, "GDN head rms norm gate assumes 128-element value heads");
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows || head >= heads )
		return;
	const uint64_t base = ((uint64_t)row * heads + head) * QH_VALUE_DIM;
	const uint64_t gbase = (uint64_t)row * gate_row_stride + gate_col_offset + (uint64_t)head * QH_VALUE_DIM;
	float in[4], sq[4];
	#pragma unroll
	for ( uint32_t k = 0u; k < 4u; ++k )
	{
		const float v = LmBf16ToFloat(input_bf16[base + k * 32u + lane]);
		in[k] = v;
		sq[k] = QwenWarpSum(v * v);
	}
	const float total = (sq[0] + sq[2]) + (sq[1] + sq[3]);
	const float scale = SparkMathRmsScale(total, (float)QH_VALUE_DIM, QH_EPSILON);
	#pragma unroll
	for ( uint32_t k = 0u; k < 4u; ++k )
	{
		const uint32_t i = k * 32u + lane;
		const float w = LmBf16ToFloat(__ldg(weight_bf16 + i));
		const float g = LmBf16ToFloat(gate_bf16[gbase + i]);
		output_bf16[base + i] = LmFloatToBf16(SparkMathRoundBf16(SparkMathRoundBf16(in[k] * scale) * w) *
			(SIGMOID ? SparkMathSigmoidFast(g) : SparkMathSiluFast(g)));
	}
}

template<uint32_t WIDTH, bool HAS_ADD>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenRmsNormKernel(uint16_t *__restrict__ hidden, const uint16_t *__restrict__ partial, const float *__restrict__ weight,
	uint16_t *__restrict__ normed, float epsilon)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= QH_THREADS * 32u, "width fits in staged vectorized waves");
	constexpr uint32_t VEC_SLOTS = (WIDTH + QH_THREADS * 8u - 1u) / (QH_THREADS * 8u);
	__shared__ float sq_wave[QH_THREADS * 8u];
	__shared__ float reduction[LmBlockWarps<QH_THREADS>()];
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
		const uint32_t wave_base = slot * QH_THREADS * 8u;
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
			if ( wave_base + s * QH_THREADS + threadIdx.x < WIDTH )
				total += sq_wave[s * QH_THREADS + threadIdx.x];
		if ( slot + 1u < VEC_SLOTS )
			__syncthreads();
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<QH_THREADS>(total, reduction), (float)WIDTH, epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t offset = (slot * QH_THREADS + threadIdx.x) * 8u;
		if ( offset < WIDTH )
		{
			float vals[8];
			LmLoadBf16x8AsFloat(staged[slot], vals);
			LmRmsStore8(vals, weight, offset, scale, n_row + offset, LmRmsGainOp<false>{});
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void QwenSiluMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(gate_up_bf16 + base, gate_up_bf16 + base + dimension, output_bf16 + (uint64_t)blockIdx.x * dimension, dimension, LmSiluOp{});
}

static __device__ __forceinline__ uint32_t QwenE2m1PairToBf16(uint32_t byte_val, uint32_t scale_bf16x2)
{
	uint32_t widened, bits, out;
	asm("{\n\t.reg .b8 narrow;\n\tcvt.u8.u32 narrow, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(widened) : "r"(byte_val));
	bits = ((widened >> 3u) & 0x0fff0fffu) | (widened & 0x80008000u);
	asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(out) : "r"(bits), "r"(scale_bf16x2));
	return out;
}

static __device__ __forceinline__ uint32_t QwenScaleToBf16x2(uint8_t ue4m3, float global_scaled)
{
	const float s = LmUe4m3ToFloat(ue4m3) * global_scaled;
	uint32_t packed;
	asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(packed) : "f"(s));
	return packed;
}

static __device__ __forceinline__ void QwenCpAsync16(void *dst, const void *src, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(src), "r"(bytes) : "memory");
}

template<class Format, uint32_t LANES, uint32_t ROWS_PER_GROUP, uint32_t NPG>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenExactLinearKernel(const uint8_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
	constexpr uint32_t GROUPS = QH_THREADS / LANES;
	constexpr uint32_t TILE_ROWS = GROUPS * ROWS_PER_GROUP;
	constexpr uint32_t ELEMENTS = LmSkinnyFormat<Format>::kElements;
	constexpr uint32_t TOTAL_W = NPG * LANES;
	__shared__ __align__(16) uint4 s_w[2][NPG][LANES];
	const uint32_t n0 = blockIdx.x * NPG;
	const uint32_t grp = threadIdx.x / LANES;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	const uint32_t warp_first_row = blockIdx.y * TILE_ROWS + grp * ROWS_PER_GROUP;
	const uint32_t row_cnt = warp_first_row < rows ? (rows - warp_first_row < ROWS_PER_GROUP ? rows - warp_first_row : ROWS_PER_GROUP) : 0u;
	const uint32_t chunks = input_dim / ELEMENTS;
	const uint4 *w4 = (const uint4 *)weight;
	LmDependentWait();
	auto stage_w = [&](uint32_t st, uint32_t k_base) {
		if ( threadIdx.x < TOTAL_W )
		{
			const uint32_t wn = threadIdx.x / LANES, wc = threadIdx.x & (LANES - 1u);
			const uint32_t neuron = n0 + wn, c = k_base + wc;
			const bool ok = neuron < output_dim && c < chunks;
			const uint4 *src = ok ? (w4 + (uint64_t)neuron * chunks + c) : w4;
			QwenCpAsync16(&s_w[st][wn][wc], src, ok ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	stage_w(0u, 0u);
	float acc[NPG][ROWS_PER_GROUP];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
		#pragma unroll
		for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
			acc[n][r] = 0.0f;
	for ( uint32_t k_base = 0u; k_base < chunks; k_base += LANES )
	{
		const uint32_t cur = (k_base / LANES) & 1u;
		if ( k_base + LANES < chunks )
			stage_w(cur ^ 1u, k_base + LANES);
		if ( k_base + LANES < chunks )
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t c = k_base + sub;
		if ( row_cnt != 0u && c < chunks )
		{
			uint4 w[NPG];
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				w[n] = s_w[cur][n][sub];
			#pragma unroll
			for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
			{
				const uint32_t row = r < row_cnt ? warp_first_row + r : 0u;
				const uint4 a = LmSkinnyFormat<LmBf16Format>::Load(activation + (uint64_t)row * input_dim + c * 8u);
				float af[8];
				LmSkinnyUnpackBf16x8(a, af);
				#pragma unroll
				for ( uint32_t n = 0u; n < NPG; ++n )
					acc[n][r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(w[n], af, 0.0f), acc[n][r]);
			}
		}
		if ( k_base + LANES < chunks )
			__syncthreads();
	}
	if ( row_cnt != 0u )
	{
		#pragma unroll
		for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				#pragma unroll
				for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
					acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], step, LANES);
		if ( sub == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < ROWS_PER_GROUP; ++r )
				if ( r < row_cnt )
				{
					const uint64_t row_off = (uint64_t)(warp_first_row + r) * output_dim + n0;
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						if ( n0 + n < output_dim )
							output[row_off + n] = LmFloatToBf16(acc[n][r]);
				}
	}
	LmDependentRelease();
}

template<uint32_t MAX_M_TILES, uint32_t WARPS, bool ACTIVATION_PACKED, bool FUSED_GATE_UP, uint32_t SCALE_TILES = 4u>
__global__ __launch_bounds__(WARPS * 32u, MAX_M_TILES <= 2u ? 2 : 1)
static void QwenNvfp4ExpertsKernel(const uint8_t *__restrict__ w1, const uint8_t *__restrict__ w1_scale_blob,
	const uint8_t *__restrict__ w3, const uint8_t *__restrict__ w3_scale_blob, const uint16_t *__restrict__ activation,
	const float *__restrict__ packed_weight, uint16_t *__restrict__ output, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, uint32_t groups, uint32_t input_dim, uint32_t output_dim)
{
	constexpr uint32_t THREADS = WARPS * 32u;
	constexpr uint32_t TILE_M = MAX_M_TILES * 16u;
	constexpr uint32_t TILE_K = 128u;
	constexpr uint32_t STAGES = 2u;
	constexpr uint32_t N_PER_WARP = FUSED_GATE_UP ? 8u : 16u;
	constexpr uint32_t TILE_N = WARPS * N_PER_WARP;
	constexpr uint32_t W_MATS = FUSED_GATE_UP ? 2u : 1u;
	constexpr uint32_t W_ROWS_TOTAL = W_MATS * TILE_N;
	constexpr uint32_t A_PITCH = TILE_K + 8u;
	constexpr uint32_t B_ROW_BYTES = TILE_K / 2u;
	constexpr uint32_t SC_BYTES = SCALE_TILES * 8u;
	constexpr uint32_t SC_VECS = SC_BYTES / 16u;
	constexpr uint32_t SC_SHIFT = SCALE_TILES == 4u ? 2u : 1u;
	__shared__ __align__(16) uint16_t s_a[STAGES][TILE_M][A_PITCH];
	__shared__ __align__(16) uint8_t s_b[STAGES][W_ROWS_TOTAL][B_ROW_BYTES];
	__shared__ __align__(16) uint8_t s_sc[2][W_ROWS_TOTAL][SC_BYTES];
	const uint32_t n_blocks = (output_dim + TILE_N - 1u) / TILE_N;
	const uint32_t group = blockIdx.x / n_blocks;
	const uint32_t n_block = blockIdx.x % n_blocks;
	LmDependentWait();
	if ( group >= groups )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t row_begin = group_row_offset[group];
	const uint32_t row_end = group_row_offset[group + 1u];
	if ( row_begin >= row_end )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, g = lane >> 2u, c = lane & 3u;
	const uint32_t n_base = n_block * TILE_N, k_tiles = input_dim / TILE_K;
	const uint64_t payload_pitch = (uint64_t)(input_dim >> 1u);
	const uint64_t scale_pitch = (uint64_t)(input_dim >> 4u);
	const uint64_t scale_group_stride = (uint64_t)output_dim * scale_pitch;
	const uint8_t *w1_sc_data = w1_scale_blob + (uint64_t)groups * sizeof(float);
	const uint8_t *w3_sc_data = FUSED_GATE_UP ? (w3_scale_blob + (uint64_t)groups * sizeof(float)) : (const uint8_t *)0;
	const float g1_scaled = ((const float *)w1_scale_blob)[group] * 5.1922968585348276e33f;
	const float g3_scaled = FUSED_GATE_UP ? (((const float *)w3_scale_blob)[group] * 5.1922968585348276e33f) : 0.0f;
	const uint32_t b_row0 = FUSED_GATE_UP ? (warp * 8u + g) : (warp * 16u + g);
	const uint32_t b_row1 = FUSED_GATE_UP ? (TILE_N + warp * 8u + g) : (warp * 16u + 8u + g);
	for ( uint32_t first = row_begin; first < row_end; first += TILE_M )
	{
		const uint32_t count = row_end - first < TILE_M ? row_end - first : TILE_M;
		const uint32_t m_tiles_live = (count + 15u) >> 4u;
		constexpr uint32_t A_VEC_THREADS = TILE_K / 8u;
		constexpr uint32_t A_ROW_STEP = THREADS / A_VEC_THREADS;
		constexpr uint32_t A_COPIES = (TILE_M + A_ROW_STEP - 1u) / A_ROW_STEP;
		const uint32_t a_vec = threadIdx.x % A_VEC_THREADS, a_row0 = threadIdx.x / A_VEC_THREADS;
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
					QwenCpAsync16(&s_a[st][lr][a_vec * 8u], a_ptrs[i] + (uint64_t)kt * TILE_K, 16u);
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
				QwenCpAsync16(&s_b[st][wr][bv * 16u], src, ok ? 16u : 0u);
			}
			if ( (kt & (SCALE_TILES - 1u)) == 0u )
			{
				const uint32_t sc_off = kt * 8u, avail = sc_off < scale_pitch ? (uint32_t)(scale_pitch - sc_off) : 0u;
				const uint32_t sb_slot = (kt >> SC_SHIFT) & 1u;
				#pragma unroll
				for ( uint32_t idx = threadIdx.x; idx < W_ROWS_TOTAL * SC_VECS; idx += THREADS )
				{
					const uint32_t wr = idx / SC_VECS, half = idx % SC_VECS;
					const uint32_t mat = FUSED_GATE_UP ? (wr / TILE_N) : 0u;
					const uint32_t neuron = n_base + (FUSED_GATE_UP ? (wr % TILE_N) : wr);
					const uint32_t h_off = half * 16u;
					const uint32_t bytes = (neuron < output_dim && h_off < avail) ? (avail - h_off >= 16u ? 16u : avail - h_off) : 0u;
					const uint8_t *sbase = (mat == 0u ? w1_sc_data : w3_sc_data);
					const uint8_t *src = bytes != 0u ? sbase + (uint64_t)group * scale_group_stride + (uint64_t)neuron * scale_pitch + sc_off + h_off : sbase;
					QwenCpAsync16(&s_sc[sb_slot][wr][h_off], src, bytes);
				}
			}
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		};
		stage_tile(0u, 0u);
		float acc0[MAX_M_TILES][4], acc1[MAX_M_TILES][4];
		#pragma unroll
		for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
			acc0[m][0] = acc0[m][1] = acc0[m][2] = acc0[m][3] = acc1[m][0] = acc1[m][1] = acc1[m][2] = acc1[m][3] = 0.0f;
		for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
		{
			const uint32_t cur = kt & 1u;
			if ( kt + 1u < k_tiles )
				stage_tile(cur ^ 1u, kt + 1u);
			if ( kt + 1u < k_tiles )
				asm volatile("cp.async.wait_group 1;\n" ::: "memory");
			else
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			const uint32_t sb_slot = (kt >> SC_SHIFT) & 1u, sb_sub = (kt & (SCALE_TILES - 1u)) * 8u;
			const uint64_t sc0_all = *(const uint64_t *)&s_sc[sb_slot][b_row0][sb_sub];
			const uint64_t sc1_all = *(const uint64_t *)&s_sc[sb_slot][b_row1][sb_sub];
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 8u; ++ks )
			{
				const uint32_t s0_2 = QwenScaleToBf16x2((uint8_t)(sc0_all >> (ks * 8u)), g1_scaled);
				const uint32_t s1_2 = QwenScaleToBf16x2((uint8_t)(sc1_all >> (ks * 8u)), FUSED_GATE_UP ? g3_scaled : g1_scaled);
				const uint32_t byte_off = ks * 8u + c;
				const uint32_t b0[2] = { QwenE2m1PairToBf16(s_b[cur][b_row0][byte_off], s0_2), QwenE2m1PairToBf16(s_b[cur][b_row0][byte_off + 4u], s0_2) };
				const uint32_t b1[2] = { QwenE2m1PairToBf16(s_b[cur][b_row1][byte_off], s1_2), QwenE2m1PairToBf16(s_b[cur][b_row1][byte_off + 4u], s1_2) };
				const uint32_t k_off = ks * 16u;
				#pragma unroll
				for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
				{
					if ( m < m_tiles_live )
					{
						const uint16_t *ap0 = &s_a[cur][m * 16u + g][k_off + c * 2u];
						const uint16_t *ap1 = &s_a[cur][m * 16u + g + 8u][k_off + c * 2u];
						const uint32_t af[4] = { *(const uint32_t *)ap0, *(const uint32_t *)ap1, *(const uint32_t *)(ap0 + 8u), *(const uint32_t *)(ap1 + 8u) };
						LmMmaBf16(acc0[m], af, b0);
						LmMmaBf16(acc1[m], af, b1);
					}
				}
			}
			__syncthreads();
		}
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
								const float w = packed_weight[pr];
								const float v0 = SparkMathSiluFast(LmBf16ToFloat(LmFloatToBf16(acc0[m][h * 2u]))) * LmBf16ToFloat(LmFloatToBf16(acc1[m][h * 2u])) * w;
								const float v1 = SparkMathSiluFast(LmBf16ToFloat(LmFloatToBf16(acc0[m][h * 2u + 1u]))) * LmBf16ToFloat(LmFloatToBf16(acc1[m][h * 2u + 1u])) * w;
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
	}
	LmDependentRelease();
}

static __device__ __forceinline__ float QwenNvfp4HalfDot(const LmSkinnyE2m1HalfDecoded &d, const uint16_t *act, uint32_t half, float sc, float acc)
{
	const LmSkinnyWide a = LmSkinnyFormat<LmNvfp4>::LoadHalf(act, half);
	float af[16];
	LmSkinnyUnpackBf16x8(a.low, af);
	LmSkinnyUnpackBf16x8(a.high, af + 8);
	return fmaf(sc, LmSkinnyDotPairs16(d.pair, af), acc);
}

static __device__ __forceinline__ LmSkinnyE2m1HalfDecoded QwenNvfp4DecodeHalf(uint4 vw, uint32_t half)
{
	return half == 0u ? LmSkinnyDecodeE2m1Half(vw.x, vw.y) : LmSkinnyDecodeE2m1Half(vw.z, vw.w);
}

static __device__ __forceinline__ float2 QwenNvfp4LoadSc(const uint8_t *s_ptr, uint32_t c, float g)
{
	return LmSkinnyDecodeUe4m3Raw(LmSkinnyUe4m3Raw{ __ldg(s_ptr + 2u * c), __ldg(s_ptr + 2u * c + 1u) }, g);
}

template<uint32_t K, uint32_t N, uint32_t GROUPS>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenNvfp4SkinnyW1W3SingleKernel(const uint8_t *__restrict__ w1, const uint8_t *__restrict__ w1_scale_blob,
	const uint8_t *__restrict__ w3, const uint8_t *__restrict__ w3_scale_blob, const uint16_t *__restrict__ activation,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ packed_expert, uint16_t *__restrict__ output)
{
	constexpr uint32_t CHUNKS = K >> 5u, LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u), PER_BLK = QH_THREADS / LANES;
	const uint32_t pr = blockIdx.y, neuron = blockIdx.x * PER_BLK + (threadIdx.x / LANES), sub = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	if ( pr >= __ldg(group_row_offset + GROUPS) || neuron >= N )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t group = __ldg(packed_expert + pr);
	const uint64_t row_off = ((uint64_t)group * N + neuron) * CHUNKS;
	const uint4 *w1_ptr = (const uint4 *)w1 + row_off, *w3_ptr = (const uint4 *)w3 + row_off;
	const uint8_t *s1_ptr = w1_scale_blob + (uint64_t)GROUPS * sizeof(float) + row_off * 2u;
	const uint8_t *s3_ptr = w3_scale_blob + (uint64_t)GROUPS * sizeof(float) + row_off * 2u;
	const float g1 = __ldg((const float *)w1_scale_blob + group), g3 = __ldg((const float *)w3_scale_blob + group);
	float acc1 = 0.0f, acc3 = 0.0f;
	for ( uint32_t c = sub; c < CHUNKS; c += LANES )
	{
		const uint4 vw1 = __ldcs(w1_ptr + c), vw3 = __ldcs(w3_ptr + c);
		const float2 sc1 = QwenNvfp4LoadSc(s1_ptr, c, g1), sc3 = QwenNvfp4LoadSc(s3_ptr, c, g3);
		#pragma unroll
		for ( uint32_t half = 0u; half < 2u; ++half )
		{
			const LmSkinnyWide a = LmSkinnyFormat<LmNvfp4>::LoadHalf(activation + c * 32u, half);
			float af[16];
			LmSkinnyUnpackBf16x8(a.low, af);
			LmSkinnyUnpackBf16x8(a.high, af + 8);
			acc1 = fmaf(half == 0u ? sc1.x : sc1.y, LmSkinnyDotPairs16(QwenNvfp4DecodeHalf(vw1, half).pair, af), acc1);
			acc3 = fmaf(half == 0u ? sc3.x : sc3.y, LmSkinnyDotPairs16(QwenNvfp4DecodeHalf(vw3, half).pair, af), acc3);
		}
	}
	#pragma unroll
	for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
	{
		acc1 += __shfl_xor_sync(0xffffffffu, acc1, step, LANES);
		acc3 += __shfl_xor_sync(0xffffffffu, acc3, step, LANES);
	}
	if ( sub == 0u )
		output[(uint64_t)pr * N + neuron] = LmFloatToBf16(
			SparkMathGateLimitedGlu(LmBf16ToFloat(LmFloatToBf16(acc1)), LmBf16ToFloat(LmFloatToBf16(acc3)), INFINITY) * __ldg(packed_weight + pr));
	LmDependentRelease();
}

template<uint32_t K, uint32_t N, uint32_t GROUPS>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenNvfp4SkinnyW1W3BatchedKernel(const uint8_t *__restrict__ w1, const uint8_t *__restrict__ w1_scale_blob,
	const uint8_t *__restrict__ w3, const uint8_t *__restrict__ w3_scale_blob, const uint16_t *__restrict__ activation,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ output)
{
	constexpr uint32_t CHUNKS = K >> 5u, LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u), PER_BLK = QH_THREADS / LANES;
	const uint32_t group = blockIdx.y, neuron = blockIdx.x * PER_BLK + (threadIdx.x / LANES), sub = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	const uint32_t row_begin = __ldg(group_row_offset + group), row_end = __ldg(group_row_offset + group + 1u);
	if ( row_begin >= row_end || neuron >= N )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t row_off = ((uint64_t)group * N + neuron) * CHUNKS;
	const uint4 *w1_ptr = (const uint4 *)w1 + row_off, *w3_ptr = (const uint4 *)w3 + row_off;
	const uint8_t *s1_ptr = w1_scale_blob + (uint64_t)GROUPS * sizeof(float) + row_off * 2u;
	const uint8_t *s3_ptr = w3_scale_blob + (uint64_t)GROUPS * sizeof(float) + row_off * 2u;
	const float g1 = __ldg((const float *)w1_scale_blob + group), g3 = __ldg((const float *)w3_scale_blob + group);
	for ( uint32_t first = row_begin; first < row_end; )
	{
		const uint32_t rem = row_end - first;
		if ( rem == 1u )
		{
			const uint16_t *act = activation + (uint64_t)__ldg(route_source_token + first) * K;
			float acc1 = 0.0f, acc3 = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw1 = __ldcs(w1_ptr + c), vw3 = __ldcs(w3_ptr + c);
				const float2 sc1 = QwenNvfp4LoadSc(s1_ptr, c, g1), sc3 = QwenNvfp4LoadSc(s3_ptr, c, g3);
				#pragma unroll
				for ( uint32_t half = 0u; half < 2u; ++half )
				{
					const LmSkinnyWide a = LmSkinnyFormat<LmNvfp4>::LoadHalf(act + c * 32u, half);
					float af[16];
					LmSkinnyUnpackBf16x8(a.low, af);
					LmSkinnyUnpackBf16x8(a.high, af + 8);
					acc1 = fmaf(half == 0u ? sc1.x : sc1.y, LmSkinnyDotPairs16(QwenNvfp4DecodeHalf(vw1, half).pair, af), acc1);
					acc3 = fmaf(half == 0u ? sc3.x : sc3.y, LmSkinnyDotPairs16(QwenNvfp4DecodeHalf(vw3, half).pair, af), acc3);
				}
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
			{
				acc1 += __shfl_xor_sync(0xffffffffu, acc1, step, LANES);
				acc3 += __shfl_xor_sync(0xffffffffu, acc3, step, LANES);
			}
			if ( sub == 0u )
				output[(uint64_t)first * N + neuron] = LmFloatToBf16(
					SparkMathGateLimitedGlu(LmBf16ToFloat(LmFloatToBf16(acc1)), LmBf16ToFloat(LmFloatToBf16(acc3)), INFINITY) * __ldg(packed_weight + first));
			first += 1u;
		}
		else
		{
			const uint32_t cnt = rem < 4u ? rem : 4u;
			uint32_t src[4];
			float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, acc3[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			#pragma unroll
			for ( uint32_t r = 0u; r < 4u; ++r )
				src[r] = __ldg(route_source_token + first + (r < cnt ? r : 0u));
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw1 = __ldcs(w1_ptr + c), vw3 = __ldcs(w3_ptr + c);
				const float2 sc1 = QwenNvfp4LoadSc(s1_ptr, c, g1), sc3 = QwenNvfp4LoadSc(s3_ptr, c, g3);
				#pragma unroll
				for ( uint32_t half = 0u; half < 2u; ++half )
				{
					const LmSkinnyE2m1HalfDecoded d1 = QwenNvfp4DecodeHalf(vw1, half);
					#pragma unroll
					for ( uint32_t r = 0u; r < 4u; ++r )
						acc1[r] = QwenNvfp4HalfDot(d1, activation + (uint64_t)src[r] * K + c * 32u, half, half == 0u ? sc1.x : sc1.y, acc1[r]);
					const LmSkinnyE2m1HalfDecoded d3 = QwenNvfp4DecodeHalf(vw3, half);
					#pragma unroll
					for ( uint32_t r = 0u; r < 4u; ++r )
						acc3[r] = QwenNvfp4HalfDot(d3, activation + (uint64_t)src[r] * K + c * 32u, half, half == 0u ? sc3.x : sc3.y, acc3[r]);
				}
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
				{
					acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], step, LANES);
					acc3[r] += __shfl_xor_sync(0xffffffffu, acc3[r], step, LANES);
				}
			if ( sub == 0u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					if ( r < cnt )
						output[(uint64_t)(first + r) * N + neuron] = LmFloatToBf16(
							SparkMathGateLimitedGlu(LmBf16ToFloat(LmFloatToBf16(acc1[r])), LmBf16ToFloat(LmFloatToBf16(acc3[r])), INFINITY) * __ldg(packed_weight + first + r));
			first += cnt;
		}
	}
	LmDependentRelease();
}

template<uint32_t K, uint32_t N, uint32_t GROUPS>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenNvfp4SkinnyW2SingleKernel(const uint8_t *__restrict__ w2, const uint8_t *__restrict__ w2_scale_blob,
	const uint16_t *__restrict__ activation, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ packed_expert, uint16_t *__restrict__ output)
{
	constexpr uint32_t CHUNKS = K >> 5u, LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u), PER_BLK = QH_THREADS / LANES, NPG = 4u;
	const uint32_t pr = blockIdx.y, n0 = (blockIdx.x * PER_BLK + (threadIdx.x / LANES)) * NPG, sub = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	if ( pr >= __ldg(group_row_offset + GROUPS) || n0 >= N )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t group = __ldg(packed_expert + pr);
	const uint64_t off0 = ((uint64_t)group * N + n0) * CHUNKS;
	const uint4 *w_ptr[NPG];
	const uint8_t *s_ptr[NPG];
	bool live[NPG];
	#pragma unroll
	for ( uint32_t k = 0u; k < NPG; ++k )
	{
		live[k] = (n0 + k) < N;
		const uint64_t off = live[k] ? (off0 + (uint64_t)k * CHUNKS) : off0;
		w_ptr[k] = (const uint4 *)w2 + off;
		s_ptr[k] = w2_scale_blob + (uint64_t)GROUPS * sizeof(float) + off * 2u;
	}
	const float glob = __ldg((const float *)w2_scale_blob + group);
	const uint16_t *act = activation + (uint64_t)pr * K;
	float acc[NPG] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for ( uint32_t c = sub; c < CHUNKS; c += LANES )
	{
		uint4 vw[NPG];
		float2 sc[NPG];
		#pragma unroll
		for ( uint32_t k = 0u; k < NPG; ++k )
		{
			vw[k] = live[k] ? __ldcs(w_ptr[k] + c) : make_uint4(0u, 0u, 0u, 0u);
			sc[k] = live[k] ? QwenNvfp4LoadSc(s_ptr[k], c, glob) : make_float2(0.0f, 0.0f);
		}
		#pragma unroll
		for ( uint32_t half = 0u; half < 2u; ++half )
		{
			const LmSkinnyWide a = LmSkinnyFormat<LmNvfp4>::LoadHalf(act + c * 32u, half);
			float af[16];
			LmSkinnyUnpackBf16x8(a.low, af);
			LmSkinnyUnpackBf16x8(a.high, af + 8);
			#pragma unroll
			for ( uint32_t k = 0u; k < NPG; ++k )
				if ( live[k] )
					acc[k] = fmaf(half == 0u ? sc[k].x : sc[k].y, LmSkinnyDotPairs16(QwenNvfp4DecodeHalf(vw[k], half).pair, af), acc[k]);
		}
	}
	#pragma unroll
	for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
		#pragma unroll
		for ( uint32_t k = 0u; k < NPG; ++k )
			if ( live[k] )
				acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], step, LANES);
	if ( sub == 0u )
		#pragma unroll
		for ( uint32_t k = 0u; k < NPG; ++k )
			if ( live[k] )
				output[(uint64_t)pr * N + n0 + k] = LmFloatToBf16(acc[k]);
	LmDependentRelease();
}

template<uint32_t K, uint32_t N, uint32_t GROUPS>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenNvfp4SkinnyW2BatchedKernel(const uint8_t *__restrict__ w2, const uint8_t *__restrict__ w2_scale_blob,
	const uint16_t *__restrict__ activation, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	constexpr uint32_t CHUNKS = K >> 5u, LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u), PER_BLK = QH_THREADS / LANES;
	const uint32_t group = blockIdx.y, neuron = blockIdx.x * PER_BLK + (threadIdx.x / LANES), sub = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	const uint32_t row_begin = __ldg(group_row_offset + group), row_end = __ldg(group_row_offset + group + 1u);
	if ( row_begin >= row_end || neuron >= N )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t off0 = ((uint64_t)group * N + neuron) * CHUNKS;
	const uint4 *w_ptr = (const uint4 *)w2 + off0;
	const uint8_t *s_ptr = w2_scale_blob + (uint64_t)GROUPS * sizeof(float) + off0 * 2u;
	const float glob = __ldg((const float *)w2_scale_blob + group);
	for ( uint32_t first = row_begin; first < row_end; )
	{
		const uint32_t rem = row_end - first;
		if ( rem == 1u )
		{
			const uint16_t *act = activation + (uint64_t)first * K;
			float acc = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw = __ldcs(w_ptr + c);
				const float2 sc = QwenNvfp4LoadSc(s_ptr, c, glob);
				#pragma unroll
				for ( uint32_t half = 0u; half < 2u; ++half )
					acc = QwenNvfp4HalfDot(QwenNvfp4DecodeHalf(vw, half), act + c * 32u, half, half == 0u ? sc.x : sc.y, acc);
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				acc += __shfl_xor_sync(0xffffffffu, acc, step, LANES);
			if ( sub == 0u )
				output[(uint64_t)first * N + neuron] = LmFloatToBf16(acc);
			first += 1u;
		}
		else
		{
			const uint32_t cnt = rem < 4u ? rem : 4u;
			float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw = __ldcs(w_ptr + c);
				const float2 sc = QwenNvfp4LoadSc(s_ptr, c, glob);
				#pragma unroll
				for ( uint32_t half = 0u; half < 2u; ++half )
				{
					const LmSkinnyE2m1HalfDecoded d = QwenNvfp4DecodeHalf(vw, half);
					#pragma unroll
					for ( uint32_t r = 0u; r < 4u; ++r )
						acc[r] = QwenNvfp4HalfDot(d, activation + (uint64_t)(first + (r < cnt ? r : 0u)) * K + c * 32u, half, half == 0u ? sc.x : sc.y, acc[r]);
				}
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], step, LANES);
			if ( sub == 0u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					if ( r < cnt )
						output[(uint64_t)(first + r) * N + neuron] = LmFloatToBf16(acc[r]);
			first += cnt;
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
static void QwenRouteBuildPackedKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint32_t routes,
	uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t *__restrict__ packed_expert, float *__restrict__ packed_weight, uint32_t tile_m, uint32_t n_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t n_down, uint32_t *__restrict__ tile_prefix_down)
{
	LmDependentWait();
	LmRouteBuildBlock<THREADS, EXPERTS>(route_expert, routes, top_k, group_row_offset, route_packed_row, route_source_token, tile_m,
		n_up, tile_prefix_up, n_down, tile_prefix_down);
	__syncthreads();
	for ( uint32_t route = threadIdx.x; route < routes; route += THREADS )
	{
		const uint32_t pr = route_packed_row[route];
		packed_weight[pr] = route_weight[route];
		packed_expert[pr] = route_expert[route];
	}
	LmDependentRelease();
}

template<class G>
static __device__ __forceinline__ void QwenRouterSoftmaxShared(const uint16_t *__restrict__ logits, uint32_t row, float *__restrict__ s_probs, float *__restrict__ reduction)
{
	constexpr uint32_t WIDTH = G::kExperts;
	constexpr uint32_t per_thread = (WIDTH + QH_THREADS - 1u) / QH_THREADS;
	const uint64_t row_base = (uint64_t)row * WIDTH;
	float top = -INFINITY, sum = 0.0f;
	float staged[per_thread];
	#pragma unroll
	for ( uint32_t slot = 0u; slot < per_thread; ++slot )
	{
		const uint32_t index = threadIdx.x + slot * QH_THREADS;
		staged[slot] = index < WIDTH ? LmBf16ToFloat(logits[row_base + index]) : -INFINITY;
		top = fmaxf(top, staged[slot]);
	}
	top = LmBlockMax<QH_THREADS>(top, reduction);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < per_thread; ++slot )
	{
		staged[slot] = threadIdx.x + slot * QH_THREADS < WIDTH ? expf(staged[slot] - top) : 0.0f;
		sum += staged[slot];
	}
	sum = LmBlockSumOnce<QH_THREADS>(sum, reduction);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < per_thread; ++slot )
		if ( threadIdx.x + slot * QH_THREADS < WIDTH )
			s_probs[threadIdx.x + slot * QH_THREADS] = staged[slot] / sum;
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 1)
static void QwenMoeRouteSmallKernel(const uint16_t *__restrict__ logits, uint32_t rows, uint32_t expert_first, uint32_t *__restrict__ fault,
	uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t *__restrict__ packed_expert, float *__restrict__ packed_weight, uint32_t tile_m, uint32_t n_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t n_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ float s_probs[G::kExperts];
	__shared__ float reduction[LmBlockWarps<QH_THREADS>()];
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	const uint32_t routes = rows * G::kTopK;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		QwenRouterSoftmaxShared<G>(logits, r, s_probs, reduction);
		__syncthreads();
		if ( warp == 0u )
			LmTopkWarpRow<G::kTopK, true, LM_TOPK_SCORE_IDENTITY>(s_probs, G::kExperts, s_exp + r * G::kTopK, s_wt + r * G::kTopK,
				(const float *)0, (const uint16_t *)0, 1.0f, fault, 0u, lane);
		__syncthreads();
	}
	if ( threadIdx.x < routes )
	{
		const uint32_t global_exp = s_exp[threadIdx.x];
		route_global[threadIdx.x] = global_exp;
		if ( global_exp >= expert_first && global_exp < expert_first + G::kLocalExperts )
			s_exp[threadIdx.x] = global_exp - expert_first;
		else
		{
			s_exp[threadIdx.x] = G::kLocalExperts;
			s_wt[threadIdx.x] = 0.0f;
		}
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	__syncthreads();
	LmRouteBuildSmallBody<QH_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
		tile_m, n_up, tile_prefix_up, n_down, tile_prefix_down);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		const uint32_t pr = route_packed_row[threadIdx.x];
		packed_weight[pr] = s_wt[threadIdx.x];
		packed_expert[pr] = s_exp[threadIdx.x];
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenMoeSoftmaxTopkLocalKernel(const uint16_t *__restrict__ logits, uint32_t expert_first, uint32_t *__restrict__ fault,
	uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global)
{
	__shared__ float s_probs[G::kExperts];
	__shared__ float reduction[LmBlockWarps<QH_THREADS>()];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	QwenRouterSoftmaxShared<G>(logits, row, s_probs, reduction);
	__syncthreads();
	if ( warp == 0u )
	{
		const uint64_t base = (uint64_t)row * G::kTopK;
		LmTopkWarpRow<G::kTopK, true, LM_TOPK_SCORE_IDENTITY>(s_probs, G::kExperts, route_expert + base, route_weight + base,
			(const float *)0, (const uint16_t *)0, 1.0f, fault, 0u, lane);
		if ( lane < G::kTopK )
		{
			const uint32_t global_exp = route_expert[base + lane];
			route_global[base + lane] = global_exp;
			if ( global_exp >= expert_first && global_exp < expert_first + G::kLocalExperts )
				route_expert[base + lane] = global_exp - expert_first;
			else
			{
				route_expert[base + lane] = G::kLocalExperts;
				route_weight[base + lane] = 0.0f;
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenMoeSwigluPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint16_t *__restrict__ output, uint32_t width)
{
	LmDependentWait();
	LmDependentRelease();
	if ( blockIdx.y >= group_row_offset[local_experts] )
		return;
	LmPackedGateUpBody(gate, up, output, width, LmSwigluPackedOp{weight[blockIdx.y], INFINITY});
}

__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenMoeLocalFinalizeGatedKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, const uint16_t *__restrict__ shared_logit,
	uint16_t *__restrict__ output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK], active_count;
	const uint32_t row = blockIdx.y;
	const uint64_t row_base = (uint64_t)row * width;
	const uint16_t *shared_row = shared + row_base;
	uint16_t *out_row = output + row_base;
	LmDependentWait();
	LmDependentRelease();
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
	const float gate = SparkMathSigmoid(LmBf16ToFloat(shared_logit[row]));
	if ( (width & 7u) == 0u && LmIsAligned16(packed) && LmIsAligned16(out_row) && LmIsAligned16(shared_row) )
	{
		const uint32_t chunks = width >> 3u;
		const uint4 *packed4 = (const uint4 *)packed, *shared4 = (const uint4 *)shared_row;
		uint4 *out4 = (uint4 *)out_row;
		for ( uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += gridDim.x * blockDim.x )
		{
			float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t k = 0u; k < count; ++k )
				LmAddBf16x8(acc, packed4[(uint64_t)active_rows[k] * chunks + c]);
			float s[8];
			LmLoadBf16x8AsFloat(shared4[c], s);
			#pragma unroll
			for ( uint32_t i = 0u; i < 8u; ++i )
				acc[i] += SparkMathRoundBf16(s[i] * gate);
			out4[c] = LmPackUint4FloatToBf16(acc);
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < count; ++k )
			total += LmBf16ToFloat(packed[(uint64_t)active_rows[k] * width + element]);
		total += SparkMathRoundBf16(LmBf16ToFloat(shared_row[element]) * gate);
		out_row[element] = LmFloatToBf16(total);
	}
}

template<uint32_t THREADS, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 2)
static void QwenHybridHcInjectNormKernel(uint16_t *__restrict__ residual_bf16, const uint16_t *__restrict__ partial_bf16,
	const float *__restrict__ inject, const float *__restrict__ norm_weight, uint16_t *__restrict__ hc_normed_bf16,
	uint32_t groups)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= THREADS * LM_RMS_NORM_STAGED, "HC width fits in staged vectorized waves");
	constexpr uint32_t VEC_SLOTS = (WIDTH + THREADS * 8u - 1u) / (THREADS * 8u);
	__shared__ float reduction[LmBlockWarps<THREADS>()], sq_wave[THREADS * 8u];
	const uint32_t stream = blockIdx.x, row = blockIdx.y;
	const uint64_t res_base = ((uint64_t)row * groups + stream) * WIDTH;
	const uint64_t part_base = (uint64_t)row * WIDTH;
	uint16_t *res_row = residual_bf16 + res_base;
	const uint16_t *part_row = partial_bf16 + part_base;
	uint16_t *out_row = hc_normed_bf16 + res_base;
	const float *group_weight = norm_weight + (uint64_t)stream * WIDTH;
	const float weight = inject[(uint64_t)row * groups + stream];
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
			float r[8], p[8], sq[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), r);
			LmLoadBf16x8AsFloat(*(const uint4 *)(part_row + offset), p);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				r[k] += SparkMathHyperInjected(p[k], weight);
			const uint4 packed = LmPackUint4FloatToBf16(r);
			*(uint4 *)(res_row + offset) = packed;
			staged[slot] = packed;
			LmLoadBf16x8AsFloat(packed, r);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				sq[k] = r[k] * r[k];
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
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)WIDTH, QH_EPSILON);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
		if ( (slot * THREADS + threadIdx.x) * 8u < WIDTH )
			LmGroupRmsStore8(staged[slot], group_weight, (slot * THREADS + threadIdx.x) * 8u, scale, out_row);
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH, uint32_t LOW_RANK, uint32_t RG>
__global__ __launch_bounds__(128u, 2)
static void QwenHcDownGateInjectKernel(const uint16_t *__restrict__ down, const uint16_t *__restrict__ inject_w,
	const uint16_t *__restrict__ hc_normed, uint16_t *__restrict__ hc_low, float *__restrict__ hc_inject, uint32_t rows)
{
	constexpr uint32_t K = STREAMS * WIDTH;
	constexpr uint32_t CHUNKS = K >> 3u;
	constexpr uint32_t TOTAL_N = LOW_RANK + STREAMS;
	constexpr uint32_t NPG = 2u;
	constexpr uint32_t WARPS = 4u;
	const uint32_t row_groups = (rows + RG - 1u) / RG;
	const uint32_t task = (blockIdx.x / row_groups) * WARPS + (threadIdx.x >> 5u);
	const uint32_t first_row = (blockIdx.x % row_groups) * RG;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t n0 = task * NPG;
	LmDependentWait();
	if ( n0 < TOTAL_N )
	{
		const uint32_t count = rows - first_row < RG ? rows - first_row : RG;
		const bool has1 = n0 + 1u < TOTAL_N;
		const uint4 *w0 = (const uint4 *)(n0 < LOW_RANK ? down + (uint64_t)n0 * K : inject_w + (uint64_t)(n0 - LOW_RANK) * K);
		const uint4 *w1 = has1 ? (const uint4 *)(n0 + 1u < LOW_RANK ? down + (uint64_t)(n0 + 1u) * K : inject_w + (uint64_t)(n0 + 1u - LOW_RANK) * K) : w0;
		const uint4 *a[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			a[r] = (const uint4 *)(hc_normed + (uint64_t)(first_row + (r < count ? r : 0u)) * K);
		float acc0[RG], acc1[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			acc0[r] = acc1[r] = 0.0f;
		for ( uint32_t c = lane; c < CHUNKS; c += 32u )
		{
			const uint4 vw0 = __ldcs(w0 + c);
			const uint4 vw1 = has1 ? __ldcs(w1 + c) : make_uint4(0u, 0u, 0u, 0u);
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				const uint4 va = __ldg(a[r] + c);
				acc0[r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw0, va, 0.0f), acc0[r]);
				if ( has1 )
					acc1[r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw1, va, 0.0f), acc1[r]);
			}
		}
		#pragma unroll
		for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				acc0[r] += __shfl_xor_sync(0xffffffffu, acc0[r], step, 32);
				if ( has1 )
					acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], step, 32);
			}
		if ( lane == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < count )
				{
					const uint32_t row = first_row + r;
					const float v0 = LmBf16ToFloat(LmFloatToBf16(acc0[r]));
					if ( n0 < LOW_RANK )
						hc_low[(uint64_t)row * LOW_RANK + n0] = LmFloatToBf16(SparkMathHyperGate(v0, (float)STREAMS));
					else
						hc_inject[(uint64_t)row * STREAMS + (n0 - LOW_RANK)] = SparkMathHyperInjectWeight(v0, (float)STREAMS);
					if ( has1 )
					{
						const float v1 = LmBf16ToFloat(LmFloatToBf16(acc1[r]));
						if ( n0 + 1u < LOW_RANK )
							hc_low[(uint64_t)row * LOW_RANK + (n0 + 1u)] = LmFloatToBf16(SparkMathHyperGate(v1, (float)STREAMS));
						else
							hc_inject[(uint64_t)row * STREAMS + (n0 + 1u - LOW_RANK)] = SparkMathHyperInjectWeight(v1, (float)STREAMS);
					}
				}
	}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH, uint32_t LOW_RANK, uint32_t RG>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenHcUpMixKernel(const uint16_t *__restrict__ up, const uint16_t *__restrict__ hc_low,
	const uint16_t *__restrict__ hc_normed, uint16_t *__restrict__ mixed, uint32_t rows)
{
	constexpr uint32_t CHUNKS = LOW_RANK >> 3u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = QH_THREADS / LANES;
	__shared__ __align__(16) uint4 s_low[RG][CHUNKS];
	const uint32_t row_groups = (rows + RG - 1u) / RG;
	const uint32_t col = (blockIdx.x / row_groups) * GROUPS + (threadIdx.x / LANES);
	const uint32_t first_row = (blockIdx.x % row_groups) * RG;
	const uint32_t count = rows - first_row < RG ? rows - first_row : RG;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	for ( uint32_t idx = threadIdx.x; idx < count * CHUNKS; idx += QH_THREADS )
	{
		const uint32_t r = idx / CHUNKS, c = idx % CHUNKS;
		s_low[r][c] = ((const uint4 *)(hc_low + (uint64_t)(first_row + r) * LOW_RANK))[c];
	}
	__syncthreads();
	if ( col < WIDTH )
	{
		const uint4 *w[STREAMS];
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			w[s] = (const uint4 *)(up + (uint64_t)(s * WIDTH + col) * LOW_RANK);
		float acc[STREAMS][RG];
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[s][r] = 0.0f;
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		{
			uint4 vw[STREAMS];
			#pragma unroll
			for ( uint32_t s = 0u; s < STREAMS; ++s )
				vw[s] = __ldcs(w[s] + c);
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				const uint4 va = s_low[r < count ? r : 0u][c];
				#pragma unroll
				for ( uint32_t s = 0u; s < STREAMS; ++s )
					acc[s][r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw[s], va, 0.0f), acc[s][r]);
			}
		}
		#pragma unroll
		for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t s = 0u; s < STREAMS; ++s )
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					acc[s][r] += __shfl_xor_sync(0xffffffffu, acc[s][r], step, LANES);
		if ( sub == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < count )
				{
					const uint32_t row = first_row + r;
					float total = 0.0f;
					#pragma unroll
					for ( uint32_t s = 0u; s < STREAMS; ++s )
					{
						const float u = LmBf16ToFloat(LmFloatToBf16(acc[s][r]));
						const float n = LmBf16ToFloat(hc_normed[((uint64_t)row * STREAMS + s) * WIDTH + col]);
						total += SparkMathHyperMixTerm(SparkMathHyperMixGate(u), n);
					}
					mixed[(uint64_t)row * WIDTH + col] = LmFloatToBf16(total / (float)STREAMS);
				}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t STREAMS, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 4)
static void QwenHyperMixKernel(const uint16_t *__restrict__ up_bf16, const uint16_t *__restrict__ normed_bf16,
	uint16_t *__restrict__ mixed_bf16)
{
	static_assert((WIDTH & 7u) == 0u, "hyper mix width is whole uint4 vectors");
	constexpr uint32_t VECS = WIDTH >> 3u;
	const uint32_t row = blockIdx.x;
	const uint64_t base = (uint64_t)row * STREAMS * WIDTH;
	LmDependentWait();
	for ( uint32_t v = threadIdx.x; v < VECS; v += THREADS )
	{
		const uint32_t element = v << 3u;
		float total[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
		{
			const uint64_t at = base + (uint64_t)s * WIDTH + element;
			float u[8], n[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(up_bf16 + at), u);
			LmLoadBf16x8AsFloat(*(const uint4 *)(normed_bf16 + at), n);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				total[k] += SparkMathHyperMixTerm(SparkMathHyperMixGate(u[k]), n[k]);
		}
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			total[k] /= (float)STREAMS;
		*(uint4 *)(mixed_bf16 + (uint64_t)row * WIDTH + element) = LmPackUint4FloatToBf16(total);
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t STREAMS>
__global__ __launch_bounds__(THREADS, 2)
static void QwenHyperInjectWeightsVecKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ weight_bf16,
	float *__restrict__ inject, uint32_t width)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint32_t row = blockIdx.x, total_v = (STREAMS * width) >> 3u;
	const uint4 *src4 = (const uint4 *)(normed_bf16 + (uint64_t)row * STREAMS * width);
	float sums[STREAMS];
	LmDependentWait();
	LmDependentRelease();
	#pragma unroll
	for ( uint32_t s = 0u; s < STREAMS; ++s )
		sums[s] = 0.0f;
	for ( uint32_t c = threadIdx.x; c < total_v; c += THREADS )
	{
		const uint4 x = src4[c];
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			sums[s] = LmDotBf16x8(x, ((const uint4 *)weight_bf16)[(uint64_t)s * total_v + c], sums[s]);
	}
	#pragma unroll
	for ( uint32_t s = 0u; s < STREAMS; ++s )
	{
		const float logit = SparkMathRoundBf16(LmBlockSum<THREADS>(sums[s], reduction));
		if ( threadIdx.x == 0u )
			inject[(uint64_t)row * STREAMS + s] = SparkMathHyperInjectWeight(logit, (float)STREAMS);
	}
}

template<uint32_t THREADS, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 1)
static void QwenPleKeyQueryGateNormKernel(const uint16_t *__restrict__ ple_key, const uint16_t *__restrict__ residual,
	const uint16_t *__restrict__ ple_value, const float *__restrict__ norm_key, const float *__restrict__ norm_query,
	const float *__restrict__ norm_conv, uint16_t *__restrict__ ple_gated, uint16_t *__restrict__ ple_conv_in, uint32_t streams)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= THREADS * LM_RMS_NORM_STAGED, "PLE width fits in staged group rms norm");
	constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
	__shared__ float reduction[LmBlockWarps<THREADS>()], sq_wave[THREADS * 8u];
	__shared__ uint16_t k_buf[WIDTH], q_buf[WIDTH];
	const uint32_t stream = blockIdx.x, row = blockIdx.y;
	const uint64_t base = ((uint64_t)row * streams + stream) * WIDTH;
	const uint16_t *key_row = ple_key + base, *res_row = residual + base, *val_row = ple_value + (uint64_t)row * WIDTH;
	uint16_t *gated_row = ple_gated + base, *conv_row = ple_conv_in + base;
	const float *wk = norm_key + (uint64_t)stream * WIDTH;
	const float *wq = norm_query + (uint64_t)stream * WIDTH;
	const float *wc = norm_conv + (uint64_t)stream * WIDTH;
	uint4 k_staged[vec_slots], q_staged[vec_slots], g_staged[vec_slots];
	float k_tot = 0.0f, q_tot = 0.0f, dot = 0.0f, g_tot = 0.0f;
	LmDependentWait();
	LmDependentRelease();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
	{
		const uint32_t wave_base = slot * THREADS * 8u;
		if ( wave_base >= WIDTH )
			k_staged[slot] = make_uint4(0u, 0u, 0u, 0u);
		else
			LmGroupRmsWaveSq<THREADS>(key_row, sq_wave, wave_base, WIDTH, &k_staged[slot], &k_tot);
	}
	const float k_scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(k_tot, reduction), (float)WIDTH, QH_EPSILON);
	__syncthreads();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < WIDTH )
			LmGroupRmsStore8(k_staged[slot], wk, offset, k_scale, k_buf);
	}
	#pragma unroll
	for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
	{
		const uint32_t wave_base = slot * THREADS * 8u;
		if ( wave_base >= WIDTH )
			q_staged[slot] = make_uint4(0u, 0u, 0u, 0u);
		else
			LmGroupRmsWaveSq<THREADS>(res_row, sq_wave, wave_base, WIDTH, &q_staged[slot], &q_tot);
	}
	const float q_scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(q_tot, reduction), (float)WIDTH, QH_EPSILON);
	__syncthreads();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < WIDTH )
			LmGroupRmsStore8(q_staged[slot], wq, offset, q_scale, q_buf);
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += THREADS )
		dot += SparkMathRoundBf16(LmBf16ToFloat(k_buf[index]) * LmBf16ToFloat(q_buf[index]));
	const float gate = SparkMathHyperKeyGate(LmBlockSumOnce<THREADS>(dot, reduction), WIDTH);
	__syncthreads();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
	{
		const uint32_t wave_base = slot * THREADS * 8u;
		if ( wave_base >= WIDTH )
			g_staged[slot] = make_uint4(0u, 0u, 0u, 0u);
		else
		{
			const uint32_t offset = wave_base + threadIdx.x * 8u;
			if ( offset < WIDTH )
			{
				float v[8], sq[8];
				LmLoadBf16x8AsFloat(*(const uint4 *)(val_row + offset), v);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					v[k] = gate * v[k];
				const uint4 packed = LmPackUint4FloatToBf16(v);
				*(uint4 *)(gated_row + offset) = packed;
				g_staged[slot] = packed;
				LmLoadBf16x8AsFloat(packed, v);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					sq[k] = v[k] * v[k];
				LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
			}
			else
				g_staged[slot] = make_uint4(0u, 0u, 0u, 0u);
			__syncthreads();
			#pragma unroll
			for ( uint32_t s = 0u; s < 8u; ++s )
				if ( wave_base + s * THREADS + threadIdx.x < WIDTH )
					g_tot += sq_wave[s * THREADS + threadIdx.x];
			if ( wave_base + THREADS * 8u < WIDTH )
				__syncthreads();
		}
	}
	const float c_scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(g_tot, reduction), (float)WIDTH, QH_EPSILON);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
		if ( (slot * THREADS + threadIdx.x) * 8u < WIDTH )
			LmGroupRmsStore8(g_staged[slot], wc, (slot * THREADS + threadIdx.x) * 8u, c_scale, conv_row);
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenIndexQPrepKernel(const uint16_t *__restrict__ index_qk, const float *__restrict__ q_norm,
	const uint32_t *__restrict__ positions, uint16_t *__restrict__ index_query, uint16_t *__restrict__ index_key, uint32_t rows)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, D_V = D >> 3u, H = G::kIndexHeads, HALF = G::kRopeDim / 2u;
	__shared__ __align__(16) uint16_t s_q[H][G::kRopeDim];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows )
		return;
	const uint4 *src4 = (const uint4 *)(index_qk + (uint64_t)row * S::kIndexWidth);
	if ( warp < H )
	{
		float total = 0.0f;
		uint4 v = make_uint4(0u, 0u, 0u, 0u);
		if ( lane < D_V )
		{
			v = src4[warp * D_V + lane];
			total = LmSumSqBf16x8(v, 0.0f);
		}
		const float scale = SparkMathRmsScale(QwenWarpSum(total), (float)D, QH_EPSILON);
		uint4 normed = make_uint4(0u, 0u, 0u, 0u);
		if ( lane < D_V )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(v, x);
			LmLoadWeight8(q_norm, lane << 3u, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathRoundBf16(x[k] * scale * w[k]);
			normed = LmPackUint4FloatToBf16(out);
			if ( lane < (G::kRopeDim >> 3u) )
				*(uint4 *)(s_q[warp] + (lane << 3u)) = normed;
		}
		__syncwarp();
		if ( lane < HALF )
		{
			const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)lane / (float)G::kRopeDim);
			LmRopeRotate<LM_ROPE_HALF_SPLIT, LM_ROPE_FP32>(s_q[warp], 0u, lane, HALF, angle, 1.0f);
		}
		__syncwarp();
		if ( lane < D_V )
			((uint4 *)index_query)[((uint64_t)row * H + warp) * D_V + lane] =
				lane < (G::kRopeDim >> 3u) ? *(const uint4 *)(s_q[warp] + (lane << 3u)) : normed;
	}
	else if ( warp == H && lane < D_V )
		((uint4 *)index_key)[(uint64_t)row * D_V + lane] = src4[H * D_V + lane];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void QwenHybridRowDotKernel(const uint16_t *rows_bf16, const uint16_t *weight, uint16_t *output, uint32_t width)
{
	__shared__ float reduce[LmBlockWarps<THREADS>()];
	const uint32_t row = blockIdx.x;
	const uint16_t *row_ptr = rows_bf16 + (uint64_t)row * width;
	float sum = 0.0f;
	LmDependentWait();
	LmDependentRelease();
	if ( (width & 7u) == 0u && LmIsAligned16(row_ptr) && LmIsAligned16(weight) )
		for ( uint32_t c = threadIdx.x; c < (width >> 3u); c += THREADS )
			sum = LmDotBf16x8(((const uint4 *)row_ptr)[c], ((const uint4 *)weight)[c], sum);
	else
		for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
			sum += LmBf16ToFloat(row_ptr[index]) * LmBf16ToFloat(weight[index]);
	sum = LmBlockSumOnce<THREADS>(sum, reduce);
	if ( threadIdx.x == 0u )
		output[row] = LmFloatToBf16(sum);
}

template<class G>
__global__ static void QwenIndexBlockStoreKernel(const uint16_t *raw, const uint16_t *tail, const float *norm, LmKvShardView view,
	const uint32_t *sequence_of_row, const uint32_t *positions, const uint32_t *row_lane, const uint32_t *row_ordinal,
	const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, float theta)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, R = G::kIndexRatio, HALF = G::kRopeDim / 2u;
	__shared__ float reduction[D / LM_WARP_LANES];
	__shared__ float normed[D];
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.x, d = threadIdx.x, position = positions[row];
	if ( position % R != R - 1u || SparkKvShardOwns(view.shard, position / R) == 0u )
		return;
	uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::IndexKv>(view, sequence_of_row[row], position / R, row, LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
		return;
	const uint32_t lane = row_lane[row], ordinal = row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = positions[lane_rows[0]], tail_base = first / R * R, start = position - (R - 1u);
	const uint16_t *carried = tail + (uint64_t)recurrent_slot[lane] * S::kIndexTail * D;
	float sum = 0.0f;
	for ( uint32_t k = 0u; k < R; ++k )
	{
		const uint32_t at = start + k;
		sum += LmBf16ToFloat(at >= first ? raw[(uint64_t)lane_rows[ordinal - (position - at)] * D + d] : carried[(uint64_t)(at - tail_base) * D + d]);
	}
	const float pooled = LmBf16ToFloat(LmFloatToBf16(sum / (float)R));
	const float scale = SparkMathRmsScale(LmBlockSum<D>(pooled * pooled, reduction), (float)D, QH_EPSILON);
	normed[d] = LmBf16ToFloat(LmFloatToBf16(pooled * scale * norm[d]));
	__syncthreads();
	float out = normed[d];
	if ( d < G::kRopeDim )
	{
		const uint32_t pair = d % HALF;
		const float angle = (float)start * powf(theta, -2.0f * (float)pair / (float)G::kRopeDim);
		const float partner = d < HALF ? -normed[d + HALF] : normed[d - HALF];
		float s, c;
		sincosf(angle, &s, &c);
		out = normed[d] * c + partner * s;
	}
	slot[d] = LmFloatToBf16(out);
}

template<class G>
__global__ static void QwenIndexTailKernel(const uint16_t *raw, uint16_t *tail, const uint32_t *positions, const uint32_t *sequence_row_begin,
	const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, uint32_t commit, const uint32_t *row_count)
{
	constexpr uint32_t D = G::kIndexDim, R = G::kIndexRatio, T = QwenHybridShape<G>::kIndexTail, V = T > 0u ? T : 1u;
	const uint32_t lane = blockIdx.x, d = threadIdx.x;
	LmDependentWait();
	const uint32_t begin = sequence_row_begin[lane], count = row_count != 0 ? row_count[lane] : sequence_row_begin[lane + 1u] - begin;
	if ( commit == 0u || count == 0u )
		return;
	const uint32_t *lane_rows = sequence_row_indices + begin;
	const uint32_t first = positions[lane_rows[0]], end = positions[lane_rows[count - 1u]] + 1u;
	const uint32_t old_base = first / R * R, new_base = end / R * R, keep = end - new_base;
	uint16_t *slot = tail + (uint64_t)recurrent_slot[lane] * T * D;
	if ( (D & 7u) == 0u && LmIsAligned16(raw) && LmIsAligned16(slot) )
	{
		if ( d >= (D >> 3u) ) return;
		constexpr uint32_t D4 = D >> 3u;
		const uint4 *raw4 = (const uint4 *)raw;
		uint4 *slot4 = (uint4 *)slot, v[V];
		#pragma unroll
		for ( uint32_t t = 0u; t < T; ++t )
		{
			const uint32_t at = new_base + t;
			v[t] = t < keep ? (at >= first ? raw4[(uint64_t)lane_rows[at - first] * D4 + d] : slot4[(uint64_t)(at - old_base) * D4 + d]) : make_uint4(0u, 0u, 0u, 0u);
		}
		#pragma unroll
		for ( uint32_t t = 0u; t < T; ++t ) slot4[(uint64_t)t * D4 + d] = v[t];
		return;
	}
	uint16_t values[V];
	#pragma unroll
	for ( uint32_t t = 0u; t < T; ++t )
	{
		const uint32_t at = new_base + t;
		values[t] = t < keep ? (at >= first ? raw[(uint64_t)lane_rows[at - first] * D + d] : slot[(uint64_t)(at - old_base) * D + d]) : (uint16_t)0u;
	}
	#pragma unroll
	for ( uint32_t t = 0u; t < T; ++t ) slot[(uint64_t)t * D + d] = values[t];
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenIndexScoreKernel(LmKvShardView view, const uint16_t *__restrict__ query, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, float *__restrict__ scores,
	uint2 *__restrict__ candidates, uint32_t stride, float scale)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, H = G::kIndexHeads;
	__shared__ alignas(16) float shared_query[H * D];
	const uint32_t row = blockIdx.y, sequence = sequence_of_row[row];
	LmDependentWait();
	uint32_t limit = context_length[sequence];
	if ( positions[row] + 1u < limit )
		limit = positions[row] + 1u;
	const uint32_t blocks = limit / G::kIndexRatio, local_count = SparkKvShardLocalKeys(view.shard, blocks);
	if ( blocks <= S::kIndexBlocks )
	{
		LmDependentRelease();
		return;
	}
	const bool direct_pack = local_count <= S::kIndexBlocks;
	const uint32_t active_limit = direct_pack ? S::kIndexBlocks : local_count;
	if ( blockIdx.x * blockDim.x >= active_limit )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *row_query = query + (uint64_t)row * H * D;
	if ( ((H * D) & 7u) == 0u && LmIsAligned16(row_query) )
	{
		constexpr uint32_t q_chunks = (H * D) >> 3u;
		const uint4 *q4 = (const uint4 *)row_query;
		for ( uint32_t c = threadIdx.x; c < q_chunks; c += blockDim.x )
		{
			float qf[8];
			LmLoadBf16x8AsFloat(q4[c], qf);
			LmStoreFloat8(shared_query + (c << 3u), qf);
		}
	}
	else
	{
		for ( uint32_t index = threadIdx.x; index < H * D; index += blockDim.x )
			shared_query[index] = LmBf16ToFloat(row_query[index]);
	}
	__syncthreads();
	for ( uint32_t local = blockIdx.x * blockDim.x + threadIdx.x; local < active_limit; local += gridDim.x * blockDim.x )
	{
		float score = -INFINITY;
		if ( local < local_count )
		{
			const uint16_t *key = (const uint16_t *)LmKvShardSlotRequired<typename S::IndexKv>(view, sequence,
				SparkKvShardLocalPosition(view.shard, local), row, LM_KV_ACCESS_READ);
			if ( key != 0 )
			{
				if constexpr ( H > 0u && H <= 16u )
				{
					float dots[H];
					#pragma unroll
					for ( uint32_t head = 0u; head < H; ++head )
						dots[head] = 0.0f;
					if ( (D & 7u) == 0u && LmIsAligned16(key) )
					{
						constexpr uint32_t chunks = D >> 3u;
						const uint4 *k4 = (const uint4 *)key;
						for ( uint32_t c = 0u; c < chunks; ++c )
						{
							float kv[8];
							LmLoadBf16x8AsFloat(__ldg(&k4[c]), kv);
							const uint32_t d_base = c << 3u;
							#pragma unroll
							for ( uint32_t head = 0u; head < H; ++head )
							{
								float qv[8], acc = dots[head];
								LmLoadFloat8(shared_query + head * D + d_base, qv);
								#pragma unroll
								for ( uint32_t k = 0u; k < 8u; ++k )
									acc = fmaf(qv[k], kv[k], acc);
								dots[head] = acc;
							}
						}
					}
					else
					{
						for ( uint32_t d = 0u; d < D; ++d )
						{
							const float kv = LmBf16ToFloat(key[d]);
							#pragma unroll
							for ( uint32_t head = 0u; head < H; ++head )
								dots[head] = fmaf(shared_query[head * D + d], kv, dots[head]);
						}
					}
					score = 0.0f;
					#pragma unroll
					for ( uint32_t head = 0u; head < H; ++head )
						score += fmaxf(dots[head], 0.0f);
					score *= scale;
				}
				else
				{
					score = 0.0f;
					for ( uint32_t head = 0u; head < H; ++head )
					{
						float dot = 0.0f;
						for ( uint32_t d = 0u; d < D; ++d )
							dot = fmaf(shared_query[head * D + d], LmBf16ToFloat(key[d]), dot);
						score += fmaxf(dot, 0.0f);
					}
					score *= scale;
				}
			}
		}
		if ( direct_pack )
			candidates[(uint64_t)row * S::kIndexBlocks + local] = make_uint2(__float_as_uint(score),
				local < local_count && score > -INFINITY ? SparkKvShardLocalPosition(view.shard, local) : LM_INDEX_SHARD_NO_POSITION);
		else
			scores[(uint64_t)row * stride + local] = score;
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenIndexLocalTopkPackKernel(SparkKvShard shard, const float *__restrict__ scores, uint32_t stride,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions,
	uint32_t *__restrict__ local_selected, uint2 *__restrict__ candidates)
{
	using S = QwenHybridShape<G>;
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[QH_THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	const uint32_t row = blockIdx.x, sequence = sequence_of_row[row];
	LmDependentWait();
	uint32_t limit = context_length[sequence];
	if ( positions[row] + 1u < limit )
		limit = positions[row] + 1u;
	const uint32_t blocks = limit / G::kIndexRatio, local_count = SparkKvShardLocalKeys(shard, blocks);
	if ( blocks <= S::kIndexBlocks || local_count <= S::kIndexBlocks )
	{
		LmDependentRelease();
		return;
	}
	const float *row_scores = scores + (uint64_t)row * stride;
	uint32_t *out = local_selected + (uint64_t)row * S::kIndexBlocks;
	LmTopkExactSelect<QH_THREADS>(row_scores, local_count, S::kIndexBlocks, out, histogram, scan, state);
	__syncthreads();
	for ( uint32_t entry = threadIdx.x; entry < S::kIndexBlocks; entry += QH_THREADS )
	{
		const uint32_t local = out[entry];
		const float score = local < local_count ? row_scores[local] : -INFINITY;
		candidates[(uint64_t)row * S::kIndexBlocks + entry] = make_uint2(__float_as_uint(score),
			local < local_count && score > -INFINITY ? SparkKvShardLocalPosition(shard, local) : LM_INDEX_SHARD_NO_POSITION);
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 4)
static void QwenIndexGatheredSelectMaskKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride, uint32_t degree,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions,
	uint32_t *__restrict__ mask, uint32_t mask_words)
{
	using S = QwenHybridShape<G>;
	__shared__ uint32_t histogram[256];
	__shared__ uint32_t s_state[6];
	const uint32_t row = blockIdx.x, sequence = sequence_of_row[row];
	LmDependentWait();
	uint32_t limit = context_length[sequence];
	if ( positions[row] + 1u < limit )
		limit = positions[row] + 1u;
	const uint32_t blocks = limit / G::kIndexRatio;
	uint32_t *row_mask = mask + (uint64_t)row * mask_words;
	if ( blocks <= S::kIndexBlocks )
	{
		auto word_bits = [&](uint32_t low) { return blocks >= low + 32u ? 0xffffffffu : blocks > low ? (1u << (blocks - low)) - 1u : 0u; };
		if ( (mask_words & 3u) == 0u && LmIsAligned16(row_mask) )
		{
			uint4 *m4 = (uint4 *)row_mask;
			for ( uint32_t q = threadIdx.x; q < (mask_words >> 2u); q += QH_THREADS )
				m4[q] = make_uint4(word_bits(q * 128u), word_bits(q * 128u + 32u), word_bits(q * 128u + 64u), word_bits(q * 128u + 96u));
		}
		else
			for ( uint32_t word = threadIdx.x; word < mask_words; word += QH_THREADS )
				row_mask[word] = word_bits(word * 32u);
		LmDependentRelease();
		return;
	}
	if ( (mask_words & 3u) == 0u && LmIsAligned16(row_mask) )
	{
		uint4 *m4 = (uint4 *)row_mask;
		for ( uint32_t q = threadIdx.x; q < (mask_words >> 2u); q += QH_THREADS )
			m4[q] = make_uint4(0u, 0u, 0u, 0u);
	}
	else
		for ( uint32_t word = threadIdx.x; word < mask_words; word += QH_THREADS )
			row_mask[word] = 0u;
	const uint32_t total = degree * S::kIndexBlocks;
	if ( threadIdx.x == 0u )
	{
		s_state[0] = 0u;
		s_state[1] = 0u;
		s_state[2] = S::kIndexBlocks;
		s_state[3] = 0u;
		s_state[4] = 0u;
		s_state[5] = 0u;
	}
	__syncthreads();
	for ( uint32_t pass = 0u; pass < 4u; ++pass )
	{
		const uint32_t shift = 24u - 8u * pass;
		histogram[threadIdx.x] = 0u;
		__syncthreads();
		const uint32_t pfx = s_state[0], msk = s_state[1];
		for ( uint32_t idx = threadIdx.x; idx < total; idx += QH_THREADS )
		{
			const uint32_t r = idx / S::kIndexBlocks, e = idx % S::kIndexBlocks;
			const uint2 c = __ldg(&gathered[(uint64_t)r * rank_stride + (uint64_t)row * S::kIndexBlocks + e]);
			if ( c.y < blocks )
			{
				const uint32_t key = LmTopkKey(__uint_as_float(c.x));
				if ( (key & msk) == pfx )
					atomicAdd(&histogram[(key >> shift) & 255u], 1u);
			}
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			if ( pass == 0u )
			{
				uint32_t valid = 0u;
				for ( uint32_t b = 0u; b < 256u; ++b )
					valid += histogram[b];
				if ( valid <= S::kIndexBlocks )
				{
					s_state[5] = 1u;
					s_state[4] = valid;
				}
			}
			if ( s_state[5] == 0u )
			{
				uint32_t running = 0u, bucket = 255u;
				for ( ; bucket > 0u && running + histogram[bucket] < s_state[2]; --bucket )
					running += histogram[bucket];
				s_state[0] |= bucket << shift;
				s_state[1] |= 255u << shift;
				s_state[2] -= running;
				s_state[3] = histogram[bucket];
			}
		}
		__syncthreads();
		if ( s_state[5] != 0u )
			break;
	}
	if ( s_state[5] != 0u )
	{
		for ( uint32_t idx = threadIdx.x; idx < total; idx += QH_THREADS )
		{
			const uint32_t r = idx / S::kIndexBlocks, e = idx % S::kIndexBlocks;
			const uint2 c = __ldg(&gathered[(uint64_t)r * rank_stride + (uint64_t)row * S::kIndexBlocks + e]);
			if ( c.y < blocks )
				atomicOr(&row_mask[c.y >> 5u], 1u << (c.y & 31u));
		}
		__syncthreads();
		if ( threadIdx.x == 0u && s_state[4] < S::kIndexBlocks )
		{
			uint32_t have = s_state[4];
			for ( uint32_t p = 0u; p < blocks && have < S::kIndexBlocks; ++p )
			{
				const uint32_t bit = 1u << (p & 31u);
				if ( (row_mask[p >> 5u] & bit) == 0u )
				{
					row_mask[p >> 5u] |= bit;
					have++;
				}
			}
		}
		LmDependentRelease();
		return;
	}
	const uint32_t thresh = s_state[0], need = s_state[2], eq_count = s_state[3];
	uint32_t pos_thresh = 0xffffffffu;
	if ( eq_count > need )
	{
		if ( threadIdx.x == 0u )
		{
			s_state[1] = 0u;
			s_state[3] = 0u;
			s_state[4] = need;
		}
		__syncthreads();
		for ( uint32_t pass = 0u; pass < 4u; ++pass )
		{
			const uint32_t shift = 24u - 8u * pass;
			histogram[threadIdx.x] = 0u;
			__syncthreads();
			const uint32_t ppfx = s_state[1], pmsk = s_state[3];
			for ( uint32_t idx = threadIdx.x; idx < total; idx += QH_THREADS )
			{
				const uint32_t r = idx / S::kIndexBlocks, e = idx % S::kIndexBlocks;
				const uint2 c = __ldg(&gathered[(uint64_t)r * rank_stride + (uint64_t)row * S::kIndexBlocks + e]);
				if ( c.y < blocks && LmTopkKey(__uint_as_float(c.x)) == thresh && (c.y & pmsk) == ppfx )
					atomicAdd(&histogram[(c.y >> shift) & 255u], 1u);
			}
			__syncthreads();
			if ( threadIdx.x == 0u )
			{
				uint32_t running = 0u, bucket = 0u;
				for ( ; bucket < 255u && running + histogram[bucket] < s_state[4]; ++bucket )
					running += histogram[bucket];
				s_state[1] |= bucket << shift;
				s_state[3] |= 255u << shift;
				s_state[4] -= running;
			}
			__syncthreads();
		}
		pos_thresh = s_state[1];
	}
	for ( uint32_t idx = threadIdx.x; idx < total; idx += QH_THREADS )
	{
		const uint32_t r = idx / S::kIndexBlocks, e = idx % S::kIndexBlocks;
		const uint2 c = __ldg(&gathered[(uint64_t)r * rank_stride + (uint64_t)row * S::kIndexBlocks + e]);
		if ( c.y < blocks )
		{
			const uint32_t key = LmTopkKey(__uint_as_float(c.x));
			if ( key > thresh || (key == thresh && c.y <= pos_thresh) )
				atomicOr(&row_mask[c.y >> 5u], 1u << (c.y & 31u));
		}
	}
	LmDependentRelease();
}

#define QH_MOE_LANE_ROWS 1024u

template<class G>
struct QwenHybridModel : CudaStage<QwenHybridModel<G>>
{
	using S = QwenHybridShape<G>;
	using State = QwenHybridState<G>;
	using Layer = QwenHybridLayer;
	using Schedule = QwenHybridSchedule<G, QwenHybridModel>;
	using KvViews = CudaStageKvViews<typename S::Kv, &State::kv_views, 0u, S::kAttnLayers>;
	using IndexViews = CudaStageKvViews<typename S::IndexKv, &State::index_views, 1u, S::kAttnLayers>;
	using Kv = typename std::conditional<S::kIndexer, CudaStageKv<KvViews, IndexViews>, CudaStageKv<KvViews>>::type;
	using LaneShared = CudaStageFields<&State::gdn_state, &State::gdn_window, &State::ple_window, &State::ple_history, &State::index_tail, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = G::kVerifyRows != 0u;
	static constexpr bool kTreeVerify = !S::kPle && !S::kIndexer && G::kVerifyRows != 0u;
	static constexpr bool kTapPlan = true;
	static constexpr bool kGraphKey = true;
	static constexpr uint32_t kChunkRows = G::kMoe ? QH_MOE_LANE_ROWS : 0u;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = QH_EPSILON;
	static constexpr uint32_t kPackAlignment = QH_PACK_ALIGNMENT;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
	};

	static const char *Tag(void)
	{
		return G::Tag();
	}

	static const char *ReportTag(void)
	{
		return G::ReportTag();
	}

	static const char *WeightdModel(void)
	{
		return G::Name();
	}

	template<class Format>
	static int32_t ExactLinear(const void *weight, const float *, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream)
	{
		const uint32_t chunks = input_dimension / LmSkinnyFormat<Format>::kElements;
		if ( chunks >= 32u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactLinearKernel<LmBf16Format, 32u, 8u, 4u>),
				dim3((output_dimension + 3u) / 4u, (rows + 63u) / 64u), QH_THREADS, 0, stream,
				(const uint8_t *)weight, source, destination, rows, input_dimension, output_dimension);
		else
			(void)LM_LAUNCH_DEPENDENT((QwenExactLinearKernel<LmBf16Format, 16u, 4u, 4u>),
				dim3((output_dimension + 3u) / 4u, (rows + 63u) / 64u), QH_THREADS, 0, stream,
				(const uint8_t *)weight, source, destination, rows, input_dimension, output_dimension);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<class Format>
	static int32_t Project(const void *weight, const float *scale_rows, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, uint32_t multiprocessors, SparkGpuStream stream)
	{
		if constexpr ( std::is_same<Format, LmBf16Format>::value )
		{
			constexpr uint32_t ELEMENTS = LmSkinnyFormat<Format>::kElements;
			if ( exact != 0u && rows > LM_SKINNY_ROWS_WIDE && (input_dimension % ELEMENTS) == 0u &&
				LmSkinnyAligned(weight) != 0u && LmSkinnyAligned(source) != 0u )
			{
				const uint32_t chunks = input_dimension / ELEMENTS;
				if ( chunks >= 16u && (output_dimension > LM_SKINNY_SPLIT_MAX_OUTPUTS || chunks < LM_SKINNY_SPLIT_WARPS * LM_WARP_LANES) )
					return ExactLinear<Format>(weight, 0, source, destination, rows, input_dimension, output_dimension, stream);
			}
		}
		const LmScaleTensor scale = scale_rows != 0 ?
			LmScaleTensorBuild(scale_rows, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u, QH_SCALE_K) : LmScaleTensorNone();
		return LmDenseRows<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension, output_dimension, exact, multiprocessors,
			stream);
	}

	static int32_t Linear(const State *state, const uint8_t *weight, const float *scale_rows, const uint16_t *source,
		uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, uint32_t multiprocessors,
		SparkGpuStream stream)
	{
		return state->linear_bytes == 1u ?
			Project<LmFp8>(weight, scale_rows, source, destination, rows, input_dimension, output_dimension, exact, multiprocessors, stream) :
			Project<LmBf16Format>(weight, 0, source, destination, rows, input_dimension, output_dimension, exact, multiprocessors, stream);
	}

	static uint32_t Exact(const SparkStageRunnerStep *step)
	{
		return step->commit == 0u ? 1u : 0u;
	}

	static SparkStatus GdnRecur(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, const uint16_t *qkv, const uint16_t *ba,
		const uint32_t *kept, uint32_t commit, SparkGpuStream stream, uint32_t qkv_stride = 0u)
	{
		const uint32_t rows = step->rows, sequences = step->sequences, heads = state->gdn_heads;
		const uint32_t tree = step->tree != 0u && commit == 0u ? 1u : 0u;
		uint8_t *state_pool = state->gdn_state + (uint64_t)layer->recurrent_index * state->slots * state->gdn_state_bytes;
		uint16_t *window = state->gdn_window + (uint64_t)layer->recurrent_index * state->slots * (state->gdn_window_bytes / sizeof(uint16_t));
		if ( qkv_stride != 0u && qkv_stride != state->gdn_channels )
		{
			(void)LM_LAUNCH_DEPENDENT((QwenGdnStridedConvTiledKernel<QH_CONV>),
				dim3(sequences, (state->gdn_channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
				LM_CAUSAL_CONV_TILE_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, qkv, layer->gdn_conv,
				state->qkv, state->gdn_channels, qkv_stride, sequences, commit, step->sequence_row_indices);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( LmCausalConvLaunch<QH_CONV,LM_CONV_SWISH,uint16_t>(window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
			state->qkv, state->gdn_channels, sequences, commit, step->sequence_row_indices, rows, stream,
			tree != 0u ? step->parent_rows : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((QwenGdnSplitExpandGateKernel<QH_THREADS,QH_KEY_DIM>), rows, QH_THREADS, 0, stream,
			state->qkv, ba, layer->gdn_a_log, layer->gdn_dt_bias, state->query_heads, state->key_heads, state->value,
			state->retention, state->write_gate, state->gdn_qk, state->gdn_v, heads, rows);
		if ( LmDeltaRuleLaunch<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>(rows, state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
				step->sequence_row_begin, kept, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
				heads, 1u, sequences, commit, step->sequence_row_indices, stream,
				tree != 0u ? step->state_loads : (const uint32_t *)0, tree != 0u ? step->state_saves : (const uint32_t *)0,
				tree != 0u ? state->gdn_snapshots : (float *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return CudaStageLaunched();
	}

	static SparkStatus Gdn(State *state, const Frame &frame, const QwenHybridLayer &entry)
	{
		const QwenHybridLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, heads = state->gdn_heads, exact = Exact(step);
		const uint32_t sms = step->multiprocessors;
		const uint64_t replay = (uint64_t)layer->recurrent_index * G::kVerifyRows;
		const uint32_t qkvz_width = state->gdn_channels + state->gdn_v;
		const bool fused_qkvz = rows > LM_SKINNY_ROWS_WIDE && exact == 0u && step->tree == 0u && qkvz_width <= S::kQkvWidth;
		SparkStatus status;
		if ( exact != 0u && rows > LM_SKINNY_ROWS_WIDE && state->linear_bytes == 2u )
		{
			const uint8_t *z_weight = layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * 2u;
			if ( ExactLinear<LmBf16Format>(layer->mix_weight, 0, state->normed, state->qkv, rows, G::kHidden, state->gdn_channels, stream) != LM_LAUNCH_OK ||
				ExactLinear<LmBf16Format>(z_weight, 0, state->normed, state->z, rows, G::kHidden, state->gdn_v, stream) != LM_LAUNCH_OK ||
				ExactLinear<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( rows <= LM_SKINNY_ROWS_WIDE || exact != 0u )
		{
			if ( state->linear_bytes == 2u )
			{
				const LmSkinnyDenseTarget targets[3] =
				{
					{ layer->mix_weight, state->qkv, (float *)0, state->gdn_channels, state->gdn_channels, 0u, LmScaleTensorNone() },
					{ layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * 2u, state->z, (float *)0, state->gdn_v, state->gdn_v, 0u, LmScaleTensorNone() },
					{ layer->gdn_ba, state->ba, (float *)0, 2u * heads, 2u * heads, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 3u, state->normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				const LmScaleTensor s0 = LmScaleTensorBuild(layer->mix_scale, LM_SCALE_ENCODING_F32, 1u, state->gdn_channels, G::kHidden, 1u, QH_SCALE_K);
				const LmScaleTensor s1 = LmScaleTensorBuild(layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
					LM_SCALE_ENCODING_F32, 1u, state->gdn_v, G::kHidden, 1u, QH_SCALE_K);
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->mix_weight, state->qkv, (float *)0, state->gdn_channels, state->gdn_channels, 0u, s0 },
					{ layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden, state->z, (float *)0, state->gdn_v, state->gdn_v, 0u, s1 }
				};
				if ( LmSkinnyDenseMulti<LmFp8>(targets, 2u, state->normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK ||
					Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, exact, sms, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else if ( fused_qkvz )
		{
			if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->attn_gathered, rows, G::kHidden, qkvz_width, 0u, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( rows <= 128u && 2u * heads <= 32u && (G::kHidden & 7u) == 0u )
				(void)LM_LAUNCH_DEPENDENT((QwenSmallNBf16ProjKernel<G::kHidden>), rows, QH_THREADS, 0, stream, layer->gdn_ba, state->normed, state->ba, rows, 2u * heads);
			else if ( Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, 0u, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->qkv, rows, G::kHidden, state->gdn_channels, exact, sms, stream) != LM_LAUNCH_OK ||
				Linear(state, layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * state->linear_bytes,
					layer->mix_scale == 0 ? (const float *)0 : layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
					state->normed, state->z, rows, G::kHidden, state->gdn_v, exact, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( rows <= 128u && 2u * heads <= 32u && exact == 0u && (G::kHidden & 7u) == 0u )
				(void)LM_LAUNCH_DEPENDENT((QwenSmallNBf16ProjKernel<G::kHidden>), rows, QH_THREADS, 0, stream, layer->gdn_ba, state->normed, state->ba, rows, 2u * heads);
			else if ( Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, exact, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( step->commit == 0u &&
			(SparkGpuMemcpyAsync(state->replay_qkv + replay * state->gdn_channels, state->qkv, (uint64_t)rows * state->gdn_channels * sizeof(uint16_t),
				SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpyAsync(state->replay_ba + replay * 2u * heads, state->ba, (uint64_t)rows * 2u * heads * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE,
				stream) != SPARK_GPU_SUCCESS) )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		status = GdnRecur(state, layer, step,
			fused_qkvz ? state->attn_gathered : (step->tree != 0u ? state->replay_qkv + replay * state->gdn_channels : state->qkv),
			state->ba, (const uint32_t *)0, step->commit, stream, fused_qkvz ? qkvz_width : 0u);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( status != SPARK_STATUS_OK )
			return status;
		(void)LM_LAUNCH_DEPENDENT((QwenGdnHeadRmsNormGateKernel<G::kGdnSigmoidGate>), dim3((heads + 7u) >> 3u, rows), QH_THREADS, 0, stream,
			state->mixed, fused_qkvz ? state->attn_gathered : state->z, layer->gdn_norm, state->mixed, rows, heads,
			fused_qkvz ? qkvz_width : state->gdn_v, fused_qkvz ? state->gdn_channels : 0u);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->mixed, state->partial, rows, state->gdn_v, G::kHidden, exact, sms, stream) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus IndexPrepare(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		uint16_t *tail = state->index_tail + (uint64_t)layer->kv_index * state->slots * S::kIndexTail * G::kIndexDim;
		if ( Project<LmBf16Format>(layer->index_qk, 0, state->normed, state->index_qk, rows, G::kHidden, S::kIndexWidth, Exact(step), sms, stream) !=
			LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kIndexDim <= 256u && (G::kIndexDim & 7u) == 0u && G::kIndexHeads + 1u <= QH_THREADS / LM_WARP_LANES && G::kRopeDim == 64u )
		{
			(void)LM_LAUNCH_DEPENDENT((QwenIndexQPrepKernel<G>), rows, QH_THREADS, 0, stream, state->index_qk, layer->index_q_norm, step->positions,
				state->index_query, state->index_key, rows);
		}
		else
		{
			LM_LAUNCH((LmSplit2RowsKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->index_qk, state->index_query, state->index_key, rows,
				G::kIndexHeads * G::kIndexDim, G::kIndexDim);
			LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kIndexHeads,rows), QH_THREADS, 0, stream,
				state->index_query, layer->index_q_norm, state->index_query, rows, G::kIndexHeads, G::kIndexDim, QH_EPSILON, 1.0f);
			LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kIndexHeads), QH_THREADS, 0, stream,
				state->index_query, step->positions, G::kIndexHeads, G::kIndexDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
		}
		if ( step->commit == 0u && SparkGpuMemcpyAsync(state->replay_index + (uint64_t)layer->kv_index * G::kVerifyRows * G::kIndexDim, state->index_key,
			(uint64_t)rows * G::kIndexDim * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		(void)LM_LAUNCH_DEPENDENT((QwenIndexBlockStoreKernel<G>), rows, G::kIndexDim, 0, stream, state->index_key, tail, layer->index_k_norm,
			state->index_views[layer->kv_index], step->sequence_of_row, step->positions, state->row_lane, state->row_ordinal,
			step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, G::kRopeTheta);
		(void)LM_LAUNCH_DEPENDENT((QwenIndexTailKernel<G>), step->sequences, G::kIndexDim, 0, stream, state->index_key, tail, step->positions, step->sequence_row_begin,
			step->sequence_row_indices, step->recurrent_index, step->commit, (const uint32_t *)0);
		return CudaStageLaunched();
	}

	static SparkStatus IndexSelect(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows;
		const LmKvShardView *view = &state->index_views[layer->kv_index];
		SparkStatus status;
		(void)LM_LAUNCH_DEPENDENT((QwenIndexScoreKernel<G>), dim3((state->index_local_stride + QH_THREADS - 1u) / QH_THREADS, rows), QH_THREADS, 0, stream, *view,
			state->index_query, step->sequence_of_row, step->context_length, step->positions, state->index_scores, state->index_candidates, state->index_local_stride,
			1.0f / sqrtf((float)G::kIndexDim));
		(void)LM_LAUNCH_DEPENDENT((QwenIndexLocalTopkPackKernel<G>), rows, QH_THREADS, 0, stream, view->shard, state->index_scores, state->index_local_stride,
			step->sequence_of_row, step->context_length, step->positions, state->index_local_selected, state->index_candidates);
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows,
			S::kIndexBlocks * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->index_candidates, state->index_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		(void)LM_LAUNCH_DEPENDENT((QwenIndexGatheredSelectMaskKernel<G>), rows, QH_THREADS, 0, stream, state->index_gathered,
			(uint64_t)rows * S::kIndexBlocks, state->tp_degree, step->sequence_of_row, step->context_length, step->positions,
			state->index_mask, state->index_mask_words);
		return CudaStageLaunched();
	}

	static SparkStatus Attention(State *state, const Frame &frame, const QwenHybridLayer &entry)
	{
		const QwenHybridLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->attn_local, rows, G::kHidden,
			state->attn_rows, Exact(step), sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * S::kGatherParts, state->attn_rows / S::kGatherParts,
			state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		{
			constexpr uint32_t tasks = G::kHeads + 2u * G::kKvHeads;
			constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((QwenHybridAttnPrepKernel<G>), dim3((tasks + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
				state->attn_gathered, layer->q_norm, layer->k_norm, step->positions, step->tree != 0u ? step->slots : step->positions,
				step->sequence_of_row, *view, state->attn_query, state->merged_gate, rows, state->attn_rows, state->tp_rank, state->attn_width);
		}
		if constexpr ( S::kIndexer )
		{
			status = IndexPrepare(state, layer, step, stream);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( status == SPARK_STATUS_OK )
				status = IndexSelect(state, layer, step, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		else
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kKvHeads,G::kHeadDim,G::kHeadDim>(*view, state->attn_query, G::kHeads,
				step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kHeadDim), state->send, stride, rows, stream,
				S::kIndexer ? state->index_mask : (const uint32_t *)0, state->index_mask_words, G::kIndexRatio, 0u,
				step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
				step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kHeadDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((QwenHybridOutputGateKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->merged, state->merged_gate, state->attn_width);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, Exact(step), sms,
				stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static LmLocalExpertRoutes ExpertRoutes(const State *state, uint32_t rows, uint32_t routes, uint32_t sms)
	{
		LmLocalExpertRoutes expert_routes;
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = G::kTopK;
		expert_routes.local_experts = G::kLocalExperts;
		expert_routes.multiprocessors = sms;
		return expert_routes;
	}

	static SparkStatus Moe(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, uint32_t first, uint32_t rows,
		SparkGpuStream stream)
	{
		const uint32_t sms = step->multiprocessors, routes = rows * G::kTopK, exact = Exact(step);
		const uint16_t *normed = state->normed + (uint64_t)first * G::kHidden;
		using ExpertFormat = typename LmWeightCodec<G::kExpertCodec>::Format;
		const LmScaleTensor w2_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w2_scale, G::kLocalExperts, G::kHidden, G::kMoeInter);
		const LmLocalExpertRoutes expert_routes = ExpertRoutes(state, rows, routes, sms);
		const bool multi_shared_done = rows <= LM_SKINNY_ROWS_WIDE || exact != 0u;
		int32_t launch;
		if ( exact != 0u && rows > LM_SKINNY_ROWS_WIDE )
		{
			if ( ExactLinear<LmBf16Format>(layer->moe_router, 0, normed, state->router_logits, rows, G::kHidden, G::kExperts, stream) != LM_LAUNCH_OK ||
				ExactLinear<LmBf16Format>(layer->moe_shared_gate_up, 0, normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( multi_shared_done )
		{
			const LmSkinnyDenseTarget targets[2] =
			{
				{ layer->moe_router, state->router_logits, (float *)0, G::kExperts, G::kExperts, 0u, LmScaleTensorNone() },
				{ layer->moe_shared_gate_up, state->shared_gate_up, (float *)0, 2u * G::kSharedRows, 2u * G::kSharedRows, 0u, LmScaleTensorNone() }
			};
			if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Project<LmBf16Format>(layer->moe_router, 0, normed, state->router_logits, rows, G::kHidden, G::kExperts, exact, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kExperts <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES )
		{
			if ( routes <= 32u )
			{
				const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
				const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
				const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
				(void)LM_LAUNCH_DEPENDENT((QwenMoeRouteSmallKernel<G>), 1u, QH_THREADS, 0, stream, state->router_logits, rows, state->expert_first,
					step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
					state->route_packed_row, state->route_source_token, state->packed_expert, state->packed_weight, tile_m, n_up,
					state->tile_prefix_up, n_down, state->tile_prefix_down);
			}
			else
			{
				(void)LM_LAUNCH_DEPENDENT((QwenMoeSoftmaxTopkLocalKernel<G>), rows, QH_THREADS, 0, stream, state->router_logits, state->expert_first,
					step->route_fault, state->route_expert, state->route_weight, state->route_global);
				const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
				const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
				const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
				(void)LM_LAUNCH_DEPENDENT((QwenRouteBuildPackedKernel<QH_THREADS, G::kLocalExperts + 1u>), 1u, QH_THREADS, 0, stream,
					state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row,
					state->route_source_token, state->packed_expert, state->packed_weight, tile_m, n_up, state->tile_prefix_up,
					n_down, state->tile_prefix_down);
			}
		}
		else
		{
			LM_LAUNCH((LmRouterSoftmaxKernel<QH_THREADS,G::kExperts>), rows, QH_THREADS, 0, stream, state->router_logits, state->router_probs);
			if ( LmTopkRouteLaunch<QH_THREADS,G::kTopK,true,LM_TOPK_SCORE_IDENTITY>(rows, state->router_probs, G::kExperts, state->route_expert,
				state->route_weight, (const float *)0, (const uint16_t *)0, 1.0f, step->route_fault, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			LM_LAUNCH((LmRouteLocalKernel), (routes + 255u) / 256u, 256u, 0, stream, state->route_expert, state->route_weight, state->route_global,
				routes, state->expert_first, G::kLocalExperts);
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((QwenRouteBuildPackedKernel<QH_THREADS, G::kLocalExperts + 1u>), 1u, QH_THREADS, 0, stream,
				state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row,
				state->route_source_token, state->packed_expert, state->packed_weight, tile_m, n_up, state->tile_prefix_up,
				n_down, state->tile_prefix_down);
		}
		if constexpr ( G::kExpertCodec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 && (G::kHidden % 128u) == 0u && (G::kMoeInter % 128u) == 0u )
		{
			constexpr uint32_t UP_CHUNKS = G::kHidden >> 5u;
			constexpr uint32_t UP_LANES = UP_CHUNKS >= 32u ? 32u : (UP_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t UP_PER_BLK = QH_THREADS / UP_LANES;
			constexpr uint32_t DN_CHUNKS = G::kMoeInter >> 5u;
			constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t DN_PER_BLK = QH_THREADS / DN_LANES;
			constexpr uint32_t DN_SINGLE_PER_BLK = DN_PER_BLK * 4u;
			if ( exact == 0u && rows > 128u )
			{
				const uint32_t up_blocks = G::kLocalExperts * ((G::kMoeInter + 63u) / 64u);
				if ( rows >= 512u )
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<3u, 8u, false, true, 2u>), up_blocks, 256u, 0, stream,
						layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight, state->expert_act,
						state->group_row_offset, state->route_source_token, G::kLocalExperts, G::kHidden, G::kMoeInter);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<2u, 8u, false, true, 4u>), up_blocks, 256u, 0, stream,
						layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight, state->expert_act,
						state->group_row_offset, state->route_source_token, G::kLocalExperts, G::kHidden, G::kMoeInter);
			}
			else if ( rows == 1u )
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW1W3SingleKernel<G::kHidden, G::kMoeInter, G::kLocalExperts>),
					dim3((G::kMoeInter + UP_PER_BLK - 1u) / UP_PER_BLK, routes), QH_THREADS, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight,
					state->group_row_offset, state->packed_expert, state->expert_act);
			else
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW1W3BatchedKernel<G::kHidden, G::kMoeInter, G::kLocalExperts>),
					dim3((G::kMoeInter + UP_PER_BLK - 1u) / UP_PER_BLK, G::kLocalExperts), QH_THREADS, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight,
					state->group_row_offset, state->route_source_token, state->expert_act);
			if ( exact == 0u && rows > 32u )
			{
				const uint32_t dn_blocks = G::kLocalExperts * ((G::kHidden + 127u) / 128u);
				if ( rows >= 512u )
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<3u, 8u, true, false, 2u>), dn_blocks, 256u, 0, stream,
						layer->moe_w2, layer->moe_w2_scale, (const uint8_t *)0, (const uint8_t *)0, state->expert_act, (const float *)0,
						state->expert_out, state->group_row_offset, (const uint32_t *)0, G::kLocalExperts, G::kMoeInter, G::kHidden);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<2u, 8u, true, false, 4u>), dn_blocks, 256u, 0, stream,
						layer->moe_w2, layer->moe_w2_scale, (const uint8_t *)0, (const uint8_t *)0, state->expert_act, (const float *)0,
						state->expert_out, state->group_row_offset, (const uint32_t *)0, G::kLocalExperts, G::kMoeInter, G::kHidden);
			}
			else if ( rows == 1u )
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW2SingleKernel<G::kMoeInter, G::kHidden, G::kLocalExperts>),
					dim3((G::kHidden + DN_SINGLE_PER_BLK - 1u) / DN_SINGLE_PER_BLK, routes), QH_THREADS, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, state->expert_act, state->group_row_offset, state->packed_expert, state->expert_out);
			else
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW2BatchedKernel<G::kMoeInter, G::kHidden, G::kLocalExperts>),
					dim3((G::kHidden + DN_PER_BLK - 1u) / DN_PER_BLK, G::kLocalExperts), QH_THREADS, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, state->expert_act, state->group_row_offset, state->expert_out);
		}
		else
		{
			launch = LmLocalExpertsUp<ExpertFormat>(expert_routes, layer->moe_w1,
				LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w1_scale, G::kLocalExperts, G::kMoeInter, G::kHidden), normed,
				state->expert_gate, G::kHidden, G::kMoeInter, stream, exact);
			if ( launch == LM_LAUNCH_OK )
				launch = LmLocalExpertsUp<ExpertFormat>(expert_routes, layer->moe_w3,
					LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w3_scale, G::kLocalExperts, G::kMoeInter, G::kHidden), normed,
					state->expert_up, G::kHidden, G::kMoeInter, stream, exact);
			if ( launch != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((QwenMoeSwigluPackedKernel), dim3(LmGridBlocks<QH_THREADS>(G::kMoeInter), routes), QH_THREADS, 0, stream,
				state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter);
			if ( LmLocalExpertsDown<ExpertFormat>(expert_routes, layer->moe_w2, w2_scale, state->expert_act, state->expert_out, G::kMoeInter, G::kHidden,
				stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( !multi_shared_done )
		{
			if ( Project<LmBf16Format>(layer->moe_shared_gate_up, 0, normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, exact, sms, stream) !=
				LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		(void)LM_LAUNCH_DEPENDENT((QwenHybridRowDotKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, normed, layer->moe_shared_gate, state->shared_logit, G::kHidden);
		(void)LM_LAUNCH_DEPENDENT((QwenSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Project<LmBf16Format>(layer->moe_shared_down, 0, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, exact, sms, stream) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((QwenMoeLocalFinalizeGatedKernel), dim3(LmVecBlocks<QH_THREADS>(G::kHidden), rows), QH_THREADS, 0, stream,
			state->expert_out, state->route_packed_row, state->route_expert, state->shared_out, state->shared_logit,
			state->partial + (uint64_t)first * G::kHidden, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus Ffn(State *state, const Frame &frame, const QwenHybridLayer &entry)
	{
		const QwenHybridLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		if constexpr ( G::kMoe )
			return Moe(state, layer, step, 0u, step->rows, stream);
		else
		{
			const uint32_t rows = step->rows, sms = step->multiprocessors, exact = Exact(step);
			if ( Linear(state, layer->ffn_gate_up, layer->ffn_gate_up_scale, state->normed, state->gate_up, rows, G::kHidden,
				2u * state->ffn_rows, exact, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((QwenSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows);
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				Linear(state, layer->ffn_down, layer->ffn_down_scale, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, exact, sms,
					stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return SPARK_STATUS_OK;
		}
	}

	static SparkStatus DumpLayer(State *state, uint32_t layer, const SparkStageRunnerStep *step, const uint16_t *hidden,
		const uint16_t *partial, SparkGpuStream stream)
	{
		const uint32_t width = S::kHyper && layer != 0xffffffffu ? S::kResidual : G::kHidden;
		const uint32_t header[4] = { S::kHyper ? 0x484c5751u : 0x444c5751u, layer, step->rows, width };
		const uint64_t values = (uint64_t)step->rows * width, partials = (uint64_t)step->rows * G::kHidden;
		if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpy(state->dump_positions, step->positions, (size_t)step->rows * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpy(state->dump_rows, hidden, (size_t)values * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS ||
			(partial != 0 && SparkGpuMemcpy(state->dump_rows + values, partial, (size_t)partials * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS) )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( partial == 0 )
			memset(state->dump_rows + values, 0, (size_t)partials * sizeof(uint16_t));
		fwrite(header, sizeof(uint32_t), S::kHyper ? 4u : 3u, state->dump);
		fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		if ( S::kHyper && layer == 0xffffffffu )
		{
			if ( step->token_ids == 0 ||
				SparkGpuMemcpy(state->dump_positions, step->token_ids, (size_t)step->rows * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		}
		fwrite(state->dump_rows, sizeof(uint16_t), (size_t)(values + partials), state->dump);
		fflush(state->dump);
		return SPARK_STATUS_OK;
	}

	static SparkStatus HcPre(State *state, const float *norm, const uint16_t *down, const uint16_t *up, const uint16_t *inject,
		uint16_t *mixed, uint32_t rows, uint32_t exact, uint32_t sms, SparkGpuStream stream)
	{
		if constexpr ( S::kHyper )
		{
			const bool exact_wide = exact != 0u && rows > LM_SKINNY_ROWS_WIDE;
			const bool skinny = rows <= LM_SKINNY_ROWS_WIDE;
			const bool fuse_down_inject = inject != 0 && skinny;
			if ( state->hc_pending_inject != 0u )
			{
				state->hc_pending_inject = 0u;
				(void)LM_LAUNCH_DEPENDENT((QwenHybridHcInjectNormKernel<QH_THREADS, G::kHidden>), dim3(G::kStreams, rows), QH_THREADS, 0, stream,
					state->residual, state->partial, state->hc_inject, norm, state->hc_normed, G::kStreams);
			}
			else
				LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->residual, norm, state->hc_normed,
					G::kStreams, G::kHidden, QH_EPSILON);
			if ( fuse_down_inject )
			{
				constexpr uint32_t TOTAL_N = G::kHcLowRank + G::kStreams;
				constexpr uint32_t N_BLOCKS = (TOTAL_N + 7u) / 8u;
				if ( rows == 1u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 1u>), N_BLOCKS, 128u, 0, stream,
						down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
				else if ( rows <= 4u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 4u>), N_BLOCKS, 128u, 0, stream,
						down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 8u>),
						N_BLOCKS * ((rows + 7u) / 8u), 128u, 0, stream, down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
			}
			else
			{
				if ( exact_wide && inject != 0 )
				{
					if ( ExactLinear<LmBf16Format>(down, 0, state->hc_normed, state->hc_low, rows, S::kResidual, G::kHcLowRank, stream) != LM_LAUNCH_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else if ( Project<LmBf16Format>(down, 0, state->hc_normed, state->hc_low, rows, S::kResidual, G::kHcLowRank, exact, sms, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				LM_LAUNCH((LmHyperGateKernel<QH_THREADS>), LmGridBlocks<QH_THREADS>((uint64_t)rows * G::kHcLowRank), QH_THREADS, 0, stream, state->hc_low,
					(uint64_t)rows * G::kHcLowRank, (float)G::kStreams);
			}
			if ( skinny )
			{
				constexpr uint32_t CHUNKS = G::kHcLowRank >> 3u;
				constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t GROUPS = QH_THREADS / LANES;
				constexpr uint32_t COL_BLOCKS = (G::kHidden + GROUPS - 1u) / GROUPS;
				if ( rows == 1u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixKernel<G::kStreams, G::kHidden, G::kHcLowRank, 1u>), COL_BLOCKS, QH_THREADS, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows);
				else if ( rows <= 4u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixKernel<G::kStreams, G::kHidden, G::kHcLowRank, 4u>), COL_BLOCKS, QH_THREADS, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixKernel<G::kStreams, G::kHidden, G::kHcLowRank, 8u>),
						COL_BLOCKS * ((rows + 7u) / 8u), QH_THREADS, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
			}
			else
			{
				if ( Project<LmBf16Format>(up, 0, state->hc_low, state->hc_up, rows, G::kHcLowRank, S::kResidual, exact, sms, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				(void)LM_LAUNCH_DEPENDENT((QwenHyperMixKernel<QH_THREADS, G::kStreams, G::kHidden>), rows, QH_THREADS, 0, stream,
					state->hc_up, state->hc_normed, mixed);
			}
			if ( inject != 0 && !fuse_down_inject )
			{
				if ( exact == 0u && rows > LM_HYPER_INJECT_CLUSTER_ROWS && ((S::kResidual & 7u) == 0u) )
					(void)LM_LAUNCH_DEPENDENT((QwenHyperInjectWeightsVecKernel<QH_THREADS, G::kStreams>), rows, QH_THREADS, 0, stream,
						state->hc_normed, inject, state->hc_inject, G::kHidden);
				else
					LmHyperInjectWeightsLaunch<QH_THREADS, G::kStreams>(rows, stream, state->hc_normed, inject, state->hc_inject, G::kHidden);
			}
		}
		return CudaStageLaunched();
	}

	static SparkStatus HcInject(State *state, uint32_t rows, SparkGpuStream stream)
	{
		LM_LAUNCH((LmHyperInjectKernel<QH_THREADS>), dim3(LmGridBlocks<QH_THREADS>(S::kResidual), rows), QH_THREADS, 0, stream, state->residual, state->partial,
			state->hc_inject, G::kStreams, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus PleRecur(State *state, const SparkStageRunnerStep *step, const uint16_t *conv_in, const uint32_t *kept, uint32_t commit,
		SparkGpuStream stream)
	{
		if ( LmCausalConvLaunch<G::kPleConv,LM_CONV_SWISH,uint16_t,G::kPleDilation>(state->ple_window, step->recurrent_index, step->sequence_row_begin,
			kept, conv_in, state->ple_conv_weight, state->ple_conv_out, S::kResidual, step->sequences, commit, step->sequence_row_indices, step->rows,
			stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmNgramHistoryKernel), (step->sequences + 63u) / 64u, 64u, 0, stream, step->token_ids, (const int32_t *)0, step->sequence_row_begin,
			step->sequence_row_indices, step->recurrent_index, state->ple_history, S::kPleHistory, step->sequences, G::kPleOrders, commit, kept);
		return CudaStageLaunched();
	}

	static SparkStatus Ple(State *state, const Frame &frame)
	{
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, exact = Exact(step);
		const uint64_t elements = (uint64_t)rows * S::kResidual;
		SparkStatus status;
		if ( step->token_ids == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( state->hc_pending_inject != 0u )
		{
			state->hc_pending_inject = 0u;
			status = HcInject(state, rows, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		CudaStageRowState(state, G::kPleLayer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LM_LAUNCH((LmNgramHashKernel), (rows + 63u) / 64u, 64u, 0, stream, step->token_ids, (const int32_t *)0, step->positions, state->row_lane,
			state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, state->ple_history, S::kPleHistory,
			state->ple_multipliers, state->ple_vocab_sizes, state->ple_offsets, state->ple_ids, rows, G::kPleOrders, G::kPleHeadsPerOrder,
			(int32_t)G::kPleEos, (int32_t)G::kPleEos, 1u);
		if constexpr ( G::kPleTableCodec == SPARK_WEIGHT_CODEC_BF16 )
			LM_LAUNCH((LmNgramGatherBf16Kernel), dim3(rows, S::kPleColumns), G::kPleHeadDim, 0, stream, state->ple_ids, (const uint16_t *)state->ple_table,
				state->ple_embed, state->ple_table_first, state->ple_table_rows, S::kPleColumns, G::kPleHeadDim);
		else
			LM_LAUNCH((LmNgramGatherScaledKernel), dim3(rows, S::kPleColumns), G::kPleHeadDim, 0, stream, state->ple_ids, state->ple_table, state->ple_table_scale,
				state->ple_embed, state->ple_table_first, state->ple_table_rows, S::kPleColumns, G::kPleHeadDim);
		status = CudaStageLaunched();
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->ple_embed, state->ple_embed);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( exact != 0u && rows > LM_SKINNY_ROWS_WIDE )
		{
			if ( ExactLinear<LmBf16Format>(state->ple_key_weight, 0, state->ple_embed, state->ple_key, rows, G::kHidden, S::kResidual, stream) != LM_LAUNCH_OK ||
				ExactLinear<LmBf16Format>(state->ple_value_weight, 0, state->ple_embed, state->ple_value, rows, G::kHidden, G::kHidden, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( rows <= LM_SKINNY_ROWS_WIDE || exact != 0u )
		{
			const LmSkinnyDenseTarget targets[2] =
			{
				{ state->ple_key_weight, state->ple_key, (float *)0, S::kResidual, S::kResidual, 0u, LmScaleTensorNone() },
				{ state->ple_value_weight, state->ple_value, (float *)0, G::kHidden, G::kHidden, 0u, LmScaleTensorNone() }
			};
			if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->ple_embed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Project<LmBf16Format>(state->ple_key_weight, 0, state->ple_embed, state->ple_key, rows, G::kHidden, S::kResidual, exact, sms, stream) !=
				LM_LAUNCH_OK ||
			Project<LmBf16Format>(state->ple_value_weight, 0, state->ple_embed, state->ple_value, rows, G::kHidden, G::kHidden, exact, sms, stream) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( (G::kHidden & 7u) == 0u && G::kHidden <= QH_THREADS * LM_RMS_NORM_STAGED )
			(void)LM_LAUNCH_DEPENDENT((QwenPleKeyQueryGateNormKernel<QH_THREADS, G::kHidden>), dim3(G::kStreams, rows), QH_THREADS, 0, stream,
				state->ple_key, state->residual, state->ple_value, state->ple_norm_key, state->ple_norm_query, state->ple_norm_conv,
				state->ple_gated, state->ple_conv_in, G::kStreams);
		else
		{
			LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->ple_key, state->ple_norm_key, state->ple_key,
				G::kStreams, G::kHidden, QH_EPSILON);
			LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->residual, state->ple_norm_query, state->ple_query,
				G::kStreams, G::kHidden, QH_EPSILON);
			LM_LAUNCH((LmHyperKeyGateKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->ple_key, state->ple_query, state->ple_value,
				state->ple_gated, G::kStreams, G::kHidden);
			LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->ple_gated, state->ple_norm_conv, state->ple_conv_in,
				G::kStreams, G::kHidden, QH_EPSILON);
		}
		if ( step->commit == 0u && SparkGpuMemcpyAsync(state->replay_ple, state->ple_conv_in, elements * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) !=
			SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		status = PleRecur(state, step, state->ple_conv_in, (const uint32_t *)0, step->commit, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
		LM_LAUNCH((LmHyperAddKernel<QH_THREADS>), LmGridBlocks<QH_THREADS>(elements), QH_THREADS, 0, stream, state->residual, state->ple_gated, state->ple_conv_out, elements);
		CudaStageRowState(state, G::kPleLayer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		return CudaStageLaunched();
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream_void, Frame *frame)
	{
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		SparkStatus status = SPARK_STATUS_OK;
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots || (step->commit == 0u && step->rows > G::kVerifyRows) )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( step->tree != 0u && (!kTreeVerify || step->commit != 0u || step->sequences != 1u || state->gdn_snapshots == 0 || step->slots == 0 ||
			step->fork_positions == 0 || step->slot_offsets == 0 || step->parent_rows == 0 || step->state_loads == 0 || step->state_saves == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		state->hc_pending_inject = 0u;
		frame->step = step;
		frame->stream = stream;
		if ( state->dump != 0 )
		{
			SparkGpuCaptureStatus capturing = SPARK_GPU_CAPTURE_STATUS_NONE;
			if ( SparkGpuStreamIsCapturing(stream, &capturing) != SPARK_GPU_SUCCESS || capturing != SPARK_GPU_CAPTURE_STATUS_NONE )
				SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
			status = DumpLayer(state, 0xffffffffu, step, step->hidden_bf16, 0, stream);
		}
		return status;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		if constexpr ( (G::kHidden & 7u) == 0u && G::kHidden <= QH_THREADS * 32u )
			(void)LM_LAUNCH_DEPENDENT((QwenRmsNormKernel<G::kHidden, false>), frame.step->rows, QH_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, (const uint16_t *)0, weight, state->normed, QH_EPSILON);
		else
			LM_LAUNCH((LmFusedResidualRmsNormKernel<QH_THREADS,float>), frame.step->rows, QH_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
				frame.step->hidden_bf16, (const uint16_t *)0, weight, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, QH_EPSILON);
		return CudaStageLaunched();
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, const float *weight)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		if constexpr ( (G::kHidden & 7u) == 0u && G::kHidden <= QH_THREADS * 32u )
			(void)LM_LAUNCH_DEPENDENT((QwenRmsNormKernel<G::kHidden, true>), frame.step->rows, QH_THREADS, 0, frame.stream,
				hidden, state->partial, weight, state->normed, QH_EPSILON);
		else
			LM_LAUNCH((LmFusedResidualRmsNormKernel<QH_THREADS,float>), frame.step->rows, QH_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
				hidden, state->partial, weight, hidden, state->normed, G::kHidden, G::kHidden, QH_EPSILON);
		return CudaStageLaunched();
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		SparkStatus status = SPARK_STATUS_OK;
		if ( state->dump != 0 )
		{
			const uint16_t *rows;
			if constexpr ( S::kHyper )
				rows = state->residual;
			else
				rows = frame.step->hidden_bf16;
			status = DumpLayer(state, index, frame.step, rows, state->partial, frame.stream);
		}
		if ( status == SPARK_STATUS_OK && state->tap_slab != 0 && state->tap_output[index] >= 0 )
		{
			LM_LAUNCH((LmAddRowsStridedKernel<QH_THREADS>), dim3(LmGridBlocks<QH_THREADS>(G::kHidden), frame.step->rows), QH_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, state->tap_slab + (uint64_t)frame.step->dispatch_row_offset * state->tap_stride +
				(uint64_t)state->tap_output[index] * G::kHidden, frame.step->rows, G::kHidden, state->tap_stride);
			status = CudaStageLaunched();
		}
		if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, index);
		return status;
	}

	static SparkStatus AddRows(State *state, const Frame &frame)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		LM_LAUNCH((LmAddElementsKernel), LmVecBlocks<QH_THREADS>(width), QH_THREADS, 0, frame.stream, frame.step->hidden_bf16, state->partial, width);
		return SPARK_STATUS_OK;
	}

	static SparkStatus HyperBegin(State *state, const Frame &frame)
	{
		const SparkStageRunnerStep *step = frame.step;
		LM_LAUNCH((LmHyperExpandKernel<QH_THREADS>), dim3(LmGridBlocks<QH_THREADS>(S::kResidual), step->rows), QH_THREADS, 0, frame.stream, step->hidden_bf16, state->residual,
			G::kStreams, G::kHidden);
		if constexpr ( S::kPle || S::kIndexer )
			LM_LAUNCH((LmRowLanesKernel), step->sequences, QH_THREADS, 0, frame.stream, step->sequence_row_begin, step->sequence_row_indices, state->row_lane,
				state->row_ordinal);
		return CudaStageLaunched();
	}

	static SparkStatus HyperPre(State *state, const Frame &frame, const QwenHybridLayer &layer, uint32_t part)
	{
		return HcPre(state, layer.hc_norm[part], layer.hc_down[part], layer.hc_up[part], layer.hc_inject[part], state->normed, frame.step->rows,
			Exact(frame.step), frame.step->multiprocessors, frame.stream);
	}

	static SparkStatus HyperPost(State *state, const Frame &frame)
	{
		if ( state->dump == 0 )
		{
			state->hc_pending_inject = 1u;
			return SPARK_STATUS_OK;
		}
		return HcInject(state, frame.step->rows, frame.stream);
	}

	static SparkStatus HyperEnd(State *state, const Frame &frame)
	{
		return HcPre(state, state->final_hc_norm, state->final_hc_down, state->final_hc_up, (const uint16_t *)0, frame.step->hidden_bf16,
			frame.step->rows, Exact(frame.step), frame.step->multiprocessors, frame.stream);
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, slots = lane == 0u ? state->slots : 0u, routes = rows * (G::kMoe ? G::kTopK : 0u);
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t), replay = lane == 0u ? G::kVerifyRows : 0u;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->qkv, rows * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->z, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->ba, rows * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->query, rows * state->gdn_qk * sizeof(uint16_t) },
			{ (void **)&state->key, rows * state->gdn_qk * sizeof(uint16_t) },
			{ (void **)&state->value, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->query_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->key_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->retention, rows * state->gdn_heads * QH_KEY_DIM * sizeof(float) },
			{ (void **)&state->write_gate, rows * state->gdn_heads * sizeof(float) },
			{ (void **)&state->mixed, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->attn_local, rows * state->attn_rows * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kQkvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kQkvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_query, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_gate, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_key, rows * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_value, rows * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_slot, rows * 2u * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->merged_gate, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->router_logits, rows * (G::kMoe ? G::kExperts : 0u) * sizeof(uint16_t) },
			{ (void **)&state->router_probs, rows * (G::kMoe ? G::kExperts : 0u) * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, (G::kMoe ? G::kLocalExperts + 2u : 0u) * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->packed_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, (G::kMoe ? G::kLocalExperts + 2u : 0u) * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, (G::kMoe ? G::kLocalExperts + 2u : 0u) * sizeof(uint32_t) },
			{ (void **)&state->packed_weight, routes * sizeof(float) },
			{ (void **)&state->expert_gate, routes * (G::kMoe ? G::kMoeInter : 0u) * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * (G::kMoe ? G::kMoeInter : 0u) * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * (G::kMoe ? G::kMoeInter : 0u) * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->shared_gate_up, rows * (G::kMoe ? 2u * G::kSharedRows : 0u) * sizeof(uint16_t) },
			{ (void **)&state->shared_act, rows * (G::kMoe ? G::kSharedRows : 0u) * sizeof(uint16_t) },
			{ (void **)&state->shared_out, (G::kMoe ? width : 0u) },
			{ (void **)&state->shared_logit, rows * (G::kMoe ? 1u : 0u) * sizeof(uint16_t) },
			{ (void **)&state->gdn_state, (uint64_t)S::kGdnLayers * slots * state->gdn_state_bytes },
			{ (void **)&state->gdn_window, (uint64_t)S::kGdnLayers * slots * state->gdn_window_bytes },
			{ (void **)&state->gdn_snapshots, replay != 0u && kTreeVerify ? LmDeltaRuleSnapshotBytes(state->gdn_heads, QH_KEY_DIM, QH_VALUE_DIM) : 0u },
			{ (void **)&state->replay_qkv, (uint64_t)S::kGdnLayers * replay * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->replay_ba, (uint64_t)S::kGdnLayers * replay * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->replay_ple, S::kPle ? replay * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->replay_index, (uint64_t)S::kAttnLayers * replay * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->residual, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_normed, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_low, rows * G::kHcLowRank * sizeof(uint16_t) },
			{ (void **)&state->hc_up, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_inject, S::kHyper ? rows * G::kStreams * sizeof(float) : 0u },
			{ (void **)&state->ple_ids, rows * S::kPleColumns * sizeof(int64_t) },
			{ (void **)&state->ple_embed, S::kPle ? width : 0u },
			{ (void **)&state->ple_key, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_query, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_value, S::kPle ? width : 0u },
			{ (void **)&state->ple_gated, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_conv_in, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_conv_out, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_window, slots * state->ple_window_bytes },
			{ (void **)&state->ple_history, S::kPle ? slots * S::kPleHistory * sizeof(int32_t) : 0u },
			{ (void **)&state->row_lane, S::kPle || S::kIndexer ? rows * sizeof(uint32_t) : 0u },
			{ (void **)&state->row_ordinal, S::kPle || S::kIndexer ? rows * sizeof(uint32_t) : 0u },
			{ (void **)&state->index_qk, rows * S::kIndexWidth * sizeof(uint16_t) },
			{ (void **)&state->index_query, rows * G::kIndexHeads * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_key, rows * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_block, rows * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_tail, (uint64_t)S::kAttnLayers * slots * S::kIndexTail * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_scores, S::kIndexer ? rows * state->index_local_stride * sizeof(float) : 0u },
			{ (void **)&state->index_local_selected, rows * S::kIndexBlocks * sizeof(uint32_t) },
			{ (void **)&state->index_candidates, rows * S::kIndexBlocks * sizeof(uint2) },
			{ (void **)&state->index_gathered, (uint64_t)state->tp_degree * rows * S::kIndexBlocks * sizeof(uint2) },
			{ (void **)&state->index_mask, rows * state->index_mask_words * sizeof(uint32_t) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static void Release(State *state)
	{
		if ( state->dump != 0 )
			fclose(state->dump);
		free(state->dump_rows);
		free(state->dump_positions);
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if constexpr ( G::kMoe )
		{
			if ( SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK ||
				experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		else if ( SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK || state->ffn_rows == 0u ||
			state->ffn_rows % QH_SCALE_K != 0u )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if constexpr ( S::kHyper )
		{
			uint32_t streams = 0u, lowrank = 0u, shared = 0u;
			if ( SparkNamedPackConfigU32(&state->pack, "hc_streams", &streams) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "hc_lowrank", &lowrank) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "shared_rows", &shared) != SPARK_STATUS_OK ||
				streams != G::kStreams || lowrank != G::kHcLowRank || shared != G::kSharedRows )
			{
				fprintf(stderr, "%s: pack hyper-connection streams %u lowrank %u shared rows %u do not match the model\n", G::Tag(),
					streams, lowrank, shared);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		if constexpr ( S::kPle )
		{
			uint32_t layer = 0u, eos = 0u;
			if ( SparkNamedPackConfigU32(&state->pack, "ple_layer", &layer) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "ple_eos", &eos) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "ple_table_first", &state->ple_table_first) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "ple_table_rows", &state->ple_table_rows) != SPARK_STATUS_OK ||
				layer != G::kPleLayer || eos != G::kPleEos || state->ple_table_rows == 0u )
			{
				fprintf(stderr, "%s: pack n-gram embedding layer %u eos %u rows %u do not match the model\n", G::Tag(), layer, eos,
					state->ple_table_rows);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Shape(State *state)
	{
		state->gdn_heads = G::kGdnValueHeads / state->tp_degree;
		state->gdn_qk = G::kGdnKeyHeads / state->tp_degree * QH_KEY_DIM;
		state->gdn_v = state->gdn_heads * QH_VALUE_DIM;
		state->gdn_channels = 2u * state->gdn_qk + state->gdn_v;
		state->attn_rows = S::kQkvWidth / state->tp_degree;
		state->attn_width = S::kQWidth / state->tp_degree;
		state->gdn_state_bytes = (uint64_t)state->gdn_heads * QH_KEY_DIM * QH_VALUE_DIM * sizeof(float);
		state->gdn_window_bytes = (uint64_t)state->gdn_channels * QH_CONV * sizeof(uint16_t);
		state->ple_window_bytes = S::kPle ? (uint64_t)S::kResidual * S::kPleSpan * sizeof(uint16_t) : 0u;
		if constexpr ( S::kIndexer )
		{
			SparkKvShard shard;
			shard.degree = state->tp_degree;
			shard.rank = state->tp_rank;
			shard.grain = 1u;
			const uint32_t blocks = state->max_positions / G::kIndexRatio;
			const uint32_t local = ((SparkKvShardGatherKeys(shard, blocks) + 63u) / 64u) * 64u, global = ((blocks + 63u) / 64u) * 64u;
			state->index_local_stride = local > S::kIndexBlocks ? local : S::kIndexBlocks;
			state->index_global_stride = global > S::kIndexBlocks ? global : S::kIndexBlocks;
			state->index_mask_words = state->index_global_stride / 32u;
		}
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, state->tp_degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration)
	{
#ifdef DEBUG
		if ( getenv(G::DumpEnv()) != 0 )
		{
			state->dump = fopen(getenv(G::DumpEnv()), "wb");
			state->dump_rows = (uint16_t *)malloc((size_t)state->max_rows * (S::kResidual + G::kHidden) * sizeof(uint16_t));
			state->dump_positions = (uint32_t *)malloc((size_t)state->max_rows * sizeof(uint32_t));
			if ( state->dump == 0 || state->dump_rows == 0 || state->dump_positions == 0 )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fprintf(stderr, "%s: this rank dumps every layer's residual and sublayer output to %s (debug; graphs off)\n",
				G::Tag(), getenv(G::DumpEnv()));
		}
#endif
		if ( configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 && !G::kMoe )
		{
			state->linear_bytes = 1u;
			state->linear_kind = "fp8_e4m3";
		}
		else if ( configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 )
		{
			state->linear_bytes = 2u;
			state->linear_kind = "bf16";
		}
		else
		{
			fprintf(stderr, "%s: linear weight codec %u is not supported by this model\n", G::Tag(), configuration->linear_weight_codec);
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
		state->max_positions = configuration->kv_pages_per_sequence * G::kPageSlots;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = kTreeVerify ? 1u : 0u;
		geometry->graph_ignores_gather = 1u;
		geometry->head_prenormed = S::kHyper ? 1u : 0u;
		return CudaStage<QwenHybridModel>::HeadRanges(state, geometry, false, S::kHyper);
	}

	template<class T>
	static void Elements(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t elements)
	{
		tensors->Add(name, kind, elements * sizeof(T), field);
	}

	static void LinearTensors(const State *state, CudaStageTensors *tensors, const char *name, const uint8_t **field, const float **scale_field, uint64_t rows,
		uint64_t columns)
	{
		tensors->Add(name, state->linear_kind, rows * columns * state->linear_bytes, field);
		*scale_field = 0;
		if ( state->linear_bytes == 1u )
			tensors->AddScaled(name, ".scale", "f32", rows * ((columns + QH_SCALE_K - 1u) / QH_SCALE_K) * sizeof(float), scale_field);
	}

	static void ExpertTensors(CudaStageTensors *tensors, const char *name, const uint8_t **field, const uint8_t **scale_field, uint64_t rows, uint64_t columns)
	{
		constexpr bool scaled = LmWeightCodec<G::kExpertCodec>::kScaleEncoding != LM_SCALE_ENCODING_NONE;
		tensors->Add(name, scaled ? "nvfp4_e2m1" : "bf16", SparkWeightCodecPayloadBytes(G::kExpertCodec, G::kLocalExperts * rows, columns), field);
		*scale_field = 0;
		if ( scaled )
			tensors->AddScaled(name, ".scale", "ue4m3_f32_global", SparkWeightCodecScaleBytes(G::kExpertCodec, G::kLocalExperts, rows, columns), scale_field);
	}

	static void PleTensors(State *state, CudaStageTensors *t)
	{
		Elements(t, "ple_key", "bf16", &state->ple_key_weight, (uint64_t)S::kResidual * G::kHidden);
		Elements(t, "ple_value", "bf16", &state->ple_value_weight, (uint64_t)G::kHidden * G::kHidden);
		Elements(t, "ple_norm_key", "f32", &state->ple_norm_key, S::kResidual);
		Elements(t, "ple_norm_query", "f32", &state->ple_norm_query, S::kResidual);
		Elements(t, "ple_norm_conv", "f32", &state->ple_norm_conv, S::kResidual);
		Elements(t, "ple_conv", "bf16", &state->ple_conv_weight, (uint64_t)S::kResidual * G::kPleConv);
		if constexpr ( G::kPleTableCodec == SPARK_WEIGHT_CODEC_BF16 )
		{
			state->ple_table_scale = 0;
			Elements(t, "ple_table", "bf16", &state->ple_table, (uint64_t)state->ple_table_rows * G::kPleHeadDim * 2u);
		}
		else
		{
			static_assert(G::kPleTableCodec == SPARK_WEIGHT_CODEC_FP8_E4M3, "the PLE table is fp8 with one scale or bf16");
			Elements(t, "ple_table", "fp8_e4m3", &state->ple_table, (uint64_t)state->ple_table_rows * G::kPleHeadDim);
			Elements(t, "ple_table_scale", "f32", &state->ple_table_scale, 1u);
		}
		Elements(t, "ple_multipliers", "i64", &state->ple_multipliers, G::kPleOrders + 1u);
		Elements(t, "ple_vocab_sizes", "i64", &state->ple_vocab_sizes, S::kPleColumns);
		Elements(t, "ple_offsets", "i64", &state->ple_offsets, S::kPleColumns);
	}

	static SparkStatus GlobalTensors(State *state, CudaStageTensors *t)
	{
		if constexpr ( S::kHyper )
		{
			Elements(t, "final_hc_norm", "f32", &state->final_hc_norm, S::kResidual);
			Elements(t, "final_hc_down", "bf16", &state->final_hc_down, (uint64_t)G::kHcLowRank * S::kResidual);
			Elements(t, "final_hc_up", "bf16", &state->final_hc_up, (uint64_t)S::kResidual * G::kHcLowRank);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *t)
	{
		QwenHybridLayer *layer = &state->layers[index];
		const uint32_t attention_layer = index % G::kPeriod == G::kPhase;
		if constexpr ( G::kMoe )
		{
			Elements(t, "moe_router", "bf16", &layer->moe_router, (uint64_t)G::kExperts * G::kHidden);
			ExpertTensors(t, "moe_w1", &layer->moe_w1, &layer->moe_w1_scale, G::kMoeInter, G::kHidden);
			ExpertTensors(t, "moe_w3", &layer->moe_w3, &layer->moe_w3_scale, G::kMoeInter, G::kHidden);
			ExpertTensors(t, "moe_w2", &layer->moe_w2, &layer->moe_w2_scale, G::kHidden, G::kMoeInter);
			Elements(t, "moe_shared_gate_up", "bf16", &layer->moe_shared_gate_up, (uint64_t)2u * G::kSharedRows * G::kHidden);
			Elements(t, "moe_shared_down", "bf16", &layer->moe_shared_down, (uint64_t)G::kHidden * G::kSharedRows);
			Elements(t, "moe_shared_gate", "bf16", &layer->moe_shared_gate, G::kHidden);
		}
		else
		{
			LinearTensors(state, t, "ffn_gate_up", &layer->ffn_gate_up, &layer->ffn_gate_up_scale, 2u * state->ffn_rows, G::kHidden);
			LinearTensors(state, t, "ffn_down", &layer->ffn_down, &layer->ffn_down_scale, G::kHidden, state->ffn_rows);
		}
		if ( attention_layer )
		{
			LinearTensors(state, t, "attn_qkv", &layer->mix_weight, &layer->mix_scale, S::kQkvWidth / G::kTpDegree, G::kHidden);
			LinearTensors(state, t, "attn_out", &layer->out_weight, &layer->out_scale, G::kHidden, S::kQWidth / G::kTpDegree);
		}
		else
		{
			LinearTensors(state, t, "gdn_qkvz", &layer->mix_weight, &layer->mix_scale, 2u * state->gdn_qk + 2u * state->gdn_v, G::kHidden);
			LinearTensors(state, t, "gdn_out", &layer->out_weight, &layer->out_scale, G::kHidden, state->gdn_v);
		}
		if ( !S::kHyper )
		{
			Elements(t, "input_norm", "f32", &layer->input_norm, G::kHidden);
			Elements(t, "post_norm", "f32", &layer->post_norm, G::kHidden);
		}
		for ( uint32_t part = 0u; part < 2u && S::kHyper; ++part )
		{
			const char *prefix = part == 0u ? "attn_hc_" : "mlp_hc_";
			char name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
			snprintf(name, sizeof(name), "%snorm", prefix);
			Elements(t, name, "f32", &layer->hc_norm[part], S::kResidual);
			snprintf(name, sizeof(name), "%sdown", prefix);
			Elements(t, name, "bf16", &layer->hc_down[part], (uint64_t)G::kHcLowRank * S::kResidual);
			snprintf(name, sizeof(name), "%sup", prefix);
			Elements(t, name, "bf16", &layer->hc_up[part], (uint64_t)S::kResidual * G::kHcLowRank);
			snprintf(name, sizeof(name), "%sinject", prefix);
			Elements(t, name, "bf16", &layer->hc_inject[part], (uint64_t)G::kStreams * S::kResidual);
		}
		if constexpr ( S::kPle )
			if ( index == G::kPleLayer )
				PleTensors(state, t);
		if ( attention_layer )
		{
			Elements(t, "attn_q_norm", "f32", &layer->q_norm, G::kHeadDim);
			Elements(t, "attn_k_norm", "f32", &layer->k_norm, G::kHeadDim);
			if ( S::kIndexer )
			{
				Elements(t, "attn_index_qk", "bf16", &layer->index_qk, (uint64_t)S::kIndexWidth * G::kHidden);
				Elements(t, "attn_index_q_norm", "f32", &layer->index_q_norm, G::kIndexDim);
				Elements(t, "attn_index_k_norm", "f32", &layer->index_k_norm, G::kIndexDim);
			}
		}
		else
		{
			Elements(t, "gdn_ba", "bf16", &layer->gdn_ba, (uint64_t)2u * state->gdn_heads * G::kHidden);
			Elements(t, "gdn_conv", "bf16", &layer->gdn_conv, (uint64_t)state->gdn_channels * QH_CONV);
			Elements(t, "gdn_norm", "bf16", &layer->gdn_norm, QH_VALUE_DIM);
			Elements(t, "gdn_a_log", "f32", &layer->gdn_a_log, state->gdn_heads);
			Elements(t, "gdn_dt_bias", "f32", &layer->gdn_dt_bias, state->gdn_heads);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t recurrent = 0u, attention = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			if ( index % G::kPeriod == G::kPhase )
				state->layers[index].kv_index = attention++;
			else
				state->layers[index].recurrent_index = recurrent++;
		return SPARK_STATUS_OK;
	}

	static uint32_t LaneReady(const State *state)
	{
		return state->lane_arena != 0 && state->dump == 0 ? 1u : 0u;
	}

	static SparkStatus LaneAdmit(State *state, const SparkStageRunnerStep *step)
	{
		(void)state;
		(void)step;
		return SPARK_STATUS_OK;
	}

	static uint32_t RecurrentParts(const State *state, CudaStageRecurrent *parts)
	{
		parts[0] = { state->gdn_state, state->gdn_state_bytes, S::kGdnLayers };
		parts[1] = { (uint8_t *)state->gdn_window, state->gdn_window_bytes, S::kGdnLayers };
		uint32_t count = 2u;
		if constexpr ( S::kPle )
		{
			parts[count++] = { (uint8_t *)state->ple_window, state->ple_window_bytes, 1u };
			parts[count++] = { (uint8_t *)state->ple_history, S::kPleHistory * sizeof(int32_t), 1u };
		}
		if constexpr ( S::kIndexer )
			parts[count++] = { (uint8_t *)state->index_tail, (uint64_t)S::kIndexTail * G::kIndexDim * sizeof(uint16_t), S::kAttnLayers };
		return count;
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

	static SparkStatus TapPlan(void *model, const SparkStageRunnerTapPlan *plan)
	{
		State *state = (State *)model;
		int16_t outputs[G::kLayers];
		if ( state == 0 || plan == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if constexpr ( S::kHyper )
		{
			fprintf(stderr, "%s: taps refused: a layer output here is several hyper-connection streams, not one residual row\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
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

	static uint32_t GraphKey(void *model, const SparkStageRunnerStep *step)
	{
		if ( ((const State *)model)->dump != 0 )
			return 0u;
		if ( step->commit == 0u && step->last_rows_only == 0u && step->sequences == 1u && step->rows > 1u && step->rows <= G::kVerifyRows )
			return step->tree != 0u && !kTreeVerify ? 0u : 2u;
		return 1u;
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		SparkStatus status = SPARK_STATUS_OK;
		if ( G::kVerifyRows == 0u || kept == 0 || step->commit != 0u || step->rows == 0u || step->rows > G::kVerifyRows || step->sequences == 0u ||
			step->sequences > state->slots || (S::kPle && step->token_ids == 0) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->tree != 0u && (!kTreeVerify || step->sequences != 1u || step->slots == 0 || step->sequence_row_indices == 0) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const QwenHybridLayer *layer = &state->layers[index];
			const uint64_t replay = (uint64_t)layer->recurrent_index * G::kVerifyRows;
			if ( index % G::kPeriod == G::kPhase )
			{
				if constexpr ( S::kIndexer )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenIndexTailKernel<G>), step->sequences, G::kIndexDim, 0, stream, state->replay_index +
						(uint64_t)layer->kv_index * G::kVerifyRows * G::kIndexDim, state->index_tail + (uint64_t)layer->kv_index * state->slots * S::kIndexTail *
						G::kIndexDim, step->positions, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, 1u, kept);
					status = CudaStageLaunched();
				}
				else if ( step->tree != 0u )
				{
					CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
					LM_LAUNCH((LmKvShardPathMoveKernel<typename S::Kv,QH_THREADS>), G::kVerifyRows, QH_THREADS, 0, stream,
						state->kv_views[layer->kv_index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, 2u * S::kKvWidth);
					CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
					status = CudaStageLaunched();
				}
				continue;
			}
			status = GdnRecur(state, layer, step, state->replay_qkv + replay * state->gdn_channels, state->replay_ba + replay * 2u * state->gdn_heads, kept,
				1u, stream);
		}
		if constexpr ( S::kPle )
			if ( status == SPARK_STATUS_OK )
				status = PleRecur(state, step, state->replay_ple, kept, 1u, stream);
		return status;
	}
};
