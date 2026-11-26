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
	uint16_t *qkv, *z, *ba, *gdn_qkvz, *value, *query_heads, *key_heads, *mixed;
	uint16_t *attn_local, *attn_gathered, *attn_query, *merged, *merged_gate;
	uint16_t *router_logits, *expert_gate, *expert_up, *expert_act, *expert_out, *shared_gate_up, *shared_act, *shared_out, *shared_logit;
	uint16_t *gdn_window, *replay_qkv, *replay_ba, *replay_ple, *replay_index, *residual, *hc_normed, *hc_low, *hc_up;
	uint16_t *ple_embed, *ple_key, *ple_query, *ple_value, *ple_gated, *ple_conv_in, *ple_conv_out, *ple_window;
	uint16_t *index_qk, *index_query, *index_key, *index_tail;
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
	uint64_t split_gqa_floats;
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
static void QwenPackLocalQkvNoGateKernel(const uint16_t *__restrict__ attn_local, uint16_t *__restrict__ qkv_send, uint32_t rows, uint32_t attn_rows)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t HD_V = G::kHeadDim >> 3u;
	constexpr uint32_t Q_LOCAL_V = (S::kQWidth / G::kTpDegree) >> 3u;
	constexpr uint32_t KV_LOCAL_V = (2u * S::kKvWidth / G::kTpDegree) >> 3u;
	constexpr uint32_t PACK_V = Q_LOCAL_V + KV_LOCAL_V;
	const uint32_t attn_rows_v = attn_rows >> 3u;
	const uint64_t total = (uint64_t)rows * PACK_V;
	const uint4 *src4 = (const uint4 *)attn_local;
	uint4 *dst4 = (uint4 *)qkv_send;
	LmDependentWait();
	for ( uint64_t idx = (uint64_t)blockIdx.x * QH_THREADS + threadIdx.x; idx < total; idx += (uint64_t)gridDim.x * QH_THREADS )
	{
		const uint64_t r = idx / PACK_V;
		const uint32_t c = (uint32_t)(idx % PACK_V);
		uint32_t src_c;
		if ( c < Q_LOCAL_V )
			src_c = ((c / HD_V) * 2u) * HD_V + (c % HD_V);
		else
			src_c = 2u * Q_LOCAL_V + (c - Q_LOCAL_V);
		dst4[idx] = __ldg(src4 + r * attn_rows_v + src_c);
	}
	LmDependentRelease();
}

template<class G, bool QKV_NO_GATE = false>
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
	constexpr uint32_t Q_RANK_V = (S::kQWidth / G::kTpDegree) >> 3u;
	constexpr uint32_t KV_RANK_V = KV_V / G::kTpDegree;
	constexpr uint32_t PACK_V = Q_RANK_V + 2u * KV_RANK_V;
	constexpr uint32_t TASKS = G::kHeads + 2u * G::kKvHeads;
	__shared__ __align__(16) uint16_t rope_smem[WARPS][G::kRopeDim];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( task >= TASKS || row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint4 *gathered4 = (const uint4 *)gathered;
	const uint32_t attn_rows_v = attn_rows >> 3u;
	if ( task < G::kHeads )
	{
		const uint32_t h = task;
		uint4 qv;
		if constexpr ( QKV_NO_GATE )
		{
			const uint32_t qv_idx = h * HD_V + lane;
			const uint32_t rk = qv_idx / Q_RANK_V, el = qv_idx % Q_RANK_V;
			qv = gathered4[(((uint64_t)rk * rows) + row) * PACK_V + el];
		}
		else
			qv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, (h * 2u) * HD_V + lane);
		const uint4 q_out = QwenAttnNormRopeHead<G>(qv, q_norm, rope_smem[warp], (float)positions[row], lane);
		((uint4 *)attn_query)[((uint64_t)row * G::kHeads + h) * HD_V + lane] = q_out;
		if constexpr ( G::kTpDegree > 1u && !QKV_NO_GATE )
		{
			const uint32_t gate_begin_v = (tp_rank * attn_width) >> 3u;
			const uint32_t attn_width_v = attn_width >> 3u;
			const uint32_t gate_vec = h * HD_V + lane;
			if ( gate_vec >= gate_begin_v && gate_vec < gate_begin_v + attn_width_v )
				((uint4 *)merged_gate)[(uint64_t)row * attn_width_v + (gate_vec - gate_begin_v)] =
					QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, (h * 2u + 1u) * HD_V + lane);
		}
		LmDependentRelease();
		return;
	}
	const uint32_t store_pos = store_positions[row];
	if ( SparkKvShardOwns(kv_view.shard, store_pos) == 0u )
	{
		LmDependentRelease();
		return;
	}
	uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
	{
		LmDependentRelease();
		return;
	}
	if ( task < G::kHeads + G::kKvHeads )
	{
		const uint32_t kh = task - G::kHeads;
		uint4 kv;
		if constexpr ( QKV_NO_GATE )
		{
			const uint32_t k_v = kh * HD_V + lane;
			const uint32_t rk = k_v / KV_RANK_V, el = Q_RANK_V + (k_v % KV_RANK_V);
			kv = gathered4[(((uint64_t)rk * rows) + row) * PACK_V + el];
		}
		else
			kv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, Q2_V + kh * HD_V + lane);
		const uint4 k_out = QwenAttnNormRopeHead<G>(kv, k_norm, rope_smem[warp], (float)positions[row], lane);
		((uint4 *)slot)[kh * HD_V + lane] = k_out;
	}
	else
	{
		const uint32_t vh = task - G::kHeads - G::kKvHeads;
		uint4 vv;
		if constexpr ( QKV_NO_GATE )
		{
			const uint32_t v_v = vh * HD_V + lane;
			const uint32_t rk = v_v / KV_RANK_V, el = Q_RANK_V + KV_RANK_V + (v_v % KV_RANK_V);
			vv = gathered4[(((uint64_t)rk * rows) + row) * PACK_V + el];
		}
		else
			vv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, Q2_V + KV_V + vh * HD_V + lane);
		((uint4 *)slot)[KV_V + vh * HD_V + lane] = vv;
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenPackLocalKvKernel(const uint16_t *__restrict__ attn_local, uint16_t *__restrict__ kv_send, uint32_t rows, uint32_t attn_rows)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t Q2_LOCAL_V = (2u * S::kQWidth / G::kTpDegree) >> 3u;
	constexpr uint32_t KV_LOCAL_V = (2u * S::kKvWidth / G::kTpDegree) >> 3u;
	const uint32_t attn_rows_v = attn_rows >> 3u;
	const uint64_t total = (uint64_t)rows * KV_LOCAL_V;
	const uint4 *src4 = (const uint4 *)attn_local;
	uint4 *dst4 = (uint4 *)kv_send;
	LmDependentWait();
	for ( uint64_t idx = (uint64_t)blockIdx.x * QH_THREADS + threadIdx.x; idx < total; idx += (uint64_t)gridDim.x * QH_THREADS )
	{
		const uint64_t r = idx / KV_LOCAL_V;
		const uint32_t c = (uint32_t)(idx % KV_LOCAL_V);
		dst4[idx] = __ldg(src4 + r * attn_rows_v + Q2_LOCAL_V + c);
	}
	LmDependentRelease();
}

template<class G, bool KV_ONLY_GATHER>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenAttnPrepLocalKernel(const uint16_t *__restrict__ attn_local, const uint16_t *__restrict__ gathered,
	const float *__restrict__ q_norm, const float *__restrict__ k_norm, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view, uint16_t *__restrict__ attn_query,
	uint16_t *__restrict__ merged_gate, uint16_t *__restrict__ kv_cur, uint32_t rows, uint32_t attn_rows,
	uint32_t tp_rank, uint32_t attn_width, uint32_t q_first, uint32_t rank_heads, uint32_t kv_first, uint32_t local_kvh)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	constexpr uint32_t HD_V = G::kHeadDim >> 3u;
	constexpr uint32_t Q2_V = (2u * S::kQWidth) >> 3u;
	constexpr uint32_t KV_V = S::kKvWidth >> 3u;
	constexpr uint32_t KV_RANK_V = KV_V / G::kTpDegree;
	constexpr uint32_t KV_LOCAL_V = 2u * KV_RANK_V;
	__shared__ __align__(16) uint16_t rope_smem[WARPS][G::kRopeDim];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * WARPS + warp, row = blockIdx.y;
	const uint32_t tasks = rank_heads + 2u * G::kKvHeads;
	LmDependentWait();
	if ( task >= tasks || row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint4 *local4 = (const uint4 *)attn_local;
	const uint4 *gathered4 = (const uint4 *)gathered;
	const uint32_t attn_rows_v = attn_rows >> 3u;
	if ( task < rank_heads )
	{
		const uint32_t h = q_first + task;
		const uint4 qv = KV_ONLY_GATHER ?
			local4[(uint64_t)row * attn_rows_v + (task * 2u) * HD_V + lane] :
			QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, (h * 2u) * HD_V + lane);
		const uint4 q_out = QwenAttnNormRopeHead<G>(qv, q_norm, rope_smem[warp], (float)__ldg(positions + row), lane);
		((uint4 *)attn_query)[((uint64_t)row * rank_heads + task) * HD_V + lane] = q_out;
		if constexpr ( G::kTpDegree > 1u && !KV_ONLY_GATHER )
		{
			const uint32_t gate_begin_v = (tp_rank * attn_width) >> 3u;
			const uint32_t attn_width_v = attn_width >> 3u;
			const uint32_t gate_vec = h * HD_V + lane;
			if ( gate_vec >= gate_begin_v && gate_vec < gate_begin_v + attn_width_v )
				((uint4 *)merged_gate)[(uint64_t)row * attn_width_v + (gate_vec - gate_begin_v)] =
					QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, (h * 2u + 1u) * HD_V + lane);
		}
		LmDependentRelease();
		return;
	}
	const uint32_t store_pos = __ldg(positions + row);
	const bool owns = SparkKvShardOwns(kv_view.shard, store_pos) != 0u;
	if ( task < rank_heads + G::kKvHeads )
	{
		const uint32_t kh = task - rank_heads;
		const bool need = kh >= kv_first && kh < kv_first + local_kvh;
		if ( owns || need )
		{
			uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, __ldg(sequence_of_row + row), store_pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
			uint4 kv;
			if constexpr ( KV_ONLY_GATHER )
			{
				const uint32_t k_v = kh * HD_V + lane;
				const uint32_t rk = k_v / KV_RANK_V;
				const uint32_t el = k_v % KV_RANK_V;
				kv = gathered4[(((uint64_t)rk * rows) + row) * KV_LOCAL_V + el];
			}
			else
				kv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, Q2_V + kh * HD_V + lane);
			const uint4 k_out = QwenAttnNormRopeHead<G>(kv, k_norm, rope_smem[warp], (float)store_pos, lane);
			if ( slot != 0 )
				((uint4 *)slot)[kh * HD_V + lane] = k_out;
			if ( need )
				((uint4 *)kv_cur)[(((uint64_t)(kh - kv_first) * rows + row) * 2u) * HD_V + lane] = k_out;
		}
	}
	else
	{
		const uint32_t vh = task - rank_heads - G::kKvHeads;
		const bool need = vh >= kv_first && vh < kv_first + local_kvh;
		if ( owns || need )
		{
			uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, __ldg(sequence_of_row + row), store_pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
			uint4 vv;
			if constexpr ( KV_ONLY_GATHER )
			{
				const uint32_t v_v = vh * HD_V + lane;
				const uint32_t rk = v_v / KV_RANK_V;
				const uint32_t el = KV_RANK_V + (v_v % KV_RANK_V);
				vv = gathered4[(((uint64_t)rk * rows) + row) * KV_LOCAL_V + el];
			}
			else
				vv = QwenAttnLoadVec8<G>(gathered4, rows, row, attn_rows_v, Q2_V + KV_V + vh * HD_V + lane);
			if ( slot != 0 )
				((uint4 *)slot)[KV_V + vh * HD_V + lane] = vv;
			if ( need )
				((uint4 *)kv_cur)[(((uint64_t)(vh - kv_first) * rows + row) * 2u + 1u) * HD_V + lane] = vv;
		}
	}
	LmDependentRelease();
}

#define QH_PREFILL_MAX_SEQS 64u

typedef struct QwenPrefillSeqPlan
{
	uint32_t count;
	uint32_t total_prior;
	uint32_t total_peer;
	uint32_t seq_id[QH_PREFILL_MAX_SEQS];
	uint32_t row_begin[QH_PREFILL_MAX_SEQS];
	uint32_t row_len[QH_PREFILL_MAX_SEQS];
	uint32_t old_ctx[QH_PREFILL_MAX_SEQS];
	uint32_t prior_off[QH_PREFILL_MAX_SEQS];
	uint32_t peer_off[QH_PREFILL_MAX_SEQS];
	uint32_t peer_keys[QH_PREFILL_MAX_SEQS];
}
QwenPrefillSeqPlan;

template<class Kv, uint32_t KVH, uint32_t HD>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenPriorKvDirectKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row,
	const __grid_constant__ QwenPrefillSeqPlan plan, uint32_t kv_first, uint32_t local_kvh, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t KV_V = KVH * HD_V;
	const uint32_t k_idx = blockIdx.x, lh = blockIdx.y, s = blockIdx.z;
	LmDependentWait();
	if ( s < plan.count && k_idx < plan.old_ctx[s] && lh < local_kvh )
	{
		const uint32_t seq = plan.count == 1u ? __ldg(sequence_of_row) : plan.seq_id[s];
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, seq, k_idx, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint32_t kh = kv_first + lh;
			const uint4 *ks = (const uint4 *)slot + kh * HD_V;
			const uint4 *vs = (const uint4 *)slot + KV_V + kh * HD_V;
			uint4 *kd = (uint4 *)kv_prior + (((uint64_t)lh * plan.total_prior + (plan.prior_off[s] + k_idx)) * 2u) * HD_V;
			uint4 *vd = kd + HD_V;
			for ( uint32_t c = threadIdx.x; c < HD_V; c += QH_THREADS )
			{
				kd[c] = __ldg(ks + c);
				vd[c] = __ldg(vs + c);
			}
		}
	}
	LmDependentRelease();
}

template<class Kv, uint32_t KVH, uint32_t HD>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenPriorKvPackKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row,
	const __grid_constant__ QwenPrefillSeqPlan plan, uint32_t attn_width, uint32_t group, uint32_t local_kvh,
	uint64_t a2a_stride_bytes, uint8_t *__restrict__ send_bytes)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t KV_V = KVH * HD_V;
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	static_assert(HD_V == 32u, "one warp copies one head-dim K/V vector");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t idx = blockIdx.x * WARPS + warp, peer = blockIdx.y, s = blockIdx.z;
	const uint32_t degree = kv_view.shard.degree, grain = kv_view.shard.grain, my_rank = kv_view.shard.rank;
	const uint32_t pos = (my_rank + (idx / grain) * degree) * grain + (idx % grain);
	LmDependentWait();
	if ( s < plan.count && idx < plan.peer_keys[s] && peer < degree && pos < plan.old_ctx[s] )
	{
		const uint32_t seq = plan.count == 1u ? __ldg(sequence_of_row) : plan.seq_id[s];
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, seq, pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
		{
			const uint32_t peer_kv_first = ((peer * attn_width) / HD) / group;
			uint4 *peer_dst = (uint4 *)(send_bytes + (uint64_t)peer * a2a_stride_bytes);
			for ( uint32_t lh = 0u; lh < local_kvh; ++lh )
			{
				const uint32_t kh = peer_kv_first + lh;
				const uint4 *ks = (const uint4 *)slot + kh * HD_V;
				const uint4 *vs = (const uint4 *)slot + KV_V + kh * HD_V;
				uint4 *kd = peer_dst + (((uint64_t)lh * plan.total_peer + (plan.peer_off[s] + idx)) * 2u) * HD_V;
				kd[lane] = __ldg(ks + lane);
				kd[HD_V + lane] = __ldg(vs + lane);
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t HD>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenPriorKvUnpackKernel(SparkKvShard shard, const uint8_t *__restrict__ recv_bytes, uint64_t a2a_stride_bytes,
	const __grid_constant__ QwenPrefillSeqPlan plan, uint32_t local_kvh, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	static_assert(HD_V == 32u, "one warp unpacks one key-value pair");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t idx = blockIdx.x * WARPS + warp, src_rank = blockIdx.y, s = blockIdx.z;
	const uint32_t degree = shard.degree, grain = shard.grain;
	const uint32_t pos = (src_rank + (idx / grain) * degree) * grain + (idx % grain);
	LmDependentWait();
	if ( s < plan.count && idx < plan.peer_keys[s] && src_rank < degree && pos < plan.old_ctx[s] )
	{
		const uint4 *src_base = (const uint4 *)(recv_bytes + (uint64_t)src_rank * a2a_stride_bytes);
		for ( uint32_t lh = 0u; lh < local_kvh; ++lh )
		{
			const uint4 *s4 = src_base + (((uint64_t)lh * plan.total_peer + (plan.peer_off[s] + idx)) * 2u) * HD_V;
			uint4 *d4 = (uint4 *)kv_prior + (((uint64_t)lh * plan.total_prior + (plan.prior_off[s] + pos)) * 2u) * HD_V;
			d4[lane] = __ldg(s4 + lane);
			d4[HD_V + lane] = __ldg(s4 + HD_V + lane);
		}
	}
	LmDependentRelease();
}

template<uint32_t HD>
static __host__ __device__ constexpr uint32_t QwenLocalPrefillSharedBytes(void)
{
	constexpr uint32_t V_WARPS = HD / 128u;
	constexpr uint32_t Q_WARPS = 2u;
	constexpr uint32_t KV_TILE = 32u;
	constexpr uint32_t ROW_BYTES = HD * 2u;
	constexpr uint32_t KV_BYTES = 2u * KV_TILE * ROW_BYTES;
	constexpr uint32_t SCORE_BYTES = V_WARPS > 1u ? Q_WARPS * V_WARPS * (KV_TILE / 2u) * 32u * 4u : 0u;
	return KV_BYTES + SCORE_BYTES;
}

template<class Kv, uint32_t KVH, uint32_t HD, bool HAS_MASK, bool GATE_FROM_LOCAL, bool DIRECT_PAGED = false>
__global__ __launch_bounds__(128u, 2)
static void QwenLocalPrefillAttnKernel(const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ kv_prior,
	const uint16_t *__restrict__ kv_cur, LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row,
	const uint16_t *__restrict__ gate_base, uint32_t gate_stride,
	const uint32_t *__restrict__ positions, const __grid_constant__ QwenPrefillSeqPlan plan, uint32_t rows, uint32_t rank_heads,
	uint32_t q_first, uint32_t group, uint32_t kv_first, uint32_t elem_first, uint32_t attn_width, float qk_scale,
	const uint32_t *__restrict__ block_mask, uint32_t mask_words, uint32_t block_tokens, uint32_t index_blocks,
	uint16_t *__restrict__ merged_out)
{
	constexpr uint32_t KV_TILE = 32u;
	constexpr uint32_t V_WARPS = HD / 128u;
	constexpr uint32_t Q_WARPS = 2u;
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
	static_assert(HD == 256u && THREADS == 128u && Q_TILE * ROW_BYTES <= 2u * TILE_BYTES, "valid qwen local prefill MMA geometry");
	static_assert(!DIRECT_PAGED || (Kv::kPageSlots % KV_TILE) == 0u, "direct paged KV requires page slots divisible by KV_TILE");
	extern __shared__ __align__(128) uint8_t qh_prefill_smem[];
	uint8_t *k_smem = qh_prefill_smem;
	uint8_t *v_smem = qh_prefill_smem + TILE_BYTES;
	float *score_buf = (float *)(qh_prefill_smem + 2u * TILE_BYTES);
	const uint32_t k_base = (uint32_t)__cvta_generic_to_shared(k_smem);
	const uint32_t v_base = (uint32_t)__cvta_generic_to_shared(v_smem);
	const uint32_t local_h = DIRECT_PAGED ? (blockIdx.y * group + (blockIdx.x % group)) : blockIdx.x;
	const uint32_t q_block = DIRECT_PAGED ? (blockIdx.x / group) : blockIdx.y;
	const uint32_t seq_idx = blockIdx.z;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t q_warp = warp / V_WARPS, v_warp = warp % V_WARPS;
	LmDependentWait();
	if ( seq_idx >= plan.count )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t s_row_begin = plan.row_begin[seq_idx], s_rows = plan.row_len[seq_idx];
	const uint32_t s_prior_keys = plan.old_ctx[seq_idx], s_prior_off = plan.prior_off[seq_idx];
	const uint32_t s_seq_id = DIRECT_PAGED ? (plan.count == 1u ? __ldg(sequence_of_row) : plan.seq_id[seq_idx]) : 0u;
	const uint32_t local_first = q_block * Q_TILE;
	if ( local_first >= s_rows )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t row_first = s_row_begin + local_first;
	const uint32_t row_end = s_row_begin + (local_first + Q_TILE < s_rows ? local_first + Q_TILE : s_rows);
	const uint32_t kh_global = (q_first + local_h) / group;
	const uint32_t lkvh = kh_global - kv_first;
	const uint32_t total_kv = s_prior_keys + s_rows;
	const bool seq_masked = HAS_MASK && block_tokens != 0u && (total_kv / block_tokens) > index_blocks;
	const float scale_log2 = qk_scale * 1.4426950408889634f;
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
	const uint32_t tile_end = (max_pos + 1u + KV_TILE - 1u) / KV_TILE;
	const uint32_t row_lo = row_first + q_warp * 16u + (lane >> 2u);
	const uint32_t row_hi = row_lo + 8u;
	const bool valid_row[2] = { row_lo < row_end, row_hi < row_end };
	const uint32_t rpos[2] = { valid_row[0] ? __ldg(positions + row_lo) : 0u, valid_row[1] ? __ldg(positions + row_hi) : 0u };
	const uint32_t rrow[2] = { row_lo, row_hi };
	auto tile_live = [&](uint32_t t) -> bool
	{
		if ( !seq_masked )
			return true;
		const uint32_t tile_blks = KV_TILE / block_tokens;
		const uint32_t blk0 = t * tile_blks;
		if ( blk0 + tile_blks > (min_pos + 1u) / block_tokens )
			return true;
		const uint32_t widx = blk0 >> 5u, sh = blk0 & 31u;
		const uint32_t bits_mask = (1u << tile_blks) - 1u;
		uint32_t any = 0u;
		if ( threadIdx.x < (row_end - row_first) )
			any = (__ldg(block_mask + (uint64_t)(row_first + threadIdx.x) * mask_words + widx) >> sh) & bits_mask;
		return __syncthreads_or(any != 0u) != 0;
	};
	auto next_live = [&](uint32_t start) -> uint32_t
	{
		uint32_t t = start;
		if ( seq_masked )
			while ( t < tile_end && !tile_live(t) )
				++t;
		return t;
	};
	auto copy_kv = [&](uint32_t tile, uint32_t dst_base, uint32_t is_v)
	{
		const uint32_t t0 = tile * KV_TILE;
		const uint8_t *paged_base = (DIRECT_PAGED && t0 < s_prior_keys) ?
			LmKvShardSlotRequired<Kv>(kv_view, s_seq_id, t0, 0u, LM_KV_ACCESS_READ) : (const uint8_t *)0;
		for ( uint32_t idx = threadIdx.x; idx < KV_TILE * CHUNKS; idx += THREADS )
		{
			const uint32_t s = idx / CHUNKS, c = idx % CHUNKS;
			const uint32_t k_idx = t0 + s;
			const bool valid = k_idx < total_kv;
			const uint16_t *src = query_bf16;
			if ( valid )
			{
				if ( k_idx < s_prior_keys )
				{
					if constexpr ( DIRECT_PAGED )
						src = paged_base != 0 ?
							((const uint16_t *)(paged_base + (uint64_t)s * Kv::kSlotBytes) + (is_v * KVH + kh_global) * HD + c * 8u) :
							query_bf16;
					else
						src = kv_prior + (((uint64_t)lkvh * plan.total_prior + (s_prior_off + k_idx)) * 2u + is_v) * HD + c * 8u;
				}
				else
					src = kv_cur + (((uint64_t)lkvh * rows + (s_row_begin + k_idx - s_prior_keys)) * 2u + is_v) * HD + c * 8u;
			}
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
	uint32_t cur_tile = next_live(0u);
	if ( cur_tile < tile_end )
		copy_kv(cur_tile, k_base, 0u);
	while ( cur_tile < tile_end )
	{
		copy_kv(cur_tile, v_base, 1u);
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
		const uint32_t nt = next_live(cur_tile + 1u);
		if ( nt < tile_end )
			copy_kv(nt, k_base, 0u);
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
		const uint32_t tile_pos0 = cur_tile * KV_TILE;
		const uint32_t tile_pos1 = tile_pos0 + KV_TILE - 1u;
		const bool whole = !seq_masked && __all_sync(0xffffffffu, valid_row[0] && valid_row[1] && tile_pos1 <= rpos[0] && tile_pos1 <= rpos[1]);
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
					bool vis = valid_row[side] && key_pos <= rpos[side];
					if ( seq_masked && vis )
					{
						const uint32_t blk = key_pos / block_tokens;
						if ( blk < (rpos[side] + 1u) / block_tokens )
							vis = ((__ldg(block_mask + (uint64_t)rrow[side] * mask_words + (blk >> 5u)) >> (blk & 31u)) & 1u) != 0u;
					}
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
		cur_tile = nt;
		if ( cur_tile < tile_end )
			__syncthreads();
	}
	#pragma unroll
	for ( uint32_t side = 0u; side < 2u; ++side )
	{
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 2u);
	}
	const float denom_lo = fmaxf(running_sum[0], 1.0e-20f);
	const float denom_hi = fmaxf(running_sum[1], 1.0e-20f);
	const uint32_t head_base = (q_first + local_h) * HD;
	const uint32_t elem_end = elem_first + attn_width;
	#pragma unroll
	for ( uint32_t f = 0u; f < VALUE_FRAGS; ++f )
	{
		const uint32_t col = v_warp * PART + f * 8u + (lane & 3u) * 2u;
		const uint32_t elem = head_base + col;
		if ( elem >= elem_first && elem < elem_end )
		{
			const uint32_t dst_col = elem - elem_first;
			const uint64_t gcol = GATE_FROM_LOCAL ? ((uint64_t)(local_h * 2u + 1u) * HD + col) : (uint64_t)dst_col;
			if ( valid_row[0] )
			{
				const uint32_t g = *(const uint32_t *)(gate_base + (uint64_t)row_lo * gate_stride + gcol);
				const float g0 = __uint_as_float(g << 16u), g1 = __uint_as_float(g & 0xffff0000u);
				const float o0 = SparkMathRoundBf16(output[f][0] / denom_lo) * SparkMathSigmoidFast(g0);
				const float o1 = SparkMathRoundBf16(output[f][1] / denom_lo) * SparkMathSigmoidFast(g1);
				__stcs((uint32_t *)(merged_out + (uint64_t)row_lo * attn_width + dst_col), LmPackBf16Pair(o0, o1));
			}
			if ( valid_row[1] )
			{
				const uint32_t g = *(const uint32_t *)(gate_base + (uint64_t)row_hi * gate_stride + gcol);
				const float g0 = __uint_as_float(g << 16u), g1 = __uint_as_float(g & 0xffff0000u);
				const float o0 = SparkMathRoundBf16(output[f][2] / denom_hi) * SparkMathSigmoidFast(g0);
				const float o1 = SparkMathRoundBf16(output[f][3] / denom_hi) * SparkMathSigmoidFast(g1);
				__stcs((uint32_t *)(merged_out + (uint64_t)row_hi * attn_width + dst_col), LmPackBf16Pair(o0, o1));
			}
		}
	}
	LmDependentRelease();
}

static constexpr uint32_t QH_GQA_MAX_SPLITS = 12u;
static constexpr uint32_t QH_GQA_TILE_T = 8u;
static constexpr uint32_t QH_VERIFY_INVARIANT_SEQS = 8u;

static __device__ __forceinline__ uint32_t QwenGqaSplitLen(uint32_t limit)
{
	if ( limit <= 64u )
		return limit > 0u ? limit : 1u;
	const uint32_t raw = (limit + QH_GQA_MAX_SPLITS - 1u) / QH_GQA_MAX_SPLITS;
	const uint32_t aligned = (raw + 63u) & ~63u;
	return aligned < 64u ? 64u : aligned;
}

template<class Geometry, uint32_t HEADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t QKV_WIDTH>
__global__ __launch_bounds__((HEADS / KV_HEADS) * 32u, 2)
static void QwenGqaSplitDecodeKernel(
	LmKvShardView view, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ attn_local,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position, const uint32_t *__restrict__ fork_positions,
	const uint32_t *__restrict__ slot_offsets, float qk_scale, float *__restrict__ partials,
	uint16_t *__restrict__ merged_out, uint32_t rows)
{
	constexpr uint32_t GROUP = HEADS / KV_HEADS;
	constexpr uint32_t THREADS = GROUP * 32u;
	constexpr uint32_t ELEMS = HEAD_DIM / 32u;
	constexpr uint32_t SLOT_STRIDE = HEAD_DIM + 4u;
	constexpr uint32_t QKV_V = QKV_WIDTH >> 3u;
	constexpr uint32_t HD_V = HEAD_DIM >> 3u;
	__shared__ __align__(16) uint4 s_k[2][QH_GQA_TILE_T][32];
	__shared__ __align__(16) uint4 s_v[2][QH_GQA_TILE_T][32];
	const uint32_t row = blockIdx.x, kv_head = blockIdx.y, split = blockIdx.z;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = kv_head * GROUP + warp;
	LmDependentWait();
	if ( row >= rows || warp >= GROUP )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t seq = __ldg(sequence_of_row + row);
	if ( !LmKvCheckSequence(view.pages, row, seq) )
	{
		LmDependentRelease();
		return;
	}
	uint32_t limit = __ldg(context_length + seq);
	if ( row_position != 0 )
	{
		const uint32_t rp = __ldg(row_position + row);
		if ( rp != LM_KV_POSITION_UNUSED && rp + 1u < limit )
			limit = rp + 1u;
	}
	const uint32_t slen = QwenGqaSplitLen(limit);
	const uint32_t begin = split * slen;
	if ( begin >= limit )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t end = begin + slen < limit ? begin + slen : limit;
	const uint32_t fork = fork_positions != 0 && __ldg(fork_positions + row) < limit ? __ldg(fork_positions + row) : limit;
	const uint32_t slot_off = slot_offsets != 0 ? __ldg(slot_offsets + row) : 0u;
	float qf[ELEMS], acc[ELEMS];
	LmLoadBf16x8AsFloat(((const uint4 *)(query_bf16 + ((uint64_t)row * HEADS + head) * HEAD_DIM))[lane], qf);
	#pragma unroll
	for ( uint32_t i = 0u; i < ELEMS; ++i )
		acc[i] = 0.0f;
	float rmax = -INFINITY, rsum = 0.0f;
	auto stage_tile = [&](uint32_t st, uint32_t t0, uint32_t cnt) {
		const bool contiguous = (t0 + cnt <= fork || t0 >= fork);
		const uint32_t phys0 = t0 < fork ? t0 : t0 + slot_off;
		const uint8_t *page_base = contiguous ? LmKvShardSlotRequired<Geometry>(view, seq, phys0, row, LM_KV_ACCESS_READ) : (const uint8_t *)0;
		const uint32_t page_rem = Geometry::kPageSlots - (phys0 % Geometry::kPageSlots);
		const bool same_page = page_base != 0 && cnt <= page_rem;
		for ( uint32_t idx = threadIdx.x; idx < cnt * 32u; idx += THREADS )
		{
			const uint32_t dt = idx >> 5u, ln = idx & 31u;
			const uint32_t sem = t0 + dt;
			const uint8_t *slot = same_page ? (page_base + (uint64_t)dt * Geometry::kSlotBytes) :
				LmKvShardSlotRequired<Geometry>(view, seq, sem < fork ? sem : sem + slot_off, row, LM_KV_ACCESS_READ);
			if ( slot != 0 )
			{
				const uint4 *kp = (const uint4 *)((const uint16_t *)slot + kv_head * HEAD_DIM) + ln;
				const uint4 *vp = (const uint4 *)((const uint16_t *)slot + (KV_HEADS + kv_head) * HEAD_DIM) + ln;
				const uint32_t dk = (uint32_t)__cvta_generic_to_shared(&s_k[st][dt][ln]);
				const uint32_t dv = (uint32_t)__cvta_generic_to_shared(&s_v[st][dt][ln]);
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(kp) : "memory");
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dv), "l"(vp) : "memory");
			}
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	const uint32_t cnt0 = end - begin < QH_GQA_TILE_T ? end - begin : QH_GQA_TILE_T;
	stage_tile(0u, begin, cnt0);
	uint32_t cur = 0u;
	for ( uint32_t t = begin; t < end; t += QH_GQA_TILE_T )
	{
		const uint32_t cnt = end - t < QH_GQA_TILE_T ? end - t : QH_GQA_TILE_T;
		const uint32_t nt = t + QH_GQA_TILE_T;
		if ( nt < end )
			stage_tile(cur ^ 1u, nt, end - nt < QH_GQA_TILE_T ? end - nt : QH_GQA_TILE_T);
		if ( nt < end )
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		for ( uint32_t dt = 0u; dt < cnt; ++dt )
		{
			float kf[ELEMS], vf[ELEMS];
			LmLoadBf16x8AsFloat(s_k[cur][dt][lane], kf);
			LmLoadBf16x8AsFloat(s_v[cur][dt][lane], vf);
			float score = 0.0f;
			#pragma unroll
			for ( uint32_t i = 0u; i < ELEMS; ++i )
				score = fmaf(qf[i], kf[i], score);
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				score += __shfl_xor_sync(0xffffffffu, score, off);
			score *= qk_scale;
			const float prev = rmax;
			rmax = fmaxf(prev, score);
			const float rescale = __expf(prev - rmax);
			const float w = __expf(score - rmax);
			rsum = fmaf(rsum, rescale, w);
			const float vw = SparkMathAttentionValueWeight(w);
			#pragma unroll
			for ( uint32_t i = 0u; i < ELEMS; ++i )
				acc[i] = fmaf(acc[i], rescale, vw * vf[i]);
		}
		cur ^= 1u;
		if ( nt < end )
			__syncthreads();
	}
	if ( begin == 0u && end == limit )
	{
		const float denom = fmaxf(rsum, 1.0e-20f);
		float gf[ELEMS], out[ELEMS];
		LmLoadBf16x8AsFloat(((const uint4 *)attn_local)[(uint64_t)row * QKV_V + (head * 2u + 1u) * HD_V + lane], gf);
		#pragma unroll
		for ( uint32_t i = 0u; i < ELEMS; ++i )
			out[i] = SparkMathRoundBf16(acc[i] / denom) * SparkMathSigmoidFast(gf[i]);
		((uint4 *)merged_out)[((uint64_t)row * HEADS + head) * HD_V + lane] = LmPackUint4FloatToBf16(out);
		LmDependentRelease();
		return;
	}
	float *out = partials + (((uint64_t)row * QH_GQA_MAX_SPLITS + split) * HEADS + head) * SLOT_STRIDE;
	*(float4 *)(out + lane * ELEMS) = make_float4(acc[0], acc[1], acc[2], acc[3]);
	*(float4 *)(out + lane * ELEMS + 4u) = make_float4(acc[4], acc[5], acc[6], acc[7]);
	if ( lane == 0u )
	{
		out[HEAD_DIM] = rmax;
		out[HEAD_DIM + 1u] = rsum;
	}
	LmDependentRelease();
}

template<uint32_t HEADS, uint32_t HEAD_DIM, uint32_t QKV_WIDTH>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenGqaSplitReduceGateKernel(
	const float *__restrict__ partials, const uint16_t *__restrict__ attn_local,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position, uint16_t *__restrict__ merged_out, uint32_t rows)
{
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	constexpr uint32_t ELEMS = HEAD_DIM / 32u;
	constexpr uint32_t SLOT_STRIDE = HEAD_DIM + 4u;
	constexpr uint32_t QKV_V = QKV_WIDTH >> 3u;
	constexpr uint32_t HD_V = HEAD_DIM >> 3u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( row >= rows || head >= HEADS )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t seq = __ldg(sequence_of_row + row);
	uint32_t limit = __ldg(context_length + seq);
	if ( row_position != 0 )
	{
		const uint32_t rp = __ldg(row_position + row);
		if ( rp != LM_KV_POSITION_UNUSED && rp + 1u < limit )
			limit = rp + 1u;
	}
	const uint32_t slen = QwenGqaSplitLen(limit);
	const uint32_t nsplits = limit == 0u ? 1u : (limit + slen - 1u) / slen;
	if ( nsplits <= 1u )
	{
		LmDependentRelease();
		return;
	}
	float gmax = -INFINITY, gsum = 0.0f;
	float acc[ELEMS] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	for ( uint32_t s = 0u; s < nsplits; ++s )
	{
		const float *in = partials + (((uint64_t)row * QH_GQA_MAX_SPLITS + s) * HEADS + head) * SLOT_STRIDE;
		const float sm = in[HEAD_DIM];
		const float sl = in[HEAD_DIM + 1u];
		const float prev = gmax;
		gmax = fmaxf(prev, sm);
		const float sc_old = prev == -INFINITY ? 0.0f : __expf(prev - gmax);
		const float sc_new = sm == -INFINITY ? 0.0f : __expf(sm - gmax);
		gsum = gsum * sc_old + sl * sc_new;
		const float4 a0 = *(const float4 *)(in + lane * ELEMS);
		const float4 a1 = *(const float4 *)(in + lane * ELEMS + 4u);
		acc[0] = fmaf(acc[0], sc_old, a0.x * sc_new);
		acc[1] = fmaf(acc[1], sc_old, a0.y * sc_new);
		acc[2] = fmaf(acc[2], sc_old, a0.z * sc_new);
		acc[3] = fmaf(acc[3], sc_old, a0.w * sc_new);
		acc[4] = fmaf(acc[4], sc_old, a1.x * sc_new);
		acc[5] = fmaf(acc[5], sc_old, a1.y * sc_new);
		acc[6] = fmaf(acc[6], sc_old, a1.z * sc_new);
		acc[7] = fmaf(acc[7], sc_old, a1.w * sc_new);
	}
	const float denom = fmaxf(gsum, 1.0e-20f);
	float gf[ELEMS], out[ELEMS];
	LmLoadBf16x8AsFloat(((const uint4 *)attn_local)[(uint64_t)row * QKV_V + (head * 2u + 1u) * HD_V + lane], gf);
	#pragma unroll
	for ( uint32_t i = 0u; i < ELEMS; ++i )
		out[i] = SparkMathRoundBf16(acc[i] / denom) * SparkMathSigmoidFast(gf[i]);
	((uint4 *)merged_out)[((uint64_t)row * HEADS + head) * HD_V + lane] = LmPackUint4FloatToBf16(out);
	LmDependentRelease();
}

template<class Geometry, uint32_t HEADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t QKV_WIDTH>
__global__ __launch_bounds__((HEADS / KV_HEADS) * 32u, 2)
static void QwenGqaDirectDecodeGateKernel(
	LmKvShardView view, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ attn_local,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position, float qk_scale, uint16_t *__restrict__ merged_out, uint32_t rows)
{
	constexpr uint32_t GROUP = HEADS / KV_HEADS;
	constexpr uint32_t THREADS = GROUP * 32u;
	constexpr uint32_t ELEMS = HEAD_DIM / 32u;
	constexpr uint32_t QKV_V = QKV_WIDTH >> 3u;
	constexpr uint32_t HD_V = HEAD_DIM >> 3u;
	__shared__ __align__(16) uint4 s_k[2][QH_GQA_TILE_T][32];
	__shared__ __align__(16) uint4 s_v[2][QH_GQA_TILE_T][32];
	const uint32_t row = blockIdx.x, kv_head = blockIdx.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = kv_head * GROUP + warp;
	LmDependentWait();
	if ( row >= rows || warp >= GROUP )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t seq = __ldg(sequence_of_row + row);
	if ( !LmKvCheckSequence(view.pages, row, seq) )
	{
		LmDependentRelease();
		return;
	}
	uint32_t limit = __ldg(context_length + seq);
	if ( row_position != 0 )
	{
		const uint32_t rp = __ldg(row_position + row);
		if ( rp != LM_KV_POSITION_UNUSED && rp + 1u < limit )
			limit = rp + 1u;
	}
	float qf[ELEMS], acc[ELEMS];
	LmLoadBf16x8AsFloat(((const uint4 *)(query_bf16 + ((uint64_t)row * HEADS + head) * HEAD_DIM))[lane], qf);
	#pragma unroll
	for ( uint32_t i = 0u; i < ELEMS; ++i )
		acc[i] = 0.0f;
	float rmax = -INFINITY, rsum = 0.0f;
	auto stage_tile = [&](uint32_t st, uint32_t t0, uint32_t cnt) {
		const uint8_t *page_base = LmKvShardSlotRequired<Geometry>(view, seq, t0, row, LM_KV_ACCESS_READ);
		const uint32_t page_rem = Geometry::kPageSlots - (t0 % Geometry::kPageSlots);
		const bool same_page = page_base != 0 && cnt <= page_rem;
		for ( uint32_t idx = threadIdx.x; idx < cnt * 32u; idx += THREADS )
		{
			const uint32_t dt = idx >> 5u, ln = idx & 31u;
			const uint8_t *slot = same_page ? (page_base + (uint64_t)dt * Geometry::kSlotBytes) :
				LmKvShardSlotRequired<Geometry>(view, seq, t0 + dt, row, LM_KV_ACCESS_READ);
			if ( slot != 0 )
			{
				const uint4 *kp = (const uint4 *)((const uint16_t *)slot + kv_head * HEAD_DIM) + ln;
				const uint4 *vp = (const uint4 *)((const uint16_t *)slot + (KV_HEADS + kv_head) * HEAD_DIM) + ln;
				const uint32_t dk = (uint32_t)__cvta_generic_to_shared(&s_k[st][dt][ln]);
				const uint32_t dv = (uint32_t)__cvta_generic_to_shared(&s_v[st][dt][ln]);
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dk), "l"(kp) : "memory");
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dv), "l"(vp) : "memory");
			}
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	if ( limit > 0u )
	{
		const uint32_t cnt0 = limit < QH_GQA_TILE_T ? limit : QH_GQA_TILE_T;
		stage_tile(0u, 0u, cnt0);
		uint32_t cur = 0u;
		for ( uint32_t t = 0u; t < limit; t += QH_GQA_TILE_T )
		{
			const uint32_t cnt = limit - t < QH_GQA_TILE_T ? limit - t : QH_GQA_TILE_T;
			const uint32_t nt = t + QH_GQA_TILE_T;
			if ( nt < limit )
				stage_tile(cur ^ 1u, nt, limit - nt < QH_GQA_TILE_T ? limit - nt : QH_GQA_TILE_T);
			if ( nt < limit )
				asm volatile("cp.async.wait_group 1;\n" ::: "memory");
			else
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			for ( uint32_t dt = 0u; dt < cnt; ++dt )
			{
				float kf[ELEMS], vf[ELEMS];
				LmLoadBf16x8AsFloat(s_k[cur][dt][lane], kf);
				LmLoadBf16x8AsFloat(s_v[cur][dt][lane], vf);
				float score = 0.0f;
				#pragma unroll
				for ( uint32_t i = 0u; i < ELEMS; ++i )
					score = fmaf(qf[i], kf[i], score);
				#pragma unroll
				for ( uint32_t off = 16u; off > 0u; off >>= 1u )
					score += __shfl_xor_sync(0xffffffffu, score, off);
				score *= qk_scale;
				const float prev = rmax;
				rmax = fmaxf(prev, score);
				const float rescale = __expf(prev - rmax);
				const float w = __expf(score - rmax);
				rsum = fmaf(rsum, rescale, w);
				const float vw = SparkMathAttentionValueWeight(w);
				#pragma unroll
				for ( uint32_t i = 0u; i < ELEMS; ++i )
					acc[i] = fmaf(acc[i], rescale, vw * vf[i]);
			}
			cur ^= 1u;
			if ( nt < limit )
				__syncthreads();
		}
	}
	const float denom = fmaxf(rsum, 1.0e-20f);
	float gf[ELEMS], out[ELEMS];
	LmLoadBf16x8AsFloat(((const uint4 *)attn_local)[(uint64_t)row * QKV_V + (head * 2u + 1u) * HD_V + lane], gf);
	#pragma unroll
	for ( uint32_t i = 0u; i < ELEMS; ++i )
		out[i] = SparkMathRoundBf16(acc[i] / denom) * SparkMathSigmoidFast(gf[i]);
	((uint4 *)merged_out)[((uint64_t)row * HEADS + head) * HD_V + lane] = LmPackUint4FloatToBf16(out);
	LmDependentRelease();
}

template<uint32_t HEADS, uint32_t HEAD_DIM, uint32_t QKV_WIDTH>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenHybridTp1MergeGateKernel(const float *__restrict__ received, const uint16_t *__restrict__ attn_local,
	uint16_t *__restrict__ output_bf16, uint32_t rows)
{
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	constexpr uint32_t WIDTH = HEADS * HEAD_DIM;
	constexpr uint32_t SLOTS = HEADS + 1u;
	constexpr uint32_t RECORD = WIDTH + 2u * SLOTS;
	constexpr uint32_t VECS = HEAD_DIM >> 1u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( row >= rows || head >= HEADS )
	{
		LmDependentRelease();
		return;
	}
	const float *row_base = received + (uint64_t)row * RECORD;
	const float denom = fmaxf(row_base[WIDTH + 2u * head + 1u], 1.0e-20f);
	const float2 *h2 = (const float2 *)(row_base + (uint64_t)head * HEAD_DIM);
	const uint32_t *g2 = (const uint32_t *)(attn_local + (uint64_t)row * QKV_WIDTH + (uint64_t)(head * 2u + 1u) * HEAD_DIM);
	uint32_t *o2 = (uint32_t *)(output_bf16 + ((uint64_t)row * HEADS + head) * HEAD_DIM);
	#pragma unroll
	for ( uint32_t v = lane; v < VECS; v += LM_WARP_LANES )
	{
		const float2 a = h2[v];
		const uint32_t g = g2[v];
		const float g0 = __uint_as_float(g << 16u), g1 = __uint_as_float(g & 0xffff0000u);
		const float o0 = SparkMathRoundBf16(a.x / denom) * SparkMathSigmoidFast(g0);
		const float o1 = SparkMathRoundBf16(a.y / denom) * SparkMathSigmoidFast(g1);
		o2[v] = LmPackBf16Pair(o0, o1);
	}
	LmDependentRelease();
}

template<uint32_t VALUE_DIM, bool GATE_FROM_LOCAL = false>
__global__ __launch_bounds__(QH_THREADS, 1)
static void QwenHybridMultiMergeGateKernel(const float *__restrict__ received, uint64_t source_stride, uint32_t sources,
	uint32_t heads, uint32_t rank, const uint16_t *__restrict__ gate_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows,
	uint32_t gate_stride = 0u)
{
	__shared__ float scales[LM_GQA_SHARD_MERGE_SLOTS][SPARK_KV_SHARD_MAX_DEGREE];
	__shared__ float denom_shared[LM_GQA_SHARD_MERGE_SLOTS];
	const uint32_t row = blockIdx.x;
	const uint32_t width = LmGqaShardWidth(heads, VALUE_DIM, sources), slots = LmGqaShardHeadSlots(heads, VALUE_DIM, sources);
	const uint32_t record = width + 2u * slots, origin = (rank * width) / VALUE_DIM;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row >= rows || width == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t used_slots = (rank * width + width - 1u) / VALUE_DIM - origin + 1u;
	const float *row_base = received + (uint64_t)row * record;
	const uint16_t *gate_row = gate_bf16 + (uint64_t)row * (GATE_FROM_LOCAL ? gate_stride : width);
	uint16_t *out_row = output_bf16 + (uint64_t)row * width;
	for ( uint32_t slot_base = 0u; slot_base < used_slots; slot_base += LM_GQA_SHARD_MERGE_SLOTS )
	{
		const uint32_t slot_end = slot_base + LM_GQA_SHARD_MERGE_SLOTS < used_slots ? slot_base + LM_GQA_SHARD_MERGE_SLOTS : used_slots;
		const uint32_t chunk_slots = slot_end - slot_base;
		const uint32_t elem_begin = slot_base == 0u ? 0u : (origin + slot_base) * VALUE_DIM - rank * width;
		const uint32_t elem_end = slot_end == used_slots ? width : (origin + slot_end) * VALUE_DIM - rank * width;
		for ( uint32_t local_slot = warp; local_slot < chunk_slots; local_slot += 8u )
		{
			const uint32_t slot_index = slot_base + local_slot;
			float gm = -INFINITY;
			const float den = LmWarpMergeScales(row_base + width + 2u * slot_index, source_stride, sources, scales[local_slot], &gm, -INFINITY);
			if ( lane == 0u )
				denom_shared[local_slot] = fmaxf(den, 1.0e-20f);
		}
		__syncthreads();
		for ( uint32_t v = (elem_begin >> 1u) + threadIdx.x; v < (elem_end >> 1u); v += QH_THREADS )
		{
			const uint32_t elem = v << 1u;
			const uint32_t slot = (rank * width + elem) / VALUE_DIM - origin - slot_base;
			const float d = denom_shared[slot];
			float m0 = 0.0f, m1 = 0.0f;
			for ( uint32_t s = 0u; s < sources; ++s )
			{
				const float w = scales[slot][s];
				const float2 x = __ldg(((const float2 *)(row_base + (uint64_t)s * source_stride)) + v);
				m0 = fmaf(x.x, w, m0);
				m1 = fmaf(x.y, w, m1);
			}
			uint32_t g;
			if constexpr ( GATE_FROM_LOCAL )
			{
				const uint32_t local_h = elem / VALUE_DIM, within = elem % VALUE_DIM;
				g = *(const uint32_t *)(gate_row + (uint64_t)(local_h * 2u + 1u) * VALUE_DIM + within);
			}
			else
				g = ((const uint32_t *)gate_row)[v];
			const float g0 = __uint_as_float(g << 16u), g1 = __uint_as_float(g & 0xffff0000u);
			const float o0 = SparkMathRoundBf16(m0 / d) * SparkMathSigmoidFast(g0);
			const float o1 = SparkMathRoundBf16(m1 / d) * SparkMathSigmoidFast(g1);
			((uint32_t *)out_row)[v] = LmPackBf16Pair(o0, o1);
		}
		if ( slot_end < used_slots )
			__syncthreads();
	}
	LmDependentRelease();
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
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
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
	LmDependentRelease();
}

template<uint32_t KEY_DIM>
static __device__ __forceinline__ void QwenGdnWriteSplitChannel(uint32_t channel, uint32_t row, uint32_t gdn_qk, uint32_t gdn_v,
	uint32_t heads, uint32_t repeat, uint16_t val, uint16_t *__restrict__ query_heads, uint16_t *__restrict__ key_heads,
	uint16_t *__restrict__ value)
{
	if ( channel < gdn_qk )
	{
		const uint32_t head = channel / KEY_DIM, within = channel % KEY_DIM;
		uint16_t *dst = query_heads + ((uint64_t)row * heads + head * repeat) * KEY_DIM + within;
		for ( uint32_t r = 0u; r < repeat; ++r )
			dst[(uint64_t)r * KEY_DIM] = val;
	}
	else if ( channel < 2u * gdn_qk )
	{
		const uint32_t kc = channel - gdn_qk;
		const uint32_t head = kc / KEY_DIM, within = kc % KEY_DIM;
		uint16_t *dst = key_heads + ((uint64_t)row * heads + head * repeat) * KEY_DIM + within;
		for ( uint32_t r = 0u; r < repeat; ++r )
			dst[(uint64_t)r * KEY_DIM] = val;
	}
	else
		value[(uint64_t)row * gdn_v + (channel - 2u * gdn_qk)] = val;
}

template<uint32_t KERNEL, uint32_t KEY_DIM>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenGdnConvSplitSerialKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count,
	const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight, const uint16_t *__restrict__ ba,
	const float *__restrict__ a_log, const float *__restrict__ dt_bias, uint16_t *__restrict__ query_heads,
	uint16_t *__restrict__ key_heads, uint16_t *__restrict__ value, float *__restrict__ retention,
	float *__restrict__ write_gate, uint16_t *__restrict__ replay_qkv, uint32_t gdn_qk, uint32_t gdn_v,
	uint32_t heads, uint32_t input_stride, uint32_t conv_blocks, uint32_t sequences, uint32_t commit,
	const uint32_t *__restrict__ sequence_row_indices)
{
	const uint32_t sequence = blockIdx.x;
	if ( sequence >= sequences )
	{
		LmDependentWait();
		LmDependentRelease();
		return;
	}
	if ( blockIdx.y >= conv_blocks )
	{
		constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
		constexpr uint32_t KEY_V = KEY_DIM >> 2u;
		const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
		const uint32_t head = (blockIdx.y - conv_blocks) * WARPS + warp;
		LmDependentWait();
		if ( head < heads )
		{
			const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
			uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
			if ( sequence_row_count != 0 )
				end = begin + sequence_row_count[sequence];
			const float a = a_log[head], dt = dt_bias[head];
			for ( uint32_t ordinal = begin; ordinal < end; ++ordinal )
			{
				const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
				const uint64_t ba_base = (uint64_t)row * 2u * heads;
				float factor = 0.0f;
				if ( lane == 0u )
				{
					const float shifted = LmBf16ToFloat(ba[ba_base + heads + head]) + dt;
					factor = SparkMathGatedDeltaDecay(a, shifted);
					write_gate[(uint64_t)row * heads + head] = SparkMathSigmoidFast(LmBf16ToFloat(ba[ba_base + head]));
				}
				factor = __shfl_sync(0xffffffffu, factor, 0);
				const float4 f4 = make_float4(factor, factor, factor, factor);
				float4 *ret4 = (float4 *)(retention + ((uint64_t)row * heads + head) * KEY_DIM);
				#pragma unroll
				for ( uint32_t v = lane; v < KEY_V; v += LM_WARP_LANES )
					ret4[v] = f4;
			}
		}
		LmDependentRelease();
		return;
	}
	const uint32_t channels = 2u * gdn_qk + gdn_v;
	const uint32_t channel = blockIdx.y * QH_THREADS + threadIdx.x;
	float taps_weight[KERNEL];
	if ( channel >= channels )
	{
		LmDependentWait();
		LmDependentRelease();
		return;
	}
	LmCausalConvLoadWeights<KERNEL, uint16_t>(weight, channel, taps_weight);
	LmDependentWait();
	const uint32_t repeat = heads / (gdn_qk / KEY_DIM);
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	uint16_t *slot = window + (uint64_t)state_index[sequence] * channels * KERNEL;
	uint16_t taps[KERNEL];
	if constexpr ( KERNEL == 4u )
	{
		const uint2 w = *(const uint2 *)(slot + (uint64_t)channel * 4u);
		taps[0] = (uint16_t)(w.x & 0xffffu);
		taps[1] = (uint16_t)(w.x >> 16u);
		taps[2] = (uint16_t)(w.y & 0xffffu);
		taps[3] = (uint16_t)(w.y >> 16u);
	}
	else
	{
		#pragma unroll
		for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
			taps[tap] = slot[(uint64_t)channel * KERNEL + tap];
	}
	for ( uint32_t ordinal = begin; ordinal < end; ++ordinal )
	{
		const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
		#pragma unroll
		for ( uint32_t tap = 0u; tap + 1u < KERNEL; ++tap )
			taps[tap] = taps[tap + 1u];
		const uint16_t in_val = input_bf16[(uint64_t)row * input_stride + channel];
		taps[KERNEL - 1u] = in_val;
		if ( replay_qkv != 0 )
			replay_qkv[(uint64_t)row * channels + channel] = in_val;
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
			total += LmBf16ToFloat(taps[tap]) * taps_weight[tap];
		const uint16_t out = LmFloatToBf16(SparkMathSwishFast(LmBf16ToFloat(LmFloatToBf16(total))));
		QwenGdnWriteSplitChannel<KEY_DIM>(channel, row, gdn_qk, gdn_v, heads, repeat, out, query_heads, key_heads, value);
	}
	if ( commit != 0u )
	{
		if constexpr ( KERNEL == 4u )
		{
			uint2 w;
			w.x = (uint32_t)taps[0] | ((uint32_t)taps[1] << 16u);
			w.y = (uint32_t)taps[2] | ((uint32_t)taps[3] << 16u);
			*(uint2 *)(slot + (uint64_t)channel * 4u) = w;
		}
		else
		{
			#pragma unroll
			for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
				slot[(uint64_t)channel * KERNEL + tap] = taps[tap];
		}
	}
	LmDependentRelease();
}

template<uint32_t KERNEL, uint32_t KEY_DIM>
__global__ __launch_bounds__(LM_CAUSAL_CONV_TILE_THREADS, 1)
static void QwenGdnConvSplitTiledKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count,
	const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight, uint16_t *__restrict__ query_heads,
	uint16_t *__restrict__ key_heads, uint16_t *__restrict__ value, uint16_t *__restrict__ replay_qkv,
	uint32_t gdn_qk, uint32_t gdn_v, uint32_t heads, uint32_t input_stride, uint32_t sequences, uint32_t commit,
	const uint32_t *__restrict__ sequence_row_indices)
{
	constexpr uint32_t HALO = KERNEL - 1u;
	constexpr uint32_t GROUPS = LM_CAUSAL_CONV_TILE_THREADS / LM_CAUSAL_CONV_TILE_CHANNELS;
	__shared__ uint16_t tile[HALO + LM_CAUSAL_CONV_TILE_ROWS][LM_CAUSAL_CONV_TILE_CHANNELS];
	const uint32_t channels = 2u * gdn_qk + gdn_v;
	const uint32_t sequence = blockIdx.x, lane = threadIdx.x % LM_CAUSAL_CONV_TILE_CHANNELS, group = threadIdx.x / LM_CAUSAL_CONV_TILE_CHANNELS;
	const uint32_t channel = blockIdx.y * LM_CAUSAL_CONV_TILE_CHANNELS + lane;
	const bool live = channel < channels;
	float taps_weight[KERNEL];
	uint16_t carried[HALO];
	if ( sequence >= sequences )
	{
		LmDependentWait();
		LmDependentRelease();
		return;
	}
	if ( live )
		LmCausalConvLoadWeights<KERNEL, uint16_t>(weight, channel, taps_weight);
	else
		#pragma unroll
		for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
			taps_weight[tap] = 0.0f;
	LmDependentWait();
	const uint32_t repeat = heads / (gdn_qk / KEY_DIM);
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	uint16_t *slot = window + (uint64_t)state_index[sequence] * channels * KERNEL;
	if ( group == 0u && live )
	{
		if constexpr ( KERNEL == 4u )
		{
			const uint2 w = *(const uint2 *)(slot + (uint64_t)channel * 4u);
			tile[0][lane] = (uint16_t)(w.x >> 16u);
			tile[1][lane] = (uint16_t)(w.y & 0xffffu);
			tile[2][lane] = (uint16_t)(w.y >> 16u);
		}
		else
		{
			#pragma unroll
			for ( uint32_t tap = 0u; tap < HALO; ++tap )
				tile[tap][lane] = slot[(uint64_t)channel * KERNEL + 1u + tap];
		}
	}
	for ( uint32_t base = begin; base < end; base += LM_CAUSAL_CONV_TILE_ROWS )
	{
		const uint32_t count = end - base < LM_CAUSAL_CONV_TILE_ROWS ? end - base : LM_CAUSAL_CONV_TILE_ROWS;
		if ( base != begin )
			__syncthreads();
		for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
			const uint16_t in_val = input_bf16[(uint64_t)row * input_stride + channel];
			tile[HALO + offset][lane] = in_val;
			if ( replay_qkv != 0 )
				replay_qkv[(uint64_t)row * channels + channel] = in_val;
		}
		__syncthreads();
		for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
				total += LmBf16ToFloat(tile[offset + tap][lane]) * taps_weight[tap];
			const uint16_t out = LmFloatToBf16(SparkMathSwishFast(LmBf16ToFloat(LmFloatToBf16(total))));
			QwenGdnWriteSplitChannel<KEY_DIM>(channel, row, gdn_qk, gdn_v, heads, repeat, out, query_heads, key_heads, value);
		}
		const bool last_tile = (base + count == end);
		if ( !last_tile || commit != 0u )
		{
			__syncthreads();
			if ( group == 0u && live )
			{
				if ( commit != 0u && last_tile )
				{
					if constexpr ( KERNEL == 4u )
					{
						uint2 w;
						w.x = (uint32_t)tile[count - 1u][lane] | ((uint32_t)tile[count][lane] << 16u);
						w.y = (uint32_t)tile[count + 1u][lane] | ((uint32_t)tile[count + 2u][lane] << 16u);
						*(uint2 *)(slot + (uint64_t)channel * 4u) = w;
					}
					else
					{
						#pragma unroll
						for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
							slot[(uint64_t)channel * KERNEL + tap] = tile[count - 1u + tap][lane];
					}
				}
				if ( !last_tile )
				{
					#pragma unroll
					for ( uint32_t tap = 0u; tap < HALO; ++tap )
						carried[tap] = tile[count + tap][lane];
					#pragma unroll
					for ( uint32_t tap = 0u; tap < HALO; ++tap )
						tile[tap][lane] = carried[tap];
				}
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t KERNEL, uint32_t KEY_DIM>
__global__ __launch_bounds__(LM_CAUSAL_CONV_TILE_THREADS, 2)
static void QwenGdnConvPrefill2DKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count,
	const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight, uint16_t *__restrict__ query_qk,
	uint16_t *__restrict__ key_qk, uint16_t *__restrict__ value, uint32_t gdn_qk, uint32_t gdn_v,
	uint32_t input_stride, uint32_t sequences, uint32_t commit_inline, const uint32_t *__restrict__ sequence_row_indices)
{
	constexpr uint32_t HALO = KERNEL - 1u;
	constexpr uint32_t GROUPS = LM_CAUSAL_CONV_TILE_THREADS / LM_CAUSAL_CONV_TILE_CHANNELS;
	__shared__ uint16_t tile[HALO + LM_CAUSAL_CONV_TILE_ROWS][LM_CAUSAL_CONV_TILE_CHANNELS];
	const uint32_t channels = 2u * gdn_qk + gdn_v;
	const uint32_t sequence = blockIdx.x, lane = threadIdx.x % LM_CAUSAL_CONV_TILE_CHANNELS, group = threadIdx.x / LM_CAUSAL_CONV_TILE_CHANNELS;
	const uint32_t channel = blockIdx.y * LM_CAUSAL_CONV_TILE_CHANNELS + lane;
	const uint32_t tile_idx = blockIdx.z;
	const bool live = channel < channels;
	float taps_weight[KERNEL];
	if ( sequence >= sequences )
	{
		LmDependentWait();
		LmDependentRelease();
		return;
	}
	if ( live )
		LmCausalConvLoadWeights<KERNEL, uint16_t>(weight, channel, taps_weight);
	else
		#pragma unroll
		for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
			taps_weight[tap] = 0.0f;
	LmDependentWait();
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	const uint32_t base = begin + tile_idx * LM_CAUSAL_CONV_TILE_ROWS;
	if ( base >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t count = end - base < LM_CAUSAL_CONV_TILE_ROWS ? end - base : LM_CAUSAL_CONV_TILE_ROWS;
	uint16_t *slot = window + (uint64_t)state_index[sequence] * channels * KERNEL;
	if ( group == 0u && live )
	{
		if ( tile_idx == 0u )
		{
			if constexpr ( KERNEL == 4u )
			{
				const uint2 w = *(const uint2 *)(slot + (uint64_t)channel * 4u);
				tile[0][lane] = (uint16_t)(w.x >> 16u);
				tile[1][lane] = (uint16_t)(w.y & 0xffffu);
				tile[2][lane] = (uint16_t)(w.y >> 16u);
			}
			else
				#pragma unroll
				for ( uint32_t tap = 0u; tap < HALO; ++tap )
					tile[tap][lane] = slot[(uint64_t)channel * KERNEL + 1u + tap];
		}
		else
		{
			#pragma unroll
			for ( uint32_t tap = 0u; tap < HALO; ++tap )
			{
				const uint32_t h_ord = base - HALO + tap;
				const uint32_t h_row = sequence_row_indices != 0 ? sequence_row_indices[h_ord] : h_ord;
				tile[tap][lane] = __ldg(input_bf16 + (uint64_t)h_row * input_stride + channel);
			}
		}
	}
	for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
	{
		const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
		tile[HALO + offset][lane] = __ldg(input_bf16 + (uint64_t)row * input_stride + channel);
	}
	__syncthreads();
	for ( uint32_t offset = group; offset < count && live; offset += GROUPS )
	{
		const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
			total += LmBf16ToFloat(tile[offset + tap][lane]) * taps_weight[tap];
		const uint16_t out = LmFloatToBf16(SparkMathSwishFast(LmBf16ToFloat(LmFloatToBf16(total))));
		if ( channel < gdn_qk )
			query_qk[(uint64_t)row * gdn_qk + channel] = out;
		else if ( channel < 2u * gdn_qk )
			key_qk[(uint64_t)row * gdn_qk + (channel - gdn_qk)] = out;
		else
			value[(uint64_t)row * gdn_v + (channel - 2u * gdn_qk)] = out;
	}
	if ( commit_inline != 0u && base + count == end )
	{
		__syncthreads();
		if ( group == 0u && live )
		{
			if constexpr ( KERNEL == 4u )
			{
				uint2 w;
				w.x = (uint32_t)tile[count - 1u][lane] | ((uint32_t)tile[count][lane] << 16u);
				w.y = (uint32_t)tile[count + 1u][lane] | ((uint32_t)tile[count + 2u][lane] << 16u);
				*(uint2 *)(slot + (uint64_t)channel * 4u) = w;
			}
			else
				#pragma unroll
				for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
					slot[(uint64_t)channel * KERNEL + tap] = tile[count - 1u + tap][lane];
		}
	}
	LmDependentRelease();
}

template<uint32_t KERNEL>
__global__ __launch_bounds__(QH_THREADS, 4)
static void QwenGdnConvWindowCommitKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count,
	const uint16_t *__restrict__ input_bf16, uint32_t channels, uint32_t input_stride, uint32_t sequences,
	const uint32_t *__restrict__ sequence_row_indices)
{
	const uint32_t sequence = blockIdx.x, channel = blockIdx.y * QH_THREADS + threadIdx.x;
	LmDependentWait();
	if ( sequence < sequences && channel < channels )
	{
		const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
		uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
		if ( sequence_row_count != 0 )
			end = begin + sequence_row_count[sequence];
		const uint32_t count = end > begin ? end - begin : 0u;
		if ( count != 0u )
		{
			uint16_t *slot = window + (uint64_t)state_index[sequence] * channels * KERNEL;
			if constexpr ( KERNEL == 4u )
			{
				uint16_t o0, o1, o2, o3;
				if ( count >= 4u )
				{
					const uint32_t r0 = sequence_row_indices != 0 ? sequence_row_indices[end - 4u] : (end - 4u);
					const uint32_t r1 = sequence_row_indices != 0 ? sequence_row_indices[end - 3u] : (end - 3u);
					const uint32_t r2 = sequence_row_indices != 0 ? sequence_row_indices[end - 2u] : (end - 2u);
					const uint32_t r3 = sequence_row_indices != 0 ? sequence_row_indices[end - 1u] : (end - 1u);
					o0 = __ldg(input_bf16 + (uint64_t)r0 * input_stride + channel);
					o1 = __ldg(input_bf16 + (uint64_t)r1 * input_stride + channel);
					o2 = __ldg(input_bf16 + (uint64_t)r2 * input_stride + channel);
					o3 = __ldg(input_bf16 + (uint64_t)r3 * input_stride + channel);
				}
				else
				{
					const uint2 prev = *(const uint2 *)(slot + (uint64_t)channel * 4u);
					const uint64_t p64 = (uint64_t)prev.x | ((uint64_t)prev.y << 32u);
					auto pick = [&](uint32_t tap) -> uint16_t
					{
						if ( count + tap < 4u )
							return (uint16_t)((p64 >> ((count + tap) * 16u)) & 0xffffu);
						const uint32_t ord = begin + (count + tap - 4u);
						const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ord] : ord;
						return __ldg(input_bf16 + (uint64_t)row * input_stride + channel);
					};
					o0 = pick(0u);
					o1 = pick(1u);
					o2 = pick(2u);
					o3 = pick(3u);
				}
				uint2 w;
				w.x = (uint32_t)o0 | ((uint32_t)o1 << 16u);
				w.y = (uint32_t)o2 | ((uint32_t)o3 << 16u);
				*(uint2 *)(slot + (uint64_t)channel * 4u) = w;
			}
			else
			{
				uint16_t out[KERNEL];
				if ( count >= KERNEL )
				{
					#pragma unroll
					for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
					{
						const uint32_t ord = end - KERNEL + tap;
						const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ord] : ord;
						out[tap] = __ldg(input_bf16 + (uint64_t)row * input_stride + channel);
					}
				}
				else
				{
					#pragma unroll
					for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
					{
						if ( count + tap < KERNEL )
							out[tap] = slot[(uint64_t)channel * KERNEL + count + tap];
						else
						{
							const uint32_t ord = begin + (count + tap - KERNEL);
							const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[ord] : ord;
							out[tap] = __ldg(input_bf16 + (uint64_t)row * input_stride + channel);
						}
					}
				}
				#pragma unroll
				for ( uint32_t tap = 0u; tap < KERNEL; ++tap )
					slot[(uint64_t)channel * KERNEL + tap] = out[tap];
			}
		}
	}
	LmDependentRelease();
}

#define QH_GDN_CHUNK 32u

template<uint32_t KEY_DIM, uint32_t COL_SLICE, uint32_t K_SPLIT>
struct QwenGdnChunkShared
{
	static constexpr uint32_t V_GROUPS = COL_SLICE / 16u;
	static constexpr uint32_t REDUCE_STRIDE = (QH_GDN_CHUNK / 8u) * 4u * 32u;
	__align__(16) float k_hat[QH_GDN_CHUNK][KEY_DIM + 4u];
	__align__(16) float q_hat[QH_GDN_CHUNK][KEY_DIM + 4u];
	__align__(16) float val[QH_GDN_CHUNK][COL_SLICE + 4u];
	float tri_a[QH_GDN_CHUNK][QH_GDN_CHUNK + 1u];
	float tri_m[QH_GDN_CHUNK][QH_GDN_CHUNK + 1u];
	float tri_x[QH_GDN_CHUNK][QH_GDN_CHUNK + 1u];
	__align__(16) float g_cum[QH_GDN_CHUNK];
	__align__(16) float exp_g[QH_GDN_CHUNK];
	__align__(16) float exp_tail_diff[QH_GDN_CHUNK];
	__align__(16) float beta[QH_GDN_CHUNK];
	float tail_decay;
	__align__(16) uint16_t raw_key[QH_GDN_CHUNK][KEY_DIM];
	__align__(16) uint16_t raw_query[QH_GDN_CHUNK][KEY_DIM];
	__align__(16) uint16_t raw_value[QH_GDN_CHUNK][COL_SLICE];
	float reduce[K_SPLIT > 1u ? V_GROUPS * (K_SPLIT - 1u) * REDUCE_STRIDE : 1u];
};

template<uint32_t KEY_DIM, uint32_t VALUE_DIM, uint32_t COL_SLICE, uint32_t K_SPLIT, uint32_t THREADS>
static __device__ __forceinline__ void QwenGdnChunkFetch(QwenGdnChunkShared<KEY_DIM, COL_SLICE, K_SPLIT> &sh,
	const uint16_t *__restrict__ query_qk, const uint16_t *__restrict__ key_qk, const uint16_t *__restrict__ value_bf16,
	const uint32_t *__restrict__ sequence_row_indices, uint32_t first_row, uint32_t rows, uint32_t qk_head, uint32_t qk_heads,
	uint32_t head, uint32_t heads, uint32_t col_base)
{
	constexpr uint32_t KV = KEY_DIM / 8u, VV = COL_SLICE / 8u, PER = 2u * KV + VV;
	for ( uint32_t index = threadIdx.x; index < rows * PER; index += THREADS )
	{
		const uint32_t t = index / PER, piece = index % PER;
		const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[first_row + t] : first_row + t;
		if ( piece < KV )
		{
			const uint64_t base = ((uint64_t)row * qk_heads + qk_head) * KEY_DIM + piece * 8u;
			LmStreamCopy((uint32_t)__cvta_generic_to_shared(&sh.raw_key[t][piece * 8u]), key_qk + base, 16u);
		}
		else if ( piece < 2u * KV )
		{
			const uint32_t p = piece - KV;
			const uint64_t base = ((uint64_t)row * qk_heads + qk_head) * KEY_DIM + p * 8u;
			LmStreamCopy((uint32_t)__cvta_generic_to_shared(&sh.raw_query[t][p * 8u]), query_qk + base, 16u);
		}
		else
		{
			const uint32_t p = piece - 2u * KV;
			const uint64_t base = ((uint64_t)row * heads + head) * VALUE_DIM + col_base + p * 8u;
			LmStreamCopy((uint32_t)__cvta_generic_to_shared(&sh.raw_value[t][p * 8u]), value_bf16 + base, 16u);
		}
	}
	LmStreamCommit();
}

template<uint32_t KEY_DIM, uint32_t COL_SLICE, uint32_t K_SPLIT>
static __device__ __forceinline__ void QwenGdnChunkGramAndSolve(QwenGdnChunkShared<KEY_DIM, COL_SLICE, K_SPLIT> &sh,
	uint32_t rows, uint32_t warp, uint32_t lane, uint32_t gid, uint32_t tg)
{
	if ( warp < 6u )
	{
		const uint32_t tr = warp < 2u ? 0u : 16u;
		const uint32_t tc = warp < 2u ? (warp * 8u) : ((warp - 2u) * 8u);
		float da[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, dm[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		const uint32_t a_r0 = tr + gid, a_r1 = a_r0 + 8u, b_r = tc + gid;
		for ( uint32_t k0 = 0u; k0 < KEY_DIM; k0 += 8u )
		{
			const uint32_t kc0 = k0 + tg, kc1 = kc0 + 4u;
			uint32_t akh[4], akl[4], aqh[4], aql[4], bkh[2], bkl[2];
			LmDeltaSplitTf32(sh.k_hat[a_r0][kc0], akh[0], akl[0]);
			LmDeltaSplitTf32(sh.k_hat[a_r1][kc0], akh[1], akl[1]);
			LmDeltaSplitTf32(sh.k_hat[a_r0][kc1], akh[2], akl[2]);
			LmDeltaSplitTf32(sh.k_hat[a_r1][kc1], akh[3], akl[3]);
			LmDeltaSplitTf32(sh.q_hat[a_r0][kc0], aqh[0], aql[0]);
			LmDeltaSplitTf32(sh.q_hat[a_r1][kc0], aqh[1], aql[1]);
			LmDeltaSplitTf32(sh.q_hat[a_r0][kc1], aqh[2], aql[2]);
			LmDeltaSplitTf32(sh.q_hat[a_r1][kc1], aqh[3], aql[3]);
			LmDeltaSplitTf32(sh.k_hat[b_r][kc0], bkh[0], bkl[0]);
			LmDeltaSplitTf32(sh.k_hat[b_r][kc1], bkh[1], bkl[1]);
			LmDeltaMmaFp32(da, akh, akl, bkh, bkl);
			LmDeltaMmaFp32(dm, aqh, aql, bkh, bkl);
		}
		auto store_g = [&](uint32_t r, uint32_t c, float va, float vm)
		{
			const bool in = r < rows && c <= r;
			const float d = in ? expf(sh.g_cum[r] - sh.g_cum[c]) : 0.0f;
			sh.tri_a[r][c] = (in && c < r) ? (sh.beta[r] * d * va) : 0.0f;
			sh.tri_m[r][c] = in ? (d * vm) : 0.0f;
		};
		const uint32_t c0 = tc + 2u * tg, c1 = c0 + 1u;
		store_g(a_r0, c0, da[0], dm[0]);
		store_g(a_r0, c1, da[1], dm[1]);
		store_g(a_r1, c0, da[2], dm[2]);
		store_g(a_r1, c1, da[3], dm[3]);
	}
	else
	{
		const uint32_t tc = 16u + (warp - 6u) * 8u;
		const uint32_t r0 = gid, r1 = gid + 8u, c0 = tc + 2u * tg, c1 = c0 + 1u;
		sh.tri_a[r0][c0] = 0.0f; sh.tri_a[r0][c1] = 0.0f;
		sh.tri_a[r1][c0] = 0.0f; sh.tri_a[r1][c1] = 0.0f;
		sh.tri_m[r0][c0] = 0.0f; sh.tri_m[r0][c1] = 0.0f;
		sh.tri_m[r1][c0] = 0.0f; sh.tri_m[r1][c1] = 0.0f;
	}
	__syncthreads();
	if ( warp < 2u && lane < 16u )
	{
		const uint32_t blk = warp * 16u;
		float x[16];
		#pragma unroll
		for ( uint32_t t = 0u; t < 16u; ++t )
		{
			float total = t == lane ? 1.0f : 0.0f;
			#pragma unroll
			for ( uint32_t j = 0u; j < t; ++j )
				total = fmaf(-sh.tri_a[blk + t][blk + j], x[j], total);
			x[t] = (blk + t < rows) ? total : 0.0f;
		}
		if ( warp == 0u )
		{
			const float bj = sh.beta[lane];
			#pragma unroll
			for ( uint32_t t = 0u; t < 16u; ++t )
			{
				sh.tri_x[t][lane] = x[t] * bj;
				sh.tri_x[t][16u + lane] = 0.0f;
			}
		}
		else
		{
			const float bj = sh.beta[16u + lane];
			#pragma unroll
			for ( uint32_t t = 0u; t < 16u; ++t )
			{
				sh.tri_a[t][16u + lane] = x[t];
				sh.tri_x[16u + t][16u + lane] = x[t] * bj;
			}
		}
	}
	__syncthreads();
	if ( warp < 2u )
	{
		const uint32_t tc = warp * 8u;
		float tt[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		#pragma unroll
		for ( uint32_t k0 = 0u; k0 < 16u; k0 += 8u )
		{
			uint32_t ah[4], al[4], bh[2], bl[2];
			LmDeltaSplitTf32(sh.tri_a[16u + gid][k0 + tg], ah[0], al[0]);
			LmDeltaSplitTf32(sh.tri_a[24u + gid][k0 + tg], ah[1], al[1]);
			LmDeltaSplitTf32(sh.tri_a[16u + gid][k0 + tg + 4u], ah[2], al[2]);
			LmDeltaSplitTf32(sh.tri_a[24u + gid][k0 + tg + 4u], ah[3], al[3]);
			LmDeltaSplitTf32(sh.tri_x[k0 + tg][tc + gid], bh[0], bl[0]);
			LmDeltaSplitTf32(sh.tri_x[k0 + tg + 4u][tc + gid], bh[1], bl[1]);
			LmDeltaMmaFp32(tt, ah, al, bh, bl);
		}
		const uint32_t c0 = tc + 2u * tg, c1 = c0 + 1u;
		sh.tri_x[16u + gid][c0] = tt[0];
		sh.tri_x[16u + gid][c1] = tt[1];
		sh.tri_x[24u + gid][c0] = tt[2];
		sh.tri_x[24u + gid][c1] = tt[3];
		__syncwarp();
		float xx[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		#pragma unroll
		for ( uint32_t k0 = 0u; k0 < 16u; k0 += 8u )
		{
			uint32_t ah[4], al[4], bh[2], bl[2];
			LmDeltaSplitTf32(sh.tri_a[gid][16u + k0 + tg], ah[0], al[0]);
			LmDeltaSplitTf32(sh.tri_a[8u + gid][16u + k0 + tg], ah[1], al[1]);
			LmDeltaSplitTf32(sh.tri_a[gid][16u + k0 + tg + 4u], ah[2], al[2]);
			LmDeltaSplitTf32(sh.tri_a[8u + gid][16u + k0 + tg + 4u], ah[3], al[3]);
			LmDeltaSplitTf32(sh.tri_x[16u + k0 + tg][tc + gid], bh[0], bl[0]);
			LmDeltaSplitTf32(sh.tri_x[16u + k0 + tg + 4u][tc + gid], bh[1], bl[1]);
			LmDeltaMmaFp32(xx, ah, al, bh, bl);
		}
		__syncwarp();
		sh.tri_x[16u + gid][c0] = -xx[0];
		sh.tri_x[16u + gid][c1] = -xx[1];
		sh.tri_x[24u + gid][c0] = -xx[2];
		sh.tri_x[24u + gid][c1] = -xx[3];
	}
	__syncthreads();
}

template<uint32_t KEY_DIM, uint32_t VALUE_DIM, bool PRE_EDGE>
__global__ __launch_bounds__(LM_DELTA_COLUMN_THREADS, 2)
static void QwenGdnExactEdgeKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ query_qk, const uint16_t *__restrict__ key_qk, const uint16_t *__restrict__ value_bf16,
	const uint16_t *__restrict__ ba_bf16, const float *__restrict__ a_log, const float *__restrict__ dt_bias,
	uint16_t *__restrict__ output_bf16, uint32_t qk_heads, uint32_t heads, uint32_t sequences, uint32_t commit,
	const uint32_t *__restrict__ sequence_row_indices)
{
	constexpr uint32_t C = QH_GDN_CHUNK;
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	constexpr uint32_t PARTS = KEY_DIM / LM_WARP_LANES;
	__shared__ __align__(16) float state_s[KEY_DIM][LM_WARP_LANES];
	__shared__ __align__(16) float key_s[KEY_DIM];
	__shared__ __align__(16) float query_s[KEY_DIM];
	__shared__ __align__(16) float forget_s[KEY_DIM];
	__shared__ float value_s[LM_WARP_LANES];
	__shared__ float beta_s;
	__shared__ float coefficient_s[LM_WARP_LANES];
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	const uint32_t sequence = blockIdx.x, head = blockIdx.y, column_base = blockIdx.z * LM_WARP_LANES;
	if ( sequence >= sequences || head >= heads || column_base >= VALUE_DIM )
	{
		LmDependentWait();
		LmDependentRelease();
		return;
	}
	const float a = __ldg(a_log + head);
	const float dt_b = __ldg(dt_bias + head);
	LmDependentWait();
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	const uint32_t count = end > begin ? end - begin : 0u;
	uint32_t edge_begin = begin, edge_count = 0u;
	if ( count != 0u )
	{
		const uint32_t row0 = sequence_row_indices != 0 ? sequence_row_indices[begin] : begin;
		const uint32_t pos0 = positions != 0 ? positions[row0] : 0u;
		const uint32_t rem = pos0 & (C - 1u);
		const uint32_t lead = rem != 0u ? (C - rem < count ? C - rem : count) : 0u;
		const uint32_t mid_end = lead + ((count - lead) & ~(C - 1u));
		if constexpr ( PRE_EDGE )
		{
			edge_begin = begin;
			edge_count = lead;
		}
		else
		{
			edge_begin = begin + mid_end;
			edge_count = count - mid_end;
		}
	}
	if ( edge_count == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t repeat = heads / qk_heads;
	const uint32_t qk_head = head / repeat;
	const float q_scale = rsqrtf((float)KEY_DIM);
	float *state = (float *)(state_pool + ((uint64_t)state_index[sequence] * slot_bytes) + ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(float)));
	const uint32_t vec_col = (threadIdx.x & 7u) * 4u;
	#pragma unroll
	for ( uint32_t index = threadIdx.x >> 3u; index < KEY_DIM; index += LM_DELTA_COLUMN_THREADS >> 3u )
		LmLinearCopyAsync(&state_s[index][vec_col], state + (index * VALUE_DIM) + column_base + vec_col, 16u);
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	LmLinearCopyWait0();
	__syncthreads();
	for ( uint32_t step = 0u; step < edge_count; ++step )
	{
		const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[edge_begin + step] : (edge_begin + step);
		const uint64_t qk_base = (((uint64_t)row * qk_heads) + qk_head) * KEY_DIM;
		if ( warp == 0u )
		{
			float parts[PARTS], squares[PARTS];
			#pragma unroll
			for ( uint32_t part = 0u; part < PARTS; ++part )
			{
				parts[part] = LmBf16ToFloat(LmLinearLdg(key_qk + qk_base + part * LM_WARP_LANES + lane));
				squares[part] = parts[part] * parts[part];
			}
			const float inv = rsqrtf(LmDeltaRuleWarpTotal<WARPS, PARTS>(squares, lane) + 1e-6f);
			float factor = 0.0f;
			if ( lane == 0u )
			{
				const uint64_t ba_base = (uint64_t)row * 2u * heads;
				const float shifted = LmBf16ToFloat(__ldg(ba_bf16 + ba_base + heads + head)) + dt_b;
				factor = SparkMathGatedDeltaDecay(a, shifted);
				beta_s = SparkMathSigmoidFast(LmBf16ToFloat(__ldg(ba_bf16 + ba_base + head)));
			}
			factor = __shfl_sync(0xffffffffu, factor, 0);
			#pragma unroll
			for ( uint32_t part = 0u; part < PARTS; ++part )
			{
				key_s[part * LM_WARP_LANES + lane] = parts[part] * inv;
				forget_s[part * LM_WARP_LANES + lane] = factor;
			}
		}
		else if ( warp == 1u )
		{
			float parts[PARTS], squares[PARTS];
			#pragma unroll
			for ( uint32_t part = 0u; part < PARTS; ++part )
			{
				parts[part] = LmBf16ToFloat(LmLinearLdg(query_qk + qk_base + part * LM_WARP_LANES + lane));
				squares[part] = parts[part] * parts[part];
			}
			value_s[lane] = LmBf16ToFloat(LmLinearLdg(value_bf16 + (((uint64_t)row * heads) + head) * VALUE_DIM + column_base + lane));
			const float inv = rsqrtf(LmDeltaRuleWarpTotal<WARPS, PARTS>(squares, lane) + 1e-6f) * q_scale;
			#pragma unroll
			for ( uint32_t part = 0u; part < PARTS; ++part )
				query_s[part * LM_WARP_LANES + lane] = parts[part] * inv;
		}
		__syncthreads();
		if ( warp == 0u )
			coefficient_s[lane] = beta_s * (value_s[lane] - LmDeltaRulePredict4<KEY_DIM>(state_s, key_s, forget_s, lane));
		__syncthreads();
		{
			const float coefficient = coefficient_s[lane];
			#pragma unroll
			for ( uint32_t index = warp * (KEY_DIM / LM_DELTA_COLUMN_WARPS); index < (warp + 1u) * (KEY_DIM / LM_DELTA_COLUMN_WARPS); index += 4u )
			{
				const float4 forget4 = *(const float4 *)&forget_s[index];
				const float4 key4 = *(const float4 *)&key_s[index];
				state_s[index][lane] = __fmaf_rn(forget4.x, state_s[index][lane], __fmul_rn(coefficient, key4.x));
				state_s[index + 1u][lane] = __fmaf_rn(forget4.y, state_s[index + 1u][lane], __fmul_rn(coefficient, key4.y));
				state_s[index + 2u][lane] = __fmaf_rn(forget4.z, state_s[index + 2u][lane], __fmul_rn(coefficient, key4.z));
				state_s[index + 3u][lane] = __fmaf_rn(forget4.w, state_s[index + 3u][lane], __fmul_rn(coefficient, key4.w));
			}
		}
		__syncthreads();
		if ( warp == 0u )
		{
			float total = 0.0f;
			#pragma unroll 4
			for ( uint32_t index = 0u; index < KEY_DIM; index += 4u )
			{
				const float4 query4 = *(const float4 *)&query_s[index];
				total = __fmaf_rn(state_s[index][lane], query4.x, total);
				total = __fmaf_rn(state_s[index + 1u][lane], query4.y, total);
				total = __fmaf_rn(state_s[index + 2u][lane], query4.z, total);
				total = __fmaf_rn(state_s[index + 3u][lane], query4.w, total);
			}
			output_bf16[(((uint64_t)row * heads) + head) * VALUE_DIM + column_base + lane] = LmFloatToBf16(total);
		}
		__syncthreads();
	}
	if ( commit != 0u )
	{
		#pragma unroll
		for ( uint32_t index = threadIdx.x >> 3u; index < KEY_DIM; index += LM_DELTA_COLUMN_THREADS >> 3u )
			LmLinearStoreFloat4Cs(state + (index * VALUE_DIM) + column_base + vec_col, *(const float4 *)&state_s[index][vec_col]);
	}
	LmDependentRelease();
}

template<uint32_t KEY_DIM, uint32_t VALUE_DIM, uint32_t COL_SLICE, uint32_t THREADS = 256u>
__global__ __launch_bounds__(THREADS, 1)
static void QwenGdnScalarChunkKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ query_qk, const uint16_t *__restrict__ key_qk, const uint16_t *__restrict__ value_bf16,
	const uint16_t *__restrict__ ba_bf16, const float *__restrict__ a_log, const float *__restrict__ dt_bias,
	uint16_t *__restrict__ output_bf16, uint32_t qk_heads, uint32_t heads, uint32_t sequences, uint32_t commit,
	const uint32_t *__restrict__ sequence_row_indices)
{
	constexpr uint32_t C = QH_GDN_CHUNK, WARPS = THREADS / LM_WARP_LANES;
	constexpr uint32_t V_GROUPS = COL_SLICE / 16u, K_SPLIT = WARPS / V_GROUPS;
	constexpr uint32_t K_BLOCKS = (KEY_DIM / 8u) / K_SPLIT, C_BLOCKS = C / 8u;
	constexpr uint32_t REDUCE_STRIDE = C_BLOCKS * 4u * 32u;
	static_assert(V_GROUPS * K_SPLIT == WARPS && (KEY_DIM % (8u * K_SPLIT)) == 0u && (VALUE_DIM % COL_SLICE) == 0u && (COL_SLICE % 16u) == 0u,
		"valid qwen scalar-decay GDN chunk-32 geometry");
	static_assert(sizeof(QwenGdnChunkShared<KEY_DIM, COL_SLICE, K_SPLIT>) <= 98304u, "GDN chunk-32 shared memory fits on sm_121a");
	extern __shared__ __align__(16) uint8_t qh_gdn_chunk_smem[];
	QwenGdnChunkShared<KEY_DIM, COL_SLICE, K_SPLIT> &sh = *(QwenGdnChunkShared<KEY_DIM, COL_SLICE, K_SPLIT> *)qh_gdn_chunk_smem;
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	const uint32_t warp_v = warp / K_SPLIT, warp_k = warp % K_SPLIT;
	const uint32_t gid = lane >> 2u, tg = lane & 3u;
	const uint32_t k_perm = (gid >> 1u) + ((gid & 1u) << 2u);
	const uint32_t sequence = blockIdx.x, head = blockIdx.y, col_base = blockIdx.z * COL_SLICE;
	const uint32_t k_base = warp_k * K_BLOCKS * 8u;
	const uint32_t v0_local = warp_v * 16u + gid, v1_local = v0_local + 8u;
	const uint32_t v0 = col_base + v0_local, v1 = col_base + v1_local;
	if ( sequence >= sequences || head >= heads || col_base >= VALUE_DIM )
	{
		LmDependentWait();
		LmDependentRelease();
		return;
	}
	const float a_exp = expf(__ldg(a_log + head));
	const float dt_b = __ldg(dt_bias + head);
	LmDependentWait();
	const uint32_t repeat = heads / qk_heads;
	const uint32_t qk_head = head / repeat;
	uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	uint32_t count = end > begin ? end - begin : 0u;
	if ( count != 0u && positions != 0 )
	{
		const uint32_t row0 = sequence_row_indices != 0 ? sequence_row_indices[begin] : begin;
		const uint32_t pos0 = positions[row0];
		const uint32_t rem = pos0 & (C - 1u);
		const uint32_t skip = rem != 0u ? (C - rem < count ? C - rem : count) : 0u;
		begin += skip;
		count = (count - skip) & ~(C - 1u);
	}
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	float *state = (float *)(state_pool + ((uint64_t)state_index[sequence] * slot_bytes) + ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(float)));
	if ( count != 0u )
		QwenGdnChunkFetch<KEY_DIM, VALUE_DIM, COL_SLICE, K_SPLIT, THREADS>(sh, query_qk, key_qk, value_bf16, sequence_row_indices,
			begin, count < C ? count : C, qk_head, qk_heads, head, heads, col_base);
	float s[K_BLOCKS][4];
	#pragma unroll
	for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
	{
		const uint32_t k0 = k_base + kb * 8u + tg, k1 = k0 + 4u;
		s[kb][0] = __ldcs(state + k0 * VALUE_DIM + v0);
		s[kb][1] = __ldcs(state + k1 * VALUE_DIM + v0);
		s[kb][2] = __ldcs(state + k0 * VALUE_DIM + v1);
		s[kb][3] = __ldcs(state + k1 * VALUE_DIM + v1);
	}
	const float q_scale = rsqrtf((float)KEY_DIM);
	for ( uint32_t first = 0u; first < count; first += C )
	{
		const uint32_t rows = count - first < C ? count - first : C;
		LmStreamWait<0u>();
		__syncthreads();
		{
			const uint32_t c0 = lane * 4u;
			#pragma unroll
			for ( uint32_t row_w = warp; row_w < C; row_w += WARPS )
			{
				if ( row_w < rows )
				{
					const uint2 rk = *(const uint2 *)&sh.raw_key[row_w][c0];
					const uint2 rq = *(const uint2 *)&sh.raw_query[row_w][c0];
					const float ka = __uint_as_float(rk.x << 16u), kb_f = __uint_as_float(rk.x & 0xffff0000u);
					const float kc = __uint_as_float(rk.y << 16u), kd = __uint_as_float(rk.y & 0xffff0000u);
					const float qa = __uint_as_float(rq.x << 16u), qb = __uint_as_float(rq.x & 0xffff0000u);
					const float qc = __uint_as_float(rq.y << 16u), qd = __uint_as_float(rq.y & 0xffff0000u);
					const float kinv = rsqrtf(LmLinearWarpSum4(ka * ka, kb_f * kb_f, kc * kc, kd * kd) + 1e-6f);
					const float qinv = rsqrtf(LmLinearWarpSum4(qa * qa, qb * qb, qc * qc, qd * qd) + 1e-6f) * q_scale;
					*(float4 *)&sh.k_hat[row_w][c0] = make_float4(ka * kinv, kb_f * kinv, kc * kinv, kd * kinv);
					*(float4 *)&sh.q_hat[row_w][c0] = make_float4(qa * qinv, qb * qinv, qc * qinv, qd * qinv);
					if constexpr ( COL_SLICE == KEY_DIM )
					{
						const uint2 rv = *(const uint2 *)&sh.raw_value[row_w][c0];
						*(float4 *)&sh.val[row_w][c0] = make_float4(__uint_as_float(rv.x << 16u), __uint_as_float(rv.x & 0xffff0000u),
							__uint_as_float(rv.y << 16u), __uint_as_float(rv.y & 0xffff0000u));
					}
				}
				else
				{
					const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
					*(float4 *)&sh.k_hat[row_w][c0] = z4;
					*(float4 *)&sh.q_hat[row_w][c0] = z4;
				}
			}
		}
		if constexpr ( COL_SLICE != KEY_DIM )
			for ( uint32_t idx = threadIdx.x; idx < rows * COL_SLICE; idx += THREADS )
				sh.val[idx / COL_SLICE][idx % COL_SLICE] = LmBf16ToFloat(sh.raw_value[idx / COL_SLICE][idx % COL_SLICE]);
		if ( warp == 0u )
		{
			float log_g = 0.0f, bv = 0.0f;
			if ( lane < rows )
			{
				const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + lane] : begin + first + lane;
				const uint64_t ba_base = (uint64_t)row * 2u * heads;
				const float shifted = LmBf16ToFloat(__ldg(ba_bf16 + ba_base + heads + head)) + dt_b;
				log_g = -a_exp * SparkMathSoftplus(shifted);
				bv = SparkMathSigmoidFast(LmBf16ToFloat(__ldg(ba_bf16 + ba_base + head)));
			}
			float g_cum = log_g;
			#pragma unroll
			for ( uint32_t off = 1u; off < 32u; off <<= 1u )
			{
				const float prev = __shfl_up_sync(0xffffffffu, g_cum, off);
				if ( lane >= off )
					g_cum += prev;
			}
			const float tail_g = __shfl_sync(0xffffffffu, g_cum, rows - 1u);
			sh.g_cum[lane] = g_cum;
			sh.exp_g[lane] = lane < rows ? expf(g_cum) : 0.0f;
			sh.exp_tail_diff[lane] = lane < rows ? expf(tail_g - g_cum) : 0.0f;
			sh.beta[lane] = bv;
			if ( lane == 0u )
				sh.tail_decay = expf(tail_g);
		}
		__syncthreads();
		if ( first + C < count )
			QwenGdnChunkFetch<KEY_DIM, VALUE_DIM, COL_SLICE, K_SPLIT, THREADS>(sh, query_qk, key_qk, value_bf16, sequence_row_indices,
				begin + first + C, count - first - C < C ? count - first - C : C, qk_head, qk_heads, head, heads, col_base);
		QwenGdnChunkGramAndSolve<KEY_DIM, COL_SLICE, K_SPLIT>(sh, rows, warp, lane, gid, tg);
		float p[C_BLOCKS][4], o[C_BLOCKS][4], u[C_BLOCKS][4];
		#pragma unroll
		for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
		{
			p[ct][0] = p[ct][1] = p[ct][2] = p[ct][3] = 0.0f;
			o[ct][0] = o[ct][1] = o[ct][2] = o[ct][3] = 0.0f;
			u[ct][0] = u[ct][1] = u[ct][2] = u[ct][3] = 0.0f;
		}
		#pragma unroll
		for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
		{
			uint32_t ah[4], al[4];
			LmDeltaSplitTf32(s[kb][0], ah[0], al[0]);
			LmDeltaSplitTf32(s[kb][2], ah[1], al[1]);
			LmDeltaSplitTf32(s[kb][1], ah[2], al[2]);
			LmDeltaSplitTf32(s[kb][3], ah[3], al[3]);
			const uint32_t k0 = k_base + kb * 8u + tg, k1 = k0 + 4u;
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t c_row = ct * 8u + gid;
				uint32_t bkh[2], bkl[2], bqh[2], bql[2];
				LmDeltaSplitTf32(sh.k_hat[c_row][k0], bkh[0], bkl[0]);
				LmDeltaSplitTf32(sh.k_hat[c_row][k1], bkh[1], bkl[1]);
				LmDeltaSplitTf32(sh.q_hat[c_row][k0], bqh[0], bql[0]);
				LmDeltaSplitTf32(sh.q_hat[c_row][k1], bqh[1], bql[1]);
				LmDeltaMmaFp32(p[ct], ah, al, bkh, bkl);
				LmDeltaMmaFp32(o[ct], ah, al, bqh, bql);
			}
		}
		if constexpr ( K_SPLIT > 1u )
		{
			if ( warp_k > 0u )
			{
				float *rp = sh.reduce + (warp_v * (K_SPLIT - 1u) + (warp_k - 1u)) * REDUCE_STRIDE;
				#pragma unroll
				for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
						rp[(ct * 4u + e) * 32u + lane] = p[ct][e];
			}
			__syncthreads();
		}
		if ( warp_k == 0u )
		{
			if constexpr ( K_SPLIT > 1u )
			{
				for ( uint32_t ks = 0u; ks < K_SPLIT - 1u; ++ks )
				{
					const float *rp = sh.reduce + (warp_v * (K_SPLIT - 1u) + ks) * REDUCE_STRIDE;
					#pragma unroll
					for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
						#pragma unroll
						for ( uint32_t e = 0u; e < 4u; ++e )
							p[ct][e] += rp[(ct * 4u + e) * 32u + lane];
				}
			}
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t c_ev = ct * 8u + 2u * tg, c_od = c_ev + 1u;
				const float e_ev = sh.exp_g[c_ev], e_od = sh.exp_g[c_od];
				p[ct][0] = c_ev < rows ? sh.val[c_ev][v0_local] - p[ct][0] * e_ev : 0.0f;
				p[ct][1] = c_od < rows ? sh.val[c_od][v0_local] - p[ct][1] * e_od : 0.0f;
				p[ct][2] = c_ev < rows ? sh.val[c_ev][v1_local] - p[ct][2] * e_ev : 0.0f;
				p[ct][3] = c_od < rows ? sh.val[c_od][v1_local] - p[ct][3] * e_od : 0.0f;
			}
			#pragma unroll
			for ( uint32_t c_in = 0u; c_in < C_BLOCKS; ++c_in )
			{
				uint32_t awh[4], awl[4];
				LmDeltaSplitTf32(p[c_in][0], awh[0], awl[0]);
				LmDeltaSplitTf32(p[c_in][2], awh[1], awl[1]);
				LmDeltaSplitTf32(p[c_in][1], awh[2], awl[2]);
				LmDeltaSplitTf32(p[c_in][3], awh[3], awl[3]);
				const uint32_t j_ev = c_in * 8u + 2u * tg, j_od = j_ev + 1u;
				#pragma unroll
				for ( uint32_t c_out = c_in; c_out < C_BLOCKS; ++c_out )
				{
					const uint32_t r_out = c_out * 8u + k_perm;
					uint32_t bxh[2], bxl[2];
					LmDeltaSplitTf32(sh.tri_x[r_out][j_ev], bxh[0], bxl[0]);
					LmDeltaSplitTf32(sh.tri_x[r_out][j_od], bxh[1], bxl[1]);
					LmDeltaMmaFp32(u[c_out], awh, awl, bxh, bxl);
				}
			}
			if constexpr ( K_SPLIT > 1u )
			{
				float *ru = sh.reduce + (warp_v * (K_SPLIT - 1u)) * REDUCE_STRIDE;
				#pragma unroll
				for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
						ru[(ct * 4u + e) * 32u + lane] = u[ct][e];
			}
		}
		if constexpr ( K_SPLIT > 1u )
		{
			__syncthreads();
			if ( warp_k > 0u )
			{
				const float *ru = sh.reduce + (warp_v * (K_SPLIT - 1u)) * REDUCE_STRIDE;
				#pragma unroll
				for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
						u[ct][e] = ru[(ct * 4u + e) * 32u + lane];
			}
			__syncthreads();
			if ( warp_k > 0u )
			{
				float *ro = sh.reduce + (warp_v * (K_SPLIT - 1u) + (warp_k - 1u)) * REDUCE_STRIDE;
				#pragma unroll
				for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
						ro[(ct * 4u + e) * 32u + lane] = o[ct][e];
			}
			__syncthreads();
		}
		if ( warp_k == 0u )
		{
			if constexpr ( K_SPLIT > 1u )
			{
				for ( uint32_t ks = 0u; ks < K_SPLIT - 1u; ++ks )
				{
					const float *ro = sh.reduce + (warp_v * (K_SPLIT - 1u) + ks) * REDUCE_STRIDE;
					#pragma unroll
					for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
						#pragma unroll
						for ( uint32_t e = 0u; e < 4u; ++e )
							o[ct][e] += ro[(ct * 4u + e) * 32u + lane];
				}
			}
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t c_ev = ct * 8u + 2u * tg, c_od = c_ev + 1u;
				const float e_ev = sh.exp_g[c_ev], e_od = sh.exp_g[c_od];
				o[ct][0] *= e_ev;
				o[ct][1] *= e_od;
				o[ct][2] *= e_ev;
				o[ct][3] *= e_od;
			}
			#pragma unroll
			for ( uint32_t c_in = 0u; c_in < C_BLOCKS; ++c_in )
			{
				uint32_t auh[4], aul[4];
				LmDeltaSplitTf32(u[c_in][0], auh[0], aul[0]);
				LmDeltaSplitTf32(u[c_in][2], auh[1], aul[1]);
				LmDeltaSplitTf32(u[c_in][1], auh[2], aul[2]);
				LmDeltaSplitTf32(u[c_in][3], auh[3], aul[3]);
				const uint32_t j0 = c_in * 8u + tg, j1 = j0 + 4u;
				#pragma unroll
				for ( uint32_t ct = c_in; ct < C_BLOCKS; ++ct )
				{
					const uint32_t r_out = ct * 8u + gid;
					uint32_t bmh[2], bml[2];
					LmDeltaSplitTf32(sh.tri_m[r_out][j0], bmh[0], bml[0]);
					LmDeltaSplitTf32(sh.tri_m[r_out][j1], bmh[1], bml[1]);
					LmDeltaMmaFp32(o[ct], auh, aul, bmh, bml);
				}
			}
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t c_ev = ct * 8u + 2u * tg, c_od = c_ev + 1u;
				if ( c_ev < rows )
				{
					const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + c_ev] : begin + first + c_ev;
					const uint64_t out_row = (((uint64_t)row * heads) + head) * VALUE_DIM;
					output_bf16[out_row + v0] = LmFloatToBf16(o[ct][0]);
					output_bf16[out_row + v1] = LmFloatToBf16(o[ct][2]);
				}
				if ( c_od < rows )
				{
					const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + c_od] : begin + first + c_od;
					const uint64_t out_row = (((uint64_t)row * heads) + head) * VALUE_DIM;
					output_bf16[out_row + v0] = LmFloatToBf16(o[ct][1]);
					output_bf16[out_row + v1] = LmFloatToBf16(o[ct][3]);
				}
			}
		}
		if ( first + C >= count && commit == 0u )
			break;
		const float tail_sc = sh.tail_decay;
		#pragma unroll
		for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
		{
			s[kb][0] *= tail_sc;
			s[kb][1] *= tail_sc;
			s[kb][2] *= tail_sc;
			s[kb][3] *= tail_sc;
		}
		#pragma unroll
		for ( uint32_t c_in = 0u; c_in < C_BLOCKS; ++c_in )
		{
			const uint32_t j0 = c_in * 8u + tg, j1 = j0 + 4u;
			const float e0 = sh.exp_tail_diff[j0], e1 = sh.exp_tail_diff[j1];
			uint32_t auh[4], aul[4];
			LmDeltaSplitTf32(u[c_in][0] * e0, auh[0], aul[0]);
			LmDeltaSplitTf32(u[c_in][2] * e0, auh[1], aul[1]);
			LmDeltaSplitTf32(u[c_in][1] * e1, auh[2], aul[2]);
			LmDeltaSplitTf32(u[c_in][3] * e1, auh[3], aul[3]);
			#pragma unroll
			for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
			{
				const uint32_t k_col = k_base + kb * 8u + k_perm;
				uint32_t bch[2], bcl[2];
				LmDeltaSplitTf32(sh.k_hat[j0][k_col], bch[0], bcl[0]);
				LmDeltaSplitTf32(sh.k_hat[j1][k_col], bch[1], bcl[1]);
				LmDeltaMmaFp32(s[kb], auh, aul, bch, bcl);
			}
		}
	}
	LmStreamWait<0u>();
	if ( commit != 0u )
	{
		#pragma unroll
		for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
		{
			const uint32_t k0 = k_base + kb * 8u + tg, k1 = k0 + 4u;
			LmLinearStoreCs(state + k0 * VALUE_DIM + v0, s[kb][0]);
			LmLinearStoreCs(state + k1 * VALUE_DIM + v0, s[kb][1]);
			LmLinearStoreCs(state + k0 * VALUE_DIM + v1, s[kb][2]);
			LmLinearStoreCs(state + k1 * VALUE_DIM + v1, s[kb][3]);
		}
	}
	LmDependentRelease();
}

template<uint32_t KEY_DIM, uint32_t VALUE_DIM, uint32_t COL_SLICE>
static inline SparkGpuError QwenGdnScalarChunkDispatch(uint8_t *state_pool, uint32_t slot_bytes, const uint32_t *state_index,
	const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint32_t *positions, const uint16_t *query_qk, const uint16_t *key_qk,
	const uint16_t *value_bf16, const uint16_t *ba_bf16, const float *a_log, const float *dt_bias, uint16_t *output_bf16,
	uint32_t qk_heads, uint32_t heads, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, SparkGpuStream stream)
{
	constexpr uint32_t K_SPLIT = 8u / (COL_SLICE / 16u);
	constexpr uint32_t shared = (uint32_t)sizeof(QwenGdnChunkShared<KEY_DIM, COL_SLICE, K_SPLIT>);
	static bool granted = false;
	if ( !granted )
	{
		const SparkGpuError err = SparkGpuFuncSetAttribute((const void *)QwenGdnScalarChunkKernel<KEY_DIM, VALUE_DIM, COL_SLICE, 256u>,
			SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
		if ( err != SPARK_GPU_SUCCESS )
			return err;
		granted = true;
	}
	if ( positions != 0 )
		(void)LM_LAUNCH_DEPENDENT((QwenGdnExactEdgeKernel<KEY_DIM, VALUE_DIM, true>), dim3(sequences, heads, VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions, query_qk, key_qk,
			value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices);
	(void)LM_LAUNCH_DEPENDENT((QwenGdnScalarChunkKernel<KEY_DIM, VALUE_DIM, COL_SLICE, 256u>), dim3(sequences, heads, VALUE_DIM / COL_SLICE),
		256u, shared, stream, state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions, query_qk, key_qk, value_bf16,
		ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices);
	if ( positions != 0 )
		(void)LM_LAUNCH_DEPENDENT((QwenGdnExactEdgeKernel<KEY_DIM, VALUE_DIM, false>), dim3(sequences, heads, VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions, query_qk, key_qk,
			value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices);
	return SparkGpuPeekAtLastError();
}

template<uint32_t KEY_DIM, uint32_t VALUE_DIM>
static inline SparkGpuError QwenGdnScalarChunkLaunch(uint8_t *state_pool, uint32_t slot_bytes, const uint32_t *state_index,
	const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint32_t *positions, const uint16_t *query_qk, const uint16_t *key_qk,
	const uint16_t *value_bf16, const uint16_t *ba_bf16, const float *a_log, const float *dt_bias, uint16_t *output_bf16,
	uint32_t qk_heads, uint32_t heads, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, SparkGpuStream stream)
{
	if ( sequences == 0u || heads == 0u )
		return SPARK_GPU_ERROR_INVALID_VALUE;
	if ( heads <= 8u )
		return QwenGdnScalarChunkDispatch<KEY_DIM, VALUE_DIM, 16u>(state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions,
			query_qk, key_qk, value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices, stream);
	if ( heads <= 12u )
		return QwenGdnScalarChunkDispatch<KEY_DIM, VALUE_DIM, 32u>(state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions,
			query_qk, key_qk, value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices, stream);
	if ( heads <= 24u )
		return QwenGdnScalarChunkDispatch<KEY_DIM, VALUE_DIM, 64u>(state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions,
			query_qk, key_qk, value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices, stream);
	return QwenGdnScalarChunkDispatch<KEY_DIM, VALUE_DIM, 128u>(state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions,
		query_qk, key_qk, value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices, stream);
}

template<uint32_t KEY_DIM>
__global__ __launch_bounds__(QH_THREADS, 4)
static void QwenGdnGateOnlyKernel(const uint16_t *__restrict__ ba, const float *__restrict__ a_log,
	const float *__restrict__ dt_bias, float *__restrict__ retention, float *__restrict__ write_gate,
	uint32_t heads, uint32_t rows)
{
	constexpr uint32_t WARPS = QH_THREADS / LM_WARP_LANES;
	constexpr uint32_t KEY_V = KEY_DIM >> 2u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( row >= rows || head >= heads )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t ba_base = (uint64_t)row * 2u * heads;
	float factor = 0.0f;
	if ( lane == 0u )
	{
		const float shifted = LmBf16ToFloat(ba[ba_base + heads + head]) + dt_bias[head];
		factor = SparkMathGatedDeltaDecay(a_log[head], shifted);
		write_gate[(uint64_t)row * heads + head] = SparkMathSigmoidFast(LmBf16ToFloat(ba[ba_base + head]));
	}
	factor = __shfl_sync(0xffffffffu, factor, 0);
	const float4 f4 = make_float4(factor, factor, factor, factor);
	float4 *ret4 = (float4 *)(retention + ((uint64_t)row * heads + head) * KEY_DIM);
	#pragma unroll
	for ( uint32_t v = lane; v < KEY_V; v += LM_WARP_LANES )
		ret4[v] = f4;
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t KEY_DIM>
__global__ __launch_bounds__(THREADS, 2)
static void QwenGdnSplitExpandGateKernel(const uint16_t *__restrict__ qkv, const uint16_t *__restrict__ ba,
	const float *__restrict__ a_log, const float *__restrict__ dt_bias, uint16_t *__restrict__ query_heads,
	uint16_t *__restrict__ key_heads, uint16_t *__restrict__ value, float *__restrict__ retention, float *__restrict__ write_gate,
	uint32_t gdn_qk, uint32_t gdn_v, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES;
	constexpr uint32_t KEY_V = KEY_DIM >> 2u;
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
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
	const uint64_t ba_base = (uint64_t)row * 2u * heads;
	for ( uint32_t h = warp; h < heads; h += WARPS )
	{
		float factor = 0.0f;
		if ( lane == 0u )
		{
			const float shifted = LmBf16ToFloat(ba[ba_base + heads + h]) + dt_bias[h];
			factor = SparkMathGatedDeltaDecay(a_log[h], shifted);
			write_gate[(uint64_t)row * heads + h] = SparkMathSigmoidFast(LmBf16ToFloat(ba[ba_base + h]));
		}
		factor = __shfl_sync(0xffffffffu, factor, 0);
		const float4 f4 = make_float4(factor, factor, factor, factor);
		float4 *ret4 = (float4 *)(retention + ((uint64_t)row * heads + h) * KEY_DIM);
		#pragma unroll
		for ( uint32_t v = lane; v < KEY_V; v += LM_WARP_LANES )
			ret4[v] = f4;
	}
	LmDependentRelease();
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

static __device__ __forceinline__ void QwenCpAsync16(void *dst, const void *src, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(src), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void QwenCpAsync16Ca(void *dst, const void *src, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
	asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(src), "r"(bytes) : "memory");
}

typedef struct QwenBf16Part
{
	const uint16_t *weight;
	uint16_t *output;
	uint32_t out_dim;
	uint32_t first_block;
} QwenBf16Part;

typedef struct QwenBf16MultiArgs
{
	QwenBf16Part part[3];
	uint32_t count;
	uint32_t total_blocks;
	const uint16_t *activation;
	uint32_t rows;
	uint32_t input_dim;
} QwenBf16MultiArgs;

template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG, uint32_t MIN_BLK>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32u, MIN_BLK)
static void QwenExactBf16MultiKernel(const __grid_constant__ QwenBf16MultiArgs args)
{
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t TILE_M = WARPS_M * RPG, TILE_N = WARPS_N * NPG;
	constexpr uint32_t TOTAL_W = TILE_N * 32u, TOTAL_A = TILE_M * 32u;
	__shared__ __align__(16) uint4 s_w[2][TILE_N][32];
	__shared__ __align__(16) uint4 s_a[2][TILE_M][32];
	uint32_t p = 0u;
	if ( args.count > 1u )
	{
		if ( blockIdx.x >= args.part[1].first_block )
			p = 1u;
		if ( args.count > 2u && blockIdx.x >= args.part[2].first_block )
			p = 2u;
	}
	const QwenBf16Part &pt = args.part[p];
	const uint32_t n_blk = (blockIdx.x - pt.first_block) * TILE_N;
	const uint32_t m_blk = blockIdx.y * TILE_M;
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t wm = warp / WARPS_N, wn = warp % WARPS_N;
	const uint32_t warp_r0 = wm * RPG, warp_n0 = wn * NPG;
	const uint32_t global_r0 = m_blk + warp_r0, global_n0 = n_blk + warp_n0;
	const uint32_t row_cnt = global_r0 < args.rows ? (args.rows - global_r0 < RPG ? args.rows - global_r0 : RPG) : 0u;
	const uint32_t chunks = args.input_dim >> 3u;
	const uint4 *w4 = (const uint4 *)pt.weight, *a4 = (const uint4 *)args.activation;
	auto stage = [&](uint32_t st, uint32_t k_base) {
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_W; idx += THREADS )
		{
			const uint32_t tn = idx >> 5u, wc = idx & 31u, neuron = n_blk + tn, c = k_base + wc;
			const bool ok = neuron < pt.out_dim && c < chunks;
			QwenCpAsync16(&s_w[st][tn][wc], ok ? (w4 + (uint64_t)neuron * chunks + c) : w4, ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_A; idx += THREADS )
		{
			const uint32_t tm = idx >> 5u, ac = idx & 31u, row = m_blk + tm, c = k_base + ac;
			const bool ok = row < args.rows && c < chunks;
			QwenCpAsync16Ca(&s_a[st][tm][ac], ok ? (a4 + (uint64_t)row * chunks + c) : a4, ok ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	LmDependentWait();
	stage(0u, 0u);
	float acc[NPG][RPG];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
		#pragma unroll
		for ( uint32_t r = 0u; r < RPG; ++r )
			acc[n][r] = 0.0f;
	for ( uint32_t k_base = 0u; k_base < chunks; k_base += 32u )
	{
		const uint32_t cur = (k_base >> 5u) & 1u;
		if ( k_base + 32u < chunks )
		{
			stage(cur ^ 1u, k_base + 32u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( row_cnt != 0u && k_base + sub < chunks )
		{
			float wf[NPG][8];
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				LmSkinnyUnpackBf16x8(s_w[cur][warp_n0 + n][sub], wf[n]);
			if ( row_cnt == RPG )
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + r][sub], af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf[n], af), acc[n][r]);
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + (r < row_cnt ? r : 0u)][sub], af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						acc[n][r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wf[n], af), acc[n][r]);
				}
			}
		}
		if ( k_base + 32u < chunks )
			__syncthreads();
	}
	if ( row_cnt != 0u )
	{
		#pragma unroll
		for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
					acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], step, 32u);
		if ( sub == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RPG; ++r )
				if ( r < row_cnt )
				{
					const uint64_t row_off = (uint64_t)(global_r0 + r) * pt.out_dim + global_n0;
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						if ( global_n0 + n < pt.out_dim )
							pt.output[row_off + n] = LmFloatToBf16(acc[n][r]);
				}
	}
	LmDependentRelease();
}

template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG, uint32_t MIN_BLK>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32u, MIN_BLK)
static void QwenExactBf16GateUpSwigluKernel(const uint16_t *__restrict__ w_gate_up, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t input_dim, uint32_t ffn_rows)
{
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t TILE_M = WARPS_M * RPG, TILE_N = WARPS_N * NPG;
	constexpr uint32_t TOTAL_W = 2u * TILE_N * 32u, TOTAL_A = TILE_M * 32u;
	__shared__ __align__(16) uint4 s_w[2][2u * TILE_N][32];
	__shared__ __align__(16) uint4 s_a[2][TILE_M][32];
	const uint32_t n_blk = blockIdx.x * TILE_N, m_blk = blockIdx.y * TILE_M;
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t wm = warp / WARPS_N, wn = warp % WARPS_N;
	const uint32_t warp_r0 = wm * RPG, warp_n0 = wn * NPG;
	const uint32_t global_r0 = m_blk + warp_r0, global_n0 = n_blk + warp_n0;
	const uint32_t row_cnt = global_r0 < rows ? (rows - global_r0 < RPG ? rows - global_r0 : RPG) : 0u;
	const uint32_t chunks = input_dim >> 3u;
	const uint4 *w4 = (const uint4 *)w_gate_up, *a4 = (const uint4 *)activation;
	auto stage = [&](uint32_t st, uint32_t k_base) {
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_W; idx += THREADS )
		{
			const uint32_t tw = idx >> 5u, wc = idx & 31u;
			const uint32_t is_up = tw >= TILE_N ? 1u : 0u;
			const uint32_t tn = is_up != 0u ? tw - TILE_N : tw;
			const uint32_t col = n_blk + tn, c = k_base + wc;
			const bool ok = col < ffn_rows && c < chunks;
			const uint64_t row_w = is_up != 0u ? (uint64_t)ffn_rows + col : (uint64_t)col;
			QwenCpAsync16(&s_w[st][tw][wc], ok ? (w4 + row_w * chunks + c) : w4, ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_A; idx += THREADS )
		{
			const uint32_t tm = idx >> 5u, ac = idx & 31u, row = m_blk + tm, c = k_base + ac;
			const bool ok = row < rows && c < chunks;
			QwenCpAsync16Ca(&s_a[st][tm][ac], ok ? (a4 + (uint64_t)row * chunks + c) : a4, ok ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	LmDependentWait();
	stage(0u, 0u);
	float acc_g[NPG][RPG], acc_u[NPG][RPG];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
		#pragma unroll
		for ( uint32_t r = 0u; r < RPG; ++r )
			acc_g[n][r] = acc_u[n][r] = 0.0f;
	for ( uint32_t k_base = 0u; k_base < chunks; k_base += 32u )
	{
		const uint32_t cur = (k_base >> 5u) & 1u;
		if ( k_base + 32u < chunks )
		{
			stage(cur ^ 1u, k_base + 32u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( row_cnt != 0u && k_base + sub < chunks )
		{
			float wg[NPG][8], wu[NPG][8];
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
			{
				LmSkinnyUnpackBf16x8(s_w[cur][warp_n0 + n][sub], wg[n]);
				LmSkinnyUnpackBf16x8(s_w[cur][TILE_N + warp_n0 + n][sub], wu[n]);
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RPG; ++r )
			{
				float af[8];
				LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + (r < row_cnt ? r : 0u)][sub], af);
				#pragma unroll
				for ( uint32_t n = 0u; n < NPG; ++n )
				{
					acc_g[n][r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wg[n], af), acc_g[n][r]);
					acc_u[n][r] = fmaf(1.0f, LmSkinnyBf16DotUnpacked(wu[n], af), acc_u[n][r]);
				}
			}
		}
		if ( k_base + 32u < chunks )
			__syncthreads();
	}
	if ( row_cnt != 0u )
	{
		#pragma unroll
		for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; ++r )
				{
					acc_g[n][r] += __shfl_xor_sync(0xffffffffu, acc_g[n][r], step, 32u);
					acc_u[n][r] += __shfl_xor_sync(0xffffffffu, acc_u[n][r], step, 32u);
				}
		if ( sub == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RPG; ++r )
				if ( r < row_cnt )
				{
					const uint64_t row_off = (uint64_t)(global_r0 + r) * ffn_rows + global_n0;
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; ++n )
						if ( global_n0 + n < ffn_rows )
						{
							const float gv = LmBf16ToFloat(LmFloatToBf16(acc_g[n][r]));
							const float uv = LmBf16ToFloat(LmFloatToBf16(acc_u[n][r]));
							output[row_off + n] = LmFloatToBf16(SparkMathSiluFast(gv) * uv);
						}
				}
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(THREADS, 2)
static void QwenBf16DenseMmaKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t K, uint32_t N)
{
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	static_assert(NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS && STAGES >= 2u, "valid dense BF16 MMA geometry");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t r0 = threadIdx.x / ROW_CHUNKS, c0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	if ( row0 >= rows || col0 >= N )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *a_src[A_PER_T], *b_src[B_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t r = r0 + q * ROW_STEP;
		const uint32_t valid = row0 + r < rows ? 1u : 0u;
		a_src[q] = valid != 0u ? (activation + (uint64_t)(row0 + r) * K + c0 * 8u) : activation;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t gn = col0 + r0 + q * ROW_STEP;
		const uint32_t valid = gn < N ? 1u : 0u;
		b_src[q] = valid != 0u ? (weight + (uint64_t)gn * K + c0 * 8u) : weight;
		b_mask |= valid << q;
	}
	auto issue_k = [&](uint32_t st_off) {
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
			a_src[q] += TILE_K;
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, b_src[q], ((b_mask >> q) & 1u) << 4u);
			b_src[q] += TILE_K;
		}
	};
	auto advance = [](uint32_t &off) {
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
	float acc[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
	auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s) {
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
			uint32_t tmp[4];
			LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
			b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
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
						LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
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
						LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
			}
			advance(sc_off);
		}
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t r0_out = row0 + warp_m * WARP_M + i * 16u + quad_row, r1_out = r0_out + 8u;
			const uint32_t col = col0 + warp_n * WARP_N + j * 8u + quad * 2u;
			if ( col + 1u < N )
			{
				if ( r0_out < rows )
					*(uint32_t *)(output + (uint64_t)r0_out * N + col) = LmPackBf16Pair(acc[i][j][0], acc[i][j][1]);
				if ( r1_out < rows )
					*(uint32_t *)(output + (uint64_t)r1_out * N + col) = LmPackBf16Pair(acc[i][j][2], acc[i][j][3]);
			}
			else if ( col < N )
			{
				if ( r0_out < rows )
					output[(uint64_t)r0_out * N + col] = LmFloatToBf16(acc[i][j][0]);
				if ( r1_out < rows )
					output[(uint64_t)r1_out * N + col] = LmFloatToBf16(acc[i][j][2]);
			}
		}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(THREADS, 2)
static void QwenBf16FusedGateUpSwigluMmaKernel(const uint16_t *__restrict__ w_gate_up, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t K, uint32_t ffn_rows)
{
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t B_ROWS = 2u * TILE_N;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = B_ROWS * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (B_ROWS * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	static_assert(NF == 2u && WARPS_M * WARPS_N == WARPS && STAGES >= 2u, "valid fused gate_up SwiGLU MMA geometry");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t r0 = threadIdx.x / ROW_CHUNKS, c0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	if ( row0 >= rows || col0 >= ffn_rows )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *a_src[A_PER_T], *b_src[B_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t r = r0 + q * ROW_STEP;
		const uint32_t valid = row0 + r < rows ? 1u : 0u;
		a_src[q] = valid != 0u ? (activation + (uint64_t)(row0 + r) * K + c0 * 8u) : activation;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t br = r0 + q * ROW_STEP;
		const uint32_t is_up = br >= TILE_N ? 1u : 0u;
		const uint32_t gn = col0 + (is_up != 0u ? br - TILE_N : br);
		const uint32_t valid = gn < ffn_rows ? 1u : 0u;
		const uint64_t wrow = is_up != 0u ? (uint64_t)ffn_rows + gn : (uint64_t)gn;
		b_src[q] = valid != 0u ? (w_gate_up + wrow * K + c0 * 8u) : w_gate_up;
		b_mask |= valid << q;
	}
	auto issue_k = [&](uint32_t st_off) {
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
			a_src[q] += TILE_K;
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, b_src[q], ((b_mask >> q) & 1u) << 4u);
			b_src[q] += TILE_K;
		}
	};
	auto advance = [](uint32_t &off) {
		if constexpr ( STAGES == 2u )
			off ^= STAGE_BYTES;
		else
		{
			off += STAGE_BYTES;
			if ( off == STAGES * STAGE_BYTES )
				off = 0u;
		}
	};
	uint32_t a_reg[2][MF][4], bg_reg[2][2][2], bu_reg[2][2][2];
	float acc_g[MF][2][4], acc_u[MF][2][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			acc_g[i][j][0] = acc_g[i][j][1] = acc_g[i][j][2] = acc_g[i][j][3] = acc_u[i][j][0] = acc_u[i][j][1] = acc_u[i][j][2] = acc_u[i][j][3] = 0.0f;
	auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s) {
		#pragma unroll
		for ( uint32_t ii = 0u; ii < MF; ++ii )
		{
			const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
			const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
			LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
		}
		const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
		const uint32_t rg = warp_n * WARP_N + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
		const uint32_t ru = TILE_N + rg;
		uint32_t tg[4], tu[4];
		LmStreamLdMatrix4(tg, stage_s + A_BYTES + rg * ROW_BYTES + LmStreamSwizzle8(rg, bc) * 16u);
		LmStreamLdMatrix4(tu, stage_s + A_BYTES + ru * ROW_BYTES + LmStreamSwizzle8(ru, bc) * 16u);
		bg_reg[buf][0][0] = tg[0]; bg_reg[buf][0][1] = tg[1];
		bg_reg[buf][1][0] = tg[2]; bg_reg[buf][1][1] = tg[3];
		bu_reg[buf][0][0] = tu[0]; bu_reg[buf][0][1] = tu[1];
		bu_reg[buf][1][0] = tu[2]; bu_reg[buf][1][1] = tu[3];
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
					for ( uint32_t j = 0u; j < 2u; ++j )
					{
						LmMmaBf16(acc_g[i][j], a_reg[cur][i], bg_reg[cur][j]);
						LmMmaBf16(acc_u[i][j], a_reg[cur][i], bu_reg[cur][j]);
					}
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
					for ( uint32_t j = 0u; j < 2u; ++j )
					{
						LmMmaBf16(acc_g[i][j], a_reg[cur][i], bg_reg[cur][j]);
						LmMmaBf16(acc_u[i][j], a_reg[cur][i], bu_reg[cur][j]);
					}
			}
			advance(sc_off);
		}
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
		{
			const uint32_t r0_out = row0 + warp_m * WARP_M + i * 16u + quad_row, r1_out = r0_out + 8u;
			const uint32_t col = col0 + warp_n * WARP_N + j * 8u + quad * 2u;
			if ( col + 1u < ffn_rows )
			{
				if ( r0_out < rows )
				{
					const float o0 = SparkMathSiluFast(LmBf16ToFloat(LmFloatToBf16(acc_g[i][j][0]))) * LmBf16ToFloat(LmFloatToBf16(acc_u[i][j][0]));
					const float o1 = SparkMathSiluFast(LmBf16ToFloat(LmFloatToBf16(acc_g[i][j][1]))) * LmBf16ToFloat(LmFloatToBf16(acc_u[i][j][1]));
					*(uint32_t *)(output + (uint64_t)r0_out * ffn_rows + col) = LmPackBf16Pair(o0, o1);
				}
				if ( r1_out < rows )
				{
					const float o2 = SparkMathSiluFast(LmBf16ToFloat(LmFloatToBf16(acc_g[i][j][2]))) * LmBf16ToFloat(LmFloatToBf16(acc_u[i][j][2]));
					const float o3 = SparkMathSiluFast(LmBf16ToFloat(LmFloatToBf16(acc_g[i][j][3]))) * LmBf16ToFloat(LmFloatToBf16(acc_u[i][j][3]));
					*(uint32_t *)(output + (uint64_t)r1_out * ffn_rows + col) = LmPackBf16Pair(o2, o3);
				}
			}
		}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void QwenBf16SmallNMmaKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t K, uint32_t N)
{
	constexpr uint32_t TILE_M = 16u, TILE_N = 16u, TILE_K = 64u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS_K = 4u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, SLICE_BYTES = A_BYTES + B_BYTES;
	__shared__ __align__(128) union {
		uint8_t ab[WARPS_K * SLICE_BYTES];
		float red[WARPS_K][TILE_M][TILE_N];
	} smem;
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t lr0 = lane / ROW_CHUNKS, lc0 = lane % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t my_base = (uint32_t)__cvta_generic_to_shared(smem.ab + warp * SLICE_BYTES);
	LmDependentWait();
	if ( row0 >= rows || col0 >= N )
	{
		LmDependentRelease();
		return;
	}
	float acc[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
	for ( uint32_t kt = warp; kt < k_tiles; kt += WARPS_K )
	{
		const uint32_t k_off = kt * TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			const bool va = row0 + r < rows, vb = col0 + r < N;
			LmStreamCopy(my_base + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				va ? activation + (uint64_t)(row0 + r) * K + k_off + lc0 * 8u : activation, va ? 16u : 0u);
			LmStreamCopy(my_base + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				vb ? weight + (uint64_t)(col0 + r) * K + k_off + lc0 * 8u : weight, vb ? 16u : 0u);
		}
		LmStreamCommit();
		LmStreamWait<0u>();
		__syncwarp();
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t af[4], bf[4];
			const uint32_t ar = lane & 15u, ac = ks * 2u + ((lane >> 4u) & 1u);
			const uint32_t br = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), bc = ks * 2u + ((lane >> 3u) & 1u);
			LmStreamLdMatrix4(af, my_base + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
			LmStreamLdMatrix4(bf, my_base + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
			const uint32_t b0[2] = { bf[0], bf[1] }, b1[2] = { bf[2], bf[3] };
			LmMmaBf16(acc[0], af, b0);
			LmMmaBf16(acc[1], af, b1);
		}
		__syncwarp();
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t j = 0u; j < 2u; ++j )
	{
		const uint32_t c = j * 8u + quad * 2u;
		smem.red[warp][quad_row][c] = acc[j][0];
		smem.red[warp][quad_row][c + 1u] = acc[j][1];
		smem.red[warp][quad_row + 8u][c] = acc[j][2];
		smem.red[warp][quad_row + 8u][c + 1u] = acc[j][3];
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t idx = threadIdx.x; idx < TILE_M * TILE_N; idx += 128u )
	{
		const uint32_t r = idx >> 4u, c = idx & 15u;
		if ( row0 + r < rows && col0 + c < N )
		{
			const float v = smem.red[0][r][c] + smem.red[1][r][c] + smem.red[2][r][c] + smem.red[3][r][c];
			output[(uint64_t)(row0 + r) * N + (col0 + c)] = LmFloatToBf16(v);
		}
	}
	LmDependentRelease();
}

template<uint32_t TILE_M>
static __host__ __device__ constexpr uint32_t QwenBf16SmallNMmaWideSharedBytes(void)
{
	return 4u * 2u * (TILE_M + 16u) * 128u;
}

template<uint32_t TILE_M>
__global__ __launch_bounds__(128u, 2)
static void QwenBf16SmallNMmaWideKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t K, uint32_t N)
{
	constexpr uint32_t TILE_N = 16u, TILE_K = 64u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS_K = 4u, MF = TILE_M / 16u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, SLICE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t STAGES = 2u;
	static_assert(TILE_M == 32u, "wide small-N MMA uses TILE_M=32 double-buffered");
	extern __shared__ __align__(128) uint8_t qh_smalln_smem[];
	float (*const red)[TILE_M][TILE_N] = (float (*)[TILE_M][TILE_N])qh_smalln_smem;
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t lr0 = lane / ROW_CHUNKS, lc0 = lane % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t warp_base = (uint32_t)__cvta_generic_to_shared(qh_smalln_smem + warp * STAGES * SLICE_BYTES);
	LmDependentWait();
	if ( row0 >= rows || col0 >= N )
	{
		LmDependentRelease();
		return;
	}
	auto issue_slice = [&](uint32_t dst_base, uint32_t kt)
	{
		const uint32_t k_off = kt * TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < TILE_M / 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			const bool va = row0 + r < rows;
			LmStreamCopy(dst_base + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				va ? activation + (uint64_t)(row0 + r) * K + k_off + lc0 * 8u : activation, va ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			const bool vb = col0 + r < N;
			LmStreamCopy(dst_base + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				vb ? weight + (uint64_t)(col0 + r) * K + k_off + lc0 * 8u : weight, vb ? 16u : 0u);
		}
		LmStreamCommit();
	};
	float acc[MF][2][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
	uint32_t cur = 0u;
	if ( warp < k_tiles )
		issue_slice(warp_base, warp);
	for ( uint32_t kt = warp; kt < k_tiles; kt += WARPS_K )
	{
		const uint32_t nxt_kt = kt + WARPS_K;
		if ( nxt_kt < k_tiles )
		{
			issue_slice(warp_base + (cur ^ 1u) * SLICE_BYTES, nxt_kt);
			LmStreamWait<1u>();
		}
		else
			LmStreamWait<0u>();
		__syncwarp();
		const uint32_t my_base = warp_base + cur * SLICE_BYTES;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t bf[4];
			const uint32_t br = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), bc = ks * 2u + ((lane >> 3u) & 1u);
			LmStreamLdMatrix4(bf, my_base + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
			const uint32_t b0[2] = { bf[0], bf[1] }, b1[2] = { bf[2], bf[3] };
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				uint32_t af[4];
				const uint32_t ar = i * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(af, my_base + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
				LmMmaBf16(acc[i][0], af, b0);
				LmMmaBf16(acc[i][1], af, b1);
			}
		}
		__syncwarp();
		cur ^= 1u;
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
		{
			const uint32_t c = j * 8u + quad * 2u;
			red[warp][i * 16u + quad_row][c] = acc[i][j][0];
			red[warp][i * 16u + quad_row][c + 1u] = acc[i][j][1];
			red[warp][i * 16u + quad_row + 8u][c] = acc[i][j][2];
			red[warp][i * 16u + quad_row + 8u][c + 1u] = acc[i][j][3];
		}
	__syncthreads();
	#pragma unroll
	for ( uint32_t idx = threadIdx.x; idx < TILE_M * TILE_N; idx += 128u )
	{
		const uint32_t r = idx >> 4u, c = idx & 15u;
		if ( row0 + r < rows && col0 + c < N )
		{
			const float v = red[0][r][c] + red[1][r][c] + red[2][r][c] + red[3][r][c];
			output[(uint64_t)(row0 + r) * N + (col0 + c)] = LmFloatToBf16(v);
		}
	}
	LmDependentRelease();
}

template<uint32_t TILE_M>
static __host__ __device__ constexpr uint32_t QwenSharedGateUpSwigluWideSharedBytes(void)
{
	return 4u * 2u * (TILE_M + 32u) * 128u;
}

template<uint32_t TILE_M>
__global__ __launch_bounds__(128u, TILE_M <= 16u ? 2 : 1)
static void QwenSharedGateUpSwigluWideKernel(const uint16_t *__restrict__ w_gate_up, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t rows, uint32_t K, uint32_t shared_rows)
{
	constexpr uint32_t TILE_N = 16u, TILE_K = 64u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS_K = 4u, MF = TILE_M / 16u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, SLICE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t STAGES = 2u;
	static_assert(TILE_M == 16u || TILE_M == 32u, "shared gate_up SwiGLU supports TILE_M 16 or 32");
	extern __shared__ __align__(128) uint8_t qh_shgu_smem[];
	float (*const red)[WARPS_K][TILE_M][TILE_N] = (float (*)[WARPS_K][TILE_M][TILE_N])qh_shgu_smem;
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t lr0 = lane / ROW_CHUNKS, lc0 = lane % ROW_CHUNKS;
	const uint32_t k_tiles = K / TILE_K;
	const uint32_t warp_base = (uint32_t)__cvta_generic_to_shared(qh_shgu_smem + warp * STAGES * SLICE_BYTES);
	LmDependentWait();
	if ( row0 >= rows || col0 >= shared_rows )
	{
		LmDependentRelease();
		return;
	}
	auto issue_slice = [&](uint32_t dst_base, uint32_t kt)
	{
		const uint32_t k_off = kt * TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < TILE_M / 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			const bool va = row0 + r < rows;
			LmStreamCopy(dst_base + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				va ? activation + (uint64_t)(row0 + r) * K + k_off + lc0 * 8u : activation, va ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			const bool vb = col0 + r < shared_rows;
			const uint32_t sw = r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u;
			LmStreamCopy(dst_base + A_BYTES + sw,
				vb ? w_gate_up + (uint64_t)(col0 + r) * K + k_off + lc0 * 8u : w_gate_up, vb ? 16u : 0u);
			LmStreamCopy(dst_base + A_BYTES + B_BYTES + sw,
				vb ? w_gate_up + (uint64_t)(shared_rows + col0 + r) * K + k_off + lc0 * 8u : w_gate_up, vb ? 16u : 0u);
		}
		LmStreamCommit();
	};
	float acc_g[MF][2][4], acc_u[MF][2][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			acc_g[i][j][0] = acc_g[i][j][1] = acc_g[i][j][2] = acc_g[i][j][3] =
				acc_u[i][j][0] = acc_u[i][j][1] = acc_u[i][j][2] = acc_u[i][j][3] = 0.0f;
	uint32_t cur = 0u;
	if ( warp < k_tiles )
		issue_slice(warp_base, warp);
	for ( uint32_t kt = warp; kt < k_tiles; kt += WARPS_K )
	{
		const uint32_t nxt_kt = kt + WARPS_K;
		if ( nxt_kt < k_tiles )
		{
			issue_slice(warp_base + (cur ^ 1u) * SLICE_BYTES, nxt_kt);
			LmStreamWait<1u>();
		}
		else
			LmStreamWait<0u>();
		__syncwarp();
		const uint32_t my_base = warp_base + cur * SLICE_BYTES;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t bg[4], bu[4];
			const uint32_t br = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), bc = ks * 2u + ((lane >> 3u) & 1u);
			const uint32_t boff = br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u;
			LmStreamLdMatrix4(bg, my_base + A_BYTES + boff);
			LmStreamLdMatrix4(bu, my_base + A_BYTES + B_BYTES + boff);
			const uint32_t g0[2] = { bg[0], bg[1] }, g1[2] = { bg[2], bg[3] };
			const uint32_t u0[2] = { bu[0], bu[1] }, u1[2] = { bu[2], bu[3] };
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				uint32_t af[4];
				const uint32_t ar = i * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(af, my_base + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
				LmMmaBf16(acc_g[i][0], af, g0);
				LmMmaBf16(acc_g[i][1], af, g1);
				LmMmaBf16(acc_u[i][0], af, u0);
				LmMmaBf16(acc_u[i][1], af, u1);
			}
		}
		__syncwarp();
		cur ^= 1u;
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
		{
			const uint32_t c = j * 8u + quad * 2u;
			red[0][warp][i * 16u + quad_row][c] = acc_g[i][j][0];
			red[0][warp][i * 16u + quad_row][c + 1u] = acc_g[i][j][1];
			red[0][warp][i * 16u + quad_row + 8u][c] = acc_g[i][j][2];
			red[0][warp][i * 16u + quad_row + 8u][c + 1u] = acc_g[i][j][3];
			red[1][warp][i * 16u + quad_row][c] = acc_u[i][j][0];
			red[1][warp][i * 16u + quad_row][c + 1u] = acc_u[i][j][1];
			red[1][warp][i * 16u + quad_row + 8u][c] = acc_u[i][j][2];
			red[1][warp][i * 16u + quad_row + 8u][c + 1u] = acc_u[i][j][3];
		}
	__syncthreads();
	#pragma unroll
	for ( uint32_t idx = threadIdx.x; idx < TILE_M * TILE_N; idx += 128u )
	{
		const uint32_t r = idx >> 4u, c = idx & 15u;
		if ( row0 + r < rows && col0 + c < shared_rows )
		{
			const float gv = LmBf16ToFloat(LmFloatToBf16(red[0][0][r][c] + red[0][1][r][c] + red[0][2][r][c] + red[0][3][r][c]));
			const float uv = LmBf16ToFloat(LmFloatToBf16(red[1][0][r][c] + red[1][1][r][c] + red[1][2][r][c] + red[1][3][r][c]));
			output[(uint64_t)(row0 + r) * shared_rows + (col0 + c)] = LmFloatToBf16(SparkMathSiluFast(gv) * uv);
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
	constexpr uint32_t B_ROW_BYTES = TILE_K / 2u;
	constexpr uint32_t SC_BYTES = SCALE_TILES * 8u;
	constexpr uint32_t SC_VECS = SC_BYTES / 16u;
	constexpr uint32_t SC_SHIFT = SCALE_TILES == 4u ? 2u : 1u;
	__shared__ __align__(128) uint16_t s_a[STAGES][TILE_M][TILE_K];
	__shared__ __align__(16) uint8_t s_b[STAGES][W_ROWS_TOTAL][B_ROW_BYTES];
	__shared__ __align__(16) uint8_t s_sc[2][W_ROWS_TOTAL][SC_BYTES];
	const uint32_t s_a_base = (uint32_t)__cvta_generic_to_shared(s_a);
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
				{
					const uint32_t swz = (a_vec & 8u) | LmStreamSwizzle8(lr, a_vec & 7u);
					QwenCpAsync16Ca(&s_a[st][lr][swz * 8u], a_ptrs[i] + (uint64_t)kt * TILE_K, 16u);
				}
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
		const float g1_b = FUSED_GATE_UP ? g3_scaled : g1_scaled;
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
			const uint32_t stage_a = s_a_base + cur * TILE_M * 256u;
			#pragma unroll
			for ( uint32_t kp = 0u; kp < 4u; ++kp )
			{
				const float2 f0 = LmSkinnyDecodeUe4m3Raw(LmSkinnyUe4m3Raw{ (uint8_t)(sc0_all >> (kp * 16u)), (uint8_t)(sc0_all >> (kp * 16u + 8u)) }, g1_scaled);
				const float2 f1 = LmSkinnyDecodeUe4m3Raw(LmSkinnyUe4m3Raw{ (uint8_t)(sc1_all >> (kp * 16u)), (uint8_t)(sc1_all >> (kp * 16u + 8u)) }, g1_b);
				uint32_t s0_pair[2], s1_pair[2];
				asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(s0_pair[0]) : "f"(f0.x));
				asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(s0_pair[1]) : "f"(f0.y));
				asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(s1_pair[0]) : "f"(f1.x));
				asm("cvt.rn.bf16x2.f32 %0, %1, %1;\n" : "=r"(s1_pair[1]) : "f"(f1.y));
				#pragma unroll
				for ( uint32_t sub = 0u; sub < 2u; ++sub )
				{
					const uint32_t ks = kp * 2u + sub;
					const uint32_t s0_2 = s0_pair[sub], s1_2 = s1_pair[sub];
					const uint32_t byte_off = ks * 8u + c;
					const uint32_t b0[2] = { QwenE2m1PairToBf16(s_b[cur][b_row0][byte_off], s0_2), QwenE2m1PairToBf16(s_b[cur][b_row0][byte_off + 4u], s0_2) };
					const uint32_t b1[2] = { QwenE2m1PairToBf16(s_b[cur][b_row1][byte_off], s1_2), QwenE2m1PairToBf16(s_b[cur][b_row1][byte_off + 4u], s1_2) };
					const uint32_t k_half = (ks & 4u) << 1u;
					const uint32_t k_c = (ks & 3u) * 2u + ((lane >> 4u) & 1u);
					#pragma unroll
					for ( uint32_t m = 0u; m < MAX_M_TILES; ++m )
					{
						if ( m < m_tiles_live )
						{
							const uint32_t ar = m * 16u + (lane & 15u);
							const uint32_t ac = k_half | LmStreamSwizzle8(ar, k_c);
							uint32_t af[4];
							LmStreamLdMatrix4(af, stage_a + ar * 256u + ac * 16u);
							LmMmaBf16(acc0[m], af, b0);
							LmMmaBf16(acc1[m], af, b1);
						}
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
	constexpr uint32_t CHUNKS = K >> 5u, LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u), PER_BLK = QH_THREADS / LANES, NPG = 4u;
	const uint32_t group = blockIdx.y, n0 = (blockIdx.x * PER_BLK + (threadIdx.x / LANES)) * NPG, sub = threadIdx.x & (LANES - 1u);
	LmDependentWait();
	const uint32_t row_begin = __ldg(group_row_offset + group), row_end = __ldg(group_row_offset + group + 1u);
	if ( row_begin >= row_end || n0 >= N )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t off0 = ((uint64_t)group * N + n0) * CHUNKS;
	const uint4 *w_base = (const uint4 *)w2 + off0;
	const uint8_t *s_base = w2_scale_blob + (uint64_t)GROUPS * sizeof(float) + off0 * 2u;
	const float glob = __ldg((const float *)w2_scale_blob + group);
	for ( uint32_t first = row_begin; first < row_end; )
	{
		const uint32_t rem = row_end - first;
		if ( rem == 1u )
		{
			const uint16_t *act = activation + (uint64_t)first * K;
			float acc[NPG] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				uint4 vw[NPG];
				float2 sc[NPG];
				#pragma unroll
				for ( uint32_t k = 0u; k < NPG; ++k )
				{
					const bool live = (n0 + k) < N;
					vw[k] = live ? __ldcs(w_base + (uint64_t)k * CHUNKS + c) : make_uint4(0u, 0u, 0u, 0u);
					sc[k] = live ? QwenNvfp4LoadSc(s_base + (uint64_t)k * CHUNKS * 2u, c, glob) : make_float2(0.0f, 0.0f);
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
						if ( (n0 + k) < N )
							acc[k] = fmaf(half == 0u ? sc[k].x : sc[k].y, LmSkinnyDotPairs16(QwenNvfp4DecodeHalf(vw[k], half).pair, af), acc[k]);
				}
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				#pragma unroll
				for ( uint32_t k = 0u; k < NPG; ++k )
					if ( (n0 + k) < N )
						acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], step, LANES);
			if ( sub == 0u )
				#pragma unroll
				for ( uint32_t k = 0u; k < NPG; ++k )
					if ( (n0 + k) < N )
						output[(uint64_t)first * N + n0 + k] = LmFloatToBf16(acc[k]);
			first += 1u;
		}
		else
		{
			const uint32_t cnt = rem < 4u ? rem : 4u;
			#pragma unroll 1
			for ( uint32_t k = 0u; k < NPG; ++k )
			{
				if ( n0 + k >= N )
					break;
				const uint4 *wk = w_base + (uint64_t)k * CHUNKS;
				const uint8_t *sk = s_base + (uint64_t)k * CHUNKS * 2u;
				float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const uint4 vw = __ldcs(wk + c);
					const float2 sc = QwenNvfp4LoadSc(sk, c, glob);
					#pragma unroll
					for ( uint32_t half = 0u; half < 2u; ++half )
					{
						const LmSkinnyE2m1HalfDecoded dec = QwenNvfp4DecodeHalf(vw, half);
						const float s = half == 0u ? sc.x : sc.y;
						acc0 = QwenNvfp4HalfDot(dec, activation + (uint64_t)first * K + c * 32u, half, s, acc0);
						if ( cnt > 1u )
							acc1 = QwenNvfp4HalfDot(dec, activation + (uint64_t)(first + 1u) * K + c * 32u, half, s, acc1);
						if ( cnt > 2u )
							acc2 = QwenNvfp4HalfDot(dec, activation + (uint64_t)(first + 2u) * K + c * 32u, half, s, acc2);
						if ( cnt > 3u )
							acc3 = QwenNvfp4HalfDot(dec, activation + (uint64_t)(first + 3u) * K + c * 32u, half, s, acc3);
					}
				}
				#pragma unroll
				for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				{
					acc0 += __shfl_xor_sync(0xffffffffu, acc0, step, LANES);
					acc1 += __shfl_xor_sync(0xffffffffu, acc1, step, LANES);
					acc2 += __shfl_xor_sync(0xffffffffu, acc2, step, LANES);
					acc3 += __shfl_xor_sync(0xffffffffu, acc3, step, LANES);
				}
				if ( sub == 0u )
				{
					output[(uint64_t)first * N + n0 + k] = LmFloatToBf16(acc0);
					if ( cnt > 1u )
						output[(uint64_t)(first + 1u) * N + n0 + k] = LmFloatToBf16(acc1);
					if ( cnt > 2u )
						output[(uint64_t)(first + 2u) * N + n0 + k] = LmFloatToBf16(acc2);
					if ( cnt > 3u )
						output[(uint64_t)(first + 3u) * N + n0 + k] = LmFloatToBf16(acc3);
				}
			}
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

template<uint32_t HIDDEN, uint32_t SHARED_ROWS, uint32_t TOP_K, uint32_t LOCAL_EXPERTS>
__global__ __launch_bounds__(128u, 4)
static void QwenMoeSharedDownFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared_gate_up, const uint16_t *__restrict__ shared_down,
	const uint16_t *__restrict__ shared_logit, uint16_t *__restrict__ output, uint32_t rows)
{
	static_assert((SHARED_ROWS & 7u) == 0u && SHARED_ROWS <= 128u && TOP_K <= LM_MOE_LOCAL_MAX_TOPK, "valid fused shared down finalize");
	constexpr uint32_t CHUNKS = SHARED_ROWS >> 3u;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = 128u / LANES;
	__shared__ __align__(16) uint4 s_act[CHUNKS];
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK], active_count;
	const uint32_t col = blockIdx.x * GROUPS + (threadIdx.x / LANES);
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	if ( threadIdx.x < CHUNKS )
	{
		const uint64_t gu_base = (uint64_t)row * 2u * SHARED_ROWS;
		float g[8], u[8], a[8];
		LmLoadBf16x8AsFloat(((const uint4 *)(shared_gate_up + gu_base))[threadIdx.x], g);
		LmLoadBf16x8AsFloat(((const uint4 *)(shared_gate_up + gu_base + SHARED_ROWS))[threadIdx.x], u);
		#pragma unroll
		for ( uint32_t i = 0u; i < 8u; ++i )
			a[i] = SparkMathSiluFast(g[i]) * u[i];
		s_act[threadIdx.x] = LmPackUint4FloatToBf16(a);
	}
	if ( threadIdx.x == 0u )
	{
		uint32_t count = 0u;
		for ( uint32_t k = 0u; k < TOP_K; ++k )
		{
			const uint32_t route = row * TOP_K + k;
			if ( route_expert[route] < LOCAL_EXPERTS )
				active_rows[count++] = route_packed_row[route];
		}
		active_count = count;
	}
	__syncthreads();
	if ( col < HIDDEN )
	{
		const uint4 *w = (const uint4 *)(shared_down + (uint64_t)col * SHARED_ROWS);
		float acc = 0.0f;
		for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			acc = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w + c), s_act[c], 0.0f), acc);
		#pragma unroll
		for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
			acc += __shfl_xor_sync(0xffffffffu, acc, step, LANES);
		if ( sub == 0u )
		{
			const uint32_t count = active_count;
			float total = 0.0f;
			for ( uint32_t k = 0u; k < count; ++k )
				total += LmBf16ToFloat(packed[(uint64_t)active_rows[k] * HIDDEN + col]);
			const float gate = SparkMathSigmoid(LmBf16ToFloat(shared_logit[row]));
			total += SparkMathRoundBf16(LmBf16ToFloat(LmFloatToBf16(acc)) * gate);
			output[(uint64_t)row * HIDDEN + col] = LmFloatToBf16(total);
		}
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN, uint32_t INTER, uint32_t LOCAL_EXPERTS>
__global__ __launch_bounds__(128u, 4)
static void QwenBf16MoeSkinnyW1W3SingleKernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3,
	const uint16_t *__restrict__ act, const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ packed_expert, uint16_t *__restrict__ out_act)
{
	constexpr uint32_t CHUNKS = HIDDEN >> 3u;
	constexpr uint32_t WARPS = 4u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t n = blockIdx.x * WARPS + warp;
	const uint32_t p = blockIdx.y;
	LmDependentWait();
	if ( p >= group_row_offset[LOCAL_EXPERTS] || n >= INTER )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t e = packed_expert[p];
	const float wt = packed_weight[p];
	const uint64_t row_off = ((uint64_t)e * INTER + n) * HIDDEN;
	const uint4 *pw1 = (const uint4 *)(w1 + row_off);
	const uint4 *pw3 = (const uint4 *)(w3 + row_off);
	const uint4 *pa = (const uint4 *)act;
	float acc1 = 0.0f, acc3 = 0.0f;
	for ( uint32_t c = lane; c < CHUNKS; c += 32u )
	{
		const uint4 va = __ldg(pa + c);
		acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(pw1 + c), va, 0.0f), acc1);
		acc3 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(pw3 + c), va, 0.0f), acc3);
	}
	#pragma unroll
	for ( uint32_t step = 16u; step > 0u; step >>= 1u )
	{
		acc1 += __shfl_xor_sync(0xffffffffu, acc1, step, 32);
		acc3 += __shfl_xor_sync(0xffffffffu, acc3, step, 32);
	}
	if ( lane == 0u )
	{
		const float g = SparkMathRoundBf16(acc1), u = SparkMathRoundBf16(acc3);
		out_act[(uint64_t)p * INTER + n] = LmFloatToBf16(SparkMathGateLimitedGlu(g, u, INFINITY) * wt);
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN, uint32_t INTER, uint32_t LOCAL_EXPERTS>
__global__ __launch_bounds__(128u, 4)
static void QwenBf16MoeSkinnyW1W3BatchedKernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3,
	const uint16_t *__restrict__ act, const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, uint16_t *__restrict__ out_act)
{
	constexpr uint32_t CHUNKS = HIDDEN >> 3u;
	constexpr uint32_t WARPS = 4u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t n = blockIdx.x * WARPS + warp;
	const uint32_t e = blockIdx.y;
	LmDependentWait();
	if ( e >= LOCAL_EXPERTS || n >= INTER )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t begin = group_row_offset[e], end = group_row_offset[e + 1u];
	if ( begin >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t row_off = ((uint64_t)e * INTER + n) * HIDDEN;
	const uint4 *pw1 = (const uint4 *)(w1 + row_off);
	const uint4 *pw3 = (const uint4 *)(w3 + row_off);
	for ( uint32_t p0 = begin; p0 < end; p0 += 4u )
	{
		const uint32_t count = end - p0 < 4u ? end - p0 : 4u;
		if ( count == 1u )
		{
			const uint4 *pa0 = (const uint4 *)(act + (uint64_t)route_source_token[p0] * HIDDEN);
			float acc1 = 0.0f, acc3 = 0.0f;
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				const uint4 va = __ldg(pa0 + c);
				acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(pw1 + c), va, 0.0f), acc1);
				acc3 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(pw3 + c), va, 0.0f), acc3);
			}
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			{
				acc1 += __shfl_xor_sync(0xffffffffu, acc1, step, 32);
				acc3 += __shfl_xor_sync(0xffffffffu, acc3, step, 32);
			}
			if ( lane == 0u )
			{
				const float g = SparkMathRoundBf16(acc1), u = SparkMathRoundBf16(acc3);
				out_act[(uint64_t)p0 * INTER + n] = LmFloatToBf16(SparkMathGateLimitedGlu(g, u, INFINITY) * packed_weight[p0]);
			}
		}
		else if ( count == 2u )
		{
			const uint4 *pa0 = (const uint4 *)(act + (uint64_t)route_source_token[p0] * HIDDEN);
			const uint4 *pa1 = (const uint4 *)(act + (uint64_t)route_source_token[p0 + 1u] * HIDDEN);
			float a10 = 0.0f, a30 = 0.0f, a11 = 0.0f, a31 = 0.0f;
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				const uint4 vw1 = __ldcs(pw1 + c), vw3 = __ldcs(pw3 + c);
				const uint4 va0 = __ldg(pa0 + c), va1 = __ldg(pa1 + c);
				a10 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1, va0, 0.0f), a10);
				a30 = fmaf(1.0f, LmSkinnyBf16Chunk(vw3, va0, 0.0f), a30);
				a11 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1, va1, 0.0f), a11);
				a31 = fmaf(1.0f, LmSkinnyBf16Chunk(vw3, va1, 0.0f), a31);
			}
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			{
				a10 += __shfl_xor_sync(0xffffffffu, a10, step, 32);
				a30 += __shfl_xor_sync(0xffffffffu, a30, step, 32);
				a11 += __shfl_xor_sync(0xffffffffu, a11, step, 32);
				a31 += __shfl_xor_sync(0xffffffffu, a31, step, 32);
			}
			if ( lane == 0u )
			{
				out_act[(uint64_t)p0 * INTER + n] = LmFloatToBf16(SparkMathGateLimitedGlu(SparkMathRoundBf16(a10), SparkMathRoundBf16(a30), INFINITY) * packed_weight[p0]);
				out_act[(uint64_t)(p0 + 1u) * INTER + n] = LmFloatToBf16(SparkMathGateLimitedGlu(SparkMathRoundBf16(a11), SparkMathRoundBf16(a31), INFINITY) * packed_weight[p0 + 1u]);
			}
		}
		else
		{
			const uint4 *pa[4];
			#pragma unroll
			for ( uint32_t r = 0u; r < 4u; ++r )
				pa[r] = (const uint4 *)(act + (uint64_t)(r < count ? route_source_token[p0 + r] : route_source_token[p0]) * HIDDEN);
			float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			float acc3[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				const uint4 vw1 = __ldcs(pw1 + c);
				const uint4 vw3 = __ldcs(pw3 + c);
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
				{
					const uint4 va = __ldg(pa[r] + c);
					acc1[r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw1, va, 0.0f), acc1[r]);
					acc3[r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw3, va, 0.0f), acc3[r]);
				}
			}
			#pragma unroll
			for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
				{
					acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], step, 32);
					acc3[r] += __shfl_xor_sync(0xffffffffu, acc3[r], step, 32);
				}
			if ( lane == 0u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					if ( r < count )
					{
						const float g = SparkMathRoundBf16(acc1[r]), u = SparkMathRoundBf16(acc3[r]);
						out_act[(uint64_t)(p0 + r) * INTER + n] = LmFloatToBf16(SparkMathGateLimitedGlu(g, u, INFINITY) * packed_weight[p0 + r]);
					}
		}
	}
	LmDependentRelease();
}

template<uint32_t INTER, uint32_t HIDDEN, uint32_t LOCAL_EXPERTS>
__global__ __launch_bounds__(128u, 4)
static void QwenBf16MoeSkinnyW2SingleKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ packed_expert, uint16_t *__restrict__ out)
{
	constexpr uint32_t CHUNKS = INTER >> 3u;
	constexpr uint32_t LANES = (CHUNKS >= 32u && (CHUNKS % 32u) == 0u) ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = 128u / LANES;
	const uint32_t n = blockIdx.x * GROUPS + (threadIdx.x / LANES);
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	const uint32_t p = blockIdx.y;
	LmDependentWait();
	if ( p >= group_row_offset[LOCAL_EXPERTS] || n >= HIDDEN )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t e = packed_expert[p];
	const uint4 *pw = (const uint4 *)(w2 + ((uint64_t)e * HIDDEN + n) * INTER);
	const uint4 *pa = (const uint4 *)(expert_act + (uint64_t)p * INTER);
	float acc = 0.0f;
	for ( uint32_t c = sub; c < CHUNKS; c += LANES )
		acc = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(pw + c), __ldg(pa + c), 0.0f), acc);
	#pragma unroll
	for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
		acc += __shfl_xor_sync(0xffffffffu, acc, step, LANES);
	if ( sub == 0u )
		out[(uint64_t)p * HIDDEN + n] = LmFloatToBf16(acc);
	LmDependentRelease();
}

template<uint32_t INTER, uint32_t HIDDEN, uint32_t LOCAL_EXPERTS>
__global__ __launch_bounds__(128u, 4)
static void QwenBf16MoeSkinnyW2BatchedKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ out)
{
	constexpr uint32_t CHUNKS = INTER >> 3u;
	constexpr uint32_t LANES = (CHUNKS >= 32u && (CHUNKS % 32u) == 0u) ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = 128u / LANES;
	const uint32_t n = blockIdx.x * GROUPS + (threadIdx.x / LANES);
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	const uint32_t e = blockIdx.y;
	LmDependentWait();
	if ( e >= LOCAL_EXPERTS || n >= HIDDEN )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t begin = group_row_offset[e], end = group_row_offset[e + 1u];
	if ( begin >= end )
	{
		LmDependentRelease();
		return;
	}
	const uint4 *pw = (const uint4 *)(w2 + ((uint64_t)e * HIDDEN + n) * INTER);
	for ( uint32_t p0 = begin; p0 < end; p0 += 4u )
	{
		const uint32_t count = end - p0 < 4u ? end - p0 : 4u;
		if ( count == 1u )
		{
			const uint4 *pa0 = (const uint4 *)(expert_act + (uint64_t)p0 * INTER);
			float acc = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				acc = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(pw + c), __ldg(pa0 + c), 0.0f), acc);
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				acc += __shfl_xor_sync(0xffffffffu, acc, step, LANES);
			if ( sub == 0u )
				out[(uint64_t)p0 * HIDDEN + n] = LmFloatToBf16(acc);
		}
		else if ( count == 2u )
		{
			const uint4 *pa0 = (const uint4 *)(expert_act + (uint64_t)p0 * INTER);
			const uint4 *pa1 = (const uint4 *)(expert_act + (uint64_t)(p0 + 1u) * INTER);
			float a0 = 0.0f, a1 = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw = __ldcs(pw + c);
				a0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw, __ldg(pa0 + c), 0.0f), a0);
				a1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw, __ldg(pa1 + c), 0.0f), a1);
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
			{
				a0 += __shfl_xor_sync(0xffffffffu, a0, step, LANES);
				a1 += __shfl_xor_sync(0xffffffffu, a1, step, LANES);
			}
			if ( sub == 0u )
			{
				out[(uint64_t)p0 * HIDDEN + n] = LmFloatToBf16(a0);
				out[(uint64_t)(p0 + 1u) * HIDDEN + n] = LmFloatToBf16(a1);
			}
		}
		else
		{
			const uint4 *pa[4];
			#pragma unroll
			for ( uint32_t r = 0u; r < 4u; ++r )
				pa[r] = (const uint4 *)(expert_act + (uint64_t)(r < count ? (p0 + r) : p0) * INTER);
			float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw = __ldcs(pw + c);
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					acc[r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw, __ldg(pa[r] + c), 0.0f), acc[r]);
			}
			#pragma unroll
			for ( uint32_t step = LANES >> 1u; step > 0u; step >>= 1u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], step, LANES);
			if ( sub == 0u )
				#pragma unroll
				for ( uint32_t r = 0u; r < 4u; ++r )
					if ( r < count )
						out[(uint64_t)(p0 + r) * HIDDEN + n] = LmFloatToBf16(acc[r]);
		}
	}
	LmDependentRelease();
}

static __device__ __forceinline__ void QwenBf16TilePrefixScan(const uint32_t *__restrict__ group_row_offset, uint32_t *prefix,
	uint32_t groups, uint32_t tile_m, uint32_t n_tiles)
{
	if ( threadIdx.x < 32u )
	{
		const uint32_t lane = threadIdx.x;
		const uint32_t per_lane = (groups + 31u) >> 5u;
		const uint32_t first = lane * per_lane;
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

static __device__ __forceinline__ uint32_t QwenBf16GroupedTotalTiles(const uint32_t *__restrict__ group_row_offset, uint32_t groups,
	uint32_t tile_m, uint32_t n_tiles)
{
	uint32_t total = 0u;
	for ( uint32_t g = 0u; g < groups; ++g )
		total += (__ldg(group_row_offset + g + 1u) - __ldg(group_row_offset + g) + tile_m - 1u) / tile_m * n_tiles;
	return total;
}

static __device__ __forceinline__ void QwenBf16GroupedTileScan(const uint32_t *__restrict__ group_row_offset, uint32_t groups,
	uint32_t tile, uint32_t tile_m, uint32_t n_tiles, uint32_t &grp, uint32_t &row_base, uint32_t &row_limit, uint32_t &nt)
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

static __device__ __forceinline__ void QwenBf16GroupedTileLookup(const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ prefix,
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

template<uint32_t BYTES, bool HAS_PREFIX>
struct QwenBf16MmaSmem
{
	__align__(128) uint8_t buf[BYTES];
	uint32_t prefix[257u];
};

template<uint32_t BYTES>
struct QwenBf16MmaSmem<BYTES, false>
{
	__align__(128) uint8_t buf[BYTES];
};

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M>
__global__ __launch_bounds__(THREADS, 2)
static void QwenBf16MoeMmaW1W3ActKernel(const uint16_t *__restrict__ w_gate, const uint16_t *__restrict__ w_up, const uint16_t *__restrict__ act,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ source_row_map,
	uint16_t *__restrict__ out, uint32_t groups, uint32_t K, uint32_t N)
{
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u, STAGES = 2u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	constexpr bool USE_PREFIX = (STAGES * STAGE_BYTES <= 48000u);
	static_assert(NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS, "valid fused BF16 MoE gate-up MMA tile");
	__shared__ QwenBf16MmaSmem<STAGES * STAGE_BYTES, USE_PREFIX> smem_s;
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
		QwenBf16TilePrefixScan(group_row_offset, smem_s.prefix, groups, TILE_M, n_tiles);
		total_tiles = smem_s.prefix[groups];
	}
	else
		total_tiles = QwenBf16GroupedTotalTiles(group_row_offset, groups, TILE_M, n_tiles);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		if constexpr ( USE_PREFIX )
			QwenBf16GroupedTileLookup(group_row_offset, smem_s.prefix, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		else
			QwenBf16GroupedTileScan(group_row_offset, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
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
						const uint16_t lo = LmFloatToBf16(SparkMathGateLimitedGlu(SparkMathRoundBf16(tg[i][j][e]), SparkMathRoundBf16(tu[i][j][e]), INFINITY) * wt);
						const uint16_t hi = LmFloatToBf16(SparkMathGateLimitedGlu(SparkMathRoundBf16(tg[i][j][e + 1u]), SparkMathRoundBf16(tu[i][j][e + 1u]), INFINITY) * wt);
						*(uint32_t *)(out + (uint64_t)r * N + c) = (uint32_t)lo | ((uint32_t)hi << 16u);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M, uint32_t STAGES>
__global__ __launch_bounds__(THREADS, 2)
static void QwenBf16MoeMmaW2Kernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ act,
	uint16_t *__restrict__ out, const uint32_t *__restrict__ group_row_offset, uint32_t groups, uint32_t K, uint32_t N)
{
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	constexpr bool USE_PREFIX = (STAGES * STAGE_BYTES <= 48000u);
	static_assert(NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS, "valid grouped BF16 W2 MMA tile");
	__shared__ QwenBf16MmaSmem<STAGES * STAGE_BYTES, USE_PREFIX> smem_s;
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
		QwenBf16TilePrefixScan(group_row_offset, smem_s.prefix, groups, TILE_M, n_tiles);
		total_tiles = smem_s.prefix[groups];
	}
	else
		total_tiles = QwenBf16GroupedTotalTiles(group_row_offset, groups, TILE_M, n_tiles);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t grp, row_base, row_limit, nt;
		if constexpr ( USE_PREFIX )
			QwenBf16GroupedTileLookup(group_row_offset, smem_s.prefix, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		else
			QwenBf16GroupedTileScan(group_row_offset, groups, tile, TILE_M, n_tiles, grp, row_base, row_limit, nt);
		const uint32_t col_base = nt * TILE_N;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u, b_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t packed = row_base + row0 + q * ROW_STEP;
			const uint32_t valid = packed < row_limit ? 1u : 0u;
			a_src[q] = act + (uint64_t)(valid != 0u ? packed : 0u) * K + col0 * 8u;
			a_mask |= valid << q;
		}
		const uint16_t *b_base = weight + (((uint64_t)grp * N) + col_base + row0) * K + col0 * 8u;
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
				const uint32_t valid = (b_mask >> q) & 1u;
				LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, valid != 0u ? b_base + q * b_step : weight, valid << 4u);
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
				LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			}
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
				b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
				b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
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
					if ( r < row_limit && c < N )
						*(uint32_t *)(out + (uint64_t)r * N + c) = (uint32_t)LmFloatToBf16(total[i][j][e]) |
							((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
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

template<uint32_t STREAMS, uint32_t WIDTH, uint32_t LOW_RANK, uint32_t RG, bool HAS_INJECT = true>
__global__ __launch_bounds__((STREAMS * WIDTH >= 2048u ? 256u : (STREAMS * WIDTH >= 1024u ? 128u : 32u)), 2)
static void QwenHcDownGateInjectKernel(const uint16_t *__restrict__ down, const uint16_t *__restrict__ inject_w,
	const uint16_t *__restrict__ hc_normed, uint16_t *__restrict__ hc_low, float *__restrict__ hc_inject, uint32_t rows)
{
	constexpr uint32_t K = STREAMS * WIDTH;
	constexpr uint32_t CHUNKS = K >> 3u;
	constexpr uint32_t TOTAL_N = LOW_RANK + (HAS_INJECT ? STREAMS : 0u);
	constexpr uint32_t SPLIT_WARPS = CHUNKS >= 256u ? 8u : (CHUNKS >= 128u ? 4u : 1u);
	constexpr uint32_t THREADS = SPLIT_WARPS * 32u;
	__shared__ float partial[SPLIT_WARPS][RG];
	const uint32_t row_groups = (rows + RG - 1u) / RG;
	const uint32_t n0 = blockIdx.x / row_groups;
	const uint32_t first_row = (blockIdx.x % row_groups) * RG;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( n0 < TOTAL_N )
	{
		const uint32_t count = rows - first_row < RG ? rows - first_row : RG;
		const uint4 *w0 = (const uint4 *)(n0 < LOW_RANK ? down + (uint64_t)n0 * K : inject_w + (uint64_t)(n0 - LOW_RANK) * K);
		const uint4 *a[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			a[r] = (const uint4 *)(hc_normed + (uint64_t)(first_row + (r < count ? r : 0u)) * K);
		float acc[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			acc[r] = 0.0f;
		for ( uint32_t c = threadIdx.x; c < CHUNKS; c += THREADS )
		{
			const uint4 vw0 = __ldcs(w0 + c);
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[r] = fmaf(1.0f, LmSkinnyBf16Chunk(vw0, __ldg(a[r] + c), 0.0f), acc[r]);
		}
		#pragma unroll
		for ( uint32_t step = 16u; step > 0u; step >>= 1u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], step, 32);
		if constexpr ( SPLIT_WARPS > 1u )
		{
			if ( lane == 0u )
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					partial[warp][r] = acc[r];
			__syncthreads();
			if ( threadIdx.x == 0u )
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					if ( r < count )
					{
						float sum = partial[0][r];
						#pragma unroll
						for ( uint32_t w = 1u; w < SPLIT_WARPS; ++w )
							sum += partial[w][r];
						const uint32_t row = first_row + r;
						const float v0 = LmBf16ToFloat(LmFloatToBf16(sum));
						if ( !HAS_INJECT || n0 < LOW_RANK )
							hc_low[(uint64_t)row * LOW_RANK + n0] = LmFloatToBf16(SparkMathHyperGate(v0, (float)STREAMS));
						else
							hc_inject[(uint64_t)row * STREAMS + (n0 - LOW_RANK)] = SparkMathHyperInjectWeight(v0, (float)STREAMS);
					}
		}
		else if ( lane == 0u )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < count )
				{
					const uint32_t row = first_row + r;
					const float v0 = LmBf16ToFloat(LmFloatToBf16(acc[r]));
					if ( !HAS_INJECT || n0 < LOW_RANK )
						hc_low[(uint64_t)row * LOW_RANK + n0] = LmFloatToBf16(SparkMathHyperGate(v0, (float)STREAMS));
					else
						hc_inject[(uint64_t)row * STREAMS + (n0 - LOW_RANK)] = SparkMathHyperInjectWeight(v0, (float)STREAMS);
				}
	}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH, uint32_t LOW_RANK, bool HAS_INJECT, uint32_t TILE_M>
__global__ __launch_bounds__(128u, TILE_M <= 32u ? 3 : 2)
static void QwenHcDownGateInjectMmaSplitKernel(const uint16_t *__restrict__ down, const uint16_t *__restrict__ inject_w,
	const uint16_t *__restrict__ hc_normed, uint16_t *__restrict__ hc_low, float *__restrict__ hc_inject, uint32_t rows)
{
	constexpr uint32_t K = STREAMS * WIDTH;
	constexpr uint32_t TOTAL_N = LOW_RANK + (HAS_INJECT ? STREAMS : 0u);
	constexpr uint32_t TILE_N = 16u, TILE_K = 64u, WARPS_K = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t MF = TILE_M / 16u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, SLICE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t K_TILES = K / TILE_K;
	static_assert((K % (TILE_K * WARPS_K)) == 0u && (TILE_M % 16u) == 0u && TILE_M <= 64u, "valid HC down split MMA");
	__shared__ __align__(128) union {
		uint8_t ab[WARPS_K * SLICE_BYTES];
		float red[WARPS_K][TILE_M][TILE_N];
	} smem;
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t lr0 = lane / ROW_CHUNKS, lc0 = lane % ROW_CHUNKS;
	const uint32_t my_base = (uint32_t)__cvta_generic_to_shared(smem.ab + warp * SLICE_BYTES);
	LmDependentWait();
	if ( row0 >= rows || col0 >= TOTAL_N )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *b_base[4];
	uint32_t b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < 4u; ++q )
	{
		const uint32_t gn = col0 + lr0 + q * 4u;
		const uint32_t valid = gn < TOTAL_N ? 1u : 0u;
		b_base[q] = valid != 0u ?
			((!HAS_INJECT || gn < LOW_RANK ? down + (uint64_t)gn * K : inject_w + (uint64_t)(gn - LOW_RANK) * K) + lc0 * 8u) : down;
		b_mask |= valid << q;
	}
	float acc[MF][2][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
	for ( uint32_t kt = warp; kt < K_TILES; kt += WARPS_K )
	{
		const uint32_t k_off = kt * TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < TILE_M / 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			const bool va = row0 + r < rows;
			LmStreamCopy(my_base + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				va ? hc_normed + (uint64_t)(row0 + r) * K + k_off + lc0 * 8u : hc_normed, va ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < 4u; ++q )
		{
			const uint32_t r = lr0 + q * 4u;
			LmStreamCopy(my_base + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, lc0) * 16u,
				b_base[q] + k_off, ((b_mask >> q) & 1u) << 4u);
		}
		LmStreamCommit();
		LmStreamWait<0u>();
		__syncwarp();
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t bf[4];
			const uint32_t br = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), bc = ks * 2u + ((lane >> 3u) & 1u);
			LmStreamLdMatrix4(bf, my_base + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
			const uint32_t b0[2] = { bf[0], bf[1] }, b1[2] = { bf[2], bf[3] };
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				uint32_t af[4];
				const uint32_t ar = i * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(af, my_base + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
				LmMmaBf16(acc[i][0], af, b0);
				LmMmaBf16(acc[i][1], af, b1);
			}
		}
		__syncwarp();
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
		{
			const uint32_t c = j * 8u + quad * 2u;
			smem.red[warp][i * 16u + quad_row][c] = acc[i][j][0];
			smem.red[warp][i * 16u + quad_row][c + 1u] = acc[i][j][1];
			smem.red[warp][i * 16u + quad_row + 8u][c] = acc[i][j][2];
			smem.red[warp][i * 16u + quad_row + 8u][c + 1u] = acc[i][j][3];
		}
	__syncthreads();
	#pragma unroll
	for ( uint32_t idx = threadIdx.x; idx < TILE_M * TILE_N; idx += 128u )
	{
		const uint32_t r = idx >> 4u, c = idx & 15u;
		const uint32_t row = row0 + r, col = col0 + c;
		if ( row < rows && col < TOTAL_N )
		{
			const float v = SparkMathRoundBf16(smem.red[0][r][c] + smem.red[1][r][c] + smem.red[2][r][c] + smem.red[3][r][c]);
			if ( !HAS_INJECT || col < LOW_RANK )
				hc_low[(uint64_t)row * LOW_RANK + col] = LmFloatToBf16(SparkMathHyperGate(v, (float)STREAMS));
			else
				hc_inject[(uint64_t)row * STREAMS + (col - LOW_RANK)] = SparkMathHyperInjectWeight(v, (float)STREAMS);
		}
	}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH, uint32_t LOW_RANK, bool HAS_INJECT, uint32_t TILE_M, uint32_t TILE_N, uint32_t THREADS, uint32_t WARPS_M, uint32_t STAGES = 3u>
__global__ __launch_bounds__(THREADS, 2)
static void QwenHcDownGateInjectMmaKernel(const uint16_t *__restrict__ down, const uint16_t *__restrict__ inject_w,
	const uint16_t *__restrict__ hc_normed, uint16_t *__restrict__ hc_low, float *__restrict__ hc_inject, uint32_t rows)
{
	constexpr uint32_t K = STREAMS * WIDTH;
	constexpr uint32_t TOTAL_N = LOW_RANK + (HAS_INJECT ? STREAMS : 0u);
	constexpr uint32_t TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	constexpr uint32_t K_TILES = K / TILE_K;
	static_assert((K % TILE_K) == 0u && NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS && STAGES >= 2u, "HC down MMA geometry must divide 64/16");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t r0 = threadIdx.x / ROW_CHUNKS, c0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	if ( row0 >= rows || col0 >= TOTAL_N )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *a_src[A_PER_T];
	const uint16_t *b_src[B_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t r = r0 + q * ROW_STEP;
		const uint32_t valid = row0 + r < rows ? 1u : 0u;
		a_src[q] = valid != 0u ? (hc_normed + (uint64_t)(row0 + r) * K + c0 * 8u) : hc_normed;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t gn = col0 + r0 + q * ROW_STEP;
		const uint32_t valid = gn < TOTAL_N ? 1u : 0u;
		b_src[q] = valid != 0u ?
			((!HAS_INJECT || gn < LOW_RANK ? down + (uint64_t)gn * K : inject_w + (uint64_t)(gn - LOW_RANK) * K) + c0 * 8u) : down;
		b_mask |= valid << q;
	}
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			const uint32_t valid = (a_mask >> q) & 1u;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, a_src[q], valid << 4u);
			a_src[q] += TILE_K;
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			const uint32_t valid = (b_mask >> q) & 1u;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, b_src[q], valid << 4u);
			b_src[q] += TILE_K;
		}
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
	float acc[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
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
			uint32_t tmp[4];
			LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
			b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
		}
	};
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
	{
		if ( kp < K_TILES )
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
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			const uint32_t stage_s = shared_base + sc_off;
			#pragma unroll
			for ( uint32_t ks = 0u; ks < KSTEPS; ++ks )
			{
				const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
				if ( ks == 0u )
				{
					if ( kp < K_TILES )
					{
						issue_k(sp_off);
						kp++;
						advance(sp_off);
					}
					LmStreamCommit();
				}
				if ( ks + 1u < KSTEPS )
					load_frag(nxt, ks + 1u, stage_s);
				else if ( kt + 1u < K_TILES )
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
						LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
			}
		}
	}
	else
	{
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<0u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + sc_off;
			if ( kp < K_TILES )
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
						LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
			}
			advance(sc_off);
		}
	}
	LmStreamWait<0u>();
	auto store_pair = [&](uint32_t row, uint32_t col, float v0, float v1)
	{
		if ( row >= rows || col >= TOTAL_N )
			return;
		const float r0_val = SparkMathRoundBf16(v0);
		if ( !HAS_INJECT || col < LOW_RANK )
			hc_low[(uint64_t)row * LOW_RANK + col] = LmFloatToBf16(SparkMathHyperGate(r0_val, (float)STREAMS));
		else
			hc_inject[(uint64_t)row * STREAMS + (col - LOW_RANK)] = SparkMathHyperInjectWeight(r0_val, (float)STREAMS);
		if ( col + 1u < TOTAL_N )
		{
			const float r1_val = SparkMathRoundBf16(v1);
			if ( !HAS_INJECT || col + 1u < LOW_RANK )
				hc_low[(uint64_t)row * LOW_RANK + (col + 1u)] = LmFloatToBf16(SparkMathHyperGate(r1_val, (float)STREAMS));
			else
				hc_inject[(uint64_t)row * STREAMS + (col + 1u - LOW_RANK)] = SparkMathHyperInjectWeight(r1_val, (float)STREAMS);
		}
	};
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t r_lo = row0 + warp_m * WARP_M + i * 16u + quad_row, r_hi = r_lo + 8u;
			const uint32_t col = col0 + warp_n * WARP_N + j * 8u + quad * 2u;
			store_pair(r_lo, col, acc[i][j][0], acc[i][j][1]);
			store_pair(r_hi, col, acc[i][j][2], acc[i][j][3]);
		}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH, uint32_t LOW_RANK, uint32_t TILE_M, uint32_t THREADS, uint32_t WARPS_M, uint32_t STAGES = 3u>
__global__ __launch_bounds__(THREADS, 2)
static void QwenHcUpMixMmaKernel(const uint16_t *__restrict__ up, const uint16_t *__restrict__ hc_low,
	const uint16_t *__restrict__ hc_normed, uint16_t *__restrict__ mixed, uint32_t rows)
{
	constexpr uint32_t TILE_N = 64u, TILE_K = 64u, KSTEPS = 4u, ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t WARPS = THREADS / 32u, WARPS_N = WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / THREADS, B_PER_T = (TILE_N * ROW_CHUNKS) / THREADS;
	constexpr uint32_t ROW_STEP = THREADS / ROW_CHUNKS;
	constexpr uint32_t K_TILES = LOW_RANK / TILE_K;
	static_assert((LOW_RANK % TILE_K) == 0u && (WIDTH % TILE_N) == 0u && NF >= 2u && (NF & 1u) == 0u && WARPS_M * WARPS_N == WARPS && STAGES >= 2u, "HC up mix MMA geometry must divide 64");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t m_blocks = (rows + TILE_M - 1u) / TILE_M;
	const uint32_t n_block = blockIdx.x / m_blocks, m_block = blockIdx.x % m_blocks;
	const uint32_t row0 = m_block * TILE_M, col0 = n_block * TILE_N;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t r0 = threadIdx.x / ROW_CHUNKS, c0 = threadIdx.x % ROW_CHUNKS;
	LmDependentWait();
	if ( row0 >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *a_base[A_PER_T];
	uint32_t a_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t r = r0 + q * ROW_STEP;
		const uint32_t valid = row0 + r < rows ? 1u : 0u;
		a_base[q] = valid != 0u ? (hc_low + (uint64_t)(row0 + r) * LOW_RANK + c0 * 8u) : hc_low;
		a_mask |= valid << q;
	}
	const uint64_t b_step = (uint64_t)ROW_STEP * LOW_RANK;
	auto issue_sk = [&](uint32_t st_off, uint32_t s_idx, uint32_t kt_idx)
	{
		const uint32_t stage = shared_base + st_off;
		const uint32_t k_off = kt_idx * TILE_K;
		const uint16_t *b_ptr = up + ((uint64_t)s_idx * WIDTH + col0 + r0) * LOW_RANK + k_off + c0 * 8u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			const uint32_t valid = (a_mask >> q) & 1u;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, valid != 0u ? a_base[q] + k_off : hc_low, valid << 4u);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t r = r0 + q * ROW_STEP;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c0) * 16u, b_ptr + q * b_step, 16u);
		}
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
			LmStreamLdMatrix4(a_reg[buf][ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
		}
		#pragma unroll
		for ( uint32_t jj = 0u; jj < NF; jj += 2u )
		{
			const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
			const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
			uint32_t tmp[4];
			LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
			b_reg[buf][jj][0] = tmp[0]; b_reg[buf][jj][1] = tmp[1];
			b_reg[buf][jj + 1u][0] = tmp[2]; b_reg[buf][jj + 1u][1] = tmp[3];
		}
	};
	uint32_t ps = 0u, pk = 0u, sp_off = 0u, sc_off = 0u;
	#pragma unroll
	for ( uint32_t st_i = 0u; st_i + 1u < STAGES; ++st_i )
	{
		if ( ps < STREAMS )
		{
			issue_sk(sp_off, ps, pk);
			pk++;
			if ( pk == K_TILES )
			{
				pk = 0u;
				ps++;
			}
		}
		LmStreamCommit();
		advance(sp_off);
	}
	if constexpr ( STAGES >= 3u )
	{
		LmStreamWait<STAGES - 2u>();
		__syncthreads();
		load_frag(0u, 0u, shared_base + sc_off);
	}
	for ( uint32_t s = 0u; s < STREAMS; ++s )
	{
		float acc[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			if constexpr ( STAGES >= 3u )
			{
				const uint32_t stage_s = shared_base + sc_off;
				const bool has_next = (kt + 1u < K_TILES) || (s + 1u < STREAMS);
				#pragma unroll
				for ( uint32_t ks = 0u; ks < KSTEPS; ++ks )
				{
					const uint32_t cur = ks & 1u, nxt = cur ^ 1u;
					if ( ks == 0u )
					{
						if ( ps < STREAMS )
						{
							issue_sk(sp_off, ps, pk);
							pk++;
							if ( pk == K_TILES )
							{
								pk = 0u;
								ps++;
							}
							advance(sp_off);
						}
						LmStreamCommit();
					}
					if ( ks + 1u < KSTEPS )
						load_frag(nxt, ks + 1u, stage_s);
					else if ( has_next )
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
							LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
				}
			}
			else
			{
				LmStreamWait<0u>();
				__syncthreads();
				const uint32_t stage_s = shared_base + sc_off;
				if ( ps < STREAMS )
				{
					issue_sk(sp_off, ps, pk);
					pk++;
					if ( pk == K_TILES )
					{
						pk = 0u;
						ps++;
					}
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
							LmMmaBf16(acc[i][j], a_reg[cur][i], b_reg[cur][j]);
				}
				advance(sc_off);
			}
		}
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
		{
			const uint32_t r_lo = row0 + warp_m * WARP_M + i * 16u + quad_row, r_hi = r_lo + 8u;
			const bool ok_lo = r_lo < rows, ok_hi = r_hi < rows;
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t col = col0 + warp_n * WARP_N + j * 8u + quad * 2u;
				if ( ok_lo )
				{
					const uint32_t nr = *(const uint32_t *)(hc_normed + ((uint64_t)r_lo * STREAMS + s) * WIDTH + col);
					const float n0 = __uint_as_float(nr << 16u), n1 = __uint_as_float(nr & 0xffff0000u);
					total[i][j][0] += SparkMathHyperMixTerm(SparkMathHyperMixGate(SparkMathRoundBf16(acc[i][j][0])), n0);
					total[i][j][1] += SparkMathHyperMixTerm(SparkMathHyperMixGate(SparkMathRoundBf16(acc[i][j][1])), n1);
				}
				if ( ok_hi )
				{
					const uint32_t nr = *(const uint32_t *)(hc_normed + ((uint64_t)r_hi * STREAMS + s) * WIDTH + col);
					const float n0 = __uint_as_float(nr << 16u), n1 = __uint_as_float(nr & 0xffff0000u);
					total[i][j][2] += SparkMathHyperMixTerm(SparkMathHyperMixGate(SparkMathRoundBf16(acc[i][j][2])), n0);
					total[i][j][3] += SparkMathHyperMixTerm(SparkMathHyperMixGate(SparkMathRoundBf16(acc[i][j][3])), n1);
				}
			}
		}
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
	{
		const uint32_t r_lo = row0 + warp_m * WARP_M + i * 16u + quad_row, r_hi = r_lo + 8u;
		const bool ok_lo = r_lo < rows, ok_hi = r_hi < rows;
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t col = col0 + warp_n * WARP_N + j * 8u + quad * 2u;
			if ( ok_lo )
				*(uint32_t *)(mixed + (uint64_t)r_lo * WIDTH + col) = LmPackBf16Pair(total[i][j][0] / (float)STREAMS, total[i][j][1] / (float)STREAMS);
			if ( ok_hi )
				*(uint32_t *)(mixed + (uint64_t)r_hi * WIDTH + col) = LmPackBf16Pair(total[i][j][2] / (float)STREAMS, total[i][j][3] / (float)STREAMS);
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
__global__ __launch_bounds__(128u, 4)
static void QwenIndexQkExactSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ index_q_raw, uint16_t *__restrict__ index_key, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t rows)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t K = G::kHidden, Q_DIM = G::kIndexHeads * G::kIndexDim, K_DIM = G::kIndexDim, N_TOTAL = Q_DIM + K_DIM;
	constexpr uint32_t CHUNKS = K >> 3u, RG = 8u;
	__shared__ uint32_t s_rows[RG], s_cnt;
	const uint32_t row_base = blockIdx.y * RG, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t cnt = row_base < rows ? (rows - row_base < RG ? rows - row_base : RG) : 0u;
	const uint32_t col0 = (blockIdx.x * 4u + warp) * 2u;
	const bool is_k = (blockIdx.x * 8u) >= Q_DIM;
	LmDependentWait();
	if ( warp == 0u )
	{
		uint32_t live = 0u;
		if ( lane < cnt )
		{
			if ( is_k )
				live = 1u;
			else
			{
				const uint32_t r = row_base + lane;
				uint32_t bound = context_length[sequence_of_row[r]];
				if ( positions[r] + 1u < bound )
					bound = positions[r] + 1u;
				live = (bound / G::kIndexRatio) > S::kIndexBlocks ? 1u : 0u;
			}
		}
		const uint32_t mask = __ballot_sync(0xffffffffu, live != 0u);
		if ( lane == 0u )
		{
			uint32_t n = 0u, m = mask;
			while ( m != 0u )
			{
				const uint32_t bit = (uint32_t)(__ffs((int)m) - 1);
				s_rows[n++] = row_base + bit;
				m &= m - 1u;
			}
			s_cnt = n;
		}
	}
	__syncthreads();
	const uint32_t act_cnt = s_cnt;
	if ( act_cnt != 0u && col0 < N_TOTAL )
	{
		const uint4 *w0 = (const uint4 *)(weight + (uint64_t)col0 * K);
		const uint4 *w1 = w0 + CHUNKS;
		const bool reuse = rows > RG;
		const uint16_t *act_ptr[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			act_ptr[r] = activation + (uint64_t)(r < act_cnt ? s_rows[r] : 0u) * K;
		float a0[RG], a1[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			a0[r] = a1[r] = 0.0f;
		for ( uint32_t c = lane; c < CHUNKS; c += 32u )
		{
			const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
			const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < act_cnt )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg((const uint4 *)act_ptr[r] + c), af);
					a0[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw0, af, 0.0f), a0[r]);
					a1[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw1, af, 0.0f), a1[r]);
				}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			if ( r < act_cnt )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
				{
					a0[r] += __shfl_xor_sync(0xffffffffu, a0[r], step);
					a1[r] += __shfl_xor_sync(0xffffffffu, a1[r], step);
				}
				if ( lane == 0u )
				{
					uint16_t *dst = is_k ? (index_key + (uint64_t)s_rows[r] * K_DIM + (col0 - Q_DIM)) :
						(index_q_raw + (uint64_t)s_rows[r] * Q_DIM + col0);
					*(uint32_t *)dst = LmPackBf16Pair(a0[r], a1[r]);
				}
			}
	}
	LmDependentRelease();
}

template<class G, uint32_t TILE_M, uint32_t WARPS_M>
__global__ __launch_bounds__(256u, 2)
static void QwenIndexQMmaSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t K = G::kHidden, N = G::kIndexHeads * G::kIndexDim, TILE_N = 64u, TILE_K = 64u, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS_N = 8u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u, ROW_STEP = 256u / ROW_CHUNKS;
	__shared__ __align__(128) uint8_t smem[2u * STAGE_BYTES];
	__shared__ uint32_t s_any;
	const uint32_t row_base = blockIdx.y * TILE_M, row_limit = row_base + TILE_M < rows ? row_base + TILE_M : rows;
	LmDependentWait();
	if ( threadIdx.x == 0u )
		s_any = 0u;
	__syncthreads();
	if ( threadIdx.x < TILE_M )
	{
		const uint32_t r = row_base + threadIdx.x;
		uint32_t sparse = 0u;
		if ( r < row_limit )
		{
			uint32_t bound = context_length[sequence_of_row[r]];
			if ( positions[r] + 1u < bound )
				bound = positions[r] + 1u;
			sparse = (bound / G::kIndexRatio) > S::kIndexBlocks ? 1u : 0u;
		}
		if ( sparse != 0u )
			s_any = 1u;
	}
	__syncthreads();
	if ( s_any != 0u )
	{
		const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
		const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
		const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
		const uint32_t col_base = blockIdx.x * TILE_N;
		const uint32_t row0 = threadIdx.x / ROW_CHUNKS, col0 = threadIdx.x % ROW_CHUNKS;
		const uint16_t *a_src[A_PER_T];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			const uint32_t valid = chunk < A_CHUNKS && row_base + r < row_limit ? 1u : 0u;
			a_src[q] = activation + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint16_t *b_src = weight + (uint64_t)(col_base + row0) * K + col0 * 8u;
		const uint64_t b_step = (uint64_t)ROW_STEP * K;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_T; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * 256u;
				if ( chunk < A_CHUNKS )
				{
					const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
					LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
					a_src[q] += TILE_K;
				}
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_T; ++q )
			{
				const uint32_t r = row0 + q * ROW_STEP;
				LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, col0) * 16u, b_src + q * b_step, 16u);
			}
			b_src += TILE_K;
		};
		float acc[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
		uint32_t sp_off = 0u, sc_off = 0u;
		issue_k(sp_off);
		LmStreamCommit();
		sp_off ^= STAGE_BYTES;
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<0u>();
			__syncthreads();
			const uint32_t stage_s = shared_base + sc_off;
			if ( kt + 1u < K_TILES )
			{
				issue_k(sp_off);
				sp_off ^= STAGE_BYTES;
			}
			LmStreamCommit();
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 4u; ++ks )
			{
				uint32_t a_reg[MF][4], b_reg[NF][2];
				#pragma unroll
				for ( uint32_t ii = 0u; ii < MF; ++ii )
				{
					const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
					const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
					LmStreamLdMatrix4(a_reg[ii], stage_s + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
				}
				#pragma unroll
				for ( uint32_t jj = 0u; jj < NF; jj += 2u )
				{
					const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
					const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
					uint32_t tmp[4];
					LmStreamLdMatrix4(tmp, stage_s + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u);
					b_reg[jj][0] = tmp[0]; b_reg[jj][1] = tmp[1];
					b_reg[jj + 1u][0] = tmp[2]; b_reg[jj + 1u][1] = tmp[3];
				}
				#pragma unroll
				for ( uint32_t i = 0u; i < MF; ++i )
					#pragma unroll
					for ( uint32_t j = 0u; j < NF; ++j )
						LmMmaBf16(acc[i][j], a_reg[i], b_reg[j]);
			}
			sc_off ^= STAGE_BYTES;
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
			{
				const uint32_t r0_out = row_base + warp_m * WARP_M + i * 16u + quad_row, r1_out = r0_out + 8u;
				const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
				if ( r0_out < row_limit )
					*(uint32_t *)(output + (uint64_t)r0_out * N + col) = LmPackBf16Pair(acc[i][j][0], acc[i][j][1]);
				if ( r1_out < row_limit )
					*(uint32_t *)(output + (uint64_t)r1_out * N + col) = LmPackBf16Pair(acc[i][j][2], acc[i][j][3]);
			}
	}
	LmDependentRelease();
}

template<class G, bool Q_ONLY = false>
__global__ __launch_bounds__(Q_ONLY ? G::kIndexHeads * 32u : QH_THREADS, 2)
static void QwenIndexQPrepKernel(const uint16_t *__restrict__ index_qk, const float *__restrict__ q_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	uint16_t *__restrict__ index_query, uint16_t *__restrict__ index_key, uint32_t rows)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, D_V = D >> 3u, H = G::kIndexHeads, HALF = G::kRopeDim / 2u;
	constexpr uint32_t SRC_W = Q_ONLY ? (H * D) : S::kIndexWidth;
	__shared__ __align__(16) uint16_t s_q[H][G::kRopeDim];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const uint4 *src4 = (const uint4 *)(index_qk + (uint64_t)row * SRC_W);
	if ( warp < H )
	{
		if ( context_length != 0 )
		{
			uint32_t bound = context_length[sequence_of_row[row]];
			if ( positions[row] + 1u < bound )
				bound = positions[row] + 1u;
			if ( (bound / G::kIndexRatio) <= S::kIndexBlocks )
			{
				LmDependentRelease();
				return;
			}
		}
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
	else if ( !Q_ONLY && warp == H && lane < D_V )
		((uint4 *)index_key)[(uint64_t)row * D_V + lane] = src4[H * D_V + lane];
	LmDependentRelease();
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
	{
		LmDependentRelease();
		return;
	}
	const uint32_t *lane_rows = sequence_row_indices + begin;
	const uint32_t first = positions[lane_rows[0]], end = positions[lane_rows[count - 1u]] + 1u;
	const uint32_t old_base = first / R * R, new_base = end / R * R, keep = end - new_base;
	uint16_t *slot = tail + (uint64_t)recurrent_slot[lane] * T * D;
	if ( (D & 7u) == 0u && LmIsAligned16(raw) && LmIsAligned16(slot) )
	{
		if ( d < (D >> 3u) )
		{
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
		}
		LmDependentRelease();
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
	LmDependentRelease();
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

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenIndexPackLocalKeysKernel(LmKvShardView view, const uint32_t *__restrict__ sequence_of_row,
	uint32_t total_blocks, uint32_t max_local, uint16_t *__restrict__ out_keys)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, D_V = D >> 3u;
	const uint32_t local = blockIdx.x;
	LmDependentWait();
	if ( local < max_local )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		const uint32_t local_count = SparkKvShardLocalKeys(view.shard, total_blocks);
		uint4 *dst4 = (uint4 *)(out_keys + (uint64_t)local * D);
		if ( local < local_count )
		{
			const uint16_t *key = (const uint16_t *)LmKvShardSlotRequired<typename S::IndexKv>(view, seq,
				SparkKvShardLocalPosition(view.shard, local), 0u, LM_KV_ACCESS_READ);
			if ( key != 0 )
			{
				const uint4 *src4 = (const uint4 *)key;
				for ( uint32_t c = threadIdx.x; c < D_V; c += QH_THREADS )
					dst4[c] = __ldg(src4 + c);
			}
		}
		else
		{
			for ( uint32_t c = threadIdx.x; c < D_V; c += QH_THREADS )
				dst4[c] = make_uint4(0u, 0u, 0u, 0u);
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 2)
static void QwenIndexGatheredScoreKernel(const uint16_t *__restrict__ gathered_keys, uint32_t max_local, uint32_t degree,
	const uint16_t *__restrict__ query, const uint32_t *__restrict__ positions, uint32_t first_row, uint32_t chunk_rows,
	uint32_t total_blocks, float *__restrict__ scores, float scale)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, H = G::kIndexHeads, D_V = D >> 3u;
	__shared__ alignas(16) float shared_query[H * D];
	const uint32_t local_row = blockIdx.y;
	LmDependentWait();
	if ( local_row >= chunk_rows )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t row = first_row + local_row;
	const uint32_t blocks = (__ldg(positions + row) + 1u) / G::kIndexRatio;
	if ( blocks <= S::kIndexBlocks || blockIdx.x * QH_THREADS >= blocks )
	{
		LmDependentRelease();
		return;
	}
	const uint4 *q4 = (const uint4 *)(query + (uint64_t)row * H * D);
	constexpr uint32_t q_chunks = (H * D) >> 3u;
	for ( uint32_t c = threadIdx.x; c < q_chunks; c += QH_THREADS )
	{
		float qf[8];
		LmLoadBf16x8AsFloat(__ldg(q4 + c), qf);
		LmStoreFloat8(shared_query + (c << 3u), qf);
	}
	__syncthreads();
	for ( uint32_t b = blockIdx.x * QH_THREADS + threadIdx.x; b < blocks; b += gridDim.x * QH_THREADS )
	{
		const uint32_t rk = degree > 1u ? (b % degree) : 0u;
		const uint32_t loc = degree > 1u ? (b / degree) : b;
		const uint4 *k4 = (const uint4 *)(gathered_keys + ((uint64_t)rk * max_local + loc) * D);
		float dots[H];
		#pragma unroll
		for ( uint32_t head = 0u; head < H; ++head )
			dots[head] = 0.0f;
		for ( uint32_t c = 0u; c < D_V; ++c )
		{
			float kv[8];
			LmLoadBf16x8AsFloat(__ldg(k4 + c), kv);
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
		float score = 0.0f;
		#pragma unroll
		for ( uint32_t head = 0u; head < H; ++head )
			score += fmaxf(dots[head], 0.0f);
		scores[(uint64_t)local_row * total_blocks + b] = score * scale;
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(256u, 2)
static void QwenIndexGatheredScoreMmaKernel(const uint16_t *__restrict__ gathered_keys, uint32_t max_local, uint32_t degree,
	const uint16_t *__restrict__ query, const uint32_t *__restrict__ positions, uint32_t first_row, uint32_t chunk_rows,
	uint32_t total_blocks, float *__restrict__ scores, float scale)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, H = G::kIndexHeads;
	constexpr uint32_t TILE_Q = 32u, TILE_B = 64u, TILE_K = 64u, K_TILES = D / TILE_K;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u;
	constexpr uint32_t K_STAGE_BYTES = TILE_B * ROW_BYTES;
	constexpr uint32_t Q_STAGE_BYTES = H * TILE_Q * ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = K_STAGE_BYTES + Q_STAGE_BYTES;
	static_assert((D % TILE_K) == 0u && H <= 8u, "valid index score MMA geometry");
	extern __shared__ __align__(128) uint8_t qh_idx_smem[];
	const uint32_t q_base = blockIdx.y * TILE_Q, b_base = blockIdx.x * TILE_B;
	const uint32_t q_end = q_base + TILE_Q < chunk_rows ? q_base + TILE_Q : chunk_rows;
	LmDependentWait();
	if ( q_base >= chunk_rows || b_base >= total_blocks )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t max_blocks = (__ldg(positions + first_row + q_end - 1u) + 1u) / G::kIndexRatio;
	if ( max_blocks <= S::kIndexBlocks || b_base >= max_blocks )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(qh_idx_smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_q = warp >> 2u, warp_b = warp & 3u;
	const uint32_t r0 = threadIdx.x / ROW_CHUNKS, c0 = threadIdx.x % ROW_CHUNKS;
	auto issue_k = [&](uint32_t st, uint32_t kt)
	{
		const uint32_t stage = shared_base + st * STAGE_BYTES;
		const uint32_t k_off = kt * TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < TILE_B / 32u; ++q )
		{
			const uint32_t rb = r0 + q * 32u;
			const uint32_t b = b_base + rb;
			const bool ok = b < max_blocks;
			const uint32_t rk = degree > 1u ? (b % degree) : 0u;
			const uint32_t loc = degree > 1u ? (b / degree) : b;
			const uint16_t *src = ok ? (gathered_keys + ((uint64_t)rk * max_local + loc) * D + k_off + c0 * 8u) : gathered_keys;
			LmStreamCopy(stage + rb * ROW_BYTES + LmStreamSwizzle8(rb, c0) * 16u, src, ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t h = 0u; h < H; ++h )
		{
			const uint32_t rq = r0;
			const uint32_t lr = q_base + rq;
			const bool ok = lr < q_end;
			const uint16_t *src = ok ? (query + ((uint64_t)(first_row + lr) * H + h) * D + k_off + c0 * 8u) : query;
			LmStreamCopy(stage + K_STAGE_BYTES + h * (TILE_Q * ROW_BYTES) + rq * ROW_BYTES + LmStreamSwizzle8(rq, c0) * 16u, src, ok ? 16u : 0u);
		}
		LmStreamCommit();
	};
	float dot[H][2][4];
	#pragma unroll
	for ( uint32_t h = 0u; h < H; ++h )
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			dot[h][j][0] = dot[h][j][1] = dot[h][j][2] = dot[h][j][3] = 0.0f;
	issue_k(0u, 0u);
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		const uint32_t cur = kt & 1u;
		if ( kt + 1u < K_TILES )
		{
			issue_k(cur ^ 1u, kt + 1u);
			LmStreamWait<1u>();
		}
		else
			LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage = shared_base + cur * STAGE_BYTES;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t bf[4];
			const uint32_t br = warp_b * 16u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
			const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
			LmStreamLdMatrix4(bf, stage + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
			const uint32_t b0[2] = { bf[0], bf[1] }, b1[2] = { bf[2], bf[3] };
			const uint32_t ar = warp_q * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
			const uint32_t aoff = ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u;
			#pragma unroll
			for ( uint32_t h = 0u; h < H; ++h )
			{
				uint32_t af[4];
				LmStreamLdMatrix4(af, stage + K_STAGE_BYTES + h * (TILE_Q * ROW_BYTES) + aoff);
				LmMmaBf16(dot[h][0], af, b0);
				LmMmaBf16(dot[h][1], af, b1);
			}
		}
		if ( kt + 1u < K_TILES )
			__syncthreads();
	}
	const uint32_t lr0 = q_base + warp_q * 16u + quad_row, lr1 = lr0 + 8u;
	const uint32_t blks0 = lr0 < q_end ? ((__ldg(positions + first_row + lr0) + 1u) / G::kIndexRatio) : 0u;
	const uint32_t blks1 = lr1 < q_end ? ((__ldg(positions + first_row + lr1) + 1u) / G::kIndexRatio) : 0u;
	#pragma unroll
	for ( uint32_t j = 0u; j < 2u; ++j )
	{
		const uint32_t b0 = b_base + warp_b * 16u + j * 8u + quad * 2u, b1 = b0 + 1u;
		if ( blks0 > S::kIndexBlocks )
		{
			if ( b0 < blks0 )
			{
				float s0 = 0.0f;
				#pragma unroll
				for ( uint32_t h = 0u; h < H; ++h )
					s0 += fmaxf(dot[h][j][0], 0.0f);
				scores[(uint64_t)lr0 * total_blocks + b0] = s0 * scale;
			}
			if ( b1 < blks0 )
			{
				float s1 = 0.0f;
				#pragma unroll
				for ( uint32_t h = 0u; h < H; ++h )
					s1 += fmaxf(dot[h][j][1], 0.0f);
				scores[(uint64_t)lr0 * total_blocks + b1] = s1 * scale;
			}
		}
		if ( blks1 > S::kIndexBlocks )
		{
			if ( b0 < blks1 )
			{
				float s2 = 0.0f;
				#pragma unroll
				for ( uint32_t h = 0u; h < H; ++h )
					s2 += fmaxf(dot[h][j][2], 0.0f);
				scores[(uint64_t)lr1 * total_blocks + b0] = s2 * scale;
			}
			if ( b1 < blks1 )
			{
				float s3 = 0.0f;
				#pragma unroll
				for ( uint32_t h = 0u; h < H; ++h )
					s3 += fmaxf(dot[h][j][3], 0.0f);
				scores[(uint64_t)lr1 * total_blocks + b1] = s3 * scale;
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(QH_THREADS, 4)
static void QwenIndexLocalMaskFromScoresKernel(const float *__restrict__ scores, uint32_t total_blocks,
	const uint32_t *__restrict__ positions, uint32_t first_row, uint32_t chunk_rows, uint32_t *__restrict__ mask, uint32_t mask_words)
{
	using S = QwenHybridShape<G>;
	__shared__ uint32_t histogram[256];
	__shared__ uint32_t s_state[6];
	const uint32_t local_row = blockIdx.x;
	LmDependentWait();
	if ( local_row >= chunk_rows )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t row = first_row + local_row;
	const uint32_t blocks = (__ldg(positions + row) + 1u) / G::kIndexRatio;
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
	const float *row_scores = scores + (uint64_t)local_row * total_blocks;
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
		for ( uint32_t b = threadIdx.x; b < blocks; b += QH_THREADS )
		{
			const uint32_t key = LmTopkKey(__ldg(row_scores + b));
			if ( (key & msk) == pfx )
				atomicAdd(&histogram[(key >> shift) & 255u], 1u);
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			uint32_t running = 0u, bucket = 255u;
			for ( ; bucket > 0u && running + histogram[bucket] < s_state[2]; --bucket )
				running += histogram[bucket];
			s_state[0] |= bucket << shift;
			s_state[1] |= 255u << shift;
			s_state[2] -= running;
			s_state[3] = histogram[bucket];
		}
		__syncthreads();
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
			for ( uint32_t b = threadIdx.x; b < blocks; b += QH_THREADS )
				if ( LmTopkKey(__ldg(row_scores + b)) == thresh && (b & pmsk) == ppfx )
					atomicAdd(&histogram[(b >> shift) & 255u], 1u);
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
	for ( uint32_t b = threadIdx.x; b < blocks; b += QH_THREADS )
	{
		const uint32_t key = LmTopkKey(__ldg(row_scores + b));
		if ( key > thresh || (key == thresh && b <= pos_thresh) )
			atomicOr(&row_mask[b >> 5u], 1u << (b & 31u));
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

	static bool PrefillWave(const SparkStageRunnerStep *step)
	{
		return step->commit != 0u && step->last_rows_only != 0u && step->tree == 0u;
	}

	static bool PrefillUnified(const SparkStageRunnerStep *step)
	{
		return PrefillWave(step) && step->sequences == 1u;
	}

	static int32_t LaunchExactBf16Multi(QwenBf16MultiArgs args, SparkGpuStream stream)
	{
		const uint32_t rows = args.rows;
		const uint32_t tile_n = rows <= 8u ? 16u : (rows <= 16u ? 8u : 4u);
		uint32_t blk = 0u;
		for ( uint32_t i = 0u; i < args.count; ++i )
		{
			args.part[i].first_block = blk;
			blk += (args.part[i].out_dim + tile_n - 1u) / tile_n;
		}
		args.total_blocks = blk;
		if ( rows <= 4u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16MultiKernel<1u, 4u, 4u, 4u, 2u>), dim3(blk, 1u), 128u, 0, stream, args);
		else if ( rows <= 8u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16MultiKernel<2u, 4u, 4u, 4u, 2u>), dim3(blk, 1u), 256u, 0, stream, args);
		else if ( rows <= 16u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16MultiKernel<4u, 2u, 4u, 4u, 2u>), dim3(blk, 1u), 256u, 0, stream, args);
		else
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16MultiKernel<4u, 2u, 8u, 2u, 2u>), dim3(blk, (rows + 31u) / 32u), 256u, 0, stream, args);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t LaunchBf16Mma(const uint16_t *weight, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream)
	{
		if ( output_dimension < 512u && (input_dimension % 256u) == 0u )
		{
			const uint32_t nb = (output_dimension + 15u) / 16u;
			if ( rows > 16u && (output_dimension >= 16u || rows >= 1024u) )
			{
				constexpr uint32_t smem = QwenBf16SmallNMmaWideSharedBytes<32u>();
				static bool granted_smalln32 = false;
				if ( !granted_smalln32 )
				{
					if ( SparkGpuFuncSetAttribute((const void *)QwenBf16SmallNMmaWideKernel<32u>,
						SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)smem) != SPARK_GPU_SUCCESS )
						return LM_LAUNCH_ERR_LAUNCH;
					granted_smalln32 = true;
				}
				(void)LM_LAUNCH_DEPENDENT((QwenBf16SmallNMmaWideKernel<32u>), ((rows + 31u) / 32u) * nb, 128u, smem, stream,
					weight, source, destination, rows, input_dimension, output_dimension);
			}
			else
				(void)LM_LAUNCH_DEPENDENT((QwenBf16SmallNMmaKernel), ((rows + 15u) / 16u) * nb, 128u, 0, stream,
					weight, source, destination, rows, input_dimension, output_dimension);
			return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
		}
		if ( output_dimension < 512u || rows > 128u )
			return LM_LAUNCH_ERR_SHAPE;
		const uint32_t nb = (output_dimension + 63u) / 64u;
		if ( rows <= 16u )
			(void)LM_LAUNCH_DEPENDENT((QwenBf16DenseMmaKernel<16u, 64u, 128u, 1u, 3u>), ((rows + 15u) / 16u) * nb, 128u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		else if ( rows <= 32u )
			(void)LM_LAUNCH_DEPENDENT((QwenBf16DenseMmaKernel<32u, 64u, 256u, 2u, 3u>), ((rows + 31u) / 32u) * nb, 256u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		else if ( rows <= 64u )
			(void)LM_LAUNCH_DEPENDENT((QwenBf16DenseMmaKernel<64u, 64u, 256u, 2u, 3u>), ((rows + 63u) / 64u) * nb, 256u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		else
			(void)LM_LAUNCH_DEPENDENT((QwenBf16DenseMmaKernel<128u, 64u, 256u, 4u, 2u>), ((rows + 127u) / 128u) * nb, 256u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<uint32_t TILE_M>
	static int32_t LaunchSharedGateUpSwigluTile(const uint16_t *weight, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t shared_rows, SparkGpuStream stream)
	{
		constexpr uint32_t smem = QwenSharedGateUpSwigluWideSharedBytes<TILE_M>();
		static bool granted = false;
		if ( !granted )
		{
			if ( SparkGpuFuncSetAttribute((const void *)QwenSharedGateUpSwigluWideKernel<TILE_M>,
				SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)smem) != SPARK_GPU_SUCCESS )
				return LM_LAUNCH_ERR_LAUNCH;
			granted = true;
		}
		const uint32_t blocks = ((rows + TILE_M - 1u) / TILE_M) * ((shared_rows + 15u) / 16u);
		(void)LM_LAUNCH_DEPENDENT((QwenSharedGateUpSwigluWideKernel<TILE_M>), blocks, 128u, smem, stream,
			weight, source, destination, rows, input_dimension, shared_rows);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t LaunchFusedGateUpSwiglu(const uint16_t *weight, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t ffn_rows, uint32_t exact, bool force_stream, SparkGpuStream stream)
	{
		const bool use_mma = (force_stream || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE)) && (input_dimension % 64u) == 0u;
		if ( use_mma )
		{
			if ( rows > 128u )
				return LM_LAUNCH_ERR_SHAPE;
			if ( rows <= 16u )
				(void)LM_LAUNCH_DEPENDENT((QwenBf16FusedGateUpSwigluMmaKernel<16u, 64u, 128u, 1u, 2u>),
					((rows + 15u) / 16u) * ((ffn_rows + 63u) / 64u), 128u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			else if ( rows <= 32u )
				(void)LM_LAUNCH_DEPENDENT((QwenBf16FusedGateUpSwigluMmaKernel<32u, 64u, 256u, 2u, 2u>),
					((rows + 31u) / 32u) * ((ffn_rows + 63u) / 64u), 256u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			else if ( rows <= 64u )
				(void)LM_LAUNCH_DEPENDENT((QwenBf16FusedGateUpSwigluMmaKernel<64u, 32u, 256u, 4u, 3u>),
					((rows + 63u) / 64u) * ((ffn_rows + 31u) / 32u), 256u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			else
				(void)LM_LAUNCH_DEPENDENT((QwenBf16FusedGateUpSwigluMmaKernel<128u, 32u, 256u, 4u, 2u>),
					((rows + 127u) / 128u) * ((ffn_rows + 31u) / 32u), 256u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
		}
		if ( rows == 1u || rows > 32u )
			return LM_LAUNCH_ERR_SHAPE;
		if ( rows <= 4u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16GateUpSwigluKernel<1u, 4u, 4u, 4u, 2u>), dim3((ffn_rows + 15u) / 16u, 1u), 128u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		else if ( rows <= 8u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16GateUpSwigluKernel<2u, 4u, 4u, 4u, 2u>), dim3((ffn_rows + 15u) / 16u, 1u), 256u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		else if ( rows <= 16u )
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16GateUpSwigluKernel<4u, 2u, 4u, 4u, 2u>), dim3((ffn_rows + 7u) / 8u, 1u), 256u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		else
			(void)LM_LAUNCH_DEPENDENT((QwenExactBf16GateUpSwigluKernel<4u, 2u, 8u, 2u, 2u>), dim3((ffn_rows + 3u) / 4u, 1u), 256u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<class Format>
	static int32_t Project(const void *weight, const float *scale_rows, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, uint32_t multiprocessors, SparkGpuStream stream,
		bool force_stream = false)
	{
		if constexpr ( std::is_same<Format, LmBf16Format>::value )
		{
			const bool use_mma = force_stream || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE);
			if ( use_mma && (input_dimension % 64u) == 0u && (output_dimension & 1u) == 0u )
			{
				const int32_t status = LaunchBf16Mma((const uint16_t *)weight, source, destination, rows, input_dimension, output_dimension, stream);
				if ( status != LM_LAUNCH_ERR_SHAPE )
					return status;
			}
			else if ( !use_mma && rows >= 2u && rows <= 32u && input_dimension >= 256u && (input_dimension & 7u) == 0u && output_dimension > 384u )
			{
				QwenBf16MultiArgs args;
				args.part[0] = { (const uint16_t *)weight, destination, output_dimension, 0u };
				args.count = 1u;
				args.activation = source;
				args.rows = rows;
				args.input_dim = input_dimension;
				return LaunchExactBf16Multi(args, stream);
			}
		}
		const LmScaleTensor scale = scale_rows != 0 ?
			LmScaleTensorBuild(scale_rows, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u, QH_SCALE_K) : LmScaleTensorNone();
		if ( force_stream )
		{
			const int32_t status = LmStreamGemmDense<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension,
				output_dimension, 0u, 0u, multiprocessors, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmDenseRows<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension, output_dimension, exact, multiprocessors,
			stream);
	}

	static int32_t Linear(const State *state, const uint8_t *weight, const float *scale_rows, const uint16_t *source,
		uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, uint32_t multiprocessors,
		SparkGpuStream stream, bool force_stream = false)
	{
		return state->linear_bytes == 1u ?
			Project<LmFp8>(weight, scale_rows, source, destination, rows, input_dimension, output_dimension, exact, multiprocessors, stream, force_stream) :
			Project<LmBf16Format>(weight, 0, source, destination, rows, input_dimension, output_dimension, exact, multiprocessors, stream, force_stream);
	}

	static uint32_t Exact(const SparkStageRunnerStep *step)
	{
		return step->commit == 0u && step->sequences < QH_VERIFY_INVARIANT_SEQS ? 1u : 0u;
	}

	static SparkStatus GdnRecur(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, const uint16_t *qkv, const uint16_t *ba,
		const uint32_t *kept, uint32_t commit, SparkGpuStream stream, uint32_t qkv_stride = 0u, uint16_t *replay_qkv_out = 0)
	{
		const uint32_t rows = step->rows, sequences = step->sequences, heads = state->gdn_heads;
		const uint32_t tree = step->tree != 0u && commit == 0u ? 1u : 0u;
		uint8_t *state_pool = state->gdn_state + (uint64_t)layer->recurrent_index * state->slots * state->gdn_state_bytes;
		uint16_t *window = state->gdn_window + (uint64_t)layer->recurrent_index * state->slots * (state->gdn_window_bytes / sizeof(uint16_t));
		if ( tree == 0u && commit != 0u && kept == 0 && PrefillWave(step) )
		{
			const uint32_t input_stride = qkv_stride != 0u ? qkv_stride : state->gdn_channels;
			const uint32_t max_row_tiles = (rows + LM_CAUSAL_CONV_TILE_ROWS - 1u) / LM_CAUSAL_CONV_TILE_ROWS;
			const uint32_t ch_tiles = (state->gdn_channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS;
			(void)LM_LAUNCH_DEPENDENT((QwenGdnConvPrefill2DKernel<QH_CONV, QH_KEY_DIM>),
				dim3(sequences, ch_tiles, max_row_tiles), LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
				window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
				state->query_heads, state->key_heads, state->value, state->gdn_qk, state->gdn_v,
				input_stride, sequences, max_row_tiles == 1u ? 1u : 0u, step->sequence_row_indices);
			if ( max_row_tiles > 1u )
				(void)LM_LAUNCH_DEPENDENT((QwenGdnConvWindowCommitKernel<QH_CONV>),
					dim3(sequences, (state->gdn_channels + QH_THREADS - 1u) / QH_THREADS), QH_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, kept, qkv,
					state->gdn_channels, input_stride, sequences, step->sequence_row_indices);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS ||
				QwenGdnScalarChunkLaunch<QH_KEY_DIM, QH_VALUE_DIM>(state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
					step->sequence_row_begin, kept, step->positions, state->query_heads, state->key_heads, state->value, ba,
					layer->gdn_a_log, layer->gdn_dt_bias, state->mixed, state->gdn_qk / QH_KEY_DIM, heads,
					sequences, commit, step->sequence_row_indices, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return CudaStageLaunched();
		}
		if ( tree == 0u )
		{
			const uint32_t input_stride = qkv_stride != 0u ? qkv_stride : state->gdn_channels;
			const uint32_t gate_blocks = (heads + (QH_THREADS / LM_WARP_LANES) - 1u) / (QH_THREADS / LM_WARP_LANES);
			if ( rows <= LM_CAUSAL_CONV_SERIAL_ROWS * sequences )
			{
				const uint32_t conv_blocks = (state->gdn_channels + QH_THREADS - 1u) / QH_THREADS;
				(void)LM_LAUNCH_DEPENDENT((QwenGdnConvSplitSerialKernel<QH_CONV, QH_KEY_DIM>),
					dim3(sequences, conv_blocks + gate_blocks), QH_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
					ba, layer->gdn_a_log, layer->gdn_dt_bias,
					state->query_heads, state->key_heads, state->value, state->retention, state->write_gate,
					replay_qkv_out, state->gdn_qk, state->gdn_v, heads, input_stride, conv_blocks, sequences, commit,
					step->sequence_row_indices);
			}
			else
			{
				(void)LM_LAUNCH_DEPENDENT((QwenGdnConvSplitTiledKernel<QH_CONV, QH_KEY_DIM>),
					dim3(sequences, (state->gdn_channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
					LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
					state->query_heads, state->key_heads, state->value, replay_qkv_out,
					state->gdn_qk, state->gdn_v, heads, input_stride, sequences, commit, step->sequence_row_indices);
				(void)LM_LAUNCH_DEPENDENT((QwenGdnGateOnlyKernel<QH_KEY_DIM>),
					dim3(gate_blocks, rows), QH_THREADS, 0, stream,
					ba, layer->gdn_a_log, layer->gdn_dt_bias, state->retention, state->write_gate, heads, rows);
			}
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( LmCausalConvLaunch<QH_CONV,LM_CONV_SWISH,uint16_t>(window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
				state->qkv, state->gdn_channels, sequences, commit, step->sequence_row_indices, rows, stream, step->parent_rows) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((QwenGdnSplitExpandGateKernel<QH_THREADS,QH_KEY_DIM>), rows, QH_THREADS, 0, stream,
				state->qkv, ba, layer->gdn_a_log, layer->gdn_dt_bias, state->query_heads, state->key_heads, state->value,
				state->retention, state->write_gate, state->gdn_qk, state->gdn_v, heads, rows);
		}
		if ( commit != 0u && kept == 0 && PrefillWave(step) )
		{
			if ( LmDeltaRuleChunkLaunch<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>(state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
					step->sequence_row_begin, kept, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
					heads, 1u, sequences, commit, step->sequence_row_indices, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( LmDeltaRuleLaunch<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>(rows, state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
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
		const bool prefill_wave = PrefillWave(step);
		const bool fused_qkvz = (prefill_wave || rows > LM_SKINNY_ROWS_WIDE) && exact == 0u && step->tree == 0u;
		SparkStatus status;
		if ( !prefill_wave && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			if ( state->linear_bytes == 2u )
			{
				if ( rows >= 2u && rows <= 32u && G::kHidden >= 256u && (G::kHidden & 7u) == 0u )
				{
					QwenBf16MultiArgs args;
					args.part[0] = { (const uint16_t *)layer->mix_weight, state->qkv, state->gdn_channels, 0u };
					args.part[1] = { (const uint16_t *)(layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * 2u), state->z, state->gdn_v, 0u };
					args.part[2] = { layer->gdn_ba, state->ba, 2u * heads, 0u };
					args.count = 3u;
					args.activation = state->normed;
					args.rows = rows;
					args.input_dim = G::kHidden;
					if ( LaunchExactBf16Multi(args, stream) != LM_LAUNCH_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else
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
			if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->gdn_qkvz, rows, G::kHidden, qkvz_width, 0u, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
				Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, 0u, sms, stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->qkv, rows, G::kHidden, state->gdn_channels, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
				Linear(state, layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * state->linear_bytes,
					layer->mix_scale == 0 ? (const float *)0 : layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
					state->normed, state->z, rows, G::kHidden, state->gdn_v, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
				Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( step->commit == 0u )
		{
			if ( !fused_qkvz &&
				SparkGpuMemcpyAsync(state->replay_qkv + replay * state->gdn_channels, state->qkv, (uint64_t)rows * state->gdn_channels * sizeof(uint16_t),
					SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			if ( SparkGpuMemcpyAsync(state->replay_ba + replay * 2u * heads, state->ba, (uint64_t)rows * 2u * heads * sizeof(uint16_t),
					SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		status = GdnRecur(state, layer, step,
			fused_qkvz ? state->gdn_qkvz : (step->tree != 0u ? state->replay_qkv + replay * state->gdn_channels : state->qkv),
			state->ba, (const uint32_t *)0, step->commit, stream, fused_qkvz ? qkvz_width : 0u,
			fused_qkvz && step->commit == 0u ? state->replay_qkv + replay * state->gdn_channels : (uint16_t *)0);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( status != SPARK_STATUS_OK )
			return status;
		(void)LM_LAUNCH_DEPENDENT((QwenGdnHeadRmsNormGateKernel<G::kGdnSigmoidGate>), dim3((heads + 7u) >> 3u, rows), QH_THREADS, 0, stream,
			state->mixed, fused_qkvz ? state->gdn_qkvz : state->z, layer->gdn_norm, state->mixed, rows, heads,
			fused_qkvz ? qkvz_width : state->gdn_v, fused_qkvz ? state->gdn_channels : 0u);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->mixed, state->partial, rows, state->gdn_v, G::kHidden, exact, sms, stream, prefill_wave) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus IndexPrepare(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream,
		bool force_stream = false, bool host_dense = false)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, exact = Exact(step);
		uint16_t *tail = state->index_tail + (uint64_t)layer->kv_index * state->slots * S::kIndexTail * G::kIndexDim;
		constexpr uint32_t Q_DIM = G::kIndexHeads * G::kIndexDim;
		constexpr bool kFastIndexSkip = G::kIndexDim <= 256u && (G::kIndexDim & 7u) == 0u && G::kIndexHeads + 1u <= QH_THREADS / LM_WARP_LANES &&
			G::kRopeDim == 64u && (G::kHidden % 64u) == 0u && (Q_DIM % 64u) == 0u;
		if constexpr ( kFastIndexSkip )
		{
			if ( host_dense )
			{
				if ( Project<LmBf16Format>(layer->index_qk + (uint64_t)Q_DIM * G::kHidden, 0, state->normed, state->index_key, rows, G::kHidden,
					G::kIndexDim, exact, sms, stream, force_stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( !force_stream && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			{
				(void)LM_LAUNCH_DEPENDENT((QwenIndexQkExactSkipKernel<G>), dim3((S::kIndexWidth + 7u) >> 3u, (rows + 7u) >> 3u), 128u, 0, stream,
					layer->index_qk, state->normed, state->index_qk, state->index_key, step->sequence_of_row, step->context_length, step->positions, rows);
				(void)LM_LAUNCH_DEPENDENT((QwenIndexQPrepKernel<G, true>), rows, G::kIndexHeads * 32u, 0, stream,
					state->index_qk, layer->index_q_norm, step->positions, step->sequence_of_row, step->context_length,
					state->index_query, state->index_key, rows);
			}
			else
			{
				if ( Project<LmBf16Format>(layer->index_qk + (uint64_t)Q_DIM * G::kHidden, 0, state->normed, state->index_key, rows, G::kHidden,
					G::kIndexDim, exact, sms, stream, force_stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( rows > 32u )
					(void)LM_LAUNCH_DEPENDENT((QwenIndexQMmaSkipKernel<G, 64u, 4u>), dim3(Q_DIM / 64u, (rows + 63u) / 64u), 256u, 0, stream,
						layer->index_qk, state->normed, state->index_qk, step->sequence_of_row, step->context_length, step->positions, rows);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenIndexQMmaSkipKernel<G, 32u, 2u>), dim3(Q_DIM / 64u, (rows + 31u) / 32u), 256u, 0, stream,
						layer->index_qk, state->normed, state->index_qk, step->sequence_of_row, step->context_length, step->positions, rows);
				(void)LM_LAUNCH_DEPENDENT((QwenIndexQPrepKernel<G, true>), rows, G::kIndexHeads * 32u, 0, stream,
					state->index_qk, layer->index_q_norm, step->positions, step->sequence_of_row, step->context_length,
					state->index_query, state->index_key, rows);
			}
		}
		else
		{
			if ( Project<LmBf16Format>(layer->index_qk, 0, state->normed, state->index_qk, rows, G::kHidden, S::kIndexWidth, exact, sms, stream, force_stream) !=
				LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if constexpr ( G::kIndexDim <= 256u && (G::kIndexDim & 7u) == 0u && G::kIndexHeads + 1u <= QH_THREADS / LM_WARP_LANES && G::kRopeDim == 64u )
			{
				(void)LM_LAUNCH_DEPENDENT((QwenIndexQPrepKernel<G, false>), rows, QH_THREADS, 0, stream, state->index_qk, layer->index_q_norm, step->positions,
					step->sequence_of_row, step->context_length, state->index_query, state->index_key, rows);
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

	static SparkStatus IndexSelect(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream,
		bool prefill_unified, bool host_dense)
	{
		if ( host_dense )
			return SPARK_STATUS_OK;
		const uint32_t rows = step->rows, degree = state->tp_degree;
		const LmKvShardView *view = &state->index_views[layer->kv_index];
		SparkStatus status;
		const uint32_t total_ctx = prefill_unified ? step->gather_context : 0u;
		const uint32_t total_blocks = total_ctx / G::kIndexRatio;
		const uint32_t max_local = total_blocks > S::kIndexBlocks ? SparkKvShardGatherKeys(view->shard, total_blocks) : 0u;
		const uint64_t cand_cap_bytes = (uint64_t)state->max_rows * S::kIndexBlocks * sizeof(uint2);
		const uint64_t gath_cap_bytes = (uint64_t)degree * cand_cap_bytes;
		const uint64_t pack_bytes = (uint64_t)max_local * G::kIndexDim * sizeof(uint16_t);
		const uint64_t keys_bytes = ((uint64_t)degree * pack_bytes + 63u) & ~(uint64_t)63u;
		const uint64_t score_floats = gath_cap_bytes > keys_bytes ? (gath_cap_bytes - keys_bytes) / sizeof(float) : 0u;
		const bool use_compact_index = prefill_unified && total_blocks > S::kIndexBlocks && max_local != 0u &&
			max_local <= state->max_rows && pack_bytes <= cand_cap_bytes &&
			score_floats >= total_blocks && G::kIndexDim <= G::kHidden && ((G::kIndexHeads * G::kIndexDim) & 7u) == 0u;
		if ( use_compact_index )
		{
			uint16_t *pack_buf = (uint16_t *)state->index_candidates;
			uint16_t *keys_buf = (uint16_t *)state->index_gathered;
			float *scores_buf = (float *)((uint8_t *)state->index_gathered + keys_bytes);
			const uint32_t chunk_cap = (uint32_t)(score_floats / total_blocks);
			const uint32_t chunk_rows = chunk_cap < rows ? chunk_cap : rows;
			const float scale = 1.0f / sqrtf((float)G::kIndexDim);
			(void)LM_LAUNCH_DEPENDENT((QwenIndexPackLocalKeysKernel<G>), max_local, QH_THREADS, 0, stream, *view,
				step->sequence_of_row, total_blocks, max_local, degree > 1u ? pack_buf : keys_buf);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( degree > 1u )
			{
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, max_local, G::kIndexDim, pack_buf, keys_buf);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			for ( uint32_t first = 0u; first < rows; first += chunk_rows )
			{
				const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
				(void)LM_LAUNCH_DEPENDENT((QwenIndexGatheredScoreKernel<G>), dim3((total_blocks + QH_THREADS - 1u) / QH_THREADS, count), QH_THREADS, 0, stream,
					keys_buf, max_local, degree, state->index_query, step->positions, first, count, total_blocks, scores_buf, scale);
				(void)LM_LAUNCH_DEPENDENT((QwenIndexLocalMaskFromScoresKernel<G>), count, QH_THREADS, 0, stream,
					scores_buf, total_blocks, step->positions, first, count, state->index_mask, state->index_mask_words);
			}
			return CudaStageLaunched();
		}
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
		const uint32_t exact = Exact(step);
		const bool prefill_wave = PrefillWave(step);
		const bool prefill_unified = PrefillUnified(step);
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->attn_local, rows, G::kHidden,
			state->attn_rows, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint32_t group = G::kHeads / G::kKvHeads;
		const uint32_t elem_first = state->tp_rank * state->attn_width;
		const uint32_t q_first = elem_first / G::kHeadDim;
		const uint32_t q_last = (elem_first + state->attn_width - 1u) / G::kHeadDim;
		const uint32_t rank_heads = q_last - q_first + 1u;
		const uint32_t kv_first = q_first / group;
		const uint32_t kv_last = q_last / group;
		const uint32_t local_kvh = kv_last - kv_first + 1u;
		bool uniform_kvh = true;
		for ( uint32_t peer = 0u; peer < degree; ++peer )
		{
			const uint32_t p_first = (peer * state->attn_width) / G::kHeadDim;
			const uint32_t p_last = (peer * state->attn_width + state->attn_width - 1u) / G::kHeadDim;
			if ( (p_last / group) - (p_first / group) + 1u != local_kvh )
				uniform_kvh = false;
		}
		const uint32_t grain = view->shard.grain;
		QwenPrefillSeqPlan plan;
		memset(&plan, 0, sizeof(plan));
		bool plan_ok = false;
		uint32_t max_seq_rows = 0u, max_seq_prior = 0u, max_seq_peer = 0u, max_seq_total_ctx = 0u;
		if ( prefill_wave && uniform_kvh && G::kHeadDim == 256u && (G::kHeads % G::kKvHeads) == 0u &&
			step->sequences != 0u && step->sequences <= QH_PREFILL_MAX_SEQS )
		{
			if ( step->sequences == 1u )
			{
				const uint32_t total_ctx = step->gather_context;
				if ( total_ctx >= rows )
				{
					const uint32_t old_ctx = total_ctx - rows;
					const uint32_t b1 = (old_ctx + grain - 1u) / grain;
					const uint32_t peer_keys = old_ctx != 0u ? (((b1 + degree - 1u) / degree) * grain) : 0u;
					plan.count = 1u;
					plan.row_begin[0] = 0u;
					plan.row_len[0] = rows;
					plan.old_ctx[0] = old_ctx;
					plan.prior_off[0] = 0u;
					plan.peer_off[0] = 0u;
					plan.peer_keys[0] = peer_keys;
					plan.total_prior = old_ctx;
					plan.total_peer = peer_keys;
					max_seq_rows = rows;
					max_seq_prior = old_ctx;
					max_seq_peer = peer_keys;
					max_seq_total_ctx = total_ctx;
					plan_ok = true;
				}
			}
			else if ( step->host_positions != 0 && step->host_sequence_of_row != 0 )
			{
				uint32_t r = 0u, s = 0u;
				bool valid = true;
				while ( r < rows && s < step->sequences )
				{
					const uint32_t seq_id = step->host_sequence_of_row[r];
					const uint32_t old_ctx = step->host_positions[r];
					const uint32_t r0 = r;
					++r;
					while ( r < rows && step->host_sequence_of_row[r] == seq_id && step->host_positions[r] == old_ctx + (r - r0) )
						++r;
					const uint32_t len = r - r0;
					const uint32_t total_ctx = old_ctx + len;
					const uint32_t b1 = (old_ctx + grain - 1u) / grain;
					const uint32_t peer_keys = old_ctx != 0u ? (((b1 + degree - 1u) / degree) * grain) : 0u;
					plan.seq_id[s] = seq_id;
					plan.row_begin[s] = r0;
					plan.row_len[s] = len;
					plan.old_ctx[s] = old_ctx;
					plan.prior_off[s] = plan.total_prior;
					plan.peer_off[s] = plan.total_peer;
					plan.peer_keys[s] = peer_keys;
					plan.total_prior += old_ctx;
					plan.total_peer += peer_keys;
					if ( len > max_seq_rows )
						max_seq_rows = len;
					if ( old_ctx > max_seq_prior )
						max_seq_prior = old_ctx;
					if ( peer_keys > max_seq_peer )
						max_seq_peer = peer_keys;
					if ( total_ctx > max_seq_total_ctx )
						max_seq_total_ctx = total_ctx;
					++s;
				}
				if ( r != rows || s != step->sequences )
					valid = false;
				plan.count = s;
				plan_ok = valid;
			}
		}
		const bool host_dense = S::kIndexer && plan_ok && max_seq_total_ctx != 0u && (max_seq_total_ctx / G::kIndexRatio) <= S::kIndexBlocks;
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t peer_bytes = (uint64_t)plan.total_peer * local_kvh * (2u * G::kHeadDim) * sizeof(uint16_t);
		const uint32_t a2a_units = (degree > 1u && plan.total_prior != 0u) ? (uint32_t)((peer_bytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t a2a_stride_bytes = (uint64_t)a2a_units * unit_bytes;
		const uint64_t kv_cur_bytes = (uint64_t)local_kvh * rows * (2u * G::kHeadDim) * sizeof(uint16_t);
		const uint64_t prior_bytes = (uint64_t)local_kvh * plan.total_prior * (2u * G::kHeadDim) * sizeof(uint16_t);
		const uint64_t buf_cap_bytes = (uint64_t)degree * state->record_stride * sizeof(float);
		const uint64_t query_cap_elems = (uint64_t)state->max_rows * S::kQWidth;
		const uint64_t query_used_elems = (uint64_t)rows * rank_heads * G::kHeadDim;
		constexpr bool kDirectPaged = G::kTpDegree == 1u && (S::Kv::kPageSlots % 32u) == 0u;
		const bool kv_cur_in_query = degree > 1u && query_used_elems + (kv_cur_bytes / sizeof(uint16_t)) <= query_cap_elems;
		const bool tp1_buf_fits = degree == 1u && (kv_cur_bytes + prior_bytes) <= buf_cap_bytes;
		const bool tp_buf_fits = degree > 1u && kv_cur_in_query && (uint64_t)degree * a2a_stride_bytes <= buf_cap_bytes && prior_bytes <= buf_cap_bytes;
		const bool use_local_prefill = plan_ok && (tp1_buf_fits || tp_buf_fits);
		if ( use_local_prefill )
		{
			constexpr bool kKvOnlyGather = G::kRankHeads && G::kTpDegree > 1u && (2u * S::kKvWidth / G::kTpDegree) <= G::kHidden;
			if constexpr ( G::kTpDegree > 1u )
			{
				if constexpr ( kKvOnlyGather )
				{
					constexpr uint32_t KV_LOCAL = 2u * S::kKvWidth / G::kTpDegree;
					constexpr uint32_t KV_LOCAL_V = KV_LOCAL >> 3u;
					(void)LM_LAUNCH_DEPENDENT((QwenPackLocalKvKernel<G>),
						LmGridBlocks<QH_THREADS>((uint64_t)rows * KV_LOCAL_V), QH_THREADS, 0, stream,
						state->attn_local, state->attn_query, rows, state->attn_rows);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, KV_LOCAL,
						state->attn_query, state->attn_gathered);
				}
				else
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * S::kGatherParts,
						state->attn_rows / S::kGatherParts, state->attn_local, state->attn_gathered);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			uint16_t *kv_cur = degree > 1u ? (state->attn_query + query_used_elems) : (uint16_t *)state->send;
			uint16_t *kv_prior = degree > 1u ? (uint16_t *)state->send : (uint16_t *)((uint8_t *)state->send + kv_cur_bytes);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			{
				constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
				const uint32_t local_tasks = rank_heads + 2u * G::kKvHeads;
				(void)LM_LAUNCH_DEPENDENT((QwenAttnPrepLocalKernel<G, kKvOnlyGather>), dim3((local_tasks + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
					state->attn_local, state->attn_gathered, layer->q_norm, layer->k_norm, step->positions, step->sequence_of_row, *view,
					state->attn_query, state->merged_gate, kv_cur, rows, state->attn_rows, state->tp_rank, state->attn_width,
					q_first, rank_heads, kv_first, local_kvh);
			}
			if constexpr ( S::kIndexer )
			{
				status = IndexPrepare(state, layer, step, stream, prefill_wave, host_dense);
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
				if ( status == SPARK_STATUS_OK )
					status = IndexSelect(state, layer, step, stream, prefill_unified, host_dense);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			else
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( plan.total_prior != 0u )
			{
				if ( degree > 1u )
				{
					const uint32_t peer_blocks = (max_seq_peer + 7u) >> 3u;
					(void)LM_LAUNCH_DEPENDENT((QwenPriorKvPackKernel<typename S::Kv, G::kKvHeads, G::kHeadDim>),
						dim3(peer_blocks, degree, plan.count), QH_THREADS, 0, stream,
						*view, step->sequence_of_row, plan, state->attn_width, group, local_kvh, a2a_stride_bytes, (uint8_t *)state->send);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, a2a_units, 0u, state->send, state->received);
					if ( status != SPARK_STATUS_OK )
						return status;
					(void)LM_LAUNCH_DEPENDENT((QwenPriorKvUnpackKernel<G::kHeadDim>),
						dim3(peer_blocks, degree, plan.count), QH_THREADS, 0, stream,
						view->shard, (const uint8_t *)state->received, a2a_stride_bytes, plan, local_kvh, kv_prior);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else if constexpr ( !kDirectPaged )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenPriorKvDirectKernel<typename S::Kv, G::kKvHeads, G::kHeadDim>),
						dim3(max_seq_prior, local_kvh, plan.count), QH_THREADS, 0, stream,
						*view, step->sequence_of_row, plan, kv_first, local_kvh, kv_prior);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			constexpr bool kGateFromLocal = G::kTpDegree == 1u || kKvOnlyGather;
			constexpr uint32_t shared_bytes = QwenLocalPrefillSharedBytes<G::kHeadDim>();
			const uint32_t q_blocks = (max_seq_rows + 31u) / 32u;
			const uint16_t *gate_ptr = kGateFromLocal ? state->attn_local : state->merged_gate;
			const uint32_t gate_stride = kGateFromLocal ? state->attn_rows : state->attn_width;
			const dim3 prefill_grid = kDirectPaged ? dim3(group * q_blocks, local_kvh, plan.count) : dim3(rank_heads, q_blocks, plan.count);
			if ( S::kIndexer && !host_dense )
			{
				static bool granted_masked = false;
				if ( !granted_masked )
				{
					if ( SparkGpuFuncSetAttribute((const void *)QwenLocalPrefillAttnKernel<typename S::Kv, G::kKvHeads, G::kHeadDim, true, kGateFromLocal, kDirectPaged>,
						SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared_bytes) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					granted_masked = true;
				}
				(void)LM_LAUNCH_DEPENDENT((QwenLocalPrefillAttnKernel<typename S::Kv, G::kKvHeads, G::kHeadDim, true, kGateFromLocal, kDirectPaged>),
					prefill_grid, 128u, shared_bytes, stream,
					state->attn_query, kv_prior, kv_cur, *view, step->sequence_of_row, gate_ptr, gate_stride, step->positions, plan, rows, rank_heads,
					q_first, group, kv_first, elem_first, state->attn_width, 1.0f / sqrtf((float)G::kHeadDim),
					state->index_mask, state->index_mask_words, G::kIndexRatio, S::kIndexBlocks, state->merged);
			}
			else
			{
				static bool granted_dense = false;
				if ( !granted_dense )
				{
					if ( SparkGpuFuncSetAttribute((const void *)QwenLocalPrefillAttnKernel<typename S::Kv, G::kKvHeads, G::kHeadDim, false, kGateFromLocal, kDirectPaged>,
						SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared_bytes) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					granted_dense = true;
				}
				(void)LM_LAUNCH_DEPENDENT((QwenLocalPrefillAttnKernel<typename S::Kv, G::kKvHeads, G::kHeadDim, false, kGateFromLocal, kDirectPaged>),
					prefill_grid, 128u, shared_bytes, stream,
					state->attn_query, kv_prior, kv_cur, *view, step->sequence_of_row, gate_ptr, gate_stride, step->positions, plan, rows, rank_heads,
					q_first, group, kv_first, elem_first, state->attn_width, 1.0f / sqrtf((float)G::kHeadDim),
					(const uint32_t *)0, 0u, G::kIndexRatio, S::kIndexBlocks, state->merged);
			}
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, exact, sms,
					stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return SPARK_STATUS_OK;
		}
		constexpr bool kQkvNoGateGather = G::kRankHeads && G::kTpDegree > 1u &&
			(S::kQWidth % (G::kTpDegree * G::kHeadDim)) == 0u && (S::kKvWidth % (G::kTpDegree * 8u)) == 0u &&
			((S::kQWidth + 2u * S::kKvWidth) / G::kTpDegree) <= G::kHidden;
		if constexpr ( G::kTpDegree > 1u )
		{
			if constexpr ( kQkvNoGateGather )
			{
				constexpr uint32_t QKV_LOCAL = (S::kQWidth + 2u * S::kKvWidth) / G::kTpDegree;
				constexpr uint32_t QKV_LOCAL_V = QKV_LOCAL >> 3u;
				(void)LM_LAUNCH_DEPENDENT((QwenPackLocalQkvNoGateKernel<G>),
					LmGridBlocks<QH_THREADS>((uint64_t)rows * QKV_LOCAL_V), QH_THREADS, 0, stream,
					state->attn_local, state->attn_query, rows, state->attn_rows);
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, QKV_LOCAL,
					state->attn_query, state->attn_gathered);
			}
			else
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * S::kGatherParts, state->attn_rows / S::kGatherParts,
					state->attn_local, state->attn_gathered);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		{
			constexpr uint32_t tasks = G::kHeads + 2u * G::kKvHeads;
			constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((QwenHybridAttnPrepKernel<G, kQkvNoGateGather>), dim3((tasks + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
				state->attn_gathered, layer->q_norm, layer->k_norm, step->positions, step->tree != 0u ? step->slots : step->positions,
				step->sequence_of_row, *view, state->attn_query, state->merged_gate, rows, state->attn_rows, state->tp_rank, state->attn_width);
		}
		if constexpr ( S::kIndexer )
		{
			status = IndexPrepare(state, layer, step, stream, prefill_wave, host_dense);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( status == SPARK_STATUS_OK )
				status = IndexSelect(state, layer, step, stream, prefill_unified, host_dense);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		else
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		constexpr bool kSplitGqaCapable = G::kTpDegree == 1u && !S::kIndexer && G::kHeadDim == 256u && (G::kHeads % G::kKvHeads) == 0u;
		bool split_done = false;
		if constexpr ( kSplitGqaCapable )
		{
			const uint64_t per_row_floats = (uint64_t)QH_GQA_MAX_SPLITS * G::kHeads * (G::kHeadDim + 4u);
			const uint32_t chunk_rows = (uint32_t)(state->split_gqa_floats / per_row_floats);
			if ( !prefill_wave && chunk_rows > 0u && (rows <= 16u || (step->commit == 0u && step->sequences < QH_VERIFY_INVARIANT_SEQS)) )
			{
				constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
				for ( uint32_t first = 0u; first < rows; first += chunk_rows )
				{
					const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
					(void)LM_LAUNCH_DEPENDENT((QwenGqaSplitDecodeKernel<typename S::Kv, G::kHeads, G::kKvHeads, G::kHeadDim, S::kQkvWidth>),
						dim3(count, G::kKvHeads, QH_GQA_MAX_SPLITS), (G::kHeads / G::kKvHeads) * 32u, 0, stream,
						*view, state->attn_query + (uint64_t)first * G::kHeads * G::kHeadDim,
						state->attn_local + (uint64_t)first * S::kQkvWidth,
						step->sequence_of_row + first, step->context_length, step->positions + first,
						step->tree != 0u ? step->fork_positions + first : (const uint32_t *)0,
						step->tree != 0u ? step->slot_offsets + first : (const uint32_t *)0,
						1.0f / sqrtf((float)G::kHeadDim), state->send,
						state->merged + (uint64_t)first * G::kHeads * G::kHeadDim, count);
					(void)LM_LAUNCH_DEPENDENT((QwenGqaSplitReduceGateKernel<G::kHeads, G::kHeadDim, S::kQkvWidth>),
						dim3((G::kHeads + warps - 1u) / warps, count), QH_THREADS, 0, stream,
						state->send, state->attn_local + (uint64_t)first * S::kQkvWidth,
						step->sequence_of_row + first, step->context_length, step->positions + first,
						state->merged + (uint64_t)first * G::kHeads * G::kHeadDim, count);
				}
				split_done = true;
			}
			else if ( !prefill_wave && step->tree == 0u && rows == step->sequences && rows > 16u )
			{
				(void)LM_LAUNCH_DEPENDENT((QwenGqaDirectDecodeGateKernel<typename S::Kv, G::kHeads, G::kKvHeads, G::kHeadDim, S::kQkvWidth>),
					dim3(rows, G::kKvHeads), (G::kHeads / G::kKvHeads) * 32u, 0, stream,
					*view, state->attn_query, state->attn_local, step->sequence_of_row, step->context_length, step->positions,
					1.0f / sqrtf((float)G::kHeadDim), state->merged, rows);
				split_done = true;
			}
		}
		if ( !split_done )
		{
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kKvHeads,G::kHeadDim,G::kHeadDim>(*view, state->attn_query, G::kHeads,
					step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kHeadDim), state->send, stride, rows, stream,
					(S::kIndexer && !host_dense) ? state->index_mask : (const uint32_t *)0, state->index_mask_words, G::kIndexRatio, 0u,
					step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
					step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if constexpr ( G::kTpDegree > 1u )
			{
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
				if ( status != SPARK_STATUS_OK )
					return status;
				(void)LM_LAUNCH_DEPENDENT((QwenHybridMultiMergeGateKernel<G::kHeadDim, kQkvNoGateGather>), rows, QH_THREADS, 0, stream,
					state->received, stride, degree, G::kHeads, state->tp_rank,
					kQkvNoGateGather ? state->attn_local : state->merged_gate, state->merged, rows, kQkvNoGateGather ? state->attn_rows : 0u);
			}
			else
			{
				constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
				(void)LM_LAUNCH_DEPENDENT((QwenHybridTp1MergeGateKernel<G::kHeads, G::kHeadDim, S::kQkvWidth>),
					dim3((G::kHeads + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
					state->send, state->attn_local, state->merged, rows);
			}
		}
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, exact, sms,
				stream, prefill_wave) != LM_LAUNCH_OK )
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
		const bool prefill_wave = PrefillWave(step);
		const uint16_t *normed = state->normed + (uint64_t)first * G::kHidden;
		using ExpertFormat = typename LmWeightCodec<G::kExpertCodec>::Format;
		const LmScaleTensor w2_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w2_scale, G::kLocalExperts, G::kHidden, G::kMoeInter);
		const LmLocalExpertRoutes expert_routes = ExpertRoutes(state, rows, routes, sms);
		const bool multi_shared_done = !prefill_wave && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u);
		int32_t launch;
		if ( multi_shared_done )
		{
			const LmSkinnyDenseTarget targets[3] =
			{
				{ layer->moe_router, state->router_logits, (float *)0, G::kExperts, G::kExperts, 0u, LmScaleTensorNone() },
				{ layer->moe_shared_gate_up, state->shared_gate_up, (float *)0, 2u * G::kSharedRows, 2u * G::kSharedRows, 0u, LmScaleTensorNone() },
				{ layer->moe_shared_gate, state->shared_logit, (float *)0, 1u, 1u, 0u, LmScaleTensorNone() }
			};
			if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 3u, normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Project<LmBf16Format>(layer->moe_router, 0, normed, state->router_logits, rows, G::kHidden, G::kExperts, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
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
			const uint32_t local_mean = (routes + G::kExperts - 1u) / G::kExperts;
			const bool use_mma = prefill_wave || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE && (G::kHidden >= 4096u || local_mean > 2u));
			if ( use_mma )
			{
				const uint32_t up_blocks = G::kLocalExperts * ((G::kMoeInter + 63u) / 64u);
				const uint32_t dn_blocks = G::kLocalExperts * ((G::kHidden + 127u) / 128u);
				if ( rows <= 128u )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<1u, 8u, false, true, 4u>), up_blocks, 256u, 0, stream,
						layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight, state->expert_act,
						state->group_row_offset, state->route_source_token, G::kLocalExperts, G::kHidden, G::kMoeInter);
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<1u, 8u, true, false, 4u>), dn_blocks, 256u, 0, stream,
						layer->moe_w2, layer->moe_w2_scale, (const uint8_t *)0, (const uint8_t *)0, state->expert_act, (const float *)0,
						state->expert_out, state->group_row_offset, (const uint32_t *)0, G::kLocalExperts, G::kMoeInter, G::kHidden);
				}
				else if ( rows < 512u )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<2u, 8u, false, true, 4u>), up_blocks, 256u, 0, stream,
						layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight, state->expert_act,
						state->group_row_offset, state->route_source_token, G::kLocalExperts, G::kHidden, G::kMoeInter);
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<2u, 8u, true, false, 4u>), dn_blocks, 256u, 0, stream,
						layer->moe_w2, layer->moe_w2_scale, (const uint8_t *)0, (const uint8_t *)0, state->expert_act, (const float *)0,
						state->expert_out, state->group_row_offset, (const uint32_t *)0, G::kLocalExperts, G::kMoeInter, G::kHidden);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<3u, 8u, false, true, 2u>), up_blocks, 256u, 0, stream,
						layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight, state->expert_act,
						state->group_row_offset, state->route_source_token, G::kLocalExperts, G::kHidden, G::kMoeInter);
					(void)LM_LAUNCH_DEPENDENT((QwenNvfp4ExpertsKernel<3u, 8u, true, false, 2u>), dn_blocks, 256u, 0, stream,
						layer->moe_w2, layer->moe_w2_scale, (const uint8_t *)0, (const uint8_t *)0, state->expert_act, (const float *)0,
						state->expert_out, state->group_row_offset, (const uint32_t *)0, G::kLocalExperts, G::kMoeInter, G::kHidden);
				}
			}
			else if ( rows == 1u )
			{
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW1W3SingleKernel<G::kHidden, G::kMoeInter, G::kLocalExperts>),
					dim3((G::kMoeInter + UP_PER_BLK - 1u) / UP_PER_BLK, routes), QH_THREADS, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight,
					state->group_row_offset, state->packed_expert, state->expert_act);
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW2SingleKernel<G::kMoeInter, G::kHidden, G::kLocalExperts>),
					dim3((G::kHidden + DN_SINGLE_PER_BLK - 1u) / DN_SINGLE_PER_BLK, routes), QH_THREADS, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, state->expert_act, state->group_row_offset, state->packed_expert, state->expert_out);
			}
			else
			{
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW1W3BatchedKernel<G::kHidden, G::kMoeInter, G::kLocalExperts>),
					dim3((G::kMoeInter + UP_PER_BLK - 1u) / UP_PER_BLK, G::kLocalExperts), QH_THREADS, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight,
					state->group_row_offset, state->route_source_token, state->expert_act);
				(void)LM_LAUNCH_DEPENDENT((QwenNvfp4SkinnyW2BatchedKernel<G::kMoeInter, G::kHidden, G::kLocalExperts>),
					dim3((G::kHidden + DN_SINGLE_PER_BLK - 1u) / DN_SINGLE_PER_BLK, G::kLocalExperts), QH_THREADS, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, state->expert_act, state->group_row_offset, state->expert_out);
			}
		}
		else if constexpr ( G::kExpertCodec == SPARK_WEIGHT_CODEC_BF16 && (G::kHidden & 63u) == 0u && (G::kMoeInter & 63u) == 0u && G::kLocalExperts <= 256u )
		{
			const uint32_t local_mean = (routes + G::kExperts - 1u) / G::kExperts;
			const bool use_mma = prefill_wave || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE);
			const uint16_t *w1_bf16 = (const uint16_t *)layer->moe_w1;
			const uint16_t *w3_bf16 = (const uint16_t *)layer->moe_w3;
			const uint16_t *w2_bf16 = (const uint16_t *)layer->moe_w2;
			if ( use_mma )
			{
				if ( local_mean <= 16u )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeMmaW1W3ActKernel<16u, 64u, 128u, 1u>), sms * 4u, 128u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->route_source_token,
						state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter);
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeMmaW2Kernel<16u, 64u, 128u, 1u, 3u>), sms * 3u, 128u, 0, stream,
						w2_bf16, state->expert_act, state->expert_out, state->group_row_offset, G::kLocalExperts, G::kMoeInter, G::kHidden);
				}
				else if ( local_mean <= 32u )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeMmaW1W3ActKernel<32u, 64u, 256u, 2u>), sms * 2u, 256u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->route_source_token,
						state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter);
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeMmaW2Kernel<32u, 64u, 256u, 2u, 3u>), sms * 2u, 256u, 0, stream,
						w2_bf16, state->expert_act, state->expert_out, state->group_row_offset, G::kLocalExperts, G::kMoeInter, G::kHidden);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeMmaW1W3ActKernel<64u, 64u, 256u, 2u>), sms * 2u, 256u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->route_source_token,
						state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter);
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeMmaW2Kernel<64u, 128u, 256u, 2u, 2u>), sms * 2u, 256u, 0, stream,
						w2_bf16, state->expert_act, state->expert_out, state->group_row_offset, G::kLocalExperts, G::kMoeInter, G::kHidden);
				}
			}
			else
			{
				constexpr uint32_t W1_BLOCKS = (G::kMoeInter + 3u) / 4u;
				constexpr uint32_t W2_CHUNKS = G::kMoeInter >> 3u;
				constexpr uint32_t W2_LANES = (W2_CHUNKS >= 32u && (W2_CHUNKS % 32u) == 0u) ? 32u : (W2_CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t W2_GROUPS = 128u / W2_LANES;
				constexpr uint32_t W2_BLOCKS = (G::kHidden + W2_GROUPS - 1u) / W2_GROUPS;
				if ( rows == 1u )
				{
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeSkinnyW1W3SingleKernel<G::kHidden, G::kMoeInter, G::kLocalExperts>),
						dim3(W1_BLOCKS, routes), 128u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->packed_expert, state->expert_act);
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeSkinnyW2SingleKernel<G::kMoeInter, G::kHidden, G::kLocalExperts>),
						dim3(W2_BLOCKS, routes), 128u, 0, stream,
						w2_bf16, state->expert_act, state->group_row_offset, state->packed_expert, state->expert_out);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeSkinnyW1W3BatchedKernel<G::kHidden, G::kMoeInter, G::kLocalExperts>),
						dim3(W1_BLOCKS, G::kLocalExperts), 128u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->route_source_token, state->expert_act);
					(void)LM_LAUNCH_DEPENDENT((QwenBf16MoeSkinnyW2BatchedKernel<G::kMoeInter, G::kHidden, G::kLocalExperts>),
						dim3(W2_BLOCKS, G::kLocalExperts), 128u, 0, stream,
						w2_bf16, state->expert_act, state->group_row_offset, state->expert_out);
				}
			}
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			const bool use_stream = prefill_wave || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE);
			const LmScaleTensor w1_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w1_scale, G::kLocalExperts, G::kMoeInter, G::kHidden);
			const LmScaleTensor w3_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w3_scale, G::kLocalExperts, G::kMoeInter, G::kHidden);
			if ( use_stream )
			{
				launch = LmStreamGemmGrouped<ExpertFormat>(layer->moe_w1, w1_scale, normed, state->expert_gate, state->group_row_offset,
					state->route_source_token, G::kLocalExperts, routes, G::kHidden, G::kMoeInter, sms, stream);
				if ( launch == LM_LAUNCH_OK )
					launch = LmStreamGemmGrouped<ExpertFormat>(layer->moe_w3, w3_scale, normed, state->expert_up, state->group_row_offset,
						state->route_source_token, G::kLocalExperts, routes, G::kHidden, G::kMoeInter, sms, stream);
			}
			else
			{
				launch = LmLocalExpertsUp<ExpertFormat>(expert_routes, layer->moe_w1, w1_scale, normed, state->expert_gate, G::kHidden, G::kMoeInter, stream, exact);
				if ( launch == LM_LAUNCH_OK )
					launch = LmLocalExpertsUp<ExpertFormat>(expert_routes, layer->moe_w3, w3_scale, normed, state->expert_up, G::kHidden, G::kMoeInter, stream, exact);
			}
			if ( launch != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((QwenMoeSwigluPackedKernel), dim3(LmGridBlocks<QH_THREADS>(G::kMoeInter), routes), QH_THREADS, 0, stream,
				state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter);
			if ( use_stream )
			{
				if ( LmStreamGemmGrouped<ExpertFormat>(layer->moe_w2, w2_scale, state->expert_act, state->expert_out, state->group_row_offset,
					(const uint32_t *)0, G::kLocalExperts, routes, G::kMoeInter, G::kHidden, sms, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( LmLocalExpertsDown<ExpertFormat>(expert_routes, layer->moe_w2, w2_scale, state->expert_act, state->expert_out, G::kMoeInter, G::kHidden,
				stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		bool shared_act_ready = false;
		if ( !multi_shared_done )
		{
			if constexpr ( G::kSharedRows <= 256u && (G::kHidden % 256u) == 0u )
			{
				const int32_t sh_st = rows > 16u ?
					LaunchSharedGateUpSwigluTile<32u>(layer->moe_shared_gate_up, normed, state->shared_act, rows, G::kHidden, G::kSharedRows, stream) :
					LaunchSharedGateUpSwigluTile<16u>(layer->moe_shared_gate_up, normed, state->shared_act, rows, G::kHidden, G::kSharedRows, stream);
				if ( sh_st != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				shared_act_ready = true;
			}
			else if ( Project<LmBf16Format>(layer->moe_shared_gate_up, 0, normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, exact, sms, stream, prefill_wave) !=
				LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((QwenHybridRowDotKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, normed, layer->moe_shared_gate, state->shared_logit, G::kHidden);
		}
		if constexpr ( G::kSharedRows <= 128u && (G::kSharedRows & 7u) == 0u && G::kTopK <= LM_MOE_LOCAL_MAX_TOPK )
		{
			if ( multi_shared_done )
			{
				constexpr uint32_t S_CHUNKS = G::kSharedRows >> 3u;
				constexpr uint32_t S_LANES = S_CHUNKS >= 32u ? 32u : (S_CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t S_GROUPS = 128u / S_LANES;
				constexpr uint32_t S_BLOCKS = (G::kHidden + S_GROUPS - 1u) / S_GROUPS;
				(void)LM_LAUNCH_DEPENDENT((QwenMoeSharedDownFinalizeKernel<G::kHidden, G::kSharedRows, G::kTopK, G::kLocalExperts>),
					dim3(S_BLOCKS, rows), 128u, 0, stream, state->expert_out, state->route_packed_row, state->route_expert,
					state->shared_gate_up, layer->moe_shared_down, state->shared_logit, state->partial + (uint64_t)first * G::kHidden, rows);
				return CudaStageLaunched();
			}
		}
		if ( !shared_act_ready )
			(void)LM_LAUNCH_DEPENDENT((QwenSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Project<LmBf16Format>(layer->moe_shared_down, 0, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, exact, sms, stream, prefill_wave) !=
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
			const bool prefill_wave = PrefillWave(step);
			bool fused_done = false;
			if ( state->linear_bytes == 2u && G::kHidden >= 256u && (G::kHidden & 7u) == 0u && (state->ffn_rows & 1u) == 0u )
			{
				const int32_t status = LaunchFusedGateUpSwiglu((const uint16_t *)layer->ffn_gate_up, state->normed, state->intermediate,
					rows, G::kHidden, state->ffn_rows, exact, prefill_wave, stream);
				if ( status == LM_LAUNCH_OK )
					fused_done = true;
				else if ( status != LM_LAUNCH_ERR_SHAPE )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			if ( !fused_done )
			{
				if ( Linear(state, layer->ffn_gate_up, layer->ffn_gate_up_scale, state->normed, state->gate_up, rows, G::kHidden,
					2u * state->ffn_rows, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				(void)LM_LAUNCH_DEPENDENT((QwenSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows);
			}
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				Linear(state, layer->ffn_down, layer->ffn_down_scale, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, exact, sms,
					stream, prefill_wave) != LM_LAUNCH_OK )
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
		uint16_t *mixed, uint32_t rows, uint32_t exact, uint32_t sms, SparkGpuStream stream, bool prefill_unified = false)
	{
		if constexpr ( S::kHyper )
		{
			const bool skinny = !prefill_unified && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u);
			if ( state->hc_pending_inject != 0u )
			{
				state->hc_pending_inject = 0u;
				(void)LM_LAUNCH_DEPENDENT((QwenHybridHcInjectNormKernel<QH_THREADS, G::kHidden>), dim3(G::kStreams, rows), QH_THREADS, 0, stream,
					state->residual, state->partial, state->hc_inject, norm, state->hc_normed, G::kStreams);
			}
			else
				LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->residual, norm, state->hc_normed,
					G::kStreams, G::kHidden, QH_EPSILON);
			if ( skinny )
			{
				constexpr uint32_t HC_THREADS = S::kResidual >= 2048u ? 256u : (S::kResidual >= 1024u ? 128u : 32u);
				if ( inject != 0 )
				{
					constexpr uint32_t N_BLOCKS = G::kHcLowRank + G::kStreams;
					if ( rows == 1u )
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 1u, true>), N_BLOCKS, HC_THREADS, 0, stream,
							down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
					else if ( rows <= 4u )
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 4u, true>), N_BLOCKS, HC_THREADS, 0, stream,
							down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
					else
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 8u, true>),
							N_BLOCKS * ((rows + 7u) / 8u), HC_THREADS, 0, stream, down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
				}
				else
				{
					constexpr uint32_t N_BLOCKS = G::kHcLowRank;
					if ( rows == 1u )
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 1u, false>), N_BLOCKS, HC_THREADS, 0, stream,
							down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
					else if ( rows <= 4u )
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 4u, false>), N_BLOCKS, HC_THREADS, 0, stream,
							down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
					else
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectKernel<G::kStreams, G::kHidden, G::kHcLowRank, 8u, false>),
							N_BLOCKS * ((rows + 7u) / 8u), HC_THREADS, 0, stream, down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
				}
				constexpr uint32_t CHUNKS = G::kHcLowRank >> 3u;
				constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t GROUPS = QH_THREADS / LANES;
				constexpr uint32_t COL_BLOCKS = (G::kHidden + GROUPS - 1u) / GROUPS;
				if ( rows == 1u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixKernel<G::kStreams, G::kHidden, G::kHcLowRank, 1u>), COL_BLOCKS, QH_THREADS, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixKernel<G::kStreams, G::kHidden, G::kHcLowRank, 4u>),
						COL_BLOCKS * ((rows + 3u) / 4u), QH_THREADS, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
			}
			else if constexpr ( (S::kResidual % 256u) == 0u && (G::kHcLowRank % 64u) == 0u && (G::kHidden % 64u) == 0u )
			{
				if ( rows <= 160u )
				{
					const uint32_t m32 = (rows + 31u) / 32u;
					if ( inject != 0 )
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + G::kStreams + 31u) / 32u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, true, 32u, 32u, 128u, 2u, 3u>),
							m32 * n_blocks, 128u, 0, stream, down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
					}
					else
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + 31u) / 32u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, false, 32u, 32u, 128u, 2u, 3u>),
							m32 * n_blocks, 128u, 0, stream, down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
					}
				}
				else
				{
					const uint32_t m64 = (rows + 63u) / 64u;
					if ( inject != 0 )
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + G::kStreams + 63u) / 64u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, true, 64u, 64u, 128u, 2u, 3u>),
							m64 * n_blocks, 128u, 0, stream, down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
					}
					else
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + 63u) / 64u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, false, 64u, 64u, 128u, 2u, 3u>),
							m64 * n_blocks, 128u, 0, stream, down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
					}
				}
				if ( rows >= 128u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 128u, 256u, 4u, 2u>),
						((rows + 127u) / 128u) * (G::kHidden / 64u), 256u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
				else if ( rows > 32u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 64u, 128u, 2u, 3u>),
						((rows + 63u) / 64u) * (G::kHidden / 64u), 128u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
				else if ( rows > 16u )
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 32u, 128u, 2u, 3u>),
						((rows + 31u) / 32u) * (G::kHidden / 64u), 128u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
				else
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 16u, 128u, 1u, 3u>),
						((rows + 15u) / 16u) * (G::kHidden / 64u), 128u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
			}
			else if constexpr ( (S::kResidual % 64u) == 0u && (G::kHcLowRank % 64u) == 0u && (G::kHidden % 64u) == 0u )
			{
				if ( rows > 32u )
				{
					const uint32_t m64 = (rows + 63u) / 64u;
					if ( inject != 0 )
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + G::kStreams + 63u) / 64u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, true, 64u, 64u, 128u, 2u, 3u>),
							m64 * n_blocks, 128u, 0, stream, down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
					}
					else
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + 63u) / 64u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, false, 64u, 64u, 128u, 2u, 3u>),
							m64 * n_blocks, 128u, 0, stream, down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
					}
					if ( rows >= 128u )
					{
						const uint32_t m128 = (rows + 127u) / 128u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 128u, 256u, 4u, 2u>),
							m128 * (G::kHidden / 64u), 256u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
					}
					else
						(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 64u, 128u, 2u, 3u>),
							m64 * (G::kHidden / 64u), 128u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
				}
				else
				{
					const uint32_t m32 = (rows + 31u) / 32u;
					if ( inject != 0 )
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + G::kStreams + 31u) / 32u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, true, 32u, 32u, 128u, 2u, 3u>),
							m32 * n_blocks, 128u, 0, stream, down, inject, state->hc_normed, state->hc_low, state->hc_inject, rows);
					}
					else
					{
						constexpr uint32_t n_blocks = (G::kHcLowRank + 31u) / 32u;
						(void)LM_LAUNCH_DEPENDENT((QwenHcDownGateInjectMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, false, 32u, 32u, 128u, 2u, 3u>),
							m32 * n_blocks, 128u, 0, stream, down, (const uint16_t *)0, state->hc_normed, state->hc_low, (float *)0, rows);
					}
					(void)LM_LAUNCH_DEPENDENT((QwenHcUpMixMmaKernel<G::kStreams, G::kHidden, G::kHcLowRank, 32u, 128u, 2u, 3u>),
						m32 * (G::kHidden / 64u), 128u, 0, stream, up, state->hc_low, state->hc_normed, mixed, rows);
				}
			}
			else
			{
				if ( Project<LmBf16Format>(down, 0, state->hc_normed, state->hc_low, rows, S::kResidual, G::kHcLowRank, exact, sms, stream, prefill_unified) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				LM_LAUNCH((LmHyperGateKernel<QH_THREADS>), LmGridBlocks<QH_THREADS>((uint64_t)rows * G::kHcLowRank), QH_THREADS, 0, stream, state->hc_low,
					(uint64_t)rows * G::kHcLowRank, (float)G::kStreams);
				if ( Project<LmBf16Format>(up, 0, state->hc_low, state->hc_up, rows, G::kHcLowRank, S::kResidual, exact, sms, stream, prefill_unified) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				(void)LM_LAUNCH_DEPENDENT((QwenHyperMixKernel<QH_THREADS, G::kStreams, G::kHidden>), rows, QH_THREADS, 0, stream,
					state->hc_up, state->hc_normed, mixed);
				if ( inject != 0 )
					(void)LM_LAUNCH_DEPENDENT((QwenHyperInjectWeightsVecKernel<QH_THREADS, G::kStreams>), rows, QH_THREADS, 0, stream,
						state->hc_normed, inject, state->hc_inject, G::kHidden);
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
		if ( !PrefillWave(step) && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			if ( rows >= 2u && rows <= 32u && G::kHidden >= 256u && (G::kHidden & 7u) == 0u )
			{
				QwenBf16MultiArgs args;
				args.part[0] = { state->ple_key_weight, state->ple_key, S::kResidual, 0u };
				args.part[1] = { state->ple_value_weight, state->ple_value, G::kHidden, 0u };
				args.count = 2u;
				args.activation = state->ple_embed;
				args.rows = rows;
				args.input_dim = G::kHidden;
				if ( LaunchExactBf16Multi(args, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				const LmSkinnyDenseTarget targets[2] =
				{
					{ state->ple_key_weight, state->ple_key, (float *)0, S::kResidual, S::kResidual, 0u, LmScaleTensorNone() },
					{ state->ple_value_weight, state->ple_value, (float *)0, G::kHidden, G::kHidden, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->ple_embed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else if ( Project<LmBf16Format>(state->ple_key_weight, 0, state->ple_embed, state->ple_key, rows, G::kHidden, S::kResidual, exact, sms, stream, PrefillWave(step)) !=
				LM_LAUNCH_OK ||
			Project<LmBf16Format>(state->ple_value_weight, 0, state->ple_embed, state->ple_value, rows, G::kHidden, G::kHidden, exact, sms, stream, PrefillWave(step)) !=
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
		if constexpr ( G::kTpDegree == 1u )
			return SPARK_STATUS_OK;
		else
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
			Exact(frame.step), frame.step->multiprocessors, frame.stream, PrefillWave(frame.step));
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
			frame.step->rows, Exact(frame.step), frame.step->multiprocessors, frame.stream, PrefillWave(frame.step));
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, slots = lane == 0u ? state->slots : 0u, routes = rows * (G::kMoe ? G::kTopK : 0u);
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t), replay = lane == 0u ? G::kVerifyRows : 0u;
		constexpr bool kFusedPle = S::kPle && (G::kHidden & 7u) == 0u && G::kHidden <= QH_THREADS * LM_RMS_NORM_STAGED;
		const uint64_t send_floats = (uint64_t)state->tp_degree * state->record_stride > state->split_gqa_floats ?
			(uint64_t)state->tp_degree * state->record_stride : state->split_gqa_floats;
		const CudaStageSlice persistent[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->residual, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_inject, S::kHyper ? rows * G::kStreams * sizeof(float) : 0u },
			{ (void **)&state->row_lane, S::kPle || S::kIndexer ? rows * sizeof(uint32_t) : 0u },
			{ (void **)&state->row_ordinal, S::kPle || S::kIndexer ? rows * sizeof(uint32_t) : 0u },
			{ (void **)&state->gdn_state, (uint64_t)S::kGdnLayers * slots * state->gdn_state_bytes },
			{ (void **)&state->gdn_window, (uint64_t)S::kGdnLayers * slots * state->gdn_window_bytes },
			{ (void **)&state->gdn_snapshots, replay != 0u && kTreeVerify ? LmDeltaRuleSnapshotBytes(state->gdn_heads, QH_KEY_DIM, QH_VALUE_DIM) : 0u },
			{ (void **)&state->replay_qkv, (uint64_t)S::kGdnLayers * replay * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->replay_ba, (uint64_t)S::kGdnLayers * replay * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->replay_ple, S::kPle ? replay * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->replay_index, (uint64_t)S::kAttnLayers * replay * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->ple_window, slots * state->ple_window_bytes },
			{ (void **)&state->ple_history, S::kPle ? slots * S::kPleHistory * sizeof(int32_t) : 0u },
			{ (void **)&state->index_tail, (uint64_t)S::kAttnLayers * slots * S::kIndexTail * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		uint64_t fixed = CudaStagePlace(persistent, base);
		fixed = (fixed + CUDA_STAGE_SLICE_ALIGNMENT - 1u) & ~(uint64_t)(CUDA_STAGE_SLICE_ALIGNMENT - 1u);
		uint8_t *scratch_base = base != 0 ? base + fixed : 0;
		uint64_t scratch = 0u;
		auto overlay = [&](uint64_t bytes) { if ( bytes > scratch ) scratch = bytes; };
		if constexpr ( G::kMoe )
		{
			constexpr bool kFusedNvfp4 = G::kExpertCodec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 &&
				(G::kHidden % 128u) == 0u && (G::kMoeInter % 128u) == 0u;
			constexpr bool kWarpTopk = G::kExperts <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES;
			const CudaStageSlice ffn_moe[] =
			{
				{ (void **)&state->router_logits, rows * G::kExperts * sizeof(uint16_t) },
				{ (void **)&state->router_probs, rows * (!kWarpTopk ? G::kExperts : 0u) * sizeof(float) },
				{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
				{ (void **)&state->route_global, routes * sizeof(uint32_t) },
				{ (void **)&state->route_weight, routes * sizeof(float) },
				{ (void **)&state->group_row_offset, (G::kLocalExperts + 2u) * sizeof(uint32_t) },
				{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
				{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
				{ (void **)&state->packed_expert, routes * sizeof(uint32_t) },
				{ (void **)&state->tile_prefix_up, (G::kLocalExperts + 2u) * sizeof(uint32_t) },
				{ (void **)&state->tile_prefix_down, (G::kLocalExperts + 2u) * sizeof(uint32_t) },
				{ (void **)&state->packed_weight, routes * sizeof(float) },
				{ (void **)&state->expert_gate, routes * (!kFusedNvfp4 ? G::kMoeInter : 0u) * sizeof(uint16_t) },
				{ (void **)&state->expert_up, routes * (!kFusedNvfp4 ? G::kMoeInter : 0u) * sizeof(uint16_t) },
				{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
				{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
				{ (void **)&state->shared_gate_up, rows * 2u * G::kSharedRows * sizeof(uint16_t) },
				{ (void **)&state->shared_act, rows * G::kSharedRows * sizeof(uint16_t) },
				{ (void **)&state->shared_out, width },
				{ (void **)&state->shared_logit, rows * sizeof(uint16_t) },
			};
			overlay(CudaStagePlace(ffn_moe, scratch_base));
		}
		else
		{
			const CudaStageSlice ffn_dense[] =
			{
				{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
				{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			};
			overlay(CudaStagePlace(ffn_dense, scratch_base));
		}
		const CudaStageSlice gdn_qkv[] =
		{
			{ (void **)&state->qkv, rows * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->z, rows * state->gdn_v * sizeof(uint16_t) },
		};
		const CudaStageSlice gdn_qkvz[] =
		{
			{ (void **)&state->gdn_qkvz, rows * (state->gdn_channels + state->gdn_v) * sizeof(uint16_t) },
		};
		const uint64_t gdn_a = CudaStagePlace(gdn_qkv, scratch_base);
		const uint64_t gdn_b = CudaStagePlace(gdn_qkvz, scratch_base);
		uint64_t gdn_head = gdn_a > gdn_b ? gdn_a : gdn_b;
		gdn_head = (gdn_head + CUDA_STAGE_SLICE_ALIGNMENT - 1u) & ~(uint64_t)(CUDA_STAGE_SLICE_ALIGNMENT - 1u);
		const CudaStageSlice gdn_rest[] =
		{
			{ (void **)&state->ba, rows * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->value, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->query_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->key_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->retention, rows * state->gdn_heads * QH_KEY_DIM * sizeof(float) },
			{ (void **)&state->write_gate, rows * state->gdn_heads * sizeof(float) },
			{ (void **)&state->mixed, rows * state->gdn_v * sizeof(uint16_t) },
		};
		overlay(gdn_head + CudaStagePlace(gdn_rest, scratch_base != 0 ? scratch_base + gdn_head : 0));
		const CudaStageSlice attn_group[] =
		{
			{ (void **)&state->attn_local, rows * state->attn_rows * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, G::kTpDegree > 1u ? rows * S::kQkvWidth * sizeof(uint16_t) : 0u },
			{ (void **)&state->attn_query, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->send, send_floats * sizeof(float) },
			{ (void **)&state->received, G::kTpDegree > 1u ? (uint64_t)state->tp_degree * state->record_stride * sizeof(float) : 0u },
			{ (void **)&state->merged, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->merged_gate, G::kTpDegree > 1u ? rows * state->attn_width * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_qk, S::kIndexer ? rows * S::kIndexWidth * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_query, S::kIndexer ? rows * G::kIndexHeads * G::kIndexDim * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_key, S::kIndexer ? rows * G::kIndexDim * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_scores, S::kIndexer ? rows * state->index_local_stride * sizeof(float) : 0u },
			{ (void **)&state->index_local_selected, S::kIndexer ? rows * S::kIndexBlocks * sizeof(uint32_t) : 0u },
			{ (void **)&state->index_candidates, S::kIndexer ? rows * S::kIndexBlocks * sizeof(uint2) : 0u },
			{ (void **)&state->index_gathered, S::kIndexer ? (uint64_t)state->tp_degree * rows * S::kIndexBlocks * sizeof(uint2) : 0u },
			{ (void **)&state->index_mask, S::kIndexer ? rows * state->index_mask_words * sizeof(uint32_t) : 0u },
		};
		overlay(CudaStagePlace(attn_group, scratch_base));
		const CudaStageSlice hc_ple[] =
		{
			{ (void **)&state->hc_normed, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_low, S::kHyper ? rows * G::kHcLowRank * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_up, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_ids, S::kPle ? rows * S::kPleColumns * sizeof(int64_t) : 0u },
			{ (void **)&state->ple_embed, S::kPle ? width : 0u },
			{ (void **)&state->ple_key, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_query, S::kPle && !kFusedPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_value, S::kPle ? width : 0u },
			{ (void **)&state->ple_gated, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_conv_in, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_conv_out, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
		};
		overlay(CudaStagePlace(hc_ple, scratch_base));
		if ( base != 0 && G::kTpDegree == 1u )
		{
			state->attn_gathered = state->attn_local;
			state->received = state->send;
			state->merged_gate = 0;
		}
		return fixed + scratch;
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
		if constexpr ( G::kTpDegree == 1u && !S::kIndexer && G::kHeadDim == 256u && (G::kHeads % G::kKvHeads) == 0u )
		{
			const uint32_t split_rows = state->max_rows < 64u ? state->max_rows : 64u;
			state->split_gqa_floats = (uint64_t)split_rows * QH_GQA_MAX_SPLITS * G::kHeads * (G::kHeadDim + 4u);
		}
		else
			state->split_gqa_floats = 0u;
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
		geometry->graph_ignores_gather = 0u;
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
