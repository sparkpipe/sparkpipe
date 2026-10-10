#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/hyper_mix.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
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

#define QH_KEY_DIM 128u
#define QH_VALUE_DIM 128u
#define QH_CONV 4u
#define QH_EPSILON 1e-6f
#define QH_THREADS 256u
#define QH_SCALE_K 128u
#define QH_PACK_ALIGNMENT 128u
#define QH_GEMM_TILE_N 128u
#define QH_GEMM_STAGES 2u
#define QH_GEMM_WARPS 8u

struct QwenHybridDefaults
{
	static constexpr uint32_t kStreams = 1u;
	static constexpr uint32_t kHcLowRank = 0u;
	static constexpr uint32_t kPleLayer = 0xffffffffu;
	static constexpr uint32_t kPleOrders = 0u;
	static constexpr uint32_t kPleHeadsPerOrder = 0u;
	static constexpr uint32_t kPleHeadDim = 0u;
	static constexpr uint32_t kPleConv = 1u;
	static constexpr uint32_t kPleDilation = 1u;
	static constexpr uint32_t kPleEos = 0u;
	static constexpr bool kGdnSigmoidGate = false;
	static constexpr uint32_t kIndexHeads = 0u;
	static constexpr uint32_t kIndexDim = 0u;
	static constexpr uint32_t kIndexBudget = 0u;
	static constexpr uint32_t kIndexRatio = 1u;
};

#define QH_INDEX_CHUNK 64u

template<class G>
struct QwenHybridShape
{
	static constexpr uint32_t kGdnLayers = G::kLayers - G::kLayers / G::kPeriod;
	static constexpr uint32_t kAttnLayers = G::kLayers / G::kPeriod;
	static constexpr uint32_t kQWidth = G::kHeads * G::kHeadDim;
	static constexpr uint32_t kKvWidth = G::kKvHeads * G::kHeadDim;
	static constexpr uint32_t kQkvWidth = 2u * kQWidth + 2u * kKvWidth;
	static constexpr bool kHyper = G::kStreams > 1u;
	static constexpr uint32_t kResidual = G::kStreams * G::kHidden;
	static constexpr bool kPle = G::kPleLayer < G::kLayers;
	static constexpr uint32_t kPleColumns = G::kPleOrders * G::kPleHeadsPerOrder;
	static constexpr uint32_t kPleSpan = (G::kPleConv - 1u) * G::kPleDilation + 1u;
	static constexpr uint32_t kPleHistory = 4u;
	static constexpr bool kIndexer = G::kIndexHeads > 0u;
	static constexpr uint32_t kIndexWidth = (G::kIndexHeads + 1u) * G::kIndexDim;
	static constexpr uint32_t kIndexBlocks = G::kIndexBudget / G::kIndexRatio;
	static constexpr uint32_t kIndexTail = G::kIndexRatio - 1u;
	using Kv = LmKvHeads<16u, G::kKvHeads, G::kHeadDim, G::kPageSlots>;
	using IndexKv = LmKvGeometry<kIndexer ? G::kIndexDim * 2u : 16u, G::kPageSlots / G::kIndexRatio, true>;
};

typedef struct QwenHybridLayer
{
	const uint8_t *mix_weight;
	const float *mix_scale;
	const uint8_t *out_weight;
	const float *out_scale;
	const uint16_t *gdn_ba;
	const uint16_t *gdn_conv;
	const uint16_t *gdn_norm;
	const float *gdn_a_log;
	const float *gdn_dt_bias;
	const float *q_norm;
	const float *k_norm;
	const uint8_t *ffn_gate_up;
	const float *ffn_gate_up_scale;
	const uint8_t *ffn_down;
	const float *ffn_down_scale;
	const uint16_t *moe_router;
	const uint8_t *moe_w1;
	const uint8_t *moe_w1_scale;
	const uint8_t *moe_w3;
	const uint8_t *moe_w3_scale;
	const uint8_t *moe_w2;
	const uint8_t *moe_w2_scale;
	const uint16_t *moe_shared_gate_up;
	const uint16_t *moe_shared_down;
	const uint16_t *moe_shared_gate;
	const float *input_norm;
	const float *post_norm;
	const float *hc_norm[2];
	const uint16_t *hc_down[2];
	const uint16_t *hc_up[2];
	const uint16_t *hc_inject[2];
	const uint16_t *index_qk;
	const float *index_q_norm;
	const float *index_k_norm;
	uint32_t recurrent_index;
	uint32_t kv_index;
} QwenHybridLayer;

template<class G>
struct QwenHybridState
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t linear_bytes;
	const char *linear_kind;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint32_t gdn_heads;
	uint32_t gdn_qk;
	uint32_t gdn_v;
	uint32_t gdn_channels;
	uint32_t attn_rows;
	uint32_t attn_width;
	uint64_t gdn_state_bytes;
	uint64_t gdn_window_bytes;
	uint64_t record_stride;
	QwenHybridLayer layers[G::kLayers];
	FILE *dump;
	uint16_t *dump_rows;
	uint32_t *dump_positions;
	void *arena;
	uint64_t arena_bytes;
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *gate_up;
	uint16_t *intermediate;
	uint16_t *qkv;
	uint16_t *z;
	uint16_t *ba;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	uint16_t *query_heads;
	uint16_t *key_heads;
	float *retention;
	float *write_gate;
	uint16_t *mixed;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *attn_full;
	uint16_t *attn_query;
	uint16_t *attn_gate;
	uint16_t *attn_key;
	uint16_t *attn_value;
	uint16_t *attn_slot;
	float *send;
	float *received;
	uint16_t *merged;
	uint16_t *merged_gate;
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
	uint16_t *shared_gate_up;
	uint16_t *shared_act;
	uint16_t *shared_out;
	uint16_t *shared_logit;
	uint8_t *gdn_state;
	uint16_t *gdn_window;
	uint16_t *residual;
	uint16_t *hc_normed;
	uint16_t *hc_low;
	uint16_t *hc_up;
	float *hc_inject;
	const float *final_hc_norm;
	const uint16_t *final_hc_down;
	const uint16_t *final_hc_up;
	const uint16_t *ple_key_weight;
	const uint16_t *ple_value_weight;
	const float *ple_norm_key;
	const float *ple_norm_query;
	const float *ple_norm_conv;
	const uint16_t *ple_conv_weight;
	const uint8_t *ple_table;
	const float *ple_table_scale;
	const int64_t *ple_multipliers;
	const int64_t *ple_vocab_sizes;
	const int64_t *ple_offsets;
	uint32_t ple_table_first;
	uint32_t ple_table_rows;
	uint64_t ple_window_bytes;
	int64_t *ple_ids;
	uint16_t *ple_embed;
	uint16_t *ple_key;
	uint16_t *ple_query;
	uint16_t *ple_value;
	uint16_t *ple_gated;
	uint16_t *ple_conv_in;
	uint16_t *ple_conv_out;
	uint16_t *ple_window;
	int32_t *ple_history;
	uint32_t *row_lane;
	uint32_t *row_ordinal;
	uint32_t max_positions;
	uint32_t index_local_stride;
	uint32_t index_global_stride;
	uint32_t index_mask_words;
	uint64_t index_topk_entries;
	uint16_t *index_qk;
	uint16_t *index_query;
	uint16_t *index_key;
	uint16_t *index_block;
	uint16_t *index_tail;
	float *index_scores;
	uint32_t *index_local_selected;
	float *index_topk_values;
	uint32_t *index_topk_positions;
	uint2 *index_candidates;
	uint2 *index_gathered;
	float *index_full_scores;
	uint32_t *index_selected;
	uint32_t *index_mask;
	LmKvShardView index_views[QwenHybridShape<G>::kAttnLayers];
	LmKvAccessError *kv_error;
	LmKvShardView kv_views[QwenHybridShape<G>::kAttnLayers];
	uint32_t kv_attached;
};

__global__ static void QwenHybridAddRowsKernel(uint16_t *hidden, const uint16_t *addend, uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
}

__global__ static void QwenHybridGatherRowsKernel(const uint16_t *gathered, uint16_t *full, uint32_t rows, uint32_t width, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width * ranks; index += gridDim.x * blockDim.x )
	{
		const uint32_t rank = index / width, element = index % width;
		full[(uint64_t)row * width * ranks + index] = gathered[((uint64_t)rank * rows + row) * width + element];
	}
}

template<class G>
__global__ static void QwenHybridSplitAttentionKernel(const uint16_t *full, uint16_t *query, uint16_t *gate, uint16_t *key, uint16_t *value)
{
	using S = QwenHybridShape<G>;
	const uint32_t row = blockIdx.y;
	const uint16_t *source = full + (uint64_t)row * S::kQkvWidth;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < S::kQkvWidth; index += gridDim.x * blockDim.x )
	{
		uint32_t fused = index;
		if constexpr ( G::kRankHeads )
		{
			constexpr uint32_t per_rank = S::kQkvWidth / G::kTpDegree, q_rank = 2u * S::kQWidth / G::kTpDegree, kv_rank = S::kKvWidth / G::kTpDegree;
			const uint32_t rank = index / per_rank, element = index % per_rank;
			fused = element < q_rank ? rank * q_rank + element :
				element < q_rank + kv_rank ? 2u * S::kQWidth + rank * kv_rank + (element - q_rank) :
				2u * S::kQWidth + S::kKvWidth + rank * kv_rank + (element - q_rank - kv_rank);
		}
		if ( fused < 2u * S::kQWidth )
		{
			const uint32_t head = fused / (2u * G::kHeadDim), within = fused % (2u * G::kHeadDim);
			uint16_t *target = within < G::kHeadDim ? query : gate;
			target[(uint64_t)row * S::kQWidth + head * G::kHeadDim + within % G::kHeadDim] = source[index];
		}
		else if ( fused < 2u * S::kQWidth + S::kKvWidth )
			key[(uint64_t)row * S::kKvWidth + fused - 2u * S::kQWidth] = source[index];
		else
			value[(uint64_t)row * S::kKvWidth + fused - 2u * S::kQWidth - S::kKvWidth] = source[index];
	}
}

template<class G>
__global__ static void QwenHybridPackSlotKernel(const uint16_t *key, const uint16_t *value, uint16_t *slot)
{
	constexpr uint32_t width = QwenHybridShape<G>::kKvWidth;
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 2u * width; index += gridDim.x * blockDim.x )
		slot[(uint64_t)row * 2u * width + index] = index < width ? key[(uint64_t)row * width + index] : value[(uint64_t)row * width + index - width];
}

__global__ static void QwenHybridSliceRowsKernel(const uint16_t *source, uint16_t *target, uint32_t source_width, uint32_t offset, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		target[(uint64_t)row * width + index] = source[(uint64_t)row * source_width + offset + index];
}

template<uint32_t WIDTH>
__global__ static void QwenHybridSoftmaxRowsKernel(const uint16_t *logits, float *probs)
{
	__shared__ float reduce[QH_THREADS / 32u];
	const uint32_t row = blockIdx.x;
	float top = -INFINITY, sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		top = fmaxf(top, LmBf16ToFloat(logits[(uint64_t)row * WIDTH + index]));
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
	if ( threadIdx.x % 32u == 0u )
		reduce[threadIdx.x / 32u] = top;
	__syncthreads();
	top = reduce[0];
	for ( uint32_t warp = 1u; warp < blockDim.x / 32u; ++warp )
		top = fmaxf(top, reduce[warp]);
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		sum += expf(LmBf16ToFloat(logits[(uint64_t)row * WIDTH + index]) - top);
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu, sum, offset);
	if ( threadIdx.x % 32u == 0u )
		reduce[threadIdx.x / 32u] = sum;
	__syncthreads();
	sum = 0.0f;
	for ( uint32_t warp = 0u; warp < blockDim.x / 32u; ++warp )
		sum += reduce[warp];
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		probs[(uint64_t)row * WIDTH + index] = expf(LmBf16ToFloat(logits[(uint64_t)row * WIDTH + index]) - top) / sum;
}

__global__ static void QwenHybridRowDotKernel(const uint16_t *rows_bf16, const uint16_t *weight, uint16_t *output, uint32_t width)
{
	__shared__ float reduce[QH_THREADS / 32u];
	const uint32_t row = blockIdx.x;
	float sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += blockDim.x )
		sum += LmBf16ToFloat(rows_bf16[(uint64_t)row * width + index]) * LmBf16ToFloat(weight[index]);
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu, sum, offset);
	if ( threadIdx.x % 32u == 0u )
		reduce[threadIdx.x / 32u] = sum;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float total = 0.0f;
		for ( uint32_t warp = 0u; warp < blockDim.x / 32u; ++warp )
			total += reduce[warp];
		output[row] = LmFloatToBf16(total);
	}
}

__global__ static void QwenHybridSigmoidGateRowsKernel(uint16_t *rows_bf16, const uint16_t *logit, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	const float gate = 1.0f / (1.0f + expf(-LmBf16ToFloat(logit[row])));
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		rows_bf16[(uint64_t)row * width + index] = LmFloatToBf16(LmBf16ToFloat(rows_bf16[(uint64_t)row * width + index]) * gate);
}

template<class G>
__global__ static void QwenIndexSplitKernel(const uint16_t *qk, uint16_t *query, uint16_t *key)
{
	constexpr uint32_t width = QwenHybridShape<G>::kIndexWidth, query_width = G::kIndexHeads * G::kIndexDim;
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
	{
		const uint16_t value = qk[(uint64_t)row * width + index];
		if ( index < query_width )
			query[(uint64_t)row * query_width + index] = value;
		else
			key[(uint64_t)row * G::kIndexDim + index - query_width] = value;
	}
}

template<class G>
__global__ static void QwenIndexBlockKeysKernel(const uint16_t *raw, const uint16_t *tail, const float *norm, uint16_t *block,
	const uint32_t *positions, const uint32_t *row_lane, const uint32_t *row_ordinal, const uint32_t *sequence_row_begin,
	const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, float theta)
{
	constexpr uint32_t D = G::kIndexDim, R = G::kIndexRatio, HALF = G::kRopeDim / 2u;
	__shared__ float reduction[D / LM_WARP_LANES];
	__shared__ float normed[D];
	const uint32_t row = blockIdx.x, d = threadIdx.x, position = positions[row];
	if ( position % R != R - 1u )
		return;
	const uint32_t lane = row_lane[row], ordinal = row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = positions[lane_rows[0]], tail_base = first / R * R, start = position - (R - 1u);
	const uint16_t *carried = tail + (uint64_t)recurrent_slot[lane] * QwenHybridShape<G>::kIndexTail * D;
	float sum = 0.0f;
	for ( uint32_t k = 0u; k < R; ++k )
	{
		const uint32_t at = start + k;
		sum += LmBf16ToFloat(at >= first ? raw[(uint64_t)lane_rows[ordinal - (position - at)] * D + d] : carried[(uint64_t)(at - tail_base) * D + d]);
	}
	const float pooled = LmBf16ToFloat(LmFloatToBf16(sum / (float)R));
	const float scale = rsqrtf(LmBlockSum<D>(pooled * pooled, reduction) / (float)D + QH_EPSILON);
	normed[d] = LmBf16ToFloat(LmFloatToBf16(pooled * scale * norm[d]));
	__syncthreads();
	float out = normed[d];
	if ( d < G::kRopeDim )
	{
		const uint32_t pair = d % HALF;
		const float angle = (float)start * powf(theta, -2.0f * (float)pair / (float)G::kRopeDim);
		const float partner = d < HALF ? -normed[d + HALF] : normed[d - HALF];
		out = normed[d] * cosf(angle) + partner * sinf(angle);
	}
	block[(uint64_t)row * D + d] = LmFloatToBf16(out);
}

template<class G>
__global__ static void QwenIndexStoreKernel(LmKvShardView view, const uint16_t *block, const uint32_t *sequence_of_row, const uint32_t *positions)
{
	using S = QwenHybridShape<G>;
	const uint32_t row = blockIdx.x, position = positions[row];
	if ( position % G::kIndexRatio != G::kIndexRatio - 1u || SparkKvShardOwns(view.shard, position / G::kIndexRatio) == 0u )
		return;
	uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::IndexKv>(view, sequence_of_row[row], position / G::kIndexRatio, row,
		LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
		return;
	for ( uint32_t d = threadIdx.x; d < G::kIndexDim; d += blockDim.x )
		slot[d] = block[(uint64_t)row * G::kIndexDim + d];
}

template<class G>
__global__ static void QwenIndexTailKernel(const uint16_t *raw, uint16_t *tail, const uint32_t *positions, const uint32_t *sequence_row_begin,
	const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, uint32_t commit)
{
	constexpr uint32_t D = G::kIndexDim, R = G::kIndexRatio, T = QwenHybridShape<G>::kIndexTail;
	const uint32_t lane = blockIdx.x, d = threadIdx.x;
	const uint32_t begin = sequence_row_begin[lane], count = sequence_row_begin[lane + 1u] - begin;
	if ( commit == 0u || count == 0u )
		return;
	const uint32_t *lane_rows = sequence_row_indices + begin;
	const uint32_t first = positions[lane_rows[0]], end = positions[lane_rows[count - 1u]] + 1u;
	const uint32_t old_base = first / R * R, new_base = end / R * R;
	uint16_t *slot = tail + (uint64_t)recurrent_slot[lane] * T * D;
	uint16_t values[T > 0u ? T : 1u];
	for ( uint32_t t = 0u; t < end - new_base; ++t )
	{
		const uint32_t at = new_base + t;
		values[t] = at >= first ? raw[(uint64_t)lane_rows[at - first] * D + d] : slot[(uint64_t)(at - old_base) * D + d];
	}
	for ( uint32_t t = 0u; t < end - new_base; ++t )
		slot[(uint64_t)t * D + d] = values[t];
}

template<class G>
__global__ static void QwenIndexScoreKernel(LmKvShardView view, const uint16_t *query, const uint32_t *sequence_of_row, const uint32_t *context_length,
	const uint32_t *positions, float *scores, uint32_t stride, float scale)
{
	using S = QwenHybridShape<G>;
	constexpr uint32_t D = G::kIndexDim, H = G::kIndexHeads;
	__shared__ float shared_query[H * D];
	const uint32_t row = blockIdx.y, sequence = sequence_of_row[row];
	uint32_t limit = context_length[sequence];
	if ( positions[row] + 1u < limit )
		limit = positions[row] + 1u;
	const uint32_t blocks = limit / G::kIndexRatio, local_count = SparkKvShardLocalKeys(view.shard, blocks);
	for ( uint32_t index = threadIdx.x; index < H * D; index += blockDim.x )
		shared_query[index] = LmBf16ToFloat(query[(uint64_t)row * H * D + index]);
	__syncthreads();
	for ( uint32_t local = blockIdx.x * blockDim.x + threadIdx.x; local < stride; local += gridDim.x * blockDim.x )
	{
		float score = -INFINITY;
		if ( blocks > S::kIndexBlocks && local < local_count )
		{
			const uint16_t *key = (const uint16_t *)LmKvShardSlotRequired<typename S::IndexKv>(view, sequence,
				SparkKvShardLocalPosition(view.shard, local), row, LM_KV_ACCESS_READ);
			if ( key != 0 )
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
		scores[(uint64_t)row * stride + local] = score;
	}
}

template<class G>
__global__ static void QwenIndexMaskKernel(const uint32_t *selected, const uint32_t *sequence_of_row, const uint32_t *context_length,
	const uint32_t *positions, uint32_t *mask, uint32_t mask_words, uint32_t first_row)
{
	using S = QwenHybridShape<G>;
	const uint32_t row = first_row + blockIdx.x, sequence = sequence_of_row[row];
	uint32_t limit = context_length[sequence];
	if ( positions[row] + 1u < limit )
		limit = positions[row] + 1u;
	const uint32_t blocks = limit / G::kIndexRatio;
	uint32_t *row_mask = mask + (uint64_t)row * mask_words;
	if ( blocks <= S::kIndexBlocks )
	{
		for ( uint32_t word = threadIdx.x; word < mask_words; word += blockDim.x )
		{
			const uint32_t low = word * 32u;
			row_mask[word] = blocks >= low + 32u ? 0xffffffffu : blocks > low ? (1u << (blocks - low)) - 1u : 0u;
		}
		return;
	}
	for ( uint32_t entry = threadIdx.x; entry < S::kIndexBlocks; entry += blockDim.x )
	{
		const uint32_t block = selected[(uint64_t)row * S::kIndexBlocks + entry];
		if ( block < blocks )
			atomicOr(&row_mask[block / 32u], 1u << (block % 32u));
	}
}

template<class G>
struct QwenHybridModel
{
	using S = QwenHybridShape<G>;
	using State = QwenHybridState<G>;

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + QH_THREADS - 1u) / QH_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	template<class Format>
	static int32_t Project(const void *weight, const float *scale_rows, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		LmScaleTensor scale = scale_rows != 0 ?
			LmScaleTensorBuild(scale_rows, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u, QH_SCALE_K) : LmScaleTensorNone();
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			int32_t status;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.scale = scale;
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			status = LmSkinnyLaunch<Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension, output_dimension,
			0u, 0u, multiprocessors, stream);
	}

	static int32_t Linear(const State *state, const uint8_t *weight, const float *scale_rows, const uint16_t *source,
		uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return state->linear_bytes == 1u ?
			Project<LmFp8>(weight, scale_rows, source, destination, rows, input_dimension, output_dimension, multiprocessors, stream) :
			Project<LmBf16Format>(weight, 0, source, destination, rows, input_dimension, output_dimension, multiprocessors, stream);
	}

	static SparkStatus Round(State *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *local, void *full)
	{
		return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL,
			rows, row_elements, local, full);
	}

	static SparkStatus Launched(void)
	{
		return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}

	static SparkStatus Gdn(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sequences = step->sequences, heads = state->gdn_heads;
		const uint32_t sms = step->multiprocessors;
		uint8_t *state_pool = state->gdn_state + (uint64_t)layer->recurrent_index * state->slots * state->gdn_state_bytes;
		uint16_t *window = state->gdn_window + (uint64_t)layer->recurrent_index * state->slots * (state->gdn_window_bytes / sizeof(uint16_t));
		LmQkvLayout layout;
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->qkv, rows, G::kHidden,
				state->gdn_channels, sms, stream) != LM_LAUNCH_OK ||
			Linear(state, layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * state->linear_bytes,
				layer->mix_scale == 0 ? (const float *)0 : layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
				state->normed, state->z, rows, G::kHidden, state->gdn_v, sms, stream) != LM_LAUNCH_OK ||
			Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmCausalConvKernel<QH_THREADS,QH_CONV,LM_CONV_SWISH,uint16_t>), dim3(sequences,(state->gdn_channels + QH_THREADS - 1u) / QH_THREADS),
			QH_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0, state->qkv, layer->gdn_conv,
			state->qkv, state->gdn_channels, sequences, step->commit, step->sequence_row_indices);
		layout.query_dimension = state->gdn_qk;
		layout.key_dimension = state->gdn_qk;
		layout.value_dimension = state->gdn_v;
		layout.rope_dimension = 0u;
		layout.head_dimension = QH_KEY_DIM;
		LM_LAUNCH((LmSplitQkvKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->qkv, layout, state->query, state->key, state->value, rows, 1.0f);
		LM_LAUNCH((LmExpandHeadsKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->query, state->query_heads,
			state->gdn_qk / QH_KEY_DIM, QH_KEY_DIM, heads / (state->gdn_qk / QH_KEY_DIM), rows);
		LM_LAUNCH((LmExpandHeadsKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->key, state->key_heads,
			state->gdn_qk / QH_KEY_DIM, QH_KEY_DIM, heads / (state->gdn_qk / QH_KEY_DIM), rows);
		LM_LAUNCH((LmGdnGateKernel<QH_THREADS,QH_KEY_DIM>), dim3(rows,heads), QH_THREADS, 0, stream,
			state->ba + heads, state->ba, layer->gdn_a_log, layer->gdn_dt_bias, state->retention, state->write_gate, heads, rows, 2u * heads);
		LM_LAUNCH((LmDeltaRuleColumnKernel<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>), dim3(sequences,heads,QH_VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index, step->sequence_row_begin,
			(const uint32_t *)0, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
			heads, 1u, sequences, step->commit, step->sequence_row_indices);
		if constexpr ( G::kGdnSigmoidGate )
			LM_LAUNCH((LmHeadRmsNormSigmoidGateKernel<QH_THREADS>), dim3(heads,rows), QH_THREADS, 0, stream,
				state->mixed, state->z, layer->gdn_norm, state->mixed, rows, heads, QH_VALUE_DIM, QH_EPSILON);
		else
			LM_LAUNCH((LmHeadRmsNormSiluGateKernel<QH_THREADS>), dim3(heads,rows), QH_THREADS, 0, stream,
				state->mixed, state->z, layer->gdn_norm, state->mixed, rows, heads, QH_VALUE_DIM, QH_EPSILON);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->mixed, state->partial, rows, state->gdn_v, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus IndexPrepare(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		uint16_t *tail = state->index_tail + (uint64_t)layer->kv_index * state->slots * S::kIndexTail * G::kIndexDim;
		if ( Project<LmBf16Format>(layer->index_qk, 0, state->normed, state->index_qk, rows, G::kHidden, S::kIndexWidth, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenIndexSplitKernel<G><<<dim3(Blocks(S::kIndexWidth), rows), QH_THREADS, 0, stream>>>(state->index_qk, state->index_query, state->index_key);
		LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kIndexHeads,rows), QH_THREADS, 0, stream,
			state->index_query, layer->index_q_norm, state->index_query, rows, G::kIndexHeads, G::kIndexDim, QH_EPSILON, 1.0f);
		LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kIndexHeads), QH_THREADS, 0, stream,
			state->index_query, step->positions, G::kIndexHeads, G::kIndexDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
		QwenIndexBlockKeysKernel<G><<<rows, G::kIndexDim, 0, stream>>>(state->index_key, tail, layer->index_k_norm, state->index_block,
			step->positions, state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index,
			G::kRopeTheta);
		QwenIndexStoreKernel<G><<<rows, G::kIndexDim, 0, stream>>>(state->index_views[layer->kv_index], state->index_block, step->sequence_of_row,
			step->positions);
		QwenIndexTailKernel<G><<<step->sequences, G::kIndexDim, 0, stream>>>(state->index_key, tail, step->positions, step->sequence_row_begin,
			step->sequence_row_indices, step->recurrent_index, step->commit);
		return Launched();
	}

	static SparkStatus IndexSelect(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows;
		const LmKvShardView *view = &state->index_views[layer->kv_index];
		SparkStatus status;
		QwenIndexScoreKernel<G><<<dim3((state->index_local_stride + QH_THREADS - 1u) / QH_THREADS, rows), QH_THREADS, 0, stream>>>(*view,
			state->index_query, step->sequence_of_row, step->context_length, step->positions, state->index_scores, state->index_local_stride,
			1.0f / sqrtf((float)G::kIndexDim));
		if ( Launched() != SPARK_STATUS_OK ||
			LmTopkExactLaunch<QH_THREADS>(state->index_scores, rows, state->index_local_stride, S::kIndexBlocks, LM_TOPK_EXACT_CHUNK,
				LM_TOPK_EXACT_CHUNKED_ROWS, state->index_topk_values, state->index_topk_positions, state->index_topk_entries,
				state->index_local_selected, stream) != cudaSuccess ||
			LmIndexShardCandidatePackLaunch<QH_THREADS>(view->shard, state->index_scores, state->index_local_stride, state->index_local_selected,
				S::kIndexBlocks, rows, state->index_candidates, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows,
			S::kIndexBlocks * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->index_candidates, state->index_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( cudaMemsetAsync(state->index_mask, 0, (uint64_t)rows * state->index_mask_words * sizeof(uint32_t), stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for ( uint32_t first = 0u; first < rows; first += QH_INDEX_CHUNK )
		{
			const uint32_t count = rows - first < QH_INDEX_CHUNK ? rows - first : QH_INDEX_CHUNK;
			if ( LmIndexShardCandidateScatterLaunch<QH_THREADS>(state->index_gathered + (uint64_t)first * S::kIndexBlocks,
					(uint64_t)rows * S::kIndexBlocks, state->tp_degree, S::kIndexBlocks, count, state->index_global_stride, state->index_full_scores,
					stream) != cudaSuccess ||
				LmTopkExactLaunch<QH_THREADS>(state->index_full_scores, count, state->index_global_stride, S::kIndexBlocks, LM_TOPK_EXACT_CHUNK,
					LM_TOPK_EXACT_CHUNKED_ROWS, state->index_topk_values, state->index_topk_positions, state->index_topk_entries,
					state->index_selected + (uint64_t)first * S::kIndexBlocks, stream) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			QwenIndexMaskKernel<G><<<count, QH_THREADS, 0, stream>>>(state->index_selected, step->sequence_of_row, step->context_length,
				step->positions, state->index_mask, state->index_mask_words, first);
		}
		return Launched();
	}

	static SparkStatus Attention(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->attn_local, rows, G::kHidden,
			state->attn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, state->attn_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		QwenHybridGatherRowsKernel<<<dim3(Blocks(S::kQkvWidth),rows), QH_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, state->attn_rows, degree);
		QwenHybridSplitAttentionKernel<G><<<dim3(Blocks(S::kQkvWidth),rows), QH_THREADS, 0, stream>>>(state->attn_full, state->attn_query,
			state->attn_gate, state->attn_key, state->attn_value);
		LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kHeads,rows), QH_THREADS, 0, stream,
			state->attn_query, layer->q_norm, state->attn_query, rows, G::kHeads, G::kHeadDim, QH_EPSILON, 1.0f);
		LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kKvHeads,rows), QH_THREADS, 0, stream,
			state->attn_key, layer->k_norm, state->attn_key, rows, G::kKvHeads, G::kHeadDim, QH_EPSILON, 1.0f);
		LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kHeads), QH_THREADS, 0, stream,
			state->attn_query, step->positions, G::kHeads, G::kHeadDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
		LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kKvHeads), QH_THREADS, 0, stream,
			state->attn_key, step->positions, G::kKvHeads, G::kHeadDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
		QwenHybridPackSlotKernel<G><<<dim3(Blocks(2u * S::kKvWidth),rows), QH_THREADS, 0, stream>>>(state->attn_key, state->attn_value, state->attn_slot);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,QH_THREADS>), rows, QH_THREADS, 0, stream,
			*view, state->attn_slot, step->sequence_of_row, step->positions, rows, 2u * S::kKvWidth);
		if constexpr ( S::kIndexer )
		{
			status = IndexPrepare(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = IndexSelect(state, layer, step, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		if ( Launched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kKvHeads,G::kHeadDim,G::kHeadDim>(*view, state->attn_query, G::kHeads,
				step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kHeadDim), state->send, stride, rows, stream,
				S::kIndexer ? state->index_mask : (const uint32_t *)0, state->index_mask_words, G::kIndexRatio) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kHeadDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridSliceRowsKernel<<<dim3(Blocks(state->attn_width),rows), QH_THREADS, 0, stream>>>(state->attn_gate, state->merged_gate,
			S::kQWidth, state->tp_rank * state->attn_width, state->attn_width);
		LM_LAUNCH((LmOutputGateKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->merged, state->merged_gate, state->attn_width);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static int32_t ExpertsUp(State *state, const uint8_t *weight, const uint8_t *scale_data, uint16_t *output, uint32_t rows, uint32_t routes,
		uint32_t sms, cudaStream_t stream)
	{
		const LmScaleTensor scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_NVFP4_E2M1>(scale_data, G::kLocalExperts, G::kMoeInter, G::kHidden);
		int32_t launch = LmSkinnyGroupedExperts<LmNvfp4>(weight, scale, state->normed, output, state->group_row_offset, state->route_source_token,
			G::kLocalExperts, routes, 0u, G::kHidden, G::kMoeInter, stream);
		if ( launch == LM_LAUNCH_ERR_SHAPE )
		{
			LmGemmArguments gemm;
			memset(&gemm, 0, sizeof(gemm));
			gemm.scale_a = LmScaleTensorNone();
			gemm.scale_b = scale;
			gemm.prefix_built = 1u;
			gemm.group_row_offset = state->group_row_offset;
			gemm.group_tile_prefix = state->tile_prefix_up;
			gemm.source_row_map = state->route_source_token;
			gemm.source_row_count = rows;
			gemm.output_bf16 = output;
			launch = LmGemmWeightOnlyIndirectLaunch<LmNvfp4,QH_GEMM_TILE_N,QH_GEMM_STAGES,QH_GEMM_WARPS>(&gemm, state->normed, weight, routes, rows,
				G::kTopK, G::kLocalExperts, G::kHidden, G::kMoeInter, sms, stream);
		}
		return launch;
	}

	static SparkStatus Moe(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * G::kTopK;
		const LmScaleTensor w2_scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_NVFP4_E2M1>(layer->moe_w2_scale, G::kLocalExperts, G::kHidden, G::kMoeInter);
		int32_t launch;
		if ( Project<LmBf16Format>(layer->moe_router, 0, state->normed, state->router_logits, rows, G::kHidden, G::kExperts, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridSoftmaxRowsKernel<G::kExperts><<<rows, QH_THREADS, 0, stream>>>(state->router_logits, state->router_probs);
		if ( LmTopkRouteLaunch<QH_THREADS,G::kTopK,true,LM_TOPK_SCORE_IDENTITY>(rows, state->router_probs, G::kExperts, state->route_expert,
			state->route_weight, (const float *)0, (const uint16_t *)0, 1.0f, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<QH_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, QH_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down,
			stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		launch = ExpertsUp(state, layer->moe_w1, layer->moe_w1_scale, state->expert_gate, rows, routes, sms, stream);
		if ( launch == LM_LAUNCH_OK )
			launch = ExpertsUp(state, layer->moe_w3, layer->moe_w3_scale, state->expert_up, rows, routes, sms, stream);
		if ( launch != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmSwigluPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), QH_THREADS, 0, stream>>>(state->expert_gate, state->expert_up,
			state->packed_weight, state->expert_act, G::kMoeInter, INFINITY);
		launch = LmSkinnyGroupedExperts<LmNvfp4>(layer->moe_w2, w2_scale, state->expert_act, state->expert_out, state->group_row_offset,
			state->route_source_token, G::kLocalExperts, routes, 1u, G::kMoeInter, G::kHidden, stream);
		if ( launch == LM_LAUNCH_ERR_SHAPE )
		{
			LmGemmArguments gemm;
			memset(&gemm, 0, sizeof(gemm));
			gemm.scale_a = LmScaleTensorNone();
			gemm.scale_b = w2_scale;
			gemm.prefix_built = 1u;
			gemm.group_row_offset = state->group_row_offset;
			gemm.group_tile_prefix = state->tile_prefix_down;
			gemm.output_bf16 = state->expert_out;
			launch = LmGemmWeightOnlyLaunch<LmNvfp4,QH_GEMM_TILE_N,QH_GEMM_STAGES,QH_GEMM_WARPS>(&gemm, state->expert_act, layer->moe_w2, routes, rows,
				G::kTopK, G::kLocalExperts, G::kMoeInter, G::kHidden, sms, true, stream);
		}
		if ( launch != LM_LAUNCH_OK ||
			Project<LmBf16Format>(layer->moe_shared_gate_up, 0, state->normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridRowDotKernel<<<rows, QH_THREADS, 0, stream>>>(state->normed, layer->moe_shared_gate, state->shared_logit, G::kHidden);
		LM_LAUNCH((LmSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows, true);
		if ( Launched() != SPARK_STATUS_OK ||
			Project<LmBf16Format>(layer->moe_shared_down, 0, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridSigmoidGateRowsKernel<<<dim3(Blocks(G::kHidden), rows), QH_THREADS, 0, stream>>>(state->shared_out, state->shared_logit, G::kHidden);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), QH_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return Launched();
	}

	static SparkStatus Ffn(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		if constexpr ( G::kMoe )
			return Moe(state, layer, step, stream);
		else
		{
			const uint32_t rows = step->rows, sms = step->multiprocessors;
			if ( Linear(state, layer->ffn_gate_up, layer->ffn_gate_up_scale, state->normed, state->gate_up, rows, G::kHidden,
				2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			LM_LAUNCH((LmSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true);
			if ( Launched() != SPARK_STATUS_OK ||
				Linear(state, layer->ffn_down, layer->ffn_down_scale, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return SPARK_STATUS_OK;
		}
	}

	static SparkStatus DumpLayer(State *state, uint32_t layer, const SparkStageRunnerStep *step, const uint16_t *hidden,
		const uint16_t *partial, cudaStream_t stream)
	{
		const uint32_t width = S::kHyper && layer != 0xffffffffu ? S::kResidual : G::kHidden;
		const uint32_t header[4] = { S::kHyper ? 0x484c5751u : 0x444c5751u, layer, step->rows, width };
		const uint64_t values = (uint64_t)step->rows * width, partials = (uint64_t)step->rows * G::kHidden;
		if ( cudaStreamSynchronize(stream) != cudaSuccess ||
			cudaMemcpy(state->dump_positions, step->positions, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(state->dump_rows, hidden, (size_t)values * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
			(partial != 0 && cudaMemcpy(state->dump_rows + values, partial, (size_t)partials * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( partial == 0 )
			memset(state->dump_rows + values, 0, (size_t)partials * sizeof(uint16_t));
		fwrite(header, sizeof(uint32_t), S::kHyper ? 4u : 3u, state->dump);
		fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		if ( S::kHyper && layer == 0xffffffffu )
		{
			if ( step->token_ids == 0 ||
				cudaMemcpy(state->dump_positions, step->token_ids, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		}
		fwrite(state->dump_rows, sizeof(uint16_t), (size_t)(values + partials), state->dump);
		fflush(state->dump);
		return SPARK_STATUS_OK;
	}

	static SparkStatus HcPre(State *state, const float *norm, const uint16_t *down, const uint16_t *up, const uint16_t *inject,
		uint16_t *mixed, uint32_t rows, uint32_t sms, cudaStream_t stream)
	{
		LmGroupRmsNormKernel<QH_THREADS><<<dim3(G::kStreams, rows), QH_THREADS, 0, stream>>>(state->residual, norm, state->hc_normed,
			G::kStreams, G::kHidden, QH_EPSILON);
		if ( Project<LmBf16Format>(down, 0, state->hc_normed, state->hc_low, rows, S::kResidual, G::kHcLowRank, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmHyperGateKernel<QH_THREADS><<<Blocks((uint64_t)rows * G::kHcLowRank), QH_THREADS, 0, stream>>>(state->hc_low,
			(uint64_t)rows * G::kHcLowRank, (float)G::kStreams);
		if ( Project<LmBf16Format>(up, 0, state->hc_low, state->hc_up, rows, G::kHcLowRank, S::kResidual, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmHyperMixKernel<QH_THREADS><<<dim3(Blocks(G::kHidden), rows), QH_THREADS, 0, stream>>>(state->hc_up, state->hc_normed, mixed,
			G::kStreams, G::kHidden);
		if ( inject != 0 )
			LmHyperInjectWeightsKernel<QH_THREADS, G::kStreams><<<rows, QH_THREADS, 0, stream>>>(state->hc_normed, inject, state->hc_inject, G::kHidden);
		return Launched();
	}

	static SparkStatus HcInject(State *state, uint32_t rows, cudaStream_t stream)
	{
		LmHyperInjectKernel<QH_THREADS><<<dim3(Blocks(S::kResidual), rows), QH_THREADS, 0, stream>>>(state->residual, state->partial,
			state->hc_inject, G::kStreams, G::kHidden);
		return Launched();
	}

	static SparkStatus Ple(State *state, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		const uint64_t elements = (uint64_t)rows * S::kResidual;
		SparkStatus status;
		if ( step->token_ids == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		LmNgramHashKernel<<<(rows + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, (const int32_t *)0, step->positions, state->row_lane,
			state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, state->ple_history, S::kPleHistory,
			state->ple_multipliers, state->ple_vocab_sizes, state->ple_offsets, state->ple_ids, rows, G::kPleOrders, G::kPleHeadsPerOrder,
			(int32_t)G::kPleEos, (int32_t)G::kPleEos, 1u);
		LmNgramGatherScaledKernel<<<dim3(rows, S::kPleColumns), G::kPleHeadDim, 0, stream>>>(state->ple_ids, state->ple_table, state->ple_table_scale,
			state->ple_embed, state->ple_table_first, state->ple_table_rows, S::kPleColumns, G::kPleHeadDim);
		status = Launched();
		if ( status == SPARK_STATUS_OK )
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->ple_embed, state->ple_embed);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( Project<LmBf16Format>(state->ple_key_weight, 0, state->ple_embed, state->ple_key, rows, G::kHidden, S::kResidual, sms, stream) != LM_LAUNCH_OK ||
			Project<LmBf16Format>(state->ple_value_weight, 0, state->ple_embed, state->ple_value, rows, G::kHidden, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmGroupRmsNormKernel<QH_THREADS><<<dim3(G::kStreams, rows), QH_THREADS, 0, stream>>>(state->ple_key, state->ple_norm_key, state->ple_key,
			G::kStreams, G::kHidden, QH_EPSILON);
		LmGroupRmsNormKernel<QH_THREADS><<<dim3(G::kStreams, rows), QH_THREADS, 0, stream>>>(state->residual, state->ple_norm_query, state->ple_query,
			G::kStreams, G::kHidden, QH_EPSILON);
		LmHyperKeyGateKernel<QH_THREADS><<<dim3(G::kStreams, rows), QH_THREADS, 0, stream>>>(state->ple_key, state->ple_query, state->ple_value,
			state->ple_gated, G::kStreams, G::kHidden);
		LmGroupRmsNormKernel<QH_THREADS><<<dim3(G::kStreams, rows), QH_THREADS, 0, stream>>>(state->ple_gated, state->ple_norm_conv, state->ple_conv_in,
			G::kStreams, G::kHidden, QH_EPSILON);
		LM_LAUNCH((LmCausalConvKernel<QH_THREADS,G::kPleConv,LM_CONV_SWISH,uint16_t,G::kPleDilation>),
			dim3(step->sequences,(S::kResidual + QH_THREADS - 1u) / QH_THREADS), QH_THREADS, 0, stream, state->ple_window, step->recurrent_index,
			step->sequence_row_begin, (const uint32_t *)0, state->ple_conv_in, state->ple_conv_weight, state->ple_conv_out, S::kResidual,
			step->sequences, step->commit, step->sequence_row_indices);
		LmHyperAddKernel<QH_THREADS><<<Blocks(elements), QH_THREADS, 0, stream>>>(state->residual, state->ple_gated, state->ple_conv_out, elements);
		LmNgramHistoryKernel<<<(step->sequences + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, (const int32_t *)0, step->sequence_row_begin,
			step->sequence_row_indices, step->recurrent_index, state->ple_history, S::kPleHistory, step->sequences, G::kPleOrders, step->commit);
		return Launched();
	}

	static SparkStatus HyperStep(State *state, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		uint16_t *hidden = step->hidden_bf16;
		SparkStatus status;
		LmHyperExpandKernel<QH_THREADS><<<dim3(Blocks(S::kResidual), rows), QH_THREADS, 0, stream>>>(hidden, state->residual, G::kStreams, G::kHidden);
		if constexpr ( S::kPle || S::kIndexer )
			LmRowLanesKernel<<<step->sequences, QH_THREADS, 0, stream>>>(step->sequence_row_begin, step->sequence_row_indices, state->row_lane,
				state->row_ordinal);
		status = Launched();
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const QwenHybridLayer *layer = &state->layers[index];
			if constexpr ( S::kPle )
				if ( index == G::kPleLayer )
					status = Ple(state, step, stream);
			for ( uint32_t part = 0u; part < 2u && status == SPARK_STATUS_OK; ++part )
			{
				status = HcPre(state, layer->hc_norm[part], layer->hc_down[part], layer->hc_up[part], layer->hc_inject[part], state->normed, rows, sms, stream);
				if ( status == SPARK_STATUS_OK )
					status = part == 1u ? Ffn(state, layer, step, stream) :
						index % G::kPeriod == G::kPhase ? Attention(state, layer, step, stream) : Gdn(state, layer, step, stream);
				if ( status == SPARK_STATUS_OK )
					status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
				if ( status == SPARK_STATUS_OK )
					status = HcInject(state, rows, stream);
			}
			if ( status == SPARK_STATUS_OK && state->dump != 0 )
				status = DumpLayer(state, index, step, state->residual, state->partial, stream);
			if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		return HcPre(state, state->final_hc_norm, state->final_hc_down, state->final_hc_up, (const uint16_t *)0, hidden, rows, sms, stream);
	}

	static SparkStatus PlainStep(State *state, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows;
		uint16_t *hidden = step->hidden_bf16;
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const QwenHybridLayer *layer = &state->layers[index];
			LM_LAUNCH((LmFusedResidualRmsNormKernel<QH_THREADS,float>), rows, QH_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, index == 0u ? (const uint16_t *)0 : state->partial, layer->input_norm, index == 0u ? (uint16_t *)0 : hidden,
				state->normed, G::kHidden, G::kHidden, QH_EPSILON);
			status = index % G::kPeriod == G::kPhase ? Attention(state, layer, step, stream) : Gdn(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			LM_LAUNCH((LmFusedResidualRmsNormKernel<QH_THREADS,float>), rows, QH_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, state->partial, layer->post_norm, hidden, state->normed, G::kHidden, G::kHidden, QH_EPSILON);
			status = Ffn(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status == SPARK_STATUS_OK && state->dump != 0 )
				status = DumpLayer(state, index, step, hidden, state->partial, stream);
			if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		QwenHybridAddRowsKernel<<<Blocks((uint64_t)rows * G::kHidden), QH_THREADS, 0, stream>>>(hidden, state->partial, (uint64_t)rows * G::kHidden);
		return Launched();
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream_void)
	{
		State *state = (State *)model;
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		uint16_t *hidden = step->hidden_bf16;
		SparkStatus status = SPARK_STATUS_OK;
		if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( state->dump != 0 )
		{
			cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
			if ( cudaStreamIsCapturing(stream, &capturing) != cudaSuccess || capturing != cudaStreamCaptureStatusNone )
				return SPARK_STATUS_UNSUPPORTED;
			status = DumpLayer(state, 0xffffffffu, step, hidden, 0, stream);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		if constexpr ( S::kHyper )
			return HyperStep(state, step, stream);
		else
			return PlainStep(state, step, stream);
	}

	static SparkStatus Entry(State *state, const char *name, const char *kind, uint64_t *offset, uint64_t *bytes, uint32_t *rows)
	{
		SparkNamedPackEntry entry;
		if ( SparkNamedPackEntryLoad(&state->pack, name, &entry) != SPARK_STATUS_OK || strcmp(entry.kind, kind) != 0 || entry.shape_count < 1u )
		{
			fprintf(stderr, "%s: pack tensor %s missing or not %s\n", G::Tag(), name, kind);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		*offset = state->pack.payload_base + entry.payload_offset;
		*bytes = entry.bytes;
		if ( rows != 0 )
			*rows = entry.shape[0];
		return SPARK_STATUS_OK;
	}

	static SparkStatus Plan(State *state, uint64_t *bytes, uint8_t *base)
	{
		const uint64_t rows = state->max_rows, slots = state->slots, routes = rows * (G::kMoe ? G::kTopK : 0u);
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t);
		struct { void **target; uint64_t bytes; } plan[] =
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
			{ (void **)&state->index_topk_values, state->index_topk_entries * sizeof(float) },
			{ (void **)&state->index_topk_positions, state->index_topk_entries * sizeof(uint32_t) },
			{ (void **)&state->index_candidates, rows * S::kIndexBlocks * sizeof(uint2) },
			{ (void **)&state->index_gathered, (uint64_t)state->tp_degree * rows * S::kIndexBlocks * sizeof(uint2) },
			{ (void **)&state->index_full_scores, S::kIndexer ? (uint64_t)QH_INDEX_CHUNK * state->index_global_stride * sizeof(float) : 0u },
			{ (void **)&state->index_selected, rows * S::kIndexBlocks * sizeof(uint32_t) },
			{ (void **)&state->index_mask, rows * state->index_mask_words * sizeof(uint32_t) },
			{ (void **)&state->kv_error, sizeof(LmKvAccessError) },
		};
		uint64_t cursor = 0u;
		for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
		{
			cursor = (cursor + 255u) & ~255ull;
			if ( base != 0 )
				*plan[index].target = plan[index].bytes != 0u ? base + cursor : 0;
			cursor += plan[index].bytes;
		}
		*bytes = cursor;
		return SPARK_STATUS_OK;
	}

	static void Close(void *model)
	{
		State *state = (State *)model;
		if ( state == 0 )
			return;
		if ( state->arena != 0 )
			(void)cudaFree(state->arena);
		if ( state->pack_open != 0u )
			SparkNamedPackClose(&state->pack);
		if ( state->dump != 0 )
			fclose(state->dump);
		free(state->dump_rows);
		free(state->dump_positions);
		free(state);
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

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		State *state;
		SparkStatus status;
		uint32_t embed_rows, head_rows;
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree ||
			configuration->max_input_row_count == 0u || configuration->state_budget_bytes == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state = (State *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->max_rows = configuration->max_input_row_count;
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
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, QH_PACK_ALIGNMENT, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		status = Config(state, configuration);
		if ( status != SPARK_STATUS_OK )
			return status;
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
			state->max_positions = configuration->kv_pages_per_sequence * G::kPageSlots;
			const uint32_t blocks = state->max_positions / G::kIndexRatio;
			const uint32_t local = ((SparkKvShardGatherKeys(shard, blocks) + 63u) / 64u) * 64u, global = ((blocks + 63u) / 64u) * 64u;
			state->index_local_stride = local > S::kIndexBlocks ? local : S::kIndexBlocks;
			state->index_global_stride = global > S::kIndexBlocks ? global : S::kIndexBlocks;
			state->index_mask_words = state->index_global_stride / 32u;
			const uint64_t local_entries = (uint64_t)state->max_rows * LmTopkExactCandidateEntries(state->index_local_stride, S::kIndexBlocks, LM_TOPK_EXACT_CHUNK);
			const uint64_t global_entries = (uint64_t)QH_INDEX_CHUNK * LmTopkExactCandidateEntries(state->index_global_stride, S::kIndexBlocks, LM_TOPK_EXACT_CHUNK);
			state->index_topk_entries = 2u * (local_entries > global_entries ? local_entries : global_entries);
		}
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, state->tp_degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
		status = Plan(state, &state->arena_bytes, 0);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes is over the %llu-byte budget (rows %u, slots %u)\n", G::Tag(),
				(unsigned long long)state->arena_bytes, (unsigned long long)configuration->state_budget_bytes, state->max_rows, state->slots);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = G::kHidden;
		geometry->vocab = G::kVocab;
		geometry->total_layers = G::kLayers;
		geometry->first_layer = 0u;
		geometry->layer_count = G::kLayers;
		geometry->rms_epsilon = QH_EPSILON;
		geometry->pack_bytes = state->pack.file_bytes;
		geometry->kv_layer_count = S::kAttnLayers;
		geometry->kv_layer_page_bytes = S::Kv::kPageBytes;
		geometry->kv_second_layer_count = S::kIndexer ? S::kAttnLayers : 0u;
		geometry->kv_second_layer_page_bytes = S::kIndexer ? S::IndexKv::kPageBytes : 0u;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->recurrent_bytes = RecurrentBytes(state);
		geometry->head_norm_f32 = 1u;
		geometry->head_prenormed = S::kHyper ? 1u : 0u;
		if ( Entry(state, "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			(!S::kHyper && Entry(state, "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK) ||
			Entry(state, "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &head_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->embed_rows = embed_rows;
		geometry->head_rows = head_rows;
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		(void)model;
		if ( manifest == 0 || manifest->range_count != 0u )
		{
			fprintf(stderr, "%s: a dense pack's weightd manifest carries no expert ranges\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Slice(State *state, SparkWeightdLazyPack *lazy_pack, const char *name, const char *kind, const void **out)
	{
		uint64_t offset, bytes;
		if ( Entry(state, name, kind, &offset, &bytes, 0) != SPARK_STATUS_OK ||
			SparkWeightdLazyPackSlice(lazy_pack, offset, bytes, out) != SPARK_STATUS_OK || *out == 0 )
		{
			fprintf(stderr, "%s: weightd has no resident bytes for %s\n", G::Tag(), name);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindTensor(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, const void **field)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(key, sizeof(key), "layers.%u.%s", index, name);
		return Slice(state, lazy_pack, key, kind, field);
	}

	template<class T>
	static SparkStatus Bind(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, T **field)
	{
		const void *bound = 0;
		const SparkStatus status = BindTensor(state, lazy_pack, index, name, kind, &bound);
		*field = (T *)bound;
		return status;
	}

	template<class W>
	static SparkStatus BindLinear(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, W **field, const float **scale_field)
	{
		char scale_name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		SparkStatus status = Bind(state, lazy_pack, index, name, state->linear_kind, field);
		*scale_field = 0;
		if ( status == SPARK_STATUS_OK && state->linear_bytes == 1u )
		{
			snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
			status = Bind(state, lazy_pack, index, scale_name, "f32", scale_field);
		}
		return status;
	}

	static SparkStatus BindExperts(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const uint8_t **field,
		const uint8_t **scale_field)
	{
		char scale_name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
		SparkStatus status = Bind(state, lazy_pack, index, name, "nvfp4_e2m1", field);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, scale_name, "ue4m3_f32_global", scale_field);
		return status;
	}

	static SparkStatus BindPle(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index)
	{
		SparkStatus status = Bind(state, lazy_pack, index, "ple_key", "bf16", &state->ple_key_weight);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_value", "bf16", &state->ple_value_weight);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_norm_key", "f32", &state->ple_norm_key);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_norm_query", "f32", &state->ple_norm_query);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_norm_conv", "f32", &state->ple_norm_conv);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_conv", "bf16", &state->ple_conv_weight);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_table", "fp8_e4m3", &state->ple_table);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_table_scale", "f32", &state->ple_table_scale);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_multipliers", "i64", &state->ple_multipliers);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_vocab_sizes", "i64", &state->ple_vocab_sizes);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, "ple_offsets", "i64", &state->ple_offsets);
		return status;
	}

	static SparkStatus BindModel(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		uint32_t recurrent = 0u, attention = 0u;
		(void)geometry;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			QwenHybridLayer *layer = &state->layers[index];
			const uint32_t attention_layer = index % G::kPeriod == G::kPhase;
			SparkStatus status = SPARK_STATUS_OK;
			if constexpr ( G::kMoe )
			{
				status = Bind(state, lazy_pack, index, "moe_router", "bf16", &layer->moe_router);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w1", &layer->moe_w1, &layer->moe_w1_scale);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w3", &layer->moe_w3, &layer->moe_w3_scale);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w2", &layer->moe_w2, &layer->moe_w2_scale);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_shared_gate_up", "bf16", &layer->moe_shared_gate_up);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_shared_down", "bf16", &layer->moe_shared_down);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_shared_gate", "bf16", &layer->moe_shared_gate);
			}
			else
			{
				status = BindLinear(state, lazy_pack, index, "ffn_gate_up", &layer->ffn_gate_up, &layer->ffn_gate_up_scale);
				if ( status == SPARK_STATUS_OK )
					status = BindLinear(state, lazy_pack, index, "ffn_down", &layer->ffn_down, &layer->ffn_down_scale);
			}
			if ( status == SPARK_STATUS_OK )
				status = BindLinear(state, lazy_pack, index, attention_layer ? "attn_qkv" : "gdn_qkvz", &layer->mix_weight, &layer->mix_scale);
			if ( status == SPARK_STATUS_OK )
				status = BindLinear(state, lazy_pack, index, attention_layer ? "attn_out" : "gdn_out", &layer->out_weight, &layer->out_scale);
			if ( status == SPARK_STATUS_OK && !S::kHyper )
				status = Bind(state, lazy_pack, index, "input_norm", "f32", &layer->input_norm);
			if ( status == SPARK_STATUS_OK && !S::kHyper )
				status = Bind(state, lazy_pack, index, "post_norm", "f32", &layer->post_norm);
			for ( uint32_t part = 0u; part < 2u && status == SPARK_STATUS_OK && S::kHyper; ++part )
			{
				const char *prefix = part == 0u ? "attn_hc_" : "mlp_hc_";
				char name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
				snprintf(name, sizeof(name), "%snorm", prefix);
				status = Bind(state, lazy_pack, index, name, "f32", &layer->hc_norm[part]);
				snprintf(name, sizeof(name), "%sdown", prefix);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, name, "bf16", &layer->hc_down[part]);
				snprintf(name, sizeof(name), "%sup", prefix);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, name, "bf16", &layer->hc_up[part]);
				snprintf(name, sizeof(name), "%sinject", prefix);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, name, "bf16", &layer->hc_inject[part]);
			}
			if ( status == SPARK_STATUS_OK && S::kPle && index == G::kPleLayer )
				status = BindPle(state, lazy_pack, index);
			if ( status == SPARK_STATUS_OK && attention_layer )
			{
				status = Bind(state, lazy_pack, index, "attn_q_norm", "f32", &layer->q_norm);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "attn_k_norm", "f32", &layer->k_norm);
				if ( status == SPARK_STATUS_OK && S::kIndexer )
					status = Bind(state, lazy_pack, index, "attn_index_qk", "bf16", &layer->index_qk);
				if ( status == SPARK_STATUS_OK && S::kIndexer )
					status = Bind(state, lazy_pack, index, "attn_index_q_norm", "f32", &layer->index_q_norm);
				if ( status == SPARK_STATUS_OK && S::kIndexer )
					status = Bind(state, lazy_pack, index, "attn_index_k_norm", "f32", &layer->index_k_norm);
				layer->kv_index = attention++;
			}
			else if ( status == SPARK_STATUS_OK )
			{
				status = Bind(state, lazy_pack, index, "gdn_ba", "bf16", &layer->gdn_ba);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_conv", "bf16", &layer->gdn_conv);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_norm", "bf16", &layer->gdn_norm);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_a_log", "f32", &layer->gdn_a_log);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_dt_bias", "f32", &layer->gdn_dt_bias);
				layer->recurrent_index = recurrent++;
			}
			if ( status != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
		}
		if constexpr ( S::kHyper )
		{
			const void *norm = 0, *down = 0, *up = 0;
			if ( Slice(state, lazy_pack, "final_hc_norm", "f32", &norm) != SPARK_STATUS_OK ||
				Slice(state, lazy_pack, "final_hc_down", "bf16", &down) != SPARK_STATUS_OK ||
				Slice(state, lazy_pack, "final_hc_up", "bf16", &up) != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
			state->final_hc_norm = (const float *)norm;
			state->final_hc_down = (const uint16_t *)down;
			state->final_hc_up = (const uint16_t *)up;
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			fprintf(stderr, "%s: the %llu-byte rank state could not be allocated\n", G::Tag(), (unsigned long long)state->arena_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return Plan(state, &state->arena_bytes, (uint8_t *)state->arena);
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != S::kAttnLayers || kv->layer_page_bytes != S::Kv::kPageBytes || kv->sequence_count > state->slots ||
			(S::kIndexer && (kv->second_pool == 0 || kv->second_layer_count != S::kAttnLayers || kv->second_layer_page_bytes != S::IndexKv::kPageBytes)) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < S::kAttnLayers && S::kIndexer; ++layer )
			if ( LmKvShardViewInitialize<typename S::IndexKv>(&state->index_views[layer], kv->second_pool + (uint64_t)layer * kv->second_layer_stride_bytes,
				kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < S::kAttnLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::Kv>(&state->kv_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes, kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}

	struct RecurrentPart
	{
		uint8_t *pool;
		uint64_t width;
		uint32_t count;
	};

	static uint32_t RecurrentParts(State *state, RecurrentPart *parts)
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

	static uint64_t RecurrentBytes(State *state)
	{
		RecurrentPart parts[5];
		const uint32_t count = RecurrentParts(state, parts);
		uint64_t bytes = 0u;
		for ( uint32_t part = 0u; part < count; ++part )
			bytes += parts[part].width * parts[part].count;
		return bytes;
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		State *state = (State *)model;
		RecurrentPart parts[5];
		const uint32_t count = RecurrentParts(state, parts);
		uint8_t *packed = (uint8_t *)buffer;
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots || buffer == 0 || bytes != RecurrentBytes(state) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < count && error == cudaSuccess; ++part )
		{
			uint8_t *pool = parts[part].pool + (uint64_t)slot * parts[part].width;
			const size_t pitch = (size_t)(parts[part].width * state->slots);
			error = to_buffer != 0u ?
				cudaMemcpy2DAsync(packed, (size_t)parts[part].width, pool, pitch, (size_t)parts[part].width, parts[part].count, cudaMemcpyDefault, (cudaStream_t)stream) :
				cudaMemcpy2DAsync(pool, pitch, packed, (size_t)parts[part].width, (size_t)parts[part].width, parts[part].count, cudaMemcpyDefault, (cudaStream_t)stream);
			packed += parts[part].width * parts[part].count;
		}
		if ( error == cudaSuccess && stream == 0 )
			error = cudaStreamSynchronize(0);
		return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		State *state = (State *)model;
		RecurrentPart parts[5];
		const uint32_t count = RecurrentParts(state, parts);
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < count && error == cudaSuccess; ++part )
			error = cudaMemset2DAsync(parts[part].pool + (uint64_t)slot * parts[part].width, (size_t)(parts[part].width * state->slots), 0,
				(size_t)parts[part].width, parts[part].count, (cudaStream_t)stream);
		return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}

	static SparkStatus Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
	{
		(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static const void *ProbeBuffers(void *model)
	{
		return model;
	}

	static void Report(void *model, uint32_t rank)
	{
		const State *state = (const State *)model;
		fprintf(stderr, "%s rank=%u arena_bytes=%llu slots=%u rows=%u ffn_rows=%u expert_first=%u\n", G::ReportTag(), rank,
			(unsigned long long)state->arena_bytes, state->slots, state->max_rows, state->ffn_rows, state->expert_first);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			G::Tag(),
			G::Name(),
			"official",
			Open,
			ManifestCheck,
			BindModel,
			Close,
			Step,
			Sideband,
			AttachKv,
			RecurrentCopy,
			ResetSlot,
			ProbeBuffers,
			Report
		};
		return &interface_table;
	}
};
