#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/norm_cluster.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_expert_planes.h"
#include "sparkpipe/spark_glm5_next_index_cp.h"
#include "sparkpipe/spark_glm5_next_kv_shard.h"
#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_glm5_next_stagepack.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "sparkpipe/spark_weightd_manifest.h"

#define G5N_HIDDEN SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define G5N_LAYERS SPARK_GLM5_NEXT_MODEL_LAYER_COUNT
#define G5N_VOCAB SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT
#define G5N_EPS SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON
#define G5N_SWIGLU_LIMIT SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT
#define G5N_FIRST_ROUTED SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER
#define G5N_HEADS SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT
#define G5N_LATENT SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION
#define G5N_NOPE SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION
#define G5N_VALUE SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION
#define G5N_QA SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION
#define G5N_QK_SCALE SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE
#define G5N_SELECTED SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K
#define G5N_INDEX_HEADS SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT
#define G5N_INDEX_DIM SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION
#define G5N_INDEX_QUERY SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION
#define G5N_INDEX_PACKED SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION
#define G5N_INDEX_WIDTH SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH
#define G5N_KPOOL SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL
#define G5N_KDA_HEADS SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT
#define G5N_KDA_KEY SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION
#define G5N_KDA_VALUE SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION
#define G5N_KDA_CONV SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL
#define G5N_KDA_LOW_RANK SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK
#define G5N_KDA_GATE_LOWER SPARK_GLM5_NEXT_MODEL_KDA_GATE_LOWER_BOUND
#define G5N_HC SPARK_GLM5_NEXT_MODEL_HC_MULT
#define G5N_HC_MIX SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION
#define G5N_HC_FLAT (G5N_HC * G5N_HIDDEN)
#define G5N_HC_ITERATIONS SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS
#define G5N_HC_EPS SPARK_GLM5_NEXT_MODEL_HC_EPSILON
#define G5N_EXPERTS SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT
#define G5N_TOP_K SPARK_GLM5_NEXT_MODEL_MOE_TOP_K
#define G5N_EXPERT_INTER SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION
#define G5N_DENSE_INTER SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define G5N_ROUTED_SCALE SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR
#define G5N_PAGE_SLOTS SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS
#define G5N_DSA_LAYERS SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT
#define G5N_KDA_LAYERS SPARK_GLM5_NEXT_MODEL_KDA_LAYER_COUNT

#define G5N_THREADS 256u
#define G5N_TILE_N 128u
#define G5N_STAGES 2u
#define G5N_WARPS 8u
#define G5N_STREAM_EXPERT_ROWS 256u
#define G5N_KDA_DELTA_COLUMNS 16u
#define G5N_HC_CLUSTER 8u
#define G5N_HC_SLICE (G5N_HC_FLAT / G5N_HC_CLUSTER)
#define G5N_HC_PRE_SLICE (G5N_HIDDEN / G5N_HC_CLUSTER)
#define G5N_HC_VECTORS (G5N_HC_SLICE / 4u / LM_WARP_LANES)
#define G5N_NO_ORDINAL 0xffffffffu

static_assert(SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION == 0u, "the latent slot carries no rope part");
static_assert(G5N_HC == 4u && G5N_HC_MIX == (2u + G5N_HC) * G5N_HC, "the HC site is specialised for four streams");
static_assert(G5N_HC_SLICE % (4u * LM_WARP_LANES) == 0u && G5N_HIDDEN % G5N_HC_CLUSTER == 0u, "HC slices are whole float4 warp spans");
static_assert(G5N_HIDDEN % LmBf16Format::kTileK == 0u && G5N_QA % LmBf16Format::kTileK == 0u, "projections cover whole BF16 K tiles");

struct Glm5NextKv
{
	static constexpr uint32_t kSlotBytes = SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	static constexpr uint32_t kPageSlots = G5N_PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = kSlotBytes * kPageSlots;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / kPageSlots; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % kPageSlots; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + kPageSlots - 1u) / kPageSlots; }
	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages) { return pages * (uint64_t)kPageBytes; }
};

struct Glm5NextIndexKv
{
	static constexpr uint32_t kSlotBytes = G5N_INDEX_PACKED * 2u;
	static constexpr uint32_t kPageSlots = G5N_PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = kSlotBytes * kPageSlots;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / kPageSlots; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % kPageSlots; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + kPageSlots - 1u) / kPageSlots; }
	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages) { return pages * (uint64_t)kPageBytes; }
};

typedef struct Glm5NextLayerWeights
{
	const void *attn_norm;
	const void *q_a;
	const void *q_a_norm;
	const void *q_b;
	const void *kv_a;
	const void *kv_a_norm;
	const void *kv_b_key;
	const void *kv_b_value;
	const void *attn_output;
	const void *post_attn_norm;
	const void *index_q;
	const void *index_k;
	const void *index_head;
	const void *index_norm_weight;
	const void *index_norm_bias;
	const void *index_compress_ape;
	const void *index_compress_gate;
	const void *dense_gate_up;
	const void *dense_down;
	const void *router;
	const float *router_correction;
	const void *shared_gate_up;
	const void *shared_down;
	const void *kda_qkv_beta;
	const void *kda_decay_gate_down;
	const void *kda_decay_up;
	const void *kda_gate_up;
	const void *kda_q_conv;
	const void *kda_k_conv;
	const void *kda_v_conv;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	const void *kda_out_norm;
	const void *kda_out;
	const void *hc_attn_fn;
	const void *hc_attn_base;
	const void *hc_attn_scale;
	const void *hc_ffn_fn;
	const void *hc_ffn_base;
	const void *hc_ffn_scale;
	uint64_t expert_w1_offset;
	uint64_t expert_w1_scale_offset;
	uint64_t expert_w2_offset;
	uint64_t expert_w2_scale_offset;
	uint64_t seen;
	uint32_t kda_ordinal;
	uint32_t dsa_ordinal;
} Glm5NextLayerWeights;

struct Glm5NextLayerBuffers
{
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	const Glm5NextLayerWeights *w;
	uint32_t layer_index;
	uint32_t tp_rank;
	uint32_t attn_heads;
	uint32_t q_b_rows;
	uint32_t attn_output_columns;
	uint32_t dense_gate_up_rows;
	uint32_t dense_intermediate;
	uint32_t expert_w1_rows;
	uint32_t expert_intermediate;
	uint32_t shared_gate_up_rows;
	uint32_t shared_intermediate;
	uint32_t kda_heads;
	const void *kv_b_key;
	const void *kv_b_value;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	uint16_t *hidden_bf16;
	uint16_t *normed_bf16;
	uint16_t *q_compressed_bf16;
	uint16_t *q_bf16;
	uint16_t *query_latent_bf16;
	uint16_t *index_query_bf16;
	uint16_t *index_key_bf16;
	uint16_t *index_head_weight_bf16;
	uint16_t *index_gate_bf16;
	uint16_t *index_packed_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
	uint16_t *attention_value_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	float *router_logits;
	float *selection_scores;
	uint32_t *selected_pools;
	uint32_t *selected_positions;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint16_t *fused_qkvb_bf16;
	uint16_t *fused_decay_gate_bf16;
	uint16_t *kda_decay_latent_bf16;
	uint16_t *kda_beta_logit;
	uint16_t *kda_gate_bf16;
	uint16_t *kda_gate_latent_bf16;
	uint16_t *kda_decay_logit_bf16;
	float *kda_retention;
	float *kda_write_gate;
	uint8_t *kda_state_pool;
	uint32_t kda_state_slot_bytes;
	uint16_t *kda_q_window;
	uint16_t *kda_k_window;
	uint16_t *kda_v_window;
	const uint32_t *kda_state_index;
	const uint32_t *sequence_row_begin;
	const uint32_t *sequence_row_indices;
	float *hc_mixes_f32;
	float *hc_pre_f32;
	float *hc_post_f32;
	float *hc_comb_f32;
	uint16_t *hc_collapsed_bf16;
	uint16_t *hc_snapshot_bf16;
	LmKvShardView cache;
	LmKvShardView index_cache;
	uint16_t *shard_query_gathered_bf16;
	uint64_t shard_query_stride;
	float *shard_partials_f32;
	float *shard_partials_received_f32;
	uint64_t shard_partial_stride;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	uint32_t index_owner_rank;
	uint32_t index_owner_degree;
	float *index_local_scores;
	const float *index_gathered_scores;
};

static int32_t Glm5NextLinear(const uint16_t *activation, const void *weight, uint16_t *output, const uint32_t *row_offset,
	uint32_t *tile_prefix, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride,
	uint32_t output_column_offset, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	if ( activation == 0 || weight == 0 || output == 0 || row_offset == 0 || tile_prefix == 0 || rows == 0u ||
		input_dimension == 0u || output_dimension == 0u || multiprocessors == 0u )
		return LM_LAUNCH_ERR_SHAPE;
	status = LmSkinnyDense<LmBf16Format>(weight, activation, output, 0, rows, input_dimension, output_dimension, output_row_stride,
		output_column_offset, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = row_offset;
	gemm.group_tile_prefix = tile_prefix;
	gemm.output_bf16 = output;
	gemm.output_row_stride = output_row_stride;
	gemm.output_column_offset = output_column_offset;
	return LmGemmLaunch<LmBf16Format, G5N_TILE_N, LmBf16Format::kTileK, G5N_STAGES, G5N_WARPS>(&gemm, activation, weight, rows, rows,
		1u, 1u, input_dimension, output_dimension, multiprocessors, false, stream);
}

static int32_t Glm5NextLaunched(void)
{
	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextIndexPackKernel(const uint16_t *__restrict__ key_bf16,
	const uint16_t *__restrict__ gate_bf16, uint16_t *__restrict__ packed_bf16, uint32_t dimension)
{
	const uint32_t row = blockIdx.x;
	const uint64_t base = (uint64_t)row * (2u * dimension + 1u);
	for ( uint32_t index = threadIdx.x; index < dimension; index += THREADS )
	{
		packed_bf16[base + index] = key_bf16[(uint64_t)row * dimension + index];
		packed_bf16[base + dimension + index] = gate_bf16[(uint64_t)row * dimension + index];
	}
	if ( threadIdx.x == 0u )
		packed_bf16[base + 2u * dimension] = 0x3F80u;
}

template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextPoolScoreKernel(const uint16_t *__restrict__ query_bf16,
	const uint16_t *__restrict__ head_weight_bf16, LmKvShardView index_cache, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions, const float *__restrict__ compress_ape_f32,
	uint32_t pools, uint32_t owner_rank, uint32_t owner_degree, uint32_t local_stride, float softmax_scale, float head_scale,
	float *__restrict__ pool_scores)
{
	__shared__ float pool_key[DIM];
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t local = blockIdx.x, pool = SparkGlm5NextIndexCpGlobalPool(local, owner_rank, owner_degree), row = blockIdx.y;
	const uint32_t sequence = sequence_of_row[row];
	const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
	const uint32_t first = pool * KPOOL;
	uint32_t index, slot_in_pool, head;
	if ( local >= local_stride )
		return;
	if ( pool >= pools || first + KPOOL > context )
	{
		if ( threadIdx.x == 0u )
			pool_scores[(uint64_t)row * local_stride + local] = -INFINITY;
		return;
	}
	for ( index = threadIdx.x; index < DIM; index += THREADS )
	{
		float maximum = -INFINITY, sum = 0.0f, logits[KPOOL], mix[KPOOL], key = 0.0f;
		for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
		{
			const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Glm5NextIndexKv>(index_cache, sequence, first + slot_in_pool, row, LM_KV_ACCESS_READ);
			if ( slot == 0 )
				return;
			logits[slot_in_pool] = LmBf16ToFloat(slot[DIM + index]) + compress_ape_f32[slot_in_pool * DIM + index];
			maximum = fmaxf(maximum, logits[slot_in_pool]);
		}
		for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
			if ( logits[slot_in_pool] != -INFINITY )
				sum += __expf(logits[slot_in_pool] - maximum);
		for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
			mix[slot_in_pool] = logits[slot_in_pool] == -INFINITY ? 0.0f : __expf(logits[slot_in_pool] - maximum) / fmaxf(sum, 1.0e-20f);
		for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
		{
			if ( mix[slot_in_pool] == 0.0f )
				continue;
			const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Glm5NextIndexKv>(index_cache, sequence, first + slot_in_pool, row, LM_KV_ACCESS_READ);
			if ( slot == 0 )
				return;
			key += mix[slot_in_pool] * LmBf16ToFloat(slot[index]);
		}
		pool_key[index] = key;
	}
	__syncthreads();
	float total = 0.0f;
	for ( head = 0u; head < HEADS; ++head )
	{
		float score = 0.0f;
		for ( index = threadIdx.x; index < DIM; index += THREADS )
			score += LmBf16ToFloat(query_bf16[((uint64_t)row * HEADS + head) * DIM + index]) * pool_key[index];
		score = LmBlockSum<THREADS>(score, reduction) * softmax_scale;
		__syncthreads();
		if ( score < 0.0f )
			score = 0.0f;
		if ( threadIdx.x == 0u )
			reduction[0] = score * LmBf16ToFloat(head_weight_bf16[(uint64_t)row * HEADS + head]) * head_scale;
		__syncthreads();
		total += reduction[0];
	}
	if ( threadIdx.x == 0u )
		pool_scores[(uint64_t)row * local_stride + local] = total;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextPoolPermuteKernel(const float *__restrict__ gathered, float *__restrict__ pool_scores,
	uint32_t pools, uint32_t local_stride, uint64_t peer_stride, uint32_t degree)
{
	const uint32_t row = blockIdx.x;
	for ( uint32_t pool = threadIdx.x; pool < pools; pool += THREADS )
		pool_scores[(uint64_t)row * pools + pool] = gathered[(uint64_t)SparkGlm5NextIndexCpOwner(pool, degree) * peer_stride +
			(uint64_t)row * local_stride + SparkGlm5NextIndexCpLocalPool(pool, degree)];
}

template<uint32_t THREADS, uint32_t KPOOL, uint32_t TOPK, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextPoolExpandKernel(const uint32_t *__restrict__ selected_pools,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions,
	uint32_t *__restrict__ selected_positions, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	const uint32_t sequence = sequence_of_row[row];
	const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
	const uint32_t pooled = context - context % KPOOL;
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += THREADS )
	{
		uint32_t position = 0xFFFFFFFFu;
		if ( index < TOPK )
		{
			position = selected_pools[(uint64_t)row * (TOPK / KPOOL) + index / KPOOL] * KPOOL + index % KPOOL;
			if ( position >= pooled )
				position = 0xFFFFFFFFu;
		}
		else if ( pooled + index - TOPK < context )
			position = pooled + index - TOPK;
		selected_positions[(uint64_t)row * WIDTH + index] = position;
	}
}

template<uint32_t THREADS, uint32_t LOW_RANK>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextSplitDecayGateKernel(const uint16_t *__restrict__ fused_bf16,
	uint16_t *__restrict__ decay_latent_bf16, uint16_t *__restrict__ gate_latent_bf16, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	for ( uint32_t index = threadIdx.x; index < LOW_RANK; index += THREADS )
	{
		decay_latent_bf16[(uint64_t)row * LOW_RANK + index] = fused_bf16[(uint64_t)row * 2u * LOW_RANK + index];
		gate_latent_bf16[(uint64_t)row * LOW_RANK + index] = fused_bf16[(uint64_t)row * 2u * LOW_RANK + LOW_RANK + index];
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextSplitQkvbKernel(const uint16_t *__restrict__ qkvb_bf16, uint16_t *__restrict__ query_bf16,
	uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint16_t *__restrict__ beta_bf16, uint32_t rows, uint32_t qk_dim,
	uint32_t v_dim, uint32_t heads, uint32_t fused_rows)
{
	const uint32_t row = blockIdx.x;
	const uint64_t fused = (uint64_t)row * fused_rows, dense = (uint64_t)row * qk_dim;
	uint32_t index;
	if ( row >= rows )
		return;
	for ( index = threadIdx.x; index < qk_dim; index += THREADS )
		query_bf16[dense + index] = qkvb_bf16[fused + index];
	for ( index = threadIdx.x; index < qk_dim; index += THREADS )
		key_bf16[dense + index] = qkvb_bf16[fused + qk_dim + index];
	for ( index = threadIdx.x; index < v_dim; index += THREADS )
		value_bf16[(uint64_t)row * v_dim + index] = qkvb_bf16[fused + 2u * qk_dim + index];
	for ( index = threadIdx.x; index < heads; index += THREADS )
		beta_bf16[(uint64_t)row * heads + index] = qkvb_bf16[fused + 2u * qk_dim + v_dim + index];
}

__global__ static void Glm5NextExpandStreamsKernel(const uint16_t *__restrict__ hidden, uint16_t *__restrict__ streams, uint32_t rows)
{
	const uint32_t row = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows || element >= G5N_HIDDEN )
		return;
	const uint16_t value = hidden[(uint64_t)row * G5N_HIDDEN + element];
	for ( uint32_t stream = 0u; stream < G5N_HC; ++stream )
		streams[((uint64_t)row * G5N_HC + stream) * G5N_HIDDEN + element] = value;
}

__global__ static void Glm5NextStreamsMeanKernel(const uint16_t *__restrict__ streams_bf16, uint16_t *__restrict__ reduced_bf16, uint32_t row_count)
{
	const uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	for ( uint32_t element = threadIdx.x; element < G5N_HIDDEN; element += blockDim.x )
	{
		float value = 0.0f;
		for ( uint32_t stream = 0u; stream < G5N_HC; ++stream )
			value += LmBf16ToFloat(streams_bf16[((uint64_t)row * G5N_HC + stream) * G5N_HIDDEN + element]);
		reduced_bf16[(uint64_t)row * G5N_HIDDEN + element] = LmFloatToBf16(value / (float)G5N_HC);
	}
}

__global__ static void Glm5NextDenseOffsetsKernel(uint32_t *offsets, uint32_t rows)
{
	if ( threadIdx.x == 0u )
	{
		offsets[0] = 0u;
		offsets[1] = rows;
	}
}

typedef struct Glm5NextHcShared
{
	float staged[G5N_HC_SLICE];
	float partial[G5N_HC_CLUSTER][G5N_HC_MIX + 1u];
	float mixes[G5N_HC_MIX + 1u];
	float reduction[G5N_THREADS / LM_WARP_LANES];
	float pre[G5N_HC];
	float post[G5N_HC];
} Glm5NextHcShared;

static __device__ __forceinline__ float Glm5NextHcRowSum(float value, uint32_t lane)
{
	const uint32_t row = lane & 12u;
	float total = 0.0f;
#pragma unroll
	for ( uint32_t j = 0u; j < G5N_HC; ++j )
		total += __shfl_sync(0xffffffffu, value, row + j);
	return total;
}

static __device__ __forceinline__ float Glm5NextHcColumnSum(float value, uint32_t lane)
{
	const uint32_t column = lane & 3u;
	float total = 0.0f;
#pragma unroll
	for ( uint32_t i = 0u; i < G5N_HC; ++i )
		total += __shfl_sync(0xffffffffu, value, column + i * G5N_HC);
	return total;
}

static __device__ __forceinline__ float Glm5NextHcRowMax(float value, uint32_t lane)
{
	const uint32_t row = lane & 12u;
	float maximum = -3.0e38f;
#pragma unroll
	for ( uint32_t j = 0u; j < G5N_HC; ++j )
		maximum = fmaxf(maximum, __shfl_sync(0xffffffffu, value, row + j));
	return maximum;
}

static __device__ void Glm5NextHcGatesWarp(const float *mixes, const float *scale3, const float *base, float *pre, float *post)
{
	const uint32_t lane = threadIdx.x % LM_WARP_LANES;
	if ( lane < G5N_HC )
	{
		pre[lane] = 1.0f / (1.0f + __expf(-(mixes[lane] * scale3[0] + base[lane]))) + G5N_HC_EPS;
		post[lane] = 2.0f / (1.0f + __expf(-(mixes[G5N_HC + lane] * scale3[1] + base[G5N_HC + lane])));
	}
}

static __device__ void Glm5NextHcCombWarp(const float *mixes, const float *scale3, const float *base, float *comb_out)
{
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, cell = lane & 15u;
	float value, total;
	value = mixes[2u * G5N_HC + cell] * scale3[2] + base[2u * G5N_HC + cell];
	value = __expf(value - Glm5NextHcRowMax(value, lane));
	total = Glm5NextHcRowSum(value, lane);
	value = value / total + G5N_HC_EPS;
	for ( uint32_t iteration = 0u; iteration < G5N_HC_ITERATIONS; ++iteration )
	{
		if ( iteration != 0u )
		{
			total = Glm5NextHcRowSum(value, lane);
			value /= total + G5N_HC_EPS;
		}
		total = Glm5NextHcColumnSum(value, lane);
		value /= total + G5N_HC_EPS;
	}
	if ( lane < G5N_HC * G5N_HC )
		comb_out[cell] = value;
}

static __device__ float Glm5NextHcStage(Glm5NextHcShared *shared, const uint16_t *slice)
{
	float total = 0.0f;
	for ( uint32_t element = threadIdx.x; element < G5N_HC_SLICE; element += G5N_THREADS )
	{
		const float value = LmBf16ToFloat(slice[element]);
		shared->staged[element] = value;
		total += value * value;
	}
	return LmBlockSum<G5N_THREADS>(total, shared->reduction);
}

static __device__ void Glm5NextHcDot(Glm5NextHcShared *shared, const float *fn, uint32_t offset, uint32_t slice)
{
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	float4 weight[G5N_HC_VECTORS], value;
	for ( uint32_t mix = warp; mix < G5N_HC_MIX; mix += G5N_THREADS / LM_WARP_LANES )
	{
		float accumulator = 0.0f;
#pragma unroll
		for ( uint32_t vector = 0u; vector < G5N_HC_VECTORS; vector++ )
			weight[vector] = __ldcs((const float4 *)(fn + (uint64_t)mix * G5N_HC_FLAT + offset) + lane + vector * LM_WARP_LANES);
#pragma unroll
		for ( uint32_t vector = 0u; vector < G5N_HC_VECTORS; vector++ )
		{
			value = ((const float4 *)shared->staged)[lane + vector * LM_WARP_LANES];
			accumulator = fmaf(weight[vector].x, value.x, accumulator);
			accumulator = fmaf(weight[vector].y, value.y, accumulator);
			accumulator = fmaf(weight[vector].z, value.z, accumulator);
			accumulator = fmaf(weight[vector].w, value.w, accumulator);
		}
		for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
			accumulator += __shfl_xor_sync(0xffffffffu, accumulator, step);
		if ( lane == 0u )
			shared->partial[slice][mix] = accumulator;
	}
}

static __device__ void Glm5NextHcScatter(cooperative_groups::cluster_group &cluster, Glm5NextHcShared *shared, uint32_t slice)
{
	for ( uint32_t index = threadIdx.x; index < G5N_HC_CLUSTER * (G5N_HC_MIX + 1u); index += G5N_THREADS )
	{
		const uint32_t rank = index / (G5N_HC_MIX + 1u), mix = index % (G5N_HC_MIX + 1u);
		if ( rank != slice )
			cluster.map_shared_rank(shared->partial[slice], rank)[mix] = shared->partial[slice][mix];
	}
}

static __device__ void Glm5NextHcFinish(Glm5NextHcShared *shared, const float *scale3, const float *base, float *mixes, float *pre, float *post, uint32_t slice)
{
	if ( threadIdx.x <= G5N_HC_MIX )
	{
		float total = 0.0f;
		for ( uint32_t rank = 0u; rank < G5N_HC_CLUSTER; rank++ )
			total += shared->partial[rank][threadIdx.x];
		shared->mixes[threadIdx.x] = total;
	}
	__syncthreads();
	if ( threadIdx.x < LM_WARP_LANES )
	{
		const float inverse = rsqrtf(shared->mixes[G5N_HC_MIX] / (float)G5N_HC_FLAT + G5N_EPS);
		if ( threadIdx.x < G5N_HC_MIX )
		{
			const float value = shared->mixes[threadIdx.x] * inverse;
			shared->mixes[threadIdx.x] = value;
			if ( slice == 0u )
				mixes[threadIdx.x] = value;
		}
		__syncwarp();
		Glm5NextHcGatesWarp(shared->mixes, scale3, base, shared->pre, slice == 0u ? post : shared->post);
		__syncwarp();
		if ( slice == 0u && threadIdx.x < G5N_HC )
			pre[threadIdx.x] = shared->pre[threadIdx.x];
	}
	__syncthreads();
}

static __device__ void Glm5NextHcCollapse(const float *pre, const uint16_t *streams, uint16_t *collapsed, uint16_t *snapshot, uint32_t first,
	uint32_t thread, uint32_t threads)
{
	for ( uint32_t element = first + thread; element < first + G5N_HC_PRE_SLICE; element += threads )
	{
		float value = 0.0f;
#pragma unroll
		for ( uint32_t stream = 0u; stream < G5N_HC; ++stream )
		{
			const uint64_t index = (uint64_t)stream * G5N_HIDDEN + element;
			const uint16_t raw = streams[index];
			snapshot[index] = raw;
			value += pre[stream] * LmBf16ToFloat(raw);
		}
		collapsed[element] = LmFloatToBf16(value);
	}
}

__global__ void __cluster_dims__(G5N_HC_CLUSTER, 1, 1) __launch_bounds__(G5N_THREADS) Glm5NextHcSiteKernel(const uint16_t *__restrict__ streams_bf16,
	const float *__restrict__ fn_f32, const float *__restrict__ scale3_f32, const float *__restrict__ base_f32, float *__restrict__ mixes_f32,
	float *__restrict__ pre_f32, float *__restrict__ post_f32, float *__restrict__ comb_f32, uint16_t *__restrict__ collapsed_bf16,
	uint16_t *__restrict__ snapshot_bf16)
{
	__shared__ Glm5NextHcShared shared;
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t slice = cluster.block_rank(), row = blockIdx.x / G5N_HC_CLUSTER;
	const uint16_t *streams = streams_bf16 + (uint64_t)row * G5N_HC_FLAT;
	float total;
	asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
	if ( threadIdx.x % LM_WARP_LANES == 0u )
		for ( uint32_t mix = threadIdx.x / LM_WARP_LANES; mix < G5N_HC_MIX; mix += G5N_THREADS / LM_WARP_LANES )
			LmPrefetchL2(fn_f32 + (uint64_t)mix * G5N_HC_FLAT + slice * G5N_HC_SLICE, G5N_HC_SLICE * sizeof(float));
	LmDependentWait();
	total = Glm5NextHcStage(&shared, streams + slice * G5N_HC_SLICE);
	Glm5NextHcDot(&shared, fn_f32, slice * G5N_HC_SLICE, slice);
	if ( threadIdx.x == 0u )
		shared.partial[slice][G5N_HC_MIX] = total;
	__syncthreads();
	asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
	Glm5NextHcScatter(cluster, &shared, slice);
	cluster.sync();
	Glm5NextHcFinish(&shared, scale3_f32, base_f32, mixes_f32 + (uint64_t)row * G5N_HC_MIX, pre_f32 + (uint64_t)row * G5N_HC,
		post_f32 + (uint64_t)row * G5N_HC, slice);
	if ( slice != 0u )
		Glm5NextHcCollapse(shared.pre, streams, collapsed_bf16 + (uint64_t)row * G5N_HIDDEN, snapshot_bf16 + (uint64_t)row * G5N_HC_FLAT,
			slice * G5N_HC_PRE_SLICE, threadIdx.x, G5N_THREADS);
	else if ( threadIdx.x >= LM_WARP_LANES )
		Glm5NextHcCollapse(shared.pre, streams, collapsed_bf16 + (uint64_t)row * G5N_HIDDEN, snapshot_bf16 + (uint64_t)row * G5N_HC_FLAT,
			0u, threadIdx.x - LM_WARP_LANES, G5N_THREADS - LM_WARP_LANES);
	else
		Glm5NextHcCombWarp(shared.mixes, scale3_f32, base_f32, comb_f32 + (uint64_t)row * G5N_HC * G5N_HC);
}

static int32_t Glm5NextHcSite(const Glm5NextLayerBuffers *b, const void *fn, const void *base, const void *scale, uint32_t rows, cudaStream_t stream)
{
	if ( fn == 0 || base == 0 || scale == 0 || ((uintptr_t)fn % 16u) != 0u )
		return LM_LAUNCH_ERR_SHAPE;
	if ( LM_LAUNCH_DEPENDENT((Glm5NextHcSiteKernel), rows * G5N_HC_CLUSTER, G5N_THREADS, 0, stream, (const uint16_t *)b->hidden_bf16,
		(const float *)fn, (const float *)scale, (const float *)base, b->hc_mixes_f32, b->hc_pre_f32, b->hc_post_f32, b->hc_comb_f32,
		b->hc_collapsed_bf16, b->hc_snapshot_bf16) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextHcPost(const Glm5NextLayerBuffers *b, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmHcPostBf16Kernel), dim3(rows, G5N_HIDDEN / G5N_THREADS), G5N_THREADS, 0, stream, b->attention_out_bf16, b->hc_snapshot_bf16,
		b->hc_post_f32, b->hc_comb_f32, b->hidden_bf16, rows, G5N_HC, G5N_HIDDEN);
	return Glm5NextLaunched();
}

static int32_t Glm5NextKda(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t sms, cudaStream_t stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t heads = b->kda_heads, rank_qk = heads * G5N_KDA_KEY, rank_v = heads * G5N_KDA_VALUE;
	const uint32_t fused = rank_qk * 2u + rank_v + heads;
	int32_t status;
	if ( b->kda_state_pool == 0 || b->kda_state_slot_bytes != rank_qk * G5N_KDA_VALUE * sizeof(float) || w->kda_qkv_beta == 0 ||
		w->kda_decay_gate_down == 0 || w->kda_decay_up == 0 || w->kda_gate_up == 0 || w->kda_q_conv == 0 || w->kda_k_conv == 0 ||
		w->kda_v_conv == 0 || w->kda_decay_bias == 0 || w->kda_head_log_scale == 0 || w->kda_out_norm == 0 || w->kda_out == 0 )
		return LM_LAUNCH_ERR_SHAPE;
	LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, b->hc_collapsed_bf16,
		(const uint16_t *)w->attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	{
		const LmSkinnyDenseTarget targets[2] = {
			{w->kda_qkv_beta, b->fused_qkvb_bf16, 0, fused, fused, 0u, LmScaleTensorNone()},
			{w->kda_decay_gate_down, b->fused_decay_gate_bf16, 0, 2u * G5N_KDA_LOW_RANK, 2u * G5N_KDA_LOW_RANK, 0u, LmScaleTensorNone()}};
		status = LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, b->normed_bf16, rows, G5N_HIDDEN, stream);
	}
	if ( status == LM_LAUNCH_ERR_SHAPE )
	{
		status = Glm5NextLinear(b->normed_bf16, w->kda_qkv_beta, b->fused_qkvb_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_HIDDEN,
			fused, fused, 0u, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b->normed_bf16, w->kda_decay_gate_down, b->fused_decay_gate_bf16, b->dense_row_offset, b->dense_tile_prefix,
				rows, G5N_HIDDEN, 2u * G5N_KDA_LOW_RANK, 2u * G5N_KDA_LOW_RANK, 0u, sms, stream);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((Glm5NextSplitQkvbKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->fused_qkvb_bf16, b->q_bf16, b->kv_slot_bf16,
		b->gate_up_bf16, b->kda_beta_logit, rows, rank_qk, rank_v, heads, fused);
	LM_LAUNCH((Glm5NextSplitDecayGateKernel<G5N_THREADS, G5N_KDA_LOW_RANK>), rows, G5N_THREADS, 0, stream, b->fused_decay_gate_bf16,
		b->kda_decay_latent_bf16, b->kda_gate_latent_bf16, rows);
	{
		LmCausalConvStreams<uint16_t> conv;
		conv.window[0] = b->kda_q_window;
		conv.window[1] = b->kda_k_window;
		conv.window[2] = b->kda_v_window;
		conv.input_bf16[0] = conv.output_bf16[0] = b->q_bf16;
		conv.input_bf16[1] = conv.output_bf16[1] = b->kv_slot_bf16;
		conv.input_bf16[2] = conv.output_bf16[2] = b->gate_up_bf16;
		conv.weight[0] = (const uint16_t *)w->kda_q_conv;
		conv.weight[1] = (const uint16_t *)w->kda_k_conv;
		conv.weight[2] = (const uint16_t *)w->kda_v_conv;
		conv.channels[0] = conv.channels[1] = rank_qk;
		conv.channels[2] = rank_v;
		LM_LAUNCH((LmCausalConvStreamsKernel<G5N_THREADS, G5N_KDA_CONV, LM_CONV_SWISH, uint16_t>),
			dim3(sequences, ((rank_qk > rank_v ? rank_qk : rank_v) + G5N_THREADS - 1u) / G5N_THREADS, LM_CAUSAL_CONV_STREAMS), G5N_THREADS, 0, stream,
			conv, b->kda_state_index, b->sequence_row_begin, (const uint32_t *)0, sequences, commit, b->sequence_row_indices);
	}
	status = Glm5NextLinear(b->kda_decay_latent_bf16, w->kda_decay_up, b->kda_decay_logit_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
		G5N_KDA_LOW_RANK, rank_qk, rank_qk, 0u, sms, stream);
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextLinear(b->kda_gate_latent_bf16, w->kda_gate_up, b->kda_gate_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
			G5N_KDA_LOW_RANK, rank_v, rank_v, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmBoundedDecayKernel<G5N_THREADS, G5N_KDA_KEY>), dim3(rows, heads), G5N_THREADS, 0, stream, b->kda_decay_logit_bf16,
		w->kda_decay_bias, w->kda_head_log_scale, b->kda_retention, heads, G5N_KDA_GATE_LOWER, rows);
	LM_LAUNCH((LmBf16SigmoidRowsKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->kda_beta_logit, b->kda_write_gate, heads);
	LM_LAUNCH((LmDeltaRuleKernel<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float, G5N_KDA_DELTA_COLUMNS>),
		dim3(sequences, heads, G5N_KDA_VALUE / G5N_KDA_DELTA_COLUMNS), G5N_THREADS, G5N_KDA_KEY * G5N_KDA_DELTA_COLUMNS * sizeof(float), stream,
		b->kda_state_pool, b->kda_state_slot_bytes, b->kda_state_index, b->sequence_row_begin, (const uint32_t *)0, b->q_bf16, b->kv_slot_bf16,
		b->gate_up_bf16, b->kda_retention, b->kda_write_gate, b->attention_out_bf16, heads, 1u, sequences, commit, b->sequence_row_indices);
	LM_LAUNCH((LmRmsNormSigmoidGateKernel<G5N_THREADS>), dim3((uint64_t)rows * heads), G5N_THREADS, 0, stream, b->attention_out_bf16,
		b->kda_gate_bf16, (const float *)w->kda_out_norm, b->attention_out_bf16, G5N_KDA_VALUE, G5N_EPS);
	LM_LAUNCH((LmCopyRowsKernel<G5N_THREADS>), dim3((rank_v + G5N_THREADS - 1u) / G5N_THREADS, rows), G5N_THREADS, 0, stream,
		b->attention_out_bf16, b->kv_slot_bf16, rows, rank_v);
	return Glm5NextLinear(b->kv_slot_bf16, w->kda_out, b->attention_out_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, rank_v,
		G5N_HIDDEN, G5N_HIDDEN, 0u, sms, stream);
}

static int32_t Glm5NextIndexScore(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, uint32_t projected, cudaStream_t stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t degree = b->index_owner_degree, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	int32_t status;
	if ( w->index_q == 0 || w->index_k == 0 || w->index_head == 0 || w->index_norm_weight == 0 || w->index_norm_bias == 0 ||
		w->index_compress_gate == 0 || w->index_compress_ape == 0 || degree < 2u || b->index_owner_rank >= degree ||
		degree != b->index_cache.shard.degree || b->index_owner_rank != b->index_cache.shard.rank )
		return LM_LAUNCH_ERR_SHAPE;
	status = Glm5NextLinear(b->q_compressed_bf16, w->index_q, b->index_query_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_QA,
		G5N_INDEX_QUERY, G5N_INDEX_QUERY, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( projected == 0u )
	{
		status = Glm5NextLinear(b->normed_bf16, w->index_k, b->index_key_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_HIDDEN,
			G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b->normed_bf16, w->index_compress_gate, b->index_gate_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
				G5N_HIDDEN, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, sms, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
	}
	LM_LAUNCH((LmLayerNormKernel<G5N_THREADS, uint16_t>), rows, G5N_THREADS, (G5N_INDEX_DIM + 8u) * sizeof(float), stream, b->index_key_bf16,
		(const uint16_t *)w->index_norm_weight, (const uint16_t *)w->index_norm_bias, b->index_key_bf16, G5N_INDEX_DIM, G5N_INDEX_DIM,
		SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON);
	if ( projected == 0u )
	{
		status = Glm5NextLinear(b->normed_bf16, w->index_head, b->index_head_weight_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
			G5N_HIDDEN, G5N_INDEX_HEADS, G5N_INDEX_HEADS, 0u, sms, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
	}
	LM_LAUNCH((Glm5NextIndexPackKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->index_key_bf16, b->index_gate_bf16, b->index_packed_bf16,
		G5N_INDEX_DIM);
	LM_LAUNCH((LmKvShardStoreKernel<Glm5NextIndexKv, G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->index_cache, b->index_packed_bf16,
		b->sequence_of_row, b->positions, rows, G5N_INDEX_PACKED);
	LM_LAUNCH((Glm5NextPoolScoreKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS>), dim3(local_stride, rows), G5N_THREADS, 0, stream,
		b->index_query_bf16, b->index_head_weight_bf16, b->index_cache, b->sequence_of_row, b->context_length, b->positions,
		(const float *)w->index_compress_ape, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
		SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, b->index_local_scores);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionHead(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	const Glm5NextLayerWeights *w = b->w;
	uint32_t projected;
	int32_t status;
	if ( w->attn_norm == 0 || w->q_a == 0 || w->q_a_norm == 0 || w->q_b == 0 || w->kv_a == 0 || w->kv_a_norm == 0 || b->kv_b_key == 0 ||
		b->kv_b_value == 0 || w->attn_output == 0 )
		return LM_LAUNCH_ERR_SHAPE;
	LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, b->hc_collapsed_bf16,
		(const uint16_t *)w->attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	{
		const LmSkinnyDenseTarget targets[4] = {
			{w->q_a, b->q_compressed_bf16, 0, G5N_QA, G5N_QA, 0u, LmScaleTensorNone()},
			{w->index_k, b->index_key_bf16, 0, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, LmScaleTensorNone()},
			{w->index_compress_gate, b->index_gate_bf16, 0, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, LmScaleTensorNone()},
			{w->index_head, b->index_head_weight_bf16, 0, G5N_INDEX_HEADS, G5N_INDEX_HEADS, 0u, LmScaleTensorNone()}};
		status = LmSkinnyDenseMulti<LmBf16Format>(targets, 4u, b->normed_bf16, rows, G5N_HIDDEN, stream);
	}
	projected = status == LM_LAUNCH_OK ? 1u : 0u;
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = Glm5NextLinear(b->normed_bf16, w->q_a, b->q_compressed_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_HIDDEN, G5N_QA,
			G5N_QA, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, b->q_compressed_bf16,
		(const uint16_t *)w->q_a_norm, b->q_compressed_bf16, G5N_QA, G5N_QA, G5N_EPS);
	return Glm5NextIndexScore(b, rows, context, sms, projected, stream);
}

static int32_t Glm5NextIndexSelect(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, cudaStream_t stream)
{
	const uint32_t degree = b->index_owner_degree, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	LM_LAUNCH((Glm5NextPoolPermuteKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->index_gathered_scores, b->selection_scores, pools,
		local_stride, (uint64_t)SparkGlm5NextIndexCpGatherSequences(rows, local_stride) * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS, degree);
	LmTopkExactRowsLaunch<G5N_THREADS>(b->selection_scores, rows, pools, G5N_SELECTED / G5N_KPOOL, b->selected_pools, stream);
	LM_LAUNCH((Glm5NextPoolExpandKernel<G5N_THREADS, G5N_KPOOL, G5N_SELECTED, G5N_INDEX_WIDTH>), rows, G5N_THREADS, 0, stream, b->selected_pools,
		b->sequence_of_row, b->context_length, b->positions, b->selected_positions, rows);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionTail(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	const Glm5NextLayerWeights *w = b->w;
	int32_t status = Glm5NextIndexSelect(b, rows, context, stream);
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextLinear(b->q_compressed_bf16, w->q_b, b->q_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_QA, b->q_b_rows,
			b->q_b_rows, 0u, sms, stream);
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextLinear(b->normed_bf16, w->kv_a, b->kv_slot_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_HIDDEN, G5N_LATENT,
			G5N_LATENT, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, b->kv_slot_bf16,
		(const uint16_t *)w->kv_a_norm, b->kv_slot_bf16, G5N_LATENT, G5N_LATENT, G5N_EPS);
	if ( LmPerHeadProjectRowsLaunch<G5N_THREADS, G5N_NOPE, G5N_LATENT, G5N_NOPE, 0u>(b->q_bf16, (const uint16_t *)b->kv_b_key, b->query_latent_bf16,
		b->attn_heads, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	LM_LAUNCH((LmKvShardStoreKernel<Glm5NextKv, G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->cache, b->kv_slot_bf16, b->sequence_of_row,
		b->positions, rows, G5N_LATENT);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionPartial(const Glm5NextLayerBuffers *b, uint32_t rows, cudaStream_t stream)
{
	return LmLatentShardPartialLaunch<Glm5NextKv, LmKvShardView, G5N_LATENT, 0u>(b->cache, b->shard_query_gathered_bf16, b->shard_query_stride,
		b->attn_heads, b->sequence_of_row, b->context_length, b->positions, b->selected_positions, G5N_INDEX_WIDTH, G5N_SELECTED, G5N_QK_SCALE,
		b->shard_partials_f32, b->shard_partial_stride, rows, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t Glm5NextAttentionMerge(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	if ( LmLatentShardMergeLaunch<G5N_LATENT>(b->shard_partials_received_f32, b->shard_partial_stride, b->cache.shard.degree, b->attn_heads,
		b->attention_latent_bf16, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	if ( LmPerHeadProjectRowsLaunch<G5N_THREADS, G5N_LATENT, G5N_VALUE>(b->attention_latent_bf16, (const uint16_t *)b->kv_b_value,
		b->attention_value_bf16, b->attn_heads, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return Glm5NextLinear(b->attention_value_bf16, b->w->attn_output, b->attention_out_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
		b->attn_output_columns, G5N_HIDDEN, G5N_HIDDEN, 0u, sms, stream);
}

static int32_t Glm5NextDenseMlp(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	const Glm5NextLayerWeights *w = b->w;
	int32_t status;
	if ( w->post_attn_norm == 0 || w->dense_gate_up == 0 || w->dense_down == 0 )
		return LM_LAUNCH_ERR_SHAPE;
	LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, b->hc_collapsed_bf16,
		(const uint16_t *)w->post_attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	status = Glm5NextLinear(b->normed_bf16, w->dense_gate_up, b->gate_up_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_HIDDEN,
		b->dense_gate_up_rows, b->dense_gate_up_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->dense_intermediate,
		G5N_SWIGLU_LIMIT);
	return Glm5NextLinear(b->intermediate_bf16, w->dense_down, b->attention_out_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
		b->dense_intermediate, G5N_HIDDEN, G5N_HIDDEN, 0u, sms, stream);
}

static int32_t Glm5NextMoeRoute(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t packed_rows = rows * G5N_TOP_K;
	LmGemmArguments gemm;
	int32_t status;
	if ( w->post_attn_norm == 0 || w->router == 0 || w->router_correction == 0 || w->shared_gate_up == 0 || w->shared_down == 0 )
		return LM_LAUNCH_ERR_SHAPE;
	LmBf16RmsNormLaunch<G5N_THREADS>(rows, stream, b->hc_collapsed_bf16,
		(const uint16_t *)w->post_attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	status = LmSkinnyDense<LmBf16Format>(w->router, b->normed_bf16, 0, b->router_logits, rows, G5N_HIDDEN, G5N_EXPERTS, 0u, 0u, stream);
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_f32 = b->router_logits;
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmGemmLaunch<LmBf16Format, G5N_TILE_N, LmBf16Format::kTileK, G5N_STAGES, G5N_WARPS>(&gemm, b->normed_bf16, w->router, rows, rows,
			1u, 1u, G5N_HIDDEN, G5N_EXPERTS, sms, false, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( LmTopkRouteLaunch<G5N_THREADS, G5N_TOP_K, true, LM_TOPK_SCORE_SIGMOID>(rows, b->router_logits, G5N_EXPERTS, b->route_expert,
		b->route_weight, w->router_correction, 0, G5N_ROUTED_SCALE, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	status = LmRouteBuild<G5N_THREADS, G5N_EXPERTS>(b->route_expert, rows, packed_rows, G5N_TOP_K, b->group_row_offset, b->route_packed_row,
		b->route_source_token, b->expert_w1_rows, G5N_HIDDEN, G5N_TILE_N, b->group_tile_prefix_w1, b->group_tile_prefix_w2, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	return Glm5NextLaunched();
}

template<uint32_t ExpertCodec>
static int32_t Glm5NextMoeExperts(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
	static_assert(ExpertFormat::kScaleGroup == 0u || (G5N_HIDDEN % ExpertFormat::kScaleGroup == 0u && G5N_EXPERT_INTER % ExpertFormat::kScaleGroup == 0u),
		"routed expert dimensions hold whole codec scale groups");
	const uint32_t packed_rows = rows * G5N_TOP_K;
	const dim3 grid((G5N_HIDDEN + G5N_THREADS - 1u) / G5N_THREADS, rows);
	LmGemmArguments gemm;
	int32_t status;
	if ( b->expert_w1_weight == 0 || b->expert_w2_weight == 0 ||
		(LmWeightCodec<ExpertCodec>::kScaleEncoding != LM_SCALE_ENCODING_NONE && (b->expert_w1_scale == 0 || b->expert_w2_scale == 0)) )
		return LM_LAUNCH_ERR_SHAPE;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(b->expert_w1_scale, G5N_EXPERTS, b->expert_w1_rows, G5N_HIDDEN);
	status = LM_LAUNCH_ERR_SHAPE;
	if constexpr ( LmStreamWeight<ExpertFormat>::kSupported )
		if ( rows >= G5N_STREAM_EXPERT_ROWS )
			status = LmStreamGemmGrouped<ExpertFormat>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
				b->route_source_token, G5N_EXPERTS, packed_rows, G5N_HIDDEN, b->expert_w1_rows, sms, stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyExperts<ExpertFormat>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->route_expert,
			b->route_packed_row, packed_rows, G5N_TOP_K, 0u, G5N_HIDDEN, b->expert_w1_rows, stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyGroupedExperts<ExpertFormat>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
			b->route_source_token, G5N_EXPERTS, packed_rows, 0u, G5N_HIDDEN, b->expert_w1_rows, stream);
	gemm.prefix_built = 1u;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.source_row_map = b->route_source_token;
	gemm.source_row_count = rows;
	gemm.output_bf16 = b->gate_up_bf16;
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmGemmWeightOnlyIndirectLaunch<ExpertFormat, G5N_TILE_N, G5N_STAGES, G5N_WARPS>(&gemm, b->normed_bf16, b->expert_w1_weight,
			packed_rows, rows, G5N_TOP_K, G5N_EXPERTS, G5N_HIDDEN, b->expert_w1_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), packed_rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16,
		b->expert_intermediate, G5N_SWIGLU_LIMIT);
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(b->expert_w2_scale, G5N_EXPERTS, G5N_HIDDEN, b->expert_intermediate);
	status = LM_LAUNCH_ERR_SHAPE;
	if constexpr ( LmStreamWeight<ExpertFormat>::kSupported )
		if ( rows >= G5N_STREAM_EXPERT_ROWS )
			status = LmStreamGemmGrouped<ExpertFormat>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16,
				b->group_row_offset, 0, G5N_EXPERTS, packed_rows, b->expert_intermediate, G5N_HIDDEN, sms, stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyExperts<ExpertFormat>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->route_expert,
			b->route_packed_row, packed_rows, G5N_TOP_K, 1u, b->expert_intermediate, G5N_HIDDEN, stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyGroupedExperts<ExpertFormat>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16,
			b->group_row_offset, b->route_source_token, G5N_EXPERTS, packed_rows, 1u, b->expert_intermediate, G5N_HIDDEN, stream);
	gemm.prefix_built = 1u;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.output_bf16 = b->expert_out_bf16;
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmGemmWeightOnlyLaunch<ExpertFormat, G5N_TILE_N, G5N_STAGES, G5N_WARPS>(&gemm, b->intermediate_bf16, b->expert_w2_weight,
			packed_rows, rows, G5N_TOP_K, G5N_EXPERTS, b->expert_intermediate, G5N_HIDDEN, sms, true, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmMoeFinalizeKernel<G5N_THREADS>), grid, G5N_THREADS, 0, stream, b->expert_out_bf16, b->route_packed_row, b->route_weight,
		b->attention_out_bf16, rows, G5N_TOP_K, G5N_HIDDEN);
	status = Glm5NextLinear(b->normed_bf16, b->w->shared_gate_up, b->gate_up_bf16, b->dense_row_offset, b->dense_tile_prefix, rows, G5N_HIDDEN,
		b->shared_gate_up_rows, b->shared_gate_up_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->shared_intermediate,
		G5N_SWIGLU_LIMIT);
	status = Glm5NextLinear(b->intermediate_bf16, b->w->shared_down, b->shared_out_bf16, b->dense_row_offset, b->dense_tile_prefix, rows,
		b->shared_intermediate, G5N_HIDDEN, G5N_HIDDEN, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmAddRowsKernel<G5N_THREADS>), grid, G5N_THREADS, 0, stream, b->attention_out_bf16, b->shared_out_bf16, b->attention_out_bf16,
		rows, G5N_HIDDEN);
	return Glm5NextLaunched();
}

#include "glm5_next_stage_model.cuh"
