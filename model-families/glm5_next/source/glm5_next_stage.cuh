#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/hc_site.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/project_chain.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/runner/stage_model_cuda.cuh"
#include "sparkpipe/spark_expert_planes.h"
#include "sparkpipe/spark_glm5_next_index_cp.h"
#include "sparkpipe/spark_glm5_next_kv_shard.h"
#include "sparkpipe/spark_glm5_next_family.h"
#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_glm5_next_stagepack.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_gpu.h"

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
#define G5N_STREAM_EXPERT_ROWS 512u
#define G5N_STREAM_EXPERT_W2_ROWS 64u
#define G5N_KDA_DELTA_COLUMNS 16u
#define G5N_POOL_BLOCK_POOLS 32u
#define G5N_POOL_BLOCK_ROWS 16u
#define G5N_POOL_ROWS_MIN 64u
#define G5N_NO_ORDINAL 0xffffffffu

static_assert(SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION == 0u, "the latent slot carries no rope part");
static_assert(G5N_HC == LM_HC_SITE_STREAMS && G5N_HC_MIX == LM_HC_SITE_MIX, "the HC site is specialised for four streams");
static_assert(G5N_THREADS == LM_HC_SITE_THREADS, "the HC site runs the stage block size");
static_assert(G5N_HIDDEN % LmBf16Format::kTileK == 0u && G5N_QA % LmBf16Format::kTileK == 0u, "projections cover whole BF16 K tiles");
static_assert(G5N_KDA_KEY == G5N_KDA_VALUE, "the convolution windows and replay slabs hold q, k and v rows of one width");

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
	uint32_t row_invariant;
	uint32_t prenormed;
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
	uint32_t *route_fault;
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
	uint16_t *kda_q_input;
	uint16_t *kda_k_input;
	uint16_t *kda_v_input;
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
	const uint32_t *slots;
	const uint32_t *fork_positions;
	const uint32_t *slot_offsets;
	const uint32_t *parent_rows;
	const uint32_t *state_loads;
	const uint32_t *state_saves;
	float *kda_snapshots;
	uint32_t index_owner_rank;
	uint32_t index_owner_degree;
	float *index_local_scores;
	const float *index_gathered_scores;
};

static int32_t Glm5NextSkinnyMulti(const Glm5NextLayerBuffers *b, const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *activation,
	uint32_t rows, uint32_t input_dimension, SparkGpuStream stream)
{
	return LmSkinnyDenseMulti<LmBf16Format>(targets, count, activation, rows, input_dimension, stream,
		rows > LM_SKINNY_ROWS_WIDE && b->row_invariant != 0u ? 1u : 0u);
}

static int32_t Glm5NextLinear(const Glm5NextLayerBuffers *b, const uint16_t *activation, const void *weight, uint16_t *output, uint32_t rows,
	uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	if ( multiprocessors == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, (float *)0, rows, input_dimension, output_dimension,
		b->row_invariant, multiprocessors, stream);
}

static int32_t Glm5NextLaunched(void)
{
	if ( !(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

template<uint32_t DIM, uint32_t KPOOL>
static __device__ __forceinline__ float Glm5NextPoolKeyElement(const uint16_t *const *slots, const float *__restrict__ compress_ape_f32, uint32_t index)
{
	float maximum = -INFINITY, sum = 0.0f, logits[KPOOL], weights[KPOOL], key = 0.0f;
	uint16_t slot_key[KPOOL];
	uint32_t slot_in_pool;
	for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
	{
		const uint16_t *slot = slots[slot_in_pool];
		slot_key[slot_in_pool] = slot[index];
		logits[slot_in_pool] = LmBf16ToFloat(slot[DIM + index]) + compress_ape_f32[slot_in_pool * DIM + index];
		maximum = fmaxf(maximum, logits[slot_in_pool]);
	}
	for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
	{
		if ( logits[slot_in_pool] != -INFINITY )
		{
			weights[slot_in_pool] = __expf(logits[slot_in_pool] - maximum);
			sum += weights[slot_in_pool];
		}
		else
			weights[slot_in_pool] = 0.0f;
	}
	const float denom = fmaxf(sum, 1.0e-20f);
	for ( slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
	{
		const float mix = weights[slot_in_pool] / denom;
		if ( mix != 0.0f )
			key += mix * LmBf16ToFloat(slot_key[slot_in_pool]);
	}
	return key;
}

static __device__ __forceinline__ uint32_t Glm5NextPoolGlobal(uint32_t local, uint32_t rank, uint32_t degree) { return degree <= 1u ? local : local * degree + rank; }
static __device__ __forceinline__ uint32_t Glm5NextPoolOwner(uint32_t pool, uint32_t degree) { return degree <= 1u ? 0u : pool % degree; }
static __device__ __forceinline__ uint32_t Glm5NextPoolLocal(uint32_t pool, uint32_t degree) { return degree <= 1u ? pool : pool / degree; }

static __device__ __forceinline__ void Glm5NextLoadBf16x4(const uint16_t *__restrict__ source, float values[4])
{
	const uint2 packed = *(const uint2 *)source;
	values[0] = __uint_as_float(packed.x << 16u);
	values[1] = __uint_as_float(packed.x & 0xffff0000u);
	values[2] = __uint_as_float(packed.y << 16u);
	values[3] = __uint_as_float(packed.y & 0xffff0000u);
}

static __device__ __forceinline__ float Glm5NextPoolHeadDot(const float q[4], float4 k, float softmax_scale, float head_weight, float head_scale)
{
	float score = fmaf(q[0], k.x, fmaf(q[1], k.y, fmaf(q[2], k.z, q[3] * k.w)));
	#pragma unroll
	for ( uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u )
		score += __shfl_down_sync(0xffffffffu, score, offset);
	score *= softmax_scale;
	return (score > 0.0f ? score : 0.0f) * head_weight * head_scale;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextIndexPackKernel(const uint16_t *__restrict__ key_bf16,
	const uint16_t *__restrict__ gate_bf16, uint16_t *__restrict__ packed_bf16, uint32_t dimension)
{
	const uint16_t *key_row = key_bf16 + (uint64_t)blockIdx.x * dimension;
	const uint16_t *gate_row = gate_bf16 + (uint64_t)blockIdx.x * dimension;
	uint16_t *packed_row = packed_bf16 + (uint64_t)blockIdx.x * (2u * dimension + 1u);
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(key_row) && LmIsAligned16(gate_row) && LmIsAligned16(packed_row) )
	{
		LmCopySliceBf16<THREADS>(key_row, packed_row, dimension);
		LmCopySliceBf16<THREADS>(gate_row, packed_row + dimension, dimension);
	}
	else
		for ( uint32_t index = threadIdx.x; index < 2u * dimension; index += THREADS )
			packed_row[index] = index < dimension ? key_row[index] : gate_row[index - dimension];
	if ( threadIdx.x == 0u )
		packed_row[2u * dimension] = 0x3F80u;
}

template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextPoolScoreKernel(const uint16_t *__restrict__ query_bf16,
	const uint16_t *__restrict__ head_weight_bf16, LmKvShardView index_cache, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions, const float *__restrict__ compress_ape_f32,
	uint32_t pools, uint32_t owner_rank, uint32_t owner_degree, uint32_t local_stride, float softmax_scale, float head_scale,
	float *__restrict__ pool_scores, const uint32_t *__restrict__ fork_positions = 0, const uint32_t *__restrict__ slot_offsets = 0,
	uint32_t dense_limit = 0u)
{
	static_assert(DIM == 4u * LM_WARP_LANES && HEADS % (THREADS / LM_WARP_LANES) == 0u, "each warp scores whole heads over four elements per lane");
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES, PER_WARP = HEADS / WARPS;
	__shared__ __align__(16) float pool_key[DIM];
	__shared__ const uint16_t *pool_slots[KPOOL];
	__shared__ uint32_t pool_ready;
	__shared__ float contribution[HEADS];
	const uint32_t local = blockIdx.x, pool = Glm5NextPoolGlobal(local, owner_rank, owner_degree), row = blockIdx.y;
	const uint32_t sequence = sequence_of_row[row];
	const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
	const uint32_t first = pool * KPOOL;
	if ( local >= local_stride || (dense_limit != 0u && context <= dense_limit) )
		return;
	if ( pool >= pools || first + KPOOL > context )
	{
		if ( threadIdx.x == 0u )
			pool_scores[(uint64_t)row * local_stride + local] = -INFINITY;
		return;
	}
	if ( threadIdx.x == 0u )
	{
		const uint32_t fork = fork_positions != 0 ? fork_positions[row] : 0xffffffffu;
		const uint32_t slot_offset = slot_offsets != 0 ? slot_offsets[row] : 0u;
		uint32_t ready = 1u;
		for ( uint32_t slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
		{
			const uint32_t pos = first + slot_in_pool;
			const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Glm5NextIndexKv>(index_cache, sequence,
				pos >= fork ? pos + slot_offset : pos, row, LM_KV_ACCESS_READ);
			pool_slots[slot_in_pool] = slot;
			if ( slot == 0 )
			{
				ready = 0u;
				break;
			}
		}
		pool_ready = ready;
	}
	__syncthreads();
	if ( pool_ready == 0u )
		return;
	for ( uint32_t index = threadIdx.x; index < DIM; index += THREADS )
		pool_key[index] = Glm5NextPoolKeyElement<DIM, KPOOL>(pool_slots, compress_ape_f32, index);
	__syncthreads();
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const float4 k = *(const float4 *)(pool_key + lane * 4u);
	for ( uint32_t slot = 0u; slot < PER_WARP; ++slot )
	{
		const uint32_t head = warp + slot * WARPS;
		float q[4];
		Glm5NextLoadBf16x4(query_bf16 + ((uint64_t)row * HEADS + head) * DIM + lane * 4u, q);
		const float contrib = Glm5NextPoolHeadDot(q, k, softmax_scale, LmBf16ToFloat(head_weight_bf16[(uint64_t)row * HEADS + head]), head_scale);
		if ( lane == 0u )
			contribution[head] = contrib;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float total = 0.0f;
		for ( uint32_t head = 0u; head < HEADS; ++head )
			total += contribution[head];
		pool_scores[(uint64_t)row * local_stride + local] = total;
	}
}

template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS, uint32_t POOLS, uint32_t ROWS>
__global__ __launch_bounds__(THREADS) void Glm5NextPoolScoreRowsKernel(const uint16_t *__restrict__ query_bf16,
	const uint16_t *__restrict__ head_weight_bf16, LmKvShardView index_cache, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions, const float *__restrict__ compress_ape_f32,
	uint32_t pools, uint32_t owner_rank, uint32_t owner_degree, uint32_t local_stride, float softmax_scale, float head_scale,
	float *__restrict__ pool_scores, uint32_t rows)
{
	static_assert(THREADS == 8u * LM_WARP_LANES && DIM == 4u * LM_WARP_LANES, "each warp scores whole heads over four elements per lane");
	static_assert(HEADS % (THREADS / LM_WARP_LANES) == 0u, "every warp scores the same number of heads");
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES, PER_WARP = HEADS / WARPS;
	__shared__ __align__(16) float pool_key[POOLS][DIM];
	__shared__ const uint16_t *pool_slots[POOLS][KPOOL];
	__shared__ uint32_t pool_ready[POOLS];
	__shared__ float contribution[POOLS][HEADS];
	const uint32_t first_local = blockIdx.x * POOLS, first_row = blockIdx.y * ROWS, warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES;
	uint32_t cached = 0xffffffffu;
	for ( uint32_t offset = 0u; offset < ROWS && first_row + offset < rows; ++offset )
	{
		const uint32_t row = first_row + offset, sequence = sequence_of_row[row];
		if ( sequence != cached )
		{
			__syncthreads();
			for ( uint32_t local = threadIdx.x; local < POOLS; local += THREADS )
			{
				const uint32_t pool = Glm5NextPoolGlobal(first_local + local, owner_rank, owner_degree);
				uint32_t ready = first_local + local < local_stride && pool < pools && pool * KPOOL + KPOOL <= context_length[sequence] ? 1u : 0u;
				if ( ready != 0u )
				{
					const uint32_t first = pool * KPOOL;
					for ( uint32_t slot_in_pool = 0u; slot_in_pool < KPOOL; ++slot_in_pool )
					{
						const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Glm5NextIndexKv>(index_cache, sequence, first + slot_in_pool, row, LM_KV_ACCESS_READ);
						pool_slots[local][slot_in_pool] = slot;
						if ( slot == 0 )
						{
							ready = 0u;
							break;
						}
					}
				}
				pool_ready[local] = ready;
			}
			__syncthreads();
			for ( uint32_t item = threadIdx.x; item < POOLS * DIM; item += THREADS )
			{
				const uint32_t local = item / DIM, index = item % DIM;
				if ( pool_ready[local] == 0u )
					continue;
				pool_key[local][index] = Glm5NextPoolKeyElement<DIM, KPOOL>(pool_slots[local], compress_ape_f32, index);
			}
			__syncthreads();
			cached = sequence;
		}
		const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
		float query[PER_WARP][4], weight[PER_WARP];
		for ( uint32_t slot = 0u; slot < PER_WARP; ++slot )
		{
			const uint32_t head = warp + slot * WARPS;
			Glm5NextLoadBf16x4(query_bf16 + ((uint64_t)row * HEADS + head) * DIM + lane * 4u, query[slot]);
			weight[slot] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row * HEADS + head]);
		}
		for ( uint32_t local = 0u; local < POOLS; ++local )
		{
			if ( pool_ready[local] == 0u )
				continue;
			const float4 k = *(const float4 *)(pool_key[local] + lane * 4u);
			for ( uint32_t slot = 0u; slot < PER_WARP; ++slot )
			{
				const float contrib = Glm5NextPoolHeadDot(query[slot], k, softmax_scale, weight[slot], head_scale);
				if ( lane == 0u )
					contribution[local][warp + slot * WARPS] = contrib;
			}
		}
		__syncthreads();
		for ( uint32_t local = threadIdx.x; local < POOLS; local += THREADS )
		{
			const uint32_t pool = Glm5NextPoolGlobal(first_local + local, owner_rank, owner_degree);
			float total = 0.0f;
			if ( first_local + local >= local_stride )
				continue;
			if ( pool >= pools || pool * KPOOL + KPOOL > context )
			{
				pool_scores[(uint64_t)row * local_stride + first_local + local] = -INFINITY;
				continue;
			}
			if ( pool_ready[local] == 0u )
				continue;
			for ( uint32_t head = 0u; head < HEADS; ++head )
				total += contribution[local][head];
			pool_scores[(uint64_t)row * local_stride + first_local + local] = total;
		}
		__syncthreads();
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextPoolPermuteKernel(const float *__restrict__ gathered, float *__restrict__ pool_scores,
	uint32_t pools, uint32_t local_stride, uint64_t peer_stride, uint32_t degree, const uint32_t *__restrict__ sequence_of_row = 0,
	const uint32_t *__restrict__ context_length = 0, const uint32_t *__restrict__ row_positions = 0, uint32_t dense_limit = 0u)
{
	const uint32_t row = blockIdx.x;
	uint32_t limit = pools;
	if ( dense_limit != 0u && sequence_of_row != 0 && context_length != 0 )
	{
		uint32_t context = context_length[sequence_of_row[row]];
		if ( row_positions != 0 && row_positions[row] + 1u < context )
			context = row_positions[row] + 1u;
		if ( context <= dense_limit )
			return;
		const uint32_t active = context / G5N_KPOOL;
		if ( active < limit )
			limit = active;
	}
	for ( uint32_t pool = threadIdx.x; pool < limit; pool += THREADS )
		pool_scores[(uint64_t)row * pools + pool] = gathered[(uint64_t)Glm5NextPoolOwner(pool, degree) * peer_stride +
			(uint64_t)row * local_stride + Glm5NextPoolLocal(pool, degree)];
}

template<uint32_t THREADS, uint32_t KPOOL, uint32_t TOPK, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 1) void Glm5NextPoolExpandKernel(const uint32_t *__restrict__ selected_pools,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions,
	uint32_t *__restrict__ selected_positions, uint32_t rows, uint32_t dense_limit = 0u)
{
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	const uint32_t sequence = sequence_of_row[row];
	const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
	if ( dense_limit != 0u && context <= dense_limit )
		return;
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

struct Glm5NextHcPolicy
{
	static constexpr float kNormEpsilon = G5N_EPS;

	static __device__ void Gates(const float *mixes, const float *scale3, const float *base, float *pre, float *post)
	{
		const uint32_t lane = threadIdx.x % LM_WARP_LANES;
		if ( lane < G5N_HC )
		{
			pre[lane] = SparkMathSigmoidFast(mixes[lane] * scale3[0] + base[lane]) + G5N_HC_EPS;
			post[lane] = SparkMathScaledSigmoidFast(2.0f,mixes[G5N_HC + lane] * scale3[1] + base[G5N_HC + lane]);
		}
	}

	static __device__ void Comb(const float *mixes, const float *scale3, const float *base, float *comb_out)
	{
		const uint32_t lane = threadIdx.x % LM_WARP_LANES, cell = lane & 15u;
		float value, total;
		value = mixes[2u * G5N_HC + cell] * scale3[2] + base[2u * G5N_HC + cell];
		value = __expf(value - LmHcRowMax(value, lane));
		total = LmHcRowSum(value, lane);
		value = value / total + G5N_HC_EPS;
		for ( uint32_t iteration = 0u; iteration < G5N_HC_ITERATIONS; ++iteration )
		{
			if ( iteration != 0u )
			{
				total = LmHcRowSum(value, lane);
				value /= total + G5N_HC_EPS;
			}
			total = LmHcColumnSum(value, lane);
			value /= total + G5N_HC_EPS;
		}
		if ( lane < G5N_HC * G5N_HC )
			comb_out[cell] = value;
	}
};

static int32_t Glm5NextHcSite(const Glm5NextLayerBuffers *b, const void *fn, const void *base, const void *scale, uint32_t rows, SparkGpuStream stream)
{
	if ( fn == 0 || base == 0 || scale == 0 || ((uintptr_t)fn % 16u) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( LmHcSite<G5N_HIDDEN, Glm5NextHcPolicy>((const uint16_t *)b->hidden_bf16, (const float *)fn, (const float *)scale, (const float *)base,
			b->hc_mixes_f32, b->hc_pre_f32, b->hc_post_f32, b->hc_comb_f32, b->hc_collapsed_bf16, b->hc_snapshot_bf16, rows, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextHcSiteNorm(const Glm5NextLayerBuffers *b, const void *fn, const void *base, const void *scale, const void *norm_weight,
	uint32_t rows, SparkGpuStream stream)
{
	if ( fn == 0 || base == 0 || scale == 0 || norm_weight == 0 || ((uintptr_t)fn % 16u) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( LmHcSiteNorm<G5N_HIDDEN, Glm5NextHcPolicy>((const uint16_t *)b->hidden_bf16, (const float *)fn, (const float *)scale, (const float *)base,
			b->hc_mixes_f32, b->hc_pre_f32, b->hc_post_f32, b->hc_comb_f32, b->hc_collapsed_bf16, b->hc_snapshot_bf16, (const uint16_t *)norm_weight,
			b->normed_bf16, rows, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

static __global__ __launch_bounds__(768u, 1) void Glm5NextHcMultiDot24WarpKernel(const uint16_t *__restrict__ streams_in,
	const float *__restrict__ fn_f32, float *__restrict__ partials, uint32_t rows)
{
	using S = LmHcSiteShape<G5N_HIDDEN>;
	constexpr uint32_t SLICE_VEC = S::kSlice / 8u;
	__shared__ __align__(16) uint16_t staged[2][S::kSlice];
	__shared__ float reduction[2][8];
	const uint32_t slice = blockIdx.x, tile = blockIdx.y, tiles = gridDim.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t off = slice * S::kSlice;
	const uint32_t row_begin = (rows * tile) / tiles, row_end = (rows * (tile + 1u)) / tiles;
	float4 weight[S::kVectors];
	const float4 *fn4 = (const float4 *)(fn_f32 + (uint64_t)warp * S::kFlat + off);
	#pragma unroll
	for ( uint32_t vector = 0u; vector < S::kVectors; ++vector )
		weight[vector] = __ldcs(fn4 + lane + vector * LM_WARP_LANES);
	LmDependentWait();
	if ( row_begin < row_end )
	{
		if ( threadIdx.x < SLICE_VEC )
		{
			const uint32_t dst_addr = (uint32_t)__cvta_generic_to_shared(&staged[row_begin & 1u][threadIdx.x * 8u]);
			const void *src_ptr = (const uint4 *)(streams_in + (uint64_t)row_begin * S::kFlat + off) + threadIdx.x;
			asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src_ptr) : "memory");
			asm volatile("cp.async.commit_group;\n" ::: "memory");
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		}
		__syncthreads();
	}
	for ( uint32_t row = row_begin; row < row_end; ++row )
	{
		const uint32_t buf = row & 1u;
		if ( row + 1u < row_end && threadIdx.x < SLICE_VEC )
		{
			const uint32_t dst_addr = (uint32_t)__cvta_generic_to_shared(&staged[buf ^ 1u][threadIdx.x * 8u]);
			const void *src_ptr = (const uint4 *)(streams_in + (uint64_t)(row + 1u) * S::kFlat + off) + threadIdx.x;
			asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src_ptr) : "memory");
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		}
		float *row_partials = partials + ((uint64_t)row * LM_HC_SITE_CLUSTER + slice) * (LM_HC_SITE_MIX + 1u);
		if ( threadIdx.x < 256u )
		{
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t element = threadIdx.x; element < S::kSlice; element += 256u )
			{
				const float value = LmBf16ToFloat(staged[buf][element]);
				total += value * value;
			}
			#pragma unroll
			for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
				total += __shfl_down_sync(0xffffffffu, total, step);
			if ( lane == 0u )
				reduction[buf][warp] = total;
		}
		float acc = 0.0f;
		#pragma unroll
		for ( uint32_t vector = 0u; vector < S::kVectors; ++vector )
		{
			const uint2 packed = ((const uint2 *)staged[buf])[lane + vector * LM_WARP_LANES];
			acc = fmaf(weight[vector].x, __uint_as_float(packed.x << 16u), acc);
			acc = fmaf(weight[vector].y, __uint_as_float(packed.x & 0xffff0000u), acc);
			acc = fmaf(weight[vector].z, __uint_as_float(packed.y << 16u), acc);
			acc = fmaf(weight[vector].w, __uint_as_float(packed.y & 0xffff0000u), acc);
		}
		#pragma unroll
		for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
			acc += __shfl_xor_sync(0xffffffffu, acc, step);
		if ( lane == 0u )
			row_partials[warp] = acc;
		if ( row + 1u < row_end && threadIdx.x < SLICE_VEC )
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( warp == 0u )
		{
			float sum = lane < 8u ? reduction[buf][lane] : 0.0f;
			#pragma unroll
			for ( uint32_t step = 4u; step > 0u; step >>= 1u )
				sum += __shfl_down_sync(0xffffffffu, sum, step);
			if ( lane == 0u )
				row_partials[LM_HC_SITE_MIX] = sum;
		}
	}
	LmDependentRelease();
}

static int32_t Glm5NextHcPostNorm(const Glm5NextLayerBuffers *b, const uint16_t *streams_in, const uint16_t *post_out, uint16_t *streams_out,
	const void *fn, const void *base, const void *scale, const void *norm_weight, uint32_t rows, SparkGpuStream stream)
{
	if ( fn == 0 || base == 0 || scale == 0 || norm_weight == 0 || ((uintptr_t)fn % 16u) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( rows >= 32u && LmIsAligned16(streams_in) && LmIsAligned16(b->hc_collapsed_bf16) && LmIsAligned16(norm_weight) && LmIsAligned16(b->normed_bf16) &&
		(post_out == 0 || (streams_out != streams_in && LmIsAligned16(post_out) && LmIsAligned16(streams_out))) )
	{
		const uint16_t *active = post_out != 0 ? streams_out : streams_in;
		const uint32_t row_tiles = rows >= 64u ? 6u : 4u;
		if ( post_out != 0 )
		{
			LM_LAUNCH_DEPENDENT((LmHcPostBf16VecKernel<G5N_HIDDEN>), rows, G5N_THREADS, 0, stream, post_out, streams_in, b->hc_post_f32,
				b->hc_comb_f32, streams_out);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		LM_LAUNCH_DEPENDENT((Glm5NextHcMultiDot24WarpKernel), dim3(LM_HC_SITE_CLUSTER, row_tiles), 768u, 0, stream, active, (const float *)fn,
			(float *)b->hc_collapsed_bf16, rows);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		LM_LAUNCH_DEPENDENT((LmHcSiteMultiFinishKernel<G5N_HIDDEN, Glm5NextHcPolicy, true, false, false>), rows, G5N_THREADS, 0, stream,
			(float *)b->hc_collapsed_bf16, active, (const float *)scale, (const float *)base, (float *)0, (float *)0, b->hc_post_f32, b->hc_comb_f32,
			(uint16_t *)0, (uint16_t *)0, (const uint16_t *)norm_weight, b->normed_bf16);
		return Glm5NextLaunched();
	}
	if ( LmHcSitePostNorm<G5N_HIDDEN, Glm5NextHcPolicy>(streams_in, post_out, b->hc_post_f32, b->hc_comb_f32, streams_out, (const float *)fn,
			(const float *)scale, (const float *)base, (float *)b->hc_collapsed_bf16, b->hc_post_f32, b->hc_comb_f32, (const uint16_t *)norm_weight,
			b->normed_bf16, rows, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextHcPostFrom(const Glm5NextLayerBuffers *b, const uint16_t *src_streams, uint16_t *dst_streams, uint32_t rows, SparkGpuStream stream)
{
	if ( LmIsAligned16(b->attention_out_bf16) && LmIsAligned16(src_streams) && LmIsAligned16(dst_streams) )
	{
		LM_LAUNCH_DEPENDENT((LmHcPostBf16VecKernel<G5N_HIDDEN>), rows, G5N_THREADS, 0, stream, b->attention_out_bf16, src_streams,
			b->hc_post_f32, b->hc_comb_f32, dst_streams);
		return Glm5NextLaunched();
	}
	LM_LAUNCH((LmHcPostBf16Kernel<G5N_HC>), dim3(rows, G5N_HIDDEN / G5N_THREADS), G5N_THREADS, 0, stream, b->attention_out_bf16, src_streams,
		b->hc_post_f32, b->hc_comb_f32, dst_streams, rows, G5N_HC, G5N_HIDDEN);
	return Glm5NextLaunched();
}

static int32_t Glm5NextHcPost(const Glm5NextLayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	return Glm5NextHcPostFrom(b, b->hc_snapshot_bf16, b->hidden_bf16, rows, stream);
}

static int32_t Glm5NextKdaRecur(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sequences, const uint32_t *kept, uint32_t commit,
	SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t heads = b->kda_heads, rank_qk = heads * G5N_KDA_KEY, rank_v = heads * G5N_KDA_VALUE;
	LmCausalConvStreams<uint16_t> conv;
	conv.window[0] = b->kda_q_window;
	conv.window[1] = b->kda_k_window;
	conv.window[2] = b->kda_v_window;
	conv.input_bf16[0] = b->kda_q_input;
	conv.input_bf16[1] = b->kda_k_input;
	conv.input_bf16[2] = b->kda_v_input;
	conv.output_bf16[0] = b->q_bf16;
	conv.output_bf16[1] = b->kv_slot_bf16;
	conv.output_bf16[2] = b->gate_up_bf16;
	conv.weight[0] = (const uint16_t *)w->kda_q_conv;
	conv.weight[1] = (const uint16_t *)w->kda_k_conv;
	conv.weight[2] = (const uint16_t *)w->kda_v_conv;
	conv.channels[0] = conv.channels[1] = rank_qk;
	conv.channels[2] = rank_v;
	if ( LmCausalConvStreamsLaunch<G5N_KDA_CONV, LM_CONV_SWISH, uint16_t>(conv, b->kda_state_index, b->sequence_row_begin, kept, sequences, commit,
		b->sequence_row_indices, rows, stream, commit == 0u ? b->parent_rows : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	if constexpr ( LM_WARP_LANES == 32u )
	{
		if ( LmDeltaRuleLaunch<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float>(rows, b->kda_state_pool, b->kda_state_slot_bytes, b->kda_state_index,
				b->sequence_row_begin, kept, b->q_bf16, b->kv_slot_bf16, b->gate_up_bf16, b->kda_retention, b->kda_write_gate, b->attention_out_bf16, heads, 1u,
				sequences, commit, b->sequence_row_indices, stream, commit == 0u ? b->state_loads : (const uint32_t *)0,
				commit == 0u ? b->state_saves : (const uint32_t *)0, commit == 0u ? b->kda_snapshots : (float *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
		LM_LAUNCH((LmDeltaRuleKernel<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float, G5N_KDA_DELTA_COLUMNS>),
			dim3(sequences, heads, G5N_KDA_VALUE / G5N_KDA_DELTA_COLUMNS), G5N_THREADS, G5N_KDA_KEY * G5N_KDA_DELTA_COLUMNS * sizeof(float), stream,
			b->kda_state_pool, b->kda_state_slot_bytes, b->kda_state_index, b->sequence_row_begin, kept, b->q_bf16, b->kv_slot_bf16,
			b->gate_up_bf16, b->kda_retention, b->kda_write_gate, b->attention_out_bf16, heads, 1u, sequences, commit, b->sequence_row_indices);
	return Glm5NextLaunched();
}

__global__ __launch_bounds__(128u) void Glm5NextKdaUpDecayBetaKernel(const uint16_t *__restrict__ fused_dg,
	const uint16_t *__restrict__ decay_up_w, const uint16_t *__restrict__ gate_up_w, const float *__restrict__ decay_bias,
	const float *__restrict__ head_log_scale, const uint16_t *__restrict__ beta_src, uint32_t beta_stride, uint32_t beta_offset,
	float *__restrict__ retention, uint16_t *__restrict__ gate_out, float *__restrict__ write_gate, uint32_t rank_qk, uint32_t heads)
{
	constexpr uint32_t GROUPS = 8u;
	const uint32_t group = threadIdx.x >> 4u, lane = threadIdx.x & 15u, row = blockIdx.y;
	const uint32_t ch = blockIdx.x * GROUPS + group;
	LmDependentWait();
	if ( blockIdx.x == 0u && threadIdx.x < heads )
	{
		const float r = SparkMathRoundBf16(SparkMathSigmoidFast(LmBf16ToFloat(beta_src[(uint64_t)row * beta_stride + beta_offset + threadIdx.x])));
		write_gate[(uint64_t)row * heads + threadIdx.x] = r;
	}
	if ( ch < rank_qk )
	{
		const uint16_t *dg_row = fused_dg + (uint64_t)row * 2u * G5N_KDA_LOW_RANK;
		const uint4 ad = __ldg((const uint4 *)dg_row + lane);
		const uint4 ag = __ldg((const uint4 *)(dg_row + G5N_KDA_LOW_RANK) + lane);
		const uint4 wd = __ldcs((const uint4 *)(decay_up_w + (uint64_t)ch * G5N_KDA_LOW_RANK) + lane);
		const uint4 wg = __ldcs((const uint4 *)(gate_up_w + (uint64_t)ch * G5N_KDA_LOW_RANK) + lane);
		float sd = LmSkinnyBf16Chunk(wd, ad, 0.0f);
		float sg = LmSkinnyBf16Chunk(wg, ag, 0.0f);
#pragma unroll
		for ( uint32_t step = 8u; step > 0u; step >>= 1u )
		{
			sd += __shfl_xor_sync(0xffffffffu, sd, step);
			sg += __shfl_xor_sync(0xffffffffu, sg, step);
		}
		if ( lane == 0u )
		{
			const uint64_t out_idx = (uint64_t)row * rank_qk + ch;
			const float d_bf16 = LmBf16ToFloat(LmFloatToBf16(sd));
			retention[out_idx] = LmBoundedDecay(d_bf16, decay_bias[ch], head_log_scale[ch / G5N_KDA_KEY], G5N_KDA_GATE_LOWER);
			gate_out[out_idx] = LmFloatToBf16(sg);
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(G5N_THREADS) void Glm5NextKdaDecayBetaKernel(const uint16_t *__restrict__ logit_bf16,
	const float *__restrict__ channel_bias, const float *__restrict__ head_log_scale, const uint16_t *__restrict__ beta_bf16,
	float *__restrict__ retention, float *__restrict__ write_gate, uint32_t heads, uint32_t rows)
{
	const uint32_t row = blockIdx.x, rank_qk = heads * G5N_KDA_KEY;
	if ( row >= rows )
		return;
	const uint64_t base = (uint64_t)row * rank_qk;
	for ( uint32_t index = threadIdx.x; index < rank_qk; index += G5N_THREADS )
		retention[base + index] = LmBoundedDecay(LmBf16ToFloat(logit_bf16[base + index]),
			channel_bias[index], head_log_scale[index / G5N_KDA_KEY], G5N_KDA_GATE_LOWER);
	for ( uint32_t h = threadIdx.x; h < heads; h += G5N_THREADS )
		write_gate[(uint64_t)row * heads + h] = SparkMathRoundBf16(SparkMathSigmoidFast(LmBf16ToFloat(beta_bf16[(uint64_t)row * heads + h])));
}

static int32_t Glm5NextKda(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t sms, SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t heads = b->kda_heads, rank_qk = heads * G5N_KDA_KEY, rank_v = heads * G5N_KDA_VALUE;
	const uint32_t fused = rank_qk * 2u + rank_v + heads;
	int32_t status;
	if ( b->kda_state_pool == 0 || b->kda_state_slot_bytes != rank_qk * G5N_KDA_VALUE * sizeof(float) || w->kda_qkv_beta == 0 ||
		w->kda_decay_gate_down == 0 || w->kda_decay_up == 0 || w->kda_gate_up == 0 || w->kda_q_conv == 0 || w->kda_k_conv == 0 ||
		w->kda_v_conv == 0 || w->kda_decay_bias == 0 || w->kda_head_log_scale == 0 || w->kda_out_norm == 0 || w->kda_out == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->prenormed == 0u )
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_HIDDEN + 8u) * sizeof(float), stream, b->hc_collapsed_bf16,
			(const uint16_t *)w->attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	const uint32_t skinny_kda = rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u ? 1u : 0u;
	if ( skinny_kda != 0u )
	{
		const LmSkinnyDenseTarget targets[2] = {
			{w->kda_qkv_beta, b->fused_qkvb_bf16, 0, fused, fused, 0u, LmScaleTensorNone()},
			{w->kda_decay_gate_down, b->fused_decay_gate_bf16, 0, 2u * G5N_KDA_LOW_RANK, 2u * G5N_KDA_LOW_RANK, 0u, LmScaleTensorNone()}};
		status = Glm5NextSkinnyMulti(b, targets, 2u, b->normed_bf16, rows, G5N_HIDDEN, stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("glm5 kda: the fused qkv-beta and decay-gate launch refused %u rows (%u + %u outputs over %u inputs); waves of up to %u "
				"rows and row-invariant waves run on it, so both weights and the normed rows need 16-byte alignment\n", rows, fused,
				2u * G5N_KDA_LOW_RANK, G5N_HIDDEN, LM_SKINNY_ROWS_WIDE);
	}
	else
	{
		status = Glm5NextLinear(b, b->normed_bf16, w->kda_qkv_beta, b->fused_qkvb_bf16, rows, G5N_HIDDEN, fused, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->normed_bf16, w->kda_decay_gate_down, b->fused_decay_gate_bf16, rows, G5N_HIDDEN, 2u * G5N_KDA_LOW_RANK, sms, stream);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	Glm5NextLayerBuffers kda_b = *b;
	if ( rows == 1u && commit != 0u )
	{
		kda_b.kda_q_input = b->fused_qkvb_bf16;
		kda_b.kda_k_input = b->fused_qkvb_bf16 + rank_qk;
		kda_b.kda_v_input = b->fused_qkvb_bf16 + 2u * rank_qk;
		kda_b.kda_beta_logit = b->fused_qkvb_bf16 + 2u * rank_qk + rank_v;
	}
	else
		LM_LAUNCH((LmSplitQkvbKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->fused_qkvb_bf16, b->kda_q_input, b->kda_k_input,
			b->kda_v_input, b->kda_beta_logit, rows, rank_qk, rank_v, heads, fused);
	if ( (rows <= 64u || b->row_invariant != 0u) && LmIsAligned16(b->fused_decay_gate_bf16) && LmIsAligned16(w->kda_decay_up) && LmIsAligned16(w->kda_gate_up) )
	{
		LM_LAUNCH_DEPENDENT((Glm5NextKdaUpDecayBetaKernel), dim3((rank_qk + 7u) / 8u, rows), 128u, 0, stream,
			b->fused_decay_gate_bf16, (const uint16_t *)w->kda_decay_up, (const uint16_t *)w->kda_gate_up, w->kda_decay_bias, w->kda_head_log_scale,
			b->fused_qkvb_bf16, fused, 2u * rank_qk + rank_v, b->kda_retention, b->kda_gate_bf16, b->kda_write_gate, rank_qk, heads);
	}
	else
	{
		LM_LAUNCH((LmSplit2RowsKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->fused_decay_gate_bf16,
			b->kda_decay_latent_bf16, b->kda_gate_latent_bf16, rows, G5N_KDA_LOW_RANK);
		status = Glm5NextLinear(b, b->kda_decay_latent_bf16, w->kda_decay_up, b->kda_decay_logit_bf16, rows, G5N_KDA_LOW_RANK, rank_qk, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->kda_gate_latent_bf16, w->kda_gate_up, b->kda_gate_bf16, rows, G5N_KDA_LOW_RANK, rank_v, sms, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
		LM_LAUNCH((Glm5NextKdaDecayBetaKernel), rows, G5N_THREADS, 0, stream, b->kda_decay_logit_bf16,
			w->kda_decay_bias, w->kda_head_log_scale, kda_b.kda_beta_logit, b->kda_retention, b->kda_write_gate, heads, rows);
	}
	status = Glm5NextKdaRecur(&kda_b, rows, sequences, (const uint32_t *)0, commit, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmRmsNormSigmoidGateKernel<G5N_THREADS>), dim3((uint64_t)rows * heads), G5N_THREADS, 0, stream, b->attention_out_bf16,
		b->kda_gate_bf16, (const float *)w->kda_out_norm, b->kv_slot_bf16, G5N_KDA_VALUE, G5N_EPS);
	return Glm5NextLinear(b, b->kv_slot_bf16, w->kda_out, b->attention_out_bf16, rows, rank_v, G5N_HIDDEN, sms, stream);
}

__global__ __launch_bounds__(G5N_THREADS) void Glm5NextDsaPreStoreKernel(uint16_t *__restrict__ q_compressed,
	const uint16_t *__restrict__ q_a_norm, uint16_t *__restrict__ kv_slot, const uint16_t *__restrict__ kv_a_norm,
	uint16_t *__restrict__ index_key, const uint16_t *__restrict__ index_norm_w, const uint16_t *__restrict__ index_norm_b,
	const uint16_t *__restrict__ index_gate, uint16_t *__restrict__ index_packed, LmKvShardView cache, LmKvShardView index_cache,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ slot_of_row, uint32_t rows)
{
	__shared__ __align__(16) float norm_buf[G5N_QA + 8u];
	__shared__ float pair_red[2u * LmBlockWarps<G5N_THREADS>()];
	const uint32_t row = blockIdx.x, part = blockIdx.y;
	LmDependentRelease();
	if ( row >= rows )
		return;
	if ( part == 0u )
	{
		LmRmsNormRowBody<G5N_THREADS, uint16_t, true, false>(norm_buf, q_compressed, (const uint16_t *)0, q_a_norm,
			(uint16_t *)0, q_compressed, G5N_QA, G5N_QA, G5N_EPS, LmRmsGainOp<true>{});
		return;
	}
	if ( part == 1u )
	{
		LmRmsNormRowBody<G5N_THREADS, uint16_t, true, false>(norm_buf, kv_slot, (const uint16_t *)0, kv_a_norm,
			(uint16_t *)0, kv_slot, G5N_LATENT, G5N_LATENT, G5N_EPS, LmRmsGainOp<true>{});
		__syncthreads();
		const uint32_t pos = slot_of_row[row];
		if ( SparkKvShardOwns(cache.shard, pos) != 0u )
		{
			uint16_t *dst = (uint16_t *)LmKvShardSlotRequired<Glm5NextKv>(cache, sequence_of_row[row], pos, row, LM_KV_ACCESS_WRITE);
			if ( dst != 0 )
			{
				const uint16_t *src = kv_slot + (uint64_t)row * G5N_LATENT;
				for ( uint32_t i = threadIdx.x; i < G5N_LATENT; i += G5N_THREADS )
					dst[i] = src[i];
			}
		}
		return;
	}
	const uint64_t key_base = (uint64_t)row * G5N_INDEX_DIM;
	const uint16_t *k_in = index_key + key_base;
	const uint16_t *g_in = index_gate + key_base;
	uint16_t *p_smem = (uint16_t *)norm_buf;
	float total = 0.0f, total_sq = 0.0f, mean, inverse;
	uint4 k_vec = make_uint4(0u, 0u, 0u, 0u);
	const uint32_t vec_off = threadIdx.x * 8u;
	if ( vec_off < G5N_INDEX_DIM )
	{
		k_vec = *(const uint4 *)(k_in + vec_off);
		float v[8];
		LmLoadBf16x8AsFloat(k_vec, v);
#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
		{
			total += v[k];
			total_sq += v[k] * v[k];
		}
	}
	LmLayerNormStats<G5N_THREADS>(total, total_sq, pair_red, G5N_INDEX_DIM, SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON, &mean, &inverse);
	if ( vec_off < G5N_INDEX_DIM )
	{
		float v[8], w[8], bi[8], out_f[8];
		LmLoadBf16x8AsFloat(k_vec, v);
		LmLoadWeight8(index_norm_w, vec_off, w);
		LmLoadWeight8(index_norm_b, vec_off, bi);
#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out_f[k] = ((v[k] - mean) * inverse * w[k]) + bi[k];
		const uint4 k_out = LmPackUint4FloatToBf16(out_f);
		*(uint4 *)(index_key + key_base + vec_off) = k_out;
		*(uint4 *)(p_smem + vec_off) = k_out;
		*(uint4 *)(p_smem + G5N_INDEX_DIM + vec_off) = *(const uint4 *)(g_in + vec_off);
	}
	if ( threadIdx.x == 0u )
		p_smem[2u * G5N_INDEX_DIM] = 0x3F80u;
	__syncthreads();
	const uint32_t pos = slot_of_row[row];
	if ( SparkKvShardOwns(index_cache.shard, pos) != 0u )
	{
		uint16_t *dst = (uint16_t *)LmKvShardSlotRequired<Glm5NextIndexKv>(index_cache, sequence_of_row[row], pos, row, LM_KV_ACCESS_WRITE);
		if ( dst != 0 )
			for ( uint32_t i = threadIdx.x; i < G5N_INDEX_PACKED; i += G5N_THREADS )
				dst[i] = p_smem[i];
	}
	(void)index_packed;
}

static int32_t Glm5NextIndexScore(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, uint32_t projected, SparkGpuStream stream,
	uint32_t dense_limit = 0u, uint32_t prestored = 0u)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t degree = b->index_owner_degree, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	int32_t status;
	if ( w->index_q == 0 || w->index_k == 0 || w->index_head == 0 || w->index_norm_weight == 0 || w->index_norm_bias == 0 ||
		w->index_compress_gate == 0 || w->index_compress_ape == 0 || degree < 2u || b->index_owner_rank >= degree ||
		degree != b->index_cache.shard.degree || b->index_owner_rank != b->index_cache.shard.rank )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	const uint32_t skip_dense = dense_limit != 0u && context <= dense_limit ? 1u : 0u;
	if ( prestored == 0u )
	{
		if ( projected == 0u )
		{
			status = Glm5NextLinear(b, b->normed_bf16, w->index_k, b->index_key_bf16, rows, G5N_HIDDEN, G5N_INDEX_DIM, sms, stream);
			if ( status == LM_LAUNCH_OK )
				status = Glm5NextLinear(b, b->normed_bf16, w->index_compress_gate, b->index_gate_bf16, rows, G5N_HIDDEN, G5N_INDEX_DIM, sms, stream);
			if ( status != LM_LAUNCH_OK )
				return status;
		}
		LM_LAUNCH((LmLayerNormKernel<G5N_THREADS, uint16_t>), rows, G5N_THREADS, (G5N_INDEX_DIM + 8u) * sizeof(float), stream, b->index_key_bf16,
			(const uint16_t *)w->index_norm_weight, (const uint16_t *)w->index_norm_bias, b->index_key_bf16, G5N_INDEX_DIM, G5N_INDEX_DIM,
			SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON);
		if ( projected == 0u && skip_dense == 0u )
		{
			status = Glm5NextLinear(b, b->normed_bf16, w->index_head, b->index_head_weight_bf16, rows, G5N_HIDDEN, G5N_INDEX_HEADS, sms, stream);
			if ( status != LM_LAUNCH_OK )
				return status;
		}
		LM_LAUNCH((Glm5NextIndexPackKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->index_key_bf16, b->index_gate_bf16, b->index_packed_bf16,
			G5N_INDEX_DIM);
		LM_LAUNCH((LmKvShardStoreKernel<Glm5NextIndexKv, G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->index_cache, b->index_packed_bf16,
			b->sequence_of_row, b->slots != 0 ? b->slots : b->positions, rows, G5N_INDEX_PACKED);
	}
	if ( skip_dense != 0u )
		return Glm5NextLaunched();
	if ( dense_limit != 0u && rows == 1u )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = (const uint8_t *)w->index_q;
		args.activation = b->q_compressed_bf16;
		args.output_bf16 = b->index_query_bf16;
		args.scale = LmScaleTensorNone();
		args.rows = rows;
		args.input_dimension = G5N_QA;
		args.output_dimension = G5N_INDEX_QUERY;
		args.output_row_stride = G5N_INDEX_QUERY;
		args.skip_sequence_of_row = b->sequence_of_row;
		args.skip_context_length = b->context_length;
		args.skip_row_position = b->positions;
		args.skip_max_context = dense_limit;
		status = LmSkinnyLaunch<LmBf16Format>(&args, stream, 1u);
	}
	else
		status = Glm5NextLinear(b, b->q_compressed_bf16, w->index_q, b->index_query_bf16, rows, G5N_QA, G5N_INDEX_QUERY, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( rows >= 128u && b->fork_positions == 0 )
		LM_LAUNCH((Glm5NextPoolScoreRowsKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS, G5N_POOL_BLOCK_POOLS, G5N_POOL_BLOCK_ROWS>),
			dim3((local_stride + G5N_POOL_BLOCK_POOLS - 1u) / G5N_POOL_BLOCK_POOLS, (rows + G5N_POOL_BLOCK_ROWS - 1u) / G5N_POOL_BLOCK_ROWS), G5N_THREADS, 0,
			stream, b->index_query_bf16, b->index_head_weight_bf16, b->index_cache, b->sequence_of_row, b->context_length, b->positions,
			(const float *)w->index_compress_ape, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, b->index_local_scores, rows);
	else
		LM_LAUNCH((Glm5NextPoolScoreKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS>), dim3(local_stride, rows), G5N_THREADS, 0, stream,
			b->index_query_bf16, b->index_head_weight_bf16, b->index_cache, b->sequence_of_row, b->context_length, b->positions,
			(const float *)w->index_compress_ape, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, b->index_local_scores, b->fork_positions, b->slot_offsets, dense_limit);
	return Glm5NextLaunched();
}

static uint32_t Glm5NextIndexKeep(void)
{
	return G5N_SELECTED / G5N_KPOOL;
}

static uint32_t Glm5NextIndexCandidates(uint32_t rows, uint32_t local_stride)
{
	return rows >= G5N_POOL_ROWS_MIN && local_stride > 2u * Glm5NextIndexKeep() ? 1u : 0u;
}

static int32_t Glm5NextIndexCandidatePack(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, SparkGpuStream stream)
{
	const uint32_t degree = b->index_owner_degree, local_stride = SparkGlm5NextIndexCpLocalStride(SparkGlm5NextIndexCpPools(context), degree);
	SparkKvShard shard;
	if ( context <= G5N_SELECTED || Glm5NextIndexCandidates(rows, local_stride) == 0u )
		return LM_LAUNCH_OK;
	shard.degree = degree;
	shard.rank = b->index_owner_rank;
	shard.grain = 1u;
	LmTopkExactRowsLaunch<G5N_THREADS>(b->index_local_scores, rows, local_stride, Glm5NextIndexKeep(), b->selected_pools, stream);
	if ( LmIndexShardCandidatePackLaunch<G5N_THREADS>(shard, b->index_local_scores, local_stride, b->selected_pools, Glm5NextIndexKeep(), rows,
		(uint2 *)b->selection_scores, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionHead(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	uint32_t projected;
	int32_t status;
	if ( w->attn_norm == 0 || w->q_a == 0 || w->q_a_norm == 0 || w->q_b == 0 || w->kv_a == 0 || w->kv_a_norm == 0 || b->kv_b_key == 0 ||
		b->kv_b_value == 0 || w->attn_output == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->prenormed == 0u )
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_HIDDEN + 8u) * sizeof(float), stream, b->hc_collapsed_bf16,
			(const uint16_t *)w->attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	projected = rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u ? 1u : 0u;
	if ( projected != 0u )
	{
		const LmSkinnyDenseTarget targets[4] = {
			{w->q_a, b->q_compressed_bf16, 0, G5N_QA, G5N_QA, 0u, LmScaleTensorNone()},
			{w->index_k, b->index_key_bf16, 0, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, LmScaleTensorNone()},
			{w->index_compress_gate, b->index_gate_bf16, 0, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, LmScaleTensorNone()},
			{w->index_head, b->index_head_weight_bf16, 0, G5N_INDEX_HEADS, G5N_INDEX_HEADS, 0u, LmScaleTensorNone()}};
		status = Glm5NextSkinnyMulti(b, targets, 4u, b->normed_bf16, rows, G5N_HIDDEN, stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("glm5 mla: the fused q_a and index launch refused %u rows (%u + %u + %u + %u outputs over %u inputs); waves of up to %u "
				"rows and row-invariant waves run on it, so the four weights and the normed rows need 16-byte alignment\n", rows, G5N_QA, G5N_INDEX_DIM,
				G5N_INDEX_DIM, G5N_INDEX_HEADS, G5N_HIDDEN, LM_SKINNY_ROWS_WIDE);
	}
	else
	{
		status = Glm5NextLinear(b, b->normed_bf16, w->q_a, b->q_compressed_bf16, rows, G5N_HIDDEN, G5N_QA, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->normed_bf16, w->index_k, b->index_key_bf16, rows, G5N_HIDDEN, G5N_INDEX_DIM, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->normed_bf16, w->index_compress_gate, b->index_gate_bf16, rows, G5N_HIDDEN, G5N_INDEX_DIM, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->normed_bf16, w->index_head, b->index_head_weight_bf16, rows, G5N_HIDDEN, G5N_INDEX_HEADS, sms, stream);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	status = Glm5NextLinear(b, b->normed_bf16, w->kv_a, b->kv_slot_bf16, rows, G5N_HIDDEN, G5N_LATENT, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( LmIsAligned16(b->index_key_bf16) && LmIsAligned16(b->index_gate_bf16) && LmIsAligned16(w->index_norm_weight) && LmIsAligned16(w->index_norm_bias) )
	{
		LM_LAUNCH((Glm5NextDsaPreStoreKernel), dim3(rows, 3u), G5N_THREADS, 0, stream, b->q_compressed_bf16, (const uint16_t *)w->q_a_norm,
			b->kv_slot_bf16, (const uint16_t *)w->kv_a_norm, b->index_key_bf16, (const uint16_t *)w->index_norm_weight,
			(const uint16_t *)w->index_norm_bias, b->index_gate_bf16, b->index_packed_bf16, b->cache, b->index_cache,
			b->sequence_of_row, b->slots != 0 ? b->slots : b->positions, rows);
		status = Glm5NextIndexScore(b, rows, context, sms, 1u, stream, G5N_SELECTED, 1u);
	}
	else
	{
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_QA + 8u) * sizeof(float), stream, b->q_compressed_bf16,
			(const uint16_t *)w->q_a_norm, b->q_compressed_bf16, G5N_QA, G5N_QA, G5N_EPS);
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_LATENT + 8u) * sizeof(float), stream, b->kv_slot_bf16,
			(const uint16_t *)w->kv_a_norm, b->kv_slot_bf16, G5N_LATENT, G5N_LATENT, G5N_EPS);
		LM_LAUNCH((LmKvShardStoreKernel<Glm5NextKv, G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->cache, b->kv_slot_bf16, b->sequence_of_row,
			b->slots != 0 ? b->slots : b->positions, rows, G5N_LATENT);
		status = Glm5NextIndexScore(b, rows, context, sms, 1u, stream, G5N_SELECTED, 0u);
	}
	return status == LM_LAUNCH_OK ? Glm5NextIndexCandidatePack(b, rows, context, stream) : status;
}

static int32_t Glm5NextIndexSelect(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, SparkGpuStream stream)
{
	const uint32_t degree = b->index_owner_degree, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	if ( context <= G5N_SELECTED )
		return LM_LAUNCH_OK;
	if ( Glm5NextIndexCandidates(rows, local_stride) != 0u )
	{
		const uint64_t peer_stride = (uint64_t)SparkGlm5NextIndexCpGatherSequences(rows, 2u * Glm5NextIndexKeep()) * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS / 2u;
		if ( LmIndexShardCandidateScatterLaunch<G5N_THREADS>((const uint2 *)b->index_gathered_scores, peer_stride, degree, Glm5NextIndexKeep(), rows, pools,
			b->selection_scores, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
		LM_LAUNCH((Glm5NextPoolPermuteKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->index_gathered_scores, b->selection_scores, pools,
			local_stride, (uint64_t)SparkGlm5NextIndexCpGatherSequences(rows, local_stride) * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS, degree,
			b->sequence_of_row, b->context_length, b->positions, G5N_SELECTED);
	LmTopkExactRowsLaunch<G5N_THREADS>(b->selection_scores, rows, pools, G5N_SELECTED / G5N_KPOOL, b->selected_pools, stream,
		b->sequence_of_row, b->context_length, b->positions, G5N_SELECTED, G5N_KPOOL);
	LM_LAUNCH((Glm5NextPoolExpandKernel<G5N_THREADS, G5N_KPOOL, G5N_SELECTED, G5N_INDEX_WIDTH>), rows, G5N_THREADS, 0, stream, b->selected_pools,
		b->sequence_of_row, b->context_length, b->positions, b->selected_positions, rows, G5N_SELECTED);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionTail(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	int32_t status = Glm5NextIndexSelect(b, rows, context, stream);
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextLinear(b, b->q_compressed_bf16, w->q_b, b->q_bf16, rows, G5N_QA, b->q_b_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( LmPerHeadProjectChainLaunch<G5N_THREADS, G5N_NOPE, G5N_LATENT, G5N_NOPE, 0u>(b->q_bf16, (const uint16_t *)b->kv_b_key, b->query_latent_bf16,
		b->attn_heads, rows, stream, b->row_invariant) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionPartial(const Glm5NextLayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	if ( !(LmLatentShardAttendLaunch<Glm5NextKv, LmKvShardView, G5N_LATENT, 0u>(b->cache, b->shard_query_gathered_bf16, b->shard_query_stride, b->attn_heads, b->sequence_of_row, b->context_length, b->positions, b->selected_positions, G5N_INDEX_WIDTH, G5N_SELECTED, G5N_QK_SCALE, b->shard_partials_f32, b->shard_partial_stride, rows, stream, b->fork_positions, b->slot_offsets) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextAttentionMerge(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	if ( LmLatentShardMergeLaunch<G5N_LATENT>(b->shard_partials_received_f32, b->shard_partial_stride, b->cache.shard.degree, b->attn_heads,
		b->attention_latent_bf16, rows, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	if ( LmPerHeadProjectChainLaunch<G5N_THREADS, G5N_LATENT, G5N_VALUE>(b->attention_latent_bf16, (const uint16_t *)b->kv_b_value,
		b->attention_value_bf16, b->attn_heads, rows, stream, b->row_invariant) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLinear(b, b->attention_value_bf16, b->w->attn_output, b->attention_out_bf16, rows, b->attn_output_columns, G5N_HIDDEN, sms, stream);
}

static int32_t Glm5NextDenseMlp(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	int32_t status;
	if ( w->post_attn_norm == 0 || w->dense_gate_up == 0 || w->dense_down == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->prenormed == 0u )
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_HIDDEN + 8u) * sizeof(float), stream, b->hc_collapsed_bf16,
			(const uint16_t *)w->post_attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	status = Glm5NextLinear(b, b->normed_bf16, w->dense_gate_up, b->gate_up_bf16, rows, G5N_HIDDEN, b->dense_gate_up_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->dense_intermediate,
		G5N_SWIGLU_LIMIT);
	return Glm5NextLinear(b, b->intermediate_bf16, w->dense_down, b->attention_out_bf16, rows, b->dense_intermediate, G5N_HIDDEN, sms, stream);
}

static int32_t Glm5NextMoeRoute(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t packed_rows = rows * G5N_TOP_K;
	int32_t status;
	if ( w->post_attn_norm == 0 || w->router == 0 || w->router_correction == 0 || w->shared_gate_up == 0 || w->shared_down == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->prenormed == 0u )
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_HIDDEN + 8u) * sizeof(float), stream, b->hc_collapsed_bf16,
			(const uint16_t *)w->post_attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	status = LmDenseRows<LmBf16Format>(w->router, LmScaleTensorNone(), b->normed_bf16, (uint16_t *)0, b->router_logits, rows, G5N_HIDDEN,
		G5N_EXPERTS, b->row_invariant, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	status = LmTopkRouteBuild<G5N_THREADS, G5N_EXPERTS, G5N_TOP_K, true, LM_TOPK_SCORE_SIGMOID>(b->router_logits, w->router_correction,
		(const uint16_t *)0, G5N_ROUTED_SCALE, b->route_fault, rows, packed_rows, b->route_expert, b->route_weight, b->group_row_offset,
		b->route_packed_row, b->route_source_token, b->expert_w1_rows, G5N_HIDDEN, G5N_TILE_N, b->group_tile_prefix_w1, b->group_tile_prefix_w2, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	return Glm5NextLaunched();
}

template<uint32_t ExpertCodec>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void Glm5NextMoeB1W1ActKernel(const uint8_t *__restrict__ exp_w1, const float *__restrict__ exp_s1,
	const uint16_t *__restrict__ shared_w1, const uint16_t *__restrict__ act, const uint32_t *__restrict__ route_expert,
	uint16_t *__restrict__ inter_out, uint16_t *__restrict__ shared_inter_out, uint32_t inter)
{
	constexpr uint32_t FP8_CHUNKS = G5N_HIDDEN / 16u, BF16_CHUNKS = G5N_HIDDEN / 8u, SCALES = G5N_HIDDEN / 128u, WARPS = LM_SKINNY_SPLIT_WARPS;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t expert_blocks = G5N_TOP_K * (inter / WARPS);
	LmDependentWait();
	if ( blockIdx.x < expert_blocks )
	{
		const uint32_t slot = blockIdx.x / (inter / WARPS), ch = (blockIdx.x % (inter / WARPS)) * WARPS + warp;
		const uint64_t exp_base = (uint64_t)route_expert[slot] * (2u * inter);
		float a_up = 0.0f, a_gt = 0.0f;
		if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
		{
			const uint4 *w_up = (const uint4 *)(exp_w1 + (exp_base + ch) * G5N_HIDDEN);
			const uint4 *w_gt = (const uint4 *)(exp_w1 + (exp_base + inter + ch) * G5N_HIDDEN);
			const float *s_up = exp_s1 + (exp_base + ch) * SCALES, *s_gt = exp_s1 + (exp_base + inter + ch) * SCALES;
			uint32_t c = lane;
			for ( ; c + 3u * LM_WARP_LANES < FP8_CHUNKS; c += 4u * LM_WARP_LANES )
			{
				uint4 wu[4], wg[4];
				float su[4], sg[4];
				LmSkinnyFormat<LmFp8>::Activation a[4];
				#pragma unroll
				for ( uint32_t u = 0u; u < 4u; ++u )
				{
					const uint32_t ci = c + u * LM_WARP_LANES, sgi = ci >> 3u;
					wu[u] = __ldcs(w_up + ci);
					wg[u] = __ldcs(w_gt + ci);
					su[u] = __ldg(s_up + sgi);
					sg[u] = __ldg(s_gt + sgi);
					a[u] = LmSkinnyFormat<LmFp8>::Load(act + ci * 16u);
				}
				#pragma unroll
				for ( uint32_t u = 0u; u < 4u; ++u )
				{
					a_up = fmaf(su[u], LmSkinnyFormat<LmFp8>::Chunk(wu[u], a[u]), a_up);
					a_gt = fmaf(sg[u], LmSkinnyFormat<LmFp8>::Chunk(wg[u], a[u]), a_gt);
				}
			}
			for ( ; c < FP8_CHUNKS; c += LM_WARP_LANES )
			{
				const LmSkinnyFormat<LmFp8>::Activation a = LmSkinnyFormat<LmFp8>::Load(act + c * 16u);
				const uint32_t sg = c >> 3u;
				a_up = fmaf(__ldg(s_up + sg), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w_up + c), a), a_up);
				a_gt = fmaf(__ldg(s_gt + sg), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w_gt + c), a), a_gt);
			}
		}
		else
		{
			const uint16_t *exp_bf16 = (const uint16_t *)exp_w1;
			const uint4 *w_up = (const uint4 *)(exp_bf16 + (exp_base + ch) * G5N_HIDDEN);
			const uint4 *w_gt = (const uint4 *)(exp_bf16 + (exp_base + inter + ch) * G5N_HIDDEN);
			uint32_t c = lane;
			for ( ; c + 3u * LM_WARP_LANES < BF16_CHUNKS; c += 4u * LM_WARP_LANES )
			{
				uint4 wu[4], wg[4], a[4];
				#pragma unroll
				for ( uint32_t u = 0u; u < 4u; ++u )
				{
					const uint32_t ci = c + u * LM_WARP_LANES;
					wu[u] = __ldcs(w_up + ci);
					wg[u] = __ldcs(w_gt + ci);
					a[u] = __ldg((const uint4 *)act + ci);
				}
				#pragma unroll
				for ( uint32_t u = 0u; u < 4u; ++u )
				{
					a_up = fmaf(1.0f, LmSkinnyBf16Chunk(wu[u], a[u], 0.0f), a_up);
					a_gt = fmaf(1.0f, LmSkinnyBf16Chunk(wg[u], a[u], 0.0f), a_gt);
				}
			}
			for ( ; c < BF16_CHUNKS; c += LM_WARP_LANES )
			{
				const uint4 a = __ldg((const uint4 *)act + c);
				a_up = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w_up + c), a, 0.0f), a_up);
				a_gt = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w_gt + c), a, 0.0f), a_gt);
			}
		}
		#pragma unroll
		for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
		{
			a_up += __shfl_xor_sync(0xffffffffu, a_up, step);
			a_gt += __shfl_xor_sync(0xffffffffu, a_gt, step);
		}
		if ( lane == 0u )
			inter_out[(uint64_t)slot * inter + ch] = LmFloatToBf16(LmClampedUpGateOp{G5N_SWIGLU_LIMIT}(LmBf16ToFloat(LmFloatToBf16(a_gt)), LmBf16ToFloat(LmFloatToBf16(a_up))));
		LmDependentRelease();
		return;
	}
	__shared__ float partial[4][WARPS];
	const uint32_t ch0 = (blockIdx.x - expert_blocks) * 2u;
	const uint32_t first = BF16_CHUNKS * warp / WARPS, last = BF16_CHUNKS * (warp + 1u) / WARPS;
	const uint4 *wu0 = (const uint4 *)(shared_w1 + (uint64_t)ch0 * G5N_HIDDEN), *wg0 = (const uint4 *)(shared_w1 + (uint64_t)(inter + ch0) * G5N_HIDDEN);
	const uint4 *wu1 = (const uint4 *)(shared_w1 + (uint64_t)(ch0 + 1u) * G5N_HIDDEN), *wg1 = (const uint4 *)(shared_w1 + (uint64_t)(inter + ch0 + 1u) * G5N_HIDDEN);
	float u0 = 0.0f, g0 = 0.0f, u1 = 0.0f, g1 = 0.0f;
	for ( uint32_t c = first + lane; c < last; c += LM_WARP_LANES )
	{
		const uint4 a = __ldg((const uint4 *)act + c);
		u0 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(wu0 + c), a, 0.0f), u0);
		g0 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(wg0 + c), a, 0.0f), g0);
		u1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(wu1 + c), a, 0.0f), u1);
		g1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(wg1 + c), a, 0.0f), g1);
	}
	#pragma unroll
	for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
	{
		u0 += __shfl_xor_sync(0xffffffffu, u0, step);
		g0 += __shfl_xor_sync(0xffffffffu, g0, step);
		u1 += __shfl_xor_sync(0xffffffffu, u1, step);
		g1 += __shfl_xor_sync(0xffffffffu, g1, step);
	}
	if ( lane == 0u )
	{
		partial[0][warp] = u0;
		partial[1][warp] = g0;
		partial[2][warp] = u1;
		partial[3][warp] = g1;
	}
	__syncthreads();
	if ( threadIdx.x < 2u )
	{
		float tu = 0.0f, tg = 0.0f;
		#pragma unroll
		for ( uint32_t w = 0u; w < WARPS; ++w )
		{
			tu += partial[threadIdx.x * 2u][w];
			tg += partial[threadIdx.x * 2u + 1u][w];
		}
		shared_inter_out[ch0 + threadIdx.x] = LmFloatToBf16(LmClampedUpGateOp{G5N_SWIGLU_LIMIT}(LmBf16ToFloat(LmFloatToBf16(tg)), LmBf16ToFloat(LmFloatToBf16(tu))));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u) void Glm5NextMoeB1W2FinalizeKernel(const uint8_t *__restrict__ exp_w2, const float *__restrict__ exp_s2,
	const uint16_t *__restrict__ shared_w2, const uint16_t *__restrict__ inter_in, const uint16_t *__restrict__ shared_inter_in,
	const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t INTER = 128u, ROWS_PER_BLOCK = 32u;
	__shared__ __align__(16) uint16_t s_inter[G5N_TOP_K + 1u][INTER];
	__shared__ uint32_t s_expert[G5N_TOP_K];
	__shared__ float s_weight[G5N_TOP_K];
	const uint32_t group = threadIdx.x >> 3u, lane = threadIdx.x & 7u, out_row = blockIdx.x * ROWS_PER_BLOCK + group;
	LmDependentWait();
	if ( threadIdx.x < G5N_TOP_K )
	{
		s_expert[threadIdx.x] = route_expert[threadIdx.x];
		s_weight[threadIdx.x] = route_weight[threadIdx.x];
	}
	for ( uint32_t v = threadIdx.x; v < G5N_TOP_K * (INTER / 8u); v += 256u )
		((uint4 *)s_inter)[v] = __ldg((const uint4 *)inter_in + v);
	if ( threadIdx.x < INTER / 8u )
		((uint4 *)s_inter[G5N_TOP_K])[threadIdx.x] = __ldg((const uint4 *)shared_inter_in + threadIdx.x);
	__syncthreads();
	const uint64_t row0 = (uint64_t)s_expert[0] * G5N_HIDDEN + out_row;
	uint4 w_cur = __ldcs((const uint4 *)(exp_w2 + row0 * INTER) + lane);
	float sc_cur = __ldg(exp_s2 + row0), acc = 0.0f;
	for ( uint32_t slot = 0u; slot < G5N_TOP_K; ++slot )
	{
		uint4 w_next;
		float sc_next;
		if ( slot + 1u < G5N_TOP_K )
		{
			const uint64_t nr = (uint64_t)s_expert[slot + 1u] * G5N_HIDDEN + out_row;
			w_next = __ldcs((const uint4 *)(exp_w2 + nr * INTER) + lane);
			sc_next = __ldg(exp_s2 + nr);
		}
		LmSkinnyFormat<LmFp8>::Activation a;
		a.low = ((const uint4 *)s_inter[slot])[2u * lane];
		a.high = ((const uint4 *)s_inter[slot])[2u * lane + 1u];
		float dot = fmaf(sc_cur, LmSkinnyFormat<LmFp8>::Chunk(w_cur, a), 0.0f);
		dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
		dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
		dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
		if ( lane == 0u )
			acc = fmaf(LmBf16ToFloat(LmFloatToBf16(dot)), s_weight[slot], acc);
		if ( slot + 1u < G5N_TOP_K )
		{
			w_cur = w_next;
			sc_cur = sc_next;
		}
	}
	const uint4 *sw = (const uint4 *)(shared_w2 + (uint64_t)out_row * INTER);
	const uint4 sa0 = ((const uint4 *)s_inter[G5N_TOP_K])[lane], sa1 = ((const uint4 *)s_inter[G5N_TOP_K])[lane + 8u];
	float sdot = LmSkinnyBf16Chunk(__ldcs(sw + lane), sa0, 0.0f) + LmSkinnyBf16Chunk(__ldcs(sw + lane + 8u), sa1, 0.0f);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 4u);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 2u);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 1u);
	if ( lane == 0u )
		out_bf16[out_row] = LmFloatToBf16(SparkMathRoundBf16(acc) + LmBf16ToFloat(LmFloatToBf16(sdot)));
	LmDependentRelease();
}

__global__ __launch_bounds__(256u) void Glm5NextMoeB1W2FinalizeBf16Kernel(const uint16_t *__restrict__ exp_w2, const uint16_t *__restrict__ shared_w2,
	const uint16_t *__restrict__ inter_in, const uint16_t *__restrict__ shared_inter_in, const uint32_t *__restrict__ route_expert,
	const float *__restrict__ route_weight, uint16_t *__restrict__ out_bf16)
{
	constexpr uint32_t INTER = 128u, ROWS_PER_BLOCK = 16u;
	__shared__ __align__(16) uint16_t s_inter[G5N_TOP_K + 1u][INTER];
	__shared__ uint32_t s_expert[G5N_TOP_K];
	__shared__ float s_weight[G5N_TOP_K];
	const uint32_t group = threadIdx.x >> 4u, lane = threadIdx.x & 15u, out_row = blockIdx.x * ROWS_PER_BLOCK + group;
	LmDependentWait();
	if ( threadIdx.x < G5N_TOP_K )
	{
		s_expert[threadIdx.x] = route_expert[threadIdx.x];
		s_weight[threadIdx.x] = route_weight[threadIdx.x];
	}
	for ( uint32_t v = threadIdx.x; v < G5N_TOP_K * (INTER / 8u); v += 256u )
		((uint4 *)s_inter)[v] = __ldg((const uint4 *)inter_in + v);
	if ( threadIdx.x < INTER / 8u )
		((uint4 *)s_inter[G5N_TOP_K])[threadIdx.x] = __ldg((const uint4 *)shared_inter_in + threadIdx.x);
	__syncthreads();
	const uint64_t row0 = (uint64_t)s_expert[0] * G5N_HIDDEN + out_row;
	uint4 w_cur = __ldcs((const uint4 *)(exp_w2 + row0 * INTER) + lane);
	float acc = 0.0f;
	for ( uint32_t slot = 0u; slot < G5N_TOP_K; ++slot )
	{
		uint4 w_next;
		if ( slot + 1u < G5N_TOP_K )
		{
			const uint64_t nr = (uint64_t)s_expert[slot + 1u] * G5N_HIDDEN + out_row;
			w_next = __ldcs((const uint4 *)(exp_w2 + nr * INTER) + lane);
		}
		float dot = LmSkinnyBf16Chunk(w_cur, ((const uint4 *)s_inter[slot])[lane], 0.0f);
		dot += __shfl_xor_sync(0xffffffffu, dot, 8u);
		dot += __shfl_xor_sync(0xffffffffu, dot, 4u);
		dot += __shfl_xor_sync(0xffffffffu, dot, 2u);
		dot += __shfl_xor_sync(0xffffffffu, dot, 1u);
		if ( lane == 0u )
			acc = fmaf(LmBf16ToFloat(LmFloatToBf16(dot)), s_weight[slot], acc);
		if ( slot + 1u < G5N_TOP_K )
			w_cur = w_next;
	}
	const uint4 *sw = (const uint4 *)(shared_w2 + (uint64_t)out_row * INTER);
	float sdot = LmSkinnyBf16Chunk(__ldcs(sw + lane), ((const uint4 *)s_inter[G5N_TOP_K])[lane], 0.0f);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 8u);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 4u);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 2u);
	sdot += __shfl_xor_sync(0xffffffffu, sdot, 1u);
	if ( lane == 0u )
		out_bf16[out_row] = LmFloatToBf16(SparkMathRoundBf16(acc) + LmBf16ToFloat(LmFloatToBf16(sdot)));
	LmDependentRelease();
}

static __global__ __launch_bounds__(256u) void Glm5NextClampedUpGatePacked128Kernel(const uint16_t *__restrict__ up_gate_bf16,
	uint16_t *__restrict__ output_bf16, uint32_t rows)
{
	const uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u), chunk = threadIdx.x & 15u;
	LmDependentWait();
	if ( row < rows )
	{
		const uint64_t in_base = (uint64_t)row * 256u;
		const uint4 vu = __ldcs((const uint4 *)(up_gate_bf16 + in_base) + chunk);
		const uint4 vg = __ldcs((const uint4 *)(up_gate_bf16 + in_base + 128u) + chunk);
		float fu[8], fg[8], fo[8];
		LmLoadBf16x8AsFloat(vu, fu);
		LmLoadBf16x8AsFloat(vg, fg);
		const LmClampedUpGateOp op{G5N_SWIGLU_LIMIT};
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			fo[k] = op(fg[k], fu[k]);
		*((uint4 *)(output_bf16 + (uint64_t)row * 128u) + chunk) = LmPackUint4FloatToBf16(fo);
	}
	LmDependentRelease();
}

static __global__ __launch_bounds__(256u) void Glm5NextMoeFinalizeAddKernel(const uint16_t *__restrict__ packed_bf16,
	const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight, const uint16_t *__restrict__ addend_bf16,
	uint16_t *__restrict__ output_bf16, uint32_t tokens)
{
	__shared__ uint32_t s_row[G5N_TOP_K];
	__shared__ float s_weight[G5N_TOP_K];
	const uint32_t element = (blockIdx.x * 256u + threadIdx.x) * 8u;
	LmDependentWait();
	for ( uint32_t token = blockIdx.y; token < tokens; token += gridDim.y )
	{
		if ( threadIdx.x < G5N_TOP_K )
		{
			const uint64_t slot = (uint64_t)token * G5N_TOP_K + threadIdx.x;
			s_row[threadIdx.x] = packed_row_of_token_route[slot];
			s_weight[threadIdx.x] = route_weight[slot];
		}
		__syncthreads();
		if ( element < G5N_HIDDEN )
		{
			float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, add[8];
			uint4 v_cur = __ldcs((const uint4 *)(packed_bf16 + (uint64_t)s_row[0] * G5N_HIDDEN + element));
			#pragma unroll
			for ( uint32_t route = 0u; route < G5N_TOP_K; ++route )
			{
				uint4 v_next;
				if ( route + 1u < G5N_TOP_K )
					v_next = __ldcs((const uint4 *)(packed_bf16 + (uint64_t)s_row[route + 1u] * G5N_HIDDEN + element));
				LmFmaBf16x8(total, v_cur, s_weight[route]);
				if ( route + 1u < G5N_TOP_K )
					v_cur = v_next;
			}
			LmLoadBf16x8AsFloat(__ldcs((const uint4 *)(addend_bf16 + (uint64_t)token * G5N_HIDDEN + element)), add);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				total[k] = SparkMathRoundBf16(total[k]) + add[k];
			*(uint4 *)(output_bf16 + (uint64_t)token * G5N_HIDDEN + element) = LmPackUint4FloatToBf16(total);
		}
		__syncthreads();
	}
	LmDependentRelease();
}

template<class Format>
static uint32_t Glm5NextStreamExperts(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t threshold = G5N_STREAM_EXPERT_ROWS)
{
	if constexpr ( LmStreamWeight<Format>::kSupported )
		return rows >= threshold && b->row_invariant == 0u ? 1u : 0u;
	else
		return 0u;
}

#define G5N_EXPERTS_REFUSE(kernel, weight, rows, pairs, output_dimension, input_dimension) \
	LM_LAUNCH_REFUSE("glm5 experts: the %s refused %s for %u rows (%u routed pairs over %u experts, %u x %u); waves of %u rows or more that are " \
		"not row-invariant run on the grouped stream GEMM and all others on the skinny kernels, so the expert weights need 16-byte alignment, an " \
		"input that is a multiple of the codec's elements and scales that cover every expert\n", kernel, weight, rows, pairs, G5N_EXPERTS, \
		output_dimension, input_dimension, G5N_STREAM_EXPERT_ROWS)

template<uint32_t ExpertCodec>
static int32_t Glm5NextMoeExperts(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
	static_assert((G5N_STREAM_EXPERT_ROWS - 1u) * G5N_TOP_K <= LM_SKINNY_GROUPED_MAX_MEAN_ROWS * G5N_EXPERTS,
		"every wave below the stream threshold fits the grouped skinny expert kernel, whose mean rows per expert are bounded");
	static_assert(ExpertFormat::kScaleGroup == 0u || (G5N_HIDDEN % ExpertFormat::kScaleGroup == 0u && G5N_EXPERT_INTER % ExpertFormat::kScaleGroup == 0u),
		"routed expert dimensions hold whole codec scale groups");
	static_assert(G5N_HIDDEN % LM_MOE_FINALIZE_ADD_ELEMENTS == 0u, "the expert combine reads whole 16-byte words of a row");
	const uint32_t packed_rows = rows * G5N_TOP_K;
	const uint32_t pairwise = rows == 1u ? 1u : 0u;
	LmScaleTensor scale_b;
	int32_t status;
	if ( b->expert_w1_weight == 0 || b->expert_w2_weight == 0 ||
		(LmWeightCodec<ExpertCodec>::kScaleEncoding != LM_SCALE_ENCODING_NONE && (b->expert_w1_scale == 0 || b->expert_w2_scale == 0)) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 || ExpertCodec == SPARK_WEIGHT_CODEC_BF16 )
		if ( rows == 1u && b->expert_intermediate == 128u && b->shared_intermediate == 128u &&
			LmIsAligned16(b->expert_w1_weight) && LmIsAligned16(b->expert_w2_weight) &&
			LmIsAligned16(b->w->shared_gate_up) && LmIsAligned16(b->w->shared_down) && LmIsAligned16(b->normed_bf16) )
		{
			const uint32_t w1_blocks = G5N_TOP_K * (b->expert_intermediate / LM_SKINNY_SPLIT_WARPS) + b->shared_intermediate / 2u;
			LM_LAUNCH_DEPENDENT((Glm5NextMoeB1W1ActKernel<ExpertCodec>), w1_blocks, LM_SKINNY_THREADS, 0, stream,
				(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, (const uint16_t *)b->w->shared_gate_up,
				b->normed_bf16, b->route_expert, b->intermediate_bf16, b->gate_up_bf16, b->expert_intermediate);
			if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
				LM_LAUNCH_DEPENDENT((Glm5NextMoeB1W2FinalizeKernel), G5N_HIDDEN / 32u, 256u, 0, stream,
					(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, (const uint16_t *)b->w->shared_down,
					b->intermediate_bf16, b->gate_up_bf16, b->route_expert, b->route_weight, b->attention_out_bf16);
			else
				LM_LAUNCH_DEPENDENT((Glm5NextMoeB1W2FinalizeBf16Kernel), G5N_HIDDEN / 16u, 256u, 0, stream,
					(const uint16_t *)b->expert_w2_weight, (const uint16_t *)b->w->shared_down, b->intermediate_bf16, b->gate_up_bf16,
					b->route_expert, b->route_weight, b->attention_out_bf16);
			return Glm5NextLaunched();
		}
	scale_b = LmWeightCodecScaleTensor<ExpertCodec>(b->expert_w1_scale, G5N_EXPERTS, b->expert_w1_rows, G5N_HIDDEN);
	if ( Glm5NextStreamExperts<ExpertFormat>(b, rows) != 0u )
	{
		status = LmStreamGemmGrouped<ExpertFormat>(b->expert_w1_weight, scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
			b->route_source_token, G5N_EXPERTS, packed_rows, G5N_HIDDEN, b->expert_w1_rows, sms, stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			G5N_EXPERTS_REFUSE("grouped stream GEMM", "w1", rows, packed_rows, b->expert_w1_rows, G5N_HIDDEN);
	}
	else
	{
		if ( pairwise != 0u )
			status = LmSkinnyExperts<ExpertFormat>(b->expert_w1_weight, scale_b, b->normed_bf16, b->gate_up_bf16, b->route_expert,
				b->route_packed_row, packed_rows, G5N_TOP_K, 0u, G5N_HIDDEN, b->expert_w1_rows, stream);
		else
			status = LmSkinnyGroupedExperts<ExpertFormat>(b->expert_w1_weight, scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
				b->route_source_token, G5N_EXPERTS, packed_rows, 0u, G5N_HIDDEN, b->expert_w1_rows, stream, b->row_invariant);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			G5N_EXPERTS_REFUSE(pairwise != 0u ? "per-pair skinny kernel" : "grouped skinny kernel", "w1", rows, packed_rows, b->expert_w1_rows,
				G5N_HIDDEN);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( b->expert_intermediate == 128u && LmIsAligned16(b->gate_up_bf16) && LmIsAligned16(b->intermediate_bf16) )
		LM_LAUNCH_DEPENDENT((Glm5NextClampedUpGatePacked128Kernel), (packed_rows + 15u) / 16u, 256u, 0, stream, b->gate_up_bf16,
			b->intermediate_bf16, packed_rows);
	else
		LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), packed_rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16,
			b->expert_intermediate, G5N_SWIGLU_LIMIT);
	scale_b = LmWeightCodecScaleTensor<ExpertCodec>(b->expert_w2_scale, G5N_EXPERTS, G5N_HIDDEN, b->expert_intermediate);
	if ( Glm5NextStreamExperts<ExpertFormat>(b, rows, G5N_STREAM_EXPERT_W2_ROWS) != 0u )
	{
		status = LmStreamGemmGrouped<ExpertFormat>(b->expert_w2_weight, scale_b, b->intermediate_bf16, b->expert_out_bf16,
			b->group_row_offset, 0, G5N_EXPERTS, packed_rows, b->expert_intermediate, G5N_HIDDEN, sms, stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			G5N_EXPERTS_REFUSE("grouped stream GEMM", "w2", rows, packed_rows, G5N_HIDDEN, b->expert_intermediate);
	}
	else
	{
		if ( pairwise != 0u )
			status = LmSkinnyExperts<ExpertFormat>(b->expert_w2_weight, scale_b, b->intermediate_bf16, b->expert_out_bf16, b->route_expert,
				b->route_packed_row, packed_rows, G5N_TOP_K, 1u, b->expert_intermediate, G5N_HIDDEN, stream);
		else
			status = LmSkinnyGroupedExperts<ExpertFormat>(b->expert_w2_weight, scale_b, b->intermediate_bf16, b->expert_out_bf16,
				b->group_row_offset, b->route_source_token, G5N_EXPERTS, packed_rows, 1u, b->expert_intermediate, G5N_HIDDEN, stream, b->row_invariant);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			G5N_EXPERTS_REFUSE(pairwise != 0u ? "per-pair skinny kernel" : "grouped skinny kernel", "w2", rows, packed_rows, G5N_HIDDEN,
				b->expert_intermediate);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	status = Glm5NextLinear(b, b->normed_bf16, b->w->shared_gate_up, b->gate_up_bf16, rows, G5N_HIDDEN, b->shared_gate_up_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( b->shared_intermediate == 128u && LmIsAligned16(b->gate_up_bf16) && LmIsAligned16(b->intermediate_bf16) )
		LM_LAUNCH_DEPENDENT((Glm5NextClampedUpGatePacked128Kernel), (rows + 15u) / 16u, 256u, 0, stream, b->gate_up_bf16,
			b->intermediate_bf16, rows);
	else
		LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->shared_intermediate,
			G5N_SWIGLU_LIMIT);
	status = Glm5NextLinear(b, b->intermediate_bf16, b->w->shared_down, b->shared_out_bf16, rows, b->shared_intermediate, G5N_HIDDEN, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH_DEPENDENT((Glm5NextMoeFinalizeAddKernel), dim3((G5N_HIDDEN / 8u + 255u) / 256u, rows), 256u, 0, stream, b->expert_out_bf16,
		b->route_packed_row, b->route_weight, b->shared_out_bf16, b->attention_out_bf16, rows);
	return Glm5NextLaunched();
}

#include "glm5_next_stage_model.cuh"
