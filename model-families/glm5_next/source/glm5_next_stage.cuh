#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_prefill.cuh"
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
#include "skl2_glm5next_embedded.h"

struct Glm5NextSkl2BundleTag {};

template<typename... Args>
static inline SparkGpuError Glm5NextSkl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Args... args)
{
	return LmLaunchEmbedded<Glm5NextSkl2BundleTag>(kGlm5NextSkl2Cubin, sizeof(kGlm5NextSkl2Cubin), (const char *)kGlm5NextSkl2Manifest,
		sizeof(kGlm5NextSkl2Manifest) - 1u, kGlm5NextSkl2Names, (uint32_t)GLM5NEXTSKL2_COUNT, id, grid, block, shared, stream, args...);
}

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
#define G5N_STREAM_EXPERT_W2_ROWS 128u
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
	uint32_t single_sequence;
	uint32_t prefill_unified;
};

static int32_t Glm5NextExactLinearLaunch(const uint16_t *activation, const void *weight, uint16_t *output_bf16, float *output_f32, uint32_t rows,
	uint32_t input_dimension, uint32_t output_dimension, uint32_t allow_split, SparkGpuStream stream)
{
	const uint32_t chunks = input_dimension >> 3u;
	const uint32_t use_f32 = output_f32 != 0 ? 1u : 0u;
	if ( allow_split != 0u && output_dimension <= LM_SKINNY_SPLIT_MAX_OUTPUTS && chunks >= LM_SKINNY_SPLIT_WARPS * LM_WARP_LANES )
	{
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_exact_split_linear_256, dim3((output_dimension + 3u) / 4u, (rows + 7u) / 8u), G5N_THREADS, 0u, stream,
			(const uint16_t *)weight, activation, output_bf16, output_f32, rows, input_dimension, output_dimension, use_f32) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return LM_LAUNCH_OK;
	}
	if ( chunks >= 32u )
	{
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_exact_linear_l32_r8, dim3((output_dimension + 3u) / 4u, (rows + 63u) / 64u), G5N_THREADS, 0u, stream,
			(const uint16_t *)weight, activation, output_bf16, output_f32, rows, input_dimension, output_dimension, use_f32) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else if ( chunks >= 16u )
	{
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_exact_linear_l16_r4, dim3((output_dimension + 3u) / 4u, (rows + 63u) / 64u), G5N_THREADS, 0u, stream,
			(const uint16_t *)weight, activation, output_bf16, output_f32, rows, input_dimension, output_dimension, use_f32) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
		return LM_LAUNCH_ERR_SHAPE;
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextSkinnyMulti(const Glm5NextLayerBuffers *b, const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *activation,
	uint32_t rows, uint32_t input_dimension, SparkGpuStream stream)
{
	if ( rows > LM_SKINNY_ROWS_WIDE && b->row_invariant != 0u && (input_dimension & 7u) == 0u && (input_dimension >> 3u) >= 16u &&
		LmIsAligned16(activation) )
	{
		uint32_t ok = 1u;
		for ( uint32_t i = 0u; i < count; ++i )
			if ( targets[i].output_column_offset != 0u || targets[i].output_row_stride != targets[i].output_dimension || !LmIsAligned16(targets[i].weight) )
				ok = 0u;
		if ( ok != 0u )
		{
			for ( uint32_t i = 0u; i < count; ++i )
			{
				const int32_t st = Glm5NextExactLinearLaunch(activation, targets[i].weight, targets[i].output_bf16, targets[i].output_f32, rows,
					input_dimension, targets[i].output_dimension, 0u, stream);
				if ( st != LM_LAUNCH_OK )
					return st;
			}
			return LM_LAUNCH_OK;
		}
	}
	return LmSkinnyDenseMulti<LmBf16Format>(targets, count, activation, rows, input_dimension, stream,
		rows > LM_SKINNY_ROWS_WIDE && b->row_invariant != 0u ? 1u : 0u);
}

static int32_t Glm5NextLinear(const Glm5NextLayerBuffers *b, const uint16_t *activation, const void *weight, uint16_t *output, uint32_t rows,
	uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	if ( multiprocessors == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->prefill_unified != 0u ||
		(b->row_invariant == 0u && rows >= 16u && input_dimension >= 256u && output_dimension >= 512u) )
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, (float *)0, rows, input_dimension,
			output_dimension, 0u, 0u, multiprocessors, stream);
	if ( b->row_invariant != 0u && rows > LM_SKINNY_ROWS_WIDE && (input_dimension & 7u) == 0u && (input_dimension >> 3u) >= 16u &&
		LmIsAligned16(activation) && LmIsAligned16(weight) )
		return Glm5NextExactLinearLaunch(activation, weight, output, (float *)0, rows, input_dimension, output_dimension, 1u, stream);
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

template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS, bool STRIDE = false>
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
	const uint32_t row = blockIdx.y;
	if constexpr ( STRIDE )
		LmDependentWait();
	const uint32_t sequence = sequence_of_row[row];
	const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
	if ( dense_limit == 0u || context > dense_limit )
	{
		const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
		const uint32_t end_local = STRIDE ? local_stride : (blockIdx.x < local_stride ? blockIdx.x + 1u : blockIdx.x);
		for ( uint32_t local = blockIdx.x; local < end_local; local += STRIDE ? gridDim.x : 1u )
		{
			const uint32_t pool = Glm5NextPoolGlobal(local, owner_rank, owner_degree), first = pool * KPOOL;
			if ( pool >= pools || first + KPOOL > context )
			{
				if ( threadIdx.x == 0u )
					pool_scores[(uint64_t)row * local_stride + local] = -INFINITY;
				continue;
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
			if ( pool_ready != 0u )
			{
				for ( uint32_t index = threadIdx.x; index < DIM; index += THREADS )
					pool_key[index] = Glm5NextPoolKeyElement<DIM, KPOOL>(pool_slots, compress_ape_f32, index);
				__syncthreads();
				const float4 k = *(const float4 *)(pool_key + lane * 4u);
				for ( uint32_t slot = 0u; slot < PER_WARP; ++slot )
				{
					const uint32_t head = warp + slot * WARPS;
					float q[4];
					Glm5NextLoadBf16x4(query_bf16 + ((uint64_t)row * HEADS + head) * DIM + lane * 4u, q);
					const float contrib = Glm5NextPoolHeadDot(q, k, softmax_scale,
						LmBf16ToFloat(head_weight_bf16[(uint64_t)row * HEADS + head]), head_scale);
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
			if constexpr ( STRIDE )
				if ( local + gridDim.x < end_local )
					__syncthreads();
		}
	}
	if constexpr ( STRIDE )
		LmDependentRelease();
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

template<uint32_t DIM, uint32_t KPOOL>
__global__ __launch_bounds__(128u, 2) void Glm5NextPoolKeyBuildKernel(LmKvShardView index_cache,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const float *__restrict__ compress_ape_f32, uint32_t pools, uint32_t owner_rank, uint32_t owner_degree,
	uint32_t local_stride, float *__restrict__ pool_keys_out, uint32_t *__restrict__ pool_ready_out)
{
	__shared__ const uint16_t *pool_slots[KPOOL];
	__shared__ uint32_t ready_s;
	const uint32_t local = blockIdx.x;
	if ( local >= local_stride )
		return;
	const uint32_t sequence = sequence_of_row[0], pool = Glm5NextPoolGlobal(local, owner_rank, owner_degree);
	if ( threadIdx.x == 0u )
	{
		uint32_t ready = pool < pools && pool * KPOOL + KPOOL <= context_length[sequence] ? 1u : 0u;
		if ( ready != 0u )
		{
			const uint32_t first = pool * KPOOL;
			for ( uint32_t s = 0u; s < KPOOL; ++s )
			{
				const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<Glm5NextIndexKv>(index_cache, sequence, first + s, 0u, LM_KV_ACCESS_READ);
				pool_slots[s] = slot;
				if ( slot == 0 )
				{
					ready = 0u;
					break;
				}
			}
		}
		ready_s = ready;
		pool_ready_out[local] = ready;
	}
	__syncthreads();
	if ( ready_s == 0u )
		return;
	pool_keys_out[(uint64_t)local * DIM + threadIdx.x] = Glm5NextPoolKeyElement<DIM, KPOOL>(pool_slots, compress_ape_f32, threadIdx.x);
}

template<uint32_t THREADS, uint32_t DIM, uint32_t KPOOL, uint32_t HEADS, uint32_t POOLS, uint32_t ROWS>
__global__ __launch_bounds__(THREADS, 2) void Glm5NextPoolScoreCachedKernel(const uint16_t *__restrict__ query_bf16,
	const uint16_t *__restrict__ head_weight_bf16, const float *__restrict__ pool_keys_in, const uint32_t *__restrict__ pool_ready_in,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions,
	uint32_t pools, uint32_t owner_rank, uint32_t owner_degree, uint32_t local_stride, float softmax_scale, float head_scale,
	float *__restrict__ pool_scores, uint32_t rows)
{
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES, PER_WARP = HEADS / WARPS;
	__shared__ __align__(16) float pool_key[POOLS][DIM];
	__shared__ uint32_t pool_ready[POOLS];
	__shared__ float contribution[2][POOLS][HEADS];
	const uint32_t first_local = blockIdx.x * POOLS, first_row = blockIdx.y * ROWS;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	for ( uint32_t local = threadIdx.x; local < POOLS; local += THREADS )
		pool_ready[local] = first_local + local < local_stride ? pool_ready_in[first_local + local] : 0u;
	__syncthreads();
	for ( uint32_t item = threadIdx.x; item < POOLS * (DIM / 4u); item += THREADS )
	{
		const uint32_t local = item / (DIM / 4u), vec = item % (DIM / 4u);
		if ( pool_ready[local] != 0u )
			((float4 *)pool_key[local])[vec] = __ldg((const float4 *)(pool_keys_in + (uint64_t)(first_local + local) * DIM) + vec);
	}
	__syncthreads();
	for ( uint32_t offset = 0u; offset < ROWS && first_row + offset < rows; offset += 2u )
	{
		const uint32_t row0 = first_row + offset, has1 = (offset + 1u < ROWS && row0 + 1u < rows) ? 1u : 0u;
		const uint32_t row1 = has1 != 0u ? row0 + 1u : row0;
		float q0[PER_WARP][4], w0[PER_WARP], q1[PER_WARP][4], w1[PER_WARP];
		#pragma unroll
		for ( uint32_t slot = 0u; slot < PER_WARP; ++slot )
		{
			const uint32_t head = warp + slot * WARPS;
			Glm5NextLoadBf16x4(query_bf16 + ((uint64_t)row0 * HEADS + head) * DIM + lane * 4u, q0[slot]);
			w0[slot] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row0 * HEADS + head]);
			if ( has1 != 0u )
			{
				Glm5NextLoadBf16x4(query_bf16 + ((uint64_t)row1 * HEADS + head) * DIM + lane * 4u, q1[slot]);
				w1[slot] = LmBf16ToFloat(head_weight_bf16[(uint64_t)row1 * HEADS + head]);
			}
		}
		for ( uint32_t local = 0u; local < POOLS; ++local )
		{
			if ( pool_ready[local] == 0u )
				continue;
			const float4 k = *(const float4 *)(pool_key[local] + lane * 4u);
			#pragma unroll
			for ( uint32_t slot = 0u; slot < PER_WARP; ++slot )
			{
				const float c0 = Glm5NextPoolHeadDot(q0[slot], k, softmax_scale, w0[slot], head_scale);
				if ( lane == 0u )
					contribution[0][local][warp + slot * WARPS] = c0;
				if ( has1 != 0u )
				{
					const float c1 = Glm5NextPoolHeadDot(q1[slot], k, softmax_scale, w1[slot], head_scale);
					if ( lane == 0u )
						contribution[1][local][warp + slot * WARPS] = c1;
				}
			}
		}
		__syncthreads();
		const uint32_t pair_items = (1u + has1) * POOLS;
		for ( uint32_t idx = threadIdx.x; idx < pair_items; idx += THREADS )
		{
			const uint32_t ri = idx / POOLS, local = idx % POOLS;
			if ( first_local + local >= local_stride )
				continue;
			const uint32_t row = ri == 0u ? row0 : row1;
			const uint32_t sequence = sequence_of_row[row];
			const uint32_t context = context_length[sequence] < row_positions[row] + 1u ? context_length[sequence] : row_positions[row] + 1u;
			const uint32_t pool = Glm5NextPoolGlobal(first_local + local, owner_rank, owner_degree);
			if ( pool >= pools || pool * KPOOL + KPOOL > context )
			{
				pool_scores[(uint64_t)row * local_stride + first_local + local] = -INFINITY;
				continue;
			}
			if ( pool_ready[local] == 0u )
				continue;
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t head = 0u; head < HEADS; ++head )
				total += contribution[ri][local][head];
			pool_scores[(uint64_t)row * local_stride + first_local + local] = total;
		}
		__syncthreads();
	}
}

#if defined(__CUDACC__)
struct Glm5NextPoolScoreMmaShared
{
	__align__(16) uint32_t khi[64u][G5N_INDEX_DIM + 4u];
	__align__(16) uint32_t klo[64u][G5N_INDEX_DIM + 4u];
	__align__(16) uint16_t q[G5N_INDEX_HEADS][G5N_INDEX_DIM + 8u];
	uint32_t ready[64u];
	float hw[G5N_INDEX_HEADS];
	float sum[2u][64u];
};

__global__ __launch_bounds__(256u, 1)
static void Glm5NextPoolScoreMmaKernel(const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ head_weight_bf16,
	const float *__restrict__ pool_keys_in, const uint32_t *__restrict__ pool_ready_in, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_positions, uint32_t pools, uint32_t owner_rank,
	uint32_t owner_degree, uint32_t local_stride, float softmax_scale, float head_scale, float *__restrict__ pool_scores, uint32_t rows)
{
	constexpr uint32_t DIM = G5N_INDEX_DIM, KPOOL = G5N_KPOOL, HEADS = G5N_INDEX_HEADS, TILE_P = 64u, ROWS_PER_BLOCK = 16u;
	extern __shared__ __align__(16) uint8_t pool_mma_smem[];
	Glm5NextPoolScoreMmaShared &sh = *(Glm5NextPoolScoreMmaShared *)pool_mma_smem;
	const uint32_t first_local = blockIdx.x * TILE_P, first_row = blockIdx.y * ROWS_PER_BLOCK;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, gid = lane >> 2u, tg = lane & 3u;
	const uint32_t warp_m = warp >> 2u, warp_n = warp & 3u;
	LmDependentWait();
	if ( threadIdx.x < TILE_P )
		sh.ready[threadIdx.x] = first_local + threadIdx.x < local_stride ? __ldg(pool_ready_in + first_local + threadIdx.x) : 0u;
	__syncthreads();
	for ( uint32_t idx = threadIdx.x; idx < TILE_P * (DIM / 4u); idx += 256u )
	{
		const uint32_t lp = idx >> 5u, c0 = (idx & 31u) << 2u;
		if ( sh.ready[lp] != 0u )
		{
			const float4 kv = __ldg((const float4 *)(pool_keys_in + (uint64_t)(first_local + lp) * DIM + c0));
			uint32_t hi[4], lo[4];
			LmDeltaSplitTf32(kv.x, hi[0], lo[0]);
			LmDeltaSplitTf32(kv.y, hi[1], lo[1]);
			LmDeltaSplitTf32(kv.z, hi[2], lo[2]);
			LmDeltaSplitTf32(kv.w, hi[3], lo[3]);
			*(uint4 *)&sh.khi[lp][c0] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
			*(uint4 *)&sh.klo[lp][c0] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
		}
		else
		{
			const uint4 z = make_uint4(0u, 0u, 0u, 0u);
			*(uint4 *)&sh.khi[lp][c0] = z;
			*(uint4 *)&sh.klo[lp][c0] = z;
		}
	}
	__syncthreads();
	for ( uint32_t offset = 0u; offset < ROWS_PER_BLOCK && first_row + offset < rows; ++offset )
	{
		const uint32_t row = first_row + offset;
		const uint32_t sequence = __ldg(sequence_of_row + row);
		const uint32_t ctx_len = __ldg(context_length + sequence);
		const uint32_t rpos = __ldg(row_positions + row) + 1u;
		const uint32_t context = ctx_len < rpos ? ctx_len : rpos;
		const uint16_t *q_row = query_bf16 + (uint64_t)row * HEADS * DIM;
		for ( uint32_t v = threadIdx.x; v < HEADS * (DIM / 8u); v += 256u )
			*(uint4 *)&sh.q[v >> 4u][(v & 15u) << 3u] = __ldg((const uint4 *)q_row + v);
		if ( threadIdx.x < HEADS )
			sh.hw[threadIdx.x] = LmBf16ToFloat(__ldg(head_weight_bf16 + (uint64_t)row * HEADS + threadIdx.x)) * head_scale;
		__syncthreads();
		float acc[2][4];
		acc[0][0] = acc[0][1] = acc[0][2] = acc[0][3] = 0.0f;
		acc[1][0] = acc[1][1] = acc[1][2] = acc[1][3] = 0.0f;
		const uint32_t h_lo = warp_m * 16u + gid, h_hi = h_lo + 8u;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < DIM / 8u; ++ks )
		{
			const uint32_t k0 = ks * 8u + tg, k1 = k0 + 4u;
			uint32_t a[4];
			a[0] = (uint32_t)sh.q[h_lo][k0] << 16u;
			a[1] = (uint32_t)sh.q[h_hi][k0] << 16u;
			a[2] = (uint32_t)sh.q[h_lo][k1] << 16u;
			a[3] = (uint32_t)sh.q[h_hi][k1] << 16u;
			#pragma unroll
			for ( uint32_t j = 0u; j < 2u; ++j )
			{
				const uint32_t pr = warp_n * 16u + j * 8u + gid;
				uint32_t bh[2] = { sh.khi[pr][k0], sh.khi[pr][k1] };
				uint32_t bl[2] = { sh.klo[pr][k0], sh.klo[pr][k1] };
				LmDeltaMmaTf32(acc[j], a, bh);
				LmDeltaMmaTf32(acc[j], a, bl);
			}
		}
		const float hw_lo = sh.hw[h_lo], hw_hi = sh.hw[h_hi];
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
		{
			const float s0 = acc[j][0] * softmax_scale, s1 = acc[j][1] * softmax_scale;
			const float s2 = acc[j][2] * softmax_scale, s3 = acc[j][3] * softmax_scale;
			float v_ev = (s0 > 0.0f ? s0 : 0.0f) * hw_lo + (s2 > 0.0f ? s2 : 0.0f) * hw_hi;
			float v_od = (s1 > 0.0f ? s1 : 0.0f) * hw_lo + (s3 > 0.0f ? s3 : 0.0f) * hw_hi;
			v_ev += __shfl_xor_sync(0xffffffffu, v_ev, 16u);
			v_od += __shfl_xor_sync(0xffffffffu, v_od, 16u);
			v_ev += __shfl_xor_sync(0xffffffffu, v_ev, 8u);
			v_od += __shfl_xor_sync(0xffffffffu, v_od, 8u);
			v_ev += __shfl_xor_sync(0xffffffffu, v_ev, 4u);
			v_od += __shfl_xor_sync(0xffffffffu, v_od, 4u);
			if ( gid == 0u )
			{
				const uint32_t p_ev = warp_n * 16u + j * 8u + 2u * tg;
				sh.sum[warp_m][p_ev] = v_ev;
				sh.sum[warp_m][p_ev + 1u] = v_od;
			}
		}
		__syncthreads();
		if ( threadIdx.x < TILE_P )
		{
			const uint32_t local = threadIdx.x;
			if ( first_local + local < local_stride )
			{
				const uint32_t pool = Glm5NextPoolGlobal(first_local + local, owner_rank, owner_degree);
				if ( pool >= pools || pool * KPOOL + KPOOL > context )
					pool_scores[(uint64_t)row * local_stride + first_local + local] = -INFINITY;
				else if ( sh.ready[local] != 0u )
					pool_scores[(uint64_t)row * local_stride + first_local + local] = sh.sum[0][local] + sh.sum[1][local];
			}
		}
		if ( offset + 1u < ROWS_PER_BLOCK && first_row + offset + 1u < rows )
			__syncthreads();
	}
	LmDependentRelease();
}

static int32_t Glm5NextPoolScoreMmaLaunch(const uint16_t *query_bf16, const uint16_t *head_weight_bf16,
	const float *pool_keys_in, const uint32_t *pool_ready_in, const uint32_t *sequence_of_row, const uint32_t *context_length,
	const uint32_t *row_positions, uint32_t pools, uint32_t owner_rank, uint32_t owner_degree, uint32_t local_stride,
	float softmax_scale, float head_scale, float *pool_scores, uint32_t rows, SparkGpuStream stream)
{
	static std::atomic<bool> configured(false);
	constexpr uint32_t sh = (uint32_t)sizeof(Glm5NextPoolScoreMmaShared);
	if ( !configured.load(std::memory_order_acquire) )
	{
		if ( SparkGpuFuncSetAttribute((const void *)Glm5NextPoolScoreMmaKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		configured.store(true, std::memory_order_release);
	}
	LM_LAUNCH_DEPENDENT((Glm5NextPoolScoreMmaKernel), dim3((local_stride + 63u) / 64u, (rows + 15u) / 16u), 256u, sh, stream,
		query_bf16, head_weight_bf16, pool_keys_in, pool_ready_in, sequence_of_row, context_length, row_positions,
		pools, owner_rank, owner_degree, local_stride, softmax_scale, head_scale, pool_scores, rows);
	return Glm5NextLaunched();
}
#endif

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

template<bool FUSE_POST>
static __global__ __launch_bounds__(768u, 1) void Glm5NextHcMultiDot24WarpKernel(const uint16_t *__restrict__ streams_in,
	const uint16_t *__restrict__ post_out, const float *__restrict__ prev_post, const float *__restrict__ prev_comb,
	uint16_t *__restrict__ streams_out, const float *__restrict__ fn_f32, float *__restrict__ partials, uint32_t rows)
{
	using S = LmHcSiteShape<G5N_HIDDEN>;
	constexpr uint32_t SLICE_VEC = S::kSlice / 8u;
	constexpr uint32_t SLICE_QUADS = S::kSlice / 4u;
	__shared__ __align__(16) uint16_t staged[2][S::kSlice];
	__shared__ float reduction[2][8];
	__shared__ float post_c[5];
	const uint32_t slice = blockIdx.x, tile = blockIdx.y, tiles = gridDim.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t off = slice * S::kSlice;
	const uint32_t out_stream = slice >> 1u, col_base = (slice & 1u) * S::kSlice;
	const uint32_t row_begin = (rows * tile) / tiles, row_end = (rows * (tile + 1u)) / tiles;
	float4 weight[FUSE_POST ? 1u : S::kVectors];
	const float4 *fn4 = (const float4 *)(fn_f32 + (uint64_t)warp * S::kFlat + off);
	if constexpr ( !FUSE_POST )
	{
		#pragma unroll
		for ( uint32_t vector = 0u; vector < S::kVectors; ++vector )
			weight[vector] = __ldg(fn4 + lane + vector * LM_WARP_LANES);
	}
	LmDependentWait();
	if constexpr ( !FUSE_POST )
	{
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
	}
	for ( uint32_t row = row_begin; row < row_end; ++row )
	{
		const uint32_t buf = row & 1u;
		if constexpr ( FUSE_POST )
		{
			if ( threadIdx.x == 0u )
				post_c[0] = LmBf16ToFloat(LmFloatToBf16(__ldg(prev_post + (uint64_t)row * LM_HC_SITE_STREAMS + out_stream)));
			else if ( threadIdx.x <= 4u )
				post_c[threadIdx.x] = LmBf16ToFloat(LmFloatToBf16(__ldg(prev_comb + (uint64_t)row * 16u + (threadIdx.x - 1u) * LM_HC_SITE_STREAMS + out_stream)));
			__syncthreads();
			if ( threadIdx.x < SLICE_QUADS )
			{
				const uint16_t *src_row = streams_in + (uint64_t)row * S::kFlat;
				const uint16_t *p_row = post_out + (uint64_t)row * G5N_HIDDEN;
				uint16_t *dst_row = streams_out + (uint64_t)row * S::kFlat + off;
				for ( uint32_t half = 0u; half < 2u; ++half )
				{
					const uint32_t local = (threadIdx.x * 2u + half) * 2u, col = col_base + local;
					float v0 = 0.0f, v1 = 0.0f;
					for ( uint32_t s = 0u; s < 4u; ++s )
					{
						const uint32_t r = __ldg((const uint32_t *)(src_row + (uint64_t)s * G5N_HIDDEN + col));
						const float cs = post_c[s + 1u];
						v0 = fmaf(cs, __uint_as_float(r << 16u), v0);
						v1 = fmaf(cs, __uint_as_float(r & 0xffff0000u), v1);
					}
					v0 = LmBf16ToFloat(LmFloatToBf16(v0));
					v1 = LmBf16ToFloat(LmFloatToBf16(v1));
					const uint32_t rp = __ldg((const uint32_t *)(p_row + col));
					const float ps = post_c[0];
					const uint32_t p = LmPackBf16Pair(
						LmBf16ToFloat(LmFloatToBf16(ps * __uint_as_float(rp << 16u))) + v0,
						LmBf16ToFloat(LmFloatToBf16(ps * __uint_as_float(rp & 0xffff0000u))) + v1);
					*(uint32_t *)(staged[buf] + local) = p;
					*(uint32_t *)(dst_row + local) = p;
				}
			}
			__syncthreads();
		}
		else if ( row + 1u < row_end && threadIdx.x < SLICE_VEC )
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
			const float4 wv = FUSE_POST ? __ldg(fn4 + lane + vector * LM_WARP_LANES) : weight[FUSE_POST ? 0u : vector];
			acc = fmaf(wv.x, __uint_as_float(packed.x << 16u), acc);
			acc = fmaf(wv.y, __uint_as_float(packed.x & 0xffff0000u), acc);
			acc = fmaf(wv.z, __uint_as_float(packed.y << 16u), acc);
			acc = fmaf(wv.w, __uint_as_float(packed.y & 0xffff0000u), acc);
		}
		#pragma unroll
		for ( uint32_t step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
			acc += __shfl_xor_sync(0xffffffffu, acc, step);
		if ( lane == 0u )
			row_partials[warp] = acc;
		if constexpr ( !FUSE_POST )
		{
			if ( row + 1u < row_end && threadIdx.x < SLICE_VEC )
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		}
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
	if ( (rows >= 5u || b->prefill_unified != 0u) && LmIsAligned16(streams_in) && LmIsAligned16(b->hc_collapsed_bf16) && LmIsAligned16(norm_weight) && LmIsAligned16(b->normed_bf16) &&
		(post_out == 0 || (streams_out != streams_in && LmIsAligned16(post_out) && LmIsAligned16(streams_out))) )
	{
		const uint32_t chunk_max = rows > 512u ? 512u : rows;
		for ( uint32_t r0 = 0u; r0 < rows; r0 += chunk_max )
		{
			const uint32_t chunk = rows - r0 < chunk_max ? rows - r0 : chunk_max;
			const uint16_t *s_in = streams_in + (uint64_t)r0 * G5N_HC_FLAT;
			uint16_t *s_out = streams_out + (uint64_t)r0 * G5N_HC_FLAT;
			const uint16_t *active = post_out != 0 ? s_out : s_in;
			float *post_p = b->hc_post_f32 + (uint64_t)r0 * G5N_HC;
			float *comb_p = b->hc_comb_f32 + (uint64_t)r0 * (G5N_HC * G5N_HC);
			const uint32_t row_tiles = chunk >= 16u ? 6u : (chunk >= 8u ? 4u : (chunk >= 4u ? 2u : chunk));
			if ( post_out != 0 && chunk < 16u )
				LM_LAUNCH_DEPENDENT((Glm5NextHcMultiDot24WarpKernel<true>), dim3(LM_HC_SITE_CLUSTER, row_tiles), 768u, 0, stream,
					s_in, post_out + (uint64_t)r0 * G5N_HIDDEN, post_p, comb_p, s_out, (const float *)fn, (float *)b->hc_collapsed_bf16, chunk);
			else
			{
				if ( post_out != 0 )
				{
					LM_LAUNCH_DEPENDENT((LmHcPostBf16VecKernel<G5N_HIDDEN>), chunk, G5N_THREADS, 0, stream, post_out + (uint64_t)r0 * G5N_HIDDEN,
						s_in, post_p, comb_p, s_out);
					if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
						SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
				}
				LM_LAUNCH_DEPENDENT((Glm5NextHcMultiDot24WarpKernel<false>), dim3(LM_HC_SITE_CLUSTER, row_tiles), 768u, 0, stream,
					active, (const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, (const float *)fn, (float *)b->hc_collapsed_bf16, chunk);
			}
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			LM_LAUNCH_DEPENDENT((LmHcSiteMultiFinishKernel<G5N_HIDDEN, Glm5NextHcPolicy, true, false, false>), chunk, G5N_THREADS, 0, stream,
				(float *)b->hc_collapsed_bf16, active, (const float *)scale, (const float *)base, (float *)0, (float *)0, post_p, comb_p,
				(uint16_t *)0, (uint16_t *)0, (const uint16_t *)norm_weight, b->normed_bf16 + (uint64_t)r0 * G5N_HIDDEN);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
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

#if defined(__CUDACC__)
#define G5N_KDA_WY_C 32u

struct Glm5NextKdaWyPrepShared
{
	__align__(16) float carried[G5N_KDA_WY_C][G5N_KDA_KEY];
	__align__(16) float key[G5N_KDA_WY_C][G5N_KDA_KEY];
	__align__(16) float query[G5N_KDA_WY_C][G5N_KDA_KEY];
	union
	{
		__align__(16) float forget[G5N_KDA_WY_C][G5N_KDA_KEY];
		__align__(16) float value[G5N_KDA_WY_C][G5N_KDA_VALUE];
	};
	__align__(16) float forget_tmp[G5N_KDA_WY_C / 2u][G5N_KDA_KEY];
	float tri_a[G5N_KDA_WY_C][G5N_KDA_WY_C + 1u];
	float tri_m[G5N_KDA_WY_C][G5N_KDA_WY_C + 1u];
	float tri_x[G5N_KDA_WY_C][G5N_KDA_WY_C + 1u];
	__align__(16) float tail[G5N_KDA_KEY];
	float beta[G5N_KDA_WY_C];
};

struct Glm5NextKdaWyStateShared
{
	union
	{
		__align__(16) float w[G5N_KDA_WY_C][G5N_KDA_KEY];
		__align__(16) float carried[G5N_KDA_WY_C][G5N_KDA_KEY];
	};
	__align__(16) float qeff[G5N_KDA_WY_C][G5N_KDA_KEY];
	__align__(16) float u_in[G5N_KDA_WY_C][16u];
	__align__(16) float o_in[G5N_KDA_WY_C][16u];
	__align__(16) float tail[G5N_KDA_KEY];
	float reduce[2u][7u][16u * 32u];
};

template<uint32_t SIZE, uint32_t LANES>
static __device__ __forceinline__ void Glm5NextKdaWyMerge(Glm5NextKdaWyPrepShared &sh, uint32_t rows)
{
	constexpr uint32_t C = G5N_KDA_WY_C, TWICE = 2u * SIZE, PAIRS = C * SIZE / 2u, VECS = (G5N_KDA_KEY / 4u) / LANES, KV4 = G5N_KDA_KEY / 4u;
	const uint32_t pair = threadIdx.x / LANES, lane = threadIdx.x % LANES;
	if ( pair < PAIRS )
	{
		const uint32_t block = pair / (SIZE * SIZE), at = pair % (SIZE * SIZE);
		const uint32_t j = block * TWICE + at / SIZE, t = block * TWICE + SIZE + at % SIZE;
		float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
		if ( t < rows )
		{
			#pragma unroll
			for ( uint32_t index = 0u; index < VECS; ++index )
			{
				const uint32_t c = (index * LANES + lane) * 4u;
				const float4 w4 = *(const float4 *)&sh.carried[j][c];
				const float4 k4 = *(const float4 *)&sh.key[t][c];
				const float4 q4 = *(const float4 *)&sh.query[t][c];
				a0 = fmaf(k4.x, w4.x, a0);
				a1 = fmaf(k4.y, w4.y, a1);
				a2 = fmaf(k4.z, w4.z, a2);
				a3 = fmaf(k4.w, w4.w, a3);
				m0 = fmaf(q4.x, w4.x, m0);
				m1 = fmaf(q4.y, w4.y, m1);
				m2 = fmaf(q4.z, w4.z, m2);
				m3 = fmaf(q4.w, w4.w, m3);
			}
		}
		float sum_a = (a0 + a1) + (a2 + a3);
		float sum_m = (m0 + m1) + (m2 + m3);
		#pragma unroll
		for ( uint32_t s = 1u; s < LANES; s <<= 1u )
		{
			sum_a += __shfl_xor_sync(0xffffffffu, sum_a, s);
			sum_m += __shfl_xor_sync(0xffffffffu, sum_m, s);
		}
		if ( lane == 0u && t < rows )
		{
			sh.tri_a[t][j] = sh.beta[t] * sum_a;
			sh.tri_m[t][j] = sum_m;
		}
	}
	__syncthreads();
	constexpr bool READ_TMP = (SIZE == 2u || SIZE == 8u);
	for ( uint32_t index = threadIdx.x; index < (C / 2u) * KV4; index += 256u )
	{
		const uint32_t half = index / KV4, dim = (index % KV4) * 4u, first = (half / SIZE) * TWICE, left = first + half % SIZE, right = left + SIZE;
		const uint32_t idx_l = READ_TMP ? (first / SIZE) : first;
		const uint32_t idx_r = READ_TMP ? ((first + SIZE) / SIZE) : (first + SIZE);
		const float4 fr = *(const float4 *)(READ_TMP ? &sh.forget_tmp[idx_r][dim] : &sh.forget[idx_r][dim]);
		const float4 fl = *(const float4 *)(READ_TMP ? &sh.forget_tmp[idx_l][dim] : &sh.forget[idx_l][dim]);
		float4 c4 = *(const float4 *)&sh.carried[left][dim];
		float4 k4 = *(const float4 *)&sh.key[right][dim];
		float4 q4 = *(const float4 *)&sh.query[right][dim];
		c4.x *= fr.x; c4.y *= fr.y; c4.z *= fr.z; c4.w *= fr.w;
		k4.x *= fl.x; k4.y *= fl.y; k4.z *= fl.z; k4.w *= fl.w;
		q4.x *= fl.x; q4.y *= fl.y; q4.z *= fl.z; q4.w *= fl.w;
		*(float4 *)&sh.carried[left][dim] = c4;
		*(float4 *)&sh.key[right][dim] = k4;
		*(float4 *)&sh.query[right][dim] = q4;
		if ( (half % SIZE) == 0u )
		{
			const float4 prod = { fl.x * fr.x, fl.y * fr.y, fl.z * fr.z, fl.w * fr.w };
			if constexpr ( TWICE == C )
				*(float4 *)&sh.tail[dim] = prod;
			else if constexpr ( READ_TMP )
				*(float4 *)&sh.forget[first][dim] = prod;
			else
				*(float4 *)&sh.forget_tmp[first / TWICE][dim] = prod;
		}
	}
	__syncthreads();
}

__global__ __launch_bounds__(256u, 1)
static void Glm5NextKdaWyPrepareKernel(const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16,
	const float *__restrict__ forget_gate, const float *__restrict__ write_gate, float *__restrict__ workspace, uint32_t heads, uint32_t num_chunks, uint32_t total_rows)
{
	constexpr uint32_t C = G5N_KDA_WY_C, D = G5N_KDA_KEY;
	extern __shared__ __align__(16) uint8_t wy_prep_smem[];
	Glm5NextKdaWyPrepShared &sh = *(Glm5NextKdaWyPrepShared *)wy_prep_smem;
	const uint32_t chunk = blockIdx.x, head = blockIdx.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, gid = lane >> 2u, tg = lane & 3u;
	LmDependentWait();
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[0] : 0u;
	const uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[1] : total_rows;
	const uint32_t count = end > begin ? end - begin : 0u;
	const uint32_t first = chunk * C;
	const uint32_t chunk_rows = first < count ? (count - first < C ? count - first : C) : 0u;
	const uint32_t c0 = lane * 4u;
	const float q_scale = rsqrtf((float)D);
	#pragma unroll
	for ( uint32_t rw = warp; rw < C; rw += 8u )
	{
		if ( rw < chunk_rows )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + rw] : begin + first + rw;
			const uint64_t base = ((uint64_t)row * heads + head) * D + c0;
			const uint2 rk = __ldg((const uint2 *)(key_bf16 + base));
			const uint2 rq = __ldg((const uint2 *)(query_bf16 + base));
			const float4 fg = __ldg((const float4 *)(forget_gate + base));
			const float ka = __uint_as_float(rk.x << 16u), kb = __uint_as_float(rk.x & 0xffff0000u);
			const float kc = __uint_as_float(rk.y << 16u), kd = __uint_as_float(rk.y & 0xffff0000u);
			const float qa = __uint_as_float(rq.x << 16u), qb = __uint_as_float(rq.x & 0xffff0000u);
			const float qc = __uint_as_float(rq.y << 16u), qd = __uint_as_float(rq.y & 0xffff0000u);
			const float kinv = rsqrtf(LmLinearWarpSum4(ka * ka, kb * kb, kc * kc, kd * kd) + 1e-6f);
			const float qinv = rsqrtf(LmLinearWarpSum4(qa * qa, qb * qb, qc * qc, qd * qd) + 1e-6f) * q_scale;
			const float nka = ka * kinv, nkb = kb * kinv, nkc = kc * kinv, nkd = kd * kinv;
			const float nqa = qa * qinv, nqb = qb * qinv, nqc = qc * qinv, nqd = qd * qinv;
			const float diag = LmLinearWarpSum4(nka * nqa, nkb * nqb, nkc * nqc, nkd * nqd);
			*(float4 *)&sh.carried[rw][c0] = make_float4(nka, nkb, nkc, nkd);
			*(float4 *)&sh.key[rw][c0] = make_float4(nka * fg.x, nkb * fg.y, nkc * fg.z, nkd * fg.w);
			*(float4 *)&sh.query[rw][c0] = make_float4(nqa * fg.x, nqb * fg.y, nqc * fg.z, nqd * fg.w);
			*(float4 *)&sh.forget[rw][c0] = fg;
			if ( lane == 0u )
			{
				sh.tri_m[rw][rw] = diag;
				sh.beta[rw] = __ldg(write_gate + (uint64_t)row * heads + head);
			}
		}
		else
		{
			const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			*(float4 *)&sh.carried[rw][c0] = z4;
			*(float4 *)&sh.key[rw][c0] = z4;
			*(float4 *)&sh.query[rw][c0] = z4;
			*(float4 *)&sh.forget[rw][c0] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			if ( lane == 0u )
			{
				sh.tri_m[rw][rw] = 0.0f;
				sh.beta[rw] = 0.0f;
			}
		}
	}
	__syncthreads();
	Glm5NextKdaWyMerge<1u, 16u>(sh, chunk_rows);
	Glm5NextKdaWyMerge<2u, 8u>(sh, chunk_rows);
	Glm5NextKdaWyMerge<4u, 4u>(sh, chunk_rows);
	Glm5NextKdaWyMerge<8u, 2u>(sh, chunk_rows);
	Glm5NextKdaWyMerge<16u, 1u>(sh, chunk_rows);
	const uint64_t plane = (uint64_t)num_chunks * heads * C * D;
	const uint64_t ch_off = ((uint64_t)chunk * heads + head) * C * D;
	float *w_out = workspace + ch_off;
	float *u_out = workspace + plane + ch_off;
	float *qeff_out = workspace + 2u * plane + ch_off;
	float *ointra_out = workspace + 3u * plane + ch_off;
	float *carried_out = workspace + 4u * plane + ch_off;
	float *tail_out = workspace + 5u * plane + ((uint64_t)chunk * heads + head) * D;
	if ( warp == 0u )
	{
		for ( uint32_t t = 0u; t < C; ++t )
		{
			float val = 0.0f;
			if ( t == lane )
				val = t < chunk_rows ? sh.beta[t] : 0.0f;
			else if ( t > lane && t < chunk_rows )
			{
				for ( uint32_t j = lane; j < t; ++j )
					val = fmaf(-sh.tri_a[t][j], sh.tri_x[j][lane], val);
			}
			sh.tri_x[t][lane] = val;
		}
	}
	else
	{
		const uint32_t w_tid = threadIdx.x - 32u;
		for ( uint32_t idx = w_tid; idx < C * (D / 4u); idx += 224u )
		{
			const uint32_t t = idx >> 5u, cc = (idx & 31u) << 2u;
			*(float4 *)(carried_out + t * D + cc) = *(const float4 *)&sh.carried[t][cc];
			if ( t < chunk_rows )
			{
				const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + t] : begin + first + t;
				const uint2 rv = __ldg((const uint2 *)(value_bf16 + ((uint64_t)row * heads + head) * D + cc));
				*(float4 *)&sh.value[t][cc] = make_float4(__uint_as_float(rv.x << 16u), __uint_as_float(rv.x & 0xffff0000u),
					__uint_as_float(rv.y << 16u), __uint_as_float(rv.y & 0xffff0000u));
			}
			else
				*(float4 *)&sh.value[t][cc] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		}
		if ( w_tid < D / 4u )
			*(float4 *)(tail_out + w_tid * 4u) = *(const float4 *)&sh.tail[w_tid * 4u];
	}
	__syncthreads();
	const uint32_t d0 = warp * 16u;
	float acc_w[2][2][4], acc_u[2][2][4];
	#pragma unroll
	for ( uint32_t mb = 0u; mb < 2u; ++mb )
		#pragma unroll
		for ( uint32_t nb = 0u; nb < 2u; ++nb )
			acc_w[mb][nb][0] = acc_w[mb][nb][1] = acc_w[mb][nb][2] = acc_w[mb][nb][3] =
			acc_u[mb][nb][0] = acc_u[mb][nb][1] = acc_u[mb][nb][2] = acc_u[mb][nb][3] = 0.0f;
	#pragma unroll
	for ( uint32_t kb = 0u; kb < C / 8u; ++kb )
	{
		const uint32_t k0 = kb * 8u + tg, k1 = k0 + 4u;
		uint32_t awh[2][4], awl[2][4];
		#pragma unroll
		for ( uint32_t mb = 0u; mb < 2u; ++mb )
		{
			const uint32_t r0 = mb * 16u + gid, r1 = r0 + 8u;
			LmDeltaSplitTf32(sh.tri_x[r0][k0], awh[mb][0], awl[mb][0]);
			LmDeltaSplitTf32(sh.tri_x[r1][k0], awh[mb][1], awl[mb][1]);
			LmDeltaSplitTf32(sh.tri_x[r0][k1], awh[mb][2], awl[mb][2]);
			LmDeltaSplitTf32(sh.tri_x[r1][k1], awh[mb][3], awl[mb][3]);
		}
		#pragma unroll
		for ( uint32_t nb = 0u; nb < 2u; ++nb )
		{
			const uint32_t dc = d0 + nb * 8u + gid;
			uint32_t bkh[2], bkl[2], bvh[2], bvl[2];
			LmDeltaSplitTf32(sh.key[k0][dc], bkh[0], bkl[0]);
			LmDeltaSplitTf32(sh.key[k1][dc], bkh[1], bkl[1]);
			LmDeltaSplitTf32(sh.value[k0][dc], bvh[0], bvl[0]);
			LmDeltaSplitTf32(sh.value[k1][dc], bvh[1], bvl[1]);
			#pragma unroll
			for ( uint32_t mb = 0u; mb < 2u; ++mb )
			{
				LmDeltaMmaFp32(acc_w[mb][nb], awh[mb], awl[mb], bkh, bkl);
				LmDeltaMmaFp32(acc_u[mb][nb], awh[mb], awl[mb], bvh, bvl);
			}
		}
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t mb = 0u; mb < 2u; ++mb )
		#pragma unroll
		for ( uint32_t nb = 0u; nb < 2u; ++nb )
		{
			const uint32_t r0 = mb * 16u + gid, r1 = r0 + 8u, dc = d0 + nb * 8u + 2u * tg;
			sh.key[r0][dc] = acc_w[mb][nb][0];
			sh.key[r0][dc + 1u] = acc_w[mb][nb][1];
			sh.key[r1][dc] = acc_w[mb][nb][2];
			sh.key[r1][dc + 1u] = acc_w[mb][nb][3];
			sh.value[r0][dc] = acc_u[mb][nb][0];
			sh.value[r0][dc + 1u] = acc_u[mb][nb][1];
			sh.value[r1][dc] = acc_u[mb][nb][2];
			sh.value[r1][dc + 1u] = acc_u[mb][nb][3];
			*(float2 *)(w_out + r0 * D + dc) = make_float2(acc_w[mb][nb][0], acc_w[mb][nb][1]);
			*(float2 *)(w_out + r1 * D + dc) = make_float2(acc_w[mb][nb][2], acc_w[mb][nb][3]);
			*(float2 *)(u_out + r0 * D + dc) = make_float2(acc_u[mb][nb][0], acc_u[mb][nb][1]);
			*(float2 *)(u_out + r1 * D + dc) = make_float2(acc_u[mb][nb][2], acc_u[mb][nb][3]);
		}
	__syncthreads();
	#pragma unroll
	for ( uint32_t mb = 0u; mb < 2u; ++mb )
		#pragma unroll
		for ( uint32_t nb = 0u; nb < 2u; ++nb )
			acc_w[mb][nb][0] = acc_w[mb][nb][1] = acc_w[mb][nb][2] = acc_w[mb][nb][3] =
			acc_u[mb][nb][0] = acc_u[mb][nb][1] = acc_u[mb][nb][2] = acc_u[mb][nb][3] = 0.0f;
	#pragma unroll
	for ( uint32_t kb = 0u; kb < C / 8u; ++kb )
	{
		const uint32_t k0 = kb * 8u + tg, k1 = k0 + 4u;
		uint32_t amh[2][4], aml[2][4];
		#pragma unroll
		for ( uint32_t mb = 0u; mb < 2u; ++mb )
		{
			const uint32_t r0 = mb * 16u + gid, r1 = r0 + 8u;
			LmDeltaSplitTf32(k0 <= r0 && r0 < chunk_rows ? sh.tri_m[r0][k0] : 0.0f, amh[mb][0], aml[mb][0]);
			LmDeltaSplitTf32(k0 <= r1 && r1 < chunk_rows ? sh.tri_m[r1][k0] : 0.0f, amh[mb][1], aml[mb][1]);
			LmDeltaSplitTf32(k1 <= r0 && r0 < chunk_rows ? sh.tri_m[r0][k1] : 0.0f, amh[mb][2], aml[mb][2]);
			LmDeltaSplitTf32(k1 <= r1 && r1 < chunk_rows ? sh.tri_m[r1][k1] : 0.0f, amh[mb][3], aml[mb][3]);
		}
		#pragma unroll
		for ( uint32_t nb = 0u; nb < 2u; ++nb )
		{
			const uint32_t dc = d0 + nb * 8u + gid;
			uint32_t bwh[2], bwl[2], buh[2], bul[2];
			LmDeltaSplitTf32(sh.key[k0][dc], bwh[0], bwl[0]);
			LmDeltaSplitTf32(sh.key[k1][dc], bwh[1], bwl[1]);
			LmDeltaSplitTf32(sh.value[k0][dc], buh[0], bul[0]);
			LmDeltaSplitTf32(sh.value[k1][dc], buh[1], bul[1]);
			#pragma unroll
			for ( uint32_t mb = 0u; mb < 2u; ++mb )
			{
				LmDeltaMmaFp32(acc_w[mb][nb], amh[mb], aml[mb], bwh, bwl);
				LmDeltaMmaFp32(acc_u[mb][nb], amh[mb], aml[mb], buh, bul);
			}
		}
	}
	#pragma unroll
	for ( uint32_t mb = 0u; mb < 2u; ++mb )
		#pragma unroll
		for ( uint32_t nb = 0u; nb < 2u; ++nb )
		{
			const uint32_t r0 = mb * 16u + gid, r1 = r0 + 8u, dc = d0 + nb * 8u + 2u * tg;
			*(float2 *)(qeff_out + r0 * D + dc) = make_float2(sh.query[r0][dc] - acc_w[mb][nb][0], sh.query[r0][dc + 1u] - acc_w[mb][nb][1]);
			*(float2 *)(qeff_out + r1 * D + dc) = make_float2(sh.query[r1][dc] - acc_w[mb][nb][2], sh.query[r1][dc + 1u] - acc_w[mb][nb][3]);
			*(float2 *)(ointra_out + r0 * D + dc) = make_float2(acc_u[mb][nb][0], acc_u[mb][nb][1]);
			*(float2 *)(ointra_out + r1 * D + dc) = make_float2(acc_u[mb][nb][2], acc_u[mb][nb][3]);
		}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 1)
static void Glm5NextKdaWyStateKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices, const float *__restrict__ workspace,
	uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t num_chunks, uint32_t total_rows, uint32_t commit)
{
	constexpr uint32_t C = G5N_KDA_WY_C, D = G5N_KDA_KEY, COL_SLICE = 16u, K_BLOCKS = 2u, C_BLOCKS = C / 8u;
	extern __shared__ __align__(16) uint8_t wy_state_smem[];
	Glm5NextKdaWyStateShared &sh = *(Glm5NextKdaWyStateShared *)wy_state_smem;
	const uint32_t head = blockIdx.x, col_base = blockIdx.y * COL_SLICE;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, gid = lane >> 2u, tg = lane & 3u;
	const uint32_t k_perm = (gid >> 1u) + ((gid & 1u) << 2u);
	const uint32_t k_base = warp * K_BLOCKS * 8u;
	const uint32_t v0_local = gid, v1_local = gid + 8u, v0 = col_base + v0_local, v1 = col_base + v1_local;
	LmDependentWait();
	const uint32_t begin = sequence_row_begin != 0 ? sequence_row_begin[0] : 0u;
	const uint32_t end = sequence_row_begin != 0 ? sequence_row_begin[1] : total_rows;
	const uint32_t count = end > begin ? end - begin : 0u;
	float *state = (float *)(state_pool + (uint64_t)state_index[0] * slot_bytes) + (uint64_t)head * D * D;
	float s[K_BLOCKS][4];
	#pragma unroll
	for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
	{
		const uint32_t k0 = k_base + kb * 8u + tg, k1 = k0 + 4u;
		s[kb][0] = __ldcs(state + k0 * D + v0);
		s[kb][1] = __ldcs(state + k1 * D + v0);
		s[kb][2] = __ldcs(state + k0 * D + v1);
		s[kb][3] = __ldcs(state + k1 * D + v1);
	}
	const uint64_t plane = (uint64_t)num_chunks * heads * C * D;
	for ( uint32_t ch = 0u; ch < num_chunks; ++ch )
	{
		const uint32_t first = ch * C;
		const uint32_t chunk_rows = first < count ? (count - first < C ? count - first : C) : 0u;
		const uint64_t ch_off = ((uint64_t)ch * heads + head) * C * D;
		const float *w_in = workspace + ch_off;
		const float *u_src = workspace + plane + ch_off;
		const float *q_in = workspace + 2u * plane + ch_off;
		const float *o_src = workspace + 3u * plane + ch_off;
		const float *c_in_ptr = workspace + 4u * plane + ch_off;
		const float *t_in = workspace + 5u * plane + ((uint64_t)ch * heads + head) * D;
		for ( uint32_t idx = threadIdx.x; idx < C * (D / 4u); idx += 256u )
		{
			const uint32_t t = idx >> 5u, cc = (idx & 31u) << 2u;
			LmLinearCopyAsync(&sh.w[t][cc], w_in + t * D + cc, 16u);
			LmLinearCopyAsync(&sh.qeff[t][cc], q_in + t * D + cc, 16u);
		}
		if ( threadIdx.x < C * (COL_SLICE / 4u) )
		{
			const uint32_t t = threadIdx.x >> 2u, cc = (threadIdx.x & 3u) << 2u;
			LmLinearCopyAsync(&sh.u_in[t][cc], u_src + t * D + col_base + cc, 16u);
			LmLinearCopyAsync(&sh.o_in[t][cc], o_src + t * D + col_base + cc, 16u);
		}
		if ( threadIdx.x < D / 4u )
			LmLinearCopyAsync(&sh.tail[threadIdx.x * 4u], t_in + threadIdx.x * 4u, 16u);
		LmLinearCopyCommit();
		LmLinearCopyWait0();
		__syncthreads();
		float u_sub[C_BLOCKS][4], o_sub[C_BLOCKS][4];
		#pragma unroll
		for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
		{
			u_sub[ct][0] = u_sub[ct][1] = u_sub[ct][2] = u_sub[ct][3] = 0.0f;
			o_sub[ct][0] = o_sub[ct][1] = o_sub[ct][2] = o_sub[ct][3] = 0.0f;
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
				const uint32_t w_row = ct * 8u + k_perm;
				const uint32_t q_row = ct * 8u + gid;
				uint32_t bwh[2], bwl[2], bqh[2], bql[2];
				LmDeltaSplitTf32(sh.w[w_row][k0], bwh[0], bwl[0]);
				LmDeltaSplitTf32(sh.w[w_row][k1], bwh[1], bwl[1]);
				LmDeltaSplitTf32(sh.qeff[q_row][k0], bqh[0], bql[0]);
				LmDeltaSplitTf32(sh.qeff[q_row][k1], bqh[1], bql[1]);
				LmDeltaMmaFp32(u_sub[ct], ah, al, bwh, bwl);
				LmDeltaMmaFp32(o_sub[ct], ah, al, bqh, bql);
			}
		}
		if ( warp > 0u )
		{
			float *rp = sh.reduce[0][warp - 1u];
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					rp[(ct * 4u + e) * 32u + lane] = u_sub[ct][e];
		}
		if ( warp != 1u )
		{
			const uint32_t o_slot = warp == 0u ? 0u : (warp - 1u);
			float *ro = sh.reduce[1][o_slot];
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					ro[(ct * 4u + e) * 32u + lane] = o_sub[ct][e];
		}
		__syncthreads();
		if ( ch + 1u < num_chunks || commit != 0u )
		{
			for ( uint32_t idx = threadIdx.x; idx < C * (D / 4u); idx += 256u )
			{
				const uint32_t t = idx >> 5u, cc = (idx & 31u) << 2u;
				LmLinearCopyAsync(&sh.carried[t][cc], c_in_ptr + t * D + cc, 16u);
			}
			LmLinearCopyCommit();
		}
		if ( warp == 0u )
		{
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 7u; ++ks )
			{
				const float *rp = sh.reduce[0][ks];
				#pragma unroll
				for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
						u_sub[ct][e] += rp[(ct * 4u + e) * 32u + lane];
			}
			float *ru = sh.reduce[0][0];
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t t0 = ct * 8u + tg, t1 = t0 + 4u;
				u_sub[ct][0] = t0 < chunk_rows ? sh.u_in[t0][v0_local] - u_sub[ct][0] : 0.0f;
				u_sub[ct][1] = t1 < chunk_rows ? sh.u_in[t1][v0_local] - u_sub[ct][1] : 0.0f;
				u_sub[ct][2] = t0 < chunk_rows ? sh.u_in[t0][v1_local] - u_sub[ct][2] : 0.0f;
				u_sub[ct][3] = t1 < chunk_rows ? sh.u_in[t1][v1_local] - u_sub[ct][3] : 0.0f;
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					ru[(ct * 4u + e) * 32u + lane] = u_sub[ct][e];
			}
		}
		else if ( warp == 1u )
		{
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 7u; ++ks )
			{
				const float *ro = sh.reduce[1][ks];
				#pragma unroll
				for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
					#pragma unroll
					for ( uint32_t e = 0u; e < 4u; ++e )
						o_sub[ct][e] += ro[(ct * 4u + e) * 32u + lane];
			}
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t c_ev = ct * 8u + 2u * tg, c_od = c_ev + 1u;
				if ( c_ev < chunk_rows )
				{
					const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + c_ev] : begin + first + c_ev;
					const uint64_t out_row = ((uint64_t)row * heads + head) * D;
					output_bf16[out_row + v0] = LmFloatToBf16(sh.o_in[c_ev][v0_local] + o_sub[ct][0]);
					output_bf16[out_row + v1] = LmFloatToBf16(sh.o_in[c_ev][v1_local] + o_sub[ct][2]);
				}
				if ( c_od < chunk_rows )
				{
					const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + c_od] : begin + first + c_od;
					const uint64_t out_row = ((uint64_t)row * heads + head) * D;
					output_bf16[out_row + v0] = LmFloatToBf16(sh.o_in[c_od][v0_local] + o_sub[ct][1]);
					output_bf16[out_row + v1] = LmFloatToBf16(sh.o_in[c_od][v1_local] + o_sub[ct][3]);
				}
			}
		}
		if ( ch + 1u >= num_chunks && commit == 0u )
			break;
		LmLinearCopyWait0();
		__syncthreads();
		if ( warp > 0u )
		{
			const float *ru = sh.reduce[0][0];
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e )
					u_sub[ct][e] = ru[(ct * 4u + e) * 32u + lane];
		}
		uint32_t auh[C_BLOCKS][4], aul[C_BLOCKS][4];
		#pragma unroll
		for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
		{
			LmDeltaSplitTf32(u_sub[ct][0], auh[ct][0], aul[ct][0]);
			LmDeltaSplitTf32(u_sub[ct][2], auh[ct][1], aul[ct][1]);
			LmDeltaSplitTf32(u_sub[ct][1], auh[ct][2], aul[ct][2]);
			LmDeltaSplitTf32(u_sub[ct][3], auh[ct][3], aul[ct][3]);
		}
		#pragma unroll
		for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
		{
			const float t0 = sh.tail[k_base + kb * 8u + tg], t1 = sh.tail[k_base + kb * 8u + tg + 4u];
			s[kb][0] *= t0;
			s[kb][1] *= t1;
			s[kb][2] *= t0;
			s[kb][3] *= t1;
			const uint32_t k_col = k_base + kb * 8u + k_perm;
			#pragma unroll
			for ( uint32_t ct = 0u; ct < C_BLOCKS; ++ct )
			{
				const uint32_t j0 = ct * 8u + tg, j1 = j0 + 4u;
				uint32_t bch[2], bcl[2];
				LmDeltaSplitTf32(sh.carried[j0][k_col], bch[0], bcl[0]);
				LmDeltaSplitTf32(sh.carried[j1][k_col], bch[1], bcl[1]);
				LmDeltaMmaFp32(s[kb], auh[ct], aul[ct], bch, bcl);
			}
		}
		if ( ch + 1u < num_chunks )
			__syncthreads();
	}
	if ( commit != 0u )
	{
		#pragma unroll
		for ( uint32_t kb = 0u; kb < K_BLOCKS; ++kb )
		{
			const uint32_t k0 = k_base + kb * 8u + tg, k1 = k0 + 4u;
			LmLinearStoreCs(state + k0 * D + v0, s[kb][0]);
			LmLinearStoreCs(state + k1 * D + v0, s[kb][1]);
			LmLinearStoreCs(state + k0 * D + v1, s[kb][2]);
			LmLinearStoreCs(state + k1 * D + v1, s[kb][3]);
		}
	}
	LmDependentRelease();
}

static int32_t Glm5NextKdaWyLaunch(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t commit, SparkGpuStream stream)
{
	static std::atomic<bool> configured(false);
	constexpr uint32_t prep_sh = (uint32_t)sizeof(Glm5NextKdaWyPrepShared);
	constexpr uint32_t state_sh = (uint32_t)sizeof(Glm5NextKdaWyStateShared);
	if ( !configured.load(std::memory_order_acquire) )
	{
		if ( SparkGpuFuncSetAttribute((const void *)Glm5NextKdaWyPrepareKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)prep_sh) != SPARK_GPU_SUCCESS ||
			SparkGpuFuncSetAttribute((const void *)Glm5NextKdaWyStateKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)state_sh) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		configured.store(true, std::memory_order_release);
	}
	const uint32_t num_chunks = (rows + G5N_KDA_WY_C - 1u) / G5N_KDA_WY_C;
	float *workspace = (float *)b->expert_out_bf16;
	LM_LAUNCH_DEPENDENT((Glm5NextKdaWyPrepareKernel), dim3(num_chunks, b->kda_heads), 256u, prep_sh, stream,
		b->sequence_row_begin, b->sequence_row_indices, b->q_bf16, b->kv_slot_bf16, b->gate_up_bf16,
		b->kda_retention, b->kda_write_gate, workspace, b->kda_heads, num_chunks, rows);
	LM_LAUNCH_DEPENDENT((Glm5NextKdaWyStateKernel), dim3(b->kda_heads, G5N_KDA_VALUE / 16u), 256u, state_sh, stream,
		b->kda_state_pool, b->kda_state_slot_bytes, b->kda_state_index, b->sequence_row_begin, b->sequence_row_indices,
		workspace, b->attention_out_bf16, b->kda_heads, num_chunks, rows, commit);
	return Glm5NextLaunched();
}
#endif

template<uint32_t TILE_M, uint32_t WARPS_M, uint32_t TILE_N, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextKdaProjMmaKernel(const uint16_t *__restrict__ w_qkvb, const uint16_t *__restrict__ w_dg,
	const uint16_t *__restrict__ activation, uint16_t *__restrict__ out_q, uint16_t *__restrict__ out_k,
	uint16_t *__restrict__ out_v, uint16_t *__restrict__ out_beta, uint16_t *__restrict__ out_dg, uint32_t rows)
{
	constexpr uint32_t K = G5N_HIDDEN, RANK_QK = 512u, DG_DIM = 2u * G5N_KDA_LOW_RANK, BETA_DIM = 4u, TILE_K = 64u, K_TILES = K / TILE_K;
	constexpr uint32_t Q_TILES = RANK_QK / TILE_N, DG_TILES = DG_DIM / TILE_N;
	constexpr uint32_t WARPS_N = 8u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row_base = blockIdx.y * TILE_M, tile_id = blockIdx.x;
	const uint16_t *w_mat;
	uint16_t *dst_mat;
	uint32_t w_row_base, out_col_base, out_dim;
	if ( tile_id < Q_TILES )
	{
		w_mat = w_qkvb; dst_mat = out_q; w_row_base = tile_id * TILE_N; out_col_base = tile_id * TILE_N; out_dim = RANK_QK;
	}
	else if ( tile_id < 2u * Q_TILES )
	{
		const uint32_t t = tile_id - Q_TILES;
		w_mat = w_qkvb; dst_mat = out_k; w_row_base = RANK_QK + t * TILE_N; out_col_base = t * TILE_N; out_dim = RANK_QK;
	}
	else if ( tile_id < 3u * Q_TILES )
	{
		const uint32_t t = tile_id - 2u * Q_TILES;
		w_mat = w_qkvb; dst_mat = out_v; w_row_base = 2u * RANK_QK + t * TILE_N; out_col_base = t * TILE_N; out_dim = RANK_QK;
	}
	else if ( tile_id < 3u * Q_TILES + DG_TILES )
	{
		const uint32_t t = tile_id - 3u * Q_TILES;
		w_mat = w_dg; dst_mat = out_dg; w_row_base = t * TILE_N; out_col_base = t * TILE_N; out_dim = DG_DIM;
	}
	else
	{
		w_mat = w_qkvb; dst_mat = out_beta; w_row_base = 3u * RANK_QK; out_col_base = 0u; out_dim = BETA_DIM;
	}
	const uint32_t valid_n = out_dim - out_col_base < TILE_N ? out_dim - out_col_base : TILE_N;
	LmDependentWait();
	const uint16_t *a_src[A_PER_T], *b_src[B_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = chunk < A_CHUNKS && row_base + r < rows ? 1u : 0u;
		a_src[q] = activation + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = r < valid_n ? 1u : 0u;
		b_src[q] = w_mat + (uint64_t)(valid != 0u ? w_row_base + r : 0u) * K + c * 8u;
		b_mask |= valid << q;
	}
	auto adv = [](uint32_t &off)
	{
		off += STAGE_BYTES;
		if ( off == STAGES * STAGE_BYTES )
			off = 0u;
	};
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
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, b_src[q], ((b_mask >> q) & 1u) << 4u);
			b_src[q] += TILE_K;
		}
	};
	float acc[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
	{
		issue_k(sp_off);
		LmStreamCommit();
		kp++;
		adv(sp_off);
	}
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		if ( kp < K_TILES )
			LmStreamWait<STAGES - 2u>();
		else
			LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage_s = shared_base + sc_off;
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			LmStreamCommit();
			kp++;
			adv(sp_off);
		}
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t af[MF][4];
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t ar = warp_m * WARP_M + ii * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(af[ii], stage_s + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
			}
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t br = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
				uint32_t b0[2] = { tmp[0], tmp[1] }, b1[2] = { tmp[2], tmp[3] };
				#pragma unroll
				for ( uint32_t ii = 0u; ii < MF; ++ii )
				{
					LmMmaBf16(acc[ii][jj], af[ii], b0);
					LmMmaBf16(acc[ii][jj + 1u], af[ii], b1);
				}
			}
		}
		adv(sc_off);
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t ii = 0u; ii < MF; ++ii )
		#pragma unroll
		for ( uint32_t jj = 0u; jj < NF; ++jj )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + ii * 16u + quad_row + (e >> 1u) * 8u;
				const uint32_t c = out_col_base + warp_n * WARP_N + jj * 8u + quad * 2u;
				if ( r < rows && c < out_dim )
					*(uint32_t *)(dst_mat + (uint64_t)r * out_dim + c) = LmPackBf16Pair(acc[ii][jj][e], acc[ii][jj][e + 1u]);
			}
	LmDependentRelease();
}

static int32_t Glm5NextKdaProjMmaLaunch(const Glm5NextLayerBuffers *b, uint16_t *out_q, uint16_t *out_k, uint16_t *out_v,
	uint16_t *out_beta, uint32_t rows, SparkGpuStream stream)
{
	if ( rows >= 128u )
	{
		static std::atomic<bool> cfg_wide(false);
		constexpr uint32_t sh = 3u * (64u * 128u + 128u * 128u);
		if ( !cfg_wide.load(std::memory_order_acquire) )
		{
			if ( SparkGpuFuncSetAttribute((const void *)Glm5NextKdaProjMmaKernel<64u, 4u, 128u, 3u>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			cfg_wide.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((Glm5NextKdaProjMmaKernel<64u, 4u, 128u, 3u>), dim3(15u, (rows + 63u) / 64u), 256u, sh, stream,
			(const uint16_t *)b->w->kda_qkv_beta, (const uint16_t *)b->w->kda_decay_gate_down, b->normed_bf16,
			out_q, out_k, out_v, out_beta, b->fused_decay_gate_bf16, rows);
	}
	else
	{
		static std::atomic<bool> cfg_med(false);
		constexpr uint32_t sh = 3u * (32u * 128u + 64u * 128u);
		if ( !cfg_med.load(std::memory_order_acquire) )
		{
			if ( SparkGpuFuncSetAttribute((const void *)Glm5NextKdaProjMmaKernel<32u, 2u, 64u, 3u>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			cfg_med.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((Glm5NextKdaProjMmaKernel<32u, 2u, 64u, 3u>), dim3(29u, (rows + 31u) / 32u), 256u, sh, stream,
			(const uint16_t *)b->w->kda_qkv_beta, (const uint16_t *)b->w->kda_decay_gate_down, b->normed_bf16,
			out_q, out_k, out_v, out_beta, b->fused_decay_gate_bf16, rows);
	}
	return Glm5NextLaunched();
}

static int32_t Glm5NextKdaRecur(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sequences, const uint32_t *kept, uint32_t commit,
	SparkGpuStream stream)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t heads = b->kda_heads, rank_qk = heads * G5N_KDA_KEY, rank_v = heads * G5N_KDA_VALUE;
#if defined(__CUDACC__)
	if ( b->prefill_unified != 0u && kept == 0 && commit != 0u && sequences == 1u && heads == 4u && b->expert_out_bf16 != 0 )
	{
		const uint32_t has_sb = b->sequence_row_begin != 0 ? 1u : 0u;
		const uint32_t has_si = b->sequence_row_indices != 0 ? 1u : 0u;
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_kda_causal_conv_prefill_128, dim3((rows + 7u) / 8u, 3u * (rank_qk / 128u)), 128u, 0u, stream,
			b->kda_q_window, b->kda_k_window, b->kda_v_window, b->kda_state_index, b->sequence_row_begin, b->sequence_row_indices,
			b->kda_q_input, b->kda_k_input, b->kda_v_input, (const uint16_t *)w->kda_q_conv, (const uint16_t *)w->kda_k_conv,
			(const uint16_t *)w->kda_v_conv, b->q_bf16, b->kv_slot_bf16, b->gate_up_bf16, rank_qk, rows, has_sb, has_si) != SPARK_GPU_SUCCESS ||
			Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_kda_conv_window_commit_128, 3u * (rank_qk / 128u), 128u, 0u, stream,
			b->kda_q_window, b->kda_k_window, b->kda_v_window, b->kda_state_index, b->sequence_row_begin, b->sequence_row_indices,
			b->kda_q_input, b->kda_k_input, b->kda_v_input, rank_qk, rows, has_sb, has_si) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return Glm5NextKdaWyLaunch(b, rows, commit, stream);
	}
#endif
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
		if ( b->prefill_unified != 0u && kept == 0 && commit != 0u )
		{
			if ( LmDeltaRuleChunkLaunch<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float>(b->kda_state_pool, b->kda_state_slot_bytes, b->kda_state_index,
					b->sequence_row_begin, (const uint32_t *)0, b->q_bf16, b->kv_slot_bf16, b->gate_up_bf16, b->kda_retention, b->kda_write_gate,
					b->attention_out_bf16, heads, 1u, sequences, commit, b->sequence_row_indices, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		else if ( LmDeltaRuleLaunch<G5N_THREADS, G5N_KDA_KEY, G5N_KDA_VALUE, float>(rows, b->kda_state_pool, b->kda_state_slot_bytes, b->kda_state_index,
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

template<uint32_t TILE_R>
__global__ __launch_bounds__(128u, 4) void Glm5NextKdaUpDecayBetaKernel(const uint16_t *__restrict__ fused_dg,
	const uint16_t *__restrict__ decay_up_w, const uint16_t *__restrict__ gate_up_w, const float *__restrict__ decay_bias,
	const float *__restrict__ head_log_scale, const uint16_t *__restrict__ beta_src, uint32_t beta_stride, uint32_t beta_offset,
	float *__restrict__ retention, uint16_t *__restrict__ gate_out, float *__restrict__ write_gate, uint32_t rank_qk, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t GROUPS = 8u;
	const uint32_t group = threadIdx.x >> 4u, lane = threadIdx.x & 15u;
	const uint32_t row_base = blockIdx.y * TILE_R;
	const uint32_t cnt = row_base < rows ? (rows - row_base < TILE_R ? rows - row_base : TILE_R) : 0u;
	const uint32_t ch = blockIdx.x * GROUPS + group;
	LmDependentWait();
	if ( cnt == 0u )
	{
		LmDependentRelease();
		return;
	}
	if ( blockIdx.x == 0u && threadIdx.x < heads )
	{
		for ( uint32_t r = 0u; r < cnt; ++r )
		{
			const uint32_t row = row_base + r;
			const float rv = SparkMathRoundBf16(SparkMathSigmoidFast(LmBf16ToFloat(beta_src[(uint64_t)row * beta_stride + beta_offset + threadIdx.x])));
			write_gate[(uint64_t)row * heads + threadIdx.x] = rv;
		}
	}
	if ( ch < rank_qk )
	{
		const bool reuse = rows > TILE_R;
		const uint4 wd = reuse ? __ldg((const uint4 *)(decay_up_w + (uint64_t)ch * G5N_KDA_LOW_RANK) + lane) :
			__ldcs((const uint4 *)(decay_up_w + (uint64_t)ch * G5N_KDA_LOW_RANK) + lane);
		const uint4 wg = reuse ? __ldg((const uint4 *)(gate_up_w + (uint64_t)ch * G5N_KDA_LOW_RANK) + lane) :
			__ldcs((const uint4 *)(gate_up_w + (uint64_t)ch * G5N_KDA_LOW_RANK) + lane);
		const float bias = lane == 0u ? __ldg(decay_bias + ch) : 0.0f;
		const float hls = lane == 0u ? __ldg(head_log_scale + ch / G5N_KDA_KEY) : 0.0f;
		for ( uint32_t r = 0u; r < cnt; ++r )
		{
			const uint32_t row = row_base + r;
			const uint16_t *dg_row = fused_dg + (uint64_t)row * 2u * G5N_KDA_LOW_RANK;
			const uint4 ad = __ldg((const uint4 *)dg_row + lane);
			const uint4 ag = __ldg((const uint4 *)(dg_row + G5N_KDA_LOW_RANK) + lane);
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
				retention[out_idx] = LmBoundedDecay(d_bf16, bias, hls, G5N_KDA_GATE_LOWER);
				gate_out[out_idx] = LmFloatToBf16(sg);
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t TILE_R>
__global__ __launch_bounds__(G5N_THREADS, 2) void Glm5NextKdaNormGateOutKernel(const uint16_t *__restrict__ mixed,
	const uint16_t *__restrict__ gate, const float *__restrict__ norm_weight, const uint16_t *__restrict__ out_w,
	uint16_t *__restrict__ output, uint32_t rows)
{
	constexpr uint32_t HEADS = 4u, HD = G5N_KDA_VALUE, K = HEADS * HD, CHUNKS = K / 8u, LANES = 32u, GROUPS = G5N_THREADS / LANES;
	static_assert(HD == 128u && K == 512u && CHUNKS == 64u, "fused KDA head norm + BF16 out fits shared memory");
	__shared__ __align__(16) uint16_t s_in[TILE_R][K];
	__shared__ float s_wsum[8u];
	const uint32_t row_base = blockIdx.y * TILE_R;
	const uint32_t cnt = row_base < rows ? (rows - row_base < TILE_R ? rows - row_base : TILE_R) : 0u;
	LmDependentWait();
	if ( cnt != 0u )
	{
		const uint32_t half_id = threadIdx.x >> 7u, within = threadIdx.x & 127u;
		const uint32_t warp_in_half = within >> 5u, lane = threadIdx.x & 31u;
		const float nw = __ldg(norm_weight + within);
		for ( uint32_t r = 0u; r < cnt; ++r )
		{
			const uint32_t row = row_base + r;
			#pragma unroll
			for ( uint32_t h0 = 0u; h0 < HEADS; h0 += 2u )
			{
				const uint32_t h = h0 + half_id;
				const uint64_t idx = ((uint64_t)row * HEADS + h) * HD + within;
				const float v = LmBf16ToFloat(__ldg(mixed + idx));
				float sq = v * v;
				#pragma unroll
				for ( uint32_t off = 16u; off > 0u; off >>= 1u )
					sq += __shfl_down_sync(0xffffffffu, sq, off);
				if ( lane == 0u )
					s_wsum[half_id * 4u + warp_in_half] = sq;
				__syncthreads();
				const float sum = s_wsum[half_id * 4u] + s_wsum[half_id * 4u + 1u] + s_wsum[half_id * 4u + 2u] + s_wsum[half_id * 4u + 3u];
				const float scale = SparkMathRmsScale(sum, (float)HD, G5N_EPS);
				const float g = LmBf16ToFloat(__ldg(gate + idx));
				s_in[r][h * HD + within] = LmFloatToBf16(v * scale * nw * SparkMathSigmoidFast(g));
				if ( h0 + 2u < HEADS || r + 1u < cnt )
					__syncthreads();
			}
		}
		__syncthreads();
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
		if ( col0 < G5N_HIDDEN )
		{
			const bool has1 = col0 + 1u < G5N_HIDDEN;
			const bool reuse = rows > TILE_R;
			const uint4 *w0 = (const uint4 *)out_w + (uint64_t)col0 * CHUNKS;
			const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
			float acc0[TILE_R], acc1[TILE_R];
			#pragma unroll
			for ( uint32_t r = 0u; r < TILE_R; ++r )
				acc0[r] = acc1[r] = 0.0f;
			#pragma unroll
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const uint4 vw0 = reuse ? __ldg(w0 + c) : __ldcs(w0 + c);
				const uint4 vw1 = reuse ? __ldg(w1 + c) : __ldcs(w1 + c);
				#pragma unroll
				for ( uint32_t r = 0u; r < TILE_R; ++r )
				{
					if ( TILE_R == 1u || r < cnt )
					{
						const uint4 a = ((const uint4 *)s_in[r])[c];
						acc0[r] = LmSkinnyBf16Chunk(vw0, a, acc0[r]);
						acc1[r] = LmSkinnyBf16Chunk(vw1, a, acc1[r]);
					}
				}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < TILE_R; ++r )
			{
				if ( TILE_R == 1u || r < cnt )
				{
					#pragma unroll
					for ( uint32_t off = 16u; off > 0u; off >>= 1u )
					{
						acc0[r] += __shfl_down_sync(0xffffffffu, acc0[r], off);
						acc1[r] += __shfl_down_sync(0xffffffffu, acc1[r], off);
					}
					if ( sub == 0u )
					{
						const uint64_t dst = (uint64_t)(row_base + r) * G5N_HIDDEN + col0;
						output[dst] = LmFloatToBf16(acc0[r]);
						if ( has1 )
							output[dst + 1u] = LmFloatToBf16(acc1[r]);
					}
				}
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 2)
static void Glm5NextKdaUpDecayBetaMmaKernel(const uint16_t *__restrict__ fused_dg, const uint16_t *__restrict__ decay_up_w,
	const uint16_t *__restrict__ gate_up_w, const float *__restrict__ decay_bias, const float *__restrict__ head_log_scale,
	const uint16_t *__restrict__ beta_src, uint32_t beta_stride, uint32_t beta_offset, float *__restrict__ retention,
	uint16_t *__restrict__ gate_out, float *__restrict__ write_gate, uint32_t rank_qk, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t K = G5N_KDA_LOW_RANK, TILE_M = 64u, TILE_N = 64u, ROW_CHUNKS = K / 8u, ROW_BYTES = K * 2u;
	constexpr uint32_t A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES;
	constexpr uint32_t WARPS_N = 2u, WARP_M = 16u, WARP_N = 32u, NF = 4u;
	__shared__ __align__(128) uint8_t smem[A_BYTES + B_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row_base = blockIdx.y * TILE_M, col_base = blockIdx.x * TILE_N;
	LmDependentWait();
	if ( blockIdx.x == 0u )
	{
		for ( uint32_t idx = threadIdx.x; idx < TILE_M * heads; idx += 256u )
		{
			const uint32_t r = row_base + idx / heads, h = idx % heads;
			if ( r < rows )
				write_gate[(uint64_t)r * heads + h] = SparkMathRoundBf16(SparkMathSigmoidFast(
					LmBf16ToFloat(__ldg(beta_src + (uint64_t)r * beta_stride + beta_offset + h))));
		}
	}
	auto run_proj = [&](uint32_t a_col_off, const uint16_t *w_mat, float (&acc)[NF][4])
	{
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
		#pragma unroll
		for ( uint32_t q = 0u; q < (TILE_M * ROW_CHUNKS) / 256u; ++q )
		{
			const uint32_t idx = threadIdx.x + q * 256u;
			const uint32_t r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
			const uint32_t valid = row_base + r < rows ? 16u : 0u;
			LmStreamCopy(shared_base + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u,
				valid != 0u ? fused_dg + (uint64_t)(row_base + r) * (2u * K) + a_col_off + c * 8u : fused_dg, valid);
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < (TILE_N * ROW_CHUNKS) / 256u; ++q )
		{
			const uint32_t idx = threadIdx.x + q * 256u;
			const uint32_t r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
			const uint32_t valid = col_base + r < rank_qk ? 16u : 0u;
			LmStreamCopy(shared_base + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u,
				valid != 0u ? w_mat + (uint64_t)(col_base + r) * K + c * 8u : w_mat, valid);
		}
		LmStreamCommit();
		LmStreamWait<0u>();
		__syncthreads();
		#pragma unroll
		for ( uint32_t ks = 0u; ks < K / 16u; ++ks )
		{
			uint32_t a_reg[4];
			const uint32_t ar = warp_m * WARP_M + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
			LmStreamLdMatrix4(a_reg, shared_base + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t br = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, shared_base + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
				uint32_t b0[2] = { tmp[0], tmp[1] }, b1[2] = { tmp[2], tmp[3] };
				LmMmaBf16(acc[jj], a_reg, b0);
				LmMmaBf16(acc[jj + 1u], a_reg, b1);
			}
		}
	};
	float acc[NF][4];
	run_proj(0u, decay_up_w, acc);
	#pragma unroll
	for ( uint32_t j = 0u; j < NF; ++j )
	{
		const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
		if ( col < rank_qk )
		{
			const float b0 = __ldg(decay_bias + col), b1 = __ldg(decay_bias + col + 1u);
			const float h0 = __ldg(head_log_scale + col / G5N_KDA_KEY), h1 = __ldg(head_log_scale + (col + 1u) / G5N_KDA_KEY);
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + quad_row + (e >> 1u) * 8u;
				if ( r < rows )
				{
					const float d0 = LmBoundedDecay(LmBf16ToFloat(LmFloatToBf16(acc[j][e])), b0, h0, G5N_KDA_GATE_LOWER);
					const float d1 = LmBoundedDecay(LmBf16ToFloat(LmFloatToBf16(acc[j][e + 1u])), b1, h1, G5N_KDA_GATE_LOWER);
					*(float2 *)(retention + (uint64_t)r * rank_qk + col) = make_float2(d0, d1);
				}
			}
		}
	}
	__syncthreads();
	run_proj(K, gate_up_w, acc);
	#pragma unroll
	for ( uint32_t j = 0u; j < NF; ++j )
	{
		const uint32_t col = col_base + warp_n * WARP_N + j * 8u + quad * 2u;
		if ( col < rank_qk )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + quad_row + (e >> 1u) * 8u;
				if ( r < rows )
					*(uint32_t *)(gate_out + (uint64_t)r * rank_qk + col) = LmPackBf16Pair(acc[j][e], acc[j][e + 1u]);
			}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 4)
static void Glm5NextKdaNormGatePrefillKernel(const uint16_t *__restrict__ mixed, const uint16_t *__restrict__ gate,
	const float *__restrict__ norm_weight, uint16_t *__restrict__ output, uint32_t total_heads)
{
	const uint32_t half_warp = threadIdx.x >> 4u, sub = threadIdx.x & 15u;
	const uint32_t head_idx = blockIdx.x * 16u + half_warp;
	LmDependentWait();
	if ( head_idx < total_heads )
	{
		const uint64_t row_base = (uint64_t)head_idx * G5N_KDA_VALUE;
		const uint32_t offset = sub * 8u;
		const uint4 vec = __ldg((const uint4 *)(mixed + row_base + offset));
		float sq = LmSumSqBf16x8(vec, 0.0f);
		#pragma unroll
		for ( uint32_t s = 8u; s > 0u; s >>= 1u )
			sq += __shfl_down_sync(0xffffffffu, sq, s, 16u);
		sq = __shfl_sync(0xffffffffu, sq, 0u, 16u);
		const float sc = SparkMathRmsScale(sq, (float)G5N_KDA_VALUE, G5N_EPS);
		LmRmsSmallStore8(vec, gate + row_base, norm_weight, offset, sc, output + row_base, LmRmsSigmoidGateOp{});
	}
	LmDependentRelease();
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
	const uint32_t skinny_kda = b->prefill_unified == 0u && (rows <= 8u || b->row_invariant != 0u) ? 1u : 0u;
	const uint32_t fused_kda_proj = skinny_kda == 0u && heads == 4u && (b->prefill_unified != 0u || (b->row_invariant == 0u && rows >= 32u)) &&
		b->fused_qkvb_bf16 != 0 && LmIsAligned16(b->normed_bf16) && LmIsAligned16(w->kda_qkv_beta) && LmIsAligned16(w->kda_decay_gate_down) ? 1u : 0u;
	uint16_t *proj_q = b->fused_qkvb_bf16;
	uint16_t *proj_k = b->fused_qkvb_bf16 + (uint64_t)rows * rank_qk;
	uint16_t *proj_v = b->fused_qkvb_bf16 + (uint64_t)rows * 2u * rank_qk;
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
	else if ( fused_kda_proj != 0u )
		status = Glm5NextKdaProjMmaLaunch(b, proj_q, proj_k, proj_v, b->kda_beta_logit, rows, stream);
	else
	{
		status = Glm5NextLinear(b, b->normed_bf16, w->kda_qkv_beta, b->fused_qkvb_bf16, rows, G5N_HIDDEN, fused, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->normed_bf16, w->kda_decay_gate_down, b->fused_decay_gate_bf16, rows, G5N_HIDDEN, 2u * G5N_KDA_LOW_RANK, sms, stream);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	Glm5NextLayerBuffers kda_b = *b;
	if ( fused_kda_proj != 0u )
	{
		kda_b.kda_q_input = proj_q;
		kda_b.kda_k_input = proj_k;
		kda_b.kda_v_input = proj_v;
		kda_b.kda_beta_logit = b->kda_beta_logit;
	}
	else if ( rows == 1u && commit != 0u )
	{
		kda_b.kda_q_input = b->fused_qkvb_bf16;
		kda_b.kda_k_input = b->fused_qkvb_bf16 + rank_qk;
		kda_b.kda_v_input = b->fused_qkvb_bf16 + 2u * rank_qk;
		kda_b.kda_beta_logit = b->fused_qkvb_bf16 + 2u * rank_qk + rank_v;
	}
	else
		LM_LAUNCH((LmSplitQkvbKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->fused_qkvb_bf16, b->kda_q_input, b->kda_k_input,
			b->kda_v_input, b->kda_beta_logit, rows, rank_qk, rank_v, heads, fused);
	const uint16_t *beta_src = fused_kda_proj != 0u ? b->kda_beta_logit : b->fused_qkvb_bf16;
	const uint32_t beta_stride = fused_kda_proj != 0u ? heads : fused;
	const uint32_t beta_offset = fused_kda_proj != 0u ? 0u : (2u * rank_qk + rank_v);
	if ( (b->prefill_unified != 0u || (b->row_invariant == 0u && rows >= 32u)) &&
		LmIsAligned16(b->fused_decay_gate_bf16) && LmIsAligned16(w->kda_decay_up) && LmIsAligned16(w->kda_gate_up) )
	{
		LM_LAUNCH_DEPENDENT((Glm5NextKdaUpDecayBetaMmaKernel), dim3((rank_qk + 63u) / 64u, (rows + 63u) / 64u), 256u, 0, stream,
			b->fused_decay_gate_bf16, (const uint16_t *)w->kda_decay_up, (const uint16_t *)w->kda_gate_up, w->kda_decay_bias,
			w->kda_head_log_scale, beta_src, beta_stride, beta_offset, b->kda_retention, b->kda_gate_bf16,
			b->kda_write_gate, rank_qk, heads, rows);
	}
	else if ( b->prefill_unified == 0u && (rows <= 64u || b->row_invariant != 0u) && LmIsAligned16(b->fused_decay_gate_bf16) && LmIsAligned16(w->kda_decay_up) && LmIsAligned16(w->kda_gate_up) )
	{
		if ( rows == 1u )
			LM_LAUNCH_DEPENDENT((Glm5NextKdaUpDecayBetaKernel<1u>), dim3((rank_qk + 7u) / 8u, 1u), 128u, 0, stream,
				b->fused_decay_gate_bf16, (const uint16_t *)w->kda_decay_up, (const uint16_t *)w->kda_gate_up, w->kda_decay_bias, w->kda_head_log_scale,
				beta_src, beta_stride, beta_offset, b->kda_retention, b->kda_gate_bf16, b->kda_write_gate, rank_qk, heads, rows);
		else if ( rows <= 4u )
			LM_LAUNCH_DEPENDENT((Glm5NextKdaUpDecayBetaKernel<4u>), dim3((rank_qk + 7u) / 8u, 1u), 128u, 0, stream,
				b->fused_decay_gate_bf16, (const uint16_t *)w->kda_decay_up, (const uint16_t *)w->kda_gate_up, w->kda_decay_bias, w->kda_head_log_scale,
				beta_src, beta_stride, beta_offset, b->kda_retention, b->kda_gate_bf16, b->kda_write_gate, rank_qk, heads, rows);
		else
			LM_LAUNCH_DEPENDENT((Glm5NextKdaUpDecayBetaKernel<8u>), dim3((rank_qk + 7u) / 8u, (rows + 7u) / 8u), 128u, 0, stream,
				b->fused_decay_gate_bf16, (const uint16_t *)w->kda_decay_up, (const uint16_t *)w->kda_gate_up, w->kda_decay_bias, w->kda_head_log_scale,
				beta_src, beta_stride, beta_offset, b->kda_retention, b->kda_gate_bf16, b->kda_write_gate, rank_qk, heads, rows);
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
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_kda_decay_beta_256, rows, G5N_THREADS, 0u, stream, b->kda_decay_logit_bf16,
			w->kda_decay_bias, w->kda_head_log_scale, kda_b.kda_beta_logit, b->kda_retention, b->kda_write_gate, heads, rows) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	const bool fuse_out = b->prefill_unified == 0u && heads == 4u && (rows <= 8u || b->row_invariant != 0u) &&
		b->attention_latent_bf16 != 0 && LmIsAligned16(w->kda_out);
	if ( fuse_out )
		kda_b.attention_out_bf16 = b->attention_latent_bf16;
	status = Glm5NextKdaRecur(&kda_b, rows, sequences, (const uint32_t *)0, commit, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( fuse_out )
	{
		constexpr uint32_t COLS = (G5N_THREADS / 32u) * 2u;
		if ( rows == 1u )
			LM_LAUNCH_DEPENDENT((Glm5NextKdaNormGateOutKernel<1u>), dim3((G5N_HIDDEN + COLS - 1u) / COLS, 1u), G5N_THREADS, 0, stream,
				b->attention_latent_bf16, b->kda_gate_bf16, (const float *)w->kda_out_norm, (const uint16_t *)w->kda_out, b->attention_out_bf16, rows);
		else if ( rows <= 4u )
			LM_LAUNCH_DEPENDENT((Glm5NextKdaNormGateOutKernel<4u>), dim3((G5N_HIDDEN + COLS - 1u) / COLS, 1u), G5N_THREADS, 0, stream,
				b->attention_latent_bf16, b->kda_gate_bf16, (const float *)w->kda_out_norm, (const uint16_t *)w->kda_out, b->attention_out_bf16, rows);
		else
			LM_LAUNCH_DEPENDENT((Glm5NextKdaNormGateOutKernel<8u>), dim3((G5N_HIDDEN + COLS - 1u) / COLS, (rows + 7u) / 8u), G5N_THREADS, 0, stream,
				b->attention_latent_bf16, b->kda_gate_bf16, (const float *)w->kda_out_norm, (const uint16_t *)w->kda_out, b->attention_out_bf16, rows);
		return Glm5NextLaunched();
	}
	if ( LmIsAligned16(b->attention_out_bf16) && LmIsAligned16(b->kda_gate_bf16) && LmIsAligned16(w->kda_out_norm) && LmIsAligned16(b->kv_slot_bf16) )
		LM_LAUNCH_DEPENDENT((Glm5NextKdaNormGatePrefillKernel), (rows * heads + 15u) / 16u, 256u, 0, stream,
			b->attention_out_bf16, b->kda_gate_bf16, (const float *)w->kda_out_norm, b->kv_slot_bf16, rows * heads);
	else
		LM_LAUNCH((LmRmsNormSigmoidGateKernel<G5N_THREADS>), dim3((uint64_t)rows * heads), G5N_THREADS, 0, stream, b->attention_out_bf16,
			b->kda_gate_bf16, (const float *)w->kda_out_norm, b->kv_slot_bf16, G5N_KDA_VALUE, G5N_EPS);
	return Glm5NextLinear(b, b->kv_slot_bf16, w->kda_out, b->attention_out_bf16, rows, rank_v, G5N_HIDDEN, sms, stream);
}

__global__ __launch_bounds__(G5N_THREADS) void Glm5NextDsaPreStoreKernel(uint16_t *__restrict__ q_compressed,
	const uint16_t *__restrict__ q_a_norm, uint16_t *__restrict__ kv_slot, const uint16_t *__restrict__ kv_a_norm,
	uint16_t *__restrict__ index_key, const uint16_t *__restrict__ index_norm_w, const uint16_t *__restrict__ index_norm_b,
	const uint16_t *__restrict__ index_gate, uint16_t *__restrict__ index_packed, LmKvShardView cache, LmKvShardView index_cache,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ slot_of_row, uint32_t rows,
	uint16_t *__restrict__ kv_all = 0, uint32_t total_ctx = 0u)
{
	__shared__ __align__(16) float norm_buf[G5N_QA + 8u];
	__shared__ float pair_red[2u * LmBlockWarps<G5N_THREADS>()];
	const uint32_t row = blockIdx.x, part = blockIdx.y;
	LmDependentWait();
	if ( row < rows )
	{
		if ( part == 0u )
		{
			LmRmsNormRowBody<G5N_THREADS, uint16_t, true, false>(norm_buf, q_compressed, (const uint16_t *)0, q_a_norm,
				(uint16_t *)0, q_compressed, G5N_QA, G5N_QA, G5N_EPS, LmRmsGainOp<true>{});
		}
		else if ( part == 1u )
		{
			LmRmsNormRowBody<G5N_THREADS, uint16_t, true, false>(norm_buf, kv_slot, (const uint16_t *)0, kv_a_norm,
				(uint16_t *)0, kv_slot, G5N_LATENT, G5N_LATENT, G5N_EPS, LmRmsGainOp<true>{});
			__syncthreads();
			const uint32_t pos = slot_of_row[row];
			const uint16_t *src = kv_slot + (uint64_t)row * G5N_LATENT;
			uint16_t *dst = SparkKvShardOwns(cache.shard, pos) != 0u ?
				(uint16_t *)LmKvShardSlotRequired<Glm5NextKv>(cache, sequence_of_row[row], pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
			uint16_t *dst_all = kv_all != 0 && pos < total_ctx ? kv_all + (uint64_t)pos * G5N_LATENT : (uint16_t *)0;
			if ( dst != 0 || dst_all != 0 )
			{
				if ( threadIdx.x < G5N_LATENT / 8u )
				{
					const uint4 v = ((const uint4 *)src)[threadIdx.x];
					if ( dst != 0 )
						((uint4 *)dst)[threadIdx.x] = v;
					if ( dst_all != 0 )
						((uint4 *)dst_all)[threadIdx.x] = v;
				}
			}
		}
		else
		{
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
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(32u, 4)
static void Glm5NextIndexHeadSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows, uint32_t dense_limit)
{
	constexpr uint32_t K = G5N_HIDDEN, CHUNKS = K / 8u, RG = 8u;
	const uint32_t head = blockIdx.x, row_base = blockIdx.y * RG, lane = threadIdx.x;
	const uint32_t cnt = row_base < rows ? (rows - row_base < RG ? rows - row_base : RG) : 0u;
	LmDependentWait();
	uint32_t sparse = 0u;
	if ( lane < cnt )
	{
		const uint32_t r = row_base + lane;
		uint32_t bound = context_length[sequence_of_row[r]];
		if ( positions != 0 && positions[r] + 1u < bound )
			bound = positions[r] + 1u;
		sparse = bound > dense_limit ? 1u : 0u;
	}
	const uint32_t mask = __ballot_sync(0xffffffffu, sparse != 0u);
	if ( mask != 0u && head < G5N_INDEX_HEADS )
	{
		uint32_t s_rows[RG], act_cnt = 0u;
		uint32_t m = mask;
		while ( m != 0u )
		{
			const uint32_t bit = (uint32_t)(__ffs((int)m) - 1);
			s_rows[act_cnt++] = row_base + bit;
			m &= m - 1u;
		}
		const uint4 *w_row = (const uint4 *)(weight + (uint64_t)head * K);
		float acc[RG];
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			acc[r] = 0.0f;
		const bool reuse = rows > RG;
		for ( uint32_t c = lane; c < CHUNKS; c += 32u )
		{
			const uint4 vw = reuse ? __ldg(w_row + c) : __ldcs(w_row + c);
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				if ( r < act_cnt )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(activation + (uint64_t)s_rows[r] * K) + c), af);
					acc[r] = fmaf(1.0f, LmSkinnyBf16ChunkUnpacked(vw, af, 0.0f), acc[r]);
				}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			if ( r < act_cnt )
			{
				#pragma unroll
				for ( uint32_t step = 16u; step > 0u; step >>= 1u )
					acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], step);
				if ( lane == 0u )
					output[(uint64_t)s_rows[r] * G5N_INDEX_HEADS + head] = LmFloatToBf16(acc[r]);
			}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void Glm5NextIndexQExactSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows, uint32_t dense_limit)
{
	constexpr uint32_t K = G5N_QA, N = G5N_INDEX_QUERY, CHUNKS = K / 8u, RG = 8u;
	__shared__ uint32_t s_rows[RG], s_cnt;
	const uint32_t row_base = blockIdx.y * RG, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t cnt = row_base < rows ? (rows - row_base < RG ? rows - row_base : RG) : 0u;
	LmDependentWait();
	if ( warp == 0u )
	{
		uint32_t sparse = 0u;
		if ( lane < cnt )
		{
			const uint32_t r = row_base + lane;
			uint32_t bound = context_length[sequence_of_row[r]];
			if ( positions != 0 && positions[r] + 1u < bound )
				bound = positions[r] + 1u;
			sparse = bound > dense_limit ? 1u : 0u;
		}
		const uint32_t mask = __ballot_sync(0xffffffffu, sparse != 0u);
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
	const uint32_t col0 = (blockIdx.x * 4u + warp) * 2u;
	if ( act_cnt != 0u && col0 < N )
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
					*(uint32_t *)(output + (uint64_t)s_rows[r] * N + col0) = LmPackBf16Pair(a0[r], a1[r]);
			}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(256u, 1)
static void Glm5NextDsaIndexProjMmaKernel(const uint16_t *__restrict__ w_k, const uint16_t *__restrict__ w_gate,
	const uint16_t *__restrict__ w_head, const uint16_t *__restrict__ activation, uint16_t *__restrict__ out_k,
	uint16_t *__restrict__ out_gate, uint16_t *__restrict__ out_head, uint32_t rows, uint32_t emit_head)
{
	constexpr uint32_t K = G5N_HIDDEN, TILE_M = 64u, TILE_N = 64u, TILE_K = 64u, K_TILES = K / TILE_K, STAGES = 4u;
	constexpr uint32_t WARPS_N = 2u, WARP_M = 16u, WARP_N = 32u, NF = 4u;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_T = (TILE_M * ROW_CHUNKS) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row_base = blockIdx.y * TILE_M, tile_id = blockIdx.x;
	LmDependentWait();
	if ( tile_id == 4u && emit_head == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *w_mat = tile_id < 2u ? w_k : (tile_id < 4u ? w_gate : w_head);
	uint16_t *dst_mat = tile_id < 2u ? out_k : (tile_id < 4u ? out_gate : out_head);
	const uint32_t out_dim = tile_id < 4u ? G5N_INDEX_DIM : G5N_INDEX_HEADS;
	const uint32_t col_base = tile_id < 4u ? (tile_id & 1u) * TILE_N : 0u;
	const uint16_t *a_src[A_PER_T], *b_src[B_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = row_base + r < rows ? 1u : 0u;
		a_src[q] = activation + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = col_base + r < out_dim ? 1u : 0u;
		b_src[q] = w_mat + (uint64_t)(valid != 0u ? col_base + r : 0u) * K + c * 8u;
		b_mask |= valid << q;
	}
	auto adv = [](uint32_t &off)
	{
		off += STAGE_BYTES;
		if ( off == STAGES * STAGE_BYTES )
			off = 0u;
	};
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			LmStreamCopy(stage + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
			a_src[q] += TILE_K;
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_T; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, b_src[q], ((b_mask >> q) & 1u) << 4u);
			b_src[q] += TILE_K;
		}
	};
	float acc[NF][4];
	#pragma unroll
	for ( uint32_t j = 0u; j < NF; ++j )
		acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
	{
		issue_k(sp_off);
		LmStreamCommit();
		kp++;
		adv(sp_off);
	}
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		if ( kp < K_TILES )
			LmStreamWait<STAGES - 2u>();
		else
			LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage_s = shared_base + sc_off;
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			LmStreamCommit();
			kp++;
			adv(sp_off);
		}
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t a_reg[4];
			const uint32_t ar = warp_m * WARP_M + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
			LmStreamLdMatrix4(a_reg, stage_s + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t br = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
				uint32_t b0[2] = { tmp[0], tmp[1] }, b1[2] = { tmp[2], tmp[3] };
				LmMmaBf16(acc[jj], a_reg, b0);
				LmMmaBf16(acc[jj + 1u], a_reg, b1);
			}
		}
		adv(sc_off);
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t jj = 0u; jj < NF; ++jj )
		#pragma unroll
		for ( uint32_t e = 0u; e < 4u; e += 2u )
		{
			const uint32_t r = row_base + warp_m * WARP_M + quad_row + (e >> 1u) * 8u;
			const uint32_t c = col_base + warp_n * WARP_N + jj * 8u + quad * 2u;
			if ( r < rows && c < out_dim )
				*(uint32_t *)(dst_mat + (uint64_t)r * out_dim + c) = LmPackBf16Pair(acc[jj][e], acc[jj][e + 1u]);
		}
	LmDependentRelease();
}

static int32_t Glm5NextDsaIndexProjMmaLaunch(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t emit_head, SparkGpuStream stream)
{
	static std::atomic<bool> configured(false);
	constexpr uint32_t sh = 4u * (64u * 128u + 64u * 128u);
	if ( !configured.load(std::memory_order_acquire) )
	{
		if ( SparkGpuFuncSetAttribute((const void *)Glm5NextDsaIndexProjMmaKernel,
				SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		configured.store(true, std::memory_order_release);
	}
	LM_LAUNCH_DEPENDENT((Glm5NextDsaIndexProjMmaKernel), dim3(emit_head != 0u ? 5u : 4u, (rows + 63u) / 64u), 256u, sh, stream,
		(const uint16_t *)b->w->index_k, (const uint16_t *)b->w->index_compress_gate, (const uint16_t *)b->w->index_head,
		b->normed_bf16, b->index_key_bf16, b->index_gate_bf16, b->index_head_weight_bf16, rows, emit_head);
	return Glm5NextLaunched();
}

template<uint32_t TILE_M, uint32_t WARPS_M, uint32_t TILE_N, uint32_t STAGES>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextDsaHeadProjMmaKernel(const uint16_t *__restrict__ w_qa, const uint16_t *__restrict__ w_kva,
	const uint16_t *__restrict__ w_k, const uint16_t *__restrict__ w_gate, const uint16_t *__restrict__ w_head,
	const uint16_t *__restrict__ activation, uint16_t *__restrict__ out_qa, uint16_t *__restrict__ out_kva,
	uint16_t *__restrict__ out_k, uint16_t *__restrict__ out_gate, uint16_t *__restrict__ out_head, uint32_t rows, uint32_t emit_head)
{
	constexpr uint32_t K = G5N_HIDDEN, TILE_K = 64u, K_TILES = K / TILE_K;
	constexpr uint32_t QA_TILES = G5N_QA / TILE_N, KVA_TILES = G5N_LATENT / TILE_N, IK_TILES = G5N_INDEX_DIM / TILE_N;
	constexpr uint32_t WARPS_N = 8u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u;
	extern __shared__ __align__(128) uint8_t smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t row_base = blockIdx.y * TILE_M, tile_id = blockIdx.x;
	const uint16_t *w_mat;
	uint16_t *dst_mat;
	uint32_t col_base, out_dim;
	LmDependentWait();
	if ( tile_id < QA_TILES )
	{
		w_mat = w_qa; dst_mat = out_qa; col_base = tile_id * TILE_N; out_dim = G5N_QA;
	}
	else if ( tile_id < QA_TILES + KVA_TILES )
	{
		w_mat = w_kva; dst_mat = out_kva; col_base = (tile_id - QA_TILES) * TILE_N; out_dim = G5N_LATENT;
	}
	else if ( tile_id < QA_TILES + KVA_TILES + IK_TILES )
	{
		w_mat = w_k; dst_mat = out_k; col_base = (tile_id - QA_TILES - KVA_TILES) * TILE_N; out_dim = G5N_INDEX_DIM;
	}
	else if ( tile_id < QA_TILES + KVA_TILES + 2u * IK_TILES )
	{
		w_mat = w_gate; dst_mat = out_gate; col_base = (tile_id - QA_TILES - KVA_TILES - IK_TILES) * TILE_N; out_dim = G5N_INDEX_DIM;
	}
	else
	{
		if ( emit_head == 0u )
		{
			LmDependentRelease();
			return;
		}
		w_mat = w_head; dst_mat = out_head; col_base = 0u; out_dim = G5N_INDEX_HEADS;
	}
	const uint32_t valid_n = out_dim - col_base < TILE_N ? out_dim - col_base : TILE_N;
	const uint16_t *a_src[A_PER_T], *b_src[B_PER_T];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = chunk < A_CHUNKS && row_base + r < rows ? 1u : 0u;
		a_src[q] = activation + (uint64_t)(valid != 0u ? row_base + r : 0u) * K + c * 8u;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_T; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * 256u;
		const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
		const uint32_t valid = r < valid_n ? 1u : 0u;
		b_src[q] = w_mat + (uint64_t)(valid != 0u ? col_base + r : 0u) * K + c * 8u;
		b_mask |= valid << q;
	}
	auto adv = [](uint32_t &off)
	{
		off += STAGE_BYTES;
		if ( off == STAGES * STAGE_BYTES )
			off = 0u;
	};
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
			const uint32_t chunk = threadIdx.x + q * 256u;
			const uint32_t r = chunk / ROW_CHUNKS, c = chunk % ROW_CHUNKS;
			LmStreamCopy(stage + A_BYTES + r * ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, b_src[q], ((b_mask >> q) & 1u) << 4u);
			b_src[q] += TILE_K;
		}
	};
	float acc[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
	{
		issue_k(sp_off);
		LmStreamCommit();
		kp++;
		adv(sp_off);
	}
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		if ( kp < K_TILES )
			LmStreamWait<STAGES - 2u>();
		else
			LmStreamWait<0u>();
		__syncthreads();
		const uint32_t stage_s = shared_base + sc_off;
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			LmStreamCommit();
			kp++;
			adv(sp_off);
		}
		#pragma unroll
		for ( uint32_t ks = 0u; ks < 4u; ++ks )
		{
			uint32_t af[MF][4];
			#pragma unroll
			for ( uint32_t ii = 0u; ii < MF; ++ii )
			{
				const uint32_t ar = warp_m * WARP_M + ii * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(af[ii], stage_s + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
			}
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; jj += 2u )
			{
				const uint32_t br = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp, stage_s + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
				uint32_t b0[2] = { tmp[0], tmp[1] }, b1[2] = { tmp[2], tmp[3] };
				#pragma unroll
				for ( uint32_t ii = 0u; ii < MF; ++ii )
				{
					LmMmaBf16(acc[ii][jj], af[ii], b0);
					LmMmaBf16(acc[ii][jj + 1u], af[ii], b1);
				}
			}
		}
		adv(sc_off);
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t ii = 0u; ii < MF; ++ii )
		#pragma unroll
		for ( uint32_t jj = 0u; jj < NF; ++jj )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + ii * 16u + quad_row + (e >> 1u) * 8u;
				const uint32_t c = col_base + warp_n * WARP_N + jj * 8u + quad * 2u;
				if ( r < rows && c < out_dim )
					*(uint32_t *)(dst_mat + (uint64_t)r * out_dim + c) = LmPackBf16Pair(acc[ii][jj][e], acc[ii][jj][e + 1u]);
			}
	LmDependentRelease();
}

static int32_t Glm5NextDsaHeadProjMmaLaunch(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t emit_head, SparkGpuStream stream)
{
	if ( rows >= 128u )
	{
		static std::atomic<bool> cfg_wide(false);
		constexpr uint32_t sh = 3u * (64u * 128u + 128u * 128u);
		if ( !cfg_wide.load(std::memory_order_acquire) )
		{
			if ( SparkGpuFuncSetAttribute((const void *)Glm5NextDsaHeadProjMmaKernel<64u, 4u, 128u, 3u>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			cfg_wide.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((Glm5NextDsaHeadProjMmaKernel<64u, 4u, 128u, 3u>), dim3(emit_head != 0u ? 19u : 18u, (rows + 63u) / 64u), 256u, sh, stream,
			(const uint16_t *)b->w->q_a, (const uint16_t *)b->w->kv_a, (const uint16_t *)b->w->index_k,
			(const uint16_t *)b->w->index_compress_gate, (const uint16_t *)b->w->index_head, b->normed_bf16,
			b->q_compressed_bf16, b->kv_slot_bf16, b->index_key_bf16, b->index_gate_bf16, b->index_head_weight_bf16, rows, emit_head);
	}
	else
	{
		static std::atomic<bool> cfg_med(false);
		constexpr uint32_t sh = 3u * (32u * 128u + 64u * 128u);
		if ( !cfg_med.load(std::memory_order_acquire) )
		{
			if ( SparkGpuFuncSetAttribute((const void *)Glm5NextDsaHeadProjMmaKernel<32u, 2u, 64u, 3u>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			cfg_med.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((Glm5NextDsaHeadProjMmaKernel<32u, 2u, 64u, 3u>), dim3(emit_head != 0u ? 37u : 36u, (rows + 31u) / 32u), 256u, sh, stream,
			(const uint16_t *)b->w->q_a, (const uint16_t *)b->w->kv_a, (const uint16_t *)b->w->index_k,
			(const uint16_t *)b->w->index_compress_gate, (const uint16_t *)b->w->index_head, b->normed_bf16,
			b->q_compressed_bf16, b->kv_slot_bf16, b->index_key_bf16, b->index_gate_bf16, b->index_head_weight_bf16, rows, emit_head);
	}
	return Glm5NextLaunched();
}

template<uint32_t TILE_M, uint32_t WARPS_M, uint32_t TILE_N = 64u, uint32_t STAGES = 2u>
__global__ __launch_bounds__(256u, 1)
static void Glm5NextIndexQMmaSkipKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows, uint32_t dense_limit)
{
	constexpr uint32_t K = G5N_QA, N = G5N_INDEX_QUERY, TILE_K = 64u, K_TILES = K / TILE_K;
	constexpr uint32_t WARPS_N = 8u / WARPS_M, WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t ROW_CHUNKS = 8u, ROW_BYTES = 128u, A_BYTES = TILE_M * ROW_BYTES, B_BYTES = TILE_N * ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * ROW_CHUNKS, A_PER_T = (A_CHUNKS + 255u) / 256u, B_PER_T = (TILE_N * ROW_CHUNKS) / 256u, ROW_STEP = 256u / ROW_CHUNKS;
	extern __shared__ __align__(128) uint8_t smem[];
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
			if ( positions != 0 && positions[r] + 1u < bound )
				bound = positions[r] + 1u;
			sparse = bound > dense_limit ? 1u : 0u;
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
		auto adv = [](uint32_t &off)
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
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				kp++;
				adv(sp_off);
			}
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			if constexpr ( STAGES > 2u )
			{
				if ( kp < K_TILES )
					LmStreamWait<STAGES - 2u>();
				else
					LmStreamWait<0u>();
			}
			else
				LmStreamWait<0u>();
			__syncthreads();
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				LmStreamCommit();
				kp++;
				adv(sp_off);
			}
			const uint32_t stage_s = shared_base + sc_off;
			#pragma unroll
			for ( uint32_t ks = 0u; ks < 4u; ++ks )
			{
				uint32_t af[MF][4];
				#pragma unroll
				for ( uint32_t ii = 0u; ii < MF; ++ii )
				{
					const uint32_t ar = warp_m * WARP_M + ii * 16u + (lane & 15u), ac = ks * 2u + ((lane >> 4u) & 1u);
					LmStreamLdMatrix4(af[ii], stage_s + ar * ROW_BYTES + LmStreamSwizzle8(ar, ac) * 16u);
				}
				if constexpr ( (NF & 1u) == 0u )
				{
					#pragma unroll
					for ( uint32_t jj = 0u; jj < NF; jj += 2u )
					{
						const uint32_t br = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
						const uint32_t bc = ks * 2u + ((lane >> 3u) & 1u);
						uint32_t tmp[4];
						LmStreamLdMatrix4(tmp, stage_s + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
						uint32_t b0[2] = { tmp[0], tmp[1] }, b1[2] = { tmp[2], tmp[3] };
						#pragma unroll
						for ( uint32_t ii = 0u; ii < MF; ++ii )
						{
							LmMmaBf16(acc[ii][jj], af[ii], b0);
							LmMmaBf16(acc[ii][jj + 1u], af[ii], b1);
						}
					}
				}
				else
				{
					#pragma unroll
					for ( uint32_t jj = 0u; jj < NF; ++jj )
					{
						const uint32_t br = warp_n * WARP_N + jj * 8u + (lane & 7u), bc = ks * 2u + ((lane >> 3u) & 1u);
						uint32_t bf[2];
						LmStreamLdMatrix2(bf, stage_s + A_BYTES + br * ROW_BYTES + LmStreamSwizzle8(br, bc) * 16u);
						#pragma unroll
						for ( uint32_t ii = 0u; ii < MF; ++ii )
							LmMmaBf16(acc[ii][jj], af[ii], bf);
					}
				}
			}
			adv(sc_off);
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t ii = 0u; ii < MF; ++ii )
			#pragma unroll
			for ( uint32_t jj = 0u; jj < NF; ++jj )
			{
				const uint32_t r0 = row_base + warp_m * WARP_M + ii * 16u + quad_row, r1 = r0 + 8u;
				const uint32_t col = col_base + warp_n * WARP_N + jj * 8u + quad * 2u;
				if ( r0 < row_limit )
					*(uint32_t *)(output + (uint64_t)r0 * N + col) = LmPackBf16Pair(acc[ii][jj][0], acc[ii][jj][1]);
				if ( r1 < row_limit )
					*(uint32_t *)(output + (uint64_t)r1 * N + col) = LmPackBf16Pair(acc[ii][jj][2], acc[ii][jj][3]);
			}
	}
	LmDependentRelease();
}

static int32_t Glm5NextIndexScore(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, uint32_t projected, SparkGpuStream stream,
	uint32_t dense_limit = 0u, uint32_t prestored = 0u, float *local_scores_out = 0)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t degree = b->index_owner_degree, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	float *scores_dst = local_scores_out != 0 ? local_scores_out : b->index_local_scores;
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
			if ( dense_limit != 0u && LmIsAligned16(w->index_head) && LmIsAligned16(b->normed_bf16) )
				LM_LAUNCH_DEPENDENT((Glm5NextIndexHeadSkipKernel), dim3(G5N_INDEX_HEADS, (rows + 7u) / 8u), 32u, 0, stream,
					(const uint16_t *)w->index_head, b->normed_bf16, b->index_head_weight_bf16,
					b->sequence_of_row, b->context_length, b->positions, rows, dense_limit);
			else
			{
				status = Glm5NextLinear(b, b->normed_bf16, w->index_head, b->index_head_weight_bf16, rows, G5N_HIDDEN, G5N_INDEX_HEADS, sms, stream);
				if ( status != LM_LAUNCH_OK )
					return status;
			}
		}
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_index_pack_256, rows, G5N_THREADS, 0u, stream, b->index_key_bf16, b->index_gate_bf16,
			b->index_packed_bf16, (uint32_t)G5N_INDEX_DIM) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
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
	else if ( dense_limit != 0u && LmIsAligned16(w->index_q) && LmIsAligned16(b->q_compressed_bf16) )
	{
		if ( b->prefill_unified == 0u && (rows <= 16u || b->row_invariant != 0u) )
			LM_LAUNCH_DEPENDENT((Glm5NextIndexQExactSkipKernel), dim3(G5N_INDEX_QUERY / 8u, (rows + 7u) / 8u), 128u, 0, stream,
				(const uint16_t *)w->index_q, b->q_compressed_bf16, b->index_query_bf16,
				b->sequence_of_row, b->context_length, b->positions, rows, dense_limit);
		else if ( rows > 64u )
		{
			static std::atomic<bool> configured_iq(false);
			constexpr uint32_t sh = 3u * (64u * 128u + 128u * 128u);
			if ( !configured_iq.load(std::memory_order_acquire) )
			{
				if ( SparkGpuFuncSetAttribute((const void *)Glm5NextIndexQMmaSkipKernel<64u, 4u, 128u, 3u>,
						SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)sh) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
				configured_iq.store(true, std::memory_order_release);
			}
			LM_LAUNCH_DEPENDENT((Glm5NextIndexQMmaSkipKernel<64u, 4u, 128u, 3u>), dim3(G5N_INDEX_QUERY / 128u, (rows + 63u) / 64u), 256u, sh, stream,
				(const uint16_t *)w->index_q, b->q_compressed_bf16, b->index_query_bf16,
				b->sequence_of_row, b->context_length, b->positions, rows, dense_limit);
		}
		else
			LM_LAUNCH_DEPENDENT((Glm5NextIndexQMmaSkipKernel<32u, 2u, 64u, 2u>), dim3(G5N_INDEX_QUERY / 64u, (rows + 31u) / 32u), 256u,
				2u * (32u * 128u + 64u * 128u), stream,
				(const uint16_t *)w->index_q, b->q_compressed_bf16, b->index_query_bf16,
				b->sequence_of_row, b->context_length, b->positions, rows, dense_limit);
		status = Glm5NextLaunched();
	}
	else
		status = Glm5NextLinear(b, b->q_compressed_bf16, w->index_q, b->index_query_bf16, rows, G5N_QA, G5N_INDEX_QUERY, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( b->single_sequence != 0u && rows >= 64u && b->fork_positions == 0 && degree >= 8u && b->selection_scores != 0 )
	{
		float *pool_keys = b->selection_scores;
		uint32_t *pool_ready = (uint32_t *)(pool_keys + (uint64_t)local_stride * G5N_INDEX_DIM);
		LM_LAUNCH((Glm5NextPoolKeyBuildKernel<G5N_INDEX_DIM, G5N_KPOOL>), local_stride, G5N_INDEX_DIM, 0, stream,
			b->index_cache, b->sequence_of_row, b->context_length, (const float *)w->index_compress_ape, pools, b->index_owner_rank,
			degree, local_stride, pool_keys, pool_ready);
#if defined(__CUDACC__)
		if ( b->prefill_unified != 0u )
			return Glm5NextPoolScoreMmaLaunch(b->index_query_bf16, b->index_head_weight_bf16, pool_keys, pool_ready, b->sequence_of_row,
				b->context_length, b->positions, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
				SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, scores_dst, rows, stream);
#endif
		LM_LAUNCH((Glm5NextPoolScoreCachedKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS, G5N_POOL_BLOCK_POOLS, G5N_POOL_BLOCK_ROWS>),
			dim3((local_stride + G5N_POOL_BLOCK_POOLS - 1u) / G5N_POOL_BLOCK_POOLS, (rows + G5N_POOL_BLOCK_ROWS - 1u) / G5N_POOL_BLOCK_ROWS), G5N_THREADS, 0,
			stream, b->index_query_bf16, b->index_head_weight_bf16, pool_keys, pool_ready, b->sequence_of_row, b->context_length, b->positions,
			pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, scores_dst, rows);
	}
	else if ( rows >= 128u && b->fork_positions == 0 )
		LM_LAUNCH((Glm5NextPoolScoreRowsKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS, G5N_POOL_BLOCK_POOLS, G5N_POOL_BLOCK_ROWS>),
			dim3((local_stride + G5N_POOL_BLOCK_POOLS - 1u) / G5N_POOL_BLOCK_POOLS, (rows + G5N_POOL_BLOCK_ROWS - 1u) / G5N_POOL_BLOCK_ROWS), G5N_THREADS, 0,
			stream, b->index_query_bf16, b->index_head_weight_bf16, b->index_cache, b->sequence_of_row, b->context_length, b->positions,
			(const float *)w->index_compress_ape, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, scores_dst, rows);
	else if ( dense_limit != 0u && local_stride > 128u )
	{
		LM_LAUNCH_DEPENDENT((Glm5NextPoolScoreKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS, true>), dim3(128u, rows), G5N_THREADS, 0, stream,
			b->index_query_bf16, b->index_head_weight_bf16, b->index_cache, b->sequence_of_row, b->context_length, b->positions,
			(const float *)w->index_compress_ape, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, scores_dst, b->fork_positions, b->slot_offsets, dense_limit);
	}
	else
		LM_LAUNCH((Glm5NextPoolScoreKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS>), dim3(local_stride, rows), G5N_THREADS, 0, stream,
			b->index_query_bf16, b->index_head_weight_bf16, b->index_cache, b->sequence_of_row, b->context_length, b->positions,
			(const float *)w->index_compress_ape, pools, b->index_owner_rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, scores_dst, b->fork_positions, b->slot_offsets, dense_limit);
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

static int32_t Glm5NextIndexCandidatePack(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, SparkGpuStream stream,
	uint2 *candidates_out = 0)
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
		candidates_out != 0 ? candidates_out : (uint2 *)b->selection_scores, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionQueryProject(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	const int32_t status = Glm5NextLinear(b, b->q_compressed_bf16, b->w->q_b, b->q_bf16, rows, G5N_QA, b->q_b_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( LmPerHeadProjectChainLaunch<G5N_THREADS, G5N_NOPE, G5N_LATENT, G5N_NOPE, 0u>(b->q_bf16, (const uint16_t *)b->kv_b_key, b->query_latent_bf16,
		b->attn_heads, rows, stream, b->row_invariant, b->prefill_unified) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionHead(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, SparkGpuStream stream,
	uint32_t project_query = 0u, float *index_send_out = 0, uint16_t *kv_all = 0, uint32_t total_ctx = 0u)
{
	const Glm5NextLayerWeights *w = b->w;
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(SparkGlm5NextIndexCpPools(context), b->index_owner_degree);
	const uint32_t use_cand = Glm5NextIndexCandidates(rows, local_stride);
	uint32_t projected;
	int32_t status;
	if ( w->attn_norm == 0 || w->q_a == 0 || w->q_a_norm == 0 || w->q_b == 0 || w->kv_a == 0 || w->kv_a_norm == 0 || b->kv_b_key == 0 ||
		b->kv_b_value == 0 || w->attn_output == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->prenormed == 0u )
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_HIDDEN + 8u) * sizeof(float), stream, b->hc_collapsed_bf16,
			(const uint16_t *)w->attn_norm, b->normed_bf16, G5N_HIDDEN, G5N_HIDDEN, G5N_EPS);
	projected = b->prefill_unified == 0u && (rows <= 8u || b->row_invariant != 0u) ? 1u : 0u;
	const uint32_t fused_dsa_head = projected == 0u && (b->prefill_unified != 0u || (b->row_invariant == 0u && rows >= 32u)) &&
		LmIsAligned16(b->normed_bf16) && LmIsAligned16(w->q_a) && LmIsAligned16(w->kv_a) &&
		LmIsAligned16(w->index_k) && LmIsAligned16(w->index_compress_gate) && LmIsAligned16(w->index_head) ? 1u : 0u;
	const uint32_t fused_dsa_idx = fused_dsa_head;
	if ( projected != 0u )
	{
		const LmSkinnyDenseTarget targets[4] = {
			{w->q_a, b->q_compressed_bf16, 0, G5N_QA, G5N_QA, 0u, LmScaleTensorNone()},
			{w->kv_a, b->kv_slot_bf16, 0, G5N_LATENT, G5N_LATENT, 0u, LmScaleTensorNone()},
			{w->index_k, b->index_key_bf16, 0, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, LmScaleTensorNone()},
			{w->index_compress_gate, b->index_gate_bf16, 0, G5N_INDEX_DIM, G5N_INDEX_DIM, 0u, LmScaleTensorNone()}};
		status = Glm5NextSkinnyMulti(b, targets, 4u, b->normed_bf16, rows, G5N_HIDDEN, stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("glm5 mla: the fused q_a and index launch refused %u rows (%u + %u + %u + %u outputs over %u inputs); waves of up to %u "
				"rows and row-invariant waves run on it, so the four weights and the normed rows need 16-byte alignment\n", rows, G5N_QA, G5N_LATENT,
				G5N_INDEX_DIM, G5N_INDEX_DIM, G5N_HIDDEN, LM_SKINNY_ROWS_WIDE);
	}
	else if ( fused_dsa_head != 0u )
		status = Glm5NextDsaHeadProjMmaLaunch(b, rows, context > G5N_SELECTED ? 1u : 0u, stream);
	else
	{
		status = Glm5NextLinear(b, b->normed_bf16, w->q_a, b->q_compressed_bf16, rows, G5N_HIDDEN, G5N_QA, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = Glm5NextLinear(b, b->normed_bf16, w->kv_a, b->kv_slot_bf16, rows, G5N_HIDDEN, G5N_LATENT, sms, stream);
		if ( status == LM_LAUNCH_OK )
		{
			status = Glm5NextLinear(b, b->normed_bf16, w->index_k, b->index_key_bf16, rows, G5N_HIDDEN, G5N_INDEX_DIM, sms, stream);
			if ( status == LM_LAUNCH_OK )
				status = Glm5NextLinear(b, b->normed_bf16, w->index_compress_gate, b->index_gate_bf16, rows, G5N_HIDDEN, G5N_INDEX_DIM, sms, stream);
		}
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( context > G5N_SELECTED && fused_dsa_idx == 0u )
	{
		if ( LmIsAligned16(w->index_head) && LmIsAligned16(b->normed_bf16) )
		{
			LM_LAUNCH_DEPENDENT((Glm5NextIndexHeadSkipKernel), dim3(G5N_INDEX_HEADS, (rows + 7u) / 8u), 32u, 0, stream,
				(const uint16_t *)w->index_head, b->normed_bf16, b->index_head_weight_bf16,
				b->sequence_of_row, b->context_length, b->positions, rows, G5N_SELECTED);
			status = Glm5NextLaunched();
		}
		else
			status = Glm5NextLinear(b, b->normed_bf16, w->index_head, b->index_head_weight_bf16, rows, G5N_HIDDEN, G5N_INDEX_HEADS, sms, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
	}
	if ( LmIsAligned16(b->index_key_bf16) && LmIsAligned16(b->index_gate_bf16) && LmIsAligned16(w->index_norm_weight) && LmIsAligned16(w->index_norm_bias) )
	{
		LM_LAUNCH_DEPENDENT((Glm5NextDsaPreStoreKernel), dim3(rows, 3u), G5N_THREADS, 0, stream, b->q_compressed_bf16, (const uint16_t *)w->q_a_norm,
			b->kv_slot_bf16, (const uint16_t *)w->kv_a_norm, b->index_key_bf16, (const uint16_t *)w->index_norm_weight,
			(const uint16_t *)w->index_norm_bias, b->index_gate_bf16, b->index_packed_bf16, b->cache, b->index_cache,
			b->sequence_of_row, b->slots != 0 ? b->slots : b->positions, rows, kv_all, total_ctx);
		status = Glm5NextIndexScore(b, rows, context, sms, 1u, stream, G5N_SELECTED, 1u, use_cand == 0u ? index_send_out : (float *)0);
	}
	else
	{
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_QA + 8u) * sizeof(float), stream, b->q_compressed_bf16,
			(const uint16_t *)w->q_a_norm, b->q_compressed_bf16, G5N_QA, G5N_QA, G5N_EPS);
		LM_LAUNCH((LmBf16RmsNormKernel<G5N_THREADS>), rows, G5N_THREADS, (G5N_LATENT + 8u) * sizeof(float), stream, b->kv_slot_bf16,
			(const uint16_t *)w->kv_a_norm, b->kv_slot_bf16, G5N_LATENT, G5N_LATENT, G5N_EPS);
		LM_LAUNCH((LmKvShardStoreKernel<Glm5NextKv, G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->cache, b->kv_slot_bf16, b->sequence_of_row,
			b->slots != 0 ? b->slots : b->positions, rows, G5N_LATENT);
		if ( kv_all != 0 && total_ctx != 0u )
			if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_copy_new_kv_256, rows, G5N_THREADS, 0u, stream, b->kv_slot_bf16,
				b->slots != 0 ? b->slots : b->positions, kv_all, rows, total_ctx) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		status = Glm5NextIndexScore(b, rows, context, sms, 1u, stream, G5N_SELECTED, 0u, use_cand == 0u ? index_send_out : (float *)0);
	}
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextIndexCandidatePack(b, rows, context, stream, use_cand != 0u ? (uint2 *)index_send_out : (uint2 *)0);
	if ( status == LM_LAUNCH_OK && project_query != 0u )
		status = Glm5NextAttentionQueryProject(b, rows, sms, stream);
	return status;
}

static int32_t Glm5NextIndexSelect(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, SparkGpuStream stream,
	const float *gathered_override = 0, uint64_t peer_stride_floats = 0u, uint32_t fill_dense = 0u)
{
	const uint32_t degree = b->index_owner_degree, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	const float *gathered = gathered_override != 0 ? gathered_override : b->index_gathered_scores;
	if ( context <= G5N_SELECTED )
		return LM_LAUNCH_OK;
	if ( Glm5NextIndexCandidates(rows, local_stride) != 0u )
	{
		const uint64_t peer_stride = peer_stride_floats != 0u ? peer_stride_floats / 2u :
			(uint64_t)SparkGlm5NextIndexCpGatherSequences(rows, 2u * Glm5NextIndexKeep()) * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS / 2u;
		if ( LmIndexShardCandidateScatterLaunch<G5N_THREADS>((const uint2 *)gathered, peer_stride, degree, Glm5NextIndexKeep(), rows, pools,
			b->selection_scores, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
	{
		const uint64_t peer_stride = peer_stride_floats != 0u ? peer_stride_floats :
			(uint64_t)SparkGlm5NextIndexCpGatherSequences(rows, local_stride) * SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS;
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_pool_permute_256, rows, G5N_THREADS, 0u, stream, gathered, b->selection_scores, pools,
			local_stride, (uint64_t)(peer_stride * sizeof(float)), degree, b->sequence_of_row, b->context_length, b->positions,
			(uint32_t)G5N_SELECTED, (uint32_t)((b->sequence_of_row != 0 && b->context_length != 0) ? 1u : 0u),
			(uint32_t)(b->positions != 0 ? 1u : 0u)) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	LmTopkExactRowsLaunch<G5N_THREADS>(b->selection_scores, rows, pools, G5N_SELECTED / G5N_KPOOL, b->selected_pools, stream,
		b->sequence_of_row, b->context_length, b->positions, G5N_SELECTED, G5N_KPOOL);
	if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_pool_expand_256, rows, G5N_THREADS, 0u, stream, b->selected_pools,
		b->sequence_of_row, b->context_length, b->positions, b->selected_positions, rows, (uint32_t)G5N_SELECTED, fill_dense) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionTail(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, SparkGpuStream stream)
{
	int32_t status = Glm5NextIndexSelect(b, rows, context, stream);
	if ( status == LM_LAUNCH_OK )
		status = Glm5NextAttentionQueryProject(b, rows, sms, stream);
	return status;
}

static int32_t Glm5NextAttentionPartial(const Glm5NextLayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	if ( !(LmLatentShardAttendLaunch<Glm5NextKv, LmKvShardView, G5N_LATENT, 0u>(b->cache, b->shard_query_gathered_bf16, b->shard_query_stride, b->attn_heads, b->sequence_of_row, b->context_length, b->positions, b->selected_positions, G5N_INDEX_WIDTH, G5N_SELECTED, G5N_QK_SCALE, b->shard_partials_f32, b->shard_partial_stride, rows, stream, b->fork_positions, b->slot_offsets) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

template<bool SPARSE, uint32_t QUERIES = 32u>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1) void Glm5NextMlaGatherPrefillKernel(
	const uint16_t *__restrict__ query_bf16,
	const uint16_t *__restrict__ kv_all,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows,
	uint32_t total_ctx,
	const uint32_t *__restrict__ union_positions,
	const uint8_t *__restrict__ union_masks,
	const uint32_t *__restrict__ union_counts,
	uint32_t union_capacity)
{
	constexpr uint32_t HEADS = 4u;
	constexpr uint32_t LATENT = G5N_LATENT;
	constexpr uint32_t DIM = LATENT;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / HEADS;
	constexpr uint32_t QUERY_TILES = QUERIES / 16u;
	constexpr uint32_t KEY_FRAGS = QUERY_TILES;
	constexpr uint32_t KEY_GROUPS = TILE / 8u / KEY_FRAGS;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / LM_PREFILL_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	static_assert(HEADS == 4u && (QUERIES == 16u || QUERIES == 32u) && ROWS_PER_BLOCK <= 8u,
		"4-head grouped prefill fills one or two 16-query MMA tiles per CTA");
	static_assert(LmPrefillAttnSharedBytes<DIM, QUERIES>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "prefill tiles fit SM shared memory");
	extern __shared__ __align__(128) uint8_t g5n_prefill_smem[];
	uint8_t *keys = g5n_prefill_smem;
	float *score_parts = (float *)(keys + 2u * TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_PARTIAL_STRIDE);
	float *running_max = (float *)(weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	uint32_t *key_rows = (uint32_t *)(slots + 2u * TILE);
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t query_tile = warp / (2u * KEY_GROUPS), half = (warp / KEY_GROUPS) & 1u, key_group = warp % KEY_GROUPS;
	const uint32_t table = gridDim.x - 1u - blockIdx.x;
	const uint32_t *tile_positions = SPARSE ? union_positions + (uint64_t)table * union_capacity : (const uint32_t *)0;
	const uint8_t *tile_masks = SPARSE ? union_masks + (uint64_t)table * union_capacity : (const uint8_t *)0;
	const uint32_t first_row = table * ROWS_PER_BLOCK;
	const uint32_t row_end = LmPrefillMin(rows, first_row + ROWS_PER_BLOCK);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t query_base = key_base + TILE * ROW_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[QUERY_TILES][COLUMN_FRAGS][4];
	uint32_t row_limit[ROWS_PER_BLOCK];
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	const uint32_t sequence = __ldg(sequence_of_row + first_row);
	const uint32_t seq_ctx = __ldg(context_length + sequence);
	const uint32_t ctx_cap = seq_ctx < total_ctx ? seq_ctx : total_ctx;
	uint32_t first_position_min = 0xffffffffu, last_position = 0u;
	for ( uint32_t r = first_row; r < row_end; ++r )
	{
		const uint32_t rpos = __ldg(row_position + r);
		const uint32_t lim = rpos + 1u < ctx_cap ? rpos : (ctx_cap > 0u ? ctx_cap - 1u : 0u);
		first_position_min = lim < first_position_min ? lim : first_position_min;
		last_position = lim > last_position ? lim : last_position;
	}
	const uint32_t listed_keys = ctx_cap == 0u ? 0u : (SPARSE ? __ldg(union_counts + table) : last_position + 1u);
	const uint32_t tiles = (listed_keys + TILE - 1u) / TILE;
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS_PER_BLOCK; ++r )
	{
		const uint32_t rpos = first_row + r < row_end ? __ldg(row_position + first_row + r) : 0u;
		row_limit[r] = rpos + 1u < ctx_cap ? rpos : (ctx_cap > 0u ? ctx_cap - 1u : 0u);
	}
	auto listing = [&](uint32_t which, uint32_t &position, uint32_t &members)
	{
		const uint32_t listed = which * TILE + threadIdx.x;
		position = !SPARSE ? listed : (listed < listed_keys ? __ldg(tile_positions + listed) : 0xffffffffu);
		members = !SPARSE ? 0xffffffffu : (listed < listed_keys ? (uint32_t)__ldg(tile_masks + listed) : 0u);
	};
	auto resolve = [&](uint32_t position, uint32_t members, uint32_t &valid) -> const uint8_t *
	{
		valid = 0u;
		#pragma unroll
		for ( uint32_t r = 0u; r < ROWS_PER_BLOCK; ++r )
			valid |= (first_row + r < row_end && ((members >> r) & 1u) != 0u && position <= row_limit[r] ? 1u : 0u) << r;
		return position <= last_position && position < total_ctx ? (const uint8_t *)(kv_all + (uint64_t)position * LATENT) : (const uint8_t *)0;
	};
	auto copy = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		constexpr uint32_t SLOTS_PER_WARP = TILE / LM_PREFILL_ATTN_WARPS;
		constexpr uint32_t WARP_ITERS = (SLOTS_PER_WARP * CHUNKS) / 32u;
		const uint8_t *s0 = slots[buffer * TILE + warp * SLOTS_PER_WARP + 0u];
		const uint8_t *s1 = slots[buffer * TILE + warp * SLOTS_PER_WARP + 1u];
		const uint8_t *s2 = slots[buffer * TILE + warp * SLOTS_PER_WARP + 2u];
		const uint8_t *s3 = slots[buffer * TILE + warp * SLOTS_PER_WARP + 3u];
		#pragma unroll
		for ( uint32_t iter = 0u; iter < WARP_ITERS; ++iter )
		{
			const uint32_t k = iter * 32u + lane;
			const uint32_t s = (k >= CHUNKS) + (k >= 2u * CHUNKS) + (k >= 3u * CHUNKS);
			const uint32_t piece = k - s * CHUNKS;
			const uint32_t slot_index = warp * SLOTS_PER_WARP + s;
			const uint8_t *slot = s == 0u ? s0 : (s == 1u ? s1 : (s == 2u ? s2 : s3));
			LmStreamCopy(key_base + (buffer * TILE + slot_index) * ROW_BYTES + LmPrefillSwizzle(slot_index, piece) * 16u,
				slot != 0 ? (const void *)(slot + piece * 16u) : (const void *)query_bf16, slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	for ( uint32_t idx = threadIdx.x; idx < QUERIES * CHUNKS; idx += LM_PREFILL_ATTN_THREADS )
	{
		const uint32_t q = idx / CHUNKS, chunk = idx % CHUNKS;
		const uint32_t row = first_row + q / HEADS;
		const uint16_t *src = row >= row_end ? query_bf16 : query_bf16 + ((uint64_t)row * HEADS + q % HEADS) * DIM + chunk * 8u;
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q, chunk) * 16u, src, row >= row_end ? 0u : 16u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	uint32_t next_position = 0xffffffffu, next_members = 0u;
	if ( threadIdx.x < TILE )
	{
		uint32_t valid, p0, m0, p1 = 0xffffffffu, m1 = 0u;
		if ( tiles > 0u ) listing(0u, p0, m0);
		if ( tiles > 1u ) listing(1u, p1, m1);
		if ( tiles > 2u ) listing(2u, next_position, next_members);
		if ( tiles > 0u ) { slots[threadIdx.x] = resolve(p0, m0, valid); key_rows[threadIdx.x] = valid; }
		if ( tiles > 1u ) { slots[TILE + threadIdx.x] = resolve(p1, m1, valid); key_rows[TILE + threadIdx.x] = valid; }
	}
	__syncthreads();
	if ( tiles > 0u )
	{
		copy(0u);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
	}
	else
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = query_tile * 16u + (matrix & 1u) * 8u + within;
		#pragma unroll
		for ( uint32_t step = 0u; step < HALF_STEPS; ++step )
			LmPrefillLoadX4(query_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row, (half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * 16u,
				query_frag[step]);
	}
	if ( threadIdx.x < QUERIES )
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
	#pragma unroll
	for ( uint32_t m = 0u; m < QUERY_TILES; ++m )
		#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
			output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
	__syncthreads();
	for ( uint32_t tile = 0u; tile < tiles; ++tile )
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_base = key_base + buffer * TILE * ROW_BYTES;
		uint32_t ahead_position = 0xffffffffu, ahead_members = 0u;
		if ( tile + 1u < tiles )
		{
			copy(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		if ( threadIdx.x < TILE && tile + 3u < tiles )
			listing(tile + 3u, ahead_position, ahead_members);
		__syncthreads();
		{
			float score[KEY_FRAGS][4];
			uint32_t b[4];
			const uint32_t b_row = (key_group * KEY_FRAGS + (KEY_FRAGS == 2u ? (matrix >> 1u) : 0u)) * 8u + within;
			#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; ++frag )
				score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
			#pragma unroll
			for ( uint32_t step = 0u; step < HALF_STEPS; ++step )
			{
				const uint32_t b_chunk = (half * HALF_STEPS + step) * 2u + (matrix & 1u);
				if constexpr ( KEY_FRAGS == 2u )
				{
					uint32_t first[2], second[2];
					LmPrefillLoadX4(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(score[0], query_frag[step], first);
					LmMmaBf16(score[1], query_frag[step], second);
				}
				else
				{
					LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
					LmMmaBf16(score[0], query_frag[step], b);
				}
			}
			#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; ++frag )
			{
				const uint32_t base_col = (key_group * KEY_FRAGS + frag) * 8u + (lane & 3u) * 2u;
				const uint32_t row_lo = half * QUERIES + query_tile * 16u + (lane >> 2u);
				*(float2 *)(score_parts + row_lo * LM_PREFILL_ATTN_PARTIAL_STRIDE + base_col) = make_float2(score[frag][0], score[frag][1]);
				*(float2 *)(score_parts + (row_lo + 8u) * LM_PREFILL_ATTN_PARTIAL_STRIDE + base_col) = make_float2(score[frag][2], score[frag][3]);
			}
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
			const uint32_t key_base_idx = part * QUERY_KEYS, own_row = own / HEADS;
			const bool full_dense_tile = !SPARSE && (first_row + own_row < row_end) && ((tile + 1u) * TILE - 1u <= first_position_min);
			float value[QUERY_KEYS], weight[QUERY_KEYS];
			if constexpr ( QUERY_KEYS == 4u )
			{
				const float4 a = *(const float4 *)(score_parts + own * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
				const float4 b = *(const float4 *)(score_parts + (QUERIES + own) * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
				value[0] = a.x + b.x;
				value[1] = a.y + b.y;
				value[2] = a.z + b.z;
				value[3] = a.w + b.w;
			}
			else
			{
				const float2 a = *(const float2 *)(score_parts + own * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
				const float2 b = *(const float2 *)(score_parts + (QUERIES + own) * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
				value[0] = a.x + b.x;
				value[1] = a.y + b.y;
			}
			if ( !full_dense_tile )
			{
				#pragma unroll
				for ( uint32_t idx = 0u; idx < QUERY_KEYS; ++idx )
					value[idx] = ((key_rows[buffer * TILE + key_base_idx + idx] >> own_row) & 1u) != 0u ? value[idx] : -INFINITY;
			}
			float peak = value[0];
			#pragma unroll
			for ( uint32_t idx = 1u; idx < QUERY_KEYS; ++idx )
				peak = fmaxf(peak, value[idx]);
			#pragma unroll
			for ( uint32_t shift = 1u; shift < QUERY_THREADS; shift <<= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			const float previous_max = running_max[own];
			const float next_max = fmaxf(previous_max, peak);
			const float scale = previous_max == -INFINITY ? 0.0f : SparkMathExp2Fast((previous_max - next_max) * scale_log2);
			const float shift_val = next_max == -INFINITY ? 0.0f : next_max * scale_log2;
			if ( full_dense_tile )
			{
				#pragma unroll
				for ( uint32_t idx = 0u; idx < QUERY_KEYS; ++idx )
					weight[idx] = SparkMathExp2Fast(fmaf(value[idx], scale_log2, -shift_val));
			}
			else
			{
				#pragma unroll
				for ( uint32_t idx = 0u; idx < QUERY_KEYS; ++idx )
					weight[idx] = next_max == -INFINITY || value[idx] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(value[idx], scale_log2, -shift_val));
			}
			float total;
			if constexpr ( QUERY_KEYS == 4u )
			{
				*(uint2 *)(weight_high + own * LM_PREFILL_ATTN_WEIGHT_STRIDE + key_base_idx) = make_uint2(
					LmPackBf16Pair(weight[0], weight[1]),
					LmPackBf16Pair(weight[2], weight[3]));
				total = (weight[0] + weight[1]) + (weight[2] + weight[3]);
			}
			else
			{
				*(uint32_t *)(weight_high + own * LM_PREFILL_ATTN_WEIGHT_STRIDE + key_base_idx) = LmPackBf16Pair(weight[0], weight[1]);
				total = weight[0] + weight[1];
			}
			#pragma unroll
			for ( uint32_t shift = 1u; shift < QUERY_THREADS; shift <<= 1u )
				total += __shfl_xor_sync(0xffffffffu, total, shift);
			if ( part == 0u )
			{
				running_sum[own] = fmaf(running_sum[own], scale, total);
				running_max[own] = next_max;
				rescale[own] = scale;
			}
		}
		__syncthreads();
		{
			uint32_t a_high[QUERY_TILES][4], b[4], first[2], second[2];
			#pragma unroll
			for ( uint32_t m = 0u; m < QUERY_TILES; ++m )
			{
				const float low_scale = rescale[m * 16u + (lane >> 2u)], high_scale = rescale[m * 16u + (lane >> 2u) + 8u];
				if ( low_scale != 1.0f || high_scale != 1.0f )
				{
					#pragma unroll
					for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
					{
						output[m][frag][0] *= low_scale;
						output[m][frag][1] *= low_scale;
						output[m][frag][2] *= high_scale;
						output[m][frag][3] *= high_scale;
					}
				}
			}
			#pragma unroll
			for ( uint32_t step = 0u; step < TILE / 16u; ++step )
			{
				#pragma unroll
				for ( uint32_t m = 0u; m < QUERY_TILES; ++m )
				{
					const uint32_t a_row = m * 16u + (matrix & 1u) * 8u + within, a_col = step * 16u + (matrix >> 1u) * 8u;
					LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_col), a_high[m]);
				}
				#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u )
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					#pragma unroll
					for ( uint32_t m = 0u; m < QUERY_TILES; ++m )
					{
						LmMmaBf16(output[m][frag], a_high[m], first);
						LmMmaBf16(output[m][frag + 1u], a_high[m], second);
					}
				}
			}
		}
		if ( threadIdx.x < TILE && tile + 2u < tiles )
		{
			uint32_t valid;
			slots[buffer * TILE + threadIdx.x] = resolve(next_position, next_members, valid);
			key_rows[buffer * TILE + threadIdx.x] = valid;
		}
		next_position = ahead_position;
		next_members = ahead_members;
		__syncthreads();
	}
	#pragma unroll
	for ( uint32_t m = 0u; m < QUERY_TILES; ++m )
	{
		const uint32_t q_lo = m * 16u + (lane >> 2u), q_hi = q_lo + 8u;
		const uint32_t row_lo = first_row + q_lo / HEADS, row_hi = first_row + q_hi / HEADS;
		const float inv_lo = row_lo < row_end && running_sum[q_lo] > 0.0f ? 1.0f / running_sum[q_lo] : 0.0f;
		const float inv_hi = row_hi < row_end && running_sum[q_hi] > 0.0f ? 1.0f / running_sum[q_hi] : 0.0f;
		#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
		{
			const uint32_t col = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
			if ( row_lo < row_end )
				__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_lo * HEADS + q_lo % HEADS) * LATENT + col),
					LmPackBf16Pair(output[m][frag][0] * inv_lo, output[m][frag][1] * inv_lo));
			if ( row_hi < row_end )
				__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_hi * HEADS + q_hi % HEADS) * LATENT + col),
					LmPackBf16Pair(output[m][frag][2] * inv_hi, output[m][frag][3] * inv_hi));
		}
	}
	LmDependentRelease();
}

static int32_t Glm5NextMlaGatherPrefillLaunch(const Glm5NextLayerBuffers *b, const uint16_t *kv_all, uint32_t rows, uint32_t total_ctx,
	uint32_t sparse, uint32_t *union_positions, uint8_t *union_masks, uint32_t *union_counts, uint32_t union_capacity, SparkGpuStream stream)
{
	constexpr uint32_t QUERIES = 32u;
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / 4u;
	constexpr uint32_t SHARED_BYTES = LmPrefillAttnSharedBytes<G5N_LATENT, QUERIES>();
	const uint32_t blocks = (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK;
	if ( b->attn_heads != 4u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( sparse != 0u )
	{
		static std::atomic<bool> configured_sparse(false);
		if ( !configured_sparse.load(std::memory_order_acquire) )
		{
			if ( SparkGpuFuncSetAttribute((const void *)Glm5NextMlaGatherPrefillKernel<true, QUERIES>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)SHARED_BYTES) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			configured_sparse.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((LmPrefillSparseUnionKernel<ROWS_PER_BLOCK, G5N_INDEX_WIDTH>), blocks, LM_PREFILL_UNION_THREADS, 0, stream,
			b->selected_positions, b->positions, rows, (const uint32_t *)0, union_capacity, union_positions, union_masks, union_counts);
		LM_LAUNCH_DEPENDENT((Glm5NextMlaGatherPrefillKernel<true, QUERIES>), blocks, LM_PREFILL_ATTN_THREADS, SHARED_BYTES, stream,
			b->query_latent_bf16, kv_all, b->sequence_of_row, b->context_length, b->positions, G5N_QK_SCALE, b->attention_latent_bf16,
			rows, total_ctx, union_positions, union_masks, union_counts, union_capacity);
	}
	else
	{
		static std::atomic<bool> configured_dense(false);
		if ( !configured_dense.load(std::memory_order_acquire) )
		{
			if ( SparkGpuFuncSetAttribute((const void *)Glm5NextMlaGatherPrefillKernel<false, QUERIES>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)SHARED_BYTES) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			configured_dense.store(true, std::memory_order_release);
		}
		LM_LAUNCH_DEPENDENT((Glm5NextMlaGatherPrefillKernel<false, QUERIES>), blocks, LM_PREFILL_ATTN_THREADS, SHARED_BYTES, stream,
			b->query_latent_bf16, kv_all, b->sequence_of_row, b->context_length, b->positions, G5N_QK_SCALE, b->attention_latent_bf16,
			rows, total_ctx, (const uint32_t *)0, (const uint8_t *)0, (const uint32_t *)0, 0u);
	}
	return Glm5NextLaunched();
}

static int32_t Glm5NextAttentionOutput(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	if ( LmPerHeadProjectChainLaunch<G5N_THREADS, G5N_LATENT, G5N_VALUE>(b->attention_latent_bf16, (const uint16_t *)b->kv_b_value,
		b->attention_value_bf16, b->attn_heads, rows, stream, b->row_invariant, b->prefill_unified) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextLinear(b, b->attention_value_bf16, b->w->attn_output, b->attention_out_bf16, rows, b->attn_output_columns, G5N_HIDDEN, sms, stream);
}

static int32_t Glm5NextAttentionMerge(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream)
{
	if ( LmLatentShardMergeLaunch<G5N_LATENT>(b->shard_partials_received_f32, b->shard_partial_stride, b->cache.shard.degree, b->attn_heads,
		b->attention_latent_bf16, rows, stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return Glm5NextAttentionOutput(b, rows, sms, stream);
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
	if ( b->prefill_unified != 0u )
		status = LmStreamGemmDense<LmBf16Format>(w->router, LmScaleTensorNone(), b->normed_bf16, (uint16_t *)0, b->router_logits, rows, G5N_HIDDEN,
			G5N_EXPERTS, G5N_EXPERTS, 0u, sms, stream);
	else if ( b->row_invariant != 0u && rows > LM_SKINNY_ROWS_WIDE && LmIsAligned16(b->normed_bf16) && LmIsAligned16(w->router) )
		status = Glm5NextExactLinearLaunch(b->normed_bf16, w->router, (uint16_t *)0, b->router_logits, rows, G5N_HIDDEN, G5N_EXPERTS, 1u, stream);
	else
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

template<class Format>
static uint32_t Glm5NextStreamExperts(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t threshold = G5N_STREAM_EXPERT_ROWS)
{
	if constexpr ( LmStreamWeight<Format>::kSupported )
		return b->prefill_unified != 0u || (rows >= threshold && b->row_invariant == 0u) ? 1u : 0u;
	else
		return 0u;
}

#define G5N_EXPERTS_REFUSE(kernel, weight, rows, pairs, output_dimension, input_dimension) \
	LM_LAUNCH_REFUSE("glm5 experts: the %s refused %s for %u rows (%u routed pairs over %u experts, %u x %u); waves of %u rows or more that are " \
		"not row-invariant run on the grouped stream GEMM and all others on the skinny kernels, so the expert weights need 16-byte alignment, an " \
		"input that is a multiple of the codec's elements and scales that cover every expert\n", kernel, weight, rows, pairs, G5N_EXPERTS, \
		output_dimension, input_dimension, G5N_STREAM_EXPERT_ROWS)

#include "glm5_next_moe.cuh"

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
	{
		constexpr bool is_fp8 = ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3;
		const bool fast_aligned = b->expert_intermediate == 128u && b->shared_intermediate == 128u &&
			LmIsAligned16(b->expert_w1_weight) && LmIsAligned16(b->expert_w2_weight) &&
			LmIsAligned16(b->w->shared_gate_up) && LmIsAligned16(b->w->shared_down) && LmIsAligned16(b->normed_bf16);
		if ( b->prefill_unified == 0u && rows == 1u && fast_aligned )
		{
			const uint32_t w1_blocks = G5N_TOP_K * (b->expert_intermediate / LM_SKINNY_SPLIT_WARPS) + b->shared_intermediate / 2u;
			if ( Glm5NextSkl2Launch(is_fp8 ? GLM5NEXTSKL2_g5n_moe_b1_w1_act_fp8 : GLM5NEXTSKL2_g5n_moe_b1_w1_act_bf16,
				w1_blocks, LM_SKINNY_THREADS, 0u, stream, (const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale,
				(const uint16_t *)b->w->shared_gate_up, b->normed_bf16, b->route_expert, b->intermediate_bf16, b->gate_up_bf16,
				b->expert_intermediate) != SPARK_GPU_SUCCESS ||
				Glm5NextSkl2Launch(is_fp8 ? GLM5NEXTSKL2_g5n_moe_b1_w2_finalize_fp8 : GLM5NEXTSKL2_g5n_moe_b1_w2_finalize_bf16,
				G5N_HIDDEN / (is_fp8 ? 32u : 16u), 256u, 0u, stream, (const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale,
				(const uint16_t *)b->w->shared_down, b->intermediate_bf16, b->gate_up_bf16, b->route_expert, b->route_weight,
				b->attention_out_bf16) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			return LM_LAUNCH_OK;
		}
		if ( b->prefill_unified == 0u && rows >= 2u && rows <= 4u && fast_aligned )
		{
			constexpr uint32_t w2_cols = is_fp8 ? 32u : 16u;
			if ( Glm5NextSkl2Launch(is_fp8 ? GLM5NEXTSKL2_g5n_moe_exact_w1_act_fp8 : GLM5NEXTSKL2_g5n_moe_exact_w1_act_bf16,
				dim3(b->expert_intermediate / 4u, G5N_EXPERTS), 128u, 0u, stream, (const uint8_t *)b->expert_w1_weight,
				(const float *)b->expert_w1_scale, b->normed_bf16, b->group_row_offset, b->route_source_token, b->intermediate_bf16) != SPARK_GPU_SUCCESS ||
				Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_shared_w1_act_tile_r4, dim3(b->shared_intermediate / 2u, 1u), LM_SKINNY_THREADS, 0u, stream,
				(const uint16_t *)b->w->shared_gate_up, b->normed_bf16, b->gate_up_bf16, rows) != SPARK_GPU_SUCCESS ||
				Glm5NextSkl2Launch(is_fp8 ? GLM5NEXTSKL2_g5n_moe_small_w2_finalize_fp8 : GLM5NEXTSKL2_g5n_moe_small_w2_finalize_bf16,
				dim3(G5N_HIDDEN / w2_cols, rows), 256u, 0u, stream, (const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale,
				(const uint16_t *)b->w->shared_down, b->intermediate_bf16, b->gate_up_bf16, b->route_expert, b->route_packed_row, b->route_weight,
				b->attention_out_bf16) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			return LM_LAUNCH_OK;
		}
		if ( (rows > 4u || b->prefill_unified != 0u) && fast_aligned && LmIsAligned16(b->intermediate_bf16) && LmIsAligned16(b->expert_out_bf16) )
		{
			if ( Glm5NextStreamExperts<ExpertFormat>(b, rows) != 0u )
				status = Glm5NextMoeMmaW1ActLaunch<ExpertCodec>(b, packed_rows, sms, stream);
			else if constexpr ( is_fp8 )
			{
				if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_moe_exact_w1_act_fp8, dim3(b->expert_intermediate / 4u, G5N_EXPERTS), 128u, 0u, stream,
					(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, b->normed_bf16,
					b->group_row_offset, b->route_source_token, b->intermediate_bf16) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
				status = LM_LAUNCH_OK;
			}
			else
			{
				status = LmSkinnyGroupedExperts<LmBf16Format>(b->expert_w1_weight, LmScaleTensorNone(), b->normed_bf16, b->gate_up_bf16,
					b->group_row_offset, b->route_source_token, G5N_EXPERTS, packed_rows, 0u, G5N_HIDDEN, b->expert_w1_rows, stream, b->row_invariant);
				if ( status == LM_LAUNCH_OK )
				{
					if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_clamped_up_gate_packed128_256, (packed_rows + 15u) / 16u, 256u, 0u, stream,
						b->gate_up_bf16, b->intermediate_bf16, packed_rows) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
					status = LM_LAUNCH_OK;
				}
			}
			if ( status != LM_LAUNCH_OK )
				return status;
			if ( b->prefill_unified != 0u && rows >= 512u && b->gate_up_bf16 != 0 && LmIsAligned16(b->gate_up_bf16) )
			{
				status = Glm5NextSharedMmaW1ActLaunch(b, rows, sms, stream, b->gate_up_bf16);
				if ( status != LM_LAUNCH_OK )
					return status;
				return Glm5NextMoeMmaW2SharedSlicedLaunch<ExpertCodec, 2u>(b, rows, packed_rows, sms, stream);
			}
			if ( Glm5NextStreamExperts<ExpertFormat>(b, rows, G5N_STREAM_EXPERT_W2_ROWS) != 0u )
				status = Glm5NextMoeMmaW2K128Launch<ExpertCodec>(b, packed_rows, sms, stream);
			else if ( rows <= 12u )
			{
				scale_b = LmWeightCodecScaleTensor<ExpertCodec>(b->expert_w2_scale, G5N_EXPERTS, G5N_HIDDEN, b->expert_intermediate);
				status = LmSkinnyGroupedExperts<ExpertFormat>(b->expert_w2_weight, scale_b, b->intermediate_bf16, b->expert_out_bf16,
					b->group_row_offset, b->route_source_token, G5N_EXPERTS, packed_rows, 1u, b->expert_intermediate, G5N_HIDDEN, stream,
					b->row_invariant);
			}
			else
			{
				constexpr uint32_t w2_cols = (is_fp8 ? 32u : 16u) * 2u;
				if ( Glm5NextSkl2Launch(is_fp8 ? GLM5NEXTSKL2_g5n_moe_exact_w2_fp8 : GLM5NEXTSKL2_g5n_moe_exact_w2_bf16,
					dim3(G5N_HIDDEN / w2_cols, G5N_EXPERTS), 256u, 0u, stream, (const uint8_t *)b->expert_w2_weight,
					(const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset, b->expert_out_bf16) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
				status = LM_LAUNCH_OK;
			}
			if ( status != LM_LAUNCH_OK )
				return status;
			if ( b->prefill_unified == 0u && (rows <= 16u || b->row_invariant != 0u) )
			{
				if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_shared_w1_act_tile_r8, dim3(b->shared_intermediate / 2u, (rows + 7u) / 8u),
					LM_SKINNY_THREADS, 0u, stream, (const uint16_t *)b->w->shared_gate_up, b->normed_bf16, b->intermediate_bf16, rows) != SPARK_GPU_SUCCESS ||
					Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_shared_down_finalize_exact_r8, dim3(G5N_HIDDEN / 16u, (rows + 7u) / 8u), 256u, 0u, stream,
					(const uint16_t *)b->w->shared_down, b->intermediate_bf16, b->expert_out_bf16, b->route_packed_row, b->route_weight,
					b->attention_out_bf16, rows) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
				return LM_LAUNCH_OK;
			}
			if ( b->prefill_unified != 0u || (b->row_invariant == 0u && rows >= 32u) )
			{
				status = Glm5NextSharedMmaW1ActLaunch(b, rows, sms, stream);
				if ( status != LM_LAUNCH_OK )
					return status;
				return Glm5NextSharedDownFinalizeAddLaunch(b, rows, sms, stream);
			}
			status = Glm5NextLinear(b, b->normed_bf16, b->w->shared_gate_up, b->gate_up_bf16, rows, G5N_HIDDEN, b->shared_gate_up_rows, sms, stream);
			if ( status != LM_LAUNCH_OK )
				return status;
			if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_clamped_up_gate_packed128_256, (rows + 15u) / 16u, 256u, 0u, stream,
				b->gate_up_bf16, b->intermediate_bf16, rows) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			status = Glm5NextLinear(b, b->intermediate_bf16, b->w->shared_down, b->shared_out_bf16, rows, b->shared_intermediate, G5N_HIDDEN, sms, stream);
			if ( status != LM_LAUNCH_OK )
				return status;
			if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_moe_finalize_add_256, dim3((G5N_HIDDEN / 8u + 255u) / 256u, rows), 256u, 0u, stream,
				b->expert_out_bf16, b->route_packed_row, b->route_weight, b->shared_out_bf16, b->attention_out_bf16, rows) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			return LM_LAUNCH_OK;
		}
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
	{
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_clamped_up_gate_packed128_256, (packed_rows + 15u) / 16u, 256u, 0u, stream,
			b->gate_up_bf16, b->intermediate_bf16, packed_rows) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
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
	{
		if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_clamped_up_gate_packed128_256, (rows + 15u) / 16u, 256u, 0u, stream,
			b->gate_up_bf16, b->intermediate_bf16, rows) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
		LM_LAUNCH((LmClampedUpGateKernel<G5N_THREADS>), rows, G5N_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->shared_intermediate,
			G5N_SWIGLU_LIMIT);
	status = Glm5NextLinear(b, b->intermediate_bf16, b->w->shared_down, b->shared_out_bf16, rows, b->shared_intermediate, G5N_HIDDEN, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_moe_finalize_add_256, dim3((G5N_HIDDEN / 8u + 255u) / 256u, rows), 256u, 0u, stream,
		b->expert_out_bf16, b->route_packed_row, b->route_weight, b->shared_out_bf16, b->attention_out_bf16, rows) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

#include "glm5_next_stage_model.cuh"
