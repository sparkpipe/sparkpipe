#pragma once
#include <atomic>
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/project_chain.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_head_screen.h"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_select.cuh"
#include "inference/kernels/attn_expanded_prefill.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "model-families/k3/source/config.h"
#include "model-families/k3/source/generated_config.h"
#include "sparkpipe/spark_gpu.h"
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
#include "inference/kernels/attn_prefill.cuh"
#include "model-families/k3/source/k3_fast.cuh"
#endif

using K3GlobalKv = LmKvLatent<K3_KV_BITS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM, K3_KV_PAGE_SLOTS>;


#include "model-families/k3/source/launch_shape.h"

static_assert(K3_KDA_QK_L2NORM == 1u, "kda qk l2norm is part of the kernel contract");
static_assert(K3_KDA_A_LOG_SOURCE_HEADS == 128u && K3_KDA_HEADS == 96u, "A_log loads 128 heads and narrows to 96");

#define K3_KDA_QK_DIM (K3_KDA_HEADS * K3_KDA_KEY_DIM)
#define K3_KDA_V_DIM (K3_KDA_HEADS * K3_KDA_VALUE_DIM)

#define K3_KDA_QKVB_K_OFFSET K3_KDA_QK_DIM
#define K3_KDA_QKVB_V_OFFSET (2u * K3_KDA_QK_DIM)
#define K3_KDA_QKVB_BETA_OFFSET (2u * K3_KDA_QK_DIM + K3_KDA_V_DIM)
#define K3_KDA_GATE_DOWN_OFFSET K3_KDA_KEY_DIM
#define K3_KDA_DELTA_COLUMNS 32u
static_assert(K3_KDA_QKVB_BETA_OFFSET + K3_KDA_HEADS == K3_KDA_QKVB_FUSED_ROWS,
	"the q|k|v|beta sections must tile the fused tensor exactly");
static_assert(K3_KDA_GATE_DOWN_OFFSET + K3_KDA_KEY_DIM == K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS,
	"the decay_down|gate_down sections must tile the fused tensor exactly");

#define K3_MLA_Q_DIM (K3_MLA_HEADS * (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM))
#define K3_MLA_KV_A_DIM (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM)
#define K3_MLA_KV_B_DIM (K3_MLA_HEADS * (K3_QK_NOPE_DIM + K3_V_HEAD_DIM))
#define K3_MLA_LATENT_OUT_DIM (K3_MLA_HEADS * K3_KV_LORA_RANK)
#define K3_MLA_OUT_DIM (K3_MLA_HEADS * K3_V_HEAD_DIM)

static_assert(K3_MLA_Q_DIM == K3_MLA_HEADS * (K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM),
	"the query must be as wide as the kernel reads");
static_assert(K3_MLA_OUT_DIM == K3_MLA_HEADS * K3_V_HEAD_DIM,
	"the gate and output projection live in v-space, not the latent");

#define K3_SHARED_INTERMEDIATE (K3_EXPERT_INTERMEDIATE * K3_SHARED_EXPERTS)

static_assert(K3_KDA_HEADS == K3_MLA_HEADS,
	"the report gives one head count for both attention kinds");
static_assert(K3_LAYERS % 4u == 1u,
	"93 layers is 23 whole blocks plus the trailing MLA layer");

static_assert(K3_HIDDEN % 256u == 0u, "KDA and MLA project from the hidden");
static_assert(K3_KDA_KEY_DIM % 128u == 0u,
	"the decay bottleneck must be a whole BF16 tile");
static_assert(K3_KDA_V_DIM % 256u == 0u, "the KDA output projection");
static_assert(K3_Q_LORA_RANK % 256u == 0u, "the MLA query up-projection");
static_assert(K3_MLA_OUT_DIM % 256u == 0u, "the MLA output projection");
static_assert(K3_ROUTED_EXPERT_HIDDEN % 256u == 0u, "the routed experts' input");
static_assert(K3_EXPERT_INTERMEDIATE % 256u == 0u, "the routed down-projection");
static_assert(K3_SHARED_INTERMEDIATE % 256u == 0u, "the shared down-projection");
static_assert(K3_DENSE_INTERMEDIATE % 256u == 0u, "layer 0's dense down-projection");

struct K3LayerBuffers
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;

	const void *kda_qkv_beta_weight;
	const void *kda_decay_down_weight;
	const float *kda_q_conv_weight;
	const float *kda_k_conv_weight;
	const float *kda_v_conv_weight;
	const void *kda_decay_up_weight;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	const void *kda_gate_weight;
	const float *kda_out_norm_weight;
	const void *kda_out_weight;
	const void *kda_out_scale;

	const void *mla_q_down_weight;
	const void *mla_q_down_scale;
	const void *mla_q_norm_weight;
	const void *mla_q_up_weight;
	const void *mla_q_up_scale;
	const void *mla_kv_a_weight;
	const void *mla_kv_a_scale;
	const void *mla_kv_a_norm_weight;
	const void *mla_kv_b_value_weight;
	const void *mla_q_b_weight;
	const void *mla_kv_b_key_full_weight;
	const void *mla_kv_b_value_full_weight;
	const void *mla_kv_b_scale;
	const void *mla_gate_weight;
	const void *mla_out_weight;
	const void *mla_out_scale;
	const void *mla_index_query_weight;
	const void *mla_index_key_weight;
	const void *mla_index_head_weight;
	const void *mla_index_norm_weight;
	const void *mla_index_norm_bias;

	const void *router_weight;
	const float *router_bias;
	float *router_logits;
	const void *routed_down_weight;
	const void *routed_down_scale;
	const void *routed_up_weight;
	const void *routed_up_scale;
	const void *routed_norm_weight;
	const void *expert_w1_weight;
	const void *expert_w2_weight;
	uint32_t expert_interleave;
	uint32_t expert_tile_k;
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;

	uint16_t *hidden_bf16;
	uint16_t *normed_bf16;
	uint16_t *fused_qkvb_bf16;
	uint16_t *query_bf16;
	uint16_t *key_bf16;
	uint16_t *value_bf16;
	uint16_t *gate_bf16;
	uint16_t *decay_logit_bf16;
	uint16_t *latent_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *shared_out_bf16;
	uint16_t *latent_full_bf16;
	uint16_t *shared_mid_bf16;
	uint16_t *attnres_bank_bf16;
	uint16_t *attnres_partial_bf16;
	float *attnres_score_f32;
	const void *attnres_attn_weight;
	const void *attnres_mlp_weight;
	const void *attnres_out_weight;
	const void *attnres_next_weight;
	uint16_t *kda_beta_logit;
	float *kda_write_gate_out;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	const uint32_t *sequence_row_begin;
	const uint32_t *sequence_row_indices;
	uint16_t *replay_conv_q;
	uint16_t *replay_conv_k;
	uint16_t *replay_conv_v;
	float *replay_retention;
	float *replay_write_gate;

	uint32_t tp_sharded;
	uint32_t tp_rank;
	uint32_t kda_qkvb_rows;
	uint32_t kda_gate_rows;
	uint32_t kda_decay_up_rows;
	uint32_t kda_out_input;
	uint32_t mla_q_up_rows;
	uint32_t mla_kv_b_value_rows;
	uint32_t mla_gate_rows;
	uint32_t mla_out_input;
	uint32_t routed_down_rows;
	uint32_t routed_up_input;
	uint32_t expert_w1_output;
	uint32_t expert_w2_input;
	uint32_t shared_w1_rows;
	uint32_t shared_w2_input;
	uint32_t dense_gate_up_rows;
	uint32_t dense_down_input;
	uint32_t kda_heads_rank;
	uint32_t mla_heads_rank;
#define K3_RANK_DIM(b, field, constant) \
	((b)->field != 0u ? (b)->field : (constant))

	uint8_t *kda_state_pool;
	uint32_t kda_state_bf16;
	uint16_t *kda_q_window;
	uint16_t *kda_k_window;
	uint16_t *kda_v_window;
	const uint32_t *kda_state_index;
	float *kda_retention;

	LmKvView cache;
	SparkKvShard kv_shard;
	LmKvShardView cache_shard;
	LmKvView index_cache;
	LmKvShardView index_shard;
	uint16_t *index_query_bf16;
	uint16_t *index_key_bf16;
	uint16_t *index_head_bf16;
	float *index_scores;
	uint64_t index_score_floats;
	uint32_t index_score_stride;
	uint2 *index_send;
	uint2 *index_received;
	uint32_t *index_bitmaps;
	uint32_t index_bitmap_words;
	uint32_t *index_selected;
	uint32_t *index_selected_own;
	LmFrameError *index_error;
	uint32_t index_chunk_rows;
	uint32_t index_chunk;
	uint32_t index_keep;
	uint32_t index_active;
	uint16_t *shard_query_gathered_bf16;
	float *shard_partials_f32;
	float *shard_partials_received_f32;
	uint64_t shard_received_floats;
	uint64_t shard_query_rank_stride;
	uint64_t shard_partial_rank_stride;
	uint32_t shard_gather;
	uint32_t shard_gather_most_keys;
	const uint32_t *shard_gather_list;
	const uint32_t *shard_gather_offset;
	const uint32_t *shard_gather_context;
	uint8_t *shard_gather_send;
	const uint8_t *shard_gather_keys;
	uint64_t shard_gather_stride;
	uint32_t shard_gather_windows;
	uint32_t shard_gather_window_keys;
	uint32_t shard_gather_window;
	uint64_t shard_gather_buffer_bytes;
	float *shard_gather_running;
	uint8_t *shard_gather_values;
	uint64_t shard_gather_values_bytes;
	uint32_t shard_gather_expanded;
	uint32_t shard_gather_passes;
	uint32_t shard_gather_value_keys;
	uint32_t shard_gather_history;
	uint32_t shard_gather_prefetched;
	uint32_t shard_gather_prior_ctx;
	uint32_t shard_gather_total_ctx;
	uint32_t shard_gather_direct;
	uint32_t shard_mha;
	uint8_t *shard_mha_keys;
	uint64_t shard_mha_keys_bytes;
	uint16_t *shard_mha_query_bf16;
	uint32_t row_invariant;
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
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *route_expert;
	uint32_t *route_fault;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
	uint32_t normed_ready;
	uint32_t defer_partial_add;
	const uint16_t *pending_partial_add;
	uint32_t fuse_shared_out;
};

static inline uint32_t K3CanFuseRoutedUpShared(const K3LayerBuffers *b, uint32_t rows)
{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	const uint32_t in_dim = K3_RANK_DIM(b, routed_up_input, K3_ROUTED_EXPERT_HIDDEN);
	return(b->fuse_shared_out != 0u && b->expert_w1_output != 0u && b->expert_w1_output < K3_EXPERT_INTERMEDIATE * 2u &&
		b->tp_sharded != 0u && b->routed_up_scale == 0 && b->shared_out_bf16 != 0 && rows > 64u && b->row_invariant == 0u &&
		(in_dim % 32u) == 0u && LmIsAligned16(b->latent_bf16) && LmIsAligned16(b->routed_up_weight) &&
		LmIsAligned16(b->shared_out_bf16) && LmIsAligned16(b->hidden_bf16) ? 1u : 0u);
#else
	(void)b;
	(void)rows;
	return(0u);
#endif
}

#define K3_EXPERT_CELLS(b) \
	((b)->expert_w1_output != 0u && (b)->expert_w1_output < K3_EXPERT_INTERMEDIATE * 2u)

#define K3_EXPERT_CELL_REFUSE(name,rows,pairs,outputs,inputs) \
	LM_LAUNCH_REFUSE("k3 experts: the cell kernel refused " name " for %u rows (%u routed pairs, %u x %u); a pack interleaved in %u-deep tiles runs its " \
		"experts on it when the input is a multiple of %u and the output of %u, so the weights and activation need 16-byte alignment and the route " \
		"tables must be bound\n",(rows),(pairs),(outputs),(inputs),LM_SKINNY_CELL_TILE_K,LM_SKINNY_CELL_TILE_K,LM_SKINNY_CELL_NEURONS)

static uint32_t K3ExpertCellKernel(const K3LayerBuffers *b, uint32_t input_dimension, uint32_t output_dimension)
{
	return(b->expert_interleave != 0u && LmSkinnyCellShapeFits(input_dimension,output_dimension,b->expert_tile_k) != 0u ? 1u : 0u);
}

#define K3_ROUTER_SLICE(b,rows) \
	((float *)((b)->latent_bf16 + (uint64_t)(rows) * K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN)))

static int32_t K3SkinnyRows(const void *weight, const uint16_t *source, uint16_t *destination_bf16, float *destination_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t split, SparkGpuStream stream)
{
	return((split != 0u ? LmSkinnyDenseSplit<LmBf16Format> : LmSkinnyDense<LmBf16Format>)(weight,source,destination_bf16,destination_f32,rows,input_dimension,
		output_dimension,0u,0u,stream,1u));
}

#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
static __device__ __forceinline__ float K3Dot8Unpacked(const float *wf, const float *af)
{
	float a0 = wf[0] * af[0], a1 = wf[1] * af[1];
	a0 = fmaf(wf[2], af[2], a0);
	a1 = fmaf(wf[3], af[3], a1);
	a0 = fmaf(wf[4], af[4], a0);
	a1 = fmaf(wf[5], af[5], a1);
	a0 = fmaf(wf[6], af[6], a0);
	a1 = fmaf(wf[7], af[7], a1);
	return(a0 + a1);
}

template<uint32_t RG, uint32_t NPG>
__global__ __launch_bounds__(256u, 2)
void K3ExactSplitKernel(const __grid_constant__ LmSkinnyMultiArguments multi)
{
	constexpr uint32_t SW = LM_SKINNY_SPLIT_WARPS;
	__shared__ float partial[SW][NPG][RG];
	const LmSkinnyArguments &head = multi.part[0];
	const uint32_t rows = head.rows, chunks = head.input_dimension / 8u;
	const uint32_t neuron_task = blockIdx.x, base_row = blockIdx.y * RG;
	const uint32_t count = base_row < rows ? (rows - base_row < RG ? rows - base_row : RG) : 0u;
	LmDependentWait();
	if ( count == 0u )
	{
		LmDependentRelease();
		return;
	}
	uint32_t part = 0u;
	while ( part + 1u < multi.count && neuron_task >= multi.first_task[part + 1u] )
		part++;
	const LmSkinnyArguments &args = multi.part[part];
	const uint32_t neuron = (neuron_task - multi.first_task[part]) * NPG;
	const uint32_t warp = threadIdx.x / 32u, sub = threadIdx.x % 32u;
	const uint32_t first = chunks * warp / SW, last = chunks * (warp + 1u) / SW;
	const uint4 *w_ptrs[NPG];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
	{
		const uint32_t rn = neuron + n < args.output_dimension ? neuron + n : neuron;
		w_ptrs[n] = (const uint4 *)(args.weight + (uint64_t)rn * chunks * 16u);
	}
	float acc[NPG][RG];
	#pragma unroll
	for ( uint32_t n = 0u; n < NPG; ++n )
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
			acc[n][r] = 0.0f;
	for ( uint32_t c = first + sub; c < last; c += 32u )
	{
		float wf[NPG][8];
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
			LmSkinnyUnpackBf16x8(__ldcs(w_ptrs[n] + c), wf[n]);
		const uint16_t *act_base = head.activation + (uint64_t)base_row * head.input_dimension + c * 8u;
		#pragma unroll
		for ( uint32_t r = 0u; r < RG; ++r )
		{
			const uint32_t rr = r < count ? r : 0u;
			const uint4 a = __ldg((const uint4 *)(act_base + (uint64_t)rr * head.input_dimension));
			float af[8];
			LmSkinnyUnpackBf16x8(a, af);
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; ++n )
				acc[n][r] += K3Dot8Unpacked(wf[n], af);
		}
	}
	#pragma unroll
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], offset);
	if ( sub == 0u )
	{
		#pragma unroll
		for ( uint32_t n = 0u; n < NPG; ++n )
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				partial[warp][n][r] = acc[n][r];
	}
	__syncthreads();
	for ( uint32_t idx = threadIdx.x; idx < NPG * RG; idx += 256u )
	{
		const uint32_t n = idx / RG, r = idx % RG;
		if ( r < count && neuron + n < args.output_dimension )
		{
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < SW; ++w )
				total += partial[w][n][r];
			const uint64_t oidx = (uint64_t)(base_row + r) * args.output_row_stride + args.output_column_offset + neuron + n;
			if ( args.output_f32 != 0 )
				args.output_f32[oidx] = total;
			else
				args.output_bf16[oidx] = LmFloatToBf16(total);
		}
	}
	LmDependentRelease();
}

static int32_t K3ExactSplitMultiLaunch(const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *activation, uint32_t rows, uint32_t input_dimension, SparkGpuStream stream)
{
	LmSkinnyMultiArguments multi = {};
	if ( targets == 0 || activation == 0 || count == 0u || count > LM_SKINNY_MULTI_MAX || rows == 0u ||
		(input_dimension & 7u) != 0u || input_dimension / 8u < LM_SKINNY_SPLIT_WARPS * 32u || !LmIsAligned16(activation) )
		return(LmSkinnyDenseMulti<LmBf16Format>(targets,count,activation,rows,input_dimension,stream,1u,1u));
	multi.count = count;
	multi.first_task[0] = 0u;
	for ( uint32_t p = 0u; p < count; ++p )
	{
		LmSkinnyArguments *a = &multi.part[p];
		if ( targets[p].weight == 0 || !LmIsAligned16(targets[p].weight) || targets[p].output_dimension == 0u ||
			(targets[p].output_bf16 == 0 && targets[p].output_f32 == 0) )
			return(LmSkinnyDenseMulti<LmBf16Format>(targets,count,activation,rows,input_dimension,stream,1u,1u));
		a->weight = (const uint8_t *)targets[p].weight;
		a->activation = activation;
		a->output_bf16 = targets[p].output_bf16;
		a->output_f32 = targets[p].output_f32;
		a->rows = rows;
		a->input_dimension = input_dimension;
		a->output_dimension = targets[p].output_dimension;
		a->output_row_stride = targets[p].output_row_stride != 0u ? targets[p].output_row_stride : targets[p].output_dimension;
		a->output_column_offset = targets[p].output_column_offset;
		multi.first_task[p + 1u] = multi.first_task[p] + (a->output_dimension + 3u) / 4u;
		LmWeightPrefetchNote(stream, a->weight, (uint64_t)a->output_dimension * input_dimension * sizeof(uint16_t));
	}
	LM_LAUNCH_DEPENDENT((K3ExactSplitKernel<8u,4u>), dim3(multi.first_task[count], (rows + 7u) / 8u), 256u, 0, stream, multi);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Geometry, class Pages, uint32_t INDEX_HEADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
void K3SparseScoreSlotsKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight, Pages index_view,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position,
	uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores, uint32_t dense_limit)
{
	const LmKvView &index_cache = LmIndexScorePages(index_view);
	constexpr uint32_t HEAD_TILES = INDEX_HEADS / 16u;
	constexpr uint32_t STEPS = INDEX_DIM / 16u;
	constexpr uint32_t CHUNK_TILES = LM_INDEX_SCORE_STAGE_POSITIONS / LM_INDEX_SCORE_TILE_POSITIONS;
	constexpr uint32_t CHUNKS = LM_INDEX_SCORE_BLOCK_POSITIONS / LM_INDEX_SCORE_STAGE_POSITIONS;
	constexpr uint32_t PITCH = INDEX_DIM * (uint32_t)sizeof(uint16_t) + LM_INDEX_SCORE_STAGE_PAD;
	constexpr uint32_t STAGE_BYTES = LM_INDEX_SCORE_STAGE_POSITIONS * PITCH;
	constexpr uint32_t ROW_PIECES = (INDEX_DIM * (uint32_t)sizeof(uint16_t)) / 16u;
	constexpr uint32_t COPIES = (LM_INDEX_SCORE_STAGE_POSITIONS * ROW_PIECES) / LM_INDEX_SCORE_THREADS;
	__shared__ const uint8_t *block_slots[LM_INDEX_SCORE_BLOCK_POSITIONS];
	__shared__ alignas(16) uint8_t staged[LM_INDEX_SCORE_STAGES * STAGE_BYTES];
	__shared__ uint32_t block_sequence, block_mixed, block_bound;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t group = lane / 4u, pair = lane % 4u;
	const uint32_t row = blockIdx.y * LM_INDEX_SCORE_WARPS + warp;
	const uint32_t block_first = blockIdx.x * LM_INDEX_SCORE_BLOCK_POSITIONS;
	const uint32_t live = row < rows ? 1u : 0u;
	uint32_t a[HEAD_TILES][STEPS][4], next[STEPS][2], b[STEPS][2];
	float weight[2u * HEAD_TILES];
	uint32_t sequence = 0xffffffffu, bound = 0u, tile, step, head_tile, valid = 0u;
	if ( scores == 0 )
		return;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		block_sequence = 0xffffffffu;
		block_mixed = 0u;
		block_bound = 0u;
	}
	__syncthreads();
	if ( live != 0u )
	{
		valid = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
		sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
		if ( valid == 0u || sequence >= index_cache.sequence_count )
		{
			if ( lane == 0u && blockIdx.x == 0u )
				LmKvReportRequiredAccessFailure(index_cache, valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, block_first, 0xffffffffu);
			valid = 0u;
		}
		else
		{
			bound = context_length[sequence];
			if ( row_position != 0 && row_position[row] + 1u < bound )
				bound = row_position[row] + 1u;
			if ( dense_limit != 0u && bound <= dense_limit )
				bound = 0u;
			else
				bound = LmIndexScoreKeys(index_view, bound);
		}
	}
	if ( lane == 0u && valid != 0u && bound > block_first )
	{
		(void)atomicCAS(&block_sequence, 0xffffffffu, sequence);
		(void)atomicMax(&block_bound, bound);
	}
	__syncthreads();
	if ( block_bound <= block_first )
	{
		LmDependentRelease();
		return;
	}
	if ( lane == 0u && valid != 0u && bound > block_first && block_sequence != sequence )
		block_mixed = 1u;
	if ( live != 0u && valid != 0u && bound > block_first )
	{
		const uint16_t *query = index_query_bf16 + (uint64_t)row * INDEX_HEADS * INDEX_DIM;
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
			for (step = 0u; step < STEPS; step++)
			{
				const uint16_t *top = query + (uint64_t)(head_tile * 16u + group) * INDEX_DIM + step * 16u + pair * 2u;
				const uint16_t *bottom = top + 8u * INDEX_DIM;
				a[head_tile][step][0] = *(const uint32_t *)top;
				a[head_tile][step][1] = *(const uint32_t *)bottom;
				a[head_tile][step][2] = *(const uint32_t *)(top + 8u);
				a[head_tile][step][3] = *(const uint32_t *)(bottom + 8u);
			}
		for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
		{
			weight[2u * head_tile] = LmIndexScoreWeight(head_weight[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group]);
			weight[2u * head_tile + 1u] = LmIndexScoreWeight(head_weight[(uint64_t)row * INDEX_HEADS + head_tile * 16u + group + 8u]);
		}
	}
	__syncthreads();
	const uint32_t shared_slots = block_mixed == 0u && block_sequence != 0xffffffffu ? 1u : 0u;
	if ( shared_slots != 0u )
	{
		const uint32_t position = block_first + threadIdx.x;
		block_slots[threadIdx.x] = position < block_bound && position < score_stride ?
			LmIndexScoreSlot<Geometry>(index_view, block_sequence, position, blockIdx.y * LM_INDEX_SCORE_WARPS) : 0;
	}
	__syncthreads();
	if ( shared_slots != 0u )
	{
		const uint32_t staged_s = (uint32_t)__cvta_generic_to_shared(staged), matrix_lane = (lane & 7u) * PITCH + (lane >> 3u) * 16u;
		auto produce = [&](uint32_t chunk, uint32_t slot)
		{
			#pragma unroll
			for ( uint32_t q = 0u; q < COPIES; q++ )
			{
				const uint32_t piece = threadIdx.x + q * LM_INDEX_SCORE_THREADS, key = piece / ROW_PIECES, part = piece % ROW_PIECES;
				const uint8_t *source = block_slots[chunk * LM_INDEX_SCORE_STAGE_POSITIONS + key];
				LmIndexScoreCopy16(staged_s + slot * STAGE_BYTES + key * PITCH + part * 16u, source != 0 ? source + part * 16u : (const uint8_t *)index_query_bf16,
					source != 0 ? 16u : 0u);
			}
		};
		#pragma unroll
		for ( uint32_t stage = 0u; stage + 1u < LM_INDEX_SCORE_STAGES; stage++ )
		{
			produce(stage, stage);
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		}
		for ( uint32_t chunk = 0u; chunk < CHUNKS; chunk++ )
		{
			asm volatile("cp.async.wait_group %0;\n" :: "n"(LM_INDEX_SCORE_STAGES - 2u) : "memory");
			__syncthreads();
			if ( chunk + LM_INDEX_SCORE_STAGES - 1u < CHUNKS )
				produce(chunk + LM_INDEX_SCORE_STAGES - 1u, (chunk + LM_INDEX_SCORE_STAGES - 1u) % LM_INDEX_SCORE_STAGES);
			asm volatile("cp.async.commit_group;\n" ::: "memory");
			if ( live == 0u || valid == 0u || bound <= block_first )
				continue;
			const uint32_t base = staged_s + (chunk % LM_INDEX_SCORE_STAGES) * STAGE_BYTES + matrix_lane;
			static_assert((CHUNK_TILES & 1u) == 0u, "index score chunks hold paired key tiles");
			#pragma unroll
			for ( tile = 0u; tile < CHUNK_TILES; tile += 2u )
			{
				const uint32_t first0 = block_first + (chunk * CHUNK_TILES + tile) * LM_INDEX_SCORE_TILE_POSITIONS;
				if ( first0 >= bound || first0 >= score_stride )
					break;
				LmIndexScoreStagedTilePair<HEAD_TILES, STEPS, PITCH, false>(a, weight, base, tile, first0, valid, bound, pair, group, row, score_stride, qk_scale, scores);
			}
		}
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		LmDependentRelease();
		return;
	}
	if ( live == 0u || valid == 0u || bound <= block_first )
	{
		LmDependentRelease();
		return;
	}
	LmIndexScoreKeyFragments<STEPS>(block_first + group < bound && block_first < score_stride ?
		LmIndexScoreSlot<Geometry>(index_view, sequence, block_first + group, row) : 0, pair, next);
	for (tile = 0u; tile < LM_INDEX_SCORE_BLOCK_TILES; tile++)
	{
		const uint32_t first = block_first + tile * LM_INDEX_SCORE_TILE_POSITIONS;
		const uint32_t ahead = first + LM_INDEX_SCORE_TILE_POSITIONS + group;
		const uint32_t written = first + pair * 2u;
		float accumulator[HEAD_TILES][4] = {};
		if ( first >= bound || first >= score_stride )
			break;
		for (step = 0u; step < STEPS; step++)
		{
			b[step][0] = next[step][0];
			b[step][1] = next[step][1];
		}
		LmIndexScoreKeyFragments<STEPS>(tile + 1u < LM_INDEX_SCORE_BLOCK_TILES && ahead < bound && first + LM_INDEX_SCORE_TILE_POSITIONS < score_stride ?
			LmIndexScoreSlot<Geometry>(index_view, sequence, ahead, row) : 0, pair, next);
		for (step = 0u; step < STEPS; step++)
			for (head_tile = 0u; head_tile < HEAD_TILES; head_tile++)
				LmMmaBf16(accumulator[head_tile], a[head_tile][step], b[step]);
		LmIndexScoreStore<HEAD_TILES, false>(accumulator, weight, valid, bound, written, group, row, score_stride, qk_scale, scores);
	}
	LmDependentRelease();
}
#endif

#define K3_STREAM_INPUT_MULTIPLE LmStreamInputMultiple<LmBf16Format>()
#define K3_TILED_INPUT_MULTIPLE 32u

static uint32_t K3ProjectStreams(uint32_t input_dimension, const uint16_t *accumulate)
{
	return accumulate == 0 && input_dimension % K3_STREAM_INPUT_MULTIPLE == 0u ? 1u : 0u;
}

template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint16_t *accumulate, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	LmGemmArguments gemm;
	int32_t status;
	static_assert(Format::kScaleGroup == 0u,
		"K3Project carries the unquantised projections; experts go weight-only");
	if (weight_scale != 0)
		LM_LAUNCH_REFUSE("k3 projection: a %u x %u projection came with a weight scale; K3 projections are unquantised and experts go weight-only\n",
			output_dimension,input_dimension);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( accumulate == 0 && rows > LM_SKINNY_ROWS_WIDE && rows <= 64u && b->row_invariant == 0u &&
		output_dimension >= 2048u && (output_dimension & 1u) == 0u && (input_dimension & 7u) == 0u &&
		LmIsAligned16(source) && LmIsAligned16(weight) && LmIsAligned16(destination) )
		return(K3WideMmaLaunch(weight, source, destination, rows, input_dimension, output_dimension, stream));
#endif
	if ( accumulate == 0 && (rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u) )
	{
		status = K3SkinnyRows(weight,source,destination,(float *)0,rows,input_dimension,output_dimension,0u,stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return(status);
		LM_LAUNCH_REFUSE("k3 projection: the skinny kernel refused %u rows of a %u x %u projection; waves of up to %u rows and row-invariant waves "
			"run on it, so the weight and activation need 16-byte alignment and an input that is a multiple of 8\n",rows,output_dimension,
			input_dimension,LM_SKINNY_ROWS_WIDE);
	}
	if constexpr ( LmStreamWeight<Format>::kSupported )
		if ( K3ProjectStreams(input_dimension,accumulate) != 0u )
		{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
			LmWeightPrefetchNote(stream, weight, (uint64_t)output_dimension * input_dimension * sizeof(uint16_t));
#endif
			status = LmStreamGemmDense<Format>(weight,LmScaleTensorNone(),source,destination,(float *)0,rows,input_dimension,output_dimension,0u,0u,multiprocessors,stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return(status);
			LM_LAUNCH_REFUSE("k3 projection: the stream GEMM refused %u rows of a %u x %u projection; prefill waves with an input that is a multiple "
				"of %u run on it, so the output must be even and the operands 16-byte aligned\n",rows,output_dimension,input_dimension,
				K3_STREAM_INPUT_MULTIPLE);
		}
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	LmWeightPrefetchNote(stream, weight, (uint64_t)output_dimension * input_dimension * sizeof(uint16_t));
#endif
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = destination;
	gemm.accumulate_bf16 = accumulate;
	status = LmGemmLaunchTileK<Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,source,weight,rows,rows,1u,1u,
		input_dimension,output_dimension,multiprocessors,false,stream);
	if ( status == LM_LAUNCH_ERR_SHAPE || status == LM_LAUNCH_ERR_TILE )
		LM_LAUNCH_REFUSE("k3 projection: the tiled GEMM refused %u rows of a %u x %u projection; accumulating prefill waves run on it, so the input "
			"must be a multiple of %u\n",rows,output_dimension,input_dimension,K3_TILED_INPUT_MULTIPLE);
	return(status);
}


template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	return(K3Project<Format>(b,source,weight,weight_scale,destination,(uint16_t *)0,rows,input_dimension,output_dimension,multiprocessors,stream));
}

static inline uint32_t K3VecGrid(uint32_t dimension, const void *a, const void *b, const void *c = 0)
{
	if ( (dimension & 7u) == 0u && ((((uintptr_t)a | (uintptr_t)b | (uintptr_t)c) & 15u) == 0u) )
		return(LmVecBlocks<K3_LAYER_THREADS>(dimension));
	return((dimension + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS);
}

static void K3AttnResRows(const K3LayerBuffers *b, const void *score_weight, uint32_t sources, uint32_t rows, uint32_t first, uint32_t count, uint16_t *output, SparkGpuStream stream)
{
	const uint64_t offset = (uint64_t)first * K3_HIDDEN;
	uint32_t done, part;
	if ( sources > K3_ATTNRES_MAX_SOURCES )
		sources = K3_ATTNRES_MAX_SOURCES;
	if ( count >= 32u )
	{
		LM_LAUNCH((LmAttnResKernel<K3_LAYER_THREADS,K3_ATTNRES_MAX_SOURCES>),
			count, K3_LAYER_THREADS, 0, stream,
			b->attnres_bank_bf16 + offset,b->attnres_partial_bf16 + offset,
			(const uint16_t *)score_weight,output,sources,rows,K3_HIDDEN,
			K3_RMS_EPSILON);
		return;
	}
	LM_LAUNCH((LmAttnResScoreKernel<K3_LAYER_THREADS>), dim3(count,sources), K3_LAYER_THREADS, 0, stream,
		b->attnres_bank_bf16 + offset,b->attnres_partial_bf16 + offset,(const uint16_t *)score_weight,b->attnres_score_f32 + (uint64_t)first * K3_ATTNRES_MAX_SOURCES,
		sources,rows,K3_HIDDEN,K3_ATTNRES_MAX_SOURCES,K3_RMS_EPSILON);
	for (done = 0u; done < count; done += part)
	{
		const uint64_t row = (uint64_t)first + done;
		const uint16_t *bank_row = b->attnres_bank_bf16 + row * K3_HIDDEN;
		const uint16_t *partial_row = b->attnres_partial_bf16 + row * K3_HIDDEN;
		uint16_t *out_row = output + (uint64_t)done * K3_HIDDEN;
		part = LmLaunchGridRows(count - done);
		LM_LAUNCH((LmAttnResMixKernel<K3_LAYER_THREADS,K3_ATTNRES_MAX_SOURCES>),
			dim3(K3VecGrid(K3_HIDDEN,bank_row,partial_row,out_row),part), K3_LAYER_THREADS, 0, stream,
			bank_row,partial_row,b->attnres_score_f32 + row * K3_ATTNRES_MAX_SOURCES,
			out_row,sources,rows,K3_HIDDEN,K3_ATTNRES_MAX_SOURCES);
	}
}

static void K3AttnRes(const K3LayerBuffers *b, const void *score_weight, uint32_t sources, uint32_t rows, SparkGpuStream stream)
{
	if ( sources > K3_ATTNRES_MAX_SOURCES )
		sources = K3_ATTNRES_MAX_SOURCES;
	if ( b->attnres_score_f32 != 0 && rows < 32u )
	{
		K3AttnResRows(b,score_weight,sources,rows,0u,rows,b->hidden_bf16,stream);
		return;
	}
	LM_LAUNCH((LmAttnResKernel<K3_LAYER_THREADS,K3_ATTNRES_MAX_SOURCES>),
		rows, K3_LAYER_THREADS, 0, stream,
		b->attnres_bank_bf16,b->attnres_partial_bf16,
		(const uint16_t *)score_weight,b->hidden_bf16,sources,rows,K3_HIDDEN,
		K3_RMS_EPSILON);
}

static void K3AttnResWithNorm(K3LayerBuffers *b, const void *score_weight, const void *norm_weight, uint32_t sources, uint32_t rows, SparkGpuStream stream)
{
	const uint16_t *addend = b->pending_partial_add;
	b->pending_partial_add = 0;
	b->normed_ready = 0u;
	if ( sources > K3_ATTNRES_MAX_SOURCES )
		sources = K3_ATTNRES_MAX_SOURCES;
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( K3_LAYER_THREADS == 256u && LmIsAligned16(b->attnres_bank_bf16) && LmIsAligned16(b->attnres_partial_bf16) &&
		LmIsAligned16(score_weight) && LmIsAligned16(b->hidden_bf16) &&
		(addend == 0 || LmIsAligned16(addend)) &&
		(norm_weight == 0 || (LmIsAligned16(norm_weight) && LmIsAligned16(b->normed_bf16))) )
	{
		const uint16_t *nw = (const uint16_t *)norm_weight;
		if ( nw != 0 )
			LmWeightPrefetchNote(stream, nw, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
		if ( rows <= 64u && b->attnres_score_f32 != 0 )
		{
			LM_LAUNCH_DEPENDENT((K3AttnResScoreAddKernel), dim3(rows, sources), 256u, 0, stream,
				b->attnres_bank_bf16, b->attnres_partial_bf16, addend, (const uint16_t *)score_weight,
				b->attnres_score_f32, sources, rows, K3_RMS_EPSILON);
			LM_LAUNCH_DEPENDENT((K3AttnResMixNormKernel), rows, 256u, 0, stream,
				b->attnres_bank_bf16, b->attnres_partial_bf16, b->attnres_score_f32, nw,
				b->hidden_bf16, b->normed_bf16, sources, rows, K3_RMS_EPSILON);
			b->normed_ready = nw != 0 ? 1u : 0u;
			return;
		}
		const uint32_t blocks = rows < K3_ATTNRES_PERSISTENT_BLOCKS ? rows : K3_ATTNRES_PERSISTENT_BLOCKS;
		LM_LAUNCH_DEPENDENT((K3PartialAddAttnResNormKernel), blocks, 256u, 0, stream,
			b->attnres_bank_bf16, b->attnres_partial_bf16, addend, (const uint16_t *)score_weight, nw,
			b->hidden_bf16, b->normed_bf16, sources, rows, K3_RMS_EPSILON);
		b->normed_ready = nw != 0 ? 1u : 0u;
		return;
	}
#endif
	if ( addend != 0 )
		LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),
			dim3(K3VecGrid(K3_HIDDEN,b->attnres_partial_bf16,addend),LmLaunchGridRows(rows)),
			K3_LAYER_THREADS, 0, stream, b->attnres_partial_bf16,addend,b->attnres_partial_bf16,rows,K3_HIDDEN);
	K3AttnRes(b, score_weight, sources, rows, stream);
}

inline void K3PartialSet(const K3LayerBuffers *b, const uint16_t *value, uint32_t rows, SparkGpuStream stream)
{
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3(K3VecGrid(K3_HIDDEN,value,b->attnres_partial_bf16),LmLaunchGridRows(rows)),
		K3_LAYER_THREADS, 0, stream,
		value,b->attnres_partial_bf16,rows,K3_HIDDEN);
}

inline void K3PartialAdd(const K3LayerBuffers *b, const uint16_t *value, uint32_t rows, SparkGpuStream stream)
{
	LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),
		dim3(K3VecGrid(K3_HIDDEN,b->attnres_partial_bf16,value),LmLaunchGridRows(rows)),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,value,b->attnres_partial_bf16,rows,K3_HIDDEN);
}

static void K3BankStore(const K3LayerBuffers *b, uint32_t slot, uint32_t rows, SparkGpuStream stream)
{
	uint16_t *dst = b->attnres_bank_bf16 + ((uint64_t)slot * rows * K3_HIDDEN);
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3(K3VecGrid(K3_HIDDEN,b->attnres_partial_bf16,dst),LmLaunchGridRows(rows)),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,dst,rows,K3_HIDDEN);
}

static void K3PartialRows(const K3LayerBuffers *b, const uint16_t *value, uint32_t first, uint32_t count, uint32_t set, SparkGpuStream stream)
{
	uint16_t *partial = b->attnres_partial_bf16 + (uint64_t)first * K3_HIDDEN;
	const dim3 grid(K3VecGrid(K3_HIDDEN,value,partial),LmLaunchGridRows(count));
	if ( set != 0u )
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), grid, K3_LAYER_THREADS, 0, stream,
			value,partial,count,K3_HIDDEN);
	else
		LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>), grid, K3_LAYER_THREADS, 0, stream,
			partial,value,partial,count,K3_HIDDEN);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void K3CopyAligned4To16(const uint16_t *__restrict__ src, uint16_t *__restrict__ dst, uint16_t *__restrict__ dst2, uint32_t elements)
{
	const uint32_t *s32 = (const uint32_t *)src;
	uint4 *d4 = (uint4 *)dst;
	uint4 *d4b = (uint4 *)dst2;
	const uint32_t vecs = elements >> 3u;
	for (uint32_t i = threadIdx.x; i < vecs; i += THREADS)
	{
		const uint32_t b = i << 2u;
		const uint4 v = make_uint4(s32[b], s32[b + 1u], s32[b + 2u], s32[b + 3u]);
		d4[i] = v;
		if ( d4b != 0 )
			d4b[i] = v;
	}
}

template<uint32_t THREADS>
static __device__ __forceinline__ void K3CopySliceDualBf16(const uint16_t *__restrict__ src, uint16_t *__restrict__ dst, uint16_t *__restrict__ dst2, uint32_t elements)
{
	for (uint32_t i = threadIdx.x; i < elements; i += THREADS)
	{
		const uint16_t v = src[i];
		dst[i] = v;
		if ( dst2 != 0 )
			dst2[i] = v;
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void K3SplitFusedProjectionsKernel(const uint16_t *__restrict__ qkvb_bf16, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint16_t *__restrict__ beta_bf16, uint16_t *__restrict__ replay_q_bf16, uint16_t *__restrict__ replay_k_bf16, uint16_t *__restrict__ replay_v_bf16, uint32_t rows, uint32_t qk_dim, uint32_t v_dim, uint32_t heads, uint32_t fused_rows)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row >= rows ) { LmDependentRelease(); return; }
	const uint16_t *src = qkvb_bf16 + (uint64_t)row * fused_rows;
	uint16_t *q_dst = query_bf16 + (uint64_t)row * qk_dim;
	uint16_t *k_dst = key_bf16 + (uint64_t)row * qk_dim;
	uint16_t *v_dst = value_bf16 + (uint64_t)row * v_dim;
	uint16_t *rq_dst = replay_q_bf16 != 0 ? replay_q_bf16 + (uint64_t)row * qk_dim : (uint16_t *)0;
	uint16_t *rk_dst = replay_k_bf16 != 0 ? replay_k_bf16 + (uint64_t)row * qk_dim : (uint16_t *)0;
	uint16_t *rv_dst = replay_v_bf16 != 0 ? replay_v_bf16 + (uint64_t)row * v_dim : (uint16_t *)0;
	const bool replay_aligned = (rq_dst == 0 || LmIsAligned16(rq_dst)) && (rk_dst == 0 || LmIsAligned16(rk_dst)) && (rv_dst == 0 || LmIsAligned16(rv_dst));
	if ( ((qk_dim & 7u) == 0u) && ((v_dim & 7u) == 0u) && ((fused_rows & 1u) == 0u) &&
		((((uintptr_t)src) & 3u) == 0u) && LmIsAligned16(q_dst) && LmIsAligned16(k_dst) && LmIsAligned16(v_dst) && replay_aligned )
	{
		K3CopyAligned4To16<THREADS>(src, q_dst, rq_dst, qk_dim);
		K3CopyAligned4To16<THREADS>(src + qk_dim, k_dst, rk_dst, qk_dim);
		K3CopyAligned4To16<THREADS>(src + 2u * qk_dim, v_dst, rv_dst, v_dim);
	}
	else
	{
		K3CopySliceDualBf16<THREADS>(src, q_dst, rq_dst, qk_dim);
		K3CopySliceDualBf16<THREADS>(src + qk_dim, k_dst, rk_dst, qk_dim);
		K3CopySliceDualBf16<THREADS>(src + 2u * qk_dim, v_dst, rv_dst, v_dim);
	}
	LmCopySliceBf16<THREADS>(src + 2u * qk_dim + v_dim, beta_bf16 + (uint64_t)row * heads, heads);
	LmDependentRelease();
}

#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
__global__ __launch_bounds__(128u, 4)
void K3KdaPrepFinishVecKernel(uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16,
	const uint16_t *__restrict__ decay_logit_bf16, const float *__restrict__ channel_bias,
	const float *__restrict__ head_log_scale, const uint16_t *__restrict__ beta_logit_bf16,
	float *__restrict__ retention, float *__restrict__ write_gate,
	uint32_t heads, float minimum_log_decay, float epsilon, uint32_t rows)
{
	static_assert(K3_KDA_KEY_DIM == 128u, "one warp handles four elements per lane across 128-wide KDA key");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * 4u + warp;
	LmDependentWait();
	if ( task >= rows * heads )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t head = task % heads, off = lane * 4u;
	const uint64_t base = (uint64_t)task * K3_KDA_KEY_DIM;
	const uint2 qv = *(const uint2 *)(query_bf16 + base + off);
	const uint2 kv = *(const uint2 *)(key_bf16 + base + off);
	const float q0 = LmBf16ToFloat((uint16_t)(qv.x & 0xffffu)), q1 = LmBf16ToFloat((uint16_t)(qv.x >> 16u));
	const float q2 = LmBf16ToFloat((uint16_t)(qv.y & 0xffffu)), q3 = LmBf16ToFloat((uint16_t)(qv.y >> 16u));
	const float k0 = LmBf16ToFloat((uint16_t)(kv.x & 0xffffu)), k1 = LmBf16ToFloat((uint16_t)(kv.x >> 16u));
	const float k2 = LmBf16ToFloat((uint16_t)(kv.y & 0xffffu)), k3 = LmBf16ToFloat((uint16_t)(kv.y >> 16u));
	float q_sum = fmaf(q0, q0, fmaf(q1, q1, fmaf(q2, q2, q3 * q3)));
	float k_sum = fmaf(k0, k0, fmaf(k1, k1, fmaf(k2, k2, k3 * k3)));
	#pragma unroll
	for ( uint32_t s = 16u; s > 0u; s >>= 1u )
	{
		q_sum += __shfl_xor_sync(0xffffffffu, q_sum, s);
		k_sum += __shfl_xor_sync(0xffffffffu, k_sum, s);
	}
	const float q_inv = rsqrtf(q_sum + epsilon), k_inv = rsqrtf(k_sum + epsilon);
	*(uint2 *)(query_bf16 + base + off) = make_uint2(LmPackBf16Pair(q0 * q_inv, q1 * q_inv), LmPackBf16Pair(q2 * q_inv, q3 * q_inv));
	*(uint2 *)(key_bf16 + base + off) = make_uint2(LmPackBf16Pair(k0 * k_inv, k1 * k_inv), LmPackBf16Pair(k2 * k_inv, k3 * k_inv));
	const uint2 dv = __ldg((const uint2 *)(decay_logit_bf16 + base + off));
	const float4 cb = __ldg((const float4 *)(channel_bias + (uint64_t)head * K3_KDA_KEY_DIM + off));
	const float hls = __ldg(head_log_scale + head);
	*(float4 *)(retention + base + off) = make_float4(
		LmBoundedDecay(LmBf16ToFloat((uint16_t)(dv.x & 0xffffu)), cb.x, hls, minimum_log_decay),
		LmBoundedDecay(LmBf16ToFloat((uint16_t)(dv.x >> 16u)), cb.y, hls, minimum_log_decay),
		LmBoundedDecay(LmBf16ToFloat((uint16_t)(dv.y & 0xffffu)), cb.z, hls, minimum_log_decay),
		LmBoundedDecay(LmBf16ToFloat((uint16_t)(dv.y >> 16u)), cb.w, hls, minimum_log_decay));
	if ( lane == 0u )
		write_gate[task] = SparkMathSigmoidFast(LmBf16ToFloat(__ldg(beta_logit_bf16 + task)));
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
void K3KdaNormGateVecKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16,
	const float *__restrict__ norm_weight, uint32_t heads, float epsilon, uint32_t rows)
{
	static_assert(K3_KDA_VALUE_DIM == 128u, "one warp handles four 32-strided elements across 128-wide KDA value");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * 4u + warp;
	LmDependentWait();
	if ( task >= rows * heads )
	{
		LmDependentRelease();
		return;
	}
	const uint64_t base = (uint64_t)task * K3_KDA_VALUE_DIM;
	const float v0 = LmBf16ToFloat(output_bf16[base + lane]);
	const float v1 = LmBf16ToFloat(output_bf16[base + 32u + lane]);
	const float v2 = LmBf16ToFloat(output_bf16[base + 64u + lane]);
	const float v3 = LmBf16ToFloat(output_bf16[base + 96u + lane]);
	float s0 = v0 * v0, s1 = v1 * v1, s2 = v2 * v2, s3 = v3 * v3;
	#pragma unroll
	for ( uint32_t s = 16u; s > 0u; s >>= 1u )
	{
		s0 += __shfl_down_sync(0xffffffffu, s0, s);
		s1 += __shfl_down_sync(0xffffffffu, s1, s);
		s2 += __shfl_down_sync(0xffffffffu, s2, s);
		s3 += __shfl_down_sync(0xffffffffu, s3, s);
	}
	const float total = __shfl_sync(0xffffffffu, (s0 + s2) + (s1 + s3), 0u);
	const float scale = SparkMathRmsScale(total, (float)K3_KDA_VALUE_DIM, epsilon);
	const float n0 = LmBf16ToFloat(LmFloatToBf16(v0 * scale * __ldg(norm_weight + lane)));
	const float n1 = LmBf16ToFloat(LmFloatToBf16(v1 * scale * __ldg(norm_weight + 32u + lane)));
	const float n2 = LmBf16ToFloat(LmFloatToBf16(v2 * scale * __ldg(norm_weight + 64u + lane)));
	const float n3 = LmBf16ToFloat(LmFloatToBf16(v3 * scale * __ldg(norm_weight + 96u + lane)));
	output_bf16[base + lane] = LmFloatToBf16(n0 * SparkMathSigmoidFast(LmBf16ToFloat(__ldg(gate_bf16 + base + lane))));
	output_bf16[base + 32u + lane] = LmFloatToBf16(n1 * SparkMathSigmoidFast(LmBf16ToFloat(__ldg(gate_bf16 + base + 32u + lane))));
	output_bf16[base + 64u + lane] = LmFloatToBf16(n2 * SparkMathSigmoidFast(LmBf16ToFloat(__ldg(gate_bf16 + base + 64u + lane))));
	output_bf16[base + 96u + lane] = LmFloatToBf16(n3 * SparkMathSigmoidFast(LmBf16ToFloat(__ldg(gate_bf16 + base + 96u + lane))));
	LmDependentRelease();
}
#endif

template<uint32_t THREADS, uint32_t KEY_DIM>
__global__ __launch_bounds__(THREADS, 1)
void K3KdaPrepFinishKernel(uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ decay_logit_bf16, const float *__restrict__ channel_bias, const float *__restrict__ head_log_scale, const uint16_t *__restrict__ beta_logit_bf16, float *__restrict__ retention, float *__restrict__ write_gate, uint32_t heads, float minimum_log_decay, float epsilon, uint32_t rows)
{
	__shared__ float reduction[2u * LmBlockWarps<THREADS>()];
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	LmDependentWait();
	if ( row >= rows || head >= heads ) { LmDependentRelease(); return; }
	const uint64_t base = (((uint64_t)row * heads) + head) * KEY_DIM;
	float q_sum = 0.0f, k_sum = 0.0f, q0 = 0.0f, k0 = 0.0f;
	for (uint32_t idx = threadIdx.x; idx < KEY_DIM; idx += THREADS) { q0 = LmBf16ToFloat(query_bf16[base + idx]); k0 = LmBf16ToFloat(key_bf16[base + idx]); q_sum += q0 * q0; k_sum += k0 * k0; }
	LmBlockSum2Once<THREADS>(&q_sum, &k_sum, reduction);
	const float q_inv = rsqrtf(q_sum + epsilon), k_inv = rsqrtf(k_sum + epsilon);
	for (uint32_t idx = threadIdx.x; idx < KEY_DIM; idx += THREADS)
	{
		query_bf16[base + idx] = LmFloatToBf16((THREADS == KEY_DIM ? q0 : LmBf16ToFloat(query_bf16[base + idx])) * q_inv);
		key_bf16[base + idx] = LmFloatToBf16((THREADS == KEY_DIM ? k0 : LmBf16ToFloat(key_bf16[base + idx])) * k_inv);
		retention[base + idx] = LmBoundedDecay(LmBf16ToFloat(decay_logit_bf16[base + idx]), channel_bias[(head * KEY_DIM) + idx], head_log_scale[head], minimum_log_decay);
	}
	if ( threadIdx.x == 0u ) write_gate[(uint64_t)row * heads + head] = SparkMathSigmoidFast(LmBf16ToFloat(beta_logit_bf16[(uint64_t)row * heads + head]));
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void K3KdaNormGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, const float *__restrict__ norm_weight, uint32_t heads, float epsilon, uint32_t rows)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	LmDependentWait();
	if ( row >= rows || head >= heads ) { LmDependentRelease(); return; }
	const uint64_t base = (((uint64_t)row * heads) + head) * VALUE_DIM;
	float sum = 0.0f, v0 = 0.0f;
	for (uint32_t idx = threadIdx.x; idx < VALUE_DIM; idx += THREADS) { v0 = LmBf16ToFloat(output_bf16[base + idx]); sum += v0 * v0; }
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(sum, reduction), (float)VALUE_DIM, epsilon);
	for (uint32_t idx = threadIdx.x; idx < VALUE_DIM; idx += THREADS)
	{
		const float v = THREADS == VALUE_DIM ? v0 : LmBf16ToFloat(output_bf16[base + idx]);
		const float normed = LmBf16ToFloat(LmFloatToBf16(v * scale * norm_weight[idx]));
		output_bf16[base + idx] = LmFloatToBf16(normed * SparkMathSigmoidFast(LmBf16ToFloat(gate_bf16[base + idx])));
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t DIM>
__global__ __launch_bounds__(THREADS, 4)
void K3RmsNormSliceKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t col_offset, uint32_t slice_width, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * DIM;
	float total = 0.0f;
	LmDependentWait();
	if ( ((DIM & 7u) == 0u) && LmIsAligned16(input_bf16) )
	{
		for ( uint32_t offset = threadIdx.x * 8u; offset < DIM; offset += THREADS * 8u )
			total = LmSumSqBf16x8(__ldg((const uint4 *)(input_bf16 + base + offset)), total);
	}
	else
	{
		for ( uint32_t index = threadIdx.x; index < DIM; index += THREADS )
		{
			const float v = LmBf16ToFloat(input_bf16[base + index]);
			total = fmaf(v, v, total);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)DIM, epsilon);
	const uint64_t out_base = (uint64_t)blockIdx.x * slice_width;
	if ( ((col_offset & 7u) == 0u) && ((slice_width & 7u) == 0u) &&
		LmIsAligned16(input_bf16) && LmIsAligned16(weight_bf16) && LmIsAligned16(output_bf16) )
	{
		for ( uint32_t offset = threadIdx.x * 8u; offset < slice_width; offset += THREADS * 8u )
		{
			const uint32_t col = col_offset + offset;
			float vals[8];
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(input_bf16 + base + col)), vals);
			LmRmsStore8(vals, weight_bf16, col, scale, output_bf16 + out_base + offset, LmRmsGainOp<true>{});
		}
	}
	else
	{
		for ( uint32_t index = threadIdx.x; index < slice_width; index += THREADS )
		{
			const uint32_t col = col_offset + index;
			output_bf16[out_base + index] = LmFloatToBf16(SparkMathNormGainRounded(LmBf16ToFloat(input_bf16[base + col]) * scale, LmBf16ToFloat(weight_bf16[col])));
		}
	}
	LmDependentRelease();
}

template<class Geometry, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 4)
void K3RmsNormKvShardStoreKernel(LmKvShardView view, const uint16_t *__restrict__ rows_bf16, const uint16_t *__restrict__ norm_weight, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count, float epsilon, uint16_t *__restrict__ kv_all = 0, uint32_t total_ctx = 0u)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row >= row_count )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t pos = position_of_row[row];
	uint16_t *dst_all = kv_all != 0 && pos < total_ctx ? kv_all + (uint64_t)pos * K3_MLA_KV_A_DIM : (uint16_t *)0;
	uint16_t *slot = SparkKvShardOwns(view.shard, pos) != 0u
		? (uint16_t *)LmKvShardSlotRequired<Geometry>(view, sequence_of_row[row], pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
	if ( slot == 0 && dst_all == 0 )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *src = rows_bf16 + (uint64_t)row * K3_MLA_KV_A_DIM;
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if constexpr ( THREADS == 64u && K3_KV_LORA_RANK == 512u && K3_QK_UNROTATED_DIM == 64u )
	{
		if ( LmIsAligned16(src) && LmIsAligned16(norm_weight) && (slot == 0 || LmIsAligned16(slot)) && (dst_all == 0 || LmIsAligned16(dst_all)) )
		{
			const uint32_t off = threadIdx.x * 8u;
			const uint4 raw = __ldg((const uint4 *)(src + off));
			const float total = LmSumSqBf16x8(raw, 0.0f);
			const float scale = SparkMathRmsScale(LmBlockSumOnce<64u>(total, reduction), (float)K3_KV_LORA_RANK, epsilon);
			float vals[8], w[8], out_f[8];
			LmLoadBf16x8AsFloat(raw, vals);
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(norm_weight + off)), w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out_f[k] = SparkMathNormGainRounded(vals[k] * scale, w[k]);
			const uint4 out_v = LmPackUint4FloatToBf16(out_f);
			if ( slot != 0 )
				*(uint4 *)(slot + off) = out_v;
			if ( dst_all != 0 )
				*(uint4 *)(dst_all + off) = out_v;
			if ( threadIdx.x < 8u )
			{
				const uint4 rope = __ldg((const uint4 *)(src + K3_KV_LORA_RANK + off));
				if ( slot != 0 )
					*(uint4 *)(slot + K3_KV_LORA_RANK + off) = rope;
				if ( dst_all != 0 )
					*(uint4 *)(dst_all + K3_KV_LORA_RANK + off) = rope;
			}
			LmDependentRelease();
			return;
		}
	}
#endif
	float total = 0.0f;
	for (uint32_t index = threadIdx.x; index < K3_KV_LORA_RANK; index += THREADS)
	{
		const float v = LmBf16ToFloat(src[index]);
		total += v * v;
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)K3_KV_LORA_RANK, epsilon);
	for (uint32_t index = threadIdx.x; index < K3_KV_LORA_RANK; index += THREADS)
	{
		const uint16_t out = LmFloatToBf16(SparkMathNormGainRounded(LmBf16ToFloat(src[index]) * scale, LmBf16ToFloat(norm_weight[index])));
		if ( slot != 0 )
			slot[index] = out;
		if ( dst_all != 0 )
			dst_all[index] = out;
	}
	for (uint32_t index = threadIdx.x; index < K3_QK_UNROTATED_DIM; index += THREADS)
	{
		const uint16_t out = src[K3_KV_LORA_RANK + index];
		if ( slot != 0 )
			slot[K3_KV_LORA_RANK + index] = out;
		if ( dst_all != 0 )
			dst_all[K3_KV_LORA_RANK + index] = out;
	}
	LmDependentRelease();
}

template<class Geometry>
__global__ __launch_bounds__(32u, 1)
void K3LayerNormKvShardStoreKernel(LmKvShardView view, const uint16_t *__restrict__ rows_bf16, const uint16_t *__restrict__ weight, const uint16_t *__restrict__ bias, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count, float epsilon)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row >= row_count || SparkKvShardOwns(view.shard, position_of_row[row]) == 0u )
	{
		LmDependentRelease();
		return;
	}
	uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<Geometry>(view, sequence_of_row[row], position_of_row[row], row, LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *src = rows_bf16 + (uint64_t)row * K3_INDEX_DIM;
	const uint32_t lane = threadIdx.x, offset = lane * 8u;
	float v[8], total = 0.0f, total_sq = 0.0f;
	if ( offset < K3_INDEX_DIM )
	{
		LmLoadBf16x8AsFloat(*(const uint4 *)(src + offset), v);
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; ++k) { total += v[k]; total_sq += v[k] * v[k]; }
	}
	float mean, inverse;
	LmLayerNormStats<32u>(total, total_sq, (float *)0, K3_INDEX_DIM, epsilon, &mean, &inverse);
	if ( offset < K3_INDEX_DIM )
		LmLayerNormStore8(v, weight, bias, offset, mean, inverse, slot + offset);
	LmDependentRelease();
}

template<class Format>
static int32_t K3LayerKdaProject(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream, uint32_t decay = 1u)
{
	int32_t status;
	uint32_t state_slot_bytes;
	const uint32_t rank_heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	const uint32_t rank_qk = rank_heads * K3_KDA_KEY_DIM;
	const uint32_t rank_v = rank_heads * K3_KDA_VALUE_DIM;
	const uint32_t head_base = b->tp_rank * rank_heads;
	if ( b->kda_state_bf16 != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( head_base + rank_heads > K3_KDA_HEADS )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	state_slot_bytes = (uint32_t)K3_KDA_RANK_STATE_SLOT_BYTES(rank_heads, 0u);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( rows > LM_SKINNY_ROWS_WIDE && b->row_invariant == 0u &&
		LmIsAligned16(b->normed_bf16) && LmIsAligned16(b->kda_qkv_beta_weight) && LmIsAligned16(b->kda_gate_weight) &&
		(decay == 0u || LmIsAligned16(b->kda_decay_down_weight)) &&
		(rank_qk & 1u) == 0u && (rank_v & 1u) == 0u && (rank_heads & 1u) == 0u )
	{
		const uint32_t qkvb_rows = K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS);
		const uint32_t gate_rows = K3_RANK_DIM(b,kda_gate_rows,K3_KDA_V_DIM);
		const uint32_t decay_rows = decay != 0u ? K3_KDA_KEY_DIM : 0u;
		K3KdaProjMmaArgs mma = {};
		mma.w_qkvb = (const uint8_t *)b->kda_qkv_beta_weight;
		mma.w_gate = (const uint8_t *)b->kda_gate_weight;
		mma.w_decay = (const uint8_t *)b->kda_decay_down_weight;
		mma.activation = b->normed_bf16;
		mma.q = b->query_bf16;
		mma.k = b->key_bf16;
		mma.v = b->value_bf16;
		mma.beta = (uint16_t *)b->kda_beta_logit;
		mma.gate = b->gate_bf16;
		mma.latent = b->latent_bf16;
		mma.rq = b->replay_conv_q;
		mma.rk = b->replay_conv_q != 0 ? b->replay_conv_k : (uint16_t *)0;
		mma.rv = b->replay_conv_q != 0 ? b->replay_conv_v : (uint16_t *)0;
		mma.rows = rows;
		mma.qkvb_n = qkvb_rows;
		mma.gate_n = gate_rows;
		mma.decay_n = decay_rows;
		mma.rank_qk = rank_qk;
		mma.rank_v = rank_v;
		mma.rank_heads = rank_heads;
		(void)state_slot_bytes;
		if ( rows > 64u )
		{
			mma.tiles_qkvb = (qkvb_rows + 127u) / 128u;
			mma.tiles_gate = (gate_rows + 127u) / 128u;
			return(K3KdaProjPrefillMmaLaunch(mma, stream));
		}
		mma.tiles_qkvb = (qkvb_rows + 31u) / 32u;
		mma.tiles_gate = (gate_rows + 31u) / 32u;
		const uint32_t total_tiles = mma.tiles_qkvb + mma.tiles_gate + (decay_rows + 31u) / 32u;
		LmWeightPrefetchNote(stream, b->kda_qkv_beta_weight, (uint64_t)qkvb_rows * K3_HIDDEN * sizeof(uint16_t));
		LmWeightPrefetchNote(stream, b->kda_gate_weight, (uint64_t)gate_rows * K3_HIDDEN * sizeof(uint16_t));
		if ( decay != 0u )
			LmWeightPrefetchNote(stream, b->kda_decay_down_weight, (uint64_t)decay_rows * K3_HIDDEN * sizeof(uint16_t));
		if ( rows <= 32u )
			LM_LAUNCH_DEPENDENT((K3KdaProjMmaKernel<32u>), total_tiles, 128u, 0u, stream, mma);
		else
			LM_LAUNCH_DEPENDENT((K3KdaProjMmaKernel<64u>), total_tiles, 128u, 0u, stream, mma);
		return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
	}
#endif
#if !defined(LM_HOST_CUDA_PRELUDE)
	if ( rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u )
	{
		const uint32_t qkvb_rows = K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS), gate_rows = K3_RANK_DIM(b,kda_gate_rows,K3_KDA_V_DIM);
		LmSkinnyDenseTarget targets[3] = {
			{b->kda_qkv_beta_weight, b->fused_qkvb_bf16, (float *)0, qkvb_rows, qkvb_rows, 0u, LmScaleTensorNone()},
			{b->kda_gate_weight, b->gate_bf16, (float *)0, gate_rows, gate_rows, 0u, LmScaleTensorNone()},
			{b->kda_decay_down_weight, b->latent_bf16, (float *)0, K3_KDA_KEY_DIM, K3_KDA_KEY_DIM, 0u, LmScaleTensorNone()}};
		status = LmSkinnyDenseMulti<LmBf16Format>(targets,decay != 0u ? 3u : 2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("k3 kda projection: the multi-target skinny kernel refused %u rows; waves of up to %u rows and row-invariant waves "
				"run on it, so the weights and activation need 16-byte alignment\n",rows,LM_SKINNY_ROWS_WIDE);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	else
#endif
	{
		status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_qkv_beta_weight,0,
			b->fused_qkvb_bf16,rows,K3_HIDDEN,
			K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS),multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( decay != 0u )
			status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_decay_down_weight,0,
				b->latent_bf16,rows,K3_HIDDEN,K3_KDA_KEY_DIM,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_gate_weight,0,
			b->gate_bf16,rows,K3_HIDDEN,
			K3_RANK_DIM(b,kda_gate_rows,K3_KDA_V_DIM),multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	LM_LAUNCH_DEPENDENT((K3SplitFusedProjectionsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->fused_qkvb_bf16,
		b->query_bf16,b->key_bf16,b->value_bf16,(uint16_t *)b->kda_beta_logit,
		b->replay_conv_q,b->replay_conv_q != 0 ? b->replay_conv_k : (uint16_t *)0,b->replay_conv_q != 0 ? b->replay_conv_v : (uint16_t *)0,
		rows,rank_qk,rank_v,rank_heads,
		K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS));
	(void)state_slot_bytes;
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerKdaPrepare(const K3LayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	const uint32_t rank_heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	const uint32_t rank_qk = rank_heads * K3_KDA_KEY_DIM, rank_v = rank_heads * K3_KDA_VALUE_DIM;
	const uint32_t head_base = b->tp_rank * rank_heads;
	if ( b->kda_state_bf16 != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( head_base + rank_heads > K3_KDA_HEADS )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->parent_rows != 0 && (b->replay_conv_q == 0 || b->replay_conv_k == 0 || b->replay_conv_v == 0) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	const LmCausalConvStreams<float> conv = {
		{b->kda_q_window, b->kda_k_window, b->kda_v_window},
		{b->parent_rows != 0 ? b->replay_conv_q : b->query_bf16, b->parent_rows != 0 ? b->replay_conv_k : b->key_bf16, b->parent_rows != 0 ? b->replay_conv_v : b->value_bf16},
		{b->kda_q_conv_weight, b->kda_k_conv_weight, b->kda_v_conv_weight},
		{b->query_bf16, b->key_bf16, b->value_bf16},
		{rank_qk, rank_qk, rank_v}};
	if ( LmCausalConvStreamsLaunch<K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>(conv,b->kda_state_index,b->sequence_row_begin,0,sequences,commit,b->sequence_row_indices,rows,stream,b->parent_rows) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	float *retention = b->replay_retention != 0 ? b->replay_retention : b->kda_retention;
	float *write_gate = b->replay_write_gate != 0 ? b->replay_write_gate : b->kda_write_gate_out;
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->kda_decay_up_weight,0,
		b->decay_logit_bf16,rows,K3_KDA_KEY_DIM,
		K3_RANK_DIM(b,kda_decay_up_rows,K3_KDA_QK_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	const float *decay_bias = b->kda_decay_bias + ((uint64_t)head_base * K3_KDA_KEY_DIM);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( K3_KDA_KEY_DIM == 128u && LmIsAligned16(b->query_bf16) && LmIsAligned16(b->key_bf16) &&
		LmIsAligned16(b->decay_logit_bf16) && LmIsAligned16(decay_bias) && LmIsAligned16(retention) )
	{
		LM_LAUNCH_DEPENDENT((K3KdaPrepFinishVecKernel), (rows * rank_heads + 3u) / 4u, 128u, 0, stream,
			b->query_bf16,b->key_bf16,b->decay_logit_bf16,decay_bias,
			b->kda_head_log_scale + head_base,(const uint16_t *)b->kda_beta_logit,retention,write_gate,
			rank_heads,K3_KDA_GATE_LOWER_BOUND,K3_RMS_EPSILON,rows);
		return(LM_LAUNCH_OK);
	}
#endif
	{
		constexpr uint32_t prep_threads = K3_LAYER_THREADS < K3_KDA_KEY_DIM ? K3_LAYER_THREADS : K3_KDA_KEY_DIM;
		LM_LAUNCH_DEPENDENT((K3KdaPrepFinishKernel<prep_threads,K3_KDA_KEY_DIM>), dim3(rows,rank_heads), prep_threads, 0, stream,
			b->query_bf16,b->key_bf16,b->decay_logit_bf16,decay_bias,
			b->kda_head_log_scale + head_base,(const uint16_t *)b->kda_beta_logit,retention,write_gate,
			rank_heads,K3_KDA_GATE_LOWER_BOUND,K3_RMS_EPSILON,rows);
	}
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerKdaInputs(const K3LayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t multiprocessors, SparkGpuStream stream)
{
	const int32_t status = K3LayerKdaProject<Format>(b,rows,multiprocessors,stream);
	return(status == LM_LAUNCH_OK ? K3LayerKdaPrepare<Format>(b,rows,sequences,commit,multiprocessors,stream) : status);
}

static int32_t K3LayerKdaRecur(const K3LayerBuffers *b, const uint16_t *value_bf16, uint16_t *output_bf16, uint32_t rows, uint32_t sequences, uint32_t commit, SparkGpuStream stream)
{
	const uint32_t state_slot_bytes = (uint32_t)K3_KDA_RANK_STATE_SLOT_BYTES(K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS), 0u);
	const float *retention = b->replay_retention != 0 ? b->replay_retention : b->kda_retention;
	const float *write_gate = b->replay_write_gate != 0 ? b->replay_write_gate : b->kda_write_gate_out;
	if ( b->kda_state_bf16 != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( (b->state_loads != 0 || b->state_saves != 0) && (b->state_loads == 0 || b->state_saves == 0 || b->kda_snapshots == 0 || commit != 0u) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if constexpr ( LM_WARP_LANES == 32u )
	{
		if ( LmDeltaRuleLaunch<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM,float>(rows,b->kda_state_pool,state_slot_bytes,b->kda_state_index,b->sequence_row_begin,0,
				b->query_bf16,b->key_bf16,value_bf16,retention,write_gate,output_bf16,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),1u,sequences,commit,b->sequence_row_indices,stream,
				b->state_loads,b->state_saves,b->kda_snapshots) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
		LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM,float,K3_KDA_DELTA_COLUMNS>), dim3(sequences,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),K3_KDA_VALUE_DIM / K3_KDA_DELTA_COLUMNS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_DELTA_COLUMNS * sizeof(float)), stream,
			b->kda_state_pool,state_slot_bytes,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->key_bf16, value_bf16,retention,write_gate,output_bf16, K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),1u,sequences,commit,b->sequence_row_indices);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerKdaGate(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	const uint32_t rank_heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( K3_KDA_VALUE_DIM == 128u && LmIsAligned16(b->attention_out_bf16) && LmIsAligned16(b->gate_bf16) && LmIsAligned16(b->kda_out_norm_weight) )
	{
		LM_LAUNCH_DEPENDENT((K3KdaNormGateVecKernel), (rows * rank_heads + 3u) / 4u, 128u, 0, stream,
			b->attention_out_bf16,b->gate_bf16,b->kda_out_norm_weight,rank_heads,K3_RMS_EPSILON,rows);
		return(LM_LAUNCH_OK);
	}
#endif
	{
		constexpr uint32_t gate_threads = K3_LAYER_THREADS < K3_KDA_VALUE_DIM ? K3_LAYER_THREADS : K3_KDA_VALUE_DIM;
		LM_LAUNCH_DEPENDENT((K3KdaNormGateKernel<gate_threads,K3_KDA_VALUE_DIM>), dim3(rows,rank_heads), gate_threads, 0, stream,
			b->attention_out_bf16,b->gate_bf16,b->kda_out_norm_weight,rank_heads,K3_RMS_EPSILON,rows);
	}
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerKdaFinish(const K3LayerBuffers *b, uint32_t rows, uint16_t *destination, uint16_t *partial_accumulate, uint32_t multiprocessors, SparkGpuStream stream)
{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( b->tp_sharded != 0u && b->kda_out_scale == 0 &&
		K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS) == 6u && K3_RANK_DIM(b,kda_out_input,K3_KDA_V_DIM) == 6u * K3_KDA_VALUE_DIM &&
		(rows <= 12u || b->row_invariant != 0u) &&
		LmIsAligned16(b->kda_out_weight) && LmIsAligned16(destination) )
	{
		LmWeightPrefetchNote(stream, b->kda_out_weight, (uint64_t)K3_HIDDEN * 6u * K3_KDA_VALUE_DIM * sizeof(uint16_t));
		if ( rows == 1u )
			LM_LAUNCH_DEPENDENT((K3KdaNormGateOutKernel<1u>), dim3(K3_HIDDEN / 32u, 1u), 256u, 0u, stream,
				b->attention_out_bf16, b->gate_bf16, b->kda_out_norm_weight, (const uint16_t *)b->kda_out_weight,
				destination, rows, K3_RMS_EPSILON);
		else if ( rows <= 4u )
			LM_LAUNCH_DEPENDENT((K3KdaNormGateOutKernel<4u>), dim3(K3_HIDDEN / 32u, 1u), 256u, 0u, stream,
				b->attention_out_bf16, b->gate_bf16, b->kda_out_norm_weight, (const uint16_t *)b->kda_out_weight,
				destination, rows, K3_RMS_EPSILON);
		else if ( rows <= 8u )
			LM_LAUNCH_DEPENDENT((K3KdaNormGateOutKernel<8u>), dim3(K3_HIDDEN / 32u, 1u), 256u, 0u, stream,
				b->attention_out_bf16, b->gate_bf16, b->kda_out_norm_weight, (const uint16_t *)b->kda_out_weight,
				destination, rows, K3_RMS_EPSILON);
		else
			LM_LAUNCH_DEPENDENT((K3KdaNormGateOutKernel<12u>), dim3(K3_HIDDEN / 32u, (rows + 11u) / 12u), 256u, 0u, stream,
				b->attention_out_bf16, b->gate_bf16, b->kda_out_norm_weight, (const uint16_t *)b->kda_out_weight,
				destination, rows, K3_RMS_EPSILON);
		return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
	}
#endif
	const int32_t status = K3LayerKdaGate(b,rows,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3Project<LmBf16Format>(b,b->attention_out_bf16,b->kda_out_weight,b->kda_out_scale,
		destination,b->tp_sharded != 0u ? (uint16_t *)0 : partial_accumulate,
		rows,K3_RANK_DIM(b,kda_out_input,K3_KDA_V_DIM),K3_HIDDEN,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerKdaBody(const K3LayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *destination, uint16_t *partial_accumulate, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status = K3LayerKdaInputs<Format>(b,rows,sequences,commit,multiprocessors,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerKdaRecur(b,b->value_bf16,b->attention_out_bf16,rows,sequences,commit,stream);
	return(status == LM_LAUNCH_OK ? K3LayerKdaFinish<Format>(b,rows,destination,partial_accumulate,multiprocessors,stream) : status);
}

template<class Format>
static int32_t K3LayerKda(const K3LayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *partial_accumulate, uint32_t multiprocessors, SparkGpuStream stream)
{
	const uint32_t rank_heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	if ( b->kda_state_bf16 != 0u || b->tp_rank * rank_heads + rank_heads > K3_KDA_HEADS )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->normed_ready == 0u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->attn_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
#endif
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
			b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->attn_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	}
	return(K3LayerKdaBody<Format>(b,rows,sequences,commit,b->hidden_bf16,partial_accumulate,multiprocessors,stream));
}

#define K3_MLA_DOWN_SLICED(b, rows) \
	((b)->tp_sharded != 0u && (b)->kv_shard.degree > 1u && ((rows) <= 64u || (b)->row_invariant != 0u) && \
	(b)->mla_q_down_scale == 0 && (b)->mla_kv_a_scale == 0 && \
	(K3_Q_LORA_RANK % (b)->kv_shard.degree) == 0u && (K3_MLA_KV_A_DIM % (b)->kv_shard.degree) == 0u)

static int32_t K3LayerMlaDownSlice(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	const uint32_t degree = b->kv_shard.degree, query = K3_Q_LORA_RANK / degree, key = K3_MLA_KV_A_DIM / degree;
	const uint8_t *qw = (const uint8_t *)b->mla_q_down_weight + (uint64_t)b->tp_rank * query * K3_HIDDEN * sizeof(uint16_t);
	const uint8_t *kw = (const uint8_t *)b->mla_kv_a_weight + (uint64_t)b->tp_rank * key * K3_HIDDEN * sizeof(uint16_t);
	int32_t status;
	if ( b->normed_ready == 0u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->attn_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
#endif
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
			b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->attn_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	}
#if !defined(LM_HOST_CUDA_PRELUDE)
	{
#if defined(__CUDACC__)
		if ( rows > LM_SKINNY_ROWS_WIDE && rows <= 64u && b->row_invariant == 0u && b->gate_up_bf16 != 0 &&
			LmIsAligned16(qw) && LmIsAligned16(kw) && LmIsAligned16(b->normed_bf16) && (query & 1u) == 0u && (key & 1u) == 0u )
		{
			const K3SplitKTarget st[2] = {
				{qw, b->latent_bf16, (float *)0, query, query + key, 0u, 0u},
				{kw, b->latent_bf16, (float *)0, key, query + key, query, 0u}
			};
			return(K3SplitKMmaLaunch(st, 2u, b->normed_bf16, (float *)b->gate_up_bf16, rows, K3_HIDDEN, 10u, (uint16_t *)0, stream));
		}
#endif
		const LmSkinnyDenseTarget targets[2] = {
			{qw, b->latent_bf16, (float *)0, query, query + key, 0u, LmScaleTensorNone()},
			{kw, b->latent_bf16, (float *)0, key, query + key, query, LmScaleTensorNone()}};
#if defined(__CUDACC__)
		if ( rows > LM_SKINNY_ROWS_WIDE && b->row_invariant != 0u )
			return(K3ExactSplitMultiLaunch(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream));
#endif
		return(LmSkinnyDenseMulti<LmBf16Format>(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant,1u));
	}
#endif
	status = LmSkinnyDense<LmBf16Format>(qw,b->normed_bf16,b->latent_bf16,(float *)0,rows,K3_HIDDEN,query,query + key,0u,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(LmSkinnyDense<LmBf16Format>(kw,b->normed_bf16,b->latent_bf16,(float *)0,rows,K3_HIDDEN,key,query + key,query,stream));
}

template<class Format>
static int32_t K3LayerMlaDown(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	if ( b->normed_ready == 0u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->attn_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
#endif
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
			b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->attn_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	}
	if ( b->row_invariant != 0u && K3_MLA_DOWN_SLICED(b,1u) &&
		(K3_Q_LORA_RANK / b->kv_shard.degree <= LM_SKINNY_SPLIT_MAX_OUTPUTS || K3_MLA_KV_A_DIM / b->kv_shard.degree <= LM_SKINNY_SPLIT_MAX_OUTPUTS) )
	{
		status = K3SkinnyRows(b->mla_q_down_weight,b->normed_bf16,b->latent_bf16,(float *)0,rows,K3_HIDDEN,K3_Q_LORA_RANK,1u,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		return(K3SkinnyRows(b->mla_kv_a_weight,b->normed_bf16,b->kv_slot_bf16,(float *)0,rows,K3_HIDDEN,K3_MLA_KV_A_DIM,1u,stream));
	}
#if !defined(LM_HOST_CUDA_PRELUDE)
	if ( b->mla_q_down_scale == 0 && b->mla_kv_a_scale == 0 && (rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u) )
	{
		const LmSkinnyDenseTarget targets[2] = {
			{b->mla_q_down_weight, b->latent_bf16, (float *)0, K3_Q_LORA_RANK, K3_Q_LORA_RANK, 0u, LmScaleTensorNone()},
			{b->mla_kv_a_weight, b->kv_slot_bf16, (float *)0, K3_MLA_KV_A_DIM, K3_MLA_KV_A_DIM, 0u, LmScaleTensorNone()}};
		status = LmSkinnyDenseMulti<LmBf16Format>(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("k3 mla down: the multi-target skinny kernel refused %u rows; waves of up to %u rows and row-invariant waves "
				"run on it, so the weights and activation need 16-byte alignment\n",rows,LM_SKINNY_ROWS_WIDE);
		return(status);
	}
#endif
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_q_down_weight,b->mla_q_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_Q_LORA_RANK,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_kv_a_weight,b->mla_kv_a_scale,
		b->kv_slot_bf16,rows,K3_HIDDEN,K3_MLA_KV_A_DIM,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LayerMlaUp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,(const uint16_t *)0,(const uint16_t *)b->mla_q_norm_weight, (uint16_t *)0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_up_weight,b->mla_q_up_scale,
		b->query_bf16,rows,K3_Q_LORA_RANK,
		K3_RANK_DIM(b,mla_q_up_rows,K3_MLA_Q_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	const uint32_t *write_slots = b->slots != 0 ? b->slots : b->positions;
	if ( b->kv_shard.degree > 1u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LM_LAUNCH_DEPENDENT((K3RmsNormKvShardStoreKernel<Geometry,64u>), rows, 64u, 0, stream,
			b->cache_shard,b->kv_slot_bf16,(const uint16_t *)b->mla_kv_a_norm_weight,b->sequence_of_row,write_slots,rows,K3_LORA_RMS_EPSILON,
			b->shard_gather_direct != 0u ? b->shard_query_gathered_bf16 : (uint16_t *)0,
			b->shard_gather_direct != 0u ? b->shard_gather_total_ctx : 0u);
#else
		LM_LAUNCH_DEPENDENT((K3RmsNormKvShardStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
			b->cache_shard,b->kv_slot_bf16,(const uint16_t *)b->mla_kv_a_norm_weight,b->sequence_of_row,write_slots,rows,K3_LORA_RMS_EPSILON,
			b->shard_gather_direct != 0u ? b->shard_query_gathered_bf16 : (uint16_t *)0,
			b->shard_gather_direct != 0u ? b->shard_gather_total_ctx : 0u);
#endif
	}
	else
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
			b->kv_slot_bf16,(const uint16_t *)0,(const uint16_t *)b->mla_kv_a_norm_weight, (uint16_t *)0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
		LM_LAUNCH((LmKvStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
			b->cache,b->kv_slot_bf16,b->sequence_of_row,write_slots,rows, Geometry::kSlotBytes / 2u);
	}
	return(LM_LAUNCH_OK);
}

template<class Format, class Geometry>
static int32_t K3LayerMlaQuery(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status = K3LayerMlaDown<Format>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerMlaUp<Format,Geometry>(b,rows,multiprocessors,stream));
}

using K3IndexKv = LmKvLatent<16u, K3_INDEX_DIM, 0u, K3_KV_PAGE_SLOTS>;

static uint32_t K3LayerIndexed(const K3LayerBuffers *b)
{
	return(b->mla_index_key_weight != 0 ? 1u : 0u);
}

static uint32_t K3LayerIndexChunks(const K3LayerBuffers *b, uint32_t rows)
{
	return(b->index_active == 0u || b->index_chunk_rows == 0u ? 0u : (rows + b->index_chunk_rows - 1u) / b->index_chunk_rows);
}

static uint32_t K3LayerIndexChunkFirst(const K3LayerBuffers *b)
{
	return(b->index_chunk * b->index_chunk_rows);
}

static uint32_t K3LayerIndexChunkCount(const K3LayerBuffers *b, uint32_t rows)
{
	const uint32_t first = K3LayerIndexChunkFirst(b);
	return(first >= rows ? 0u : rows - first < b->index_chunk_rows ? rows - first : b->index_chunk_rows);
}

static uint32_t K3LayerIndexShare(const K3LayerBuffers *b, uint32_t count)
{
	const uint32_t degree = b->kv_shard.degree;
	return(degree < 2u || count < degree ? 0u : (count + degree - 1u) / degree);
}

static uint32_t K3LayerIndexOwnRows(const K3LayerBuffers *b, uint32_t count, uint32_t share)
{
	const uint32_t first = b->kv_shard.rank * share;
	return(first >= count ? 0u : count - first < share ? count - first : share);
}

static void K3LayerIndexPlan(K3LayerBuffers *b, uint32_t context)
{
	const uint32_t stride = SparkKvShardGatherKeys(b->kv_shard, context);
	b->index_active = b->index_send != 0 && context > K3_INDEX_TOPK ? 1u : 0u;
	b->index_score_stride = stride;
	b->index_keep = stride < K3_INDEX_TOPK ? stride : K3_INDEX_TOPK;
	b->index_chunk = 0u;
}

template<class Format>
static int32_t K3LayerMlaIndex(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	uint32_t head_done = 0u;
	if ( K3LayerIndexed(b) == 0u )
		return(LM_LAUNCH_OK);
	if ( b->index_key_bf16 == 0 || b->index_query_bf16 == 0 || b->index_head_bf16 == 0 || b->index_shard.shard.degree != b->kv_shard.degree ||
		LmKvViewIsConfigured(b->index_shard.pages) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	const uint32_t can_skip = b->index_active != 0u && b->fork_positions == 0 && b->sequence_of_row != 0 &&
		b->context_length != 0 && b->positions != 0 ? 1u : 0u;
#if !defined(LM_HOST_CUDA_PRELUDE)
	if ( b->index_active != 0u && (rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u || rows <= 64u) )
	{
#if defined(__CUDACC__)
		if ( can_skip != 0u && LmIsAligned16(b->mla_index_head_weight) && LmIsAligned16(b->normed_bf16) )
		{
			LmSkinnyDenseTarget key_target[1] = {
				{b->mla_index_key_weight, b->index_key_bf16, (float *)0, K3_INDEX_DIM, K3_INDEX_DIM, 0u, LmScaleTensorNone()}
			};
			status = rows > LM_SKINNY_ROWS_WIDE ?
				K3ExactSplitMultiLaunch(key_target,1u,b->normed_bf16,rows,K3_HIDDEN,stream) :
				LmSkinnyDenseMulti<LmBf16Format>(key_target,1u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant != 0u ? 1u : 0u,1u);
			if ( status != LM_LAUNCH_OK )
				return(status);
		}
		else
#endif
		{
			LmSkinnyDenseTarget targets[2] = {
				{b->mla_index_key_weight, b->index_key_bf16, (float *)0, K3_INDEX_DIM, K3_INDEX_DIM, 0u, LmScaleTensorNone()},
				{b->mla_index_head_weight, b->index_head_bf16, (float *)0, K3_INDEX_HEADS, K3_INDEX_HEADS, 0u, LmScaleTensorNone()}
			};
#if defined(__CUDACC__)
			status = rows > LM_SKINNY_ROWS_WIDE ?
				K3ExactSplitMultiLaunch(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream) :
				LmSkinnyDenseMulti<LmBf16Format>(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant != 0u ? 1u : 0u,1u);
#else
			status = LmSkinnyDenseMulti<LmBf16Format>(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant != 0u ? 1u : 0u,1u);
#endif
			if ( status != LM_LAUNCH_OK )
				return(status);
			head_done = 1u;
		}
	}
	else if ( b->index_active == 0u && rows > LM_SKINNY_ROWS_WIDE && rows <= 64u )
	{
#if defined(__CUDACC__)
		if ( b->row_invariant == 0u && b->gate_up_bf16 != 0 && LmIsAligned16(b->mla_index_key_weight) && LmIsAligned16(b->normed_bf16) )
		{
			const K3SplitKTarget st[1] = {
				{(const uint8_t *)b->mla_index_key_weight, b->index_key_bf16, (float *)0, K3_INDEX_DIM, K3_INDEX_DIM, 0u, 0u}
			};
			status = K3SplitKMmaLaunch(st, 1u, b->normed_bf16, (float *)b->gate_up_bf16, rows, K3_HIDDEN, 12u, (uint16_t *)0, stream);
		}
		else
		{
			LmSkinnyDenseTarget key_target[1] = {
				{b->mla_index_key_weight, b->index_key_bf16, (float *)0, K3_INDEX_DIM, K3_INDEX_DIM, 0u, LmScaleTensorNone()}
			};
			status = K3ExactSplitMultiLaunch(key_target,1u,b->normed_bf16,rows,K3_HIDDEN,stream);
		}
#else
		status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_index_key_weight,0,b->index_key_bf16,rows,K3_HIDDEN,K3_INDEX_DIM,multiprocessors,stream);
#endif
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	else
#endif
	{
		status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_index_key_weight,0,b->index_key_bf16,rows,K3_HIDDEN,K3_INDEX_DIM,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	LM_LAUNCH_DEPENDENT((K3LayerNormKvShardStoreKernel<K3IndexKv>), rows, 32u, 0, stream,
		b->index_shard,b->index_key_bf16,(const uint16_t *)b->mla_index_norm_weight,(const uint16_t *)b->mla_index_norm_bias,
		b->sequence_of_row,b->slots != 0 ? b->slots : b->positions,rows,K3_INDEX_EPSILON);
	if ( b->index_active == 0u )
		return(LM_LAUNCH_OK);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( can_skip != 0u && LmIsAligned16(b->mla_index_query_weight) && LmIsAligned16(b->latent_bf16) && LmIsAligned16(b->index_query_bf16) )
	{
		if ( rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u )
		{
			LM_LAUNCH_DEPENDENT((K3IndexQExactSkipKernel), K3_INDEX_QUERY / 8u, 128u, 0, stream,
				(const uint16_t *)b->mla_index_query_weight,b->latent_bf16,b->index_query_bf16,
				b->sequence_of_row,b->context_length,b->positions,rows);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		else
		{
			status = K3IndexQMmaWideSkipLaunch(b->mla_index_query_weight,b->latent_bf16,b->index_query_bf16,
				b->sequence_of_row,b->context_length,b->positions,rows,stream);
			if ( status != LM_LAUNCH_OK )
				return(status);
		}
		if ( head_done != 0u )
			return(LM_LAUNCH_OK);
		if ( LmIsAligned16(b->mla_index_head_weight) && LmIsAligned16(b->normed_bf16) )
		{
			if ( rows <= 64u || b->row_invariant != 0u )
				LM_LAUNCH_DEPENDENT((K3IndexHeadSkipKernel), dim3((K3_INDEX_HEADS + 3u) / 4u, rows), 128u, 0, stream,
					(const uint16_t *)b->mla_index_head_weight,b->normed_bf16,b->index_head_bf16,
					b->sequence_of_row,b->context_length,b->positions,rows);
			else
			{
				LmWeightPrefetchNote(stream, b->mla_index_head_weight, (uint64_t)K3_INDEX_HEADS * K3_HIDDEN * sizeof(uint16_t));
				LM_LAUNCH_DEPENDENT((K3IndexHeadMmaSkipKernel), (rows + 31u) / 32u, 128u, 0, stream,
					(const uint8_t *)b->mla_index_head_weight,b->normed_bf16,b->index_head_bf16,
					b->sequence_of_row,b->context_length,b->positions,rows);
			}
			return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		}
		return(K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_index_head_weight,0,b->index_head_bf16,rows,K3_HIDDEN,K3_INDEX_HEADS,multiprocessors,stream));
	}
#endif
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_index_query_weight,0,b->index_query_bf16,rows,K3_Q_LORA_RANK,K3_INDEX_QUERY,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK || head_done != 0u )
		return(status);
	return(K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_index_head_weight,0,b->index_head_bf16,rows,K3_HIDDEN,K3_INDEX_HEADS,multiprocessors,stream));
}

static int32_t K3LayerMlaIndexSelect(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	const uint32_t first = K3LayerIndexChunkFirst(b), count = K3LayerIndexChunkCount(b, rows), stride = b->index_score_stride;
	if ( b->index_active == 0u || count == 0u || stride == 0u || b->index_keep == 0u || (uint64_t)count * stride > b->index_score_floats )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( b->fork_positions == 0 && b->slot_offsets == 0 )
	{
		LM_LAUNCH_DEPENDENT((K3SparseScoreSlotsKernel<K3IndexKv,LmKvShardView,K3_INDEX_HEADS,K3_INDEX_DIM>),
			dim3(LmIndexScoreBlocks(stride),(count + LM_INDEX_SCORE_SHARED_ROWS - 1u) / LM_INDEX_SCORE_SHARED_ROWS), LM_INDEX_SCORE_THREADS, 0, stream,
			b->index_query_bf16 + (uint64_t)first * K3_INDEX_QUERY,b->index_head_bf16 + (uint64_t)first * K3_INDEX_HEADS,b->index_shard,
			b->sequence_of_row + first,b->context_length,b->positions + first,count,stride,K3_INDEX_SCALE / sqrtf((float)K3_INDEX_HEADS),
			b->index_scores,K3_INDEX_TOPK);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS ||
			LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(b->kv_shard,b->index_scores,stride,b->sequence_of_row + first,b->context_length,
				b->positions + first,b->index_keep,K3_INDEX_TOPK,count,b->index_send,b->index_error,stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
#endif
	if ( LmWeightedSparseScoreLaunch<K3IndexKv,K3_INDEX_HEADS,K3_INDEX_DIM,LmKvShardView>(b->index_query_bf16 + (uint64_t)first * K3_INDEX_QUERY,
			b->index_head_bf16 + (uint64_t)first * K3_INDEX_HEADS,b->index_shard,b->sequence_of_row + first,b->context_length,b->positions + first,
			count,stride,K3_INDEX_SCALE / sqrtf((float)K3_INDEX_HEADS),b->index_scores,stream,
			b->fork_positions != 0 ? b->fork_positions + first : (const uint32_t *)0,
			b->slot_offsets != 0 ? b->slot_offsets + first : (const uint32_t *)0,K3_INDEX_TOPK) != SPARK_GPU_SUCCESS ||
		LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(b->kv_shard,b->index_scores,stride,b->sequence_of_row + first,b->context_length,
			b->positions + first,b->index_keep,K3_INDEX_TOPK,count,b->index_send,b->index_error,stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaIndexMerge(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	const uint32_t count = K3LayerIndexChunkCount(b, rows), share = K3LayerIndexShare(b, count), own = K3LayerIndexOwnRows(b, count, share);
	const uint32_t first = K3LayerIndexChunkFirst(b) + b->kv_shard.rank * share, merged = share != 0u ? own : count;
	if ( b->index_active == 0u || count == 0u || (share != 0u && b->index_selected_own == 0) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( merged == 0u )
		return(LM_LAUNCH_OK);
	if ( !(LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(b->index_received,(uint64_t)(share != 0u ? share : count) * b->index_keep,b->kv_shard,b->index_keep, b->sequence_of_row + first,b->context_length,b->positions + first,K3_INDEX_TOPK,K3_INDEX_TOPK,merged,b->index_bitmap_words,b->index_bitmaps, share != 0u ? b->index_selected_own : b->index_selected + (uint64_t)first * K3_INDEX_TOPK,b->index_error,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaGate(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	const uint32_t gate_rows = K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( rows > LM_SKINNY_ROWS_WIDE && rows <= 64u && b->row_invariant == 0u && b->gate_up_bf16 != 0 &&
		LmIsAligned16(b->mla_gate_weight) && LmIsAligned16(b->normed_bf16) && (gate_rows & 1u) == 0u )
	{
		const K3SplitKTarget st[1] = {
			{(const uint8_t *)b->mla_gate_weight, b->gate_bf16, (float *)0, gate_rows, gate_rows, 0u, 0u}
		};
		return(K3SplitKMmaLaunch(st, 1u, b->normed_bf16, (float *)b->gate_up_bf16, rows, K3_HIDDEN, 4u, (uint16_t *)0, stream));
	}
#endif
	return(K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_gate_weight,0,
		b->gate_bf16,rows,K3_HIDDEN,gate_rows,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerMlaGated(const K3LayerBuffers *b, uint32_t rows, uint32_t gate_ready, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	if ( b->shard_mha == 0u && (b->shard_gather == 0u || b->shard_gather_expanded == 0u) )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->mla_kv_b_value_weight,
			(uint64_t)K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS) * K3_V_HEAD_DIM * K3_KV_LORA_RANK * sizeof(uint16_t));
#endif
		if ( LmPerHeadProjectChainLaunch<K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_V_HEAD_DIM>(
				b->attention_out_bf16,(const uint16_t *)b->mla_kv_b_value_weight,b->value_bf16,
				K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),rows,stream,b->row_invariant) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	status = gate_ready != 0u ? LM_LAUNCH_OK : K3LayerMlaGate(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->value_bf16,b->gate_bf16,K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM));
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerMlaOutput(const K3LayerBuffers *b, uint32_t rows, uint16_t *partial_accumulate, uint32_t gate_ready, uint32_t multiprocessors, SparkGpuStream stream)
{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( b->tp_sharded != 0u && b->mla_out_scale == 0 && (rows <= 12u || b->row_invariant != 0u) &&
		K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS) == 6u && K3_RANK_DIM(b,mla_out_input,K3_MLA_OUT_DIM) == 6u * K3_V_HEAD_DIM &&
		K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM) == 6u * K3_V_HEAD_DIM &&
		LmIsAligned16(b->mla_out_weight) && LmIsAligned16(b->attention_out_bf16) )
	{
		if ( b->shard_mha == 0u && (b->shard_gather == 0u || b->shard_gather_expanded == 0u) )
		{
			LmWeightPrefetchNote(stream, b->mla_kv_b_value_weight,
				(uint64_t)6u * K3_V_HEAD_DIM * K3_KV_LORA_RANK * sizeof(uint16_t));
			if ( LmPerHeadProjectChainLaunch<K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_V_HEAD_DIM>(
					b->attention_out_bf16,(const uint16_t *)b->mla_kv_b_value_weight,b->value_bf16,
					6u,rows,stream,b->row_invariant) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		if ( gate_ready == 0u )
		{
			const int32_t gs = K3LayerMlaGate(b,rows,multiprocessors,stream);
			if ( gs != LM_LAUNCH_OK )
				return(gs);
		}
		LmWeightPrefetchNote(stream, b->mla_out_weight, (uint64_t)K3_HIDDEN * 6u * K3_V_HEAD_DIM * sizeof(uint16_t));
		if ( rows == 1u )
			LM_LAUNCH_DEPENDENT((K3MlaGateOutKernel<1u>), dim3(K3_HIDDEN / 32u, 1u), 256u, 0u, stream,
				b->value_bf16, b->gate_bf16, (const uint16_t *)b->mla_out_weight, b->attention_out_bf16, rows);
		else if ( rows <= 4u )
			LM_LAUNCH_DEPENDENT((K3MlaGateOutKernel<4u>), dim3(K3_HIDDEN / 32u, 1u), 256u, 0u, stream,
				b->value_bf16, b->gate_bf16, (const uint16_t *)b->mla_out_weight, b->attention_out_bf16, rows);
		else if ( rows <= 8u )
			LM_LAUNCH_DEPENDENT((K3MlaGateOutKernel<8u>), dim3(K3_HIDDEN / 32u, 1u), 256u, 0u, stream,
				b->value_bf16, b->gate_bf16, (const uint16_t *)b->mla_out_weight, b->attention_out_bf16, rows);
		else
			LM_LAUNCH_DEPENDENT((K3MlaGateOutKernel<12u>), dim3(K3_HIDDEN / 32u, (rows + 11u) / 12u), 256u, 0u, stream,
				b->value_bf16, b->gate_bf16, (const uint16_t *)b->mla_out_weight, b->attention_out_bf16, rows);
		return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
	}
#endif
	const int32_t status = K3LayerMlaGated<Format>(b,rows,gate_ready,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3Project<LmBf16Format>(b,b->value_bf16,b->mla_out_weight,b->mla_out_scale,
		b->attention_out_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : partial_accumulate,
		rows,K3_RANK_DIM(b,mla_out_input,K3_MLA_OUT_DIM),K3_HIDDEN,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LayerMla(const K3LayerBuffers *b, uint32_t rows, uint32_t context, uint16_t *partial_accumulate, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	if ( b->kv_shard.degree > 1u || K3LayerIndexed(b) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	status = K3LayerMlaQuery<Format,Geometry>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmAttentionDecodeKernel<Geometry,K3_ATTN_THREADS,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>), dim3(rows,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS)), K3_ATTN_THREADS, 0, stream,
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),K3_MLA_QK_SCALE,b->attention_out_bf16,b->positions,b->fork_positions,b->slot_offsets);
	(void)context;
	return(K3LayerMlaOutput<Format>(b,rows,partial_accumulate,0u,multiprocessors,stream));
}

static uint32_t K3LayerMlaShardReady(const K3LayerBuffers *b, uint32_t rows)
{
	uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	return(b->kv_shard.degree > 1u && heads * b->kv_shard.degree == K3_MLA_HEADS &&
		SparkKvShardValid(b->kv_shard,K3_KV_PAGE_SLOTS) != 0u &&
		b->cache_shard.shard.degree == b->kv_shard.degree && b->cache_shard.shard.rank == b->kv_shard.rank &&
		LmKvViewIsConfigured(b->cache_shard.pages) &&
		b->shard_query_gathered_bf16 != 0 && b->shard_partials_f32 != 0 && b->shard_partials_received_f32 != 0 &&
		b->shard_query_rank_stride >= (uint64_t)rows * heads * K3_MLA_KV_A_DIM &&
		b->shard_partial_rank_stride >= (uint64_t)rows * heads * LM_LATENT_SHARD_RECORD_FLOATS(K3_KV_LORA_RANK) ? 1u : 0u);
}

template<class Geometry>
static int32_t K3LayerMlaShardPartials(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	if ( K3LayerMlaShardReady(b,rows) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->index_active != 0u )
	{
		if ( !(LmLatentShardAttendLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>( b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride, K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions, b->index_selected,K3_INDEX_TOPK,K3_INDEX_TOPK,K3_MLA_QK_SCALE,b->shard_partials_f32,b->shard_partial_rank_stride,rows,stream,b->fork_positions,b->slot_offsets) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( b->row_invariant != 0u )
	{
		const uint32_t splits = 1u;
		if ( !(LmLatentShardSplitLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>( b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride, K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions, K3_MLA_QK_SCALE,b->shard_partials_received_f32,b->shard_received_floats, b->shard_partials_f32,b->shard_partial_rank_stride,rows,splits,stream,b->fork_positions,b->slot_offsets) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS )
	{
		if ( !(LmLatentShardPrefillLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>( b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride, K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions, K3_MLA_QK_SCALE,b->shard_partials_f32,b->shard_partial_rank_stride,rows,stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( K3LayerIndexed(b) != 0u || rows <= LM_SKINNY_ROWS_WIDE )
	{
		if ( !(LmLatentShardSplitLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>( b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride, K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions, K3_MLA_QK_SCALE,b->shard_partials_received_f32,b->shard_received_floats, b->shard_partials_f32,b->shard_partial_rank_stride,rows,1u,stream,b->fork_positions,b->slot_offsets) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( !(LmLatentShardDecodeLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>( b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride, K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions, K3_MLA_QK_SCALE,b->shard_partials_received_f32,b->shard_received_floats, b->shard_partials_f32,b->shard_partial_rank_stride,rows,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static uint32_t K3LayerMlaWindowKeys(const K3LayerBuffers *b, uint32_t window)
{
	const uint32_t begin = window * b->shard_gather_window_keys;
	if ( b->shard_gather_windows <= 1u )
		return(b->shard_gather_most_keys);
	if ( begin >= b->shard_gather_most_keys )
		return(0u);
	return(b->shard_gather_most_keys - begin < b->shard_gather_window_keys ? b->shard_gather_most_keys - begin : b->shard_gather_window_keys);
}

static uint64_t K3LayerMlaWindowStride(const K3LayerBuffers *b, uint32_t window)
{
	const uint64_t unit = (uint64_t)K3_HIDDEN * sizeof(uint16_t);
	if ( b->shard_gather_windows <= 1u )
		return(b->shard_gather_stride);
	return(((uint64_t)K3LayerMlaWindowKeys(b,window) * K3_MLA_KV_A_DIM * sizeof(uint16_t) + unit - 1u) / unit * unit);
}

static const uint8_t *K3LayerMlaWindowLanding(const K3LayerBuffers *b, uint32_t window)
{
	if ( b->shard_gather_direct != 0u )
		return((const uint8_t *)b->shard_partials_received_f32);
	return(b->shard_gather_keys + (b->shard_gather_windows <= 1u ? 0u : (uint64_t)(window & 1u) * b->shard_gather_buffer_bytes));
}

static const uint8_t *K3LayerMlaWindowSend(const K3LayerBuffers *b, uint32_t window)
{
	return(b->shard_gather_send + (b->shard_gather_windows <= 1u ? 0u :
		(uint64_t)window * b->shard_gather_window_keys * K3_MLA_KV_A_DIM * sizeof(uint16_t)));
}

static uint32_t K3LayerMlaGatherReady(const K3LayerBuffers *b, uint32_t rows)
{
	const uint64_t key_bytes = (uint64_t)K3_MLA_KV_A_DIM * sizeof(uint16_t);
	const uint32_t cold_direct = b->shard_gather_direct != 0u && b->shard_gather_prior_ctx == 0u ? 1u : 0u;
	return(K3LayerMlaShardReady(b,rows) != 0u && b->shard_gather != 0u && rows > LM_SKINNY_ROWS_WIDE && b->row_invariant == 0u &&
		b->shard_gather_list != 0 && b->shard_gather_offset != 0 && b->shard_gather_context != 0 &&
		b->shard_gather_send != 0 && b->shard_gather_keys != 0 && (b->shard_gather_most_keys != 0u || cold_direct != 0u) &&
		(b->shard_gather_windows <= 1u
			? (cold_direct != 0u || b->shard_gather_stride >= (uint64_t)b->shard_gather_most_keys * key_bytes)
			: b->shard_gather_window_keys != 0u && b->shard_gather_running != 0 && b->shard_gather_buffer_bytes != 0u &&
				b->shard_gather_stride >= (uint64_t)b->shard_gather_window_keys * key_bytes &&
				(uint64_t)(b->shard_gather_windows - 1u) * b->shard_gather_window_keys < b->shard_gather_most_keys &&
				(uint64_t)b->shard_gather_windows * b->shard_gather_window_keys >= b->shard_gather_most_keys) ? 1u : 0u);
}

static uint32_t K3LayerMlaPasses(const K3LayerBuffers *b)
{
	if ( b->shard_gather_expanded != 0u )
		return(b->shard_gather_passes);
	return(b->shard_gather_windows > 1u ? b->shard_gather_windows : 1u);
}

static uint32_t K3LayerMlaPassBegin(const K3LayerBuffers *b, uint32_t pass)
{
	return(pass * (b->shard_gather_expanded != 0u ? b->shard_gather_value_keys : b->shard_gather_window_keys));
}

static uint32_t K3LayerMlaPassKeys(const K3LayerBuffers *b, uint32_t pass)
{
	if ( b->shard_gather_direct != 0u && b->shard_gather_expanded != 0u )
		return(pass == 0u ? b->shard_gather_total_ctx : 0u);
	const uint32_t begin = K3LayerMlaPassBegin(b,pass);
	if ( b->shard_gather_expanded == 0u )
		return(K3LayerMlaWindowKeys(b,pass));
	if ( begin >= b->shard_gather_most_keys )
		return(0u);
	return(b->shard_gather_most_keys - begin < b->shard_gather_value_keys ? b->shard_gather_most_keys - begin : b->shard_gather_value_keys);
}

static const uint8_t *K3LayerMlaPassLanding(const K3LayerBuffers *b, uint32_t pass)
{
	if ( b->shard_gather_windows > 1u )
		return(K3LayerMlaWindowLanding(b,pass));
	return(b->shard_gather_keys + (uint64_t)K3LayerMlaPassBegin(b,pass) * K3_MLA_KV_A_DIM * sizeof(uint16_t));
}

static uint64_t K3LayerMlaValueStride(const K3LayerBuffers *b)
{
	return((uint64_t)b->shard_gather_value_keys * K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS) * K3_V_HEAD_DIM * sizeof(uint16_t));
}

static uint32_t K3LayerMlaExpandedReady(const K3LayerBuffers *b, uint32_t rows)
{
	const uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	const uint32_t passes = b->shard_gather_passes, keys = b->shard_gather_value_keys;
	if ( b->shard_gather_direct != 0u )
		return(K3LayerMlaGatherReady(b,rows) != 0u && b->shard_gather_expanded != 0u && b->shard_gather_values != 0 &&
			b->mla_kv_b_value_weight != 0 && heads % LM_EXPANDED_PREFILL_HEADS == 0u && b->shard_gather_total_ctx != 0u &&
			passes == 1u && b->shard_gather_values_bytes >= (uint64_t)b->shard_gather_total_ctx * heads * K3_V_HEAD_DIM * sizeof(uint16_t) ? 1u : 0u);
	return(K3LayerMlaGatherReady(b,rows) != 0u && b->shard_gather_expanded != 0u && b->shard_gather_values != 0 &&
		b->mla_kv_b_value_weight != 0 && heads % LM_EXPANDED_PREFILL_HEADS == 0u && keys != 0u && keys % LM_EXPANDED_PREFILL_TILE == 0u &&
		b->shard_gather_values_bytes >= (uint64_t)b->kv_shard.degree * K3LayerMlaValueStride(b) &&
		passes != 0u && (uint64_t)(passes - 1u) * keys < b->shard_gather_most_keys && (uint64_t)passes * keys >= b->shard_gather_most_keys &&
		(passes == 1u || b->shard_gather_running != 0) &&
		(b->shard_gather_windows <= 1u || (b->shard_gather_windows == passes && b->shard_gather_window_keys == keys)) ? 1u : 0u);
}

static uint32_t K3LayerMlaGatherIndexReady(const K3LayerBuffers *b)
{
	return(K3LayerIndexed(b) == 0u || (b->shard_gather_windows <= 1u && b->shard_gather_expanded == 0u && b->shard_gather_prefetched == 0u) ? 1u : 0u);
}

#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
static __global__ __launch_bounds__(128u, 4)
void K3UnpackPriorKvKernel(SparkKvShard shard, const uint8_t *__restrict__ gathered_bytes, uint64_t rank_stride,
	uint32_t prior_local_keys, uint32_t old_ctx, uint16_t *__restrict__ kv_all)
{
	constexpr uint32_t SLOT_V = K3_MLA_KV_A_DIM >> 3u;
	static_assert(SLOT_V == 72u && SLOT_V <= 128u, "one thread per 16-byte vector of a K3 latent KV slot");
	const uint32_t local = blockIdx.x, src_rank = blockIdx.y;
	SparkKvShard src_shard = shard;
	src_shard.rank = src_rank;
	const uint32_t pos = SparkKvShardLocalPosition(src_shard, local);
	LmDependentWait();
	if ( local < prior_local_keys && src_rank < shard.degree && pos < old_ctx && threadIdx.x < SLOT_V )
	{
		const uint4 *s4 = (const uint4 *)(gathered_bytes + (uint64_t)src_rank * rank_stride) + (uint64_t)local * SLOT_V;
		uint4 *d4 = (uint4 *)(kv_all + (uint64_t)pos * K3_MLA_KV_A_DIM);
		d4[threadIdx.x] = __ldg(s4 + threadIdx.x);
	}
	LmDependentRelease();
}

template<uint32_t TOPK>
__global__ __launch_bounds__(K3_LAYER_THREADS, 2)
void K3FillDenseSelectedKernel(const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t *__restrict__ selected)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	const uint32_t pos = positions[row];
	uint32_t bound = context_length[sequence_of_row[row]];
	if ( pos + 1u < bound )
		bound = pos + 1u;
	if ( bound <= TOPK )
	{
		uint32_t *dst = selected + (uint64_t)row * TOPK;
		for ( uint32_t i = threadIdx.x; i < TOPK; i += K3_LAYER_THREADS )
			dst[i] = i < bound ? i : LM_INDEX_SHARD_NO_POSITION;
	}
	LmDependentRelease();
}

template<bool SPARSE>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
void K3MlaGatherPrefillKernel(
	const uint16_t *__restrict__ query_bf16,
	const uint16_t *__restrict__ kv_all,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows,
	uint32_t total_heads,
	uint32_t total_ctx,
	const uint32_t *__restrict__ union_positions,
	const uint8_t *__restrict__ union_masks,
	const uint32_t *__restrict__ union_counts,
	uint32_t union_capacity)
{
	constexpr uint32_t HEADS = 2u;
	constexpr uint32_t LATENT = K3_KV_LORA_RANK;
	constexpr uint32_t DIM = K3_MLA_KV_A_DIM;
	constexpr uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / HEADS;
	constexpr uint32_t KEY_GROUPS = LM_PREFILL_ATTN_WARPS / 2u;
	constexpr uint32_t KEY_FRAGS = TILE / (KEY_GROUPS * 8u);
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / LM_PREFILL_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	static_assert(HEADS == 2u && ROWS_PER_BLOCK == 8u && QUERIES == 16u && KEY_FRAGS == 1u && QUERY_KEYS == 2u,
		"2-head 8-row grouped prefill fills one 16-query MMA tile per CTA");
	static_assert(LmPrefillAttnSharedBytes<DIM, QUERIES>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "K3 prefill tiles fit SM shared memory");
	extern __shared__ __align__(128) uint8_t k3_prefill_smem[];
	uint8_t *keys = k3_prefill_smem;
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
	const uint32_t half = (warp / KEY_GROUPS) & 1u, key_group = warp % KEY_GROUPS;
	const uint32_t table = gridDim.x - 1u - blockIdx.x;
	const uint32_t head_base = blockIdx.y * HEADS;
	const uint32_t *tile_positions = SPARSE ? union_positions + (uint64_t)table * union_capacity : (const uint32_t *)0;
	const uint8_t *tile_masks = SPARSE ? union_masks + (uint64_t)table * union_capacity : (const uint8_t *)0;
	const uint32_t first_row = table * ROWS_PER_BLOCK;
	const uint32_t row_end = LmPrefillMin(rows, first_row + ROWS_PER_BLOCK);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t query_base = key_base + TILE * ROW_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[COLUMN_FRAGS][4];
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
		return position <= last_position && position < total_ctx ? (const uint8_t *)(kv_all + (uint64_t)position * DIM) : (const uint8_t *)0;
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
			K3CpAsync16L256(key_base + (buffer * TILE + slot_index) * ROW_BYTES + LmPrefillSwizzle(slot_index, piece) * 16u,
				slot != 0 ? (const void *)(slot + piece * 16u) : (const void *)query_bf16, slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	for ( uint32_t idx = threadIdx.x; idx < QUERIES * CHUNKS; idx += LM_PREFILL_ATTN_THREADS )
	{
		const uint32_t q = idx / CHUNKS, chunk = idx % CHUNKS;
		const uint32_t row = first_row + q / HEADS;
		const uint16_t *src = row >= row_end ? query_bf16 : query_bf16 + ((uint64_t)row * total_heads + head_base + q % HEADS) * DIM + chunk * 8u;
		K3CpAsync16L256(query_base + q * ROW_BYTES + LmPrefillSwizzle(q, chunk) * 16u, src, row >= row_end ? 0u : 16u);
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
		const uint32_t a_row = (matrix & 1u) * 8u + within;
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
	for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
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
			float score[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			uint32_t b[2][2];
			const uint32_t b_row = key_group * 8u + within;
			LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, half * HALF_STEPS * 2u + (matrix & 1u)) * 16u, b[0]);
			#pragma unroll
			for ( uint32_t step = 0u; step < HALF_STEPS; ++step )
			{
				const uint32_t cur = step & 1u, nxt = cur ^ 1u;
				if ( step + 1u < HALF_STEPS )
				{
					const uint32_t next_chunk = (half * HALF_STEPS + step + 1u) * 2u + (matrix & 1u);
					LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, next_chunk) * 16u, b[nxt]);
				}
				LmMmaBf16(score, query_frag[step], b[cur]);
			}
			const uint32_t base_col = key_group * 8u + (lane & 3u) * 2u;
			const uint32_t row_lo = half * QUERIES + (lane >> 2u);
			*(float2 *)(score_parts + row_lo * LM_PREFILL_ATTN_PARTIAL_STRIDE + base_col) = make_float2(score[0], score[1]);
			*(float2 *)(score_parts + (row_lo + 8u) * LM_PREFILL_ATTN_PARTIAL_STRIDE + base_col) = make_float2(score[2], score[3]);
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
			const uint32_t key_base_idx = part * QUERY_KEYS, own_row = own / HEADS;
			const bool full_dense_tile = !SPARSE && (first_row + own_row < row_end) && ((tile + 1u) * TILE - 1u <= first_position_min);
			const float2 a = *(const float2 *)(score_parts + own * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
			const float2 b = *(const float2 *)(score_parts + (QUERIES + own) * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
			float v0 = a.x + b.x, v1 = a.y + b.y;
			if ( !full_dense_tile )
			{
				v0 = ((key_rows[buffer * TILE + key_base_idx] >> own_row) & 1u) != 0u ? v0 : -INFINITY;
				v1 = ((key_rows[buffer * TILE + key_base_idx + 1u] >> own_row) & 1u) != 0u ? v1 : -INFINITY;
			}
			float peak = fmaxf(v0, v1);
			#pragma unroll
			for ( uint32_t shift = 1u; shift < QUERY_THREADS; shift <<= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			const float previous_max = running_max[own];
			const float next_max = fmaxf(previous_max, peak);
			const float scale = previous_max == -INFINITY ? 0.0f : SparkMathExp2Fast((previous_max - next_max) * scale_log2);
			const float shift_val = next_max == -INFINITY ? 0.0f : next_max * scale_log2;
			const float w0 = full_dense_tile ? SparkMathExp2Fast(fmaf(v0, scale_log2, -shift_val)) :
				(next_max == -INFINITY || v0 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v0, scale_log2, -shift_val)));
			const float w1 = full_dense_tile ? SparkMathExp2Fast(fmaf(v1, scale_log2, -shift_val)) :
				(next_max == -INFINITY || v1 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v1, scale_log2, -shift_val)));
			*(uint32_t *)(weight_high + own * LM_PREFILL_ATTN_WEIGHT_STRIDE + key_base_idx) = LmPackBf16Pair(w0, w1);
			float total = w0 + w1;
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
			uint32_t a_high[4], b[2][4], first[2], second[2];
			const float low_scale = rescale[lane >> 2u], high_scale = rescale[(lane >> 2u) + 8u];
			if ( low_scale != 1.0f || high_scale != 1.0f )
			{
				#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
				{
					output[frag][0] *= low_scale;
					output[frag][1] *= low_scale;
					output[frag][2] *= high_scale;
					output[frag][3] *= high_scale;
				}
			}
			#pragma unroll
			for ( uint32_t step = 0u; step < TILE / 16u; ++step )
			{
				const uint32_t a_row = (matrix & 1u) * 8u + within, a_col = step * 16u + (matrix >> 1u) * 8u;
				const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_col), a_high);
				LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, (warp * WARP_COLUMNS) / 8u + (matrix >> 1u)) * 16u, b[0]);
				#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u )
				{
					const uint32_t cur = (frag >> 1u) & 1u, nxt = cur ^ 1u;
					if ( frag + 2u < COLUMN_FRAGS )
					{
						const uint32_t next_chunk = (warp * WARP_COLUMNS + (frag + 2u) * 8u) / 8u + (matrix >> 1u);
						LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, next_chunk) * 16u, b[nxt]);
					}
					first[0] = b[cur][0]; first[1] = b[cur][1];
					second[0] = b[cur][2]; second[1] = b[cur][3];
					LmMmaBf16(output[frag], a_high, first);
					LmMmaBf16(output[frag + 1u], a_high, second);
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
	const uint32_t q_lo = lane >> 2u, q_hi = q_lo + 8u;
	const uint32_t row_lo = first_row + q_lo / HEADS, row_hi = first_row + q_hi / HEADS;
	const float inv_lo = row_lo < row_end && running_sum[q_lo] > 0.0f ? 1.0f / running_sum[q_lo] : 0.0f;
	const float inv_hi = row_hi < row_end && running_sum[q_hi] > 0.0f ? 1.0f / running_sum[q_hi] : 0.0f;
	#pragma unroll
	for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
	{
		const uint32_t col = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
		if ( row_lo < row_end )
			__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_lo * total_heads + head_base + q_lo % HEADS) * LATENT + col),
				LmPackBf16Pair(output[frag][0] * inv_lo, output[frag][1] * inv_lo));
		if ( row_hi < row_end )
			__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_hi * total_heads + head_base + q_hi % HEADS) * LATENT + col),
				LmPackBf16Pair(output[frag][2] * inv_hi, output[frag][3] * inv_hi));
	}
	LmDependentRelease();
}

static __host__ __device__ constexpr uint32_t K3Prefill6HeadSharedBytes(void)
{
	return(2u * 16u * K3_MLA_KV_A_DIM * 2u
		+ 48u * K3_MLA_KV_A_DIM * 2u
		+ 2u * 48u * 16u * 4u
		+ 48u * 16u * 2u
		+ 3u * 48u * 4u
		+ 2u * 16u * 8u
		+ 2u * 16u * 4u);
}

__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
void K3MlaGatherPrefill6HeadKernel(
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
	constexpr uint32_t HEADS = 6u, LATENT = K3_KV_LORA_RANK, DIM = K3_MLA_KV_A_DIM;
	constexpr uint32_t ROWS_PER_BLOCK = 8u, QUERIES = HEADS * ROWS_PER_BLOCK, Q_TILES = QUERIES / 16u;
	constexpr uint32_t TILE = 16u, ROW_BYTES = DIM * 2u, CHUNKS = DIM / 8u;
	constexpr uint32_t K_STEPS = DIM / 16u, K_SPLITS = 4u, SPLIT_STEPS = K_STEPS / K_SPLITS;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS, COLUMN_FRAGS = WARP_COLUMNS / 8u;
	static_assert(K_STEPS % K_SPLITS == 0u && K3Prefill6HeadSharedBytes() <= LM_PREFILL_ATTN_SHARED_LIMIT,
		"6-head sparse MLA prefill fits GB10 dynamic shared memory");
	extern __shared__ __align__(128) uint8_t k3_prefill6_smem[];
	uint8_t *keys = k3_prefill6_smem;
	uint8_t *queries_smem = keys + 2u * TILE * ROW_BYTES;
	float *score_parts = (float *)(queries_smem + QUERIES * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * TILE);
	float *running_max = (float *)(weight_high + QUERIES * TILE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	uint32_t *key_rows = (uint32_t *)(slots + 2u * TILE);
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t k_split = warp >> 1u, key_group = warp & 1u;
	const uint32_t table = gridDim.x - 1u - blockIdx.x;
	const uint32_t *tile_positions = union_positions + (uint64_t)table * union_capacity;
	const uint8_t *tile_masks = union_masks + (uint64_t)table * union_capacity;
	const uint32_t first_row = table * ROWS_PER_BLOCK;
	const uint32_t row_end = LmPrefillMin(rows, first_row + ROWS_PER_BLOCK);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t query_base = (uint32_t)__cvta_generic_to_shared(queries_smem);
	float output[Q_TILES][COLUMN_FRAGS][4];
	uint32_t row_limit[ROWS_PER_BLOCK];
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	const uint32_t sequence = __ldg(sequence_of_row + first_row);
	const uint32_t seq_ctx = __ldg(context_length + sequence);
	const uint32_t ctx_cap = seq_ctx < total_ctx ? seq_ctx : total_ctx;
	uint32_t last_position = 0u;
	for ( uint32_t r = first_row; r < row_end; ++r )
	{
		const uint32_t rpos = __ldg(row_position + r);
		const uint32_t lim = rpos + 1u < ctx_cap ? rpos : (ctx_cap > 0u ? ctx_cap - 1u : 0u);
		last_position = lim > last_position ? lim : last_position;
	}
	const uint32_t listed_keys = ctx_cap == 0u ? 0u : __ldg(union_counts + table);
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
		position = listed < listed_keys ? __ldg(tile_positions + listed) : 0xffffffffu;
		members = listed < listed_keys ? (uint32_t)__ldg(tile_masks + listed) : 0u;
	};
	auto resolve = [&](uint32_t position, uint32_t members, uint32_t &valid) -> const uint8_t *
	{
		valid = 0u;
		#pragma unroll
		for ( uint32_t r = 0u; r < ROWS_PER_BLOCK; ++r )
			valid |= (first_row + r < row_end && ((members >> r) & 1u) != 0u && position <= row_limit[r] ? 1u : 0u) << r;
		return position <= last_position && position < total_ctx ? (const uint8_t *)(kv_all + (uint64_t)position * DIM) : (const uint8_t *)0;
	};
	auto copy = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		const uint8_t *s0 = slots[buffer * TILE + warp * 2u];
		const uint8_t *s1 = slots[buffer * TILE + warp * 2u + 1u];
		#pragma unroll
		for ( uint32_t iter = 0u; iter < 5u; ++iter )
		{
			const uint32_t k = iter * 32u + lane;
			if ( k < 2u * CHUNKS )
			{
				const uint32_t s = k >= CHUNKS ? 1u : 0u;
				const uint32_t piece = k - s * CHUNKS;
				const uint32_t slot_index = warp * 2u + s;
				const uint8_t *slot = s == 0u ? s0 : s1;
				K3CpAsync16L256(key_base + (buffer * TILE + slot_index) * ROW_BYTES + LmPrefillSwizzle(slot_index, piece) * 16u,
					slot != 0 ? (const void *)(slot + piece * 16u) : (const void *)query_bf16, slot != 0 ? 16u : 0u);
			}
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	for ( uint32_t idx = threadIdx.x; idx < QUERIES * CHUNKS; idx += LM_PREFILL_ATTN_THREADS )
	{
		const uint32_t q = idx / CHUNKS, chunk = idx % CHUNKS;
		const uint32_t row = first_row + q / HEADS;
		const uint16_t *src = row >= row_end ? query_bf16 : query_bf16 + ((uint64_t)row * HEADS + q % HEADS) * DIM + chunk * 8u;
		K3CpAsync16L256(query_base + q * ROW_BYTES + LmPrefillSwizzle(q, chunk) * 16u, src, row >= row_end ? 0u : 16u);
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
	if ( threadIdx.x < QUERIES )
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
	#pragma unroll
	for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
		#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
			output[qt][frag][0] = output[qt][frag][1] = output[qt][frag][2] = output[qt][frag][3] = 0.0f;
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
			float score[Q_TILES][4];
			#pragma unroll
			for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
				score[qt][0] = score[qt][1] = score[qt][2] = score[qt][3] = 0.0f;
			uint32_t b[2][2];
			const uint32_t b_row = key_group * 8u + within;
			const uint32_t a_row_off = (matrix & 1u) * 8u + within;
			const uint32_t step0 = k_split * SPLIT_STEPS;
			LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, step0 * 2u + (matrix & 1u)) * 16u, b[0]);
			#pragma unroll
			for ( uint32_t s = 0u; s < SPLIT_STEPS; ++s )
			{
				const uint32_t cur = s & 1u, nxt = cur ^ 1u, step = step0 + s;
				if ( s + 1u < SPLIT_STEPS )
					LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, (step + 1u) * 2u + (matrix & 1u)) * 16u, b[nxt]);
				const uint32_t a_chunk = step * 2u + (matrix >> 1u);
				uint32_t a0[4], a1[4];
				LmPrefillLoadX4(query_base + a_row_off * ROW_BYTES + LmPrefillSwizzle(a_row_off, a_chunk) * 16u, a0);
				LmPrefillLoadX4(query_base + (16u + a_row_off) * ROW_BYTES + LmPrefillSwizzle(16u + a_row_off, a_chunk) * 16u, a1);
				LmMmaBf16(score[0], a0, b[cur]);
				LmPrefillLoadX4(query_base + (32u + a_row_off) * ROW_BYTES + LmPrefillSwizzle(32u + a_row_off, a_chunk) * 16u, a0);
				LmMmaBf16(score[1], a1, b[cur]);
				LmMmaBf16(score[2], a0, b[cur]);
			}
			const uint32_t base_col = key_group * 8u + (lane & 3u) * 2u;
			if ( k_split < 2u )
			{
				#pragma unroll
				for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
				{
					const uint32_t row_lo = k_split * QUERIES + qt * 16u + (lane >> 2u);
					*(float2 *)(score_parts + row_lo * TILE + base_col) = make_float2(score[qt][0], score[qt][1]);
					*(float2 *)(score_parts + (row_lo + 8u) * TILE + base_col) = make_float2(score[qt][2], score[qt][3]);
				}
			}
			__syncthreads();
			if ( k_split >= 2u )
			{
				#pragma unroll
				for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
				{
					const uint32_t row_lo = (k_split - 2u) * QUERIES + qt * 16u + (lane >> 2u);
					float2 *p0 = (float2 *)(score_parts + row_lo * TILE + base_col);
					float2 *p1 = (float2 *)(score_parts + (row_lo + 8u) * TILE + base_col);
					const float2 d0 = *p0, d1 = *p1;
					*p0 = make_float2(d0.x + score[qt][0], d0.y + score[qt][1]);
					*p1 = make_float2(d1.x + score[qt][2], d1.y + score[qt][3]);
				}
			}
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x >> 2u, part = threadIdx.x & 3u;
			if ( own < QUERIES )
			{
				const uint32_t k0 = part * 4u, own_row = own / HEADS;
				const float2 a0 = *(const float2 *)(score_parts + own * TILE + k0);
				const float2 a1 = *(const float2 *)(score_parts + own * TILE + k0 + 2u);
				const float2 b0 = *(const float2 *)(score_parts + (QUERIES + own) * TILE + k0);
				const float2 b1 = *(const float2 *)(score_parts + (QUERIES + own) * TILE + k0 + 2u);
				float v0 = ((key_rows[buffer * TILE + k0] >> own_row) & 1u) != 0u ? a0.x + b0.x : -INFINITY;
				float v1 = ((key_rows[buffer * TILE + k0 + 1u] >> own_row) & 1u) != 0u ? a0.y + b0.y : -INFINITY;
				float v2 = ((key_rows[buffer * TILE + k0 + 2u] >> own_row) & 1u) != 0u ? a1.x + b1.x : -INFINITY;
				float v3 = ((key_rows[buffer * TILE + k0 + 3u] >> own_row) & 1u) != 0u ? a1.y + b1.y : -INFINITY;
				float peak = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 1u));
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, 2u));
				const float previous_max = running_max[own];
				const float next_max = fmaxf(previous_max, peak);
				const float scale = previous_max == -INFINITY ? 0.0f : SparkMathExp2Fast((previous_max - next_max) * scale_log2);
				const float shift_val = next_max == -INFINITY ? 0.0f : next_max * scale_log2;
				const float w0 = next_max == -INFINITY || v0 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v0, scale_log2, -shift_val));
				const float w1 = next_max == -INFINITY || v1 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v1, scale_log2, -shift_val));
				const float w2 = next_max == -INFINITY || v2 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v2, scale_log2, -shift_val));
				const float w3 = next_max == -INFINITY || v3 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v3, scale_log2, -shift_val));
				*(uint2 *)(weight_high + own * TILE + k0) = make_uint2(LmPackBf16Pair(w0, w1), LmPackBf16Pair(w2, w3));
				float total = (w0 + w1) + (w2 + w3);
				total += __shfl_xor_sync(0xffffffffu, total, 1u);
				total += __shfl_xor_sync(0xffffffffu, total, 2u);
				if ( part == 0u )
				{
					running_sum[own] = fmaf(running_sum[own], scale, total);
					running_max[own] = next_max;
					rescale[own] = scale;
				}
			}
		}
		__syncthreads();
		{
			uint32_t a_high[Q_TILES][4], b[2][4], first[2], second[2];
			#pragma unroll
			for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
			{
				const float lo_s = rescale[qt * 16u + (lane >> 2u)], hi_s = rescale[qt * 16u + (lane >> 2u) + 8u];
				if ( lo_s != 1.0f || hi_s != 1.0f )
				{
					#pragma unroll
					for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
					{
						output[qt][frag][0] *= lo_s;
						output[qt][frag][1] *= lo_s;
						output[qt][frag][2] *= hi_s;
						output[qt][frag][3] *= hi_s;
					}
				}
				const uint32_t a_row = qt * 16u + (matrix & 1u) * 8u + within, a_col = (matrix >> 1u) * 8u;
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * TILE + a_col), a_high[qt]);
			}
			const uint32_t b_row = (matrix & 1u) * 8u + within;
			LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, (warp * WARP_COLUMNS) / 8u + (matrix >> 1u)) * 16u, b[0]);
			#pragma unroll
			for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u )
			{
				const uint32_t cur = (frag >> 1u) & 1u, nxt = cur ^ 1u;
				if ( frag + 2u < COLUMN_FRAGS )
				{
					const uint32_t next_chunk = (warp * WARP_COLUMNS + (frag + 2u) * 8u) / 8u + (matrix >> 1u);
					LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, next_chunk) * 16u, b[nxt]);
				}
				first[0] = b[cur][0]; first[1] = b[cur][1];
				second[0] = b[cur][2]; second[1] = b[cur][3];
				#pragma unroll
				for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
				{
					LmMmaBf16(output[qt][frag], a_high[qt], first);
					LmMmaBf16(output[qt][frag + 1u], a_high[qt], second);
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
	for ( uint32_t qt = 0u; qt < Q_TILES; ++qt )
	{
		const uint32_t q_lo = qt * 16u + (lane >> 2u), q_hi = q_lo + 8u;
		const uint32_t row_lo = first_row + q_lo / HEADS, row_hi = first_row + q_hi / HEADS;
		const float inv_lo = row_lo < row_end && running_sum[q_lo] > 0.0f ? 1.0f / running_sum[q_lo] : 0.0f;
		const float inv_hi = row_hi < row_end && running_sum[q_hi] > 0.0f ? 1.0f / running_sum[q_hi] : 0.0f;
		#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
		{
			const uint32_t col = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
			if ( row_lo < row_end )
				__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_lo * HEADS + q_lo % HEADS) * LATENT + col),
					LmPackBf16Pair(output[qt][frag][0] * inv_lo, output[qt][frag][1] * inv_lo));
			if ( row_hi < row_end )
				__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_hi * HEADS + q_hi % HEADS) * LATENT + col),
					LmPackBf16Pair(output[qt][frag][2] * inv_hi, output[qt][frag][3] * inv_hi));
		}
	}
	LmDependentRelease();
}
#endif

template<class Geometry>
static int32_t K3LayerMlaGatherPack(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	static_assert(Geometry::kSlotBytes == K3_MLA_KV_A_DIM * sizeof(uint16_t), "gathered keys are packed at the latent row width the value expansion reads");
	if ( K3LayerMlaGatherReady(b,rows) == 0u || K3LayerMlaGatherIndexReady(b) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->shard_gather_direct != 0u && b->shard_gather_prior_ctx == 0u )
		return(LM_LAUNCH_OK);
	if ( b->shard_gather_prefetched != 0u )
	{
		if ( !((uint64_t)b->shard_gather_prefetched * b->shard_gather_window_keys < b->shard_gather_most_keys && LmKvShardGatherPackRangeLaunch<Geometry,K3_LAYER_THREADS>(b->cache_shard,b->shard_gather_list,b->shard_gather_offset, b->shard_gather_context,1u,b->shard_gather_prefetched * b->shard_gather_window_keys,b->shard_gather_most_keys, b->shard_gather_send,stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( !(LmKvShardGatherPackLaunch<Geometry,K3_LAYER_THREADS>(b->cache_shard,b->shard_gather_list,b->shard_gather_offset, b->shard_gather_context,1u,b->shard_gather_most_keys,b->shard_gather_send,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Geometry>
static int32_t K3LayerMlaPrefetchPack(const K3LayerBuffers *b, uint32_t rows, const LmKvView &pages, uint32_t windows, SparkGpuStream stream)
{
	LmKvShardView view;
	if ( K3LayerMlaGatherReady(b,rows) == 0u || b->shard_gather_windows <= 1u || windows == 0u || windows > b->shard_gather_history ||
		!LmKvViewIsConfigured(pages) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	view.pages = pages;
	view.shard = b->kv_shard;
	if ( !(LmKvShardGatherPackRangeLaunch<Geometry,K3_LAYER_THREADS>(view,b->shard_gather_list,b->shard_gather_offset, b->shard_gather_context,1u,0u,windows * b->shard_gather_window_keys,b->shard_gather_send,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaExpandValues(const K3LayerBuffers *b, uint32_t rows, uint32_t pass, SparkGpuStream stream)
{
	const uint32_t keys = K3LayerMlaPassKeys(b,pass);
	if ( K3LayerMlaExpandedReady(b,rows) == 0u || pass >= b->shard_gather_passes || keys == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->shard_gather_direct != 0u )
	{
		uint16_t *kv_all = b->shard_query_gathered_bf16;
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		if ( b->shard_gather_prior_ctx != 0u )
		{
			LM_LAUNCH_DEPENDENT((K3UnpackPriorKvKernel), dim3(b->shard_gather_most_keys, b->kv_shard.degree), 128u, 0, stream,
				b->kv_shard, (const uint8_t *)b->shard_partials_received_f32, b->shard_gather_stride,
				b->shard_gather_most_keys, b->shard_gather_prior_ctx, kv_all);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
#endif
		if ( !(LmLatentValueExpandLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_V_HEAD_DIM>((const uint8_t *)kv_all, 0u, 1u,
				b->shard_gather_total_ctx, (const uint16_t *)b->mla_kv_b_value_weight,
				K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS), b->shard_gather_values, 0u, stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( !(LmLatentValueExpandLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_V_HEAD_DIM>(K3LayerMlaPassLanding(b,pass),K3LayerMlaWindowStride(b,pass), b->kv_shard.degree,keys,(const uint16_t *)b->mla_kv_b_value_weight,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS), b->shard_gather_values,K3LayerMlaValueStride(b),stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Geometry>
static int32_t K3LayerMlaGatherPartials(const K3LayerBuffers *b, uint32_t rows, uint32_t window, SparkGpuStream stream)
{
	LmKvShardGatherView view;
	SparkKvShard source;
	const uint32_t expanded = b->shard_gather_expanded;
	const uint8_t *landing = expanded != 0u ? K3LayerMlaPassLanding(b,window) : K3LayerMlaWindowLanding(b,window);
	const uint64_t stride = K3LayerMlaWindowStride(b,window);
	const uint32_t begin = K3LayerMlaPassBegin(b,window), keys = K3LayerMlaPassKeys(b,window);
	const uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	uint32_t rank;
	if ( K3LayerMlaGatherReady(b,rows) == 0u || window >= K3LayerMlaPasses(b) || (keys == 0u && (b->shard_gather_direct == 0u || b->shard_gather_prior_ctx != 0u)) ||
		(expanded != 0u && K3LayerMlaExpandedReady(b,rows) == 0u) || K3LayerMlaGatherIndexReady(b) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->shard_gather_direct != 0u )
	{
		if ( window != 0u || K3LayerMlaPasses(b) != 1u || (heads & 1u) != 0u )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		uint16_t *kv_all = b->shard_query_gathered_bf16;
		if ( expanded != 0u )
		{
			LmKvShardGatherView flat_view;
			SparkKvShard flat_owner = {1u, 0u, 1u};
			if ( LmKvShardGatherViewInitialize<Geometry>(&flat_view, b->cache_shard.pages, flat_owner,
					(const uint8_t *)kv_all, b->shard_gather_offset, b->context_length) != 0 )
				SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
			if ( LmExpandedGatherPrefillLaunch<Geometry,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM,K3_V_HEAD_DIM>(
					flat_view, b->shard_gather_values, b->query_bf16, heads, b->sequence_of_row,
					b->context_length, b->positions, K3_MLA_QK_SCALE, b->shard_partials_received_f32, rows, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			return(LM_LAUNCH_OK);
		}
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		constexpr uint32_t ROWS_PER_BLOCK = 8u;
		constexpr uint32_t SHARED_BYTES = LmPrefillAttnSharedBytes<K3_MLA_KV_A_DIM, LM_PREFILL_ATTN_QUERIES>();
		const uint32_t blocks = (rows + ROWS_PER_BLOCK - 1u) / ROWS_PER_BLOCK;
		if ( b->shard_gather_prior_ctx != 0u )
		{
			LM_LAUNCH_DEPENDENT((K3UnpackPriorKvKernel), dim3(b->shard_gather_most_keys, b->kv_shard.degree), 128u, 0, stream,
				b->kv_shard, (const uint8_t *)b->shard_partials_received_f32, b->shard_gather_stride,
				b->shard_gather_most_keys, b->shard_gather_prior_ctx, kv_all);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		if ( b->index_active != 0u )
		{
			constexpr uint32_t UNION_CAP = ROWS_PER_BLOCK * K3_INDEX_TOPK;
			const uint64_t pos_bytes = (uint64_t)blocks * UNION_CAP * sizeof(uint32_t);
			const uint64_t mask_bytes = (((uint64_t)blocks * UNION_CAP + 15u) & ~UINT64_C(15));
			const uint64_t need_bytes = pos_bytes + mask_bytes + (uint64_t)blocks * sizeof(uint32_t);
			if ( b->index_selected == 0 || need_bytes > b->shard_received_floats * sizeof(float) )
				SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
			uint8_t *union_base = (uint8_t *)b->shard_partials_received_f32;
			uint32_t *union_pos = (uint32_t *)union_base;
			uint8_t *union_masks = union_base + pos_bytes;
			uint32_t *union_counts = (uint32_t *)(union_base + pos_bytes + mask_bytes);
			LM_LAUNCH_DEPENDENT((K3FillDenseSelectedKernel<K3_INDEX_TOPK>), rows, K3_LAYER_THREADS, 0, stream,
				b->sequence_of_row, b->context_length, b->positions, b->index_selected);
			LM_LAUNCH_DEPENDENT((LmPrefillSparseUnionKernel<ROWS_PER_BLOCK, K3_INDEX_TOPK>), blocks, LM_PREFILL_UNION_THREADS, 0, stream,
				b->index_selected, b->positions, rows, (const uint32_t *)0, UNION_CAP, union_pos, union_masks, union_counts);
			if ( heads == 6u && rows >= 256u )
			{
				constexpr uint32_t SHARED6 = K3Prefill6HeadSharedBytes();
				static std::atomic<bool> configured_sparse6(false);
				if ( !configured_sparse6.load(std::memory_order_acquire) )
				{
					if ( SparkGpuFuncSetAttribute((const void *)K3MlaGatherPrefill6HeadKernel,
							SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)SHARED6) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
					configured_sparse6.store(true, std::memory_order_release);
				}
				LM_LAUNCH_DEPENDENT((K3MlaGatherPrefill6HeadKernel), blocks, LM_PREFILL_ATTN_THREADS, SHARED6, stream,
					b->query_bf16, kv_all, b->sequence_of_row, b->context_length, b->positions, K3_MLA_QK_SCALE, b->attention_out_bf16,
					rows, b->shard_gather_total_ctx, union_pos, union_masks, union_counts, UNION_CAP);
			}
			else
			{
				static std::atomic<bool> configured_sparse(false);
				if ( !configured_sparse.load(std::memory_order_acquire) )
				{
					if ( SparkGpuFuncSetAttribute((const void *)K3MlaGatherPrefillKernel<true>,
							SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)SHARED_BYTES) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
					configured_sparse.store(true, std::memory_order_release);
				}
				LM_LAUNCH_DEPENDENT((K3MlaGatherPrefillKernel<true>), dim3(blocks, heads / 2u), LM_PREFILL_ATTN_THREADS, SHARED_BYTES, stream,
					b->query_bf16, kv_all, b->sequence_of_row, b->context_length, b->positions, K3_MLA_QK_SCALE, b->attention_out_bf16,
					rows, heads, b->shard_gather_total_ctx, union_pos, union_masks, union_counts, UNION_CAP);
			}
		}
		else
		{
			LmKvShardGatherAllView all_view;
			SparkKvShard flat_owner = {1u, 0u, 1u};
			if ( LmKvShardGatherAllViewInitialize<Geometry>(&all_view,b->cache_shard.pages,flat_owner,(const uint8_t *)kv_all,0u,
					b->shard_gather_offset,b->context_length) != 0 )
				SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
			if ( LmLatentGatherPrefillLaunch<Geometry,LmKvShardGatherAllView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>(all_view,b->query_bf16,
					heads,b->sequence_of_row,b->context_length,b->positions,K3_MLA_QK_SCALE,b->shard_partials_received_f32,rows,stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			return(LM_LAUNCH_OK);
		}
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
#endif
		return(LM_LAUNCH_OK);
	}
	if ( b->index_active != 0u )
	{
		LmKvShardGatherAllView all_view;
		if ( window != 0u || K3LayerMlaPasses(b) != 1u || b->index_selected == 0 )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( LmKvShardGatherAllViewInitialize<Geometry>(&all_view,b->cache_shard.pages,b->kv_shard,landing,stride,
				b->shard_gather_offset,b->shard_gather_context) != 0 )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( LmLatentShardAttendHeadsLaunch<Geometry,LmKvShardGatherAllView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>(all_view,b->query_bf16,
				b->shard_query_rank_stride,heads,heads,b->sequence_of_row,b->context_length,b->positions,b->index_selected,K3_INDEX_TOPK,K3_INDEX_TOPK,
				K3_MLA_QK_SCALE,b->shard_partials_received_f32,b->shard_partial_rank_stride,heads,1u,rows,stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( expanded != 0u )
	{
		source = b->kv_shard;
		source.rank = 0u;
		if ( LmKvShardGatherViewInitialize<Geometry>(&view,b->cache_shard.pages,source,landing,b->shard_gather_offset,b->shard_gather_context) != 0 ||
			(K3LayerMlaPasses(b) > 1u && LmKvShardGatherViewWindow(&view,begin,begin + keys) != 0) )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( !(LmExpandedGatherPrefillSourcesLaunch<Geometry,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM,K3_V_HEAD_DIM>(view,b->kv_shard.degree,stride, b->shard_gather_values,K3LayerMlaValueStride(b),b->query_bf16,heads,b->sequence_of_row,b->context_length,b->positions, K3_MLA_QK_SCALE,b->shard_partials_received_f32,b->shard_partial_rank_stride,rows,stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( K3LayerMlaPasses(b) == 1u )
	{
		LmKvShardGatherAllView all_view;
		if ( LmKvShardGatherAllViewInitialize<Geometry>(&all_view,b->cache_shard.pages,b->kv_shard,landing,stride,
				b->shard_gather_offset,b->shard_gather_context) != 0 )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( LmLatentGatherPrefillLaunch<Geometry,LmKvShardGatherAllView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>(all_view,b->query_bf16,
				heads,b->sequence_of_row,b->context_length,b->positions,K3_MLA_QK_SCALE,b->shard_partials_received_f32,rows,stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	for ( rank = 0u; rank < b->kv_shard.degree; rank++ )
	{
		source = b->kv_shard;
		source.rank = rank;
		if ( LmKvShardGatherViewInitialize<Geometry>(&view,b->cache_shard.pages,source,landing + (uint64_t)rank * stride,
				b->shard_gather_offset,b->shard_gather_context) != 0 ||
			(K3LayerMlaPasses(b) > 1u && LmKvShardGatherViewWindow(&view,begin,begin + keys) != 0) )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( LmLatentGatherPrefillLaunch<Geometry,LmKvShardGatherView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>(view,b->query_bf16,
				heads,b->sequence_of_row,b->context_length,b->positions,K3_MLA_QK_SCALE,
				b->shard_partials_received_f32 + (uint64_t)rank * b->shard_partial_rank_stride,rows,stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaShardMerge(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	if ( K3LayerMlaShardReady(b,rows) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( !(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(b->shard_partials_received_f32,b->shard_partial_rank_stride, b->kv_shard.degree,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->attention_out_bf16,rows,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

#define K3_MHA_QK (K3_QK_NOPE_DIM + K3_QK_UNROTATED_DIM)
#define K3_MHA_KEY_ELEMENTS ((uint64_t)K3_MLA_HEADS * (K3_MHA_QK + K3_V_HEAD_DIM))

template<class Geometry>
struct K3MhaExpandedKv
{
	static constexpr uint32_t kSlotBytes = K3_MLA_HEADS * (K3_MHA_QK + K3_V_HEAD_DIM) * 2u;
	static constexpr uint32_t kPageSlots = Geometry::kPageSlots;
	static constexpr uint32_t kPageBytes = kSlotBytes * Geometry::kPageSlots;
	static constexpr bool kGrows = Geometry::kGrows;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return Geometry::PageOf(position); }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return Geometry::SlotInPage(position); }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return Geometry::PagesForTokens(tokens); }
};

#define K3_MHA_PARTIAL_UNIT_FLOATS ((uint64_t)K3_HIDDEN * sizeof(uint16_t) / sizeof(float))

static uint64_t K3LayerMhaPartialStride(const K3LayerBuffers *b, uint32_t rows)
{
	const uint64_t floats = (uint64_t)rows * LmGqaShardRecordFloats(K3_MLA_HEADS,K3_V_HEAD_DIM,b->kv_shard.degree);
	return((floats + K3_MHA_PARTIAL_UNIT_FLOATS - 1u) / K3_MHA_PARTIAL_UNIT_FLOATS * K3_MHA_PARTIAL_UNIT_FLOATS);
}

static uint32_t K3LayerMhaReady(const K3LayerBuffers *b, uint32_t rows)
{
	const uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	return(b->shard_mha != 0u && b->mla_q_b_weight != 0 && b->mla_kv_b_key_full_weight != 0 && b->mla_kv_b_value_full_weight != 0 &&
		K3LayerMlaShardReady(b,rows) != 0u && b->shard_mha_keys != 0 && b->shard_mha_query_bf16 != 0 && b->shard_gather_send != 0 &&
		b->shard_gather_offset != 0 && b->shard_gather_context != 0 && heads * b->kv_shard.degree == K3_MLA_HEADS &&
		b->shard_query_rank_stride >= (uint64_t)rows * heads * K3_MHA_QK &&
		b->shard_partial_rank_stride >= K3LayerMhaPartialStride(b,rows) &&
		b->shard_mha_keys_bytes >= (uint64_t)b->shard_gather_most_keys * K3_MHA_KEY_ELEMENTS * 2u ? 1u : 0u);
}

template<class Format, class Geometry>
static int32_t K3LayerMhaUp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	if ( K3LayerIndexed(b) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_b_weight,0,b->query_bf16,rows,K3_Q_LORA_RANK,
		K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS) * K3_MHA_QK,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((K3RmsNormKvShardStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->cache_shard,b->kv_slot_bf16,(const uint16_t *)b->mla_kv_a_norm_weight,b->sequence_of_row,b->slots != 0 ? b->slots : b->positions,rows,K3_LORA_RMS_EPSILON);
	return(LM_LAUNCH_OK);
}

template<class Geometry>
static int32_t K3LayerMhaPartials(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	typedef K3MhaExpandedKv<Geometry> Expanded;
	const uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	const uint64_t key_pitch = K3_MHA_KEY_ELEMENTS;
	LmKvShardGatherView view;
	if ( K3LayerMhaReady(b,rows) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmRankMajorRowsKernel<K3_LAYER_THREADS>), dim3((K3_MLA_HEADS * K3_MHA_QK + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS, LmLaunchGridRows(rows)), K3_LAYER_THREADS, 0, stream,
		b->shard_query_gathered_bf16,b->shard_mha_query_bf16,rows,heads * K3_MHA_QK,b->kv_shard.degree);
	if ( LmKvShardGatherPackLaunch<Geometry,K3_LAYER_THREADS>(b->cache_shard,b->shard_gather_list,b->shard_gather_offset,b->shard_gather_context,1u,
			b->shard_gather_most_keys,b->shard_gather_send,stream) != SPARK_GPU_SUCCESS ||
		LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_QK_NOPE_DIM>(b->shard_gather_send,b->shard_gather_most_keys,
			(const uint16_t *)b->mla_kv_b_key_full_weight,K3_MLA_HEADS,b->shard_mha_keys,key_pitch,K3_MHA_QK,stream) != SPARK_GPU_SUCCESS ||
		LmLatentRopeBroadcastLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM>(b->shard_gather_send,b->shard_gather_most_keys,K3_MLA_HEADS,b->shard_mha_keys,
			key_pitch,K3_MHA_QK,K3_QK_NOPE_DIM,stream) != SPARK_GPU_SUCCESS ||
		LmLatentExpandPitchedLaunch<K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_V_HEAD_DIM>(b->shard_gather_send,b->shard_gather_most_keys,
			(const uint16_t *)b->mla_kv_b_value_full_weight,K3_MLA_HEADS,b->shard_mha_keys + (uint64_t)K3_MLA_HEADS * K3_MHA_QK * 2u,key_pitch,K3_V_HEAD_DIM,stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	if ( LmKvShardGatherViewInitialize<Expanded>(&view,b->cache_shard.pages,b->kv_shard,b->shard_mha_keys,b->shard_gather_offset,b->shard_gather_context) != 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( !(LmGqaShardPartialLaunch<Expanded,LmKvShardGatherView,K3_MLA_HEADS,K3_MHA_QK,K3_V_HEAD_DIM>(view,b->shard_mha_query_bf16,K3_MLA_HEADS, b->sequence_of_row,b->context_length,b->positions,K3_MLA_QK_SCALE,b->shard_partials_f32,K3LayerMhaPartialStride(b,rows),rows,stream,(const uint32_t *)0,0u,1u,0u,b->fork_positions,b->slot_offsets) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMhaMerge(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	if ( K3LayerMhaReady(b,rows) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( !(LmGqaShardMergeLaunch<K3_V_HEAD_DIM>(b->shard_partials_received_f32,K3LayerMhaPartialStride(b,rows),b->kv_shard.degree,K3_MLA_HEADS, b->kv_shard.rank,b->value_bf16,rows,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaGatherMerge(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	if ( b->shard_gather_direct != 0u && b->index_active != 0u )
		return(LM_LAUNCH_OK);
	if ( b->shard_gather_expanded == 0u )
	{
		if ( K3LayerMlaShardReady(b,rows) == 0u || K3LayerMlaPasses(b) != 1u )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( !(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(b->shard_partials_received_f32,b->shard_partial_rank_stride,
				1u,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->attention_out_bf16,rows,stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( K3LayerMlaExpandedReady(b,rows) == 0u || b->shard_gather_passes != 1u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( !(LmLatentShardMergeLaunch<K3_V_HEAD_DIM>(b->shard_partials_received_f32,b->shard_partial_rank_stride,
			b->shard_gather_direct != 0u ? 1u : b->kv_shard.degree,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->value_bf16,rows,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaShardFold(const K3LayerBuffers *b, uint32_t rows, uint32_t window, SparkGpuStream stream)
{
	const uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	if ( K3LayerMlaGatherReady(b,rows) == 0u || K3LayerMlaPasses(b) <= 1u || window >= K3LayerMlaPasses(b) || b->shard_gather_running == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->shard_gather_expanded != 0u )
	{
		if ( !(LmLatentShardFoldLaunch<K3_V_HEAD_DIM>(b->shard_partials_received_f32,b->shard_partial_rank_stride, b->kv_shard.degree,heads,b->shard_gather_running,rows,window == 0u ? 1u : 0u,stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( !(LmLatentShardFoldLaunch<K3_KV_LORA_RANK>(b->shard_partials_received_f32,b->shard_partial_rank_stride, b->kv_shard.degree,heads,b->shard_gather_running,rows,window == 0u ? 1u : 0u,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMlaShardFinish(const K3LayerBuffers *b, uint32_t rows, SparkGpuStream stream)
{
	const uint32_t heads = K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS);
	if ( K3LayerMlaGatherReady(b,rows) == 0u || K3LayerMlaPasses(b) <= 1u || b->shard_gather_running == 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( b->shard_gather_expanded != 0u )
	{
		if ( !(LmLatentShardMergeLaunch<K3_V_HEAD_DIM>(b->shard_gather_running,(uint64_t)rows * heads * LM_LATENT_SHARD_RECORD_FLOATS(K3_V_HEAD_DIM), 1u,heads,b->value_bf16,rows,stream) == SPARK_GPU_SUCCESS) )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		return(LM_LAUNCH_OK);
	}
	if ( !(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(b->shard_gather_running,(uint64_t)rows * heads * LM_LATENT_SHARD_RECORD_FLOATS(K3_KV_LORA_RANK), 1u,heads,b->attention_out_bf16,rows,stream) == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3LayerMoeSelect(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, SparkGpuStream stream)
{
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( K3_EXPERT_CELLS(b) && b->tp_sharded != 0u && b->expert_interleave != 0u && rows != 0u && packed_rows == rows * K3_TOP_K )
	{
		if ( rows == 1u )
		{
			LM_LAUNCH_DEPENDENT((K3TopkRouteBuildRow1Kernel), 1u, 256u, 0, stream,
				b->router_logits, b->router_bias, b->route_expert, b->route_weight, b->route_fault,
				b->group_row_offset, b->route_packed_row, b->route_source_token, b->group_tile_prefix_w1);
			return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		}
		if ( packed_rows <= 65535u )
		{
			if ( LmTopkRouteLaunch<K3_LAYER_THREADS,K3_TOP_K,true,LM_TOPK_SCORE_SIGMOID>(rows,b->router_logits,K3_EXPERTS,
				b->route_expert,b->route_weight,b->router_bias,0,K3_ROUTED_SCALE,b->route_fault,stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
			LM_LAUNCH_DEPENDENT((K3RouteBuildFastKernel), 1u, 256u, 0, stream,
				b->route_expert, packed_rows, b->group_row_offset, b->route_packed_row,
				b->route_source_token, b->group_tile_prefix_w1);
			return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		}
	}
#endif
	if ( LmTopkRouteLaunch<K3_LAYER_THREADS,K3_TOP_K,true,LM_TOPK_SCORE_SIGMOID>(rows,b->router_logits,K3_EXPERTS,
		b->route_expert,b->route_weight,b->router_bias,0,K3_ROUTED_SCALE,b->route_fault,stream) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w1_out = K3_EXPERT_INTERMEDIATE * 2u;
	return(LmRouteBuild<K3_LAYER_THREADS,K3_EXPERTS>(
		b->route_expert,rows,packed_rows,K3_TOP_K,b->group_row_offset,
		b->route_packed_row,b->route_source_token,w1_out,
		moe_in,K3_LAYER_TILE_N,b->group_tile_prefix_w1,
		b->group_tile_prefix_w2,stream));
}

template<class Format>
static int32_t K3LayerMoeRoute(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	if ( b->normed_ready == 0u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->mlp_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
#endif
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
			b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->mlp_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	}
	if ( K3_EXPERT_CELLS(b) )
	{
		const uint32_t experts = K3_EXPERTS / (K3_ROUTED_EXPERT_HIDDEN / moe_in);
		const uint8_t *slice = (const uint8_t *)b->router_weight + (uint64_t)b->tp_rank * experts * K3_HIDDEN * sizeof(uint16_t);
		if ( rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u )
		{
			status = K3SkinnyRows(slice,b->normed_bf16,(uint16_t *)0,K3_ROUTER_SLICE(b,rows),rows,K3_HIDDEN,experts,0u,stream);
			if ( status == LM_LAUNCH_ERR_SHAPE )
				LM_LAUNCH_REFUSE("k3 router slice: the skinny kernel refused %u rows of this rank's %u x %u router slice; waves of up to %u rows and "
					"row-invariant waves run on it, so the slice and the normed rows need 16-byte alignment\n",rows,experts,K3_HIDDEN,LM_SKINNY_ROWS_WIDE);
			return(status);
		}
		status = LmStreamGemmDense<LmBf16Format>(slice,LmScaleTensorNone(),b->normed_bf16,(uint16_t *)0,K3_ROUTER_SLICE(b,rows),rows,K3_HIDDEN,experts,0u,0u,multiprocessors,stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("k3 router slice: the stream GEMM refused %u rows of this rank's %u x %u router slice; prefill waves run on it, so the "
				"slice must hold an even number of experts and the operands 16-byte alignment\n",rows,experts,K3_HIDDEN);
		return(status);
	}
	if ( rows > LM_SKINNY_ROWS_WIDE && b->row_invariant == 0u )
	{
		status = LmStreamGemmDense<LmBf16Format>(b->router_weight,LmScaleTensorNone(),b->normed_bf16,(uint16_t *)0,b->router_logits,rows,K3_HIDDEN,K3_EXPERTS,0u,0u,multiprocessors,stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("k3 router: the stream GEMM refused %u rows of the %u x %u router; prefill waves run on it, so the operands need 16-byte alignment\n",rows,K3_EXPERTS,K3_HIDDEN);
	}
	else
	{
		status = K3SkinnyRows(b->router_weight,b->normed_bf16,(uint16_t *)0,b->router_logits,rows,K3_HIDDEN,K3_EXPERTS,0u,stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			LM_LAUNCH_REFUSE("k3 router: the skinny kernel refused %u rows of the %u x %u router; waves of up to %u rows and row-invariant waves run on it, "
				"so the router weight and the normed rows need 16-byte alignment\n",rows,K3_EXPERTS,K3_HIDDEN,LM_SKINNY_ROWS_WIDE);
	}
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerMoeSelect(b,rows,packed_rows,stream));
}

template<class Format>
static int32_t K3LayerSharedUp(const K3LayerBuffers *b, uint32_t rows, uint16_t *intermediate,
	uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	const uint32_t w1_rows = K3_RANK_DIM(b,shared_w1_rows,K3_SHARED_INTERMEDIATE * 2u);
	const uint32_t ch = K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( b->shared_w1_scale == 0 && w1_rows == ch * 2u && (ch & 1u) == 0u &&
		LmIsAligned16(b->shared_w1_weight) && LmIsAligned16(b->normed_bf16) && LmIsAligned16(intermediate) )
	{
		if ( rows > 64u && b->row_invariant == 0u )
			return(K3SharedW1SituPrefillMmaLaunch(b->shared_w1_weight, b->normed_bf16, intermediate, rows, ch, stream));
		if ( rows > LM_SKINNY_ROWS_WIDE && rows <= 64u && b->row_invariant == 0u && b->gate_up_bf16 != 0 && (ch % 16u) == 0u )
		{
			const K3SplitKTarget st[1] = {
				{(const uint8_t *)b->shared_w1_weight, (uint16_t *)0, (float *)0, w1_rows, w1_rows, 0u, 0u}
			};
			return(K3SplitKMmaLaunch(st, 1u, b->normed_bf16, (float *)b->gate_up_bf16, rows, K3_HIDDEN, 4u, intermediate, stream));
		}
		if ( rows <= 12u || b->row_invariant != 0u )
		{
			LmWeightPrefetchNote(stream, b->shared_w1_weight, (uint64_t)w1_rows * K3_HIDDEN * sizeof(uint16_t));
			if ( rows == 1u )
				LM_LAUNCH_DEPENDENT((K3SharedW1SituSkinnyKernel<1u>), dim3(ch, 1u), 256u, 0u, stream,
					(const uint16_t *)b->shared_w1_weight, b->normed_bf16, intermediate, rows, ch);
			else if ( rows <= 4u )
				LM_LAUNCH_DEPENDENT((K3SharedW1SituSkinnyKernel<4u>), dim3(ch, 1u), 256u, 0u, stream,
					(const uint16_t *)b->shared_w1_weight, b->normed_bf16, intermediate, rows, ch);
			else if ( rows <= 8u )
				LM_LAUNCH_DEPENDENT((K3SharedW1SituSkinnyKernel<8u>), dim3(ch, 1u), 256u, 0u, stream,
					(const uint16_t *)b->shared_w1_weight, b->normed_bf16, intermediate, rows, ch);
			else
				LM_LAUNCH_DEPENDENT((K3SharedW1SituSkinnyKernel<12u>), dim3(ch, (rows + 11u) / 12u), 256u, 0u, stream,
					(const uint16_t *)b->shared_w1_weight, b->normed_bf16, intermediate, rows, ch);
			return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		}
	}
#endif
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->shared_w1_weight,b->shared_w1_scale,
		b->gate_up_bf16,rows,K3_HIDDEN,w1_rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,intermediate,ch,(uint64_t)ch,
		K3_SITU_BETA,K3_SITU_LINEAR_BETA,1u);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerSharedDown(const K3LayerBuffers *b, uint32_t rows, const uint16_t *intermediate,
	uint32_t multiprocessors, SparkGpuStream stream)
{
	return(K3Project<LmBf16Format>(b,intermediate,b->shared_w2_weight,b->shared_w2_scale,
		b->shared_out_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE),K3_HIDDEN,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerMoeOutput(const K3LayerBuffers *b, uint32_t rows,
	uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	const uint32_t in_dim = K3_RANK_DIM(b,routed_up_input,K3_ROUTED_EXPERT_HIDDEN);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( b->tp_sharded != 0u && b->routed_up_scale == 0 && rows > 64u && b->row_invariant == 0u &&
		(in_dim % 32u) == 0u && LmIsAligned16(b->latent_bf16) && LmIsAligned16(b->routed_up_weight) && LmIsAligned16(b->hidden_bf16) )
	{
		const uint16_t *addend = K3CanFuseRoutedUpShared(b, rows) != 0u ? b->shared_out_bf16 : (const uint16_t *)0;
		status = K3RoutedUpAddPrefillMmaLaunch(b->routed_up_weight, b->latent_bf16, addend, b->hidden_bf16, rows, in_dim, K3_HIDDEN, stream);
		if ( status != LM_LAUNCH_OK || K3_EXPERT_CELLS(b) )
			return(status);
		status = K3LayerSharedUp<Format>(b,rows,b->intermediate_bf16,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		return(K3LayerSharedDown<Format>(b,rows,b->intermediate_bf16,multiprocessors,stream));
	}
#endif
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->routed_up_weight,b->routed_up_scale,
		b->hidden_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,in_dim,K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK || K3_EXPERT_CELLS(b) )
		return(status);
	status = K3LayerSharedUp<Format>(b,rows,b->intermediate_bf16,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerSharedDown<Format>(b,rows,b->intermediate_bf16,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerMoeCells(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, uint32_t multiprocessors, SparkGpuStream stream,
	uint32_t phase)
{
	const uint32_t slice = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t channels = b->expert_w1_output / 2u;
	int32_t status;
	if ( b->tp_sharded == 0u || b->expert_interleave == 0u || b->expert_w2_input != channels )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( phase == 0u )
		return(K3Project<LmBf16Format>(b,b->normed_bf16,b->routed_down_weight,b->routed_down_scale,
			b->latent_bf16,rows,K3_HIDDEN,slice,multiprocessors,stream));
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
	if ( b->expert_tile_k == LM_SKINNY_CELL_TILE_K && b->group_tile_prefix_w1 != 0 &&
		packed_rows <= 65535u && packed_rows == rows * K3_TOP_K &&
		(channels % LM_SKINNY_CELL_TILE_K) == 0u &&
		LmSkinnyAligned(b->expert_w1_weight) != 0u && LmSkinnyAligned(b->expert_w2_weight) != 0u &&
		LmSkinnyAligned(b->latent_full_bf16) != 0u && LmSkinnyAligned(b->gate_up_bf16) != 0u &&
		LmSkinnyAligned(b->intermediate_bf16) != 0u && LmSkinnyAligned(b->latent_bf16) != 0u )
	{
		if ( rows > LM_SKINNY_ROWS_WIDE && b->row_invariant == 0u &&
			(channels % (LM_SKINNY_CELL_NEURONS * 4u)) == 0u )
			return(K3MoeCellsMmaFastLaunch(b->expert_w1_weight, b->expert_w2_weight, b->latent_full_bf16,
				b->gate_up_bf16, b->intermediate_bf16, b->latent_bf16, b->group_row_offset,
				b->route_source_token, b->route_packed_row, b->route_weight, b->group_tile_prefix_w1,
				rows, packed_rows, channels, stream));
		if ( (rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u) && (channels % LM_SKINNY_CELL_NEURONS) == 0u )
			return(K3MoeCellsSkinnyFastLaunch(b->expert_w1_weight, b->expert_w2_weight, b->latent_full_bf16,
				b->gate_up_bf16, b->intermediate_bf16, b->latent_bf16, b->group_row_offset,
				b->route_expert, b->route_source_token, b->route_packed_row, b->route_weight, b->group_tile_prefix_w1,
				rows, packed_rows, channels, stream));
	}
#endif
	status = rows > LM_SKINNY_ROWS_WIDE && b->row_invariant == 0u ?
		LmCellMmaExperts(b->expert_w1_weight,b->latent_full_bf16,b->gate_up_bf16,
			b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,0u,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_w1_output,b->expert_tile_k,stream) :
		LmSkinnyCellExperts(b->expert_w1_weight,b->latent_full_bf16,b->gate_up_bf16,
			b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,rows,0u,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_w1_output,b->expert_tile_k,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,channels,(uint64_t)channels,K3_SITU_BETA,K3_SITU_LINEAR_BETA,1u);
	status = rows > LM_SKINNY_ROWS_WIDE && b->row_invariant == 0u ?
		LmCellMmaExperts(b->expert_w2_weight,b->intermediate_bf16,b->gate_up_bf16,
			b->group_row_offset,0,K3_EXPERTS,packed_rows,1u,channels,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_tile_k,stream) :
		LmSkinnyCellExperts(b->expert_w2_weight,b->intermediate_bf16,b->gate_up_bf16,
			b->group_row_offset,0,K3_EXPERTS,packed_rows,rows,1u,channels,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_tile_k,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3(K3VecGrid(K3_ROUTED_EXPERT_HIDDEN,b->gate_up_bf16,b->latent_bf16),LmLaunchGridRows(rows)), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16,rows,K3_TOP_K,K3_ROUTED_EXPERT_HIDDEN);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerMoeWeighted(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, uint32_t multiprocessors, SparkGpuStream stream,
	uint32_t phase)
{
	LmGemmArguments gemm;
	int32_t status;
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t w1_out = K3_EXPERT_INTERMEDIATE * 2u;
	if ( K3_EXPERT_CELLS(b) && phase < 2u )
		return(K3LayerMoeCells<Format>(b,rows,packed_rows,multiprocessors,stream,phase));
	if ( phase == 0u )
	{
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->routed_down_weight,b->routed_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,moe_in,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = b->gate_up_bf16;
	gemm.source_row_map = b->route_source_token;
	gemm.source_row_count = rows;
	if ( K3ExpertCellKernel(b,moe_in,w1_out) != 0u )
	{
		status = LmSkinnyCellExperts(b->expert_w1_weight,b->latent_bf16,b->gate_up_bf16,
			b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,rows,0u,moe_in,w1_out,b->expert_tile_k,stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			K3_EXPERT_CELL_REFUSE("w1",rows,packed_rows,w1_out,moe_in);
		return(status);
	}
	if ( b->expert_interleave != 0u )
	{
		if ( b->expert_tile_k == 32u )
			status = LmGemmWeightOnlyIndirectInterleavedLaunch<
				Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS,32u>(
				&gemm,b->latent_bf16,b->expert_w1_weight,packed_rows,rows,
				K3_TOP_K,K3_EXPERTS,moe_in,w1_out,
				multiprocessors,stream);
		else
			status = LmGemmWeightOnlyIndirectInterleavedLaunch<
				Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
				&gemm,b->latent_bf16,b->expert_w1_weight,packed_rows,rows,
				K3_TOP_K,K3_EXPERTS,moe_in,w1_out,
				multiprocessors,stream);
	}
	else
		status = LmGemmWeightOnlyIndirectLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->latent_bf16,b->expert_w1_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,moe_in,w1_out,
			multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
		return(LM_LAUNCH_OK);
	}
	if ( phase == 2u )
	{
		if ( b->tp_sharded == 0u )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->routed_norm_weight, (uint64_t)K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t));
#endif
		LM_LAUNCH_DEPENDENT((K3RmsNormSliceKernel<K3_LAYER_THREADS,K3_ROUTED_EXPERT_HIDDEN>), rows, K3_LAYER_THREADS, 0, stream,
			b->latent_full_bf16,(const uint16_t *)b->routed_norm_weight,b->latent_bf16,b->tp_rank * moe_in,moe_in,K3_RMS_EPSILON);
		return(K3LayerMoeOutput<Format>(b,rows,multiprocessors,stream));
	}
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,
		K3_EXPERT_INTERMEDIATE,(uint64_t)K3_EXPERT_INTERMEDIATE,
		K3_SITU_BETA,K3_SITU_LINEAR_BETA,1u);
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = b->gate_up_bf16;
	gemm.source_row_map = 0;
	gemm.source_row_count = 0u;
	const uint32_t w2_in = K3_EXPERT_INTERMEDIATE;
	if ( K3ExpertCellKernel(b,w2_in,moe_in) != 0u )
	{
		status = LmSkinnyCellExperts(b->expert_w2_weight,b->intermediate_bf16,b->gate_up_bf16,
			b->group_row_offset,0,K3_EXPERTS,packed_rows,rows,1u,w2_in,moe_in,b->expert_tile_k,stream);
		if ( status == LM_LAUNCH_ERR_SHAPE )
			K3_EXPERT_CELL_REFUSE("w2",rows,packed_rows,moe_in,w2_in);
	}
	else if ( b->expert_interleave != 0u )
	{
		if ( b->expert_tile_k == 32u )
			status = LmGemmWeightOnlyInterleavedLaunch<
				Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS,32u>(
				&gemm,b->intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
				K3_TOP_K,K3_EXPERTS,w2_in,moe_in,
				multiprocessors,true,stream);
		else
			status = LmGemmWeightOnlyInterleavedLaunch<
				Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
				&gemm,b->intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
				K3_TOP_K,K3_EXPERTS,w2_in,moe_in,
				multiprocessors,true,stream);
	}
	else
		status = LmGemmWeightOnlyLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,w2_in,moe_in,
			multiprocessors,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3(K3VecGrid(moe_in,b->gate_up_bf16,b->latent_bf16),LmLaunchGridRows(rows)), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16, rows,K3_TOP_K,moe_in);
	if ( b->tp_sharded != 0u )
	{
		if ( moe_in == 0u || K3_ROUTED_EXPERT_HIDDEN % moe_in != 0u ||
			b->tp_rank >= K3_ROUTED_EXPERT_HIDDEN / moe_in )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		return(LM_LAUNCH_OK);
	}
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (moe_in + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_bf16,moe_in,moe_in,K3_RMS_EPSILON);
	return(K3LayerMoeOutput<Format>(b,rows,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerLatentMoe(const K3LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, SparkGpuStream stream, uint32_t phase)
{
	int32_t status;
	if ( phase == 0u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE)
		if ( K3_EXPERT_CELLS(b) && b->tp_sharded != 0u && b->expert_interleave != 0u &&
			b->expert_w2_input == b->expert_w1_output / 2u && b->routed_down_scale == 0 )
		{
			const uint32_t slice = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
			const uint32_t experts = K3_EXPERTS / (K3_ROUTED_EXPERT_HIDDEN / slice);
			const uint8_t *rslice = (const uint8_t *)b->router_weight + (uint64_t)b->tp_rank * experts * K3_HIDDEN * sizeof(uint16_t);
#if defined(__CUDACC__)
			if ( rows > 64u && b->row_invariant == 0u &&
				LmIsAligned16(b->routed_down_weight) && LmIsAligned16(rslice) && LmIsAligned16(b->normed_bf16) &&
				LmIsAligned16(b->latent_bf16) && LmIsAligned16(K3_ROUTER_SLICE(b,rows)) &&
				(slice & 1u) == 0u && (experts & 1u) == 0u )
			{
				if ( b->normed_ready == 0u )
				{
					LmWeightPrefetchNote(stream, b->mlp_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
					LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
						b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->mlp_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
				}
				return(K3MoeDownRoutePrefillMmaLaunch(b->routed_down_weight, rslice, b->normed_bf16,
					b->latent_bf16, K3_ROUTER_SLICE(b,rows), rows, slice, experts, stream));
			}
#endif
			if ( rows <= 64u || b->row_invariant != 0u )
			{
				if ( b->normed_ready == 0u )
				{
#if defined(__CUDACC__)
					LmWeightPrefetchNote(stream, b->mlp_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
#endif
					LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
						b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->mlp_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
				}
#if defined(__CUDACC__)
				if ( rows > LM_SKINNY_ROWS_WIDE && rows <= 64u && b->row_invariant == 0u && b->gate_up_bf16 != 0 &&
					LmIsAligned16(b->routed_down_weight) && LmIsAligned16(rslice) && LmIsAligned16(b->normed_bf16) &&
					(slice & 1u) == 0u && (experts & 1u) == 0u )
				{
					const K3SplitKTarget st[2] = {
						{(const uint8_t *)b->routed_down_weight, b->latent_bf16, (float *)0, slice, slice, 0u, 0u},
						{rslice, (uint16_t *)0, K3_ROUTER_SLICE(b,rows), experts, experts, 0u, 0u}
					};
					return(K3SplitKMmaLaunch(st, 2u, b->normed_bf16, (float *)b->gate_up_bf16, rows, K3_HIDDEN, 8u, (uint16_t *)0, stream));
				}
#endif
				if ( rows <= LM_SKINNY_ROWS_WIDE || b->row_invariant != 0u )
				{
					const LmSkinnyDenseTarget targets[2] = {
						{b->routed_down_weight, b->latent_bf16, (float *)0, slice, slice, 0u, LmScaleTensorNone()},
						{rslice, (uint16_t *)0, K3_ROUTER_SLICE(b,rows), experts, experts, 0u, LmScaleTensorNone()}};
#if defined(__CUDACC__)
					status = rows > LM_SKINNY_ROWS_WIDE && b->row_invariant != 0u ?
						K3ExactSplitMultiLaunch(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream) :
						LmSkinnyDenseMulti<LmBf16Format>(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant,1u);
#else
					status = LmSkinnyDenseMulti<LmBf16Format>(targets,2u,b->normed_bf16,rows,K3_HIDDEN,stream,b->row_invariant,1u);
#endif
					if ( status == LM_LAUNCH_ERR_SHAPE )
						LM_LAUNCH_REFUSE("k3 moe route+down: the multi-target skinny kernel refused %u rows; waves of up to %u rows and row-invariant waves "
							"run on it, so the weights and activation need 16-byte alignment\n",rows,LM_SKINNY_ROWS_WIDE);
					return(status);
				}
			}
		}
#endif
		status = K3LayerMoeRoute<Format>(b,rows,packed_rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(K3LayerMoeWeighted<Format>(b,rows,packed_rows,multiprocessors,stream,phase));
}

template<class Format>
static int32_t K3LayerDenseMlpBody(const K3LayerBuffers *b, uint32_t rows, uint16_t *destination, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status;
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->dense_gate_up_weight,
		b->dense_gate_up_scale,b->gate_up_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,dense_gate_up_rows,K3_DENSE_INTERMEDIATE * 2u),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,
		K3_RANK_DIM(b,dense_down_input,K3_DENSE_INTERMEDIATE),(uint64_t)K3_RANK_DIM(b,dense_down_input,K3_DENSE_INTERMEDIATE),
		K3_SITU_BETA,K3_SITU_LINEAR_BETA,1u);
	return(K3Project<LmBf16Format>(b,b->intermediate_bf16,b->dense_down_weight,
		b->dense_down_scale,destination,
		b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,dense_down_input,K3_DENSE_INTERMEDIATE),K3_HIDDEN,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerDenseMlp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, SparkGpuStream stream)
{
	if ( b->normed_ready == 0u )
	{
#if !defined(LM_HOST_CUDA_PRELUDE) && defined(__CUDACC__)
		LmWeightPrefetchNote(stream, b->mlp_norm_weight, (uint64_t)K3_HIDDEN * sizeof(uint16_t));
#endif
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
			b->hidden_bf16,(const uint16_t *)0,(const uint16_t *)b->mlp_norm_weight, (uint16_t *)0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	}
	return(K3LayerDenseMlpBody<Format>(b,rows,b->hidden_bf16,multiprocessors,stream));
}

static int32_t K3Head(const K3LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, SparkGpuStream stream)
{
	uint32_t tiles = (vocabulary + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<K3_LAYER_THREADS,K3_HEAD_TILE>), dim3(tiles,rows), K3_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,K3_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	if ( !(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS) )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t K3ProjectionShapeFits(const char *name, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( input_dimension == 0u || output_dimension == 0u || input_dimension % K3_TILED_INPUT_MULTIPLE != 0u )
		LM_LAUNCH_REFUSE("k3 projection %s: this rank's %u x %u slice has an input that is not a nonzero multiple of %u, so neither the skinny "
			"kernel nor a prefill GEMM can run it; reshard the pack\n",name,output_dimension,input_dimension,K3_TILED_INPUT_MULTIPLE);
	if ( input_dimension % K3_STREAM_INPUT_MULTIPLE == 0u && output_dimension % 2u != 0u )
		LM_LAUNCH_REFUSE("k3 projection %s: this rank's %u x %u slice takes the stream GEMM for prefill, which needs an even output; reshard the pack\n",
			name,output_dimension,input_dimension);
	return(LM_LAUNCH_OK);
}

static int32_t K3ProjectionShapes(const K3LayerBuffers *b)
{
	static_assert(K3_HIDDEN % K3_STREAM_INPUT_MULTIPLE == 0u && K3_Q_LORA_RANK % K3_STREAM_INPUT_MULTIPLE == 0u &&
		K3_KDA_KEY_DIM % K3_STREAM_INPUT_MULTIPLE == 0u, "the projections from the hidden row, the query latent and the KDA key run on the stream GEMM");
	static_assert(K3_KDA_KEY_DIM % 2u == 0u && K3_Q_LORA_RANK % 2u == 0u && K3_MLA_KV_A_DIM % 2u == 0u && K3_INDEX_DIM % 2u == 0u &&
		K3_INDEX_QUERY % 2u == 0u && K3_INDEX_HEADS % 2u == 0u, "the stream GEMM writes even outputs");
	const struct { const char *name; uint32_t input_dimension, output_dimension; } shapes[] = {
		{"kda_qkv_beta", K3_HIDDEN, K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS)},
		{"kda_gate", K3_HIDDEN, K3_RANK_DIM(b,kda_gate_rows,K3_KDA_V_DIM)},
		{"kda_decay_up", K3_KDA_KEY_DIM, K3_RANK_DIM(b,kda_decay_up_rows,K3_KDA_QK_DIM)},
		{"kda_out", K3_RANK_DIM(b,kda_out_input,K3_KDA_V_DIM), K3_HIDDEN},
		{"mla_q_up", K3_Q_LORA_RANK, K3_RANK_DIM(b,mla_q_up_rows,K3_MLA_Q_DIM)},
		{"mla_gate", K3_HIDDEN, K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM)},
		{"mla_out", K3_RANK_DIM(b,mla_out_input,K3_MLA_OUT_DIM), K3_HIDDEN},
		{"mla_q_b", K3_Q_LORA_RANK, K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS) * K3_MHA_QK},
		{"shared_w1", K3_HIDDEN, K3_RANK_DIM(b,shared_w1_rows,K3_SHARED_INTERMEDIATE * 2u)},
		{"shared_w2", K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE), K3_HIDDEN},
		{"routed_up", K3_RANK_DIM(b,routed_up_input,K3_ROUTED_EXPERT_HIDDEN), K3_HIDDEN},
		{"routed_down", K3_HIDDEN, K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN)},
		{"dense_gate_up", K3_HIDDEN, K3_RANK_DIM(b,dense_gate_up_rows,K3_DENSE_INTERMEDIATE * 2u)},
		{"dense_down", K3_RANK_DIM(b,dense_down_input,K3_DENSE_INTERMEDIATE), K3_HIDDEN}};
	for ( uint32_t index = 0u; index < sizeof(shapes) / sizeof(shapes[0]); index++ )
	{
		const int32_t status = K3ProjectionShapeFits(shapes[index].name,shapes[index].input_dimension,shapes[index].output_dimension);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}
