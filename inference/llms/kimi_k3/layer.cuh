#pragma once
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_lm_certified_launch.h"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/llms/kimi_k3/config.h"
#include "inference/llms/kimi_k3/generated_config.h"

using K3GlobalKv = LmKvLatent<K3_KV_BITS, K3_KV_LORA_RANK, K3_QK_UNROTATED_DIM, K3_KV_PAGE_SLOTS>;


#include "inference/llms/kimi_k3/launch_shape.h"

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
	const void *mla_kv_b_scale;
	const void *mla_gate_weight;
	const void *mla_out_weight;
	const void *mla_out_scale;

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
	uint16_t *fused_decay_gate_bf16;
	uint16_t *gate_latent_bf16;
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
	uint16_t *shard_query_gathered_bf16;
	float *shard_partials_f32;
	float *shard_partials_received_f32;
	uint64_t shard_query_rank_stride;
	uint64_t shard_partial_rank_stride;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *route_expert;
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
};

#define K3_EXPERT_CELLS(b) \
	((b)->expert_w1_output != 0u && (b)->expert_w1_output < K3_EXPERT_INTERMEDIATE * 2u)

#define K3_ROUTER_SLICE(b,rows) \
	((float *)((b)->latent_bf16 + (uint64_t)(rows) * K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN)))

static int32_t K3SkinnyRows(const void *weight, const uint16_t *source, uint16_t *destination_bf16, float *destination_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	uint32_t first, count;
	int32_t status = LM_LAUNCH_OK;
	for ( first = 0u; first < rows && status == LM_LAUNCH_OK; first += count )
	{
		count = rows - first < LM_SKINNY_ROWS_WIDE ? rows - first : LM_SKINNY_ROWS_WIDE;
		status = LmSkinnyDense<LmBf16Format>(weight,source + (uint64_t)first * input_dimension,
			destination_bf16 != 0 ? destination_bf16 + (uint64_t)first * output_dimension : (uint16_t *)0,
			destination_f32 != 0 ? destination_f32 + (uint64_t)first * output_dimension : (float *)0,
			count,input_dimension,output_dimension,0u,0u,stream);
	}
	return(status);
}

template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint16_t *accumulate, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	static_assert(Format::kScaleGroup == 0u,
		"K3Project carries the unquantised projections; experts go weight-only");
	if (weight_scale != 0)
		return(LM_LAUNCH_ERR_SHAPE);
	if ( accumulate == 0 && rows <= LM_SKINNY_ROWS_WIDE )
	{
		int32_t status = K3SkinnyRows(weight,source,destination,(float *)0,rows,input_dimension,output_dimension,stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return(status);
	}
	if constexpr ( LmStreamWeight<Format>::kSupported )
		if ( accumulate == 0 )
		{
			int32_t status = LmStreamGemmDense<Format>(weight,LmScaleTensorNone(),source,destination,(float *)0,rows,input_dimension,output_dimension,0u,0u,multiprocessors,stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return(status);
		}
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = destination;
	gemm.accumulate_bf16 = accumulate;
	return(LmGemmLaunchTileK<Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,source,weight,rows,rows,1u,1u,
		input_dimension,output_dimension,multiprocessors,false,stream));
}

template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	return(K3Project<Format>(b,source,weight,weight_scale,destination,(uint16_t *)0,rows,input_dimension,output_dimension,multiprocessors,stream));
}

static void K3AttnRes(const K3LayerBuffers *b, const void *score_weight, uint32_t sources, uint32_t rows, cudaStream_t stream)
{
	if ( sources > K3_ATTNRES_MAX_SOURCES )
		sources = K3_ATTNRES_MAX_SOURCES;
	if ( b->attnres_score_f32 != 0 )
	{
		LM_LAUNCH((LmAttnResScoreKernel<K3_LAYER_THREADS>), dim3(rows,sources), K3_LAYER_THREADS, 0, stream,
			b->attnres_bank_bf16,b->attnres_partial_bf16,(const uint16_t *)score_weight,b->attnres_score_f32,
			sources,rows,K3_HIDDEN,K3_ATTNRES_MAX_SOURCES,K3_RMS_EPSILON);
		LM_LAUNCH((LmAttnResMixKernel<K3_LAYER_THREADS,K3_ATTNRES_MAX_SOURCES>),
			dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->attnres_bank_bf16,b->attnres_partial_bf16,b->attnres_score_f32,b->hidden_bf16,
			sources,rows,K3_HIDDEN,K3_ATTNRES_MAX_SOURCES);
		return;
	}
	LM_LAUNCH((LmAttnResKernel<K3_LAYER_THREADS,K3_ATTNRES_MAX_SOURCES>),
		rows, K3_LAYER_THREADS, 0, stream,
		b->attnres_bank_bf16,b->attnres_partial_bf16,
		(const uint16_t *)score_weight,b->hidden_bf16,sources,rows,K3_HIDDEN,
		K3_RMS_EPSILON);
}

inline void K3PartialSet(const K3LayerBuffers *b, const uint16_t *value, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		value,b->attnres_partial_bf16,rows,K3_HIDDEN);
}

inline void K3PartialAdd(const K3LayerBuffers *b, const uint16_t *value, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,value,b->attnres_partial_bf16,rows,K3_HIDDEN);
}

static void K3BankStore(const K3LayerBuffers *b, uint32_t slot, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		b->attnres_partial_bf16,
		b->attnres_bank_bf16 + ((uint64_t)slot * rows * K3_HIDDEN),
		rows,K3_HIDDEN);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void K3LatentScatterKernel(const uint16_t *__restrict__ shard_bf16, uint16_t *__restrict__ full_bf16, uint32_t shard_dim, uint32_t shard_offset, uint32_t full_dim)
{
	uint32_t row = blockIdx.y;
	uint32_t column = (blockIdx.x * THREADS) + threadIdx.x;
	if ( column >= full_dim )
		return;
	full_bf16[((uint64_t)row * full_dim) + column] =
		(column >= shard_offset && column < shard_offset + shard_dim)
			? shard_bf16[((uint64_t)row * shard_dim) + (column - shard_offset)] : (uint16_t)0u;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void K3LatentSliceKernel(const uint16_t *__restrict__ full_bf16, uint16_t *__restrict__ shard_bf16, uint32_t shard_dim, uint32_t shard_offset, uint32_t full_dim)
{
	uint32_t row = blockIdx.y;
	uint32_t column = (blockIdx.x * THREADS) + threadIdx.x;
	if ( column >= shard_dim )
		return;
	shard_bf16[((uint64_t)row * shard_dim) + column] =
		full_bf16[((uint64_t)row * full_dim) + shard_offset + column];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void K3SplitFusedProjectionsKernel(const uint16_t *__restrict__ qkvb_bf16, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint16_t *__restrict__ beta_bf16, uint32_t rows, uint32_t qk_dim, uint32_t v_dim, uint32_t heads, uint32_t fused_rows)
{
	uint32_t row = blockIdx.x,index;
	const uint32_t k_offset = qk_dim;
	const uint32_t v_offset = 2u * qk_dim;
	const uint32_t beta_offset = v_offset + v_dim;
	uint64_t fused = (uint64_t)row * fused_rows;
	uint64_t dense = (uint64_t)row * qk_dim;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < qk_dim; index += THREADS)
		query_bf16[dense + index] = qkvb_bf16[fused + index];
	for (index = threadIdx.x; index < qk_dim; index += THREADS)
		key_bf16[dense + index] = qkvb_bf16[fused + k_offset + index];
	for (index = threadIdx.x; index < v_dim; index += THREADS)
		value_bf16[((uint64_t)row * v_dim) + index] =
			qkvb_bf16[fused + v_offset + index];
	for (index = threadIdx.x; index < heads; index += THREADS)
		beta_bf16[((uint64_t)row * heads) + index] =
			qkvb_bf16[fused + beta_offset + index];
}

template<class Format>
static int32_t K3LayerKda(const K3LayerBuffers *b, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	uint32_t state_slot_bytes;
	const uint32_t rank_heads = K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS);
	const uint32_t rank_qk = rank_heads * K3_KDA_KEY_DIM;
	const uint32_t rank_v = rank_heads * K3_KDA_VALUE_DIM;
	const uint32_t head_base = b->tp_rank * rank_heads;
	if ( b->kda_state_bf16 != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( head_base + rank_heads > K3_KDA_HEADS )
		return(LM_LAUNCH_ERR_SHAPE);
	state_slot_bytes = (uint32_t)K3_KDA_RANK_STATE_SLOT_BYTES(rank_heads, 0u);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_qkv_beta_weight,0,
		b->fused_qkvb_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_decay_down_weight,0,
		b->latent_bf16,rows,K3_HIDDEN,K3_KDA_KEY_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->kda_gate_weight,0,
		b->gate_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,kda_gate_rows,K3_KDA_V_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((K3SplitFusedProjectionsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->fused_qkvb_bf16,
		b->query_bf16,b->key_bf16,b->value_bf16,(uint16_t *)b->kda_beta_logit,
		rows,rank_qk,rank_v,rank_heads,
		K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS));
	if ( b->replay_conv_q != 0 )
	{
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((rank_qk + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->query_bf16,b->replay_conv_q,rows,rank_qk);
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((rank_qk + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->key_bf16,b->replay_conv_k,rows,rank_qk);
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((rank_v + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->value_bf16,b->replay_conv_v,rows,rank_v);
	}
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(rank_qk + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_q_window,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->kda_q_conv_weight,b->query_bf16,rank_qk,sequences,commit,b->sequence_row_indices);
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(rank_qk + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_k_window,b->kda_state_index,b->sequence_row_begin,0,b->key_bf16,b->kda_k_conv_weight,b->key_bf16,rank_qk,sequences,commit,b->sequence_row_indices);
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(rank_v + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_v_window,b->kda_state_index,b->sequence_row_begin,0,b->value_bf16,b->kda_v_conv_weight,b->value_bf16,rank_v,sequences,commit,b->sequence_row_indices);
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,rank_heads), K3_LAYER_THREADS, 0, stream,
		b->query_bf16,rank_heads,rows,K3_RMS_EPSILON);
	LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,rank_heads), K3_LAYER_THREADS, 0, stream,
		b->key_bf16,rank_heads,rows,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->kda_decay_up_weight,0,
		b->decay_logit_bf16,rows,K3_KDA_KEY_DIM,
		K3_RANK_DIM(b,kda_decay_up_rows,K3_KDA_QK_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	float *retention = b->replay_retention != 0
		? b->replay_retention : b->kda_retention;
	LM_LAUNCH((LmBoundedDecayKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(rows,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS)), K3_LAYER_THREADS, 0, stream,
		b->decay_logit_bf16,b->kda_decay_bias + ((uint64_t)head_base * K3_KDA_KEY_DIM),b->kda_head_log_scale + head_base,retention,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),K3_KDA_GATE_LOWER_BOUND,rows);
	float *write_gate = b->replay_write_gate != 0
		? b->replay_write_gate : b->kda_write_gate_out;
	LM_LAUNCH((LmSigmoidRowsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		(const uint16_t *)b->kda_beta_logit,write_gate,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS));
	LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM,float,K3_KDA_DELTA_COLUMNS>), dim3(sequences,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),K3_KDA_VALUE_DIM / K3_KDA_DELTA_COLUMNS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_DELTA_COLUMNS * sizeof(float)), stream,
		b->kda_state_pool,state_slot_bytes,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->key_bf16, b->value_bf16,retention,write_gate,b->attention_out_bf16, K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),1u,sequences,commit,b->sequence_row_indices);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,float>), dim3(rows * K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS)), K3_LAYER_THREADS, (K3_KDA_VALUE_DIM + 8u) * sizeof(float), stream,
		b->attention_out_bf16,0,b->kda_out_norm_weight,0,b->attention_out_bf16,K3_KDA_VALUE_DIM,K3_KDA_VALUE_DIM,K3_RMS_EPSILON);
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->gate_bf16,rank_v);
	return(K3Project<LmBf16Format>(b,b->attention_out_bf16,b->kda_out_weight,b->kda_out_scale,
		b->hidden_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : partial_accumulate,
		rows,K3_RANK_DIM(b,kda_out_input,K3_KDA_V_DIM),K3_HIDDEN,multiprocessors,stream));
}

#define K3_MLA_DOWN_SLICED(b, rows) \
	((b)->tp_sharded != 0u && (b)->kv_shard.degree > 1u && (rows) <= LM_SKINNY_ROWS_WIDE && \
	(b)->mla_q_down_scale == 0 && (b)->mla_kv_a_scale == 0 && \
	(K3_Q_LORA_RANK % (b)->kv_shard.degree) == 0u && (K3_MLA_KV_A_DIM % (b)->kv_shard.degree) == 0u)

static int32_t K3LayerMlaDownSlice(const K3LayerBuffers *b, uint32_t rows, cudaStream_t stream)
{
	const uint32_t degree = b->kv_shard.degree, query = K3_Q_LORA_RANK / degree, key = K3_MLA_KV_A_DIM / degree;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = LmSkinnyDense<LmBf16Format>((const uint8_t *)b->mla_q_down_weight + (uint64_t)b->tp_rank * query * K3_HIDDEN * sizeof(uint16_t),
		b->normed_bf16,b->latent_bf16,(float *)0,rows,K3_HIDDEN,query,query + key,0u,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(LmSkinnyDense<LmBf16Format>((const uint8_t *)b->mla_kv_a_weight + (uint64_t)b->tp_rank * key * K3_HIDDEN * sizeof(uint16_t),
		b->normed_bf16,b->latent_bf16,(float *)0,rows,K3_HIDDEN,key,query + key,query,stream));
}

template<class Format>
static int32_t K3LayerMlaDown(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_q_down_weight,b->mla_q_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_Q_LORA_RANK,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_kv_a_weight,b->mla_kv_a_scale,
		b->kv_slot_bf16,rows,K3_HIDDEN,K3_MLA_KV_A_DIM,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LayerMlaUp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_up_weight,b->mla_q_up_scale,
		b->query_bf16,rows,K3_Q_LORA_RANK,
		K3_RANK_DIM(b,mla_q_up_rows,K3_MLA_Q_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	if ( b->kv_shard.degree > 1u )
		LM_LAUNCH((LmKvShardStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
			b->cache_shard,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, Geometry::kSlotBytes / 2u);
	else
		LM_LAUNCH((LmKvStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
			b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, Geometry::kSlotBytes / 2u);
	return(LM_LAUNCH_OK);
}

template<class Format, class Geometry>
static int32_t K3LayerMlaQuery(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status = K3LayerMlaDown<Format>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerMlaUp<Format,Geometry>(b,rows,multiprocessors,stream));
}

static int32_t K3LayerMlaGate(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	return(K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_gate_weight,0,
		b->gate_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM),multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerMlaOutput(const K3LayerBuffers *b, uint32_t rows, uint16_t *partial_accumulate, uint32_t gate_ready, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmPerHeadProjectSplitKernel<K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_V_HEAD_DIM>), dim3(rows,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),K3_V_HEAD_DIM / LM_PER_HEAD_SPLIT_OUTPUTS), K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,(const uint16_t *)b->mla_kv_b_value_weight, b->value_bf16,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),rows);
	status = gate_ready != 0u ? LM_LAUNCH_OK : K3LayerMlaGate(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->value_bf16,b->gate_bf16,K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM));
	return(K3Project<LmBf16Format>(b,b->value_bf16,b->mla_out_weight,b->mla_out_scale,
		b->attention_out_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : partial_accumulate,
		rows,K3_RANK_DIM(b,mla_out_input,K3_MLA_OUT_DIM),K3_HIDDEN,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LayerMla(const K3LayerBuffers *b, uint32_t rows, uint32_t context, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	if ( b->kv_shard.degree > 1u )
		return(LM_LAUNCH_ERR_SHAPE);
	status = K3LayerMlaQuery<Format,Geometry>(b,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmAttentionDecodeKernel<Geometry,K3_ATTN_THREADS,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>), dim3(rows,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS)), K3_ATTN_THREADS, 0, stream,
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),K3_MLA_QK_SCALE,b->attention_out_bf16,b->positions);
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
static int32_t K3LayerMlaShardPartials(const K3LayerBuffers *b, uint32_t rows, cudaStream_t stream)
{
	if ( K3LayerMlaShardReady(b,rows) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS )
		return(LmLatentShardPrefillLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>(
			b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride,
			K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions,
			K3_MLA_QK_SCALE,b->shard_partials_f32,b->shard_partial_rank_stride,rows,stream) == cudaSuccess
			? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
	return(LmLatentShardPartialLaunch<Geometry,LmKvShardView,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>(
		b->cache_shard,b->shard_query_gathered_bf16,b->shard_query_rank_stride,
		K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->sequence_of_row,b->context_length,b->positions,
		0,0u,0u,K3_MLA_QK_SCALE,b->shard_partials_f32,b->shard_partial_rank_stride,rows,stream) == cudaSuccess
		? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t K3LayerMlaShardMerge(const K3LayerBuffers *b, uint32_t rows, cudaStream_t stream)
{
	if ( K3LayerMlaShardReady(b,rows) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	return(LmLatentShardMergeLaunch<K3_KV_LORA_RANK>(b->shard_partials_received_f32,b->shard_partial_rank_stride,
		b->kv_shard.degree,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),b->attention_out_bf16,rows,stream) == cudaSuccess
		? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t K3LayerMoeSelect(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, cudaStream_t stream)
{
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	if ( LmTopkRouteLaunch<K3_LAYER_THREADS,K3_TOP_K,true,LM_TOPK_SCORE_SIGMOID>(rows,b->router_logits,K3_EXPERTS,
		b->route_expert,b->route_weight,b->router_bias,0,K3_ROUTED_SCALE,stream) != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w1_out = K3_EXPERT_INTERMEDIATE * 2u;
	return(LmRouteBuild<K3_LAYER_THREADS,K3_EXPERTS>(
		b->route_expert,rows,packed_rows,K3_TOP_K,b->group_row_offset,
		b->route_packed_row,b->route_source_token,w1_out,
		moe_in,K3_LAYER_TILE_N,b->group_tile_prefix_w1,
		b->group_tile_prefix_w2,stream));
}

template<class Format>
static int32_t K3LayerMoeRoute(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	if ( K3_EXPERT_CELLS(b) )
	{
		const uint32_t experts = K3_EXPERTS / (K3_ROUTED_EXPERT_HIDDEN / moe_in);
		const uint8_t *slice = (const uint8_t *)b->router_weight + (uint64_t)b->tp_rank * experts * K3_HIDDEN * sizeof(uint16_t);
		if ( rows <= LM_SKINNY_ROWS_WIDE )
			return(K3SkinnyRows(slice,b->normed_bf16,(uint16_t *)0,K3_ROUTER_SLICE(b,rows),rows,K3_HIDDEN,experts,stream));
		status = LmStreamGemmDense<LmBf16Format>(slice,LmScaleTensorNone(),b->normed_bf16,(uint16_t *)0,K3_ROUTER_SLICE(b,rows),rows,K3_HIDDEN,experts,0u,0u,multiprocessors,stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return(status);
		memset(&gemm,0,sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = LmScaleTensorNone();
		gemm.group_row_offset = b->dense_row_offset;
		gemm.group_tile_prefix = b->dense_tile_prefix;
		gemm.output_f32 = K3_ROUTER_SLICE(b,rows);
		return(LmGemmLaunch<LmBf16Format,K3_LAYER_TILE_N,LmBf16Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->normed_bf16,slice,rows,rows,1u,1u,
			K3_HIDDEN,experts,multiprocessors,false,stream));
	}
	memset(&gemm,0,sizeof(gemm));
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_f32 = b->router_logits;
	status = K3SkinnyRows(b->router_weight,b->normed_bf16,(uint16_t *)0,b->router_logits,rows,K3_HIDDEN,K3_EXPERTS,stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmGemmLaunch<LmBf16Format,K3_LAYER_TILE_N,LmBf16Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->normed_bf16,b->router_weight,rows,rows,1u,1u,
			K3_HIDDEN,K3_EXPERTS,multiprocessors,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerMoeSelect(b,rows,packed_rows,stream));
}

template<class Format>
static int32_t K3LayerSharedUp(const K3LayerBuffers *b, uint32_t rows, uint16_t *intermediate,
	uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->shared_w1_weight,b->shared_w1_scale,
		b->gate_up_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,shared_w1_rows,K3_SHARED_INTERMEDIATE * 2u),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,intermediate,
		K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE),
		K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerSharedDown(const K3LayerBuffers *b, uint32_t rows, const uint16_t *intermediate,
	uint32_t multiprocessors, cudaStream_t stream)
{
	return(K3Project<LmBf16Format>(b,intermediate,b->shared_w2_weight,b->shared_w2_scale,
		b->shared_out_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE),K3_HIDDEN,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerMoeOutput(const K3LayerBuffers *b, uint32_t rows,
	uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->routed_up_weight,b->routed_up_scale,
		b->hidden_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,routed_up_input,K3_ROUTED_EXPERT_HIDDEN),K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK || K3_EXPERT_CELLS(b) )
		return(status);
	status = K3LayerSharedUp<Format>(b,rows,b->intermediate_bf16,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerSharedDown<Format>(b,rows,b->intermediate_bf16,multiprocessors,stream));
}


template<class Format>
static int32_t K3LayerMoeCells(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream,
	uint32_t phase)
{
	const uint32_t slice = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t channels = b->expert_w1_output / 2u;
	int32_t status;
	if ( b->tp_sharded == 0u || b->expert_interleave == 0u || b->expert_w2_input != channels )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( phase == 0u )
		return(K3Project<LmBf16Format>(b,b->normed_bf16,b->routed_down_weight,b->routed_down_scale,
			b->latent_bf16,rows,K3_HIDDEN,slice,multiprocessors,stream));
	status = rows > LM_SKINNY_ROWS_WIDE ?
		LmCellMmaExperts(b->expert_w1_weight,b->latent_full_bf16,b->gate_up_bf16,
			b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,0u,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_w1_output,b->expert_tile_k,stream) :
		LmSkinnyCellExperts(b->expert_w1_weight,b->latent_full_bf16,b->gate_up_bf16,
			b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,rows,0u,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_w1_output,b->expert_tile_k,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,channels,K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	status = rows > LM_SKINNY_ROWS_WIDE ?
		LmCellMmaExperts(b->expert_w2_weight,b->intermediate_bf16,b->gate_up_bf16,
			b->group_row_offset,0,K3_EXPERTS,packed_rows,1u,channels,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_tile_k,stream) :
		LmSkinnyCellExperts(b->expert_w2_weight,b->intermediate_bf16,b->gate_up_bf16,
			b->group_row_offset,0,K3_EXPERTS,packed_rows,rows,1u,channels,
			K3_ROUTED_EXPERT_HIDDEN,b->expert_tile_k,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3((K3_ROUTED_EXPERT_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16,rows,K3_TOP_K,K3_ROUTED_EXPERT_HIDDEN);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t K3LayerMoeWeighted(const K3LayerBuffers *b, uint32_t rows,
	uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream,
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
	status = b->expert_interleave != 0u ? LmSkinnyCellExperts(b->expert_w1_weight,b->latent_bf16,b->gate_up_bf16,
		b->group_row_offset,b->route_source_token,K3_EXPERTS,packed_rows,rows,0u,moe_in,w1_out,b->expert_tile_k,stream) : LM_LAUNCH_ERR_SHAPE;
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return(status);
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
			return(LM_LAUNCH_ERR_SHAPE);
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float), stream,
			b->latent_full_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_full_bf16,K3_ROUTED_EXPERT_HIDDEN,K3_ROUTED_EXPERT_HIDDEN,K3_RMS_EPSILON);
		LM_LAUNCH((K3LatentSliceKernel<K3_LAYER_THREADS>), dim3((moe_in + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
			b->latent_full_bf16,b->latent_bf16,moe_in,b->tp_rank * moe_in,K3_ROUTED_EXPERT_HIDDEN);
		return(K3LayerMoeOutput<Format>(b,rows,multiprocessors,stream));
	}
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), packed_rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,
		K3_EXPERT_INTERMEDIATE,
		K3_SITU_BETA,K3_SITU_LINEAR_BETA);
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
	status = b->expert_interleave != 0u ? LmSkinnyCellExperts(b->expert_w2_weight,b->intermediate_bf16,b->gate_up_bf16,
		b->group_row_offset,0,K3_EXPERTS,packed_rows,rows,1u,w2_in,moe_in,b->expert_tile_k,stream) : LM_LAUNCH_ERR_SHAPE;
	if ( status == LM_LAUNCH_ERR_SHAPE && b->expert_interleave != 0u )
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
	else if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmGemmWeightOnlyLaunch<
			Format,K3_LAYER_TILE_N,K3_LAYER_STAGES,K3_LAYER_WARPS>(
			&gemm,b->intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
			K3_TOP_K,K3_EXPERTS,w2_in,moe_in,
			multiprocessors,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3((moe_in + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16, rows,K3_TOP_K,moe_in);
	if ( b->tp_sharded != 0u )
	{
		if ( moe_in == 0u || K3_ROUTED_EXPERT_HIDDEN % moe_in != 0u ||
			b->tp_rank >= K3_ROUTED_EXPERT_HIDDEN / moe_in )
			return(LM_LAUNCH_ERR_SHAPE);
		return(LM_LAUNCH_OK);
	}
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (moe_in + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_bf16,moe_in,moe_in,K3_RMS_EPSILON);
	return(K3LayerMoeOutput<Format>(b,rows,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerLatentMoe(const K3LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream, uint32_t phase)
{
	int32_t status;
	if ( phase == 0u )
	{
		status = K3LayerMoeRoute<Format>(b,rows,packed_rows,multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(K3LayerMoeWeighted<Format>(b,rows,packed_rows,multiprocessors,stream,phase));
}

template<class Format>
static int32_t K3LayerDenseMlp(const K3LayerBuffers *b, uint32_t rows, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->dense_gate_up_weight,
		b->dense_gate_up_scale,b->gate_up_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,dense_gate_up_rows,K3_DENSE_INTERMEDIATE * 2u),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,
		K3_RANK_DIM(b,dense_down_input,K3_DENSE_INTERMEDIATE),
		K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	return(K3Project<LmBf16Format>(b,b->intermediate_bf16,b->dense_down_weight,
		b->dense_down_scale,b->hidden_bf16,
		b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,dense_down_input,K3_DENSE_INTERMEDIATE),K3_HIDDEN,multiprocessors,stream));
}

static int32_t K3Head(const K3LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<K3_LAYER_THREADS,K3_HEAD_TILE>), dim3(tiles,rows), K3_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,K3_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ uint32_t K3HeadOrderedScore(float score)
{
	uint32_t bits;
	if ( score != score )
		return 0u;
	bits = __float_as_uint(score == 0.0f ? 0.0f : score);
	return bits ^ ((bits & 0x80000000u) != 0u ? 0xFFFFFFFFu : 0x80000000u);
}

static __device__ __forceinline__ float K3HeadOrderedScoreInverse(uint32_t ordered)
{
	return __uint_as_float(ordered ^ ((ordered & 0x80000000u) != 0u ?
		0x80000000u : 0xFFFFFFFFu));
}

__global__ static void K3HeadMaxlocPackKernel(const float *scores,
	const uint32_t *tokens, uint64_t *maxloc, uint32_t rows)
{
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u )
		maxloc[row] = ((uint64_t)K3HeadOrderedScore(scores[row]) << 32u) |
			(uint64_t)(0xFFFFFFFFu - tokens[row]);
}

__global__ static void K3HeadMaxlocUnpackKernel(const uint64_t *maxloc,
	uint32_t *tokens, float *scores, uint32_t rows)
{
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u )
	{
		tokens[row] = 0xFFFFFFFFu - (uint32_t)maxloc[row];
		scores[row] = K3HeadOrderedScoreInverse((uint32_t)(maxloc[row] >> 32u));
	}
}

static int32_t K3HeadMaxlocPack(const float *scores,
	const uint32_t *tokens, uint64_t *maxloc, uint32_t rows,
	cudaStream_t stream)
{
	if ( rows == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH((K3HeadMaxlocPackKernel), dim3(1u,rows), K3_LAYER_THREADS, 0,
		stream, scores,tokens,maxloc,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t K3HeadMaxlocUnpack(const uint64_t *maxloc,
	uint32_t *tokens, float *scores, uint32_t rows, cudaStream_t stream)
{
	if ( rows == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH((K3HeadMaxlocUnpackKernel), dim3(1u,rows), K3_LAYER_THREADS, 0,
		stream, maxloc,tokens,scores,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

__global__ static void K3HeadRankTokenKernel(uint32_t *tokens,
	uint32_t rows, uint32_t rank_offset)
{
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u && tokens[row] != 0xFFFFFFFFu )
		tokens[row] += rank_offset;
}

static int32_t K3HeadRankSlice(const K3LayerBuffers *b, const void *head_norm_weight,
	const void *head_weight, uint32_t rank_offset, uint32_t vocabulary,
	uint32_t rows, cudaStream_t stream)
{
	int32_t status = K3Head(b,head_norm_weight,head_weight,0,vocabulary,rows,stream);
	if ( status != LM_LAUNCH_OK || rank_offset == 0u || rows == 0u )
		return(status);
	LM_LAUNCH((K3HeadRankTokenKernel), dim3(1u,rows), K3_LAYER_THREADS, 0,
		stream, b->output_token,rows,rank_offset);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<uint32_t THREADS>
__global__ static void K3EmbeddingKernel(const uint16_t *embed_weight,
	const uint32_t *token_ids,uint16_t *hidden_bf16,
	uint32_t vocab_slice_offset,uint32_t vocab_slice_rows)
{
	const uint32_t row = blockIdx.y;
	const uint32_t token = token_ids[row];
	const uint32_t local = token - vocab_slice_offset;
	const uint16_t *src = local < vocab_slice_rows
		? embed_weight + ((uint64_t)local * K3_HIDDEN) : 0;
	for ( uint32_t k = (blockIdx.x * THREADS) + threadIdx.x;
		k < K3_HIDDEN; k += gridDim.x * THREADS )
		hidden_bf16[((uint64_t)row * K3_HIDDEN) + k] =
			src != 0 ? src[k] : 0u;
}

static int32_t K3Embedding(const uint16_t *embed_weight,
	const uint32_t *token_ids,uint16_t *hidden_bf16,uint32_t rows,
	uint32_t vocab_slice_offset,uint32_t vocab_slice_rows,cudaStream_t stream)
{
	const uint32_t columns = (K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS;
	LM_LAUNCH((K3EmbeddingKernel<K3_LAYER_THREADS>), dim3(columns,rows),
		K3_LAYER_THREADS, 0, stream,
		embed_weight,token_ids,hidden_bf16,vocab_slice_offset,vocab_slice_rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}


static int32_t K3HeadCertifiedRows(
    const K3LayerBuffers *b,
    const void *head_norm_weight,
    const void *head_weight,
    const uint8_t *certified_payload,
    const float *certified_scale,
    const float *certified_norm,
    void *certified_scratch,
    uint32_t *candidate_ids,
    uint32_t *screened_count,
    uint32_t rank_offset,
    uint32_t vocabulary,
    uint32_t rows,
    cudaStream_t stream)
{
    cudaError_t status;
    if (b == 0 || head_norm_weight == 0 || head_weight == 0 ||
        certified_payload == 0 || certified_scale == 0 ||
        certified_norm == 0 || certified_scratch == 0 ||
        candidate_ids == 0 || screened_count == 0 ||
        b->hidden_bf16 == 0 || b->normed_bf16 == 0 ||
        b->output_token == 0 || b->output_score == 0)
        return LM_LAUNCH_ERR_SHAPE;
    LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>),
        rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
        b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,
        b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
    status = SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(
        stream, b->normed_bf16, head_weight, certified_payload,
        certified_scale, certified_norm, certified_scratch, candidate_ids,
        screened_count, b->output_token, b->output_score,
        rank_offset, rows, vocabulary, K3_HIDDEN);
    return status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
