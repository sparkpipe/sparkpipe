#pragma once
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_lm_certified_launch.h"
#include "inference/kernels/kv.cuh"
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
	uint16_t *attnres_bank_bf16;
	uint16_t *attnres_partial_bf16;
	const void *attnres_attn_weight;
	const void *attnres_mlp_weight;
	const void *attnres_out_weight;
	uint16_t *kda_beta_logit;
	float *kda_write_gate_out;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	const uint32_t *sequence_row_begin;
	uint16_t *replay_conv_q;
	uint16_t *replay_conv_k;
	uint16_t *replay_conv_v;
	float *replay_retention;
	float *replay_write_gate;

	uint32_t tp_sharded;
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

template<class Format>
static int32_t K3Project(const K3LayerBuffers *b, const uint16_t *source, const void *weight, const void *weight_scale, uint16_t *destination, uint16_t *accumulate, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	LmGemmArguments gemm;
	static_assert(Format::kScaleGroup == 0u,
		"K3Project carries the unquantised projections; experts go weight-only");
	if (weight_scale != 0)
		return(LM_LAUNCH_ERR_SHAPE);
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

static int32_t K3DeltaRuleOptIn(uint32_t shared_bytes)
{
	return(LmKernelSharedMemoryOptIn(
		(const void *)LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>,
		shared_bytes));
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
	if ( b->kda_state_bf16 != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	state_slot_bytes = b->kda_state_bf16 != 0u
		? K3_KDA_STATE_SLOT_BYTES_BF16 : K3_KDA_STATE_SLOT_BYTES;
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
		b->kda_q_window,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->kda_q_conv_weight,b->query_bf16,rank_qk,sequences,commit);
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(rank_qk + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_k_window,b->kda_state_index,b->sequence_row_begin,0,b->key_bf16,b->kda_k_conv_weight,b->key_bf16,rank_qk,sequences,commit);
	LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(rank_v + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
		b->kda_v_window,b->kda_state_index,b->sequence_row_begin,0,b->value_bf16,b->kda_v_conv_weight,b->value_bf16,rank_v,sequences,commit);
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
		b->decay_logit_bf16,b->kda_decay_bias,b->kda_head_log_scale,retention,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),K3_KDA_GATE_LOWER_BOUND,rows);
	float *write_gate = b->replay_write_gate != 0
		? b->replay_write_gate : b->kda_write_gate_out;
	LM_LAUNCH((LmSigmoidRowsKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		(const uint16_t *)b->kda_beta_logit,write_gate,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS));
	status = K3DeltaRuleOptIn((uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)));
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>), dim3(sequences,K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS)), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)), stream,
		b->kda_state_pool,state_slot_bytes,b->kda_state_index,b->sequence_row_begin,0,b->query_bf16,b->key_bf16, b->value_bf16,retention,write_gate,b->attention_out_bf16, K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS),1u,sequences,commit);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,float>), dim3(rows * K3_RANK_DIM(b,kda_heads_rank,K3_KDA_HEADS)), K3_LAYER_THREADS, (K3_KDA_VALUE_DIM + 8u) * sizeof(float), stream,
		b->attention_out_bf16,0,b->kda_out_norm_weight,0,b->attention_out_bf16,K3_KDA_VALUE_DIM,K3_KDA_VALUE_DIM,K3_RMS_EPSILON);
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->gate_bf16,rank_v);
	return(K3Project<LmBf16Format>(b,b->attention_out_bf16,b->kda_out_weight,b->kda_out_scale,
		b->hidden_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : partial_accumulate,
		rows,K3_RANK_DIM(b,kda_out_input,K3_KDA_V_DIM),K3_HIDDEN,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LayerMla(const K3LayerBuffers *b, uint32_t rows, uint32_t context, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->attn_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_q_down_weight,b->mla_q_down_scale,
		b->latent_bf16,rows,K3_HIDDEN,K3_Q_LORA_RANK,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_Q_LORA_RANK + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->mla_q_norm_weight, 0,b->latent_bf16,K3_Q_LORA_RANK,K3_Q_LORA_RANK,K3_LORA_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->mla_q_up_weight,b->mla_q_up_scale,
		b->query_bf16,rows,K3_Q_LORA_RANK,
		K3_RANK_DIM(b,mla_q_up_rows,K3_MLA_Q_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_kv_a_weight,b->mla_kv_a_scale,
		b->kv_slot_bf16,rows,K3_HIDDEN,K3_MLA_KV_A_DIM,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_KV_LORA_RANK + 8u) * sizeof(float), stream,
		b->kv_slot_bf16,0,(const uint16_t *)b->mla_kv_a_norm_weight, 0,b->kv_slot_bf16,K3_KV_LORA_RANK,K3_MLA_KV_A_DIM,K3_LORA_RMS_EPSILON);
	LM_LAUNCH((LmKvStoreKernel<Geometry,K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->cache,b->kv_slot_bf16,b->sequence_of_row,b->positions,rows, Geometry::kSlotBytes / 2u);
	LM_LAUNCH((LmAttentionDecodeKernel<Geometry,K3_ATTN_THREADS,K3_KV_LORA_RANK,K3_QK_UNROTATED_DIM>), dim3(rows,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS)), K3_ATTN_THREADS, 0, stream,
		b->query_bf16,b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),K3_MLA_QK_SCALE,b->attention_out_bf16,b->positions);
	(void)context;
	LM_LAUNCH((LmPerHeadProjectKernel<K3_LAYER_THREADS,K3_KV_LORA_RANK,K3_V_HEAD_DIM>), dim3(rows,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS)), K3_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,(const uint16_t *)b->mla_kv_b_value_weight, b->value_bf16,K3_RANK_DIM(b,mla_heads_rank,K3_MLA_HEADS),rows);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->mla_gate_weight,0,
		b->gate_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmOutputGateKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->value_bf16,b->gate_bf16,K3_RANK_DIM(b,mla_gate_rows,K3_MLA_OUT_DIM));
	return(K3Project<LmBf16Format>(b,b->value_bf16,b->mla_out_weight,b->mla_out_scale,
		b->attention_out_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : partial_accumulate,
		rows,K3_RANK_DIM(b,mla_out_input,K3_MLA_OUT_DIM),K3_HIDDEN,multiprocessors,stream));
}

template<class Format>
static int32_t K3LayerLatentMoe(const K3LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream, uint32_t phase)
{
	LmGemmArguments gemm;
	int32_t status;
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	if ( phase == 0u )
	{
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)b->mlp_norm_weight, 0,b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_f32 = b->router_logits;
	status = LmGemmLaunch<LmBf16Format,K3_LAYER_TILE_N,LmBf16Format::kTileK,K3_LAYER_STAGES,K3_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->router_weight,rows,rows,1u,1u,
		K3_HIDDEN,K3_EXPERTS,multiprocessors,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmTopkSmallKernel<K3_LAYER_THREADS,K3_TOP_K,true,1u,1u,LM_TOPK_SCORE_SIGMOID>), rows, K3_LAYER_THREADS, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream,
		b->router_logits,K3_EXPERTS,b->route_expert,b->route_weight,b->router_bias,0,K3_ROUTED_SCALE);
	const uint32_t w1_out = K3_EXPERT_INTERMEDIATE * 2u;
	status = LmRouteBuild<K3_LAYER_THREADS,K3_EXPERTS>(
		b->route_expert,rows,packed_rows,K3_TOP_K,b->group_row_offset,
		b->route_packed_row,b->route_source_token,w1_out,
		moe_in,K3_LAYER_TILE_N,b->group_tile_prefix_w1,
		b->group_tile_prefix_w2,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
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
	if ( b->expert_interleave != 0u )
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
	LM_LAUNCH((LmMoeFinalizeKernel<K3_LAYER_THREADS>), dim3((moe_in + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->route_packed_row,b->route_weight,b->latent_bf16, rows,K3_TOP_K,moe_in);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t>), rows, K3_LAYER_THREADS, (moe_in + 8u) * sizeof(float), stream,
		b->latent_bf16,0,(const uint16_t *)b->routed_norm_weight, 0,b->latent_bf16,moe_in,moe_in,K3_RMS_EPSILON);
	status = K3Project<LmBf16Format>(b,b->latent_bf16,b->routed_up_weight,b->routed_up_scale,
		b->hidden_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,routed_up_input,K3_ROUTED_EXPERT_HIDDEN),K3_HIDDEN,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3Project<LmBf16Format>(b,b->normed_bf16,b->shared_w1_weight,b->shared_w1_scale,
		b->gate_up_bf16,rows,K3_HIDDEN,
		K3_RANK_DIM(b,shared_w1_rows,K3_SHARED_INTERMEDIATE * 2u),multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSituMulKernel<K3_LAYER_THREADS>), rows, K3_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,
		K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE),
		K3_SITU_BETA,K3_SITU_LINEAR_BETA);
	status = K3Project<LmBf16Format>(b,b->intermediate_bf16,b->shared_w2_weight,b->shared_w2_scale,
		b->shared_out_bf16,b->tp_sharded != 0u ? (uint16_t *)0 : b->attnres_partial_bf16,
		rows,K3_RANK_DIM(b,shared_w2_input,K3_SHARED_INTERMEDIATE),K3_HIDDEN,multiprocessors,stream);
	return(status);
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


static int32_t K3HeadCertifiedB1(
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
        1u, K3_LAYER_THREADS, (K3_HIDDEN + 8u) * sizeof(float), stream,
        b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,
        b->normed_bf16,K3_HIDDEN,K3_HIDDEN,K3_RMS_EPSILON);
    status = SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
        stream, b->normed_bf16, head_weight, certified_payload,
        certified_scale, certified_norm, certified_scratch, candidate_ids,
        screened_count, b->output_token, b->output_score,
        rank_offset, 1u, vocabulary, K3_HIDDEN);
    return status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
