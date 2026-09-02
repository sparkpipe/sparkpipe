#pragma once
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/head.cuh"
#include "inference/llms/qwen_3_6/config.h"
#include "inference/kernels/kv.cuh"

using Qwen38_27bFullKv = LmKvHeads<QWEN38_27B_KV_BITS, QWEN38_27B_KV_HEADS, QWEN38_27B_HEAD_DIM, QWEN38_27B_KV_PAGE_SLOTS>;

static_assert(Qwen38_27bFullKv::kSlotBytes == QWEN38_27B_KV_HEADS * (QWEN38_27B_HEAD_DIM + QWEN38_27B_HEAD_DIM) * 2u,
	"the GQA slot is [K: heads x head_dim][V: heads x value_dim] bf16");

#ifndef QWEN38_27B_LAYER_THREADS
#define QWEN38_27B_LAYER_THREADS 256u
#endif
#define QWEN38_27B_LAYER_TILE_N 128u
#define QWEN38_27B_LAYER_STAGES 2u
#define QWEN38_27B_LAYER_WARPS 8u
#define QWEN38_27B_HEAD_TILE 1024u

#define QWEN38_27B_Q_DIM (QWEN38_27B_ATTN_HEADS * QWEN38_27B_HEAD_DIM)
#define QWEN38_27B_KV_DIM (QWEN38_27B_KV_HEADS * QWEN38_27B_HEAD_DIM)
#define QWEN38_27B_ATTN_QG_DIM (2u * QWEN38_27B_Q_DIM)
#define QWEN38_27B_ATTN_QKV_DIM (QWEN38_27B_ATTN_QG_DIM + (2u * QWEN38_27B_KV_DIM))

#define QWEN38_27B_GDN_QK_DIM (QWEN38_27B_GDN_KEY_HEADS * QWEN38_27B_GDN_KEY_DIM)
#define QWEN38_27B_GDN_V_DIM (QWEN38_27B_GDN_VALUE_HEADS * QWEN38_27B_GDN_VALUE_DIM)
#define QWEN38_27B_GDN_QKV_DIM ((2u * QWEN38_27B_GDN_QK_DIM) + QWEN38_27B_GDN_V_DIM)
#define QWEN38_27B_GDN_VALUE_PER_KEY (QWEN38_27B_GDN_VALUE_HEADS / QWEN38_27B_GDN_KEY_HEADS)

static_assert(QWEN38_27B_GDN_VALUE_HEADS % QWEN38_27B_GDN_KEY_HEADS == 0u,
	"value heads share key heads in whole groups");
static_assert(QWEN38_27B_ATTN_QKV_DIM == QWEN38_27B_QKV_DIM,
	"config and layer must agree on the fused projection width");
static_assert(QWEN38_27B_ATTN_OUTPUT_GATE == 1u,
	"the layer applies the attention output gate; an ungated config is a different model");

using Qwen38_27bGdnState = LmKvState<QWEN38_27B_GDN_STATE_BYTES>;

static_assert(QWEN38_27B_GDN_STATE_ELEMENT_BYTES == sizeof(float),
	"LmDeltaRuleKernel addresses the state pool as float; a narrower slot is out of bounds");

struct Qwen38_27bLayerBuffers
{
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *gdn_in_weight;
	const void *gdn_in_scale;
	const void *gdn_conv_weight;
	const void *gdn_out_weight;
	const void *gdn_out_scale;
	const void *gdn_beta_weight;
	const void *gdn_beta_scale;
	const void *gdn_decay_weight;
	const void *gdn_decay_scale;
	const float *gdn_a_log;
	const float *gdn_dt_bias;
	const void *mlp_norm_weight;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *fused_qkv_bf16;
	uint16_t *query_gate_bf16;
	uint16_t *query_bf16;
	uint16_t *attn_gate_bf16;
	uint16_t *key_bf16;
	uint16_t *value_bf16;
	uint16_t *attention_out_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;

	uint8_t *gdn_state_pool;
	uint16_t *gdn_conv_window;
	uint16_t *gdn_query_expanded_bf16;
	uint16_t *gdn_key_expanded_bf16;
	uint16_t *gdn_beta_logit_bf16;
	uint16_t *gdn_decay_logit_bf16;
	const uint32_t *gdn_state_index;
	float *gdn_forget_gate;
	float *gdn_write_gate;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};

template<class Format>
static const void *Qwen38_27bPrepareInput(
    const Qwen38_27bLayerBuffers *buffers,
    const uint16_t *source_bf16,
    uint32_t row_count,
    uint32_t input_dimension,
    LmScaleTensor *scale_out,
    cudaStream_t stream)
{
    if constexpr (Format::kScaleGroup == 0u)
    {
        *scale_out = LmScaleTensorNone();
        return source_bf16;
    }
    else
    {
        LM_LAUNCH(
            (LmQuantiseRowsKernel<Format, QWEN38_27B_LAYER_THREADS>),
            dim3(row_count, input_dimension / Format::kScaleGroup),
            QWEN38_27B_LAYER_THREADS,
            (Format::kScaleGroup + 8u) * sizeof(float),
            stream,
            source_bf16,
            0,
            buffers->packed_activation,
            buffers->packed_scale,
            row_count,
            input_dimension);
        *scale_out = LmScaleTensorRowsUe4m3(
            buffers->packed_scale,
            row_count,
            input_dimension,
            Format::kScaleGroup);
        return buffers->packed_activation;
    }
}

template<class Format>
static LmScaleTensor Qwen38_27bWeightScale(
    const void *scale_data,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    if constexpr (Format::kScaleGroup == 0u)
        return LmScaleTensorNone();
    else
        return LmScaleTensorBlockF32(
            scale_data,
            1u,
            output_dimension,
            input_dimension,
            Format::kScaleGroup,
            Format::kScaleGroup);
}

template<class Format, class Geometry>
static int32_t Qwen38_27bLayerAttention(const Qwen38_27bLayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	const void *activation;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_27B_LAYER_THREADS,uint16_t>), rows, QWEN38_27B_LAYER_THREADS, (QWEN38_27B_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,QWEN38_27B_HIDDEN,QWEN38_27B_HIDDEN,QWEN38_27B_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	activation = Qwen38_27bPrepareInput<Format>(
		b,b->normed_bf16,rows,QWEN38_27B_HIDDEN,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->qkv_scale,QWEN38_27B_ATTN_QKV_DIM,QWEN38_27B_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->qkv_weight,rows,rows,1u,1u,
		QWEN38_27B_HIDDEN,QWEN38_27B_ATTN_QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	layout.query_dimension = QWEN38_27B_ATTN_QG_DIM;
	layout.key_dimension = QWEN38_27B_KV_DIM;
	layout.value_dimension = QWEN38_27B_KV_DIM;
	layout.rope_dimension = QWEN38_27B_ROPE_DIM;
	layout.head_dimension = QWEN38_27B_HEAD_DIM;
	LM_LAUNCH((LmSplitQkvKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->fused_qkv_bf16,layout,b->query_gate_bf16,b->key_bf16,b->value_bf16,rows,1.0f);
	LM_LAUNCH((LmSplitQueryGateKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->query_gate_bf16,b->query_bf16,b->attn_gate_bf16,QWEN38_27B_ATTN_HEADS,QWEN38_27B_HEAD_DIM,rows);
	LM_LAUNCH((LmRopePerHeadKernel<QWEN38_27B_LAYER_THREADS>), dim3(rows,QWEN38_27B_ATTN_HEADS), QWEN38_27B_LAYER_THREADS, 0, stream,
		b->query_bf16,b->positions,QWEN38_27B_ATTN_HEADS,QWEN38_27B_HEAD_DIM, QWEN38_27B_ROPE_DIM,QWEN38_27B_ROPE_THETA);
	LM_LAUNCH((LmRopePerHeadKernel<QWEN38_27B_LAYER_THREADS>), dim3(rows,QWEN38_27B_KV_HEADS), QWEN38_27B_LAYER_THREADS, 0, stream,
		b->key_bf16,b->positions,QWEN38_27B_KV_HEADS,QWEN38_27B_HEAD_DIM, QWEN38_27B_ROPE_DIM,QWEN38_27B_ROPE_THETA);
	LM_LAUNCH((LmGqaKvStoreKernel<Geometry,QWEN38_27B_LAYER_THREADS,QWEN38_27B_KV_HEADS,QWEN38_27B_HEAD_DIM,QWEN38_27B_HEAD_DIM>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->cache,b->key_bf16,b->value_bf16,b->sequence_of_row,b->positions,rows);
	LM_LAUNCH((LmGqaAttentionDecodeKernel<Geometry,QWEN38_27B_LAYER_THREADS,QWEN38_27B_KV_HEADS,QWEN38_27B_HEAD_DIM,QWEN38_27B_HEAD_DIM>), dim3(rows,QWEN38_27B_ATTN_HEADS), QWEN38_27B_LAYER_THREADS, 0, stream,
		b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,QWEN38_27B_ATTN_HEADS,QWEN38_27B_QK_SCALE,b->attention_out_bf16,0);
	LM_LAUNCH((LmOutputGateKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->attn_gate_bf16,QWEN38_27B_Q_DIM);
	activation = Qwen38_27bPrepareInput<Format>(
		b,b->attention_out_bf16,rows,QWEN38_27B_Q_DIM,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->output_scale,QWEN38_27B_HIDDEN,QWEN38_27B_Q_DIM);
	gemm.output_bf16 = b->attention_out_bf16;
	(void)context;
	return(LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->output_weight,rows,rows,1u,1u,
		QWEN38_27B_Q_DIM,QWEN38_27B_HIDDEN,sms,false,stream));
}

template<class Format>
static int32_t Qwen38_27bLayerLinear(const Qwen38_27bLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	const void *activation;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_27B_LAYER_THREADS,uint16_t>), rows, QWEN38_27B_LAYER_THREADS, (QWEN38_27B_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,QWEN38_27B_HIDDEN,QWEN38_27B_HIDDEN,QWEN38_27B_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	activation = Qwen38_27bPrepareInput<Format>(
		b,b->normed_bf16,rows,QWEN38_27B_HIDDEN,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->gdn_in_scale,QWEN38_27B_GDN_QKV_DIM,QWEN38_27B_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->gdn_in_weight,rows,rows,1u,1u,
		QWEN38_27B_HIDDEN,QWEN38_27B_GDN_QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->gdn_beta_scale,QWEN38_27B_GDN_VALUE_HEADS,QWEN38_27B_HIDDEN);
	gemm.output_bf16 = b->gdn_beta_logit_bf16;
	status = LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->gdn_beta_weight,rows,rows,1u,1u,
		QWEN38_27B_HIDDEN,QWEN38_27B_GDN_VALUE_HEADS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->gdn_decay_scale,QWEN38_27B_GDN_VALUE_HEADS,QWEN38_27B_HIDDEN);
	gemm.output_bf16 = b->gdn_decay_logit_bf16;
	status = LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->gdn_decay_weight,rows,rows,1u,1u,
		QWEN38_27B_HIDDEN,QWEN38_27B_GDN_VALUE_HEADS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmGdnGateKernel<QWEN38_27B_LAYER_THREADS,QWEN38_27B_GDN_KEY_DIM>), dim3(rows,QWEN38_27B_GDN_VALUE_HEADS), QWEN38_27B_LAYER_THREADS, 0, stream,
		b->gdn_decay_logit_bf16,b->gdn_beta_logit_bf16,b->gdn_a_log,b->gdn_dt_bias,b->gdn_forget_gate,b->gdn_write_gate,QWEN38_27B_GDN_VALUE_HEADS,rows);
	layout.query_dimension = QWEN38_27B_GDN_QK_DIM;
	layout.key_dimension = QWEN38_27B_GDN_QK_DIM;
	layout.value_dimension = QWEN38_27B_GDN_V_DIM;
	layout.rope_dimension = 0u;
	layout.head_dimension = QWEN38_27B_GDN_KEY_DIM;
	LM_LAUNCH((LmCausalConvKernel<QWEN38_27B_LAYER_THREADS,QWEN38_27B_GDN_CONV_KERNEL,LM_CONV_SWISH,uint16_t>), dim3(rows,(QWEN38_27B_GDN_QKV_DIM + QWEN38_27B_LAYER_THREADS - 1u) / QWEN38_27B_LAYER_THREADS), QWEN38_27B_LAYER_THREADS, 0, stream,
		b->gdn_conv_window,b->gdn_state_index,0,0,b->fused_qkv_bf16, (const uint16_t *)b->gdn_conv_weight,b->fused_qkv_bf16,QWEN38_27B_GDN_QKV_DIM,rows,1u);
	LM_LAUNCH((LmSplitQkvKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->fused_qkv_bf16,layout,b->query_bf16,b->key_bf16,b->value_bf16,rows,1.0f);
	LM_LAUNCH((LmExpandHeadsKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->query_bf16,b->gdn_query_expanded_bf16,QWEN38_27B_GDN_KEY_HEADS,QWEN38_27B_GDN_KEY_DIM,QWEN38_27B_GDN_VALUE_PER_KEY,rows);
	LM_LAUNCH((LmExpandHeadsKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->key_bf16,b->gdn_key_expanded_bf16,QWEN38_27B_GDN_KEY_HEADS,QWEN38_27B_GDN_KEY_DIM,QWEN38_27B_GDN_VALUE_PER_KEY,rows);
	status = LmKernelSharedMemoryOptIn(
		(const void *)LmDeltaRuleKernel<QWEN38_27B_LAYER_THREADS,QWEN38_27B_GDN_KEY_DIM,QWEN38_27B_GDN_VALUE_DIM>,
		(uint32_t)(QWEN38_27B_GDN_KEY_DIM * QWEN38_27B_GDN_VALUE_DIM * sizeof(float)));
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmDeltaRuleKernel<QWEN38_27B_LAYER_THREADS,QWEN38_27B_GDN_KEY_DIM,QWEN38_27B_GDN_VALUE_DIM>), dim3(rows,QWEN38_27B_GDN_VALUE_HEADS), QWEN38_27B_LAYER_THREADS, (uint32_t)(QWEN38_27B_GDN_KEY_DIM * QWEN38_27B_GDN_VALUE_DIM * sizeof(float)), stream,
		b->gdn_state_pool,QWEN38_27B_GDN_STATE_BYTES,b->gdn_state_index,0,0,b->gdn_query_expanded_bf16,b->gdn_key_expanded_bf16,b->value_bf16, b->gdn_forget_gate,b->gdn_write_gate,b->attention_out_bf16, QWEN38_27B_GDN_VALUE_HEADS,1u,rows,1u);
	activation = Qwen38_27bPrepareInput<Format>(
		b,b->attention_out_bf16,rows,QWEN38_27B_GDN_V_DIM,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->gdn_out_scale,QWEN38_27B_HIDDEN,QWEN38_27B_GDN_V_DIM);
	gemm.output_bf16 = b->attention_out_bf16;
	return(LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->gdn_out_weight,rows,rows,1u,1u,
		QWEN38_27B_GDN_V_DIM,QWEN38_27B_HIDDEN,sms,false,stream));
}

template<class Format>
static int32_t Qwen38_27bLayerDenseMlp(const Qwen38_27bLayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	const void *activation;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_27B_LAYER_THREADS,uint16_t>), rows, QWEN38_27B_LAYER_THREADS, (QWEN38_27B_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,QWEN38_27B_HIDDEN,QWEN38_27B_HIDDEN,QWEN38_27B_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	activation = Qwen38_27bPrepareInput<Format>(
		b,b->normed_bf16,rows,QWEN38_27B_HIDDEN,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->dense_gate_up_scale,QWEN38_27B_FFN_INTERMEDIATE * 2u,QWEN38_27B_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->dense_gate_up_weight,rows,rows,1u,1u,
		QWEN38_27B_HIDDEN,QWEN38_27B_FFN_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSiluMulKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,QWEN38_27B_FFN_INTERMEDIATE,true);
	activation = Qwen38_27bPrepareInput<Format>(
		b,b->intermediate_bf16,rows,QWEN38_27B_FFN_INTERMEDIATE,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38_27bWeightScale<Format>(
		b->dense_down_scale,QWEN38_27B_HIDDEN,QWEN38_27B_FFN_INTERMEDIATE);
	gemm.output_bf16 = b->hidden_bf16;
	return(LmGemmLaunch<Format,QWEN38_27B_LAYER_TILE_N,Format::kTileK,QWEN38_27B_LAYER_STAGES,QWEN38_27B_LAYER_WARPS>(
		&gemm,activation,b->dense_down_weight,rows,rows,1u,1u,
		QWEN38_27B_FFN_INTERMEDIATE,QWEN38_27B_HIDDEN,sms,false,stream));
}

static int32_t Qwen38_27bHead(const Qwen38_27bLayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + QWEN38_27B_HEAD_TILE - 1u) / QWEN38_27B_HEAD_TILE;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_27B_LAYER_THREADS,uint16_t>), rows, QWEN38_27B_LAYER_THREADS, (QWEN38_27B_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,QWEN38_27B_HIDDEN,QWEN38_27B_HIDDEN,QWEN38_27B_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<QWEN38_27B_LAYER_THREADS,QWEN38_27B_HEAD_TILE>), dim3(tiles,rows), QWEN38_27B_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,QWEN38_27B_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<QWEN38_27B_LAYER_THREADS>), rows, QWEN38_27B_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
