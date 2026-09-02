#pragma once
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/llms/qwen_3_8/config.h"
#include "inference/kernels/kv.cuh"

using Qwen38FullKv = LmKvHeads<QWEN38_KV_BITS, QWEN38_KV_HEADS, QWEN38_HEAD_DIM, QWEN38_KV_PAGE_SLOTS>;

static_assert(Qwen38FullKv::kSlotBytes == QWEN38_KV_HEADS * (QWEN38_HEAD_DIM + QWEN38_HEAD_DIM) * 2u,
	"the GQA slot is [K: heads x head_dim][V: heads x value_dim] bf16");

#ifndef QWEN38_LAYER_THREADS
#define QWEN38_LAYER_THREADS 256u
#endif
#define QWEN38_LAYER_TILE_N 128u
#define QWEN38_LAYER_STAGES 2u
#define QWEN38_LAYER_WARPS 8u
#define QWEN38_HEAD_TILE 1024u

#define QWEN38_Q_DIM (QWEN38_ATTN_HEADS * QWEN38_HEAD_DIM)
#define QWEN38_KV_DIM (QWEN38_KV_HEADS * QWEN38_HEAD_DIM)
#define QWEN38_ATTN_QG_DIM (2u * QWEN38_Q_DIM)
#define QWEN38_ATTN_QKV_DIM (QWEN38_ATTN_QG_DIM + (2u * QWEN38_KV_DIM))

#define QWEN38_GDN_QK_DIM (QWEN38_GDN_KEY_HEADS * QWEN38_GDN_KEY_DIM)
#define QWEN38_GDN_V_DIM (QWEN38_GDN_VALUE_HEADS * QWEN38_GDN_VALUE_DIM)
#define QWEN38_GDN_QKV_DIM ((2u * QWEN38_GDN_QK_DIM) + QWEN38_GDN_V_DIM)
#define QWEN38_GDN_VALUE_PER_KEY (QWEN38_GDN_VALUE_HEADS / QWEN38_GDN_KEY_HEADS)

static_assert(QWEN38_GDN_VALUE_HEADS % QWEN38_GDN_KEY_HEADS == 0u,
	"value heads share key heads in whole groups");
static_assert(QWEN38_ATTN_QKV_DIM == QWEN38_QKV_DIM,
	"config and layer must agree on the fused projection width");
static_assert(QWEN38_ATTN_OUTPUT_GATE == 1u,
	"the layer applies the attention output gate; an ungated config is a different model");

using Qwen38GdnState = LmKvState<QWEN38_GDN_STATE_BYTES>;
static_assert(QWEN38_GDN_STATE_ELEMENT_BYTES == sizeof(float),
	"LmDeltaRuleKernel addresses the state pool as float; a narrower slot is out of bounds");


template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void Qwen38HeadRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon)
{
	__shared__ float reduce_scratch[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x;
	uint64_t index = ((uint64_t)row * head_count * head_dim) + ((uint64_t)head * head_dim) + column;
	float value,variance;
	if ( threadIdx.x < THREADS / LM_WARP_LANES )
		reduce_scratch[threadIdx.x] = 0.0f;
	__syncthreads();
	if ( row >= row_count || column >= head_dim )
		return;
	value = LmBf16ToFloat(input_bf16[index]);
	variance = LmBlockSum<THREADS>(value * value,reduce_scratch) / (float)head_dim;
	value = value * rsqrtf(variance + epsilon) * LmBf16ToFloat(weight_bf16[column]);
	output_bf16[index] = LmFloatToBf16(value);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void Qwen38GatedHeadNormKernel(const uint16_t *__restrict__ core_bf16, const uint16_t *__restrict__ z_bf16, const uint16_t *__restrict__ norm_weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon)
{
	__shared__ float reduce_scratch[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x;
	uint64_t index = ((uint64_t)row * head_count * head_dim) + ((uint64_t)head * head_dim) + column;
	float value,variance,z;
	if ( threadIdx.x < THREADS / LM_WARP_LANES )
		reduce_scratch[threadIdx.x] = 0.0f;
	__syncthreads();
	if ( row >= row_count || column >= head_dim )
		return;
	value = LmBf16ToFloat(core_bf16[index]);
	variance = LmBlockSum<THREADS>(value * value,reduce_scratch) / (float)head_dim;
	z = LmBf16ToFloat(z_bf16[index]);
	value = value * rsqrtf(variance + epsilon) * LmBf16ToFloat(norm_weight_bf16[column]) * (z / (1.0f + __expf(-z)));
	output_bf16[index] = LmFloatToBf16(value);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void Qwen38SharedExpertAddKernel(const uint16_t *__restrict__ routed_bf16, const uint16_t *__restrict__ shared_bf16, const uint16_t *__restrict__ gate_weight_bf16, const uint16_t *__restrict__ gate_input_bf16, uint16_t *__restrict__ hidden_out_bf16, uint32_t row_count, uint32_t dimension)
{
	__shared__ float reduce_scratch[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x;
	uint64_t row_base = (uint64_t)row * dimension;
	uint64_t index;
	float logit = 0.0f,routed,shared,gate;
	if ( row >= row_count )
		return;
	if ( threadIdx.x < THREADS / LM_WARP_LANES )
		reduce_scratch[threadIdx.x] = 0.0f;
	__syncthreads();
	for (index = threadIdx.x; index < dimension; index += THREADS)
		logit = fmaf(LmBf16ToFloat(gate_input_bf16[row_base + index]),LmBf16ToFloat(gate_weight_bf16[index]),logit);
	logit = LmBlockSum<THREADS>(logit,reduce_scratch);
	gate = 1.0f / (1.0f + __expf(-logit));
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		routed = LmBf16ToFloat(routed_bf16[row_base + index]);
		shared = LmBf16ToFloat(shared_bf16[row_base + index]);
		hidden_out_bf16[row_base + index] = LmFloatToBf16(routed + (gate * shared));
	}
}

struct Qwen38LayerBuffers
{
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *query_norm_weight;
	const void *key_norm_weight;
	const void *gdn_in_weight;
	const void *gdn_in_scale;
	const void *gdn_conv_weight;
	const void *gdn_out_weight;
	const void *gdn_out_scale;
	const void *gdn_norm_weight;
	const void *gdn_z_weight;
	const void *gdn_z_scale;
	const void *gdn_beta_weight;
	const void *gdn_beta_scale;
	const void *gdn_decay_weight;
	const void *gdn_decay_scale;
	const float *gdn_a_log;
	const float *gdn_dt_bias;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *router_scale;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	const void *shared_gate_up_weight;
	const void *shared_gate_up_scale;
	const void *shared_down_weight;
	const void *shared_down_scale;
	const void *shared_gate_coeff;

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
	uint16_t *gdn_z_logit_bf16;
	uint16_t *gdn_beta_logit_bf16;
	uint16_t *gdn_decay_logit_bf16;
	const uint32_t *gdn_state_index;
	float *gdn_forget_gate;
	float *gdn_write_gate;
	uint16_t *gdn_core_bf16;
	uint16_t *gdn_gated_bf16;

	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint16_t *packed_gate_up_bf16;
	uint16_t *packed_intermediate_bf16;
	uint16_t *packed_down_bf16;
	uint16_t *shared_out_bf16;

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
static const void *Qwen38PrepareInput(
    const Qwen38LayerBuffers *buffers,
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
            (LmQuantiseRowsKernel<Format, QWEN38_LAYER_THREADS>),
            dim3(row_count, input_dimension / Format::kScaleGroup),
            QWEN38_LAYER_THREADS,
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
static LmScaleTensor Qwen38WeightScale(
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

static LmScaleTensor Qwen38ExpertScale(
    const void *scale_data,
    uint32_t group_count,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    return LmScaleTensorBlockF32(
        scale_data,
        group_count,
        output_dimension,
        input_dimension,
        128u,
        128u);
}

template<class Format, class Geometry>
static int32_t Qwen38LayerAttention(const Qwen38LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	const void *activation;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_LAYER_THREADS,uint16_t>), rows, QWEN38_LAYER_THREADS, (QWEN38_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,QWEN38_HIDDEN,QWEN38_HIDDEN,QWEN38_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	activation = Qwen38PrepareInput<Format>(
		b,b->normed_bf16,rows,QWEN38_HIDDEN,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->qkv_scale,QWEN38_ATTN_QKV_DIM,QWEN38_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->qkv_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_ATTN_QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	layout.query_dimension = QWEN38_ATTN_QG_DIM;
	layout.key_dimension = QWEN38_KV_DIM;
	layout.value_dimension = QWEN38_KV_DIM;
	layout.rope_dimension = QWEN38_ROPE_DIM;
	layout.head_dimension = QWEN38_HEAD_DIM;
	LM_LAUNCH((LmSplitQkvKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->fused_qkv_bf16,layout,b->query_gate_bf16,b->key_bf16,b->value_bf16,rows,1.0f);
	LM_LAUNCH((LmSplitQueryGateKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->query_gate_bf16,b->query_bf16,b->attn_gate_bf16,QWEN38_ATTN_HEADS,QWEN38_HEAD_DIM,rows);
	LM_LAUNCH((Qwen38HeadRmsNormKernel<QWEN38_LAYER_THREADS>), dim3(QWEN38_ATTN_HEADS,rows), QWEN38_HEAD_DIM, 0, stream,
		b->query_bf16,(const uint16_t *)b->query_norm_weight,b->query_bf16,rows,QWEN38_ATTN_HEADS,QWEN38_HEAD_DIM,QWEN38_RMS_EPSILON);
	LM_LAUNCH((Qwen38HeadRmsNormKernel<QWEN38_LAYER_THREADS>), dim3(QWEN38_KV_HEADS,rows), QWEN38_HEAD_DIM, 0, stream,
		b->key_bf16,(const uint16_t *)b->key_norm_weight,b->key_bf16,rows,QWEN38_KV_HEADS,QWEN38_HEAD_DIM,QWEN38_RMS_EPSILON);
	LM_LAUNCH((LmRopePerHeadKernel<QWEN38_LAYER_THREADS>), dim3(rows,QWEN38_ATTN_HEADS), QWEN38_LAYER_THREADS, 0, stream,
		b->query_bf16,b->positions,QWEN38_ATTN_HEADS,QWEN38_HEAD_DIM, QWEN38_ROPE_DIM,QWEN38_ROPE_THETA);
	LM_LAUNCH((LmRopePerHeadKernel<QWEN38_LAYER_THREADS>), dim3(rows,QWEN38_KV_HEADS), QWEN38_LAYER_THREADS, 0, stream,
		b->key_bf16,b->positions,QWEN38_KV_HEADS,QWEN38_HEAD_DIM, QWEN38_ROPE_DIM,QWEN38_ROPE_THETA);
	LM_LAUNCH((LmGqaKvStoreKernel<Geometry,QWEN38_LAYER_THREADS,QWEN38_KV_HEADS,QWEN38_HEAD_DIM,QWEN38_HEAD_DIM>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->cache,b->key_bf16,b->value_bf16,b->sequence_of_row,b->positions,rows);
	LM_LAUNCH((LmGqaAttentionDecodeKernel<Geometry,QWEN38_LAYER_THREADS,QWEN38_KV_HEADS,QWEN38_HEAD_DIM,QWEN38_HEAD_DIM>), dim3(rows,QWEN38_ATTN_HEADS), QWEN38_LAYER_THREADS, 0, stream,
		b->query_bf16,b->cache,b->sequence_of_row,b->context_length, 0,0u,QWEN38_ATTN_HEADS,QWEN38_QK_SCALE,b->attention_out_bf16,0);
	LM_LAUNCH((LmOutputGateKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->attn_gate_bf16,QWEN38_Q_DIM);
	activation = Qwen38PrepareInput<Format>(
		b,b->attention_out_bf16,rows,QWEN38_Q_DIM,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->output_scale,QWEN38_HIDDEN,QWEN38_Q_DIM);
	gemm.output_bf16 = b->attention_out_bf16;
	(void)context;
	return(LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->output_weight,rows,rows,1u,1u,
		QWEN38_Q_DIM,QWEN38_HIDDEN,sms,false,stream));
}

template<class Format>
static int32_t Qwen38LayerLinear(const Qwen38LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	const void *activation;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_LAYER_THREADS,uint16_t>), rows, QWEN38_LAYER_THREADS, (QWEN38_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,QWEN38_HIDDEN,QWEN38_HIDDEN,QWEN38_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	activation = Qwen38PrepareInput<Format>(
		b,b->normed_bf16,rows,QWEN38_HIDDEN,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->gdn_in_scale,QWEN38_GDN_QKV_DIM,QWEN38_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->gdn_in_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_GDN_QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->gdn_z_scale,QWEN38_GDN_VALUE_HEADS,QWEN38_HIDDEN);
	gemm.output_bf16 = b->gdn_z_logit_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->gdn_z_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_GDN_VALUE_HEADS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->gdn_beta_scale,QWEN38_GDN_VALUE_HEADS,QWEN38_HIDDEN);
	gemm.output_bf16 = b->gdn_beta_logit_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->gdn_beta_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_GDN_VALUE_HEADS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->gdn_decay_scale,QWEN38_GDN_VALUE_HEADS,QWEN38_HIDDEN);
	gemm.output_bf16 = b->gdn_decay_logit_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->gdn_decay_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_GDN_VALUE_HEADS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmGdnGateKernel<QWEN38_LAYER_THREADS,QWEN38_GDN_KEY_DIM>), dim3(rows,QWEN38_GDN_VALUE_HEADS), QWEN38_LAYER_THREADS, 0, stream,
		b->gdn_decay_logit_bf16,b->gdn_beta_logit_bf16,b->gdn_a_log,b->gdn_dt_bias,b->gdn_forget_gate,b->gdn_write_gate,QWEN38_GDN_VALUE_HEADS,rows);
	layout.query_dimension = QWEN38_GDN_QK_DIM;
	layout.key_dimension = QWEN38_GDN_QK_DIM;
	layout.value_dimension = QWEN38_GDN_V_DIM;
	layout.rope_dimension = 0u;
	layout.head_dimension = QWEN38_GDN_KEY_DIM;
	LM_LAUNCH((LmCausalConvKernel<QWEN38_LAYER_THREADS,QWEN38_GDN_CONV_KERNEL,LM_CONV_SWISH,uint16_t>), dim3(rows,(QWEN38_GDN_QKV_DIM + QWEN38_LAYER_THREADS - 1u) / QWEN38_LAYER_THREADS), QWEN38_LAYER_THREADS, 0, stream,
		b->gdn_conv_window,b->gdn_state_index,0,0,b->fused_qkv_bf16, (const uint16_t *)b->gdn_conv_weight,b->fused_qkv_bf16,QWEN38_GDN_QKV_DIM,rows,1u);
	LM_LAUNCH((LmSplitQkvKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->fused_qkv_bf16,layout,b->query_bf16,b->key_bf16,b->value_bf16,rows,1.0f);
	LM_LAUNCH((LmExpandHeadsKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->query_bf16,b->gdn_query_expanded_bf16,QWEN38_GDN_KEY_HEADS,QWEN38_GDN_KEY_DIM,QWEN38_GDN_VALUE_PER_KEY,rows);
	LM_LAUNCH((LmExpandHeadsKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->key_bf16,b->gdn_key_expanded_bf16,QWEN38_GDN_KEY_HEADS,QWEN38_GDN_KEY_DIM,QWEN38_GDN_VALUE_PER_KEY,rows);
	status = LmKernelSharedMemoryOptIn(
		(const void *)LmDeltaRuleKernel<QWEN38_LAYER_THREADS,QWEN38_GDN_KEY_DIM,QWEN38_GDN_VALUE_DIM>,
		(uint32_t)(QWEN38_GDN_KEY_DIM * QWEN38_GDN_VALUE_DIM * sizeof(float)));
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmDeltaRuleKernel<QWEN38_LAYER_THREADS,QWEN38_GDN_KEY_DIM,QWEN38_GDN_VALUE_DIM>), dim3(rows,QWEN38_GDN_VALUE_HEADS), QWEN38_LAYER_THREADS, (uint32_t)(QWEN38_GDN_KEY_DIM * QWEN38_GDN_VALUE_DIM * sizeof(float)), stream,
		b->gdn_state_pool,QWEN38_GDN_STATE_BYTES,b->gdn_state_index,0,0,b->gdn_query_expanded_bf16,b->gdn_key_expanded_bf16,b->value_bf16, b->gdn_forget_gate,b->gdn_write_gate,b->gdn_core_bf16, QWEN38_GDN_VALUE_HEADS,1u,rows,1u);
	LM_LAUNCH((Qwen38GatedHeadNormKernel<QWEN38_LAYER_THREADS>), dim3(QWEN38_GDN_VALUE_HEADS,rows), QWEN38_LAYER_THREADS, 0, stream,
		b->gdn_core_bf16,b->gdn_z_logit_bf16,(const uint16_t *)b->gdn_norm_weight,b->gdn_gated_bf16,rows,QWEN38_GDN_VALUE_HEADS,QWEN38_GDN_VALUE_DIM,QWEN38_RMS_EPSILON);
	activation = Qwen38PrepareInput<Format>(
		b,b->gdn_gated_bf16,rows,QWEN38_GDN_V_DIM,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->gdn_out_scale,QWEN38_HIDDEN,QWEN38_GDN_V_DIM);
	gemm.output_bf16 = b->attention_out_bf16;
	return(LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->gdn_out_weight,rows,rows,1u,1u,
		QWEN38_GDN_V_DIM,QWEN38_HIDDEN,sms,false,stream));
}

template<class Format>
static int32_t Qwen38LayerMoe(const Qwen38LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	const void *activation;
	uint32_t packed_rows = rows * QWEN38_TOP_K;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_LAYER_THREADS,uint16_t>), rows, QWEN38_LAYER_THREADS, (QWEN38_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,QWEN38_HIDDEN,QWEN38_HIDDEN,QWEN38_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	activation = Qwen38PrepareInput<Format>(
		b,b->normed_bf16,rows,QWEN38_HIDDEN,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->router_scale,QWEN38_EXPERTS,QWEN38_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_f32 = b->router_logits;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->router_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_EXPERTS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmHeadSoftmaxKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, (QWEN38_LAYER_THREADS / LM_WARP_LANES) * sizeof(float), stream,
		b->router_logits,rows,QWEN38_EXPERTS,1.0f);
	LM_LAUNCH((LmTopkSmallKernel<QWEN38_LAYER_THREADS,QWEN38_TOP_K,true,1u,1u,LM_TOPK_SCORE_IDENTITY>), rows, QWEN38_LAYER_THREADS, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream,
		b->router_logits,QWEN38_EXPERTS,b->route_expert,b->route_weight,0,0,1.0f);
	status = LmRouteBuild<QWEN38_LAYER_THREADS,QWEN38_EXPERTS>(
		b->route_expert,rows,packed_rows,QWEN38_TOP_K,b->group_row_offset,
		b->route_packed_row,b->route_source_token,
		QWEN38_EXPERT_INTERMEDIATE * 2u,QWEN38_HIDDEN,QWEN38_LAYER_TILE_N,
		QWEN38_LAYER_TILE_N,b->group_tile_prefix_w1,b->group_tile_prefix_w2,
		stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = Qwen38ExpertScale(
		b->expert_w1_scale,QWEN38_EXPERTS,
		(uint32_t)((uint64_t)QWEN38_EXPERT_INTERMEDIATE * 2u * QWEN38_EXPERTS),QWEN38_HIDDEN);
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = b->packed_gate_up_bf16;
	gemm.source_row_map = b->route_source_token;
	gemm.source_row_count = rows;
	status = LmGemmWeightOnlyIndirectLaunch<
		LmFp8,QWEN38_LAYER_TILE_N,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->expert_w1_weight,packed_rows,rows,
		QWEN38_TOP_K,QWEN38_EXPERTS,
		(uint32_t)((uint64_t)QWEN38_EXPERT_INTERMEDIATE * 2u * QWEN38_EXPERTS),
		QWEN38_HIDDEN,sms,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSiluMulKernel<QWEN38_LAYER_THREADS>), packed_rows, QWEN38_LAYER_THREADS, 0, stream,
		b->packed_gate_up_bf16,b->packed_intermediate_bf16,QWEN38_EXPERT_INTERMEDIATE,true);
	gemm.source_row_map = 0;
	gemm.source_row_count = 0u;
	gemm.scale_b = Qwen38ExpertScale(
		b->expert_w2_scale,QWEN38_EXPERTS,
		(uint32_t)((uint64_t)QWEN38_HIDDEN * QWEN38_EXPERTS),QWEN38_EXPERT_INTERMEDIATE);
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.output_bf16 = b->packed_down_bf16;
	status = LmGemmWeightOnlyLaunch<
		LmFp8,QWEN38_LAYER_TILE_N,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,b->packed_intermediate_bf16,b->expert_w2_weight,packed_rows,rows,
		QWEN38_TOP_K,QWEN38_EXPERTS,(uint32_t)((uint64_t)QWEN38_HIDDEN * QWEN38_EXPERTS),
		QWEN38_EXPERT_INTERMEDIATE,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmCopyRowsKernel<QWEN38_LAYER_THREADS>),
		dim3((QWEN38_HIDDEN + QWEN38_LAYER_THREADS - 1u) / QWEN38_LAYER_THREADS,rows),
		QWEN38_LAYER_THREADS, 0, stream,
		b->attention_out_bf16,b->hidden_bf16,rows,QWEN38_HIDDEN);
	LM_LAUNCH((LmMoeFinalizeKernel<QWEN38_LAYER_THREADS>), dim3((QWEN38_HIDDEN + QWEN38_LAYER_THREADS - 1u) / QWEN38_LAYER_THREADS,rows), QWEN38_LAYER_THREADS, 0, stream,
		b->packed_down_bf16,b->route_packed_row,b->route_weight,b->hidden_bf16, rows,QWEN38_TOP_K,QWEN38_HIDDEN);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->shared_gate_up_scale,QWEN38_EXPERT_INTERMEDIATE * 2u,QWEN38_HIDDEN);
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->shared_gate_up_weight,rows,rows,1u,1u,
		QWEN38_HIDDEN,QWEN38_EXPERT_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSiluMulKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,QWEN38_EXPERT_INTERMEDIATE,true);
	activation = Qwen38PrepareInput<Format>(
		b,b->intermediate_bf16,rows,QWEN38_EXPERT_INTERMEDIATE,&gemm.scale_a,stream);
	gemm.scale_b = Qwen38WeightScale<Format>(
		b->shared_down_scale,QWEN38_HIDDEN,QWEN38_EXPERT_INTERMEDIATE);
	gemm.output_bf16 = b->shared_out_bf16;
	status = LmGemmLaunch<Format,QWEN38_LAYER_TILE_N,Format::kTileK,QWEN38_LAYER_STAGES,QWEN38_LAYER_WARPS>(
		&gemm,activation,b->shared_down_weight,rows,rows,1u,1u,
		QWEN38_EXPERT_INTERMEDIATE,QWEN38_HIDDEN,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((Qwen38SharedExpertAddKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, (QWEN38_LAYER_THREADS / LM_WARP_LANES) * sizeof(float), stream,
		b->hidden_bf16,b->shared_out_bf16,(const uint16_t *)b->shared_gate_coeff,(const uint16_t *)b->normed_bf16,b->hidden_bf16,rows,QWEN38_HIDDEN);
	return(LM_LAUNCH_OK);
}

static int32_t Qwen38Head(const Qwen38LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + QWEN38_HEAD_TILE - 1u) / QWEN38_HEAD_TILE;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<QWEN38_LAYER_THREADS,uint16_t>), rows, QWEN38_LAYER_THREADS, (QWEN38_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,QWEN38_HIDDEN,QWEN38_HIDDEN,QWEN38_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<QWEN38_LAYER_THREADS,QWEN38_HEAD_TILE>), dim3(tiles,rows), QWEN38_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,QWEN38_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<QWEN38_LAYER_THREADS>), rows, QWEN38_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
