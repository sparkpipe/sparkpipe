#pragma once

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "sparkpipe/spark_laguna_resident_decode_stage_firmware.h"
#include "modules/laguna_resident_decode_stage/source/cuda/config.h"
#include "modules/laguna_resident_decode_stage/source/cuda/launch_shape.h"

using LagunaKv = LmKvGeometry<LAGUNA_KV_SLOT_BYTES,LAGUNA_KV_PAGE_SLOTS,true>;

static_assert(
	LagunaKv::kSlotBytes == (LAGUNA_KV_HEADS * (LAGUNA_HEAD_DIM + LAGUNA_HEAD_DIM) * 2u),
	"the laguna kv slot is [K: 1 head x 128][V: 1 head x 128] bf16 and nothing else");

#define LAGUNA_LAYER_TILE_N 128u
#define LAGUNA_LAYER_STAGES 2u
#define LAGUNA_LAYER_WARPS 8u
#define LAGUNA_HEAD_TILE 1024u

static_assert(
	LAGUNA_HIDDEN % LmBf16Format::kTileK == 0u,
	"laguna hidden projections must cover every BF16 K tile");
static_assert(
	(LAGUNA_ATTN_HEADS * LAGUNA_HEAD_DIM) % LmBf16Format::kTileK == 0u,
	"laguna attention output must cover every BF16 K tile");
static_assert(
	LAGUNA_DENSE_INTERMEDIATE % LmBf16Format::kTileK == 0u,
	"laguna dense FFN down projection must cover every BF16 K tile");
static_assert(
	LAGUNA_EXPERT_INTERMEDIATE % LmBf16Format::kTileK == 0u,
	"laguna expert down projection must cover every BF16 K tile");
static_assert(
	LAGUNA_EXPERTS <= LM_TOPK_SMALL_LIMIT,
	"the laguna router is the small-topk path");
static_assert(
	LAGUNA_WINDOW <= 1024u,
	"the sliding window positions buffer is sized for the stated window");

struct LagunaLayerBuffers
{
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;

	const void *attn_norm_weight;
	const void *fused_qkv_weight;
	const void *q_norm_weight;
	const void *k_norm_weight;
	const void *output_weight;
	const void *gate_weight;
	float qk_scale;
	const float *yarn_inv_freq;
	const void *mlp_norm_weight;
	const void *router_weight;
	const float *router_correction_bias;
	const void *dense_gate_up_weight;
	const void *dense_down_weight;
	uint32_t dense_gate_up_rows;
	uint32_t dense_intermediate;
	uint32_t expert_w1_rows;
	uint32_t expert_intermediate;
	uint32_t shared_gate_up_rows;
	uint32_t shared_intermediate;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	const void *shared_gate_up_weight;
	const void *shared_down_weight;

	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t layer_index;
	uint32_t q_heads;
	uint32_t qkv_rows;
	uint32_t attn_output_columns;
	uint32_t gate_rows;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *qkv_bf16;
	uint16_t *q_bf16;
	uint16_t *k_bf16;
	uint16_t *v_bf16;
	uint16_t *attention_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *gate_bf16;
	uint32_t *window_positions;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *row_positions;
};

static inline int32_t LagunaLaunchBf16Linear(
	const uint16_t *activation_bf16,
	const void *weight_bf16,
	uint16_t *output_bf16,
	const uint32_t *row_offset,
	uint32_t *tile_prefix,
	uint32_t rows,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t output_row_stride,
	uint32_t output_column_offset,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	LmGemmArguments gemm;

	if (activation_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 ||
		row_offset == 0 || tile_prefix == 0 || rows == 0u ||
		input_dimension == 0u || output_dimension == 0u ||
		multiprocessors == 0u)
	{
		return LM_LAUNCH_ERR_SHAPE;
	}

	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = row_offset;
	gemm.group_tile_prefix = tile_prefix;
	gemm.output_bf16 = output_bf16;
	gemm.output_row_stride = output_row_stride;
	gemm.output_column_offset = output_column_offset;
	return LmGemmLaunch<
		LmBf16Format,
		LAGUNA_LAYER_TILE_N,
		LmBf16Format::kTileK,
		LAGUNA_LAYER_STAGES,
		LAGUNA_LAYER_WARPS>(
			&gemm,
			activation_bf16,
			weight_bf16,
			rows,
			rows,
			1u,
			1u,
			input_dimension,
			output_dimension,
			multiprocessors,
			false,
			stream);
}

static inline void LagunaBuildYarnInvFrequency(
	float *inv_freq,
	uint32_t rotary_dimension,
	float theta,
	float factor,
	float original_positions,
	float beta_fast,
	float beta_slow)
{
	float low_exact,high_exact;
	float low,high;
	uint32_t half,index;
	if ( inv_freq == 0 || rotary_dimension < 2u || factor <= 1.0f )
		return;
	low_exact = ((float)rotary_dimension *
		logf(original_positions / (beta_fast * 6.283185307179586f))) /
		(2.0f * logf(theta));
	high_exact = ((float)rotary_dimension *
		logf(original_positions / (beta_slow * 6.283185307179586f))) /
		(2.0f * logf(theta));
	low = floorf(fmaxf(fminf(low_exact,(float)rotary_dimension - 1.0f),0.0f));
	high = ceilf(fmaxf(fminf(high_exact,(float)rotary_dimension - 1.0f),0.0f));
	half = rotary_dimension / 2u;
	for (index = 0u; index < half; ++index)
	{
		float base = powf(theta,-2.0f * (float)index / (float)rotary_dimension);
		float ramp = ((float)index - low) / fmaxf(high - low,1e-6f);
		float blend = fminf(fmaxf(ramp,0.0f),1.0f);
		inv_freq[index] = (base * (1.0f - blend)) + ((base / factor) * blend);
	}
}

static int32_t LagunaLayerAttention(
	const LagunaLayerBuffers *buffers,
	uint32_t rows,
	uint32_t context,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	int32_t status;
	uint32_t sliding,kv_positions;
	const uint32_t *selected_positions;
	if (buffers == 0 || rows == 0u || context == 0u ||
		buffers->qk_scale <= 0.0f || buffers->hidden_bf16 == 0 ||
		buffers->residual_bf16 == 0 || buffers->normed_bf16 == 0 ||
		buffers->attn_norm_weight == 0 || buffers->fused_qkv_weight == 0 ||
		buffers->q_norm_weight == 0 || buffers->k_norm_weight == 0 ||
		buffers->output_weight == 0 || buffers->gate_weight == 0 ||
		buffers->qkv_bf16 == 0 || buffers->q_bf16 == 0 ||
		buffers->k_bf16 == 0 || buffers->v_bf16 == 0 ||
		buffers->attention_bf16 == 0 || buffers->attention_out_bf16 == 0 ||
		buffers->gate_bf16 == 0 || buffers->window_positions == 0 ||
		buffers->sequence_of_row == 0 || buffers->context_length == 0 ||
		buffers->positions == 0 || !LmKvViewIsConfigured(buffers->cache))
	{
		return LM_LAUNCH_ERR_SHAPE;
	}
	sliding = LAGUNA_LAYER_IS_SLIDING(buffers->layer_index) ? 1u : 0u;
	selected_positions = 0;
	kv_positions = 0u;

	LM_LAUNCH(
		(LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
		rows,
		LAGUNA_LAYER_THREADS,
		(LAGUNA_HIDDEN + 8u) * sizeof(float),
		stream,
		buffers->hidden_bf16,
		buffers->residual_bf16,
		(const uint16_t *)buffers->attn_norm_weight,
		buffers->residual_bf16,
		buffers->normed_bf16,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		LAGUNA_RMS_EPSILON);
	status = LagunaLaunchBf16Linear(
		buffers->normed_bf16,
		buffers->fused_qkv_weight,
		buffers->qkv_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		LAGUNA_HIDDEN,
		buffers->qkv_rows,
		buffers->qkv_rows,
		0u,
		multiprocessors,
		stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}
	{
		LmQkvLayout layout;
		layout.query_dimension = buffers->q_heads * LAGUNA_HEAD_DIM;
		layout.key_dimension = LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;
		layout.value_dimension = LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;
		layout.rope_dimension = 0u;
		layout.head_dimension = LAGUNA_HEAD_DIM;
		LM_LAUNCH(
			(LmSplitQkvKernel<LAGUNA_LAYER_THREADS>),
			rows,
			LAGUNA_LAYER_THREADS,
			0,
			stream,
			buffers->qkv_bf16,
			layout,
			buffers->q_bf16,
			buffers->k_bf16,
			buffers->v_bf16,
			rows,
			1.0f);
	}
	LM_LAUNCH(
		(LmHeadRmsNormKernel<LAGUNA_LAYER_THREADS>),
		dim3(buffers->q_heads,rows),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->q_bf16,
		(const uint16_t *)buffers->q_norm_weight,
		buffers->q_bf16,
		rows,
		buffers->q_heads,
		LAGUNA_HEAD_DIM,
		LAGUNA_RMS_EPSILON,
		1.0f);
	LM_LAUNCH(
		(LmHeadRmsNormKernel<LAGUNA_LAYER_THREADS>),
		dim3(LAGUNA_KV_HEADS,rows),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->k_bf16,
		(const uint16_t *)buffers->k_norm_weight,
		buffers->k_bf16,
		rows,
		LAGUNA_KV_HEADS,
		LAGUNA_HEAD_DIM,
		LAGUNA_RMS_EPSILON,
		1.0f);
	if ( sliding != 0u )
	{
		LM_LAUNCH(
			(LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),
			dim3(rows,buffers->q_heads),
			LAGUNA_LAYER_THREADS,
			0,
			stream,
			buffers->q_bf16,
			buffers->positions,
			buffers->q_heads,
			LAGUNA_HEAD_DIM,
			LAGUNA_ROPE_SLIDING_ROT,
			LAGUNA_ROPE_SLIDING_THETA);
		LM_LAUNCH(
			(LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),
			dim3(rows,LAGUNA_KV_HEADS),
			LAGUNA_LAYER_THREADS,
			0,
			stream,
			buffers->k_bf16,
			buffers->positions,
			LAGUNA_KV_HEADS,
			LAGUNA_HEAD_DIM,
			LAGUNA_ROPE_SLIDING_ROT,
			LAGUNA_ROPE_SLIDING_THETA);
	}
	else
	{
		LM_LAUNCH(
			(LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),
			dim3(rows,buffers->q_heads),
			LAGUNA_LAYER_THREADS,
			0,
			stream,
			buffers->q_bf16,
			buffers->positions,
			buffers->q_heads,
			LAGUNA_HEAD_DIM,
			LAGUNA_ROPE_FULL_ROT,
			0.0f,
			buffers->yarn_inv_freq,
			LAGUNA_ROPE_FULL_ATTENTION_FACTOR,
			0u);
		LM_LAUNCH(
			(LmRopePerHeadKernel<LAGUNA_LAYER_THREADS,LM_ROPE_HALF_SPLIT>),
			dim3(rows,LAGUNA_KV_HEADS),
			LAGUNA_LAYER_THREADS,
			0,
			stream,
			buffers->k_bf16,
			buffers->positions,
			LAGUNA_KV_HEADS,
			LAGUNA_HEAD_DIM,
			LAGUNA_ROPE_FULL_ROT,
			0.0f,
			buffers->yarn_inv_freq,
			LAGUNA_ROPE_FULL_ATTENTION_FACTOR,
			0u);
	}
	LM_LAUNCH(
		(LmGqaKvStoreKernel<LagunaKv,LAGUNA_LAYER_THREADS,LAGUNA_KV_HEADS,
			LAGUNA_HEAD_DIM,LAGUNA_HEAD_DIM>),
		rows,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->cache,
		buffers->k_bf16,
		buffers->v_bf16,
		buffers->sequence_of_row,
		buffers->positions,
		rows);
	if ( sliding != 0u )
	{
		LM_LAUNCH(
			(LmBuildSlidingWindowPositionsKernel<LAGUNA_LAYER_THREADS>),
			rows,
			LAGUNA_LAYER_THREADS,
			0,
			stream,
			buffers->sequence_of_row,
			buffers->context_length,
			buffers->row_positions,
			rows,
			LAGUNA_WINDOW,
			buffers->window_positions);
		selected_positions = buffers->window_positions;
		kv_positions = LAGUNA_WINDOW;
	}
	LM_LAUNCH(
		(LmGqaAttentionDecodeKernel<LagunaKv,LAGUNA_ATTN_THREADS,LAGUNA_KV_HEADS,
			LAGUNA_HEAD_DIM,LAGUNA_HEAD_DIM>),
		dim3(rows,buffers->q_heads),
		LAGUNA_ATTN_THREADS,
		0,
		stream,
		buffers->q_bf16,
		buffers->cache,
		buffers->sequence_of_row,
		buffers->context_length,
		selected_positions,
		kv_positions,
		buffers->q_heads,
		buffers->qk_scale,
		buffers->attention_bf16,
		sliding != 0u ? 0 : buffers->row_positions);
	status = LagunaLaunchBf16Linear(
		buffers->normed_bf16,
		buffers->gate_weight,
		buffers->gate_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		LAGUNA_HIDDEN,
		buffers->gate_rows,
		buffers->gate_rows,
		0u,
		multiprocessors,
		stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}
	LM_LAUNCH(
		(LmHeadGateBroadcastKernel<LAGUNA_LAYER_THREADS,LM_GATE_SOFTPLUS>),
		dim3(rows,buffers->q_heads),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->attention_bf16,
		buffers->gate_bf16,
		buffers->q_heads,
		LAGUNA_HEAD_DIM);
	status = LagunaLaunchBf16Linear(
		buffers->attention_bf16,
		buffers->output_weight,
		buffers->attention_out_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		buffers->attn_output_columns,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		0u,
		multiprocessors,
		stream);
	return status;
}

static int32_t LagunaLayerDenseMlp(
	const LagunaLayerBuffers *buffers,
	uint32_t rows,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	int32_t status;

	if (buffers == 0 || rows == 0u || buffers->attention_out_bf16 == 0 ||
		buffers->residual_bf16 == 0 || buffers->mlp_norm_weight == 0 ||
		buffers->normed_bf16 == 0 || buffers->dense_gate_up_weight == 0 ||
		buffers->dense_down_weight == 0 ||
		buffers->gate_up_bf16 == 0 || buffers->intermediate_bf16 == 0 ||
		buffers->hidden_bf16 == 0)
	{
		return LM_LAUNCH_ERR_SHAPE;
	}

	LM_LAUNCH(
		(LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
		rows,
		LAGUNA_LAYER_THREADS,
		(LAGUNA_HIDDEN + 8u) * sizeof(float),
		stream,
		buffers->attention_out_bf16,
		buffers->residual_bf16,
		(const uint16_t *)buffers->mlp_norm_weight,
		buffers->residual_bf16,
		buffers->normed_bf16,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		LAGUNA_RMS_EPSILON);
	status = LagunaLaunchBf16Linear(
		buffers->normed_bf16,
		buffers->dense_gate_up_weight,
		buffers->gate_up_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		LAGUNA_HIDDEN,
		buffers->dense_gate_up_rows,
		buffers->dense_gate_up_rows,
		0u,
		multiprocessors,
		stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}
	LM_LAUNCH(
		(LmSiluMulKernel<LAGUNA_LAYER_THREADS>),
		rows,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->gate_up_bf16,
		buffers->intermediate_bf16,
		buffers->dense_intermediate,
		true);
	status = LagunaLaunchBf16Linear(
		buffers->intermediate_bf16,
		buffers->dense_down_weight,
		buffers->attention_out_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		buffers->dense_intermediate,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		0u,
		multiprocessors,
		stream);
	return(status);
}

template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoeValidate(
	const LagunaLayerBuffers *buffers,
	uint32_t rows,
	uint32_t packed_rows)
{
	using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;

	static_assert((ExpertFormat::kScaleGroup == 0u ||
			(LAGUNA_HIDDEN % ExpertFormat::kScaleGroup == 0u &&
			 LAGUNA_EXPERT_INTERMEDIATE % ExpertFormat::kScaleGroup == 0u)),
		"laguna expert dimensions must contain complete codec scale groups");

	if (buffers == 0 || rows == 0u ||
		packed_rows != rows * LAGUNA_TOP_K ||
		buffers->attention_out_bf16 == 0 || buffers->residual_bf16 == 0 ||
		buffers->mlp_norm_weight == 0 || buffers->normed_bf16 == 0 ||
		buffers->router_weight == 0 || buffers->router_logits == 0 ||
		buffers->router_correction_bias == 0 ||
		buffers->route_expert == 0 || buffers->route_weight == 0 ||
		buffers->route_source_token == 0 || buffers->route_packed_row == 0 ||
		buffers->group_row_offset == 0 ||
		buffers->group_tile_prefix_w1 == 0 ||
		buffers->group_tile_prefix_w2 == 0 ||
		buffers->expert_out_bf16 == 0 || buffers->gate_up_bf16 == 0 ||
		buffers->intermediate_bf16 == 0 || buffers->hidden_bf16 == 0 ||
		buffers->shared_gate_up_weight == 0 ||
		buffers->shared_down_weight == 0 || buffers->shared_out_bf16 == 0)
	{
		return LM_LAUNCH_ERR_SHAPE;
	}

	return LM_LAUNCH_OK;
}

template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoeRoute(
	const LagunaLayerBuffers *buffers,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status = LagunaLayerMoeValidate<ExpertCodec>(buffers,rows,packed_rows);
	if (status != LM_LAUNCH_OK)
		return status;
	LM_LAUNCH(
		(LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
		rows,
		LAGUNA_LAYER_THREADS,
		(LAGUNA_HIDDEN + 8u) * sizeof(float),
		stream,
		buffers->attention_out_bf16,
		buffers->residual_bf16,
		(const uint16_t *)buffers->mlp_norm_weight,
		buffers->residual_bf16,
		buffers->normed_bf16,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		LAGUNA_RMS_EPSILON);

	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = buffers->dense_row_offset;
	gemm.group_tile_prefix = buffers->dense_tile_prefix;
	gemm.output_f32 = buffers->router_logits;
	status = LmGemmLaunch<
		LmBf16Format,
		LAGUNA_LAYER_TILE_N,
		LmBf16Format::kTileK,
		LAGUNA_LAYER_STAGES,
		LAGUNA_LAYER_WARPS>(
			&gemm,
			buffers->normed_bf16,
			buffers->router_weight,
			rows,
			rows,
			1u,
			1u,
			LAGUNA_HIDDEN,
			LAGUNA_EXPERTS,
			multiprocessors,
			false,
			stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}

	LM_LAUNCH(
		(LmTopkSmallKernel<
			LAGUNA_LAYER_THREADS,
			LAGUNA_TOP_K,
			true,
			1u,
			1u,
			LM_TOPK_SCORE_SIGMOID>),
		rows,
		LAGUNA_LAYER_THREADS,
		2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),
		stream,
		buffers->router_logits,
		LAGUNA_EXPERTS,
		buffers->route_expert,
		buffers->route_weight,
		buffers->router_correction_bias,
		0,
		LAGUNA_ROUTED_SCALE);
	status = LmRouteBuild<LAGUNA_LAYER_THREADS, LAGUNA_EXPERTS>(
		buffers->route_expert,
		rows,
		packed_rows,
		LAGUNA_TOP_K,
		buffers->group_row_offset,
		buffers->route_packed_row,
		buffers->route_source_token,
		buffers->expert_w1_rows,
		LAGUNA_HIDDEN,
		LAGUNA_LAYER_TILE_N,
		buffers->group_tile_prefix_w1,
		buffers->group_tile_prefix_w2,
		stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}

	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoeExperts(
	const LagunaLayerBuffers *buffers,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;
	LmGemmArguments gemm;
	int32_t status = LagunaLayerMoeValidate<ExpertCodec>(buffers,rows,packed_rows);
	if (status != LM_LAUNCH_OK)
		return status;
	if ( buffers->expert_w1_weight == 0 || buffers->expert_w1_scale == 0 ||
		buffers->expert_w2_weight == 0 || buffers->expert_w2_scale == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
		buffers->expert_w1_scale,
		LAGUNA_EXPERTS,
		buffers->expert_w1_rows,
		LAGUNA_HIDDEN);
	gemm.prefix_built = 1u;
	gemm.group_row_offset = buffers->group_row_offset;
	gemm.group_tile_prefix = buffers->group_tile_prefix_w1;
	gemm.source_row_map = buffers->route_source_token;
	gemm.source_row_count = rows;
	gemm.output_bf16 = buffers->gate_up_bf16;
	status = LmGemmWeightOnlyIndirectLaunch<
		ExpertFormat,
		LAGUNA_LAYER_TILE_N,
		LAGUNA_LAYER_STAGES,
		LAGUNA_LAYER_WARPS>(
			&gemm,
			buffers->normed_bf16,
			buffers->expert_w1_weight,
			packed_rows,
			rows,
			LAGUNA_TOP_K,
			LAGUNA_EXPERTS,
			LAGUNA_HIDDEN,
			buffers->expert_w1_rows,
			multiprocessors,
			stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}

	LM_LAUNCH(
		(LmSiluMulKernel<LAGUNA_LAYER_THREADS>),
		packed_rows,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->gate_up_bf16,
		buffers->intermediate_bf16,
		buffers->expert_intermediate,
		true);

	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<ExpertCodec>(
		buffers->expert_w2_scale,
		LAGUNA_EXPERTS,
		LAGUNA_HIDDEN,
		buffers->expert_intermediate);
	gemm.prefix_built = 1u;
	gemm.group_row_offset = buffers->group_row_offset;
	gemm.group_tile_prefix = buffers->group_tile_prefix_w2;
	gemm.output_bf16 = buffers->expert_out_bf16;
	status = LmGemmWeightOnlyLaunch<
		ExpertFormat,
		LAGUNA_LAYER_TILE_N,
		LAGUNA_LAYER_STAGES,
		LAGUNA_LAYER_WARPS>(
			&gemm,
			buffers->intermediate_bf16,
			buffers->expert_w2_weight,
			packed_rows,
			rows,
			LAGUNA_TOP_K,
			LAGUNA_EXPERTS,
			buffers->expert_intermediate,
			LAGUNA_HIDDEN,
			multiprocessors,
			true,
			stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}

	LM_LAUNCH(
		(LmMoeFinalizeKernel<LAGUNA_LAYER_THREADS>),
		dim3(
			(LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) /
				LAGUNA_LAYER_THREADS,
			rows),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->expert_out_bf16,
		buffers->route_packed_row,
		buffers->route_weight,
		buffers->attention_out_bf16,
		rows,
		LAGUNA_TOP_K,
		LAGUNA_HIDDEN);
	status = LagunaLaunchBf16Linear(
		buffers->normed_bf16,
		buffers->shared_gate_up_weight,
		buffers->gate_up_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		LAGUNA_HIDDEN,
		buffers->shared_gate_up_rows,
		buffers->shared_gate_up_rows,
		0u,
		multiprocessors,
		stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}
	LM_LAUNCH(
		(LmSiluMulKernel<LAGUNA_LAYER_THREADS>),
		rows,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->gate_up_bf16,
		buffers->intermediate_bf16,
		buffers->shared_intermediate,
		true);
	status = LagunaLaunchBf16Linear(
		buffers->intermediate_bf16,
		buffers->shared_down_weight,
		buffers->shared_out_bf16,
		buffers->dense_row_offset,
		buffers->dense_tile_prefix,
		rows,
		buffers->shared_intermediate,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		0u,
		multiprocessors,
		stream);
	if (status != LM_LAUNCH_OK)
	{
		return status;
	}
	LM_LAUNCH(
		(LmAddRowsKernel<LAGUNA_LAYER_THREADS>),
		dim3(
			(LAGUNA_HIDDEN + LAGUNA_LAYER_THREADS - 1u) /
				LAGUNA_LAYER_THREADS,
			rows),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->attention_out_bf16,
		buffers->shared_out_bf16,
		buffers->attention_out_bf16,
		rows,
		LAGUNA_HIDDEN);
	return cudaPeekAtLastError() == cudaSuccess
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH;
}

template<uint32_t ExpertCodec>
static int32_t LagunaLayerMoe(
	const LagunaLayerBuffers *buffers,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	int32_t status = LagunaLayerMoeRoute<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
	if (status != LM_LAUNCH_OK)
		return status;
	return LagunaLayerMoeExperts<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
}

static int32_t LagunaHead(
	const LagunaLayerBuffers *buffers,
	const void *head_norm_weight,
	const void *head_weight,
	const uint32_t *token_ids,
	uint32_t vocabulary,
	uint32_t rows,
	cudaStream_t stream)
{
	uint32_t tiles;

	if (buffers == 0 || head_norm_weight == 0 || head_weight == 0 ||
		rows == 0u || vocabulary == 0u || buffers->hidden_bf16 == 0 ||
		buffers->residual_bf16 == 0 || buffers->normed_bf16 == 0 ||
		buffers->head_candidate_score == 0 ||
		buffers->head_candidate_token == 0 || buffers->output_token == 0 ||
		buffers->output_score == 0)
	{
		return LM_LAUNCH_ERR_SHAPE;
	}

	tiles = (vocabulary + LAGUNA_HEAD_TILE - 1u) / LAGUNA_HEAD_TILE;
	LM_LAUNCH(
		(LmFusedResidualRmsNormKernel<LAGUNA_LAYER_THREADS,uint16_t>),
		rows,
		LAGUNA_LAYER_THREADS,
		(LAGUNA_HIDDEN + 8u) * sizeof(float),
		stream,
		buffers->attention_out_bf16,
		buffers->residual_bf16,
		(const uint16_t *)head_norm_weight,
		buffers->residual_bf16,
		buffers->normed_bf16,
		LAGUNA_HIDDEN,
		LAGUNA_HIDDEN,
		LAGUNA_RMS_EPSILON);
	LM_LAUNCH(
		(LmHeadCandidateKernel<LAGUNA_LAYER_THREADS, LAGUNA_HEAD_TILE>),
		dim3(tiles, rows),
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->normed_bf16,
		(const uint16_t *)head_weight,
		token_ids,
		buffers->head_candidate_score,
		buffers->head_candidate_token,
		rows,
		LAGUNA_HIDDEN,
		vocabulary);
	LM_LAUNCH(
		(LmHeadCommitKernel<LAGUNA_LAYER_THREADS>),
		rows,
		LAGUNA_LAYER_THREADS,
		0,
		stream,
		buffers->head_candidate_score,
		buffers->head_candidate_token,
		tiles,
		buffers->output_token,
		buffers->output_score,
		rows);
	return cudaPeekAtLastError() == cudaSuccess
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH;
}
