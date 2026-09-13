#pragma once

// One MiMo 2.5 decode layer.
//
// Written second, after llms/glm5_2/layer.cuh, and deliberately not by copying
// it. The two models differ in ways that would have made a copy wrong in exactly
// the manner the first one was wrong nine times:
//
//     glm5_2                        mimo_2_5
//     four separate projections     one fused QKV
//     latent cache, 1152 B/slot     per-head K and V, 2560 or 5120
//     one attention kind            two, chosen per layer
//     one rope theta                two, chosen per layer
//     rope on its own buffers       rope on a suffix of every head
//
// What is shared is every kernel. kernels/ did not change to accommodate this
// file; project.cuh gained two primitives that GLM 5.2 does not use and MiMo 2.5
// does, which is the difference between a shared library and a common one.

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/head.cuh"
#include "inference/llms/mimo_2_5/config.h"

// LmKvHeads prices a slot as heads x head_dim x (k+v) - it assumes the value is
// as wide as the key. MiMo's value is 128 against a 192 key, so that alias
// oversized the slot by a fifth and never said where the value lived. The
// geometry is priced from the actual sum, which is also the layout contract
// kernels/gqa.cuh static_asserts: [K: heads x 192][V: heads x 128] bf16.
using Mimo25FullKv = LmKvGeometry<(MIMO25_FULL_KV_HEADS * (MIMO25_HEAD_DIM + MIMO25_VALUE_DIM) * MIMO25_KV_BITS) / 8u, MIMO25_KV_PAGE_SLOTS, true>;
using Mimo25SwaKv  = LmKvGeometry<(MIMO25_SWA_KV_HEADS * (MIMO25_HEAD_DIM + MIMO25_VALUE_DIM) * MIMO25_KV_BITS) / 8u, MIMO25_KV_PAGE_SLOTS, true>;

// Overridable because the host harnesses instantiate the layer at one thread:
// the CPU shim's sequential schedule (tests/host_cuda/lm_host_cuda.cuh) is
// only a valid execution of these kernels when the template width IS one.
// Device builds take the default and nothing changes.
#ifndef MIMO25_LAYER_THREADS
#define MIMO25_LAYER_THREADS 256u
#endif
#define MIMO25_LAYER_TILE_N 128u
#define MIMO25_LAYER_STAGES 2u
#define MIMO25_LAYER_WARPS 8u

struct Mimo25LayerBuffers
{
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;

	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *fused_qkv_bf16;
	uint16_t *query_bf16;
	uint16_t *key_bf16;
	uint16_t *value_bf16;
	uint16_t *attention_out_bf16;
	uint8_t *packed_activation;
	uint8_t *packed_scale;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	float *router_logits;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;

	LmKvView cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	uint32_t *window_positions;
	float *head_candidate_score;
	uint32_t *head_candidate_token;
	uint32_t *output_token;
	float *output_score;
};


template<class Format>
static LmScaleTensor Mimo25ActivationScale(
    const void *scale_data,
    uint32_t row_count,
    uint32_t input_dimension)
{
    return LmScaleTensorRowsUe4m3(
        scale_data,
        row_count,
        input_dimension,
        Format::kScaleGroup);
}

template<class Format>
static LmScaleTensor Mimo25WeightScale(
    const void *scale_data,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    return LmScaleTensorBlockF32(
        scale_data,
        1u,
        output_dimension,
        input_dimension,
        Format::kScaleGroup,
        Format::kScaleGroup);
}

// Attention. The layer kind selects the projection width, the rope theta, the
// KV geometry and whether the window applies - four things from one flag, which
// is why it is a template parameter rather than an argument.
template<class Format, class Geometry, uint32_t KV_HEADS, uint32_t QKV_DIM>
static int32_t Mimo25LayerAttention(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t context, uint32_t window, float theta, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	LmQkvLayout layout;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS,uint16_t>), rows, MIMO25_LAYER_THREADS, (MIMO25_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,b->residual_bf16,(const uint16_t *)b->attn_norm_weight, b->residual_bf16,b->normed_bf16,MIMO25_HIDDEN,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>), dim3(rows,MIMO25_HIDDEN / Format::kScaleGroup), MIMO25_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,MIMO25_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = Mimo25ActivationScale<Format>(b->packed_scale,rows,MIMO25_HIDDEN);
	gemm.scale_b = Mimo25WeightScale<Format>(b->qkv_scale,QKV_DIM,MIMO25_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->fused_qkv_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->qkv_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_HIDDEN,QKV_DIM,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	layout.query_dimension = MIMO25_Q_DIM;
	layout.key_dimension = KV_HEADS * MIMO25_HEAD_DIM;
	layout.value_dimension = KV_HEADS * MIMO25_VALUE_DIM;
	layout.rope_dimension = MIMO25_ROPE_DIM;
	layout.head_dimension = MIMO25_HEAD_DIM;
	LM_LAUNCH((LmSplitQkvKernel<MIMO25_LAYER_THREADS>), rows, MIMO25_LAYER_THREADS, 0, stream,
		b->fused_qkv_bf16,layout,b->query_bf16,b->key_bf16,b->value_bf16, rows,MIMO25_VALUE_SCALE);
	// Per head, not per row: the rope part is the suffix of every head, so
	// rotating the row's tail would rotate the last head and leave 63 unrotated.
	LM_LAUNCH((LmRopePerHeadKernel<MIMO25_LAYER_THREADS>), dim3(rows,MIMO25_ATTN_HEADS), MIMO25_LAYER_THREADS, 0, stream,
		b->query_bf16,b->positions,MIMO25_ATTN_HEADS,MIMO25_HEAD_DIM,MIMO25_ROPE_DIM,theta);
	LM_LAUNCH((LmRopePerHeadKernel<MIMO25_LAYER_THREADS>), dim3(rows,KV_HEADS), MIMO25_LAYER_THREADS, 0, stream,
		b->key_bf16,b->positions,KV_HEADS,MIMO25_HEAD_DIM,MIMO25_ROPE_DIM,theta);
	// The slot holds all keys then all values, packed by the store from the
	// split's two dense tensors. What stood here read kv_slot_bf16, which no
	// launch ever wrote - the cache held allocator residue - and attended
	// through the MLA latent kernel, which returns the key's prefix as the
	// value: wrong tensor, wrong width (192 where this model's value is 128),
	// and 12288 outputs written into an 8192-wide pipeline.
	LM_LAUNCH((LmGqaKvStoreKernel<Geometry,MIMO25_LAYER_THREADS,KV_HEADS,MIMO25_HEAD_DIM,MIMO25_VALUE_DIM>), rows, MIMO25_LAYER_THREADS, 0, stream,
		b->cache,b->key_bf16,b->value_bf16,b->sequence_of_row,b->positions,rows);
	// A sliding window is an explicit device-built position list. Building it
	// from the live sequence lengths and row positions avoids stale host lists,
	// and sentinel padding keeps short rows mathematically exact.
	if ( window != 0u )
	{
		if ( b->window_positions == 0 )
			return(LM_LAUNCH_ERR_SHAPE);
		LM_LAUNCH((LmBuildSlidingWindowPositionsKernel<MIMO25_LAYER_THREADS>), rows,
			MIMO25_LAYER_THREADS, 0, stream,
			b->sequence_of_row,b->context_length,b->positions,rows,window,
			b->window_positions);
	}
	LM_LAUNCH((LmGqaAttentionDecodeKernel<Geometry,MIMO25_LAYER_THREADS,KV_HEADS,MIMO25_HEAD_DIM,MIMO25_VALUE_DIM>), dim3(rows,MIMO25_ATTN_HEADS), MIMO25_LAYER_THREADS, 0, stream,
		b->query_bf16,b->cache,b->sequence_of_row,b->context_length, window != 0u ? b->window_positions : 0,window,MIMO25_ATTN_HEADS, rsqrtf((float)MIMO25_HEAD_DIM),b->attention_out_bf16, 0);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>), dim3(rows,MIMO25_O_INPUT_DIM / Format::kScaleGroup), MIMO25_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->attention_out_bf16,0,b->packed_activation,b->packed_scale,rows,MIMO25_O_INPUT_DIM);
	gemm.scale_a = Mimo25ActivationScale<Format>(b->packed_scale,rows,MIMO25_O_INPUT_DIM);
	gemm.scale_b = Mimo25WeightScale<Format>(b->output_scale,MIMO25_HIDDEN,MIMO25_O_INPUT_DIM);
	gemm.output_bf16 = b->attention_out_bf16;
	(void)context;
	return(LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->output_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_O_INPUT_DIM,MIMO25_HIDDEN,sms,false,stream));
}

// The MLP half. Routed and dense, the same pair GLM 5.2 has, at MiMo 2.5's
// widths - 2048 per expert against a dense 16384.
//
// Identical in shape to glm5_2's because the MoE really is the same computation:
// route, expand, two GEMMs with a gated activation between, fold back. The
// models differ in attention and agree here, which is why both call the same
// kernels with different constants rather than sharing a function that takes
// both models' constants as arguments.
template<class Format>
static int32_t Mimo25LayerMoe(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS,uint16_t>), rows, MIMO25_LAYER_THREADS, (MIMO25_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,MIMO25_HIDDEN,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_f32 = b->router_logits;
	status = LmGemmLaunch<LmBf16Format,MIMO25_LAYER_TILE_N,LmBf16Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->normed_bf16,b->router_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_HIDDEN,MIMO25_EXPERTS,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmTopkSmallKernel<MIMO25_LAYER_THREADS,MIMO25_TOP_K,false,1u,1u,LM_TOPK_SCORE_IDENTITY>), rows, MIMO25_LAYER_THREADS, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream,
		b->router_logits,MIMO25_EXPERTS,b->route_expert,b->route_weight,0,0,1.0f);
	status = LmRouteBuild<MIMO25_LAYER_THREADS,MIMO25_EXPERTS>(
		b->route_expert,rows,packed_rows,MIMO25_TOP_K,b->group_row_offset,
		b->route_packed_row,b->route_source_token,MIMO25_GATE_UP_DIM,MIMO25_HIDDEN,
		MIMO25_LAYER_TILE_N,b->group_tile_prefix_w1,b->group_tile_prefix_w2,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>), dim3(packed_rows,MIMO25_HIDDEN / Format::kScaleGroup), MIMO25_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->normed_bf16,b->route_source_token,b->packed_activation,b->packed_scale, packed_rows,MIMO25_HIDDEN);
	gemm.scale_a = Mimo25ActivationScale<Format>(b->packed_scale,packed_rows,MIMO25_HIDDEN);
	gemm.scale_b = Mimo25WeightScale<Format>(b->expert_w1_scale,MIMO25_GATE_UP_DIM,MIMO25_HIDDEN);
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w1_weight,packed_rows,rows,MIMO25_TOP_K,
		MIMO25_EXPERTS,MIMO25_HIDDEN,MIMO25_GATE_UP_DIM,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSiluMulKernel<MIMO25_LAYER_THREADS>), packed_rows, MIMO25_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,MIMO25_EXPERT_INTERMEDIATE,true);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>), dim3(packed_rows,MIMO25_EXPERT_INTERMEDIATE / Format::kScaleGroup), MIMO25_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale, packed_rows,MIMO25_EXPERT_INTERMEDIATE);
	gemm.scale_a = Mimo25ActivationScale<Format>(b->packed_scale,packed_rows,MIMO25_EXPERT_INTERMEDIATE);
	gemm.scale_b = Mimo25WeightScale<Format>(b->expert_w2_scale,MIMO25_HIDDEN,MIMO25_EXPERT_INTERMEDIATE);
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->expert_w2_weight,packed_rows,rows,MIMO25_TOP_K,
		MIMO25_EXPERTS,MIMO25_EXPERT_INTERMEDIATE,MIMO25_HIDDEN,sms,true,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmMoeFinalizeKernel<MIMO25_LAYER_THREADS>), dim3((MIMO25_HIDDEN + MIMO25_LAYER_THREADS - 1u) / MIMO25_LAYER_THREADS,rows), MIMO25_LAYER_THREADS, 0, stream,
		b->expert_out_bf16,b->route_packed_row,b->route_weight,b->hidden_bf16, rows,MIMO25_TOP_K,MIMO25_HIDDEN);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format>
static int32_t Mimo25LayerDenseMlp(const Mimo25LayerBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	LmGemmArguments gemm;
	int32_t status;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS,uint16_t>), rows, MIMO25_LAYER_THREADS, (MIMO25_HIDDEN + 8u) * sizeof(float), stream,
		b->attention_out_bf16,b->residual_bf16,(const uint16_t *)b->mlp_norm_weight, b->residual_bf16,b->normed_bf16,MIMO25_HIDDEN,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>), dim3(rows,MIMO25_HIDDEN / Format::kScaleGroup), MIMO25_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->normed_bf16,0,b->packed_activation,b->packed_scale,rows,MIMO25_HIDDEN);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = Mimo25ActivationScale<Format>(b->packed_scale,rows,MIMO25_HIDDEN);
	gemm.scale_b = Mimo25WeightScale<Format>(b->dense_gate_up_scale,MIMO25_DENSE_INTERMEDIATE * 2u,MIMO25_HIDDEN);
	gemm.group_row_offset = b->dense_row_offset;
	gemm.group_tile_prefix = b->dense_tile_prefix;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_gate_up_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_HIDDEN,MIMO25_DENSE_INTERMEDIATE * 2u,sms,false,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	LM_LAUNCH((LmSiluMulKernel<MIMO25_LAYER_THREADS>), rows, MIMO25_LAYER_THREADS, 0, stream,
		b->gate_up_bf16,b->intermediate_bf16,MIMO25_DENSE_INTERMEDIATE,true);
	LM_LAUNCH((LmQuantiseRowsKernel<Format,MIMO25_LAYER_THREADS>), dim3(rows,MIMO25_DENSE_INTERMEDIATE / Format::kScaleGroup), MIMO25_LAYER_THREADS, (Format::kScaleGroup + 8u) * sizeof(float), stream,
		b->intermediate_bf16,0,b->packed_activation,b->packed_scale, rows,MIMO25_DENSE_INTERMEDIATE);
	gemm.scale_a = Mimo25ActivationScale<Format>(b->packed_scale,rows,MIMO25_DENSE_INTERMEDIATE);
	gemm.scale_b = Mimo25WeightScale<Format>(b->dense_down_scale,MIMO25_HIDDEN,MIMO25_DENSE_INTERMEDIATE);
	gemm.output_bf16 = b->hidden_bf16;
	return(LmGemmLaunch<Format,MIMO25_LAYER_TILE_N,Format::kTileK,MIMO25_LAYER_STAGES,MIMO25_LAYER_WARPS>(
		&gemm,b->packed_activation,b->dense_down_weight,rows,rows,MIMO25_TOP_K,1u,
		MIMO25_DENSE_INTERMEDIATE,MIMO25_HIDDEN,sms,false,stream));
}

// -- output head ----------------------------------------------------------------
//
// A layer stack that never reaches a vocabulary produces no token. This model
// had every layer kind and no head, so MIMO25_VOCAB was declared, carried through
// the config gate as an exemption reading "no layer sequence yet, so no head
// call", and never read by anything. The sequence is the same three kernels
// glm5_2 uses; only the widths differ.
#define MIMO25_HEAD_TILE 1024u

static int32_t Mimo25Head(const Mimo25LayerBuffers *b, const void *head_norm_weight, const void *head_weight, const uint32_t *token_ids, uint32_t vocabulary, uint32_t rows, cudaStream_t stream)
{
	uint32_t tiles = (vocabulary + MIMO25_HEAD_TILE - 1u) / MIMO25_HEAD_TILE;
	// End of the stream, so no residual in and no residual out.
	LM_LAUNCH((LmFusedResidualRmsNormKernel<MIMO25_LAYER_THREADS,uint16_t>), rows, MIMO25_LAYER_THREADS, (MIMO25_HIDDEN + 8u) * sizeof(float), stream,
		b->hidden_bf16,0,(const uint16_t *)head_norm_weight, 0,b->normed_bf16,MIMO25_HIDDEN,MIMO25_HIDDEN,MIMO25_RMS_EPSILON);
	LM_LAUNCH((LmHeadCandidateKernel<MIMO25_LAYER_THREADS,MIMO25_HEAD_TILE>), dim3(tiles,rows), MIMO25_LAYER_THREADS, 0, stream,
		b->normed_bf16,(const uint16_t *)head_weight,token_ids, b->head_candidate_score,b->head_candidate_token,rows,MIMO25_HIDDEN,vocabulary);
	LM_LAUNCH((LmHeadCommitKernel<MIMO25_LAYER_THREADS>), rows, MIMO25_LAYER_THREADS, 0, stream,
		b->head_candidate_score,b->head_candidate_token,tiles, b->output_token,b->output_score,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
