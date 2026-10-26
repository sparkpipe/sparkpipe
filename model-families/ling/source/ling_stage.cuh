#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "runtime/gemm.cuh"

#define LG_THREADS 256u
#define LG_EPSILON 1e-6f
#define LG_PACK_ALIGNMENT 128u
#define LING_ATTENTION_MLA 0u
#define LING_ATTENTION_GQA 1u

struct LingDefaults
{
	static constexpr uint32_t kAttention = LING_ATTENTION_MLA;
	static constexpr float kNormEpsilon = LG_EPSILON;
	static constexpr uint32_t kKdaRank = 0u;
	static constexpr bool kKdaDoubledBeta = false;
	static constexpr uint32_t kMlaHeads = 0u;
	static constexpr uint32_t kNope = 0u;
	static constexpr uint32_t kRope = 0u;
	static constexpr uint32_t kValueDim = 0u;
	static constexpr uint32_t kLatent = 0u;
	static constexpr float kRopeTheta = 0.0f;
	static constexpr uint32_t kGqaHeads = 0u;
	static constexpr uint32_t kGqaKvHeads = 0u;
	static constexpr uint32_t kGqaHeadDim = 0u;
	static float RoutedLimit(uint32_t layer) { (void)layer; return 0.0f; }
	static float SharedLimit(uint32_t layer) { (void)layer; return 0.0f; }
};

template<class G>
struct LingShape
{
	static constexpr bool kGqa = G::kAttention == LING_ATTENTION_GQA;
	static constexpr uint32_t kAttnLayers = G::kLayers / G::kPeriod;
	static constexpr uint32_t kKdaLayers = G::kLayers - kAttnLayers;
	static constexpr uint32_t kKdaRankHeads = G::kKdaHeads / G::kTpDegree;
	static constexpr uint32_t kKdaWidth = kKdaRankHeads * G::kKdaDim;
	static constexpr uint32_t kKdaChannels = 3u * kKdaWidth;
	static constexpr uint32_t kMlaRankHeads = G::kMlaHeads / G::kTpDegree;
	static constexpr uint32_t kQkDim = G::kNope + G::kRope;
	static constexpr uint32_t kRecord = LM_LATENT_SHARD_RECORD_FLOATS(G::kLatent);
	static constexpr uint32_t kGqaWidth = G::kGqaHeads * G::kGqaHeadDim;
	static constexpr uint32_t kGqaKvWidth = G::kGqaKvHeads * G::kGqaHeadDim;
	static constexpr uint32_t kGqaFused = kGqaWidth + 2u * kGqaKvWidth;
	static constexpr uint32_t kGqaRankFused = kGqaFused / G::kTpDegree;
	static constexpr uint32_t kGqaRankWidth = kGqaWidth / G::kTpDegree;
	static constexpr uint32_t kSlot = kGqa ? 2u * kGqaKvWidth : G::kLatent + G::kRope;
	using Kv = LmKvGeometry<kSlot * 2u, G::kPageSlots, true>;
	static_assert(G::kKdaHeads % G::kTpDegree == 0u && G::kMlaHeads % G::kTpDegree == 0u && G::kExperts % G::kTpDegree == 0u,
		"heads and experts split evenly over the ranks");
	static_assert(kMlaRankHeads * kSlot <= G::kHidden, "a rank's packed queries fit one collective row");
	static_assert(!kGqa || (G::kMlaHeads == 0u && G::kGqaHeads % G::kTpDegree == 0u && kGqaFused % G::kTpDegree == 0u &&
		kGqaRankFused <= G::kHidden && kGqaRankWidth % G::kGqaHeadDim == 0u), "a GQA rank owns whole query heads and its fused rows fit one collective row");
	static_assert(G::kKdaRank % 64u == 0u, "the low-rank decay and gate projections stream in 64-column steps");
};

typedef struct LingLayer
{
	const float *input_norm;
	const float *post_norm;
	const uint16_t *kda_qkv;
	const uint16_t *kda_f_a;
	const uint16_t *kda_f;
	const uint16_t *kda_g_a;
	const uint16_t *kda_g;
	const uint16_t *kda_beta;
	const uint16_t *kda_conv;
	const float *kda_a_log;
	const float *kda_dt_bias;
	const uint16_t *kda_norm;
	const uint16_t *kda_out;
	const uint16_t *mla_q;
	const uint16_t *mla_kv_a;
	const float *mla_kv_norm;
	const uint16_t *mla_uk;
	const uint16_t *mla_uv;
	const uint16_t *mla_gate;
	const uint16_t *mla_out;
	const uint16_t *gqa_qkv;
	const uint16_t *gqa_gate;
	const uint16_t *gqa_out;
	const uint16_t *ffn_gate_up;
	const uint16_t *ffn_down;
	const uint16_t *router;
	const float *expert_bias;
	const uint16_t *moe_w1;
	const uint16_t *moe_w3;
	const uint16_t *moe_w2;
	const uint16_t *shared_gate_up;
	const uint16_t *shared_down;
	uint32_t recurrent_index;
	uint32_t kv_index;
} LingLayer;

template<class G>
struct LingState
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint64_t partial_stride;
	uint64_t kda_state_bytes;
	uint64_t kda_window_bytes;
	LingLayer layers[G::kLayers];
	void *arena;
	uint64_t arena_bytes;
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *qkv;
	uint16_t *kda_low_f;
	uint16_t *kda_low_g;
	uint16_t *decay_logit;
	uint16_t *kda_gate;
	uint16_t *beta;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	float *retention;
	float *write_gate;
	uint16_t *kda_mixed;
	uint16_t *mla_q;
	uint16_t *mla_latent_query;
	uint16_t *mla_query;
	uint16_t *mla_kv;
	uint16_t *mla_gathered;
	float *attn_send;
	float *attn_received;
	uint16_t *mla_latent;
	uint16_t *mla_value;
	uint16_t *mla_gate;
	uint16_t *gqa_local;
	uint16_t *gqa_gathered;
	uint16_t *gqa_full;
	uint16_t *gqa_query;
	uint16_t *gqa_slot;
	uint16_t *gqa_gate;
	uint16_t *gqa_merged;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *router_logits;
	uint32_t *route_expert;
	uint32_t *route_global;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	float *packed_weight;
	uint16_t *expert_gate;
	uint16_t *expert_up;
	uint16_t *expert_act;
	uint16_t *expert_out;
	uint16_t *shared_gate_up;
	uint16_t *shared_act;
	uint16_t *shared_out;
	uint8_t *kda_state;
	uint16_t *kda_window;
	LmKvAccessError *kv_error;
	LmKvShardView kv_views[LingShape<G>::kAttnLayers];
	uint32_t kv_attached;
};

__global__ static void LingSplitRowsKernel(const uint16_t *fused, uint16_t *query, uint16_t *key, uint16_t *value, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 3u * width; index += gridDim.x * blockDim.x )
	{
		uint16_t *target = index < width ? query : index < 2u * width ? key : value;
		target[(uint64_t)row * width + index % width] = fused[(uint64_t)row * 3u * width + index];
	}
}

__global__ static void LingRopeQueryPackKernel(const uint16_t *latent, const uint16_t *q, uint16_t *query, uint32_t heads, uint32_t latent_dim,
	uint32_t nope, uint32_t rope)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t pair = (uint64_t)row * heads + head;
	for ( uint32_t element = threadIdx.x; element < latent_dim + rope; element += blockDim.x )
		query[pair * (latent_dim + rope) + element] = element < latent_dim ? latent[pair * latent_dim + element] :
			q[pair * (nope + rope) + nope + element - latent_dim];
}

__global__ static void LingHeadGateKernel(uint16_t *values, const uint16_t *gate_logits, uint32_t heads, uint32_t head_dim)
{
	const uint32_t row = blockIdx.x;
	for ( uint32_t index = threadIdx.x; index < heads * head_dim; index += blockDim.x )
	{
		const float gate = LmBf16ToFloat(LmFloatToBf16(1.0f / (1.0f + expf(-LmBf16ToFloat(gate_logits[(uint64_t)row * heads + index / head_dim])))));
		values[(uint64_t)row * heads * head_dim + index] = LmFloatToBf16(LmBf16ToFloat(values[(uint64_t)row * heads * head_dim + index]) * gate);
	}
}

__global__ static void LingSiluClampPackedKernel(const uint16_t *gate, const uint16_t *up, const float *weight, uint16_t *output, uint32_t width,
	float limit)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float g = LmBf16ToFloat(gate[(uint64_t)row * width + element]);
		const float u = fminf(fmaxf(LmBf16ToFloat(up[(uint64_t)row * width + element]), -limit), limit);
		output[(uint64_t)row * width + element] = LmFloatToBf16(fminf(g / (1.0f + expf(-g)), limit) * u * weight[row]);
	}
}

template<class G>
struct LingModel
{
	using S = LingShape<G>;
	using State = LingState<G>;

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + LG_THREADS - 1u) / LG_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			int32_t status;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.output_f32 = destination_f32;
			args.scale = LmScaleTensorNone();
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			status = LmSkinnyLaunch<LmBf16Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension,
			output_dimension, 0u, 0u, multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return Linear(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, multiprocessors, stream);
	}

	static SparkStatus Round(State *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *local, void *full)
	{
		return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL,
			rows, row_elements, local, full);
	}

	static SparkStatus Launched(void)
	{
		return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}

	static SparkStatus Kda(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sequences = step->sequences, sms = step->multiprocessors;
		uint8_t *state_pool = state->kda_state + (uint64_t)layer->recurrent_index * state->slots * state->kda_state_bytes;
		uint16_t *window = state->kda_window + (uint64_t)layer->recurrent_index * state->slots * (state->kda_window_bytes / sizeof(uint16_t));
		if ( Linear(layer->kda_qkv, state->normed, state->qkv, rows, G::kHidden, S::kKdaChannels, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->kda_beta, state->normed, state->beta, rows, G::kHidden, S::kKdaRankHeads, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kKdaRank != 0u )
		{
			if ( Linear(layer->kda_f_a, state->normed, state->kda_low_f, rows, G::kHidden, G::kKdaRank, sms, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_f, state->kda_low_f, state->decay_logit, rows, G::kKdaRank, S::kKdaWidth, sms, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_g_a, state->normed, state->kda_low_g, rows, G::kHidden, G::kKdaRank, sms, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_g, state->kda_low_g, state->kda_gate, rows, G::kKdaRank, S::kKdaWidth, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Linear(layer->kda_f, state->normed, state->decay_logit, rows, G::kHidden, S::kKdaWidth, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->kda_g, state->normed, state->kda_gate, rows, G::kHidden, S::kKdaWidth, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmCausalConvKernel<LG_THREADS,G::kConv,LM_CONV_SWISH,uint16_t>), dim3(sequences,(S::kKdaChannels + LG_THREADS - 1u) / LG_THREADS),
			LG_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0, state->qkv, layer->kda_conv,
			state->qkv, S::kKdaChannels, sequences, step->commit, step->sequence_row_indices);
		LingSplitRowsKernel<<<dim3(Blocks(S::kKdaChannels), rows), LG_THREADS, 0, stream>>>(state->qkv, state->query, state->key, state->value,
			S::kKdaWidth);
		LM_LAUNCH((LmBoundedDecayKernel<LG_THREADS,G::kKdaDim>), dim3(rows,S::kKdaRankHeads), LG_THREADS, 0, stream,
			state->decay_logit, layer->kda_dt_bias, layer->kda_a_log, state->retention, S::kKdaRankHeads, G::kKdaLowerBound, rows);
		if constexpr ( G::kKdaDoubledBeta )
			LM_LAUNCH((LmBf16SigmoidRowsKernel<LG_THREADS,true>), rows, LG_THREADS, 0, stream, state->beta, state->write_gate, S::kKdaRankHeads);
		else
			LM_LAUNCH((LmSigmoidRowsKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->beta, state->write_gate, S::kKdaRankHeads);
		LM_LAUNCH((LmDeltaRuleColumnKernel<LG_THREADS,G::kKdaDim,G::kKdaDim,float>), dim3(sequences,S::kKdaRankHeads,G::kKdaDim / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, (uint32_t)state->kda_state_bytes, step->recurrent_index, step->sequence_row_begin,
			(const uint32_t *)0, state->query, state->key, state->value, state->retention, state->write_gate, state->kda_mixed,
			S::kKdaRankHeads, 1u, sequences, step->commit, step->sequence_row_indices);
		LM_LAUNCH((LmHeadRmsNormSigmoidGateKernel<LG_THREADS>), dim3(S::kKdaRankHeads,rows), LG_THREADS, 0, stream,
			state->kda_mixed, state->kda_gate, layer->kda_norm, state->kda_mixed, rows, S::kKdaRankHeads, G::kKdaDim, G::kNormEpsilon);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->kda_out, state->kda_mixed, state->partial, rows, S::kKdaWidth, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mla(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t query_stride = (uint64_t)rows * S::kMlaRankHeads * S::kSlot;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * S::kMlaRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		const float scale = 1.0f / sqrtf((float)S::kQkDim);
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(layer->mla_q, state->normed, state->mla_q, rows, G::kHidden, S::kMlaRankHeads * S::kQkDim, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->mla_kv_a, state->normed, state->mla_kv, rows, G::kHidden, S::kSlot, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->mla_gate, state->normed, state->mla_gate, rows, G::kHidden, S::kMlaRankHeads, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmPerHeadProjectSplitKernel<LG_THREADS,G::kNope,G::kLatent,S::kQkDim,0u>), dim3(rows,S::kMlaRankHeads,G::kLatent / LM_PER_HEAD_SPLIT_OUTPUTS),
			LG_THREADS, 0, stream, state->mla_q, layer->mla_uk, state->mla_latent_query, S::kMlaRankHeads, rows);
		LM_LAUNCH((LmRopePerHeadKernel<LG_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,S::kMlaRankHeads), LG_THREADS, 0, stream,
			state->mla_q, step->positions, S::kMlaRankHeads, S::kQkDim, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kNope);
		LingRopeQueryPackKernel<<<dim3(rows, S::kMlaRankHeads), LG_THREADS, 0, stream>>>(state->mla_latent_query, state->mla_q, state->mla_query,
			S::kMlaRankHeads, G::kLatent, G::kNope, G::kRope);
		LM_LAUNCH((LmFusedResidualRmsNormKernel<LG_THREADS,float>), rows, LG_THREADS, (G::kLatent + 8u) * sizeof(float), stream,
			state->mla_kv, (const uint16_t *)0, layer->mla_kv_norm, (uint16_t *)0, state->mla_kv, G::kLatent, S::kSlot, G::kNormEpsilon);
		LM_LAUNCH((LmRopePerHeadKernel<LG_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,1u), LG_THREADS, 0, stream,
			state->mla_kv, step->positions, 1u, S::kSlot, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kLatent);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,LG_THREADS>), rows, LG_THREADS, 0, stream, *view, state->mla_kv, step->sequence_of_row,
			step->positions, rows, S::kSlot);
		if ( Launched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kMlaRankHeads * S::kSlot, state->mla_query,
			state->mla_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( (rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS ?
				LmLatentShardPrefillLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->mla_gathered, query_stride, S::kMlaRankHeads,
					step->sequence_of_row, step->context_length, step->positions, scale, state->attn_send, partial_stride, rows, stream) :
				LmLatentShardPartialLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->mla_gathered, query_stride, S::kMlaRankHeads,
					step->sequence_of_row, step->context_length, step->positions, (const uint32_t *)0, 0u, 0u, scale, state->attn_send, partial_stride,
					rows, stream)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->attn_send,
			state->attn_received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardMergeLaunch<G::kLatent>(state->attn_received, partial_stride, degree, S::kMlaRankHeads, state->mla_latent, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmPerHeadProjectSplitKernel<LG_THREADS,G::kLatent,G::kValueDim>), dim3(rows,S::kMlaRankHeads,G::kValueDim / LM_PER_HEAD_SPLIT_OUTPUTS),
			LG_THREADS, 0, stream, state->mla_latent, layer->mla_uv, state->mla_value, S::kMlaRankHeads, rows);
		LingHeadGateKernel<<<rows, LG_THREADS, 0, stream>>>(state->mla_value, state->mla_gate, S::kMlaRankHeads, G::kValueDim);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->mla_out, state->mla_value, state->partial, rows, S::kMlaRankHeads * G::kValueDim, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Gqa(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * LmGqaShardRecordFloats(G::kGqaHeads, G::kGqaHeadDim, degree) + record_unit - 1u) /
			record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(layer->gqa_qkv, state->normed, state->gqa_local, rows, G::kHidden, S::kGqaRankFused, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->gqa_gate, state->normed, state->gqa_gate, rows, G::kHidden, S::kGqaRankWidth, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kGqaRankFused, state->gqa_local, state->gqa_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		LmRankMajorRowsKernel<LG_THREADS><<<dim3(Blocks(S::kGqaFused), rows), LG_THREADS, 0, stream>>>(state->gqa_gathered, state->gqa_full, rows, S::kGqaRankFused,
			degree);
		LmSliceColumnsKernel<LG_THREADS><<<dim3(Blocks(S::kGqaWidth), rows), LG_THREADS, 0, stream>>>(state->gqa_full, state->gqa_query, S::kGqaFused, 0u,
			S::kGqaWidth);
		LmSliceColumnsKernel<LG_THREADS><<<dim3(Blocks(S::kSlot), rows), LG_THREADS, 0, stream>>>(state->gqa_full, state->gqa_slot, S::kGqaFused, S::kGqaWidth,
			S::kSlot);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,LG_THREADS>), rows, LG_THREADS, 0, stream, *view, state->gqa_slot, step->sequence_of_row,
			step->positions, rows, S::kSlot);
		if ( Launched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kGqaKvHeads,G::kGqaHeadDim,G::kGqaHeadDim>(*view, state->gqa_query, G::kGqaHeads,
				step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kGqaHeadDim), state->attn_send, partial_stride, rows,
				stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->attn_send,
			state->attn_received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kGqaHeadDim>(state->attn_received, partial_stride, degree, G::kGqaHeads, state->tp_rank, state->gqa_merged, rows,
			stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmOutputGateKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->gqa_merged, state->gqa_gate, S::kGqaRankWidth);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->gqa_out, state->gqa_merged, state->partial, rows, S::kGqaRankWidth, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Attention(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		if constexpr ( S::kGqa )
			return Gqa(state, layer, step, stream);
		else
			return Mla(state, layer, step, stream);
	}

	static SparkStatus Dense(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		if ( Linear(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmSiluMulLimitKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true, INFINITY);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->ffn_down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Moe(State *state, const LingLayer *layer, uint32_t index, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * G::kTopK;
		const float routed_limit = G::RoutedLimit(index), shared_limit = G::SharedLimit(index);
		LmLocalExpertRoutes expert_routes;
		if ( Linear(layer->router, state->normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, sms, stream) != LM_LAUNCH_OK ||
			LmTopkGroupLimitedLaunch<LG_THREADS,G::kExperts,G::kGroups,G::kTopkGroups,G::kTopK>(rows, state->router_logits, layer->expert_bias,
				state->route_expert, state->route_weight, G::kRouteScale, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<LG_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, LM_MOE_GEMM_TILE_N, state->tile_prefix_up,
			state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = G::kTopK;
		expert_routes.local_experts = G::kLocalExperts;
		expert_routes.multiprocessors = sms;
		if ( Launched() != SPARK_STATUS_OK ||
			LmLocalExpertsUp<LmBf16Format>(expert_routes, layer->moe_w1, LmScaleTensorNone(), state->normed, state->expert_gate, G::kHidden,
				G::kMoeInter, stream) != LM_LAUNCH_OK ||
			LmLocalExpertsUp<LmBf16Format>(expert_routes, layer->moe_w3, LmScaleTensorNone(), state->normed, state->expert_up, G::kHidden,
				G::kMoeInter, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LingSiluClampPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), LG_THREADS, 0, stream>>>(state->expert_gate, state->expert_up,
			state->packed_weight, state->expert_act, G::kMoeInter, routed_limit > 0.0f ? routed_limit : INFINITY);
		if ( Launched() != SPARK_STATUS_OK ||
			LmLocalExpertsDown<LmBf16Format>(expert_routes, layer->moe_w2, LmScaleTensorNone(), state->expert_act, state->expert_out, G::kMoeInter,
				G::kHidden, stream) != LM_LAUNCH_OK ||
			Linear(layer->shared_gate_up, state->normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmSiluMulLimitKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows, true,
			shared_limit > 0.0f ? shared_limit : INFINITY);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->shared_down, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), LG_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return Launched();
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream_void)
	{
		State *state = (State *)model;
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		uint16_t *hidden = step->hidden_bf16;
		SparkStatus status = SPARK_STATUS_OK;
		if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const LingLayer *layer = &state->layers[index];
			LM_LAUNCH((LmFusedResidualRmsNormKernel<LG_THREADS,float>), rows, LG_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, index == 0u ? (const uint16_t *)0 : state->partial, layer->input_norm, index == 0u ? (uint16_t *)0 : hidden,
				state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
			status = index % G::kPeriod == G::kPhase ? Attention(state, layer, step, stream) : Kda(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			LM_LAUNCH((LmFusedResidualRmsNormKernel<LG_THREADS,float>), rows, LG_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, state->partial, layer->post_norm, hidden, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
			status = index < G::kDenseLayers ? Dense(state, layer, step, stream) : Moe(state, layer, index, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		LmAddRowsKernel<LG_THREADS><<<dim3(Blocks(G::kHidden), rows), LG_THREADS, 0, stream>>>(hidden, state->partial, hidden, rows, G::kHidden);
		return Launched();
	}

	static SparkStatus Entry(State *state, const char *name, const char *kind, uint64_t *offset, uint64_t *bytes, uint32_t *rows)
	{
		SparkNamedPackEntry entry;
		if ( SparkNamedPackEntryLoad(&state->pack, name, &entry) != SPARK_STATUS_OK || strcmp(entry.kind, kind) != 0 || entry.shape_count < 1u )
		{
			fprintf(stderr, "%s: pack tensor %s missing or not %s\n", G::Tag(), name, kind);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		*offset = state->pack.payload_base + entry.payload_offset;
		*bytes = entry.bytes;
		if ( rows != 0 )
			*rows = entry.shape[0];
		return SPARK_STATUS_OK;
	}

	static SparkStatus Plan(State *state, uint64_t *bytes, uint8_t *base)
	{
		const uint64_t rows = state->max_rows, slots = state->slots, routes = rows * G::kTopK, groups = G::kLocalExperts + 2u, degree = state->tp_degree;
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t);
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->qkv, rows * S::kKdaChannels * sizeof(uint16_t) },
			{ (void **)&state->kda_low_f, rows * G::kKdaRank * sizeof(uint16_t) },
			{ (void **)&state->kda_low_g, rows * G::kKdaRank * sizeof(uint16_t) },
			{ (void **)&state->decay_logit, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->kda_gate, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->beta, rows * S::kKdaRankHeads * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->key, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->retention, rows * S::kKdaWidth * sizeof(float) },
			{ (void **)&state->write_gate, rows * S::kKdaRankHeads * sizeof(float) },
			{ (void **)&state->kda_mixed, rows * S::kKdaWidth * sizeof(uint16_t) },
			{ (void **)&state->mla_q, rows * S::kMlaRankHeads * S::kQkDim * sizeof(uint16_t) },
			{ (void **)&state->mla_latent_query, rows * S::kMlaRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->mla_query, rows * S::kMlaRankHeads * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->mla_kv, rows * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->mla_gathered, (uint64_t)state->tp_degree * rows * S::kMlaRankHeads * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->attn_send, (uint64_t)state->tp_degree * state->partial_stride * sizeof(float) },
			{ (void **)&state->attn_received, (uint64_t)state->tp_degree * state->partial_stride * sizeof(float) },
			{ (void **)&state->gqa_local, rows * S::kGqaRankFused * sizeof(uint16_t) },
			{ (void **)&state->gqa_gathered, degree * rows * S::kGqaRankFused * sizeof(uint16_t) },
			{ (void **)&state->gqa_full, rows * S::kGqaFused * sizeof(uint16_t) },
			{ (void **)&state->gqa_query, rows * S::kGqaWidth * sizeof(uint16_t) },
			{ (void **)&state->gqa_slot, S::kGqa ? rows * S::kSlot * sizeof(uint16_t) : 0u },
			{ (void **)&state->gqa_gate, rows * S::kGqaRankWidth * sizeof(uint16_t) },
			{ (void **)&state->gqa_merged, rows * S::kGqaRankWidth * sizeof(uint16_t) },
			{ (void **)&state->mla_latent, rows * S::kMlaRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->mla_value, rows * S::kMlaRankHeads * G::kValueDim * sizeof(uint16_t) },
			{ (void **)&state->mla_gate, rows * S::kMlaRankHeads * sizeof(uint16_t) },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->packed_weight, routes * sizeof(float) },
			{ (void **)&state->expert_gate, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->shared_gate_up, rows * 2u * G::kSharedRows * sizeof(uint16_t) },
			{ (void **)&state->shared_act, rows * G::kSharedRows * sizeof(uint16_t) },
			{ (void **)&state->shared_out, width },
			{ (void **)&state->kda_state, (uint64_t)S::kKdaLayers * slots * state->kda_state_bytes },
			{ (void **)&state->kda_window, (uint64_t)S::kKdaLayers * slots * state->kda_window_bytes },
			{ (void **)&state->kv_error, sizeof(LmKvAccessError) },
		};
		uint64_t cursor = 0u;
		for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
		{
			cursor = (cursor + 255u) & ~255ull;
			if ( base != 0 )
				*plan[index].target = plan[index].bytes != 0u ? base + cursor : 0;
			cursor += plan[index].bytes;
		}
		*bytes = cursor;
		return SPARK_STATUS_OK;
	}

	static void Close(void *model)
	{
		State *state = (State *)model;
		if ( state == 0 )
			return;
		if ( state->arena != 0 )
			(void)cudaFree(state->arena);
		if ( state->pack_open != 0u )
			SparkNamedPackClose(&state->pack);
		free(state);
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		State *state;
		SparkStatus status;
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u, embed_rows = 0u;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree || configuration->max_input_row_count == 0u ||
			configuration->state_budget_bytes == 0u || configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state = (State *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->max_rows = configuration->max_input_row_count;
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, LG_PACK_ALIGNMENT, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || (G::kDenseLayers != 0u && (state->ffn_rows == 0u || state->ffn_rows % 64u != 0u)) || experts != G::kExperts ||
			local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != (G::kLocalExperts == G::kExperts ? 0u : rank * G::kLocalExperts) )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, experts, local, state->expert_first, top_k, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		state->kda_state_bytes = (uint64_t)S::kKdaRankHeads * G::kKdaDim * G::kKdaDim * sizeof(float);
		state->kda_window_bytes = (uint64_t)S::kKdaChannels * G::kConv * sizeof(uint16_t);
		if ( S::kGqa && LmGqaShardGeometryValid(G::kGqaHeads, G::kGqaKvHeads, G::kGqaHeadDim, state->tp_degree) == 0u )
		{
			fprintf(stderr, "%s: %u query heads over %u KV heads do not shard over %u ranks\n", G::Tag(), G::kGqaHeads, G::kGqaKvHeads, state->tp_degree);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		state->partial_stride = ((uint64_t)state->max_rows * (S::kGqa ? LmGqaShardRecordFloats(G::kGqaHeads, G::kGqaHeadDim, state->tp_degree) :
			S::kMlaRankHeads * S::kRecord) + record_unit - 1u) / record_unit * record_unit;
		status = Plan(state, &state->arena_bytes, 0);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes is over the %llu-byte budget\n", G::Tag(), (unsigned long long)state->arena_bytes,
				(unsigned long long)configuration->state_budget_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = G::kHidden;
		geometry->vocab = G::kVocab;
		geometry->total_layers = G::kLayers;
		geometry->layer_count = G::kLayers;
		geometry->rms_epsilon = G::kNormEpsilon;
		geometry->pack_bytes = state->pack.file_bytes;
		geometry->kv_layer_count = S::kAttnLayers;
		geometry->kv_layer_page_bytes = S::Kv::kPageBytes;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->recurrent_bytes = RecurrentBytes(state);
		geometry->head_norm_f32 = 1u;
		if ( Entry(state, "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			Entry(state, "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
			Entry(state, "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->embed_rows = embed_rows;
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		(void)model;
		if ( manifest == 0 || manifest->range_count != 0u )
		{
			fprintf(stderr, "%s: the pack carries its experts as plain tensors, so the weightd manifest has no expert ranges\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	template<class T>
	static SparkStatus Bind(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, T **field)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		uint64_t offset, bytes;
		const void *bound = 0;
		snprintf(key, sizeof(key), "layers.%u.%s", index, name);
		if ( Entry(state, key, kind, &offset, &bytes, 0) != SPARK_STATUS_OK ||
			SparkWeightdLazyPackSlice(lazy_pack, offset, bytes, &bound) != SPARK_STATUS_OK || bound == 0 )
		{
			fprintf(stderr, "%s: weightd has no resident bytes for %s\n", G::Tag(), key);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		*field = (T *)bound;
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindModel(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		uint32_t kda = 0u, attention = 0u;
		(void)geometry;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			LingLayer *layer = &state->layers[index];
			SparkStatus status = Bind(state, lazy_pack, index, "input_norm", "f32", &layer->input_norm);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "post_norm", "f32", &layer->post_norm);
			if ( index % G::kPeriod == G::kPhase )
			{
				if constexpr ( S::kGqa )
				{
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "gqa_qkv", "bf16", &layer->gqa_qkv);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "gqa_gate", "bf16", &layer->gqa_gate);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "gqa_out", "bf16", &layer->gqa_out);
				}
				else
				{
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_q", "bf16", &layer->mla_q);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_kv_a", "bf16", &layer->mla_kv_a);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_kv_norm", "f32", &layer->mla_kv_norm);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_uk", "bf16", &layer->mla_uk);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_uv", "bf16", &layer->mla_uv);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_gate", "bf16", &layer->mla_gate);
					if ( status == SPARK_STATUS_OK )
						status = Bind(state, lazy_pack, index, "mla_out", "bf16", &layer->mla_out);
				}
				layer->kv_index = attention++;
			}
			else
			{
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_qkv", "bf16", &layer->kda_qkv);
				if ( G::kKdaRank != 0u && status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_f_a", "bf16", &layer->kda_f_a);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_f", "bf16", &layer->kda_f);
				if ( G::kKdaRank != 0u && status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_g_a", "bf16", &layer->kda_g_a);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_g", "bf16", &layer->kda_g);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_beta", "bf16", &layer->kda_beta);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_conv", "bf16", &layer->kda_conv);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_a_log", "f32", &layer->kda_a_log);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_dt_bias", "f32", &layer->kda_dt_bias);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_norm", "bf16", &layer->kda_norm);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "kda_out", "bf16", &layer->kda_out);
				layer->recurrent_index = kda++;
			}
			if ( index < G::kDenseLayers )
			{
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "ffn_gate_up", "bf16", &layer->ffn_gate_up);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "ffn_down", "bf16", &layer->ffn_down);
			}
			else
			{
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "router", "bf16", &layer->router);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "expert_bias", "f32", &layer->expert_bias);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_w1", "bf16", &layer->moe_w1);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_w3", "bf16", &layer->moe_w3);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_w2", "bf16", &layer->moe_w2);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "shared_gate_up", "bf16", &layer->shared_gate_up);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "shared_down", "bf16", &layer->shared_down);
			}
			if ( status != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || Plan(state, &state->arena_bytes, (uint8_t *)state->arena) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != S::kAttnLayers || kv->layer_page_bytes != S::Kv::kPageBytes || kv->sequence_count > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < S::kAttnLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::Kv>(&state->kv_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes, kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}

	static uint64_t RecurrentBytes(const State *state)
	{
		return (uint64_t)S::kKdaLayers * (state->kda_state_bytes + state->kda_window_bytes);
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		State *state = (State *)model;
		uint8_t *pools[2] = { state->kda_state, (uint8_t *)state->kda_window };
		const uint64_t widths[2] = { state->kda_state_bytes, state->kda_window_bytes };
		uint8_t *packed = (uint8_t *)buffer;
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots || buffer == 0 || bytes != RecurrentBytes(state) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < 2u && error == cudaSuccess; ++part )
		{
			uint8_t *pool = pools[part] + (uint64_t)slot * widths[part];
			const size_t pitch = (size_t)(widths[part] * state->slots);
			error = to_buffer != 0u ?
				cudaMemcpy2DAsync(packed, (size_t)widths[part], pool, pitch, (size_t)widths[part], S::kKdaLayers, cudaMemcpyDefault, (cudaStream_t)stream) :
				cudaMemcpy2DAsync(pool, pitch, packed, (size_t)widths[part], (size_t)widths[part], S::kKdaLayers, cudaMemcpyDefault, (cudaStream_t)stream);
			packed += widths[part] * S::kKdaLayers;
		}
		if ( error == cudaSuccess && stream == 0 )
			error = cudaStreamSynchronize(0);
		return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		State *state = (State *)model;
		uint8_t *pools[2] = { state->kda_state, (uint8_t *)state->kda_window };
		const uint64_t widths[2] = { state->kda_state_bytes, state->kda_window_bytes };
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < 2u && error == cudaSuccess; ++part )
			error = cudaMemset2DAsync(pools[part] + (uint64_t)slot * widths[part], (size_t)(widths[part] * state->slots), 0, (size_t)widths[part],
				S::kKdaLayers, (cudaStream_t)stream);
		return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}

	static SparkStatus Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
	{
		(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static const void *ProbeBuffers(void *model)
	{
		return model;
	}

	static void Report(void *model, uint32_t rank)
	{
		const State *state = (const State *)model;
		fprintf(stderr, "%s rank=%u arena_bytes=%llu slots=%u rows=%u ffn_rows=%u expert_first=%u\n", G::ReportTag(), rank,
			(unsigned long long)state->arena_bytes, state->slots, state->max_rows, state->ffn_rows, state->expert_first);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			G::Tag(),
			G::Name(),
			"official",
			Open,
			ManifestCheck,
			BindModel,
			Close,
			Step,
			Sideband,
			AttachKv,
			RecurrentCopy,
			ResetSlot,
			ProbeBuffers,
			Report
		};
		return &interface_table;
	}
};
