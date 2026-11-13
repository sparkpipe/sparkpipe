#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/dense_rows.cuh"
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

#include "sparkpipe/spark_ling_family.h"

#define LG_THREADS 256u

template<class G>
struct LingShape : LingLayout<G>
{
	static constexpr uint32_t kRecord = LM_LATENT_SHARD_RECORD_FLOATS(G::kLatent);
	using Kv = LmKvGeometry<LingLayout<G>::kSlot * 2u, G::kPageSlots, true>;
	static_assert(LingLayout<G>::kMlaRankHeads * LingLayout<G>::kSlot <= G::kHidden, "a rank's packed queries fit one collective row");
	static_assert(!LingLayout<G>::kGqa || (G::kMlaHeads == 0u && G::kGqaHeads % G::kTpDegree == 0u && LingLayout<G>::kGqaFused % G::kTpDegree == 0u &&
		LingLayout<G>::kGqaRankFused <= G::kHidden && LingLayout<G>::kGqaRankWidth % G::kGqaHeadDim == 0u),
		"a GQA rank owns whole query heads and its fused rows fit one collective row");
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
struct LingState : CudaStagePackState
{
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint64_t partial_stride;
	uint64_t kda_state_bytes;
	uint64_t kda_window_bytes;
	LingLayer layers[G::kLayers];
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
	uint16_t *replay_qkv;
	float *replay_retention;
	float *replay_write_gate;
	float *kda_snapshots;
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
	LmKvShardView kv_views[LingShape<G>::kAttnLayers];
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
		const float gate = LmBf16ToFloat(LmFloatToBf16(SparkMathSigmoid(LmBf16ToFloat(gate_logits[(uint64_t)row * heads + index / head_dim]))));
		values[(uint64_t)row * heads * head_dim + index] = LmFloatToBf16(LmBf16ToFloat(values[(uint64_t)row * heads * head_dim + index]) * gate);
	}
}

template<bool ROUNDED, bool WEIGHTED>
__global__ static void LingSiluClampPackedKernel(const uint16_t *gate, const uint16_t *up, const float *weight, uint16_t *output, uint32_t width,
	float limit)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float g = LmBf16ToFloat(gate[(uint64_t)row * width + element]);
		const float u = fminf(fmaxf(LmBf16ToFloat(up[(uint64_t)row * width + element]), -limit), limit);
		const float activated = fminf(ROUNDED ? SparkMathRoundBf16(SparkMathSilu(g)) : SparkMathSilu(g), limit) * u;
		if constexpr ( WEIGHTED )
			output[(uint64_t)row * width + element] = LmFloatToBf16(activated * weight[row]);
		else
			output[(uint64_t)row * width + element] = LmFloatToBf16(activated);
	}
}

template<class G>
struct LingModel : CudaStage<LingModel<G>>
{
	using S = LingShape<G>;
	using State = LingState<G>;
	using Layer = LingLayer;
	using Schedule = LingSchedule<G, LingModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::Kv, &State::kv_views, 0u, S::kAttnLayers>>;
	using LaneShared = CudaStageFields<&State::kda_state, &State::kda_window, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = G::kVerifyRows != 0u;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kNormEpsilon;
	static constexpr uint32_t kPackAlignment = LG_PACK_ALIGNMENT;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		cudaStream_t stream;
	};

	static const char *Tag(void)
	{
		return G::Tag();
	}

	static const char *ReportTag(void)
	{
		return G::ReportTag();
	}

	static const char *WeightdModel(void)
	{
		return G::Name();
	}

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + LG_THREADS - 1u) / LG_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			step->commit == 0u ? 1u : 0u, step->multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return Linear(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, step, stream);
	}

	static uint16_t *ReplayQkv(State *state, const LingLayer *layer)
	{
		return state->replay_qkv + (uint64_t)layer->recurrent_index * G::kVerifyRows * S::kKdaChannels;
	}

	static float *ReplayRetention(State *state, const LingLayer *layer)
	{
		return state->replay_retention + (uint64_t)layer->recurrent_index * G::kVerifyRows * S::kKdaWidth;
	}

	static float *ReplayWriteGate(State *state, const LingLayer *layer)
	{
		return state->replay_write_gate + (uint64_t)layer->recurrent_index * G::kVerifyRows * S::kKdaRankHeads;
	}

	static SparkStatus Kda(State *state, const Frame &frame, const LingLayer &entry)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows, sequences = step->sequences;
		uint8_t *state_pool = state->kda_state + (uint64_t)layer->recurrent_index * state->slots * state->kda_state_bytes;
		uint16_t *window = state->kda_window + (uint64_t)layer->recurrent_index * state->slots * (state->kda_window_bytes / sizeof(uint16_t));
		if ( Linear(layer->kda_qkv, state->normed, state->qkv, rows, G::kHidden, S::kKdaChannels, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->kda_beta, state->normed, state->beta, rows, G::kHidden, S::kKdaRankHeads, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kKdaRank != 0u )
		{
			if ( Linear(layer->kda_f_a, state->normed, state->kda_low_f, rows, G::kHidden, G::kKdaRank, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_f, state->kda_low_f, state->decay_logit, rows, G::kKdaRank, S::kKdaWidth, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_g_a, state->normed, state->kda_low_g, rows, G::kHidden, G::kKdaRank, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->kda_g, state->kda_low_g, state->kda_gate, rows, G::kKdaRank, S::kKdaWidth, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Linear(layer->kda_f, state->normed, state->decay_logit, rows, G::kHidden, S::kKdaWidth, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->kda_g, state->normed, state->kda_gate, rows, G::kHidden, S::kKdaWidth, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const bool replay = G::kVerifyRows != 0u && step->commit == 0u;
		if ( step->commit == 0u && (G::kVerifyRows == 0u || rows > G::kVerifyRows) )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( replay && cudaMemcpyAsync(ReplayQkv(state, layer), state->qkv, (uint64_t)rows * S::kKdaChannels * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
			stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LmCausalConvLaunch<G::kConv,(G::kKdaConvRoundedSum ? LM_CONV_SWISH : LM_CONV_SWISH_ONCE),uint16_t>(window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0,
			step->tree != 0u ? ReplayQkv(state, layer) : state->qkv, layer->kda_conv, state->qkv, S::kKdaChannels, sequences, step->commit, step->sequence_row_indices,
			rows, stream, step->tree != 0u ? step->parent_rows : (const uint32_t *)0);
		LingSplitRowsKernel<<<dim3(Blocks(S::kKdaChannels), rows), LG_THREADS, 0, stream>>>(state->qkv, state->query, state->key, state->value,
			S::kKdaWidth);
		LM_LAUNCH((LmBoundedDecayKernel<LG_THREADS,G::kKdaDim>), dim3(rows,S::kKdaRankHeads), LG_THREADS, 0, stream,
			state->decay_logit, layer->kda_dt_bias, layer->kda_a_log, state->retention, S::kKdaRankHeads, G::kKdaLowerBound, rows);
		if constexpr ( G::kKdaDoubledBeta )
			LM_LAUNCH((LmBf16SigmoidRowsKernel<LG_THREADS,true>), rows, LG_THREADS, 0, stream, state->beta, state->write_gate, S::kKdaRankHeads);
		else
			LM_LAUNCH((LmSigmoidRowsKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->beta, state->write_gate, S::kKdaRankHeads);
		if ( replay && (cudaMemcpyAsync(ReplayRetention(state, layer), state->retention, (uint64_t)rows * S::kKdaWidth * sizeof(float),
				cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(ReplayWriteGate(state, layer), state->write_gate, (uint64_t)rows * S::kKdaRankHeads * sizeof(float),
				cudaMemcpyDeviceToDevice, stream) != cudaSuccess) )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( LmDeltaRuleLaunch<LG_THREADS,G::kKdaDim,G::kKdaDim,float>(step->rows, state_pool, (uint32_t)state->kda_state_bytes, step->recurrent_index,
				step->sequence_row_begin, (const uint32_t *)0, state->query, state->key, state->value, state->retention, state->write_gate, state->kda_mixed,
				S::kKdaRankHeads, 1u, sequences, step->commit, step->sequence_row_indices, stream, step->tree != 0u ? step->state_loads : (const uint32_t *)0,
				step->tree != 0u ? step->state_saves : (const uint32_t *)0, step->tree != 0u ? state->kda_snapshots : (float *)0) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if constexpr ( G::kKdaNormRoundOnce )
			LM_LAUNCH((LmHeadRmsNormSigmoidGateOnceKernel<LG_THREADS>), dim3(S::kKdaRankHeads,rows), LG_THREADS, 0, stream,
				state->kda_mixed, state->kda_gate, layer->kda_norm, state->kda_mixed, rows, S::kKdaRankHeads, G::kKdaDim, G::kNormEpsilon);
		else
			LM_LAUNCH((LmHeadRmsNormSigmoidGateKernel<LG_THREADS>), dim3(S::kKdaRankHeads,rows), LG_THREADS, 0, stream,
				state->kda_mixed, state->kda_gate, layer->kda_norm, state->kda_mixed, rows, S::kKdaRankHeads, G::kKdaDim, G::kNormEpsilon);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->kda_out, state->kda_mixed, state->partial, rows, S::kKdaWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mla(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, degree = state->tp_degree;
		const uint64_t query_stride = (uint64_t)rows * S::kMlaRankHeads * S::kSlot;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * S::kMlaRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		const float scale = 1.0f / sqrtf((float)S::kQkDim);
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->tree != 0u )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		if ( Linear(layer->mla_q, state->normed, state->mla_q, rows, G::kHidden, S::kMlaRankHeads * S::kQkDim, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->mla_kv_a, state->normed, state->mla_kv, rows, G::kHidden, S::kSlot, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->mla_gate, state->normed, state->mla_gate, rows, G::kHidden, S::kMlaRankHeads, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmPerHeadProjectSplitKernel<LG_THREADS,G::kNope,G::kLatent,S::kQkDim,0u>), dim3(rows,S::kMlaRankHeads,G::kLatent / LM_PER_HEAD_SPLIT_OUTPUTS),
			LG_THREADS, 0, stream, state->mla_q, layer->mla_uk, state->mla_latent_query, S::kMlaRankHeads, rows);
		LM_LAUNCH((LmRopePerHeadKernel<LG_THREADS,LM_ROPE_INTERLEAVED,LM_ROPE_ROUNDED_TERMS>), dim3(rows,S::kMlaRankHeads), LG_THREADS, 0, stream,
			state->mla_q, step->positions, S::kMlaRankHeads, S::kQkDim, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kNope);
		LingRopeQueryPackKernel<<<dim3(rows, S::kMlaRankHeads), LG_THREADS, 0, stream>>>(state->mla_latent_query, state->mla_q, state->mla_query,
			S::kMlaRankHeads, G::kLatent, G::kNope, G::kRope);
		LM_LAUNCH((LmFusedResidualRmsNormKernel<LG_THREADS,float,G::kNormRoundFirst>), rows, LG_THREADS, (G::kLatent + 8u) * sizeof(float), stream,
			state->mla_kv, (const uint16_t *)0, layer->mla_kv_norm, (uint16_t *)0, state->mla_kv, G::kLatent, S::kSlot, G::kNormEpsilon);
		LM_LAUNCH((LmRopePerHeadKernel<LG_THREADS,LM_ROPE_INTERLEAVED,LM_ROPE_ROUNDED_TERMS>), dim3(rows,1u), LG_THREADS, 0, stream,
			state->mla_kv, step->positions, 1u, S::kSlot, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kLatent);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,LG_THREADS>), rows, LG_THREADS, 0, stream, *view, state->mla_kv, step->sequence_of_row,
			step->positions, rows, S::kSlot);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kMlaRankHeads * S::kSlot, state->mla_query,
			state->mla_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( (rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS && step->commit != 0u ?
				LmLatentShardPrefillLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->mla_gathered, query_stride, S::kMlaRankHeads,
					step->sequence_of_row, step->context_length, step->positions, scale, state->attn_send, partial_stride, rows, stream) :
				LmLatentShardAttendLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->mla_gathered, query_stride, S::kMlaRankHeads,
					step->sequence_of_row, step->context_length, step->positions, (const uint32_t *)0, 0u, 0u, scale, state->attn_send, partial_stride,
					rows, stream)) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->attn_send,
			state->attn_received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardMergeLaunch<G::kLatent>(state->attn_received, partial_stride, degree, S::kMlaRankHeads, state->mla_latent, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmPerHeadProjectSplitKernel<LG_THREADS,G::kLatent,G::kValueDim>), dim3(rows,S::kMlaRankHeads,G::kValueDim / LM_PER_HEAD_SPLIT_OUTPUTS),
			LG_THREADS, 0, stream, state->mla_latent, layer->mla_uv, state->mla_value, S::kMlaRankHeads, rows);
		LingHeadGateKernel<<<rows, LG_THREADS, 0, stream>>>(state->mla_value, state->mla_gate, S::kMlaRankHeads, G::kValueDim);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->mla_out, state->mla_value, state->partial, rows, S::kMlaRankHeads * G::kValueDim, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Gqa(State *state, const LingLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, degree = state->tp_degree;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * LmGqaShardRecordFloats(G::kGqaHeads, G::kGqaHeadDim, degree) + record_unit - 1u) /
			record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(layer->gqa_qkv, state->normed, state->gqa_local, rows, G::kHidden, S::kGqaRankFused, step, stream) != LM_LAUNCH_OK ||
			Linear(layer->gqa_gate, state->normed, state->gqa_gate, rows, G::kHidden, S::kGqaRankWidth, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kGqaRankFused, state->gqa_local, state->gqa_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		LmRankMajorRowsKernel<LG_THREADS><<<dim3(Blocks(S::kGqaFused), rows), LG_THREADS, 0, stream>>>(state->gqa_gathered, state->gqa_full, rows, S::kGqaRankFused,
			degree);
		LmSliceColumnsKernel<LG_THREADS><<<dim3(Blocks(S::kGqaWidth), rows), LG_THREADS, 0, stream>>>(state->gqa_full, state->gqa_query, S::kGqaFused, 0u,
			S::kGqaWidth);
		LmSliceColumnsKernel<LG_THREADS><<<dim3(Blocks(S::kSlot), rows), LG_THREADS, 0, stream>>>(state->gqa_full, state->gqa_slot, S::kGqaFused, S::kGqaWidth,
			S::kSlot);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,LG_THREADS>), rows, LG_THREADS, 0, stream, *view, state->gqa_slot, step->sequence_of_row,
			step->tree != 0u ? step->slots : step->positions, rows, S::kSlot);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kGqaKvHeads,G::kGqaHeadDim,G::kGqaHeadDim>(*view, state->gqa_query, G::kGqaHeads,
				step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kGqaHeadDim), state->attn_send, partial_stride, rows,
				stream, (const uint32_t *)0, 0u, 1u, 0u, step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
				step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->attn_send,
			state->attn_received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kGqaHeadDim>(state->attn_received, partial_stride, degree, G::kGqaHeads, state->tp_rank, state->gqa_merged, rows,
			stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmOutputGateKernel<LG_THREADS,true>), rows, LG_THREADS, 0, stream, state->gqa_merged, state->gqa_gate, S::kGqaRankWidth);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->gqa_out, state->gqa_merged, state->partial, rows, S::kGqaRankWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Attention(State *state, const Frame &frame, const LingLayer &entry)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		if constexpr ( S::kGqa )
			return Gqa(state, layer, step, stream);
		else
			return Mla(state, layer, step, stream);
	}

	static SparkStatus Dense(State *state, const Frame &frame, const LingLayer &entry)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows;
		if ( Linear(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kFfnSiluRounded )
			LM_LAUNCH((LmSiluRoundedMulLimitKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true, INFINITY);
		else
			LM_LAUNCH((LmSiluMulLimitKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true, INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->ffn_down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Experts(State *state, const Frame &frame, const LingLayer &entry, uint32_t index)
	{
		const LingLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows, routes = rows * G::kTopK, exact = step->commit == 0u ? 1u : 0u;
		const uint16_t *normed = state->normed;
		const float routed_limit = G::RoutedLimit(index), shared_limit = G::SharedLimit(index);
		LmLocalExpertRoutes expert_routes;
		if ( Linear(layer->router, normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK ||
			LmTopkGroupLimitedLaunch<LG_THREADS,G::kExperts,G::kGroups,G::kTopkGroups,G::kTopK>(rows, state->router_logits, layer->expert_bias,
				state->route_expert, state->route_weight, G::kRouteScale, step->route_fault, stream) != cudaSuccess )
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
		expert_routes.multiprocessors = step->multiprocessors;
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmLocalExpertsUp<LmBf16Format>(expert_routes, layer->moe_w1, LmScaleTensorNone(), normed, state->expert_gate, G::kHidden,
				G::kMoeInter, stream, exact) != LM_LAUNCH_OK ||
			LmLocalExpertsUp<LmBf16Format>(expert_routes, layer->moe_w3, LmScaleTensorNone(), normed, state->expert_up, G::kHidden,
				G::kMoeInter, stream, exact) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LingSiluClampPackedKernel<G::kFfnSiluRounded,!G::kExpertWeightAfterDown><<<dim3(Blocks(G::kMoeInter), routes), LG_THREADS, 0, stream>>>(state->expert_gate,
			state->expert_up, state->packed_weight, state->expert_act, G::kMoeInter, routed_limit > 0.0f ? routed_limit : INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmLocalExpertsDown<LmBf16Format>(expert_routes, layer->moe_w2, LmScaleTensorNone(), state->expert_act, state->expert_out, G::kMoeInter,
				G::kHidden, stream, exact) != LM_LAUNCH_OK ||
			Linear(layer->shared_gate_up, normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kFfnSiluRounded )
			LM_LAUNCH((LmSiluRoundedMulLimitKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows, true,
				shared_limit > 0.0f ? shared_limit : INFINITY);
		else
			LM_LAUNCH((LmSiluMulLimitKernel<LG_THREADS>), rows, LG_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows, true,
				shared_limit > 0.0f ? shared_limit : INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->shared_down, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kExpertWeightAfterDown )
			LmMoeLocalWeightedFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), LG_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
				state->route_expert, state->route_weight, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		else
			LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), LG_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
				state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( step->tree != 0u && (!S::kGqa || G::kVerifyRows == 0u || step->commit != 0u || step->sequences != 1u) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		frame->step = step;
		frame->stream = (cudaStream_t)stream;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<LG_THREADS,float,G::kNormRoundFirst>), frame.step->rows, LG_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
			frame.step->hidden_bf16, (const uint16_t *)0, weight, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, const float *weight)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		LM_LAUNCH((LmFusedResidualRmsNormKernel<LG_THREADS,float,G::kNormRoundFirst>), frame.step->rows, LG_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
			hidden, state->partial, weight, hidden, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus AddRows(State *state, const Frame &frame)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		LmAddRowsKernel<LG_THREADS><<<dim3(Blocks(G::kHidden), frame.step->rows), LG_THREADS, 0, frame.stream>>>(hidden, state->partial, hidden,
			frame.step->rows, G::kHidden);
		return SPARK_STATUS_OK;
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows, sequences = step->sequences;
		if ( G::kVerifyRows == 0u || kept == 0 || rows == 0u || rows > G::kVerifyRows || sequences == 0u || sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			const LingLayer *layer = &state->layers[index];
			if ( index % G::kPeriod == G::kPhase )
				continue;
			uint8_t *state_pool = state->kda_state + (uint64_t)layer->recurrent_index * state->slots * state->kda_state_bytes;
			uint16_t *window = state->kda_window + (uint64_t)layer->recurrent_index * state->slots * (state->kda_window_bytes / sizeof(uint16_t));
			LmCausalConvLaunch<G::kConv,(G::kKdaConvRoundedSum ? LM_CONV_SWISH : LM_CONV_SWISH_ONCE),uint16_t>(window, step->recurrent_index, step->sequence_row_begin, kept, ReplayQkv(state, layer), layer->kda_conv,
				state->qkv, S::kKdaChannels, sequences, 1u, step->sequence_row_indices, rows, stream);
			LingSplitRowsKernel<<<dim3(Blocks(S::kKdaChannels), rows), LG_THREADS, 0, stream>>>(state->qkv, state->query, state->key, state->value,
				S::kKdaWidth);
			LM_LAUNCH((LmDeltaRuleColumnKernel<LG_THREADS,G::kKdaDim,G::kKdaDim,float>), dim3(sequences,S::kKdaRankHeads,G::kKdaDim / LM_WARP_LANES),
				LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, (uint32_t)state->kda_state_bytes, step->recurrent_index, step->sequence_row_begin,
				kept, state->query, state->key, state->value, ReplayRetention(state, layer), ReplayWriteGate(state, layer), state->kda_mixed,
				S::kKdaRankHeads, 1u, sequences, 1u, step->sequence_row_indices);
		}
		if ( step->tree != 0u )
		{
			if ( !S::kGqa || sequences != 1u || step->slots == 0 || step->sequence_row_indices == 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			for ( uint32_t index = 0u; index < G::kLayers; ++index )
			{
				const LingLayer *layer = &state->layers[index];
				if ( index % G::kPeriod != G::kPhase )
					continue;
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
				LM_LAUNCH((LmKvShardPathMoveKernel<typename S::Kv,LG_THREADS>), G::kVerifyRows, LG_THREADS, 0, stream, state->kv_views[layer->kv_index],
					step->sequence_of_row, step->sequence_row_indices, kept, step->slots, S::kSlot);
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			}
		}
		return CudaStageLaunched();
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, slots = lane == 0u ? state->slots : 0u, routes = rows * G::kTopK, groups = G::kLocalExperts + 2u;
		const uint64_t degree = state->tp_degree, verify_rows = lane == 0u ? G::kVerifyRows : 0u;
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t);
		const CudaStageSlice plan[] =
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
			{ (void **)&state->replay_qkv, (uint64_t)S::kKdaLayers * verify_rows * S::kKdaChannels * sizeof(uint16_t) },
			{ (void **)&state->replay_retention, (uint64_t)S::kKdaLayers * verify_rows * S::kKdaWidth * sizeof(float) },
			{ (void **)&state->replay_write_gate, (uint64_t)S::kKdaLayers * verify_rows * S::kKdaRankHeads * sizeof(float) },
			{ (void **)&state->kda_snapshots, verify_rows != 0u && S::kGqa ? LmDeltaRuleSnapshotBytes(S::kKdaRankHeads, G::kKdaDim, G::kKdaDim) : 0u },
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
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		state->kda_state_bytes = (uint64_t)S::kKdaRankHeads * G::kKdaDim * G::kKdaDim * sizeof(float);
		state->kda_window_bytes = (uint64_t)S::kKdaChannels * G::kConv * sizeof(uint16_t);
		state->partial_stride = ((uint64_t)state->max_rows * (S::kGqa ? LmGqaShardRecordFloats(G::kGqaHeads, G::kGqaHeadDim, state->tp_degree) :
			S::kMlaRankHeads * S::kRecord) + record_unit - 1u) / record_unit * record_unit;
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration)
	{
		return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 ? 1u : 0u;
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "vocab_rows", &vocab_rows },
			{ "ffn_rows", &state->ffn_rows }, { "experts", &experts }, { "local_experts", &local }, { "expert_first", &state->expert_first },
			{ "top_k", &top_k }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || (G::kDenseLayers != 0u && (state->ffn_rows == 0u || state->ffn_rows % 64u != 0u)) || experts != G::kExperts ||
			local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != (G::kLocalExperts == G::kExperts ? 0u : rank * G::kLocalExperts) )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, experts, local, state->expert_first, top_k, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( S::kGqa && LmGqaShardGeometryValid(G::kGqaHeads, G::kGqaKvHeads, G::kGqaHeadDim, state->tp_degree) == 0u )
		{
			fprintf(stderr, "%s: %u query heads over %u KV heads do not shard over %u ranks\n", G::Tag(), G::kGqaHeads, G::kGqaKvHeads, state->tp_degree);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = S::kGqa && G::kVerifyRows != 0u ? 1u : 0u;
		geometry->head_norm_round_first = G::kNormRoundFirst ? 1u : 0u;
		if ( CudaStage<LingModel>::HeadRanges(state, geometry, false, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_rows * state->tp_degree != G::kVocab || geometry->head_rows * state->tp_degree != G::kVocab ||
			geometry->embed_bytes != (uint64_t)geometry->embed_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)geometry->head_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the stage geometry\n", G::Tag(),
				geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	template<class T>
	static void Tensor(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t bytes)
	{
		tensors->Add(name, kind, bytes, field);
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		LingLayer *layer = &state->layers[index];
		Tensor(tensors, "input_norm", "f32", &layer->input_norm, (uint64_t)G::kHidden * sizeof(float));
		Tensor(tensors, "post_norm", "f32", &layer->post_norm, (uint64_t)G::kHidden * sizeof(float));
		if ( index % G::kPeriod == G::kPhase )
		{
			if constexpr ( S::kGqa )
			{
				Tensor(tensors, "gqa_qkv", "bf16", &layer->gqa_qkv, (uint64_t)S::kGqaRankFused * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "gqa_gate", "bf16", &layer->gqa_gate, (uint64_t)S::kGqaRankWidth * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "gqa_out", "bf16", &layer->gqa_out, (uint64_t)G::kHidden * S::kGqaRankWidth * sizeof(uint16_t));
			}
			else
			{
				Tensor(tensors, "mla_q", "bf16", &layer->mla_q, (uint64_t)S::kMlaRankHeads * S::kQkDim * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "mla_kv_a", "bf16", &layer->mla_kv_a, (uint64_t)S::kSlot * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "mla_kv_norm", "f32", &layer->mla_kv_norm, (uint64_t)G::kLatent * sizeof(float));
				Tensor(tensors, "mla_uk", "bf16", &layer->mla_uk, (uint64_t)S::kMlaRankHeads * G::kLatent * G::kNope * sizeof(uint16_t));
				Tensor(tensors, "mla_uv", "bf16", &layer->mla_uv, (uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent * sizeof(uint16_t));
				Tensor(tensors, "mla_gate", "bf16", &layer->mla_gate, (uint64_t)S::kMlaRankHeads * G::kHidden * sizeof(uint16_t));
				Tensor(tensors, "mla_out", "bf16", &layer->mla_out, (uint64_t)G::kHidden * S::kMlaRankHeads * G::kValueDim * sizeof(uint16_t));
			}
		}
		else
		{
			Tensor(tensors, "kda_qkv", "bf16", &layer->kda_qkv, (uint64_t)S::kKdaChannels * G::kHidden * sizeof(uint16_t));
			if ( G::kKdaRank != 0u )
				Tensor(tensors, "kda_f_a", "bf16", &layer->kda_f_a, (uint64_t)G::kKdaRank * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "kda_f", "bf16", &layer->kda_f, (uint64_t)S::kKdaWidth * (G::kKdaRank != 0u ? G::kKdaRank : G::kHidden) * sizeof(uint16_t));
			if ( G::kKdaRank != 0u )
				Tensor(tensors, "kda_g_a", "bf16", &layer->kda_g_a, (uint64_t)G::kKdaRank * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "kda_g", "bf16", &layer->kda_g, (uint64_t)S::kKdaWidth * (G::kKdaRank != 0u ? G::kKdaRank : G::kHidden) * sizeof(uint16_t));
			Tensor(tensors, "kda_beta", "bf16", &layer->kda_beta, (uint64_t)S::kKdaRankHeads * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "kda_conv", "bf16", &layer->kda_conv, (uint64_t)S::kKdaChannels * G::kConv * sizeof(uint16_t));
			Tensor(tensors, "kda_a_log", "f32", &layer->kda_a_log, (uint64_t)S::kKdaRankHeads * sizeof(float));
			Tensor(tensors, "kda_dt_bias", "f32", &layer->kda_dt_bias, (uint64_t)S::kKdaWidth * sizeof(float));
			Tensor(tensors, "kda_norm", "bf16", &layer->kda_norm, (uint64_t)G::kKdaDim * sizeof(uint16_t));
			Tensor(tensors, "kda_out", "bf16", &layer->kda_out, (uint64_t)G::kHidden * S::kKdaWidth * sizeof(uint16_t));
		}
		if ( index < G::kDenseLayers )
		{
			Tensor(tensors, "ffn_gate_up", "bf16", &layer->ffn_gate_up, (uint64_t)2u * state->ffn_rows * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "ffn_down", "bf16", &layer->ffn_down, (uint64_t)G::kHidden * state->ffn_rows * sizeof(uint16_t));
		}
		else
		{
			Tensor(tensors, "router", "bf16", &layer->router, (uint64_t)G::kExperts * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "expert_bias", "f32", &layer->expert_bias, (uint64_t)G::kExperts * sizeof(float));
			Tensor(tensors, "moe_w1", "bf16", &layer->moe_w1, (uint64_t)G::kLocalExperts * G::kMoeInter * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "moe_w3", "bf16", &layer->moe_w3, (uint64_t)G::kLocalExperts * G::kMoeInter * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "moe_w2", "bf16", &layer->moe_w2, (uint64_t)G::kLocalExperts * G::kMoeInter * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "shared_gate_up", "bf16", &layer->shared_gate_up, (uint64_t)2u * G::kSharedRows * G::kHidden * sizeof(uint16_t));
			Tensor(tensors, "shared_down", "bf16", &layer->shared_down, (uint64_t)G::kSharedRows * G::kHidden * sizeof(uint16_t));
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t attention = 0u, kda = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			if ( index % G::kPeriod == G::kPhase )
				state->layers[index].kv_index = attention++;
			else
				state->layers[index].recurrent_index = kda++;
		}
		return SPARK_STATUS_OK;
	}

	static uint32_t RecurrentParts(const State *state, CudaStageRecurrent *parts)
	{
		parts[0] = { state->kda_state, state->kda_state_bytes, S::kKdaLayers };
		parts[1] = { (uint8_t *)state->kda_window, state->kda_window_bytes, S::kKdaLayers };
		return 2u;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "ffn_rows", state->ffn_rows },
			{ "expert_first", state->expert_first }
		};
		return CudaStageReportOut(fields, report);
	}
};
