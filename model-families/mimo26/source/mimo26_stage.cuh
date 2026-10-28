#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "runtime/launch.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_named_pack_bind.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_mimo26_family.h"

#define MM_THREADS 256u
#define MM_PACK_ALIGNMENT 128u
#define MM_FP8_GROUP 128u

template<class G>
struct MimoShape : MimoLayout<G>
{
	using SlidingKv = LmKvGeometry<G::kSlidingKvHeads * (G::kHeadDim + G::kValueDim) * 2u, G::kPageSlots, true>;
	using FullKv = LmKvGeometry<G::kFullKvHeads * (G::kHeadDim + G::kValueDim) * 2u, G::kPageSlots, true>;
	static_assert(MimoLayout<G>::kMaxFused / G::kTpDegree <= G::kHidden, "a rank's fused rows fit one collective row");
	static_assert(G::kHidden % MM_FP8_GROUP == 0u && MimoLayout<G>::kDenseRows % MM_FP8_GROUP == 0u, "FP8 inputs are whole 128-wide scale groups");
	static_assert(G::kHidden % 32u == 0u && G::kMoeInter % 32u == 0u, "MXFP4 inputs are whole 32-wide scale groups");
};

typedef struct MimoScaled
{
	const uint8_t *weight;
	const void *scale;
} MimoScaled;

typedef struct MimoLayer
{
	const uint16_t *input_norm;
	const uint16_t *post_attn_norm;
	MimoScaled qkv;
	const uint16_t *out;
	const float *sinks;
	MimoScaled ffn_gate_up;
	MimoScaled ffn_down;
	const uint16_t *router;
	const float *router_bias;
	MimoScaled moe_w1;
	MimoScaled moe_w3;
	MimoScaled moe_w2;
	uint32_t full;
	uint32_t kv_index;
} MimoLayer;

template<class G>
struct MimoState
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t expert_first;
	uint64_t record_stride;
	MimoLayer layers[G::kLayers];
	void *arena;
	uint64_t arena_bytes;
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *attn_full;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	uint16_t *slot;
	float *send;
	float *received;
	uint16_t *merged;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *sliding_inv_freq;
	float *full_inv_freq;
	float *router_logits;
	uint32_t *route_expert;
	uint32_t *route_global;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	uint16_t *expert_gate;
	uint16_t *expert_up;
	uint16_t *expert_act;
	uint16_t *expert_out;
	LmKvAccessError *kv_error;
	LmKvShardView sliding_views[MimoShape<G>::kSlidingLayers];
	LmKvShardView full_views[G::kFullLayers];
	uint32_t kv_attached;
};

__global__ static void MimoGatherRowsKernel(const uint16_t *gathered, uint16_t *full, uint32_t rows, uint32_t width, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width * ranks; index += gridDim.x * blockDim.x )
		full[(uint64_t)row * width * ranks + index] = gathered[((uint64_t)(index / width) * rows + row) * width + index % width];
}

__global__ static void MimoSplitKernel(const uint16_t *full, uint16_t *query, uint16_t *key, uint16_t *value, uint32_t query_width,
	uint32_t key_width, uint32_t value_width, float value_scale)
{
	const uint32_t row = blockIdx.y, fused = query_width + key_width + value_width;
	const uint16_t *source = full + (uint64_t)row * fused;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < fused; index += gridDim.x * blockDim.x )
	{
		if ( index < query_width )
			query[(uint64_t)row * query_width + index] = source[index];
		else if ( index < query_width + key_width )
			key[(uint64_t)row * key_width + index - query_width] = source[index];
		else
			value[(uint64_t)row * value_width + index - query_width - key_width] = LmFloatToBf16(LmBf16ToFloat(source[index]) * value_scale);
	}
}

__global__ static void MimoPackSlotKernel(const uint16_t *key, const uint16_t *value, uint16_t *slot, uint32_t key_width, uint32_t value_width)
{
	const uint32_t row = blockIdx.y, width = key_width + value_width;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		slot[(uint64_t)row * width + index] = index < key_width ? key[(uint64_t)row * key_width + index] :
			value[(uint64_t)row * value_width + index - key_width];
}

__global__ static void MimoSwigluKernel(const uint16_t *gate, const uint16_t *up, uint32_t source_stride, uint16_t *output, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
	{
		const float g = LmBf16ToFloat(gate[(uint64_t)row * source_stride + index]);
		const float activated = LmBf16ToFloat(LmFloatToBf16(SparkMathSilu(g)));
		output[(uint64_t)row * width + index] = LmFloatToBf16(activated * LmBf16ToFloat(up[(uint64_t)row * source_stride + index]));
	}
}

__global__ static void MimoMoeCombineKernel(const uint16_t *expert_out, const uint32_t *route_packed_row, const uint32_t *route_expert,
	const float *route_weight, uint16_t *output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				total = fmaf(LmBf16ToFloat(expert_out[(uint64_t)route_packed_row[route] * width + index]), route_weight[route], total);
		}
		output[(uint64_t)row * width + index] = LmFloatToBf16(total);
	}
}

__global__ static void MimoResidualKernel(uint16_t *hidden, const uint16_t *partial, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(partial[index]));
}

template<class G>
struct MimoModel
{
	using S = MimoShape<G>;
	using State = MimoState<G>;
	using Layer = MimoLayer;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		cudaStream_t stream;
	};

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + MM_THREADS - 1u) / MM_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	template<class Format>
	static int32_t Project(const void *weight, LmScaleTensor scale, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
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
			args.scale = scale;
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			status = LmSkinnyLaunch<Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<Format>(weight, scale, source, destination, destination_f32, rows, input_dimension, output_dimension, 0u, 0u,
			multiprocessors, stream);
	}

	static int32_t Fp8(const MimoScaled &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return Project<LmFp8>(weight.weight, LmScaleTensorBuild(weight.scale, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u,
			MM_FP8_GROUP), source, destination, (float *)0, rows, input_dimension, output_dimension, multiprocessors, stream);
	}

	static int32_t Bf16(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return Project<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			multiprocessors, stream);
	}

	static LmScaleTensor ExpertScales(const void *scale, uint32_t rows, uint32_t input_dimension)
	{
		return LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(scale, G::kLocalExperts, rows, input_dimension);
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

	static void Norm(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t rows, cudaStream_t stream)
	{
		LmBf16RmsNormKernel<MM_THREADS><<<rows, MM_THREADS, (G::kHidden + MM_THREADS / 32u) * sizeof(float), stream>>>(input, weight, output,
			G::kHidden, G::kHidden, G::kEpsilon);
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const MimoLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		using Kv = typename std::conditional<FULL, typename S::FullKv, typename S::SlidingKv>::type;
		constexpr uint32_t KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
		constexpr uint32_t KW = FULL ? S::kFullKey : S::kSlidingKey, VW = FULL ? S::kFullValue : S::kSlidingValue;
		constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kValueDim, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Fp8(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		MimoGatherRowsKernel<<<dim3(Blocks(FUSED), rows), MM_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, local_rows, degree);
		MimoSplitKernel<<<dim3(Blocks(FUSED), rows), MM_THREADS, 0, stream>>>(state->attn_full, state->query, state->key, state->value, S::kQuery,
			KW, VW, G::kValueScale);
		LM_LAUNCH((LmRopePerHeadKernel<MM_THREADS>), dim3(rows,G::kHeads), MM_THREADS, 0, stream,
			state->query, step->positions, G::kHeads, G::kHeadDim, G::kRopeDim, FULL ? G::kFullTheta : G::kSlidingTheta,
			FULL ? state->full_inv_freq : state->sliding_inv_freq, 1.0f, 0u);
		LM_LAUNCH((LmRopePerHeadKernel<MM_THREADS>), dim3(rows,KVH), MM_THREADS, 0, stream,
			state->key, step->positions, KVH, G::kHeadDim, G::kRopeDim, FULL ? G::kFullTheta : G::kSlidingTheta,
			FULL ? state->full_inv_freq : state->sliding_inv_freq, 1.0f, 0u);
		MimoPackSlotKernel<<<dim3(Blocks(KW + VW), rows), MM_THREADS, 0, stream>>>(state->key, state->value, state->slot, KW, VW);
		LM_LAUNCH((LmKvShardStoreKernel<Kv,MM_THREADS>), rows, MM_THREADS, 0, stream, *view, state->slot, step->sequence_of_row, step->positions,
			rows, KW + VW);
		if ( Launched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,G::kHeadDim,G::kValueDim>(*view, state->query, G::kHeads, step->sequence_of_row,
				step->context_length, step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u,
				FULL ? 0u : G::kWindow) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kValueDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream,
				FULL ? (const float *)0 : layer->sinks) != cudaSuccess ||
			Bf16(layer->out, state->merged, state->partial, (float *)0, rows, S::kRankWidth, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Dense(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const MimoLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		if ( Fp8(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * S::kDenseRows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		MimoSwigluKernel<<<dim3(Blocks(S::kDenseRows), rows), MM_THREADS, 0, stream>>>(state->gate_up, state->gate_up + S::kDenseRows,
			2u * S::kDenseRows, state->intermediate, S::kDenseRows);
		if ( Launched() != SPARK_STATUS_OK ||
			Fp8(layer->ffn_down, state->intermediate, state->partial, rows, S::kDenseRows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Experts(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const MimoLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * G::kTopK;
		LmLocalExpertRoutes expert_routes;
		if ( Bf16(layer->router, state->normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, sms, stream) != LM_LAUNCH_OK ||
			LmTopkGroupLimitedLaunch<MM_THREADS,G::kExperts,1u,1u,G::kTopK>(rows, state->router_logits, layer->router_bias, state->route_expert,
				state->route_weight, 1.0f, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<MM_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, LM_MOE_GEMM_TILE_N, state->tile_prefix_up,
			state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
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
			LmLocalExpertsUp<LmMxfp4>(expert_routes, layer->moe_w1.weight, ExpertScales(layer->moe_w1.scale, G::kMoeInter, G::kHidden),
				state->normed, state->expert_gate, G::kHidden, G::kMoeInter, stream) != LM_LAUNCH_OK ||
			LmLocalExpertsUp<LmMxfp4>(expert_routes, layer->moe_w3.weight, ExpertScales(layer->moe_w3.scale, G::kMoeInter, G::kHidden),
				state->normed, state->expert_up, G::kHidden, G::kMoeInter, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		MimoSwigluKernel<<<dim3(Blocks(G::kMoeInter), routes), MM_THREADS, 0, stream>>>(state->expert_gate, state->expert_up, G::kMoeInter,
			state->expert_act, G::kMoeInter);
		if ( Launched() != SPARK_STATUS_OK ||
			LmLocalExpertsDown<LmMxfp4>(expert_routes, layer->moe_w2.weight, ExpertScales(layer->moe_w2.scale, G::kHidden, G::kMoeInter),
				state->expert_act, state->expert_out, G::kMoeInter, G::kHidden, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		MimoMoeCombineKernel<<<dim3(Blocks(G::kHidden), rows), MM_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->route_weight, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return Launched();
	}

	static SparkStatus Residual(State *state, uint16_t *hidden, uint32_t rows, cudaStream_t stream)
	{
		MimoResidualKernel<<<Blocks((uint64_t)rows * G::kHidden), MM_THREADS, 0, stream>>>(hidden, state->partial, (uint64_t)rows * G::kHidden);
		return Launched();
	}

	static const MimoLayer &LayerOf(State *state, uint32_t index)
	{
		return state->layers[index];
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		frame->step = step;
		frame->stream = (cudaStream_t)stream;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const uint16_t *weight)
	{
		Norm(frame.step->hidden_bf16, weight, state->normed, frame.step->rows, frame.stream);
		return Launched();
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return Round(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus AddResidual(State *state, const Frame &frame)
	{
		return Residual(state, frame.step->hidden_bf16, frame.step->rows, frame.stream);
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		(void)frame;
		if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, index);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream)
	{
		return MimoSchedule<G, MimoModel>::Step((State *)model, step, stream);
	}

	static SparkStatus Plan(State *state, uint64_t *bytes, uint8_t *base)
	{
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kLocalExperts + 2u;
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->attn_local, rows * (S::kMaxFused / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kQuery * sizeof(uint16_t) },
			{ (void **)&state->key, rows * S::kMaxKey * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kMaxValue * sizeof(uint16_t) },
			{ (void **)&state->slot, rows * (S::kMaxKey + S::kMaxValue) * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * S::kRankWidth * sizeof(uint16_t) },
			{ (void **)&state->gate_up, rows * 2u * S::kDenseRows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * S::kDenseRows * sizeof(uint16_t) },
			{ (void **)&state->sliding_inv_freq, (G::kRopeDim / 2u) * sizeof(float) },
			{ (void **)&state->full_inv_freq, (G::kRopeDim / 2u) * sizeof(float) },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->expert_gate, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
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
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u, embed_rows = 0u, full_layers = 0u;
		uint32_t ffn_rows = 0u, experts = 0u, local = 0u, top_k = 0u, pattern_full = 0u;
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree || configuration->max_input_row_count == 0u ||
			configuration->state_budget_bytes == 0u || configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			pattern_full += G::FullLayer(index) ? 1u : 0u;
		if ( pattern_full != G::kFullLayers || !G::FullLayer(0u) )
		{
			fprintf(stderr, "%s: the layer pattern has %u full layers, the geometry declares %u\n", G::Tag(), pattern_full, G::kFullLayers);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
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
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, MM_PACK_ALIGNMENT, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "full_layers", &full_layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "ffn_rows", &ffn_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "sliding_window", &window) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			full_layers != G::kFullLayers || vocab_rows * degree != G::kVocab || ffn_rows != S::kDenseRows || window != G::kWindow ||
			experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u/%u full window %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, full_layers, window, experts, local, state->expert_first, top_k, configuration->tp_degree,
				configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kValueDim, state->tp_degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
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
		geometry->rms_epsilon = G::kEpsilon;
		geometry->pack_bytes = state->pack.file_bytes;
		geometry->kv_layer_count = S::kSlidingLayers;
		geometry->kv_layer_page_bytes = S::SlidingKv::kPageBytes;
		geometry->kv_second_layer_count = G::kFullLayers;
		geometry->kv_second_layer_page_bytes = S::FullKv::kPageBytes;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->head_norm_f32 = 1u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			SparkNamedPackTensorRange(&state->pack, G::Tag(), "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
			SparkNamedPackTensorRange(&state->pack, G::Tag(), "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->embed_rows = embed_rows;
		if ( embed_rows != vocab_rows || geometry->head_rows != vocab_rows || geometry->embed_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the pack's %u vocab rows of %u\n",
				G::Tag(), embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes, vocab_rows, G::kHidden);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
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

	static SparkStatus Slice(State *state, SparkWeightdLazyPack *lazy_pack, const char *key, const char *kind, uint32_t rows, uint64_t expected_bytes,
		const void **field)
	{
		uint64_t offset, bytes;
		uint32_t found = 0u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, kind, &offset, &bytes, &found) != SPARK_STATUS_OK )
			return SPARK_STATUS_PARSE_ERROR;
		if ( rows != 0u && found != rows )
		{
			fprintf(stderr, "%s: pack tensor %s has %u rows, the model needs %u\n", G::Tag(), key, found, rows);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SparkNamedPackBindSized(&state->pack, lazy_pack, G::Tag(), key, kind, expected_bytes, field);
	}

	template<class T>
	static SparkStatus Bind(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, uint32_t rows, T **field,
		uint64_t expected_bytes)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(key, sizeof(key), "layers.%u.%s", index, name);
		return Slice(state, lazy_pack, key, kind, rows, expected_bytes, (const void **)field);
	}

	static SparkStatus BindScaled(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind,
		const char *scale_kind, uint32_t rows, MimoScaled *field, uint64_t weight_bytes, uint64_t scale_bytes)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		SparkStatus status = Bind(state, lazy_pack, index, name, kind, rows, &field->weight, weight_bytes);
		snprintf(key, sizeof(key), "%s.scale", name);
		return status == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, key, scale_kind, rows, &field->scale, scale_bytes) : status;
	}

	static uint64_t Fp8Bytes(uint32_t rows, uint32_t columns)
	{
		return (uint64_t)rows * columns;
	}

	static uint64_t Fp8ScaleBytes(uint32_t rows, uint32_t columns)
	{
		return (uint64_t)rows * ((columns + MM_FP8_GROUP - 1u) / MM_FP8_GROUP) * sizeof(float);
	}

	static SparkStatus BindExperts(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, MimoScaled *field, uint32_t rows,
		uint32_t columns)
	{
		return BindScaled(state, lazy_pack, index, name, "mxfp4_e2m1", "e8m0", G::kLocalExperts, field,
			SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1, (uint64_t)G::kLocalExperts * rows, columns),
			SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1, G::kLocalExperts, rows, columns));
	}

	static SparkStatus BindModel(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		uint32_t sliding = 0u, full = 0u;
		float inv_freq[2][G::kRopeDim / 2u];
		(void)geometry;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			MimoLayer *layer = &state->layers[index];
			char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
			SparkNamedPackEntry entry;
			SparkStatus status;
			layer->full = G::FullLayer(index) ? 1u : 0u;
			status = Bind(state, lazy_pack, index, "input_norm", "bf16", 0u, &layer->input_norm, (uint64_t)G::kHidden * sizeof(uint16_t));
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "post_attn_norm", "bf16", 0u, &layer->post_attn_norm, (uint64_t)G::kHidden * sizeof(uint16_t));
			const uint32_t qkv_rows = (layer->full != 0u ? S::kFullFused : S::kSlidingFused) / G::kTpDegree;
			if ( status == SPARK_STATUS_OK )
				status = BindScaled(state, lazy_pack, index, "attn_qkv", "fp8_e4m3", "f32", qkv_rows, &layer->qkv, Fp8Bytes(qkv_rows, G::kHidden),
					Fp8ScaleBytes(qkv_rows, G::kHidden));
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "attn_out", "bf16", G::kHidden, &layer->out, (uint64_t)G::kHidden * S::kRankWidth * sizeof(uint16_t));
			snprintf(key, sizeof(key), "layers.%u.sinks", index);
			if ( status == SPARK_STATUS_OK && layer->full == 0u )
				status = Bind(state, lazy_pack, index, "sinks", "f32", G::kHeads, &layer->sinks, (uint64_t)G::kHeads * sizeof(float));
			else if ( status == SPARK_STATUS_OK && SparkNamedPackEntryLoad(&state->pack, key, &entry) == SPARK_STATUS_OK )
			{
				fprintf(stderr, "%s: layer %u is a full-attention layer in the model but the pack gives it sinks\n", G::Tag(), index);
				status = SPARK_STATUS_TARGET_MISMATCH;
			}
			if ( index < G::kDenseLayers )
			{
				if ( status == SPARK_STATUS_OK )
					status = BindScaled(state, lazy_pack, index, "ffn_gate_up", "fp8_e4m3", "f32", 2u * S::kDenseRows, &layer->ffn_gate_up,
						Fp8Bytes(2u * S::kDenseRows, G::kHidden), Fp8ScaleBytes(2u * S::kDenseRows, G::kHidden));
				if ( status == SPARK_STATUS_OK )
					status = BindScaled(state, lazy_pack, index, "ffn_down", "fp8_e4m3", "f32", G::kHidden, &layer->ffn_down, Fp8Bytes(G::kHidden, S::kDenseRows),
						Fp8ScaleBytes(G::kHidden, S::kDenseRows));
			}
			else
			{
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "router", "bf16", G::kExperts, &layer->router, (uint64_t)G::kExperts * G::kHidden * sizeof(uint16_t));
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "router_bias", "f32", G::kExperts, &layer->router_bias, (uint64_t)G::kExperts * sizeof(float));
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w1", &layer->moe_w1, G::kMoeInter, G::kHidden);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w3", &layer->moe_w3, G::kMoeInter, G::kHidden);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w2", &layer->moe_w2, G::kHidden, G::kMoeInter);
			}
			if ( status != SPARK_STATUS_OK )
				return status == SPARK_STATUS_TARGET_MISMATCH ? status : SPARK_STATUS_PARSE_ERROR;
			layer->kv_index = layer->full != 0u ? full++ : sliding++;
		}
		if ( full != G::kFullLayers || sliding != S::kSlidingLayers )
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || Plan(state, &state->arena_bytes, (uint8_t *)state->arena) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for ( uint32_t pair = 0u; pair < G::kRopeDim / 2u; ++pair )
		{
			const float exponent = (float)(2u * pair) / (float)G::kRopeDim;
			inv_freq[0][pair] = 1.0f / (float)pow((double)G::kSlidingTheta, (double)exponent);
			inv_freq[1][pair] = 1.0f / (float)pow((double)G::kFullTheta, (double)exponent);
		}
		if ( cudaMemcpy(state->sliding_inv_freq, inv_freq[0], sizeof(inv_freq[0]), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(state->full_inv_freq, inv_freq[1], sizeof(inv_freq[1]), cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != S::kSlidingLayers || kv->layer_page_bytes != S::SlidingKv::kPageBytes || kv->second_pool == 0 ||
			kv->second_layer_count != G::kFullLayers || kv->second_layer_page_bytes != S::FullKv::kPageBytes || kv->sequence_count > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < S::kSlidingLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::SlidingKv>(&state->sliding_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes,
				kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < G::kFullLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::FullKv>(&state->full_views[layer], kv->second_pool + (uint64_t)layer * kv->second_layer_stride_bytes,
				kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		(void)model; (void)to_buffer; (void)slot; (void)buffer; (void)bytes; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		(void)model; (void)slot; (void)stream;
		return SPARK_STATUS_OK;
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
		fprintf(stderr, "%s rank=%u arena_bytes=%llu slots=%u rows=%u expert_first=%u record_stride=%llu\n", G::ReportTag(), rank,
			(unsigned long long)state->arena_bytes, state->slots, state->max_rows, state->expert_first, (unsigned long long)state->record_stride);
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
