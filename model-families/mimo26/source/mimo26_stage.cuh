#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_mimo26_family.h"

#define MM_THREADS 256u
#define MM_FP8_GROUP 128u

template<class G>
struct MimoShape : MimoLayout<G>
{
	using SlidingKv = LmKvGeometry<G::kSlidingKvHeads * (G::kHeadDim + G::kValueDim) * 2u, G::kPageSlots, true>;
	using FullKv = LmKvGeometry<G::kFullKvHeads * (G::kHeadDim + G::kValueDim) * 2u, G::kPageSlots, true>;
	static_assert(MimoLayout<G>::kMaxFused / G::kTpDegree <= G::kHidden, "a rank's fused rows fit one collective row");
	static_assert(G::kHidden % MM_FP8_GROUP == 0u && MimoLayout<G>::kDenseRows % MM_FP8_GROUP == 0u, "FP8 inputs are whole 128-wide scale groups");
	static_assert(G::kHidden % 32u == 0u && G::kMoeInter % 32u == 0u, "MXFP4 inputs are whole 32-wide scale groups");
	static_assert(LM_SKINNY_ROWS_WIDE * G::kTopK <= LM_SKINNY_GROUPED_MAX_MEAN_ROWS * G::kLocalExperts,
		"a 16-row verify chunk keeps its routes within the grouped skinny expert kernel");
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
struct MimoState : CudaStagePackState
{
	uint32_t expert_first;
	uint64_t record_stride;
	MimoLayer layers[G::kLayers];
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
	LmKvShardView sliding_views[MimoShape<G>::kSlidingLayers];
	LmKvShardView full_views[G::kFullLayers];
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
struct MimoModel : CudaStage<MimoModel<G>>
{
	using S = MimoShape<G>;
	using State = MimoState<G>;
	using Layer = MimoLayer;
	using Schedule = MimoSchedule<G, MimoModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::SlidingKv, &State::sliding_views, 0u, S::kSlidingLayers>,
		CudaStageKvViews<typename S::FullKv, &State::full_views, 1u, G::kFullLayers>>;

	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kEpsilon;
	static constexpr uint32_t kPackAlignment = MM_PACK_ALIGNMENT;

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
		const uint64_t blocks = (elements + MM_THREADS - 1u) / MM_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	static uint32_t RowExact(const SparkStageRunnerStep *step)
	{
		return step->commit == 0u ? 1u : 0u;
	}

	static int32_t Fp8(const MimoScaled &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return LmDenseRows<LmFp8>(weight.weight, LmScaleTensorBuild(weight.scale, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u,
			MM_FP8_GROUP), source, destination, (float *)0, rows, input_dimension, output_dimension, RowExact(step), step->multiprocessors, stream);
	}

	static int32_t Bf16(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			RowExact(step), step->multiprocessors, stream);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const MimoScaled &weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const LmScaleTensor scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(weight.scale, G::kLocalExperts, output_dimension,
			input_dimension);
		return down != 0u ? LmLocalExpertsDown<LmMxfp4>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream,
			RowExact(step)) : LmLocalExpertsUp<LmMxfp4>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream,
			RowExact(step));
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
		const uint32_t rows = step->rows, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kValueDim, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Fp8(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		MimoGatherRowsKernel<<<dim3(Blocks(FUSED), rows), MM_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, local_rows, degree);
		MimoSplitKernel<<<dim3(Blocks(FUSED), rows), MM_THREADS, 0, stream>>>(state->attn_full, state->query, state->key, state->value, S::kQuery,
			KW, VW, G::kValueScale);
		LM_LAUNCH((LmRopePerHeadKernel<MM_THREADS,LM_ROPE_HALF_SPLIT,LM_ROPE_ROUNDED_TERMS>), dim3(rows,G::kHeads), MM_THREADS, 0, stream,
			state->query, step->positions, G::kHeads, G::kHeadDim, G::kRopeDim, FULL ? G::kFullTheta : G::kSlidingTheta,
			FULL ? state->full_inv_freq : state->sliding_inv_freq, 1.0f, 0u);
		LM_LAUNCH((LmRopePerHeadKernel<MM_THREADS,LM_ROPE_HALF_SPLIT,LM_ROPE_ROUNDED_TERMS>), dim3(rows,KVH), MM_THREADS, 0, stream,
			state->key, step->positions, KVH, G::kHeadDim, G::kRopeDim, FULL ? G::kFullTheta : G::kSlidingTheta,
			FULL ? state->full_inv_freq : state->sliding_inv_freq, 1.0f, 0u);
		MimoPackSlotKernel<<<dim3(Blocks(KW + VW), rows), MM_THREADS, 0, stream>>>(state->key, state->value, state->slot, KW, VW);
		LM_LAUNCH((LmKvShardStoreKernel<Kv,MM_THREADS>), rows, MM_THREADS, 0, stream, *view, state->slot, step->sequence_of_row, step->positions,
			rows, KW + VW);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,G::kHeadDim,G::kValueDim>(*view, state->query, G::kHeads, step->sequence_of_row,
				step->context_length, step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u,
				FULL ? 0u : G::kWindow) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kValueDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream,
				FULL ? (const float *)0 : layer->sinks) != cudaSuccess ||
			Bf16(layer->out, state->merged, state->partial, (float *)0, rows, S::kRankWidth, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Dense(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const MimoLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows;
		if ( Fp8(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * S::kDenseRows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		MimoSwigluKernel<<<dim3(Blocks(S::kDenseRows), rows), MM_THREADS, 0, stream>>>(state->gate_up, state->gate_up + S::kDenseRows,
			2u * S::kDenseRows, state->intermediate, S::kDenseRows);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Fp8(layer->ffn_down, state->intermediate, state->partial, rows, S::kDenseRows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus ExpertRows(State *state, const MimoLayer *layer, const SparkStageRunnerStep *step, uint32_t first, uint32_t rows, cudaStream_t stream)
	{
		const uint32_t routes = rows * G::kTopK;
		const uint16_t *normed = state->normed + (uint64_t)first * G::kHidden;
		LmLocalExpertRoutes expert_routes;
		if ( Bf16(layer->router, normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK ||
			LmTopkGroupLimitedLaunch<MM_THREADS,G::kExperts,1u,1u,G::kTopK>(rows, state->router_logits, layer->router_bias, state->route_expert,
				state->route_weight, 1.0f, step->route_fault, stream) != cudaSuccess )
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
		expert_routes.multiprocessors = step->multiprocessors;
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w1, normed, state->expert_gate, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, normed, state->expert_up, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		MimoSwigluKernel<<<dim3(Blocks(G::kMoeInter), routes), MM_THREADS, 0, stream>>>(state->expert_gate, state->expert_up, G::kMoeInter,
			state->expert_act, G::kMoeInter);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		MimoMoeCombineKernel<<<dim3(Blocks(G::kHidden), rows), MM_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->route_weight, state->partial + (uint64_t)first * G::kHidden, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus Experts(State *state, const Frame &frame, const MimoLayer &entry)
	{
		const SparkStageRunnerStep *step = frame.step;
		const uint32_t chunk = step->commit == 0u ? LM_SKINNY_ROWS_WIDE : step->rows;
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t first = 0u; first < step->rows && status == SPARK_STATUS_OK; first += chunk )
			status = ExpertRows(state, &entry, step, first, step->rows - first < chunk ? step->rows - first : chunk, frame.stream);
		return status;
	}

	static SparkStatus Residual(State *state, uint16_t *hidden, uint32_t rows, cudaStream_t stream)
	{
		MimoResidualKernel<<<Blocks((uint64_t)rows * G::kHidden), MM_THREADS, 0, stream>>>(hidden, state->partial, (uint64_t)rows * G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const uint16_t *weight)
	{
		Norm(frame.step->hidden_bf16, weight, state->normed, frame.step->rows, frame.stream);
		return CudaStageLaunched();
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus AddResidual(State *state, const Frame &frame)
	{
		return Residual(state, frame.step->hidden_bf16, frame.step->rows, frame.stream);
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		(void)lane;
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kLocalExperts + 2u;
		const CudaStageSlice plan[] =
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
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t row_bytes = G::kHidden * sizeof(uint16_t);
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kValueDim, state->tp_degree) * sizeof(float) +
			row_bytes - 1u) / row_bytes) * row_bytes / sizeof(float);
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration)
	{
		return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 ? 1u : 0u;
	}

	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t pattern_full = 0u;
		(void)state;
		(void)configuration;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			pattern_full += G::FullLayer(index) ? 1u : 0u;
		if ( pattern_full != G::kFullLayers || !G::FullLayer(0u) )
		{
			fprintf(stderr, "%s: the layer pattern has %u full layers, the geometry declares %u\n", G::Tag(), pattern_full, G::kFullLayers);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u, full_layers = 0u;
		uint32_t ffn_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "full_layers", &full_layers },
			{ "vocab_rows", &vocab_rows }, { "ffn_rows", &ffn_rows }, { "sliding_window", &window }, { "experts", &experts }, { "local_experts", &local },
			{ "expert_first", &state->expert_first }, { "top_k", &top_k }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
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
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		const uint32_t vocab_rows = G::kVocab / state->tp_degree;
		if ( CudaStage<MimoModel>::HeadRanges(state, geometry, false, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_rows != vocab_rows || geometry->head_rows != vocab_rows ||
			geometry->embed_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the pack's %u vocab rows of %u\n",
				G::Tag(), geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes, vocab_rows, G::kHidden);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static uint64_t Fp8Bytes(uint32_t rows, uint32_t columns)
	{
		return (uint64_t)rows * columns;
	}

	static uint64_t Fp8ScaleBytes(uint32_t rows, uint32_t columns)
	{
		return (uint64_t)rows * ((columns + MM_FP8_GROUP - 1u) / MM_FP8_GROUP) * sizeof(float);
	}

	static void AddFp8(CudaStageTensors *tensors, const char *name, uint32_t rows, uint32_t columns, MimoScaled *field)
	{
		tensors->Add(name, "fp8_e4m3", Fp8Bytes(rows, columns), &field->weight, rows);
		tensors->AddScaled(name, ".scale", "f32", Fp8ScaleBytes(rows, columns), &field->scale, rows);
	}

	static void AddExperts(CudaStageTensors *tensors, const char *name, MimoScaled *field, uint32_t rows, uint32_t columns)
	{
		tensors->Add(name, "mxfp4_e2m1", SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1, (uint64_t)G::kLocalExperts * rows, columns), &field->weight,
			G::kLocalExperts);
		tensors->AddScaled(name, ".scale", "e8m0", SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1, G::kLocalExperts, rows, columns), &field->scale,
			G::kLocalExperts);
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		MimoLayer *layer = &state->layers[index];
		const uint32_t full = G::FullLayer(index) ? 1u : 0u;
		const uint32_t qkv_rows = (full != 0u ? S::kFullFused : S::kSlidingFused) / G::kTpDegree;
		tensors->Add("input_norm", "bf16", (uint64_t)G::kHidden * sizeof(uint16_t), &layer->input_norm);
		tensors->Add("post_attn_norm", "bf16", (uint64_t)G::kHidden * sizeof(uint16_t), &layer->post_attn_norm);
		AddFp8(tensors, "attn_qkv", qkv_rows, G::kHidden, &layer->qkv);
		tensors->Add("attn_out", "bf16", (uint64_t)G::kHidden * S::kRankWidth * sizeof(uint16_t), &layer->out, G::kHidden);
		if ( full == 0u )
			tensors->Add("sinks", "f32", (uint64_t)G::kHeads * sizeof(float), &layer->sinks, G::kHeads);
		else
		{
			char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
			SparkNamedPackEntry entry;
			snprintf(key, sizeof(key), "layers.%u.sinks", index);
			if ( SparkNamedPackEntryLoad(&state->pack, key, &entry) == SPARK_STATUS_OK )
			{
				fprintf(stderr, "%s: layer %u is a full-attention layer in the model but the pack gives it sinks\n", G::Tag(), index);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		if ( index < G::kDenseLayers )
		{
			AddFp8(tensors, "ffn_gate_up", 2u * S::kDenseRows, G::kHidden, &layer->ffn_gate_up);
			AddFp8(tensors, "ffn_down", G::kHidden, S::kDenseRows, &layer->ffn_down);
		}
		else
		{
			tensors->Add("router", "bf16", (uint64_t)G::kExperts * G::kHidden * sizeof(uint16_t), &layer->router, G::kExperts);
			tensors->Add("router_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->router_bias, G::kExperts);
			AddExperts(tensors, "moe_w1", &layer->moe_w1, G::kMoeInter, G::kHidden);
			AddExperts(tensors, "moe_w3", &layer->moe_w3, G::kMoeInter, G::kHidden);
			AddExperts(tensors, "moe_w2", &layer->moe_w2, G::kHidden, G::kMoeInter);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindRefusal(SparkStatus status)
	{
		return status == SPARK_STATUS_TARGET_MISMATCH ? status : SPARK_STATUS_PARSE_ERROR;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t sliding = 0u, full = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			MimoLayer *layer = &state->layers[index];
			layer->full = G::FullLayer(index) ? 1u : 0u;
			layer->kv_index = layer->full != 0u ? full++ : sliding++;
		}
		if ( full != G::kFullLayers || sliding != S::kSlidingLayers )
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[2][G::kRopeDim / 2u];
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

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "expert_first", state->expert_first },
			{ "record_stride", state->record_stride }
		};
		return CudaStageReportOut(fields, report);
	}
};
