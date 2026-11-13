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
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_gemma4_family.h"

#define GM_THREADS 256u

template<class G>
struct GemmaShape : GemmaLayout<G>
{
	using SlidingKvGeometry = LmKvHeads<16u, G::kSlidingKvHeads, G::kSlidingDim, G::kPageSlots>;
	using FullKvGeometry = LmKvHeads<16u, G::kFullKvHeads, G::kFullDim, G::kPageSlots>;
};

typedef struct GemmaLayer
{
	const uint16_t *qkv;
	const uint16_t *out;
	const uint16_t *gate_up;
	const uint16_t *down;
	const float *input_norm;
	const float *post_attn_norm;
	const float *pre_ff_norm;
	const float *post_ff_norm;
	const float *layer_scalar;
	const float *q_norm;
	const float *k_norm;
	const uint16_t *gate;
	const uint16_t *router_proj;
	const float *router_scale;
	const float *expert_scale;
	const uint16_t *moe_w1;
	const uint16_t *moe_w3;
	const uint16_t *moe_w2;
	const float *post_ff_norm_1;
	const float *pre_ff_norm_2;
	const float *post_ff_norm_2;
	uint32_t kv_index;
	const float *expert_bias;
	uint32_t ffn_rows;
	uint32_t gate_up_rows;
	uint32_t window;
	float swiglu_limit;
} GemmaLayer;

template<class G>
struct GemmaState : CudaStagePackState
{
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint64_t record_stride;
	GemmaLayer layers[G::kLayers];
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
	uint16_t *gate_logits;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *full_inv_freq;
	uint16_t *moe_normed;
	uint16_t *router_logits;
	float *router_probs;
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
	LmKvShardView sliding_views[GemmaShape<G>::kSlidingLayers];
	LmKvShardView full_views[GemmaShape<G>::kFullLayers];
};

__global__ static void GemmaScaleRowsKernel(uint16_t *rows, uint64_t elements, float scale)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		rows[index] = LmFloatToBf16(LmBf16ToFloat(rows[index]) * scale);
}

__global__ static void GemmaGatherRowsKernel(const uint16_t *gathered, uint16_t *full, uint32_t rows, uint32_t width, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width * ranks; index += gridDim.x * blockDim.x )
		full[(uint64_t)row * width * ranks + index] = gathered[((uint64_t)(index / width) * rows + row) * width + index % width];
}

__global__ static void GemmaSplitKernel(const uint16_t *full, uint16_t *query, uint16_t *key, uint16_t *value, uint32_t fused, uint32_t q_width,
	uint32_t kv_width, uint32_t key_is_value)
{
	const uint32_t row = blockIdx.y;
	const uint16_t *source = full + (uint64_t)row * fused;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < q_width + 2u * kv_width; index += gridDim.x * blockDim.x )
	{
		if ( index < q_width )
			query[(uint64_t)row * q_width + index] = source[index];
		else if ( index < q_width + kv_width )
			key[(uint64_t)row * kv_width + index - q_width] = source[index];
		else
			value[(uint64_t)row * kv_width + index - q_width - kv_width] = source[key_is_value != 0u ? index - kv_width : index];
	}
}

__global__ static void GemmaPackSlotKernel(const uint16_t *key, const uint16_t *value, uint16_t *slot, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 2u * width; index += gridDim.x * blockDim.x )
		slot[(uint64_t)row * 2u * width + index] = index < width ? key[(uint64_t)row * width + index] : value[(uint64_t)row * width + index - width];
}

__global__ static void GemmaHeadSoftplusGateKernel(uint16_t *values, const uint16_t *gate_logits, uint32_t heads, uint32_t head_dim, uint32_t gate_stride)
{
	const uint32_t row = blockIdx.y, head = blockIdx.x;
	const float gate = SparkMathSoftplusGate(LmBf16ToFloat(gate_logits[(uint64_t)row * gate_stride + head]));
	uint16_t *base = values + ((uint64_t)row * heads + head) * head_dim;
	for ( uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x )
		base[index] = LmFloatToBf16(LmBf16ToFloat(base[index]) * gate);
}

__global__ static void GemmaAddRowsKernel(uint16_t *residual, const uint16_t *partial, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
		residual[index] = LmFloatToBf16(LmBf16ToFloat(residual[index]) + LmBf16ToFloat(partial[index]));
}

__global__ static void GemmaWidenKernel(const uint16_t *source, float *target, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
		target[index] = LmBf16ToFloat(source[index]);
}

__global__ static void GemmaSigmoidGateKernel(uint16_t *values, const uint16_t *gate_logits, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const float gate = LmBf16ToFloat(LmFloatToBf16(SparkMathSigmoid(LmBf16ToFloat(gate_logits[index]))));
		values[index] = LmFloatToBf16(LmBf16ToFloat(values[index]) * gate);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GemmaPostNormAddKernel(uint16_t *residual, const uint16_t *partial, const float *weight, const float *scalar, uint32_t width, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x;
	const uint16_t *source = partial + (uint64_t)row * width;
	uint16_t *target = residual + (uint64_t)row * width;
	float total = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float value = LmBf16ToFloat(source[index]);
		total += value * value;
	}
	const float scale = SparkMathRmsScale(LmBlockSum<THREADS>(total, reduction), (float)width, epsilon);
	const float multiplier = scalar != 0 ? scalar[0] : 1.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float normed = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(source[index]) * scale * weight[index]));
		const float added = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(target[index]) + normed));
		target[index] = LmFloatToBf16(scalar != 0 ? added * multiplier : added);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GemmaGeluTanhMulKernel(const uint16_t *gate_up, uint16_t *output, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < width; index += gridDim.x * THREADS )
	{
		const float gate = LmBf16ToFloat(gate_up[(uint64_t)row * 2u * width + index]);
		const float up = LmBf16ToFloat(gate_up[(uint64_t)row * 2u * width + width + index]);
		const float activated = LmBf16ToFloat(LmFloatToBf16(SparkMathGeluTanh(gate)));
		output[(uint64_t)row * width + index] = LmFloatToBf16(activated * up);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GemmaRouterInputKernel(const uint16_t *residual, const float *scale, uint16_t *output, uint32_t width, float epsilon, float root)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	float total = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float value = LmBf16ToFloat(residual[base + index]);
		total += value * value;
	}
	const float inverse = SparkMathRmsScale(LmBlockSum<THREADS>(total, reduction), (float)width, epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float normed = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(residual[base + index]) * inverse));
		output[base + index] = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(normed * scale[index])) * root);
	}
}

__global__ static void GemmaExpertScaleKernel(const uint32_t *route_expert, float *route_weight, const float *expert_scale, uint32_t routes)
{
	const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
	if ( route < routes )
		route_weight[route] *= expert_scale[route_expert[route]];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GemmaMoeCombineKernel(uint16_t *residual, const uint16_t *dense, const uint16_t *moe, const float *dense_norm, const float *moe_norm,
	const float *post_norm, const float *scalar, uint32_t width, float epsilon)
{
	extern __shared__ float gemma_combined[];
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	float dense_total = 0.0f, moe_total = 0.0f, total = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float d = LmBf16ToFloat(dense[base + index]), m = LmBf16ToFloat(moe[base + index]);
		dense_total += d * d;
		moe_total += m * m;
	}
	const float dense_scale = SparkMathRmsScale(LmBlockSum<THREADS>(dense_total, reduction), (float)width, epsilon);
	const float moe_scale = SparkMathRmsScale(LmBlockSum<THREADS>(moe_total, reduction), (float)width, epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float d = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(dense[base + index]) * dense_scale * dense_norm[index]));
		const float m = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(moe[base + index]) * moe_scale * moe_norm[index]));
		const float combined = LmBf16ToFloat(LmFloatToBf16(d + m));
		gemma_combined[index] = combined;
		total += combined * combined;
	}
	const float scale = SparkMathRmsScale(LmBlockSum<THREADS>(total, reduction), (float)width, epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float normed = LmBf16ToFloat(LmFloatToBf16(gemma_combined[index] * scale * post_norm[index]));
		const float added = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(residual[base + index]) + normed));
		residual[base + index] = LmFloatToBf16(added * scalar[0]);
	}
}

template<class G>
struct GemmaModel : CudaStage<GemmaModel<G>>
{
	using S = GemmaShape<G>;
	using State = GemmaState<G>;
	using Layer = GemmaLayer;
	using Schedule = GemmaSchedule<G, GemmaModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::SlidingKvGeometry, &State::sliding_views, 0u, S::kSlidingLayers>,
		CudaStageKvViews<typename S::FullKvGeometry, &State::full_views, 1u, S::kFullLayers>>;
	using LaneShared = CudaStageFields<&State::full_inv_freq, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kNormEpsilon;
	static constexpr uint32_t kPackAlignment = GEMMA_PACK_ALIGNMENT;

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
		const uint64_t blocks = (elements + GM_THREADS - 1u) / GM_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	static int32_t Project(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			step->commit == 0u ? 1u : 0u, step->multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return Project(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, float *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		return Project(weight, source, (uint16_t *)0, destination, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const uint16_t *weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		return down != 0u ? LmLocalExpertsDown<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, input_dimension, output_dimension, stream, exact) :
			LmLocalExpertsUp<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, input_dimension, output_dimension, stream, exact);
	}

	static void ExpertRoutes(const State *state, const SparkStageRunnerStep *step, uint32_t rows, LmLocalExpertRoutes *routes)
	{
		routes->group_row_offset = state->group_row_offset;
		routes->route_source_token = state->route_source_token;
		routes->tile_prefix_up = state->tile_prefix_up;
		routes->tile_prefix_down = state->tile_prefix_down;
		routes->rows = rows;
		routes->routes = rows * G::kTopK;
		routes->top_k = G::kTopK;
		routes->local_experts = G::kLocalExperts;
		routes->multiprocessors = step->multiprocessors;
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		const GemmaLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
		constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim, KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
		constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ, KVW = FULL ? S::kFullKv : S::kSlidingKv;
		constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
		constexpr uint32_t HEADS = FULL ? G::kFullHeads : G::kSlidingHeads;
		const uint32_t rows = step->rows, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint32_t width = QW / degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(HEADS, HD, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		GemmaGatherRowsKernel<<<dim3(Blocks(FUSED), rows), GM_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, local_rows, degree);
		GemmaSplitKernel<<<dim3(Blocks(QW + 2u * KVW), rows), GM_THREADS, 0, stream>>>(state->attn_full, state->query, state->key, state->value, FUSED,
			QW, KVW, FULL && G::kKeyIsValue ? 1u : 0u);
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float,G::kNormRoundFirst>), dim3(HEADS,rows), GM_THREADS, 0, stream,
			state->query, G::kQkNormWeighted ? layer->q_norm : (const float *)0, state->query, rows, HEADS, HD, G::kNormEpsilon, G::kQueryMultiply);
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float,G::kNormRoundFirst>), dim3(KVH,rows), GM_THREADS, 0, stream,
			state->key, G::kQkNormWeighted ? layer->k_norm : (const float *)0, state->key, rows, KVH, HD, G::kNormEpsilon, 1.0f);
		if constexpr ( G::kValueNorm )
			LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(KVH,rows), GM_THREADS, 0, stream,
				state->value, (const float *)0, state->value, rows, KVH, HD, G::kNormEpsilon, 1.0f);
		if ( !FULL || G::kFullRope )
		{
			LM_LAUNCH((LmRopePerHeadKernel<GM_THREADS,LM_ROPE_HALF_SPLIT,LM_ROPE_ROUNDED_TERMS>), dim3(rows,HEADS), GM_THREADS, 0, stream,
				state->query, step->positions, HEADS, HD, FULL ? S::kFullRopeWidth : HD, FULL ? G::kFullTheta : G::kSlidingTheta,
				FULL ? state->full_inv_freq : (const float *)0, FULL ? G::kFullRopeScale : 1.0f, 0u);
			LM_LAUNCH((LmRopePerHeadKernel<GM_THREADS,LM_ROPE_HALF_SPLIT,LM_ROPE_ROUNDED_TERMS>), dim3(rows,KVH), GM_THREADS, 0, stream,
				state->key, step->positions, KVH, HD, FULL ? S::kFullRopeWidth : HD, FULL ? G::kFullTheta : G::kSlidingTheta,
				FULL ? state->full_inv_freq : (const float *)0, FULL ? G::kFullRopeScale : 1.0f, 0u);
		}
		GemmaPackSlotKernel<<<dim3(Blocks(2u * KVW), rows), GM_THREADS, 0, stream>>>(state->key, state->value, state->slot, KVW);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LM_LAUNCH((LmKvShardStoreKernel<Kv,GM_THREADS>), rows, GM_THREADS, 0, stream, *view, state->slot, step->sequence_of_row, step->positions,
			rows, 2u * KVW);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,HD,HD>(*view, state->query, HEADS, step->sequence_of_row, step->context_length,
				step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u, FULL ? 0u : layer->window) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<HD>(state->received, stride, degree, HEADS, state->tp_rank, state->merged, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kOutputGate )
		{
			if ( Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, width, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			GemmaSigmoidGateKernel<<<Blocks((uint64_t)rows * width), GM_THREADS, 0, stream>>>(state->merged, state->gate_logits, (uint64_t)rows * width);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( G::kHeadGate )
		{
			constexpr uint32_t GATE_ROWS = FULL ? S::kFullGateRows : S::kSlidingGateRows;
			if ( Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, GATE_ROWS, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			GemmaHeadSoftplusGateKernel<<<dim3(width / HD, rows), 128u, 0, stream>>>(state->merged, state->gate_logits, width / HD, HD, GATE_ROWS);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Linear(layer->out, state->merged, state->partial, rows, width, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mlp(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		const GemmaLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows, ffn_rows = layer->ffn_rows;
		if ( Linear(layer->gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * ffn_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kGeluMlp )
			GemmaGeluTanhMulKernel<GM_THREADS><<<dim3(Blocks(ffn_rows), rows), GM_THREADS, 0, stream>>>(state->gate_up, state->intermediate, ffn_rows);
		else if ( layer->swiglu_limit != INFINITY )
			LM_LAUNCH((LmSiluMulGateLimitKernel<GM_THREADS>), rows, GM_THREADS, 0, stream, state->gate_up, state->intermediate, ffn_rows,
				layer->swiglu_limit);
		else
			LM_LAUNCH((LmSiluMulKernel<GM_THREADS>), rows, GM_THREADS, 0, stream, state->gate_up, state->intermediate, ffn_rows, true);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->down, state->intermediate, state->partial, rows, ffn_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus ExpertRows(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		const uint16_t *hidden = step->hidden_bf16;
		uint16_t *output = state->partial + (uint64_t)rows * G::kHidden;
		LmLocalExpertRoutes expert_routes;
		GemmaRouterInputKernel<GM_THREADS><<<rows, GM_THREADS, 0, stream>>>(hidden, layer->router_scale, state->moe_normed, G::kHidden, G::kNormEpsilon,
			(float)(1.0 / sqrt((double)G::kHidden)));
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->router_proj, state->moe_normed, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouterSoftmaxKernel<GM_THREADS,G::kExperts><<<rows, GM_THREADS, 0, stream>>>(state->router_logits, state->router_probs);
		if ( LmTopkRouteLaunch<GM_THREADS,G::kTopK,true,LM_TOPK_SCORE_IDENTITY>(rows, state->router_probs, G::kExperts, state->route_expert,
			state->route_weight, (const float *)0, (const uint16_t *)0, 1.0f, step->route_fault, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		GemmaExpertScaleKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, layer->expert_scale, routes);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<GM_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, LM_MOE_GEMM_TILE_N, state->tile_prefix_up,
			state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GM_THREADS,float,G::kNormRoundFirst>), rows, GM_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
			hidden, (const uint16_t *)0, layer->pre_ff_norm_2, (uint16_t *)0, state->moe_normed, G::kHidden, G::kHidden, G::kNormEpsilon);
		ExpertRoutes(state, step, rows, &expert_routes);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w1, state->moe_normed, state->expert_gate, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, state->moe_normed, state->expert_up, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmGeluTanhPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), GM_THREADS, 0, stream>>>(state->expert_gate, state->expert_up, state->packed_weight,
			state->expert_act, G::kMoeInter);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), GM_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, (const uint16_t *)0, output, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus SigmoidExpertRows(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		const uint16_t *normed = state->normed;
		uint16_t *output = state->partial + (uint64_t)rows * G::kHidden;
		LmLocalExpertRoutes expert_routes;
		if constexpr ( G::kRouterFloat )
		{
			if ( Linear(layer->router_proj, normed, state->router_probs, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( Linear(layer->router_proj, normed, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			GemmaWidenKernel<<<Blocks((uint64_t)rows * G::kExperts), GM_THREADS, 0, stream>>>(state->router_logits, state->router_probs,
				(uint64_t)rows * G::kExperts);
		}
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmTopkGroupLimitedLaunch<GM_THREADS,G::kExperts,1u,1u,G::kTopK>(rows, state->router_probs, layer->expert_bias, state->route_expert,
				state->route_weight, G::kRoutedScale, step->route_fault, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<GM_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, LM_MOE_GEMM_TILE_N, state->tile_prefix_up,
			state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		ExpertRoutes(state, step, rows, &expert_routes);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w1, normed, state->expert_gate, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, normed, state->expert_up, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmSwigluPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), GM_THREADS, 0, stream>>>(state->expert_gate, state->expert_up, state->packed_weight,
			state->expert_act, G::kMoeInter, layer->swiglu_limit);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), GM_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->partial, output, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus Experts(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		return ExpertRows(state, &entry, frame.step, frame.stream);
	}

	static SparkStatus SigmoidExperts(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		return SigmoidExpertRows(state, &entry, frame.step, frame.stream);
	}

	static uint16_t *BranchRows(State *state, const Frame &frame, GemmaBranch branch)
	{
		return state->partial + (uint64_t)branch * frame.step->rows * G::kHidden;
	}

	static SparkStatus EmbedNorm(State *state, const Frame &frame)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		(void)state;
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(1u,frame.step->rows), GM_THREADS, 0, frame.stream,
			hidden, (const float *)0, hidden, frame.step->rows, 1u, G::kHidden, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus EmbedScale(State *state, const Frame &frame, float scale)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		(void)state;
		GemmaScaleRowsKernel<<<Blocks(width), GM_THREADS, 0, frame.stream>>>(frame.step->hidden_bf16, width, scale);
		return SPARK_STATUS_OK;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GM_THREADS,float,G::kNormRoundFirst>), frame.step->rows, GM_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
			frame.step->hidden_bf16, (const uint16_t *)0, weight, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, GemmaBranch branch, const float *weight)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GM_THREADS,float,G::kNormRoundFirst>), frame.step->rows, GM_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
			BranchRows(state, frame, branch), hidden, weight, hidden, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus NormAdd(State *state, const Frame &frame, const float *weight)
	{
		GemmaPostNormAddKernel<GM_THREADS><<<frame.step->rows, GM_THREADS, 0, frame.stream>>>(frame.step->hidden_bf16, state->partial, weight,
			(const float *)0, G::kHidden, G::kPostEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus NormAddScaled(State *state, const Frame &frame, const float *weight, const float *scalar)
	{
		GemmaPostNormAddKernel<GM_THREADS><<<frame.step->rows, GM_THREADS, 0, frame.stream>>>(frame.step->hidden_bf16, state->partial, weight,
			scalar, G::kHidden, G::kPostEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus BranchNormAdd(State *state, const Frame &frame, const float *first_norm, const float *second_norm, const float *norm,
		const float *scalar)
	{
		const uint32_t rows = frame.step->rows;
		GemmaMoeCombineKernel<GM_THREADS><<<rows, GM_THREADS, G::kHidden * sizeof(float), frame.stream>>>(frame.step->hidden_bf16, state->partial,
			state->partial + (uint64_t)rows * G::kHidden, first_norm, second_norm, norm, scalar, G::kHidden, G::kPostEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRows(State *state, const Frame &frame, GemmaBranch branch)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		GemmaAddRowsKernel<<<Blocks(width), GM_THREADS, 0, frame.stream>>>(frame.step->hidden_bf16, BranchRows(state, frame, branch), width);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame, GemmaBranch branch, uint32_t blocks)
	{
		uint16_t *rows = BranchRows(state, frame, branch);
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, blocks * frame.step->rows, 0u, rows, rows);
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kMoe ? G::kLocalExperts + 2u : 0u;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, G::kMoe ? 2u * width : width },
			{ (void **)&state->attn_local, rows * (S::kMaxFused / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kMaxQ * sizeof(uint16_t) },
			{ (void **)&state->key, rows * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->slot, rows * 2u * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->gate_logits, G::kOutputGate ? rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) : G::kHeadGate ? rows * S::kMaxGateRows * sizeof(uint16_t) : 0u },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->full_inv_freq, lane == 0u ? (G::kFullDim / 2u) * sizeof(float) : 0u },
			{ (void **)&state->moe_normed, G::kMoe ? width : 0u },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(uint16_t) },
			{ (void **)&state->router_probs, rows * G::kExperts * sizeof(float) },
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
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t full = LmGqaShardRecordFloats(G::kFullHeads, G::kFullDim, state->tp_degree);
		const uint64_t sliding = LmGqaShardRecordFloats(G::kSlidingHeads, G::kSlidingDim, state->tp_degree);
		const uint64_t row_bytes = G::kHidden * sizeof(uint16_t);
		state->record_stride = (((uint64_t)state->max_rows * (full > sliding ? full : sliding) * sizeof(float) + row_bytes - 1u) / row_bytes) * row_bytes /
			sizeof(float);
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration)
	{
		return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 ? 1u : 0u;
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "vocab_rows", &vocab_rows },
			{ "ffn_rows", &state->ffn_rows }, { "sliding_window", &window }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || window != G::kWindow || state->ffn_rows == 0u || state->ffn_rows % 64u != 0u )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u window %u, the runner is tp %u rank %u\n", G::Tag(), degree, rank,
				hidden, layers, window, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( G::kMoe )
		{
			uint32_t experts = 0u, local = 0u, top_k = 0u;
			const CudaStagePackValue moe[] =
			{
				{ "experts", &experts }, { "local_experts", &local }, { "expert_first", &state->expert_first }, { "top_k", &top_k }
			};
			if ( CudaStagePackValues(&state->pack, moe) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK ||
				state->expert_first != (G::kLocalExperts == G::kExperts ? 0u : rank * G::kLocalExperts) )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->head_norm_round_first = G::kNormRoundFirst ? 1u : 0u;
		geometry->logit_softcap = G::kLogitSoftcap;
		geometry->logit_scale = G::kLogitScale;
		if constexpr ( G::kEmbedAliasCount != 0u )
		{
			geometry->embed_alias_first = G::kEmbedAliasFirst;
			geometry->embed_alias_count = G::kEmbedAliasCount;
			geometry->embed_alias_row = G::kEmbedAliasRow;
		}
		if ( CudaStage<GemmaModel>::HeadRanges(state, geometry, !G::kUntiedHead, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_bytes != (uint64_t)geometry->embed_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)geometry->head_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the stage geometry\n", G::Tag(),
				geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		GemmaLayer *layer = &state->layers[index];
		const bool full = S::Full(index);
		const uint64_t hidden = G::kHidden, norm = hidden * sizeof(float);
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		uint64_t offset, bytes;
		uint32_t gate_rows = 0u;
		snprintf(key, sizeof(key), "layers.%u.ffn_gate_up", index);
		layer->gate_up_rows = 0u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, "bf16", &offset, &bytes, &layer->gate_up_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		tensors->Add("attn_qkv", "bf16", (uint64_t)((full ? S::kFullFused : S::kSlidingFused) / state->tp_degree) * hidden * sizeof(uint16_t), &layer->qkv);
		tensors->Add("attn_out", "bf16", hidden * ((full ? S::kFullQ : S::kSlidingQ) / state->tp_degree) * sizeof(uint16_t), &layer->out);
		tensors->Add("ffn_gate_up", "bf16", (uint64_t)layer->gate_up_rows * hidden * sizeof(uint16_t), &layer->gate_up);
		tensors->Add("ffn_down", "bf16", hidden * (layer->gate_up_rows / 2u) * sizeof(uint16_t), &layer->down);
		tensors->Add("input_norm", "f32", norm, &layer->input_norm);
		if ( !G::kPreNorm )
			tensors->Add("post_attn_norm", "f32", norm, &layer->post_attn_norm);
		tensors->Add("pre_ff_norm", "f32", norm, &layer->pre_ff_norm);
		if ( !G::kPreNorm )
			tensors->Add("post_ff_norm", "f32", norm, &layer->post_ff_norm);
		if ( G::kLayerScalar )
			tensors->Add("layer_scalar", "f32", (uint64_t)sizeof(float), &layer->layer_scalar);
		if ( G::kQkNormWeighted )
		{
			tensors->Add("attn_q_norm", "f32", (uint64_t)(full ? G::kFullDim : G::kSlidingDim) * sizeof(float), &layer->q_norm);
			tensors->Add("attn_k_norm", "f32", (uint64_t)(full ? G::kFullDim : G::kSlidingDim) * sizeof(float), &layer->k_norm);
		}
		if ( G::kOutputGate )
			tensors->Add("attn_gate", "bf16", ((full ? S::kFullQ : S::kSlidingQ) / state->tp_degree) * hidden * sizeof(uint16_t), &layer->gate);
		if ( G::kHeadGate )
		{
			snprintf(key, sizeof(key), "layers.%u.attn_gate", index);
			if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, "bf16", &offset, &bytes, &gate_rows) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( gate_rows != (full ? S::kFullGateRows : S::kSlidingGateRows) )
			{
				fprintf(stderr, "%s: layer %u attn_gate has %u rows, the rank's padded head count is %u\n", G::Tag(), index, gate_rows,
					full ? S::kFullGateRows : S::kSlidingGateRows);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
			tensors->Add("attn_gate", "bf16", (uint64_t)gate_rows * hidden * sizeof(uint16_t), &layer->gate);
		}
		if ( (G::kSigmoidRouter && index >= G::kFirstMoeLayer) || (G::kMoe && !G::kSigmoidRouter) )
		{
			const uint64_t experts = (uint64_t)G::kLocalExperts * G::kMoeInter * hidden * sizeof(uint16_t);
			tensors->Add("router_proj", "bf16", (uint64_t)G::kExperts * hidden * sizeof(uint16_t), &layer->router_proj);
			if ( G::kSigmoidRouter )
				tensors->Add("expert_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->expert_bias);
			else
			{
				tensors->Add("router_scale", "f32", norm, &layer->router_scale);
				tensors->Add("expert_scale", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->expert_scale);
			}
			tensors->Add("moe_w1", "bf16", experts, &layer->moe_w1);
			tensors->Add("moe_w3", "bf16", experts, &layer->moe_w3);
			tensors->Add("moe_w2", "bf16", experts, &layer->moe_w2);
			if ( !G::kSigmoidRouter )
			{
				tensors->Add("post_ff_norm_1", "f32", norm, &layer->post_ff_norm_1);
				tensors->Add("pre_ff_norm_2", "f32", norm, &layer->pre_ff_norm_2);
				tensors->Add("post_ff_norm_2", "f32", norm, &layer->post_ff_norm_2);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerBound(State *state, uint32_t index)
	{
		GemmaLayer *layer = &state->layers[index];
		layer->ffn_rows = layer->gate_up_rows / 2u;
		if ( layer->gate_up_rows % 128u != 0u || layer->ffn_rows == 0u || layer->ffn_rows > state->ffn_rows )
		{
			fprintf(stderr, "%s: layer %u ffn_gate_up has %u rows; a rank holds gate|up in 64-row blocks of at most %u rows each\n", G::Tag(),
				index, layer->gate_up_rows, state->ffn_rows);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t sliding = 0u, full = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			GemmaLayer *layer = &state->layers[index];
			layer->kv_index = S::Full(index) ? full++ : sliding++;
			layer->window = S::Window(index);
			layer->swiglu_limit = S::SwigluLimit(index);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[G::kFullDim / 2u];
		for ( uint32_t pair = 0u; pair < G::kFullDim / 2u; ++pair )
			inv_freq[pair] = pair < G::kFullRopePairs ? (float)(1.0 / pow((double)G::kFullTheta, 2.0 * pair / (double)G::kFullDim)) : 0.0f;
		if constexpr ( G::kYarnFactor > 1.0f )
		{
			const SparkRopePlanDomain yarn = { G::kFullTheta, G::kYarnFactor, G::kYarnOriginal, G::kYarnBetaFast, G::kYarnBetaSlow, G::kFullRopeScale,
				S::kFullRopeWidth };
			memset(inv_freq, 0, sizeof(inv_freq));
			SparkRopePlanBuildYarnInvFrequency(&yarn, inv_freq);
			if ( SparkRopePlanYarnTableIsValid(&yarn, inv_freq) == 0u )
			{
				fprintf(stderr, "%s: the YaRN inverse-frequency table is invalid\n", G::Tag());
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			}
		}
		if ( cudaMemcpy(state->full_inv_freq, inv_freq, sizeof(inv_freq), cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
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
