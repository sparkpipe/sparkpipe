#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "runtime/launch.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
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
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "runtime/gemm.cuh"

#define GM_THREADS 256u
#define GM_PACK_ALIGNMENT 128u

struct GemmaDefaults
{
	static constexpr bool kEmbedNorm = false;
	static constexpr float kNormEpsilon = 1e-6f;
	static constexpr float kPostEpsilon = 1e-6f;
	static constexpr bool kQkNormWeighted = true;
	static constexpr float kQueryMultiply = 1.0f;
	static constexpr bool kValueNorm = true;
	static constexpr float kAttentionScale = 1.0f;
	static constexpr bool kFullRope = true;
	static constexpr bool kOutputGate = false;
	static constexpr bool kGeluMlp = true;
	static constexpr bool kUntiedHead = false;
	static constexpr bool kLayerScalar = true;
	static constexpr float kLogitScale = 0.0f;
	static constexpr bool kMoe = false;
	static constexpr uint32_t kExperts = 0u;
	static constexpr uint32_t kLocalExperts = 0u;
	static constexpr uint32_t kTopK = 0u;
	static constexpr uint32_t kMoeInter = 0u;
};

template<class G>
struct GemmaShape
{
	static constexpr uint32_t kFullLayers = G::kLayers / G::kPeriod;
	static constexpr uint32_t kSlidingLayers = G::kLayers - kFullLayers;
	static constexpr uint32_t kSlidingQ = G::kHeads * G::kSlidingDim;
	static constexpr uint32_t kSlidingKv = G::kSlidingKvHeads * G::kSlidingDim;
	static constexpr uint32_t kSlidingFused = kSlidingQ + 2u * kSlidingKv;
	static constexpr uint32_t kFullQ = G::kHeads * G::kFullDim;
	static constexpr uint32_t kFullKv = G::kFullKvHeads * G::kFullDim;
	static constexpr uint32_t kFullFused = kFullQ + (G::kKeyIsValue ? 1u : 2u) * kFullKv;
	static constexpr uint32_t kMaxFused = kSlidingFused > kFullFused ? kSlidingFused : kFullFused;
	static constexpr uint32_t kMaxQ = kSlidingQ > kFullQ ? kSlidingQ : kFullQ;
	static constexpr uint32_t kMaxKv = kSlidingKv > kFullKv ? kSlidingKv : kFullKv;
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
} GemmaLayer;

template<class G>
struct GemmaState
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
	uint64_t record_stride;
	GemmaLayer layers[G::kLayers];
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
	LmKvAccessError *kv_error;
	LmKvShardView sliding_views[GemmaShape<G>::kSlidingLayers];
	LmKvShardView full_views[GemmaShape<G>::kFullLayers];
	uint32_t kv_attached;
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

__global__ static void GemmaSigmoidGateKernel(uint16_t *values, const uint16_t *gate_logits, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const float gate = LmBf16ToFloat(LmFloatToBf16(1.0f / (1.0f + expf(-LmBf16ToFloat(gate_logits[index])))));
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
	const float scale = rsqrtf(LmBlockSum<THREADS>(total, reduction) / (float)width + epsilon);
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
		const float cube = gate * gate * gate;
		const float activated = LmBf16ToFloat(LmFloatToBf16(0.5f * gate * (1.0f + tanhf(0.7978845608028654f * (gate + 0.044715f * cube)))));
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
	const float inverse = rsqrtf(LmBlockSum<THREADS>(total, reduction) / (float)width + epsilon);
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
	const float dense_scale = rsqrtf(LmBlockSum<THREADS>(dense_total, reduction) / (float)width + epsilon);
	const float moe_scale = rsqrtf(LmBlockSum<THREADS>(moe_total, reduction) / (float)width + epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float d = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(dense[base + index]) * dense_scale * dense_norm[index]));
		const float m = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(moe[base + index]) * moe_scale * moe_norm[index]));
		const float combined = LmBf16ToFloat(LmFloatToBf16(d + m));
		gemma_combined[index] = combined;
		total += combined * combined;
	}
	const float scale = rsqrtf(LmBlockSum<THREADS>(total, reduction) / (float)width + epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float normed = LmBf16ToFloat(LmFloatToBf16(gemma_combined[index] * scale * post_norm[index]));
		const float added = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(residual[base + index]) + normed));
		residual[base + index] = LmFloatToBf16(added * scalar[0]);
	}
}

template<class G>
struct GemmaModel
{
	using S = GemmaShape<G>;
	using State = GemmaState<G>;

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + GM_THREADS - 1u) / GM_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			int32_t status;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.scale = LmScaleTensorNone();
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			status = LmSkinnyLaunch<LmBf16Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension, output_dimension,
			0u, 0u, multiprocessors, stream);
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

	template<bool FULL>
	static SparkStatus Attention(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
		constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim, KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
		constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ, KVW = FULL ? S::kFullKv : S::kSlidingKv;
		constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint32_t width = QW / degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, HD, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		GemmaGatherRowsKernel<<<dim3(Blocks(FUSED), rows), GM_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, local_rows, degree);
		GemmaSplitKernel<<<dim3(Blocks(QW + 2u * KVW), rows), GM_THREADS, 0, stream>>>(state->attn_full, state->query, state->key, state->value, FUSED,
			QW, KVW, FULL && G::kKeyIsValue ? 1u : 0u);
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(G::kHeads,rows), GM_THREADS, 0, stream,
			state->query, G::kQkNormWeighted ? layer->q_norm : (const float *)0, state->query, rows, G::kHeads, HD, G::kNormEpsilon, G::kQueryMultiply);
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(KVH,rows), GM_THREADS, 0, stream,
			state->key, G::kQkNormWeighted ? layer->k_norm : (const float *)0, state->key, rows, KVH, HD, G::kNormEpsilon, 1.0f);
		if constexpr ( G::kValueNorm )
			LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(KVH,rows), GM_THREADS, 0, stream,
				state->value, (const float *)0, state->value, rows, KVH, HD, G::kNormEpsilon, 1.0f);
		if ( !FULL || G::kFullRope )
		{
			LM_LAUNCH((LmRopePerHeadKernel<GM_THREADS>), dim3(rows,G::kHeads), GM_THREADS, 0, stream,
				state->query, step->positions, G::kHeads, HD, HD, FULL ? G::kFullTheta : G::kSlidingTheta, FULL ? state->full_inv_freq : (const float *)0, 1.0f, 0u);
			LM_LAUNCH((LmRopePerHeadKernel<GM_THREADS>), dim3(rows,KVH), GM_THREADS, 0, stream,
				state->key, step->positions, KVH, HD, HD, FULL ? G::kFullTheta : G::kSlidingTheta, FULL ? state->full_inv_freq : (const float *)0, 1.0f, 0u);
		}
		GemmaPackSlotKernel<<<dim3(Blocks(2u * KVW), rows), GM_THREADS, 0, stream>>>(state->key, state->value, state->slot, KVW);
		LM_LAUNCH((LmKvShardStoreKernel<Kv,GM_THREADS>), rows, GM_THREADS, 0, stream, *view, state->slot, step->sequence_of_row, step->positions,
			rows, 2u * KVW);
		if ( Launched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,HD,HD>(*view, state->query, G::kHeads, step->sequence_of_row, step->context_length,
				step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u, FULL ? 0u : G::kWindow) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<HD>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kOutputGate )
		{
			if ( Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, width, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			GemmaSigmoidGateKernel<<<Blocks((uint64_t)rows * width), GM_THREADS, 0, stream>>>(state->merged, state->gate_logits, (uint64_t)rows * width);
			if ( Launched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Linear(layer->out, state->merged, state->partial, rows, width, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mlp(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		if ( Linear(layer->gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kGeluMlp )
			GemmaGeluTanhMulKernel<GM_THREADS><<<dim3(Blocks(state->ffn_rows), rows), GM_THREADS, 0, stream>>>(state->gate_up, state->intermediate, state->ffn_rows);
		else
			LM_LAUNCH((LmSiluMulKernel<GM_THREADS>), rows, GM_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Moe(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * G::kTopK;
		const uint16_t *hidden = step->hidden_bf16;
		LmLocalExpertRoutes expert_routes;
		GemmaRouterInputKernel<GM_THREADS><<<rows, GM_THREADS, 0, stream>>>(hidden, layer->router_scale, state->moe_normed, G::kHidden, G::kNormEpsilon,
			(float)(1.0 / sqrt((double)G::kHidden)));
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->router_proj, state->moe_normed, state->router_logits, rows, G::kHidden, G::kExperts, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouterSoftmaxKernel<GM_THREADS,G::kExperts><<<rows, GM_THREADS, 0, stream>>>(state->router_logits, state->router_probs);
		if ( LmTopkRouteLaunch<GM_THREADS,G::kTopK,true,LM_TOPK_SCORE_IDENTITY>(rows, state->router_probs, G::kExperts, state->route_expert,
			state->route_weight, (const float *)0, (const uint16_t *)0, 1.0f, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		GemmaExpertScaleKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, layer->expert_scale, routes);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<GM_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, LM_MOE_GEMM_TILE_N, state->tile_prefix_up,
			state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GM_THREADS,float>), rows, GM_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
			hidden, (const uint16_t *)0, layer->pre_ff_norm_2, (uint16_t *)0, state->moe_normed, G::kHidden, G::kHidden, G::kNormEpsilon);
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
			LmLocalExpertsUp<LmBf16Format>(expert_routes, layer->moe_w1, LmScaleTensorNone(), state->moe_normed, state->expert_gate, G::kHidden,
				G::kMoeInter, stream) != LM_LAUNCH_OK ||
			LmLocalExpertsUp<LmBf16Format>(expert_routes, layer->moe_w3, LmScaleTensorNone(), state->moe_normed, state->expert_up, G::kHidden,
				G::kMoeInter, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmGeluTanhPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), GM_THREADS, 0, stream>>>(state->expert_gate, state->expert_up, state->packed_weight,
			state->expert_act, G::kMoeInter);
		if ( Launched() != SPARK_STATUS_OK ||
			LmLocalExpertsDown<LmBf16Format>(expert_routes, layer->moe_w2, LmScaleTensorNone(), state->expert_act, state->expert_out, G::kMoeInter,
				G::kHidden, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), GM_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, (const uint16_t *)0, state->partial + (uint64_t)rows * G::kHidden, G::kTopK, G::kLocalExperts, G::kHidden);
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
		if constexpr ( G::kEmbedNorm )
			LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(1u,rows), GM_THREADS, 0, stream,
				hidden, (const float *)0, hidden, rows, 1u, G::kHidden, G::kNormEpsilon, 1.0f);
		else
			GemmaScaleRowsKernel<<<Blocks((uint64_t)rows * G::kHidden), GM_THREADS, 0, stream>>>(hidden, (uint64_t)rows * G::kHidden, G::kEmbedScale);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const GemmaLayer *layer = &state->layers[index];
			LM_LAUNCH((LmFusedResidualRmsNormKernel<GM_THREADS,float>), rows, GM_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, (const uint16_t *)0, layer->input_norm, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
			status = index % G::kPeriod == G::kPhase ? Attention<true>(state, layer, step, stream) : Attention<false>(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			GemmaPostNormAddKernel<GM_THREADS><<<rows, GM_THREADS, 0, stream>>>(hidden, state->partial, layer->post_attn_norm, (const float *)0,
				G::kHidden, G::kPostEpsilon);
			LM_LAUNCH((LmFusedResidualRmsNormKernel<GM_THREADS,float>), rows, GM_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, (const uint16_t *)0, layer->pre_ff_norm, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, G::kNormEpsilon);
			status = Mlp(state, layer, step, stream);
			if constexpr ( G::kMoe )
				if ( status == SPARK_STATUS_OK )
					status = Moe(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, G::kMoe ? 2u * rows : rows, 0u, state->partial,
					state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			if ( G::kMoe )
				GemmaMoeCombineKernel<GM_THREADS><<<rows, GM_THREADS, G::kHidden * sizeof(float), stream>>>(hidden, state->partial,
					state->partial + (uint64_t)rows * G::kHidden, layer->post_ff_norm_1, layer->post_ff_norm_2, layer->post_ff_norm, layer->layer_scalar,
					G::kHidden, G::kPostEpsilon);
			else
				GemmaPostNormAddKernel<GM_THREADS><<<rows, GM_THREADS, 0, stream>>>(hidden, state->partial, layer->post_ff_norm,
					G::kLayerScalar ? layer->layer_scalar : (const float *)0, G::kHidden, G::kPostEpsilon);
			status = Launched();
			if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		return status;
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
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kMoe ? G::kLocalExperts + 2u : 0u;
		struct { void **target; uint64_t bytes; } plan[] =
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
			{ (void **)&state->gate_logits, G::kOutputGate ? rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) : 0u },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->full_inv_freq, (G::kFullDim / 2u) * sizeof(float) },
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
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u, embed_rows = 0u;
		uint64_t record_floats;
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
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, GM_PACK_ALIGNMENT, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "sliding_window", &window) != SPARK_STATUS_OK )
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
			if ( SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		record_floats = LmGqaShardRecordFloats(G::kHeads, G::kFullDim, state->tp_degree) > LmGqaShardRecordFloats(G::kHeads, G::kSlidingDim, state->tp_degree) ?
			LmGqaShardRecordFloats(G::kHeads, G::kFullDim, state->tp_degree) : LmGqaShardRecordFloats(G::kHeads, G::kSlidingDim, state->tp_degree);
		state->record_stride = (((uint64_t)state->max_rows * record_floats * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
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
		geometry->kv_layer_count = S::kSlidingLayers;
		geometry->kv_layer_page_bytes = S::SlidingKvGeometry::kPageBytes;
		geometry->kv_second_layer_count = S::kFullLayers;
		geometry->kv_second_layer_page_bytes = S::FullKvGeometry::kPageBytes;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->head_norm_f32 = 1u;
		geometry->logit_softcap = G::kLogitSoftcap;
		geometry->logit_scale = G::kLogitScale;
		if ( Entry(state, "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			Entry(state, "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->embed_rows = embed_rows;
		if constexpr ( G::kUntiedHead )
		{
			if ( Entry(state, "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		else
		{
			geometry->head_offset = geometry->embed_offset;
			geometry->head_bytes = geometry->embed_bytes;
			geometry->head_rows = embed_rows;
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		(void)model;
		if ( manifest == 0 || manifest->range_count != 0u )
		{
			fprintf(stderr, "%s: a dense pack's weightd manifest carries no expert ranges\n", G::Tag());
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
		uint32_t sliding = 0u, full = 0u;
		float inv_freq[G::kFullDim / 2u];
		(void)geometry;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			GemmaLayer *layer = &state->layers[index];
			SparkStatus status = Bind(state, lazy_pack, index, "attn_qkv", "bf16", &layer->qkv);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "attn_out", "bf16", &layer->out);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "ffn_gate_up", "bf16", &layer->gate_up);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "ffn_down", "bf16", &layer->down);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "input_norm", "f32", &layer->input_norm);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "post_attn_norm", "f32", &layer->post_attn_norm);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "pre_ff_norm", "f32", &layer->pre_ff_norm);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "post_ff_norm", "f32", &layer->post_ff_norm);
			if ( G::kLayerScalar && status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "layer_scalar", "f32", &layer->layer_scalar);
			if ( G::kQkNormWeighted && status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "attn_q_norm", "f32", &layer->q_norm);
			if ( G::kQkNormWeighted && status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "attn_k_norm", "f32", &layer->k_norm);
			if ( G::kOutputGate && status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "attn_gate", "bf16", &layer->gate);
			if ( G::kMoe )
			{
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "router_proj", "bf16", &layer->router_proj);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "router_scale", "f32", &layer->router_scale);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "expert_scale", "f32", &layer->expert_scale);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_w1", "bf16", &layer->moe_w1);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_w3", "bf16", &layer->moe_w3);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_w2", "bf16", &layer->moe_w2);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "post_ff_norm_1", "f32", &layer->post_ff_norm_1);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "pre_ff_norm_2", "f32", &layer->pre_ff_norm_2);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "post_ff_norm_2", "f32", &layer->post_ff_norm_2);
			}
			if ( status != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
			layer->kv_index = index % G::kPeriod == G::kPhase ? full++ : sliding++;
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || Plan(state, &state->arena_bytes, (uint8_t *)state->arena) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for ( uint32_t pair = 0u; pair < G::kFullDim / 2u; ++pair )
			inv_freq[pair] = pair < G::kFullRopePairs ? (float)(1.0 / pow((double)G::kFullTheta, 2.0 * pair / (double)G::kFullDim)) : 0.0f;
		if ( cudaMemcpy(state->full_inv_freq, inv_freq, sizeof(inv_freq), cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != S::kSlidingLayers || kv->layer_page_bytes != S::SlidingKvGeometry::kPageBytes || kv->second_pool == 0 ||
			kv->second_layer_count != S::kFullLayers || kv->second_layer_page_bytes != S::FullKvGeometry::kPageBytes || kv->sequence_count > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < S::kSlidingLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::SlidingKvGeometry>(&state->sliding_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes,
				kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < S::kFullLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::FullKvGeometry>(&state->full_views[layer], kv->second_pool + (uint64_t)layer * kv->second_layer_stride_bytes,
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
