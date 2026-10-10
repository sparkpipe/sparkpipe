#pragma once

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
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"

#define QH_KEY_DIM 128u
#define QH_VALUE_DIM 128u
#define QH_CONV 4u
#define QH_EPSILON 1e-6f
#define QH_THREADS 256u
#define QH_SCALE_K 128u
#define QH_PACK_ALIGNMENT 128u
#define QH_GEMM_TILE_N 128u
#define QH_GEMM_STAGES 2u
#define QH_GEMM_WARPS 8u

template<class G>
struct QwenHybridShape
{
	static constexpr uint32_t kGdnLayers = G::kLayers - G::kLayers / G::kPeriod;
	static constexpr uint32_t kAttnLayers = G::kLayers / G::kPeriod;
	static constexpr uint32_t kQWidth = G::kHeads * G::kHeadDim;
	static constexpr uint32_t kKvWidth = G::kKvHeads * G::kHeadDim;
	static constexpr uint32_t kQkvWidth = 2u * kQWidth + 2u * kKvWidth;
	using Kv = LmKvHeads<16u, G::kKvHeads, G::kHeadDim, G::kPageSlots>;
};

typedef struct QwenHybridLayer
{
	const uint8_t *mix_weight;
	const float *mix_scale;
	const uint8_t *out_weight;
	const float *out_scale;
	const uint16_t *gdn_ba;
	const uint16_t *gdn_conv;
	const uint16_t *gdn_norm;
	const float *gdn_a_log;
	const float *gdn_dt_bias;
	const float *q_norm;
	const float *k_norm;
	const uint8_t *ffn_gate_up;
	const float *ffn_gate_up_scale;
	const uint8_t *ffn_down;
	const float *ffn_down_scale;
	const uint16_t *moe_router;
	const uint8_t *moe_w1;
	const uint8_t *moe_w1_scale;
	const uint8_t *moe_w3;
	const uint8_t *moe_w3_scale;
	const uint8_t *moe_w2;
	const uint8_t *moe_w2_scale;
	const uint16_t *moe_shared_gate_up;
	const uint16_t *moe_shared_down;
	const uint16_t *moe_shared_gate;
	const float *input_norm;
	const float *post_norm;
	uint32_t recurrent_index;
	uint32_t kv_index;
} QwenHybridLayer;

template<class G>
struct QwenHybridState
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t linear_bytes;
	const char *linear_kind;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint32_t gdn_heads;
	uint32_t gdn_qk;
	uint32_t gdn_v;
	uint32_t gdn_channels;
	uint32_t attn_rows;
	uint32_t attn_width;
	uint64_t gdn_state_bytes;
	uint64_t gdn_window_bytes;
	uint64_t record_stride;
	QwenHybridLayer layers[G::kLayers];
	FILE *dump;
	uint16_t *dump_rows;
	uint32_t *dump_positions;
	void *arena;
	uint64_t arena_bytes;
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *gate_up;
	uint16_t *intermediate;
	uint16_t *qkv;
	uint16_t *z;
	uint16_t *ba;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	uint16_t *query_heads;
	uint16_t *key_heads;
	float *retention;
	float *write_gate;
	uint16_t *mixed;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *attn_full;
	uint16_t *attn_query;
	uint16_t *attn_gate;
	uint16_t *attn_key;
	uint16_t *attn_value;
	uint16_t *attn_slot;
	float *send;
	float *received;
	uint16_t *merged;
	uint16_t *merged_gate;
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
	uint16_t *shared_gate_up;
	uint16_t *shared_act;
	uint16_t *shared_out;
	uint16_t *shared_logit;
	uint8_t *gdn_state;
	uint16_t *gdn_window;
	LmKvAccessError *kv_error;
	LmKvShardView kv_views[QwenHybridShape<G>::kAttnLayers];
	uint32_t kv_attached;
};

__global__ static void QwenHybridAddRowsKernel(uint16_t *hidden, const uint16_t *addend, uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
}

__global__ static void QwenHybridGatherRowsKernel(const uint16_t *gathered, uint16_t *full, uint32_t rows, uint32_t width, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width * ranks; index += gridDim.x * blockDim.x )
	{
		const uint32_t rank = index / width, element = index % width;
		full[(uint64_t)row * width * ranks + index] = gathered[((uint64_t)rank * rows + row) * width + element];
	}
}

template<class G>
__global__ static void QwenHybridSplitAttentionKernel(const uint16_t *full, uint16_t *query, uint16_t *gate, uint16_t *key, uint16_t *value)
{
	using S = QwenHybridShape<G>;
	const uint32_t row = blockIdx.y;
	const uint16_t *source = full + (uint64_t)row * S::kQkvWidth;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < S::kQkvWidth; index += gridDim.x * blockDim.x )
	{
		uint32_t fused = index;
		if constexpr ( G::kRankHeads )
		{
			constexpr uint32_t per_rank = S::kQkvWidth / G::kTpDegree, q_rank = 2u * S::kQWidth / G::kTpDegree, kv_rank = S::kKvWidth / G::kTpDegree;
			const uint32_t rank = index / per_rank, element = index % per_rank;
			fused = element < q_rank ? rank * q_rank + element :
				element < q_rank + kv_rank ? 2u * S::kQWidth + rank * kv_rank + (element - q_rank) :
				2u * S::kQWidth + S::kKvWidth + rank * kv_rank + (element - q_rank - kv_rank);
		}
		if ( fused < 2u * S::kQWidth )
		{
			const uint32_t head = fused / (2u * G::kHeadDim), within = fused % (2u * G::kHeadDim);
			uint16_t *target = within < G::kHeadDim ? query : gate;
			target[(uint64_t)row * S::kQWidth + head * G::kHeadDim + within % G::kHeadDim] = source[index];
		}
		else if ( fused < 2u * S::kQWidth + S::kKvWidth )
			key[(uint64_t)row * S::kKvWidth + fused - 2u * S::kQWidth] = source[index];
		else
			value[(uint64_t)row * S::kKvWidth + fused - 2u * S::kQWidth - S::kKvWidth] = source[index];
	}
}

template<class G>
__global__ static void QwenHybridPackSlotKernel(const uint16_t *key, const uint16_t *value, uint16_t *slot)
{
	constexpr uint32_t width = QwenHybridShape<G>::kKvWidth;
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 2u * width; index += gridDim.x * blockDim.x )
		slot[(uint64_t)row * 2u * width + index] = index < width ? key[(uint64_t)row * width + index] : value[(uint64_t)row * width + index - width];
}

__global__ static void QwenHybridSliceRowsKernel(const uint16_t *source, uint16_t *target, uint32_t source_width, uint32_t offset, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		target[(uint64_t)row * width + index] = source[(uint64_t)row * source_width + offset + index];
}

template<uint32_t WIDTH>
__global__ static void QwenHybridSoftmaxRowsKernel(const uint16_t *logits, float *probs)
{
	__shared__ float reduce[QH_THREADS / 32u];
	const uint32_t row = blockIdx.x;
	float top = -INFINITY, sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		top = fmaxf(top, LmBf16ToFloat(logits[(uint64_t)row * WIDTH + index]));
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
	if ( threadIdx.x % 32u == 0u )
		reduce[threadIdx.x / 32u] = top;
	__syncthreads();
	top = reduce[0];
	for ( uint32_t warp = 1u; warp < blockDim.x / 32u; ++warp )
		top = fmaxf(top, reduce[warp]);
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		sum += expf(LmBf16ToFloat(logits[(uint64_t)row * WIDTH + index]) - top);
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu, sum, offset);
	if ( threadIdx.x % 32u == 0u )
		reduce[threadIdx.x / 32u] = sum;
	__syncthreads();
	sum = 0.0f;
	for ( uint32_t warp = 0u; warp < blockDim.x / 32u; ++warp )
		sum += reduce[warp];
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		probs[(uint64_t)row * WIDTH + index] = expf(LmBf16ToFloat(logits[(uint64_t)row * WIDTH + index]) - top) / sum;
}

__global__ static void QwenHybridRowDotKernel(const uint16_t *rows_bf16, const uint16_t *weight, uint16_t *output, uint32_t width)
{
	__shared__ float reduce[QH_THREADS / 32u];
	const uint32_t row = blockIdx.x;
	float sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += blockDim.x )
		sum += LmBf16ToFloat(rows_bf16[(uint64_t)row * width + index]) * LmBf16ToFloat(weight[index]);
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu, sum, offset);
	if ( threadIdx.x % 32u == 0u )
		reduce[threadIdx.x / 32u] = sum;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float total = 0.0f;
		for ( uint32_t warp = 0u; warp < blockDim.x / 32u; ++warp )
			total += reduce[warp];
		output[row] = LmFloatToBf16(total);
	}
}

__global__ static void QwenHybridSigmoidGateRowsKernel(uint16_t *rows_bf16, const uint16_t *logit, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	const float gate = 1.0f / (1.0f + expf(-LmBf16ToFloat(logit[row])));
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		rows_bf16[(uint64_t)row * width + index] = LmFloatToBf16(LmBf16ToFloat(rows_bf16[(uint64_t)row * width + index]) * gate);
}

template<class G>
struct QwenHybridModel
{
	using S = QwenHybridShape<G>;
	using State = QwenHybridState<G>;

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + QH_THREADS - 1u) / QH_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	template<class Format>
	static int32_t Project(const void *weight, const float *scale_rows, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		LmScaleTensor scale = scale_rows != 0 ?
			LmScaleTensorBuild(scale_rows, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u, QH_SCALE_K) : LmScaleTensorNone();
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			int32_t status;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.scale = scale;
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			status = LmSkinnyLaunch<Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension, output_dimension,
			0u, 0u, multiprocessors, stream);
	}

	static int32_t Linear(const State *state, const uint8_t *weight, const float *scale_rows, const uint16_t *source,
		uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return state->linear_bytes == 1u ?
			Project<LmFp8>(weight, scale_rows, source, destination, rows, input_dimension, output_dimension, multiprocessors, stream) :
			Project<LmBf16Format>(weight, 0, source, destination, rows, input_dimension, output_dimension, multiprocessors, stream);
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

	static SparkStatus Gdn(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sequences = step->sequences, heads = state->gdn_heads;
		const uint32_t sms = step->multiprocessors;
		uint8_t *state_pool = state->gdn_state + (uint64_t)layer->recurrent_index * state->slots * state->gdn_state_bytes;
		uint16_t *window = state->gdn_window + (uint64_t)layer->recurrent_index * state->slots * (state->gdn_window_bytes / sizeof(uint16_t));
		LmQkvLayout layout;
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->qkv, rows, G::kHidden,
				state->gdn_channels, sms, stream) != LM_LAUNCH_OK ||
			Linear(state, layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * state->linear_bytes,
				layer->mix_scale == 0 ? (const float *)0 : layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
				state->normed, state->z, rows, G::kHidden, state->gdn_v, sms, stream) != LM_LAUNCH_OK ||
			Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmCausalConvKernel<QH_THREADS,QH_CONV,LM_CONV_SWISH,uint16_t>), dim3(sequences,(state->gdn_channels + QH_THREADS - 1u) / QH_THREADS),
			QH_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0, state->qkv, layer->gdn_conv,
			state->qkv, state->gdn_channels, sequences, step->commit, step->sequence_row_indices);
		layout.query_dimension = state->gdn_qk;
		layout.key_dimension = state->gdn_qk;
		layout.value_dimension = state->gdn_v;
		layout.rope_dimension = 0u;
		layout.head_dimension = QH_KEY_DIM;
		LM_LAUNCH((LmSplitQkvKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->qkv, layout, state->query, state->key, state->value, rows, 1.0f);
		LM_LAUNCH((LmExpandHeadsKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->query, state->query_heads,
			state->gdn_qk / QH_KEY_DIM, QH_KEY_DIM, heads / (state->gdn_qk / QH_KEY_DIM), rows);
		LM_LAUNCH((LmExpandHeadsKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->key, state->key_heads,
			state->gdn_qk / QH_KEY_DIM, QH_KEY_DIM, heads / (state->gdn_qk / QH_KEY_DIM), rows);
		LM_LAUNCH((LmGdnGateKernel<QH_THREADS,QH_KEY_DIM>), dim3(rows,heads), QH_THREADS, 0, stream,
			state->ba + heads, state->ba, layer->gdn_a_log, layer->gdn_dt_bias, state->retention, state->write_gate, heads, rows, 2u * heads);
		LM_LAUNCH((LmDeltaRuleColumnKernel<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>), dim3(sequences,heads,QH_VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index, step->sequence_row_begin,
			(const uint32_t *)0, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
			heads, 1u, sequences, step->commit, step->sequence_row_indices);
		LM_LAUNCH((LmHeadRmsNormSiluGateKernel<QH_THREADS>), dim3(heads,rows), QH_THREADS, 0, stream,
			state->mixed, state->z, layer->gdn_norm, state->mixed, rows, heads, QH_VALUE_DIM, QH_EPSILON);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->mixed, state->partial, rows, state->gdn_v, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Attention(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->attn_local, rows, G::kHidden,
			state->attn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, state->attn_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		QwenHybridGatherRowsKernel<<<dim3(Blocks(S::kQkvWidth),rows), QH_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, state->attn_rows, degree);
		QwenHybridSplitAttentionKernel<G><<<dim3(Blocks(S::kQkvWidth),rows), QH_THREADS, 0, stream>>>(state->attn_full, state->attn_query,
			state->attn_gate, state->attn_key, state->attn_value);
		LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kHeads,rows), QH_THREADS, 0, stream,
			state->attn_query, layer->q_norm, state->attn_query, rows, G::kHeads, G::kHeadDim, QH_EPSILON, 1.0f);
		LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kKvHeads,rows), QH_THREADS, 0, stream,
			state->attn_key, layer->k_norm, state->attn_key, rows, G::kKvHeads, G::kHeadDim, QH_EPSILON, 1.0f);
		LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kHeads), QH_THREADS, 0, stream,
			state->attn_query, step->positions, G::kHeads, G::kHeadDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
		LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kKvHeads), QH_THREADS, 0, stream,
			state->attn_key, step->positions, G::kKvHeads, G::kHeadDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
		QwenHybridPackSlotKernel<G><<<dim3(Blocks(2u * S::kKvWidth),rows), QH_THREADS, 0, stream>>>(state->attn_key, state->attn_value, state->attn_slot);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,QH_THREADS>), rows, QH_THREADS, 0, stream,
			*view, state->attn_slot, step->sequence_of_row, step->positions, rows, 2u * S::kKvWidth);
		if ( Launched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kKvHeads,G::kHeadDim,G::kHeadDim>(*view, state->attn_query, G::kHeads,
				step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kHeadDim), state->send, stride, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kHeadDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridSliceRowsKernel<<<dim3(Blocks(state->attn_width),rows), QH_THREADS, 0, stream>>>(state->attn_gate, state->merged_gate,
			S::kQWidth, state->tp_rank * state->attn_width, state->attn_width);
		LM_LAUNCH((LmOutputGateKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->merged, state->merged_gate, state->attn_width);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static int32_t ExpertsUp(State *state, const uint8_t *weight, const uint8_t *scale_data, uint16_t *output, uint32_t rows, uint32_t routes,
		uint32_t sms, cudaStream_t stream)
	{
		const LmScaleTensor scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_NVFP4_E2M1>(scale_data, G::kLocalExperts, G::kMoeInter, G::kHidden);
		int32_t launch = LmSkinnyGroupedExperts<LmNvfp4>(weight, scale, state->normed, output, state->group_row_offset, state->route_source_token,
			G::kLocalExperts, routes, 0u, G::kHidden, G::kMoeInter, stream);
		if ( launch == LM_LAUNCH_ERR_SHAPE )
		{
			LmGemmArguments gemm;
			memset(&gemm, 0, sizeof(gemm));
			gemm.scale_a = LmScaleTensorNone();
			gemm.scale_b = scale;
			gemm.prefix_built = 1u;
			gemm.group_row_offset = state->group_row_offset;
			gemm.group_tile_prefix = state->tile_prefix_up;
			gemm.source_row_map = state->route_source_token;
			gemm.source_row_count = rows;
			gemm.output_bf16 = output;
			launch = LmGemmWeightOnlyIndirectLaunch<LmNvfp4,QH_GEMM_TILE_N,QH_GEMM_STAGES,QH_GEMM_WARPS>(&gemm, state->normed, weight, routes, rows,
				G::kTopK, G::kLocalExperts, G::kHidden, G::kMoeInter, sms, stream);
		}
		return launch;
	}

	static SparkStatus Moe(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * G::kTopK;
		const LmScaleTensor w2_scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_NVFP4_E2M1>(layer->moe_w2_scale, G::kLocalExperts, G::kHidden, G::kMoeInter);
		int32_t launch;
		if ( Project<LmBf16Format>(layer->moe_router, 0, state->normed, state->router_logits, rows, G::kHidden, G::kExperts, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridSoftmaxRowsKernel<G::kExperts><<<rows, QH_THREADS, 0, stream>>>(state->router_logits, state->router_probs);
		if ( LmTopkRouteLaunch<QH_THREADS,G::kTopK,true,LM_TOPK_SCORE_IDENTITY>(rows, state->router_probs, G::kExperts, state->route_expert,
			state->route_weight, (const float *)0, (const uint16_t *)0, 1.0f, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<QH_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, QH_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down,
			stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		launch = ExpertsUp(state, layer->moe_w1, layer->moe_w1_scale, state->expert_gate, rows, routes, sms, stream);
		if ( launch == LM_LAUNCH_OK )
			launch = ExpertsUp(state, layer->moe_w3, layer->moe_w3_scale, state->expert_up, rows, routes, sms, stream);
		if ( launch != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmSwigluPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), QH_THREADS, 0, stream>>>(state->expert_gate, state->expert_up,
			state->packed_weight, state->expert_act, G::kMoeInter, INFINITY);
		launch = LmSkinnyGroupedExperts<LmNvfp4>(layer->moe_w2, w2_scale, state->expert_act, state->expert_out, state->group_row_offset,
			state->route_source_token, G::kLocalExperts, routes, 1u, G::kMoeInter, G::kHidden, stream);
		if ( launch == LM_LAUNCH_ERR_SHAPE )
		{
			LmGemmArguments gemm;
			memset(&gemm, 0, sizeof(gemm));
			gemm.scale_a = LmScaleTensorNone();
			gemm.scale_b = w2_scale;
			gemm.prefix_built = 1u;
			gemm.group_row_offset = state->group_row_offset;
			gemm.group_tile_prefix = state->tile_prefix_down;
			gemm.output_bf16 = state->expert_out;
			launch = LmGemmWeightOnlyLaunch<LmNvfp4,QH_GEMM_TILE_N,QH_GEMM_STAGES,QH_GEMM_WARPS>(&gemm, state->expert_act, layer->moe_w2, routes, rows,
				G::kTopK, G::kLocalExperts, G::kMoeInter, G::kHidden, sms, true, stream);
		}
		if ( launch != LM_LAUNCH_OK ||
			Project<LmBf16Format>(layer->moe_shared_gate_up, 0, state->normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridRowDotKernel<<<rows, QH_THREADS, 0, stream>>>(state->normed, layer->moe_shared_gate, state->shared_logit, G::kHidden);
		LM_LAUNCH((LmSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows, true);
		if ( Launched() != SPARK_STATUS_OK ||
			Project<LmBf16Format>(layer->moe_shared_down, 0, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		QwenHybridSigmoidGateRowsKernel<<<dim3(Blocks(G::kHidden), rows), QH_THREADS, 0, stream>>>(state->shared_out, state->shared_logit, G::kHidden);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), QH_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return Launched();
	}

	static SparkStatus Ffn(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		if constexpr ( G::kMoe )
			return Moe(state, layer, step, stream);
		else
		{
			const uint32_t rows = step->rows, sms = step->multiprocessors;
			if ( Linear(state, layer->ffn_gate_up, layer->ffn_gate_up_scale, state->normed, state->gate_up, rows, G::kHidden,
				2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			LM_LAUNCH((LmSiluMulKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true);
			if ( Launched() != SPARK_STATUS_OK ||
				Linear(state, layer->ffn_down, layer->ffn_down_scale, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return SPARK_STATUS_OK;
		}
	}

	static SparkStatus DumpLayer(State *state, uint32_t layer, const SparkStageRunnerStep *step, const uint16_t *hidden,
		const uint16_t *partial, cudaStream_t stream)
	{
		const uint32_t header[3] = { 0x444c5751u, layer, step->rows };
		const uint64_t values = (uint64_t)step->rows * G::kHidden;
		if ( cudaStreamSynchronize(stream) != cudaSuccess ||
			cudaMemcpy(state->dump_positions, step->positions, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(state->dump_rows, hidden, (size_t)values * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
			(partial != 0 && cudaMemcpy(state->dump_rows + values, partial, (size_t)values * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( partial == 0 )
			memset(state->dump_rows + values, 0, (size_t)values * sizeof(uint16_t));
		fwrite(header, sizeof(header), 1u, state->dump);
		fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		fwrite(state->dump_rows, sizeof(uint16_t), (size_t)(2u * values), state->dump);
		fflush(state->dump);
		return SPARK_STATUS_OK;
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
		if ( state->dump != 0 )
		{
			cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
			if ( cudaStreamIsCapturing(stream, &capturing) != cudaSuccess || capturing != cudaStreamCaptureStatusNone )
				return SPARK_STATUS_UNSUPPORTED;
			status = DumpLayer(state, 0xffffffffu, step, hidden, 0, stream);
		}
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const QwenHybridLayer *layer = &state->layers[index];
			LM_LAUNCH((LmFusedResidualRmsNormKernel<QH_THREADS,float>), rows, QH_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, index == 0u ? (const uint16_t *)0 : state->partial, layer->input_norm, index == 0u ? (uint16_t *)0 : hidden,
				state->normed, G::kHidden, G::kHidden, QH_EPSILON);
			status = index % G::kPeriod == G::kPhase ? Attention(state, layer, step, stream) : Gdn(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			LM_LAUNCH((LmFusedResidualRmsNormKernel<QH_THREADS,float>), rows, QH_THREADS, (G::kHidden + 8u) * sizeof(float), stream,
				hidden, state->partial, layer->post_norm, hidden, state->normed, G::kHidden, G::kHidden, QH_EPSILON);
			status = Ffn(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status == SPARK_STATUS_OK && state->dump != 0 )
				status = DumpLayer(state, index, step, hidden, state->partial, stream);
			if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		QwenHybridAddRowsKernel<<<Blocks((uint64_t)rows * G::kHidden), QH_THREADS, 0, stream>>>(hidden, state->partial, (uint64_t)rows * G::kHidden);
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
		const uint64_t rows = state->max_rows, slots = state->slots, routes = rows * (G::kMoe ? G::kTopK : 0u);
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t);
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->qkv, rows * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->z, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->ba, rows * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->query, rows * state->gdn_qk * sizeof(uint16_t) },
			{ (void **)&state->key, rows * state->gdn_qk * sizeof(uint16_t) },
			{ (void **)&state->value, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->query_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->key_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->retention, rows * state->gdn_heads * QH_KEY_DIM * sizeof(float) },
			{ (void **)&state->write_gate, rows * state->gdn_heads * sizeof(float) },
			{ (void **)&state->mixed, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->attn_local, rows * state->attn_rows * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kQkvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kQkvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_query, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_gate, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_key, rows * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_value, rows * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_slot, rows * 2u * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->merged_gate, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->router_logits, rows * (G::kMoe ? G::kExperts : 0u) * sizeof(uint16_t) },
			{ (void **)&state->router_probs, rows * (G::kMoe ? G::kExperts : 0u) * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, (G::kMoe ? G::kLocalExperts + 2u : 0u) * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, (G::kMoe ? G::kLocalExperts + 2u : 0u) * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, (G::kMoe ? G::kLocalExperts + 2u : 0u) * sizeof(uint32_t) },
			{ (void **)&state->packed_weight, routes * sizeof(float) },
			{ (void **)&state->expert_gate, routes * (G::kMoe ? G::kMoeInter : 0u) * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * (G::kMoe ? G::kMoeInter : 0u) * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * (G::kMoe ? G::kMoeInter : 0u) * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->shared_gate_up, rows * (G::kMoe ? 2u * G::kSharedRows : 0u) * sizeof(uint16_t) },
			{ (void **)&state->shared_act, rows * (G::kMoe ? G::kSharedRows : 0u) * sizeof(uint16_t) },
			{ (void **)&state->shared_out, (G::kMoe ? width : 0u) },
			{ (void **)&state->shared_logit, rows * (G::kMoe ? 1u : 0u) * sizeof(uint16_t) },
			{ (void **)&state->gdn_state, (uint64_t)S::kGdnLayers * slots * state->gdn_state_bytes },
			{ (void **)&state->gdn_window, (uint64_t)S::kGdnLayers * slots * state->gdn_window_bytes },
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
		if ( state->dump != 0 )
			fclose(state->dump);
		free(state->dump_rows);
		free(state->dump_positions);
		free(state);
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if constexpr ( G::kMoe )
		{
			if ( SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK ||
				experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		else if ( SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK || state->ffn_rows == 0u ||
			state->ffn_rows % QH_SCALE_K != 0u )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		State *state;
		SparkStatus status;
		uint32_t embed_rows, head_rows;
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree ||
			configuration->max_input_row_count == 0u || configuration->state_budget_bytes == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state = (State *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->max_rows = configuration->max_input_row_count;
		if ( getenv(G::DumpEnv()) != 0 )
		{
			state->dump = fopen(getenv(G::DumpEnv()), "wb");
			state->dump_rows = (uint16_t *)malloc((size_t)2u * state->max_rows * G::kHidden * sizeof(uint16_t));
			state->dump_positions = (uint32_t *)malloc((size_t)state->max_rows * sizeof(uint32_t));
			if ( state->dump == 0 || state->dump_rows == 0 || state->dump_positions == 0 )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fprintf(stderr, "%s: this rank dumps every layer's residual and sublayer output to %s (debug; graphs off)\n",
				G::Tag(), getenv(G::DumpEnv()));
		}
		if ( configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 && !G::kMoe )
		{
			state->linear_bytes = 1u;
			state->linear_kind = "fp8_e4m3";
		}
		else if ( configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 )
		{
			state->linear_bytes = 2u;
			state->linear_kind = "bf16";
		}
		else
		{
			fprintf(stderr, "%s: linear weight codec %u is not supported by this model\n", G::Tag(), configuration->linear_weight_codec);
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, QH_PACK_ALIGNMENT, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		status = Config(state, configuration);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->gdn_heads = G::kGdnValueHeads / state->tp_degree;
		state->gdn_qk = G::kGdnKeyHeads / state->tp_degree * QH_KEY_DIM;
		state->gdn_v = state->gdn_heads * QH_VALUE_DIM;
		state->gdn_channels = 2u * state->gdn_qk + state->gdn_v;
		state->attn_rows = S::kQkvWidth / state->tp_degree;
		state->attn_width = S::kQWidth / state->tp_degree;
		state->gdn_state_bytes = (uint64_t)state->gdn_heads * QH_KEY_DIM * QH_VALUE_DIM * sizeof(float);
		state->gdn_window_bytes = (uint64_t)state->gdn_channels * QH_CONV * sizeof(uint16_t);
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, state->tp_degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
		status = Plan(state, &state->arena_bytes, 0);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes is over the %llu-byte budget (rows %u, slots %u)\n", G::Tag(),
				(unsigned long long)state->arena_bytes, (unsigned long long)configuration->state_budget_bytes, state->max_rows, state->slots);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = G::kHidden;
		geometry->vocab = G::kVocab;
		geometry->total_layers = G::kLayers;
		geometry->first_layer = 0u;
		geometry->layer_count = G::kLayers;
		geometry->rms_epsilon = QH_EPSILON;
		geometry->pack_bytes = state->pack.file_bytes;
		geometry->kv_layer_count = S::kAttnLayers;
		geometry->kv_layer_page_bytes = S::Kv::kPageBytes;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->recurrent_bytes = (uint64_t)S::kGdnLayers * (state->gdn_state_bytes + state->gdn_window_bytes);
		geometry->head_norm_f32 = 1u;
		if ( Entry(state, "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			Entry(state, "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
			Entry(state, "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &head_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->embed_rows = embed_rows;
		geometry->head_rows = head_rows;
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

	static SparkStatus Slice(State *state, SparkWeightdLazyPack *lazy_pack, const char *name, const char *kind, const void **out)
	{
		uint64_t offset, bytes;
		if ( Entry(state, name, kind, &offset, &bytes, 0) != SPARK_STATUS_OK ||
			SparkWeightdLazyPackSlice(lazy_pack, offset, bytes, out) != SPARK_STATUS_OK || *out == 0 )
		{
			fprintf(stderr, "%s: weightd has no resident bytes for %s\n", G::Tag(), name);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindTensor(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, const void **field)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(key, sizeof(key), "layers.%u.%s", index, name);
		return Slice(state, lazy_pack, key, kind, field);
	}

	template<class T>
	static SparkStatus Bind(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, T **field)
	{
		const void *bound = 0;
		const SparkStatus status = BindTensor(state, lazy_pack, index, name, kind, &bound);
		*field = (T *)bound;
		return status;
	}

	template<class W>
	static SparkStatus BindLinear(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, W **field, const float **scale_field)
	{
		char scale_name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		SparkStatus status = Bind(state, lazy_pack, index, name, state->linear_kind, field);
		*scale_field = 0;
		if ( status == SPARK_STATUS_OK && state->linear_bytes == 1u )
		{
			snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
			status = Bind(state, lazy_pack, index, scale_name, "f32", scale_field);
		}
		return status;
	}

	static SparkStatus BindExperts(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const uint8_t **field,
		const uint8_t **scale_field)
	{
		char scale_name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
		SparkStatus status = Bind(state, lazy_pack, index, name, "nvfp4_e2m1", field);
		if ( status == SPARK_STATUS_OK )
			status = Bind(state, lazy_pack, index, scale_name, "ue4m3_f32_global", scale_field);
		return status;
	}

	static SparkStatus BindModel(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		uint32_t recurrent = 0u, attention = 0u;
		(void)geometry;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			QwenHybridLayer *layer = &state->layers[index];
			const uint32_t attention_layer = index % G::kPeriod == G::kPhase;
			SparkStatus status = SPARK_STATUS_OK;
			if constexpr ( G::kMoe )
			{
				status = Bind(state, lazy_pack, index, "moe_router", "bf16", &layer->moe_router);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w1", &layer->moe_w1, &layer->moe_w1_scale);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w3", &layer->moe_w3, &layer->moe_w3_scale);
				if ( status == SPARK_STATUS_OK )
					status = BindExperts(state, lazy_pack, index, "moe_w2", &layer->moe_w2, &layer->moe_w2_scale);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_shared_gate_up", "bf16", &layer->moe_shared_gate_up);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_shared_down", "bf16", &layer->moe_shared_down);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "moe_shared_gate", "bf16", &layer->moe_shared_gate);
			}
			else
			{
				status = BindLinear(state, lazy_pack, index, "ffn_gate_up", &layer->ffn_gate_up, &layer->ffn_gate_up_scale);
				if ( status == SPARK_STATUS_OK )
					status = BindLinear(state, lazy_pack, index, "ffn_down", &layer->ffn_down, &layer->ffn_down_scale);
			}
			if ( status == SPARK_STATUS_OK )
				status = BindLinear(state, lazy_pack, index, attention_layer ? "attn_qkv" : "gdn_qkvz", &layer->mix_weight, &layer->mix_scale);
			if ( status == SPARK_STATUS_OK )
				status = BindLinear(state, lazy_pack, index, attention_layer ? "attn_out" : "gdn_out", &layer->out_weight, &layer->out_scale);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "input_norm", "f32", &layer->input_norm);
			if ( status == SPARK_STATUS_OK )
				status = Bind(state, lazy_pack, index, "post_norm", "f32", &layer->post_norm);
			if ( status == SPARK_STATUS_OK && attention_layer )
			{
				status = Bind(state, lazy_pack, index, "attn_q_norm", "f32", &layer->q_norm);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "attn_k_norm", "f32", &layer->k_norm);
				layer->kv_index = attention++;
			}
			else if ( status == SPARK_STATUS_OK )
			{
				status = Bind(state, lazy_pack, index, "gdn_ba", "bf16", &layer->gdn_ba);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_conv", "bf16", &layer->gdn_conv);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_norm", "bf16", &layer->gdn_norm);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_a_log", "f32", &layer->gdn_a_log);
				if ( status == SPARK_STATUS_OK )
					status = Bind(state, lazy_pack, index, "gdn_dt_bias", "f32", &layer->gdn_dt_bias);
				layer->recurrent_index = recurrent++;
			}
			if ( status != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			fprintf(stderr, "%s: the %llu-byte rank state could not be allocated\n", G::Tag(), (unsigned long long)state->arena_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return Plan(state, &state->arena_bytes, (uint8_t *)state->arena);
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

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		State *state = (State *)model;
		const uint64_t widths[2] = { state->gdn_state_bytes, state->gdn_window_bytes };
		uint8_t *pools[2] = { state->gdn_state, (uint8_t *)state->gdn_window };
		uint8_t *packed = (uint8_t *)buffer;
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots || buffer == 0 || bytes != (uint64_t)S::kGdnLayers * (widths[0] + widths[1]) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < 2u && error == cudaSuccess; ++part )
		{
			uint8_t *pool = pools[part] + (uint64_t)slot * widths[part];
			const size_t pitch = (size_t)(widths[part] * state->slots);
			error = to_buffer != 0u ?
				cudaMemcpy2DAsync(packed, (size_t)widths[part], pool, pitch, (size_t)widths[part], S::kGdnLayers, cudaMemcpyDefault, (cudaStream_t)stream) :
				cudaMemcpy2DAsync(pool, pitch, packed, (size_t)widths[part], (size_t)widths[part], S::kGdnLayers, cudaMemcpyDefault, (cudaStream_t)stream);
			packed += widths[part] * S::kGdnLayers;
		}
		if ( error == cudaSuccess && stream == 0 )
			error = cudaStreamSynchronize(0);
		return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		State *state = (State *)model;
		const uint64_t widths[2] = { state->gdn_state_bytes, state->gdn_window_bytes };
		uint8_t *pools[2] = { state->gdn_state, (uint8_t *)state->gdn_window };
		cudaError_t error = cudaSuccess;
		if ( slot >= state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t part = 0u; part < 2u && error == cudaSuccess; ++part )
			error = cudaMemset2DAsync(pools[part] + (uint64_t)slot * widths[part], (size_t)(widths[part] * state->slots), 0,
				(size_t)widths[part], S::kGdnLayers, (cudaStream_t)stream);
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
