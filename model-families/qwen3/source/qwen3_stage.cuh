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
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_qwen3_family.h"
#include "sparkpipe/spark_gpu.h"

#define Q3_THREADS 256u

constexpr uint32_t Qwen3GatherParts(uint32_t ranks, uint32_t width, uint32_t limit)
{
	uint32_t parts = ranks == 1u ? 1u : (width + limit - 1u) / limit;
	while ( width % parts != 0u )
		++parts;
	return parts;
}

template<class G>
struct Qwen3Shape : Qwen3Layout<G>
{
	using Kv = LmKvHeads<16u, G::kKvHeads, G::kHeadDim, G::kPageSlots>;
	static constexpr uint32_t kGatherParts = Qwen3GatherParts(G::kTpDegree, Qwen3Layout<G>::kQkvWidth / G::kTpDegree, G::kHidden);
	static_assert(Qwen3Layout<G>::kQkvWidth % G::kTpDegree == 0u && Qwen3Layout<G>::kQWidth % G::kTpDegree == 0u,
		"the fused attention rows and the query heads split evenly over the ranks");
	static_assert(Qwen3Layout<G>::kQkvWidth / G::kTpDegree % kGatherParts == 0u &&
		(G::kTpDegree == 1u || Qwen3Layout<G>::kQkvWidth / G::kTpDegree / kGatherParts <= G::kHidden),
		"a rank's fused attention rows gather as whole collective rows of at most the hidden width");
};

typedef struct Qwen3Layer
{
	const uint16_t *qkv;
	const uint16_t *out;
	const float *q_norm;
	const float *k_norm;
	const uint16_t *ffn_gate_up;
	const uint16_t *ffn_down;
	const float *input_norm;
	const float *post_norm;
} Qwen3Layer;

template<class G>
struct Qwen3State : CudaStagePackState
{
	uint32_t ffn_rows;
	uint32_t attn_rows;
	uint32_t attn_width;
	uint64_t record_stride;
	Qwen3Layer layers[G::kLayers];
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *gate_up;
	uint16_t *intermediate;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *attn_full;
	uint16_t *attn_query;
	uint16_t *attn_key;
	uint16_t *attn_value;
	uint16_t *attn_slot;
	float *send;
	float *received;
	uint16_t *merged;
	float *inv_freq;
	LmKvShardView kv_views[G::kLayers];
};

__global__ static void Qwen3AddRowsKernel(uint16_t *hidden, const uint16_t *addend, uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
}

__global__ static void Qwen3PackSlotKernel(const uint16_t *key, const uint16_t *value, uint16_t *slot, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 2u * width; index += gridDim.x * blockDim.x )
		slot[(uint64_t)row * 2u * width + index] = index < width ? key[(uint64_t)row * width + index] : value[(uint64_t)row * width + index - width];
}

template<class G>
struct Qwen3Model : CudaStage<Qwen3Model<G>>
{
	using S = Qwen3Shape<G>;
	using State = Qwen3State<G>;
	using Layer = Qwen3Layer;
	using Schedule = Qwen3Schedule<G, Qwen3Model>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::Kv, &State::kv_views, 0u, G::kLayers>>;

	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = Q3_EPSILON;
	static constexpr uint32_t kPackAlignment = Q3_PACK_ALIGNMENT;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
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
		const uint64_t blocks = (elements + Q3_THREADS - 1u) / Q3_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	static int32_t Project(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
	{
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension, output_dimension,
			0u, 0u, multiprocessors, stream);
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->commit == 0u )
		{
			fprintf(stderr, "%s: a verify frame is refused: this stage commits every row it computes\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		frame->step = step;
		frame->stream = (SparkGpuStream)stream;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Attention(State *state, const Frame &frame, const Qwen3Layer &entry)
	{
		const Qwen3Layer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = &state->kv_views[layer - state->layers];
		LmQkvLayout layout;
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Project(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, state->attn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * S::kGatherParts, state->attn_rows / S::kGatherParts,
			state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		layout.query_dimension = S::kQWidth;
		layout.key_dimension = S::kKvWidth;
		layout.value_dimension = S::kKvWidth;
		layout.rope_dimension = 0u;
		layout.head_dimension = G::kHeadDim;
		LM_LAUNCH((LmRankMajorRowsKernel<Q3_THREADS>), dim3(Blocks(S::kQkvWidth),rows), Q3_THREADS, 0, stream, state->attn_gathered, state->attn_full, rows,
			state->attn_rows, degree);
		LM_LAUNCH((LmSplitQkvKernel<Q3_THREADS>), rows, Q3_THREADS, 0, stream, state->attn_full, layout, state->attn_query, state->attn_key,
			state->attn_value, rows, 1.0f);
		LM_LAUNCH((LmHeadRmsNormKernel<Q3_THREADS,float,true>), dim3(G::kHeads,rows), Q3_THREADS, 0, stream,
			state->attn_query, layer->q_norm, state->attn_query, rows, G::kHeads, G::kHeadDim, Q3_EPSILON, 1.0f);
		LM_LAUNCH((LmHeadRmsNormKernel<Q3_THREADS,float,true>), dim3(G::kKvHeads,rows), Q3_THREADS, 0, stream,
			state->attn_key, layer->k_norm, state->attn_key, rows, G::kKvHeads, G::kHeadDim, Q3_EPSILON, 1.0f);
		LM_LAUNCH((LmRopePerHeadKernel<Q3_THREADS,LM_ROPE_HALF_SPLIT,LM_ROPE_ROUNDED_TERMS>), dim3(rows,G::kHeads), Q3_THREADS, 0, stream,
			state->attn_query, step->positions, G::kHeads, G::kHeadDim, G::kHeadDim, G::kRopeTheta, (const float *)state->inv_freq, 1.0f, 0u);
		LM_LAUNCH((LmRopePerHeadKernel<Q3_THREADS,LM_ROPE_HALF_SPLIT,LM_ROPE_ROUNDED_TERMS>), dim3(rows,G::kKvHeads), Q3_THREADS, 0, stream,
			state->attn_key, step->positions, G::kKvHeads, G::kHeadDim, G::kHeadDim, G::kRopeTheta, (const float *)state->inv_freq, 1.0f, 0u);
		LM_LAUNCH((Qwen3PackSlotKernel), dim3(Blocks(2u * S::kKvWidth),rows), Q3_THREADS, 0, stream, state->attn_key, state->attn_value, state->attn_slot,
			S::kKvWidth);
		CudaStageRowState(state, (uint32_t)(layer - state->layers), SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,Q3_THREADS>), rows, Q3_THREADS, 0, stream,
			*view, state->attn_slot, step->sequence_of_row, step->positions, rows, 2u * S::kKvWidth);
		CudaStageRowState(state, (uint32_t)(layer - state->layers), SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kKvHeads,G::kHeadDim,G::kHeadDim>(*view, state->attn_query, G::kHeads,
				step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kHeadDim), state->send, stride, rows, stream) !=
				SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<G::kHeadDim>(state->received, stride, degree, G::kHeads, state->tp_rank, state->merged, rows, stream) != SPARK_GPU_SUCCESS ||
			Project(layer->out, state->merged, state->partial, rows, state->attn_width, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Ffn(State *state, const Frame &frame, const Qwen3Layer &entry)
	{
		const Qwen3Layer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		if ( Project(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmSiluRoundedMulLimitKernel<Q3_THREADS>), rows, Q3_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true, INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Project(layer->ffn_down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<Q3_THREADS,float,true>), frame.step->rows, Q3_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
			frame.step->hidden_bf16, (const uint16_t *)0, weight, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, Q3_EPSILON);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, const float *weight)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		LM_LAUNCH((LmFusedResidualRmsNormKernel<Q3_THREADS,float,true>), frame.step->rows, Q3_THREADS, (G::kHidden + 8u) * sizeof(float), frame.stream,
			hidden, state->partial, weight, hidden, state->normed, G::kHidden, G::kHidden, Q3_EPSILON);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus AddRows(State *state, const Frame &frame)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		LM_LAUNCH((Qwen3AddRowsKernel), Blocks(width), Q3_THREADS, 0, frame.stream, frame.step->hidden_bf16, state->partial, width);
		return SPARK_STATUS_OK;
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t);
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->attn_local, rows * state->attn_rows * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kQkvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kQkvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_query, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_key, rows * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_value, rows * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->attn_slot, rows * 2u * S::kKvWidth * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->inv_freq, (G::kHeadDim / 2u) * sizeof(float) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			(uint64_t)vocab_rows * degree != G::kVocab || state->ffn_rows == 0u || state->ffn_rows * degree != G::kFfn )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u vocab rows %u ffn rows %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, vocab_rows, state->ffn_rows, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 )
		{
			fprintf(stderr, "%s: linear weight codec %u is not supported by this model\n", G::Tag(), configuration->linear_weight_codec);
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Shape(State *state)
	{
		state->attn_rows = S::kQkvWidth / state->tp_degree;
		state->attn_width = S::kQWidth / state->tp_degree;
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, state->tp_degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->head_norm_round_first = 1u;
		if constexpr ( G::kHeadless )
		{
			geometry->headless = 1u;
			if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &geometry->embed_rows) !=
				SPARK_STATUS_OK || SparkNamedPackTensorRange(&state->pack, G::Tag(), "head_norm", "f32", &geometry->head_norm_offset,
				&geometry->head_norm_bytes, 0) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			return SPARK_STATUS_OK;
		}
		else
			return CudaStage<Qwen3Model>::HeadRanges(state, geometry, G::kTiedHead, false);
	}

	template<class T>
	static void Elements(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t elements)
	{
		tensors->Add(name, kind, elements * sizeof(T), field);
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *t)
	{
		Qwen3Layer *layer = &state->layers[index];
		Elements(t, "attn_qkv", "bf16", &layer->qkv, (uint64_t)state->attn_rows * G::kHidden);
		Elements(t, "attn_out", "bf16", &layer->out, (uint64_t)G::kHidden * state->attn_width);
		Elements(t, "attn_q_norm", "f32", &layer->q_norm, G::kHeadDim);
		Elements(t, "attn_k_norm", "f32", &layer->k_norm, G::kHeadDim);
		Elements(t, "ffn_gate_up", "bf16", &layer->ffn_gate_up, (uint64_t)2u * state->ffn_rows * G::kHidden);
		Elements(t, "ffn_down", "bf16", &layer->ffn_down, (uint64_t)G::kHidden * state->ffn_rows);
		Elements(t, "input_norm", "f32", &layer->input_norm, G::kHidden);
		Elements(t, "post_norm", "f32", &layer->post_norm, G::kHidden);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[G::kHeadDim / 2u];
		for ( uint32_t pair = 0u; pair < G::kHeadDim / 2u; ++pair )
			inv_freq[pair] = 1.0f / (float)pow((double)G::kRopeTheta, (double)((float)(2u * pair) / (float)G::kHeadDim));
		if ( SparkGpuMemcpy(state->inv_freq, inv_freq, sizeof(inv_freq), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "ffn_rows", state->ffn_rows }
		};
		return CudaStageReportOut(fields, report);
	}
};
