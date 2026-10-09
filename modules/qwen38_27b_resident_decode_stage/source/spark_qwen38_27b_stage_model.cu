#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "runtime/launch.h"

#define QW_HIDDEN SPARK_QWEN38_27B_STAGE_HIDDEN
#define QW_LAYERS 64u
#define QW_VOCAB 248320u
#define QW_PERIOD 4u
#define QW_PHASE 3u
#define QW_GDN_LAYERS (QW_LAYERS - QW_LAYERS / QW_PERIOD)
#define QW_ATTN_LAYERS (QW_LAYERS / QW_PERIOD)
#define QW_KEY_DIM 128u
#define QW_VALUE_DIM 128u
#define QW_GDN_KEY_HEADS 16u
#define QW_GDN_VALUE_HEADS 48u
#define QW_CONV 4u
#define QW_HEADS 24u
#define QW_KV_HEADS SPARK_QWEN38_27B_STAGE_KV_HEADS
#define QW_HEAD_DIM SPARK_QWEN38_27B_STAGE_HEAD_DIM
#define QW_Q_WIDTH (QW_HEADS * QW_HEAD_DIM)
#define QW_KV_WIDTH (QW_KV_HEADS * QW_HEAD_DIM)
#define QW_QKV_WIDTH SPARK_QWEN38_27B_STAGE_QKV_WIDTH
#define QW_ROPE_DIM 64u
#define QW_ROPE_THETA 10000000.0f
#define QW_EPSILON 1e-6f
#define QW_THREADS 256u
#define QW_SCALE_K 128u
#define QW_PACK_ALIGNMENT 128u

using QwenKv = LmKvHeads<16u, QW_KV_HEADS, QW_HEAD_DIM, SPARK_QWEN38_27B_STAGE_PAGE_SLOTS>;

static_assert(QwenKv::kSlotBytes == SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES, "a KV slot is [K: 4 x 256][V: 4 x 256] bf16");
static_assert(QW_QKV_WIDTH == 2u * QW_Q_WIDTH + 2u * QW_KV_WIDTH, "the fused attention projection is query|gate, key, value");

typedef struct QwenLayer
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
	const float *input_norm;
	const float *post_norm;
	uint32_t recurrent_index;
	uint32_t kv_index;
} QwenLayer;

typedef struct QwenModelState
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t ffn_rows;
	uint32_t gdn_heads;
	uint32_t gdn_qk;
	uint32_t gdn_v;
	uint32_t gdn_channels;
	uint32_t attn_rows;
	uint32_t attn_width;
	uint64_t gdn_state_bytes;
	uint64_t gdn_window_bytes;
	uint64_t record_stride;
	QwenLayer layers[QW_LAYERS];
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
	uint8_t *gdn_state;
	uint16_t *gdn_window;
	LmKvAccessError *kv_error;
	LmKvShardView kv_views[QW_ATTN_LAYERS];
	uint32_t kv_attached;
} QwenModelState;

__global__ static void QwenAddRowsKernel(uint16_t *hidden, const uint16_t *addend, uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
}

__global__ static void QwenGatherRowsKernel(const uint16_t *gathered, uint16_t *full, uint32_t rows, uint32_t width, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width * ranks; index += gridDim.x * blockDim.x )
	{
		const uint32_t rank = index / width, element = index % width;
		full[(uint64_t)row * width * ranks + index] = gathered[((uint64_t)rank * rows + row) * width + element];
	}
}

__global__ static void QwenSplitAttentionKernel(const uint16_t *full, uint16_t *query, uint16_t *gate, uint16_t *key, uint16_t *value)
{
	const uint32_t row = blockIdx.y;
	const uint16_t *source = full + (uint64_t)row * QW_QKV_WIDTH;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < QW_QKV_WIDTH; index += gridDim.x * blockDim.x )
	{
		if ( index < 2u * QW_Q_WIDTH )
		{
			const uint32_t head = index / (2u * QW_HEAD_DIM), within = index % (2u * QW_HEAD_DIM);
			uint16_t *target = within < QW_HEAD_DIM ? query : gate;
			target[(uint64_t)row * QW_Q_WIDTH + head * QW_HEAD_DIM + within % QW_HEAD_DIM] = source[index];
		}
		else if ( index < 2u * QW_Q_WIDTH + QW_KV_WIDTH )
			key[(uint64_t)row * QW_KV_WIDTH + index - 2u * QW_Q_WIDTH] = source[index];
		else
			value[(uint64_t)row * QW_KV_WIDTH + index - 2u * QW_Q_WIDTH - QW_KV_WIDTH] = source[index];
	}
}

__global__ static void QwenPackSlotKernel(const uint16_t *key, const uint16_t *value, uint16_t *slot)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < 2u * QW_KV_WIDTH; index += gridDim.x * blockDim.x )
		slot[(uint64_t)row * 2u * QW_KV_WIDTH + index] = index < QW_KV_WIDTH ? key[(uint64_t)row * QW_KV_WIDTH + index] :
			value[(uint64_t)row * QW_KV_WIDTH + index - QW_KV_WIDTH];
}

__global__ static void QwenSliceRowsKernel(const uint16_t *source, uint16_t *target, uint32_t source_width, uint32_t offset, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		target[(uint64_t)row * width + index] = source[(uint64_t)row * source_width + offset + index];
}

static uint32_t QwenBlocks(uint64_t elements)
{
	const uint64_t blocks = (elements + QW_THREADS - 1u) / QW_THREADS;
	return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
}

template<class Format>
static int32_t QwenProject(const void *weight, const float *scale_rows, const uint16_t *source, uint16_t *destination,
	uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	LmScaleTensor scale = scale_rows != 0 ?
		LmScaleTensorBuild(scale_rows, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u, QW_SCALE_K) : LmScaleTensorNone();
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

static SparkStatus QwenRound(QwenModelState *state, cudaStream_t stream, uint32_t operation, uint32_t rows,
	uint32_t row_elements, const void *local, void *full)
{
	return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL,
		rows, row_elements, local, full);
}

static SparkStatus QwenLaunched(void)
{
	return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus QwenGdn(QwenModelState *state, const QwenLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sequences = step->sequences, heads = state->gdn_heads;
	const uint32_t sms = step->multiprocessors;
	uint8_t *state_pool = state->gdn_state + (uint64_t)layer->recurrent_index * state->slots * state->gdn_state_bytes;
	uint16_t *window = state->gdn_window + (uint64_t)layer->recurrent_index * state->slots * (state->gdn_window_bytes / sizeof(uint16_t));
	LmQkvLayout layout;
	if ( QwenProject<LmFp8>(layer->mix_weight, layer->mix_scale, state->normed, state->qkv, rows, QW_HIDDEN,
			state->gdn_channels, sms, stream) != LM_LAUNCH_OK ||
		QwenProject<LmFp8>(layer->mix_weight + (uint64_t)state->gdn_channels * QW_HIDDEN, layer->mix_scale + (uint64_t)state->gdn_channels * (QW_HIDDEN / QW_SCALE_K),
			state->normed, state->z, rows, QW_HIDDEN, state->gdn_v, sms, stream) != LM_LAUNCH_OK ||
		QwenProject<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, QW_HIDDEN, 2u * heads, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LM_LAUNCH((LmCausalConvKernel<QW_THREADS,QW_CONV,LM_CONV_SWISH,uint16_t>), dim3(sequences,(state->gdn_channels + QW_THREADS - 1u) / QW_THREADS),
		QW_THREADS, 0, stream, window, step->recurrent_index, step->sequence_row_begin, (const uint32_t *)0, state->qkv, layer->gdn_conv,
		state->qkv, state->gdn_channels, sequences, step->commit, step->sequence_row_indices);
	layout.query_dimension = state->gdn_qk;
	layout.key_dimension = state->gdn_qk;
	layout.value_dimension = state->gdn_v;
	layout.rope_dimension = 0u;
	layout.head_dimension = QW_KEY_DIM;
	LM_LAUNCH((LmSplitQkvKernel<QW_THREADS>), rows, QW_THREADS, 0, stream, state->qkv, layout, state->query, state->key, state->value, rows, 1.0f);
	LM_LAUNCH((LmExpandHeadsKernel<QW_THREADS>), rows, QW_THREADS, 0, stream, state->query, state->query_heads,
		state->gdn_qk / QW_KEY_DIM, QW_KEY_DIM, heads / (state->gdn_qk / QW_KEY_DIM), rows);
	LM_LAUNCH((LmExpandHeadsKernel<QW_THREADS>), rows, QW_THREADS, 0, stream, state->key, state->key_heads,
		state->gdn_qk / QW_KEY_DIM, QW_KEY_DIM, heads / (state->gdn_qk / QW_KEY_DIM), rows);
	LM_LAUNCH((LmGdnGateKernel<QW_THREADS,QW_KEY_DIM>), dim3(rows,heads), QW_THREADS, 0, stream,
		state->ba + heads, state->ba, layer->gdn_a_log, layer->gdn_dt_bias, state->retention, state->write_gate, heads, rows, 2u * heads);
	LM_LAUNCH((LmDeltaRuleColumnKernel<QW_THREADS,QW_KEY_DIM,QW_VALUE_DIM,float>), dim3(sequences,heads,QW_VALUE_DIM / LM_WARP_LANES),
		LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index, step->sequence_row_begin,
		(const uint32_t *)0, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
		heads, 1u, sequences, step->commit, step->sequence_row_indices);
	LM_LAUNCH((LmHeadRmsNormSiluGateKernel<QW_THREADS>), dim3(heads,rows), QW_THREADS, 0, stream,
		state->mixed, state->z, layer->gdn_norm, state->mixed, rows, heads, QW_VALUE_DIM, QW_EPSILON);
	if ( QwenLaunched() != SPARK_STATUS_OK ||
		QwenProject<LmFp8>(layer->out_weight, layer->out_scale, state->mixed, state->partial, rows, state->gdn_v, QW_HIDDEN, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus QwenAttention(QwenModelState *state, const QwenLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
	const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(QW_HEADS, QW_HEAD_DIM, degree) * sizeof(float) +
		QW_HIDDEN * sizeof(uint16_t) - 1u) / (QW_HIDDEN * sizeof(uint16_t));
	const uint64_t stride = units * QW_HIDDEN * sizeof(uint16_t) / sizeof(float);
	const LmKvShardView *view = &state->kv_views[layer->kv_index];
	SparkStatus status;
	if ( state->kv_attached == 0u || stride > state->record_stride )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( QwenProject<LmFp8>(layer->mix_weight, layer->mix_scale, state->normed, state->attn_local, rows, QW_HIDDEN,
		state->attn_rows, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = QwenRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, state->attn_rows, state->attn_local, state->attn_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	QwenGatherRowsKernel<<<dim3(QwenBlocks(QW_QKV_WIDTH),rows), QW_THREADS, 0, stream>>>(state->attn_gathered, state->attn_full, rows, state->attn_rows, degree);
	QwenSplitAttentionKernel<<<dim3(QwenBlocks(QW_QKV_WIDTH),rows), QW_THREADS, 0, stream>>>(state->attn_full, state->attn_query,
		state->attn_gate, state->attn_key, state->attn_value);
	LM_LAUNCH((LmHeadRmsNormKernel<QW_THREADS,float>), dim3(QW_HEADS,rows), QW_THREADS, 0, stream,
		state->attn_query, layer->q_norm, state->attn_query, rows, QW_HEADS, QW_HEAD_DIM, QW_EPSILON, 1.0f);
	LM_LAUNCH((LmHeadRmsNormKernel<QW_THREADS,float>), dim3(QW_KV_HEADS,rows), QW_THREADS, 0, stream,
		state->attn_key, layer->k_norm, state->attn_key, rows, QW_KV_HEADS, QW_HEAD_DIM, QW_EPSILON, 1.0f);
	LM_LAUNCH((LmRopePerHeadKernel<QW_THREADS>), dim3(rows,QW_HEADS), QW_THREADS, 0, stream,
		state->attn_query, step->positions, QW_HEADS, QW_HEAD_DIM, QW_ROPE_DIM, QW_ROPE_THETA);
	LM_LAUNCH((LmRopePerHeadKernel<QW_THREADS>), dim3(rows,QW_KV_HEADS), QW_THREADS, 0, stream,
		state->attn_key, step->positions, QW_KV_HEADS, QW_HEAD_DIM, QW_ROPE_DIM, QW_ROPE_THETA);
	QwenPackSlotKernel<<<dim3(QwenBlocks(2u * QW_KV_WIDTH),rows), QW_THREADS, 0, stream>>>(state->attn_key, state->attn_value, state->attn_slot);
	LM_LAUNCH((LmKvShardStoreKernel<QwenKv,QW_THREADS>), rows, QW_THREADS, 0, stream,
		*view, state->attn_slot, step->sequence_of_row, step->positions, rows, 2u * QW_KV_WIDTH);
	if ( QwenLaunched() != SPARK_STATUS_OK ||
		LmGqaShardPartialLaunch<QwenKv,LmKvShardView,QW_KV_HEADS,QW_HEAD_DIM,QW_HEAD_DIM>(*view, state->attn_query, QW_HEADS,
			step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)QW_HEAD_DIM), state->send, stride, rows, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = QwenRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( LmGqaShardMergeLaunch<QW_HEAD_DIM>(state->received, stride, degree, QW_HEADS, state->tp_rank, state->merged, rows, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	QwenSliceRowsKernel<<<dim3(QwenBlocks(state->attn_width),rows), QW_THREADS, 0, stream>>>(state->attn_gate, state->merged_gate,
		QW_Q_WIDTH, state->tp_rank * state->attn_width, state->attn_width);
	LM_LAUNCH((LmOutputGateKernel<QW_THREADS>), rows, QW_THREADS, 0, stream, state->merged, state->merged_gate, state->attn_width);
	if ( QwenLaunched() != SPARK_STATUS_OK ||
		QwenProject<LmFp8>(layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, QW_HIDDEN, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus QwenFfn(QwenModelState *state, const QwenLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors;
	if ( QwenProject<LmFp8>(layer->ffn_gate_up, layer->ffn_gate_up_scale, state->normed, state->gate_up, rows, QW_HIDDEN,
		2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LM_LAUNCH((LmSiluMulKernel<QW_THREADS>), rows, QW_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true);
	if ( QwenLaunched() != SPARK_STATUS_OK ||
		QwenProject<LmFp8>(layer->ffn_down, layer->ffn_down_scale, state->intermediate, state->partial, rows, state->ffn_rows, QW_HIDDEN, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus QwenStep(void *model, const SparkStageRunnerStep *step, void *stream_void)
{
	QwenModelState *state = (QwenModelState *)model;
	cudaStream_t stream = (cudaStream_t)stream_void;
	const uint32_t rows = step->rows;
	uint16_t *hidden = step->hidden_bf16;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for ( uint32_t index = 0u; index < QW_LAYERS && status == SPARK_STATUS_OK; ++index )
	{
		const QwenLayer *layer = &state->layers[index];
		LM_LAUNCH((LmFusedResidualRmsNormKernel<QW_THREADS,float>), rows, QW_THREADS, (QW_HIDDEN + 8u) * sizeof(float), stream,
			hidden, index == 0u ? (const uint16_t *)0 : state->partial, layer->input_norm, index == 0u ? (uint16_t *)0 : hidden,
			state->normed, QW_HIDDEN, QW_HIDDEN, QW_EPSILON);
		status = index % QW_PERIOD == QW_PHASE ? QwenAttention(state, layer, step, stream) : QwenGdn(state, layer, step, stream);
		if ( status == SPARK_STATUS_OK )
			status = QwenRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((LmFusedResidualRmsNormKernel<QW_THREADS,float>), rows, QW_THREADS, (QW_HIDDEN + 8u) * sizeof(float), stream,
			hidden, state->partial, layer->post_norm, hidden, state->normed, QW_HIDDEN, QW_HIDDEN, QW_EPSILON);
		status = QwenFfn(state, layer, step, stream);
		if ( status == SPARK_STATUS_OK )
			status = QwenRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
		if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, index);
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	QwenAddRowsKernel<<<QwenBlocks((uint64_t)rows * QW_HIDDEN), QW_THREADS, 0, stream>>>(hidden, state->partial, (uint64_t)rows * QW_HIDDEN);
	return QwenLaunched();
}

static SparkStatus QwenEntry(QwenModelState *state, const char *name, const char *kind, uint64_t *offset, uint64_t *bytes, uint32_t *rows)
{
	SparkNamedPackEntry entry;
	if ( SparkNamedPackEntryLoad(&state->pack, name, &entry) != SPARK_STATUS_OK || strcmp(entry.kind, kind) != 0 || entry.shape_count < 1u )
	{
		fprintf(stderr, "sparkpipe_qwen38_27b: pack tensor %s missing or not %s\n", name, kind);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	*offset = state->pack.payload_base + entry.payload_offset;
	*bytes = entry.bytes;
	if ( rows != 0 )
		*rows = entry.shape[0];
	return SPARK_STATUS_OK;
}

static SparkStatus QwenPlan(QwenModelState *state, uint64_t *bytes, uint8_t *base)
{
	const uint64_t rows = state->max_rows, slots = state->slots;
	const uint64_t width = rows * QW_HIDDEN * sizeof(uint16_t);
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
		{ (void **)&state->query_heads, rows * state->gdn_heads * QW_KEY_DIM * sizeof(uint16_t) },
		{ (void **)&state->key_heads, rows * state->gdn_heads * QW_KEY_DIM * sizeof(uint16_t) },
		{ (void **)&state->retention, rows * state->gdn_heads * QW_KEY_DIM * sizeof(float) },
		{ (void **)&state->write_gate, rows * state->gdn_heads * sizeof(float) },
		{ (void **)&state->mixed, rows * state->gdn_v * sizeof(uint16_t) },
		{ (void **)&state->attn_local, rows * state->attn_rows * sizeof(uint16_t) },
		{ (void **)&state->attn_gathered, rows * QW_QKV_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->attn_full, rows * QW_QKV_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->attn_query, rows * QW_Q_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->attn_gate, rows * QW_Q_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->attn_key, rows * QW_KV_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->attn_value, rows * QW_KV_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->attn_slot, rows * 2u * QW_KV_WIDTH * sizeof(uint16_t) },
		{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
		{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
		{ (void **)&state->merged, rows * state->attn_width * sizeof(uint16_t) },
		{ (void **)&state->merged_gate, rows * state->attn_width * sizeof(uint16_t) },
		{ (void **)&state->gdn_state, (uint64_t)QW_GDN_LAYERS * slots * state->gdn_state_bytes },
		{ (void **)&state->gdn_window, (uint64_t)QW_GDN_LAYERS * slots * state->gdn_window_bytes },
		{ (void **)&state->kv_error, sizeof(LmKvAccessError) },
	};
	uint64_t cursor = 0u;
	for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
	{
		cursor = (cursor + 255u) & ~255ull;
		if ( base != 0 )
			*plan[index].target = base + cursor;
		cursor += plan[index].bytes;
	}
	*bytes = cursor;
	return SPARK_STATUS_OK;
}

static void QwenClose(void *model)
{
	QwenModelState *state = (QwenModelState *)model;
	if ( state == 0 )
		return;
	if ( state->arena != 0 )
		(void)cudaFree(state->arena);
	if ( state->pack_open != 0u )
		SparkNamedPackClose(&state->pack);
	free(state);
}

static SparkStatus QwenConfig(QwenModelState *state, const SparkStageRunnerConfiguration *configuration)
{
	uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u;
	if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != QW_HIDDEN || layers != QW_LAYERS ||
		vocab_rows * degree != QW_VOCAB || state->ffn_rows == 0u || state->ffn_rows % QW_SCALE_K != 0u )
	{
		fprintf(stderr, "sparkpipe_qwen38_27b: pack is tp %u rank %u hidden %u layers %u, the runner is tp %u rank %u\n",
			degree, rank, hidden, layers, configuration->tp_degree, configuration->tp_rank);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus QwenOpen(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
{
	const SparkStageRunnerConfiguration *configuration = request->configuration;
	QwenModelState *state;
	SparkStatus status;
	uint64_t offset, bytes;
	uint32_t embed_rows, head_rows;
	if ( configuration->stage_count != 1u || configuration->tp_degree != SPARK_QWEN38_27B_STAGE_TP_DEGREE ||
		configuration->max_input_row_count == 0u || configuration->state_budget_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (QwenModelState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	*model = state;
	state->services = request->services;
	state->tp_degree = configuration->tp_degree;
	state->tp_rank = configuration->tp_rank;
	state->max_rows = configuration->max_input_row_count;
	state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
		configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
	status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, QW_PACK_ALIGNMENT, &state->pack);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->pack_open = 1u;
	status = QwenConfig(state, configuration);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->gdn_heads = QW_GDN_VALUE_HEADS / state->tp_degree;
	state->gdn_qk = QW_GDN_KEY_HEADS / state->tp_degree * QW_KEY_DIM;
	state->gdn_v = state->gdn_heads * QW_VALUE_DIM;
	state->gdn_channels = 2u * state->gdn_qk + state->gdn_v;
	state->attn_rows = QW_QKV_WIDTH / state->tp_degree;
	state->attn_width = QW_Q_WIDTH / state->tp_degree;
	state->gdn_state_bytes = (uint64_t)state->gdn_heads * QW_KEY_DIM * QW_VALUE_DIM * sizeof(float);
	state->gdn_window_bytes = (uint64_t)state->gdn_channels * QW_CONV * sizeof(uint16_t);
	state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(QW_HEADS, QW_HEAD_DIM, state->tp_degree) * sizeof(float) +
		QW_HIDDEN * sizeof(uint16_t) - 1u) / (QW_HIDDEN * sizeof(uint16_t))) * QW_HIDDEN * sizeof(uint16_t) / sizeof(float);
	status = QwenPlan(state, &state->arena_bytes, 0);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->arena_bytes > configuration->state_budget_bytes )
	{
		fprintf(stderr, "sparkpipe_qwen38_27b: rank state %llu bytes is over the %llu-byte budget (rows %u, slots %u)\n",
			(unsigned long long)state->arena_bytes, (unsigned long long)configuration->state_budget_bytes, state->max_rows, state->slots);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	memset(geometry, 0, sizeof(*geometry));
	geometry->hidden = QW_HIDDEN;
	geometry->vocab = QW_VOCAB;
	geometry->total_layers = QW_LAYERS;
	geometry->first_layer = 0u;
	geometry->layer_count = QW_LAYERS;
	geometry->rms_epsilon = QW_EPSILON;
	geometry->pack_bytes = state->pack.file_bytes;
	geometry->kv_layer_count = QW_ATTN_LAYERS;
	geometry->kv_layer_page_bytes = QwenKv::kPageBytes;
	geometry->recurrent_bytes = (uint64_t)QW_GDN_LAYERS * (state->gdn_state_bytes + state->gdn_window_bytes);
	geometry->head_norm_f32 = 1u;
	if ( QwenEntry(state, "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
		QwenEntry(state, "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
		QwenEntry(state, "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &head_rows) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	geometry->embed_rows = embed_rows;
	geometry->head_rows = head_rows;
	(void)offset;
	(void)bytes;
	return SPARK_STATUS_OK;
}

static SparkStatus QwenManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
{
	(void)model;
	if ( manifest == 0 || manifest->range_count != 0u )
	{
		fprintf(stderr, "sparkpipe_qwen38_27b: a dense pack's weightd manifest carries no expert ranges\n");
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus QwenSlice(QwenModelState *state, SparkWeightdLazyPack *lazy_pack, const char *name, const char *kind, const void **out)
{
	uint64_t offset, bytes;
	if ( QwenEntry(state, name, kind, &offset, &bytes, 0) != SPARK_STATUS_OK ||
		SparkWeightdLazyPackSlice(lazy_pack, offset, bytes, out) != SPARK_STATUS_OK || *out == 0 )
	{
		fprintf(stderr, "sparkpipe_qwen38_27b: weightd has no resident bytes for %s\n", name);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	return SPARK_STATUS_OK;
}

#define QW_BIND(name, kind, field) \
	do { const void *bound = 0; snprintf(key, sizeof(key), "layers.%u." name, index); \
		if ( QwenSlice(state, lazy_pack, key, kind, &bound) != SPARK_STATUS_OK ) return SPARK_STATUS_PARSE_ERROR; \
		*(const void **)&layer->field = bound; } while ( 0 )

static SparkStatus QwenBind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
{
	QwenModelState *state = (QwenModelState *)model;
	char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
	uint32_t recurrent = 0u, attention = 0u;
	(void)geometry;
	for ( uint32_t index = 0u; index < QW_LAYERS; ++index )
	{
		QwenLayer *layer = &state->layers[index];
		QW_BIND("ffn_gate_up", "fp8_e4m3", ffn_gate_up);
		QW_BIND("ffn_gate_up.scale", "f32", ffn_gate_up_scale);
		QW_BIND("ffn_down", "fp8_e4m3", ffn_down);
		QW_BIND("ffn_down.scale", "f32", ffn_down_scale);
		QW_BIND("input_norm", "f32", input_norm);
		QW_BIND("post_norm", "f32", post_norm);
		if ( index % QW_PERIOD == QW_PHASE )
		{
			QW_BIND("attn_qkv", "fp8_e4m3", mix_weight);
			QW_BIND("attn_qkv.scale", "f32", mix_scale);
			QW_BIND("attn_out", "fp8_e4m3", out_weight);
			QW_BIND("attn_out.scale", "f32", out_scale);
			QW_BIND("attn_q_norm", "f32", q_norm);
			QW_BIND("attn_k_norm", "f32", k_norm);
			layer->kv_index = attention++;
		}
		else
		{
			QW_BIND("gdn_qkvz", "fp8_e4m3", mix_weight);
			QW_BIND("gdn_qkvz.scale", "f32", mix_scale);
			QW_BIND("gdn_out", "fp8_e4m3", out_weight);
			QW_BIND("gdn_out.scale", "f32", out_scale);
			QW_BIND("gdn_ba", "bf16", gdn_ba);
			QW_BIND("gdn_conv", "bf16", gdn_conv);
			QW_BIND("gdn_norm", "bf16", gdn_norm);
			QW_BIND("gdn_a_log", "f32", gdn_a_log);
			QW_BIND("gdn_dt_bias", "f32", gdn_dt_bias);
			layer->recurrent_index = recurrent++;
		}
	}
	if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
	{
		state->arena = 0;
		fprintf(stderr, "sparkpipe_qwen38_27b: the %llu-byte rank state could not be allocated\n", (unsigned long long)state->arena_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return QwenPlan(state, &state->arena_bytes, (uint8_t *)state->arena);
}

#undef QW_BIND

static SparkStatus QwenAttachKv(void *model, const SparkStageRunnerKv *kv)
{
	QwenModelState *state = (QwenModelState *)model;
	if ( kv == 0 || kv->layer_count != QW_ATTN_LAYERS || kv->layer_page_bytes != QwenKv::kPageBytes || kv->sequence_count > state->slots )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t layer = 0u; layer < QW_ATTN_LAYERS; ++layer )
		if ( LmKvShardViewInitialize<QwenKv>(&state->kv_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes, kv->page_table,
			kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->kv_attached = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus QwenRecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	QwenModelState *state = (QwenModelState *)model;
	const uint64_t widths[2] = { state->gdn_state_bytes, state->gdn_window_bytes };
	uint8_t *pools[2] = { state->gdn_state, (uint8_t *)state->gdn_window };
	uint8_t *packed = (uint8_t *)buffer;
	cudaError_t error = cudaSuccess;
	if ( slot >= state->slots || buffer == 0 || bytes != (uint64_t)QW_GDN_LAYERS * (widths[0] + widths[1]) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t part = 0u; part < 2u && error == cudaSuccess; ++part )
	{
		uint8_t *pool = pools[part] + (uint64_t)slot * widths[part];
		const size_t pitch = (size_t)(widths[part] * state->slots);
		error = to_buffer != 0u ?
			cudaMemcpy2DAsync(packed, (size_t)widths[part], pool, pitch, (size_t)widths[part], QW_GDN_LAYERS, cudaMemcpyDefault, (cudaStream_t)stream) :
			cudaMemcpy2DAsync(pool, pitch, packed, (size_t)widths[part], (size_t)widths[part], QW_GDN_LAYERS, cudaMemcpyDefault, (cudaStream_t)stream);
		packed += widths[part] * QW_GDN_LAYERS;
	}
	if ( error == cudaSuccess && stream == 0 )
		error = cudaStreamSynchronize(0);
	return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus QwenResetSlot(void *model, uint32_t slot, void *stream)
{
	QwenModelState *state = (QwenModelState *)model;
	const uint64_t widths[2] = { state->gdn_state_bytes, state->gdn_window_bytes };
	uint8_t *pools[2] = { state->gdn_state, (uint8_t *)state->gdn_window };
	cudaError_t error = cudaSuccess;
	if ( slot >= state->slots )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t part = 0u; part < 2u && error == cudaSuccess; ++part )
		error = cudaMemset2DAsync(pools[part] + (uint64_t)slot * widths[part], (size_t)(widths[part] * state->slots), 0,
			(size_t)widths[part], QW_GDN_LAYERS, (cudaStream_t)stream);
	return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus QwenSideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
{
	(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

static const void *QwenProbeBuffers(void *model)
{
	return model;
}

static void QwenReport(void *model, uint32_t rank)
{
	const QwenModelState *state = (const QwenModelState *)model;
	fprintf(stderr, "QWEN38_27B-STATE rank=%u arena_bytes=%llu slots=%u rows=%u ffn_rows=%u\n", rank,
		(unsigned long long)state->arena_bytes, state->slots, state->max_rows, state->ffn_rows);
}

static const SparkStageRunnerModelInterface spark_qwen38_27b_stage_model =
{
	SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
	"sparkpipe_qwen38_27b",
	"qwen3.8-27b",
	"fp8",
	QwenOpen,
	QwenManifestCheck,
	QwenBind,
	QwenClose,
	QwenStep,
	QwenSideband,
	QwenAttachKv,
	QwenRecurrentCopy,
	QwenResetSlot,
	QwenProbeBuffers,
	QwenReport
};

const SparkStageRunnerModelInterface *SparkQwen38_27bStageModel(void)
{
	return &spark_qwen38_27b_stage_model;
}
