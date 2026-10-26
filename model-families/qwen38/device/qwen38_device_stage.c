#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_ops.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_manifest.h"

#define QD_TAG "sparkpipe_qwen38_27b_device"
#define QD_LAYERS 64u
#define QD_HIDDEN 5120u
#define QD_VOCAB 248320u
#define QD_PERIOD 4u
#define QD_PHASE 3u
#define QD_GDN_LAYERS 48u
#define QD_ATTN_LAYERS 16u
#define QD_KEY_HEADS 16u
#define QD_VALUE_HEADS 48u
#define QD_KEY_DIM 128u
#define QD_VALUE_DIM 128u
#define QD_QK (QD_KEY_HEADS * QD_KEY_DIM)
#define QD_V (QD_VALUE_HEADS * QD_VALUE_DIM)
#define QD_CHANNELS (2u * QD_QK + QD_V)
#define QD_MIX (QD_CHANNELS + QD_V)
#define QD_GATES (2u * QD_VALUE_HEADS)
#define QD_HEADS 24u
#define QD_KV_HEADS 4u
#define QD_HEAD_DIM 256u
#define QD_Q (QD_HEADS * QD_HEAD_DIM)
#define QD_KV (QD_KV_HEADS * QD_HEAD_DIM)
#define QD_QKV (2u * QD_Q + 2u * QD_KV)
#define QD_ROPE 64u
#define QD_THETA 10000000.0
#define QD_FFN 17408u
#define QD_CONV 4u
#define QD_EPSILON 1e-6f
#define QD_PAGE_SLOTS 64u
#define QD_SLOT_BYTES (2u * QD_KV * 2u)
#define QD_PAGE_BYTES (QD_PAGE_SLOTS * QD_SLOT_BYTES)
#define QD_STATE_BYTES ((uint64_t)QD_VALUE_HEADS * QD_KEY_DIM * QD_VALUE_DIM * 4u)
#define QD_WINDOW_BYTES ((uint64_t)QD_CHANNELS * QD_CONV * 2u)
#define QD_DECODE_PARTITION_KEYS 512u
#define QD_PREFILL_PARTITION_KEYS 1024u
#define QD_PACK_ALIGNMENT 128u

typedef struct QwenDeviceLayer
{
	SparkTensor mix;
	SparkTensor out;
	SparkTensor ba;
	SparkTensor conv;
	SparkTensor a_log;
	SparkTensor dt_bias;
	SparkTensor gdn_norm;
	SparkTensor q_norm;
	SparkTensor k_norm;
	SparkTensor gate_up;
	SparkTensor down;
	SparkTensor input_norm;
	SparkTensor post_norm;
	uint32_t attention;
	uint32_t index;
} QwenDeviceLayer;

typedef struct QwenDeviceScratch
{
	SparkTensor normed;
	SparkTensor partial;
	SparkTensor mix;
	SparkTensor ba;
	SparkTensor delta;
	SparkTensor core;
	SparkTensor gated;
	SparkTensor query;
	SparkTensor key;
	SparkTensor records;
	SparkTensor gate_up;
	SparkTensor act;
} QwenDeviceScratch;

typedef struct QwenDeviceModel
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	SparkDeviceOps *ops;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t max_positions;
	uint64_t pack_bytes;
	uint64_t scratch_bytes;
	QwenDeviceLayer layers[QD_LAYERS];
	SparkDeviceBuffer scratch_buffer;
	SparkDeviceBuffer state;
	SparkDeviceBuffer window;
	QwenDeviceScratch scratch;
	SparkTensor kv_pool;
	SparkTensor page_table;
	uint64_t kv_layer_stride;
	uint32_t pages_per_sequence;
	uint32_t pool_pages;
	uint32_t kv_attached;
	SparkHeadRopeParams rope_query;
	SparkHeadRopeParams rope_key;
} QwenDeviceModel;

static SparkStatus QwenRefuse(SparkStatus status, const char *what)
{
	fprintf(stderr,"%s: %s\n",QD_TAG,what);
	return(status);
}

static uint64_t QwenAttentionRecordBytes(uint32_t rows, uint32_t positions)
{
	const uint64_t decode = (uint64_t)SPARK_OPS_GEMV_MAX_ROWS * ((positions + QD_DECODE_PARTITION_KEYS - 1u) / QD_DECODE_PARTITION_KEYS);
	const uint64_t prefill = (uint64_t)rows * ((positions + QD_PREFILL_PARTITION_KEYS - 1u) / QD_PREFILL_PARTITION_KEYS);
	return((decode > prefill ? decode : prefill) * QD_HEADS * SPARK_OPS_PARTIAL_FLOATS * 4u);
}

static uint64_t QwenScratchPlan(QwenDeviceModel *model, const SparkDeviceBuffer *buffer)
{
	const uint64_t rows = model->max_rows;
	const uint64_t sizes[] = { rows * QD_HIDDEN * 2u, rows * QD_HIDDEN * 2u, rows * QD_MIX * 2u, rows * QD_GATES * 2u,
		SparkOpsDeltaScratchBytes(model->max_rows,QD_KEY_HEADS,QD_VALUE_HEADS,QD_KEY_DIM), rows * QD_V * 2u, rows * QD_V * 2u,
		rows * QD_Q * 2u, rows * QD_KV * 2u, QwenAttentionRecordBytes(model->max_rows,model->max_positions), rows * 2u * QD_FFN * 2u,
		rows * QD_FFN * 2u };
	SparkTensor *targets[] = { &model->scratch.normed, &model->scratch.partial, &model->scratch.mix, &model->scratch.ba, &model->scratch.delta,
		&model->scratch.core, &model->scratch.gated, &model->scratch.query, &model->scratch.key, &model->scratch.records,
		&model->scratch.gate_up, &model->scratch.act };
	uint64_t cursor = 0u;
	uint32_t index;
	for ( index = 0u; index < sizeof(sizes) / sizeof(sizes[0]); index++ )
	{
		cursor = (cursor + 255u) / 256u * 256u;
		if ( buffer != 0 )
			*targets[index] = SparkTensorAt(buffer,cursor,sizes[index]);
		cursor += sizes[index];
	}
	return(cursor);
}

static uint64_t QwenRecurrentBytes(void)
{
	return((uint64_t)QD_GDN_LAYERS * (QD_STATE_BYTES + QD_WINDOW_BYTES));
}

static SparkStatus QwenEntry(QwenDeviceModel *model, const char *name, const char *kind, uint64_t *offset, uint64_t *bytes, uint32_t *rows)
{
	SparkNamedPackEntry entry;
	if ( SparkNamedPackEntryLoad(&model->pack,name,&entry) != SPARK_STATUS_OK || strcmp(entry.kind,kind) != 0 || entry.shape_count < 1u )
	{
		fprintf(stderr,"%s: pack tensor %s missing or not %s\n",QD_TAG,name,kind);
		return(SPARK_STATUS_PARSE_ERROR);
	}
	*offset = model->pack.payload_base + entry.payload_offset;
	*bytes = entry.bytes;
	if ( rows != 0 )
		*rows = entry.shape[0];
	return(SPARK_STATUS_OK);
}

static SparkStatus QwenConfig(QwenDeviceModel *model, const SparkStageRunnerConfiguration *configuration)
{
	uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, ffn_rows = 0u;
	if ( SparkNamedPackConfigU32(&model->pack,"tp_degree",&degree) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&model->pack,"tp_rank",&rank) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&model->pack,"hidden",&hidden) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&model->pack,"layers",&layers) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&model->pack,"vocab_rows",&vocab_rows) != SPARK_STATUS_OK ||
		SparkNamedPackConfigU32(&model->pack,"ffn_rows",&ffn_rows) != SPARK_STATUS_OK )
		return(QwenRefuse(SPARK_STATUS_PARSE_ERROR,"the pack configuration lacks its topology or geometry"));
	if ( degree != 1u || rank != 0u || degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != QD_HIDDEN ||
		layers != QD_LAYERS || vocab_rows != QD_VOCAB || ffn_rows != QD_FFN )
	{
		fprintf(stderr,"%s: pack is tp %u rank %u hidden %u layers %u vocab rows %u ffn rows %u; the device model is the whole TP1 Qwen3.8-27B\n",
			QD_TAG,degree,rank,hidden,layers,vocab_rows,ffn_rows);
		return(SPARK_STATUS_TARGET_MISMATCH);
	}
	return(SPARK_STATUS_OK);
}

static void QwenRopeParams(SparkHeadRopeParams *params, uint32_t heads, uint32_t input_offset, uint32_t head_stride)
{
	uint32_t pair;
	memset(params,0,sizeof(*params));
	params->heads = heads;
	params->head_dimension = QD_HEAD_DIM;
	params->input_stride = QD_QKV;
	params->input_offset = input_offset;
	params->input_head_stride = head_stride;
	params->output_stride = heads * QD_HEAD_DIM;
	params->rope_dimension = QD_ROPE;
	params->epsilon = QD_EPSILON;
	for ( pair = 0u; pair < QD_ROPE / 2u; pair++ )
		params->inverse_frequency[pair] = (float)(1.0 / pow(QD_THETA,(double)(2u * pair) / (double)QD_ROPE));
}

static SparkStatus QwenOpen(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
{
	const SparkStageRunnerConfiguration *configuration = request->configuration;
	QwenDeviceModel *model;
	uint32_t embed_rows = 0u, head_rows = 0u;
	uint64_t state_bytes;
	SparkStatus status;
	*out = 0;
	if ( configuration->stage_count != 1u || configuration->tp_degree != 1u || configuration->max_input_row_count == 0u ||
		configuration->state_budget_bytes == 0u || configuration->kv_pages_per_sequence == 0u || configuration->resident_sequence_capacity == 0u )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"the device model runs one TP1 stage with rows, sequences, KV pages and a state budget"));
	if ( configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 )
		return(QwenRefuse(SPARK_STATUS_UNSUPPORTED,"the device model reads the publisher's BF16 projections; this codec has no Metal kernels"));
	model = (QwenDeviceModel *)calloc(1u,sizeof(*model));
	if ( model == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*out = model;
	model->services = request->services;
	model->max_rows = configuration->max_input_row_count;
	model->slots = configuration->resident_sequence_capacity;
	model->pages_per_sequence = configuration->kv_pages_per_sequence;
	model->max_positions = configuration->kv_pages_per_sequence * QD_PAGE_SLOTS;
	status = SparkNamedPackOpen(configuration->rank_pack_path,SPARK_NAMED_PACK_MAGIC,SPARK_NAMED_PACK_VERSION,QD_PACK_ALIGNMENT,&model->pack);
	if ( status != SPARK_STATUS_OK )
		return(QwenRefuse(status,"cannot open the rank pack"));
	model->pack_open = 1u;
	status = QwenConfig(model,configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	model->scratch_bytes = QwenScratchPlan(model,0);
	state_bytes = (uint64_t)model->slots * QwenRecurrentBytes();
	if ( model->scratch_bytes + state_bytes > configuration->state_budget_bytes )
	{
		fprintf(stderr,"%s: state %llu bytes is over the %llu-byte budget (rows %u, slots %u)\n",QD_TAG,
			(unsigned long long)(model->scratch_bytes + state_bytes),(unsigned long long)configuration->state_budget_bytes,model->max_rows,model->slots);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkDeviceOpsAcquire(&model->ops);
	if ( status != SPARK_STATUS_OK )
		return(QwenRefuse(status,"the device kernel library is not available"));
	QwenRopeParams(&model->rope_query,QD_HEADS,0u,2u * QD_HEAD_DIM);
	QwenRopeParams(&model->rope_key,QD_KV_HEADS,2u * QD_Q,QD_HEAD_DIM);
	memset(geometry,0,sizeof(*geometry));
	geometry->hidden = QD_HIDDEN;
	geometry->vocab = QD_VOCAB;
	geometry->total_layers = QD_LAYERS;
	geometry->first_layer = 0u;
	geometry->layer_count = QD_LAYERS;
	geometry->rms_epsilon = QD_EPSILON;
	geometry->pack_bytes = model->pack.file_bytes;
	geometry->kv_layer_count = QD_ATTN_LAYERS;
	geometry->kv_layer_page_bytes = QD_PAGE_BYTES;
	geometry->kv_shard.degree = 1u;
	geometry->kv_shard.rank = 0u;
	geometry->kv_shard.grain = 1u;
	geometry->recurrent_bytes = QwenRecurrentBytes();
	geometry->head_norm_f32 = 1u;
	geometry->head_prenormed = 0u;
	model->pack_bytes = model->pack.file_bytes;
	status = QwenEntry(model,"embed","bf16",&geometry->embed_offset,&geometry->embed_bytes,&embed_rows);
	if ( status == SPARK_STATUS_OK )
		status = QwenEntry(model,"head_norm","f32",&geometry->head_norm_offset,&geometry->head_norm_bytes,0);
	if ( status == SPARK_STATUS_OK )
		status = QwenEntry(model,"head","bf16",&geometry->head_offset,&geometry->head_bytes,&head_rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( embed_rows != QD_VOCAB || head_rows != QD_VOCAB || geometry->embed_bytes != (uint64_t)QD_VOCAB * QD_HIDDEN * 2u ||
		geometry->head_bytes != (uint64_t)QD_VOCAB * QD_HIDDEN * 2u || geometry->head_norm_bytes != QD_HIDDEN * 4u )
		return(QwenRefuse(SPARK_STATUS_SCHEMA_ERROR,"the pack's embedding, head or head norm has the wrong shape"));
	geometry->embed_rows = embed_rows;
	geometry->head_rows = head_rows;
	return(SPARK_STATUS_OK);
}

static SparkStatus QwenManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
{
	(void)model;
	if ( manifest == 0 || manifest->range_count != 0u )
		return(QwenRefuse(SPARK_STATUS_SCHEMA_ERROR,"a dense pack's weightd manifest carries no expert ranges"));
	return(SPARK_STATUS_OK);
}

static SparkStatus QwenBindWeightd(void *model, struct SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
{
	(void)model;
	(void)lazy_pack;
	(void)geometry;
	return(QwenRefuse(SPARK_STATUS_UNSUPPORTED,"weightd lazy packs are CUDA arenas; the device model binds the device runner's mapping through bind_device"));
}

static SparkStatus QwenTensor(QwenDeviceModel *model, const SparkStageRunnerDeviceWeights *weights, const char *name, const char *kind,
	uint64_t expected_bytes, SparkTensor *tensor)
{
	SparkDeviceRange range;
	uint64_t offset = 0u, bytes = 0u;
	SparkStatus status = QwenEntry(model,name,kind,&offset,&bytes,0);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( bytes != expected_bytes || offset + bytes > weights->pack_bytes || offset % 16u != 0u )
	{
		fprintf(stderr,"%s: pack tensor %s holds %llu bytes at %llu; the model needs %llu aligned bytes inside the pack\n",QD_TAG,name,
			(unsigned long long)bytes,(unsigned long long)offset,(unsigned long long)expected_bytes);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	status = SparkDeviceResolve(weights->pack_address + offset,bytes,&range);
	if ( status != SPARK_STATUS_OK )
		return(QwenRefuse(status,"the pack mapping does not hold a tensor's bytes"));
	tensor->range = range;
	tensor->bytes = bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus QwenLayerTensor(QwenDeviceModel *model, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *name,
	const char *kind, uint64_t bytes, SparkTensor *tensor)
{
	char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
	snprintf(key,sizeof(key),"layers.%u.%s",layer,name);
	return(QwenTensor(model,weights,key,kind,bytes,tensor));
}

static SparkStatus QwenBindLayer(QwenDeviceModel *model, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
{
	QwenDeviceLayer *layer = &model->layers[index];
	const SparkStageRunnerDeviceWeights *w = weights;
	SparkStatus status;
	layer->attention = index % QD_PERIOD == QD_PHASE ? 1u : 0u;
	status = QwenLayerTensor(model,w,index,"input_norm","f32",QD_HIDDEN * 4u,&layer->input_norm);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"post_norm","f32",QD_HIDDEN * 4u,&layer->post_norm);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"ffn_gate_up","bf16",(uint64_t)2u * QD_FFN * QD_HIDDEN * 2u,&layer->gate_up);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"ffn_down","bf16",(uint64_t)QD_HIDDEN * QD_FFN * 2u,&layer->down);
	if ( status != SPARK_STATUS_OK || layer->attention != 0u )
	{
		if ( status == SPARK_STATUS_OK )
			status = QwenLayerTensor(model,w,index,"attn_qkv","bf16",(uint64_t)QD_QKV * QD_HIDDEN * 2u,&layer->mix);
		if ( status == SPARK_STATUS_OK )
			status = QwenLayerTensor(model,w,index,"attn_out","bf16",(uint64_t)QD_HIDDEN * QD_Q * 2u,&layer->out);
		if ( status == SPARK_STATUS_OK )
			status = QwenLayerTensor(model,w,index,"attn_q_norm","f32",QD_HEAD_DIM * 4u,&layer->q_norm);
		if ( status == SPARK_STATUS_OK )
			status = QwenLayerTensor(model,w,index,"attn_k_norm","f32",QD_HEAD_DIM * 4u,&layer->k_norm);
		return(status);
	}
	status = QwenLayerTensor(model,w,index,"gdn_qkvz","bf16",(uint64_t)QD_MIX * QD_HIDDEN * 2u,&layer->mix);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"gdn_out","bf16",(uint64_t)QD_HIDDEN * QD_V * 2u,&layer->out);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"gdn_ba","bf16",(uint64_t)QD_GATES * QD_HIDDEN * 2u,&layer->ba);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"gdn_conv","bf16",(uint64_t)QD_CHANNELS * QD_CONV * 2u,&layer->conv);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"gdn_a_log","f32",QD_VALUE_HEADS * 4u,&layer->a_log);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"gdn_dt_bias","f32",QD_VALUE_HEADS * 4u,&layer->dt_bias);
	if ( status == SPARK_STATUS_OK )
		status = QwenLayerTensor(model,w,index,"gdn_norm","bf16",QD_VALUE_DIM * 2u,&layer->gdn_norm);
	return(status);
}

static SparkStatus QwenBindDevice(void *opaque, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	uint32_t index, recurrent = 0u, attention = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	(void)geometry;
	if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != model->pack_bytes )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"bind_device needs the device mapping of the whole rank pack"));
	for ( index = 0u; index < QD_LAYERS && status == SPARK_STATUS_OK; index++ )
	{
		status = QwenBindLayer(model,weights,index);
		model->layers[index].index = model->layers[index].attention != 0u ? attention++ : recurrent++;
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDeviceAllocate(&model->scratch_buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,model->scratch_bytes,"qwen38 device scratch");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAllocate(&model->state,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,(uint64_t)QD_GDN_LAYERS * model->slots * QD_STATE_BYTES,
			"qwen38 recurrent state");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAllocate(&model->window,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,(uint64_t)QD_GDN_LAYERS * model->slots * QD_WINDOW_BYTES,
			"qwen38 convolution window");
	if ( status != SPARK_STATUS_OK )
		return(QwenRefuse(status,"cannot allocate the device scratch and recurrent state"));
	QwenScratchPlan(model,&model->scratch_buffer);
	return(SPARK_STATUS_OK);
}

static void QwenClose(void *opaque)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	if ( model == 0 )
		return;
	if ( model->scratch_buffer.handle != 0u )
		(void)SparkDeviceFree(&model->scratch_buffer);
	if ( model->state.handle != 0u )
		(void)SparkDeviceFree(&model->state);
	if ( model->window.handle != 0u )
		(void)SparkDeviceFree(&model->window);
	if ( model->ops != 0 )
		SparkDeviceOpsRelease(model->ops);
	if ( model->pack_open != 0u )
		SparkNamedPackClose(&model->pack);
	free(model);
}

static SparkStatus QwenResolve(const void *address, uint64_t bytes, SparkTensor *tensor)
{
	SparkDeviceRange range;
	SparkStatus status = SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)address,bytes,&range);
	if ( status != SPARK_STATUS_OK )
		return(QwenRefuse(status,"a step table is not device memory of the expected size"));
	tensor->range = range;
	tensor->bytes = bytes;
	return(SPARK_STATUS_OK);
}

typedef struct QwenStepTensors
{
	SparkTensor hidden;
	SparkTensor positions;
	SparkTensor sequence_of_row;
	SparkOpsSequences sequences;
	const uint32_t *host_positions;
	const uint32_t *host_sequence_of_row;
	uint32_t rows;
} QwenStepTensors;

static SparkStatus QwenLinear(QwenDeviceModel *model, SparkDeviceQueue *queue, SparkTensor x, uint32_t x_stride, SparkTensor w, SparkTensor y,
	uint32_t y_stride, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t gated)
{
	SparkGemvParams params;
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = input_dimension;
	params.output_dimension = output_dimension;
	params.input_stride = x_stride;
	params.output_stride = y_stride;
	params.gated_input = gated;
	return(SparkOpsLinear(model->ops,queue,x,w,y,&params));
}

static SparkStatus QwenNorm(QwenDeviceModel *model, SparkDeviceQueue *queue, const QwenStepTensors *t, const SparkTensor *addend, SparkTensor weight)
{
	SparkNormParams params;
	memset(&params,0,sizeof(params));
	params.rows = t->rows;
	params.dimension = QD_HIDDEN;
	params.input_stride = params.output_stride = QD_HIDDEN;
	params.has_addend = addend != 0 ? 1u : 0u;
	params.write_sum = addend != 0 ? 1u : 0u;
	params.epsilon = QD_EPSILON;
	return(SparkOpsRmsNormResidual(model->ops,queue,t->hidden,addend,weight,model->scratch.normed,&params));
}

static SparkStatus QwenGdn(QwenDeviceModel *model, SparkDeviceQueue *queue, const QwenStepTensors *t, const QwenDeviceLayer *layer)
{
	const QwenDeviceScratch *s = &model->scratch;
	const uint32_t rows = t->rows;
	SparkConvParams conv;
	SparkDeltaParams delta;
	SparkGatedNormParams gate;
	SparkTensor state = SparkTensorAt(&model->state,(uint64_t)layer->index * model->slots * QD_STATE_BYTES,(uint64_t)model->slots * QD_STATE_BYTES);
	SparkTensor window = SparkTensorAt(&model->window,(uint64_t)layer->index * model->slots * QD_WINDOW_BYTES,(uint64_t)model->slots * QD_WINDOW_BYTES);
	SparkStatus status = QwenLinear(model,queue,s->normed,QD_HIDDEN,layer->mix,s->mix,QD_MIX,rows,QD_HIDDEN,QD_MIX,0u);
	if ( status == SPARK_STATUS_OK )
		status = QwenLinear(model,queue,s->normed,QD_HIDDEN,layer->ba,s->ba,QD_GATES,rows,QD_HIDDEN,QD_GATES,0u);
	memset(&conv,0,sizeof(conv));
	conv.channels = QD_CHANNELS;
	conv.row_stride = QD_MIX;
	conv.sequences = t->sequences.sequences;
	conv.kernel_taps = QD_CONV;
	conv.commit = t->sequences.commit;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsCausalConv(model->ops,queue,window,&t->sequences,s->mix,layer->conv,s->mix,&conv);
	memset(&delta,0,sizeof(delta));
	delta.rows = rows;
	delta.sequences = t->sequences.sequences;
	delta.key_heads = QD_KEY_HEADS;
	delta.value_heads = QD_VALUE_HEADS;
	delta.key_dimension = QD_KEY_DIM;
	delta.value_dimension = QD_VALUE_DIM;
	delta.mix_stride = QD_MIX;
	delta.query_offset = 0u;
	delta.key_offset = QD_QK;
	delta.value_offset = 2u * QD_QK;
	delta.gate_stride = QD_GATES;
	delta.beta_offset = 0u;
	delta.decay_offset = QD_VALUE_HEADS;
	delta.output_stride = QD_V;
	delta.commit = t->sequences.commit;
	delta.epsilon = QD_EPSILON;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsGatedDelta(model->ops,queue,state,&t->sequences,s->mix,s->ba,layer->a_log,layer->dt_bias,s->delta,s->core,&delta);
	memset(&gate,0,sizeof(gate));
	gate.rows = rows;
	gate.heads = QD_VALUE_HEADS;
	gate.head_dimension = QD_VALUE_DIM;
	gate.input_stride = QD_V;
	gate.gate_stride = QD_MIX;
	gate.gate_offset = QD_CHANNELS;
	gate.output_stride = QD_V;
	gate.epsilon = QD_EPSILON;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormGate(model->ops,queue,s->core,s->mix,layer->gdn_norm,s->gated,&gate);
	if ( status == SPARK_STATUS_OK )
		status = QwenLinear(model,queue,s->gated,QD_V,layer->out,s->partial,QD_HIDDEN,rows,QD_V,QD_HIDDEN,0u);
	return(status);
}

static SparkOpsPaged QwenPaged(const QwenDeviceModel *model, const QwenStepTensors *t, uint32_t kv_layer)
{
	SparkOpsPaged paged;
	paged.pool = SparkTensorSlice(model->kv_pool,(uint64_t)kv_layer * model->kv_layer_stride,model->kv_layer_stride);
	paged.page_table = model->page_table;
	paged.sequence_of_row = t->sequence_of_row;
	paged.positions = t->positions;
	paged.page_slots = QD_PAGE_SLOTS;
	paged.pages_per_sequence = model->pages_per_sequence;
	paged.pool_pages = model->pool_pages;
	paged.kv_heads = QD_KV_HEADS;
	return(paged);
}

static SparkStatus QwenAttention(QwenDeviceModel *model, SparkDeviceQueue *queue, const QwenStepTensors *t, const QwenDeviceLayer *layer)
{
	const QwenDeviceScratch *s = &model->scratch;
	const uint32_t rows = t->rows;
	const SparkOpsPaged paged = QwenPaged(model,t,layer->index);
	SparkHeadRopeParams query = model->rope_query, key = model->rope_key;
	SparkKvParams store;
	SparkAttentionParams attention;
	SparkStatus status = QwenLinear(model,queue,s->normed,QD_HIDDEN,layer->mix,s->mix,QD_QKV,rows,QD_HIDDEN,QD_QKV,0u);
	query.rows = key.rows = rows;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(model->ops,queue,s->mix,layer->q_norm,t->positions,s->query,&query);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(model->ops,queue,s->mix,layer->k_norm,t->positions,s->key,&key);
	memset(&store,0,sizeof(store));
	store.rows = rows;
	store.kv_heads = QD_KV_HEADS;
	store.head_dimension = QD_HEAD_DIM;
	store.page_slots = QD_PAGE_SLOTS;
	store.pages_per_sequence = model->pages_per_sequence;
	store.key_stride = QD_KV;
	store.value_stride = QD_QKV;
	store.value_offset = 2u * QD_Q + QD_KV;
	store.pool_pages = model->pool_pages;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsKvStore(model->ops,queue,&paged,s->key,s->mix,&store);
	memset(&attention,0,sizeof(attention));
	attention.rows = rows;
	attention.heads = QD_HEADS;
	attention.kv_heads = QD_KV_HEADS;
	attention.head_dimension = QD_HEAD_DIM;
	attention.page_slots = QD_PAGE_SLOTS;
	attention.pages_per_sequence = model->pages_per_sequence;
	attention.pool_pages = model->pool_pages;
	attention.query_stride = QD_Q;
	attention.gate_stride = QD_QKV;
	attention.gate_offset = QD_HEAD_DIM;
	attention.gate_head_stride = 2u * QD_HEAD_DIM;
	attention.output_stride = QD_Q;
	attention.scale = 1.0f / 16.0f;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAttention(model->ops,queue,&paged,s->query,s->mix,s->records,s->gated,&attention,t->host_positions,t->host_sequence_of_row);
	if ( status == SPARK_STATUS_OK )
		status = QwenLinear(model,queue,s->gated,QD_Q,layer->out,s->partial,QD_HIDDEN,rows,QD_Q,QD_HIDDEN,0u);
	return(status);
}

static SparkStatus QwenFfn(QwenDeviceModel *model, SparkDeviceQueue *queue, const QwenStepTensors *t, const QwenDeviceLayer *layer)
{
	const QwenDeviceScratch *s = &model->scratch;
	const uint32_t rows = t->rows;
	SparkSiluMulParams silu;
	SparkStatus status = QwenLinear(model,queue,s->normed,QD_HIDDEN,layer->gate_up,s->gate_up,2u * QD_FFN,rows,QD_HIDDEN,2u * QD_FFN,0u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( rows <= SPARK_OPS_GEMV_MAX_ROWS )
		return(QwenLinear(model,queue,s->gate_up,2u * QD_FFN,layer->down,s->partial,QD_HIDDEN,rows,QD_FFN,QD_HIDDEN,1u));
	silu.rows = rows;
	silu.dimension = QD_FFN;
	silu.input_stride = 2u * QD_FFN;
	silu.output_stride = QD_FFN;
	status = SparkOpsSiluMul(model->ops,queue,s->gate_up,s->act,&silu);
	if ( status == SPARK_STATUS_OK )
		status = QwenLinear(model,queue,s->act,QD_FFN,layer->down,s->partial,QD_HIDDEN,rows,QD_FFN,QD_HIDDEN,0u);
	return(status);
}

static SparkStatus QwenStepTables(QwenDeviceModel *model, const SparkStageRunnerStep *step, QwenStepTensors *t)
{
	SparkStatus status;
	uint32_t ordinal, groups;
	if ( step->rows == 0u || step->rows > model->max_rows || step->sequences == 0u || step->sequences > model->slots || model->kv_attached == 0u ||
		step->host_positions == 0 || step->host_sequence_of_row == 0 || step->recurrent_index == 0 || step->sequence_row_begin == 0 )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"a step needs rows and sequences within capacity, an attached KV pool and its row tables"));
	for ( ordinal = 0u, groups = 0u; ordinal < step->rows; ordinal++ )
	{
		if ( step->host_sequence_of_row[ordinal] >= model->slots )
			return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"a row names a sequence outside the model's slots"));
		if ( ordinal == 0u || step->host_sequence_of_row[ordinal] != step->host_sequence_of_row[ordinal - 1u] )
			groups++;
	}
	if ( groups != step->sequences )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"each step sequence must own one contiguous run of rows"));
	t->rows = step->rows;
	t->host_positions = step->host_positions;
	t->host_sequence_of_row = step->host_sequence_of_row;
	status = QwenResolve(step->hidden_bf16,(uint64_t)step->rows * QD_HIDDEN * 2u,&t->hidden);
	if ( status == SPARK_STATUS_OK )
		status = QwenResolve(step->positions,(uint64_t)step->rows * 4u,&t->positions);
	if ( status == SPARK_STATUS_OK )
		status = QwenResolve(step->sequence_of_row,(uint64_t)step->rows * 4u,&t->sequence_of_row);
	if ( status == SPARK_STATUS_OK )
		status = QwenResolve(step->recurrent_index,(uint64_t)step->sequences * 4u,&t->sequences.state_index);
	if ( status == SPARK_STATUS_OK )
		status = QwenResolve(step->sequence_row_begin,((uint64_t)step->sequences + 1u) * 4u,&t->sequences.row_begin);
	t->sequences.sequences = step->sequences;
	t->sequences.commit = step->commit;
	return(status);
}

static SparkStatus QwenStep(void *opaque, const SparkStageRunnerStep *step, void *device_queue)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	QwenStepTensors t;
	uint32_t index;
	SparkStatus status = QwenStepTables(model,step,&t);
	for ( index = 0u; index < QD_LAYERS && status == SPARK_STATUS_OK; index++ )
	{
		const QwenDeviceLayer *layer = &model->layers[index];
		status = QwenNorm(model,queue,&t,index == 0u ? 0 : &model->scratch.partial,layer->input_norm);
		if ( status == SPARK_STATUS_OK )
			status = layer->attention != 0u ? QwenAttention(model,queue,&t,layer) : QwenGdn(model,queue,&t,layer);
		if ( status == SPARK_STATUS_OK )
			status = QwenNorm(model,queue,&t,&model->scratch.partial,layer->post_norm);
		if ( status == SPARK_STATUS_OK )
			status = QwenFfn(model,queue,&t,layer);
		if ( status == SPARK_STATUS_OK && model->services != 0 && model->services->layer_done != 0 )
			model->services->layer_done(model->services->context,index);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAddRows(model->ops,queue,t.hidden,model->scratch.partial,t.rows,QD_HIDDEN,QD_HIDDEN,QD_HIDDEN);
	return(status);
}

static SparkStatus QwenSideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *device_queue)
{
	(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)device_queue;
	return(QwenRefuse(SPARK_STATUS_UNSUPPORTED,"the dense hybrid model has no sideband"));
}

static SparkStatus QwenAttachKv(void *opaque, const SparkStageRunnerKv *kv)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkStatus status;
	if ( kv == 0 || kv->layer_count != QD_ATTN_LAYERS || kv->layer_page_bytes != QD_PAGE_BYTES || kv->sequence_count > model->slots ||
		kv->second_pool != 0 || kv->page_table_stride != model->pages_per_sequence || kv->pool_page_count == 0u ||
		kv->layer_stride_bytes < (uint64_t)kv->pool_page_count * QD_PAGE_BYTES || kv->context_shard.degree != 1u )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"the KV pool must hold 16 layers of 64-slot pages for the model's slots, unsharded"));
	status = QwenResolve(kv->pool,(uint64_t)kv->layer_count * kv->layer_stride_bytes,&model->kv_pool);
	if ( status == SPARK_STATUS_OK )
		status = QwenResolve(kv->page_table,(uint64_t)kv->sequence_count * kv->page_table_stride * 4u,&model->page_table);
	if ( status != SPARK_STATUS_OK )
		return(status);
	model->kv_layer_stride = kv->layer_stride_bytes;
	model->pool_pages = kv->pool_page_count;
	model->kv_attached = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus QwenRecurrentCopy(void *opaque, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *device_queue)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	SparkTensor packed;
	uint64_t cursor = 0u;
	uint32_t part, layer;
	SparkStatus status;
	if ( slot >= model->slots || bytes != QwenRecurrentBytes() || queue == 0 )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"recurrent copy needs a slot, the model's recurrent size and a queue"));
	status = QwenResolve(buffer,bytes,&packed);
	for ( part = 0u; part < 2u && status == SPARK_STATUS_OK; part++ )
	{
		const SparkDeviceBuffer *pool = part == 0u ? &model->state : &model->window;
		const uint64_t width = part == 0u ? QD_STATE_BYTES : QD_WINDOW_BYTES;
		for ( layer = 0u; layer < QD_GDN_LAYERS && status == SPARK_STATUS_OK; layer++ )
		{
			SparkDeviceRange resident;
			resident.buffer = pool;
			resident.offset = ((uint64_t)layer * model->slots + slot) * width;
			status = to_buffer != 0u ? SparkDeviceCopy(queue,(SparkDeviceRange){ packed.range.buffer, packed.range.offset + cursor },resident,width) :
				SparkDeviceCopy(queue,resident,(SparkDeviceRange){ packed.range.buffer, packed.range.offset + cursor },width);
			cursor += width;
		}
	}
	return(status);
}

static SparkStatus QwenResetSlot(void *opaque, uint32_t slot, void *device_queue)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	uint32_t layer;
	SparkStatus status = SPARK_STATUS_OK;
	if ( slot >= model->slots || queue == 0 )
		return(QwenRefuse(SPARK_STATUS_INVALID_ARGUMENT,"reset needs a slot and a queue"));
	for ( layer = 0u; layer < QD_GDN_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		SparkDeviceRange state, window;
		state.buffer = &model->state;
		state.offset = ((uint64_t)layer * model->slots + slot) * QD_STATE_BYTES;
		window.buffer = &model->window;
		window.offset = ((uint64_t)layer * model->slots + slot) * QD_WINDOW_BYTES;
		status = SparkDeviceFill(queue,state,0u,QD_STATE_BYTES);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFill(queue,window,0u,QD_WINDOW_BYTES);
	}
	return(status);
}

static const void *QwenProbeBuffers(void *model)
{
	return(model);
}

static void QwenReport(void *opaque, uint32_t rank)
{
	const QwenDeviceModel *model = (const QwenDeviceModel *)opaque;
	fprintf(stderr,"QWEN38_27B-DEVICE rank=%u scratch_bytes=%llu recurrent_bytes=%llu slots=%u rows=%u positions=%u\n",rank,
		(unsigned long long)model->scratch_bytes,(unsigned long long)((uint64_t)model->slots * QwenRecurrentBytes()),model->slots,model->max_rows,
		model->max_positions);
}

static SparkStatus QwenLane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services, void **lane_model)
{
	(void)model; (void)lane; (void)step; (void)services; (void)lane_model;
	return(QwenRefuse(SPARK_STATUS_UNSUPPORTED,"the device runner has one execution lane"));
}

const SparkStageRunnerModelInterface *SparkQwen38_27bDeviceStageModel(void)
{
	static const SparkStageRunnerModelInterface interface_table =
	{
		SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
		QD_TAG,
		"qwen3.8-27b",
		"official",
		QwenOpen,
		QwenManifestCheck,
		QwenBindWeightd,
		QwenClose,
		QwenStep,
		QwenSideband,
		QwenAttachKv,
		QwenRecurrentCopy,
		QwenResetSlot,
		QwenProbeBuffers,
		QwenReport,
		0,
		QwenLane,
		0,
		0,
		QwenBindDevice
	};
	return(&interface_table);
}
