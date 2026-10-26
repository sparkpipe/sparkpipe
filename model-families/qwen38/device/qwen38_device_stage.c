#include <stdlib.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_qwen38_device_stage.h"

#define QD_TAG "sparkpipe_qwen38_27b_device"
#define QD_LAYERS 64u
#define QD_PERIOD 4u
#define QD_PHASE 3u
#define QD_GDN_LAYERS 48u
#define QD_FFN 17408u

static const SparkDeviceStageShape qwen_shape =
{
	.tag = QD_TAG,
	.hidden = 5120u,
	.vocab = 248320u,
	.layers = QD_LAYERS,
	.kv_layers = 16u,
	.kv_heads = 4u,
	.kv_head_dimension = 256u,
	.pack_alignment = 128u,
	.epsilon = 1e-6f
};

static const SparkDeviceGatedDeltaShape qwen_delta =
{
	.key_heads = 16u,
	.value_heads = 48u,
	.key_dimension = 128u,
	.value_dimension = 128u,
	.conv_taps = 4u,
	.epsilon = 1e-6f
};

static const SparkDeviceAttentionShape qwen_attention =
{
	.heads = 24u,
	.kv_heads = 4u,
	.head_dimension = 256u,
	.rope_dimension = 64u,
	.rope_theta = 10000000.0,
	.epsilon = 1e-6f,
	.scale = 1.0f / 16.0f,
	.output_gate = 1u
};

typedef struct QwenDeviceLayer
{
	SparkTensor input_norm;
	SparkTensor post_norm;
	SparkDeviceFfnWeights ffn;
	SparkDeviceGatedDeltaWeights delta;
	SparkDeviceAttentionWeights attention;
	uint32_t is_attention;
	uint32_t index;
} QwenDeviceLayer;

typedef struct QwenDeviceModel
{
	SparkDeviceStage stage;
	QwenDeviceLayer layers[QD_LAYERS];
	SparkTensor normed;
	SparkTensor partial;
	SparkDeviceGatedDeltaScratch delta;
	SparkDeviceAttentionScratch attention;
	SparkDeviceFfnScratch ffn;
	SparkDeviceAttentionRope rope;
} QwenDeviceModel;

static uint32_t QwenMixWidth(void)
{
	const uint32_t delta = SparkDeviceGatedDeltaMix(&qwen_delta), attention = SparkDeviceAttentionFusedWidth(&qwen_attention);
	return(delta > attention ? delta : attention);
}

static uint32_t QwenGatedWidth(void)
{
	const uint32_t delta = SparkDeviceGatedDeltaValues(&qwen_delta), attention = SparkDeviceAttentionQueryWidth(&qwen_attention);
	return(delta > attention ? delta : attention);
}

static uint32_t QwenScratchSlices(QwenDeviceModel *model, SparkDeviceStageSlice *slices)
{
	const uint64_t rows = model->stage.max_rows, hidden = qwen_shape.hidden;
	const SparkDeviceStageSlice plan[] =
	{
		{ rows * hidden * 2u, &model->normed },
		{ rows * hidden * 2u, &model->partial },
		{ rows * QwenMixWidth() * 2u, &model->delta.mix },
		{ rows * SparkDeviceGatedDeltaGates(&qwen_delta) * 2u, &model->delta.gates },
		{ SparkOpsDeltaScratchBytes(model->stage.max_rows,qwen_delta.key_heads,qwen_delta.value_heads,qwen_delta.key_dimension), &model->delta.delta },
		{ rows * SparkDeviceGatedDeltaValues(&qwen_delta) * 2u, &model->delta.core },
		{ rows * QwenGatedWidth() * 2u, &model->delta.gated },
		{ rows * SparkDeviceAttentionQueryWidth(&qwen_attention) * 2u, &model->attention.query },
		{ rows * SparkDeviceAttentionKvWidth(&qwen_attention) * 2u, &model->attention.key },
		{ SparkOpsAttentionScratchBytes(model->stage.max_rows,qwen_attention.heads,model->stage.max_positions), &model->attention.records },
		{ rows * 2u * QD_FFN * 2u, &model->ffn.gate_up },
		{ rows * QD_FFN * 2u, &model->ffn.act },
		{ SparkDeviceGatedDeltaKeepBytes(&qwen_delta,model->stage.verify_rows,QD_GDN_LAYERS), &model->delta.keep }
	};
	const uint32_t count = sizeof(plan) / sizeof(plan[0]);
	uint32_t index;
	for ( index = 0u; index < count; index++ )
		slices[index] = plan[index];
	return(count);
}

static SparkStatus QwenOpen(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
{
	const SparkDeviceStageConfigValue config[] =
	{
		{ "hidden", qwen_shape.hidden }, { "layers", QD_LAYERS }, { "vocab_rows", qwen_shape.vocab }, { "ffn_rows", QD_FFN }
	};
	const uint64_t widths[SPARK_DEVICE_GATED_DELTA_POOLS] = { SparkDeviceGatedDeltaStateBytes(&qwen_delta), SparkDeviceGatedDeltaWindowBytes(&qwen_delta) };
	SparkDeviceStageSlice slices[16];
	QwenDeviceModel *model = (QwenDeviceModel *)calloc(1u,sizeof(*model));
	SparkStatus status;
	*out = model;
	if ( model == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkDeviceStageOpen(&model->stage,&qwen_shape,request);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageExpectConfig(&model->stage,config,sizeof(config) / sizeof(config[0]));
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageReserve(&model->stage,SparkDeviceStagePlan(0,slices,QwenScratchSlices(model,slices)),QD_GDN_LAYERS,widths,
			SPARK_DEVICE_GATED_DELTA_POOLS,request->configuration->state_budget_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkDeviceAttentionRopePrepare(&qwen_attention,&model->rope);
	return(SparkDeviceStageGeometry(&model->stage,geometry));
}

static SparkStatus QwenBindLayer(QwenDeviceModel *model, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
{
	QwenDeviceLayer *layer = &model->layers[index];
	SparkStatus status;
	layer->is_attention = index % QD_PERIOD == QD_PHASE ? 1u : 0u;
	status = SparkDeviceStageLayerTensor(&model->stage,weights,index,"input_norm","f32",qwen_shape.hidden * 4u,&layer->input_norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(&model->stage,weights,index,"post_norm","f32",qwen_shape.hidden * 4u,&layer->post_norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceFfnBind(&model->stage,weights,index,QD_FFN,&layer->ffn);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( layer->is_attention != 0u )
		return(SparkDeviceAttentionBind(&model->stage,weights,index,&qwen_attention,&layer->attention));
	return(SparkDeviceGatedDeltaBind(&model->stage,weights,index,&qwen_delta,&layer->delta));
}

static SparkStatus QwenBindDevice(void *opaque, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkDeviceStageSlice slices[16];
	uint32_t index, recurrent = 0u, attention = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	(void)geometry;
	if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != model->stage.pack_bytes )
		return(SparkDeviceStageRefuse(&model->stage,SPARK_STATUS_INVALID_ARGUMENT,"bind_device needs the device mapping of the whole rank pack"));
	for ( index = 0u; index < QD_LAYERS && status == SPARK_STATUS_OK; index++ )
	{
		status = QwenBindLayer(model,weights,index);
		model->layers[index].index = model->layers[index].is_attention != 0u ? attention++ : recurrent++;
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDeviceStageAllocate(&model->stage,slices,QwenScratchSlices(model,slices));
	model->attention.qkv = model->delta.mix;
	model->attention.gated = model->delta.gated;
	return(status);
}

static SparkStatus QwenStep(void *opaque, const SparkStageRunnerStep *step, void *device_queue)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkDeviceStage *stage = &model->stage;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	SparkDeviceStepTensors t;
	uint32_t index;
	SparkStatus status = SparkDeviceStageStepTables(stage,step,&t);
	for ( index = 0u; index < QD_LAYERS && status == SPARK_STATUS_OK; index++ )
	{
		const QwenDeviceLayer *layer = &model->layers[index];
		status = SparkDeviceResidualNorm(stage,queue,&t,index == 0u ? 0 : &model->partial,layer->input_norm,model->normed);
		if ( status == SPARK_STATUS_OK )
			status = layer->is_attention != 0u ?
				SparkDeviceAttention(stage,queue,&qwen_attention,&model->rope,&t,&layer->attention,layer->index,&model->attention,model->normed,model->partial) :
				SparkDeviceGatedDelta(stage,queue,&qwen_delta,&t,&layer->delta,layer->index,&model->delta,model->normed,model->partial);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceResidualNorm(stage,queue,&t,&model->partial,layer->post_norm,model->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(stage,queue,QD_FFN,&t,&layer->ffn,&model->ffn,model->normed,model->partial);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageLayerDone(stage,index);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAddRows(stage->ops,queue,t.hidden,model->partial,t.rows,qwen_shape.hidden,qwen_shape.hidden,qwen_shape.hidden);
	return(status);
}

static SparkStatus QwenKeepRows(void *opaque, const SparkStageRunnerStep *step, const uint32_t *kept, void *device_queue)
{
	QwenDeviceModel *model = (QwenDeviceModel *)opaque;
	SparkDeviceKeepTensors keep;
	uint32_t index;
	SparkStatus status = SparkDeviceStageKeepTables(&model->stage,step,kept,&keep);
	for ( index = 0u; index < QD_LAYERS && status == SPARK_STATUS_OK; index++ )
		if ( model->layers[index].is_attention == 0u )
			status = SparkDeviceGatedDeltaKeep(&model->stage,(SparkDeviceQueue *)device_queue,&qwen_delta,&keep,&model->layers[index].delta,
				model->layers[index].index,&model->delta);
	return(status);
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
		SparkDeviceStageManifestCheck,
		SparkDeviceStageBindWeightd,
		SparkDeviceStageClose,
		QwenStep,
		SparkDeviceStageSideband,
		SparkDeviceStageAttachKv,
		SparkDeviceStageRecurrentCopy,
		SparkDeviceStageResetSlot,
		SparkDeviceStageProbeBuffers,
		SparkDeviceStageReport,
		0,
		SparkDeviceStageLane,
		QwenKeepRows,
		0,
		QwenBindDevice
	};
	return(&interface_table);
}
