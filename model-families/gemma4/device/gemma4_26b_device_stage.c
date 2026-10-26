#include <stdlib.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_gemma4_26b_device_stage.h"

#define GD_TAG "sparkpipe_gemma4_26b_device"
#define GD_LAYERS 30u
#define GD_PERIOD 6u
#define GD_PHASE 5u
#define GD_FFN 2112u
#define GD_EMBED_SCALE 53.0f
#define GD_SLICES 24u

static const SparkDeviceStageShape gemma_shape =
{
	.tag = GD_TAG,
	.hidden = 2816u,
	.vocab = 262144u,
	.layers = GD_LAYERS,
	.kv_layers = GD_LAYERS,
	.kv_heads = 8u,
	.kv_head_dimension = 256u,
	.pack_alignment = 128u,
	.epsilon = 1e-6f,
	.tied_head = 1u,
	.logit_softcap = 30.0f
};

static const SparkDeviceAttentionShape gemma_sliding =
{
	.heads = 16u,
	.kv_heads = 8u,
	.head_dimension = 256u,
	.rope_dimension = 256u,
	.rope_theta = 10000.0,
	.epsilon = 1e-6f,
	.scale = 1.0f,
	.value_norm = 1u,
	.window = 1024u,
	.rope_rounded_terms = 1u
};

static const SparkDeviceAttentionShape gemma_full =
{
	.heads = 16u,
	.kv_heads = 2u,
	.head_dimension = 512u,
	.rope_dimension = 512u,
	.rope_theta = 1000000.0,
	.epsilon = 1e-6f,
	.scale = 1.0f,
	.key_is_value = 1u,
	.value_norm = 1u,
	.rope_pairs = 64u,
	.rope_rounded_terms = 1u
};

static const SparkDeviceExpertsShape gemma_experts =
{
	.experts = 128u,
	.top_k = 8u,
	.intermediate = 704u,
	.activation = SPARK_GATED_GELU_TANH,
	.epsilon = 1e-6f
};

typedef struct GemmaDeviceLayer
{
	SparkTensor input_norm;
	SparkTensor post_attn_norm;
	SparkTensor pre_ff_norm;
	SparkTensor post_ff_norm;
	SparkTensor post_ff_norm_1;
	SparkTensor post_ff_norm_2;
	SparkTensor layer_scalar;
	SparkDeviceAttentionWeights attention;
	SparkDeviceFfnWeights ffn;
	SparkDeviceExpertsWeights experts;
	uint32_t full;
} GemmaDeviceLayer;

typedef struct GemmaDeviceModel
{
	SparkDeviceStage stage;
	GemmaDeviceLayer layers[GD_LAYERS];
	SparkTensor normed;
	SparkTensor partial;
	SparkTensor expert_out;
	SparkDeviceAttentionScratch attention;
	SparkDeviceFfnScratch ffn;
	SparkDeviceExpertsScratch experts;
	SparkDeviceAttentionRope sliding_rope;
	SparkDeviceAttentionRope full_rope;
} GemmaDeviceModel;

static uint32_t GemmaMax(uint32_t a, uint32_t b)
{
	return(a > b ? a : b);
}

static uint32_t GemmaScratchSlices(GemmaDeviceModel *model, SparkDeviceStageSlice *slices)
{
	const uint64_t rows = model->stage.max_rows, hidden = gemma_shape.hidden;
	const uint64_t fused = GemmaMax(SparkDeviceAttentionFusedWidth(&gemma_sliding),SparkDeviceAttentionFusedWidth(&gemma_full));
	const uint64_t query = GemmaMax(SparkDeviceAttentionQueryWidth(&gemma_sliding),SparkDeviceAttentionQueryWidth(&gemma_full));
	const uint64_t kv = GemmaMax(SparkDeviceAttentionKvWidth(&gemma_sliding),SparkDeviceAttentionKvWidth(&gemma_full));
	SparkTensor *experts[] = { &model->experts.routed, &model->experts.logits, &model->experts.experts, &model->experts.weights, &model->experts.normed,
		&model->experts.gate_up, &model->experts.act, &model->experts.out };
	const SparkDeviceStageSlice plan[] =
	{
		{ rows * hidden * 2u, &model->normed },
		{ rows * hidden * 2u, &model->partial },
		{ rows * hidden * 2u, &model->expert_out },
		{ rows * fused * 2u, &model->attention.qkv },
		{ rows * query * 2u, &model->attention.query },
		{ rows * kv * 2u, &model->attention.key },
		{ rows * kv * 2u, &model->attention.value },
		{ SparkOpsAttentionScratchBytes(model->stage.max_rows,gemma_full.heads,gemma_full.head_dimension,model->stage.max_positions),
			&model->attention.records },
		{ rows * query * 2u, &model->attention.gated },
		{ rows * 2u * GD_FFN * 2u, &model->ffn.gate_up },
		{ rows * GD_FFN * 2u, &model->ffn.act }
	};
	const uint32_t count = sizeof(plan) / sizeof(plan[0]);
	uint32_t index;
	for ( index = 0u; index < count; index++ )
		slices[index] = plan[index];
	for ( index = 0u; index < sizeof(experts) / sizeof(experts[0]); index++ )
	{
		slices[count + index].bytes = SparkDeviceExpertsScratchBytes(&model->stage,&gemma_experts,index);
		slices[count + index].target = experts[index];
	}
	return(count + index);
}

static SparkStatus GemmaOpen(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
{
	const SparkDeviceStageConfigValue config[] =
	{
		{ "hidden", gemma_shape.hidden }, { "layers", GD_LAYERS }, { "vocab_rows", gemma_shape.vocab }, { "ffn_rows", GD_FFN },
		{ "experts", gemma_experts.experts }, { "top_k", gemma_experts.top_k }, { "full_period", GD_PERIOD }, { "sliding_window", gemma_sliding.window }
	};
	SparkDeviceStageSlice slices[GD_SLICES];
	GemmaDeviceModel *model = (GemmaDeviceModel *)calloc(1u,sizeof(*model));
	SparkStatus status;
	*out = model;
	if ( model == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkDeviceStageOpen(&model->stage,&gemma_shape,request);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageExpectConfig(&model->stage,config,sizeof(config) / sizeof(config[0]));
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageReserve(&model->stage,SparkDeviceStagePlan(0,slices,GemmaScratchSlices(model,slices)),0u,0,0u,
			request->configuration->state_budget_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkDeviceAttentionRopePrepare(&gemma_sliding,&model->sliding_rope);
	SparkDeviceAttentionRopePrepare(&gemma_full,&model->full_rope);
	return(SparkDeviceStageGeometry(&model->stage,geometry));
}

static SparkStatus GemmaBindLayer(GemmaDeviceModel *model, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
{
	static const char *const norms[] = { "input_norm", "post_attn_norm", "pre_ff_norm", "post_ff_norm", "post_ff_norm_1", "post_ff_norm_2" };
	GemmaDeviceLayer *layer = &model->layers[index];
	SparkTensor *targets[] = { &layer->input_norm, &layer->post_attn_norm, &layer->pre_ff_norm, &layer->post_ff_norm, &layer->post_ff_norm_1,
		&layer->post_ff_norm_2 };
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t norm;
	layer->full = index % GD_PERIOD == GD_PHASE ? 1u : 0u;
	for ( norm = 0u; norm < sizeof(norms) / sizeof(norms[0]) && status == SPARK_STATUS_OK; norm++ )
		status = SparkDeviceStageLayerTensor(&model->stage,weights,index,norms[norm],"f32",gemma_shape.hidden * sizeof(float),targets[norm]);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(&model->stage,weights,index,"layer_scalar","f32",sizeof(float),&layer->layer_scalar);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAttentionBind(&model->stage,weights,index,layer->full != 0u ? &gemma_full : &gemma_sliding,&layer->attention);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceFfnBind(&model->stage,weights,index,GD_FFN,&layer->ffn);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceExpertsBind(&model->stage,weights,index,&gemma_experts,&layer->experts);
	return(status);
}

static SparkStatus GemmaBindDevice(void *opaque, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
{
	GemmaDeviceModel *model = (GemmaDeviceModel *)opaque;
	SparkDeviceStageSlice slices[GD_SLICES];
	uint32_t index;
	SparkStatus status = SPARK_STATUS_OK;
	(void)geometry;
	if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != model->stage.pack_bytes )
		return(SparkDeviceStageRefuse(&model->stage,SPARK_STATUS_INVALID_ARGUMENT,"bind_device needs the device mapping of the whole rank pack"));
	for ( index = 0u; index < GD_LAYERS && status == SPARK_STATUS_OK; index++ )
		status = GemmaBindLayer(model,weights,index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceStageAllocate(&model->stage,slices,GemmaScratchSlices(model,slices)));
}

static SparkStatus GemmaStep(void *opaque, const SparkStageRunnerStep *step, void *device_queue)
{
	GemmaDeviceModel *model = (GemmaDeviceModel *)opaque;
	SparkDeviceStage *stage = &model->stage;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	SparkDeviceStepTensors t;
	uint32_t index;
	SparkStatus status = SparkDeviceStageStepTables(stage,step,&t);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsScaleRows(stage->ops,queue,t.hidden,(uint64_t)t.rows * gemma_shape.hidden,GD_EMBED_SCALE);
	for ( index = 0u; index < GD_LAYERS && status == SPARK_STATUS_OK; index++ )
	{
		const GemmaDeviceLayer *layer = &model->layers[index];
		status = SparkDeviceResidualNorm(stage,queue,&t,0,layer->input_norm,model->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceAttention(stage,queue,layer->full != 0u ? &gemma_full : &gemma_sliding,layer->full != 0u ? &model->full_rope : &model->sliding_rope,
				&t,&layer->attention,index,&model->attention,model->normed,model->partial);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceBranchNorm(stage,queue,&t,model->partial,0,0,0,layer->post_attn_norm,0);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceResidualNorm(stage,queue,&t,0,layer->pre_ff_norm,model->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(stage,queue,GD_FFN,SPARK_GATED_GELU_TANH,&t,&layer->ffn,&model->ffn,model->normed,model->partial);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceExperts(stage,queue,&gemma_experts,&t,&layer->experts,&model->experts,model->expert_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceBranchNorm(stage,queue,&t,model->partial,&model->expert_out,&layer->post_ff_norm_1,&layer->post_ff_norm_2,
				layer->post_ff_norm,&layer->layer_scalar);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageLayerDone(stage,index);
	}
	return(status);
}

const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModel(void)
{
	static const SparkStageRunnerModelInterface interface_table =
	{
		SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
		GD_TAG,
		"gemma4-26b-a4b",
		"official",
		GemmaOpen,
		SparkDeviceStageManifestCheck,
		SparkDeviceStageBindWeightd,
		SparkDeviceStageClose,
		GemmaStep,
		SparkDeviceStageSideband,
		SparkDeviceStageAttachKv,
		SparkDeviceStageRecurrentCopy,
		SparkDeviceStageResetSlot,
		SparkDeviceStageProbeBuffers,
		SparkDeviceStageReport,
		0,
		SparkDeviceStageLane,
		0,
		0,
		GemmaBindDevice
	};
	return(&interface_table);
}
