#include <stdlib.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_gemma4_device_stage.h"

#define GD_MAX_LAYERS 60u
#define GD_SLICES 24u

typedef struct GemmaGeometry
{
	const char *weightd_model;
	SparkDeviceStageShape shape;
	SparkDeviceAttentionShape sliding;
	SparkDeviceAttentionShape full;
	SparkDeviceExpertsShape experts;
	uint32_t period;
	uint32_t phase;
	uint32_t ffn;
	float embed_scale;
} GemmaGeometry;

static const GemmaGeometry gemma_26b =
{
	.weightd_model = "gemma4-26b-a4b",
	.shape = { .tag = "sparkpipe_gemma4_26b_device", .hidden = 2816u, .vocab = 262144u, .layers = 30u, .kv_layers = 30u, .kv_heads = 8u,
		.kv_head_dimension = 256u, .pack_alignment = 128u, .epsilon = 1e-6f, .tied_head = 1u, .logit_softcap = 30.0f,
		.projection = SPARK_PROJECTION_GEMV },
	.sliding = { .heads = 16u, .kv_heads = 8u, .head_dimension = 256u, .rope_dimension = 256u, .rope_theta = 10000.0, .epsilon = 1e-6f,
		.scale = 1.0f, .value_norm = 1u, .window = 1024u, .rope_rounded_terms = 1u },
	.full = { .heads = 16u, .kv_heads = 2u, .head_dimension = 512u, .rope_dimension = 512u, .rope_theta = 1000000.0, .epsilon = 1e-6f,
		.scale = 1.0f, .key_is_value = 1u, .value_norm = 1u, .rope_pairs = 64u, .rope_rounded_terms = 1u },
	.experts = { .experts = 128u, .top_k = 8u, .intermediate = 704u, .activation = SPARK_GATED_GELU_TANH, .epsilon = 1e-6f },
	.period = 6u,
	.phase = 5u,
	.ffn = 2112u,
	.embed_scale = 53.0f
};

static const GemmaGeometry gemma_31b =
{
	.weightd_model = "gemma4-31b",
	.shape = { .tag = "sparkpipe_gemma4_31b_device", .hidden = 5376u, .vocab = 262144u, .layers = 60u, .kv_layers = 60u, .kv_heads = 16u,
		.kv_head_dimension = 256u, .pack_alignment = 128u, .epsilon = 1e-6f, .tied_head = 1u, .logit_softcap = 30.0f,
		.projection = SPARK_PROJECTION_ROW_TILES },
	.sliding = { .heads = 32u, .kv_heads = 16u, .head_dimension = 256u, .rope_dimension = 256u, .rope_theta = 10000.0, .epsilon = 1e-6f,
		.scale = 1.0f, .value_norm = 1u, .window = 1024u, .rope_rounded_terms = 1u },
	.full = { .heads = 32u, .kv_heads = 4u, .head_dimension = 512u, .rope_dimension = 512u, .rope_theta = 1000000.0, .epsilon = 1e-6f,
		.scale = 1.0f, .key_is_value = 1u, .value_norm = 1u, .rope_pairs = 64u, .rope_rounded_terms = 1u },
	.period = 6u,
	.phase = 5u,
	.ffn = 21504u,
	.embed_scale = 73.5f
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
	const GemmaGeometry *geometry;
	GemmaDeviceLayer layers[GD_MAX_LAYERS];
	SparkTensor normed;
	SparkTensor partial;
	SparkTensor expert_out;
	SparkDeviceAttentionScratch attention;
	SparkDeviceFfnScratch ffn;
	SparkDeviceExpertsScratch experts;
	SparkDeviceAttentionRope sliding_rope;
	SparkDeviceAttentionRope full_rope;
} GemmaDeviceModel;

static uint64_t GemmaMax(uint64_t a, uint64_t b)
{
	return(a > b ? a : b);
}

static uint32_t GemmaScratchSlices(GemmaDeviceModel *model, SparkDeviceStageSlice *slices)
{
	const GemmaGeometry *g = model->geometry;
	const uint64_t rows = model->stage.max_rows, hidden = g->shape.hidden;
	const uint64_t fused = GemmaMax(SparkDeviceAttentionFusedWidth(&g->sliding),SparkDeviceAttentionFusedWidth(&g->full));
	const uint64_t query = GemmaMax(SparkDeviceAttentionQueryWidth(&g->sliding),SparkDeviceAttentionQueryWidth(&g->full));
	const uint64_t kv = GemmaMax(SparkDeviceAttentionKvWidth(&g->sliding),SparkDeviceAttentionKvWidth(&g->full));
	SparkTensor *experts[] = { &model->experts.routed, &model->experts.logits, &model->experts.experts, &model->experts.weights, &model->experts.normed,
		&model->experts.gate_up, &model->experts.act, &model->experts.out };
	const SparkDeviceStageSlice plan[] =
	{
		{ rows * hidden * 2u, &model->normed },
		{ rows * hidden * 2u, &model->partial },
		{ rows * fused * 2u, &model->attention.qkv },
		{ rows * query * 2u, &model->attention.query },
		{ rows * kv * 2u, &model->attention.key },
		{ rows * kv * 2u, &model->attention.value },
		{ GemmaMax(SparkOpsAttentionScratchBytes(model->stage.max_rows,g->full.heads,g->full.head_dimension,model->stage.max_positions),
			SparkOpsAttentionScratchBytes(model->stage.max_rows,g->sliding.heads,g->sliding.head_dimension,model->stage.max_positions)),
			&model->attention.records },
		{ rows * query * 2u, &model->attention.gated },
		{ rows * 2u * g->ffn * 2u, &model->ffn.gate_up },
		{ rows * g->ffn * 2u, &model->ffn.act }
	};
	uint32_t count = sizeof(plan) / sizeof(plan[0]), index;
	for ( index = 0u; index < count; index++ )
		slices[index] = plan[index];
	if ( g->experts.experts == 0u )
		return(count);
	slices[count].bytes = rows * hidden * 2u;
	slices[count++].target = &model->expert_out;
	for ( index = 0u; index < sizeof(experts) / sizeof(experts[0]); index++ )
	{
		slices[count].bytes = SparkDeviceExpertsScratchBytes(&model->stage,&g->experts,index);
		slices[count++].target = experts[index];
	}
	return(count);
}

static SparkStatus GemmaOpen(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry, const GemmaGeometry *g)
{
	const SparkDeviceStageConfigValue config[] =
	{
		{ "hidden", g->shape.hidden }, { "layers", g->shape.layers }, { "vocab_rows", g->shape.vocab }, { "ffn_rows", g->ffn },
		{ "full_period", g->period }, { "sliding_window", g->sliding.window }, { "experts", g->experts.experts }, { "top_k", g->experts.top_k }
	};
	SparkDeviceStageSlice slices[GD_SLICES];
	GemmaDeviceModel *model = (GemmaDeviceModel *)calloc(1u,sizeof(*model));
	SparkStatus status;
	*out = model;
	if ( model == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	model->geometry = g;
	if ( g->shape.layers > GD_MAX_LAYERS )
		return(SparkDeviceStageRefuse(&model->stage,SPARK_STATUS_UNSUPPORTED,"the Gemma 4 device stage holds at most 60 layers"));
	status = SparkDeviceStageOpen(&model->stage,&g->shape,request);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageExpectConfig(&model->stage,config,g->experts.experts != 0u ? 8u : 6u);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageReserve(&model->stage,SparkDeviceStagePlan(0,slices,GemmaScratchSlices(model,slices)),0u,0,0u,
			request->configuration->state_budget_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkDeviceAttentionRopePrepare(&g->sliding,&model->sliding_rope);
	SparkDeviceAttentionRopePrepare(&g->full,&model->full_rope);
	return(SparkDeviceStageGeometry(&model->stage,geometry));
}

static SparkStatus Gemma26bOpen(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
{
	return(GemmaOpen(request,out,geometry,&gemma_26b));
}

static SparkStatus Gemma31bOpen(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
{
	return(GemmaOpen(request,out,geometry,&gemma_31b));
}

static SparkStatus GemmaBindLayer(GemmaDeviceModel *model, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
{
	static const char *const norms[] = { "input_norm", "post_attn_norm", "pre_ff_norm", "post_ff_norm", "post_ff_norm_1", "post_ff_norm_2" };
	const GemmaGeometry *g = model->geometry;
	GemmaDeviceLayer *layer = &model->layers[index];
	SparkTensor *targets[] = { &layer->input_norm, &layer->post_attn_norm, &layer->pre_ff_norm, &layer->post_ff_norm, &layer->post_ff_norm_1,
		&layer->post_ff_norm_2 };
	const uint32_t norm_count = g->experts.experts != 0u ? 6u : 4u;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t norm;
	layer->full = index % g->period == g->phase ? 1u : 0u;
	for ( norm = 0u; norm < norm_count && status == SPARK_STATUS_OK; norm++ )
		status = SparkDeviceStageLayerTensor(&model->stage,weights,index,norms[norm],"f32",g->shape.hidden * sizeof(float),targets[norm]);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(&model->stage,weights,index,"layer_scalar","f32",sizeof(float),&layer->layer_scalar);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceAttentionBind(&model->stage,weights,index,layer->full != 0u ? &g->full : &g->sliding,&layer->attention);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceFfnBind(&model->stage,weights,index,g->ffn,&layer->ffn);
	if ( status == SPARK_STATUS_OK && g->experts.experts != 0u )
		status = SparkDeviceExpertsBind(&model->stage,weights,index,&g->experts,&layer->experts);
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
	for ( index = 0u; index < model->geometry->shape.layers && status == SPARK_STATUS_OK; index++ )
		status = GemmaBindLayer(model,weights,index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkDeviceStageAllocate(&model->stage,slices,GemmaScratchSlices(model,slices)));
}

static SparkStatus GemmaStep(void *opaque, const SparkStageRunnerStep *step, void *device_queue)
{
	GemmaDeviceModel *model = (GemmaDeviceModel *)opaque;
	const GemmaGeometry *g = model->geometry;
	SparkDeviceStage *stage = &model->stage;
	SparkDeviceQueue *queue = (SparkDeviceQueue *)device_queue;
	SparkDeviceStepTensors t;
	uint32_t index;
	SparkStatus status = SparkDeviceStageStepTables(stage,step,&t);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsScaleRows(stage->ops,queue,t.hidden,(uint64_t)t.rows * g->shape.hidden,g->embed_scale);
	for ( index = 0u; index < g->shape.layers && status == SPARK_STATUS_OK; index++ )
	{
		const GemmaDeviceLayer *layer = &model->layers[index];
		status = SparkDeviceResidualNorm(stage,queue,&t,0,layer->input_norm,model->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceAttention(stage,queue,layer->full != 0u ? &g->full : &g->sliding,layer->full != 0u ? &model->full_rope : &model->sliding_rope,
				&t,&layer->attention,index,&model->attention,model->normed,model->partial);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceBranchNorm(stage,queue,&t,model->partial,0,0,0,layer->post_attn_norm,0);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceResidualNorm(stage,queue,&t,0,layer->pre_ff_norm,model->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(stage,queue,g->ffn,SPARK_GATED_GELU_TANH,&t,&layer->ffn,&model->ffn,model->normed,model->partial);
		if ( status == SPARK_STATUS_OK && g->experts.experts != 0u )
			status = SparkDeviceExperts(stage,queue,&g->experts,&t,index,&layer->experts,&model->experts,model->expert_out);
		if ( status == SPARK_STATUS_OK && g->experts.experts != 0u )
			status = SparkDeviceBranchNorm(stage,queue,&t,model->partial,&model->expert_out,&layer->post_ff_norm_1,&layer->post_ff_norm_2,
				layer->post_ff_norm,&layer->layer_scalar);
		else if ( status == SPARK_STATUS_OK )
			status = SparkDeviceBranchNorm(stage,queue,&t,model->partial,0,0,0,layer->post_ff_norm,&layer->layer_scalar);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerDone(stage,queue,&t,index,0);
	}
	return(status);
}

const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModel(void)
{
	static const SparkStageRunnerModelInterface interface_table =
	{
		SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
		"sparkpipe_gemma4_26b_device",
		"gemma4-26b-a4b",
		"official",
		Gemma26bOpen,
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
		GemmaBindDevice,
		0
	};
	return(&interface_table);
}

const SparkStageRunnerModelInterface *SparkGemma4_31bDeviceStageModel(void)
{
	static const SparkStageRunnerModelInterface interface_table =
	{
		SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
		"sparkpipe_gemma4_31b_device",
		"gemma4-31b",
		"official",
		Gemma31bOpen,
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
		GemmaBindDevice,
		0
	};
	return(&interface_table);
}
