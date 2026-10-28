#include <math.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "inference/runner/device_stage_runner.h"

uint32_t SparkDeviceGatedDeltaQk(const SparkDeviceGatedDeltaShape *shape)
{
	return(shape->key_heads * shape->key_dimension);
}

uint32_t SparkDeviceGatedDeltaValues(const SparkDeviceGatedDeltaShape *shape)
{
	return(shape->value_heads * shape->value_dimension);
}

uint32_t SparkDeviceGatedDeltaChannels(const SparkDeviceGatedDeltaShape *shape)
{
	return(2u * SparkDeviceGatedDeltaQk(shape) + SparkDeviceGatedDeltaValues(shape));
}

uint32_t SparkDeviceGatedDeltaMix(const SparkDeviceGatedDeltaShape *shape)
{
	return(SparkDeviceGatedDeltaChannels(shape) + SparkDeviceGatedDeltaValues(shape));
}

uint32_t SparkDeviceGatedDeltaGates(const SparkDeviceGatedDeltaShape *shape)
{
	return(2u * shape->value_heads);
}

uint64_t SparkDeviceGatedDeltaStateBytes(const SparkDeviceGatedDeltaShape *shape)
{
	return((uint64_t)shape->value_heads * shape->key_dimension * shape->value_dimension * 4u);
}

uint64_t SparkDeviceGatedDeltaWindowBytes(const SparkDeviceGatedDeltaShape *shape)
{
	return((uint64_t)SparkDeviceGatedDeltaChannels(shape) * shape->conv_taps * 2u);
}

uint64_t SparkDeviceGatedDeltaKeepBytes(const SparkDeviceGatedDeltaShape *shape, uint32_t rows, uint32_t layers)
{
	return((uint64_t)layers * rows * (SparkDeviceGatedDeltaMix(shape) + SparkDeviceGatedDeltaGates(shape)) * sizeof(uint16_t));
}

static SparkTensor GatedDeltaKeepSlab(const SparkDeviceStage *stage, const SparkDeviceGatedDeltaShape *shape, const SparkDeviceGatedDeltaScratch *scratch,
	uint32_t recurrent_layer, uint32_t gates)
{
	const uint64_t mix_bytes = (uint64_t)stage->verify_rows * SparkDeviceGatedDeltaMix(shape) * sizeof(uint16_t);
	const uint64_t gate_bytes = (uint64_t)stage->verify_rows * SparkDeviceGatedDeltaGates(shape) * sizeof(uint16_t);
	return(SparkTensorSlice(scratch->keep,(uint64_t)recurrent_layer * (mix_bytes + gate_bytes) + (gates != 0u ? mix_bytes : 0u),
		gates != 0u ? gate_bytes : mix_bytes));
}

static SparkDeltaParams GatedDeltaParams(const SparkDeviceGatedDeltaShape *shape, uint32_t rows, const SparkOpsSequences *sequences)
{
	const uint32_t qk = SparkDeviceGatedDeltaQk(shape);
	SparkDeltaParams delta;
	memset(&delta,0,sizeof(delta));
	delta.rows = rows;
	delta.sequences = sequences->sequences;
	delta.key_heads = shape->key_heads;
	delta.value_heads = shape->value_heads;
	delta.key_dimension = shape->key_dimension;
	delta.value_dimension = shape->value_dimension;
	delta.mix_stride = SparkDeviceGatedDeltaMix(shape);
	delta.query_offset = 0u;
	delta.key_offset = qk;
	delta.value_offset = 2u * qk;
	delta.gate_stride = SparkDeviceGatedDeltaGates(shape);
	delta.beta_offset = 0u;
	delta.decay_offset = shape->value_heads;
	delta.output_stride = SparkDeviceGatedDeltaValues(shape);
	delta.commit = sequences->commit;
	delta.epsilon = shape->epsilon;
	return(delta);
}

static SparkConvParams GatedDeltaConvParams(const SparkDeviceGatedDeltaShape *shape, const SparkOpsSequences *sequences)
{
	SparkConvParams conv;
	memset(&conv,0,sizeof(conv));
	conv.channels = SparkDeviceGatedDeltaChannels(shape);
	conv.row_stride = SparkDeviceGatedDeltaMix(shape);
	conv.sequences = sequences->sequences;
	conv.kernel_taps = shape->conv_taps;
	conv.commit = sequences->commit;
	return(conv);
}

SparkStatus SparkDeviceGatedDeltaBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceGatedDeltaShape *shape, SparkDeviceGatedDeltaWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	SparkStatus status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_qkvz","bf16",(uint64_t)SparkDeviceGatedDeltaMix(shape) * hidden * 2u,
		&bound->mix);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_out","bf16",hidden * SparkDeviceGatedDeltaValues(shape) * 2u,&bound->out);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_ba","bf16",(uint64_t)SparkDeviceGatedDeltaGates(shape) * hidden * 2u,&bound->gates);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_conv","bf16",SparkDeviceGatedDeltaWindowBytes(shape),&bound->conv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_a_log","f32",(uint64_t)shape->value_heads * 4u,&bound->log_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_dt_bias","f32",(uint64_t)shape->value_heads * 4u,&bound->bias);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"gdn_norm","bf16",(uint64_t)shape->value_dimension * 2u,&bound->norm);
	return(status);
}

SparkStatus SparkDeviceGatedDelta(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceGatedDeltaShape *shape,
	const SparkDeviceStepTensors *tensors, const SparkDeviceGatedDeltaWeights *weights, uint32_t recurrent_layer,
	const SparkDeviceGatedDeltaScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, mix = SparkDeviceGatedDeltaMix(shape);
	const uint32_t gates = SparkDeviceGatedDeltaGates(shape), values = SparkDeviceGatedDeltaValues(shape);
	const SparkConvParams conv = GatedDeltaConvParams(shape,&tensors->sequences);
	const SparkDeltaParams delta = GatedDeltaParams(shape,rows,&tensors->sequences);
	SparkGatedNormParams gate;
	SparkStatus status = SparkDeviceLinear(stage,queue,normed,hidden,weights->mix,scratch->mix,mix,rows,hidden,mix);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,normed,hidden,weights->gates,scratch->gates,gates,rows,hidden,gates);
	if ( status == SPARK_STATUS_OK && tensors->sequences.commit == 0u )
		status = SparkOpsGather(stage->ops,queue,scratch->mix,tensors->row_identity,GatedDeltaKeepSlab(stage,shape,scratch,recurrent_layer,0u),rows,mix,rows);
	if ( status == SPARK_STATUS_OK && tensors->sequences.commit == 0u )
		status = SparkOpsGather(stage->ops,queue,scratch->gates,tensors->row_identity,GatedDeltaKeepSlab(stage,shape,scratch,recurrent_layer,1u),rows,gates,
			rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsCausalConv(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_WINDOW_POOL,recurrent_layer),&tensors->sequences,scratch->mix,weights->conv,
			scratch->mix,&conv);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsGatedDelta(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_STATE_POOL,recurrent_layer),&tensors->sequences,scratch->mix,scratch->gates,
			weights->log_scale,weights->bias,scratch->delta,scratch->core,&delta);
	memset(&gate,0,sizeof(gate));
	gate.rows = rows;
	gate.heads = shape->value_heads;
	gate.head_dimension = shape->value_dimension;
	gate.input_stride = values;
	gate.gate_stride = mix;
	gate.gate_offset = SparkDeviceGatedDeltaChannels(shape);
	gate.output_stride = values;
	gate.epsilon = shape->epsilon;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormGate(stage->ops,queue,scratch->core,scratch->mix,weights->norm,scratch->gated,&gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,scratch->gated,values,weights->out,output,hidden,rows,values,hidden);
	return(status);
}

SparkStatus SparkDeviceGatedDeltaKeep(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceGatedDeltaShape *shape,
	const SparkDeviceKeepTensors *keep, const SparkDeviceGatedDeltaWeights *weights, uint32_t recurrent_layer, const SparkDeviceGatedDeltaScratch *scratch)
{
	const uint32_t mix = SparkDeviceGatedDeltaMix(shape), gates = SparkDeviceGatedDeltaGates(shape);
	const SparkConvParams conv = GatedDeltaConvParams(shape,&keep->sequences);
	const SparkDeltaParams delta = GatedDeltaParams(shape,keep->rows,&keep->sequences);
	SparkStatus status = SPARK_STATUS_OK;
	if ( keep->rows != 0u )
		status = SparkOpsGather(stage->ops,queue,GatedDeltaKeepSlab(stage,shape,scratch,recurrent_layer,0u),keep->kept_rows,scratch->mix,keep->rows,mix,
			keep->frame_rows);
	if ( status == SPARK_STATUS_OK && keep->rows != 0u )
		status = SparkOpsGather(stage->ops,queue,GatedDeltaKeepSlab(stage,shape,scratch,recurrent_layer,1u),keep->kept_rows,scratch->gates,keep->rows,gates,
			keep->frame_rows);
	if ( status == SPARK_STATUS_OK && keep->rows != 0u )
		status = SparkOpsCausalConv(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_WINDOW_POOL,recurrent_layer),&keep->sequences,
			scratch->mix,weights->conv,scratch->mix,&conv);
	if ( status == SPARK_STATUS_OK && keep->rows != 0u )
		status = SparkOpsGatedDelta(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_STATE_POOL,recurrent_layer),&keep->sequences,
			scratch->mix,scratch->gates,weights->log_scale,weights->bias,scratch->delta,scratch->core,&delta);
	return(status);
}

uint32_t SparkDeviceAttentionQueryWidth(const SparkDeviceAttentionShape *shape)
{
	return(shape->heads * shape->head_dimension);
}

uint32_t SparkDeviceAttentionKvWidth(const SparkDeviceAttentionShape *shape)
{
	return(shape->kv_heads * shape->head_dimension);
}

static uint32_t AttentionKeyOffset(const SparkDeviceAttentionShape *shape)
{
	return((shape->output_gate != 0u && shape->gate_projection == 0u ? 2u : 1u) * SparkDeviceAttentionQueryWidth(shape));
}

static uint32_t AttentionValueOffset(const SparkDeviceAttentionShape *shape)
{
	return(AttentionKeyOffset(shape) + (shape->key_is_value != 0u ? 0u : SparkDeviceAttentionKvWidth(shape)));
}

uint32_t SparkDeviceAttentionFusedWidth(const SparkDeviceAttentionShape *shape)
{
	return(AttentionKeyOffset(shape) + (shape->key_is_value != 0u ? 1u : 2u) * SparkDeviceAttentionKvWidth(shape));
}

static void AttentionRopeParams(const SparkDeviceAttentionShape *shape, SparkHeadRopeParams *params, uint32_t heads, uint32_t input_offset,
	uint32_t head_stride, uint32_t rotate, uint32_t weighted, float multiply)
{
	const uint32_t pairs = shape->rope_pairs != 0u ? shape->rope_pairs : shape->rope_dimension / 2u;
	uint32_t pair;
	memset(params,0,sizeof(*params));
	params->heads = heads;
	params->head_dimension = shape->head_dimension;
	params->input_stride = SparkDeviceAttentionFusedWidth(shape);
	params->input_offset = input_offset;
	params->input_head_stride = head_stride;
	params->output_stride = heads * shape->head_dimension;
	params->rope_dimension = rotate != 0u ? shape->rope_dimension : 0u;
	params->epsilon = shape->epsilon;
	params->has_weight = weighted;
	params->rope_rounded_terms = shape->rope_rounded_terms;
	params->multiply = multiply;
	for ( pair = 0u; rotate != 0u && pair < pairs && pair < SPARK_KERNEL_ROPE_PAIRS; pair++ )
		params->inverse_frequency[pair] = (float)(1.0 / pow(shape->rope_theta,(double)(2u * pair) / (double)shape->rope_dimension));
}

void SparkDeviceAttentionRopePrepare(const SparkDeviceAttentionShape *shape, SparkDeviceAttentionRope *rope)
{
	const uint32_t query_head_stride = (shape->output_gate != 0u && shape->gate_projection == 0u ? 2u : 1u) * shape->head_dimension;
	const uint32_t weighted = shape->qk_weightless != 0u ? 0u : 1u;
	AttentionRopeParams(shape,&rope->query,shape->heads,0u,query_head_stride,1u,weighted,shape->query_multiply);
	AttentionRopeParams(shape,&rope->key,shape->kv_heads,AttentionKeyOffset(shape),shape->head_dimension,1u,weighted,0.0f);
	AttentionRopeParams(shape,&rope->value,shape->kv_heads,AttentionValueOffset(shape),shape->head_dimension,0u,0u,0.0f);
}

SparkStatus SparkDeviceAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceAttentionShape *shape, SparkDeviceAttentionWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	const uint32_t pairs = shape->rope_pairs != 0u ? shape->rope_pairs : shape->rope_dimension / 2u;
	SparkStatus status;
	if ( shape->rope_dimension % 2u != 0u || shape->rope_dimension > shape->head_dimension || pairs > shape->rope_dimension / 2u ||
		pairs > SPARK_KERNEL_ROPE_PAIRS || (shape->key_is_value != 0u && shape->value_norm == 0u) ||
		(uint64_t)shape->kv_heads * shape->head_dimension > (uint64_t)stage->shape->kv_heads * stage->shape->kv_head_dimension )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,
			"device attention needs an even rotary span within the head and the kernel's table, a normed value when keys are values, "
			"and a KV slot within the stage's"));
	status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_qkv","bf16",(uint64_t)SparkDeviceAttentionFusedWidth(shape) * hidden * 2u,&bound->qkv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_out","bf16",hidden * SparkDeviceAttentionQueryWidth(shape) * 2u,&bound->out);
	if ( status == SPARK_STATUS_OK && shape->gate_projection != 0u )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_gate","bf16",hidden * SparkDeviceAttentionQueryWidth(shape) * 2u,&bound->gate);
	if ( status == SPARK_STATUS_OK && shape->qk_weightless == 0u )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_q_norm","f32",(uint64_t)shape->head_dimension * 4u,&bound->q_norm);
	if ( status == SPARK_STATUS_OK && shape->qk_weightless == 0u )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_k_norm","f32",(uint64_t)shape->head_dimension * 4u,&bound->k_norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageNoteKvRead(stage,shape->window,(uint64_t)SparkDeviceAttentionKvWidth(shape) * 2u * sizeof(uint16_t));
	return(status);
}

SparkStatus SparkDeviceAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceAttentionShape *shape,
	const SparkDeviceAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceAttentionScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, fused = SparkDeviceAttentionFusedWidth(shape);
	const uint32_t q = SparkDeviceAttentionQueryWidth(shape), kv = SparkDeviceAttentionKvWidth(shape);
	SparkOpsPaged paged = SparkDeviceStagePaged(stage,tensors,kv_layer);
	SparkHeadRopeParams query = rope->query, key = rope->key, value = rope->value;
	SparkKvParams store;
	SparkAttentionParams attention;
	SparkStatus status = SparkDeviceLinear(stage,queue,normed,hidden,weights->qkv,scratch->qkv,fused,rows,hidden,fused);
	if ( status == SPARK_STATUS_OK && shape->gate_projection != 0u )
		status = SparkDeviceLinear(stage,queue,normed,hidden,weights->gate,scratch->gate,q,rows,hidden,q);
	paged.kv_heads = shape->kv_heads;
	query.rows = key.rows = value.rows = rows;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->q_norm,tensors->positions,scratch->query,&query);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->k_norm,tensors->positions,scratch->key,&key);
	if ( status == SPARK_STATUS_OK && shape->value_norm != 0u )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->k_norm,tensors->positions,scratch->value,&value);
	memset(&store,0,sizeof(store));
	store.rows = rows;
	store.kv_heads = shape->kv_heads;
	store.head_dimension = shape->head_dimension;
	store.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	store.pages_per_sequence = stage->pages_per_sequence;
	store.key_stride = kv;
	store.value_stride = shape->value_norm != 0u ? kv : fused;
	store.value_offset = shape->value_norm != 0u ? 0u : AttentionValueOffset(shape);
	store.pool_pages = stage->pool_pages;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsKvStore(stage->ops,queue,&paged,scratch->key,shape->value_norm != 0u ? scratch->value : scratch->qkv,&store);
	memset(&attention,0,sizeof(attention));
	attention.rows = rows;
	attention.heads = shape->heads;
	attention.kv_heads = shape->kv_heads;
	attention.head_dimension = shape->head_dimension;
	attention.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	attention.pages_per_sequence = stage->pages_per_sequence;
	attention.pool_pages = stage->pool_pages;
	attention.query_stride = q;
	attention.gate_stride = shape->gate_projection != 0u ? q : fused;
	attention.gate_offset = shape->gate_projection != 0u ? 0u : shape->head_dimension;
	attention.gate_head_stride = (shape->gate_projection != 0u ? 1u : 2u) * shape->head_dimension;
	attention.output_stride = q;
	attention.scale = shape->scale;
	attention.row_exact = stage->step_exact;
	attention.window = shape->window;
	attention.output_gate = shape->output_gate;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAttention(stage->ops,queue,&paged,scratch->query,shape->gate_projection != 0u ? scratch->gate : scratch->qkv,scratch->records,
			scratch->gated,&attention,tensors->host_positions,tensors->host_sequence_of_row);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,scratch->gated,q,weights->out,output,hidden,rows,q,hidden);
	return(status);
}

SparkStatus SparkDeviceFfnBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, uint32_t ffn,
	SparkDeviceFfnWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	SparkStatus status = SparkDeviceStageLayerTensor(stage,weights,layer,"ffn_gate_up","bf16",(uint64_t)2u * ffn * hidden * 2u,&bound->gate_up);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"ffn_down","bf16",hidden * ffn * 2u,&bound->down);
	return(status);
}

SparkStatus SparkDeviceGatedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t ffn, uint32_t activation, const SparkDeviceStepTensors *tensors,
	const SparkDeviceFfnWeights *weights, const SparkDeviceFfnScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows;
	SparkSiluMulParams silu;
	SparkStatus status = SparkDeviceLinear(stage,queue,normed,hidden,weights->gate_up,scratch->gate_up,2u * ffn,rows,hidden,2u * ffn);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&silu,0,sizeof(silu));
	silu.rows = rows;
	silu.dimension = ffn;
	silu.input_stride = 2u * ffn;
	silu.output_stride = ffn;
	status = SparkOpsGatedMul(stage->ops,queue,scratch->gate_up,scratch->act,&silu,activation);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,scratch->act,ffn,weights->down,output,hidden,rows,ffn,hidden);
	return(status);
}

uint64_t SparkDeviceExpertsScratchBytes(const SparkDeviceStage *stage, const SparkDeviceExpertsShape *shape, uint32_t part)
{
	const uint64_t rows = stage->max_rows, routes = rows * shape->top_k, hidden = stage->shape->hidden;
	const uint64_t bytes[] = { rows * hidden * sizeof(uint16_t), rows * shape->experts * sizeof(uint16_t), routes * sizeof(uint32_t),
		routes * sizeof(float), rows * hidden * sizeof(uint16_t), routes * 2u * shape->intermediate * sizeof(uint16_t),
		routes * shape->intermediate * sizeof(uint16_t), routes * hidden * sizeof(uint16_t) };
	return(part < sizeof(bytes) / sizeof(bytes[0]) ? bytes[part] : 0u);
}

SparkStatus SparkDeviceExpertsBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden, expert_bytes = (uint64_t)shape->experts * shape->intermediate * hidden * sizeof(uint16_t);
	SparkStatus status;
	if ( shape->top_k == 0u || shape->top_k > SPARK_ROUTE_MAX_TOP_K || shape->experts > SPARK_ROUTE_MAX_EXPERTS || shape->intermediate % 8u != 0u )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,"device experts need at most 256 experts, a top-k of at most 16 and an 8-aligned width"));
	status = SparkDeviceStageLayerTensor(stage,weights,layer,"router_proj","bf16",(uint64_t)shape->experts * hidden * sizeof(uint16_t),&bound->router);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"router_scale","f32",hidden * sizeof(float),&bound->router_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"expert_scale","f32",(uint64_t)shape->experts * sizeof(float),&bound->expert_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"pre_ff_norm_2","f32",hidden * sizeof(float),&bound->input_norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"moe_w1","bf16",expert_bytes,&bound->gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"moe_w3","bf16",expert_bytes,&bound->up);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"moe_w2","bf16",expert_bytes,&bound->down);
	if ( status == SPARK_STATUS_OK )
	{
		stage->routed_bytes += 3u * expert_bytes;
		stage->routed_bytes_per_token += 3u * expert_bytes / shape->experts * shape->top_k;
	}
	return(status);
}

static SparkGemvParams ExpertsRouted(uint32_t routes, uint32_t divisor, uint32_t input_dimension, uint32_t input_stride, uint32_t output_dimension,
	uint32_t output_stride)
{
	SparkGemvParams params;
	memset(&params,0,sizeof(params));
	params.rows = routes;
	params.input_dimension = input_dimension;
	params.output_dimension = output_dimension;
	params.input_stride = input_stride;
	params.output_stride = output_stride;
	params.route_divisor = divisor;
	return(params);
}

static SparkStatus ExpertsOverride(SparkDeviceStage *stage, const SparkDeviceExpertsShape *shape, const SparkDeviceStepTensors *tensors, uint32_t layer,
	SparkTensor *pinned)
{
	const uint64_t count = (uint64_t)tensors->rows * shape->top_k, bytes = count * sizeof(uint32_t);
	const uint32_t *experts = 0;
	uint64_t index;
	SparkStatus status;
	if ( tensors->route_override_width != shape->top_k || layer >= stage->shape->layers )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a route override names top-k experts for every layer of the stage"));
	*pinned = SparkTensorSlice(tensors->route_override,(uint64_t)layer * bytes,bytes);
	status = SparkDeviceHostOf(pinned->range,bytes,(void **)&experts);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"a route override must be host-visible device memory"));
	for ( index = 0u; index < count; index++ )
	{
		uint64_t other;
		if ( experts[index] >= shape->experts )
			return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a route override names an expert outside the layer"));
		for ( other = index - index % shape->top_k; other < index; other++ )
			if ( experts[other] == experts[index] )
				return(SparkDeviceStageRefuse(stage,SPARK_STATUS_INVALID_ARGUMENT,"a route override names an expert twice in one row"));
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDeviceExperts(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape, const SparkDeviceStepTensors *tensors,
	uint32_t layer, const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor output)
{
	SparkTensor pinned;
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, routes = rows * shape->top_k, inter = shape->intermediate;
	SparkBranchNormParams routed;
	SparkNormParams norm;
	SparkRouteParams route;
	SparkSiluMulParams gated;
	SparkGemvParams gate = ExpertsRouted(routes,shape->top_k,hidden,hidden,inter,2u * inter), down = ExpertsRouted(routes,1u,inter,inter,hidden,hidden);
	SparkStatus status;
	memset(&routed,0,sizeof(routed));
	routed.rows = rows;
	routed.dimension = hidden;
	routed.branch_stride = routed.hidden_stride = hidden;
	routed.epsilon = shape->epsilon;
	routed.output_scale = (float)(1.0 / sqrt((double)hidden));
	status = SparkOpsRmsNormScaled(stage->ops,queue,tensors->hidden,weights->router_scale,scratch->routed,&routed);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,scratch->routed,hidden,weights->router,scratch->logits,shape->experts,rows,hidden,shape->experts);
	memset(&route,0,sizeof(route));
	route.rows = rows;
	route.experts = shape->experts;
	route.top_k = shape->top_k;
	route.width = hidden;
	route.logit_stride = shape->experts;
	if ( status == SPARK_STATUS_OK && tensors->route_override_width != 0u )
		status = ExpertsOverride(stage,shape,tensors,layer,&pinned);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRouteTopk(stage->ops,queue,scratch->logits,weights->expert_scale,tensors->route_override_width != 0u ? &pinned : 0,scratch->experts,
			scratch->weights,&route);
	memset(&norm,0,sizeof(norm));
	norm.rows = rows;
	norm.dimension = hidden;
	norm.input_stride = norm.output_stride = hidden;
	norm.epsilon = shape->epsilon;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormResidual(stage->ops,queue,tensors->hidden,0,weights->input_norm,scratch->normed,&norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinearRouted(stage->ops,queue,scratch->normed,weights->gate,scratch->gate_up,scratch->experts,shape->experts,&gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinearRouted(stage->ops,queue,scratch->normed,weights->up,
			SparkTensorSlice(scratch->gate_up,(uint64_t)inter * sizeof(uint16_t),scratch->gate_up.bytes - (uint64_t)inter * sizeof(uint16_t)),scratch->experts,
			shape->experts,&gate);
	memset(&gated,0,sizeof(gated));
	gated.rows = routes;
	gated.dimension = inter;
	gated.input_stride = 2u * inter;
	gated.output_stride = inter;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsGatedMulRouted(stage->ops,queue,scratch->gate_up,scratch->weights,scratch->act,&gated,shape->activation);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinearRouted(stage->ops,queue,scratch->act,weights->down,scratch->out,scratch->experts,shape->experts,&down);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRouteCombine(stage->ops,queue,scratch->out,output,&route);
	return(status);
}

SparkStatus SparkDeviceBranchNorm(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceStepTensors *tensors, SparkTensor first,
	const SparkTensor *second, const SparkTensor *first_weight, const SparkTensor *second_weight, SparkTensor output_weight, const SparkTensor *scalar)
{
	SparkBranchNormParams params;
	memset(&params,0,sizeof(params));
	params.rows = tensors->rows;
	params.dimension = stage->shape->hidden;
	params.branch_stride = params.hidden_stride = stage->shape->hidden;
	params.branches = second != 0 ? 2u : 1u;
	params.has_scalar = scalar != 0 ? 1u : 0u;
	params.epsilon = stage->shape->post_epsilon != 0.0f ? stage->shape->post_epsilon : stage->shape->epsilon;
	return(SparkOpsRmsNormBranches(stage->ops,queue,tensors->hidden,first,second,first_weight,second_weight,output_weight,scalar,&params));
}

SparkStatus SparkDeviceResidualNorm(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceStepTensors *tensors,
	const SparkTensor *addend, SparkTensor weight, SparkTensor normed)
{
	SparkNormParams params;
	memset(&params,0,sizeof(params));
	params.rows = tensors->rows;
	params.dimension = stage->shape->hidden;
	params.input_stride = params.output_stride = stage->shape->hidden;
	params.has_addend = addend != 0 ? 1u : 0u;
	params.write_sum = addend != 0 ? 1u : 0u;
	params.epsilon = stage->shape->epsilon;
	return(SparkOpsRmsNormResidual(stage->ops,queue,tensors->hidden,addend,weight,normed,&params));
}

SparkStatus SparkDeviceLinear(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor x, uint32_t x_stride, SparkTensor w, SparkTensor y,
	uint32_t y_stride, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension)
{
	SparkGemvParams params;
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = input_dimension;
	params.output_dimension = output_dimension;
	params.input_stride = x_stride;
	params.output_stride = y_stride;
	params.projection = stage->shape->projection;
	params.row_exact = stage->step_exact;
	return(SparkOpsLinear(stage->ops,queue,x,w,y,&params));
}
