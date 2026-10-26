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
	const uint32_t gates = SparkDeviceGatedDeltaGates(shape), values = SparkDeviceGatedDeltaValues(shape), qk = SparkDeviceGatedDeltaQk(shape);
	SparkConvParams conv;
	SparkDeltaParams delta;
	SparkGatedNormParams gate;
	SparkStatus status = SparkDeviceLinear(stage,queue,normed,hidden,weights->mix,scratch->mix,mix,rows,hidden,mix,0u);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,normed,hidden,weights->gates,scratch->gates,gates,rows,hidden,gates,0u);
	memset(&conv,0,sizeof(conv));
	conv.channels = SparkDeviceGatedDeltaChannels(shape);
	conv.row_stride = mix;
	conv.sequences = tensors->sequences.sequences;
	conv.kernel_taps = shape->conv_taps;
	conv.commit = tensors->sequences.commit;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsCausalConv(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_WINDOW_POOL,recurrent_layer),&tensors->sequences,scratch->mix,weights->conv,
			scratch->mix,&conv);
	memset(&delta,0,sizeof(delta));
	delta.rows = rows;
	delta.sequences = tensors->sequences.sequences;
	delta.key_heads = shape->key_heads;
	delta.value_heads = shape->value_heads;
	delta.key_dimension = shape->key_dimension;
	delta.value_dimension = shape->value_dimension;
	delta.mix_stride = mix;
	delta.query_offset = 0u;
	delta.key_offset = qk;
	delta.value_offset = 2u * qk;
	delta.gate_stride = gates;
	delta.beta_offset = 0u;
	delta.decay_offset = shape->value_heads;
	delta.output_stride = values;
	delta.commit = tensors->sequences.commit;
	delta.epsilon = shape->epsilon;
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
		status = SparkDeviceLinear(stage,queue,scratch->gated,values,weights->out,output,hidden,rows,values,hidden,0u);
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
	return((shape->output_gate != 0u ? 2u : 1u) * SparkDeviceAttentionQueryWidth(shape));
}

uint32_t SparkDeviceAttentionFusedWidth(const SparkDeviceAttentionShape *shape)
{
	return(AttentionKeyOffset(shape) + 2u * SparkDeviceAttentionKvWidth(shape));
}

static void AttentionRopeParams(const SparkDeviceAttentionShape *shape, SparkHeadRopeParams *params, uint32_t heads, uint32_t input_offset,
	uint32_t head_stride)
{
	uint32_t pair;
	memset(params,0,sizeof(*params));
	params->heads = heads;
	params->head_dimension = shape->head_dimension;
	params->input_stride = SparkDeviceAttentionFusedWidth(shape);
	params->input_offset = input_offset;
	params->input_head_stride = head_stride;
	params->output_stride = heads * shape->head_dimension;
	params->rope_dimension = shape->rope_dimension;
	params->epsilon = shape->epsilon;
	for ( pair = 0u; pair < shape->rope_dimension / 2u && pair < SPARK_KERNEL_ROPE_PAIRS; pair++ )
		params->inverse_frequency[pair] = (float)(1.0 / pow(shape->rope_theta,(double)(2u * pair) / (double)shape->rope_dimension));
}

void SparkDeviceAttentionRopePrepare(const SparkDeviceAttentionShape *shape, SparkDeviceAttentionRope *rope)
{
	const uint32_t query_head_stride = (shape->output_gate != 0u ? 2u : 1u) * shape->head_dimension;
	AttentionRopeParams(shape,&rope->query,shape->heads,0u,query_head_stride);
	AttentionRopeParams(shape,&rope->key,shape->kv_heads,AttentionKeyOffset(shape),shape->head_dimension);
}

SparkStatus SparkDeviceAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceAttentionShape *shape, SparkDeviceAttentionWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	SparkStatus status;
	if ( shape->output_gate == 0u || shape->rope_dimension / 2u > SPARK_KERNEL_ROPE_PAIRS || shape->rope_dimension % 2u != 0u ||
		shape->kv_heads != stage->shape->kv_heads || shape->head_dimension != stage->shape->kv_head_dimension )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,
			"device attention needs a sigmoid output gate, an even rotary width within the kernel's table and the stage's KV heads"));
	status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_qkv","bf16",(uint64_t)SparkDeviceAttentionFusedWidth(shape) * hidden * 2u,&bound->qkv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_out","bf16",hidden * SparkDeviceAttentionQueryWidth(shape) * 2u,&bound->out);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_q_norm","f32",(uint64_t)shape->head_dimension * 4u,&bound->q_norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_k_norm","f32",(uint64_t)shape->head_dimension * 4u,&bound->k_norm);
	return(status);
}

SparkStatus SparkDeviceAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceAttentionShape *shape,
	const SparkDeviceAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceAttentionScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, fused = SparkDeviceAttentionFusedWidth(shape);
	const uint32_t q = SparkDeviceAttentionQueryWidth(shape), kv = SparkDeviceAttentionKvWidth(shape);
	const SparkOpsPaged paged = SparkDeviceStagePaged(stage,tensors,kv_layer);
	SparkHeadRopeParams query = rope->query, key = rope->key;
	SparkKvParams store;
	SparkAttentionParams attention;
	SparkStatus status = SparkDeviceLinear(stage,queue,normed,hidden,weights->qkv,scratch->qkv,fused,rows,hidden,fused,0u);
	query.rows = key.rows = rows;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->q_norm,tensors->positions,scratch->query,&query);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->k_norm,tensors->positions,scratch->key,&key);
	memset(&store,0,sizeof(store));
	store.rows = rows;
	store.kv_heads = shape->kv_heads;
	store.head_dimension = shape->head_dimension;
	store.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	store.pages_per_sequence = stage->pages_per_sequence;
	store.key_stride = kv;
	store.value_stride = fused;
	store.value_offset = AttentionKeyOffset(shape) + kv;
	store.pool_pages = stage->pool_pages;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsKvStore(stage->ops,queue,&paged,scratch->key,scratch->qkv,&store);
	memset(&attention,0,sizeof(attention));
	attention.rows = rows;
	attention.heads = shape->heads;
	attention.kv_heads = shape->kv_heads;
	attention.head_dimension = shape->head_dimension;
	attention.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	attention.pages_per_sequence = stage->pages_per_sequence;
	attention.pool_pages = stage->pool_pages;
	attention.query_stride = q;
	attention.gate_stride = fused;
	attention.gate_offset = shape->head_dimension;
	attention.gate_head_stride = 2u * shape->head_dimension;
	attention.output_stride = q;
	attention.scale = shape->scale;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAttention(stage->ops,queue,&paged,scratch->query,scratch->qkv,scratch->records,scratch->gated,&attention,tensors->host_positions,
			tensors->host_sequence_of_row);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,scratch->gated,q,weights->out,output,hidden,rows,q,hidden,0u);
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

SparkStatus SparkDeviceGatedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t ffn, const SparkDeviceStepTensors *tensors,
	const SparkDeviceFfnWeights *weights, const SparkDeviceFfnScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows;
	SparkSiluMulParams silu;
	SparkStatus status = SparkDeviceLinear(stage,queue,normed,hidden,weights->gate_up,scratch->gate_up,2u * ffn,rows,hidden,2u * ffn,0u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( rows <= SPARK_OPS_GEMV_MAX_ROWS )
		return(SparkDeviceLinear(stage,queue,scratch->gate_up,2u * ffn,weights->down,output,hidden,rows,ffn,hidden,1u));
	memset(&silu,0,sizeof(silu));
	silu.rows = rows;
	silu.dimension = ffn;
	silu.input_stride = 2u * ffn;
	silu.output_stride = ffn;
	status = SparkOpsSiluMul(stage->ops,queue,scratch->gate_up,scratch->act,&silu);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinear(stage,queue,scratch->act,ffn,weights->down,output,hidden,rows,ffn,hidden,0u);
	return(status);
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
	return(SparkOpsLinear(stage->ops,queue,x,w,y,&params));
}
