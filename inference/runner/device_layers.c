#include <math.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_rope_plan.h"
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
	SparkStatus status = SparkDeviceStageLayerWeight(stage,weights,layer,"gdn_qkvz",SparkDeviceGatedDeltaMix(shape),hidden,&bound->mix);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"gdn_out",hidden,SparkDeviceGatedDeltaValues(shape),&bound->out);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"gdn_ba",SparkDeviceGatedDeltaGates(shape),hidden,&bound->gates);
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
	SparkStatus status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->mix,scratch->mix,mix,rows,hidden,mix);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gates,scratch->gates,gates,rows,hidden,gates);
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
	gate.sigmoid_gate = shape->sigmoid_gate;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormGate(stage->ops,queue,scratch->core,scratch->mix,weights->norm,scratch->gated,&gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,scratch->gated,values,&weights->out,output,hidden,rows,values,hidden);
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

uint32_t SparkDeviceChannelDeltaWidth(const SparkDeviceChannelDeltaShape *shape)
{
	return(shape->heads * shape->dimension);
}

uint64_t SparkDeviceChannelDeltaStateBytes(const SparkDeviceChannelDeltaShape *shape)
{
	return((uint64_t)shape->heads * shape->dimension * shape->dimension * 4u);
}

uint64_t SparkDeviceChannelDeltaWindowBytes(const SparkDeviceChannelDeltaShape *shape)
{
	return((uint64_t)3u * SparkDeviceChannelDeltaWidth(shape) * shape->conv_taps * 2u);
}

SparkStatus SparkDeviceChannelDeltaBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceChannelDeltaShape *shape, SparkDeviceChannelDeltaWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden, width = SparkDeviceChannelDeltaWidth(shape), source = shape->low_rank != 0u ? shape->low_rank : hidden;
	SparkStatus status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_qkv",3u * width,hidden,&bound->qkv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_beta",shape->heads,hidden,&bound->beta);
	if ( status == SPARK_STATUS_OK && shape->low_rank != 0u )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_f_a",shape->low_rank,hidden,&bound->decay_down);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_f",width,source,&bound->decay);
	if ( status == SPARK_STATUS_OK && shape->low_rank != 0u )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_g_a",shape->low_rank,hidden,&bound->gate_down);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_g",width,source,&bound->gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"kda_conv","bf16",3u * width * shape->conv_taps * 2u,&bound->conv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"kda_a_log","f32",(uint64_t)shape->heads * 4u,&bound->log_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"kda_dt_bias","f32",width * 4u,&bound->bias);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"kda_norm","bf16",(uint64_t)shape->dimension * 2u,&bound->norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"kda_out",hidden,width,&bound->out);
	return(status);
}

SparkStatus SparkDeviceChannelDelta(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceChannelDeltaShape *shape,
	const SparkDeviceStepTensors *tensors, const SparkDeviceChannelDeltaWeights *weights, uint32_t recurrent_layer,
	const SparkDeviceChannelDeltaScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, width = SparkDeviceChannelDeltaWidth(shape), channels = 3u * width;
	SparkConvParams conv;
	SparkDeltaParams delta;
	SparkGatedNormParams gate;
	SparkStatus status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->qkv,scratch->qkv,channels,rows,hidden,channels);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->beta,scratch->beta,shape->heads,rows,hidden,shape->heads);
	if ( status == SPARK_STATUS_OK && shape->low_rank != 0u )
	{
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->decay_down,scratch->low,shape->low_rank,rows,hidden,shape->low_rank);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(stage,queue,scratch->low,shape->low_rank,&weights->decay,scratch->decay,width,rows,shape->low_rank,width);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gate_down,scratch->low,shape->low_rank,rows,hidden,shape->low_rank);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(stage,queue,scratch->low,shape->low_rank,&weights->gate,scratch->gate,width,rows,shape->low_rank,width);
	}
	else if ( status == SPARK_STATUS_OK )
	{
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->decay,scratch->decay,width,rows,hidden,width);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gate,scratch->gate,width,rows,hidden,width);
	}
	memset(&conv,0,sizeof(conv));
	conv.channels = channels;
	conv.row_stride = channels;
	conv.sequences = tensors->sequences.sequences;
	conv.kernel_taps = shape->conv_taps;
	conv.commit = tensors->sequences.commit;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsCausalConv(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_WINDOW_POOL,recurrent_layer),&tensors->sequences,
			scratch->qkv,weights->conv,scratch->qkv,&conv);
	memset(&delta,0,sizeof(delta));
	delta.rows = rows;
	delta.sequences = tensors->sequences.sequences;
	delta.key_heads = delta.value_heads = shape->heads;
	delta.key_dimension = delta.value_dimension = shape->dimension;
	delta.mix_stride = channels;
	delta.key_offset = width;
	delta.value_offset = 2u * width;
	delta.gate_stride = width;
	delta.beta_stride = shape->heads;
	delta.output_stride = width;
	delta.commit = tensors->sequences.commit;
	delta.epsilon = shape->epsilon;
	delta.minimum_log_decay = shape->minimum_log_decay;
	delta.doubled_beta = shape->doubled_beta;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsChannelDelta(stage->ops,queue,SparkDeviceStageRecurrent(stage,SPARK_DEVICE_GATED_DELTA_STATE_POOL,recurrent_layer),&tensors->sequences,
			scratch->qkv,scratch->decay,scratch->beta,weights->log_scale,weights->bias,scratch->delta,scratch->core,&delta);
	memset(&gate,0,sizeof(gate));
	gate.rows = rows;
	gate.heads = shape->heads;
	gate.head_dimension = shape->dimension;
	gate.input_stride = gate.gate_stride = gate.output_stride = width;
	gate.epsilon = shape->epsilon;
	gate.sigmoid_gate = 1u;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormGate(stage->ops,queue,scratch->core,scratch->gate,weights->norm,scratch->gated,&gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,scratch->gated,width,&weights->out,output,hidden,rows,width,hidden);
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

static uint32_t AttentionValueDimension(const SparkDeviceAttentionShape *shape)
{
	return(shape->value_dimension != 0u ? shape->value_dimension : shape->head_dimension);
}

uint32_t SparkDeviceAttentionValueWidth(const SparkDeviceAttentionShape *shape)
{
	return(shape->kv_heads * AttentionValueDimension(shape));
}

uint32_t SparkDeviceAttentionOutputWidth(const SparkDeviceAttentionShape *shape)
{
	return(shape->heads * AttentionValueDimension(shape));
}

uint32_t SparkDeviceAttentionHeadGateRows(const SparkDeviceAttentionShape *shape, uint32_t degree)
{
	const uint32_t local = shape->heads / degree;
	return(local + (local & 1u));
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
	return(AttentionKeyOffset(shape) + SparkDeviceAttentionKvWidth(shape) + (shape->key_is_value != 0u ? 0u : SparkDeviceAttentionValueWidth(shape)));
}

static void AttentionRopeParams(const SparkDeviceAttentionShape *shape, SparkHeadRopeParams *params, uint32_t heads, uint32_t head_dimension,
	uint32_t input_offset, uint32_t head_stride, uint32_t rotate, uint32_t weighted, uint32_t plain, float multiply)
{
	const uint32_t pairs = shape->rope_pairs != 0u ? shape->rope_pairs : shape->rope_dimension / 2u;
	uint32_t pair;
	memset(params,0,sizeof(*params));
	params->heads = heads;
	params->head_dimension = head_dimension;
	params->input_stride = SparkDeviceAttentionFusedWidth(shape);
	params->input_offset = input_offset;
	params->input_head_stride = head_stride;
	params->output_stride = heads * head_dimension;
	params->plain = plain;
	params->rope_dimension = rotate != 0u ? shape->rope_dimension : 0u;
	params->epsilon = shape->epsilon;
	params->has_weight = weighted;
	params->rope_rounded_terms = shape->rope_rounded_terms;
	params->multiply = multiply;
	params->scale = shape->rope_scale != 0.0f ? shape->rope_scale : 1.0f;
	for ( pair = 0u; rotate != 0u && pair < pairs && pair < SPARK_KERNEL_ROPE_PAIRS; pair++ )
		params->inverse_frequency[pair] = (float)(1.0 / pow(shape->rope_theta,(double)(2u * pair) / (double)shape->rope_dimension));
	if ( rotate != 0u && shape->yarn_factor > 1.0f )
	{
		const SparkRopePlanDomain yarn = { (float)shape->rope_theta, shape->yarn_factor, shape->yarn_original, shape->yarn_beta_fast, shape->yarn_beta_slow,
			params->scale, shape->rope_dimension };
		memset(params->inverse_frequency,0,sizeof(params->inverse_frequency));
		SparkRopePlanBuildYarnInvFrequency(&yarn,params->inverse_frequency);
	}
}

void SparkDeviceAttentionRopePrepare(const SparkDeviceAttentionShape *shape, SparkDeviceAttentionRope *rope)
{
	const uint32_t query_head_stride = (shape->output_gate != 0u && shape->gate_projection == 0u ? 2u : 1u) * shape->head_dimension;
	const uint32_t weighted = shape->qk_weightless != 0u || shape->qk_plain != 0u ? 0u : 1u, values = AttentionValueDimension(shape);
	AttentionRopeParams(shape,&rope->query,shape->heads,shape->head_dimension,0u,query_head_stride,1u,weighted,shape->qk_plain,shape->query_multiply);
	AttentionRopeParams(shape,&rope->key,shape->kv_heads,shape->head_dimension,AttentionKeyOffset(shape),shape->head_dimension,1u,weighted,shape->qk_plain,0.0f);
	AttentionRopeParams(shape,&rope->value,shape->kv_heads,values,AttentionValueOffset(shape),values,0u,0u,shape->value_norm != 0u ? 0u : 1u,
		shape->value_scale);
}

SparkStatus SparkDeviceAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceAttentionShape *shape, SparkDeviceAttentionWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	const uint32_t pairs = shape->rope_pairs != 0u ? shape->rope_pairs : shape->rope_dimension / 2u, degree = stage->tp_degree;
	SparkStatus status;
	if ( shape->rope_dimension % 2u != 0u || shape->rope_dimension > shape->head_dimension || pairs > shape->rope_dimension / 2u ||
		pairs > SPARK_KERNEL_ROPE_PAIRS || (shape->key_is_value != 0u && (shape->value_norm == 0u || shape->value_dimension != 0u)) ||
		(uint64_t)shape->kv_heads * (shape->head_dimension + AttentionValueDimension(shape)) >
		(uint64_t)2u * stage->shape->kv_heads * stage->shape->kv_head_dimension )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,
			"device attention needs an even rotary span within the head and the kernel's table, a normed value when keys are values, "
			"and a KV slot within the stage's"));
	if ( SparkDeviceAttentionFusedWidth(shape) % degree != 0u || shape->heads % degree != 0u )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,"device attention splits the fused projection rows and whole heads evenly over the ranks"));
	status = SparkDeviceStageLayerWeight(stage,weights,layer,"attn_qkv",SparkDeviceAttentionFusedWidth(shape) / degree,hidden,&bound->qkv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"attn_out",hidden,SparkDeviceAttentionOutputWidth(shape) / degree,&bound->out);
	if ( status == SPARK_STATUS_OK && shape->gate_projection != 0u )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"attn_gate",SparkDeviceAttentionQueryWidth(shape) / degree,hidden,&bound->gate);
	if ( status == SPARK_STATUS_OK && shape->head_gate != 0u )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"attn_gate",SparkDeviceAttentionHeadGateRows(shape,degree),hidden,&bound->gate);
	if ( status == SPARK_STATUS_OK && shape->qk_weightless == 0u && shape->qk_plain == 0u )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_q_norm","f32",(uint64_t)shape->head_dimension * 4u,&bound->q_norm);
	if ( status == SPARK_STATUS_OK && shape->qk_weightless == 0u && shape->qk_plain == 0u )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"attn_k_norm","f32",(uint64_t)shape->head_dimension * 4u,&bound->k_norm);
	if ( status == SPARK_STATUS_OK && shape->sinks != 0u )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"sinks","f32",(uint64_t)shape->heads * 4u,&bound->sinks);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageNoteKvRead(stage,shape->window,((uint64_t)SparkDeviceAttentionKvWidth(shape) + SparkDeviceAttentionValueWidth(shape)) * sizeof(uint16_t));
	return(status);
}

SparkStatus SparkDeviceAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceAttentionShape *shape,
	const SparkDeviceAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceAttentionScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, fused = SparkDeviceAttentionFusedWidth(shape);
	const uint32_t q = SparkDeviceAttentionQueryWidth(shape), kv = SparkDeviceAttentionKvWidth(shape), values = SparkDeviceAttentionValueWidth(shape);
	const uint32_t separate_value = shape->value_norm != 0u || shape->value_scale != 0.0f ? 1u : 0u, degree = stage->tp_degree;
	const uint32_t heads_out = SparkDeviceAttentionOutputWidth(shape) / degree, local_heads = shape->heads / degree, first_head = stage->tp_rank * local_heads;
	const uint32_t gate_width = shape->gate_projection != 0u ? q / degree : 0u;
	SparkOpsPaged paged = SparkDeviceStagePaged(stage,tensors,kv_layer);
	SparkHeadRopeParams query = rope->query, key = rope->key, value = rope->value;
	SparkTensor sinks = weights->sinks;
	SparkKvParams store;
	SparkAttentionParams attention;
	SparkStatus status;
	if ( degree > 1u )
	{
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->qkv,scratch->qkv_local,fused / degree,rows,hidden,fused / degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageGatherRows(stage,queue,scratch->qkv_local,scratch->qkv_gathered,scratch->qkv,rows,fused / degree);
	}
	else
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->qkv,scratch->qkv,fused,rows,hidden,fused);
	if ( status == SPARK_STATUS_OK && shape->gate_projection != 0u )
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gate,scratch->gate,gate_width,rows,hidden,gate_width);
	paged.kv_heads = shape->kv_heads;
	query.rows = key.rows = value.rows = rows;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->q_norm,tensors->positions,scratch->query,&query);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->k_norm,tensors->positions,scratch->key,&key);
	if ( status == SPARK_STATUS_OK && separate_value != 0u )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->qkv,weights->k_norm,tensors->positions,scratch->value,&value);
	memset(&store,0,sizeof(store));
	store.rows = rows;
	store.kv_heads = shape->kv_heads;
	store.head_dimension = shape->head_dimension;
	store.value_dimension = shape->value_dimension;
	store.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	store.pages_per_sequence = stage->pages_per_sequence;
	store.key_stride = kv;
	store.value_stride = separate_value != 0u ? values : fused;
	store.value_offset = separate_value != 0u ? 0u : AttentionValueOffset(shape);
	store.pool_pages = stage->pool_pages;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsKvStore(stage->ops,queue,&paged,scratch->key,separate_value != 0u ? scratch->value : scratch->qkv,&store);
	memset(&attention,0,sizeof(attention));
	attention.rows = rows;
	attention.heads = local_heads;
	attention.first_head = first_head;
	attention.group = shape->heads / shape->kv_heads;
	attention.kv_heads = shape->kv_heads;
	attention.head_dimension = shape->head_dimension;
	attention.value_dimension = shape->value_dimension;
	attention.has_sinks = shape->sinks;
	attention.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	attention.pages_per_sequence = stage->pages_per_sequence;
	attention.pool_pages = stage->pool_pages;
	attention.query_stride = q;
	attention.gate_stride = shape->gate_projection != 0u ? gate_width : fused;
	attention.gate_head_stride = (shape->gate_projection != 0u ? 1u : 2u) * shape->head_dimension;
	attention.gate_offset = shape->gate_projection != 0u ? 0u : shape->head_dimension + first_head * attention.gate_head_stride;
	attention.output_stride = heads_out;
	attention.scale = shape->scale;
	attention.row_exact = stage->step_exact;
	attention.window = shape->window;
	attention.output_gate = shape->output_gate;
	if ( shape->sinks != 0u )
		sinks = SparkTensorSlice(weights->sinks,(uint64_t)first_head * sizeof(float),(uint64_t)local_heads * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAttention(stage->ops,queue,&paged,SparkTensorSlice(scratch->query,(uint64_t)first_head * shape->head_dimension * sizeof(uint16_t),
			scratch->query.bytes - (uint64_t)first_head * shape->head_dimension * sizeof(uint16_t)),shape->gate_projection != 0u ? scratch->gate : scratch->qkv,
			shape->sinks != 0u ? &sinks : 0,scratch->records,scratch->gated,&attention,tensors->host_positions,tensors->host_sequence_of_row);
	if ( status == SPARK_STATUS_OK && shape->head_gate != 0u )
	{
		const uint32_t gate_rows = SparkDeviceAttentionHeadGateRows(shape,degree);
		SparkHeadGateParams head_gate;
		memset(&head_gate,0,sizeof(head_gate));
		head_gate.rows = rows;
		head_gate.heads = local_heads;
		head_gate.head_dimension = AttentionValueDimension(shape);
		head_gate.gate_stride = gate_rows;
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gate,scratch->gate,gate_rows,rows,hidden,gate_rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsHeadGate(stage->ops,queue,scratch->gated,scratch->gate,&head_gate);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,scratch->gated,heads_out,&weights->out,output,hidden,rows,heads_out,hidden);
	return(status);
}

uint32_t SparkDeviceLatentAttentionSlot(const SparkDeviceLatentAttentionShape *shape)
{
	return(shape->latent + shape->rope);
}

static void LatentRopeParams(const SparkDeviceLatentAttentionShape *shape, SparkHeadRopeParams *params, uint32_t heads, uint32_t input_stride,
	uint32_t input_offset, uint32_t input_head_stride, uint32_t output_stride, uint32_t output_head_stride)
{
	uint32_t pair;
	memset(params,0,sizeof(*params));
	params->heads = heads;
	params->head_dimension = shape->rope;
	params->input_stride = input_stride;
	params->input_offset = input_offset;
	params->input_head_stride = input_head_stride;
	params->output_stride = output_stride;
	params->output_head_stride = output_head_stride;
	params->rope_dimension = shape->rope;
	params->plain = 1u;
	params->interleaved = 1u;
	params->scale = 1.0f;
	for ( pair = 0u; pair < shape->rope / 2u && pair < SPARK_KERNEL_ROPE_PAIRS; pair++ )
		params->inverse_frequency[pair] = (float)(1.0 / pow(shape->rope_theta,(double)(2u * pair) / (double)shape->rope));
}

void SparkDeviceLatentAttentionRopePrepare(const SparkDeviceLatentAttentionShape *shape, uint32_t local_heads, SparkDeviceLatentAttentionRope *rope)
{
	const uint32_t qk = shape->nope + shape->rope, slot = SparkDeviceLatentAttentionSlot(shape);
	LatentRopeParams(shape,&rope->query,local_heads,local_heads * qk,shape->nope,qk,local_heads * slot,slot);
	LatentRopeParams(shape,&rope->key,1u,slot,shape->latent,shape->rope,slot,shape->rope);
}

SparkStatus SparkDeviceLatentAttentionBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceLatentAttentionShape *shape, SparkDeviceLatentAttentionWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden, slot = SparkDeviceLatentAttentionSlot(shape);
	const uint32_t degree = stage->tp_degree, heads = shape->heads / degree;
	SparkStatus status;
	if ( shape->heads % degree != 0u || shape->rope % 2u != 0u || shape->rope / 2u > SPARK_KERNEL_ROPE_PAIRS ||
		slot != SPARK_OPS_ATTENTION_LATENT_KEY_DIM || shape->latent != SPARK_OPS_ATTENTION_LATENT_VALUE_DIM ||
		slot + shape->latent > (uint64_t)2u * stage->shape->kv_heads * stage->shape->kv_head_dimension )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,
			"device latent attention needs whole heads per rank, an even rotary span, the kernel's 576-wide slot over a 512-wide latent and a KV slot within the stage's"));
	status = SparkDeviceStageLayerWeight(stage,weights,layer,"mla_q",(uint64_t)heads * (shape->nope + shape->rope),hidden,&bound->query);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"mla_kv_a",slot,hidden,&bound->kv);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"mla_kv_norm","f32",(uint64_t)shape->latent * 4u,&bound->kv_norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"mla_uk","bf16",(uint64_t)heads * shape->latent * shape->nope * 2u,&bound->key_up);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"mla_uv","bf16",(uint64_t)heads * shape->value_dimension * shape->latent * 2u,&bound->value_up);
	if ( status == SPARK_STATUS_OK && shape->head_gate != 0u )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"mla_gate",heads,hidden,&bound->gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,"mla_out",hidden,(uint64_t)heads * shape->value_dimension,&bound->out);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageNoteKvRead(stage,0u,(slot + shape->latent) * sizeof(uint16_t));
	return(status);
}

static SparkStatus LatentHeads(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor input, uint32_t input_head_stride, uint32_t input_offset,
	SparkTensor weight, SparkTensor output, uint32_t output_head_stride, uint32_t heads, uint32_t rows, uint32_t in, uint32_t out)
{
	const uint64_t weight_bytes = (uint64_t)in * out * sizeof(uint16_t);
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t head;
	for ( head = 0u; status == SPARK_STATUS_OK && head < heads; head++ )
	{
		const uint64_t input_at = ((uint64_t)head * input_head_stride + input_offset) * sizeof(uint16_t), output_at = (uint64_t)head * output_head_stride * sizeof(uint16_t);
		status = SparkDeviceLinear(stage,queue,SparkTensorSlice(input,input_at,input.bytes - input_at),heads * input_head_stride,
			SparkTensorSlice(weight,(uint64_t)head * weight_bytes,weight_bytes),SparkTensorSlice(output,output_at,output.bytes - output_at),
			heads * output_head_stride,rows,in,out);
	}
	return(status);
}

SparkStatus SparkDeviceLatentAttention(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceLatentAttentionShape *shape,
	const SparkDeviceLatentAttentionRope *rope, const SparkDeviceStepTensors *tensors, const SparkDeviceLatentAttentionWeights *weights, uint32_t kv_layer,
	const SparkDeviceLatentAttentionScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows, degree = stage->tp_degree, heads = shape->heads / degree;
	const uint32_t qk = shape->nope + shape->rope, slot = SparkDeviceLatentAttentionSlot(shape), values = heads * shape->value_dimension;
	SparkOpsPaged paged = SparkDeviceStagePaged(stage,tensors,kv_layer);
	SparkHeadRopeParams query = rope->query, key = rope->key;
	SparkNormParams norm;
	SparkKvParams store;
	SparkAttentionParams attention;
	SparkHeadGateParams gate;
	SparkStatus status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->query,scratch->query,heads * qk,rows,hidden,heads * qk);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->kv,scratch->kv,slot,rows,hidden,slot);
	if ( status == SPARK_STATUS_OK && shape->head_gate != 0u )
		status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gate,scratch->gate,heads,rows,hidden,heads);
	if ( status == SPARK_STATUS_OK )
		status = LatentHeads(stage,queue,scratch->query,qk,0u,weights->key_up,scratch->packed,slot,heads,rows,shape->nope,shape->latent);
	query.rows = key.rows = rows;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->query,scratch->query,tensors->positions,
			SparkTensorSlice(scratch->packed,(uint64_t)shape->latent * sizeof(uint16_t),scratch->packed.bytes - (uint64_t)shape->latent * sizeof(uint16_t)),&query);
	memset(&norm,0,sizeof(norm));
	norm.rows = rows;
	norm.dimension = shape->latent;
	norm.input_stride = norm.output_stride = slot;
	norm.epsilon = shape->epsilon;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRmsNormResidual(stage->ops,queue,scratch->kv,0,weights->kv_norm,scratch->kv,&norm);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsHeadRmsNormRope(stage->ops,queue,scratch->kv,scratch->kv,tensors->positions,
			SparkTensorSlice(scratch->kv,(uint64_t)shape->latent * sizeof(uint16_t),scratch->kv.bytes - (uint64_t)shape->latent * sizeof(uint16_t)),&key);
	paged.kv_heads = 1u;
	memset(&store,0,sizeof(store));
	store.rows = rows;
	store.kv_heads = 1u;
	store.head_dimension = slot;
	store.value_dimension = shape->latent;
	store.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	store.pages_per_sequence = stage->pages_per_sequence;
	store.key_stride = store.value_stride = slot;
	store.pool_pages = stage->pool_pages;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsKvStore(stage->ops,queue,&paged,scratch->kv,scratch->kv,&store);
	memset(&attention,0,sizeof(attention));
	attention.rows = rows;
	attention.heads = heads;
	attention.first_head = stage->tp_rank * heads;
	attention.group = shape->heads;
	attention.kv_heads = 1u;
	attention.head_dimension = slot;
	attention.value_dimension = shape->latent;
	attention.page_slots = SPARK_DEVICE_RUNNER_PAGE_SLOTS;
	attention.pages_per_sequence = stage->pages_per_sequence;
	attention.pool_pages = stage->pool_pages;
	attention.query_stride = heads * slot;
	attention.output_stride = heads * shape->latent;
	attention.scale = shape->scale;
	attention.row_exact = stage->step_exact;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsAttention(stage->ops,queue,&paged,scratch->packed,scratch->packed,0,scratch->records,scratch->latent,&attention,tensors->host_positions,
			tensors->host_sequence_of_row);
	if ( status == SPARK_STATUS_OK )
		status = LatentHeads(stage,queue,scratch->latent,shape->latent,0u,weights->value_up,scratch->value,shape->value_dimension,heads,rows,shape->latent,
			shape->value_dimension);
	memset(&gate,0,sizeof(gate));
	gate.rows = rows;
	gate.heads = heads;
	gate.head_dimension = shape->value_dimension;
	gate.gate_stride = heads;
	gate.activation = SPARK_HEAD_GATE_SIGMOID_BF16;
	if ( status == SPARK_STATUS_OK && shape->head_gate != 0u )
		status = SparkOpsHeadGate(stage->ops,queue,scratch->value,scratch->gate,&gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,scratch->value,values,&weights->out,output,hidden,rows,values,hidden);
	return(status);
}

SparkStatus SparkDeviceFfnBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, uint32_t ffn,
	SparkDeviceFfnWeights *bound)
{
	return(SparkDeviceFfnBindNamed(stage,weights,layer,"ffn_gate_up","ffn_down",ffn,bound));
}

SparkStatus SparkDeviceFfnBindNamed(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *gate_up,
	const char *down, uint32_t ffn, SparkDeviceFfnWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	SparkStatus status = SparkDeviceStageLayerWeight(stage,weights,layer,gate_up,2u * (uint64_t)ffn,hidden,&bound->gate_up);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerWeight(stage,weights,layer,down,hidden,ffn,&bound->down);
	return(status);
}

SparkStatus SparkDeviceGatedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, uint32_t ffn, uint32_t activation, float limit, const SparkDeviceStepTensors *tensors,
	const SparkDeviceFfnWeights *weights, const SparkDeviceFfnScratch *scratch, SparkTensor normed, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows;
	SparkSiluMulParams silu;
	SparkStatus status = SparkDeviceLinearWeight(stage,queue,normed,hidden,&weights->gate_up,scratch->gate_up,2u * ffn,rows,hidden,2u * ffn);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&silu,0,sizeof(silu));
	silu.rows = rows;
	silu.dimension = ffn;
	silu.input_stride = 2u * ffn;
	silu.output_stride = ffn;
	silu.limit = limit;
	status = SparkOpsGatedMul(stage->ops,queue,scratch->gate_up,scratch->act,&silu,activation);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceLinearWeight(stage,queue,scratch->act,ffn,&weights->down,output,hidden,rows,ffn,hidden);
	return(status);
}

uint64_t SparkDeviceExpertsScratchBytes(const SparkDeviceStage *stage, const SparkDeviceExpertsShape *shape, uint32_t part)
{
	const uint64_t rows = stage->max_rows, routes = rows * shape->top_k, hidden = stage->shape->hidden;
	const uint64_t bytes[] = { rows * hidden * sizeof(uint16_t), rows * shape->experts * sizeof(float), routes * sizeof(uint32_t),
		routes * sizeof(float), rows * hidden * sizeof(uint16_t), routes * 2u * shape->intermediate * sizeof(uint16_t),
		routes * shape->intermediate * sizeof(uint16_t), routes * hidden * sizeof(uint16_t), (routes + 1u) * hidden * sizeof(uint16_t) };
	return(part < sizeof(bytes) / sizeof(bytes[0]) ? bytes[part] : 0u);
}

static SparkStatus ExpertsLocalBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound)
{
	const uint32_t local = shape->experts / stage->tp_degree;
	const uint64_t hidden = stage->shape->hidden, expert_bytes = (uint64_t)local * shape->intermediate * hidden * sizeof(uint16_t);
	SparkStatus status = SparkDeviceStageLayerTensor(stage,weights,layer,"moe_w1","bf16",expert_bytes,&bound->gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"moe_w3","bf16",expert_bytes,&bound->up);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"moe_w2","bf16",expert_bytes,&bound->down);
	if ( status == SPARK_STATUS_OK )
		SparkDeviceStageNoteRouted(stage,3u * expert_bytes,local,shape->top_k);
	return(status);
}

SparkStatus SparkDeviceExpertsBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	SparkStatus status;
	if ( shape->top_k == 0u || shape->top_k > SPARK_ROUTE_MAX_TOP_K || shape->experts > SPARK_ROUTE_MAX_EXPERTS || shape->intermediate % 8u != 0u ||
		shape->experts % stage->tp_degree != 0u )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,
			"device experts need at most 512 experts split evenly over the ranks, a top-k of at most 16 and an 8-aligned width"));
	status = SparkDeviceStageLayerTensor(stage,weights,layer,"router_proj","bf16",(uint64_t)shape->experts * hidden * sizeof(uint16_t),&bound->router);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"router_scale","f32",hidden * sizeof(float),&bound->router_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"expert_scale","f32",(uint64_t)shape->experts * sizeof(float),&bound->expert_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"pre_ff_norm_2","f32",hidden * sizeof(float),&bound->input_norm);
	if ( status == SPARK_STATUS_OK )
		status = ExpertsLocalBind(stage,weights,layer,shape,bound);
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

static SparkStatus ExpertsGroupedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape, uint32_t rows,
	const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor input, const SparkRouteParams *route)
{
	const uint32_t hidden = stage->shape->hidden, routes = rows * shape->top_k, inter = shape->intermediate, local = route->local_experts;
	const uint64_t table = (uint64_t)routes * sizeof(uint32_t), expert_bytes = (uint64_t)inter * hidden * sizeof(uint16_t);
	uint32_t *experts, *ids, *inverse, *offsets, index, expert, total = 0u;
	float *weights_routed, *weights_sorted;
	SparkSiluMulParams gated;
	SparkGemvParams projection;
	SparkStatus status = SPARK_STATUS_OK;
	if ( stage->route_tables.handle == 0u )
		status = SparkDeviceAllocate(&stage->route_tables,SPARK_DEVICE_SPACE_SHARED_COHERENT,
			(uint64_t)stage->max_rows * SPARK_ROUTE_MAX_TOP_K * 5u * sizeof(uint32_t) + (SPARK_ROUTE_MAX_EXPERTS + 1u) * sizeof(uint32_t),"grouped expert routes");
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceCopy(queue,SparkTensorAt(&stage->route_tables,0u,table).range,scratch->experts.range,table);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceCopy(queue,SparkTensorAt(&stage->route_tables,table,table).range,scratch->weights.range,table);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER);
	if ( status != SPARK_STATUS_OK )
		return(SparkDeviceStageRefuse(stage,status,"cannot read the step's routes back for the grouped experts"));
	experts = (uint32_t *)stage->route_tables.host;
	weights_routed = (float *)(experts + routes);
	ids = (uint32_t *)(weights_routed + routes);
	weights_sorted = (float *)(ids + routes);
	inverse = (uint32_t *)(weights_sorted + routes);
	offsets = inverse + routes;
	memset(offsets,0,(size_t)(local + 1u) * sizeof(uint32_t));
	for ( index = 0u; index < routes; index++ )
		if ( experts[index] < local )
			offsets[experts[index] + 1u]++;
	for ( expert = 0u; expert < local; expert++ )
		offsets[expert + 1u] += offsets[expert];
	total = offsets[local];
	for ( index = 0u; index < routes; index++ )
	{
		inverse[index] = total;
		if ( experts[index] < local )
		{
			const uint32_t slot = offsets[experts[index]]++;
			ids[slot] = index / shape->top_k;
			weights_sorted[slot] = weights_routed[index];
			inverse[index] = slot;
		}
	}
	for ( expert = local; expert > 0u; expert-- )
		offsets[expert] = offsets[expert - 1u];
	offsets[0] = 0u;
	if ( total != 0u )
		status = SparkOpsGather(stage->ops,queue,input,SparkTensorAt(&stage->route_tables,2u * table,(uint64_t)total * sizeof(uint32_t)),scratch->sorted,
			total,hidden,rows);
	memset(&projection,0,sizeof(projection));
	projection.projection = stage->shape->projection;
	for ( expert = 0u; status == SPARK_STATUS_OK && expert < local; expert++ )
	{
		const uint32_t first = offsets[expert], count = offsets[expert + 1u] - first;
		if ( count == 0u )
			continue;
		projection.rows = count;
		projection.input_dimension = projection.input_stride = hidden;
		projection.output_dimension = inter;
		projection.output_stride = 2u * inter;
		status = SparkOpsLinear(stage->ops,queue,SparkTensorSlice(scratch->sorted,(uint64_t)first * hidden * sizeof(uint16_t),(uint64_t)count * hidden * sizeof(uint16_t)),
			SparkTensorSlice(weights->gate,(uint64_t)expert * expert_bytes,expert_bytes),
			SparkTensorSlice(scratch->gate_up,(uint64_t)first * 2u * inter * sizeof(uint16_t),(uint64_t)count * 2u * inter * sizeof(uint16_t)),&projection);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsLinear(stage->ops,queue,SparkTensorSlice(scratch->sorted,(uint64_t)first * hidden * sizeof(uint16_t),(uint64_t)count * hidden * sizeof(uint16_t)),
				SparkTensorSlice(weights->up,(uint64_t)expert * expert_bytes,expert_bytes),
				SparkTensorSlice(scratch->gate_up,((uint64_t)first * 2u + 1u) * inter * sizeof(uint16_t),
					((uint64_t)count * 2u - 1u) * inter * sizeof(uint16_t)),&projection);
	}
	memset(&gated,0,sizeof(gated));
	gated.rows = total;
	gated.dimension = inter;
	gated.input_stride = 2u * inter;
	gated.output_stride = inter;
	gated.limit = shape->limit;
	if ( status == SPARK_STATUS_OK && total != 0u )
		status = SparkOpsGatedMulRouted(stage->ops,queue,scratch->gate_up,SparkTensorAt(&stage->route_tables,3u * table,(uint64_t)total * sizeof(float)),
			scratch->act,&gated,shape->activation);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceFill(queue,SparkTensorSlice(scratch->sorted,(uint64_t)total * hidden * sizeof(uint16_t),(uint64_t)hidden * sizeof(uint16_t)).range,0u,
			(uint64_t)hidden * sizeof(uint16_t));
	for ( expert = 0u; status == SPARK_STATUS_OK && expert < local; expert++ )
	{
		const uint32_t first = offsets[expert], count = offsets[expert + 1u] - first;
		if ( count == 0u )
			continue;
		projection.rows = count;
		projection.input_dimension = projection.input_stride = inter;
		projection.output_dimension = projection.output_stride = hidden;
		status = SparkOpsLinear(stage->ops,queue,SparkTensorSlice(scratch->act,(uint64_t)first * inter * sizeof(uint16_t),(uint64_t)count * inter * sizeof(uint16_t)),
			SparkTensorSlice(weights->down,(uint64_t)expert * expert_bytes,expert_bytes),
			SparkTensorSlice(scratch->sorted,(uint64_t)first * hidden * sizeof(uint16_t),(uint64_t)count * hidden * sizeof(uint16_t)),&projection);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsGather(stage->ops,queue,scratch->sorted,SparkTensorAt(&stage->route_tables,4u * table,table),scratch->out,routes,hidden,total + 1u);
	return(status);
}

static SparkStatus ExpertsRoutedFfn(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape, uint32_t rows,
	const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor input, SparkRouteParams *route)
{
	const uint32_t hidden = stage->shape->hidden, routes = rows * shape->top_k, inter = shape->intermediate;
	SparkGemvParams gate = ExpertsRouted(routes,shape->top_k,hidden,hidden,inter,2u * inter), down = ExpertsRouted(routes,1u,inter,inter,hidden,hidden);
	SparkSiluMulParams gated;
	SparkStatus status = SPARK_STATUS_OK;
	route->local_experts = shape->experts / stage->tp_degree;
	route->first_expert = stage->tp_rank * route->local_experts;
	if ( stage->tp_degree > 1u )
		status = SparkOpsRouteLocal(stage->ops,queue,scratch->experts,route);
	if ( status == SPARK_STATUS_OK && rows > SPARK_OPS_DECODE_MAX_ROWS && stage->step_exact == 0u && scratch->sorted.bytes != 0u )
		return(ExpertsGroupedFfn(stage,queue,shape,rows,weights,scratch,input,route));
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinearRouted(stage->ops,queue,input,weights->gate,scratch->gate_up,scratch->experts,route->local_experts,&gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinearRouted(stage->ops,queue,input,weights->up,
			SparkTensorSlice(scratch->gate_up,(uint64_t)inter * sizeof(uint16_t),scratch->gate_up.bytes - (uint64_t)inter * sizeof(uint16_t)),scratch->experts,
			route->local_experts,&gate);
	memset(&gated,0,sizeof(gated));
	gated.rows = routes;
	gated.dimension = inter;
	gated.input_stride = 2u * inter;
	gated.output_stride = inter;
	gated.limit = shape->limit;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsGatedMulRouted(stage->ops,queue,scratch->gate_up,scratch->weights,scratch->act,&gated,shape->activation);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsLinearRouted(stage->ops,queue,scratch->act,weights->down,scratch->out,scratch->experts,route->local_experts,&down);
	return(status);
}

SparkStatus SparkDeviceExperts(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape, const SparkDeviceStepTensors *tensors,
	uint32_t layer, const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor output)
{
	SparkTensor pinned;
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows;
	SparkBranchNormParams routed;
	SparkNormParams norm;
	SparkRouteParams route;
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
		status = ExpertsRoutedFfn(stage,queue,shape,rows,weights,scratch,scratch->normed,&route);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRouteCombine(stage->ops,queue,scratch->out,output,&route);
	return(status);
}

SparkStatus SparkDeviceSigmoidExpertsBind(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer,
	const SparkDeviceExpertsShape *shape, SparkDeviceExpertsWeights *bound)
{
	const uint64_t hidden = stage->shape->hidden;
	SparkStatus status;
	if ( shape->top_k == 0u || shape->top_k > SPARK_ROUTE_MAX_TOP_K || shape->experts > SPARK_ROUTE_MAX_EXPERTS || shape->intermediate % 8u != 0u ||
		shape->experts % stage->tp_degree != 0u )
		return(SparkDeviceStageRefuse(stage,SPARK_STATUS_UNSUPPORTED,
			"device experts need at most 512 experts split evenly over the ranks, a top-k of at most 16 and an 8-aligned width"));
	status = SparkDeviceStageLayerTensor(stage,weights,layer,shape->router != 0 ? shape->router : "router_proj","bf16",(uint64_t)shape->experts * hidden * sizeof(uint16_t),
		&bound->router);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageLayerTensor(stage,weights,layer,"expert_bias","f32",(uint64_t)shape->experts * sizeof(float),&bound->bias);
	if ( status == SPARK_STATUS_OK )
		status = ExpertsLocalBind(stage,weights,layer,shape,bound);
	return(status);
}

SparkStatus SparkDeviceSigmoidExperts(SparkDeviceStage *stage, SparkDeviceQueue *queue, const SparkDeviceExpertsShape *shape,
	const SparkDeviceStepTensors *tensors, const SparkDeviceExpertsWeights *weights, const SparkDeviceExpertsScratch *scratch, SparkTensor normed,
	SparkTensor shared, SparkTensor output)
{
	const uint32_t hidden = stage->shape->hidden, rows = tensors->rows;
	SparkGemvParams router;
	SparkRouteParams route;
	SparkStatus status;
	memset(&router,0,sizeof(router));
	router.rows = rows;
	router.input_dimension = router.input_stride = hidden;
	router.output_dimension = router.output_stride = shape->experts;
	router.output_f32 = 1u;
	router.projection = stage->shape->projection;
	router.row_exact = stage->step_exact;
	status = SparkOpsLinear(stage->ops,queue,normed,weights->router,scratch->logits,&router);
	memset(&route,0,sizeof(route));
	route.rows = rows;
	route.experts = shape->experts;
	route.top_k = shape->top_k;
	route.width = hidden;
	route.logit_stride = shape->experts;
	route.scale = shape->routed_scale;
	route.round_logits = shape->round_logits;
	route.groups = shape->groups;
	route.topk_groups = shape->topk_groups;
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRouteSigmoid(stage->ops,queue,scratch->logits,weights->bias,scratch->experts,scratch->weights,&route);
	if ( status == SPARK_STATUS_OK )
		status = ExpertsRoutedFfn(stage,queue,shape,rows,weights,scratch,normed,&route);
	if ( status == SPARK_STATUS_OK )
		status = SparkOpsRouteCombineShared(stage->ops,queue,scratch->out,shared,output,&route);
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

SparkStatus SparkDeviceLinearWeight(SparkDeviceStage *stage, SparkDeviceQueue *queue, SparkTensor x, uint32_t x_stride, const SparkDeviceWeight *w,
	SparkTensor y, uint32_t y_stride, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension)
{
	SparkGemvParams params;
	if ( w->scale.bytes == 0u )
		return(SparkDeviceLinear(stage,queue,x,x_stride,w->values,y,y_stride,rows,input_dimension,output_dimension));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.input_dimension = input_dimension;
	params.output_dimension = output_dimension;
	params.input_stride = x_stride;
	params.output_stride = y_stride;
	params.projection = stage->shape->projection;
	params.row_exact = stage->step_exact;
	params.scale_stride = (input_dimension + SPARK_FP8_SCALE_K - 1u) / SPARK_FP8_SCALE_K;
	return(SparkOpsLinearFp8(stage->ops,queue,x,w->values,w->scale,stage->dequant,y,&params));
}
