
#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/qwen_3_6/layer.cuh"

struct Qwen38_27bLayerWeights
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *gdn_in_weight;
	const void *gdn_in_scale;
	const void *gdn_conv_weight;
	const void *gdn_out_weight;
	const void *gdn_out_scale;
	const void *gdn_beta_weight;
	const void *gdn_beta_scale;
	const void *gdn_decay_weight;
	const void *gdn_decay_scale;
	const float *gdn_a_log;
	const float *gdn_dt_bias;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
};

static void Qwen38_27bBindLayer(const Qwen38_27bLayerWeights *weights, Qwen38_27bLayerBuffers *buffers)
{
	buffers->attn_norm_weight = weights->attn_norm_weight;
	buffers->mlp_norm_weight = weights->mlp_norm_weight;
	buffers->qkv_weight = weights->qkv_weight;
	buffers->qkv_scale = weights->qkv_scale;
	buffers->output_weight = weights->output_weight;
	buffers->output_scale = weights->output_scale;
	buffers->gdn_in_weight = weights->gdn_in_weight;
	buffers->gdn_in_scale = weights->gdn_in_scale;
	buffers->gdn_conv_weight = weights->gdn_conv_weight;
	buffers->gdn_out_weight = weights->gdn_out_weight;
	buffers->gdn_out_scale = weights->gdn_out_scale;
	buffers->gdn_beta_weight = weights->gdn_beta_weight;
	buffers->gdn_beta_scale = weights->gdn_beta_scale;
	buffers->gdn_decay_weight = weights->gdn_decay_weight;
	buffers->gdn_decay_scale = weights->gdn_decay_scale;
	buffers->gdn_a_log = weights->gdn_a_log;
	buffers->gdn_dt_bias = weights->gdn_dt_bias;
	buffers->dense_gate_up_weight = weights->dense_gate_up_weight;
	buffers->dense_gate_up_scale = weights->dense_gate_up_scale;
	buffers->dense_down_weight = weights->dense_down_weight;
	buffers->dense_down_scale = weights->dense_down_scale;
}

template<class Format>
static int32_t Qwen38_27bLaunchAttentionHalf(const Qwen38_27bLayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)QWEN38_27B_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(Qwen38_27bLayerLinear<Format>(buffers,rows,multiprocessors,stream));
	case LM_LAYER_FULL:
		return(Qwen38_27bLayerAttention<Format,Qwen38_27bFullKv>(buffers,rows,context,
			multiprocessors,stream));
	case LM_LAYER_WINDOW:
	case LM_LAYER_SPARSE:
	case LM_LAYER_COMPRESSED:
	case LM_LAYER_LATENT:
	case LM_LAYER_KIND_COUNT:
	default:
		return(LM_LAUNCH_ERR_SHAPE);
	}
}

template<class Format>
static int32_t Qwen38_27bLaunchSlice(const Qwen38_27bLayerWeights *weights, Qwen38_27bLayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if (layer >= QWEN38_27B_LAYERS)
			return(LM_LAUNCH_ERR_SHAPE);
		Qwen38_27bBindLayer(&weights[offset],buffers);
		status = Qwen38_27bLaunchAttentionHalf<Format>(buffers,layer,rows,context,
			multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
		status = Qwen38_27bLayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t Qwen38_27bStageSlice(const void *layer_weights, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(Qwen38_27bLaunchSlice<LmBf16Format>(
		(const Qwen38_27bLayerWeights *)layer_weights,
		(Qwen38_27bLayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,context,multiprocessors,
		(cudaStream_t)stream));
}
