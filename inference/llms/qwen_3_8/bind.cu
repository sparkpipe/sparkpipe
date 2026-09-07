
#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/qwen_3_8/layer.cuh"

struct Qwen38LayerWeights
{
	const void *attn_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *query_norm_weight;
	const void *key_norm_weight;
	const void *gdn_in_weight;
	const void *gdn_in_scale;
	const void *gdn_conv_weight;
	const void *gdn_out_weight;
	const void *gdn_out_scale;
	const void *gdn_norm_weight;
	const void *gdn_z_weight;
	const void *gdn_z_scale;
	const void *gdn_beta_weight;
	const void *gdn_beta_scale;
	const void *gdn_decay_weight;
	const void *gdn_decay_scale;
	const float *gdn_a_log;
	const float *gdn_dt_bias;
	const void *mlp_norm_weight;
	const void *router_weight;
	const void *router_scale;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	const void *shared_gate_up_weight;
	const void *shared_gate_up_scale;
	const void *shared_down_weight;
	const void *shared_down_scale;
	const void *shared_gate_coeff;
};

static void Qwen38BindLayer(const Qwen38LayerWeights *weights, Qwen38LayerBuffers *buffers)
{
	buffers->attn_norm_weight = weights->attn_norm_weight;
	buffers->qkv_weight = weights->qkv_weight;
	buffers->qkv_scale = weights->qkv_scale;
	buffers->output_weight = weights->output_weight;
	buffers->output_scale = weights->output_scale;
	buffers->query_norm_weight = weights->query_norm_weight;
	buffers->key_norm_weight = weights->key_norm_weight;
	buffers->gdn_in_weight = weights->gdn_in_weight;
	buffers->gdn_in_scale = weights->gdn_in_scale;
	buffers->gdn_conv_weight = weights->gdn_conv_weight;
	buffers->gdn_out_weight = weights->gdn_out_weight;
	buffers->gdn_out_scale = weights->gdn_out_scale;
	buffers->gdn_norm_weight = weights->gdn_norm_weight;
	buffers->gdn_z_weight = weights->gdn_z_weight;
	buffers->gdn_z_scale = weights->gdn_z_scale;
	buffers->gdn_beta_weight = weights->gdn_beta_weight;
	buffers->gdn_beta_scale = weights->gdn_beta_scale;
	buffers->gdn_decay_weight = weights->gdn_decay_weight;
	buffers->gdn_decay_scale = weights->gdn_decay_scale;
	buffers->gdn_a_log = weights->gdn_a_log;
	buffers->gdn_dt_bias = weights->gdn_dt_bias;
	buffers->mlp_norm_weight = weights->mlp_norm_weight;
	buffers->router_weight = weights->router_weight;
	buffers->router_scale = weights->router_scale;
	buffers->expert_w1_weight = weights->expert_w1_weight;
	buffers->expert_w1_scale = weights->expert_w1_scale;
	buffers->expert_w2_weight = weights->expert_w2_weight;
	buffers->expert_w2_scale = weights->expert_w2_scale;
	buffers->shared_gate_up_weight = weights->shared_gate_up_weight;
	buffers->shared_gate_up_scale = weights->shared_gate_up_scale;
	buffers->shared_down_weight = weights->shared_down_weight;
	buffers->shared_down_scale = weights->shared_down_scale;
	buffers->shared_gate_coeff = weights->shared_gate_coeff;
}

template<class Format>
static int32_t Qwen38LaunchAttentionHalf(const Qwen38LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)QWEN38_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(Qwen38LayerLinear<Format>(buffers,rows,multiprocessors,stream));
	case LM_LAYER_FULL:
		return(Qwen38LayerAttention<Format,Qwen38FullKv>(buffers,rows,context,
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
static int32_t Qwen38LaunchSlice(const Qwen38LayerWeights *weights, Qwen38LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if (layer >= QWEN38_LAYERS)
			return(LM_LAUNCH_ERR_SHAPE);
		Qwen38BindLayer(&weights[offset],buffers);
		status = Qwen38LaunchAttentionHalf<Format>(buffers,layer,rows,context,
			multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
		status = Qwen38LayerMoe<Format>(buffers,rows,multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t Qwen38StageSlice(const void *layer_weights, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(Qwen38LaunchSlice<LmBf16Format>(
		(const Qwen38LayerWeights *)layer_weights,
		(Qwen38LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,context,multiprocessors,
		(cudaStream_t)stream));
}
