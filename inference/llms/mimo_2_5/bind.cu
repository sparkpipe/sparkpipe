
#include "inference/kernels/formats/fp8.cuh"
#include "inference/llms/mimo_2_5/layer.cuh"

struct Mimo25LayerWeights
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *router_weight;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
};

static void Mimo25BindLayer(const Mimo25LayerWeights *weights, Mimo25LayerBuffers *buffers)
{
	buffers->attn_norm_weight = weights->attn_norm_weight;
	buffers->mlp_norm_weight = weights->mlp_norm_weight;
	buffers->qkv_weight = weights->qkv_weight;
	buffers->qkv_scale = weights->qkv_scale;
	buffers->output_weight = weights->output_weight;
	buffers->output_scale = weights->output_scale;
	buffers->router_weight = weights->router_weight;
	buffers->expert_w1_weight = weights->expert_w1_weight;
	buffers->expert_w1_scale = weights->expert_w1_scale;
	buffers->expert_w2_weight = weights->expert_w2_weight;
	buffers->expert_w2_scale = weights->expert_w2_scale;
	buffers->dense_gate_up_weight = weights->dense_gate_up_weight;
	buffers->dense_gate_up_scale = weights->dense_gate_up_scale;
	buffers->dense_down_weight = weights->dense_down_weight;
	buffers->dense_down_scale = weights->dense_down_scale;
}

template<class Format, class FullKv, class SwaKv>
static int32_t Mimo25LaunchAttentionHalf(const Mimo25LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)MIMO25_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_FULL:
		return(Mimo25LayerAttention<Format,FullKv,MIMO25_FULL_KV_HEADS,MIMO25_FULL_QKV_DIM>(
			buffers,rows,context,0u,MIMO25_FULL_ROPE_THETA,multiprocessors,stream));
	case LM_LAYER_WINDOW:
		return(Mimo25LayerAttention<Format,SwaKv,MIMO25_SWA_KV_HEADS,MIMO25_SWA_QKV_DIM>(
			buffers,rows,context,MIMO25_SLIDING_WINDOW,MIMO25_SWA_ROPE_THETA,
			multiprocessors,stream));
	case LM_LAYER_SPARSE:
	case LM_LAYER_COMPRESSED:
	case LM_LAYER_LATENT:
	case LM_LAYER_RECURRENT:
	case LM_LAYER_KIND_COUNT:
	default:
		return(LM_LAUNCH_ERR_SHAPE);
	}
}

template<class Format, class FullKv, class SwaKv>
static int32_t Mimo25LaunchSlice(const Mimo25LayerWeights *weights, Mimo25LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if (layer >= MIMO25_LAYERS)
			return(LM_LAUNCH_ERR_SHAPE);
		Mimo25BindLayer(&weights[offset],buffers);
		status = Mimo25LaunchAttentionHalf<Format,FullKv,SwaKv>(buffers,layer,rows,
			context,multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
		if (layer < MIMO25_FIRST_ROUTED_LAYER)
			status = Mimo25LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		else
			status = Mimo25LayerMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t Mimo25StageSlice(const void *layer_weights, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(Mimo25LaunchSlice<LmFp8,Mimo25FullKv,Mimo25SwaKv>(
		(const Mimo25LayerWeights *)layer_weights,
		(Mimo25LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,packed_rows,context,multiprocessors,
		(cudaStream_t)stream));
}
