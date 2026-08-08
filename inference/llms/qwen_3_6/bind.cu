// Bind weights to the layer and run a rank's slice of Qwen 3.6.
//
// glm5_2/bind.cu is the model for this file and its header comment is the
// warning: its author stopped writing it once because the first attempt mapped
// 137 node-context fields onto a layer that did one projection where the model
// does four, and "an adapter over that would have buried them under
// plausible-looking plumbing".
//
// So this file does NOT invent a Qwen node context. glm5_2 binds from
// SparkResidentDecodeStageNodeContext, a struct the host already fills
// because a glm5_2 pack format exists. No qwen pack format exists, and writing
// one from the layer's requirements would be guessing at what a packer will
// produce. The weights arrive here as an explicit per-layer table instead -
// which is what any pack format resolves to anyway, and leaves the packer free
// to disagree with my guess about its layout.
//
// What IS real here is the loop and the dispatch, and that is the half that was
// missing: config.h has declared QWEN36_LAYER_KIND since the layer-kind commit
// and nothing read it.

#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/qwen_3_6/layer.cuh"

// One layer's weights. Every pointer here is a tensor the packer must place;
// naming them in one struct is what lets the loop below be a loop rather than
// 64 special cases.
struct Qwen36LayerWeights
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;
	// Full-attention layers use the fused QKV and output projections. Linear
	// layers use the gated-DeltaNet in and out projections and the convolution.
	// A layer carries one pair or the other, never both, and which is decided
	// by QWEN36_LAYER_KIND rather than by which pointers are non-null - a
	// missing tensor should fail loudly, not silently select the other path.
	const void *qkv_weight;
	const void *qkv_scale;
	const void *output_weight;
	const void *output_scale;
	const void *gdn_in_weight;
	const void *gdn_in_scale;
	const void *gdn_conv_weight;
	const void *gdn_out_weight;
	const void *gdn_out_scale;
	// The gate producers: beta and decay are 48-row projections (the
	// checkpoint's fused in_proj_ba, split), A_log and dt_bias the per-head
	// fp32 tensors of the decay mapping.
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

// Copy one layer's tensors into the buffer struct the kernels read. Each
// assignment is a claim that two names mean the same tensor, which is where a
// wrong one produces fluent output rather than a crash.
static void Qwen36BindLayer(const Qwen36LayerWeights *weights, Qwen36LayerBuffers *buffers)
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

// The attention half, chosen by the layer's kind.
//
// THE DEFAULT RETURNS AN ERROR. deepseek_v4 exported one entry point for three
// kinds and nothing said so, because there was no place where a kind without a
// kernel had to be handled. This switch is that place: a sixth kind added to
// LmLayerKind without an arm here stops the model instead of running the wrong
// one, and the compiler warns about the unhandled enum value first.
template<class Format>
static int32_t Qwen36LaunchAttentionHalf(const Qwen36LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)QWEN36_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(Qwen36LayerLinear<Format>(buffers,rows,multiprocessors,stream));
	case LM_LAYER_FULL:
		return(Qwen36LayerAttention<Format,Qwen36FullKv>(buffers,rows,context,
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

// One stage slice: the layers this rank owns, in order.
//
// The kind is computed from the ABSOLUTE layer index, not the offset into this
// rank's slice. Qwen's period is four and a thirteen-rank split does not divide
// by four, so a rank starting mid-period would otherwise run the wrong kind for
// every layer it owns - and produce fluent output while doing it.
template<class Format>
static int32_t Qwen36LaunchSlice(const Qwen36LayerWeights *weights, Qwen36LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if (layer >= QWEN36_LAYERS)
			return(LM_LAUNCH_ERR_SHAPE);
		Qwen36BindLayer(&weights[offset],buffers);
		status = Qwen36LaunchAttentionHalf<Format>(buffers,layer,rows,context,
			multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
		// Every layer has the same dense SwiGLU. This configuration has no
		// routed experts, so unlike glm5_2 there is no second branch here.
		status = Qwen36LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		if (status != LM_LAUNCH_OK)
			return(status);
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t Qwen36StageSlice(const void *layer_weights, void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t context, uint32_t multiprocessors, void *stream)
{
	return(Qwen36LaunchSlice<LmBf16Format>(
		(const Qwen36LayerWeights *)layer_weights,
		(Qwen36LayerBuffers *)layer_buffers,
		first_layer,layer_count,rows,context,multiprocessors,
		(cudaStream_t)stream));
}
