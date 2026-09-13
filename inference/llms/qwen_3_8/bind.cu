// Bind weights to the layer and run a rank's slice of Qwen 3.8 Max.
//
// qwen_3_6/bind.cu is the model for this file: no invented node context, an
// explicit per-layer table the packer must fill, and a dispatch that reads
// QWEN38_LAYER_KIND rather than guessing from which pointers are non-null.
// What is new here is the MoE: EVERY layer has one, so the slice loop runs
// the attention half, then Qwen38LayerMoe, then the residual bookkeeping the
// layer functions share with qwen_3_6's dense path.
//
// Pack contract the bind pins (mirrors the module's pack):
//   - routed experts arrive expert-major stacked FP8_E4M3 with block-128
//     F32 scales (w1 = fused gate|up, 2 x intermediate per expert; w2 =
//     hidden per expert), the vendor FP8 checkpoint's layout verbatim;
//   - shared_expert.gate_proj and up_proj are fused at pack time into one
//     [2 x intermediate, hidden] BF16 tensor;
//   - everything else (norms, qkv, GDN, router, shared down, head) is BF16.

#include "inference/kernels/formats/bf16.cuh"
#include "inference/llms/qwen_3_8/layer.cuh"

// One layer's weights. Every pointer is a tensor the packer must place.
struct Qwen38LayerWeights
{
	const void *attn_norm_weight;
	// Full-attention layers use the fused QKV and output projections and the
	// per-head q/k norm weights. Linear layers use the GDN in/out
	// projections, the conv, and the gate producers. A layer carries one
	// pair or the other, never both - QWEN38_LAYER_KIND decides.
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
	// The MoE: router gate, routed experts (stacked FP8 + block-128 F32
	// scales), shared expert (fused gate|up + down, BF16), and the learned
	// per-channel shared gate.
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

// Copy one layer's tensors into the buffer struct the kernels read. Each
// assignment claims two names mean the same tensor; a wrong one produces
// fluent output rather than a crash, so the names are the pack's names.
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

// The attention half, chosen by the layer's kind. Same shape as qwen_3_6:
// a sixth kind without an arm here stops the model instead of running the
// wrong one.
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

// One stage slice: the layers this rank owns, in order. The kind is computed
// from the ABSOLUTE layer index (the period does not divide a rank slice),
// and every layer runs the same routed MoE.
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
