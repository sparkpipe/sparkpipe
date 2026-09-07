#pragma once
#include "inference/llms/kimi_k3/layer.cuh"
#include "inference/llms/kimi_k3/dspark.h"

struct K3LayerWeights
{
	const void *attn_norm_weight;
	const void *mlp_norm_weight;

	const void *kda_qkv_beta_weight;
	const void *kda_decay_down_weight;
	const float *kda_q_conv_weight;
	const float *kda_k_conv_weight;
	const float *kda_v_conv_weight;
	const void *kda_decay_up_weight;
	const float *kda_decay_bias;
	const float *kda_head_log_scale;
	const void *kda_gate_weight;
	const float *kda_out_norm_weight;
	const void *kda_out_weight;
	const void *kda_out_scale;

	const void *mla_q_down_weight;
	const void *mla_q_down_scale;
	const void *mla_q_norm_weight;
	const void *mla_q_up_weight;
	const void *mla_q_up_scale;
	const void *mla_kv_a_weight;
	const void *mla_kv_a_scale;
	const void *mla_kv_a_norm_weight;
	const void *mla_kv_b_value_weight;
	const void *mla_kv_b_scale;
	const void *mla_gate_weight;
	const void *mla_out_weight;
	const void *mla_out_scale;

	const void *router_weight;
	const void *routed_down_weight;
	const void *routed_down_scale;
	const void *routed_up_weight;
	const void *routed_up_scale;
	const void *routed_norm_weight;
	const void *expert_w1_weight;
	const void *expert_w2_weight;
	uint32_t expert_interleave;
	uint32_t expert_tile_k;
	const void *shared_w1_weight;
	const void *shared_w1_scale;
	const void *shared_w2_weight;
	const void *shared_w2_scale;
	const void *dense_gate_up_weight;
	const void *dense_gate_up_scale;
	const void *dense_down_weight;
	const void *dense_down_scale;
	const void *attnres_attn_weight;
	const void *attnres_mlp_weight;
};

struct K3SliceState
{
	uint8_t *kda_state;
	uint16_t *kda_q_window;
	uint16_t *kda_k_window;
	uint16_t *kda_v_window;
	const LmKvView *mla_cache;
	uint16_t *replay_conv_q;
	uint16_t *replay_conv_k;
	uint16_t *replay_conv_v;
	float *replay_retention;
	float *replay_write_gate;
	uint32_t verify_rows;
	uint16_t *dspark_aux;
	uint32_t aux_rows;
	uint32_t sequences;
	uint32_t kda_state_bf16;
	void (*layer_collective)(void *context, void *stream, uint32_t layer,
		uint32_t phase);
	void *collective_context;
};

static_assert(K3_KDA_LAYER_COUNT + K3_MLA_LAYER_COUNT == K3_LAYERS,
	"every layer is exactly one of the two kinds");
static_assert((90u - (90u / 4u)) == K3_KDA_LAYER_COUNT - 1u,
	"the last KDA layer must land on the last KDA pool slot");
static_assert(((K3_LAYERS - 1u) / 4u) == K3_MLA_LAYER_COUNT - 1u,
	"the trailing MLA layer must land on the last cache view");

static void K3BindLayer(const K3LayerWeights *weights, K3LayerBuffers *buffers)
{
	buffers->attn_norm_weight = weights->attn_norm_weight;
	buffers->mlp_norm_weight = weights->mlp_norm_weight;
	buffers->kda_qkv_beta_weight = weights->kda_qkv_beta_weight;
	buffers->kda_decay_down_weight = weights->kda_decay_down_weight;
	buffers->kda_q_conv_weight = weights->kda_q_conv_weight;
	buffers->kda_k_conv_weight = weights->kda_k_conv_weight;
	buffers->kda_v_conv_weight = weights->kda_v_conv_weight;
	buffers->kda_decay_up_weight = weights->kda_decay_up_weight;
	buffers->kda_decay_bias = weights->kda_decay_bias;
	buffers->kda_head_log_scale = weights->kda_head_log_scale;
	buffers->kda_gate_weight = weights->kda_gate_weight;
	buffers->kda_out_norm_weight = weights->kda_out_norm_weight;
	buffers->kda_out_weight = weights->kda_out_weight;
	buffers->kda_out_scale = weights->kda_out_scale;
	buffers->mla_q_down_weight = weights->mla_q_down_weight;
	buffers->mla_q_down_scale = weights->mla_q_down_scale;
	buffers->mla_q_norm_weight = weights->mla_q_norm_weight;
	buffers->mla_q_up_weight = weights->mla_q_up_weight;
	buffers->mla_q_up_scale = weights->mla_q_up_scale;
	buffers->mla_kv_a_weight = weights->mla_kv_a_weight;
	buffers->mla_kv_a_scale = weights->mla_kv_a_scale;
	buffers->mla_kv_a_norm_weight = weights->mla_kv_a_norm_weight;
	buffers->mla_kv_b_value_weight = weights->mla_kv_b_value_weight;
	buffers->mla_kv_b_scale = weights->mla_kv_b_scale;
	buffers->mla_gate_weight = weights->mla_gate_weight;
	buffers->mla_out_weight = weights->mla_out_weight;
	buffers->mla_out_scale = weights->mla_out_scale;
	buffers->router_weight = weights->router_weight;
	buffers->routed_down_weight = weights->routed_down_weight;
	buffers->routed_down_scale = weights->routed_down_scale;
	buffers->routed_up_weight = weights->routed_up_weight;
	buffers->routed_up_scale = weights->routed_up_scale;
	buffers->routed_norm_weight = weights->routed_norm_weight;
	buffers->expert_w1_weight = weights->expert_w1_weight;
	buffers->expert_w2_weight = weights->expert_w2_weight;
	buffers->expert_interleave = weights->expert_interleave;
	buffers->expert_tile_k = weights->expert_tile_k;
	buffers->shared_w1_weight = weights->shared_w1_weight;
	buffers->shared_w1_scale = weights->shared_w1_scale;
	buffers->shared_w2_weight = weights->shared_w2_weight;
	buffers->shared_w2_scale = weights->shared_w2_scale;
	buffers->dense_gate_up_weight = weights->dense_gate_up_weight;
	buffers->dense_gate_up_scale = weights->dense_gate_up_scale;
	buffers->dense_down_weight = weights->dense_down_weight;
	buffers->dense_down_scale = weights->dense_down_scale;
	buffers->attnres_attn_weight = weights->attnres_attn_weight;
	buffers->attnres_mlp_weight = weights->attnres_mlp_weight;
}

static void K3BindLayerState(const K3SliceState *state, uint32_t layer, K3LayerBuffers *buffers)
{
	uint32_t mla_index = layer / 4u;
	uint32_t kda_index = layer - mla_index;
	uint64_t sequences = state->sequences;
	uint64_t slot_bytes = state->kda_state_bf16 != 0u
		? (uint64_t)K3_KDA_STATE_SLOT_BYTES_BF16 : (uint64_t)K3_KDA_STATE_SLOT_BYTES;
	buffers->kda_state_bf16 = state->kda_state_bf16;
	buffers->kda_state_pool = state->kda_state
		+ ((uint64_t)kda_index * sequences * slot_bytes);
	buffers->kda_q_window = state->kda_q_window
		+ ((uint64_t)kda_index * sequences * K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL);
	buffers->kda_k_window = state->kda_k_window
		+ ((uint64_t)kda_index * sequences * K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL);
	buffers->kda_v_window = state->kda_v_window
		+ ((uint64_t)kda_index * sequences * K3_KDA_V_DIM * K3_KDA_CONV_KERNEL);
	buffers->replay_conv_q = state->replay_conv_q == 0 ? 0 : state->replay_conv_q
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_QK_DIM);
	buffers->replay_conv_k = state->replay_conv_k == 0 ? 0 : state->replay_conv_k
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_QK_DIM);
	buffers->replay_conv_v = state->replay_conv_v == 0 ? 0 : state->replay_conv_v
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_V_DIM);
	buffers->replay_retention = state->replay_retention == 0 ? 0 : state->replay_retention
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_QK_DIM);
	buffers->replay_write_gate = state->replay_write_gate == 0 ? 0 : state->replay_write_gate
		+ ((uint64_t)kda_index * sequences * state->verify_rows * K3_KDA_HEADS);
	if ( K3_LAYER_KIND(layer) == LM_LAYER_LATENT )
		buffers->cache = state->mla_cache[mla_index];
}

template<class Format, class Geometry>
static int32_t K3LaunchAttentionHalf(const K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *partial_accumulate, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)K3_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(K3LayerKda<Format>(buffers,rows,sequences,commit,partial_accumulate,multiprocessors,stream));
	case LM_LAYER_LATENT:
		return(K3LayerMla<Format,Geometry>(buffers,rows,context,partial_accumulate,multiprocessors,stream));
	case LM_LAYER_FULL:
	case LM_LAYER_WINDOW:
	case LM_LAYER_SPARSE:
	case LM_LAYER_COMPRESSED:
	case LM_LAYER_KIND_COUNT:
	default:
		return(LM_LAUNCH_ERR_SHAPE);
	}
}

template<class Format, class Geometry>
static int32_t K3LaunchSlice(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer,boundary;
	int32_t status;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if ( layer >= K3_LAYERS )
			return(LM_LAUNCH_ERR_SHAPE);
		K3BindLayer(&weights[offset],buffers);
		K3BindLayerState(state,layer,buffers);
		boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u ? 1u : 0u;
		if ( layer > 0u )
			K3AttnRes(buffers,buffers->attnres_attn_weight,
				((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( boundary != 0u )
		{
			if ( layer == 0u )
				K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
			K3BankStore(buffers,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
		}
		status = K3LaunchAttentionHalf<Format,Geometry>(buffers,layer,rows,sequences,commit,
			boundary != 0u ? (uint16_t *)0 : buffers->attnres_partial_bf16,context,
			multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( boundary != 0u && buffers->tp_sharded == 0u )
			K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
		if ( state->layer_collective != 0 )
			state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,0u);
		K3AttnRes(buffers,buffers->attnres_mlp_weight,
			(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( layer < K3_FIRST_ROUTED_LAYER )
			status = K3LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		else
		{
			status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,0u);
			if ( status == LM_LAUNCH_OK && state->layer_collective != 0 )
				state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,2u);
			if ( status == LM_LAUNCH_OK )
				status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,1u);
		}
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( state->layer_collective != 0 )
			state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,1u);
		if ( state->dspark_aux != 0 )
		{
			static const uint32_t aux_ids[K3_DSPARK_AUX_LAYER_COUNT] = K3_DSPARK_AUX_LAYER_IDS_INITIALIZER;
			uint32_t aux;
			for (aux = 0u; aux < K3_DSPARK_AUX_LAYER_COUNT; ++aux)
				if ( aux_ids[aux] == layer )
					LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0, stream,
						buffers->attnres_partial_bf16,state->dspark_aux + ((uint64_t)aux * state->aux_rows * K3_HIDDEN),rows,K3_HIDDEN);
		}
	}
	if ( first_layer + layer_count == K3_LAYERS )
	{
		K3AttnRes(buffers,buffers->attnres_out_weight,
			((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		return(LM_LAUNCH_OK);
	}
	LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
		dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
		K3_LAYER_THREADS, 0, stream,
		buffers->attnres_partial_bf16,buffers->hidden_bf16,rows,K3_HIDDEN);
	return(LM_LAUNCH_OK);
}

template<class Format, class Geometry>
static int32_t K3LaunchSliceHalf(const K3LayerWeights *weights, const K3SliceState *state,
	K3LayerBuffers *buffers, uint32_t layer, uint32_t phase, uint32_t rows,
	uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context,
	uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u ? 1u : 0u;
	if ( phase == 0u )
	{
		K3BindLayer(weights,buffers);
		K3BindLayerState(state,layer,buffers);
		if ( layer > 0u )
			K3AttnRes(buffers,buffers->attnres_attn_weight,
				((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( boundary != 0u )
		{
			if ( layer == 0u )
				K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
			K3BankStore(buffers,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
		}
		status = K3LaunchAttentionHalf<Format,Geometry>(buffers,layer,rows,sequences,commit,
			boundary != 0u ? (uint16_t *)0 : buffers->attnres_partial_bf16,context,
			multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( boundary != 0u && buffers->tp_sharded == 0u )
			K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
		return(LM_LAUNCH_OK);
	}
	if ( layer < K3_FIRST_ROUTED_LAYER )
	{
		K3AttnRes(buffers,buffers->attnres_mlp_weight,
			(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		return(K3LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream));
	}
	if ( phase == 1u )
	{
		K3AttnRes(buffers,buffers->attnres_mlp_weight,
			(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		return(K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,0u));
	}
	return(K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,1u));
}

template<class Format>
static int32_t K3FoldAccepted(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t layer;
	uint64_t replay_capacity;
	int32_t status;
	(void)multiprocessors;
	if ( weights == 0 || state == 0 || buffers == 0
		|| verify_row_begin == 0 || accepted == 0
		|| first_layer > K3_LAYERS || layer_count > K3_LAYERS - first_layer
		|| sequences == 0u || sequences > state->sequences
		|| state->verify_rows == 0u
		|| state->replay_conv_q == 0 || state->replay_conv_k == 0
		|| state->replay_conv_v == 0 || state->replay_retention == 0
		|| state->replay_write_gate == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	replay_capacity = (uint64_t)sequences * state->verify_rows;
	if ( slab_rows == 0u || (uint64_t)slab_rows > replay_capacity )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( state->kda_state_bf16 != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	status = K3DeltaRuleOptIn((uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)));
	if ( status != LM_LAUNCH_OK )
		return(status);
	for (layer = first_layer; layer < first_layer + layer_count; ++layer)
	{
		if ( K3_LAYER_KIND(layer) != LM_LAYER_RECURRENT )
			continue;
		K3BindLayer(&weights[layer - first_layer],buffers);
		K3BindLayerState(state,layer,buffers);
		LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
			buffers->kda_q_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_q,buffers->kda_q_conv_weight,buffers->query_bf16,K3_KDA_QK_DIM,sequences,1u);
		LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_QK_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
			buffers->kda_k_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_k,buffers->kda_k_conv_weight,buffers->key_bf16,K3_KDA_QK_DIM,sequences,1u);
		LM_LAUNCH((LmCausalConvKernel<K3_LAYER_THREADS,K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>), dim3(sequences,(K3_KDA_V_DIM + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS), K3_LAYER_THREADS, 0, stream,
			buffers->kda_v_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_v,buffers->kda_v_conv_weight,buffers->value_bf16,K3_KDA_V_DIM,sequences,1u);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(slab_rows,K3_KDA_HEADS), K3_LAYER_THREADS, 0, stream,
			buffers->key_bf16,K3_KDA_HEADS,slab_rows,K3_RMS_EPSILON);
		LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM>), dim3(sequences,K3_KDA_HEADS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float)), stream,
			buffers->kda_state_pool,state->kda_state_bf16 != 0u ? K3_KDA_STATE_SLOT_BYTES_BF16 : K3_KDA_STATE_SLOT_BYTES,buffers->kda_state_index,verify_row_begin,accepted,buffers->query_bf16,buffers->key_bf16, buffers->value_bf16,buffers->replay_retention,buffers->replay_write_gate,buffers->attention_out_bf16, K3_KDA_HEADS,1u,sequences,1u);
	}
	return(LM_LAUNCH_OK);
}
