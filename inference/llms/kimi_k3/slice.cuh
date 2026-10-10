#pragma once
#include <stdio.h>
#include "inference/llms/kimi_k3/layer.cuh"
#include "inference/llms/kimi_k3/dspark.h"
#include "inference/kernels/l2_load.cuh"

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
	const float *router_bias;
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

#define K3_COLLECTIVE_MLA_QUERY 4u
#define K3_COLLECTIVE_MLA_PARTIALS 5u
#define K3_COLLECTIVE_MLA_DOWN 6u
#define K3_COLLECTIVE_MLA_KEYS 7u
#define K3_COLLECTIVE_FINISH 0x100u
#define K3_COLLECTIVE_BEGIN 0x200u
#define K3_L2_LOAD_BYTES (8u << 20)
#define K3_L2_LOAD_BLOCKS 48u

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
	uint32_t kda_rank_heads;
	uint32_t kda_state_bf16;
	uint32_t first_mla_index;
	uint32_t first_kda_index;
	void (*layer_collective)(void *context, void *stream, uint32_t layer,
		uint32_t phase);
	uint32_t (*collective_published)(void *context);
	void *collective_context;
	void *collective_pair_context;
	void *pair_stream_a;
	void *pair_stream_b;
	void *pair_fence;
	void *load_stream;
	void *load_fork;
	void *load_join;
	void *lazy_context;
	int32_t (*lazy_acquire)(void *context, uint32_t layer, void *buffers);
	void (*lazy_release)(void *context, uint32_t layer);
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
	buffers->router_bias = weights->router_bias;
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
	uint32_t mla_index = (layer / 4u) - state->first_mla_index;
	uint32_t kda_index = (layer - (layer / 4u)) - state->first_kda_index;
	uint64_t sequences = state->sequences;
	uint64_t heads = state->kda_rank_heads;
	uint64_t slot_bytes = K3_KDA_RANK_STATE_SLOT_BYTES(state->kda_rank_heads,
		state->kda_state_bf16);
	buffers->kda_state_bf16 = state->kda_state_bf16;
	buffers->kda_state_pool = state->kda_state
		+ ((uint64_t)kda_index * sequences * slot_bytes);
	buffers->kda_q_window = state->kda_q_window
		+ ((uint64_t)kda_index * sequences * heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL);
	buffers->kda_k_window = state->kda_k_window
		+ ((uint64_t)kda_index * sequences * heads * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL);
	buffers->kda_v_window = state->kda_v_window
		+ ((uint64_t)kda_index * sequences * heads * K3_KDA_VALUE_DIM * K3_KDA_CONV_KERNEL);
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
	{
		buffers->cache = state->mla_cache[mla_index];
		buffers->cache_shard.pages = buffers->cache;
		buffers->cache_shard.shard = buffers->kv_shard;
	}
}

template<class Format, class Geometry>
static int32_t K3LaunchMlaGather(const K3SliceState *state, const K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status = K3LayerMlaQuery<Format,Geometry>(buffers,rows,multiprocessors,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerMlaGatherPack<Geometry>(buffers,rows,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,K3_COLLECTIVE_MLA_KEYS);
	status = K3LayerMlaGate(buffers,rows,multiprocessors,stream);
	state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,K3_COLLECTIVE_MLA_KEYS | K3_COLLECTIVE_FINISH);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerMlaGatherPartials<Geometry>(buffers,rows,stream);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerMlaShardMerge(buffers,rows,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerMlaOutput<Format>(buffers,rows,partial_accumulate,1u,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LaunchMlaShard(const K3SliceState *state, const K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint16_t *partial_accumulate, uint32_t multiprocessors, cudaStream_t stream)
{
	int32_t status;
	if ( state->layer_collective == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( buffers->shard_gather != 0u )
		return(K3LaunchMlaGather<Format,Geometry>(state,buffers,layer,rows,partial_accumulate,multiprocessors,stream));
	if ( K3_MLA_DOWN_SLICED(buffers,rows) )
	{
		status = K3LayerMlaDownSlice(buffers,rows,stream);
		if ( status != LM_LAUNCH_OK )
			return(status);
		state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,K3_COLLECTIVE_MLA_DOWN);
	}
	else
		status = K3LayerMlaDown<Format>(buffers,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3LayerMlaUp<Format,Geometry>(buffers,rows,multiprocessors,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,K3_COLLECTIVE_MLA_QUERY);
	status = K3LayerMlaGate(buffers,rows,multiprocessors,stream);
	state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,K3_COLLECTIVE_MLA_QUERY | K3_COLLECTIVE_FINISH);
	if ( status != LM_LAUNCH_OK )
		return(status);
	status = K3LayerMlaShardPartials<Geometry>(buffers,rows,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,K3_COLLECTIVE_MLA_PARTIALS);
	status = K3LayerMlaShardMerge(buffers,rows,stream);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerMlaOutput<Format>(buffers,rows,partial_accumulate,1u,multiprocessors,stream));
}

template<class Format, class Geometry>
static int32_t K3LaunchAttentionHalf(const K3SliceState *state, const K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t sequences, uint32_t commit, uint16_t *partial_accumulate, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	enum LmLayerKind kind = (enum LmLayerKind)K3_LAYER_KIND(layer);
	switch (kind)
	{
	case LM_LAYER_RECURRENT:
		return(K3LayerKda<Format>(buffers,rows,sequences,commit,partial_accumulate,multiprocessors,stream));
	case LM_LAYER_LATENT:
		if ( buffers->kv_shard.degree > 1u )
			return(K3LaunchMlaShard<Format,Geometry>(state,buffers,layer,rows,partial_accumulate,multiprocessors,stream));
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


static void K3L2PlanMoe(const K3LayerBuffers *b, uint32_t layer, LmL2LoadPlan *plan)
{
	const uint64_t row = (uint64_t)K3_HIDDEN * sizeof(uint16_t);
	const uint32_t moe_in = K3_RANK_DIM(b,routed_down_rows,K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t experts = K3_EXPERT_CELLS(b) ? K3_EXPERTS / (K3_ROUTED_EXPERT_HIDDEN / moe_in) : K3_EXPERTS;
	LmL2LoadBegin(plan,K3_L2_LOAD_BYTES);
	LmL2LoadAdd(plan,b->mlp_norm_weight,row);
	if ( layer < K3_FIRST_ROUTED_LAYER )
	{
		LmL2LoadAdd(plan,b->dense_gate_up_weight,(uint64_t)K3_RANK_DIM(b,dense_gate_up_rows,K3_DENSE_INTERMEDIATE * 2u) * row);
		return;
	}
	LmL2LoadAdd(plan,(const uint8_t *)b->router_weight + (K3_EXPERT_CELLS(b) ? (uint64_t)b->tp_rank * experts * row : 0u),(uint64_t)experts * row);
	LmL2LoadAdd(plan,b->routed_down_weight,(uint64_t)moe_in * row);
	LmL2LoadAdd(plan,b->shared_w1_weight,(uint64_t)K3_RANK_DIM(b,shared_w1_rows,K3_SHARED_INTERMEDIATE * 2u) * row);
}

static void K3L2PlanAttention(const K3LayerBuffers *b, uint32_t layer, LmL2LoadPlan *plan)
{
	const uint64_t row = (uint64_t)K3_HIDDEN * sizeof(uint16_t);
	LmL2LoadBegin(plan,K3_L2_LOAD_BYTES);
	LmL2LoadAdd(plan,b->attn_norm_weight,row);
	if ( K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT )
		LmL2LoadAdd(plan,b->kda_qkv_beta_weight,(uint64_t)K3_RANK_DIM(b,kda_qkvb_rows,K3_KDA_QKVB_FUSED_ROWS) * row);
	else
		LmL2LoadAdd(plan,b->mla_q_down_weight,(uint64_t)K3_Q_LORA_RANK * row);
}

static int32_t K3L2LoadRound(const K3SliceState *state, uint32_t layer, uint32_t phase, const LmL2LoadPlan *plan, cudaStream_t stream)
{
	void *stream_word = (void *)(uintptr_t)stream;
	cudaStream_t side = (cudaStream_t)(uintptr_t)state->load_stream;
	uint32_t forked = 0u;
	cudaError_t error = cudaSuccess;
	state->layer_collective(state->collective_context,stream_word,layer,phase | K3_COLLECTIVE_BEGIN);
	if ( plan != 0 && plan->count != 0u && side != 0 && state->collective_published != 0 &&
		state->collective_published(state->collective_context) != 0u )
	{
		error = cudaEventRecord((cudaEvent_t)state->load_fork,stream);
		if ( error == cudaSuccess )
			error = cudaStreamWaitEvent(side,(cudaEvent_t)state->load_fork,0u);
		if ( error == cudaSuccess )
			error = LmL2LoadLaunch(plan,K3_L2_LOAD_BLOCKS,side);
		if ( error == cudaSuccess )
			error = cudaEventRecord((cudaEvent_t)state->load_join,side);
		forked = error == cudaSuccess ? 1u : 0u;
	}
	state->layer_collective(state->collective_context,stream_word,layer,phase | K3_COLLECTIVE_FINISH);
	if ( forked != 0u )
		error = cudaStreamWaitEvent(stream,(cudaEvent_t)state->load_join,0u);
	return(error == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<class Format>
static int32_t K3LaunchCellsMoe(const K3SliceState *state, K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, uint32_t packed_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	void *stream_word = (void *)(uintptr_t)stream;
	int32_t status;
	if ( state->layer_collective == 0 || buffers->tp_sharded == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,0u);
	if ( status != LM_LAUNCH_OK )
		return(status);
	state->layer_collective(state->collective_context,stream_word,layer,2u);
	status = K3LayerSharedUp<Format>(buffers,rows,buffers->shared_mid_bf16,multiprocessors,stream);
	state->layer_collective(state->collective_context,stream_word,layer,2u | K3_COLLECTIVE_FINISH);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerMoeSelect(buffers,rows,packed_rows,stream);
	if ( status == LM_LAUNCH_OK && state->lazy_acquire != 0 )
		status = state->lazy_acquire(state->lazy_context,layer,(void *)buffers);
	if ( status == LM_LAUNCH_OK )
		status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,1u);
	if ( status != LM_LAUNCH_OK )
		return(status);
	state->layer_collective(state->collective_context,stream_word,layer,3u);
	status = K3LayerSharedDown<Format>(buffers,rows,buffers->shared_mid_bf16,multiprocessors,stream);
	state->layer_collective(state->collective_context,stream_word,layer,3u | K3_COLLECTIVE_FINISH);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,2u));
}

static int32_t K3SliceFailure(uint32_t layer,const char *phase,int32_t status)
{
	fprintf(stderr,"k3 slice failed layer=%u phase=%s status=%d\n",layer,phase,status);
	return(status);
}

template<class Format, class Geometry>
static int32_t K3LaunchSlice(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t offset,layer,boundary;
	int32_t status;
	LmL2LoadPlan load;
	for (offset = 0u; offset < layer_count; ++offset)
	{
		layer = first_layer + offset;
		if ( layer >= K3_LAYERS )
			return(K3SliceFailure(layer,"shape",LM_LAUNCH_ERR_SHAPE));
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
		status = K3LaunchAttentionHalf<Format,Geometry>(state,buffers,layer,rows,sequences,commit,
			boundary != 0u ? (uint16_t *)0 : buffers->attnres_partial_bf16,context,
			multiprocessors,stream);
		if ( status != LM_LAUNCH_OK )
			return(K3SliceFailure(layer,"attention",status));
		if ( boundary != 0u && buffers->tp_sharded == 0u )
			K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
		if ( state->layer_collective != 0 )
		{
			K3L2PlanMoe(buffers,layer,&load);
			status = K3L2LoadRound(state,layer,0u,&load,stream);
			if ( status != LM_LAUNCH_OK )
				return(K3SliceFailure(layer,"attention-round",status));
		}
		K3AttnRes(buffers,buffers->attnres_mlp_weight,
			(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
		if ( layer < K3_FIRST_ROUTED_LAYER )
			status = K3LayerDenseMlp<Format>(buffers,rows,multiprocessors,stream);
		else if ( K3_EXPERT_CELLS(buffers) )
			status = K3LaunchCellsMoe<Format>(state,buffers,layer,rows,packed_rows,multiprocessors,stream);
		else if ( state->lazy_acquire != 0 )
		{
			status = K3LayerMoeRoute<Format>(buffers,rows,packed_rows,multiprocessors,stream);
			if ( status == LM_LAUNCH_OK )
				status = state->lazy_acquire(state->lazy_context,layer,(void *)buffers);
			if ( status == LM_LAUNCH_OK )
				status = K3LayerMoeWeighted<Format>(buffers,rows,packed_rows,multiprocessors,stream,0u);
			if ( status == LM_LAUNCH_OK && state->layer_collective != 0 )
				state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,2u);
			if ( status == LM_LAUNCH_OK )
				status = K3LayerMoeWeighted<Format>(buffers,rows,packed_rows,multiprocessors,stream,1u);
			if ( status == LM_LAUNCH_OK && buffers->tp_sharded != 0u )
			{
				if ( state->layer_collective == 0 )
					status = LM_LAUNCH_ERR_SHAPE;
				else
				{
					state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,3u);
					status = K3LayerMoeWeighted<Format>(buffers,rows,packed_rows,multiprocessors,stream,2u);
				}
			}
		}
		else
		{
			status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,0u);
			if ( status == LM_LAUNCH_OK && state->layer_collective != 0 )
				state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,2u);
			if ( status == LM_LAUNCH_OK )
				status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,1u);
			if ( status == LM_LAUNCH_OK && buffers->tp_sharded != 0u )
			{
				if ( state->layer_collective == 0 )
					status = LM_LAUNCH_ERR_SHAPE;
				else
				{
					state->layer_collective(state->collective_context,(void *)(uintptr_t)stream,layer,3u);
					status = K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,2u);
				}
			}
		}
		if ( status != LM_LAUNCH_OK )
			return(K3SliceFailure(layer,layer < K3_FIRST_ROUTED_LAYER ? "dense" : state->lazy_acquire != 0 ? "routed-lazy" : "routed",status));
		if ( state->layer_collective != 0 )
		{
			load.count = 0u;
			if ( offset + 1u < layer_count )
			{
				K3LayerBuffers next = *buffers;
				K3BindLayer(&weights[offset + 1u],&next);
				K3L2PlanAttention(&next,layer + 1u,&load);
			}
			status = K3L2LoadRound(state,layer,1u,&load,stream);
			if ( status != LM_LAUNCH_OK )
				return(K3SliceFailure(layer,"mlp-round",status));
		}
		if ( state->lazy_release != 0 )
			state->lazy_release(state->lazy_context,layer);
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

static void K3PairPre(K3LayerBuffers *buffers, uint32_t layer, uint32_t rows, cudaStream_t stream)
{
	if ( layer > 0u )
		K3AttnRes(buffers,buffers->attnres_attn_weight,
			((layer - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows,stream);
	if ( (layer % K3_ATTNRES_BLOCK_SIZE) == 0u )
	{
		if ( layer == 0u )
			K3PartialSet(buffers,buffers->hidden_bf16,rows,stream);
		K3BankStore(buffers,layer / K3_ATTNRES_BLOCK_SIZE,rows,stream);
	}
}

static uint32_t K3PairReady(const K3SliceState *state, const K3LayerBuffers *a, const K3LayerBuffers *b, uint32_t rows_a, uint32_t rows_b)
{
	return(state->layer_collective != 0 && state->collective_pair_context != 0 && state->dspark_aux == 0 &&
		state->pair_stream_a != 0 && state->pair_stream_b != 0 && state->pair_fence != 0 &&
		a->tp_sharded != 0u && b->tp_sharded != 0u && K3_EXPERT_CELLS(a) && a->kv_shard.degree > 1u &&
		K3LayerMlaGatherReady(a,rows_a) != 0u && K3LayerMlaGatherReady(b,rows_b) != 0u ? 1u : 0u);
}

static int32_t K3PairFence(const K3SliceState *state, cudaStream_t waiter, cudaStream_t signaler)
{
	cudaEvent_t fence = (cudaEvent_t)state->pair_fence;
	return(cudaEventRecord(fence,signaler) == cudaSuccess && cudaStreamWaitEvent(waiter,fence,0u) == cudaSuccess
		? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t K3PairRound(const K3SliceState *state, void *lane, cudaStream_t main, cudaStream_t producer, cudaStream_t consumer, uint32_t layer, uint32_t phase, uint32_t split)
{
	void *const word = (void *)(uintptr_t)main;
	if ( K3PairFence(state,main,producer) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_LAUNCH);
	state->layer_collective(lane,word,layer,phase);
	if ( split != 0u )
		state->layer_collective(lane,word,layer,phase | K3_COLLECTIVE_FINISH);
	return(consumer != 0 ? K3PairFence(state,consumer,main) : LM_LAUNCH_OK);
}

template<class Format, class Geometry>
static int32_t K3LaunchSlicePair(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *a, K3LayerBuffers *b, uint32_t first_layer, uint32_t layer_count, uint32_t rows_a, uint32_t rows_b, uint32_t commit, uint32_t multiprocessors, cudaStream_t stream)
{
	void *const lane_a = state->collective_context;
	void *const lane_b = state->collective_pair_context;
	const cudaStream_t sa = (cudaStream_t)state->pair_stream_a;
	const cudaStream_t sb = (cudaStream_t)state->pair_stream_b;
	const uint32_t total = rows_a + rows_b;
	uint32_t offset, layer = first_layer;
	int32_t status = LM_LAUNCH_OK;
	if ( K3PairReady(state,a,b,rows_a,rows_b) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
#define K3_PAIR_DO(call) do { if ( status == LM_LAUNCH_OK ) status = (call); } while ( 0 )
	K3_PAIR_DO(K3PairFence(state,sa,stream));
	K3_PAIR_DO(K3PairFence(state,sb,stream));
	for (offset = 0u; offset < layer_count && status == LM_LAUNCH_OK; ++offset)
	{
		layer = first_layer + offset;
		if ( layer >= K3_LAYERS )
			return(K3SliceFailure(layer,"shape",LM_LAUNCH_ERR_SHAPE));
		K3BindLayer(&weights[offset],a);
		K3BindLayerState(state,layer,a);
		K3BindLayer(&weights[offset],b);
		K3BindLayerState(state,layer,b);
		K3PairPre(a,layer,rows_a,sa);
		if ( K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT )
			K3_PAIR_DO(K3LayerKda<Format>(a,rows_a,1u,commit,(uint16_t *)0,multiprocessors,sa));
		else
			K3_PAIR_DO((K3LayerMlaQuery<Format,Geometry>(a,rows_a,multiprocessors,sa)));
		K3_PAIR_DO(K3PairFence(state,sb,sa));
		if ( K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT )
		{
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sa,sa,layer,0u,0u));
			K3PairPre(b,layer,rows_b,sb);
			K3_PAIR_DO(K3LayerKda<Format>(b,rows_b,1u,commit,(uint16_t *)0,multiprocessors,sb));
		}
		else
		{
			K3PairPre(b,layer,rows_b,sb);
			K3_PAIR_DO((K3LayerMlaQuery<Format,Geometry>(b,rows_b,multiprocessors,sb)));
			K3_PAIR_DO(K3LayerMlaGatherPack<Geometry>(a,rows_a,sb));
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sb,0,layer,K3_COLLECTIVE_MLA_KEYS,1u));
			K3_PAIR_DO(K3LayerMlaGate(a,rows_a,multiprocessors,sa));
			K3_PAIR_DO(K3PairFence(state,sa,stream));
			K3_PAIR_DO(K3LayerMlaGate(b,rows_b,multiprocessors,sb));
			K3_PAIR_DO(K3PairFence(state,sb,stream));
			K3_PAIR_DO(K3LayerMlaGatherPartials<Geometry>(a,rows_a,sa));
			K3_PAIR_DO(K3LayerMlaShardMerge(a,rows_a,sa));
			K3_PAIR_DO(K3LayerMlaOutput<Format>(a,rows_a,(uint16_t *)0,1u,multiprocessors,sa));
			K3_PAIR_DO(K3PairFence(state,sb,sa));
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sa,sa,layer,0u,0u));
			K3_PAIR_DO(K3LayerMlaGatherPartials<Geometry>(b,rows_b,sb));
			K3_PAIR_DO(K3LayerMlaShardMerge(b,rows_b,sb));
			K3_PAIR_DO(K3LayerMlaOutput<Format>(b,rows_b,(uint16_t *)0,1u,multiprocessors,sb));
		}
		K3_PAIR_DO(K3PairRound(state,lane_b,stream,sb,sb,layer,0u,0u));
		if ( status != LM_LAUNCH_OK )
			return(K3SliceFailure(layer,"pair-attention",status));
		K3AttnRes(a,a->attnres_mlp_weight,(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows_a,sa);
		if ( layer < K3_FIRST_ROUTED_LAYER )
		{
			K3_PAIR_DO(K3LayerDenseMlp<Format>(a,rows_a,multiprocessors,sa));
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sa,sa,layer,1u,0u));
			K3AttnRes(b,b->attnres_mlp_weight,(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows_b,sb);
			K3_PAIR_DO(K3LayerDenseMlp<Format>(b,rows_b,multiprocessors,sb));
			K3_PAIR_DO(K3PairRound(state,lane_b,stream,sb,sb,layer,1u,0u));
			if ( status != LM_LAUNCH_OK )
				return(K3SliceFailure(layer,"pair-dense",status));
		}
		else
		{
			K3_PAIR_DO(K3LayerLatentMoe<Format>(a,rows_a,rows_a * K3_TOP_K,multiprocessors,sa,0u));
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sa,0,layer,2u,1u));
			K3_PAIR_DO(K3LayerSharedUp<Format>(a,rows_a,a->shared_mid_bf16,multiprocessors,sa));
			K3_PAIR_DO(K3PairFence(state,sa,stream));
			K3AttnRes(b,b->attnres_mlp_weight,(layer / K3_ATTNRES_BLOCK_SIZE) + 2u,rows_b,sb);
			K3_PAIR_DO(K3LayerLatentMoe<Format>(b,rows_b,rows_b * K3_TOP_K,multiprocessors,sb,0u));
			K3_PAIR_DO(K3PairRound(state,lane_b,stream,sb,0,layer,2u,1u));
			K3_PAIR_DO(K3LayerSharedUp<Format>(b,rows_b,b->shared_mid_bf16,multiprocessors,sb));
			K3_PAIR_DO(K3PairFence(state,sa,stream));
			K3_PAIR_DO(K3PairFence(state,sa,sb));
			K3_PAIR_DO(K3LayerMoeSelect(a,total,total * K3_TOP_K,sa));
			if ( status == LM_LAUNCH_OK && state->lazy_acquire != 0 )
				status = state->lazy_acquire(state->lazy_context,layer,(void *)a);
			K3_PAIR_DO(K3LayerLatentMoe<Format>(a,total,total * K3_TOP_K,multiprocessors,sa,1u));
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sa,sa,layer,3u,1u));
			K3_PAIR_DO(K3LayerSharedDown<Format>(b,rows_b,b->shared_mid_bf16,multiprocessors,sb));
			K3_PAIR_DO(K3PairRound(state,lane_b,stream,sa,sb,layer,3u,1u));
			K3_PAIR_DO(K3LayerSharedDown<Format>(a,rows_a,a->shared_mid_bf16,multiprocessors,sa));
			K3_PAIR_DO(K3LayerLatentMoe<Format>(a,rows_a,rows_a * K3_TOP_K,multiprocessors,sa,2u));
			K3_PAIR_DO(K3PairRound(state,lane_a,stream,sa,sa,layer,1u,0u));
			K3_PAIR_DO(K3LayerLatentMoe<Format>(b,rows_b,rows_b * K3_TOP_K,multiprocessors,sb,2u));
			K3_PAIR_DO(K3PairRound(state,lane_b,stream,sb,sb,layer,1u,0u));
			if ( status != LM_LAUNCH_OK )
				return(K3SliceFailure(layer,"pair-routed",status));
		}
		if ( state->lazy_release != 0 )
			state->lazy_release(state->lazy_context,layer);
	}
	if ( status != LM_LAUNCH_OK )
		return(K3SliceFailure(layer,"pair",status));
	if ( first_layer + layer_count == K3_LAYERS )
	{
		K3AttnRes(a,a->attnres_out_weight,((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows_a,sa);
		K3AttnRes(b,b->attnres_out_weight,((K3_LAYERS - 1u) / K3_ATTNRES_BLOCK_SIZE) + 2u,rows_b,sb);
	}
	else
	{
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
			dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows_a),
			K3_LAYER_THREADS, 0, sa,
			a->attnres_partial_bf16,a->hidden_bf16,rows_a,K3_HIDDEN);
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>),
			dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows_b),
			K3_LAYER_THREADS, 0, sb,
			b->attnres_partial_bf16,b->hidden_bf16,rows_b,K3_HIDDEN);
	}
	K3_PAIR_DO(K3PairFence(state,stream,sa));
	K3_PAIR_DO(K3PairFence(state,stream,sb));
#undef K3_PAIR_DO
	return(status);
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
		status = K3LaunchAttentionHalf<Format,Geometry>(state,buffers,layer,rows,sequences,commit,
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
	if ( phase == 2u )
		return(K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,1u));
	return(K3LayerLatentMoe<Format>(buffers,rows,packed_rows,multiprocessors,stream,2u));
}

template<class Format>
static int32_t K3FoldAccepted(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t layer;
	uint64_t replay_capacity;
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
	if ( state->kda_rank_heads != K3_KDA_HEADS )
		return(LM_LAUNCH_ERR_SHAPE);
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
		LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM,float,K3_KDA_DELTA_COLUMNS>), dim3(sequences,K3_KDA_HEADS,K3_KDA_VALUE_DIM / K3_KDA_DELTA_COLUMNS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_DELTA_COLUMNS * sizeof(float)), stream,
			buffers->kda_state_pool,state->kda_state_bf16 != 0u ? K3_KDA_STATE_SLOT_BYTES_BF16 : K3_KDA_STATE_SLOT_BYTES,buffers->kda_state_index,verify_row_begin,accepted,buffers->query_bf16,buffers->key_bf16, buffers->value_bf16,buffers->replay_retention,buffers->replay_write_gate,buffers->attention_out_bf16, K3_KDA_HEADS,1u,sequences,1u);
	}
	return(LM_LAUNCH_OK);
}
