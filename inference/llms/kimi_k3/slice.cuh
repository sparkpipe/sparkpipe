#pragma once
#include <stdio.h>
#include "inference/llms/kimi_k3/layer.cuh"
#include "inference/llms/kimi_k3/dspark.h"
#include "inference/kernels/l2_load.cuh"
#include "sparkpipe/spark_k3_family.h"

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
	uint32_t replay_active;
	uint16_t *tap_slab;
	uint64_t tap_row_stride;
	uint32_t tap_active;
	int16_t tap_output[K3_LAYERS];
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
	const uint64_t replay_rows = state->replay_active != 0u ? state->verify_rows : 0u;
	buffers->replay_conv_q = state->replay_conv_q == 0 || replay_rows == 0u ? 0 : state->replay_conv_q
		+ ((uint64_t)kda_index * replay_rows * heads * K3_KDA_KEY_DIM);
	buffers->replay_conv_k = state->replay_conv_k == 0 || replay_rows == 0u ? 0 : state->replay_conv_k
		+ ((uint64_t)kda_index * replay_rows * heads * K3_KDA_KEY_DIM);
	buffers->replay_conv_v = state->replay_conv_v == 0 || replay_rows == 0u ? 0 : state->replay_conv_v
		+ ((uint64_t)kda_index * replay_rows * heads * K3_KDA_VALUE_DIM);
	buffers->replay_retention = state->replay_retention == 0 || replay_rows == 0u ? 0 : state->replay_retention
		+ ((uint64_t)kda_index * replay_rows * heads * K3_KDA_KEY_DIM);
	buffers->replay_write_gate = state->replay_write_gate == 0 || replay_rows == 0u ? 0 : state->replay_write_gate
		+ ((uint64_t)kda_index * replay_rows * heads);
	if ( K3_LAYER_KIND(layer) == LM_LAYER_LATENT )
	{
		buffers->cache = state->mla_cache[mla_index];
		buffers->cache_shard.pages = buffers->cache;
		buffers->cache_shard.shard = buffers->kv_shard;
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

static int32_t K3SliceFailure(uint32_t layer,const char *phase,int32_t status)
{
	fprintf(stderr,"k3 slice failed layer=%u phase=%s status=%d\n",layer,phase,status);
	return(status);
}

static uint32_t K3PairReady(const K3SliceState *state, const K3LayerBuffers *a, const K3LayerBuffers *b, uint32_t rows_a, uint32_t rows_b)
{
	return(state->layer_collective != 0 && state->collective_pair_context != 0 &&
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
struct K3Cuda
{
	static constexpr int32_t kOk = LM_LAUNCH_OK;
	static constexpr int32_t kShapeError = LM_LAUNCH_ERR_SHAPE;
	static constexpr bool kShardedMla = true;
	static constexpr bool kExpertCells = true;
	static constexpr bool kLazyExperts = true;

	struct Context
	{
		const K3LayerWeights *weights;
		const K3SliceState *state;
		K3LayerBuffers *buffers[2];
		cudaStream_t streams[3];
		uint32_t rows[2];
		uint32_t sequences;
		uint32_t commit;
		uint32_t packed_rows;
		uint32_t context;
		uint32_t multiprocessors;
		LmL2LoadPlan load;
	};

	static K3LayerBuffers *Lane(const Context &c, uint32_t lane)
	{
		return c.buffers[lane];
	}

	static cudaStream_t Stream(const Context &c, uint32_t stream)
	{
		return stream == K3_STREAM_NONE ? (cudaStream_t)0 : c.streams[stream];
	}

	static void *CollectiveLane(const Context &c, uint32_t lane)
	{
		return lane == K3_LANE_A ? c.state->collective_context : c.state->collective_pair_context;
	}

	static uint16_t *Accumulate(const K3LayerBuffers *b, uint32_t accumulate)
	{
		return accumulate != 0u ? b->attnres_partial_bf16 : (uint16_t *)0;
	}

	static uint32_t Rows(const Context &c, uint32_t lane) { return c.rows[lane]; }
	static uint32_t Sequences(const Context &c) { return c.sequences; }
	static uint32_t PackedRows(const Context &c) { return c.packed_rows; }
	static bool HasCollective(const Context &c) { return c.state->layer_collective != 0; }
	static bool HasLazy(const Context &c) { return c.state->lazy_acquire != 0; }
	static uint32_t TpSharded(const Context &c, uint32_t lane) { return Lane(c, lane)->tp_sharded; }
	static bool KvSharded(const Context &c, uint32_t lane) { return Lane(c, lane)->kv_shard.degree > 1u; }
	static uint32_t ShardGather(const Context &c, uint32_t lane) { return Lane(c, lane)->shard_gather; }
	static uint32_t GatherWindows(const Context &c, uint32_t lane) { return Lane(c, lane)->shard_gather_windows; }
	static uint32_t GatherExpanded(const Context &c, uint32_t lane) { return Lane(c, lane)->shard_gather_expanded; }
	static uint32_t GatherHistory(const Context &c, uint32_t lane) { return Lane(c, lane)->shard_gather_history; }
	static uint32_t GatherPrefetched(const Context &c, uint32_t lane) { return Lane(c, lane)->shard_gather_prefetched; }
	static void SetGatherWindow(Context &c, uint32_t lane, uint32_t window) { Lane(c, lane)->shard_gather_window = window; }
	static void SetGatherPrefetched(Context &c, uint32_t lane, uint32_t value) { Lane(c, lane)->shard_gather_prefetched = value; }
	static uint32_t MlaPasses(const Context &c, uint32_t lane) { return K3LayerMlaPasses(Lane(c, lane)); }

	static bool ExpertCells(const Context &c, uint32_t lane)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3_EXPERT_CELLS(b);
	}

	static bool MlaDownSliced(const Context &c, uint32_t lane)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3_MLA_DOWN_SLICED(b, c.rows[lane]);
	}

	static uint32_t RowShard(const Context &c, uint32_t first_layer, uint32_t layer_count)
	{
		const K3LayerBuffers *a = c.buffers[K3_LANE_A], *b = c.buffers[K3_LANE_B];
		return first_layer == 0u && layer_count == K3_LAYERS && a->attnres_score_f32 != 0 && b->attnres_score_f32 != 0 ? K3_COLLECTIVE_ROWS : 0u;
	}

	static bool PairReady(const Context &c)
	{
		return K3PairReady(c.state, c.buffers[K3_LANE_A], c.buffers[K3_LANE_B], c.rows[K3_LANE_A], c.rows[K3_LANE_B]) != 0u;
	}

	static int32_t Failure(uint32_t layer, const char *phase, int32_t status)
	{
		return K3SliceFailure(layer, phase, status);
	}

	static void Bind(Context &c, uint32_t lane, uint32_t offset, uint32_t layer)
	{
		K3BindLayer(&c.weights[offset], Lane(c, lane));
		K3BindLayerState(c.state, layer, Lane(c, lane));
	}

	static void SetNextAttnRes(Context &c, uint32_t lane, uint32_t offset, uint32_t layer_count)
	{
		K3LayerBuffers *b = Lane(c, lane);
		b->attnres_next_weight = offset + 1u < layer_count ? c.weights[offset + 1u].attnres_attn_weight : b->attnres_out_weight;
	}

	static void AttnRes(Context &c, uint32_t lane, uint32_t site, uint32_t sources, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		const void *weight = site == K3_ATTNRES_ATTENTION ? b->attnres_attn_weight : site == K3_ATTNRES_MLP ? b->attnres_mlp_weight : b->attnres_out_weight;
		K3AttnRes(b, weight, sources, c.rows[lane], Stream(c, stream));
	}

	static void PartialSet(Context &c, uint32_t lane, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		K3PartialSet(b, b->hidden_bf16, c.rows[lane], Stream(c, stream));
	}

	static void PartialSetUnsharded(Context &c, uint32_t lane, uint32_t stream)
	{
		if ( Lane(c, lane)->tp_sharded == 0u )
			PartialSet(c, lane, stream);
	}

	static void BankStore(Context &c, uint32_t lane, uint32_t bank, uint32_t stream)
	{
		K3BankStore(Lane(c, lane), bank, c.rows[lane], Stream(c, stream));
	}

	static int32_t Kda(Context &c, uint32_t lane, uint32_t sequences, uint32_t accumulate, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3LayerKda<Format>(b, c.rows[lane], sequences, c.commit, Accumulate(b, accumulate), c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaLocal(Context &c, uint32_t lane, uint32_t accumulate, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3LayerMla<Format,Geometry>(b, c.rows[lane], c.context, Accumulate(b, accumulate), c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaQuery(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaQuery<Format,Geometry>(Lane(c, lane), c.rows[lane], c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaGatherPack(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaGatherPack<Geometry>(Lane(c, lane), c.rows[lane], Stream(c, stream));
	}

	static int32_t MlaPrefetchPack(Context &c, uint32_t lane, uint32_t next_layer, uint32_t stream)
	{
		return K3LayerMlaPrefetchPack<Geometry>(Lane(c, lane), c.rows[lane], c.state->mla_cache[(next_layer / 4u) - c.state->first_mla_index], 1u,
			Stream(c, stream));
	}

	static int32_t MlaGate(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaGate(Lane(c, lane), c.rows[lane], c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaExpandValues(Context &c, uint32_t lane, uint32_t window, uint32_t stream)
	{
		return K3LayerMlaExpandValues(Lane(c, lane), c.rows[lane], window, Stream(c, stream));
	}

	static int32_t MlaGatherPartials(Context &c, uint32_t lane, uint32_t window, uint32_t stream)
	{
		return K3LayerMlaGatherPartials<Geometry>(Lane(c, lane), c.rows[lane], window, Stream(c, stream));
	}

	static int32_t MlaGatherMerge(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaGatherMerge(Lane(c, lane), c.rows[lane], Stream(c, stream));
	}

	static int32_t MlaShardFold(Context &c, uint32_t lane, uint32_t window, uint32_t stream)
	{
		return K3LayerMlaShardFold(Lane(c, lane), c.rows[lane], window, Stream(c, stream));
	}

	static int32_t MlaShardFinish(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaShardFinish(Lane(c, lane), c.rows[lane], Stream(c, stream));
	}

	static int32_t MlaOutput(Context &c, uint32_t lane, uint32_t accumulate, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3LayerMlaOutput<Format>(b, c.rows[lane], Accumulate(b, accumulate), 1u, c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaDownSlice(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaDownSlice(Lane(c, lane), c.rows[lane], Stream(c, stream));
	}

	static int32_t MlaDown(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaDown<Format>(Lane(c, lane), c.rows[lane], c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaUp(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaUp<Format,Geometry>(Lane(c, lane), c.rows[lane], c.multiprocessors, Stream(c, stream));
	}

	static int32_t MlaShardPartials(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaShardPartials<Geometry>(Lane(c, lane), c.rows[lane], Stream(c, stream));
	}

	static int32_t MlaShardMerge(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMlaShardMerge(Lane(c, lane), c.rows[lane], Stream(c, stream));
	}

	static int32_t Dense(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerDenseMlp<Format>(Lane(c, lane), c.rows[lane], c.multiprocessors, Stream(c, stream));
	}

	static int32_t LatentMoe(Context &c, uint32_t lane, uint32_t rows, uint32_t packed_rows, uint32_t phase, uint32_t stream)
	{
		return K3LayerLatentMoe<Format>(Lane(c, lane), rows, packed_rows, c.multiprocessors, Stream(c, stream), phase);
	}

	static int32_t SharedUp(Context &c, uint32_t lane, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3LayerSharedUp<Format>(b, c.rows[lane], b->shared_mid_bf16, c.multiprocessors, Stream(c, stream));
	}

	static int32_t SharedDown(Context &c, uint32_t lane, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		return K3LayerSharedDown<Format>(b, c.rows[lane], b->shared_mid_bf16, c.multiprocessors, Stream(c, stream));
	}

	static int32_t MoeSelect(Context &c, uint32_t lane, uint32_t rows, uint32_t packed_rows, uint32_t stream)
	{
		return K3LayerMoeSelect(Lane(c, lane), rows, packed_rows, Stream(c, stream));
	}

	static int32_t MoeRoute(Context &c, uint32_t lane, uint32_t stream)
	{
		return K3LayerMoeRoute<Format>(Lane(c, lane), c.rows[lane], c.packed_rows, c.multiprocessors, Stream(c, stream));
	}

	static int32_t MoeWeighted(Context &c, uint32_t lane, uint32_t phase, uint32_t stream)
	{
		return K3LayerMoeWeighted<Format>(Lane(c, lane), c.rows[lane], c.packed_rows, c.multiprocessors, Stream(c, stream), phase);
	}

	static int32_t Acquire(Context &c, uint32_t lane, uint32_t layer)
	{
		return c.state->lazy_acquire(c.state->lazy_context, layer, (void *)Lane(c, lane));
	}

	static void Release(Context &c, uint32_t layer)
	{
		if ( c.state->lazy_release != 0 )
			c.state->lazy_release(c.state->lazy_context, layer);
	}

	static void Collective(Context &c, uint32_t lane, uint32_t stream, uint32_t layer, uint32_t phase)
	{
		c.state->layer_collective(CollectiveLane(c, lane), (void *)(uintptr_t)Stream(c, stream), layer, phase);
	}

	static int32_t Fence(Context &c, uint32_t waiter, uint32_t signaler)
	{
		return K3PairFence(c.state, Stream(c, waiter), Stream(c, signaler));
	}

	static int32_t PairRound(Context &c, uint32_t lane, uint32_t producer, uint32_t consumer, uint32_t layer, uint32_t phase, uint32_t split)
	{
		return K3PairRound(c.state, CollectiveLane(c, lane), Stream(c, K3_STREAM_MAIN), Stream(c, producer), Stream(c, consumer), layer, phase, split);
	}

	static int32_t MlpPrefetchRound(Context &c, uint32_t layer)
	{
		K3L2PlanMoe(Lane(c, K3_LANE_A), layer, &c.load);
		return K3L2LoadRound(c.state, layer, 0u, &c.load, Stream(c, K3_STREAM_MAIN));
	}

	static int32_t AttentionPrefetchRound(Context &c, uint32_t offset, uint32_t layer, uint32_t layer_count)
	{
		c.load.count = 0u;
		if ( offset + 1u < layer_count )
		{
			K3LayerBuffers next = *Lane(c, K3_LANE_A);
			K3BindLayer(&c.weights[offset + 1u], &next);
			K3L2PlanAttention(&next, layer + 1u, &c.load);
		}
		return K3L2LoadRound(c.state, layer, 1u, &c.load, Stream(c, K3_STREAM_MAIN));
	}

	static void Tap(Context &c, uint32_t lane, uint32_t layer, uint32_t row_offset, uint32_t stream)
	{
		const K3SliceState *state = c.state;
		const K3LayerBuffers *b = Lane(c, lane);
		const uint32_t rows = c.rows[lane];
		if ( state->tap_slab != 0 && state->tap_output[layer] >= 0 )
			LM_LAUNCH((LmCopyRowsStridedKernel<K3_LAYER_THREADS>), dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0,
				Stream(c, stream), b->attnres_partial_bf16, state->tap_slab + (uint64_t)row_offset * state->tap_row_stride +
				((uint64_t)state->tap_output[layer] * K3_HIDDEN), rows, K3_HIDDEN, state->tap_row_stride);
	}

	static void CopyRows(Context &c, uint32_t lane, uint32_t stream)
	{
		const K3LayerBuffers *b = Lane(c, lane);
		const uint32_t rows = c.rows[lane];
		LM_LAUNCH((LmCopyRowsKernel<K3_LAYER_THREADS>), dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows), K3_LAYER_THREADS, 0,
			Stream(c, stream), b->attnres_partial_bf16, b->hidden_bf16, rows, K3_HIDDEN);
	}
};

template<class Format, class Geometry>
static void K3CudaContext(typename K3Cuda<Format,Geometry>::Context *c, const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *a,
	K3LayerBuffers *b, uint32_t rows_a, uint32_t rows_b, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context,
	uint32_t multiprocessors, cudaStream_t stream)
{
	memset(c, 0, sizeof(*c));
	c->weights = weights;
	c->state = state;
	c->buffers[K3_LANE_A] = a;
	c->buffers[K3_LANE_B] = b;
	c->streams[K3_STREAM_MAIN] = stream;
	c->rows[K3_LANE_A] = rows_a;
	c->rows[K3_LANE_B] = rows_b;
	c->sequences = sequences;
	c->commit = commit;
	c->packed_rows = packed_rows;
	c->context = context;
	c->multiprocessors = multiprocessors;
}

template<class Format, class Geometry>
static int32_t K3LaunchSlice(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	typename K3Cuda<Format,Geometry>::Context c;
	K3CudaContext<Format,Geometry>(&c, weights, state, buffers, (K3LayerBuffers *)0, rows, 0u, sequences, commit, packed_rows, context, multiprocessors,
		stream);
	return(K3Schedule<K3Cuda<Format,Geometry>>::Slice(c, first_layer, layer_count));
}

template<class Format, class Geometry>
static int32_t K3LaunchSlicePair(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *a, K3LayerBuffers *b, uint32_t first_layer, uint32_t layer_count, uint32_t rows_a, uint32_t rows_b, uint32_t commit, uint32_t multiprocessors, cudaStream_t stream)
{
	typename K3Cuda<Format,Geometry>::Context c;
	K3CudaContext<Format,Geometry>(&c, weights, state, a, b, rows_a, rows_b, 1u, commit, 0u, 0u, multiprocessors, stream);
	c.streams[K3_STREAM_A] = (cudaStream_t)state->pair_stream_a;
	c.streams[K3_STREAM_B] = (cudaStream_t)state->pair_stream_b;
	return(K3Schedule<K3Cuda<Format,Geometry>>::SlicePair(c, first_layer, layer_count));
}

template<class Format, class Geometry>
static int32_t K3LaunchSliceHalf(const K3LayerWeights *weights, const K3SliceState *state,
	K3LayerBuffers *buffers, uint32_t layer, uint32_t phase, uint32_t rows,
	uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context,
	uint32_t multiprocessors, cudaStream_t stream)
{
	typename K3Cuda<Format,Geometry>::Context c;
	K3CudaContext<Format,Geometry>(&c, weights, state, buffers, (K3LayerBuffers *)0, rows, 0u, sequences, commit, packed_rows, context, multiprocessors,
		stream);
	return(K3Schedule<K3Cuda<Format,Geometry>>::SliceHalf(c, layer, phase));
}

template<class Format>
static int32_t K3FoldAccepted(const K3LayerWeights *weights, const K3SliceState *state, K3LayerBuffers *buffers, uint32_t first_layer, uint32_t layer_count, uint32_t sequences, const uint32_t *verify_row_begin, const uint32_t *verify_row_indices, const uint32_t *accepted, uint32_t slab_rows, uint32_t multiprocessors, cudaStream_t stream)
{
	uint32_t layer;
	(void)multiprocessors;
	if ( weights == 0 || state == 0 || buffers == 0
		|| verify_row_begin == 0 || accepted == 0
		|| first_layer > K3_LAYERS || layer_count > K3_LAYERS - first_layer
		|| sequences == 0u || sequences > state->sequences
		|| state->verify_rows == 0u || state->replay_active == 0u
		|| state->replay_conv_q == 0 || state->replay_conv_k == 0
		|| state->replay_conv_v == 0 || state->replay_retention == 0
		|| state->replay_write_gate == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( slab_rows == 0u || slab_rows > state->verify_rows )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( state->kda_state_bf16 != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	for (layer = first_layer; layer < first_layer + layer_count; ++layer)
	{
		if ( !K3Layout::Kda(layer) )
			continue;
		K3BindLayer(&weights[layer - first_layer],buffers);
		K3BindLayerState(state,layer,buffers);
		const uint32_t rank_heads = K3_RANK_DIM(buffers,kda_heads_rank,K3_KDA_HEADS);
		const uint32_t rank_qk = rank_heads * K3_KDA_KEY_DIM, rank_v = rank_heads * K3_KDA_VALUE_DIM;
		const uint32_t slot_bytes = (uint32_t)K3_KDA_RANK_STATE_SLOT_BYTES(rank_heads, 0u);
		if ( buffers->tp_rank * rank_heads + rank_heads > K3_KDA_HEADS || rank_heads != state->kda_rank_heads )
			return(LM_LAUNCH_ERR_SHAPE);
		LmCausalConvLaunch<K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>(buffers->kda_q_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_q,buffers->kda_q_conv_weight,buffers->query_bf16,rank_qk,sequences,1u,verify_row_indices, slab_rows, stream);
		LmCausalConvLaunch<K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>(buffers->kda_k_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_k,buffers->kda_k_conv_weight,buffers->key_bf16,rank_qk,sequences,1u,verify_row_indices, slab_rows, stream);
		LmCausalConvLaunch<K3_KDA_CONV_KERNEL,LM_CONV_SWISH,float>(buffers->kda_v_window,buffers->kda_state_index,verify_row_begin,accepted,buffers->replay_conv_v,buffers->kda_v_conv_weight,buffers->value_bf16,rank_v,sequences,1u,verify_row_indices, slab_rows, stream);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(slab_rows,rank_heads), K3_LAYER_THREADS, 0, stream,
			buffers->query_bf16,rank_heads,slab_rows,K3_RMS_EPSILON);
		LM_LAUNCH((LmL2NormalisePerHeadKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM>), dim3(slab_rows,rank_heads), K3_LAYER_THREADS, 0, stream,
			buffers->key_bf16,rank_heads,slab_rows,K3_RMS_EPSILON);
		if constexpr ( LM_WARP_LANES == 32u )
			LM_LAUNCH((LmDeltaRuleColumnKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM,float>), dim3(sequences,rank_heads,K3_KDA_VALUE_DIM / LM_WARP_LANES), LM_DELTA_COLUMN_THREADS, 0, stream,
				buffers->kda_state_pool,slot_bytes,buffers->kda_state_index,verify_row_begin,accepted,buffers->query_bf16,buffers->key_bf16,buffers->value_bf16,buffers->replay_retention,buffers->replay_write_gate,buffers->attention_out_bf16,rank_heads,1u,sequences,1u,verify_row_indices);
		else
			LM_LAUNCH((LmDeltaRuleKernel<K3_LAYER_THREADS,K3_KDA_KEY_DIM,K3_KDA_VALUE_DIM,float,K3_KDA_DELTA_COLUMNS>), dim3(sequences,rank_heads,K3_KDA_VALUE_DIM / K3_KDA_DELTA_COLUMNS), K3_LAYER_THREADS, (uint32_t)(K3_KDA_KEY_DIM * K3_KDA_DELTA_COLUMNS * sizeof(float)), stream,
				buffers->kda_state_pool,slot_bytes,buffers->kda_state_index,verify_row_begin,accepted,buffers->query_bf16,buffers->key_bf16,buffers->value_bf16,buffers->replay_retention,buffers->replay_write_gate,buffers->attention_out_bf16,rank_heads,1u,sequences,1u,verify_row_indices);
	}
	return(LM_LAUNCH_OK);
}
