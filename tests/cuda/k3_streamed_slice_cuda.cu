#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"

#define SL_SHARDS 16u
#define SL_ROWS 256u
#define SL_LAYERS 2u
#define SL_PACKED (SL_ROWS * K3_TOP_K)
#define SL_HEADS (K3_KDA_HEADS / SL_SHARDS)
#define SL_QK (SL_HEADS * K3_KDA_KEY_DIM)
#define SL_V (SL_HEADS * K3_KDA_VALUE_DIM)
#define SL_QKVB (2u * SL_QK + SL_V + SL_HEADS)
#define SL_SLICE (K3_ROUTED_EXPERT_HIDDEN / SL_SHARDS)
#define SL_EXPERT_SLICE (K3_EXPERTS / SL_SHARDS)
#define SL_W1_OUT (2u * K3_EXPERT_INTERMEDIATE / SL_SHARDS)
#define SL_CHANNELS (K3_EXPERT_INTERMEDIATE / SL_SHARDS)
#define SL_SHARED_ROWS (2u * K3_SHARED_INTERMEDIATE / SL_SHARDS)
#define SL_SHARED_IN (K3_SHARED_INTERMEDIATE / SL_SHARDS)
#define SL_DENSE_ROWS (2u * K3_DENSE_INTERMEDIATE / SL_SHARDS)
#define SL_DENSE_IN (K3_DENSE_INTERMEDIATE / SL_SHARDS)
#define SL_SLOT_BYTES K3_KDA_RANK_STATE_SLOT_BYTES(SL_HEADS,0u)
#define SL_MLA_HEADS (K3_MLA_HEADS / SL_SHARDS)
#define SL_NODES 4u
#define SL_PASSES 2u
#define SL_NODE_ROWS (SL_ROWS / (SL_NODES * SL_PASSES))
#define SL_TAIL (2u * SL_QK + SL_V) * K3_KDA_CONV_KERNEL

#define SL_CUDA(call) do { cudaError_t sl_error = (call); if ( sl_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(sl_error)); exit(1); } } while (0)

static int sl_failures;

static void SlCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	sl_failures++;
}

__global__ void SlFillBf16(uint16_t *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = LmFloatToBf16(offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

__global__ void SlFillF32(float *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f);
	}
}

__global__ void SlFillCells(uint8_t *data,uint64_t count,uint32_t salt)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = (uint8_t)(((index / LM_SKINNY_CHUNK_BYTES) % LM_SKINNY_CELL_ROWS) == LM_SKINNY_CELL_NEURONS ? 0x76u + (x & 3u) : x);
	}
}

static double SlHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

template<class T>
static T *SlAlloc(uint64_t count)
{
	T *device;
	SL_CUDA(cudaMalloc((void **)&device,count * sizeof(T) + 256u));
	SL_CUDA(cudaMemset(device,0,count * sizeof(T) + 256u));
	return device;
}

static uint16_t *SlBf16(uint64_t count,uint32_t salt,float scale,float offset)
{
	uint16_t *data = SlAlloc<uint16_t>(count);
	SlFillBf16<<<512,256>>>(data,count,salt,scale,offset);
	return data;
}

static float *SlF32(uint64_t count,uint32_t salt,float scale,float offset)
{
	float *data = SlAlloc<float>(count);
	SlFillF32<<<64,256>>>(data,count,salt,scale,offset);
	return data;
}

static void SlBuffers(K3LayerBuffers *b,uint32_t rank,const uint32_t *state_index,const uint32_t *row_begin,uint32_t rows)
{
	uint32_t *dense_offsets = SlAlloc<uint32_t>(2u),offsets[2] = {0u,rows};
	memset(b,0,sizeof(*b));
	b->hidden_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
	b->normed_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
	b->fused_qkvb_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_QKVB);
	b->query_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_QK);
	b->key_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_QK);
	b->value_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_V);
	b->kda_beta_logit = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_HEADS);
	b->gate_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_V);
	b->decay_logit_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_QK);
	b->kda_retention = SlAlloc<float>((uint64_t)SL_ROWS * SL_QK);
	b->kda_write_gate_out = SlAlloc<float>((uint64_t)SL_ROWS * SL_HEADS);
	b->attention_out_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_V);
	b->latent_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * (K3_ROUTED_EXPERT_HIDDEN + 2u * K3_EXPERTS));
	b->latent_full_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_ROUTED_EXPERT_HIDDEN);
	b->router_logits = SlAlloc<float>((uint64_t)SL_ROWS * K3_EXPERTS);
	b->route_expert = SlAlloc<uint32_t>(SL_PACKED);
	b->route_weight = SlAlloc<float>(SL_PACKED);
	b->route_packed_row = SlAlloc<uint32_t>(SL_PACKED);
	b->route_source_token = SlAlloc<uint32_t>(SL_PACKED);
	b->group_row_offset = SlAlloc<uint32_t>(K3_EXPERTS + 1u);
	b->group_tile_prefix_w1 = SlAlloc<uint32_t>(K3_EXPERTS + 1u);
	b->group_tile_prefix_w2 = SlAlloc<uint32_t>(K3_EXPERTS + 1u);
	b->gate_up_bf16 = SlAlloc<uint16_t>((uint64_t)SL_PACKED * K3_ROUTED_EXPERT_HIDDEN);
	b->intermediate_bf16 = SlAlloc<uint16_t>((uint64_t)SL_PACKED * SL_DENSE_IN);
	b->shared_mid_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * SL_SHARED_IN);
	b->shared_out_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
	b->attnres_partial_bf16 = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
	b->attnres_bank_bf16 = SlAlloc<uint16_t>((uint64_t)K3_ATTNRES_MAX_SOURCES * SL_ROWS * K3_HIDDEN);
	SL_CUDA(cudaMemcpy(dense_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice));
	b->dense_row_offset = dense_offsets;
	b->dense_tile_prefix = SlAlloc<uint32_t>(2u);
	b->kda_state_index = state_index;
	b->sequence_row_begin = row_begin;
	b->tp_sharded = 1u;
	b->tp_rank = rank;
	b->kda_heads_rank = SL_HEADS;
	b->kda_qkvb_rows = SL_QKVB;
	b->kda_gate_rows = SL_V;
	b->kda_decay_up_rows = SL_QK;
	b->kda_out_input = SL_V;
	b->expert_w1_output = SL_W1_OUT;
	b->expert_w2_input = SL_CHANNELS;
	b->routed_down_rows = SL_SLICE;
	b->routed_up_input = SL_SLICE;
	b->shared_w1_rows = SL_SHARED_ROWS;
	b->shared_w2_input = SL_SHARED_IN;
	b->dense_gate_up_rows = SL_DENSE_ROWS;
	b->dense_down_input = SL_DENSE_IN;
}

struct SlRank
{
	K3LayerBuffers buffers;
	K3SliceState state;
	K3LayerWeights weights[SL_LAYERS];
	uint16_t *fused;
	uint16_t *sum;
	cudaStream_t stream;
	cudaEvent_t contributed;
	cudaEvent_t consumed;
	int32_t status;
};

static SlRank sl_ranks[SL_SHARDS];
static pthread_barrier_t sl_barrier;

static void SlWaitAll(SlRank *self,cudaEvent_t SlRank::*which)
{
	for (uint32_t peer = 0u; peer < SL_SHARDS; peer++)
		SL_CUDA(cudaStreamWaitEvent(self->stream,sl_ranks[peer].*which,0u));
}

static void SlCollective(void *context,void *stream_word,uint32_t layer,uint32_t phase)
{
	SlRank *self = (SlRank *)context;
	K3LayerBuffers *b = &self->buffers;
	const uint32_t base = phase & ~(K3_COLLECTIVE_FINISH | K3_COLLECTIVE_BEGIN | K3_COLLECTIVE_ROWS);
	(void)stream_word;
	if ( (phase & K3_COLLECTIVE_FINISH) == 0u )
	{
		if ( base == 0u )
			SL_CUDA(cudaMemcpyAsync(self->fused,b->hidden_bf16,(uint64_t)SL_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice,self->stream));
		else if ( base == 1u && layer == 0u )
			SL_CUDA(cudaMemcpyAsync(self->fused,b->hidden_bf16,(uint64_t)SL_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice,self->stream));
		else if ( base == 1u )
			LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,SL_ROWS),K3_LAYER_THREADS,0,self->stream,
				b->hidden_bf16,b->shared_out_bf16,self->fused,SL_ROWS,K3_HIDDEN);
		SL_CUDA(cudaEventRecord(self->contributed,self->stream));
		return;
	}
	pthread_barrier_wait(&sl_barrier);
	SlWaitAll(self,&SlRank::contributed);
	if ( base == 2u )
		for (uint32_t peer = 0u; peer < SL_SHARDS; peer++)
		{
			const K3LayerBuffers *p = &sl_ranks[peer].buffers;
			SL_CUDA(cudaMemcpy2DAsync(b->latent_full_bf16 + peer * SL_SLICE,K3_ROUTED_EXPERT_HIDDEN * 2u,p->latent_bf16,SL_SLICE * 2u,SL_SLICE * 2u,SL_ROWS,
				cudaMemcpyDeviceToDevice,self->stream));
			SL_CUDA(cudaMemcpy2DAsync(b->router_logits + peer * SL_EXPERT_SLICE,K3_EXPERTS * 4u,K3_ROUTER_SLICE(p,SL_ROWS),SL_EXPERT_SLICE * 4u,
				SL_EXPERT_SLICE * 4u,SL_ROWS,cudaMemcpyDeviceToDevice,self->stream));
		}
	else
	{
		LmTpBf16Contributions<SL_SHARDS> inputs;
		for (uint32_t peer = 0u; peer < SL_SHARDS; peer++)
			inputs.rank[peer] = base == 3u ? sl_ranks[peer].buffers.latent_bf16 : sl_ranks[peer].fused;
		if ( base == 3u )
			LmTpBf16SumKernel<SL_SHARDS><<<SL_ROWS,256u,0,self->stream>>>(b->latent_full_bf16,inputs,SL_ROWS,K3_ROUTED_EXPERT_HIDDEN);
		else
		{
			LmTpBf16SumKernel<SL_SHARDS><<<SL_ROWS,256u,0,self->stream>>>(self->sum,inputs,SL_ROWS,K3_HIDDEN);
			if ( base == 0u && (layer % K3_ATTNRES_BLOCK_SIZE) == 0u )
				K3PartialSet(b,self->sum,SL_ROWS,self->stream);
			else
				K3PartialAdd(b,self->sum,SL_ROWS,self->stream);
		}
		SL_CUDA(cudaGetLastError());
	}
	SL_CUDA(cudaEventRecord(self->consumed,self->stream));
	pthread_barrier_wait(&sl_barrier);
	SlWaitAll(self,&SlRank::consumed);
}

static void *SlRankThread(void *argument)
{
	SlRank *self = (SlRank *)argument;
	self->status = K3LaunchSlice<LmMxfp4,K3GlobalKv>(self->weights,&self->state,&self->buffers,0u,SL_LAYERS,SL_ROWS,1u,1u,SL_PACKED,SL_ROWS,48u,self->stream);
	SL_CUDA(cudaStreamSynchronize(self->stream));
	return 0;
}


struct SlNode
{
	uint32_t node;
	const K3LayerWeights *weights;
	K3StreamedKdaState *states;
	K3StreamedKdaExchange exchange;
	K3StreamedJoin join;
	K3LayerBuffers buffers;
	uint16_t *partial;
	float *latent_accumulator;
	float *hidden_accumulator;
	const uint16_t *hidden;
	uint16_t *gathered;
	int32_t status;
	cudaStream_t stream;
};

static SlNode sl_nodes[SL_NODES];
static pthread_barrier_t sl_node_barrier;

static void SlNodeExchange(void *context,void *stream,uint32_t phase,uint32_t shard)
{
	(void)context;
	(void)phase;
	(void)shard;
	SL_CUDA(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream));
	pthread_barrier_wait(&sl_node_barrier);
}

static void *SlNodeThread(void *argument)
{
	SlNode *n = (SlNode *)argument;
	const uint64_t row_bytes = (uint64_t)K3_HIDDEN * 2u;
	n->status = LM_LAUNCH_OK;
	for (uint32_t pass = 0u; pass < SL_PASSES && n->status == LM_LAUNCH_OK; pass++)
	{
		const uint64_t first = (uint64_t)pass * SL_NODES * SL_NODE_ROWS + (uint64_t)n->node * SL_NODE_ROWS;
		SL_CUDA(cudaMemcpyAsync(n->buffers.hidden_bf16,n->hidden + first * K3_HIDDEN,SL_NODE_ROWS * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		n->status = K3StreamedSlice<LmMxfp4,K3GlobalKv,SL_MLA_HEADS>(&n->buffers,n->weights,n->states,SL_SHARDS,&n->exchange,0,&n->join,0u,SL_LAYERS,n->partial,
			n->latent_accumulator,n->hidden_accumulator,SL_NODE_ROWS,SL_NODE_ROWS * K3_TOP_K,48u,n->stream);
		SL_CUDA(cudaMemcpyAsync(n->gathered + first * K3_HIDDEN,n->buffers.hidden_bf16,SL_NODE_ROWS * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		SL_CUDA(cudaStreamSynchronize(n->stream));
	}
	return 0;
}

int main(void)
{
	const uint64_t w1_bytes = (uint64_t)K3_EXPERTS * (K3_ROUTED_EXPERT_HIDDEN / 32u) * (SL_W1_OUT / 16u) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	const uint64_t w2_bytes = (uint64_t)K3_EXPERTS * (SL_CHANNELS / 32u) * (K3_ROUTED_EXPERT_HIDDEN / 16u) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	std::vector<K3LayerWeights> shard_weights((uint64_t)SL_LAYERS * SL_SHARDS);
	std::vector<K3StreamedKdaState> states((uint64_t)SL_LAYERS * SL_SHARDS);
	uint16_t *hidden = SlBf16((uint64_t)SL_ROWS * K3_HIDDEN,1u,1.0f,0.0f);
	uint32_t *state_index = SlAlloc<uint32_t>(1u),*row_begin = SlAlloc<uint32_t>(2u),begin[2] = {0u,SL_ROWS};
	uint16_t *partial = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
	float *latent_accumulator = SlAlloc<float>((uint64_t)SL_ROWS * K3_ROUTED_EXPERT_HIDDEN),*hidden_accumulator = SlAlloc<float>((uint64_t)SL_ROWS * K3_HIDDEN);
	K3LayerBuffers streamed;
	cudaStream_t stream;
	SL_CUDA(cudaMemcpy(row_begin,begin,sizeof(begin),cudaMemcpyHostToDevice));
	for (uint32_t layer = 0u; layer < SL_LAYERS; layer++)
	{
		const uint32_t salt = 1000u * (layer + 1u);
		uint16_t *attn_norm = SlBf16(K3_HIDDEN,salt + 1u,0.1f,1.0f),*mlp_norm = SlBf16(K3_HIDDEN,salt + 2u,0.1f,1.0f);
		uint16_t *attnres_attn = SlBf16(K3_HIDDEN,salt + 3u,0.5f,0.0f),*attnres_mlp = SlBf16(K3_HIDDEN,salt + 4u,0.5f,0.0f);
		uint16_t *decay_down = SlBf16((uint64_t)K3_KDA_KEY_DIM * K3_HIDDEN,salt + 5u,0.02f,0.0f);
		uint16_t *router = SlBf16((uint64_t)K3_EXPERTS * K3_HIDDEN,salt + 6u,0.02f,0.0f),*routed_norm = SlBf16(K3_ROUTED_EXPERT_HIDDEN,salt + 7u,0.1f,1.0f);
		float *decay_bias = SlF32((uint64_t)K3_KDA_HEADS * K3_KDA_KEY_DIM,salt + 8u,0.5f,0.0f),*log_scale = SlF32(K3_KDA_HEADS,salt + 9u,0.5f,0.0f);
		float *out_norm = SlF32(K3_KDA_VALUE_DIM,salt + 10u,0.1f,1.0f),*bias = SlF32(K3_EXPERTS,salt + 11u,0.01f,0.0f);
		for (uint32_t shard = 0u; shard < SL_SHARDS; shard++)
		{
			K3LayerWeights *w = &shard_weights[(uint64_t)layer * SL_SHARDS + shard];
			const uint32_t s = salt + 100u + 40u * shard;
			memset(w,0,sizeof(*w));
			w->attn_norm_weight = attn_norm;
			w->mlp_norm_weight = mlp_norm;
			w->attnres_attn_weight = attnres_attn;
			w->attnres_mlp_weight = attnres_mlp;
			w->kda_qkv_beta_weight = SlBf16((uint64_t)SL_QKVB * K3_HIDDEN,s + 1u,0.02f,0.0f);
			w->kda_decay_down_weight = decay_down;
			w->kda_decay_up_weight = SlBf16((uint64_t)SL_QK * K3_KDA_KEY_DIM,s + 2u,0.1f,0.0f);
			w->kda_decay_bias = decay_bias;
			w->kda_head_log_scale = log_scale;
			w->kda_gate_weight = SlBf16((uint64_t)SL_V * K3_HIDDEN,s + 3u,0.02f,0.0f);
			w->kda_out_norm_weight = out_norm;
			w->kda_out_weight = SlBf16((uint64_t)K3_HIDDEN * SL_V,s + 4u,0.05f,0.0f);
			w->kda_q_conv_weight = SlF32((uint64_t)SL_QK * K3_KDA_CONV_KERNEL,s + 5u,0.5f,0.0f);
			w->kda_k_conv_weight = SlF32((uint64_t)SL_QK * K3_KDA_CONV_KERNEL,s + 6u,0.5f,0.0f);
			w->kda_v_conv_weight = SlF32((uint64_t)SL_V * K3_KDA_CONV_KERNEL,s + 7u,0.5f,0.0f);
			if ( layer < K3_FIRST_ROUTED_LAYER )
			{
				w->dense_gate_up_weight = SlBf16((uint64_t)SL_DENSE_ROWS * K3_HIDDEN,s + 8u,0.02f,0.0f);
				w->dense_down_weight = SlBf16((uint64_t)K3_HIDDEN * SL_DENSE_IN,s + 9u,0.02f,0.0f);
			}
			else
			{
				uint8_t *w1 = SlAlloc<uint8_t>(w1_bytes),*w2 = SlAlloc<uint8_t>(w2_bytes);
				SlFillCells<<<1024,256>>>(w1,w1_bytes,s + 10u);
				SlFillCells<<<1024,256>>>(w2,w2_bytes,s + 11u);
				w->router_weight = router;
				w->router_bias = bias;
				w->routed_norm_weight = routed_norm;
				w->routed_down_weight = SlBf16((uint64_t)SL_SLICE * K3_HIDDEN,s + 12u,0.02f,0.0f);
				w->routed_up_weight = SlBf16((uint64_t)K3_HIDDEN * SL_SLICE,s + 13u,0.05f,0.0f);
				w->shared_w1_weight = SlBf16((uint64_t)SL_SHARED_ROWS * K3_HIDDEN,s + 14u,0.02f,0.0f);
				w->shared_w2_weight = SlBf16((uint64_t)K3_HIDDEN * SL_SHARED_IN,s + 15u,0.05f,0.0f);
				w->expert_w1_weight = w1;
				w->expert_w2_weight = w2;
				w->expert_interleave = 1u;
				w->expert_tile_k = 32u;
			}
			K3StreamedKdaState *st = &states[(uint64_t)layer * SL_SHARDS + shard];
			st->state_pool = SlAlloc<uint8_t>(SL_SLOT_BYTES);
			st->q_window = SlAlloc<uint16_t>((uint64_t)SL_QK * K3_KDA_CONV_KERNEL);
			st->k_window = SlAlloc<uint16_t>((uint64_t)SL_QK * K3_KDA_CONV_KERNEL);
			st->v_window = SlAlloc<uint16_t>((uint64_t)SL_V * K3_KDA_CONV_KERNEL);
		}
	}
	SL_CUDA(cudaDeviceSynchronize());
	pthread_barrier_init(&sl_barrier,0,SL_SHARDS);
	for (uint32_t rank = 0u; rank < SL_SHARDS; rank++)
	{
		SlRank *r = &sl_ranks[rank];
		SlBuffers(&r->buffers,rank,state_index,row_begin,SL_ROWS);
		for (uint32_t layer = 0u; layer < SL_LAYERS; layer++)
			r->weights[layer] = shard_weights[(uint64_t)layer * SL_SHARDS + rank];
		SL_CUDA(cudaMemcpy(r->buffers.hidden_bf16,hidden,(uint64_t)SL_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
		r->fused = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
		r->sum = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
		memset(&r->state,0,sizeof(r->state));
		r->state.layer_collective = SlCollective;
		r->state.collective_context = r;
		r->state.sequences = 1u;
		r->state.kda_rank_heads = SL_HEADS;
		r->state.kda_state = SlAlloc<uint8_t>((uint64_t)SL_LAYERS * SL_SLOT_BYTES);
		r->state.kda_q_window = SlAlloc<uint16_t>((uint64_t)SL_LAYERS * SL_QK * K3_KDA_CONV_KERNEL);
		r->state.kda_k_window = SlAlloc<uint16_t>((uint64_t)SL_LAYERS * SL_QK * K3_KDA_CONV_KERNEL);
		r->state.kda_v_window = SlAlloc<uint16_t>((uint64_t)SL_LAYERS * SL_V * K3_KDA_CONV_KERNEL);
		SL_CUDA(cudaStreamCreateWithFlags(&r->stream,cudaStreamNonBlocking));
		SL_CUDA(cudaEventCreateWithFlags(&r->contributed,cudaEventDisableTiming));
		SL_CUDA(cudaEventCreateWithFlags(&r->consumed,cudaEventDisableTiming));
	}
	{
		pthread_t threads[SL_SHARDS];
		for (uint32_t rank = 0u; rank < SL_SHARDS; rank++)
			pthread_create(&threads[rank],0,SlRankThread,&sl_ranks[rank]);
		for (uint32_t rank = 0u; rank < SL_SHARDS; rank++)
			pthread_join(threads[rank],0);
	}
	for (uint32_t rank = 0u; rank < SL_SHARDS; rank++)
		SlCheck(sl_ranks[rank].status == LM_LAUNCH_OK,"every TP16 rank runs the two-layer slice");
	SlBuffers(&streamed,0u,state_index,row_begin,SL_ROWS);
	K3StreamedJoin join;
	join.weight = SlAlloc<uint16_t>((uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM);
	join.weight_capacity = (uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM;
	join.activation = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_MLA_HEADS * K3_V_HEAD_DIM);
	join.activation_capacity = (uint64_t)SL_ROWS * K3_MLA_HEADS * K3_V_HEAD_DIM;
	SL_CUDA(cudaMemcpy(streamed.hidden_bf16,hidden,(uint64_t)SL_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	SL_CUDA(cudaStreamCreate(&stream));
	SlCheck(K3StreamedSlice<LmMxfp4,K3GlobalKv,SL_MLA_HEADS>(&streamed,shard_weights.data(),states.data(),SL_SHARDS,0,0,&join,0u,SL_LAYERS,partial,latent_accumulator,hidden_accumulator,
		SL_ROWS,SL_PACKED,48u,stream) == LM_LAUNCH_OK,"the streamed slice runs both layers' sixteen shards on one device");
	SL_CUDA(cudaStreamSynchronize(stream));
	{
		std::vector<uint16_t> want((uint64_t)SL_ROWS * K3_HIDDEN),got(want.size());
		double worst = 0.0,scale = 0.0,squared = 0.0,error = 0.0;
		uint64_t nonfinite = 0u;
		SL_CUDA(cudaMemcpy(want.data(),sl_ranks[0].buffers.hidden_bf16,want.size() * 2u,cudaMemcpyDeviceToHost));
		SL_CUDA(cudaMemcpy(got.data(),streamed.hidden_bf16,got.size() * 2u,cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < want.size(); index++)
		{
			const double r = SlHostFloat(want[index]),c = SlHostFloat(got[index]);
			if ( !isfinite(r) || !isfinite(c) )
			{
				nonfinite++;
				continue;
			}
			scale = fmax(scale,fabs(r));
			worst = fmax(worst,fabs(r - c));
			squared += r * r;
			error += (r - c) * (r - c);
		}
		SlCheck(nonfinite == 0u,"every output element is finite");
		SlCheck(scale > 0.0,"the slice produces a hidden state");
		SlCheck(sqrt(error / (squared > 0.0 ? squared : 1.0)) <= 0.01,"the streamed slice matches TP16 within 1% relative RMS");
		printf("%s streamed two-layer slice vs TP16: worst |diff| %.4g, largest output %.4g, relative RMS %.3g, nonfinite %llu\n",
			sl_failures == 0 ? "PASS" : "FAIL",worst,scale,sqrt(error / (squared > 0.0 ? squared : 1.0)),(unsigned long long)nonfinite);
	}
	{
		uint16_t *tails = SlAlloc<uint16_t>((uint64_t)SL_NODES * SL_SHARDS * SL_TAIL),*gathered = SlAlloc<uint16_t>((uint64_t)SL_ROWS * K3_HIDDEN);
		uint8_t *summaries = SlAlloc<uint8_t>((uint64_t)SL_NODES * SL_SHARDS * 2u * SL_SLOT_BYTES);
		uint32_t *node_begin = SlAlloc<uint32_t>(2u),node_rows[2] = {0u,SL_NODE_ROWS};
		std::vector<uint16_t> want((uint64_t)SL_ROWS * K3_HIDDEN),got(want.size());
		double squared = 0.0,error = 0.0,worst = 0.0;
		uint64_t nonfinite = 0u;
		SL_CUDA(cudaMemcpy(node_begin,node_rows,sizeof(node_rows),cudaMemcpyHostToDevice));
		for (uint32_t node = 0u; node < SL_NODES; node++)
		{
			SlNode *n = &sl_nodes[node];
			K3StreamedKdaExchange *x = &n->exchange;
			n->node = node;
			n->weights = shard_weights.data();
			n->states = new K3StreamedKdaState[(uint64_t)SL_LAYERS * SL_SHARDS];
			for (uint64_t index = 0u; index < (uint64_t)SL_LAYERS * SL_SHARDS; index++)
			{
				n->states[index].state_pool = SlAlloc<uint8_t>(SL_SLOT_BYTES);
				n->states[index].q_window = SlAlloc<uint16_t>((uint64_t)SL_QK * K3_KDA_CONV_KERNEL);
				n->states[index].k_window = SlAlloc<uint16_t>((uint64_t)SL_QK * K3_KDA_CONV_KERNEL);
				n->states[index].v_window = SlAlloc<uint16_t>((uint64_t)SL_V * K3_KDA_CONV_KERNEL);
			}
			SlBuffers(&n->buffers,0u,state_index,node_begin,SL_NODE_ROWS);
			n->partial = SlAlloc<uint16_t>((uint64_t)SL_NODE_ROWS * K3_HIDDEN);
			n->join.weight = SlAlloc<uint16_t>((uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM);
			n->join.weight_capacity = (uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM;
			n->join.activation = SlAlloc<uint16_t>((uint64_t)SL_NODE_ROWS * K3_MLA_HEADS * K3_V_HEAD_DIM);
			n->join.activation_capacity = (uint64_t)SL_NODE_ROWS * K3_MLA_HEADS * K3_V_HEAD_DIM;
			n->latent_accumulator = SlAlloc<float>((uint64_t)SL_NODE_ROWS * K3_ROUTED_EXPERT_HIDDEN);
			n->hidden_accumulator = SlAlloc<float>((uint64_t)SL_NODE_ROWS * K3_HIDDEN);
			n->hidden = hidden;
			n->gathered = gathered;
			memset(x,0,sizeof(*x));
			x->node = node;
			x->nodes = SL_NODES;
			x->tails = tails;
			x->summaries = summaries;
			x->chunks = 1u;
			x->chunk_row_begin = node_begin;
			x->chunk_slots = state_index;
			x->ends = SlAlloc<float>(SL_SLOT_BYTES / 4u);
			x->transforms = SlAlloc<float>(SL_SLOT_BYTES / 4u);
			x->incoming = SlAlloc<float>(SL_SLOT_BYTES / 4u);
			x->running = SlAlloc<float>(2u * SL_SLOT_BYTES / 4u);
			x->q_windows = SlAlloc<uint16_t>((uint64_t)SL_QK * K3_KDA_CONV_KERNEL);
			x->k_windows = SlAlloc<uint16_t>((uint64_t)SL_QK * K3_KDA_CONV_KERNEL);
			x->v_windows = SlAlloc<uint16_t>((uint64_t)SL_V * K3_KDA_CONV_KERNEL);
			x->zeros = SlAlloc<uint16_t>((uint64_t)SL_NODE_ROWS * SL_V);
			x->scratch = SlAlloc<uint16_t>((uint64_t)SL_NODE_ROWS * SL_V);
			x->exchange = SlNodeExchange;
			SL_CUDA(cudaStreamCreate(&n->stream));
		}
		SL_CUDA(cudaDeviceSynchronize());
		pthread_barrier_init(&sl_node_barrier,0,SL_NODES);
		{
			pthread_t threads[SL_NODES];
			for (uint32_t node = 0u; node < SL_NODES; node++)
				pthread_create(&threads[node],0,SlNodeThread,&sl_nodes[node]);
			for (uint32_t node = 0u; node < SL_NODES; node++)
				pthread_join(threads[node],0);
		}
		for (uint32_t node = 0u; node < SL_NODES; node++)
			SlCheck(sl_nodes[node].status == LM_LAUNCH_OK,"every node runs the two-layer slice for both passes");
		SL_CUDA(cudaMemcpy(want.data(),streamed.hidden_bf16,want.size() * 2u,cudaMemcpyDeviceToHost));
		SL_CUDA(cudaMemcpy(got.data(),gathered,got.size() * 2u,cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < want.size(); index++)
		{
			const double r = SlHostFloat(want[index]),c = SlHostFloat(got[index]);
			if ( !isfinite(r) || !isfinite(c) )
			{
				nonfinite++;
				continue;
			}
			worst = fmax(worst,fabs(r - c));
			squared += r * r;
			error += (r - c) * (r - c);
		}
		SlCheck(nonfinite == 0u,"every node output element is finite");
		SlCheck(sqrt(error / (squared > 0.0 ? squared : 1.0)) <= 0.002,"the slice on four nodes over two passes matches the one-device streamed slice within 0.2% relative RMS");
		printf("%s streamed two-layer slice on %u nodes x %u passes vs one device: worst |diff| %.4g, relative RMS %.3g, nonfinite %llu\n",
			sl_failures == 0 ? "PASS" : "FAIL",SL_NODES,SL_PASSES,worst,sqrt(error / (squared > 0.0 ? squared : 1.0)),(unsigned long long)nonfinite);
	}
	return sl_failures != 0;
}
