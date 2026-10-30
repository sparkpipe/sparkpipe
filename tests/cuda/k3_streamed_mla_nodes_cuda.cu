#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"
#include "tests/cuda/node_collective_emulation.cuh"

#define ML_SHARDS 16u
#ifndef ML_ROWS
#define ML_ROWS 256u
#endif
#define ML_NODES 4u
#define ML_PASSES 2u
#define ML_PASS_ROWS (ML_ROWS / ML_PASSES)
#define ML_NODE_ROWS (ML_PASS_ROWS / ML_NODES)
#define ML_HEADS (K3_MLA_HEADS / ML_SHARDS)
#define ML_PAGES ((ML_ROWS + K3_KV_PAGE_SLOTS - 1u) / K3_KV_PAGE_SLOTS)

#define ML_CUDA(call) do { cudaError_t ml_error = (call); if ( ml_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(ml_error)); exit(1); } } while (0)

static int ml_failures;

static void MlCheck(int condition,const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	ml_failures++;
}

__global__ void MlFillBf16(uint16_t *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = LmFloatToBf16(offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

static double MlHostFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

template<class T>
static T *MlAlloc(uint64_t count)
{
	T *device;
	ML_CUDA(cudaMalloc((void **)&device,count * sizeof(T) + 256u));
	ML_CUDA(cudaMemset(device,0,count * sizeof(T) + 256u));
	return device;
}

static uint16_t *MlBf16(uint64_t count,uint32_t salt,float scale,float offset)
{
	uint16_t *data = MlAlloc<uint16_t>(count);
	MlFillBf16<<<512,256>>>(data,count,salt,scale,offset);
	return data;
}

static uint32_t *MlDevice(const std::vector<uint32_t> &host)
{
	uint32_t *device = MlAlloc<uint32_t>(host.size());
	ML_CUDA(cudaMemcpy(device,host.data(),host.size() * 4u,cudaMemcpyHostToDevice));
	return device;
}

static void MlBuffers(K3LayerBuffers *b,uint32_t rows,const uint32_t *sequence_of_row,const uint32_t *context,const uint32_t *positions)
{
	memset(b,0,sizeof(*b));
	b->hidden_bf16 = MlAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->normed_bf16 = MlAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->latent_bf16 = MlAlloc<uint16_t>((uint64_t)rows * K3_Q_LORA_RANK);
	b->kv_slot_bf16 = MlAlloc<uint16_t>((uint64_t)rows * K3_MLA_KV_A_DIM);
	b->query_bf16 = MlAlloc<uint16_t>((uint64_t)rows * ML_HEADS * K3_MHA_QK);
	b->attention_out_bf16 = MlAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->value_bf16 = MlAlloc<uint16_t>((uint64_t)rows * ML_HEADS * K3_V_HEAD_DIM);
	b->gate_bf16 = MlAlloc<uint16_t>((uint64_t)rows * ML_HEADS * K3_V_HEAD_DIM);
	b->attnres_partial_bf16 = MlAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->dense_row_offset = MlDevice(std::vector<uint32_t>{0u,rows});
	b->dense_tile_prefix = MlAlloc<uint32_t>(2u);
	b->sequence_of_row = sequence_of_row;
	b->context_length = context;
	b->positions = positions;
	b->tp_sharded = 1u;
	b->mla_heads_rank = ML_HEADS;
	b->mla_gate_rows = ML_HEADS * K3_V_HEAD_DIM;
	b->mla_out_input = ML_HEADS * K3_V_HEAD_DIM;
}

static LmKvShardView MlCache(uint32_t *table,LmKvAccessError *error)
{
	SparkKvShard shard = {1u,0u,1u};
	LmKvShardView view;
	const uint64_t bytes = SparkKvShardPoolBytes(shard,K3GlobalKv::kPageSlots,K3GlobalKv::kSlotBytes,ML_PAGES);
	uint8_t *pool = MlAlloc<uint8_t>(bytes);
	if ( LmKvShardViewInitialize<K3GlobalKv>(&view,pool,table,ML_PAGES,1u,ML_PAGES,error,shard) != 0 )
	{
		printf("FAIL cache view\n");
		exit(1);
	}
	return view;
}

struct MlNode
{
	uint32_t node;
	uint32_t nodes;
	uint32_t rows;
	const K3LayerWeights *shards;
	K3StreamedMlaExchange exchange[ML_PASSES];
	K3LayerBuffers buffers[ML_PASSES];
	K3StreamedJoin join;
	NodeEmulationNode peer;
	uint16_t *partial;
	const uint16_t *hidden;
	const uint16_t *residual;
	uint16_t *gathered;
	uint8_t *latent_keys;
	uint32_t stored;
	int32_t status;
	cudaStream_t stream;
};

static NodeEmulationWorld ml_world,ml_alone_world;

static void *MlNodeThread(void *argument)
{
	MlNode *n = (MlNode *)argument;
	const uint64_t row_bytes = (uint64_t)K3_HIDDEN * 2u;
	n->status = LM_LAUNCH_OK;
	for (uint32_t pass = 0u; pass < ML_PASSES && n->status == LM_LAUNCH_OK; pass++)
	{
		K3LayerBuffers *b = &n->buffers[pass];
		const uint64_t first = (uint64_t)pass * ML_PASS_ROWS + (uint64_t)n->node * n->rows;
		ML_CUDA(cudaMemcpyAsync(b->hidden_bf16,n->hidden + first * K3_HIDDEN,n->rows * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		ML_CUDA(cudaMemcpyAsync(b->attnres_partial_bf16,n->residual + first * K3_HIDDEN,n->rows * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		if ( n->stored != 0u )
			ML_CUDA(cudaMemsetAsync(n->latent_keys,0xff,(uint64_t)ML_ROWS * K3GlobalKv::kSlotBytes,n->stream));
		n->status = K3StreamedMlaNode<LmMxfp4,K3GlobalKv,ML_HEADS>(b,n->shards,ML_SHARDS,&n->exchange[pass],n->partial,&n->join,0u,n->rows,48u,n->stream);
		ML_CUDA(cudaMemcpyAsync(n->gathered + first * K3_HIDDEN,b->attnres_partial_bf16,n->rows * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		ML_CUDA(cudaStreamSynchronize(n->stream));
	}
	return 0;
}

static void MlSetup(MlNode *n,uint32_t node,uint32_t nodes,NodeEmulationWorld *world,const K3LayerWeights *shards,uint32_t *table_device,
	LmKvAccessError *error_device,const uint16_t *hidden,const uint16_t *partial_in,uint16_t *gathered,uint32_t stored)
{
	const SparkKvShard store_shard = {nodes,node,4u};
	LmKvShardView store;
	const uint32_t rows = ML_PASS_ROWS / nodes;
	n->nodes = nodes;
	n->rows = rows;
	uint16_t *query = MlAlloc<uint16_t>((uint64_t)ML_PASS_ROWS * ML_HEADS * K3_MHA_QK);
	uint8_t *expanded = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3StreamedHeadKv<ML_HEADS,K3GlobalKv>::kSlotBytes);
	float *partials = MlAlloc<float>((uint64_t)ML_PASS_ROWS * LmGqaShardRecordFloats(ML_HEADS,K3_V_HEAD_DIM,1u));
	const uint32_t *pass_row_offset = MlDevice(std::vector<uint32_t>{0u,ML_PASS_ROWS});
	uint32_t *pass_tile_prefix = MlAlloc<uint32_t>(2u);
	const uint32_t *pass_sequence = MlDevice(std::vector<uint32_t>(ML_PASS_ROWS,0u)),*node_sequence = MlDevice(std::vector<uint32_t>(rows,0u));
	const uint32_t *key_offset = MlDevice(std::vector<uint32_t>{0u});
	LmKvShardView cache = MlCache(table_device,error_device);
	uint16_t *q_latents = MlAlloc<uint16_t>((uint64_t)ML_PASS_ROWS * K3_Q_LORA_RANK);
	uint8_t *latent_keys = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3GlobalKv::kSlotBytes);
	uint16_t *merged = MlAlloc<uint16_t>((uint64_t)ML_PASS_ROWS * ML_HEADS * K3_V_HEAD_DIM);
	uint16_t *value_send = MlAlloc<uint16_t>((uint64_t)ML_PASS_ROWS * (ML_SHARDS / nodes) * ML_HEADS * K3_V_HEAD_DIM);
	uint16_t *values = MlAlloc<uint16_t>((uint64_t)ML_SHARDS * rows * ML_HEADS * K3_V_HEAD_DIM);
	n->node = node;
	n->peer.world = world;
	n->peer.node = node;
	n->shards = shards;
	n->partial = MlAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	n->join.weight = MlAlloc<uint16_t>((uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM);
	n->join.weight_capacity = (uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM;
	n->join.activation = MlAlloc<uint16_t>((uint64_t)(rows) * K3_MLA_HEADS * K3_V_HEAD_DIM);
	n->join.activation_capacity = (uint64_t)(rows) * K3_MLA_HEADS * K3_V_HEAD_DIM;
	n->hidden = hidden;
	n->residual = partial_in;
	n->gathered = gathered;
	n->latent_keys = latent_keys;
	n->stored = stored;
	memset(&store,0,sizeof(store));
	if ( stored != 0u && LmKvShardViewInitialize<K3GlobalKv>(&store,MlAlloc<uint8_t>(SparkKvShardPoolBytes(store_shard,K3GlobalKv::kPageSlots,K3GlobalKv::kSlotBytes,ML_PAGES)),
			table_device,ML_PAGES,1u,ML_PAGES,error_device,store_shard) != 0 )
	{
		printf("FAIL store view\n");
		exit(1);
	}
	for (uint32_t pass = 0u; pass < ML_PASSES; pass++)
	{
		K3StreamedMlaExchange *x = &n->exchange[pass];
		std::vector<uint32_t> pass_positions(ML_PASS_ROWS),node_positions(rows);
		for (uint32_t row = 0u; row < ML_PASS_ROWS; row++)
			pass_positions[row] = pass * ML_PASS_ROWS + row;
		for (uint32_t row = 0u; row < rows; row++)
			node_positions[row] = pass * ML_PASS_ROWS + node * rows + row;
		MlBuffers(&n->buffers[pass],rows,node_sequence,MlDevice(std::vector<uint32_t>{(pass + 1u) * ML_PASS_ROWS}),MlDevice(node_positions));
		n->buffers[pass].cache_shard = cache;
		n->buffers[pass].kv_shard = cache.shard;
		memset(x,0,sizeof(*x));
		x->node = node;
		x->nodes = nodes;
		x->pass_rows = ML_PASS_ROWS;
		x->first_key = pass * ML_PASS_ROWS;
		x->key_capacity = ML_ROWS;
		x->q_latents = q_latents;
		x->latent_keys = latent_keys;
		x->query = query;
		x->expanded = expanded;
		x->partials = partials;
		x->merged = merged;
		x->value_send = value_send;
		x->values = values;
		x->pass_positions = MlDevice(pass_positions);
		x->pass_sequence_of_row = pass_sequence;
		x->pass_row_offset = pass_row_offset;
		x->pass_tile_prefix = pass_tile_prefix;
		x->key_offset = key_offset;
		x->key_context = MlDevice(std::vector<uint32_t>{(pass + 1u) * ML_PASS_ROWS});
		x->collective = NodeEmulationCollective;
		x->context = &n->peer;
		if ( stored != 0u )
		{
			x->store = store;
			x->store_sequence_of_row = pass_sequence;
			x->older_sequence = key_offset;
			x->older_offset = key_offset;
			x->older_context = MlDevice(std::vector<uint32_t>{pass * ML_PASS_ROWS});
			x->older_send = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3GlobalKv::kSlotBytes);
			x->older_receive = expanded;
			x->older_capacity = stored > 1u ? (uint64_t)nodes * 5u * K3GlobalKv::kSlotBytes : (uint64_t)ML_ROWS * K3StreamedHeadKv<ML_HEADS,K3GlobalKv>::kSlotBytes;
		}
		if ( stored > 1u )
		{
			x->key_block = 48u;
			x->expanded_keys = 48u;
			x->running = MlAlloc<float>((uint64_t)ML_PASS_ROWS * LmGqaShardRecordFloats(ML_HEADS,K3_V_HEAD_DIM,1u));
		}
	}
	ML_CUDA(cudaStreamCreate(&n->stream));
}

int main(void)
{
	std::vector<K3LayerWeights> shards(ML_SHARDS);
	std::vector<uint32_t> table(ML_PAGES),positions(ML_ROWS);
	static MlNode nodes[ML_NODES],stored[ML_NODES],blocked[ML_NODES];
	static NodeEmulationWorld ml_stored_world,ml_blocked_world;
	uint16_t *hidden = MlBf16((uint64_t)ML_ROWS * K3_HIDDEN,1u,1.0f,0.0f),*partial_in = MlBf16((uint64_t)ML_ROWS * K3_HIDDEN,2u,1.0f,0.0f);
	uint16_t *attn_norm = MlBf16(K3_HIDDEN,3u,0.1f,1.0f),*q_norm = MlBf16(K3_Q_LORA_RANK,4u,0.1f,1.0f),*kv_norm = MlBf16(K3_KV_LORA_RANK,5u,0.1f,1.0f);
	uint16_t *q_down = MlBf16((uint64_t)K3_Q_LORA_RANK * K3_HIDDEN,6u,0.02f,0.0f),*kv_a = MlBf16((uint64_t)K3_MLA_KV_A_DIM * K3_HIDDEN,7u,0.02f,0.0f);
	uint16_t *q_b = MlBf16((uint64_t)K3_MLA_HEADS * K3_MHA_QK * K3_Q_LORA_RANK,8u,0.03f,0.0f);
	uint16_t *kv_b_key = MlBf16((uint64_t)K3_MLA_HEADS * K3_QK_NOPE_DIM * K3_KV_LORA_RANK,9u,0.05f,0.0f);
	uint16_t *kv_b_value = MlBf16((uint64_t)K3_MLA_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK,10u,0.05f,0.0f);
	uint16_t *scratch = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN),*gathered = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	float *accumulator = MlAlloc<float>((uint64_t)ML_ROWS * K3_HIDDEN);
	LmKvAccessError *error_device = MlAlloc<LmKvAccessError>(1u),error_host;
	uint32_t *table_device,*plan_device;
	K3LayerBuffers streamed;
	K3StreamedMlaScratch mla;
	cudaStream_t stream;
	for (uint32_t page = 0u; page < ML_PAGES; page++)
		table[page] = page;
	for (uint32_t row = 0u; row < ML_ROWS; row++)
		positions[row] = row;
	table_device = MlDevice(table);
	plan_device = MlDevice(std::vector<uint32_t>{0u,0u,ML_ROWS});
	LmKvAccessErrorReset(&error_host);
	ML_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	for (uint32_t shard = 0u; shard < ML_SHARDS; shard++)
	{
		K3LayerWeights *w = &shards[shard];
		memset(w,0,sizeof(*w));
		w->attn_norm_weight = attn_norm;
		w->mla_q_down_weight = q_down;
		w->mla_q_norm_weight = q_norm;
		w->mla_kv_a_weight = kv_a;
		w->mla_kv_a_norm_weight = kv_norm;
		w->mla_q_b_weight = q_b + (uint64_t)shard * ML_HEADS * K3_MHA_QK * K3_Q_LORA_RANK;
		w->mla_kv_b_key_full_weight = kv_b_key;
		w->mla_kv_b_value_full_weight = kv_b_value;
		w->mla_gate_weight = MlBf16((uint64_t)ML_HEADS * K3_V_HEAD_DIM * K3_HIDDEN,20u + shard,0.02f,0.0f);
		w->mla_out_weight = MlBf16((uint64_t)K3_HIDDEN * ML_HEADS * K3_V_HEAD_DIM,40u + shard,0.05f,0.0f);
	}
	MlBuffers(&streamed,ML_ROWS,MlDevice(std::vector<uint32_t>(ML_ROWS,0u)),MlDevice(std::vector<uint32_t>{ML_ROWS}),MlDevice(positions));
	streamed.cache_shard = MlCache(table_device,error_device);
	streamed.kv_shard = streamed.cache_shard.shard;
	streamed.shard_gather_list = plan_device;
	streamed.shard_gather_offset = plan_device + 1u;
	streamed.shard_gather_context = plan_device + 2u;
	mla.latent_keys = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3GlobalKv::kSlotBytes);
	mla.expanded = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3StreamedHeadKv<ML_HEADS,K3GlobalKv>::kSlotBytes);
	mla.partials = MlAlloc<float>((uint64_t)ML_ROWS * LmGqaShardRecordFloats(ML_HEADS,K3_V_HEAD_DIM,1u));
	mla.key_capacity = ML_ROWS;
	ML_CUDA(cudaMemcpy(streamed.hidden_bf16,hidden,(uint64_t)ML_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	ML_CUDA(cudaMemcpy(streamed.attnres_partial_bf16,partial_in,(uint64_t)ML_ROWS * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	ML_CUDA(cudaDeviceSynchronize());
	ML_CUDA(cudaStreamCreate(&stream));
	MlCheck(K3StreamedMla<LmMxfp4,K3GlobalKv,ML_HEADS>(&streamed,shards.data(),ML_SHARDS,&mla,scratch,accumulator,0u,ML_ROWS,ML_ROWS,48u,stream) == LM_LAUNCH_OK,
		"the one-device streamed MLA runs all rows");
	ML_CUDA(cudaStreamSynchronize(stream));
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlSetup(&nodes[node],node,ML_NODES,&ml_world,shards.data(),table_device,error_device,hidden,partial_in,gathered,0u);
	uint16_t *stored_gathered = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlSetup(&stored[node],node,ML_NODES,&ml_stored_world,shards.data(),table_device,error_device,hidden,partial_in,stored_gathered,1u);
	uint16_t *blocked_gathered = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlSetup(&blocked[node],node,ML_NODES,&ml_blocked_world,shards.data(),table_device,error_device,hidden,partial_in,blocked_gathered,2u);
	static MlNode single;
	uint16_t *single_gathered = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	MlSetup(&single,0u,1u,&ml_alone_world,shards.data(),table_device,error_device,hidden,partial_in,single_gathered,0u);
	ML_CUDA(cudaDeviceSynchronize());
	NodeEmulationStart(&ml_alone_world,1u);
	MlNodeThread(&single);
	MlCheck(single.status == LM_LAUNCH_OK,"one node runs both passes");
	NodeEmulationStart(&ml_world,ML_NODES);
	{
		pthread_t threads[ML_NODES];
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_create(&threads[node],0,MlNodeThread,&nodes[node]);
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_join(threads[node],0);
	}
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlCheck(nodes[node].status == LM_LAUNCH_OK,"every node runs both passes");
	NodeEmulationStart(&ml_stored_world,ML_NODES);
	{
		pthread_t threads[ML_NODES];
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_create(&threads[node],0,MlNodeThread,&stored[node]);
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_join(threads[node],0);
	}
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlCheck(stored[node].status == LM_LAUNCH_OK,"every node runs both passes with keys in the decode store");
	NodeEmulationStart(&ml_blocked_world,ML_NODES);
	{
		pthread_t threads[ML_NODES];
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_create(&threads[node],0,MlNodeThread,&blocked[node]);
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_join(threads[node],0);
	}
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlCheck(blocked[node].status == LM_LAUNCH_OK,"every node runs both passes expanding keys in blocks of 48 and gathering older keys in chunks of 5");
	{
		std::vector<uint16_t> want((uint64_t)ML_ROWS * K3_HIDDEN),got(want.size()),before(want.size());
		double worst = 0.0,squared = 0.0,error = 0.0;
		uint64_t nonfinite = 0u;
		ML_CUDA(cudaMemcpy(want.data(),streamed.attnres_partial_bf16,want.size() * 2u,cudaMemcpyDeviceToHost));
		ML_CUDA(cudaMemcpy(got.data(),gathered,got.size() * 2u,cudaMemcpyDeviceToHost));
		ML_CUDA(cudaMemcpy(before.data(),partial_in,before.size() * 2u,cudaMemcpyDeviceToHost));
		ML_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < want.size(); index++)
		{
			const double r = MlHostFloat(want[index]),c = MlHostFloat(got[index]),b = MlHostFloat(before[index]);
			if ( !isfinite(r) || !isfinite(c) )
			{
				nonfinite++;
				continue;
			}
			worst = fmax(worst,fabs(r - c));
			squared += (r - b) * (r - b);
			error += (r - c) * (r - c);
		}
		const double rms = sqrt(error / (squared > 0.0 ? squared : 1.0));
		double node_squared = 0.0,node_error = 0.0;
		ML_CUDA(cudaMemcpy(want.data(),single_gathered,want.size() * 2u,cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < want.size(); index++)
		{
			const double r = MlHostFloat(want[index]),c = MlHostFloat(got[index]),b = MlHostFloat(before[index]);
			if ( isfinite(r) && isfinite(c) )
			{
				node_squared += (r - b) * (r - b);
				node_error += (r - c) * (r - c);
			}
		}
		const double node_rms = sqrt(node_error / (node_squared > 0.0 ? node_squared : 1.0));
		std::vector<uint16_t> from_store(want.size());
		ML_CUDA(cudaMemcpy(from_store.data(),stored_gathered,from_store.size() * 2u,cudaMemcpyDeviceToHost));
		MlCheck(memcmp(from_store.data(),got.data(),got.size() * 2u) == 0,
			"nodes that keep no keys between passes (older keys gathered back from the owners' decode store, every node's key buffer wiped) give bit-identical output");
		std::vector<uint16_t> from_blocks(want.size());
		double block_squared = 0.0,block_error = 0.0;
		ML_CUDA(cudaMemcpy(from_blocks.data(),blocked_gathered,from_blocks.size() * 2u,cudaMemcpyDeviceToHost));
		for (uint64_t index = 0u; index < got.size(); index++)
		{
			const double r = MlHostFloat(got[index]),c = MlHostFloat(from_blocks[index]),b = MlHostFloat(before[index]);
			if ( isfinite(r) && isfinite(c) )
			{
				block_squared += (r - b) * (r - b);
				block_error += (r - c) * (r - c);
			}
			else
				nonfinite++;
		}
		const double block_rms = sqrt(block_error / (block_squared > 0.0 ? block_squared : 1.0));
		MlCheck(block_rms <= 0.01,"expanding keys in blocks with the running softmax fold matches one expansion of all keys within 1% relative RMS of the attention contribution (probabilities enter the value product in bf16 against each block's own maximum)");
		MlCheck(node_rms <= 0.002,"four nodes over two passes match the same MLA run as one node within 0.2% relative RMS of the attention contribution");
		MlCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access errors");
		MlCheck(nonfinite == 0u,"every output element is finite");
		MlCheck(squared > 0.0,"the MLA layer changes the residual");
		MlCheck(rms <= 0.01,"four nodes over two passes, gathering latents and returning head-group values, match the one-device streamed MLA (sixteen bf16-rounded output partials) within 1% relative RMS of the attention contribution");
		printf("%s MLA on %u nodes x %u passes vs one node: relative RMS %.3g; vs one-device streamed MLA: %.3g (worst %.4g), nonfinite %llu; keys from the decode store bit-identical; key blocks %.3g\n",
			ml_failures == 0 ? "PASS" : "FAIL",ML_NODES,ML_PASSES,node_rms,rms,worst,(unsigned long long)nonfinite,block_rms);
	}
	return ml_failures != 0;
}
