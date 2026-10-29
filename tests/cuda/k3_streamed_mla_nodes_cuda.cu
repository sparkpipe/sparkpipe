#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"

#define ML_SHARDS 16u
#define ML_ROWS 256u
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
	const K3LayerWeights *shards;
	K3StreamedMlaExchange exchange[ML_PASSES];
	K3LayerBuffers buffers[ML_PASSES];
	uint16_t *partial;
	float *accumulator;
	const uint16_t *hidden;
	const uint16_t *residual;
	uint16_t *gathered;
	int32_t status;
	cudaStream_t stream;
};

static pthread_barrier_t ml_barrier;

static void MlExchange(void *context,void *stream,uint32_t phase,uint32_t shard)
{
	(void)context;
	(void)phase;
	(void)shard;
	ML_CUDA(cudaStreamSynchronize((cudaStream_t)(uintptr_t)stream));
	pthread_barrier_wait(&ml_barrier);
}

static void *MlNodeThread(void *argument)
{
	MlNode *n = (MlNode *)argument;
	const uint64_t row_bytes = (uint64_t)K3_HIDDEN * 2u;
	n->status = LM_LAUNCH_OK;
	for (uint32_t pass = 0u; pass < ML_PASSES && n->status == LM_LAUNCH_OK; pass++)
	{
		K3LayerBuffers *b = &n->buffers[pass];
		const uint64_t first = (uint64_t)pass * ML_PASS_ROWS + (uint64_t)n->node * ML_NODE_ROWS;
		ML_CUDA(cudaMemcpyAsync(b->hidden_bf16,n->hidden + first * K3_HIDDEN,ML_NODE_ROWS * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		ML_CUDA(cudaMemcpyAsync(b->attnres_partial_bf16,n->residual + first * K3_HIDDEN,ML_NODE_ROWS * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		n->status = K3StreamedMlaNode<LmMxfp4,K3GlobalKv,ML_HEADS>(b,n->shards,ML_SHARDS,&n->exchange[pass],n->partial,n->accumulator,0u,ML_NODE_ROWS,48u,n->stream);
		ML_CUDA(cudaMemcpyAsync(n->gathered + first * K3_HIDDEN,b->attnres_partial_bf16,ML_NODE_ROWS * row_bytes,cudaMemcpyDeviceToDevice,n->stream));
		ML_CUDA(cudaStreamSynchronize(n->stream));
	}
	return 0;
}

int main(void)
{
	std::vector<K3LayerWeights> shards(ML_SHARDS);
	std::vector<uint32_t> table(ML_PAGES),positions(ML_ROWS);
	static MlNode nodes[ML_NODES];
	uint16_t *hidden = MlBf16((uint64_t)ML_ROWS * K3_HIDDEN,1u,1.0f,0.0f),*partial_in = MlBf16((uint64_t)ML_ROWS * K3_HIDDEN,2u,1.0f,0.0f);
	uint16_t *attn_norm = MlBf16(K3_HIDDEN,3u,0.1f,1.0f),*q_norm = MlBf16(K3_Q_LORA_RANK,4u,0.1f,1.0f),*kv_norm = MlBf16(K3_KV_LORA_RANK,5u,0.1f,1.0f);
	uint16_t *q_down = MlBf16((uint64_t)K3_Q_LORA_RANK * K3_HIDDEN,6u,0.02f,0.0f),*kv_a = MlBf16((uint64_t)K3_MLA_KV_A_DIM * K3_HIDDEN,7u,0.02f,0.0f);
	uint16_t *q_b = MlBf16((uint64_t)K3_MLA_HEADS * K3_MHA_QK * K3_Q_LORA_RANK,8u,0.03f,0.0f);
	uint16_t *kv_b_key = MlBf16((uint64_t)K3_MLA_HEADS * K3_QK_NOPE_DIM * K3_KV_LORA_RANK,9u,0.05f,0.0f);
	uint16_t *kv_b_value = MlBf16((uint64_t)K3_MLA_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK,10u,0.05f,0.0f);
	uint16_t *scratch = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN),*gathered = MlAlloc<uint16_t>((uint64_t)ML_ROWS * K3_HIDDEN);
	uint16_t *q_latents = MlAlloc<uint16_t>((uint64_t)ML_PASS_ROWS * K3_Q_LORA_RANK);
	uint16_t *values = MlAlloc<uint16_t>((uint64_t)ML_SHARDS * ML_PASS_ROWS * ML_HEADS * K3_V_HEAD_DIM);
	uint8_t *latent_keys = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3GlobalKv::kSlotBytes);
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
	{
		MlNode *n = &nodes[node];
		uint16_t *query = MlAlloc<uint16_t>((uint64_t)ML_PASS_ROWS * ML_HEADS * K3_MHA_QK);
		uint8_t *expanded = MlAlloc<uint8_t>((uint64_t)ML_ROWS * K3StreamedHeadKv<ML_HEADS,K3GlobalKv>::kSlotBytes);
		float *partials = MlAlloc<float>((uint64_t)ML_PASS_ROWS * LmGqaShardRecordFloats(ML_HEADS,K3_V_HEAD_DIM,1u));
		const uint32_t *pass_row_offset = MlDevice(std::vector<uint32_t>{0u,ML_PASS_ROWS});
		uint32_t *pass_tile_prefix = MlAlloc<uint32_t>(2u);
		const uint32_t *pass_sequence = MlDevice(std::vector<uint32_t>(ML_PASS_ROWS,0u)),*node_sequence = MlDevice(std::vector<uint32_t>(ML_NODE_ROWS,0u));
		const uint32_t *key_offset = MlDevice(std::vector<uint32_t>{0u});
		LmKvShardView cache = MlCache(table_device,error_device);
		n->node = node;
		n->shards = shards.data();
		n->partial = MlAlloc<uint16_t>((uint64_t)ML_NODE_ROWS * K3_HIDDEN);
		n->accumulator = MlAlloc<float>((uint64_t)ML_NODE_ROWS * K3_HIDDEN);
		n->hidden = hidden;
		n->residual = partial_in;
		n->gathered = gathered;
		for (uint32_t pass = 0u; pass < ML_PASSES; pass++)
		{
			K3StreamedMlaExchange *x = &n->exchange[pass];
			std::vector<uint32_t> pass_positions(ML_PASS_ROWS),node_positions(ML_NODE_ROWS);
			for (uint32_t row = 0u; row < ML_PASS_ROWS; row++)
				pass_positions[row] = pass * ML_PASS_ROWS + row;
			for (uint32_t row = 0u; row < ML_NODE_ROWS; row++)
				node_positions[row] = pass * ML_PASS_ROWS + node * ML_NODE_ROWS + row;
			MlBuffers(&n->buffers[pass],ML_NODE_ROWS,node_sequence,MlDevice(std::vector<uint32_t>{(pass + 1u) * ML_PASS_ROWS}),MlDevice(node_positions));
			n->buffers[pass].cache_shard = cache;
			n->buffers[pass].kv_shard = cache.shard;
			memset(x,0,sizeof(*x));
			x->node = node;
			x->nodes = ML_NODES;
			x->pass_rows = ML_PASS_ROWS;
			x->first_row = node * ML_NODE_ROWS;
			x->first_key = pass * ML_PASS_ROWS;
			x->key_capacity = ML_ROWS;
			x->q_latents = q_latents;
			x->latent_keys = latent_keys;
			x->query = query;
			x->expanded = expanded;
			x->partials = partials;
			x->values = values;
			x->pass_positions = MlDevice(pass_positions);
			x->pass_sequence_of_row = pass_sequence;
			x->pass_row_offset = pass_row_offset;
			x->pass_tile_prefix = pass_tile_prefix;
			x->key_offset = key_offset;
			x->key_context = MlDevice(std::vector<uint32_t>{(pass + 1u) * ML_PASS_ROWS});
			x->exchange = MlExchange;
		}
		ML_CUDA(cudaStreamCreate(&n->stream));
	}
	ML_CUDA(cudaDeviceSynchronize());
	pthread_barrier_init(&ml_barrier,0,ML_NODES);
	{
		pthread_t threads[ML_NODES];
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_create(&threads[node],0,MlNodeThread,&nodes[node]);
		for (uint32_t node = 0u; node < ML_NODES; node++)
			pthread_join(threads[node],0);
	}
	for (uint32_t node = 0u; node < ML_NODES; node++)
		MlCheck(nodes[node].status == LM_LAUNCH_OK,"every node runs both passes");
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
		MlCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access errors");
		MlCheck(nonfinite == 0u,"every output element is finite");
		MlCheck(squared > 0.0,"the MLA layer changes the residual");
		MlCheck(rms <= 0.01,"four nodes over two passes, gathering latents and returning head-group values, match the one-device streamed MLA within 1% relative RMS of the attention contribution");
		printf("%s MLA on %u nodes x %u passes vs one device: worst |diff| %.4g, relative RMS %.3g, nonfinite %llu\n",
			ml_failures == 0 ? "PASS" : "FAIL",ML_NODES,ML_PASSES,worst,rms,(unsigned long long)nonfinite);
	}
	return ml_failures != 0;
}
