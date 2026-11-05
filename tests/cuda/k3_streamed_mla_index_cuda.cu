#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "inference/llms/kimi_k3/streamed.cuh"
#include "tests/cuda/node_collective_emulation.cuh"

#define MI_SHARDS 16u
#define MI_ROWS 4096u
#define MI_NODES 4u
#define MI_HEADS (K3_MLA_HEADS / MI_SHARDS)
#define MI_PAGES (MI_ROWS / K3_KV_PAGE_SLOTS)
#define MI_CHUNK 128u
#define MI_SPARSE_ROWS 1024u

#define MI_CUDA(call) do { cudaError_t mi_error = (call); if ( mi_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(mi_error)); exit(1); } } while (0)

static int mi_failures;

static void MiCheck(int condition,const char *what)
{
	if ( !condition )
	{
		printf("FAIL %s\n",what);
		mi_failures++;
	}
}

__global__ void MiFillBf16(uint16_t *data,uint64_t count,uint32_t salt,float scale,float offset)
{
	for (uint64_t index = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(index * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		data[index] = LmFloatToBf16(offset + scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

__global__ void MiAbsorbKernel(const uint16_t *q_b,const uint16_t *kv_b_key,uint16_t *q_up)
{
	const uint32_t head = blockIdx.y, row = blockIdx.x;
	for (uint32_t column = threadIdx.x; column < K3_Q_LORA_RANK; column += blockDim.x)
	{
		float total = 0.0f;
		if ( row < K3_KV_LORA_RANK )
			for (uint32_t nope = 0u; nope < K3_QK_NOPE_DIM; nope++)
				total += LmBf16ToFloat(kv_b_key[((uint64_t)head * K3_QK_NOPE_DIM + nope) * K3_KV_LORA_RANK + row]) *
					LmBf16ToFloat(q_b[((uint64_t)head * K3_MHA_QK + nope) * K3_Q_LORA_RANK + column]);
		else
			total = LmBf16ToFloat(q_b[((uint64_t)head * K3_MHA_QK + K3_QK_NOPE_DIM + row - K3_KV_LORA_RANK) * K3_Q_LORA_RANK + column]);
		q_up[((uint64_t)head * K3_MLA_KV_A_DIM + row) * K3_Q_LORA_RANK + column] = LmFloatToBf16(total);
	}
}

static double MiHost(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

static double MiRound(double value)
{
	float single = (float)value;
	uint32_t bits;
	memcpy(&bits,&single,4u);
	bits = (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u;
	memcpy(&single,&bits,4u);
	return single;
}

template<class T>
static T *MiAlloc(uint64_t count)
{
	T *device;
	MI_CUDA(cudaMalloc((void **)&device,count * sizeof(T) + 256u));
	MI_CUDA(cudaMemset(device,0,count * sizeof(T) + 256u));
	return device;
}

static uint16_t *MiBf16(uint64_t count,uint32_t salt,float scale,float offset)
{
	uint16_t *data = MiAlloc<uint16_t>(count);
	MiFillBf16<<<512,256>>>(data,count,salt,scale,offset);
	return data;
}

static uint32_t *MiDevice(const std::vector<uint32_t> &host)
{
	uint32_t *device = MiAlloc<uint32_t>(host.size());
	MI_CUDA(cudaMemcpy(device,host.data(),host.size() * 4u,cudaMemcpyHostToDevice));
	return device;
}

template<class Geometry>
static LmKvShardView MiView(SparkKvShard shard,const uint32_t *table,LmKvAccessError *error)
{
	LmKvShardView view;
	uint8_t *pool = MiAlloc<uint8_t>(SparkKvShardPoolBytes(shard,Geometry::kPageSlots,Geometry::kSlotBytes,MI_PAGES));
	if ( LmKvShardViewInitialize<Geometry>(&view,pool,table,MI_PAGES,1u,MI_PAGES,error,shard) != 0 )
	{
		printf("FAIL kv view\n");
		exit(1);
	}
	return view;
}

struct MiNode
{
	uint32_t rows;
	const K3LayerWeights *shards;
	K3StreamedMlaExchange x;
	K3LayerBuffers b;
	K3StreamedJoin join;
	NodeEmulationNode peer;
	uint16_t *partial;
	int32_t status;
	cudaStream_t stream;
};

static void *MiThread(void *argument)
{
	MiNode *n = (MiNode *)argument;
	n->status = K3StreamedMlaNode<LmMxfp4,K3GlobalKv,MI_HEADS>(&n->b,n->shards,MI_SHARDS,&n->x,n->partial,&n->join,0u,n->rows,48u,n->stream);
	MI_CUDA(cudaStreamSynchronize(n->stream));
	return 0;
}

static void MiSetup(MiNode *n,uint32_t node,uint32_t nodes,NodeEmulationWorld *world,const K3LayerWeights *shards,const uint32_t *table,LmKvAccessError *error,
	const uint16_t *hidden,const uint16_t *residual)
{
	const uint32_t rows = MI_ROWS / nodes;
	const SparkKvShard local = {1u,0u,1u},owned = {nodes,node,4u};
	std::vector<uint32_t> pass_positions(MI_ROWS),node_positions(rows);
	K3LayerBuffers *b = &n->b;
	K3StreamedMlaExchange *x = &n->x;
	for (uint32_t row = 0u; row < MI_ROWS; row++)
		pass_positions[row] = row;
	for (uint32_t row = 0u; row < rows; row++)
		node_positions[row] = node * rows + row;
	n->rows = rows;
	n->shards = shards;
	n->peer.world = world;
	n->peer.node = node;
	memset(b,0,sizeof(*b));
	b->hidden_bf16 = MiAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->normed_bf16 = MiAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->latent_bf16 = MiAlloc<uint16_t>((uint64_t)rows * K3_Q_LORA_RANK);
	b->kv_slot_bf16 = MiAlloc<uint16_t>((uint64_t)rows * K3_MLA_KV_A_DIM);
	b->query_bf16 = MiAlloc<uint16_t>((uint64_t)rows * MI_HEADS * K3_MHA_QK);
	b->attention_out_bf16 = MiAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->value_bf16 = MiAlloc<uint16_t>((uint64_t)rows * MI_HEADS * K3_V_HEAD_DIM);
	b->gate_bf16 = MiAlloc<uint16_t>((uint64_t)rows * MI_HEADS * K3_V_HEAD_DIM);
	b->attnres_partial_bf16 = MiAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	b->dense_row_offset = MiDevice(std::vector<uint32_t>{0u,rows});
	b->dense_tile_prefix = MiAlloc<uint32_t>(2u);
	b->sequence_of_row = MiDevice(std::vector<uint32_t>(rows,0u));
	b->context_length = MiDevice(std::vector<uint32_t>{MI_ROWS});
	b->positions = MiDevice(node_positions);
	b->tp_sharded = 1u;
	b->mla_heads_rank = MI_HEADS;
	b->mla_gate_rows = MI_HEADS * K3_V_HEAD_DIM;
	b->mla_out_input = MI_HEADS * K3_V_HEAD_DIM;
	b->cache_shard = MiView<K3GlobalKv>(local,table,error);
	b->kv_shard = local;
	MI_CUDA(cudaMemcpy(b->hidden_bf16,hidden + (uint64_t)node * rows * K3_HIDDEN,(uint64_t)rows * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	MI_CUDA(cudaMemcpy(b->attnres_partial_bf16,residual + (uint64_t)node * rows * K3_HIDDEN,(uint64_t)rows * K3_HIDDEN * 2u,cudaMemcpyDeviceToDevice));
	n->partial = MiAlloc<uint16_t>((uint64_t)rows * K3_HIDDEN);
	n->join.weight = MiAlloc<uint16_t>((uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM);
	n->join.weight_capacity = (uint64_t)K3_HIDDEN * K3_MLA_HEADS * K3_V_HEAD_DIM;
	n->join.activation = MiAlloc<uint16_t>((uint64_t)rows * K3_MLA_HEADS * K3_V_HEAD_DIM);
	n->join.activation_capacity = (uint64_t)rows * K3_MLA_HEADS * K3_V_HEAD_DIM;
	memset(x,0,sizeof(*x));
	x->node = node;
	x->nodes = nodes;
	x->pass_rows = MI_ROWS;
	x->pass_tokens = MI_ROWS;
	x->key_capacity = MI_ROWS;
	x->q_latents = MiAlloc<uint16_t>((uint64_t)MI_ROWS * K3_Q_LORA_RANK);
	x->latent_keys = MiAlloc<uint8_t>((uint64_t)MI_ROWS * K3GlobalKv::kSlotBytes);
	x->query = MiAlloc<uint16_t>((uint64_t)MI_ROWS * MI_HEADS * K3_MHA_QK);
	x->expanded = MiAlloc<uint8_t>((uint64_t)MI_ROWS * K3StreamedHeadKv<MI_HEADS,K3GlobalKv>::kSlotBytes);
	x->partials = MiAlloc<float>((uint64_t)MI_ROWS * LmGqaShardRecordFloats(MI_HEADS,K3_V_HEAD_DIM,1u));
	x->merged = MiAlloc<uint16_t>((uint64_t)MI_ROWS * MI_HEADS * K3_V_HEAD_DIM);
	x->value_send = MiAlloc<uint16_t>((uint64_t)MI_ROWS * (MI_SHARDS / nodes) * MI_HEADS * K3_V_HEAD_DIM);
	x->values = MiAlloc<uint16_t>((uint64_t)MI_SHARDS * rows * MI_HEADS * K3_V_HEAD_DIM);
	x->pass_positions = MiDevice(pass_positions);
	x->pass_sequence_of_row = MiDevice(std::vector<uint32_t>(MI_ROWS,0u));
	x->pass_row_offset = MiDevice(std::vector<uint32_t>{0u,MI_ROWS});
	x->pass_tile_prefix = MiAlloc<uint32_t>(2u);
	x->key_offset = MiDevice(std::vector<uint32_t>{0u});
	x->key_context = MiDevice(std::vector<uint32_t>{MI_ROWS});
	x->collective = NodeEmulationCollective;
	x->context = &n->peer;
	x->help_values = MiAlloc<uint16_t>((uint64_t)MI_ROWS * MI_HEADS * K3_V_HEAD_DIM);
	x->help_capacity = MI_ROWS;
	x->store = MiView<K3GlobalKv>(owned,table,error);
	x->store_sequence_of_row = x->pass_sequence_of_row;
	x->older_sequence = x->key_offset;
	x->older_offset = x->key_offset;
	x->older_context = MiDevice(std::vector<uint32_t>{0u});
	x->older_send = MiAlloc<uint8_t>((uint64_t)MI_ROWS * K3GlobalKv::kSlotBytes);
	x->older_receive = x->expanded;
	x->older_capacity = (uint64_t)MI_ROWS * K3StreamedHeadKv<MI_HEADS,K3GlobalKv>::kSlotBytes;
	x->index_store = MiView<K3IndexKv>(owned,table,error);
	x->index_context = x->key_context;
	x->index_local = MiAlloc<uint16_t>((uint64_t)rows * K3_INDEX_DIM);
	x->index_keys = MiAlloc<uint16_t>((uint64_t)MI_ROWS * K3_INDEX_DIM);
	x->index_heads_local = MiAlloc<uint16_t>((uint64_t)rows * K3_INDEX_HEADS);
	x->index_heads = MiAlloc<uint16_t>((uint64_t)MI_ROWS * K3_INDEX_HEADS);
	x->index_query = MiAlloc<uint16_t>((uint64_t)MI_CHUNK * K3_INDEX_QUERY);
	x->index_score_floats = (uint64_t)MI_CHUNK * SparkKvShardGatherKeys(owned,MI_ROWS);
	x->index_scores = MiAlloc<float>(x->index_score_floats);
	x->index_send = MiAlloc<uint2>((uint64_t)nodes * MI_CHUNK * K3_INDEX_TOPK);
	x->index_received = MiAlloc<uint2>((uint64_t)nodes * MI_CHUNK * K3_INDEX_TOPK);
	x->index_bitmap_words = LmIndexSelectBitmapWords(MI_ROWS);
	x->index_bitmaps = MiAlloc<uint32_t>((uint64_t)MI_CHUNK * 2u * x->index_bitmap_words);
	x->index_selected_local = MiAlloc<uint32_t>((uint64_t)rows * K3_INDEX_TOPK);
	x->index_selected = MiAlloc<uint32_t>((uint64_t)MI_ROWS * K3_INDEX_TOPK);
	x->index_error = (LmFrameError *)MiAlloc<uint8_t>(sizeof(LmFrameError));
	x->index_chunk = MI_CHUNK;
	x->key_pages = table;
	x->zero_rows = x->pass_sequence_of_row;
	x->key_error = error;
	x->sparse_rows = MI_SPARSE_ROWS;
	x->sparse_query = MiAlloc<uint16_t>((uint64_t)MI_SPARSE_ROWS * MI_HEADS * K3_MLA_KV_A_DIM);
	x->sparse_partials = MiAlloc<float>((uint64_t)MI_SPARSE_ROWS * MI_HEADS * LM_LATENT_SHARD_RECORD_FLOATS(K3_KV_LORA_RANK));
	x->sparse_latent = MiAlloc<uint16_t>((uint64_t)MI_SPARSE_ROWS * MI_HEADS * K3_KV_LORA_RANK);
	MI_CUDA(cudaStreamCreate(&n->stream));
}

static std::vector<uint16_t> MiRun(MiNode *nodes,uint32_t count,NodeEmulationWorld *world)
{
	std::vector<uint16_t> output((uint64_t)MI_ROWS * K3_HIDDEN);
	std::vector<pthread_t> threads(count);
	NodeEmulationStart(world,count);
	for (uint32_t node = 0u; node < count; node++)
		pthread_create(&threads[node],0,MiThread,&nodes[node]);
	for (uint32_t node = 0u; node < count; node++)
		pthread_join(threads[node],0);
	for (uint32_t node = 0u; node < count; node++)
	{
		MiCheck(nodes[node].status == LM_LAUNCH_OK,"every node runs the indexed MLA layer");
		MI_CUDA(cudaMemcpy(output.data() + (uint64_t)node * nodes[node].rows * K3_HIDDEN,nodes[node].b.attnres_partial_bf16,(uint64_t)nodes[node].rows * K3_HIDDEN * 2u,
			cudaMemcpyDeviceToHost));
	}
	return output;
}

static double MiRelative(const std::vector<uint16_t> &want,const std::vector<uint16_t> &got,const std::vector<uint16_t> &before,uint32_t first,uint32_t last,
	uint64_t *nonfinite)
{
	double squared = 0.0,error = 0.0;
	for (uint64_t index = (uint64_t)first * K3_HIDDEN; index < (uint64_t)last * K3_HIDDEN; index++)
	{
		const double r = MiHost(want[index]),c = MiHost(got[index]),b = MiHost(before[index]);
		if ( !isfinite(r) || !isfinite(c) )
		{
			(*nonfinite)++;
			continue;
		}
		squared += (r - b) * (r - b);
		error += (r - c) * (r - c);
	}
	return sqrt(error / (squared > 0.0 ? squared : 1.0));
}

int main(void)
{
	static MiNode one[1],four[MI_NODES],dense[1];
	static NodeEmulationWorld one_world,four_world,dense_world;
	std::vector<K3LayerWeights> shards(MI_SHARDS),plain(MI_SHARDS);
	std::vector<uint32_t> table(MI_PAGES);
	for (uint32_t page = 0u; page < MI_PAGES; page++)
		table[page] = page;
	const uint32_t *table_device = MiDevice(table);
	LmKvAccessError *error = MiAlloc<LmKvAccessError>(1u),error_host;
	LmKvAccessErrorReset(&error_host);
	MI_CUDA(cudaMemcpy(error,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	uint16_t *hidden = MiBf16((uint64_t)MI_ROWS * K3_HIDDEN,1u,1.0f,0.0f),*residual = MiBf16((uint64_t)MI_ROWS * K3_HIDDEN,2u,1.0f,0.0f);
	uint16_t *attn_norm = MiBf16(K3_HIDDEN,3u,0.1f,1.0f),*q_norm = MiBf16(K3_Q_LORA_RANK,4u,0.1f,1.0f),*kv_norm = MiBf16(K3_KV_LORA_RANK,5u,0.1f,1.0f);
	uint16_t *q_down = MiBf16((uint64_t)K3_Q_LORA_RANK * K3_HIDDEN,6u,0.02f,0.0f),*kv_a = MiBf16((uint64_t)K3_MLA_KV_A_DIM * K3_HIDDEN,7u,0.02f,0.0f);
	uint16_t *q_b = MiBf16((uint64_t)K3_MLA_HEADS * K3_MHA_QK * K3_Q_LORA_RANK,8u,0.03f,0.0f);
	uint16_t *kv_b_key = MiBf16((uint64_t)K3_MLA_HEADS * K3_QK_NOPE_DIM * K3_KV_LORA_RANK,9u,0.05f,0.0f);
	uint16_t *kv_b_value = MiBf16((uint64_t)K3_MLA_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK,10u,0.05f,0.0f);
	uint16_t *q_up = MiAlloc<uint16_t>((uint64_t)K3_MLA_HEADS * K3_MLA_KV_A_DIM * K3_Q_LORA_RANK);
	uint16_t *index_query = MiBf16((uint64_t)K3_INDEX_QUERY * K3_Q_LORA_RANK,11u,0.03f,0.0f),*index_key = MiBf16((uint64_t)K3_INDEX_DIM * K3_HIDDEN,12u,0.02f,0.0f);
	uint16_t *index_head = MiBf16((uint64_t)K3_INDEX_HEADS * K3_HIDDEN,13u,0.02f,0.0f);
	uint16_t *index_norm = MiBf16(K3_INDEX_DIM,14u,0.1f,1.0f),*index_bias = MiBf16(K3_INDEX_DIM,15u,0.1f,0.0f);
	MiAbsorbKernel<<<dim3(K3_MLA_KV_A_DIM,K3_MLA_HEADS),256>>>(q_b,kv_b_key,q_up);
	MI_CUDA(cudaDeviceSynchronize());
	for (uint32_t shard = 0u; shard < MI_SHARDS; shard++)
	{
		K3LayerWeights *w = &plain[shard];
		memset(w,0,sizeof(*w));
		w->attn_norm_weight = attn_norm;
		w->mla_q_down_weight = q_down;
		w->mla_q_norm_weight = q_norm;
		w->mla_kv_a_weight = kv_a;
		w->mla_kv_a_norm_weight = kv_norm;
		w->mla_q_b_weight = q_b + (uint64_t)shard * MI_HEADS * K3_MHA_QK * K3_Q_LORA_RANK;
		w->mla_kv_b_key_full_weight = kv_b_key;
		w->mla_kv_b_value_full_weight = kv_b_value;
		w->mla_q_up_weight = q_up + (uint64_t)shard * MI_HEADS * K3_MLA_KV_A_DIM * K3_Q_LORA_RANK;
		w->mla_kv_b_value_weight = kv_b_value + (uint64_t)shard * MI_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK;
		w->mla_gate_weight = MiBf16((uint64_t)MI_HEADS * K3_V_HEAD_DIM * K3_HIDDEN,20u + shard,0.02f,0.0f);
		w->mla_out_weight = MiBf16((uint64_t)K3_HIDDEN * MI_HEADS * K3_V_HEAD_DIM,40u + shard,0.05f,0.0f);
		shards[shard] = *w;
		shards[shard].mla_index_query_weight = index_query;
		shards[shard].mla_index_key_weight = index_key;
		shards[shard].mla_index_head_weight = index_head;
		shards[shard].mla_index_norm_weight = index_norm;
		shards[shard].mla_index_norm_bias = index_bias;
	}
	MiSetup(&one[0],0u,1u,&one_world,shards.data(),table_device,error,hidden,residual);
	for (uint32_t node = 0u; node < MI_NODES; node++)
		MiSetup(&four[node],node,MI_NODES,&four_world,shards.data(),table_device,error,hidden,residual);
	MiSetup(&dense[0],0u,1u,&dense_world,plain.data(),table_device,error,hidden,residual);
	std::vector<uint16_t> before((uint64_t)MI_ROWS * K3_HIDDEN);
	MI_CUDA(cudaMemcpy(before.data(),residual,before.size() * 2u,cudaMemcpyDeviceToHost));
	const std::vector<uint16_t> alone = MiRun(one,1u,&one_world),split = MiRun(four,MI_NODES,&four_world),full = MiRun(dense,1u,&dense_world);
	uint64_t nonfinite = 0u;
	const double nodes_rms = MiRelative(alone,split,before,0u,MI_ROWS,&nonfinite);
	const double early_rms = MiRelative(full,alone,before,0u,K3_INDEX_TOPK,&nonfinite),late_rms = MiRelative(full,alone,before,K3_INDEX_TOPK,MI_ROWS,&nonfinite);
	std::vector<uint32_t> selected((uint64_t)MI_ROWS * K3_INDEX_TOPK);
	std::vector<uint16_t> latents((uint64_t)MI_ROWS * K3_Q_LORA_RANK),keys((uint64_t)MI_ROWS * K3_INDEX_DIM),heads((uint64_t)MI_ROWS * K3_INDEX_HEADS);
	std::vector<uint16_t> wq((uint64_t)K3_INDEX_QUERY * K3_Q_LORA_RANK);
	MI_CUDA(cudaMemcpy(selected.data(),one[0].x.index_selected,selected.size() * 4u,cudaMemcpyDeviceToHost));
	MI_CUDA(cudaMemcpy(latents.data(),one[0].x.q_latents,latents.size() * 2u,cudaMemcpyDeviceToHost));
	MI_CUDA(cudaMemcpy(keys.data(),one[0].x.index_keys,keys.size() * 2u,cudaMemcpyDeviceToHost));
	MI_CUDA(cudaMemcpy(heads.data(),one[0].x.index_heads,heads.size() * 2u,cudaMemcpyDeviceToHost));
	MI_CUDA(cudaMemcpy(wq.data(),index_query,wq.size() * 2u,cudaMemcpyDeviceToHost));
	uint32_t worst_overlap = K3_INDEX_TOPK,invalid = 0u;
	for (uint32_t row : {2600u,3500u,MI_ROWS - 1u})
	{
		std::vector<double> query(K3_INDEX_QUERY),score(row + 1u);
		std::vector<uint32_t> order(row + 1u),device;
		for (uint32_t out = 0u; out < K3_INDEX_QUERY; out++)
		{
			double total = 0.0;
			for (uint32_t in = 0u; in < K3_Q_LORA_RANK; in++)
				total += MiHost(wq[(uint64_t)out * K3_Q_LORA_RANK + in]) * MiHost(latents[(uint64_t)row * K3_Q_LORA_RANK + in]);
			query[out] = MiRound(total);
		}
		for (uint32_t key = 0u; key <= row; key++)
		{
			double total = 0.0;
			for (uint32_t head = 0u; head < K3_INDEX_HEADS; head++)
			{
				double dot = 0.0;
				for (uint32_t d = 0u; d < K3_INDEX_DIM; d++)
					dot += query[head * K3_INDEX_DIM + d] * MiHost(keys[(uint64_t)key * K3_INDEX_DIM + d]);
				total += MiHost(heads[(uint64_t)row * K3_INDEX_HEADS + head]) * (dot > 0.0 ? dot : 0.0);
			}
			score[key] = total;
			order[key] = key;
		}
		std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) { return score[a] != score[b] ? score[a] > score[b] : a < b; });
		std::vector<uint8_t> want(row + 1u,0u);
		for (uint32_t index = 0u; index < K3_INDEX_TOPK; index++)
			want[order[index]] = 1u;
		uint32_t overlap = 0u;
		for (uint32_t index = 0u; index < K3_INDEX_TOPK; index++)
		{
			const uint32_t position = selected[(uint64_t)row * K3_INDEX_TOPK + index];
			if ( position > row )
			{
				invalid++;
				continue;
			}
			overlap += want[position];
			device.push_back(position);
		}
		std::sort(device.begin(),device.end());
		invalid += (uint32_t)(device.end() - std::unique(device.begin(),device.end())) + (K3_INDEX_TOPK - (uint32_t)device.size());
		worst_overlap = overlap < worst_overlap ? overlap : worst_overlap;
	}
	MI_CUDA(cudaMemcpy(&error_host,error,sizeof(error_host),cudaMemcpyDeviceToHost));
	MiCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access errors");
	MiCheck(nonfinite == 0u,"every output element is finite");
	MiCheck(nodes_rms <= 0.002,"four nodes, each scoring only the index keys it owns and merging the exchanged candidates, match one node within 0.2% relative RMS");
	MiCheck(invalid == 0u,"each row past 2,048 positions selects 2,048 distinct earlier positions");
	MiCheck(worst_overlap >= K3_INDEX_TOPK * 99u / 100u,"the selection is the top 2,048 of d^-0.5 H^-0.5 sum_h w_h relu(q_h . k) recomputed on the host (99% overlap; the rest are bf16 near-ties)");
	MiCheck(early_rms <= 0.015,"rows within 2,048 positions attend every earlier key, so the sparse latent path matches the dense expanded path within 1.5% (the absorbed query rounds once more than the expanded one)");
	MiCheck(late_rms >= 0.02,"rows past 2,048 positions attend only their selection, so they differ from dense attention");
	printf("%s streamed MLA indexer: 4 nodes vs 1 %.3g; rows < 2048 sparse vs dense %.3g, rows >= 2048 %.3g; selection overlap with the host top-2048 >= %u, invalid %u, nonfinite %llu\n",
		mi_failures == 0 ? "PASS" : "FAIL",nodes_rms,early_rms,late_rms,worst_overlap,invalid,(unsigned long long)nonfinite);
	return mi_failures != 0;
}
