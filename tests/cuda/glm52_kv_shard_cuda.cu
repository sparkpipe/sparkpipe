#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "sparkpipe/spark_glm52_kv_shard.h"

#define G52_DEGREE 16u
#define G52_HEADS_PER_RANK 4u
#define G52_HEADS (G52_DEGREE * G52_HEADS_PER_RANK)
#define G52_LATENT SPARK_GLM52_MODEL_LATENT_DIMENSION
#define G52_ROPE SPARK_GLM52_MODEL_ROPE_DIMENSION
#define G52_WIDTH SPARK_GLM52_KV_SHARD_QUERY_WIDTH
#define G52_RECORD SPARK_GLM52_KV_SHARD_RECORD_FLOATS
#define G52_PAGE SPARK_GLM52_KV_SHARD_PAGE_SLOTS
#define G52_LAYERS SPARK_GLM52_MODEL_LAYER_COUNT
#define G52_SELECTED SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT
#define G52_THREADS 256u
#define G52_SPLIT_THRESHOLD 64u
#define G52_REPEATS 10u

struct Glm52TestKv
{
	static constexpr uint32_t kSlotBytes = G52_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = G52_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * G52_PAGE * G52_LAYERS;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / G52_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % G52_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + G52_PAGE - 1u) / G52_PAGE; }
};

static_assert(Glm52TestKv::kPageBytes == SPARK_GLM52_KV_SHARD_LATENT_TOKEN_BYTES * G52_PAGE, "the test page is the glm52 block-major page");

static int g52_failures;

static __host__ __device__ uint16_t G52Value(uint64_t key)
{
	uint64_t z = key * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	z ^= z >> 31u;
	float value = (float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 1024.0f;
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static double G52Float(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

__global__ void G52FillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	uint64_t index;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
		values[index] = G52Value(index ^ salt);
}

static void G52Check(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	g52_failures++;
}

#define G52_CUDA(call) do { cudaError_t g52_error = (call); if ( g52_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(g52_error)); exit(1); } } while (0)

template<class T>
static T *G52Device(const std::vector<T> &host)
{
	T *device;
	G52_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	G52_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static uint32_t G52Ulps(uint16_t a,uint16_t b)
{
	int32_t x = (a & 0x8000u) != 0u ? -(int32_t)(a & 0x7fffu) : (int32_t)a;
	int32_t y = (b & 0x8000u) != 0u ? -(int32_t)(b & 0x7fffu) : (int32_t)b;
	return (uint32_t)(x > y ? x - y : y - x);
}

static void G52Case(uint32_t rows,uint32_t context,uint32_t layer)
{
	typedef Glm52TestKv Geometry;
	const uint32_t pages_per_sequence = (context + G52_PAGE - 1u) / G52_PAGE,pages = pages_per_sequence * rows;
	const uint64_t tokens = (uint64_t)rows * context;
	const uint64_t rank_stride = (uint64_t)rows * G52_HEADS_PER_RANK * G52_RECORD,query_stride = (uint64_t)rows * G52_HEADS_PER_RANK * G52_WIDTH;
	const uint64_t out_stride = (uint64_t)rows * G52_HEADS_PER_RANK * G52_LATENT;
	const uint64_t replicated_block = SparkGlm52KvShardLayerBlockBytes(1u),shard_block = SparkGlm52KvShardLayerBlockBytes(G52_DEGREE);
	const uint64_t shard_pool_bytes = (uint64_t)pages * SparkGlm52KvShardPageBytes(G52_DEGREE);
	const uint32_t listed_case = context > G52_SELECTED ? 1u : 0u;
	const float scale = SPARK_GLM52_MODEL_QK_SCALE;
	std::vector<uint32_t> table(pages),order(pages),contexts(rows),row_position(rows),sequences(rows),token_sequence(tokens),token_position(tokens);
	std::vector<uint32_t> selected(listed_case != 0u ? (uint64_t)rows * G52_SELECTED : 1u,0xffffffffu);
	std::vector<float> oracle_send((uint64_t)G52_DEGREE * G52_DEGREE * rank_stride),shard_send(oracle_send.size());
	std::vector<uint16_t> oracle_out((uint64_t)G52_DEGREE * out_stride),shard_out(oracle_out.size()),current_out(oracle_out.size()),query_host((uint64_t)G52_DEGREE * query_stride);
	std::vector<uint8_t *> shard_pool(G52_DEGREE);
	std::vector<uint8_t> probe_bytes;
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device,*selected_device;
	uint16_t *values_device,*query_device,*latent_device,*rope_device,*out_device;
	uint8_t *replicated_device;
	float *send_device,*receive_device,*split_device;
	LmKvAccessError *error_device,error_host;
	LmKvView view;
	cudaDeviceProp properties;
	cudaEvent_t begin,end;
	uint32_t row,page,rank,index,state = 11u + rows * 131u + context,worst_ulps = 0u,differing = 0u;
	double shard_worst = 0.0,current_worst = 0.0;
	float shard_ms = 0.0f,current_ms = 0.0f;
	char label[128];

	snprintf(label,sizeof(label),"glm52 B%u ctx%u layer%u %s",rows,context,layer,listed_case != 0u ? "dsa-listed" : "dense");
	G52_CUDA(cudaGetDeviceProperties(&properties,0));
	for (page=0u; page<pages; page++)
		order[page] = page;
	for (page=pages; page>1u; page--)
	{
		uint32_t pick,swap;
		state = state * 1664525u + 1013904223u;
		pick = (state >> 8u) % page;
		swap = order[page - 1u];
		order[page - 1u] = order[pick];
		order[pick] = swap;
	}
	for (row=0u; row<rows; row++)
	{
		sequences[row] = row;
		contexts[row] = context - (row * 37u) % (context / 4u);
		row_position[row] = contexts[row] - 1u;
		for (page=0u; page<pages_per_sequence; page++)
			table[row * pages_per_sequence + page] = order[row * pages_per_sequence + page];
		for (index=0u; index<context; index++)
		{
			token_sequence[(uint64_t)row * context + index] = row;
			token_position[(uint64_t)row * context + index] = index;
		}
		if ( listed_case != 0u )
		{
			std::vector<uint32_t> pool(context);
			for (index=0u; index<context; index++)
				pool[index] = index;
			for (index=0u; index<G52_SELECTED; index++)
			{
				uint32_t pick,swap;
				state = state * 1664525u + 1013904223u;
				pick = index + (state >> 8u) % (context - index);
				swap = pool[index];
				pool[index] = pool[pick];
				pool[pick] = swap;
				selected[(uint64_t)row * G52_SELECTED + index] = pool[index];
			}
		}
	}
	table_device = G52Device(table);
	contexts_device = G52Device(contexts);
	positions_device = G52Device(row_position);
	sequences_device = G52Device(sequences);
	token_sequence_device = G52Device(token_sequence);
	token_position_device = G52Device(token_position);
	selected_device = G52Device(selected);
	G52_CUDA(cudaMalloc((void **)&values_device,tokens * G52_WIDTH * 2u));
	G52_CUDA(cudaMalloc((void **)&query_device,(uint64_t)G52_DEGREE * query_stride * 2u));
	G52_CUDA(cudaMalloc((void **)&latent_device,(uint64_t)G52_DEGREE * out_stride * 2u));
	G52_CUDA(cudaMalloc((void **)&rope_device,(uint64_t)G52_DEGREE * rows * G52_HEADS_PER_RANK * G52_ROPE * 2u));
	G52_CUDA(cudaMalloc((void **)&replicated_device,(uint64_t)pages * Geometry::kPageBytes));
	G52_CUDA(cudaMalloc((void **)&send_device,oracle_send.size() * sizeof(float)));
	G52_CUDA(cudaMalloc((void **)&receive_device,oracle_send.size() * sizeof(float)));
	G52_CUDA(cudaMalloc((void **)&split_device,(uint64_t)rows * G52_HEADS_PER_RANK * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS * G52_RECORD * sizeof(float)));
	G52_CUDA(cudaMalloc((void **)&out_device,oracle_out.size() * 2u));
	G52_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	G52_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	G52FillKernel<<<1024,256>>>(values_device,tokens * G52_WIDTH,0u);
	G52FillKernel<<<256,256>>>(query_device,(uint64_t)G52_DEGREE * query_stride,1ull << 60u);
	G52_CUDA(cudaGetLastError());
	G52_CUDA(cudaMemcpy2D(latent_device,G52_LATENT * 2u,query_device,G52_WIDTH * 2u,G52_LATENT * 2u,(uint64_t)G52_DEGREE * rows * G52_HEADS_PER_RANK,cudaMemcpyDeviceToDevice));
	G52_CUDA(cudaMemcpy2D(rope_device,G52_ROPE * 2u,query_device + G52_LATENT,G52_WIDTH * 2u,G52_ROPE * 2u,(uint64_t)G52_DEGREE * rows * G52_HEADS_PER_RANK,cudaMemcpyDeviceToDevice));
	G52_CUDA(cudaMemcpy(query_host.data(),query_device,query_host.size() * 2u,cudaMemcpyDeviceToHost));
	G52_CUDA(cudaMemset(replicated_device,0,(uint64_t)pages * Geometry::kPageBytes));
	G52Check(LmKvViewInitialize(&view,replicated_device + (uint64_t)layer * replicated_block,table_device,pages_per_sequence,rows,pages,error_device) == 0,"replicated view",label);
	LmKvStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,G52_WIDTH);
	G52_CUDA(cudaGetLastError());
	for (rank=0u; rank<G52_DEGREE; rank++)
	{
		SparkKvShard shard = SparkGlm52KvShardLatent(rank,G52_DEGREE);
		LmKvShardView local;
		LmKvShardReplicaView replica;
		G52_CUDA(cudaMalloc((void **)&shard_pool[rank],shard_pool_bytes));
		G52_CUDA(cudaMemset(shard_pool[rank],0,shard_pool_bytes));
		G52Check(LmKvShardViewInitialize<Geometry>(&local,shard_pool[rank] + (uint64_t)layer * shard_block,table_device,pages_per_sequence,rows,pages,error_device,shard) == 0,"shard view",label);
		G52Check(LmKvShardReplicaViewInitialize<Geometry>(&replica,view,shard) == 0,"replica view",label);
		LmKvShardStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(local,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,G52_WIDTH);
		G52_CUDA(cudaGetLastError());
		G52Check(LmLatentShardPartialLaunch<Geometry,LmKvShardReplicaView,G52_LATENT,G52_ROPE>(replica,query_device,query_stride,G52_HEADS_PER_RANK,sequences_device,contexts_device,positions_device,listed_case != 0u ? selected_device : 0,listed_case != 0u ? G52_SELECTED : 0u,G52_SELECTED,scale,receive_device + (uint64_t)rank * G52_DEGREE * rank_stride,rank_stride,rows,0) == cudaSuccess,"oracle partial launch",label);
		G52Check(LmLatentShardPartialLaunch<Geometry,LmKvShardView,G52_LATENT,G52_ROPE>(local,query_device,query_stride,G52_HEADS_PER_RANK,sequences_device,contexts_device,positions_device,listed_case != 0u ? selected_device : 0,listed_case != 0u ? G52_SELECTED : 0u,G52_SELECTED,scale,send_device + (uint64_t)rank * G52_DEGREE * rank_stride,rank_stride,rows,0) == cudaSuccess,"shard partial launch",label);
	}
	G52_CUDA(cudaDeviceSynchronize());
	{
		uint64_t owned = 0u,matching = 0u,foreign_layers = 0u;
		std::vector<uint8_t> replicated_host((uint64_t)pages * Geometry::kPageBytes);
		G52_CUDA(cudaMemcpy(replicated_host.data(),replicated_device,replicated_host.size(),cudaMemcpyDeviceToHost));
		for (rank=0u; rank<G52_DEGREE; rank++)
		{
			SparkKvShard shard = SparkGlm52KvShardLatent(rank,G52_DEGREE);
			probe_bytes.assign(shard_pool_bytes,0u);
			G52_CUDA(cudaMemcpy(probe_bytes.data(),shard_pool[rank],shard_pool_bytes,cudaMemcpyDeviceToHost));
			for (page=0u; page<pages; page++)
			{
				uint32_t other;
				for (other=0u; other<G52_LAYERS; other++)
				{
					const uint8_t *block = probe_bytes.data() + (uint64_t)page * SparkGlm52KvShardPageBytes(G52_DEGREE) + (uint64_t)other * shard_block;
					if ( other == layer )
						continue;
					for (index=0u; index<shard_block; index++)
						foreign_layers += block[index] != 0u ? 1u : 0u;
				}
			}
			for (row=0u; row<rows; row++)
				for (index=0u; index<context; index++)
				{
					uint32_t physical;
					const uint8_t *mine,*reference;
					if ( SparkKvShardOwns(shard,index) == 0u )
						continue;
					physical = table[row * pages_per_sequence + index / G52_PAGE];
					mine = probe_bytes.data() + (uint64_t)physical * SparkGlm52KvShardPageBytes(G52_DEGREE) + (uint64_t)layer * shard_block + (uint64_t)SparkKvShardSlotInPage(shard,G52_PAGE,index) * Geometry::kSlotBytes;
					reference = replicated_host.data() + (uint64_t)physical * Geometry::kPageBytes + (uint64_t)layer * replicated_block + (uint64_t)(index % G52_PAGE) * Geometry::kSlotBytes;
					owned++;
					matching += memcmp(mine,reference,Geometry::kSlotBytes) == 0 ? 1u : 0u;
				}
		}
		G52Check(owned == tokens && matching == owned,"every token is stored once, by its owner, with the replicated slot bytes",label);
		G52Check(foreign_layers == 0u,"a layer store touches only its own block of the rank page",label);
		G52Check(shard_pool_bytes * G52_DEGREE == (uint64_t)pages * Geometry::kPageBytes,"per-rank KV bytes are the replicated bytes / 16",label);
	}
	G52_CUDA(cudaMemcpy(oracle_send.data(),receive_device,oracle_send.size() * sizeof(float),cudaMemcpyDeviceToHost));
	G52_CUDA(cudaMemcpy(shard_send.data(),send_device,shard_send.size() * sizeof(float),cudaMemcpyDeviceToHost));
	G52Check(memcmp(oracle_send.data(),shard_send.data(),oracle_send.size() * sizeof(float)) == 0,"sharded partials equal the replicated-storage partials bit for bit",label);
	for (rank=0u; rank<G52_DEGREE; rank++)
		for (index=0u; index<G52_DEGREE; index++)
			G52_CUDA(cudaMemcpy(receive_device + ((uint64_t)rank * G52_DEGREE + index) * rank_stride,send_device + ((uint64_t)index * G52_DEGREE + rank) * rank_stride,rank_stride * sizeof(float),cudaMemcpyDeviceToDevice));
	for (rank=0u; rank<G52_DEGREE; rank++)
		G52Check(LmLatentShardMergeLaunch<G52_LATENT>(receive_device + (uint64_t)rank * G52_DEGREE * rank_stride,rank_stride,G52_DEGREE,G52_HEADS_PER_RANK,out_device + (uint64_t)rank * out_stride,rows,0) == cudaSuccess,"merge",label);
	G52_CUDA(cudaMemcpy(shard_out.data(),out_device,shard_out.size() * 2u,cudaMemcpyDeviceToHost));
	G52_CUDA(cudaMemcpy(receive_device,oracle_send.data(),oracle_send.size() * sizeof(float),cudaMemcpyHostToDevice));
	for (rank=0u; rank<G52_DEGREE; rank++)
		G52Check(LmLatentShardMergeLaunch<G52_LATENT>(receive_device + (uint64_t)rank * rank_stride,(uint64_t)G52_DEGREE * rank_stride,G52_DEGREE,G52_HEADS_PER_RANK,out_device + (uint64_t)rank * out_stride,rows,0) == cudaSuccess,"oracle merge",label);
	G52_CUDA(cudaMemcpy(oracle_out.data(),out_device,oracle_out.size() * 2u,cudaMemcpyDeviceToHost));
	G52Check(memcmp(oracle_out.data(),shard_out.data(),shard_out.size() * 2u) == 0,"merged latent equals the replicated-storage merge bit for bit",label);
	for (rank=0u; rank<G52_DEGREE; rank++)
		G52Check((LmLatentAttentionDecodeSplitLaunch<Geometry,G52_THREADS,G52_LATENT,G52_ROPE,true>(latent_device + (uint64_t)rank * out_stride,rope_device + (uint64_t)rank * rows * G52_HEADS_PER_RANK * G52_ROPE,view,sequences_device,contexts_device,listed_case != 0u ? selected_device : 0,listed_case != 0u ? G52_SELECTED : 0u,G52_HEADS_PER_RANK,scale,out_device + (uint64_t)rank * out_stride,positions_device,rows,context > G52_SELECTED ? G52_SELECTED : context,G52_SPLIT_THRESHOLD,split_device,rows * G52_HEADS_PER_RANK * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS,(uint32_t)properties.multiProcessorCount,0)) == cudaSuccess,"current replicated kernel launch",label);
	G52_CUDA(cudaDeviceSynchronize());
	G52_CUDA(cudaMemcpy(current_out.data(),out_device,current_out.size() * 2u,cudaMemcpyDeviceToHost));
	G52_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	G52Check(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	for (index=0u; index<shard_out.size(); index++)
	{
		uint32_t ulps = G52Ulps(shard_out[index],current_out[index]);
		worst_ulps = ulps > worst_ulps ? ulps : worst_ulps;
		differing += ulps != 0u ? 1u : 0u;
	}
	for (row=0u; row<rows; row+=(rows > 4u ? rows / 4u : 1u))
	{
		uint32_t head;
		std::vector<uint32_t> keys;
		if ( listed_case != 0u )
		{
			for (index=0u; index<G52_SELECTED; index++)
				if ( selected[(uint64_t)row * G52_SELECTED + index] <= row_position[row] )
					keys.push_back(selected[(uint64_t)row * G52_SELECTED + index]);
		}
		else
			for (index=0u; index<contexts[row]; index++)
				keys.push_back(index);
		for (head=0u; head<G52_HEADS; head+=7u)
		{
			uint64_t q_base = (uint64_t)(head / G52_HEADS_PER_RANK) * query_stride + ((uint64_t)row * G52_HEADS_PER_RANK + head % G52_HEADS_PER_RANK) * G52_WIDTH;
			uint64_t o_base = (uint64_t)(head / G52_HEADS_PER_RANK) * out_stride + ((uint64_t)row * G52_HEADS_PER_RANK + head % G52_HEADS_PER_RANK) * G52_LATENT;
			std::vector<double> scores(keys.size()),result(G52_LATENT,0.0);
			double top = -1.0e300,total = 0.0;
			uint32_t key,element;
			for (key=0u; key<keys.size(); key++)
			{
				uint64_t k_base = ((uint64_t)row * context + keys[key]) * G52_WIDTH;
				double dot = 0.0;
				for (element=0u; element<G52_WIDTH; element++)
					dot += G52Float(query_host[q_base + element]) * G52Float(G52Value(k_base + element));
				scores[key] = dot * scale;
				top = scores[key] > top ? scores[key] : top;
			}
			for (key=0u; key<keys.size(); key++)
			{
				uint64_t k_base = ((uint64_t)row * context + keys[key]) * G52_WIDTH;
				double weight = exp(scores[key] - top);
				total += weight;
				for (element=0u; element<G52_LATENT; element++)
					result[element] += weight * G52Float(G52Value(k_base + element));
			}
			for (element=0u; element<G52_LATENT; element++)
			{
				double expected = result[element] / total;
				double a = fabs(expected - G52Float(shard_out[o_base + element])),b = fabs(expected - G52Float(current_out[o_base + element]));
				shard_worst = a > shard_worst ? a : shard_worst;
				current_worst = b > current_worst ? b : current_worst;
			}
		}
	}
	G52Check(shard_worst <= current_worst * 1.5 + 1.0e-3,"the sharded order is as close to the f64 reference as the served kernel",label);
	G52Check(shard_worst < 1.6e-2,"the sharded output matches the f64 attention reference to bf16 output rounding",label);
	G52_CUDA(cudaEventCreate(&begin));
	G52_CUDA(cudaEventCreate(&end));
	{
		SparkKvShard shard = SparkGlm52KvShardLatent(0u,G52_DEGREE);
		LmKvShardView local;
		(void)LmKvShardViewInitialize<Geometry>(&local,shard_pool[0] + (uint64_t)layer * shard_block,table_device,pages_per_sequence,rows,pages,error_device,shard);
		G52_CUDA(cudaEventRecord(begin));
		for (index=0u; index<G52_REPEATS; index++)
		{
			G52_CUDA((LmLatentShardPartialLaunch<Geometry,LmKvShardView,G52_LATENT,G52_ROPE>(local,query_device,query_stride,G52_HEADS_PER_RANK,sequences_device,contexts_device,positions_device,listed_case != 0u ? selected_device : 0,listed_case != 0u ? G52_SELECTED : 0u,G52_SELECTED,scale,send_device,rank_stride,rows,0)));
			G52_CUDA(LmLatentShardMergeLaunch<G52_LATENT>(receive_device,rank_stride,G52_DEGREE,G52_HEADS_PER_RANK,out_device,rows,0));
		}
		G52_CUDA(cudaEventRecord(end));
		G52_CUDA(cudaEventSynchronize(end));
		G52_CUDA(cudaEventElapsedTime(&shard_ms,begin,end));
		G52_CUDA(cudaEventRecord(begin));
		for (index=0u; index<G52_REPEATS; index++)
			G52_CUDA((LmLatentAttentionDecodeSplitLaunch<Geometry,G52_THREADS,G52_LATENT,G52_ROPE,true>(latent_device,rope_device,view,sequences_device,contexts_device,listed_case != 0u ? selected_device : 0,listed_case != 0u ? G52_SELECTED : 0u,G52_HEADS_PER_RANK,scale,out_device,positions_device,rows,context > G52_SELECTED ? G52_SELECTED : context,G52_SPLIT_THRESHOLD,split_device,rows * G52_HEADS_PER_RANK * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS,(uint32_t)properties.multiProcessorCount,0)));
		G52_CUDA(cudaEventRecord(end));
		G52_CUDA(cudaEventSynchronize(end));
		G52_CUDA(cudaEventElapsedTime(&current_ms,begin,end));
	}
	{
		uint64_t keys_read = 0u;
		for (row=0u; row<rows; row++)
			keys_read += listed_case != 0u ? G52_SELECTED : contexts[row];
		printf("%s %s: KV/rank %.1f of %.1f MiB | sharded == replicated-storage merge bitwise | vs served split kernel: %u of %zu outputs differ, worst %u bf16 ulp | |out-f64| sharded %.2e served %.2e | per rank per layer: sharded 64-head partial+merge %.1f us (%.1f MB KV) vs served 4-head %.1f us (%.1f MB KV)\n",
			g52_failures == 0 ? "PASS" : "FAIL",label,(double)shard_pool_bytes / 1048576.0,(double)pages * Geometry::kPageBytes / 1048576.0,differing,shard_out.size(),worst_ulps,shard_worst,current_worst,
			shard_ms * 1000.0f / G52_REPEATS,(double)keys_read * Geometry::kSlotBytes / G52_DEGREE / 1.0e6,current_ms * 1000.0f / G52_REPEATS,(double)keys_read * Geometry::kSlotBytes / 1.0e6);
	}
	for (rank=0u; rank<G52_DEGREE; rank++)
		cudaFree(shard_pool[rank]);
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(selected_device); cudaFree(values_device);
	cudaFree(query_device); cudaFree(latent_device); cudaFree(rope_device); cudaFree(replicated_device); cudaFree(send_device);
	cudaFree(receive_device); cudaFree(split_device); cudaFree(out_device); cudaFree(error_device);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
}

int main(void)
{
	G52Case(1u,1024u,0u);
	G52Case(1u,8192u,77u);
	G52Case(1u,32768u,41u);
	G52Case(8u,2048u,77u);
	G52Case(4u,4096u,5u);
	G52Case(64u,256u,77u);
	G52Case(16u,1024u,30u);
	if ( g52_failures != 0 )
	{
		printf("FAIL %d checks\n",g52_failures);
		return 1;
	}
	printf("PASS glm52 latent KV 1/16 per rank at TP16: block-major 78-layer pages shard per layer block, sharded store and 64-head partial merge equal the replicated-storage order bit for bit at B1/B4/B8/B16/B64, dense and DSA-listed up to 32k context\n");
	return 0;
}
