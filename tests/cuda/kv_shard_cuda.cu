#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"

#define PROBE_PAGE 64u
#define PROBE_LATENT 512u
#define PROBE_DENSE_LIMIT 2048u
#define PROBE_SELECTED 2051u
#define PROBE_REPEATS 20u

template<uint32_t WIDTH>
struct ProbeKv
{
	static constexpr uint32_t kSlotBytes = WIDTH * 2u;
	static constexpr uint32_t kPageSlots = PROBE_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * PROBE_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / PROBE_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % PROBE_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + PROBE_PAGE - 1u) / PROBE_PAGE; }
};

static int probe_failures;

static __host__ __device__ uint16_t ProbeValue(uint64_t key)
{
	uint64_t z = key * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	z ^= z >> 31u;
	float value = (float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 2048.0f;
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static double ProbeFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,4u);
	return result;
}

__global__ void ProbeFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	uint64_t index;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
		values[index] = ProbeValue(index ^ salt);
}

static void ProbeCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	probe_failures++;
}

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<class T>
static T *ProbeDevice(const std::vector<T> &host)
{
	T *device;
	PROBE_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	PROBE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

template<uint32_t ROPE>
static void ProbeCase(uint32_t rows,uint32_t context,uint32_t degree,uint32_t heads_per_rank,uint32_t grain,uint32_t compare)
{
	typedef ProbeKv<PROBE_LATENT + ROPE> Geometry;
	const uint32_t width = PROBE_LATENT + ROPE,record = PROBE_LATENT + 2u,heads = heads_per_rank * degree;
	const uint32_t pages_per_sequence = (context + PROBE_PAGE - 1u) / PROBE_PAGE,pages = pages_per_sequence * rows;
	const uint64_t tokens = (uint64_t)rows * context;
	const uint64_t rank_stride = (uint64_t)rows * heads_per_rank * record,query_stride = (uint64_t)rows * heads_per_rank * width;
	const uint64_t out_stride = (uint64_t)rows * heads_per_rank * PROBE_LATENT;
	const float scale = 1.0f / sqrtf((float)width);
	std::vector<uint32_t> table(pages),order(pages),contexts(rows),row_position(rows),sequences(rows),token_sequence(tokens),token_position(tokens);
	std::vector<uint32_t> selected((uint64_t)rows * PROBE_SELECTED,0xffffffffu);
	std::vector<float> oracle_send((uint64_t)degree * degree * rank_stride),shard_send(oracle_send.size());
	std::vector<uint16_t> oracle_out((uint64_t)degree * out_stride),shard_out(oracle_out.size()),gather_out(oracle_out.size()),single_out(oracle_out.size());
	std::vector<uint8_t *> shard_pool(degree);
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device,*selected_device;
	uint16_t *values_device,*query_device,*out_device;
	uint8_t *replicated_device;
	float *send_device,*receive_device,*split_device;
	LmKvAccessError *error_device,error_host;
	LmKvView view;
	uint32_t row,page,rank,index,listed = 0u,state = 7u + rows * 131u + context;
	uint64_t shard_bytes = 0u;
	double worst = 0.0,single_worst = 0.0;
	float shard_ms = 0.0f,single_ms = 0.0f;
	cudaEvent_t begin,end;
	char label[160];

	snprintf(label,sizeof(label),"B%u ctx%u degree%u heads%u grain%u rope%u",rows,context,degree,heads,grain,ROPE);
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
		if ( contexts[row] > PROBE_DENSE_LIMIT )
		{
			uint32_t pools = contexts[row] / 4u,chosen = 0u,start;
			state = state * 1664525u + 1013904223u;
			start = (state >> 8u) % pools;
			listed++;
			for (index=0u; index<pools && chosen<PROBE_SELECTED / 4u; index++)
			{
				uint32_t pool = (start + index * 7919u) % pools,token;
				for (token=0u; token<4u; token++)
					selected[(uint64_t)row * PROBE_SELECTED + chosen * 4u + token] = pool * 4u + token;
				chosen++;
			}
			for (index=0u; index<contexts[row] % 4u; index++)
				selected[(uint64_t)row * PROBE_SELECTED + chosen * 4u + index] = pools * 4u + index;
		}
	}
	table_device = ProbeDevice(table);
	contexts_device = ProbeDevice(contexts);
	positions_device = ProbeDevice(row_position);
	sequences_device = ProbeDevice(sequences);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	selected_device = ProbeDevice(selected);
	PROBE_CUDA(cudaMalloc((void **)&values_device,tokens * width * 2u));
	PROBE_CUDA(cudaMalloc((void **)&query_device,(uint64_t)degree * query_stride * 2u));
	PROBE_CUDA(cudaMalloc((void **)&replicated_device,(uint64_t)pages * Geometry::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&send_device,oracle_send.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&receive_device,oracle_send.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&split_device,(uint64_t)rows * heads * 64u * record * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out_device,oracle_out.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,tokens * width,0u);
	ProbeFillKernel<<<256,256>>>(query_device,(uint64_t)degree * query_stride,1ull << 60u);
	PROBE_CUDA(cudaMemset(replicated_device,0,(uint64_t)pages * Geometry::kPageBytes));
	ProbeCheck(LmKvViewInitialize(&view,replicated_device,table_device,pages_per_sequence,rows,pages,error_device) == 0,"replicated view",label);
	LmKvStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,width);
	PROBE_CUDA(cudaGetLastError());
	for (rank=0u; rank<degree; rank++)
	{
		SparkKvShard shard = {degree,rank,grain};
		LmKvShardView local;
		LmKvShardReplicaView replica;
		uint64_t bytes = SparkKvShardPoolBytes(shard,PROBE_PAGE,Geometry::kSlotBytes,pages);
		shard_bytes += bytes;
		ProbeCheck(bytes * degree == (uint64_t)pages * Geometry::kPageBytes,"per-rank KV bytes are total / degree",label);
		PROBE_CUDA(cudaMalloc((void **)&shard_pool[rank],bytes));
		PROBE_CUDA(cudaMemset(shard_pool[rank],0,bytes));
		ProbeCheck(LmKvShardViewInitialize<Geometry>(&local,shard_pool[rank],table_device,pages_per_sequence,rows,pages,error_device,shard) == 0,"shard view",label);
		ProbeCheck(LmKvShardReplicaViewInitialize<Geometry>(&replica,view,shard) == 0,"replica view",label);
		LmKvShardStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(local,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,width);
		PROBE_CUDA(cudaGetLastError());
		ProbeCheck(LmLatentShardPartialLaunch<Geometry,LmKvShardReplicaView,PROBE_LATENT,ROPE>(replica,query_device,query_stride,heads_per_rank,sequences_device,contexts_device,positions_device,selected_device,PROBE_SELECTED,PROBE_DENSE_LIMIT,scale,receive_device + (uint64_t)rank * degree * rank_stride,rank_stride,rows,0) == cudaSuccess,"oracle launch",label);
		ProbeCheck(LmLatentShardPartialLaunch<Geometry,LmKvShardView,PROBE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,sequences_device,contexts_device,positions_device,selected_device,PROBE_SELECTED,PROBE_DENSE_LIMIT,scale,send_device + (uint64_t)rank * degree * rank_stride,rank_stride,rows,0) == cudaSuccess,"shard launch",label);
	}
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(oracle_send.data(),receive_device,oracle_send.size() * sizeof(float),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(shard_send.data(),send_device,shard_send.size() * sizeof(float),cudaMemcpyDeviceToHost));
	ProbeCheck(memcmp(oracle_send.data(),shard_send.data(),oracle_send.size() * sizeof(float)) == 0,"sharded partials equal the replicated-storage oracle bit for bit",label);
	for (rank=0u; rank<degree; rank++)
		for (index=0u; index<degree; index++)
			PROBE_CUDA(cudaMemcpy(receive_device + ((uint64_t)rank * degree + index) * rank_stride,send_device + ((uint64_t)index * degree + rank) * rank_stride,rank_stride * sizeof(float),cudaMemcpyDeviceToDevice));
	for (rank=0u; rank<degree; rank++)
		ProbeCheck(LmLatentShardMergeLaunch<PROBE_LATENT>(receive_device + (uint64_t)rank * degree * rank_stride,rank_stride,degree,heads_per_rank,out_device + (uint64_t)rank * out_stride,rows,0) == cudaSuccess,"merge",label);
	PROBE_CUDA(cudaMemcpy(shard_out.data(),out_device,shard_out.size() * 2u,cudaMemcpyDeviceToHost));
	for (rank=0u; rank<degree; rank++)
		ProbeCheck(LmLatentShardMergeLaunch<PROBE_LATENT>(send_device + (uint64_t)rank * rank_stride,(uint64_t)degree * rank_stride,degree,heads_per_rank,out_device + (uint64_t)rank * out_stride,rows,0) == cudaSuccess,"gather merge",label);
	PROBE_CUDA(cudaMemcpy(gather_out.data(),out_device,gather_out.size() * 2u,cudaMemcpyDeviceToHost));
	ProbeCheck(memcmp(gather_out.data(),shard_out.data(),shard_out.size() * 2u) == 0,"all-gather and all-to-all layouts merge to the same bits",label);
	PROBE_CUDA(cudaMemcpy(receive_device,oracle_send.data(),oracle_send.size() * sizeof(float),cudaMemcpyHostToDevice));
	for (rank=0u; rank<degree; rank++)
		ProbeCheck(LmLatentShardMergeLaunch<PROBE_LATENT>(receive_device + (uint64_t)rank * rank_stride,(uint64_t)degree * rank_stride,degree,heads_per_rank,out_device + (uint64_t)rank * out_stride,rows,0) == cudaSuccess,"oracle merge",label);
	PROBE_CUDA(cudaMemcpy(oracle_out.data(),out_device,oracle_out.size() * 2u,cudaMemcpyDeviceToHost));
	ProbeCheck(memcmp(oracle_out.data(),shard_out.data(),shard_out.size() * 2u) == 0,"merged output equals the oracle bit for bit",label);
	if ( rows > 1u )
	{
		const uint64_t alone_stride = (uint64_t)heads_per_rank * record;
		const uint32_t picks[3] = {0u,rows / 2u,rows - 1u};
		std::vector<uint16_t> alone_out((uint64_t)degree * heads_per_rank * PROBE_LATENT);
		float *alone_send;
		uint16_t *alone_out_device;
		uint32_t pick;
		PROBE_CUDA(cudaMalloc((void **)&alone_send,(uint64_t)degree * degree * alone_stride * sizeof(float)));
		PROBE_CUDA(cudaMalloc((void **)&alone_out_device,alone_out.size() * 2u));
		for (pick=0u; pick<3u; pick++)
		{
			row = picks[pick];
			for (rank=0u; rank<degree; rank++)
			{
				SparkKvShard shard = {degree,rank,grain};
				LmKvShardView local;
				ProbeCheck(LmKvShardViewInitialize<Geometry>(&local,shard_pool[rank],table_device,pages_per_sequence,rows,pages,error_device,shard) == 0,"row-alone shard view",label);
				ProbeCheck(LmLatentShardPartialLaunch<Geometry,LmKvShardView,PROBE_LATENT,ROPE>(local,query_device + (uint64_t)row * heads_per_rank * width,query_stride,heads_per_rank,sequences_device + row,contexts_device,positions_device + row,selected_device + (uint64_t)row * PROBE_SELECTED,PROBE_SELECTED,PROBE_DENSE_LIMIT,scale,alone_send + (uint64_t)rank * degree * alone_stride,alone_stride,1u,0) == cudaSuccess,"row-alone partial launch",label);
			}
			for (rank=0u; rank<degree; rank++)
				ProbeCheck(LmLatentShardMergeLaunch<PROBE_LATENT>(alone_send + (uint64_t)rank * alone_stride,(uint64_t)degree * alone_stride,degree,heads_per_rank,alone_out_device + (uint64_t)rank * heads_per_rank * PROBE_LATENT,1u,0) == cudaSuccess,"row-alone merge",label);
			PROBE_CUDA(cudaMemcpy(alone_out.data(),alone_out_device,alone_out.size() * 2u,cudaMemcpyDeviceToHost));
			for (rank=0u; rank<degree; rank++)
				ProbeCheck(memcmp(alone_out.data() + (uint64_t)rank * heads_per_rank * PROBE_LATENT,shard_out.data() + (uint64_t)rank * out_stride + (uint64_t)row * heads_per_rank * PROBE_LATENT,(uint64_t)heads_per_rank * PROBE_LATENT * 2u) == 0,"a row decoded alone has the bits it has inside the batch",label);
		}
		cudaFree(alone_send);
		cudaFree(alone_out_device);
	}
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	ProbeCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	for (row=0u; row<rows; row+=(rows > 8u ? rows / 8u : 1u))
	{
		uint32_t head;
		std::vector<uint32_t> keys;
		if ( contexts[row] > PROBE_DENSE_LIMIT )
		{
			for (index=0u; index<PROBE_SELECTED; index++)
				if ( selected[(uint64_t)row * PROBE_SELECTED + index] <= row_position[row] )
					keys.push_back(selected[(uint64_t)row * PROBE_SELECTED + index]);
		}
		else
			for (index=0u; index<contexts[row]; index++)
				keys.push_back(index);
		for (head=0u; head<heads; head+=heads / 4u)
		{
			uint64_t q_base = (uint64_t)(head / heads_per_rank) * query_stride + ((uint64_t)row * heads_per_rank + head % heads_per_rank) * width;
			const uint16_t *out = shard_out.data() + (uint64_t)(head / heads_per_rank) * out_stride + ((uint64_t)row * heads_per_rank + head % heads_per_rank) * PROBE_LATENT;
			std::vector<double> scores(keys.size()),result(PROBE_LATENT,0.0);
			double top = -1.0e300,total = 0.0;
			uint32_t key,element;
			for (key=0u; key<keys.size(); key++)
			{
				uint64_t k_base = ((uint64_t)row * context + keys[key]) * width;
				double dot = 0.0;
				for (element=0u; element<width; element++)
					dot += ProbeFloat(ProbeValue((q_base + element) ^ (1ull << 60u))) * ProbeFloat(ProbeValue(k_base + element));
				scores[key] = dot * scale;
				top = scores[key] > top ? scores[key] : top;
			}
			for (key=0u; key<keys.size(); key++)
			{
				uint64_t k_base = ((uint64_t)row * context + keys[key]) * width;
				double weight = exp(scores[key] - top);
				total += weight;
				for (element=0u; element<PROBE_LATENT; element++)
					result[element] += weight * ProbeFloat(ProbeValue(k_base + element));
			}
			for (element=0u; element<PROBE_LATENT; element++)
			{
				double difference = fabs(result[element] / total - ProbeFloat(out[element]));
				worst = difference > worst ? difference : worst;
			}
		}
	}
	ProbeCheck(worst < 2.0e-3,"merged output matches the f64 attention reference",label);
	PROBE_CUDA(cudaEventCreate(&begin));
	PROBE_CUDA(cudaEventCreate(&end));
	if ( compare != 0u && ROPE == 0u )
	{
		cudaDeviceProp properties;
		const uint32_t *list = listed == rows ? selected_device : 0;
		PROBE_CUDA(cudaGetDeviceProperties(&properties,0));
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
			for (rank=0u; rank<degree; rank++)
				PROBE_CUDA((LmLatentAttentionHeadsLaunch<Geometry,PROBE_LATENT>(query_device + (uint64_t)rank * query_stride,view,sequences_device,contexts_device,list,list != 0 ? PROBE_SELECTED : 0u,heads_per_rank,scale,out_device + (uint64_t)rank * out_stride,positions_device,rows,context,64u,split_device,rows * heads * 64u,(uint32_t)properties.multiProcessorCount,0)));
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&single_ms,begin,end));
		single_ms /= (float)(PROBE_REPEATS * degree);
		PROBE_CUDA(cudaMemcpy(single_out.data(),out_device,single_out.size() * 2u,cudaMemcpyDeviceToHost));
		if ( list != 0 || listed == 0u )
			for (index=0u; index<single_out.size(); index++)
			{
				double difference = fabs(ProbeFloat(single_out[index]) - ProbeFloat(shard_out[index]));
				single_worst = difference > single_worst ? difference : single_worst;
			}
	}
	{
		SparkKvShard shard = {degree,0u,grain};
		LmKvShardView local;
		(void)LmKvShardViewInitialize<Geometry>(&local,shard_pool[0],table_device,pages_per_sequence,rows,pages,error_device,shard);
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
		{
			PROBE_CUDA((LmLatentShardPartialLaunch<Geometry,LmKvShardView,PROBE_LATENT,ROPE>(local,query_device,query_stride,heads_per_rank,sequences_device,contexts_device,positions_device,selected_device,PROBE_SELECTED,PROBE_DENSE_LIMIT,scale,send_device,rank_stride,rows,0)));
			PROBE_CUDA(LmLatentShardMergeLaunch<PROBE_LATENT>(receive_device,rank_stride,degree,heads_per_rank,out_device,rows,0));
		}
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&shard_ms,begin,end));
		shard_ms /= (float)PROBE_REPEATS;
	}
	printf("%s %s: listed rows %u/%u, KV per rank %.1f MiB of %.1f MiB, worst |out-f64| %.2e, |sharded-replicated kernel| %.2e, rank us: sharded partial+merge %.1f, replicated heads kernel %.1f\n",
		probe_failures == 0 ? "PASS" : "FAIL",label,listed,rows,(double)shard_bytes / degree / 1048576.0,(double)pages * Geometry::kPageBytes / 1048576.0,worst,single_worst,shard_ms * 1000.0f,single_ms * 1000.0f);
	for (rank=0u; rank<degree; rank++)
		cudaFree(shard_pool[rank]);
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(selected_device); cudaFree(values_device);
	cudaFree(query_device); cudaFree(replicated_device); cudaFree(send_device); cudaFree(receive_device); cudaFree(split_device);
	cudaFree(out_device); cudaFree(error_device);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
}

int main(void)
{
	uint32_t contexts[2] = {1024u,8192u},rows[3] = {1u,8u,64u},c,r;
	for (c=0u; c<2u; c++)
		for (r=0u; r<3u; r++)
			ProbeCase<0u>(rows[r],contexts[c],16u,4u,1u,1u);
	ProbeCase<64u>(8u,1024u,16u,4u,1u,0u);
	ProbeCase<64u>(8u,8192u,16u,4u,1u,0u);
	ProbeCase<0u>(8u,8192u,4u,16u,1u,0u);
	ProbeCase<0u>(8u,8192u,8u,8u,1u,0u);
	ProbeCase<0u>(8u,1024u,16u,4u,4u,0u);
	ProbeCase<0u>(8u,8192u,16u,4u,4u,0u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS latent KV shard on the device: B1/B8/B64 at 1k and 8k, grains 1 and 4, sharded store and attention equal the replicated oracle bit for bit, a row alone equals its bits inside the batch\n");
	return 0;
}
