#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"

#define PROBE_PAGE 64u

template<uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
struct ProbeGqaKv
{
	static constexpr uint32_t kSlotBytes = KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u;
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

template<uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static void ProbeCase(uint32_t heads,uint32_t sequences,uint32_t rows_per_sequence,uint32_t context,uint32_t degree,uint32_t grain,uint32_t window = 0u,bool sinks = false)
{
	typedef ProbeGqaKv<KV_HEADS,HEAD_DIM,VALUE_DIM> Geometry;
	const uint32_t slot_elements = KV_HEADS * (HEAD_DIM + VALUE_DIM),rows = sequences * rows_per_sequence;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,degree),record = LmGqaShardRecordFloats(heads,VALUE_DIM,degree);
	const uint32_t pages_per_sequence = (context + PROBE_PAGE - 1u) / PROBE_PAGE,pages = pages_per_sequence * sequences;
	const uint64_t tokens = (uint64_t)sequences * context,stride = (uint64_t)rows * record;
	const uint64_t query_salt = 1ull << 60u;
	const float scale = 1.0f / sqrtf((float)HEAD_DIM);
	std::vector<uint32_t> table(pages),order(pages),contexts(sequences),row_position(rows),row_sequence(rows),token_sequence(tokens),token_position(tokens);
	std::vector<float> oracle_send((uint64_t)degree * degree * stride),shard_send(oracle_send.size());
	std::vector<uint16_t> shard_out((uint64_t)rows * heads * VALUE_DIM),single_out(shard_out.size()),rank_out((uint64_t)rows * width);
	std::vector<uint8_t *> shard_pool(degree);
	std::vector<float> sink_values(heads);
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device;
	uint16_t *values_device,*query_device,*out_device,*single_device;
	uint8_t *replicated_device;
	float *send_device,*receive_device,*oracle_device,*sinks_device = 0;
	LmKvAccessError *error_device,error_host;
	LmKvView view;
	uint32_t row,page,rank,index,sequence,state = 11u + rows * 131u + context + heads;
	uint64_t shard_bytes = 0u;
	double worst = 0.0,single_worst = 0.0;
	char label[200];

	snprintf(label,sizeof(label),"heads%u kv%u dim%u/%u seq%u x rows%u ctx%u degree%u grain%u window%u sinks%u",heads,KV_HEADS,HEAD_DIM,VALUE_DIM,sequences,rows_per_sequence,context,degree,grain,window,sinks ? 1u : 0u);
	for (index=0u; index<heads; index++)
		sink_values[index] = (float)((index * 37u) % 13u) * 0.5f - 2.0f;
	if ( sinks )
		sinks_device = ProbeDevice(sink_values);
	ProbeCheck(LmGqaShardGeometryValid(heads,KV_HEADS,VALUE_DIM,degree) != 0u,"the probe geometry is valid",label);
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
	for (sequence=0u; sequence<sequences; sequence++)
	{
		contexts[sequence] = context - (sequence * 37u) % (context / 4u);
		for (page=0u; page<pages_per_sequence; page++)
			table[sequence * pages_per_sequence + page] = order[sequence * pages_per_sequence + page];
		for (index=0u; index<context; index++)
		{
			token_sequence[(uint64_t)sequence * context + index] = sequence;
			token_position[(uint64_t)sequence * context + index] = index;
		}
		for (index=0u; index<rows_per_sequence; index++)
		{
			row = sequence * rows_per_sequence + index;
			row_sequence[row] = sequence;
			row_position[row] = contexts[sequence] - rows_per_sequence + index;
		}
	}
	table_device = ProbeDevice(table);
	contexts_device = ProbeDevice(contexts);
	positions_device = ProbeDevice(row_position);
	sequences_device = ProbeDevice(row_sequence);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	PROBE_CUDA(cudaMalloc((void **)&values_device,tokens * slot_elements * 2u));
	PROBE_CUDA(cudaMalloc((void **)&query_device,(uint64_t)rows * heads * HEAD_DIM * 2u));
	PROBE_CUDA(cudaMalloc((void **)&replicated_device,(uint64_t)pages * Geometry::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&send_device,oracle_send.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&receive_device,oracle_send.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&oracle_device,oracle_send.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out_device,(uint64_t)rows * width * 2u));
	PROBE_CUDA(cudaMalloc((void **)&single_device,single_out.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	PROBE_CUDA(cudaMemset(send_device,0xff,oracle_send.size() * sizeof(float)));
	PROBE_CUDA(cudaMemset(oracle_device,0xff,oracle_send.size() * sizeof(float)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,tokens * slot_elements,0u);
	ProbeFillKernel<<<256,256>>>(query_device,(uint64_t)rows * heads * HEAD_DIM,query_salt);
	PROBE_CUDA(cudaMemset(replicated_device,0,(uint64_t)pages * Geometry::kPageBytes));
	ProbeCheck(LmKvViewInitialize(&view,replicated_device,table_device,pages_per_sequence,sequences,pages,error_device) == 0,"replicated view",label);
	LmKvStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,slot_elements);
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
		ProbeCheck(LmKvShardViewInitialize<Geometry>(&local,shard_pool[rank],table_device,pages_per_sequence,sequences,pages,error_device,shard) == 0,"shard view",label);
		ProbeCheck(LmKvShardReplicaViewInitialize<Geometry>(&replica,view,shard) == 0,"replica view",label);
		LmKvShardStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(local,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,slot_elements);
		PROBE_CUDA(cudaGetLastError());
		ProbeCheck(LmGqaShardPartialLaunch<Geometry,LmKvShardReplicaView,KV_HEADS,HEAD_DIM,VALUE_DIM>(replica,query_device,heads,sequences_device,contexts_device,positions_device,scale,oracle_device + (uint64_t)rank * degree * stride,stride,rows,0,0,0u,1u,window) == cudaSuccess,"oracle partial launch",label);
		ProbeCheck(LmGqaShardPartialLaunch<Geometry,LmKvShardView,KV_HEADS,HEAD_DIM,VALUE_DIM>(local,query_device,heads,sequences_device,contexts_device,positions_device,scale,send_device + (uint64_t)rank * degree * stride,stride,rows,0,0,0u,1u,window) == cudaSuccess,"shard partial launch",label);
	}
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(oracle_send.data(),oracle_device,oracle_send.size() * sizeof(float),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(shard_send.data(),send_device,shard_send.size() * sizeof(float),cudaMemcpyDeviceToHost));
	ProbeCheck(memcmp(oracle_send.data(),shard_send.data(),oracle_send.size() * sizeof(float)) == 0,"partials from the rank's own KV equal the replicated-storage oracle bit for bit",label);
	for (rank=0u; rank<degree; rank++)
		for (index=0u; index<degree; index++)
			PROBE_CUDA(cudaMemcpy(receive_device + ((uint64_t)rank * degree + index) * stride,send_device + ((uint64_t)index * degree + rank) * stride,stride * sizeof(float),cudaMemcpyDeviceToDevice));
	for (rank=0u; rank<degree; rank++)
	{
		ProbeCheck(LmGqaShardMergeLaunch<VALUE_DIM>(receive_device + (uint64_t)rank * degree * stride,stride,degree,heads,rank,out_device,rows,0,sinks_device) == cudaSuccess,"merge launch",label);
		PROBE_CUDA(cudaMemcpy(rank_out.data(),out_device,rank_out.size() * 2u,cudaMemcpyDeviceToHost));
		for (row=0u; row<rows; row++)
			memcpy(shard_out.data() + (uint64_t)row * heads * VALUE_DIM + (uint64_t)rank * width,rank_out.data() + (uint64_t)row * width,(uint64_t)width * 2u);
	}
	if ( rows > 1u )
	{
		const uint32_t picks[3] = {0u,rows / 2u,rows - 1u};
		const uint64_t alone_stride = record;
		std::vector<uint16_t> alone_out(width);
		float *alone_send,*alone_receive;
		uint32_t pick;
		PROBE_CUDA(cudaMalloc((void **)&alone_send,(uint64_t)degree * degree * alone_stride * sizeof(float)));
		PROBE_CUDA(cudaMalloc((void **)&alone_receive,(uint64_t)degree * alone_stride * sizeof(float)));
		for (pick=0u; pick<3u; pick++)
		{
			row = picks[pick];
			for (rank=0u; rank<degree; rank++)
			{
				SparkKvShard shard = {degree,rank,grain};
				LmKvShardView local;
				ProbeCheck(LmKvShardViewInitialize<Geometry>(&local,shard_pool[rank],table_device,pages_per_sequence,sequences,pages,error_device,shard) == 0,"row-alone shard view",label);
				ProbeCheck(LmGqaShardPartialLaunch<Geometry,LmKvShardView,KV_HEADS,HEAD_DIM,VALUE_DIM>(local,query_device + (uint64_t)row * heads * HEAD_DIM,heads,sequences_device + row,contexts_device,positions_device + row,scale,alone_send + (uint64_t)rank * degree * alone_stride,alone_stride,1u,0,0,0u,1u,window) == cudaSuccess,"row-alone partial launch",label);
			}
			for (rank=0u; rank<degree; rank++)
			{
				for (index=0u; index<degree; index++)
					PROBE_CUDA(cudaMemcpy(alone_receive + (uint64_t)index * alone_stride,alone_send + ((uint64_t)index * degree + rank) * alone_stride,alone_stride * sizeof(float),cudaMemcpyDeviceToDevice));
				ProbeCheck(LmGqaShardMergeLaunch<VALUE_DIM>(alone_receive,alone_stride,degree,heads,rank,out_device,1u,0,sinks_device) == cudaSuccess,"row-alone merge",label);
				PROBE_CUDA(cudaMemcpy(alone_out.data(),out_device,(uint64_t)width * 2u,cudaMemcpyDeviceToHost));
				ProbeCheck(memcmp(alone_out.data(),shard_out.data() + (uint64_t)row * heads * VALUE_DIM + (uint64_t)rank * width,(uint64_t)width * 2u) == 0,"a row computed alone has the bits it has inside the batch",label);
			}
		}
		cudaFree(alone_send);
		cudaFree(alone_receive);
	}
	LmGqaAttentionDecodeKernel<Geometry,256u,KV_HEADS,HEAD_DIM,VALUE_DIM><<<dim3(rows,heads),256>>>(query_device,view,sequences_device,contexts_device,0,0u,heads,scale,single_device,positions_device);
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(single_out.data(),single_device,single_out.size() * 2u,cudaMemcpyDeviceToHost));
	for (index=0u; index<single_out.size(); index++)
	{
		double difference = fabs(ProbeFloat(single_out[index]) - ProbeFloat(shard_out[index]));
		single_worst = difference > single_worst ? difference : single_worst;
	}
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	ProbeCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	for (row=0u; row<rows; row+=(rows > 8u ? rows / 8u : 1u))
	{
		uint32_t head;
		sequence = row_sequence[row];
		for (head=0u; head<heads; head+=(heads > 6u ? heads / 6u : 1u))
		{
			const uint32_t kv_head = head / (heads / KV_HEADS);
			const uint64_t q_base = ((uint64_t)row * heads + head) * HEAD_DIM;
			const uint16_t *out = shard_out.data() + ((uint64_t)row * heads + head) * VALUE_DIM;
			const uint32_t keys = row_position[row] + 1u;
			const uint32_t lowest = window != 0u && keys > window ? keys - window : 0u;
			std::vector<double> scores(keys),result(VALUE_DIM,0.0);
			double top = -1.0e300,total = 0.0;
			uint32_t key,element;
			for (key=lowest; key<keys; key++)
			{
				const uint64_t base = ((uint64_t)sequence * context + key) * slot_elements + (uint64_t)kv_head * HEAD_DIM;
				double dot = 0.0;
				for (element=0u; element<HEAD_DIM; element++)
					dot += ProbeFloat(ProbeValue((q_base + element) ^ query_salt)) * ProbeFloat(ProbeValue(base + element));
				scores[key] = dot * scale;
				top = scores[key] > top ? scores[key] : top;
			}
			if ( sinks )
			{
				top = sink_values[head] > top ? sink_values[head] : top;
				total = exp(sink_values[head] - top);
			}
			for (key=lowest; key<keys; key++)
			{
				const uint64_t base = ((uint64_t)sequence * context + key) * slot_elements + (uint64_t)KV_HEADS * HEAD_DIM + (uint64_t)kv_head * VALUE_DIM;
				const double weight = exp(scores[key] - top);
				total += weight;
				for (element=0u; element<VALUE_DIM; element++)
					result[element] += weight * ProbeFloat(ProbeValue(base + element));
			}
			for (element=0u; element<VALUE_DIM; element++)
			{
				const double difference = fabs(result[element] / total - ProbeFloat(out[element]));
				worst = difference > worst ? difference : worst;
			}
		}
	}
	ProbeCheck(worst < 2.0e-3,"merged output matches the f64 attention reference",label);
	ProbeCheck(window != 0u || sinks || single_worst < 4.0e-3,"merged output matches the unsharded attention kernel",label);
	printf("%s %s: KV per rank %.1f MiB of %.1f MiB, worst |out-f64| %.2e, |sharded-unsharded| %.2e\n",
		probe_failures == 0 ? "PASS" : "FAIL",label,(double)shard_bytes / degree / 1048576.0,(double)pages * Geometry::kPageBytes / 1048576.0,worst,single_worst);
	for (rank=0u; rank<degree; rank++)
		cudaFree(shard_pool[rank]);
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(values_device); cudaFree(query_device);
	cudaFree(replicated_device); cudaFree(send_device); cudaFree(receive_device); cudaFree(oracle_device);
	cudaFree(out_device); cudaFree(single_device); cudaFree(error_device);
	if ( sinks_device != 0 )
		cudaFree(sinks_device);
}

template<uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static void ProbeTiled(uint32_t heads,uint32_t sequences,uint32_t rows_per_sequence,uint32_t context,uint32_t degree,uint32_t rank,uint32_t grain,uint32_t window,uint32_t block_tokens,bool timed)
{
	typedef ProbeGqaKv<KV_HEADS,HEAD_DIM,VALUE_DIM> Geometry;
	const uint32_t group = heads / KV_HEADS,slot_elements = KV_HEADS * (HEAD_DIM + VALUE_DIM),rows = sequences * rows_per_sequence;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,degree),slot_count = LmGqaShardHeadSlots(heads,VALUE_DIM,degree),record = width + 2u * slot_count;
	const uint32_t pages_per_sequence = (context + PROBE_PAGE - 1u) / PROBE_PAGE,pages = pages_per_sequence * sequences;
	const uint32_t mask_words = block_tokens != 0u ? (context / block_tokens + 32u) / 32u : 0u;
	const uint64_t tokens = (uint64_t)sequences * context,stride = (uint64_t)rows * record;
	const float scale = 1.0f / sqrtf((float)HEAD_DIM);
	SparkKvShard shard = {degree,rank,grain};
	std::vector<uint32_t> table(pages),contexts(sequences),row_position(rows),row_sequence(rows),token_sequence(tokens),token_position(tokens),mask((uint64_t)rows * (mask_words != 0u ? mask_words : 1u));
	std::vector<float> tiled((uint64_t)degree * stride),scalar(tiled.size());
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device,*mask_device;
	uint16_t *values_device,*query_device;
	uint8_t *pool;
	float *tiled_device,*scalar_device;
	LmKvAccessError *error_device,error_host;
	LmKvShardView local;
	uint32_t row,sequence,index,destination,slot,state = 7u + rows * 17u + context;
	double worst = 0.0,worst_max = 0.0;
	char label[200];

	snprintf(label,sizeof(label),"tiled heads%u kv%u dim%u/%u seq%u x rows%u ctx%u degree%u rank%u grain%u window%u mask%u",heads,KV_HEADS,HEAD_DIM,VALUE_DIM,sequences,rows_per_sequence,context,degree,rank,grain,window,block_tokens);
	for (index=0u; index<pages; index++)
		table[index] = index;
	for (sequence=0u; sequence<sequences; sequence++)
	{
		contexts[sequence] = context - (sequence * 101u) % (context / 3u + 1u);
		for (index=0u; index<context; index++)
		{
			token_sequence[(uint64_t)sequence * context + index] = sequence;
			token_position[(uint64_t)sequence * context + index] = index;
		}
		for (index=0u; index<rows_per_sequence; index++)
		{
			row = sequence * rows_per_sequence + index;
			row_sequence[row] = sequence;
			row_position[row] = contexts[sequence] - rows_per_sequence + index;
		}
	}
	for (index=0u; index<mask.size(); index++)
	{
		state = state * 1664525u + 1013904223u;
		mask[index] = state ^ (state >> 13u);
	}
	table_device = ProbeDevice(table);
	contexts_device = ProbeDevice(contexts);
	positions_device = ProbeDevice(row_position);
	sequences_device = ProbeDevice(row_sequence);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	mask_device = ProbeDevice(mask);
	PROBE_CUDA(cudaMalloc((void **)&values_device,tokens * slot_elements * 2u));
	PROBE_CUDA(cudaMalloc((void **)&query_device,(uint64_t)rows * heads * HEAD_DIM * 2u));
	PROBE_CUDA(cudaMalloc((void **)&pool,SparkKvShardPoolBytes(shard,PROBE_PAGE,Geometry::kSlotBytes,pages)));
	PROBE_CUDA(cudaMalloc((void **)&tiled_device,tiled.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&scalar_device,scalar.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	PROBE_CUDA(cudaMemset(tiled_device,0,tiled.size() * sizeof(float)));
	PROBE_CUDA(cudaMemset(scalar_device,0,scalar.size() * sizeof(float)));
	PROBE_CUDA(cudaMemset(pool,0,SparkKvShardPoolBytes(shard,PROBE_PAGE,Geometry::kSlotBytes,pages)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,tokens * slot_elements,(uint64_t)rows << 40u);
	ProbeFillKernel<<<256,256>>>(query_device,(uint64_t)rows * heads * HEAD_DIM,(1ull << 61u) + context);
	ProbeCheck(LmKvShardViewInitialize<Geometry>(&local,pool,table_device,pages_per_sequence,sequences,pages,error_device,shard) == 0,"shard view",label);
	LmKvShardStoreKernel<Geometry,256u><<<(uint32_t)tokens,256>>>(local,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,slot_elements);
	PROBE_CUDA(cudaGetLastError());
	ProbeCheck(LmGqaShardPartialLaunch<Geometry,LmKvShardView,KV_HEADS,HEAD_DIM,VALUE_DIM>(local,query_device,heads,sequences_device,contexts_device,positions_device,scale,tiled_device,stride,rows,0,
		block_tokens != 0u ? mask_device : 0,mask_words,block_tokens != 0u ? block_tokens : 1u,window) == cudaSuccess,"tiled launch",label);
	LmGqaShardPartialKernel<Geometry,LmKvShardView,KV_HEADS,HEAD_DIM,VALUE_DIM><<<dim3(rows,KV_HEADS),group * 32u>>>(local,query_device,heads,sequences_device,contexts_device,positions_device,scale,
		scalar_device,stride,rows,block_tokens != 0u ? mask_device : 0,mask_words,block_tokens != 0u ? block_tokens : 1u,window);
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(tiled.data(),tiled_device,tiled.size() * sizeof(float),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(scalar.data(),scalar_device,scalar.size() * sizeof(float),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	ProbeCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	for (destination=0u; destination<degree; destination++)
		for (row=0u; row<rows; row++)
		{
			const float *a = tiled.data() + (uint64_t)destination * stride + (uint64_t)row * record,*b = scalar.data() + (uint64_t)destination * stride + (uint64_t)row * record;
			for (index=0u; index<width; index++)
			{
				const uint32_t head = (destination * width + index) / VALUE_DIM;
				slot = head - (destination * width) / VALUE_DIM;
				const float a_max = a[width + 2u * slot],b_max = b[width + 2u * slot];
				if ( a_max == -INFINITY || b_max == -INFINITY )
				{
					ProbeCheck(a_max == b_max && a[width + 2u * slot + 1u] == 0.0f && a[index] == 0.0f,"a head with no visible key on this rank has max -inf, sum 0 and value 0 in both kernels",label);
					continue;
				}
				const double a_value = a[index] / a[width + 2u * slot + 1u],b_value = b[index] / b[width + 2u * slot + 1u];
				worst = fabs(a_value - b_value) > worst ? fabs(a_value - b_value) : worst;
				worst_max = fabs((double)a_max - b_max) > worst_max ? fabs((double)a_max - b_max) : worst_max;
			}
		}
	ProbeCheck(worst < ldexp(1.0,-9) && worst_max < 1.0e-4,"tiled partials agree with the scalar kernel within the bf16 rounding of the attention weights (normalized values within 2^-9 of values in [-1, 1), running maxima)",label);
	if ( timed )
	{
		cudaEvent_t start,stop;
		float tiled_ms = 0.0f,scalar_ms = 0.0f;
		const uint32_t repeats = 5u;
		const double flops = 4.0 * rows * heads * (double)SparkKvShardLocalKeys(shard,context) * HEAD_DIM;
		PROBE_CUDA(cudaEventCreate(&start));
		PROBE_CUDA(cudaEventCreate(&stop));
		PROBE_CUDA(cudaEventRecord(start));
		for (index=0u; index<repeats; index++)
			LmGqaShardPartialLaunch<Geometry,LmKvShardView,KV_HEADS,HEAD_DIM,VALUE_DIM>(local,query_device,heads,sequences_device,contexts_device,positions_device,scale,tiled_device,stride,rows,0);
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&tiled_ms,start,stop));
		PROBE_CUDA(cudaEventRecord(start));
		for (index=0u; index<repeats; index++)
			LmGqaShardPartialKernel<Geometry,LmKvShardView,KV_HEADS,HEAD_DIM,VALUE_DIM><<<dim3(rows,KV_HEADS),group * 32u>>>(local,query_device,heads,sequences_device,contexts_device,positions_device,scale,
				scalar_device,stride,rows,0,0u,1u,0u);
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&scalar_ms,start,stop));
		printf("TIME %s: tiled %.3f ms (%.1f TFLOP/s), scalar %.3f ms (%.1f TFLOP/s), %.1fx\n",label,tiled_ms / repeats,flops / (tiled_ms / repeats * 1.0e-3) / 1.0e12,
			scalar_ms / repeats,flops / (scalar_ms / repeats * 1.0e-3) / 1.0e12,scalar_ms / tiled_ms);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	printf("%s %s: worst |tiled-scalar| normalized %.2e, max %.2e\n",probe_failures == 0 ? "PASS" : "FAIL",label,worst,worst_max);
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device); cudaFree(token_sequence_device);
	cudaFree(token_position_device); cudaFree(mask_device); cudaFree(values_device); cudaFree(query_device); cudaFree(pool);
	cudaFree(tiled_device); cudaFree(scalar_device); cudaFree(error_device);
}

int main(void)
{
	ProbeCase<4u,256u,256u>(24u,1u,1u,1024u,16u,1u);
	ProbeCase<4u,256u,256u>(24u,8u,1u,8192u,16u,1u);
	ProbeCase<4u,256u,256u>(24u,1u,64u,2048u,16u,1u);
	ProbeCase<4u,256u,256u>(24u,2u,33u,4096u,16u,4u);
	ProbeCase<2u,256u,256u>(24u,4u,1u,4096u,16u,1u);
	ProbeCase<2u,128u,128u>(16u,1u,48u,1024u,8u,1u);
	ProbeCase<8u,128u,128u>(64u,4u,1u,2048u,16u,1u);
	ProbeCase<16u,128u,128u>(32u,2u,17u,1024u,4u,2u);
	ProbeCase<16u,256u,256u>(32u,2u,33u,4096u,16u,1u,1024u);
	ProbeCase<16u,256u,256u>(32u,1u,1u,1500u,16u,1u,1024u);
	ProbeCase<4u,512u,512u>(32u,2u,9u,2048u,16u,1u);
	ProbeCase<4u,192u,128u>(64u,2u,9u,2048u,16u,1u);
	ProbeCase<8u,192u,128u>(64u,2u,33u,1024u,16u,1u,128u,true);
	ProbeCase<8u,192u,128u>(64u,4u,1u,700u,16u,1u,128u,true);
	ProbeCase<8u,192u,128u>(128u,1u,17u,512u,16u,1u,0u,true);
	ProbeTiled<8u,128u,128u>(64u,1u,1u,4096u,16u,3u,1u,0u,0u,false);
	ProbeTiled<8u,128u,128u>(64u,4u,1u,4096u,16u,0u,1u,0u,0u,false);
	ProbeTiled<8u,128u,128u>(64u,3u,37u,6000u,16u,5u,1u,0u,0u,false);
	ProbeTiled<8u,128u,128u>(64u,2u,300u,5000u,16u,15u,4u,128u,0u,false);
	ProbeTiled<8u,128u,128u>(64u,2u,200u,9000u,4u,1u,1u,4096u,0u,false);
	ProbeTiled<4u,128u,128u>(24u,3u,41u,3000u,16u,2u,1u,0u,64u,false);
	ProbeTiled<8u,192u,128u>(64u,2u,90u,4000u,16u,7u,1u,128u,0u,false);
	ProbeTiled<4u,256u,256u>(24u,3u,41u,3000u,16u,6u,1u,0u,0u,false);
	ProbeTiled<16u,256u,256u>(32u,2u,70u,5000u,16u,9u,1u,1024u,0u,false);
	ProbeTiled<4u,256u,256u>(24u,2u,40u,3000u,16u,1u,1u,0u,64u,false);
	ProbeTiled<4u,128u,512u>(32u,2u,33u,2048u,16u,4u,1u,0u,0u,false);
	ProbeTiled<4u,256u,256u>(24u,1u,1u,65536u,16u,0u,1u,0u,0u,true);
	ProbeTiled<4u,256u,256u>(24u,1u,2048u,65536u,16u,0u,1u,0u,0u,true);
	ProbeTiled<8u,128u,128u>(64u,1u,2048u,131072u,16u,0u,1u,0u,0u,true);
	ProbeTiled<8u,128u,128u>(64u,1u,1u,131072u,16u,0u,1u,0u,0u,true);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS context-split GQA attention on the device: decode and causal prefill rows, 2/4/8/16 KV heads, degrees 4/8/16, grains 1/2/4, partials from each rank's own KV equal the replicated oracle bit for bit, merged output matches f64 and the unsharded kernel, a row alone equals its bits inside the batch, a sliding window keeps exactly the last window positions, 512-wide heads, 192/128 key/value heads, per-head sink logits in the merge denominator; the tiled kernel (bf16 attention weights in P.V, as the reference computes them) agrees with the f32 scalar kernel within 2^-9 for decode rows, mixed sequences inside one block, windows, block masks and shard ranks\n");
	return 0;
}
