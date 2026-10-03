#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_rope_heads.cuh"

#define PROBE_PAGE 64u
#define PROBE_LATENT 512u
#define PROBE_ROPE 64u
#define PROBE_WIDTH (PROBE_LATENT + PROBE_ROPE)
#define PROBE_SELECTED 2048u
#define PROBE_THRESHOLD 64u
#define PROBE_REPEATS 50u

struct ProbeKv
{
	static constexpr uint32_t kSlotBytes = PROBE_WIDTH * 2u;
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

typedef cudaError_t (*ProbeLaunch)(const uint16_t *,const uint16_t *,LmKvView,const uint32_t *,const uint32_t *,const uint32_t *,uint32_t,uint32_t,float,uint16_t *,const uint32_t *,uint32_t,uint32_t,uint32_t,float *,uint32_t,uint32_t,cudaStream_t);

static void ProbeCase(uint32_t rows,uint32_t context,uint32_t heads,uint32_t listed)
{
	const uint32_t pages_per_sequence = (context + PROBE_PAGE - 1u) / PROBE_PAGE,pages = pages_per_sequence * rows;
	const uint64_t tokens = (uint64_t)rows * context,query_count = (uint64_t)rows * heads;
	const uint32_t partial_blocks = rows * heads * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS;
	const float scale = 1.0f / sqrtf((float)PROBE_WIDTH);
	const ProbeLaunch launches[2] = {LmLatentAttentionDecodeSplitLaunch<ProbeKv,256u,PROBE_LATENT,PROBE_ROPE,true>,LmLatentRopeHeadsSplitLaunch<ProbeKv,256u,PROBE_LATENT,PROBE_ROPE,true>};
	std::vector<uint32_t> table(pages),contexts(rows),row_position(rows),sequences(rows),token_sequence(tokens),token_position(tokens);
	std::vector<uint32_t> selected((uint64_t)rows * PROBE_SELECTED,0u);
	std::vector<uint16_t> out[2],alone(query_count * PROBE_LATENT);
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device,*selected_device,*list;
	uint16_t *values_device,*latent_device,*rope_device,*out_device;
	uint8_t *cache_device;
	float *split_device,elapsed[2] = {0.0f,0.0f};
	LmKvAccessError *error_device,error_host;
	LmKvView view;
	cudaDeviceProp properties;
	cudaEvent_t begin,end;
	uint32_t row,page,index,kernel,bound,multiprocessors;
	double worst[2] = {0.0,0.0},between = 0.0;
	char label[160];

	snprintf(label,sizeof(label),"B%u ctx%u heads%u %s",rows,context,heads,listed != 0u ? "selected" : "dense");
	PROBE_CUDA(cudaGetDeviceProperties(&properties,0));
	multiprocessors = (uint32_t)properties.multiProcessorCount;
	for (row=0u; row<rows; row++)
	{
		sequences[row] = row;
		contexts[row] = context - (row * 37u) % (context / 4u);
		row_position[row] = contexts[row] - 1u;
		for (page=0u; page<pages_per_sequence; page++)
			table[row * pages_per_sequence + page] = (row * pages_per_sequence + page) * 7u % pages;
		for (index=0u; index<context; index++)
		{
			token_sequence[(uint64_t)row * context + index] = row;
			token_position[(uint64_t)row * context + index] = index;
		}
		for (index=0u; index<PROBE_SELECTED; index++)
			selected[(uint64_t)row * PROBE_SELECTED + index] = (uint32_t)(((uint64_t)index * 7919u + row) % contexts[row]);
	}
	table_device = ProbeDevice(table);
	contexts_device = ProbeDevice(contexts);
	positions_device = ProbeDevice(row_position);
	sequences_device = ProbeDevice(sequences);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	selected_device = ProbeDevice(selected);
	list = listed != 0u ? selected_device : 0;
	bound = listed != 0u ? PROBE_SELECTED : context;
	PROBE_CUDA(cudaMalloc((void **)&values_device,tokens * PROBE_WIDTH * 2u));
	PROBE_CUDA(cudaMalloc((void **)&latent_device,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&rope_device,query_count * PROBE_ROPE * 2u));
	PROBE_CUDA(cudaMalloc((void **)&cache_device,(uint64_t)pages * ProbeKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&split_device,(uint64_t)partial_blocks * (PROBE_LATENT + 2u) * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out_device,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,tokens * PROBE_WIDTH,0u);
	ProbeFillKernel<<<256,256>>>(latent_device,query_count * PROBE_LATENT,1ull << 60u);
	ProbeFillKernel<<<256,256>>>(rope_device,query_count * PROBE_ROPE,1ull << 61u);
	PROBE_CUDA(cudaMemset(cache_device,0,(uint64_t)pages * ProbeKv::kPageBytes));
	ProbeCheck(LmKvViewInitialize(&view,cache_device,table_device,pages_per_sequence,rows,pages,error_device) == 0,"view",label);
	LmKvStoreKernel<ProbeKv,256u><<<(uint32_t)tokens,256>>>(view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,PROBE_WIDTH);
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaEventCreate(&begin));
	PROBE_CUDA(cudaEventCreate(&end));
	for (kernel=0u; kernel<2u; kernel++)
	{
		out[kernel].resize(query_count * PROBE_LATENT);
		PROBE_CUDA(cudaMemset(out_device,0xff,query_count * PROBE_LATENT * 2u));
		PROBE_CUDA(launches[kernel](latent_device,rope_device,view,sequences_device,contexts_device,list,listed != 0u ? PROBE_SELECTED : 0u,heads,scale,out_device,positions_device,rows,bound,PROBE_THRESHOLD,split_device,partial_blocks,multiprocessors,0));
		PROBE_CUDA(cudaDeviceSynchronize());
		PROBE_CUDA(cudaMemcpy(out[kernel].data(),out_device,out[kernel].size() * 2u,cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
			PROBE_CUDA(launches[kernel](latent_device,rope_device,view,sequences_device,contexts_device,list,listed != 0u ? PROBE_SELECTED : 0u,heads,scale,out_device,positions_device,rows,bound,PROBE_THRESHOLD,split_device,partial_blocks,multiprocessors,0));
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed[kernel],begin,end));
		elapsed[kernel] /= (float)PROBE_REPEATS;
	}
	for (row=0u; row<rows; row++)
	{
		PROBE_CUDA(launches[1](latent_device + (uint64_t)row * heads * PROBE_LATENT,rope_device + (uint64_t)row * heads * PROBE_ROPE,view,sequences_device + row,contexts_device,list != 0 ? list + (uint64_t)row * PROBE_SELECTED : 0,listed != 0u ? PROBE_SELECTED : 0u,heads,scale,out_device,positions_device + row,1u,bound,PROBE_THRESHOLD,split_device,partial_blocks,multiprocessors,0));
		PROBE_CUDA(cudaDeviceSynchronize());
		PROBE_CUDA(cudaMemcpy(alone.data(),out_device,(uint64_t)heads * PROBE_LATENT * 2u,cudaMemcpyDeviceToHost));
		ProbeCheck(memcmp(alone.data(),out[1].data() + (uint64_t)row * heads * PROBE_LATENT,(uint64_t)heads * PROBE_LATENT * 2u) == 0,"a row decoded alone has the bits it has inside the batch",label);
	}
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	ProbeCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	for (index=0u; index<out[0].size(); index++)
	{
		double difference = fabs(ProbeFloat(out[0][index]) - ProbeFloat(out[1][index]));
		between = difference > between ? difference : between;
	}
	for (row=0u; row<rows; row++)
	{
		uint32_t head,element,key;
		std::vector<uint32_t> keys;
		if ( listed != 0u )
		{
			for (index=0u; index<PROBE_SELECTED; index++)
				if ( selected[(uint64_t)row * PROBE_SELECTED + index] <= row_position[row] )
					keys.push_back(selected[(uint64_t)row * PROBE_SELECTED + index]);
		}
		else
			for (index=0u; index<contexts[row]; index++)
				keys.push_back(index);
		for (head=0u; head<heads; head++)
		{
			uint64_t q = (uint64_t)row * heads + head;
			std::vector<double> scores(keys.size()),result(PROBE_LATENT,0.0);
			double top = -1.0e300,total = 0.0;
			for (key=0u; key<keys.size(); key++)
			{
				uint64_t k_base = ((uint64_t)row * context + keys[key]) * PROBE_WIDTH;
				double dot = 0.0;
				for (element=0u; element<PROBE_LATENT; element++)
					dot += ProbeFloat(ProbeValue((q * PROBE_LATENT + element) ^ (1ull << 60u))) * ProbeFloat(ProbeValue(k_base + element));
				for (element=0u; element<PROBE_ROPE; element++)
					dot += ProbeFloat(ProbeValue((q * PROBE_ROPE + element) ^ (1ull << 61u))) * ProbeFloat(ProbeValue(k_base + PROBE_LATENT + element));
				scores[key] = dot * scale;
				top = scores[key] > top ? scores[key] : top;
			}
			for (key=0u; key<keys.size(); key++)
			{
				uint64_t k_base = ((uint64_t)row * context + keys[key]) * PROBE_WIDTH;
				double weight = exp(scores[key] - top);
				total += weight;
				for (element=0u; element<PROBE_LATENT; element++)
					result[element] += weight * ProbeFloat(ProbeValue(k_base + element));
			}
			for (kernel=0u; kernel<2u; kernel++)
				for (element=0u; element<PROBE_LATENT; element++)
				{
					double difference = fabs(result[element] / total - ProbeFloat(out[kernel][q * PROBE_LATENT + element]));
					worst[kernel] = difference > worst[kernel] ? difference : worst[kernel];
				}
		}
	}
	ProbeCheck(worst[0] < 2.0e-3,"reference split kernel matches the f64 attention",label);
	ProbeCheck(worst[1] < 2.0e-3,"all-heads split kernel matches the f64 attention",label);
	printf("%s %s: worst |out-f64| split %.2e heads %.2e, |split-heads| %.2e, us split %.1f heads %.1f (%.2fx), KV read once %.0f GB/s\n",
		probe_failures == 0 ? "PASS" : "FAIL",label,worst[0],worst[1],between,elapsed[0] * 1000.0f,elapsed[1] * 1000.0f,elapsed[0] / elapsed[1],
		(double)rows * (listed != 0u ? PROBE_SELECTED : context) * PROBE_WIDTH * 2.0 / (elapsed[1] * 1.0e-3) / 1.0e9);
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(selected_device); cudaFree(values_device);
	cudaFree(latent_device); cudaFree(rope_device); cudaFree(cache_device); cudaFree(split_device); cudaFree(out_device); cudaFree(error_device);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
}

int main(void)
{
	uint32_t contexts[3] = {100u,1024u,2048u},rows[3] = {1u,4u,8u},heads[3] = {1u,2u,4u},c,r,h;
	for (c=0u; c<3u; c++)
		for (r=0u; r<3u; r++)
			ProbeCase(rows[r],contexts[c],4u,0u);
	for (h=0u; h<2u; h++)
		ProbeCase(4u,1024u,heads[h],0u);
	ProbeCase(1u,8192u,4u,1u);
	ProbeCase(8u,8192u,4u,1u);
	ProbeCase(2u,1024u,8u,0u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS latent all-heads split attention on the device: B1/B4/B8 at 100/1k/2k dense and 8k selected, 1/2/4/8 heads, matches the f64 reference, a row alone equals its bits inside the batch\n");
	return 0;
}
