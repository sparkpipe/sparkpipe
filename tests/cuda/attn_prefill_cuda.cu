#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_rope_heads.cuh"
#include "inference/kernels/attn_prefill.cuh"

#define PROBE_PAGE 64u
#define PROBE_LATENT 512u
#define PROBE_ROPE 64u
#define PROBE_WIDTH (PROBE_LATENT + PROBE_ROPE)
#define PROBE_HEADS 4u
#define PROBE_THRESHOLD 64u
#define PROBE_SAMPLES 12u
#define PROBE_REPEATS 10u

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

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<class T>
static T *ProbeDevice(const std::vector<T> &host)
{
	T *device;
	PROBE_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T)));
	PROBE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static double ProbeUlp(double value)
{
	int exponent;
	frexp(fabs(value) > 1.0e-30 ? value : 1.0e-30,&exponent);
	return(ldexp(1.0,exponent - 8));
}

static void ProbeCase(uint32_t rows,uint32_t prefix,uint32_t sequence,uint32_t timing)
{
	const uint32_t context = prefix + rows,sequence_pages = (context + PROBE_PAGE - 1u) / PROBE_PAGE,sequence_count = sequence + 2u,pages = sequence_pages * sequence_count;
	const uint64_t query_count = (uint64_t)rows * PROBE_HEADS;
	const uint32_t partial_blocks = rows * PROBE_HEADS * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS;
	const float scale = 1.0f / sqrtf((float)PROBE_WIDTH);
	const uint32_t half = rows / 2u;
	std::vector<uint32_t> table(pages),contexts(sequence_count,context),row_position(rows),sequences(rows,sequence),token_sequence((uint64_t)context * sequence_count),token_position((uint64_t)context * sequence_count);
	std::vector<uint16_t> prefill(query_count * PROBE_LATENT),split(query_count * PROBE_LATENT),part((uint64_t)half * PROBE_HEADS * PROBE_LATENT);
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device;
	uint16_t *values_device,*latent_device,*rope_device,*out_device;
	uint8_t *cache_device;
	float *split_device,elapsed[2] = { 0.0f, 0.0f };
	LmKvAccessError *error_device,error_host;
	LmKvView view;
	cudaDeviceProp properties;
	cudaEvent_t begin,end;
	uint32_t row,page,index,sample,multiprocessors;
	double worst_prefill = 0.0,worst_split = 0.0,worst_ulps = 0.0,between = 0.0;
	int failed = probe_failures;
	char label[96];

	snprintf(label,sizeof(label),"rows%u prefix%u sequence%u",rows,prefix,sequence);
	PROBE_CUDA(cudaGetDeviceProperties(&properties,0));
	multiprocessors = (uint32_t)properties.multiProcessorCount;
	for (page=0u; page<pages; page++)
		table[page] = pages - 1u - page;
	for (index=0u; index<context * sequence_count; index++)
	{
		token_sequence[index] = index / context;
		token_position[index] = index % context;
	}
	for (row=0u; row<rows; row++)
		row_position[row] = prefix + row;
	table_device = ProbeDevice(table);
	contexts_device = ProbeDevice(contexts);
	positions_device = ProbeDevice(row_position);
	sequences_device = ProbeDevice(sequences);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	PROBE_CUDA(cudaMalloc((void **)&values_device,(uint64_t)context * sequence_count * PROBE_WIDTH * 2u));
	PROBE_CUDA(cudaMalloc((void **)&latent_device,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&rope_device,query_count * PROBE_ROPE * 2u));
	PROBE_CUDA(cudaMalloc((void **)&cache_device,(uint64_t)pages * ProbeKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&split_device,(uint64_t)partial_blocks * (PROBE_LATENT + 2u) * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out_device,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,(uint64_t)context * sequence_count * PROBE_WIDTH,0u);
	ProbeFillKernel<<<256,256>>>(latent_device,query_count * PROBE_LATENT,1ull << 60u);
	ProbeFillKernel<<<256,256>>>(rope_device,query_count * PROBE_ROPE,1ull << 61u);
	PROBE_CUDA(cudaMemset(cache_device,0,(uint64_t)pages * ProbeKv::kPageBytes));
	if ( LmKvViewInitialize(&view,cache_device,table_device,sequence_pages,sequence_count,pages,error_device) != 0 )
	{
		printf("FAIL %s: view\n",label);
		probe_failures++;
		return;
	}
	LmKvStoreKernel<ProbeKv,256u><<<context * sequence_count,256>>>(view,values_device,token_sequence_device,token_position_device,context * sequence_count,PROBE_WIDTH);
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaMemset(out_device,0xff,query_count * PROBE_LATENT * 2u));
	PROBE_CUDA((LmLatentAttentionPrefillLaunch<ProbeKv,PROBE_LATENT,PROBE_ROPE>(latent_device,rope_device,view,sequences_device,positions_device,PROBE_HEADS,scale,out_device,rows,0,0u,0)));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(prefill.data(),out_device,prefill.size() * 2u,cudaMemcpyDeviceToHost));
	PROBE_CUDA((LmLatentRopeHeadsSplitLaunch<ProbeKv,256u,PROBE_LATENT,PROBE_ROPE,true>(latent_device,rope_device,view,sequences_device,contexts_device,0,0u,PROBE_HEADS,scale,out_device,positions_device,rows,context,PROBE_THRESHOLD,split_device,partial_blocks,multiprocessors,0)));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(split.data(),out_device,split.size() * 2u,cudaMemcpyDeviceToHost));
	if ( half != 0u )
	{
		const uint32_t first = rows - half - 1u;
		PROBE_CUDA((LmLatentAttentionPrefillLaunch<ProbeKv,PROBE_LATENT,PROBE_ROPE>(latent_device + (uint64_t)first * PROBE_HEADS * PROBE_LATENT,rope_device + (uint64_t)first * PROBE_HEADS * PROBE_ROPE,view,sequences_device,positions_device + first,PROBE_HEADS,scale,out_device,half,0,0u,0)));
		PROBE_CUDA(cudaDeviceSynchronize());
		PROBE_CUDA(cudaMemcpy(part.data(),out_device,part.size() * 2u,cudaMemcpyDeviceToHost));
		if ( memcmp(part.data(),prefill.data() + (uint64_t)first * PROBE_HEADS * PROBE_LATENT,part.size() * 2u) != 0 )
		{
			printf("FAIL %s: rows computed in a shifted, shorter wave differ from their bits in the full wave\n",label);
			probe_failures++;
		}
	}
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	if ( error_host.error_code != LM_FRAME_ERROR_NONE )
	{
		printf("FAIL %s: KV access error %u\n",label,(unsigned)error_host.error_code);
		probe_failures++;
	}
	for (index=0u; index<prefill.size(); index++)
	{
		double difference = fabs(ProbeFloat(prefill[index]) - ProbeFloat(split[index]));
		between = difference > between ? difference : between;
	}
	for (sample=0u; sample<PROBE_SAMPLES; sample++)
	{
		uint32_t head,element,key;
		row = sample == 0u ? 0u : sample == 1u ? rows - 1u : (uint32_t)(((uint64_t)sample * 2654435761u) % rows);
		for (head=0u; head<PROBE_HEADS; head++)
		{
			uint64_t q = (uint64_t)row * PROBE_HEADS + head;
			uint32_t keys = row_position[row] + 1u;
			std::vector<double> scores(keys),result(PROBE_LATENT,0.0);
			double top = -1.0e300,total = 0.0;
			for (key=0u; key<keys; key++)
			{
				uint64_t k_base = ((uint64_t)sequence * context + key) * PROBE_WIDTH;
				double dot = 0.0;
				for (element=0u; element<PROBE_LATENT; element++)
					dot += ProbeFloat(ProbeValue((q * PROBE_LATENT + element) ^ (1ull << 60u))) * ProbeFloat(ProbeValue(k_base + element));
				for (element=0u; element<PROBE_ROPE; element++)
					dot += ProbeFloat(ProbeValue((q * PROBE_ROPE + element) ^ (1ull << 61u))) * ProbeFloat(ProbeValue(k_base + PROBE_LATENT + element));
				scores[key] = dot * scale;
				top = scores[key] > top ? scores[key] : top;
			}
			for (key=0u; key<keys; key++)
			{
				double weight = exp(scores[key] - top);
				total += weight;
				for (element=0u; element<PROBE_LATENT; element++)
					result[element] += weight * ProbeFloat(ProbeValue(((uint64_t)sequence * context + key) * PROBE_WIDTH + element));
			}
			for (element=0u; element<PROBE_LATENT; element++)
			{
				double reference = result[element] / total;
				double prefill_error = fabs(reference - ProbeFloat(prefill[q * PROBE_LATENT + element]));
				double split_error = fabs(reference - ProbeFloat(split[q * PROBE_LATENT + element]));
				worst_prefill = prefill_error > worst_prefill ? prefill_error : worst_prefill;
				worst_split = split_error > worst_split ? split_error : worst_split;
				worst_ulps = prefill_error / ProbeUlp(reference) > worst_ulps ? prefill_error / ProbeUlp(reference) : worst_ulps;
			}
		}
	}
	if ( !(worst_prefill <= worst_split * 1.5 + 1.0e-6) || !(worst_prefill < 2.0e-3) )
	{
		printf("FAIL %s: prefill attention error %.2e exceeds the split kernel's %.2e\n",label,worst_prefill,worst_split);
		probe_failures++;
	}
	if ( timing != 0u )
	{
		PROBE_CUDA(cudaEventCreate(&begin));
		PROBE_CUDA(cudaEventCreate(&end));
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
			PROBE_CUDA((LmLatentAttentionPrefillLaunch<ProbeKv,PROBE_LATENT,PROBE_ROPE>(latent_device,rope_device,view,sequences_device,positions_device,PROBE_HEADS,scale,out_device,rows,0,0u,0)));
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed[0],begin,end));
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
			PROBE_CUDA((LmLatentRopeHeadsSplitLaunch<ProbeKv,256u,PROBE_LATENT,PROBE_ROPE,true>(latent_device,rope_device,view,sequences_device,contexts_device,0,0u,PROBE_HEADS,scale,out_device,positions_device,rows,context,PROBE_THRESHOLD,split_device,partial_blocks,multiprocessors,0)));
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed[1],begin,end));
		elapsed[0] /= (float)PROBE_REPEATS;
		elapsed[1] /= (float)PROBE_REPEATS;
		cudaEventDestroy(begin);
		cudaEventDestroy(end);
	}
	printf("%s %s: worst |out-f64| prefill %.2e (%.2f bf16 ulp) split %.2e, |prefill-split| %.2e%s",failed == probe_failures ? "PASS" : "FAIL",label,worst_prefill,worst_ulps,worst_split,between,half != 0u ? ", rows equal in a shifted shorter wave" : "");
	if ( timing != 0u )
		printf(", us prefill %.1f split %.1f (%.1fx)",elapsed[0] * 1000.0f,elapsed[1] * 1000.0f,elapsed[1] / elapsed[0]);
	printf("\n");
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(values_device);
	cudaFree(latent_device); cudaFree(rope_device); cudaFree(cache_device); cudaFree(split_device); cudaFree(out_device); cudaFree(error_device);
}

int main(int argc,char **argv)
{
	static const uint32_t cases[][3] = { { 9u, 0u, 0u }, { 16u, 5u, 3u }, { 14u, 320u, 5u }, { 22u, 256u, 2u }, { 64u, 0u, 1u }, { 128u, 700u, 0u }, { 256u, 0u, 7u }, { 1024u, 0u, 0u }, { 1024u, 1024u, 2u } };
	uint32_t index,timing = argc > 1 && strcmp(argv[1],"--time") == 0 ? 1u : 0u;
	for (index=0u; index<sizeof(cases) / sizeof(cases[0]); index++)
		ProbeCase(cases[index][0],cases[index][1],cases[index][2],timing);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS latent prefill attention on the device: one sequence of 9..1024 causal rows after 0..1024 cached positions matches the f64 reference at least as closely as the split decode kernel, and a row's bits do not depend on the wave it runs in\n");
	return 0;
}
