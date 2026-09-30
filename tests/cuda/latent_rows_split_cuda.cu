#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/attn_rows.cuh"

#define PROBE_PAGE 64u
#define PROBE_LATENT 512u
#define PROBE_ROPE 64u
#define PROBE_WIDTH (PROBE_LATENT + PROBE_ROPE)
#define PROBE_THREADS 256u
#define PROBE_SPLIT_THRESHOLD 64u
#define PROBE_SELECTED 2048u
#define PROBE_REPEATS 20u

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
	float value = (float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 512.0f;
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
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
	PROBE_CUDA(cudaMalloc((void **)&device,host.size() * sizeof(T) + 4u));
	PROBE_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

static void ProbeCheck(int condition,const char *what,const char *label)
{
	if ( condition )
		return;
	printf("FAIL %s: %s\n",label,what);
	probe_failures++;
}

static uint32_t ProbeNext(uint32_t *state)
{
	*state = *state * 1664525u + 1013904223u;
	return *state >> 8u;
}

static void ProbeCase(const char *kind,uint32_t sequences,uint32_t context,uint32_t rows,uint32_t selected,uint32_t heads,uint32_t time_it)
{
	const uint32_t pages_per_sequence = (context + PROBE_PAGE - 1u) / PROBE_PAGE,pages = pages_per_sequence * sequences;
	const uint64_t tokens = (uint64_t)sequences * context;
	const float scale = 1.0f / sqrtf((float)PROBE_WIDTH);
	std::vector<uint32_t> table(pages),order(pages),contexts(sequences),row_position(rows),sequence_of_row(rows),token_sequence(tokens),token_position(tokens);
	std::vector<uint32_t> selection(selected != 0u ? (uint64_t)rows * selected : 1u,0u);
	std::vector<uint16_t> reference_out((uint64_t)rows * heads * PROBE_LATENT),rows_out(reference_out.size());
	uint32_t *table_device,*contexts_device,*positions_device,*sequences_device,*token_sequence_device,*token_position_device,*selected_device;
	uint16_t *values_device,*latent_device,*rope_device,*out_device;
	uint8_t *pool_device;
	float *split_device,reference_ms = 0.0f,rows_ms = 0.0f;
	uint32_t split_blocks = rows * heads * LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS,row,page,index,state = 11u + rows * 7u + context,multiprocessors,bound = 0u;
	LmKvAccessError *error_device,error_host;
	LmKvView view;
	cudaDeviceProp properties;
	cudaEvent_t begin,end;
	char label[160];

	PROBE_CUDA(cudaGetDeviceProperties(&properties,0));
	multiprocessors = (uint32_t)properties.multiProcessorCount;
	snprintf(label,sizeof(label),"%s seq%u ctx%u rows%u heads%u selected%u",kind,sequences,context,rows,heads,selected);
	for (page=0u; page<pages; page++)
		order[page] = page;
	for (page=pages; page>1u; page--)
	{
		uint32_t pick = ProbeNext(&state) % page,swap = order[page - 1u];
		order[page - 1u] = order[pick];
		order[pick] = swap;
	}
	for (index=0u; index<sequences; index++)
	{
		contexts[index] = context - (index * 131u) % (context / 4u + 1u);
		for (page=0u; page<pages_per_sequence; page++)
			table[index * pages_per_sequence + page] = order[index * pages_per_sequence + page];
	}
	for (index=0u; index<tokens; index++)
	{
		token_sequence[index] = (uint32_t)(index / context);
		token_position[index] = (uint32_t)(index % context);
	}
	for (row=0u; row<rows; row++)
	{
		uint32_t sequence = row % sequences,round = row / sequences,per = (rows + sequences - 1u) / sequences;
		sequence_of_row[row] = sequence;
		row_position[row] = contexts[sequence] - per + round;
		bound = row_position[row] + 1u > bound ? row_position[row] + 1u : bound;
		if ( selected != 0u )
			for (index=0u; index<selected; index++)
				selection[(uint64_t)row * selected + index] = ProbeNext(&state) % (row_position[row] + 1u + row_position[row] / 64u);
	}
	table_device = ProbeDevice(table);
	contexts_device = ProbeDevice(contexts);
	positions_device = ProbeDevice(row_position);
	sequences_device = ProbeDevice(sequence_of_row);
	token_sequence_device = ProbeDevice(token_sequence);
	token_position_device = ProbeDevice(token_position);
	selected_device = ProbeDevice(selection);
	PROBE_CUDA(cudaMalloc((void **)&values_device,tokens * PROBE_WIDTH * 2u));
	PROBE_CUDA(cudaMalloc((void **)&latent_device,(uint64_t)rows * heads * PROBE_LATENT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&rope_device,(uint64_t)rows * heads * PROBE_ROPE * 2u));
	PROBE_CUDA(cudaMalloc((void **)&pool_device,(uint64_t)pages * ProbeKv::kPageBytes));
	PROBE_CUDA(cudaMalloc((void **)&split_device,(uint64_t)split_blocks * (PROBE_LATENT + 2u) * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out_device,reference_out.size() * 2u));
	PROBE_CUDA(cudaMalloc((void **)&error_device,sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error_host);
	PROBE_CUDA(cudaMemcpy(error_device,&error_host,sizeof(error_host),cudaMemcpyHostToDevice));
	ProbeFillKernel<<<1024,256>>>(values_device,tokens * PROBE_WIDTH,0u);
	ProbeFillKernel<<<256,256>>>(latent_device,(uint64_t)rows * heads * PROBE_LATENT,1ull << 60u);
	ProbeFillKernel<<<256,256>>>(rope_device,(uint64_t)rows * heads * PROBE_ROPE,1ull << 61u);
	PROBE_CUDA(cudaMemset(pool_device,0,(uint64_t)pages * ProbeKv::kPageBytes));
	ProbeCheck(LmKvViewInitialize(&view,pool_device,table_device,pages_per_sequence,sequences,pages,error_device) == 0,"view",label);
	LmKvStoreKernel<ProbeKv,256u><<<(uint32_t)tokens,256>>>(view,values_device,token_sequence_device,token_position_device,(uint32_t)tokens,PROBE_WIDTH);
	PROBE_CUDA(cudaGetLastError());
	PROBE_CUDA(cudaMemset(out_device,0,reference_out.size() * 2u));
	PROBE_CUDA((LmLatentAttentionDecodeSplitLaunch<ProbeKv,PROBE_THREADS,PROBE_LATENT,PROBE_ROPE,true>(latent_device,rope_device,view,sequences_device,contexts_device,selected != 0u ? selected_device : 0,selected,heads,scale,out_device,positions_device,rows,selected != 0u ? selected : bound,PROBE_SPLIT_THRESHOLD,split_device,split_blocks,multiprocessors,0)));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(reference_out.data(),out_device,reference_out.size() * 2u,cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemset(out_device,0xff,reference_out.size() * 2u));
	PROBE_CUDA(cudaMemset(split_device,0xff,(uint64_t)split_blocks * (PROBE_LATENT + 2u) * sizeof(float)));
	PROBE_CUDA((LmLatentAttentionRowsSplitLaunch<ProbeKv,PROBE_THREADS,PROBE_LATENT,PROBE_ROPE>(latent_device,rope_device,view,sequences_device,contexts_device,selected != 0u ? selected_device : 0,selected,heads,scale,out_device,positions_device,rows,selected != 0u ? selected : bound,PROBE_SPLIT_THRESHOLD,split_device,split_blocks,multiprocessors,0)));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(rows_out.data(),out_device,rows_out.size() * 2u,cudaMemcpyDeviceToHost));
	ProbeCheck(memcmp(reference_out.data(),rows_out.data(),rows_out.size() * 2u) == 0,"the rows kernel equals the row-invariant split kernel bit for bit",label);
	{
		uint64_t nonzero = 0u;
		for (index=0u; index<reference_out.size(); index++)
			nonzero += (reference_out[index] & 0x7fffu) != 0u;
		ProbeCheck(nonzero * 2u > reference_out.size(),"the reference output is not trivially zero",label);
	}
	PROBE_CUDA(cudaMemcpy(&error_host,error_device,sizeof(error_host),cudaMemcpyDeviceToHost));
	ProbeCheck(error_host.error_code == LM_FRAME_ERROR_NONE,"no KV access error",label);
	if ( time_it != 0u )
	{
		PROBE_CUDA(cudaEventCreate(&begin));
		PROBE_CUDA(cudaEventCreate(&end));
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
			PROBE_CUDA((LmLatentAttentionDecodeSplitLaunch<ProbeKv,PROBE_THREADS,PROBE_LATENT,PROBE_ROPE,true>(latent_device,rope_device,view,sequences_device,contexts_device,selected != 0u ? selected_device : 0,selected,heads,scale,out_device,positions_device,rows,selected != 0u ? selected : bound,PROBE_SPLIT_THRESHOLD,split_device,split_blocks,multiprocessors,0)));
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&reference_ms,begin,end));
		PROBE_CUDA(cudaEventRecord(begin));
		for (index=0u; index<PROBE_REPEATS; index++)
			PROBE_CUDA((LmLatentAttentionRowsSplitLaunch<ProbeKv,PROBE_THREADS,PROBE_LATENT,PROBE_ROPE>(latent_device,rope_device,view,sequences_device,contexts_device,selected != 0u ? selected_device : 0,selected,heads,scale,out_device,positions_device,rows,selected != 0u ? selected : bound,PROBE_SPLIT_THRESHOLD,split_device,split_blocks,multiprocessors,0)));
		PROBE_CUDA(cudaEventRecord(end));
		PROBE_CUDA(cudaEventSynchronize(end));
		PROBE_CUDA(cudaEventElapsedTime(&rows_ms,begin,end));
		cudaEventDestroy(begin);
		cudaEventDestroy(end);
	}
	printf("%s %s: split kernel %.1f us, rows kernel %.1f us\n",probe_failures == 0 ? "PASS" : "FAIL",label,reference_ms * 1000.0f / PROBE_REPEATS,rows_ms * 1000.0f / PROBE_REPEATS);
	cudaFree(table_device); cudaFree(contexts_device); cudaFree(positions_device); cudaFree(sequences_device);
	cudaFree(token_sequence_device); cudaFree(token_position_device); cudaFree(selected_device); cudaFree(values_device);
	cudaFree(latent_device); cudaFree(rope_device); cudaFree(pool_device); cudaFree(split_device); cudaFree(out_device); cudaFree(error_device);
}

int main(int argc,char **argv)
{
	uint32_t time_it = argc > 1 && strcmp(argv[1],"--time") == 0 ? 1u : 0u;
	ProbeCase("prefill",1u,300u,128u,0u,4u,time_it);
	ProbeCase("prefill",1u,1024u,128u,0u,4u,time_it);
	ProbeCase("prefill",1u,1024u,8u,0u,4u,time_it);
	ProbeCase("prefill",1u,2048u,128u,0u,4u,time_it);
	ProbeCase("prefill",1u,2048u,77u,0u,8u,time_it);
	ProbeCase("rounds",4u,700u,64u,0u,4u,time_it);
	ProbeCase("decode",16u,1500u,16u,0u,4u,time_it);
	ProbeCase("decode",1u,1500u,1u,0u,4u,time_it);
	ProbeCase("selected",1u,4096u,32u,PROBE_SELECTED,4u,time_it);
	ProbeCase("selected",8u,3000u,16u,PROBE_SELECTED,4u,time_it);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS latent rows split attention on the device: prefill tiles, round-major waves, decode rows and DSA-selected rows equal the row-invariant split kernel bit for bit\n");
	return 0;
}
