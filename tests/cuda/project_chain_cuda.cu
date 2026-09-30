#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/project_chain.cuh"

#define PROBE_HEADS 4u
#define PROBE_REPEATS 200u

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

__global__ void ProbeFillKernel(uint16_t *values,uint64_t count,uint64_t salt)
{
	uint64_t index,z;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z ^= z >> 31u;
		values[index] = (uint16_t)(0x3c00u + (z & 0x3ffu) + ((z >> 20u) & 1u) * 0x8000u - ((z >> 12u) & 7u) * 0x80u);
	}
}

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET>
static void ProbeShape(uint32_t rows)
{
	const uint64_t input_count = (uint64_t)rows * PROBE_HEADS * INPUT_HEAD_DIM,weight_count = (uint64_t)PROBE_HEADS * OUT_DIM * IN_DIM,output_count = (uint64_t)rows * PROBE_HEADS * OUT_DIM;
	std::vector<uint16_t> tile(output_count),chain(output_count);
	uint16_t *input,*weight,*output;
	cudaEvent_t begin,end;
	float tile_ms,chain_ms;
	uint32_t index;
	char label[96];
	snprintf(label,sizeof(label),"in%u out%u head%u offset%u rows%u",IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET,rows);
	PROBE_CUDA(cudaMalloc((void **)&input,input_count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,weight_count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&output,output_count * 2u));
	ProbeFillKernel<<<256,256>>>(input,input_count,rows * 7919u);
	ProbeFillKernel<<<256,256>>>(weight,weight_count,1ull << 50u);
	PROBE_CUDA(cudaEventCreate(&begin));
	PROBE_CUDA(cudaEventCreate(&end));
	PROBE_CUDA(cudaMemset(output,0xff,output_count * 2u));
	PROBE_CUDA((LmPerHeadProjectRowsLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaMemcpy(tile.data(),output,output_count * 2u,cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemset(output,0xff,output_count * 2u));
	PROBE_CUDA((LmPerHeadProjectChainLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaMemcpy(chain.data(),output,output_count * 2u,cudaMemcpyDeviceToHost));
	if ( memcmp(tile.data(),chain.data(),output_count * 2u) != 0 )
	{
		for (index=0u; index<output_count && tile[index] == chain[index]; index++)
			;
		printf("FAIL %s: chain output differs from the tile kernel at %u (%04x vs %04x)\n",label,index,chain[index],tile[index]);
		probe_failures++;
	}
	PROBE_CUDA(cudaEventRecord(begin));
	for (index=0u; index<PROBE_REPEATS; index++)
		PROBE_CUDA((LmPerHeadProjectRowsLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaEventRecord(end));
	PROBE_CUDA(cudaEventSynchronize(end));
	PROBE_CUDA(cudaEventElapsedTime(&tile_ms,begin,end));
	PROBE_CUDA(cudaEventRecord(begin));
	for (index=0u; index<PROBE_REPEATS; index++)
		PROBE_CUDA((LmPerHeadProjectChainLaunch<256u,IN_DIM,OUT_DIM,INPUT_HEAD_DIM,INPUT_OFFSET>(input,weight,output,PROBE_HEADS,rows,0)));
	PROBE_CUDA(cudaEventRecord(end));
	PROBE_CUDA(cudaEventSynchronize(end));
	PROBE_CUDA(cudaEventElapsedTime(&chain_ms,begin,end));
	printf("%s %s: bit-equal to the tile kernel, us tile %.1f chain %.1f\n",probe_failures == 0 ? "PASS" : "FAIL",label,tile_ms * 1000.0f / PROBE_REPEATS,chain_ms * 1000.0f / PROBE_REPEATS);
	cudaFree(input); cudaFree(weight); cudaFree(output);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
}

int main(void)
{
	uint32_t rows;
	for (rows=1u; rows<=8u; rows++)
	{
		ProbeShape<512u,256u,512u,0u>(rows);
		ProbeShape<192u,512u,256u,0u>(rows);
		ProbeShape<64u,128u,256u,192u>(rows);
	}
	ProbeShape<512u,256u,512u,0u>(16u);
	ProbeShape<192u,512u,256u,0u>(13u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS per-head chain projection on the device: rows 1..8 and the tile fallback above 8 are bit-equal to the tile kernel\n");
	return 0;
}
