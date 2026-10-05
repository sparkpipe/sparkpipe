#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"

#define PROBE_HIDDEN 2048u
#define PROBE_MATRICES 4u
#define PROBE_STAGES 48u
#define PROBE_EXPERTS 256u
#define PROBE_TOP_K 8u
#define PROBE_EXPERT_INPUT 6144u
#define PROBE_EXPERT_OUTPUT 256u
#define PROBE_MAX_ROWS 8u

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)
#define PROBE_OK(call) do { int32_t probe_status = (call); if ( probe_status != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n",#call,probe_status); exit(1); } } while (0)

__global__ void ProbeFillKernel(uint16_t *values,uint64_t count,uint64_t salt,uint32_t fp8)
{
	uint64_t index,z;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z ^= z >> 31u;
		if ( fp8 != 0u )
			((uint8_t *)values)[index] = (uint8_t)((z & 0x80u) | (0x20u + (z >> 8u) % 0x18u));
		else
			values[index] = (uint16_t)(0x3c00u + (z & 0x3ffu) + ((z >> 20u) & 1u) * 0x8000u - ((z >> 12u) & 15u) * 0x80u);
	}
}

typedef struct ProbeChain
{
	uint16_t *hidden,*residual,*normed,*norm_weight,*weight;
}
ProbeChain;

static void ProbeChainWalk(const ProbeChain *chain,uint32_t rows,uint32_t synchronize,cudaStream_t stream)
{
	uint32_t stage;
	for (stage=0u; stage<PROBE_STAGES; stage++)
	{
		LmFusedResidualRmsNormKernel<256u,uint16_t><<<rows,256u,(PROBE_HIDDEN + 8u) * sizeof(float),stream>>>(chain->hidden,chain->residual,chain->norm_weight,chain->residual,chain->normed,PROBE_HIDDEN,PROBE_HIDDEN,1.0e-5f);
		if ( synchronize != 0u )
			PROBE_CUDA(cudaStreamSynchronize(stream));
		PROBE_OK(LmSkinnyDense<LmBf16Format>(chain->weight + (uint64_t)(stage % PROBE_MATRICES) * PROBE_HIDDEN * PROBE_HIDDEN,chain->normed,chain->hidden,0,rows,PROBE_HIDDEN,PROBE_HIDDEN,0u,0u,stream));
		if ( synchronize != 0u )
			PROBE_CUDA(cudaStreamSynchronize(stream));
	}
}

static void ProbeChainCase(uint32_t rows,cudaStream_t stream)
{
	const uint64_t count = (uint64_t)rows * PROBE_HIDDEN;
	std::vector<uint16_t> hidden[2],residual[2];
	cudaGraph_t graph;
	cudaGraphExec_t exec;
	ProbeChain chain;
	uint32_t mode;
	int failed = probe_failures;
	PROBE_CUDA(cudaMalloc((void **)&chain.hidden,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&chain.residual,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&chain.normed,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&chain.norm_weight,PROBE_HIDDEN * 2u));
	PROBE_CUDA(cudaMalloc((void **)&chain.weight,(uint64_t)PROBE_MATRICES * PROBE_HIDDEN * PROBE_HIDDEN * 2u));
	ProbeFillKernel<<<256,256>>>(chain.weight,(uint64_t)PROBE_MATRICES * PROBE_HIDDEN * PROBE_HIDDEN,7u,0u);
	ProbeFillKernel<<<64,256>>>(chain.norm_weight,PROBE_HIDDEN,3u,0u);
	for (mode=0u; mode<2u; mode++)
	{
		ProbeFillKernel<<<64,256>>>(chain.hidden,count,1u,0u);
		ProbeFillKernel<<<64,256>>>(chain.residual,count,2u,0u);
		PROBE_CUDA(cudaDeviceSynchronize());
		if ( mode == 0u )
			ProbeChainWalk(&chain,rows,1u,stream);
		else
		{
			PROBE_CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
			ProbeChainWalk(&chain,rows,0u,stream);
			PROBE_CUDA(cudaStreamEndCapture(stream,&graph));
			PROBE_CUDA(cudaGraphInstantiate(&exec,graph,0));
			PROBE_CUDA(cudaGraphLaunch(exec,stream));
			PROBE_CUDA(cudaStreamSynchronize(stream));
			PROBE_CUDA(cudaGraphExecDestroy(exec));
			PROBE_CUDA(cudaGraphDestroy(graph));
		}
		hidden[mode].resize(count);
		residual[mode].resize(count);
		PROBE_CUDA(cudaMemcpy(hidden[mode].data(),chain.hidden,count * 2u,cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(residual[mode].data(),chain.residual,count * 2u,cudaMemcpyDeviceToHost));
	}
	if ( hidden[0] != hidden[1] || residual[0] != residual[1] )
	{
		printf("FAIL rows%u: the overlapped norm -> GEMV chain differs from the serialized chain\n",rows);
		probe_failures++;
	}
	printf("%s rows%u: %u norm -> GEMV stages with early dependent launch and weight prefetch equal the serialized chain bit for bit\n",failed == probe_failures ? "PASS" : "FAIL",rows,PROBE_STAGES);
	cudaFree(chain.hidden); cudaFree(chain.residual); cudaFree(chain.normed); cudaFree(chain.norm_weight); cudaFree(chain.weight);
}

static void ProbeExpertsCase(uint32_t rows,cudaStream_t stream)
{
	const uint32_t pairs = rows * PROBE_TOP_K;
	std::vector<uint32_t> route(pairs);
	std::vector<uint16_t> out[2];
	uint32_t *device_route,*offset,*packed,*source,*prefix_up,*prefix_down,token,k,path;
	uint16_t *activation,*output,*weight;
	int failed = probe_failures;
	for (token=0u; token<rows; token++)
		for (k=0u; k<PROBE_TOP_K; k++)
			route[token * PROBE_TOP_K + k] = (k * 37u + (token % 3u) * 37u + (token / 3u) * 5u) % PROBE_EXPERTS;
	PROBE_CUDA(cudaMalloc((void **)&device_route,pairs * 4u));
	PROBE_CUDA(cudaMalloc((void **)&offset,(PROBE_EXPERTS + 1u) * 4u));
	PROBE_CUDA(cudaMalloc((void **)&packed,pairs * 4u));
	PROBE_CUDA(cudaMalloc((void **)&source,pairs * 4u));
	PROBE_CUDA(cudaMalloc((void **)&prefix_up,(PROBE_EXPERTS + 1u) * 4u));
	PROBE_CUDA(cudaMalloc((void **)&prefix_down,(PROBE_EXPERTS + 1u) * 4u));
	PROBE_CUDA(cudaMalloc((void **)&activation,(uint64_t)rows * PROBE_EXPERT_INPUT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&output,(uint64_t)pairs * PROBE_EXPERT_OUTPUT * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,(uint64_t)PROBE_EXPERTS * PROBE_EXPERT_OUTPUT * PROBE_EXPERT_INPUT));
	PROBE_CUDA(cudaMemcpy(device_route,route.data(),pairs * 4u,cudaMemcpyHostToDevice));
	ProbeFillKernel<<<256,256>>>(weight,(uint64_t)PROBE_EXPERTS * PROBE_EXPERT_OUTPUT * PROBE_EXPERT_INPUT,11u,1u);
	ProbeFillKernel<<<64,256>>>(activation,(uint64_t)rows * PROBE_EXPERT_INPUT,5u,0u);
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_OK((LmRouteBuild<256u,PROBE_EXPERTS>(device_route,rows,pairs,PROBE_TOP_K,offset,packed,source,PROBE_EXPERT_OUTPUT,PROBE_EXPERT_INPUT,64u,64u,prefix_up,prefix_down,stream)));
	for (path=0u; path<2u; path++)
	{
		PROBE_CUDA(cudaMemsetAsync(output,0,(uint64_t)pairs * PROBE_EXPERT_OUTPUT * 2u,stream));
		if ( path == 0u )
			PROBE_OK(LmSkinnyExperts<LmFp8>(weight,LmScaleTensorNone(),activation,output,device_route,packed,pairs,PROBE_TOP_K,0u,PROBE_EXPERT_INPUT,PROBE_EXPERT_OUTPUT,stream));
		else
			PROBE_OK(LmSkinnyGroupedExperts<LmFp8>(weight,LmScaleTensorNone(),activation,output,offset,source,PROBE_EXPERTS,pairs,0u,PROBE_EXPERT_INPUT,PROBE_EXPERT_OUTPUT,stream));
		PROBE_CUDA(cudaStreamSynchronize(stream));
		out[path].resize((uint64_t)pairs * PROBE_EXPERT_OUTPUT);
		PROBE_CUDA(cudaMemcpy(out[path].data(),output,(uint64_t)pairs * PROBE_EXPERT_OUTPUT * 2u,cudaMemcpyDeviceToHost));
	}
	if ( out[0] != out[1] )
	{
		printf("FAIL rows%u: grouped experts differ from the per-pair experts\n",rows);
		probe_failures++;
	}
	printf("%s rows%u: grouped FP8 gate/up experts (each expert read once for its rows) equal the per-pair kernel bit for bit\n",failed == probe_failures ? "PASS" : "FAIL",rows);
	cudaFree(device_route); cudaFree(offset); cudaFree(packed); cudaFree(source); cudaFree(prefix_up); cudaFree(prefix_down);
	cudaFree(activation); cudaFree(output); cudaFree(weight);
}

int main(void)
{
	cudaStream_t stream;
	uint32_t rows;
	PROBE_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
	for (rows=1u; rows<=PROBE_MAX_ROWS; rows++)
	{
		ProbeChainCase(rows,stream);
		ProbeExpertsCase(rows,stream);
	}
	PROBE_CUDA(cudaStreamDestroy(stream));
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS skinny early dependent launch and grouped experts on the device: rows 1..8 bitwise equal to the serialized and per-pair paths\n");
	return 0;
}
