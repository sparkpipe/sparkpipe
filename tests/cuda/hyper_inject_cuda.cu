#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/hyper_mix.cuh"

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
		values[index] = (uint16_t)(0x3c00u + (z & 0x3ffu) + ((z >> 20u) & 1u) * 0x8000u - ((z >> 12u) & 15u) * 0x80u);
	}
}

template<uint32_t STREAMS>
static void ProbeCase(uint32_t rows,uint32_t width)
{
	const uint64_t total = (uint64_t)STREAMS * width;
	std::vector<float> expected((uint64_t)rows * STREAMS),got((uint64_t)rows * STREAMS);
	uint16_t *normed,*weight;
	float *inject;
	PROBE_CUDA(cudaMalloc((void **)&normed,rows * total * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,STREAMS * total * 2u));
	PROBE_CUDA(cudaMalloc((void **)&inject,rows * STREAMS * sizeof(float)));
	ProbeFillKernel<<<64,256>>>(normed,rows * total,rows * 7u + width);
	ProbeFillKernel<<<64,256>>>(weight,STREAMS * total,width * 3u + 1u);
	LmHyperInjectWeightsKernel<256u,STREAMS><<<rows,256u>>>(normed,weight,inject,width);
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(expected.data(),inject,expected.size() * sizeof(float),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemset(inject,0xff,rows * STREAMS * sizeof(float)));
	PROBE_CUDA((LmHyperInjectWeightsLaunch<256u,STREAMS>(rows,0,normed,weight,inject,width)));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(got.data(),inject,got.size() * sizeof(float),cudaMemcpyDeviceToHost));
	if ( memcmp(expected.data(),got.data(),got.size() * sizeof(float)) != 0 )
	{
		printf("FAIL rows%u width%u streams%u: the cluster inject weights differ from the one-block kernel\n",rows,width,STREAMS);
		probe_failures++;
	}
	else
		printf("PASS rows%u width%u streams%u: cluster inject weights bitwise equal to the one-block kernel\n",rows,width,STREAMS);
	cudaFree(normed); cudaFree(weight); cudaFree(inject);
}

int main(void)
{
	uint32_t rows;
	for (rows = 1u; rows <= 10u; rows++)
	{
		ProbeCase<4u>(rows,2560u);
		ProbeCase<4u>(rows,2048u);
		ProbeCase<4u>(rows,1000u);
		ProbeCase<2u>(rows,3000u);
	}
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS hyper-connection inject weights on the device: 1..10 rows, cluster launch bitwise equal to the one-block kernel\n");
	return 0;
}
