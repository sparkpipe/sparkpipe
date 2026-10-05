#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

__global__ void ProbeFillKernel(uint16_t *values,uint64_t count,uint64_t salt,uint32_t row_spread)
{
	uint64_t index,z;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z ^= z >> 31u;
		values[index] = (uint16_t)(0x3c00u + (z & 0x3ffu) + ((z >> 20u) & 1u) * 0x8000u - ((z >> 12u) & 15u) * 0x80u + (row_spread != 0u ? (uint16_t)(((index / row_spread) % 7u) * 0x80u) : 0u));
	}
}

static float ProbeBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return(result);
}

static void ProbeCase(uint32_t rows,uint32_t dimension,uint32_t in_place,uint32_t launches)
{
	const uint64_t count = (uint64_t)rows * dimension;
	std::vector<uint16_t> input(count),weight(dimension),bias(dimension),first(count),again(count);
	uint16_t *device_input,*device_weight,*device_bias,*device_output;
	uint32_t row,column,launch,wrong_rows = 0u,unstable_launches = 0u;
	double worst = 0.0;
	PROBE_CUDA(cudaMalloc((void **)&device_input,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&device_weight,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&device_bias,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&device_output,count * 2u));
	ProbeFillKernel<<<64,256>>>(device_weight,dimension,3u,0u);
	ProbeFillKernel<<<64,256>>>(device_bias,dimension,4u,0u);
	PROBE_CUDA(cudaMemcpy(weight.data(),device_weight,dimension * 2u,cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(bias.data(),device_bias,dimension * 2u,cudaMemcpyDeviceToHost));
	for (launch=0u; launch<launches; launch++)
	{
		ProbeFillKernel<<<64,256>>>(device_input,count,1u,dimension);
		if ( launch == 0u )
			PROBE_CUDA(cudaMemcpy(input.data(),device_input,count * 2u,cudaMemcpyDeviceToHost));
		LmLayerNormKernel<256u,uint16_t><<<rows,256u,(dimension + 8u) * sizeof(float)>>>(device_input,device_weight,device_bias,in_place != 0u ? device_input : device_output,dimension,dimension,1.0e-6f);
		PROBE_CUDA(cudaDeviceSynchronize());
		PROBE_CUDA(cudaMemcpy(launch == 0u ? first.data() : again.data(),in_place != 0u ? device_input : device_output,count * 2u,cudaMemcpyDeviceToHost));
		if ( launch != 0u && memcmp(first.data(),again.data(),count * 2u) != 0 )
			unstable_launches++;
	}
	for (row=0u; row<rows; row++)
	{
		double total = 0.0,squares = 0.0,mean,inverse,expected,got,error,row_worst = 0.0;
		for (column=0u; column<dimension; column++)
		{
			double value = ProbeBf16(input[(uint64_t)row * dimension + column]);
			total += value;
			squares += value * value;
		}
		mean = total / dimension;
		inverse = 1.0 / sqrt(fmax(squares / dimension - mean * mean,0.0) + 1.0e-6);
		for (column=0u; column<dimension; column++)
		{
			expected = (ProbeBf16(input[(uint64_t)row * dimension + column]) - mean) * inverse * ProbeBf16(weight[column]) + ProbeBf16(bias[column]);
			got = ProbeBf16(first[(uint64_t)row * dimension + column]);
			error = fabs(got - expected) / (fabs(expected) + 1.0e-2);
			row_worst = error > row_worst ? error : row_worst;
		}
		if ( row_worst > 1.6e-2 )
			wrong_rows++;
		worst = row_worst > worst ? row_worst : worst;
	}
	printf("%s layer norm rows=%u dim=%u in_place=%u: rows off the reference %u, worst relative error %.3g, launches differing from the first %u of %u\n",
		wrong_rows == 0u && unstable_launches == 0u ? "PASS" : "FAIL",rows,dimension,in_place,wrong_rows,worst,unstable_launches,launches - 1u);
	if ( wrong_rows != 0u || unstable_launches != 0u )
		probe_failures++;
	cudaFree(device_input);
	cudaFree(device_weight);
	cudaFree(device_bias);
	cudaFree(device_output);
}

int main(void)
{
	ProbeCase(65536u,128u,1u,8u);
	ProbeCase(65536u,128u,0u,8u);
	ProbeCase(8192u,512u,1u,4u);
	ProbeCase(1024u,6144u,0u,4u);
	if ( probe_failures != 0 )
	{
		printf("FAIL layer norm: back-to-back block reductions over one shared buffer must give every warp the block's own totals\n");
		return(1);
	}
	printf("PASS layer norm: every row matches the reference and repeated launches are bit-identical\n");
	return(0);
}
