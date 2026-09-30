#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

template<uint32_t THREADS, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void ProbeReferenceNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared;
	float *reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float total = 0.0f,scale;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[base + index]);
		if ( residual_bf16 != 0 )
			value += LmBf16ToFloat(residual_bf16[base + index]);
		row[index] = value;
		total += value * value;
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[base + index] = LmFloatToBf16(value);
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] =
			LmFloatToBf16(row[index] * scale * LmScalarToFloat(weight[index]));
}

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

static void ProbeCase(uint32_t rows,uint32_t dimension,uint32_t stride,uint32_t residual,uint32_t in_place)
{
	const uint64_t count = (uint64_t)rows * stride;
	std::vector<uint16_t> output[2],residual_out[2];
	uint16_t *input,*residual_in,*weight,*out,*res_out;
	uint32_t kernel;
	int failed = probe_failures;
	char label[96];
	snprintf(label,sizeof(label),"rows%u dim%u stride%u residual%u in_place%u",rows,dimension,stride,residual,in_place);
	PROBE_CUDA(cudaMalloc((void **)&input,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&residual_in,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&out,count * 2u));
	for (kernel=0u; kernel<2u; kernel++)
	{
		ProbeFillKernel<<<64,256>>>(input,count,1u);
		ProbeFillKernel<<<64,256>>>(residual_in,count,2u);
		ProbeFillKernel<<<64,256>>>(weight,dimension,3u);
		PROBE_CUDA(cudaMemset(out,0,count * 2u));
		res_out = residual != 0u ? residual_in : 0;
		if ( kernel == 0u )
			ProbeReferenceNormKernel<256u,uint16_t><<<rows,256u,(dimension + 8u) * sizeof(float)>>>(input,residual != 0u ? residual_in : 0,weight,res_out,in_place != 0u ? input : out,dimension,stride,1.0e-5f);
		else
			LmFusedResidualRmsNormKernel<256u,uint16_t><<<rows,256u,(dimension + 8u) * sizeof(float)>>>(input,residual != 0u ? residual_in : 0,weight,res_out,in_place != 0u ? input : out,dimension,stride,1.0e-5f);
		PROBE_CUDA(cudaDeviceSynchronize());
		output[kernel].resize(count);
		residual_out[kernel].resize(count);
		PROBE_CUDA(cudaMemcpy(output[kernel].data(),in_place != 0u ? input : out,count * 2u,cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(residual_out[kernel].data(),residual_in,count * 2u,cudaMemcpyDeviceToHost));
	}
	if ( output[0] != output[1] || residual_out[0] != residual_out[1] )
	{
		printf("FAIL %s: output or residual differs from the single-pass kernel\n",label);
		probe_failures++;
	}
	printf("%s %s: normed output and residual bitwise equal to the single-pass kernel\n",failed == probe_failures ? "PASS" : "FAIL",label);
	cudaFree(input); cudaFree(residual_in); cudaFree(weight); cudaFree(out);
}

int main(void)
{
	uint32_t rows[3] = {1u,8u,16u},r;
	for (r=0u; r<3u; r++)
	{
		ProbeCase(rows[r],6144u,6144u,1u,0u);
		ProbeCase(rows[r],1536u,1536u,0u,1u);
		ProbeCase(rows[r],512u,576u,0u,1u);
		ProbeCase(rows[r],2048u,2048u,0u,0u);
	}
	ProbeCase(3u,1000u,1024u,1u,0u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS fused residual RMS norm on the device: 1..16 rows, strided and in-place rows, bitwise equal to the single-pass kernel\n");
	return 0;
}
