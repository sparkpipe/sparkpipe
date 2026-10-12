#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm_cluster.cuh"

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

static void ProbeCase(uint32_t rows,uint32_t dimension,uint32_t stride,uint32_t residual,uint32_t in_place,uint32_t residual_into_input = 0u)
{
	const uint64_t count = (uint64_t)rows * stride;
	std::vector<uint16_t> output[3],residual_out[3];
	uint16_t *input,*residual_in,*weight,*out,*res_out;
	uint32_t kernel;
	int failed = probe_failures;
	char label[96];
	snprintf(label,sizeof(label),"rows%u dim%u stride%u residual%u in_place%u residual_into_input%u",rows,dimension,stride,residual,in_place,residual_into_input);
	PROBE_CUDA(cudaMalloc((void **)&input,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&residual_in,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&out,count * 2u));
	for (kernel=0u; kernel<3u; kernel++)
	{
		ProbeFillKernel<<<64,256>>>(input,count,1u);
		ProbeFillKernel<<<64,256>>>(residual_in,count,2u);
		ProbeFillKernel<<<64,256>>>(weight,dimension,3u);
		PROBE_CUDA(cudaMemset(out,0,count * 2u));
		res_out = residual != 0u ? (residual_into_input != 0u ? input : residual_in) : 0;
		if ( kernel == 0u )
			ProbeReferenceNormKernel<256u,uint16_t><<<rows,256u,(dimension + 8u) * sizeof(float)>>>(input,residual != 0u ? residual_in : 0,weight,res_out,in_place != 0u ? input : out,dimension,stride,1.0e-5f);
		else if ( kernel == 1u )
			LmFusedResidualRmsNormKernel<256u,uint16_t><<<rows,256u,(dimension + 8u) * sizeof(float)>>>(input,residual != 0u ? residual_in : 0,weight,res_out,in_place != 0u ? input : out,dimension,stride,1.0e-5f);
		else
			PROBE_CUDA((LmFusedResidualRmsNormLaunch<256u,uint16_t>(rows,0,input,residual != 0u ? residual_in : 0,weight,res_out,in_place != 0u ? input : out,dimension,stride,1.0e-5f)));
		PROBE_CUDA(cudaDeviceSynchronize());
		output[kernel].resize(count);
		residual_out[kernel].resize(count);
		PROBE_CUDA(cudaMemcpy(output[kernel].data(),in_place != 0u ? input : out,count * 2u,cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(residual_out[kernel].data(),residual_into_input != 0u ? input : residual_in,count * 2u,cudaMemcpyDeviceToHost));
	}
	if ( output[0] != output[1] || residual_out[0] != residual_out[1] || output[0] != output[2] || residual_out[0] != residual_out[2] )
	{
		printf("FAIL %s: output or residual of the per-row kernel or the cluster launch differs from the single-pass kernel\n",label);
		probe_failures++;
	}
	printf("%s %s: normed output and residual bitwise equal to the single-pass kernel\n",failed == probe_failures ? "PASS" : "FAIL",label);
	cudaFree(input); cudaFree(residual_in); cudaFree(weight); cudaFree(out);
}

static void ProbeBf16Case(uint32_t rows,uint32_t dimension,uint32_t stride,uint32_t in_place)
{
	const uint64_t count = (uint64_t)rows * stride;
	std::vector<uint16_t> output[2];
	uint16_t *input,*weight,*out;
	uint32_t kernel;
	PROBE_CUDA(cudaMalloc((void **)&input,count * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&out,count * 2u));
	for (kernel=0u; kernel<2u; kernel++)
	{
		ProbeFillKernel<<<64,256>>>(input,count,11u + dimension);
		ProbeFillKernel<<<64,256>>>(weight,dimension,12u);
		PROBE_CUDA(cudaMemset(out,0,count * 2u));
		if ( kernel == 0u )
			LmBf16RmsNormKernel<256u><<<rows,256u,(dimension + 8u) * sizeof(float)>>>(input,weight,in_place != 0u ? input : out,dimension,stride,1.0e-5f);
		else
			PROBE_CUDA((LmBf16RmsNormLaunch<256u>(rows,0,input,weight,in_place != 0u ? input : out,dimension,stride,1.0e-5f)));
		PROBE_CUDA(cudaDeviceSynchronize());
		output[kernel].resize(count);
		PROBE_CUDA(cudaMemcpy(output[kernel].data(),in_place != 0u ? input : out,count * 2u,cudaMemcpyDeviceToHost));
	}
	if ( output[0] != output[1] )
	{
		printf("FAIL bf16 norm rows%u dim%u stride%u in_place%u: the cluster launch differs from the per-row kernel\n",rows,dimension,stride,in_place);
		probe_failures++;
	}
	else
		printf("PASS bf16 norm rows%u dim%u stride%u in_place%u: cluster launch bitwise equal to the per-row kernel\n",rows,dimension,stride,in_place);
	cudaFree(input); cudaFree(weight); cudaFree(out);
}

static void ProbeTime(void)
{
	const uint32_t dimension = 5120u,repeats = 2000u;
	uint16_t *hidden,*partial,*normed;
	float *weight,elapsed[2];
	cudaEvent_t start,stop;
	uint32_t kernel,repeat;
	PROBE_CUDA(cudaMalloc((void **)&hidden,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&partial,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&normed,dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&weight,dimension * sizeof(float)));
	PROBE_CUDA(cudaMemset(weight,0,dimension * sizeof(float)));
	ProbeFillKernel<<<64,256>>>(hidden,dimension,4u);
	ProbeFillKernel<<<64,256>>>(partial,dimension,5u);
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	for (kernel=0u; kernel<2u; kernel++)
	{
		PROBE_CUDA(cudaEventRecord(start));
		for (repeat=0u; repeat<repeats; repeat++)
			if ( kernel == 0u )
				LmFusedResidualRmsNormKernel<256u,float><<<1,256u,(dimension + 8u) * sizeof(float)>>>(hidden,partial,weight,hidden,normed,dimension,dimension,1.0e-6f);
			else
				PROBE_CUDA((LmFusedResidualRmsNormLaunch<256u,float>(1u,0,hidden,partial,weight,hidden,normed,dimension,dimension,1.0e-6f)));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed[kernel],start,stop));
	}
	printf("TIME rows1 dim%u fused residual norm: per-row kernel %.2f us, cluster launch %.2f us per call (back to back)\n",dimension,1000.0f * elapsed[0] / repeats,1000.0f * elapsed[1] / repeats);
	cudaFree(hidden); cudaFree(partial); cudaFree(normed); cudaFree(weight);
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
	for (r=1u; r<=9u; r++)
	{
		ProbeCase(r,5120u,5120u,1u,0u,1u);
		ProbeCase(r,4096u,4096u,1u,0u);
		ProbeCase(r,2816u,2816u,1u,0u,1u);
		ProbeCase(r,512u,512u,0u,1u);
	}
	for (r=1u; r<=9u; r++)
	{
		ProbeBf16Case(r,4096u,4096u,0u);
		ProbeBf16Case(r,1536u,1536u,1u);
		ProbeBf16Case(r,512u,576u,1u);
		ProbeBf16Case(r,6144u,6144u,0u);
	}
	ProbeTime();
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS fused residual RMS norm on the device: 1..16 rows, strided and in-place rows, per-row kernel and cluster launch bitwise equal to the single-pass kernel\n");
	return 0;
}
