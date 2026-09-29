#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/norm.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define NORM_THREADS 256u

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1) void ReferenceRmsNormKernel(const uint16_t *input,const uint16_t *weight,uint16_t *output,uint32_t dimension,uint32_t row_stride,float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared,*reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float total = 0.0f,scale,value;
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		value = LmBf16ToFloat(input[base + index]);
		row[index] = value;
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		value = LmBf16ToFloat(LmFloatToBf16(row[index] * scale));
		output[base + index] = LmFloatToBf16(value * LmBf16ToFloat(weight[index]));
	}
}

static uint32_t random_state = 20260928u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits+0x7fffu+((bits>>16u)&1u))>>16u);
}

static void Case(uint32_t rows,uint32_t dimension,uint32_t row_stride,float amplitude,cudaStream_t stream)
{
	std::vector<uint16_t> input((uint64_t)rows*row_stride),weight(dimension),staged,reference;
	uint16_t *device_input,*device_weight,*device_staged,*device_reference;
	const size_t shared=(dimension+8u)*sizeof(float);
	for (auto &value : input) value=Bf16(amplitude*(((int32_t)(Random()%2049u)-1024)/1024.0f));
	for (auto &value : weight) value=Bf16(1.0f+(((int32_t)(Random()%2049u)-1024)/4096.0f));
	CUDA(cudaMalloc(&device_input,input.size()*2u)); CUDA(cudaMalloc(&device_weight,weight.size()*2u));
	CUDA(cudaMalloc(&device_staged,input.size()*2u)); CUDA(cudaMalloc(&device_reference,input.size()*2u));
	CUDA(cudaMemcpy(device_input,input.data(),input.size()*2u,cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(device_weight,weight.data(),weight.size()*2u,cudaMemcpyHostToDevice));
	CUDA(cudaMemset(device_staged,0,input.size()*2u)); CUDA(cudaMemset(device_reference,0,input.size()*2u));
	if (shared > 48u*1024u)
	{
		CUDA(cudaFuncSetAttribute((const void *)LmBf16RmsNormKernel<NORM_THREADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared));
		CUDA(cudaFuncSetAttribute((const void *)ReferenceRmsNormKernel<NORM_THREADS>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared));
	}
	LmBf16RmsNormKernel<NORM_THREADS><<<rows,NORM_THREADS,shared,stream>>>(device_input,device_weight,device_staged,dimension,row_stride,1e-5f);
	ReferenceRmsNormKernel<NORM_THREADS><<<rows,NORM_THREADS,shared,stream>>>(device_input,device_weight,device_reference,dimension,row_stride,1e-5f);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaStreamSynchronize(stream));
	staged.resize(input.size()); reference.resize(input.size());
	CUDA(cudaMemcpy(staged.data(),device_staged,staged.size()*2u,cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(reference.data(),device_reference,reference.size()*2u,cudaMemcpyDeviceToHost));
	REQUIRE(memcmp(staged.data(),reference.data(),staged.size()*2u) == 0);
	printf("PASS rows=%u dimension=%u row_stride=%u amplitude=%g\n",rows,dimension,row_stride,amplitude);
	CUDA(cudaFree(device_input)); CUDA(cudaFree(device_weight)); CUDA(cudaFree(device_staged)); CUDA(cudaFree(device_reference));
}

int main(int argc,char **argv)
{
	cudaStream_t stream;
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_rms_norm_staged --run\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	Case(1u,4096u,4096u,4.0f,stream);
	Case(8u,4096u,4096u,0.01f,stream);
	Case(3u,1536u,1536u,30.0f,stream);
	Case(2u,512u,576u,1.0f,stream);
	Case(5u,1000u,1024u,2.0f,stream);
	Case(2u,4097u,4100u,1.0f,stream);
	Case(1u,12288u,12288u,1.0f,stream);
	printf("test_rms_norm_staged PASS\n");
	return 0;
}
