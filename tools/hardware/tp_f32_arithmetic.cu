#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "inference/kernels/tp_reduce.cuh"

static uint16_t host_bf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float host_float(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return(result);
}

static int32_t run_case(uint16_t *input,uint16_t *output,float *sums,float *groups,float *result,uint32_t degree,uint32_t elements)
{
	LmTpF32Contributions<16> sources = {};
	uint32_t rank,group,index,seed = 1616u,blocks = (elements+255u)/256u;
	double expected;
	for (rank=0; rank<degree; rank++)
		for (index=0; index<elements; index++)
		{
			seed = (seed * 1664525u) + 1013904223u;
			input[rank*elements+index] = host_bf16(((int32_t)(seed % 16385u)-8192)/64.0f);
		}
	for (rank=0; rank<degree; rank++)
		input[rank*elements] = host_bf16(rank == 0 ? 256.0f : (rank == 1 ? 1.0f : (rank == 2 ? -256.0f : 0.0f)));
	LmTpBf16ToF32Kernel<<<(degree*elements+255u)/256u,256u>>>(sums,input,degree*elements);
	for (rank=0; rank<degree; rank+=2)
	{
		sources = {}; sources.rank[rank] = sums+rank*elements; sources.rank[rank+1] = sums+(rank+1)*elements;
		LmTpF32SumKernel<<<blocks,256u>>>(sums+rank*elements,sources,elements);
	}
	for (rank=0; rank<degree; rank+=4)
	{
		sources = {}; sources.rank[rank] = sums+rank*elements; sources.rank[rank+2] = sums+(rank+2)*elements;
		LmTpF32SumKernel<<<blocks,256u>>>(groups+(rank/4)*elements,sources,elements);
	}
	sources = {};
	for (group=0; group<degree/4; group++)
		sources.rank[group*4] = groups+group*elements;
	LmTpF32SumKernel<<<blocks,256u>>>(result,sources,elements);
	LmTpF32ToBf16Kernel<<<blocks,256u>>>(output,result,elements);
	if ( cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
		return(-1);
	for (index=0; index<elements; index++)
	{
		expected = 0.0;
		for (rank=0; rank<degree; rank++)
			expected += (double)host_float(input[rank*elements+index]);
		if ( output[index] != host_bf16((float)expected) )
			return(-2);
	}
	return(0);
}

int main(void)
{
	void *allocation = 0;
	uint16_t *input,*output;
	float *sums,*groups,*result;
	uint32_t degrees[2] = {4,16},rows[3] = {1,3,17},widths[3] = {7,257,4096},d,b,w,capacity = 17*4096,cases = 0;
	int32_t status = 0;
	if ( cudaMallocManaged(&allocation,(16u+1u)*capacity*sizeof(uint16_t)+(16u+4u+1u)*capacity*sizeof(float)) != cudaSuccess )
		return(1);
	input = (uint16_t *)allocation; output = input+16u*capacity;
	sums = (float *)(output+capacity); groups = sums+16u*capacity; result = groups+4u*capacity;
	for (d=0; d<2 && status == 0; d++)
		for (b=0; b<3 && status == 0; b++)
			for (w=0; w<3 && status == 0; w++)
			{
				status = run_case(input,output,sums,groups,result,degrees[d],rows[b]*widths[w]);
				printf("TP%u B%u width=%u status=%d\n",degrees[d],rows[b],widths[w],status);
				cases++;
			}
	if ( cudaFree(allocation) != cudaSuccess )
		return(2);
	printf("cases=%u status=%d\n",cases,status);
	return(status == 0 && cases == 18 ? 0 : 3);
}
