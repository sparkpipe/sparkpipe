#pragma once

#include "dtype.cuh"

template<uint32_t MaxRanks>
struct LmTpF32Contributions
{
	const float *rank[MaxRanks];
};

static __global__ void LmTpBf16ToF32Kernel(float *destination,const uint16_t *source,uint32_t elements)
{
	uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index < elements )
		destination[index] = LmBf16ToFloat(source[index]);
}

static __global__ void LmTpF32ToBf16Kernel(uint16_t *destination,const float *source,uint32_t elements)
{
	uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index < elements )
		destination[index] = LmFloatToBf16(source[index]);
}

template<uint32_t MaxRanks>
__global__ void LmTpF32SumKernel(float *destination,LmTpF32Contributions<MaxRanks> inputs,uint32_t elements)
{
	uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x,rank;
	float value = 0.0f;
	if ( index >= elements )
		return;
	for (rank=0u; rank<MaxRanks; rank++)
		if ( inputs.rank[rank] != 0 )
			value = value + inputs.rank[rank][index];
	destination[index] = value;
}

template<uint32_t MaxRanks>
struct LmTpBf16Contributions
{
	const uint16_t *rank[MaxRanks];
};

template<uint32_t MaxRanks>
__global__ void LmTpBf16SumKernel(uint16_t *destination,LmTpBf16Contributions<MaxRanks> inputs,uint32_t rows,uint32_t width)
{
	uint32_t row = blockIdx.x,element,rank;
	uint64_t offset;
	float value;
	if ( row >= rows )
		return;
	for (element=threadIdx.x; element<width; element+=blockDim.x)
	{
		offset = ((uint64_t)row * width) + element;
		value = 0.0f;
		for (rank=0u; rank<MaxRanks; rank++)
			if ( inputs.rank[rank] != 0 )
				value = value + LmBf16ToFloat(inputs.rank[rank][offset]);
		destination[offset] = LmFloatToBf16(value);
	}
}
