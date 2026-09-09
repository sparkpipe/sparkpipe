#pragma once

#include "dtype.cuh"

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
