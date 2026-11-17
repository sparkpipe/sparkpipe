#pragma once

#include "dtype.cuh"

template<uint32_t MaxRanks>
struct LmTpF32Contributions
{
	const float *rank[MaxRanks];
};

static __global__ void LmTpBf16ToF32Kernel(float *destination,const uint16_t *source,uint32_t elements)
{
#if defined(__CUDA_ARCH__)
	if ( (((uintptr_t)destination & 15u) == 0u) && (((uintptr_t)source & 7u) == 0u) && (elements & 3u) == 0u )
	{
		const uint32_t chunk = (blockIdx.x * blockDim.x) + threadIdx.x, chunks = elements >> 2u;
		if ( chunk < chunks )
		{
			const uint2 packed = ((const uint2 *)source)[chunk];
			((float4 *)destination)[chunk] = make_float4(
				__uint_as_float(packed.x << 16u), __uint_as_float(packed.x & 0xffff0000u),
				__uint_as_float(packed.y << 16u), __uint_as_float(packed.y & 0xffff0000u));
		}
		return;
	}
#endif
	uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index < elements )
		destination[index] = LmBf16ToFloat(source[index]);
}

static __global__ void LmTpF32ToBf16Kernel(uint16_t *destination,const float *source,uint32_t elements)
{
#if defined(__CUDA_ARCH__)
	if ( (((uintptr_t)destination | (uintptr_t)source) & 15u) == 0u && (elements & 7u) == 0u )
	{
		const uint32_t chunk = (blockIdx.x * blockDim.x) + threadIdx.x, chunks = elements >> 3u;
		if ( chunk < chunks )
		{
			const float4 *in4 = (const float4 *)source;
			const float4 lo = in4[chunk * 2u], hi = in4[chunk * 2u + 1u];
			((uint4 *)destination)[chunk] = make_uint4(
				(uint32_t)LmFloatToBf16(lo.x) | ((uint32_t)LmFloatToBf16(lo.y) << 16u),
				(uint32_t)LmFloatToBf16(lo.z) | ((uint32_t)LmFloatToBf16(lo.w) << 16u),
				(uint32_t)LmFloatToBf16(hi.x) | ((uint32_t)LmFloatToBf16(hi.y) << 16u),
				(uint32_t)LmFloatToBf16(hi.z) | ((uint32_t)LmFloatToBf16(hi.w) << 16u));
		}
		return;
	}
#endif
	uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index < elements )
		destination[index] = LmFloatToBf16(source[index]);
}

static __global__ void LmTpAccumulateBf16Kernel(float *accumulator,const uint16_t *value,uint64_t elements,uint32_t first)
{
#if defined(__CUDA_ARCH__)
	if ( (((uintptr_t)accumulator & 15u) == 0u) && (((uintptr_t)value & 7u) == 0u) && (elements & 3u) == 0u )
	{
		const uint64_t chunk = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x, chunks = elements >> 2u;
		if ( chunk < chunks )
		{
			const uint2 packed = ((const uint2 *)value)[chunk];
			float4 *acc4 = (float4 *)accumulator;
			const float4 v = make_float4(
				__uint_as_float(packed.x << 16u), __uint_as_float(packed.x & 0xffff0000u),
				__uint_as_float(packed.y << 16u), __uint_as_float(packed.y & 0xffff0000u));
			if ( first != 0u )
				acc4[chunk] = v;
			else
			{
				const float4 c = acc4[chunk];
				acc4[chunk] = make_float4(c.x + v.x, c.y + v.y, c.z + v.z, c.w + v.w);
			}
		}
		return;
	}
#endif
	const uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index < elements )
		accumulator[index] = (first != 0u ? 0.0f : accumulator[index]) + LmBf16ToFloat(value[index]);
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
