#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/dtype.cuh"

#ifdef __CUDACC__
static __global__ void LmRouteLocalKernel(uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ route_global, uint32_t routes, uint32_t first_expert, uint32_t local_experts)
{
	const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
	if ( route >= routes )
		return;
	const uint32_t expert = route_expert[route];
	route_global[route] = expert;
	if ( expert >= first_expert && expert < first_expert + local_experts )
		route_expert[route] = expert - first_expert;
	else
	{
		route_expert[route] = local_experts;
		route_weight[route] = 0.0f;
	}
}

static __global__ void LmPackedRouteWeightKernel(const uint32_t *__restrict__ route_packed_row, const float *__restrict__ route_weight,
	float *__restrict__ packed_weight, uint32_t routes)
{
	const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
	if ( route < routes )
		packed_weight[route_packed_row[route]] = route_weight[route];
}

static __global__ void LmSwigluPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float g = fminf(LmBf16ToFloat(gate[(uint64_t)row * width + element]), limit);
		const float u = fminf(fmaxf(LmBf16ToFloat(up[(uint64_t)row * width + element]), -limit), limit);
		output[(uint64_t)row * width + element] = LmFloatToBf16((g / (1.0f + expf(-g))) * u * weight[row]);
	}
}

static __global__ void LmMoeLocalFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				total += LmBf16ToFloat(packed[(uint64_t)route_packed_row[route] * width + element]);
		}
		if ( shared != 0 )
			total += LmBf16ToFloat(shared[(uint64_t)row * width + element]);
		output[(uint64_t)row * width + element] = LmFloatToBf16(total);
	}
}
#endif
