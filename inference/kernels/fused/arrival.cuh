#pragma once

#include <stdint.h>

static __device__ __forceinline__ bool LmFusedLastArrival(uint32_t *counter, uint32_t arrivals)
{
	__shared__ uint32_t lm_fused_last;
	__threadfence();
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		lm_fused_last = atomicAdd(counter, 1u) + 1u == arrivals ? 1u : 0u;
		if ( lm_fused_last != 0u )
			atomicExch(counter, 0u);
	}
	__syncthreads();
	if ( lm_fused_last == 0u )
		return false;
	__threadfence();
	return true;
}
