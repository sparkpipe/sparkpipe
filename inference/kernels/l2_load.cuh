#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#define LM_L2_LOAD_RANGES 8u
#define LM_L2_LOAD_THREADS 256u

typedef struct LmL2LoadPlan
{
	const uint8_t *base[LM_L2_LOAD_RANGES];
	uint32_t bytes[LM_L2_LOAD_RANGES];
	uint32_t count;
	uint32_t total;
	uint32_t cap;
} LmL2LoadPlan;

static __device__ uint32_t lm_l2_load_sink;

static __global__ void __launch_bounds__(LM_L2_LOAD_THREADS) LmL2LoadKernel(const __grid_constant__ LmL2LoadPlan plan)
{
	const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
	uint32_t range, index, count, folded = 0u;
	uint4 value;
	for (range = 0u; range < plan.count; range++)
	{
		count = plan.bytes[range] / 16u;
		for (index = first; index < count; index += stride)
		{
			value = __ldcg((const uint4 *)plan.base[range] + index);
			folded ^= value.x ^ value.y ^ value.z ^ value.w;
		}
	}
	if ( folded == 0x9e3779b9u )
		lm_l2_load_sink = folded;
}

static inline void LmL2LoadBegin(LmL2LoadPlan *plan, uint32_t cap)
{
	uint32_t range;
	for (range = 0u; range < LM_L2_LOAD_RANGES; range++)
	{
		plan->base[range] = 0;
		plan->bytes[range] = 0u;
	}
	plan->count = 0u;
	plan->total = 0u;
	plan->cap = cap;
}

static inline void LmL2LoadAdd(LmL2LoadPlan *plan, const void *base, uint64_t bytes)
{
	uint64_t room;
	if ( base == 0 || ((uintptr_t)base % 16u) != 0u || plan->count == LM_L2_LOAD_RANGES || plan->total >= plan->cap )
		return;
	room = plan->cap - plan->total;
	bytes = (bytes < room ? bytes : room) & ~(uint64_t)15u;
	if ( bytes == 0u )
		return;
	plan->base[plan->count] = (const uint8_t *)base;
	plan->bytes[plan->count] = (uint32_t)bytes;
	plan->count++;
	plan->total += (uint32_t)bytes;
}

static inline cudaError_t LmL2LoadLaunch(const LmL2LoadPlan *plan, uint32_t blocks, cudaStream_t stream)
{
	if ( plan->count == 0u )
		return(cudaSuccess);
	LmL2LoadKernel<<<blocks,LM_L2_LOAD_THREADS,0u,stream>>>(*plan);
	return(cudaPeekAtLastError());
}
