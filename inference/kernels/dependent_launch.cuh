#pragma once

#include <stdint.h>

static __device__ __forceinline__ void LmDependentWait(void)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ void LmDependentRelease(void)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ void LmPrefetchL2(const void *address, uint32_t bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(address), "r"(bytes) : "memory");
#else
	(void)address;
	(void)bytes;
#endif
}
