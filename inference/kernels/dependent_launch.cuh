#pragma once

#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "inference/kernels/dependent_launch.cuh: programmatic dependent launch, cluster barriers and bulk L2 prefetch need sm_90 or newer; build for sm_120a or sm_121a"
#endif

static __device__ __forceinline__ void LmDependentWait(void)
{
#if defined(__CUDA_ARCH__)
	asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ void LmDependentRelease(void)
{
#if defined(__CUDA_ARCH__)
	asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ void LmClusterArrive(void)
{
#if defined(__CUDA_ARCH__)
	asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
#endif
}

static __device__ __forceinline__ void LmClusterWait(void)
{
#if defined(__CUDA_ARCH__)
	asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}

static __device__ __forceinline__ void LmClusterBarrier(void)
{
	LmClusterArrive();
	LmClusterWait();
}

static __device__ __forceinline__ void LmPrefetchL2(const void *address, uint32_t bytes)
{
#if defined(__CUDA_ARCH__)
	asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(address), "r"(bytes) : "memory");
#else
	(void)address;
	(void)bytes;
#endif
}
