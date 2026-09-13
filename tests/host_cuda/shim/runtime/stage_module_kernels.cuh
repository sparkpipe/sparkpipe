#pragma once

// Host-syntax shim for runtime/stage_module_kernels.cuh. The real header
// defines __global__ kernels plus inline launchers whose bodies use launch
// syntax, which is not C++. This mirror keeps the two extern "C" launcher
// signatures EXACTLY (they are what family TUs - here the glm52 stage's
// cuda.cu - compile against) and drops only the kernel bodies and the
// launches themselves; the paired declarations header
// runtime/stage_module_kernels.h stays the neutral contract.
//
// Kept in lockstep by hand: a signature drift between the real header and
// this shim fails the TU compile here instead of hiding behind the shim.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef SPARK_HOST_SYNTAX_STAGE_KERNELS_SHIM
#define SPARK_HOST_SYNTAX_STAGE_KERNELS_SHIM

#ifdef __cplusplus
extern "C" {
#endif

inline cudaError_t SparkStageLaunchAccumAddInline(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	(void)stream;(void)destination_bf16;(void)source_bf16;(void)row_count;(void)width;
	return(cudaErrorInvalidValue);
}

inline cudaError_t SparkStageLaunchAccumU64MaxInline(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;(void)destination;(void)source;(void)element_count;
	return(cudaErrorInvalidValue);
}

#ifdef __cplusplus
}
#endif

#endif /* SPARK_HOST_SYNTAX_STAGE_KERNELS_SHIM */
