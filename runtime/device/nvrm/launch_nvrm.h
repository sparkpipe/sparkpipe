#pragma once

#include <stdint.h>

#include "runtime/device/nvrm/spark_gpu_nvrm.h"

#ifdef __CUDACC__
#include <string.h>
#include <utility>

template<typename... Parameters>
constexpr uint32_t LmParameterBytes(void)
{
	uint32_t offset = 0u;
	((offset = (offset + (uint32_t)alignof(Parameters) - 1u) / (uint32_t)alignof(Parameters) * (uint32_t)alignof(Parameters) + (uint32_t)sizeof(Parameters)), ...);
	return(offset);
}

template<typename Parameter, typename Argument>
static inline void LmPackParameter(uint8_t *packed, SparkDeviceParameter *layout, uint32_t &offset, uint32_t &index, Argument &&argument)
{
	const Parameter value = std::forward<Argument>(argument);
	offset = (offset + (uint32_t)alignof(Parameter) - 1u) / (uint32_t)alignof(Parameter) * (uint32_t)alignof(Parameter);
	memcpy(packed + offset,&value,sizeof(Parameter));
	layout[index].offset = offset;
	layout[index].bytes = (uint32_t)sizeof(Parameter);
	layout[index].align = (uint32_t)alignof(Parameter);
	offset += (uint32_t)sizeof(Parameter);
	index++;
}

template<typename... Parameters, typename... Arguments>
static SparkGpuError LmLaunchKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Arguments &&... arguments)
{
	static_assert(sizeof...(Parameters) == sizeof...(Arguments), "LM_LAUNCH passes one argument per kernel parameter");
	alignas(16) uint8_t packed[LmParameterBytes<Parameters...>() + 1u];
	SparkDeviceParameter layout[sizeof...(Parameters) + 1u];
	const uint32_t shape[6] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
	uint32_t offset = 0u,index = 0u;
	(LmPackParameter<Parameters>(packed,layout,offset,index,std::forward<Arguments>(arguments)), ...);
	return(SparkNvrmLaunchPacked((const void *)kernel,shape,shape + 3,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset));
}

#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) \
	((void)LmLaunchKernel(LM_UNPAREN kernel, dim3(grid), dim3(block), (size_t)(shared), (stream), ##__VA_ARGS__))

template<typename... Parameters, typename... Arguments>
static SparkGpuError LmLaunchDependentKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Arguments &&... arguments)
{
	static_assert(sizeof...(Parameters) == sizeof...(Arguments), "LM_LAUNCH_DEPENDENT passes one argument per kernel parameter");
	(void)grid;
	(void)block;
	(void)shared;
	(void)stream;
	((void)arguments, ...);
	return(SparkNvrmLaunchUnsupported((const void *)kernel,"a programmatic dependent launch"));
}

#define LM_LAUNCH_DEPENDENT(kernel, grid, block, shared, stream, ...) \
	LmLaunchDependentKernel(LM_UNPAREN kernel, dim3(grid), dim3(block), (size_t)(shared), (stream), ##__VA_ARGS__)

template<typename... Parameters, typename... Arguments>
static SparkGpuError LmLaunchClusterKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, uint32_t cluster,
	uint32_t dependent, Arguments &&... arguments)
{
	static_assert(sizeof...(Parameters) == sizeof...(Arguments), "LmLaunchClusterKernel passes one argument per kernel parameter");
	(void)grid;
	(void)block;
	(void)shared;
	(void)stream;
	(void)cluster;
	(void)dependent;
	((void)arguments, ...);
	return(SparkNvrmLaunchUnsupported((const void *)kernel,"a cluster launch"));
}
#endif

static inline int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	return(SparkNvrmSharedMemoryOptIn(kernel,shared_bytes) == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_ATTRIBUTE);
}
