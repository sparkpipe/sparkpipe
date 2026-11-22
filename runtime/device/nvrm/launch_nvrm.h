#pragma once

#include <stdint.h>

#include "runtime/device/nvrm/spark_gpu_nvrm.h"

#ifdef __CUDACC__
#include <string.h>

template<typename... Parameters>
constexpr uint32_t LmParameterBytes(void)
{
	uint32_t offset = 0u;
	((offset = (offset + (uint32_t)alignof(Parameters) - 1u) / (uint32_t)alignof(Parameters) * (uint32_t)alignof(Parameters) + (uint32_t)sizeof(Parameters)), ...);
	return(offset);
}

template<typename Parameter>
struct LmParameterOf
{
	using Type = Parameter;
};

template<typename Parameter>
static inline void LmPackParameter(uint8_t *packed, SparkDeviceParameter *layout, uint32_t &offset, uint32_t &index, const Parameter &value)
{
	offset = (offset + (uint32_t)alignof(Parameter) - 1u) / (uint32_t)alignof(Parameter) * (uint32_t)alignof(Parameter);
	memcpy(packed + offset,&value,sizeof(Parameter));
	layout[index].offset = offset;
	layout[index].bytes = (uint32_t)sizeof(Parameter);
	layout[index].align = (uint32_t)alignof(Parameter);
	offset += (uint32_t)sizeof(Parameter);
	index++;
}

template<typename... Parameters>
static SparkGpuError LmLaunchKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	typename LmParameterOf<Parameters>::Type... arguments)
{
	alignas(16) uint8_t packed[LmParameterBytes<Parameters...>() + 1u];
	SparkDeviceParameter layout[sizeof...(Parameters) + 1u];
	const uint32_t shape[6] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
	uint32_t offset = 0u,index = 0u;
	(LmPackParameter<Parameters>(packed,layout,offset,index,arguments), ...);
	return(SparkNvrmLaunchPacked((const void *)kernel,shape,shape + 3,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset));
}

#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) \
	((void)LmLaunchKernel(LM_UNPAREN kernel, dim3(grid), dim3(block), (size_t)(shared), (stream), ##__VA_ARGS__))

template<typename... Parameters>
static SparkGpuError LmLaunchDependentKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	typename LmParameterOf<Parameters>::Type... arguments)
{
	return(LmLaunchKernel(kernel,grid,block,shared,stream,arguments...));
}

#define LM_LAUNCH_DEPENDENT(kernel, grid, block, shared, stream, ...) \
	LmLaunchDependentKernel(LM_UNPAREN kernel, dim3(grid), dim3(block), (size_t)(shared), (stream), ##__VA_ARGS__)

template<typename... Parameters>
static SparkGpuError LmLaunchClusterKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, uint32_t cluster,
	uint32_t dependent, typename LmParameterOf<Parameters>::Type... arguments)
{
	alignas(16) uint8_t packed[LmParameterBytes<Parameters...>() + 1u];
	SparkDeviceParameter layout[sizeof...(Parameters) + 1u];
	const uint32_t shape[6] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
	uint32_t offset = 0u,index = 0u;
	(void)dependent;
	(LmPackParameter<Parameters>(packed,layout,offset,index,arguments), ...);
	return(SparkNvrmLaunchCluster((const void *)kernel,shape,shape + 3,cluster,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset));
}
#endif

static inline int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	return(SparkNvrmSharedMemoryOptIn(kernel,shared_bytes) == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_ATTRIBUTE);
}
