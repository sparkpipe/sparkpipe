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
struct LmParameterOf { using Type = Parameter; };
template<bool Enable> struct LmLaunchIf {};
template<> struct LmLaunchIf<true> { using Type = int; };

template<typename Parameter>
static inline void LmLayoutParameter(SparkDeviceParameter *layout, uint32_t &offset, uint32_t &index)
{
	offset = (offset + (uint32_t)alignof(Parameter) - 1u) / (uint32_t)alignof(Parameter) * (uint32_t)alignof(Parameter);
	layout[index].offset = offset; layout[index].bytes = (uint32_t)sizeof(Parameter); layout[index].align = (uint32_t)alignof(Parameter);
	offset += (uint32_t)sizeof(Parameter); index++;
}

template<typename Parameter>
static inline void LmPackParameter(uint8_t *packed, SparkDeviceParameter *layout, uint32_t &offset, uint32_t &index, const Parameter &value)
{
	LmLayoutParameter<Parameter>(layout,offset,index);
	memcpy(packed + layout[index - 1u].offset,&value,sizeof(Parameter));
}

template<typename... Arguments>
struct LmLaunchArgPack { static constexpr size_t kCount = sizeof...(Arguments); };
template<typename... Arguments>
LmLaunchArgPack<Arguments...> LmLaunchCountArgs(Arguments...);

template<bool Dependent, typename... Parameters>
static SparkGpuError LmLaunchDirect(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	typename LmParameterOf<Parameters>::Type... arguments)
{
	alignas(16) uint8_t packed[LmParameterBytes<Parameters...>() + 1u];
	SparkDeviceParameter layout[sizeof...(Parameters) + 1u];
	const uint32_t shape[6] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
	uint32_t offset = 0u,index = 0u;
	(LmPackParameter<Parameters>(packed,layout,offset,index,arguments), ...);
	return(Dependent ? SparkNvrmLaunchDependent((const void *)kernel,shape,shape + 3,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset)
		: SparkNvrmLaunchPacked((const void *)kernel,shape,shape + 3,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset));
}

template<bool Dependent, size_t ArgCount, typename DirectCall, typename Fallback, typename... Parameters,
	typename LmLaunchIf<(ArgCount == sizeof...(Parameters))>::Type = 0>
static SparkGpuError LmLaunchDispatch(int, DirectCall &&direct, Fallback &&, void (*kernel)(Parameters...), dim3, dim3, size_t, SparkGpuStream)
{
	return(direct(kernel));
}

template<bool Dependent, size_t ArgCount, typename DirectCall, typename Fallback, typename... Parameters,
	typename LmLaunchIf<(ArgCount < sizeof...(Parameters))>::Type = 0>
static SparkGpuError LmLaunchDispatch(long, DirectCall &&, Fallback &&fallback, void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream)
{
	alignas(16) uint8_t packed[LmParameterBytes<Parameters...>() + 1u];
	SparkDeviceParameter layout[sizeof...(Parameters) + 1u];
	const uint32_t shape[6] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
	uint32_t offset = 0u,index = 0u;
	SparkNvrmArgumentCapture capture = {packed, layout, (uint32_t)sizeof...(Parameters), 0u, 0u, 0u, {0u, 0u, 0u}, {0u, 0u, 0u}, 0u, 0, SPARK_GPU_SUCCESS},*previous;
	(LmLayoutParameter<Parameters>(layout,offset,index), ...);
	capture.bytes = offset;
	previous = SparkNvrmCaptureSet(&capture);
	fallback(0);
	(void)SparkNvrmCaptureSet(previous);
	if ( capture.captured == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return(Dependent ? SparkNvrmLaunchDependent((const void *)kernel,shape,shape + 3,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset)
		: SparkNvrmLaunchPacked((const void *)kernel,shape,shape + 3,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset));
}

template<bool Dependent, typename Fallback>
static SparkGpuError LmLaunchByPointers(dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Fallback &&fallback)
{
	SparkNvrmArgumentCapture capture = {0, 0, 0u, 0u, 0u, Dependent ? 1u : 0u, {grid.x, grid.y, grid.z}, {block.x, block.y, block.z}, shared, stream, SPARK_GPU_ERROR_INVALID_VALUE},*previous;
	previous = SparkNvrmCaptureSet(&capture);
	fallback(0);
	(void)SparkNvrmCaptureSet(previous);
	return(capture.captured != 0u ? capture.error : SPARK_GPU_ERROR_INVALID_VALUE);
}

template<typename Direct, typename Fallback>
static auto LmLaunchSelect(int, Direct &&direct, Fallback &&) -> decltype(direct(0)) { return(direct(0)); }
template<typename Direct, typename Fallback>
static SparkGpuError LmLaunchSelect(long, Direct &&, Fallback &&fallback) { return(fallback(0)); }

#define LM_LAUNCH_CALL(dependent, kernel, grid, block, shared, stream, ...) \
	([&](auto __lm_run) -> SparkGpuError { \
		const dim3 __lm_grid(grid); \
		const dim3 __lm_block(block); \
		const size_t __lm_shared = (size_t)(shared); \
		const SparkGpuStream __lm_stream = (stream); \
		auto __lm_direct = [&](auto __lm_k) { return(LmLaunchDirect<dependent>(__lm_k, __lm_grid, __lm_block, __lm_shared, __lm_stream, ##__VA_ARGS__)); }; \
		auto __lm_fb = [&](auto) { LM_UNPAREN kernel<<<dim3(1u), dim3(1u)>>>(__VA_ARGS__); }; \
		return(LmLaunchSelect(__lm_run, \
			[&](auto __lm_z) -> decltype(LmLaunchDispatch<dependent, decltype(LmLaunchCountArgs(__VA_ARGS__))::kCount>(__lm_z, __lm_direct, __lm_fb, LM_UNPAREN kernel, __lm_grid, __lm_block, __lm_shared, __lm_stream)) { \
				return(LmLaunchDispatch<dependent, decltype(LmLaunchCountArgs(__VA_ARGS__))::kCount>(__lm_z, __lm_direct, __lm_fb, LM_UNPAREN kernel, __lm_grid, __lm_block, __lm_shared, __lm_stream)); \
			}, \
			[&](auto) { return(LmLaunchByPointers<dependent>(__lm_grid, __lm_block, __lm_shared, __lm_stream, __lm_fb)); })); \
	}(0))

#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) ((void)LM_LAUNCH_CALL(false, kernel, grid, block, shared, stream, ##__VA_ARGS__))
#define LM_LAUNCH_DEPENDENT(kernel, grid, block, shared, stream, ...) LM_LAUNCH_CALL(true, kernel, grid, block, shared, stream, ##__VA_ARGS__)

template<typename... Parameters>
static SparkGpuError LmLaunchClusterKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, uint32_t cluster,
	uint32_t dependent, typename LmParameterOf<Parameters>::Type... arguments)
{
	alignas(16) uint8_t packed[LmParameterBytes<Parameters...>() + 1u];
	SparkDeviceParameter layout[sizeof...(Parameters) + 1u];
	const uint32_t shape[6] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
	uint32_t offset = 0u,index = 0u;
	(LmPackParameter<Parameters>(packed,layout,offset,index,arguments), ...);
	return(SparkNvrmLaunchCluster((const void *)kernel,shape,shape + 3,cluster,dependent,shared,stream,packed,layout,(uint32_t)sizeof...(Parameters),offset));
}
#endif

static inline int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	return(SparkNvrmSharedMemoryOptIn(kernel,shared_bytes) == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_ATTRIBUTE);
}
