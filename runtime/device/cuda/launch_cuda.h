#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#include <mutex>
#include <utility>
#include <vector>

#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) \
	LM_UNPAREN kernel<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__)

template<typename... Parameters, typename... Arguments>
static cudaError_t LmLaunchDependentKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, cudaStream_t stream, Arguments &&... arguments)
{
	cudaLaunchConfig_t config;
	cudaLaunchAttribute attribute;
	attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
	attribute.val.programmaticStreamSerializationAllowed = 1;
	config.gridDim = grid;
	config.blockDim = block;
	config.dynamicSmemBytes = shared;
	config.stream = stream;
	config.attrs = &attribute;
	config.numAttrs = 1u;
	return(cudaLaunchKernelEx(&config,kernel,std::forward<Arguments>(arguments)...));
}

#define LM_LAUNCH_DEPENDENT(kernel, grid, block, shared, stream, ...) \
	LmLaunchDependentKernel(LM_UNPAREN kernel, dim3(grid), dim3(block), (size_t)(shared), (stream), __VA_ARGS__)

template<typename... Parameters, typename... Arguments>
static cudaError_t LmLaunchClusterKernel(void (*kernel)(Parameters...), dim3 grid, dim3 block, size_t shared, cudaStream_t stream, uint32_t cluster, uint32_t dependent, Arguments &&... arguments)
{
	cudaLaunchConfig_t config;
	cudaLaunchAttribute attributes[2];
	attributes[0].id = cudaLaunchAttributeClusterDimension;
	attributes[0].val.clusterDim.x = cluster;
	attributes[0].val.clusterDim.y = 1u;
	attributes[0].val.clusterDim.z = 1u;
	attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
	attributes[1].val.programmaticStreamSerializationAllowed = 1;
	config.gridDim = grid;
	config.blockDim = block;
	config.dynamicSmemBytes = shared;
	config.stream = stream;
	config.attrs = attributes;
	config.numAttrs = dependent != 0u ? 2u : 1u;
	return(cudaLaunchKernelEx(&config,kernel,std::forward<Arguments>(arguments)...));
}

typedef struct LmKernelOptInEntry
{
	const void *kernel;
	uint64_t device_mask;
}
LmKernelOptInEntry;

static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	static std::mutex grant_mutex;
	static LmKernelOptInEntry grants[LM_LAUNCH_MAX_OPTIN_KERNELS];
	static uint32_t grant_count = 0u;
	cudaError_t status;
	uint64_t device_bit;
	uint32_t index;
	int device_index;

	if ( kernel == 0 )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	status = cudaGetDevice(&device_index);
	if ( status != cudaSuccess )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( device_index < 0
		|| (uint32_t)device_index >= LM_LAUNCH_MAX_TRACKED_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	device_bit = UINT64_C(1) << (uint32_t)device_index;
	{
		std::lock_guard<std::mutex> lock(grant_mutex);
		for ( index = 0u; index < grant_count; index++ )
		{
			if ( grants[index].kernel == kernel )
			{
				if ( (grants[index].device_mask & device_bit) != 0u )
					return(LM_LAUNCH_OK);
				break;
			}
		}
		if ( index == grant_count )
		{
			if ( grant_count >= LM_LAUNCH_MAX_OPTIN_KERNELS )
				return(LM_LAUNCH_ERR_ATTRIBUTE);
			grants[grant_count].kernel = kernel;
			grants[grant_count].device_mask = 0u;
			grant_count++;
		}
		status = cudaFuncSetAttribute(kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared_bytes);
		if ( status == cudaSuccess )
			grants[index].device_mask |= device_bit;
	}
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_ATTRIBUTE);
}

template<typename BundleTag, typename... Args>
static inline cudaError_t LmLaunchEmbedded(const uint8_t *cubin, uint64_t cubin_bytes, const char *manifest, uint64_t manifest_bytes,
	const char *const *names, uint32_t count, uint32_t id, dim3 grid, dim3 block, size_t shared, cudaStream_t stream, Args... args)
{
	(void)cubin_bytes; (void)manifest; (void)manifest_bytes;
	static std::mutex lock;
	static CUcontext cached_ctx = 0;
	static CUmodule mod = 0;
	static std::vector<CUfunction> fns;
	static std::vector<uint32_t> max_dyn;
	CUcontext ctx = 0;
	if ( id >= count )
		return cudaErrorInvalidValue;
	if ( (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == 0) && (cudaFree(0) != cudaSuccess || cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == 0) )
		return cudaErrorInvalidValue;
	CUfunction fn = 0;
	{
		std::lock_guard<std::mutex> guard(lock);
		if ( ctx != cached_ctx || mod == 0 )
		{
			if ( cuModuleLoadData(&mod, cubin) != CUDA_SUCCESS )
				return cudaErrorInvalidValue;
			fns.assign(count, 0);
			max_dyn.assign(count, 49152u);
			for ( uint32_t i = 0u; i < count; ++i )
				if ( cuModuleGetFunction(&fns[i], mod, names[i]) != CUDA_SUCCESS )
					return cudaErrorInvalidDeviceFunction;
			cached_ctx = ctx;
		}
		fn = fns[id];
		if ( shared > max_dyn[id] )
		{
			if ( cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shared) != CUDA_SUCCESS )
				return cudaErrorInvalidValue;
			max_dyn[id] = (uint32_t)shared;
		}
	}
	void *ptrs[] = { (void *)&args... };
	CUlaunchAttribute attr = {};
	attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
	attr.value.programmaticStreamSerializationAllowed = 1;
	CUlaunchConfig cfg = { grid.x, grid.y, grid.z, block.x, block.y, block.z, (unsigned int)shared, (CUstream)stream, &attr, 1u };
	return cuLaunchKernelEx(&cfg, fn, ptrs, 0) == CUDA_SUCCESS ? cudaSuccess : cudaErrorLaunchFailure;
}
#else
static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	(void)kernel;
	(void)shared_bytes;
	return(LM_LAUNCH_OK);
}
#endif
