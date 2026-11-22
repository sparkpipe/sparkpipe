#include "runtime/device/nvrm/tensor_map_build.cuh"

extern "C" __global__ void SparkDeviceTensorMapBuild(const SparkDeviceTensorMapRequest *requests, uint64_t *maps, uint32_t count)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		SparkTensorMapBuild(maps + (uint64_t)index * SPARK_TENSOR_MAP_WORDS,requests[index]);
}
