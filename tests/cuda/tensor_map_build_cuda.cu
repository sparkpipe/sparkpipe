#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/tensor_map_build.cuh"

#define BUILD_HOST 0
#define BUILD_DEVICE 1
#define BUILD_STRIDES_SHIFTED 2

static int build_mode;
static unsigned build_failures;

static __global__ void TestBuild(SparkDeviceTensorMapRequest request, uint64_t *map, int shifted)
{
	SparkTensorMapBuild(map,request);
	if ( shifted != 0 )
	{
		asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], 0, %1;" :: "l"(map), "l"(0ull) : "memory");
		asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], 1, %1;" :: "l"(map), "l"(request.stride_bytes[0]) : "memory");
		asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
	}
}

static CUresult TestTensorMapEncode(CUtensorMap *map, CUtensorMapDataType type, cuuint32_t rank, void *address, const cuuint64_t *dimension,
	const cuuint64_t *stride, const cuuint32_t *box, const cuuint32_t *element_stride, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
	CUtensorMapL2promotion promotion, CUtensorMapFloatOOBfill fill)
{
	SparkDeviceTensorMapRequest request;
	(void)promotion;
	if ( type != CU_TENSOR_MAP_DATA_TYPE_UINT8 || interleave != CU_TENSOR_MAP_INTERLEAVE_NONE || fill != CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE ||
		rank < 2u || rank > SPARK_DEVICE_TENSOR_MAP_RANK_MAX )
		return(CUDA_ERROR_NOT_SUPPORTED);
	memset(&request,0,sizeof(request));
	request.address = (SparkDeviceAddress)(uintptr_t)address;
	request.rank = rank;
	request.swizzle_bytes = swizzle == CU_TENSOR_MAP_SWIZZLE_128B ? 128u : swizzle == CU_TENSOR_MAP_SWIZZLE_64B ? 64u : swizzle == CU_TENSOR_MAP_SWIZZLE_32B ? 32u : 0u;
	for ( uint32_t axis = 0u; axis < rank; ++axis )
	{
		request.dimension[axis] = dimension[axis];
		request.box[axis] = box[axis];
		request.element_stride[axis] = element_stride[axis];
		if ( axis + 1u < rank )
			request.stride_bytes[axis] = stride[axis];
	}
	if ( build_mode == BUILD_HOST )
	{
		SparkDeviceTensorMap encoded;
		if ( SparkDeviceTensorMapEncode(&request,1u,&encoded,"tensor map test") != SPARK_STATUS_OK )
			return(CUDA_ERROR_INVALID_VALUE);
		memcpy(map,&encoded,sizeof(*map));
		return(CUDA_SUCCESS);
	}
	uint64_t *device = 0;
	if ( cudaMalloc(&device,SPARK_DEVICE_TENSOR_MAP_BYTES) != cudaSuccess )
		return(CUDA_ERROR_OUT_OF_MEMORY);
	TestBuild<<<1,1>>>(request,device,build_mode == BUILD_STRIDES_SHIFTED ? 1 : 0);
	const cudaError_t built = cudaDeviceSynchronize();
	const cudaError_t copied = built == cudaSuccess ? cudaMemcpy(map,device,SPARK_DEVICE_TENSOR_MAP_BYTES,cudaMemcpyDeviceToHost) : built;
	cudaFree(device);
	return(copied == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN);
}

#define cuTensorMapEncodeTiled TestTensorMapEncode
#include "tests/nvrm/tensor_map_gemms.cuh"
#undef cuTensorMapEncodeTiled

static void BuildRefusals(void)
{
	SparkDeviceTensorMapRequest request;
	SparkDeviceTensorMap map;
	memset(&request,0,sizeof(request));
	request.address = 4096u;
	request.rank = 2u;
	request.dimension[0] = 256u;
	request.dimension[1] = 8u;
	request.stride_bytes[0] = 256u;
	request.box[0] = 128u;
	request.box[1] = 8u;
	request.element_stride[0] = request.element_stride[1] = 1u;
	request.swizzle_bytes = 128u;
	if ( SparkDeviceTensorMapEncode(&request,1u,&map,"tensor map test") != SPARK_STATUS_OK )
	{
		printf("FAIL a valid request was refused: %s\n",SparkDeviceLastError()->text);
		build_failures++;
	}
	request.swizzle_bytes = 64u;
	if ( SparkDeviceTensorMapEncode(&request,1u,&map,"tensor map test") != SPARK_STATUS_INVALID_ARGUMENT ||
		strstr(SparkDeviceLastError()->text,"swizzle span") == 0 )
	{
		printf("FAIL a 128-byte box with a 64-byte swizzle was not refused by the device layer\n");
		build_failures++;
	}
}

int main(int argc, char **argv)
{
	SparkDeviceConfiguration configuration;
	int multiprocessors = 0;
	cudaStream_t stream = 0;
	if ( argc < 3 || (strcmp(argv[1],"host") != 0 && strcmp(argv[1],"device") != 0 && strcmp(argv[1],"strides-shifted") != 0) )
	{
		printf("usage: %s host|device|strides-shifted OUTPUT\n",argv[0]);
		return(2);
	}
	build_mode = strcmp(argv[1],"host") == 0 ? BUILD_HOST : strcmp(argv[1],"device") == 0 ? BUILD_DEVICE : BUILD_STRIDES_SHIFTED;
	memset(&configuration,0,sizeof(configuration));
	if ( (build_out = fopen(argv[2],"w")) == 0 || cudaSetDevice(0) != cudaSuccess || SparkDeviceInitialize(&configuration) != SPARK_STATUS_OK ||
		cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) != cudaSuccess )
	{
		printf("FAIL setup: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	if ( (SparkDeviceGetProperties()->properties & SPARK_DEVICE_PROPERTY_TENSOR_MAPS) == 0u )
	{
		printf("SKIP tensor map build (this device has no tensor maps)\n");
		return(0);
	}
	cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,0);
	if ( build_mode == BUILD_HOST )
		BuildRefusals();
	BuildGemms(stream,(uint32_t)multiprocessors);
	fclose(build_out);
	cudaStreamDestroy(stream);
	printf("%s tensor map build %s: %u failures\n",build_failures == 0u ? "PASS" : "FAIL",argv[1],build_failures);
	return(build_failures == 0u ? 0 : 1);
}
