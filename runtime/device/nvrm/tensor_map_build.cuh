#pragma once

#include <stdint.h>

#include "sparkpipe/spark_device.h"

#define SPARK_TENSOR_MAP_WORDS (SPARK_DEVICE_TENSOR_MAP_BYTES / 8u)

template<int ORDINAL>
static __device__ __forceinline__ void SparkTensorMapAxis(uint64_t *map, const SparkDeviceTensorMapRequest &request)
{
	if ( ORDINAL < (int)request.rank )
	{
		asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], %1, %2;" :: "l"(map), "n"(ORDINAL), "r"((uint32_t)request.dimension[ORDINAL]) : "memory");
		asm volatile("tensormap.replace.tile.box_dim.global.b1024.b32 [%0], %1, %2;" :: "l"(map), "n"(ORDINAL), "r"(request.box[ORDINAL]) : "memory");
		asm volatile("tensormap.replace.tile.element_stride.global.b1024.b32 [%0], %1, %2;" :: "l"(map), "n"(ORDINAL), "r"(request.element_stride[ORDINAL]) : "memory");
	}
	if ( ORDINAL + 1 < (int)request.rank )
		asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], %1, %2;" :: "l"(map), "n"(ORDINAL), "l"(request.stride_bytes[ORDINAL]) : "memory");
}

static __device__ __forceinline__ void SparkTensorMapBuild(uint64_t *map, const SparkDeviceTensorMapRequest &request)
{
	for ( uint32_t word = 0u; word < SPARK_TENSOR_MAP_WORDS; ++word )
		map[word] = 0ull;
	asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;" :: "l"(map), "l"(request.address) : "memory");
	asm volatile("tensormap.replace.tile.rank.global.b1024.b32 [%0], %1;" :: "l"(map), "r"(request.rank - 1u) : "memory");
	SparkTensorMapAxis<0>(map,request);
	SparkTensorMapAxis<1>(map,request);
	SparkTensorMapAxis<2>(map,request);
	asm volatile("tensormap.replace.tile.elemtype.global.b1024.b32 [%0], 0;" :: "l"(map) : "memory");
	asm volatile("tensormap.replace.tile.interleave_layout.global.b1024.b32 [%0], 0;" :: "l"(map) : "memory");
	asm volatile("tensormap.replace.tile.fill_mode.global.b1024.b32 [%0], 0;" :: "l"(map) : "memory");
	if ( request.swizzle_bytes == 32u )
		asm volatile("tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%0], 1;" :: "l"(map) : "memory");
	else if ( request.swizzle_bytes == 64u )
		asm volatile("tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%0], 2;" :: "l"(map) : "memory");
	else if ( request.swizzle_bytes == 128u )
		asm volatile("tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%0], 3;" :: "l"(map) : "memory");
	else
		asm volatile("tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%0], 0;" :: "l"(map) : "memory");
	asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}
