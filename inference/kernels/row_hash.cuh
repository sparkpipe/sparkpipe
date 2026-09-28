#pragma once

#include "inference/kernels/splitmix.cuh"
#include "runtime/launch.h"
#include <cuda_runtime.h>
#include <stdint.h>

#define LM_ROW_HASH_THREADS 256u
#define LM_ROW_HASH_SITE_LIMIT 256u

static __host__ __device__ __forceinline__ uint64_t LmRowHashWord(uint32_t site, uint64_t index, uint16_t word)
{
	return(LmSplitMix64(((uint64_t)site << 56u) ^ (index << 16u) ^ (uint64_t)word));
}

static __global__ __launch_bounds__(LM_ROW_HASH_THREADS) void LmRowHashKernel(const uint8_t *data, uint64_t row_stride_bytes, uint32_t row_words, uint32_t site, uint64_t *hashes)
{
	__shared__ uint64_t partial[LM_ROW_HASH_THREADS];
	const uint16_t *row = (const uint16_t *)(data + (uint64_t)blockIdx.x * row_stride_bytes);
	uint64_t sum;
	uint32_t index, stride;
	sum = 0u;
	for ( index = threadIdx.x; index < row_words; index += LM_ROW_HASH_THREADS )
		sum += LmRowHashWord(site,index,row[index]);
	partial[threadIdx.x] = sum;
	__syncthreads();
	for ( stride = LM_ROW_HASH_THREADS / 2u; stride > 0u; stride >>= 1u )
	{
		if ( threadIdx.x < stride )
			partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
		hashes[blockIdx.x] = partial[0];
}

static int32_t LmRowHashLaunch(const void *data, uint64_t row_stride_bytes, uint32_t row_words, uint32_t rows, uint32_t site, uint64_t *hashes, cudaStream_t stream)
{
	if ( data == 0 || hashes == 0 || rows == 0u || rows > (uint32_t)INT32_MAX || row_words == 0u || site >= LM_ROW_HASH_SITE_LIMIT )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( ((uintptr_t)data % sizeof(uint16_t)) != 0u || (row_stride_bytes % sizeof(uint16_t)) != 0u || (rows > 1u && row_stride_bytes < (uint64_t)row_words * sizeof(uint16_t)) )
		return(LM_LAUNCH_ERR_SHAPE);
	LM_LAUNCH((LmRowHashKernel),rows,LM_ROW_HASH_THREADS,0u,stream,(const uint8_t *)data,row_stride_bytes,row_words,site,hashes);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
