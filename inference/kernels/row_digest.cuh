#pragma once

#include <stdint.h>
#include "inference/kernels/frame_error.cuh"

static __device__ __forceinline__ uint64_t LmRowDigestMix(uint64_t value)
{
	value ^= value >> 33u;
	value *= UINT64_C(0xff51afd7ed558ccd);
	value ^= value >> 33u;
	value *= UINT64_C(0xc4ceb9fe1a85ec53);
	value ^= value >> 33u;
	return(value);
}

#ifdef __CUDACC__
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmRowDigestKernel(const uint16_t *__restrict__ rows_bf16, uint32_t elements, const uint32_t *__restrict__ position_of_row, uint32_t row_count, uint64_t salt, unsigned long long *digest)
{
	__shared__ unsigned long long partial[THREADS];
	uint32_t row = blockIdx.x, index, lane, stride;
	uint64_t value = 0u, word;
	if ( row >= row_count )
		return;
	for (index = threadIdx.x; index < (elements + 3u) / 4u; index += THREADS)
	{
		word = 0u;
		for (lane = 0u; lane < 4u && index * 4u + lane < elements; lane++)
			word |= (uint64_t)rows_bf16[(uint64_t)row * elements + index * 4u + lane] << (16u * lane);
		value += LmRowDigestMix(word ^ LmRowDigestMix((salt << 40u) ^ ((uint64_t)position_of_row[row] << 16u) ^ index));
	}
	partial[threadIdx.x] = value;
	__syncthreads();
	for (stride = THREADS / 2u; stride != 0u; stride /= 2u)
	{
		if ( threadIdx.x < stride )
			partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
		(void)atomicAdd(digest, partial[0]);
}

__global__ void LmRowDigestFinishKernel(const unsigned long long *__restrict__ digest, uint32_t count, unsigned long long *__restrict__ words)
{
	uint32_t index;
	for (index = threadIdx.x; index < count; index += blockDim.x)
	{
		words[2u * index] = digest[index];
		words[2u * index + 1u] = ~digest[index];
	}
}

__global__ void LmRowDigestCheckKernel(const unsigned long long *__restrict__ words, uint32_t count, LmFrameError *error)
{
	uint32_t index;
	for (index = threadIdx.x; index < count; index += blockDim.x)
		if ( words[2u * index] != ~words[2u * index + 1u] )
			LmFrameErrorReport(error, LM_FRAME_ERROR_ROW_DIGEST_MISMATCH, 0u, index, 0u, 0u, 0u);
}

template<uint32_t THREADS>
static inline cudaError_t LmRowDigestLaunch(
	const uint16_t *rows_bf16,
	uint32_t elements,
	const uint32_t *position_of_row,
	uint32_t row_count,
	uint64_t salt,
	unsigned long long *digest,
	cudaStream_t stream)
{
	if ( rows_bf16 == 0 || position_of_row == 0 || digest == 0 || row_count == 0u || elements == 0u )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmRowDigestKernel<THREADS>), row_count, THREADS, 0, stream, rows_bf16, elements, position_of_row, row_count, salt, digest);
	return(cudaPeekAtLastError());
}

static inline cudaError_t LmRowDigestFinishLaunch(const unsigned long long *digest, uint32_t count, unsigned long long *words, cudaStream_t stream)
{
	if ( digest == 0 || words == 0 || count == 0u )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmRowDigestFinishKernel), 1, 32, 0, stream, digest, count, words);
	return(cudaPeekAtLastError());
}

static inline cudaError_t LmRowDigestCheckLaunch(const unsigned long long *words, uint32_t count, LmFrameError *error, cudaStream_t stream)
{
	if ( words == 0 || count == 0u || error == 0 )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmRowDigestCheckKernel), 1, 32, 0, stream, words, count, error);
	return(cudaPeekAtLastError());
}
#endif
