#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Model-neutral row compaction for heads and other sparse row consumers.
 * Source and destination maps are selected on the host; these kernels only
 * move rows and publish compact scalar results without model policy.
 */

static __global__ void SparkGatherBf16RowsKernel(
	const uint16_t *__restrict__ source_bf16,
	const uint32_t *__restrict__ source_row_indices,
	uint16_t *__restrict__ destination_bf16,
	uint32_t row_count,
	uint32_t row_width)
{
	uint32_t column,row;
	row = blockIdx.y;
	column = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count && column < row_width )
		destination_bf16[(uint64_t)row * row_width + column] = source_bf16[(uint64_t)source_row_indices[row] * row_width + column];
}

static __global__ void SparkScatterU32RowsKernel(
	const uint32_t *__restrict__ source,
	const uint32_t *__restrict__ destination_indices,
	uint32_t *__restrict__ destination,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		destination[destination_indices[row]] = source[row];
}

static inline cudaError_t SparkLaunchGatherBf16Rows(
	cudaStream_t stream,
	const void *source_bf16,
	const uint32_t *source_row_indices,
	void *destination_bf16,
	uint32_t row_count,
	uint32_t row_width)
{
	dim3 grid;
	if ( source_bf16 == 0 || source_row_indices == 0 || destination_bf16 == 0 || source_bf16 == destination_bf16 || row_count == 0u || row_width == 0u )
		return(cudaErrorInvalidValue);
	grid = dim3((row_width + 255u) / 256u,row_count);
	SparkGatherBf16RowsKernel<<<grid,256u,0,stream>>>((const uint16_t *)source_bf16,source_row_indices,(uint16_t *)destination_bf16,row_count,row_width);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLaunchScatterU32Rows(
	cudaStream_t stream,
	const uint32_t *source,
	const uint32_t *destination_indices,
	uint32_t *destination,
	uint32_t row_count)
{
	if ( source == 0 || destination_indices == 0 || destination == 0 || source == destination || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkScatterU32RowsKernel<<<(row_count + 255u) / 256u,256u,0,stream>>>(source,destination_indices,destination,row_count);
	return(cudaGetLastError());
}
