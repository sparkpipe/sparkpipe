#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_state_span.h"

#define LM_STATE_SNAPSHOT_THREADS 256u

static __global__ void LmStateSpansCopyKernel(
	const SparkStateSpan *spans,
	uint32_t span_count,
	uint8_t *snapshot,
	const uint32_t *state_index,
	uint32_t rows,
	uint32_t restore)
{
	uint32_t span = blockIdx.z,row = blockIdx.y,word,slot;
	uint4 *state,*saved;
	SparkStateSpan item;
	if ( span >= span_count || row >= rows )
		return;
	item = spans[span];
	slot = state_index[row];
	if ( slot >= item.state_rows )
		__trap();
	state = (uint4 *)(item.base + (uint64_t)slot * item.row_stride);
	saved = (uint4 *)(snapshot + item.snapshot_offset + (uint64_t)row * item.row_bytes);
	for (word=blockIdx.x * blockDim.x + threadIdx.x; word<item.row_bytes / SPARK_STATE_SPAN_ALIGN; word+=gridDim.x * blockDim.x)
	{
		if ( restore != 0u )
			state[word] = saved[word];
		else
			saved[word] = state[word];
	}
}

static inline cudaError_t LmStateSpansCopy(
	cudaStream_t stream,
	const SparkStateSpan *device_spans,
	uint32_t span_count,
	uint32_t row_words_max,
	uint8_t *snapshot,
	const uint32_t *state_index,
	uint32_t rows,
	uint32_t restore)
{
	dim3 grid;
	if ( device_spans == 0 || snapshot == 0 || state_index == 0 || span_count == 0u || rows == 0u || row_words_max == 0u || span_count > 65535u || rows > 65535u || ((uintptr_t)snapshot % SPARK_STATE_SPAN_ALIGN) != 0u )
		return(cudaErrorInvalidValue);
	grid.x = (row_words_max + LM_STATE_SNAPSHOT_THREADS - 1u) / LM_STATE_SNAPSHOT_THREADS;
	grid.y = rows;
	grid.z = span_count;
	LmStateSpansCopyKernel<<<grid,LM_STATE_SNAPSHOT_THREADS,0u,stream>>>(device_spans,span_count,snapshot,state_index,rows,restore);
	return(cudaPeekAtLastError());
}
