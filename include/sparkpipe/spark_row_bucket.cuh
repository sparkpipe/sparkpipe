#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#define SPARK_ROW_BUCKET_THREADS 256u

__global__ static void SparkRowBucketPadKernel(uint32_t *first,uint32_t *second,uint32_t *third,uint32_t rows,uint32_t bucket)
{
	uint32_t row;
	row = rows + blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= bucket )
		return;
	first[row] = first[rows - 1u];
	if ( second != 0 )
		second[row] = second[rows - 1u];
	if ( third != 0 )
		third[row] = third[rows - 1u];
}

static inline cudaError_t SparkRowBucketPad(cudaStream_t stream,uint32_t *first,uint32_t *second,uint32_t *third,uint32_t rows,uint32_t bucket)
{
	if ( first == 0 || rows == 0u || bucket < rows )
		return(cudaErrorInvalidValue);
	if ( bucket == rows )
		return(cudaSuccess);
	SparkRowBucketPadKernel<<<(bucket - rows + SPARK_ROW_BUCKET_THREADS - 1u) / SPARK_ROW_BUCKET_THREADS,SPARK_ROW_BUCKET_THREADS,0,stream>>>(first,second,third,rows,bucket);
	return(cudaPeekAtLastError());
}
