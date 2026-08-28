/*
 * Memory-M1 implementation: typed buffer handles + the space-aware copy.
 * See include/sparkpipe/spark_memory_buffer.h for the contract. The
 * device paths reach <cuda_runtime.h> the same way stage_module_common
 * does: the real header on GPU builds, the house stub on host builds,
 * so the space rules are enforced identically everywhere.
 */

#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_memory_buffer.h"

#include <cuda_runtime.h>

void SparkMemoryBufferReset(SparkMemoryBuffer *buffer)
{
	if ( buffer == 0 )
		return;
	buffer->pointer = 0;
	buffer->space = (SparkMemorySpace)0u;
	buffer->bytes = 0u;
}

SparkStatus SparkMemoryBufferAllocate(
	SparkMemoryBuffer *buffer,
	SparkMemorySpace space,
	uint64_t bytes)
{
	if ( buffer == 0 || bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkMemoryBufferReset(buffer);
	if ( space == SPARK_MEMORY_SPACE_HOST_COHERENT )
	{
		buffer->pointer = malloc((size_t)bytes);
		if ( buffer->pointer == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	else if ( space == SPARK_MEMORY_SPACE_HOST_PINNED )
	{
		if ( cudaHostAlloc(&buffer->pointer,(size_t)bytes,
			cudaHostAllocDefault) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	else if ( space == SPARK_MEMORY_SPACE_DEVICE_PRIVATE )
	{
		if ( cudaMalloc(&buffer->pointer,(size_t)bytes) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	else
	{
		/* FILE_BACKED is the M2 surface (register/map_file): named in the
		 * model, refusing here until it lands. */
		return(SPARK_STATUS_UNSUPPORTED);
	}
	buffer->space = space;
	buffer->bytes = bytes;
	return(SPARK_STATUS_OK);
}

void SparkMemoryBufferFree(SparkMemoryBuffer *buffer)
{
	if ( buffer == 0 || buffer->pointer == 0 )
		return;
	if ( buffer->space == SPARK_MEMORY_SPACE_HOST_COHERENT )
		free(buffer->pointer);
	else if ( buffer->space == SPARK_MEMORY_SPACE_HOST_PINNED ||
		buffer->space == SPARK_MEMORY_SPACE_DEVICE_PRIVATE )
		(void)cudaFree(buffer->pointer);
	SparkMemoryBufferReset(buffer);
}

static SparkStatus SparkMemoryBufferCopyDevice(
	SparkMemoryBuffer *destination,
	const SparkMemoryBuffer *source,
	uint64_t bytes,
	void *execution_stream)
{
	cudaError_t error;
	if ( execution_stream != 0 )
		error = cudaMemcpyAsync(destination->pointer,source->pointer,
			(size_t)bytes,cudaMemcpyDeviceToDevice,
			(cudaStream_t)execution_stream);
	else
		error = cudaMemcpy(destination->pointer,source->pointer,
			(size_t)bytes,cudaMemcpyDeviceToDevice);
	return(error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}

SparkStatus SparkMemoryBufferCopy(
	SparkMemoryBuffer *destination,
	const SparkMemoryBuffer *source,
	uint64_t bytes,
	void *execution_stream)
{
	cudaError_t error;
	if ( destination == 0 || source == 0 || bytes == 0u ||
		destination->pointer == 0 || source->pointer == 0 ||
		bytes > destination->bytes || bytes > source->bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( destination->space == SPARK_MEMORY_SPACE_FILE_BACKED ||
		source->space == SPARK_MEMORY_SPACE_FILE_BACKED )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( destination->space == SPARK_MEMORY_SPACE_HOST_COHERENT &&
		source->space == SPARK_MEMORY_SPACE_HOST_COHERENT )
	{
		memcpy(destination->pointer,source->pointer,(size_t)bytes);
		return(SPARK_STATUS_OK);
	}
	if ( destination->space == SPARK_MEMORY_SPACE_DEVICE_PRIVATE &&
		source->space == SPARK_MEMORY_SPACE_DEVICE_PRIVATE )
		return(SparkMemoryBufferCopyDevice(destination,source,bytes,
			execution_stream));
	if ( execution_stream != 0 )
		error = cudaMemcpyAsync(destination->pointer,source->pointer,
			(size_t)bytes,
			destination->space == SPARK_MEMORY_SPACE_DEVICE_PRIVATE ?
				cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
			(cudaStream_t)execution_stream);
	else
		error = cudaMemcpy(destination->pointer,source->pointer,
			(size_t)bytes,
			destination->space == SPARK_MEMORY_SPACE_DEVICE_PRIVATE ?
				cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
	return(error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
}
