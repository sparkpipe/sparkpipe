#ifndef SPARKPIPE_SPARK_MEMORY_BUFFER_H
#define SPARKPIPE_SPARK_MEMORY_BUFFER_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum SparkMemorySpace
{
	SPARK_MEMORY_SPACE_HOST_COHERENT = 1,
	SPARK_MEMORY_SPACE_HOST_PINNED = 2,
	SPARK_MEMORY_SPACE_DEVICE_PRIVATE = 3,
	SPARK_MEMORY_SPACE_FILE_BACKED = 4
} SparkMemorySpace;

typedef struct SparkMemoryBuffer
{
	void *pointer;
	SparkMemorySpace space;
	uint64_t bytes;
} SparkMemoryBuffer;

#define SPARK_MEMORY_BUFFER_VIEW(pointer_value, space_value, bytes_value) \
	((SparkMemoryBuffer){(pointer_value), (space_value), (bytes_value)})

void SparkMemoryBufferReset(SparkMemoryBuffer *buffer);

SparkStatus SparkMemoryBufferAllocate(
	SparkMemoryBuffer *buffer,
	SparkMemorySpace space,
	uint64_t bytes);

void SparkMemoryBufferFree(SparkMemoryBuffer *buffer);

SparkStatus SparkMemoryBufferCopy(
	SparkMemoryBuffer *destination,
	const SparkMemoryBuffer *source,
	uint64_t bytes,
	void *execution_stream);

#ifdef __cplusplus
}
#endif

#endif
