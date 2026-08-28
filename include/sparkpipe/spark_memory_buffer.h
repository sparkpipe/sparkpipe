#ifndef SPARKPIPE_SPARK_MEMORY_BUFFER_H
#define SPARKPIPE_SPARK_MEMORY_BUFFER_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Memory-M1: the typed buffer handle. The lane contract used to be a
 * one-line comment ("every allocation names its space"); this makes the
 * comment a type. A SparkMemoryBuffer is {pointer, space, bytes}: every
 * allocation in template-adopting code produces one, every free consumes
 * one, and every cross-space copy goes through the space-aware copy so
 * "a host pointer" and "a device pointer" can never be confused in
 * open-coded cudaMemcpy calls again (the o_norm-overread and
 * 110GiB-OOM classes get their compile-time teeth here, and the first
 * discrete-GPU port inherits an inventory instead of an audit).
 *
 * The spaces are the inference-OS memory model's first four; FILE_BACKED
 * (mapped pack files, the M2 register/map surface) is named here so the
 * inventory is complete, and refuses at allocate until M2 lands.
 */

typedef enum SparkMemorySpace
{
	SPARK_MEMORY_SPACE_HOST_COHERENT = 1, /* plain host memory (malloc) */
	SPARK_MEMORY_SPACE_HOST_PINNED = 2,   /* page-locked host memory */
	SPARK_MEMORY_SPACE_DEVICE_PRIVATE = 3,/* device-local memory */
	SPARK_MEMORY_SPACE_FILE_BACKED = 4    /* mapped pack file (M2) */
} SparkMemorySpace;

typedef struct SparkMemoryBuffer
{
	void *pointer;
	SparkMemorySpace space;
	uint64_t bytes;
} SparkMemoryBuffer;

/* A typed view of memory this code does not own (a submission boundary,
 * a state-embedded array): names the space for copies without taking
 * ownership. Never pass a view to SparkMemoryBufferFree. */
#define SPARK_MEMORY_BUFFER_VIEW(pointer_value, space_value, bytes_value) \
	((SparkMemoryBuffer){(pointer_value), (space_value), (bytes_value)})

void SparkMemoryBufferReset(SparkMemoryBuffer *buffer);

/* Allocates and tags. HOST_COHERENT and DEVICE_PRIVATE return
 * CAPACITY_EXCEEDED on exhaustion, matching the adapters' pasted
 * allocation-failure statuses; contents are UNDEFINED (zero explicitly
 * where the pasted code used calloc). */
SparkStatus SparkMemoryBufferAllocate(
	SparkMemoryBuffer *buffer,
	SparkMemorySpace space,
	uint64_t bytes);

/* Frees and resets the handle; a NULL or zeroed handle is a no-op, and a
 * double free is impossible through the type (the handle is cleared). */
void SparkMemoryBufferFree(SparkMemoryBuffer *buffer);

/*
 * Space-aware copy: resolves the copy direction FROM THE SPACE TAGS and
 * refuses (INVALID_ARGUMENT) any pairing the model does not define, and
 * any request larger than either buffer's named bytes. execution_stream
 * non-zero selects the stream-ordered copy for device pairings; host to
 * host is always synchronous memcpy. Device-involving failures map to
 * IO_ERROR, exactly as the pasted adapter copies reported.
 */
SparkStatus SparkMemoryBufferCopy(
	SparkMemoryBuffer *destination,
	const SparkMemoryBuffer *source,
	uint64_t bytes,
	void *execution_stream);

#ifdef __cplusplus
}
#endif

#endif
