#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_DEVICE_COPY_MAX_REGIONS 2u
#define SPARK_KV_DEVICE_COPY_PAGE_MAJOR 1u
#define SPARK_KV_DEVICE_COPY_LAYER_MAJOR 2u

typedef struct SparkKvDeviceCopyRegion
{
	uint32_t layout;
	uint32_t layer_count;
	uintptr_t device_base;
	uint64_t layer_page_bytes;
	uint64_t layer_stride_bytes;
} SparkKvDeviceCopyRegion;

typedef struct SparkKvDeviceCopyConfiguration
{
	const char *module_tag;
	void *stream;
	SparkKvCacheArena *arena;
	uint32_t physical_page_count;
	uint32_t region_count;
	SparkKvDeviceCopyRegion regions[SPARK_KV_DEVICE_COPY_MAX_REGIONS];
	uint32_t pending_capacity;
	uint32_t reserved0;
} SparkKvDeviceCopyConfiguration;

typedef struct SparkKvDeviceCopyPending
{
	void *event;
	uint32_t source_logical_page;
	uint32_t destination_logical_page;
	uint32_t free_destination;
	uint32_t active;
} SparkKvDeviceCopyPending;

typedef struct SparkKvDeviceCopier
{
	const char *module_tag;
	void *stream;
	void *fence_event;
	SparkKvCacheArena *arena;
	uint32_t physical_page_count;
	uint32_t region_count;
	SparkKvDeviceCopyRegion regions[SPARK_KV_DEVICE_COPY_MAX_REGIONS];
	uint32_t pending_capacity;
	uint32_t pending_count;
	uint32_t full_logged;
	uint32_t reserved0;
	SparkKvDeviceCopyPending *pending;
	uint64_t copy_count;
	uint64_t copy_bytes;
	uint64_t retire_count;
	uint64_t deferred_free_count;
	uint64_t busy_count;
} SparkKvDeviceCopier;

SparkStatus SparkKvDeviceCopierInitialize(SparkKvDeviceCopier *copier, const SparkKvDeviceCopyConfiguration *configuration);
SparkStatus SparkKvDeviceCopierCopyPage(void *context, uint32_t source_logical_page, uint32_t destination_logical_page);
SparkStatus SparkKvDeviceCopierRetire(void *context, uint32_t require_all);
uint32_t SparkKvDeviceCopierDestinationPins(void *context, uint32_t logical_page);
SparkStatus SparkKvDeviceCopierDeferFree(void *context, uint32_t logical_page);
SparkStatus SparkKvDeviceCopierFence(SparkKvDeviceCopier *copier, void *stream);
SparkStatus SparkKvDeviceCopierDrain(SparkKvDeviceCopier *copier);
void SparkKvDeviceCopierDestroy(SparkKvDeviceCopier *copier);

#ifdef __cplusplus
}
#endif
