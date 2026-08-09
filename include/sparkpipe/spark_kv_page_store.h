#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_PAGE_STORE_ABI_VERSION 2u
#define SPARK_KV_PAGE_STORE_PATH_BYTES 1024u
#define SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkKvPageStoreConfiguration))
#define SPARK_KV_PAGE_STORE_BYTES ((uint32_t)sizeof(SparkKvPageStore))

#define SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST 1u
#define SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE 2u

#define SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE UINT32_C(0x00000001)
#define SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS UINT32_C(0x00000002)
#define SPARK_KV_PAGE_STORE_KNOWN_FLAGS \
	(SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE | \
	 SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS)

/*
 * Device-neutral backing for opaque fixed-size KV pages. The cache decides
 * what to evict and when to fetch it; a model driver supplies only the copy
 * primitive needed to move its physical page between device and host memory.
 */
typedef SparkStatus (*SparkKvPageStoreCopyFunction)(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes);

typedef struct SparkKvPageStoreConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t logical_page_capacity;
	uint32_t transfer_capacity;
	uint32_t reserved0;
	uint64_t page_bytes;
	uint64_t maximum_backing_bytes;
	const char *backing_path;
	void *staging_address;
	uint64_t staging_bytes;
	SparkKvPageStoreCopyFunction copy_function;
	void *copy_context;
}
SparkKvPageStoreConfiguration;

typedef struct SparkKvPageStore
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t logical_page_capacity;
	uint32_t transfer_capacity;
	uint32_t reserved0;
	uint64_t page_bytes;
	uint64_t maximum_backing_bytes;
	int32_t file_descriptor;
	uint32_t reserved1;
	void *staging_address;
	uint64_t staging_bytes;
	SparkKvPageStoreCopyFunction copy_function;
	void *copy_context;
	void *worker_state;
	uint64_t *generations;
	uint8_t *valid_pages;
	uint64_t write_count;
	uint64_t read_count;
	uint64_t write_bytes;
	uint64_t read_bytes;
}
SparkKvPageStore;

SparkStatus SparkKvPageStoreInitialize(
	SparkKvPageStore *store,
	const SparkKvPageStoreConfiguration *configuration);
SparkStatus SparkKvPageStoreBuildPath(
	char *path,
	uint32_t path_capacity,
	const char *backing_directory,
	const char *model_id,
	const char *model_revision,
	const char *node_id,
	uint32_t stage_index);
void SparkKvPageStoreDestroy(SparkKvPageStore *store);
SparkStatus SparkKvPageStoreWriteback(
	void *context,
	uint32_t logical_page_index,
	uint32_t physical_page_index,
	uint64_t generation,
	uintptr_t key_device_address,
	uint64_t key_bytes,
	uintptr_t value_device_address,
	uint64_t value_bytes);
SparkStatus SparkKvPageStorePrefetch(
	SparkKvPageStore *store,
	SparkKvCacheArena *arena,
	uint32_t logical_page_index);

#ifdef __cplusplus
}
#endif
