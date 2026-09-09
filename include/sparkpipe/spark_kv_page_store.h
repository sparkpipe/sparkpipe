#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_PAGE_STORE_ABI_VERSION 3u
#define SPARK_KV_PAGE_STORE_PATH_BYTES 1024u
#define SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkKvPageStoreConfiguration))
#define SPARK_KV_PAGE_STORE_BYTES ((uint32_t)sizeof(SparkKvPageStore))

#define SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST 1u
#define SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE 2u

#define SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE UINT32_C(0x00000001)
#define SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS UINT32_C(0x00000002)
#define SPARK_KV_PAGE_STORE_FLAG_DIRECT_IO UINT32_C(0x00000004)
#define SPARK_KV_PAGE_STORE_KNOWN_FLAGS \
	(SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE | \
	 SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS | \
	 SPARK_KV_PAGE_STORE_FLAG_DIRECT_IO)

#define SPARK_KV_PAGE_STORE_DIRECT_IO_ALIGNMENT UINT64_C(4096)

typedef SparkStatus (*SparkKvPageStoreCopyFunction)(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes);

// A packed backing page contains one page slice from each native layer.
// The device copy callback is the hardware boundary; this layout has no CUDA
// dependency. Native layer slabs may have padding after their last page.
typedef struct SparkKvLayeredPageLayout
{
	uintptr_t device_base;
	uint64_t device_bytes;
	uint64_t layer_stride_bytes;
	uint64_t layer_page_bytes;
	uint32_t layer_count;
	uint32_t page_count;
} SparkKvLayeredPageLayout;

static inline SparkStatus SparkKvPageStoreCopyLayered(
	const SparkKvLayeredPageLayout *layout,
	uint32_t direction,
	uint32_t physical_page,
	void *host_address,
	uint64_t bytes,
	SparkKvPageStoreCopyFunction copy_function,
	void *copy_context)
{
	uint64_t layer_bytes,last_layer_offset,page_offset;
	uint32_t layer;
	SparkStatus status;
	if ( layout == 0 || host_address == 0 || copy_function == 0 || (direction != SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST && direction != SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( layout->device_base == 0u || layout->device_bytes == 0u || layout->layer_count == 0u || layout->page_count == 0u || physical_page >= layout->page_count || layout->layer_page_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( layout->layer_page_bytes > UINT64_MAX / layout->page_count || layout->layer_page_bytes > UINT64_MAX / layout->layer_count || bytes != layout->layer_page_bytes * layout->layer_count || bytes > UINTPTR_MAX - (uintptr_t)host_address )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	layer_bytes = layout->layer_page_bytes * layout->page_count;
	if ( layout->layer_stride_bytes < layer_bytes || layout->layer_stride_bytes > UINT64_MAX / layout->layer_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	last_layer_offset = (layout->layer_count - 1u) * layout->layer_stride_bytes;
	if ( last_layer_offset > layout->device_bytes || layer_bytes > layout->device_bytes - last_layer_offset || layout->device_bytes > UINTPTR_MAX - layout->device_base )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	page_offset = physical_page * layout->layer_page_bytes;
	for (layer=0u; layer<layout->layer_count; layer++)
	{
		status = copy_function(copy_context,direction,layout->device_base + layer * layout->layer_stride_bytes + page_offset,(uint8_t *)host_address + layer * layout->layer_page_bytes,layout->layer_page_bytes);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

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
	uint32_t backing_page_count;
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
// Host-worker wait: finishes queued transfers without consuming their results.
// Caller retains buffers and excludes destruction; poll the original operation
// afterward to consume its terminal status. Not callable from a copy callback.
SparkStatus SparkKvPageStoreWaitForTransfers(SparkKvPageStore *store);
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
// Repeat the identical request while BUSY; destination remains owned until a
// terminal result or store destruction. Does not change KV arena residency.
SparkStatus SparkKvPageStoreReadback(
	SparkKvPageStore *store,
	uint32_t logical_page_index,
	uint64_t generation,
	uintptr_t destination,
	uint64_t bytes);
SparkStatus SparkKvPageStoreProgress(
	SparkKvPageStore *store,
	SparkKvCacheArena *arena,
	uint32_t maximum_job_count);
// Requires a completed record of this generation; does not schedule a copy.
SparkStatus SparkKvPageStoreValidateRecord(SparkKvPageStore *store,uint32_t logical_page_index,uint64_t generation);
SparkStatus SparkKvPageStoreInvalidate(
	SparkKvPageStore *store,
	uint32_t logical_page_index,
	uint64_t generation);
// Validate both records under both worker locks before invalidating either.
SparkStatus SparkKvPageStoreInvalidatePair(
	SparkKvPageStore *first,
	SparkKvPageStore *second,
	uint32_t logical_page_index,
	uint64_t generation);

#ifdef __cplusplus
}
#endif
