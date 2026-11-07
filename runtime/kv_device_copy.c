#include "sparkpipe/spark_kv_device_copy.h"
#include "sparkpipe/spark_error_site.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t SparkKvDeviceCopyRegionIsValid(const SparkKvDeviceCopyRegion *region, uint32_t physical_page_count)
{
	if ( (region->layout != SPARK_KV_DEVICE_COPY_PAGE_MAJOR && region->layout != SPARK_KV_DEVICE_COPY_LAYER_MAJOR) ||
		region->layer_count == 0u || region->layer_page_bytes == 0u || region->base.buffer == 0 )
		return(0u);
	if ( region->layer_page_bytes > UINT64_MAX / region->layer_count )
		return(0u);
	if ( region->layout == SPARK_KV_DEVICE_COPY_LAYER_MAJOR &&
		(region->layer_stride_bytes < region->layer_page_bytes * physical_page_count ||
		 region->layer_stride_bytes > UINT64_MAX / region->layer_count) )
		return(0u);
	return(1u);
}

static void SparkKvDeviceCopierReleaseEvents(SparkKvDeviceCopier *copier)
{
	uint32_t index;
	if ( copier->pending != 0 )
		for (index=0u; index<copier->pending_capacity; index++)
			if ( copier->pending[index].event != 0 )
				(void)SparkDeviceEventDestroy(copier->pending[index].event);
	if ( copier->fence_event != 0 )
		(void)SparkDeviceEventDestroy(copier->fence_event);
	free(copier->pending);
}

SparkStatus SparkKvDeviceCopierInitialize(SparkKvDeviceCopier *copier, const SparkKvDeviceCopyConfiguration *configuration)
{
	uint32_t index;
	if ( copier == 0 || configuration == 0 || configuration->queue == 0 || configuration->arena == 0 ||
		configuration->region_count == 0u || configuration->region_count > SPARK_KV_DEVICE_COPY_MAX_REGIONS ||
		configuration->pending_capacity == 0u || configuration->physical_page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<configuration->region_count; index++)
		if ( SparkKvDeviceCopyRegionIsValid(&configuration->regions[index],configuration->physical_page_count) == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(copier,0,sizeof(*copier));
	copier->module_tag = configuration->module_tag != 0 ? configuration->module_tag : "kv";
	copier->queue = configuration->queue;
	copier->arena = configuration->arena;
	copier->physical_page_count = configuration->physical_page_count;
	copier->region_count = configuration->region_count;
	memcpy(copier->regions,configuration->regions,sizeof(copier->regions));
	copier->pending_capacity = configuration->pending_capacity;
	copier->pending = (SparkKvDeviceCopyPending *)calloc(copier->pending_capacity,sizeof(*copier->pending));
	if ( copier->pending == 0 )
	{
		memset(copier,0,sizeof(*copier));
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (index=0u; index<=copier->pending_capacity; index++)
		if ( SparkDeviceEventCreate(index < copier->pending_capacity ? &copier->pending[index].event : &copier->fence_event) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"%s kv copy-on-write event create failed device=%s\n",copier->module_tag,SparkDeviceLastError()->text);
			SparkKvDeviceCopierReleaseEvents(copier);
			memset(copier,0,sizeof(*copier));
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvDeviceCopierBlockIsPinned(const SparkKvDeviceCopier *copier, uint32_t logical_page)
{
	const SparkKvCacheBlock *block;
	if ( logical_page >= copier->arena->logical_block_count )
		return(0u);
	block = &copier->arena->blocks[logical_page];
	return((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u && block->residency_reference_count >= 1u ? 1u : 0u);
}

static SparkDeviceRange SparkKvDeviceCopierRange(const SparkKvDeviceCopyRegion *region, uint64_t offset)
{
	SparkDeviceRange range = region->base;
	range.offset += offset;
	return(range);
}

static SparkStatus SparkKvDeviceCopierEnqueue(SparkKvDeviceCopier *copier, uint32_t source_slot, uint32_t destination_slot, uint64_t *bytes)
{
	const SparkKvDeviceCopyRegion *region;
	SparkStatus status;
	uint64_t page_bytes;
	uint32_t index,layer;
	*bytes = 0u;
	status = SPARK_STATUS_OK;
	for (index=0u; status == SPARK_STATUS_OK && index<copier->region_count; index++)
	{
		region = &copier->regions[index];
		page_bytes = region->layer_page_bytes * region->layer_count;
		if ( region->layout == SPARK_KV_DEVICE_COPY_PAGE_MAJOR )
			status = SparkDeviceCopy(copier->queue,SparkKvDeviceCopierRange(region,(uint64_t)destination_slot * page_bytes),
				SparkKvDeviceCopierRange(region,(uint64_t)source_slot * page_bytes),page_bytes);
		else
			for (layer=0u; status == SPARK_STATUS_OK && layer<region->layer_count; layer++)
				status = SparkDeviceCopy(copier->queue,
					SparkKvDeviceCopierRange(region,layer * region->layer_stride_bytes + (uint64_t)destination_slot * region->layer_page_bytes),
					SparkKvDeviceCopierRange(region,layer * region->layer_stride_bytes + (uint64_t)source_slot * region->layer_page_bytes),
					region->layer_page_bytes);
		*bytes += page_bytes;
	}
	return(status);
}

SparkStatus SparkKvDeviceCopierCopyPage(void *context, uint32_t source_logical_page, uint32_t destination_logical_page)
{
	SparkKvDeviceCopier *copier;
	SparkKvDeviceCopyPending *entry;
	SparkKvCacheBlockView source,destination;
	uint64_t bytes;
	uint32_t index;
	SparkStatus status;
	copier = (SparkKvDeviceCopier *)context;
	if ( copier == 0 || copier->pending == 0 ||
		SparkKvCacheArenaResolveBlock(copier->arena,source_logical_page,&source) != SPARK_STATUS_OK ||
		SparkKvCacheArenaResolveBlock(copier->arena,destination_logical_page,&destination) != SPARK_STATUS_OK ||
		SparkKvDeviceCopierBlockIsPinned(copier,source_logical_page) == 0u ||
		SparkKvDeviceCopierBlockIsPinned(copier,destination_logical_page) == 0u ||
		source.resident_slot_index >= copier->physical_page_count ||
		destination.resident_slot_index >= copier->physical_page_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( copier->pending_count == copier->pending_capacity )
	{
		status = SparkKvDeviceCopierRetire(copier,0u);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	if ( copier->pending_count == copier->pending_capacity )
	{
		copier->busy_count++;
		if ( copier->full_logged == 0u )
			fprintf(stderr,"%s kv copy-on-write pending full capacity=%u\n",copier->module_tag,copier->pending_capacity);
		copier->full_logged = 1u;
		SPARK_FAIL(SPARK_STATUS_BUSY);
	}
	entry = 0;
	for (index=0u; entry == 0 && index<copier->pending_capacity; index++)
		if ( copier->pending[index].active == 0u )
			entry = &copier->pending[index];
	if ( entry == 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvDeviceCopierEnqueue(copier,source.resident_slot_index,destination.resident_slot_index,&bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceEventRecord(entry->event,copier->queue);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceEventRecord(copier->fence_event,copier->queue);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv copy-on-write enqueue failed source=%u destination=%u device=%s\n",
			copier->module_tag,source_logical_page,destination_logical_page,SparkDeviceLastError()->text);
		(void)SparkDeviceQueueWait(copier->queue,SPARK_DEVICE_WAIT_FOREVER);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	entry->source_logical_page = source_logical_page;
	entry->destination_logical_page = destination_logical_page;
	entry->free_destination = 0u;
	entry->active = 1u;
	copier->pending_count++;
	copier->copy_count++;
	copier->copy_bytes += bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvDeviceCopierRetireEntry(SparkKvDeviceCopier *copier, SparkKvDeviceCopyPending *entry)
{
	SparkStatus status;
	status = SparkKvCacheArenaUnpinResidentBlock(copier->arena,entry->source_logical_page);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaUnpinResidentBlock(copier->arena,entry->destination_logical_page);
	if ( status == SPARK_STATUS_OK && entry->free_destination != 0u )
	{
		status = SparkKvCacheArenaFreeBlock(copier->arena,entry->destination_logical_page);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"%s kv copy-on-write retire free failed page=%u status=%s\n",
				copier->module_tag,entry->destination_logical_page,SparkStatusToString(status));
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	entry->active = 0u;
	entry->free_destination = 0u;
	copier->pending_count--;
	copier->retire_count++;
	copier->full_logged = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvDeviceCopierRetire(void *context, uint32_t require_all)
{
	SparkKvDeviceCopier *copier;
	SparkKvDeviceCopyPending *entry;
	SparkStatus status,result;
	uint32_t index;
	copier = (SparkKvDeviceCopier *)context;
	if ( copier == 0 || copier->pending == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	result = SPARK_STATUS_OK;
	for (index=0u; index<copier->pending_capacity; index++)
	{
		entry = &copier->pending[index];
		if ( entry->active == 0u )
			continue;
		status = SparkDeviceEventQuery(entry->event);
		if ( status == SPARK_STATUS_OK )
		{
			status = SparkKvDeviceCopierRetireEntry(copier,entry);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		else if ( status == SPARK_STATUS_PENDING )
		{
			if ( require_all != 0u )
				result = SPARK_STATUS_BUSY;
		}
		else
		{
			fprintf(stderr,"%s kv copy-on-write event failed page=%u device=%s\n",
				copier->module_tag,entry->destination_logical_page,SparkDeviceLastError()->text);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	}
	SPARK_RETURN(result);
}

uint32_t SparkKvDeviceCopierDestinationPins(void *context, uint32_t logical_page)
{
	SparkKvDeviceCopier *copier;
	uint32_t index,pins;
	copier = (SparkKvDeviceCopier *)context;
	pins = 0u;
	if ( copier == 0 || copier->pending == 0 )
		return(0u);
	for (index=0u; index<copier->pending_capacity; index++)
		if ( copier->pending[index].active != 0u && copier->pending[index].free_destination == 0u &&
			copier->pending[index].destination_logical_page == logical_page )
			pins++;
	return(pins);
}

SparkStatus SparkKvDeviceCopierDeferFree(void *context, uint32_t logical_page)
{
	SparkKvDeviceCopier *copier;
	uint32_t index;
	copier = (SparkKvDeviceCopier *)context;
	if ( copier == 0 || copier->pending == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<copier->pending_capacity; index++)
		if ( copier->pending[index].active != 0u && copier->pending[index].free_destination == 0u &&
			copier->pending[index].destination_logical_page == logical_page )
		{
			copier->pending[index].free_destination = 1u;
			copier->deferred_free_count++;
			return(SPARK_STATUS_OK);
		}
	SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkKvDeviceCopierFence(SparkKvDeviceCopier *copier, SparkDeviceQueue *queue)
{
	if ( copier == 0 || copier->fence_event == 0 || queue == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkDeviceQueueWaitEvent(queue,copier->fence_event) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s kv copy-on-write fence failed device=%s\n",copier->module_tag,SparkDeviceLastError()->text);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvDeviceCopierDrain(SparkKvDeviceCopier *copier)
{
	uint32_t index;
	if ( copier == 0 || copier->pending == 0 )
		return(SPARK_STATUS_OK);
	for (index=0u; index<copier->pending_capacity; index++)
		if ( copier->pending[index].active != 0u &&
			SparkDeviceEventWait(copier->pending[index].event,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(SparkKvDeviceCopierRetire(copier,1u));
}

void SparkKvDeviceCopierDestroy(SparkKvDeviceCopier *copier)
{
	if ( copier == 0 )
		return;
	SparkKvDeviceCopierReleaseEvents(copier);
	memset(copier,0,sizeof(*copier));
}
