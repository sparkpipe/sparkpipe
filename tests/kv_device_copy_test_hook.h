#pragma once

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"

#define SPARK_TEST_KV_DEVICE_COPY_CAPACITY 8u

typedef struct SparkTestKvDeviceCopyEntry
{
	uint32_t source;
	uint32_t destination;
	uint32_t deferred;
	uint32_t done;
	uint32_t active;
}
SparkTestKvDeviceCopyEntry;

typedef struct SparkTestKvDeviceCopy
{
	SparkKvCacheArena *arena;
	SparkTestKvDeviceCopyEntry entries[SPARK_TEST_KV_DEVICE_COPY_CAPACITY];
	uint32_t fail_next;
	uint32_t copies;
	uint32_t retired;
	uint32_t deferred_frees;
}
SparkTestKvDeviceCopy;

static SparkStatus SparkTestKvDeviceCopyPage(void *context,uint32_t source,uint32_t destination)
{
	SparkTestKvDeviceCopy *copy = context;
	uint32_t index;
	if ( copy->fail_next != 0u )
	{
		copy->fail_next = 0u;
		return(SPARK_STATUS_IO_ERROR);
	}
	for (index=0u; index<SPARK_TEST_KV_DEVICE_COPY_CAPACITY; index++)
		if ( copy->entries[index].active == 0u )
		{
			copy->entries[index] = (SparkTestKvDeviceCopyEntry){source,destination,0u,0u,1u};
			copy->copies++;
			return(SPARK_STATUS_OK);
		}
	return(SPARK_STATUS_BUSY);
}

static void SparkTestKvDeviceCopyComplete(SparkTestKvDeviceCopy *copy)
{
	SparkKvCacheBlockView source,destination;
	uint32_t index;
	for (index=0u; index<SPARK_TEST_KV_DEVICE_COPY_CAPACITY; index++)
	{
		SparkTestKvDeviceCopyEntry *entry = &copy->entries[index];
		if ( entry->active == 0u || entry->done != 0u )
			continue;
		assert(SparkKvCacheArenaResolveBlock(copy->arena,entry->source,&source) == SPARK_STATUS_OK);
		assert(SparkKvCacheArenaResolveBlock(copy->arena,entry->destination,&destination) == SPARK_STATUS_OK);
		memcpy((void *)destination.key_device_address,(const void *)source.key_device_address,(size_t)copy->arena->key_block_stride_bytes);
		if ( source.value_device_address != 0u )
			memcpy((void *)destination.value_device_address,(const void *)source.value_device_address,(size_t)copy->arena->value_block_stride_bytes);
		entry->done = 1u;
	}
}

static SparkStatus SparkTestKvDeviceCopyRetire(void *context,uint32_t require_all)
{
	SparkTestKvDeviceCopy *copy = context;
	uint32_t index;
	if ( require_all != 0u )
		SparkTestKvDeviceCopyComplete(copy);
	for (index=0u; index<SPARK_TEST_KV_DEVICE_COPY_CAPACITY; index++)
	{
		SparkTestKvDeviceCopyEntry *entry = &copy->entries[index];
		if ( entry->active == 0u || entry->done == 0u )
			continue;
		assert(SparkKvCacheArenaUnpinResidentBlock(copy->arena,entry->source) == SPARK_STATUS_OK);
		assert(SparkKvCacheArenaUnpinResidentBlock(copy->arena,entry->destination) == SPARK_STATUS_OK);
		if ( entry->deferred != 0u )
			assert(SparkKvCacheArenaFreeBlock(copy->arena,entry->destination) == SPARK_STATUS_OK);
		entry->active = 0u;
		copy->retired++;
	}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkTestKvDeviceCopyDestinationPins(void *context,uint32_t page)
{
	SparkTestKvDeviceCopy *copy = context;
	uint32_t index,pins = 0u;
	for (index=0u; index<SPARK_TEST_KV_DEVICE_COPY_CAPACITY; index++)
		if ( copy->entries[index].active != 0u && copy->entries[index].deferred == 0u && copy->entries[index].destination == page )
			pins++;
	return(pins);
}

static SparkStatus SparkTestKvDeviceCopyDeferFree(void *context,uint32_t page)
{
	SparkTestKvDeviceCopy *copy = context;
	uint32_t index,found = 0u;
	for (index=0u; index<SPARK_TEST_KV_DEVICE_COPY_CAPACITY; index++)
		if ( copy->entries[index].active != 0u && copy->entries[index].deferred == 0u && copy->entries[index].destination == page )
		{
			copy->entries[index].deferred = 1u;
			found = 1u;
			break;
		}
	if ( found == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	copy->deferred_frees++;
	return(SPARK_STATUS_OK);
}

static void SparkTestKvAttachDeviceCopy(SparkKvPageCache *cache,SparkTestKvDeviceCopy *copy,SparkKvCacheArena *arena)
{
	SparkKvPageCacheDeviceCopy hook;
	memset(copy,0,sizeof(*copy));
	copy->arena = arena;
	hook.copy_page = SparkTestKvDeviceCopyPage;
	hook.retire_copies = SparkTestKvDeviceCopyRetire;
	hook.destination_pins = SparkTestKvDeviceCopyDestinationPins;
	hook.defer_free = SparkTestKvDeviceCopyDeferFree;
	hook.context = copy;
	assert(SparkKvPageCacheAttachDeviceCopy(cache,&hook) == SPARK_STATUS_OK);
}

static void SparkTestKvDeviceCopySettle(SparkTestKvDeviceCopy *copy)
{
	SparkTestKvDeviceCopyComplete(copy);
	assert(SparkTestKvDeviceCopyRetire(copy,0u) == SPARK_STATUS_OK);
}
