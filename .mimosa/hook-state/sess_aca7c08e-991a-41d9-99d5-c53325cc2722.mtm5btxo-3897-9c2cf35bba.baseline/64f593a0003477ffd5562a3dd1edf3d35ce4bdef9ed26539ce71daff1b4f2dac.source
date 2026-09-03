#include "spark_dsv4_paged_cache.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static uint32_t SparkDsv4PagedCacheConfigurationIsValid(
	const SparkDsv4PagedCacheConfiguration *configuration)
{
	uint32_t lane_page_capacity;
	if ( configuration == 0 || configuration->layer_count == 0u ||
		configuration->first_layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT ||
		configuration->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT -
			configuration->first_layer_index ||
		configuration->resident_sequence_capacity == 0u ||
		configuration->maximum_sequence_positions == 0u ||
		configuration->maximum_sequence_positions > SPARK_DSV4_MODEL_MAX_POSITIONS ||
		configuration->logical_page_capacity == 0u ||
		configuration->physical_page_capacity == 0u )
		return(0u);
	lane_page_capacity = (configuration->maximum_sequence_positions +
		SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS - 1u) /
		SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS;
	return(configuration->logical_page_capacity >=
		configuration->physical_page_capacity &&
		configuration->logical_page_capacity >= lane_page_capacity ? 1u : 0u);
}

static void *SparkDsv4PagedCacheCalloc(uint64_t count,uint64_t bytes)
{
	if ( count == 0u || bytes == 0u || count > SIZE_MAX / bytes )
		return(0);
	return(calloc((size_t)count,(size_t)bytes));
}

void SparkDsv4PagedCacheDestroyHost(SparkDsv4PagedCache *cache)
{
	if ( cache == 0 )
		return;
	free(cache->physical_page_generations);
	free(cache->physical_page_content_logical_pages);
	free(cache->host_device_page_table);
	free(cache->hash_bucket_heads);
	free(cache->entry_indices_by_logical_page);
	free(cache->resident_page_owners);
	free(cache->sequences);
	free(cache->entries);
	free(cache->blocks);
	cache->physical_page_generations = 0;
	cache->physical_page_content_logical_pages = 0;
	cache->host_device_page_table = 0;
	cache->hash_bucket_heads = 0;
	cache->entry_indices_by_logical_page = 0;
	cache->resident_page_owners = 0;
	cache->sequences = 0;
	cache->entries = 0;
	cache->blocks = 0;
}

static SparkStatus SparkDsv4PagedCacheAllocateHost(
	SparkDsv4PagedCache *cache,
	uint32_t sequence_capacity)
{
	cache->blocks = (SparkKvCacheBlock *)SparkDsv4PagedCacheCalloc(
		cache->logical_page_capacity,sizeof(cache->blocks[0]));
	cache->entries = (SparkKvPageCacheEntry *)SparkDsv4PagedCacheCalloc(
		cache->logical_page_capacity,sizeof(cache->entries[0]));
	cache->sequences = (SparkKvPageCacheSequence *)SparkDsv4PagedCacheCalloc(
		sequence_capacity,sizeof(cache->sequences[0]));
	cache->resident_page_owners = (uint32_t *)SparkDsv4PagedCacheCalloc(
		cache->physical_page_capacity,sizeof(cache->resident_page_owners[0]));
	cache->hash_bucket_heads = (uint32_t *)SparkDsv4PagedCacheCalloc(
		cache->logical_page_capacity,sizeof(cache->hash_bucket_heads[0]));
	cache->entry_indices_by_logical_page =
		(uint32_t *)SparkDsv4PagedCacheCalloc(cache->logical_page_capacity,
			sizeof(cache->entry_indices_by_logical_page[0]));
	cache->physical_page_generations = (uint64_t *)SparkDsv4PagedCacheCalloc(
		cache->physical_page_capacity,sizeof(cache->physical_page_generations[0]));
	cache->physical_page_content_logical_pages =
		(uint32_t *)SparkDsv4PagedCacheCalloc(cache->physical_page_capacity,
			sizeof(cache->physical_page_content_logical_pages[0]));
	cache->host_device_page_table = (uint32_t *)SparkDsv4PagedCacheCalloc(
		(uint64_t)sequence_capacity * cache->lane_page_capacity,
		sizeof(cache->host_device_page_table[0]));
	if ( cache->blocks == 0 || cache->entries == 0 || cache->sequences == 0 ||
		cache->resident_page_owners == 0 || cache->hash_bucket_heads == 0 ||
		cache->entry_indices_by_logical_page == 0 ||
		cache->physical_page_generations == 0 ||
		cache->physical_page_content_logical_pages == 0 ||
		cache->host_device_page_table == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(cache->physical_page_content_logical_pages,0xff,
		(uint64_t)cache->physical_page_capacity *
		sizeof(cache->physical_page_content_logical_pages[0]));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4PagedCacheInitializeArena(
	SparkDsv4PagedCache *cache,
	uint32_t layer_count)
{
	SparkKvCacheConfiguration configuration;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.logical_block_count = cache->logical_page_capacity;
	configuration.block_token_count = SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS;
	configuration.resident_block_capacity = cache->physical_page_capacity;
	configuration.layer_count = layer_count;
	configuration.kv_head_count = 1u;
	configuration.head_dim = 1u;
	configuration.bytes_per_scalar = sizeof(uint8_t);
	configuration.key_block_stride_bytes = cache->layout.page_stride_bytes;
	configuration.key_device_base = cache->device_page_pool;
	configuration.blocks = cache->blocks;
	configuration.resident_slot_logical_block_indices =
		cache->resident_page_owners;
	return(SparkKvCacheArenaInitialize(&cache->arena,&configuration));
}

static SparkStatus SparkDsv4PagedCacheInitializeDirectory(
	SparkDsv4PagedCache *cache,
	uint32_t sequence_capacity)
{
	SparkKvPageCacheConfiguration configuration;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	configuration.sequence_capacity = sequence_capacity;
	configuration.entry_capacity = cache->logical_page_capacity;
	configuration.hash_bucket_count = cache->logical_page_capacity;
	configuration.kv_cache_arena = &cache->arena;
	configuration.entries = cache->entries;
	configuration.sequences = cache->sequences;
	configuration.hash_bucket_heads = cache->hash_bucket_heads;
	configuration.entry_indices_by_logical_page =
		cache->entry_indices_by_logical_page;
	return(SparkKvPageCacheInitialize(&cache->page_cache,&configuration));
}

SparkStatus SparkDsv4PagedCacheInitialize(
	SparkDsv4PagedCache *cache,
	const SparkDsv4PagedCacheConfiguration *configuration,
	SparkStageModuleLedger *ledger)
{
	uint64_t device_bytes,page_table_bytes;
	SparkStatus status;
	if ( cache == 0 || ledger == 0 ||
		SparkDsv4PagedCacheConfigurationIsValid(configuration) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(cache,0,sizeof(*cache));
	status = SparkDsv4PagedPoolBuildLayout(configuration->first_layer_index,
		configuration->layer_count,&cache->layout) == 0 ?
		SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
	cache->logical_page_capacity = configuration->logical_page_capacity;
	cache->physical_page_capacity = configuration->physical_page_capacity;
	cache->lane_page_capacity = (configuration->maximum_sequence_positions +
		SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS - 1u) /
		SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS;
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4PagedCacheAllocateHost(cache,
			configuration->resident_sequence_capacity);
	device_bytes = (uint64_t)cache->physical_page_capacity *
		cache->layout.page_stride_bytes;
	page_table_bytes = (uint64_t)configuration->resident_sequence_capacity *
		cache->lane_page_capacity * sizeof(uint32_t);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(ledger,device_bytes,
			&cache->device_page_pool);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(ledger,page_table_bytes,
			(void **)&cache->device_page_table);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4PagedCacheInitializeArena(cache,
			configuration->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4PagedCacheInitializeDirectory(cache,
			configuration->resident_sequence_capacity);
	if ( status != SPARK_STATUS_OK )
		SparkDsv4PagedCacheDestroyHost(cache);
	return(status);
}

static SparkStatus SparkDsv4PagedCacheResolvePages(
	SparkDsv4PagedCache *cache,
	const uint32_t *logical_pages,
	uint32_t page_count,
	uint32_t *physical_pages)
{
	SparkKvCacheBlockView view;
	uint32_t page;
	SparkStatus status;
	for (page=0u; page<page_count; page++)
	{
		status = SparkKvCacheArenaResolveBlock(&cache->arena,
			logical_pages[page],&view);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
			view.resident_slot_index >= cache->physical_page_capacity )
			return(SPARK_STATUS_BUSY);
		physical_pages[page] = view.resident_slot_index;
		if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u )
		{
			cache->physical_page_content_logical_pages[
				view.resident_slot_index] = logical_pages[page];
			cache->physical_page_generations[view.resident_slot_index] =
				view.generation;
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4PagedCachePrepareLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_pages,
	uint32_t *physical_pages,
	uint32_t page_capacity,
	SparkDsv4PagedCacheLane *prepared_lane)
{
	SparkKvCacheBlockView view;
	uint32_t mutable_page,page_count,pinned_page_count,prepared_page_count;
	SparkStatus status;
	if ( cache == 0 || lane == 0 || logical_pages == 0 ||
		physical_pages == 0 || prepared_lane == 0 ||
		page_capacity < cache->lane_page_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(prepared_lane,0,sizeof(*prepared_lane));
	prepared_lane->mutable_logical_page = SPARK_DSV4_PAGED_CACHE_NO_PAGE;
	prepared_lane->mutable_physical_page = SPARK_DSV4_PAGED_CACHE_NO_PAGE;
	prepared_lane->parent_physical_page = SPARK_DSV4_PAGED_CACHE_NO_PAGE;
	mutable_page = SPARK_DSV4_PAGED_CACHE_NO_PAGE;
	status = SparkKvPageCachePrepareLane(&cache->page_cache,lane,logical_pages,
		page_capacity,&page_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4PagedCachePinLane(cache,logical_pages,page_count);
	pinned_page_count = status == SPARK_STATUS_OK ? page_count : 0u;
	prepared_page_count = page_count;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheBeginLaneTransaction(&cache->page_cache,lane,
			&mutable_page,&prepared_lane->mutation_flags);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheBuildLaneTable(&cache->page_cache,
			lane->resident_sequence_slot,lane->sequence_id,logical_pages,
			page_capacity,&page_count);
	if ( status == SPARK_STATUS_OK && page_count < prepared_page_count )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4PagedCachePinLane(cache,
			logical_pages + prepared_page_count,
			page_count - prepared_page_count);
	if ( status == SPARK_STATUS_OK )
		pinned_page_count = page_count;
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4PagedCacheResolvePages(cache,logical_pages,page_count,
			physical_pages);
	if ( status != SPARK_STATUS_OK )
	{
		if ( pinned_page_count != 0u )
			(void)SparkDsv4PagedCacheUnpinLane(cache,logical_pages,
				pinned_page_count);
		(void)SparkKvPageCacheRollbackLaneTransaction(&cache->page_cache,lane,
			prepared_lane->mutation_flags);
		prepared_lane->mutation_flags = 0u;
		return(status);
	}
	prepared_lane->logical_page_count = page_count;
	prepared_lane->mutable_logical_page = mutable_page;
	if ( mutable_page == SPARK_KV_CACHE_NO_BLOCK )
		return(SPARK_STATUS_OK);
	status = SparkKvCacheArenaResolveBlock(&cache->arena,mutable_page,&view);
	if ( status != SPARK_STATUS_OK || page_count == 0u ||
		logical_pages[page_count - 1u] != mutable_page )
	{
		(void)SparkDsv4PagedCacheUnpinLane(cache,logical_pages,page_count);
		(void)SparkKvPageCacheRollbackLaneTransaction(&cache->page_cache,lane,
			prepared_lane->mutation_flags);
		prepared_lane->logical_page_count = 0u;
		prepared_lane->mutation_flags = 0u;
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_INTERNAL_ERROR : status);
	}
	prepared_lane->mutable_physical_page = view.resident_slot_index;
	prepared_lane->mutable_generation = view.generation;
	if ( page_count > 1u )
		prepared_lane->parent_physical_page = physical_pages[page_count - 2u];
	prepared_lane->requires_initialization =
		cache->physical_page_content_logical_pages[view.resident_slot_index] !=
		mutable_page ||
		cache->physical_page_generations[view.resident_slot_index] !=
		view.generation ? 1u : 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4PagedCachePrefetchLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_pages,
	uint32_t page_capacity)
{
	uint32_t page_count;
	if ( cache == 0 || lane == 0 || logical_pages == 0 ||
		page_capacity < cache->lane_page_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageCachePrepareLane(&cache->page_cache,lane,logical_pages,
		page_capacity,&page_count));
}

SparkStatus SparkDsv4PagedCachePinLane(
	SparkDsv4PagedCache *cache,
	const uint32_t *logical_pages,
	uint32_t logical_page_count)
{
	uint32_t page;
	SparkStatus status;
	if ( cache == 0 || (logical_page_count != 0u && logical_pages == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (page=0u; page<logical_page_count; page++)
	{
		status = SparkKvCacheArenaPinResidentBlock(&cache->arena,
			logical_pages[page]);
		if ( status != SPARK_STATUS_OK )
		{
			while ( page != 0u )
			{
				page--;
				(void)SparkKvCacheArenaUnpinResidentBlock(&cache->arena,
					logical_pages[page]);
			}
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4PagedCacheUnpinLane(
	SparkDsv4PagedCache *cache,
	const uint32_t *logical_pages,
	uint32_t logical_page_count)
{
	uint32_t page;
	SparkStatus status,result;
	if ( cache == 0 || (logical_page_count != 0u && logical_pages == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	result = SPARK_STATUS_OK;
	for (page=0u; page<logical_page_count; page++)
	{
		status = SparkKvCacheArenaUnpinResidentBlock(&cache->arena,
			logical_pages[page]);
		if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
			result = status;
	}
	return(result);
}

SparkStatus SparkDsv4PagedCacheCompleteLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	const SparkDsv4PagedCacheLane *prepared_lane)
{
	if ( cache == 0 || lane == 0 || prepared_lane == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( prepared_lane->requires_initialization != 0u )
	{
		cache->physical_page_content_logical_pages[
			prepared_lane->mutable_physical_page] =
			prepared_lane->mutable_logical_page;
		cache->physical_page_generations[
			prepared_lane->mutable_physical_page] =
			prepared_lane->mutable_generation;
	}
	return(SparkKvPageCacheCompleteLane(&cache->page_cache,lane));
}

SparkStatus SparkDsv4PagedCacheAbortLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane)
{
	if ( cache == 0 || lane == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageCacheReleaseLane(&cache->page_cache,
		lane->resident_sequence_slot,lane->sequence_id));
}

SparkStatus SparkDsv4PagedCacheRollbackLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	const SparkDsv4PagedCacheLane *prepared_lane)
{
	if ( cache == 0 || lane == 0 || prepared_lane == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageCacheRollbackLaneTransaction(&cache->page_cache,lane,
		prepared_lane->mutation_flags));
}

SparkStatus SparkDsv4PagedCacheReleaseLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane)
{
	if ( cache == 0 || lane == 0 ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageCacheReleaseLane(&cache->page_cache,
		lane->resident_sequence_slot,lane->sequence_id));
}
