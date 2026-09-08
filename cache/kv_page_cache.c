#include "sparkpipe/spark_kv_page_cache.h"

#include <string.h>

#include "sparkpipe/spark_model_driver_support.h"

static uint64_t SparkKvPageCacheHashIdentity(
	const SparkModelDriverCacheIdentity *identity,
	uint32_t token_count)
{
	uint64_t hash;
	uint32_t index;
	hash = UINT64_C(1469598103934665603);
	for (index=0u; index<sizeof(identity->sha256); index++)
	{
		hash ^= identity->sha256[index];
		hash *= UINT64_C(1099511628211);
	}
	hash ^= token_count;
	hash *= UINT64_C(1099511628211);
	hash ^= hash >> 33u;
	hash *= UINT64_C(0xff51afd7ed558ccd);
	return(hash ^ (hash >> 33u));
}

static uint32_t SparkKvPageCacheBucket(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheIdentity *identity,
	uint32_t token_count)
{
	return((uint32_t)(SparkKvPageCacheHashIdentity(identity,token_count) %
		cache->hash_bucket_count));
}

static uint32_t SparkKvPageCacheConfigurationIsValid(
	const SparkKvPageCacheConfiguration *configuration)
{
	return(configuration != 0 &&
		configuration->abi_version == SPARK_KV_PAGE_CACHE_ABI_VERSION &&
		configuration->descriptor_bytes == SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES &&
		configuration->sequence_capacity != 0u &&
		configuration->entry_capacity != 0u &&
		configuration->hash_bucket_count != 0u &&
		configuration->reserved0 == 0u &&
		configuration->kv_cache_arena != 0 &&
		configuration->kv_cache_arena->block_token_count != 0u &&
		configuration->kv_cache_arena->block_token_count <=
			SPARK_KV_CACHE_MAX_BLOCK_TOKENS &&
		(configuration->page_store == 0 ||
		 (configuration->page_store->abi_version ==
			SPARK_KV_PAGE_STORE_ABI_VERSION &&
		  configuration->page_store->logical_page_capacity >=
			configuration->kv_cache_arena->logical_block_count)) &&
		configuration->entry_capacity <= configuration->kv_cache_arena->logical_block_count &&
		configuration->entries != 0 && configuration->sequences != 0 &&
		configuration->hash_bucket_heads != 0 &&
		configuration->entry_indices_by_logical_page != 0 ? 1u : 0u);
}

static uint32_t SparkKvPageCacheIsValid(const SparkKvPageCache *cache)
{
	return(cache != 0 && cache->abi_version == SPARK_KV_PAGE_CACHE_ABI_VERSION &&
		cache->descriptor_bytes == SPARK_KV_PAGE_CACHE_BYTES &&
		cache->sequence_capacity != 0u && cache->entry_capacity != 0u &&
		cache->hash_bucket_count != 0u && cache->kv_cache_arena != 0 &&
		cache->entries != 0 && cache->sequences != 0 &&
		cache->hash_bucket_heads != 0 &&
		cache->entry_indices_by_logical_page != 0 ? 1u : 0u);
}

SparkStatus SparkKvPageCacheAttachStateStore(SparkKvPageCache *cache,SparkKvPageStore *store)
{
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->page_store == 0 || cache->state_store != 0 || store == 0 || store == cache->page_store || store->abi_version != SPARK_KV_PAGE_STORE_ABI_VERSION || store->descriptor_bytes != SPARK_KV_PAGE_STORE_BYTES || store->worker_state == 0 || store->logical_page_capacity < cache->kv_cache_arena->logical_block_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cache->live_sequence_count != 0u || cache->published_page_count != 0u || store->backing_page_count != 0u )
		return(SPARK_STATUS_BUSY);
	cache->state_store = store;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheInitialize(
	SparkKvPageCache *cache,
	const SparkKvPageCacheConfiguration *configuration)
{
	uint32_t index;
	if ( cache == 0 || SparkKvPageCacheConfigurationIsValid(configuration) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(cache,0,sizeof(*cache));
	cache->abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	cache->descriptor_bytes = SPARK_KV_PAGE_CACHE_BYTES;
	cache->sequence_capacity = configuration->sequence_capacity;
	cache->entry_capacity = configuration->entry_capacity;
	cache->hash_bucket_count = configuration->hash_bucket_count;
	cache->kv_cache_arena = configuration->kv_cache_arena;
	cache->page_store = configuration->page_store;
	cache->entries = configuration->entries;
	cache->sequences = configuration->sequences;
	cache->hash_bucket_heads = configuration->hash_bucket_heads;
	cache->entry_indices_by_logical_page =
		configuration->entry_indices_by_logical_page;
	cache->free_entry_head = 0u;
	memset(cache->sequences,0,(uint64_t)cache->sequence_capacity * sizeof(cache->sequences[0]));
	for (index=0u; index<cache->sequence_capacity; index++)
	{
		cache->sequences[index].terminal_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->sequences[index].mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	}
	memset(cache->entries,0,(uint64_t)cache->entry_capacity * sizeof(cache->entries[0]));
	for (index=0u; index<cache->entry_capacity; index++)
	{
		cache->entries[index].parent_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->entries[index].logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
		cache->entries[index].hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->entries[index].free_next = index + 1u < cache->entry_capacity ?
			index + 1u : SPARK_KV_PAGE_CACHE_NO_INDEX;
	}
	for (index=0u; index<cache->hash_bucket_count; index++)
		cache->hash_bucket_heads[index] = SPARK_KV_PAGE_CACHE_NO_INDEX;
	for (index=0u; index<cache->kv_cache_arena->logical_block_count; index++)
		cache->entry_indices_by_logical_page[index] =
			SPARK_KV_PAGE_CACHE_NO_INDEX;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheEnsureResidentPages(
	SparkKvPageCache *cache,
	const uint32_t *logical_page_indices,
	uint32_t logical_page_count)
{
	SparkKvCacheBlockView view;
	SparkStatus result,status;
	uint32_t page;
	result = SPARK_STATUS_OK;
	if ( cache->page_store != 0 )
	{
		status = SparkKvPageStoreProgress(cache->page_store,
			cache->kv_cache_arena,cache->page_store->transfer_capacity);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	for (page=0u; page<logical_page_count; page++)
	{
		status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,
			logical_page_indices[page],&view);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
			continue;
		if ( cache->page_store == 0 )
			return(SPARK_STATUS_BUSY);
		status = SparkKvPageStorePrefetch(cache->page_store,
			cache->kv_cache_arena,logical_page_indices[page]);
		if ( status == SPARK_STATUS_BUSY )
			result = SPARK_STATUS_BUSY;
		else if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(result);
}

static uint32_t SparkKvPageCacheFindEntry(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheIdentity *identity,
	uint32_t token_count,
	uint32_t record_stats)
{
	SparkKvPageCacheEntry *entry;
	uint32_t entry_index;
	entry_index = cache->hash_bucket_heads[
		SparkKvPageCacheBucket(cache,identity,token_count)];
	while ( entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( entry_index >= cache->entry_capacity )
			return(SPARK_KV_PAGE_CACHE_NO_INDEX);
		entry = &cache->entries[entry_index];
		if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u &&
			entry->token_count == token_count &&
			memcmp(&entry->identity,identity,sizeof(*identity)) == 0 )
		{
			cache->epoch++;
			entry->last_used_epoch = cache->epoch;
			if ( record_stats != 0u )
				cache->prefix_hit_count++;
			return(entry_index);
		}
		entry_index = entry->hash_next;
	}
	if ( record_stats != 0u )
		cache->prefix_miss_count++;
	return(SPARK_KV_PAGE_CACHE_NO_INDEX);
}

static uint32_t SparkKvPageCacheFindEntryConst(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheIdentity *identity,
	uint32_t token_count)
{
	const SparkKvPageCacheEntry *entry;
	uint32_t entry_index;
	entry_index = cache->hash_bucket_heads[
		SparkKvPageCacheBucket(cache,identity,token_count)];
	while ( entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( entry_index >= cache->entry_capacity )
			return(SPARK_KV_PAGE_CACHE_NO_INDEX);
		entry = &cache->entries[entry_index];
		if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u &&
			entry->token_count == token_count &&
			memcmp(&entry->identity,identity,sizeof(*identity)) == 0 )
			return(entry_index);
		entry_index = entry->hash_next;
	}
	return(SPARK_KV_PAGE_CACHE_NO_INDEX);
}

static void SparkKvPageCacheUnlinkEntry(
	SparkKvPageCache *cache,
	uint32_t entry_index)
{
	SparkKvPageCacheEntry *entry;
	uint32_t bucket,*link;
	entry = &cache->entries[entry_index];
	bucket = SparkKvPageCacheBucket(cache,&entry->identity,entry->token_count);
	link = &cache->hash_bucket_heads[bucket];
	while ( *link != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( *link == entry_index )
		{
			*link = entry->hash_next;
			return;
		}
		link = &cache->entries[*link].hash_next;
	}
}

static uint32_t SparkKvPageCachePageCanDiscard(const SparkKvPageCache *cache,uint32_t logical_page_index)
{
	const SparkKvCacheBlock *block;
	if ( logical_page_index >= cache->kv_cache_arena->logical_block_count )
		return(0u);
	block = &cache->kv_cache_arena->blocks[logical_page_index];
	return((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u && block->reference_count == 1u && block->residency_reference_count == 0u && (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) == 0u);
}

static SparkStatus SparkKvPageCacheDiscardLogicalPage(
	SparkKvPageCache *cache,
	uint32_t logical_page_index)
{
	SparkKvCacheBlockView view;
	SparkStatus status;
	if ( SparkKvPageCachePageCanDiscard(cache,logical_page_index) == 0u )
		return(SPARK_STATUS_BUSY);
	if ( cache->page_store != 0 )
	{
		status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,
			logical_page_index,&view);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( cache->state_store != 0 )
			status = SparkKvPageStoreInvalidatePair(cache->page_store,cache->state_store,logical_page_index,view.generation);
		else
			status = SparkKvPageStoreInvalidate(cache->page_store,logical_page_index,view.generation);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	status = SparkKvCacheArenaReleaseBlockReference(
		cache->kv_cache_arena,logical_page_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkKvCacheArenaFreeBlock(cache->kv_cache_arena,logical_page_index));
}

static uint32_t SparkKvPageCacheEvictionCandidate(
	const SparkKvPageCache *cache,
	uint32_t require_resident)
{
	const SparkKvPageCacheEntry *entry,*victim;
	uint32_t entry_index,logical_page_index,resident_slot,victim_index;
	victim = 0;
	victim_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	for (resident_slot=0u; require_resident != 0u &&
		resident_slot<cache->kv_cache_arena->resident_block_capacity;
		resident_slot++)
	{
		logical_page_index = cache->kv_cache_arena->
			resident_slot_logical_block_indices[resident_slot];
		if ( logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
			continue;
		if ( logical_page_index >= cache->kv_cache_arena->logical_block_count )
			return(SPARK_KV_PAGE_CACHE_NO_INDEX);
		entry_index = cache->entry_indices_by_logical_page[logical_page_index];
		if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
			continue;
		if ( entry_index >= cache->entry_capacity )
			return(SPARK_KV_PAGE_CACHE_NO_INDEX);
		entry = &cache->entries[entry_index];
		if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) == 0u ||
			entry->reference_count != 0u || SparkKvPageCachePageCanDiscard(cache,entry->logical_page_index) == 0u )
			continue;
		if ( victim == 0 || entry->last_used_epoch < victim->last_used_epoch )
		{
			victim = entry;
			victim_index = entry_index;
		}
	}
	for (entry_index=0u; require_resident == 0u &&
		entry_index<cache->entry_capacity; entry_index++)
	{
		entry = &cache->entries[entry_index];
		if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) == 0u ||
			entry->reference_count != 0u || SparkKvPageCachePageCanDiscard(cache,entry->logical_page_index) == 0u )
			continue;
		if ( victim == 0 || entry->last_used_epoch < victim->last_used_epoch )
		{
			victim = entry;
			victim_index = entry_index;
		}
	}
	return(victim_index);
}

static SparkStatus SparkKvPageCacheEvictEntry(
	SparkKvPageCache *cache,
	uint32_t entry_index)
{
	SparkKvPageCacheEntry *entry;
	uint32_t parent;
	SparkStatus status;
	if ( entry_index >= cache->entry_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	entry = &cache->entries[entry_index];
	if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) == 0u ||
		entry->reference_count != 0u )
		return(SPARK_STATUS_BUSY);
	parent = entry->parent_entry_index;
	status = SparkKvPageCacheDiscardLogicalPage(cache,entry->logical_page_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	cache->entry_indices_by_logical_page[entry->logical_page_index] =
		SPARK_KV_PAGE_CACHE_NO_INDEX;
	SparkKvPageCacheUnlinkEntry(cache,entry_index);
	if ( parent != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( parent >= cache->entry_capacity || cache->entries[parent].reference_count == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		cache->entries[parent].reference_count--;
	}
	memset(entry,0,sizeof(*entry));
	entry->parent_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	entry->hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->free_next = cache->free_entry_head;
	cache->free_entry_head = entry_index;
	cache->evicted_entry_count++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheEvictUnused(SparkKvPageCache *cache)
{
	uint32_t entry;
	if ( SparkKvPageCacheIsValid(cache) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	entry = SparkKvPageCacheEvictionCandidate(cache,0u);
	return(entry == SPARK_KV_PAGE_CACHE_NO_INDEX ? SPARK_STATUS_CAPACITY_EXCEEDED : SparkKvPageCacheEvictEntry(cache,entry));
}

static SparkStatus SparkKvPageCacheAcquireEntry(
	SparkKvPageCache *cache,
	uint32_t *entry_index_out)
{
	SparkKvPageCacheEntry *entry;
	uint32_t entry_index;
	SparkStatus status;
	entry_index = cache->free_entry_head;
	if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		entry_index = SparkKvPageCacheEvictionCandidate(cache,0u);
		if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = SparkKvPageCacheEvictEntry(cache,entry_index);
		if ( status != SPARK_STATUS_OK )
			return(status);
		entry_index = cache->free_entry_head;
	}
	if ( entry_index >= cache->entry_capacity )
		return(SPARK_STATUS_INTERNAL_ERROR);
	entry = &cache->entries[entry_index];
	cache->free_entry_head = entry->free_next;
	memset(entry,0,sizeof(*entry));
	entry->parent_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	entry->hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->free_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	*entry_index_out = entry_index;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvPageCacheEntryIsAncestor(
	const SparkKvPageCache *cache,
	uint32_t terminal_entry_index,
	uint32_t ancestor_entry_index)
{
	while ( terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( terminal_entry_index == ancestor_entry_index )
			return(1u);
		if ( terminal_entry_index >= cache->entry_capacity )
			return(0u);
		terminal_entry_index = cache->entries[terminal_entry_index].parent_entry_index;
	}
	return(0u);
}

static SparkStatus SparkKvPageCacheResolveLanePrefix(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t record_stats,
	uint32_t *entry_index_out)
{
	uint32_t entry_index;
	*entry_index_out = SPARK_KV_PAGE_CACHE_NO_INDEX;
	if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) == 0u )
		return(SPARK_STATUS_OK);
	entry_index = SparkKvPageCacheFindEntry(cache,&lane->prefix_identity,
		lane->prefix_token_count,record_stats);
	if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
		return(SPARK_STATUS_NOT_FOUND);
	*entry_index_out = entry_index;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheValidateExistingSequence(
	const SparkKvPageCache *cache,
	const SparkKvPageCacheSequence *sequence,
	const SparkModelDriverCacheLane *lane,
	uint32_t prefix_entry_index)
{
	if ( sequence->sequence_id != lane->sequence_id ||
		sequence->next_token_position != lane->sequence_position )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( prefix_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX &&
		SparkKvPageCacheEntryIsAncestor(cache,sequence->terminal_entry_index,
			prefix_entry_index) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheResolveLaneChain(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t record_stats,
	uint32_t *terminal_entry_index_out)
{
	SparkKvPageCacheSequence *sequence;
	uint32_t prefix_entry_index;
	SparkStatus status;
	status = SparkKvPageCacheResolveLanePrefix(cache,lane,record_stats,
		&prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id )
	{
		status = SparkKvPageCacheValidateExistingSequence(cache,sequence,lane,
			prefix_entry_index);
		if ( status != SPARK_STATUS_OK )
			return(status);
		*terminal_entry_index_out = sequence->terminal_entry_index;
		return(SPARK_STATUS_OK);
	}
	if ( prefix_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( lane->sequence_position != 0u )
			return(SPARK_STATUS_NOT_FOUND);
	}
	else if ( lane->prefix_token_count != lane->sequence_position )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*terminal_entry_index_out = prefix_entry_index;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheResolveLaneChainConst(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *terminal_entry_index_out)
{
	const SparkKvPageCacheSequence *sequence;
	uint32_t prefix_entry_index;
	if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u )
	{
		prefix_entry_index = SparkKvPageCacheFindEntryConst(cache,
			&lane->prefix_identity,lane->prefix_token_count);
		if ( prefix_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
			return(SPARK_STATUS_NOT_FOUND);
	}
	else
		prefix_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id )
	{
		SparkStatus status;
		status = SparkKvPageCacheValidateExistingSequence(cache,sequence,lane,
			prefix_entry_index);
		if ( status != SPARK_STATUS_OK )
			return(status);
		*terminal_entry_index_out = sequence->terminal_entry_index;
		return(SPARK_STATUS_OK);
	}
	if ( prefix_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( lane->sequence_position != 0u )
			return(SPARK_STATUS_NOT_FOUND);
	}
	else if ( lane->prefix_token_count != lane->sequence_position )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*terminal_entry_index_out = prefix_entry_index;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheAppendEntryPages(
	const SparkKvPageCache *cache,
	uint32_t terminal_entry_index,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *page_count_out)
{
	const SparkKvPageCacheEntry *entry;
	uint32_t cursor,page_count;
	if ( terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX &&
		terminal_entry_index >= cache->entry_capacity )
		return(SPARK_STATUS_INTERNAL_ERROR);
	page_count = terminal_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX ? 0u :
		cache->entries[terminal_entry_index].page_count;
	if ( page_count > logical_page_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	cursor = page_count;
	while ( terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( terminal_entry_index >= cache->entry_capacity || cursor == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		entry = &cache->entries[terminal_entry_index];
		logical_page_indices[--cursor] = entry->logical_page_index;
		terminal_entry_index = entry->parent_entry_index;
	}
	if ( cursor != 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	*page_count_out = page_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheResolveLanePages(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out)
{
	const SparkKvPageCacheSequence *sequence;
	uint32_t page_count,terminal_entry_index;
	SparkStatus status;
	if ( logical_page_count_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*logical_page_count_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity ||
		(logical_page_capacity != 0u && logical_page_indices == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChainConst(cache,lane,
		&terminal_entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkKvPageCacheAppendEntryPages(cache,terminal_entry_index,
		logical_page_indices,logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id &&
		sequence->mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
	{
		if ( page_count >= logical_page_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		logical_page_indices[page_count++] =
			sequence->mutable_logical_page_index;
	}
	*logical_page_count_out = page_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheGetLaneMutablePageDemand(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_page_demand_out)
{
	const SparkKvPageCacheSequence *sequence;
	uint32_t first_token,terminal_entry_index;
	SparkStatus status;
	if ( mutable_page_demand_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*mutable_page_demand_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChainConst(cache,lane,
		&terminal_entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	(void)terminal_entry_index;
	if ( lane->context_token_count == lane->sequence_position )
		return(SPARK_STATUS_OK);
	first_token = ((uint32_t)lane->sequence_position /
		cache->kv_cache_arena->block_token_count) *
		cache->kv_cache_arena->block_token_count;
	if ( (lane->context_token_count - 1u) /
		cache->kv_cache_arena->block_token_count !=
		first_token / cache->kv_cache_arena->block_token_count )
		return(SPARK_STATUS_UNSUPPORTED);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id &&
		sequence->mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
		return(sequence->mutable_first_token_index == first_token ?
			SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( lane->sequence_position != first_token )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*mutable_page_demand_out = 1u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCachePrepareLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out)
{
	SparkKvPageCacheSequence *sequence;
	uint32_t page_count,prefix_entry_index;
	SparkStatus status;
	if ( logical_page_count_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*logical_page_count_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity ||
		(logical_page_capacity != 0u && logical_page_indices == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChain(cache,lane,1u,&prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	status = SparkKvPageCacheAppendEntryPages(cache,prefix_entry_index,
		logical_page_indices,logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( sequence->sequence_id == lane->sequence_id &&
		sequence->mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
	{
		if ( page_count >= logical_page_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		logical_page_indices[page_count++] = sequence->mutable_logical_page_index;
	}
	status = SparkKvPageCacheEnsureResidentPages(cache,logical_page_indices,
		page_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*logical_page_count_out = page_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheReleaseMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence)
{
	SparkStatus status;
	if ( sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
		return(SPARK_STATUS_OK);
	status = SparkKvPageCacheDiscardLogicalPage(cache,
		sequence->mutable_logical_page_index);
	if ( status == SPARK_STATUS_OK )
	{
		sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
		sequence->mutable_first_token_index = 0u;
	}
	return(status);
}

SparkStatus SparkKvPageCacheReleaseLane(
	SparkKvPageCache *cache,
	uint32_t resident_sequence_slot,
	uint64_t sequence_id)
{
	SparkKvPageCacheSequence *sequence;
	uint64_t generation;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		resident_sequence_slot >= cache->sequence_capacity || sequence_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = &cache->sequences[resident_sequence_slot];
	if ( sequence->sequence_id == 0u )
		return(SPARK_STATUS_OK);
	if ( sequence->sequence_id != sequence_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cache->live_sequence_count == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvPageCacheReleaseMutable(cache,sequence);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( sequence->terminal_entry_index >= cache->entry_capacity ||
			cache->entries[sequence->terminal_entry_index].reference_count == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		cache->entries[sequence->terminal_entry_index].reference_count--;
	}
	cache->live_sequence_count--;
	generation = sequence->generation + 1u;
	memset(sequence,0,sizeof(*sequence));
	sequence->generation = generation != 0u ? generation : 1u;
	sequence->terminal_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	cache->released_sequence_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheBindLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t prefix_entry_index)
{
	SparkKvPageCacheSequence *sequence;
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id )
		return(SparkKvPageCacheValidateExistingSequence(cache,sequence,lane,
			prefix_entry_index));
	if ( sequence->sequence_id != 0u )
		return(SPARK_STATUS_BUSY);
	sequence->generation++;
	if ( sequence->generation == 0u )
		sequence->generation = 1u;
	sequence->sequence_id = lane->sequence_id;
	sequence->next_token_position = (uint32_t)lane->sequence_position;
	sequence->terminal_entry_index = prefix_entry_index;
	sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	if ( prefix_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
		cache->entries[prefix_entry_index].reference_count++;
	cache->live_sequence_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheAllocateMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	uint32_t first_token_index)
{
	uint32_t logical_page_index;
	SparkStatus status;
	logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	status = SparkKvCacheArenaAcquireBlock(cache->kv_cache_arena,
		&logical_page_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaRetainBlock(cache->kv_cache_arena,
			logical_page_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaMarkBlockResident(cache->kv_cache_arena,
			logical_page_index);
	if ( status == SPARK_STATUS_CAPACITY_EXCEEDED ||
		status == SPARK_STATUS_BUSY )
	{
		uint32_t victim;
		victim = SparkKvPageCacheEvictionCandidate(cache,1u);
		if ( victim != SPARK_KV_PAGE_CACHE_NO_INDEX &&
			SparkKvPageCacheEvictEntry(cache,victim) == SPARK_STATUS_OK )
			status = SparkKvCacheArenaMarkBlockResident(cache->kv_cache_arena,
				logical_page_index);
	}
	if ( status != SPARK_STATUS_OK )
	{
		if ( logical_page_index != SPARK_KV_CACHE_NO_BLOCK &&
			logical_page_index < cache->kv_cache_arena->logical_block_count &&
			cache->kv_cache_arena->blocks[logical_page_index].reference_count != 0u )
			(void)SparkKvCacheArenaReleaseBlockReference(cache->kv_cache_arena,
				logical_page_index);
		if ( logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
			(void)SparkKvCacheArenaFreeBlock(cache->kv_cache_arena,logical_page_index);
		return(status);
	}
	sequence->mutable_logical_page_index = logical_page_index;
	sequence->mutable_first_token_index = first_token_index;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheBeginLaneInternal(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out,
	uint32_t *mutation_flags_out)
{
	SparkKvPageCacheSequence *sequence;
	uint32_t first_token,mutation_flags,newly_bound,prefix_entry_index;
	SparkStatus status;
	if ( mutable_logical_page_index_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*mutable_logical_page_index_out = SPARK_KV_CACHE_NO_BLOCK;
	if ( mutation_flags_out != 0 )
		*mutation_flags_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChain(cache,lane,0u,&prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	newly_bound = cache->sequences[lane->resident_sequence_slot].sequence_id == 0u ? 1u : 0u;
	status = SparkKvPageCacheBindLane(cache,lane,prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	mutation_flags = newly_bound != 0u ?
		SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE : 0u;
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( lane->context_token_count == lane->sequence_position )
	{
		if ( mutation_flags_out != 0 )
			*mutation_flags_out = mutation_flags;
		return(SPARK_STATUS_OK);
	}
	first_token = ((uint32_t)lane->sequence_position /
		cache->kv_cache_arena->block_token_count) *
		cache->kv_cache_arena->block_token_count;
	if ( (lane->context_token_count - 1u) /
		cache->kv_cache_arena->block_token_count !=
		first_token / cache->kv_cache_arena->block_token_count )
		status = SPARK_STATUS_UNSUPPORTED;
	else if ( sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
	{
		if ( lane->sequence_position != first_token )
			status = SPARK_STATUS_INVALID_ARGUMENT;
		else
		{
			status = SparkKvPageCacheAllocateMutable(cache,sequence,first_token);
			if ( status == SPARK_STATUS_OK )
				mutation_flags |=
					SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE;
		}
	}
	if ( status == SPARK_STATUS_OK &&
		sequence->mutable_first_token_index != first_token )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK )
	{
		if ( newly_bound != 0u )
			(void)SparkKvPageCacheReleaseLane(cache,
				lane->resident_sequence_slot,lane->sequence_id);
		return(status);
	}
	*mutable_logical_page_index_out = sequence->mutable_logical_page_index;
	if ( mutation_flags_out != 0 )
		*mutation_flags_out = mutation_flags;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheBeginLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out)
{
	return(SparkKvPageCacheBeginLaneInternal(cache,lane,
		mutable_logical_page_index_out,0));
}

SparkStatus SparkKvPageCacheBeginLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out,
	uint32_t *mutation_flags_out)
{
	if ( mutation_flags_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageCacheBeginLaneInternal(cache,lane,
		mutable_logical_page_index_out,mutation_flags_out));
}

SparkStatus SparkKvPageCacheRollbackLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t mutation_flags)
{
	SparkKvPageCacheSequence *sequence;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(mutation_flags & ~SPARK_KV_PAGE_CACHE_KNOWN_MUTATIONS) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( mutation_flags == 0u )
		return(SPARK_STATUS_OK);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id != lane->sequence_id ||
		sequence->next_token_position != lane->sequence_position )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u )
		return(SparkKvPageCacheReleaseLane(cache,
			lane->resident_sequence_slot,lane->sequence_id));
	if ( (mutation_flags &
		SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE) != 0u )
		return(SparkKvPageCacheReleaseMutable(cache,sequence));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCachePublishNewEntry(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	const SparkModelDriverCacheLane *lane)
{
	SparkKvPageCacheEntry *entry;
	uint32_t bucket,entry_index,parent;
	SparkStatus status;
	status = SparkKvPageCacheAcquireEntry(cache,&entry_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	parent = sequence->terminal_entry_index;
	entry = &cache->entries[entry_index];
	entry->flags = SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID;
	entry->token_count = lane->publish_token_count;
	entry->page_count = lane->publish_token_count /
		cache->kv_cache_arena->block_token_count;
	entry->parent_entry_index = parent;
	entry->logical_page_index = sequence->mutable_logical_page_index;
	entry->reference_count = 1u;
	entry->identity = lane->publish_identity;
	if ( cache->entry_indices_by_logical_page[entry->logical_page_index] !=
		SPARK_KV_PAGE_CACHE_NO_INDEX )
		return(SPARK_STATUS_INTERNAL_ERROR);
	cache->entry_indices_by_logical_page[entry->logical_page_index] =
		entry_index;
	cache->epoch++;
	entry->last_used_epoch = cache->epoch;
	bucket = SparkKvPageCacheBucket(cache,&entry->identity,entry->token_count);
	entry->hash_next = cache->hash_bucket_heads[bucket];
	cache->hash_bucket_heads[bucket] = entry_index;
	sequence->terminal_entry_index = entry_index;
	sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	sequence->mutable_first_token_index = 0u;
	cache->published_page_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCachePublishDeduplicated(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	uint32_t entry_index)
{
	SparkKvPageCacheEntry *entry;
	SparkStatus status;
	entry = &cache->entries[entry_index];
	if ( entry->parent_entry_index != sequence->terminal_entry_index )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkKvPageCacheReleaseMutable(cache,sequence);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( cache->entries[sequence->terminal_entry_index].reference_count == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		cache->entries[sequence->terminal_entry_index].reference_count--;
	}
	entry->reference_count++;
	sequence->terminal_entry_index = entry_index;
	cache->deduplicated_page_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCachePublishMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	const SparkModelDriverCacheLane *lane)
{
	uint32_t entry_index,expected_parent_count;
	if ( sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK ||
		lane->publish_token_count == 0u ||
		lane->publish_token_count % cache->kv_cache_arena->block_token_count != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	expected_parent_count = lane->publish_token_count -
		cache->kv_cache_arena->block_token_count;
	if ( (sequence->terminal_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX) !=
		(expected_parent_count == 0u) ||
		(sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX &&
		 cache->entries[sequence->terminal_entry_index].token_count !=
		 expected_parent_count) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	entry_index = SparkKvPageCacheFindEntry(cache,&lane->publish_identity,
		lane->publish_token_count,0u);
	if ( entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
		return(SparkKvPageCachePublishDeduplicated(cache,sequence,entry_index));
	return(SparkKvPageCachePublishNewEntry(cache,sequence,lane));
}

SparkStatus SparkKvPageCacheCompleteLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane)
{
	SparkKvPageCacheSequence *sequence;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id != lane->sequence_id ||
		sequence->next_token_position != lane->sequence_position )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( lane->context_token_count != lane->sequence_position &&
		sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( lane->context_token_count != lane->sequence_position )
	{
		status = SparkKvCacheArenaMarkBlockDirty(cache->kv_cache_arena,
			sequence->mutable_logical_page_index);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) != 0u )
	{
		if ( lane->publish_token_count != lane->context_token_count )
			return(SPARK_STATUS_UNSUPPORTED);
		status = SparkKvPageCachePublishMutable(cache,sequence,lane);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	sequence->next_token_position = lane->context_token_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheBuildLaneTable(
	SparkKvPageCache *cache,
	uint32_t resident_sequence_slot,
	uint64_t sequence_id,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out)
{
	SparkKvPageCacheSequence *sequence;
	uint32_t page_count;
	SparkStatus status;
	if ( logical_page_count_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*logical_page_count_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		resident_sequence_slot >= cache->sequence_capacity || sequence_id == 0u ||
		(logical_page_capacity != 0u && logical_page_indices == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = &cache->sequences[resident_sequence_slot];
	if ( sequence->sequence_id != sequence_id )
		return(SPARK_STATUS_NOT_FOUND);
	status = SparkKvPageCacheAppendEntryPages(cache,
		sequence->terminal_entry_index,logical_page_indices,
		logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( sequence->mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
	{
		if ( page_count >= logical_page_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		logical_page_indices[page_count++] = sequence->mutable_logical_page_index;
	}
	*logical_page_count_out = page_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheRollbackPinnedLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	const uint32_t *logical_pages,
	uint32_t pinned_count,
	uint32_t mutation_flags,
	SparkStatus failure)
{
	SparkStatus status;
	status = SparkKvCacheArenaUnpinResidentTable(cache->kv_cache_arena,logical_pages,pinned_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkKvPageCacheRollbackLaneTransaction(cache,lane,mutation_flags);
	return(status == SPARK_STATUS_OK ? failure : status);
}

SparkStatus SparkKvPageCacheBeginPinnedLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_pages,
	uint32_t *physical_pages,
	uint32_t page_capacity,
	uint32_t *page_count_out,
	uint32_t *mutation_flags_out)
{
	uint32_t prepared_count,page_count,pinned_count,mutable_page,mutation_flags = 0u;
	SparkStatus status;
	if ( page_count_out == 0 || mutation_flags_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*page_count_out = 0u;
	*mutation_flags_out = 0u;
	if ( logical_pages == 0 || physical_pages == 0 || page_capacity == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCachePrepareLane(cache,lane,logical_pages,page_capacity,&prepared_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkKvCacheArenaPinResidentTable(cache->kv_cache_arena,logical_pages,prepared_count,physical_pages);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pinned_count = prepared_count;
	status = SparkKvPageCacheBeginLaneTransaction(cache,lane,&mutable_page,&mutation_flags);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheBuildLaneTable(cache,lane->resident_sequence_slot,lane->sequence_id,logical_pages,page_capacity,&page_count);
	if ( status == SPARK_STATUS_OK && page_count < prepared_count )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaPinResidentTable(cache->kv_cache_arena,logical_pages + prepared_count,(page_count - prepared_count),physical_pages + prepared_count);
	if ( status != SPARK_STATUS_OK )
		return(SparkKvPageCacheRollbackPinnedLane(cache,lane,logical_pages,pinned_count,mutation_flags,status));
	*page_count_out = page_count;
	*mutation_flags_out = mutation_flags;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvLaneTransactionMatches(const SparkKvLaneTransaction *owner,const SparkModelDriverAdmissionRequest *request,const SparkModelDriverCacheLane *lane)
{
	const SparkModelDriverAdmissionRequest *saved = &owner->request;
	if ( saved->program_id != request->program_id || saved->submission_id != request->submission_id || saved->control_generation != request->control_generation || saved->transaction_id != request->transaction_id )
		return(0u);
	if ( saved->request_generation != request->request_generation || saved->step_generation != request->step_generation || saved->request_id != request->request_id || saved->sequence_id != request->sequence_id || saved->sequence_position != request->sequence_position )
		return(0u);
	if ( saved->deadline_time_ns != request->deadline_time_ns || saved->active_slot_count != request->active_slot_count || saved->new_token_count != request->new_token_count || saved->priority != request->priority || saved->frame_flags != request->frame_flags || saved->cache_lane_count != request->cache_lane_count )
		return(0u);
	return(memcmp(&saved->residency,&request->residency,sizeof(saved->residency)) == 0 && memcmp(&owner->lane,lane,sizeof(*lane)) == 0);
}

static void SparkKvLaneTransactionsNextEpoch(SparkKvLaneTransactions *transactions)
{
	uint32_t index;
	transactions->validation_epoch++;
	if ( transactions->validation_epoch == 0u )
	{
		for (index=0u; index<transactions->cache->sequence_capacity; index++)
			transactions->lanes[index].validation_epoch = 0u;
		transactions->validation_epoch = 1u;
	}
}

static SparkStatus SparkKvLaneTransactionsValidate(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t index,slot;
	if ( transactions == 0 || SparkKvPageCacheIsValid(transactions->cache) == 0u || transactions->lanes == 0 || transactions->logical_pages == 0 || transactions->physical_pages == 0 || transactions->page_capacity == 0u || request == 0 || request->program_id == 0u || request->cache_lane_count == 0u || request->cache_lane_count > transactions->cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLaneTransactionsNextEpoch(transactions);
	for (index=0u; index<request->cache_lane_count; index++)
	{
		slot = request->cache_lanes[index].resident_sequence_slot;
		if ( slot >= transactions->cache->sequence_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( transactions->lanes[slot].validation_epoch == transactions->validation_epoch )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		transactions->lanes[slot].validation_epoch = transactions->validation_epoch;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvLaneTransactionAbort(SparkKvLaneTransactions *transactions,SparkKvLaneTransaction *owner)
{
	SparkStatus status;
	uint64_t offset = ((uint64_t)owner->lane.resident_sequence_slot * transactions->page_capacity);
	status = SparkKvCacheArenaUnpinResidentTable(transactions->cache->kv_cache_arena,transactions->logical_pages + offset,owner->page_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	owner->page_count = 0u;
	status = SparkKvPageCacheRollbackLaneTransaction(transactions->cache,&owner->lane,owner->mutation_flags);
	if ( status == SPARK_STATUS_OK )
		owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
	return(status);
}

static SparkStatus SparkKvLaneTransactionsRequire(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request,uint32_t phase)
{
	SparkKvLaneTransaction *owner;
	uint32_t index;
	for (index=0u; index<request->cache_lane_count; index++)
	{
		owner = &transactions->lanes[request->cache_lanes[index].resident_sequence_slot];
		if ( owner->phase != phase )
			return(SPARK_STATUS_BUSY);
		if ( SparkKvLaneTransactionMatches(owner,request,&request->cache_lanes[index]) == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvLaneTransactionsPrepare(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	SparkKvLaneTransaction *owner;
	const SparkModelDriverCacheLane *lane;
	uint32_t index,owned = 0u;
	uint64_t offset;
	SparkStatus status,rollback;
	for (index=0u; index<request->cache_lane_count; index++)
		owned += transactions->lanes[request->cache_lanes[index].resident_sequence_slot].phase != SPARK_KV_LANE_TRANSACTION_EMPTY ? 1u : 0u;
	if ( owned != 0u )
		return(SparkKvLaneTransactionsRequire(transactions,request,SPARK_KV_LANE_TRANSACTION_PREPARED));
	for (index=0u; index<request->cache_lane_count; index++)
	{
		lane = &request->cache_lanes[index];
		owner = &transactions->lanes[lane->resident_sequence_slot];
		offset = ((uint64_t)lane->resident_sequence_slot * transactions->page_capacity);
		status = SparkKvPageCacheBeginPinnedLaneTransaction(transactions->cache,lane,transactions->logical_pages + offset,transactions->physical_pages + offset,transactions->page_capacity,&owner->page_count,&owner->mutation_flags);
		if ( status != SPARK_STATUS_OK )
			break;
		owner->request = *request;
		owner->request.cache_lanes = 0;
		owner->lane = *lane;
		owner->phase = SPARK_KV_LANE_TRANSACTION_PREPARED;
	}
	if ( index == request->cache_lane_count )
		return(SPARK_STATUS_OK);
	while ( index != 0u )
	{
		index--;
		rollback = SparkKvLaneTransactionAbort(transactions,&transactions->lanes[request->cache_lanes[index].resident_sequence_slot]);
		if ( rollback != SPARK_STATUS_OK )
			status = rollback;
	}
	return(status);
}

static SparkStatus SparkKvLaneTransactionsRelease(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	const SparkModelDriverCacheLane *lane;
	uint32_t index;
	SparkStatus status;
	for (index=0u; index<request->cache_lane_count; index++)
	{
		lane = &request->cache_lanes[index];
		if ( transactions->lanes[lane->resident_sequence_slot].phase != SPARK_KV_LANE_TRANSACTION_EMPTY )
			return(SPARK_STATUS_BUSY);
		if ( transactions->cache->sequences[lane->resident_sequence_slot].sequence_id != lane->sequence_id )
			return(SPARK_STATUS_NOT_FOUND);
	}
	for (index=0u; index<request->cache_lane_count; index++)
	{
		lane = &request->cache_lanes[index];
		status = SparkKvPageCacheReleaseLane(transactions->cache,lane->resident_sequence_slot,lane->sequence_id);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvLaneTransactionsAdmit(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	SparkKvLaneTransaction *owner;
	uint32_t index,phase;
	SparkStatus status,result;
	status = SparkKvLaneTransactionsValidate(transactions,request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		return(request->admission_flags == 0u ? SparkKvLaneTransactionsRelease(transactions,request) : SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE )
		return(SparkKvLaneTransactionsPrepare(transactions,request));
	phase = request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT ? SPARK_KV_LANE_TRANSACTION_PREPARED : SPARK_KV_LANE_TRANSACTION_COMMITTED;
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
		phase = transactions->lanes[request->cache_lanes[0].resident_sequence_slot].phase;
	if ( phase != SPARK_KV_LANE_TRANSACTION_PREPARED && phase != SPARK_KV_LANE_TRANSACTION_COMMITTED )
		return(SPARK_STATUS_BUSY);
	status = SparkKvLaneTransactionsRequire(transactions,request,phase);
	if ( status != SPARK_STATUS_OK )
		return(status);
	result = SPARK_STATUS_OK;
	for (index=0u; index<request->cache_lane_count; index++)
	{
		owner = &transactions->lanes[request->cache_lanes[index].resident_sequence_slot];
		if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
		{
			status = SparkKvLaneTransactionAbort(transactions,owner);
			if ( result == SPARK_STATUS_OK )
				result = status;
		}
		else if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT )
			owner->phase = SPARK_KV_LANE_TRANSACTION_COMMITTED;
	}
	return(result);
}

SparkStatus SparkKvLaneTransactionsClaim(SparkKvLaneTransactions *transactions,const SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	uint32_t index,slot;
	SparkStatus status;
	if ( transactions == 0 || frame == 0 || transactions->cache == 0 || transactions->lanes == 0 || frame->cache_lanes == 0 || frame->cache_lane_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = frame->cache_lanes[0].resident_sequence_slot;
	if ( slot >= transactions->cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	request = transactions->lanes[slot].request;
	request.submission_id = frame->driver_dispatch_cookie1;
	request.control_generation = frame->driver_dispatch_generation;
	request.transaction_id = frame->driver_dispatch_cookie0;
	request.program_id = frame->program_id;
	request.request_id = frame->request_id;
	request.sequence_id = frame->sequence_id;
	request.sequence_position = frame->sequence_position;
	request.deadline_time_ns = frame->deadline_time_ns;
	request.active_slot_count = frame->active_slot_count;
	request.new_token_count = frame->new_token_count;
	request.priority = frame->priority;
	request.frame_flags = frame->flags & ~SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
	request.admission_flags = 0u;
	request.residency = frame->residency;
	request.cache_lanes = frame->cache_lanes;
	request.cache_lane_count = frame->cache_lane_count;
	status = SparkKvLaneTransactionsValidate(transactions,&request);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvLaneTransactionsRequire(transactions,&request,SPARK_KV_LANE_TRANSACTION_COMMITTED);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (index=0u; index<frame->cache_lane_count; index++)
		transactions->lanes[frame->cache_lanes[index].resident_sequence_slot].phase = SPARK_KV_LANE_TRANSACTION_EXECUTING;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvLaneTransactionFinish(SparkKvLaneTransactions *transactions,uint32_t resident_slot,SparkStatus execution_status,uint32_t extra_tokens)
{
	SparkKvLaneTransaction *owner;
	SparkModelDriverCacheLane lane;
	SparkStatus status,release_status;
	uint64_t offset;
	if ( transactions == 0 || SparkKvPageCacheIsValid(transactions->cache) == 0u || transactions->lanes == 0 || resident_slot >= transactions->cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	owner = &transactions->lanes[resident_slot];
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	offset = ((uint64_t)resident_slot * transactions->page_capacity);
	status = SparkKvCacheArenaUnpinResidentTable(transactions->cache->kv_cache_arena,transactions->logical_pages + offset,owner->page_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	owner->page_count = 0u;
	lane = owner->lane;
	status = execution_status;
	if ( status == SPARK_STATUS_OK && extra_tokens > UINT32_MAX - lane.context_token_count )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
	{
		lane.context_token_count += extra_tokens;
		status = SparkKvPageCacheCompleteLane(transactions->cache,&lane);
	}
	if ( status != SPARK_STATUS_OK )
	{
		// Failed GPU execution may have overwritten existing mutable state;
		// metadata rollback cannot make that sequence safe to continue.
		release_status = SparkKvPageCacheReleaseLane(transactions->cache,resident_slot,lane.sequence_id);
		if ( release_status != SPARK_STATUS_OK )
			return(release_status);
	}
	owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
	return(status);
}

static SparkStatus SparkKvLaneTransactionsDiscardCompleted(SparkKvLaneTransactions *transactions,const uint32_t *resident_slots,uint32_t lane_count,SparkStatus failure)
{
	SparkKvLaneTransaction *owner;
	SparkStatus status;
	uint32_t index,slot;
	for (index=0u; index<lane_count; index++)
	{
		slot = resident_slots[index];
		owner = &transactions->lanes[slot];
		if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EMPTY || transactions->cache->sequences[slot].sequence_id == 0u )
			continue;
		status = SparkKvPageCacheReleaseLane(transactions->cache,slot,owner->lane.sequence_id);
		if ( status != SPARK_STATUS_OK )
		{
			owner->phase = SPARK_KV_LANE_TRANSACTION_EXECUTING;
			failure = status;
		}
	}
	return(failure);
}

SparkStatus SparkKvLaneTransactionsFinish(SparkKvLaneTransactions *transactions,const uint32_t *resident_slots,uint32_t lane_count,SparkStatus execution_status,uint32_t extra_tokens)
{
	uint32_t index,slot;
	const SparkModelDriverAdmissionRequest *request;
	SparkStatus status,result = execution_status;
	if ( transactions == 0 || SparkKvPageCacheIsValid(transactions->cache) == 0u || transactions->lanes == 0 || resident_slots == 0 || lane_count == 0u || lane_count > transactions->cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( resident_slots[0] >= transactions->cache->sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	request = &transactions->lanes[resident_slots[0]].request;
	if ( request->cache_lane_count != lane_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLaneTransactionsNextEpoch(transactions);
	for (index=0u; index<lane_count; index++)
	{
		slot = resident_slots[index];
		if ( slot >= transactions->cache->sequence_capacity || transactions->lanes[slot].phase != SPARK_KV_LANE_TRANSACTION_EXECUTING || transactions->lanes[slot].validation_epoch == transactions->validation_epoch )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( SparkKvLaneTransactionMatches(&transactions->lanes[slot],request,&transactions->lanes[slot].lane) == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		transactions->lanes[slot].validation_epoch = transactions->validation_epoch;
	}
	for (index=0u; index<lane_count; index++)
	{
		status = SparkKvLaneTransactionFinish(transactions,resident_slots[index],execution_status,extra_tokens);
		if ( result == SPARK_STATUS_OK )
			result = status;
	}
	if ( result != SPARK_STATUS_OK )
		return(SparkKvLaneTransactionsDiscardCompleted(transactions,resident_slots,lane_count,result));
	return(result);
}
