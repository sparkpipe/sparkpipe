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

static SparkStatus SparkKvPageCacheDiscardLogicalPage(
	SparkKvPageCache *cache,
	uint32_t logical_page_index)
{
	SparkKvCacheBlockView view;
	SparkStatus status;
	if ( cache->page_store != 0 )
	{
		status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,
			logical_page_index,&view);
		if ( status != SPARK_STATUS_OK )
			return(status);
		status = SparkKvPageStoreInvalidate(cache->page_store,
			logical_page_index,view.generation);
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
			entry->reference_count != 0u )
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
			entry->reference_count != 0u )
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
