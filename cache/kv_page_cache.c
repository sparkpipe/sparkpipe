#include "sparkpipe/spark_kv_page_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_error_site.h"
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cache->live_sequence_count != 0u || cache->published_page_count != 0u || store->backing_page_count != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	cache->state_store = store;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheInitialize(
	SparkKvPageCache *cache,
	const SparkKvPageCacheConfiguration *configuration)
{
	uint32_t index;
	if ( cache == 0 || SparkKvPageCacheConfigurationIsValid(configuration) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
	cache->lru_head = SPARK_KV_PAGE_CACHE_NO_INDEX;
	cache->lru_tail = SPARK_KV_PAGE_CACHE_NO_INDEX;
	memset(cache->sequences,0,(uint64_t)cache->sequence_capacity * sizeof(cache->sequences[0]));
	for (index=0u; index<cache->sequence_capacity; index++)
	{
		cache->sequences[index].terminal_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->sequences[index].mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
		cache->sequences[index].mutable_page_count = 0u;
	}
	memset(cache->entries,0,(uint64_t)cache->entry_capacity * sizeof(cache->entries[0]));
	for (index=0u; index<cache->entry_capacity; index++)
	{
		cache->entries[index].parent_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->entries[index].logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
		cache->entries[index].hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->entries[index].lru_prev = SPARK_KV_PAGE_CACHE_NO_INDEX;
		cache->entries[index].lru_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
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

static SparkStatus SparkKvPageCacheRelieveBacking(SparkKvPageCache *cache,const uint32_t *protected_pages,uint32_t protected_count);
static void SparkKvPageCacheBackingOutcome(SparkKvPageCache *cache,SparkStatus status,uint32_t backing_full);

static SparkStatus SparkKvPageCachePrefetchRelieved(SparkKvPageCache *cache,const uint32_t *pages,uint32_t page_count,uint32_t page)
{
	uint64_t before = cache->kv_cache_arena->park_backing_full_count;
	uint32_t backing_full = 0u;
	SparkStatus status;
	status = SparkKvPageStorePrefetch(cache->page_store,cache->kv_cache_arena,pages[page]);
	while ( status == SPARK_STATUS_CAPACITY_EXCEEDED && cache->kv_cache_arena->park_backing_full_count != before )
	{
		backing_full = 1u;
		before = cache->kv_cache_arena->park_backing_full_count;
		if ( SparkKvPageCacheRelieveBacking(cache,pages,page_count) != SPARK_STATUS_OK )
			break;
		status = SparkKvPageStorePrefetch(cache->page_store,cache->kv_cache_arena,pages[page]);
	}
	SparkKvPageCacheBackingOutcome(cache,status,backing_full);
	return(status);
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
			SPARK_RETURN(status);
	}
	for (page=0u; page<logical_page_count; page++)
	{
		status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,
			logical_page_indices[page],&view);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
			continue;
		if ( cache->page_store == 0 )
			SPARK_FAIL(SPARK_STATUS_BUSY);
		status = SparkKvPageCachePrefetchRelieved(cache,logical_page_indices,logical_page_count,page);
		if ( status == SPARK_STATUS_BUSY )
			result = SPARK_STATUS_BUSY;
		else if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(result);
}

static void SparkKvPageCacheLruUnlink(SparkKvPageCache *cache, uint32_t entry_index)
{
	SparkKvPageCacheEntry *entry;
	entry = &cache->entries[entry_index];
	if ( entry->lru_prev != SPARK_KV_PAGE_CACHE_NO_INDEX )
		cache->entries[entry->lru_prev].lru_next = entry->lru_next;
	else
		cache->lru_head = entry->lru_next;
	if ( entry->lru_next != SPARK_KV_PAGE_CACHE_NO_INDEX )
		cache->entries[entry->lru_next].lru_prev = entry->lru_prev;
	else
		cache->lru_tail = entry->lru_prev;
	entry->lru_prev = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->lru_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
}

static void SparkKvPageCacheLruLinkAfter(SparkKvPageCache *cache, uint32_t entry_index, uint32_t previous)
{
	SparkKvPageCacheEntry *entry;
	uint32_t next;
	entry = &cache->entries[entry_index];
	next = previous == SPARK_KV_PAGE_CACHE_NO_INDEX ? cache->lru_head : cache->entries[previous].lru_next;
	entry->lru_prev = previous;
	entry->lru_next = next;
	if ( previous == SPARK_KV_PAGE_CACHE_NO_INDEX )
		cache->lru_head = entry_index;
	else
		cache->entries[previous].lru_next = entry_index;
	if ( next == SPARK_KV_PAGE_CACHE_NO_INDEX )
		cache->lru_tail = entry_index;
	else
		cache->entries[next].lru_prev = entry_index;
}

static uint32_t SparkKvPageCacheLruPredecessor(const SparkKvPageCache *cache, uint64_t epoch)
{
	uint32_t forward,backward;
	forward = cache->lru_head;
	backward = cache->lru_tail;
	for (;;)
	{
		if ( backward == SPARK_KV_PAGE_CACHE_NO_INDEX || cache->entries[backward].last_used_epoch < epoch )
			return(backward);
		if ( forward == SPARK_KV_PAGE_CACHE_NO_INDEX )
			return(cache->lru_tail);
		if ( cache->entries[forward].last_used_epoch > epoch )
			return(cache->entries[forward].lru_prev);
		backward = cache->entries[backward].lru_prev;
		forward = cache->entries[forward].lru_next;
	}
}

static void SparkKvPageCacheReference(SparkKvPageCache *cache, uint32_t entry_index)
{
	if ( cache->entries[entry_index].reference_count++ == 0u )
		SparkKvPageCacheLruUnlink(cache,entry_index);
}

static void SparkKvPageCacheDereference(SparkKvPageCache *cache, uint32_t entry_index)
{
	if ( --cache->entries[entry_index].reference_count == 0u )
		SparkKvPageCacheLruLinkAfter(cache,entry_index,SparkKvPageCacheLruPredecessor(cache,cache->entries[entry_index].last_used_epoch));
}

static void SparkKvPageCacheTouch(SparkKvPageCache *cache, uint32_t entry_index)
{
	cache->epoch++;
	cache->entries[entry_index].last_used_epoch = cache->epoch;
	if ( cache->entries[entry_index].reference_count != 0u )
		return;
	SparkKvPageCacheLruUnlink(cache,entry_index);
	SparkKvPageCacheLruLinkAfter(cache,entry_index,cache->lru_tail);
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
			SparkKvPageCacheTouch(cache,entry_index);
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

static uint32_t SparkKvPageCacheCopyPins(const SparkKvPageCache *cache,uint32_t logical_page_index)
{
	return(cache->device_copy.destination_pins != 0 ?
		cache->device_copy.destination_pins(cache->device_copy.context,logical_page_index) : 0u);
}

static uint32_t SparkKvPageCachePageCanDiscard(const SparkKvPageCache *cache,uint32_t logical_page_index)
{
	const SparkKvCacheBlock *block;
	uint32_t pins;
	if ( logical_page_index >= cache->kv_cache_arena->logical_block_count )
		return(0u);
	pins = SparkKvPageCacheCopyPins(cache,logical_page_index);
	block = &cache->kv_cache_arena->blocks[logical_page_index];
	return((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u && block->reference_count == 1u && block->residency_reference_count == pins && (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) == 0u);
}

static SparkStatus SparkKvPageCacheDiscardLogicalPage(
	SparkKvPageCache *cache,
	uint32_t logical_page_index)
{
	SparkKvCacheBlockView view;
	SparkStatus status;
	uint32_t pins;
	pins = SparkKvPageCacheCopyPins(cache,logical_page_index);
	if ( SparkKvPageCachePageCanDiscard(cache,logical_page_index) == 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	if ( cache->page_store != 0 )
	{
		status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,
			logical_page_index,&view);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( cache->state_store != 0 )
			status = SparkKvPageStoreInvalidatePair(cache->page_store,cache->state_store,logical_page_index,view.generation);
		else
			status = SparkKvPageStoreInvalidate(cache->page_store,logical_page_index,view.generation);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	status = SparkKvCacheArenaReleaseBlockReference(
		cache->kv_cache_arena,logical_page_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pins != 0u )
		return(cache->device_copy.defer_free(cache->device_copy.context,logical_page_index) == SPARK_STATUS_OK ?
			SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR);
	return(SparkKvCacheArenaFreeBlock(cache->kv_cache_arena,logical_page_index));
}

#define SPARK_KV_PAGE_CACHE_DEMOTE_WINDOW 32u

static uint32_t SparkKvPageCacheSaveEligible(const SparkKvPageCache *cache,uint32_t entry_index);
static uint32_t SparkKvPageCacheSaveIsPending(const SparkKvPageCacheSnapshot *snapshot,uint32_t terminal);
static SparkStatus SparkKvPageCacheSavePush(SparkKvPageCache *cache,uint32_t terminal);
static uint32_t SparkKvPageCachePageIsProtected(const uint32_t *pages,uint32_t count,uint32_t page);

static uint32_t SparkKvPageCacheDiscardIsSafe(const SparkKvPageCache *cache,uint32_t entry_index)
{
	if ( cache->snapshot == 0 || (cache->entries[entry_index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED) != 0u )
		return(1u);
	return(SparkKvPageCacheSaveEligible(cache,entry_index) == 0u ? 1u : 0u);
}

static uint32_t SparkKvPageCacheSelectVictim(SparkKvPageCache *cache,const uint32_t *protected_pages,uint32_t protected_count)
{
	uint32_t candidates[SPARK_KV_PAGE_CACHE_DEMOTE_WINDOW],count = 0u,entry,index,best = SPARK_KV_PAGE_CACHE_NO_INDEX,best_safe = 0u,safe;
	for (entry=cache->lru_head; entry != SPARK_KV_PAGE_CACHE_NO_INDEX && count < SPARK_KV_PAGE_CACHE_DEMOTE_WINDOW; entry=cache->entries[entry].lru_next)
		if ( SparkKvPageCachePageCanDiscard(cache,cache->entries[entry].logical_page_index) != 0u &&
			SparkKvPageCachePageIsProtected(protected_pages,protected_count,cache->entries[entry].logical_page_index) == 0u )
			candidates[count++] = entry;
	for (index=0u; index<count; index++)
	{
		safe = SparkKvPageCacheDiscardIsSafe(cache,candidates[index]);
		if ( best == SPARK_KV_PAGE_CACHE_NO_INDEX || cache->entries[candidates[index]].priority < cache->entries[best].priority ||
			(cache->entries[candidates[index]].priority == cache->entries[best].priority && safe != 0u && best_safe == 0u) )
		{
			best = candidates[index];
			best_safe = safe;
		}
	}
	if ( best == SPARK_KV_PAGE_CACHE_NO_INDEX )
		return(best);
	for (index=0u; index<count; index++)
		if ( candidates[index] != best && SparkKvPageCacheDiscardIsSafe(cache,candidates[index]) == 0u &&
			SparkKvPageCacheSaveIsPending(cache->snapshot,candidates[index]) == 0u && SparkKvPageCacheSavePush(cache,candidates[index]) == SPARK_STATUS_OK )
			cache->snapshot->demote_queued_count++;
	if ( best_safe == 0u )
		cache->snapshot->evicted_unsaved_count++;
	return(best);
}

static uint32_t SparkKvPageCacheResidentVictim(const SparkKvPageCache *cache)
{
	const SparkKvPageCacheEntry *entry,*victim;
	uint32_t entry_index,logical_page_index,resident_slot,victim_index;
	victim = 0;
	victim_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	for (resident_slot=0u; resident_slot<cache->kv_cache_arena->resident_block_capacity; resident_slot++)
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
		if ( victim == 0 || entry->priority < victim->priority ||
			(entry->priority == victim->priority && entry->last_used_epoch < victim->last_used_epoch) )
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	entry = &cache->entries[entry_index];
	if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) == 0u ||
		entry->reference_count != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	parent = entry->parent_entry_index;
	status = SparkKvPageCacheDiscardLogicalPage(cache,entry->logical_page_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	cache->entry_indices_by_logical_page[entry->logical_page_index] =
		SPARK_KV_PAGE_CACHE_NO_INDEX;
	SparkKvPageCacheUnlinkEntry(cache,entry_index);
	SparkKvPageCacheLruUnlink(cache,entry_index);
	if ( parent != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( parent >= cache->entry_capacity || cache->entries[parent].reference_count == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		SparkKvPageCacheDereference(cache,parent);
	}
	memset(entry,0,sizeof(*entry));
	entry->parent_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	entry->hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->lru_prev = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->lru_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->free_next = cache->free_entry_head;
	cache->free_entry_head = entry_index;
	cache->evicted_entry_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheVacatePage(SparkKvPageCache *cache,uint32_t page)
{
	SparkKvCacheArena *arena = cache->kv_cache_arena;
	const SparkKvCacheBlock *block;
	uint32_t entry_index,discardable = 0u;
	SparkStatus status;
	if ( page >= arena->logical_block_count )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	block = &arena->blocks[page];
	entry_index = cache->entry_indices_by_logical_page[page];
	if ( entry_index < cache->entry_capacity && (cache->entries[entry_index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u &&
		cache->entries[entry_index].reference_count == 0u && SparkKvPageCachePageCanDiscard(cache,page) != 0u )
		discardable = 1u;
	if ( discardable != 0u && SparkKvPageCacheDiscardIsSafe(cache,entry_index) != 0u )
		return(SparkKvPageCacheEvictEntry(cache,entry_index));
	if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u || block->residency_reference_count != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	status = SparkKvCacheArenaParkResidentBlock(arena,page);
	if ( status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY || discardable == 0u )
		SPARK_RETURN(status);
	if ( cache->snapshot != 0 )
		cache->snapshot->evicted_unsaved_count++;
	return(SparkKvPageCacheEvictEntry(cache,entry_index));
}

SparkStatus SparkKvPageCacheVacateResident(SparkKvPageCache *cache,uint32_t limit,uint32_t *kept_limit,uint32_t *vacated_pages)
{
	SparkKvCacheArena *arena;
	uint64_t held;
	uint32_t slot,page;
	if ( SparkKvPageCacheIsValid(cache) == 0u || kept_limit == 0 || vacated_pages == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	arena = cache->kv_cache_arena;
	*vacated_pages = 0u;
	slot = arena->resident_block_capacity;
	while ( slot > limit )
	{
		page = arena->resident_slot_logical_block_indices[slot - 1u];
		if ( page != SPARK_KV_CACHE_NO_BLOCK )
		{
			if ( SparkKvPageCacheVacatePage(cache,page) != SPARK_STATUS_OK || arena->resident_slot_logical_block_indices[slot - 1u] != SPARK_KV_CACHE_NO_BLOCK )
				break;
			(*vacated_pages)++;
		}
		slot--;
	}
	held = (uint64_t)arena->resident_block_count + arena->reserved_block_count + atomic_load(&arena->unassigned_resident_block_count);
	if ( held > slot )
		slot = held < arena->resident_block_capacity ? (uint32_t)held : arena->resident_block_capacity;
	*kept_limit = slot;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvPageCachePageIsProtected(const uint32_t *pages,uint32_t count,uint32_t page)
{
	uint32_t index;
	for (index=0u; index<count; index++)
		if ( pages[index] == page )
			return(1u);
	return(0u);
}

static SparkStatus SparkKvPageCacheEvictUnusedExcept(SparkKvPageCache *cache,const uint32_t *protected_pages,uint32_t protected_count)
{
	uint32_t entry = SparkKvPageCacheSelectVictim(cache,protected_pages,protected_count);
	return(entry == SPARK_KV_PAGE_CACHE_NO_INDEX ? SPARK_STATUS_CAPACITY_EXCEEDED : SparkKvPageCacheEvictEntry(cache,entry));
}

SparkStatus SparkKvPageCacheEvictUnused(SparkKvPageCache *cache)
{
	if ( SparkKvPageCacheIsValid(cache) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageCacheEvictUnusedExcept(cache,0,0u));
}

static SparkStatus SparkKvPageCacheRelieveBacking(SparkKvPageCache *cache,const uint32_t *protected_pages,uint32_t protected_count)
{
	SparkKvCacheArena *arena = cache->kv_cache_arena;
	const SparkKvCacheBlock *block;
	uint32_t slot,page;
	SparkStatus status;
	for (slot=0u; cache->page_store != 0 && slot<arena->resident_block_capacity; slot++)
	{
		page = arena->resident_slot_logical_block_indices[slot];
		if ( page >= arena->logical_block_count )
			continue;
		block = &arena->blocks[page];
		if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
			(block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u )
			continue;
		status = SparkKvPageStoreReleaseIdle(cache->page_store,page,block->generation);
		if ( status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_BUSY )
			continue;
		if ( status == SPARK_STATUS_OK )
			status = SparkKvCacheArenaMarkBlockDirty(arena,page);
		if ( status == SPARK_STATUS_OK )
			cache->backing_reclaim_count++;
		SPARK_RETURN(status);
	}
	return(SparkKvPageCacheEvictUnusedExcept(cache,protected_pages,protected_count));
}

static void SparkKvPageCacheBackingOutcome(SparkKvPageCache *cache,SparkStatus status,uint32_t backing_full)
{
	if ( status == SPARK_STATUS_OK )
	{
		cache->backing_full_logged = 0u;
		return;
	}
	if ( status != SPARK_STATUS_CAPACITY_EXCEEDED || backing_full == 0u )
		return;
	cache->backing_full_count++;
	if ( cache->backing_full_logged == 0u )
		fprintf(stderr,"KV-BACKING-FULL backing_pages=%u capacity=%llu reclaimed=%llu\n",
			cache->page_store != 0 ? cache->page_store->backing_page_count : 0u,
			(unsigned long long)(cache->page_store != 0 ? cache->page_store->logical_page_capacity : 0u),
			(unsigned long long)cache->backing_reclaim_count);
	cache->backing_full_logged = 1u;
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
		entry_index = SparkKvPageCacheSelectVictim(cache,0,0u);
		if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = SparkKvPageCacheEvictEntry(cache,entry_index);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		entry_index = cache->free_entry_head;
	}
	if ( entry_index >= cache->entry_capacity )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	entry = &cache->entries[entry_index];
	cache->free_entry_head = entry->free_next;
	memset(entry,0,sizeof(*entry));
	entry->parent_entry_index = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	entry->hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->free_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->lru_prev = SPARK_KV_PAGE_CACHE_NO_INDEX;
	entry->lru_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
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
	if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX ||
		((cache->entries[entry_index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u && cache->state_store != 0) )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( cache->entries[entry_index].priority < cache->admission_priority )
		cache->entries[entry_index].priority = cache->admission_priority;
	*entry_index_out = entry_index;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvPageCacheSequenceAt(
	const SparkKvPageCacheSequence *sequence,
	uint64_t position)
{
	return(position == sequence->next_token_position ||
		(sequence->rewind_ceiling != 0u && sequence->rewind_ceiling == sequence->next_token_position &&
		 position >= sequence->rewind_floor && position < sequence->next_token_position) ? 1u : 0u);
}

static SparkStatus SparkKvPageCacheValidateExistingSequence(
	const SparkKvPageCache *cache,
	const SparkKvPageCacheSequence *sequence,
	const SparkModelDriverCacheLane *lane,
	uint32_t prefix_entry_index)
{
	if ( sequence->sequence_id != lane->sequence_id ||
		SparkKvPageCacheSequenceAt(sequence,lane->sequence_position) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( prefix_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX &&
		SparkKvPageCacheEntryIsAncestor(cache,sequence->terminal_entry_index,
			prefix_entry_index) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		SPARK_RETURN(status);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id )
	{
		status = SparkKvPageCacheValidateExistingSequence(cache,sequence,lane,
			prefix_entry_index);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		*terminal_entry_index_out = sequence->terminal_entry_index;
		return(SPARK_STATUS_OK);
	}
	if ( prefix_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( lane->sequence_position != 0u )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	else if ( lane->prefix_token_count != lane->sequence_position )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
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
			SPARK_RETURN(status);
		*terminal_entry_index_out = sequence->terminal_entry_index;
		return(SPARK_STATUS_OK);
	}
	if ( prefix_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( lane->sequence_position != 0u )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	else if ( lane->prefix_token_count != lane->sequence_position )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	page_count = terminal_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX ? 0u :
		cache->entries[terminal_entry_index].page_count;
	if ( page_count > logical_page_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	cursor = page_count;
	while ( terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( terminal_entry_index >= cache->entry_capacity || cursor == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		entry = &cache->entries[terminal_entry_index];
		logical_page_indices[--cursor] = entry->logical_page_index;
		terminal_entry_index = entry->parent_entry_index;
	}
	if ( cursor != 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	*page_count_out = page_count;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvPageCacheMutablePage(const SparkKvPageCacheSequence *sequence,uint32_t index)
{
	return(index == 0u ? sequence->mutable_logical_page_index : sequence->mutable_following_pages[index - 1u]);
}

static uint32_t SparkKvPageCacheSpanBlocks(const SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane)
{
	uint32_t block = cache->kv_cache_arena->block_token_count;
	return((lane->context_token_count - 1u) / block - (uint32_t)lane->sequence_position / block + 1u);
}

static SparkStatus SparkKvPageCacheAppendMutablePage(
	const SparkKvPageCache *cache,
	const SparkKvPageCacheSequence *sequence,
	uint32_t *pages,
	uint32_t capacity,
	uint32_t *count)
{
	uint32_t index;
	if ( sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
		return(SPARK_STATUS_OK);
	if ( sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX &&
		cache->entries[sequence->terminal_entry_index].token_count > sequence->mutable_first_token_index )
	{
		if ( *count == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		pages[*count - 1u] = sequence->mutable_logical_page_index;
	}
	else
	{
		if ( *count >= capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		pages[(*count)++] = sequence->mutable_logical_page_index;
	}
	for (index=1u; index<sequence->mutable_page_count; index++)
	{
		if ( *count >= capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		pages[(*count)++] = sequence->mutable_following_pages[index - 1u];
	}
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*logical_page_count_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity ||
		(logical_page_capacity != 0u && logical_page_indices == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChainConst(cache,lane,
		&terminal_entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvPageCacheAppendEntryPages(cache,terminal_entry_index,
		logical_page_indices,logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id )
	{
		status = SparkKvPageCacheAppendMutablePage(cache,sequence,logical_page_indices,logical_page_capacity,&page_count);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
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
	uint32_t first_token,terminal_entry_index,blocks;
	SparkStatus status;
	if ( mutable_page_demand_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*mutable_page_demand_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChainConst(cache,lane,
		&terminal_entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( lane->context_token_count == lane->sequence_position )
		return(SPARK_STATUS_OK);
	first_token = ((uint32_t)lane->sequence_position /
		cache->kv_cache_arena->block_token_count) *
		cache->kv_cache_arena->block_token_count;
	blocks = SparkKvPageCacheSpanBlocks(cache,lane);
	if ( blocks > SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id == lane->sequence_id &&
		sequence->mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
	{
		if ( sequence->mutable_first_token_index != first_token || sequence->mutable_page_count > blocks )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		*mutable_page_demand_out = blocks - sequence->mutable_page_count;
		return(SPARK_STATUS_OK);
	}
	if ( lane->sequence_position != first_token &&
		(terminal_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX || cache->entries[terminal_entry_index].token_count != lane->sequence_position || cache->page_store == 0) )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	*mutable_page_demand_out = blocks;
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*logical_page_count_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity ||
		(logical_page_capacity != 0u && logical_page_indices == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChain(cache,lane,1u,&prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	status = SparkKvPageCacheAppendEntryPages(cache,prefix_entry_index,
		logical_page_indices,logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( sequence->sequence_id == lane->sequence_id )
	{
		status = SparkKvPageCacheAppendMutablePage(cache,sequence,logical_page_indices,logical_page_capacity,&page_count);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	status = SparkKvPageCacheEnsureResidentPages(cache,logical_page_indices,
		page_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*logical_page_count_out = page_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheTrimMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	uint32_t keep)
{
	SparkStatus status;
	while ( sequence->mutable_page_count > keep )
	{
		status = SparkKvPageCacheDiscardLogicalPage(cache,
			SparkKvPageCacheMutablePage(sequence,sequence->mutable_page_count - 1u));
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		sequence->mutable_page_count--;
	}
	if ( sequence->mutable_page_count == 0u )
	{
		sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
		sequence->mutable_first_token_index = 0u;
	}
	sequence->mutable_block_identity_count = 0u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheReleaseMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence)
{
	return(SparkKvPageCacheTrimMutable(cache,sequence,0u));
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = &cache->sequences[resident_sequence_slot];
	if ( sequence->sequence_id == 0u )
		return(SPARK_STATUS_OK);
	if ( sequence->sequence_id != sequence_id )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cache->live_sequence_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvPageCacheReleaseMutable(cache,sequence);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( sequence->terminal_entry_index >= cache->entry_capacity ||
			cache->entries[sequence->terminal_entry_index].reference_count == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		SparkKvPageCacheDereference(cache,sequence->terminal_entry_index);
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
		SPARK_FAIL(SPARK_STATUS_BUSY);
	sequence->generation++;
	if ( sequence->generation == 0u )
		sequence->generation = 1u;
	sequence->sequence_id = lane->sequence_id;
	sequence->next_token_position = (uint32_t)lane->sequence_position;
	sequence->rewind_floor = (uint32_t)lane->sequence_position;
	sequence->rewind_ceiling = 0u;
	sequence->terminal_entry_index = prefix_entry_index;
	sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	sequence->mutable_page_count = 0u;
	if ( prefix_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
		SparkKvPageCacheReference(cache,prefix_entry_index);
	cache->live_sequence_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheAcquireLogicalPage(SparkKvPageCache *cache,uint32_t *logical_page_index)
{
	SparkStatus status;
	status = SparkKvCacheArenaAcquireBlock(cache->kv_cache_arena,logical_page_index);
	if ( status == SPARK_STATUS_CAPACITY_EXCEEDED && SparkKvPageCacheEvictUnused(cache) == SPARK_STATUS_OK )
		status = SparkKvCacheArenaAcquireBlock(cache->kv_cache_arena,logical_page_index);
	return(status);
}

static void SparkKvPageCacheDiscardForBudget(SparkKvPageCache *cache,uint64_t slots_needed);

static SparkStatus SparkKvPageCacheMarkPageResident(SparkKvPageCache *cache,uint32_t logical_page_index)
{
	uint64_t before = cache->kv_cache_arena->park_backing_full_count;
	uint32_t victim,backing_full = 0u;
	SparkStatus status;
	SparkKvPageCacheDiscardForBudget(cache,1u);
	status = SparkKvCacheArenaMarkBlockResident(cache->kv_cache_arena,logical_page_index);
	while ( status == SPARK_STATUS_CAPACITY_EXCEEDED && cache->kv_cache_arena->park_backing_full_count != before )
	{
		backing_full = 1u;
		before = cache->kv_cache_arena->park_backing_full_count;
		if ( SparkKvPageCacheRelieveBacking(cache,&logical_page_index,1u) != SPARK_STATUS_OK )
			break;
		status = SparkKvCacheArenaMarkBlockResident(cache->kv_cache_arena,logical_page_index);
	}
	SparkKvPageCacheBackingOutcome(cache,status,backing_full);
	if ( status != SPARK_STATUS_CAPACITY_EXCEEDED || backing_full != 0u )
		return(status);
	victim = SparkKvPageCacheResidentVictim(cache);
	if ( victim == SPARK_KV_PAGE_CACHE_NO_INDEX || SparkKvPageCacheEvictEntry(cache,victim) != SPARK_STATUS_OK )
		return(status);
	return(SparkKvCacheArenaMarkBlockResident(cache->kv_cache_arena,logical_page_index));
}

static uint32_t SparkKvPageCacheResidentLivePages(const SparkKvPageCache *cache)
{
	const SparkKvCacheArena *arena = cache->kv_cache_arena;
	uint32_t slot,page,count = 0u;
	for (slot=0u; slot<arena->resident_block_capacity; slot++)
	{
		page = arena->resident_slot_logical_block_indices[slot];
		if ( page != SPARK_KV_CACHE_NO_BLOCK && (page >= arena->logical_block_count ||
			cache->entry_indices_by_logical_page[page] == SPARK_KV_PAGE_CACHE_NO_INDEX) )
			count++;
	}
	return(count);
}

static uint64_t SparkKvPageCacheNowNs(void);

static void SparkKvPageCacheDiscardForBudget(SparkKvPageCache *cache,uint64_t slots_needed)
{
	SparkKvCacheArena *arena = cache->kv_cache_arena;
	uint64_t free_slots;
	uint32_t victim;
	if ( cache->write_budget == 0 )
		return;
	for (;;)
	{
		free_slots = arena->resident_block_capacity > arena->resident_block_count ? arena->resident_block_capacity - arena->resident_block_count : 0u;
		if ( free_slots >= slots_needed || SparkKvWriteBudgetAllows(cache->write_budget,cache->page_store->page_bytes,SparkKvPageCacheNowNs()) != 0u )
			return;
		victim = SparkKvPageCacheResidentVictim(cache);
		if ( victim == SPARK_KV_PAGE_CACHE_NO_INDEX || SparkKvPageCacheEvictEntry(cache,victim) != SPARK_STATUS_OK )
			return;
		cache->write_budget->discarded_pages++;
	}
}

static SparkStatus SparkKvPageCacheSpanRoom(SparkKvPageCache *cache,const SparkKvPageCacheSequence *sequence,uint32_t pages)
{
	SparkKvCacheArena *arena = cache->kv_cache_arena;
	uint64_t fixed = (uint64_t)arena->reserved_block_count + atomic_load(&arena->unassigned_resident_block_count) + pages;
	uint32_t held[SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES],index;
	if ( fixed + SparkKvPageCacheResidentLivePages(cache) > arena->resident_block_capacity )
	{
		arena->resident_capacity_stall_count++;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( fixed + arena->resident_block_count <= arena->resident_block_capacity )
		return(SPARK_STATUS_OK);
	uint64_t before = arena->park_backing_full_count;
	uint32_t backing_full = 0u;
	SparkStatus status;
	for (index=0u; index<sequence->mutable_page_count; index++)
		held[index] = SparkKvPageCacheMutablePage(sequence,index);
	SparkKvPageCacheDiscardForBudget(cache,fixed);
	status = SparkKvCacheArenaTrimResidentBlocks(arena,held,sequence->mutable_page_count,
		(uint32_t)(arena->resident_block_capacity - fixed),0);
	while ( status == SPARK_STATUS_CAPACITY_EXCEEDED && arena->park_backing_full_count != before )
	{
		backing_full = 1u;
		before = arena->park_backing_full_count;
		if ( SparkKvPageCacheRelieveBacking(cache,held,sequence->mutable_page_count) != SPARK_STATUS_OK )
			break;
		status = SparkKvCacheArenaTrimResidentBlocks(arena,held,sequence->mutable_page_count,
			(uint32_t)(arena->resident_block_capacity - fixed),0);
	}
	SparkKvPageCacheBackingOutcome(cache,status,backing_full);
	return(status);
}

static SparkStatus SparkKvPageCacheAllocateMutable(SparkKvPageCache *cache,SparkKvPageCacheSequence *sequence,uint32_t first_token_index)
{
	uint32_t logical_page_index;
	SparkStatus status;
	if ( sequence->mutable_page_count >= SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
	status = SparkKvPageCacheAcquireLogicalPage(cache,&logical_page_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaRetainBlock(cache->kv_cache_arena,logical_page_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheMarkPageResident(cache,logical_page_index);
	if ( status != SPARK_STATUS_OK )
	{
		if ( logical_page_index != SPARK_KV_CACHE_NO_BLOCK && logical_page_index < cache->kv_cache_arena->logical_block_count && cache->kv_cache_arena->blocks[logical_page_index].reference_count != 0u )
			(void)SparkKvCacheArenaReleaseBlockReference(cache->kv_cache_arena,logical_page_index);
		if ( logical_page_index != SPARK_KV_CACHE_NO_BLOCK )
			(void)SparkKvCacheArenaFreeBlock(cache->kv_cache_arena,logical_page_index);
		SPARK_RETURN(status);
	}
	if ( sequence->mutable_page_count == 0u )
	{
		sequence->mutable_logical_page_index = logical_page_index;
		sequence->mutable_first_token_index = first_token_index;
	}
	else
		sequence->mutable_following_pages[sequence->mutable_page_count - 1u] = logical_page_index;
	sequence->mutable_page_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheCloneMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	uint32_t first_token)
{
	uint32_t pages[2],physical,pinned = 0u,allocated = 0u;
	SparkStatus status;
	if ( sequence->terminal_entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX ||
		cache->entries[sequence->terminal_entry_index].token_count != sequence->next_token_position )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( cache->device_copy.copy_page == 0 )
	{
		if ( cache->copy_unavailable_logged == 0u )
			fprintf(stderr,"KV-COPY-ON-WRITE-UNAVAILABLE tokens=%u\n",first_token);
		cache->copy_unavailable_logged = 1u;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	pages[0] = cache->entries[sequence->terminal_entry_index].logical_page_index;
	status = SparkKvPageCacheEnsureResidentPages(cache,pages,1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaPinResidentTable(cache->kv_cache_arena,pages,1u,&physical);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pinned = 1u;
	status = SparkKvPageCacheAllocateMutable(cache,sequence,first_token);
	if ( status == SPARK_STATUS_OK )
	{
		allocated = 1u;
		pages[1] = sequence->mutable_logical_page_index;
		status = SparkKvCacheArenaPinResidentTable(cache->kv_cache_arena,pages + 1u,1u,&physical);
		if ( status == SPARK_STATUS_OK )
			pinned++;
	}
	if ( status == SPARK_STATUS_OK )
		status = cache->device_copy.copy_page(cache->device_copy.context,pages[0],pages[1]);
	if ( status == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	(void)SparkKvCacheArenaUnpinResidentTable(cache->kv_cache_arena,pages,pinned);
	if ( allocated != 0u )
		(void)SparkKvPageCacheReleaseMutable(cache,sequence);
	SPARK_RETURN(status);
}

static SparkStatus SparkKvPageCacheBeginLaneInternal(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out,
	uint32_t *mutation_flags_out)
{
	SparkKvPageCacheSequence *sequence;
	uint32_t first_token,mutation_flags,newly_bound,prefix_entry_index,blocks,held,index;
	SparkStatus status;
	if ( mutable_logical_page_index_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*mutable_logical_page_index_out = SPARK_KV_CACHE_NO_BLOCK;
	if ( mutation_flags_out != 0 )
		*mutation_flags_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		SparkModelDriverCacheLaneIsValid(lane) == 0u ||
		(lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE) != 0u ||
		lane->resident_sequence_slot >= cache->sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheResolveLaneChain(cache,lane,0u,&prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	newly_bound = cache->sequences[lane->resident_sequence_slot].sequence_id == 0u ? 1u : 0u;
	status = SparkKvPageCacheBindLane(cache,lane,prefix_entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	mutation_flags = newly_bound != 0u ?
		SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE : 0u;
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( lane->context_token_count == lane->sequence_position )
	{
		if ( lane->block_identity_count != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( mutation_flags_out != 0 )
			*mutation_flags_out = mutation_flags;
		return(SPARK_STATUS_OK);
	}
	first_token = ((uint32_t)lane->sequence_position /
		cache->kv_cache_arena->block_token_count) *
		cache->kv_cache_arena->block_token_count;
	blocks = SparkKvPageCacheSpanBlocks(cache,lane);
	held = sequence->mutable_page_count;
	if ( blocks > SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES || held > blocks ||
		(lane->block_identity_count != 0u && lane->block_identity_count + 1u != blocks) )
		status = blocks > SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES ? SPARK_STATUS_UNSUPPORTED : SPARK_STATUS_INVALID_ARGUMENT;
	else if ( blocks > 1u && held < blocks )
		status = SparkKvPageCacheSpanRoom(cache,sequence,blocks - held);
	if ( status == SPARK_STATUS_OK && sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
	{
		status = lane->sequence_position == first_token ?
			SparkKvPageCacheAllocateMutable(cache,sequence,first_token) :
			SparkKvPageCacheCloneMutable(cache,sequence,first_token);
		if ( status == SPARK_STATUS_OK )
			mutation_flags |= SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE;
	}
	if ( status == SPARK_STATUS_OK &&
		sequence->mutable_first_token_index != first_token )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	while ( status == SPARK_STATUS_OK && sequence->mutable_page_count < blocks )
		status = SparkKvPageCacheAllocateMutable(cache,sequence,first_token);
	for (index=0u; status == SPARK_STATUS_OK && index<sequence->mutable_page_count; index++)
		if ( (cache->kv_cache_arena->blocks[SparkKvPageCacheMutablePage(sequence,index)].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK && held != 0u && sequence->mutable_page_count > held )
		mutation_flags |= SPARK_KV_PAGE_CACHE_MUTATION_EXTENDED_MUTABLE;
	if ( status != SPARK_STATUS_OK )
	{
		if ( sequence->mutable_page_count > held )
			(void)SparkKvPageCacheTrimMutable(cache,sequence,held);
		if ( newly_bound != 0u )
			(void)SparkKvPageCacheReleaseLane(cache,
				lane->resident_sequence_slot,lane->sequence_id);
		SPARK_RETURN(status);
	}
	sequence->mutable_block_identity_count = lane->block_identity_count;
	if ( lane->block_identity_count != 0u )
		memcpy(sequence->mutable_block_identities,lane->block_identities,(uint64_t)lane->block_identity_count * sizeof(sequence->mutable_block_identities[0]));
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( mutation_flags == 0u )
		return(SPARK_STATUS_OK);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id != lane->sequence_id ||
		SparkKvPageCacheSequenceAt(sequence,lane->sequence_position) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u )
		return(SparkKvPageCacheReleaseLane(cache,
			lane->resident_sequence_slot,lane->sequence_id));
	if ( (mutation_flags &
		SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE) != 0u )
		return(SparkKvPageCacheReleaseMutable(cache,sequence));
	if ( (mutation_flags &
		SPARK_KV_PAGE_CACHE_MUTATION_EXTENDED_MUTABLE) != 0u )
		return(SparkKvPageCacheTrimMutable(cache,sequence,1u));
	return(SPARK_STATUS_OK);
}

static void SparkKvPageCacheShiftMutable(const SparkKvPageCache *cache,SparkKvPageCacheSequence *sequence)
{
	uint32_t index;
	if ( sequence->mutable_page_count <= 1u )
	{
		sequence->mutable_logical_page_index = SPARK_KV_CACHE_NO_BLOCK;
		sequence->mutable_first_token_index = 0u;
		sequence->mutable_page_count = 0u;
		return;
	}
	sequence->mutable_logical_page_index = sequence->mutable_following_pages[0];
	for (index=1u; index + 1u < sequence->mutable_page_count; index++)
		sequence->mutable_following_pages[index - 1u] = sequence->mutable_following_pages[index];
	sequence->mutable_first_token_index += cache->kv_cache_arena->block_token_count;
	sequence->mutable_page_count--;
}

static uint32_t SparkKvPageCacheMutableParent(const SparkKvPageCache *cache,const SparkKvPageCacheSequence *sequence)
{
	uint32_t parent = sequence->terminal_entry_index;
	if ( parent != SPARK_KV_PAGE_CACHE_NO_INDEX && cache->entries[parent].token_count > sequence->mutable_first_token_index )
		parent = cache->entries[parent].parent_entry_index;
	return(parent);
}

static SparkStatus SparkKvPageCachePublishNewEntry(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	uint32_t token_count,
	const SparkModelDriverCacheIdentity *identity)
{
	SparkKvPageCacheEntry *entry;
	uint32_t bucket,entry_index,parent;
	SparkStatus status;
	status = SparkKvPageCacheAcquireEntry(cache,&entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	parent = SparkKvPageCacheMutableParent(cache,sequence);
	if ( parent != sequence->terminal_entry_index )
	{
		SparkKvPageCacheDereference(cache,sequence->terminal_entry_index);
		if ( parent != SPARK_KV_PAGE_CACHE_NO_INDEX )
			SparkKvPageCacheReference(cache,parent);
	}
	entry = &cache->entries[entry_index];
	entry->flags = SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID | (identity == 0 ? SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE : 0u);
	entry->token_count = token_count;
	entry->page_count = (token_count - 1u) /
		cache->kv_cache_arena->block_token_count + 1u;
	entry->parent_entry_index = parent;
	entry->logical_page_index = sequence->mutable_logical_page_index;
	entry->reference_count = 1u;
	entry->priority = cache->admission_priority;
	if ( identity != 0 )
		entry->identity = *identity;
	else
		memset(&entry->identity,0,sizeof(entry->identity));
	if ( cache->entry_indices_by_logical_page[entry->logical_page_index] !=
		SPARK_KV_PAGE_CACHE_NO_INDEX )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	cache->entry_indices_by_logical_page[entry->logical_page_index] =
		entry_index;
	cache->epoch++;
	entry->last_used_epoch = cache->epoch;
	entry->hash_next = SPARK_KV_PAGE_CACHE_NO_INDEX;
	if ( identity != 0 )
	{
		bucket = SparkKvPageCacheBucket(cache,&entry->identity,entry->token_count);
		entry->hash_next = cache->hash_bucket_heads[bucket];
		cache->hash_bucket_heads[bucket] = entry_index;
	}
	sequence->terminal_entry_index = entry_index;
	SparkKvPageCacheShiftMutable(cache,sequence);
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
	if ( entry->parent_entry_index != SparkKvPageCacheMutableParent(cache,sequence) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkKvPageCacheReleaseMutable(cache,sequence);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( cache->entries[sequence->terminal_entry_index].reference_count == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		SparkKvPageCacheDereference(cache,sequence->terminal_entry_index);
	}
	SparkKvPageCacheReference(cache,entry_index);
	sequence->terminal_entry_index = entry_index;
	cache->deduplicated_page_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCachePublishMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	const SparkModelDriverCacheLane *lane)
{
	uint32_t entry_index,expected_parent_count,parent;
	SparkStatus status;
	if ( sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK ||
		lane->publish_token_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cache->state_store != 0 )
	{
		status = SparkKvPageStoreValidateRecord(cache->state_store,sequence->mutable_logical_page_index,cache->kv_cache_arena->blocks[sequence->mutable_logical_page_index].generation);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	expected_parent_count = ((lane->publish_token_count - 1u) /
		cache->kv_cache_arena->block_token_count) * cache->kv_cache_arena->block_token_count;
	parent = SparkKvPageCacheMutableParent(cache,sequence);
	if ( sequence->mutable_first_token_index != expected_parent_count ||
		(parent == SPARK_KV_PAGE_CACHE_NO_INDEX) != (expected_parent_count == 0u) ||
		(parent != SPARK_KV_PAGE_CACHE_NO_INDEX && cache->entries[parent].token_count != expected_parent_count) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	entry_index = SparkKvPageCacheFindEntry(cache,&lane->publish_identity,
		lane->publish_token_count,0u);
	if ( entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
		return(SparkKvPageCachePublishDeduplicated(cache,sequence,entry_index));
	return(SparkKvPageCachePublishNewEntry(cache,sequence,lane->publish_token_count,&lane->publish_identity));
}

static SparkStatus SparkKvPageCacheSealDeduplicated(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence,
	uint32_t entry_index)
{
	SparkStatus status;
	if ( cache->entries[entry_index].parent_entry_index != SparkKvPageCacheMutableParent(cache,sequence) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkKvPageCacheDiscardLogicalPage(cache,sequence->mutable_logical_page_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( sequence->terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		if ( cache->entries[sequence->terminal_entry_index].reference_count == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		SparkKvPageCacheDereference(cache,sequence->terminal_entry_index);
	}
	SparkKvPageCacheReference(cache,entry_index);
	sequence->terminal_entry_index = entry_index;
	SparkKvPageCacheShiftMutable(cache,sequence);
	cache->deduplicated_page_count++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheSealMutable(
	SparkKvPageCache *cache,
	SparkKvPageCacheSequence *sequence)
{
	const SparkModelDriverCacheIdentity *identity;
	uint32_t sealed,token_count,entry_index;
	SparkStatus status;
	for (sealed=0u; sequence->mutable_page_count > 1u; sealed++)
	{
		if ( cache->state_store != 0 )
		{
			status = SparkKvPageStoreValidateRecord(cache->state_store,sequence->mutable_logical_page_index,cache->kv_cache_arena->blocks[sequence->mutable_logical_page_index].generation);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		token_count = sequence->mutable_first_token_index + cache->kv_cache_arena->block_token_count;
		identity = sealed < sequence->mutable_block_identity_count ? &sequence->mutable_block_identities[sealed] : 0;
		entry_index = identity != 0 ? SparkKvPageCacheFindEntry(cache,identity,token_count,0u) : SPARK_KV_PAGE_CACHE_NO_INDEX;
		status = entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX ?
			SparkKvPageCacheSealDeduplicated(cache,sequence,entry_index) :
			SparkKvPageCachePublishNewEntry(cache,sequence,token_count,identity);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	sequence->mutable_block_identity_count = 0u;
	return(SPARK_STATUS_OK);
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = &cache->sequences[lane->resident_sequence_slot];
	if ( sequence->sequence_id != lane->sequence_id ||
		SparkKvPageCacheSequenceAt(sequence,lane->sequence_position) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( lane->context_token_count != lane->sequence_position &&
		sequence->mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( lane->context_token_count != lane->sequence_position )
	{
		uint32_t index;
		for (index=0u; index<sequence->mutable_page_count; index++)
		{
			uint32_t page = SparkKvPageCacheMutablePage(sequence,index);
			if ( cache->page_store != 0 && page < cache->kv_cache_arena->logical_block_count )
			{
				status = SparkKvPageStoreSupersede(cache->page_store,page,
					cache->kv_cache_arena->blocks[page].generation);
				if ( status != SPARK_STATUS_OK )
					SPARK_RETURN(status);
			}
			status = SparkKvCacheArenaMarkBlockDirty(cache->kv_cache_arena,page);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		status = SparkKvPageCacheSealMutable(cache,sequence);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) != 0u )
	{
		if ( lane->publish_token_count != lane->context_token_count )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		status = SparkKvPageCachePublishMutable(cache,sequence,lane);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	sequence->next_token_position = lane->context_token_count;
	sequence->rewind_floor = (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_VERIFY) != 0u &&
		lane->context_token_count > lane->sequence_position ?
		(uint32_t)lane->sequence_position + 1u : lane->context_token_count;
	sequence->rewind_ceiling = (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_VERIFY) != 0u ? lane->context_token_count : 0u;
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*logical_page_count_out = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u ||
		resident_sequence_slot >= cache->sequence_capacity || sequence_id == 0u ||
		(logical_page_capacity != 0u && logical_page_indices == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	sequence = &cache->sequences[resident_sequence_slot];
	if ( sequence->sequence_id != sequence_id )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	status = SparkKvPageCacheAppendEntryPages(cache,
		sequence->terminal_entry_index,logical_page_indices,
		logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvPageCacheAppendMutablePage(cache,sequence,logical_page_indices,logical_page_capacity,&page_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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
		SPARK_RETURN(status);
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
	uint32_t prepared_count,page_count,pinned_count,prepared_last = SPARK_KV_CACHE_NO_BLOCK,mutable_page,mutation_flags = 0u;
	SparkStatus status;
	if ( page_count_out == 0 || mutation_flags_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*page_count_out = 0u;
	*mutation_flags_out = 0u;
	if ( logical_pages == 0 || physical_pages == 0 || page_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCachePrepareLane(cache,lane,logical_pages,page_capacity,&prepared_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvCacheArenaPinResidentTable(cache->kv_cache_arena,logical_pages,prepared_count,physical_pages);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pinned_count = prepared_count;
	if ( prepared_count != 0u )
		prepared_last = logical_pages[prepared_count - 1u];
	status = SparkKvPageCacheBeginLaneTransaction(cache,lane,&mutable_page,&mutation_flags);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheBuildLaneTable(cache,lane->resident_sequence_slot,lane->sequence_id,logical_pages,page_capacity,&page_count);
	if ( status == SPARK_STATUS_OK && prepared_count != 0u && logical_pages[prepared_count - 1u] != prepared_last )
	{
		status = SparkKvCacheArenaUnpinResidentTable(cache->kv_cache_arena,&prepared_last,1u);
		if ( status == SPARK_STATUS_OK )
		{
			prepared_count--;
			pinned_count--;
		}
		else
			logical_pages[prepared_count - 1u] = prepared_last;
	}
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

static uint64_t SparkKvPageCacheNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

SparkStatus SparkKvPageCacheAttachDeviceCopy(SparkKvPageCache *cache,const SparkKvPageCacheDeviceCopy *copy)
{
	if ( SparkKvPageCacheIsValid(cache) == 0u || copy == 0 || copy->copy_page == 0 || copy->retire_copies == 0 ||
		copy->destination_pins == 0 || copy->defer_free == 0 || cache->device_copy.copy_page != 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cache->live_sequence_count != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	cache->device_copy = *copy;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheAttachSnapshot(SparkKvPageCache *cache,SparkKvPageCacheSnapshot *snapshot)
{
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->page_store == 0 || cache->snapshot != 0 || snapshot == 0 || snapshot->store == 0 || snapshot->store->abi_version != SPARK_KV_SNAPSHOT_ABI_VERSION || snapshot->page_capacity == 0u || snapshot->reserved0 != 0u || snapshot->links == 0 || snapshot->page == 0 || snapshot->pending_terminals == 0 || snapshot->pending_capacity == 0u || (cache->state_store != 0 && snapshot->state == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	cache->snapshot = snapshot;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheSnapshotCopy(SparkKvPageStore *store,uint32_t direction,uintptr_t device_address,uint8_t *host,uint64_t bytes)
{
	if ( bytes == 0u )
		return(SPARK_STATUS_OK);
	if ( device_address == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( store->copy_function != 0 )
		return(store->copy_function(store->copy_context,direction,device_address,host,bytes));
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		memcpy(host,(const void *)device_address,(size_t)bytes);
	else
		memcpy((void *)device_address,host,(size_t)bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheSnapshotPage(SparkKvPageCache *cache,uint32_t direction,uint32_t logical_page_index,uint8_t *host)
{
	SparkKvCacheBlockView view;
	SparkStatus status;
	status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,logical_page_index,&view);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( view.key_block_stride_bytes + view.value_block_stride_bytes != cache->page_store->page_bytes ||
		(view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u || direction != SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvPageCacheSnapshotCopy(cache->page_store,direction,view.key_device_address,host,view.key_block_stride_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheSnapshotCopy(cache->page_store,direction,view.value_device_address,host + view.key_block_stride_bytes,view.value_block_stride_bytes);
	SPARK_RETURN(status);
}

static SparkStatus SparkKvPageCacheWriteState(SparkKvPageCache *cache,uint32_t logical_page_index,const uint8_t *host)
{
	const SparkKvCacheBlock *block = &cache->kv_cache_arena->blocks[logical_page_index];
	SparkStatus status = SparkKvPageStoreWriteback(cache->state_store,logical_page_index,block->resident_slot_index,block->generation,(uintptr_t)host,cache->state_store->page_bytes,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(cache->state_store);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreWriteback(cache->state_store,logical_page_index,block->resident_slot_index,block->generation,(uintptr_t)host,cache->state_store->page_bytes,0u,0u);
	}
	SPARK_RETURN(status);
}

static void SparkKvPageCacheSnapshotKey(const SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count,SparkKvSnapshotKey *key)
{
	memset(key,0,sizeof(*key));
	memcpy(key->layout_sha256,cache->snapshot->layout_sha256,sizeof(key->layout_sha256));
	memcpy(key->identity_sha256,identity->sha256,sizeof(key->identity_sha256));
	key->token_count = token_count;
}

static void SparkKvPageCacheSaveLog(SparkKvPageCache *cache,SparkStatus status,uint32_t terminal)
{
	SparkKvPageCacheSnapshot *snapshot = cache->snapshot;
	snapshot->save_failure_count++;
	if ( status != snapshot->last_save_status )
		fprintf(stderr,"KV-SNAPSHOT save status=%d tokens=%u failures=%llu\n",(int)status,terminal < cache->entry_capacity ? cache->entries[terminal].token_count : 0u,(unsigned long long)snapshot->save_failure_count);
	snapshot->last_save_status = status;
}

static uint32_t SparkKvPageCacheSaveIsPending(const SparkKvPageCacheSnapshot *snapshot,uint32_t terminal)
{
	uint32_t index;
	for (index=0u; index<snapshot->pending_count; index++)
		if ( snapshot->pending_terminals[(snapshot->pending_head + index) % snapshot->pending_capacity] == terminal )
			return(1u);
	return(0u);
}

static SparkStatus SparkKvPageCacheSavePush(SparkKvPageCache *cache,uint32_t terminal)
{
	SparkKvPageCacheSnapshot *snapshot = cache->snapshot;
	if ( snapshot->pending_count == snapshot->pending_capacity )
	{
		snapshot->save_skipped_count++;
		if ( snapshot->full_logged == 0u )
			fprintf(stderr,"KV-SNAPSHOT save queue full pending=%u skipped=%llu\n",snapshot->pending_count,(unsigned long long)snapshot->save_skipped_count);
		snapshot->full_logged = 1u;
		SPARK_FAIL(SPARK_STATUS_BUSY);
	}
	SparkKvPageCacheReference(cache,terminal);
	snapshot->pending_terminals[(snapshot->pending_head + snapshot->pending_count) % snapshot->pending_capacity] = terminal;
	snapshot->pending_count++;
	snapshot->save_mark_count++;
	return(SPARK_STATUS_OK);
}

static void SparkKvPageCacheSavePop(SparkKvPageCache *cache)
{
	SparkKvPageCacheSnapshot *snapshot = cache->snapshot;
	uint32_t terminal = snapshot->pending_terminals[snapshot->pending_head];
	snapshot->pending_head = (snapshot->pending_head + 1u) % snapshot->pending_capacity;
	snapshot->pending_count--;
	snapshot->head_start_ns = 0u;
	snapshot->head_saved_pages = 0u;
	snapshot->full_logged = 0u;
	SparkKvPageCacheDereference(cache,terminal);
}

static void SparkKvPageCacheMarkSave(SparkKvPageCache *cache,uint32_t resident_sequence_slot)
{
	uint32_t terminal = cache->sequences[resident_sequence_slot].terminal_entry_index;
	if ( cache->snapshot == 0 || terminal == SPARK_KV_PAGE_CACHE_NO_INDEX )
		return;
	if ( terminal >= cache->entry_capacity )
	{
		SparkKvPageCacheSaveLog(cache,SPARK_STATUS_INTERNAL_ERROR,terminal);
		return;
	}
	if ( (cache->entries[terminal].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u && cache->state_store != 0 )
		return;
	if ( (cache->entries[terminal].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED) != 0u || SparkKvPageCacheSaveIsPending(cache->snapshot,terminal) != 0u )
		return;
	(void)SparkKvPageCacheSavePush(cache,terminal);
}

static SparkStatus SparkKvPageCacheSaveCheck(const SparkKvPageCache *cache,uint32_t entry_index)
{
	const SparkKvPageCacheEntry *entry = &cache->entries[entry_index];
	uint32_t index;
	if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u && cache->state_store != 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	for (index=entry_index; index != SPARK_KV_PAGE_CACHE_NO_INDEX && index < cache->entry_capacity; index=cache->entries[index].parent_entry_index)
		if ( (cache->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE) != 0u )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( entry->page_count == 0u || entry->page_count > cache->snapshot->page_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvPageCacheSaveEligible(const SparkKvPageCache *cache,uint32_t entry_index)
{
	const SparkKvPageCacheEntry *entry = &cache->entries[entry_index];
	uint32_t index;
	if ( (entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u && cache->state_store != 0 )
		return(0u);
	for (index=entry_index; index != SPARK_KV_PAGE_CACHE_NO_INDEX && index < cache->entry_capacity; index=cache->entries[index].parent_entry_index)
		if ( (cache->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE) != 0u )
			return(0u);
	return(entry->page_count != 0u && entry->page_count <= cache->snapshot->page_capacity ? 1u : 0u);
}

static int SparkKvPageCacheSaveOrderCompare(const void *left,const void *right)
{
	const SparkKvPageCacheSaveOrder *a = left,*b = right;
	if ( a->last_used_epoch != b->last_used_epoch )
		return(a->last_used_epoch > b->last_used_epoch ? -1 : 1);
	return(a->entry_index < b->entry_index ? -1 : (a->entry_index > b->entry_index ? 1 : 0));
}

static uint32_t SparkKvPageCacheEntryUnsaved(const SparkKvPageCache *cache,uint32_t entry_index)
{
	uint32_t flags = cache->entries[entry_index].flags;
	return((flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && (flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED) == 0u ? 1u : 0u);
}

uint32_t SparkKvPageCacheSaveParked(SparkKvPageCache *cache,uint32_t logical_page_index)
{
	uint32_t entry_index;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || logical_page_index >= cache->kv_cache_arena->logical_block_count )
		return(0u);
	entry_index = cache->entry_indices_by_logical_page[logical_page_index];
	if ( entry_index >= cache->entry_capacity || SparkKvPageCacheEntryUnsaved(cache,entry_index) == 0u ||
		((cache->entries[entry_index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u && cache->state_store != 0) ||
		SparkKvPageCacheSaveIsPending(cache->snapshot,entry_index) != 0u || SparkKvPageCacheSaveEligible(cache,entry_index) == 0u ||
		cache->snapshot->pending_count == cache->snapshot->pending_capacity || SparkKvPageCacheSavePush(cache,entry_index) != SPARK_STATUS_OK )
		return(0u);
	cache->snapshot->park_save_queued_count++;
	return(1u);
}

SparkStatus SparkKvPageCacheMarkAllUnsaved(SparkKvPageCache *cache,SparkKvPageCacheSaveOrder *order,uint32_t order_capacity,uint32_t *marked_out,uint32_t *deferred_out,uint32_t *ineligible_out)
{
	uint32_t entry,count = 0u,index,marked = 0u,deferred = 0u,ineligible = 0u;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || order == 0 || order_capacity < cache->entry_capacity || marked_out == 0 || deferred_out == 0 || ineligible_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (entry=0u; entry<cache->entry_capacity; entry++)
	{
		if ( SparkKvPageCacheEntryUnsaved(cache,entry) == 0u || SparkKvPageCacheSaveIsPending(cache->snapshot,entry) != 0u )
			continue;
		if ( SparkKvPageCacheSaveEligible(cache,entry) == 0u )
		{
			ineligible++;
			continue;
		}
		order[count++] = (SparkKvPageCacheSaveOrder){.last_used_epoch=cache->entries[entry].last_used_epoch,.entry_index=entry};
	}
	qsort(order,count,sizeof(order[0]),SparkKvPageCacheSaveOrderCompare);
	for (index=0u; index<count; index++)
	{
		if ( SparkKvPageCacheSaveIsPending(cache->snapshot,order[index].entry_index) != 0u || SparkKvPageCacheEntryUnsaved(cache,order[index].entry_index) == 0u )
			continue;
		if ( cache->snapshot->pending_count == cache->snapshot->pending_capacity )
		{
			deferred += count - index;
			break;
		}
		status = SparkKvPageCacheSavePush(cache,order[index].entry_index);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		marked++;
	}
	*marked_out = marked;
	*deferred_out = deferred;
	*ineligible_out = ineligible;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheCountUnsaved(const SparkKvPageCache *cache,uint32_t *unsaved_out,uint32_t *ineligible_out)
{
	uint32_t entry,unsaved = 0u,ineligible = 0u;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || unsaved_out == 0 || ineligible_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (entry=0u; entry<cache->entry_capacity; entry++)
	{
		if ( SparkKvPageCacheEntryUnsaved(cache,entry) == 0u )
			continue;
		if ( SparkKvPageCacheSaveEligible(cache,entry) != 0u )
			unsaved++;
		else
			ineligible++;
	}
	*unsaved_out = unsaved;
	*ineligible_out = ineligible;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheSaveLinks(const SparkKvPageCache *cache,uint32_t entry_index,SparkKvPageCacheSnapshotLink *links)
{
	uint32_t cursor = cache->entries[entry_index].page_count,index;
	for (index=entry_index; index != SPARK_KV_PAGE_CACHE_NO_INDEX; index=cache->entries[index].parent_entry_index)
	{
		if ( index >= cache->entry_capacity || cursor == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		links[--cursor] = (SparkKvPageCacheSnapshotLink){.token_count=cache->entries[index].token_count,.identity=cache->entries[index].identity};
	}
	return(cursor == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR);
}

static SparkStatus SparkKvPageCacheSaveSource(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work)
{
	SparkKvCacheBlockView view;
	SparkStatus status = SparkKvCacheArenaResolveBlock(cache->kv_cache_arena,work->logical_page_index,&view);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( view.key_block_stride_bytes + view.value_block_stride_bytes != cache->page_store->page_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	work->generation = view.generation;
	if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
	{
		status = SparkKvCacheArenaPinResidentBlock(cache->kv_cache_arena,work->logical_page_index);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		work->pinned = 1u;
		work->source = SPARK_KV_PAGE_CACHE_SAVE_DEVICE;
		work->key_device_address = view.key_device_address;
		work->key_bytes = view.key_block_stride_bytes;
		work->value_device_address = view.value_device_address;
		work->value_bytes = view.value_block_stride_bytes;
		return(SPARK_STATUS_OK);
	}
	if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) == 0u )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	work->source = SPARK_KV_PAGE_CACHE_SAVE_BACKING;
	return(SPARK_STATUS_OK);
}

static void SparkKvPageCacheSaveDropHead(SparkKvPageCache *cache,SparkStatus status)
{
	SparkKvPageCacheSnapshot *snapshot = cache->snapshot;
	SparkKvPageCacheSaveLog(cache,status,snapshot->pending_terminals[snapshot->pending_head]);
	SparkKvPageCacheSavePop(cache);
}

static uint32_t SparkKvPageCacheSaveUnsavedEntry(const SparkKvPageCache *cache,uint32_t terminal)
{
	uint32_t index;
	for (index=terminal; index != SPARK_KV_PAGE_CACHE_NO_INDEX && index < cache->entry_capacity; index=cache->entries[index].parent_entry_index)
		if ( (cache->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED) == 0u )
			return(index);
	return(SPARK_KV_PAGE_CACHE_NO_INDEX);
}

static SparkStatus SparkKvPageCacheSaveBegin(SparkKvPageCache *cache,uint32_t entry_index,SparkKvPageCacheSaveWork *work)
{
	const SparkKvPageCacheEntry *entry = &cache->entries[entry_index];
	uint32_t kinds[3],segment_count = cache->state_store != 0 ? 3u : 2u;
	uint64_t bytes[3];
	SparkKvSnapshotKey key;
	kinds[0] = SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_CHAIN;
	kinds[1] = SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES;
	kinds[2] = SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_STATE;
	bytes[0] = (uint64_t)entry->page_count * sizeof(SparkKvPageCacheSnapshotLink);
	bytes[1] = cache->page_store->page_bytes;
	bytes[2] = cache->state_store != 0 ? cache->state_store->page_bytes : 0u;
	SparkKvPageCacheSnapshotKey(cache,&entry->identity,entry->token_count,&key);
	return(SparkKvSnapshotWriteBegin(cache->snapshot->store,&key,kinds,bytes,segment_count,&work->ticket));
}

SparkStatus SparkKvPageCacheSaveTake(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work)
{
	SparkKvPageCacheSnapshot *snapshot;
	uint32_t terminal,entry_index;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || work == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	snapshot = cache->snapshot;
	if ( snapshot->in_flight != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	while ( snapshot->pending_count != 0u )
	{
		terminal = snapshot->pending_terminals[snapshot->pending_head];
		if ( snapshot->head_start_ns == 0u )
		{
			snapshot->head_start_ns = SparkKvPageCacheNowNs();
			snapshot->head_saved_pages = 0u;
		}
		entry_index = SparkKvPageCacheSaveUnsavedEntry(cache,terminal);
		if ( entry_index == SPARK_KV_PAGE_CACHE_NO_INDEX )
		{
			if ( snapshot->head_saved_pages != 0u )
			{
				snapshot->save_count++;
				snapshot->save_ns += SparkKvPageCacheNowNs() - snapshot->head_start_ns;
			}
			SparkKvPageCacheSavePop(cache);
			continue;
		}
		status = SparkKvPageCacheSaveCheck(cache,entry_index);
		if ( status != SPARK_STATUS_OK )
		{
			SparkKvPageCacheSaveDropHead(cache,status);
			continue;
		}
		if ( cache->write_budget != 0 && SparkKvWriteBudgetAllows(cache->write_budget,cache->page_store->page_bytes,SparkKvPageCacheNowNs()) == 0u )
		{
			cache->write_budget->refused_saves++;
			SparkKvPageCacheSavePop(cache);
			continue;
		}
		memset(work,0,sizeof(*work));
		status = SparkKvPageCacheSaveBegin(cache,entry_index,work);
		if ( status == SPARK_STATUS_DUPLICATE )
		{
			cache->entries[entry_index].flags |= SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED;
			continue;
		}
		if ( status == SPARK_STATUS_BUSY )
		{
			snapshot->save_deferred_count++;
			SPARK_FAIL(SPARK_STATUS_BUSY);
		}
		if ( status != SPARK_STATUS_OK )
		{
			SparkKvPageCacheSaveDropHead(cache,status);
			continue;
		}
		work->terminal_entry_index = terminal;
		work->entry_index = entry_index;
		work->logical_page_index = cache->entries[entry_index].logical_page_index;
		status = SparkKvPageCacheSaveLinks(cache,entry_index,(SparkKvPageCacheSnapshotLink *)work->ticket.segments[0].data);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageCacheSaveSource(cache,work);
		if ( status != SPARK_STATUS_OK )
		{
			SparkKvSnapshotWriteCancel(snapshot->store,&work->ticket);
			SparkKvPageCacheSaveDropHead(cache,status);
			continue;
		}
		work->page = (uint8_t *)work->ticket.segments[1].data;
		work->state = cache->state_store != 0 ? (uint8_t *)work->ticket.segments[2].data : 0;
		if ( cache->write_budget != 0 )
			SparkKvWriteBudgetCharge(cache->write_budget,work->ticket.file_bytes,SparkKvPageCacheNowNs());
		snapshot->in_flight = 1u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkKvPageCacheSaveFinish(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work,SparkStatus copy_status)
{
	SparkKvPageCacheSnapshot *snapshot;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || work == 0 || cache->snapshot->in_flight == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	snapshot = cache->snapshot;
	status = SPARK_STATUS_OK;
	if ( work->pinned != 0u )
		status = SparkKvCacheArenaUnpinResidentBlock(cache->kv_cache_arena,work->logical_page_index);
	work->pinned = 0u;
	if ( status == SPARK_STATUS_OK )
		status = copy_status;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotWriteCommit(snapshot->store,&work->ticket);
	else
		SparkKvSnapshotWriteCancel(snapshot->store,&work->ticket);
	snapshot->in_flight = 0u;
	if ( status != SPARK_STATUS_OK )
	{
		if ( snapshot->pending_count != 0u && snapshot->pending_terminals[snapshot->pending_head] == work->terminal_entry_index )
			SparkKvPageCacheSaveDropHead(cache,status);
		else
			SparkKvPageCacheSaveLog(cache,status,work->terminal_entry_index);
		SPARK_RETURN(status);
	}
	cache->entries[work->entry_index].flags |= SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED;
	snapshot->save_page_count++;
	snapshot->head_saved_pages++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheReadBacking(SparkKvPageStore *store,uint32_t logical_page_index,uint64_t generation,uint8_t *host)
{
	SparkStatus status = SparkKvPageStoreReadback(store,logical_page_index,generation,(uintptr_t)host,store->page_bytes);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(store);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreReadback(store,logical_page_index,generation,(uintptr_t)host,store->page_bytes);
	}
	SPARK_RETURN(status);
}

SparkStatus SparkKvPageCacheSaveCopy(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work)
{
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u || work == 0 || work->page == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( work->source == SPARK_KV_PAGE_CACHE_SAVE_DEVICE )
	{
		status = SparkKvPageCacheSnapshotCopy(cache->page_store,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,work->key_device_address,work->page,work->key_bytes);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageCacheSnapshotCopy(cache->page_store,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,work->value_device_address,work->page + work->key_bytes,work->value_bytes);
	}
	else
		status = SparkKvPageCacheReadBacking(cache->page_store,work->logical_page_index,work->generation,work->page);
	if ( status == SPARK_STATUS_OK && work->state != 0 )
		status = SparkKvPageCacheReadBacking(cache->state_store,work->logical_page_index,work->generation,work->state);
	SPARK_RETURN(status);
}

SparkStatus SparkKvPageCacheSaveDrain(SparkKvPageCache *cache)
{
	SparkKvPageCacheSaveWork work;
	SparkStatus status;
	for (;;)
	{
		status = SparkKvPageCacheSaveTake(cache,&work);
		if ( status == SPARK_STATUS_NOT_FOUND )
			return(SPARK_STATUS_OK);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		(void)SparkKvPageCacheSaveFinish(cache,&work,SparkKvPageCacheSaveCopy(cache,&work));
	}
}

uint32_t SparkKvPageCacheSavePending(const SparkKvPageCache *cache)
{
	return(cache != 0 && cache->snapshot != 0 ? cache->snapshot->pending_count : 0u);
}

void SparkKvPageCacheSaveCancelAll(SparkKvPageCache *cache)
{
	uint32_t count;
	if ( cache == 0 || cache->snapshot == 0 || cache->snapshot->in_flight != 0u )
		return;
	count = cache->snapshot->pending_count;
	while ( cache->snapshot->pending_count != 0u )
		SparkKvPageCacheSavePop(cache);
	cache->snapshot->save_cancelled_count += count;
}

SparkStatus SparkKvPageCacheSavePrefix(SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	uint32_t entry;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || identity == 0 || token_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	entry = SparkKvPageCacheFindEntryConst(cache,identity,token_count);
	if ( entry == SPARK_KV_PAGE_CACHE_NO_INDEX )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( (cache->entries[entry].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED) == 0u && SparkKvPageCacheSaveIsPending(cache->snapshot,entry) == 0u )
	{
		status = SparkKvPageCacheSavePush(cache,entry);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SparkKvPageCacheSaveDrain(cache));
}

static SparkStatus SparkKvPageCacheValidateChain(const SparkKvPageCache *cache,const SparkKvPageCacheSnapshotLink *links,uint32_t page_count,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	uint32_t index,block = cache->kv_cache_arena->block_token_count;
	if ( page_count == 0u || links[page_count - 1u].token_count != token_count || memcmp(&links[page_count - 1u].identity,identity,sizeof(*identity)) != 0 )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	for (index=0u; index<page_count; index++)
	{
		if ( links[index].reserved0 != 0u || links[index].token_count <= index * block || links[index].token_count > (index + 1u) * block || (index + 1u < page_count && links[index].token_count != (index + 1u) * block) )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheImportPage(SparkKvPageCache *cache,const SparkKvPageCacheSnapshotLink *link,uint32_t parent,uint32_t page_count,const uint8_t *payload,const uint8_t *state,uint32_t *entry_index_out)
{
	SparkKvPageCacheEntry *entry;
	uint32_t entry_index,page = SPARK_KV_CACHE_NO_BLOCK,bucket;
	SparkStatus status;
	status = SparkKvPageCacheAcquireEntry(cache,&entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvPageCacheAcquireLogicalPage(cache,&page);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaRetainBlock(cache->kv_cache_arena,page);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheMarkPageResident(cache,page);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheSnapshotPage(cache,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,page,(uint8_t *)payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaMarkBlockDirty(cache->kv_cache_arena,page);
	if ( status == SPARK_STATUS_OK && state != 0 )
		status = SparkKvPageCacheWriteState(cache,page,state);
	if ( status == SPARK_STATUS_OK && cache->entry_indices_by_logical_page[page] != SPARK_KV_PAGE_CACHE_NO_INDEX )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		if ( page != SPARK_KV_CACHE_NO_BLOCK && cache->kv_cache_arena->blocks[page].reference_count != 0u )
			(void)SparkKvCacheArenaReleaseBlockReference(cache->kv_cache_arena,page);
		if ( page != SPARK_KV_CACHE_NO_BLOCK )
			(void)SparkKvCacheArenaFreeBlock(cache->kv_cache_arena,page);
		cache->entries[entry_index].free_next = cache->free_entry_head;
		cache->free_entry_head = entry_index;
		SPARK_RETURN(status);
	}
	entry = &cache->entries[entry_index];
	entry->flags = SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID | (state == 0 && cache->state_store != 0 ? SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS : 0u);
	entry->token_count = link->token_count;
	entry->page_count = page_count;
	entry->parent_entry_index = parent;
	entry->logical_page_index = page;
	entry->reference_count = 1u;
	entry->identity = link->identity;
	cache->entry_indices_by_logical_page[page] = entry_index;
	cache->epoch++;
	entry->last_used_epoch = cache->epoch;
	bucket = SparkKvPageCacheBucket(cache,&entry->identity,entry->token_count);
	entry->hash_next = cache->hash_bucket_heads[bucket];
	cache->hash_bucket_heads[bucket] = entry_index;
	cache->published_page_count++;
	*entry_index_out = entry_index;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvPageCacheRestoreOutcome(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkStatus status,SparkKvPageCacheRestoreJob *job)
{
	if ( status == SPARK_STATUS_OK || status == SPARK_STATUS_NOT_FOUND || status == SPARK_STATUS_HASH_MISMATCH )
		return(status);
	(void)SparkKvSnapshotRemove(store,key);
	if ( status != SPARK_STATUS_IO_ERROR )
		return(SPARK_STATUS_HASH_MISMATCH);
	job->read_errors++;
	return(SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkKvPageCacheRestoreBegin(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	uint32_t existing;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 || job == 0 || identity == 0 || token_count == 0u || job->links == 0 || job->page == 0 ||
		(cache->state_store != 0 && job->state == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	job->identity = *identity;
	job->token_count = token_count;
	job->page_count = 0u;
	job->next_page = 0u;
	job->parent = SPARK_KV_PAGE_CACHE_NO_INDEX;
	job->need = SPARK_KV_PAGE_CACHE_RESTORE_DONE;
	job->imported_pages = 0u;
	job->read_errors = 0u;
	job->start_ns = SparkKvPageCacheNowNs();
	existing = SparkKvPageCacheFindEntryConst(cache,identity,token_count);
	if ( existing != SPARK_KV_PAGE_CACHE_NO_INDEX && ((cache->entries[existing].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) == 0u || cache->state_store == 0) )
		return(SPARK_STATUS_OK);
	return(SPARK_STATUS_PENDING);
}

uint32_t SparkKvPageCachePrefixReady(const SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	uint32_t existing = SparkKvPageCacheFindEntryConst(cache,identity,token_count);
	return(existing != SPARK_KV_PAGE_CACHE_NO_INDEX && ((cache->entries[existing].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) == 0u || cache->state_store == 0) ? 1u : 0u);
}

SparkStatus SparkKvPageCacheRestoreReadChain(const SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job)
{
	SparkKvSnapshotKey key;
	uint64_t bytes = 0u;
	SparkStatus status;
	SparkKvPageCacheSnapshotKey(cache,&job->identity,job->token_count,&key);
	status = SparkKvSnapshotReadSegment(cache->snapshot->store,&key,0u,SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_CHAIN,job->links,(uint64_t)cache->snapshot->page_capacity * sizeof(job->links[0]),&bytes);
	if ( status == SPARK_STATUS_OK && (bytes == 0u || bytes % sizeof(job->links[0]) != 0u) )
		status = SPARK_STATUS_VALIDATION_FAILED;
	job->page_count = (uint32_t)(bytes / sizeof(job->links[0]));
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheValidateChain(cache,job->links,job->page_count,&job->identity,job->token_count);
	return(SparkKvPageCacheRestoreOutcome(cache->snapshot->store,&key,status,job));
}

SparkStatus SparkKvPageCacheRestoreAdvance(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job)
{
	const SparkKvPageCacheSnapshotLink *link;
	uint32_t existing;
	while ( job->next_page < job->page_count )
	{
		link = &job->links[job->next_page];
		existing = SparkKvPageCacheFindEntryConst(cache,&link->identity,link->token_count);
		if ( existing == SPARK_KV_PAGE_CACHE_NO_INDEX )
			break;
		if ( cache->entries[existing].parent_entry_index != job->parent )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		SparkKvPageCacheReference(cache,existing);
		if ( job->parent != SPARK_KV_PAGE_CACHE_NO_INDEX )
			SparkKvPageCacheDereference(cache,job->parent);
		job->parent = existing;
		job->next_page++;
	}
	if ( job->next_page < job->page_count )
		job->need = SPARK_KV_PAGE_CACHE_RESTORE_NEED_PAGE;
	else if ( job->parent != SPARK_KV_PAGE_CACHE_NO_INDEX && cache->state_store != 0 && (cache->entries[job->parent].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u )
		job->need = SPARK_KV_PAGE_CACHE_RESTORE_NEED_STATE;
	else
		job->need = SPARK_KV_PAGE_CACHE_RESTORE_DONE;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheRestoreRead(const SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job)
{
	const SparkKvPageCacheSnapshotLink *link;
	SparkKvSnapshotKey key;
	uint64_t bytes = 0u;
	uint32_t page_wanted,state_wanted;
	SparkStatus status = SPARK_STATUS_OK;
	if ( job->need == SPARK_KV_PAGE_CACHE_RESTORE_DONE || job->page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	page_wanted = job->need == SPARK_KV_PAGE_CACHE_RESTORE_NEED_PAGE ? 1u : 0u;
	link = &job->links[page_wanted != 0u ? job->next_page : job->page_count - 1u];
	state_wanted = cache->state_store != 0 && (page_wanted == 0u || job->next_page + 1u == job->page_count) ? 1u : 0u;
	SparkKvPageCacheSnapshotKey(cache,&link->identity,link->token_count,&key);
	if ( page_wanted != 0u )
	{
		status = SparkKvSnapshotReadSegment(cache->snapshot->store,&key,1u,SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES,job->page,cache->page_store->page_bytes,&bytes);
		if ( status == SPARK_STATUS_OK && bytes != cache->page_store->page_bytes )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status == SPARK_STATUS_OK && state_wanted != 0u )
	{
		status = SparkKvSnapshotReadSegment(cache->snapshot->store,&key,2u,SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_STATE,job->state,cache->state_store->page_bytes,&bytes);
		if ( status == SPARK_STATUS_OK && bytes != cache->state_store->page_bytes )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	return(SparkKvPageCacheRestoreOutcome(cache->snapshot->store,&key,status,job));
}

SparkStatus SparkKvPageCacheRestoreApply(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job)
{
	const SparkKvPageCacheSnapshotLink *link;
	uint32_t created,with_state;
	SparkStatus status;
	if ( job->need == SPARK_KV_PAGE_CACHE_RESTORE_NEED_STATE )
	{
		if ( job->parent == SPARK_KV_PAGE_CACHE_NO_INDEX || (cache->entries[job->parent].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) == 0u )
			return(SPARK_STATUS_OK);
		status = SparkKvPageCacheWriteState(cache,cache->entries[job->parent].logical_page_index,job->state);
		if ( status == SPARK_STATUS_OK )
			cache->entries[job->parent].flags &= ~SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS;
		SPARK_RETURN(status);
	}
	if ( job->need != SPARK_KV_PAGE_CACHE_RESTORE_NEED_PAGE || job->next_page >= job->page_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	link = &job->links[job->next_page];
	if ( SparkKvPageCacheFindEntryConst(cache,&link->identity,link->token_count) != SPARK_KV_PAGE_CACHE_NO_INDEX )
		return(SPARK_STATUS_OK);
	with_state = job->next_page + 1u == job->page_count && cache->state_store != 0 ? 1u : 0u;
	status = SparkKvPageCacheImportPage(cache,link,job->parent,job->next_page + 1u,job->page,with_state != 0u ? job->state : 0,&created);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	cache->entries[created].flags |= SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED;
	job->parent = created;
	job->next_page++;
	job->imported_pages++;
	cache->snapshot->restore_page_count++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheRestoreFinish(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job,SparkStatus status)
{
	SparkKvPageCacheSnapshot *snapshot = cache->snapshot;
	if ( job->parent != SPARK_KV_PAGE_CACHE_NO_INDEX )
		SparkKvPageCacheDereference(cache,job->parent);
	job->parent = SPARK_KV_PAGE_CACHE_NO_INDEX;
	if ( job->read_errors != 0u )
	{
		snapshot->restore_read_error_count += job->read_errors;
		fprintf(stderr,"KV-SNAPSHOT restore read_error tokens=%u read_errors=%llu\n",job->token_count,(unsigned long long)snapshot->restore_read_error_count);
	}
	if ( status == SPARK_STATUS_HASH_MISMATCH )
	{
		snapshot->restore_corrupt_count++;
		if ( (snapshot->restore_corrupt_count & (snapshot->restore_corrupt_count - 1u)) == 0u )
			fprintf(stderr,"KV-SNAPSHOT restore corrupt tokens=%u corrupt=%llu\n",job->token_count,(unsigned long long)snapshot->restore_corrupt_count);
		status = SPARK_STATUS_NOT_FOUND;
	}
	if ( status == SPARK_STATUS_NOT_FOUND )
	{
		snapshot->restore_miss_count++;
		return(status);
	}
	snapshot->last_restore_status = status;
	if ( status != SPARK_STATUS_OK )
	{
		snapshot->restore_failure_count++;
		fprintf(stderr,"KV-SNAPSHOT restore status=%d tokens=%u failures=%llu\n",(int)status,job->token_count,(unsigned long long)snapshot->restore_failure_count);
		SPARK_RETURN(status);
	}
	snapshot->restore_count++;
	snapshot->restore_ns += SparkKvPageCacheNowNs() - job->start_ns;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheRestorePrefix(SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count)
{
	SparkKvPageCacheRestoreJob job;
	SparkStatus status;
	if ( SparkKvPageCacheIsValid(cache) == 0u || cache->snapshot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&job,0,sizeof(job));
	job.links = cache->snapshot->links;
	job.page = cache->snapshot->page;
	job.state = cache->snapshot->state;
	status = SparkKvPageCacheRestoreBegin(cache,&job,identity,token_count);
	if ( status != SPARK_STATUS_PENDING )
		return(status);
	status = SparkKvPageCacheRestoreReadChain(cache,&job);
	while ( status == SPARK_STATUS_OK )
	{
		status = SparkKvPageCacheRestoreAdvance(cache,&job);
		if ( status != SPARK_STATUS_OK || job.need == SPARK_KV_PAGE_CACHE_RESTORE_DONE )
			break;
		status = SparkKvPageCacheRestoreRead(cache,&job);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageCacheRestoreApply(cache,&job);
	}
	return(SparkKvPageCacheRestoreFinish(cache,&job,status));
}

SparkStatus SparkKvPageCacheExportResident(const SparkKvPageCache *cache,SparkKvPageCacheResidentRecord *records,uint32_t capacity,uint32_t *count_out)
{
	const SparkKvPageCacheEntry *entry;
	const SparkKvCacheBlock *block;
	uint32_t *record_of,index,depth,depth_max = 0u,count = 0u,parent;
	if ( SparkKvPageCacheIsValid(cache) == 0u || records == 0 || count_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*count_out = 0u;
	record_of = (uint32_t *)malloc((size_t)cache->entry_capacity * sizeof(*record_of));
	if ( record_of == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<cache->entry_capacity; index++)
	{
		record_of[index] = SPARK_KV_PAGE_CACHE_NO_INDEX;
		if ( (cache->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && cache->entries[index].page_count > depth_max )
			depth_max = cache->entries[index].page_count;
	}
	for (depth=1u; depth<=depth_max && count<capacity; depth++)
	{
		for (index=0u; index<cache->entry_capacity && count<capacity; index++)
		{
			entry = &cache->entries[index];
			if ( (entry->flags & (SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID | SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE)) != SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID ||
				entry->page_count != depth || entry->logical_page_index >= cache->kv_cache_arena->logical_block_count )
				continue;
			parent = entry->parent_entry_index;
			if ( (depth == 1u) != (parent == SPARK_KV_PAGE_CACHE_NO_INDEX) || (parent != SPARK_KV_PAGE_CACHE_NO_INDEX && (parent >= cache->entry_capacity || record_of[parent] == SPARK_KV_PAGE_CACHE_NO_INDEX)) )
				continue;
			block = &cache->kv_cache_arena->blocks[entry->logical_page_index];
			if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u || block->resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT )
				continue;
			memset(&records[count],0,sizeof(records[count]));
			records[count].identity = entry->identity;
			records[count].token_count = entry->token_count;
			records[count].page_count = entry->page_count;
			records[count].resident_slot = block->resident_slot_index;
			records[count].parent_record = parent == SPARK_KV_PAGE_CACHE_NO_INDEX ? SPARK_KV_PAGE_CACHE_NO_INDEX : record_of[parent];
			records[count].flags = entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED;
			record_of[index] = count++;
		}
	}
	free(record_of);
	*count_out = count;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvPageCacheResidentRecordFits(const SparkKvPageCache *cache,const SparkKvPageCacheResidentRecord *record,uint32_t parent)
{
	const SparkKvCacheArena *arena = cache->kv_cache_arena;
	uint32_t block = arena->block_token_count;
	if ( record->reserved0 != 0u || (record->flags & ~SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED) != 0u || record->page_count == 0u ||
		record->token_count <= (record->page_count - 1u) * block || record->token_count > record->page_count * block )
		return(0u);
	if ( (parent == SPARK_KV_PAGE_CACHE_NO_INDEX) != (record->page_count == 1u) )
		return(0u);
	if ( parent != SPARK_KV_PAGE_CACHE_NO_INDEX && (cache->entries[parent].page_count + 1u != record->page_count || cache->entries[parent].token_count != (record->page_count - 1u) * block) )
		return(0u);
	if ( record->resident_slot >= arena->resident_block_capacity || arena->resident_slot_logical_block_indices[record->resident_slot] != SPARK_KV_CACHE_NO_BLOCK )
		return(0u);
	return(SparkKvPageCacheFindEntryConst(cache,&record->identity,record->token_count) == SPARK_KV_PAGE_CACHE_NO_INDEX ? 1u : 0u);
}

static SparkStatus SparkKvPageCacheAdoptOne(SparkKvPageCache *cache,const SparkKvPageCacheResidentRecord *record,uint32_t parent,uint32_t *entry_index_out)
{
	SparkKvPageCacheEntry *entry;
	uint32_t entry_index,page = SPARK_KV_CACHE_NO_BLOCK,bucket;
	SparkStatus status;
	status = SparkKvPageCacheAcquireEntry(cache,&entry_index);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvCacheArenaAcquireBlock(cache->kv_cache_arena,&page);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaRetainBlock(cache->kv_cache_arena,page);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaAdoptResidentSlot(cache->kv_cache_arena,page,record->resident_slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvCacheArenaMarkBlockDirty(cache->kv_cache_arena,page);
	if ( status == SPARK_STATUS_OK && cache->entry_indices_by_logical_page[page] != SPARK_KV_PAGE_CACHE_NO_INDEX )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		if ( page != SPARK_KV_CACHE_NO_BLOCK && cache->kv_cache_arena->blocks[page].reference_count != 0u )
			(void)SparkKvCacheArenaReleaseBlockReference(cache->kv_cache_arena,page);
		if ( page != SPARK_KV_CACHE_NO_BLOCK )
			(void)SparkKvCacheArenaFreeBlock(cache->kv_cache_arena,page);
		cache->entries[entry_index].free_next = cache->free_entry_head;
		cache->free_entry_head = entry_index;
		SPARK_RETURN(status);
	}
	entry = &cache->entries[entry_index];
	entry->flags = SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID | record->flags | (cache->state_store != 0 ? SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS : 0u);
	entry->token_count = record->token_count;
	entry->page_count = record->page_count;
	entry->parent_entry_index = parent;
	entry->logical_page_index = page;
	entry->reference_count = 1u;
	entry->identity = record->identity;
	cache->entry_indices_by_logical_page[page] = entry_index;
	cache->epoch++;
	entry->last_used_epoch = cache->epoch;
	bucket = SparkKvPageCacheBucket(cache,&entry->identity,entry->token_count);
	entry->hash_next = cache->hash_bucket_heads[bucket];
	cache->hash_bucket_heads[bucket] = entry_index;
	cache->published_page_count++;
	if ( parent != SPARK_KV_PAGE_CACHE_NO_INDEX )
		SparkKvPageCacheReference(cache,parent);
	*entry_index_out = entry_index;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheAdoptResident(SparkKvPageCache *cache,const SparkKvPageCacheResidentRecord *records,uint32_t count,uint32_t *adopted_out)
{
	uint32_t *entry_of,index,parent,adopted = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if ( SparkKvPageCacheIsValid(cache) == 0u || (records == 0 && count != 0u) || adopted_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*adopted_out = 0u;
	if ( count == 0u )
		return(SPARK_STATUS_OK);
	entry_of = (uint32_t *)malloc((size_t)count * sizeof(*entry_of));
	if ( entry_of == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<count; index++)
		entry_of[index] = SPARK_KV_PAGE_CACHE_NO_INDEX;
	for (index=0u; index<count && status==SPARK_STATUS_OK; index++)
	{
		parent = SPARK_KV_PAGE_CACHE_NO_INDEX;
		if ( records[index].parent_record != SPARK_KV_PAGE_CACHE_NO_INDEX )
		{
			if ( records[index].parent_record >= index || entry_of[records[index].parent_record] == SPARK_KV_PAGE_CACHE_NO_INDEX )
				continue;
			parent = entry_of[records[index].parent_record];
		}
		if ( SparkKvPageCacheResidentRecordFits(cache,&records[index],parent) == 0u )
			continue;
		status = SparkKvPageCacheAdoptOne(cache,&records[index],parent,&entry_of[index]);
		if ( status == SPARK_STATUS_OK )
			adopted++;
	}
	for (index=0u; index<count; index++)
		if ( entry_of[index] != SPARK_KV_PAGE_CACHE_NO_INDEX )
			SparkKvPageCacheDereference(cache,entry_of[index]);
	free(entry_of);
	*adopted_out = adopted;
	if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
		return(SPARK_STATUS_OK);
	SPARK_RETURN(status);
}

static uint32_t SparkKvLaneContentEqual(const SparkModelDriverCacheLane *left,const SparkModelDriverCacheLane *right)
{
	if ( left->sequence_id != right->sequence_id || left->sequence_position != right->sequence_position || left->request_generation != right->request_generation || left->step_generation != right->step_generation )
		return(0u);
	if ( left->resident_sequence_slot != right->resident_sequence_slot || left->context_token_count != right->context_token_count || left->prefix_token_count != right->prefix_token_count || left->publish_token_count != right->publish_token_count || left->flags != right->flags || left->block_identity_count != right->block_identity_count )
		return(0u);
	if ( memcmp(&left->prefix_identity,&right->prefix_identity,sizeof(left->prefix_identity)) != 0 || memcmp(&left->publish_identity,&right->publish_identity,sizeof(left->publish_identity)) != 0 )
		return(0u);
	return(left->block_identity_count == 0u || (left->block_identities != 0 && right->block_identities != 0 &&
		memcmp(left->block_identities,right->block_identities,(size_t)left->block_identity_count * sizeof(*left->block_identities)) == 0) ? 1u : 0u);
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
	return(memcmp(&saved->residency,&request->residency,sizeof(saved->residency)) == 0 && SparkKvLaneContentEqual(&owner->lane,lane) != 0u);
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLaneTransactionsNextEpoch(transactions);
	for (index=0u; index<request->cache_lane_count; index++)
	{
		slot = request->cache_lanes[index].resident_sequence_slot;
		if ( slot >= transactions->cache->sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( transactions->lanes[slot].validation_epoch == transactions->validation_epoch )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		SPARK_RETURN(status);
	owner->page_count = 0u;
	status = SparkKvPageCacheRollbackLaneTransaction(transactions->cache,&owner->lane,owner->mutation_flags);
	if ( status == SPARK_STATUS_OK )
		owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
	SPARK_RETURN(status);
}

static SparkStatus SparkKvLaneTransactionsRequire(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request,uint32_t phase)
{
	SparkKvLaneTransaction *owner;
	uint32_t index;
	for (index=0u; index<request->cache_lane_count; index++)
	{
		owner = &transactions->lanes[request->cache_lanes[index].resident_sequence_slot];
		if ( owner->phase != phase )
		{
#ifdef DEBUG
			fprintf(stderr,"ADMIT9-KVPHASE slot=%u owner_phase=%u want_phase=%u owner_req=%llu owner_seq=%llu req=%llu seq=%llu pos=%u\n",
				(unsigned)request->cache_lanes[index].resident_sequence_slot,
				(unsigned)owner->phase,(unsigned)phase,
				(unsigned long long)owner->request.request_id,
				(unsigned long long)owner->lane.sequence_id,
				(unsigned long long)request->request_id,
				(unsigned long long)request->sequence_id,
				(unsigned)request->cache_lanes[index].sequence_position);
#endif
			SPARK_FAIL(SPARK_STATUS_BUSY);
		}
		if ( SparkKvLaneTransactionMatches(owner,request,&request->cache_lanes[index]) == 0u )
		{
#ifdef DEBUG
			fprintf(stderr,"ADMIT9-KVMATCH slot=%u owner_req=%llu owner_sub=%llu owner_seq=%llu req=%llu sub=%llu seq=%llu phase=%u\n",
				(unsigned)request->cache_lanes[index].resident_sequence_slot,
				(unsigned long long)owner->request.request_id,
				(unsigned long long)owner->request.submission_id,
				(unsigned long long)owner->request.sequence_id,
				(unsigned long long)request->request_id,
				(unsigned long long)request->submission_id,
				(unsigned long long)request->sequence_id,
				(unsigned)owner->phase);
#endif
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvLaneTransactionsPrepareLanes(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
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
	for (index=0u; transactions->cache->snapshot != 0 && transactions->restore_async == 0u && index<request->cache_lane_count; index++)
	{
		lane = &request->cache_lanes[index];
		if ( (lane->flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) == 0u || lane->prefix_token_count == 0u || transactions->cache->sequences[lane->resident_sequence_slot].sequence_id == lane->sequence_id )
			continue;
		status = SparkKvPageCacheRestorePrefix(transactions->cache,&lane->prefix_identity,lane->prefix_token_count);
		if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
			SPARK_RETURN(status);
	}
	for (index=0u; index<request->cache_lane_count; index++)
	{
		lane = &request->cache_lanes[index];
		owner = &transactions->lanes[lane->resident_sequence_slot];
		offset = ((uint64_t)lane->resident_sequence_slot * transactions->page_capacity);
		if ( lane->block_identity_count > SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES - 1u || (lane->block_identity_count != 0u && lane->block_identities == 0) )
		{
			status = SPARK_STATUS_INVALID_ARGUMENT;
			break;
		}
		status = SparkKvPageCacheBeginPinnedLaneTransaction(transactions->cache,lane,transactions->logical_pages + offset,transactions->physical_pages + offset,transactions->page_capacity,&owner->page_count,&owner->mutation_flags);
		if ( status != SPARK_STATUS_OK )
			break;
		owner->request = *request;
		owner->request.cache_lanes = 0;
		owner->lane = *lane;
		if ( lane->block_identity_count != 0u )
			memcpy(owner->block_identities,lane->block_identities,(size_t)lane->block_identity_count * sizeof(*lane->block_identities));
		owner->lane.block_identities = lane->block_identity_count != 0u ? owner->block_identities : 0;
		owner->phase = SPARK_KV_LANE_TRANSACTION_PREPARED;
		{
			struct timespec prepared_ts;
			owner->prepared_since_ns =
			    clock_gettime(CLOCK_MONOTONIC,&prepared_ts) == 0 ?
			    (uint64_t)prepared_ts.tv_sec *
			        UINT64_C(1000000000) +
			        (uint64_t)prepared_ts.tv_nsec : 0u;
		}
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
	SPARK_RETURN(status);
}

static uint32_t SparkKvLaneTransactionsNewWorkCanWait(const SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t slot,index,member;
	for (index=0u; index<request->cache_lane_count; index++)
		if ( transactions->cache->sequences[request->cache_lanes[index].resident_sequence_slot].sequence_id == request->cache_lanes[index].sequence_id )
			return(0u);
	for (slot=0u; slot<transactions->cache->sequence_capacity; slot++)
	{
		if ( transactions->cache->sequences[slot].sequence_id == 0u )
			continue;
		for (member=0u,index=0u; index<request->cache_lane_count; index++)
			member |= request->cache_lanes[index].resident_sequence_slot == slot ? 1u : 0u;
		if ( member == 0u )
			return(1u);
	}
	return(0u);
}

static SparkStatus SparkKvLaneTransactionsPrepare(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	uint64_t backing_full_before = transactions->cache->backing_full_count;
	SparkStatus status = SparkKvLaneTransactionsPrepareLanes(transactions,request);
	if ( status != SPARK_STATUS_CAPACITY_EXCEEDED || transactions->cache->backing_full_count == backing_full_before ||
		SparkKvLaneTransactionsNewWorkCanWait(transactions,request) == 0u )
		return(status);
	transactions->cache->backing_full_queued_count++;
	return(SPARK_STATUS_BUSY);
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
			SPARK_FAIL(SPARK_STATUS_BUSY);
		if ( transactions->cache->sequences[lane->resident_sequence_slot].sequence_id != lane->sequence_id )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	for (index=0u; index<request->cache_lane_count; index++)
	{
		lane = &request->cache_lanes[index];
		SparkKvPageCacheMarkSave(transactions->cache,lane->resident_sequence_slot);
		status = SparkKvPageCacheReleaseLane(transactions->cache,lane->resident_sequence_slot,lane->sequence_id);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvLaneTransactionsAdmit(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request)
{
	SparkKvLaneTransaction *owner;
	uint32_t index,phase;
	SparkStatus status,result;
	status = SparkKvLaneTransactionsValidate(transactions,request);
	if ( status == SPARK_STATUS_OK && transactions->cache->device_copy.retire_copies != 0 )
		status = transactions->cache->device_copy.retire_copies(transactions->cache->device_copy.context,0u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	transactions->cache->admission_priority = request->priority;
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		return(request->admission_flags == 0u ? SparkKvLaneTransactionsRelease(transactions,request) : SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE )
		return(SparkKvLaneTransactionsPrepare(transactions,request));
	phase = request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT ? SPARK_KV_LANE_TRANSACTION_PREPARED : SPARK_KV_LANE_TRANSACTION_COMMITTED;
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
	{
		for (index=0u; index<request->cache_lane_count && transactions->lanes[request->cache_lanes[index].resident_sequence_slot].phase == SPARK_KV_LANE_TRANSACTION_EMPTY; index++)
			;
		if ( index == request->cache_lane_count )
			return(SPARK_STATUS_OK);
		phase = transactions->lanes[request->cache_lanes[0].resident_sequence_slot].phase;
	}
	if ( phase != SPARK_KV_LANE_TRANSACTION_PREPARED && phase != SPARK_KV_LANE_TRANSACTION_COMMITTED )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	status = SparkKvLaneTransactionsRequire(transactions,request,phase);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	slot = frame->cache_lanes[0].resident_sequence_slot;
	if ( slot >= transactions->cache->sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		SPARK_RETURN(status);
	for (index=0u; index<frame->cache_lane_count; index++)
	{
		struct timespec executing_since;
		transactions->lanes[frame->cache_lanes[index].resident_sequence_slot].phase = SPARK_KV_LANE_TRANSACTION_EXECUTING;
		transactions->lanes[frame->cache_lanes[index].resident_sequence_slot].executing_since_ns =
		    clock_gettime(CLOCK_MONOTONIC,&executing_since) == 0 ?
		    (uint64_t)executing_since.tv_sec * UINT64_C(1000000000) + (uint64_t)executing_since.tv_nsec : 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvLaneTransactionFinish(SparkKvLaneTransactions *transactions,uint32_t resident_slot,SparkStatus execution_status,uint32_t extra_tokens)
{
	SparkKvLaneTransaction *owner;
	SparkModelDriverCacheLane lane;
	SparkStatus status,release_status;
	uint64_t offset;
	if ( transactions == 0 || SparkKvPageCacheIsValid(transactions->cache) == 0u || transactions->lanes == 0 || resident_slot >= transactions->cache->sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	owner = &transactions->lanes[resident_slot];
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	offset = ((uint64_t)resident_slot * transactions->page_capacity);
	status = SparkKvCacheArenaUnpinResidentTable(transactions->cache->kv_cache_arena,transactions->logical_pages + offset,owner->page_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	owner->page_count = 0u;
	lane = owner->lane;
	status = execution_status;
	if ( status == SPARK_STATUS_OK && extra_tokens > UINT32_MAX - lane.context_token_count )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
	{
		lane.context_token_count += extra_tokens;
		transactions->cache->admission_priority = owner->request.priority;
		status = SparkKvPageCacheCompleteLane(transactions->cache,&lane);
		if ( status == SPARK_STATUS_OK && (lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) != 0u )
			SparkKvPageCacheMarkSave(transactions->cache,resident_slot);
	}
	if ( status != SPARK_STATUS_OK )
	{
		release_status = SparkKvPageCacheReleaseLane(transactions->cache,resident_slot,lane.sequence_id);
		if ( release_status != SPARK_STATUS_OK )
			return(release_status);
	}
	owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
	SPARK_RETURN(status);
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
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( resident_slots[0] >= transactions->cache->sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	request = &transactions->lanes[resident_slots[0]].request;
	if ( request->cache_lane_count != lane_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvLaneTransactionsNextEpoch(transactions);
	for (index=0u; index<lane_count; index++)
	{
		slot = resident_slots[index];
		if ( slot >= transactions->cache->sequence_capacity || transactions->lanes[slot].phase != SPARK_KV_LANE_TRANSACTION_EXECUTING || transactions->lanes[slot].validation_epoch == transactions->validation_epoch )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( SparkKvLaneTransactionMatches(&transactions->lanes[slot],request,&transactions->lanes[slot].lane) == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
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

static SparkStatus SparkKvPageCacheEvictUnreferencedChains(SparkKvPageCache *cache)
{
	uint32_t index,entry,parent;
	SparkStatus status;
	for (index=0u; index<cache->entry_capacity; index++)
	{
		entry = index;
		while ( entry != SPARK_KV_PAGE_CACHE_NO_INDEX && (cache->entries[entry].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && cache->entries[entry].reference_count == 0u )
		{
			parent = cache->entries[entry].parent_entry_index;
			status = SparkKvPageCacheEvictEntry(cache,entry);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			entry = parent;
		}
	}
	for (index=0u; index<cache->entry_capacity; index++)
		if ( (cache->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u )
			SPARK_FAIL(SPARK_STATUS_BUSY);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPageCacheReleaseAll(SparkKvPageCache *cache)
{
	SparkStatus status;
	uint32_t slot;
	if ( SparkKvPageCacheIsValid(cache) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (slot=0u; slot<cache->sequence_capacity; slot++)
		if ( cache->sequences[slot].sequence_id != 0u )
		{
			status = SparkKvPageCacheReleaseLane(cache,slot,cache->sequences[slot].sequence_id);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	return(SparkKvPageCacheEvictUnreferencedChains(cache));
}

SparkStatus SparkKvLaneTransactionsReset(SparkKvLaneTransactions *transactions)
{
	SparkKvPageCache *cache;
	SparkStatus status;
	uint32_t slot;
	if ( transactions == 0 || SparkKvPageCacheIsValid(transactions->cache) == 0u || transactions->lanes == 0 || transactions->logical_pages == 0 || transactions->physical_pages == 0 || transactions->page_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	cache = transactions->cache;
	for (slot=0u; slot<cache->sequence_capacity; slot++)
		if ( transactions->lanes[slot].phase >= SPARK_KV_LANE_TRANSACTION_EXECUTING )
			SPARK_FAIL(SPARK_STATUS_BUSY);
	if ( cache->snapshot != 0 && cache->snapshot->in_flight != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	for (slot=0u; slot<cache->sequence_capacity; slot++)
	{
		if ( transactions->lanes[slot].phase != SPARK_KV_LANE_TRANSACTION_EMPTY )
		{
			status = SparkKvLaneTransactionAbort(transactions,&transactions->lanes[slot]);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
	}
	if ( cache->device_copy.retire_copies != 0 )
	{
		status = cache->device_copy.retire_copies(cache->device_copy.context,1u);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	SparkKvPageCacheSaveCancelAll(cache);
	return(SparkKvPageCacheReleaseAll(cache));
}
