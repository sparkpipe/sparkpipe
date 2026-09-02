#pragma once


#include <stdint.h>
#include <string.h>

#define LM_CACHE_OK 0
#define LM_CACHE_ERR_SHAPE (-61)
#define LM_CACHE_ERR_FULL (-62)
#define LM_CACHE_ERR_NOT_FOUND (-63)
#define LM_CACHE_ERR_BUSY (-64)
#define LM_CACHE_ERR_CAPACITY (-65)

#define LM_CACHE_NO_BLOCK 0xffffffffu
#define LM_CACHE_NO_ENTRY 0xffffffffu

typedef struct LmCachePartition
{
	uint64_t total_bytes;
	uint32_t block_tokens;
	uint32_t slot_bytes;
	uint32_t index_entries;
	uint32_t jit_reserve_blocks;
	uint32_t resident_limit;
}

LmCachePartition;

typedef struct LmCacheBlock
{
	uint64_t content_hash;
	uint32_t reference_count;
	uint32_t index_entry;
	uint64_t last_use;
	uint32_t reuse_count;
	uint32_t sequence_hint;
	uint8_t resident;
	uint8_t protected_from_eviction;
}
LmCacheBlock;

typedef struct LmCacheIndexEntry
{
	uint64_t content_hash;
	uint32_t block;
	uint32_t next;
}
LmCacheIndexEntry;

typedef struct LmCache
{
	LmCachePartition partition;
	LmCacheBlock *blocks;
	LmCacheIndexEntry *index;
	uint32_t *buckets;
	uint32_t block_count;
	uint32_t bucket_count;
	uint32_t free_hint;
	uint32_t live_blocks;
	uint32_t jit_held;
	uint64_t tick;
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	uint32_t resident_blocks;
	uint64_t fetches;
	uint64_t resident_evictions;
}
LmCache;

static uint64_t LmCacheMix(uint64_t value)
{
	value ^= value >> 33;
	value *= 0xff51afd7ed558ccdULL;
	value ^= value >> 33;
	value *= 0xc4ceb9fe1a85ec53ULL;
	value ^= value >> 33;
	return(value);
}

static uint64_t LmCacheHashBlock(uint64_t previous_hash, const uint32_t *tokens, uint32_t count)
{
	uint64_t hash = previous_hash ^ 0x9e3779b97f4a7c15ULL;
	uint32_t index;
	for (index = 0u; index < count; ++index)
		hash = LmCacheMix(hash ^ (uint64_t)tokens[index]);
	return(hash | 1u);
}

static uint64_t LmCacheBlocksAvailable(const LmCachePartition *partition)
{
	uint64_t per_block;
	if ( partition->block_tokens == 0u || partition->slot_bytes == 0u )
		return(0u);
	per_block = (uint64_t)partition->block_tokens * partition->slot_bytes;
	{
		uint64_t overhead = ((uint64_t)partition->index_entries * sizeof(LmCacheIndexEntry))
			+ ((uint64_t)partition->index_entries * sizeof(uint32_t));
		uint64_t usable = partition->total_bytes > overhead ? partition->total_bytes - overhead : 0u;
		uint64_t with_table = per_block + sizeof(LmCacheBlock);
		return(usable / with_table);
	}
}

static int32_t LmCacheInitialise(LmCache *cache, const LmCachePartition *partition, void *block_table, void *index_table, void *bucket_table)
{
	uint64_t count;
	uint32_t index;
	if ( cache == 0 || partition == 0 )
		return(LM_CACHE_ERR_SHAPE);
	count = LmCacheBlocksAvailable(partition);
	if ( count == 0u || count > 0xfffffffeULL )
		return(LM_CACHE_ERR_SHAPE);
	if ( partition->jit_reserve_blocks >= count )
		return(LM_CACHE_ERR_SHAPE);
	cache->partition = *partition;
	cache->blocks = (LmCacheBlock *)block_table;
	cache->index = (LmCacheIndexEntry *)index_table;
	cache->buckets = (uint32_t *)bucket_table;
	cache->block_count = (uint32_t)count;
	cache->bucket_count = partition->index_entries;
	cache->free_hint = 0u;
	cache->live_blocks = 0u;
	cache->jit_held = 0u;
	cache->tick = 1u;
	cache->hits = 0u;
	cache->misses = 0u;
	cache->evictions = 0u;
	for (index = 0u; index < cache->block_count; ++index)
	{
		cache->blocks[index].content_hash = 0u;
		cache->blocks[index].reference_count = 0u;
		cache->blocks[index].index_entry = LM_CACHE_NO_ENTRY;
		cache->blocks[index].last_use = 0u;
		cache->blocks[index].reuse_count = 0u;
		cache->blocks[index].sequence_hint = 0u;
	}
	for (index = 0u; index < cache->bucket_count; ++index)
		cache->buckets[index] = LM_CACHE_NO_ENTRY;
	return(LM_CACHE_OK);
}

static uint64_t LmCacheScore(const LmCacheBlock *block)
{
	return(block->last_use + ((uint64_t)block->reuse_count << 8));
}

static uint32_t LmCacheFindEvictable(LmCache *cache)
{
	uint32_t best = LM_CACHE_NO_BLOCK,scanned,index;
	uint64_t best_score = 0u;
	for (scanned = 0u; scanned < cache->block_count; ++scanned)
	{
		index = (cache->free_hint + scanned) % cache->block_count;
		if ( cache->blocks[index].reference_count != 0u )
			continue;
		if ( cache->blocks[index].content_hash == 0u )
		{
			cache->free_hint = (index + 1u) % cache->block_count;
			return(index);
		}
		if ( best == LM_CACHE_NO_BLOCK || LmCacheScore(&cache->blocks[index]) < best_score )
		{
			best = index;
			best_score = LmCacheScore(&cache->blocks[index]);
		}
	}
	if ( best != LM_CACHE_NO_BLOCK )
		cache->free_hint = (best + 1u) % cache->block_count;
	return(best);
}

static void LmCacheUnlink(LmCache *cache, uint32_t block)
{
	uint32_t entry = cache->blocks[block].index_entry,bucket,walk,previous;
	if ( entry == LM_CACHE_NO_ENTRY || cache->bucket_count == 0u )
		return;
	bucket = (uint32_t)(cache->index[entry].content_hash % cache->bucket_count);
	walk = cache->buckets[bucket];
	previous = LM_CACHE_NO_ENTRY;
	while ( walk != LM_CACHE_NO_ENTRY && walk != entry )
	{
		previous = walk;
		walk = cache->index[walk].next;
	}
	if ( walk == entry )
	{
		if ( previous == LM_CACHE_NO_ENTRY )
			cache->buckets[bucket] = cache->index[entry].next;
		else
			cache->index[previous].next = cache->index[entry].next;
	}
	cache->blocks[block].index_entry = LM_CACHE_NO_ENTRY;
}

static int32_t LmCacheAcquire(LmCache *cache, uint64_t content_hash, uint32_t sequence, uint32_t *block_out, int32_t *was_hit)
{
	uint32_t bucket,entry,block;
	*was_hit = 0;
	if ( cache->bucket_count != 0u && content_hash != 0u )
	{
		bucket = (uint32_t)(content_hash % cache->bucket_count);
		for (entry = cache->buckets[bucket]; entry != LM_CACHE_NO_ENTRY; entry = cache->index[entry].next)
		{
			if ( cache->index[entry].content_hash != content_hash )
				continue;
			block = cache->index[entry].block;
			if ( cache->blocks[block].reference_count == 0u )
				cache->live_blocks++;
			cache->blocks[block].reference_count++;
			cache->blocks[block].last_use = cache->tick++;
			cache->blocks[block].reuse_count++;
			cache->hits++;
			*block_out = block;
			*was_hit = 1;
			return(LM_CACHE_OK);
		}
	}
	cache->misses++;
	if ( cache->live_blocks + cache->jit_held >= cache->block_count )
		return(LM_CACHE_ERR_FULL);
	block = LmCacheFindEvictable(cache);
	if ( block == LM_CACHE_NO_BLOCK )
		return(LM_CACHE_ERR_FULL);
	if ( cache->blocks[block].content_hash != 0u )
	{
		LmCacheUnlink(cache,block);
		cache->evictions++;
	}
	cache->blocks[block].content_hash = 0u;
	cache->blocks[block].reference_count = 1u;
	cache->blocks[block].last_use = cache->tick++;
	cache->blocks[block].reuse_count = 0u;
	cache->blocks[block].sequence_hint = sequence;
	cache->live_blocks++;
	*block_out = block;
	return(LM_CACHE_OK);
}

static int32_t LmCachePublish(LmCache *cache, uint32_t block, uint64_t content_hash)
{
	uint32_t bucket,entry;
	if ( cache->bucket_count == 0u || content_hash == 0u )
		return(LM_CACHE_OK);
	if ( block >= cache->block_count || cache->blocks[block].reference_count == 0u )
		return(LM_CACHE_ERR_SHAPE);
	entry = block % cache->bucket_count;
	if ( cache->index[entry].block < cache->block_count
		&& cache->index[entry].block != block
		&& cache->blocks[cache->index[entry].block].index_entry == entry )
		LmCacheUnlink(cache,cache->index[entry].block);
	bucket = (uint32_t)(content_hash % cache->bucket_count);
	cache->index[entry].content_hash = content_hash;
	cache->index[entry].block = block;
	cache->index[entry].next = cache->buckets[bucket];
	cache->buckets[bucket] = entry;
	cache->blocks[block].content_hash = content_hash;
	cache->blocks[block].index_entry = entry;
	return(LM_CACHE_OK);
}

static int32_t LmCacheRelease(LmCache *cache, uint32_t block)
{
	if ( block >= cache->block_count || cache->blocks[block].reference_count == 0u )
		return(LM_CACHE_ERR_SHAPE);
	cache->blocks[block].reference_count--;
	if ( cache->blocks[block].reference_count == 0u )
		cache->live_blocks--;
	return(LM_CACHE_OK);
}

static int32_t LmCacheFork(LmCache *cache, uint32_t block, uint32_t sequence, uint32_t *forked_out)
{
	int32_t hit,status;
	if ( block >= cache->block_count )
		return(LM_CACHE_ERR_SHAPE);
	if ( cache->blocks[block].reference_count == 1u )
	{
		LmCacheUnlink(cache,block);
		cache->blocks[block].content_hash = 0u;
		*forked_out = block;
		return(LM_CACHE_OK);
	}
	status = LmCacheAcquire(cache,0u,sequence,forked_out,&hit);
	if ( status != LM_CACHE_OK )
		return(status);
	return(LmCacheRelease(cache,block));
}

static int32_t LmCacheReserve(LmCache *cache, uint32_t blocks)
{
	if ( cache->live_blocks + cache->jit_held + blocks > cache->block_count )
		return(LM_CACHE_ERR_FULL);
	if ( cache->jit_held + blocks > cache->partition.jit_reserve_blocks )
		return(LM_CACHE_ERR_BUSY);
	cache->jit_held += blocks;
	return(LM_CACHE_OK);
}

static void LmCacheReleaseReservation(LmCache *cache, uint32_t blocks)
{
	cache->jit_held = cache->jit_held > blocks ? cache->jit_held - blocks : 0u;
}

static uint32_t LmCacheBlocksForTokens(const LmCache *cache, uint32_t tokens)
{
	return((tokens + cache->partition.block_tokens - 1u) / cache->partition.block_tokens);
}

static uint32_t LmCacheHitPercent(const LmCache *cache)
{
	uint64_t total = cache->hits + cache->misses;
	return(total == 0u ? 0u : (uint32_t)((cache->hits * 100u) / total));
}


static uint32_t LmCacheSelectResidentVictim(LmCache *cache)
{
	uint32_t best = LM_CACHE_NO_BLOCK,index;
	uint64_t best_score = 0u;
	for (index = 0u; index < cache->block_count; ++index)
	{
		if ( cache->blocks[index].resident == 0u )
			continue;
		if ( cache->blocks[index].protected_from_eviction )
			continue;
		if ( best == LM_CACHE_NO_BLOCK || LmCacheScore(&cache->blocks[index]) < best_score )
		{
			best = index;
			best_score = LmCacheScore(&cache->blocks[index]);
		}
	}
	return(best);
}

static int32_t LmCacheMakeResident(LmCache *cache, uint32_t block, int32_t *needs_fetch)
{
	*needs_fetch = 0;
	if ( block >= cache->block_count )
		return(LM_CACHE_ERR_SHAPE);
	if ( cache->blocks[block].resident )
	{
		cache->blocks[block].last_use = cache->tick++;
		return(LM_CACHE_OK);
	}
	if ( cache->resident_blocks >= cache->partition.resident_limit )
	{
		uint32_t victim = LmCacheSelectResidentVictim(cache);
		if ( victim == LM_CACHE_NO_BLOCK )
			return(LM_CACHE_ERR_FULL);
		cache->blocks[victim].resident = 0u;
		cache->resident_blocks--;
		cache->resident_evictions++;
	}
	cache->blocks[block].resident = 1u;
	cache->blocks[block].last_use = cache->tick++;
	cache->resident_blocks++;
	cache->fetches++;
	*needs_fetch = 1;
	return(LM_CACHE_OK);
}

static int32_t LmCacheProtect(LmCache *cache, uint32_t block, int32_t protect)
{
	if ( block >= cache->block_count )
		return(LM_CACHE_ERR_SHAPE);
	cache->blocks[block].protected_from_eviction = protect ? 1u : 0u;
	return(LM_CACHE_OK);
}

static int32_t LmCacheBlocksAreResident(const LmCache *cache, const uint32_t *blocks, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; ++index)
	{
		if ( blocks[index] >= cache->block_count )
			return(0);
		if ( cache->blocks[blocks[index]].resident == 0u )
			return(0);
	}
	return(1);
}

static uint32_t LmCacheResidentPercent(const LmCache *cache)
{
	return(cache->block_count == 0u ? 0u
		: (cache->resident_blocks * 100u) / cache->block_count);
}


#define LM_CACHE_PREFETCH_LANES 8u
#define LM_CACHE_PREFETCH_CAPACITY 256u

typedef enum LmCacheSource
{
	LM_CACHE_SOURCE_NONE = 0,
	LM_CACHE_SOURCE_MEMORY = 1,
	LM_CACHE_SOURCE_FILE = 2,
	LM_CACHE_SOURCE_PEER = 3
}
LmCacheSource;

typedef struct LmCachePrefetchBlock
{
	uint32_t block;
	uint32_t lane;
	uint32_t source;
	uint64_t source_offset;
}
LmCachePrefetchBlock;

typedef struct LmCachePrefetchPlan
{
	LmCachePrefetchBlock blocks[LM_CACHE_PREFETCH_CAPACITY];
	uint32_t lane_counts[LM_CACHE_PREFETCH_LANES];
	uint32_t prefetch_count;
	uint32_t resident_count;
	uint32_t duplicate_count;
	uint32_t missing_count;
}
LmCachePrefetchPlan;

typedef int32_t (*LmCacheResolveSource)(void *context, uint32_t block, uint32_t *source_out, uint64_t *offset_out);

static int32_t LmCachePlanPrefetch(const LmCache *cache, const uint32_t *blocks, uint32_t count, LmCacheResolveSource resolve, void *resolve_context, LmCachePrefetchPlan *plan)
{
	uint32_t index,scan,lane = 0u;
	if ( cache == 0 || blocks == 0 || plan == 0 )
		return(LM_CACHE_ERR_SHAPE);
	memset(plan,0,sizeof(*plan));
	for (index = 0u; index < count; ++index)
	{
		uint32_t block = blocks[index],source = LM_CACHE_SOURCE_NONE;
		uint64_t offset = 0u;
		int32_t duplicate = 0;
		if ( block >= cache->block_count )
			return(LM_CACHE_ERR_SHAPE);
		if ( cache->blocks[block].resident )
		{
			plan->resident_count++;
			continue;
		}
		for (scan = 0u; scan < plan->prefetch_count; ++scan)
			if ( plan->blocks[scan].block == block )
				duplicate = 1;
		if ( duplicate )
		{
			plan->duplicate_count++;
			continue;
		}
		if ( resolve == 0 || resolve(resolve_context,block,&source,&offset) != LM_CACHE_OK
			|| source == LM_CACHE_SOURCE_NONE )
		{
			plan->missing_count++;
			continue;
		}
		if ( plan->prefetch_count >= LM_CACHE_PREFETCH_CAPACITY )
			return(LM_CACHE_ERR_CAPACITY);
		plan->blocks[plan->prefetch_count].block = block;
		plan->blocks[plan->prefetch_count].lane = lane;
		plan->blocks[plan->prefetch_count].source = source;
		plan->blocks[plan->prefetch_count].source_offset = offset;
		plan->lane_counts[lane]++;
		lane = (lane + 1u) % LM_CACHE_PREFETCH_LANES;
		plan->prefetch_count++;
	}
	return(LM_CACHE_OK);
}

static int32_t LmCacheProtectPlan(LmCache *cache, const LmCachePrefetchPlan *plan, int32_t protect)
{
	uint32_t index;
	for (index = 0u; index < plan->prefetch_count; ++index)
	{
		int32_t status = LmCacheProtect(cache,plan->blocks[index].block,protect);
		if ( status != LM_CACHE_OK )
			return(status);
	}
	return(LM_CACHE_OK);
}

static int32_t LmCacheCompletePlan(LmCache *cache, const LmCachePrefetchPlan *plan)
{
	uint32_t index;
	for (index = 0u; index < plan->prefetch_count; ++index)
	{
		int32_t fetch;
		int32_t status = LmCacheMakeResident(cache,plan->blocks[index].block,&fetch);
		if ( status != LM_CACHE_OK )
			return(status);
	}
	return(LM_CACHE_OK);
}
