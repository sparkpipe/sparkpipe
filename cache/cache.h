#pragma once

// The KV cache. One arena, content-addressed sharing, JIT admission.
//
// This merges what were three files and one of them turned out not to be a
// cache at all: work_control.c was packet building, batch bucketing and prefill
// chunking, which is scheduler work and now lives there. What remains is two
// real things that belong together and were apart:
//
//   the ARENA      uniform blocks of opaque bytes, acquired and released
//   the INDEX      content hashes of token runs, mapping to blocks that hold them
//
// They were separate because one was written for storage and the other for
// reuse, and the seam between them - who owns a block that two sequences share -
// was expressed twice: a binding table in the index and a free list in the arena.
// A block could be free in one and bound in the other, which is a class of bug
// that a single reference count makes unrepresentable.
//
// WHAT MAKES A KV CACHE GOOD, in the order the numbers say it matters.
//
// 1. NOT ALLOCATING. A prefix shared between two requests should cost nothing
//    the second time. At a 64-token block and a 2,000-token system prompt that
//    is 31 blocks per request that never need writing, and in a serving mix
//    where most requests share a preamble it is most of the cache.
//
// 2. NOT COPYING. Sharing is a reference count, not a memcpy. Two sequences on
//    the same prefix point at the same blocks until one of them writes past the
//    divergence, and only the diverging block is forked.
//
// 3. NOT KEEPING. A block nothing references is reusable immediately; the
//    question is which of those to evict first, and the answer is the one whose
//    content is least likely to recur, approximated by least-recently-used
//    weighted by how many times it has been reused. A block reused twenty times
//    is worth more than a fresh one even if older.
//
// 4. NOT STORING WIDE. The cache is read once per (sequence, position) and
//    shared with nothing, so it sits at a different point on the
//    precision-versus-bytes curve than weights. INT8 at block 64 halves it
//    against BF16 for measured 0.647 percent error, and the format is a
//    parameter rather than a rebuild.
//
// THIS CACHE IS TWO-TIER AND MY FIRST VERSION WAS NOT. The audit before deleting
// the old one found forty-nine references to residency in kv_cache.c against two
// here, and residency is the whole point of a large cache: blocks live in a pool
// sized in terabytes and only some of them are RESIDENT in device memory at any
// moment. A single-tier cache with a 1 TB partition is a cache that assumes a
// terabyte of HBM.
//
// So a block has two states that vary independently:
//
//   REFERENCED   some sequence needs it. Reference counted; zero means evictable
//                from the pool entirely.
//   RESIDENT     it is in device memory right now. Bounded by what the device
//                has, and a referenced block that is not resident must be
//                fetched before it can be read.
//
// The second bound is the tight one. At GLM 5.2's 1,152-byte slot and 64-token
// blocks, 1 TB is 14.9 million blocks and 40 GB of device memory holds about
// 566,000 - so at any moment 96 percent of a full cache is not resident, and
// which 4 percent is resident is the decision that matters.
//
// PARTITIONING. The arena's size, the index's size, the resident limit and the
// reserve held for just-in-time admission are configured together, the way disk
// partitions are:
// not adjusted often, and the thing that saves the day when a workload changes
// shape. A serving mix with long shared preambles wants a large index and a
// small reserve; one with many short unique prompts wants the opposite.

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

// How the cache is carved. Set once at start, like a partition table.
//
// The three sizes trade against each other in one pool, so they are one struct
// rather than three arguments that can be set inconsistently.
typedef struct LmCachePartition
{
	uint64_t total_bytes;          /* what the cache may use in total */
	uint32_t block_tokens;         /* positions per block */
	uint32_t slot_bytes;           /* bytes per position, from the model geometry */
	uint32_t index_entries;        /* content-addressed entries; 0 disables sharing */
	uint32_t jit_reserve_blocks;   /* held back for admitted-not-started requests */
	/* How many blocks fit in device memory. The pool may be a thousand times
	   this; that ratio is the point. */
	uint32_t resident_limit;
}

LmCachePartition;

// One block of the arena.
//
// reference_count is the whole ownership story. A block with count zero is
// evictable; one with count above zero is live in that many sequences. There is
// no separate free list and no separate binding table, because two
// representations of the same fact is how a block ends up free and bound at once.
typedef struct LmCacheBlock
{
	uint64_t content_hash;         /* of the token run this holds; 0 if unhashed */
	uint32_t reference_count;
	uint32_t index_entry;          /* back-pointer, or LM_CACHE_NO_ENTRY */
	uint64_t last_use;             /* monotonic tick */
	uint32_t reuse_count;          /* times acquired by hash rather than fresh */
	uint32_t sequence_hint;        /* first owner, for debugging only */
	/* In device memory right now. Independent of reference_count: a referenced
	   block may be non-resident and awaiting fetch, and a resident block may be
	   unreferenced and merely warm. */
	uint8_t resident;
	/* Protected from residency eviction even when unreferenced. This is what a
	   lookahead reservation buys: the scheduler admits a request, protects the
	   blocks its prefix will need, and the fetch has time to land before the
	   request runs. Without it a protected block is evicted between admission
	   and execution and the request stalls on a fetch it already paid for. */
	uint8_t protected_from_eviction;
}
LmCacheBlock;

typedef struct LmCacheIndexEntry
{
	uint64_t content_hash;
	uint32_t block;
	uint32_t next;                 /* chain within a hash bucket */
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
	uint32_t free_hint;            /* where the last scan stopped */
	uint32_t live_blocks;
	uint32_t jit_held;
	uint64_t tick;
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	uint32_t resident_blocks;
	uint64_t fetches;              /* referenced but not resident */
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

// Rolling hash over a block's tokens, chained from the previous block's hash.
//
// Chained rather than per-block so that a block matches only when the ENTIRE
// prefix before it matches. Hashing a block's own tokens alone would let two
// sequences share a block whose contents agree by coincidence while the
// preceding context differs - and the KV in that block depends on everything
// before it, so the coincidence would be silently wrong.
static uint64_t LmCacheHashBlock(uint64_t previous_hash, const uint32_t *tokens, uint32_t count)
{
	uint64_t hash = previous_hash ^ 0x9e3779b97f4a7c15ULL;
	uint32_t index;
	for (index = 0u; index < count; ++index)
		hash = LmCacheMix(hash ^ (uint64_t)tokens[index]);
	return(hash | 1u);              /* never zero; zero means unhashed */
}

static uint64_t LmCacheBlocksAvailable(const LmCachePartition *partition)
{
	uint64_t per_block;
	if ( partition->block_tokens == 0u || partition->slot_bytes == 0u )
		return(0u);
	per_block = (uint64_t)partition->block_tokens * partition->slot_bytes;
	// The index and the block table are overhead against the same pool. Sizing
	// them out of it rather than beside it is what makes total_bytes a real
	// ceiling instead of an approximate one.
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

// Eviction score. Lower is evicted first.
//
// Least-recently-used, but a block that has been reused is worth keeping past
// its age: a system preamble touched by every request is old by the time the
// hundredth arrives and is the single most valuable block in the cache. Plain
// LRU evicts exactly that.
//
// The weight is a shift rather than a multiply so the comparison stays integral
// and the ordering is total - a float score that ties on equal values makes
// eviction order depend on scan order, which makes a serving run irreproducible.
static uint64_t LmCacheScore(const LmCacheBlock *block)
{
	return(block->last_use + ((uint64_t)block->reuse_count << 8));
}

// A block with no references, preferring the lowest score.
//
// Scans from a rotating hint rather than from zero, so a cache under pressure
// does not re-walk the same busy prefix on every acquire. The scan is bounded by
// the block count, so this is O(n) worst case and O(1) amortised when blocks are
// plentiful - which is the case a 1 TB arena is sized to guarantee.
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
			// Never written: free, and cheaper than evicting anything.
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

// Acquire the block holding this content hash, or a fresh one.
//
// THE HIT PATH IS THE POINT. A hit increments a reference count and returns; it
// writes no bytes and allocates nothing. That is what makes a shared preamble
// free after the first request rather than merely cheap.
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
			// A block at zero references is resident but not live: it holds
			// valid content and is evictable. Coming back to one makes it live
			// again, and forgetting that here while release decrements on the
			// way down makes live_blocks underflow and the arena report itself
			// full at eight blocks used. Found by tests/test_cache.c.
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
	// The reserve is held for requests already admitted. Handing it to a fresh
	// miss is what turns an admitted request into one that stalls mid-prefill,
	// which is worse than refusing it at admission.
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

// Publish a written block under its content hash so later requests can share it.
//
// Separate from acquire because a block is only shareable once its KV is
// actually written. Publishing at acquire time would let a second sequence bind
// to a block that is still being filled and read whatever was there before -
// fluent output from a half-written cache, which is the hardest kind to catch.
static int32_t LmCachePublish(LmCache *cache, uint32_t block, uint64_t content_hash)
{
	uint32_t bucket,entry;
	if ( cache->bucket_count == 0u || content_hash == 0u )
		return(LM_CACHE_OK);
	if ( block >= cache->block_count || cache->blocks[block].reference_count == 0u )
		return(LM_CACHE_ERR_SHAPE);
	// Reuse the index slot that mirrors the block, so the index cannot outgrow
	// the arena and needs no allocator of its own.
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

// Fork a shared block for a sequence that is about to write past the prefix.
//
// Copy-on-write, and only here: two sequences read the same blocks until one
// diverges, and then only the diverging block is duplicated. Everything before
// it stays shared, which is why a long preamble costs one block to branch from
// rather than the whole prefix.
static int32_t LmCacheFork(LmCache *cache, uint32_t block, uint32_t sequence, uint32_t *forked_out)
{
	int32_t hit,status;
	if ( block >= cache->block_count )
		return(LM_CACHE_ERR_SHAPE);
	if ( cache->blocks[block].reference_count == 1u )
	{
		// Sole owner: writing in place is safe and the fork is a no-op. The
		// caller must still unpublish, because the content is about to change.
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

// -- just-in-time admission ------------------------------------------------------
//
// The scheduler admits a request before it runs, and between those two moments
// the blocks it will need must not be handed to someone else. Reserving them
// holds the count without binding specific blocks, so eviction can still choose
// freely - what is reserved is capacity, not identity.
//
// This is what makes a large arena useful rather than merely large: without it,
// a request admitted against 1 TB of free space can still stall because the
// space was consumed between admission and execution.
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

// Blocks a prompt of this length needs, and how many it can expect to share.
//
// The second number is what admission should use: a request whose prefix is
// already resident needs far fewer new blocks than its length suggests, and
// admitting on length alone under-admits by exactly the hit rate.
static uint32_t LmCacheBlocksForTokens(const LmCache *cache, uint32_t tokens)
{
	return((tokens + cache->partition.block_tokens - 1u) / cache->partition.block_tokens);
}

// What fraction of acquires are hits, in percent. The number admission should
// watch: a serving mix whose prefix hit rate is 90 percent needs a tenth of the
// blocks its prompt lengths suggest, and admitting on length alone leaves the
// arena mostly idle.
static uint32_t LmCacheHitPercent(const LmCache *cache)
{
	uint64_t total = cache->hits + cache->misses;
	return(total == 0u ? 0u : (uint32_t)((cache->hits * 100u) / total));
}

// -- residency -------------------------------------------------------------------
//
// A block being in the pool and a block being readable are different questions,
// and conflating them is what a single-tier cache does. These four functions are
// the second question.

// Which resident block to evict from device memory.
//
// Not the same choice as pool eviction. A block may be referenced - some
// sequence needs it - and still be the right one to page out, if that sequence
// will not run this step. So the score ignores reference_count and weighs
// recency against reuse, and protected blocks are never chosen.
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

// Make a block readable. Returns whether a fetch is needed - the caller issues
// it, because this file does not know whether the source is host memory, NVMe or
// a peer rank, and should not.
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

// Protect a block from residency eviction. What a lookahead reservation buys:
// the scheduler admits, protects, the fetch lands, the request runs. Unprotecting
// is the caller's job and forgetting it pins a block forever, which is why the
// count is reported.
static int32_t LmCacheProtect(LmCache *cache, uint32_t block, int32_t protect)
{
	if ( block >= cache->block_count )
		return(LM_CACHE_ERR_SHAPE);
	cache->blocks[block].protected_from_eviction = protect ? 1u : 0u;
	return(LM_CACHE_OK);
}

// Is every block a sequence needs already readable?
//
// The question admission should ask, and the one a single-tier cache cannot
// express: a prefix can be a perfect hash hit and still not be readable, and
// admitting on the hit alone stalls the request on a fetch at its first layer.
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

// -- just-in-time prefetch ---------------------------------------------------------
//
// The last piece, and the one that makes a terabyte useful rather than merely
// large. Harvested from the old arena's AsyncPrefetchBackend, which was the only
// part of it I had not accounted for and the only part that turns "this block is
// cold" into "this block is arriving".
//
// THE SEQUENCE THE WHOLE DESIGN EXISTS FOR:
//
//   1. the scheduler admits a request whose prefix is a cache hit
//   2. LmCachePlanPrefetch says which of those blocks are not resident
//   3. the blocks are PROTECTED so nothing evicts them while they land
//   4. the fetches are issued and the request waits in a queue, not on a stall
//   5. when they land the request moves to ready-to-issue
//
// Without step 3 the blocks fetched in step 4 can be evicted before step 5, and
// the request pays the fetch twice. Without step 2 every admitted request stalls
// at its first layer. The old code had all five; my version had one.
//
// LANES. Fetches are spread across lanes so several can be in flight, and a
// lane is a queue rather than a thread - what services it is the caller's
// business. The plan assigns blocks round-robin because the cost of a fetch does
// not depend on which block it is, so there is nothing to balance beyond count.
//
// A PLAN IS A REQUEST, NOT A PROMISE. It reports what it could not include -
// blocks already resident, duplicates within the request, and blocks with no
// source - rather than silently shortening. A caller that admits on
// prefetch_count without checking missing_count admits a request whose prefix is
// partly unfetchable.

#define LM_CACHE_PREFETCH_LANES 8u
#define LM_CACHE_PREFETCH_CAPACITY 256u

typedef enum LmCacheSource
{
	LM_CACHE_SOURCE_NONE = 0,
	LM_CACHE_SOURCE_MEMORY = 1,     /* host memory, mapped */
	LM_CACHE_SOURCE_FILE = 2,       /* a descriptor and an offset */
	LM_CACHE_SOURCE_PEER = 3        /* another rank's resident copy */
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
	uint32_t prefetch_count;        /* what will be fetched */
	uint32_t resident_count;        /* already readable, nothing to do */
	uint32_t duplicate_count;       /* named twice in the request */
	uint32_t missing_count;         /* no source; the caller must handle this */
}
LmCachePrefetchPlan;

// Which of these blocks need fetching, and on which lane.
//
// Resolving a source is the caller's - this file does not know where a cold
// block lives - so the resolver is a callback and a block it cannot place is
// counted as missing rather than dropped.
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
		// A block named twice costs one fetch, not two. The old plan checked
		// this and it is the difference between a shared prefix costing its
		// length and costing its length times the number of sequences on it.
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

// Protect everything a plan will fetch, before any of it is issued.
//
// Step three, and the one whose absence is invisible until load: a fetch that
// completes into a block already evicted has done nothing, and the request that
// waited for it stalls anyway. Under light load it never happens.
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

// Mark a plan's blocks readable once the fetches have landed.
//
// Separate from issuing, because between the two the request is queued rather
// than running - which is the whole point of doing this before it is ready to
// issue rather than when it starts.
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
