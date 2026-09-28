#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_kv_page_cache.h"

#define LRU_SEQUENCES 6u
#define LRU_BLOCKS 24u
#define LRU_BLOCK_TOKENS 4u
#define LRU_ROUNDS 300u
#define LRU_STEPS 600u
#define LRU_ALPHABET 3u

#define CHECK(cond) do { if ( !(cond) ) { fprintf(stderr,"FAIL seed=%llu step=%u line=%u %s\n",(unsigned long long)lru_seed,lru_step,__LINE__,#cond); exit(1); } } while (0)

typedef struct LruHarness
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[LRU_BLOCKS];
	SparkKvPageCacheEntry entries[LRU_BLOCKS];
	SparkKvPageCacheSequence sequences[LRU_SEQUENCES];
	uint32_t bucket_heads[LRU_BLOCKS],resident_slots[LRU_BLOCKS],entry_map[LRU_BLOCKS];
	uint8_t device[LRU_BLOCKS * 4u];
	SparkKvPageCache cache;
	SparkModelDriverCacheIdentity identity[LRU_SEQUENCES];
	uint64_t next_sequence_id;
} LruHarness;

static uint64_t lru_seed,lru_state;
static uint32_t lru_step,lru_evictions,lru_refusals,lru_prefix_binds;

static uint32_t LruRand(uint32_t bound)
{
	lru_state = lru_state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
	return((uint32_t)(lru_state >> 33u) % bound);
}

static void LruInit(LruHarness *h)
{
	SparkKvCacheConfiguration arena = {0};
	SparkKvPageCacheConfiguration config = {0};
	memset(h,0,sizeof(*h));
	arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	arena.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	arena.logical_block_count = arena.resident_block_capacity = LRU_BLOCKS;
	arena.block_token_count = LRU_BLOCK_TOKENS;
	arena.layer_count = arena.kv_head_count = arena.head_dim = arena.bytes_per_scalar = 1u;
	arena.key_block_stride_bytes = 4u;
	arena.key_device_base = h->device;
	arena.blocks = h->blocks;
	arena.resident_slot_logical_block_indices = h->resident_slots;
	CHECK(SparkKvCacheArenaInitialize(&h->arena,&arena) == SPARK_STATUS_OK);
	config.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	config.sequence_capacity = LRU_SEQUENCES;
	config.entry_capacity = config.hash_bucket_count = LRU_BLOCKS;
	config.kv_cache_arena = &h->arena;
	config.entries = h->entries;
	config.sequences = h->sequences;
	config.hash_bucket_heads = h->bucket_heads;
	config.entry_indices_by_logical_page = h->entry_map;
	CHECK(SparkKvPageCacheInitialize(&h->cache,&config) == SPARK_STATUS_OK);
	h->next_sequence_id = 1000u;
}

static uint32_t LruUnreferenced(const SparkKvPageCacheEntry *entry)
{
	return((entry->flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && entry->reference_count == 0u ? 1u : 0u);
}

static uint32_t LruDiscardable(const LruHarness *h, uint32_t page)
{
	const SparkKvCacheBlock *block;
	block = &h->blocks[page];
	return((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u && block->reference_count == 1u && block->residency_reference_count == 0u && (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) == 0u ? 1u : 0u);
}

static void LruCheckList(const LruHarness *h)
{
	uint32_t index,count,members,previous;
	uint64_t epoch;
	count = members = 0u;
	previous = SPARK_KV_PAGE_CACHE_NO_INDEX;
	epoch = 0u;
	for (index=h->cache.lru_head; index != SPARK_KV_PAGE_CACHE_NO_INDEX; index=h->entries[index].lru_next)
	{
		CHECK(index < LRU_BLOCKS && count < LRU_BLOCKS);
		CHECK(h->entries[index].lru_prev == previous && LruUnreferenced(&h->entries[index]) != 0u);
		CHECK(count == 0u || h->entries[index].last_used_epoch > epoch);
		epoch = h->entries[index].last_used_epoch;
		previous = index;
		count++;
	}
	CHECK(h->cache.lru_tail == previous);
	for (index=0u; index<LRU_BLOCKS; index++)
	{
		members += LruUnreferenced(&h->entries[index]);
		if ( LruUnreferenced(&h->entries[index]) == 0u )
			CHECK(h->entries[index].lru_prev == SPARK_KV_PAGE_CACHE_NO_INDEX && h->entries[index].lru_next == SPARK_KV_PAGE_CACHE_NO_INDEX);
	}
	CHECK(members == count);
}

static uint32_t LruReferenceVictim(const LruHarness *h)
{
	uint32_t index,victim;
	victim = SPARK_KV_PAGE_CACHE_NO_INDEX;
	for (index=0u; index<LRU_BLOCKS; index++)
		if ( LruUnreferenced(&h->entries[index]) != 0u && LruDiscardable(h,h->entries[index].logical_page_index) != 0u && (victim == SPARK_KV_PAGE_CACHE_NO_INDEX || h->entries[index].last_used_epoch < h->entries[victim].last_used_epoch) )
			victim = index;
	return(victim);
}

static uint32_t LruValidCount(const LruHarness *h)
{
	uint32_t index,count;
	count = 0u;
	for (index=0u; index<LRU_BLOCKS; index++)
		count += (h->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u ? 1u : 0u;
	return(count);
}

static void LruEvict(LruHarness *h)
{
	uint32_t victim,valid;
	SparkStatus status;
	victim = LruReferenceVictim(h);
	valid = LruValidCount(h);
	status = SparkKvPageCacheEvictUnused(&h->cache);
	if ( victim == SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		CHECK(status == SPARK_STATUS_CAPACITY_EXCEEDED);
		lru_refusals++;
		return;
	}
	CHECK(status == SPARK_STATUS_OK && (h->entries[victim].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) == 0u && LruValidCount(h) == valid - 1u);
	lru_evictions++;
}

static void LruIdentity(SparkModelDriverCacheIdentity *identity, const SparkModelDriverCacheIdentity *parent, uint32_t token, uint32_t context)
{
	uint64_t state;
	uint32_t index;
	state = UINT64_C(1469598103934665603) ^ ((uint64_t)token << 32u) ^ context;
	for (index=0u; index<sizeof(*parent); index++)
		state = (state ^ ((const uint8_t *)parent)[index]) * UINT64_C(1099511628211);
	for (index=0u; index<sizeof(*identity); index++)
	{
		state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
		((uint8_t *)identity)[index] = (uint8_t)(state >> 56u);
	}
}

static uint32_t LruPrefixEntry(const LruHarness *h)
{
	uint32_t start,index,candidate;
	start = LruRand(LRU_BLOCKS);
	for (index=0u; index<LRU_BLOCKS; index++)
	{
		candidate = (start + index) % LRU_BLOCKS;
		if ( (h->entries[candidate].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && h->entries[candidate].token_count % LRU_BLOCK_TOKENS == 0u )
			return(candidate);
	}
	return(SPARK_KV_PAGE_CACHE_NO_INDEX);
}

static void LruStartLane(LruHarness *h, SparkModelDriverCacheLane *lane, uint32_t slot)
{
	uint32_t prefix;
	memset(&h->identity[slot],0,sizeof(h->identity[slot]));
	lane->sequence_id = h->next_sequence_id++;
	lane->sequence_position = 0u;
	prefix = LruRand(2u) == 0u ? LruPrefixEntry(h) : SPARK_KV_PAGE_CACHE_NO_INDEX;
	if ( prefix == SPARK_KV_PAGE_CACHE_NO_INDEX )
		return;
	lane->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane->prefix_token_count = h->entries[prefix].token_count;
	lane->prefix_identity = h->entries[prefix].identity;
	lane->sequence_position = h->entries[prefix].token_count;
	h->identity[slot] = h->entries[prefix].identity;
	lru_prefix_binds++;
}

static void LruStep(LruHarness *h, uint32_t slot)
{
	SparkModelDriverCacheLane lane;
	SparkKvPageCacheSequence *sequence;
	uint32_t page,room;
	memset(&lane,0,sizeof(lane));
	sequence = &h->sequences[slot];
	lane.request_generation = lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.sequence_id = sequence->sequence_id;
	lane.sequence_position = sequence->next_token_position;
	if ( sequence->sequence_id == 0u )
		LruStartLane(h,&lane,slot);
	room = LRU_BLOCK_TOKENS - (uint32_t)(lane.sequence_position % LRU_BLOCK_TOKENS);
	lane.context_token_count = lane.publish_token_count = (uint32_t)lane.sequence_position + 1u + LruRand(room);
	lane.flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	LruIdentity(&lane.publish_identity,&h->identity[slot],LruRand(LRU_ALPHABET),lane.context_token_count);
	if ( SparkKvPageCacheBeginLane(&h->cache,&lane,&page) != SPARK_STATUS_OK )
		return;
	CHECK(SparkKvPageCacheCompleteLane(&h->cache,&lane) == SPARK_STATUS_OK);
	if ( lane.context_token_count % LRU_BLOCK_TOKENS == 0u )
		h->identity[slot] = lane.publish_identity;
}

static void LruRelease(LruHarness *h, uint32_t slot)
{
	if ( h->sequences[slot].sequence_id != 0u )
		CHECK(SparkKvPageCacheReleaseLane(&h->cache,slot,h->sequences[slot].sequence_id) == SPARK_STATUS_OK);
}

static void LruRound(void)
{
	LruHarness h;
	uint32_t operation,slot;
	LruInit(&h);
	for (lru_step=0u; lru_step<LRU_STEPS; lru_step++)
	{
		operation = LruRand(10u);
		slot = LruRand(LRU_SEQUENCES);
		if ( operation < 6u )
			LruStep(&h,slot);
		else if ( operation < 8u )
			LruRelease(&h,slot);
		else
			LruEvict(&h);
		LruCheckList(&h);
	}
	for (slot=0u; slot<LRU_SEQUENCES; slot++)
		LruRelease(&h,slot);
	LruCheckList(&h);
	while ( LruReferenceVictim(&h) != SPARK_KV_PAGE_CACHE_NO_INDEX )
	{
		LruEvict(&h);
		LruCheckList(&h);
	}
	CHECK(LruValidCount(&h) == 0u && h.cache.lru_head == SPARK_KV_PAGE_CACHE_NO_INDEX);
}

int main(void)
{
	uint32_t round;
	for (round=0u; round<LRU_ROUNDS; round++)
	{
		lru_seed = 17u + round;
		lru_state = lru_seed;
		LruRound();
	}
	CHECK(lru_evictions > 1000u && lru_refusals > 0u && lru_prefix_binds > 1000u);
	printf("test_kv_page_cache_lru PASS rounds=%u evictions=%u refusals=%u prefix_binds=%u\n",LRU_ROUNDS,lru_evictions,lru_refusals,lru_prefix_binds);
	return(0);
}
