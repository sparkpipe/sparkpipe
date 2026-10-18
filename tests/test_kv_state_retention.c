#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"

#define RETAIN_PAGES 256u
#define RETAIN_BLOCK_TOKENS 4u
#define RETAIN_PAGE_BYTES 32u
#define RETAIN_SEQUENCES 4u

typedef struct RetainFixture
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[RETAIN_PAGES];
	uint32_t resident_owners[RETAIN_PAGES];
	uint8_t device[RETAIN_PAGES * RETAIN_PAGE_BYTES];
	SparkKvPageCache cache;
	SparkKvPageCacheEntry entries[RETAIN_PAGES];
	SparkKvPageCacheSequence sequences[RETAIN_SEQUENCES];
	uint32_t hash_heads[RETAIN_PAGES];
	uint32_t entry_indices_by_logical_page[RETAIN_PAGES];
	SparkKvPageStore stores[2];
	uint8_t staging[2][RETAIN_PAGE_BYTES];
	char paths[2][64];
	uint8_t record[RETAIN_PAGE_BYTES];
	uint64_t pressure_writes;
}
RetainFixture;

static void RetainIdentity(SparkModelDriverCacheIdentity *identity,uint32_t chain,uint32_t step)
{
	memset(identity,0,sizeof(*identity));
	identity->sha256[0] = (uint8_t)chain;
	identity->sha256[1] = (uint8_t)(step & 0xffu);
	identity->sha256[2] = (uint8_t)(step >> 8);
	identity->sha256[3] = 0xa5u;
}

static void RetainOpen(RetainFixture *fixture,uint32_t state_slots,uint32_t anchor_tokens)
{
	SparkKvCacheConfiguration arena = {0};
	SparkKvPageCacheConfiguration cache = {0};
	SparkKvPageStoreConfiguration store = {0};
	uint32_t index;
	int descriptor;
	memset(fixture,0,sizeof(*fixture));
	arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	arena.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	arena.logical_block_count = RETAIN_PAGES;
	arena.block_token_count = RETAIN_BLOCK_TOKENS;
	arena.resident_block_capacity = RETAIN_PAGES;
	arena.layer_count = 1u;
	arena.kv_head_count = 1u;
	arena.head_dim = 8u;
	arena.bytes_per_scalar = 1u;
	arena.key_block_stride_bytes = RETAIN_PAGE_BYTES;
	arena.key_device_base = fixture->device;
	arena.blocks = fixture->blocks;
	arena.resident_slot_logical_block_indices = fixture->resident_owners;
	assert(SparkKvCacheArenaInitialize(&fixture->arena,&arena) == SPARK_STATUS_OK);
	cache.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	cache.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	cache.sequence_capacity = RETAIN_SEQUENCES;
	cache.entry_capacity = RETAIN_PAGES;
	cache.hash_bucket_count = RETAIN_PAGES;
	cache.kv_cache_arena = &fixture->arena;
	cache.entries = fixture->entries;
	cache.sequences = fixture->sequences;
	cache.hash_bucket_heads = fixture->hash_heads;
	cache.entry_indices_by_logical_page = fixture->entry_indices_by_logical_page;
	assert(SparkKvPageCacheInitialize(&fixture->cache,&cache) == SPARK_STATUS_OK);
	store.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	store.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	store.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	store.logical_page_capacity = RETAIN_PAGES;
	store.transfer_capacity = 1u;
	store.page_bytes = store.staging_bytes = RETAIN_PAGE_BYTES;
	for (index=0u; index<2u; index++)
	{
		strcpy(fixture->paths[index],"/tmp/sparkpipe-kv-retain-XXXXXX");
		descriptor = mkstemp(fixture->paths[index]);
		assert(descriptor >= 0 && close(descriptor) == 0 && unlink(fixture->paths[index]) == 0);
		store.backing_path = fixture->paths[index];
		store.staging_address = fixture->staging[index];
		store.maximum_backing_bytes = (uint64_t)(index == 0u ? RETAIN_PAGES : state_slots) * RETAIN_PAGE_BYTES;
		assert(SparkKvPageStoreInitialize(&fixture->stores[index],&store) == SPARK_STATUS_OK);
	}
	fixture->cache.page_store = &fixture->stores[0];
	assert(SparkKvPageCacheAttachStateStore(&fixture->cache,&fixture->stores[1],anchor_tokens) == SPARK_STATUS_OK);
	for (index=0u; index<RETAIN_PAGE_BYTES; index++)
		fixture->record[index] = (uint8_t)(0x40u + index);
}

static void RetainClose(RetainFixture *fixture)
{
	uint32_t index;
	assert(SparkKvPageCacheReleaseAll(&fixture->cache) == SPARK_STATUS_OK);
	for (index=0u; index<2u; index++)
	{
		SparkKvPageStoreDestroy(&fixture->stores[index]);
		assert(unlink(fixture->paths[index]) == 0);
	}
}

static SparkStatus RetainWriteRecord(RetainFixture *fixture,uint32_t page)
{
	SparkKvPageStore *store = &fixture->stores[1];
	SparkStatus status;
	for (;;)
	{
		status = SparkKvPageStoreWriteback(store,page,0u,fixture->blocks[page].generation,(uintptr_t)fixture->record,RETAIN_PAGE_BYTES,0u,0u);
		if ( status == SPARK_STATUS_BUSY )
			status = SparkKvPageStoreWaitForTransfers(store);
		else if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
		{
			fixture->pressure_writes++;
			status = SparkKvPageCacheDemoteState(&fixture->cache,0,0u);
		}
		else
			break;
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageStoreWaitForTransfers(store);
	return(status);
}

static void RetainLane(SparkModelDriverCacheLane *lane,uint64_t sequence_id,uint32_t slot,uint32_t chain,uint32_t step,uint32_t stateless)
{
	memset(lane,0,sizeof(*lane));
	lane->sequence_id = sequence_id;
	lane->request_generation = 1u;
	lane->step_generation = 1u;
	lane->resident_sequence_slot = slot;
	lane->sequence_position = (step - 1u) * RETAIN_BLOCK_TOKENS;
	lane->context_token_count = step * RETAIN_BLOCK_TOKENS;
	lane->flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH | (stateless != 0u ? SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_STATELESS : 0u);
	lane->publish_token_count = step * RETAIN_BLOCK_TOKENS;
	RetainIdentity(&lane->publish_identity,chain,step);
	if ( step > 1u )
	{
		lane->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
		lane->prefix_token_count = (step - 1u) * RETAIN_BLOCK_TOKENS;
		RetainIdentity(&lane->prefix_identity,chain,step - 1u);
	}
}

static uint32_t RetainBegin(RetainFixture *fixture,const SparkModelDriverCacheLane *lane)
{
	uint32_t page = SPARK_KV_CACHE_NO_BLOCK;
	assert(SparkKvPageCacheBeginLane(&fixture->cache,lane,&page) == SPARK_STATUS_OK && page < RETAIN_PAGES);
	return(page);
}

static void RetainStep(RetainFixture *fixture,uint64_t sequence_id,uint32_t chain,uint32_t step,uint32_t stateless)
{
	SparkModelDriverCacheLane lane;
	uint32_t page;
	RetainLane(&lane,sequence_id,0u,chain,step,stateless);
	page = RetainBegin(fixture,&lane);
	if ( stateless == 0u )
		assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&fixture->cache,0u,sequence_id) == SPARK_STATUS_OK);
}

static uint32_t RetainResumable(RetainFixture *fixture,uint32_t chain,uint32_t step)
{
	SparkModelDriverCacheIdentity identity;
	RetainIdentity(&identity,chain,step);
	return(SparkKvPageCachePrefixReady(&fixture->cache,&identity,step * RETAIN_BLOCK_TOKENS));
}

static void TestChainRetention(void)
{
	RetainFixture *fixture = calloc(1u,sizeof(*fixture));
	SparkModelDriverCacheLane lane;
	const uint32_t steps = 200u,anchor_blocks = 32u;
	uint32_t step,kept,expected;
	assert(fixture != 0);
	RetainOpen(fixture,12u,anchor_blocks * RETAIN_BLOCK_TOKENS);
	for (step=1u; step<=steps; step++)
	{
		RetainStep(fixture,1000u + step,1u,step,0u);
		expected = (step < SPARK_KV_PAGE_CACHE_RECENT_STATES ? step : SPARK_KV_PAGE_CACHE_RECENT_STATES) +
			(step > SPARK_KV_PAGE_CACHE_RECENT_STATES ? (step - SPARK_KV_PAGE_CACHE_RECENT_STATES) / anchor_blocks : 0u);
		assert(fixture->stores[1].backing_page_count == expected);
	}
	assert(fixture->pressure_writes == 0u && fixture->cache.state_pressure_demoted_count == 0u);
	for (step=1u,kept=0u; step<=steps; step++)
	{
		expected = step > steps - SPARK_KV_PAGE_CACHE_RECENT_STATES || step % anchor_blocks == 0u ? 1u : 0u;
		assert(RetainResumable(fixture,1u,step) == expected);
		kept += expected;
	}
	assert(kept == fixture->stores[1].backing_page_count && fixture->cache.state_superseded_count == steps - kept);
	RetainLane(&lane,5000u,1u,1u,steps + 1u,1u);
	(void)RetainBegin(fixture,&lane);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&fixture->cache,1u,5000u) == SPARK_STATUS_OK);
	RetainLane(&lane,5001u,1u,1u,51u,0u);
	assert(SparkKvPageCacheBeginLane(&fixture->cache,&lane,&step) == SPARK_STATUS_NOT_FOUND);
	printf("state retention: a 200-checkpoint chain keeps the newest %u records plus every %u-block anchor, releases each superseded record as the next one lands, never needs pressure demotion, resumes at kept points and refuses a released one: ok\n",
		SPARK_KV_PAGE_CACHE_RECENT_STATES,anchor_blocks);
	RetainClose(fixture);
	free(fixture);
}

static void TestPressureOrder(void)
{
	RetainFixture *fixture = calloc(1u,sizeof(*fixture));
	SparkModelDriverCacheLane live,next;
	uint32_t step,page;
	assert(fixture != 0);
	RetainOpen(fixture,8u,RETAIN_PAGES * RETAIN_BLOCK_TOKENS);
	for (step=1u; step<=4u; step++)
		RetainStep(fixture,100u + step,1u,step,0u);
	for (step=1u; step<=4u; step++)
		RetainStep(fixture,200u + step,2u,step,0u);
	assert(fixture->stores[1].backing_page_count == 8u);
	RetainLane(&live,300u,2u,1u,5u,0u);
	page = RetainBegin(fixture,&live);
	for (step=1u; step<=4u; step++)
	{
		RetainLane(&next,400u + step,1u,3u,step,0u);
		(void)RetainBegin(fixture,&next);
		assert(RetainWriteRecord(fixture,fixture->cache.sequences[1].mutable_logical_page_index) == SPARK_STATUS_OK);
		assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK);
		assert(SparkKvPageCacheReleaseLane(&fixture->cache,1u,400u + step) == SPARK_STATUS_OK);
		assert(RetainResumable(fixture,2u,step) == 0u && RetainResumable(fixture,1u,4u) == 1u);
	}
	assert(RetainResumable(fixture,1u,1u) == 1u && RetainResumable(fixture,1u,2u) == 1u && RetainResumable(fixture,1u,3u) == 1u);
	assert(fixture->cache.state_pressure_demoted_count == 4u && fixture->pressure_writes == 4u);
	RetainLane(&next,500u,1u,3u,5u,0u);
	(void)RetainBegin(fixture,&next);
	assert(RetainWriteRecord(fixture,fixture->cache.sequences[1].mutable_logical_page_index) == SPARK_STATUS_OK);
	assert(RetainResumable(fixture,1u,4u) == 1u && RetainResumable(fixture,3u,4u) == 1u && fixture->cache.state_pinned_skip_count == 0u);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK && RetainResumable(fixture,3u,5u) == 1u && RetainResumable(fixture,3u,1u) == 0u);
	assert(SparkKvPageCacheReleaseLane(&fixture->cache,1u,500u) == SPARK_STATUS_OK);
	assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK && SparkKvPageCacheCompleteLane(&fixture->cache,&live) == SPARK_STATUS_OK);
	assert(RetainResumable(fixture,1u,5u) == 1u && RetainResumable(fixture,1u,1u) == 0u);
	assert(SparkKvPageCacheReleaseLane(&fixture->cache,2u,300u) == SPARK_STATUS_OK);
	printf("state pressure: a full store releases the records of a chain no lane uses, oldest first, before any record on a bound lane's path, and never the record a bound lane resumes from: ok\n");
	RetainClose(fixture);
	free(fixture);
}

static void TestRekeyOntoStatelessEntry(void)
{
	RetainFixture *fixture = calloc(1u,sizeof(*fixture));
	assert(fixture != 0);
	RetainOpen(fixture,4u,RETAIN_PAGES * RETAIN_BLOCK_TOKENS);
	RetainStep(fixture,10u,7u,1u,1u);
	assert(RetainResumable(fixture,7u,1u) == 0u && fixture->stores[1].backing_page_count == 0u);
	RetainStep(fixture,11u,7u,1u,0u);
	assert(RetainResumable(fixture,7u,1u) == 1u && fixture->stores[1].backing_page_count == 1u);
	assert(fixture->cache.state_rekeyed_count == 1u && fixture->cache.stateful_dedup_lost_count == 0u && fixture->cache.deduplicated_page_count == 1u);
	RetainStep(fixture,12u,7u,2u,0u);
	assert(RetainResumable(fixture,7u,2u) == 1u && RetainResumable(fixture,7u,1u) == 1u);
	printf("state rekey: a checkpoint published onto an existing stateless entry of the same prefix moves its record onto that entry instead of losing it: ok\n");
	RetainClose(fixture);
	free(fixture);
}

int main(void)
{
	TestChainRetention();
	TestPressureOrder();
	TestRekeyOntoStatelessEntry();
	printf("PASS kv state retention\n");
	return(0);
}
