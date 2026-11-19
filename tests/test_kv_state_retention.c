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

typedef struct FoldCopy
{
	SparkKvCacheArena *arena;
	uint32_t copies;
}
FoldCopy;

static SparkStatus FoldCopyPage(void *context,uint32_t source,uint32_t destination)
{
	FoldCopy *copy = (FoldCopy *)context;
	SparkKvCacheBlockView from,to;
	if ( SparkKvCacheArenaResolveBlock(copy->arena,source,&from) != SPARK_STATUS_OK || SparkKvCacheArenaResolveBlock(copy->arena,destination,&to) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	memcpy((void *)to.key_device_address,(const void *)from.key_device_address,RETAIN_PAGE_BYTES);
	assert(SparkKvCacheArenaUnpinResidentBlock(copy->arena,source) == SPARK_STATUS_OK && SparkKvCacheArenaUnpinResidentBlock(copy->arena,destination) == SPARK_STATUS_OK);
	copy->copies++;
	return(SPARK_STATUS_OK);
}

static SparkStatus FoldCopyRetire(void *context,uint32_t require_all)
{
	(void)context;
	(void)require_all;
	return(SPARK_STATUS_OK);
}

static uint32_t FoldCopyPins(void *context,uint32_t page)
{
	(void)context;
	(void)page;
	return(0u);
}

static SparkStatus FoldCopyDeferFree(void *context,uint32_t page)
{
	(void)context;
	(void)page;
	return(SPARK_STATUS_NOT_FOUND);
}

static void FoldLane(SparkModelDriverCacheLane *lane,uint64_t sequence_id,uint32_t slot,uint32_t chain,uint32_t prompt,SparkModelDriverCacheIdentity *blocks)
{
	memset(lane,0,sizeof(*lane));
	lane->sequence_id = sequence_id;
	lane->request_generation = 1u;
	lane->step_generation = 1u;
	lane->resident_sequence_slot = slot;
	lane->sequence_position = 0u;
	lane->context_token_count = prompt;
	lane->publish_token_count = prompt - 1u;
	lane->flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_FOLDED;
	RetainIdentity(&lane->publish_identity,chain,prompt - 1u);
	RetainIdentity(&blocks[0],chain,RETAIN_BLOCK_TOKENS);
	lane->block_identity_count = (prompt - 2u) / RETAIN_BLOCK_TOKENS;
	lane->block_identities = blocks;
}

static void FoldDecodeLane(SparkModelDriverCacheLane *lane,uint64_t sequence_id,uint32_t slot,uint32_t position)
{
	memset(lane,0,sizeof(*lane));
	lane->sequence_id = sequence_id;
	lane->request_generation = 1u;
	lane->step_generation = 2u;
	lane->resident_sequence_slot = slot;
	lane->sequence_position = position;
	lane->context_token_count = position + 1u;
}

static uint8_t *FoldTokenBytes(RetainFixture *fixture,uint32_t page,uint32_t position)
{
	return(fixture->device + (uint64_t)page * RETAIN_PAGE_BYTES + (uint64_t)(position % RETAIN_BLOCK_TOKENS) * (RETAIN_PAGE_BYTES / RETAIN_BLOCK_TOKENS));
}

static void TestFoldedPublish(void)
{
	RetainFixture *fixture = calloc(1u,sizeof(*fixture));
	FoldCopy copy = {0};
	SparkKvPageCacheDeviceCopy hook = {FoldCopyPage,FoldCopyRetire,FoldCopyPins,FoldCopyDeferFree,&copy};
	SparkModelDriverCacheIdentity blocks[2],identity;
	SparkModelDriverCacheLane lane,next;
	const uint32_t prompt = 7u,unit = RETAIN_PAGE_BYTES / RETAIN_BLOCK_TOKENS;
	uint32_t page,clone,terminal;
	assert(fixture != 0);
	RetainOpen(fixture,8u,RETAIN_PAGES * RETAIN_BLOCK_TOKENS);
	copy.arena = &fixture->arena;
	assert(SparkKvPageCacheAttachDeviceCopy(&fixture->cache,&hook) == SPARK_STATUS_OK);
	FoldLane(&lane,60u,0u,9u,prompt,blocks);
	(void)RetainBegin(fixture,&lane);
	page = fixture->cache.sequences[0].mutable_following_pages[0];
	memset(FoldTokenBytes(fixture,page,prompt - 1u),0x6a,unit);
	assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	terminal = fixture->cache.sequences[0].terminal_entry_index;
	assert(fixture->cache.folded_publish_count == 1u && fixture->cache.folded_duplicate_count == 0u);
	assert(fixture->cache.sequences[0].folded_token_count == 1u && fixture->cache.sequences[0].next_token_position == prompt &&
		fixture->cache.sequences[0].mutable_page_count == 0u);
	assert(fixture->entries[terminal].token_count == prompt - 1u && fixture->entries[terminal].logical_page_index == page);
	RetainIdentity(&identity,9u,prompt - 1u);
	assert(SparkKvPageCachePrefixReady(&fixture->cache,&identity,prompt - 1u) == 1u);
	FoldDecodeLane(&next,60u,0u,prompt);
	clone = RetainBegin(fixture,&next);
	assert(clone != page && copy.copies == 1u && fixture->cache.sequences[0].folded_token_count == 0u);
	assert(memcmp(FoldTokenBytes(fixture,clone,prompt - 1u),FoldTokenBytes(fixture,page,prompt - 1u),unit) == 0 && FoldTokenBytes(fixture,clone,prompt - 1u)[0] == 0x6a);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK && fixture->cache.sequences[0].next_token_position == prompt + 1u);
	memset(&next,0,sizeof(next));
	next.sequence_id = 61u;
	next.request_generation = 1u;
	next.step_generation = 1u;
	next.resident_sequence_slot = 1u;
	next.sequence_position = prompt - 1u;
	next.context_token_count = prompt;
	next.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	next.prefix_token_count = prompt - 1u;
	next.prefix_identity = identity;
	clone = RetainBegin(fixture,&next);
	assert(clone != page && copy.copies == 2u && fixture->cache.sequences[1].folded_token_count == 0u);
	memset(FoldTokenBytes(fixture,clone,prompt - 1u),0x5b,unit);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK && FoldTokenBytes(fixture,page,prompt - 1u)[0] == 0x6a);
	FoldLane(&lane,62u,2u,9u,prompt,blocks);
	(void)RetainBegin(fixture,&lane);
	page = fixture->cache.sequences[2].mutable_following_pages[0];
	memset(FoldTokenBytes(fixture,page,prompt - 1u),0x7c,unit);
	assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	assert(fixture->cache.folded_duplicate_count == 1u && fixture->cache.sequences[2].mutable_page_count == 1u &&
		fixture->cache.sequences[2].mutable_logical_page_index == page && fixture->cache.sequences[2].folded_token_count == 0u);
	assert(SparkKvPageStoreValidateRecord(&fixture->stores[1],page,fixture->blocks[page].generation) == SPARK_STATUS_NOT_FOUND);
	FoldDecodeLane(&next,62u,2u,prompt);
	assert(RetainBegin(fixture,&next) == page && copy.copies == 2u && FoldTokenBytes(fixture,page,prompt - 1u)[0] == 0x7c);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK);
	FoldLane(&lane,63u,3u,10u,prompt,blocks);
	lane.flags &= ~SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_FOLDED;
	(void)RetainBegin(fixture,&lane);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_UNSUPPORTED);
	assert(SparkKvPageCacheReleaseLane(&fixture->cache,3u,63u) == SPARK_STATUS_OK);
	printf("folded publish: a step that ends one token past its checkpoint publishes the checkpoint with its state, the next step copies the page with that token, a resume at the checkpoint gets its own copy, a repeat keeps its page and drops its record, and a fold without the flag is refused: ok\n");
	RetainClose(fixture);
	free(fixture);
}

static void TestFoldedPublishOnBlockEdges(void)
{
	RetainFixture *fixture = calloc(1u,sizeof(*fixture));
	FoldCopy copy = {0};
	SparkKvPageCacheDeviceCopy hook = {FoldCopyPage,FoldCopyRetire,FoldCopyPins,FoldCopyDeferFree,&copy};
	SparkModelDriverCacheIdentity blocks[3],identity,whole;
	SparkModelDriverCacheLane lane,next;
	const uint32_t unit = RETAIN_PAGE_BYTES / RETAIN_BLOCK_TOKENS;
	uint32_t first,last,page,terminal,entry;
	assert(fixture != 0);
	RetainOpen(fixture,8u,RETAIN_PAGES * RETAIN_BLOCK_TOKENS);
	copy.arena = &fixture->arena;
	assert(SparkKvPageCacheAttachDeviceCopy(&fixture->cache,&hook) == SPARK_STATUS_OK);
	FoldLane(&lane,70u,0u,21u,2u * RETAIN_BLOCK_TOKENS + 1u,blocks);
	RetainIdentity(&blocks[1],21u,2u * RETAIN_BLOCK_TOKENS);
	lane.block_identity_count = 2u;
	(void)RetainBegin(fixture,&lane);
	page = fixture->cache.sequences[0].mutable_following_pages[0];
	last = fixture->cache.sequences[0].mutable_following_pages[1];
	memset(FoldTokenBytes(fixture,last,2u * RETAIN_BLOCK_TOKENS),0x3d,unit);
	assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	assert(fixture->cache.folded_publish_count == 1u && copy.copies == 0u);
	assert(fixture->cache.sequences[0].mutable_page_count == 1u && fixture->cache.sequences[0].mutable_logical_page_index == last &&
		fixture->cache.sequences[0].mutable_first_token_index == 2u * RETAIN_BLOCK_TOKENS && fixture->cache.sequences[0].folded_token_count == 0u &&
		fixture->cache.sequences[0].next_token_position == 2u * RETAIN_BLOCK_TOKENS + 1u);
	RetainIdentity(&identity,21u,2u * RETAIN_BLOCK_TOKENS);
	assert(SparkKvPageCachePrefixReady(&fixture->cache,&identity,2u * RETAIN_BLOCK_TOKENS) == 1u);
	FoldDecodeLane(&next,70u,0u,2u * RETAIN_BLOCK_TOKENS + 1u);
	assert(RetainBegin(fixture,&next) == last && copy.copies == 0u && FoldTokenBytes(fixture,last,2u * RETAIN_BLOCK_TOKENS)[0] == 0x3d);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK);
	FoldLane(&lane,71u,1u,22u,2u * RETAIN_BLOCK_TOKENS,blocks);
	RetainIdentity(&blocks[1],22u,2u * RETAIN_BLOCK_TOKENS);
	lane.block_identity_count = 2u;
	(void)RetainBegin(fixture,&lane);
	first = fixture->cache.sequences[1].mutable_logical_page_index;
	page = fixture->cache.sequences[1].mutable_following_pages[0];
	memset(FoldTokenBytes(fixture,page,2u * RETAIN_BLOCK_TOKENS - 2u),0x4e,unit);
	memset(FoldTokenBytes(fixture,page,2u * RETAIN_BLOCK_TOKENS - 1u),0x5f,unit);
	assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	assert(fixture->cache.folded_publish_count == 2u && copy.copies == 1u);
	terminal = fixture->cache.sequences[1].terminal_entry_index;
	assert(fixture->cache.sequences[1].mutable_page_count == 0u && fixture->cache.sequences[1].next_token_position == 2u * RETAIN_BLOCK_TOKENS &&
		fixture->entries[terminal].token_count == 2u * RETAIN_BLOCK_TOKENS && fixture->entries[terminal].logical_page_index == page &&
		fixture->entries[fixture->entries[terminal].parent_entry_index].logical_page_index == first);
	RetainIdentity(&identity,22u,2u * RETAIN_BLOCK_TOKENS - 1u);
	RetainIdentity(&whole,22u,2u * RETAIN_BLOCK_TOKENS);
	assert(SparkKvPageCachePrefixReady(&fixture->cache,&identity,2u * RETAIN_BLOCK_TOKENS - 1u) == 1u);
	assert(SparkKvPageCachePrefixReady(&fixture->cache,&whole,2u * RETAIN_BLOCK_TOKENS) == 0u);
	for (entry=0u; entry<RETAIN_PAGES; entry++)
		if ( fixture->entries[entry].flags != 0u && fixture->entries[entry].token_count == 2u * RETAIN_BLOCK_TOKENS - 1u &&
			memcmp(&fixture->entries[entry].identity,&identity,sizeof(identity)) == 0 )
			break;
	assert(entry < RETAIN_PAGES && fixture->entries[entry].logical_page_index != page &&
		fixture->entries[entry].parent_entry_index == fixture->entries[terminal].parent_entry_index &&
		memcmp(FoldTokenBytes(fixture,fixture->entries[entry].logical_page_index,2u * RETAIN_BLOCK_TOKENS - 2u),FoldTokenBytes(fixture,page,2u * RETAIN_BLOCK_TOKENS - 2u),unit) == 0);
	assert(SparkKvPageStoreValidateRecord(&fixture->stores[1],page,fixture->blocks[page].generation) == SPARK_STATUS_NOT_FOUND);
	FoldDecodeLane(&next,71u,1u,2u * RETAIN_BLOCK_TOKENS);
	(void)RetainBegin(fixture,&next);
	assert(copy.copies == 1u && fixture->cache.sequences[1].mutable_first_token_index == 2u * RETAIN_BLOCK_TOKENS);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&next) == SPARK_STATUS_OK);
	FoldLane(&lane,72u,2u,22u,2u * RETAIN_BLOCK_TOKENS,blocks);
	RetainIdentity(&blocks[1],22u,2u * RETAIN_BLOCK_TOKENS);
	lane.block_identity_count = 2u;
	(void)RetainBegin(fixture,&lane);
	page = fixture->cache.sequences[2].mutable_following_pages[0];
	assert(RetainWriteRecord(fixture,page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
	assert(copy.copies == 1u && fixture->cache.folded_duplicate_count == 1u && fixture->cache.sequences[2].mutable_page_count == 0u &&
		fixture->cache.sequences[2].next_token_position == 2u * RETAIN_BLOCK_TOKENS);
	assert(SparkKvPageCacheReleaseLane(&fixture->cache,2u,72u) == SPARK_STATUS_OK);
	printf("folded publish on block edges: a checkpoint that ends a block is that block's entry with its state and the next token stays in the open page; a prompt that ends a block keeps the whole block for itself and gives the checkpoint a copied page that holds the state, and a repeat copies nothing: ok\n");
	RetainClose(fixture);
	free(fixture);
}

int main(void)
{
	TestChainRetention();
	TestPressureOrder();
	TestRekeyOntoStatelessEntry();
	TestFoldedPublish();
	TestFoldedPublishOnBlockEdges();
	printf("PASS kv state retention\n");
	return(0);
}
