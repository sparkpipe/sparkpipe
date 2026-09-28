#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_kv_snapshot.h"

#define BENCH_BLOCK_TOKENS 64u

typedef struct BenchFixture
{
	uint32_t pages;
	uint64_t page_bytes;
	uint64_t state_bytes;
	SparkKvCacheArena arena;
	SparkKvCacheBlock *blocks;
	uint32_t *resident;
	uint8_t *device;
	SparkKvPageStore stores[2];
	uint8_t *staging[2];
	char paths[2][1100];
	SparkKvPageCache cache;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence sequences[4];
	uint32_t *heads;
	uint32_t *by_page;
	SparkKvPageCacheSnapshot snapshot;
	SparkKvPageCacheSnapshotLink *links;
	SparkKvSnapshotKey *keys;
	uint8_t *page;
	uint8_t *state;
} BenchFixture;

static uint64_t BenchNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static int32_t BenchStoreOpen(SparkKvPageStore *store,char *path,const char *directory,uint32_t index,uint8_t *staging,uint64_t page_bytes,uint32_t pages)
{
	SparkKvPageStoreConfiguration configuration = {0};
	(void)snprintf(path,1100,"%s/bench-backing-%ld-%u",directory,(long)getpid(),index);
	(void)unlink(path);
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = pages;
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = configuration.staging_bytes = page_bytes;
	configuration.maximum_backing_bytes = page_bytes * pages;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	return(SparkKvPageStoreInitialize(store,&configuration) == SPARK_STATUS_OK ? 0 : -1);
}

static int32_t BenchFixtureOpen(BenchFixture *fixture,SparkKvSnapshotStore *store,const char *directory,uint32_t flags)
{
	SparkKvCacheConfiguration arena = {0};
	SparkKvPageCacheConfiguration config = {0};
	uint32_t pages = fixture->pages;
	fixture->blocks = calloc(pages,sizeof(*fixture->blocks));
	fixture->resident = calloc(pages,sizeof(*fixture->resident));
	fixture->device = malloc((size_t)(pages * fixture->page_bytes));
	fixture->staging[0] = malloc((size_t)fixture->page_bytes);
	fixture->staging[1] = malloc((size_t)fixture->state_bytes);
	fixture->entries = calloc(pages,sizeof(*fixture->entries));
	fixture->heads = calloc(pages,sizeof(*fixture->heads));
	fixture->by_page = calloc(pages,sizeof(*fixture->by_page));
	fixture->links = calloc(pages,sizeof(*fixture->links));
	fixture->keys = calloc(pages,sizeof(*fixture->keys));
	fixture->page = malloc((size_t)fixture->page_bytes);
	fixture->state = malloc((size_t)fixture->state_bytes);
	if ( fixture->blocks == 0 || fixture->resident == 0 || fixture->device == 0 || fixture->staging[0] == 0 || fixture->staging[1] == 0 || fixture->entries == 0 || fixture->heads == 0 || fixture->by_page == 0 || fixture->links == 0 || fixture->keys == 0 || fixture->page == 0 || fixture->state == 0 )
		return(-1);
	if ( BenchStoreOpen(&fixture->stores[0],fixture->paths[0],directory,0u,fixture->staging[0],fixture->page_bytes,pages) != 0 || BenchStoreOpen(&fixture->stores[1],fixture->paths[1],directory,1u,fixture->staging[1],fixture->state_bytes,pages) != 0 )
		return(-2);
	arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	arena.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	arena.logical_block_count = arena.resident_block_capacity = pages;
	arena.block_token_count = BENCH_BLOCK_TOKENS;
	arena.layer_count = arena.kv_head_count = 1u;
	arena.bytes_per_scalar = (uint32_t)sizeof(uint8_t);
	arena.head_dim = (uint32_t)fixture->page_bytes;
	arena.key_block_stride_bytes = fixture->page_bytes;
	arena.key_device_base = fixture->device;
	arena.blocks = fixture->blocks;
	arena.resident_slot_logical_block_indices = fixture->resident;
	arena.evict_function = SparkKvPageStoreWriteback;
	arena.evict_context = &fixture->stores[0];
	if ( SparkKvCacheArenaInitialize(&fixture->arena,&arena) != SPARK_STATUS_OK )
		return(-3);
	config.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	config.sequence_capacity = 4u;
	config.entry_capacity = config.hash_bucket_count = pages;
	config.kv_cache_arena = &fixture->arena;
	config.page_store = &fixture->stores[0];
	config.entries = fixture->entries;
	config.sequences = fixture->sequences;
	config.hash_bucket_heads = fixture->heads;
	config.entry_indices_by_logical_page = fixture->by_page;
	if ( SparkKvPageCacheInitialize(&fixture->cache,&config) != SPARK_STATUS_OK || SparkKvPageCacheAttachStateStore(&fixture->cache,&fixture->stores[1]) != SPARK_STATUS_OK )
		return(-4);
	fixture->snapshot.store = store;
	memset(fixture->snapshot.layout_sha256,0x42,sizeof(fixture->snapshot.layout_sha256));
	fixture->snapshot.flags = flags;
	fixture->snapshot.page_capacity = pages;
	fixture->snapshot.links = fixture->links;
	fixture->snapshot.keys = fixture->keys;
	fixture->snapshot.page = fixture->page;
	fixture->snapshot.state = fixture->state;
	return(SparkKvPageCacheAttachSnapshot(&fixture->cache,&fixture->snapshot) == SPARK_STATUS_OK ? 0 : -5);
}

static void BenchFixtureClose(BenchFixture *fixture)
{
	SparkKvPageStoreDestroy(&fixture->stores[0]);
	SparkKvPageStoreDestroy(&fixture->stores[1]);
	(void)unlink(fixture->paths[0]);
	(void)unlink(fixture->paths[1]);
	free(fixture->blocks);
	free(fixture->resident);
	free(fixture->device);
	free(fixture->staging[0]);
	free(fixture->staging[1]);
	free(fixture->entries);
	free(fixture->heads);
	free(fixture->by_page);
	free(fixture->links);
	free(fixture->keys);
	free(fixture->page);
	free(fixture->state);
}

static void BenchIdentity(SparkModelDriverCacheIdentity *identity,uint32_t block)
{
	uint32_t index;
	for (index=0u; index<sizeof(identity->sha256); index++)
		identity->sha256[index] = (uint8_t)(block * 7u + index + 1u);
}

static void BenchFill(uint8_t *data,uint64_t bytes,uint32_t seed)
{
	uint64_t index;
	uint32_t value = seed * 2654435761u + 1u;
	for (index=0u; index<bytes; index++)
	{
		value = value * 1664525u + 1013904223u;
		data[index] = (uint8_t)(value >> 24);
	}
}

static SparkStatus BenchWriteState(SparkKvPageStore *store,uint32_t page,uint64_t generation,const uint8_t *source,uint64_t bytes)
{
	SparkStatus status = SparkKvPageStoreWriteback(store,page,0u,generation,(uintptr_t)source,bytes,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(store);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreWriteback(store,page,0u,generation,(uintptr_t)source,bytes,0u,0u);
	}
	return(status);
}

static SparkStatus BenchReadState(SparkKvPageStore *store,uint32_t page,uint64_t generation,uint8_t *output,uint64_t bytes)
{
	SparkStatus status = SparkKvPageStoreReadback(store,page,generation,(uintptr_t)output,bytes);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(store);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreReadback(store,page,generation,(uintptr_t)output,bytes);
	}
	return(status);
}

static int32_t BenchPrefill(BenchFixture *fixture,uint32_t blocks,uint8_t *state)
{
	SparkModelDriverCacheLane lane;
	SparkKvCacheBlockView view;
	uint32_t block,page;
	for (block=0u; block<blocks; block++)
	{
		memset(&lane,0,sizeof(lane));
		lane.sequence_id = 1u;
		lane.request_generation = 1u;
		lane.step_generation = block + 1u;
		lane.sequence_position = block * BENCH_BLOCK_TOKENS;
		lane.context_token_count = (block + 1u) * BENCH_BLOCK_TOKENS;
		lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
		lane.publish_token_count = lane.context_token_count;
		BenchIdentity(&lane.publish_identity,block);
		if ( SparkKvPageCacheBeginLane(&fixture->cache,&lane,&page) != SPARK_STATUS_OK || SparkKvCacheArenaResolveBlock(&fixture->arena,page,&view) != SPARK_STATUS_OK )
			return(-1);
		BenchFill((uint8_t *)view.key_device_address,fixture->page_bytes,block);
		BenchFill(state,fixture->state_bytes,block + 1000u);
		if ( BenchWriteState(&fixture->stores[1],page,view.generation,state,fixture->state_bytes) != SPARK_STATUS_OK || SparkKvPageCacheCompleteLane(&fixture->cache,&lane) != SPARK_STATUS_OK )
			return(-2);
	}
	return(0);
}

static void BenchDropCache(const char *directory)
{
	char path[1200];
	struct dirent *item;
	DIR *listing = opendir(directory);
	int descriptor;
	if ( listing == 0 )
		return;
	while ( (item = readdir(listing)) != 0 )
	{
		if ( item->d_name[0] == '.' )
			continue;
		(void)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name);
		descriptor = open(path,O_RDONLY);
		if ( descriptor < 0 )
			continue;
		(void)fdatasync(descriptor);
		(void)posix_fadvise(descriptor,0,0,POSIX_FADV_DONTNEED);
		(void)close(descriptor);
	}
	(void)closedir(listing);
}

static int32_t BenchVerify(BenchFixture *fixture,uint32_t blocks,uint8_t *expected,uint8_t *actual)
{
	SparkModelDriverCacheIdentity identity;
	SparkKvCacheBlockView view;
	uint32_t block,entry;
	for (block=0u; block<blocks; block++)
	{
		BenchIdentity(&identity,block);
		for (entry=0u; entry<fixture->pages; entry++)
			if ( (fixture->entries[entry].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && fixture->entries[entry].token_count == (block + 1u) * BENCH_BLOCK_TOKENS && memcmp(&fixture->entries[entry].identity,&identity,sizeof(identity)) == 0 )
				break;
		if ( entry == fixture->pages || SparkKvCacheArenaResolveBlock(&fixture->arena,fixture->entries[entry].logical_page_index,&view) != SPARK_STATUS_OK )
			return(-1);
		BenchFill(expected,fixture->page_bytes,block);
		if ( memcmp((const void *)view.key_device_address,expected,fixture->page_bytes) != 0 )
			return(-2);
		if ( block + 1u == blocks )
		{
			BenchFill(expected,fixture->state_bytes,block + 1000u);
			if ( BenchReadState(&fixture->stores[1],fixture->entries[entry].logical_page_index,view.generation,actual,fixture->state_bytes) != SPARK_STATUS_OK || memcmp(actual,expected,fixture->state_bytes) != 0 )
				return(-3);
		}
	}
	return(0);
}

int main(int argc,char **argv)
{
	static BenchFixture source,fresh;
	SparkKvSnapshotStore store;
	SparkModelDriverCacheIdentity terminal;
	uint64_t page_bytes,state_bytes,start,publish_ns,writer_ns,sync_cold_ns,sync_warm_ns,prefetch_first_ns,prefetch_background_ns,prefetch_join_ns,file_bytes,prefetch_bytes;
	uint32_t blocks,pages;
	uint8_t *scratch,*actual;
	SparkStatus status;
	int32_t failures = 0;
	if ( argc != 5 || argv[1][0] != '/' )
	{
		fprintf(stderr,"usage: %s ABSOLUTE_DIRECTORY PAGE_BYTES STATE_BYTES BLOCKS\n",argv[0]);
		return(2);
	}
	page_bytes = strtoull(argv[2],0,10);
	state_bytes = strtoull(argv[3],0,10);
	blocks = (uint32_t)strtoul(argv[4],0,10);
	if ( page_bytes == 0u || state_bytes == 0u || blocks == 0u )
		return(2);
	pages = blocks + 2u;
	scratch = malloc((size_t)(page_bytes > state_bytes ? page_bytes : state_bytes));
	actual = malloc((size_t)state_bytes);
	file_bytes = SPARK_KV_SNAPSHOT_ALIGNMENT * 4u + (uint64_t)pages * sizeof(SparkKvPageCacheSnapshotLink) + page_bytes + state_bytes;
	prefetch_bytes = 2u * ((uint64_t)pages * page_bytes + state_bytes);
	if ( scratch == 0 || actual == 0 || SparkKvSnapshotStoreOpen(&store,argv[1],UINT64_C(1) << 40u,32u * file_bytes) != SPARK_STATUS_OK )
		return(3);
	source.pages = pages;
	source.page_bytes = page_bytes;
	source.state_bytes = state_bytes;
	if ( BenchFixtureOpen(&source,&store,argv[1],0u) != 0 || BenchPrefill(&source,blocks,scratch) != 0 )
		return(4);
	BenchIdentity(&terminal,blocks - 1u);
	start = BenchNow();
	status = SparkKvPageCacheSavePrefix(&source.cache,&terminal,blocks * BENCH_BLOCK_TOKENS);
	publish_ns = BenchNow() - start;
	failures += status != SPARK_STATUS_OK;
	start = BenchNow();
	failures += SparkKvSnapshotFlush(&store) != SPARK_STATUS_OK || store.write_count != blocks || store.write_failure_count != 0u;
	writer_ns = BenchNow() - start + publish_ns;
	BenchFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);

	BenchDropCache(argv[1]);
	fresh.pages = pages;
	fresh.page_bytes = page_bytes;
	fresh.state_bytes = state_bytes;
	failures += SparkKvSnapshotStoreOpen(&store,argv[1],UINT64_C(1) << 40u,32u * file_bytes) != SPARK_STATUS_OK || BenchFixtureOpen(&fresh,&store,argv[1],0u) != 0;
	start = BenchNow();
	failures += SparkKvPageCacheRestorePrefix(&fresh.cache,&terminal,blocks * BENCH_BLOCK_TOKENS) != SPARK_STATUS_OK;
	sync_cold_ns = BenchNow() - start;
	failures += BenchVerify(&fresh,blocks,scratch,actual) != 0;
	BenchFixtureClose(&fresh);
	failures += BenchFixtureOpen(&fresh,&store,argv[1],0u) != 0;
	start = BenchNow();
	failures += SparkKvPageCacheRestorePrefix(&fresh.cache,&terminal,blocks * BENCH_BLOCK_TOKENS) != SPARK_STATUS_OK;
	sync_warm_ns = BenchNow() - start;
	BenchFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);

	BenchDropCache(argv[1]);
	failures += SparkKvSnapshotStoreOpen(&store,argv[1],UINT64_C(1) << 40u,32u * file_bytes) != SPARK_STATUS_OK || SparkKvSnapshotPrefetcherStart(&store,prefetch_bytes) != SPARK_STATUS_OK;
	failures += BenchFixtureOpen(&fresh,&store,argv[1],SPARK_KV_PAGE_CACHE_SNAPSHOT_FLAG_PREFETCH_JOIN) != 0;
	start = BenchNow();
	failures += SparkKvPageCacheRestorePrefix(&fresh.cache,&terminal,blocks * BENCH_BLOCK_TOKENS) != SPARK_STATUS_PENDING;
	prefetch_first_ns = BenchNow() - start;
	start = BenchNow();
	failures += SparkKvSnapshotPrefetchWait(&store) != SPARK_STATUS_OK;
	prefetch_background_ns = BenchNow() - start;
	start = BenchNow();
	failures += SparkKvPageCacheRestorePrefix(&fresh.cache,&terminal,blocks * BENCH_BLOCK_TOKENS) != SPARK_STATUS_OK;
	prefetch_join_ns = BenchNow() - start;
	failures += BenchVerify(&fresh,blocks,scratch,actual) != 0;
	BenchFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);
	printf("kv_snapshot_bench blocks=%u page_bytes=%llu state_bytes=%llu file_bytes=%llu publish_thread_us=%llu publish_thread_us_per_block=%llu writer_us=%llu writer_us_per_block=%llu sync_restore_cold_us=%llu sync_restore_warm_us=%llu prefetch_first_admit_us=%llu prefetch_background_us=%llu prefetch_join_admit_us=%llu verified=%s failures=%d\n",blocks,(unsigned long long)page_bytes,(unsigned long long)state_bytes,(unsigned long long)file_bytes,(unsigned long long)(publish_ns / 1000u),(unsigned long long)(publish_ns / 1000u / blocks),(unsigned long long)(writer_ns / 1000u),(unsigned long long)(writer_ns / 1000u / blocks),(unsigned long long)(sync_cold_ns / 1000u),(unsigned long long)(sync_warm_ns / 1000u),(unsigned long long)(prefetch_first_ns / 1000u),(unsigned long long)(prefetch_background_ns / 1000u),(unsigned long long)(prefetch_join_ns / 1000u),failures == 0 ? "bitwise" : "no",failures);
	free(scratch);
	free(actual);
	return(failures == 0 ? 0 : 1);
}
