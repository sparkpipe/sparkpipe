#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_kv_snapshot.h"
#include "kv_device_copy_test_hook.h"

#define SNAP_PAGES 8u
#define SNAP_BLOCK_TOKENS 4u
#define SNAP_PAGE_BYTES 32u
#define SNAP_STATE_BYTES 48u

static uint32_t snap_checks,snap_failures;

#define CHECK(condition) do { snap_checks++; if ( !(condition) ) { snap_failures++; fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#condition); } } while (0)

typedef struct SnapFixture
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[SNAP_PAGES];
	uint32_t resident[SNAP_PAGES];
	uint8_t device[SNAP_PAGES * SNAP_PAGE_BYTES];
	SparkKvPageStore stores[2];
	uint8_t page_staging[SNAP_PAGE_BYTES];
	uint8_t state_staging[SNAP_STATE_BYTES];
	char paths[2][64];
	SparkKvPageCache cache;
	SparkKvPageCacheEntry entries[SNAP_PAGES];
	SparkKvPageCacheSequence sequences[4];
	uint32_t heads[SNAP_PAGES];
	uint32_t by_page[SNAP_PAGES];
	SparkKvPageCacheSnapshot snapshot;
	SparkKvPageCacheSnapshotLink links[SNAP_PAGES];
	uint8_t page[SNAP_PAGE_BYTES];
	uint8_t state[SNAP_STATE_BYTES];
	uint32_t pending[4];
	uint32_t copy_calls;
	SparkTestKvDeviceCopy device_copy;
} SnapFixture;

static SparkStatus SnapCountingCopy(void *context,uint32_t direction,uintptr_t device,void *host,uint64_t bytes)
{
	SnapFixture *fixture = context;
	fixture->copy_calls++;
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		memcpy(host,(const void *)device,(size_t)bytes);
	else
		memcpy((void *)device,host,(size_t)bytes);
	return(SPARK_STATUS_OK);
}

static void SnapKey(SparkKvSnapshotKey *key,uint8_t layout,uint8_t identity,uint32_t tokens)
{
	memset(key,0,sizeof(*key));
	memset(key->layout_sha256,layout,sizeof(key->layout_sha256));
	memset(key->identity_sha256,identity,sizeof(key->identity_sha256));
	key->token_count = tokens;
}

static uint64_t SnapFileBytes(const char *path)
{
	struct stat info;
	return(stat(path,&info) == 0 ? (uint64_t)info.st_size : 0u);
}

static void SnapFlip(const char *path,uint64_t offset)
{
	uint8_t byte = 0u;
	int descriptor = open(path,O_RDWR);
	CHECK(descriptor >= 0);
	CHECK(pread(descriptor,&byte,1u,(off_t)offset) == 1);
	byte ^= 0x5au;
	CHECK(pwrite(descriptor,&byte,1u,(off_t)offset) == 1);
	CHECK(close(descriptor) == 0);
}

static void SnapRemoveTree(const char *directory)
{
	char path[1200];
	struct dirent *item;
	DIR *listing = opendir(directory);
	if ( listing == 0 )
		return;
	while ( (item = readdir(listing)) != 0 )
	{
		if ( strcmp(item->d_name,".") == 0 || strcmp(item->d_name,"..") == 0 )
			continue;
		(void)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name);
		(void)unlink(path);
	}
	(void)closedir(listing);
	(void)rmdir(directory);
}

static void SnapTestStoreFormat(const char *directory)
{
	SparkKvSnapshotStore store,small;
	SparkKvSnapshotSegment segments[3],readback[3];
	SparkKvSnapshotFileHeader header;
	SparkKvSnapshotKey key,other;
	uint8_t chain[100],payload[5000],state[7],out_chain[100],out_payload[5000],out_state[7];
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES],stale[1200];
	struct stat info;
	uint64_t bytes,used;
	uint32_t index;
	int descriptor;
	CHECK(SparkKvSnapshotStoreOpen(&store,"relative/dir",1u << 20u,1u << 20u) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,0u,1u << 20u) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,"/nonexistent-sparkpipe-snapshot-dir",1u << 20u,1u << 20u) == SPARK_STATUS_NOT_FOUND);
	(void)snprintf(stale,sizeof(stale),"%s/%sdead-1",directory,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX);
	descriptor = open(stale,O_WRONLY | O_CREAT | O_EXCL,0600);
	CHECK(descriptor >= 0 && close(descriptor) == 0);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(store.removed_temporary_count == 1u && access(stale,F_OK) != 0 && store.used_bytes == 0u);
	for (index=0u; index<sizeof(chain); index++)
		chain[index] = (uint8_t)(index * 3u + 1u);
	for (index=0u; index<sizeof(payload); index++)
		payload[index] = (uint8_t)(index * 7u + 5u);
	memcpy(state,"kdastat",sizeof(state));
	segments[0] = (SparkKvSnapshotSegment){.kind=1u,.bytes=sizeof(chain),.data=chain};
	segments[1] = (SparkKvSnapshotSegment){.kind=2u,.bytes=sizeof(payload),.data=payload};
	segments[2] = (SparkKvSnapshotSegment){.kind=3u,.bytes=sizeof(state),.data=state};
	SnapKey(&key,0x11u,0x22u,9u);
	CHECK(SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_NOT_FOUND && store.miss_count == 1u);
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotPath(&store,&key,path,sizeof(path)) == SPARK_STATUS_OK);
	CHECK(stat(path,&info) == 0 && (info.st_mode & 0777) == 0600 && (uint64_t)info.st_size % SPARK_KV_SNAPSHOT_ALIGNMENT == 0u);
	CHECK(store.used_bytes == (uint64_t)info.st_size && store.file_count == 1u && store.write_count == 1u);
	CHECK(SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_OK && header.segment_count == 3u && header.segments[1].bytes == sizeof(payload));
	readback[0] = (SparkKvSnapshotSegment){.kind=1u,.bytes=sizeof(out_chain),.data=out_chain};
	readback[1] = (SparkKvSnapshotSegment){.kind=2u,.bytes=sizeof(out_payload),.data=out_payload};
	readback[2] = (SparkKvSnapshotSegment){.kind=3u,.bytes=sizeof(out_state),.data=out_state};
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_OK);
	CHECK(memcmp(out_chain,chain,sizeof(chain)) == 0 && memcmp(out_payload,payload,sizeof(payload)) == 0 && memcmp(out_state,state,sizeof(state)) == 0);
	readback[1].bytes = sizeof(out_payload) - 1u;
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	readback[1].bytes = sizeof(out_payload);
	readback[2].kind = 4u;
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_VALIDATION_FAILED);
	readback[2].kind = 3u;
	CHECK(SparkKvSnapshotReadSegment(&store,&key,2u,3u,out_state,sizeof(out_state),&bytes) == SPARK_STATUS_OK && bytes == sizeof(state));
	CHECK(SparkKvSnapshotReadSegment(&store,&key,1u,3u,out_payload,sizeof(out_payload),&bytes) == SPARK_STATUS_VALIDATION_FAILED);
	used = store.used_bytes;
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK && store.duplicate_count == 1u && store.used_bytes == used && store.write_count == 1u);
	SnapKey(&other,0x12u,0x22u,9u);
	CHECK(SparkKvSnapshotRead(&store,&other,readback,3u) == SPARK_STATUS_NOT_FOUND);
	SnapKey(&other,0x11u,0x22u,8u);
	CHECK(SparkKvSnapshotReadSegment(&store,&other,0u,1u,out_chain,sizeof(out_chain),&bytes) == SPARK_STATUS_NOT_FOUND);
	SnapFlip(path,header.segments[1].offset + 17u);
	CHECK(SparkKvSnapshotReadSegment(&store,&key,0u,1u,out_chain,sizeof(out_chain),&bytes) == SPARK_STATUS_OK && memcmp(out_chain,chain,sizeof(chain)) == 0);
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_HASH_MISMATCH);
	CHECK(store.checksum_failure_count == 1u && access(path,F_OK) != 0 && store.used_bytes == 0u && store.file_count == 0u);
	CHECK(SparkKvSnapshotReadSegment(&store,&key,0u,1u,out_chain,sizeof(out_chain),&bytes) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK && store.write_count == 2u && store.used_bytes == used && store.file_count == 1u);
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_OK && memcmp(out_payload,payload,sizeof(payload)) == 0);
	SnapFlip(path,header.segments[2].offset);
	CHECK(SparkKvSnapshotReadSegment(&store,&key,2u,3u,out_state,sizeof(out_state),&bytes) == SPARK_STATUS_HASH_MISMATCH && store.checksum_failure_count == 2u && access(path,F_OK) != 0);
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK);
	SnapFlip(path,40u);
	CHECK(SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_HASH_MISMATCH && store.checksum_failure_count == 3u && access(path,F_OK) != 0);
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK && SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_OK);
	CHECK(truncate(path,(off_t)(SnapFileBytes(path) - SPARK_KV_SNAPSHOT_ALIGNMENT)) == 0);
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_HASH_MISMATCH && store.checksum_failure_count == 4u);
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK && SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,1u << 20u) == SPARK_STATUS_OK && store.used_bytes == used && store.file_count == 1u);
	CHECK(SparkKvSnapshotStoreOpen(&small,directory,1u << 20u,1u << 20u) == SPARK_STATUS_BUSY);
	CHECK(SparkKvSnapshotRemove(&store,&key) == SPARK_STATUS_OK && store.used_bytes == 0u && store.file_count == 0u);
	CHECK(SparkKvSnapshotRemove(&store,&key) == SPARK_STATUS_NOT_FOUND && SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&small,directory,used + SPARK_KV_SNAPSHOT_ALIGNMENT,1u << 20u) == SPARK_STATUS_OK && small.used_bytes == used && small.file_count == 1u);
	SnapKey(&other,0x11u,0x33u,9u);
	CHECK(SparkKvSnapshotWrite(&small,&other,segments,3u) == SPARK_STATUS_OK && small.eviction_count == 1u && small.evicted_bytes == used && small.used_bytes == used && small.file_count == 1u);
	CHECK(SparkKvSnapshotStat(&small,&key,&header) == SPARK_STATUS_NOT_FOUND && access(path,F_OK) != 0 && SparkKvSnapshotStat(&small,&other,&header) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&small);
	CHECK(SparkKvSnapshotStoreOpen(&small,directory,used - 1u,1u << 20u) == SPARK_STATUS_OK && small.file_count == 0u && small.eviction_count == 1u);
	CHECK(SparkKvSnapshotWrite(&small,&other,segments,3u) == SPARK_STATUS_CAPACITY_EXCEEDED && small.write_failure_count == 0u && small.eviction_count == 1u && small.failed_status == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&small);
}

static void SnapTestEvictionAndQueue(const char *directory)
{
	SparkKvSnapshotStore store,sample;
	SparkKvSnapshotWriteTicket ticket,second;
	SparkKvSnapshotFileHeader header;
	SparkKvSnapshotSegment readback[2];
	SparkKvSnapshotKey keys[4],stale;
	uint32_t kinds[2] = {1u,2u},index;
	uint64_t bytes[2] = {100u,5000u},file_bytes = SPARK_KV_SNAPSHOT_ALIGNMENT * 4u;
	uint8_t out_chain[100],out_payload[5000];
	for (index=0u; index<4u; index++)
		SnapKey(&keys[index],0x21u,(uint8_t)(0x40u + index),4u * (index + 1u));
	SnapKey(&stale,0x22u,0x40u,4u);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,3u * file_bytes,file_bytes) == SPARK_STATUS_OK);
	for (index=0u; index<3u; index++)
	{
		CHECK(SparkKvSnapshotWriteBegin(&store,index == 2u ? &stale : &keys[index],kinds,bytes,2u,&ticket) == SPARK_STATUS_OK && ticket.file_bytes == file_bytes);
		CHECK(SparkKvSnapshotWriteBegin(&store,&keys[3],kinds,bytes,2u,&second) == SPARK_STATUS_BUSY);
		memset(ticket.segments[0].data,(int)(index + 1u),100u);
		memset(ticket.segments[1].data,(int)(index + 7u),5000u);
		CHECK(SparkKvSnapshotWriteCommit(&store,&ticket) == SPARK_STATUS_OK && ticket.job == 0);
		CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	}
	CHECK(SparkKvSnapshotStoreSample(&store,&sample) == SPARK_STATUS_OK && sample.write_count == 3u && sample.file_count == 3u && sample.used_bytes == 3u * file_bytes && sample.queue_full_count == 3u && sample.queued_bytes == 0u && sample.runtime == 0);
	CHECK(SparkKvSnapshotWriteBegin(&store,&keys[0],kinds,bytes,2u,&ticket) == SPARK_STATUS_DUPLICATE && ticket.job == 0 && store.duplicate_count == 1u);
	readback[0] = (SparkKvSnapshotSegment){.kind=1u,.bytes=sizeof(out_chain),.data=out_chain};
	readback[1] = (SparkKvSnapshotSegment){.kind=2u,.bytes=sizeof(out_payload),.data=out_payload};
	CHECK(SparkKvSnapshotRead(&store,&keys[1],readback,2u) == SPARK_STATUS_OK && out_chain[0] == 2u && out_payload[4999] == 8u);
	CHECK(SparkKvSnapshotRead(&store,&keys[0],readback,2u) == SPARK_STATUS_OK && out_chain[0] == 1u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&keys[3],kinds,bytes,2u,&ticket) == SPARK_STATUS_OK);
	memset(ticket.segments[0].data,9,100u);
	memset(ticket.segments[1].data,9,5000u);
	CHECK(SparkKvSnapshotWriteCommit(&store,&ticket) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(store.eviction_count == 1u && store.file_count == 3u && store.used_bytes == 3u * file_bytes);
	CHECK(SparkKvSnapshotStat(&store,&stale,&header) == SPARK_STATUS_NOT_FOUND && SparkKvSnapshotStat(&store,&keys[1],&header) == SPARK_STATUS_OK && SparkKvSnapshotStat(&store,&keys[3],&header) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotWriteBegin(&store,&stale,kinds,bytes,2u,&ticket) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotWriteBegin(&store,&stale,kinds,bytes,2u,&second) == SPARK_STATUS_DUPLICATE);
	SparkKvSnapshotWriteCancel(&store,&ticket);
	CHECK(ticket.job == 0 && store.queued_bytes == 0u && store.queued_count == 0u && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 4u);
	CHECK(SparkKvSnapshotWrite(&store,&stale,readback,2u) == SPARK_STATUS_OK && store.eviction_count == 2u && SparkKvSnapshotStat(&store,&keys[1],&header) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotPrune(&store,keys[0].layout_sha256) == SPARK_STATUS_OK && store.pruned_count == 1u && SparkKvSnapshotStat(&store,&stale,&header) == SPARK_STATUS_NOT_FOUND && store.file_count == 2u);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,3u * file_bytes,file_bytes) == SPARK_STATUS_OK && store.file_count == 2u && store.used_bytes == 2u * file_bytes);
	CHECK(SparkKvSnapshotStat(&store,&keys[0],&header) == SPARK_STATUS_OK && SparkKvSnapshotStat(&store,&keys[3],&header) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,file_bytes,file_bytes) == SPARK_STATUS_OK && store.file_count == 1u && store.eviction_count == 1u);
	SparkKvSnapshotStoreClose(&store);
}

static int32_t SnapStoreOpen(SparkKvPageStore *store,char *path,void *staging,uint64_t page_bytes)
{
	SparkKvPageStoreConfiguration configuration = {0};
	int descriptor;
	strcpy(path,"/tmp/sparkpipe-kv-snapshot-store-XXXXXX");
	descriptor = mkstemp(path);
	if ( descriptor < 0 || close(descriptor) != 0 || unlink(path) != 0 )
		return(-1);
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = SNAP_PAGES;
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = configuration.staging_bytes = page_bytes;
	configuration.maximum_backing_bytes = page_bytes * SNAP_PAGES;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	return(SparkKvPageStoreInitialize(store,&configuration) == SPARK_STATUS_OK ? 0 : -2);
}

static int32_t SnapFixtureOpen(SnapFixture *fixture,SparkKvSnapshotStore *store,uint8_t layout)
{
	SparkKvCacheConfiguration arena = {0};
	SparkKvPageCacheConfiguration config = {0};
	memset(fixture,0,sizeof(*fixture));
	if ( SnapStoreOpen(&fixture->stores[0],fixture->paths[0],fixture->page_staging,SNAP_PAGE_BYTES) != 0 || SnapStoreOpen(&fixture->stores[1],fixture->paths[1],fixture->state_staging,SNAP_STATE_BYTES) != 0 )
		return(-1);
	fixture->stores[0].copy_function = SnapCountingCopy;
	fixture->stores[0].copy_context = fixture;
	arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	arena.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	arena.logical_block_count = arena.resident_block_capacity = SNAP_PAGES;
	arena.block_token_count = SNAP_BLOCK_TOKENS;
	arena.layer_count = arena.kv_head_count = arena.bytes_per_scalar = 1u;
	arena.head_dim = SNAP_PAGE_BYTES;
	arena.key_block_stride_bytes = SNAP_PAGE_BYTES;
	arena.key_device_base = fixture->device;
	arena.blocks = fixture->blocks;
	arena.resident_slot_logical_block_indices = fixture->resident;
	arena.evict_function = SparkKvPageStoreWriteback;
	arena.evict_context = &fixture->stores[0];
	if ( SparkKvCacheArenaInitialize(&fixture->arena,&arena) != SPARK_STATUS_OK )
		return(-2);
	config.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	config.sequence_capacity = 4u;
	config.entry_capacity = config.hash_bucket_count = SNAP_PAGES;
	config.kv_cache_arena = &fixture->arena;
	config.page_store = &fixture->stores[0];
	config.entries = fixture->entries;
	config.sequences = fixture->sequences;
	config.hash_bucket_heads = fixture->heads;
	config.entry_indices_by_logical_page = fixture->by_page;
	if ( SparkKvPageCacheInitialize(&fixture->cache,&config) != SPARK_STATUS_OK || SparkKvPageCacheAttachStateStore(&fixture->cache,&fixture->stores[1]) != SPARK_STATUS_OK )
		return(-3);
	SparkTestKvAttachDeviceCopy(&fixture->cache,&fixture->device_copy,&fixture->arena);
	fixture->snapshot.store = store;
	fixture->snapshot.pending_terminals = fixture->pending;
	fixture->snapshot.pending_capacity = 4u;
	memset(fixture->snapshot.layout_sha256,layout,sizeof(fixture->snapshot.layout_sha256));
	fixture->snapshot.page_capacity = SNAP_PAGES;
	fixture->snapshot.links = fixture->links;
	fixture->snapshot.page = fixture->page;
	fixture->snapshot.state = fixture->state;
	return(SparkKvPageCacheAttachSnapshot(&fixture->cache,&fixture->snapshot) == SPARK_STATUS_OK ? 0 : -4);
}

static void SnapFixtureClose(SnapFixture *fixture)
{
	SparkKvPageStoreDestroy(&fixture->stores[0]);
	SparkKvPageStoreDestroy(&fixture->stores[1]);
}

static void SnapIdentity(SparkModelDriverCacheIdentity *identity,uint8_t seed)
{
	uint32_t index;
	for (index=0u; index<sizeof(identity->sha256); index++)
		identity->sha256[index] = (uint8_t)(seed + index);
}

static void SnapLane(SparkModelDriverCacheLane *lane,uint64_t sequence,uint32_t slot,uint32_t position,uint32_t context)
{
	memset(lane,0,sizeof(*lane));
	lane->sequence_id = sequence;
	lane->request_generation = 1u;
	lane->step_generation = position + 1u;
	lane->resident_sequence_slot = slot;
	lane->sequence_position = position;
	lane->context_token_count = context;
}

static SparkStatus SnapWriteState(SparkKvPageStore *store,uint32_t page,uint64_t generation,const uint8_t *source)
{
	SparkStatus status = SparkKvPageStoreWriteback(store,page,0u,generation,(uintptr_t)source,SNAP_STATE_BYTES,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(store);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreWriteback(store,page,0u,generation,(uintptr_t)source,SNAP_STATE_BYTES,0u,0u);
	}
	return(status);
}

static SparkStatus SnapReadState(SparkKvPageStore *store,uint32_t page,uint64_t generation,uint8_t *output)
{
	SparkStatus status = SparkKvPageStoreReadback(store,page,generation,(uintptr_t)output,SNAP_STATE_BYTES);
	while ( status == SPARK_STATUS_BUSY )
	{
		status = SparkKvPageStoreWaitForTransfers(store);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageStoreReadback(store,page,generation,(uintptr_t)output,SNAP_STATE_BYTES);
	}
	return(status);
}

static void SnapPrefill(SnapFixture *fixture,uint8_t pages[3][SNAP_PAGE_BYTES],uint8_t states[3][SNAP_STATE_BYTES])
{
	static const uint32_t ends[3] = {4u,8u,9u};
	SparkModelDriverCacheLane lane;
	SparkKvCacheBlockView view;
	uint32_t step,page,position = 0u,byte;
	for (step=0u; step<3u; step++)
	{
		SnapLane(&lane,1u,0u,position,ends[step]);
		lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
		lane.publish_token_count = ends[step];
		SnapIdentity(&lane.publish_identity,(uint8_t)(10u * (step + 1u)));
		CHECK(SparkKvPageCacheBeginLane(&fixture->cache,&lane,&page) == SPARK_STATUS_OK);
		CHECK(SparkKvCacheArenaResolveBlock(&fixture->arena,page,&view) == SPARK_STATUS_OK);
		for (byte=0u; byte<SNAP_PAGE_BYTES; byte++)
			pages[step][byte] = (uint8_t)(step * 50u + byte + 1u);
		for (byte=0u; byte<SNAP_STATE_BYTES; byte++)
			states[step][byte] = (uint8_t)(step * 70u + byte * 3u + 2u);
		memcpy((void *)view.key_device_address,pages[step],SNAP_PAGE_BYTES);
		CHECK(SnapWriteState(&fixture->stores[1],page,view.generation,states[step]) == SPARK_STATUS_OK);
		CHECK(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
		position = ends[step];
	}
}

static uint32_t SnapEntryCount(const SnapFixture *fixture)
{
	uint32_t index,count = 0u;
	for (index=0u; index<SNAP_PAGES; index++)
		count += (fixture->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u ? 1u : 0u;
	return(count);
}

static uint32_t SnapEvictAll(SnapFixture *fixture)
{
	uint32_t evicted = 0u;
	while ( SparkKvPageCacheEvictUnused(&fixture->cache) == SPARK_STATUS_OK )
		evicted++;
	return(evicted);
}

static uint32_t SnapAllocatedPages(const SnapFixture *fixture)
{
	uint32_t index,count = 0u;
	for (index=0u; index<SNAP_PAGES; index++)
		count += (fixture->blocks[index].flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u ? 1u : 0u;
	return(count);
}

static void SnapCheckRestoredChain(SnapFixture *fixture,uint8_t pages[3][SNAP_PAGE_BYTES],uint8_t states[3][SNAP_STATE_BYTES])
{
	SparkModelDriverCacheLane lane;
	SparkKvCacheBlockView view;
	uint32_t logical[4],physical[4],count,flags,index,entry;
	uint8_t state[SNAP_STATE_BYTES];
	SnapLane(&lane,7u,1u,9u,10u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane.prefix_token_count = 9u;
	SnapIdentity(&lane.prefix_identity,30u);
	CHECK(SparkKvPageCacheBeginPinnedLaneTransaction(&fixture->cache,&lane,logical,physical,4u,&count,&flags) == SPARK_STATUS_OK && count == 3u);
	SparkTestKvDeviceCopySettle(&fixture->device_copy);
	for (index=0u; index<3u && count == 3u; index++)
	{
		CHECK(SparkKvCacheArenaResolveBlock(&fixture->arena,logical[index],&view) == SPARK_STATUS_OK);
		CHECK(memcmp((const void *)view.key_device_address,pages[index],SNAP_PAGE_BYTES) == 0);
	}
	entry = fixture->by_page[logical[1]];
	CHECK(entry < SNAP_PAGES && (fixture->entries[entry].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u);
	for (entry=0u; entry<SNAP_PAGES && fixture->entries[entry].token_count != 9u; entry++)
		;
	CHECK(entry < SNAP_PAGES && (fixture->entries[entry].flags & (SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS | SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED)) == SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED);
	if ( entry < SNAP_PAGES )
		CHECK(SnapReadState(&fixture->stores[1],fixture->entries[entry].logical_page_index,fixture->blocks[fixture->entries[entry].logical_page_index].generation,state) == SPARK_STATUS_OK && memcmp(state,states[2],SNAP_STATE_BYTES) == 0);
	CHECK(SparkKvCacheArenaUnpinResidentTable(&fixture->arena,logical,count) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheRollbackLaneTransaction(&fixture->cache,&lane,flags) == SPARK_STATUS_OK);
	SnapLane(&lane,8u,2u,8u,9u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane.prefix_token_count = 8u;
	SnapIdentity(&lane.prefix_identity,20u);
	CHECK(SparkKvPageCacheBeginLane(&fixture->cache,&lane,&index) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvPageCacheRestorePrefix(&fixture->cache,&lane.prefix_identity,8u) == SPARK_STATUS_OK);
	CHECK(SnapReadState(&fixture->stores[1],logical[1],fixture->blocks[logical[1]].generation,state) == SPARK_STATUS_OK && memcmp(state,states[1],SNAP_STATE_BYTES) == 0);
	CHECK((fixture->entries[fixture->by_page[logical[1]]].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) == 0u);
	CHECK(SparkKvPageCacheBeginPinnedLaneTransaction(&fixture->cache,&lane,logical,physical,4u,&count,&flags) == SPARK_STATUS_OK && count == 3u);
	CHECK(SparkKvCacheArenaUnpinResidentTable(&fixture->arena,logical,count) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheRollbackLaneTransaction(&fixture->cache,&lane,flags) == SPARK_STATUS_OK);
}

static void SnapTestPageCacheRoundTrip(const char *directory)
{
	static SnapFixture source,fresh,other;
	SparkKvSnapshotStore store;
	SparkModelDriverCacheIdentity identity;
	SparkKvSnapshotKey key;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x44u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheRestorePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && store.read_count == 0u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(source.snapshot.save_count == 1u && source.snapshot.save_page_count == 3u && store.file_count == 3u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 3u && store.duplicate_count == 0u);
	SnapFixtureClose(&source);

	CHECK(SnapFixtureOpen(&fresh,&store,0x44u) == 0);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_OK);
	CHECK(fresh.snapshot.restore_count == 1u && fresh.snapshot.restore_page_count == 3u && SnapEntryCount(&fresh) == 3u);
	SnapCheckRestoredChain(&fresh,pages,states);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_OK && fresh.snapshot.restore_count == 2u);
	SnapFixtureClose(&fresh);

	CHECK(SnapFixtureOpen(&other,&store,0x45u) == 0);
	CHECK(SparkKvPageCacheRestorePrefix(&other.cache,&identity,9u) == SPARK_STATUS_NOT_FOUND && other.snapshot.restore_miss_count == 1u && SnapEntryCount(&other) == 0u);
	SnapFixtureClose(&other);

	CHECK(SnapFixtureOpen(&fresh,&store,0x44u) == 0);
	memset(key.layout_sha256,0x44u,sizeof(key.layout_sha256));
	SnapIdentity((SparkModelDriverCacheIdentity *)key.identity_sha256,20u);
	key.token_count = 8u;
	key.reserved0 = 0u;
	CHECK(SparkKvSnapshotPath(&store,&key,path,sizeof(path)) == SPARK_STATUS_OK);
	SnapFlip(path,SPARK_KV_SNAPSHOT_ALIGNMENT * 2u + 3u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_NOT_FOUND);
	CHECK(SnapEntryCount(&fresh) == 1u && SnapAllocatedPages(&fresh) == 1u && fresh.cache.lru_head != SPARK_KV_PAGE_CACHE_NO_INDEX && fresh.entries[fresh.cache.lru_head].token_count == 4u);
	CHECK(fresh.snapshot.restore_failure_count == 0u && fresh.snapshot.restore_corrupt_count == 1u && fresh.snapshot.restore_miss_count == 1u && fresh.snapshot.restore_page_count == 1u);
	CHECK(access(path,F_OK) != 0 && store.checksum_failure_count == 1u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_NOT_FOUND);
	CHECK(SnapEntryCount(&fresh) == 1u && SnapAllocatedPages(&fresh) == 1u && fresh.cache.entries[fresh.cache.lru_head].reference_count == 0u);
	SnapFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestSaveDefersWhenQueueIsFull(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvSnapshotWriteTicket blocker;
	SparkKvSnapshotKey blocker_key;
	SparkModelDriverCacheIdentity identity;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t kind = 1u;
	uint64_t bytes = 40000u;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,SPARK_KV_SNAPSHOT_ALIGNMENT * 12u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x77u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	SnapKey(&blocker_key,0x78u,0x01u,1u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&blocker_key,&kind,&bytes,1u,&blocker) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_BUSY);
	CHECK(source.snapshot.save_page_count == 0u && store.queue_full_count == 1u && store.write_count == 0u);
	SparkKvSnapshotWriteCancel(&store,&blocker);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(store.write_count == 3u && source.snapshot.save_page_count == 3u && store.file_count == 3u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestReleaseSavesAndPrepareRestores(const char *directory)
{
	static SnapFixture source,fresh;
	SparkKvSnapshotStore store;
	SparkKvLaneTransaction owners[4];
	SparkKvLaneTransactions transactions;
	SparkModelDriverCacheLane lane;
	SparkModelDriverAdmissionRequest request;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x66u) == 0);
	SnapPrefill(&source,pages,states);
	memset(owners,0,sizeof(owners));
	memset(&transactions,0,sizeof(transactions));
	transactions.cache = &source.cache;
	transactions.lanes = owners;
	transactions.logical_pages = logical;
	transactions.physical_pages = physical;
	transactions.page_capacity = 4u;
	SnapLane(&lane,1u,0u,9u,9u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.request_id = request.submission_id = request.transaction_id = 1u;
	request.control_generation = request.request_generation = request.step_generation = 1u;
	request.cache_lane_count = 1u;
	request.cache_lanes = &lane;
	request.active_slot_count = 1u;
	request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	source.copy_calls = 0u;
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheSavePending(&source.cache) == 1u && source.copy_calls == 0u && source.snapshot.save_count == 0u && source.cache.live_sequence_count == 0u);
	CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_CAPACITY_EXCEEDED && SnapEntryCount(&source) == 3u);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && SparkKvPageCacheSavePending(&source.cache) == 0u && source.copy_calls != 0u);
	CHECK(source.snapshot.save_count == 1u && source.snapshot.save_page_count == 3u);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 3u);
	CHECK(SnapEvictAll(&source) == 3u && SnapEntryCount(&source) == 0u);
	SnapFixtureClose(&source);

	CHECK(SnapFixtureOpen(&fresh,&store,0x66u) == 0);
	memset(owners,0,sizeof(owners));
	memset(&transactions,0,sizeof(transactions));
	transactions.cache = &fresh.cache;
	transactions.lanes = owners;
	transactions.logical_pages = logical;
	transactions.physical_pages = physical;
	transactions.page_capacity = 4u;
	SnapLane(&lane,5u,1u,9u,10u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane.prefix_token_count = 9u;
	SnapIdentity(&lane.prefix_identity,30u);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	request.frame_flags = 0u;
	request.active_slot_count = request.new_token_count = 1u;
	request.submission_id = request.transaction_id = request.request_id = 2u;
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK);
	CHECK(fresh.snapshot.restore_count == 1u && owners[1].phase == SPARK_KV_LANE_TRANSACTION_PREPARED && owners[1].page_count == 3u);
	CHECK(memcmp(fresh.device + (uint64_t)fresh.blocks[logical[4]].resident_slot_index * SNAP_PAGE_BYTES,pages[0],SNAP_PAGE_BYTES) == 0);
	CHECK(fresh.device_copy.copies == 1u);
	SparkTestKvDeviceCopySettle(&fresh.device_copy);
	CHECK(memcmp(fresh.device + (uint64_t)fresh.blocks[logical[6]].resident_slot_index * SNAP_PAGE_BYTES,pages[2],SNAP_PAGE_BYTES) == 0);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK && owners[1].phase == SPARK_KV_LANE_TRANSACTION_EMPTY);
	SparkTestKvDeviceCopySettle(&fresh.device_copy);
	CHECK(fresh.device_copy.retired == 1u);
	SnapFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTransactionsInit(SparkKvLaneTransactions *transactions,SparkKvLaneTransaction *owners,SparkKvPageCache *cache,uint32_t *logical,uint32_t *physical)
{
	memset(owners,0,4u * sizeof(*owners));
	memset(transactions,0,sizeof(*transactions));
	transactions->cache = cache;
	transactions->lanes = owners;
	transactions->logical_pages = logical;
	transactions->physical_pages = physical;
	transactions->page_capacity = 4u;
}

static void SnapReleaseRequest(SparkModelDriverAdmissionRequest *request,SparkModelDriverCacheLane *lanes,uint32_t count,uint64_t id)
{
	memset(request,0,sizeof(*request));
	request->descriptor_bytes = sizeof(*request);
	request->program_id = 1u;
	request->request_id = request->submission_id = request->transaction_id = id;
	request->control_generation = request->request_generation = request->step_generation = 1u;
	request->cache_lane_count = count;
	request->cache_lanes = lanes;
	request->active_slot_count = count;
	request->frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
}

static void SnapPublishSecond(SnapFixture *fixture)
{
	SparkModelDriverCacheLane lane;
	SparkKvCacheBlockView view;
	uint8_t state[SNAP_STATE_BYTES];
	uint32_t page;
	memset(state,0x5d,sizeof(state));
	SnapLane(&lane,2u,1u,0u,4u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	lane.publish_token_count = 4u;
	SnapIdentity(&lane.publish_identity,90u);
	CHECK(SparkKvPageCacheBeginLane(&fixture->cache,&lane,&page) == SPARK_STATUS_OK);
	CHECK(SparkKvCacheArenaResolveBlock(&fixture->arena,page,&view) == SPARK_STATUS_OK);
	memset((void *)view.key_device_address,0x3c,SNAP_PAGE_BYTES);
	CHECK(SnapWriteState(&fixture->stores[1],page,view.generation,state) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
}

static void SnapPublishOne(SnapFixture *fixture,uint64_t sequence,uint32_t slot,uint8_t identity_seed,uint8_t fill)
{
	SparkModelDriverCacheLane lane;
	SparkKvCacheBlockView view;
	uint8_t state[SNAP_STATE_BYTES];
	uint32_t page;
	memset(state,fill,sizeof(state));
	SnapLane(&lane,sequence,slot,0u,4u);
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	lane.publish_token_count = 4u;
	SnapIdentity(&lane.publish_identity,identity_seed);
	CHECK(SparkKvPageCacheBeginLane(&fixture->cache,&lane,&page) == SPARK_STATUS_OK);
	CHECK(SparkKvCacheArenaResolveBlock(&fixture->arena,page,&view) == SPARK_STATUS_OK);
	memset((void *)view.key_device_address,fill,SNAP_PAGE_BYTES);
	CHECK(SnapWriteState(&fixture->stores[1],page,view.generation,state) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheCompleteLane(&fixture->cache,&lane) == SPARK_STATUS_OK);
}

static uint32_t SnapIdentityValid(SnapFixture *fixture,uint8_t identity_seed)
{
	SparkModelDriverCacheIdentity identity;
	uint32_t index;
	SnapIdentity(&identity,identity_seed);
	for (index=0u; index<SNAP_PAGES; index++)
		if ( (fixture->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && memcmp(&fixture->entries[index].identity,&identity,sizeof(identity)) == 0 )
			return(1u);
	return(0u);
}

static void SnapReleaseThree(SnapFixture *fixture,SparkKvLaneTransactions *transactions,SparkKvLaneTransaction *owners,uint32_t *logical,uint32_t *physical)
{
	SparkModelDriverCacheLane lanes[3];
	SparkModelDriverAdmissionRequest request;
	SnapTransactionsInit(transactions,owners,&fixture->cache,logical,physical);
	SnapLane(&lanes[0],1u,0u,9u,9u);
	SnapLane(&lanes[1],2u,1u,4u,4u);
	SnapLane(&lanes[2],3u,2u,4u,4u);
	lanes[0].flags = lanes[1].flags = lanes[2].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	SnapReleaseRequest(&request,lanes,3u,1u);
	CHECK(SparkKvLaneTransactionsAdmit(transactions,&request) == SPARK_STATUS_OK);
	CHECK(fixture->cache.live_sequence_count == 0u && fixture->snapshot.save_skipped_count == 2u);
}

static void SnapTestEvictionDemotes(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvLaneTransaction owners[4];
	SparkKvLaneTransactions transactions;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16],evicted;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x68u) == 0);
	source.snapshot.pending_capacity = 1u;
	SnapPrefill(&source,pages,states);
	SnapPublishOne(&source,2u,1u,90u,0x3cu);
	SnapPublishOne(&source,3u,2u,91u,0x4du);
	SnapReleaseThree(&source,&transactions,owners,logical,physical);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && source.snapshot.save_count == 1u && SparkKvPageCacheSavePending(&source.cache) == 0u);
	source.snapshot.pending_capacity = 0u;
	for (evicted=0u; evicted<3u; evicted++)
		CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_OK);
	CHECK(SnapEntryCount(&source) == 2u && SnapIdentityValid(&source,90u) != 0u && SnapIdentityValid(&source,91u) != 0u);
	CHECK(source.snapshot.evicted_unsaved_count == 0u && source.snapshot.demote_queued_count == 0u);
	CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_OK && SnapEntryCount(&source) == 1u && source.snapshot.evicted_unsaved_count == 1u);
	CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_OK && SnapEntryCount(&source) == 0u && source.snapshot.evicted_unsaved_count == 2u);
	source.snapshot.pending_capacity = 4u;
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestEvictionQueuesDemotion(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvLaneTransaction owners[4];
	SparkKvLaneTransactions transactions;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x69u) == 0);
	source.snapshot.pending_capacity = 1u;
	SnapPrefill(&source,pages,states);
	SnapPublishOne(&source,2u,1u,90u,0x3cu);
	SnapPublishOne(&source,3u,2u,91u,0x4du);
	SnapReleaseThree(&source,&transactions,owners,logical,physical);
	source.snapshot.pending_capacity = 4u;
	CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_OK && SnapEntryCount(&source) == 4u);
	CHECK(source.snapshot.evicted_unsaved_count == 1u && source.snapshot.demote_queued_count == 1u && SparkKvPageCacheSavePending(&source.cache) == 2u);
	CHECK(SnapIdentityValid(&source,90u) + SnapIdentityValid(&source,91u) == 1u);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && source.snapshot.save_count == 2u);
	CHECK(SnapEvictAll(&source) == 4u && source.snapshot.evicted_unsaved_count == 1u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestEvictionPrefersLowPriority(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x6au) == 0);
	source.cache.admission_priority = 7u;
	SnapPublishOne(&source,2u,1u,90u,0x3cu);
	source.cache.admission_priority = 1u;
	SnapPublishOne(&source,3u,2u,91u,0x4du);
	source.cache.admission_priority = 0u;
	CHECK(SparkKvPageCacheReleaseLane(&source.cache,1u,2u) == SPARK_STATUS_OK && SparkKvPageCacheReleaseLane(&source.cache,2u,3u) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_OK && SnapIdentityValid(&source,90u) != 0u && SnapIdentityValid(&source,91u) == 0u);
	CHECK(source.snapshot.evicted_unsaved_count == 1u && source.snapshot.demote_queued_count == 1u);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && source.snapshot.save_count == 1u);
	CHECK(SparkKvPageCacheEvictUnused(&source.cache) == SPARK_STATUS_OK && SnapEntryCount(&source) == 0u && source.snapshot.evicted_unsaved_count == 1u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestQueueFullSkips(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvLaneTransaction owners[4];
	SparkKvLaneTransactions transactions;
	SparkModelDriverCacheLane lanes[2];
	SparkModelDriverAdmissionRequest request;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x67u) == 0);
	source.snapshot.pending_capacity = 1u;
	SnapPrefill(&source,pages,states);
	SnapPublishSecond(&source);
	SnapTransactionsInit(&transactions,owners,&source.cache,logical,physical);
	SnapLane(&lanes[0],1u,0u,9u,9u);
	SnapLane(&lanes[1],2u,1u,4u,4u);
	lanes[0].flags = lanes[1].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	SnapReleaseRequest(&request,lanes,2u,1u);
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK);
	CHECK(source.cache.live_sequence_count == 0u && SparkKvPageCacheSavePending(&source.cache) == 1u);
	CHECK(source.snapshot.save_skipped_count == 1u && source.snapshot.save_mark_count == 1u && source.snapshot.full_logged == 1u);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && source.snapshot.save_count == 1u);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 3u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestResetCancelsPending(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvLaneTransaction owners[4];
	SparkKvLaneTransactions transactions;
	SparkModelDriverCacheLane lanes[2];
	SparkModelDriverAdmissionRequest request;
	SparkKvPageCacheSaveWork work;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x68u) == 0);
	SnapPrefill(&source,pages,states);
	SnapPublishSecond(&source);
	SnapTransactionsInit(&transactions,owners,&source.cache,logical,physical);
	SnapLane(&lanes[0],1u,0u,9u,9u);
	lanes[0].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	SnapReleaseRequest(&request,lanes,1u,1u);
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK && SparkKvPageCacheSavePending(&source.cache) == 1u);
	CHECK(SparkKvLaneTransactionsReset(&transactions) == SPARK_STATUS_OK);
	CHECK(source.snapshot.save_cancelled_count == 1u && SparkKvPageCacheSavePending(&source.cache) == 0u && source.snapshot.save_count == 0u);
	CHECK(SnapEntryCount(&source) == 0u && SnapAllocatedPages(&source) == 0u);
	SnapPrefill(&source,pages,states);
	SnapReleaseRequest(&request,lanes,1u,2u);
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK && SparkKvPageCacheSavePending(&source.cache) == 1u);
	CHECK(SparkKvPageCacheSaveTake(&source.cache,&work) == SPARK_STATUS_OK && source.snapshot.in_flight == 1u);
	CHECK(SparkKvLaneTransactionsReset(&transactions) == SPARK_STATUS_BUSY && source.snapshot.save_cancelled_count == 1u);
	CHECK(SparkKvPageCacheSaveFinish(&source.cache,&work,SparkKvPageCacheSaveCopy(&source.cache,&work)) == SPARK_STATUS_OK);
	CHECK(source.snapshot.in_flight == 0u && SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK);
	CHECK(source.snapshot.in_flight == 0u && SparkKvPageCacheSavePending(&source.cache) == 0u && source.snapshot.save_count == 1u);
	CHECK(SparkKvLaneTransactionsReset(&transactions) == SPARK_STATUS_OK && source.snapshot.save_cancelled_count == 1u);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 3u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapLinkKey(SparkKvSnapshotKey *key,uint8_t layout,uint8_t identity_seed,uint32_t tokens)
{
	memset(key,0,sizeof(*key));
	memset(key->layout_sha256,layout,sizeof(key->layout_sha256));
	SnapIdentity((SparkModelDriverCacheIdentity *)key->identity_sha256,identity_seed);
	key->token_count = tokens;
}

static void SnapWriteFile(SparkKvSnapshotStore *store,uint8_t layout,uint8_t identity_seed,uint32_t tokens,const SparkKvPageCacheSnapshotLink *links,uint32_t link_count,uint64_t page_bytes)
{
	static uint8_t page[2u * SNAP_PAGE_BYTES],state[SNAP_STATE_BYTES];
	SparkKvSnapshotSegment segments[3];
	SparkKvSnapshotKey key;
	SnapLinkKey(&key,layout,identity_seed,tokens);
	segments[0] = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_CHAIN,.bytes=(uint64_t)link_count * sizeof(links[0]),.data=(void *)links};
	segments[1] = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES,.bytes=page_bytes,.data=page};
	segments[2] = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_STATE,.bytes=SNAP_STATE_BYTES,.data=state};
	CHECK(SparkKvSnapshotWrite(store,&key,segments,3u) == SPARK_STATUS_OK);
}

static void SnapUnreadable(SparkKvSnapshotStore *store,uint8_t identity_seed,uint32_t tokens)
{
	SparkKvSnapshotKey key;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SnapLinkKey(&key,0x44u,identity_seed,tokens);
	CHECK(SparkKvSnapshotPath(store,&key,path,sizeof(path)) == SPARK_STATUS_OK && chmod(path,0) == 0);
}

static uint32_t SnapKeyPresent(SparkKvSnapshotStore *store,uint8_t identity_seed,uint32_t tokens)
{
	SparkKvSnapshotKey key;
	SparkKvSnapshotFileHeader header;
	SnapLinkKey(&key,0x44u,identity_seed,tokens);
	return(SparkKvSnapshotStat(store,&key,&header) == SPARK_STATUS_OK ? 1u : 0u);
}

static void SnapTestReadOutcomes(const char *directory)
{
	static SnapFixture source,fresh;
	SparkKvSnapshotStore store,sample;
	SparkModelDriverCacheIdentity identity;
	SparkKvPageCacheSnapshotLink links[2];
	SparkKvSnapshotSegment segments[3];
	SparkKvSnapshotKey key;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES],state[SNAP_STATE_BYTES];
	uint64_t writes,duplicates;
	CHECK(geteuid() != 0);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x44u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.file_count == 3u);
	SnapFixtureClose(&source);
	memset(links,0,sizeof(links));
	links[0].token_count = 3u;
	SnapIdentity(&links[0].identity,90u);
	SnapWriteFile(&store,0x44u,90u,4u,links,1u,SNAP_PAGE_BYTES);
	links[0].token_count = 4u;
	SnapIdentity(&links[0].identity,91u);
	SnapWriteFile(&store,0x44u,91u,4u,links,1u,SNAP_PAGE_BYTES - 16u);
	CHECK(SnapFixtureOpen(&fresh,&store,0x44u) == 0);
	SnapIdentity(&identity,90u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,4u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_corrupt_count == 1u && fresh.snapshot.restore_miss_count == 1u && fresh.snapshot.restore_failure_count == 0u && SnapKeyPresent(&store,90u,4u) == 0u);
	SnapIdentity(&identity,91u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,4u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_corrupt_count == 2u && fresh.snapshot.restore_failure_count == 0u && SnapKeyPresent(&store,91u,4u) == 0u);
	SnapUnreadable(&store,30u,9u);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_read_error_count == 1u && fresh.snapshot.restore_miss_count == 3u && fresh.snapshot.restore_failure_count == 0u && SnapKeyPresent(&store,30u,9u) == 0u);
	CHECK(SparkKvSnapshotStoreSample(&store,&sample) == SPARK_STATUS_OK);
	writes = sample.write_count;
	duplicates = sample.duplicate_count;
	SnapLinkKey(&key,0x44u,30u,9u);
	memset(state,0,sizeof(state));
	segments[0] = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_CHAIN,.bytes=sizeof(links[0]),.data=links};
	segments[1] = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES,.bytes=SNAP_PAGE_BYTES,.data=pages[0]};
	segments[2] = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_STATE,.bytes=SNAP_STATE_BYTES,.data=state};
	CHECK(SparkKvSnapshotWrite(&store,&key,segments,3u) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotStoreSample(&store,&sample) == SPARK_STATUS_OK && sample.write_count == writes + 1u && sample.duplicate_count == duplicates);
	SnapUnreadable(&store,10u,4u);
	SnapIdentity(&identity,20u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,8u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_read_error_count == 2u && fresh.snapshot.restore_failure_count == 0u && SnapEntryCount(&fresh) == 0u && SnapAllocatedPages(&fresh) == 0u);
	SnapFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestMarkAllOrder(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvPageCacheSaveOrder order[SNAP_PAGES];
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t marked,deferred,ineligible,unsaved,index,last;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x46u) == 0);
	SnapPrefill(&source,pages,states);
	SnapPublishSecond(&source);
	source.entries[source.by_page[source.sequences[1].terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX ? source.entries[source.sequences[1].terminal_entry_index].logical_page_index : 0u]].flags |= SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE;
	CHECK(SparkKvPageCacheCountUnsaved(&source.cache,&unsaved,&ineligible) == SPARK_STATUS_OK && unsaved == 3u && ineligible == 1u);
	CHECK(SparkKvPageCacheMarkAllUnsaved(&source.cache,order,SNAP_PAGES,&marked,&deferred,&ineligible) == SPARK_STATUS_OK);
	CHECK(marked == 3u && deferred == 0u && ineligible == 1u && SparkKvPageCacheSavePending(&source.cache) == 3u);
	last = UINT32_MAX;
	for (index=0u; index<3u; index++)
	{
		uint32_t terminal = source.pending[(source.snapshot.pending_head + index) % source.snapshot.pending_capacity];
		CHECK(index == 0u || source.entries[terminal].last_used_epoch <= source.entries[last].last_used_epoch);
		last = terminal;
	}
	CHECK(SparkKvPageCacheMarkAllUnsaved(&source.cache,order,SNAP_PAGES - 1u,&marked,&deferred,&ineligible) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheCountUnsaved(&source.cache,&unsaved,&ineligible) == SPARK_STATUS_OK && unsaved == 0u && ineligible == 1u && store.write_count == 3u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

int main(void)
{
	char directory[] = "/tmp/sparkpipe-kv-snapshot-XXXXXX";
	if ( mkdtemp(directory) == 0 )
		return(2);
	SnapTestStoreFormat(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionAndQueue(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestSaveDefersWhenQueueIsFull(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestPageCacheRoundTrip(directory);
	SnapTestReleaseSavesAndPrepareRestores(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestQueueFullSkips(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionDemotes(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionQueuesDemotion(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionPrefersLowPriority(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestResetCancelsPending(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestReadOutcomes(directory);
	SnapRemoveTree(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestMarkAllOrder(directory);
	SnapRemoveTree(directory);
	printf("test_kv_snapshot: %u checks, %u failures\n",snap_checks,snap_failures);
	return(snap_failures == 0u ? 0 : 1);
}
