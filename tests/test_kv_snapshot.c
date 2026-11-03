#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
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
#define SNAP_RESERVE (UINT64_C(1) << 20)

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

static void SnapRemoveAll(const char *directory)
{
	char path[1200];
	struct dirent *item;
	struct stat info;
	DIR *listing = opendir(directory);
	if ( listing == 0 )
		return;
	while ( (item = readdir(listing)) != 0 )
	{
		if ( strcmp(item->d_name,".") == 0 || strcmp(item->d_name,"..") == 0 )
			continue;
		(void)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name);
		if ( lstat(path,&info) == 0 && S_ISDIR(info.st_mode) )
			SnapRemoveAll(path);
		else
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
	CHECK(SparkKvSnapshotStoreOpen(&store,"relative/dir",1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,0u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,"/nonexistent-sparkpipe-snapshot-dir",1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_NOT_FOUND);
	(void)snprintf(stale,sizeof(stale),"%s/%sdead-1",directory,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX);
	descriptor = open(stale,O_WRONLY | O_CREAT | O_EXCL,0600);
	CHECK(descriptor >= 0 && close(descriptor) == 0);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,0u,SNAP_RESERVE) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotPath(&store,&key,0,path,sizeof(path)) == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK && store.duplicate_count == 1u && store.used_bytes == used && store.write_count == 1u);
	SnapKey(&other,0x12u,0x22u,9u);
	CHECK(SparkKvSnapshotRead(&store,&other,readback,3u) == SPARK_STATUS_NOT_FOUND);
	SnapKey(&other,0x11u,0x22u,8u);
	CHECK(SparkKvSnapshotReadSegment(&store,&other,0u,1u,out_chain,sizeof(out_chain),&bytes) == SPARK_STATUS_NOT_FOUND);
	SnapFlip(path,header.segments[1].offset + 17u);
	CHECK(SparkKvSnapshotReadSegment(&store,&key,0u,1u,out_chain,sizeof(out_chain),&bytes) == SPARK_STATUS_OK && memcmp(out_chain,chain,sizeof(chain)) == 0);
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_HASH_MISMATCH);
	CHECK(store.checksum_failure_count == 1u && access(path,F_OK) != 0 && store.used_bytes == 0u && store.file_count == 0u);
	CHECK(SparkKvSnapshotReadSegment(&store,&key,0u,1u,out_chain,sizeof(out_chain),&bytes) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK && store.write_count == 2u && store.used_bytes == used && store.file_count == 1u);
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_OK && memcmp(out_payload,payload,sizeof(payload)) == 0);
	SnapFlip(path,header.segments[2].offset);
	CHECK(SparkKvSnapshotReadSegment(&store,&key,2u,3u,out_state,sizeof(out_state),&bytes) == SPARK_STATUS_HASH_MISMATCH && store.checksum_failure_count == 2u && access(path,F_OK) != 0);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK);
	SnapFlip(path,40u);
	CHECK(SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_HASH_MISMATCH && store.checksum_failure_count == 3u && access(path,F_OK) != 0);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK && SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_OK);
	CHECK(truncate(path,(off_t)(SnapFileBytes(path) - SPARK_KV_SNAPSHOT_ALIGNMENT)) == 0);
	CHECK(SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_HASH_MISMATCH && store.checksum_failure_count == 4u);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK && SparkKvSnapshotRead(&store,&key,readback,3u) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && store.used_bytes == used && store.file_count == 1u);
	CHECK(SparkKvSnapshotStoreOpen(&small,directory,1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_BUSY);
	CHECK(SparkKvSnapshotRemove(&store,&key) == SPARK_STATUS_OK && store.used_bytes == 0u && store.file_count == 0u);
	CHECK(SparkKvSnapshotRemove(&store,&key) == SPARK_STATUS_NOT_FOUND && SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,segments,3u) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&small,directory,used + SPARK_KV_SNAPSHOT_ALIGNMENT,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && small.used_bytes == used && small.file_count == 1u);
	SnapKey(&other,0x11u,0x33u,9u);
	CHECK(SparkKvSnapshotWrite(&small,&other,0,segments,3u) == SPARK_STATUS_OK && small.eviction_count == 1u && small.evicted_bytes == used && small.used_bytes == used && small.file_count == 1u);
	CHECK(SparkKvSnapshotStat(&small,&key,&header) == SPARK_STATUS_NOT_FOUND && access(path,F_OK) != 0 && SparkKvSnapshotStat(&small,&other,&header) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&small);
	CHECK(SparkKvSnapshotStoreOpen(&small,directory,used - 1u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && small.file_count == 0u && small.eviction_count == 1u);
	CHECK(SparkKvSnapshotWrite(&small,&other,0,segments,3u) == SPARK_STATUS_CAPACITY_EXCEEDED && small.write_failure_count == 0u && small.eviction_count == 1u && small.failed_status == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,3u * file_bytes,file_bytes,SNAP_RESERVE) == SPARK_STATUS_OK);
	for (index=0u; index<3u; index++)
	{
		CHECK(SparkKvSnapshotWriteBegin(&store,index == 2u ? &stale : &keys[index],0,kinds,bytes,2u,0u,&ticket) == SPARK_STATUS_OK && ticket.file_bytes == file_bytes);
		CHECK(SparkKvSnapshotWriteBegin(&store,&keys[3],0,kinds,bytes,2u,0u,&second) == SPARK_STATUS_BUSY);
		memset(ticket.segments[0].data,(int)(index + 1u),100u);
		memset(ticket.segments[1].data,(int)(index + 7u),5000u);
		CHECK(SparkKvSnapshotWriteCommit(&store,&ticket) == SPARK_STATUS_OK && ticket.job == 0);
		CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	}
	CHECK(SparkKvSnapshotStoreSample(&store,&sample) == SPARK_STATUS_OK && sample.write_count == 3u && sample.file_count == 3u && sample.used_bytes == 3u * file_bytes && sample.queue_full_count == 3u && sample.queued_bytes == 0u && sample.runtime == 0);
	CHECK(SparkKvSnapshotWriteBegin(&store,&keys[0],0,kinds,bytes,2u,0u,&ticket) == SPARK_STATUS_DUPLICATE && ticket.job == 0 && store.duplicate_count == 1u);
	readback[0] = (SparkKvSnapshotSegment){.kind=1u,.bytes=sizeof(out_chain),.data=out_chain};
	readback[1] = (SparkKvSnapshotSegment){.kind=2u,.bytes=sizeof(out_payload),.data=out_payload};
	CHECK(SparkKvSnapshotRead(&store,&keys[1],readback,2u) == SPARK_STATUS_OK && out_chain[0] == 2u && out_payload[4999] == 8u);
	CHECK(SparkKvSnapshotRead(&store,&keys[0],readback,2u) == SPARK_STATUS_OK && out_chain[0] == 1u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&keys[3],0,kinds,bytes,2u,0u,&ticket) == SPARK_STATUS_OK);
	memset(ticket.segments[0].data,9,100u);
	memset(ticket.segments[1].data,9,5000u);
	CHECK(SparkKvSnapshotWriteCommit(&store,&ticket) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(store.eviction_count == 1u && store.file_count == 3u && store.used_bytes == 3u * file_bytes);
	CHECK(SparkKvSnapshotStat(&store,&stale,&header) == SPARK_STATUS_NOT_FOUND && SparkKvSnapshotStat(&store,&keys[1],&header) == SPARK_STATUS_OK && SparkKvSnapshotStat(&store,&keys[3],&header) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotWriteBegin(&store,&stale,0,kinds,bytes,2u,0u,&ticket) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotWriteBegin(&store,&stale,0,kinds,bytes,2u,0u,&second) == SPARK_STATUS_PENDING && second.job == 0);
	SparkKvSnapshotWriteCancel(&store,&ticket);
	CHECK(ticket.job == 0 && store.queued_bytes == 0u && store.queued_count == 0u && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 4u);
	CHECK(SparkKvSnapshotStat(&store,&stale,&header) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotWriteBegin(&store,&stale,0,kinds,bytes,2u,0u,&second) == SPARK_STATUS_OK);
	SparkKvSnapshotWriteCancel(&store,&second);
	CHECK(SparkKvSnapshotWrite(&store,&stale,0,readback,2u) == SPARK_STATUS_OK && store.eviction_count == 2u && SparkKvSnapshotStat(&store,&keys[1],&header) == SPARK_STATUS_NOT_FOUND);
	CHECK(SparkKvSnapshotPrune(&store,keys[0].layout_sha256) == SPARK_STATUS_OK && store.pruned_count == 1u && SparkKvSnapshotStat(&store,&stale,&header) == SPARK_STATUS_NOT_FOUND && store.file_count == 2u);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,3u * file_bytes,file_bytes,SNAP_RESERVE) == SPARK_STATUS_OK && store.file_count == 2u && store.used_bytes == 2u * file_bytes);
	CHECK(SparkKvSnapshotStat(&store,&keys[0],&header) == SPARK_STATUS_OK && SparkKvSnapshotStat(&store,&keys[3],&header) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,file_bytes,file_bytes,SNAP_RESERVE) == SPARK_STATUS_OK && store.file_count == 1u && store.eviction_count == 1u);
	SparkKvSnapshotStoreClose(&store);
}

static SparkStatus SnapTreeWrite(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent)
{
	static uint8_t payload[100];
	SparkKvSnapshotSegment segment = {.kind=2u,.bytes=sizeof(payload),.data=payload};
	return(SparkKvSnapshotWrite(store,key,parent,&segment,1u));
}

static uint32_t SnapTreeTouch(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	uint8_t payload[100];
	uint64_t bytes;
	return(SparkKvSnapshotReadSegment(store,key,0u,2u,payload,sizeof(payload),&bytes) == SPARK_STATUS_OK ? 1u : 0u);
}

static void SnapTestTreeEviction(const char *directory)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotKey root,child,grandchild,other,state,orphan,missing,later,last,parent;
	const uint64_t file_bytes = 2u * SPARK_KV_SNAPSHOT_ALIGNMENT;
	char legacy[1200];
	uint32_t index;
	int descriptor;
	SnapKey(&root,0x31u,0x01u,4u);
	SnapKey(&child,0x31u,0x02u,8u);
	SnapKey(&grandchild,0x31u,0x03u,12u);
	SnapKey(&other,0x31u,0x04u,4u);
	SnapKey(&orphan,0x31u,0x05u,16u);
	SnapKey(&missing,0x31u,0x06u,12u);
	SnapKey(&later,0x31u,0x07u,4u);
	SnapKey(&last,0x31u,0x08u,4u);
	state = child;
	state.file_kind = SPARK_KV_SNAPSHOT_FILE_STATE;
	index = (uint32_t)snprintf(legacy,sizeof(legacy),"%s/",directory);
	for (uint32_t byte=0u; byte<32u; byte++)
		index += (uint32_t)snprintf(legacy + index,sizeof(legacy) - index,"31");
	index += (uint32_t)snprintf(legacy + index,sizeof(legacy) - index,"-");
	for (uint32_t byte=0u; byte<32u; byte++)
		index += (uint32_t)snprintf(legacy + index,sizeof(legacy) - index,"01");
	(void)snprintf(legacy + index,sizeof(legacy) - index,"-4%s",SPARK_KV_SNAPSHOT_SUFFIX);
	descriptor = open(legacy,O_WRONLY | O_CREAT | O_EXCL,0600);
	CHECK(descriptor >= 0 && close(descriptor) == 0);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,5u * file_bytes,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(store.removed_legacy_count == 1u && access(legacy,F_OK) != 0 && store.file_count == 0u);
	CHECK(SnapTreeWrite(&store,&root,0) == SPARK_STATUS_OK && SnapTreeWrite(&store,&child,&root) == SPARK_STATUS_OK && SnapTreeWrite(&store,&grandchild,&child) == SPARK_STATUS_OK);
	CHECK(SnapTreeWrite(&store,&other,0) == SPARK_STATUS_OK && SnapTreeWrite(&store,&state,0) == SPARK_STATUS_OK);
	CHECK(store.file_count == 5u && store.used_bytes == 5u * file_bytes && store.orphan_count == 0u);
	CHECK(SparkKvSnapshotParent(&store,&grandchild,&parent) == SPARK_STATUS_OK && memcmp(&parent,&child,sizeof(parent)) == 0);
	CHECK(SparkKvSnapshotParent(&store,&root,&parent) == SPARK_STATUS_OK && parent.token_count == 0u);
	CHECK(SparkKvSnapshotParent(&store,&missing,&parent) == SPARK_STATUS_NOT_FOUND);
	CHECK(SnapTreeWrite(&store,&child,&other) == SPARK_STATUS_OK && store.duplicate_count == 1u);
	CHECK(SnapTreeWrite(&store,&orphan,&missing) == SPARK_STATUS_OK && store.eviction_count == 1u && store.orphan_count == 1u);
	CHECK(SparkKvSnapshotContains(&store,&root) == 1u && SparkKvSnapshotContains(&store,&child) == 1u && SparkKvSnapshotContains(&store,&grandchild) == 0u);
	CHECK(SnapTreeTouch(&store,&child) == 1u && SnapTreeTouch(&store,&other) == 1u && SnapTreeTouch(&store,&state) == 1u);
	CHECK(SnapTreeWrite(&store,&later,0) == SPARK_STATUS_OK && store.eviction_count == 2u && store.orphan_count == 0u && SparkKvSnapshotContains(&store,&orphan) == 0u);
	CHECK(SparkKvSnapshotContains(&store,&root) == 1u && SparkKvSnapshotContains(&store,&child) == 1u && SparkKvSnapshotContains(&store,&other) == 1u);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,5u * file_bytes,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && store.file_count == 5u && store.orphan_count == 0u);
	CHECK(SparkKvSnapshotParent(&store,&child,&parent) == SPARK_STATUS_OK && memcmp(&parent,&root,sizeof(parent)) == 0);
	CHECK(SnapTreeTouch(&store,&other) == 1u && SnapTreeTouch(&store,&state) == 1u && SnapTreeTouch(&store,&later) == 1u);
	CHECK(SnapTreeWrite(&store,&last,0) == SPARK_STATUS_OK && SparkKvSnapshotContains(&store,&child) == 0u && SparkKvSnapshotContains(&store,&root) == 1u);
	CHECK(SnapTreeWrite(&store,&grandchild,0) == SPARK_STATUS_OK && SparkKvSnapshotContains(&store,&root) == 0u);
	CHECK(SparkKvSnapshotContains(&store,&other) == 1u && SparkKvSnapshotContains(&store,&state) == 1u && SparkKvSnapshotContains(&store,&later) == 1u && SparkKvSnapshotContains(&store,&last) == 1u);
	SparkKvSnapshotStoreClose(&store);
}

static uint64_t SnapNow(void)
{
	struct timespec now;
	return(clock_gettime(CLOCK_REALTIME,&now) == 0 ? (uint64_t)now.tv_sec : 0u);
}

static SparkStatus SnapPathIn(const char *directory,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,char *path,uint32_t capacity)
{
	SparkKvSnapshotStore shadow;
	memset(&shadow,0,sizeof(shadow));
	(void)snprintf(shadow.directory,sizeof(shadow.directory),"%s",directory);
	return(SparkKvSnapshotPath(&shadow,key,parent,path,capacity));
}

static void SnapAge(const char *directory,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,uint64_t seconds_ago)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct timespec stamps[2];
	stamps[0].tv_sec = stamps[1].tv_sec = (time_t)(SnapNow() - seconds_ago);
	stamps[0].tv_nsec = stamps[1].tv_nsec = 0;
	CHECK(SnapPathIn(directory,key,parent,path,sizeof(path)) == SPARK_STATUS_OK && utimensat(AT_FDCWD,path,stamps,0) == 0);
}

static uint64_t SnapAgeOf(const char *directory,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct stat info;
	if ( SnapPathIn(directory,key,parent,path,sizeof(path)) != SPARK_STATUS_OK || stat(path,&info) != 0 )
		return(UINT64_MAX);
	return(SnapNow() - (uint64_t)info.st_mtime);
}

static void SnapTestRetention(const char *directory)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotKey root,child,idle,state,stale,grandchild;
	const uint64_t day = SPARK_KV_RETENTION_SECONDS,file_bytes = 2u * SPARK_KV_SNAPSHOT_ALIGNMENT;
	uint64_t now;
	CHECK(day == UINT64_C(86400));
	SnapKey(&root,0x51u,0x01u,4u);
	SnapKey(&child,0x51u,0x02u,8u);
	SnapKey(&idle,0x51u,0x03u,4u);
	SnapKey(&stale,0x51u,0x04u,4u);
	SnapKey(&grandchild,0x51u,0x05u,12u);
	state = child;
	state.file_kind = SPARK_KV_SNAPSHOT_FILE_STATE;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,64u * file_bytes,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && store.expired_count == 0u);
	CHECK(SnapTreeWrite(&store,&root,0) == SPARK_STATUS_OK && SnapTreeWrite(&store,&child,&root) == SPARK_STATUS_OK);
	CHECK(SnapTreeWrite(&store,&idle,0) == SPARK_STATUS_OK && SnapTreeWrite(&store,&state,0) == SPARK_STATUS_OK && SnapTreeWrite(&store,&stale,0) == SPARK_STATUS_OK);
	CHECK(store.file_count == 5u);
	now = SnapNow();
	CHECK(SparkKvSnapshotExpire(&store,now) == SPARK_STATUS_OK && SparkKvSnapshotExpire(&store,now + day - 120u) == SPARK_STATUS_OK && store.expired_count == 0u && store.file_count == 5u);
	SparkKvSnapshotStoreClose(&store);
	SnapAge(directory,&root,0,3600u);
	SnapAge(directory,&child,&root,3600u);
	SnapAge(directory,&idle,0,3600u);
	SnapAge(directory,&state,0,3600u);
	SnapAge(directory,&stale,0,day + 10u);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,64u * file_bytes,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(store.expired_count == 1u && store.expired_bytes == file_bytes && store.file_count == 4u && SnapAgeOf(directory,&stale,0) == UINT64_MAX && SparkKvSnapshotContains(&store,&stale) == 0u);
	CHECK(SnapTreeTouch(&store,&state) == 1u && SnapAgeOf(directory,&state,0) < 60u);
	CHECK(SnapTreeWrite(&store,&grandchild,&child) == SPARK_STATUS_OK && SnapAgeOf(directory,&child,&root) < 60u && SnapAgeOf(directory,&root,0) >= 3500u);
	CHECK(SnapAgeOf(directory,&idle,0) >= 3500u);
	now = SnapNow();
	CHECK(SparkKvSnapshotExpire(&store,now + day - 1800u) == SPARK_STATUS_OK);
	CHECK(store.expired_count == 3u && store.expired_bytes == 3u * file_bytes && store.file_count == 3u);
	CHECK(SparkKvSnapshotContains(&store,&idle) == 0u && SparkKvSnapshotContains(&store,&root) == 0u && SnapAgeOf(directory,&idle,0) == UINT64_MAX && SnapAgeOf(directory,&root,0) == UINT64_MAX);
	CHECK(SparkKvSnapshotContains(&store,&state) == 1u && SparkKvSnapshotContains(&store,&child) == 1u && SparkKvSnapshotContains(&store,&grandchild) == 1u);
	CHECK(SparkKvSnapshotExpire(&store,now + day + 1u) == SPARK_STATUS_OK && store.file_count == 0u && store.used_bytes == 0u && store.expired_count == 6u);
	CHECK(SnapAgeOf(directory,&state,0) == UINT64_MAX && SnapAgeOf(directory,&grandchild,&child) == UINT64_MAX);
	CHECK(SparkKvSnapshotExpire(0,now) == SPARK_STATUS_INVALID_ARGUMENT);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a file unused for 24 hours is deleted, at open by its mtime and while open by the sweep; a read stamps its use and a child write stamps its parent, in the file's mtime so a restart keeps it\n");
}

static void SnapTestChainExpiry(const char *directory)
{
	enum { depth = 2500 };
	SparkKvSnapshotStore store;
	SparkKvSnapshotKey keys[2];
	const uint64_t file_bytes = 2u * SPARK_KV_SNAPSHOT_ALIGNMENT;
	uint32_t index;
	uint64_t now;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,(depth + 4u) * file_bytes,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	for (index=0u; index<depth; index++)
	{
		SnapKey(&keys[index & 1u],0x61u,0x07u,4u * (index + 1u));
		CHECK(SnapTreeWrite(&store,&keys[index & 1u],index == 0u ? 0 : &keys[(index + 1u) & 1u]) == SPARK_STATUS_OK);
	}
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.file_count == depth);
	now = SnapNow();
	CHECK(SparkKvSnapshotExpire(&store,now + SPARK_KV_RETENTION_SECONDS - 60u) == SPARK_STATUS_OK && store.file_count == depth && store.expired_count == 0u);
	CHECK(SparkKvSnapshotExpire(&store,now + SPARK_KV_RETENTION_SECONDS + 60u) == SPARK_STATUS_OK);
	CHECK(store.file_count == 0u && store.used_bytes == 0u && store.expired_count == depth && store.orphan_count == 0u);
	SnapKey(&keys[0],0x61u,0x07u,4u);
	CHECK(SparkKvSnapshotContains(&store,&keys[0]) == 0u && SnapAgeOf(directory,&keys[0],0) == UINT64_MAX);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a %u-file chain idle for 24 hours expires completely in one call, more than one sweep batch, leaving no orphan\n",(unsigned)depth);
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
	if ( SparkKvPageCacheInitialize(&fixture->cache,&config) != SPARK_STATUS_OK || SparkKvPageCacheAttachStateStore(&fixture->cache,&fixture->stores[1],4u * SNAP_BLOCK_TOKENS) != SPARK_STATUS_OK )
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

static void SnapPublishChain(SnapFixture *fixture,const uint32_t *ends,uint32_t count,uint8_t pages[][SNAP_PAGE_BYTES],uint8_t states[][SNAP_STATE_BYTES])
{
	SparkModelDriverCacheLane lane;
	SparkKvCacheBlockView view;
	uint32_t step,page,position = 0u,byte;
	for (step=0u; step<count; step++)
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

static void SnapPrefill(SnapFixture *fixture,uint8_t pages[3][SNAP_PAGE_BYTES],uint8_t states[3][SNAP_STATE_BYTES])
{
	static const uint32_t ends[3] = {4u,8u,9u};
	SnapPublishChain(fixture,ends,3u,pages,states);
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

static void SnapLinkKey(SparkKvSnapshotKey *key,uint8_t layout,uint8_t identity_seed,uint32_t tokens)
{
	memset(key,0,sizeof(*key));
	memset(key->layout_sha256,layout,sizeof(key->layout_sha256));
	SnapIdentity((SparkModelDriverCacheIdentity *)key->identity_sha256,identity_seed);
	key->token_count = tokens;
}

static void SnapPagePath(SparkKvSnapshotStore *store,uint8_t identity_seed,uint32_t tokens,uint32_t file_kind,char *path,uint32_t capacity)
{
	SparkKvSnapshotKey key,parent;
	SnapLinkKey(&key,0x44u,identity_seed,tokens);
	memset(&parent,0,sizeof(parent));
	if ( file_kind == SPARK_KV_SNAPSHOT_FILE_PAGES )
		CHECK(SparkKvSnapshotParent(store,&key,&parent) == SPARK_STATUS_OK);
	key.file_kind = file_kind;
	CHECK(SparkKvSnapshotPath(store,&key,&parent,path,capacity) == SPARK_STATUS_OK);
}

static void SnapTestPageCacheRoundTrip(const char *directory)
{
	static SnapFixture source,fresh,other;
	SparkKvSnapshotStore store;
	SparkModelDriverCacheIdentity identity;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x44u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheRestorePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && store.read_count == 0u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(source.snapshot.save_count == 1u && source.snapshot.save_page_count == 6u && store.file_count == 6u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 6u && store.duplicate_count == 0u);
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
	SnapPagePath(&store,20u,8u,SPARK_KV_SNAPSHOT_FILE_PAGES,path,sizeof(path));
	SnapFlip(path,SPARK_KV_SNAPSHOT_ALIGNMENT + 3u);
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
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,SPARK_KV_SNAPSHOT_ALIGNMENT * 12u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x77u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	SnapKey(&blocker_key,0x78u,0x01u,1u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&blocker_key,0,&kind,&bytes,1u,0u,&blocker) == SPARK_STATUS_OK);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_BUSY);
	CHECK(source.snapshot.save_page_count == 0u && store.queue_full_count == 1u && store.write_count == 0u);
	SparkKvSnapshotWriteCancel(&store,&blocker);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(store.write_count == 6u && source.snapshot.save_page_count == 6u && store.file_count == 6u);
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
	SparkModelDriverCacheIdentity identity;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16];
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
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
	CHECK(source.snapshot.save_count == 1u && source.snapshot.save_page_count == 4u);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 4u);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(source.snapshot.save_page_count == 6u && store.write_count == 6u);
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
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
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
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 4u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestResetKeepsPublishedSaves(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvLaneTransaction owners[4];
	SparkKvLaneTransactions transactions;
	SparkModelDriverCacheLane lanes[2];
	SparkModelDriverAdmissionRequest request;
	SparkKvPageCacheSaveWork work;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t logical[16],physical[16],entries,allocated;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x68u) == 0);
	SnapPrefill(&source,pages,states);
	SnapPublishSecond(&source);
	SnapTransactionsInit(&transactions,owners,&source.cache,logical,physical);
	SnapLane(&lanes[0],1u,0u,9u,9u);
	lanes[0].flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE;
	SnapReleaseRequest(&request,lanes,1u,1u);
	CHECK(SparkKvLaneTransactionsAdmit(&transactions,&request) == SPARK_STATUS_OK && SparkKvPageCacheSavePending(&source.cache) == 1u);
	entries = SnapEntryCount(&source);
	allocated = SnapAllocatedPages(&source);
	CHECK(entries != 0u && allocated != 0u);
	CHECK(SparkKvPageCacheSaveTake(&source.cache,&work) == SPARK_STATUS_OK && source.snapshot.in_flight == 1u);
	CHECK(SparkKvLaneTransactionsReset(&transactions) == SPARK_STATUS_BUSY && source.snapshot.save_cancelled_count == 0u);
	CHECK(SparkKvPageCacheSaveFinish(&source.cache,&work,SparkKvPageCacheSaveCopy(&source.cache,&work)) == SPARK_STATUS_OK);
	CHECK(SparkKvLaneTransactionsReset(&transactions) == SPARK_STATUS_OK && source.snapshot.save_cancelled_count == 0u);
	CHECK(source.cache.live_sequence_count == 0u && SnapEntryCount(&source) == entries && SnapAllocatedPages(&source) == allocated);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && source.snapshot.in_flight == 0u && SparkKvPageCacheSavePending(&source.cache) == 0u && source.snapshot.save_count == 1u);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count != 0u && store.write_count == source.snapshot.save_page_count);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapWriteFile(SparkKvSnapshotStore *store,uint8_t layout,uint8_t identity_seed,uint32_t tokens,uint8_t parent_seed,uint32_t parent_tokens,uint64_t page_bytes,uint64_t state_bytes)
{
	static uint8_t page[2u * SNAP_PAGE_BYTES],state[2u * SNAP_STATE_BYTES];
	SparkKvSnapshotSegment segment;
	SparkKvSnapshotKey key,parent;
	SnapLinkKey(&key,layout,identity_seed,tokens);
	SnapLinkKey(&parent,layout,parent_seed,parent_tokens);
	segment = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES,.bytes=page_bytes,.data=page};
	CHECK(SparkKvSnapshotWrite(store,&key,parent_tokens != 0u ? &parent : 0,&segment,1u) == SPARK_STATUS_OK);
	if ( state_bytes == 0u )
		return;
	key.file_kind = SPARK_KV_SNAPSHOT_FILE_STATE;
	segment = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_STATE,.bytes=state_bytes,.data=state};
	CHECK(SparkKvSnapshotWrite(store,&key,0,&segment,1u) == SPARK_STATUS_OK);
}

static void SnapUnreadable(SparkKvSnapshotStore *store,uint8_t identity_seed,uint32_t tokens)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SnapPagePath(store,identity_seed,tokens,SPARK_KV_SNAPSHOT_FILE_PAGES,path,sizeof(path));
	CHECK(chmod(path,0) == 0);
}

static uint32_t SnapKeyPresent(SparkKvSnapshotStore *store,uint8_t identity_seed,uint32_t tokens,uint32_t file_kind)
{
	SparkKvSnapshotKey key;
	SnapLinkKey(&key,0x44u,identity_seed,tokens);
	key.file_kind = file_kind;
	return(SparkKvSnapshotContains(store,&key));
}

static void SnapTestReadOutcomes(const char *directory)
{
	static SnapFixture source,fresh;
	SparkKvSnapshotStore store,sample;
	SparkModelDriverCacheIdentity identity;
	SparkKvSnapshotSegment segment;
	SparkKvSnapshotKey key,parent;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint64_t writes,duplicates;
	CHECK(geteuid() != 0);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x44u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.file_count == 6u);
	SnapFixtureClose(&source);
	SnapWriteFile(&store,0x44u,90u,4u,0u,0u,SNAP_PAGE_BYTES - 16u,SNAP_STATE_BYTES);
	SnapWriteFile(&store,0x44u,91u,4u,0u,0u,SNAP_PAGE_BYTES,SNAP_STATE_BYTES - 16u);
	CHECK(SnapFixtureOpen(&fresh,&store,0x44u) == 0);
	SnapIdentity(&identity,90u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,4u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_corrupt_count == 1u && fresh.snapshot.restore_miss_count == 1u && fresh.snapshot.restore_failure_count == 0u && SnapKeyPresent(&store,90u,4u,SPARK_KV_SNAPSHOT_FILE_PAGES) == 0u);
	SnapIdentity(&identity,91u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,4u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_corrupt_count == 2u && fresh.snapshot.restore_failure_count == 0u && SnapKeyPresent(&store,91u,4u,SPARK_KV_SNAPSHOT_FILE_STATE) == 0u &&
		SnapKeyPresent(&store,91u,4u,SPARK_KV_SNAPSHOT_FILE_PAGES) == 1u);
	SnapUnreadable(&store,30u,9u);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_read_error_count == 1u && fresh.snapshot.restore_miss_count == 3u && fresh.snapshot.restore_failure_count == 0u && SnapKeyPresent(&store,30u,9u,SPARK_KV_SNAPSHOT_FILE_PAGES) == 0u);
	CHECK(SparkKvSnapshotStoreSample(&store,&sample) == SPARK_STATUS_OK);
	writes = sample.write_count;
	duplicates = sample.duplicate_count;
	SnapLinkKey(&key,0x44u,30u,9u);
	SnapLinkKey(&parent,0x44u,20u,8u);
	segment = (SparkKvSnapshotSegment){.kind=SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES,.bytes=SNAP_PAGE_BYTES,.data=pages[0]};
	CHECK(SparkKvSnapshotWrite(&store,&key,&parent,&segment,1u) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotStoreSample(&store,&sample) == SPARK_STATUS_OK && sample.write_count == writes + 1u && sample.duplicate_count == duplicates);
	SnapFixtureClose(&fresh);
	CHECK(SnapFixtureOpen(&fresh,&store,0x44u) == 0);
	SnapUnreadable(&store,10u,4u);
	SnapIdentity(&identity,20u);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,8u) == SPARK_STATUS_NOT_FOUND);
	CHECK(fresh.snapshot.restore_read_error_count == 1u && fresh.snapshot.restore_failure_count == 0u && SnapEntryCount(&fresh) == 0u && SnapAllocatedPages(&fresh) == 0u);
	SnapFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);
}

static uint32_t SnapStateFileHeld(SparkKvSnapshotStore *store,uint8_t identity_seed,uint32_t tokens)
{
	SparkKvSnapshotKey key;
	SnapLinkKey(&key,0x48u,identity_seed,tokens);
	key.file_kind = SPARK_KV_SNAPSHOT_FILE_STATE;
	return(SparkKvSnapshotContains(store,&key));
}

static uint32_t SnapEntryAt(const SnapFixture *fixture,uint32_t tokens)
{
	uint32_t index;
	for (index=0u; index<SNAP_PAGES; index++)
		if ( (fixture->entries[index].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID) != 0u && fixture->entries[index].token_count == tokens )
			return(index);
	return(SPARK_KV_PAGE_CACHE_NO_INDEX);
}

static void SnapTestSpareRetention(const char *directory)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotWriteTicket tickets[8];
	SparkKvSnapshotKey key;
	uint32_t kinds[1] = {1u},index,serial = 0u;
	uint64_t big[1] = {200000u},small[1] = {1000u},big_file = 0u;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,UINT64_C(1) << 26,UINT64_C(1) << 21,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(store.spare_maximum_bytes == store.queue_maximum_bytes / 4u && store.spare_bytes == 0u);
	for (index=0u; index<8u; index++)
	{
		SnapKey(&key,0x5au,(uint8_t)(index + 1u),4u * (index + 1u));
		CHECK(SparkKvSnapshotWriteBegin(&store,&key,0,kinds,big,1u,0u,&tickets[index]) == SPARK_STATUS_OK);
		big_file = tickets[index].file_bytes;
		memset(tickets[index].segments[0].data,(int)index,(size_t)big[0]);
	}
	for (index=0u; index<8u; index++)
		CHECK(SparkKvSnapshotWriteCommit(&store,&tickets[index]) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.write_count == 8u);
	CHECK(store.spare_bytes != 0u && store.spare_bytes <= store.spare_maximum_bytes && store.spare_maximum_bytes < 8u * big_file);
	SparkKvSnapshotStoreClose(&store);

	CHECK(SparkKvSnapshotStoreOpen(&store,directory,UINT64_C(1) << 26,UINT64_C(1) << 21,SNAP_RESERVE) == SPARK_STATUS_OK);
	SnapKey(&key,0x5bu,(uint8_t)++serial,4u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&key,0,kinds,small,1u,0u,&tickets[0]) == SPARK_STATUS_OK && SparkKvSnapshotWriteCommit(&store,&tickets[0]) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	SnapKey(&key,0x5bu,(uint8_t)++serial,8u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&key,0,kinds,big,1u,0u,&tickets[0]) == SPARK_STATUS_OK && SparkKvSnapshotWriteCommit(&store,&tickets[0]) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(store.spare_bytes == big[0] + small[0]);
	SnapKey(&key,0x5bu,(uint8_t)++serial,12u);
	CHECK(SparkKvSnapshotWriteBegin(&store,&key,0,kinds,small,1u,0u,&tickets[1]) == SPARK_STATUS_OK);
	CHECK(store.spare_bytes == big[0]);
	CHECK(SparkKvSnapshotWriteCommit(&store,&tickets[1]) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a burst of large writes keeps at most a quarter of the queue as spare buffers, and a small write reuses the smallest spare that fits, not a large one\n");
}

static void SnapTestStatesSpillBeforeDemotion(const char *directory)
{
	static const uint32_t ends[4] = {4u,8u,12u,16u};
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvPageCacheSaveOrder order[SNAP_PAGES];
	uint8_t pages[4][SNAP_PAGE_BYTES],states[4][SNAP_STATE_BYTES];
	uint32_t marked,deferred,ineligible,index,unsaved_entry;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x48u) == 0);
	SnapPublishChain(&source,ends,4u,pages,states);
	CHECK(SparkKvPageCacheMarkAllUnsaved(&source.cache,order,SNAP_PAGES,&marked,&deferred,&ineligible) == SPARK_STATUS_OK && marked != 0u && deferred == 0u);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(store.write_count == 5u && SnapStateFileHeld(&store,40u,16u) == 1u);
	CHECK(SnapStateFileHeld(&store,10u,4u) == 0u && SnapStateFileHeld(&store,20u,8u) == 0u && SnapStateFileHeld(&store,30u,12u) == 0u);
	SparkKvPageCacheSpillStates(&source.cache);
	CHECK(source.cache.state_spill_queued_count == 3u && SparkKvPageCacheSavePending(&source.cache) == 3u);
	CHECK(SparkKvPageCacheSaveDrain(&source.cache) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(SnapStateFileHeld(&store,10u,4u) == 1u && SnapStateFileHeld(&store,20u,8u) == 1u && SnapStateFileHeld(&store,30u,12u) == 1u);
	CHECK((source.entries[SnapEntryAt(&source,16u)].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PERSIST_STATE) == 0u);
	CHECK(SparkKvPageCacheDemoteState(&source.cache,0,0u) == SPARK_STATUS_OK && SparkKvPageCacheDemoteState(&source.cache,0,0u) == SPARK_STATUS_OK &&
		SparkKvPageCacheDemoteState(&source.cache,0,0u) == SPARK_STATUS_OK && SparkKvPageCacheDemoteState(&source.cache,0,0u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	source.sequences[0].next_token_position = 0u;
	CHECK(SparkKvPageCacheDemoteState(&source.cache,0,0u) == SPARK_STATUS_OK);
	CHECK(source.cache.state_pressure_demoted_count == 4u && source.cache.state_dropped_unsaved_count == 0u && source.stores[1].backing_page_count == 0u);
	for (index=0u; index<4u; index++)
		CHECK((source.entries[SnapEntryAt(&source,ends[index])].flags & (SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS | SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED_STATE)) ==
			(SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS | SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED_STATE));
	SnapFixtureClose(&source);

	CHECK(SnapFixtureOpen(&source,&store,0x49u) == 0);
	SnapPublishChain(&source,ends,2u,pages,states);
	unsaved_entry = SnapEntryAt(&source,4u);
	source.sequences[0].next_token_position = 0u;
	CHECK(SparkKvPageCacheDemoteState(&source.cache,0,0u) == SPARK_STATUS_OK);
	CHECK(source.cache.state_dropped_unsaved_count == 1u && (source.entries[unsaved_entry].flags & SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS) != 0u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a state off the anchor stride stays in memory while its pages save, the anchor saves at once, a store near its capacity spills its oldest records, demotion takes saved records first and keeps them restorable, and only with nothing saved left does it drop an unsaved record, counted\n");
}

static void SnapTestMarkAllOrder(const char *directory)
{
	static SnapFixture source;
	SparkKvSnapshotStore store;
	SparkKvPageCacheSaveOrder order[SNAP_PAGES];
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES];
	uint32_t marked,deferred,ineligible,unsaved,index,last;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SnapFixtureOpen(&source,&store,0x46u) == 0);
	SnapPrefill(&source,pages,states);
	SnapPublishSecond(&source);
	source.entries[source.by_page[source.sequences[1].terminal_entry_index != SPARK_KV_PAGE_CACHE_NO_INDEX ? source.entries[source.sequences[1].terminal_entry_index].logical_page_index : 0u]].flags |= SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE;
	SparkKvPageCachePersistAllStates(&source.cache);
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
	CHECK(SparkKvPageCacheCountUnsaved(&source.cache,&unsaved,&ineligible) == SPARK_STATUS_OK && unsaved == 0u && ineligible == 1u && store.write_count == 6u);
	SnapFixtureClose(&source);
	SparkKvSnapshotStoreClose(&store);
}

static void SnapTestFailedStoreRecovers(const char *directory)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotSegment segment;
	SparkKvSnapshotKey first,second,third;
	uint8_t payload[512];
	char shard[1200];
	struct timespec pause = {0,150000000};
	memset(payload,0x6b,sizeof(payload));
	segment = (SparkKvSnapshotSegment){.kind=1u,.bytes=sizeof(payload),.data=payload};
	SnapKey(&first,0x41u,0x51u,1u);
	SnapKey(&second,0x42u,0x52u,2u);
	SnapKey(&third,0x43u,0x53u,3u);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	store.failure_retry_ns = 100000000u;
	(void)snprintf(shard,sizeof(shard),"%s/%02x",directory,(unsigned)first.identity_sha256[0]);
	CHECK(chmod(shard,0500) == 0);
	CHECK(SparkKvSnapshotWrite(&store,&first,0,&segment,1u) == SPARK_STATUS_IO_ERROR && store.failed_status == SPARK_STATUS_IO_ERROR && store.write_failure_count == 1u);
	CHECK(chmod(shard,0700) == 0);
	CHECK(SparkKvSnapshotWrite(&store,&second,0,&segment,1u) == SPARK_STATUS_IO_ERROR && store.write_failure_count == 1u);
	(void)nanosleep(&pause,0);
	CHECK(SparkKvSnapshotWrite(&store,&second,0,&segment,1u) == SPARK_STATUS_OK && store.failed_status == SPARK_STATUS_OK && store.recovered_count == 1u);
	CHECK(SparkKvSnapshotWrite(&store,&third,0,&segment,1u) == SPARK_STATUS_OK && store.write_count == 2u);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a failed write refuses writes for the retry window, then a probe that succeeds clears the failure\n");
}

static void SnapTestFailedWriteResaves(const char *directory)
{
	static SnapFixture source,fresh;
	SparkKvSnapshotStore store;
	SparkKvSnapshotSegment segment;
	SparkModelDriverCacheIdentity identity;
	SparkKvSnapshotKey root;
	uint8_t pages[3][SNAP_PAGE_BYTES],states[3][SNAP_STATE_BYTES],payload[64];
	char shard[1200];
	uint32_t attempt;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 24u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	store.failure_retry_ns = 0u;
	CHECK(SnapFixtureOpen(&source,&store,0x44u) == 0);
	SnapPrefill(&source,pages,states);
	SnapIdentity(&identity,30u);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK && SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK && store.file_count == 6u);
	SnapLinkKey(&root,0x44u,10u,4u);
	root.file_kind = SPARK_KV_SNAPSHOT_FILE_PAGES;
	CHECK(SparkKvSnapshotRemove(&store,&root) == SPARK_STATUS_OK && SparkKvSnapshotContains(&store,&root) == 0u);
	memset(payload,0x5a,sizeof(payload));
	segment = (SparkKvSnapshotSegment){.kind=1u,.bytes=sizeof(payload),.data=payload};
	(void)snprintf(shard,sizeof(shard),"%s/%02x",directory,(unsigned)root.identity_sha256[0]);
	CHECK(chmod(shard,0500) == 0);
	CHECK(SparkKvSnapshotWrite(&store,&root,0,&segment,1u) == SPARK_STATUS_IO_ERROR && store.write_failure_count == 1u && SparkKvSnapshotContains(&store,&root) == 0u);
	CHECK(chmod(shard,0700) == 0);
	CHECK(SparkKvPageCacheSavePrefix(&source.cache,&identity,9u) == SPARK_STATUS_OK);
	for (attempt=0u; attempt<50u && (SparkKvPageCacheSavePending(&source.cache) != 0u || SparkKvSnapshotContains(&store,&root) == 0u); attempt++)
	{
		CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
		(void)SparkKvPageCacheSaveDrain(&source.cache);
	}
	CHECK(SparkKvSnapshotFlush(&store) == SPARK_STATUS_OK);
	CHECK(source.snapshot.write_failed_count == 1u && source.snapshot.write_resaved_count == 1u && SparkKvSnapshotContains(&store,&root) != 0u && SparkKvPageCacheSavePending(&source.cache) == 0u);
	SnapFixtureClose(&source);
	CHECK(SnapFixtureOpen(&fresh,&store,0x44u) == 0);
	CHECK(SparkKvPageCacheRestorePrefix(&fresh.cache,&identity,9u) == SPARK_STATUS_OK && fresh.snapshot.restore_page_count == 3u && SnapEntryCount(&fresh) == 3u);
	if ( fresh.snapshot.restore_page_count == 3u )
		SnapCheckRestoredChain(&fresh,pages,states);
	SnapFixtureClose(&fresh);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a failed write of a page the cache holds as saved, under saved descendants, clears its saved mark and saves it again, so the chain restores whole\n");
}

static void SnapTestSyncWriteSeesQueuedWrite(const char *directory)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotSegment segment;
	SparkKvSnapshotWriteTicket ticket;
	SparkKvSnapshotFileHeader header;
	SparkKvSnapshotKey key;
	uint8_t payload[256];
	const uint32_t kind = 1u;
	const uint64_t bytes = sizeof(payload);
	memset(payload,0x2d,sizeof(payload));
	segment = (SparkKvSnapshotSegment){.kind=kind,.bytes=bytes,.data=payload};
	SnapKey(&key,0x61u,0x71u,4u);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,1u << 20u,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotWriteBegin(&store,&key,0,&kind,&bytes,1u,0u,&ticket) == SPARK_STATUS_OK);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,&segment,1u) == SPARK_STATUS_PENDING && store.write_count == 0u &&
		SparkKvSnapshotStat(&store,&key,&header) == SPARK_STATUS_NOT_FOUND);
	SparkKvSnapshotWriteCancel(&store,&ticket);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,&segment,1u) == SPARK_STATUS_OK && store.write_count == 1u);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: a synchronous write of a key with a queued write answers PENDING and writes nothing\n");
}

static void SnapScaleKey(SparkKvSnapshotKey *key,uint32_t chain,uint32_t page)
{
	memset(key,0,sizeof(*key));
	memset(key->layout_sha256,0x61u,sizeof(key->layout_sha256));
	key->identity_sha256[0] = (uint8_t)(chain * 37u + page * 11u);
	memcpy(key->identity_sha256 + 1u,&chain,sizeof(chain));
	memcpy(key->identity_sha256 + 5u,&page,sizeof(page));
	key->token_count = 4u * (page + 1u);
}

static void SnapScaleFile(const char *path)
{
	int descriptor = open(path,O_WRONLY | O_CREAT | O_TRUNC,0600);
	CHECK(descriptor >= 0 && ftruncate(descriptor,(off_t)SPARK_KV_SNAPSHOT_ALIGNMENT) == 0 && close(descriptor) == 0);
}

static void SnapTestScale(const char *directory)
{
	const uint32_t chains = 40u,pages = 100u,heads = chains / 4u;
	const uint64_t files = (uint64_t)chains * pages - heads,unlimited = UINT64_C(1) << 40;
	SparkKvSnapshotStore store,names;
	SparkKvSnapshotKey key,parent;
	SparkKvSnapshotSegment segment;
	struct statvfs partition;
	struct stat info;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES],prefix[1200],flat[1200],misplaced[1200],temporary[1200];
	const char *name;
	uint8_t payload[100];
	uint32_t chain,page;
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,unlimited,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	(void)snprintf(prefix,sizeof(prefix),"%s/00",directory);
	CHECK(stat(prefix,&info) == 0 && S_ISDIR(info.st_mode));
	(void)snprintf(prefix,sizeof(prefix),"%s/ff",directory);
	CHECK(stat(prefix,&info) == 0 && S_ISDIR(info.st_mode));
	memset(&names,0,sizeof(names));
	(void)snprintf(names.directory,sizeof(names.directory),"%s",directory);
	for (chain=0u; chain<chains; chain++)
		for (page=0u; page<pages; page++)
		{
			if ( chain % 4u == 0u && page == 0u )
				continue;
			SnapScaleKey(&key,chain,page);
			SnapScaleKey(&parent,chain,page - 1u);
			CHECK(SparkKvSnapshotPath(&names,&key,page != 0u ? &parent : 0,path,sizeof(path)) == SPARK_STATUS_OK);
			(void)snprintf(prefix,sizeof(prefix),"%s/%02x/",directory,(unsigned)key.identity_sha256[0]);
			CHECK(strncmp(path,prefix,strlen(prefix)) == 0);
			SnapScaleFile(path);
		}
	SnapScaleKey(&key,chains + 1u,0u);
	CHECK(SparkKvSnapshotPath(&names,&key,0,path,sizeof(path)) == SPARK_STATUS_OK);
	name = strrchr(path,'/') + 1;
	(void)snprintf(flat,sizeof(flat),"%s/%s",directory,name);
	(void)snprintf(misplaced,sizeof(misplaced),"%s/%02x/%s",directory,(unsigned)(uint8_t)(key.identity_sha256[0] + 1u),name);
	(void)snprintf(temporary,sizeof(temporary),"%s/%02x/%sdead-2",directory,(unsigned)key.identity_sha256[0],SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX);
	SnapScaleFile(flat);
	SnapScaleFile(misplaced);
	SnapScaleFile(temporary);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,unlimited,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK);
	CHECK(store.file_count == files && store.used_bytes == files * SPARK_KV_SNAPSHOT_ALIGNMENT && store.orphan_count == heads);
	CHECK(store.removed_legacy_count == 2u && store.removed_temporary_count == 1u && access(flat,F_OK) != 0 && access(misplaced,F_OK) != 0 && access(temporary,F_OK) != 0);
	memset(payload,7,sizeof(payload));
	segment = (SparkKvSnapshotSegment){.kind=2u,.bytes=sizeof(payload),.data=payload};
	for (chain=0u; chain<chains; chain+=4u)
	{
		SnapScaleKey(&key,chain,0u);
		CHECK(SparkKvSnapshotWrite(&store,&key,0,&segment,1u) == SPARK_STATUS_OK);
	}
	CHECK(store.orphan_count == 0u && store.file_count == files + heads);
	SnapScaleKey(&key,5u,50u);
	CHECK(SparkKvSnapshotRemove(&store,&key) == SPARK_STATUS_OK && store.orphan_count == 1u);
	SnapScaleKey(&key,5u,51u);
	CHECK(SparkKvSnapshotParent(&store,&key,&parent) == SPARK_STATUS_OK && parent.token_count == 4u * 51u);
	SnapScaleKey(&key,5u,50u);
	SnapScaleKey(&parent,5u,49u);
	CHECK(SparkKvSnapshotWrite(&store,&key,&parent,&segment,1u) == SPARK_STATUS_OK && store.orphan_count == 0u && store.file_count == files + heads);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,unlimited,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && store.file_count == files + heads && store.orphan_count == 0u);
	SparkKvSnapshotStoreClose(&store);
	CHECK(statvfs(directory,&partition) == 0);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,unlimited,1u << 20u,(uint64_t)partition.f_blocks * (uint64_t)partition.f_frsize) == SPARK_STATUS_OK);
	CHECK(store.file_count == 0u && store.used_bytes == 0u && store.reserve_eviction_count == files + heads && store.reserve_refusal_count == 1u);
	SnapScaleKey(&key,1u,0u);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,&segment,1u) == SPARK_STATUS_CAPACITY_EXCEEDED && store.reserve_refusal_count == 2u && store.file_count == 0u);
	CHECK(store.write_failure_count == 0u && store.failed_status == SPARK_STATUS_OK);
	SparkKvSnapshotStoreClose(&store);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,unlimited,1u << 20u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotStoreOpen(&store,directory,unlimited,1u << 20u,SNAP_RESERVE) == SPARK_STATUS_OK && store.file_count == 0u);
	CHECK(SparkKvSnapshotWrite(&store,&key,0,&segment,1u) == SPARK_STATUS_OK && store.file_count == 1u && store.reserve_refusal_count == 0u);
	SparkKvSnapshotStoreClose(&store);
	printf("kv snapshot: files land in 256 identity shards, a scan of %llu files keeps chains linked, a written parent adopts its orphans, and the partition reserve evicts then refuses\n",
		(unsigned long long)files);
}

static void SnapPurgeFile(const char *directory,const char *name,uint64_t age_seconds)
{
	char path[1200];
	struct timespec times[2];
	int descriptor;
	(void)snprintf(path,sizeof(path),"%s/%s",directory,name);
	descriptor = open(path,O_WRONLY | O_CREAT | O_TRUNC,0600);
	CHECK(descriptor >= 0 && write(descriptor,"kvdata",6) == 6);
	if ( descriptor >= 0 )
		(void)close(descriptor);
	times[0].tv_sec = times[1].tv_sec = time(0) - (time_t)age_seconds;
	times[0].tv_nsec = times[1].tv_nsec = 0;
	CHECK(utimensat(AT_FDCWD,path,times,0) == 0);
}

static uint32_t SnapPurgeExists(const char *directory,const char *name)
{
	char path[1200];
	struct stat info;
	(void)snprintf(path,sizeof(path),"%s/%s",directory,name);
	return(lstat(path,&info) == 0 ? 1u : 0u);
}

static void SnapTestPurge(const char *root)
{
	const uint64_t old = SPARK_KV_RETENTION_SECONDS + 60u,fresh = SPARK_KV_RETENTION_SECONDS - 600u;
	char store[1024],nested[1024],busy[1024],outside[1024],link_path[1024],lock_path[1200],store_shard[1100],busy_shard[1100];
	SparkKvSnapshotPurgeResult result;
	int lock;
	(void)snprintf(store,sizeof(store),"%s/lanes/old/root.failed-1/kvsnapshot",root);
	(void)snprintf(nested,sizeof(nested),"%s/kvsnapshot/flash-tip",root);
	(void)snprintf(busy,sizeof(busy),"%s/lanes/live/root/kvsnapshot",root);
	(void)snprintf(outside,sizeof(outside),"%s-outside",root);
	(void)snprintf(link_path,sizeof(link_path),"%s/lanes/linked",root);
	CHECK(mkdir(root,0700) == 0 || access(root,F_OK) == 0);
	{
		char build[1024];
		const char *parts[] = {"/lanes","/lanes/old","/lanes/old/root.failed-1","/lanes/old/root.failed-1/kvsnapshot","/kvsnapshot","/kvsnapshot/flash-tip",
			"/lanes/live","/lanes/live/root","/lanes/live/root/kvsnapshot"};
		uint32_t index;
		for (index=0u; index<sizeof(parts)/sizeof(parts[0]); index++)
		{
			(void)snprintf(build,sizeof(build),"%s%s",root,parts[index]);
			CHECK(mkdir(build,0700) == 0);
		}
	}
	CHECK(mkdir(outside,0700) == 0);
	CHECK(symlink(outside,link_path) == 0);
	(void)snprintf(store_shard,sizeof(store_shard),"%s/3a",store);
	(void)snprintf(busy_shard,sizeof(busy_shard),"%s/c4",busy);
	CHECK(mkdir(store_shard,0700) == 0 && mkdir(busy_shard,0700) == 0);
	SnapPurgeFile(store_shard,"aa-cc-64.kvs",old);
	SnapPurgeFile(store_shard,"aa-cc-128.kvs",fresh);
	SnapPurgeFile(busy_shard,"cc-ee-64.kvs",old);
	SnapPurgeFile(store,"aa-bb-64.kvs",old);
	SnapPurgeFile(store,"aa-bb-12.kvst",old);
	SnapPurgeFile(store,"aa-bb-128.kvs",fresh);
	SnapPurgeFile(store,".kvs-writing-7",0u);
	SnapPurgeFile(store,"notes.txt",old);
	SnapPurgeFile(nested,"legacy-name.kvs",old);
	SnapPurgeFile(busy,"cc-dd-64.kvs",old);
	SnapPurgeFile(outside,"ee-ff-64.kvs",old);
	(void)snprintf(lock_path,sizeof(lock_path),"%s/%s",busy,SPARK_KV_SNAPSHOT_LOCK_NAME);
	lock = open(lock_path,O_RDWR | O_CREAT,0600);
	CHECK(lock >= 0 && flock(lock,LOCK_EX | LOCK_NB) == 0);
	CHECK(SparkKvSnapshotPurgeTree(root,(uint64_t)time(0),&result) == SPARK_STATUS_OK);
	CHECK(result.stores == 2u && result.busy_stores == 1u && result.deleted_files == 5u && result.deleted_bytes == 30u && result.failed_unlinks == 0u);
	CHECK(SnapPurgeExists(store_shard,"aa-cc-64.kvs") == 0u && SnapPurgeExists(store_shard,"aa-cc-128.kvs") == 1u && SnapPurgeExists(busy_shard,"cc-ee-64.kvs") == 1u);
	CHECK(SnapPurgeExists(store,"aa-bb-64.kvs") == 0u && SnapPurgeExists(store,"aa-bb-12.kvst") == 0u && SnapPurgeExists(store,".kvs-writing-7") == 0u);
	CHECK(SnapPurgeExists(store,"aa-bb-128.kvs") == 1u && SnapPurgeExists(store,"notes.txt") == 1u);
	CHECK(SnapPurgeExists(nested,"legacy-name.kvs") == 0u);
	CHECK(SnapPurgeExists(busy,"cc-dd-64.kvs") == 1u);
	CHECK(SnapPurgeExists(outside,"ee-ff-64.kvs") == 1u);
	(void)close(lock);
	CHECK(SparkKvSnapshotPurgeDirectory(busy,(uint64_t)time(0),&result) == SPARK_STATUS_OK && SnapPurgeExists(busy,"cc-dd-64.kvs") == 0u && SnapPurgeExists(busy_shard,"cc-ee-64.kvs") == 0u);
	CHECK(SparkKvSnapshotPurgeDirectory(store,(uint64_t)time(0) + 600u,&result) == SPARK_STATUS_OK && SnapPurgeExists(store,"aa-bb-128.kvs") == 0u);
	CHECK(SparkKvSnapshotPurgeTree("relative",(uint64_t)time(0),&result) == SPARK_STATUS_INVALID_ARGUMENT);
	CHECK(SparkKvSnapshotIsCacheFileName("x.kvs") == 1u && SparkKvSnapshotIsCacheFileName("x.kvst") == 1u && SparkKvSnapshotIsCacheFileName(".kvs-writing-1") == 1u &&
		SparkKvSnapshotIsCacheFileName(".kvs-lock") == 0u && SparkKvSnapshotIsCacheFileName("x.kvs.txt") == 0u);
	CHECK(SparkKvSnapshotIsShardName("00") == 1u && SparkKvSnapshotIsShardName("f9") == 1u && SparkKvSnapshotIsShardName("F9") == 0u && SparkKvSnapshotIsShardName("0") == 0u &&
		SparkKvSnapshotIsShardName("abc") == 0u && SparkKvSnapshotIsShardName("g0") == 0u);
	(void)unlink(link_path);
}

int main(void)
{
	char directory[] = "/tmp/sparkpipe-kv-snapshot-XXXXXX";
	if ( mkdtemp(directory) == 0 )
		return(2);
	SnapTestStoreFormat(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestFailedStoreRecovers(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestSyncWriteSeesQueuedWrite(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionAndQueue(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestTreeEviction(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestScale(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestRetention(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestChainExpiry(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestSaveDefersWhenQueueIsFull(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestPageCacheRoundTrip(directory);
	SnapTestReleaseSavesAndPrepareRestores(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestFailedWriteResaves(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestQueueFullSkips(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionDemotes(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionQueuesDemotion(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestEvictionPrefersLowPriority(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestResetKeepsPublishedSaves(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestReadOutcomes(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	SnapTestMarkAllOrder(directory);
	SnapTestSpareRetention(directory);
	SnapTestStatesSpillBeforeDemotion(directory);
	SnapRemoveAll(directory);
	if ( mkdtemp(strcpy(directory,"/tmp/sparkpipe-kv-snapshot-XXXXXX")) == 0 )
		return(2);
	{
		char root[256];
		(void)snprintf(root,sizeof(root),"%s/home",directory);
		SnapTestPurge(root);
	}
	SnapRemoveAll(directory);
	printf("test_kv_snapshot: %u checks, %u failures\n",snap_checks,snap_failures);
	return(snap_failures == 0u ? 0 : 1);
}
