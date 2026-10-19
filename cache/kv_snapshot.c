#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_error_site.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SPARK_KV_SNAPSHOT_NO_ENTRY UINT32_MAX
#define SPARK_KV_SNAPSHOT_SPARE_SHARE 4u
#define SPARK_KV_SNAPSHOT_CHUNK_SHIFT 14u
#define SPARK_KV_SNAPSHOT_CHUNK_ENTRIES (UINT32_C(1) << SPARK_KV_SNAPSHOT_CHUNK_SHIFT)
#define SPARK_KV_SNAPSHOT_MAX_ENTRIES (UINT32_MAX - SPARK_KV_SNAPSHOT_CHUNK_ENTRIES)
#define SPARK_KV_SNAPSHOT_FIRST_BUCKETS 1024u
#define SPARK_KV_SNAPSHOT_MAX_BUCKETS (UINT32_C(1) << 31u)
#define SPARK_KV_SNAPSHOT_MAX_LAYOUTS 65536u
#define SPARK_KV_SNAPSHOT_ENTRY_LIVE 1u
#define SPARK_KV_SNAPSHOT_ENTRY_PHANTOM 2u
#define SPARK_KV_SNAPSHOT_ENTRY_STATE 4u
#define SPARK_KV_SNAPSHOT_ENTRY_IN_LRU 8u
#define SPARK_KV_SNAPSHOT_ENTRY_KIND (SPARK_KV_SNAPSHOT_ENTRY_LIVE | SPARK_KV_SNAPSHOT_ENTRY_PHANTOM | SPARK_KV_SNAPSHOT_ENTRY_STATE)

typedef struct SparkKvSnapshotIndexEntry
{
	uint8_t identity_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t token_count;
	uint32_t parent;
	uint32_t first_child;
	uint32_t next_sibling;
	uint32_t hash_next;
	uint32_t lru_prev;
	uint32_t lru_next;
	uint32_t units;
	uint16_t layout;
	uint16_t pins;
	uint32_t flags;
} SparkKvSnapshotIndexEntry;

typedef struct SparkKvSnapshotLayout
{
	uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint64_t file_count;
	uint64_t references;
} SparkKvSnapshotLayout;

typedef struct SparkKvSnapshotJob
{
	struct SparkKvSnapshotJob *next;
	uint32_t ready;
	uint32_t segment_count;
	uint64_t capacity;
	uint64_t file_bytes;
	SparkKvSnapshotKey key;
	SparkKvSnapshotKey parent;
	SparkKvSnapshotSegment segments[SPARK_KV_SNAPSHOT_MAX_SEGMENTS];
} SparkKvSnapshotJob;

typedef struct SparkKvSnapshotRuntime
{
	pthread_mutex_t mutex;
	pthread_cond_t wake;
	pthread_cond_t idle;
	pthread_t writer;
	uint32_t stopping;
	uint32_t writing;
	int lock_descriptor;
	uint64_t sequence;
	SparkKvSnapshotIndexEntry **chunks;
	uint32_t chunk_count;
	uint32_t chunk_capacity;
	uint32_t entry_count;
	uint32_t entry_high;
	uint32_t free_head;
	uint32_t *buckets;
	uint32_t bucket_count;
	uint32_t lru_head;
	uint32_t lru_tail;
	SparkKvSnapshotLayout *layouts;
	uint32_t layout_count;
	uint32_t layout_capacity;
	SparkKvSnapshotJob *jobs;
	SparkKvSnapshotJob *spare_jobs;
	uint64_t spare_bytes;
} SparkKvSnapshotRuntime;

static uint64_t SparkKvSnapshotNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static uint32_t SparkKvSnapshotStoreIsValid(const SparkKvSnapshotStore *store)
{
	return(store != 0 && store->abi_version == SPARK_KV_SNAPSHOT_ABI_VERSION && store->descriptor_bytes == SPARK_KV_SNAPSHOT_STORE_BYTES && store->directory[0] != '\0' && store->maximum_bytes != 0u && store->runtime != 0 ? 1u : 0u);
}

static SparkKvSnapshotRuntime *SparkKvSnapshotRuntimeOf(SparkKvSnapshotStore *store)
{
	return((SparkKvSnapshotRuntime *)store->runtime);
}

static uint64_t SparkKvSnapshotAlign(uint64_t bytes)
{
	return((bytes + SPARK_KV_SNAPSHOT_ALIGNMENT - 1u) / SPARK_KV_SNAPSHOT_ALIGNMENT * SPARK_KV_SNAPSHOT_ALIGNMENT);
}

static void SparkKvSnapshotHeaderDigest(const SparkKvSnapshotFileHeader *header,uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
	SparkSha256Context context;
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,header,offsetof(SparkKvSnapshotFileHeader,header_sha256));
	SparkSha256Finalize(&context,digest);
}

static void SparkKvSnapshotDigest(const void *data,uint64_t bytes,uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
	SparkSha256Context context;
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,data,(size_t)bytes);
	SparkSha256Finalize(&context,digest);
}

static uint32_t SparkKvSnapshotHasParent(const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent)
{
	return(key->file_kind == SPARK_KV_SNAPSHOT_FILE_PAGES && parent != 0 && parent->token_count != 0u ? 1u : 0u);
}

static uint32_t SparkKvSnapshotParseCount(const char *text,size_t length,uint32_t *value)
{
	uint64_t count = 0u;
	size_t cursor;
	if ( length == 0u )
		return(0u);
	for (cursor=0u; cursor<length; cursor++)
	{
		if ( text[cursor] < '0' || text[cursor] > '9' )
			return(0u);
		count = count * 10u + (uint64_t)(text[cursor] - '0');
		if ( count > UINT32_MAX )
			return(0u);
	}
	*value = (uint32_t)count;
	return(1u);
}

static uint32_t SparkKvSnapshotParseName(const char *name,SparkKvSnapshotKey *key,SparkKvSnapshotKey *parent)
{
	const size_t hex = 2u * SPARK_SHA256_DIGEST_BYTES;
	size_t length = strlen(name),suffix,body,cursor;
	const char *dash;
	memset(key,0,sizeof(*key));
	memset(parent,0,sizeof(*parent));
	if ( length > strlen(SPARK_KV_SNAPSHOT_STATE_SUFFIX) && strcmp(name + length - strlen(SPARK_KV_SNAPSHOT_STATE_SUFFIX),SPARK_KV_SNAPSHOT_STATE_SUFFIX) == 0 )
	{
		key->file_kind = SPARK_KV_SNAPSHOT_FILE_STATE;
		suffix = strlen(SPARK_KV_SNAPSHOT_STATE_SUFFIX);
	}
	else if ( length > strlen(SPARK_KV_SNAPSHOT_SUFFIX) && strcmp(name + length - strlen(SPARK_KV_SNAPSHOT_SUFFIX),SPARK_KV_SNAPSHOT_SUFFIX) == 0 )
	{
		key->file_kind = SPARK_KV_SNAPSHOT_FILE_PAGES;
		suffix = strlen(SPARK_KV_SNAPSHOT_SUFFIX);
	}
	else
		return(0u);
	body = length - suffix;
	if ( body <= 2u * hex + 2u || name[hex] != '-' || name[2u * hex + 1u] != '-' )
		return(0u);
	if ( SparkSha256HexToDigest(name,key->layout_sha256) != SPARK_STATUS_OK || SparkSha256HexToDigest(name + hex + 1u,key->identity_sha256) != SPARK_STATUS_OK )
		return(0u);
	cursor = 2u * hex + 2u;
	dash = memchr(name + cursor,'-',body - cursor);
	if ( key->file_kind == SPARK_KV_SNAPSHOT_FILE_STATE )
		return(dash == 0 && SparkKvSnapshotParseCount(name + cursor,body - cursor,&key->token_count) != 0u && key->token_count != 0u ? 1u : 0u);
	if ( dash == 0 || SparkKvSnapshotParseCount(name + cursor,(size_t)(dash - name) - cursor,&key->token_count) == 0u || key->token_count == 0u )
		return(0u);
	cursor = (size_t)(dash - name) + 1u;
	if ( body < cursor + hex + 2u || name[cursor + hex] != '-' )
		return(0u);
	memcpy(parent->layout_sha256,key->layout_sha256,sizeof(parent->layout_sha256));
	parent->file_kind = SPARK_KV_SNAPSHOT_FILE_PAGES;
	if ( SparkSha256HexToDigest(name + cursor,parent->identity_sha256) != SPARK_STATUS_OK ||
		SparkKvSnapshotParseCount(name + cursor + hex + 1u,body - cursor - hex - 1u,&parent->token_count) == 0u || parent->token_count >= key->token_count )
		return(0u);
	if ( parent->token_count == 0u )
		memset(parent,0,sizeof(*parent));
	return(1u);
}

static uint32_t SparkKvSnapshotIsLegacyName(const char *name)
{
	const size_t hex = 2u * SPARK_SHA256_DIGEST_BYTES,suffix = strlen(SPARK_KV_SNAPSHOT_SUFFIX);
	size_t length = strlen(name);
	return(length > 2u * hex + 2u + suffix && strcmp(name + length - suffix,SPARK_KV_SNAPSHOT_SUFFIX) == 0 && name[hex] == '-' && name[2u * hex + 1u] == '-' &&
		memchr(name + 2u * hex + 2u,'-',length - suffix - 2u * hex - 2u) == 0 ? 1u : 0u);
}

static SparkKvSnapshotIndexEntry *SparkKvSnapshotEntryAt(const SparkKvSnapshotRuntime *runtime,uint32_t index)
{
	return(&runtime->chunks[index >> SPARK_KV_SNAPSHOT_CHUNK_SHIFT][index & (SPARK_KV_SNAPSHOT_CHUNK_ENTRIES - 1u)]);
}

static uint32_t SparkKvSnapshotHashEntry(const uint8_t identity_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t token_count,uint32_t layout,uint32_t state,uint32_t bucket_count)
{
	uint64_t hash = UINT64_C(1469598103934665603);
	uint32_t index;
	for (index=0u; index<8u; index++)
		hash = (hash ^ identity_sha256[index]) * UINT64_C(1099511628211);
	hash = (hash ^ layout) * UINT64_C(1099511628211);
	hash = (hash ^ token_count) * UINT64_C(1099511628211);
	hash = (hash ^ state) * UINT64_C(1099511628211);
	return((uint32_t)(hash ^ (hash >> 32u)) & (bucket_count - 1u));
}

static uint32_t SparkKvSnapshotEntryBucket(const SparkKvSnapshotIndexEntry *entry,uint32_t bucket_count)
{
	return(SparkKvSnapshotHashEntry(entry->identity_sha256,entry->token_count,entry->layout,entry->flags & SPARK_KV_SNAPSHOT_ENTRY_STATE,bucket_count));
}

static uint32_t SparkKvSnapshotKeyFlags(const SparkKvSnapshotKey *key)
{
	return(key->file_kind == SPARK_KV_SNAPSHOT_FILE_STATE ? SPARK_KV_SNAPSHOT_ENTRY_LIVE | SPARK_KV_SNAPSHOT_ENTRY_STATE : SPARK_KV_SNAPSHOT_ENTRY_LIVE);
}

static uint32_t SparkKvSnapshotLayoutFind(const SparkKvSnapshotRuntime *runtime,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES])
{
	uint32_t index;
	for (index=0u; index<runtime->layout_count; index++)
		if ( runtime->layouts[index].references != 0u && memcmp(runtime->layouts[index].layout_sha256,layout_sha256,SPARK_SHA256_DIGEST_BYTES) == 0 )
			return(index);
	return(SPARK_KV_SNAPSHOT_NO_ENTRY);
}

static SparkStatus SparkKvSnapshotLayoutAcquire(SparkKvSnapshotRuntime *runtime,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES],uint16_t *layout_out)
{
	SparkKvSnapshotLayout *grown;
	uint32_t index = SparkKvSnapshotLayoutFind(runtime,layout_sha256),capacity;
	if ( index == SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		for (index=0u; index<runtime->layout_count && runtime->layouts[index].references != 0u; index++)
			;
		if ( index == runtime->layout_capacity )
		{
			if ( index == SPARK_KV_SNAPSHOT_MAX_LAYOUTS )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			capacity = index != 0u ? index * 2u : 8u;
			if ( capacity > SPARK_KV_SNAPSHOT_MAX_LAYOUTS )
				capacity = SPARK_KV_SNAPSHOT_MAX_LAYOUTS;
			grown = (SparkKvSnapshotLayout *)realloc(runtime->layouts,(size_t)capacity * sizeof(*grown));
			if ( grown == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			runtime->layouts = grown;
			runtime->layout_capacity = capacity;
		}
		if ( index == runtime->layout_count )
			runtime->layout_count++;
		memset(&runtime->layouts[index],0,sizeof(runtime->layouts[index]));
		memcpy(runtime->layouts[index].layout_sha256,layout_sha256,SPARK_SHA256_DIGEST_BYTES);
	}
	runtime->layouts[index].references++;
	*layout_out = (uint16_t)index;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkKvSnapshotIndexLookup(const SparkKvSnapshotRuntime *runtime,const uint8_t identity_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t token_count,uint32_t layout,uint32_t flags)
{
	const SparkKvSnapshotIndexEntry *entry;
	uint32_t index;
	for (index=runtime->buckets[SparkKvSnapshotHashEntry(identity_sha256,token_count,layout,flags & SPARK_KV_SNAPSHOT_ENTRY_STATE,runtime->bucket_count)]; index != SPARK_KV_SNAPSHOT_NO_ENTRY; index=entry->hash_next)
	{
		entry = SparkKvSnapshotEntryAt(runtime,index);
		if ( entry->token_count == token_count && entry->layout == layout && (entry->flags & SPARK_KV_SNAPSHOT_ENTRY_KIND) == flags && memcmp(entry->identity_sha256,identity_sha256,SPARK_SHA256_DIGEST_BYTES) == 0 )
			return(index);
	}
	return(SPARK_KV_SNAPSHOT_NO_ENTRY);
}

static uint32_t SparkKvSnapshotIndexFind(const SparkKvSnapshotRuntime *runtime,const SparkKvSnapshotKey *key)
{
	uint32_t layout;
	if ( key->file_kind != SPARK_KV_SNAPSHOT_FILE_PAGES && key->file_kind != SPARK_KV_SNAPSHOT_FILE_STATE )
		return(SPARK_KV_SNAPSHOT_NO_ENTRY);
	layout = SparkKvSnapshotLayoutFind(runtime,key->layout_sha256);
	if ( layout == SPARK_KV_SNAPSHOT_NO_ENTRY )
		return(SPARK_KV_SNAPSHOT_NO_ENTRY);
	return(SparkKvSnapshotIndexLookup(runtime,key->identity_sha256,key->token_count,layout,SparkKvSnapshotKeyFlags(key)));
}

static void SparkKvSnapshotEntryKey(const SparkKvSnapshotRuntime *runtime,uint32_t index,SparkKvSnapshotKey *key)
{
	const SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	memcpy(key->layout_sha256,runtime->layouts[entry->layout].layout_sha256,SPARK_SHA256_DIGEST_BYTES);
	memcpy(key->identity_sha256,entry->identity_sha256,SPARK_SHA256_DIGEST_BYTES);
	key->token_count = entry->token_count;
	key->file_kind = (entry->flags & SPARK_KV_SNAPSHOT_ENTRY_STATE) != 0u ? SPARK_KV_SNAPSHOT_FILE_STATE : SPARK_KV_SNAPSHOT_FILE_PAGES;
}

static void SparkKvSnapshotEntryParentKey(const SparkKvSnapshotRuntime *runtime,uint32_t index,SparkKvSnapshotKey *parent)
{
	uint32_t owner = SparkKvSnapshotEntryAt(runtime,index)->parent;
	memset(parent,0,sizeof(*parent));
	if ( owner != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotEntryKey(runtime,owner,parent);
}

static void SparkKvSnapshotLruUnlink(SparkKvSnapshotRuntime *runtime,uint32_t index)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	if ( (entry->flags & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) == 0u )
		return;
	if ( entry->lru_prev != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotEntryAt(runtime,entry->lru_prev)->lru_next = entry->lru_next;
	else
		runtime->lru_head = entry->lru_next;
	if ( entry->lru_next != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotEntryAt(runtime,entry->lru_next)->lru_prev = entry->lru_prev;
	else
		runtime->lru_tail = entry->lru_prev;
	entry->lru_prev = entry->lru_next = SPARK_KV_SNAPSHOT_NO_ENTRY;
	entry->flags &= ~SPARK_KV_SNAPSHOT_ENTRY_IN_LRU;
}

static void SparkKvSnapshotLruPush(SparkKvSnapshotRuntime *runtime,uint32_t index,uint32_t at_head)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	SparkKvSnapshotLruUnlink(runtime,index);
	if ( at_head != 0u )
	{
		entry->lru_next = runtime->lru_head;
		if ( runtime->lru_head != SPARK_KV_SNAPSHOT_NO_ENTRY )
			SparkKvSnapshotEntryAt(runtime,runtime->lru_head)->lru_prev = index;
		else
			runtime->lru_tail = index;
		runtime->lru_head = index;
	}
	else
	{
		entry->lru_prev = runtime->lru_tail;
		if ( runtime->lru_tail != SPARK_KV_SNAPSHOT_NO_ENTRY )
			SparkKvSnapshotEntryAt(runtime,runtime->lru_tail)->lru_next = index;
		else
			runtime->lru_head = index;
		runtime->lru_tail = index;
	}
	entry->flags |= SPARK_KV_SNAPSHOT_ENTRY_IN_LRU;
}

static uint32_t SparkKvSnapshotIsOrphan(const SparkKvSnapshotRuntime *runtime,uint32_t index)
{
	uint32_t owner = SparkKvSnapshotEntryAt(runtime,index)->parent;
	return(owner != SPARK_KV_SNAPSHOT_NO_ENTRY && (SparkKvSnapshotEntryAt(runtime,owner)->flags & SPARK_KV_SNAPSHOT_ENTRY_PHANTOM) != 0u ? 1u : 0u);
}

static SparkStatus SparkKvSnapshotBucketsFit(SparkKvSnapshotRuntime *runtime,uint32_t entries)
{
	SparkKvSnapshotIndexEntry *entry;
	uint32_t count = runtime->bucket_count != 0u ? runtime->bucket_count : SPARK_KV_SNAPSHOT_FIRST_BUCKETS,*heads,index,bucket;
	while ( count < entries && count < SPARK_KV_SNAPSHOT_MAX_BUCKETS )
		count *= 2u;
	if ( count == runtime->bucket_count )
		return(SPARK_STATUS_OK);
	heads = (uint32_t *)malloc((size_t)count * sizeof(*heads));
	if ( heads == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<count; index++)
		heads[index] = SPARK_KV_SNAPSHOT_NO_ENTRY;
	for (index=0u; index<runtime->entry_high; index++)
	{
		entry = SparkKvSnapshotEntryAt(runtime,index);
		if ( (entry->flags & (SPARK_KV_SNAPSHOT_ENTRY_LIVE | SPARK_KV_SNAPSHOT_ENTRY_PHANTOM)) == 0u )
			continue;
		bucket = SparkKvSnapshotEntryBucket(entry,count);
		entry->hash_next = heads[bucket];
		heads[bucket] = index;
	}
	free(runtime->buckets);
	runtime->buckets = heads;
	runtime->bucket_count = count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotSlotTake(SparkKvSnapshotRuntime *runtime,uint32_t *index_out)
{
	SparkKvSnapshotIndexEntry **grown;
	uint32_t capacity;
	if ( runtime->free_head != SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		*index_out = runtime->free_head;
		runtime->free_head = SparkKvSnapshotEntryAt(runtime,runtime->free_head)->hash_next;
		return(SPARK_STATUS_OK);
	}
	if ( runtime->entry_high >= SPARK_KV_SNAPSHOT_MAX_ENTRIES )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (runtime->entry_high >> SPARK_KV_SNAPSHOT_CHUNK_SHIFT) == runtime->chunk_count )
	{
		if ( runtime->chunk_count == runtime->chunk_capacity )
		{
			capacity = runtime->chunk_capacity != 0u ? runtime->chunk_capacity * 2u : 16u;
			grown = (SparkKvSnapshotIndexEntry **)realloc(runtime->chunks,(size_t)capacity * sizeof(*grown));
			if ( grown == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			runtime->chunks = grown;
			runtime->chunk_capacity = capacity;
		}
		runtime->chunks[runtime->chunk_count] = (SparkKvSnapshotIndexEntry *)calloc(SPARK_KV_SNAPSHOT_CHUNK_ENTRIES,sizeof(SparkKvSnapshotIndexEntry));
		if ( runtime->chunks[runtime->chunk_count] == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		runtime->chunk_count++;
	}
	*index_out = runtime->entry_high++;
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotSlotRelease(SparkKvSnapshotRuntime *runtime,uint32_t index)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	memset(entry,0,sizeof(*entry));
	entry->hash_next = runtime->free_head;
	runtime->free_head = index;
}

static void SparkKvSnapshotEntryInit(SparkKvSnapshotRuntime *runtime,uint32_t index,const uint8_t identity_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t token_count,uint16_t layout,uint32_t flags)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	uint32_t bucket;
	memset(entry,0,sizeof(*entry));
	memcpy(entry->identity_sha256,identity_sha256,SPARK_SHA256_DIGEST_BYTES);
	entry->token_count = token_count;
	entry->layout = layout;
	entry->flags = flags;
	entry->parent = entry->first_child = entry->next_sibling = SPARK_KV_SNAPSHOT_NO_ENTRY;
	entry->lru_prev = entry->lru_next = SPARK_KV_SNAPSHOT_NO_ENTRY;
	bucket = SparkKvSnapshotEntryBucket(entry,runtime->bucket_count);
	entry->hash_next = runtime->buckets[bucket];
	runtime->buckets[bucket] = index;
	runtime->entry_count++;
}

static void SparkKvSnapshotEntryFree(SparkKvSnapshotRuntime *runtime,uint32_t index)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	uint32_t *link;
	for (link=&runtime->buckets[SparkKvSnapshotEntryBucket(entry,runtime->bucket_count)]; *link != SPARK_KV_SNAPSHOT_NO_ENTRY; link=&SparkKvSnapshotEntryAt(runtime,*link)->hash_next)
		if ( *link == index )
		{
			*link = entry->hash_next;
			break;
		}
	runtime->layouts[entry->layout].references--;
	runtime->entry_count--;
	SparkKvSnapshotSlotRelease(runtime,index);
}

static void SparkKvSnapshotChildAttach(SparkKvSnapshotRuntime *runtime,uint32_t owner,uint32_t child)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,child),*parent = SparkKvSnapshotEntryAt(runtime,owner);
	entry->parent = owner;
	entry->next_sibling = parent->first_child;
	parent->first_child = child;
}

static void SparkKvSnapshotChildDetach(SparkKvSnapshotRuntime *runtime,uint32_t child)
{
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,child);
	uint32_t *link;
	for (link=&SparkKvSnapshotEntryAt(runtime,entry->parent)->first_child; *link != SPARK_KV_SNAPSHOT_NO_ENTRY; link=&SparkKvSnapshotEntryAt(runtime,*link)->next_sibling)
		if ( *link == child )
		{
			*link = entry->next_sibling;
			break;
		}
	entry->parent = entry->next_sibling = SPARK_KV_SNAPSHOT_NO_ENTRY;
}

static SparkStatus SparkKvSnapshotIndexInsert(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,uint64_t bytes,uint32_t push,uint32_t *index_out)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotIndexEntry *entry;
	uint64_t units = bytes / SPARK_KV_SNAPSHOT_ALIGNMENT + (bytes % SPARK_KV_SNAPSHOT_ALIGNMENT != 0u ? 1u : 0u);
	uint32_t parented = SparkKvSnapshotHasParent(key,parent),flags = SparkKvSnapshotKeyFlags(key),index = SPARK_KV_SNAPSHOT_NO_ENTRY,owner = SPARK_KV_SNAPSHOT_NO_ENTRY,fresh = 0u,child;
	uint16_t layout;
	SparkStatus status;
	if ( units > UINT32_MAX || (key->file_kind != SPARK_KV_SNAPSHOT_FILE_PAGES && key->file_kind != SPARK_KV_SNAPSHOT_FILE_STATE) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( runtime->entry_count > SPARK_KV_SNAPSHOT_MAX_ENTRIES - 2u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkKvSnapshotBucketsFit(runtime,runtime->entry_count + 2u);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotLayoutAcquire(runtime,key->layout_sha256,&layout);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( flags == SPARK_KV_SNAPSHOT_ENTRY_LIVE )
		index = SparkKvSnapshotIndexLookup(runtime,key->identity_sha256,key->token_count,layout,SPARK_KV_SNAPSHOT_ENTRY_PHANTOM);
	if ( parented != 0u )
	{
		owner = SparkKvSnapshotIndexLookup(runtime,parent->identity_sha256,parent->token_count,layout,SPARK_KV_SNAPSHOT_ENTRY_LIVE);
		if ( owner == SPARK_KV_SNAPSHOT_NO_ENTRY )
			owner = SparkKvSnapshotIndexLookup(runtime,parent->identity_sha256,parent->token_count,layout,SPARK_KV_SNAPSHOT_ENTRY_PHANTOM);
		if ( owner == SPARK_KV_SNAPSHOT_NO_ENTRY )
		{
			status = SparkKvSnapshotSlotTake(runtime,&owner);
			fresh = status == SPARK_STATUS_OK ? 1u : 0u;
		}
	}
	if ( status == SPARK_STATUS_OK && index == SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		status = SparkKvSnapshotSlotTake(runtime,&index);
		if ( status != SPARK_STATUS_OK && fresh != 0u )
			SparkKvSnapshotSlotRelease(runtime,owner);
		if ( status == SPARK_STATUS_OK )
			SparkKvSnapshotEntryInit(runtime,index,key->identity_sha256,key->token_count,layout,flags);
	}
	else if ( status == SPARK_STATUS_OK )
	{
		entry = SparkKvSnapshotEntryAt(runtime,index);
		entry->flags = SPARK_KV_SNAPSHOT_ENTRY_LIVE;
		runtime->layouts[layout].references--;
		for (child=entry->first_child; child != SPARK_KV_SNAPSHOT_NO_ENTRY; child=SparkKvSnapshotEntryAt(runtime,child)->next_sibling)
			store->orphan_count--;
	}
	if ( status != SPARK_STATUS_OK )
	{
		runtime->layouts[layout].references--;
		SPARK_RETURN(status);
	}
	if ( fresh != 0u )
	{
		SparkKvSnapshotEntryInit(runtime,owner,parent->identity_sha256,parent->token_count,layout,SPARK_KV_SNAPSHOT_ENTRY_PHANTOM);
		runtime->layouts[layout].references++;
	}
	entry = SparkKvSnapshotEntryAt(runtime,index);
	entry->units = (uint32_t)units;
	runtime->layouts[layout].file_count++;
	store->used_bytes += units * SPARK_KV_SNAPSHOT_ALIGNMENT;
	store->file_count++;
	if ( owner != SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		SparkKvSnapshotChildAttach(runtime,owner,index);
		if ( (SparkKvSnapshotEntryAt(runtime,owner)->flags & SPARK_KV_SNAPSHOT_ENTRY_PHANTOM) != 0u )
			store->orphan_count++;
		else
			SparkKvSnapshotLruUnlink(runtime,owner);
	}
	if ( push != 0u && ((flags & SPARK_KV_SNAPSHOT_ENTRY_STATE) != 0u || entry->first_child == SPARK_KV_SNAPSHOT_NO_ENTRY) )
		SparkKvSnapshotLruPush(runtime,index,SparkKvSnapshotIsOrphan(runtime,index));
	if ( index_out != 0 )
		*index_out = index;
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotIndexRemoveAt(SparkKvSnapshotStore *store,uint32_t index)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotIndexEntry *entry = SparkKvSnapshotEntryAt(runtime,index);
	uint32_t owner = entry->parent,child;
	if ( owner != SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		SparkKvSnapshotChildDetach(runtime,index);
		if ( (SparkKvSnapshotEntryAt(runtime,owner)->flags & SPARK_KV_SNAPSHOT_ENTRY_PHANTOM) != 0u )
		{
			store->orphan_count--;
			if ( SparkKvSnapshotEntryAt(runtime,owner)->first_child == SPARK_KV_SNAPSHOT_NO_ENTRY )
				SparkKvSnapshotEntryFree(runtime,owner);
		}
		else if ( SparkKvSnapshotEntryAt(runtime,owner)->first_child == SPARK_KV_SNAPSHOT_NO_ENTRY )
			SparkKvSnapshotLruPush(runtime,owner,1u);
	}
	SparkKvSnapshotLruUnlink(runtime,index);
	store->used_bytes -= (uint64_t)entry->units * SPARK_KV_SNAPSHOT_ALIGNMENT;
	store->file_count--;
	runtime->layouts[entry->layout].file_count--;
	if ( entry->first_child == SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		SparkKvSnapshotEntryFree(runtime,index);
		return;
	}
	entry->flags = SPARK_KV_SNAPSHOT_ENTRY_PHANTOM;
	entry->units = 0u;
	entry->pins = 0u;
	for (child=entry->first_child; child != SPARK_KV_SNAPSHOT_NO_ENTRY; child=SparkKvSnapshotEntryAt(runtime,child)->next_sibling)
	{
		store->orphan_count++;
		if ( SparkKvSnapshotEntryAt(runtime,child)->first_child == SPARK_KV_SNAPSHOT_NO_ENTRY )
			SparkKvSnapshotLruPush(runtime,child,1u);
	}
}

static void SparkKvSnapshotIndexForget(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	uint32_t index = SparkKvSnapshotIndexFind(SparkKvSnapshotRuntimeOf(store),key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotIndexRemoveAt(store,index);
}

static void SparkKvSnapshotIndexTouch(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	uint32_t index = SparkKvSnapshotIndexFind(runtime,key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY && (SparkKvSnapshotEntryAt(runtime,index)->flags & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u && SparkKvSnapshotIsOrphan(runtime,index) == 0u )
		SparkKvSnapshotLruPush(runtime,index,0u);
}

static int SparkKvSnapshotCompareOrder(const void *left,const void *right)
{
	uint64_t a = *(const uint64_t *)left,b = *(const uint64_t *)right;
	return(a < b ? -1 : (a > b ? 1 : 0));
}

SparkStatus SparkKvSnapshotPath(const SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,char *path,uint32_t path_capacity)
{
	char layout[SPARK_SHA256_HEX_BYTES],identity[SPARK_SHA256_HEX_BYTES],parent_identity[SPARK_SHA256_HEX_BYTES];
	SparkKvSnapshotKey root;
	int written;
	if ( store == 0 || store->directory[0] == '\0' || key == 0 || path == 0 || key->token_count == 0u ||
		(key->file_kind != SPARK_KV_SNAPSHOT_FILE_PAGES && key->file_kind != SPARK_KV_SNAPSHOT_FILE_STATE) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkSha256DigestToHex(key->layout_sha256,layout);
	SparkSha256DigestToHex(key->identity_sha256,identity);
	if ( key->file_kind == SPARK_KV_SNAPSHOT_FILE_STATE )
		written = snprintf(path,path_capacity,"%s/%s-%s-%u%s",store->directory,layout,identity,key->token_count,SPARK_KV_SNAPSHOT_STATE_SUFFIX);
	else
	{
		memset(&root,0,sizeof(root));
		if ( SparkKvSnapshotHasParent(key,parent) == 0u )
			parent = &root;
		else if ( parent->token_count >= key->token_count || parent->file_kind != SPARK_KV_SNAPSHOT_FILE_PAGES || memcmp(parent->layout_sha256,key->layout_sha256,sizeof(parent->layout_sha256)) != 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		SparkSha256DigestToHex(parent->identity_sha256,parent_identity);
		written = snprintf(path,path_capacity,"%s/%s-%s-%u-%s-%u%s",store->directory,layout,identity,key->token_count,parent_identity,parent->token_count,SPARK_KV_SNAPSHOT_SUFFIX);
	}
	if ( written < 0 || (uint32_t)written >= path_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotEntryPath(const SparkKvSnapshotStore *store,uint32_t index,char *path,uint32_t path_capacity)
{
	const SparkKvSnapshotRuntime *runtime = (const SparkKvSnapshotRuntime *)store->runtime;
	SparkKvSnapshotKey key,parent;
	SparkKvSnapshotEntryKey(runtime,index,&key);
	SparkKvSnapshotEntryParentKey(runtime,index,&parent);
	return(SparkKvSnapshotPath(store,&key,&parent,path,path_capacity));
}

static void SparkKvSnapshotUnlinkEntryLocked(SparkKvSnapshotStore *store,uint32_t index)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( SparkKvSnapshotEntryPath(store,index,path,sizeof(path)) == SPARK_STATUS_OK )
		(void)unlink(path);
	SparkKvSnapshotIndexRemoveAt(store,index);
}

static SparkStatus SparkKvSnapshotEvictLocked(SparkKvSnapshotStore *store,uint64_t incoming_bytes)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	uint32_t victim;
	while ( store->used_bytes > store->maximum_bytes - incoming_bytes )
	{
		for (victim=runtime->lru_head; victim != SPARK_KV_SNAPSHOT_NO_ENTRY && SparkKvSnapshotEntryAt(runtime,victim)->pins != 0u; victim=SparkKvSnapshotEntryAt(runtime,victim)->lru_next)
			;
		if ( victim == SPARK_KV_SNAPSHOT_NO_ENTRY )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		store->eviction_count++;
		store->evicted_bytes += (uint64_t)SparkKvSnapshotEntryAt(runtime,victim)->units * SPARK_KV_SNAPSHOT_ALIGNMENT;
		SparkKvSnapshotUnlinkEntryLocked(store,victim);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotWriteExact(int descriptor,const void *data,uint64_t bytes,uint64_t offset)
{
	const uint8_t *cursor = (const uint8_t *)data;
	ssize_t written;
	while ( bytes != 0u )
	{
		written = pwrite(descriptor,cursor,(size_t)(bytes > (UINT64_C(1) << 30u) ? (UINT64_C(1) << 30u) : bytes),(off_t)offset);
		if ( written < 0 && errno == EINTR )
			continue;
		if ( written < 0 && errno == ENOSPC )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( written <= 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		cursor += written;
		offset += (uint64_t)written;
		bytes -= (uint64_t)written;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotReadExact(int descriptor,void *data,uint64_t bytes,uint64_t offset)
{
	uint8_t *cursor = (uint8_t *)data;
	ssize_t received;
	while ( bytes != 0u )
	{
		received = pread(descriptor,cursor,(size_t)(bytes > (UINT64_C(1) << 30u) ? (UINT64_C(1) << 30u) : bytes),(off_t)offset);
		if ( received < 0 && errno == EINTR )
			continue;
		if ( received == 0 )
			return(SPARK_STATUS_HASH_MISMATCH);
		if ( received < 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		cursor += received;
		offset += (uint64_t)received;
		bytes -= (uint64_t)received;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotSyncDirectory(const SparkKvSnapshotStore *store)
{
	int descriptor = open(store->directory,O_RDONLY | O_CLOEXEC);
	SparkStatus status = SPARK_STATUS_OK;
	if ( descriptor < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( fsync(descriptor) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	(void)close(descriptor);
	SPARK_RETURN(status);
}

static SparkStatus SparkKvSnapshotFileBytes(const uint64_t *bytes,uint32_t segment_count,uint64_t *file_bytes)
{
	uint64_t cursor = SparkKvSnapshotAlign(sizeof(SparkKvSnapshotFileHeader));
	uint32_t index;
	for (index=0u; index<segment_count; index++)
	{
		if ( bytes[index] > UINT64_MAX / 2u - cursor )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		cursor = SparkKvSnapshotAlign(cursor + bytes[index]);
	}
	*file_bytes = cursor;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotBuildHeader(const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,const SparkKvSnapshotSegment *segments,uint32_t segment_count,SparkKvSnapshotFileHeader *header)
{
	uint64_t cursor;
	uint32_t index;
	memset(header,0,sizeof(*header));
	header->magic = SPARK_KV_SNAPSHOT_MAGIC;
	header->format_version = SPARK_KV_SNAPSHOT_FORMAT_VERSION;
	header->header_bytes = sizeof(*header);
	header->key = *key;
	if ( SparkKvSnapshotHasParent(key,parent) != 0u )
		header->parent = *parent;
	header->segment_count = segment_count;
	cursor = SparkKvSnapshotAlign(sizeof(*header));
	for (index=0u; index<segment_count; index++)
	{
		if ( segments[index].kind == 0u || segments[index].reserved0 != 0u || (segments[index].bytes != 0u && segments[index].data == 0) || segments[index].bytes > UINT64_MAX / 2u - cursor )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		header->segments[index].kind = segments[index].kind;
		header->segments[index].offset = cursor;
		header->segments[index].bytes = segments[index].bytes;
		SparkKvSnapshotDigest(segments[index].data,segments[index].bytes,header->segments[index].sha256);
		cursor = SparkKvSnapshotAlign(cursor + segments[index].bytes);
	}
	header->file_bytes = cursor;
	SparkKvSnapshotHeaderDigest(header,header->header_sha256);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotWriteFile(SparkKvSnapshotStore *store,const char *path,const SparkKvSnapshotFileHeader *header,const SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	char temporary[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	int descriptor;
	if ( (size_t)snprintf(temporary,sizeof(temporary),"%s/%s%ld-%llu",store->directory,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX,(long)getpid(),(unsigned long long)__atomic_add_fetch(&SparkKvSnapshotRuntimeOf(store)->sequence,1u,__ATOMIC_RELAXED)) >= sizeof(temporary) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	descriptor = open(temporary,O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0600);
	if ( descriptor < 0 )
		SPARK_FAIL(errno == ENOSPC ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_IO_ERROR);
	for (index=0u; status == SPARK_STATUS_OK && index<segment_count; index++)
		status = SparkKvSnapshotWriteExact(descriptor,segments[index].data,segments[index].bytes,header->segments[index].offset);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotWriteExact(descriptor,header,sizeof(*header),0u);
	if ( status == SPARK_STATUS_OK && ftruncate(descriptor,(off_t)header->file_bytes) != 0 )
		status = errno == ENOSPC ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && fsync(descriptor) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( close(descriptor) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && rename(temporary,path) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		(void)unlink(temporary);
		SPARK_RETURN(status);
	}
	return(SparkKvSnapshotSyncDirectory(store));
}

static SparkStatus SparkKvSnapshotFailureGateLocked(const SparkKvSnapshotStore *store)
{
	if ( store->failed_status != SPARK_STATUS_OK && SparkKvSnapshotNowNs() - store->failed_at_ns < store->failure_retry_ns )
		return(store->failed_status);
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotPinLocked(SparkKvSnapshotRuntime *runtime,const SparkKvSnapshotKey *key,int delta)
{
	uint32_t index = SparkKvSnapshotIndexFind(runtime,key);
	SparkKvSnapshotIndexEntry *entry;
	if ( index == SPARK_KV_SNAPSHOT_NO_ENTRY )
		return;
	entry = SparkKvSnapshotEntryAt(runtime,index);
	if ( delta > 0 || entry->pins != 0u )
		entry->pins = (uint16_t)(entry->pins + delta);
}

static SparkStatus SparkKvSnapshotCommitFile(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,const SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotFileHeader header;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t start_ns = SparkKvSnapshotNowNs();
	uint32_t parented = SparkKvSnapshotHasParent(key,parent);
	SparkStatus status;
	status = SparkKvSnapshotPath(store,key,parent,path,sizeof(path));
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotBuildHeader(key,parent,segments,segment_count,&header);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pthread_mutex_lock(&runtime->mutex);
	SparkKvSnapshotIndexForget(store,key);
	if ( parented != 0u )
		SparkKvSnapshotPinLocked(runtime,parent,1);
	status = SparkKvSnapshotEvictLocked(store,header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		store->used_bytes += header.file_bytes;
	else if ( parented != 0u )
		SparkKvSnapshotPinLocked(runtime,parent,-1);
	pthread_mutex_unlock(&runtime->mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvSnapshotWriteFile(store,path,&header,segments,segment_count);
	pthread_mutex_lock(&runtime->mutex);
	store->used_bytes -= header.file_bytes;
	if ( parented != 0u )
		SparkKvSnapshotPinLocked(runtime,parent,-1);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotIndexInsert(store,key,parent,header.file_bytes,1u,0);
	if ( status == SPARK_STATUS_OK )
	{
		store->write_count++;
		store->write_bytes += header.file_bytes;
		store->write_ns += SparkKvSnapshotNowNs() - start_ns;
		if ( store->failed_status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"kv snapshot store %s writes again after status %u\n",store->directory,(unsigned)store->failed_status);
			store->failed_status = SPARK_STATUS_OK;
			store->recovered_count++;
		}
	}
	else
	{
		store->write_failure_count++;
		if ( status == SPARK_STATUS_IO_ERROR || status == SPARK_STATUS_CAPACITY_EXCEEDED )
		{
			store->failed_status = status;
			store->failed_at_ns = SparkKvSnapshotNowNs();
			fprintf(stderr,"kv snapshot store %s write failed status=%u; writes are refused for %llu ms, then one probes the store\n",
				store->directory,(unsigned)status,(unsigned long long)(store->failure_retry_ns / 1000000u));
		}
	}
	pthread_mutex_unlock(&runtime->mutex);
	SPARK_RETURN(status);
}

static SparkKvSnapshotJob *SparkKvSnapshotNextReadyJob(SparkKvSnapshotRuntime *runtime)
{
	SparkKvSnapshotJob *job;
	for (job=runtime->jobs; job != 0; job=job->next)
		if ( job->ready != 0u )
			return(job);
	return(0);
}

static void SparkKvSnapshotUnlinkJob(SparkKvSnapshotStore *store,SparkKvSnapshotJob *target)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotJob **link;
	for (link=&runtime->jobs; *link != 0; link=&(*link)->next)
		if ( *link == target )
		{
			*link = target->next;
			store->queued_count--;
			store->queued_bytes -= target->file_bytes;
			return;
		}
}

static void SparkKvSnapshotRecycleJob(SparkKvSnapshotStore *store,SparkKvSnapshotJob *job)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	if ( runtime->spare_bytes > store->spare_maximum_bytes || job->capacity > store->spare_maximum_bytes - runtime->spare_bytes )
	{
		free(job);
		return;
	}
	job->next = runtime->spare_jobs;
	runtime->spare_jobs = job;
	runtime->spare_bytes += job->capacity;
	store->spare_bytes = runtime->spare_bytes;
}

static SparkKvSnapshotJob *SparkKvSnapshotTakeSpareJob(SparkKvSnapshotStore *store,SparkKvSnapshotRuntime *runtime,uint64_t payload)
{
	SparkKvSnapshotJob **link,**best = 0,*job;
	for (link=&runtime->spare_jobs; *link != 0; link=&(*link)->next)
		if ( (*link)->capacity >= payload && (best == 0 || (*link)->capacity < (*best)->capacity) )
			best = link;
	if ( best == 0 )
		return(0);
	job = *best;
	*best = job->next;
	runtime->spare_bytes -= job->capacity;
	store->spare_bytes = runtime->spare_bytes;
	return(job);
}

static void *SparkKvSnapshotWriterMain(void *context)
{
	SparkKvSnapshotStore *store = (SparkKvSnapshotStore *)context;
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotJob *job;
	SparkStatus status;
	pthread_mutex_lock(&runtime->mutex);
	for (;;)
	{
		job = SparkKvSnapshotNextReadyJob(runtime);
		if ( job == 0 )
		{
			runtime->writing = 0u;
			pthread_cond_broadcast(&runtime->idle);
			if ( runtime->stopping != 0u )
				break;
			pthread_cond_wait(&runtime->wake,&runtime->mutex);
			continue;
		}
		runtime->writing = 1u;
		pthread_mutex_unlock(&runtime->mutex);
		status = SparkKvSnapshotCommitFile(store,&job->key,&job->parent,job->segments,job->segment_count);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"KV-SNAPSHOT write status=%d tokens=%u bytes=%llu\n",(int)status,job->key.token_count,(unsigned long long)job->file_bytes);
		pthread_mutex_lock(&runtime->mutex);
		SparkKvSnapshotUnlinkJob(store,job);
		SparkKvSnapshotRecycleJob(store,job);
	}
	pthread_mutex_unlock(&runtime->mutex);
	return(0);
}

static SparkStatus SparkKvSnapshotScan(SparkKvSnapshotStore *store)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SparkKvSnapshotKey key,parent;
	SparkKvSnapshotIndexEntry *entry;
	struct dirent *item;
	struct stat info;
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t *order = 0,*grown,seconds;
	size_t count = 0u,capacity = 0u,cursor;
	uint32_t index;
	DIR *listing = opendir(store->directory);
	if ( listing == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	while ( status == SPARK_STATUS_OK && (item = readdir(listing)) != 0 )
	{
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",store->directory,item->d_name) >= sizeof(path) )
			continue;
		if ( strncmp(item->d_name,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX,strlen(SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX)) == 0 )
		{
			if ( unlink(path) == 0 )
				store->removed_temporary_count++;
			continue;
		}
		if ( SparkKvSnapshotParseName(item->d_name,&key,&parent) == 0u )
		{
			if ( SparkKvSnapshotIsLegacyName(item->d_name) != 0u && unlink(path) == 0 )
				store->removed_legacy_count++;
			continue;
		}
		if ( stat(path,&info) != 0 || !S_ISREG(info.st_mode) )
			continue;
		if ( count == capacity )
		{
			capacity = capacity != 0u ? capacity * 2u : 1024u;
			grown = (uint64_t *)realloc(order,capacity * sizeof(*order));
			if ( grown == 0 )
			{
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
				break;
			}
			order = grown;
		}
		status = SparkKvSnapshotIndexInsert(store,&key,&parent,(uint64_t)info.st_size,0u,&index);
		seconds = info.st_mtime > 0 ? (uint64_t)info.st_mtime : 0u;
		if ( status == SPARK_STATUS_OK )
			order[count++] = (seconds < UINT32_MAX ? seconds : UINT32_MAX) << 32u | index;
	}
	(void)closedir(listing);
	if ( status != SPARK_STATUS_OK )
	{
		free(order);
		SPARK_RETURN(status);
	}
	if ( count != 0u )
		qsort(order,count,sizeof(order[0]),SparkKvSnapshotCompareOrder);
	for (cursor=0u; cursor<count; cursor++)
	{
		index = (uint32_t)order[cursor];
		entry = SparkKvSnapshotEntryAt(runtime,index);
		if ( (entry->flags & SPARK_KV_SNAPSHOT_ENTRY_STATE) != 0u || entry->first_child == SPARK_KV_SNAPSHOT_NO_ENTRY )
			SparkKvSnapshotLruPush(runtime,index,0u);
	}
	for (cursor=0u; cursor<count; cursor++)
	{
		index = (uint32_t)order[cursor];
		if ( SparkKvSnapshotIsOrphan(runtime,index) != 0u && (SparkKvSnapshotEntryAt(runtime,index)->flags & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u )
			SparkKvSnapshotLruPush(runtime,index,1u);
	}
	free(order);
	return(SparkKvSnapshotEvictLocked(store,0u));
}

static void SparkKvSnapshotRuntimeFree(SparkKvSnapshotRuntime *runtime)
{
	uint32_t index;
	if ( runtime->lock_descriptor >= 0 )
		(void)close(runtime->lock_descriptor);
	for (index=0u; index<runtime->chunk_count; index++)
		free(runtime->chunks[index]);
	free(runtime->chunks);
	free(runtime->buckets);
	free(runtime->layouts);
	free(runtime);
}

SparkStatus SparkKvSnapshotStoreOpen(SparkKvSnapshotStore *store,const char *directory,uint64_t maximum_bytes,uint64_t queue_maximum_bytes)
{
	SparkKvSnapshotRuntime *runtime;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct stat info;
	SparkStatus status;
	if ( store == 0 || directory == 0 || directory[0] != '/' || maximum_bytes == 0u || queue_maximum_bytes == 0u || strlen(directory) >= SPARK_KV_SNAPSHOT_PATH_BYTES / 2u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( stat(directory,&info) != 0 || !S_ISDIR(info.st_mode) )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(store,0,sizeof(*store));
	runtime = (SparkKvSnapshotRuntime *)calloc(1u,sizeof(*runtime));
	if ( runtime == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	runtime->lock_descriptor = -1;
	runtime->free_head = runtime->lru_head = runtime->lru_tail = SPARK_KV_SNAPSHOT_NO_ENTRY;
	(void)snprintf(path,sizeof(path),"%s/%s",directory,SPARK_KV_SNAPSHOT_LOCK_NAME);
	runtime->lock_descriptor = open(path,O_RDWR | O_CREAT | O_CLOEXEC,0600);
	if ( runtime->lock_descriptor < 0 )
	{
		SparkKvSnapshotRuntimeFree(runtime);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	if ( flock(runtime->lock_descriptor,LOCK_EX | LOCK_NB) != 0 )
	{
		SparkKvSnapshotRuntimeFree(runtime);
		SPARK_FAIL(SPARK_STATUS_BUSY);
	}
	store->abi_version = SPARK_KV_SNAPSHOT_ABI_VERSION;
	store->descriptor_bytes = SPARK_KV_SNAPSHOT_STORE_BYTES;
	store->maximum_bytes = maximum_bytes;
	store->queue_maximum_bytes = queue_maximum_bytes;
	store->spare_maximum_bytes = queue_maximum_bytes / SPARK_KV_SNAPSHOT_SPARE_SHARE;
	store->failure_retry_ns = SPARK_KV_SNAPSHOT_FAILURE_RETRY_NS;
	memcpy(store->directory,directory,strlen(directory) + 1u);
	store->runtime = runtime;
	status = SparkKvSnapshotBucketsFit(runtime,1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotScan(store);
	if ( status == SPARK_STATUS_OK && (pthread_mutex_init(&runtime->mutex,0) != 0 || pthread_cond_init(&runtime->wake,0) != 0 || pthread_cond_init(&runtime->idle,0) != 0 || pthread_create(&runtime->writer,0,SparkKvSnapshotWriterMain,store) != 0) )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		SparkKvSnapshotRuntimeFree(runtime);
		memset(store,0,sizeof(*store));
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

void SparkKvSnapshotStoreClose(SparkKvSnapshotStore *store)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotJob *job;
	if ( store == 0 || store->runtime == 0 )
		return;
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	runtime->stopping = 1u;
	pthread_cond_broadcast(&runtime->wake);
	pthread_mutex_unlock(&runtime->mutex);
	(void)pthread_join(runtime->writer,0);
	while ( (job = runtime->jobs) != 0 )
	{
		runtime->jobs = job->next;
		free(job);
	}
	while ( (job = runtime->spare_jobs) != 0 )
	{
		runtime->spare_jobs = job->next;
		free(job);
	}
	pthread_cond_destroy(&runtime->wake);
	pthread_cond_destroy(&runtime->idle);
	pthread_mutex_destroy(&runtime->mutex);
	SparkKvSnapshotRuntimeFree(runtime);
	memset(store,0,sizeof(*store));
}

SparkStatus SparkKvSnapshotStoreSample(SparkKvSnapshotStore *store,SparkKvSnapshotStore *sample)
{
	SparkKvSnapshotRuntime *runtime;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || sample == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	*sample = *store;
	pthread_mutex_unlock(&runtime->mutex);
	sample->runtime = 0;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotPrune(SparkKvSnapshotStore *store,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES])
{
	SparkKvSnapshotRuntime *runtime;
	uint32_t index,layout;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || layout_sha256 == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	layout = SparkKvSnapshotLayoutFind(runtime,layout_sha256);
	if ( store->file_count != (layout != SPARK_KV_SNAPSHOT_NO_ENTRY ? runtime->layouts[layout].file_count : 0u) )
		for (index=0u; index<runtime->entry_high; index++)
		{
			if ( (SparkKvSnapshotEntryAt(runtime,index)->flags & SPARK_KV_SNAPSHOT_ENTRY_LIVE) == 0u || SparkKvSnapshotEntryAt(runtime,index)->layout == layout )
				continue;
			store->pruned_count++;
			SparkKvSnapshotUnlinkEntryLocked(store,index);
		}
	pthread_mutex_unlock(&runtime->mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotCountLayout(SparkKvSnapshotStore *store,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES],uint64_t *matching_files,uint64_t *foreign_files)
{
	SparkKvSnapshotRuntime *runtime;
	uint64_t matching;
	uint32_t layout;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || layout_sha256 == 0 || matching_files == 0 || foreign_files == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	layout = SparkKvSnapshotLayoutFind(runtime,layout_sha256);
	matching = layout != SPARK_KV_SNAPSHOT_NO_ENTRY ? runtime->layouts[layout].file_count : 0u;
	*matching_files = matching;
	*foreign_files = store->file_count - matching;
	pthread_mutex_unlock(&runtime->mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotBinaryDigest(const void *symbol,uint8_t digest[SPARK_SHA256_DIGEST_BYTES],char *path,uint32_t path_capacity)
{
	char hex[SPARK_SHA256_HEX_BYTES];
	Dl_info info;
	size_t length;
	SparkStatus status;
	if ( symbol == 0 || digest == 0 || path == 0 || path_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&info,0,sizeof(info));
	if ( dladdr(symbol,&info) == 0 || info.dli_fname == 0 || info.dli_fname[0] == '\0' )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	length = strlen(info.dli_fname);
	if ( length >= path_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(path,info.dli_fname,length + 1u);
	status = SparkSha256File(path,hex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkSha256HexToDigest(hex,digest));
}

static void SparkKvSnapshotDiscard(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t failures;
	uint32_t index;
	pthread_mutex_lock(&runtime->mutex);
	failures = ++store->checksum_failure_count;
	index = SparkKvSnapshotIndexFind(runtime,key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotUnlinkEntryLocked(store,index);
	else if ( key->file_kind == SPARK_KV_SNAPSHOT_FILE_STATE && SparkKvSnapshotPath(store,key,0,path,sizeof(path)) == SPARK_STATUS_OK )
		(void)unlink(path);
	pthread_mutex_unlock(&runtime->mutex);
	fprintf(stderr,"KV-SNAPSHOT discarded corrupt file tokens=%u checksum_failures=%llu\n",key->token_count,(unsigned long long)failures);
}

static SparkStatus SparkKvSnapshotValidateHeader(const SparkKvSnapshotFileHeader *header,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,uint64_t file_bytes)
{
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	SparkKvSnapshotKey expected;
	uint64_t floor;
	uint32_t index;
	memset(&expected,0,sizeof(expected));
	if ( SparkKvSnapshotHasParent(key,parent) != 0u )
		expected = *parent;
	if ( header->magic != SPARK_KV_SNAPSHOT_MAGIC || header->format_version != SPARK_KV_SNAPSHOT_FORMAT_VERSION || header->header_bytes != sizeof(*header) || header->reserved0 != 0u )
		return(SPARK_STATUS_HASH_MISMATCH);
	SparkKvSnapshotHeaderDigest(header,digest);
	if ( memcmp(digest,header->header_sha256,sizeof(digest)) != 0 || memcmp(&header->key,key,sizeof(*key)) != 0 || memcmp(&header->parent,&expected,sizeof(expected)) != 0 )
		return(SPARK_STATUS_HASH_MISMATCH);
	if ( header->segment_count == 0u || header->segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS || header->file_bytes != file_bytes )
		return(SPARK_STATUS_HASH_MISMATCH);
	floor = SparkKvSnapshotAlign(sizeof(*header));
	for (index=0u; index<header->segment_count; index++)
	{
		const SparkKvSnapshotFileSegment *segment = &header->segments[index];
		if ( segment->kind == 0u || segment->reserved0 != 0u || segment->offset < floor || segment->offset % SPARK_KV_SNAPSHOT_ALIGNMENT != 0u || segment->bytes > file_bytes || segment->offset > file_bytes - segment->bytes )
			return(SPARK_STATUS_HASH_MISMATCH);
		floor = segment->offset + segment->bytes;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotOpenValidated(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header,int *descriptor_out,uint32_t record_miss)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotKey parent;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct stat info;
	SparkStatus status;
	uint32_t index;
	int descriptor;
	memset(&parent,0,sizeof(parent));
	*descriptor_out = -1;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	index = SparkKvSnapshotIndexFind(runtime,key);
	status = index == SPARK_KV_SNAPSHOT_NO_ENTRY ? SPARK_STATUS_NOT_FOUND : SparkKvSnapshotEntryPath(store,index,path,sizeof(path));
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotEntryParentKey(runtime,index,&parent);
	else
		store->miss_count += record_miss;
	pthread_mutex_unlock(&runtime->mutex);
	if ( status == SPARK_STATUS_NOT_FOUND )
		return(status);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	descriptor = open(path,O_RDONLY | O_CLOEXEC);
	if ( descriptor < 0 )
	{
		if ( errno == ENOENT )
		{
			pthread_mutex_lock(&runtime->mutex);
			store->miss_count += record_miss;
			SparkKvSnapshotIndexForget(store,key);
			pthread_mutex_unlock(&runtime->mutex);
			return(SPARK_STATUS_NOT_FOUND);
		}
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = fstat(descriptor,&info) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && (uint64_t)info.st_size < sizeof(*header) )
		status = SPARK_STATUS_HASH_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotReadExact(descriptor,header,sizeof(*header),0u);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotValidateHeader(header,key,&parent,(uint64_t)info.st_size);
	if ( status != SPARK_STATUS_OK )
	{
		(void)close(descriptor);
		if ( status == SPARK_STATUS_HASH_MISMATCH )
			SparkKvSnapshotDiscard(store,key);
		return(status);
	}
	*descriptor_out = descriptor;
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotRecordRead(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,uint64_t bytes,uint64_t start_ns)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	SparkKvSnapshotIndexTouch(store,key);
	store->read_count++;
	store->read_bytes += bytes;
	store->read_ns += SparkKvSnapshotNowNs() - start_ns;
	pthread_mutex_unlock(&runtime->mutex);
}

SparkStatus SparkKvSnapshotStat(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header)
{
	SparkStatus status;
	int descriptor;
	if ( header == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotOpenValidated(store,key,header,&descriptor,1u);
	if ( status == SPARK_STATUS_OK )
		(void)close(descriptor);
	return(status);
}

SparkStatus SparkKvSnapshotParent(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotKey *parent_out)
{
	SparkKvSnapshotRuntime *runtime;
	uint32_t index;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || key == 0 || parent_out == 0 || key->file_kind != SPARK_KV_SNAPSHOT_FILE_PAGES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	index = SparkKvSnapshotIndexFind(runtime,key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotEntryParentKey(runtime,index,parent_out);
	pthread_mutex_unlock(&runtime->mutex);
	return(index != SPARK_KV_SNAPSHOT_NO_ENTRY ? SPARK_STATUS_OK : SPARK_STATUS_NOT_FOUND);
}

uint32_t SparkKvSnapshotContains(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotJob *job;
	uint32_t found;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || key == 0 )
		return(0u);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	found = SparkKvSnapshotIndexFind(runtime,key) != SPARK_KV_SNAPSHOT_NO_ENTRY ? 1u : 0u;
	for (job=runtime->jobs; found == 0u && job != 0; job=job->next)
		found = memcmp(&job->key,key,sizeof(*key)) == 0 ? 1u : 0u;
	pthread_mutex_unlock(&runtime->mutex);
	return(found);
}

SparkStatus SparkKvSnapshotRemove(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime;
	uint32_t index;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || key == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	index = SparkKvSnapshotIndexFind(runtime,key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotUnlinkEntryLocked(store,index);
	pthread_mutex_unlock(&runtime->mutex);
	return(index != SPARK_KV_SNAPSHOT_NO_ENTRY ? SPARK_STATUS_OK : SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkKvSnapshotWrite(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,const SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotJob *job;
	uint64_t bytes[SPARK_KV_SNAPSHOT_MAX_SEGMENTS],file_bytes;
	SparkStatus status;
	uint32_t index;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || segments == 0 || segment_count == 0u || segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPath(store,key,parent,path,sizeof(path));
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (index=0u; index<segment_count; index++)
		bytes[index] = segments[index].bytes;
	status = SparkKvSnapshotFileBytes(bytes,segment_count,&file_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	for (job=runtime->jobs; job != 0 && memcmp(&job->key,key,sizeof(*key)) != 0; job=job->next)
		;
	if ( SparkKvSnapshotFailureGateLocked(store) != SPARK_STATUS_OK )
		status = store->failed_status;
	else if ( job != 0 )
		status = SPARK_STATUS_PENDING;
	else if ( SparkKvSnapshotIndexFind(runtime,key) != SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		store->duplicate_count++;
		SparkKvSnapshotIndexTouch(store,key);
		status = SPARK_STATUS_DUPLICATE;
	}
	else if ( file_bytes > store->maximum_bytes )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	pthread_mutex_unlock(&runtime->mutex);
	if ( status == SPARK_STATUS_DUPLICATE )
		return(SPARK_STATUS_OK);
	if ( status == SPARK_STATUS_PENDING )
		return(status);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkKvSnapshotCommitFile(store,key,parent,segments,segment_count));
}

SparkStatus SparkKvSnapshotWriteBegin(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,const uint32_t *kinds,const uint64_t *bytes,uint32_t segment_count,SparkKvSnapshotWriteTicket *ticket)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotJob *job,**tail;
	uint64_t file_bytes = 0u,payload = 0u;
	uint8_t *cursor;
	SparkStatus status;
	uint32_t index;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( ticket != 0 )
		memset(ticket,0,sizeof(*ticket));
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || ticket == 0 || kinds == 0 || bytes == 0 || segment_count == 0u || segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPath(store,key,parent,path,sizeof(path));
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotFileBytes(bytes,segment_count,&file_bytes);
	for (index=0u; status == SPARK_STATUS_OK && index<segment_count; index++)
	{
		if ( kinds[index] == 0u )
			status = SPARK_STATUS_INVALID_ARGUMENT;
		payload += bytes[index];
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	for (job=runtime->jobs; job != 0 && memcmp(&job->key,key,sizeof(*key)) != 0; job=job->next)
		;
	if ( SparkKvSnapshotFailureGateLocked(store) != SPARK_STATUS_OK )
		status = store->failed_status;
	else if ( job != 0 )
		status = SPARK_STATUS_PENDING;
	else if ( SparkKvSnapshotIndexFind(runtime,key) != SPARK_KV_SNAPSHOT_NO_ENTRY )
	{
		store->duplicate_count++;
		SparkKvSnapshotIndexTouch(store,key);
		status = SPARK_STATUS_DUPLICATE;
	}
	else if ( file_bytes > store->maximum_bytes )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	else if ( store->queued_bytes > store->queue_maximum_bytes || file_bytes > store->queue_maximum_bytes - store->queued_bytes )
	{
		store->queue_full_count++;
		status = SPARK_STATUS_BUSY;
	}
	job = 0;
	if ( status == SPARK_STATUS_OK && (job = SparkKvSnapshotTakeSpareJob(store,runtime,payload)) == 0 && (job = (SparkKvSnapshotJob *)malloc(sizeof(*job) + (size_t)payload)) != 0 )
		job->capacity = payload;
	if ( status == SPARK_STATUS_OK && job == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
	{
		job->ready = 0u;
		memset(job->segments,0,sizeof(job->segments));
		job->key = *key;
		memset(&job->parent,0,sizeof(job->parent));
		if ( SparkKvSnapshotHasParent(key,parent) != 0u )
			job->parent = *parent;
		job->segment_count = segment_count;
		job->file_bytes = file_bytes;
		cursor = (uint8_t *)(job + 1);
		for (index=0u; index<segment_count; index++)
		{
			job->segments[index] = (SparkKvSnapshotSegment){.kind=kinds[index],.bytes=bytes[index],.data=cursor};
			cursor += bytes[index];
		}
		job->next = 0;
		for (tail=&runtime->jobs; *tail != 0; tail=&(*tail)->next)
			;
		*tail = job;
		store->queued_count++;
		store->queued_bytes += file_bytes;
	}
	pthread_mutex_unlock(&runtime->mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	ticket->key = *key;
	ticket->parent = job->parent;
	ticket->segment_count = segment_count;
	ticket->file_bytes = file_bytes;
	memcpy(ticket->segments,job->segments,sizeof(job->segments));
	ticket->job = job;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotWriteCommit(SparkKvSnapshotStore *store,SparkKvSnapshotWriteTicket *ticket)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotJob *job;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || ticket == 0 || ticket->job == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	job = (SparkKvSnapshotJob *)ticket->job;
	pthread_mutex_lock(&runtime->mutex);
	job->ready = 1u;
	pthread_cond_signal(&runtime->wake);
	pthread_mutex_unlock(&runtime->mutex);
	memset(ticket,0,sizeof(*ticket));
	return(SPARK_STATUS_OK);
}

void SparkKvSnapshotWriteCancel(SparkKvSnapshotStore *store,SparkKvSnapshotWriteTicket *ticket)
{
	SparkKvSnapshotRuntime *runtime;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || ticket == 0 || ticket->job == 0 )
		return;
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	SparkKvSnapshotUnlinkJob(store,(SparkKvSnapshotJob *)ticket->job);
	SparkKvSnapshotRecycleJob(store,(SparkKvSnapshotJob *)ticket->job);
	pthread_mutex_unlock(&runtime->mutex);
	memset(ticket,0,sizeof(*ticket));
}

SparkStatus SparkKvSnapshotFlush(SparkKvSnapshotStore *store)
{
	SparkKvSnapshotRuntime *runtime;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	while ( runtime->writing != 0u || SparkKvSnapshotNextReadyJob(runtime) != 0 )
		pthread_cond_wait(&runtime->idle,&runtime->mutex);
	pthread_mutex_unlock(&runtime->mutex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotRead(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	SparkKvSnapshotFileHeader header;
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	uint64_t start_ns = SparkKvSnapshotNowNs(),bytes = 0u;
	SparkStatus status;
	uint32_t index;
	int descriptor;
	if ( segments == 0 || segment_count == 0u || segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotOpenValidated(store,key,&header,&descriptor,1u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( header.segment_count != segment_count )
		status = SPARK_STATUS_VALIDATION_FAILED;
	for (index=0u; status == SPARK_STATUS_OK && index<segment_count; index++)
	{
		if ( segments[index].kind != header.segments[index].kind )
			status = SPARK_STATUS_VALIDATION_FAILED;
		else if ( segments[index].bytes < header.segments[index].bytes || (header.segments[index].bytes != 0u && segments[index].data == 0) )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		else
			status = SparkKvSnapshotReadExact(descriptor,segments[index].data,header.segments[index].bytes,header.segments[index].offset);
		if ( status != SPARK_STATUS_OK )
			break;
		SparkKvSnapshotDigest(segments[index].data,header.segments[index].bytes,digest);
		if ( memcmp(digest,header.segments[index].sha256,sizeof(digest)) != 0 )
		{
			status = SPARK_STATUS_HASH_MISMATCH;
			break;
		}
		segments[index].bytes = header.segments[index].bytes;
		bytes += header.segments[index].bytes;
	}
	(void)close(descriptor);
	if ( status == SPARK_STATUS_HASH_MISMATCH )
		SparkKvSnapshotDiscard(store,key);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkKvSnapshotRecordRead(store,key,bytes,start_ns);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotReadSegment(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,uint32_t segment_index,uint32_t kind,void *data,uint64_t capacity,uint64_t *bytes_out)
{
	SparkKvSnapshotFileHeader header;
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	uint64_t start_ns = SparkKvSnapshotNowNs();
	SparkStatus status;
	int descriptor;
	if ( bytes_out == 0 || segment_index >= SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*bytes_out = 0u;
	status = SparkKvSnapshotOpenValidated(store,key,&header,&descriptor,1u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( segment_index >= header.segment_count )
		status = SPARK_STATUS_NOT_FOUND;
	else if ( header.segments[segment_index].kind != kind )
		status = SPARK_STATUS_VALIDATION_FAILED;
	else if ( capacity < header.segments[segment_index].bytes || (header.segments[segment_index].bytes != 0u && data == 0) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	else
		status = SparkKvSnapshotReadExact(descriptor,data,header.segments[segment_index].bytes,header.segments[segment_index].offset);
	(void)close(descriptor);
	if ( status == SPARK_STATUS_OK )
	{
		SparkKvSnapshotDigest(data,header.segments[segment_index].bytes,digest);
		if ( memcmp(digest,header.segments[segment_index].sha256,sizeof(digest)) != 0 )
			status = SPARK_STATUS_HASH_MISMATCH;
	}
	if ( status == SPARK_STATUS_HASH_MISMATCH )
		SparkKvSnapshotDiscard(store,key);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*bytes_out = header.segments[segment_index].bytes;
	SparkKvSnapshotRecordRead(store,key,header.segments[segment_index].bytes,start_ns);
	return(SPARK_STATUS_OK);
}
