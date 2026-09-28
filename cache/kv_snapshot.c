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

typedef struct SparkKvSnapshotIndexEntry
{
	SparkKvSnapshotKey key;
	uint64_t bytes;
	uint64_t recency;
} SparkKvSnapshotIndexEntry;

typedef struct SparkKvSnapshotJob
{
	struct SparkKvSnapshotJob *next;
	uint32_t ready;
	uint32_t segment_count;
	uint64_t capacity;
	uint64_t file_bytes;
	SparkKvSnapshotKey key;
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
	uint64_t recency;
	uint64_t sequence;
	SparkKvSnapshotIndexEntry *entries;
	uint32_t entry_count;
	uint32_t entry_capacity;
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

static int32_t SparkKvSnapshotHexNibble(char value)
{
	if ( value >= '0' && value <= '9' )
		return(value - '0');
	if ( value >= 'a' && value <= 'f' )
		return(value - 'a' + 10);
	return(-1);
}

static uint32_t SparkKvSnapshotParseHex(const char *text,uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
	uint32_t index;
	int32_t high,low;
	for (index=0u; index<SPARK_SHA256_DIGEST_BYTES; index++)
	{
		high = SparkKvSnapshotHexNibble(text[2u * index]);
		low = SparkKvSnapshotHexNibble(text[2u * index + 1u]);
		if ( high < 0 || low < 0 )
			return(0u);
		digest[index] = (uint8_t)((high << 4) | low);
	}
	return(1u);
}

static uint32_t SparkKvSnapshotParseName(const char *name,SparkKvSnapshotKey *key)
{
	const size_t hex = 2u * SPARK_SHA256_DIGEST_BYTES,suffix = strlen(SPARK_KV_SNAPSHOT_SUFFIX);
	size_t length = strlen(name),cursor;
	uint64_t tokens = 0u;
	memset(key,0,sizeof(*key));
	if ( length <= 2u * hex + 2u + suffix || name[hex] != '-' || name[2u * hex + 1u] != '-' || strcmp(name + length - suffix,SPARK_KV_SNAPSHOT_SUFFIX) != 0 )
		return(0u);
	if ( SparkKvSnapshotParseHex(name,key->layout_sha256) == 0u || SparkKvSnapshotParseHex(name + hex + 1u,key->identity_sha256) == 0u )
		return(0u);
	for (cursor=2u * hex + 2u; cursor<length - suffix; cursor++)
	{
		if ( name[cursor] < '0' || name[cursor] > '9' )
			return(0u);
		tokens = tokens * 10u + (uint64_t)(name[cursor] - '0');
		if ( tokens > UINT32_MAX )
			return(0u);
	}
	key->token_count = (uint32_t)tokens;
	return(tokens != 0u ? 1u : 0u);
}

static uint32_t SparkKvSnapshotIndexFind(const SparkKvSnapshotRuntime *runtime,const SparkKvSnapshotKey *key)
{
	uint32_t index;
	for (index=0u; index<runtime->entry_count; index++)
		if ( memcmp(&runtime->entries[index].key,key,sizeof(*key)) == 0 )
			return(index);
	return(SPARK_KV_SNAPSHOT_NO_ENTRY);
}

static SparkStatus SparkKvSnapshotIndexInsert(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,uint64_t bytes,uint64_t recency)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotIndexEntry *grown;
	uint32_t capacity;
	if ( runtime->entry_count == runtime->entry_capacity )
	{
		capacity = runtime->entry_capacity == 0u ? 256u : runtime->entry_capacity * 2u;
		grown = (SparkKvSnapshotIndexEntry *)realloc(runtime->entries,(size_t)capacity * sizeof(*grown));
		if ( grown == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		runtime->entries = grown;
		runtime->entry_capacity = capacity;
	}
	runtime->entries[runtime->entry_count++] = (SparkKvSnapshotIndexEntry){.key=*key,.bytes=bytes,.recency=recency};
	store->used_bytes += bytes;
	store->file_count++;
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotIndexRemoveAt(SparkKvSnapshotStore *store,uint32_t index)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	store->used_bytes -= runtime->entries[index].bytes;
	store->file_count--;
	runtime->entries[index] = runtime->entries[--runtime->entry_count];
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
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		runtime->entries[index].recency = ++runtime->recency;
}

static int SparkKvSnapshotCompareRecency(const void *left,const void *right)
{
	const SparkKvSnapshotIndexEntry *a = (const SparkKvSnapshotIndexEntry *)left,*b = (const SparkKvSnapshotIndexEntry *)right;
	return(a->recency < b->recency ? -1 : (a->recency > b->recency ? 1 : 0));
}

SparkStatus SparkKvSnapshotPath(const SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,char *path,uint32_t path_capacity)
{
	char layout[SPARK_SHA256_HEX_BYTES],identity[SPARK_SHA256_HEX_BYTES];
	if ( store == 0 || store->directory[0] == '\0' || key == 0 || path == 0 || key->token_count == 0u || key->reserved0 != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkSha256DigestToHex(key->layout_sha256,layout);
	SparkSha256DigestToHex(key->identity_sha256,identity);
	if ( (uint32_t)snprintf(path,path_capacity,"%s/%s-%s-%u%s",store->directory,layout,identity,key->token_count,SPARK_KV_SNAPSHOT_SUFFIX) >= path_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotUnlinkEntryLocked(SparkKvSnapshotStore *store,uint32_t index)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( SparkKvSnapshotPath(store,&SparkKvSnapshotRuntimeOf(store)->entries[index].key,path,sizeof(path)) == SPARK_STATUS_OK )
		(void)unlink(path);
	SparkKvSnapshotIndexRemoveAt(store,index);
}

static SparkStatus SparkKvSnapshotEvictLocked(SparkKvSnapshotStore *store,uint64_t incoming_bytes)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	uint32_t index,victim;
	while ( store->used_bytes > store->maximum_bytes - incoming_bytes )
	{
		if ( runtime->entry_count == 0u )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		victim = 0u;
		for (index=1u; index<runtime->entry_count; index++)
			if ( runtime->entries[index].recency < runtime->entries[victim].recency )
				victim = index;
		store->eviction_count++;
		store->evicted_bytes += runtime->entries[victim].bytes;
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

static SparkStatus SparkKvSnapshotBuildHeader(const SparkKvSnapshotKey *key,const SparkKvSnapshotSegment *segments,uint32_t segment_count,SparkKvSnapshotFileHeader *header)
{
	uint64_t cursor;
	uint32_t index;
	memset(header,0,sizeof(*header));
	header->magic = SPARK_KV_SNAPSHOT_MAGIC;
	header->format_version = SPARK_KV_SNAPSHOT_FORMAT_VERSION;
	header->header_bytes = sizeof(*header);
	header->key = *key;
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

static SparkStatus SparkKvSnapshotCommitFile(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkKvSnapshotFileHeader header;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t start_ns = SparkKvSnapshotNowNs();
	SparkStatus status;
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotBuildHeader(key,segments,segment_count,&header);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pthread_mutex_lock(&runtime->mutex);
	SparkKvSnapshotIndexForget(store,key);
	status = SparkKvSnapshotEvictLocked(store,header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		store->used_bytes += header.file_bytes;
	pthread_mutex_unlock(&runtime->mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvSnapshotWriteFile(store,path,&header,segments,segment_count);
	pthread_mutex_lock(&runtime->mutex);
	store->used_bytes -= header.file_bytes;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotIndexInsert(store,key,header.file_bytes,++runtime->recency);
	if ( status == SPARK_STATUS_OK )
	{
		store->write_count++;
		store->write_bytes += header.file_bytes;
		store->write_ns += SparkKvSnapshotNowNs() - start_ns;
	}
	else
	{
		store->write_failure_count++;
		if ( status == SPARK_STATUS_IO_ERROR || status == SPARK_STATUS_CAPACITY_EXCEEDED )
			store->failed_status = status;
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
	if ( runtime->spare_bytes > store->queue_maximum_bytes || job->capacity > store->queue_maximum_bytes - runtime->spare_bytes )
	{
		free(job);
		return;
	}
	job->next = runtime->spare_jobs;
	runtime->spare_jobs = job;
	runtime->spare_bytes += job->capacity;
}

static SparkKvSnapshotJob *SparkKvSnapshotTakeSpareJob(SparkKvSnapshotRuntime *runtime,uint64_t payload)
{
	SparkKvSnapshotJob **link,*job;
	for (link=&runtime->spare_jobs; *link != 0; link=&(*link)->next)
		if ( (*link)->capacity >= payload )
		{
			job = *link;
			*link = job->next;
			runtime->spare_bytes -= job->capacity;
			return(job);
		}
	return(0);
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
		status = SparkKvSnapshotCommitFile(store,&job->key,job->segments,job->segment_count);
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
	SparkKvSnapshotKey key;
	struct dirent *item;
	struct stat info;
	SparkStatus status = SPARK_STATUS_OK;
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
		if ( SparkKvSnapshotParseName(item->d_name,&key) == 0u || stat(path,&info) != 0 || !S_ISREG(info.st_mode) )
			continue;
		status = SparkKvSnapshotIndexInsert(store,&key,(uint64_t)info.st_size,(uint64_t)info.st_mtime);
	}
	(void)closedir(listing);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( runtime->entry_count != 0u )
		qsort(runtime->entries,runtime->entry_count,sizeof(runtime->entries[0]),SparkKvSnapshotCompareRecency);
	for (index=0u; index<runtime->entry_count; index++)
		runtime->entries[index].recency = index + 1u;
	runtime->recency = runtime->entry_count;
	return(SparkKvSnapshotEvictLocked(store,0u));
}

static void SparkKvSnapshotRuntimeFree(SparkKvSnapshotRuntime *runtime)
{
	if ( runtime->lock_descriptor >= 0 )
		(void)close(runtime->lock_descriptor);
	free(runtime->entries);
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
	memcpy(store->directory,directory,strlen(directory) + 1u);
	store->runtime = runtime;
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

SparkStatus SparkKvSnapshotBinaryDigest(const void *symbol,uint8_t digest[SPARK_SHA256_DIGEST_BYTES],char *path,uint32_t path_capacity)
{
	Dl_info info;
	char hex[SPARK_SHA256_HEX_BYTES];
	SparkStatus status;
	if ( symbol == 0 || digest == 0 || path == 0 || path_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&info,0,sizeof(info));
	if ( dladdr(symbol,&info) == 0 || info.dli_fname == 0 || info.dli_fname[0] == '\0' )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( (uint32_t)snprintf(path,path_capacity,"%s",info.dli_fname) >= path_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkSha256File(path,hex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( SparkKvSnapshotParseHex(hex,digest) == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvSnapshotPrune(SparkKvSnapshotStore *store,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES])
{
	SparkKvSnapshotRuntime *runtime;
	uint32_t index = 0u;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || layout_sha256 == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	while ( index < runtime->entry_count )
	{
		if ( memcmp(runtime->entries[index].key.layout_sha256,layout_sha256,SPARK_SHA256_DIGEST_BYTES) == 0 )
		{
			index++;
			continue;
		}
		store->pruned_count++;
		SparkKvSnapshotUnlinkEntryLocked(store,index);
	}
	pthread_mutex_unlock(&runtime->mutex);
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotDiscard(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t failures;
	pthread_mutex_lock(&runtime->mutex);
	failures = ++store->checksum_failure_count;
	if ( SparkKvSnapshotPath(store,key,path,sizeof(path)) == SPARK_STATUS_OK )
		(void)unlink(path);
	SparkKvSnapshotIndexForget(store,key);
	pthread_mutex_unlock(&runtime->mutex);
	fprintf(stderr,"KV-SNAPSHOT discarded corrupt file tokens=%u checksum_failures=%llu\n",key->token_count,(unsigned long long)failures);
}

static SparkStatus SparkKvSnapshotValidateHeader(const SparkKvSnapshotFileHeader *header,const SparkKvSnapshotKey *key,uint64_t file_bytes)
{
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	uint64_t floor;
	uint32_t index;
	if ( header->magic != SPARK_KV_SNAPSHOT_MAGIC || header->format_version != SPARK_KV_SNAPSHOT_FORMAT_VERSION || header->header_bytes != sizeof(*header) || header->reserved0 != 0u )
		return(SPARK_STATUS_HASH_MISMATCH);
	SparkKvSnapshotHeaderDigest(header,digest);
	if ( memcmp(digest,header->header_sha256,sizeof(digest)) != 0 || memcmp(&header->key,key,sizeof(*key)) != 0 )
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
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct stat info;
	SparkStatus status;
	int descriptor;
	*descriptor_out = -1;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	runtime = SparkKvSnapshotRuntimeOf(store);
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
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
		status = SparkKvSnapshotValidateHeader(header,key,(uint64_t)info.st_size);
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

SparkStatus SparkKvSnapshotRemove(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint32_t index;
	SparkStatus status;
	if ( SparkKvSnapshotStoreIsValid(store) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	index = SparkKvSnapshotIndexFind(runtime,key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotUnlinkEntryLocked(store,index);
	pthread_mutex_unlock(&runtime->mutex);
	return(index != SPARK_KV_SNAPSHOT_NO_ENTRY ? SPARK_STATUS_OK : SPARK_STATUS_NOT_FOUND);
}

SparkStatus SparkKvSnapshotWrite(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	SparkKvSnapshotRuntime *runtime;
	uint64_t bytes[SPARK_KV_SNAPSHOT_MAX_SEGMENTS],file_bytes;
	SparkStatus status;
	uint32_t index;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || segments == 0 || segment_count == 0u || segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (index=0u; index<segment_count; index++)
		bytes[index] = segments[index].bytes;
	status = SparkKvSnapshotFileBytes(bytes,segment_count,&file_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	runtime = SparkKvSnapshotRuntimeOf(store);
	pthread_mutex_lock(&runtime->mutex);
	if ( store->failed_status != SPARK_STATUS_OK )
		status = store->failed_status;
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
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkKvSnapshotCommitFile(store,key,segments,segment_count));
}

SparkStatus SparkKvSnapshotWriteBegin(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const uint32_t *kinds,const uint64_t *bytes,uint32_t segment_count,SparkKvSnapshotWriteTicket *ticket)
{
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotJob *job;
	uint64_t file_bytes = 0u,payload = 0u;
	uint8_t *cursor;
	SparkStatus status;
	uint32_t index;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( ticket != 0 )
		memset(ticket,0,sizeof(*ticket));
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || ticket == 0 || kinds == 0 || bytes == 0 || segment_count == 0u || segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
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
	if ( store->failed_status != SPARK_STATUS_OK )
		status = store->failed_status;
	else if ( job != 0 || SparkKvSnapshotIndexFind(runtime,key) != SPARK_KV_SNAPSHOT_NO_ENTRY )
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
	if ( status == SPARK_STATUS_OK && (job = SparkKvSnapshotTakeSpareJob(runtime,payload)) == 0 && (job = (SparkKvSnapshotJob *)malloc(sizeof(*job) + (size_t)payload)) != 0 )
		job->capacity = payload;
	if ( status == SPARK_STATUS_OK && job == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
	{
		job->ready = 0u;
		memset(job->segments,0,sizeof(job->segments));
		job->key = *key;
		job->segment_count = segment_count;
		job->file_bytes = file_bytes;
		cursor = (uint8_t *)(job + 1);
		for (index=0u; index<segment_count; index++)
		{
			job->segments[index] = (SparkKvSnapshotSegment){.kind=kinds[index],.bytes=bytes[index],.data=cursor};
			cursor += bytes[index];
		}
		job->next = runtime->jobs;
		runtime->jobs = job;
		store->queued_count++;
		store->queued_bytes += file_bytes;
	}
	pthread_mutex_unlock(&runtime->mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	ticket->key = *key;
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
	if ( segment_index >= header.segment_count || header.segments[segment_index].kind != kind )
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
