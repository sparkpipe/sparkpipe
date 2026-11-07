#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_error_site.h"

#define SPARK_KV_PURGE_BUCKET_SECONDS UINT64_C(60)
#define SPARK_KV_PURGE_BUCKETS 4096u

typedef void (*SparkKvSnapshotPurgeVisitor)(void *context,const char *path,const char *name,const struct stat *info);

typedef struct SparkKvSnapshotPurgeTally
{
	SparkKvSnapshotPurgeResult *result;
	const SparkWorkingStateDirective *directive;
	uint64_t now_seconds;
	uint64_t remaining_bytes;
	uint64_t oldest_seconds;
	uint64_t freed_bytes;
	uint32_t cutoff_bucket;
	uint64_t *bucket_bytes;
} SparkKvSnapshotPurgeTally;

static uint32_t SparkKvSnapshotPurgeEndsWith(const char *name,const char *suffix)
{
	size_t length = strlen(name),tail = strlen(suffix);
	return(length > tail && strcmp(name + length - tail,suffix) == 0 ? 1u : 0u);
}

static uint32_t SparkKvSnapshotPurgeIsTemporary(const char *name)
{
	return(strncmp(name,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX,strlen(SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX)) == 0 ? 1u : 0u);
}

uint32_t SparkKvSnapshotIsCacheFileName(const char *name)
{
	if ( name == 0 )
		return(0u);
	return(SparkKvSnapshotPurgeEndsWith(name,SPARK_KV_SNAPSHOT_SUFFIX) != 0u || SparkKvSnapshotPurgeEndsWith(name,SPARK_KV_SNAPSHOT_STATE_SUFFIX) != 0u ||
		SparkKvSnapshotPurgeIsTemporary(name) != 0u ? 1u : 0u);
}

uint32_t SparkKvSnapshotIsShardName(const char *name)
{
	static const char digits[] = "0123456789abcdef";
	return(name != 0 && name[0] != '\0' && name[1] != '\0' && name[2] == '\0' && strchr(digits,name[0]) != 0 && strchr(digits,name[1]) != 0 ? 1u : 0u);
}

static void SparkKvSnapshotPurgeVisitFiles(const char *directory,SparkKvSnapshotPurgeVisitor visitor,void *context)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct dirent *item;
	struct stat info;
	DIR *listing = opendir(directory);
	if ( listing == 0 )
		return;
	while ( (item = readdir(listing)) != 0 )
	{
		if ( SparkKvSnapshotIsCacheFileName(item->d_name) == 0u )
			continue;
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name) >= sizeof(path) )
			continue;
		if ( lstat(path,&info) == 0 && S_ISREG(info.st_mode) )
			visitor(context,path,item->d_name,&info);
	}
	(void)closedir(listing);
}

static SparkStatus SparkKvSnapshotPurgeVisit(const char *directory,SparkKvSnapshotPurgeVisitor visitor,void *context)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct dirent *item;
	struct stat info;
	DIR *listing = opendir(directory);
	if ( listing == 0 )
		SPARK_FAIL(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	while ( (item = readdir(listing)) != 0 )
	{
		if ( SparkKvSnapshotIsShardName(item->d_name) == 0u )
			continue;
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name) >= sizeof(path) )
			continue;
		if ( lstat(path,&info) == 0 && S_ISDIR(info.st_mode) )
			SparkKvSnapshotPurgeVisitFiles(path,visitor,context);
	}
	(void)closedir(listing);
	SparkKvSnapshotPurgeVisitFiles(directory,visitor,context);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotPurgeLock(const char *directory,int *lock_out)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	int lock;
	*lock_out = -1;
	if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,SPARK_KV_SNAPSHOT_LOCK_NAME) >= sizeof(path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	lock = open(path,O_RDWR | O_CLOEXEC | O_NOFOLLOW);
	if ( lock < 0 && errno != ENOENT )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( lock >= 0 && flock(lock,LOCK_EX | LOCK_NB) != 0 )
	{
		(void)close(lock);
		if ( errno != EWOULDBLOCK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return(SPARK_STATUS_BUSY);
	}
	*lock_out = lock;
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotPurgeKeep(SparkKvSnapshotPurgeTally *tally,const struct stat *info)
{
	tally->remaining_bytes += (uint64_t)info->st_size;
	if ( (uint64_t)info->st_mtime < tally->oldest_seconds )
		tally->oldest_seconds = (uint64_t)info->st_mtime;
}

static void SparkKvSnapshotPurgeUnlink(SparkKvSnapshotPurgeTally *tally,const char *path,const struct stat *info)
{
	if ( unlink(path) != 0 )
	{
		tally->result->failed_unlinks++;
		SparkKvSnapshotPurgeKeep(tally,info);
		return;
	}
	tally->result->deleted_files++;
	tally->result->deleted_bytes += (uint64_t)info->st_size;
	tally->freed_bytes += (uint64_t)info->st_size;
}

static void SparkKvSnapshotPurgeExpired(void *context,const char *path,const char *name,const struct stat *info)
{
	SparkKvSnapshotPurgeTally *tally = (SparkKvSnapshotPurgeTally *)context;
	if ( SparkKvSnapshotPurgeIsTemporary(name) == 0u && SparkWorkingStateExpired(tally->directive,(uint64_t)info->st_mtime,tally->now_seconds) == 0u )
		SparkKvSnapshotPurgeKeep(tally,info);
	else
		SparkKvSnapshotPurgeUnlink(tally,path,info);
}

static uint32_t SparkKvSnapshotPurgeBucket(const SparkKvSnapshotPurgeTally *tally,const struct stat *info)
{
	uint64_t age = tally->now_seconds > (uint64_t)info->st_mtime ? tally->now_seconds - (uint64_t)info->st_mtime : 0u;
	age /= SPARK_KV_PURGE_BUCKET_SECONDS;
	return(age >= SPARK_KV_PURGE_BUCKETS ? SPARK_KV_PURGE_BUCKETS - 1u : (uint32_t)age);
}

static void SparkKvSnapshotPurgeCount(void *context,const char *path,const char *name,const struct stat *info)
{
	SparkKvSnapshotPurgeTally *tally = (SparkKvSnapshotPurgeTally *)context;
	(void)path;
	if ( SparkKvSnapshotPurgeIsTemporary(name) == 0u )
		tally->bucket_bytes[SparkKvSnapshotPurgeBucket(tally,info)] += (uint64_t)info->st_size;
}

static void SparkKvSnapshotPurgeCut(void *context,const char *path,const char *name,const struct stat *info)
{
	SparkKvSnapshotPurgeTally *tally = (SparkKvSnapshotPurgeTally *)context;
	if ( SparkKvSnapshotPurgeIsTemporary(name) != 0u || SparkKvSnapshotPurgeBucket(tally,info) >= tally->cutoff_bucket )
		SparkKvSnapshotPurgeUnlink(tally,path,info);
	else
		SparkKvSnapshotPurgeKeep(tally,info);
}

static uint64_t SparkKvSnapshotPurgeWallSeconds(void)
{
	struct timespec now;
	return(clock_gettime(CLOCK_REALTIME,&now) == 0 ? (uint64_t)now.tv_sec : 0u);
}

SparkStatus SparkKvSnapshotPurgeOldest(SparkKvLedger *ledger,const char *directory,uint64_t needed_bytes,uint64_t *freed_bytes)
{
	SparkKvSnapshotPurgeResult result;
	SparkKvSnapshotPurgeTally tally;
	uint64_t sum = 0u;
	uint32_t bucket;
	SparkStatus status;
	int lock;
	if ( ledger == 0 || directory == 0 || freed_bytes == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*freed_bytes = 0u;
	status = SparkKvSnapshotPurgeLock(directory,&lock);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&result,0,sizeof(result));
	memset(&tally,0,sizeof(tally));
	tally.result = &result;
	tally.now_seconds = SparkKvSnapshotPurgeWallSeconds();
	tally.oldest_seconds = SPARK_KV_LEDGER_NO_OLDEST;
	tally.bucket_bytes = (uint64_t *)calloc(SPARK_KV_PURGE_BUCKETS,sizeof(uint64_t));
	status = tally.bucket_bytes != 0 ? SparkKvSnapshotPurgeVisit(directory,SparkKvSnapshotPurgeCount,&tally) : SPARK_STATUS_CAPACITY_EXCEEDED;
	for (bucket=SPARK_KV_PURGE_BUCKETS; status == SPARK_STATUS_OK && bucket != 0u && sum < needed_bytes; bucket--)
		sum += tally.bucket_bytes[bucket - 1u];
	tally.cutoff_bucket = bucket;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotPurgeVisit(directory,SparkKvSnapshotPurgeCut,&tally);
	free(tally.bucket_bytes);
	if ( status == SPARK_STATUS_OK || status == SPARK_STATUS_NOT_FOUND )
		(void)SparkKvLedgerRecord(ledger,directory,tally.remaining_bytes,tally.freed_bytes != 0u ? tally.oldest_seconds : SPARK_KV_LEDGER_NO_OLDEST);
	if ( lock >= 0 )
		(void)close(lock);
	*freed_bytes = tally.freed_bytes;
	if ( status == SPARK_STATUS_OK )
		fprintf(stderr,"kv snapshot store %s was closed and held the node's oldest KV: deleted files=%llu bytes=%llu for the %llu-byte node quota shortfall\n",directory,
			(unsigned long long)result.deleted_files,(unsigned long long)result.deleted_bytes,(unsigned long long)needed_bytes);
	return(status);
}

SparkStatus SparkKvSnapshotPurgeDirectory(SparkKvLedger *ledger,const char *directory,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result)
{
	SparkKvSnapshotPurgeTally tally;
	SparkStatus status;
	int lock;
	if ( ledger == 0 || directory == 0 || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPurgeLock(directory,&lock);
	if ( status == SPARK_STATUS_BUSY )
	{
		result->busy_stores++;
		return(SPARK_STATUS_OK);
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&tally,0,sizeof(tally));
	tally.result = result;
	tally.directive = directive;
	tally.now_seconds = now_seconds;
	tally.oldest_seconds = SPARK_KV_LEDGER_NO_OLDEST;
	status = SparkKvSnapshotPurgeVisit(directory,SparkKvSnapshotPurgeExpired,&tally);
	if ( status == SPARK_STATUS_OK )
	{
		result->stores++;
		if ( SparkKvLedgerRecord(ledger,directory,tally.remaining_bytes,tally.oldest_seconds) != SPARK_STATUS_OK )
			result->uncounted_stores++;
	}
	if ( lock >= 0 )
		(void)close(lock);
	SPARK_RETURN(status);
}

static SparkStatus SparkKvSnapshotPurgeWalk(SparkKvLedger *ledger,const char *directory,dev_t device,uint32_t depth,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct dirent *item;
	struct stat info;
	uint32_t holds_cache = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	DIR *listing;
	if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,SPARK_KV_SNAPSHOT_LOCK_NAME) < sizeof(path) && lstat(path,&info) == 0 )
		SPARK_RETURN(SparkKvSnapshotPurgeDirectory(ledger,directory,directive,now_seconds,result));
	listing = opendir(directory);
	if ( listing == 0 )
		return(errno == EACCES || errno == ENOENT ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
	while ( holds_cache == 0u && (item = readdir(listing)) != 0 )
		holds_cache = SparkKvSnapshotIsCacheFileName(item->d_name);
	rewinddir(listing);
	while ( (item = readdir(listing)) != 0 )
	{
		if ( strcmp(item->d_name,".") == 0 || strcmp(item->d_name,"..") == 0 || SparkKvSnapshotIsCacheFileName(item->d_name) != 0u ||
			(holds_cache != 0u && SparkKvSnapshotIsShardName(item->d_name) != 0u) )
			continue;
		if ( depth >= SPARK_KV_PURGE_DEPTH_MAX || (item->d_type != DT_DIR && item->d_type != DT_UNKNOWN) )
			continue;
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name) >= sizeof(path) )
			continue;
		if ( lstat(path,&info) != 0 || !S_ISDIR(info.st_mode) || info.st_dev != device )
			continue;
		if ( SparkKvSnapshotPurgeWalk(ledger,path,device,depth + 1u,directive,now_seconds,result) != SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
	}
	(void)closedir(listing);
	if ( holds_cache != 0u && SparkKvSnapshotPurgeDirectory(ledger,directory,directive,now_seconds,result) != SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

SparkStatus SparkKvSnapshotPurgeTree(SparkKvLedger *ledger,const char *root,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result)
{
	struct stat info;
	if ( ledger == 0 || root == 0 || root[0] != '/' || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(result,0,sizeof(*result));
	if ( lstat(root,&info) != 0 || !S_ISDIR(info.st_mode) )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	SparkKvLedgerForgetMissing(ledger);
	SPARK_RETURN(SparkKvSnapshotPurgeWalk(ledger,root,info.st_dev,0u,directive,now_seconds,result));
}
