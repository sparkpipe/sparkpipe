#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_error_site.h"

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

SparkStatus SparkKvSnapshotPurgeDirectory(const char *directory,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct dirent *item;
	struct stat info;
	DIR *listing;
	int lock = -1;
	if ( directory == 0 || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		result->busy_stores++;
		return(SPARK_STATUS_OK);
	}
	listing = opendir(directory);
	if ( listing == 0 )
	{
		if ( lock >= 0 )
			(void)close(lock);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	result->stores++;
	while ( (item = readdir(listing)) != 0 )
	{
		if ( SparkKvSnapshotIsCacheFileName(item->d_name) == 0u )
			continue;
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name) >= sizeof(path) )
			continue;
		if ( lstat(path,&info) != 0 || !S_ISREG(info.st_mode) )
			continue;
		if ( SparkKvSnapshotPurgeIsTemporary(item->d_name) == 0u && (uint64_t)info.st_mtime + SPARK_KV_RETENTION_SECONDS > now_seconds )
			continue;
		if ( unlink(path) != 0 )
		{
			result->failed_unlinks++;
			continue;
		}
		result->deleted_files++;
		result->deleted_bytes += (uint64_t)info.st_size;
	}
	(void)closedir(listing);
	if ( lock >= 0 )
		(void)close(lock);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotPurgeWalk(const char *directory,dev_t device,uint32_t depth,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct dirent *item;
	struct stat info;
	uint32_t holds_cache = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	DIR *listing = opendir(directory);
	if ( listing == 0 )
		return(errno == EACCES || errno == ENOENT ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
	while ( (item = readdir(listing)) != 0 )
	{
		if ( strcmp(item->d_name,".") == 0 || strcmp(item->d_name,"..") == 0 )
			continue;
		if ( SparkKvSnapshotIsCacheFileName(item->d_name) != 0u || strcmp(item->d_name,SPARK_KV_SNAPSHOT_LOCK_NAME) == 0 )
		{
			holds_cache = 1u;
			continue;
		}
		if ( depth >= SPARK_KV_PURGE_DEPTH_MAX || (item->d_type != DT_DIR && item->d_type != DT_UNKNOWN) )
			continue;
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name) >= sizeof(path) )
			continue;
		if ( lstat(path,&info) != 0 || !S_ISDIR(info.st_mode) || info.st_dev != device )
			continue;
		if ( SparkKvSnapshotPurgeWalk(path,device,depth + 1u,now_seconds,result) != SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
	}
	(void)closedir(listing);
	if ( holds_cache != 0u && SparkKvSnapshotPurgeDirectory(directory,now_seconds,result) != SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

SparkStatus SparkKvSnapshotPurgeTree(const char *root,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result)
{
	struct stat info;
	if ( root == 0 || root[0] != '/' || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(result,0,sizeof(*result));
	if ( lstat(root,&info) != 0 || !S_ISDIR(info.st_mode) )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	SPARK_RETURN(SparkKvSnapshotPurgeWalk(root,info.st_dev,0u,now_seconds,result));
}
