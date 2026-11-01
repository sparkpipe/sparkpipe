#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_working_state.h"
#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_error_site.h"

static const char SparkWorkingStateHeader[] = "sparkpipe-working-state 1\n";

SparkStatus SparkWorkingStateOwnerRoot(char *path,uint32_t capacity)
{
	const struct passwd *owner = getpwuid(geteuid());
	if ( path == 0 || capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( owner == 0 || owner->pw_dir == 0 || owner->pw_dir[0] != '/' )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( strlen(owner->pw_dir) >= capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(path,owner->pw_dir,strlen(owner->pw_dir) + 1u);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWorkingStateDirectivePath(const char *home,char *path,uint32_t capacity)
{
	int written;
	if ( home == 0 || home[0] != '/' || path == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	written = snprintf(path,capacity,"%s/%s/%s",home,SPARK_WORKING_STATE_ROOT,SPARK_WORKING_STATE_DIRECTIVE_NAME);
	if ( written < 0 || (uint32_t)written >= capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkWorkingStateParseLine(const char **cursor,const char *end,const char *name,uint64_t *value)
{
	const char *at = *cursor;
	size_t length = strlen(name);
	uint32_t digits = 0u;
	*value = 0u;
	if ( (size_t)(end - at) <= length || memcmp(at,name,length) != 0 || at[length] != ' ' )
		return(0u);
	at += length + 1u;
	while ( at < end && *at >= '0' && *at <= '9' )
	{
		if ( (digits != 0u && *value == 0u) || *value > (SPARK_WORKING_STATE_SECONDS_MAX - (uint64_t)(*at - '0')) / 10u )
			return(0u);
		*value = *value * 10u + (uint64_t)(*at - '0');
		digits++;
		at++;
	}
	if ( digits == 0u || at >= end || *at != '\n' )
		return(0u);
	*cursor = at + 1;
	return(1u);
}

SparkStatus SparkWorkingStateDirectiveParse(const char *text,uint32_t bytes,SparkWorkingStateDirective *directive)
{
	const char *cursor,*end;
	SparkWorkingStateDirective parsed;
	if ( directive == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(directive,0,sizeof(*directive));
	if ( text == 0 || bytes < sizeof(SparkWorkingStateHeader) - 1u || memcmp(text,SparkWorkingStateHeader,sizeof(SparkWorkingStateHeader) - 1u) != 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	cursor = text + sizeof(SparkWorkingStateHeader) - 1u;
	end = text + bytes;
	if ( SparkWorkingStateParseLine(&cursor,end,"wiped",&parsed.wiped_seconds) == 0u ||
		SparkWorkingStateParseLine(&cursor,end,"held_until",&parsed.held_until_seconds) == 0u || cursor != end )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	*directive = parsed;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWorkingStateDirectiveRead(const char *path,SparkWorkingStateDirective *directive)
{
	char text[SPARK_WORKING_STATE_DIRECTIVE_BYTES_MAX + 1u];
	ssize_t got;
	size_t bytes = 0u;
	int descriptor;
	if ( directive == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(directive,0,sizeof(*directive));
	if ( path == 0 || path[0] != '/' )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	descriptor = open(path,O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
	if ( descriptor < 0 )
	{
		if ( errno == ENOENT )
			return(SPARK_STATUS_OK);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	while ( bytes < sizeof(text) && (got = read(descriptor,text + bytes,sizeof(text) - bytes)) != 0 )
	{
		if ( got < 0 )
		{
			if ( errno == EINTR )
				continue;
			(void)close(descriptor);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		bytes += (size_t)got;
	}
	(void)close(descriptor);
	if ( bytes > SPARK_WORKING_STATE_DIRECTIVE_BYTES_MAX )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	SPARK_RETURN(SparkWorkingStateDirectiveParse(text,(uint32_t)bytes,directive));
}

uint32_t SparkWorkingStateDirectiveEqual(const SparkWorkingStateDirective *left,const SparkWorkingStateDirective *right)
{
	return(left->wiped_seconds == right->wiped_seconds && left->held_until_seconds == right->held_until_seconds ? 1u : 0u);
}

uint32_t SparkWorkingStateExpired(const SparkWorkingStateDirective *directive,uint64_t used_seconds,uint64_t now_seconds)
{
	uint64_t since = used_seconds;
	if ( directive != 0 && used_seconds < directive->wiped_seconds )
		return(1u);
	if ( directive != 0 && directive->held_until_seconds > since )
		since = directive->held_until_seconds;
	return(since + SPARK_KV_RETENTION_SECONDS <= now_seconds ? 1u : 0u);
}

static void SparkWorkingStateEmpty(int directory,dev_t device,uint32_t depth,SparkWorkingStatePurgeResult *result)
{
	struct dirent *item;
	struct stat info;
	DIR *listing;
	int copy = dup(directory),child;
	if ( copy < 0 || (listing = fdopendir(copy)) == 0 )
	{
		if ( copy >= 0 )
			(void)close(copy);
		result->failed_unlinks++;
		return;
	}
	while ( (item = readdir(listing)) != 0 )
	{
		if ( strcmp(item->d_name,".") == 0 || strcmp(item->d_name,"..") == 0 )
			continue;
		if ( fstatat(directory,item->d_name,&info,AT_SYMLINK_NOFOLLOW) != 0 )
			continue;
		if ( S_ISDIR(info.st_mode) )
		{
			if ( info.st_dev != device || depth >= SPARK_WORKING_STATE_SCRATCH_DEPTH_MAX ||
				(child = openat(directory,item->d_name,O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)) < 0 )
			{
				result->failed_unlinks++;
				continue;
			}
			SparkWorkingStateEmpty(child,device,depth + 1u,result);
			(void)close(child);
			if ( unlinkat(directory,item->d_name,AT_REMOVEDIR) != 0 )
				result->failed_unlinks++;
			else
				result->deleted_entries++;
			continue;
		}
		if ( unlinkat(directory,item->d_name,0) != 0 )
		{
			result->failed_unlinks++;
			continue;
		}
		result->deleted_entries++;
		if ( S_ISREG(info.st_mode) )
			result->deleted_bytes += (uint64_t)info.st_size;
	}
	(void)closedir(listing);
}

SparkStatus SparkWorkingStatePurgeScratchArea(const char *area,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkWorkingStatePurgeResult *result)
{
	struct stat info,stamp;
	uint64_t used;
	int directory;
	if ( area == 0 || area[0] != '/' || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	directory = open(area,O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
	if ( directory < 0 )
	{
		if ( errno == ENOENT )
			return(SPARK_STATUS_OK);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	if ( fstat(directory,&info) != 0 )
	{
		(void)close(directory);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	result->areas++;
	used = (uint64_t)info.st_mtime;
	if ( fstatat(directory,SPARK_WORKING_STATE_LAST_USE_NAME,&stamp,AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(stamp.st_mode) )
		used = (uint64_t)stamp.st_mtime;
	if ( SparkWorkingStateExpired(directive,used,now_seconds) != 0u )
	{
		SparkWorkingStateEmpty(directory,info.st_dev,0u,result);
		result->emptied_areas++;
	}
	(void)close(directory);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWorkingStatePurgeScratch(const char *home,const SparkWorkingStateDirective *directive,uint64_t now_seconds,SparkWorkingStatePurgeResult *result)
{
	char path[SPARK_WORKING_STATE_PATH_BYTES];
	struct dirent *item;
	DIR *listing;
	SparkStatus status;
	if ( home == 0 || home[0] != '/' || result == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(result,0,sizeof(*result));
	if ( (uint32_t)snprintf(path,sizeof(path),"%s/%s/%s",home,SPARK_WORKING_STATE_ROOT,SPARK_WORKING_STATE_SCRATCH_NAME) >= sizeof(path) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkWorkingStatePurgeScratchArea(path,directive,now_seconds,result);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( (uint32_t)snprintf(path,sizeof(path),"%s/%s/%s",home,SPARK_WORKING_STATE_ROOT,SPARK_WORKING_STATE_MIRROR_NAME) >= sizeof(path) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	listing = opendir(path);
	if ( listing == 0 )
		return(errno == ENOENT ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR);
	while ( status == SPARK_STATUS_OK && (item = readdir(listing)) != 0 )
	{
		char area[SPARK_WORKING_STATE_PATH_BYTES];
		if ( item->d_name[0] == '.' )
			continue;
		if ( (uint32_t)snprintf(area,sizeof(area),"%s/%s/%s/%s",path,item->d_name,SPARK_WORKING_STATE_ROOT,SPARK_WORKING_STATE_SCRATCH_NAME) >= sizeof(area) )
			continue;
		status = SparkWorkingStatePurgeScratchArea(area,directive,now_seconds,result);
	}
	(void)closedir(listing);
	SPARK_RETURN(status);
}
