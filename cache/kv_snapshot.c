#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_error_site.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static uint64_t SparkKvSnapshotNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static uint32_t SparkKvSnapshotStoreIsValid(const SparkKvSnapshotStore *store)
{
	return(store != 0 && store->abi_version == SPARK_KV_SNAPSHOT_ABI_VERSION && store->descriptor_bytes == SPARK_KV_SNAPSHOT_STORE_BYTES && store->directory[0] != '\0' && store->maximum_bytes != 0u ? 1u : 0u);
}

static uint32_t SparkKvSnapshotHasSuffix(const char *name,const char *suffix)
{
	size_t name_bytes = strlen(name),suffix_bytes = strlen(suffix);
	return(name_bytes > suffix_bytes && strcmp(name + name_bytes - suffix_bytes,suffix) == 0 ? 1u : 0u);
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

SparkStatus SparkKvSnapshotStoreOpen(SparkKvSnapshotStore *store,const char *directory,uint64_t maximum_bytes)
{
	struct stat info;
	struct dirent *item;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	DIR *listing;
	if ( store == 0 || directory == 0 || directory[0] != '/' || maximum_bytes == 0u || strlen(directory) >= SPARK_KV_SNAPSHOT_PATH_BYTES / 2u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( stat(directory,&info) != 0 || !S_ISDIR(info.st_mode) )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(store,0,sizeof(*store));
	store->abi_version = SPARK_KV_SNAPSHOT_ABI_VERSION;
	store->descriptor_bytes = SPARK_KV_SNAPSHOT_STORE_BYTES;
	store->maximum_bytes = maximum_bytes;
	memcpy(store->directory,directory,strlen(directory) + 1u);
	listing = opendir(directory);
	if ( listing == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	while ( (item = readdir(listing)) != 0 )
	{
		if ( (size_t)snprintf(path,sizeof(path),"%s/%s",directory,item->d_name) >= sizeof(path) )
			continue;
		if ( strncmp(item->d_name,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX,strlen(SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX)) == 0 )
		{
			if ( unlink(path) == 0 )
				store->removed_temporary_count++;
			continue;
		}
		if ( SparkKvSnapshotHasSuffix(item->d_name,SPARK_KV_SNAPSHOT_SUFFIX) == 0u || stat(path,&info) != 0 || !S_ISREG(info.st_mode) )
			continue;
		store->used_bytes += (uint64_t)info.st_size;
		store->file_count++;
	}
	(void)closedir(listing);
	return(SPARK_STATUS_OK);
}

void SparkKvSnapshotStoreClose(SparkKvSnapshotStore *store)
{
	if ( store != 0 )
		memset(store,0,sizeof(*store));
}

SparkStatus SparkKvSnapshotPath(const SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,char *path,uint32_t path_capacity)
{
	char layout[SPARK_SHA256_HEX_BYTES],identity[SPARK_SHA256_HEX_BYTES];
	if ( SparkKvSnapshotStoreIsValid(store) == 0u || key == 0 || path == 0 || key->token_count == 0u || key->reserved0 != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkSha256DigestToHex(key->layout_sha256,layout);
	SparkSha256DigestToHex(key->identity_sha256,identity);
	if ( (uint32_t)snprintf(path,path_capacity,"%s/%s-%s-%u%s",store->directory,layout,identity,key->token_count,SPARK_KV_SNAPSHOT_SUFFIX) >= path_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
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
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
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

static SparkStatus SparkKvSnapshotRemovePath(SparkKvSnapshotStore *store,const char *path)
{
	struct stat info;
	if ( stat(path,&info) != 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	if ( unlink(path) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	store->used_bytes = store->used_bytes >= (uint64_t)info.st_size ? store->used_bytes - (uint64_t)info.st_size : 0u;
	if ( store->file_count != 0u )
		store->file_count--;
	return(SPARK_STATUS_OK);
}

static void SparkKvSnapshotDiscard(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	store->checksum_failure_count++;
	if ( SparkKvSnapshotPath(store,key,path,sizeof(path)) == SPARK_STATUS_OK )
		(void)SparkKvSnapshotRemovePath(store,path);
}

static SparkStatus SparkKvSnapshotValidateHeader(const SparkKvSnapshotFileHeader *header,const SparkKvSnapshotKey *key,uint64_t file_bytes)
{
	uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
	uint64_t floor;
	uint32_t index;
	if ( header->magic != SPARK_KV_SNAPSHOT_MAGIC || header->format_version != SPARK_KV_SNAPSHOT_FORMAT_VERSION || header->header_bytes != sizeof(*header) || header->reserved0 != 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	SparkKvSnapshotHeaderDigest(header,digest);
	if ( memcmp(digest,header->header_sha256,sizeof(digest)) != 0 || memcmp(&header->key,key,sizeof(*key)) != 0 )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( header->segment_count == 0u || header->segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS || header->file_bytes != file_bytes )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	floor = SparkKvSnapshotAlign(sizeof(*header));
	for (index=0u; index<header->segment_count; index++)
	{
		const SparkKvSnapshotFileSegment *segment = &header->segments[index];
		if ( segment->kind == 0u || segment->reserved0 != 0u || segment->offset < floor || segment->offset % SPARK_KV_SNAPSHOT_ALIGNMENT != 0u || segment->bytes > file_bytes || segment->offset > file_bytes - segment->bytes )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		floor = segment->offset + segment->bytes;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotOpenValidated(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header,int *descriptor_out,uint32_t record_miss)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	struct stat info;
	SparkStatus status;
	int descriptor;
	*descriptor_out = -1;
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	descriptor = open(path,O_RDONLY | O_CLOEXEC);
	if ( descriptor < 0 )
	{
		if ( errno == ENOENT )
		{
			store->miss_count += record_miss;
			return(SPARK_STATUS_NOT_FOUND);
		}
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = fstat(descriptor,&info) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && (uint64_t)info.st_size < sizeof(*header) )
		status = SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotReadExact(descriptor,header,sizeof(*header),0u);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotValidateHeader(header,key,(uint64_t)info.st_size);
	if ( status != SPARK_STATUS_OK )
	{
		(void)close(descriptor);
		if ( status == SPARK_STATUS_VALIDATION_FAILED )
			SparkKvSnapshotDiscard(store,key);
		SPARK_RETURN(status);
	}
	*descriptor_out = descriptor;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkKvSnapshotStatRecord(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header,uint32_t record_miss)
{
	SparkStatus status;
	int descriptor;
	if ( header == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotOpenValidated(store,key,header,&descriptor,record_miss);
	if ( status == SPARK_STATUS_OK )
		(void)close(descriptor);
	return(status);
}

SparkStatus SparkKvSnapshotStat(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header)
{
	return(SparkKvSnapshotStatRecord(store,key,header,1u));
}

SparkStatus SparkKvSnapshotRemove(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SparkStatus status = SparkKvSnapshotPath(store,key,path,sizeof(path));
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkKvSnapshotRemovePath(store,path));
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

SparkStatus SparkKvSnapshotWrite(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotSegment *segments,uint32_t segment_count)
{
	static uint64_t sequence;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES],temporary[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SparkKvSnapshotFileHeader header,existing;
	uint64_t start_ns = SparkKvSnapshotNowNs();
	SparkStatus status;
	uint32_t index;
	int descriptor;
	if ( segments == 0 || segment_count == 0u || segment_count > SPARK_KV_SNAPSHOT_MAX_SEGMENTS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkKvSnapshotPath(store,key,path,sizeof(path));
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvSnapshotStatRecord(store,key,&existing,0u);
	if ( status == SPARK_STATUS_OK )
	{
		store->duplicate_count++;
		return(SPARK_STATUS_OK);
	}
	if ( status == SPARK_STATUS_VALIDATION_FAILED )
		status = SparkKvSnapshotRemovePath(store,path);
	else if ( status == SPARK_STATUS_NOT_FOUND )
		status = SPARK_STATUS_OK;
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvSnapshotBuildHeader(key,segments,segment_count,&header);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( header.file_bytes > store->maximum_bytes || store->used_bytes > store->maximum_bytes - header.file_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (size_t)snprintf(temporary,sizeof(temporary),"%s/%s%ld-%llu",store->directory,SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX,(long)getpid(),(unsigned long long)__atomic_add_fetch(&sequence,1u,__ATOMIC_RELAXED)) >= sizeof(temporary) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	descriptor = open(temporary,O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0600);
	if ( descriptor < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	for (index=0u; status == SPARK_STATUS_OK && index<segment_count; index++)
		status = SparkKvSnapshotWriteExact(descriptor,segments[index].data,segments[index].bytes,header.segments[index].offset);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvSnapshotWriteExact(descriptor,&header,sizeof(header),0u);
	if ( status == SPARK_STATUS_OK && ftruncate(descriptor,(off_t)header.file_bytes) != 0 )
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
	status = SparkKvSnapshotSyncDirectory(store);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	store->used_bytes += header.file_bytes;
	store->file_count++;
	store->write_count++;
	store->write_bytes += header.file_bytes;
	store->write_ns += SparkKvSnapshotNowNs() - start_ns;
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
	{
		SparkKvSnapshotDiscard(store,key);
		status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	store->read_count++;
	store->read_bytes += bytes;
	store->read_ns += SparkKvSnapshotNowNs() - start_ns;
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
		{
			SparkKvSnapshotDiscard(store,key);
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*bytes_out = header.segments[segment_index].bytes;
	store->read_count++;
	store->read_bytes += header.segments[segment_index].bytes;
	store->read_ns += SparkKvSnapshotNowNs() - start_ns;
	return(SPARK_STATUS_OK);
}
