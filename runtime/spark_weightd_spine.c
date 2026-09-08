#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_weightd_spine.h"
#include "sparkpipe/spark_sha256.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static SparkStatus spine_read(int32_t fd,uint8_t *buffer,uint64_t offset,uint32_t bytes)
{
	uint32_t done = 0u;
	ssize_t count;
	while ( done < bytes )
	{
		count = pread(fd,buffer + done,bytes - done,(off_t)(offset + done));
		if ( count < 0 && errno == EINTR )
			continue;
		if ( count <= 0 )
			return(SPARK_STATUS_IO_ERROR);
		done += (uint32_t)count;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus spine_copy(const SparkWeightdManifest *manifest,uint32_t *index,const uint8_t *buffer,uint64_t offset,uint32_t bytes,uint8_t *destination)
{
	const SparkWeightdSpan *span;
	uint64_t start,end,limit = (offset + bytes);
	while ( *index < manifest->spine_count )
	{
		span = &manifest->spine[*index];
		if ( span->offset >= limit )
			break;
		start = span->offset > offset ? span->offset : offset;
		end = (span->offset + span->bytes);
		if ( end > limit )
			end = limit;
		if ( end > start && cudaMemcpy(destination + span->compact_offset + (start - span->offset),buffer + (start - offset),(size_t)(end - start),cudaMemcpyHostToDevice) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
		if ( (span->offset + span->bytes) > limit )
			break;
		(*index)++;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus spine_stream(int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *expected,uint8_t *destination)
{
	SparkSha256Context hash;
	uint8_t buffer[65536],digest[32];
	char hex[SPARK_SHA256_HEX_BYTES];
	uint64_t offset = 0u;
	uint32_t bytes,index = 0u;
	SparkStatus status;
	SparkSha256Initialize(&hash);
	while ( offset < pack_bytes )
	{
		bytes = (uint32_t)((pack_bytes - offset) < sizeof(buffer) ? (pack_bytes - offset) : sizeof(buffer));
		status = spine_read(fd,buffer,offset,bytes);
		if ( status != SPARK_STATUS_OK )
			return(status);
		SparkSha256Update(&hash,buffer,bytes);
		status = spine_copy(manifest,&index,buffer,offset,bytes,destination);
		if ( status != SPARK_STATUS_OK )
			return(status);
		offset += bytes;
	}
	SparkSha256Finalize(&hash,digest);
	SparkSha256DigestToHex(digest,hex);
	return(strcmp(hex,expected) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_HASH_MISMATCH);
}

static int32_t spine_unchanged(const struct stat *before,const struct stat *after)
{
	if ( before->st_dev != after->st_dev || before->st_ino != after->st_ino || before->st_size != after->st_size )
		return(0);
#if defined(__APPLE__)
	return(before->st_mtimespec.tv_sec == after->st_mtimespec.tv_sec && before->st_mtimespec.tv_nsec == after->st_mtimespec.tv_nsec && before->st_ctimespec.tv_sec == after->st_ctimespec.tv_sec && before->st_ctimespec.tv_nsec == after->st_ctimespec.tv_nsec);
#else
	return(before->st_mtim.tv_sec == after->st_mtim.tv_sec && before->st_mtim.tv_nsec == after->st_mtim.tv_nsec && before->st_ctim.tv_sec == after->st_ctim.tv_sec && before->st_ctim.tv_nsec == after->st_ctim.tv_nsec);
#endif
}

SparkStatus SparkWeightdSpineLoad(int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *sha256,void *destination,uint64_t capacity)
{
	struct stat before,after;
	SparkStatus status;
	if ( fd < 0 || manifest == 0 || sha256 == 0 || SparkSha256HexIsValid(sha256) == 0 || pack_bytes == 0u || pack_bytes > INT64_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( manifest->spine_allocation_bytes > capacity || capacity > SIZE_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( manifest->spine_allocation_bytes != 0u && (destination == 0 || ((uintptr_t)destination & 255u) != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( fstat(fd,&before) != 0 || S_ISREG(before.st_mode) == 0 || before.st_size < 0 || (uint64_t)before.st_size != pack_bytes )
		return(SPARK_STATUS_IO_ERROR);
	status = spine_stream(fd,manifest,pack_bytes,sha256,destination);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( fstat(fd,&after) != 0 || spine_unchanged(&before,&after) == 0 )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}
