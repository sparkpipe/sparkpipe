#define _DARWIN_C_SOURCE
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_spine.h"
#include "sparkpipe/spark_weightd_receipt.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
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
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( (span->offset + span->bytes) > limit )
			break;
		(*index)++;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus spine_copy_spans(int32_t fd,const SparkWeightdManifest *manifest,uint8_t *buffer,uint32_t capacity,uint8_t *destination)
{
	uint64_t span_offset;
	uint32_t bytes,index,copy_index = 0u;
	SparkStatus status;
	for (index = 0u; index < manifest->spine_count; index++)
	{
		const SparkWeightdSpan *span = &manifest->spine[index];
		span_offset = span->offset;
		while ( span_offset < span->offset + span->bytes )
		{
			uint64_t remain = span->offset + span->bytes - span_offset;
			bytes = (uint32_t)(remain < capacity ? remain : capacity);
			status = spine_read(fd,buffer,span_offset,bytes);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			status = spine_copy(manifest,&copy_index,buffer,span_offset,bytes,destination);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			span_offset += bytes;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus spine_stream(const char *pack_path,int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *expected,uint8_t *destination,char ck_hex[SPARK_CK128_HEX_BYTES])
{
	SparkSha256Context hash;
	SparkCk128Context quick;
	static _Thread_local uint8_t buffer[1048576];
	uint8_t digest[32],ck[16];
	char hex[SPARK_SHA256_HEX_BYTES];
	const char *reason = "absent";
	uint64_t offset = 0u;
	uint32_t bytes,index = 0u;
	SparkStatus status;
	ck_hex[0] = 0;
	status = SparkWeightdReceiptCheck(pack_path,fd,expected,0,&reason);
	if ( status == SPARK_STATUS_OK )
		return(spine_copy_spans(fd,manifest,buffer,(uint32_t)sizeof(buffer),destination));
	fprintf(stderr,"weightd spine path=%s receipt=%s status=%s: full sha256 verification\n",pack_path,reason,SparkStatusToString(status));
	SparkCk128Initialize(&quick);
	SparkSha256Initialize(&hash);
	while ( offset < pack_bytes )
	{
		bytes = (uint32_t)((pack_bytes - offset) < sizeof(buffer) ? (pack_bytes - offset) : sizeof(buffer));
		status = spine_read(fd,buffer,offset,bytes);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkCk128Update(&quick,buffer,bytes);
		SparkSha256Update(&hash,buffer,bytes);
		status = spine_copy(manifest,&index,buffer,offset,bytes,destination);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		offset += bytes;
	}
	SparkSha256Finalize(&hash,digest);
	SparkSha256DigestToHex(digest,hex);
	if ( strcmp(hex,expected) != 0 )
	{
		fprintf(stderr,"weightd spine HASH MISMATCH path=%s expected=%s computed=%s\n",pack_path,expected,hex);
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	SparkCk128Finalize(&quick,ck);
	SparkCk128DigestToHex(ck,ck_hex);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdSpineLoad(const char *pack_path,int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *sha256,void *destination,uint64_t capacity)
{
	struct stat before,after;
	char ck_hex[SPARK_CK128_HEX_BYTES];
	SparkStatus status;
	if ( pack_path == 0 || fd < 0 || manifest == 0 || sha256 == 0 || SparkSha256HexIsValid(sha256) == 0 || pack_bytes == 0u || pack_bytes > INT64_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( manifest->spine_allocation_bytes > capacity || capacity > SIZE_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( manifest->spine_allocation_bytes != 0u && (destination == 0 || ((uintptr_t)destination & 255u) != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( fstat(fd,&before) != 0 || S_ISREG(before.st_mode) == 0 || before.st_size < 0 || (uint64_t)before.st_size != pack_bytes )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = spine_stream(pack_path,fd,manifest,pack_bytes,sha256,destination,ck_hex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( fstat(fd,&after) != 0 || SparkWeightdPackStatSame(&before,&after) == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	if ( ck_hex[0] != 0 )
	{
		status = SparkWeightdReceiptRecord(pack_path,fd,&before,sha256,ck_hex,"client");
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"weightd spine path=%s receipt not written status=%s: the next load verifies again\n",pack_path,SparkStatusToString(status));
	}
	return(SPARK_STATUS_OK);
}
