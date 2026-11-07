#define _DARWIN_C_SOURCE
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_spine.h"
#include "sparkpipe/spark_weightd_receipt.h"
#include "sparkpipe/spark_weightd_direct.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

static SparkStatus spine_copy(const SparkWeightdManifest *manifest,uint32_t *index,const uint8_t *buffer,uint64_t offset,uint32_t bytes,uint8_t *destination,uint32_t pack_offsets)
{
	const SparkWeightdSpan *span;
	uint64_t start,end,placed,limit = (offset + bytes);
	while ( *index < manifest->spine_count )
	{
		span = &manifest->spine[*index];
		if ( span->offset >= limit )
			break;
		start = span->offset > offset ? span->offset : offset;
		end = (span->offset + span->bytes);
		if ( end > limit )
			end = limit;
		placed = pack_offsets != 0u ? span->offset : span->compact_offset;
		if ( end > start && cudaMemcpy(destination + placed + (start - span->offset),buffer + (start - offset),(size_t)(end - start),cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( (span->offset + span->bytes) > limit )
			break;
		(*index)++;
	}
	return(SPARK_STATUS_OK);
}

#define SPINE_STREAM_BLOCK_BYTES (UINT64_C(16) << 20)
#define SPINE_STREAM_READERS 4u

typedef struct SpineSink
{
	const SparkWeightdManifest *manifest;
	const SparkWeightdDirectSpan *spans;
	uint8_t *destination;
	SparkSha256Context *hash;
	SparkCk128Context *quick;
	SparkWeightdSpinePace pace;
	void *pace_context;
	uint32_t pack_offsets;
	uint32_t index;
} SpineSink;

static SparkStatus spine_sink(void *context,uint32_t span_index,uint64_t span_offset,const uint8_t *data,uint64_t bytes)
{
	SpineSink *sink = context;
	if ( sink->pace != 0 )
	{
		SparkStatus paced = sink->pace(sink->pace_context,bytes);
		if ( paced != SPARK_STATUS_OK )
			SPARK_RETURN(paced);
	}
	if ( sink->quick != 0 )
		SparkCk128Update(sink->quick,data,(size_t)bytes);
	if ( sink->hash != 0 )
		SparkSha256Update(sink->hash,data,(size_t)bytes);
	return(spine_copy(sink->manifest,&sink->index,data,sink->spans[span_index].offset + span_offset,(uint32_t)bytes,sink->destination,sink->pack_offsets));
}

static SparkStatus spine_stream_spans(int32_t fd,SpineSink *sink,const SparkWeightdDirectSpan *spans,uint32_t span_count)
{
	SparkWeightdDirect *direct = 0;
	SparkWeightdDirectStats stats;
	uint32_t is_direct = 0u;
	int32_t stream_fd;
	SparkStatus status;
	if ( span_count == 0u )
		return(SPARK_STATUS_OK);
	stream_fd = SparkWeightdDirectReopen(fd,&is_direct);
	if ( stream_fd < 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdDirectCreate(SPINE_STREAM_BLOCK_BYTES,SPINE_STREAM_READERS,&direct);
	if ( status == SPARK_STATUS_OK )
	{
		sink->spans = spans;
		status = SparkWeightdDirectStream(direct,stream_fd,is_direct,spans,span_count,spine_sink,sink,&stats);
		if ( status == SPARK_STATUS_OK && stats.wall_ns != 0u )
			fprintf(stderr,"weightd spine-stream bytes=%llu seconds=%.3f gbps=%.2f direct=%u readers=%u hashed=%u\n",(unsigned long long)stats.bytes_read,(double)stats.wall_ns / 1e9,(double)stats.bytes_read / (double)stats.wall_ns,stats.direct,stats.readers,sink->hash != 0 ? 1u : 0u);
	}
	SparkWeightdDirectDestroy(direct);
	(void)close(stream_fd);
	SPARK_RETURN(status);
}

static SparkWeightdDirectSpan *spine_direct_spans(const SparkWeightdManifest *manifest)
{
	SparkWeightdDirectSpan *spans = calloc((size_t)manifest->spine_count + 1u,sizeof(*spans));
	uint32_t index;
	if ( spans == 0 )
		return(0);
	for (index = 0u; index < manifest->spine_count; index++)
	{
		spans[index].offset = manifest->spine[index].offset;
		spans[index].bytes = manifest->spine[index].bytes;
	}
	return(spans);
}

static SparkStatus spine_stream(const char *pack_path,int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *expected,uint8_t *destination,char ck_hex[SPARK_CK128_HEX_BYTES])
{
	SparkSha256Context hash;
	SparkCk128Context quick;
	SparkWeightdDirectSpan whole;
	SparkWeightdDirectSpan *spans;
	uint8_t digest[32],ck[16];
	char hex[SPARK_SHA256_HEX_BYTES];
	const char *reason = "absent";
	SpineSink sink;
	SparkStatus status;
	ck_hex[0] = 0;
	memset(&sink,0,sizeof(sink));
	sink.manifest = manifest;
	sink.destination = destination;
	status = SparkWeightdReceiptCheck(pack_path,fd,expected,0,&reason);
	if ( status == SPARK_STATUS_OK )
	{
		spans = spine_direct_spans(manifest);
		if ( spans == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = spine_stream_spans(fd,&sink,spans,manifest->spine_count);
		free(spans);
		SPARK_RETURN(status);
	}
	fprintf(stderr,"weightd spine path=%s receipt=%s status=%s: full sha256 verification\n",pack_path,reason,SparkStatusToString(status));
	SparkCk128Initialize(&quick);
	SparkSha256Initialize(&hash);
	whole.offset = 0u;
	whole.bytes = pack_bytes;
	sink.hash = &hash;
	sink.quick = &quick;
	status = spine_stream_spans(fd,&sink,&whole,1u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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

SparkStatus SparkWeightdSpinePreload(int32_t fd,const SparkWeightdManifest *manifest,void *device_base,SparkWeightdSpinePace pace,void *pace_context)
{
	SparkWeightdDirectSpan *spans;
	SpineSink sink;
	SparkStatus status;
	if ( fd < 0 || manifest == 0 || device_base == 0 || pace == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	spans = spine_direct_spans(manifest);
	if ( spans == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&sink,0,sizeof(sink));
	sink.manifest = manifest;
	sink.destination = device_base;
	sink.pace = pace;
	sink.pace_context = pace_context;
	sink.pack_offsets = 1u;
	status = spine_stream_spans(fd,&sink,spans,manifest->spine_count);
	free(spans);
	SPARK_RETURN(status);
}
