#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t read32(const uint8_t *p)
{
	uint32_t value;
	memcpy(&value,p,sizeof(value));
	return(value);
}

static uint64_t read64(const uint8_t *p)
{
	uint64_t value;
	memcpy(&value,p,sizeof(value));
	return(value);
}

static int32_t compare_offset(const void *a,const void *b)
{
	const SparkWeightdRange *left = a,*right = b;
	return(left->offset < right->offset ? -1 : left->offset > right->offset);
}

static int32_t compare_group(const void *a,const void *b)
{
	const SparkWeightdRange *left = a,*right = b;
	if ( left->layer != right->layer )
		return(left->layer < right->layer ? -1 : 1);
	if ( left->expert != right->expert )
		return(left->expert < right->expert ? -1 : 1);
	return(left->kind < right->kind ? -1 : left->kind > right->kind);
}

void SparkWeightdManifestDestroy(SparkWeightdManifest *manifest)
{
	if ( manifest == 0 )
		return;
	free(manifest->ranges);
	free(manifest->groups);
	free(manifest->spine);
	memset(manifest,0,sizeof(*manifest));
}

static SparkStatus read_ranges(FILE *file,uint64_t pack_bytes,SparkWeightdManifest *out)
{
	uint8_t record[48];
	uint32_t i;
	SparkWeightdRange *range;
	for (i=0u; i<out->range_count; i++)
	{
		if ( fread(record,1u,sizeof(record),file) != sizeof(record) )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		range = &out->ranges[i];
		range->layer = read32(record);
		range->expert = read32(record + 4u);
		range->kind = read32(record + 8u);
		range->offset = read64(record + 16u);
		range->bytes = read64(record + 24u);
		memcpy(range->digest,record + 32u,16u);
		if ( read32(record + 12u) != 0u || range->bytes == 0u || range->bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( range->offset > pack_bytes || range->bytes > (pack_bytes - range->offset) )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	if ( fgetc(file) != EOF || ferror(file) != 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus build_spine(SparkWeightdManifest *out,uint64_t pack_bytes)
{
	uint64_t cursor = 0u,end,padding;
	uint32_t i;
	SparkWeightdSpan *span;
	out->spine = calloc(out->range_count + 1u,sizeof(*out->spine));
	if ( out->spine == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (i=0u; i<=out->range_count; i++)
	{
		end = i < out->range_count ? out->ranges[i].offset : pack_bytes;
		if ( end > cursor )
		{
			span = &out->spine[out->spine_count++];
			span->offset = cursor;
			span->bytes = (end - cursor);
			padding = ((cursor - out->spine_allocation_bytes) & UINT64_C(255));
			if ( padding > (UINT64_MAX - out->spine_allocation_bytes) )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			span->compact_offset = (out->spine_allocation_bytes + padding);
			if ( span->bytes > (UINT64_MAX - span->compact_offset) )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			out->spine_allocation_bytes = (span->compact_offset + span->bytes);
			out->spine_bytes += span->bytes;
		}
		if ( i < out->range_count )
			cursor = (out->ranges[i].offset + out->ranges[i].bytes);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus group_ranges(SparkWeightdManifest *out,uint64_t pack_bytes)
{
	SparkWeightdRange *range,*previous;
	SparkWeightdRangeGroup *group = 0;
	SparkStatus status;
	uint32_t i;
	qsort(out->ranges,out->range_count,sizeof(*out->ranges),compare_offset);
	for (i=1u; i<out->range_count; i++)
	{
		previous = &out->ranges[i - 1u];
		if ( out->ranges[i].offset < (previous->offset + previous->bytes) )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	status = build_spine(out,pack_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	qsort(out->ranges,out->range_count,sizeof(*out->ranges),compare_group);
	for (i=0u; i<out->range_count; i++)
	{
		range = &out->ranges[i];
		if ( group == 0 || group->layer != range->layer || group->expert != range->expert )
		{
			group = &out->groups[out->group_count++];
			group->layer = range->layer;
			group->expert = range->expert;
			group->first_range = i;
		}
		else if ( range->kind == out->ranges[i - 1u].kind )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		group->range_count++;
		if ( group->range_count > SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus load_manifest(const char *path,FILE *file,uint64_t pack_bytes,SparkWeightdManifest *out)
{
	uint8_t header[16];
	SparkStatus status;
	if ( fread(header,1u,sizeof(header),file) != sizeof(header) )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( read32(header) != SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC || read32(header + 4u) != SPARK_WEIGHTD_RANGE_MANIFEST_VERSION || read32(header + 12u) != 0u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	out->range_count = read32(header + 8u);
	if ( out->range_count > SPARK_WEIGHTD_RANGE_COUNT_MAX )
	{
		fprintf(stderr,"weightd manifest refused: %s has %u ranges, above SPARK_WEIGHTD_RANGE_COUNT_MAX %u (%llu table bytes per range, %llu-byte manifest table budget)\n",
		    path,out->range_count,SPARK_WEIGHTD_RANGE_COUNT_MAX,(unsigned long long)SPARK_WEIGHTD_MANIFEST_RANGE_TABLE_BYTES,(unsigned long long)SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	out->ranges = calloc(out->range_count + 1u,sizeof(*out->ranges));
	out->groups = calloc(out->range_count + 1u,sizeof(*out->groups));
	if ( out->ranges == 0 || out->groups == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = read_ranges(file,pack_bytes,out);
	if ( status == SPARK_STATUS_OK )
		status = group_ranges(out,pack_bytes);
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdManifestLoad(const char *path,uint64_t pack_bytes,SparkWeightdManifest *out)
{
	FILE *file;
	struct stat info;
	int32_t fd;
	SparkStatus status;
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(out,0,sizeof(*out));
	if ( path == 0 || pack_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	fd = open(path,O_RDONLY | O_NONBLOCK);
	if ( fd < 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	if ( fstat(fd,&info) != 0 || S_ISREG(info.st_mode) == 0 )
	{
		(void)close(fd);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	file = fdopen(fd,"rb");
	if ( file == 0 )
	{
		(void)close(fd);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = load_manifest(path,file,pack_bytes,out);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
		SparkWeightdManifestDestroy(out);
	SPARK_RETURN(status);
}

const SparkWeightdRangeGroup *SparkWeightdManifestFind(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert)
{
	const SparkWeightdRangeGroup *group;
	uint32_t low = 0u,high,middle;
	if ( manifest == 0 )
		return(0);
	high = manifest->group_count;
	while ( low < high )
	{
		middle = (low + ((high - low) / 2u));
		group = &manifest->groups[middle];
		if ( group->layer == layer && group->expert == expert )
			return(group);
		if ( group->layer < layer || (group->layer == layer && group->expert < expert) )
			low = (middle + 1u);
		else
			high = middle;
	}
	return(0);
}

SparkStatus SparkWeightdManifestSpineSlice(const SparkWeightdManifest *manifest,uint64_t offset,uint64_t bytes,uint64_t *compact_offset)
{
	const SparkWeightdSpan *span;
	uint32_t low = 0u,high,middle;
	uint64_t relative;
	if ( compact_offset == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*compact_offset = 0u;
	if ( manifest == 0 || bytes == 0u || bytes > (UINT64_MAX - offset) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	high = manifest->spine_count;
	while ( low < high )
	{
		middle = (low + ((high - low) / 2u));
		span = &manifest->spine[middle];
		if ( offset < span->offset )
			high = middle;
		else if ( (offset - span->offset) >= span->bytes )
			low = (middle + 1u);
		else
		{
			relative = (offset - span->offset);
			if ( bytes > (span->bytes - relative) )
				SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
			*compact_offset = (span->compact_offset + relative);
			return(SPARK_STATUS_OK);
		}
	}
	SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
}

typedef struct ChunkSpan
{
	uint64_t first;
	uint64_t last;
} ChunkSpan;

static int32_t compare_chunk_span(const void *a,const void *b)
{
	const ChunkSpan *left = a,*right = b;
	return(left->first < right->first ? -1 : left->first > right->first);
}

SparkStatus SparkWeightdManifestPoolBytes(const SparkWeightdManifest *manifest,uint64_t chunk_bytes,uint64_t *bytes_out)
{
	const SparkWeightdRange *range;
	ChunkSpan *spans;
	uint64_t count = 0u,next = 0u,first;
	uint32_t i;
	if ( bytes_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*bytes_out = 0u;
	if ( manifest == 0 || chunk_bytes == 0u || (manifest->range_count != 0u && manifest->ranges == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( manifest->range_count == 0u )
	{
		*bytes_out = chunk_bytes;
		return(SPARK_STATUS_OK);
	}
	spans = calloc(manifest->range_count,sizeof(*spans));
	if ( spans == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (i=0u; i<manifest->range_count; i++)
	{
		range = &manifest->ranges[i];
		if ( range->bytes == 0u || range->bytes > (UINT64_MAX - range->offset) )
		{
			free(spans);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		spans[i].first = (range->offset / chunk_bytes);
		spans[i].last = ((range->offset + range->bytes - 1u) / chunk_bytes);
	}
	qsort(spans,manifest->range_count,sizeof(*spans),compare_chunk_span);
	for (i=0u; i<manifest->range_count; i++)
	{
		first = spans[i].first < next ? next : spans[i].first;
		if ( spans[i].last >= first )
		{
			count += (spans[i].last - first + 1u);
			next = (spans[i].last + 1u);
		}
	}
	free(spans);
	if ( count > (UINT64_MAX / chunk_bytes) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	*bytes_out = (count * chunk_bytes);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdManifestPoolPlan(const SparkWeightdManifest *manifest,uint64_t arena_bytes,uint64_t cap_bytes,SparkWeightdPoolPlan *plan)
{
	const uint64_t chunk = SPARK_WEIGHTD_EXPERT_CHUNK_BYTES;
	char cap[24];
	SparkStatus status;
	if ( plan == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(plan,0,sizeof(*plan));
	if ( manifest == 0 || arena_bytes == 0u || arena_bytes > (UINT64_MAX - chunk) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkWeightdManifestPoolBytes(manifest,chunk,&plan->exact_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( cap_bytes > plan->exact_bytes )
	{
		fprintf(stderr,"weightd pool refused: cap=%llu is larger than exact=%llu, the %llu-byte chunks the pack's expert ranges touch; a larger cap only wastes memory\n",
		    (unsigned long long)cap_bytes,(unsigned long long)plan->exact_bytes,(unsigned long long)chunk);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( (cap_bytes % chunk) != 0u )
	{
		fprintf(stderr,"weightd pool refused: cap=%llu is not a whole number of %llu-byte chunks (exact=%llu)\n",
		    (unsigned long long)cap_bytes,(unsigned long long)chunk,(unsigned long long)plan->exact_bytes);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	plan->cap_bytes = cap_bytes;
	plan->request_bytes = cap_bytes != 0u ? cap_bytes : (manifest->range_count != 0u ? (arena_bytes + chunk) : plan->exact_bytes);
	plan->whole_arena = plan->request_bytes > arena_bytes ? 1u : 0u;
	plan->used_bytes = plan->whole_arena != 0u ? (((arena_bytes + chunk - 1u) / chunk) * chunk) : plan->request_bytes;
	if ( cap_bytes != 0u )
		(void)snprintf(cap,sizeof(cap),"%llu",(unsigned long long)cap_bytes);
	else
		(void)snprintf(cap,sizeof(cap),"none");
	fprintf(stderr,"weightd pool exact=%llu cap=%s used=%llu mode=%s\n",(unsigned long long)plan->exact_bytes,cap,
	    (unsigned long long)plan->used_bytes,plan->whole_arena != 0u ? "whole-arena" : "slots");
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdPackPoolPlan(const char *pack_path,uint64_t arena_bytes,uint64_t cap_bytes,SparkWeightdPoolPlan *plan)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u];
	SparkWeightdManifest manifest;
	SparkStatus status;
	if ( plan == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(plan,0,sizeof(*plan));
	if ( pack_path == 0 || pack_path[0] == 0 || strlen(pack_path) >= SPARK_WEIGHTD_PATH_BYTES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	(void)snprintf(path,sizeof(path),"%s.experts",pack_path);
	status = SparkWeightdManifestLoad(path,arena_bytes,&manifest);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"weightd pool refused: %s status=%d; the pool is sized from the pack's expert manifest\n",path,(int)status);
		SPARK_RETURN(status);
	}
	status = SparkWeightdManifestPoolPlan(&manifest,arena_bytes,cap_bytes,plan);
	SparkWeightdManifestDestroy(&manifest);
	SPARK_RETURN(status);
}
