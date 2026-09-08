#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_weightd_manifest.h"
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
			return(SPARK_STATUS_PARSE_ERROR);
		range = &out->ranges[i];
		range->layer = read32(record);
		range->expert = read32(record + 4u);
		range->kind = read32(record + 8u);
		range->offset = read64(record + 16u);
		range->bytes = read64(record + 24u);
		memcpy(range->digest,record + 32u,16u);
		if ( read32(record + 12u) != 0u || range->bytes == 0u || range->bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX )
			return(SPARK_STATUS_PARSE_ERROR);
		if ( range->offset > pack_bytes || range->bytes > (pack_bytes - range->offset) )
			return(SPARK_STATUS_PARSE_ERROR);
	}
	if ( fgetc(file) != EOF || ferror(file) != 0 )
		return(SPARK_STATUS_PARSE_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus build_spine(SparkWeightdManifest *out,uint64_t pack_bytes)
{
	uint64_t cursor = 0u,end;
	uint32_t i;
	SparkWeightdSpan *span;
	out->spine = calloc(out->range_count + 1u,sizeof(*out->spine));
	if ( out->spine == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (i=0u; i<=out->range_count; i++)
	{
		end = i < out->range_count ? out->ranges[i].offset : pack_bytes;
		if ( end > cursor )
		{
			span = &out->spine[out->spine_count++];
			span->offset = cursor;
			span->bytes = (end - cursor);
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
			return(SPARK_STATUS_PARSE_ERROR);
	}
	status = build_spine(out,pack_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
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
			return(SPARK_STATUS_PARSE_ERROR);
		group->range_count++;
		if ( group->range_count > SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus load_manifest(FILE *file,uint64_t pack_bytes,SparkWeightdManifest *out)
{
	uint8_t header[16];
	SparkStatus status;
	if ( fread(header,1u,sizeof(header),file) != sizeof(header) )
		return(SPARK_STATUS_PARSE_ERROR);
	if ( read32(header) != SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC || read32(header + 4u) != SPARK_WEIGHTD_RANGE_MANIFEST_VERSION || read32(header + 12u) != 0u )
		return(SPARK_STATUS_PARSE_ERROR);
	out->range_count = read32(header + 8u);
	if ( out->range_count == 0u || out->range_count > SPARK_WEIGHTD_RANGE_COUNT_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	out->ranges = calloc(out->range_count,sizeof(*out->ranges));
	out->groups = calloc(out->range_count,sizeof(*out->groups));
	if ( out->ranges == 0 || out->groups == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = read_ranges(file,pack_bytes,out);
	if ( status == SPARK_STATUS_OK )
		status = group_ranges(out,pack_bytes);
	return(status);
}

SparkStatus SparkWeightdManifestLoad(const char *path,uint64_t pack_bytes,SparkWeightdManifest *out)
{
	FILE *file;
	struct stat info;
	int32_t fd;
	SparkStatus status;
	if ( out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(out,0,sizeof(*out));
	if ( path == 0 || pack_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	fd = open(path,O_RDONLY | O_NONBLOCK);
	if ( fd < 0 )
		return(errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR);
	if ( fstat(fd,&info) != 0 || S_ISREG(info.st_mode) == 0 )
	{
		(void)close(fd);
		return(SPARK_STATUS_PARSE_ERROR);
	}
	file = fdopen(fd,"rb");
	if ( file == 0 )
	{
		(void)close(fd);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = load_manifest(file,pack_bytes,out);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
		SparkWeightdManifestDestroy(out);
	return(status);
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
