#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd_direct.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct BenchRange
{
	uint64_t offset;
	uint64_t bytes;
	const uint8_t *digest;
} BenchRange;

typedef struct BenchSink
{
	const BenchRange *ranges;
	SparkCk128Context context;
	uint64_t verified;
	uint64_t mismatches;
	uint32_t verify;
} BenchSink;

static int compare_range(const void *left,const void *right)
{
	const BenchRange *a = left,*b = right;
	return(a->offset < b->offset ? -1 : a->offset > b->offset);
}

static SparkStatus bench_sink(void *context,uint32_t span_index,uint64_t span_offset,const uint8_t *data,uint64_t bytes)
{
	BenchSink *sink = context;
	const BenchRange *range = &sink->ranges[span_index];
	uint8_t digest[16];
	if ( sink->verify == 0u )
		return(SPARK_STATUS_OK);
	if ( span_offset == 0u )
		SparkCk128Initialize(&sink->context);
	SparkCk128Update(&sink->context,data,(size_t)bytes);
	if ( span_offset + bytes == range->bytes && range->digest != 0 )
	{
		SparkCk128Finalize(&sink->context,digest);
		if ( memcmp(digest,range->digest,sizeof(digest)) != 0 )
			sink->mismatches++;
		sink->verified++;
	}
	return(SPARK_STATUS_OK);
}

int main(int argc,char **argv)
{
	SparkWeightdManifest manifest;
	SparkWeightdDirect *direct = 0;
	SparkWeightdDirectSpan *spans;
	SparkWeightdDirectStats stats;
	BenchRange *ranges;
	BenchSink sink;
	struct stat info;
	char manifest_path[4096];
	uint32_t index,count,is_direct = 0u,readers;
	uint64_t block_bytes;
	int32_t fd;
	SparkStatus status;
	if ( argc != 7 )
	{
		fprintf(stderr,"usage: %s PACK experts|spine|whole direct|buffered READERS BLOCK_MIB verify|noverify\n",argv[0]);
		return(2);
	}
	readers = (uint32_t)strtoul(argv[4],0,10);
	block_bytes = (uint64_t)strtoull(argv[5],0,10) << 20;
	if ( stat(argv[1],&info) != 0 || info.st_size <= 0 )
	{
		fprintf(stderr,"pack_stream_bench: cannot stat %s\n",argv[1]);
		return(1);
	}
	(void)snprintf(manifest_path,sizeof(manifest_path),"%s.experts",argv[1]);
	memset(&manifest,0,sizeof(manifest));
	if ( strcmp(argv[2],"whole") != 0 && SparkWeightdManifestLoad(manifest_path,(uint64_t)info.st_size,&manifest) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"pack_stream_bench: cannot load %s\n",manifest_path);
		return(1);
	}
	count = strcmp(argv[2],"experts") == 0 ? manifest.range_count : strcmp(argv[2],"spine") == 0 ? manifest.spine_count : 1u;
	ranges = calloc(count,sizeof(*ranges));
	spans = calloc(count,sizeof(*spans));
	if ( ranges == 0 || spans == 0 )
		return(1);
	for (index = 0u; index < count; index++)
	{
		if ( strcmp(argv[2],"experts") == 0 )
		{
			ranges[index].offset = manifest.ranges[index].offset;
			ranges[index].bytes = manifest.ranges[index].bytes;
			ranges[index].digest = manifest.ranges[index].digest;
		}
		else if ( strcmp(argv[2],"spine") == 0 )
		{
			ranges[index].offset = manifest.spine[index].offset;
			ranges[index].bytes = manifest.spine[index].bytes;
		}
		else
		{
			ranges[index].offset = 0u;
			ranges[index].bytes = (uint64_t)info.st_size;
		}
	}
	qsort(ranges,count,sizeof(*ranges),compare_range);
	for (index = 0u; index < count; index++)
	{
		spans[index].offset = ranges[index].offset;
		spans[index].bytes = ranges[index].bytes;
	}
	if ( strcmp(argv[3],"direct") == 0 )
		fd = SparkWeightdDirectOpen(argv[1],&is_direct);
	else
		fd = open(argv[1],O_RDONLY);
	if ( fd < 0 || SparkWeightdDirectCreate(block_bytes,readers,&direct) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"pack_stream_bench: open or create failed\n");
		return(1);
	}
	memset(&sink,0,sizeof(sink));
	sink.ranges = ranges;
	sink.verify = strcmp(argv[6],"verify") == 0 ? 1u : 0u;
	status = SparkWeightdDirectStream(direct,fd,is_direct,spans,count,bench_sink,&sink,&stats);
	printf("pack_stream_bench what=%s mode=%s direct=%u readers=%u block_mib=%llu spans=%u blocks=%llu bytes_read=%llu seconds=%.3f gbps=%.2f sink_seconds=%.3f verified=%llu mismatches=%llu status=%d\n",
		argv[2],argv[3],stats.direct,stats.readers,(unsigned long long)(block_bytes >> 20),count,
		(unsigned long long)stats.blocks,(unsigned long long)stats.bytes_read,(double)stats.wall_ns / 1e9,
		(double)stats.bytes_read / ((double)stats.wall_ns + 1.0),(double)stats.sink_ns / 1e9,
		(unsigned long long)sink.verified,(unsigned long long)sink.mismatches,(int)status);
	SparkWeightdDirectDestroy(direct);
	close(fd);
	free(spans);
	free(ranges);
	SparkWeightdManifestDestroy(&manifest);
	return(status == SPARK_STATUS_OK && sink.mismatches == 0u ? 0 : 1);
}
