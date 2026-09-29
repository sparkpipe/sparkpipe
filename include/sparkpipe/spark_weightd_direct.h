#pragma once
#include <stdint.h>
#include "sparkpipe/spark_status.h"

#define SPARK_WEIGHTD_DIRECT_ALIGNMENT UINT64_C(4096)
#define SPARK_WEIGHTD_DIRECT_READERS_MAX 8u
#define SPARK_WEIGHTD_DIRECT_GAP_BYTES (UINT64_C(1) << 20)

typedef struct SparkWeightdDirectSpan
{
	uint64_t offset;
	uint64_t bytes;
} SparkWeightdDirectSpan;

typedef struct SparkWeightdDirectStats
{
	uint64_t bytes_read;
	uint64_t blocks;
	uint64_t wall_ns;
	uint64_t sink_ns;
	uint32_t direct;
	uint32_t readers;
} SparkWeightdDirectStats;

typedef SparkStatus (*SparkWeightdDirectSink)(void *context,uint32_t span_index,uint64_t span_offset,const uint8_t *data,uint64_t bytes);

typedef struct SparkWeightdDirect SparkWeightdDirect;

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkWeightdDirectCreate(uint64_t block_bytes,uint32_t readers,SparkWeightdDirect **out);
void SparkWeightdDirectDestroy(SparkWeightdDirect *direct);
int32_t SparkWeightdDirectOpen(const char *path,uint32_t *is_direct);
int32_t SparkWeightdDirectReopen(int32_t fd,uint32_t *is_direct);
SparkStatus SparkWeightdDirectStream(SparkWeightdDirect *direct,int32_t fd,uint32_t is_direct,const SparkWeightdDirectSpan *spans,uint32_t span_count,SparkWeightdDirectSink sink,void *context,SparkWeightdDirectStats *stats);

#ifdef __cplusplus
}
#endif
