#pragma once
#include <stdint.h>
#include "sparkpipe/spark_status.h"

#define SPARK_WEIGHTD_RANGE_MANIFEST_VERSION 2u
#define SPARK_WEIGHTD_RANGE_COUNT_MAX 262144u
#define SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX (32ull * 1024ull * 1024ull)
#define SPARK_WEIGHTD_MANIFEST_GROUP_STATE_BYTES_MAX 32u
#define SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX 16u

typedef struct SparkWeightdRange
{
	uint64_t offset;
	uint64_t bytes;
	uint8_t digest[16];
	uint32_t layer;
	uint32_t expert;
	uint32_t kind;
} SparkWeightdRange;

typedef struct SparkWeightdRangeGroup
{
	uint32_t layer;
	uint32_t expert;
	uint32_t first_range;
	uint32_t range_count;
} SparkWeightdRangeGroup;

typedef struct SparkWeightdSpan
{
	uint64_t offset;
	uint64_t bytes;
	uint64_t compact_offset;
} SparkWeightdSpan;

typedef struct SparkWeightdManifest
{
	SparkWeightdRange *ranges;
	SparkWeightdRangeGroup *groups;
	uint32_t range_count;
	uint32_t group_count;
	SparkWeightdSpan *spine;
	uint64_t spine_bytes;
	uint32_t spine_count;
	uint64_t spine_allocation_bytes;
} SparkWeightdManifest;

#define SPARK_WEIGHTD_MANIFEST_RANGE_TABLE_BYTES (sizeof(SparkWeightdRange) + sizeof(SparkWeightdRangeGroup) + sizeof(SparkWeightdSpan) + SPARK_WEIGHTD_MANIFEST_GROUP_STATE_BYTES_MAX)

#if !defined(__cplusplus)
_Static_assert((uint64_t)SPARK_WEIGHTD_RANGE_COUNT_MAX * SPARK_WEIGHTD_MANIFEST_RANGE_TABLE_BYTES <= SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX,
    "SPARK_WEIGHTD_RANGE_COUNT_MAX must keep one manifest's host tables inside SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX");
_Static_assert((uint64_t)SPARK_WEIGHTD_RANGE_COUNT_MAX * 2u * SPARK_WEIGHTD_MANIFEST_RANGE_TABLE_BYTES > SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX,
    "SPARK_WEIGHTD_RANGE_COUNT_MAX is the largest power of two inside SPARK_WEIGHTD_MANIFEST_TABLE_BYTES_MAX");
_Static_assert((SPARK_WEIGHTD_RANGE_COUNT_MAX & (SPARK_WEIGHTD_RANGE_COUNT_MAX - 1u)) == 0u,
    "SPARK_WEIGHTD_RANGE_COUNT_MAX is a power of two");
#endif

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkWeightdManifestLoad(const char *path,uint64_t pack_bytes,SparkWeightdManifest *out);
void SparkWeightdManifestDestroy(SparkWeightdManifest *manifest);
const SparkWeightdRangeGroup *SparkWeightdManifestFind(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert);

SparkStatus SparkWeightdManifestSpineSlice(const SparkWeightdManifest *manifest,uint64_t offset,uint64_t bytes,uint64_t *compact_offset);

#ifdef __cplusplus
}
#endif
