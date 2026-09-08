#pragma once
#include <stdint.h>
#include "sparkpipe/spark_status.h"

// Version 2 groups all ranges of a logical expert; version 1 is not inferred.
#define SPARK_WEIGHTD_RANGE_MANIFEST_VERSION 2u
#define SPARK_WEIGHTD_RANGE_COUNT_MAX 131072u
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

typedef struct SparkWeightdManifest
{
	SparkWeightdRange *ranges;
	SparkWeightdRangeGroup *groups;
	uint32_t range_count;
	uint32_t group_count;
} SparkWeightdManifest;

// Startup allocation only. The caller owns a successful result until Destroy.
// Wire header: magic, version, range_count, zero (four little-endian u32).
// Each 48-byte record: layer, expert, kind, zero, offset, bytes, ck128[16].
SparkStatus SparkWeightdManifestLoad(const char *path,uint64_t pack_bytes,SparkWeightdManifest *out);
void SparkWeightdManifestDestroy(SparkWeightdManifest *manifest);
const SparkWeightdRangeGroup *SparkWeightdManifestFind(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert);
