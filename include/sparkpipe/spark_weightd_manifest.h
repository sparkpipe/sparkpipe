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

typedef struct SparkWeightdSpan
{
	uint64_t offset;
	uint64_t bytes;
	uint64_t compact_offset;
} SparkWeightdSpan;

// Spine spans are the sorted exact complement of expert ranges in the pack.
// They include headers/padding, carry no per-span checksum, and require pack
// identity validation by the loader before publication.
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

// Startup allocation only. The caller owns a successful result until Destroy.
// Wire header: magic, version, range_count, zero (four little-endian u32).
// Each 48-byte record: layer, expert, kind, zero, offset, bytes, ck128[16].
SparkStatus SparkWeightdManifestLoad(const char *path,uint64_t pack_bytes,SparkWeightdManifest *out);
void SparkWeightdManifestDestroy(SparkWeightdManifest *manifest);
const SparkWeightdRangeGroup *SparkWeightdManifestFind(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t expert);

// Translate a wholly non-expert slice in O(log N). Compact offsets preserve
// source alignment modulo 256; allocation size includes alignment padding.
SparkStatus SparkWeightdManifestSpineSlice(const SparkWeightdManifest *manifest,uint64_t offset,uint64_t bytes,uint64_t *compact_offset);
