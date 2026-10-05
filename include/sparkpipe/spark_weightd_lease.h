#pragma once
#include "sparkpipe/spark_weightd_manifest.h"

#define SPARK_WEIGHTD_LEASE_COUNT_MAX 256u
#define SPARK_WEIGHTD_LEASE_GROUPS_MAX 512u
#define SPARK_WEIGHTD_LANE_NONE UINT32_MAX

typedef struct SparkWeightdExpertKey
{
	uint32_t layer;
	uint32_t expert;
} SparkWeightdExpertKey;

typedef struct SparkWeightdLease
{
	uint64_t identifier;
	uint64_t owner;
	uint32_t lane;
	uint32_t count;
	uint32_t groups[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
} SparkWeightdLease;

typedef struct SparkWeightdLeaseTable
{
	const SparkWeightdManifest *manifest;
	uint32_t *pins;
	uint64_t next_identifier;
	SparkWeightdLease leases[SPARK_WEIGHTD_LEASE_COUNT_MAX];
} SparkWeightdLeaseTable;

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkWeightdLeaseTableCreate(const SparkWeightdManifest *manifest,SparkWeightdLeaseTable **out);
SparkStatus SparkWeightdLeaseTableDestroy(SparkWeightdLeaseTable *table);
SparkStatus SparkWeightdLeaseAcquire(SparkWeightdLeaseTable *table,uint64_t owner,uint32_t lane,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier);
SparkStatus SparkWeightdLeaseReleaseForLane(SparkWeightdLeaseTable *table,uint32_t lane,uint32_t *released_count);
const SparkWeightdLease *SparkWeightdLeaseFind(const SparkWeightdLeaseTable *table,uint64_t owner,uint64_t identifier);
SparkStatus SparkWeightdLeaseRelease(SparkWeightdLeaseTable *table,uint64_t owner,uint64_t identifier);
SparkStatus SparkWeightdLeaseReleaseOwner(SparkWeightdLeaseTable *table,uint64_t owner);

SparkStatus SparkWeightdRouteKeys(uint32_t layer,const uint32_t *offsets,uint32_t expert_count,uint32_t packed_rows,SparkWeightdExpertKey *keys,uint32_t capacity,uint32_t *count);

#ifdef __cplusplus
}
#endif
