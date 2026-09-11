#pragma once
#include "sparkpipe/spark_weightd_manifest.h"

#define SPARK_WEIGHTD_LEASE_COUNT_MAX 64u
#define SPARK_WEIGHTD_LEASE_GROUPS_MAX 512u

typedef struct SparkWeightdExpertKey
{
	uint32_t layer;
	uint32_t expert;
} SparkWeightdExpertKey;

typedef struct SparkWeightdLease
{
	uint64_t identifier;
	uint64_t owner;
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

// Startup allocation; the manifest must outlive the table. Calls are serialized
// by the daemon thread. Owner identifiers must not be reused for new connections.
SparkStatus SparkWeightdLeaseTableCreate(const SparkWeightdManifest *manifest,SparkWeightdLeaseTable **out);
SparkStatus SparkWeightdLeaseTableDestroy(SparkWeightdLeaseTable *table);
SparkStatus SparkWeightdLeaseAcquire(SparkWeightdLeaseTable *table,uint64_t owner,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier);
const SparkWeightdLease *SparkWeightdLeaseFind(const SparkWeightdLeaseTable *table,uint64_t owner,uint64_t identifier);
// Release only after consumer GPU completion and unmapping are established.
// Connection loss alone is not permission to release an in-flight lease.
SparkStatus SparkWeightdLeaseRelease(SparkWeightdLeaseTable *table,uint64_t owner,uint64_t identifier);

// Completed host routing offsets, length expert_count + 1. Validate the full
// prefix before emitting keys. No allocation; failures leave count zero.
SparkStatus SparkWeightdRouteKeys(uint32_t layer,const uint32_t *offsets,uint32_t expert_count,uint32_t packed_rows,SparkWeightdExpertKey *keys,uint32_t capacity,uint32_t *count);

#ifdef __cplusplus
}
#endif
