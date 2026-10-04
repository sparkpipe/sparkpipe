#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS 60000000000ull
#define SPARK_WEIGHTD_KV_POOL_MINIMUM_POLL_NS 50000000ull

typedef struct SparkWeightdKvPoolMapping
{
    SparkWeightdClient *client;
    void *device_base;
    uint64_t device_bytes;
    uint64_t chunk_bytes;
    uint32_t chunk_capacity;
    uint32_t chunk_count;
    uint32_t mapped_count;
    uint32_t reattached;
    uint64_t pool_generation;
    uint64_t kv_reserve_bytes;
    uint64_t kv_committed_bytes;
    uint64_t write_budget_bytes_per_day;
    uint8_t *metadata;
    uint64_t metadata_bytes;
    uint64_t *chunk_offsets;
    void **chunk_handles;
    int device;
} SparkWeightdKvPoolMapping;

SparkStatus SparkWeightdKvPoolGranularity(uint64_t *bytes);
SparkStatus SparkWeightdKvPoolMap(const SparkWeightdKvPoolRequest *request,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping);
SparkStatus SparkWeightdKvPoolGrow(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdKvPoolShrink(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdKvPoolStatus(SparkWeightdKvPoolMapping *mapping,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping);

#ifdef __cplusplus
}
#endif
