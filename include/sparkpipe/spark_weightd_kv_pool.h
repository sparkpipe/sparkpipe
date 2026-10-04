#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_KV_POOL_TIMEOUT_DEFAULT_NS 60000000000ull

typedef struct SparkWeightdKvPoolMapping
{
    SparkWeightdClient *client;
    void *device_base;
    uint64_t device_bytes;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    uint32_t mapped_count;
    uint32_t reattached;
    uint32_t reserved0;
    uint64_t pool_generation;
    uint64_t kv_reserve_bytes;
    uint64_t kv_committed_bytes;
    uint64_t write_budget_bytes_per_day;
    uint8_t *metadata;
    uint64_t metadata_bytes;
    void *chunk_handles[SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX];
} SparkWeightdKvPoolMapping;

SparkStatus SparkWeightdKvPoolMap(const SparkWeightdKvPoolRequest *request,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping);
void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping);

#ifdef __cplusplus
}
#endif
