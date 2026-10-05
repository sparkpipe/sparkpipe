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
    uint32_t external_reservation;
} SparkWeightdKvPoolMapping;

typedef struct SparkWeightdKvSharedMapping
{
    SparkWeightdClient *client;
    void *device_base;
    uint64_t pool_generation;
    uint64_t chunk_bytes;
    uint64_t device_bytes;
    uint64_t metadata_bytes;
    uint32_t chunk_count;
    uint32_t mapped_count;
    uint32_t slot_count;
    uint32_t holder;
    uint32_t created;
    uint8_t *metadata;
    uint64_t *chunk_offsets;
    void **chunk_handles;
    int device;
} SparkWeightdKvSharedMapping;

SparkStatus SparkWeightdKvPoolGranularity(uint64_t *bytes);
SparkStatus SparkWeightdKvPoolMap(const SparkWeightdKvPoolRequest *request,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds,SparkWeightdKvPoolMapping *mapping);
SparkStatus SparkWeightdKvPoolGrow(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdKvPoolShrink(SparkWeightdKvPoolMapping *mapping,uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdKvPoolStatus(SparkWeightdKvPoolMapping *mapping,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
void SparkWeightdKvPoolUnmap(SparkWeightdKvPoolMapping *mapping);
SparkStatus SparkWeightdKvSharedAttach(const SparkWeightdKvSharedRequest *request,uint64_t timeout_nanoseconds,SparkWeightdKvSharedMapping *mapping);
SparkStatus SparkWeightdKvSharedMapChunks(SparkWeightdKvSharedMapping *mapping,void *device_base,const uint64_t *chunk_offsets,uint64_t timeout_nanoseconds);
void SparkWeightdKvSharedUnmap(SparkWeightdKvSharedMapping *mapping);
SparkStatus SparkWeightdKvReserveAddress(uint64_t bytes,void **base_out);
void SparkWeightdKvFreeAddress(void *base,uint64_t bytes);

#ifdef __cplusplus
}
#endif
