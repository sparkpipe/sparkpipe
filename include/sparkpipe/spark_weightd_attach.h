#pragma once


#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_ATTACH_ENV_SOCKET "SPARK_WEIGHTD_SOCKET"
#define SPARK_WEIGHTD_ATTACH_ENV_SWITCH "SPARK_WEIGHTD_ATTACH"
#define SPARK_WEIGHTD_ATTACH_ENV_MODEL "SPARK_WEIGHTD_IDENTITY_MODEL"
#define SPARK_WEIGHTD_ATTACH_ENV_REVISION "SPARK_WEIGHTD_IDENTITY_REVISION"
#define SPARK_WEIGHTD_ATTACH_ENV_SHA256 "SPARK_WEIGHTD_PACK_SHA256"

#define SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS 120000000000ull

#define SPARK_WEIGHTD_ATTACH_REASON_BYTES 32u

typedef struct SparkWeightdPackSlice
{
    const char *model;
    const char *revision;
    uint32_t topology;
    uint64_t geometry_fingerprint;
    uint64_t pack_bytes;
} SparkWeightdPackSlice;

typedef struct SparkWeightdAttachOutcome
{
    SparkWeightdClient *client;
    uint64_t device_handle;
    uint64_t arena_bytes;
    uint64_t arena_generation;
    uint32_t loaded_from_pack;
    uint32_t refcount;
    void *map_base;
    uint64_t map_span_bytes;
    uint64_t map_chunk_bytes;
    void **map_handles;
    uint32_t map_chunk_count;
    uint32_t map_handle_count;
    uint32_t map_mapped_count;
} SparkWeightdAttachOutcome;

SparkStatus SparkWeightdAttachRequested(void);

SparkStatus SparkWeightdAttachPack(const SparkWeightdPackSlice *slice,
    const char *pack_path,
    uint64_t timeout_nanoseconds,
    SparkWeightdAttachOutcome *outcome,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES]);

SparkStatus SparkWeightdAttachImportMap(SparkWeightdAttachOutcome *outcome,
    uint64_t expected_arena_bytes,
    uint64_t timeout_nanoseconds,
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES]);

void SparkWeightdAttachRelease(SparkWeightdAttachOutcome *outcome);

#ifdef __cplusplus
}
#endif
