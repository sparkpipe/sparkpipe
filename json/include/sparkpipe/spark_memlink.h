#ifndef SPARKPIPE_SPARK_MEMLINK_H
#define SPARKPIPE_SPARK_MEMLINK_H

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MEMLINK_DEFAULT_BASE_PORT 55200u
#define SPARK_MEMLINK_DEFAULT_LANE_COUNT 8u
#define SPARK_MEMLINK_MAX_LANE_COUNT 64u
#define SPARK_MEMLINK_MAX_KEY_BYTES 240u
#define SPARK_MEMLINK_MAX_HOST_BYTES 256u
#define SPARK_MEMLINK_DEFAULT_STORE_BYTES (1024ull * 1024ull * 1024ull)
#define SPARK_MEMLINK_DEFAULT_IO_CHUNK_BYTES (1024ull * 1024ull)

typedef enum SparkMemlinkNeighborDirection
{
    SPARK_MEMLINK_NEIGHBOR_PREVIOUS = 0,
    SPARK_MEMLINK_NEIGHBOR_NEXT = 1
} SparkMemlinkNeighborDirection;

typedef struct SparkMemlinkTransferPartition
{
    uint32_t lane_index;
    uint32_t lane_count;
    uint64_t offset;
    uint64_t byte_count;
} SparkMemlinkTransferPartition;

typedef struct SparkMemlinkEndpoint
{
    char host[SPARK_MEMLINK_MAX_HOST_BYTES];
    uint16_t base_port;
    uint32_t lane_count;
} SparkMemlinkEndpoint;

SparkStatus SparkMemlinkValidateLaneCount(uint32_t lane_count);

SparkStatus SparkMemlinkBuildTransferPartition(
    uint64_t total_bytes,
    uint32_t lane_count,
    uint32_t lane_index,
    SparkMemlinkTransferPartition *partition);

SparkStatus SparkMemlinkResolveNeighborRank(
    uint32_t current_rank,
    uint32_t rank_count,
    SparkMemlinkNeighborDirection direction,
    uint32_t *neighbor_rank);

SparkStatus SparkMemlinkFormatHostFromTemplate(
    const char *host_template,
    uint32_t rank,
    char *host,
    size_t host_capacity);

SparkStatus SparkMemlinkResolveNeighborEndpoint(
    uint32_t current_rank,
    uint32_t rank_count,
    SparkMemlinkNeighborDirection direction,
    const char *host_template,
    uint16_t base_port,
    uint32_t lane_count,
    SparkMemlinkEndpoint *endpoint);

#ifdef __cplusplus
}
#endif

#endif
