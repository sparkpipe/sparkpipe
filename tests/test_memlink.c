#include "sparkpipe/spark_memlink.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int SparkTestExpect(int condition, const char *message)
{
    if (!condition)
    {
        fprintf(stderr, "test_memlink failed: %s\n", message);
        return 0;
    }
    return 1;
}

static int SparkTestPartitions(void)
{
    SparkMemlinkTransferPartition partition;
    uint64_t total;
    uint64_t cursor;
    uint32_t lane;

    total = 1003ull;
    cursor = 0ull;
    for (lane = 0u; lane < 8u; ++lane)
    {
        if (!SparkTestExpect(
                SparkMemlinkBuildTransferPartition(total, 8u, lane, &partition) == SPARK_STATUS_OK,
                "partition build"))
        {
            return 0;
        }
        if (!SparkTestExpect(partition.offset == cursor, "partition offset continuity"))
        {
            return 0;
        }
        if (!SparkTestExpect(partition.lane_index == lane && partition.lane_count == 8u, "partition lane identity"))
        {
            return 0;
        }
        if (!SparkTestExpect(partition.byte_count == (lane < 3u ? 126ull : 125ull), "partition balanced with remainder first"))
        {
            return 0;
        }
        cursor += partition.byte_count;
    }

    if (!SparkTestExpect(cursor == total, "partition total coverage"))
    {
        return 0;
    }

    cursor = 0ull;
    for (lane = 0u; lane < SPARK_MEMLINK_MAX_LANE_COUNT; ++lane)
    {
        if (!SparkTestExpect(
                SparkMemlinkBuildTransferPartition(40ull, SPARK_MEMLINK_MAX_LANE_COUNT, lane, &partition) == SPARK_STATUS_OK &&
                    partition.offset == cursor && partition.byte_count == (lane < 40u ? 1ull : 0ull),
                "maximum lanes split fewer bytes than lanes"))
        {
            return 0;
        }
        cursor += partition.byte_count;
    }
    if (!SparkTestExpect(cursor == 40ull, "maximum lanes total coverage"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkValidateLaneCount(SPARK_MEMLINK_MAX_LANE_COUNT) == SPARK_STATUS_OK &&
                SparkMemlinkValidateLaneCount(SPARK_MEMLINK_MAX_LANE_COUNT + 1u) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkMemlinkValidateLaneCount(0u) == SPARK_STATUS_INVALID_ARGUMENT,
            "lane count bounds"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkBuildTransferPartition(total, SPARK_MEMLINK_MAX_LANE_COUNT + 1u, 0u, &partition) == SPARK_STATUS_INVALID_ARGUMENT,
            "lanes above maximum rejected"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkBuildTransferPartition(total, 0u, 0u, &partition) == SPARK_STATUS_INVALID_ARGUMENT,
            "zero lanes rejected"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkBuildTransferPartition(total, 8u, 8u, &partition) == SPARK_STATUS_INVALID_ARGUMENT,
            "lane out of range rejected"))
    {
        return 0;
    }

    return 1;
}

static int SparkTestNeighborResolution(void)
{
    SparkMemlinkEndpoint endpoint;
    char host[SPARK_MEMLINK_MAX_HOST_BYTES];
    uint32_t rank;

    static const char *const fleet_hosts[16] = {
        "spark0", "spark1", "spark2", "spark3", "spark4", "spark5", "spark6", "spark7",
        "spark8", "spark9", "sparka", "sparkb", "sparkc", "sparkd", "sparke", "sparkf"};

    for (rank = 0u; rank < 16u; ++rank)
    {
        uint32_t previous;
        uint32_t next;

        if (!SparkTestExpect(
                SparkMemlinkResolveNeighborRank(rank, 16u, SPARK_MEMLINK_NEIGHBOR_PREVIOUS, &previous) == SPARK_STATUS_OK &&
                    SparkMemlinkResolveNeighborRank(rank, 16u, SPARK_MEMLINK_NEIGHBOR_NEXT, &next) == SPARK_STATUS_OK,
                "fleet neighbor resolve"))
        {
            return 0;
        }
        if (!SparkTestExpect(previous == (rank + 15u) % 16u && next == (rank + 1u) % 16u, "fleet ring neighbors wrap"))
        {
            return 0;
        }
        if (!SparkTestExpect(
                SparkMemlinkFormatHostFromTemplate("spark%x", rank, host, sizeof(host)) == SPARK_STATUS_OK &&
                    strcmp(host, fleet_hosts[rank]) == 0,
                "hex host template names every fleet node"))
        {
            return 0;
        }
    }

    if (!SparkTestExpect(
            SparkMemlinkResolveNeighborRank(16u, 16u, SPARK_MEMLINK_NEIGHBOR_NEXT, &rank) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkMemlinkResolveNeighborRank(0u, 0u, SPARK_MEMLINK_NEIGHBOR_NEXT, &rank) == SPARK_STATUS_INVALID_ARGUMENT,
            "rank outside the ring rejected"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkFormatHostFromTemplate("spark%x%u", 1u, host, sizeof(host)) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkMemlinkFormatHostFromTemplate("spark%s", 1u, host, sizeof(host)) == SPARK_STATUS_INVALID_ARGUMENT &&
                SparkMemlinkFormatHostFromTemplate("spark%x", 15u, host, 6u) == SPARK_STATUS_CAPACITY_EXCEEDED,
            "host template refuses ambiguous placeholders and truncation"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkResolveNeighborEndpoint(
                15u,
                16u,
                SPARK_MEMLINK_NEIGHBOR_NEXT,
                "spark%u.local",
                55200u,
                16u,
                &endpoint) == SPARK_STATUS_OK,
            "neighbor endpoint resolve"))
    {
        return 0;
    }
    if (!SparkTestExpect(strcmp(endpoint.host, "spark0.local") == 0, "endpoint host wraps to rank 0"))
    {
        return 0;
    }
    if (!SparkTestExpect(endpoint.base_port == 55200u, "endpoint port value"))
    {
        return 0;
    }
    if (!SparkTestExpect(endpoint.lane_count == 16u, "endpoint lane value"))
    {
        return 0;
    }

    return 1;
}

int main(void)
{
    if (!SparkTestPartitions())
    {
        return 1;
    }
    if (!SparkTestNeighborResolution())
    {
        return 1;
    }

    return 0;
}
