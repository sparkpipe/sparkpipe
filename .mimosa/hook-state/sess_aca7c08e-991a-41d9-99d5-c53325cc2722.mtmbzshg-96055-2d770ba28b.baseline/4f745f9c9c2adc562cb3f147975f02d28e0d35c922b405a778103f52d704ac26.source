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
        cursor += partition.byte_count;
    }

    if (!SparkTestExpect(cursor == total, "partition total coverage"))
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

    if (!SparkTestExpect(
            SparkMemlinkResolveNeighborRank(0u, 13u, SPARK_MEMLINK_NEIGHBOR_PREVIOUS, &rank) == SPARK_STATUS_OK,
            "previous rank resolve"))
    {
        return 0;
    }
    if (!SparkTestExpect(rank == 12u, "previous rank wraps"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkResolveNeighborRank(12u, 13u, SPARK_MEMLINK_NEIGHBOR_NEXT, &rank) == SPARK_STATUS_OK,
            "next rank resolve"))
    {
        return 0;
    }
    if (!SparkTestExpect(rank == 0u, "next rank wraps"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkFormatHostFromTemplate("spark%x", 12u, host, sizeof(host)) == SPARK_STATUS_OK,
            "hex host format"))
    {
        return 0;
    }
    if (!SparkTestExpect(strcmp(host, "sparkc") == 0, "hex host value"))
    {
        return 0;
    }

    if (!SparkTestExpect(
            SparkMemlinkResolveNeighborEndpoint(
                5u,
                13u,
                SPARK_MEMLINK_NEIGHBOR_NEXT,
                "spark%u.local",
                55200u,
                16u,
                &endpoint) == SPARK_STATUS_OK,
            "neighbor endpoint resolve"))
    {
        return 0;
    }
    if (!SparkTestExpect(strcmp(endpoint.host, "spark6.local") == 0, "endpoint host value"))
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
