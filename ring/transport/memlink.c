#include "sparkpipe/spark_memlink.h"

#include <stdio.h>
#include <string.h>

SparkStatus SparkMemlinkValidateLaneCount(uint32_t lane_count)
{
    if (lane_count == 0u || lane_count > SPARK_MEMLINK_MAX_LANE_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkMemlinkBuildTransferPartition(
    uint64_t total_bytes,
    uint32_t lane_count,
    uint32_t lane_index,
    SparkMemlinkTransferPartition *partition)
{
    uint64_t base_bytes;
    uint64_t extra_bytes;
    uint64_t offset;

    if (partition == NULL || SparkMemlinkValidateLaneCount(lane_count) != SPARK_STATUS_OK || lane_index >= lane_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    base_bytes = total_bytes / (uint64_t)lane_count;
    extra_bytes = total_bytes % (uint64_t)lane_count;

    if ((uint64_t)lane_index < extra_bytes)
    {
        partition->byte_count = base_bytes + 1ull;
        offset = ((base_bytes + 1ull) * (uint64_t)lane_index);
    }
    else
    {
        partition->byte_count = base_bytes;
        offset = ((base_bytes + 1ull) * extra_bytes) + (base_bytes * ((uint64_t)lane_index - extra_bytes));
    }

    partition->lane_index = lane_index;
    partition->lane_count = lane_count;
    partition->offset = offset;

    return SPARK_STATUS_OK;
}

SparkStatus SparkMemlinkResolveNeighborRank(
    uint32_t current_rank,
    uint32_t rank_count,
    SparkMemlinkNeighborDirection direction,
    uint32_t *neighbor_rank)
{
    if (neighbor_rank == NULL || rank_count == 0u || current_rank >= rank_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (direction == SPARK_MEMLINK_NEIGHBOR_PREVIOUS)
    {
        *neighbor_rank = (current_rank == 0u) ? (rank_count - 1u) : (current_rank - 1u);
        return SPARK_STATUS_OK;
    }

    if (direction == SPARK_MEMLINK_NEIGHBOR_NEXT)
    {
        *neighbor_rank = (current_rank + 1u == rank_count) ? 0u : (current_rank + 1u);
        return SPARK_STATUS_OK;
    }

    return SPARK_STATUS_INVALID_ARGUMENT;
}

static int SparkMemlinkTemplatePlaceholder(
    const char *host_template,
    const char **placeholder)
{
	const char *found = NULL;
	const char *cursor = host_template;

	*placeholder = NULL;
	if (host_template == NULL)
		return(-1);
	while (*cursor != '\0')
	{
		if (cursor[0] != '%' || cursor[1] == '\0')
		{
			++cursor;
			continue;
		}
		if (cursor[1] == 'x' || cursor[1] == 'u' || cursor[1] == 'd')
		{
			if (found != NULL)
				return(-1);
			found = cursor;
			cursor += 2u;
			continue;
		}
		if (cursor[1] == '%')
		{
			cursor += 2u;
			continue;
		}
		return(-1);
	}
	if (found != NULL)
	{
		*placeholder = found;
		return(1);
	}
	return(0);
}

SparkStatus SparkMemlinkFormatHostFromTemplate(
    const char *host_template,
    uint32_t rank,
    char *host,
    size_t host_capacity)
{
    const char *placeholder;
    int written;

    if (host_template == NULL || host == NULL || host_capacity == 0u)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    {
        int template_kind = SparkMemlinkTemplatePlaceholder(
            host_template,&placeholder);
        if (template_kind < 0)
        {
            return(SPARK_STATUS_INVALID_ARGUMENT);
        }
        if (template_kind == 0)
        {
            written = snprintf(host, host_capacity, "%s", host_template);
        }
        else if (placeholder[1] == 'd')
        {
            written = snprintf(host, host_capacity, host_template, (int)rank);
        }
        else
        {
            written = snprintf(host, host_capacity, host_template, rank);
        }
    }

    if (written < 0 || (size_t)written >= host_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkMemlinkResolveNeighborEndpoint(
    uint32_t current_rank,
    uint32_t rank_count,
    SparkMemlinkNeighborDirection direction,
    const char *host_template,
    uint16_t base_port,
    uint32_t lane_count,
    SparkMemlinkEndpoint *endpoint)
{
    SparkStatus status;
    uint32_t neighbor_rank;

    if (endpoint == NULL || host_template == NULL || SparkMemlinkValidateLaneCount(lane_count) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkMemlinkResolveNeighborRank(current_rank, rank_count, direction, &neighbor_rank);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkMemlinkFormatHostFromTemplate(host_template, neighbor_rank, endpoint->host, sizeof(endpoint->host));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    endpoint->base_port = base_port;
    endpoint->lane_count = lane_count;

    return SPARK_STATUS_OK;
}
