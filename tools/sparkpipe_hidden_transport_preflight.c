#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_hidden_transport.h"

static void SparkPreflightInitializeEndpoint(
    SparkHiddenTransportEndpoint *endpoint)
{
    memset(endpoint, 0, sizeof(*endpoint));
    endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint->capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS;
    endpoint->hidden_dimension = 6144u;
    endpoint->bytes_per_sequence =
        endpoint->hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    endpoint->max_active_sequence_count = 1024u;
    endpoint->max_packet_bytes =
        (uint64_t)endpoint->bytes_per_sequence *
        (uint64_t)endpoint->max_active_sequence_count;
    endpoint->validated_latency_ns = 0u;
    endpoint->transport_module_id =
        SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_VERBS_MODULE_ID;
    endpoint->route_name = "sparkpipe_pp13_hidden_transport";
}

static int SparkPreflightParseUint32(
    const char *value,
    uint32_t *result)
{
    char *end;
    unsigned long parsed;

    if (value == 0 || result == 0 || value[0] == '\0')
    {
        return -1;
    }
    end = 0;
    parsed = strtoul(value, &end, 10);
    if (end == value || end == 0 || end[0] != '\0' || parsed > 0xfffffffful)
    {
        return -2;
    }
    *result = (uint32_t)parsed;
    return 0;
}

static int SparkPreflightApplyArgument(
    SparkHiddenTransportEndpoint *endpoint,
    const char **peermem_path,
    const char **infiniband_path,
    int argc,
    char **argv,
    int *index)
{
    uint32_t parsed;

    if (*index + 1 >= argc)
    {
        return -1;
    }
    if (strcmp(argv[*index], "--route") == 0)
    {
        endpoint->route_name = argv[*index + 1];
    }
    else if (strcmp(argv[*index], "--peermem-sysfs") == 0)
    {
        *peermem_path = argv[*index + 1];
    }
    else if (strcmp(argv[*index], "--infiniband-sysfs") == 0)
    {
        *infiniband_path = argv[*index + 1];
    }
    else if (strcmp(argv[*index], "--max-active") == 0)
    {
        if (SparkPreflightParseUint32(argv[*index + 1], &parsed) < 0)
        {
            return -2;
        }
        endpoint->max_active_sequence_count = parsed;
    }
    else if (strcmp(argv[*index], "--hidden-dimension") == 0)
    {
        if (SparkPreflightParseUint32(argv[*index + 1], &parsed) < 0)
        {
            return -3;
        }
        endpoint->hidden_dimension = parsed;
        endpoint->bytes_per_sequence =
            endpoint->hidden_dimension *
            SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    }
    else
    {
        return -4;
    }
    *index += 1;
    return 0;
}

int main(int argc, char **argv)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkStatus status;
    const char *peermem_path;
    const char *infiniband_path;
    int index;

    SparkPreflightInitializeEndpoint(&endpoint);
    peermem_path = 0;
    infiniband_path = 0;
    for (index = 1; index < argc; ++index)
    {
        if (SparkPreflightApplyArgument(
                &endpoint,
                &peermem_path,
                &infiniband_path,
                argc,
                argv,
                &index) < 0)
        {
            fprintf(stderr, "invalid argument: %s\n", argv[index]);
            return 2;
        }
    }
    endpoint.max_packet_bytes =
        (uint64_t)endpoint.bytes_per_sequence *
        (uint64_t)endpoint.max_active_sequence_count;
    status = SparkHiddenTransportGpudirectRdmaVerbsPreflight(
        &endpoint,
        peermem_path,
        infiniband_path);
    printf("gpudirect_rdma_verbs_preflight=%s\n", SparkStatusToString(status));
    printf("module=%s\n", endpoint.transport_module_id);
    printf("route=%s\n", endpoint.route_name);
    printf("hidden_dimension=%u\n", endpoint.hidden_dimension);
    printf("max_active_sequence_count=%u\n", endpoint.max_active_sequence_count);
    printf("peermem_sysfs=%s\n",
        peermem_path != 0 ? peermem_path :
        SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_PEERMEM_SYSFS_PATH);
    printf("infiniband_sysfs=%s\n",
        infiniband_path != 0 ? infiniband_path :
        SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_INFINIBAND_SYSFS_PATH);
    return status == SPARK_STATUS_OK ? 0 : 1;
}
