#include "sparkpipe/spark_hidden_transport.h"

SparkStatus SparkHiddenTransportValidateEndpoint(
    const SparkHiddenTransportEndpoint *endpoint)
{
    uint64_t maximum_payload_bytes;

    if (endpoint == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (endpoint->abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        endpoint->descriptor_bytes != SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if ((endpoint->capability_flags &
         SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS) !=
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS ||
        endpoint->transport_module_id == 0 ||
        endpoint->transport_module_id[0] == '\0' ||
        endpoint->route_name == 0 ||
        endpoint->route_name[0] == '\0' ||
        endpoint->hidden_dimension == 0u ||
        endpoint->bytes_per_sequence == 0u ||
        endpoint->max_active_sequence_count == 0u ||
        endpoint->max_packet_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (endpoint->bytes_per_sequence !=
        (endpoint->hidden_dimension *
         SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    maximum_payload_bytes =
        (uint64_t)endpoint->bytes_per_sequence *
        (uint64_t)endpoint->max_active_sequence_count;
    if (maximum_payload_bytes > endpoint->max_packet_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportValidateInterface(
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags)
{
    if (transport_interface == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (transport_interface->abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        transport_interface->descriptor_bytes !=
            SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if ((transport_interface->capability_flags & required_capability_flags) !=
            required_capability_flags ||
        transport_interface->initialize == 0 ||
        transport_interface->destroy == 0 ||
        transport_interface->post_receive == 0 ||
        transport_interface->send == 0 ||
        transport_interface->poll == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}
