#include "sparkpipe/spark_hidden_transport.h"

#include <stdlib.h>
#include <string.h>

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


SparkStatus SparkHiddenTransportValidatePacket(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportPacket *packet)
{
    SparkStatus status;
    uint64_t transfer_bytes;
    uint32_t required_packet_flags;
    uint32_t known_packet_flags;

    status = SparkHiddenTransportValidateEndpoint(endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (packet->abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        packet->descriptor_bytes != SPARK_HIDDEN_TRANSPORT_PACKET_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }

    known_packet_flags =
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_END_OF_SEQUENCE;
    if ((packet->flags & ~known_packet_flags) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    required_packet_flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16;
    if ((endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_DEVICE_POINTER_IO) != 0u ||
        (endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_NO_HOST_STAGING) != 0u)
    {
        required_packet_flags |=
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    }
    if ((packet->flags & required_packet_flags) != required_packet_flags)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_STREAM_ORDERED) != 0u &&
        packet->cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (packet->hidden_bf16 == 0 ||
        packet->active_sequence_count == 0u ||
        packet->active_sequence_count > endpoint->max_active_sequence_count ||
        packet->hidden_dimension != endpoint->hidden_dimension ||
        packet->bytes_per_sequence != endpoint->bytes_per_sequence)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    transfer_bytes =
        (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    if (packet->active_sequence_count != 0u &&
        transfer_bytes / packet->active_sequence_count !=
            (uint64_t)packet->bytes_per_sequence)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (transfer_bytes > endpoint->max_packet_bytes)
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

struct SparkHiddenTransportSession
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    void *transport_state;
};

SparkStatus SparkHiddenTransportOpen(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags,
    SparkHiddenTransportSession **session_out)
{
    SparkHiddenTransportSession *session;
    SparkStatus status;

    if (session_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *session_out = 0;

    status = SparkHiddenTransportValidateEndpoint(endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkHiddenTransportValidateInterface(
        transport_interface,
        required_capability_flags);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((transport_interface->capability_flags & endpoint->capability_flags) !=
        endpoint->capability_flags)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    session = (SparkHiddenTransportSession *)calloc(1u, sizeof(*session));
    if (session == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    session->endpoint = *endpoint;
    session->transport_interface = *transport_interface;
    status = session->transport_interface.initialize(
        &session->endpoint,
        &session->transport_state);
    if (status != SPARK_STATUS_OK)
    {
        free(session);
        return status;
    }
    if (session->transport_state == 0)
    {
        free(session);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *session_out = session;
    return SPARK_STATUS_OK;
}

void SparkHiddenTransportClose(SparkHiddenTransportSession *session)
{
    if (session == 0)
    {
        return;
    }
    if (session->transport_interface.destroy != 0 &&
        session->transport_state != 0)
    {
        session->transport_interface.destroy(session->transport_state);
    }
    free(session);
}

SparkStatus SparkHiddenTransportPostReceive(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packet)
{
    SparkStatus status;

    if (session == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&session->endpoint, packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return session->transport_interface.post_receive(
        session->transport_state,
        packet);
}

SparkStatus SparkHiddenTransportSend(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportPacket *packet)
{
    SparkStatus status;

    if (session == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&session->endpoint, packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return session->transport_interface.send(
        session->transport_state,
        packet);
}

SparkStatus SparkHiddenTransportPoll(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportCompletion *completion)
{
    SparkStatus status;

    if (session == 0 || completion == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(completion, 0, sizeof(*completion));
    completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    status = session->transport_interface.poll(
        session->transport_state,
        completion);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (completion->abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        completion->descriptor_bytes != SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (completion->active_sequence_count >
            session->endpoint.max_active_sequence_count ||
        completion->transfer_bytes > session->endpoint.max_packet_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}
