#include "sparkpipe/spark_hidden_transport.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct SparkHiddenTransportSession
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    void *transport_state;
    char transport_module_id[512];
    char route_name[512];
    char source_host[512];
    char sink_host[512];
};

static SparkStatus SparkHiddenTransportCopySessionText(
    char *destination,
    uint32_t destination_bytes,
    const char *source,
    uint32_t required)
{
    uint64_t bytes;
    if (destination == 0 || destination_bytes == 0u ||
        (required != 0u && (source == 0 || source[0] == '\0')))
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (source == 0)
    {
        destination[0] = '\0';
        return SPARK_STATUS_OK;
    }
    bytes = (uint64_t)strlen(source) + 1u;
    if (bytes > destination_bytes)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    memcpy(destination,source,(size_t)bytes);
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTransportOwnEndpointText(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportEndpoint *endpoint)
{
    SparkStatus status;
    session->endpoint = *endpoint;
    status = SparkHiddenTransportCopySessionText(session->transport_module_id,
        sizeof(session->transport_module_id),endpoint->transport_module_id,1u);
    if (status == SPARK_STATUS_OK)
        status = SparkHiddenTransportCopySessionText(session->route_name,
            sizeof(session->route_name),endpoint->route_name,1u);
    if (status == SPARK_STATUS_OK)
        status = SparkHiddenTransportCopySessionText(session->source_host,
            sizeof(session->source_host),endpoint->source_host,0u);
    if (status == SPARK_STATUS_OK)
        status = SparkHiddenTransportCopySessionText(session->sink_host,
            sizeof(session->sink_host),endpoint->sink_host,0u);
    session->endpoint.transport_module_id = session->transport_module_id;
    session->endpoint.route_name = session->route_name;
    session->endpoint.source_host = endpoint->source_host != 0 ? session->source_host : 0;
    session->endpoint.sink_host = endpoint->sink_host != 0 ? session->sink_host : 0;
    return status;
}

static uint32_t SparkHiddenTransportInterfaceRequiresBatchFunctions(
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags)
{
    return ((transport_interface->capability_flags | required_capability_flags) &
        SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION) != 0u;
}

static uint32_t SparkHiddenTransportSessionCanUseBatchSubmission(
    const SparkHiddenTransportSession *session)
{
    return (session->transport_interface.capability_flags &
        SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION) != 0u &&
        session->transport_interface.post_receive_batch != 0 &&
        session->transport_interface.send_batch != 0;
}

static uint32_t SparkHiddenTransportSessionCanUsePollDescriptors(
    const SparkHiddenTransportSession *session)
{
    return session != 0 &&
        (session->transport_interface.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_POLL_DESCRIPTORS) != 0u &&
        session->transport_interface.get_poll_descriptors != 0;
}

static uint32_t SparkHiddenTransportSessionCanUsePersistentReceiveCredits(
    const SparkHiddenTransportSession *session)
{
    return session != 0 &&
        (session->transport_interface.capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS) != 0u &&
        session->transport_interface.register_persistent_receive != 0 &&
        session->transport_interface.persistent_remote_credit_ready != 0 &&
        session->transport_interface.reserve_persistent_send != 0 &&
        session->transport_interface.cancel_persistent_send != 0 &&
        session->transport_interface.activate_persistent_receive != 0 &&
        session->transport_interface.cancel_persistent_receive != 0 &&
        session->transport_interface.send_persistent != 0 &&
        session->transport_interface.release_persistent_receive != 0;
}

static uint32_t SparkHiddenTransportCapabilitiesAreSimulationOnly(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY) != 0u;
}

static uint32_t SparkHiddenTransportCapabilitiesMeetProduction(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS;
}


static uint32_t SparkHiddenTransportCapabilitiesMeetSparkHostRdma(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS;
}

static uint32_t SparkHiddenTransportCapabilitiesMeetSparkGpudirectRdma(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS;
}

static void SparkHiddenTransportCopyRouteConfiguration(
    SparkHiddenTransportEndpoint *destination,
    const SparkHiddenTransportEndpoint *source)
{
    destination->configuration_flags = source->configuration_flags;
    destination->local_rank_index = source->local_rank_index;
    destination->source_rank_index = source->source_rank_index;
    destination->sink_rank_index = source->sink_rank_index;
    destination->control_port_base = source->control_port_base;
    destination->reserved0 = source->reserved0;
    destination->source_host = source->source_host;
    destination->sink_host = source->sink_host;
    destination->route_identifier = source->route_identifier;
}

static void SparkHiddenTransportBuildEffectiveEndpoint(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportInterface *transport_interface,
    SparkHiddenTransportEndpoint *effective_endpoint)
{
    if (endpoint == 0 || effective_endpoint == 0)
    {
        return;
    }
    *effective_endpoint = *endpoint;
    if (transport_interface != 0 &&
        SparkHiddenTransportCapabilitiesMeetSparkGpudirectRdma(
            transport_interface->capability_flags))
    {
        SparkHiddenTransportInitializeSparkGpudirectRdmaEndpoint(
            effective_endpoint,
            endpoint->hidden_dimension,
            endpoint->max_active_sequence_count,
            endpoint->validated_latency_ns,
            endpoint->route_name);
        effective_endpoint->bytes_per_sequence = endpoint->bytes_per_sequence;
        effective_endpoint->max_packet_bytes = endpoint->max_packet_bytes;
        SparkHiddenTransportCopyRouteConfiguration(effective_endpoint,endpoint);
        effective_endpoint->capability_flags |= endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    }
    else if (transport_interface != 0 &&
        SparkHiddenTransportCapabilitiesMeetSparkHostRdma(
            transport_interface->capability_flags))
    {
        SparkHiddenTransportInitializeSparkHostRdmaEndpoint(
            effective_endpoint,
            endpoint->hidden_dimension,
            endpoint->max_active_sequence_count,
            endpoint->validated_latency_ns,
            endpoint->route_name);
        effective_endpoint->bytes_per_sequence = endpoint->bytes_per_sequence;
        effective_endpoint->max_packet_bytes = endpoint->max_packet_bytes;
        SparkHiddenTransportCopyRouteConfiguration(effective_endpoint,endpoint);
        effective_endpoint->capability_flags |= endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    }
}

static uint32_t SparkHiddenTransportCapabilitiesMeetSimulation(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS;
}

static uint32_t SparkHiddenTransportCapabilitiesMeetPipelineHostStaged(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS;
}

static uint32_t SparkHiddenTransportStringsEqual(
    const char *left,
    const char *right)
{
    if (left == 0 || right == 0)
    {
        return 0u;
    }
    return strcmp(left, right) == 0;
}

static uint32_t SparkHiddenTransportRouteHostIsValid(const char *host)
{
    return host != 0 && host[0] != '\0' && strcmp(host,"0.0.0.0") != 0 &&
        strcmp(host,"::") != 0 && strcmp(host,"*") != 0;
}

SparkStatus SparkHiddenTransportValidateEndpoint(
    const SparkHiddenTransportEndpoint *endpoint)
{
    uint64_t bytes_per_sequence;
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
    if (SparkHiddenTransportCapabilitiesAreSimulationOnly(
            endpoint->capability_flags))
    {
        if (!SparkHiddenTransportCapabilitiesMeetSimulation(
                endpoint->capability_flags) ||
            SparkHiddenTransportCapabilitiesMeetProduction(
                endpoint->capability_flags))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (!SparkHiddenTransportCapabilitiesMeetProduction(
            endpoint->capability_flags) &&
        !SparkHiddenTransportCapabilitiesMeetPipelineHostStaged(
            endpoint->capability_flags))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (endpoint->transport_module_id == 0 ||
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
    if ((endpoint->configuration_flags &
            ~SPARK_HIDDEN_TRANSPORT_ENDPOINT_KNOWN_FLAGS) != 0u ||
        (((endpoint->configuration_flags &
            SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_OPEN_TIMEOUT) != 0u) !=
            (endpoint->reserved0 != 0u)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((endpoint->configuration_flags &
            SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION) != 0u)
    {
        if (endpoint->local_rank_index > (uint32_t)INT32_MAX ||
            endpoint->source_rank_index > (uint32_t)INT32_MAX ||
            endpoint->sink_rank_index > (uint32_t)INT32_MAX ||
            endpoint->source_rank_index == endpoint->sink_rank_index ||
            (endpoint->local_rank_index != endpoint->source_rank_index &&
             endpoint->local_rank_index != endpoint->sink_rank_index) ||
            endpoint->control_port_base == 0u ||
            endpoint->control_port_base >
                UINT16_MAX - endpoint->sink_rank_index ||
            SparkHiddenTransportRouteHostIsValid(endpoint->source_host) == 0u ||
            SparkHiddenTransportRouteHostIsValid(endpoint->sink_host) == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (endpoint->local_rank_index != 0u ||
        endpoint->source_rank_index != 0u ||
        endpoint->sink_rank_index != 0u ||
        endpoint->control_port_base != 0u || endpoint->source_host != 0 ||
        endpoint->sink_host != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    bytes_per_sequence = (uint64_t)endpoint->hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    if (bytes_per_sequence > UINT32_MAX ||
        endpoint->bytes_per_sequence != bytes_per_sequence)
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

SparkStatus SparkHiddenTransportConfigureEndpointOpenTimeout(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t timeout_milli)
{
    if (endpoint == 0 || timeout_milli == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    endpoint->configuration_flags |=
        SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_OPEN_TIMEOUT;
    endpoint->reserved0 = timeout_milli;
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportValidatePacket(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportPacket *packet)
{
    SparkStatus status;
    uint64_t hidden_transfer_bytes;
    uint64_t sideband_transfer_bytes;
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
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_END_OF_SEQUENCE |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SUBRANGE_SHAPE;
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
        fprintf(stderr,"G5N-TP validate-stream-zero caps=%x\n",
            endpoint->capability_flags);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (packet->hidden_bf16 == 0 ||
        packet->active_sequence_count == 0u ||
        packet->active_sequence_count > endpoint->max_active_sequence_count ||
        packet->hidden_dimension == 0u ||
        packet->bytes_per_sequence !=
            (uint64_t)packet->hidden_dimension *
                SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT ||
        (((packet->flags &
                SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SUBRANGE_SHAPE) == 0u) &&
            (packet->hidden_dimension != endpoint->hidden_dimension ||
             packet->bytes_per_sequence != endpoint->bytes_per_sequence)) ||
        (((packet->flags &
                SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SUBRANGE_SHAPE) != 0u) &&
            (packet->hidden_dimension > endpoint->hidden_dimension ||
             packet->bytes_per_sequence > endpoint->bytes_per_sequence)))
    {
        fprintf(stderr,
            "G5N-TP validate-shape hidden=%p rows=%u/%u dim=%u bps=%llu/%llu flags=%x ep_dim=%u\n",
            packet->hidden_bf16,packet->active_sequence_count,
            endpoint->max_active_sequence_count,packet->hidden_dimension,
            (unsigned long long)packet->bytes_per_sequence,
            (unsigned long long)endpoint->bytes_per_sequence,packet->flags,
            endpoint->hidden_dimension);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    hidden_transfer_bytes =
        (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    if (packet->active_sequence_count != 0u &&
        hidden_transfer_bytes / packet->active_sequence_count !=
            (uint64_t)packet->bytes_per_sequence)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    sideband_transfer_bytes = 0u;
    if ((packet->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD) != 0u)
    {
        sideband_transfer_bytes =
            (uint64_t)packet->sideband_bytes_per_sequence *
            (uint64_t)packet->active_sequence_count;
        if (packet->sideband_payload == 0 ||
            packet->sideband_kind == 0u ||
            packet->sideband_bytes_per_sequence == 0u ||
            sideband_transfer_bytes / packet->active_sequence_count !=
                (uint64_t)packet->sideband_bytes_per_sequence)
        {
        return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (packet->sideband_payload != 0 ||
             packet->sideband_kind != 0u ||
             packet->sideband_bytes_per_sequence != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    transfer_bytes = hidden_transfer_bytes + sideband_transfer_bytes;
    if (transfer_bytes < hidden_transfer_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (transfer_bytes > endpoint->max_packet_bytes)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportValidatePacketBatch(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;
    uint32_t packet_index;

    if (packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenTransportValidatePacket(
            endpoint,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
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
    if (SparkHiddenTransportCapabilitiesAreSimulationOnly(
            transport_interface->capability_flags) &&
        (required_capability_flags &
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS) ==
            SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((transport_interface->capability_flags & required_capability_flags) !=
            required_capability_flags ||
        transport_interface->reserved != 0u ||
        transport_interface->initialize == 0 ||
        transport_interface->destroy == 0 ||
        transport_interface->post_receive == 0 ||
        transport_interface->send == 0 ||
        transport_interface->poll == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkHiddenTransportInterfaceRequiresBatchFunctions(
            transport_interface,
            required_capability_flags) &&
        (transport_interface->post_receive_batch == 0 ||
         transport_interface->send_batch == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((transport_interface->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS) != 0u &&
        (transport_interface->register_persistent_receive == 0 ||
         transport_interface->persistent_remote_credit_ready == 0 ||
         transport_interface->reserve_persistent_send == 0 ||
         transport_interface->cancel_persistent_send == 0 ||
         transport_interface->activate_persistent_receive == 0 ||
         transport_interface->cancel_persistent_receive == 0 ||
         transport_interface->send_persistent == 0 ||
         transport_interface->release_persistent_receive == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportLoadInterfaceFromSharedObject(
    const char *shared_object_path,
    uint32_t required_capability_flags,
    SparkHiddenTransportDynamicLibrary *library)
{
    SparkHiddenTransportGetInterfaceFunction get_interface;
    const SparkHiddenTransportInterface *transport_interface;
    SparkStatus status;
    void *dynamic_library;

    if (shared_object_path == 0 || shared_object_path[0] == '\0' ||
        library == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(library, 0, sizeof(*library));
    dynamic_library = dlopen(shared_object_path, RTLD_NOW | RTLD_LOCAL);
    if (dynamic_library == 0)
    {
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    get_interface = (SparkHiddenTransportGetInterfaceFunction)dlsym(
        dynamic_library,
        SPARK_HIDDEN_TRANSPORT_INTERFACE_SYMBOL);
    if (get_interface == 0)
    {
        dlclose(dynamic_library);
        return SPARK_STATUS_DRIVER_LOAD_ERROR;
    }
    transport_interface = get_interface();
    status = SparkHiddenTransportValidateInterface(
        transport_interface,
        required_capability_flags);
    if (status != SPARK_STATUS_OK)
    {
        dlclose(dynamic_library);
        return status;
    }
    library->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    library->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_DYNAMIC_LIBRARY_BYTES;
    library->dynamic_library = dynamic_library;
    library->transport_interface = *transport_interface;
    return SPARK_STATUS_OK;
}

void SparkHiddenTransportUnloadInterface(
    SparkHiddenTransportDynamicLibrary *library)
{
    if (library == 0)
    {
        return;
    }
    if (library->dynamic_library != 0)
    {
        dlclose(library->dynamic_library);
    }
    memset(library, 0, sizeof(*library));
}

SparkStatus SparkHiddenTransportOpen(
    const SparkHiddenTransportEndpoint *endpoint,
    const SparkHiddenTransportInterface *transport_interface,
    uint32_t required_capability_flags,
    SparkHiddenTransportSession **session_out)
{
    SparkHiddenTransportSession *session;
    SparkHiddenTransportEndpoint effective_endpoint;
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
    SparkHiddenTransportBuildEffectiveEndpoint(
        endpoint,
        transport_interface,
        &effective_endpoint);
    status = SparkHiddenTransportValidateEndpoint(&effective_endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((transport_interface->capability_flags &
            effective_endpoint.capability_flags) !=
        effective_endpoint.capability_flags)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    session = (SparkHiddenTransportSession *)calloc(1u, sizeof(*session));
    if (session == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkHiddenTransportOwnEndpointText(session,&effective_endpoint);
    if (status != SPARK_STATUS_OK)
    {
        free(session);
        return status;
    }
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

SparkStatus SparkHiddenTransportPostReceiveBatch(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;

    if (session == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacketBatch(
        &session->endpoint,
        packets,
        packet_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkHiddenTransportSessionCanUseBatchSubmission(session) == 0u)
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    return session->transport_interface.post_receive_batch(
        session->transport_state,
        packets,
        packet_count);
}

SparkStatus SparkHiddenTransportSendBatch(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;

    if (session == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacketBatch(
        &session->endpoint,
        packets,
        packet_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkHiddenTransportSessionCanUseBatchSubmission(session) == 0u)
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    return session->transport_interface.send_batch(
        session->transport_state,
        packets,
        packet_count);
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

SparkStatus SparkHiddenTransportGetPollDescriptors(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPollDescriptor *descriptors,
    uint32_t descriptor_capacity,
    uint32_t *descriptor_count_out)
{
    SparkStatus status;
    uint32_t descriptor_index;

    if (session == 0 || descriptor_count_out == 0 ||
        (descriptors == 0 && descriptor_capacity != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *descriptor_count_out = 0u;
    if (SparkHiddenTransportSessionCanUsePollDescriptors(session) == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    status = session->transport_interface.get_poll_descriptors(
        session->transport_state,
        descriptors,
        descriptor_capacity,
        descriptor_count_out);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (*descriptor_count_out > descriptor_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (descriptor_index = 0u;
         descriptor_index < *descriptor_count_out;
         ++descriptor_index)
    {
        if (descriptors[descriptor_index].abi_version !=
                SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
            descriptors[descriptor_index].descriptor_bytes !=
                SPARK_HIDDEN_TRANSPORT_POLL_DESCRIPTOR_BYTES ||
            descriptors[descriptor_index].fd < 0 ||
            (descriptors[descriptor_index].events &
                ~(SPARK_HIDDEN_TRANSPORT_POLL_READ |
                  SPARK_HIDDEN_TRANSPORT_POLL_WRITE)) != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportRegisterPersistentReceive(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    SparkHiddenTransportPacket *packet_template)
{
    SparkStatus status;

    if (session == 0 || packet_template == 0 ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(
        &session->endpoint,packet_template);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return session->transport_interface.register_persistent_receive(
        session->transport_state,credit_index,packet_template);
}

SparkStatus SparkHiddenTransportPersistentRemoteCreditReady(
    SparkHiddenTransportSession *session,
    uint32_t credit_index)
{
    if (session == 0 ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return session->transport_interface.persistent_remote_credit_ready(
        session->transport_state,credit_index);
}

SparkStatus SparkHiddenTransportReservePersistentSend(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    SparkStatus status;

    if (session == 0 || packet == 0 || generation == 0u ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        fprintf(stderr,
            "G5N-TP w-reserve-guard session=%p packet=%p gen=%llu caps=%x\n",
            (void *)session,(const void *)packet,
            (unsigned long long)generation,
            session != 0 ? session->transport_interface.capability_flags : 0u);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&session->endpoint,packet);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "G5N-TP w-reserve-validate status=%u rows=%u hidden=%u flags=%x stream=%p\n",
            (unsigned)status,packet != 0 ? packet->active_sequence_count : 0u,
            packet != 0 ? packet->hidden_dimension : 0u,
            packet != 0 ? packet->flags : 0u,
            packet != 0 ? packet->cuda_stream : 0);
        return status;
    }
    return session->transport_interface.reserve_persistent_send(
        session->transport_state,credit_index,generation,packet);
}

SparkStatus SparkHiddenTransportCancelPersistentSend(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    uint64_t generation)
{
    if (session == 0 || generation == 0u ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return session->transport_interface.cancel_persistent_send(
        session->transport_state,credit_index,generation);
}

SparkStatus SparkHiddenTransportActivatePersistentReceive(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    uint64_t generation,
    SparkHiddenTransportPacket *packet)
{
    SparkStatus status;

    if (session == 0 || packet == 0 || generation == 0u ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        fprintf(stderr,
            "G5N-TP w-activate-guard session=%p gen=%llu caps=%x\n",
            (void *)session,(unsigned long long)generation,
            session != 0 ? session->transport_interface.capability_flags : 0u);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&session->endpoint,packet);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,
            "G5N-TP w-activate-validate status=%u rows=%u hidden=%u flags=%x\n",
            (unsigned)status,packet != 0 ? packet->active_sequence_count : 0u,
            packet != 0 ? packet->hidden_dimension : 0u,
            packet != 0 ? packet->flags : 0u);
        return status;
    }
    return session->transport_interface.activate_persistent_receive(
        session->transport_state,credit_index,generation,packet);
}

SparkStatus SparkHiddenTransportCancelPersistentReceive(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    uint64_t generation)
{
    if (session == 0 || generation == 0u ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return session->transport_interface.cancel_persistent_receive(
        session->transport_state,credit_index,generation);
}

SparkStatus SparkHiddenTransportSendPersistent(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    SparkStatus status;

    if (session == 0 || packet == 0 || generation == 0u ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportValidatePacket(&session->endpoint,packet);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return session->transport_interface.send_persistent(
        session->transport_state,credit_index,generation,packet);
}

SparkStatus SparkHiddenTransportReleasePersistentReceive(
    SparkHiddenTransportSession *session,
    uint32_t credit_index,
    uint64_t generation,
    void *consumer_cuda_stream)
{
    if (session == 0 || generation == 0u || consumer_cuda_stream == 0 ||
        SparkHiddenTransportSessionCanUsePersistentReceiveCredits(session) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return session->transport_interface.release_persistent_receive(
        session->transport_state,credit_index,generation,
        consumer_cuda_stream);
}

void SparkHiddenTransportCompletionQueueInitialize(
    SparkHiddenTransportCompletionQueue *queue)
{
    if (queue != 0)
        memset(queue,0,sizeof(*queue));
}

uint32_t SparkHiddenTransportCompletionQueueIsFull(
    const SparkHiddenTransportCompletionQueue *queue)
{
    return queue != 0 &&
        queue->count >= SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH;
}

SparkStatus SparkHiddenTransportCompletionQueuePush(
    SparkHiddenTransportCompletionQueue *queue,
    const SparkHiddenTransportCompletion *completion)
{
    uint32_t tail;
    if (queue == 0 || completion == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (SparkHiddenTransportCompletionQueueIsFull(queue) != 0u)
    {
        queue->dropped_count += 1u;
        return SPARK_STATUS_BUSY;
    }
    tail = (queue->head + queue->count) %
        SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH;
    queue->entries[tail] = *completion;
    queue->count += 1u;
    queue->total_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportCompletionQueuePushPacket(
    SparkHiddenTransportCompletionQueue *queue,
    const SparkHiddenTransportPacket *packet,
    SparkStatus status,
    uint64_t service_time_ns)
{
    SparkHiddenTransportCompletion completion;
    uint64_t transfer_bytes;
    if (queue == 0 || packet == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    transfer_bytes = (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    if ((packet->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD) != 0u)
        transfer_bytes +=
            (uint64_t)packet->sideband_bytes_per_sequence *
            (uint64_t)packet->active_sequence_count;
    memset(&completion,0,sizeof(completion));
    completion.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    completion.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    completion.status = status;
    completion.active_sequence_count = packet->active_sequence_count;
    completion.sequence_id = packet->sequence_id;
    completion.token_index = packet->token_index;
    completion.transfer_bytes = transfer_bytes;
    completion.service_time_ns = service_time_ns;
    return SparkHiddenTransportCompletionQueuePush(queue,&completion);
}

SparkStatus SparkHiddenTransportCompletionQueuePop(
    SparkHiddenTransportCompletionQueue *queue,
    SparkHiddenTransportCompletion *completion)
{
    if (queue == 0 || completion == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (queue->count == 0u)
    {
        memset(completion,0,sizeof(*completion));
        completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
        completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
        completion->status = SPARK_STATUS_BUSY;
        return SPARK_STATUS_OK;
    }
    *completion = queue->entries[queue->head];
    queue->head = (queue->head + 1u) %
        SPARK_HIDDEN_TRANSPORT_COMPLETION_QUEUE_DEPTH;
    queue->count -= 1u;
    return SPARK_STATUS_OK;
}

typedef struct SparkHiddenTransportPersistentRingState
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportCompletionQueue completion_queue;
    uint64_t send_count;
    uint64_t receive_count;
} SparkHiddenTransportPersistentRingState;

static SparkStatus SparkHiddenTransportPersistentRingPushCompletion(
    SparkHiddenTransportPersistentRingState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus packet_status)
{
    if (state == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkHiddenTransportCompletionQueuePushPacket(
        &state->completion_queue,
        packet,
        packet_status,
        state->endpoint.validated_latency_ns);
}

static SparkStatus SparkHiddenTransportPersistentRingInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    SparkHiddenTransportPersistentRingState *state;

    if (endpoint == 0 || transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenTransportPersistentRingState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    state->endpoint = *endpoint;
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void SparkHiddenTransportPersistentRingDestroy(void *transport_state)
{
    free(transport_state);
}

static SparkStatus SparkHiddenTransportPersistentRingPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    SparkHiddenTransportPersistentRingState *state;
    SparkStatus status;

    if (transport_state == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenTransportPersistentRingState *)transport_state;
    status = SparkHiddenTransportPersistentRingPushCompletion(
        state,
        packet,
        SPARK_STATUS_OK);
    if (status == SPARK_STATUS_OK)
    {
        state->receive_count += 1u;
    }
    return status;
}

static SparkStatus SparkHiddenTransportPersistentRingSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    SparkHiddenTransportPersistentRingState *state;
    SparkStatus status;

    if (transport_state == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenTransportPersistentRingState *)transport_state;
    status = SparkHiddenTransportPersistentRingPushCompletion(
        state,
        packet,
        SPARK_STATUS_OK);
    if (status == SPARK_STATUS_OK)
    {
        state->send_count += 1u;
    }
    return status;
}

static SparkStatus SparkHiddenTransportPersistentRingPostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    uint32_t packet_index;
    SparkStatus status;

    if (transport_state == 0 || packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenTransportPersistentRingPostReceive(
            transport_state,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTransportPersistentRingSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    uint32_t packet_index;
    SparkStatus status;

    if (transport_state == 0 || packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = SparkHiddenTransportPersistentRingSend(
            transport_state,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTransportPersistentRingPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    SparkHiddenTransportPersistentRingState *state;

    if (transport_state == 0 || completion == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (SparkHiddenTransportPersistentRingState *)transport_state;
    return SparkHiddenTransportCompletionQueuePop(
        &state->completion_queue,completion);
}

SparkStatus SparkHiddenTransportPersistentRingGetInterface(
    SparkHiddenTransportInterface *transport_interface)
{
    if (transport_interface == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(transport_interface, 0, sizeof(*transport_interface));
    transport_interface->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface->descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface->capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS;
    transport_interface->initialize =
        SparkHiddenTransportPersistentRingInitialize;
    transport_interface->destroy =
        SparkHiddenTransportPersistentRingDestroy;
    transport_interface->post_receive =
        SparkHiddenTransportPersistentRingPostReceive;
    transport_interface->send = SparkHiddenTransportPersistentRingSend;
    transport_interface->poll = SparkHiddenTransportPersistentRingPoll;
    transport_interface->post_receive_batch =
        SparkHiddenTransportPersistentRingPostReceiveBatch;
    transport_interface->send_batch = SparkHiddenTransportPersistentRingSendBatch;
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportPersistentRingGetStatistics(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPersistentRingStatistics *statistics)
{
    SparkHiddenTransportPersistentRingState *state;

    if (session == 0 || statistics == 0 || session->transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (session->transport_interface.initialize !=
            SparkHiddenTransportPersistentRingInitialize ||
        session->transport_interface.poll != SparkHiddenTransportPersistentRingPoll)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    state = (SparkHiddenTransportPersistentRingState *)session->transport_state;
    memset(statistics, 0, sizeof(*statistics));
    statistics->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    statistics->descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_STATISTICS_BYTES;
    statistics->send_count = state->send_count;
    statistics->receive_count = state->receive_count;
    statistics->completion_count = state->completion_queue.total_count;
    statistics->dropped_completion_count =
        state->completion_queue.dropped_count;
    statistics->queued_completion_count = state->completion_queue.count;
    statistics->queue_depth =
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH;
    return SPARK_STATUS_OK;
}


static void SparkHiddenTransportInitializeRdmaEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t hidden_dimension,
    uint32_t max_active_sequence_count,
    uint64_t validated_latency_ns,
    const char *route_name,
    const char *transport_module_id,
    uint32_t capability_flags)
{
    if (endpoint == 0)
    {
        return;
    }
    memset(endpoint, 0, sizeof(*endpoint));
    endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint->capability_flags = capability_flags;
    endpoint->hidden_dimension = hidden_dimension;
    endpoint->bytes_per_sequence = hidden_dimension <=
        UINT32_MAX / SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT ?
        hidden_dimension * SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT : 0u;
    endpoint->max_active_sequence_count = max_active_sequence_count;
    endpoint->max_packet_bytes =
        (uint64_t)endpoint->bytes_per_sequence *
        (uint64_t)max_active_sequence_count;
    endpoint->validated_latency_ns = validated_latency_ns;
    endpoint->transport_module_id = transport_module_id;
    endpoint->route_name = route_name;
}

static SparkStatus SparkHiddenTransportValidateRdmaEndpoint(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *transport_module_id,
    uint32_t required_capability_flags)
{
    SparkStatus status;

    status = SparkHiddenTransportValidateEndpoint(endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkHiddenTransportStringsEqual(
            endpoint->transport_module_id,
            transport_module_id))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((endpoint->capability_flags & required_capability_flags) !=
        required_capability_flags)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkHiddenTransportRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *infiniband_sysfs_path,
    const char *transport_module_id,
    uint32_t required_capability_flags)
{
    SparkStatus status;
    const char *infiniband_path;

    status = SparkHiddenTransportValidateRdmaEndpoint(
        endpoint,
        transport_module_id,
        required_capability_flags);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    infiniband_path = infiniband_sysfs_path;
    if (infiniband_path == 0 || infiniband_path[0] == '\0')
    {
        infiniband_path =
            SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_INFINIBAND_SYSFS_PATH;
    }
    if (access(infiniband_path, F_OK) != 0)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    return SPARK_STATUS_OK;
}

void SparkHiddenTransportInitializeSparkHostRdmaEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t hidden_dimension,
    uint32_t max_active_sequence_count,
    uint64_t validated_latency_ns,
    const char *route_name)
{
    SparkHiddenTransportInitializeRdmaEndpoint(
        endpoint,
        hidden_dimension,
        max_active_sequence_count,
        validated_latency_ns,
        route_name,
        SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS);
}

SparkStatus SparkHiddenTransportValidateSparkHostRdmaEndpoint(
    const SparkHiddenTransportEndpoint *endpoint)
{
    return SparkHiddenTransportValidateRdmaEndpoint(
        endpoint,
        SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS);
}

SparkStatus SparkHiddenTransportSparkHostRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *infiniband_sysfs_path)
{
    return SparkHiddenTransportRdmaVerbsPreflight(
        endpoint,
        infiniband_sysfs_path,
        SPARK_HIDDEN_TRANSPORT_SPARK_HOST_RDMA_VERBS_MODULE_ID,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_HOST_RDMA_CAPS);
}

void SparkHiddenTransportInitializeSparkGpudirectRdmaEndpoint(
    SparkHiddenTransportEndpoint *endpoint,
    uint32_t hidden_dimension,
    uint32_t max_active_sequence_count,
    uint64_t validated_latency_ns,
    const char *route_name)
{
    SparkHiddenTransportInitializeRdmaEndpoint(
        endpoint,
        hidden_dimension,
        max_active_sequence_count,
        validated_latency_ns,
        route_name,
        SPARK_HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA_VERBS_MODULE_ID,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_GPUDIRECT_RDMA_CAPS);
}

SparkStatus SparkHiddenTransportValidateSparkGpudirectRdmaEndpoint(
    const SparkHiddenTransportEndpoint *endpoint)
{
    return SparkHiddenTransportValidateRdmaEndpoint(
        endpoint,
        SPARK_HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA_VERBS_MODULE_ID,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS);
}

SparkStatus SparkHiddenTransportSparkGpudirectRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *infiniband_sysfs_path)
{
    return SparkHiddenTransportRdmaVerbsPreflight(
        endpoint,
        infiniband_sysfs_path,
        SPARK_HIDDEN_TRANSPORT_SPARK_GPUDIRECT_RDMA_VERBS_MODULE_ID,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS);
}
