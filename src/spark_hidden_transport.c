#include "sparkpipe/spark_hidden_transport.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct SparkHiddenTransportSession
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    void *transport_state;
};

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

static uint32_t SparkHiddenTransportCapabilitiesMeetSimulation(
    uint32_t capability_flags)
{
    return (capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS;
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
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
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
    return SPARK_STATUS_OK;
}

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

SparkStatus SparkHiddenTransportPostReceiveBatch(
    SparkHiddenTransportSession *session,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;
    uint32_t packet_index;

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
    if (SparkHiddenTransportSessionCanUseBatchSubmission(session))
    {
        return session->transport_interface.post_receive_batch(
            session->transport_state,
            packets,
            packet_count);
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = session->transport_interface.post_receive(
            session->transport_state,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportSendBatch(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    SparkStatus status;
    uint32_t packet_index;

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
    if (SparkHiddenTransportSessionCanUseBatchSubmission(session))
    {
        return session->transport_interface.send_batch(
            session->transport_state,
            packets,
            packet_count);
    }
    for (packet_index = 0u; packet_index < packet_count; ++packet_index)
    {
        status = session->transport_interface.send(
            session->transport_state,
            &packets[packet_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
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

typedef struct SparkHiddenTransportPersistentRingState
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportCompletion completions[
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH];
    uint32_t completion_head;
    uint32_t completion_tail;
    uint32_t completion_count;
    uint64_t send_count;
    uint64_t receive_count;
    uint64_t completion_total_count;
    uint64_t dropped_completion_count;
} SparkHiddenTransportPersistentRingState;

static SparkStatus SparkHiddenTransportPersistentRingPushCompletion(
    SparkHiddenTransportPersistentRingState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus packet_status)
{
    SparkHiddenTransportCompletion *completion;
    uint64_t transfer_bytes;

    if (state == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->completion_count >=
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH)
    {
        state->dropped_completion_count += 1u;
        return SPARK_STATUS_BUSY;
    }

    transfer_bytes = (uint64_t)packet->bytes_per_sequence *
        (uint64_t)packet->active_sequence_count;
    if ((packet->flags &
            SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD) != 0u)
    {
        transfer_bytes +=
            (uint64_t)packet->sideband_bytes_per_sequence *
            (uint64_t)packet->active_sequence_count;
    }
    completion = &state->completions[state->completion_tail];
    memset(completion, 0, sizeof(*completion));
    completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    completion->status = packet_status;
    completion->active_sequence_count = packet->active_sequence_count;
    completion->sequence_id = packet->sequence_id;
    completion->token_index = packet->token_index;
    completion->transfer_bytes = transfer_bytes;
    completion->service_time_ns = state->endpoint.validated_latency_ns;

    state->completion_tail =
        (state->completion_tail + 1u) %
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH;
    state->completion_count += 1u;
    state->completion_total_count += 1u;
    return SPARK_STATUS_OK;
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
    if (state->completion_count == 0u)
    {
        completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
        completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
        completion->status = SPARK_STATUS_BUSY;
        return SPARK_STATUS_OK;
    }

    *completion = state->completions[state->completion_head];
    state->completion_head =
        (state->completion_head + 1u) %
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH;
    state->completion_count -= 1u;
    return SPARK_STATUS_OK;
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
    statistics->completion_count = state->completion_total_count;
    statistics->dropped_completion_count = state->dropped_completion_count;
    statistics->queued_completion_count = state->completion_count;
    statistics->queue_depth =
        SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_DEFAULT_QUEUE_DEPTH;
    return SPARK_STATUS_OK;
}

SparkStatus SparkHiddenTransportGpudirectRdmaVerbsPreflight(
    const SparkHiddenTransportEndpoint *endpoint,
    const char *peermem_sysfs_path,
    const char *infiniband_sysfs_path)
{
    SparkStatus status;
    const char *peermem_path;
    const char *infiniband_path;

    status = SparkHiddenTransportValidateEndpoint(endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (!SparkHiddenTransportStringsEqual(
            endpoint->transport_module_id,
            SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_VERBS_MODULE_ID))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((endpoint->capability_flags &
            SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS) !=
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    peermem_path = peermem_sysfs_path;
    if (peermem_path == 0 || peermem_path[0] == '\0')
    {
        peermem_path =
            SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_PEERMEM_SYSFS_PATH;
    }
    infiniband_path = infiniband_sysfs_path;
    if (infiniband_path == 0 || infiniband_path[0] == '\0')
    {
        infiniband_path =
            SPARK_HIDDEN_TRANSPORT_GPUDIRECT_RDMA_INFINIBAND_SYSFS_PATH;
    }
    if (access(peermem_path, F_OK) != 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (access(infiniband_path, F_OK) != 0)
    {
        return SPARK_STATUS_ROUTE_NOT_FOUND;
    }
    return SPARK_STATUS_OK;
}
