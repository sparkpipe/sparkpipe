#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_device_collective.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SPARK_TP_DEVICE_COLLECTIVE_PORT_STRIDE 64u
#define SPARK_TP_DEVICE_COLLECTIVE_POLL_SLEEP_NANOSECONDS 10000L

static int SparkTpDeviceCollectiveDegreeIsSupported(uint32_t tp_degree)
{
    return tp_degree == 1u || tp_degree == 2u || tp_degree == 4u ||
        tp_degree == 8u || tp_degree == 16u;
}

static uint32_t SparkTpDeviceCollectiveStepCount(uint32_t tp_degree)
{
    uint32_t step_count;

    step_count = 0u;
    while ((tp_degree >> (step_count + 1u)) != 0u)
    {
        step_count += 1u;
    }
    return step_count;
}

static uint64_t SparkTpDeviceCollectiveNowMilli(void)
{
    struct timespec current_time;

    if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0)
    {
        return UINT64_MAX;
    }
    return ((uint64_t)current_time.tv_sec * 1000u) +
        ((uint64_t)current_time.tv_nsec / 1000000u);
}

static SparkStatus SparkTpDeviceCollectiveBuildDeadline(
    uint32_t timeout_milli,
    uint64_t *deadline_milli)
{
    uint64_t now_milli;

    if (deadline_milli == NULL || timeout_milli == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    now_milli = SparkTpDeviceCollectiveNowMilli();
    if (now_milli == UINT64_MAX)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (UINT64_MAX - now_milli < (uint64_t)timeout_milli)
    {
        *deadline_milli = UINT64_MAX;
    }
    else
    {
        *deadline_milli = now_milli + (uint64_t)timeout_milli;
    }
    return SPARK_STATUS_OK;
}

static int SparkTpDeviceCollectiveTextIsValid(const char *text)
{
    return text != NULL && text[0] != '\0';
}

static SparkStatus SparkTpDeviceCollectiveValidateConfig(
    const SparkTpDeviceCollectiveConfig *config)
{
    uint32_t rank_index;

    if (config == NULL ||
        config->abi_version != SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION ||
        !SparkTpDeviceCollectiveDegreeIsSupported(config->tp_degree) ||
        config->tp_rank >= config->tp_degree ||
        config->local_hidden_dimension == 0u ||
        config->max_active_sequence_count == 0u ||
        config->connect_timeout_milli == 0u ||
        config->operation_timeout_milli == 0u ||
        config->control_port_base == 0u ||
        config->collective_identifier == 0u ||
        !SparkTpDeviceCollectiveTextIsValid(config->transport_module_path) ||
        !SparkTpDeviceCollectiveTextIsValid(config->local_host))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (rank_index = 0u;
         rank_index < config->tp_degree;
         ++rank_index)
    {
        if (!SparkTpDeviceCollectiveTextIsValid(config->rank_hosts[rank_index]))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveBuildEndpoint(
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t step_index,
    uint32_t source_rank,
    uint32_t sink_rank,
    uint32_t hidden_dimension,
    char *route_name,
    SparkHiddenTransportEndpoint *endpoint)
{
    uint32_t port_base;
    int written;

    if (config == NULL || route_name == NULL || endpoint == NULL ||
        source_rank >= config->tp_degree || sink_rank >= config->tp_degree ||
        source_rank == sink_rank || step_index >=
            SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (step_index > (UINT32_MAX - config->control_port_base) /
            SPARK_TP_DEVICE_COLLECTIVE_PORT_STRIDE)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    port_base = config->control_port_base +
        step_index * SPARK_TP_DEVICE_COLLECTIVE_PORT_STRIDE;
    if (port_base > UINT16_MAX - sink_rank)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    written = snprintf(
        route_name,
        SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES,
        "tp-device.%u.%u.%u",
        step_index,
        source_rank,
        sink_rank);
    if (written < 0 || (uint32_t)written >=
        SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    SparkHiddenTransportInitializeSparkGpudirectRdmaEndpoint(
        endpoint,
        hidden_dimension,
        config->max_active_sequence_count,
        0u,
        route_name);
    endpoint->configuration_flags =
        SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION;
    endpoint->local_rank_index = config->tp_rank;
    endpoint->source_rank_index = source_rank;
    endpoint->sink_rank_index = sink_rank;
    endpoint->control_port_base = port_base;
    endpoint->source_host = source_rank == config->tp_rank ?
        config->local_host : config->rank_hosts[source_rank];
    endpoint->sink_host = sink_rank == config->tp_rank ?
        config->local_host : config->rank_hosts[sink_rank];
    return SparkHiddenTransportValidateEndpoint(endpoint);
}

static void SparkTpDeviceCollectiveCloseSessions(
    SparkTpDeviceCollective *collective)
{
    uint32_t step_index;

    if (collective == NULL)
    {
        return;
    }
    for (step_index = 0u;
         step_index < SPARK_TP_DEVICE_COLLECTIVE_MAX_STEPS;
         ++step_index)
    {
        if (collective->send_sessions[step_index] != NULL)
        {
            SparkHiddenTransportClose(collective->send_sessions[step_index]);
            collective->send_sessions[step_index] = NULL;
        }
        if (collective->receive_sessions[step_index] != NULL)
        {
            SparkHiddenTransportClose(
                collective->receive_sessions[step_index]);
            collective->receive_sessions[step_index] = NULL;
        }
    }
}

static SparkStatus SparkTpDeviceCollectiveFail(
    SparkTpDeviceCollective *collective,
    SparkStatus status)
{
    SparkTpDeviceCollectiveCloseSessions(collective);
    if (collective != NULL)
    {
        SparkHiddenTransportUnloadInterface(&collective->transport_library);
        collective->failed = 1u;
    }
    return status;
}

static SparkStatus SparkTpDeviceCollectiveOpenStep(
    SparkTpDeviceCollective *collective,
    const SparkTpDeviceCollectiveConfig *config,
    uint32_t step_index)
{
    uint32_t partner_rank;
    uint32_t source_rank;
    uint32_t sink_rank;
    SparkHiddenTransportEndpoint endpoint;
    SparkStatus status;

    partner_rank = config->tp_rank ^ (1u << step_index);
    source_rank = config->tp_rank;
    sink_rank = partner_rank;
    memset(&endpoint, 0, sizeof(endpoint));
    status = SparkTpDeviceCollectiveBuildEndpoint(
        config,
        step_index,
        source_rank,
        sink_rank,
        collective->step_hidden_dimensions[step_index],
        collective->send_route_names[step_index],
        &endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkHiddenTransportOpen(
        &endpoint,
        &collective->transport_library.transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS,
        &collective->send_sessions[step_index]);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    memset(&endpoint, 0, sizeof(endpoint));
    status = SparkTpDeviceCollectiveBuildEndpoint(
        config,
        step_index,
        partner_rank,
        config->tp_rank,
        collective->step_hidden_dimensions[step_index],
        collective->receive_route_names[step_index],
        &endpoint);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkHiddenTransportOpen(
        &endpoint,
        &collective->transport_library.transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS,
        &collective->receive_sessions[step_index]);
    return status;
}

SparkStatus SparkTpDeviceCollectiveCreate(
    const SparkTpDeviceCollectiveConfig *config,
    SparkTpDeviceCollective *collective_out)
{
    uint32_t step_index;
    uint32_t step_count;
    SparkStatus status;

    if (collective_out == NULL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(collective_out, 0, sizeof(*collective_out));
    status = SparkTpDeviceCollectiveValidateConfig(config);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    step_count = SparkTpDeviceCollectiveStepCount(config->tp_degree);
    collective_out->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    collective_out->tp_degree = config->tp_degree;
    collective_out->tp_rank = config->tp_rank;
    collective_out->step_count = step_count;
    collective_out->local_hidden_dimension = config->local_hidden_dimension;
    collective_out->max_active_sequence_count =
        config->max_active_sequence_count;
    collective_out->operation_timeout_milli = config->operation_timeout_milli;
    collective_out->collective_identifier = config->collective_identifier;
    collective_out->next_operation_sequence = 1u;
    for (step_index = 0u; step_index < step_count; ++step_index)
    {
        uint64_t step_hidden_dimension;

        step_hidden_dimension = (uint64_t)config->local_hidden_dimension <<
            step_index;
        if (step_hidden_dimension > UINT32_MAX)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        collective_out->step_hidden_dimensions[step_index] =
            (uint32_t)step_hidden_dimension;
    }
    status = SparkHiddenTransportLoadInterfaceFromSharedObject(
        config->transport_module_path,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_SPARK_GPUDIRECT_RDMA_CAPS,
        &collective_out->transport_library);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (step_index = 0u;
         status == SPARK_STATUS_OK && step_index < step_count;
         ++step_index)
    {
        status = SparkTpDeviceCollectiveOpenStep(
            collective_out,
            config,
            step_index);
    }
    if (status != SPARK_STATUS_OK)
    {
        return SparkTpDeviceCollectiveFail(collective_out, status);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectiveBuildPacket(
    const void *device_pointer,
    uint32_t rows,
    uint32_t hidden_dimension,
    uint64_t operation_sequence,
    uint32_t step_index,
    void *cuda_stream,
    SparkHiddenTransportPacket *packet)
{
    if (device_pointer == NULL || rows == 0u || hidden_dimension == 0u ||
        packet == NULL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(packet, 0, sizeof(*packet));
    packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    packet->active_sequence_count = rows;
    packet->hidden_dimension = hidden_dimension;
    packet->bytes_per_sequence = hidden_dimension *
        SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT;
    packet->sequence_id = operation_sequence;
    packet->token_index = step_index;
    packet->hidden_bf16 = device_pointer;
    packet->cuda_stream = cuda_stream;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTpDeviceCollectivePollCompletion(
    SparkHiddenTransportSession *session,
    const SparkHiddenTransportPacket *packet,
    uint32_t *complete)
{
    SparkHiddenTransportCompletion completion;
    SparkStatus status;

    if (session == NULL || packet == NULL || complete == NULL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkHiddenTransportPoll(session, &completion);
    if (status == SPARK_STATUS_BUSY)
    {
        return SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (completion.status == SPARK_STATUS_BUSY)
    {
        return SPARK_STATUS_OK;
    }
    if (completion.status != SPARK_STATUS_OK ||
        completion.sequence_id != packet->sequence_id ||
        completion.token_index != packet->token_index ||
        completion.active_sequence_count != packet->active_sequence_count ||
        completion.transfer_bytes != (uint64_t)packet->bytes_per_sequence *
            packet->active_sequence_count)
    {
        return completion.status != SPARK_STATUS_OK ? completion.status :
            SPARK_STATUS_VALIDATION_FAILED;
    }
    *complete = 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveExchangeBf16(
    SparkTpDeviceCollective *collective,
    const void *send_device,
    void *receive_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    uint32_t step_index,
    void *cuda_stream)
{
    SparkHiddenTransportPacket send_packet;
    SparkHiddenTransportPacket receive_packet;
    uint64_t deadline_milli;
    uint64_t operation_sequence;
    uint32_t send_posted;
    uint32_t receive_posted;
    uint32_t send_complete;
    uint32_t receive_complete;

    if (collective == NULL || collective->abi_version !=
        SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION || collective->failed != 0u ||
        step_index >= collective->step_count || send_device == NULL ||
        receive_device == NULL || active_sequence_count == 0u ||
        active_sequence_count > collective->max_active_sequence_count ||
        hidden_dimension != collective->step_hidden_dimensions[step_index] ||
        cuda_stream == NULL || collective->next_operation_sequence == 0u ||
        collective->next_operation_sequence == UINT64_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    operation_sequence = collective->next_operation_sequence;
    if (SparkTpDeviceCollectiveBuildPacket(
            send_device,
            active_sequence_count,
            hidden_dimension,
            operation_sequence,
            step_index,
            cuda_stream,
            &send_packet) != SPARK_STATUS_OK ||
        SparkTpDeviceCollectiveBuildPacket(
            receive_device,
            active_sequence_count,
            hidden_dimension,
            operation_sequence,
            step_index,
            cuda_stream,
            &receive_packet) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkTpDeviceCollectiveBuildDeadline(
            collective->operation_timeout_milli,
            &deadline_milli) != SPARK_STATUS_OK)
    {
        return SparkTpDeviceCollectiveFail(
            collective,
            SPARK_STATUS_IO_ERROR);
    }
    send_posted = 0u;
    receive_posted = 0u;
    send_complete = 0u;
    receive_complete = 0u;
    while (send_complete == 0u || receive_complete == 0u)
    {
        SparkStatus status;
        struct timespec delay;

        if (SparkTpDeviceCollectiveNowMilli() >= deadline_milli)
        {
            return SparkTpDeviceCollectiveFail(
                collective,
                SPARK_STATUS_IO_ERROR);
        }
        if (receive_posted == 0u)
        {
            status = SparkHiddenTransportPostReceive(
                collective->receive_sessions[step_index],
                &receive_packet);
            if (status == SPARK_STATUS_OK)
                receive_posted = 1u;
            else if (status != SPARK_STATUS_BUSY)
                return SparkTpDeviceCollectiveFail(collective, status);
        }
        if (send_posted == 0u)
        {
            status = SparkHiddenTransportSend(
                collective->send_sessions[step_index],
                &send_packet);
            if (status == SPARK_STATUS_OK)
                send_posted = 1u;
            else if (status != SPARK_STATUS_BUSY)
                return SparkTpDeviceCollectiveFail(collective, status);
        }
        if (receive_posted != 0u && receive_complete == 0u)
        {
            status = SparkTpDeviceCollectivePollCompletion(
                collective->receive_sessions[step_index],
                &receive_packet,
                &receive_complete);
            if (status != SPARK_STATUS_OK)
                return SparkTpDeviceCollectiveFail(collective, status);
        }
        if (send_posted != 0u && send_complete == 0u)
        {
            status = SparkTpDeviceCollectivePollCompletion(
                collective->send_sessions[step_index],
                &send_packet,
                &send_complete);
            if (status != SPARK_STATUS_OK)
                return SparkTpDeviceCollectiveFail(collective, status);
        }
        if (send_complete == 0u || receive_complete == 0u)
        {
            delay.tv_sec = 0;
            delay.tv_nsec = SPARK_TP_DEVICE_COLLECTIVE_POLL_SLEEP_NANOSECONDS;
            while (nanosleep(&delay, &delay) != 0)
            {
            }
        }
    }
    collective->next_operation_sequence += 1u;
    return SPARK_STATUS_OK;
}

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    if (collective == NULL)
    {
        return;
    }
    SparkTpDeviceCollectiveCloseSessions(collective);
    SparkHiddenTransportUnloadInterface(&collective->transport_library);
    memset(collective, 0, sizeof(*collective));
}
