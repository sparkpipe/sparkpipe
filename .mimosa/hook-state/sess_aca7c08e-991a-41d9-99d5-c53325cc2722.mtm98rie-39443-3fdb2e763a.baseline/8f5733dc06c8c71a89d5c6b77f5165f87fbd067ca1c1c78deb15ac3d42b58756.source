#include "sparkpipe/spark_draft_bridge.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define SPARK_DRAFT_BRIDGE_HOST_BYTES 256u
#define SPARK_DRAFT_BRIDGE_PORT_TEXT_BYTES 8u
#define SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC "DFT3"
#define SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES 4u
#define SPARK_DRAFT_BRIDGE_MS_PER_SECOND 1000u
#define SPARK_DRAFT_BRIDGE_US_PER_MS 1000u
#define SPARK_DRAFT_BRIDGE_DISCONNECTED_FD (-1)

#ifdef MSG_NOSIGNAL
#define SPARK_DRAFT_BRIDGE_SEND_FLAGS MSG_NOSIGNAL
#else
#define SPARK_DRAFT_BRIDGE_SEND_FLAGS 0
#endif

struct SparkDraftBridge
{
    char host[SPARK_DRAFT_BRIDGE_HOST_BYTES];
    char target_model[SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES];
    uint8_t *request_buffer;
    uint8_t *response_buffer;
    uint64_t request_buffer_bytes;
    uint64_t response_buffer_bytes;
    uint32_t port;
    uint32_t speculator_mask;
    uint32_t tap_row_bytes;
    uint32_t max_committed_tokens;
    uint32_t max_tap_rows;
    uint32_t max_nodes;
    uint32_t connect_timeout_ms;
    uint32_t io_timeout_ms;
    int socket_fd;
};

static void SparkDraftBridgePutU32Le(
    uint8_t *destination,
    uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFFu);
    destination[1] = (uint8_t)((value >> 8u) & 0xFFu);
    destination[2] = (uint8_t)((value >> 16u) & 0xFFu);
    destination[3] = (uint8_t)((value >> 24u) & 0xFFu);
}

static void SparkDraftBridgePutU64Le(
    uint8_t *destination,
    uint64_t value)
{
    uint32_t shift;

    for (shift = 0u; shift < 64u; shift += 8u)
    {
        destination[shift / 8u] = (uint8_t)((value >> shift) & 0xFFu);
    }
}

static uint32_t SparkDraftBridgeGetU32Le(
    const uint8_t *source)
{
    return (uint32_t)source[0] |
        ((uint32_t)source[1] << 8u) |
        ((uint32_t)source[2] << 16u) |
        ((uint32_t)source[3] << 24u);
}

static uint64_t SparkDraftBridgeGetU64Le(
    const uint8_t *source)
{
    uint64_t value;
    uint32_t shift;

    value = 0u;
    for (shift = 0u; shift < 64u; shift += 8u)
    {
        value |= (uint64_t)source[shift / 8u] << shift;
    }
    return value;
}

static float SparkDraftBridgeGetF32Le(
    const uint8_t *source)
{
    uint32_t bits;
    float value;

    bits = SparkDraftBridgeGetU32Le(source);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void SparkDraftBridgeDropConnection(
    SparkDraftBridge *bridge)
{
    if (bridge->socket_fd != SPARK_DRAFT_BRIDGE_DISCONNECTED_FD)
    {
        (void)close(bridge->socket_fd);
        bridge->socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    }
}

static SparkStatus SparkDraftBridgeFailTransport(
    SparkDraftBridge *bridge)
{
    SparkDraftBridgeDropConnection(bridge);
    return SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkDraftBridgeWaitConnected(
    int socket_fd,
    const struct sockaddr *address,
    socklen_t address_bytes,
    uint32_t timeout_ms)
{
    struct pollfd poll_descriptor;
    int flags;
    int connect_status;
    int poll_status;
    int socket_error;
    socklen_t socket_error_bytes;

    flags = fcntl(socket_fd, F_GETFL, 0);
    if (flags < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    connect_status = connect(socket_fd, address, address_bytes);
    if (connect_status != 0)
    {
        if (errno != EINPROGRESS)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        memset(&poll_descriptor, 0, sizeof(poll_descriptor));
        poll_descriptor.fd = socket_fd;
        poll_descriptor.events = POLLOUT;
        do
        {
            poll_status = poll(
                &poll_descriptor,
                1u,
                (int)timeout_ms);
        } while (poll_status < 0 && errno == EINTR);
        if (poll_status <= 0 ||
            (poll_descriptor.revents & (POLLOUT | POLLERR | POLLHUP)) == 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        socket_error = 0;
        socket_error_bytes = (socklen_t)sizeof(socket_error);
        if (getsockopt(
                socket_fd,
                SOL_SOCKET,
                SO_ERROR,
                &socket_error,
                &socket_error_bytes) != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        if (socket_error != 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
    }
    if (fcntl(socket_fd, F_SETFL, flags) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeConfigureSocket(
    int socket_fd,
    uint32_t io_timeout_ms)
{
    struct timeval timeout;
    int enabled;

    enabled = 1;
    if (setsockopt(
            socket_fd,
            IPPROTO_TCP,
            TCP_NODELAY,
            &enabled,
            (socklen_t)sizeof(enabled)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    timeout.tv_sec = (time_t)(io_timeout_ms / SPARK_DRAFT_BRIDGE_MS_PER_SECOND);
    timeout.tv_usec =
        (suseconds_t)((io_timeout_ms % SPARK_DRAFT_BRIDGE_MS_PER_SECOND) *
            SPARK_DRAFT_BRIDGE_US_PER_MS);
    if (setsockopt(
            socket_fd,
            SOL_SOCKET,
            SO_RCVTIMEO,
            &timeout,
            (socklen_t)sizeof(timeout)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (setsockopt(
            socket_fd,
            SOL_SOCKET,
            SO_SNDTIMEO,
            &timeout,
            (socklen_t)sizeof(timeout)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
#ifdef SO_NOSIGPIPE
    enabled = 1;
    if (setsockopt(
            socket_fd,
            SOL_SOCKET,
            SO_NOSIGPIPE,
            &enabled,
            (socklen_t)sizeof(enabled)) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
#endif
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeConnect(
    SparkDraftBridge *bridge)
{
    struct addrinfo hints;
    struct addrinfo *address_list;
    struct addrinfo *address;
    char port_text[SPARK_DRAFT_BRIDGE_PORT_TEXT_BYTES];
    int socket_fd;
    int address_status;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    (void)snprintf(
        port_text,
        sizeof(port_text),
        "%u",
        (unsigned)bridge->port);
    address_list = 0;
    address_status = getaddrinfo(
        bridge->host,
        port_text,
        &hints,
        &address_list);
    if (address_status != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    for (address = address_list; address != 0; address = address->ai_next)
    {
        socket_fd = socket(
            address->ai_family,
            address->ai_socktype,
            address->ai_protocol);
        if (socket_fd < 0)
        {
            continue;
        }
        if (SparkDraftBridgeWaitConnected(
                socket_fd,
                address->ai_addr,
                address->ai_addrlen,
                bridge->connect_timeout_ms) == SPARK_STATUS_OK &&
            SparkDraftBridgeConfigureSocket(
                socket_fd,
                bridge->io_timeout_ms) == SPARK_STATUS_OK)
        {
            break;
        }
        (void)close(socket_fd);
        socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    }
    freeaddrinfo(address_list);
    if (socket_fd == SPARK_DRAFT_BRIDGE_DISCONNECTED_FD)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    bridge->socket_fd = socket_fd;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeWriteAll(
    int socket_fd,
    const uint8_t *data,
    uint64_t data_bytes)
{
    uint64_t offset;
    ssize_t written;

    offset = 0u;
    while (offset < data_bytes)
    {
        written = send(
            socket_fd,
            data + offset,
            (size_t)(data_bytes - offset),
            SPARK_DRAFT_BRIDGE_SEND_FLAGS);
        if (written < 0 && errno == EINTR)
        {
            continue;
        }
        if (written <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        offset += (uint64_t)written;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeReadAll(
    int socket_fd,
    uint8_t *data,
    uint64_t data_bytes)
{
    uint64_t offset;
    ssize_t received;

    offset = 0u;
    while (offset < data_bytes)
    {
        received = recv(
            socket_fd,
            data + offset,
            (size_t)(data_bytes - offset),
            0);
        if (received < 0 && errno == EINTR)
        {
            continue;
        }
        if (received <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        offset += (uint64_t)received;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkDraftBridgeValidateConfig(
    const SparkDraftBridgeConfig *config)
{
    if (config == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (config->abi_version != SPARK_DRAFT_BRIDGE_ABI_VERSION ||
        config->descriptor_bytes != SPARK_DRAFT_BRIDGE_CONFIG_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (config->host == 0 ||
        config->host[0] == '\0' ||
        strlen(config->host) >= SPARK_DRAFT_BRIDGE_HOST_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (config->port == 0u || config->port > SPARK_DRAFT_BRIDGE_MAX_PORT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (config->target_model[0] == '\0' ||
        memchr(
            config->target_model,
            '\0',
            SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES) == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (config->tap_row_bytes == 0u ||
        config->max_committed_tokens == 0u ||
        config->max_tap_rows == 0u ||
        config->max_nodes == 0u ||
        config->max_nodes > SPARK_DRAFT_BRIDGE_MAX_TREE_NODES ||
        config->connect_timeout_ms == 0u ||
        config->io_timeout_ms == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkDraftBridgeInitialize(
    const SparkDraftBridgeConfig *config,
    SparkDraftBridge **bridge_out)
{
    SparkDraftBridge *bridge;
    SparkStatus status;
    uint64_t request_bytes;
    uint64_t response_bytes;

    if (bridge_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *bridge_out = 0;
    status = SparkDraftBridgeValidateConfig(config);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    bridge = (SparkDraftBridge *)calloc(1u, sizeof(*bridge));
    if (bridge == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    bridge->socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    memcpy(bridge->host, config->host, strlen(config->host) + 1u);
    memcpy(
        bridge->target_model,
        config->target_model,
        SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES);
    bridge->port = config->port;
    bridge->speculator_mask = config->speculator_mask;
    bridge->tap_row_bytes = config->tap_row_bytes;
    bridge->max_committed_tokens = config->max_committed_tokens;
    bridge->max_tap_rows = config->max_tap_rows;
    bridge->max_nodes = config->max_nodes;
    bridge->connect_timeout_ms = config->connect_timeout_ms;
    bridge->io_timeout_ms = config->io_timeout_ms;
    request_bytes =
        (uint64_t)SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES +
        (uint64_t)config->max_committed_tokens * sizeof(uint32_t) +
        (uint64_t)config->max_tap_rows * config->tap_row_bytes;
    response_bytes =
        (uint64_t)SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES +
        (uint64_t)config->max_nodes *
            SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES +
        (uint64_t)SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES;
    bridge->request_buffer = (uint8_t *)malloc((size_t)request_bytes);
    bridge->response_buffer = (uint8_t *)malloc((size_t)response_bytes);
    if (bridge->request_buffer == 0 || bridge->response_buffer == 0)
    {
        free(bridge->request_buffer);
        free(bridge->response_buffer);
        free(bridge);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    bridge->request_buffer_bytes = request_bytes;
    bridge->response_buffer_bytes = response_bytes;
    status = SparkDraftBridgeConnect(bridge);
    if (status != SPARK_STATUS_OK)
    {
        SparkDraftBridgeDestroy(bridge);
        return status;
    }
    *bridge_out = bridge;
    return SPARK_STATUS_OK;
}

void SparkDraftBridgeDestroy(
    SparkDraftBridge *bridge)
{
    if (bridge == 0)
    {
        return;
    }
    SparkDraftBridgeDropConnection(bridge);
    free(bridge->request_buffer);
    free(bridge->response_buffer);
    free(bridge);
}

SparkStatus SparkDraftBridgeProposeTree(
    SparkDraftBridge *bridge,
    uint64_t sequence_id,
    uint64_t generation,
    uint64_t position,
    uint32_t anchor_token_id,
    const uint32_t *committed_token_ids,
    uint32_t committed_token_count,
    const void *tap_rows,
    uint32_t tap_row_count,
    uint32_t depth,
    uint32_t time_budget_ms,
    uint32_t max_depth,
    SparkDraftBridgeNode *nodes_out,
    uint32_t node_capacity,
    uint32_t *node_count_out,
    SparkDraftBridgeProposalInfo *info_out)
{
    SparkStatus status;
    uint8_t *cursor;
    const uint8_t *reader;
    const uint8_t *record;
    const uint8_t *footer;
    uint64_t request_bytes;
    uint64_t response_node_bytes;
    uint64_t response_sequence_id;
    uint64_t response_generation;
    uint32_t server_status;
    uint32_t node_count;
    uint32_t node_index;
    uint32_t stage_index;

    if (bridge == 0 ||
        committed_token_ids == 0 ||
        nodes_out == 0 ||
        node_count_out == 0 ||
        info_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (committed_token_count == 0u ||
        node_capacity == 0u ||
        (uint64_t)committed_token_count != position + 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (committed_token_count > bridge->max_committed_tokens ||
        tap_row_count > bridge->max_tap_rows ||
        node_capacity > bridge->max_nodes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (tap_row_count != 0u && tap_rows == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (depth == 0u ||
        depth > SPARK_DRAFT_BRIDGE_MAX_DEPTH ||
        max_depth == 0u ||
        max_depth > SPARK_DRAFT_BRIDGE_MAX_TREE_DEPTH ||
        time_budget_ms > SPARK_DRAFT_BRIDGE_MAX_TIME_BUDGET_MS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (committed_token_ids[position] != anchor_token_id)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *node_count_out = 0u;
    memset(info_out, 0, sizeof(*info_out));

    if (bridge->socket_fd == SPARK_DRAFT_BRIDGE_DISCONNECTED_FD)
    {
        status = SparkDraftBridgeConnect(bridge);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    cursor = bridge->request_buffer;
    memcpy(
        cursor,
        SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC,
        SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES);
    cursor += SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES;
    SparkDraftBridgePutU32Le(cursor, SPARK_DRAFT_BRIDGE_PROTOCOL_VERSION);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, bridge->speculator_mask);
    cursor += sizeof(uint32_t);
    memcpy(
        cursor,
        bridge->target_model,
        SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES);
    cursor += SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES;
    SparkDraftBridgePutU64Le(cursor, sequence_id);
    cursor += sizeof(uint64_t);
    SparkDraftBridgePutU64Le(cursor, generation);
    cursor += sizeof(uint64_t);
    SparkDraftBridgePutU64Le(cursor, position);
    cursor += sizeof(uint64_t);
    SparkDraftBridgePutU32Le(cursor, anchor_token_id);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, committed_token_count);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, tap_row_count);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, depth);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, time_budget_ms);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, max_depth);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, node_capacity);
    cursor += sizeof(uint32_t);
    for (node_index = 0u; node_index < committed_token_count; ++node_index)
    {
        SparkDraftBridgePutU32Le(cursor, committed_token_ids[node_index]);
        cursor += sizeof(uint32_t);
    }
    if (tap_row_count != 0u)
    {
        memcpy(
            cursor,
            tap_rows,
            (size_t)tap_row_count * bridge->tap_row_bytes);
        cursor += (size_t)tap_row_count * bridge->tap_row_bytes;
    }
    request_bytes = (uint64_t)(cursor - bridge->request_buffer);

    status = SparkDraftBridgeWriteAll(
        bridge->socket_fd,
        bridge->request_buffer,
        request_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    status = SparkDraftBridgeReadAll(
        bridge->socket_fd,
        bridge->response_buffer,
        SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES);
    if (status != SPARK_STATUS_OK)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    if (memcmp(
            bridge->response_buffer,
            SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC,
            SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES) != 0)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    reader = bridge->response_buffer + SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES;
    server_status = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    node_count = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    response_sequence_id = SparkDraftBridgeGetU64Le(reader);
    reader += sizeof(uint64_t);
    response_generation = SparkDraftBridgeGetU64Le(reader);
    if (node_count > node_capacity)
    {
        SparkDraftBridgeDropConnection(bridge);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    response_node_bytes =
        (uint64_t)node_count * SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES;
    status = SparkDraftBridgeReadAll(
        bridge->socket_fd,
        bridge->response_buffer + SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES,
        response_node_bytes + SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES);
    if (status != SPARK_STATUS_OK)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    if (response_sequence_id != sequence_id ||
        response_generation != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    info_out->server_status = server_status;
    footer = bridge->response_buffer +
        SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES + response_node_bytes;
    reader = footer;
    for (stage_index = 0u;
         stage_index < SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT;
         ++stage_index)
    {
        info_out->drafter_us[stage_index] = SparkDraftBridgeGetU32Le(reader);
        reader += sizeof(uint32_t);
    }
    info_out->graft_us = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    info_out->expansions = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    info_out->elapsed_us = SparkDraftBridgeGetU32Le(reader);
    if (server_status != SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    for (node_index = 0u; node_index < node_count; ++node_index)
    {
        record = bridge->response_buffer +
            SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES +
            (uint64_t)node_index * SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES;
        nodes_out[node_index].token_id = SparkDraftBridgeGetU32Le(record);
        record += sizeof(uint32_t);
        nodes_out[node_index].parent_index = SparkDraftBridgeGetU32Le(record);
        record += sizeof(uint32_t);
        nodes_out[node_index].depth = SparkDraftBridgeGetU32Le(record);
        record += sizeof(uint32_t);
        nodes_out[node_index].source_bit = SparkDraftBridgeGetU32Le(record);
        record += sizeof(uint32_t);
        nodes_out[node_index].score = SparkDraftBridgeGetF32Le(record);
    }
    *node_count_out = node_count;
    return SPARK_STATUS_OK;
}
