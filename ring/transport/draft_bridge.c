#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_io.h"
#include "sparkpipe/spark_error_site.h"

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
#include <unistd.h>

#define SPARK_DRAFT_BRIDGE_HOST_BYTES 256u
#define SPARK_DRAFT_BRIDGE_PORT_TEXT_BYTES 8u
#define SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC "DFT3"
#define SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES 4u
#define SPARK_DRAFT_BRIDGE_NS_PER_MS UINT64_C(1000000)
#define SPARK_DRAFT_BRIDGE_DISCONNECTED_FD (-1)

#define SPARK_DRAFT_BRIDGE_PHASE_IDLE 0u
#define SPARK_DRAFT_BRIDGE_PHASE_CONNECT 1u
#define SPARK_DRAFT_BRIDGE_PHASE_SEND 2u
#define SPARK_DRAFT_BRIDGE_PHASE_HEADER 3u
#define SPARK_DRAFT_BRIDGE_PHASE_BODY 4u
#define SPARK_DRAFT_BRIDGE_PHASE_FAILED 5u


struct SparkDraftBridge
{
    char host[SPARK_DRAFT_BRIDGE_HOST_BYTES];
    char target_model[SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES];
    uint8_t *request_buffer;
    uint8_t *response_buffer;
    uint64_t request_buffer_bytes;
    uint64_t response_buffer_bytes;
    uint32_t port;
    uint32_t tap_row_bytes;
    uint32_t max_committed_tokens;
    uint32_t max_tap_rows;
    uint32_t max_nodes;
    uint32_t connect_timeout_ms;
    uint32_t io_timeout_ms;
    int socket_fd;
    struct addrinfo *address_list;
    struct addrinfo *address;
    uint64_t deadline_ns;
    uint64_t request_bytes;
    uint64_t request_sent;
    uint64_t response_bytes;
    uint64_t response_expected;
    uint64_t sequence_id;
    uint64_t generation;
    uint32_t node_capacity;
    uint32_t node_count;
    uint32_t server_status;
    uint32_t stale;
    uint32_t phase;
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

static void SparkDraftBridgeFreeAddresses(
    SparkDraftBridge *bridge)
{
    if (bridge->address_list != 0)
    {
        freeaddrinfo(bridge->address_list);
    }
    bridge->address_list = 0;
    bridge->address = 0;
}

static void SparkDraftBridgeDropConnection(
    SparkDraftBridge *bridge)
{
    if (bridge->socket_fd != SPARK_DRAFT_BRIDGE_DISCONNECTED_FD)
    {
        (void)close(bridge->socket_fd);
        bridge->socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    }
    SparkDraftBridgeFreeAddresses(bridge);
}

static SparkStatus SparkDraftBridgeFailTransport(
    SparkDraftBridge *bridge)
{
    SparkDraftBridgeDropConnection(bridge);
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
    SPARK_FAIL(SPARK_STATUS_IO_ERROR);
}

static uint64_t SparkDraftBridgeDeadline(
    uint32_t timeout_ms)
{
    return SparkMonotonicNs() +
        (uint64_t)timeout_ms * SPARK_DRAFT_BRIDGE_NS_PER_MS;
}

static SparkStatus SparkDraftBridgeConfigureSocket(
    int socket_fd)
{
    int enabled;

    enabled = 1;
    if (setsockopt(
            socket_fd,
            IPPROTO_TCP,
            TCP_NODELAY,
            &enabled,
            (socklen_t)sizeof(enabled)) != 0)
    {
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
#endif
    return SPARK_STATUS_OK;
}

static void SparkDraftBridgeConnected(
    SparkDraftBridge *bridge)
{
    SparkDraftBridgeFreeAddresses(bridge);
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_SEND;
    bridge->deadline_ns = SparkDraftBridgeDeadline(bridge->io_timeout_ms);
}

static SparkStatus SparkDraftBridgeConnectNext(
    SparkDraftBridge *bridge)
{
    const struct addrinfo *address;
    int socket_fd;
    int flags;

    while (bridge->address != 0)
    {
        address = bridge->address;
        bridge->address = address->ai_next;
        socket_fd = socket(
            address->ai_family,
            address->ai_socktype,
            address->ai_protocol);
        if (socket_fd < 0)
        {
            continue;
        }
        flags = fcntl(socket_fd, F_GETFL, 0);
        if (flags < 0 ||
            fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) != 0 ||
            SparkDraftBridgeConfigureSocket(socket_fd) != SPARK_STATUS_OK)
        {
            (void)close(socket_fd);
            continue;
        }
        if (connect(socket_fd, address->ai_addr, address->ai_addrlen) == 0)
        {
            bridge->socket_fd = socket_fd;
            SparkDraftBridgeConnected(bridge);
            return SPARK_STATUS_OK;
        }
        if (errno == EINPROGRESS || errno == EINTR)
        {
            bridge->socket_fd = socket_fd;
            bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_CONNECT;
            bridge->deadline_ns =
                SparkDraftBridgeDeadline(bridge->connect_timeout_ms);
            return SPARK_STATUS_OK;
        }
        (void)close(socket_fd);
    }
    return SparkDraftBridgeFailTransport(bridge);
}

static SparkStatus SparkDraftBridgeConnect(
    SparkDraftBridge *bridge)
{
    struct addrinfo hints;
    char port_text[SPARK_DRAFT_BRIDGE_PORT_TEXT_BYTES];

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    (void)snprintf(
        port_text,
        sizeof(port_text),
        "%u",
        (unsigned)bridge->port);
    bridge->address_list = 0;
    if (getaddrinfo(
            bridge->host,
            port_text,
            &hints,
            &bridge->address_list) != 0)
    {
        bridge->address_list = 0;
        return SparkDraftBridgeFailTransport(bridge);
    }
    bridge->address = bridge->address_list;
    return SparkDraftBridgeConnectNext(bridge);
}

static SparkStatus SparkDraftBridgeAwaitConnect(
    SparkDraftBridge *bridge)
{
    struct pollfd poll_descriptor;
    int poll_status;
    int socket_error;
    socklen_t socket_error_bytes;

    memset(&poll_descriptor, 0, sizeof(poll_descriptor));
    poll_descriptor.fd = bridge->socket_fd;
    poll_descriptor.events = POLLOUT;
    do
    {
        poll_status = poll(&poll_descriptor, 1u, 0);
    } while (poll_status < 0 && errno == EINTR);
    if (poll_status == 0 && SparkMonotonicNs() < bridge->deadline_ns)
    {
        return SPARK_STATUS_PENDING;
    }
    socket_error = 0;
    socket_error_bytes = (socklen_t)sizeof(socket_error);
    if (poll_status > 0 &&
        getsockopt(
            bridge->socket_fd,
            SOL_SOCKET,
            SO_ERROR,
            &socket_error,
            &socket_error_bytes) == 0 &&
        socket_error == 0)
    {
        SparkDraftBridgeConnected(bridge);
        return SPARK_STATUS_OK;
    }
    (void)close(bridge->socket_fd);
    bridge->socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    return SparkDraftBridgeConnectNext(bridge);
}

static SparkStatus SparkDraftBridgeBlocked(
    SparkDraftBridge *bridge)
{
    if (SparkMonotonicNs() < bridge->deadline_ns)
    {
        return SPARK_STATUS_PENDING;
    }
    return SparkDraftBridgeFailTransport(bridge);
}

static SparkStatus SparkDraftBridgeSend(
    SparkDraftBridge *bridge)
{
    ssize_t put;

    while (bridge->request_sent < bridge->request_bytes)
    {
        put = SparkIoSend(
            bridge->socket_fd,
            bridge->request_buffer + bridge->request_sent,
            (size_t)(bridge->request_bytes - bridge->request_sent));
        if (put < 0 && errno == EINTR)
        {
            continue;
        }
        if (put < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            return SparkDraftBridgeBlocked(bridge);
        }
        if (put <= 0)
        {
            return SparkDraftBridgeFailTransport(bridge);
        }
        bridge->request_sent += (uint64_t)put;
    }
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_HEADER;
    bridge->response_bytes = 0u;
    bridge->response_expected = SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeReceive(
    SparkDraftBridge *bridge)
{
    ssize_t got;

    while (bridge->response_bytes < bridge->response_expected)
    {
        got = recv(
            bridge->socket_fd,
            bridge->response_buffer + bridge->response_bytes,
            (size_t)(bridge->response_expected - bridge->response_bytes),
            0);
        if (got < 0 && errno == EINTR)
        {
            continue;
        }
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            return SparkDraftBridgeBlocked(bridge);
        }
        if (got <= 0)
        {
            return SparkDraftBridgeFailTransport(bridge);
        }
        bridge->response_bytes += (uint64_t)got;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeReadHeader(
    SparkDraftBridge *bridge)
{
    const uint8_t *reader;
    uint64_t response_sequence_id;
    uint64_t response_generation;

    if (memcmp(
            bridge->response_buffer,
            SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC,
            SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES) != 0)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    reader = bridge->response_buffer + SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES;
    bridge->server_status = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    bridge->node_count = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    response_sequence_id = SparkDraftBridgeGetU64Le(reader);
    reader += sizeof(uint64_t);
    response_generation = SparkDraftBridgeGetU64Le(reader);
    if (bridge->node_count > bridge->node_capacity)
    {
        SparkDraftBridgeDropConnection(bridge);
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    bridge->stale =
        response_sequence_id != bridge->sequence_id ||
        response_generation != bridge->generation ? 1u : 0u;
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_BODY;
    bridge->response_expected =
        (uint64_t)SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES +
        (uint64_t)bridge->node_count * SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES +
        (uint64_t)SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeAdvance(
    SparkDraftBridge *bridge)
{
    SparkStatus status;

    status = SPARK_STATUS_OK;
    while (status == SPARK_STATUS_OK)
    {
        if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_CONNECT)
        {
            status = SparkDraftBridgeAwaitConnect(bridge);
        }
        else if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND)
        {
            status = SparkDraftBridgeSend(bridge);
        }
        else if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_HEADER)
        {
            status = SparkDraftBridgeReceive(bridge);
            if (status == SPARK_STATUS_OK)
            {
                status = SparkDraftBridgeReadHeader(bridge);
            }
        }
        else
        {
            return SparkDraftBridgeReceive(bridge);
        }
    }
    return status;
}

SparkStatus SparkDraftBridgeValidateConfig(
    const SparkDraftBridgeConfig *config)
{
    if (config == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (config->abi_version != SPARK_DRAFT_BRIDGE_ABI_VERSION ||
        config->descriptor_bytes != SPARK_DRAFT_BRIDGE_CONFIG_BYTES)
    {
        SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
    }
    if (config->host == 0 ||
        config->host[0] == '\0' ||
        strlen(config->host) >= SPARK_DRAFT_BRIDGE_HOST_BYTES)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (config->port == 0u || config->port > SPARK_DRAFT_BRIDGE_MAX_PORT)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (config->target_model[0] == '\0' ||
        memchr(
            config->target_model,
            '\0',
            SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES) == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (config->max_committed_tokens == 0u ||
        config->max_nodes == 0u ||
        config->max_nodes > SPARK_DRAFT_BRIDGE_MAX_TREE_NODES ||
        config->connect_timeout_ms == 0u ||
        config->io_timeout_ms == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if ((config->tap_row_bytes == 0u) != (config->max_tap_rows == 0u))
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    }
    bridge->socket_fd = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    memcpy(bridge->host, config->host, strlen(config->host) + 1u);
    memcpy(
        bridge->target_model,
        config->target_model,
        SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES);
    bridge->port = config->port;
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
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    }
    bridge->request_buffer_bytes = request_bytes;
    bridge->response_buffer_bytes = response_bytes;
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

SparkStatus SparkDraftBridgeRequestTree(
    SparkDraftBridge *bridge,
    uint32_t speculator_mask,
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
    const SparkRowSampling *sampling,
    uint32_t node_capacity)
{
    SparkStatus status;
    uint8_t *cursor;
    uint32_t token_index;

    if (bridge == 0 || committed_token_ids == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (committed_token_count == 0u ||
        node_capacity == 0u ||
        speculator_mask == 0u ||
        (uint64_t)committed_token_count != position + 1u)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (committed_token_count > bridge->max_committed_tokens ||
        tap_row_count > bridge->max_tap_rows ||
        node_capacity > bridge->max_nodes)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (tap_row_count != 0u && bridge->tap_row_bytes == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (tap_row_count != 0u && tap_rows == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (depth == 0u ||
        depth > SPARK_DRAFT_BRIDGE_MAX_DEPTH ||
        max_depth == 0u ||
        max_depth > SPARK_DRAFT_BRIDGE_MAX_TREE_DEPTH ||
        time_budget_ms > SPARK_DRAFT_BRIDGE_MAX_TIME_BUDGET_MS)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (committed_token_ids[position] != anchor_token_id)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (sampling != 0 && SparkSamplingRuleValid(sampling) == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (bridge->phase != SPARK_DRAFT_BRIDGE_PHASE_IDLE)
    {
        SPARK_FAIL(SPARK_STATUS_BUSY);
    }

    cursor = bridge->request_buffer;
    memcpy(
        cursor,
        SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC,
        SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES);
    cursor += SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES;
    SparkDraftBridgePutU32Le(cursor, SPARK_DRAFT_BRIDGE_PROTOCOL_VERSION);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, speculator_mask);
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
    {
        const uint64_t seed = sampling != 0 ? sampling->seed : 0u;
        const float inverse_temperature = sampling != 0 ? sampling->inverse_temperature : 0.0f;
        const float nucleus = sampling != 0 ? sampling->nucleus : 0.0f;
        uint32_t bits;
        SparkDraftBridgePutU64Le(cursor, seed);
        cursor += sizeof(uint64_t);
        memcpy(&bits, &inverse_temperature, sizeof(bits));
        SparkDraftBridgePutU32Le(cursor, bits);
        cursor += sizeof(uint32_t);
        memcpy(&bits, &nucleus, sizeof(bits));
        SparkDraftBridgePutU32Le(cursor, bits);
        cursor += sizeof(uint32_t);
        SparkDraftBridgePutU32Le(cursor, sampling != 0 ? sampling->top_k : 0u);
        cursor += sizeof(uint32_t);
        SparkDraftBridgePutU32Le(cursor, 0u);
        cursor += sizeof(uint32_t);
    }
    for (token_index = 0u; token_index < committed_token_count; ++token_index)
    {
        SparkDraftBridgePutU32Le(cursor, committed_token_ids[token_index]);
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
    bridge->request_bytes = (uint64_t)(cursor - bridge->request_buffer);
    bridge->request_sent = 0u;
    bridge->response_bytes = 0u;
    bridge->response_expected = 0u;
    bridge->sequence_id = sequence_id;
    bridge->generation = generation;
    bridge->node_capacity = node_capacity;
    bridge->node_count = 0u;
    bridge->server_status = 0u;
    bridge->stale = 0u;
    if (bridge->socket_fd == SPARK_DRAFT_BRIDGE_DISCONNECTED_FD)
    {
        status = SparkDraftBridgeConnect(bridge);
    }
    else
    {
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_SEND;
        bridge->deadline_ns = SparkDraftBridgeDeadline(bridge->io_timeout_ms);
        status = SPARK_STATUS_OK;
    }
    if (status == SPARK_STATUS_OK &&
        bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND)
    {
        status = SparkDraftBridgeSend(bridge);
    }
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_PENDING)
    {
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_FAILED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkDraftBridgeReply(
    SparkDraftBridge *bridge,
    SparkDraftBridgeNode *nodes_out,
    uint32_t *node_count_out,
    SparkDraftBridgeProposalInfo *info_out)
{
    SparkStatus status;
    const uint8_t *reader;
    const uint8_t *record;
    uint32_t node_index;
    uint32_t stage_index;

    if (bridge == 0 ||
        nodes_out == 0 ||
        node_count_out == 0 ||
        info_out == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_IDLE)
    {
        SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
    }
    *node_count_out = 0u;
    memset(info_out, 0, sizeof(*info_out));
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_FAILED)
    {
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    status = SparkDraftBridgeAdvance(bridge);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
    if (bridge->stale != 0u)
    {
        SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
    }
    info_out->server_status = bridge->server_status;
    reader = bridge->response_buffer +
        SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES +
        (uint64_t)bridge->node_count * SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES;
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
    if (bridge->server_status != SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK)
    {
        SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
    }
    for (node_index = 0u; node_index < bridge->node_count; ++node_index)
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
    *node_count_out = bridge->node_count;
    return SPARK_STATUS_OK;
}

void SparkDraftBridgeCancel(
    SparkDraftBridge *bridge)
{
    if (bridge == 0 || bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_IDLE)
    {
        return;
    }
    if (bridge->phase != SPARK_DRAFT_BRIDGE_PHASE_SEND ||
        bridge->request_sent != 0u)
    {
        SparkDraftBridgeDropConnection(bridge);
    }
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
}

uint32_t SparkDraftBridgePoll(
    const SparkDraftBridge *bridge,
    int *fd_out,
    uint32_t *events_out,
    uint64_t *deadline_ns_out)
{
    if (fd_out != 0)
    {
        *fd_out = SPARK_DRAFT_BRIDGE_DISCONNECTED_FD;
    }
    if (events_out != 0)
    {
        *events_out = 0u;
    }
    if (deadline_ns_out != 0)
    {
        *deadline_ns_out = 0u;
    }
    if (bridge == 0 || bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_IDLE)
    {
        return 0u;
    }
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_FAILED)
    {
        if (deadline_ns_out != 0)
        {
            *deadline_ns_out = 1u;
        }
        return 1u;
    }
    if (fd_out != 0)
    {
        *fd_out = bridge->socket_fd;
    }
    if (events_out != 0)
    {
        *events_out =
            bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_CONNECT ||
            bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND ?
            SPARK_DRAFT_BRIDGE_POLL_WRITE : SPARK_DRAFT_BRIDGE_POLL_READ;
    }
    if (deadline_ns_out != 0)
    {
        *deadline_ns_out = bridge->deadline_ns;
    }
    return 1u;
}
