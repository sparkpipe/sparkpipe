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
#include <sys/uio.h>
#include <unistd.h>

#define SPARK_DRAFT_BRIDGE_HOST_BYTES 256u
#define SPARK_DRAFT_BRIDGE_PORT_TEXT_BYTES 8u
#define SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC "DFT3"
#define SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES 4u
#define SPARK_DRAFT_BRIDGE_NS_PER_MS UINT64_C(1000000)
#define SPARK_DRAFT_BRIDGE_DISCONNECTED_FD (-1)
#define SPARK_DRAFT_BRIDGE_RX_BUFFER_BYTES 65536u
#define SPARK_DRAFT_BRIDGE_SOCKBUF_BYTES (2 * 1024 * 1024)

#define SPARK_DRAFT_BRIDGE_PHASE_IDLE 0u
#define SPARK_DRAFT_BRIDGE_PHASE_CONNECT 1u
#define SPARK_DRAFT_BRIDGE_PHASE_SEND 2u
#define SPARK_DRAFT_BRIDGE_PHASE_HEADER 3u
#define SPARK_DRAFT_BRIDGE_PHASE_BODY 4u
#define SPARK_DRAFT_BRIDGE_PHASE_FAILED 5u


typedef struct SparkDraftBridgeInflight
{
    uint64_t sequence_id;
    uint64_t generation;
    uint32_t node_capacity;
} SparkDraftBridgeInflight;

struct SparkDraftBridge
{
    char host[SPARK_DRAFT_BRIDGE_HOST_BYTES];
    char target_model[SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES];
    uint8_t *request_buffer;
    uint8_t *response_buffer;
    uint8_t *rx_buffer;
    uint64_t request_buffer_bytes;
    uint64_t response_buffer_bytes;
    uint64_t rx_head;
    uint64_t rx_tail;
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
    SparkDraftBridgeInflight inflight[SPARK_DRAFT_BRIDGE_MAX_PIPELINE];
    uint32_t inflight_head;
    uint32_t inflight_count;
    uint32_t sent_any;
    uint32_t node_count;
    uint32_t server_status;
    uint32_t stale;
    uint32_t phase;
    uint32_t batch_cork;
};

static void SparkDraftBridgePutU32Le(
    uint8_t *destination,
    uint32_t value)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    memcpy(destination, &value, sizeof(value));
#else
    destination[0] = (uint8_t)(value & 0xFFu);
    destination[1] = (uint8_t)((value >> 8u) & 0xFFu);
    destination[2] = (uint8_t)((value >> 16u) & 0xFFu);
    destination[3] = (uint8_t)((value >> 24u) & 0xFFu);
#endif
}

static void SparkDraftBridgePutU64Le(
    uint8_t *destination,
    uint64_t value)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    memcpy(destination, &value, sizeof(value));
#else
    uint32_t shift;

    for (shift = 0u; shift < 64u; shift += 8u)
    {
        destination[shift / 8u] = (uint8_t)((value >> shift) & 0xFFu);
    }
#endif
}

static uint32_t SparkDraftBridgeGetU32Le(
    const uint8_t *source)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint32_t value;
    memcpy(&value, source, sizeof(value));
    return value;
#else
    return (uint32_t)source[0] |
        ((uint32_t)source[1] << 8u) |
        ((uint32_t)source[2] << 16u) |
        ((uint32_t)source[3] << 24u);
#endif
}

static uint64_t SparkDraftBridgeGetU64Le(
    const uint8_t *source)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint64_t value;
    memcpy(&value, source, sizeof(value));
    return value;
#else
    uint64_t value;
    uint32_t shift;

    value = 0u;
    for (shift = 0u; shift < 64u; shift += 8u)
    {
        value |= (uint64_t)source[shift / 8u] << shift;
    }
    return value;
#endif
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
    bridge->rx_head = 0u;
    bridge->rx_tail = 0u;
    SparkDraftBridgeFreeAddresses(bridge);
}

static void SparkDraftBridgeResetPipeline(
    SparkDraftBridge *bridge)
{
    bridge->inflight_head = 0u;
    bridge->inflight_count = 0u;
    bridge->request_bytes = 0u;
    bridge->request_sent = 0u;
    bridge->response_bytes = 0u;
    bridge->response_expected = 0u;
    bridge->sent_any = 0u;
    bridge->batch_cork = 0u;
    bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
}

static SparkStatus SparkDraftBridgeFailTransport(
    SparkDraftBridge *bridge)
{
    SparkDraftBridgeDropConnection(bridge);
    SparkDraftBridgeResetPipeline(bridge);
    SPARK_FAIL(SPARK_STATUS_IO_ERROR);
}

static uint64_t SparkDraftBridgeDeadline(
    uint32_t timeout_ms)
{
    return SparkMonotonicNs() +
        (uint64_t)timeout_ms * SPARK_DRAFT_BRIDGE_NS_PER_MS;
}

static void SparkDraftBridgeQuickAck(
    int socket_fd)
{
#ifdef TCP_QUICKACK
    int enabled = 1;
    (void)setsockopt(
        socket_fd,
        IPPROTO_TCP,
        TCP_QUICKACK,
        &enabled,
        (socklen_t)sizeof(enabled));
#else
    (void)socket_fd;
#endif
}

static SparkStatus SparkDraftBridgeConfigureSocket(
    int socket_fd)
{
    int enabled;
    int buf_bytes;

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
    SparkDraftBridgeQuickAck(socket_fd);
    buf_bytes = SPARK_DRAFT_BRIDGE_SOCKBUF_BYTES;
    (void)setsockopt(
        socket_fd,
        SOL_SOCKET,
        SO_SNDBUF,
        &buf_bytes,
        (socklen_t)sizeof(buf_bytes));
    (void)setsockopt(
        socket_fd,
        SOL_SOCKET,
        SO_RCVBUF,
        &buf_bytes,
        (socklen_t)sizeof(buf_bytes));
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
        bridge->sent_any = 1u;
    }
    bridge->request_bytes = 0u;
    bridge->request_sent = 0u;
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND)
    {
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_HEADER;
        bridge->response_bytes = 0u;
        bridge->response_expected = SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeReceive(
    SparkDraftBridge *bridge)
{
    ssize_t got;
    uint64_t need;
    uint64_t avail;
    uint64_t take;
    uint32_t received_any = 0u;

    while (bridge->response_bytes < bridge->response_expected)
    {
        need = bridge->response_expected - bridge->response_bytes;
        avail = bridge->rx_tail - bridge->rx_head;
        if (avail != 0u)
        {
            take = avail < need ? avail : need;
            memcpy(
                bridge->response_buffer + bridge->response_bytes,
                bridge->rx_buffer + bridge->rx_head,
                (size_t)take);
            bridge->response_bytes += take;
            bridge->rx_head += take;
            if (bridge->rx_head == bridge->rx_tail)
            {
                bridge->rx_head = 0u;
                bridge->rx_tail = 0u;
            }
            continue;
        }
        bridge->rx_head = 0u;
        bridge->rx_tail = 0u;
        if (need >= SPARK_DRAFT_BRIDGE_RX_BUFFER_BYTES || bridge->rx_buffer == 0)
        {
            got = recv(
                bridge->socket_fd,
                bridge->response_buffer + bridge->response_bytes,
                (size_t)need,
                0);
        }
        else
        {
            got = recv(
                bridge->socket_fd,
                bridge->rx_buffer,
                SPARK_DRAFT_BRIDGE_RX_BUFFER_BYTES,
                0);
        }
        if (got < 0 && errno == EINTR)
        {
            continue;
        }
        if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            if (received_any != 0u)
            {
                SparkDraftBridgeQuickAck(bridge->socket_fd);
            }
            return SparkDraftBridgeBlocked(bridge);
        }
        if (got <= 0)
        {
            return SparkDraftBridgeFailTransport(bridge);
        }
        received_any = 1u;
        if (need >= SPARK_DRAFT_BRIDGE_RX_BUFFER_BYTES || bridge->rx_buffer == 0)
        {
            bridge->response_bytes += (uint64_t)got;
        }
        else
        {
            bridge->rx_tail = (uint64_t)got;
        }
    }
    if (received_any != 0u)
    {
        SparkDraftBridgeQuickAck(bridge->socket_fd);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDraftBridgeReadHeader(
    SparkDraftBridge *bridge)
{
    const SparkDraftBridgeInflight *head;
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
    head = &bridge->inflight[bridge->inflight_head];
    reader = bridge->response_buffer + SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES;
    bridge->server_status = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    bridge->node_count = SparkDraftBridgeGetU32Le(reader);
    reader += sizeof(uint32_t);
    response_sequence_id = SparkDraftBridgeGetU64Le(reader);
    reader += sizeof(uint64_t);
    response_generation = SparkDraftBridgeGetU64Le(reader);
    if (bridge->node_count > head->node_capacity)
    {
        SparkDraftBridgeDropConnection(bridge);
        SparkDraftBridgeResetPipeline(bridge);
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    bridge->stale =
        response_sequence_id != head->sequence_id ||
        response_generation != head->generation ? 1u : 0u;
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
            if (status == SPARK_STATUS_PENDING &&
                bridge->inflight_count > 1u &&
                bridge->sent_any != 0u)
            {
                bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_HEADER;
                bridge->response_bytes = 0u;
                bridge->response_expected =
                    SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES;
                status = SPARK_STATUS_OK;
            }
        }
        else
        {
            if (bridge->request_sent < bridge->request_bytes)
            {
                status = SparkDraftBridgeSend(bridge);
                if (status != SPARK_STATUS_OK && status != SPARK_STATUS_PENDING)
                {
                    return status;
                }
            }
            if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_HEADER)
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
    bridge->rx_buffer = (uint8_t *)malloc(SPARK_DRAFT_BRIDGE_RX_BUFFER_BYTES);
    if (bridge->request_buffer == 0 || bridge->response_buffer == 0 || bridge->rx_buffer == 0)
    {
        free(bridge->request_buffer);
        free(bridge->response_buffer);
        free(bridge->rx_buffer);
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
    free(bridge->rx_buffer);
    free(bridge);
}

static SparkStatus SparkDraftBridgeReserveRequest(
    SparkDraftBridge *bridge,
    uint64_t message_bytes)
{
    uint64_t pending;
    uint64_t needed;
    uint8_t *grown;

    if (bridge->request_sent == bridge->request_bytes)
    {
        bridge->request_sent = 0u;
        bridge->request_bytes = 0u;
    }
    pending = bridge->request_bytes - bridge->request_sent;
    if (bridge->request_sent != 0u &&
        message_bytes > bridge->request_buffer_bytes - bridge->request_bytes)
    {
        memmove(
            bridge->request_buffer,
            bridge->request_buffer + bridge->request_sent,
            (size_t)pending);
        bridge->request_bytes = pending;
        bridge->request_sent = 0u;
    }
    needed = bridge->request_bytes + message_bytes;
    if (needed > bridge->request_buffer_bytes)
    {
        grown = (uint8_t *)realloc(bridge->request_buffer, (size_t)needed);
        if (grown == 0)
        {
            SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
        }
        bridge->request_buffer = grown;
        bridge->request_buffer_bytes = needed;
    }
    return SPARK_STATUS_OK;
}

static void SparkDraftBridgeEncodeHeader(
    uint8_t *cursor,
    const SparkDraftBridge *bridge,
    uint32_t speculator_mask,
    uint64_t sequence_id,
    uint64_t generation,
    uint64_t position,
    uint32_t anchor_token_id,
    uint32_t committed_token_count,
    uint32_t tap_row_count,
    uint32_t depth,
    uint32_t time_budget_ms,
    uint32_t max_depth,
    const SparkRowSampling *sampling,
    uint32_t node_capacity)
{
    const uint64_t seed = sampling != 0 ? sampling->seed : 0u;
    const float inverse_temperature = sampling != 0 ? sampling->inverse_temperature : 0.0f;
    const float nucleus = sampling != 0 ? sampling->nucleus : 0.0f;
    uint32_t bits;

    memcpy(cursor, SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC, SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES);
    cursor += SPARK_DRAFT_BRIDGE_PROTOCOL_MAGIC_BYTES;
    SparkDraftBridgePutU32Le(cursor, SPARK_DRAFT_BRIDGE_PROTOCOL_VERSION);
    cursor += sizeof(uint32_t);
    SparkDraftBridgePutU32Le(cursor, speculator_mask);
    cursor += sizeof(uint32_t);
    memcpy(cursor, bridge->target_model, SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES);
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
}

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
static SparkStatus SparkDraftBridgeSendDirect(
    SparkDraftBridge *bridge,
    const uint8_t *header,
    const uint32_t *committed_token_ids,
    uint64_t token_bytes,
    const void *tap_rows,
    uint64_t tap_bytes,
    uint64_t message_bytes)
{
    struct iovec iov[3];
    struct msghdr msg;
    ssize_t put;
    uint64_t written = 0u;
    uint64_t remaining;
    uint8_t *dst;
    uint32_t seg;
#ifdef MSG_NOSIGNAL
    const int send_flags = MSG_NOSIGNAL;
#else
    const int send_flags = 0;
#endif
    SparkStatus status;

    iov[0].iov_base = (void *)header;
    iov[0].iov_len = SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES;
    iov[1].iov_base = (void *)committed_token_ids;
    iov[1].iov_len = (size_t)token_bytes;
    iov[2].iov_base = (void *)tap_rows;
    iov[2].iov_len = (size_t)tap_bytes;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iov;
    msg.msg_iovlen = tap_bytes != 0u ? 3 : 2;
    do
    {
        put = sendmsg(bridge->socket_fd, &msg, send_flags);
    } while (put < 0 && errno == EINTR);
    if (put == (ssize_t)message_bytes)
    {
        bridge->request_bytes = 0u;
        bridge->request_sent = 0u;
        bridge->sent_any = 1u;
        if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND)
        {
            bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_HEADER;
            bridge->response_bytes = 0u;
            bridge->response_expected = SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES;
        }
        return SPARK_STATUS_OK;
    }
    if (put < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    if (put == 0)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    if (put > 0)
    {
        written = (uint64_t)put;
        bridge->sent_any = 1u;
    }
    remaining = message_bytes - written;
    bridge->request_bytes = 0u;
    bridge->request_sent = 0u;
    status = SparkDraftBridgeReserveRequest(bridge, remaining);
    if (status != SPARK_STATUS_OK)
    {
        return SparkDraftBridgeFailTransport(bridge);
    }
    dst = bridge->request_buffer;
    for (seg = 0u; seg < (tap_bytes != 0u ? 3u : 2u); ++seg)
    {
        const uint64_t seg_len = (uint64_t)iov[seg].iov_len;
        if (written >= seg_len)
        {
            written -= seg_len;
            continue;
        }
        memcpy(dst, (const uint8_t *)iov[seg].iov_base + written, (size_t)(seg_len - written));
        dst += seg_len - written;
        written = 0u;
    }
    bridge->request_bytes = remaining;
    return SparkDraftBridgeBlocked(bridge);
}
#endif

static SparkStatus SparkDraftBridgeSubmitTree(
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
    uint32_t node_capacity,
    uint32_t max_inflight)
{
    SparkStatus status;
    uint8_t *cursor;
    uint64_t token_bytes;
    uint64_t tap_bytes;
    uint64_t message_bytes;
    uint32_t tail;

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
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_FAILED)
    {
        SPARK_FAIL(max_inflight == 1u ? SPARK_STATUS_BUSY : SPARK_STATUS_IO_ERROR);
    }
    if (bridge->inflight_count >= max_inflight)
    {
        SPARK_FAIL(SPARK_STATUS_BUSY);
    }

    token_bytes = (uint64_t)committed_token_count * sizeof(uint32_t);
    tap_bytes = (uint64_t)tap_row_count * bridge->tap_row_bytes;
    message_bytes =
        (uint64_t)SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES +
        token_bytes +
        tap_bytes;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    if (bridge->batch_cork == 0u &&
        bridge->request_sent == bridge->request_bytes &&
        bridge->socket_fd != SPARK_DRAFT_BRIDGE_DISCONNECTED_FD &&
        bridge->phase != SPARK_DRAFT_BRIDGE_PHASE_CONNECT)
    {
        uint8_t header[SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES];
        SparkDraftBridgeEncodeHeader(
            header, bridge, speculator_mask, sequence_id, generation, position,
            anchor_token_id, committed_token_count, tap_row_count, depth,
            time_budget_ms, max_depth, sampling, node_capacity);
        tail = (bridge->inflight_head + bridge->inflight_count) %
            SPARK_DRAFT_BRIDGE_MAX_PIPELINE;
        bridge->inflight[tail].sequence_id = sequence_id;
        bridge->inflight[tail].generation = generation;
        bridge->inflight[tail].node_capacity = node_capacity;
        bridge->inflight_count++;
        if (bridge->inflight_count == 1u)
        {
            bridge->response_bytes = 0u;
            bridge->response_expected = 0u;
            bridge->node_count = 0u;
            bridge->server_status = 0u;
            bridge->stale = 0u;
            bridge->sent_any = 0u;
            bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_SEND;
            bridge->deadline_ns = SparkDraftBridgeDeadline(bridge->io_timeout_ms);
        }
        status = SparkDraftBridgeSendDirect(
            bridge, header, committed_token_ids, token_bytes, tap_rows, tap_bytes, message_bytes);
        if (status != SPARK_STATUS_OK && status != SPARK_STATUS_PENDING)
        {
            bridge->inflight_count = 1u;
            bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_FAILED;
        }
        return SPARK_STATUS_OK;
    }
#endif
    status = SparkDraftBridgeReserveRequest(bridge, message_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    cursor = bridge->request_buffer + bridge->request_bytes;
    SparkDraftBridgeEncodeHeader(
        cursor, bridge, speculator_mask, sequence_id, generation, position,
        anchor_token_id, committed_token_count, tap_row_count, depth,
        time_budget_ms, max_depth, sampling, node_capacity);
    cursor += SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    memcpy(cursor, committed_token_ids, (size_t)token_bytes);
    cursor += token_bytes;
#else
    {
        uint32_t token_index;
        for (token_index = 0u; token_index < committed_token_count; ++token_index)
        {
            SparkDraftBridgePutU32Le(cursor, committed_token_ids[token_index]);
            cursor += sizeof(uint32_t);
        }
    }
#endif
    if (tap_bytes != 0u)
    {
        memcpy(cursor, tap_rows, (size_t)tap_bytes);
        cursor += tap_bytes;
    }
    bridge->request_bytes = (uint64_t)(cursor - bridge->request_buffer);
    tail = (bridge->inflight_head + bridge->inflight_count) %
        SPARK_DRAFT_BRIDGE_MAX_PIPELINE;
    bridge->inflight[tail].sequence_id = sequence_id;
    bridge->inflight[tail].generation = generation;
    bridge->inflight[tail].node_capacity = node_capacity;
    bridge->inflight_count++;
    if (bridge->inflight_count == 1u)
    {
        bridge->response_bytes = 0u;
        bridge->response_expected = 0u;
        bridge->node_count = 0u;
        bridge->server_status = 0u;
        bridge->stale = 0u;
        bridge->sent_any = 0u;
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
            bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND &&
            bridge->batch_cork == 0u)
        {
            status = SparkDraftBridgeSend(bridge);
        }
    }
    else if (bridge->phase != SPARK_DRAFT_BRIDGE_PHASE_CONNECT &&
             bridge->batch_cork == 0u)
    {
        status = SparkDraftBridgeSend(bridge);
    }
    else
    {
        status = SPARK_STATUS_OK;
    }
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_PENDING)
    {
        bridge->inflight_count = 1u;
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_FAILED;
    }
    return SPARK_STATUS_OK;
}

void SparkDraftBridgeBeginBatch(
    SparkDraftBridge *bridge)
{
    if (bridge != 0)
    {
        bridge->batch_cork = 1u;
    }
}

SparkStatus SparkDraftBridgeFlushBatch(
    SparkDraftBridge *bridge)
{
    SparkStatus status;

    if (bridge == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    bridge->batch_cork = 0u;
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_FAILED ||
        bridge->inflight_count == 0u ||
        bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_CONNECT ||
        bridge->request_sent >= bridge->request_bytes)
    {
        return SPARK_STATUS_OK;
    }
    status = SparkDraftBridgeSend(bridge);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_PENDING)
    {
        bridge->inflight_count = 1u;
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_FAILED;
    }
    return SPARK_STATUS_OK;
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
    return SparkDraftBridgeSubmitTree(
        bridge,
        speculator_mask,
        sequence_id,
        generation,
        position,
        anchor_token_id,
        committed_token_ids,
        committed_token_count,
        tap_rows,
        tap_row_count,
        depth,
        time_budget_ms,
        max_depth,
        sampling,
        node_capacity,
        1u);
}

SparkStatus SparkDraftBridgeEnqueueTree(
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
    return SparkDraftBridgeSubmitTree(
        bridge,
        speculator_mask,
        sequence_id,
        generation,
        position,
        anchor_token_id,
        committed_token_ids,
        committed_token_count,
        tap_rows,
        tap_row_count,
        depth,
        time_budget_ms,
        max_depth,
        sampling,
        node_capacity,
        SPARK_DRAFT_BRIDGE_MAX_PIPELINE);
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
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_IDLE ||
        bridge->inflight_count == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
    }
    *node_count_out = 0u;
    memset(info_out, 0, sizeof(*info_out));
    if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_FAILED)
    {
        SparkDraftBridgeResetPipeline(bridge);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    bridge->batch_cork = 0u;
    status = SparkDraftBridgeAdvance(bridge);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    bridge->inflight_head =
        (bridge->inflight_head + 1u) % SPARK_DRAFT_BRIDGE_MAX_PIPELINE;
    bridge->inflight_count--;
    if (bridge->inflight_count != 0u)
    {
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_HEADER;
        bridge->response_bytes = 0u;
        bridge->response_expected = SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES;
        bridge->deadline_ns = SparkDraftBridgeDeadline(bridge->io_timeout_ms);
    }
    else
    {
        bridge->phase = SPARK_DRAFT_BRIDGE_PHASE_IDLE;
        bridge->response_bytes = 0u;
        bridge->response_expected = 0u;
        bridge->sent_any = 0u;
    }
    if (bridge->stale != 0u)
    {
        if (bridge->inflight_count != 0u)
        {
            SparkDraftBridgeDropConnection(bridge);
            SparkDraftBridgeResetPipeline(bridge);
        }
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
        bridge->sent_any != 0u)
    {
        SparkDraftBridgeDropConnection(bridge);
    }
    SparkDraftBridgeResetPipeline(bridge);
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
    if (bridge == 0 ||
        bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_IDLE ||
        bridge->inflight_count == 0u)
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
        if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_CONNECT)
        {
            *events_out = SPARK_DRAFT_BRIDGE_POLL_WRITE;
        }
        else if (bridge->phase == SPARK_DRAFT_BRIDGE_PHASE_SEND)
        {
            *events_out = bridge->inflight_count > 1u && bridge->sent_any != 0u ?
                (SPARK_DRAFT_BRIDGE_POLL_READ | SPARK_DRAFT_BRIDGE_POLL_WRITE) :
                SPARK_DRAFT_BRIDGE_POLL_WRITE;
        }
        else
        {
            *events_out = SPARK_DRAFT_BRIDGE_POLL_READ |
                (bridge->request_sent < bridge->request_bytes ?
                    SPARK_DRAFT_BRIDGE_POLL_WRITE : 0u);
        }
    }
    if (deadline_ns_out != 0)
    {
        *deadline_ns_out = bridge->deadline_ns;
    }
    return 1u;
}
