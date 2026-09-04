#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "sparkpipe/spark_draft_bridge.h"

#define SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES 16u
#define SPARK_TEST_DRAFT_BRIDGE_MAX_COMMITTED 8u
#define SPARK_TEST_DRAFT_BRIDGE_MAX_TAP_ROWS 4u
#define SPARK_TEST_DRAFT_BRIDGE_MAX_NODES 16u
#define SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY 8u
#define SPARK_TEST_DRAFT_BRIDGE_OVER_CAPACITY_COUNT 9u
#define SPARK_TEST_DRAFT_BRIDGE_TIMEOUT_MS 2000u
#define SPARK_TEST_DRAFT_BRIDGE_SILENCE_MS 500u
#define SPARK_TEST_DRAFT_BRIDGE_LISTEN_BACKLOG 4
#define SPARK_TEST_DRAFT_BRIDGE_CAPTURE_BYTES 512u
#define SPARK_TEST_DRAFT_BRIDGE_RESPONSE_BYTES 512u
#define SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT 3u
#define SPARK_TEST_DRAFT_BRIDGE_TAP_COUNT 1u
#define SPARK_TEST_DRAFT_BRIDGE_SEQUENCE_ID 0x1122334455667788ull
#define SPARK_TEST_DRAFT_BRIDGE_GENERATION 7ull
#define SPARK_TEST_DRAFT_BRIDGE_POSITION 2ull
#define SPARK_TEST_DRAFT_BRIDGE_ANCHOR_TOKEN 33u
#define SPARK_TEST_DRAFT_BRIDGE_DEPTH 4u
#define SPARK_TEST_DRAFT_BRIDGE_TIME_BUDGET_MS 250u
#define SPARK_TEST_DRAFT_BRIDGE_MAX_DEPTH 8u
#define SPARK_TEST_DRAFT_BRIDGE_TARGET_MODEL "glm5.3-test-drafter"
#define SPARK_TEST_DRAFT_BRIDGE_HEADER_OFF_COMMITTED_COUNT 72u
#define SPARK_TEST_DRAFT_BRIDGE_HEADER_OFF_TAP_COUNT 76u
#define SPARK_TEST_DRAFT_BRIDGE_EXPECTED_REQUEST_BYTES \
    (SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES + \
        SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT * sizeof(uint32_t) + \
        SPARK_TEST_DRAFT_BRIDGE_TAP_COUNT * \
            SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES)

#define SPARK_TEST_DRAFT_BRIDGE_MODE_GOOD 0u
#define SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_SEQUENCE 1u
#define SPARK_TEST_DRAFT_BRIDGE_MODE_STALE_GENERATION 2u
#define SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_MAGIC 3u
#define SPARK_TEST_DRAFT_BRIDGE_MODE_OVER_CAPACITY 4u

#define SPARK_TEST_DRAFT_BRIDGE_CHECK(condition) \
    do \
    { \
        if (!(condition)) \
        { \
            fprintf( \
                stderr, \
                "CHECK-FAIL %s:%u: %s\n", \
                __FILE__, \
                (unsigned)__LINE__, \
                #condition); \
            return 1u; \
        } \
    } while (0)

typedef struct TestDraftBridgeStub
{
    const uint32_t *modes;
    uint32_t exchange_count;
    uint32_t expect_silence;
    int listen_fd;
    uint32_t port;
    pthread_t thread;
    uint32_t accept_count;
    uint32_t exchanges_completed;
    uint64_t bytes_received;
    uint32_t failed;
    uint8_t last_request[SPARK_TEST_DRAFT_BRIDGE_CAPTURE_BYTES];
    uint32_t last_request_bytes;
} TestDraftBridgeStub;

static void TestDraftBridgePutU32Le(
    uint8_t *destination,
    uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFFu);
    destination[1] = (uint8_t)((value >> 8u) & 0xFFu);
    destination[2] = (uint8_t)((value >> 16u) & 0xFFu);
    destination[3] = (uint8_t)((value >> 24u) & 0xFFu);
}

static void TestDraftBridgePutU64Le(
    uint8_t *destination,
    uint64_t value)
{
    uint32_t shift;

    for (shift = 0u; shift < 64u; shift += 8u)
    {
        destination[shift / 8u] = (uint8_t)((value >> shift) & 0xFFu);
    }
}

static void TestDraftBridgePutF32Le(
    uint8_t *destination,
    float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    TestDraftBridgePutU32Le(destination, bits);
}

static uint32_t TestDraftBridgeGetU32Le(
    const uint8_t *source)
{
    return (uint32_t)source[0] |
        ((uint32_t)source[1] << 8u) |
        ((uint32_t)source[2] << 16u) |
        ((uint32_t)source[3] << 24u);
}

static uint32_t TestDraftBridgeFloatBits(
    float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int TestDraftBridgeStubReadExact(
    int connection,
    uint8_t *data,
    uint32_t data_bytes)
{
    uint32_t offset;
    ssize_t received;

    offset = 0u;
    while (offset < data_bytes)
    {
        received = recv(connection, data + offset, data_bytes - offset, 0);
        if (received < 0 && errno == EINTR)
        {
            continue;
        }
        if (received < 0)
        {
            return -1;
        }
        if (received == 0)
        {
            return offset == 0u ? 0 : -1;
        }
        offset += (uint32_t)received;
    }
    return 1;
}

static int TestDraftBridgeStubWriteAll(
    int connection,
    const uint8_t *data,
    uint32_t data_bytes)
{
    uint32_t offset;
    ssize_t written;

    offset = 0u;
    while (offset < data_bytes)
    {
        written = send(connection, data + offset, data_bytes - offset, 0);
        if (written < 0 && errno == EINTR)
        {
            continue;
        }
        if (written <= 0)
        {
            return -1;
        }
        offset += (uint32_t)written;
    }
    return 0;
}

static uint32_t TestDraftBridgeStubBuildResponse(
    uint32_t mode,
    uint64_t sequence_id,
    uint64_t generation,
    uint8_t *response)
{
    uint8_t *cursor;
    uint32_t stage_index;

    cursor = response;
    if (mode == SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_MAGIC)
    {
        memcpy(cursor, "XXXX", 4u);
        cursor += 4u;
        memset(cursor, 0, 24u);
        cursor += 24u;
        return (uint32_t)(cursor - response);
    }
    memcpy(cursor, "DFT3", 4u);
    cursor += 4u;
    TestDraftBridgePutU32Le(
        cursor,
        mode == SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_SEQUENCE ?
            SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_SEQUENCE :
            SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK);
    cursor += sizeof(uint32_t);
    if (mode == SPARK_TEST_DRAFT_BRIDGE_MODE_OVER_CAPACITY)
    {
        TestDraftBridgePutU32Le(
            cursor,
            SPARK_TEST_DRAFT_BRIDGE_OVER_CAPACITY_COUNT);
        cursor += sizeof(uint32_t);
        TestDraftBridgePutU64Le(cursor, sequence_id);
        cursor += sizeof(uint64_t);
        TestDraftBridgePutU64Le(cursor, generation);
        cursor += sizeof(uint64_t);
        return (uint32_t)(cursor - response);
    }
    if (mode == SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_SEQUENCE)
    {
        TestDraftBridgePutU32Le(cursor, 0u);
        cursor += sizeof(uint32_t);
        TestDraftBridgePutU64Le(cursor, sequence_id);
        cursor += sizeof(uint64_t);
        TestDraftBridgePutU64Le(cursor, generation);
        cursor += sizeof(uint64_t);
        memset(cursor, 0, SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES);
        cursor += SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES;
        return (uint32_t)(cursor - response);
    }
    TestDraftBridgePutU32Le(cursor, 3u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU64Le(cursor, sequence_id);
    cursor += sizeof(uint64_t);
    TestDraftBridgePutU64Le(
        cursor,
        mode == SPARK_TEST_DRAFT_BRIDGE_MODE_STALE_GENERATION ?
            generation + 1u :
            generation);
    cursor += sizeof(uint64_t);
    TestDraftBridgePutU32Le(cursor, 100u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 0u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 0u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutF32Le(cursor, 1.0f);
    cursor += sizeof(float);
    TestDraftBridgePutU32Le(cursor, 101u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 0u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 1u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 1u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutF32Le(cursor, 0.625f);
    cursor += sizeof(float);
    TestDraftBridgePutU32Le(cursor, 102u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 1u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 2u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 2u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutF32Le(cursor, -1.25f);
    cursor += sizeof(float);
    for (stage_index = 0u;
         stage_index < SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT;
         ++stage_index)
    {
        TestDraftBridgePutU32Le(cursor, 10u + stage_index * 10u);
        cursor += sizeof(uint32_t);
    }
    TestDraftBridgePutU32Le(cursor, 90u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 2u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 123u);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 0u);
    cursor += sizeof(uint32_t);
    return (uint32_t)(cursor - response);
}

static void *TestDraftBridgeStubMain(
    void *argument)
{
    TestDraftBridgeStub *stub;
    struct timeval timeout;
    struct pollfd poll_descriptor;
    uint8_t header[SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES];
    uint8_t response[SPARK_TEST_DRAFT_BRIDGE_RESPONSE_BYTES];
    uint8_t discard[256];
    uint64_t body_bytes;
    uint64_t total_bytes;
    uint32_t exchange_index;
    uint32_t n_committed;
    uint32_t n_taps;
    uint32_t response_bytes;
    ssize_t received;
    int connection;
    int read_result;

    stub = (TestDraftBridgeStub *)argument;
    if (stub->expect_silence != 0u)
    {
        memset(&poll_descriptor, 0, sizeof(poll_descriptor));
        poll_descriptor.fd = stub->listen_fd;
        poll_descriptor.events = POLLIN;
        if (poll(
                &poll_descriptor,
                1u,
                (int)SPARK_TEST_DRAFT_BRIDGE_TIMEOUT_MS) <= 0)
        {
            stub->failed = 1u;
            return 0;
        }
        connection = accept(stub->listen_fd, 0, 0);
        if (connection < 0)
        {
            stub->failed = 1u;
            return 0;
        }
        stub->accept_count += 1u;
        timeout.tv_sec = 0;
        timeout.tv_usec =
            (suseconds_t)(SPARK_TEST_DRAFT_BRIDGE_SILENCE_MS * 1000u);
        (void)setsockopt(
            connection,
            SOL_SOCKET,
            SO_RCVTIMEO,
            &timeout,
            (socklen_t)sizeof(timeout));
        received = recv(connection, discard, sizeof(discard), 0);
        if (received > 0)
        {
            stub->bytes_received += (uint64_t)received;
            stub->failed = 1u;
        }
        (void)close(connection);
        return 0;
    }
    connection = -1;
    for (exchange_index = 0u;
         exchange_index < stub->exchange_count;
         ++exchange_index)
    {
        for (;;)
        {
            if (connection < 0)
            {
                do
                {
                    connection = accept(stub->listen_fd, 0, 0);
                } while (connection < 0 && errno == EINTR);
                if (connection < 0)
                {
                    stub->failed = 1u;
                    return 0;
                }
                stub->accept_count += 1u;
            }
            read_result = TestDraftBridgeStubReadExact(
                connection,
                header,
                SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES);
            if (read_result == 0)
            {
                (void)close(connection);
                connection = -1;
                continue;
            }
            if (read_result < 0)
            {
                stub->failed = 1u;
                (void)close(connection);
                return 0;
            }
            break;
        }
        n_committed = TestDraftBridgeGetU32Le(
            header + SPARK_TEST_DRAFT_BRIDGE_HEADER_OFF_COMMITTED_COUNT);
        n_taps = TestDraftBridgeGetU32Le(
            header + SPARK_TEST_DRAFT_BRIDGE_HEADER_OFF_TAP_COUNT);
        body_bytes =
            (uint64_t)n_committed * sizeof(uint32_t) +
            (uint64_t)n_taps * SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES;
        total_bytes = SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES + body_bytes;
        if (total_bytes > sizeof(stub->last_request))
        {
            stub->failed = 1u;
            (void)close(connection);
            return 0;
        }
        memcpy(
            stub->last_request,
            header,
            SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES);
        if (body_bytes != 0u)
        {
            read_result = TestDraftBridgeStubReadExact(
                connection,
                stub->last_request + SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES,
                (uint32_t)body_bytes);
            if (read_result != 1)
            {
                stub->failed = 1u;
                (void)close(connection);
                return 0;
            }
        }
        stub->bytes_received += total_bytes;
        stub->last_request_bytes = (uint32_t)total_bytes;
        response_bytes = TestDraftBridgeStubBuildResponse(
            stub->modes[exchange_index],
            SPARK_TEST_DRAFT_BRIDGE_SEQUENCE_ID,
            SPARK_TEST_DRAFT_BRIDGE_GENERATION,
            response);
        if (TestDraftBridgeStubWriteAll(
                connection,
                response,
                response_bytes) != 0)
        {
            (void)close(connection);
            connection = -1;
        }
        stub->exchanges_completed += 1u;
    }
    if (connection >= 0)
    {
        (void)close(connection);
    }
    return 0;
}

static uint32_t TestDraftBridgeStubStart(
    TestDraftBridgeStub *stub,
    const uint32_t *modes,
    uint32_t exchange_count,
    uint32_t expect_silence)
{
    struct sockaddr_in address;
    socklen_t address_bytes;
    int reuse;

    memset(stub, 0, sizeof(*stub));
    stub->modes = modes;
    stub->exchange_count = exchange_count;
    stub->expect_silence = expect_silence;
    stub->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (stub->listen_fd < 0)
    {
        return 1u;
    }
    reuse = 1;
    (void)setsockopt(
        stub->listen_fd,
        SOL_SOCKET,
        SO_REUSEADDR,
        &reuse,
        (socklen_t)sizeof(reuse));
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(0u);
    if (bind(
            stub->listen_fd,
            (const struct sockaddr *)&address,
            (socklen_t)sizeof(address)) != 0)
    {
        return 1u;
    }
    if (listen(stub->listen_fd, SPARK_TEST_DRAFT_BRIDGE_LISTEN_BACKLOG) != 0)
    {
        return 1u;
    }
    address_bytes = (socklen_t)sizeof(address);
    if (getsockname(
            stub->listen_fd,
            (struct sockaddr *)&address,
            &address_bytes) != 0)
    {
        return 1u;
    }
    stub->port = (uint32_t)ntohs(address.sin_port);
    if (pthread_create(&stub->thread, 0, TestDraftBridgeStubMain, stub) != 0)
    {
        return 1u;
    }
    return 0u;
}

static void TestDraftBridgeStubStop(
    TestDraftBridgeStub *stub)
{
    (void)pthread_join(stub->thread, 0);
    (void)close(stub->listen_fd);
}

static void TestDraftBridgeInitializeConfig(
    SparkDraftBridgeConfig *config,
    uint32_t port)
{
    memset(config, 0, sizeof(*config));
    config->abi_version = SPARK_DRAFT_BRIDGE_ABI_VERSION;
    config->descriptor_bytes = SPARK_DRAFT_BRIDGE_CONFIG_BYTES;
    config->host = "127.0.0.1";
    config->port = port;
    config->speculator_mask = 0x5u;
    memcpy(
        config->target_model,
        SPARK_TEST_DRAFT_BRIDGE_TARGET_MODEL,
        sizeof(SPARK_TEST_DRAFT_BRIDGE_TARGET_MODEL));
    config->tap_row_bytes = SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES;
    config->max_committed_tokens = SPARK_TEST_DRAFT_BRIDGE_MAX_COMMITTED;
    config->max_tap_rows = SPARK_TEST_DRAFT_BRIDGE_MAX_TAP_ROWS;
    config->max_nodes = SPARK_TEST_DRAFT_BRIDGE_MAX_NODES;
    config->connect_timeout_ms = SPARK_TEST_DRAFT_BRIDGE_TIMEOUT_MS;
    config->io_timeout_ms = SPARK_TEST_DRAFT_BRIDGE_TIMEOUT_MS;
}

static SparkStatus TestDraftBridgeProposeStandard(
    SparkDraftBridge *bridge,
    SparkDraftBridgeNode *nodes,
    uint32_t node_capacity,
    uint32_t *node_count_out,
    SparkDraftBridgeProposalInfo *info)
{
    static const uint32_t committed[SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT] =
        { 11u, 22u, 33u };
    uint8_t tap_rows[SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES];
    uint32_t byte_index;

    for (byte_index = 0u; byte_index < sizeof(tap_rows); ++byte_index)
    {
        tap_rows[byte_index] = (uint8_t)(0xA0u + byte_index);
    }
    return SparkDraftBridgeProposeTree(
        bridge,
        SPARK_TEST_DRAFT_BRIDGE_SEQUENCE_ID,
        SPARK_TEST_DRAFT_BRIDGE_GENERATION,
        SPARK_TEST_DRAFT_BRIDGE_POSITION,
        SPARK_TEST_DRAFT_BRIDGE_ANCHOR_TOKEN,
        committed,
        SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT,
        tap_rows,
        SPARK_TEST_DRAFT_BRIDGE_TAP_COUNT,
        SPARK_TEST_DRAFT_BRIDGE_DEPTH,
        SPARK_TEST_DRAFT_BRIDGE_TIME_BUDGET_MS,
        SPARK_TEST_DRAFT_BRIDGE_MAX_DEPTH,
        nodes,
        node_capacity,
        node_count_out,
        info);
}

static uint32_t TestDraftBridgeCheckGoodTree(
    const SparkDraftBridgeNode *nodes,
    uint32_t node_count,
    const SparkDraftBridgeProposalInfo *info)
{
    uint32_t stage_index;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(node_count == 3u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[0].token_id == 100u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        nodes[0].parent_index == SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[0].depth == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[0].source_bit == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeFloatBits(nodes[0].score) ==
        TestDraftBridgeFloatBits(1.0f));
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[1].token_id == 101u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[1].parent_index == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[1].depth == 1u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[1].source_bit == 1u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeFloatBits(nodes[1].score) ==
        TestDraftBridgeFloatBits(0.625f));
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[2].token_id == 102u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[2].parent_index == 1u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[2].depth == 2u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(nodes[2].source_bit == 2u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeFloatBits(nodes[2].score) ==
        TestDraftBridgeFloatBits(-1.25f));
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        info->server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK);
    for (stage_index = 0u;
         stage_index < SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT;
         ++stage_index)
    {
        SPARK_TEST_DRAFT_BRIDGE_CHECK(
            info->drafter_us[stage_index] == 10u + stage_index * 10u);
    }
    SPARK_TEST_DRAFT_BRIDGE_CHECK(info->graft_us == 90u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(info->expansions == 2u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(info->elapsed_us == 123u);
    return 0u;
}

static uint32_t TestDraftBridgeHappyPath(void)
{
    static const uint32_t modes[] = { SPARK_TEST_DRAFT_BRIDGE_MODE_GOOD };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    SparkStatus status;
    uint32_t node_count;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, modes, 1u, 0u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    status = SparkDraftBridgeInitialize(&config, &bridge);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(status == SPARK_STATUS_OK);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(bridge != 0);
    node_count = 0u;
    status = TestDraftBridgeProposeStandard(
        bridge,
        nodes,
        SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
        &node_count,
        &info);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(status == SPARK_STATUS_OK);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeCheckGoodTree(nodes, node_count, &info) == 0u);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.exchanges_completed == 1u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.accept_count == 1u);
    return 0u;
}

static uint32_t TestDraftBridgeWireFormat(void)
{
    static const uint32_t modes[] = { SPARK_TEST_DRAFT_BRIDGE_MODE_GOOD };
    static const uint32_t committed[SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT] =
        { 11u, 22u, 33u };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    uint8_t expected[SPARK_TEST_DRAFT_BRIDGE_EXPECTED_REQUEST_BYTES];
    uint8_t *cursor;
    uint32_t node_count;
    uint32_t byte_index;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, modes, 1u, 0u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_OK);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        stub.last_request_bytes ==
        SPARK_TEST_DRAFT_BRIDGE_EXPECTED_REQUEST_BYTES);

    cursor = expected;
    memcpy(cursor, "DFT3", 4u);
    cursor += 4u;
    TestDraftBridgePutU32Le(cursor, SPARK_DRAFT_BRIDGE_PROTOCOL_VERSION);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, 0x5u);
    cursor += sizeof(uint32_t);
    memset(cursor, 0, SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES);
    memcpy(
        cursor,
        SPARK_TEST_DRAFT_BRIDGE_TARGET_MODEL,
        sizeof(SPARK_TEST_DRAFT_BRIDGE_TARGET_MODEL) - 1u);
    cursor += SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES;
    TestDraftBridgePutU64Le(cursor, SPARK_TEST_DRAFT_BRIDGE_SEQUENCE_ID);
    cursor += sizeof(uint64_t);
    TestDraftBridgePutU64Le(cursor, SPARK_TEST_DRAFT_BRIDGE_GENERATION);
    cursor += sizeof(uint64_t);
    TestDraftBridgePutU64Le(cursor, SPARK_TEST_DRAFT_BRIDGE_POSITION);
    cursor += sizeof(uint64_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_ANCHOR_TOKEN);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_TAP_COUNT);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_DEPTH);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_TIME_BUDGET_MS);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_MAX_DEPTH);
    cursor += sizeof(uint32_t);
    TestDraftBridgePutU32Le(cursor, SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY);
    cursor += sizeof(uint32_t);
    for (byte_index = 0u;
         byte_index < SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT;
         ++byte_index)
    {
        TestDraftBridgePutU32Le(cursor, committed[byte_index]);
        cursor += sizeof(uint32_t);
    }
    for (byte_index = 0u;
         byte_index < SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES;
         ++byte_index)
    {
        *cursor = (uint8_t)(0xA0u + byte_index);
        cursor += 1u;
    }
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        (uint32_t)(cursor - expected) ==
        SPARK_TEST_DRAFT_BRIDGE_EXPECTED_REQUEST_BYTES);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        memcmp(
            stub.last_request,
            expected,
            SPARK_TEST_DRAFT_BRIDGE_EXPECTED_REQUEST_BYTES) == 0);
    return 0u;
}

static uint32_t TestDraftBridgeServerRejectionKeepsConnection(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_SEQUENCE,
          SPARK_TEST_DRAFT_BRIDGE_MODE_GOOD };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    uint32_t node_count;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, modes, 2u, 0u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_VALIDATION_FAILED);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        info.server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_SEQUENCE);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(node_count == 0u);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_OK);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeCheckGoodTree(nodes, node_count, &info) == 0u);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.exchanges_completed == 2u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.accept_count == 1u);
    return 0u;
}

static uint32_t TestDraftBridgeStaleGenerationRejected(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_DRAFT_BRIDGE_MODE_STALE_GENERATION };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    uint32_t node_count;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, modes, 1u, 0u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_VALIDATION_FAILED);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.exchanges_completed == 1u);
    return 0u;
}

static uint32_t TestDraftBridgeBadMagicReconnects(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_DRAFT_BRIDGE_MODE_BAD_MAGIC,
          SPARK_TEST_DRAFT_BRIDGE_MODE_GOOD };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    uint32_t node_count;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, modes, 2u, 0u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_IO_ERROR);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_OK);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeCheckGoodTree(nodes, node_count, &info) == 0u);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.exchanges_completed == 2u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.accept_count == 2u);
    return 0u;
}

static uint32_t TestDraftBridgeClientValidationSkipsSocket(void)
{
    static const uint32_t committed[SPARK_TEST_DRAFT_BRIDGE_COMMITTED_COUNT] =
        { 11u, 22u, 33u };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    uint8_t tap_rows[SPARK_TEST_DRAFT_BRIDGE_TAP_ROW_BYTES];
    uint32_t node_count;

    memset(tap_rows, 0, sizeof(tap_rows));
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, 0, 0u, 1u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeProposeTree(
            bridge, 1u, 1u, 2u, 34u, committed, 3u,
            tap_rows, 1u, 4u, 250u, 8u,
            nodes, SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count, &info) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeProposeTree(
            bridge, 1u, 1u, 3u, 33u, committed, 3u,
            tap_rows, 1u, 4u, 250u, 8u,
            nodes, SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count, &info) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeProposeTree(
            bridge, 1u, 1u, 2u, 33u, committed, 3u,
            tap_rows, 1u, 0u, 250u, 8u,
            nodes, SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count, &info) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeProposeTree(
            bridge, 1u, 1u, 2u, 33u, committed, 3u,
            tap_rows, 1u, 4u, 250u, 8u,
            nodes, 0u,
            &node_count, &info) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeProposeTree(
            bridge, 1u, 1u, 2u, 33u, committed, 3u,
            tap_rows, SPARK_TEST_DRAFT_BRIDGE_MAX_TAP_ROWS + 1u, 4u, 250u, 8u,
            nodes, SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count, &info) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.bytes_received == 0u);
    return 0u;
}

static uint32_t TestDraftBridgeOverCapacityDrops(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_DRAFT_BRIDGE_MODE_OVER_CAPACITY,
          SPARK_TEST_DRAFT_BRIDGE_MODE_GOOD };
    TestDraftBridgeStub stub;
    SparkDraftBridgeConfig config;
    SparkDraftBridgeProposalInfo info;
    SparkDraftBridge *bridge;
    SparkDraftBridgeNode nodes[SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY];
    uint32_t node_count;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeStubStart(&stub, modes, 2u, 0u) == 0u);
    TestDraftBridgeInitializeConfig(&config, stub.port);
    bridge = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_CAPACITY_EXCEEDED);
    node_count = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        TestDraftBridgeProposeStandard(
            bridge,
            nodes,
            SPARK_TEST_DRAFT_BRIDGE_NODE_CAPACITY,
            &node_count,
            &info) == SPARK_STATUS_OK);
    SparkDraftBridgeDestroy(bridge);
    TestDraftBridgeStubStop(&stub);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.failed == 0u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.exchanges_completed == 2u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(stub.accept_count == 2u);
    return 0u;
}

static uint32_t TestDraftBridgeConfigValidation(void)
{
    SparkDraftBridgeConfig config;
    SparkDraftBridge *bridge;

    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(0) == SPARK_STATUS_INVALID_ARGUMENT);
    TestDraftBridgeInitializeConfig(&config, 1u);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(&config) == SPARK_STATUS_OK);

    TestDraftBridgeInitializeConfig(&config, 1u);
    config.host = 0;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(&config) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    bridge = (SparkDraftBridge *)0x1;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(bridge == 0);

    TestDraftBridgeInitializeConfig(&config, 1u);
    config.port = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(&config) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    TestDraftBridgeInitializeConfig(&config, 1u);
    config.tap_row_bytes = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(&config) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    TestDraftBridgeInitializeConfig(&config, 1u);
    memset(
        config.target_model,
        'x',
        SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(&config) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeInitialize(&config, &bridge) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    TestDraftBridgeInitializeConfig(&config, 1u);
    config.abi_version = 0u;
    SPARK_TEST_DRAFT_BRIDGE_CHECK(
        SparkDraftBridgeValidateConfig(&config) == SPARK_STATUS_ABI_MISMATCH);
    return 0u;
}

static uint32_t TestDraftBridgeRunCase(
    const char *name,
    uint32_t (*case_fn)(void))
{
    uint32_t result;

    result = case_fn();
    if (result != 0u)
    {
        printf("FAIL %s\n", name);
        return 1u;
    }
    printf("PASS %s\n", name);
    return 0u;
}

int main(void)
{
    uint32_t failures;

    (void)signal(SIGPIPE, SIG_IGN);
    failures = 0u;
    failures += TestDraftBridgeRunCase(
        "happy_path",
        TestDraftBridgeHappyPath);
    failures += TestDraftBridgeRunCase(
        "wire_format",
        TestDraftBridgeWireFormat);
    failures += TestDraftBridgeRunCase(
        "server_rejection_keeps_connection",
        TestDraftBridgeServerRejectionKeepsConnection);
    failures += TestDraftBridgeRunCase(
        "stale_generation_rejected",
        TestDraftBridgeStaleGenerationRejected);
    failures += TestDraftBridgeRunCase(
        "bad_magic_reconnects",
        TestDraftBridgeBadMagicReconnects);
    failures += TestDraftBridgeRunCase(
        "client_validation_skips_socket",
        TestDraftBridgeClientValidationSkipsSocket);
    failures += TestDraftBridgeRunCase(
        "over_capacity_drops",
        TestDraftBridgeOverCapacityDrops);
    failures += TestDraftBridgeRunCase(
        "config_validation",
        TestDraftBridgeConfigValidation);
    if (failures != 0u)
    {
        fprintf(stderr, "FAILURES %u\n", (unsigned)failures);
        return 1;
    }
    return 0;
}
