#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_speculation_seam.h"

#define SPARK_TEST_SEAM_TAP_ROW_BYTES 16u
#define SPARK_TEST_SEAM_MAX_COMMITTED 8u
#define SPARK_TEST_SEAM_MAX_TAP_ROWS 4u
#define SPARK_TEST_SEAM_MAX_NODES 16u
#define SPARK_TEST_SEAM_DRAFT_CAPACITY 8u
#define SPARK_TEST_SEAM_HIDDEN_DIMENSION 8u
#define SPARK_TEST_SEAM_VOCAB_SIZE 1000u
#define SPARK_TEST_SEAM_MAX_SPECULATIVE 8u
#define SPARK_TEST_SEAM_VERIFIER_ACCEPT_K 4u
#define SPARK_TEST_SEAM_TIME_BUDGET_MS 250u
#define SPARK_TEST_SEAM_TIMEOUT_MS 2000u
#define SPARK_TEST_SEAM_LISTEN_BACKLOG 4
#define SPARK_TEST_SEAM_CAPTURE_BYTES 512u
#define SPARK_TEST_SEAM_RESPONSE_BYTES 512u
#define SPARK_TEST_SEAM_SEQUENCE_ID UINT64_C(0xA0A1A2A3A4A5A6A7)
#define SPARK_TEST_SEAM_TARGET_MODEL "spec-seam-test-drafter"
#define SPARK_TEST_SEAM_HEADER_OFF_SPECULATOR_MASK 8u
#define SPARK_TEST_SEAM_HEADER_OFF_SEQUENCE_ID 44u
#define SPARK_TEST_SEAM_HEADER_OFF_GENERATION 52u
#define SPARK_TEST_SEAM_HEADER_OFF_COMMITTED_COUNT 72u
#define SPARK_TEST_SEAM_HEADER_OFF_TAP_COUNT 76u

#define SPARK_TEST_SEAM_MODE_DEEP_TREE 0u
#define SPARK_TEST_SEAM_MODE_MALFORMED_TREE 1u

#define SPARK_TEST_SEAM_DEEP_NODE_COUNT 5u
#define SPARK_TEST_SEAM_MALFORMED_NODE_COUNT 3u

#define SPARK_TEST_SEAM_CHECK(condition) \
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

typedef struct TestSeamStubNode
{
    uint32_t token_id;
    uint32_t parent_index;
    uint32_t depth;
    uint32_t source_bit;
    float score;
} TestSeamStubNode;

typedef struct TestSeamStub
{
    const uint32_t *modes;
    uint32_t exchange_count;
    int listen_fd;
    uint32_t port;
    pthread_t thread;
    uint32_t accept_count;
    uint32_t exchanges_completed;
    uint64_t bytes_received;
    uint32_t failed;
    uint32_t last_speculator_mask;
    uint64_t last_sequence_id;
    uint64_t last_generation;
    uint8_t last_request[SPARK_TEST_SEAM_CAPTURE_BYTES];
    uint32_t last_request_bytes;
} TestSeamStub;

static void TestSeamPutU32Le(
    uint8_t *destination,
    uint32_t value)
{
    destination[0] = (uint8_t)(value & 0xFFu);
    destination[1] = (uint8_t)((value >> 8u) & 0xFFu);
    destination[2] = (uint8_t)((value >> 16u) & 0xFFu);
    destination[3] = (uint8_t)((value >> 24u) & 0xFFu);
}

static void TestSeamPutU64Le(
    uint8_t *destination,
    uint64_t value)
{
    uint32_t shift;

    for (shift = 0u; shift < 64u; shift += 8u)
    {
        destination[shift / 8u] = (uint8_t)((value >> shift) & 0xFFu);
    }
}

static void TestSeamPutF32Le(
    uint8_t *destination,
    float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    TestSeamPutU32Le(destination, bits);
}

static uint32_t TestSeamGetU32Le(
    const uint8_t *source)
{
    return (uint32_t)source[0] |
        ((uint32_t)source[1] << 8u) |
        ((uint32_t)source[2] << 16u) |
        ((uint32_t)source[3] << 24u);
}

static uint64_t TestSeamGetU64Le(
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

static int TestSeamStubReadExact(
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

static int TestSeamStubWriteAll(
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

static uint32_t TestSeamStubBuildResponse(
    uint32_t mode,
    uint64_t sequence_id,
    uint64_t generation,
    uint8_t *response)
{
    static const TestSeamStubNode deep_nodes[SPARK_TEST_SEAM_DEEP_NODE_COUNT] =
    {
        { 900u, SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX, 0u, 0u, 1.0f },
        { 901u, 0u, 1u, 8u, 0.5f },
        { 902u, 1u, 2u, 16u, 0.625f },
        { 903u, 2u, 3u, 8u, 0.75f },
        { 911u, 0u, 1u, 16u, 0.875f }
    };
    static const TestSeamStubNode
        malformed_nodes[SPARK_TEST_SEAM_MALFORMED_NODE_COUNT] =
    {
        { 900u, SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX, 0u, 0u, 1.0f },
        { 901u, 2u, 2u, 8u, 0.5f },
        { 902u, 0u, 1u, 16u, 0.625f }
    };
    const TestSeamStubNode *nodes;
    uint8_t *cursor;
    uint32_t node_count;
    uint32_t node_index;

    if (mode == SPARK_TEST_SEAM_MODE_MALFORMED_TREE)
    {
        nodes = malformed_nodes;
        node_count = SPARK_TEST_SEAM_MALFORMED_NODE_COUNT;
    }
    else
    {
        nodes = deep_nodes;
        node_count = SPARK_TEST_SEAM_DEEP_NODE_COUNT;
    }
    cursor = response;
    memcpy(cursor, "DFT3", 4u);
    cursor += 4u;
    TestSeamPutU32Le(cursor, SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK);
    cursor += sizeof(uint32_t);
    TestSeamPutU32Le(cursor, node_count);
    cursor += sizeof(uint32_t);
    TestSeamPutU64Le(cursor, sequence_id);
    cursor += sizeof(uint64_t);
    TestSeamPutU64Le(cursor, generation);
    cursor += sizeof(uint64_t);
    for (node_index = 0u; node_index < node_count; ++node_index)
    {
        TestSeamPutU32Le(cursor, nodes[node_index].token_id);
        cursor += sizeof(uint32_t);
        TestSeamPutU32Le(cursor, nodes[node_index].parent_index);
        cursor += sizeof(uint32_t);
        TestSeamPutU32Le(cursor, nodes[node_index].depth);
        cursor += sizeof(uint32_t);
        TestSeamPutU32Le(cursor, nodes[node_index].source_bit);
        cursor += sizeof(uint32_t);
        TestSeamPutF32Le(cursor, nodes[node_index].score);
        cursor += sizeof(float);
    }
    memset(cursor, 0, SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES);
    cursor += SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES;
    return (uint32_t)(cursor - response);
}

static void *TestSeamStubMain(
    void *argument)
{
    TestSeamStub *stub;
    uint8_t header[SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES];
    uint8_t response[SPARK_TEST_SEAM_RESPONSE_BYTES];
    uint64_t body_bytes;
    uint64_t total_bytes;
    uint32_t exchange_index;
    uint32_t n_committed;
    uint32_t n_taps;
    uint32_t response_bytes;
    int connection;
    int read_result;

    stub = (TestSeamStub *)argument;
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
            read_result = TestSeamStubReadExact(
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
        stub->last_speculator_mask = TestSeamGetU32Le(
            header + SPARK_TEST_SEAM_HEADER_OFF_SPECULATOR_MASK);
        stub->last_sequence_id = TestSeamGetU64Le(
            header + SPARK_TEST_SEAM_HEADER_OFF_SEQUENCE_ID);
        stub->last_generation = TestSeamGetU64Le(
            header + SPARK_TEST_SEAM_HEADER_OFF_GENERATION);
        n_committed = TestSeamGetU32Le(
            header + SPARK_TEST_SEAM_HEADER_OFF_COMMITTED_COUNT);
        n_taps = TestSeamGetU32Le(
            header + SPARK_TEST_SEAM_HEADER_OFF_TAP_COUNT);
        body_bytes =
            (uint64_t)n_committed * sizeof(uint32_t) +
            (uint64_t)n_taps * SPARK_TEST_SEAM_TAP_ROW_BYTES;
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
            read_result = TestSeamStubReadExact(
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
        response_bytes = TestSeamStubBuildResponse(
            stub->modes[exchange_index],
            stub->last_sequence_id,
            stub->last_generation,
            response);
        if (TestSeamStubWriteAll(
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

static uint32_t TestSeamStubStart(
    TestSeamStub *stub,
    const uint32_t *modes,
    uint32_t exchange_count)
{
    struct sockaddr_in address;
    socklen_t address_bytes;
    int reuse;

    memset(stub, 0, sizeof(*stub));
    stub->modes = modes;
    stub->exchange_count = exchange_count;
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
    if (listen(stub->listen_fd, SPARK_TEST_SEAM_LISTEN_BACKLOG) != 0)
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
    if (pthread_create(&stub->thread, 0, TestSeamStubMain, stub) != 0)
    {
        return 1u;
    }
    return 0u;
}

static void TestSeamStubStop(
    TestSeamStub *stub)
{
    (void)pthread_join(stub->thread, 0);
    (void)close(stub->listen_fd);
}

static void TestSeamWriteModelContract(
    SparkSpeculationModelContract *model_contract,
    uint32_t aux_layer_count)
{
    uint32_t layer_index;

    memset(model_contract, 0, sizeof(*model_contract));
    model_contract->abi_version = SPARK_SPECULATION_ABI_VERSION;
    model_contract->descriptor_bytes =
        SPARK_SPECULATION_MODEL_CONTRACT_DESCRIPTOR_BYTES;
    model_contract->verifier_hidden_dtype =
        SPARK_SPECULATION_VERIFIER_HIDDEN_DTYPE_BF16;
    model_contract->draft_dtype = SPARK_SPECULATION_DRAFT_DTYPE_BF16;
    model_contract->draft_layer_count = 1u;
    model_contract->block_size = 16u;
    model_contract->hidden_dimension = SPARK_TEST_SEAM_HIDDEN_DIMENSION;
    model_contract->intermediate_dimension = 16u;
    model_contract->attention_head_count = 2u;
    model_contract->kv_head_count = 1u;
    model_contract->head_dimension = 4u;
    model_contract->vocab_size = SPARK_TEST_SEAM_VOCAB_SIZE;
    model_contract->draft_vocab_size = SPARK_TEST_SEAM_VOCAB_SIZE;
    model_contract->maximum_speculative_token_count =
        SPARK_TEST_SEAM_MAX_SPECULATIVE;
    model_contract->verifier_accept_k = SPARK_TEST_SEAM_VERIFIER_ACCEPT_K;
    model_contract->aux_layer_count = aux_layer_count;
    for (layer_index = 0u; layer_index < aux_layer_count; ++layer_index)
    {
        model_contract->aux_layer_ids[layer_index] = 3u + layer_index;
    }
}

static void TestSeamWriteConfiguration(
    SparkSpeculationSeamConfiguration *configuration,
    uint32_t available_source_mask,
    uint32_t bridge_port,
    uint32_t aux_layer_count,
    uint32_t max_tap_row_count)
{
    memset(configuration, 0, sizeof(*configuration));
    configuration->abi_version = SPARK_SPECULATION_SEAM_ABI_VERSION;
    configuration->descriptor_bytes = SPARK_SPECULATION_SEAM_DESCRIPTOR_BYTES;
    configuration->available_source_mask = available_source_mask;
    configuration->default_speculative_token_count = 4u;
    configuration->lane_count = 2u;
    configuration->max_committed_token_count = SPARK_TEST_SEAM_MAX_COMMITTED;
    configuration->max_tap_row_count = max_tap_row_count;
    configuration->draft_time_budget_ms = SPARK_TEST_SEAM_TIME_BUDGET_MS;
    configuration->draft_max_depth = SPARK_DRAFT_BRIDGE_MAX_DEPTH;
    configuration->draft_max_node_count = SPARK_TEST_SEAM_MAX_NODES;
    configuration->connect_timeout_ms = SPARK_TEST_SEAM_TIMEOUT_MS;
    configuration->io_timeout_ms = SPARK_TEST_SEAM_TIMEOUT_MS;
    configuration->bridge_host = "127.0.0.1";
    configuration->bridge_port = bridge_port;
    memcpy(
        configuration->target_model,
        SPARK_TEST_SEAM_TARGET_MODEL,
        sizeof(SPARK_TEST_SEAM_TARGET_MODEL));
    TestSeamWriteModelContract(
        &configuration->model_contract,
        aux_layer_count);
}

static uint32_t TestSeamDraftStandard(
    SparkSpeculationSeam *seam,
    uint64_t request_id,
    uint32_t with_taps,
    uint32_t *draft_token_ids_out,
    uint32_t *draft_token_count_out)
{
    static const uint32_t committed[3u] = { 11u, 22u, 33u };
    uint8_t tap_rows[SPARK_TEST_SEAM_TAP_ROW_BYTES];
    uint32_t byte_index;

    for (byte_index = 0u; byte_index < sizeof(tap_rows); ++byte_index)
    {
        tap_rows[byte_index] = (uint8_t)(0xA0u + byte_index);
    }
    return SparkSpeculationSeamDraftRemoteChain(
        seam,
        request_id,
        SPARK_TEST_SEAM_SEQUENCE_ID,
        2u,
        33u,
        committed,
        3u,
        with_taps != 0u ? tap_rows : 0,
        with_taps != 0u ? 1u : 0u,
        3u,
        draft_token_ids_out,
        SPARK_TEST_SEAM_DRAFT_CAPACITY,
        draft_token_count_out);
}

static uint32_t TestSeamCheckDeepChain(
    const uint32_t *draft_token_ids,
    uint32_t draft_token_count)
{
    SPARK_TEST_SEAM_CHECK(draft_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(draft_token_ids[0] == 901u);
    SPARK_TEST_SEAM_CHECK(draft_token_ids[1] == 902u);
    SPARK_TEST_SEAM_CHECK(draft_token_ids[2] == 903u);
    return 0u;
}

static uint32_t TestSeamParseControlMatrix(void)
{
    uint32_t enabled;

    enabled = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            0,
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(enabled == SPARK_SPECULATION_SEAM_KNOWN_SOURCES);
    enabled = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "1",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(enabled == SPARK_SPECULATION_SEAM_KNOWN_SOURCES);
    enabled = 0xFFFFFFFFu;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "0",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(enabled == 0u);
    enabled = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "0x4",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(enabled == SPARK_SPECULATION_SEAM_SOURCE_DFLASH2);
    enabled = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "8",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(enabled == SPARK_SPECULATION_SEAM_SOURCE_NGRAM);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "0x40",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "yes",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "0x",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            "0x100000000",
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            &enabled) == SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamParseControl(
            0,
            SPARK_SPECULATION_SEAM_KNOWN_SOURCES,
            0) == SPARK_STATUS_INVALID_ARGUMENT);
    return 0u;
}

static uint32_t TestSeamInitializeValidation(void)
{
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationSeam *seam;

    seam = 0;
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        1u,
        0u,
        0u);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(0, &seam) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, 0) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    seam = 0;
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        1u,
        0u,
        0u);
    configuration.abi_version = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(&configuration, 0x40u, 1u, 0u, 0u);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP,
        0u,
        0u,
        0u);
    configuration.lane_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM,
        1u,
        0u,
        0u);
    configuration.bridge_host = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_DFLASH2,
        1u,
        0u,
        SPARK_TEST_SEAM_MAX_TAP_ROWS);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_DFLASH2,
        1u,
        1u,
        0u);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM,
        1u,
        0u,
        SPARK_TEST_SEAM_MAX_TAP_ROWS);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM,
        1u,
        0u,
        0u);
    memset(
        configuration.target_model,
        'x',
        SPARK_SPECULATION_SEAM_TARGET_MODEL_BYTES);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SPARK_TEST_SEAM_CHECK(seam == 0);

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK,
        0u,
        0u,
        0u);
    configuration.bridge_host = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(seam != 0);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamEnabledSources(seam) ==
        (SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK));
    SPARK_TEST_SEAM_CHECK(SparkSpeculationSeamSpeculator(seam) != 0);
    SparkSpeculationSeamDestroy(seam);
    SparkSpeculationSeamDestroy(0);
    SPARK_TEST_SEAM_CHECK(SparkSpeculationSeamEnabledSources(0) == 0u);
    SPARK_TEST_SEAM_CHECK(SparkSpeculationSeamSpeculator(0) == 0);
    return 0u;
}

static uint32_t TestSeamDraftAcceptRejectedBumpsGeneration(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_SEAM_MODE_DEEP_TREE, SPARK_TEST_SEAM_MODE_DEEP_TREE };
    static const uint32_t verifier_ids[3u] = { 901u, 902u, 777u };
    static const uint32_t committed_after[6u] =
        { 11u, 22u, 33u, 901u, 902u, 777u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSpeculator *speculator;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 2u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamEnabledSources(seam) ==
        (SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX));

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 11u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 0u);
    SPARK_TEST_SEAM_CHECK(
        stub.last_speculator_mask ==
        (SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX));

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_ids,
            3u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.proposed_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 2u);
    SPARK_TEST_SEAM_CHECK(verify_result.committed_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 777u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    speculator = SparkSpeculationSeamSpeculator(seam);
    SPARK_TEST_SEAM_CHECK(speculator->verify_dispatch_count == 1u);
    SPARK_TEST_SEAM_CHECK(speculator->accepted_draft_token_count == 2u);
    SPARK_TEST_SEAM_CHECK(speculator->rejected_token_count == 1u);
    SPARK_TEST_SEAM_CHECK(speculator->committed_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(speculator->draft_request_count == 1u);
    SPARK_TEST_SEAM_CHECK(speculator->draft_success_count == 1u);
    SPARK_TEST_SEAM_CHECK(speculator->tap_ready_count == 1u);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamDraftRemoteChain(
            seam,
            12u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            5u,
            777u,
            committed_after,
            6u,
            0,
            0u,
            3u,
            draft_ids,
            SPARK_TEST_SEAM_DRAFT_CAPACITY,
            &draft_count) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 1u);
    SPARK_TEST_SEAM_CHECK(
        stub.last_speculator_mask ==
        (SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX));

    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 2u);
    return 0u;
}

static uint32_t TestSeamFullAcceptKeepsGeneration(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_SEAM_MODE_DEEP_TREE, SPARK_TEST_SEAM_MODE_DEEP_TREE };
    static const uint32_t verifier_ids[4u] = { 901u, 902u, 903u, 444u };
    static const uint32_t committed_after[7u] =
        { 11u, 22u, 33u, 901u, 902u, 903u, 444u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 2u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 21u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 0u);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_ids,
            4u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.proposed_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.committed_token_count == 4u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 444u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_ACCEPTED_ALL) != 0u);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamDraftRemoteChain(
            seam,
            22u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            6u,
            444u,
            committed_after,
            7u,
            0,
            0u,
            3u,
            draft_ids,
            SPARK_TEST_SEAM_DRAFT_CAPACITY,
            &draft_count) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 0u);

    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 2u);
    return 0u;
}

static uint32_t TestSeamRepeatDraftReusesStagedChain(void)
{
    static const uint32_t modes[] = { SPARK_TEST_SEAM_MODE_DEEP_TREE };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t first_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t second_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t first_count;
    uint32_t second_count;

    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 1u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    first_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 31u, 0u, first_ids, &first_count) ==
        SPARK_STATUS_OK);
    second_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 32u, 0u, second_ids, &second_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(first_count == second_count);
    SPARK_TEST_SEAM_CHECK(
        memcmp(
            first_ids,
            second_ids,
            (size_t)first_count * sizeof(uint32_t)) == 0);
    SPARK_TEST_SEAM_CHECK(
        TestSeamCheckDeepChain(second_ids, second_count) == 0u);

    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 1u);
    SPARK_TEST_SEAM_CHECK(stub.accept_count == 1u);
    return 0u;
}

static uint32_t TestSeamAcceptWithoutDraftFails(void)
{
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t verifier_ids[1u];

    verifier_ids[0] = 901u;
    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, 0, 0u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_ids,
            1u,
            &verify_result) == SPARK_STATUS_NOT_FOUND);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            0,
            1u,
            &verify_result) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_ids,
            1u,
            0) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.bytes_received == 0u);
    return 0u;
}

static uint32_t TestSeamCancelResetsSequence(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_SEAM_MODE_DEEP_TREE, SPARK_TEST_SEAM_MODE_DEEP_TREE };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t verifier_ids[1u];
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    verifier_ids[0] = 901u;
    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 2u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 41u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 0u);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamCancelSequence(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_ids,
            1u,
            &verify_result) == SPARK_STATUS_NOT_FOUND);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamCancelSequence(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID) == SPARK_STATUS_NOT_FOUND);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 42u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 1u);

    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 2u);
    return 0u;
}

static uint32_t TestSeamMalformedTreeFailsValidation(void)
{
    static const uint32_t modes[] = { SPARK_TEST_SEAM_MODE_MALFORMED_TREE };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 1u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 51u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_VALIDATION_FAILED);
    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 1u);
    return 0u;
}

static uint32_t TestSeamNoRemoteSourcesUnsupported(void)
{
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationSeam *seam;
    uint32_t committed[3u];
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    committed[0] = 11u;
    committed[1] = 22u;
    committed[2] = 33u;
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK,
        0u,
        0u,
        0u);
    configuration.bridge_host = 0;
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamEnabledSources(seam) ==
        (SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK));
    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamDraftRemoteChain(
            seam,
            61u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            33u,
            committed,
            3u,
            0,
            0u,
            3u,
            draft_ids,
            SPARK_TEST_SEAM_DRAFT_CAPACITY,
            &draft_count) == SPARK_STATUS_UNSUPPORTED);
    SPARK_TEST_SEAM_CHECK(draft_count == 0u);
    SparkSpeculationSeamDestroy(seam);
    return 0u;
}

static uint32_t TestSeamTapSourceWireMaskTracksTaps(void)
{
    static const uint32_t modes[] =
        { SPARK_TEST_SEAM_MODE_DEEP_TREE, SPARK_TEST_SEAM_MODE_DEEP_TREE };
    static const uint32_t verifier_ids[4u] = { 901u, 902u, 903u, 444u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 2u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_DFLASH2,
        stub.port,
        1u,
        SPARK_TEST_SEAM_MAX_TAP_ROWS);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamEnabledSources(seam) ==
        (SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_DFLASH2));

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 81u, 1u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(
        stub.last_speculator_mask ==
        (SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_DFLASH2));

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_ids,
            4u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 3u);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 82u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(
        stub.last_speculator_mask == SPARK_SPECULATION_SEAM_SOURCE_NGRAM);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 0u);

    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 2u);
    return 0u;
}

static uint32_t TestSeamTapSourceWithoutTapsUnsupported(void)
{
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t committed[3u];
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    committed[0] = 11u;
    committed[1] = 22u;
    committed[2] = 33u;
    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, 0, 0u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_DFLASH2,
        stub.port,
        1u,
        SPARK_TEST_SEAM_MAX_TAP_ROWS);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamEnabledSources(seam) ==
        SPARK_SPECULATION_SEAM_SOURCE_DFLASH2);
    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamDraftRemoteChain(
            seam,
            71u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            33u,
            committed,
            3u,
            0,
            0u,
            3u,
            draft_ids,
            SPARK_TEST_SEAM_DRAFT_CAPACITY,
            &draft_count) == SPARK_STATUS_UNSUPPORTED);
    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.bytes_received == 0u);
    return 0u;
}

static uint32_t TestSeamLocalDraftAcceptPaths(void)
{
    static const uint32_t local_chain[3u] = { 201u, 202u, 203u };
    static const uint32_t verifier_mid[3u] = { 201u, 202u, 777u };
    static const uint32_t verifier_full[4u] = { 201u, 202u, 203u, 444u };
    static const uint32_t verifier_reject[3u] = { 999u, 998u, 997u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSpeculator *speculator;
    SparkSpeculationSeam *seam;

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK,
        0u,
        0u,
        0u);
    configuration.bridge_host = 0;
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);
    speculator = SparkSpeculationSeamSpeculator(seam);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            101u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            3u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_mid,
            3u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.proposed_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 2u);
    SPARK_TEST_SEAM_CHECK(verify_result.committed_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 777u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    SPARK_TEST_SEAM_CHECK(speculator->verify_dispatch_count == 1u);
    SPARK_TEST_SEAM_CHECK(speculator->accepted_draft_token_count == 2u);
    SPARK_TEST_SEAM_CHECK(speculator->rejected_token_count == 1u);
    SPARK_TEST_SEAM_CHECK(speculator->committed_token_count == 3u);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            102u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            5u,
            local_chain,
            3u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_full,
            4u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(verify_result.committed_token_count == 4u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 444u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_ACCEPTED_ALL) != 0u);
    SPARK_TEST_SEAM_CHECK(speculator->verify_dispatch_count == 2u);
    SPARK_TEST_SEAM_CHECK(speculator->accepted_draft_token_count == 5u);
    SPARK_TEST_SEAM_CHECK(speculator->committed_token_count == 7u);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            103u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            6u,
            local_chain,
            3u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_reject,
            3u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 0u);
    SPARK_TEST_SEAM_CHECK(verify_result.committed_token_count == 1u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 999u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);
    SPARK_TEST_SEAM_CHECK(speculator->verify_dispatch_count == 3u);
    SPARK_TEST_SEAM_CHECK(speculator->accepted_draft_token_count == 5u);
    SPARK_TEST_SEAM_CHECK(speculator->rejected_token_count == 4u);
    SPARK_TEST_SEAM_CHECK(speculator->committed_token_count == 8u);
    SPARK_TEST_SEAM_CHECK(speculator->draft_request_count == 0u);
    SPARK_TEST_SEAM_CHECK(speculator->draft_success_count == 0u);

    SparkSpeculationSeamDestroy(seam);
    return 0u;
}

static uint32_t TestSeamLocalDraftDoubleStageBusy(void)
{
    static const uint32_t local_chain[2u] = { 201u, 202u };
    static const uint32_t verifier_full[3u] = { 201u, 202u, 333u };
    static const uint32_t verifier_single[2u] = { 201u, 555u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK,
        0u,
        0u,
        0u);
    configuration.bridge_host = 0;
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            111u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            112u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_BUSY);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_full,
            3u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 2u);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            113u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            4u,
            local_chain,
            1u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_single,
            2u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.proposed_token_count == 1u);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 1u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 555u);

    SparkSpeculationSeamDestroy(seam);
    return 0u;
}

static uint32_t TestSeamLocalDraftBadTokenRejected(void)
{
    static const uint32_t local_chain[2u] = { 201u, 202u };
    static const uint32_t bad_chain[2u] =
        { 201u, SPARK_TEST_SEAM_VOCAB_SIZE };
    static const uint32_t verifier_full[3u] = { 201u, 202u, 333u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK,
        0u,
        0u,
        0u);
    configuration.bridge_host = 0;
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            121u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            bad_chain,
            2u) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            122u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            0u) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            123u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            SPARK_TEST_SEAM_MAX_SPECULATIVE + 1u) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            124u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            0,
            2u) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            0,
            125u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_INVALID_ARGUMENT);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_full,
            3u,
            &verify_result) == SPARK_STATUS_NOT_FOUND);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            126u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_full,
            3u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 2u);

    SparkSpeculationSeamDestroy(seam);
    return 0u;
}

static uint32_t TestSeamLocalThenRemoteSameLane(void)
{
    static const uint32_t modes[] = { SPARK_TEST_SEAM_MODE_DEEP_TREE };
    static const uint32_t local_chain[2u] = { 201u, 202u };
    static const uint32_t verifier_mid[2u] = { 201u, 999u };
    static const uint32_t verifier_remote[4u] = { 901u, 902u, 903u, 444u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;
    TestSeamStub stub;
    uint32_t draft_ids[SPARK_TEST_SEAM_DRAFT_CAPACITY];
    uint32_t draft_count;

    SPARK_TEST_SEAM_CHECK(TestSeamStubStart(&stub, modes, 1u) == 0u);
    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX,
        stub.port,
        0u,
        0u);
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            131u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_mid,
            2u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 1u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_REJECTED) != 0u);

    draft_count = 0u;
    SPARK_TEST_SEAM_CHECK(
        TestSeamDraftStandard(seam, 132u, 0u, draft_ids, &draft_count) ==
        SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(TestSeamCheckDeepChain(draft_ids, draft_count) == 0u);
    SPARK_TEST_SEAM_CHECK(stub.last_generation == 1u);
    SPARK_TEST_SEAM_CHECK(
        stub.last_speculator_mask ==
        (SPARK_SPECULATION_SEAM_SOURCE_NGRAM |
            SPARK_SPECULATION_SEAM_SOURCE_SUFFIX));

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_remote,
            4u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 3u);
    SPARK_TEST_SEAM_CHECK(
        (verify_result.flags &
            SPARK_SPECULATION_VERIFY_RESULT_FLAG_ACCEPTED_ALL) != 0u);

    SparkSpeculationSeamDestroy(seam);
    TestSeamStubStop(&stub);
    SPARK_TEST_SEAM_CHECK(stub.failed == 0u);
    SPARK_TEST_SEAM_CHECK(stub.exchanges_completed == 1u);
    return 0u;
}

static uint32_t TestSeamLocalStageAfterCancel(void)
{
    static const uint32_t local_chain[2u] = { 201u, 202u };
    static const uint32_t verifier_full[3u] = { 201u, 202u, 333u };
    SparkSpeculationSeamConfiguration configuration;
    SparkSpeculationPolicyVerifyResult verify_result;
    SparkSpeculationSeam *seam;

    TestSeamWriteConfiguration(
        &configuration,
        SPARK_SPECULATION_SEAM_SOURCE_MTP |
            SPARK_SPECULATION_SEAM_SOURCE_DSPARK,
        0u,
        0u,
        0u);
    configuration.bridge_host = 0;
    seam = 0;
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamInitialize(&configuration, &seam) ==
        SPARK_STATUS_OK);

    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            141u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamCancelSequence(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_full,
            3u,
            &verify_result) == SPARK_STATUS_NOT_FOUND);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamStageLocalDraft(
            seam,
            142u,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            2u,
            local_chain,
            2u) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(
        SparkSpeculationSeamAcceptChain(
            seam,
            SPARK_TEST_SEAM_SEQUENCE_ID,
            verifier_full,
            3u,
            &verify_result) == SPARK_STATUS_OK);
    SPARK_TEST_SEAM_CHECK(verify_result.accepted_draft_token_count == 2u);
    SPARK_TEST_SEAM_CHECK(verify_result.fallback_token_id == 333u);

    SparkSpeculationSeamDestroy(seam);
    return 0u;
}

static uint32_t TestSeamRunCase(
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
    failures += TestSeamRunCase(
        "parse_control_matrix",
        TestSeamParseControlMatrix);
    failures += TestSeamRunCase(
        "initialize_validation",
        TestSeamInitializeValidation);
    failures += TestSeamRunCase(
        "draft_accept_rejected_bumps_generation",
        TestSeamDraftAcceptRejectedBumpsGeneration);
    failures += TestSeamRunCase(
        "full_accept_keeps_generation",
        TestSeamFullAcceptKeepsGeneration);
    failures += TestSeamRunCase(
        "repeat_draft_reuses_staged_chain",
        TestSeamRepeatDraftReusesStagedChain);
    failures += TestSeamRunCase(
        "accept_without_draft_fails",
        TestSeamAcceptWithoutDraftFails);
    failures += TestSeamRunCase(
        "cancel_resets_sequence",
        TestSeamCancelResetsSequence);
    failures += TestSeamRunCase(
        "malformed_tree_fails_validation",
        TestSeamMalformedTreeFailsValidation);
    failures += TestSeamRunCase(
        "no_remote_sources_unsupported",
        TestSeamNoRemoteSourcesUnsupported);
    failures += TestSeamRunCase(
        "tap_source_wire_mask_tracks_taps",
        TestSeamTapSourceWireMaskTracksTaps);
    failures += TestSeamRunCase(
        "tap_source_without_taps_unsupported",
        TestSeamTapSourceWithoutTapsUnsupported);
    failures += TestSeamRunCase(
        "local_draft_accept_paths",
        TestSeamLocalDraftAcceptPaths);
    failures += TestSeamRunCase(
        "local_draft_double_stage_busy",
        TestSeamLocalDraftDoubleStageBusy);
    failures += TestSeamRunCase(
        "local_draft_bad_token_rejected",
        TestSeamLocalDraftBadTokenRejected);
    failures += TestSeamRunCase(
        "local_then_remote_same_lane",
        TestSeamLocalThenRemoteSameLane);
    failures += TestSeamRunCase(
        "local_stage_after_cancel",
        TestSeamLocalStageAfterCancel);
    if (failures != 0u)
    {
        fprintf(stderr, "FAILURES %u\n", (unsigned)failures);
        return 1;
    }
    return 0;
}
