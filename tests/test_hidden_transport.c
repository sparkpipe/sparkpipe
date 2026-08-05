#include <assert.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_hidden_transport.h"

#define SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT 2u
#define SPARK_TEST_HIDDEN_TRANSPORT_MAX_ACTIVE_SEQUENCE_COUNT 128u

#ifndef SPARK_TEST_HIDDEN_TRANSPORT_MODULE_PATH
#define SPARK_TEST_HIDDEN_TRANSPORT_MODULE_PATH \
    "build/test_modules/libhidden_transport_module.so"
#endif

typedef struct TestHiddenTransportState
{
    uint32_t send_count;
    uint32_t receive_count;
    uint32_t send_batch_count;
    uint32_t receive_batch_count;
    uint32_t last_batch_packet_count;
} TestHiddenTransportState;

static TestHiddenTransportState g_test_hidden_transport_state;

static SparkStatus TestHiddenTransportInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    assert(endpoint != 0);
    assert(transport_state != 0);
    memset(&g_test_hidden_transport_state, 0, sizeof(g_test_hidden_transport_state));
    *transport_state = &g_test_hidden_transport_state;
    return SPARK_STATUS_OK;
}

static void TestHiddenTransportDestroy(void *transport_state)
{
    assert(transport_state == &g_test_hidden_transport_state);
}

static SparkStatus TestHiddenTransportPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    TestHiddenTransportState *state;

    assert(transport_state != 0);
    assert(packet != 0);
    state = (TestHiddenTransportState *)transport_state;
    state->receive_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    TestHiddenTransportState *state;

    assert(transport_state != 0);
    assert(packet != 0);
    state = (TestHiddenTransportState *)transport_state;
    state->send_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportPostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    TestHiddenTransportState *state;

    assert(transport_state != 0);
    assert(packets != 0);
    assert(packet_count != 0u);
    state = (TestHiddenTransportState *)transport_state;
    state->receive_batch_count += 1u;
    state->last_batch_packet_count = packet_count;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    TestHiddenTransportState *state;

    assert(transport_state != 0);
    assert(packets != 0);
    assert(packet_count != 0u);
    state = (TestHiddenTransportState *)transport_state;
    state->send_batch_count += 1u;
    state->last_batch_packet_count = packet_count;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportPoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    assert(transport_state != 0);
    assert(completion != 0);
    completion->status = SPARK_STATUS_BUSY;
    return SPARK_STATUS_OK;
}

static void SparkTestInitializeEndpoint(
    SparkHiddenTransportEndpoint *endpoint)
{
    memset(endpoint, 0, sizeof(*endpoint));
    endpoint->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint->capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS;
    endpoint->hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
    endpoint->bytes_per_sequence = SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES;
    endpoint->max_active_sequence_count =
        SPARK_TEST_HIDDEN_TRANSPORT_MAX_ACTIVE_SEQUENCE_COUNT;
    endpoint->max_packet_bytes =
        SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES *
        SPARK_TEST_HIDDEN_TRANSPORT_MAX_ACTIVE_SEQUENCE_COUNT;
    endpoint->validated_latency_ns = 200000u;
    endpoint->transport_module_id =
        "spark.glm52.hidden_stage_transport.100g.persistent.v1";
    endpoint->route_name = "spark2_to_sparka";
}

static void SparkTestInitializeSparkHostRdmaEndpoint(
    SparkHiddenTransportEndpoint *endpoint)
{
    SparkHiddenTransportInitializeSparkHostRdmaEndpoint(
        endpoint,
        SPARK_GLM52_MODEL_HIDDEN_DIMENSION,
        SPARK_TEST_HIDDEN_TRANSPORT_MAX_ACTIVE_SEQUENCE_COUNT,
        200000u,
        "spark2_to_sparka_hidden");
}

static void SparkTestInitializePacket(
    SparkHiddenTransportPacket *packet,
    const SparkHiddenTransportEndpoint *endpoint,
    uint16_t *hidden_payload,
    uint64_t sequence_id)
{
    memset(packet, 0, sizeof(*packet));
    packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    packet->flags =
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    packet->active_sequence_count =
        SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT;
    packet->hidden_dimension = endpoint->hidden_dimension;
    packet->bytes_per_sequence = endpoint->bytes_per_sequence;
    packet->sequence_id = sequence_id;
    packet->token_index = 19u;
    packet->hidden_bf16 = hidden_payload;
    packet->cuda_stream = (void *)0x1;
}

static void SparkTestInitializeTransportInterface(
    SparkHiddenTransportInterface *transport_interface,
    uint32_t capability_flags)
{
    memset(transport_interface, 0, sizeof(*transport_interface));
    transport_interface->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface->descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface->capability_flags = capability_flags;
    transport_interface->initialize = TestHiddenTransportInitialize;
    transport_interface->destroy = TestHiddenTransportDestroy;
    transport_interface->post_receive = TestHiddenTransportPostReceive;
    transport_interface->send = TestHiddenTransportSend;
    transport_interface->poll = TestHiddenTransportPoll;
    if ((capability_flags &
            SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION) != 0u)
    {
        transport_interface->post_receive_batch =
            TestHiddenTransportPostReceiveBatch;
        transport_interface->send_batch = TestHiddenTransportSendBatch;
    }
}

static void SparkTestHiddenTransportValidatesEndpointAndPacket(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportPacket packet;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];

    SparkTestInitializeEndpoint(&endpoint);
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) == SPARK_STATUS_OK);
    endpoint.capability_flags &= ~SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS;
    endpoint.max_packet_bytes = 12287u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_CAPACITY_EXCEEDED);
    endpoint.max_packet_bytes = SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES * SPARK_TEST_HIDDEN_TRANSPORT_MAX_ACTIVE_SEQUENCE_COUNT;
    endpoint.bytes_per_sequence = 12290u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.bytes_per_sequence = SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES;
    endpoint.local_rank_index = 1u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.configuration_flags =
        SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_EXPLICIT_ROUTE_CONFIGURATION;
    endpoint.source_rank_index = 0u;
    endpoint.sink_rank_index = 1u;
    endpoint.control_port_base = 59000u;
    endpoint.source_host = "node-alpha";
    endpoint.sink_host = "node-beta";
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) == SPARK_STATUS_OK);
    endpoint.source_rank_index = 1u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.source_rank_index = 0u;
    endpoint.sink_host = "0.0.0.0";
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.sink_host = "node-beta";
    endpoint.reserved0 = 1u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.reserved0 = 0u;

    SparkTestInitializePacket(&packet, &endpoint, hidden_payload, 7u);
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_OK);
    packet.flags &= ~SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    packet.flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    packet.active_sequence_count = 129u;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    packet.active_sequence_count = 2u;
    packet.cuda_stream = 0;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    packet.cuda_stream = (void *)0x1;
    packet.hidden_dimension = 4096u;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestHiddenTransportValidatesSidebandPayload(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportPacket packet;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];
    uint32_t sideband_payload[SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];

    SparkTestInitializeEndpoint(&endpoint);
    SparkTestInitializePacket(&packet, &endpoint, hidden_payload, 9u);
    packet.flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
    packet.sideband_payload = sideband_payload;
    packet.sideband_kind = 1u;
    packet.sideband_bytes_per_sequence = SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_OK);

    packet.sideband_payload = 0;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    packet.sideband_payload = sideband_payload;
    packet.sideband_kind = 0u;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    packet.sideband_kind = 1u;
    packet.sideband_bytes_per_sequence = 0u;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    packet.sideband_bytes_per_sequence = SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES;
    packet.flags &= ~SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
    assert(SparkHiddenTransportValidatePacket(&endpoint, &packet) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestHiddenTransportRejectsMissingNativeBatch(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportPacket packets[3];
    SparkHiddenTransportCompletion completion;
    SparkHiddenTransportSession *session;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];
    uint32_t packet_index;

    SparkTestInitializeEndpoint(&endpoint);
    for (packet_index = 0u; packet_index < 3u; ++packet_index)
    {
        SparkTestInitializePacket(
            &packets[packet_index],
            &endpoint,
            hidden_payload,
            (uint64_t)packet_index + 1u);
    }
    assert(SparkHiddenTransportValidatePacketBatch(
        &endpoint,
        packets,
        3u) == SPARK_STATUS_OK);

    SparkTestInitializeTransportInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS);
    assert(SparkHiddenTransportValidateInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS) == SPARK_STATUS_OK);

    session = 0;
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
        &session) == SPARK_STATUS_OK);
    assert(session != 0);
    assert(SparkHiddenTransportSend(session, &packets[0]) == SPARK_STATUS_OK);
    assert(g_test_hidden_transport_state.send_count == 1u);
    assert(SparkHiddenTransportPostReceive(session, &packets[0]) ==
        SPARK_STATUS_OK);
    assert(g_test_hidden_transport_state.receive_count == 1u);
    assert(SparkHiddenTransportSendBatch(session, packets, 3u) ==
        SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(g_test_hidden_transport_state.send_count == 1u);
    assert(g_test_hidden_transport_state.send_batch_count == 0u);
    assert(SparkHiddenTransportPostReceiveBatch(session, packets, 3u) ==
        SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(g_test_hidden_transport_state.receive_count == 1u);
    assert(g_test_hidden_transport_state.receive_batch_count == 0u);
    memset(&completion, 0, sizeof(completion));
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.abi_version == SPARK_HIDDEN_TRANSPORT_ABI_VERSION);
    assert(completion.descriptor_bytes == SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES);
    assert(completion.status == SPARK_STATUS_BUSY);
    SparkHiddenTransportClose(session);
}

static void SparkTestHiddenTransportUsesNativeBatchSubmission(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportPacket packets[2];
    SparkHiddenTransportSession *session;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];

    SparkTestInitializeEndpoint(&endpoint);
    endpoint.capability_flags |= SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION;
    SparkTestInitializePacket(&packets[0], &endpoint, hidden_payload, 11u);
    SparkTestInitializePacket(&packets[1], &endpoint, hidden_payload, 12u);

    SparkTestInitializeTransportInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS);
    transport_interface.post_receive_batch = TestHiddenTransportPostReceiveBatch;
    transport_interface.send_batch = TestHiddenTransportSendBatch;
    assert(SparkHiddenTransportValidateInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS) == SPARK_STATUS_OK);

    session = 0;
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS,
        &session) == SPARK_STATUS_OK);
    assert(session != 0);
    assert(SparkHiddenTransportSendBatch(session, packets, 2u) == SPARK_STATUS_OK);
    assert(g_test_hidden_transport_state.send_count == 0u);
    assert(g_test_hidden_transport_state.send_batch_count == 1u);
    assert(g_test_hidden_transport_state.last_batch_packet_count == 2u);
    assert(SparkHiddenTransportPostReceiveBatch(session, packets, 2u) ==
        SPARK_STATUS_OK);
    assert(g_test_hidden_transport_state.receive_count == 0u);
    assert(g_test_hidden_transport_state.receive_batch_count == 1u);
    SparkHiddenTransportClose(session);

    transport_interface.send_batch = 0;
    assert(SparkHiddenTransportValidateInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS) ==
            SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestHiddenTransportCompletionQueuePreservesOrder(void)
{
    SparkHiddenTransportCompletionQueue queue;
    SparkHiddenTransportCompletion completion;
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportPacket packet;
    uint16_t hidden_payload[
        SPARK_GLM52_MODEL_HIDDEN_DIMENSION *
        SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];
    uint32_t packet_index;

    SparkHiddenTransportCompletionQueueInitialize(&queue);
    SparkTestInitializeEndpoint(&endpoint);
    for (packet_index = 0u; packet_index < 3u; ++packet_index)
    {
        SparkTestInitializePacket(
            &packet,&endpoint,hidden_payload,(uint64_t)packet_index + 31u);
        packet.token_index = packet_index + 7u;
        assert(SparkHiddenTransportCompletionQueuePushPacket(
            &queue,&packet,SPARK_STATUS_OK,packet_index + 1u) ==
            SPARK_STATUS_OK);
    }
    assert(queue.count == 3u);
    assert(queue.total_count == 3u);
    assert(queue.dropped_count == 0u);
    for (packet_index = 0u; packet_index < 3u; ++packet_index)
    {
        assert(SparkHiddenTransportCompletionQueuePop(
            &queue,&completion) == SPARK_STATUS_OK);
        assert(completion.status == SPARK_STATUS_OK);
        assert(completion.sequence_id == (uint64_t)packet_index + 31u);
        assert(completion.token_index == packet_index + 7u);
        assert(completion.service_time_ns == packet_index + 1u);
    }
    assert(SparkHiddenTransportCompletionQueuePop(
        &queue,&completion) == SPARK_STATUS_OK);
    assert(completion.status == SPARK_STATUS_BUSY);
}

static void SparkTestHiddenTransportPersistentRingBackend(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportPacket packets[2u];
    SparkHiddenTransportCompletion completion;
    SparkHiddenTransportPersistentRingStatistics statistics;
    SparkHiddenTransportSession *session;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];

    SparkTestInitializeEndpoint(&endpoint);
    endpoint.capability_flags = SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS;
    endpoint.transport_module_id = SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_MODULE_ID;
    SparkTestInitializePacket(&packets[0], &endpoint, hidden_payload, 41u);
    SparkTestInitializePacket(&packets[1], &endpoint, hidden_payload, 42u);

    assert(SparkHiddenTransportPersistentRingGetInterface(&transport_interface) ==
        SPARK_STATUS_OK);
    assert(SparkHiddenTransportValidateInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS) ==
            SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS,
        &session) == SPARK_STATUS_OK);
    assert(session != 0);
    assert(SparkHiddenTransportSendBatch(session, packets, 2u) == SPARK_STATUS_OK);
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.status == SPARK_STATUS_OK);
    assert(completion.sequence_id == 41u);
    assert(completion.transfer_bytes == SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT);
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.status == SPARK_STATUS_OK);
    assert(completion.sequence_id == 42u);
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.status == SPARK_STATUS_BUSY);

    assert(SparkHiddenTransportPersistentRingGetStatistics(
        session,
        &statistics) == SPARK_STATUS_OK);
    assert(statistics.abi_version == SPARK_HIDDEN_TRANSPORT_ABI_VERSION);
    assert(statistics.send_count == 2u);
    assert(statistics.receive_count == 0u);
    assert(statistics.completion_count == 2u);
    assert(statistics.queued_completion_count == 0u);
    SparkHiddenTransportClose(session);
}

static void SparkTestHiddenTransportPersistentRingAccountsSidebandBytes(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportPacket packet;
    SparkHiddenTransportCompletion completion;
    SparkHiddenTransportSession *session;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];
    uint32_t sideband_payload[SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];

    SparkTestInitializeEndpoint(&endpoint);
    endpoint.capability_flags = SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS;
    endpoint.transport_module_id = SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_MODULE_ID;
    SparkTestInitializePacket(&packet, &endpoint, hidden_payload, 43u);
    packet.flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
    packet.sideband_payload = sideband_payload;
    packet.sideband_kind = 1u;
    packet.sideband_bytes_per_sequence = SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES;

    assert(SparkHiddenTransportPersistentRingGetInterface(&transport_interface) ==
        SPARK_STATUS_OK);
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SIMULATION_CAPS,
        &session) == SPARK_STATUS_OK);
    assert(SparkHiddenTransportSend(session, &packet) == SPARK_STATUS_OK);
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.status == SPARK_STATUS_OK);
    assert(completion.transfer_bytes == (SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES + SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES) * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT);
    SparkHiddenTransportClose(session);
}

static void SparkTestHiddenTransportRejectsInvalidInterface(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportSession *session;

    SparkTestInitializeEndpoint(&endpoint);
    SparkTestInitializeTransportInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS);
    transport_interface.send = 0;
    assert(SparkHiddenTransportValidateInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS) ==
            SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
        &session) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestHiddenTransportLoadsProductionModule(void)
{
    SparkHiddenTransportDynamicLibrary library;
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportPacket packet;
    SparkHiddenTransportCompletion completion;
    SparkHiddenTransportSession *session;
    uint16_t hidden_payload[SPARK_GLM52_MODEL_HIDDEN_DIMENSION * SPARK_TEST_HIDDEN_TRANSPORT_ACTIVE_SEQUENCE_COUNT];

    memset(&library, 0, sizeof(library));
    assert(SparkHiddenTransportLoadInterfaceFromSharedObject(
        "build/test_modules/missing_hidden_transport_module.so",
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
        &library) == SPARK_STATUS_DRIVER_LOAD_ERROR);
    assert(library.dynamic_library == 0);

    assert(SparkHiddenTransportLoadInterfaceFromSharedObject(
        SPARK_TEST_HIDDEN_TRANSPORT_MODULE_PATH,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
        &library) == SPARK_STATUS_OK);
    assert(library.abi_version == SPARK_HIDDEN_TRANSPORT_ABI_VERSION);
    assert(library.descriptor_bytes ==
        SPARK_HIDDEN_TRANSPORT_DYNAMIC_LIBRARY_BYTES);
    assert(library.dynamic_library != 0);
    assert((library.transport_interface.capability_flags &
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS) ==
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS);

    SparkTestInitializeEndpoint(&endpoint);
    SparkTestInitializePacket(&packet, &endpoint, hidden_payload, 77u);
    session = 0;
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &library.transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS,
        &session) == SPARK_STATUS_OK);
    assert(session != 0);
    assert(SparkHiddenTransportSend(session, &packet) == SPARK_STATUS_OK);
    memset(&completion, 0, sizeof(completion));
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.status == SPARK_STATUS_BUSY);
    SparkHiddenTransportClose(session);
    SparkHiddenTransportUnloadInterface(&library);
    assert(library.dynamic_library == 0);
}



static void SparkTestHiddenTransportOpensZeroCopyInterfaceFromHostPlan(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportSession *session;

    SparkTestInitializeEndpoint(&endpoint);
    SparkTestInitializeTransportInterface(
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS);
    transport_interface.get_poll_descriptors = 0;
    session = 0;
    assert(SparkHiddenTransportOpen(
        &endpoint,
        &transport_interface,
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        &session) == SPARK_STATUS_OK);
    assert(session != 0);
    SparkHiddenTransportClose(session);
}

static void SparkTestHiddenTransportValidatesSparkHostRdmaEndpoint(void)
{
    SparkHiddenTransportEndpoint endpoint;

    SparkTestInitializeSparkHostRdmaEndpoint(&endpoint);
    assert((endpoint.capability_flags &
        SPARK_HIDDEN_TRANSPORT_CAP_BATCHED_SUBMISSION) != 0u);
    assert(SparkHiddenTransportValidateSparkHostRdmaEndpoint(&endpoint) ==
        SPARK_STATUS_OK);
    endpoint.capability_flags &=
        ~SPARK_HIDDEN_TRANSPORT_CAP_CUDA_MAPPED_HOST_MEMORY;
    assert(SparkHiddenTransportValidateSparkHostRdmaEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestInitializeSparkHostRdmaEndpoint(&endpoint);
    endpoint.capability_flags |= SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY;
    assert(SparkHiddenTransportValidateSparkHostRdmaEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestInitializeSparkHostRdmaEndpoint(&endpoint);
    endpoint.transport_module_id = SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_MODULE_ID;
    assert(SparkHiddenTransportValidateSparkHostRdmaEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestHiddenTransportSparkHostRdmaPreflight(void)
{
    SparkHiddenTransportEndpoint endpoint;
    const char *infiniband_path;

    infiniband_path = "build/test_hidden_transport_infiniband";
    rmdir(infiniband_path);

    SparkTestInitializeSparkHostRdmaEndpoint(&endpoint);
    assert(SparkHiddenTransportSparkHostRdmaVerbsPreflight(
        &endpoint,
        infiniband_path) == SPARK_STATUS_ROUTE_NOT_FOUND);

    assert(mkdir(infiniband_path, 0777) == 0);
    assert(SparkHiddenTransportSparkHostRdmaVerbsPreflight(
        &endpoint,
        infiniband_path) == SPARK_STATUS_OK);

    endpoint.transport_module_id = "spark.hidden_transport.not_spark_host_rdma.v1";
    assert(SparkHiddenTransportSparkHostRdmaVerbsPreflight(
        &endpoint,
        infiniband_path) == SPARK_STATUS_INVALID_ARGUMENT);

    rmdir(infiniband_path);
}

int main(void)
{
    SparkTestHiddenTransportValidatesEndpointAndPacket();
    SparkTestHiddenTransportValidatesSidebandPayload();
    SparkTestHiddenTransportRejectsMissingNativeBatch();
    SparkTestHiddenTransportUsesNativeBatchSubmission();
    SparkTestHiddenTransportCompletionQueuePreservesOrder();
    SparkTestHiddenTransportPersistentRingBackend();
    SparkTestHiddenTransportPersistentRingAccountsSidebandBytes();
    SparkTestHiddenTransportRejectsInvalidInterface();
    SparkTestHiddenTransportLoadsProductionModule();
    SparkTestHiddenTransportOpensZeroCopyInterfaceFromHostPlan();
    SparkTestHiddenTransportValidatesSparkHostRdmaEndpoint();
    SparkTestHiddenTransportSparkHostRdmaPreflight();
    return 0;
}
