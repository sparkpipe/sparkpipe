#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_hidden_transport.h"

static SparkStatus TestHiddenTransportInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    assert(endpoint != 0);
    assert(transport_state != 0);
    *transport_state = (void *)endpoint;
    return SPARK_STATUS_OK;
}

static void TestHiddenTransportDestroy(void *transport_state)
{
    assert(transport_state != 0);
}

static SparkStatus TestHiddenTransportPostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    assert(transport_state != 0);
    assert(packet != 0);
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    assert(transport_state != 0);
    assert(packet != 0);
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

int main(void)
{
    SparkHiddenTransportEndpoint endpoint;
    SparkHiddenTransportInterface transport_interface;
    SparkHiddenTransportPacket packet;
    SparkHiddenTransportCompletion completion;
    SparkHiddenTransportSession *session;
    uint16_t hidden_payload[6144u * 2u];

    memset(&endpoint, 0, sizeof(endpoint));
    endpoint.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    endpoint.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES;
    endpoint.capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS;
    endpoint.hidden_dimension = 6144u;
    endpoint.bytes_per_sequence = 12288u;
    endpoint.max_active_sequence_count = 128u;
    endpoint.max_packet_bytes = 1572864u;
    endpoint.validated_latency_ns = 200000u;
    endpoint.transport_module_id =
        "spark.glm52.hidden_stage_transport.100g.persistent.v1";
    endpoint.route_name = "spark2_to_sparka";
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) == SPARK_STATUS_OK);
    endpoint.capability_flags &= ~SPARK_HIDDEN_TRANSPORT_CAP_NO_SHELL_TRANSPORT;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS;
    endpoint.max_packet_bytes = 12287u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_CAPACITY_EXCEEDED);
    endpoint.max_packet_bytes = 1572864u;
    endpoint.bytes_per_sequence = 12290u;
    assert(SparkHiddenTransportValidateEndpoint(&endpoint) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    endpoint.bytes_per_sequence = 12288u;

    memset(&packet, 0, sizeof(packet));
    packet.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    packet.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    packet.flags =
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    packet.active_sequence_count = 2u;
    packet.hidden_dimension = endpoint.hidden_dimension;
    packet.bytes_per_sequence = endpoint.bytes_per_sequence;
    packet.sequence_id = 7u;
    packet.token_index = 19u;
    packet.hidden_bf16 = hidden_payload;
    packet.cuda_stream = (void *)0x1;
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
    packet.hidden_dimension = endpoint.hidden_dimension;

    memset(&transport_interface, 0, sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS;
    transport_interface.initialize = TestHiddenTransportInitialize;
    transport_interface.destroy = TestHiddenTransportDestroy;
    transport_interface.post_receive = TestHiddenTransportPostReceive;
    transport_interface.send = TestHiddenTransportSend;
    transport_interface.poll = TestHiddenTransportPoll;
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
    assert(SparkHiddenTransportSend(session, &packet) == SPARK_STATUS_OK);
    assert(SparkHiddenTransportPostReceive(session, &packet) == SPARK_STATUS_OK);
    memset(&completion, 0, sizeof(completion));
    assert(SparkHiddenTransportPoll(session, &completion) == SPARK_STATUS_OK);
    assert(completion.abi_version == SPARK_HIDDEN_TRANSPORT_ABI_VERSION);
    assert(completion.descriptor_bytes == SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES);
    assert(completion.status == SPARK_STATUS_BUSY);
    SparkHiddenTransportClose(session);

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
    return 0;
}
