#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_hidden_transport.h"

typedef struct TestTpDeviceCollectiveState
{
    SparkHiddenTransportPacket packet;
    uint32_t active;
    uint32_t sender;
} TestTpDeviceCollectiveState;

static SparkStatus TestTpDeviceCollectiveInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    TestTpDeviceCollectiveState *state;

    if (endpoint == 0 || transport_state == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state = (TestTpDeviceCollectiveState *)calloc(1u,sizeof(*state));
    if (state == 0)
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    state->sender = endpoint->source_rank_index == endpoint->local_rank_index;
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void TestTpDeviceCollectiveDestroy(void *transport_state)
{
    free(transport_state);
}

static SparkStatus TestTpDeviceCollectivePostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    TestTpDeviceCollectiveState *state =
        (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || packet == 0 || state->sender != 0u ||
        state->active != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state->packet = *packet;
    state->active = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectiveSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    TestTpDeviceCollectiveState *state =
        (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || packet == 0 || state->sender == 0u ||
        state->active != 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    state->packet = *packet;
    state->active = 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectivePoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    TestTpDeviceCollectiveState *state =
        (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || completion == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (state->active == 0u)
    {
        completion->status = SPARK_STATUS_BUSY;
        return SPARK_STATUS_OK;
    }
    completion->status = SPARK_STATUS_OK;
    completion->active_sequence_count = state->packet.active_sequence_count;
    completion->sequence_id = state->packet.sequence_id;
    completion->token_index = state->packet.token_index;
    completion->transfer_bytes = (uint64_t)state->packet.bytes_per_sequence *
        state->packet.active_sequence_count;
    state->active = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectivePostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    uint32_t packet_index;
    SparkStatus status;

    for (packet_index=0u; packet_index<packet_count; ++packet_index)
    {
        status = TestTpDeviceCollectivePostReceive(
            transport_state,&packets[packet_index]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectiveSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    uint32_t packet_index;
    SparkStatus status;

    for (packet_index=0u; packet_index<packet_count; ++packet_index)
    {
        status = TestTpDeviceCollectiveSend(
            transport_state,&packets[packet_index]);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return SPARK_STATUS_OK;
}

const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface interface;
    const char *host_mode;

    memset(&interface,0,sizeof(interface));
    host_mode = getenv("SPARK_TEST_TP_DEVICE_COLLECTIVE_HOST_MODE");
    interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    interface.capability_flags =
        host_mode != 0 && host_mode[0] == '1' ?
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS :
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_GPUDIRECT_RDMA_CAPS;
    interface.initialize = TestTpDeviceCollectiveInitialize;
    interface.destroy = TestTpDeviceCollectiveDestroy;
    interface.post_receive = TestTpDeviceCollectivePostReceive;
    interface.send = TestTpDeviceCollectiveSend;
    interface.poll = TestTpDeviceCollectivePoll;
    interface.post_receive_batch = TestTpDeviceCollectivePostReceiveBatch;
    interface.send_batch = TestTpDeviceCollectiveSendBatch;
    return &interface;
}
