#include <string.h>

#include "sparkpipe/spark_hidden_transport.h"

typedef struct TestHiddenTransportModuleState
{
    uint32_t send_count;
    uint32_t receive_count;
} TestHiddenTransportModuleState;

static TestHiddenTransportModuleState g_test_hidden_transport_module_state;

static SparkStatus TestHiddenTransportModuleInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    if (endpoint == 0 || transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(&g_test_hidden_transport_module_state,0,
        sizeof(g_test_hidden_transport_module_state));
    *transport_state = &g_test_hidden_transport_module_state;
    return SPARK_STATUS_OK;
}

static void TestHiddenTransportModuleDestroy(
    void *transport_state)
{
    (void)transport_state;
}

static SparkStatus TestHiddenTransportModulePostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    TestHiddenTransportModuleState *state;

    if (transport_state == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (TestHiddenTransportModuleState *)transport_state;
    state->receive_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportModuleSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    TestHiddenTransportModuleState *state;

    if (transport_state == 0 || packet == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (TestHiddenTransportModuleState *)transport_state;
    state->send_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportModulePostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    TestHiddenTransportModuleState *state;

    if (transport_state == 0 || packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (TestHiddenTransportModuleState *)transport_state;
    state->receive_count += packet_count;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportModuleSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    TestHiddenTransportModuleState *state;

    if (transport_state == 0 || packets == 0 || packet_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    state = (TestHiddenTransportModuleState *)transport_state;
    state->send_count += packet_count;
    return SPARK_STATUS_OK;
}

static SparkStatus TestHiddenTransportModulePoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    if (transport_state == 0 || completion == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    completion->status = SPARK_STATUS_BUSY;
    return SPARK_STATUS_OK;
}

const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface transport_interface;

    memset(&transport_interface,0,sizeof(transport_interface));
    transport_interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    transport_interface.descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    transport_interface.capability_flags =
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_PRODUCTION_CAPS;
    transport_interface.initialize = TestHiddenTransportModuleInitialize;
    transport_interface.destroy = TestHiddenTransportModuleDestroy;
    transport_interface.post_receive = TestHiddenTransportModulePostReceive;
    transport_interface.send = TestHiddenTransportModuleSend;
    transport_interface.poll = TestHiddenTransportModulePoll;
    transport_interface.post_receive_batch =
        TestHiddenTransportModulePostReceiveBatch;
    transport_interface.send_batch = TestHiddenTransportModuleSendBatch;
    return &transport_interface;
}
