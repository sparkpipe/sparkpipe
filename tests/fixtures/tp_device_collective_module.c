#include <pthread.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_hidden_transport.h"

#define TEST_TP_DEVICE_COLLECTIVE_MAX_STEPS 16u
#define TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT 64u
#define TEST_TP_DEVICE_COLLECTIVE_COMPLETION_COUNT 256u

#define TEST_TP_DEVICE_COLLECTIVE_METRIC_REGISTER 1u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_RESERVE 2u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_SEND 3u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_RELEASE 4u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_CANCEL_SEND 5u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_CANCEL_RECEIVE 6u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_CALLBACK_CLOSE_WITH_OWNER 7u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_DESTROY 8u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_GENERATION_REUSE_ERROR 9u
#define TEST_TP_DEVICE_COLLECTIVE_METRIC_SEND_BUSY 10u

typedef struct TestTpDeviceCollectiveState
{
    uint32_t sender;
    uint32_t step_index;
    uint32_t registered[TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint32_t reserved[TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint32_t activated[TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint64_t generations[TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    uint64_t returned_generations[TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    SparkHiddenTransportPacket receive_packets[
        TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
    SparkHiddenTransportCompletion completions[
        TEST_TP_DEVICE_COLLECTIVE_COMPLETION_COUNT];
    uint32_t completion_count;
} TestTpDeviceCollectiveState;

typedef struct TestTpDeviceCollectiveGlobal
{
    pthread_mutex_t mutex;
    uint32_t initialized;
    TestTpDeviceCollectiveState *senders[TEST_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    TestTpDeviceCollectiveState *receivers[TEST_TP_DEVICE_COLLECTIVE_MAX_STEPS];
    atomic_uint reserve_gate_credit_plus_one;
    atomic_uint send_gate_credit_plus_one;
    atomic_uint release_gate_credit_plus_one;
    atomic_uint reverse_completion_order;
    atomic_uint host_memory_mode;
    atomic_ullong metrics[11];
} TestTpDeviceCollectiveGlobal;

static TestTpDeviceCollectiveGlobal TestTpDeviceCollectiveGlobalState;
static pthread_once_t TestTpDeviceCollectiveGlobalOnce = PTHREAD_ONCE_INIT;

static void TestTpDeviceCollectiveInitializeGlobal(void)
{
    uint32_t index;

    memset(&TestTpDeviceCollectiveGlobalState,0,
        sizeof(TestTpDeviceCollectiveGlobalState));
    (void)pthread_mutex_init(&TestTpDeviceCollectiveGlobalState.mutex,0);
    atomic_init(
        &TestTpDeviceCollectiveGlobalState.reserve_gate_credit_plus_one,0u);
    atomic_init(
        &TestTpDeviceCollectiveGlobalState.send_gate_credit_plus_one,0u);
    atomic_init(
        &TestTpDeviceCollectiveGlobalState.release_gate_credit_plus_one,0u);
    atomic_init(
        &TestTpDeviceCollectiveGlobalState.reverse_completion_order,0u);
    atomic_init(&TestTpDeviceCollectiveGlobalState.host_memory_mode,0u);
    for (index = 0u; index < 11u; ++index)
    {
        atomic_init(&TestTpDeviceCollectiveGlobalState.metrics[index],0u);
    }
    TestTpDeviceCollectiveGlobalState.initialized = 1u;
}

static TestTpDeviceCollectiveGlobal *TestTpDeviceCollectiveGlobalGet(void)
{
    (void)pthread_once(&TestTpDeviceCollectiveGlobalOnce,
        TestTpDeviceCollectiveInitializeGlobal);
    return &TestTpDeviceCollectiveGlobalState;
}

void TestTpDeviceCollectiveReset(void)
{
    TestTpDeviceCollectiveGlobal *global;
    uint32_t index;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->reserve_gate_credit_plus_one,0u,
        memory_order_release);
    atomic_store_explicit(&global->send_gate_credit_plus_one,0u,
        memory_order_release);
    atomic_store_explicit(&global->release_gate_credit_plus_one,0u,
        memory_order_release);
    atomic_store_explicit(&global->reverse_completion_order,0u,
        memory_order_release);
    atomic_store_explicit(&global->host_memory_mode,0u,memory_order_release);
    for (index = 0u; index < 11u; ++index)
    {
        atomic_store_explicit(&global->metrics[index],0u,
            memory_order_release);
    }
}

void TestTpDeviceCollectiveSetReserveGate(uint32_t credit_index)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->reserve_gate_credit_plus_one,
        credit_index + 1u,memory_order_release);
}

void TestTpDeviceCollectiveReleaseReserveGate(void)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->reserve_gate_credit_plus_one,0u,
        memory_order_release);
}

void TestTpDeviceCollectiveSetSendGate(uint32_t credit_index)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->send_gate_credit_plus_one,
        credit_index + 1u,memory_order_release);
}

void TestTpDeviceCollectiveReleaseSendGate(void)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->send_gate_credit_plus_one,0u,
        memory_order_release);
}

void TestTpDeviceCollectiveSetReleaseGate(uint32_t credit_index)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->release_gate_credit_plus_one,
        credit_index + 1u,memory_order_release);
}

void TestTpDeviceCollectiveReleaseReleaseGate(void)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->release_gate_credit_plus_one,0u,
        memory_order_release);
}

void TestTpDeviceCollectiveSetReverseCompletionOrder(uint32_t enabled)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->reverse_completion_order,
        enabled != 0u ? 1u : 0u,memory_order_release);
}

void TestTpDeviceCollectiveSetHostMemoryMode(uint32_t enabled)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    atomic_store_explicit(&global->host_memory_mode,
        enabled != 0u ? 1u : 0u,memory_order_release);
}

uint64_t TestTpDeviceCollectiveMetric(uint32_t metric)
{
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    if (metric >= 11u)
    {
        return 0u;
    }
    return atomic_load_explicit(&global->metrics[metric],
        memory_order_acquire);
}

static uint32_t TestTpDeviceCollectiveStepFromRoute(const char *route_name)
{
    unsigned long long collective_identifier;
    uint32_t source_rank;
    uint32_t sink_rank;
    uint32_t step_index;

    collective_identifier = 0u;
    step_index = TEST_TP_DEVICE_COLLECTIVE_MAX_STEPS;
    source_rank = 0u;
    sink_rank = 0u;
    if (route_name == 0 || sscanf(route_name,"tp-device.%llx.%u.%u.%u",
            &collective_identifier,&step_index,&source_rank,&sink_rank) != 4)
    {
        return TEST_TP_DEVICE_COLLECTIVE_MAX_STEPS;
    }
    (void)collective_identifier;
    (void)source_rank;
    (void)sink_rank;
    return step_index;
}

static SparkStatus TestTpDeviceCollectiveInitialize(
    const SparkHiddenTransportEndpoint *endpoint,
    void **transport_state)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;

    if (endpoint == 0 || transport_state == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((endpoint->configuration_flags &
            SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_OPEN_TIMEOUT) == 0u ||
        endpoint->reserved0 != 1000u || endpoint->route_identifier == 0u)
        return SPARK_STATUS_VALIDATION_FAILED;
    state = (TestTpDeviceCollectiveState *)calloc(1u,sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    state->sender = endpoint->source_rank_index ==
        endpoint->local_rank_index;
    state->step_index = TestTpDeviceCollectiveStepFromRoute(
        endpoint->route_name);
    if (state->step_index >= TEST_TP_DEVICE_COLLECTIVE_MAX_STEPS)
    {
        free(state);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    (void)pthread_mutex_lock(&global->mutex);
    if (state->sender != 0u)
    {
        global->senders[state->step_index] = state;
    }
    else
    {
        global->receivers[state->step_index] = state;
    }
    (void)pthread_mutex_unlock(&global->mutex);
    *transport_state = state;
    return SPARK_STATUS_OK;
}

static void TestTpDeviceCollectiveDestroy(void *transport_state)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;
    uint32_t credit_index;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0)
    {
        return;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    for (credit_index = 0u;
         credit_index < TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT;
         ++credit_index)
    {
        if (state->reserved[credit_index] != 0u ||
            state->activated[credit_index] != 0u)
        {
            atomic_fetch_add_explicit(
                &global->metrics[
                    TEST_TP_DEVICE_COLLECTIVE_METRIC_CALLBACK_CLOSE_WITH_OWNER],
                1u,memory_order_relaxed);
        }
    }
    (void)pthread_mutex_lock(&global->mutex);
    if (state->sender != 0u &&
        global->senders[state->step_index] == state)
    {
        global->senders[state->step_index] = 0;
    }
    if (state->sender == 0u &&
        global->receivers[state->step_index] == state)
    {
        global->receivers[state->step_index] = 0;
    }
    (void)pthread_mutex_unlock(&global->mutex);
    atomic_fetch_add_explicit(
        &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_DESTROY],
        1u,memory_order_relaxed);
    free(state);
}

static SparkStatus TestTpDeviceCollectivePushCompletion(
    TestTpDeviceCollectiveState *state,
    const SparkHiddenTransportPacket *packet,
    SparkStatus completion_status)
{
    SparkHiddenTransportCompletion *completion;

    if (state == 0 || packet == 0 ||
        state->completion_count >= TEST_TP_DEVICE_COLLECTIVE_COMPLETION_COUNT)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    completion = &state->completions[state->completion_count];
    memset(completion,0,sizeof(*completion));
    completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
    completion->status = completion_status;
    completion->active_sequence_count = packet->active_sequence_count;
    completion->sequence_id = packet->sequence_id;
    completion->token_index = packet->token_index;
    completion->transfer_bytes = (uint64_t)packet->bytes_per_sequence *
        packet->active_sequence_count;
    state->completion_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectivePostReceive(
    void *transport_state,
    SparkHiddenTransportPacket *packet)
{
    (void)transport_state;
    (void)packet;
    return SPARK_STATUS_UNSUPPORTED;
}

static SparkStatus TestTpDeviceCollectiveSend(
    void *transport_state,
    const SparkHiddenTransportPacket *packet)
{
    (void)transport_state;
    (void)packet;
    return SPARK_STATUS_UNSUPPORTED;
}

static SparkStatus TestTpDeviceCollectivePoll(
    void *transport_state,
    SparkHiddenTransportCompletion *completion)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;
    uint32_t completion_index;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || completion == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    if (state->completion_count == 0u)
    {
        memset(completion,0,sizeof(*completion));
        completion->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
        completion->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_COMPLETION_BYTES;
        completion->status = SPARK_STATUS_BUSY;
        return SPARK_STATUS_OK;
    }
    completion_index = atomic_load_explicit(
        &global->reverse_completion_order,memory_order_acquire) != 0u ?
        state->completion_count - 1u : 0u;
    *completion = state->completions[completion_index];
    if (completion_index + 1u < state->completion_count)
    {
        memmove(&state->completions[completion_index],
            &state->completions[completion_index + 1u],
            (state->completion_count - completion_index - 1u) *
                sizeof(state->completions[0]));
    }
    state->completion_count -= 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectivePostReceiveBatch(
    void *transport_state,
    SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    (void)transport_state;
    (void)packets;
    (void)packet_count;
    return SPARK_STATUS_UNSUPPORTED;
}

static SparkStatus TestTpDeviceCollectiveSendBatch(
    void *transport_state,
    const SparkHiddenTransportPacket *packets,
    uint32_t packet_count)
{
    (void)transport_state;
    (void)packets;
    (void)packet_count;
    return SPARK_STATUS_UNSUPPORTED;
}

static SparkStatus TestTpDeviceCollectiveRegisterPersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    SparkHiddenTransportPacket *packet_template)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender != 0u || packet_template == 0 ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        state->registered[credit_index] != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    state->registered[credit_index] = 1u;
    state->receive_packets[credit_index] = *packet_template;
    atomic_fetch_add_explicit(
        &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_REGISTER],
        1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectivePersistentRemoteCreditReady(
    void *transport_state,
    uint32_t credit_index)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *receiver;
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender == 0u ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    receiver = global->receivers[state->step_index];
    return receiver != 0 && receiver->registered[credit_index] != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_BUSY;
}

static SparkStatus TestTpDeviceCollectiveReservePersistentSend(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender == 0u || packet == 0 ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->reserved[credit_index] != 0u)
    {
        return state->generations[credit_index] == generation ?
            SPARK_STATUS_OK : SPARK_STATUS_BUSY;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    if (atomic_load_explicit(&global->reserve_gate_credit_plus_one,
            memory_order_acquire) == credit_index + 1u)
    {
        return SPARK_STATUS_BUSY;
    }
    if (generation <= state->returned_generations[credit_index])
    {
        atomic_fetch_add_explicit(
            &global->metrics[
                TEST_TP_DEVICE_COLLECTIVE_METRIC_GENERATION_REUSE_ERROR],
            1u,memory_order_relaxed);
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    state->reserved[credit_index] = 1u;
    state->generations[credit_index] = generation;
    atomic_fetch_add_explicit(
        &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_RESERVE],
        1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectiveCancelPersistentSend(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender == 0u ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->reserved[credit_index] == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (state->generations[credit_index] != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    state->reserved[credit_index] = 0u;
    global = TestTpDeviceCollectiveGlobalGet();
    atomic_fetch_add_explicit(
        &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_CANCEL_SEND],
        1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectiveActivatePersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    SparkHiddenTransportPacket *packet)
{
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender != 0u || packet == 0 ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        state->registered[credit_index] == 0u || generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->activated[credit_index] != 0u)
    {
        return state->generations[credit_index] == generation ?
            SPARK_STATUS_OK : SPARK_STATUS_BUSY;
    }
    state->activated[credit_index] = 1u;
    state->generations[credit_index] = generation;
    state->receive_packets[credit_index] = *packet;
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectiveCancelPersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender != 0u ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        generation == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->activated[credit_index] == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (state->generations[credit_index] != generation)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    state->activated[credit_index] = 0u;
    global = TestTpDeviceCollectiveGlobalGet();
    atomic_fetch_add_explicit(
        &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_CANCEL_RECEIVE],
        1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestTpDeviceCollectiveSendPersistent(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    const SparkHiddenTransportPacket *packet)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *receiver;
    TestTpDeviceCollectiveState *state;
    SparkStatus status;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender == 0u || packet == 0 ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        state->reserved[credit_index] == 0u ||
        state->generations[credit_index] != generation)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    if (atomic_load_explicit(&global->send_gate_credit_plus_one,
            memory_order_acquire) == credit_index + 1u)
    {
        atomic_fetch_add_explicit(
            &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_SEND_BUSY],
            1u,memory_order_relaxed);
        return SPARK_STATUS_BUSY;
    }
    receiver = global->receivers[state->step_index];
    if (receiver == 0 || receiver->activated[credit_index] == 0u ||
        receiver->generations[credit_index] != generation)
    {
        return SPARK_STATUS_BUSY;
    }
    memcpy((void *)receiver->receive_packets[credit_index].hidden_bf16,
        packet->hidden_bf16,
        (size_t)packet->bytes_per_sequence * packet->active_sequence_count);
    status = TestTpDeviceCollectivePushCompletion(
        state,packet,SPARK_STATUS_OK);
    if (status == SPARK_STATUS_OK)
    {
        status = TestTpDeviceCollectivePushCompletion(receiver,
            &receiver->receive_packets[credit_index],SPARK_STATUS_OK);
    }
    if (status == SPARK_STATUS_OK)
    {
        atomic_fetch_add_explicit(
            &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_SEND],
            1u,memory_order_relaxed);
    }
    return status;
}

static SparkStatus TestTpDeviceCollectiveReleasePersistentReceive(
    void *transport_state,
    uint32_t credit_index,
    uint64_t generation,
    void *consumer_cuda_stream)
{
    TestTpDeviceCollectiveGlobal *global;
    TestTpDeviceCollectiveState *sender;
    TestTpDeviceCollectiveState *state;

    state = (TestTpDeviceCollectiveState *)transport_state;
    if (state == 0 || state->sender != 0u || consumer_cuda_stream == 0 ||
        credit_index >= TEST_TP_DEVICE_COLLECTIVE_CREDIT_COUNT ||
        state->activated[credit_index] == 0u ||
        state->generations[credit_index] != generation)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    global = TestTpDeviceCollectiveGlobalGet();
    if (atomic_load_explicit(&global->release_gate_credit_plus_one,
            memory_order_acquire) == credit_index + 1u)
    {
        return SPARK_STATUS_BUSY;
    }
    state->activated[credit_index] = 0u;
    state->returned_generations[credit_index] = generation;
    sender = global->senders[state->step_index];
    if (sender != 0 && sender->reserved[credit_index] != 0u &&
        sender->generations[credit_index] == generation)
    {
        sender->reserved[credit_index] = 0u;
        sender->returned_generations[credit_index] = generation;
    }
    atomic_fetch_add_explicit(
        &global->metrics[TEST_TP_DEVICE_COLLECTIVE_METRIC_RELEASE],
        1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

const SparkHiddenTransportInterface *SparkHiddenTransportGetInterface(void)
{
    static SparkHiddenTransportInterface interface;
    TestTpDeviceCollectiveGlobal *global;

    global = TestTpDeviceCollectiveGlobalGet();
    memset(&interface,0,sizeof(interface));
    interface.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    interface.descriptor_bytes = SPARK_HIDDEN_TRANSPORT_INTERFACE_BYTES;
    interface.capability_flags = atomic_load_explicit(
        &global->host_memory_mode,memory_order_acquire) != 0u ?
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_HOST_RDMA_CAPS :
        SPARK_HIDDEN_TRANSPORT_RECOMMENDED_SPARK_GPUDIRECT_RDMA_CAPS;
    interface.capability_flags |=
        SPARK_HIDDEN_TRANSPORT_CAP_PERSISTENT_RECEIVE_CREDITS;
    interface.initialize = TestTpDeviceCollectiveInitialize;
    interface.destroy = TestTpDeviceCollectiveDestroy;
    interface.post_receive = TestTpDeviceCollectivePostReceive;
    interface.send = TestTpDeviceCollectiveSend;
    interface.poll = TestTpDeviceCollectivePoll;
    interface.post_receive_batch = TestTpDeviceCollectivePostReceiveBatch;
    interface.send_batch = TestTpDeviceCollectiveSendBatch;
    interface.register_persistent_receive =
        TestTpDeviceCollectiveRegisterPersistentReceive;
    interface.persistent_remote_credit_ready =
        TestTpDeviceCollectivePersistentRemoteCreditReady;
    interface.reserve_persistent_send =
        TestTpDeviceCollectiveReservePersistentSend;
    interface.cancel_persistent_send =
        TestTpDeviceCollectiveCancelPersistentSend;
    interface.activate_persistent_receive =
        TestTpDeviceCollectiveActivatePersistentReceive;
    interface.cancel_persistent_receive =
        TestTpDeviceCollectiveCancelPersistentReceive;
    interface.send_persistent = TestTpDeviceCollectiveSendPersistent;
    interface.release_persistent_receive =
        TestTpDeviceCollectiveReleasePersistentReceive;
    return &interface;
}
