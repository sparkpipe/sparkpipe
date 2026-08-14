#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_tp_device_collective.h"

#ifndef SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH
#define SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH \
    "build/test_modules/libtp_device_collective_module.so"
#endif

#define TEST_STEP_COUNT 2u
#define TEST_REDUCE_CREDIT_COUNT 4u
#define TEST_WAIT_MILLI 4000u

#define TEST_METRIC_REGISTER 1u
#define TEST_METRIC_RESERVE 2u
#define TEST_METRIC_SEND 3u
#define TEST_METRIC_RELEASE 4u
#define TEST_METRIC_CANCEL_SEND 5u
#define TEST_METRIC_CANCEL_RECEIVE 6u
#define TEST_METRIC_CLOSE_WITH_OWNER 7u
#define TEST_METRIC_DESTROY 8u
#define TEST_METRIC_GENERATION_REUSE_ERROR 9u
#define TEST_METRIC_SEND_BUSY 10u

typedef void (*TestResetFunction)(void);
typedef void (*TestCreditFunction)(uint32_t credit_index);
typedef void (*TestVoidFunction)(void);
typedef void (*TestToggleFunction)(uint32_t enabled);
typedef uint64_t (*TestMetricFunction)(uint32_t metric);

typedef struct TestTransportControls
{
    void *library;
    TestResetFunction reset;
    TestCreditFunction set_reserve_gate;
    TestVoidFunction release_reserve_gate;
    TestCreditFunction set_send_gate;
    TestVoidFunction release_send_gate;
    TestCreditFunction set_release_gate;
    TestVoidFunction release_release_gate;
    TestToggleFunction set_reverse_completion_order;
    TestToggleFunction set_host_memory_mode;
    TestMetricFunction metric;
} TestTransportControls;

typedef struct TestCompletionState
{
    atomic_uint count;
    atomic_int status;
    atomic_ullong ordinal;
    atomic_ullong generation;
    atomic_uint credit_index;
    atomic_uint slot_index;
} TestCompletionState;

typedef struct TestFailureHook
{
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint32_t observed;
    uint32_t release;
} TestFailureHook;

typedef struct TestFailureThread
{
    SparkTpDeviceCollective *collective;
    uint64_t ordinal;
    SparkStatus result;
} TestFailureThread;

typedef struct TestSubmissionThread
{
    SparkTpDeviceCollective *collective;
    SparkTpDeviceCollectiveSubmission submission;
    SparkStatus result;
} TestSubmissionThread;

typedef struct TestSubmissionClaimHook
{
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint32_t credit_index;
    uint64_t generation;
    uint32_t observed;
    uint32_t release;
} TestSubmissionClaimHook;

typedef struct TestBlockingCompletion
{
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    uint32_t entered;
    uint32_t release;
    TestCompletionState completion;
} TestBlockingCompletion;

typedef struct TestCombineState
{
    atomic_uint count;
} TestCombineState;

typedef struct TestRelayState
{
    TestCombineState combine;
    atomic_uint relay_count;
} TestRelayState;

static uint8_t TestSendBuffers[TEST_STEP_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestReceiveBuffers[TEST_STEP_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestSendTransportBuffers[TEST_STEP_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestReceiveTransportBuffers[TEST_STEP_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestLocalBuffers[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][64u];
static uint8_t TestFullBuffers[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][256u];
static SparkTpDeviceCollectiveCreditBinding TestBindings[
    TEST_STEP_COUNT * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
static SparkTpDeviceCollectiveCreditBinding TestReduceBindings[
    TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT];
static SparkTpDeviceCollectiveCreditBinding TestMappedReduceBindings[
    TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT];
static void TestSubmissionClaimed(
    void *context,
    uint32_t credit_index,
    uint64_t generation)
{
    TestSubmissionClaimHook *hook;

    hook = (TestSubmissionClaimHook *)context;
    if (hook == 0 || hook->credit_index != credit_index ||
        hook->generation != generation)
    {
        return;
    }
    assert(pthread_mutex_lock(&hook->mutex) == 0);
    hook->observed = 1u;
    assert(pthread_cond_broadcast(&hook->condition) == 0);
    while (hook->release == 0u)
    {
        assert(pthread_cond_wait(&hook->condition,&hook->mutex) == 0);
    }
    assert(pthread_mutex_unlock(&hook->mutex) == 0);
}

static uint64_t TestNowMilli(void)
{
    struct timespec now;

    assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
    return (uint64_t)now.tv_sec * 1000u +
        (uint64_t)now.tv_nsec / 1000000u;
}

static void TestInitializeBindings(void)
{
    uint32_t credit_index;
    uint32_t step_index;

    memset(TestBindings,0,sizeof(TestBindings));
    for (step_index = 0u; step_index < TEST_STEP_COUNT; ++step_index)
    {
        for (credit_index = 0u;
             credit_index < SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT;
             ++credit_index)
        {
            SparkTpDeviceCollectiveCreditBinding *binding;
            uint32_t binding_index;

            binding_index = step_index *
                SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT + credit_index;
            binding = &TestBindings[binding_index];
            binding->step_index = step_index;
            binding->credit_index = credit_index;
            binding->send_device = TestSendBuffers[step_index][credit_index];
            binding->receive_device =
                TestReceiveBuffers[step_index][credit_index];
            binding->send_transport = binding->send_device;
            binding->receive_transport = binding->receive_device;
        }
    }
    for (step_index = 0u; step_index < TEST_STEP_COUNT; ++step_index)
    {
        for (credit_index = 0u;
             credit_index < TEST_REDUCE_CREDIT_COUNT;
             ++credit_index)
        {
            SparkTpDeviceCollectiveCreditBinding *binding;
            uint32_t binding_index;

            binding_index = step_index * TEST_REDUCE_CREDIT_COUNT +
                credit_index;
            binding = &TestReduceBindings[binding_index];
            binding->step_index = step_index;
            binding->credit_index = credit_index;
            binding->send_device = TestSendBuffers[step_index][credit_index];
            binding->receive_device =
                TestReceiveBuffers[step_index][credit_index];
            binding->send_transport = binding->send_device;
            binding->receive_transport = binding->receive_device;
            TestMappedReduceBindings[binding_index] = *binding;
            TestMappedReduceBindings[binding_index].send_transport =
                TestSendTransportBuffers[step_index][credit_index];
            TestMappedReduceBindings[binding_index].receive_transport =
                TestReceiveTransportBuffers[step_index][credit_index];
        }
    }
}

static SparkStatus TestCombineBf16(
    void *context,
    void *destination_device,
    const void *source_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream)
{
    TestCombineState *state;
    uint16_t *destination;
    const uint16_t *source;
    uint32_t element;

    state = (TestCombineState *)context;
    destination = (uint16_t *)destination_device;
    source = (const uint16_t *)source_device;
    assert(state != 0 && destination != 0 && source != 0);
    assert(active_sequence_count != 0u && hidden_dimension != 0u);
    assert(active_sequence_count * hidden_dimension <= 8u);
    assert(cuda_stream != 0);
    for (element = 0u;
         element < active_sequence_count * hidden_dimension;
         ++element)
    {
        destination[element] = (uint16_t)(destination[element] +
            source[element]);
    }
    atomic_fetch_add_explicit(&state->count,1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestCombineRelayBf16(
    void *context,
    void *destination_device,
    const void *source_device,
    void *relay_device,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream)
{
    TestRelayState *state;
    SparkStatus status;
    uint32_t bytes;

    state = (TestRelayState *)context;
    assert(state != 0 && relay_device != 0);
    status = TestCombineBf16(&state->combine,destination_device,
        source_device,active_sequence_count,hidden_dimension,cuda_stream);
    if (status != SPARK_STATUS_OK)
        return status;
    bytes = active_sequence_count * hidden_dimension * sizeof(uint16_t);
    memcpy(relay_device,destination_device,bytes);
    atomic_fetch_add_explicit(&state->relay_count,1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestCombineU64Max(
    void *context,
    uint64_t *destination_device,
    const uint64_t *source_device,
    uint32_t element_count,
    void *cuda_stream)
{
    TestCombineState *state;
    uint64_t destination;
    uint64_t source;
    uint32_t element;

    state = (TestCombineState *)context;
    assert(state != 0 && destination_device != 0 && source_device != 0);
    assert(element_count != 0u && element_count <= 8u);
    assert(cuda_stream != 0);
    for (element=0u; element<element_count; element++)
    {
        memcpy(&destination,destination_device + element,sizeof(destination));
        memcpy(&source,source_device + element,sizeof(source));
        if (source > destination)
            memcpy(destination_device + element,&source,sizeof(source));
    }
    atomic_fetch_add_explicit(&state->count,1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestSignalU32(
    void *context,void *device_word,uint32_t value,void *cuda_stream)
{
    (void)context;
    assert(device_word != 0 && value != 0u && cuda_stream != 0);
    __atomic_store_n((uint32_t *)device_word,value,__ATOMIC_RELEASE);
    return SPARK_STATUS_OK;
}

static void *TestRequiredSymbol(void *library, const char *name)
{
    void *symbol;

    dlerror();
    symbol = dlsym(library,name);
    assert(symbol != 0);
    assert(dlerror() == 0);
    return symbol;
}

static void TestLoadControls(TestTransportControls *controls)
{
    memset(controls,0,sizeof(*controls));
    controls->library = dlopen(
        SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH,RTLD_NOW | RTLD_LOCAL);
    assert(controls->library != 0);
    controls->reset = (TestResetFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveReset");
    controls->set_reserve_gate = (TestCreditFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveSetReserveGate");
    controls->release_reserve_gate = (TestVoidFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveReleaseReserveGate");
    controls->set_send_gate = (TestCreditFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveSetSendGate");
    controls->release_send_gate = (TestVoidFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveReleaseSendGate");
    controls->set_release_gate = (TestCreditFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveSetReleaseGate");
    controls->release_release_gate = (TestVoidFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveReleaseReleaseGate");
    controls->set_reverse_completion_order =
        (TestToggleFunction)TestRequiredSymbol(controls->library,
            "TestTpDeviceCollectiveSetReverseCompletionOrder");
    controls->set_host_memory_mode =
        (TestToggleFunction)TestRequiredSymbol(controls->library,
            "TestTpDeviceCollectiveSetHostMemoryMode");
    controls->metric = (TestMetricFunction)TestRequiredSymbol(
        controls->library,"TestTpDeviceCollectiveMetric");
}

static void TestCompletionInitialize(TestCompletionState *state)
{
    atomic_init(&state->count,0u);
    atomic_init(&state->status,SPARK_STATUS_PENDING);
    atomic_init(&state->ordinal,UINT64_MAX);
    atomic_init(&state->generation,0u);
    atomic_init(&state->credit_index,UINT32_MAX);
    atomic_init(&state->slot_index,UINT32_MAX);
}

static void TestCompletionCallback(
    void *context,
    const SparkTpDeviceCollectiveCompletion *completion)
{
    TestCompletionState *state;

    state = (TestCompletionState *)context;
    assert(state != 0);
    assert(completion != 0);
    assert(completion->abi_version == SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION);
    assert(completion->descriptor_bytes == sizeof(*completion));
    atomic_store_explicit(&state->status,(int)completion->status,
        memory_order_relaxed);
    atomic_store_explicit(&state->ordinal,completion->ordinal,
        memory_order_relaxed);
    atomic_store_explicit(&state->generation,completion->generation,
        memory_order_relaxed);
    atomic_store_explicit(&state->credit_index,completion->credit_index,
        memory_order_relaxed);
    atomic_store_explicit(&state->slot_index,completion->slot_index,
        memory_order_relaxed);
    atomic_fetch_add_explicit(&state->count,1u,memory_order_release);
}

static void TestBlockingCompletionCallback(
    void *context,
    const SparkTpDeviceCollectiveCompletion *completion)
{
    TestBlockingCompletion *blocking;

    blocking = (TestBlockingCompletion *)context;
    assert(blocking != 0);
    assert(pthread_mutex_lock(&blocking->mutex) == 0);
    blocking->entered = 1u;
    assert(pthread_cond_broadcast(&blocking->condition) == 0);
    while (blocking->release == 0u)
    {
        assert(pthread_cond_wait(&blocking->condition,&blocking->mutex) == 0);
    }
    assert(pthread_mutex_unlock(&blocking->mutex) == 0);
    TestCompletionCallback(&blocking->completion,completion);
}

static void TestWaitForCompletion(TestCompletionState *state)
{
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    while (atomic_load_explicit(&state->count,memory_order_acquire) == 0u)
    {
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
    assert(atomic_load_explicit(&state->count,memory_order_acquire) == 1u);
}

static void TestWaitForAtomicCount(atomic_uint *count,uint32_t wanted)
{
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    while (atomic_load_explicit(count,memory_order_acquire) != wanted)
    {
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
}

static void TestWaitForPhase(
    SparkTpDeviceCollective *collective,
    uint64_t ordinal,
    uint32_t wanted_phase)
{
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    for (;;)
    {
        uint32_t failure_requested;
        uint32_t phase;

        if (SparkTpDeviceCollectiveOperationPhase(collective,ordinal,
                &phase,&failure_requested) == SPARK_STATUS_OK &&
            phase == wanted_phase)
        {
            return;
        }
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
}

static void TestWaitForMetric(
    const TestTransportControls *controls,
    uint32_t metric,
    uint64_t minimum)
{
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    while (controls->metric(metric) < minimum)
    {
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
}

static void TestConfigure(
    SparkTpDeviceCollectiveConfig *configuration,
    const SparkTpDeviceCollectiveDebugHooks *debug_hooks)
{
    static const char *hosts[4] = {
        "rank0","rank1","rank2","rank3"
    };

    memset(configuration,0,sizeof(*configuration));
    configuration->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    configuration->backend_kind =
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
    configuration->tp_degree = 4u;
    configuration->tp_rank = 1u;
    configuration->local_hidden_dimension = 4u;
    configuration->max_active_sequence_count = 8u;
    configuration->connect_timeout_milli = 1000u;
    configuration->operation_timeout_milli = TEST_WAIT_MILLI;
    configuration->control_port_base = 60000u;
    configuration->collective_identifier = 0x123456789abcdef0ull;
    configuration->backend_module_path =
        SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH;
    configuration->local_host = hosts[1];
    memcpy(configuration->rank_hosts,hosts,sizeof(hosts));
    configuration->credit_bindings = TestBindings;
    configuration->credit_binding_count =
        TEST_STEP_COUNT * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT;
    configuration->registration_cuda_stream = (void *)(uintptr_t)0x10000u;
    configuration->debug_hooks = debug_hooks;
}

static void TestCreate(
    TestTransportControls *controls,
    const SparkTpDeviceCollectiveDebugHooks *debug_hooks,
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveConfig configuration;

    controls->reset();
    TestConfigure(&configuration,debug_hooks);
    assert(SparkTpDeviceCollectiveCreate(&configuration,collective) ==
        SPARK_STATUS_OK);
    assert(collective->memory_mode ==
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE);
    assert(controls->metric(TEST_METRIC_REGISTER) ==
        TEST_STEP_COUNT * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT);
}

static void TestSubmitAndWait(
    SparkTpDeviceCollective *collective,
    uint32_t slot_index,
    uint64_t ordinal,
    TestCompletionState *completion)
{
    SparkTpDeviceCollectiveSubmission submission;
    uint32_t credit_index;

    TestCompletionInitialize(completion);
    credit_index = (uint32_t)(ordinal %
        SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT);
    memset(TestLocalBuffers[credit_index],(int)(ordinal + 1u),
        sizeof(TestLocalBuffers[credit_index]));
    memset(TestFullBuffers[credit_index],0,sizeof(TestFullBuffers[credit_index]));
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.slot_index = slot_index;
    submission.active_sequence_count = 2u;
    submission.ordinal = ordinal;
    submission.local_device = TestLocalBuffers[credit_index];
    submission.full_device = TestFullBuffers[credit_index];
    submission.cuda_stream = (void *)(uintptr_t)(0x20000u + credit_index);
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(completion);
    TestWaitForPhase(collective,ordinal,SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
}

static void TestBuildSubmission(
    SparkTpDeviceCollectiveSubmission *submission,
    TestCompletionState *completion,
    uint32_t slot_index,
    uint64_t ordinal,
    uint32_t rows)
{
    uint32_t credit_index;

    credit_index = (uint32_t)(ordinal %
        SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT);
    memset(TestLocalBuffers[credit_index],(int)(ordinal + 1u),
        sizeof(TestLocalBuffers[credit_index]));
    memset(TestFullBuffers[credit_index],0,sizeof(TestFullBuffers[credit_index]));
    memset(submission,0,sizeof(*submission));
    submission->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission->descriptor_bytes = sizeof(*submission);
    submission->slot_index = slot_index;
    submission->active_sequence_count = rows;
    submission->ordinal = ordinal;
    submission->local_device = TestLocalBuffers[credit_index];
    submission->full_device = TestFullBuffers[credit_index];
    submission->cuda_stream = (void *)(uintptr_t)(0x20000u + credit_index);
    submission->completion_function = TestCompletionCallback;
    submission->completion_context = completion;
}

static void TestSuccessfulOperation(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    TestCompletionState completion;
    uint32_t memory_mode;
    uint32_t byte_index;
    uint8_t buffer[16u];

    assert(SparkTpDeviceCollectiveProbeMemoryMode(
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT,
        SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH,&memory_mode) ==
        SPARK_STATUS_OK);
    assert(memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_DEVICE);
    TestCreate(controls,0,&collective);
    TestSubmitAndWait(&collective,17u,0u,&completion);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    assert(atomic_load_explicit(&completion.ordinal,memory_order_acquire) == 0u);
    assert(atomic_load_explicit(&completion.generation,memory_order_acquire) ==
        1u);
    assert(atomic_load_explicit(&completion.credit_index,memory_order_acquire) ==
        0u);
    assert(atomic_load_explicit(&completion.slot_index,memory_order_acquire) ==
        17u);
    for (byte_index = 0u; byte_index < 64u; ++byte_index)
    {
        assert(TestFullBuffers[0u][byte_index] == 1u);
    }
    assert(controls->metric(TEST_METRIC_RESERVE) == TEST_STEP_COUNT);
    assert(controls->metric(TEST_METRIC_SEND) == TEST_STEP_COUNT);
    assert(controls->metric(TEST_METRIC_RELEASE) == TEST_STEP_COUNT);
    assert(SparkTpDeviceCollectiveExchangeBf16(&collective,buffer,buffer,1u,
        4u,0u,(void *)(uintptr_t)1u) == SPARK_STATUS_UNSUPPORTED);
    assert(SparkTpDeviceCollectivePrepareReceiveBf16(&collective,buffer,1u,
        4u,0u,(void *)(uintptr_t)1u) == SPARK_STATUS_UNSUPPORTED);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
    assert(controls->metric(TEST_METRIC_DESTROY) == TEST_STEP_COUNT * 2u);
}

static void TestAllReduceSumAndBoundedCredits(
    TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveSubmission submission;
    TestCombineState combine;
    TestCompletionState completion;
    uint16_t *values;
    uint32_t element;

    controls->reset();
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestReduceBindings;
    configuration.credit_binding_count =
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_context = &combine;
    atomic_init(&combine.count,0u);
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    assert(collective.operation_kind ==
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
    assert(collective.credit_count == TEST_REDUCE_CREDIT_COUNT);
    assert(controls->metric(TEST_METRIC_REGISTER) ==
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT);
    values = (uint16_t *)TestFullBuffers[0u];
    for (element = 0u; element < 8u; ++element)
    {
        values[element] = 3u;
    }
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.slot_index = 2u;
    submission.active_sequence_count = 2u;
    submission.flags =
        SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x30000u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&combine.count,memory_order_acquire) ==
        TEST_STEP_COUNT);
    for (element = 0u; element < 8u; ++element)
    {
        assert(values[element] == 12u);
    }
    SparkTpDeviceCollectiveDestroy(&collective);
    configuration.combine_bf16_function = 0;
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestAllReduceU64Max(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveSubmission submission;
    TestCombineState combine;
    TestCompletionState completion;
    uint64_t values[2] = {UINT64_C(0x100000002),UINT64_C(0x300000004)};

    controls->reset();
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestReduceBindings;
    configuration.credit_binding_count =
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_u64_max_function = TestCombineU64Max;
    configuration.combine_context = &combine;
    atomic_init(&combine.count,0u);
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = 2u;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x30800u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitU64Max(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    assert(atomic_load_explicit(&combine.count,memory_order_acquire) ==
        TEST_STEP_COUNT);
    assert(values[0] == UINT64_C(0x100000002));
    assert(values[1] == UINT64_C(0x300000004));
    SparkTpDeviceCollectiveDestroy(&collective);
    configuration.combine_u64_max_function = 0;
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveSubmitU64Max(&collective,&submission) ==
        SPARK_STATUS_UNSUPPORTED);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestAdaptiveSplitRing(TestTransportControls *controls)
{
    static const char *direct_hosts[4] = {
        "direct0","direct1","direct2","direct3"
    };
    static const char *switch_hosts[4] = {
        "switch0","switch1","switch2","switch3"
    };
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveTopology sliced;
    SparkTpDeviceCollectiveTopology topology;
    SparkTpDeviceCollectiveSubmission submission;
    TestCombineState combine;
    TestCompletionState completion;
    uint16_t *values;
    uint32_t element;
    uint32_t rank;

    controls->reset();
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestReduceBindings;
    configuration.credit_binding_count =
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_context = &combine;
    memset(&topology,0,sizeof(topology));
    topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
    topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
    topology.rank_count = 4u;
    topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS;
    topology.split_ring_min_payload_bytes = 16u;
    topology.rail_count = 2u;
    topology.step_rail_indices[0] = 0u;
    topology.step_rail_indices[1] = 1u;
    topology.step_rail_indices[2] = 1u;
    for (rank=0u; rank<4u; rank++)
    {
        (void)snprintf(topology.rank_hosts[rank],
            sizeof(topology.rank_hosts[rank]),"rank%u",rank);
        (void)snprintf(topology.rail_rank_hosts[0][rank],
            sizeof(topology.rail_rank_hosts[0][rank]),"%s",direct_hosts[rank]);
        (void)snprintf(topology.rail_rank_hosts[1][rank],
            sizeof(topology.rail_rank_hosts[1][rank]),"%s",switch_hosts[rank]);
    }
    assert(SparkTpDeviceCollectiveApplyTopology(&topology,&configuration) ==
        SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveSliceTopology(&topology,2u,2u,&sliced) ==
        SPARK_STATUS_OK);
    assert(sliced.rank_count == 2u);
    assert(strcmp(sliced.rank_hosts[0],"rank2") == 0);
    assert(strcmp(sliced.rail_rank_hosts[0][1],"direct3") == 0);
    atomic_init(&combine.count,0u);
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    assert(collective.algorithm_mask ==
        SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS);
    assert(collective.split_ring_min_payload_bytes == 16u);
    assert(controls->metric(TEST_METRIC_REGISTER) ==
        SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_COUNT *
            TEST_REDUCE_CREDIT_COUNT);
    values = (uint16_t *)TestFullBuffers[0u];
    for (element=0u; element<8u; element++)
        values[element] = 3u;
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = 2u;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x30500u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    assert(atomic_load_explicit(&combine.count,memory_order_acquire) == 6u);
    assert(controls->metric(TEST_METRIC_SEND) == 12u);
    assert(controls->metric(TEST_METRIC_RELEASE) == 12u);
    for (element=0u; element<8u; element++)
        assert(values[element] == 12u);
    values = (uint16_t *)TestFullBuffers[1u];
    for (element=0u; element<4u; element++)
        values[element] = 3u;
    TestCompletionInitialize(&completion);
    submission.active_sequence_count = 1u;
    submission.ordinal = 1u;
    submission.local_device = values;
    submission.full_device = values;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,1u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    assert(atomic_load_explicit(&combine.count,memory_order_acquire) == 8u);
    assert(controls->metric(TEST_METRIC_SEND) == 14u);
    assert(controls->metric(TEST_METRIC_RELEASE) == 14u);
    for (element=0u; element<4u; element++)
        assert(values[element] == 12u);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
}

static void TestMappedHostStaging(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveSubmission submission;
    TestCombineState combine;
    TestCompletionState completion;
    uint16_t *values;
    uint32_t element;
    uint32_t memory_mode;

    controls->reset();
    controls->set_host_memory_mode(1u);
    assert(SparkTpDeviceCollectiveProbeMemoryMode(
        SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT,
        SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH,&memory_mode) ==
        SPARK_STATUS_OK);
    assert(memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST);
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestMappedReduceBindings;
    configuration.credit_binding_count =
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_context = &combine;
    atomic_init(&combine.count,0u);
    memset(TestSendTransportBuffers,0,sizeof(TestSendTransportBuffers));
    memset(TestReceiveTransportBuffers,0,sizeof(TestReceiveTransportBuffers));
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    assert(collective.memory_mode ==
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST);
    values = (uint16_t *)TestFullBuffers[0u];
    for (element = 0u; element < 8u; ++element)
    {
        values[element] = 3u;
    }
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.slot_index = 3u;
    submission.active_sequence_count = 2u;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x31000u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&combine.count,memory_order_acquire) ==
        TEST_STEP_COUNT);
    for (element = 0u; element < 8u; ++element)
    {
        assert(values[element] == 12u);
    }
    assert(((uint16_t *)TestSendTransportBuffers[0u][0u])[0] == 3u);
    assert(((uint16_t *)TestSendTransportBuffers[1u][0u])[0] == 6u);
    assert(((uint16_t *)TestReceiveBuffers[1u][0u])[0] == 6u);
    SparkTpDeviceCollectiveDestroy(&collective);
    controls->set_host_memory_mode(0u);
}

static void TestDirectBf16Relay(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveSubmission submission;
    TestCompletionState completion;
    TestRelayState relay;
    uint16_t *values;
    uint32_t element;

    controls->reset();
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestReduceBindings;
    configuration.credit_binding_count =
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_relay_bf16_function = TestCombineRelayBf16;
    configuration.combine_context = &relay;
    atomic_init(&relay.combine.count,0u);
    atomic_init(&relay.relay_count,0u);
    memset(TestSendBuffers,0,sizeof(TestSendBuffers));
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    values = (uint16_t *)TestFullBuffers[0u];
    for (element=0u; element<8u; element++)
        values[element] = 3u;
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.slot_index = 3u;
    submission.active_sequence_count = 2u;
    submission.flags =
        SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x31200u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&relay.combine.count,memory_order_acquire) ==
        TEST_STEP_COUNT);
    assert(atomic_load_explicit(&relay.relay_count,memory_order_acquire) ==
        1u);
    for (element=0u; element<8u; element++)
        assert(values[element] == 12u);
    assert(((uint16_t *)TestSendBuffers[1u][0u])[0] == 6u);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestExternalGraphOrder(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveSubmission submission;
    TestCombineState combine;
    TestCompletionState completion;
    uint16_t *values;
    uint32_t completion_word,producer_word;

    controls->reset();
    controls->set_release_gate(0u);
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestReduceBindings;
    configuration.credit_binding_count =
        TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_context = &combine;
    configuration.signal_u32_function = TestSignalU32;
    atomic_init(&combine.count,0u);
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    values = (uint16_t *)TestFullBuffers[0u];
    memset(values,0,16u);
    producer_word = 0u;
    completion_word = 0u;
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = 2u;
    submission.flags =
        SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION |
        SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_EXTERNAL_GRAPH_ORDER;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x31500u;
    submission.producer_ready_host = &producer_word;
    submission.completion_ready_host = &completion_word;
    submission.completion_ready_device = &completion_word;
    submission.producer_ready_value = 7u;
    submission.completion_ready_value = 7u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
    assert(controls->metric(TEST_METRIC_SEND) == 0u);
    __atomic_store_n(&producer_word,7u,__ATOMIC_RELEASE);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_CONSUME_BUILDING);
    TestWaitForAtomicCount(&combine.count,1u);
    assert(controls->metric(TEST_METRIC_SEND) == 1u);
    controls->release_release_gate();
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(__atomic_load_n(&completion_word,__ATOMIC_ACQUIRE) == 7u);
    assert(atomic_load_explicit(&combine.count,memory_order_acquire) ==
        TEST_STEP_COUNT);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestOutOfOrderCompletions(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission first_submission;
    SparkTpDeviceCollectiveSubmission second_submission;
    TestCompletionState first;
    TestCompletionState second;

    TestCreate(controls,0,&collective);
    controls->set_reverse_completion_order(1u);
    TestCompletionInitialize(&first);
    TestCompletionInitialize(&second);
    TestBuildSubmission(&first_submission,&first,1u,1u,8u);
    TestBuildSubmission(&second_submission,&second,2u,2u,1u);
    assert(SparkTpDeviceCollectiveSubmitBf16(
        &collective,&first_submission) == SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveSubmitBf16(
        &collective,&second_submission) == SPARK_STATUS_OK);
    TestWaitForCompletion(&first);
    TestWaitForCompletion(&second);
    TestWaitForPhase(&collective,1u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    TestWaitForPhase(&collective,2u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&first.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    assert(atomic_load_explicit(&second.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    assert(controls->metric(TEST_METRIC_SEND) == TEST_STEP_COUNT * 2u);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
}

static void TestRotatingGenerationReuse(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission stale_submission;
    TestCompletionState first;
    TestCompletionState second;

    TestCreate(controls,0,&collective);
    TestSubmitAndWait(&collective,0u,0u,&first);
    TestSubmitAndWait(&collective,64u,64u,&second);
    assert(atomic_load_explicit(&first.generation,memory_order_acquire) == 1u);
    assert(atomic_load_explicit(&second.generation,memory_order_acquire) == 2u);
    assert(controls->metric(TEST_METRIC_GENERATION_REUSE_ERROR) == 0u);
    TestBuildSubmission(&stale_submission,&first,0u,0u,1u);
    assert(SparkTpDeviceCollectiveSubmitBf16(
        &collective,&stale_submission) == SPARK_STATUS_BUSY);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
}

static void TestFailureObservedHook(
    void *context,
    uint32_t credit_index,
    uint64_t observed_state_word)
{
    TestFailureHook *hook;

    hook = (TestFailureHook *)context;
    assert(credit_index == 3u);
    assert((observed_state_word & 0xffu) ==
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
    assert(pthread_mutex_lock(&hook->mutex) == 0);
    hook->observed = 1u;
    assert(pthread_cond_broadcast(&hook->condition) == 0);
    while (hook->release == 0u)
    {
        assert(pthread_cond_wait(&hook->condition,&hook->mutex) == 0);
    }
    assert(pthread_mutex_unlock(&hook->mutex) == 0);
}

static void *TestRequestFailureMain(void *context)
{
    TestFailureThread *thread;

    thread = (TestFailureThread *)context;
    thread->result = SparkTpDeviceCollectiveRequestOperationFailure(
        thread->collective,thread->ordinal,SPARK_STATUS_IO_ERROR);
    return 0;
}

static void *TestSubmitMain(void *context)
{
    TestSubmissionThread *thread;

    thread = (TestSubmissionThread *)context;
    thread->result = SparkTpDeviceCollectiveSubmitBf16(
        thread->collective,&thread->submission);
    return 0;
}

static void TestActiveToSendBuildingFailureRace(
    TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission submission;
    SparkTpDeviceCollectiveDebugHooks debug_hooks;
    TestCompletionState completion;
    TestFailureHook hook;
    TestFailureThread failure_thread;
    pthread_t thread;

    memset(&hook,0,sizeof(hook));
    assert(pthread_mutex_init(&hook.mutex,0) == 0);
    assert(pthread_cond_init(&hook.condition,0) == 0);
    memset(&debug_hooks,0,sizeof(debug_hooks));
    debug_hooks.failure_observed_function = TestFailureObservedHook;
    debug_hooks.hook_context = &hook;
    TestCreate(controls,&debug_hooks,&collective);
    controls->set_reserve_gate(3u);
    controls->set_send_gate(3u);
    TestCompletionInitialize(&completion);
    TestBuildSubmission(&submission,&completion,3u,3u,2u);
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForPhase(&collective,3u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
    memset(&failure_thread,0,sizeof(failure_thread));
    failure_thread.collective = &collective;
    failure_thread.ordinal = 3u;
    assert(pthread_create(&thread,0,TestRequestFailureMain,&failure_thread) ==
        0);
    assert(pthread_mutex_lock(&hook.mutex) == 0);
    while (hook.observed == 0u)
    {
        assert(pthread_cond_wait(&hook.condition,&hook.mutex) == 0);
    }
    assert(pthread_mutex_unlock(&hook.mutex) == 0);

    controls->release_reserve_gate();
    TestWaitForPhase(&collective,3u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING);
    TestWaitForMetric(controls,TEST_METRIC_SEND_BUSY,1u);
    assert(pthread_mutex_lock(&hook.mutex) == 0);
    hook.release = 1u;
    assert(pthread_cond_broadcast(&hook.condition) == 0);
    assert(pthread_mutex_unlock(&hook.mutex) == 0);
    assert(pthread_join(thread,0) == 0);
    assert(failure_thread.result == SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,3u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    controls->release_send_gate();
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_IO_ERROR);
    assert(controls->metric(TEST_METRIC_SEND) == 0u);
    assert(controls->metric(TEST_METRIC_CANCEL_SEND) == TEST_STEP_COUNT);
    assert(controls->metric(TEST_METRIC_CANCEL_RECEIVE) == TEST_STEP_COUNT);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
    assert(pthread_cond_destroy(&hook.condition) == 0);
    assert(pthread_mutex_destroy(&hook.mutex) == 0);
}

static void TestTeardownWhileSendBuilding(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission submission;
    TestCompletionState completion;

    TestCreate(controls,0,&collective);
    controls->set_send_gate(4u);
    TestCompletionInitialize(&completion);
    TestBuildSubmission(&submission,&completion,4u,4u,2u);
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForPhase(&collective,4u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_SEND_BUILDING);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(atomic_load_explicit(&completion.count,memory_order_acquire) == 1u);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_IO_ERROR);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
    assert(controls->metric(TEST_METRIC_DESTROY) == TEST_STEP_COUNT * 2u);
}

static void TestStaleFailureCannotPoisonReusedCredit(
    TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission submission;
    TestCompletionState first;
    TestCompletionState second;
    uint32_t failure_requested;
    uint32_t phase;

    TestCreate(controls,0,&collective);
    TestSubmitAndWait(&collective,0u,0u,&first);
    controls->set_reserve_gate(0u);
    TestCompletionInitialize(&second);
    TestBuildSubmission(&submission,&second,64u,64u,2u);
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForPhase(&collective,64u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
    assert(SparkTpDeviceCollectiveRequestOperationFailure(
        &collective,0u,SPARK_STATUS_IO_ERROR) == SPARK_STATUS_NOT_FOUND);
    assert(SparkTpDeviceCollectiveOperationPhase(
        &collective,64u,&phase,&failure_requested) == SPARK_STATUS_OK);
    assert(phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
    assert(failure_requested == 0u);
    controls->release_reserve_gate();
    TestWaitForCompletion(&second);
    TestWaitForPhase(&collective,64u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&second.status,memory_order_acquire) ==
        SPARK_STATUS_OK);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
}

static void TestSubmitBuildingFailureIsNotOverwritten(
    TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveDebugHooks debug_hooks;
    TestCompletionState completion;
    TestSubmissionClaimHook hook;
    TestSubmissionThread submission_thread;
    pthread_t thread;
    uint32_t failure_requested;
    uint32_t phase;

    TestCompletionInitialize(&completion);
    memset(&hook,0,sizeof(hook));
    hook.credit_index = 5u;
    hook.generation = 1u;
    assert(pthread_mutex_init(&hook.mutex,0) == 0);
    assert(pthread_cond_init(&hook.condition,0) == 0);
    memset(&debug_hooks,0,sizeof(debug_hooks));
    debug_hooks.submission_claimed_function = TestSubmissionClaimed;
    debug_hooks.hook_context = &hook;
    TestCreate(controls,&debug_hooks,&collective);
    memset(&submission_thread,0,sizeof(submission_thread));
    submission_thread.collective = &collective;
    TestBuildSubmission(&submission_thread.submission,&completion,5u,5u,2u);
    assert(pthread_create(&thread,0,TestSubmitMain,&submission_thread) == 0);
    assert(pthread_mutex_lock(&hook.mutex) == 0);
    while (hook.observed == 0u)
    {
        assert(pthread_cond_wait(&hook.condition,&hook.mutex) == 0);
    }
    assert(pthread_mutex_unlock(&hook.mutex) == 0);
    assert(SparkTpDeviceCollectiveOperationPhase(
        &collective,5u,&phase,&failure_requested) == SPARK_STATUS_OK);
    assert(phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING);
    assert(failure_requested == 0u);
    assert(SparkTpDeviceCollectiveRequestOperationFailure(
        &collective,5u,SPARK_STATUS_IO_ERROR) == SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveRequestOperationFailure(
        &collective,5u,SPARK_STATUS_INTERNAL_ERROR) == SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveOperationPhase(
        &collective,5u,&phase,&failure_requested) == SPARK_STATUS_OK);
    assert(phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_SUBMIT_BUILDING);
    assert(failure_requested == 1u);
    assert(pthread_mutex_lock(&hook.mutex) == 0);
    hook.release = 1u;
    assert(pthread_cond_broadcast(&hook.condition) == 0);
    assert(pthread_mutex_unlock(&hook.mutex) == 0);
    assert(pthread_join(thread,0) == 0);
    assert(submission_thread.result == SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,5u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_IO_ERROR);
    assert(controls->metric(TEST_METRIC_SEND) == 0u);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
    assert(pthread_cond_destroy(&hook.condition) == 0);
    assert(pthread_mutex_destroy(&hook.mutex) == 0);
}

static void TestFailureLosesToCallbackClaim(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveSubmission submission;
    TestBlockingCompletion blocking;
    uint32_t failure_requested;
    uint32_t phase;

    TestCreate(controls,0,&collective);
    memset(&blocking,0,sizeof(blocking));
    assert(pthread_mutex_init(&blocking.mutex,0) == 0);
    assert(pthread_cond_init(&blocking.condition,0) == 0);
    TestCompletionInitialize(&blocking.completion);
    TestBuildSubmission(&submission,&blocking.completion,6u,6u,2u);
    submission.completion_function = TestBlockingCompletionCallback;
    submission.completion_context = &blocking;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    assert(pthread_mutex_lock(&blocking.mutex) == 0);
    while (blocking.entered == 0u)
    {
        assert(pthread_cond_wait(&blocking.condition,&blocking.mutex) == 0);
    }
    assert(pthread_mutex_unlock(&blocking.mutex) == 0);
    assert(SparkTpDeviceCollectiveOperationPhase(
        &collective,6u,&phase,&failure_requested) == SPARK_STATUS_OK);
    assert(phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_CALLBACK_CLAIMED);
    assert(failure_requested == 0u);
    assert(SparkTpDeviceCollectiveRequestOperationFailure(
        &collective,6u,SPARK_STATUS_IO_ERROR) == SPARK_STATUS_NOT_FOUND);
    assert(pthread_mutex_lock(&blocking.mutex) == 0);
    blocking.release = 1u;
    assert(pthread_cond_broadcast(&blocking.condition) == 0);
    assert(pthread_mutex_unlock(&blocking.mutex) == 0);
    TestWaitForCompletion(&blocking.completion);
    TestWaitForPhase(&collective,6u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(
        &blocking.completion.status,memory_order_acquire) == SPARK_STATUS_OK);
    assert(atomic_load_explicit(
        &blocking.completion.count,memory_order_acquire) == 1u);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
    assert(pthread_cond_destroy(&blocking.condition) == 0);
    assert(pthread_mutex_destroy(&blocking.mutex) == 0);
}

static void TestGraphFenceHost(void)
{
	SparkTpDeviceCollectiveGraphFence fence;
	volatile uint32_t completion,producer;
	completion = 7u;
	producer = 0u;
	memset(&fence,0,sizeof(fence));
	fence.producer_ready_host = &producer;
	fence.completion_ready_host = &completion;
	fence.ready_value = 7u;
	SparkTpDeviceCollectiveGraphFenceHost(&fence);
	assert(producer == 7u);
}

int main(void)
{
    TestTransportControls controls;

    TestInitializeBindings();
    TestLoadControls(&controls);
    TestSuccessfulOperation(&controls);
    TestAllReduceSumAndBoundedCredits(&controls);
    TestAllReduceU64Max(&controls);
    TestAdaptiveSplitRing(&controls);
    TestMappedHostStaging(&controls);
    TestDirectBf16Relay(&controls);
    TestExternalGraphOrder(&controls);
    TestOutOfOrderCompletions(&controls);
    TestRotatingGenerationReuse(&controls);
    TestActiveToSendBuildingFailureRace(&controls);
    TestTeardownWhileSendBuilding(&controls);
    TestStaleFailureCannotPoisonReusedCredit(&controls);
    TestSubmitBuildingFailureIsNotOverwritten(&controls);
    TestFailureLosesToCallbackClaim(&controls);
	TestGraphFenceHost();
    assert(dlclose(controls.library) == 0);
    puts("tp_device_collective: ok");
    return 0;
}
