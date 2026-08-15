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
#define TEST_DIRECT_ROUTE_COUNT 3u
#define TEST_REDUCE_CREDIT_COUNT 4u
#define TEST_PROGRAM_CREDIT_COUNT 8u
#define TEST_PROGRAM_STEP_COUNT 6u
#define TEST_PROGRAM_STORAGE_BYTES 16384u
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

typedef struct TestProgramCancelThread
{
	SparkTpDeviceCollectiveProgram *program;
	atomic_uint entered;
	atomic_uint complete;
	SparkStatus result;
} TestProgramCancelThread;

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

typedef struct TestTp4CombineState
{
    TestCombineState recursive;
    atomic_uint direct_count;
	atomic_uint u64_direct_count;
} TestTp4CombineState;

typedef struct TestProgramFixture
{
    SparkTpDeviceCollectiveProgramStep steps[TEST_PROGRAM_STEP_COUNT];
	uint32_t producer_ready[TEST_PROGRAM_CREDIT_COUNT];
	uint32_t receive_ready[TEST_PROGRAM_CREDIT_COUNT];
	uint32_t result_ready[TEST_PROGRAM_CREDIT_COUNT];
	uint32_t send_reuse[TEST_PROGRAM_CREDIT_COUNT];
    uint32_t run_state[
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_STATE_WORD_COUNT];
    _Alignas(SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_STORAGE_ALIGNMENT)
        uint8_t storage[TEST_PROGRAM_STORAGE_BYTES];
} TestProgramFixture;

static uint8_t TestSendBuffers[TEST_DIRECT_ROUTE_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestReceiveBuffers[TEST_DIRECT_ROUTE_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestSendTransportBuffers[TEST_DIRECT_ROUTE_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestReceiveTransportBuffers[TEST_DIRECT_ROUTE_COUNT]
    [SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][128u];
static uint8_t TestLocalBuffers[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][64u];
static uint8_t TestFullBuffers[SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT][256u];
static SparkTpDeviceCollectiveCreditBinding TestBindings[
    TEST_STEP_COUNT * SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT];
static SparkTpDeviceCollectiveCreditBinding TestReduceBindings[
    TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT];
static SparkTpDeviceCollectiveCreditBinding TestMappedReduceBindings[
    TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT];
static SparkTpDeviceCollectiveCreditBinding TestDirectReduceBindings[
    TEST_DIRECT_ROUTE_COUNT * TEST_REDUCE_CREDIT_COUNT];
static SparkTpDeviceCollectiveCreditBinding TestProgramBindings[
	TEST_DIRECT_ROUTE_COUNT * TEST_PROGRAM_CREDIT_COUNT];
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
    for (step_index=0u; step_index<TEST_DIRECT_ROUTE_COUNT; step_index++)
        for (credit_index=0u; credit_index<TEST_REDUCE_CREDIT_COUNT;
            credit_index++)
        {
            SparkTpDeviceCollectiveCreditBinding *binding;
            uint32_t binding_index;

            binding_index = step_index * TEST_REDUCE_CREDIT_COUNT +
                credit_index;
            binding = &TestDirectReduceBindings[binding_index];
            binding->step_index = step_index;
            binding->credit_index = credit_index;
            binding->send_device = TestSendBuffers[step_index][credit_index];
            binding->receive_device =
                TestReceiveBuffers[step_index][credit_index];
            binding->send_transport = binding->send_device;
            binding->receive_transport = binding->receive_device;
        }
	for (step_index=0u; step_index<TEST_DIRECT_ROUTE_COUNT; step_index++)
		for (credit_index=0u; credit_index<TEST_PROGRAM_CREDIT_COUNT;
			credit_index++)
		{
			SparkTpDeviceCollectiveCreditBinding *binding;
			uint32_t binding_index;

			binding_index = step_index * TEST_PROGRAM_CREDIT_COUNT +
				credit_index;
			binding = &TestProgramBindings[binding_index];
			binding->step_index = step_index;
			binding->credit_index = credit_index;
			binding->send_device = TestSendBuffers[step_index][credit_index];
			binding->receive_device = TestReceiveBuffers[step_index][credit_index];
			binding->send_transport = binding->send_device;
			binding->receive_transport = binding->receive_device;
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
    assert(active_sequence_count * hidden_dimension <= 32u);
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

static SparkStatus TestCombineTp4Bf16(
    void *context,
    void *destination_device,
    const void *const rank_devices[4],
    uint32_t tp_rank,
    uint32_t active_sequence_count,
    uint32_t hidden_dimension,
    void *cuda_stream)
{
    TestTp4CombineState *state;
    uint16_t *destination;
    const uint16_t *rank0;
    const uint16_t *rank1;
    const uint16_t *rank2;
    const uint16_t *rank3;
    uint32_t element;

    state = (TestTp4CombineState *)context;
    destination = (uint16_t *)destination_device;
    rank0 = (const uint16_t *)rank_devices[0];
    rank1 = (const uint16_t *)rank_devices[1];
    rank2 = (const uint16_t *)rank_devices[2];
    rank3 = (const uint16_t *)rank_devices[3];
    assert(state != 0 && destination != 0 && tp_rank == 1u);
    assert(rank0 == (const uint16_t *)TestReceiveBuffers[0u][0u]);
    assert(rank1 == destination);
    assert(rank2 == (const uint16_t *)TestReceiveBuffers[2u][0u]);
    assert(rank3 == (const uint16_t *)TestReceiveBuffers[1u][0u]);
    assert(active_sequence_count * hidden_dimension <= 8u);
    assert(cuda_stream != 0);
    for (element=0u; element<active_sequence_count * hidden_dimension;
        element++)
    {
        uint16_t pair01;
        uint16_t pair23;

        pair01 = (uint16_t)(rank0[element] + rank1[element]);
        pair23 = (uint16_t)(rank2[element] + rank3[element]);
        destination[element] = (uint16_t)(pair01 + pair23);
    }
    atomic_fetch_add_explicit(&state->direct_count,1u,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

static SparkStatus TestCombineTp4U64Max(
	void *context,
	uint64_t *destination_device,
	const uint64_t *const rank_devices[4],
	uint32_t tp_rank,
	uint32_t element_count,
	void *cuda_stream)
{
	TestTp4CombineState *state;
	uint64_t maximum;
	uint32_t element,rank;

	state = (TestTp4CombineState *)context;
	assert(state != 0 && destination_device != 0 && rank_devices != 0);
	assert(tp_rank == 1u && element_count != 0u && element_count <= 8u);
	assert(cuda_stream != 0);
	for (element=0u; element<element_count; element++)
	{
		maximum = rank_devices[0][element];
		for (rank=1u; rank<4u; rank++)
			if ( rank_devices[rank][element] > maximum )
				maximum = rank_devices[rank][element];
		destination_device[element] = maximum;
	}
	atomic_fetch_add_explicit(&state->u64_direct_count,1u,
		memory_order_relaxed);
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
    topology.algorithm_mask =
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING |
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING;
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
    assert(collective.algorithm_mask == topology.algorithm_mask);
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

static void TestAdaptiveDirectAllToAll(TestTransportControls *controls)
{
    static const char *direct_hosts[4] = {
        "direct0","direct1","direct2","direct3"
    };
    static const char *switch_hosts[4] = {
        "switch0","switch1","switch2","switch3"
    };
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveConfig invalid;
    SparkTpDeviceCollectiveSubmission submission;
    SparkTpDeviceCollectiveTopology sliced;
    SparkTpDeviceCollectiveTopology topology;
    TestCompletionState completion;
    TestTp4CombineState combine;
    uint16_t *values;
    uint32_t element;
    uint32_t rank;
    uint32_t route_count;

    controls->reset();
    TestConfigure(&configuration,0);
    configuration.operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration.credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration.credit_bindings = TestDirectReduceBindings;
    configuration.credit_binding_count =
        TEST_DIRECT_ROUTE_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration.combine_bf16_function = TestCombineBf16;
    configuration.combine_tp4_bf16_function = TestCombineTp4Bf16;
    configuration.combine_context = &combine;
    memset(&topology,0,sizeof(topology));
    topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
    topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
    topology.rank_count = 4u;
    topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS;
    topology.direct_all_to_all_max_payload_bytes = 16u;
    topology.split_ring_min_payload_bytes = 32u;
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
    assert(SparkTpDeviceCollectiveSliceTopology(&topology,0u,4u,&sliced) ==
        SPARK_STATUS_OK);
    assert(sliced.direct_all_to_all_max_payload_bytes == 16u);
    assert(SparkTpDeviceCollectiveCreditBindingRouteCount(
        &configuration,&route_count) == SPARK_STATUS_OK);
    assert(route_count == TEST_DIRECT_ROUTE_COUNT);
    invalid = configuration;
    invalid.combine_tp4_bf16_function = 0;
    assert(SparkTpDeviceCollectiveCreate(&invalid,&collective) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    invalid = configuration;
    invalid.credit_binding_count = TEST_STEP_COUNT * TEST_REDUCE_CREDIT_COUNT;
    assert(SparkTpDeviceCollectiveCreate(&invalid,&collective) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    invalid = configuration;
    invalid.direct_all_to_all_max_payload_bytes =
        invalid.split_ring_min_payload_bytes;
    assert(SparkTpDeviceCollectiveCreate(&invalid,&collective) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    atomic_init(&combine.recursive.count,0u);
    atomic_init(&combine.direct_count,0u);
    memset(TestSendBuffers,0x5a,sizeof(TestSendBuffers));
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    assert(collective.direct_all_to_all_max_payload_bytes == 16u);
    assert(controls->metric(TEST_METRIC_REGISTER) ==
        TEST_DIRECT_ROUTE_COUNT * TEST_REDUCE_CREDIT_COUNT);
    values = (uint16_t *)TestFullBuffers[0u];
    for (element=0u; element<8u; element++)
        values[element] = 3u;
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = 2u;
    submission.flags =
        SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x30600u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&combine.direct_count,memory_order_acquire) ==
        1u);
    assert(atomic_load_explicit(&combine.recursive.count,memory_order_acquire) ==
        0u);
    assert(controls->metric(TEST_METRIC_SEND) == TEST_DIRECT_ROUTE_COUNT);
    assert(controls->metric(TEST_METRIC_RELEASE) == TEST_DIRECT_ROUTE_COUNT);
    for (element=0u; element<8u; element++)
    {
        assert(values[element] == 12u);
        assert(((uint16_t *)TestSendBuffers[0u][0u])[element] == 3u);
        assert(((uint16_t *)TestSendBuffers[1u][0u])[element] == 0x5a5au);
        assert(((uint16_t *)TestSendBuffers[2u][0u])[element] == 0x5a5au);
    }
    values = (uint16_t *)TestFullBuffers[1u];
    for (element=0u; element<12u; element++)
        values[element] = 3u;
    TestCompletionInitialize(&completion);
    submission.active_sequence_count = 3u;
    submission.ordinal = 1u;
    submission.local_device = values;
    submission.full_device = values;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_OK);
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,1u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    assert(atomic_load_explicit(&combine.direct_count,memory_order_acquire) ==
        1u);
    assert(atomic_load_explicit(&combine.recursive.count,memory_order_acquire) ==
        TEST_STEP_COUNT);
    assert(controls->metric(TEST_METRIC_SEND) ==
        TEST_DIRECT_ROUTE_COUNT + TEST_STEP_COUNT);
    assert(controls->metric(TEST_METRIC_RELEASE) ==
        TEST_DIRECT_ROUTE_COUNT + TEST_STEP_COUNT);
    for (element=0u; element<12u; element++)
    {
        assert(values[element] == 12u);
        assert(((uint16_t *)TestSendBuffers[0u][1u])[element] == 3u);
        assert(((uint16_t *)TestSendBuffers[1u][1u])[element] == 6u);
        assert(((uint16_t *)TestSendBuffers[2u][1u])[element] == 0x5a5au);
    }
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
}

static void TestConfigureProducerLease(
    SparkTpDeviceCollectiveConfig *configuration,
    TestTp4CombineState *combine)
{
    static const char *direct_hosts[4] = {
        "direct0","direct1","direct2","direct3"
    };
    static const char *switch_hosts[4] = {
        "switch0","switch1","switch2","switch3"
    };
    static SparkTpDeviceCollectiveTopology topology;
    uint32_t rank;

    TestConfigure(configuration,0);
    configuration->operation_kind =
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
    configuration->credit_count = TEST_REDUCE_CREDIT_COUNT;
    configuration->credit_bindings = TestDirectReduceBindings;
    configuration->credit_binding_count =
        TEST_DIRECT_ROUTE_COUNT * TEST_REDUCE_CREDIT_COUNT;
    configuration->combine_bf16_function = TestCombineBf16;
    configuration->combine_tp4_bf16_function = TestCombineTp4Bf16;
	configuration->combine_tp4_u64_max_function = TestCombineTp4U64Max;
    configuration->combine_context = combine;
    memset(&topology,0,sizeof(topology));
    topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
    topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
    topology.rank_count = 4u;
    topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS;
    topology.direct_all_to_all_max_payload_bytes = 16u;
    topology.split_ring_min_payload_bytes = 32u;
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
    assert(SparkTpDeviceCollectiveApplyTopology(&topology,configuration) ==
        SPARK_STATUS_OK);
}

static SparkStatus TestPrepareProgramAt(
    SparkTpDeviceCollective *collective,
    TestProgramFixture *fixture,
	SparkTpDeviceCollectiveCompletionFunction completion_function,
	void *completion_context,
	uint32_t step_count,uint32_t final_u64,uint32_t rows,uint64_t base_ordinal,
    SparkTpDeviceCollectiveProgram *program)
{
    SparkTpDeviceCollectiveProgramConfig configuration;
    uint64_t storage_bytes;
    uint32_t credit_index;
    uint32_t step_index;

    assert(step_count != 0u && step_count <= TEST_PROGRAM_STEP_COUNT);
	assert(final_u64 <= 1u);
    memset(fixture,0,sizeof(*fixture));
    for (step_index=0u; step_index<step_count; step_index++)
    {
        credit_index = step_index % TEST_REDUCE_CREDIT_COUNT;
        fixture->steps[step_index].abi_version =
            SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
        fixture->steps[step_index].descriptor_bytes =
            sizeof(fixture->steps[step_index]);
        fixture->steps[step_index].slot_index = step_index;
        fixture->steps[step_index].active_sequence_count = rows;
        fixture->steps[step_index].operation_kind =
            SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
		if ( final_u64 != 0u && step_index + 1u == step_count )
			fixture->steps[step_index].operation_kind =
				SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64;
        fixture->steps[step_index].ordinal = base_ordinal + step_index;
        fixture->steps[step_index].local_device = TestFullBuffers[credit_index];
        fixture->steps[step_index].full_device = TestFullBuffers[credit_index];
        fixture->steps[step_index].cuda_stream =
            (void *)(uintptr_t)(0x50000u + credit_index);
    }
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    configuration.descriptor_bytes = sizeof(configuration);
    configuration.step_count = step_count;
    configuration.window_count = 2u;
    configuration.steps = fixture->steps;
    configuration.producer_ready_host = fixture->producer_ready;
    configuration.receive_ready_host = fixture->receive_ready;
    configuration.result_ready_host = fixture->result_ready;
    configuration.send_reuse_host = fixture->send_reuse;
    configuration.producer_ready_device = fixture->producer_ready;
    configuration.receive_ready_device = fixture->receive_ready;
    configuration.result_ready_device = fixture->result_ready;
    configuration.send_reuse_device = fixture->send_reuse;
    configuration.run_state_host = fixture->run_state;
    configuration.run_state_device = fixture->run_state;
    configuration.storage = fixture->storage;
    configuration.storage_bytes = sizeof(fixture->storage);
    configuration.terminal_completion_function = completion_function;
    configuration.terminal_completion_context = completion_context;
    assert(SparkTpDeviceCollectiveProgramStorageBytes(
        step_count,&storage_bytes) == SPARK_STATUS_OK);
    assert(storage_bytes <= sizeof(fixture->storage));
    return SparkTpDeviceCollectivePrepareProgram(
		collective,&configuration,program);
}

static void TestPrepareProgram(
	SparkTpDeviceCollective *collective,TestProgramFixture *fixture,
	TestCompletionState *completion,uint32_t step_count,uint32_t final_u64,
	SparkTpDeviceCollectiveProgram *program)
{
	assert(TestPrepareProgramAt(collective,fixture,TestCompletionCallback,
		completion,step_count,final_u64,2u,0u,program) == SPARK_STATUS_OK);
}

static void TestCreateProgramCollective(
    TestTransportControls *controls,
    TestTp4CombineState *combine,
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveConfig configuration;

    controls->reset();
    atomic_init(&combine->recursive.count,0u);
    atomic_init(&combine->direct_count,0u);
	atomic_init(&combine->u64_direct_count,0u);
    TestConfigureProducerLease(&configuration,combine);
    assert(SparkTpDeviceCollectiveCreate(&configuration,collective) ==
        SPARK_STATUS_OK);
}

static void TestCreateConcurrentProgramCollective(
	TestTransportControls *controls,TestTp4CombineState *combine,
	SparkTpDeviceCollective *collective)
{
	SparkTpDeviceCollectiveConfig configuration;

	controls->reset();
	atomic_init(&combine->recursive.count,0u);
	atomic_init(&combine->direct_count,0u);
	atomic_init(&combine->u64_direct_count,0u);
	TestConfigureProducerLease(&configuration,combine);
	configuration.credit_count = TEST_PROGRAM_CREDIT_COUNT;
	configuration.credit_bindings = TestProgramBindings;
	configuration.credit_binding_count =
		TEST_DIRECT_ROUTE_COUNT * TEST_PROGRAM_CREDIT_COUNT;
	assert(SparkTpDeviceCollectiveCreate(&configuration,collective) ==
		SPARK_STATUS_OK);
}

static void TestWaitForProgramWord(uint32_t *word,uint32_t generation)
{
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    while (__atomic_load_n(word,__ATOMIC_ACQUIRE) < generation)
    {
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
}

static void TestWaitForProgramDestroy(SparkTpDeviceCollectiveProgram *program)
{
    SparkStatus status;
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    for (;;)
    {
        status = SparkTpDeviceCollectiveDestroyProgram(program);
        if (status == SPARK_STATUS_OK)
            return;
        assert(status == SPARK_STATUS_BUSY);
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
}

static void TestWaitForProgramRearm(
    SparkTpDeviceCollectiveProgram *program,uint64_t base_ordinal)
{
    SparkStatus status;
    uint64_t deadline;

    deadline = TestNowMilli() + TEST_WAIT_MILLI;
    for (;;)
    {
        status = SparkTpDeviceCollectiveRearmProgram(
            program,base_ordinal);
        if (status == SPARK_STATUS_OK)
            return;
        assert(status == SPARK_STATUS_BUSY);
        assert(TestNowMilli() < deadline);
        sched_yield();
    }
}

static void TestProgramPublishGraphSignals(TestProgramFixture *fixture)
{
    uint32_t credit_index;

    for (credit_index=0u; credit_index<TEST_REDUCE_CREDIT_COUNT;
        credit_index++)
    {
        __atomic_store_n(&fixture->producer_ready[credit_index],4u,
            __ATOMIC_RELEASE);
        __atomic_store_n(&fixture->result_ready[credit_index],4u,
            __ATOMIC_RELEASE);
    }
    __atomic_store_n(&fixture->run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
}

static void TestProgramPublishGraphDrained(TestProgramFixture *fixture)
{
    __atomic_store_n(&fixture->run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_GRAPH_DRAINED,
        __ATOMIC_RELEASE);
    __atomic_store_n(&fixture->run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_CONSUMER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_CONSUMER_GRAPH_DRAINED,
        __ATOMIC_RELEASE);
}

static void TestProgramResetGraphSignals(TestProgramFixture *fixture)
{
    memset(fixture->producer_ready,0,sizeof(fixture->producer_ready));
    memset(fixture->receive_ready,0,sizeof(fixture->receive_ready));
    memset(fixture->result_ready,0,sizeof(fixture->result_ready));
    memset(fixture->send_reuse,0,sizeof(fixture->send_reuse));
}

static void TestProgramPublishProducerStep(
    TestProgramFixture *fixture,uint32_t step_index)
{
    uint32_t credit_index;
    uint32_t generation;

    credit_index = step_index % TEST_REDUCE_CREDIT_COUNT;
    generation = step_index / TEST_REDUCE_CREDIT_COUNT + 1u;
    __atomic_store_n(&fixture->producer_ready[credit_index],generation,
        __ATOMIC_RELEASE);
}

static void TestProgramPublishResultStep(
    TestProgramFixture *fixture,uint32_t step_index)
{
    uint32_t credit_index;
    uint32_t generation;

    credit_index = step_index % TEST_REDUCE_CREDIT_COUNT;
    generation = step_index / TEST_REDUCE_CREDIT_COUNT + 1u;
    __atomic_store_n(&fixture->result_ready[credit_index],generation,
        __ATOMIC_RELEASE);
}

static void TestProgramWaitReceiveStep(
    TestProgramFixture *fixture,uint32_t step_index)
{
    uint32_t credit_index;
    uint32_t generation;

    credit_index = step_index % TEST_REDUCE_CREDIT_COUNT;
    generation = step_index / TEST_REDUCE_CREDIT_COUNT + 1u;
    TestWaitForProgramWord(&fixture->receive_ready[credit_index],generation);
}

static void TestProgramWaitCancellation(TestProgramFixture *fixture)
{
    uint32_t credit_index;
    uint32_t maximum_generation;

    for (credit_index=0u; credit_index<TEST_REDUCE_CREDIT_COUNT;
        credit_index++)
    {
        maximum_generation = credit_index < 2u ? 2u : 1u;
        TestWaitForProgramWord(
            &fixture->receive_ready[credit_index],maximum_generation);
        TestWaitForProgramWord(
            &fixture->result_ready[credit_index],maximum_generation);
        TestWaitForProgramWord(
            &fixture->send_reuse[credit_index],maximum_generation);
    }
}

static void TestProducerLeaseAlgorithmSpans(
    SparkTpDeviceCollective *collective)
{
    SparkTpDeviceCollectiveProducerLease recursive;
    SparkTpDeviceCollectiveProducerLease split;

    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        collective,1u,3u,&recursive) == SPARK_STATUS_OK);
    assert(recursive.algorithm ==
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING);
    assert(recursive.span_count == 1u && recursive.source_bytes == 24u);
    assert(recursive.spans[0].destination_device ==
        TestSendBuffers[0u][1u]);
    assert(recursive.spans[0].source_offset_bytes == 0u);
    assert(recursive.spans[0].byte_count == 24u);
    assert(SparkTpDeviceCollectiveCancelBf16ProducerLease(
        collective,&recursive) == SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        collective,2u,4u,&split) == SPARK_STATUS_OK);
    assert(split.algorithm ==
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING);
    assert(split.span_count == 2u && split.source_bytes == 32u);
    assert(split.spans[0].destination_device == TestSendBuffers[0u][2u]);
    assert(split.spans[1].destination_device == TestSendBuffers[2u][2u]);
    assert(split.spans[0].byte_count == 4u);
    assert(split.spans[1].byte_count == 4u);
    assert(split.spans[0].source_offset_bytes < split.source_bytes);
    assert(split.spans[1].source_offset_bytes < split.source_bytes);
    assert(split.spans[0].source_offset_bytes !=
        split.spans[1].source_offset_bytes);
    assert(SparkTpDeviceCollectiveCancelBf16ProducerLease(
        collective,&split) == SPARK_STATUS_OK);
}

static void TestProducerLeaseLifecycle(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveProducerLease lease;
    SparkTpDeviceCollectiveProducerLease next;
    SparkTpDeviceCollectiveProducerLease mutated;
    SparkTpDeviceCollectiveSubmission submission;
    TestCompletionState completion;
    TestTp4CombineState combine;
    uint16_t *values;
    uint32_t element;

    controls->reset();
    atomic_init(&combine.recursive.count,0u);
    atomic_init(&combine.direct_count,0u);
    TestConfigureProducerLease(&configuration,&combine);
    memset(TestSendBuffers,0x5a,sizeof(TestSendBuffers));
    assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
        SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        &collective,0u,2u,&lease) == SPARK_STATUS_OK);
    assert(lease.credit_index == 0u && lease.generation == 1u);
    assert(lease.algorithm ==
        SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL);
    assert(lease.span_count == 1u && lease.source_bytes == 16u);
    assert(lease.spans[0].destination_device == TestSendBuffers[0u][0u]);
    assert(lease.spans[0].source_offset_bytes == 0u);
    assert(lease.spans[0].byte_count == 16u);
    assert(lease.producer_ready_event != 0 && lease.consumer_ready_event != 0);
    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        &collective,0u,2u,&next) == SPARK_STATUS_BUSY);
    values = (uint16_t *)TestFullBuffers[0u];
    for (element=0u; element<8u; element++)
    {
        values[element] = 3u;
        ((uint16_t *)lease.spans[0].destination_device)[element] = 5u;
    }
    TestCompletionInitialize(&completion);
    memset(&submission,0,sizeof(submission));
    submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    submission.descriptor_bytes = sizeof(submission);
    submission.active_sequence_count = 2u;
    submission.flags =
        SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
    submission.ordinal = 0u;
    submission.local_device = values;
    submission.full_device = values;
    submission.cuda_stream = (void *)(uintptr_t)0x31800u;
    submission.completion_function = TestCompletionCallback;
    submission.completion_context = &completion;
    assert(SparkTpDeviceCollectiveSubmitBf16(&collective,&submission) ==
        SPARK_STATUS_BUSY);
    assert(SparkTpDeviceCollectiveSubmitLeasedBf16(
        &collective,&lease,&submission) == SPARK_STATUS_VALIDATION_FAILED);
    mutated = lease;
    mutated.spans[0].byte_count -= 2u;
    assert(SparkTpDeviceCollectiveRecordBf16ProducerReady(
        &collective,&mutated,submission.cuda_stream) ==
        SPARK_STATUS_VALIDATION_FAILED);
    mutated = lease;
    mutated.ordinal += TEST_REDUCE_CREDIT_COUNT;
    assert(SparkTpDeviceCollectiveRecordBf16ProducerReady(
        &collective,&mutated,submission.cuda_stream) ==
        SPARK_STATUS_VALIDATION_FAILED);
    mutated = lease;
    mutated.generation += 1u;
    assert(SparkTpDeviceCollectiveRecordBf16ProducerReady(
        &collective,&mutated,submission.cuda_stream) ==
        SPARK_STATUS_NOT_FOUND);
    assert(SparkTpDeviceCollectiveRecordBf16ProducerReady(
        &collective,&lease,submission.cuda_stream) == SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveRecordBf16ProducerReady(
        &collective,&lease,submission.cuda_stream) == SPARK_STATUS_DUPLICATE);
    controls->set_reserve_gate(0u);
    assert(SparkTpDeviceCollectiveSubmitLeasedBf16(
        &collective,&lease,&submission) == SPARK_STATUS_OK);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        &collective,4u,2u,&next) == SPARK_STATUS_BUSY);
    for (element=0u; element<8u; element++)
        assert(((uint16_t *)TestSendBuffers[0u][0u])[element] == 5u);
    controls->release_reserve_gate();
    TestWaitForCompletion(&completion);
    TestWaitForPhase(&collective,0u,
        SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE);
    for (element=0u; element<8u; element++)
        assert(values[element] == 18u);
    assert(SparkTpDeviceCollectiveCancelBf16ProducerLease(
        &collective,&lease) == SPARK_STATUS_NOT_FOUND);
    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        &collective,4u,2u,&next) == SPARK_STATUS_OK);
    assert(next.producer_ready_event == lease.producer_ready_event);
    assert(next.consumer_ready_event == lease.consumer_ready_event);
    assert(SparkTpDeviceCollectiveCancelBf16ProducerLease(
        &collective,&next) == SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveRecordBf16ProducerReady(
        &collective,&lease,submission.cuda_stream) == SPARK_STATUS_NOT_FOUND);
    TestProducerLeaseAlgorithmSpans(&collective);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(controls->metric(TEST_METRIC_CLOSE_WITH_OWNER) == 0u);
}

static void TestProgramAdmissionFailureDrain(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveProgram program;
    TestCompletionState completion;
    TestProgramFixture fixture;
    TestTp4CombineState combine;

    TestCreateProgramCollective(controls,&combine,&collective);
    TestCompletionInitialize(&completion);
    TestPrepareProgram(&collective,&fixture,&completion,1u,0u,&program);
    assert(SparkTpDeviceCollectiveStartProgram(&program) ==
        SPARK_STATUS_OK);
    __atomic_store_n(&fixture.run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
    assert(SparkTpDeviceCollectiveRequestFailure(
        &collective,SPARK_STATUS_VALIDATION_FAILED) == SPARK_STATUS_OK);
    __atomic_store_n(&fixture.producer_ready[0u],1u,__ATOMIC_RELEASE);
    TestProgramPublishGraphDrained(&fixture);
    TestWaitForCompletion(&completion);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_VALIDATION_FAILED);
    TestWaitForProgramDestroy(&program);
    assert(controls->metric(TEST_METRIC_RESERVE) == 0u);
    assert(controls->metric(TEST_METRIC_SEND) == 0u);
    assert(controls->metric(TEST_METRIC_RELEASE) == 0u);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestProgramStartContention(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveProgram first;
    SparkTpDeviceCollectiveProgram second;
    TestCompletionState first_completion;
    TestCompletionState second_completion;
    TestProgramFixture first_fixture;
    TestProgramFixture second_fixture;
    TestTp4CombineState combine;

    TestCreateProgramCollective(controls,&combine,&collective);
    TestCompletionInitialize(&first_completion);
    TestCompletionInitialize(&second_completion);
    TestPrepareProgram(&collective,&first_fixture,&first_completion,1u,0u,
		&first);
    TestPrepareProgram(&collective,&second_fixture,&second_completion,1u,0u,
		&second);
    assert(SparkTpDeviceCollectiveStartProgram(&first) ==
        SPARK_STATUS_OK);
    assert(SparkTpDeviceCollectiveStartProgram(&second) ==
        SPARK_STATUS_BUSY);
	assert(SparkTpDeviceCollectiveCancelProgram(&second,
		SPARK_STATUS_VALIDATION_FAILED) == SPARK_STATUS_BUSY);
    __atomic_store_n(&first_fixture.run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
    __atomic_store_n(&first_fixture.producer_ready[0u],1u,__ATOMIC_RELEASE);
    __atomic_store_n(&first_fixture.result_ready[0u],1u,__ATOMIC_RELEASE);
    TestProgramPublishGraphDrained(&first_fixture);
    TestWaitForCompletion(&first_completion);
    TestWaitForProgramDestroy(&first);
    assert(atomic_load_explicit(&first_completion.status,
        memory_order_acquire) == SPARK_STATUS_OK);
    assert(atomic_load_explicit(&second_completion.count,
        memory_order_acquire) == 0u);
	assert(SparkTpDeviceCollectiveRearmProgram(&second,4u) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveStartProgram(&second) == SPARK_STATUS_OK);
	__atomic_store_n(&second_fixture.run_state[
			SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
		SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
	__atomic_store_n(&second_fixture.producer_ready[0u],1u,__ATOMIC_RELEASE);
	__atomic_store_n(&second_fixture.result_ready[0u],1u,__ATOMIC_RELEASE);
	TestProgramPublishGraphDrained(&second_fixture);
	TestWaitForCompletion(&second_completion);
	assert(atomic_load_explicit(&second_completion.status,
		memory_order_acquire) == SPARK_STATUS_OK);
	TestWaitForProgramDestroy(&second);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestConcurrentDisjointPrograms(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveProgram programs[4u];
	TestCompletionState completions[4u];
	TestProgramFixture fixtures[4u];
	TestTp4CombineState combine;
	uint32_t credit,index,step;

	TestCreateConcurrentProgramCollective(controls,&combine,&collective);
	for (index=0u; index<4u; index++)
	{
		TestCompletionInitialize(&completions[index]);
		assert(TestPrepareProgramAt(&collective,&fixtures[index],
			TestCompletionCallback,&completions[index],2u,0u,2u,2u * index,
			&programs[index]) == SPARK_STATUS_OK);
		assert(SparkTpDeviceCollectiveStartProgram(&programs[index]) ==
			SPARK_STATUS_OK);
		__atomic_store_n(&fixtures[index].run_state[
				SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
			SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
	}
	for (step=0u; step<2u; step++)
		for (index=0u; index<4u; index++)
	{
		credit = 2u * index + step;
		TestWaitForProgramWord(&fixtures[index].send_reuse[credit],1u);
		__atomic_store_n(&fixtures[index].producer_ready[credit],1u,
			__ATOMIC_RELEASE);
		__atomic_store_n(&fixtures[index].result_ready[credit],1u,
			__ATOMIC_RELEASE);
	}
	for (index=0u; index<4u; index++)
		TestProgramPublishGraphDrained(&fixtures[index]);
	for (index=0u; index<4u; index++)
	{
		TestWaitForCompletion(&completions[index]);
		assert(atomic_load_explicit(&completions[index].status,
			memory_order_acquire) == SPARK_STATUS_OK);
		assert(atomic_load_explicit(&completions[index].credit_index,
			memory_order_acquire) == 2u * index + 1u);
		TestWaitForProgramDestroy(&programs[index]);
	}
	assert(controls->metric(TEST_METRIC_SEND) == 24u);
	assert(controls->metric(TEST_METRIC_RELEASE) == 24u);
	SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestSplitProgramPhases(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveProgram program;
	SparkTpDeviceCollectiveProgramConsumerPhase phase;
	SparkTpDeviceCollectiveProgramConsumerStep consumer;
	SparkTpDeviceCollectiveProgramProducerStep producer;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestTp4CombineState combine;
	uint32_t index,span;

	TestCreateProgramCollective(controls,&combine,&collective);
	TestCompletionInitialize(&completion);
	assert(TestPrepareProgramAt(&collective,&fixture,TestCompletionCallback,
		&completion,1u,0u,4u,2u,&program) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramProducerStep(&program,0u,
		&producer) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramConsumerStep(&program,0u,
		&consumer) == SPARK_STATUS_OK);
	assert(producer.span_count == 2u && producer.generation == 1u &&
		producer.completion_generation == 6u);
	assert(consumer.algorithm ==
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING);
	assert(consumer.phase_count == 6u);
	for (index=0u; index<consumer.phase_count; index++)
	{
		assert(SparkTpDeviceCollectiveGetProgramConsumerPhase(&program,0u,
			index,&phase) == SPARK_STATUS_OK);
		assert(phase.generation == index + 1u);
		assert(phase.receive_span_count == 2u);
		assert(phase.next_send_span_count == (index < 5u ? 2u : 0u));
		for (span=0u; span<phase.receive_span_count; span++)
		{
			assert(phase.receive_spans[span].element_count != 0u);
			assert(phase.receive_spans[span].destination_offset_bytes +
				(uint64_t)phase.receive_spans[span].element_count * 2u <= 32u);
			assert(phase.receive_spans[span].operation == (index < 3u ?
				SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_SPAN_ADD_BF16 :
				SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_SPAN_COPY_BF16));
		}
	}
	assert(SparkTpDeviceCollectiveStartProgram(&program) == SPARK_STATUS_OK);
	__atomic_store_n(&fixture.run_state[
			SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
		SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
	TestWaitForProgramWord(&fixture.send_reuse[2u],1u);
	__atomic_store_n(&fixture.producer_ready[2u],1u,__ATOMIC_RELEASE);
	for (index=0u; index<consumer.phase_count; index++)
	{
		TestWaitForProgramWord(&fixture.receive_ready[2u],index + 1u);
		assert(controls->metric(TEST_METRIC_SEND) == 2u * (index + 1u));
		assert(controls->metric(TEST_METRIC_RELEASE) == 2u * index);
		__atomic_store_n(&fixture.result_ready[2u],index + 1u,
			__ATOMIC_RELEASE);
	}
	TestProgramPublishGraphDrained(&fixture);
	TestWaitForCompletion(&completion);
	TestWaitForProgramDestroy(&program);
	assert(controls->metric(TEST_METRIC_SEND) == 12u);
	assert(controls->metric(TEST_METRIC_RELEASE) == 12u);
	SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestRecursiveProgramPhases(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveProgram program;
	SparkTpDeviceCollectiveProgramConsumerStep consumer;
	SparkTpDeviceCollectiveProgramProducerStep producer;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestTp4CombineState combine;

	TestCreateProgramCollective(controls,&combine,&collective);
	TestCompletionInitialize(&completion);
	assert(TestPrepareProgramAt(&collective,&fixture,TestCompletionCallback,
		&completion,1u,0u,3u,1u,&program) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramProducerStep(&program,0u,
		&producer) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramConsumerStep(&program,0u,
		&consumer) == SPARK_STATUS_OK);
	assert(producer.span_count == 1u && producer.generation == 1u &&
		producer.completion_generation == TEST_STEP_COUNT);
	assert(consumer.algorithm ==
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING);
	assert(consumer.phase_count == TEST_STEP_COUNT);
	assert(SparkTpDeviceCollectiveStartProgram(&program) == SPARK_STATUS_OK);
	__atomic_store_n(&fixture.run_state[
			SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
		SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
	TestWaitForProgramWord(&fixture.send_reuse[1u],1u);
	__atomic_store_n(&fixture.producer_ready[1u],1u,__ATOMIC_RELEASE);
	TestWaitForProgramWord(&fixture.result_ready[1u],TEST_STEP_COUNT);
	TestProgramPublishGraphDrained(&fixture);
	TestWaitForCompletion(&completion);
	TestWaitForProgramDestroy(&program);
	assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
		SPARK_STATUS_OK);
	assert(controls->metric(TEST_METRIC_SEND) == TEST_STEP_COUNT);
	assert(controls->metric(TEST_METRIC_RELEASE) == TEST_STEP_COUNT);
	SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestB1024ProgramPreparation(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveProgram program;
	SparkTpDeviceCollectiveProgramConsumerPhase phase;
	SparkTpDeviceCollectiveProgramConsumerStep consumer;
	SparkTpDeviceCollectiveProgramProducerStep producer;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestTp4CombineState combine;
	uint32_t index,span;

	controls->reset();
	atomic_init(&combine.recursive.count,0u);
	atomic_init(&combine.direct_count,0u);
	atomic_init(&combine.u64_direct_count,0u);
	TestConfigureProducerLease(&configuration,&combine);
	configuration.local_hidden_dimension = 4096u;
	configuration.max_active_sequence_count = 1024u;
	assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
		SPARK_STATUS_OK);
	TestCompletionInitialize(&completion);
	assert(TestPrepareProgramAt(&collective,&fixture,TestCompletionCallback,
		&completion,1u,0u,1024u,0u,&program) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramProducerStep(&program,0u,
		&producer) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramConsumerStep(&program,0u,
		&consumer) == SPARK_STATUS_OK);
	assert(producer.completion_generation == 6u);
	assert(consumer.algorithm ==
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING);
	assert(consumer.phase_count == 6u && consumer.hidden_dimension == 4096u &&
		consumer.active_sequence_count == 1024u);
	for (index=0u; index<consumer.phase_count; index++)
	{
		assert(SparkTpDeviceCollectiveGetProgramConsumerPhase(&program,0u,
			index,&phase) == SPARK_STATUS_OK);
		assert(phase.receive_span_count == 2u);
		for (span=0u; span<phase.receive_span_count; span++)
			assert(phase.receive_spans[span].destination_offset_bytes +
				(uint64_t)phase.receive_spans[span].element_count * 2u <=
				UINT64_C(1024) * 4096u * 2u);
	}
	assert(SparkTpDeviceCollectiveDestroyProgram(&program) == SPARK_STATUS_OK);
	SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestProgramReceiveAliasContract(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveProgram program;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestTp4CombineState combine;
	void *original_transport;
	uint32_t original_flags;

	controls->reset();
	atomic_init(&combine.recursive.count,0u);
	atomic_init(&combine.direct_count,0u);
	atomic_init(&combine.u64_direct_count,0u);
	original_transport = TestDirectReduceBindings[0u].receive_transport;
	original_flags = TestDirectReduceBindings[0u].flags;
	TestDirectReduceBindings[0u].receive_transport =
		TestReceiveTransportBuffers[0u][0u];
	TestDirectReduceBindings[0u].flags = 0u;
	TestConfigureProducerLease(&configuration,&combine);
	assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
		SPARK_STATUS_OK);
	TestCompletionInitialize(&completion);
	assert(TestPrepareProgramAt(&collective,&fixture,TestCompletionCallback,
		&completion,1u,0u,2u,0u,&program) == SPARK_STATUS_UNSUPPORTED);
	assert(program.implementation == 0);
	SparkTpDeviceCollectiveDestroy(&collective);
	TestDirectReduceBindings[0u].flags =
		SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS;
	TestConfigureProducerLease(&configuration,&combine);
	assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
		SPARK_STATUS_OK);
	assert(TestPrepareProgramAt(&collective,&fixture,TestCompletionCallback,
		&completion,1u,0u,2u,0u,&program) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveDestroyProgram(&program) == SPARK_STATUS_OK);
	SparkTpDeviceCollectiveDestroy(&collective);
	TestDirectReduceBindings[0u].receive_transport = original_transport;
	TestDirectReduceBindings[0u].flags = original_flags;
}

static void TestProgramCallbackDrainBarrier(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveProgram program;
	TestBlockingCompletion blocking;
	TestProgramFixture fixture;
	TestTp4CombineState combine;

	TestCreateProgramCollective(controls,&combine,&collective);
	memset(&blocking,0,sizeof(blocking));
	assert(pthread_mutex_init(&blocking.mutex,0) == 0);
	assert(pthread_cond_init(&blocking.condition,0) == 0);
	TestCompletionInitialize(&blocking.completion);
	assert(TestPrepareProgramAt(&collective,&fixture,
		TestBlockingCompletionCallback,&blocking,1u,0u,2u,0u,&program) ==
		SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveStartProgram(&program) == SPARK_STATUS_OK);
	TestProgramPublishGraphSignals(&fixture);
	TestProgramPublishGraphDrained(&fixture);
	assert(pthread_mutex_lock(&blocking.mutex) == 0);
	while (blocking.entered == 0u)
		assert(pthread_cond_wait(&blocking.condition,&blocking.mutex) == 0);
	assert(pthread_mutex_unlock(&blocking.mutex) == 0);
	assert(SparkTpDeviceCollectiveDestroyProgram(&program) == SPARK_STATUS_BUSY);
	assert(SparkTpDeviceCollectiveRearmProgram(&program,4u) ==
		SPARK_STATUS_BUSY);
	assert(pthread_mutex_lock(&blocking.mutex) == 0);
	blocking.release = 1u;
	assert(pthread_cond_broadcast(&blocking.condition) == 0);
	assert(pthread_mutex_unlock(&blocking.mutex) == 0);
	TestWaitForCompletion(&blocking.completion);
	TestWaitForProgramRearm(&program,4u);
	assert(SparkTpDeviceCollectiveDestroyProgram(&program) == SPARK_STATUS_OK);
	SparkTpDeviceCollectiveDestroy(&collective);
	assert(pthread_cond_destroy(&blocking.condition) == 0);
	assert(pthread_mutex_destroy(&blocking.mutex) == 0);
}

static void TestProgramRearmReuse(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveProgram program;
    SparkTpDeviceCollectiveProgramProducerStep first;
    SparkTpDeviceCollectiveProgramProducerStep rebound;
    TestCompletionState completion;
    TestProgramFixture fixture;
    TestTp4CombineState combine;
    void *implementation;

    TestCreateProgramCollective(controls,&combine,&collective);
    TestCompletionInitialize(&completion);
    TestPrepareProgram(&collective,&fixture,&completion,
		TEST_PROGRAM_STEP_COUNT,0u,&program);
    implementation = program.implementation;
    assert(SparkTpDeviceCollectiveGetProgramProducerStep(
        &program,0u,&first) == SPARK_STATUS_OK);
    assert(first.ordinal == 0u && first.generation == 1u);
    assert(SparkTpDeviceCollectiveStartProgram(&program) ==
        SPARK_STATUS_OK);
    TestProgramPublishGraphSignals(&fixture);
    TestProgramPublishGraphDrained(&fixture);
    TestWaitForCompletion(&completion);
    TestWaitForProgramRearm(&program,8u);
    assert(program.implementation == implementation);
    assert(program.base_ordinal == 8u);
    assert(SparkTpDeviceCollectiveGetProgramProducerStep(
        &program,0u,&rebound) == SPARK_STATUS_OK);
    assert(rebound.ordinal == 8u && rebound.generation == first.generation);
    assert(rebound.producer_ready_device == first.producer_ready_device);
    assert(rebound.result_ready_device == first.result_ready_device);
    TestProgramResetGraphSignals(&fixture);
    TestCompletionInitialize(&completion);
    assert(SparkTpDeviceCollectiveStartProgram(&program) ==
        SPARK_STATUS_OK);
    TestProgramPublishGraphSignals(&fixture);
    TestProgramPublishGraphDrained(&fixture);
    TestWaitForCompletion(&completion);
    assert(atomic_load_explicit(&completion.ordinal,memory_order_acquire) ==
        13u);
    assert(atomic_load_explicit(&completion.generation,memory_order_acquire) ==
        4u);
    TestWaitForProgramDestroy(&program);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestProgramMixedOperationPreparedRebase(
	TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveProgram program;
	SparkTpDeviceCollectiveProgramConsumerStep consumer;
	SparkTpDeviceCollectiveProgramProducerStep producer;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestTp4CombineState combine;
	uint32_t span;

	TestCreateProgramCollective(controls,&combine,&collective);
	TestCompletionInitialize(&completion);
	TestPrepareProgram(&collective,&fixture,&completion,3u,1u,&program);
	assert(SparkTpDeviceCollectiveRearmProgram(&program,8u) ==
		SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramProducerStep(
		&program,2u,&producer) == SPARK_STATUS_OK);
	assert(SparkTpDeviceCollectiveGetProgramConsumerStep(
		&program,2u,&consumer) == SPARK_STATUS_OK);
	assert(producer.operation_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
	assert(consumer.operation_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
	assert(producer.ordinal == 10u && consumer.ordinal == 10u);
	assert(producer.span_count == 1u);
	for (span=0u; span<producer.span_count; span++)
	{
		assert(producer.spans[span].source_offset_bytes == 0u);
		assert(producer.spans[span].byte_count == 2u * sizeof(uint64_t));
	}
	assert(consumer.active_sequence_count == 2u);
	assert(consumer.hidden_dimension == 1u);
	assert(SparkTpDeviceCollectiveStartProgram(&program) ==
		SPARK_STATUS_OK);
	TestProgramPublishGraphSignals(&fixture);
	TestProgramPublishGraphDrained(&fixture);
	TestWaitForCompletion(&completion);
	assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
		SPARK_STATUS_OK);
	assert(atomic_load_explicit(&completion.ordinal,memory_order_acquire) ==
		10u);
	assert(atomic_load_explicit(&completion.generation,memory_order_acquire) ==
		3u);
	TestWaitForProgramDestroy(&program);
	SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestProgramPartialWindowFailure(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveProgram program;
    TestCompletionState completion;
    TestProgramFixture fixture;
    TestTp4CombineState combine;
    uint32_t credit_index;

    TestCreateProgramCollective(controls,&combine,&collective);
    TestCompletionInitialize(&completion);
    TestPrepareProgram(&collective,&fixture,&completion,
		TEST_PROGRAM_STEP_COUNT,0u,&program);
    for (credit_index=0u; credit_index<TEST_REDUCE_CREDIT_COUNT; credit_index++)
        __atomic_store_n(&fixture.producer_ready[credit_index],2u,
            __ATOMIC_RELEASE);
    controls->set_release_gate(0u);
    assert(SparkTpDeviceCollectiveStartProgram(&program) ==
        SPARK_STATUS_OK);
    __atomic_store_n(&fixture.run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
    TestWaitForProgramWord(&fixture.receive_ready[0u],1u);
    TestWaitForProgramWord(&fixture.receive_ready[1u],1u);
    assert(SparkTpDeviceCollectiveRequestFailure(
        &collective,SPARK_STATUS_VALIDATION_FAILED) == SPARK_STATUS_OK);
    __atomic_store_n(&fixture.result_ready[1u],1u,__ATOMIC_RELEASE);
    assert(SparkTpDeviceCollectiveDestroyProgram(&program) ==
        SPARK_STATUS_BUSY);
    __atomic_store_n(&fixture.result_ready[0u],1u,__ATOMIC_RELEASE);
    TestProgramPublishGraphDrained(&fixture);
    controls->release_release_gate();
    TestWaitForCompletion(&completion);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_VALIDATION_FAILED);
    TestWaitForProgramDestroy(&program);
    assert(atomic_load_explicit(&completion.count,memory_order_acquire) == 1u);
    assert(controls->metric(TEST_METRIC_RELEASE) ==
        TEST_DIRECT_ROUTE_COUNT * 2u);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestProgramFailureAfterSubmitted(
    TestTransportControls *controls,uint32_t submitted_count)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveProgram program;
    TestCompletionState completion;
    TestProgramFixture fixture;
    TestTp4CombineState combine;
    uint32_t step_index;

    assert(submitted_count < TEST_PROGRAM_STEP_COUNT);
    TestCreateProgramCollective(controls,&combine,&collective);
    TestCompletionInitialize(&completion);
    TestPrepareProgram(&collective,&fixture,&completion,
		TEST_PROGRAM_STEP_COUNT,0u,&program);
    assert(SparkTpDeviceCollectiveStartProgram(&program) ==
        SPARK_STATUS_OK);
    __atomic_store_n(&fixture.run_state[
            SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
        SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
    for (step_index=0u; step_index<submitted_count; step_index++)
    {
        TestProgramPublishProducerStep(&fixture,step_index);
        TestProgramWaitReceiveStep(&fixture,step_index);
        if (step_index + 1u >= 2u && step_index + 1u < submitted_count)
            TestProgramPublishResultStep(&fixture,step_index - 1u);
    }
    assert(SparkTpDeviceCollectiveRequestFailure(
        &collective,SPARK_STATUS_VALIDATION_FAILED) == SPARK_STATUS_OK);
    TestProgramPublishProducerStep(&fixture,submitted_count);
    for (step_index=0u; step_index<submitted_count; step_index++)
        TestProgramPublishResultStep(&fixture,step_index);
    TestProgramWaitCancellation(&fixture);
    assert(SparkTpDeviceCollectiveDestroyProgram(&program) ==
        SPARK_STATUS_BUSY);
    TestProgramPublishGraphDrained(&fixture);
    TestWaitForCompletion(&completion);
    TestWaitForProgramDestroy(&program);
    assert(atomic_load_explicit(&completion.count,memory_order_acquire) == 1u);
    assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
        SPARK_STATUS_VALIDATION_FAILED);
    SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestProgramOperationFailureWithoutResultSignal(
	TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveProgram program;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestTp4CombineState combine;

	TestCreateProgramCollective(controls,&combine,&collective);
	TestCompletionInitialize(&completion);
	TestPrepareProgram(&collective,&fixture,&completion,
		TEST_PROGRAM_STEP_COUNT,0u,&program);
	controls->set_reserve_gate(0u);
	assert(SparkTpDeviceCollectiveStartProgram(&program) == SPARK_STATUS_OK);
	__atomic_store_n(&fixture.run_state[
			SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
		SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
	TestProgramPublishProducerStep(&fixture,0u);
	TestWaitForPhase(&collective,0u,SPARK_TP_DEVICE_COLLECTIVE_PHASE_ACTIVE);
	assert(SparkTpDeviceCollectiveRequestOperationFailure(
		&collective,0u,SPARK_STATUS_VALIDATION_FAILED) == SPARK_STATUS_OK);
	controls->release_reserve_gate();
	TestProgramWaitCancellation(&fixture);
	assert(SparkTpDeviceCollectiveDestroyProgram(&program) ==
		SPARK_STATUS_BUSY);
	TestProgramPublishGraphDrained(&fixture);
	TestWaitForCompletion(&completion);
	assert(atomic_load_explicit(&completion.count,memory_order_acquire) == 1u);
	assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
		SPARK_STATUS_VALIDATION_FAILED);
	TestWaitForProgramDestroy(&program);
	SparkTpDeviceCollectiveDestroy(&collective);
}

static void TestMappedHostStaging(TestTransportControls *controls)
{
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollectiveProducerLease lease;
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
    assert(SparkTpDeviceCollectiveAcquireBf16ProducerLease(
        &collective,0u,2u,&lease) == SPARK_STATUS_UNSUPPORTED);
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

static void *TestProgramCancelMain(void *context)
{
	TestProgramCancelThread *thread;

	thread = (TestProgramCancelThread *)context;
	atomic_store_explicit(&thread->entered,1u,memory_order_release);
	thread->result = SparkTpDeviceCollectiveCancelProgram(thread->program,
		SPARK_STATUS_VALIDATION_FAILED);
	atomic_store_explicit(&thread->complete,1u,memory_order_release);
	return 0;
}

static void TestProgramActivationCancelRace(TestTransportControls *controls)
{
	SparkTpDeviceCollective collective;
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveDebugHooks debug_hooks;
	SparkTpDeviceCollectiveProgram program;
	TestCompletionState completion;
	TestProgramFixture fixture;
	TestProgramCancelThread cancel;
	TestSubmissionClaimHook hook;
	TestTp4CombineState combine;
	pthread_t thread;

	controls->reset();
	atomic_init(&combine.recursive.count,0u);
	atomic_init(&combine.direct_count,0u);
	atomic_init(&combine.u64_direct_count,0u);
	memset(&hook,0,sizeof(hook));
	hook.credit_index = 0u;
	hook.generation = 1u;
	assert(pthread_mutex_init(&hook.mutex,0) == 0);
	assert(pthread_cond_init(&hook.condition,0) == 0);
	memset(&debug_hooks,0,sizeof(debug_hooks));
	debug_hooks.submission_claimed_function = TestSubmissionClaimed;
	debug_hooks.hook_context = &hook;
	TestConfigureProducerLease(&configuration,&combine);
	configuration.debug_hooks = &debug_hooks;
	assert(SparkTpDeviceCollectiveCreate(&configuration,&collective) ==
		SPARK_STATUS_OK);
	TestCompletionInitialize(&completion);
	TestPrepareProgram(&collective,&fixture,&completion,1u,0u,&program);
	assert(SparkTpDeviceCollectiveStartProgram(&program) == SPARK_STATUS_OK);
	__atomic_store_n(&fixture.run_state[
			SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_PRODUCER_STATE_INDEX],
		SPARK_TP_DEVICE_COLLECTIVE_PROGRAM_RUN_ACTIVE,__ATOMIC_RELEASE);
	TestWaitForProgramWord(&fixture.send_reuse[0u],1u);
	__atomic_store_n(&fixture.producer_ready[0u],1u,__ATOMIC_RELEASE);
	assert(pthread_mutex_lock(&hook.mutex) == 0);
	while (hook.observed == 0u)
		assert(pthread_cond_wait(&hook.condition,&hook.mutex) == 0);
	assert(pthread_mutex_unlock(&hook.mutex) == 0);
	memset(&cancel,0,sizeof(cancel));
	cancel.program = &program;
	atomic_init(&cancel.entered,0u);
	atomic_init(&cancel.complete,0u);
	assert(pthread_create(&thread,0,TestProgramCancelMain,&cancel) == 0);
	while (atomic_load_explicit(&cancel.entered,memory_order_acquire) == 0u)
		sched_yield();
	assert(atomic_load_explicit(&cancel.complete,memory_order_acquire) == 0u);
	assert(pthread_mutex_lock(&hook.mutex) == 0);
	hook.release = 1u;
	assert(pthread_cond_broadcast(&hook.condition) == 0);
	assert(pthread_mutex_unlock(&hook.mutex) == 0);
	assert(pthread_join(thread,0) == 0);
	assert(cancel.result == SPARK_STATUS_OK);
	__atomic_store_n(&fixture.result_ready[0u],1u,__ATOMIC_RELEASE);
	TestProgramPublishGraphDrained(&fixture);
	TestWaitForCompletion(&completion);
	TestWaitForProgramDestroy(&program);
	assert(atomic_load_explicit(&completion.status,memory_order_acquire) ==
		SPARK_STATUS_VALIDATION_FAILED);
	assert(atomic_load_explicit(&completion.count,memory_order_acquire) == 1u);
	SparkTpDeviceCollectiveDestroy(&collective);
	assert(pthread_cond_destroy(&hook.condition) == 0);
	assert(pthread_mutex_destroy(&hook.mutex) == 0);
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

int main(void)
{
    TestTransportControls controls;

    TestInitializeBindings();
    TestLoadControls(&controls);
    TestSuccessfulOperation(&controls);
    TestAllReduceSumAndBoundedCredits(&controls);
    TestAllReduceU64Max(&controls);
    TestAdaptiveSplitRing(&controls);
    TestAdaptiveDirectAllToAll(&controls);
    TestProducerLeaseLifecycle(&controls);
    TestProgramAdmissionFailureDrain(&controls);
    TestProgramStartContention(&controls);
    TestConcurrentDisjointPrograms(&controls);
    TestSplitProgramPhases(&controls);
    TestB1024ProgramPreparation(&controls);
    TestRecursiveProgramPhases(&controls);
    TestProgramReceiveAliasContract(&controls);
    TestProgramCallbackDrainBarrier(&controls);
    TestProgramRearmReuse(&controls);
	TestProgramMixedOperationPreparedRebase(&controls);
    TestProgramPartialWindowFailure(&controls);
    TestProgramFailureAfterSubmitted(&controls,0u);
    TestProgramFailureAfterSubmitted(&controls,1u);
    TestProgramFailureAfterSubmitted(&controls,2u);
    TestProgramFailureAfterSubmitted(&controls,
		TEST_PROGRAM_STEP_COUNT - 1u);
	TestProgramOperationFailureWithoutResultSignal(&controls);
    TestProgramActivationCancelRace(&controls);
    TestMappedHostStaging(&controls);
    TestDirectBf16Relay(&controls);
    TestOutOfOrderCompletions(&controls);
    TestRotatingGenerationReuse(&controls);
    TestActiveToSendBuildingFailureRace(&controls);
    TestTeardownWhileSendBuilding(&controls);
    TestStaleFailureCannotPoisonReusedCredit(&controls);
    TestSubmitBuildingFailureIsNotOverwritten(&controls);
    TestFailureLosesToCallbackClaim(&controls);
    assert(dlclose(controls.library) == 0);
    puts("tp_device_collective: ok");
    return 0;
}
