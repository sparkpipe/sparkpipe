#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_runner.h"

typedef struct SparkDsv4RunnerTestState
{
    uint32_t admit_count;
    uint32_t submit_count;
    uint32_t last_frame_flags;
    uint32_t last_buffer_count;
    uint32_t prefill_row_count;
    const uint32_t *prefill_token_ids;
} SparkDsv4RunnerTestState;

static SparkDsv4RunnerTestState TestState;

static SparkStatus SparkDsv4RunnerTestSend(
    SparkHiddenTransportSession *transport_session,
    const SparkHiddenTransportPacket *packet)
{
    (void)transport_session;
    (void)packet;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDsv4RunnerTestAdmit(
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkDsv4RunnerTestState *state = driver_instance;

    assert(state != 0);
    assert(request != 0);
    assert(decision != 0);
    state->admit_count += 1u;
    memset(decision, 0, sizeof(*decision));
    decision->descriptor_bytes = sizeof(*decision);
    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkDsv4RunnerTestSubmit(
    void *driver_instance,
    SparkModelDriverFrame *frame)
{
    SparkDsv4RunnerTestState *state = driver_instance;
    SparkDsv4ResidentDecodeStageFrameContext *context;

    assert(state != 0);
    assert(frame != 0);
    assert(frame->user_context != 0);
    context = (SparkDsv4ResidentDecodeStageFrameContext *)
        frame->user_context;
    state->submit_count += 1u;
    state->last_frame_flags = frame->flags;
    state->last_buffer_count = frame->buffer_count;
    assert((context->flags &
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW) != 0u);
    assert((context->flags &
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u);
    assert(context->prefill_batch != 0);
    state->prefill_row_count = context->prefill_batch->row_count;
    state->prefill_token_ids = context->prefill_batch->token_ids;
    return SPARK_STATUS_OK;
}

static const SparkModelDriverInterface TestInterface =
{
    SPARK_MODEL_DRIVER_ABI_VERSION,
    (uint32_t)sizeof(SparkModelDriverInterface),
    0,
    0,
    0,
    SparkDsv4RunnerTestAdmit,
    0
};

static const SparkModelDriverProgramProfile TestProfile =
{
    (uint32_t)sizeof(SparkModelDriverProgramProfile),
    0u,
    1u,
    1u,
    1u,
    1u,
    128u,
    0u,
    0u,
    0u,
    0u,
    0u,
    0u,
    0u,
    0u,
    0u
};

static const SparkModelDriverProgramDescriptor TestProgram =
{
    1u,
    0u,
    1u,
    0u,
    "resident_decode",
    &TestProfile,
    SparkDsv4RunnerTestSubmit
};

static void SparkDsv4RunnerTestPrefillMapping(void)
{
    SparkDsv4StageRunner runner;
    SparkDsv4StageRunnerConfiguration configuration;
    SparkDsv4StageRunnerDispatch dispatch;
    uint32_t token_id = 10397u;
    uint32_t lane = 0u;
    uint64_t position = 0u;
    uint64_t sequence = 1u;

    memset(&TestState, 0, sizeof(TestState));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES;
    configuration.flags = SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION |
        SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
    configuration.stage_index = 0u;
    configuration.stage_count = 13u;
    configuration.max_active_sequence_count = 1u;
    configuration.driver_interface = &TestInterface;
    configuration.driver_instance = &TestState;
    configuration.program = &TestProgram;
    configuration.hidden_output_send_function = SparkDsv4RunnerTestSend;
    configuration.execution_stream = (void *)(uintptr_t)1u;
    assert(SparkDsv4StageRunnerInitialize(&runner, &configuration) ==
        SPARK_STATUS_OK);
    memset(&dispatch, 0, sizeof(dispatch));
    dispatch.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
    dispatch.descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_DISPATCH_BYTES;
    dispatch.flags = SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL;
    dispatch.request_id = 1u;
    dispatch.sequence_id = sequence;
    dispatch.active_sequence_count = 1u;
    dispatch.new_token_count = 1u;
    dispatch.row_count = 1u;
    dispatch.lane_count = 1u;
    dispatch.token_ids = &token_id;
    dispatch.row_lane_indices = &lane;
    dispatch.row_positions = &position;
    dispatch.row_sequence_ids = &sequence;
    dispatch.hidden_output_transport_session =
        (SparkHiddenTransportSession *)(uintptr_t)1u;
    assert(SparkDsv4StageRunnerSubmit(&runner, &dispatch) ==
        SPARK_STATUS_OK);
    assert(TestState.admit_count == 1u);
    assert(TestState.submit_count == 1u);
    assert(TestState.last_frame_flags == SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL);
    assert(TestState.last_buffer_count == 1u);
    assert(TestState.prefill_row_count == 1u);
    assert(TestState.prefill_token_ids == &token_id);
}

static void SparkDsv4RunnerTestIntermediateRequiresInput(void)
{
    SparkDsv4StageRunner runner;
    SparkDsv4StageRunnerConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES;
    configuration.stage_index = 1u;
    configuration.stage_count = 13u;
    configuration.max_active_sequence_count = 1u;
    configuration.driver_interface = &TestInterface;
    configuration.driver_instance = &TestState;
    configuration.program = &TestProgram;
    assert(SparkDsv4StageRunnerInitialize(&runner, &configuration) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkDsv4RunnerTestPrefillMapping();
    SparkDsv4RunnerTestIntermediateRequiresInput();
    return 0;
}
