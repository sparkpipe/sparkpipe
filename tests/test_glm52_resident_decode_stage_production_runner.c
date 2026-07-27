#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"

typedef struct SparkTestProductionRunnerState
{
    uint32_t admit_count;
    uint32_t submit_count;
    uint32_t admit_accept;
    SparkModelDriverFrame last_frame;
    SparkGlm52ResidentDecodeStageFrameContext last_frame_context;
} SparkTestProductionRunnerState;

static SparkTestProductionRunnerState TestState;

static SparkStatus SparkTestProductionRunnerAdmit(
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkTestProductionRunnerState *state;

    state = (SparkTestProductionRunnerState *)driver_instance;
    assert(state != 0);
    assert(request != 0);
    assert(decision != 0);
    state->admit_count += 1u;
    decision->descriptor_bytes =
        ((uint32_t)sizeof(SparkModelDriverAdmissionDecision));
    decision->accepted = state->admit_accept;
    decision->driver_dispatch_slot = 11u;
    decision->driver_dispatch_generation = 99u;
    decision->driver_dispatch_cookie0 = 123u;
    decision->driver_dispatch_cookie1 = 456u;
    if ( state->admit_accept == 0u )
        decision->rejection_reason =
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTestProductionRunnerSubmit(
    void *driver_instance,
    SparkModelDriverFrame *frame)
{
    SparkTestProductionRunnerState *state;
    SparkGlm52ResidentDecodeStageFrameContext *frame_context;

    state = (SparkTestProductionRunnerState *)driver_instance;
    assert(state != 0);
    assert(frame != 0);
    assert(frame->user_context != 0);
    frame_context =
        (SparkGlm52ResidentDecodeStageFrameContext *)frame->user_context;
    state->submit_count += 1u;
    state->last_frame = *frame;
    state->last_frame_context = *frame_context;
    return SPARK_STATUS_OK;
}

static const SparkModelDriverInterface TestInterface =
{
    SPARK_MODEL_DRIVER_ABI_VERSION,
    ((uint32_t)sizeof(SparkModelDriverInterface)),
    0,
    0,
    0,
    SparkTestProductionRunnerAdmit,
    0
};

static const SparkModelDriverProgramProfile TestProfile =
{
    ((uint32_t)sizeof(SparkModelDriverProgramProfile)),
    0u,
    16u,
    128u,
    7u,
    128u,
    4096u,
    0u,
    1000000u,
    1u,
    1u,
    1u,
    0u,
    0u,
    1u,
    0u
};

static const SparkModelDriverProgramDescriptor TestProgram =
{
    7u,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_REQUIRED_PROGRAM_FLAGS,
    16u,
    0u,
    "glm52.ring.rank.production",
    &TestProfile,
    SparkTestProductionRunnerSubmit
};

static SparkGlm52KvBlockTableView TestKvTable;
static uint32_t TestPhysicalBlockIndices[4];
static uint32_t TestLaneBlockCounts[2];
static SparkHiddenTransportSession *TestInputSession =
    (SparkHiddenTransportSession *)(uintptr_t)0x1000u;
static SparkHiddenTransportSession *TestOutputSession =
    (SparkHiddenTransportSession *)(uintptr_t)0x2000u;

static void SparkTestProductionRunnerInitializeKvTable(void)
{
    memset(&TestKvTable, 0, sizeof(TestKvTable));
    TestPhysicalBlockIndices[0] = 3u;
    TestLaneBlockCounts[0] = 1u;
    TestKvTable.descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
    TestKvTable.block_token_count = 64u;
    TestKvTable.lane_count = 1u;
    TestKvTable.lane_stride = 4u;
    TestKvTable.lane_capacity = 2u;
    TestKvTable.physical_block_indices = TestPhysicalBlockIndices;
    TestKvTable.lane_physical_block_counts = TestLaneBlockCounts;
    TestKvTable.host_physical_block_indices = TestPhysicalBlockIndices;
    TestKvTable.host_lane_physical_block_counts = TestLaneBlockCounts;
}

static void SparkTestProductionRunnerInitializeRunner(
    SparkGlm52ResidentDecodeStageProductionRunner *runner,
    uint32_t flags)
{
    SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
    configuration.flags = flags;
    configuration.driver_interface = &TestInterface;
    configuration.driver_instance = &TestState;
    configuration.program = &TestProgram;
    configuration.execution_stream = (void *)(uintptr_t)0x7777u;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
        runner,
        &configuration) == SPARK_STATUS_OK);
}

static void SparkTestProductionRunnerInitializeDispatch(
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch *dispatch)
{
    memset(dispatch, 0, sizeof(*dispatch));
    dispatch->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    dispatch->descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES;
    dispatch->request_id = 1001u;
    dispatch->sequence_id = 2002u;
    dispatch->sequence_position = 33u;
    dispatch->active_sequence_count = 1u;
    dispatch->logical_lane_count = 1u;
    dispatch->rows_per_lane = 1u;
    dispatch->new_token_count = 1u;
    dispatch->pipeline_slot = 5u;
    dispatch->priority = 9u;
    dispatch->kv_block_table = &TestKvTable;
    dispatch->hidden_input_transport_session = TestInputSession;
    dispatch->hidden_output_transport_session = TestOutputSession;
    dispatch->hidden_input_packet.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    dispatch->hidden_input_packet.descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    dispatch->hidden_output_packet.abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    dispatch->hidden_output_packet.descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
}

static void SparkTestProductionRunnerSubmitsFrame(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;

    memset(&TestState, 0, sizeof(TestState));
    TestState.admit_accept = 1u;
    SparkTestProductionRunnerInitializeKvTable();
    SparkTestProductionRunnerInitializeRunner(
        &runner,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS);
    SparkTestProductionRunnerInitializeDispatch(&dispatch);
    dispatch.flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,
        &dispatch) == SPARK_STATUS_OK);
    assert(TestState.admit_count == 1u);
    assert(TestState.submit_count == 1u);
    assert((TestState.last_frame.flags &
        SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u);
    assert(TestState.last_frame.driver_dispatch_slot == 11u);
    assert(TestState.last_frame.program_id == 7u);
    assert(TestState.last_frame.scalar[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX] == 5u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_PRERECEIVED) != 0u);
    assert(TestState.last_frame_context.hidden_input_post_receive_function ==
        SparkHiddenTransportPostReceive);
    assert(TestState.last_frame_context.hidden_output_send_function ==
        SparkHiddenTransportSend);
}

static void SparkTestProductionRunnerRejectsMissingTransport(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;

    memset(&TestState, 0, sizeof(TestState));
    TestState.admit_accept = 1u;
    SparkTestProductionRunnerInitializeKvTable();
    SparkTestProductionRunnerInitializeRunner(
        &runner,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS);
    SparkTestProductionRunnerInitializeDispatch(&dispatch);
    dispatch.hidden_output_transport_session = 0;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,
        &dispatch) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(TestState.admit_count == 0u);
    assert(TestState.submit_count == 0u);
}

static void SparkTestProductionRunnerCarriesLayerMajorVerifyShape(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;
    uint32_t mtp_draft_token_budgets[16];

    memset(&TestState, 0, sizeof(TestState));
    TestState.admit_accept = 1u;
    SparkTestProductionRunnerInitializeKvTable();
    SparkTestProductionRunnerInitializeRunner(
        &runner,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS);
    SparkTestProductionRunnerInitializeDispatch(&dispatch);
    dispatch.logical_lane_count = 16u;
    dispatch.rows_per_lane = 7u;
    dispatch.active_sequence_count = 112u;
    dispatch.new_token_count = 7u;
    dispatch.mtp_draft_token_budgets = mtp_draft_token_budgets;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,&dispatch) == SPARK_STATUS_OK);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_LAYER_MAJOR_SPECULATIVE_VERIFY) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_BUDGETS) != 0u);
    assert(TestState.last_frame_context.mtp_draft_token_budgets ==
        mtp_draft_token_budgets);
    assert(TestState.last_frame_context.logical_lane_count == 16u);
    assert(TestState.last_frame_context.rows_per_lane == 7u);
    dispatch.active_sequence_count -= 1u;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,&dispatch) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestProductionRunnerCarriesParallelPrefillShape(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;

    memset(&TestState, 0, sizeof(TestState));
    TestState.admit_accept = 1u;
    SparkTestProductionRunnerInitializeKvTable();
    SparkTestProductionRunnerInitializeRunner(
        &runner,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS);
    SparkTestProductionRunnerInitializeDispatch(&dispatch);
    dispatch.flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL;
    dispatch.active_sequence_count = 64u;
    dispatch.logical_lane_count = 1u;
    dispatch.rows_per_lane = 64u;
    dispatch.new_token_count = 64u;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,
        &dispatch) == SPARK_STATUS_OK);
    assert((TestState.last_frame.flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_LAYER_MAJOR_SPECULATIVE_VERIFY) == 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_VIEW) == 0u);
    assert(TestState.last_frame_context.logical_lane_count == 1u);
    assert(TestState.last_frame_context.rows_per_lane == 64u);
}

static void SparkTestProductionRunnerMarksSingleRowPrefill(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;

    memset(&TestState, 0, sizeof(TestState));
    TestState.admit_accept = 1u;
    SparkTestProductionRunnerInitializeKvTable();
    SparkTestProductionRunnerInitializeRunner(
        &runner,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS);
    SparkTestProductionRunnerInitializeDispatch(&dispatch);
    dispatch.flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,
        &dispatch) == SPARK_STATUS_OK);
    assert((TestState.last_frame.flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME) != 0u);
    assert((TestState.last_frame_context.flags &
        SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_VIEW) == 0u);
}

static void SparkTestProductionRunnerRejectsAdmissionFailure(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;

    memset(&TestState, 0, sizeof(TestState));
    SparkTestProductionRunnerInitializeKvTable();
    SparkTestProductionRunnerInitializeRunner(
        &runner,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS);
    SparkTestProductionRunnerInitializeDispatch(&dispatch);
    assert(SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
        &runner,
        &dispatch) == SPARK_STATUS_BUSY);
    assert(TestState.admit_count == 1u);
    assert(TestState.submit_count == 0u);
}

static void SparkTestProductionRunnerRejectsSlowProgram(void)
{
    SparkGlm52ResidentDecodeStageProductionRunner runner;
    SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;
    SparkModelDriverProgramDescriptor slow_program;

    memset(&configuration, 0, sizeof(configuration));
    slow_program = TestProgram;
    slow_program.flags &= ~SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT;
    configuration.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
    configuration.flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS;
    configuration.driver_interface = &TestInterface;
    configuration.driver_instance = &TestState;
    configuration.program = &slow_program;
    assert(SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
        &runner,
        &configuration) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestProductionRunnerSubmitsFrame();
    SparkTestProductionRunnerCarriesLayerMajorVerifyShape();
    SparkTestProductionRunnerRejectsMissingTransport();
    SparkTestProductionRunnerCarriesParallelPrefillShape();
    SparkTestProductionRunnerMarksSingleRowPrefill();
    SparkTestProductionRunnerRejectsAdmissionFailure();
    SparkTestProductionRunnerRejectsSlowProgram();
    return 0;
}
