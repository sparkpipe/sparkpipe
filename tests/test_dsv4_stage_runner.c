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
	uint32_t invalid_admission;
	uint32_t expect_hidden_input;
	void *last_execution_stream;
	const uint32_t *prefill_token_ids;
} SparkDsv4RunnerTestState;

static SparkDsv4RunnerTestState TestState;

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
	if ( state->invalid_admission != 0u )
		decision->driver_dispatch_slot = 0u;
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
	state->last_execution_stream = frame->execution_stream;
    assert((context->flags &
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW) != 0u);
	assert((context->flags &
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER) != 0u);
	if ( state->expect_hidden_input != 0u )
	{
		assert((context->flags &
			SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER) != 0u);
		assert(context->hidden_input_bf16 != 0);
		assert(context->hidden_input_bytes ==
			SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
			SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES);
	}
    assert(context->hidden_output_bf16 != 0);
    assert(context->hidden_output_bytes ==
        SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
        SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES);
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
    uint16_t hidden_output[SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
    uint64_t position = 0u;
    uint64_t sequence = 1u;

    memset(&TestState, 0, sizeof(TestState));
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES;
    configuration.flags = SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION |
        SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY;
    configuration.stage_index = 0u;
    configuration.stage_count = 13u;
	configuration.max_active_sequence_count = 1u;
	configuration.max_input_row_count = 1u;
    configuration.driver_interface = &TestInterface;
    configuration.driver_instance = &TestState;
    configuration.program = &TestProgram;
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
    dispatch.hidden_output_bf16 = hidden_output;
    dispatch.hidden_output_bytes = sizeof(hidden_output);
    assert(SparkDsv4StageRunnerSubmit(&runner, &dispatch) ==
        SPARK_STATUS_OK);
    assert(TestState.admit_count == 1u);
    assert(TestState.submit_count == 1u);
    assert(TestState.last_frame_flags == SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL);
	assert(TestState.last_buffer_count == 1u);
	assert(TestState.last_execution_stream == configuration.execution_stream);
    assert(TestState.prefill_row_count == 1u);
    assert(TestState.prefill_token_ids == &token_id);
	TestState.invalid_admission = 1u;
	assert(SparkDsv4StageRunnerSubmit(&runner, &dispatch) ==
		SPARK_STATUS_ABI_MISMATCH);
	assert(TestState.submit_count == 1u);
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
	configuration.max_input_row_count = 1u;
    configuration.driver_interface = &TestInterface;
    configuration.driver_instance = &TestState;
    configuration.program = &TestProgram;
    assert(SparkDsv4StageRunnerInitialize(&runner, &configuration) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkDsv4RunnerTestIntermediateTokenRouting(void)
{
	SparkDsv4StageRunner runner;
	SparkDsv4StageRunnerConfiguration configuration;
	SparkDsv4StageRunnerDispatch dispatch;
	uint16_t hidden_input[SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
	uint16_t hidden_output[SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
	uint32_t token_id,lane;
	uint64_t position,sequence;
	memset(&TestState,0,sizeof(TestState));
	TestState.expect_hidden_input = 1u;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES;
	configuration.flags = SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION |
		SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY |
		SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY;
	configuration.stage_index = 1u;
	configuration.stage_count = 13u;
	configuration.max_active_sequence_count = 1u;
	configuration.max_input_row_count = 1u;
	configuration.driver_interface = &TestInterface;
	configuration.driver_instance = &TestState;
	configuration.program = &TestProgram;
	configuration.execution_stream = (void *)(uintptr_t)1u;
	assert(SparkDsv4StageRunnerInitialize(&runner,&configuration) == SPARK_STATUS_OK);
	token_id = 10397u;
	lane = 0u;
	position = 0u;
	sequence = 1u;
	memset(&dispatch,0,sizeof(dispatch));
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
	dispatch.hidden_input_bf16 = hidden_input;
	dispatch.hidden_input_bytes = sizeof(hidden_input);
	dispatch.hidden_output_bf16 = hidden_output;
	dispatch.hidden_output_bytes = sizeof(hidden_output);
	assert(SparkDsv4StageRunnerSubmit(&runner,&dispatch) == SPARK_STATUS_OK);
	assert(TestState.submit_count == 1u);
	assert(TestState.last_buffer_count == 1u);
	assert(TestState.prefill_token_ids == &token_id);
}

static void SparkDsv4RunnerTestRoundMajorPrefill(void)
{
	uint32_t round_major[6] = {7u,3u,11u,3u,11u,3u};
	uint32_t lane_major[6] = {7u,7u,3u,3u,3u,11u};
	assert(SparkDsv4ValidateRoundMajorPrefillRows(6u,3u,round_major) == SPARK_STATUS_OK);
	assert(SparkDsv4ValidateRoundMajorPrefillRows(6u,3u,lane_major) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkDsv4RoundMajorPrefillWaveRowCount(6u,3u,round_major,0u) == 3u);
	assert(SparkDsv4RoundMajorPrefillWaveRowCount(6u,3u,round_major,3u) == 2u);
	assert(SparkDsv4RoundMajorPrefillWaveRowCount(6u,3u,round_major,5u) == 1u);
}

int main(void)
{
	SparkDsv4RunnerTestPrefillMapping();
	SparkDsv4RunnerTestIntermediateRequiresInput();
	SparkDsv4RunnerTestIntermediateTokenRouting();
	SparkDsv4RunnerTestRoundMajorPrefill();
    return 0;
}
