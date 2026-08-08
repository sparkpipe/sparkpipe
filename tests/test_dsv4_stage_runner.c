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
	uint32_t prefill_emit_count;
	uint32_t invalid_admission;
	uint32_t expect_hidden_input;
	void *last_execution_stream;
	const uint32_t *prefill_token_ids;
	const uint32_t *prefill_emit_rows;
	const uint32_t *prefill_emit_lanes;
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
	state->prefill_emit_count = context->prefill_batch->emit_count;
    state->prefill_token_ids = context->prefill_batch->token_ids;
	state->prefill_emit_rows = context->prefill_batch->emit_row_indices;
	state->prefill_emit_lanes = context->prefill_batch->emit_lane_indices;
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
	dispatch.emit_count = 1u;
	dispatch.emit_row_indices = &lane;
	dispatch.emit_lane_indices = &lane;
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
	assert(TestState.prefill_emit_count == 1u);
    assert(TestState.prefill_token_ids == &token_id);
	assert(TestState.prefill_emit_rows == &lane);
	assert(TestState.prefill_emit_lanes == &lane);
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

static uint32_t SparkDsv4RunnerRingSum(const uint32_t *ring, uint32_t lane, uint64_t position)
{
	uint32_t column,sum = 0u;
	for (column=0u; column<4u; column++)
		sum += ring[lane * 4u + SparkDsv4AttentionWindowSlot(position,column,4u)];
	return(sum);
}

static void SparkDsv4RunnerTestCausalBulkWaves(void)
{
	uint32_t lanes[4] = {0u,1u,0u,1u},values[4] = {4u,40u,5u,50u};
	uint32_t serial_ring[8] = {1u,2u,3u,0u,10u,20u,30u,0u},wave_ring[8],bulk_ring[8];
	uint32_t serial_out[4],wave_out[4],bulk_out[4],first,wave,row;
	uint64_t positions[4] = {3u,3u,4u,4u};
	memcpy(wave_ring,serial_ring,sizeof(serial_ring));
	memcpy(bulk_ring,serial_ring,sizeof(serial_ring));
	for (row=0u; row<4u; row++)
	{
		serial_ring[lanes[row] * 4u + positions[row] % 4u] = values[row];
		serial_out[row] = SparkDsv4RunnerRingSum(serial_ring,lanes[row],positions[row]);
		bulk_ring[lanes[row] * 4u + positions[row] % 4u] = values[row];
	}
	for (row=0u; row<4u; row++)
		bulk_out[row] = SparkDsv4RunnerRingSum(bulk_ring,lanes[row],positions[row]);
	for (first=0u; first<4u; first+=wave)
	{
		wave = SparkDsv4RoundMajorPrefillWaveRowCount(4u,2u,lanes,first);
		for (row=first; row<first+wave; row++)
			wave_ring[lanes[row] * 4u + positions[row] % 4u] = values[row];
		for (row=first; row<first+wave; row++)
			wave_out[row] = SparkDsv4RunnerRingSum(wave_ring,lanes[row],positions[row]);
	}
	assert(memcmp(serial_out,wave_out,sizeof(serial_out)) == 0);
	assert(bulk_out[0] != serial_out[0] && bulk_out[1] != serial_out[1]);
}

static void SparkDsv4RunnerTestPrefillOffsets(void)
{
	assert(SparkDsv4PrefillRowElementOffset(0u,512u) == 0u);
	assert(SparkDsv4PrefillRowElementOffset(3u,512u) == 1536u);
	assert(SparkDsv4PrefillRowElementOffset(UINT32_MAX,UINT32_MAX) > UINT32_MAX);
	assert(SparkDsv4AttentionWindowSlot(127u,0u,128u) == 0u);
	assert(SparkDsv4AttentionWindowSlot(127u,127u,128u) == 127u);
	assert(SparkDsv4AttentionWindowSlot(128u,0u,128u) == 1u);
	assert(SparkDsv4AttentionWindowSlot(128u,127u,128u) == 0u);
	assert(SparkDsv4AttentionWindowSlot(129u,0u,128u) == 2u);
	assert(SparkDsv4AttentionWindowSlot(129u,126u,128u) == 0u);
	assert(SparkDsv4AttentionWindowSlot(129u,127u,128u) == 1u);
	assert(SparkDsv4AttentionWindowSlot(0u,0u,0u) == UINT32_MAX);
}

int main(void)
{
	SparkDsv4RunnerTestPrefillMapping();
	SparkDsv4RunnerTestIntermediateRequiresInput();
	SparkDsv4RunnerTestIntermediateTokenRouting();
	SparkDsv4RunnerTestRoundMajorPrefill();
	SparkDsv4RunnerTestCausalBulkWaves();
	SparkDsv4RunnerTestPrefillOffsets();
    return 0;
}
