#include "inference/runner/stage_runner.cu"

#include <stdio.h>

#define LONE_HIDDEN 256u
#define LONE_VOCAB 256u
#define LONE_ROWS 2u
#define LONE_INPUT_WIDTH (3u * LONE_HIDDEN)

static int failures;
static uint32_t lone_steps;
static uint64_t lone_digest = 1469598103934665603ull;
static uint16_t lone_seen[LONE_ROWS * LONE_INPUT_WIDTH];

extern "C" {

void SparkTpDeviceCollectiveRegisterLaunchers(const SparkTpMeshLaunchers *launchers)
{
	(void)launchers;
}

SparkStatus SparkTpDeviceCollectiveCreate(const SparkTpDeviceCollectiveConfig *config, SparkTpDeviceCollective *collective_out)
{
	(void)config; (void)collective_out;
	return SPARK_STATUS_UNSUPPORTED;
}

void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
	(void)collective;
}

SparkStatus SparkTpDeviceCollectivePrepareReceiveBf16(SparkTpDeviceCollective *collective, void *receive_device,
	uint32_t active_sequence_count, uint32_t hidden_dimension, uint32_t step_index, void *cuda_stream)
{
	(void)collective; (void)receive_device; (void)active_sequence_count; (void)hidden_dimension; (void)step_index; (void)cuda_stream;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission,
	uint32_t operation_kind)
{
	(void)collective; (void)submission; (void)operation_kind;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveBegin(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission,
	uint32_t operation_kind)
{
	(void)collective; (void)submission; (void)operation_kind;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveFinish(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_OK;
}

uint32_t SparkTpDeviceCollectiveSupportsPairAllReduce(const SparkTpDeviceCollective *collective, uint32_t rows, uint32_t row_elements)
{
	(void)collective; (void)rows; (void)row_elements;
	return 0u;
}

uint32_t SparkTpDeviceCollectivePublished(const SparkTpDeviceCollective *collective)
{
	(void)collective;
	return 0u;
}

SparkStatus SparkTpDeviceCollectiveVerifyDeferred(SparkTpDeviceCollective *collective, void *stream)
{
	(void)collective; (void)stream;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveHardwareStats(SparkTpDeviceCollective *collective, SparkTpDeviceCollectiveHardwareTiming *timing_out)
{
	(void)collective; (void)timing_out;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveGraphSettle(SparkTpDeviceCollective *collective, void *stream, uint64_t *error_out)
{
	(void)collective; (void)stream; (void)error_out;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveGraphPreLaunch(SparkTpDeviceCollective *collective, void *stream)
{
	(void)collective; (void)stream;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveGraphCancelSeed(SparkTpDeviceCollective *collective, void *stream)
{
	(void)collective; (void)stream;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveEndChain(SparkTpDeviceCollective *collective, void *cuda_stream)
{
	(void)collective; (void)cuda_stream;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveChainKey(SparkTpDeviceCollective *collective, uint64_t request_id)
{
	(void)collective; (void)request_id;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveArmCapture(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkTpDeviceCollectiveDisarmCapture(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveSetOperationTimeout(SparkTpDeviceCollective *collective, uint32_t timeout_milli, uint32_t *previous_milli)
{
	(void)collective; (void)timeout_milli;
	if ( previous_milli != 0 )
		*previous_milli = 0u;
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveReady(SparkTpDeviceCollective *collective)
{
	(void)collective;
	return SPARK_STATUS_OK;
}

}

static SparkStatus LoneIdentityStep(void *model, const SparkStageRunnerStep *step, void *stream)
{
	(void)model;
	(void)step;
	(void)stream;
	lone_steps++;
	return SPARK_STATUS_OK;
}

static SparkStatus LoneInputStep(void *model, const SparkStageRunnerStep *step, void *stream)
{
	(void)model;
	lone_steps++;
	if ( SparkGpuMemcpyAsync(lone_seen, step->hidden_bf16, (size_t)step->rows * LONE_INPUT_WIDTH * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_HOST,
			(SparkGpuStream)stream) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize((SparkGpuStream)stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static void LoneDigest(const void *data, size_t bytes)
{
	const uint8_t *byte = (const uint8_t *)data;
	for ( size_t index = 0u; index < bytes; ++index )
		lone_digest = (lone_digest ^ byte[index]) * 1099511628211ull;
}

static void Expect(int condition, const char *name)
{
	if ( condition )
		return;
	printf("FAIL %s\n", name);
	failures++;
}

typedef struct LoneCompletion
{
	uint32_t calls;
	uint32_t tokens[LONE_ROWS];
} LoneCompletion;

static void LoneComplete(void *context, const SparkModelDriverCompletion *completion)
{
	LoneCompletion *seen = (LoneCompletion *)context;
	seen->calls++;
	for ( uint32_t row = 0u; row < LONE_ROWS && row < completion->token_count; ++row )
		seen->tokens[row] = completion->token_ids[row];
}

static uint16_t *LoneUpload(const uint16_t *host, size_t count)
{
	uint16_t *device = 0;
	if ( SparkGpuMalloc((void **)&device, count * sizeof(uint16_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(device, host, count * sizeof(uint16_t), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return 0;
	return device;
}

static SparkStatus LoneHead(SparkStageRunnerState *state)
{
	static uint16_t embed[LONE_VOCAB * LONE_HIDDEN], head[LONE_VOCAB * LONE_HIDDEN], norm[LONE_HIDDEN];
	const uint64_t rows = LONE_VOCAB, dim = LONE_HIDDEN;
	memset(embed, 0, sizeof(embed));
	memset(head, 0, sizeof(head));
	for ( uint32_t token = 0u; token < LONE_VOCAB; ++token )
	{
		embed[(size_t)token * LONE_HIDDEN + token] = 0x3f80u;
		head[(size_t)token * LONE_HIDDEN + (token + LONE_VOCAB - 1u) % LONE_VOCAB] = 0x3f80u;
	}
	for ( uint32_t index = 0u; index < LONE_HIDDEN; ++index )
		norm[index] = 0x3f80u;
	state->embed_weight = LoneUpload(embed, LONE_VOCAB * LONE_HIDDEN);
	state->head_weight = LoneUpload(head, LONE_VOCAB * LONE_HIDDEN);
	state->head_norm_weight = LoneUpload(norm, LONE_HIDDEN);
	if ( state->embed_weight == 0 || state->head_weight == 0 || state->head_norm_weight == 0 ||
		SparkGpuMalloc(&state->head_certified_fp8_payload, rows * dim) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_fp8_scale_f32, rows * (dim / 32u) * sizeof(float)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_fp8_norm_f32, rows * (dim / 32u) * sizeof(float)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_scratch, SparkHeadCertifiedFp8ScratchBytes(rows, dim) * state->head_capacity) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_certified_candidates, SparkHeadCertifiedFp8CandidateBytes(rows) * state->head_capacity) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&state->head_screened_count, sizeof(uint32_t) * state->head_capacity) != SPARK_GPU_SUCCESS ||
		SparkLmHostLaunchHeadCertifiedFp8Quantize(0, state->head_weight, state->head_certified_fp8_payload, state->head_certified_fp8_scale_f32,
			state->head_certified_fp8_norm_f32, (uint32_t)rows, (uint32_t)dim) != SPARK_GPU_SUCCESS || SparkGpuDeviceSynchronize() != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStageRunnerState *LoneState(SparkStageRunnerModelInterface *model_interface, SparkStageRunner *runner, uint32_t input_width)
{
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunnerConfiguration configuration;
	memset(state, 0, sizeof(*state));
	memset(&configuration, 0, sizeof(configuration));
	configuration.max_input_row_count = LONE_ROWS;
	configuration.max_active_sequence_count = LONE_ROWS;
	configuration.resident_sequence_capacity = LONE_ROWS;
	state->model_interface = model_interface;
	state->tp_degree = 1u;
	state->max_rows = LONE_ROWS;
	state->head_capacity = LONE_ROWS;
	state->vocab = LONE_VOCAB;
	state->output_vocab = LONE_VOCAB;
	state->vocab_slice_rows = LONE_VOCAB;
	state->head_output_rows = LONE_VOCAB;
	state->sequence_capacity = LONE_ROWS;
	state->tap_final_entry = UINT32_MAX;
	state->geometry.hidden = LONE_HIDDEN;
	state->geometry.vocab = LONE_VOCAB;
	state->geometry.rms_epsilon = 1e-6f;
	state->geometry.input_width = input_width;
	runner->private_state = state;
	runner->tp_degree = 1u;
	if ( SparkGpuStreamCreateWithFlags(&state->stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		(runner->owns_final_head != 0u && LoneHead(state) != SPARK_STATUS_OK) ||
		StageRunnerCreateBuffers(state, runner, &configuration) != SPARK_STATUS_OK || StageRunnerAdoptQueue(state) != SPARK_STATUS_OK )
		return 0;
	return state;
}

static SparkStatus LoneDecode(SparkStageRunner *runner, const uint32_t *tokens, uint32_t *positions_device, uint32_t *context_device,
	uint32_t *sequence_device, LoneCompletion *completion)
{
	SparkStageRunnerDispatch dispatch;
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.row_count = LONE_ROWS;
	dispatch.active_sequence_count = LONE_ROWS;
	dispatch.token_ids = tokens;
	dispatch.positions = positions_device;
	dispatch.context_length = context_device;
	dispatch.sequence_of_row = sequence_device;
	dispatch.completion_function = LoneComplete;
	dispatch.completion_context = completion;
	return SparkStageRunnerSubmit(runner, &dispatch);
}

static void LoneGraphs(void)
{
	SparkStageRunnerModelInterface model_interface;
	SparkStageRunner runner;
	SparkStageRunnerState *state;
	LoneCompletion completion;
	const uint32_t first[LONE_ROWS] = {5u, 9u}, second[LONE_ROWS] = {20u, 255u}, positions[LONE_ROWS] = {3u, 7u}, sequences[LONE_ROWS] = {0u, 1u};
	const uint32_t contexts[LONE_ROWS] = {4u, 8u};
	uint32_t *positions_device = 0, *context_device = 0, *sequence_device = 0, slot;
	SparkDeviceProgram *captured;
	memset(&model_interface, 0, sizeof(model_interface));
	memset(&runner, 0, sizeof(runner));
	memset(&completion, 0, sizeof(completion));
	model_interface.tag = "lone-gpu";
	model_interface.step = LoneIdentityStep;
	runner.owns_embedding = 1u;
	runner.owns_final_head = 1u;
	state = LoneState(&model_interface, &runner, 0u);
	Expect(state != 0, "a single-GPU runner with its own embedding and head is set up without a collective");
	if ( state == 0 )
		return;
	if ( SparkGpuMalloc((void **)&positions_device, sizeof(positions)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&context_device, sizeof(contexts)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&sequence_device, sizeof(sequences)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(positions_device, positions, sizeof(positions), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(context_device, contexts, sizeof(contexts), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(sequence_device, sequences, sizeof(sequences), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
	{
		Expect(0, "the decode rows are uploaded");
		return;
	}
	slot = StageRunnerGraphSlot(1u, LONE_ROWS, 0u, 0u);
	Expect(StageRunnerGraphEligible(state, &runner, 1u, LONE_ROWS, LONE_ROWS) != 0u, "a TP1 decode step without a collective is eligible for a graph");
	Expect(LoneDecode(&runner, first, positions_device, context_device, sequence_device, &completion) == SPARK_STATUS_OK && completion.calls == 1u &&
		completion.tokens[0] == 6u && completion.tokens[1] == 10u, "the first decode embeds, runs the model and takes the head's argmax");
	LoneDigest(completion.tokens, sizeof(completion.tokens));
	LoneDigest(state->output_score_host, LONE_ROWS * sizeof(float));
	captured = state->graph_program[slot];
	Expect(captured != 0 && state->graph_launches == 1u && lone_steps == 1u, "the first decode recorded its program and ran it once");
	Expect(LoneDecode(&runner, second, positions_device, context_device, sequence_device, &completion) == SPARK_STATUS_OK && completion.calls == 2u &&
		completion.tokens[0] == 21u && completion.tokens[1] == 0u, "the second decode runs the program on new tokens");
	Expect(state->graph_program[slot] == captured && state->graph_launches == 2u && lone_steps == 1u,
		"the second decode reused the recorded program without running the model on the host again");
	LoneDigest(completion.tokens, sizeof(completion.tokens));
	LoneDigest(state->output_score_host, LONE_ROWS * sizeof(float));
	SparkGpuFree(positions_device);
	SparkGpuFree(context_device);
	SparkGpuFree(sequence_device);
}

static void LoneWideInput(void)
{
	static uint16_t input[LONE_ROWS * LONE_INPUT_WIDTH];
	SparkStageRunnerModelInterface model_interface;
	SparkStageRunnerDispatch dispatch;
	SparkStageRunner runner;
	SparkStageRunnerState *state;
	uint32_t same = 1u;
	memset(&model_interface, 0, sizeof(model_interface));
	memset(&runner, 0, sizeof(runner));
	model_interface.tag = "lone-input";
	model_interface.step = LoneInputStep;
	state = LoneState(&model_interface, &runner, LONE_INPUT_WIDTH);
	Expect(state != 0, "a runner whose rows are wider than its hidden width is set up");
	if ( state == 0 )
		return;
	for ( uint32_t index = 0u; index < LONE_ROWS * LONE_INPUT_WIDTH; ++index )
		input[index] = (uint16_t)(index * 7u + 3u);
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.row_count = LONE_ROWS;
	dispatch.active_sequence_count = LONE_ROWS;
	dispatch.hidden_input_bf16 = input;
	dispatch.hidden_input_bytes = sizeof(input);
	lone_steps = 0u;
	Expect(SparkStageRunnerSubmit(&runner, &dispatch) == SPARK_STATUS_OK && lone_steps == 1u, "a dispatch of wide input rows runs the model");
	for ( uint32_t index = 0u; index < LONE_ROWS * LONE_INPUT_WIDTH; ++index )
		same &= lone_seen[index] == input[index] ? 1u : 0u;
	Expect(same != 0u, "the model sees every element of the input rows, past the hidden width");
	LoneDigest(lone_seen, sizeof(lone_seen));
	dispatch.hidden_input_bytes = sizeof(input) - 2u;
	Expect(SparkStageRunnerSubmit(&runner, &dispatch) == SPARK_STATUS_INVALID_ARGUMENT, "input rows shorter than the input width are refused");
}

int main(void)
{
	LoneGraphs();
	LoneWideInput();
	if ( failures != 0 )
	{
		printf("FAIL stage_runner_single_gpu_cuda: %d failures\n", failures);
		return 1;
	}
	printf("digest stage runner %016llx\n", (unsigned long long)lone_digest);
	printf("PASS stage_runner_single_gpu_cuda: a TP1 runner without a collective records a decode program once and runs it again, and its input rows may be wider than its hidden width\n");
	return 0;
}
