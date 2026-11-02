#include "inference/runner/stage_runner.cu"

#include <stdio.h>

#define FAKE_ROUNDS 64u
#define FAKE_WIDTH 6144u
#define FAKE_DEGREE 16u

typedef struct FakeRound
{
	uint32_t operation;
	uint32_t stage;
	uint32_t rows;
} FakeRound;

typedef struct FakeModel
{
	SparkStageRunnerState *state;
	const SparkStageRunnerServices *services;
	uint32_t index;
	uint32_t lane;
	uint16_t *buffer;
} FakeModel;

static pthread_mutex_t fake_lock = PTHREAD_MUTEX_INITIALIZER;
static FakeRound fake_rounds[FAKE_ROUNDS];
static uint32_t fake_count;
static uint32_t fake_paired = 1u;
static FakeModel fake_lanes[SPARK_STAGE_RUNNER_LANES];
static int failures;

static void FakeRecord(uint32_t operation, uint32_t stage, uint32_t rows)
{
	pthread_mutex_lock(&fake_lock);
	if ( fake_count < FAKE_ROUNDS )
	{
		fake_rounds[fake_count].operation = operation;
		fake_rounds[fake_count].stage = stage;
		fake_rounds[fake_count].rows = rows;
	}
	fake_count++;
	pthread_mutex_unlock(&fake_lock);
}

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
	(void)collective;
	FakeRecord(operation_kind, SPARK_STAGE_RUNNER_ROUND_ALL, submission->active_sequence_count);
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveBegin(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission,
	uint32_t operation_kind)
{
	(void)collective;
	FakeRecord(operation_kind, SPARK_STAGE_RUNNER_ROUND_BEGIN, submission->active_sequence_count);
	return SPARK_STATUS_OK;
}

SparkStatus SparkTpDeviceCollectiveFinish(SparkTpDeviceCollective *collective)
{
	(void)collective;
	FakeRecord(UINT32_MAX, SPARK_STAGE_RUNNER_ROUND_FINISH, 0u);
	return SPARK_STATUS_OK;
}

uint32_t SparkTpDeviceCollectiveSupportsPairAllReduce(const SparkTpDeviceCollective *collective, uint32_t rows, uint32_t row_elements)
{
	(void)collective;
	return fake_paired != 0u && SparkTpMeshAllReducePairFirst((uint64_t)rows * (row_elements != 0u ? row_elements : FAKE_WIDTH), FAKE_DEGREE,
		SPARK_WEIGHTD_MESH_SLOT_BYTES, SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) != 0u ? 1u : 0u;
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

}

static SparkStatus FakeStep(void *model, const SparkStageRunnerStep *step, void *stream)
{
	FakeModel *fake = (FakeModel *)model;
	SparkStatus status;
	if ( fake->lane != 0u )
	{
		fake->services->row_state(fake->services->context, 0u, fake->index == 0u ? SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN : SPARK_STAGE_RUNNER_ROW_STATE_READ);
		return fake->services->round(fake->services->context, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
			SPARK_STAGE_RUNNER_ROUND_ALL, step->rows, 0u, fake->buffer, fake->buffer);
	}
	status = fake->services->round(fake->services->context, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
		SPARK_STAGE_RUNNER_ROUND_ALL, step->rows, 0u, fake->buffer, fake->buffer);
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerReduceBf16(fake->state, (cudaStream_t)stream, fake->buffer, step->rows);
	if ( status == SPARK_STATUS_OK )
		status = fake->services->round(fake->services->context, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,
			SPARK_STAGE_RUNNER_ROUND_ALL, step->rows, FAKE_WIDTH / FAKE_DEGREE, fake->buffer, fake->buffer + FAKE_WIDTH);
	if ( status == SPARK_STATUS_OK )
		status = fake->services->round(fake->services->context, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
			SPARK_STAGE_RUNNER_ROUND_BEGIN, step->rows, 0u, fake->buffer, fake->buffer);
	if ( status == SPARK_STATUS_OK )
		status = fake->services->round(fake->services->context, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
			SPARK_STAGE_RUNNER_ROUND_FINISH, step->rows, 0u, fake->buffer, fake->buffer);
	return status;
}

static SparkStatus FakeLane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services, void **lane_model)
{
	FakeModel *fake = (FakeModel *)model;
	(void)step;
	fake_lanes[lane].state = fake->state;
	fake_lanes[lane].services = services;
	fake_lanes[lane].index = lane;
	fake_lanes[lane].lane = 1u;
	fake_lanes[lane].buffer = fake->buffer;
	*lane_model = &fake_lanes[lane];
	return SPARK_STATUS_OK;
}

static void Expect(int condition, const char *name)
{
	if ( condition )
		return;
	printf("FAIL %s\n", name);
	failures++;
}

static void ExpectRounds(const uint32_t *operations, uint32_t count, const char *name)
{
	uint32_t index;
	int same = fake_count == count;
	for ( index = 0u; same && index < count; ++index )
		same = fake_rounds[index].operation == operations[index];
	if ( !same )
	{
		printf("FAIL %s:", name);
		for ( index = 0u; index < fake_count && index < FAKE_ROUNDS; ++index )
			printf(" %u/%u/%u", fake_rounds[index].operation, fake_rounds[index].stage, fake_rounds[index].rows);
		printf("\n");
		failures++;
	}
	fake_count = 0u;
}

typedef struct LaneCall
{
	StageRunnerLane *lane;
	uint32_t rows;
	uint16_t *buffer;
	SparkStatus status;
} LaneCall;

static void *LaneCallMain(void *argument)
{
	LaneCall *call = (LaneCall *)argument;
	call->status = StageRunnerLaneRound(call->lane, call->lane->stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
		SPARK_STAGE_RUNNER_ROUND_ALL, call->rows, 0u, call->buffer, call->buffer);
	return 0;
}

int main(void)
{
	const uint32_t pair = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_ALL_REDUCE_SUM_BF16;
	const uint32_t plain = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	const uint32_t gather = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER;
	const uint32_t prefill = SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL, verify = SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY;
	SparkStageRunnerState *state = new SparkStageRunnerState;
	SparkStageRunnerModelInterface model_interface;
	SparkStageRunner runner;
	SparkStageRunnerStep in;
	FakeModel model;
	uint32_t *positions, *slots, *context_device;
	uint16_t *buffer;
	memset(state, 0, sizeof(*state));
	memset(&model_interface, 0, sizeof(model_interface));
	memset(&runner, 0, sizeof(runner));
	memset(&model, 0, sizeof(model));
	model_interface.tag = "sum-rule";
	model_interface.step = FakeStep;
	positions = new uint32_t[2048];
	slots = new uint32_t[2048];
	for ( uint32_t row = 0u; row < 2048u; ++row )
	{
		positions[row] = row;
		slots[row] = 0u;
	}
	if ( cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking) != cudaSuccess ||
		cudaMalloc((void **)&buffer, (size_t)2048u * FAKE_WIDTH * 2u * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&context_device, 4u * sizeof(uint32_t)) != cudaSuccess ||
		cudaMemset(context_device, 0, 4u * sizeof(uint32_t)) != cudaSuccess )
	{
		printf("FAIL stage_runner_sum_rule_cuda: no CUDA device\n");
		return 1;
	}
	state->model_interface = &model_interface;
	state->model = &model;
	state->device_collective_created = 1;
	state->device_collective_deferred = 1u;
	state->tp_degree = FAKE_DEGREE;
	state->sequence_capacity = 4u;
	state->tap_final_entry = UINT32_MAX;
	state->services.context = state;
	state->services.round = StageRunnerRoundService;
	model.state = state;
	model.services = &state->services;
	model.buffer = buffer;
	memset(&in, 0, sizeof(in));
	in.hidden_bf16 = buffer;
	in.rows = 2048u;
	in.sequences = 1u;
	in.last_rows_only = 1u;
	in.context_length = context_device;
	in.host_positions = positions;
	in.host_sequence_of_row = slots;

	{
		const uint32_t expected[5] = {pair, pair, gather, plain, UINT32_MAX};
		Expect(StageRunnerStepBody(state, &runner, &in, prefill, state->stream) == SPARK_STATUS_OK, "a prefill step runs");
		Expect(state->wave_pair_sums == 2u && state->wave_plain_sums == 1u, "a prefill wave counts its pair-first and plain sums");
		ExpectRounds(expected, 5u, "a prefill wave sums pair-first, keeps gathers and split rounds as they are");
		state->wave_pair_sums = 0u;
		state->wave_plain_sums = 0u;
	}
	{
		const uint32_t expected[5] = {plain, plain, gather, plain, UINT32_MAX};
		Expect(StageRunnerStepBody(state, &runner, &in, prefill | verify, state->stream) == SPARK_STATUS_OK, "a prefill-verify step runs");
		ExpectRounds(expected, 5u, "a prefill dispatch that also verifies keeps the plain all-reduce");
		Expect(StageRunnerStepBody(state, &runner, &in, 0u, state->stream) == SPARK_STATUS_OK, "a decode step runs");
		ExpectRounds(expected, 5u, "a decode dispatch keeps the plain all-reduce");
		Expect(StageRunnerStepBody(state, &runner, &in, verify, state->stream) == SPARK_STATUS_OK, "a verify step runs");
		ExpectRounds(expected, 5u, "a verify frame keeps the plain all-reduce");
		in.rows = 256u;
		Expect(StageRunnerStepBody(state, &runner, &in, prefill, state->stream) == SPARK_STATUS_OK, "a short prefill step runs");
		ExpectRounds(expected, 5u, "a prefill wave below the pair-first size keeps the plain all-reduce");
		in.rows = 2048u;
		fake_paired = 0u;
		Expect(StageRunnerStepBody(state, &runner, &in, prefill, state->stream) == SPARK_STATUS_OK, "an unpaired prefill step runs");
		ExpectRounds(expected, 5u, "a prefill wave on an unpaired mesh keeps the plain all-reduce");
		fake_paired = 1u;
	}
	{
		const uint32_t expected[1] = {plain};
		Expect(StageRunnerRoundService(state, state->stream, plain, SPARK_STAGE_RUNNER_ROUND_ALL, 2048u, 0u, buffer, buffer) == SPARK_STATUS_OK,
			"a round outside any dispatch runs");
		ExpectRounds(expected, 1u, "a round issued after the prefill step ended keeps the plain all-reduce");
	}
	model_interface.lane = FakeLane;
	Expect(StageRunnerCreateLanes(state) == SPARK_STATUS_OK && state->lanes.created != 0u, "two lanes start");
	{
		const uint32_t lanes_pair[2] = {pair, pair}, lanes_plain[2] = {plain, plain};
		Expect(StageRunnerStepBody(state, &runner, &in, prefill, state->stream) == SPARK_STATUS_OK, "a two-lane prefill step runs");
		Expect(fake_count == 2u && fake_rounds[0].rows + fake_rounds[1].rows == 2048u, "each lane sums its own half of the wave");
		ExpectRounds(lanes_pair, 2u, "both lanes of a prefill wave sum pair-first");
		Expect(StageRunnerStepBody(state, &runner, &in, prefill | verify, state->stream) == SPARK_STATUS_OK, "a two-lane verify step runs");
		ExpectRounds(lanes_plain, 2u, "both lanes of a prefill dispatch that verifies keep the plain all-reduce");
		in.rows = 512u;
		Expect(StageRunnerStepBody(state, &runner, &in, prefill, state->stream) == SPARK_STATUS_OK, "a short two-lane prefill step runs");
		ExpectRounds(lanes_plain, 2u, "lanes whose halves are below the pair-first size keep the plain all-reduce");
		in.rows = 2048u;
	}
	{
		StageRunnerLanes *lanes = &state->lanes;
		LaneCall calls[2];
		pthread_t threads[2];
		uint32_t index, found = 0u;
		for ( index = 0u; index < SPARK_STAGE_RUNNER_LANES; ++index )
		{
			lanes->lane[index].epoch = 0u;
			lanes->lane[index].group_rounds = 0u;
			lanes->lane[index].holding = 0u;
			lanes->lane[index].finished = 0u;
		}
		lanes->failed = 0u;
		lanes->lane[0].sum_flags = prefill;
		lanes->lane[1].sum_flags = 0u;
		state->sum_flags = prefill | verify;
		calls[0].lane = &lanes->lane[0];
		calls[0].rows = 1024u;
		calls[1].lane = &lanes->lane[1];
		calls[1].rows = 1280u;
		for ( index = 0u; index < 2u; ++index )
		{
			calls[index].buffer = buffer;
			calls[index].status = SPARK_STATUS_INTERNAL_ERROR;
			pthread_create(&threads[index], 0, LaneCallMain, &calls[index]);
		}
		for ( index = 0u; index < 2u; ++index )
			pthread_join(threads[index], 0);
		state->sum_flags = 0u;
		for ( index = 0u; index < fake_count && index < FAKE_ROUNDS; ++index )
			found |= (fake_rounds[index].rows == 1024u && fake_rounds[index].operation == pair ? 1u : 0u) |
				(fake_rounds[index].rows == 1280u && fake_rounds[index].operation == plain ? 2u : 0u);
		Expect(calls[0].status == SPARK_STATUS_OK && calls[1].status == SPARK_STATUS_OK && fake_count == 2u && found == 3u,
			"two concurrent lanes each sum by their own dispatch's rule, not the runner's");
		fake_count = 0u;
	}
	StageRunnerDestroyLanes(state);
	cudaFree(buffer);
	cudaFree(context_device);
	cudaStreamDestroy(state->stream);
	delete[] positions;
	delete[] slots;
	delete state;
	if ( failures != 0 )
	{
		printf("FAIL stage_runner_sum_rule_cuda: %d failures\n", failures);
		return 1;
	}
	printf("PASS stage_runner_sum_rule_cuda: prefill waves sum pair-first; prefill-verify, decode, verify, small and unpaired waves keep the plain all-reduce; each lane follows its own dispatch\n");
	return 0;
}
