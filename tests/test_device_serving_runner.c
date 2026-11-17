#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_speculation_tree_frame.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_runner_model.h"

#define TEST_ROWS 64u
#define TEST_SLOT 2u
#define TEST_TOKEN_BASE 1000u

struct SparkDeviceStageRunner
{
	SparkDeviceQueue *queue;
	SparkStageRunnerModelGeometry geometry;
	SparkDeviceStageRunnerInfo info;
	uint32_t external_kv;
	uint32_t argmax[TEST_ROWS];
	float logprob[4u * TEST_ROWS];
	uint32_t steps;
	uint32_t rows;
	uint32_t verify;
	uint32_t tree;
	uint32_t parents[TEST_ROWS];
	uint32_t positions[TEST_ROWS];
	uint32_t tokens[TEST_ROWS];
	uint32_t sequence_of_row[TEST_ROWS];
	uint32_t head_rows;
	uint32_t keep_paths;
	uint32_t path_rows[TEST_ROWS];
	uint32_t path_begin[2];
	uint32_t path_sequences;
};

static uint32_t failures;
static uint32_t checks;
static uint32_t device_tree_verify;
static uint64_t device_recurrent_bytes;

static void Check(int condition, const char *what)
{
	checks++;
	if ( condition )
		return;
	failures++;
	fprintf(stderr, "FAIL %s\n", what);
}

SparkStatus SparkDeviceStageRunnerOpen(SparkDeviceStageRunner **runner, const SparkDeviceStageRunnerConfig *config)
{
	SparkDeviceStageRunner *fake = (SparkDeviceStageRunner *)calloc(1u, sizeof(*fake));
	if ( fake == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( SparkDeviceQueueCreate(&fake->queue, "test device serving runner") != SPARK_STATUS_OK )
	{
		free(fake);
		return SPARK_STATUS_IO_ERROR;
	}
	fake->external_kv = config->external_kv;
	fake->geometry.recurrent_bytes = device_recurrent_bytes;
	fake->info.tree_verify = device_tree_verify;
	*runner = fake;
	return SPARK_STATUS_OK;
}

void SparkDeviceStageRunnerClose(SparkDeviceStageRunner *runner)
{
	(void)SparkDeviceQueueDestroy(runner->queue);
	free(runner);
}

SparkStatus SparkDeviceStageRunnerStep(SparkDeviceStageRunner *runner, const SparkDeviceStageRunnerBatch *batch, SparkDeviceStageRunnerResult *result)
{
	runner->steps++;
	runner->rows = batch->rows;
	runner->verify = batch->verify;
	runner->tree = batch->parent_rows != 0 ? 1u : 0u;
	runner->head_rows = batch->head_rows;
	for ( uint32_t row = 0u; row < batch->rows; row++ )
	{
		runner->parents[row] = batch->parent_rows != 0 ? batch->parent_rows[row] : SPARK_SPECULATION_TREE_FRAME_NO_ROW;
		runner->positions[row] = batch->positions[row];
		runner->tokens[row] = batch->token_ids[row];
		runner->sequence_of_row[row] = batch->sequence_of_row[row];
	}
	for ( uint32_t head = 0u; head < batch->head_rows; head++ )
		runner->argmax[head] = TEST_TOKEN_BASE + batch->head_row_index[head];
	memset(result, 0, sizeof(*result));
	result->argmax = runner->argmax;
	result->logprob = runner->logprob;
	return SPARK_STATUS_OK;
}

SparkStatus SparkDeviceStageRunnerKeepRows(SparkDeviceStageRunner *runner, const uint32_t *kept, uint32_t sequences)
{
	(void)runner;
	(void)kept;
	(void)sequences;
	return SPARK_STATUS_OK;
}

SparkStatus SparkDeviceStageRunnerKeepPath(SparkDeviceStageRunner *runner, const uint32_t *path_rows, const uint32_t *path_begin, uint32_t sequences)
{
	runner->keep_paths++;
	runner->path_sequences = sequences;
	runner->path_begin[0] = path_begin[0];
	runner->path_begin[1] = path_begin[1];
	memcpy(runner->path_rows, path_rows, (size_t)path_begin[sequences] * sizeof(uint32_t));
	return SPARK_STATUS_OK;
}

SparkStatus SparkDeviceStageRunnerResetSequence(SparkDeviceStageRunner *runner, uint32_t sequence)
{
	(void)runner;
	(void)sequence;
	return SPARK_STATUS_OK;
}

SparkStatus SparkDeviceStageRunnerSample(SparkDeviceStageRunner *runner, uint32_t rows, const SparkSampleRow *rules, SparkDeviceStageRunnerSamples *samples)
{
	(void)runner;
	(void)rows;
	(void)rules;
	(void)samples;
	return SPARK_STATUS_UNSUPPORTED;
}

SparkStatus SparkDeviceStageRunnerAttachKv(SparkDeviceStageRunner *runner, const SparkStageRunnerKv *kv)
{
	(void)runner;
	(void)kv;
	return SPARK_STATUS_OK;
}

SparkStatus SparkDeviceStageRunnerRecurrentCopy(SparkDeviceStageRunner *runner, uint32_t to_buffer, uint32_t sequence, SparkDeviceAddress buffer,
	uint64_t bytes)
{
	(void)runner;
	(void)to_buffer;
	(void)sequence;
	(void)buffer;
	(void)bytes;
	return SPARK_STATUS_OK;
}

SparkDeviceQueue *SparkDeviceStageRunnerQueue(const SparkDeviceStageRunner *runner)
{
	return runner->queue;
}

const SparkStageRunnerModelGeometry *SparkDeviceStageRunnerGeometry(const SparkDeviceStageRunner *runner)
{
	return &runner->geometry;
}

SparkStatus SparkDeviceStageRunnerInfoGet(const SparkDeviceStageRunner *runner, SparkDeviceStageRunnerInfo *info)
{
	*info = runner->info;
	return SPARK_STATUS_OK;
}

void *SparkDeviceStageRunnerModel(const SparkDeviceStageRunner *runner)
{
	return (void *)runner;
}

static SparkStatus TestKeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream)
{
	(void)model;
	(void)step;
	(void)kept;
	(void)stream;
	return SPARK_STATUS_OK;
}

static SparkStageRunnerModelInterface keeping_model;
static SparkStageRunnerModelInterface plain_model;

static SparkDeviceStageRunner *Fake(const SparkStageRunner *runner)
{
	return (SparkDeviceStageRunner *)SparkStageRunnerModel(runner);
}

static SparkStatus Open(SparkStageRunner *runner, const SparkStageRunnerModelInterface *model, uint32_t tree_verify, uint64_t recurrent_bytes)
{
	SparkStageRunnerConfiguration configuration;
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	configuration.stage_count = 1u;
	configuration.tp_degree = 1u;
	configuration.max_input_row_count = TEST_ROWS;
	configuration.resident_sequence_capacity = 4u;
	configuration.kv_pages_per_sequence = 8u;
	configuration.head_row_capacity = TEST_ROWS;
	configuration.rank_pack_path = "test.pack";
	device_tree_verify = tree_verify;
	device_recurrent_bytes = recurrent_bytes;
	return SparkStageRunnerInitialize(runner, &configuration, model);
}

static void TestCapability(void)
{
	SparkStageRunner runner;
	Check(Open(&runner, &keeping_model, 1u, 4096u) == SPARK_STATUS_OK, "a device stage that verifies trees opens");
	Check(Fake(&runner)->external_kv == 1u, "serving opens the device runner on the binding's external KV");
	Check(SparkStageRunnerVerifyCapable(&runner) == 1u, "a recurrent model that keeps rows verifies");
	Check(SparkStageRunnerTreeVerifyCapable(&runner) == 1u, "a device stage whose runner verifies trees reports the tree capability");
	Check(SparkStageRunnerTreeSlotGrain(&runner) == SPARK_DEVICE_RUNNER_TREE_SLOT_GRAIN, "the tree slot grain is the device runner's");
	SparkStageRunnerDestroy(&runner);
	Check(Open(&runner, &keeping_model, 0u, 4096u) == SPARK_STATUS_OK, "a device stage without tree frames opens");
	Check(SparkStageRunnerTreeVerifyCapable(&runner) == 0u && SparkStageRunnerTreeSlotGrain(&runner) == 0u,
		"a runner without tree frames reports neither the capability nor a grain");
	SparkStageRunnerDestroy(&runner);
	Check(Open(&runner, &plain_model, 1u, 4096u) == SPARK_STATUS_OK, "a recurrent model without keep_rows opens");
	Check(SparkStageRunnerVerifyCapable(&runner) == 0u && SparkStageRunnerTreeVerifyCapable(&runner) == 0u && SparkStageRunnerTreeSlotGrain(&runner) == 0u,
		"a model that cannot keep verified rows verifies neither chains nor trees");
	SparkStageRunnerDestroy(&runner);
}

typedef struct TestFrame
{
	SparkDeviceBuffer lane_slots;
	SparkDeviceBuffer output_tokens;
	uint32_t tokens[TEST_ROWS];
	uint32_t positions[TEST_ROWS];
	uint32_t sequence_of_row[TEST_ROWS];
	uint32_t parents[TEST_ROWS];
	SparkStageRunnerDispatch dispatch;
} TestFrame;

static const uint32_t tree_parents[6] = { SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 0u, 3u, 1u };
static const uint32_t tree_depths[6] = { 0u, 1u, 2u, 1u, 2u, 2u };
static const uint32_t tree_tokens[6] = { 10u, 11u, 12u, 21u, 22u, 31u };

static void FrameOpen(TestFrame *frame)
{
	memset(frame, 0, sizeof(*frame));
	if ( SparkDeviceAllocate(&frame->lane_slots, SPARK_DEVICE_SPACE_SHARED_COHERENT, 4u * sizeof(uint32_t), "test lane slots") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&frame->output_tokens, SPARK_DEVICE_SPACE_SHARED_COHERENT, TEST_ROWS * sizeof(uint32_t), "test output tokens") != SPARK_STATUS_OK )
	{
		fprintf(stderr, "FAIL the host device backend allocates the test buffers\n");
		exit(1);
	}
	((uint32_t *)frame->lane_slots.host)[0] = TEST_SLOT;
	((uint32_t *)frame->lane_slots.host)[1] = TEST_SLOT + 1u;
	for ( uint32_t row = 0u; row < 6u; row++ )
	{
		frame->tokens[row] = tree_tokens[row];
		frame->positions[row] = 100u + tree_depths[row];
		frame->sequence_of_row[row] = TEST_SLOT;
		frame->parents[row] = tree_parents[row];
	}
	frame->dispatch.abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	frame->dispatch.descriptor_bytes = (uint32_t)sizeof(frame->dispatch);
	frame->dispatch.flags = SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY;
	frame->dispatch.row_count = 6u;
	frame->dispatch.active_sequence_count = 1u;
	frame->dispatch.chain_steps = 1u;
	frame->dispatch.token_ids = frame->tokens;
	frame->dispatch.host_positions = frame->positions;
	frame->dispatch.host_sequence_of_row = frame->sequence_of_row;
	frame->dispatch.recurrent_index = (const uint32_t *)(uintptr_t)frame->lane_slots.device;
	frame->dispatch.output_token_ids = (uint32_t *)(uintptr_t)frame->output_tokens.device;
	frame->dispatch.tree = 1u;
	frame->dispatch.host_parent_rows = frame->parents;
}

static void FrameClose(TestFrame *frame)
{
	(void)SparkDeviceFree(&frame->lane_slots);
	(void)SparkDeviceFree(&frame->output_tokens);
}

static void TestTreeFrame(void)
{
	SparkStageRunner runner;
	TestFrame frame;
	SparkDeviceStageRunner *fake;
	uint32_t row, same = 1u;
	if ( Open(&runner, &keeping_model, 1u, 4096u) != SPARK_STATUS_OK )
	{
		Check(0, "the tree-frame runner opens");
		return;
	}
	fake = Fake(&runner);
	FrameOpen(&frame);
	Check(SparkStageRunnerSubmit(&runner, &frame.dispatch) == SPARK_STATUS_OK, "a tree verify frame of one sequence is served");
	Check(fake->steps == 1u && fake->verify == 1u && fake->tree == 1u && fake->rows == 6u,
		"the tree frame reaches the device runner as one verify step with parent rows");
	for ( row = 0u; row < 6u; row++ )
		same &= fake->parents[row] == tree_parents[row] && fake->positions[row] == 100u + tree_depths[row] && fake->tokens[row] == tree_tokens[row] &&
			fake->sequence_of_row[row] == TEST_SLOT;
	Check(same != 0u, "every row keeps its parent, semantic position, token and sequence slot in dispatch order");
	Check(fake->head_rows == 6u, "every row of a tree frame is a head row");
	Check(SparkDeviceQueueWait(SparkStageRunnerQueue(&runner), SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK, "the token copy completes");
	for ( row = 0u, same = 1u; row < 6u; row++ )
		same &= ((const uint32_t *)frame.output_tokens.host)[row] == TEST_TOKEN_BASE + row;
	Check(same != 0u, "each row's verified token returns at that row");
	{
		const uint32_t path_rows[3] = { 0u, 3u, 4u }, path_begin[2] = { 0u, 3u }, kept[1] = { 3u }, short_kept[1] = { 2u };
		const uint32_t backwards[2] = { 3u, 0u };
		Check(SparkStageRunnerKeepPath(&runner, path_rows, path_begin, short_kept, 1u) == SPARK_STATUS_INVALID_ARGUMENT && fake->keep_paths == 0u,
			"a kept count that is not its path's length is refused before the device runner");
		Check(SparkStageRunnerKeepPath(&runner, path_rows, backwards, kept, 1u) == SPARK_STATUS_INVALID_ARGUMENT && fake->keep_paths == 0u,
			"a path range that runs backwards is refused");
		Check(SparkStageRunnerKeepPath(&runner, path_rows, path_begin, kept, 1u) == SPARK_STATUS_OK, "the accepted root path is kept");
		Check(fake->keep_paths == 1u && fake->path_sequences == 1u && fake->path_begin[0] == 0u && fake->path_begin[1] == 3u &&
			fake->path_rows[0] == 0u && fake->path_rows[1] == 3u && fake->path_rows[2] == 4u, "the device runner receives the path rows and their ranges");
	}
	FrameClose(&frame);
	SparkStageRunnerDestroy(&runner);
}

static void TestTreeRefusals(void)
{
	SparkStageRunner runner;
	TestFrame frame;
	SparkDeviceStageRunner *fake;
	if ( Open(&runner, &keeping_model, 1u, 4096u) != SPARK_STATUS_OK )
	{
		Check(0, "the refusal runner opens");
		return;
	}
	fake = Fake(&runner);
	FrameOpen(&frame);
	frame.dispatch.flags = 0u;
	Check(SparkStageRunnerSubmit(&runner, &frame.dispatch) == SPARK_STATUS_INVALID_ARGUMENT, "a tree frame that is not a verify step is refused");
	frame.dispatch.flags = SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY | SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL;
	Check(SparkStageRunnerSubmit(&runner, &frame.dispatch) == SPARK_STATUS_INVALID_ARGUMENT, "a tree frame that prefills is refused");
	frame.dispatch.flags = SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY;
	frame.dispatch.host_parent_rows = 0;
	Check(SparkStageRunnerSubmit(&runner, &frame.dispatch) == SPARK_STATUS_INVALID_ARGUMENT, "a tree frame without parent rows is refused");
	frame.dispatch.host_parent_rows = frame.parents;
	frame.sequence_of_row[5] = TEST_SLOT + 1u;
	frame.dispatch.active_sequence_count = 2u;
	Check(SparkStageRunnerSubmit(&runner, &frame.dispatch) == SPARK_STATUS_INVALID_ARGUMENT, "a tree frame over two sequences is refused");
	Check(fake->steps == 0u, "no refused tree frame reaches the device runner");
	frame.dispatch.tree = 0u;
	frame.dispatch.host_parent_rows = 0;
	Check(SparkStageRunnerSubmit(&runner, &frame.dispatch) == SPARK_STATUS_OK, "a chain verify frame over two sequences is served");
	Check(fake->steps == 1u && fake->tree == 0u && fake->verify == 1u, "a chain verify frame carries no parent rows");
	FrameClose(&frame);
	SparkStageRunnerDestroy(&runner);
}

int main(void)
{
	keeping_model.keep_rows = TestKeepRows;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "FAIL the host device backend initializes\n");
		return 1;
	}
	TestCapability();
	TestTreeFrame();
	TestTreeRefusals();
	(void)SparkDeviceShutdown();
	printf("test_device_serving_runner: %u checks, %u failures\n", checks, failures);
	if ( failures != 0u )
		return 1;
	printf("PASS device serving runner: a device stage that verifies trees reports the capability and the device runner's slot grain, a tree verify frame "
		"reaches the device runner with its parent rows in dispatch order and its tokens return per row, keep_path forwards the accepted root path after "
		"checking each kept count against its path, and tree frames that are not one sequence's verify step with parents are refused before the device\n");
	return 0;
}
