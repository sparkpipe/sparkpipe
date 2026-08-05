#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_pipeline_client.h"

#ifndef TEST_MODEL_RESIDENTD_PATH
#define TEST_MODEL_RESIDENTD_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_BATCH_PATH
#define TEST_MODEL_BATCH_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

#define TEST_MODEL_PIPELINE_RANK_COUNT 3u

static const char *const TestModelPipelineTransportHosts[
	TEST_MODEL_PIPELINE_RANK_COUNT] =
{
	"test-stage-a","test-stage-b","test-stage-c"
};

typedef struct TestModelPipelineState
{
	uint32_t result_count;
	uint32_t completion_count;
	uint32_t callback_submitted;
	SparkStatus callback_submit_status;
	uint64_t result_submission_ids[9];
	SparkStatus result_statuses[9];
	SparkModelServingCompletion completions[9];
	SparkModelPipelineClient *pipeline;
	const SparkModelServingSubmission *callback_submission;
} TestModelPipelineState;

typedef struct TestModelBatchState
{
	uint32_t accepted_count;
	uint32_t token_count;
	uint32_t completed_count;
	uint32_t cancelled_count;
	uint32_t error_count;
	uint32_t stop_token_count;
	uint32_t token_ids[8];
} TestModelBatchState;

static void TestModelBatchEvent(
	void *event_context,
	const SparkModelBatchEvent *event)
{
	TestModelBatchState *state;
	state = (TestModelBatchState *)event_context;
	assert(event != 0);
	assert(event->abi_version == SPARK_MODEL_BATCH_ENGINE_ABI_VERSION);
	assert(event->descriptor_bytes == SPARK_MODEL_BATCH_EVENT_BYTES);
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED )
		state->accepted_count++;
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
	{
		assert(state->token_count < 8u);
		state->token_ids[state->token_count++] = event->token_id;
		if ( (event->flags & SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN) != 0u )
			state->stop_token_count++;
	}
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
		state->completed_count++;
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED )
		state->cancelled_count++;
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
		state->error_count++;
	else
		assert(0 && "unknown model batch event");
}

static void TestModelPipelineResult(
	void *result_context,
	uint64_t submission_id,
	SparkStatus status)
{
	TestModelPipelineState *state;
	state = (TestModelPipelineState *)result_context;
	assert(state->result_count < 9u);
	state->result_submission_ids[state->result_count] = submission_id;
	state->result_statuses[state->result_count] = status;
	state->result_count++;
}

static void TestModelPipelineCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestModelPipelineState *state;
	state = (TestModelPipelineState *)completion_context;
	assert(state->completion_count < 9u);
	state->completions[state->completion_count] = *completion;
	state->completion_count++;
	if ( state->callback_submitted == 0u && state->callback_submission != 0 )
	{
		state->callback_submitted = 1u;
		state->callback_submit_status = SparkModelPipelineClientSubmit(state->pipeline,state->callback_submission);
	}
}

static pid_t TestModelPipelineStartResident(
	const char *deployment_path,
	uint32_t rank_index)
{
	pid_t child;
	char rank[16];
	assert(snprintf(rank,sizeof(rank),"%u",rank_index) > 0);
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		execl(TEST_MODEL_RESIDENTD_PATH,TEST_MODEL_RESIDENTD_PATH,
			"--deployment",deployment_path,
			"--rank-index",rank,
			(char *)0);
		_exit(127);
	}
	return(child);
}

static void TestModelPipelineWaitForSockets(char paths[][108])
{
	struct stat status;
	struct timespec delay;
	uint32_t attempt,rank,ready;
	delay.tv_sec = 0;
	delay.tv_nsec = 10000000;
	for (attempt=0u; attempt<500u; attempt++)
	{
		ready = 1u;
		for (rank=1u; rank<TEST_MODEL_PIPELINE_RANK_COUNT; rank++)
			if ( lstat(paths[rank],&status) != 0 || !S_ISSOCK(status.st_mode) )
				ready = 0u;
		if ( ready != 0u )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "model pipeline sockets did not become ready");
}

static void TestModelPipelineBuildSubmission(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lanes,
	uint32_t *tokens,
	uint32_t *row_lanes,
	uint64_t *positions,
	uint64_t *sequences,
	uint64_t submission_id)
{
	memset(lanes,0,2u * sizeof(lanes[0]));
	lanes[0].request_id = 900u;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = submission_id + 3000u;
	lanes[0].sequence_id = 100u;
	lanes[0].resident_sequence_slot = 7u;
	lanes[1].request_id = 901u;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = submission_id + 3000u;
	lanes[1].sequence_id = 101u;
	lanes[1].resident_sequence_slot = 3u;
	tokens[0] = 11u;
	tokens[1] = 12u;
	row_lanes[0] = 0u;
	row_lanes[1] = 1u;
	positions[0] = 0u;
	positions[1] = 0u;
	sequences[0] = 100u;
	sequences[1] = 101u;
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->submission_id = submission_id;
	submission->request_id = 9u;
	submission->sequence_id = 100u;
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = submission_id + 2000u;
	submission->request_generation = 1u;
	submission->step_generation = submission_id + 3000u;
	submission->residency.word0 = submission_id;
	submission->residency.word1 = submission_id + 100u;
	submission->residency.generation = submission_id + 200u;
	submission->residency.owner = 13u;
	submission->active_sequence_count = 2u;
	submission->new_token_count = 2u;
	submission->lane_count = 2u;
	submission->row_count = 2u;
	submission->token_count = 2u;
	submission->lanes = lanes;
	submission->token_ids = tokens;
	submission->row_lane_indices = row_lanes;
	submission->row_positions = positions;
	submission->row_sequence_ids = sequences;
}

static void TestModelPipelineBuildPrefill(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lanes,
	uint32_t *tokens,
	uint32_t *row_lanes,
	uint64_t *positions,
	uint64_t *sequences,
	uint64_t submission_id)
{
	uint32_t row;
	memset(lanes,0,2u * sizeof(lanes[0]));
	lanes[0].request_id = 902u;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = submission_id + 3000u;
	lanes[0].sequence_id = 200u;
	lanes[0].resident_sequence_slot = 5u;
	for (row=0u; row<4u; row++)
	{
		tokens[row] = 21u + row;
		row_lanes[row] = 0u;
		positions[row] = row;
		sequences[row] = 200u;
	}
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission->submission_id = submission_id;
	submission->request_id = 10u;
	submission->sequence_id = 200u;
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = submission_id + 2000u;
	submission->request_generation = 1u;
	submission->step_generation = submission_id + 3000u;
	submission->residency.word0 = submission_id;
	submission->residency.word1 = submission_id + 100u;
	submission->residency.generation = submission_id + 200u;
	submission->residency.owner = 13u;
	submission->active_sequence_count = 1u;
	submission->new_token_count = 4u;
	submission->lane_count = 1u;
	submission->row_count = 4u;
	submission->token_count = 4u;
	submission->lanes = lanes;
	submission->token_ids = tokens;
	submission->row_lane_indices = row_lanes;
	submission->row_positions = positions;
	submission->row_sequence_ids = sequences;
}

static void TestModelPipelineBuildRelease(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lane,
	uint64_t submission_id)
{
	memset(lane,0,sizeof(*lane));
	lane->request_id = 902u;
	lane->request_generation = 1u;
	lane->step_generation = submission_id + 3000u;
	lane->sequence_id = 200u;
	lane->resident_sequence_slot = 5u;
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission->submission_id = submission_id;
	submission->request_id = 10u;
	submission->sequence_id = 200u;
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = submission_id + 2000u;
	submission->request_generation = 1u;
	submission->step_generation = submission_id + 3000u;
	submission->residency.word0 = submission_id;
	submission->residency.word1 = submission_id + 100u;
	submission->residency.generation = submission_id + 200u;
	submission->residency.owner = 13u;
	submission->active_sequence_count = 1u;
	submission->lane_count = 1u;
	submission->lanes = lane;
}

static void TestModelPipelineRetargetPrefill(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lane,
	uint64_t *sequences,
	uint64_t request_id,
	uint64_t sequence_id)
{
	uint32_t row;
	lane->request_id = request_id;
	lane->sequence_id = sequence_id;
	submission->request_id = request_id;
	submission->sequence_id = sequence_id;
	for (row=0u; row<submission->row_count; row++)
		sequences[row] = sequence_id;
}

static SparkModelPipelineClient *TestModelPipelineConnect(
	const SparkModelResidentDeployment *deployment,
	TestModelPipelineState *state)
{
	SparkModelPipelineClientConfiguration configuration;
	SparkModelPipelineClient *pipeline;
	struct timespec delay;
	SparkStatus status;
	uint32_t attempt;
	char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 100u;
	configuration.deployment = deployment;
	configuration.runtime_root = runtime_root;
	configuration.submit_result_function = TestModelPipelineResult;
	configuration.submit_result_context = state;
	configuration.completion_function = TestModelPipelineCompletion;
	configuration.completion_context = state;
	delay.tv_sec = 0;
	delay.tv_nsec = 10000000;
	pipeline = 0;
	for (attempt=0u; attempt<500u && pipeline==0; attempt++)
	{
		status = SparkModelPipelineClientConnect(&configuration,&pipeline);
		if ( status != SPARK_STATUS_OK )
			nanosleep(&delay,0);
	}
	assert(pipeline != 0);
	return(pipeline);
}

static void TestModelPipelineRejectMissingClientRuntimeRoot(
	const SparkModelResidentDeployment *deployment,
	TestModelPipelineState *state)
{
	SparkModelPipelineClientConfiguration configuration;
	SparkModelPipelineClient *pipeline;
	char runtime_root[108];
	assert(snprintf(runtime_root,sizeof(runtime_root),"/tmp/sparkpipe-missing-client-root-%ld",(long)getpid()) > 0);
	rmdir(runtime_root);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 100u;
	configuration.deployment = deployment;
	configuration.runtime_root = runtime_root;
	configuration.submit_result_function = TestModelPipelineResult;
	configuration.submit_result_context = state;
	configuration.completion_function = TestModelPipelineCompletion;
	configuration.completion_context = state;
	pipeline = 0;
	assert(SparkModelPipelineClientConnect(&configuration,&pipeline) == SPARK_STATUS_NOT_FOUND);
	assert(pipeline == 0);
}

static void TestModelPipelineWaitForCompletion(
	SparkModelPipelineClient *pipeline,
	const TestModelPipelineState *state,
	uint32_t completion_count)
{
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (attempt=0u; attempt<5000u && state->completion_count<completion_count; attempt++)
	{
		assert(SparkModelPipelineClientProgress(pipeline,8u) == SPARK_STATUS_OK);
		nanosleep(&delay,0);
	}
	assert(state->completion_count == completion_count);
}

static SparkStatus TestModelPipelineWaitForFailure(
	SparkModelPipelineClient *pipeline)
{
	struct timespec delay;
	SparkStatus status;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	status = SPARK_STATUS_OK;
	for (attempt=0u; attempt<5000u && status==SPARK_STATUS_OK; attempt++)
	{
		status = SparkModelPipelineClientProgress(pipeline,8u);
		if ( status == SPARK_STATUS_OK )
			nanosleep(&delay,0);
	}
	return(status);
}

static void TestModelPipelineWriteDeployment(
	const char *path,
	const SparkModelResidentEndpoint *endpoints)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_MODEL_PIPELINE_RANK_COUNT];
	uint32_t stage_indices[TEST_MODEL_PIPELINE_RANK_COUNT];
	char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	uint32_t rank;
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	for (rank=0u; rank<TEST_MODEL_PIPELINE_RANK_COUNT; rank++)
	{
		runtime_roots[rank] = runtime_root;
		stage_indices[rank] = rank;
	}
	stage_indices[1] = 2u;
	stage_indices[2] = 1u;
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "test.model.serving.target";
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestModelPipelineTransportHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 2u;
	fixture.runtime_limits.max_active_sequence_count = 2u;
	fixture.runtime_limits.max_input_row_count = 4u;
	fixture.runtime_limits.resident_sequence_capacity = 8u;
	fixture.control_port_base = 59000u;
	fixture.node_count = TEST_MODEL_PIPELINE_RANK_COUNT;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static SparkModelBatchEngine *TestModelBatchConnect(
	const SparkModelResidentDeployment *deployment,
	TestModelBatchState *state,
	uint32_t stop_token_count,
	uint32_t stop_token_id,
	uint32_t max_prefill_rows)
{
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine;
	char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 100u;
	configuration.request_capacity = 2u;
	configuration.max_context_tokens = 16u;
	configuration.max_prefill_rows_per_submission = max_prefill_rows;
	configuration.maximum_messages_per_rank_per_progress = 8u;
	configuration.stop_token_count = stop_token_count;
	configuration.stop_token_ids[0] = stop_token_id;
	configuration.deployment = deployment;
	configuration.runtime_root = runtime_root;
	configuration.event_function = TestModelBatchEvent;
	configuration.event_context = state;
	engine = 0;
	assert(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK);
	assert(engine != 0);
	return(engine);
}

static SparkModelBatchRequestHandle TestModelBatchSubmit(
	SparkModelBatchEngine *engine,
	uint64_t request_id,
	uint64_t sequence_id,
	const uint32_t *tokens,
	uint32_t token_count,
	uint32_t output_budget)
{
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES;
	request.priority = 10u;
	request.output_token_budget = output_budget;
	request.request_id = request_id;
	request.sequence_id = sequence_id;
	request.prompt_token_ids = tokens;
	request.prompt_token_count = token_count;
	handle = 0u;
	assert(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK);
	assert(handle != SPARK_MODEL_BATCH_ENGINE_INVALID_REQUEST_HANDLE);
	return(handle);
}

static void TestModelBatchWaitIdle(
	SparkModelBatchEngine *engine,
	uint32_t completed_count)
{
	SparkModelBatchEngineView view;
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (attempt=0u; attempt<10000u; attempt++)
	{
		assert(SparkModelBatchEngineProgress(engine,4u) == SPARK_STATUS_OK);
		assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
		if ( view.live_request_count == 0u && view.completed_request_count == completed_count )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "model batch engine did not drain");
}

static void TestModelBatchWaitForToken(
	SparkModelBatchEngine *engine,
	const TestModelBatchState *state)
{
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (attempt=0u; attempt<10000u && state->token_count == 0u; attempt++)
	{
		assert(SparkModelBatchEngineProgress(engine,1u) == SPARK_STATUS_OK);
		nanosleep(&delay,0);
	}
	assert(state->token_count == 1u);
}

static void TestModelBatchWaitShutdown(SparkModelBatchEngine *engine)
{
	SparkModelBatchEngineView view;
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (attempt=0u; attempt<10000u; attempt++)
	{
		assert(SparkModelBatchEngineProgress(engine,4u) == SPARK_STATUS_OK);
		assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
		if ( view.live_request_count == 0u && view.inflight_submission_count == 0u )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "model batch engine shutdown did not drain");
}

static void TestModelBatchEngineRun(
	const SparkModelResidentDeployment *deployment)
{
	SparkModelBatchEngineView view;
	SparkModelBatchRequestHandle cancelled,first,third;
	SparkModelBatchEngine *engine;
	TestModelBatchState state;
	uint32_t prompt_a[3] = {11u,12u,13u};
	uint32_t prompt_b[2] = {21u,22u};
	uint32_t prompt_c[1] = {31u};
	memset(&state,0,sizeof(state));
	engine = TestModelBatchConnect(deployment,&state,0u,0u,4u);
	first = TestModelBatchSubmit(engine,1001u,2001u,prompt_a,3u,2u);
	(void)TestModelBatchSubmit(engine,1002u,2002u,prompt_b,2u,2u);
	TestModelBatchWaitIdle(engine,2u);
	assert(state.accepted_count == 2u);
	assert(state.token_count == 4u);
	assert(state.completed_count == 2u);
	assert(state.cancelled_count == 0u);
	assert(state.error_count == 0u);
	third = TestModelBatchSubmit(engine,1003u,2003u,prompt_c,1u,1u);
	assert(third != first);
	TestModelBatchWaitIdle(engine,3u);
	assert(state.accepted_count == 3u);
	assert(state.token_count == 5u);
	assert(state.completed_count == 3u);
	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.live_request_count == 0u);
	assert(view.failed_status == SPARK_STATUS_OK);
	assert(view.pipeline.active_transaction_count == 0u);
	assert(view.pipeline.submitted_count >= 7u);
	cancelled = TestModelBatchSubmit(engine,1004u,2004u,prompt_c,1u,4u);
	assert(SparkModelBatchEngineCancel(engine,cancelled) == SPARK_STATUS_OK);
	assert(state.accepted_count == 4u);
	assert(state.cancelled_count == 1u);
	assert(SparkModelBatchEngineCancel(engine,cancelled) == SPARK_STATUS_NOT_FOUND);
	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.live_request_count == 0u);
	assert(view.cancelled_request_count == 1u);
	assert(SparkModelBatchEngineCloseAdmission(engine) == SPARK_STATUS_OK);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
	engine = TestModelBatchConnect(deployment,&state,1u,4203u,4u);
	(void)TestModelBatchSubmit(engine,1005u,2005u,prompt_c,1u,4u);
	TestModelBatchWaitIdle(engine,1u);
	assert(state.accepted_count == 1u);
	assert(state.token_count == 1u);
	assert(state.token_ids[0] == 4203u);
	assert(state.stop_token_count == 1u);
	assert(state.completed_count == 1u);
	assert(state.cancelled_count == 0u);
	assert(state.error_count == 0u);
	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.emitted_token_count == 1u);
	assert(view.pipeline.submitted_count == 2u);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
	engine = TestModelBatchConnect(deployment,&state,0u,0u,1u);
	(void)TestModelBatchSubmit(engine,1006u,2006u,prompt_c,1u,2u);
	(void)TestModelBatchSubmit(engine,1007u,2007u,prompt_c,1u,2u);
	TestModelBatchWaitIdle(engine,2u);
	assert(state.accepted_count == 2u);
	assert(state.token_count == 4u);
	assert(state.completed_count == 2u);
	assert(state.error_count == 0u);
	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.live_request_count == 0u);
	assert(view.failed_status == SPARK_STATUS_OK);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK);
}

static void TestModelBatchEngineShutdown(
	const SparkModelResidentDeployment *deployment)
{
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine;
	TestModelBatchState state;
	uint32_t prompt[1] = {41u};
	memset(&state,0,sizeof(state));
	engine = TestModelBatchConnect(deployment,&state,0u,0u,4u);
	(void)TestModelBatchSubmit(engine,1101u,2101u,prompt,1u,4u);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_BUSY);
	TestModelBatchWaitForToken(engine,&state);
	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.inflight_submission_count == 1u);
	assert(SparkModelBatchEngineBeginShutdown(engine) == SPARK_STATUS_PENDING);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_BUSY);
	TestModelBatchWaitShutdown(engine);
	assert(state.completed_count == 0u);
	assert(state.cancelled_count == 1u);
	assert(state.error_count == 0u);
	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.admission_open == 0u);
	assert(view.pipeline.submitted_count == 3u);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK);
}

static uint32_t TestModelBatchCountText(const char *text,const char *needle)
{
	uint32_t count;
	const char *cursor;
	count = 0u;
	cursor = text;
	while ( (cursor=strstr(cursor,needle)) != 0 )
	{
		count++;
		cursor += strlen(needle);
	}
	return(count);
}

static void TestModelBatchProcess(const char *deployment_path)
{
	char batch_path[108],output_path[108],output[16384],runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	FILE *file;
	pid_t child;
	size_t bytes;
	int32_t child_status;
	assert(snprintf(batch_path,sizeof(batch_path),"/tmp/sparkpipe-model-batch-%ld.json",(long)getpid()) > 0);
	assert(snprintf(output_path,sizeof(output_path),"/tmp/sparkpipe-model-batch-%ld.ndjson",(long)getpid()) > 0);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	file = fopen(batch_path,"wb");
	assert(file != 0);
	assert(fputs("{\"schema_version\":1,\"connect_timeout_ms\":100,\"request_capacity\":2,\"max_context_tokens\":16,\"max_prefill_rows_per_submission\":4,\"maximum_messages_per_rank_per_progress\":8,\"maximum_new_submissions_per_progress\":4,\"stop_token_ids\":[],\"requests\":[{\"request_id\":3101,\"sequence_id\":4101,\"priority\":10,\"output_token_budget\":2,\"prompt_token_ids\":[11,12]},{\"request_id\":3102,\"sequence_id\":4102,\"priority\":10,\"output_token_budget\":1,\"prompt_token_ids\":[21]}]}\n",file) != EOF);
	assert(fclose(file) == 0);
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		if ( freopen(output_path,"wb",stdout) == 0 )
			_exit(120);
		execl(TEST_MODEL_BATCH_PATH,TEST_MODEL_BATCH_PATH,"--deployment",deployment_path,"--runtime-root",runtime_root,"--batch",batch_path,(char *)0);
		_exit(121);
	}
	assert(waitpid(child,&child_status,0) == child);
	if ( WIFSIGNALED(child_status) )
		fprintf(stderr,"sparkpipe_model_batch terminated by signal %d\n",WTERMSIG(child_status));
	else if ( !WIFEXITED(child_status) )
		fprintf(stderr,"sparkpipe_model_batch returned unknown wait status %d\n",child_status);
	assert(WIFEXITED(child_status));
	assert(WEXITSTATUS(child_status) == 0);
	file = fopen(output_path,"rb");
	assert(file != 0);
	bytes = fread(output,1u,sizeof(output) - 1u,file);
	assert(feof(file));
	assert(fclose(file) == 0);
	output[bytes] = '\0';
	assert(TestModelBatchCountText(output,"\"event\":\"ready\"") == 1u);
	assert(TestModelBatchCountText(output,"\"event\":\"accepted\"") == 2u);
	assert(TestModelBatchCountText(output,"\"event\":\"token\"") == 3u);
	assert(TestModelBatchCountText(output,"\"event\":\"completed\"") == 2u);
	assert(TestModelBatchCountText(output,"\"event\":\"error\"") == 0u);
	unlink(output_path);
	unlink(batch_path);
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	SparkModelResidentEndpoint endpoints[TEST_MODEL_PIPELINE_RANK_COUNT];
	SparkModelResidentClientPollDescriptor poll_descriptors[TEST_MODEL_PIPELINE_RANK_COUNT];
	SparkModelPipelineClientView view;
	SparkModelServingSubmission submissions[9];
	SparkModelServingLane lanes[9][2];
	SparkModelPipelineClient *pipeline;
	TestModelPipelineState state;
	uint32_t tokens[9][4],row_lanes[9][4];
	uint64_t positions[9][4],sequences[9][4];
	uint8_t rejected_extension,missing_tokens_extension;
	char deployment_path[108];
	char paths[TEST_MODEL_PIPELINE_RANK_COUNT][108];
	pid_t children[TEST_MODEL_PIPELINE_RANK_COUNT];
	uint32_t descriptor_count,rank,tcp_port;
	int32_t child_status;
	tcp_port = 30000u + ((uint32_t)getpid() % 20000u);
	memset(&state,0,sizeof(state));
	memset(endpoints,0,sizeof(endpoints));
	for (rank=0u; rank<TEST_MODEL_PIPELINE_RANK_COUNT; rank++)
	{
		assert(snprintf(paths[rank],sizeof(paths[rank]),"/tmp/sparkpipe-pipeline-%ld-%u.sock",(long)getpid(),rank) > 0);
		unlink(paths[rank]);
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = rank == 0u ? SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP : SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
		endpoints[rank].tcp_port = rank == 0u ? tcp_port : 0u;
		endpoints[rank].tcp_host = rank == 0u ? "127.0.0.1" : 0;
		endpoints[rank].unix_socket_path = rank == 0u ? 0 : paths[rank];
	}
	assert(snprintf(deployment_path,sizeof(deployment_path),"/tmp/sparkpipe-pipeline-%ld.json",(long)getpid()) > 0);
	unlink(deployment_path);
	TestModelPipelineWriteDeployment(deployment_path,endpoints);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(deployment_path,&deployment) == SPARK_STATUS_OK);
	assert(SparkModelResidentDeploymentFindRank(&deployment,1u)->stage_index == 2u);
	assert(SparkModelResidentDeploymentFindStage(&deployment,1u)->rank_index == 2u);
	TestModelPipelineRejectMissingClientRuntimeRoot(&deployment,&state);
	for (rank=0u; rank<TEST_MODEL_PIPELINE_RANK_COUNT; rank++)
		children[rank] = TestModelPipelineStartResident(deployment_path,rank);
	TestModelPipelineWaitForSockets(paths);
	pipeline = TestModelPipelineConnect(&deployment,&state);
	state.pipeline = pipeline;
	assert(SparkModelPipelineClientGetPollDescriptors(pipeline,poll_descriptors,TEST_MODEL_PIPELINE_RANK_COUNT,&descriptor_count) == SPARK_STATUS_OK);
	assert(descriptor_count == TEST_MODEL_PIPELINE_RANK_COUNT);
	TestModelPipelineBuildSubmission(&submissions[0],lanes[0],tokens[0],row_lanes[0],positions[0],sequences[0],501u);
	TestModelPipelineBuildPrefill(&submissions[1],lanes[1],tokens[1],row_lanes[1],positions[1],sequences[1],502u);
	TestModelPipelineBuildSubmission(&submissions[2],lanes[2],tokens[2],row_lanes[2],positions[2],sequences[2],503u);
	state.callback_submission = &submissions[2];
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[0]) == SPARK_STATUS_OK);
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[1]) == SPARK_STATUS_OK);
	TestModelPipelineWaitForCompletion(pipeline,&state,3u);
	assert(state.callback_submitted == 1u);
	assert(state.callback_submit_status == SPARK_STATUS_OK);
	assert(state.result_count == 3u);
	assert(state.result_submission_ids[0] == 501u);
	assert(state.result_submission_ids[1] == 502u);
	assert(state.result_submission_ids[2] == 503u);
	assert(state.result_statuses[0] == SPARK_STATUS_OK);
	assert(state.result_statuses[1] == SPARK_STATUS_OK);
	assert(state.result_statuses[2] == SPARK_STATUS_OK);
	assert(state.completions[0].submission_id == 501u);
	assert(memcmp(&state.completions[0].residency,&submissions[0].residency,
		sizeof(submissions[0].residency)) == 0);
	assert(state.completions[0].token_count == 2u);
	assert(state.completions[0].token_ids[0] == 4200u);
	assert(state.completions[0].token_ids[1] == 4201u);
	assert(state.completions[1].submission_id == 502u);
	assert(state.completions[1].token_count == 1u);
	assert(state.completions[1].token_ids[0] == 4203u);
	assert(state.completions[2].submission_id == 503u);
	assert(state.completions[2].token_count == 2u);
	assert(state.completions[2].token_ids[0] == 4200u);
	assert(state.completions[2].token_ids[1] == 4201u);
	TestModelPipelineBuildPrefill(&submissions[3],lanes[3],tokens[3],row_lanes[3],positions[3],sequences[3],504u);
	TestModelPipelineRetargetPrefill(&submissions[3],&lanes[3][0],sequences[3],903u,201u);
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[3]) == SPARK_STATUS_OK);
	TestModelPipelineWaitForCompletion(pipeline,&state,4u);
	assert(state.result_count == 4u);
	assert(state.result_submission_ids[3] == 504u);
	assert(state.result_statuses[3] == SPARK_STATUS_INVALID_ARGUMENT);
	assert(state.completions[3].submission_id == 504u);
	assert(state.completions[3].status == SPARK_STATUS_INVALID_ARGUMENT);
	assert(state.completions[3].completion_flags == 0u);
	assert(state.completions[3].token_count == 0u);
	TestModelPipelineBuildRelease(&submissions[4],&lanes[4][0],505u);
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[4]) == SPARK_STATUS_OK);
	TestModelPipelineWaitForCompletion(pipeline,&state,5u);
	assert(state.result_count == 5u);
	assert(state.result_submission_ids[4] == 505u);
	assert(state.result_statuses[4] == SPARK_STATUS_OK);
	assert(state.completions[4].submission_id == 505u);
	assert(memcmp(&state.completions[4].residency,&submissions[4].residency,
		sizeof(submissions[4].residency)) == 0);
	assert(state.completions[4].completion_flags == 0u);
	assert(state.completions[4].token_count == 0u);
	TestModelPipelineBuildPrefill(&submissions[5],lanes[5],tokens[5],row_lanes[5],positions[5],sequences[5],506u);
	TestModelPipelineRetargetPrefill(&submissions[5],&lanes[5][0],sequences[5],903u,201u);
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[5]) == SPARK_STATUS_OK);
	TestModelPipelineWaitForCompletion(pipeline,&state,6u);
	assert(state.result_count == 6u);
	assert(state.result_submission_ids[5] == 506u);
	assert(state.result_statuses[5] == SPARK_STATUS_OK);
	assert(state.completions[5].submission_id == 506u);
	assert(state.completions[5].token_count == 1u);
	assert(state.completions[5].token_ids[0] == 4203u);
	rejected_extension = 1u;
	TestModelPipelineBuildSubmission(&submissions[6],lanes[6],tokens[6],row_lanes[6],positions[6],sequences[6],507u);
	submissions[6].model_extension_kind = 77u;
	submissions[6].model_extension_bytes = sizeof(rejected_extension);
	submissions[6].model_extension = &rejected_extension;
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[6]) == SPARK_STATUS_OK);
	TestModelPipelineWaitForCompletion(pipeline,&state,7u);
	assert(state.result_count == 7u);
	assert(state.result_submission_ids[6] == 507u);
	assert(state.result_statuses[6] == SPARK_STATUS_UNSUPPORTED);
	assert(state.completions[6].submission_id == 507u);
	assert(state.completions[6].status == SPARK_STATUS_UNSUPPORTED);
	assert(state.completions[6].completion_flags == 0u);
	TestModelPipelineBuildSubmission(&submissions[7],lanes[7],tokens[7],row_lanes[7],positions[7],sequences[7],508u);
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[7]) == SPARK_STATUS_OK);
	TestModelPipelineWaitForCompletion(pipeline,&state,8u);
	assert(state.result_count == 8u);
	assert(state.result_submission_ids[7] == 508u);
	assert(state.result_statuses[7] == SPARK_STATUS_OK);
	assert(state.completions[7].submission_id == 508u);
	assert(state.completions[7].token_count == 2u);
	assert(state.completions[7].token_ids[0] == 4200u);
	assert(state.completions[7].token_ids[1] == 4201u);
	assert(SparkModelPipelineClientGetView(pipeline,&view) == SPARK_STATUS_OK);
	assert(view.rank_count == TEST_MODEL_PIPELINE_RANK_COUNT);
	assert(view.connected_rank_count == TEST_MODEL_PIPELINE_RANK_COUNT);
	assert(view.active_transaction_count == 0u);
	assert(view.failed_status == SPARK_STATUS_OK);
	assert(view.submitted_count == 8u);
	assert(view.admitted_count == 6u);
	assert(view.rejected_count == 2u);
	assert(view.completed_count == 8u);
	missing_tokens_extension = 1u;
	TestModelPipelineBuildSubmission(&submissions[8],lanes[8],tokens[8],row_lanes[8],positions[8],sequences[8],509u);
	submissions[8].model_extension_kind = 88u;
	submissions[8].model_extension_bytes = sizeof(missing_tokens_extension);
	submissions[8].model_extension = &missing_tokens_extension;
	assert(SparkModelPipelineClientSubmit(pipeline,&submissions[8]) == SPARK_STATUS_OK);
	assert(TestModelPipelineWaitForFailure(pipeline) == SPARK_STATUS_SCHEMA_ERROR);
	assert(state.result_count == 9u);
	assert(state.result_submission_ids[8] == 509u);
	assert(state.result_statuses[8] == SPARK_STATUS_SCHEMA_ERROR);
	assert(state.completion_count == 9u);
	assert(state.completions[8].submission_id == 509u);
	assert(state.completions[8].status == SPARK_STATUS_SCHEMA_ERROR);
	assert(state.completions[8].completion_flags == 0u);
	assert(SparkModelPipelineClientGetView(pipeline,&view) == SPARK_STATUS_OK);
	assert(view.active_transaction_count == 0u);
	assert(view.failed_status == SPARK_STATUS_SCHEMA_ERROR);
	assert(view.submitted_count == 9u);
	assert(view.admitted_count == 6u);
	assert(view.rejected_count == 3u);
	assert(view.completed_count == 9u);
	SparkModelPipelineClientDestroy(pipeline);
	TestModelBatchEngineRun(&deployment);
	TestModelBatchEngineShutdown(&deployment);
	TestModelBatchProcess(deployment_path);
	for (rank=0u; rank<TEST_MODEL_PIPELINE_RANK_COUNT; rank++)
	{
		assert(kill(children[rank],SIGTERM) == 0);
		assert(waitpid(children[rank],&child_status,0) == children[rank]);
		assert(WIFEXITED(child_status));
		assert(WEXITSTATUS(child_status) == 0);
		unlink(paths[rank]);
	}
	SparkModelResidentDeploymentDestroy(&deployment);
	unlink(deployment_path);
	return(0);
}
