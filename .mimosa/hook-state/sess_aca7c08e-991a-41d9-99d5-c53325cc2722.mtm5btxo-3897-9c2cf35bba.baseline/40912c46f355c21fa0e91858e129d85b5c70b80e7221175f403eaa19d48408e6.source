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
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_resident_client.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_RESIDENTD_PATH
#define TEST_MODEL_RESIDENTD_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_ADAPTER_PATH
#define TEST_DSV4_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_DRIVER_PATH
#define TEST_DSV4_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_CONFIG_PATH
#define TEST_DSV4_SERVING_CONFIG_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

#define TEST_MODEL_RESIDENT_RANK_COUNT 13u

static const char *const TestModelResidentTransportHosts[
	TEST_MODEL_RESIDENT_RANK_COUNT] =
{
	"spark0","spark1","spark2","spark3","spark4","spark5","spark6",
	"spark7","spark8","spark9","sparka","sparkb","sparkc"
};

typedef struct TestModelResidentState
{
	uint32_t result_count;
	uint32_t completion_count;
	uint64_t result_submission_id;
	SparkStatus result_status;
	SparkModelServingCompletion completion;
} TestModelResidentState;

static void TestModelResidentResult(
	void *result_context,
	uint64_t submission_id,
	SparkStatus status)
{
	TestModelResidentState *state;
	state = (TestModelResidentState *)result_context;
	state->result_submission_id = submission_id;
	state->result_status = status;
	state->result_count++;
}

static void TestModelResidentCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestModelResidentState *state;
	state = (TestModelResidentState *)completion_context;
	state->completion = *completion;
	state->completion_count++;
}

static pid_t TestModelResidentStart(
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

static void TestModelResidentWaitForSocket(const char *socket_path)
{
	struct stat status;
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 10000000;
	for (attempt=0u; attempt<500u; attempt++)
	{
		if ( lstat(socket_path,&status) == 0 && S_ISSOCK(status.st_mode) )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "model resident socket did not become ready");
}

static void TestModelResidentBuildSubmission(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lanes,
	uint32_t *tokens,
	uint32_t *row_lanes,
	uint64_t *positions,
	uint64_t *sequences)
{
	memset(lanes,0,2u * sizeof(lanes[0]));
	lanes[0].request_id = 900u;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = 1u;
	lanes[0].sequence_id = 100u;
	lanes[0].resident_sequence_slot = 7u;
	lanes[0].context_token_count = 1u;
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lanes[1].request_id = 901u;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = 1u;
	lanes[1].sequence_id = 101u;
	lanes[1].resident_sequence_slot = 3u;
	lanes[1].context_token_count = 1u;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
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
	submission->tokens_per_sequence = 1u;
	submission->submission_id = 501u;
	submission->request_id = 9u;
	submission->sequence_id = 100u;
	submission->control_generation = 1u;
	submission->transaction_id = 1501u;
	submission->dispatch_generation = 2501u;
	submission->request_generation = 1u;
	submission->step_generation = 3501u;
	submission->residency.word0 = 501u;
	submission->residency.word1 = 601u;
	submission->residency.generation = 701u;
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

static void TestModelResidentBuildPrefill(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lanes,
	uint32_t *tokens,
	uint32_t *row_lanes,
	uint64_t *positions,
	uint64_t *sequences)
{
	uint32_t row;
	memset(lanes,0,2u * sizeof(lanes[0]));
	lanes[0].request_id = 902u;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = 2u;
	lanes[0].sequence_id = 200u;
	lanes[0].resident_sequence_slot = 5u;
	lanes[0].context_token_count = 4u;
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
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
	submission->tokens_per_sequence = 1u;
	submission->submission_id = 503u;
	submission->request_id = 10u;
	submission->sequence_id = 200u;
	submission->control_generation = 1u;
	submission->transaction_id = 1503u;
	submission->dispatch_generation = 2503u;
	submission->request_generation = 1u;
	submission->step_generation = 3503u;
	submission->residency.word0 = 503u;
	submission->residency.word1 = 603u;
	submission->residency.generation = 703u;
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

static void TestModelResidentBuildRelease(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lanes,
	const uint64_t *request_ids,
	const uint64_t *sequence_ids,
	const uint64_t *sequence_positions,
	const uint32_t *resident_slots,
	uint32_t lane_count,
	uint64_t submission_id)
{
	uint32_t lane;
	memset(lanes,0,lane_count * sizeof(lanes[0]));
	for (lane=0u; lane<lane_count; lane++)
	{
		lanes[lane].request_id = request_ids[lane];
		lanes[lane].request_generation = 1u;
		lanes[lane].step_generation = submission_id + 3000u;
		lanes[lane].sequence_id = sequence_ids[lane];
		lanes[lane].sequence_position = sequence_positions[lane];
		lanes[lane].resident_sequence_slot = resident_slots[lane];
		lanes[lane].context_token_count = sequence_positions[lane];
	}
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission->submission_id = submission_id;
	submission->request_id = request_ids[0];
	submission->sequence_id = sequence_ids[0];
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = submission_id + 2000u;
	submission->request_generation = 1u;
	submission->step_generation = submission_id + 3000u;
	submission->residency.word0 = submission_id;
	submission->residency.word1 = submission_id + 100u;
	submission->residency.generation = submission_id + 200u;
	submission->residency.owner = 13u;
	submission->active_sequence_count = lane_count;
	submission->lane_count = lane_count;
	submission->lanes = lanes;
}

static void TestModelResidentWaitForCompletion(
	SparkModelResidentClient *client,
	const TestModelResidentState *state,
	uint32_t completion_count)
{
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (attempt=0u; attempt<5000u && state->completion_count < completion_count; attempt++)
	{
		assert(SparkModelResidentClientProgress(client,8u) == SPARK_STATUS_OK);
		nanosleep(&delay,0);
	}
	assert(state->completion_count == completion_count);
}

static void TestModelResidentWaitForResult(
	SparkModelResidentClient *client,
	const TestModelResidentState *state,
	uint32_t result_count)
{
	struct timespec delay;
	uint32_t attempt;
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (attempt=0u; attempt<5000u && state->result_count < result_count; attempt++)
	{
		assert(SparkModelResidentClientProgress(client,8u) == SPARK_STATUS_OK);
		nanosleep(&delay,0);
	}
	assert(state->result_count == result_count);
}

static void TestModelResidentWriteDeployment(
	const char *path,
	const SparkModelResidentEndpoint *endpoints)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_MODEL_RESIDENT_RANK_COUNT];
	char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	uint32_t rank;
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	for (rank=0u; rank<TEST_MODEL_RESIDENT_RANK_COUNT; rank++)
		runtime_roots[rank] = runtime_root;
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = TEST_DSV4_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_DSV4_SERVING_DRIVER_PATH;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
	fixture.adapter_configuration_path = TEST_DSV4_SERVING_CONFIG_PATH;
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestModelResidentTransportHosts;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 2u;
	fixture.runtime_limits.max_active_sequence_count = 2u;
	fixture.runtime_limits.max_input_row_count = 4u;
	fixture.runtime_limits.resident_sequence_capacity = 8u;
	fixture.runtime_limits.kv_logical_page_capacity = 16u;
	fixture.runtime_limits.kv_physical_page_capacity = 8u;
	fixture.control_port_base = 59000u;
	fixture.node_count = TEST_MODEL_RESIDENT_RANK_COUNT;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static void TestModelResidentRunCase(
	uint32_t rank_index,
	uint32_t expect_tokens)
{
	SparkModelServingAdapterDynamicLibrary adapter;
	SparkModelResidentDeployment deployment;
	SparkModelResidentEndpoint endpoints[TEST_MODEL_RESIDENT_RANK_COUNT];
	SparkModelResidentClientConfiguration configuration;
	SparkModelResidentClientView view;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	SparkModelResidentClient *client;
	TestModelResidentState state;
	uint32_t tokens[4],row_lanes[4];
	uint32_t release_slots[2];
	uint64_t positions[4],sequences[4];
	uint64_t release_positions[2],release_request_ids[2];
	uint64_t release_sequence_ids[2];
	const SparkModelResidentDeploymentNode *node;
	char deployment_path[108];
	char socket_paths[TEST_MODEL_RESIDENT_RANK_COUNT][108];
	pid_t child;
	uint32_t rank;
	int32_t child_status;
	memset(endpoints,0,sizeof(endpoints));
	for (rank=0u; rank<TEST_MODEL_RESIDENT_RANK_COUNT; rank++)
	{
		assert(snprintf(socket_paths[rank],sizeof(socket_paths[rank]),"/tmp/sparkpipe-model-resident-%ld-%u-%u.sock",(long)getpid(),rank_index,rank) > 0);
		unlink(socket_paths[rank]);
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
		endpoints[rank].unix_socket_path = socket_paths[rank];
	}
	assert(snprintf(deployment_path,sizeof(deployment_path),"/tmp/sparkpipe-model-resident-%ld-%u.json",(long)getpid(),rank_index) > 0);
	unlink(deployment_path);
	TestModelResidentWriteDeployment(deployment_path,endpoints);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(deployment_path,&deployment) == SPARK_STATUS_OK);
	node = SparkModelResidentDeploymentFindRank(&deployment,rank_index);
	assert(node != 0);
	child = TestModelResidentStart(deployment_path,rank_index);
	TestModelResidentWaitForSocket(socket_paths[rank_index]);
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_DSV4_SERVING_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT,&adapter) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_RESIDENT_CLIENT_CONFIGURATION_BYTES;
	configuration.rank_index = rank_index;
	configuration.stage_index = rank_index;
	configuration.connect_timeout_ms = 1000u;
	configuration.runtime_limits = deployment.runtime_limits;
	configuration.endpoint = node->control_endpoint;
	configuration.adapter_descriptor = adapter.adapter_interface.descriptor;
	configuration.submit_result_function = TestModelResidentResult;
	configuration.submit_result_context = &state;
	configuration.completion_function = TestModelResidentCompletion;
	configuration.completion_context = &state;
	client = 0;
	assert(SparkModelResidentClientConnect(&configuration,&client) == SPARK_STATUS_OK);
	TestModelResidentBuildSubmission(&submission,lanes,tokens,row_lanes,positions,sequences);
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_OK);
	submission.submission_id = 502u;
	submission.transaction_id = 1502u;
	submission.dispatch_generation = 2502u;
	submission.step_generation = 3502u;
	submission.residency.word0 = 502u;
	submission.residency.word1 = 602u;
	submission.residency.generation = 702u;
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,2u);
	assert(state.result_count == 2u);
	assert(state.result_submission_id == 502u);
	assert(state.result_status == SPARK_STATUS_BUSY);
	assert(SparkModelResidentClientCommit(client,501u) == SPARK_STATUS_OK);
	TestModelResidentWaitForCompletion(client,&state,1u);
	assert(state.completion.submission_id == 501u);
	assert(state.completion.residency.word0 == 501u);
	assert(state.completion.residency.word1 == 601u);
	assert(state.completion.residency.generation == 701u);
	assert(state.completion.residency.owner == 13u);
	assert(state.completion.service_time_ns != 0u);
	assert(state.completion.completion_flags == (expect_tokens != 0u ? SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS : 0u));
	assert(state.completion.token_count == (expect_tokens != 0u ? 2u : 0u));
	if ( expect_tokens != 0u )
	{
		assert(state.completion.token_ids[0] == 4200u);
		assert(state.completion.token_ids[1] == 4201u);
	}
	TestModelResidentBuildPrefill(&submission,lanes,tokens,row_lanes,positions,sequences);
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,3u);
	assert(state.result_submission_id == 503u);
	assert(state.result_status == SPARK_STATUS_OK);
	assert(SparkModelResidentClientCommit(client,503u) == SPARK_STATUS_OK);
	TestModelResidentWaitForCompletion(client,&state,2u);
	assert(state.result_count == 3u);
	assert(state.result_submission_id == 503u);
	assert(state.result_status == SPARK_STATUS_OK);
	assert(state.completion.submission_id == 503u);
	assert(memcmp(&state.completion.residency,&submission.residency,
		sizeof(submission.residency)) == 0);
	assert(state.completion.completion_flags == (expect_tokens != 0u ? SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS : 0u));
	assert(state.completion.token_count == (expect_tokens != 0u ? 1u : 0u));
	if ( expect_tokens != 0u )
		assert(state.completion.token_ids[0] == 4203u);
	submission.submission_id = 504u;
	submission.transaction_id = 1504u;
	submission.dispatch_generation = 2504u;
	submission.step_generation = 3504u;
	positions[1] = 3u;
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,4u);
	assert(state.result_submission_id == 504u);
	assert(state.result_status == SPARK_STATUS_INVALID_ARGUMENT);
	assert(state.completion_count == 2u);
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	release_request_ids[0] = 900u;
	release_request_ids[1] = 901u;
	release_sequence_ids[0] = 100u;
	release_sequence_ids[1] = 101u;
	release_positions[0] = 1u;
	release_positions[1] = 1u;
	release_slots[0] = 7u;
	release_slots[1] = 3u;
	TestModelResidentBuildRelease(&submission,lanes,release_request_ids,
		release_sequence_ids,release_positions,release_slots,2u,505u);
	assert(SparkModelResidentClientContinue(client,&submission) ==
		SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,5u);
	TestModelResidentWaitForCompletion(client,&state,3u);
	release_request_ids[0] = 902u;
	release_sequence_ids[0] = 200u;
	release_positions[0] = 4u;
	release_slots[0] = 5u;
	TestModelResidentBuildRelease(&submission,lanes,release_request_ids,
		release_sequence_ids,release_positions,release_slots,1u,506u);
	assert(SparkModelResidentClientContinue(client,&submission) ==
		SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,6u);
	TestModelResidentWaitForCompletion(client,&state,4u);
	assert(SparkModelResidentClientGetView(client,&view) == SPARK_STATUS_OK);
	assert(view.connected == 1u);
	assert(view.pending_submission_count == 0u);
	assert(view.queue_capacity == 2u);
	assert(view.max_active_sequence_count == 2u);
	assert(view.max_input_row_count == 4u);
	assert(view.resident_sequence_capacity == 8u);
	assert(view.kv_logical_page_capacity == 16u);
	assert(view.kv_physical_page_capacity == 8u);
	assert(view.submitted_count == 6u);
	assert(view.admitted_count == 4u);
	assert(view.rejected_count == 2u);
	assert(view.continued_count == 2u);
	assert(view.completed_count == 4u);
	SparkModelResidentClientDestroy(client);
	SparkModelServingAdapterUnloadInterface(&adapter);
	assert(kill(child,SIGTERM) == 0);
	assert(waitpid(child,&child_status,0) == child);
	assert(WIFEXITED(child_status));
	assert(WEXITSTATUS(child_status) == 0);
	SparkModelResidentDeploymentDestroy(&deployment);
	for (rank=0u; rank<TEST_MODEL_RESIDENT_RANK_COUNT; rank++)
		unlink(socket_paths[rank]);
	unlink(deployment_path);
}

int main(void)
{
	TestModelResidentRunCase(0u,0u);
	TestModelResidentRunCase(12u,1u);
	TestModelResidentRunCase(6u,0u);
	return(0);
}
