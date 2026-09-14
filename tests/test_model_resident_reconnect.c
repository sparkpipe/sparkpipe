#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
#define TEST_MODEL_RESIDENT_RECONNECT_ATTEMPTS 5000u

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

static void TestModelResidentBuildDecode(
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

int main(void)
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
	struct timespec delay;
	uint32_t tokens[2],row_lanes[2];
	uint64_t positions[2],sequences[2];
	const SparkModelResidentDeploymentNode *node;
	char deployment_path[108];
	char socket_path[108];
	char rank_path[108];
	pid_t child;
	uint32_t rank,attempt,status_count;
	int32_t child_status;
	SparkStatus status;
	delay.tv_sec = 0;
	delay.tv_nsec = 2000000;
	memset(endpoints,0,sizeof(endpoints));
	assert(snprintf(socket_path,sizeof(socket_path),
		"/tmp/sparkpipe-model-resident-d1-%ld-rank0.sock",
		(long)getpid()) > 0);
	assert(snprintf(deployment_path,sizeof(deployment_path),
		"/tmp/sparkpipe-model-resident-d1-%ld.json",
		(long)getpid()) > 0);
	for (rank=0u; rank<TEST_MODEL_RESIDENT_RANK_COUNT; rank++)
	{
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
		if ( rank == 0u )
		{
			endpoints[rank].unix_socket_path = socket_path;
			continue;
		}
		assert(snprintf(rank_path,sizeof(rank_path),"%s-%u",socket_path,rank) > 0);
		endpoints[rank].unix_socket_path = strdup(rank_path);
		assert(endpoints[rank].unix_socket_path != 0);
	}
	unlink(socket_path);
	TestModelResidentWriteDeployment(deployment_path,endpoints);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(deployment_path,&deployment) == SPARK_STATUS_OK);
	node = SparkModelResidentDeploymentFindRank(&deployment,0u);
	assert(node != 0);
	child = TestModelResidentStart(deployment_path,0u);
	TestModelResidentWaitForSocket(socket_path);
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_DSV4_SERVING_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT,&adapter) == SPARK_STATUS_OK);
	memset(&state,0,sizeof(state));
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_RESIDENT_CLIENT_CONFIGURATION_BYTES;
	configuration.rank_index = 0u;
	configuration.stage_index = 0u;
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
	TestModelResidentBuildDecode(&submission,lanes,tokens,row_lanes,positions,sequences,501u);
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,1u);
	assert(state.result_status == SPARK_STATUS_OK);
	assert(SparkModelResidentClientCommit(client,501u) == SPARK_STATUS_OK);
	TestModelResidentWaitForCompletion(client,&state,1u);
	assert(state.completion.submission_id == 501u);
	assert(kill(child,SIGKILL) == 0);
	assert(waitpid(child,&child_status,0) == child);
	assert(WIFSIGNALED(child_status) && WTERMSIG(child_status) == SIGKILL);
	status = SPARK_STATUS_OK;
	status_count = 0u;
	while ( status == SPARK_STATUS_OK && status_count < 100u )
	{
		status = SparkModelResidentClientProgress(client,8u);
		status_count++;
	}
	assert(status != SPARK_STATUS_OK);
	assert(SparkModelResidentClientGetView(client,&view) == SPARK_STATUS_OK);
	assert(view.connected == 0u);
	unlink(socket_path);
	child = TestModelResidentStart(deployment_path,0u);
	TestModelResidentWaitForSocket(socket_path);
	status = SPARK_STATUS_IO_ERROR;
	for (attempt=0u; attempt<TEST_MODEL_RESIDENT_RECONNECT_ATTEMPTS; attempt++)
	{
		status = SparkModelResidentClientProgress(client,8u);
		if ( status == SPARK_STATUS_OK )
			break;
		nanosleep(&delay,0);
	}
	assert(status == SPARK_STATUS_OK);
	assert(SparkModelResidentClientGetView(client,&view) == SPARK_STATUS_OK);
	assert(view.connected == 1u);
	TestModelResidentBuildDecode(&submission,lanes,tokens,row_lanes,positions,sequences,502u);
	assert(SparkModelResidentClientPrepare(client,&submission) == SPARK_STATUS_OK);
	TestModelResidentWaitForResult(client,&state,2u);
	assert(state.result_status == SPARK_STATUS_OK);
	assert(state.result_submission_id == 502u);
	assert(SparkModelResidentClientCommit(client,502u) == SPARK_STATUS_OK);
	TestModelResidentWaitForCompletion(client,&state,2u);
	assert(state.completion.submission_id == 502u);
	SparkModelResidentClientDestroy(client);
	SparkModelServingAdapterUnloadInterface(&adapter);
	assert(kill(child,SIGTERM) == 0);
	assert(waitpid(child,&child_status,0) == child);
	assert(WIFEXITED(child_status));
	assert(WEXITSTATUS(child_status) == 0);
	SparkModelResidentDeploymentDestroy(&deployment);
	for (rank=0u; rank<TEST_MODEL_RESIDENT_RANK_COUNT; rank++)
	{
		if ( rank == 0u )
		{
			unlink(socket_path);
			continue;
		}
		unlink(endpoints[rank].unix_socket_path);
		free((void *)endpoints[rank].unix_socket_path);
	}
	unlink(deployment_path);
	return(0);
}
