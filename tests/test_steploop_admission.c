#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "sparkpipe/spark_model_batch_engine.h"

#ifndef TEST_MODEL_RESIDENTD_PATH
#define TEST_MODEL_RESIDENTD_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

/*
 * P1/D2 step-loop oracle (BUG_LEDGER PATTERN B): the serving loop must
 * feed an ASYNC_COMPLETION adapter at the adapter's own backpressure
 * (submit returns BUSY), not one frame per resident Progress pass.
 *
 * The batch below is sized so every request is dispatched in ONE engine
 * Progress (all frames ready simultaneously). Under the serialized
 * one-submit-per-pass bound the completions trickle at one per resident
 * wakeup (10 ms poll cap), so the drain needs many client round trips.
 * With adapter-contract admission the committed FIFO drains in one
 * resident pass and the drain needs only the protocol's own round
 * trips. The ceiling below pins the post-fix drain; the report carries
 * the pre-fix floor receipt that proves the bound discriminates.
 */
#define TEST_STEPLOOP_RANK_COUNT 3u
/* The fixture adapter's descriptor caps max_inflight_submission_count
 * at 4, and runtime limits must not exceed it — the batch is exactly
 * one full inflight window so every frame is ready simultaneously. */
#define TEST_STEPLOOP_REQUEST_COUNT 4u
#define TEST_STEPLOOP_DRAIN_CALL_CEILING 24u

static const char *const TestSteploopTransportHosts[TEST_STEPLOOP_RANK_COUNT] =
{
	"test-steploop-a","test-steploop-b","test-steploop-c"
};

typedef struct TestSteploopState
{
	uint32_t accepted_count;
	uint32_t token_count;
	uint32_t completed_count;
	uint32_t cancelled_count;
	uint32_t error_count;
	uint64_t token_request_ids[TEST_STEPLOOP_REQUEST_COUNT];
	uint32_t token_ids[TEST_STEPLOOP_REQUEST_COUNT];
} TestSteploopState;

static void TestSteploopEvent(
	void *event_context,
	const SparkModelBatchEvent *event)
{
	TestSteploopState *state;
	state = (TestSteploopState *)event_context;
	assert(event != 0);
	assert(event->abi_version == SPARK_MODEL_BATCH_ENGINE_ABI_VERSION);
	assert(event->descriptor_bytes == SPARK_MODEL_BATCH_EVENT_BYTES);
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED )
		state->accepted_count++;
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
	{
		if ( state->token_count < TEST_STEPLOOP_REQUEST_COUNT )
		{
			state->token_request_ids[state->token_count] = event->request_id;
			state->token_ids[state->token_count] = event->token_id;
		}
		state->token_count++;
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

static pid_t TestSteploopStartResident(
	const char *deployment_path,
	uint32_t rank_index,
	const char *stderr_path)
{
	pid_t child;
	char rank[16];
	assert(snprintf(rank,sizeof(rank),"%u",rank_index) > 0);
	child = fork();
	assert(child >= 0);
	if ( child == 0 )
	{
		if ( freopen(stderr_path,"wb",stderr) == 0 )
			_exit(126);
		execl(TEST_MODEL_RESIDENTD_PATH,TEST_MODEL_RESIDENTD_PATH,
			"--deployment",deployment_path,
			"--rank-index",rank,
			(char *)0);
		_exit(127);
	}
	return(child);
}

/* The step-loop receipt: the resident prints its adapter-ops-per-pass
 * maximum at exit. The serialized bound pins it to 1; adapter-contract
 * admission must show a drained FIFO (>= 2 in one pass). */
static uint32_t TestSteploopMaxOpsPerPass(
	const char *stderr_path,
	uint32_t rank)
{
	char line[512];
	uint32_t value;
	FILE *file;
	file = fopen(stderr_path,"rb");
	assert(file != 0);
	value = 0u;
	while ( fgets(line,sizeof(line),file) != 0 )
	{
		const char *marker;
		if ( strncmp(line,"model_residentd_exit ",21u) != 0 )
			continue;
		marker = strstr(line," max_ops_per_pass=");
		assert(marker != 0);
		if ( sscanf(marker," max_ops_per_pass=%u",&value) == 1 )
			break;
	}
	assert(fclose(file) == 0);
	fprintf(stderr,"test_steploop_admission rank=%u max_ops_per_pass=%u\n",rank,value);
	return(value);
}

static void TestSteploopWaitForSockets(char paths[][108])
{
	struct stat status;
	struct timespec delay;
	uint32_t attempt,rank,ready;
	delay.tv_sec = 0;
	delay.tv_nsec = 10000000;
	for (attempt=0u; attempt<500u; attempt++)
	{
		ready = 1u;
		for (rank=1u; rank<TEST_STEPLOOP_RANK_COUNT; rank++)
			if ( lstat(paths[rank],&status) != 0 || !S_ISSOCK(status.st_mode) )
				ready = 0u;
		if ( ready != 0u )
			return;
		nanosleep(&delay,0);
	}
	assert(0 && "steploop resident sockets did not become ready");
}

static void TestSteploopStopResidents(
	pid_t children[TEST_STEPLOOP_RANK_COUNT],
	char paths[][108])
{
	uint32_t rank;
	int32_t child_status;
	for (rank=0u; rank<TEST_STEPLOOP_RANK_COUNT; rank++)
	{
		assert(kill(children[rank],SIGTERM) == 0 || errno == ESRCH);
		assert(waitpid(children[rank],&child_status,0) == children[rank]);
		assert(WIFEXITED(child_status));
		assert((uint32_t)WEXITSTATUS(child_status) == 0u);
		unlink(paths[rank]);
	}
}

static uint32_t TestSteploopProbeFreeTcpPort(void)
{
	struct sockaddr_in address;
	socklen_t address_length;
	int32_t fd;
	uint32_t port;
	port = 0u;
	fd = socket(AF_INET,SOCK_STREAM,0);
	if ( fd >= 0 )
	{
		memset(&address,0,sizeof(address));
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl(0x7f000001u);
		address.sin_port = htons(0u);
		address_length = (socklen_t)sizeof(address);
		if ( bind(fd,(const struct sockaddr *)&address,(socklen_t)sizeof(address)) == 0 && getsockname(fd,(struct sockaddr *)&address,&address_length) == 0 )
			port = (uint32_t)ntohs(address.sin_port);
		close(fd);
	}
	return(port);
}

static void TestSteploopWriteDeployment(
	const char *path,
	const SparkModelResidentEndpoint *endpoints,
	const char *runtime_root)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_STEPLOOP_RANK_COUNT];
	uint32_t stage_indices[TEST_STEPLOOP_RANK_COUNT];
	uint32_t rank;
	for (rank=0u; rank<TEST_STEPLOOP_RANK_COUNT; rank++)
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
	fixture.node_target = "test.steploop.serving.target";
	/* Hold mode: submit enqueues, completions release on adapter
	 * progress — the async shape the fixture descriptor declares. The
	 * admission rate of the resident loop is then the only variable. */
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config_hold.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestSteploopTransportHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	/* The whole batch must fit the inflight window in ONE dispatch: this
	 * is what makes the resident-side admission rate the observable. */
	/* The whole batch must fit the inflight window in ONE dispatch, and
	 * one lane per frame (max_active_sequence_count=1, the B1 serving
	 * pattern) so the four requests are FOUR routes on the resident —
	 * a committed FIFO deeper than one is the observable. */
	fixture.runtime_limits.max_inflight_submission_count = TEST_STEPLOOP_REQUEST_COUNT;
	fixture.runtime_limits.max_active_sequence_count = 1u;
	fixture.runtime_limits.max_input_row_count = 32u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 128u;
	fixture.runtime_limits.kv_physical_page_capacity = 32u;
	fixture.control_port_base = TestSteploopProbeFreeTcpPort();
	if ( fixture.control_port_base == 0u || fixture.control_port_base > UINT16_MAX - (TEST_STEPLOOP_RANK_COUNT - 1u) )
		fixture.control_port_base = 58000u;
	fixture.node_count = TEST_STEPLOOP_RANK_COUNT;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

int main(void)
{
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine;
	SparkModelResidentDeployment deployment;
	SparkModelResidentEndpoint endpoints[TEST_STEPLOOP_RANK_COUNT];
	TestSteploopState state;
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle;
	uint32_t prompt[TEST_STEPLOOP_REQUEST_COUNT][1];
	uint32_t drain_calls,index;
	struct timespec delay;
	char deployment_path[108];
	char runtime_root[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	char stderr_paths[TEST_STEPLOOP_RANK_COUNT][108];
	char paths[TEST_STEPLOOP_RANK_COUNT][108];
	pid_t children[TEST_STEPLOOP_RANK_COUNT];
	uint32_t tcp_port;

	tcp_port = TestSteploopProbeFreeTcpPort();
	if ( tcp_port == 0u )
		tcp_port = 30000u + ((uint32_t)getpid() % 20000u);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	memset(&state,0,sizeof(state));
	memset(endpoints,0,sizeof(endpoints));
	for (index=0u; index<TEST_STEPLOOP_RANK_COUNT; index++)
	{
		assert(snprintf(paths[index],sizeof(paths[index]),"/tmp/sparkpipe-steploop-%ld-%u.sock",(long)getpid(),index) > 0);
		unlink(paths[index]);
		endpoints[index].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[index].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[index].kind = index == 0u ? SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP : SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
		endpoints[index].tcp_port = index == 0u ? tcp_port : 0u;
		endpoints[index].tcp_host = index == 0u ? "127.0.0.1" : 0;
		endpoints[index].unix_socket_path = index == 0u ? 0 : paths[index];
	}
	assert(snprintf(deployment_path,sizeof(deployment_path),"/tmp/sparkpipe-steploop-%ld.json",(long)getpid()) > 0);
	unlink(deployment_path);
	TestSteploopWriteDeployment(deployment_path,endpoints,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad(deployment_path,&deployment) == SPARK_STATUS_OK);
	for (index=0u; index<TEST_STEPLOOP_RANK_COUNT; index++)
	{
		char stderr_path[108];
		assert(snprintf(stderr_path,sizeof(stderr_path),"/tmp/sparkpipe-steploop-%ld-%u.stderr",(long)getpid(),index) > 0);
		unlink(stderr_path);
		children[index] = TestSteploopStartResident(deployment_path,index,stderr_path);
		assert(snprintf(stderr_paths[index],sizeof(stderr_paths[index]),"%s",stderr_path) > 0);
	}
	TestSteploopWaitForSockets(paths);

	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 100u;
	configuration.request_capacity = TEST_STEPLOOP_REQUEST_COUNT;
	configuration.max_context_tokens = 16u;
	configuration.max_prefill_rows_per_submission = 16u;
	configuration.maximum_messages_per_rank_per_progress = 16u;
	configuration.deployment = &deployment;
	configuration.runtime_root = runtime_root;
	configuration.event_function = TestSteploopEvent;
	configuration.event_context = &state;
	assert(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK);
	assert(engine != 0);

	for (index=0u; index<TEST_STEPLOOP_REQUEST_COUNT; index++)
	{
		memset(&request,0,sizeof(request));
		request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
		request.descriptor_bytes = SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES;
		request.priority = 10u;
		request.output_token_budget = 1u;
		request.request_id = 5100u + index;
		request.sequence_id = 6100u + index;
		prompt[index][0] = 71u + index;
		request.prompt_token_ids = prompt[index];
		request.prompt_token_count = 1u;
		handle = 0u;
		assert(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK);
		assert(handle != SPARK_MODEL_BATCH_ENGINE_INVALID_REQUEST_HANDLE);
	}
	assert(SparkModelBatchEngineCloseAdmission(engine) == SPARK_STATUS_OK);

	/* The oracle: one dispatching Progress, then count the client round
	 * trips until every request reaches terminal (informational — the
	 * protocol's own round trips dominate), and the resident's receipt
	 * of adapter ops per Progress pass, which is the pinned invariant. */
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	assert(SparkModelBatchEngineProgress(engine,TEST_STEPLOOP_REQUEST_COUNT) == SPARK_STATUS_OK);
	drain_calls = 1u;
	while ( state.completed_count < TEST_STEPLOOP_REQUEST_COUNT )
	{
		assert(SparkModelBatchEngineProgress(engine,TEST_STEPLOOP_REQUEST_COUNT) == SPARK_STATUS_OK);
		drain_calls++;
		assert(drain_calls <= TEST_STEPLOOP_DRAIN_CALL_CEILING);
		nanosleep(&delay,0);
	}

	assert(state.accepted_count == TEST_STEPLOOP_REQUEST_COUNT);
	assert(state.token_count == TEST_STEPLOOP_REQUEST_COUNT);
	assert(state.completed_count == TEST_STEPLOOP_REQUEST_COUNT);
	assert(state.cancelled_count == 0u);
	assert(state.error_count == 0u);
	/* FIFO order survives the drain: oldest request emits first. */
	for (index=0u; index<TEST_STEPLOOP_REQUEST_COUNT; index++)
		assert(state.token_request_ids[index] == 5100u + index);

	assert(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK);
	assert(view.live_request_count == 0u);
	assert(view.failed_status == SPARK_STATUS_OK);
	assert(view.pipeline.active_transaction_count == 0u);
	assert(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK);
	fprintf(stderr,"test_steploop_admission drain_calls=%u ceiling=%u\n",
		drain_calls,TEST_STEPLOOP_DRAIN_CALL_CEILING);

	TestSteploopStopResidents(children,paths);
	/* THE pinned invariant: under adapter-contract admission the first
	 * stage sees a committed FIFO deeper than one and drains it in a
	 * single Progress pass. The serialized one-op-per-pass bound
	 * (pre-redesign behavior) measures exactly 1 here. */
	for (index=0u; index<TEST_STEPLOOP_RANK_COUNT; index++)
		assert(TestSteploopMaxOpsPerPass(stderr_paths[index],index) >= 2u);
	SparkModelResidentDeploymentDestroy(&deployment);
	unlink(deployment_path);
	return(0);
}
