#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "mock_model_resident_client.h"
#include "sparkpipe/spark_speculation_frame.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_PATH
#define TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_SPECULATIVE_INLINE_PATH
#define TEST_MODEL_SERVING_SPECULATIVE_INLINE_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_PATH
#define TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_SPECULATIVE_VERIFY_PATH
#define TEST_MODEL_SERVING_SPECULATIVE_VERIFY_PATH ""
#endif
#ifndef TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_PATH
#define TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_PATH ""
#endif

#define TEST_RANKS 3u
#define TEST_MAX_REQUESTS 8u

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

typedef struct TestBatchState
{
	uint32_t token_events[TEST_MAX_REQUESTS + 1u];
	uint32_t completed_events[TEST_MAX_REQUESTS + 1u];
	uint32_t error_events[TEST_MAX_REQUESTS + 1u];
	uint32_t cancelled_events[TEST_MAX_REQUESTS + 1u];
	uint32_t total_terminals;
	uint32_t cached_tokens[TEST_MAX_REQUESTS + 1u];
	uint32_t stale_recomputes[TEST_MAX_REQUESTS + 1u];
	uint64_t accepted_ns[TEST_MAX_REQUESTS + 1u];
	uint64_t first_dispatch_ns[TEST_MAX_REQUESTS + 1u];
	uint64_t first_token_ns[TEST_MAX_REQUESTS + 1u];
	uint32_t token_event_order;
	uint32_t first_token_order[TEST_MAX_REQUESTS + 1u];
	uint32_t second_token_order[TEST_MAX_REQUESTS + 1u];
	uint32_t logprob_matches[TEST_MAX_REQUESTS + 1u];
	uint32_t logprob_events[TEST_MAX_REQUESTS + 1u];
	uint32_t token_ids[TEST_MAX_REQUESTS + 1u][128];
} TestBatchState;

static void TestBatchEvent(void *context, const SparkModelBatchEvent *event)
{
	TestBatchState *s = (TestBatchState *)context;
	uint64_t id = event->request_id;
	if ( id > TEST_MAX_REQUESTS )
		return;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED )
		s->accepted_ns[id] = event->monotonic_ns;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
	{
		s->token_event_order++;
		if ( s->token_events[id] == 0u )
			s->first_token_order[id] = s->token_event_order;
		if ( s->token_events[id] == 1u )
			s->second_token_order[id] = s->token_event_order;
		if ( s->token_events[id] < 128u )
			s->token_ids[id][s->token_events[id]] = event->token_id;
		if ( s->token_events[id]++ == 0u )
			s->first_token_ns[id] = event->monotonic_ns;
		s->cached_tokens[id] = event->cached_prompt_token_count;
		s->first_dispatch_ns[id] = event->first_dispatch_ns;
		if ( event->logprob_count != 0u )
			s->logprob_events[id]++;
		if ( event->logprob_count == 3u && event->logprobs[0].token == event->token_id && event->logprobs[0].logprob == -0.25f &&
			event->logprobs[1].token == 1001u && event->logprobs[1].logprob == -0.5f && event->logprobs[2].token == 1002u )
			s->logprob_matches[id]++;
	}
	s->stale_recomputes[id] = event->stale_prefix_recompute_count;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
	{
		s->completed_events[id]++;
		s->total_terminals++;
	}
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED )
		s->cancelled_events[id]++;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED ||
	     event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
	{
		s->error_events[id]++;
		s->total_terminals++;
	}
}

static const char *const TestTransportHosts[TEST_RANKS] =
{
	"mock-stage-a","mock-stage-b","mock-stage-c"
};

static void TestWriteDeployment(const char *path, const char *runtime_root, const char *adapter_path)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_RANKS];
	uint32_t stage_indices[TEST_RANKS];
	SparkModelResidentEndpoint endpoints[TEST_RANKS];
	uint32_t rank;
	for (rank=0u; rank<TEST_RANKS; rank++)
	{
		runtime_roots[rank] = runtime_root;
		stage_indices[rank] = rank;
		memset(&endpoints[rank],0,sizeof(endpoints[rank]));
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
		endpoints[rank].tcp_host = TestTransportHosts[rank];
		endpoints[rank].tcp_port = (uint32_t)(59100u + rank);
	}
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = adapter_path;
	fixture.driver_shared_object_path = adapter_path;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "test.model.serving.target";
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestTransportHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 4u;
	fixture.runtime_limits.max_active_sequence_count = 8u;
	fixture.runtime_limits.max_input_row_count = 256u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.control_port_base = 59200u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static uint32_t TestKeepPrefixIndex;
static const char *TestPeerPrefixIndex;

static const char *TestPrefixIndexPath(void)
{
	static char path[1024];
	char directory[900];
	if ( path[0] == '\0' )
	{
		assert(getcwd(directory,sizeof(directory)) != 0);
		(void)snprintf(path,sizeof(path),"%s/build/engine-mock-%d",directory,(int)getpid());
		(void)mkdir(path,0700);
		(void)snprintf(path,sizeof(path),"%s/build/engine-mock-%d/prefix_index.spi",directory,(int)getpid());
	}
	return(path);
}

static void TestConfigure(SparkModelBatchEngineConfiguration *configuration,const SparkModelResidentDeployment *deployment,TestBatchState *state,const char *runtime_root,uint32_t prefill_rows,uint32_t request_capacity,uint32_t flags)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration->descriptor_bytes = sizeof(*configuration);
	configuration->flags = flags;
	configuration->connect_timeout_ms = 1000u;
	configuration->request_capacity = request_capacity;
	configuration->max_context_tokens = 256u;
	configuration->max_prefill_rows_per_submission = prefill_rows;
	configuration->maximum_messages_per_rank_per_progress = 8u;
	configuration->inflight_budget_ns = SPARK_MODEL_BATCH_ENGINE_DEFAULT_INFLIGHT_BUDGET_NS;
	configuration->deployment = deployment;
	configuration->runtime_root = runtime_root;
	configuration->prefix_index_path = TestPrefixIndexPath();
	if ( TestKeepPrefixIndex == 0u )
		(void)unlink(configuration->prefix_index_path);
	if ( TestPeerPrefixIndex != 0 )
	{
		configuration->peer_prefix_index_paths = &TestPeerPrefixIndex;
		configuration->peer_prefix_index_count = 1u;
	}
	configuration->event_function = TestBatchEvent;
	configuration->event_context = state;
}

static SparkModelBatchEngine *TestConnectFlags(const SparkModelResidentDeployment *deployment,TestBatchState *state,const char *runtime_root,uint32_t prefill_rows,uint32_t request_capacity,uint32_t flags)
{
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine;
	SparkStatus status;
	TestConfigure(&configuration,deployment,state,runtime_root,prefill_rows,request_capacity,flags);
	engine = 0;
	status = SparkModelBatchEngineConnect(&configuration,&engine);
	CHECK(status == SPARK_STATUS_OK, "batch engine connect");
	return(engine);
}

static SparkModelBatchEngine *TestConnectRows(const SparkModelResidentDeployment *deployment,TestBatchState *state,const char *runtime_root,uint32_t prefill_rows)
{
	return(TestConnectFlags(deployment,state,runtime_root,prefill_rows,8u,0u));
}

static SparkModelBatchEngine *TestConnect(const SparkModelResidentDeployment *deployment,TestBatchState *state,const char *runtime_root)
{
	return(TestConnectRows(deployment,state,runtime_root,4u));
}

static SparkStatus TestSubmitPromptStatus(SparkModelBatchEngine *engine,uint64_t request_id,uint64_t sequence_id,uint32_t budget,const uint32_t *prompt,uint32_t prompt_count)
{
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.top_p = 1.0f;
	request.request_id = request_id;
	request.sequence_id = sequence_id;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = prompt_count;
	request.output_token_budget = budget;
	handle = 0;
	return(SparkModelBatchEngineSubmit(engine,&request,&handle));
}

static void TestSubmitPrompt(SparkModelBatchEngine *engine, uint64_t request_id, uint64_t sequence_id, uint32_t budget,const uint32_t *prompt,uint32_t prompt_count)
{
	CHECK(TestSubmitPromptStatus(engine,request_id,sequence_id,budget,prompt,prompt_count) == SPARK_STATUS_OK, "submit request");
}

static void TestSubmit(SparkModelBatchEngine *engine, uint64_t request_id, uint64_t sequence_id, uint32_t budget)
{
	static const uint32_t prompt[4] = {11u,12u,13u,14u};
	TestSubmitPrompt(engine,request_id,sequence_id,budget,prompt,4u);
}

/* One drive step: engine progress (which pumps the pipeline and the mock
 * residents' reconnect logic), then the mock answers everything in flight.
 * The 1ms pause lets the engine's busy-retry backoff elapse on failure
 * scenarios without making the happy path slow. */
static void TestDrive(SparkModelBatchEngine *engine, uint32_t steps)
{
	uint32_t step;
	for (step=0u; step<steps; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
}

static void TestDriveUntilTerminal(SparkModelBatchEngine *engine, TestBatchState *state, uint32_t terminals, uint32_t max_steps)
{
	uint32_t step;
	for (step=0u; step<max_steps && state->total_terminals < terminals; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
}

static void TestScenarioHappyPath(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,2u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK( state.token_events[1] != 0u,
		"happy: the request produced tokens through the full stack");
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"happy: the request completed cleanly");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioDeploymentPositionLimit(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[4] = {11u,12u,13u,14u};
	SparkModelResidentDeployment limited;
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	limited = *deployment;
	limited.max_sequence_positions = 8u;
	engine = TestConnect(&limited,&state,runtime_root);
	if ( engine == 0 )
		return;
	CHECK(TestSubmitPromptStatus(engine,2u,502u,5u,prompt,4u) == SPARK_STATUS_CAPACITY_EXCEEDED,"positions: a budget past the deployment's positions is refused at admission");
	CHECK(TestSubmitPromptStatus(engine,3u,503u,4u,prompt,4u) == SPARK_STATUS_OK,"positions: a budget ending at the deployment's positions is admitted");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDriveUntilTerminal(engine,&state,1u,800u);
	CHECK(state.completed_events[3] == 1u && state.error_events[3] == 0u && state.token_events[3] == 4u,"positions: the admitted request completes with its whole budget");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioRankDiesMidDecode(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	uint32_t step,tokens_before;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,4u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	for (step=0u; step<400u && state.token_events[1] == 0u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
	CHECK( state.token_events[1] != 0u, "chaos: prefill produced a token before the kill");
	tokens_before = state.token_events[1];
	MockResidentClientDisconnect(1u);
	TestDriveUntilTerminal(engine,&state,1u,2000u);
	CHECK( state.completed_events[1] == 0u && state.error_events[1] == 1u,
		"disconnect: started stream fails exactly once");
	CHECK( state.token_events[1] == tokens_before,
		"disconnect: emitted tokens are never replayed");
	{
		SparkModelBatchEngineView view;
		CHECK( SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK &&
			view.inflight_submission_count == 0u &&
			view.pipeline.active_transaction_count == 0u,
			"disconnect: old submissions and transactions retire before reuse");
	}
	TestSubmit(engine,2u,501u,2u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK( state.completed_events[2] == 1u && state.error_events[2] == 0u,
		"chaos: a fresh request completes after the recovery");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestWaitLane(SparkModelBatchEngine *engine,uint64_t request_id,uint64_t position,SparkModelServingLane *lane)
{
	for (uint32_t step = 0u; step < 2000u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		if ( MockResidentClientLastLane(0u,lane) != 0u &&
		     lane->request_id == request_id && lane->sequence_position == position )
			return(1u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
	return(0u);
}

static void TestScenarioCachedPrefixSessionReset(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane canonical = {0},cached = {0},rebuilt = {0};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,500u,1u);
	CHECK(TestWaitLane(engine,1u,0u,&canonical) != 0u &&
		canonical.cache_publish_token_count == 4u,
		"prefix reset: capture canonical first block identity");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u,"prefix reset: warm request completes");
	TestSubmitPrompt(engine,2u,501u,1u,prompt,8u);
	CHECK(TestWaitLane(engine,2u,4u,&cached) != 0u &&
		cached.cache_prefix_token_count == 4u &&
		(cached.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u,
		"prefix reset: next request actually uses cached prefix");
	CHECK(state.token_events[2] == 0u,"prefix reset: session dies before emitted token");
	MockResidentClientDisconnect(1u);
	CHECK(TestWaitLane(engine,2u,4u,&rebuilt) != 0u &&
		rebuilt.cache_prefix_token_count == 4u && (rebuilt.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u,
		"prefix reset: the kept prefix index sends the first block as a prefix after the session change");
	CHECK(memcmp(&canonical.cache_publish_identity,&rebuilt.cache_prefix_identity,
		sizeof(canonical.cache_publish_identity)) == 0,
		"prefix reset: the kept prefix identity equals the canonical prompt digest");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u &&
		state.token_events[2] == 1u,"prefix reset: recovered request completes exactly once");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioPrefixIndexSurvivesRestart(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane canonical = {0},restored = {0};
	struct stat info;
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,500u,1u);
	CHECK(TestWaitLane(engine,1u,0u,&canonical) != 0u && canonical.cache_publish_token_count == 4u,"index restart: the first engine publishes the first block");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u,"index restart: the first engine completes the warm request");
	CHECK(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK,"index restart: the first engine is destroyed");
	CHECK(stat(TestPrefixIndexPath(),&info) == 0 && info.st_size > 0,"index restart: destroy wrote the prefix index file");
	memset(&state,0,sizeof(state));
	MockResidentClientReset();
	TestKeepPrefixIndex = 1u;
	engine = TestConnect(deployment,&state,runtime_root);
	TestKeepPrefixIndex = 0u;
	if ( engine == 0 )
		return;
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.prefix_index_loaded_record_count == 1u && view.prefix_index_refused_count == 0u,"index restart: the second engine loads one record");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,2u,501u,1u,prompt,8u);
	CHECK(TestWaitLane(engine,2u,4u,&restored) != 0u && restored.cache_prefix_token_count == 4u &&
		(restored.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u &&
		memcmp(&canonical.cache_publish_identity,&restored.cache_prefix_identity,sizeof(canonical.cache_publish_identity)) == 0,
		"index restart: the restarted engine sends the saved block as a prefix lane");
	{
		uint32_t rank,hints = 0u;
		for (rank=0u; rank<TEST_RANKS; rank++)
			hints += MockResidentClientCalls(rank,MOCK_CALL_CACHE_HINT) == 1u ? 1u : 0u;
		CHECK(hints == TEST_RANKS,"index restart: the engine hints the queued prefix to every rank once so restores start before dispatch");
	}
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 4u,"index restart: the request completes with four cached prompt tokens");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioPeerPrefixIndex(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	static char peer[512];
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane canonical = {0},imported = {0};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,600u,1u);
	CHECK(TestWaitLane(engine,1u,0u,&canonical) != 0u && canonical.cache_publish_token_count == 4u,"peer index: the peer engine publishes the first block");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(SparkModelBatchEngineDestroy(engine) == SPARK_STATUS_OK,"peer index: the peer engine is destroyed and saves its index");
	(void)snprintf(peer,sizeof(peer),"%s.peer",TestPrefixIndexPath());
	CHECK(rename(TestPrefixIndexPath(),peer) == 0,"peer index: the peer's index file moves to its own path");
	memset(&state,0,sizeof(state));
	MockResidentClientReset();
	{
		SparkModelBatchEngineConfiguration configuration;
		TestPeerPrefixIndex = TestPrefixIndexPath();
		TestConfigure(&configuration,deployment,&state,runtime_root,4u,8u,0u);
		engine = 0;
		CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_INVALID_ARGUMENT && engine == 0,"peer index: an engine refuses its own index as a peer");
	}
	TestPeerPrefixIndex = peer;
	engine = TestConnect(deployment,&state,runtime_root);
	TestPeerPrefixIndex = 0;
	if ( engine == 0 )
		return;
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.prefix_index_loaded_record_count == 0u,"peer index: the engine's own index is empty");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,2u,601u,1u,prompt,8u);
	CHECK(TestWaitLane(engine,2u,4u,&imported) != 0u && SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK &&
		view.peer_prefix_imported_record_count == 1u && view.peer_prefix_import_count == 1u,"peer index: the engine imports the peer's committed block before it schedules the prompt");
	CHECK(imported.cache_prefix_token_count == 4u &&
		(imported.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u &&
		memcmp(&canonical.cache_publish_identity,&imported.cache_prefix_identity,sizeof(canonical.cache_publish_identity)) == 0,
		"peer index: the engine sends the peer's block as a prefix lane, for the shared KV window to serve");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 4u,"peer index: the request completes with four cached prompt tokens");
	SparkModelBatchEngineDestroy(engine);
	(void)unlink(peer);
}

static void TestScenarioPrefixIndexRemovesStaleTemporaries(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	char stale[4200],live[4200];
	FILE *file;
	pid_t child = fork();
	if ( child == 0 )
		_exit(0);
	CHECK(child > 0 && waitpid(child,0,0) == child,"index temporaries: a finished writer process exists");
	(void)snprintf(stale,sizeof(stale),"%s.tmp-%d",TestPrefixIndexPath(),(int)child);
	(void)snprintf(live,sizeof(live),"%s.tmp-%d",TestPrefixIndexPath(),(int)getppid());
	file = fopen(stale,"w");
	CHECK(file != 0 && fputs("partial",file) >= 0 && fclose(file) == 0,"index temporaries: a dead writer's temporary is written");
	file = fopen(live,"w");
	CHECK(file != 0 && fputs("partial",file) >= 0 && fclose(file) == 0,"index temporaries: a live writer's temporary is written");
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	CHECK(access(stale,F_OK) != 0 && access(live,F_OK) == 0,"index temporaries: start removes a dead writer's temporary and keeps a live writer's");
	SparkModelBatchEngineDestroy(engine);
	(void)unlink(live);
}

static void TestScenarioPrefixIndexRefusesCorruptFile(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane lane = {0};
	uint8_t byte = 0u;
	FILE *file;
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,500u,1u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	SparkModelBatchEngineDestroy(engine);
	file = fopen(TestPrefixIndexPath(),"r+b");
	CHECK(file != 0,"index corrupt: the index file exists");
	if ( file == 0 )
		return;
	CHECK(fseek(file,-1L,SEEK_END) == 0 && fread(&byte,1u,1u,file) == 1u,"index corrupt: read the last record byte");
	byte ^= 0x5au;
	CHECK(fseek(file,-1L,SEEK_END) == 0 && fwrite(&byte,1u,1u,file) == 1u,"index corrupt: flip the last record byte");
	fclose(file);
	memset(&state,0,sizeof(state));
	MockResidentClientReset();
	TestKeepPrefixIndex = 1u;
	engine = TestConnect(deployment,&state,runtime_root);
	TestKeepPrefixIndex = 0u;
	if ( engine == 0 )
		return;
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.prefix_index_refused_count == 1u && view.prefix_index_loaded_record_count == 0u,"index corrupt: the digest mismatch refuses the file");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,2u,501u,1u);
	CHECK(TestWaitLane(engine,2u,0u,&lane) != 0u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) == 0u,"index corrupt: no prefix lane comes from a refused file");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u,"index corrupt: the request completes");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioStatusReport(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingAdapterSnapshot snapshot;
	SparkModelResidentStatusReport report;
	uint32_t step;
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	(void)SparkModelBatchEngineProgress(engine,1u);
	CHECK(MockResidentClientCalls(TEST_RANKS - 1u,MOCK_CALL_STATUS_REQUEST) == 0u,"status: no status request before the first interval");
	CHECK(SparkModelBatchEngineNextProgressNs(engine) == 0u,"status: polling adds no wake-up deadline");
	for (step=0u; step<120u && MockResidentClientCalls(TEST_RANKS - 1u,MOCK_CALL_STATUS_REQUEST) == 0u; step++)
	{
		struct timespec delay = {0,10000000};
		(void)nanosleep(&delay,0);
		(void)SparkModelBatchEngineProgress(engine,1u);
	}
	CHECK(MockResidentClientCalls(TEST_RANKS - 1u,MOCK_CALL_STATUS_REQUEST) > 0u,"status: progress requests rank status after one interval");
	memset(&snapshot,0,sizeof(snapshot));
	snapshot.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	snapshot.kv_store.attached = 1u;
	snapshot.kv_store.restore_count = 7u;
	MockResidentClientDeliverStatus(TEST_RANKS - 1u,&snapshot);
	CHECK(SparkModelBatchEngineGetRankStatus(engine,TEST_RANKS - 1u,&report) == SPARK_STATUS_OK && report.generation == 1u &&
		report.adapter_snapshot.kv_store.attached == 1u && report.adapter_snapshot.kv_store.restore_count == 7u,"status: the engine returns the delivered report");
	CHECK(SparkModelBatchEngineGetRankStatus(engine,0u,&report) == SPARK_STATUS_OK && report.generation == 0u,"status: a rank with no report reads generation zero");
	CHECK(SparkModelBatchEngineGetRankStatus(engine,TEST_RANKS,&report) == SPARK_STATUS_NOT_FOUND,"status: an unknown rank is not found");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestWaitFirstRequestLane(SparkModelBatchEngine *engine,uint64_t request_id,SparkModelServingLane *lane)
{
	for (uint32_t step=0u; step<2000u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		if ( MockResidentClientLastLane(0u,lane) != 0u && lane->request_id == request_id )
			return(1u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
	return(0u);
}

static void TestScenarioPartialPrefixAppend(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0},prefix = {0};
	uint32_t tokens[65],divergent[65];
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	for (uint32_t i=0u; i<65u; i++)
		tokens[i] = divergent[i] = 11u + i;
	divergent[63] += 100u;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,500u,1u,tokens,63u);
	CHECK(TestWaitLane(engine,1u,60u,&prefix) != 0u &&
		prefix.cache_publish_token_count == 63u,"partial: publish exact 63-token checkpoint");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.token_events[1] == 1u,
		"partial: canonical prefix completes exactly once");
	TestSubmitPrompt(engine,2u,501u,2u,tokens,64u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u &&
		lane.sequence_position == 63u && lane.context_token_count == 64u &&
		lane.cache_prefix_token_count == 63u && lane.cache_publish_token_count == 64u &&
		memcmp(&lane.cache_prefix_identity,&prefix.cache_publish_identity,
			sizeof(lane.cache_prefix_identity)) == 0,
		"partial: required hit appends token 64 without replaying cached tokens");
	CHECK(TestWaitLane(engine,2u,64u,&lane) != 0u &&
		lane.context_token_count == 65u && lane.cache_publish_token_count == 65u,
		"partial: next decode publishes token 65 across the page boundary");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.token_events[2] == 2u &&
		state.cached_tokens[2] == 63u,"partial: output oracle records exact required hit");
	TestSubmitPrompt(engine,3u,502u,1u,divergent,65u);
	CHECK(TestWaitFirstRequestLane(engine,3u,&lane) != 0u &&
		lane.sequence_position == 63u && lane.cache_prefix_token_count == 63u &&
		lane.input_token_id == divergent[63],
		"partial: divergent append reuses immutable 63-token source");
	TestDriveUntilTerminal(engine,&state,3u,400u);
	CHECK(state.completed_events[3] == 1u && state.cached_tokens[3] == 63u,
		"partial: divergent append completes from required hit");
	TestSubmitPrompt(engine,4u,503u,1u,tokens,65u);
	CHECK(TestWaitFirstRequestLane(engine,4u,&lane) != 0u &&
		lane.sequence_position == 64u && lane.cache_prefix_token_count == 64u,
		"partial: original full-page branch remains reusable");
	TestDriveUntilTerminal(engine,&state,4u,400u);
	CHECK(state.completed_events[4] == 1u && state.cached_tokens[4] == 64u,
		"partial: full-page baseline completes without replay");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestLaneLogFind(uint64_t request_id,uint64_t position,uint32_t flags,SparkModelServingLane *found)
{
	SparkModelServingLane lane;
	uint32_t index;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
		if ( lane.request_id == request_id && lane.sequence_position == position && lane.flags == flags )
		{
			*found = lane;
			return(1u);
		}
	return(0u);
}

static SparkModelBatchEngine *TestConnectCapturingLog(const SparkModelResidentDeployment *deployment,TestBatchState *state,const char *runtime_root,char *log,size_t log_bytes)
{
	SparkModelBatchEngine *engine;
	FILE *capture;
	size_t read_bytes;
	int saved;
	log[0] = '\0';
	capture = tmpfile();
	CHECK(capture != 0,"cache mode log: capture file opens");
	if ( capture == 0 )
		return(TestConnect(deployment,state,runtime_root));
	fflush(stderr);
	saved = dup(2);
	(void)dup2(fileno(capture),2);
	engine = TestConnect(deployment,state,runtime_root);
	fflush(stderr);
	(void)dup2(saved,2);
	close(saved);
	rewind(capture);
	read_bytes = fread(log,1u,log_bytes - 1u,capture);
	log[read_bytes] = '\0';
	fclose(capture);
	return(engine);
}

static void TestScenarioAdapterRefused(const SparkModelResidentDeployment *deployment,const char *runtime_root,const char *name)
{
	TestBatchState state = {0};
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	engine = 0;
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,8u,0u);
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_UNSUPPORTED && engine == 0,name);
}

static void TestScenarioSpeculationOffPublishesDecodeCheckpoint(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0},published = {0};
	uint32_t prompt[4] = {11u,12u,13u,14u};
	uint32_t extended[9] = {11u,12u,13u,14u,1000u,1000u,1000u,1000u,31u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,720u,6u,prompt,4u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && state.token_events[1] == 6u,"speculation off: the request completes one token per frame");
	CHECK(TestLaneLogFind(1u,7u,SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN | SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH,&published) != 0u && published.cache_publish_token_count == 8u,"speculation off: the boundary decode lane publishes inline");
	TestSubmitPrompt(engine,2u,721u,1u,extended,9u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 8u && lane.sequence_position == 8u && memcmp(&lane.cache_prefix_identity,&published.cache_publish_identity,sizeof(lane.cache_prefix_identity)) == 0,"speculation off: a later prompt reuses the generated decode block");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 8u,"speculation off: the extending request reports the decode checkpoint");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioSpeculativePublishAdapterDefers(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0},published = {0};
	uint32_t prompt[4] = {11u,12u,13u,14u};
	uint32_t extended[9] = {11u,12u,13u,14u,1000u,1000u,1000u,1000u,31u};
	char log[4096];
	MockResidentClientReset();
	engine = TestConnectCapturingLog(deployment,&state,runtime_root,log,sizeof(log));
	if ( engine == 0 )
		return;
	CHECK(strstr(log,"adapter=test.model.serving.adapter.speculative-deferred.v1 decode_checkpoints=deferred") != 0,"speculative deferred: startup names the adapter's cache mode");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,730u,6u,prompt,4u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && state.token_events[1] == 6u,"speculative deferred: the request completes");
	CHECK(TestLaneLogFind(1u,7u,SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN,&lane) != 0u && lane.cache_publish_token_count == 0u,"speculative deferred: the boundary decode lane carries no inline checkpoint");
	CHECK(TestLaneLogFind(1u,8u,SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH,&published) != 0u && published.context_token_count == 8u && published.cache_publish_token_count == 8u,"speculative deferred: a publish frame names the boundary checkpoint");
	TestSubmitPrompt(engine,2u,731u,1u,extended,9u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 8u && memcmp(&lane.cache_prefix_identity,&published.cache_publish_identity,sizeof(lane.cache_prefix_identity)) == 0,"speculative deferred: the deferred checkpoint is reusable");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 8u,"speculative deferred: the extending request reports the deferred checkpoint");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioChainPublishesFinalCheckpoint(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0},published = {0};
	uint32_t prompt[8] = {11u,12u,13u,14u,1000u,1000u,1001u,1002u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,600u,4u,prompt,4u);
	CHECK(TestWaitLane(engine,1u,4u,&lane) != 0u && lane.cache_publish_token_count == 0u,
		"chain: intermediate decode has no requested checkpoint");
	MockResidentClientSetAutoTokens(3u);
	CHECK(TestWaitLane(engine,1u,7u,&published) != 0u && published.flags == SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH && published.context_token_count == 7u && published.cache_publish_token_count == 7u,
		"chain: zero-row publication names actual processed context before release");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.token_events[1] == 4u,
		"chain: three returned tokens do not imply three checkpoints");
	MockResidentClientSetAutoTokens(1u);
	TestSubmitPrompt(engine,2u,601u,1u,prompt,8u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u &&
		lane.sequence_position == 7u && lane.cache_prefix_token_count == 7u && lane.input_token_id == prompt[7] &&
		memcmp(&lane.cache_prefix_identity,&published.cache_publish_identity,sizeof(lane.cache_prefix_identity)) == 0,
		"chain: final processed checkpoint is published before release");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 7u,
		"chain: required hit consumes final checkpoint without replay");
	TestSubmitPrompt(engine,3u,602u,1u,prompt,6u);
	CHECK(TestWaitFirstRequestLane(engine,3u,&lane) != 0u && lane.cache_prefix_token_count == 4u,
		"chain: final publication does not invent intermediate recurrent checkpoints");
	TestDriveUntilTerminal(engine,&state,3u,400u);
	CHECK(state.completed_events[3] == 1u && state.cached_tokens[3] == 4u,
		"chain: unavailable intermediate state is an explicit miss");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioChainPublishesAtBlockBoundary(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0},published = {0};
	uint32_t prompt[9] = {11u,12u,13u,14u,1000u,1000u,1001u,1000u,1001u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,620u,8u,prompt,4u);
	CHECK(TestWaitLane(engine,1u,4u,&lane) != 0u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) == 0u,"chain block: the first decode frame carries no checkpoint");
	MockResidentClientSetAutoTokens(2u);
	CHECK(TestWaitLane(engine,1u,6u,&lane) != 0u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) == 0u && lane.cache_publish_token_count == 0u,"chain block: a chain that ends inside a block decodes on without a publish frame");
	CHECK(TestWaitLane(engine,1u,8u,&published) != 0u && published.flags == SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH && published.context_token_count == 8u && published.cache_publish_token_count == 8u,"chain block: a chain that reaches the block boundary publishes it");
	MockResidentClientSetAutoTokens(3u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && state.token_events[1] == 8u,"chain block: every chained token is emitted once");
	MockResidentClientSetAutoTokens(1u);
	TestSubmitPrompt(engine,2u,621u,1u,prompt,9u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.sequence_position == 8u && lane.cache_prefix_token_count == 8u && memcmp(&lane.cache_prefix_identity,&published.cache_publish_identity,sizeof(lane.cache_prefix_identity)) == 0,"chain block: the boundary checkpoint is reusable");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 8u,"chain block: a required hit consumes the boundary checkpoint");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioGeneratedCheckpointIdentity(const SparkModelResidentDeployment *deployment,const char *runtime_root,uint32_t chain)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t prefix[5] = {11u,12u,13u,14u,15u};
	uint32_t divergent[12] = {11u,12u,13u,14u,15u,21u,22u,23u,24u,25u,26u,27u};
	uint32_t extended[9] = {11u,12u,13u,14u,15u,1000u,1000u,chain == 1u ? 1000u : 1001u,31u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,640u,4u,prefix,5u);
	CHECK(TestWaitLane(engine,1u,5u,&lane) != 0u,"generated identity: prefill emits before decode");
	MockResidentClientSetAutoTokens(chain);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.token_events[1] == 4u,"generated identity: the source request completes");
	MockResidentClientSetAutoTokens(1u);
	TestSubmitPrompt(engine,2u,641u,1u,divergent,12u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 5u && lane.sequence_position == 5u && lane.input_token_id == divergent[5],"generated identity: a prompt that diverges from the generated tokens reuses only the shared prompt");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 5u,"generated identity: the divergent request reports the shared prompt only");
	TestSubmitPrompt(engine,3u,642u,1u,extended,9u);
	CHECK(TestWaitFirstRequestLane(engine,3u,&lane) != 0u && lane.cache_prefix_token_count == 8u && lane.sequence_position == 8u && lane.input_token_id == extended[8],"generated identity: a prompt that repeats the generated tokens reuses the generated checkpoint");
	TestDriveUntilTerminal(engine,&state,3u,400u);
	CHECK(state.completed_events[3] == 1u && state.cached_tokens[3] == 8u,"generated identity: the extending request reports the generated checkpoint");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestPrefillChunksAreCanonical(uint64_t request_id,uint32_t block,uint32_t prompt_count)
{
	SparkModelServingLane lane;
	uint32_t index,chunks;
	chunks = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
	{
		if ( lane.request_id != request_id || lane.context_token_count <= lane.sequence_position || lane.context_token_count > prompt_count )
			continue;
		if ( lane.sequence_position % block != 0u || (lane.context_token_count % block != 0u && lane.context_token_count != prompt_count) )
			return(0u);
		chunks++;
	}
	return(chunks);
}

static void TestScenarioCanonicalPrefillChunks(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	uint32_t longer[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	uint32_t shorter[3] = {21u,22u,23u};
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,6u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,650u,1u,longer,8u);
	TestSubmitPrompt(engine,2u,651u,1u,shorter,3u);
	TestDriveUntilTerminal(engine,&state,2u,800u);
	CHECK(state.completed_events[1] == 1u && state.completed_events[2] == 1u,"canonical chunks: both prompts complete");
	CHECK(TestPrefillChunksAreCanonical(1u,4u,8u) == 2u,"canonical chunks: a prompt sharing the row budget still prefills whole blocks");
	CHECK(TestPrefillChunksAreCanonical(2u,4u,3u) == 1u,"canonical chunks: a short prompt prefills in one chunk");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioSharedPrefillRows(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	SparkModelResidentDeployment limited = *deployment;
	uint32_t longer[3] = {31u,32u,33u};
	uint32_t first[1] = {41u},second[3] = {51u,52u,53u};
	uint32_t pass,submissions[2] = {0u,0u},rows[2] = {0u,0u};
	for (pass=0u; pass<2u; pass++)
	{
		TestBatchState state = {0};
		SparkModelBatchEngine *engine;
		uint32_t base = 660u + 10u * pass;
		limited.max_shared_input_row_count = pass == 0u ? 0u : 4u;
		MockResidentClientReset();
		engine = TestConnectRows(&limited,&state,runtime_root,8u);
		if ( engine == 0 )
			return;
		MockResidentClientSetAutoTokens(1u);
		MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
		TestSubmitPrompt(engine,1u,base,1u,longer,3u);
		TestSubmitPrompt(engine,2u,base + 1u,1u,first,1u);
		TestSubmitPrompt(engine,3u,base + 2u,1u,second,3u);
		TestDriveUntilTerminal(engine,&state,3u,800u);
		CHECK(state.completed_events[1] == 1u && state.completed_events[2] == 1u && state.completed_events[3] == 1u,
			"shared rows: every prompt completes with and without a shared-wave limit");
		MockResidentClientTakePrefillShared(&submissions[pass],&rows[pass]);
		SparkModelBatchEngineDestroy(engine);
	}
	CHECK(submissions[0] != 0u && rows[0] > 4u,"shared rows: without a limit, one prefill wave carries several prompts past the shared size");
	CHECK(submissions[1] != 0u && rows[1] <= 4u,"shared rows: a wave carrying several prompts stays within the shared limit, while one prompt alone may use the full budget");
}

static uint32_t TestCoordinatorFrames(void)
{
	return(MockResidentClientCalls(0u,MOCK_CALL_SUBMIT) + MockResidentClientCalls(0u,MOCK_CALL_PREPARE) + MockResidentClientCalls(0u,MOCK_CALL_CONTINUE));
}

static void TestScenarioSingleTokenEosPublishesTail(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t frames;
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,630u,8u);
	CHECK(TestWaitLane(engine,1u,4u,&lane) != 0u,"single EOS: a decode frame follows prefill");
	frames = TestCoordinatorFrames();
	MockResidentClientSetTokenStart(154820u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	TestDrive(engine,8u);
	CHECK(state.completed_events[1] == 1u && state.token_events[1] == 2u && TestCoordinatorFrames() == frames + 2u,"single EOS: a reply that stops on EOS mid-block publishes its partial block before the release");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioChainEosCheckpoint(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t prompt[8] = {11u,12u,13u,14u,154819u,154819u,154820u,154821u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 ) return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(154819u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,610u,4u,prompt,4u);
	CHECK(TestWaitLane(engine,1u,4u,&lane) != 0u,"chain EOS: prefill emits before decode chain");
	MockResidentClientSetAutoTokens(3u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && state.token_events[1] == 3u,
		"chain EOS: early output stop completes without pretending to rewind resident state");
	MockResidentClientSetAutoTokens(1u);
	TestSubmitPrompt(engine,2u,611u,1u,prompt,8u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 4u,
		"chain EOS: truncated emitted token count is never indexed as a checkpoint");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 4u,"chain EOS: subsequent request uses valid prefill checkpoint");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioSamplingValidation(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmitRequest request = {0};
	SparkModelBatchRequestHandle handle = 0u;
	uint32_t prompt[2] = {11u,12u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.top_p = 1.0f;
	request.request_id = request.sequence_id = 1u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 2u;
	request.output_token_budget = 2u;
	request.temperature = 2.5f;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: temperature above the maximum is rejected");
	request.temperature = 0.0f;
	request.seed = 5u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: a greedy request carries no seed");
	request.temperature = 0.00001f;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: temperature below the minimum is rejected");
	request.seed = 0u;
	request.temperature = 0.0f;
	request.top_p = 0.5f;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: a greedy request carries no top-p");
	request.top_p = 1.0f;
	request.top_k = 3u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: a greedy request carries no top-k");
	request.top_k = 0u;
	request.temperature = 0.7f;
	request.seed = 5u;
	request.top_p = 0.0f;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: top-p must be above zero");
	request.top_p = 1.5f;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: top-p above one is rejected");
	request.top_p = 1.0f;
	request.top_k = SPARK_SAMPLING_MAX_TOP_K + 1u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: top-k above the maximum is rejected");
	request.top_k = 0u;
	request.logprobs = SPARK_SAMPLING_MAX_LOGPROBS + 1u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_INVALID_ARGUMENT,"sampling: more than twenty top logprobs are rejected");
	request.logprobs = 0u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK && handle != 0u,"sampling: a seeded request is admitted by a sampling adapter");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioChainYieldsToPrefill(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	uint32_t prompt[2] = {11u,12u},other[2] = {21u,22u},chained,yielded,steps;
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,950u,24u,prompt,2u);
	for (steps=0u; steps<200u && state.token_events[1] < 2u; steps++)
		TestDrive(engine,1u);
	chained = MockResidentClientTakeMaxTokensPerSequence();
	TestSubmitPrompt(engine,2u,951u,2u,other,2u);
	(void)SparkModelBatchEngineProgress(engine,8u);
	yielded = MockResidentClientTakeMaxTokensPerSequence();
	TestDriveUntilTerminal(engine,&state,2u,800u);
	CHECK(chained > 1u,"chain yield: a lone decoding request chains several tokens per step");
	CHECK(yielded == 1u,"chain yield: a decode wave built while an admissible prefill waits runs one step so the prefill is not held behind a chain");
	CHECK(state.completed_events[1] == 1u && state.completed_events[2] == 1u,"chain yield: both requests complete");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioAdapterContextLimit(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	uint32_t prompt[2] = {11u,12u};
	MockResidentClientReset();
	MockResidentClientSetMaxSequencePositions(0u,64u);
	MockResidentClientSetMaxSequencePositions(TEST_RANKS - 1u,40u);
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestDrive(engine,3u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.context_limit == 40u,"context limit: the engine takes the smallest adapter limit across ranks");
	CHECK(TestSubmitPromptStatus(engine,1u,951u,39u,prompt,2u) == SPARK_STATUS_CAPACITY_EXCEEDED,"context limit: a request past the smallest adapter limit is refused at admission");
	CHECK(TestSubmitPromptStatus(engine,2u,952u,38u,prompt,2u) == SPARK_STATUS_OK,"context limit: a request that fits the smallest adapter limit is admitted");
	MockResidentClientSetMaxSequencePositions(TEST_RANKS - 1u,0u);
	TestDrive(engine,3u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.context_limit == 64u,"context limit: the limit follows the ranks' current adapters");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioTruncatedChains(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmitRequest request = {0};
	SparkModelBatchRequestHandle handle = 0u;
	uint32_t prompt[2] = {11u,12u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.request_id = 1u;
	request.sequence_id = 911u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 2u;
	request.output_token_budget = 4u;
	request.temperature = 0.7f;
	request.seed = 9u;
	request.top_p = 0.9f;
	request.top_k = 5u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"chains: a sampled request with top-p and top-k is admitted");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.token_events[1] == 4u && MockResidentClientTakeMaxTokensPerSequence() > 1u,
		"chains: a top-p and top-k lane without logprobs chains decode steps on a chaining adapter");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioLogprobs(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmitRequest request = {0};
	SparkModelBatchRequestHandle handle = 0u;
	uint32_t prompt[2] = {11u,12u};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,901u,4u,prompt,2u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && MockResidentClientTakeMaxTokensPerSequence() > 1u && state.logprob_events[1] == 0u,"logprobs: a plain request chains decode steps and carries no logprobs");
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	MockResidentClientSetLogprobs(3u,0u);
	request.descriptor_bytes = sizeof(request);
	request.request_id = 2u;
	request.sequence_id = 902u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 2u;
	request.output_token_budget = 4u;
	request.temperature = 0.7f;
	request.seed = 9u;
	request.top_p = 0.9f;
	request.top_k = 5u;
	request.logprobs = 3u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"logprobs: a sampled request with top-p, top-k and logprobs is admitted");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.token_events[2] == 4u,"logprobs: the request completes with its four tokens");
	CHECK(state.logprob_matches[2] == 4u,"logprobs: every token event carries three entries led by the emitted token");
	CHECK(MockResidentClientTakeMaxTokensPerSequence() == 1u,"logprobs: a logprob lane decodes one token per step on a chaining adapter");
	MockResidentClientSetLogprobs(3u,7u);
	request.request_id = 3u;
	request.sequence_id = 903u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"logprobs: a second logprob request is admitted");
	TestDriveUntilTerminal(engine,&state,3u,400u);
	CHECK(state.error_events[3] == 1u && state.token_events[3] == 0u,"logprobs: entries that do not lead with the emitted token fail the request");
	CHECK(SparkModelBatchEngineReopenAdmission(engine) == SPARK_STATUS_VALIDATION_FAILED,"latch: a protocol failure stays latched while the rank session is unchanged");
	request.request_id = 4u;
	request.sequence_id = 904u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) != SPARK_STATUS_OK,"latch: a latched engine admits nothing");
	MockResidentClientKill(1u);
	TestDrive(engine,5u);
	MockResidentClientRevive(1u);
	TestDrive(engine,50u);
	CHECK(SparkModelBatchEngineReopenAdmission(engine) == SPARK_STATUS_OK,"latch: a new rank session clears the latched failure");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioPartialCopyCapacity(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	SparkModelResidentDeployment bounded = *deployment;
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmitRequest request = {0};
	SparkModelBatchRequestHandle handle = 0u;
	uint32_t prompt[4] = {11u,12u,13u,14u};
	MockResidentClientReset();
	bounded.runtime_limits.max_active_sequence_count = 1u;
	bounded.runtime_limits.kv_physical_page_capacity = 1u;
	engine = TestConnect(&bounded,&state,runtime_root);
	if ( engine == 0 )
		return;
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.top_p = 1.0f;
	request.request_id = request.sequence_id = 1u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 3u;
	request.output_token_budget = 2u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_CAPACITY_EXCEEDED && handle == 0u,
		"partial capacity: immutable prompt append requires a copy page");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,701u,1u,prompt,3u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u,"partial capacity: seed fits one page");
	TestSubmitPrompt(engine,2u,702u,1u,prompt,4u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.token_events[2] == 1u && state.cached_tokens[2] == 0u,
		"partial capacity: a mid-block hit whose copy page does not fit falls back to its whole blocks and recomputes the rest");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioMeasurements(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,800u,1u,prompt,4u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.prefix_miss_count == 1u && view.prefix_hit_count == 0u && view.first_token_count == 1u,"measure: a cold prompt counts one miss and one first token");
	CHECK(view.ttft_ns_total != 0u && view.ttft_ns_total == view.ttft_ns_maximum && view.queue_ns_total + view.prefill_ns_total == view.ttft_ns_total,"measure: queue and prefill split the first-token latency");
	CHECK(state.first_dispatch_ns[1] >= state.accepted_ns[1] && state.first_token_ns[1] >= state.first_dispatch_ns[1] && state.accepted_ns[1] != 0u,"measure: the token event carries the request's first dispatch time");
	TestSubmitPrompt(engine,2u,801u,1u,prompt,8u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.prefix_hit_count == 1u && view.prefix_hit_token_count == 4u && view.prefix_miss_count == 1u && view.first_token_count == 2u,"measure: a shared prefix counts one hit with its cached tokens");
	CHECK(view.rejected_lane_count == 0u && view.rejected_submission_count_by_status[SPARK_STATUS_BUSY] == 0u,"measure: an unrejected run counts no rejections");
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_BUSY);
	TestSubmitPrompt(engine,3u,802u,1u,prompt,3u);
	TestDrive(engine,20u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_OK);
	TestDriveUntilTerminal(engine,&state,3u,400u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.rejected_submission_count_by_status[SPARK_STATUS_BUSY] != 0u && view.rejected_lane_count == view.rejected_submission_count_by_status[SPARK_STATUS_BUSY] && view.stale_prefix_recompute_count == 0u,"measure: a BUSY rank counts its rejected waves by status");
	CHECK(state.completed_events[3] == 1u && view.prefix_miss_count == 2u && view.first_token_count == 3u,"measure: retried waves count the prefix outcome once");
	CHECK(state.first_dispatch_ns[3] >= state.accepted_ns[3] + UINT64_C(10000000),"measure: BUSY retry time before admission is booked as queue, not prefill");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestFindLane(uint64_t request_id,uint64_t position,uint32_t prefix_tokens,SparkModelServingLane *found)
{
	SparkModelServingLane lane;
	uint32_t index;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
		if ( lane.request_id == request_id && lane.sequence_position == position && lane.cache_prefix_token_count == prefix_tokens )
		{
			*found = lane;
			return(1u);
		}
	return(0u);
}

static void TestScenarioStalePrefixRecomputes(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u},plain_a[3] = {31u,32u,33u},plain_b[3] = {41u,42u,43u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane canonical = {0},stale = {0},rebuilt = {0},reused = {0};
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,16u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,900u,1u,prompt,4u);
	CHECK(TestWaitLane(engine,1u,0u,&canonical) != 0u && canonical.cache_publish_token_count == 4u,"stale: capture the canonical first block identity");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_NOT_FOUND,1u);
	TestSubmitPrompt(engine,2u,901u,1u,prompt,8u);
	TestSubmitPrompt(engine,3u,902u,1u,plain_a,3u);
	TestSubmitPrompt(engine,4u,903u,1u,plain_b,3u);
	TestDriveUntilTerminal(engine,&state,4u,800u);
	CHECK(state.completed_events[2] == 1u && state.completed_events[3] == 1u && state.completed_events[4] == 1u && state.error_events[2] + state.error_events[3] + state.error_events[4] == 0u,"stale: a stale prefix hint fails no lane of its wave");
	CHECK(state.token_events[2] == 1u && state.token_events[3] == 1u && state.token_events[4] == 1u,"stale: every lane of the rejected wave emits its token exactly once");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.rejected_submission_count_by_status[SPARK_STATUS_NOT_FOUND] == 1u && view.rejected_lane_count == 3u && view.stale_prefix_recompute_count == 1u,"stale: one rejected wave of three lanes recomputes one prefix");
	CHECK(state.stale_recomputes[2] == 1u && state.stale_recomputes[3] == 0u && state.stale_recomputes[4] == 0u && state.cached_tokens[2] == 0u,"stale: only the prefix lane is reset and reports no cached tokens");
	CHECK(TestFindLane(2u,4u,4u,&stale) != 0u && (stale.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u,"stale: the rejected lane carried the cached prefix");
	{
		SparkModelServingCacheIdentity first_block = {0};
		CHECK(TestFindLane(2u,0u,0u,&rebuilt) != 0u && (rebuilt.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) == 0u && rebuilt.cache_publish_token_count == 8u &&
			rebuilt.cache_block_identity_count == 1u && MockResidentClientIdentityLog(rebuilt.cache_block_identity_first,&first_block) != 0u &&
			memcmp(&first_block,&canonical.cache_publish_identity,sizeof(canonical.cache_publish_identity)) == 0,"stale: the recompute starts at position zero and republishes the canonical block inside its span");
	}
	TestSubmitPrompt(engine,5u,904u,1u,prompt,8u);
	TestDriveUntilTerminal(engine,&state,5u,400u);
	CHECK(state.completed_events[5] == 1u && state.cached_tokens[5] == 4u && TestFindLane(5u,4u,4u,&reused) != 0u && memcmp(&reused.cache_prefix_identity,&canonical.cache_publish_identity,sizeof(canonical.cache_publish_identity)) == 0,"stale: the republished block serves the next hit");
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_BUSY,1u);
	TestSubmitPrompt(engine,6u,905u,1u,prompt,8u);
	TestDriveUntilTerminal(engine,&state,6u,400u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.rejected_submission_count_by_status[SPARK_STATUS_BUSY] == 1u && view.stale_prefix_recompute_count == 1u,"stale: BUSY keeps its backoff retry and recomputes nothing");
	CHECK(state.completed_events[6] == 1u && state.cached_tokens[6] == 4u && state.stale_recomputes[6] == 0u,"stale: a BUSY-rejected hit still completes from its prefix");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioStalePrefixIsolatesLanes(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt_a[8] = {11u,12u,13u,14u,15u,16u,17u,18u},prompt_b[8] = {51u,52u,53u,54u,55u,56u,57u,58u},plain[3] = {61u,62u,63u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane lane = {0};
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,16u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,920u,1u,prompt_a,4u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	TestSubmitPrompt(engine,2u,921u,1u,prompt_b,4u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	MockResidentClientScriptStalePrefixRequest(1u,3u);
	TestSubmitPrompt(engine,3u,922u,1u,prompt_a,8u);
	TestSubmitPrompt(engine,4u,923u,1u,prompt_b,8u);
	TestSubmitPrompt(engine,5u,924u,1u,plain,3u);
	TestDriveUntilTerminal(engine,&state,5u,800u);
	MockResidentClientScriptStalePrefixRequest(1u,0u);
	CHECK(state.completed_events[3] == 1u && state.completed_events[4] == 1u && state.completed_events[5] == 1u && state.error_events[3] + state.error_events[4] + state.error_events[5] == 0u,"stale isolate: every lane of a wave with two prefix lanes completes");
	CHECK(state.stale_recomputes[3] == 1u && state.cached_tokens[3] == 0u,"stale isolate: the stale prefix lane alone is recomputed");
	CHECK(state.stale_recomputes[4] == 0u && state.cached_tokens[4] == 4u && TestFindLane(4u,4u,4u,&lane) != 0u,"stale isolate: the valid prefix lane keeps its prefix");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.stale_prefix_recompute_count == 1u && view.stale_prefix_isolation_count == 1u && view.rejected_submission_count_by_status[SPARK_STATUS_NOT_FOUND] == 2u,"stale isolate: one isolation, then one single-lane NOT_FOUND tombstones exactly one prefix");
	CHECK(view.prefix_hit_count == 1u && view.prefix_miss_count == 4u,"stale isolate: hits count only admitted prefix waves");
	TestSubmitPrompt(engine,6u,925u,1u,prompt_b,8u);
	TestDriveUntilTerminal(engine,&state,6u,400u);
	CHECK(state.completed_events[6] == 1u && state.cached_tokens[6] == 4u,"stale isolate: the untouched prefix still serves the next hit");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestLaneLogEdge(uint64_t request_id,uint32_t last)
{
	SparkModelServingLane lane;
	uint32_t index,found = UINT32_MAX;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
		if ( lane.request_id == request_id && (found == UINT32_MAX || last != 0u) )
			found = index;
	return(found);
}

static void TestScenarioSharedPrefixReservation(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	SparkModelResidentDeployment bounded = *deployment;
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t shared_a[18],shared_b[18],other[12],whole[36],index;
	for (index=0u; index<16u; index++)
		shared_a[index] = shared_b[index] = 100u + index;
	shared_a[16] = 201u;
	shared_a[17] = 202u;
	shared_b[16] = 211u;
	shared_b[17] = 212u;
	for (index=0u; index<12u; index++)
		other[index] = 300u + index;
	for (index=0u; index<36u; index++)
		whole[index] = 400u + index;
	MockResidentClientReset();
	bounded.runtime_limits.kv_physical_page_capacity = 9u;
	engine = TestConnectRows(&bounded,&state,runtime_root,16u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,960u,1u,shared_a,16u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	TestSubmitPrompt(engine,2u,961u,4u,shared_a,18u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u,"shared reservation: the first sharer of the cached prefix is bound and holds its pages");
	TestSubmitPrompt(engine,3u,962u,8u,shared_b,18u);
	TestDriveUntilTerminal(engine,&state,2u,800u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 16u,"shared reservation: the holder reuses the cached 16-token prefix");
	CHECK(TestLaneLogEdge(3u,0u) < TestLaneLogEdge(2u,1u),"shared reservation: the second request is bound beside the first, reserving 6 + 3 of 9 pages, where charging its whole prompt (6 + 7) would wait for the first to finish");
	TestSubmitPrompt(engine,4u,963u,1u,other,12u);
	TestDriveUntilTerminal(engine,&state,4u,800u);
	CHECK(state.completed_events[3] == 1u && state.cached_tokens[3] == 16u && state.completed_events[4] == 1u && state.error_events[3] + state.error_events[4] == 0u,"shared reservation: the sharer reuses the same prefix and it and a later request complete");
	CHECK(TestLaneLogEdge(4u,0u) > TestLaneLogEdge(3u,1u),"shared reservation: when the holder finishes first the sharer inherits the shared pages, so a 3-page request waits (7 + 3 > 9)");
	TestSubmitPrompt(engine,5u,964u,1u,whole,36u);
	TestDriveUntilTerminal(engine,&state,5u,800u);
	CHECK(state.completed_events[5] == 1u,"shared reservation: every page is returned, so a request needing all 9 pages is admitted");
	TestSubmitPrompt(engine,6u,965u,4u,shared_a,18u);
	CHECK(TestWaitFirstRequestLane(engine,6u,&lane) != 0u,"shared reservation: a new holder of the cached prefix is bound");
	MockResidentClientScriptStalePrefixRequest(1u,7u);
	TestSubmitPrompt(engine,7u,966u,8u,shared_b,18u);
	TestDriveUntilTerminal(engine,&state,7u,800u);
	MockResidentClientScriptStalePrefixRequest(1u,0u);
	CHECK(state.completed_events[6] == 1u && state.error_events[6] == 0u,"shared reservation: the holder completes when a sharer's prefix turns stale");
	CHECK(state.error_events[7] == 1u && state.token_events[7] == 0u && state.stale_recomputes[7] == 1u,"shared reservation: a sharer whose shared prefix is gone and whose own copy does not fit beside the holder (6 + 7 > 9) fails instead of overcommitting the pages");
	TestSubmitPrompt(engine,8u,967u,1u,whole,36u);
	TestDriveUntilTerminal(engine,&state,8u,800u);
	CHECK(state.completed_events[8] == 1u,"shared reservation: the failed sharer returns its pages too");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioVerificationFailureIsFatal(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,16u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,930u,1u,prompt,4u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_VALIDATION_FAILED,1u);
	TestSubmitPrompt(engine,2u,931u,1u,prompt,8u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_OK,0u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.stale_prefix_recompute_count == 0u && view.rejected_submission_count_by_status[SPARK_STATUS_VALIDATION_FAILED] == 1u,"validation: VALIDATION_FAILED is never treated as a stale prefix");
	CHECK(state.error_events[2] == 1u && state.completed_events[2] == 0u && state.stale_recomputes[2] == 0u,"validation: the request fails loudly instead of recomputing");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioStalePrefixTerminates(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[13] = {11u,12u,13u,14u,15u,16u,17u,18u,19u,20u,21u,22u,23u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane lane = {0};
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,16u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,910u,1u,prompt,12u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_NOT_FOUND,10u);
	TestSubmitPrompt(engine,2u,911u,1u,prompt,13u);
	TestDriveUntilTerminal(engine,&state,2u,800u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_OK,0u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u && state.token_events[2] == 1u,"stale chain: a request whose every cached block is stale still completes once");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.stale_prefix_recompute_count == 3u && state.stale_recomputes[2] == 3u && view.rejected_submission_count_by_status[SPARK_STATUS_NOT_FOUND] == 3u,"stale chain: recomputes stop within the prompt's block count");
	CHECK(TestFindLane(2u,12u,12u,&lane) != 0u && TestFindLane(2u,8u,8u,&lane) != 0u && TestFindLane(2u,4u,4u,&lane) != 0u && TestFindLane(2u,0u,0u,&lane) != 0u,"stale chain: each retry drops exactly the deepest stale block");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioRankKilledAndRevived(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,3u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	/* The rank is dead from the start (residentd down, agent respawning). */
	MockResidentClientKill(1u);
	TestDrive(engine,50u);
	CHECK( state.total_terminals == 0u,
		"chaos: a dead rank holds the request without failing it");
	MockResidentClientRevive(1u);
	TestDriveUntilTerminal(engine,&state,1u,2000u);
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"chaos: request completed once the killed rank revived");
	SparkModelBatchEngineDestroy(engine);
}

/* A rank answering BUSY is transient backpressure, not a fault: the
 * pipeline must NOT fail-stop (that reconnects every rank and resets every
 * engine session, killing all in-flight chains fleet-wide). The request
 * retries and completes; no rank ever reconnects. */
static void TestScenarioBusyRestoreDeadline(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine = 0;
	struct timespec start,now;
	uint64_t elapsed_ns = 0u;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,8u,0u);
	configuration.inflight_budget_ns = SPARK_MODEL_BATCH_ENGINE_MIN_INFLIGHT_BUDGET_NS;
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK,"busy deadline: connect");
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_BUSY);
	TestSubmit(engine,1u,510u,2u);
	clock_gettime(CLOCK_MONOTONIC,&start);
	while ( state.total_terminals == 0u && elapsed_ns < UINT64_C(5000000000) )
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveAll();
		usleep(5000);
		clock_gettime(CLOCK_MONOTONIC,&now);
		elapsed_ns = (uint64_t)(now.tv_sec - start.tv_sec) * UINT64_C(1000000000) + (uint64_t)now.tv_nsec - (uint64_t)start.tv_nsec;
	}
	CHECK(state.error_events[1] == 1u && elapsed_ns >= SPARK_MODEL_BATCH_ENGINE_MIN_INFLIGHT_BUDGET_NS && elapsed_ns < UINT64_C(5000000000),
		"busy deadline: a rank that stays BUSY fails the request once the in-flight budget passes, not after a retry count");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioPendingRestoreRetries(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine = 0;
	struct timespec start,now;
	uint64_t elapsed_ns = 0u;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,8u,0u);
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK,"pending restore: connect");
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	MockResidentClientScriptSubmitStatusOnce(1u,SPARK_STATUS_PENDING);
	TestSubmit(engine,1u,520u,2u);
	clock_gettime(CLOCK_MONOTONIC,&start);
	while ( state.total_terminals == 0u && elapsed_ns < UINT64_C(5000000000) )
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
		clock_gettime(CLOCK_MONOTONIC,&now);
		elapsed_ns = (uint64_t)(now.tv_sec - start.tv_sec) * UINT64_C(1000000000) + (uint64_t)now.tv_nsec - (uint64_t)start.tv_nsec;
	}
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u,"pending restore: a rank answering PENDING while it restores is retried and the request completes");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.rejected_submission_count_by_status[SPARK_STATUS_PENDING] == 1u,"pending restore: the PENDING answer is counted once");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioRankBusyBackpressure(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	uint32_t step;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_BUSY);
	TestSubmit(engine,1u,500u,2u);
	for (step=0u; step<50u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(2000);
	}
	CHECK( state.total_terminals == 0u,
		"busy: a BUSY rank holds the request without failing it");
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_OK);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	{
		SparkModelBatchEngineView view;
		if ( SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK )
			fprintf(stderr,"DBG busy-end live=%u tokens=%u terminals=%u active_txn=%u submitted=%llu completed=%llu admitted=%llu rejected=%llu\n",
				(unsigned)view.live_request_count,(unsigned)state.token_events[1],
				(unsigned)state.total_terminals,
				(unsigned)view.pipeline.active_transaction_count,
				(unsigned long long)view.pipeline.submitted_count,
				(unsigned long long)view.pipeline.completed_count,
				(unsigned long long)view.pipeline.admitted_count,
				(unsigned long long)view.pipeline.rejected_count);
	}
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"busy: the request completes once the rank drains");
	{
		uint32_t rank;
		uint32_t reconnected = 0u;
		for (rank=0u; rank<TEST_RANKS; rank++)
			if ( MockResidentClientGeneration(rank) != 1u )
				reconnected = 1u;
		CHECK( reconnected == 0u,
			"busy: no rank reconnected — no pipeline fail-stop on backpressure");
	}
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioInflightExpiryHoldsSlot(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	static const uint32_t prompt[4] = {11u,12u,13u,14u};
	TestBatchState state;
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine = 0;
	uint64_t ids[TEST_RANKS];
	struct timespec start,now;
	uint64_t elapsed_ns = 0u;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,1u,0u);
	configuration.inflight_budget_ns = SPARK_MODEL_BATCH_ENGINE_MIN_INFLIGHT_BUDGET_NS;
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK,"inflight expiry: connect");
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,530u,2u);
	for (uint32_t step=0u; step<20u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveResults();
		(void)MockResidentClientDriveDecisions();
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		ids[rank] = MockResidentClientPendingEvent(rank,MOCK_EVENT_COMPLETION,0u);
		CHECK(ids[rank] != 0u,"inflight expiry: the submission is in flight on every rank");
	}
	clock_gettime(CLOCK_MONOTONIC,&start);
	while ( state.error_events[1] == 0u && elapsed_ns < UINT64_C(5000000000) )
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveResults();
		(void)MockResidentClientDriveDecisions();
		usleep(5000);
		clock_gettime(CLOCK_MONOTONIC,&now);
		elapsed_ns = (uint64_t)(now.tv_sec - start.tv_sec) * UINT64_C(1000000000) + (uint64_t)now.tv_nsec - (uint64_t)start.tv_nsec;
	}
	CHECK(state.error_events[1] == 1u,"inflight expiry: the client is failed once the in-flight budget passes");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.live_request_count == 1u,
		"inflight expiry: the request slot stays held while its submission is still in flight");
	CHECK(TestSubmitPromptStatus(engine,2u,531u,2u,prompt,4u) == SPARK_STATUS_BUSY,
		"inflight expiry: a new request cannot take the only slot while it is held");
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
		CHECK(MockResidentClientDeliverEvent(rank,ids[rank],MOCK_EVENT_COMPLETION,SPARK_STATUS_OK,1u) != 0u,
			"inflight expiry: deliver the late completion");
	TestDrive(engine,200u);
	CHECK(state.token_events[1] == 0u && state.error_events[1] == 1u && state.completed_events[1] == 0u,
		"inflight expiry: the late completion emits no token and no second terminal to the expired client");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.live_request_count == 0u,
		"inflight expiry: the slot is released once the submission completes");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioBusyAdmittedPrefillReleases(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	static const uint32_t prompt[4] = {11u,12u,13u,14u};
	TestBatchState state;
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle = 0u;
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine = 0;
	uint64_t ids[TEST_RANKS];
	uint32_t releases_before = 0u;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,1u,0u);
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK,"busy admitted prefill: connect");
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.top_p = 1.0f;
	request.request_id = 1u;
	request.sequence_id = 540u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 4u;
	request.output_token_budget = 2u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"busy admitted prefill: submit");
	for (uint32_t step=0u; step<20u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveResults();
		(void)MockResidentClientDriveDecisions();
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		ids[rank] = MockResidentClientPendingEvent(rank,MOCK_EVENT_COMPLETION,0u);
		CHECK(ids[rank] != 0u,"busy admitted prefill: every rank admitted the prefill");
		releases_before += MockResidentClientReleaseSubmissions(rank);
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
		CHECK(MockResidentClientDeliverEvent(rank,ids[rank],MOCK_EVENT_COMPLETION,SPARK_STATUS_BUSY,1u) != 0u,
			"busy admitted prefill: the admitted prefill completes BUSY");
	(void)SparkModelBatchEngineProgress(engine,8u);
	CHECK(SparkModelBatchEngineCancel(engine,handle) == SPARK_STATUS_OK,"busy admitted prefill: cancel while the retry waits");
	TestDrive(engine,200u);
	{
		uint32_t releases_after = 0u;
		for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
			releases_after += MockResidentClientReleaseSubmissions(rank);
		CHECK(releases_after > releases_before,
			"busy admitted prefill: a cancel after an admitted BUSY prefill releases the sequence on the ranks");
	}
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.live_request_count == 0u,
		"busy admitted prefill: the slot is free after the release");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioCancelledInflightPrefillBusy(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	static const uint32_t prompt[4] = {21u,22u,23u,24u};
	TestBatchState state;
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle = 0u;
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine = 0;
	uint64_t ids[TEST_RANKS];
	uint32_t submits_before = 0u,releases_before = 0u,submits_after = 0u,releases_after = 0u;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,1u,0u);
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK,"cancelled busy prefill: connect");
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.top_p = 1.0f;
	request.request_id = 1u;
	request.sequence_id = 560u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 4u;
	request.output_token_budget = 2u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"cancelled busy prefill: submit");
	for (uint32_t step=0u; step<20u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveResults();
		(void)MockResidentClientDriveDecisions();
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		ids[rank] = MockResidentClientPendingEvent(rank,MOCK_EVENT_COMPLETION,0u);
		CHECK(ids[rank] != 0u,"cancelled busy prefill: every rank admitted the prefill");
	}
	CHECK(SparkModelBatchEngineCancel(engine,handle) == SPARK_STATUS_PENDING,"cancelled busy prefill: a cancel while the prefill is in flight is pending");
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		submits_before += MockResidentClientCalls(rank,MOCK_CALL_SUBMIT) + MockResidentClientCalls(rank,MOCK_CALL_PREPARE);
		releases_before += MockResidentClientReleaseSubmissions(rank);
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
		CHECK(MockResidentClientDeliverEvent(rank,ids[rank],MOCK_EVENT_COMPLETION,SPARK_STATUS_BUSY,1u) != 0u,
			"cancelled busy prefill: the admitted prefill completes BUSY");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	TestDrive(engine,50u);
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		submits_after += MockResidentClientCalls(rank,MOCK_CALL_SUBMIT) + MockResidentClientCalls(rank,MOCK_CALL_PREPARE);
		releases_after += MockResidentClientReleaseSubmissions(rank);
	}
	CHECK(state.cancelled_events[1] == 1u && state.error_events[1] == 1u && state.completed_events[1] == 0u && state.token_events[1] == 0u,
		"cancelled busy prefill: the client gets exactly one CANCELLED and no tokens");
	CHECK(submits_after - submits_before == releases_after - releases_before && releases_after > releases_before,
		"cancelled busy prefill: the cancelled prefill is not submitted again; only its release is");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.live_request_count == 0u && view.inflight_submission_count == 0u,
		"cancelled busy prefill: the slot is free after the release");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioPartiallyAdmittedPrefillReleases(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	uint64_t ids[TEST_RANKS];
	uint32_t releases[TEST_RANKS];
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,560u,2u);
	for (uint32_t step=0u; step<20u; step++)
		(void)SparkModelBatchEngineProgress(engine,8u);
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		ids[rank] = MockResidentClientPendingEvent(rank,MOCK_EVENT_RESULT,0u);
		releases[rank] = MockResidentClientReleaseSubmissions(rank);
		CHECK(ids[rank] != 0u,"partial admission: the prefill waits for its result on every rank");
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
		CHECK(MockResidentClientDeliverEvent(rank,ids[rank],MOCK_EVENT_RESULT,rank == 1u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK,1u) != 0u,
			"partial admission: rank 1 fails the prefill result, the others admit it");
	TestDrive(engine,200u);
	CHECK(state.error_events[1] == 1u && state.completed_events[1] == 0u,"partial admission: the request fails once");
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
		CHECK(MockResidentClientReleaseSubmissions(rank) == releases[rank] + 1u,
			"partial admission: the failed prefill is released on every rank, since some of them admitted it");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.live_request_count == 0u,
		"partial admission: the request slot is free after the release");
	TestSubmit(engine,2u,561u,2u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u,"partial admission: the next request on the slot completes");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioCompletionBeforeResult(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngineView view;
	SparkModelBatchEngine *engine = 0;
	SparkModelServingCompletion completion;
	uint64_t ids[TEST_RANKS];
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,1u,0u);
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_OK,"early completion: connect");
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,550u,2u);
	for (uint32_t step=0u; step<20u; step++)
		(void)SparkModelBatchEngineProgress(engine,8u);
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		ids[rank] = MockResidentClientPendingEvent(rank,MOCK_EVENT_RESULT,0u);
		CHECK(ids[rank] != 0u,"early completion: the prefill waits for its result on every rank");
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		memset(&completion,0,sizeof(completion));
		completion.submission_id = ids[rank];
		completion.status = SPARK_STATUS_OK;
		MockResidentClientFireCompletion(rank,&completion);
	}
	for (uint32_t step=0u; step<50u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		usleep(1000);
	}
	CHECK(state.error_events[1] == 1u,"early completion: the client of the broken submission gets one terminal error");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.inflight_submission_count == 0u && view.live_request_count == 0u,
		"early completion: the broken submission and its request slot are released");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioDriverIoError(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 ) return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,500u,2u);
	for (uint32_t step=0u; step<20u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		(void)MockResidentClientDriveResults();
		(void)MockResidentClientDriveDecisions();
	}
	for (uint32_t rank=0u; rank<TEST_RANKS; rank++)
	{
		uint64_t id=MockResidentClientPendingEvent(rank,MOCK_EVENT_COMPLETION,0u);
		CHECK(id != 0u,"driver IO fixture reached committed work");
		CHECK(MockResidentClientDeliverEvent(rank,id,MOCK_EVENT_COMPLETION,
			rank == 1u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK,1u) != 0u,
			"deliver real completion status from each rank");
	}
	TestDrive(engine,200u);
	CHECK(state.error_events[1] == 1u && state.completed_events[1] == 0u && state.token_events[1] == 0u,
		"a driver IO error terminates the request instead of resubmitting inference");
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.live_request_count == 0u,
		"failed driver request releases its request slot");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioEosEarlyStop(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,8u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(154820u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK( state.token_events[1] == 1u,
		"eos: generation stopped at the EOS token instead of the budget");
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"eos: the request completed cleanly on EOS");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioTwoRequestsRankDies(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,3u);
	TestSubmit(engine,2u,501u,3u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDrive(engine,20u);
	MockResidentClientKill(0u);
	TestDrive(engine,30u);
	MockResidentClientRevive(0u);
	TestDriveUntilTerminal(engine,&state,2u,2000u);
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"chaos: first concurrent request completed across the rank death");
	CHECK( state.completed_events[2] == 1u && state.error_events[2] == 0u,
		"chaos: second concurrent request completed across the rank death");
	SparkModelBatchEngineDestroy(engine);
}

static uint64_t TestNowNs(void)
{
	struct timespec now;
	assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void TestScenarioEventDeadlines(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	uint64_t deadline,now;
	uint32_t step;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	CHECK(SparkModelBatchEngineNextProgressNs(engine) == 0u,"idle engine has no polling deadline");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_BUSY);
	TestSubmit(engine,1u,901u,2u);
	CHECK(SparkModelBatchEngineNextProgressNs(engine) == 1u,"new submission is immediately runnable");
	CHECK(SparkModelBatchEngineProgress(engine,8u) == SPARK_STATUS_OK,"dispatch for deadline test");
	CHECK(SparkModelBatchEngineNextProgressNs(engine) > TestNowNs() + UINT64_C(1000000000),"inflight math waits for socket completion or actual timeout");
	deadline = 0u;
	for (step=0u; step<16u; step++)
	{
		(void)MockResidentClientDriveAll();
		CHECK(SparkModelBatchEngineProgress(engine,8u) == SPARK_STATUS_OK,"prepare rejection drains for retry");
		now = TestNowNs();
		deadline = SparkModelBatchEngineNextProgressNs(engine);
		if ( deadline > now && deadline - now <= UINT64_C(320000000) )
			break;
	}
	CHECK(step < 16u,"BUSY advertises a finite retry deadline without polling");
	if ( step < 16u )
	{
		struct timespec delay;
		uint64_t remaining = deadline - now + UINT64_C(1000000);
		delay.tv_sec = (time_t)(remaining / UINT64_C(1000000000));
		delay.tv_nsec = (long)(remaining % UINT64_C(1000000000));
		CHECK(SparkModelBatchEngineNextProgressNs(engine) == deadline,"reading readiness does not extend retry time");
		MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_OK);
		nanosleep(&delay,0);
		TestDriveUntilTerminal(engine,&state,1u,400u);
		CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u,"deadline wake resumes and completes request");
		CHECK(SparkModelBatchEngineProgress(engine,8u) == SPARK_STATUS_OK,"drain terminal progress");
		CHECK(SparkModelBatchEngineNextProgressNs(engine) == 0u,"drained engine returns to event-only wait");
	}
	SparkModelBatchEngineDestroy(engine);
}

static void TestLoadVariantDeployment(const char *runtime_root,const char *name,const char *adapter_path,char *path,size_t path_bytes,SparkModelResidentDeployment *deployment)
{
	(void)snprintf(path,path_bytes,"%s/mock-batch-deployment-%s.json",runtime_root,name);
	TestWriteDeployment(path,runtime_root,adapter_path);
	assert(SparkModelResidentDeploymentLoad(path,deployment) == SPARK_STATUS_OK);
	deployment->eos_token_count = 1u;
	deployment->eos_token_ids[0] = 154820u;
}

static uint32_t TestLongestPrefillChunk(uint64_t request_id,uint32_t prompt_count)
{
	SparkModelServingLane lane;
	uint32_t index,longest = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
		if ( lane.request_id == request_id && lane.context_token_count > lane.sequence_position && lane.context_token_count <= prompt_count &&
			lane.context_token_count - lane.sequence_position > longest )
			longest = lane.context_token_count - lane.sequence_position;
	return(longest);
}

static uint32_t TestPrefillBlockIdentities(uint64_t request_id,const uint32_t *prompt,uint32_t block,uint32_t prompt_count)
{
	SparkModelServingLane lane;
	SparkModelServingCacheIdentity logged;
	SparkSha256Context context;
	uint8_t expected[32];
	uint32_t index,identity,boundary,checked;
	checked = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
	{
		if ( lane.request_id != request_id || lane.context_token_count <= lane.sequence_position || lane.context_token_count > prompt_count )
			continue;
		if ( lane.cache_block_identity_count != (lane.context_token_count - 1u) / block - (uint32_t)lane.sequence_position / block )
			return(UINT32_MAX);
		for (identity=0u; identity<lane.cache_block_identity_count; identity++)
		{
			boundary = ((uint32_t)lane.sequence_position / block + 1u + identity) * block;
			SparkSha256Initialize(&context);
			SparkSha256Update(&context,prompt,(size_t)boundary * sizeof(prompt[0]));
			SparkSha256Finalize(&context,expected);
			if ( MockResidentClientIdentityLog(lane.cache_block_identity_first + identity,&logged) == 0u || memcmp(logged.sha256,expected,sizeof(expected)) != 0 )
				return(UINT32_MAX);
			checked++;
		}
	}
	return(checked);
}

static uint32_t TestPrefillSpans(uint64_t request_id,uint32_t prompt_count,uint32_t *spans,uint32_t capacity)
{
	SparkModelServingLane lane;
	uint32_t index,count = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u && count<capacity; index++)
		if ( lane.request_id == request_id && lane.context_token_count > lane.sequence_position && lane.context_token_count <= prompt_count )
			spans[count++] = lane.context_token_count - lane.sequence_position;
	return(count);
}

static void TestScenarioCapacityRefusedPrefill(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t expected[6] = {8u,4u,4u,4u,4u,2u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	uint32_t prompt[18],spans[16],count,index;
	for (index=0u; index<18u; index++)
		prompt[index] = 50u + index;
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,10u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,670u,1u,prompt,18u);
	MockResidentClientScriptSubmitStatusOnce(1u,SPARK_STATUS_CAPACITY_EXCEEDED);
	TestDriveUntilTerminal(engine,&state,1u,800u);
	count = TestPrefillSpans(1u,18u,spans,16u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK &&
		view.rejected_submission_count_by_status[SPARK_STATUS_CAPACITY_EXCEEDED] == 1u,"capacity refusal: a refused multi-block prefill is requeued and the request completes");
	CHECK(count == 6u && memcmp(spans,expected,sizeof(expected)) == 0,"capacity refusal: the requeued prefill continues in spans half the refused size, still on block boundaries");
	TestSubmitPrompt(engine,2u,671u,1u,prompt,3u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_CAPACITY_EXCEEDED);
	TestDriveUntilTerminal(engine,&state,2u,800u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_OK);
	CHECK(state.error_events[2] == 1u && state.token_events[2] == 0u,"capacity refusal: a refused span of one block or less fails the request instead of retrying forever");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioMultiBlockPrefill(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	uint32_t prompt[18],index;
	for (index=0u; index<18u; index++)
		prompt[index] = 30u + index;
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,10u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,660u,1u,prompt,18u);
	TestDriveUntilTerminal(engine,&state,1u,800u);
	CHECK(state.completed_events[1] == 1u,"multi-block prefill: the prompt completes");
	CHECK(TestPrefillChunksAreCanonical(1u,4u,18u) == 3u && TestLongestPrefillChunk(1u,18u) == 8u,"multi-block prefill: every adapter gets lanes spanning whole cache blocks up to the row budget, ending on a block boundary or the prompt end");
	CHECK(TestPrefillBlockIdentities(1u,prompt,4u,18u) == 2u,"multi-block prefill: each block boundary inside a span carries the identity of the prompt prefix it closes");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestPrefillPublishes(uint64_t request_id,uint32_t prompt_count,uint32_t *ends,uint32_t *stateful,uint32_t capacity)
{
	SparkModelServingLane lane;
	uint32_t index,count = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u && count<capacity; index++)
		if ( lane.request_id == request_id && lane.context_token_count > lane.sequence_position && lane.context_token_count <= prompt_count &&
			(lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) != 0u )
		{
			ends[count] = lane.cache_publish_token_count;
			stateful[count++] = (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_STATELESS) == 0u ? 1u : 0u;
		}
	return(count);
}

static uint32_t TestRowTargets(uint64_t request_id,uint32_t prompt_count,uint32_t *targets,uint32_t capacity,uint32_t *decode_exact)
{
	SparkModelServingLane lane;
	uint32_t index,count = 0u;
	*decode_exact = 1u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
	{
		if ( lane.request_id != request_id || lane.context_token_count <= lane.sequence_position )
			continue;
		if ( lane.context_token_count <= prompt_count && count < capacity )
			targets[count++] = lane.state_target_token_count;
		else if ( lane.context_token_count > prompt_count && lane.state_target_token_count != lane.context_token_count )
			*decode_exact = 0u;
	}
	return(count);
}

static void TestScenarioRecurrentCheckpoints(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t expected_spans[4] = {8u,4u,5u,1u},expected_ends[4] = {8u,12u,17u,18u},expected_stateful[4] = {0u,1u,1u,0u};
	static const uint32_t expected_targets[4] = {12u,12u,17u,18u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t prompt[18],diverge_late[18],diverge_early[18],extended[26],spans[8],ends[8],stateful[8],targets[8],count,index,decode_exact;
	for (index=0u; index<18u; index++)
		prompt[index] = diverge_late[index] = diverge_early[index] = 300u + index;
	diverge_late[14] = 9u;
	diverge_early[10] = 9u;
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,10u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,690u,6u,prompt,18u);
	TestDriveUntilTerminal(engine,&state,1u,800u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && state.token_events[1] == 6u,"recurrent checkpoints: a deferred-publication decode across a block boundary completes");
	count = TestPrefillSpans(1u,18u,spans,8u);
	CHECK(state.completed_events[1] == 1u && count == 4u && memcmp(spans,expected_spans,sizeof(expected_spans)) == 0,"recurrent checkpoints: prefill spans stop at the checkpoint stride and one token before the prompt end, so the last wave is one row");
	count = TestPrefillPublishes(1u,18u,ends,stateful,8u);
	CHECK(count == 4u && memcmp(ends,expected_ends,sizeof(expected_ends)) == 0 && memcmp(stateful,expected_stateful,sizeof(expected_stateful)) == 0,"recurrent checkpoints: every wave end is published but only the stride and the token before the prompt end keep state");
	count = TestRowTargets(1u,18u,targets,8u,&decode_exact);
	CHECK(count == 4u && memcmp(targets,expected_targets,sizeof(expected_targets)) == 0 && decode_exact != 0u,"recurrent checkpoints: each prefill lane names the next position whose state is kept or the prompt end, and decode lanes name their own context");
	TestSubmitPrompt(engine,2u,691u,1u,prompt,18u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 17u,"recurrent checkpoints: a repeated prompt resumes one token before its end");
	TestDriveUntilTerminal(engine,&state,2u,800u);
	TestSubmitPrompt(engine,3u,692u,1u,diverge_late,18u);
	CHECK(TestWaitFirstRequestLane(engine,3u,&lane) != 0u && lane.cache_prefix_token_count == 12u,"recurrent checkpoints: a prompt that diverges after the stride resumes at the stride checkpoint");
	TestDriveUntilTerminal(engine,&state,3u,800u);
	for (index=0u; index<8u; index++)
		extended[18u + index] = 600u + index;
	memcpy(extended,prompt,sizeof(prompt));
	TestSubmitPrompt(engine,5u,694u,1u,extended,26u);
	CHECK(TestWaitFirstRequestLane(engine,5u,&lane) != 0u && lane.cache_prefix_token_count == 17u,"recurrent checkpoints: a prompt that extends a published one resumes exactly at its last checkpoint, inside the partial block");
	TestDriveUntilTerminal(engine,&state,4u,800u);
	TestSubmitPrompt(engine,4u,693u,1u,diverge_early,18u);
	CHECK(TestWaitFirstRequestLane(engine,4u,&lane) != 0u && lane.cache_prefix_token_count == 0u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) == 0u,"recurrent checkpoints: shared blocks without a checkpoint are recomputed instead of resumed");
	TestDriveUntilTerminal(engine,&state,5u,800u);
	CHECK(state.error_events[1] == 0u && state.error_events[2] == 0u && state.error_events[3] == 0u && state.error_events[4] == 0u && state.error_events[5] == 0u,"recurrent checkpoints: no request fails, including the stateless publishes at each request end");
	{
		uint32_t aligned[20],aligned_extended[24];
		for (index=0u; index<20u; index++)
			aligned[index] = aligned_extended[index] = 700u + index;
		for (index=20u; index<24u; index++)
			aligned_extended[index] = 900u + index;
		TestSubmitPrompt(engine,6u,695u,3u,aligned,20u);
		TestDriveUntilTerminal(engine,&state,6u,800u);
		TestSubmitPrompt(engine,7u,696u,1u,aligned_extended,24u);
		CHECK(TestWaitFirstRequestLane(engine,7u,&lane) != 0u && lane.cache_prefix_token_count == 19u,"recurrent checkpoints: a prompt that extends a block-aligned published prompt resumes one token before that prompt's end, inside its last full block");
		TestDriveUntilTerminal(engine,&state,7u,800u);
		CHECK(state.error_events[6] == 0u && state.error_events[7] == 0u && state.completed_events[7] == 1u,"recurrent checkpoints: the block-aligned prompt and its extension complete");
	}
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioRecurrentStaleResume(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelBatchEngineView view;
	SparkModelServingLane lane = {0};
	uint32_t prompt[18],index;
	for (index=0u; index<18u; index++)
		prompt[index] = 500u + index;
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,10u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,695u,1u,prompt,18u);
	TestDriveUntilTerminal(engine,&state,1u,800u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_NOT_FOUND,1u);
	TestSubmitPrompt(engine,2u,696u,1u,prompt,18u);
	TestDriveUntilTerminal(engine,&state,2u,800u);
	MockResidentClientScriptPrefixResult(1u,SPARK_STATUS_OK,0u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u && state.stale_recomputes[2] == 1u,"recurrent stale resume: a resume point the ranks no longer hold costs one retry and the request completes");
	CHECK(TestFindLane(2u,17u,17u,&lane) != 0u && TestFindLane(2u,12u,12u,&lane) != 0u && TestFindLane(2u,0u,0u,&lane) == 0u,"recurrent stale resume: the retry resumes at the previous checkpoint instead of recomputing the prompt");
	TestSubmitPrompt(engine,3u,697u,1u,prompt,18u);
	CHECK(TestWaitFirstRequestLane(engine,3u,&lane) != 0u && lane.cache_prefix_token_count == 17u,"recurrent stale resume: the recomputed checkpoint is a resume point again, so no block of the chain was cut");
	TestDriveUntilTerminal(engine,&state,3u,800u);
	CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK && view.stale_prefix_recompute_count == 1u && state.error_events[3] == 0u,"recurrent stale resume: one stale resume point is one recompute");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioReplyCheckpoint(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t prompt[18],follow[30],index,reply;
	for (index=0u; index<18u; index++)
		prompt[index] = 400u + index;
	MockResidentClientReset();
	engine = TestConnectRows(deployment,&state,runtime_root,10u);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,790u,4u,prompt,18u);
	TestDriveUntilTerminal(engine,&state,1u,800u);
	reply = state.token_events[1];
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && reply == 4u,"reply checkpoint: a four-token reply completes");
	memcpy(follow,prompt,sizeof(prompt));
	for (index=0u; index<reply; index++)
		follow[18u + index] = state.token_ids[1][index];
	for (index=18u + reply; index<30u; index++)
		follow[index] = 700u + index;
	TestSubmitPrompt(engine,2u,791u,1u,follow,30u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 17u + reply,"reply checkpoint: a follow-up that carries the reply resumes after every computed token of the reply, partial last block included, not at the previous prompt's checkpoint");
	TestDriveUntilTerminal(engine,&state,2u,800u);
	CHECK(state.error_events[2] == 0u && state.completed_events[2] == 1u,"reply checkpoint: the follow-up completes");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioLookupVerify(const SparkModelResidentDeployment *deployment,const SparkModelResidentDeployment *plain_deployment,const char *runtime_root);
static void TestScenarioLookupVerifyMixedLanes(const SparkModelResidentDeployment *deployment,const char *runtime_root);
static void TestScenarioModelDrafts(const SparkModelResidentDeployment *deployment,const char *runtime_root);
static void TestScenarioLongVerify(const SparkModelResidentDeployment *deployment,const char *runtime_root);

static void TestScenarioAdapterCacheModes(const char *runtime_root)
{
	SparkModelResidentDeployment deployment;
	char path[512];
	TestLoadVariantDeployment(runtime_root,"without-prefix-reuse",TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_PATH,path,sizeof(path),&deployment);
	TestScenarioAdapterRefused(&deployment,runtime_root,"required cache: an adapter that cannot restore cached prefixes is refused at connect");
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"speculative-inline",TEST_MODEL_SERVING_SPECULATIVE_INLINE_PATH,path,sizeof(path),&deployment);
	TestScenarioAdapterRefused(&deployment,runtime_root,"required cache: a speculating adapter that cannot publish decode checkpoints is refused at connect");
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"inline-checkpoints",TEST_MODEL_SERVING_ADAPTER_PATH,path,sizeof(path),&deployment);
	TestScenarioSpeculationOffPublishesDecodeCheckpoint(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"speculative-deferred",TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_PATH,path,sizeof(path),&deployment);
	TestScenarioSpeculativePublishAdapterDefers(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	{
		SparkModelResidentDeployment plain;
		char plain_path[512];
		TestLoadVariantDeployment(runtime_root,"speculative-verify",TEST_MODEL_SERVING_SPECULATIVE_VERIFY_PATH,path,sizeof(path),&deployment);
		TestLoadVariantDeployment(runtime_root,"plain-decode",TEST_MODEL_SERVING_ADAPTER_PATH,plain_path,sizeof(plain_path),&plain);
		TestScenarioLookupVerify(&deployment,&plain,runtime_root);
		TestScenarioLookupVerifyMixedLanes(&deployment,runtime_root);
		TestScenarioModelDrafts(&deployment,runtime_root);
		SparkModelResidentDeploymentReset(&plain);
		(void)unlink(plain_path);
		SparkModelResidentDeploymentReset(&deployment);
		(void)unlink(path);
	}
	TestLoadVariantDeployment(runtime_root,"speculative-long-verify",TEST_MODEL_SERVING_SPECULATIVE_LONG_VERIFY_PATH,path,sizeof(path),&deployment);
	TestScenarioLongVerify(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"recurrent-checkpoints",TEST_MODEL_SERVING_RECURRENT_CHECKPOINTS_PATH,path,sizeof(path),&deployment);
	TestScenarioRecurrentCheckpoints(&deployment,runtime_root);
	TestScenarioRecurrentStaleResume(&deployment,runtime_root);
	TestScenarioReplyCheckpoint(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"multi-block-prefill",TEST_MODEL_SERVING_ADAPTER_PATH,path,sizeof(path),&deployment);
	TestScenarioMultiBlockPrefill(&deployment,runtime_root);
	TestScenarioCapacityRefusedPrefill(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
}

static uint32_t TestScriptedOutputMatches(const TestBatchState *state,uint64_t request_id,const uint32_t *script,uint32_t prompt_count,uint32_t expected_count)
{
	uint32_t index;
	if ( state->token_events[request_id] != expected_count )
		return(0u);
	for (index=0u; index<expected_count; index++)
		if ( state->token_ids[request_id][index] != script[prompt_count + index] )
			return(0u);
	return(1u);
}

static uint32_t TestLaneInputsFollowScript(uint64_t request_id,const uint32_t *script,uint32_t script_count)
{
	SparkModelServingLane lane;
	uint32_t index,checked = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
	{
		if ( lane.request_id != request_id || (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) == 0u || lane.sequence_position < 44u )
			continue;
		if ( lane.sequence_position >= script_count || lane.input_token_id != script[lane.sequence_position] )
			return(0u);
		checked++;
	}
	return(checked);
}

static void TestRunScript(const SparkModelResidentDeployment *deployment,const char *runtime_root,const uint32_t *script,uint32_t script_count,uint32_t prompt_count,uint32_t budget,TestBatchState *state,SparkModelBatchEngineView *view,uint32_t *verify_submissions,uint32_t *verify_rows)
{
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(state,0,sizeof(*state));
	engine = TestConnectFlags(deployment,state,runtime_root,32u,8u,0u);
	if ( engine == 0 )
		return;
	MockResidentClientSetScript(script,script_count);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,900u,budget,script,prompt_count);
	TestDriveUntilTerminal(engine,state,1u,2000u);
	memset(view,0,sizeof(*view));
	CHECK(SparkModelBatchEngineGetView(engine,view) == SPARK_STATUS_OK,"lookup: engine view");
	MockResidentClientTakeVerifyStats(verify_submissions,verify_rows);
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioLookupVerify(const SparkModelResidentDeployment *deployment,const SparkModelResidentDeployment *plain_deployment,const char *runtime_root)
{
	uint32_t script[256],index,submissions,rows,plain_submissions,plain_rows;
	SparkModelBatchEngineView view,plain_view;
	TestBatchState state,plain_state;
	for (index=0u; index<40u; index++)
		script[index] = 1000u + (index * 7u) % 97u;
	for (index=40u; index<44u; index++)
		script[index] = 2000u + index;
	for (index=44u; index<256u; index++)
		script[index] = script[(index - 44u) % 40u];
	TestRunScript(deployment,runtime_root,script,256u,44u,80u,&state,&view,&submissions,&rows);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,44u,80u) != 0u,"lookup: an edit-style request emits exactly the model's greedy stream");
	CHECK(submissions != 0u && rows > submissions && view.speculative_verify_lane_count != 0u,"lookup: a repeated passage is verified in multi-row waves");
	CHECK(view.speculative_accepted_token_count * 2u > view.speculative_draft_token_count && view.speculative_accepted_token_count >= 30u,"lookup: drafts from the repeated passage are accepted");
	CHECK(TestLaneInputsFollowScript(1u,script,256u) != 0u,"lookup: every decode lane's input token is a committed model token");
	TestRunScript(plain_deployment,runtime_root,script,256u,44u,80u,&plain_state,&plain_view,&plain_submissions,&plain_rows);
	CHECK(TestScriptedOutputMatches(&plain_state,1u,script,44u,80u) != 0u && plain_submissions == 0u && plain_view.speculative_draft_token_count == 0u,"lookup: an adapter without verify decodes the same stream with no verify waves");
	for (index=0u; index<256u; index++)
		script[index] = 3000u + (index * 37u + index * index * 11u) % 4093u;
	TestRunScript(deployment,runtime_root,script,256u,44u,40u,&state,&view,&submissions,&rows);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,40u) != 0u && submissions == 0u && view.speculative_draft_token_count == 0u,"lookup: a prompt with no repeated suffix issues no verify rows");
	for (index=0u; index<40u; index++)
		script[index] = 1000u + (index * 7u) % 97u;
	for (index=40u; index<44u; index++)
		script[index] = 2000u + index;
	for (index=44u; index<256u; index++)
		script[index] = script[(index - 44u) % 40u];
	for (index=60u; index<256u; index+=9u)
		script[index] = 5000u + index;
	TestRunScript(deployment,runtime_root,script,256u,44u,90u,&state,&view,&submissions,&rows);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,90u) != 0u,"lookup: an edit that diverges from its source still emits exactly the greedy stream");
	CHECK(view.speculative_accepted_token_count < view.speculative_draft_token_count && view.speculative_accepted_token_count != 0u,"lookup: drafts past a divergence are rejected and earlier ones accepted");
	CHECK(TestLaneInputsFollowScript(1u,script,256u) != 0u,"lookup: a rejected draft never becomes a lane's input");
	for (index=44u; index<256u; index++)
		script[index] = script[(index - 44u) % 40u];
	TestRunScript(deployment,runtime_root,script,256u,44u,23u,&state,&view,&submissions,&rows);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,44u,23u) != 0u,"lookup: drafts never run past the output budget");
	script[44u + 30u] = 154820u;
	TestRunScript(deployment,runtime_root,script,256u,44u,80u,&state,&view,&submissions,&rows);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,44u,31u) != 0u,"lookup: a stop token inside accepted drafts ends the request there");
}

#define TEST_DRAFT_CORRECT 0u
#define TEST_DRAFT_LOW 1u
#define TEST_DRAFT_WRONG 2u
#define TEST_DRAFT_UNAVAILABLE 3u
#define TEST_DRAFT_CLOSE 4u
#define TEST_DRAFT_FIRST 5u

typedef struct TestDraftServer
{
	int listener;
	uint32_t port;
	uint32_t mode;
	const uint32_t *script;
	uint32_t script_count;
	uint32_t requests;
	uint32_t bad_requests;
	uint32_t connections;
	uint32_t stop;
	pthread_t thread;
} TestDraftServer;

static uint32_t TestDraftRead(int fd,uint8_t *data,size_t bytes)
{
	size_t got = 0u;
	while ( got < bytes )
	{
		const ssize_t part = recv(fd,data + got,bytes - got,0);
		if ( part <= 0 )
			return(0u);
		got += (size_t)part;
	}
	return(1u);
}

static uint32_t TestDraftU32(const uint8_t *data)
{
	return((uint32_t)data[0] | ((uint32_t)data[1] << 8u) | ((uint32_t)data[2] << 16u) | ((uint32_t)data[3] << 24u));
}

static uint64_t TestDraftU64(const uint8_t *data)
{
	return((uint64_t)TestDraftU32(data) | ((uint64_t)TestDraftU32(data + 4u) << 32u));
}

static void TestDraftPut32(uint8_t **cursor,uint32_t value)
{
	memcpy(*cursor,&value,sizeof(value));
	*cursor += sizeof(value);
}

static void TestDraftServe(TestDraftServer *server,int fd)
{
	static uint8_t committed[4u * 4096u];
	uint8_t header[120],reply[28u + 9u * 20u + 48u],*cursor;
	for (;;)
	{
		uint64_t sequence,generation,position;
		uint32_t anchor,count,depth,nodes = 0u,status = 0u,index;
		if ( TestDraftRead(fd,header,sizeof(header)) == 0u )
			return;
		sequence = TestDraftU64(header + 44u);
		generation = TestDraftU64(header + 52u);
		position = TestDraftU64(header + 60u);
		anchor = TestDraftU32(header + 68u);
		count = TestDraftU32(header + 72u);
		depth = TestDraftU32(header + 80u);
		if ( count > sizeof(committed) / 4u || TestDraftRead(fd,committed,(size_t)count * 4u) == 0u )
			return;
		server->requests++;
		if ( memcmp(header,"DFT3",4u) != 0 || TestDraftU32(header + 4u) != 2u || (TestDraftU32(header + 8u) & 2u) == 0u ||
			strncmp((const char *)header + 12u,"test-target",32u) != 0 || count != position + 1u || position >= server->script_count ||
			anchor != server->script[position] || TestDraftU32(header + 76u) != 0u || depth == 0u || depth > 7u )
			server->bad_requests++;
		for (index=0u; index<count && index<server->script_count; index++)
			if ( TestDraftU32(committed + 4u * index) != server->script[index] )
			{
				server->bad_requests++;
				break;
			}
		if ( server->mode == TEST_DRAFT_CLOSE )
			return;
		cursor = reply;
		memcpy(cursor,"DFT3",4u);
		cursor += 4u;
		if ( server->mode == TEST_DRAFT_UNAVAILABLE )
			status = 4u;
		else
			nodes = 1u + depth;
		TestDraftPut32(&cursor,status);
		TestDraftPut32(&cursor,nodes);
		memcpy(cursor,&sequence,8u);
		cursor += 8u;
		memcpy(cursor,&generation,8u);
		cursor += 8u;
		for (index=0u; index<nodes; index++)
		{
			const uint64_t at = position + index;
			const uint32_t wrong = server->mode == TEST_DRAFT_WRONG || (server->mode == TEST_DRAFT_FIRST && index >= 2u) ? 1u : 0u;
			const uint32_t token = index == 0u ? anchor : at < server->script_count ? server->script[at] + wrong : 7u;
			const float score = index == 0u ? 1.0f : server->mode == TEST_DRAFT_LOW ? 0.1f : 0.9f;
			TestDraftPut32(&cursor,token);
			TestDraftPut32(&cursor,index == 0u ? UINT32_MAX : index - 1u);
			TestDraftPut32(&cursor,index);
			TestDraftPut32(&cursor,2u);
			memcpy(cursor,&score,4u);
			cursor += 4u;
		}
		memset(cursor,0,48u);
		cursor += 48u;
		if ( send(fd,reply,(size_t)(cursor - reply),0) != (ssize_t)(cursor - reply) )
			return;
	}
}

static void *TestDraftThread(void *argument)
{
	TestDraftServer *server = (TestDraftServer *)argument;
	while ( server->stop == 0u )
	{
		const int fd = accept(server->listener,0,0);
		if ( fd < 0 )
			break;
		server->connections++;
		TestDraftServe(server,fd);
		(void)close(fd);
	}
	return(0);
}

static uint32_t TestDraftStart(TestDraftServer *server,uint32_t mode,const uint32_t *script,uint32_t script_count)
{
	struct sockaddr_in address;
	socklen_t bytes = (socklen_t)sizeof(address);
	memset(server,0,sizeof(*server));
	server->mode = mode;
	server->script = script;
	server->script_count = script_count;
	server->listener = socket(AF_INET,SOCK_STREAM,0);
	memset(&address,0,sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if ( server->listener < 0 || bind(server->listener,(const struct sockaddr *)&address,bytes) != 0 || listen(server->listener,4) != 0 ||
		getsockname(server->listener,(struct sockaddr *)&address,&bytes) != 0 )
		return(0u);
	server->port = ntohs(address.sin_port);
	return(pthread_create(&server->thread,0,TestDraftThread,server) == 0 ? 1u : 0u);
}

static void TestDraftStop(TestDraftServer *server)
{
	server->stop = 1u;
	(void)shutdown(server->listener,SHUT_RDWR);
	(void)close(server->listener);
	(void)pthread_join(server->thread,0);
}

static uint32_t TestHistoryMatch(const uint32_t *script,uint32_t position)
{
	uint32_t end,length,best = 0u;
	for (end=0u; end<position; end++)
	{
		for (length=0u; length<64u && length<=end && script[end - length] == script[position - length]; length++)
			;
		if ( length > best )
			best = length;
	}
	return(best);
}

static uint32_t TestModelFramesPastDepth(const uint32_t *script,uint32_t depth,uint32_t *short_match_frames)
{
	uint32_t index,position,rows,past = 0u;
	*short_match_frames = 0u;
	for (index=0u; MockResidentClientVerifyFrame(index,&position,&rows) != 0u; index++)
		if ( TestHistoryMatch(script,position) < 8u )
		{
			(*short_match_frames)++;
			if ( rows > depth + 1u )
				past++;
		}
	return(past);
}

static void TestRunModelDrafts(const SparkModelResidentDeployment *deployment,const char *runtime_root,uint32_t mode,const uint32_t *script,uint32_t budget,
	TestBatchState *state,SparkModelBatchEngineView *view,uint32_t *verify_submissions,uint32_t *verify_rows,TestDraftServer *server)
{
	SparkModelResidentDeployment drafted = *deployment;
	static char host[] = "127.0.0.1";
	CHECK(TestDraftStart(server,mode,script,256u) != 0u,"model drafts: a stub draft server listens");
	drafted.draft_server.host = host;
	drafted.draft_server.port = server->port;
	strcpy(drafted.draft_server.target_model,"test-target");
	drafted.draft_server.depth = 4u;
	drafted.draft_server.wait_ms = 50u;
	drafted.draft_server.min_confidence = 0.05f;
	MockResidentClientSetStepDelay(2000u);
	TestRunScript(&drafted,runtime_root,script,256u,44u,budget,state,view,verify_submissions,verify_rows);
	MockResidentClientSetStepDelay(0u);
	TestDraftStop(server);
}

static void TestScenarioModelDrafts(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	uint32_t script[256],index,submissions,rows,short_frames;
	SparkModelBatchEngineView view;
	TestBatchState state;
	TestDraftServer server;
	for (index=0u; index<256u; index++)
		script[index] = 3000u + (index * 37u + index * index * 11u) % 4093u;
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_CORRECT,script,80u,&state,&view,&submissions,&rows,&server);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,44u,80u) != 0u,"model drafts: a stream with no repeats emits exactly the model's greedy stream");
	CHECK(TestModelFramesPastDepth(script,4u,&short_frames) == 0u && short_frames != 0u,
		"model drafts: with no earlier passage to continue, a model frame never runs past the drafter's depth");
	CHECK(submissions != 0u && rows > submissions && view.speculative_accepted_token_count >= 40u,"model drafts: confident drafts are verified in multi-row frames and accepted");
	CHECK(server.requests != 0u && server.bad_requests == 0u,"model drafts: each request names the target, the anchor and every committed token");
	CHECK(TestLaneInputsFollowScript(1u,script,256u) != 0u,"model drafts: every decode lane's input token is a committed model token");
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_LOW,script,40u,&state,&view,&submissions,&rows,&server);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,40u) != 0u && submissions == 0u && server.requests != 0u,"model drafts: drafts below the confidence cut are never verified");
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_WRONG,script,40u,&state,&view,&submissions,&rows,&server);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,40u) != 0u && submissions != 0u && view.speculative_accepted_token_count == 0u,
		"model drafts: wrong drafts are rejected and the stream stays exact");
	CHECK(submissions <= 3u && server.requests >= submissions + 3u,"model drafts: confident drafts that keep failing stop being verified once their acceptance is learned");
	CHECK(TestLaneInputsFollowScript(1u,script,256u) != 0u,"model drafts: a rejected model draft never becomes a lane's input");
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_FIRST,script,80u,&state,&view,&submissions,&rows,&server);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,80u) != 0u && submissions >= 20u && view.speculative_accepted_token_count * 2u + 8u >= submissions * 2u,
		"model drafts: a draft whose first token is right keeps that token in every frame");
	CHECK(rows <= 2u * submissions + 12u,
		"model drafts: confident deep drafts that fail stop being verified while the first draft that holds keeps its row");
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_UNAVAILABLE,script,40u,&state,&view,&submissions,&rows,&server);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,40u) != 0u && submissions == 0u && server.requests > 4u,
		"model drafts: an unavailable answer declines one round and the engine keeps asking");
	MockResidentClientSetVerifyRowCost(21u);
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_CORRECT,script,80u,&state,&view,&submissions,&rows,&server);
	MockResidentClientSetVerifyRowCost(1u);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,80u) != 0u && submissions != 0u && submissions <= 10u && server.requests == submissions,
		"model drafts: frames that turn out slower than chains back the request off to chains, with no draft requests during the backoff");
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_CLOSE,script,40u,&state,&view,&submissions,&rows,&server);
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,40u) != 0u && submissions == 0u && server.requests == 1u,
		"model drafts: a broken draft server pauses model drafts and decoding continues");
	{
		SparkModelResidentDeployment drafted = *deployment;
		static char host[] = "127.0.0.1";
		struct sockaddr_in address;
		socklen_t bytes = (socklen_t)sizeof(address);
		int closed = socket(AF_INET,SOCK_STREAM,0);
		memset(&address,0,sizeof(address));
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
		CHECK(closed >= 0 && bind(closed,(const struct sockaddr *)&address,bytes) == 0 && getsockname(closed,(struct sockaddr *)&address,&bytes) == 0,
			"model drafts: a bound port with no listener stands in for a draft server that is down");
		drafted.draft_server.host = host;
		drafted.draft_server.port = ntohs(address.sin_port);
		strcpy(drafted.draft_server.target_model,"test-target");
		drafted.draft_server.depth = 4u;
		drafted.draft_server.wait_ms = 50u;
		drafted.draft_server.min_confidence = 0.05f;
		memset(&state,0,sizeof(state));
		TestRunScript(&drafted,runtime_root,script,256u,44u,40u,&state,&view,&submissions,&rows);
		CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,44u,40u) != 0u && submissions == 0u,
			"model drafts: an engine whose draft server is down at start still starts and emits exactly the model's greedy stream");
		(void)close(closed);
	}
	for (index=0u; index<40u; index++)
		script[index] = 1000u + (index * 7u) % 97u;
	for (index=40u; index<50u; index++)
		script[index] = 6000u + index;
	for (index=50u; index<80u; index++)
		script[index] = script[index - 40u];
	for (index=80u; index<256u; index++)
		script[index] = 3000u + (index * 37u + index * index * 11u) % 4093u;
	TestRunModelDrafts(deployment,runtime_root,TEST_DRAFT_CORRECT,script,60u,&state,&view,&submissions,&rows,&server);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,44u,60u) != 0u,
		"model drafts: an edit that resumes an earlier passage after an insertion emits exactly the model's greedy stream");
	CHECK(TestModelFramesPastDepth(script,4u,&short_frames) != 0u,
		"model drafts: a model draft that completes an earlier passage carries the passage's continuation in the same frame");
	CHECK(view.speculative_accepted_token_count >= 40u && TestLaneInputsFollowScript(1u,script,256u) != 0u,
		"model drafts: the carried continuation is accepted and only committed tokens become lane inputs");
}

static void TestScenarioLookupVerifyMixedLanes(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	uint32_t script[256],index,submissions,rows;
	SparkModelBatchRequestHandle handle;
	SparkModelBatchSubmitRequest request;
	SparkModelBatchEngine *engine;
	TestBatchState state;
	for (index=0u; index<40u; index++)
		script[index] = 1000u + (index * 7u) % 97u;
	for (index=40u; index<44u; index++)
		script[index] = 2000u + index;
	for (index=44u; index<256u; index++)
		script[index] = script[(index - 44u) % 40u];
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnectFlags(deployment,&state,runtime_root,32u,8u,0u);
	if ( engine == 0 )
		return;
	MockResidentClientSetScript(script,256u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,910u,60u,script,44u);
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.request_id = 2u;
	request.sequence_id = 911u;
	request.prompt_token_ids = script;
	request.prompt_token_count = 44u;
	request.output_token_budget = 60u;
	request.temperature = 0.7f;
	request.seed = 5u;
	request.top_p = 1.0f;
	handle = 0u;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"lookup mixed: sampled request admitted");
	TestDriveUntilTerminal(engine,&state,2u,3000u);
	MockResidentClientTakeVerifyStats(&submissions,&rows);
	CHECK(state.completed_events[1] == 1u && state.completed_events[2] == 1u,"lookup mixed: both requests complete");
	CHECK(TestScriptedOutputMatches(&state,1u,script,44u,60u) != 0u && TestScriptedOutputMatches(&state,2u,script,44u,60u) != 0u,"lookup mixed: greedy and sampled lanes both emit the model stream");
	CHECK(MockResidentClientTakeVerifyShared() == 0u,"lookup mixed: a verify frame never carries a co-batched lane, which would lose its chain");
	CHECK(TestLaneInputsFollowScript(2u,script,256u) != 0u,"lookup mixed: every sampled-lane input is a committed model token");
	SparkModelBatchEngineDestroy(engine);
}

static void TestLongVerifyScript(uint32_t *script,uint32_t count,uint32_t passage,uint32_t prompt)
{
	uint32_t index;
	for (index=0u; index<passage; index++)
		script[index] = 3000u + (index * 37u + index * index * 11u) % 4093u;
	for (index=passage; index<prompt; index++)
		script[index] = 9000u + index;
	for (index=prompt; index<count; index++)
		script[index] = script[(index - prompt) % passage];
}

static void TestRunLongVerify(const SparkModelResidentDeployment *deployment,const char *runtime_root,const uint32_t *script,uint32_t prompt,uint32_t budget,
	TestBatchState *state,SparkModelBatchEngineView *view,uint32_t *submissions,uint32_t *rows,uint32_t *stride,uint32_t *continues)
{
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(state,0,sizeof(*state));
	engine = TestConnectFlags(deployment,state,runtime_root,256u,8u,0u);
	if ( engine == 0 )
		return;
	MockResidentClientSetScript(script,512u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,920u,budget,script,prompt);
	TestDriveUntilTerminal(engine,state,1u,4000u);
	memset(view,0,sizeof(*view));
	CHECK(SparkModelBatchEngineGetView(engine,view) == SPARK_STATUS_OK,"long verify: engine view");
	MockResidentClientTakeVerifyStats(submissions,rows);
	*stride = MockResidentClientTakeVerifyMaxStride();
	*continues = MockResidentClientCalls(0u,MOCK_CALL_CONTINUE);
	MockResidentClientSetVerifyCorrupt(0u);
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioLongVerify(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	uint32_t script[512],submissions,rows,stride,continues,index;
	SparkModelBatchEngineView view;
	TestBatchState state;
	TestLongVerifyScript(script,512u,120u,124u);
	TestRunLongVerify(deployment,runtime_root,script,124u,120u,&state,&view,&submissions,&rows,&stride,&continues);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,124u,120u) != 0u,"long verify: a whole-file copy emits exactly the model's greedy stream");
	CHECK(stride > SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE + 1u && rows > 4u * submissions,"long verify: a long copy is verified in frames longer than a decode chain");
	CHECK(view.speculative_accepted_token_count >= 90u && view.speculative_accepted_token_count == view.speculative_draft_token_count,"long verify: every draft of an exact copy is kept");
	CHECK(submissions != 0u && continues >= submissions,"long verify: every frame after a verify continues on the lease");
	CHECK(TestLaneInputsFollowScript(1u,script,512u) != 0u,"long verify: every decode lane's input token is a committed model token");
	for (index=124u + 30u; index<512u; index+=45u)
		script[index] = 6000u + index;
	TestRunLongVerify(deployment,runtime_root,script,124u,120u,&state,&view,&submissions,&rows,&stride,&continues);
	CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,124u,120u) != 0u,"long verify: an edited copy emits exactly the greedy stream");
	CHECK(view.speculative_accepted_token_count < view.speculative_draft_token_count && view.speculative_accepted_token_count >= 60u,"long verify: drafts past an edit are rejected and the rest kept");
	CHECK(submissions != 0u && continues >= submissions,"long verify: the lease holds after a rejecting verify");
	CHECK(TestLaneInputsFollowScript(1u,script,512u) != 0u,"long verify: a rejected draft never becomes a lane's input");
	TestLongVerifyScript(script,512u,120u,124u);
	MockResidentClientReset();
	{
		SparkModelBatchSubmitRequest request;
		SparkModelBatchRequestHandle handle = 0u;
		SparkModelBatchEngine *engine;
		memset(&state,0,sizeof(state));
		engine = TestConnectFlags(deployment,&state,runtime_root,256u,8u,0u);
		if ( engine == 0 )
			return;
		MockResidentClientSetScript(script,512u);
		MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
		memset(&request,0,sizeof(request));
		request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
		request.descriptor_bytes = sizeof(request);
		request.request_id = 1u;
		request.sequence_id = 922u;
		request.prompt_token_ids = script;
		request.prompt_token_count = 124u;
		request.output_token_budget = 120u;
		request.temperature = 1.0f;
		request.top_p = 0.95f;
		request.seed = 7u;
		CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"long verify: a sampled request is admitted");
		TestDriveUntilTerminal(engine,&state,1u,4000u);
		memset(&view,0,sizeof(view));
		CHECK(SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK,"long verify: engine view");
		CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,124u,120u) != 0u && view.speculative_accepted_token_count >= 90u,
			"long verify: a sampled request drafts from its history and emits exactly the model's sampled stream");
		SparkModelBatchEngineDestroy(engine);
	}
	{
		uint32_t pass,strides[2];
		for (pass=0u; pass<2u; pass++)
		{
			SparkModelBatchSubmitRequest request;
			SparkModelBatchRequestHandle handle = 0u;
			SparkModelBatchEngine *engine;
			MockResidentClientReset();
			memset(&state,0,sizeof(state));
			engine = TestConnectFlags(deployment,&state,runtime_root,256u,8u,0u);
			if ( engine == 0 )
				return;
			MockResidentClientSetScript(script,512u);
			MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
			MockResidentClientSetRowCost(10000u);
			if ( pass == 1u )
			{
				memset(&request,0,sizeof(request));
				request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
				request.descriptor_bytes = sizeof(request);
				request.request_id = 2u;
				request.sequence_id = 924u;
				request.prompt_token_ids = script;
				request.prompt_token_count = 124u;
				request.output_token_budget = 1u;
				request.top_p = 1.0f;
				request.priority = 1u;
				CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK,"long verify: a foreground request is admitted");
				TestDriveUntilTerminal(engine,&state,1u,4000u);
			}
			TestSubmitPrompt(engine,1u,923u,120u,script,124u);
			TestDriveUntilTerminal(engine,&state,1u + pass,4000u);
			CHECK(state.completed_events[1] == 1u && TestScriptedOutputMatches(&state,1u,script,124u,120u) != 0u,"long verify: a copy at either priority emits exactly the greedy stream");
			strides[pass] = MockResidentClientTakeVerifyMaxStride();
			SparkModelBatchEngineDestroy(engine);
		}
		CHECK(strides[0] > 32u,"long verify: a request at the default priority verifies frames longer than eight chained tokens cost");
		CHECK(strides[1] != 0u && strides[1] <= SPARK_SPECULATION_FRAME_PROBE_DRAFTS + 1u,"long verify: a request below a served foreground priority keeps its frames priced within eight chained tokens");
	}
	script[124u + 40u] = 6100u;
	MockResidentClientReset();
	{
		SparkModelBatchEngine *engine;
		uint32_t emitted,prefix = 1u;
		memset(&state,0,sizeof(state));
		engine = TestConnectFlags(deployment,&state,runtime_root,256u,8u,0u);
		if ( engine == 0 )
			return;
		MockResidentClientSetScript(script,512u);
		MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
		MockResidentClientSetVerifyCorrupt(1u);
		TestSubmitPrompt(engine,1u,921u,120u,script,124u);
		TestDriveUntilTerminal(engine,&state,1u,4000u);
		MockResidentClientSetVerifyCorrupt(0u);
		emitted = state.token_events[1] < 128u ? state.token_events[1] : 128u;
		for (index=0u; index<emitted; index++)
			if ( state.token_ids[1][index] != script[124u + index] )
				prefix = 0u;
		CHECK(state.completed_events[1] == 0u && state.error_events[1] == 1u && emitted < 120u && prefix != 0u,
			"long verify: a node kept count that disagrees with the drafts fails the request and emits no wrong token");
		(void)SparkModelBatchEngineDestroy(engine);
	}
}

static void TestScenarioDecodeAfterPrefill(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine;
	uint32_t prompt[16] = {11u,12u,13u,14u,15u,16u,17u,18u,19u,20u,21u,22u,23u,24u,25u,26u};
	uint32_t index,last_first,first_second;
#ifndef DEBUG
	MockResidentClientReset();
	engine = 0;
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,8u,SPARK_MODEL_BATCH_ENGINE_FLAG_DECODE_AFTER_PREFILL);
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_INVALID_ARGUMENT && engine == 0,"decode after prefill: a release engine refuses the DEBUG measurement barrier");
	return;
#endif
	MockResidentClientReset();
	engine = TestConnectFlags(deployment,&state,runtime_root,4u,8u,SPARK_MODEL_BATCH_ENGINE_FLAG_DECODE_AFTER_PREFILL);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,701u,3u,prompt,4u);
	for (index=2u; index<=6u; index++)
		TestSubmitPrompt(engine,index,700u + index,3u,prompt,16u);
	TestDriveUntilTerminal(engine,&state,6u,4000u);
	last_first = 0u;
	first_second = UINT32_MAX;
	for (index=1u; index<=6u; index++)
	{
		CHECK(state.completed_events[index] == 1u && state.token_events[index] == 3u,"decode after prefill: every request completes its budget");
		last_first = state.first_token_order[index] > last_first ? state.first_token_order[index] : last_first;
		first_second = state.second_token_order[index] < first_second ? state.second_token_order[index] : first_second;
	}
	CHECK(last_first != 0u && last_first < first_second,"decode after prefill: no request decodes before every prompt has its first token");
	SparkModelBatchEngineDestroy(engine);
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnectFlags(deployment,&state,runtime_root,4u,40u,SPARK_MODEL_BATCH_ENGINE_FLAG_DECODE_AFTER_PREFILL);
	if ( engine == 0 )
		return;
	for (index=1u; index<=32u; index++)
		CHECK(TestSubmitPromptStatus(engine,index,800u + index,3u,prompt,8u) == SPARK_STATUS_OK,"decode after prefill: requests up to the resident capacity are admitted");
	CHECK(TestSubmitPromptStatus(engine,33u,833u,3u,prompt,8u) == SPARK_STATUS_CAPACITY_EXCEEDED,"decode after prefill: a request past the resident capacity is refused before it can deadlock the barrier");
	SparkModelBatchEngineDestroy(engine);
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = 0;
	TestConfigure(&configuration,deployment,&state,runtime_root,4u,8u,UINT32_C(0x00000002));
	CHECK(SparkModelBatchEngineConnect(&configuration,&engine) == SPARK_STATUS_INVALID_ARGUMENT && engine == 0,"decode after prefill: an unknown engine flag is refused");
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	char path[512];
	char runtime_root[256];

	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	(void)snprintf(path,sizeof(path),"%s/mock-batch-deployment.json",runtime_root);
	TestWriteDeployment(path,runtime_root,TEST_MODEL_SERVING_ADAPTER_PATH);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == SPARK_STATUS_OK);
	deployment.eos_token_count = 1u;
	deployment.eos_token_ids[0] = 154820u;

	char *coordinator_root = deployment.nodes[0].runtime_root;
	deployment.nodes[0].runtime_root = "/remote-resident-only/runtime";
	TestScenarioHappyPath(&deployment,runtime_root);
	deployment.nodes[0].runtime_root = coordinator_root;
	TestScenarioEventDeadlines(&deployment,runtime_root);
	TestScenarioHappyPath(&deployment,runtime_root);
	TestScenarioDeploymentPositionLimit(&deployment,runtime_root);
	TestScenarioRankDiesMidDecode(&deployment,runtime_root);
	TestScenarioRankKilledAndRevived(&deployment,runtime_root);
	TestScenarioCachedPrefixSessionReset(&deployment,runtime_root);
	TestScenarioPrefixIndexSurvivesRestart(&deployment,runtime_root);
	TestScenarioPeerPrefixIndex(&deployment,runtime_root);
	TestScenarioPrefixIndexRefusesCorruptFile(&deployment,runtime_root);
	TestScenarioPrefixIndexRemovesStaleTemporaries(&deployment,runtime_root);
	TestScenarioStatusReport(&deployment,runtime_root);
	TestScenarioPartialPrefixAppend(&deployment,runtime_root);
	TestScenarioChainPublishesFinalCheckpoint(&deployment,runtime_root);
	TestScenarioChainPublishesAtBlockBoundary(&deployment,runtime_root);
	TestScenarioSingleTokenEosPublishesTail(&deployment,runtime_root);
	TestScenarioChainEosCheckpoint(&deployment,runtime_root);
	TestScenarioGeneratedCheckpointIdentity(&deployment,runtime_root,1u);
	TestScenarioGeneratedCheckpointIdentity(&deployment,runtime_root,3u);
	TestScenarioCanonicalPrefillChunks(&deployment,runtime_root);
	TestScenarioSharedPrefillRows(&deployment,runtime_root);
	TestScenarioDecodeAfterPrefill(&deployment,runtime_root);
	TestScenarioPartialCopyCapacity(&deployment,runtime_root);
	TestScenarioSamplingValidation(&deployment,runtime_root);
	TestScenarioLogprobs(&deployment,runtime_root);
	TestScenarioTruncatedChains(&deployment,runtime_root);
	TestScenarioAdapterContextLimit(&deployment,runtime_root);
	TestScenarioChainYieldsToPrefill(&deployment,runtime_root);
	TestScenarioMeasurements(&deployment,runtime_root);
	TestScenarioStalePrefixRecomputes(&deployment,runtime_root);
	TestScenarioStalePrefixTerminates(&deployment,runtime_root);
	TestScenarioStalePrefixIsolatesLanes(&deployment,runtime_root);
	TestScenarioSharedPrefixReservation(&deployment,runtime_root);
	TestScenarioVerificationFailureIsFatal(&deployment,runtime_root);
	TestScenarioRankBusyBackpressure(&deployment,runtime_root);
	TestScenarioBusyRestoreDeadline(&deployment,runtime_root);
	TestScenarioPendingRestoreRetries(&deployment,runtime_root);
	TestScenarioDriverIoError(&deployment,runtime_root);
	TestScenarioPartiallyAdmittedPrefillReleases(&deployment,runtime_root);
	TestScenarioInflightExpiryHoldsSlot(&deployment,runtime_root);
	TestScenarioBusyAdmittedPrefillReleases(&deployment,runtime_root);
	TestScenarioCancelledInflightPrefillBusy(&deployment,runtime_root);
	TestScenarioCompletionBeforeResult(&deployment,runtime_root);
	TestScenarioEosEarlyStop(&deployment,runtime_root);
	TestScenarioTwoRequestsRankDies(&deployment,runtime_root);
	TestScenarioAdapterCacheModes(runtime_root);

	SparkModelResidentDeploymentReset(&deployment);
	MockResidentClientReset();
	(void)unlink(path);

	fprintf(stderr,"%s: %u checks, %u failures\n",
		"test_model_batch_engine_mock", test_checks, test_failures);
	return( test_failures != 0u ? 1 : 0 );
}
