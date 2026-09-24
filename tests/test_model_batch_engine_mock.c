#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "mock_model_resident_client.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
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
	uint32_t total_terminals;
	uint32_t cached_tokens[TEST_MAX_REQUESTS + 1u];
} TestBatchState;

static void TestBatchEvent(void *context, const SparkModelBatchEvent *event)
{
	TestBatchState *s = (TestBatchState *)context;
	uint64_t id = event->request_id;
	if ( id > TEST_MAX_REQUESTS )
		return;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
	{
		s->token_events[id]++;
		s->cached_tokens[id] = event->cached_prompt_token_count;
	}
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
	{
		s->completed_events[id]++;
		s->total_terminals++;
	}
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

static void TestWriteDeployment(const char *path, const char *runtime_root)
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
	fixture.adapter_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
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
	fixture.runtime_limits.max_input_row_count = 32u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.control_port_base = 59200u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static SparkModelBatchEngine *TestConnect(const SparkModelResidentDeployment *deployment, TestBatchState *state, const char *runtime_root)
{
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine;
	SparkStatus status;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.connect_timeout_ms = 1000u;
	configuration.request_capacity = 8u;
	configuration.max_context_tokens = 256u;
	configuration.max_prefill_rows_per_submission = 4u;
	configuration.maximum_messages_per_rank_per_progress = 8u;
	configuration.inflight_budget_ns = SPARK_MODEL_BATCH_ENGINE_DEFAULT_INFLIGHT_BUDGET_NS;
	configuration.deployment = deployment;
	configuration.runtime_root = runtime_root;
	configuration.event_function = TestBatchEvent;
	configuration.event_context = state;
	engine = 0;
	status = SparkModelBatchEngineConnect(&configuration,&engine);
	CHECK(status == SPARK_STATUS_OK, "batch engine connect");
	return(engine);
}

static void TestSubmitPrompt(SparkModelBatchEngine *engine, uint64_t request_id, uint64_t sequence_id, uint32_t budget,const uint32_t *prompt,uint32_t prompt_count)
{
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.request_id = request_id;
	request.sequence_id = sequence_id;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = prompt_count;
	request.output_token_budget = budget;
	handle = 0;
	status = SparkModelBatchEngineSubmit(engine,&request,&handle);
	CHECK(status == SPARK_STATUS_OK, "submit request");
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
	CHECK(TestWaitLane(engine,2u,0u,&rebuilt) != 0u &&
		rebuilt.cache_prefix_token_count == 0u && rebuilt.cache_publish_token_count == 4u,
		"prefix reset: recovered session recomputes first block");
	CHECK(memcmp(&canonical.cache_publish_identity,&rebuilt.cache_publish_identity,
		sizeof(canonical.cache_publish_identity)) == 0,
		"prefix reset: rebuilt digest equals canonical prompt digest");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u &&
		state.token_events[2] == 1u,"prefix reset: recovered request completes exactly once");
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
	CHECK(state.error_events[2] == 1u && state.token_events[2] == 0u,
		"partial capacity: required hit fails rather than spinning or recomputing");
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

int main(void)
{
	SparkModelResidentDeployment deployment;
	char path[512];
	char runtime_root[256];

	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	(void)snprintf(path,sizeof(path),"%s/mock-batch-deployment.json",runtime_root);
	TestWriteDeployment(path,runtime_root);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == SPARK_STATUS_OK);
	deployment.eos_token_count = 1u;
	deployment.eos_token_ids[0] = 154820u;

	if ( getenv("ONLY_BUSY") != 0 )
	{
		TestScenarioRankBusyBackpressure(&deployment,runtime_root);
	}
	else
	{
		char *coordinator_root = deployment.nodes[0].runtime_root;
		deployment.nodes[0].runtime_root = "/remote-resident-only/runtime";
		TestScenarioHappyPath(&deployment,runtime_root);
		deployment.nodes[0].runtime_root = coordinator_root;
		TestScenarioEventDeadlines(&deployment,runtime_root);
		TestScenarioHappyPath(&deployment,runtime_root);
		TestScenarioRankDiesMidDecode(&deployment,runtime_root);
		TestScenarioRankKilledAndRevived(&deployment,runtime_root);
		TestScenarioCachedPrefixSessionReset(&deployment,runtime_root);
		TestScenarioPartialPrefixAppend(&deployment,runtime_root);
		TestScenarioChainPublishesFinalCheckpoint(&deployment,runtime_root);
		TestScenarioChainEosCheckpoint(&deployment,runtime_root);
		TestScenarioPartialCopyCapacity(&deployment,runtime_root);
		TestScenarioRankBusyBackpressure(&deployment,runtime_root);
		TestScenarioDriverIoError(&deployment,runtime_root);
		TestScenarioEosEarlyStop(&deployment,runtime_root);
		TestScenarioTwoRequestsRankDies(&deployment,runtime_root);
	}

	SparkModelResidentDeploymentReset(&deployment);
	MockResidentClientReset();
	(void)unlink(path);

	fprintf(stderr,"%s: %u checks, %u failures\n",
		"test_model_batch_engine_mock", test_checks, test_failures);
	return( test_failures != 0u ? 1 : 0 );
}
