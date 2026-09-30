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
	uint32_t stale_recomputes[TEST_MAX_REQUESTS + 1u];
	uint64_t accepted_ns[TEST_MAX_REQUESTS + 1u];
	uint64_t first_dispatch_ns[TEST_MAX_REQUESTS + 1u];
	uint64_t first_token_ns[TEST_MAX_REQUESTS + 1u];
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
		if ( s->token_events[id]++ == 0u )
			s->first_token_ns[id] = event->monotonic_ns;
		s->cached_tokens[id] = event->cached_prompt_token_count;
		s->first_dispatch_ns[id] = event->first_dispatch_ns;
	}
	s->stale_recomputes[id] = event->stale_prefix_recompute_count;
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
	fixture.runtime_limits.max_input_row_count = 32u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.control_port_base = 59200u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static SparkModelBatchEngine *TestConnectRows(const SparkModelResidentDeployment *deployment,TestBatchState *state,const char *runtime_root,uint32_t prefill_rows)
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
	configuration.max_prefill_rows_per_submission = prefill_rows;
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

static uint32_t TestLaneLogCount(uint64_t request_id,uint32_t flag)
{
	SparkModelServingLane lane;
	uint32_t index,count;
	count = 0u;
	for (index=0u; MockResidentClientLaneLog(index,&lane) != 0u; index++)
		if ( (request_id == 0u || lane.request_id == request_id) && (lane.flags & flag) != 0u )
			count++;
	return(count);
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

static void TestScenarioAdapterWithoutPrefixReuse(const SparkModelResidentDeployment *deployment,const char *runtime_root,const char *expected_mode)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0};
	uint32_t prompt[9] = {11u,12u,13u,14u,15u,16u,17u,18u,19u};
	char log[4096];
	MockResidentClientReset();
	engine = TestConnectCapturingLog(deployment,&state,runtime_root,log,sizeof(log));
	if ( engine == 0 )
		return;
	CHECK(strstr(log,expected_mode) != 0,"no prefix reuse: startup names the adapter's cache mode");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,700u,1u,prompt,8u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && TestLaneLogCount(1u,SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH) != 0u,"no prefix reuse: the first prompt completes and its checkpoints are still named");
	TestSubmitPrompt(engine,2u,701u,1u,prompt,9u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.sequence_position == 0u && lane.cache_prefix_token_count == 0u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) == 0u,"no prefix reuse: a prompt sharing a published prefix prefills from position zero");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u && state.cached_tokens[2] == 0u,"no prefix reuse: the second prompt reports no cached tokens");
	CHECK(TestLaneLogCount(2u,SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u && TestLaneLogCount(0u,SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) == 0u,"no prefix reuse: the adapter never receives a cache prefix lane");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioSpeculativeCompletionSkipsDecodeCheckpoint(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane lane = {0},requested = {0};
	uint32_t prompt[4] = {11u,12u,13u,14u};
	uint32_t extended[10] = {11u,12u,13u,14u,1000u,1000u,1001u,1002u,1000u,31u};
	char log[4096];
	MockResidentClientReset();
	engine = TestConnectCapturingLog(deployment,&state,runtime_root,log,sizeof(log));
	if ( engine == 0 )
		return;
	CHECK(strstr(log,"adapter=test.model.serving.adapter.speculative-inline.v1 prefix_reuse=on decode_checkpoints=inline-until-speculative") != 0,"speculative inline: startup names the adapter's cache mode");
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(1000u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmitPrompt(engine,1u,710u,8u,prompt,4u);
	CHECK(TestWaitLane(engine,1u,4u,&lane) != 0u,"speculative inline: prefill emits before decode");
	MockResidentClientSetAutoTokens(3u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u && state.error_events[1] == 0u && state.token_events[1] == 8u,"speculative inline: a speculative completion across the block boundary is not UNSUPPORTED");
	CHECK(TestLaneLogFind(1u,7u,SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN | SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH,&requested) != 0u && requested.cache_publish_token_count == 8u,"speculative inline: the boundary decode lane names its checkpoint before the completion is known");
	CHECK(TestLaneLogFind(1u,10u,SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN,&lane) != 0u,"speculative inline: after a speculative completion the request names no further decode checkpoints");
	MockResidentClientSetAutoTokens(1u);
	TestSubmitPrompt(engine,2u,711u,1u,extended,10u);
	CHECK(TestWaitFirstRequestLane(engine,2u,&lane) != 0u && lane.cache_prefix_token_count == 4u && lane.sequence_position == 4u && (lane.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u,"speculative inline: the prompt checkpoint is reused and the speculative decode block is not indexed");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.cached_tokens[2] == 4u,"speculative inline: the extending request reports only the prompt checkpoint");
	SparkModelBatchEngineDestroy(engine);
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
	CHECK(strstr(log,"adapter=test.model.serving.adapter.speculative-deferred.v1 prefix_reuse=on decode_checkpoints=deferred") != 0,"speculative deferred: startup names the adapter's cache mode");
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

static uint32_t TestCoordinatorFrames(void)
{
	return(MockResidentClientCalls(0u,MOCK_CALL_SUBMIT) + MockResidentClientCalls(0u,MOCK_CALL_PREPARE) + MockResidentClientCalls(0u,MOCK_CALL_CONTINUE));
}

static void TestScenarioSingleTokenEosSkipsPublish(const SparkModelResidentDeployment *deployment,const char *runtime_root)
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
	CHECK(state.completed_events[1] == 1u && state.token_events[1] == 2u && TestCoordinatorFrames() == frames + 1u,"single EOS: a one-token completion that stops on EOS releases without a publish frame");
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
	request.temperature = 0.7f;
	CHECK(SparkModelBatchEngineSubmit(engine,&request,&handle) == SPARK_STATUS_OK && handle != 0u,"sampling: a seeded request is admitted by a sampling adapter");
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
	CHECK(TestFindLane(2u,0u,0u,&rebuilt) != 0u && (rebuilt.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) == 0u && rebuilt.cache_publish_token_count == 4u && memcmp(&rebuilt.cache_publish_identity,&canonical.cache_publish_identity,sizeof(canonical.cache_publish_identity)) == 0,"stale: the recompute starts at position zero and republishes the canonical block");
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

static void TestScenarioMultiBlockPrefill(const SparkModelResidentDeployment *deployment,const char *runtime_root,uint32_t multi_block)
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
	if ( multi_block != 0u )
	{
		CHECK(TestPrefillChunksAreCanonical(1u,4u,18u) == 3u && TestLongestPrefillChunk(1u,18u) == 8u,"multi-block prefill: a lane spans whole cache blocks up to the row budget and still ends on a block boundary or the prompt end");
		CHECK(TestPrefillBlockIdentities(1u,prompt,4u,18u) == 2u,"multi-block prefill: each block boundary inside a span carries the identity of the prompt prefix it closes");
	}
	else
	{
		CHECK(TestPrefillChunksAreCanonical(1u,4u,18u) == 5u && TestLongestPrefillChunk(1u,18u) == 4u,"multi-block prefill: an adapter without the capability keeps one cache block per prefill lane");
		CHECK(TestPrefillBlockIdentities(1u,prompt,4u,18u) == 0u,"multi-block prefill: one-block lanes carry no block identities");
	}
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioAdapterCacheModes(const char *runtime_root)
{
	SparkModelResidentDeployment deployment;
	char path[512];
	TestLoadVariantDeployment(runtime_root,"without-prefix-reuse",TEST_MODEL_SERVING_WITHOUT_PREFIX_REUSE_PATH,path,sizeof(path),&deployment);
	TestScenarioAdapterWithoutPrefixReuse(&deployment,runtime_root,"adapter=test.model.serving.adapter.without-prefix-reuse.v1 prefix_reuse=off decode_checkpoints=inline");
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"prefix-reuse-deployment-off",TEST_MODEL_SERVING_ADAPTER_PATH,path,sizeof(path),&deployment);
	deployment.prefix_reuse_disabled = 1u;
	TestScenarioAdapterWithoutPrefixReuse(&deployment,runtime_root,"adapter=test.model.serving.adapter.v1 prefix_reuse=deployment-off decode_checkpoints=inline");
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"speculative-inline",TEST_MODEL_SERVING_SPECULATIVE_INLINE_PATH,path,sizeof(path),&deployment);
	TestScenarioSpeculativeCompletionSkipsDecodeCheckpoint(&deployment,runtime_root);
	TestScenarioSpeculationOffPublishesDecodeCheckpoint(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"speculative-deferred",TEST_MODEL_SERVING_SPECULATIVE_DEFERRED_PATH,path,sizeof(path),&deployment);
	TestScenarioSpeculativePublishAdapterDefers(&deployment,runtime_root);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"multi-block-prefill",TEST_MODEL_SERVING_MULTI_BLOCK_PREFILL_PATH,path,sizeof(path),&deployment);
	TestScenarioMultiBlockPrefill(&deployment,runtime_root,1u);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
	TestLoadVariantDeployment(runtime_root,"one-block-prefill",TEST_MODEL_SERVING_ADAPTER_PATH,path,sizeof(path),&deployment);
	TestScenarioMultiBlockPrefill(&deployment,runtime_root,0u);
	SparkModelResidentDeploymentReset(&deployment);
	(void)unlink(path);
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
	TestScenarioPartialPrefixAppend(&deployment,runtime_root);
	TestScenarioChainPublishesFinalCheckpoint(&deployment,runtime_root);
	TestScenarioChainPublishesAtBlockBoundary(&deployment,runtime_root);
	TestScenarioSingleTokenEosSkipsPublish(&deployment,runtime_root);
	TestScenarioChainEosCheckpoint(&deployment,runtime_root);
	TestScenarioGeneratedCheckpointIdentity(&deployment,runtime_root,1u);
	TestScenarioGeneratedCheckpointIdentity(&deployment,runtime_root,3u);
	TestScenarioCanonicalPrefillChunks(&deployment,runtime_root);
	TestScenarioPartialCopyCapacity(&deployment,runtime_root);
	TestScenarioSamplingValidation(&deployment,runtime_root);
	TestScenarioMeasurements(&deployment,runtime_root);
	TestScenarioStalePrefixRecomputes(&deployment,runtime_root);
	TestScenarioStalePrefixTerminates(&deployment,runtime_root);
	TestScenarioStalePrefixIsolatesLanes(&deployment,runtime_root);
	TestScenarioVerificationFailureIsFatal(&deployment,runtime_root);
	TestScenarioRankBusyBackpressure(&deployment,runtime_root);
	TestScenarioDriverIoError(&deployment,runtime_root);
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
