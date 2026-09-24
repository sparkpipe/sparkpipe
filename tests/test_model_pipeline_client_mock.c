#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "mock_model_resident_client.h"
#include "sparkpipe/spark_model_pipeline_client.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

#define TEST_RANKS 3u
#define TEST_MAX_SEQ 4u
#define TEST_MAX_ROWS 8u

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

typedef struct TestCallbackState
{
	uint32_t result_count;
	uint32_t completion_count;
	uint32_t stage_completion_count;
	SparkStatus last_result_status;
	SparkStatus last_completion_status;
	uint64_t last_submission_id;
	uint32_t last_token_count;
} TestCallbackState;

static void TestSubmitResult(void *context, uint64_t submission_id, SparkStatus status)
{
	TestCallbackState *s = (TestCallbackState *)context;
	s->result_count++;
	s->last_result_status = status;
	s->last_submission_id = submission_id;
}

static void TestCompletion(void *context, const SparkModelServingCompletion *completion)
{
	TestCallbackState *s = (TestCallbackState *)context;
	s->completion_count++;
	s->last_completion_status = (SparkStatus)completion->status;
	s->last_token_count = completion->token_count;
}

static void TestStageCompletion(void *context, const SparkModelPipelineStageCompletion *stage_completion)
{
	TestCallbackState *s = (TestCallbackState *)context;
	(void)stage_completion;
	s->stage_completion_count++;
}

static const char *const TestTransportHosts[TEST_RANKS] =
{
	"mock-stage-a","mock-stage-b","mock-stage-c"
};

static void TestBuildDeployment(SparkModelResidentDeployment *deployment, const char *path, const char *runtime_root)
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
		endpoints[rank].tcp_port = (uint32_t)(59000u + rank);
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
	fixture.runtime_limits.max_active_sequence_count = 4u;
	fixture.runtime_limits.max_input_row_count = 8u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.control_port_base = 59000u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
	assert(SparkModelResidentDeploymentLoad(path,deployment) == SPARK_STATUS_OK);
}

static void TestBuildSubmission(SparkModelServingSubmission *submission, SparkModelServingLane *lanes, uint64_t submission_id)
{
	uint32_t slot_base = (uint32_t)((submission_id * 2u) % 28u);
	uint64_t sequence_id = 100u + submission_id;
	static uint32_t token_ids[2];
	static uint32_t row_lane_indices[2];
	static uint64_t row_positions[2];
	static uint64_t row_sequence_ids[2];
	memset(lanes,0,2u * sizeof(lanes[0]));
	lanes[0].request_id = 900u + submission_id;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = submission_id + 3000u;
	lanes[0].sequence_id = sequence_id;
	lanes[0].resident_sequence_slot = slot_base;
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lanes[1].request_id = 901u + submission_id;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = submission_id + 3000u;
	lanes[1].sequence_id = sequence_id + 1u;
	lanes[1].resident_sequence_slot = slot_base + 1u;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	token_ids[0] = 11u;
	token_ids[1] = 12u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_sequence_ids[0] = sequence_id;
	row_sequence_ids[1] = sequence_id + 1u;
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->tokens_per_sequence = 1u;
	submission->submission_id = submission_id;
	submission->request_id = 900u + submission_id;
	submission->sequence_id = sequence_id;
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = 1u;
	submission->request_generation = 1u;
	submission->step_generation = submission_id + 3000u;
	submission->residency.word0 = submission_id;
	submission->residency.word1 = submission_id + 100u;
	submission->residency.generation = submission_id + 200u;
	submission->residency.owner = 1u;
	submission->active_sequence_count = 2u;
	submission->new_token_count = 2u;
	submission->lane_count = 2u;
	submission->row_count = 2u;
	submission->token_count = 2u;
	submission->lanes = lanes;
	submission->token_ids = token_ids;
	submission->row_lane_indices = row_lane_indices;
	submission->row_positions = row_positions;
	submission->row_sequence_ids = row_sequence_ids;
}

static void TestFireAllRanksResult(uint64_t submission_id, SparkStatus status)
{
	uint32_t rank;
	for (rank=0u; rank<TEST_RANKS; rank++)
		MockResidentClientFireResult(rank, submission_id, status);
}

static void TestFireAllRanksDecision(uint64_t submission_id, uint32_t decision_kind)
{
	uint32_t rank;
	for (rank=0u; rank<TEST_RANKS; rank++)
		MockResidentClientFireDecision(rank, submission_id, decision_kind, SPARK_STATUS_OK);
}

static void TestFireAllRanksCompletion(uint64_t submission_id, SparkStatus status, const TestCallbackState *cb)
{
	SparkModelServingCompletion completion;
	uint32_t rank;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = (uint32_t)status;
	completion.submission_id = submission_id;
	completion.request_id = 900u + submission_id;
	completion.sequence_id = 100u + submission_id;
	completion.control_generation = 1u;
	completion.transaction_id = submission_id + 1000u;
	completion.dispatch_generation = 1u;
	completion.request_generation = 1u;
	completion.step_generation = submission_id + 3000u;
	completion.residency.word0 = submission_id;
	completion.residency.word1 = submission_id + 100u;
	completion.residency.generation = submission_id + 200u;
	completion.residency.owner = 1u;
	(void)cb;
	for (rank=0u; rank<TEST_RANKS; rank++)
	{
		if ( rank == TEST_RANKS - 1u )
		{
			completion.token_count = 2u;
			completion.tokens_per_sequence = 1u;
			completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		}
		MockResidentClientFireCompletion(rank, &completion);
	}
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	SparkModelPipelineClientConfiguration configuration;
	SparkModelPipelineClient *pipeline;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	TestCallbackState cb;
	SparkModelPipelineClientView view;
	SparkStatus status;
	uint64_t fingerprint_a, fingerprint_b;

	MockResidentClientReset();
	{
		char deploy_path[512];
		char runtime_root[256];
		assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
		(void)snprintf(deploy_path,sizeof(deploy_path),"%s/build/mock-pipeline-deployment.json",runtime_root);
		TestBuildDeployment(&deployment, deploy_path, runtime_root);
		memset(&configuration,0,sizeof(configuration));
		configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
		configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
		configuration.connect_timeout_ms = 1000u;
		configuration.deployment = &deployment;
		configuration.runtime_root = runtime_root;
		memset(&cb,0,sizeof(cb));
		configuration.submit_result_function = TestSubmitResult;
		configuration.submit_result_context = &cb;
		configuration.completion_function = TestCompletion;
		configuration.completion_context = &cb;
		configuration.stage_completion_function = TestStageCompletion;
		configuration.stage_completion_context = &cb;

		pipeline = 0;
		status = SparkModelPipelineClientConnect(&configuration, &pipeline);
		CHECK(status == SPARK_STATUS_OK, "connect");
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"connect failed: %u\n",(unsigned)status);
			return(1);
		}
	}

	TestBuildSubmission(&submission, lanes, 99u);
	submission.submission_id = 0u;
	CHECK( SparkModelPipelineClientSubmit(pipeline, &submission) == SPARK_STATUS_INVALID_ARGUMENT, "zero submission id rejected");

	TestBuildSubmission(&submission, lanes, 1u);
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	CHECK(status == SPARK_STATUS_OK, "submit 1");
	CHECK( MockResidentClientCalls(0, MOCK_CALL_PREPARE) == 1u, "rank 0 prepared");
	CHECK( MockResidentClientCalls(2, MOCK_CALL_PREPARE) == 1u, "rank 2 prepared");

	TestFireAllRanksResult(1u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	TestFireAllRanksDecision(1u, SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);

	fingerprint_a = SparkModelPipelineClientSessionFingerprint(pipeline);

	TestFireAllRanksCompletion(1u, SPARK_STATUS_OK, &cb);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	CHECK( cb.completion_count == 1u, "completion fired once");
	CHECK( cb.last_token_count == 2u, "two tokens for two lanes");

	TestBuildSubmission(&submission, lanes, 2u);
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	CHECK(status == SPARK_STATUS_OK, "submit 2");

	MockResidentClientFireResult(0u, 2u, SPARK_STATUS_OK);
	MockResidentClientFireResult(0u, 2u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status == SPARK_STATUS_OK, "duplicate result is not fatal");

	TestFireAllRanksResult(2u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	TestFireAllRanksDecision(2u, SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	TestFireAllRanksCompletion(2u, SPARK_STATUS_OK, &cb);
	MockResidentClientFireCompletion(1u, &(const SparkModelServingCompletion){
		.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
		.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES,
		.submission_id = 999u,
	});
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status == SPARK_STATUS_OK, "unknown-submission completion is not fatal");

	MockResidentClientFireResult(0u, 3u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status == SPARK_STATUS_OK, "late result after invalidation is not fatal");

	TestBuildSubmission(&submission, lanes, 3u);
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	MockResidentClientFireResult(0u, 3u, SPARK_STATUS_OK);
	MockResidentClientFireResult(1u, 3u, SPARK_STATUS_BUSY);
	MockResidentClientFireResult(2u, 3u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	MockResidentClientFireDecision(0u, 3u, SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT, SPARK_STATUS_OK);
	MockResidentClientFireDecision(2u, 3u, SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
	{
		CHECK( view.rejected_count != 0u || view.failed_status != SPARK_STATUS_OK,
			"a rank's real error surfaces (not swallowed)");
	}

	fingerprint_b = SparkModelPipelineClientSessionFingerprint(pipeline);
	CHECK( fingerprint_a == fingerprint_b, "fingerprint stable without disconnects");
	TestBuildSubmission(&submission,lanes,4u);
	CHECK(SparkModelPipelineClientSubmit(pipeline,&submission) == SPARK_STATUS_OK,
		"submit before transport loss");
	uint32_t completions_before = cb.completion_count;
	MockResidentClientKill(1u);
	CHECK(SparkModelPipelineClientSessionFingerprint(pipeline) == fingerprint_b,
		"disconnect does not publish a partial session");
	CHECK(SparkModelPipelineClientRecover(pipeline) == SPARK_STATUS_IO_ERROR,
		"recovery stays failed while a rank is unavailable");
	CHECK(cb.completion_count == completions_before + 1u &&
		cb.last_completion_status == SPARK_STATUS_IO_ERROR,
		"lost submission completes with failure exactly once");
	CHECK(SparkModelPipelineClientSessionFingerprint(pipeline) == fingerprint_b &&
		SparkModelPipelineClientAllRanksReady(pipeline) == 0u,
		"partial reconnection retains old fingerprint and closed admission");
	CHECK(SparkModelPipelineClientGetView(pipeline,&view) == SPARK_STATUS_OK &&
		view.active_transaction_count == 0u,
		"failed transaction accounting is released");
	(void)SparkModelPipelineClientRecover(pipeline);
	CHECK(cb.completion_count == completions_before + 1u,
		"retry does not duplicate completion");
	MockResidentClientRevive(1u);
	CHECK(SparkModelPipelineClientRecover(pipeline) == SPARK_STATUS_OK,
		"all ranks recover");
	CHECK(SparkModelPipelineClientSessionFingerprint(pipeline) != fingerprint_b &&
		SparkModelPipelineClientAllRanksReady(pipeline) != 0u,
		"complete new session publishes one fingerprint");

	SparkModelPipelineClientDestroy(pipeline);
	MockResidentClientReset();

	fprintf(stderr,"%s: %u checks, %u failures\n",
		"test_model_pipeline_client_mock", test_checks, test_failures);
	return( test_failures != 0u ? 1 : 0 );
}
