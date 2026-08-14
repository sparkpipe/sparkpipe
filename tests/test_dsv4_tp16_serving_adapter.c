#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_DSV4_TP16_ADAPTER_PATH
#define TEST_DSV4_TP16_ADAPTER_PATH ""
#endif
#ifndef TEST_DSV4_TP16_DRIVER_PATH
#define TEST_DSV4_TP16_DRIVER_PATH ""
#endif
#ifndef TEST_DSV4_TP16_CONFIG_PATH
#define TEST_DSV4_TP16_CONFIG_PATH ""
#endif

typedef struct TestDsv4Tp16State
{
	uint32_t completion_count;
	SparkModelServingCompletion completion;
} TestDsv4Tp16State;

static void TestDsv4Tp16Completion(void *context,
	const SparkModelServingCompletion *completion)
{
	TestDsv4Tp16State *state;
	state = (TestDsv4Tp16State *)context;
	assert(state != 0 && completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingAdapterSnapshot snapshot;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	TestDsv4Tp16State test_state;
	uint32_t row_lane,step,token_id;
	uint64_t row_position,row_sequence;
	void *adapter_state;
	char runtime_root[4096];
	SparkStatus status;
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(
		TEST_DSV4_TP16_ADAPTER_PATH,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV,
		&library) == SPARK_STATUS_OK);
	assert(library.adapter_interface.descriptor->stage_count == 16u);
	assert(library.adapter_interface.descriptor->max_inflight_submission_count == 16u);
	assert(library.adapter_interface.descriptor->minimum_efficient_submission_row_count == 1u);
	assert((library.adapter_interface.descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT) == 0u);
	assert((library.adapter_interface.descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN) != 0u);
	assert(library.adapter_interface.descriptor->max_output_token_count ==
		SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT);
	memset(&test_state,0,sizeof(test_state));
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration.rank_index = 15u;
	configuration.stage_index = 15u;
	configuration.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration.runtime_limits.max_inflight_submission_count = 1u;
	configuration.runtime_limits.max_active_sequence_count = 1u;
	configuration.runtime_limits.max_input_row_count = 1u;
	configuration.runtime_limits.resident_sequence_capacity = 128u;
	configuration.runtime_limits.kv_logical_page_capacity = 128u;
	configuration.runtime_limits.kv_physical_page_capacity = 128u;
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	configuration.runtime_root = runtime_root;
	configuration.node_id = "spark15";
	configuration.node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
	configuration.adapter_configuration_path = TEST_DSV4_TP16_CONFIG_PATH;
	configuration.driver_shared_object_path = TEST_DSV4_TP16_DRIVER_PATH;
	configuration.driver_program_name = "resident_decode";
	configuration.execution_stream = (void *)(uintptr_t)1u;
	configuration.completion_function = TestDsv4Tp16Completion;
	configuration.completion_context = &test_state;
	adapter_state = 0;
	status = library.adapter_interface.initialize(&configuration,&adapter_state);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"TP16 adapter initialize status=%d (%s)\n",(int)status,SparkStatusToString(status));
	assert(status == SPARK_STATUS_OK);
	assert(adapter_state != 0);
	assert(library.adapter_interface.snapshot(adapter_state,&snapshot) ==
		SPARK_STATUS_OK);
	assert(snapshot.kv_token_capacity == 130u);
	memset(&lane,0,sizeof(lane));
	lane.request_id = 900u;
	lane.request_generation = 1u;
	lane.step_generation = 4000u;
	lane.sequence_id = 100u;
	lane.resident_sequence_slot = 127u;
	lane.context_token_count = 1u;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	token_id = 11u;
	row_lane = 0u;
	row_position = 0u;
	row_sequence = 100u;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 1000u;
	submission.request_id = lane.request_id;
	submission.sequence_id = lane.sequence_id;
	submission.control_generation = 1u;
	submission.transaction_id = 2000u;
	submission.dispatch_generation = 3000u;
	submission.request_generation = 1u;
	submission.step_generation = lane.step_generation;
	submission.active_sequence_count = 1u;
	submission.new_token_count = 1u;
	submission.lane_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	submission.tokens_per_sequence = 8u;
	submission.lanes = &lane;
	submission.token_ids = &token_id;
	submission.row_lane_indices = &row_lane;
	submission.row_positions = &row_position;
	submission.row_sequence_ids = &row_sequence;
	assert(library.adapter_interface.validate_submission(adapter_state,
		&submission) == SPARK_STATUS_OK);
	assert(SparkModelServingAdapterPrepareSubmission(
		&library.adapter_interface,adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(SparkModelServingAdapterResolvePrefetch(
		&library.adapter_interface,adapter_state,&submission,
		SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
	assert(library.adapter_interface.submit(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(test_state.completion_count == 1u);
	assert(test_state.completion.tokens_per_sequence == 8u);
	assert(test_state.completion.token_count == 8u);
	for (step=0u; step<8u; step++)
		assert(test_state.completion.token_ids[step] == 4200u + step);
	library.adapter_interface.destroy(adapter_state);
	SparkModelServingAdapterUnloadInterface(&library);
	return(0);
}
