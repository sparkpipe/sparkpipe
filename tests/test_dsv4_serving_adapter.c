#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_DSV4_SERVING_ADAPTER_PATH
#define TEST_DSV4_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_DRIVER_PATH
#define TEST_DSV4_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_CONFIG_PATH
#define TEST_DSV4_SERVING_CONFIG_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_STALE_CONFIG_PATH
#define TEST_DSV4_SERVING_STALE_CONFIG_PATH ""
#endif
#ifndef TEST_DSV4_SERVING_ABSOLUTE_CONFIG_PATH
#define TEST_DSV4_SERVING_ABSOLUTE_CONFIG_PATH ""
#endif

typedef struct TestDsv4ServingState
{
	uint32_t completion_count;
	SparkModelServingCompletion completion;
} TestDsv4ServingState;

static void TestDsv4ServingCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestDsv4ServingState *state;
	state = (TestDsv4ServingState *)completion_context;
	assert(state != 0);
	assert(completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingAdapterSnapshot snapshot;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	TestDsv4ServingState test_state;
	SparkStatus initialize_status;
	void *adapter_state;
	void *hidden_input;
	uint32_t token_ids[4],row_lane_indices[4];
	uint64_t row_positions[4],row_sequence_ids[4];
	uint64_t hidden_input_bytes;
	char runtime_root[4096];
	memset(&test_state,0,sizeof(test_state));
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_DSV4_SERVING_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT,&library) == SPARK_STATUS_OK);
	assert(strcmp(library.adapter_interface.descriptor->model_id,"deepseek-ai/DeepSeek-V4-Flash-0731") == 0);
	assert(library.adapter_interface.descriptor->max_speculative_token_count == 0u);
	assert(library.adapter_interface.descriptor->max_inflight_submission_count == 13u);
	assert(library.adapter_interface.descriptor->stage_count == 13u);
	assert(library.adapter_interface.descriptor->minimum_efficient_submission_row_count == 16u);
	hidden_input_bytes = 4u * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	hidden_input = calloc(1u,(size_t)hidden_input_bytes);
	assert(hidden_input != 0);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration.rank_index = 12u;
	configuration.stage_index = 12u;
	configuration.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration.runtime_limits.max_inflight_submission_count = 2u;
	configuration.runtime_limits.max_active_sequence_count = 2u;
	configuration.runtime_limits.max_input_row_count = 4u;
	configuration.runtime_limits.resident_sequence_capacity = 8u;
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	configuration.runtime_root = runtime_root;
	configuration.node_id = "spark12";
	configuration.node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
	configuration.adapter_configuration_path = TEST_DSV4_SERVING_CONFIG_PATH;
	configuration.driver_shared_object_path = TEST_DSV4_SERVING_DRIVER_PATH;
	configuration.driver_program_name = "resident_decode";
	configuration.execution_stream = (void *)(uintptr_t)1u;
	configuration.completion_function = TestDsv4ServingCompletion;
	configuration.completion_context = &test_state;
	adapter_state = 0;
	initialize_status = library.adapter_interface.initialize(&configuration,&adapter_state);
	if ( initialize_status != SPARK_STATUS_OK )
		fprintf(stderr,"DSV4 adapter initialize status=%d\n",(int)initialize_status);
	assert(initialize_status == SPARK_STATUS_OK);
	memset(lanes,0,sizeof(lanes));
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
	token_ids[0] = 11u;
	token_ids[1] = 12u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 101u;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 77u;
	submission.request_id = 9u;
	submission.sequence_id = 100u;
	submission.control_generation = 1u;
	submission.transaction_id = 1077u;
	submission.dispatch_generation = 2077u;
	submission.request_generation = 1u;
	submission.step_generation = 3077u;
	submission.residency.word0 = 77u;
	submission.residency.word1 = 177u;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 2u;
	submission.new_token_count = 2u;
	submission.lane_count = 2u;
	submission.row_count = 2u;
	submission.token_count = 2u;
	submission.lanes = lanes;
	submission.token_ids = token_ids;
	submission.row_lane_indices = row_lane_indices;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequence_ids;
	submission.hidden_input_address = hidden_input;
	submission.hidden_input_bytes = hidden_input_bytes;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 1u);
	assert(test_state.completion.submission_id == 77u);
	assert(test_state.completion.dispatch_generation == 2077u);
	assert(test_state.completion.transaction_id == 1077u);
	assert(memcmp(&test_state.completion.residency,&submission.residency,
		sizeof(submission.residency)) == 0);
	assert(test_state.completion.completion_flags == SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS);
	assert(test_state.completion.token_count == 2u);
	assert(test_state.completion.token_ids[0] == 4200u);
	assert(test_state.completion.token_ids[1] == 4201u);
	row_positions[0] = 4096u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	row_positions[0] = 0u;
	token_ids[0] = 21u;
	token_ids[1] = 22u;
	token_ids[2] = 23u;
	token_ids[3] = 24u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_lane_indices[2] = 0u;
	row_lane_indices[3] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_positions[2] = 1u;
	row_positions[3] = 1u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 101u;
	row_sequence_ids[2] = 100u;
	row_sequence_ids[3] = 101u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	lanes[0].context_token_count = 2u;
	lanes[1].context_token_count = 2u;
	submission.submission_id = 78u;
	submission.transaction_id = 1078u;
	submission.dispatch_generation = 2078u;
	submission.step_generation = 3078u;
	submission.new_token_count = 4u;
	submission.row_count = 4u;
	submission.token_count = 4u;
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 2u);
	assert(test_state.completion.submission_id == 78u);
	assert(test_state.completion.token_count == 2u);
	assert(test_state.completion.token_ids[0] == 4202u);
	assert(test_state.completion.token_ids[1] == 4203u);
	lanes[1].flags = 0u;
	submission.submission_id = 79u;
	submission.transaction_id = 1079u;
	submission.dispatch_generation = 2079u;
	submission.step_generation = 3079u;
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 3u);
	assert(test_state.completion.token_ids[0] == 4202u);
	assert(test_state.completion.token_ids[1] == 0u);
	lanes[0].flags = 0u;
	submission.submission_id = 80u;
	submission.transaction_id = 1080u;
	submission.dispatch_generation = 2080u;
	submission.step_generation = 3080u;
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 4u);
	assert(test_state.completion.token_ids[0] == 0u);
	assert(test_state.completion.token_ids[1] == 0u);
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 0u;
	row_lane_indices[2] = 1u;
	row_lane_indices[3] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 1u;
	row_positions[2] = 0u;
	row_positions[3] = 1u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 100u;
	row_sequence_ids[2] = 101u;
	row_sequence_ids[3] = 101u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_lane_indices[2] = 0u;
	row_lane_indices[3] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_positions[2] = 1u;
	row_positions[3] = 1u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 101u;
	row_sequence_ids[2] = 100u;
	row_sequence_ids[3] = 101u;
	submission.submission_id = 81u;
	submission.transaction_id = 1081u;
	submission.dispatch_generation = 2081u;
	submission.step_generation = 3081u;
	row_positions[0] = 1u;
	row_positions[2] = 0u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(test_state.completion_count == 4u);
	assert(library.adapter_interface.snapshot(adapter_state,&snapshot) == SPARK_STATUS_OK);
	assert(snapshot.submitted_count == 4u);
	assert(snapshot.completed_count == 4u);
	assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
	assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_BUSY);
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_BUSY);
	assert(library.adapter_interface.snapshot(adapter_state,&snapshot) == SPARK_STATUS_OK);
	assert(snapshot.active_submission_count == 0u);
	assert(snapshot.available_submission_count == 0u);
	library.adapter_interface.destroy(adapter_state);
	configuration.adapter_configuration_path = TEST_DSV4_SERVING_STALE_CONFIG_PATH;
	adapter_state = 0;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_SCHEMA_ERROR);
	assert(adapter_state == 0);
	configuration.adapter_configuration_path = TEST_DSV4_SERVING_ABSOLUTE_CONFIG_PATH;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(adapter_state == 0);
	SparkModelServingAdapterUnloadInterface(&library);
	free(hidden_input);
	return(0);
}
