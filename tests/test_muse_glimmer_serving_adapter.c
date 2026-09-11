#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_muse_glimmer_model.h"
#include "sparkpipe/spark_muse_glimmer_resident_decode_stage_firmware.h"

#ifndef TEST_MUSE_GLIMMER_SERVING_ADAPTER_PATH
#define TEST_MUSE_GLIMMER_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MUSE_GLIMMER_SERVING_DRIVER_PATH
#define TEST_MUSE_GLIMMER_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_MUSE_GLIMMER_SERVING_CONFIG_PATH
#define TEST_MUSE_GLIMMER_SERVING_CONFIG_PATH ""
#endif
#ifndef TEST_MUSE_GLIMMER_MODEL_REVISION
#define TEST_MUSE_GLIMMER_MODEL_REVISION ""
#endif

#define TEST_MUSE_GLIMMER_MANDATORY_MEMBER_COUNT 10u

typedef struct TestMuseGlimmerServingState
{
	uint32_t completion_count;
	void *execution_stream;
	SparkModelServingCompletion completion;
} TestMuseGlimmerServingState;

static void TestMuseGlimmerServingCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestMuseGlimmerServingState *state;
	state = (TestMuseGlimmerServingState *)completion_context;
	assert(state != 0);
	assert(completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

static void TestMuseGlimmerServingInterfaceCompleteness(
	const SparkModelServingAdapterInterface *adapter_interface)
{
	SparkModelServingAdapterInterface probe;
	void **first,**copy_first;
	uint32_t index,member_count;
	assert(adapter_interface->abi_version == SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION);
	assert(adapter_interface->interface_bytes == SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES);
	assert(SparkModelServingAdapterValidateInterface(adapter_interface,0u) ==
		SPARK_STATUS_OK);
	assert(SparkModelServingAdapterValidateInterface(adapter_interface,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT) ==
		SPARK_STATUS_OK);
	first = (void **)&adapter_interface->initialize;
	member_count = (uint32_t)((void **)&adapter_interface->reset - first) + 1u;
	assert(member_count == TEST_MUSE_GLIMMER_MANDATORY_MEMBER_COUNT);
	for (index=0u; index<member_count; index++)
		assert(first[index] != 0);
	for (index=0u; index<member_count; index++)
	{
		probe = *adapter_interface;
		copy_first = (void **)&probe.initialize;
		copy_first[index] = 0;
		assert(SparkModelServingAdapterValidateInterface(&probe,0u) ==
			SPARK_STATUS_INVALID_ARGUMENT);
	}
}

static void TestMuseGlimmerServingConfiguration(
	SparkModelServingAdapterConfiguration *configuration,
	uint32_t stage_index,
	const char *config_path,
	const char *runtime_root,
	const char *driver_path,
	TestMuseGlimmerServingState *test_state)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = stage_index;
	configuration->stage_index = stage_index;
	configuration->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration->runtime_limits.max_inflight_submission_count = 1u;
	configuration->runtime_limits.max_active_sequence_count = 8u;
	configuration->runtime_limits.max_input_row_count = 8u;
	configuration->runtime_limits.resident_sequence_capacity = 8u;
	configuration->runtime_limits.kv_logical_page_capacity = 64u;
	configuration->runtime_limits.kv_physical_page_capacity = 64u;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "spark-test";
	configuration->node_target = "cuda.sm121.muse_glimmer.resident_decode_stage.bf16";
	configuration->adapter_configuration_path = config_path;
	configuration->driver_shared_object_path = driver_path;
	configuration->driver_program_name = "resident_decode";
	configuration->execution_stream = test_state->execution_stream;
	configuration->completion_function = TestMuseGlimmerServingCompletion;
	configuration->completion_context = test_state;
}

static void TestMuseGlimmerServingDecodeSubmission(
	SparkModelServingSubmission *submission,
	SparkModelServingLane *lanes,
	const uint32_t *token_ids,
	const uint32_t *row_lane_indices,
	const uint64_t *row_positions,
	const uint64_t *row_sequence_ids,
	uint64_t control_generation)
{
	memset(submission,0,sizeof(*submission));
	memset(lanes,0,sizeof(SparkModelServingLane) * 2u);
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
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->tokens_per_sequence = 1u;
	submission->submission_id = 77u;
	submission->request_id = 9u;
	submission->sequence_id = 100u;
	submission->control_generation = control_generation;
	submission->transaction_id = 1077u;
	submission->dispatch_generation = 2077u;
	submission->request_generation = 1u;
	submission->step_generation = 3077u;
	submission->residency.word0 = 77u;
	submission->residency.word1 = 177u;
	submission->residency.generation = 277u;
	submission->residency.owner = 13u;
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

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingAdapterSnapshot snapshot;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	TestMuseGlimmerServingState test_state;
	void *adapter_state;
	uint32_t token_ids[2],row_lane_indices[2];
	uint64_t row_positions[2],row_sequence_ids[2];
	char runtime_root[4096];
	memset(&test_state,0,sizeof(test_state));
	assert(cudaStreamCreate((cudaStream_t *)&test_state.execution_stream) == cudaSuccess);
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(
		TEST_MUSE_GLIMMER_SERVING_ADAPTER_PATH,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT,
		&library) == SPARK_STATUS_OK);
	assert(strcmp(library.adapter_interface.descriptor->adapter_id,
		"spark.muse.serving-adapter.tp16.v1") == 0);
	assert(strcmp(library.adapter_interface.descriptor->model_id,
		"meta-models/Muse-Glimmer-30B") == 0);
	assert(strcmp(library.adapter_interface.descriptor->model_revision,
		TEST_MUSE_GLIMMER_MODEL_REVISION) == 0);
	assert(library.adapter_interface.descriptor->capability_flags ==
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT);
	assert(library.adapter_interface.descriptor->max_speculative_token_count == 0u);
	assert(library.adapter_interface.descriptor->cache_block_token_count ==
		SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	assert(library.adapter_interface.descriptor->stage_layer_counts[0] ==
		SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT);
	assert(library.adapter_interface.descriptor->layer_count ==
		SPARK_MUSE_GLIMMER_MODEL_LAYER_COUNT);
	TestMuseGlimmerServingInterfaceCompleteness(&library.adapter_interface);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	TestMuseGlimmerServingConfiguration(&configuration,0u,
		TEST_MUSE_GLIMMER_SERVING_CONFIG_PATH,runtime_root,
		TEST_MUSE_GLIMMER_SERVING_DRIVER_PATH,&test_state);
	adapter_state = 0;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) ==
		SPARK_STATUS_OK);
	assert(adapter_state != 0);
	token_ids[0] = 11u;
	token_ids[1] = 12u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 101u;
	TestMuseGlimmerServingDecodeSubmission(&submission,lanes,token_ids,
		row_lane_indices,row_positions,row_sequence_ids,1u);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(library.adapter_interface.prefetch(0,&submission,1u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.prefetch(adapter_state,&submission,0u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.prefetch(adapter_state,&submission,1u) ==
		SPARK_STATUS_OK);
	assert(library.adapter_interface.resolve_prefetch(adapter_state,&submission,0u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.resolve_prefetch(adapter_state,&submission,
		SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
	assert(library.adapter_interface.prefetch(adapter_state,&submission,1u) ==
		SPARK_STATUS_OK);
	assert(library.adapter_interface.resolve_prefetch(adapter_state,&submission,
		SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) == SPARK_STATUS_OK);
	assert(library.adapter_interface.submit(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(test_state.completion_count == 1u);
	assert(test_state.completion.submission_id == 77u);
	assert(test_state.completion.completion_flags ==
		SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS);
	assert(test_state.completion.token_count == 2u);
	assert(test_state.completion.token_ids[0] == 4200u);
	assert(test_state.completion.token_ids[1] == 4201u);
	assert(library.adapter_interface.snapshot(adapter_state,&snapshot) ==
		SPARK_STATUS_OK);
	assert(snapshot.submitted_count == 1u);
	assert(snapshot.completed_count == 1u);
	assert(library.adapter_interface.reset(adapter_state,0u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.reset(adapter_state,1u) == SPARK_STATUS_OK);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	submission.control_generation = 0u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_VALIDATION_FAILED);
	assert(library.adapter_interface.prefetch(adapter_state,&submission,1u) ==
		SPARK_STATUS_VALIDATION_FAILED);
	submission.control_generation = 1u;
	assert(library.adapter_interface.reset(adapter_state,1u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.reset(adapter_state,2u) == SPARK_STATUS_OK);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_VALIDATION_FAILED);
	submission.control_generation = 2u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) ==
		SPARK_STATUS_OK);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_BUSY);
	assert(library.adapter_interface.reset(adapter_state,2u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.reset(adapter_state,3u) == SPARK_STATUS_OK);
	submission.control_generation = 2u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_VALIDATION_FAILED);
	submission.control_generation = 3u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(library.adapter_interface.submit(adapter_state,&submission) ==
		SPARK_STATUS_OK);
	assert(test_state.completion_count == 2u);
	library.adapter_interface.destroy(adapter_state);
	SparkModelServingAdapterUnloadInterface(&library);
	assert(cudaStreamDestroy((cudaStream_t)test_state.execution_stream) == cudaSuccess);
	return(0);
}
