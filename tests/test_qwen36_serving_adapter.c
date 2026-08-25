#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_27b_model.h"
#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"

#ifndef TEST_QWEN38_27B_SERVING_ADAPTER_PATH
#define TEST_QWEN38_27B_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_QWEN38_27B_SERVING_DRIVER_PATH
#define TEST_QWEN38_27B_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_QWEN38_27B_SERVING_CONFIG_PATH
#define TEST_QWEN38_27B_SERVING_CONFIG_PATH ""
#endif
#ifndef TEST_QWEN38_27B_SERVING_STALE_CONFIG_PATH
#define TEST_QWEN38_27B_SERVING_STALE_CONFIG_PATH ""
#endif
#ifndef TEST_QWEN38_27B_SERVING_ABSOLUTE_CONFIG_PATH
#define TEST_QWEN38_27B_SERVING_ABSOLUTE_CONFIG_PATH ""
#endif
#ifndef TEST_QWEN38_27B_SERVING_OVERRUN_CONFIG_PATH
#define TEST_QWEN38_27B_SERVING_OVERRUN_CONFIG_PATH ""
#endif

typedef struct TestQwen38_27bServingState
{
	uint32_t completion_count;
	/* A real stream: the adapter's transport shim issues stream-ordered
	 * copies through the configuration's execution stream, and a CUDA host
	 * dereferences the handle (the cuda stub ignores it, which is why a
	 * forged handle only ever "worked" off-device). */
	void *execution_stream;
	SparkModelServingCompletion completion;
} TestQwen38_27bServingState;

static void TestQwen38_27bServingCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	TestQwen38_27bServingState *state;
	state = (TestQwen38_27bServingState *)completion_context;
	assert(state != 0);
	assert(completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

static void TestQwen38_27bServingConfiguration(
	SparkModelServingAdapterConfiguration *configuration,
	uint32_t stage_index,
	const char *config_path,
	const char *runtime_root,
	const char *driver_path,
	TestQwen38_27bServingState *test_state)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = stage_index;
	configuration->stage_index = stage_index;
	configuration->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration->runtime_limits.max_inflight_submission_count = 2u;
	configuration->runtime_limits.max_active_sequence_count = 8u;
	configuration->runtime_limits.max_input_row_count = 8u;
	configuration->runtime_limits.resident_sequence_capacity = 8u;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "spark-test";
	configuration->node_target = "cuda.sm121.qwen38_27b.resident_decode_stage.bf16";
	configuration->adapter_configuration_path = config_path;
	configuration->driver_shared_object_path = driver_path;
	configuration->driver_program_name = "resident_decode";
	configuration->execution_stream = test_state->execution_stream;
	configuration->completion_function = TestQwen38_27bServingCompletion;
	configuration->completion_context = test_state;
}

static void TestQwen38_27bServingLanes(SparkModelServingLane *lanes)
{
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
}

static void TestQwen38_27bServingDecodeSubmission(
	SparkModelServingSubmission *submission,
	const SparkModelServingLane *lanes,
	const uint32_t *token_ids,
	const uint32_t *row_lane_indices,
	const uint64_t *row_positions,
	const uint64_t *row_sequence_ids,
	const void *hidden_input,
	uint64_t hidden_bytes)
{
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->tokens_per_sequence = 1u;
	submission->submission_id = 77u;
	submission->request_id = 9u;
	submission->sequence_id = 100u;
	submission->control_generation = 1u;
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
	submission->hidden_input_address = hidden_input;
	submission->hidden_input_bytes = hidden_input != 0 ? hidden_bytes : 0u;
}

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingAdapterSnapshot snapshot;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	TestQwen38_27bServingState test_state;
	void *adapter_state;
	uint32_t token_ids[4],row_lane_indices[4];
	uint64_t row_positions[4],row_sequence_ids[4];
	char runtime_root[4096];
	memset(&test_state,0,sizeof(test_state));
	assert(cudaStreamCreate((cudaStream_t *)&test_state.execution_stream) == cudaSuccess);
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_QWEN38_27B_SERVING_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&library) == SPARK_STATUS_OK);
	/* TP4 descriptor: four whole-stack ranks, no hidden transport, MTP armed. */
	assert(strcmp(library.adapter_interface.descriptor->adapter_id,"spark.qwen38_27b.serving-adapter.tp4.v1") == 0);
	assert(strcmp(library.adapter_interface.descriptor->model_id,"Qwen/Qwen3.8-27B") == 0);
	assert(library.adapter_interface.descriptor->stage_count == 4u);
	assert(library.adapter_interface.descriptor->layer_count == SPARK_QWEN38_27B_MODEL_LAYER_COUNT);
	assert(library.adapter_interface.descriptor->max_inflight_submission_count == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
	assert(library.adapter_interface.descriptor->max_active_sequence_count == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT);
	assert(library.adapter_interface.descriptor->max_speculative_token_count == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS);
	assert(library.adapter_interface.descriptor->expert_weight_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(library.adapter_interface.descriptor->stage_layer_counts[0] == SPARK_QWEN38_27B_MODEL_LAYER_COUNT);
	assert(library.adapter_interface.descriptor->stage_layer_counts[3] == SPARK_QWEN38_27B_MODEL_LAYER_COUNT);
	assert(library.adapter_interface.descriptor->stage_layer_counts[4] == 0u);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);

	/* Head stage (every TP4 rank owns the embedding and the head): decode,
	 * prefill, emit gating, validation refusals. */
	TestQwen38_27bServingConfiguration(&configuration,3u,TEST_QWEN38_27B_SERVING_CONFIG_PATH,runtime_root,TEST_QWEN38_27B_SERVING_DRIVER_PATH,&test_state);
	adapter_state = 0;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
	assert(adapter_state != 0);
	TestQwen38_27bServingLanes(lanes);
	token_ids[0] = 11u;
	token_ids[1] = 12u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 101u;
	TestQwen38_27bServingDecodeSubmission(&submission,lanes,token_ids,row_lane_indices,row_positions,row_sequence_ids,0,0);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 1u);
	assert(test_state.completion.submission_id == 77u);
	assert(test_state.completion.dispatch_generation == 2077u);
	assert(test_state.completion.transaction_id == 1077u);
	assert(memcmp(&test_state.completion.residency,&submission.residency,sizeof(submission.residency)) == 0);
	assert(test_state.completion.completion_flags == SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS);
	assert(test_state.completion.token_count == 2u);
	assert(test_state.completion.token_ids[0] == 4200u);
	assert(test_state.completion.token_ids[1] == 4201u);
	assert(test_state.completion.accepted_token_count == 2u);
	/* The positions cap is the adapter configuration's 4096, not the module's. */
	row_positions[0] = 4096u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	row_positions[0] = 0u;
	/* Duplicate resident slots across lanes are refused. */
	lanes[1].resident_sequence_slot = 7u;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	lanes[1].resident_sequence_slot = 3u;
	/* Head stage prefill: one frame per lane, one token per emitting lane. */
	token_ids[0] = 21u;
	token_ids[1] = 22u;
	token_ids[2] = 23u;
	token_ids[3] = 24u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_lane_indices[2] = 0u;
	row_lane_indices[3] = 1u;
	row_positions[0] = 1u;
	row_positions[1] = 1u;
	row_positions[2] = 2u;
	row_positions[3] = 2u;
	row_sequence_ids[0] = 100u;
	row_sequence_ids[1] = 101u;
	row_sequence_ids[2] = 100u;
	row_sequence_ids[3] = 101u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.submission_id = 78u;
	submission.transaction_id = 1078u;
	submission.dispatch_generation = 2078u;
	submission.step_generation = 3078u;
	submission.new_token_count = 4u;
	submission.row_count = 4u;
	submission.token_count = 4u;
	lanes[0].context_token_count = 3u;
	lanes[1].context_token_count = 3u;
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 2u);
	assert(test_state.completion.submission_id == 78u);
	assert(test_state.completion.token_count == 2u);
	assert(test_state.completion.token_ids[0] == 4242u);
	assert(test_state.completion.token_ids[1] == 4242u);
	assert(test_state.completion.accepted_token_count == 4u);
	lanes[1].flags = 0u;
	submission.submission_id = 79u;
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_OK);
	assert(test_state.completion_count == 3u);
	assert(test_state.completion.token_ids[0] == 4242u);
	assert(test_state.completion.token_ids[1] == 0u);
	/* Wave-major violations are refused in both validate and submit. */
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 0u;
	row_lane_indices[2] = 1u;
	row_lane_indices[3] = 1u;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(test_state.completion_count == 3u);
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_lane_indices[2] = 0u;
	row_lane_indices[3] = 1u;
	assert(library.adapter_interface.snapshot(adapter_state,&snapshot) == SPARK_STATUS_OK);
	/* One decode frame plus two frames per two-lane prefill submission. */
	assert(snapshot.submitted_count == 5u);
	assert(snapshot.completed_count == 5u);
	/* 8 resident lanes x 64 blocks (4096 positions / 64) x 64 tokens. */
	assert(snapshot.kv_token_capacity == 8u * 64u * 64u);
	assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
	assert(library.adapter_interface.validate_submission(adapter_state,&submission) == SPARK_STATUS_BUSY);
	assert(library.adapter_interface.submit(adapter_state,&submission) == SPARK_STATUS_BUSY);
	library.adapter_interface.destroy(adapter_state);

	/* Configuration refusals: stale schema, absolute pack path, positions
	 * beyond the 8192 serving cap. */
	TestQwen38_27bServingConfiguration(&configuration,3u,TEST_QWEN38_27B_SERVING_STALE_CONFIG_PATH,runtime_root,TEST_QWEN38_27B_SERVING_DRIVER_PATH,&test_state);
	adapter_state = 0;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_SCHEMA_ERROR);
	assert(adapter_state == 0);
	configuration.adapter_configuration_path = TEST_QWEN38_27B_SERVING_ABSOLUTE_CONFIG_PATH;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(adapter_state == 0);
	configuration.adapter_configuration_path = TEST_QWEN38_27B_SERVING_OVERRUN_CONFIG_PATH;
	assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_SCHEMA_ERROR);
	assert(adapter_state == 0);
	SparkModelServingAdapterUnloadInterface(&library);
	assert(cudaStreamDestroy((cudaStream_t)test_state.execution_stream) == cudaSuccess);
	return(0);
}
