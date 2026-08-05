#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_MODULE_PATH
#define TEST_MODEL_SERVING_ADAPTER_MODULE_PATH ""
#endif

static SparkStatus TestInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	(void)configuration;
	(void)adapter_state;
	return(SPARK_STATUS_OK);
}

static void TestDestroy(void *adapter_state)
{
	(void)adapter_state;
}

static SparkStatus TestValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	(void)adapter_state;
	(void)submission;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	(void)adapter_state;
	(void)submission;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestProgress(void *adapter_state,uint32_t maximum_step_count)
{
	(void)adapter_state;
	(void)maximum_step_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestQuiesce(void *adapter_state,uint64_t deadline_time_ns)
{
	(void)adapter_state;
	return(deadline_time_ns != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus TestSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	(void)adapter_state;
	(void)snapshot;
	return(SPARK_STATUS_OK);
}

static void TestBuildDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	static const uint32_t layer_counts[13] = {3u,3u,3u,3u,3u,3u,3u,4u,4u,4u,4u,4u,2u};
	uint32_t index;
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV;
	descriptor->stage_count = 13u;
	descriptor->layer_count = 43u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 16384u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 4u;
	descriptor->max_active_sequence_count = 128u;
	descriptor->max_input_row_count = 256u;
	descriptor->max_resident_sequence_count = 512u;
	descriptor->max_output_token_count = 1u;
	descriptor->resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO;
	descriptor->adapter_id = "spark.dsv4.flash.serving.v1";
	descriptor->model_id = "deepseek-ai/DeepSeek-V4-Flash";
	descriptor->model_revision = "fixture";
	descriptor->driver_program_name = "resident_decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	for (index=0u; index<13u; index++)
		descriptor->stage_layer_counts[index] = layer_counts[index];
}

static void TestDescriptor(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	TestBuildDescriptor(&descriptor);
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_OK);
	descriptor.stage_layer_counts[12] = 1u;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_INVALID_ARGUMENT);
	TestBuildDescriptor(&descriptor);
	descriptor.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_INVALID_ARGUMENT);
	descriptor.capability_flags |= SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_OK);
}

static void TestRuntimeLimits(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	TestBuildDescriptor(&descriptor);
	memset(&limits,0,sizeof(limits));
	limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits.max_inflight_submission_count = 2u;
	limits.max_active_sequence_count = 32u;
	limits.max_input_row_count = 128u;
	limits.resident_sequence_capacity = 256u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) == SPARK_STATUS_OK);
	limits.max_inflight_submission_count = 5u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) == SPARK_STATUS_INVALID_ARGUMENT);
	limits.max_inflight_submission_count = 2u;
	limits.max_input_row_count = 16u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestIndependentPrefillCapacity(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	TestBuildDescriptor(&descriptor);
	descriptor.max_input_row_count = SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_OK);
	descriptor.max_input_row_count++;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_INVALID_ARGUMENT);
	descriptor.max_input_row_count = SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT;
	memset(&limits,0,sizeof(limits));
	limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits.max_inflight_submission_count = 1u;
	limits.max_active_sequence_count = 128u;
	limits.max_input_row_count = 32768u;
	limits.resident_sequence_capacity = 128u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) == SPARK_STATUS_OK);
}

static void TestInterfaceValidation(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingAdapterInterface adapter_interface;
	TestBuildDescriptor(&descriptor);
	memset(&adapter_interface,0,sizeof(adapter_interface));
	adapter_interface.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	adapter_interface.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES;
	adapter_interface.descriptor = &descriptor;
	adapter_interface.initialize = TestInitialize;
	adapter_interface.destroy = TestDestroy;
	adapter_interface.validate_submission = TestValidateSubmission;
	adapter_interface.submit = TestSubmit;
	adapter_interface.progress = TestProgress;
	adapter_interface.quiesce = TestQuiesce;
	adapter_interface.snapshot = TestSnapshot;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE) == SPARK_STATUS_OK);
	adapter_interface.quiesce = 0;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	adapter_interface.quiesce = TestQuiesce;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE) == SPARK_STATUS_INVALID_ARGUMENT);
	descriptor.capability_flags |= SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,0u) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestSubmissionValidation(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	uint32_t token_id,row_lane;
	uint64_t row_position,row_sequence;
	TestBuildDescriptor(&descriptor);
	memset(&submission,0,sizeof(submission));
	memset(&lane,0,sizeof(lane));
	lane.request_id = 6u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.sequence_id = 7u;
	lane.resident_sequence_slot = 9u;
	token_id = 10397u;
	row_lane = 0u;
	row_position = 0u;
	row_sequence = 7u;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.submission_id = 9u;
	submission.request_id = 6u;
	submission.sequence_id = 7u;
	submission.control_generation = 1u;
	submission.transaction_id = 2u;
	submission.dispatch_generation = 3u;
	submission.request_generation = 1u;
	submission.step_generation = 4u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = 1u;
	submission.lane_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	submission.lanes = &lane;
	submission.token_ids = &token_id;
	submission.row_lane_indices = &row_lane;
	submission.row_positions = &row_position;
	submission.row_sequence_ids = &row_sequence;
	assert(SparkModelServingAdapterValidateSubmission(&descriptor,&submission) == SPARK_STATUS_OK);
	row_sequence = 8u;
	assert(SparkModelServingAdapterValidateSubmission(&descriptor,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	row_sequence = 7u;
	row_lane = 1u;
	assert(SparkModelServingAdapterValidateSubmission(&descriptor,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	row_lane = 0u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	submission.row_count = 0u;
	submission.token_count = 0u;
	submission.new_token_count = 0u;
	assert(SparkModelServingAdapterValidateSubmission(&descriptor,&submission) == SPARK_STATUS_UNSUPPORTED);
	descriptor.capability_flags |= SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE;
	assert(SparkModelServingAdapterValidateSubmission(&descriptor,&submission) == SPARK_STATUS_OK);
}

static void TestRuntimeSubmissionValidation(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	uint32_t token_ids[2],row_lanes[2];
	uint64_t row_positions[2],row_sequences[2];
	TestBuildDescriptor(&descriptor);
	memset(&limits,0,sizeof(limits));
	limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits.max_inflight_submission_count = 2u;
	limits.max_active_sequence_count = 2u;
	limits.max_input_row_count = 2u;
	limits.resident_sequence_capacity = 8u;
	memset(lanes,0,sizeof(lanes));
	lanes[0].request_id = 10u;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = 1u;
	lanes[0].sequence_id = 100u;
	lanes[0].resident_sequence_slot = 7u;
	lanes[1].request_id = 11u;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = 1u;
	lanes[1].sequence_id = 101u;
	lanes[1].resident_sequence_slot = 3u;
	token_ids[0] = 20u;
	token_ids[1] = 21u;
	row_lanes[0] = 0u;
	row_lanes[1] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_sequences[0] = 100u;
	row_sequences[1] = 101u;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 10u;
	submission.request_id = 10u;
	submission.sequence_id = 100u;
	submission.control_generation = 1u;
	submission.transaction_id = 2u;
	submission.dispatch_generation = 3u;
	submission.request_generation = 1u;
	submission.step_generation = 4u;
	submission.active_sequence_count = 2u;
	submission.new_token_count = 2u;
	submission.lane_count = 2u;
	submission.row_count = 2u;
	submission.token_count = 2u;
	submission.lanes = lanes;
	submission.token_ids = token_ids;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequences;
	assert(SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission) == SPARK_STATUS_OK);
	lanes[1].resident_sequence_slot = 7u;
	assert(SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	lanes[1].resident_sequence_slot = 3u;
	limits.resident_sequence_capacity = 4u;
	assert(SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission) == SPARK_STATUS_CAPACITY_EXCEEDED);
	limits.resident_sequence_capacity = 8u;
	memset(&lanes[1],0,sizeof(lanes[1]));
	lanes[1].resident_sequence_slot = SPARK_MODEL_SERVING_NO_RESIDENT_SEQUENCE_SLOT;
	submission.active_sequence_count = 1u;
	submission.new_token_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	assert(SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission) == SPARK_STATUS_OK);
	lanes[1].resident_sequence_slot = 0u;
	assert(SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestCompletionValidation(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingCompletion completion;
	TestBuildDescriptor(&descriptor);
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = SPARK_STATUS_OK;
	completion.submission_id = 9u;
	completion.control_generation = 1u;
	completion.transaction_id = 2u;
	completion.dispatch_generation = 3u;
	completion.request_generation = 1u;
	completion.step_generation = 4u;
	completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
	completion.token_count = 1u;
	assert(SparkModelServingAdapterValidateCompletion(&descriptor,&completion) == SPARK_STATUS_OK);
	completion.completion_flags = 0u;
	assert(SparkModelServingAdapterValidateCompletion(&descriptor,&completion) == SPARK_STATUS_INVALID_ARGUMENT);
	completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
	completion.token_count = 2u;
	assert(SparkModelServingAdapterValidateCompletion(&descriptor,&completion) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestDynamicLoader(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_MODEL_SERVING_ADAPTER_MODULE_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&library) == SPARK_STATUS_OK);
	assert(strcmp(library.adapter_interface.descriptor->model_id,"test/model") == 0);
	SparkModelServingAdapterUnloadInterface(&library);
	assert(library.dynamic_library == 0);
}

int main(void)
{
	TestDescriptor();
	TestRuntimeLimits();
	TestIndependentPrefillCapacity();
	TestInterfaceValidation();
	TestSubmissionValidation();
	TestRuntimeSubmissionValidation();
	TestCompletionValidation();
	TestDynamicLoader();
	return(0);
}
