#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_row_layout.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_MODULE_PATH
#define TEST_MODEL_SERVING_ADAPTER_MODULE_PATH ""
#endif

_Static_assert(SPARK_MODEL_DRIVER_ABI_VERSION == 11u,
	"model-driver admission identity requires ABI 11");

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

static uint32_t TestPrefetchCallCount;
static SparkStatus TestPrefetchStatus;
static uint32_t TestResolveCallCount;
static uint32_t TestLastResolution;
static SparkStatus TestResolveStatus;

static SparkStatus TestPrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submissions,
	uint32_t submission_count)
{
	assert(adapter_state != 0);
	assert(submissions != 0);
	assert(submission_count == 1u);
	TestPrefetchCallCount++;
	return(TestPrefetchStatus);
}

static SparkStatus TestResolvePrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submission,
	uint32_t resolution)
{
	assert(adapter_state != 0);
	assert(submission != 0);
	TestResolveCallCount++;
	TestLastResolution = resolution;
	return(TestResolveStatus);
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
	descriptor->minimum_efficient_submission_row_count = 16u;
	descriptor->adapter_id = "spark.dsv4.flash.serving.v1";
	descriptor->model_id = "deepseek-ai/DeepSeek-V4-Flash-0731";
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
	descriptor.minimum_efficient_submission_row_count = descriptor.max_input_row_count + 1u;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_INVALID_ARGUMENT);
	TestBuildDescriptor(&descriptor);
	descriptor.stage_layer_counts[12] = 1u;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_INVALID_ARGUMENT);
	TestBuildDescriptor(&descriptor);
	descriptor.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_INVALID_ARGUMENT);
	descriptor.capability_flags |= SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_OK);
	TestBuildDescriptor(&descriptor);
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) == SPARK_STATUS_OK);
	descriptor.capability_flags &=
		~SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV;
	descriptor.resident_sequence_slot_reuse =
		SPARK_MODEL_SERVING_SLOT_REUSE_NONE;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestHybridDescriptor(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	uint32_t index;
	TestBuildDescriptor(&descriptor);
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP;
	descriptor.stage_count = 16u;
	descriptor.layer_count = 8u;
	descriptor.parallel_group_size = 4u;
	memset(descriptor.stage_layer_counts,0,
		sizeof(descriptor.stage_layer_counts));
	for (index=0u; index<descriptor.stage_count; index++)
		descriptor.stage_layer_counts[index] = 2u;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) ==
		SPARK_STATUS_OK);
	descriptor.stage_layer_counts[5] = 3u;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	descriptor.stage_layer_counts[5] = 2u;
	descriptor.parallel_group_size = 0u;
	assert(SparkModelServingAdapterValidateDescriptor(&descriptor) ==
		SPARK_STATUS_INVALID_ARGUMENT);
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

static void TestJitKvRuntimeLimits(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	TestBuildDescriptor(&descriptor);
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV;
	descriptor.cache_block_token_count = 128u;
	memset(&limits,0,sizeof(limits));
	limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits.max_inflight_submission_count = 2u;
	limits.max_active_sequence_count = 32u;
	limits.max_input_row_count = 128u;
	limits.resident_sequence_capacity = 256u;
	limits.kv_logical_page_capacity = 1024u;
	limits.kv_physical_page_capacity = 64u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) ==
		SPARK_STATUS_OK);
	limits.kv_physical_page_capacity = 31u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	limits.kv_physical_page_capacity = 64u;
	limits.kv_logical_page_capacity = 255u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	limits.kv_logical_page_capacity = 1024u;
	descriptor.capability_flags &=
		~SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV;
	descriptor.capability_flags &=
		~SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH;
	descriptor.cache_block_token_count = 0u;
	assert(SparkModelServingAdapterValidateRuntimeLimits(&descriptor,&limits) ==
		SPARK_STATUS_INVALID_ARGUMENT);
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
	adapter_interface.prefetch = TestPrefetch;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,0u) ==
		SPARK_STATUS_OK);
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV;
	descriptor.cache_block_token_count = 4u;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,0u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	adapter_interface.resolve_prefetch = TestResolvePrefetch;
	assert(SparkModelServingAdapterValidateInterface(&adapter_interface,0u) ==
		SPARK_STATUS_OK);
}

static void TestPreparedResolution(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingAdapterInterface adapter_interface;
	SparkModelServingSubmission submission;
	uint32_t adapter_state;
	TestBuildDescriptor(&descriptor);
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV;
	descriptor.cache_block_token_count = 4u;
	memset(&adapter_interface,0,sizeof(adapter_interface));
	adapter_interface.descriptor = &descriptor;
	adapter_interface.resolve_prefetch = TestResolvePrefetch;
	memset(&submission,0,sizeof(submission));
	adapter_state = 1u;
	TestResolveCallCount = 0u;
	TestResolveStatus = SPARK_STATUS_OK;
	assert(SparkModelServingAdapterResolvePrefetch(&adapter_interface,
		&adapter_state,&submission,
		SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT) == SPARK_STATUS_OK);
	assert(TestResolveCallCount == 1u);
	assert(TestLastResolution ==
		SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT);
	TestResolveStatus = SPARK_STATUS_BUSY;
	assert(SparkModelServingAdapterResolvePrefetch(&adapter_interface,
		&adapter_state,&submission,
		SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) ==
		SPARK_STATUS_INTERNAL_ERROR);
	assert(SparkModelServingAdapterResolvePrefetch(&adapter_interface,
		&adapter_state,&submission,0u) == SPARK_STATUS_INVALID_ARGUMENT);
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

static void TestPreparationInvokesOptionalPrefetch(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingAdapterInterface adapter_interface;
	SparkModelServingSubmission submission;
	uint32_t adapter_state;
	TestBuildDescriptor(&descriptor);
	memset(&adapter_interface,0,sizeof(adapter_interface));
	memset(&submission,0,sizeof(submission));
	adapter_interface.descriptor = &descriptor;
	adapter_interface.validate_submission = TestValidateSubmission;
	adapter_interface.prefetch = TestPrefetch;
	adapter_state = 1u;
	TestPrefetchCallCount = 0u;
	TestPrefetchStatus = SPARK_STATUS_OK;
	assert(SparkModelServingAdapterPrepareSubmission(&adapter_interface,&adapter_state,&submission) == SPARK_STATUS_OK);
	assert(TestPrefetchCallCount == 0u);
	descriptor.capability_flags |= SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH;
	TestPrefetchStatus = SPARK_STATUS_BUSY;
	assert(SparkModelServingAdapterPrepareSubmission(&adapter_interface,&adapter_state,&submission) == SPARK_STATUS_BUSY);
	assert(TestPrefetchCallCount == 1u);
	adapter_interface.prefetch = 0;
	assert(SparkModelServingAdapterPrepareSubmission(&adapter_interface,&adapter_state,&submission) == SPARK_STATUS_ABI_MISMATCH);
}

static void TestDriverCacheLaneMapping(void)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane serving_lanes[2];
	SparkModelDriverCacheLane driver_lanes[2];
	uint32_t lane_count;
	memset(&submission,0,sizeof(submission));
	memset(serving_lanes,0,sizeof(serving_lanes));
	serving_lanes[0].sequence_id = 101u;
	serving_lanes[0].sequence_position = 128u;
	serving_lanes[0].request_generation = 1001u;
	serving_lanes[0].step_generation = 2001u;
	serving_lanes[0].resident_sequence_slot = 7u;
	serving_lanes[0].context_token_count = 256u;
	serving_lanes[0].cache_prefix_token_count = 128u;
	serving_lanes[0].cache_publish_token_count = 256u;
	serving_lanes[0].cache_prefix_identity.sha256[0] = 1u;
	serving_lanes[0].cache_publish_identity.sha256[0] = 2u;
	serving_lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX |
		SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
	serving_lanes[1].sequence_id = 102u;
	serving_lanes[1].sequence_position = 5u;
	serving_lanes[1].request_generation = 1002u;
	serving_lanes[1].step_generation = 2002u;
	serving_lanes[1].resident_sequence_slot = 8u;
	serving_lanes[1].context_token_count = 6u;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.active_sequence_count = 2u;
	submission.lane_count = 2u;
	submission.lanes = serving_lanes;
	assert(SparkModelServingAdapterBuildDriverCacheLanes(&submission,driver_lanes,2u,&lane_count) == SPARK_STATUS_OK);
	assert(lane_count == 2u);
	assert(driver_lanes[0].flags == (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH));
	assert(driver_lanes[0].request_generation == 1001u);
	assert(driver_lanes[0].step_generation == 2001u);
	assert(driver_lanes[0].prefix_token_count == 128u);
	assert(driver_lanes[0].publish_identity.sha256[0] == 2u);
	assert(driver_lanes[1].flags == 0u);
	assert(driver_lanes[1].request_generation == 1002u);
	assert(driver_lanes[1].step_generation == 2002u);
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	assert(SparkModelServingAdapterBuildDriverCacheLanes(&submission,driver_lanes,2u,&lane_count) == SPARK_STATUS_OK);
	assert(driver_lanes[0].flags == SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE);
	assert(driver_lanes[0].prefix_token_count == 0u);
	assert(SparkModelServingAdapterBuildDriverCacheLanes(&submission,driver_lanes,1u,&lane_count) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestDriverCacheAdmissionIdentity(void)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverCacheLane lane;
	memset(&request,0,sizeof(request));
	memset(&lane,0,sizeof(lane));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.submission_id = 10u;
	request.control_generation = 11u;
	request.transaction_id = 12u;
	request.request_generation = 13u;
	request.step_generation = 14u;
	request.active_slot_count = 1u;
	request.admission_flags =
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	request.cache_lane_count = 1u;
	request.cache_lanes = &lane;
	lane.sequence_id = 20u;
	lane.request_generation = request.request_generation;
	lane.step_generation = request.step_generation;
	lane.resident_sequence_slot = 0u;
	lane.context_token_count = 1u;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) != 0u);
	request.submission_id = 0u;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) == 0u);
	request.submission_id = 10u;
	lane.request_generation++;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) != 0u);
	lane.request_generation = 0u;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) == 0u);
	lane.request_generation = request.request_generation + 1u;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) != 0u);
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) != 0u);
	request.admission_flags = 0u;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) != 0u);
	request.transaction_id = 0u;
	assert(SparkModelDriverAdmissionRequestIsValid(&request) == 0u);
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
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lanes[1].request_id = 11u;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = 1u;
	lanes[1].sequence_id = 101u;
	lanes[1].resident_sequence_slot = 3u;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
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
	lanes[1].flags = 0u;
	assert(SparkModelServingAdapterValidateRuntimeSubmission(&descriptor,&limits,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
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

static void TestEmitRowSelection(void)
{
	static SparkModelServingLane lanes[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	static uint32_t row_lanes[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT * 2u];
	static uint32_t emit_rows[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	static uint32_t emit_lanes[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	static uint64_t row_positions[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT * 2u];
	static uint64_t row_sequences[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT * 2u];
	SparkModelServingSubmission submission;
	uint32_t emit_count,lane,row;
	memset(&submission,0,sizeof(submission));
	memset(lanes,0,sizeof(lanes));
	for (lane=0u; lane<SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		lanes[lane].sequence_id = 1000u + lane;
		lanes[lane].context_token_count = lane == 0u ? 1u : 2u;
		row_lanes[lane] = lane;
		row_positions[lane] = 0u;
		row_sequences[lane] = lanes[lane].sequence_id;
	}
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	row = SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT;
	for (lane=1u; lane<SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		row_lanes[row] = lane;
		row_positions[row] = 1u;
		row_sequences[row] = lanes[lane].sequence_id;
		row++;
	}
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.active_sequence_count = SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT;
	submission.lane_count = SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT;
	submission.row_count = row;
	submission.lanes = lanes;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequences;
	assert(SparkModelServingAdapterSelectEmitRows(&submission,emit_rows,emit_lanes,SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,&emit_count) == SPARK_STATUS_OK);
	assert(emit_count == 1u);
	assert(emit_rows[0] == 0u);
	assert(emit_lanes[0] == 0u);
	for (lane=0u; lane<SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	assert(SparkModelServingAdapterSelectEmitRows(&submission,emit_rows,emit_lanes,SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT - 1u,&emit_count) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(emit_count == 0u);
	assert(SparkModelServingAdapterSelectEmitRows(&submission,emit_rows,emit_lanes,SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,&emit_count) == SPARK_STATUS_OK);
	assert(emit_count == SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT);
	assert(emit_rows[0] == 0u && emit_lanes[0] == 0u);
	assert(emit_rows[1] == SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT && emit_lanes[1] == 1u);
	assert(emit_rows[emit_count - 1u] == row - 1u);
	assert(SparkModelServingAdapterSelectEmitRows(&submission,0,0,0u,&emit_count) == SPARK_STATUS_OK);
	assert(emit_count == SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT);
	lanes[7].flags |= UINT32_C(0x80000000);
	assert(SparkModelServingAdapterSelectEmitRows(&submission,emit_rows,emit_lanes,SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,&emit_count) == SPARK_STATUS_INVALID_ARGUMENT);
	lanes[7].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	row_positions[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT + 6u] = 2u;
	assert(SparkModelServingAdapterSelectEmitRows(&submission,emit_rows,emit_lanes,SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,&emit_count) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestMaximumRoundMajorRowLayout(void)
{
	static uint32_t row_lanes[SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT];
	static uint32_t occurrences[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	static uint32_t last_rows[SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRowLayoutDenseLaneContext dense;
	uint32_t lane,row;
	dense.lane_count = SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT;
	for (lane=0u; lane<dense.lane_count; lane++)
		row_lanes[lane] = lane;
	for (row=dense.lane_count; row<SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT; row++)
		row_lanes[row] = 0u;
	assert(SparkRowLayoutValidateRoundMajor(SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,dense.lane_count,row_lanes,SparkRowLayoutDenseLaneOrdinal,&dense,occurrences,last_rows) == SPARK_STATUS_OK);
	assert(occurrences[0] == SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT - dense.lane_count + 1u);
	assert(last_rows[0] == SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT - 1u);
	assert(last_rows[dense.lane_count - 1u] == dense.lane_count - 1u);
	assert(SparkRowLayoutRoundMajorWaveRowCount(0u,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,row_lanes,SparkRowLayoutDenseLaneOrdinal,&dense) == dense.lane_count);
	assert(SparkRowLayoutRoundMajorWaveRowCount(dense.lane_count,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,row_lanes,SparkRowLayoutDenseLaneOrdinal,&dense) == 1u);
	row_lanes[0] = 1u;
	assert(SparkRowLayoutValidateRoundMajor(SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,dense.lane_count,row_lanes,SparkRowLayoutDenseLaneOrdinal,&dense,occurrences,last_rows) == SPARK_STATUS_INVALID_ARGUMENT);
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
	TestHybridDescriptor();
	TestRuntimeLimits();
	TestIndependentPrefillCapacity();
	TestJitKvRuntimeLimits();
	TestInterfaceValidation();
	TestPreparedResolution();
	TestSubmissionValidation();
	TestPreparationInvokesOptionalPrefetch();
	TestDriverCacheLaneMapping();
	TestDriverCacheAdmissionIdentity();
	TestRuntimeSubmissionValidation();
	TestEmitRowSelection();
	TestMaximumRoundMajorRowLayout();
	TestCompletionValidation();
	TestDynamicLoader();
	return(0);
}
