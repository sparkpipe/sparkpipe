#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_model_resident_ipc.h"

static void TestBuildDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE;
	descriptor->stage_count = 2u;
	descriptor->layer_count = 4u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 32u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 4u;
	descriptor->max_active_sequence_count = 32u;
	descriptor->max_input_row_count = 32u;
	descriptor->max_resident_sequence_count = 128u;
	descriptor->max_output_token_count = 32u;
	descriptor->adapter_id = "test.adapter";
	descriptor->model_id = "test/model";
	descriptor->model_revision = "revision-1";
	descriptor->driver_program_name = "decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	descriptor->stage_layer_counts[0] = 2u;
	descriptor->stage_layer_counts[1] = 2u;
	descriptor->boundary_sideband_kinds[0] = 1u;
	descriptor->boundary_sideband_bytes_per_sequence[0] = 8192u;
}

static void TestHello(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	SparkModelServingRuntimeLimits limits;
	SparkModelResidentIpcHello hello;
	SparkModelResidentIpcHelloAck ack;
	TestBuildDescriptor(&descriptor);
	memset(&limits,0,sizeof(limits));
	limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	limits.max_inflight_submission_count = 2u;
	limits.max_active_sequence_count = 8u;
	limits.max_input_row_count = 16u;
	limits.resident_sequence_capacity = 64u;
	assert(SparkModelResidentIpcInitializeHello(&hello,7u,1u,1u,&descriptor) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateHello(&hello,sizeof(hello),1u,1u,&descriptor) == SPARK_STATUS_OK);
	hello.model_revision[0] = 'x';
	assert(SparkModelResidentIpcValidateHello(&hello,sizeof(hello),1u,1u,&descriptor) == SPARK_STATUS_TARGET_MISMATCH);
	assert(SparkModelResidentIpcInitializeHelloAck(&ack,7u,SPARK_STATUS_OK,1u,1u,&descriptor,&limits) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),7u,1u,1u,&descriptor,&limits) == SPARK_STATUS_OK);
	assert(ack.header.kind == SPARK_MODEL_RESIDENT_IPC_KIND_HELLO_ACK);
	assert(ack.max_inflight_submission_count == 2u);
	assert(ack.max_active_sequence_count == 8u);
	assert(ack.max_input_row_count == 16u);
	assert(ack.resident_sequence_capacity == 64u);
	assert(ack.kv_logical_page_capacity == 0u);
	assert(ack.kv_physical_page_capacity == 0u);
	assert(ack.boundary_format == SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16);
	assert(ack.linear_weight_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(ack.expert_weight_codec == SPARK_WEIGHT_CODEC_INT8);
	assert(ack.kv_cache_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(ack.input_sideband_kind == 1u);
	assert(ack.input_sideband_bytes_per_sequence == 8192u);
	assert(ack.output_sideband_kind == 0u);
	ack.boundary_element_count++;
	assert(SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),7u,1u,1u,&descriptor,&limits) == SPARK_STATUS_TARGET_MISMATCH);
}

static void TestSubmitResult(void)
{
	SparkModelResidentIpcSubmitResult result;
	assert(SparkModelResidentIpcInitializeSubmitResult(&result,19u,88u,SPARK_STATUS_BUSY) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),19u,88u) == SPARK_STATUS_OK);
	assert(result.status == SPARK_STATUS_BUSY);
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),19u,89u) == SPARK_STATUS_SCHEMA_ERROR);
}

static void TestDecision(void)
{
	SparkModelResidentIpcDecision decision;
	SparkModelResidentIpcDecisionResult result;
	SparkModelServingSubmission submission;
	memset(&submission,0,sizeof(submission));
	submission.submission_id = 88u;
	submission.control_generation = 3u;
	submission.transaction_id = 4u;
	submission.dispatch_generation = 5u;
	assert(SparkModelResidentIpcInitializeDecision(&decision,20u,SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT,&submission) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateDecision(&decision,sizeof(decision)) == SPARK_STATUS_OK);
	assert(decision.header.kind == SPARK_MODEL_RESIDENT_IPC_KIND_DECISION);
	assert(decision.transaction_id == 4u);
	assert(SparkModelResidentIpcInitializeDecisionResult(&result,&decision,SPARK_STATUS_OK) == SPARK_STATUS_OK);
	assert(result.header.kind == SPARK_MODEL_RESIDENT_IPC_KIND_DECISION_RESULT);
	assert(SparkModelResidentIpcValidateDecisionResult(&result,sizeof(result),20u,SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT,&submission) == SPARK_STATUS_OK);
	result.dispatch_generation++;
	assert(SparkModelResidentIpcValidateDecisionResult(&result,sizeof(result),20u,SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT,&submission) == SPARK_STATUS_SCHEMA_ERROR);
	decision.decision = 0u;
	assert(SparkModelResidentIpcValidateDecision(&decision,sizeof(decision)) == SPARK_STATUS_SCHEMA_ERROR);
}

static void TestSubmissionRoundTrip(void)
{
	uint8_t buffer[8192];
	SparkModelServingSubmission submission,decoded;
	SparkModelServingLane lanes[2];
	uint32_t tokens[2],row_lanes[2],message_bytes;
	uint64_t positions[2],sequences[2];
	uint8_t extension[3];
	SparkModelResidentIpcSubmit *wire;
	memset(&submission,0,sizeof(submission));
	memset(lanes,0,sizeof(lanes));
	lanes[0].request_id = 21u;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = 1u;
	lanes[0].sequence_id = 31u;
	lanes[0].resident_sequence_slot = 7u;
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lanes[1].request_id = 22u;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = 1u;
	lanes[1].sequence_id = 32u;
	lanes[1].resident_sequence_slot = 11u;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	tokens[0] = 100u;
	tokens[1] = 101u;
	row_lanes[0] = 0u;
	row_lanes[1] = 1u;
	positions[0] = 9u;
	positions[1] = 12u;
	sequences[0] = 31u;
	sequences[1] = 32u;
	extension[0] = 4u;
	extension[1] = 5u;
	extension[2] = 6u;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 44u;
	submission.request_id = 50u;
	submission.sequence_id = 31u;
	submission.control_generation = 1u;
	submission.transaction_id = 2u;
	submission.dispatch_generation = 3u;
	submission.request_generation = 4u;
	submission.step_generation = 5u;
	submission.active_sequence_count = 2u;
	submission.new_token_count = 2u;
	submission.lane_count = 2u;
	submission.row_count = 2u;
	submission.token_count = 2u;
	submission.model_extension_kind = 9u;
	submission.model_extension_bytes = sizeof(extension);
	submission.residency.word0 = UINT64_C(0x1020304050607080);
	submission.residency.word1 = UINT64_C(0x90a0b0c0d0e0f001);
	submission.residency.generation = 17u;
	submission.residency.owner = 29u;
	submission.lanes = lanes;
	submission.token_ids = tokens;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = positions;
	submission.row_sequence_ids = sequences;
	submission.model_extension = extension;
	assert(SparkModelResidentIpcEncodeSubmission(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_OK);
	assert(decoded.submission_id == 44u);
	assert(decoded.lane_count == 2u);
	assert(decoded.lanes[1].sequence_id == 32u);
	assert(decoded.lanes[0].flags == SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN);
	assert(decoded.token_ids[1] == 101u);
	assert(decoded.row_positions[0] == 9u);
	assert(((const uint8_t *)decoded.model_extension)[2] == 6u);
	assert(memcmp(&decoded.residency,&submission.residency,
		sizeof(decoded.residency)) == 0);
	wire = (SparkModelResidentIpcSubmit *)buffer;
	wire->header.abi_version--;
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_ABI_MISMATCH);
	wire->header.abi_version = SPARK_MODEL_RESIDENT_IPC_ABI_VERSION;
	assert(SparkModelResidentIpcEncodePreparation(&submission,6u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(((const SparkModelResidentIpcHeader *)buffer)->kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE);
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_OK);
	submission.hidden_input_address = buffer;
	submission.hidden_input_bytes = sizeof(buffer);
	assert(SparkModelResidentIpcEncodeSubmission(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	submission.hidden_input_address = 0;
	submission.hidden_input_bytes = 0u;
	submission.boundary_sideband_input_address = buffer;
	submission.boundary_sideband_input_bytes = sizeof(buffer);
	assert(SparkModelResidentIpcEncodeSubmission(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	submission.boundary_sideband_input_address = 0;
	submission.boundary_sideband_input_bytes = 0u;
	wire = (SparkModelResidentIpcSubmit *)buffer;
	wire->row_positions_offset++;
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_SCHEMA_ERROR);
}

static void TestCompletionRoundTrip(void)
{
	uint8_t buffer[8192];
	SparkModelServingCompletion completion,decoded;
	uint32_t index,message_bytes;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = SPARK_STATUS_OK;
	completion.submission_id = 91u;
	completion.control_generation = 1u;
	completion.transaction_id = 2u;
	completion.dispatch_generation = 3u;
	completion.request_generation = 4u;
	completion.step_generation = 5u;
	completion.residency.word0 = UINT64_C(0x0102030405060708);
	completion.residency.word1 = UINT64_C(0x1112131415161718);
	completion.residency.generation = 23u;
	completion.residency.owner = 41u;
	completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS | SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION;
	completion.token_count = 16u;
	completion.model_extension_kind = 3u;
	completion.model_extension_bytes = 2u;
	completion.model_extension[0] = 8u;
	completion.model_extension[1] = 9u;
	for (index=0u; index<completion.token_count; index++)
		completion.token_ids[index] = 700u + index;
	assert(SparkModelResidentIpcEncodeCompletion(&completion,9u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcDecodeCompletion(buffer,message_bytes,&decoded) == SPARK_STATUS_OK);
	assert(decoded.submission_id == 91u);
	assert(decoded.dispatch_generation == 3u);
	assert(decoded.transaction_id == 2u);
	assert(memcmp(&decoded.residency,&completion.residency,
		sizeof(decoded.residency)) == 0);
	assert(decoded.token_count == 16u);
	assert(decoded.token_ids[15] == 715u);
	assert(decoded.model_extension[1] == 9u);
}

static void TestIndependentPrefillMessageCapacity(void)
{
	uint32_t message_bytes;
	assert(SparkModelResidentIpcCalculateSubmitBytes(SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&message_bytes) == SPARK_STATUS_OK);
	assert(message_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MODEL_SERVING_LANE_BYTES);
	assert(SparkModelResidentIpcCalculateSubmitBytes(SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT + 1u,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcCalculateSubmitBytes(SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT + 1u,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestDirectSubmitDescriptorGuard(void)
{
	SparkModelServingAdapterDescriptor descriptor;
	TestBuildDescriptor(&descriptor);
	assert(SparkModelResidentIpcValidateDirectSubmitDescriptor(&descriptor) ==
		SPARK_STATUS_UNSUPPORTED);
	descriptor.stage_count = 1u;
	descriptor.stage_layer_counts[0] = descriptor.layer_count;
	descriptor.stage_layer_counts[1] = 0u;
	descriptor.boundary_sideband_kinds[0] = 0u;
	descriptor.boundary_sideband_bytes_per_sequence[0] = 0u;
	assert(SparkModelResidentIpcValidateDirectSubmitDescriptor(&descriptor) ==
		SPARK_STATUS_OK);
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT;
	assert(SparkModelResidentIpcValidateDirectSubmitDescriptor(&descriptor) ==
		SPARK_STATUS_UNSUPPORTED);
	descriptor.capability_flags &=
		~SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT;
	descriptor.capability_flags |=
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT;
	assert(SparkModelResidentIpcValidateDirectSubmitDescriptor(&descriptor) ==
		SPARK_STATUS_UNSUPPORTED);
}

int main(void)
{
	TestHello();
	TestSubmitResult();
	TestDecision();
	TestSubmissionRoundTrip();
	TestCompletionRoundTrip();
	TestIndependentPrefillMessageCapacity();
	TestDirectSubmitDescriptorGuard();
	return(0);
}
