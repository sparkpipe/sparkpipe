#include <assert.h>
#include <string.h>
#include <time.h>

#include "runtime/model_continuation_lease.h"
#include "sparkpipe/spark_model_resident_ipc.h"

static void TestBuildDescriptor(SparkModelServingAdapterDescriptor *descriptor)
{
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->cache_block_token_count = 4u;
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH;
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
	limits.kv_logical_page_capacity = 256u;
	limits.kv_physical_page_capacity = 8u;
	limits.output_vocabulary = 32000u;
	assert(SparkModelResidentIpcInitializeHello(&hello,7u,1u,1u,0x1234u,&descriptor) == SPARK_STATUS_OK);
	assert(hello.session_epoch == 0x1234u);
	assert(SparkModelResidentIpcValidateHello(&hello,sizeof(hello),1u,1u,&descriptor) == SPARK_STATUS_OK);
	hello.model_revision[0] = 'x';
	assert(SparkModelResidentIpcValidateHello(&hello,sizeof(hello),1u,1u,&descriptor) == SPARK_STATUS_TARGET_MISMATCH);
	hello.model_revision[0] = '\0';
	hello.session_epoch = 0u;
	assert(SparkModelResidentIpcValidateHello(&hello,sizeof(hello),1u,1u,&descriptor) == SPARK_STATUS_TARGET_MISMATCH);
	hello.session_epoch = 0x1234u;
	assert(SparkModelResidentIpcInitializeHelloAck(&ack,7u,SPARK_STATUS_OK,1u,
		1u,13u,0x1234u,&descriptor,&limits,0u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),7u,1u,1u,0x1234u,&descriptor,&limits) == SPARK_STATUS_TARGET_MISMATCH);
	assert(SparkModelResidentIpcInitializeHelloAck(&ack,7u,SPARK_STATUS_OK,1u,
		1u,13u,0x1234u,&descriptor,&limits,8192u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),7u,1u,1u,0x1234u,&descriptor,&limits) == SPARK_STATUS_OK);
	assert(ack.max_sequence_positions == 8192u);
	assert(ack.header.kind == SPARK_MODEL_RESIDENT_IPC_KIND_HELLO_ACK);
	assert(ack.client_generation == 13u);
	assert(ack.max_inflight_submission_count == 2u);
	assert(ack.max_active_sequence_count == 8u);
	assert(ack.max_input_row_count == 16u);
	assert(ack.resident_sequence_capacity == 64u);
	assert(ack.kv_logical_page_capacity == 256u);
	assert(ack.kv_physical_page_capacity == 8u);
	assert(ack.boundary_format == SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16);
	assert(ack.linear_weight_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(ack.expert_weight_codec == SPARK_WEIGHT_CODEC_INT8);
	assert(ack.kv_cache_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(ack.input_sideband_kind == 1u);
	assert(ack.input_sideband_bytes_per_sequence == 8192u);
	assert(ack.output_sideband_kind == 0u);
	ack.boundary_element_count++;
	assert(SparkModelResidentIpcValidateHelloAck(&ack,sizeof(ack),7u,1u,1u,0x1234u,&descriptor,&limits) == SPARK_STATUS_TARGET_MISMATCH);
}

static void TestSubmitResult(void)
{
	SparkModelResidentIpcSubmitResult result;
	assert(SparkModelResidentIpcInitializeSubmitResult(&result,19u,88u,SPARK_STATUS_BUSY) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),19u,88u) == SPARK_STATUS_OK);
	assert(result.status == SPARK_STATUS_BUSY);
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),19u,89u) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentIpcInitializeSubmitResult(&result,20u,90u,SPARK_STATUS_NO_LANE) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),20u,90u) == SPARK_STATUS_OK && result.status == SPARK_STATUS_NO_LANE);
	assert(SparkModelResidentIpcInitializeSubmitResult(&result,21u,91u,SPARK_STATUS_EVICT_DENIED) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),21u,91u) == SPARK_STATUS_OK && result.status == SPARK_STATUS_EVICT_DENIED);
	assert(SparkModelResidentIpcInitializeSubmitResult(&result,22u,92u,(SparkStatus)(SPARK_STATUS_MAXIMUM + 1)) != SPARK_STATUS_OK);
	result.status = (uint32_t)SPARK_STATUS_MAXIMUM + 1u;
	assert(SparkModelResidentIpcValidateSubmitResult(&result,sizeof(result),21u,91u) == SPARK_STATUS_SCHEMA_ERROR);
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
	assert(SparkModelResidentIpcEncodePreparation(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
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
	assert(decoded.deadline_time_ns == 0u && ((SparkModelResidentIpcSubmit *)buffer)->deadline_remaining_ns == 0u);
	{
		struct timespec now;
		uint64_t before,after;
		assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
		before = (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
		submission.deadline_time_ns = before + UINT64_C(5000000000);
		assert(SparkModelResidentIpcEncodePreparation(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
		assert(((SparkModelResidentIpcSubmit *)buffer)->deadline_remaining_ns != 0u && ((SparkModelResidentIpcSubmit *)buffer)->deadline_remaining_ns <= UINT64_C(5000000000));
		((SparkModelResidentIpcSubmit *)buffer)->deadline_remaining_ns = UINT64_C(2000000000);
		assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_OK);
		assert(clock_gettime(CLOCK_MONOTONIC,&now) == 0);
		after = (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
		assert(decoded.deadline_time_ns >= before + UINT64_C(2000000000) && decoded.deadline_time_ns <= after + UINT64_C(2000000000));
		submission.deadline_time_ns = before - 1u;
		assert(SparkModelResidentIpcEncodePreparation(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
		assert(((SparkModelResidentIpcSubmit *)buffer)->deadline_remaining_ns == 1u);
		submission.deadline_time_ns = 0u;
		assert(SparkModelResidentIpcEncodePreparation(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	}
	wire = (SparkModelResidentIpcSubmit *)buffer;
	wire->header.abi_version--;
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_ABI_MISMATCH);
	wire->header.abi_version = SPARK_MODEL_RESIDENT_IPC_ABI_VERSION;
	assert(SparkModelResidentIpcEncodePreparation(&submission,6u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(((const SparkModelResidentIpcHeader *)buffer)->kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE);
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodeContinuation(&submission,7u,13u,buffer,
		sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	wire = (SparkModelResidentIpcSubmit *)buffer;
	assert(wire->header.kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE);
	assert(wire->client_generation == 13u);
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) ==
		SPARK_STATUS_OK);
	wire->client_generation = 0u;
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentIpcEncodeContinuation(&submission,7u,0u,buffer,
		sizeof(buffer),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	submission.hidden_input_address = buffer;
	submission.hidden_input_bytes = sizeof(buffer);
	assert(SparkModelResidentIpcEncodePreparation(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	submission.hidden_input_address = 0;
	submission.hidden_input_bytes = 0u;
	submission.boundary_sideband_input_address = buffer;
	submission.boundary_sideband_input_bytes = sizeof(buffer);
	assert(SparkModelResidentIpcEncodePreparation(&submission,5u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	submission.boundary_sideband_input_address = 0;
	submission.boundary_sideband_input_bytes = 0u;
	wire = (SparkModelResidentIpcSubmit *)buffer;
	wire->row_positions_offset++;
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_SCHEMA_ERROR);
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH;
	submission.tokens_per_sequence = submission.new_token_count = submission.row_count = submission.token_count = 0u;
	for (uint32_t lane=0u; lane<2u; lane++)
	{
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
		lanes[lane].sequence_position = lanes[lane].context_token_count = lanes[lane].cache_publish_token_count = 17u + lane;
		memset(&lanes[lane].cache_publish_identity,(int)(0x60u + lane),sizeof(lanes[lane].cache_publish_identity));
	}
	assert(SparkModelResidentIpcEncodePreparation(&submission,8u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(((const SparkModelResidentIpcHeader *)buffer)->kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE);
	assert(SparkModelResidentIpcDecodeSubmission(buffer,message_bytes,&decoded) == SPARK_STATUS_OK);
	assert(decoded.work_kind == SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH);
	assert(decoded.row_count == 0u && decoded.token_count == 0u && decoded.new_token_count == 0u && decoded.tokens_per_sequence == 0u);
	assert(decoded.hidden_input_address == 0 && decoded.hidden_output_address == 0);
	assert(decoded.boundary_sideband_input_address == 0 && decoded.boundary_sideband_output_address == 0);
	assert(memcmp(decoded.lanes,lanes,sizeof(lanes)) == 0);
	assert(memcmp(&decoded.residency,&submission.residency,sizeof(decoded.residency)) == 0);
	assert(decoded.control_generation == submission.control_generation && decoded.transaction_id == submission.transaction_id && decoded.dispatch_generation == submission.dispatch_generation);

}

static void TestContinuationLease(void)
{
	SparkModelContinuationLease leases[2];
	uint64_t next_sequence_position;
	uint32_t order[2] = {1u,0u};
	uint32_t index;
	memset(leases,0,sizeof(leases));
	assert(SparkModelContinuationLeaseDecodePosition(41u,1u,
		&next_sequence_position) == SPARK_STATUS_OK);
	assert(next_sequence_position == 41u);
	assert(SparkModelContinuationLeaseDecodePosition(41u,8u,
		&next_sequence_position) == SPARK_STATUS_OK);
	assert(next_sequence_position == 48u);
	assert(SparkModelContinuationLeaseDecodePosition(41u,0u,
		&next_sequence_position) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelContinuationLeaseDecodePosition(UINT64_MAX,2u,
		&next_sequence_position) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkModelContinuationLeaseVerifyPosition(40u,3u,
		&next_sequence_position) == SPARK_STATUS_OK);
	assert(next_sequence_position == 43u);
	assert(SparkModelContinuationLeaseVerifyPosition(40u,0u,
		&next_sequence_position) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelContinuationLeaseVerifyPosition(UINT64_MAX,1u,
		&next_sequence_position) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkModelContinuationLeaseIsActive(&leases[0]) == 0u);
	assert(SparkModelContinuationLeaseEstablish(&leases[0],7u,3u,41u,11u) ==
		SPARK_STATUS_OK);
	assert(SparkModelContinuationLeaseEstablish(&leases[1],7u,3u,99u,14u) ==
		SPARK_STATUS_OK);
	for (index=0u; index<2u; index++)
		assert(SparkModelContinuationLeaseValidate(&leases[order[index]],7u,3u,
			order[index] == 0u ? 41u : 99u,
			order[index] == 0u ? 15u : 16u) == SPARK_STATUS_OK);
	assert(SparkModelContinuationLeaseValidate(&leases[0],8u,3u,41u,15u) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelContinuationLeaseValidate(&leases[0],7u,4u,41u,15u) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelContinuationLeaseValidate(&leases[0],7u,3u,42u,15u) ==
		SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelContinuationLeaseValidate(&leases[0],7u,3u,41u,11u) ==
		SPARK_STATUS_SCHEMA_ERROR);
	SparkModelContinuationLeaseInvalidate(&leases[0]);
	assert(SparkModelContinuationLeaseIsActive(&leases[0]) == 0u);
	assert(SparkModelContinuationLeaseValidate(&leases[0],7u,3u,41u,15u) ==
		SPARK_STATUS_NOT_FOUND);
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

static void TestCompletionFill(SparkModelServingCompletion *completion,uint32_t tokens,uint32_t stride,uint32_t extension_bytes,uint32_t seed)
{
	uint32_t index;
	memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion->status = SPARK_STATUS_OK;
	completion->submission_id = seed;
	completion->control_generation = 1u;
	completion->transaction_id = 2u;
	completion->dispatch_generation = 3u;
	completion->request_generation = 4u;
	completion->step_generation = seed + 5u;
	completion->residency.word0 = seed + 6u;
	completion->completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS | (extension_bytes != 0u ? SPARK_MODEL_SERVING_COMPLETION_FLAG_MODEL_EXTENSION : 0u);
	completion->token_count = tokens;
	completion->tokens_per_sequence = 1u;
	for (index=0u; index<tokens; index++)
		completion->token_ids[index] = seed * 100u + index;
	completion->logprob_stride = stride;
	completion->logprob_entry_count = tokens * stride;
	for (index=0u; index<tokens * stride; index++)
	{
		completion->logprobs[index].token = seed + index;
		completion->logprobs[index].logprob = -0.25f * (float)(index + 1u);
	}
	completion->model_extension_kind = extension_bytes != 0u ? 3u : 0u;
	completion->model_extension_bytes = extension_bytes;
	for (index=0u; index<extension_bytes; index++)
		completion->model_extension[index] = (uint8_t)(seed + index);
	completion->queue_delay_ns = seed + 7u;
	completion->service_time_ns = seed + 8u;
}

static void TestCompletionDecodeReplacesHeldCompletion(void)
{
	static uint8_t buffer[65536];
	static SparkModelServingCompletion large,small,held,fresh;
	uint32_t message_bytes;
	TestCompletionFill(&large,512u,8u,64u,7u);
	TestCompletionFill(&small,3u,2u,0u,11u);
	memset(&held,0,sizeof(held));
	assert(SparkModelResidentIpcEncodeCompletion(&large,9u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcDecodeCompletionReplacing(buffer,message_bytes,&held) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcDecodeCompletion(buffer,message_bytes,&fresh) == SPARK_STATUS_OK);
	assert(memcmp(&held,&fresh,sizeof(held)) == 0);
	assert(SparkModelResidentIpcEncodeCompletion(&small,10u,buffer,sizeof(buffer),&message_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcDecodeCompletionReplacing(buffer,message_bytes,&held) == SPARK_STATUS_OK);
	memset(&fresh,0xa5,sizeof(fresh));
	assert(SparkModelResidentIpcDecodeCompletion(buffer,message_bytes,&fresh) == SPARK_STATUS_OK);
	assert(memcmp(&held,&fresh,sizeof(held)) == 0);
	buffer[message_bytes - 1u] ^= 0xffu;
	((SparkModelResidentIpcCompletion *)buffer)->reserved = 1u;
	assert(SparkModelResidentIpcDecodeCompletionReplacing(buffer,message_bytes,&held) == SPARK_STATUS_SCHEMA_ERROR);
	assert(memcmp(&held,&fresh,sizeof(held)) == 0);
}

static void TestStampedSubmissionMatchesDirectEncoding(void)
{
	static uint8_t encoded[8192],stamped[8192];
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	uint32_t tokens[2],row_lanes[2],encoded_bytes,stamped_bytes;
	uint64_t positions[2],sequences[2];
	memset(&submission,0,sizeof(submission));
	memset(lanes,0,sizeof(lanes));
	for (uint32_t lane=0u; lane<2u; lane++)
	{
		lanes[lane].request_id = 21u + lane;
		lanes[lane].request_generation = 1u;
		lanes[lane].step_generation = 5u;
		lanes[lane].sequence_id = 31u + lane;
		lanes[lane].sequence_position = 9u;
		lanes[lane].context_token_count = 10u;
		lanes[lane].resident_sequence_slot = 7u + lane;
		lanes[lane].input_token_id = 100u + lane;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
		tokens[lane] = 100u + lane;
		row_lanes[lane] = lane;
		positions[lane] = 9u;
		sequences[lane] = 31u + lane;
	}
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.submission_id = 44u;
	submission.request_id = 44u;
	submission.sequence_id = 44u;
	submission.control_generation = 1u;
	submission.transaction_id = 44u;
	submission.dispatch_generation = 44u;
	submission.request_generation = 1u;
	submission.step_generation = 5u;
	submission.active_sequence_count = 2u;
	submission.new_token_count = 2u;
	submission.lane_count = 2u;
	submission.row_count = 2u;
	submission.token_count = 2u;
	submission.tokens_per_sequence = 8u;
	submission.lanes = lanes;
	submission.token_ids = tokens;
	submission.row_lane_indices = row_lanes;
	submission.row_positions = positions;
	submission.row_sequence_ids = sequences;
	assert(SparkModelResidentIpcEncodeContinuation(&submission,17u,23u,encoded,sizeof(encoded),&encoded_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodeContinuation(&submission,1u,1u,stamped,sizeof(stamped),&stamped_bytes) == SPARK_STATUS_OK);
	assert(stamped_bytes == encoded_bytes);
	assert(SparkModelResidentIpcStampSubmission(stamped,stamped_bytes,17u,23u) == SPARK_STATUS_OK);
	assert(memcmp(stamped,encoded,encoded_bytes) == 0);
	assert(SparkModelResidentIpcStampSubmission(stamped,stamped_bytes,17u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcStampSubmission(stamped,stamped_bytes,0u,23u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcStampSubmission(stamped,stamped_bytes - 4u,17u,23u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcEncodePreparation(&submission,17u,encoded,sizeof(encoded),&encoded_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodePreparation(&submission,1u,stamped,sizeof(stamped),&stamped_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcStampSubmission(stamped,stamped_bytes,17u,23u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcStampSubmission(stamped,stamped_bytes,17u,0u) == SPARK_STATUS_OK);
	assert(memcmp(stamped,encoded,encoded_bytes) == 0);
}

static void TestTokenContinuationBuild(SparkModelServingSubmission *submission,SparkModelServingLane *lanes,uint32_t *tokens,uint32_t *row_lanes,uint64_t *positions,uint64_t *sequences,uint32_t lane_count)
{
	memset(submission,0,sizeof(*submission));
	memset(lanes,0,lane_count * sizeof(lanes[0]));
	for (uint32_t lane=0u; lane<lane_count; lane++)
	{
		lanes[lane].request_id = 21u + lane;
		lanes[lane].request_generation = 3u;
		lanes[lane].step_generation = 77u;
		lanes[lane].sequence_id = 31u + lane;
		lanes[lane].sequence_position = 90u + lane;
		lanes[lane].context_token_count = 91u + lane;
		lanes[lane].state_target_token_count = 91u + lane;
		lanes[lane].resident_sequence_slot = 5u + 2u * lane;
		lanes[lane].input_token_id = 400u + lane;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN | (lane == 1u ? SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX : 0u);
		lanes[lane].cache_prefix_token_count = lane == 1u ? 64u : 0u;
		memset(&lanes[lane].cache_prefix_identity,lane == 1u ? 0x31 : 0,sizeof(lanes[lane].cache_prefix_identity));
		lanes[lane].sampling = SparkSamplingRuleWith(lane == 2u ? 0.7f : 0.0f,lane == 2u ? 5u : 0u,1.0f,0u,0u);
		tokens[lane] = lanes[lane].input_token_id;
		row_lanes[lane] = lane;
		positions[lane] = lanes[lane].sequence_position;
		sequences[lane] = lanes[lane].sequence_id;
	}
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->submission_id = 70u;
	submission->request_id = 70u;
	submission->sequence_id = 70u;
	submission->control_generation = 2u;
	submission->transaction_id = 70u;
	submission->dispatch_generation = 70u;
	submission->request_generation = 1u;
	submission->step_generation = 77u;
	submission->priority = 2u;
	submission->tokens_per_sequence = 8u;
	submission->residency.word0 = 70u;
	submission->residency.owner = 1u;
	submission->active_sequence_count = lane_count;
	submission->new_token_count = lane_count;
	submission->lane_count = lane_count;
	submission->row_count = lane_count;
	submission->token_count = lane_count;
	submission->lanes = lanes;
	submission->token_ids = tokens;
	submission->row_lane_indices = row_lanes;
	submission->row_positions = positions;
	submission->row_sequence_ids = sequences;
}

static void TestTokenContinuationCarriesEveryLane(void)
{
	static uint8_t full[16384],compact[16384],expanded[16384];
	SparkModelServingSubmission submission,decoded;
	SparkModelServingLane lanes[3],previous[3];
	SparkModelResidentIpcContinueToken token;
	const SparkModelResidentIpcContinueToken *tokens_view;
	uint32_t tokens[3],row_lanes[3],full_bytes,compact_bytes,expanded_bytes,count;
	uint64_t positions[3],sequences[3];
	TestTokenContinuationBuild(&submission,lanes,tokens,row_lanes,positions,sequences,3u);
	for (uint32_t lane=0u; lane<3u; lane++)
	{
		previous[lane] = lanes[lane];
		previous[lane].step_generation = 60u;
		previous[lane].sequence_position -= 8u;
		previous[lane].context_token_count -= 8u;
		previous[lane].state_target_token_count -= 8u;
		previous[lane].input_token_id = 7u;
		SparkModelResidentIpcContinueTokenOf(&lanes[lane],&token);
		SparkModelServingLane rebuilt;
		SparkModelResidentIpcContinuedLane(&previous[lane],&token,77u,&rebuilt);
		assert(memcmp(&rebuilt,&lanes[lane],sizeof(rebuilt)) == 0);
	}
	assert(SparkModelResidentIpcEncodeContinuation(&submission,13u,19u,full,sizeof(full),&full_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodeTokenContinuation(&submission,13u,19u,compact,sizeof(compact),&compact_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcCalculateTokenContinuationBytes(3u,&count) == SPARK_STATUS_OK && count == compact_bytes);
	assert(compact_bytes - SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES == 3u * (uint32_t)sizeof(SparkModelResidentIpcContinueToken));
	assert(full_bytes - SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES == 3u * (SPARK_MODEL_SERVING_LANE_BYTES + 2u * (uint32_t)sizeof(uint32_t) + 2u * (uint32_t)sizeof(uint64_t)));
	assert(SparkModelResidentIpcTokenContinuationTokens(compact,compact_bytes,&tokens_view,&count) == SPARK_STATUS_OK && count == 3u &&
		tokens_view[2].input_token_id == 402u && tokens_view[1].resident_sequence_slot == 7u && tokens_view[0].sequence_position == 90u);
	assert(SparkModelResidentIpcExpandTokenContinuation(compact,compact_bytes,previous,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_OK);
	assert(expanded_bytes == full_bytes);
	assert(memcmp(expanded,full,full_bytes) == 0);
	assert(SparkModelResidentIpcDecodeSubmission(expanded,expanded_bytes,&decoded) == SPARK_STATUS_OK && decoded.lanes[1].cache_prefix_token_count == 64u);
	assert(SparkModelResidentIpcStampSubmission(compact,compact_bytes,14u,20u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcStampSubmission(compact,compact_bytes,14u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcExpandTokenContinuation(compact,compact_bytes - 8u,previous,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentIpcExpandTokenContinuation(compact,compact_bytes,previous,expanded,full_bytes - 1u,&expanded_bytes) == SPARK_STATUS_CAPACITY_EXCEEDED);
	((SparkModelResidentIpcSubmit *)compact)->flags = SPARK_MODEL_SERVING_SUBMISSION_FLAG_VERIFY;
	assert(SparkModelResidentIpcTokenContinuationTokens(compact,compact_bytes,&tokens_view,&count) == SPARK_STATUS_SCHEMA_ERROR);
	((SparkModelResidentIpcSubmit *)compact)->flags = 0u;
	((SparkModelResidentIpcSubmit *)compact)->row_count = 2u;
	assert(SparkModelResidentIpcTokenContinuationTokens(compact,compact_bytes,&tokens_view,&count) == SPARK_STATUS_SCHEMA_ERROR);
	((SparkModelResidentIpcSubmit *)compact)->row_count = 3u;
	((SparkModelResidentIpcSubmit *)compact)->client_generation = 0u;
	assert(SparkModelResidentIpcTokenContinuationTokens(compact,compact_bytes,&tokens_view,&count) == SPARK_STATUS_SCHEMA_ERROR);
	((SparkModelResidentIpcSubmit *)compact)->client_generation = 20u;
	((SparkModelResidentIpcSubmit *)compact)->model_extension_offset = 0u;
	assert(SparkModelResidentIpcTokenContinuationTokens(compact,compact_bytes,&tokens_view,&count) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentIpcDecodeSubmission(compact,compact_bytes,&decoded) == SPARK_STATUS_SCHEMA_ERROR);
	submission.flags = SPARK_MODEL_SERVING_SUBMISSION_FLAG_VERIFY;
	assert(SparkModelResidentIpcEncodeTokenContinuation(&submission,13u,19u,compact,sizeof(compact),&compact_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	submission.flags = 0u;
	submission.row_count = 2u;
	assert(SparkModelResidentIpcEncodeTokenContinuation(&submission,13u,19u,compact,sizeof(compact),&compact_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
}

#define TEST_SPAN_CHUNK_ROWS 16384u

static void TestSpanBuild(SparkModelServingSubmission *submission,SparkModelServingLane *lanes,const uint32_t *lane_rows,uint32_t lane_count,uint32_t *tokens,uint32_t *row_lanes,uint64_t *positions,uint64_t *sequences)
{
	uint32_t lane,row,step;
	memset(submission,0,sizeof(*submission));
	memset(lanes,0,lane_count * sizeof(lanes[0]));
	row = 0u;
	for (lane=0u; lane<lane_count; lane++)
	{
		lanes[lane].request_id = 41u + lane;
		lanes[lane].request_generation = 2u;
		lanes[lane].step_generation = 90u;
		lanes[lane].sequence_id = 51u + lane;
		lanes[lane].sequence_position = 1000u * (lane + 1u);
		lanes[lane].context_token_count = 1000u * (lane + 1u) + lane_rows[lane];
		lanes[lane].state_target_token_count = lanes[lane].context_token_count;
		lanes[lane].resident_sequence_slot = 3u + lane;
		lanes[lane].input_token_id = 700u + lane;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PUBLISH;
		lanes[lane].cache_publish_token_count = lanes[lane].context_token_count;
		memset(&lanes[lane].cache_publish_identity,0x20 + (int)lane,sizeof(lanes[lane].cache_publish_identity));
		for (step=0u; step<lane_rows[lane]; step++,row++)
		{
			tokens[row] = 5000u + row * 7u;
			row_lanes[row] = lane;
			positions[row] = lanes[lane].sequence_position + step;
			sequences[row] = lanes[lane].sequence_id;
		}
	}
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission->submission_id = 80u;
	submission->request_id = 80u;
	submission->sequence_id = 80u;
	submission->control_generation = 3u;
	submission->transaction_id = 80u;
	submission->dispatch_generation = 80u;
	submission->request_generation = 1u;
	submission->step_generation = 90u;
	submission->priority = 1u;
	submission->tokens_per_sequence = 1u;
	submission->residency.word0 = 80u;
	submission->residency.owner = 1u;
	submission->active_sequence_count = lane_count;
	submission->lane_count = lane_count;
	submission->row_count = row;
	submission->token_count = row;
	submission->new_token_count = row;
	submission->lanes = lanes;
	submission->token_ids = tokens;
	submission->row_lane_indices = row_lanes;
	submission->row_positions = positions;
	submission->row_sequence_ids = sequences;
}

static void TestSpansRoundTrip(const SparkModelServingSubmission *submission,uint64_t client_generation,uint8_t *full,uint8_t *compact,uint8_t *expanded,uint32_t capacity,uint32_t *full_bytes,uint32_t *compact_bytes)
{
	SparkModelServingSubmission decoded;
	uint32_t expanded_bytes,span_bytes;
	if ( client_generation == 0u )
	{
		assert(SparkModelResidentIpcEncodePreparation(submission,21u,full,capacity,full_bytes) == SPARK_STATUS_OK);
		assert(SparkModelResidentIpcEncodeSpanPreparation(submission,21u,compact,capacity,compact_bytes) == SPARK_STATUS_OK);
		assert(((const SparkModelResidentIpcHeader *)compact)->kind == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE_SPANS);
	}
	else
	{
		assert(SparkModelResidentIpcEncodeContinuation(submission,21u,client_generation,full,capacity,full_bytes) == SPARK_STATUS_OK);
		assert(SparkModelResidentIpcEncodeSpanContinuation(submission,21u,client_generation,compact,capacity,compact_bytes) == SPARK_STATUS_OK);
		assert(((const SparkModelResidentIpcHeader *)compact)->kind == SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE_SPANS);
	}
	assert(SparkModelResidentIpcCalculateSpanBytes(submission->lane_count,submission->row_count,submission->cache_block_identity_count,submission->model_extension_bytes,&span_bytes) == SPARK_STATUS_OK);
	assert(span_bytes == *compact_bytes && *compact_bytes < *full_bytes);
	assert(SparkModelResidentIpcDecodeSubmission(compact,*compact_bytes,&decoded) == SPARK_STATUS_SCHEMA_ERROR);
	memset(expanded,0xa5,capacity);
	assert(SparkModelResidentIpcExpandSpans(compact,*compact_bytes,expanded,capacity,&expanded_bytes) == SPARK_STATUS_OK);
	assert(expanded_bytes == *full_bytes && memcmp(expanded,full,*full_bytes) == 0);
	assert(SparkModelResidentIpcDecodeSubmission(expanded,expanded_bytes,&decoded) == SPARK_STATUS_OK);
	assert(decoded.row_count == submission->row_count && decoded.lane_count == submission->lane_count);
	for (uint32_t row=0u; row<submission->row_count; row++)
		assert(decoded.token_ids[row] == submission->token_ids[row] && decoded.row_lane_indices[row] == submission->row_lane_indices[row] &&
			decoded.row_positions[row] == submission->row_positions[row] && decoded.row_sequence_ids[row] == submission->row_sequence_ids[row]);
}

static void TestSpansExpandToTheFullForm(void)
{
	static uint8_t full[65536],compact[65536],expanded[65536];
	static uint32_t tokens[64],row_lanes[64];
	static uint64_t positions[64],sequences[64];
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[4];
	SparkModelServingCacheIdentity identities[3];
	const uint32_t lane_rows[4] = {5u,0u,9u,1u};
	const uint8_t extension[5] = {1u,2u,3u,4u,5u};
	uint32_t compact_bytes,full_bytes;
	TestSpanBuild(&submission,lanes,lane_rows,4u,tokens,row_lanes,positions,sequences);
	memset(identities,0x6b,sizeof(identities));
	identities[2].sha256[31] = 0x11u;
	submission.cache_block_identity_count = 3u;
	submission.cache_block_identities = identities;
	submission.model_extension_kind = 4u;
	submission.model_extension_bytes = sizeof(extension);
	submission.model_extension = extension;
	assert(submission.row_count == 15u && SparkModelResidentIpcRowsAreLaneSpans(&submission) == 1u);
	TestSpansRoundTrip(&submission,0u,full,compact,expanded,sizeof(full),&full_bytes,&compact_bytes);
	assert(compact_bytes == SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES + 4u * SPARK_MODEL_SERVING_LANE_BYTES + 3u * (uint32_t)sizeof(SparkModelServingCacheIdentity) + 4u * 4u + 15u * 4u + 5u);
	assert(full_bytes - compact_bytes == 15u * 20u - 4u * 4u);
	TestSpansRoundTrip(&submission,6u,full,compact,expanded,sizeof(full),&full_bytes,&compact_bytes);
	submission.cache_block_identity_count = 0u;
	submission.cache_block_identities = 0;
	submission.model_extension_kind = 0u;
	submission.model_extension_bytes = 0u;
	submission.model_extension = 0;
	TestSpansRoundTrip(&submission,0u,full,compact,expanded,sizeof(full),&full_bytes,&compact_bytes);
}

static void TestSixteenThousandRowChunkCarriesItsTokens(void)
{
	static uint8_t full[400000],compact[400000],expanded[400000];
	static uint32_t tokens[TEST_SPAN_CHUNK_ROWS],row_lanes[TEST_SPAN_CHUNK_ROWS];
	static uint64_t positions[TEST_SPAN_CHUNK_ROWS],sequences[TEST_SPAN_CHUNK_ROWS];
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	SparkModelServingCacheIdentity identities[16];
	const uint32_t lane_rows[1] = {TEST_SPAN_CHUNK_ROWS};
	uint32_t compact_bytes,full_bytes;
	TestSpanBuild(&submission,&lane,lane_rows,1u,tokens,row_lanes,positions,sequences);
	lane.sequence_position = 131072u;
	for (uint32_t row=0u; row<TEST_SPAN_CHUNK_ROWS; row++)
		positions[row] = 131072u + row;
	memset(identities,0x3c,sizeof(identities));
	submission.cache_block_identity_count = 16u;
	submission.cache_block_identities = identities;
	TestSpansRoundTrip(&submission,0u,full,compact,expanded,sizeof(full),&full_bytes,&compact_bytes);
	TestSpansRoundTrip(&submission,11u,full,compact,expanded,sizeof(full),&full_bytes,&compact_bytes);
	assert(full_bytes == SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES + SPARK_MODEL_SERVING_LANE_BYTES + TEST_SPAN_CHUNK_ROWS * 24u + 16u * (uint32_t)sizeof(SparkModelServingCacheIdentity));
	assert(compact_bytes == SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES + SPARK_MODEL_SERVING_LANE_BYTES + 16u * (uint32_t)sizeof(SparkModelServingCacheIdentity) + 4u + TEST_SPAN_CHUNK_ROWS * 4u);
	assert(compact_bytes * 5u < full_bytes);
}

static void TestRowsThatAreNotLaneSpansTravelWhole(void)
{
	static uint8_t message[16384];
	uint32_t tokens[8],row_lanes[8],message_bytes;
	uint64_t positions[8],sequences[8];
	uint32_t parents[8] = {0u},slots[8] = {0u};
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	const uint32_t lane_rows[2] = {3u,3u};
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 1u);
	row_lanes[2] = 1u;
	row_lanes[3] = 0u;
	positions[2] = lanes[1].sequence_position;
	sequences[2] = lanes[1].sequence_id;
	positions[3] = lanes[0].sequence_position + 2u;
	sequences[3] = lanes[0].sequence_id;
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 0u);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,5u,message,sizeof(message),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	positions[4] += 1u;
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 0u);
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	positions[0] += 1u;
	positions[1] += 1u;
	positions[2] += 1u;
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 0u);
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	sequences[5] = 99u;
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 0u);
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	row_lanes[5] = 2u;
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 0u);
	assert(SparkModelResidentIpcEncodeSpanContinuation(&submission,5u,3u,message,sizeof(message),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	submission.flags = SPARK_MODEL_SERVING_SUBMISSION_FLAG_TREE;
	submission.row_parent_rows = parents;
	submission.row_slots = slots;
	assert(SparkModelResidentIpcRowsAreLaneSpans(&submission) == 0u);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,5u,message,sizeof(message),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	assert(SparkModelResidentIpcEncodeSpanContinuation(&submission,5u,0u,message,sizeof(message),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,0u,message,sizeof(message),&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,5u,message,SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES,&message_bytes) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,5u,message,sizeof(message),&message_bytes) == SPARK_STATUS_OK);
}

static void TestMalformedSpansAreRefused(void)
{
	static uint8_t full[16384],compact[16384],expanded[16384],stamped[16384];
	SparkModelResidentIpcSubmit *wire = (SparkModelResidentIpcSubmit *)compact;
	uint32_t tokens[8],row_lanes[8],compact_bytes,expanded_bytes,full_bytes,*lane_rows_wire;
	uint64_t positions[8],sequences[8];
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	const uint32_t lane_rows[2] = {2u,4u};
	TestSpanBuild(&submission,lanes,lane_rows,2u,tokens,row_lanes,positions,sequences);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,5u,compact,sizeof(compact),&compact_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,expanded_bytes - 1u,&expanded_bytes) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes - 4u,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	lane_rows_wire = (uint32_t *)(compact + wire->row_lane_indices_offset);
	lane_rows_wire[0] = 3u;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	lane_rows_wire[1] = 3u;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_OK);
	lane_rows_wire[0] = 2u;
	lane_rows_wire[1] = 4u;
	wire->token_ids_offset += 4u;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->token_ids_offset -= 4u;
	wire->row_positions_offset = SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->row_positions_offset = compact_bytes;
	wire->client_generation = 4u;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->client_generation = 0u;
	wire->flags = SPARK_MODEL_SERVING_SUBMISSION_FLAG_TREE;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->flags = 0u;
	wire->token_count = wire->row_count + 1u;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->token_count = wire->row_count;
	wire->header.kind = SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->header.kind = SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE_SPANS;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
	wire->header.kind = SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE_SPANS;
	wire->header.abi_version -= 1u;
	assert(SparkModelResidentIpcExpandSpans(compact,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_ABI_MISMATCH);
	wire->header.abi_version += 1u;
	assert(SparkModelResidentIpcStampSubmission(compact,compact_bytes,17u,9u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcEncodeSpanPreparation(&submission,1u,stamped,sizeof(stamped),&compact_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcStampSubmission(stamped,compact_bytes,17u,0u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodePreparation(&submission,17u,full,sizeof(full),&full_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcExpandSpans(stamped,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_OK);
	assert(expanded_bytes == full_bytes && memcmp(expanded,full,full_bytes) == 0);
	assert(SparkModelResidentIpcEncodeSpanContinuation(&submission,1u,1u,stamped,sizeof(stamped),&compact_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcStampSubmission(stamped,compact_bytes,18u,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcStampSubmission(stamped,compact_bytes,18u,23u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcEncodeContinuation(&submission,18u,23u,full,sizeof(full),&full_bytes) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcExpandSpans(stamped,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_OK);
	assert(expanded_bytes == full_bytes && memcmp(expanded,full,full_bytes) == 0);
	((SparkModelResidentIpcSubmit *)stamped)->client_generation = 0u;
	assert(SparkModelResidentIpcExpandSpans(stamped,compact_bytes,expanded,sizeof(expanded),&expanded_bytes) == SPARK_STATUS_SCHEMA_ERROR);
}

static void TestSubmissionKinds(void)
{
	const uint32_t submissions[] = {SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE,SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE,SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE_TOKENS,
		SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE_SPANS,SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE_SPANS};
	const uint32_t others[] = {0u,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO_ACK,SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT_RESULT,
		SPARK_MODEL_RESIDENT_IPC_KIND_COMPLETION,SPARK_MODEL_RESIDENT_IPC_KIND_DECISION,SPARK_MODEL_RESIDENT_IPC_KIND_DECISION_RESULT,
		SPARK_MODEL_RESIDENT_IPC_KIND_STATUS_REQUEST,SPARK_MODEL_RESIDENT_IPC_KIND_STATUS_REPORT,SPARK_MODEL_RESIDENT_IPC_KIND_CACHE_HINT,16u};
	for (uint32_t index=0u; index<sizeof(submissions) / sizeof(submissions[0]); index++)
		assert(SparkModelResidentIpcIsSubmissionKind(submissions[index]) == 1u &&
			SparkModelResidentIpcIsPreparationKind(submissions[index]) == (submissions[index] == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE || submissions[index] == SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE_SPANS ? 1u : 0u));
	for (uint32_t index=0u; index<sizeof(others) / sizeof(others[0]); index++)
		assert(SparkModelResidentIpcIsSubmissionKind(others[index]) == 0u && SparkModelResidentIpcIsPreparationKind(others[index]) == 0u);
}

static void TestIndependentPrefillMessageCapacity(void)
{
	uint32_t message_bytes;
	assert(SparkModelResidentIpcCalculateSubmitBytes(SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_CACHE_BLOCK_IDENTITY_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,SPARK_MODEL_SERVING_ADAPTER_MAX_TREE_ROW_COUNT,&message_bytes) == SPARK_STATUS_OK);
	assert(message_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MODEL_SERVING_LANE_BYTES);
	assert(SparkModelResidentIpcCalculateSubmitBytes(SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT + 1u,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT,0u,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,0u,&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcCalculateSubmitBytes(SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT,SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT + 1u,0u,SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES,0u,&message_bytes) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void TestCacheProbeWire(void)
{
	SparkModelResidentIpcCacheProbe probe;
	SparkModelResidentIpcCacheProbeReport report,bad;
	SparkModelServingCacheIdentity identity;
	SparkModelServingCacheProbe held = {4096u,2048u,4096u,4032u,4096u};
	memset(&identity,0x5a,sizeof(identity));
	assert(SparkModelResidentIpcInitializeCacheProbe(&probe,77u,&identity,4096u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateCacheProbe(&probe,sizeof(probe)) == SPARK_STATUS_OK && probe.header.kind == SPARK_MODEL_RESIDENT_IPC_KIND_CACHE_PROBE);
	assert(SparkModelResidentIpcInitializeCacheProbe(&probe,77u,&identity,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkModelResidentIpcInitializeCacheProbe(&probe,77u,&identity,4096u) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateCacheProbe(&probe,sizeof(probe) - 1u) != SPARK_STATUS_OK);
	assert(SparkModelResidentIpcInitializeCacheProbeReport(&report,&probe,SPARK_STATUS_OK,5u,&held) == SPARK_STATUS_OK);
	assert(SparkModelResidentIpcValidateCacheProbeReport(&report,sizeof(report),77u,5u) == SPARK_STATUS_OK);
	assert(report.token_count == 4096u && memcmp(&report.identity,&identity,sizeof(identity)) == 0 && report.probe.resident_token_count == 2048u &&
		report.probe.state_token_count == 4032u);
	assert(SparkModelResidentIpcValidateCacheProbeReport(&report,sizeof(report),78u,5u) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentIpcValidateCacheProbeReport(&report,sizeof(report),77u,6u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = report;
	bad.probe.saved_token_count = 4097u;
	assert(SparkModelResidentIpcValidateCacheProbeReport(&bad,sizeof(bad),77u,5u) == SPARK_STATUS_SCHEMA_ERROR);
	bad = report;
	bad.probe.read_token_count = 4097u;
	assert(SparkModelResidentIpcValidateCacheProbeReport(&bad,sizeof(bad),77u,5u) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentIpcIsSubmissionKind(SPARK_MODEL_RESIDENT_IPC_KIND_CACHE_PROBE) == 0u &&
		SparkModelResidentIpcIsSubmissionKind(SPARK_MODEL_RESIDENT_IPC_KIND_CACHE_PROBE_REPORT) == 0u);
}

int main(void)
{
	TestCacheProbeWire();
	TestHello();
	TestSubmitResult();
	TestDecision();
	TestSubmissionRoundTrip();
	TestCompletionRoundTrip();
	TestCompletionDecodeReplacesHeldCompletion();
	TestStampedSubmissionMatchesDirectEncoding();
	TestTokenContinuationCarriesEveryLane();
	TestSpansExpandToTheFullForm();
	TestSixteenThousandRowChunkCarriesItsTokens();
	TestRowsThatAreNotLaneSpansTravelWhole();
	TestMalformedSpansAreRefused();
	TestSubmissionKinds();
	TestIndependentPrefillMessageCapacity();
	TestContinuationLease();
	return(0);
}
