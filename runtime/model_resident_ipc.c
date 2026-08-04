#include "sparkpipe/spark_model_resident_ipc.h"

#include <string.h>

static SparkStatus SparkModelResidentIpcAddBytes(
	uint32_t *total,
	uint32_t count,
	uint32_t element_bytes)
{
	uint64_t next;
	next = (uint64_t)*total + ((uint64_t)count * element_bytes);
	if ( next > SPARK_MODEL_RESIDENT_IPC_MAX_MESSAGE_BYTES )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*total = (uint32_t)next;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelResidentIpcCopyText(
	char *destination,
	uint32_t destination_bytes,
	const char *source)
{
	uint32_t source_bytes;
	if ( destination == 0 || destination_bytes == 0u || source == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	source_bytes = (uint32_t)strlen(source);
	if ( source_bytes + 1u > destination_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(destination,source,source_bytes + 1u);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelResidentIpcTextMatches(
	const char *wire_text,
	uint32_t wire_bytes,
	const char *expected)
{
	const void *terminator;
	if ( wire_text == 0 || expected == 0 )
		return(0u);
	terminator = memchr(wire_text,'\0',wire_bytes);
	return(terminator != 0 && strcmp(wire_text,expected) == 0 ? 1u : 0u);
}

static void SparkModelResidentIpcInitializeHeader(
	SparkModelResidentIpcHeader *header,
	uint32_t kind,
	uint32_t descriptor_bytes,
	uint32_t message_bytes,
	uint64_t message_id)
{
	memset(header,0,sizeof(*header));
	header->magic = SPARK_MODEL_RESIDENT_IPC_MAGIC;
	header->abi_version = SPARK_MODEL_RESIDENT_IPC_ABI_VERSION;
	header->kind = kind;
	header->descriptor_bytes = descriptor_bytes;
	header->message_bytes = message_bytes;
	header->message_id = message_id;
}

SparkStatus SparkModelResidentIpcValidateHeader(
	const SparkModelResidentIpcHeader *header,
	uint32_t message_bytes,
	uint32_t expected_kind,
	uint32_t expected_descriptor_bytes)
{
	if ( header == 0 || message_bytes < SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES || message_bytes > SPARK_MODEL_RESIDENT_IPC_MAX_MESSAGE_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_MODEL_RESIDENT_IPC_MAGIC || header->abi_version != SPARK_MODEL_RESIDENT_IPC_ABI_VERSION || header->descriptor_bytes != expected_descriptor_bytes )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( header->kind != expected_kind || header->message_bytes != message_bytes || header->reserved0 != 0u || header->message_id == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcInitializeHello(
	SparkModelResidentIpcHello *hello,
	uint64_t message_id,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor)
{
	SparkStatus status;
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( hello == 0 || message_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(hello,0,sizeof(*hello));
	SparkModelResidentIpcInitializeHeader(&hello->header,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO,SPARK_MODEL_RESIDENT_IPC_HELLO_BYTES,SPARK_MODEL_RESIDENT_IPC_HELLO_BYTES,message_id);
	hello->rank_index = rank_index;
	hello->stage_index = stage_index;
	status = SparkModelResidentIpcCopyText(hello->adapter_id,sizeof(hello->adapter_id),descriptor->adapter_id);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcCopyText(hello->model_id,sizeof(hello->model_id),descriptor->model_id);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcCopyText(hello->model_revision,sizeof(hello->model_revision),descriptor->model_revision);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcCopyText(hello->artifact_sha256,sizeof(hello->artifact_sha256),descriptor->artifact_sha256);
	return(status);
}

SparkStatus SparkModelResidentIpcValidateHello(
	const SparkModelResidentIpcHello *hello,
	uint32_t message_bytes,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor)
{
	SparkStatus status;
	status = SparkModelResidentIpcValidateHeader(hello != 0 ? &hello->header : 0,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO,SPARK_MODEL_RESIDENT_IPC_HELLO_BYTES);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkModelServingAdapterValidateDescriptor(descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( hello->rank_index != rank_index || hello->stage_index != stage_index || SparkModelResidentIpcTextMatches(hello->adapter_id,sizeof(hello->adapter_id),descriptor->adapter_id) == 0u || SparkModelResidentIpcTextMatches(hello->model_id,sizeof(hello->model_id),descriptor->model_id) == 0u || SparkModelResidentIpcTextMatches(hello->model_revision,sizeof(hello->model_revision),descriptor->model_revision) == 0u || SparkModelResidentIpcTextMatches(hello->artifact_sha256,sizeof(hello->artifact_sha256),descriptor->artifact_sha256) == 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcInitializeHelloAck(
	SparkModelResidentIpcHelloAck *ack,
	uint64_t message_id,
	SparkStatus status,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits)
{
	SparkStatus copy_status;
	copy_status = SparkModelServingAdapterValidateRuntimeLimits(descriptor,runtime_limits);
	if ( ack == 0 || message_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( copy_status != SPARK_STATUS_OK )
		return(copy_status);
	memset(ack,0,sizeof(*ack));
	SparkModelResidentIpcInitializeHeader(&ack->header,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO_ACK,SPARK_MODEL_RESIDENT_IPC_HELLO_ACK_BYTES,SPARK_MODEL_RESIDENT_IPC_HELLO_ACK_BYTES,message_id);
	ack->status = (uint32_t)status;
	ack->rank_index = rank_index;
	ack->stage_index = stage_index;
	ack->adapter_capability_flags = descriptor->capability_flags;
	ack->max_inflight_submission_count = runtime_limits->max_inflight_submission_count;
	ack->max_active_sequence_count = runtime_limits->max_active_sequence_count;
	ack->max_input_row_count = runtime_limits->max_input_row_count;
	ack->resident_sequence_capacity = runtime_limits->resident_sequence_capacity;
	ack->boundary_format = descriptor->boundary_format;
	ack->boundary_element_count = descriptor->boundary_element_count;
	ack->boundary_element_bytes = descriptor->boundary_element_bytes;
	copy_status = SparkModelResidentIpcCopyText(ack->adapter_id,sizeof(ack->adapter_id),descriptor->adapter_id);
	if ( copy_status == SPARK_STATUS_OK )
		copy_status = SparkModelResidentIpcCopyText(ack->model_id,sizeof(ack->model_id),descriptor->model_id);
	if ( copy_status == SPARK_STATUS_OK )
		copy_status = SparkModelResidentIpcCopyText(ack->model_revision,sizeof(ack->model_revision),descriptor->model_revision);
	if ( copy_status == SPARK_STATUS_OK )
		copy_status = SparkModelResidentIpcCopyText(ack->artifact_sha256,sizeof(ack->artifact_sha256),descriptor->artifact_sha256);
	return(copy_status);
}

SparkStatus SparkModelResidentIpcValidateHelloAck(
	const SparkModelResidentIpcHelloAck *ack,
	uint32_t message_bytes,
	uint64_t message_id,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits)
{
	SparkStatus status;
	status = SparkModelResidentIpcValidateHeader(ack != 0 ? &ack->header : 0,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_HELLO_ACK,SPARK_MODEL_RESIDENT_IPC_HELLO_ACK_BYTES);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkModelServingAdapterValidateRuntimeLimits(descriptor,runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( ack->status > SPARK_STATUS_UNSUPPORTED || ack->header.message_id != message_id || ack->rank_index != rank_index || ack->stage_index != stage_index || ack->adapter_capability_flags != descriptor->capability_flags || ack->max_inflight_submission_count != runtime_limits->max_inflight_submission_count || ack->max_active_sequence_count != runtime_limits->max_active_sequence_count || ack->max_input_row_count != runtime_limits->max_input_row_count || ack->resident_sequence_capacity != runtime_limits->resident_sequence_capacity || ack->boundary_format != descriptor->boundary_format || ack->boundary_element_count != descriptor->boundary_element_count || ack->boundary_element_bytes != descriptor->boundary_element_bytes )
		return(SPARK_STATUS_TARGET_MISMATCH);
	if ( SparkModelResidentIpcTextMatches(ack->adapter_id,sizeof(ack->adapter_id),descriptor->adapter_id) == 0u || SparkModelResidentIpcTextMatches(ack->model_id,sizeof(ack->model_id),descriptor->model_id) == 0u || SparkModelResidentIpcTextMatches(ack->model_revision,sizeof(ack->model_revision),descriptor->model_revision) == 0u || SparkModelResidentIpcTextMatches(ack->artifact_sha256,sizeof(ack->artifact_sha256),descriptor->artifact_sha256) == 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcInitializeSubmitResult(
	SparkModelResidentIpcSubmitResult *result,
	uint64_t message_id,
	uint64_t submission_id,
	SparkStatus status)
{
	if ( result == 0 || message_id == 0u || submission_id == 0u || status > SPARK_STATUS_UNSUPPORTED )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(result,0,sizeof(*result));
	SparkModelResidentIpcInitializeHeader(&result->header,SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT_RESULT,SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES,SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES,message_id);
	result->submission_id = submission_id;
	result->status = (uint32_t)status;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcValidateSubmitResult(
	const SparkModelResidentIpcSubmitResult *result,
	uint32_t message_bytes,
	uint64_t message_id,
	uint64_t submission_id)
{
	SparkStatus status;
	status = SparkModelResidentIpcValidateHeader(result != 0 ? &result->header : 0,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT_RESULT,SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( result->header.message_id != message_id || result->submission_id != submission_id || result->status > SPARK_STATUS_UNSUPPORTED || result->reserved0 != 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcInitializeDecision(
	SparkModelResidentIpcDecision *decision,
	uint64_t message_id,
	uint32_t decision_kind,
	const SparkModelServingSubmission *submission)
{
	if ( decision == 0 || submission == 0 || message_id == 0u || (decision_kind != SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT && decision_kind != SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT) || submission->submission_id == 0u || submission->control_generation == 0u || submission->transaction_id == 0u || submission->dispatch_generation == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(decision,0,sizeof(*decision));
	SparkModelResidentIpcInitializeHeader(&decision->header,SPARK_MODEL_RESIDENT_IPC_KIND_DECISION,SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES,SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES,message_id);
	decision->decision = decision_kind;
	decision->submission_id = submission->submission_id;
	decision->control_generation = submission->control_generation;
	decision->transaction_id = submission->transaction_id;
	decision->dispatch_generation = submission->dispatch_generation;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcValidateDecision(
	const SparkModelResidentIpcDecision *decision,
	uint32_t message_bytes)
{
	SparkStatus status;
	status = SparkModelResidentIpcValidateHeader(decision != 0 ? &decision->header : 0,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_DECISION,SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( (decision->decision != SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT && decision->decision != SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT) || decision->reserved0 != 0u || decision->submission_id == 0u || decision->control_generation == 0u || decision->transaction_id == 0u || decision->dispatch_generation == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcInitializeDecisionResult(
	SparkModelResidentIpcDecisionResult *result,
	const SparkModelResidentIpcDecision *decision,
	SparkStatus status)
{
	SparkStatus validation_status;
	validation_status = SparkModelResidentIpcValidateDecision(decision,SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES);
	if ( result == 0 || validation_status != SPARK_STATUS_OK || status > SPARK_STATUS_UNSUPPORTED )
		return(validation_status != SPARK_STATUS_OK ? validation_status : SPARK_STATUS_INVALID_ARGUMENT);
	memset(result,0,sizeof(*result));
	SparkModelResidentIpcInitializeHeader(&result->header,SPARK_MODEL_RESIDENT_IPC_KIND_DECISION_RESULT,SPARK_MODEL_RESIDENT_IPC_DECISION_RESULT_BYTES,SPARK_MODEL_RESIDENT_IPC_DECISION_RESULT_BYTES,decision->header.message_id);
	result->decision = decision->decision;
	result->status = (uint32_t)status;
	result->submission_id = decision->submission_id;
	result->control_generation = decision->control_generation;
	result->transaction_id = decision->transaction_id;
	result->dispatch_generation = decision->dispatch_generation;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcValidateDecisionResult(
	const SparkModelResidentIpcDecisionResult *result,
	uint32_t message_bytes,
	uint64_t message_id,
	uint32_t decision_kind,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	status = SparkModelResidentIpcValidateHeader(result != 0 ? &result->header : 0,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_DECISION_RESULT,SPARK_MODEL_RESIDENT_IPC_DECISION_RESULT_BYTES);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission == 0 || result->header.message_id != message_id || result->decision != decision_kind || result->status > SPARK_STATUS_UNSUPPORTED || result->submission_id != submission->submission_id || result->control_generation != submission->control_generation || result->transaction_id != submission->transaction_id || result->dispatch_generation != submission->dispatch_generation )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcCalculateSubmitBytes(
	uint32_t lane_count,
	uint32_t row_count,
	uint32_t model_extension_bytes,
	uint32_t *message_bytes_out)
{
	uint32_t total;
	SparkStatus status;
	if ( message_bytes_out == 0 || lane_count == 0u || lane_count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT || row_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT || model_extension_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	total = SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES;
	status = SparkModelResidentIpcAddBytes(&total,lane_count,SPARK_MODEL_SERVING_LANE_BYTES);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcAddBytes(&total,row_count,sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcAddBytes(&total,row_count,sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcAddBytes(&total,row_count,sizeof(uint64_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcAddBytes(&total,row_count,sizeof(uint64_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcAddBytes(&total,model_extension_bytes,sizeof(uint8_t));
	if ( status == SPARK_STATUS_OK )
		*message_bytes_out = total;
	return(status);
}

static void SparkModelResidentIpcCopySubmissionScalars(
	SparkModelResidentIpcSubmit *wire,
	const SparkModelServingSubmission *submission)
{
	wire->work_kind = submission->work_kind;
	wire->flags = submission->flags;
	wire->submission_id = submission->submission_id;
	wire->request_id = submission->request_id;
	wire->sequence_id = submission->sequence_id;
	wire->sequence_position = submission->sequence_position;
	wire->deadline_time_ns = submission->deadline_time_ns;
	wire->control_generation = submission->control_generation;
	wire->transaction_id = submission->transaction_id;
	wire->dispatch_generation = submission->dispatch_generation;
	wire->request_generation = submission->request_generation;
	wire->step_generation = submission->step_generation;
	wire->priority = submission->priority;
	wire->active_sequence_count = submission->active_sequence_count;
	wire->new_token_count = submission->new_token_count;
	wire->lane_count = submission->lane_count;
	wire->row_count = submission->row_count;
	wire->token_count = submission->token_count;
	wire->model_extension_kind = submission->model_extension_kind;
	wire->model_extension_bytes = submission->model_extension_bytes;
	wire->residency = submission->residency;
}

static SparkStatus SparkModelResidentIpcEncodeSubmissionKind(
	const SparkModelServingSubmission *submission,
	uint64_t message_id,
	uint32_t message_kind,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out)
{
	SparkModelResidentIpcSubmit *wire;
	uint8_t *bytes;
	uint32_t message_bytes,offset;
	SparkStatus status;
	if ( submission == 0 || message_id == 0u || message_buffer == 0 || message_bytes_out == 0 || (message_kind != SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT && message_kind != SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE) || submission->lanes == 0 || submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || (submission->row_count != 0u && (submission->token_ids == 0 || submission->row_lane_indices == 0 || submission->row_positions == 0 || submission->row_sequence_ids == 0)) || (submission->model_extension_bytes != 0u && submission->model_extension == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelResidentIpcCalculateSubmitBytes(submission->lane_count,submission->row_count,submission->model_extension_bytes,&message_bytes);
	if ( status != SPARK_STATUS_OK || message_capacity < message_bytes )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(message_buffer,0,message_bytes);
	wire = (SparkModelResidentIpcSubmit *)message_buffer;
	bytes = (uint8_t *)message_buffer;
	SparkModelResidentIpcInitializeHeader(&wire->header,message_kind,SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES,message_bytes,message_id);
	SparkModelResidentIpcCopySubmissionScalars(wire,submission);
	offset = SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES;
	wire->lanes_offset = offset;
	memcpy(bytes + offset,submission->lanes,submission->lane_count * SPARK_MODEL_SERVING_LANE_BYTES);
	offset += submission->lane_count * SPARK_MODEL_SERVING_LANE_BYTES;
	wire->token_ids_offset = offset;
	if ( submission->row_count != 0u )
		memcpy(bytes + offset,submission->token_ids,submission->row_count * sizeof(uint32_t));
	offset += submission->row_count * sizeof(uint32_t);
	wire->row_lane_indices_offset = offset;
	if ( submission->row_count != 0u )
		memcpy(bytes + offset,submission->row_lane_indices,submission->row_count * sizeof(uint32_t));
	offset += submission->row_count * sizeof(uint32_t);
	wire->row_positions_offset = offset;
	if ( submission->row_count != 0u )
		memcpy(bytes + offset,submission->row_positions,submission->row_count * sizeof(uint64_t));
	offset += submission->row_count * sizeof(uint64_t);
	wire->row_sequence_ids_offset = offset;
	if ( submission->row_count != 0u )
		memcpy(bytes + offset,submission->row_sequence_ids,submission->row_count * sizeof(uint64_t));
	offset += submission->row_count * sizeof(uint64_t);
	wire->model_extension_offset = offset;
	if ( submission->model_extension_bytes != 0u )
		memcpy(bytes + offset,submission->model_extension,submission->model_extension_bytes);
	*message_bytes_out = message_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcEncodeSubmission(
	const SparkModelServingSubmission *submission,
	uint64_t message_id,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out)
{
	return(SparkModelResidentIpcEncodeSubmissionKind(submission,message_id,SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT,message_buffer,message_capacity,message_bytes_out));
}

SparkStatus SparkModelResidentIpcEncodePreparation(
	const SparkModelServingSubmission *submission,
	uint64_t message_id,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out)
{
	return(SparkModelResidentIpcEncodeSubmissionKind(submission,message_id,SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE,message_buffer,message_capacity,message_bytes_out));
}

static SparkStatus SparkModelResidentIpcValidateSubmitLayout(
	const SparkModelResidentIpcSubmit *wire,
	uint32_t message_bytes)
{
	uint32_t expected,offset;
	SparkStatus status;
	status = SparkModelResidentIpcCalculateSubmitBytes(wire->lane_count,wire->row_count,wire->model_extension_bytes,&expected);
	if ( status != SPARK_STATUS_OK || expected != message_bytes || wire->token_count != wire->row_count || wire->reserved0 != 0u )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_SCHEMA_ERROR);
	offset = SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES;
	if ( wire->lanes_offset != offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	offset += wire->lane_count * SPARK_MODEL_SERVING_LANE_BYTES;
	if ( wire->token_ids_offset != offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	offset += wire->row_count * sizeof(uint32_t);
	if ( wire->row_lane_indices_offset != offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	offset += wire->row_count * sizeof(uint32_t);
	if ( wire->row_positions_offset != offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	offset += wire->row_count * sizeof(uint64_t);
	if ( wire->row_sequence_ids_offset != offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	offset += wire->row_count * sizeof(uint64_t);
	return(wire->model_extension_offset == offset ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

SparkStatus SparkModelResidentIpcDecodeSubmission(
	const void *message_buffer,
	uint32_t message_bytes,
	SparkModelServingSubmission *submission_out)
{
	const SparkModelResidentIpcSubmit *wire;
	const uint8_t *bytes;
	SparkStatus status;
	if ( message_buffer == 0 || submission_out == 0 || message_bytes < SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	wire = (const SparkModelResidentIpcSubmit *)message_buffer;
	if ( wire->header.kind != SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT && wire->header.kind != SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkModelResidentIpcValidateHeader(&wire->header,message_bytes,wire->header.kind,SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcValidateSubmitLayout(wire,message_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	bytes = (const uint8_t *)message_buffer;
	memset(submission_out,0,sizeof(*submission_out));
	submission_out->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission_out->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission_out->work_kind = wire->work_kind;
	submission_out->flags = wire->flags;
	submission_out->submission_id = wire->submission_id;
	submission_out->request_id = wire->request_id;
	submission_out->sequence_id = wire->sequence_id;
	submission_out->sequence_position = wire->sequence_position;
	submission_out->deadline_time_ns = wire->deadline_time_ns;
	submission_out->control_generation = wire->control_generation;
	submission_out->transaction_id = wire->transaction_id;
	submission_out->dispatch_generation = wire->dispatch_generation;
	submission_out->request_generation = wire->request_generation;
	submission_out->step_generation = wire->step_generation;
	submission_out->priority = wire->priority;
	submission_out->active_sequence_count = wire->active_sequence_count;
	submission_out->new_token_count = wire->new_token_count;
	submission_out->lane_count = wire->lane_count;
	submission_out->row_count = wire->row_count;
	submission_out->token_count = wire->token_count;
	submission_out->model_extension_kind = wire->model_extension_kind;
	submission_out->model_extension_bytes = wire->model_extension_bytes;
	submission_out->residency = wire->residency;
	submission_out->lanes = (const SparkModelServingLane *)(bytes + wire->lanes_offset);
	submission_out->token_ids = (const uint32_t *)(bytes + wire->token_ids_offset);
	submission_out->row_lane_indices = (const uint32_t *)(bytes + wire->row_lane_indices_offset);
	submission_out->row_positions = (const uint64_t *)(bytes + wire->row_positions_offset);
	submission_out->row_sequence_ids = (const uint64_t *)(bytes + wire->row_sequence_ids_offset);
	submission_out->model_extension = wire->model_extension_bytes != 0u ? bytes + wire->model_extension_offset : 0;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcCalculateCompletionBytes(
	uint32_t token_count,
	uint32_t model_extension_bytes,
	uint32_t *message_bytes_out)
{
	uint32_t total;
	SparkStatus status;
	if ( message_bytes_out == 0 || token_count > SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT || model_extension_bytes > SPARK_MODEL_SERVING_ADAPTER_MAX_EXTENSION_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	total = SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES;
	status = SparkModelResidentIpcAddBytes(&total,token_count,sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcAddBytes(&total,model_extension_bytes,sizeof(uint8_t));
	if ( status == SPARK_STATUS_OK )
		*message_bytes_out = total;
	return(status);
}

SparkStatus SparkModelResidentIpcEncodeCompletion(
	const SparkModelServingCompletion *completion,
	uint64_t message_id,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out)
{
	SparkModelResidentIpcCompletion *wire;
	uint8_t *bytes;
	uint32_t message_bytes;
	SparkStatus status;
	if ( completion == 0 || message_id == 0u || message_buffer == 0 || message_bytes_out == 0 || completion->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || completion->descriptor_bytes != SPARK_MODEL_SERVING_COMPLETION_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelResidentIpcCalculateCompletionBytes(completion->token_count,completion->model_extension_bytes,&message_bytes);
	if ( status != SPARK_STATUS_OK || message_capacity < message_bytes )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(message_buffer,0,message_bytes);
	wire = (SparkModelResidentIpcCompletion *)message_buffer;
	bytes = (uint8_t *)message_buffer;
	SparkModelResidentIpcInitializeHeader(&wire->header,SPARK_MODEL_RESIDENT_IPC_KIND_COMPLETION,SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES,message_bytes,message_id);
	wire->status = completion->status;
	wire->completion_flags = completion->completion_flags;
	wire->submission_id = completion->submission_id;
	wire->request_id = completion->request_id;
	wire->sequence_id = completion->sequence_id;
	wire->sequence_position = completion->sequence_position;
	wire->control_generation = completion->control_generation;
	wire->transaction_id = completion->transaction_id;
	wire->dispatch_generation = completion->dispatch_generation;
	wire->request_generation = completion->request_generation;
	wire->step_generation = completion->step_generation;
	wire->residency = completion->residency;
	wire->accepted_token_count = completion->accepted_token_count;
	wire->token_count = completion->token_count;
	wire->model_extension_kind = completion->model_extension_kind;
	wire->model_extension_bytes = completion->model_extension_bytes;
	wire->token_ids_offset = SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES;
	wire->model_extension_offset = wire->token_ids_offset + (completion->token_count * sizeof(uint32_t));
	wire->queue_delay_ns = completion->queue_delay_ns;
	wire->service_time_ns = completion->service_time_ns;
	wire->device_memcpy_bytes = completion->device_memcpy_bytes;
	wire->host_staging_bytes = completion->host_staging_bytes;
	memcpy(bytes + wire->token_ids_offset,completion->token_ids,completion->token_count * sizeof(uint32_t));
	if ( completion->model_extension_bytes != 0u )
		memcpy(bytes + wire->model_extension_offset,completion->model_extension,completion->model_extension_bytes);
	*message_bytes_out = message_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentIpcDecodeCompletion(
	const void *message_buffer,
	uint32_t message_bytes,
	SparkModelServingCompletion *completion_out)
{
	const SparkModelResidentIpcCompletion *wire;
	const uint8_t *bytes;
	uint32_t expected;
	SparkStatus status;
	if ( message_buffer == 0 || completion_out == 0 || message_bytes < SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	wire = (const SparkModelResidentIpcCompletion *)message_buffer;
	status = SparkModelResidentIpcValidateHeader(&wire->header,message_bytes,SPARK_MODEL_RESIDENT_IPC_KIND_COMPLETION,SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentIpcCalculateCompletionBytes(wire->token_count,wire->model_extension_bytes,&expected);
	if ( status != SPARK_STATUS_OK || expected != message_bytes || wire->token_ids_offset != SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES || wire->model_extension_offset != wire->token_ids_offset + (wire->token_count * sizeof(uint32_t)) )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_SCHEMA_ERROR);
	bytes = (const uint8_t *)message_buffer;
	memset(completion_out,0,sizeof(*completion_out));
	completion_out->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion_out->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion_out->status = wire->status;
	completion_out->completion_flags = wire->completion_flags;
	completion_out->submission_id = wire->submission_id;
	completion_out->request_id = wire->request_id;
	completion_out->sequence_id = wire->sequence_id;
	completion_out->sequence_position = wire->sequence_position;
	completion_out->control_generation = wire->control_generation;
	completion_out->transaction_id = wire->transaction_id;
	completion_out->dispatch_generation = wire->dispatch_generation;
	completion_out->request_generation = wire->request_generation;
	completion_out->step_generation = wire->step_generation;
	completion_out->residency = wire->residency;
	completion_out->accepted_token_count = wire->accepted_token_count;
	completion_out->token_count = wire->token_count;
	completion_out->model_extension_kind = wire->model_extension_kind;
	completion_out->model_extension_bytes = wire->model_extension_bytes;
	completion_out->queue_delay_ns = wire->queue_delay_ns;
	completion_out->service_time_ns = wire->service_time_ns;
	completion_out->device_memcpy_bytes = wire->device_memcpy_bytes;
	completion_out->host_staging_bytes = wire->host_staging_bytes;
	memcpy(completion_out->token_ids,bytes + wire->token_ids_offset,wire->token_count * sizeof(uint32_t));
	memcpy(completion_out->model_extension,bytes + wire->model_extension_offset,wire->model_extension_bytes);
	return(SPARK_STATUS_OK);
}
