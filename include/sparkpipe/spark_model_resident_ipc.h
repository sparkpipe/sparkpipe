#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_RESIDENT_IPC_ABI_VERSION 15u
#define SPARK_MODEL_RESIDENT_IPC_MAGIC UINT32_C(0x52444D53)
#define SPARK_MODEL_RESIDENT_IPC_MAX_MESSAGE_BYTES UINT32_C(2097152)
#define SPARK_MODEL_RESIDENT_IPC_ID_BYTES 128u
#define SPARK_MODEL_RESIDENT_IPC_REVISION_BYTES 128u
#define SPARK_MODEL_RESIDENT_IPC_SHA256_BYTES 65u

#define SPARK_MODEL_RESIDENT_IPC_KIND_HELLO 1u
#define SPARK_MODEL_RESIDENT_IPC_KIND_HELLO_ACK 2u
#define SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT 3u
#define SPARK_MODEL_RESIDENT_IPC_KIND_SUBMIT_RESULT 4u
#define SPARK_MODEL_RESIDENT_IPC_KIND_COMPLETION 5u
#define SPARK_MODEL_RESIDENT_IPC_KIND_PREPARE 6u
#define SPARK_MODEL_RESIDENT_IPC_KIND_DECISION 7u
#define SPARK_MODEL_RESIDENT_IPC_KIND_DECISION_RESULT 8u
#define SPARK_MODEL_RESIDENT_IPC_KIND_CONTINUE 9u

#define SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT 1u
#define SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT 2u

/* Distributed adapters must enter through PREPARE and a terminal decision. */
SparkStatus SparkModelResidentIpcValidateDirectSubmitDescriptor(
	const SparkModelServingAdapterDescriptor *descriptor);

typedef struct SparkModelResidentIpcHeader
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t kind;
	uint32_t descriptor_bytes;
	uint32_t message_bytes;
	uint32_t reserved0;
	uint64_t message_id;
} SparkModelResidentIpcHeader;

typedef struct SparkModelResidentIpcHello
{
	SparkModelResidentIpcHeader header;
	uint32_t rank_index;
	uint32_t stage_index;
	char adapter_id[SPARK_MODEL_RESIDENT_IPC_ID_BYTES];
	char model_id[SPARK_MODEL_RESIDENT_IPC_ID_BYTES];
	char model_revision[SPARK_MODEL_RESIDENT_IPC_REVISION_BYTES];
	char artifact_sha256[SPARK_MODEL_RESIDENT_IPC_SHA256_BYTES];
} SparkModelResidentIpcHello;

typedef struct SparkModelResidentIpcHelloAck
{
	SparkModelResidentIpcHeader header;
	uint64_t client_generation;
	uint32_t status;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t adapter_capability_flags;
	uint32_t max_inflight_submission_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint32_t boundary_format;
	uint32_t boundary_element_count;
	uint32_t boundary_element_bytes;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t kv_cache_codec;
	uint32_t input_sideband_kind;
	uint32_t input_sideband_bytes_per_sequence;
	uint32_t output_sideband_kind;
	uint32_t output_sideband_bytes_per_sequence;
	char adapter_id[SPARK_MODEL_RESIDENT_IPC_ID_BYTES];
	char model_id[SPARK_MODEL_RESIDENT_IPC_ID_BYTES];
	char model_revision[SPARK_MODEL_RESIDENT_IPC_REVISION_BYTES];
	char artifact_sha256[SPARK_MODEL_RESIDENT_IPC_SHA256_BYTES];
} SparkModelResidentIpcHelloAck;

typedef struct SparkModelResidentIpcSubmit
{
	SparkModelResidentIpcHeader header;
	uint32_t work_kind;
	uint32_t flags;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t deadline_time_ns;
	uint64_t client_generation;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	uint32_t priority;
	uint32_t active_sequence_count;
	uint32_t new_token_count;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t token_count;
	uint32_t tokens_per_sequence;
	uint32_t model_extension_kind;
	uint32_t model_extension_bytes;
	SparkModelDriverResidencyToken residency;
	uint32_t lanes_offset;
	uint32_t token_ids_offset;
	uint32_t row_lane_indices_offset;
	uint32_t row_positions_offset;
	uint32_t row_sequence_ids_offset;
	uint32_t model_extension_offset;
} SparkModelResidentIpcSubmit;

typedef struct SparkModelResidentIpcSubmitResult
{
	SparkModelResidentIpcHeader header;
	uint64_t submission_id;
	uint32_t status;
	uint32_t reserved0;
} SparkModelResidentIpcSubmitResult;

typedef struct SparkModelResidentIpcDecision
{
	SparkModelResidentIpcHeader header;
	uint32_t decision;
	uint32_t reserved0;
	uint64_t submission_id;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
} SparkModelResidentIpcDecision;

typedef struct SparkModelResidentIpcDecisionResult
{
	SparkModelResidentIpcHeader header;
	uint32_t decision;
	uint32_t status;
	uint64_t submission_id;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
} SparkModelResidentIpcDecisionResult;

typedef struct SparkModelResidentIpcCompletion
{
	SparkModelResidentIpcHeader header;
	uint32_t status;
	uint32_t completion_flags;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	SparkModelDriverResidencyToken residency;
	uint32_t accepted_token_count;
	uint32_t token_count;
	uint32_t tokens_per_sequence;
	uint32_t model_extension_kind;
	uint32_t model_extension_bytes;
	uint32_t token_ids_offset;
	uint32_t model_extension_offset;
	uint64_t queue_delay_ns;
	uint64_t service_time_ns;
	uint64_t device_memcpy_bytes;
	uint64_t host_staging_bytes;
} SparkModelResidentIpcCompletion;

#define SPARK_MODEL_RESIDENT_IPC_HEADER_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcHeader))
#define SPARK_MODEL_RESIDENT_IPC_HELLO_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcHello))
#define SPARK_MODEL_RESIDENT_IPC_HELLO_ACK_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcHelloAck))
#define SPARK_MODEL_RESIDENT_IPC_SUBMIT_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcSubmit))
#define SPARK_MODEL_RESIDENT_IPC_SUBMIT_RESULT_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcSubmitResult))
#define SPARK_MODEL_RESIDENT_IPC_DECISION_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcDecision))
#define SPARK_MODEL_RESIDENT_IPC_DECISION_RESULT_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcDecisionResult))
#define SPARK_MODEL_RESIDENT_IPC_COMPLETION_BYTES \
	((uint32_t)sizeof(SparkModelResidentIpcCompletion))

SparkStatus SparkModelResidentIpcValidateHeader(
	const SparkModelResidentIpcHeader *header,
	uint32_t message_bytes,
	uint32_t expected_kind,
	uint32_t expected_descriptor_bytes);
SparkStatus SparkModelResidentIpcInitializeHello(
	SparkModelResidentIpcHello *hello,
	uint64_t message_id,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor);
SparkStatus SparkModelResidentIpcValidateHello(
	const SparkModelResidentIpcHello *hello,
	uint32_t message_bytes,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor);
SparkStatus SparkModelResidentIpcInitializeHelloAck(
	SparkModelResidentIpcHelloAck *ack,
	uint64_t message_id,
	SparkStatus status,
	uint32_t rank_index,
	uint32_t stage_index,
	uint64_t client_generation,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits);
SparkStatus SparkModelResidentIpcValidateHelloAck(
	const SparkModelResidentIpcHelloAck *ack,
	uint32_t message_bytes,
	uint64_t message_id,
	uint32_t rank_index,
	uint32_t stage_index,
	const SparkModelServingAdapterDescriptor *descriptor,
	const SparkModelServingRuntimeLimits *runtime_limits);
SparkStatus SparkModelResidentIpcInitializeSubmitResult(
	SparkModelResidentIpcSubmitResult *result,
	uint64_t message_id,
	uint64_t submission_id,
	SparkStatus status);
SparkStatus SparkModelResidentIpcValidateSubmitResult(
	const SparkModelResidentIpcSubmitResult *result,
	uint32_t message_bytes,
	uint64_t message_id,
	uint64_t submission_id);
SparkStatus SparkModelResidentIpcInitializeDecision(
	SparkModelResidentIpcDecision *decision,
	uint64_t message_id,
	uint32_t decision_kind,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelResidentIpcValidateDecision(
	const SparkModelResidentIpcDecision *decision,
	uint32_t message_bytes);
SparkStatus SparkModelResidentIpcInitializeDecisionResult(
	SparkModelResidentIpcDecisionResult *result,
	const SparkModelResidentIpcDecision *decision,
	SparkStatus status);
SparkStatus SparkModelResidentIpcValidateDecisionResult(
	const SparkModelResidentIpcDecisionResult *result,
	uint32_t message_bytes,
	uint64_t message_id,
	uint32_t decision_kind,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelResidentIpcCalculateSubmitBytes(
	uint32_t lane_count,
	uint32_t row_count,
	uint32_t model_extension_bytes,
	uint32_t *message_bytes_out);
SparkStatus SparkModelResidentIpcEncodeSubmission(
	const SparkModelServingSubmission *submission,
	uint64_t message_id,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out);
SparkStatus SparkModelResidentIpcEncodePreparation(
	const SparkModelServingSubmission *submission,
	uint64_t message_id,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out);
SparkStatus SparkModelResidentIpcEncodeContinuation(
	const SparkModelServingSubmission *submission,
	uint64_t message_id,
	uint64_t client_generation,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out);
SparkStatus SparkModelResidentIpcDecodeSubmission(
	const void *message_buffer,
	uint32_t message_bytes,
	SparkModelServingSubmission *submission_out);
SparkStatus SparkModelResidentIpcCalculateCompletionBytes(
	uint32_t token_count,
	uint32_t model_extension_bytes,
	uint32_t *message_bytes_out);
SparkStatus SparkModelResidentIpcEncodeCompletion(
	const SparkModelServingCompletion *completion,
	uint64_t message_id,
	void *message_buffer,
	uint32_t message_capacity,
	uint32_t *message_bytes_out);
SparkStatus SparkModelResidentIpcDecodeCompletion(
	const void *message_buffer,
	uint32_t message_bytes,
	SparkModelServingCompletion *completion_out);

#ifdef __cplusplus
}
#endif
