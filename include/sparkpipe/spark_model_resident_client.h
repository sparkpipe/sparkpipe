#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_resident_endpoint.h"
#include "sparkpipe/spark_model_resident_ipc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION 9u
#define SPARK_MODEL_RESIDENT_CLIENT_MAX_QUEUE_CAPACITY 256u
#define SPARK_MODEL_RESIDENT_CLIENT_POLL_READ UINT32_C(0x00000001)
#define SPARK_MODEL_RESIDENT_CLIENT_POLL_WRITE UINT32_C(0x00000002)

typedef struct SparkModelResidentClient SparkModelResidentClient;

typedef void (*SparkModelResidentSubmitResultFunction)(
	void *result_context,
	uint64_t submission_id,
	SparkStatus status);
typedef void (*SparkModelResidentDecisionResultFunction)(
	void *result_context,
	uint64_t submission_id,
	uint32_t decision_kind,
	SparkStatus status);

typedef struct SparkModelResidentClientConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t connect_timeout_ms;
	uint32_t reserved0;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkModelResidentEndpoint endpoint;
	const SparkModelServingAdapterDescriptor *adapter_descriptor;
	SparkModelResidentSubmitResultFunction submit_result_function;
	void *submit_result_context;
	SparkModelResidentDecisionResultFunction decision_result_function;
	void *decision_result_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
} SparkModelResidentClientConfiguration;

/*
 * The client is an event-loop object: one thread owns submit/progress/destroy.
 * adapter_descriptor and every callback context must outlive the client. A
 * disconnect is terminal; reconnecting creates a new client and handshake.
 * Callbacks may submit more work after the client has updated its queue state.
 * They must not recursively call progress or destroy the client.
 */

typedef struct SparkModelResidentClientPollDescriptor
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	int32_t fd;
	uint32_t events;
} SparkModelResidentClientPollDescriptor;

typedef struct SparkModelResidentClientView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t connected;
	uint32_t queued_message_count;
	uint32_t pending_submission_count;
	uint32_t queue_capacity;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint32_t prepared_submission_count;
	uint64_t submitted_count;
	uint64_t prepared_count;
	uint64_t admitted_count;
	uint64_t rejected_count;
	uint64_t aborted_count;
	uint64_t completed_count;
} SparkModelResidentClientView;

#define SPARK_MODEL_RESIDENT_CLIENT_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkModelResidentClientConfiguration))
#define SPARK_MODEL_RESIDENT_CLIENT_POLL_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkModelResidentClientPollDescriptor))
#define SPARK_MODEL_RESIDENT_CLIENT_VIEW_BYTES \
	((uint32_t)sizeof(SparkModelResidentClientView))

SparkStatus SparkModelResidentClientConnect(
	const SparkModelResidentClientConfiguration *configuration,
	SparkModelResidentClient **client_out);
void SparkModelResidentClientDestroy(SparkModelResidentClient *client);
SparkStatus SparkModelResidentClientSubmit(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelResidentClientPrepare(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission);
SparkStatus SparkModelResidentClientCommit(
	SparkModelResidentClient *client,
	uint64_t submission_id);
SparkStatus SparkModelResidentClientAbort(
	SparkModelResidentClient *client,
	uint64_t submission_id);
SparkStatus SparkModelResidentClientProgress(
	SparkModelResidentClient *client,
	uint32_t maximum_message_count);
SparkStatus SparkModelResidentClientGetPollDescriptor(
	const SparkModelResidentClient *client,
	SparkModelResidentClientPollDescriptor *descriptor);
SparkStatus SparkModelResidentClientGetView(
	const SparkModelResidentClient *client,
	SparkModelResidentClientView *view);

#ifdef __cplusplus
}
#endif
