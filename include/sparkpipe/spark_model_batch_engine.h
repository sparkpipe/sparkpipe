#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_pipeline_client.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_BATCH_ENGINE_ABI_VERSION 2u
#define SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT 16u
#define SPARK_MODEL_BATCH_ENGINE_INVALID_REQUEST_HANDLE 0u

#define SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED 1u
#define SPARK_MODEL_BATCH_EVENT_TOKEN 2u
#define SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED 3u
#define SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED 4u
#define SPARK_MODEL_BATCH_EVENT_ERROR 5u

#define SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN UINT32_C(0x00000001)
#define SPARK_MODEL_BATCH_EVENT_KNOWN_FLAGS \
	SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN

typedef uint64_t SparkModelBatchRequestHandle;
typedef struct SparkModelBatchEngine SparkModelBatchEngine;

typedef struct SparkModelBatchEvent
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t kind;
	uint32_t flags;
	uint32_t status;
	uint32_t token_id;
	uint32_t token_index;
	uint32_t generated_token_count;
	uint64_t request_id;
	uint64_t sequence_id;
	SparkModelBatchRequestHandle request_handle;
} SparkModelBatchEvent;

typedef void (*SparkModelBatchEventFunction)(
	void *event_context,
	const SparkModelBatchEvent *event);

typedef struct SparkModelBatchEngineConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t connect_timeout_ms;
	uint32_t request_capacity;
	uint32_t max_context_tokens;
	uint32_t max_prefill_rows_per_submission;
	uint32_t maximum_messages_per_rank_per_progress;
	uint32_t stop_token_count;
	uint32_t stop_token_ids[SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT];
	const SparkModelResidentDeployment *deployment;
	const char *runtime_root;
	SparkModelBatchEventFunction event_function;
	void *event_context;
} SparkModelBatchEngineConfiguration;

typedef struct SparkModelBatchSubmitRequest
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t priority;
	uint32_t output_token_budget;
	uint64_t request_id;
	uint64_t sequence_id;
	const uint32_t *prompt_token_ids;
	uint32_t prompt_token_count;
	uint32_t reserved0;
} SparkModelBatchSubmitRequest;

typedef struct SparkModelBatchEngineView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t admission_open;
	uint32_t request_capacity;
	uint32_t live_request_count;
	uint32_t queued_prefill_count;
	uint32_t ready_decode_count;
	uint32_t inflight_submission_count;
	uint32_t failed_status;
	uint64_t submitted_request_count;
	uint64_t completed_request_count;
	uint64_t cancelled_request_count;
	uint64_t emitted_token_count;
	SparkModelPipelineClientView pipeline;
} SparkModelBatchEngineView;

#define SPARK_MODEL_BATCH_EVENT_BYTES \
	((uint32_t)sizeof(SparkModelBatchEvent))
#define SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkModelBatchEngineConfiguration))
#define SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES \
	((uint32_t)sizeof(SparkModelBatchSubmitRequest))
#define SPARK_MODEL_BATCH_ENGINE_VIEW_BYTES \
	((uint32_t)sizeof(SparkModelBatchEngineView))

/*
 * One event-loop thread owns submit/cancel/progress/shutdown/destroy. Event
 * callbacks may submit new requests, but must not recursively progress,
 * shutdown, or destroy the engine. Request token storage and all dispatch
 * workspaces are allocated once during connect; no request or dispatch
 * allocation occurs on the hot path. BeginShutdown closes admission and marks
 * every live request for cancellation. The owner must continue Progress until
 * live_request_count and inflight_submission_count are zero; Destroy returns
 * BUSY instead of abandoning model-owned resident state before then.
 */
SparkStatus SparkModelBatchEngineConnect(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine **engine_out);
SparkStatus SparkModelBatchEngineDestroy(SparkModelBatchEngine *engine);
SparkStatus SparkModelBatchEngineSubmit(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmitRequest *request,
	SparkModelBatchRequestHandle *request_handle_out);
SparkStatus SparkModelBatchEngineCancel(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestHandle request_handle);
SparkStatus SparkModelBatchEngineProgress(
	SparkModelBatchEngine *engine,
	uint32_t maximum_new_submission_count);
SparkStatus SparkModelBatchEngineCloseAdmission(
	SparkModelBatchEngine *engine);
SparkStatus SparkModelBatchEngineBeginShutdown(
	SparkModelBatchEngine *engine);
SparkStatus SparkModelBatchEngineGetPollDescriptors(
	const SparkModelBatchEngine *engine,
	SparkModelResidentClientPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out);
SparkStatus SparkModelBatchEngineGetView(
	const SparkModelBatchEngine *engine,
	SparkModelBatchEngineView *view);
const SparkModelServingAdapterDescriptor *SparkModelBatchEngineGetAdapterDescriptor(
	const SparkModelBatchEngine *engine);

#ifdef __cplusplus
}
#endif
