#include "sparkpipe/spark_model_batch_engine.h"

#include <stdlib.h>
#include <string.h>

#define SPARK_MODEL_BATCH_REQUEST_FREE 0u
#define SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL 1u
#define SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT 2u
#define SPARK_MODEL_BATCH_REQUEST_READY_DECODE 3u
#define SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT 4u
#define SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE 5u
#define SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT 6u
#define SPARK_MODEL_BATCH_REQUEST_COMPLETING 7u
#define SPARK_MODEL_BATCH_NO_SLOT UINT32_MAX

typedef struct SparkModelBatchRequestState
{
	uint32_t state;
	uint32_t generation;
	uint32_t next_free_slot;
	uint32_t priority;
	uint32_t prompt_token_count;
	uint32_t computed_prompt_token_count;
	uint32_t generated_token_count;
	uint32_t output_token_budget;
	uint32_t cancel_pending;
	uint32_t resident_bound;
	uint32_t terminal_event_kind;
	uint32_t terminal_status;
	uint64_t request_id;
	uint64_t sequence_id;
	SparkModelBatchRequestHandle handle;
} SparkModelBatchRequestState;

typedef struct SparkModelBatchSubmissionState
{
	uint32_t active;
	uint32_t slot_index;
	uint32_t work_kind;
	uint32_t lane_count;
	uint32_t result_received;
	uint32_t admitted;
	uint32_t result_status;
	uint32_t reserved0;
	uint64_t submission_id;
} SparkModelBatchSubmissionState;

struct SparkModelBatchEngine
{
	SparkModelPipelineClient *pipeline;
	const SparkModelServingAdapterDescriptor *adapter_descriptor;
	SparkModelBatchEventFunction event_function;
	void *event_context;
	SparkModelBatchRequestState *requests;
	SparkModelBatchSubmissionState *submissions;
	uint32_t *request_token_storage;
	uint32_t *submission_request_slots;
	uint32_t *submission_prefill_counts;
	SparkModelServingLane *scratch_lanes;
	uint32_t *scratch_token_ids;
	uint32_t *scratch_row_lane_indices;
	uint64_t *scratch_row_positions;
	uint64_t *scratch_row_sequence_ids;
	uint32_t *scratch_request_slots;
	uint32_t *scratch_prefill_counts;
	uint32_t request_capacity;
	uint32_t max_context_tokens;
	uint32_t max_prefill_rows;
	uint32_t max_active_sequence_count;
	uint32_t scratch_row_capacity;
	uint32_t submission_capacity;
	uint32_t maximum_messages_per_rank;
	uint32_t stop_token_count;
	uint32_t stop_token_ids[SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT];
	uint32_t free_request_head;
	uint32_t next_request_scan;
	uint32_t admission_open;
	uint32_t live_request_count;
	uint32_t inflight_submission_count;
	uint32_t failed_status;
	uint32_t next_work_kind;
	uint64_t next_submission_id;
	uint64_t submitted_request_count;
	uint64_t completed_request_count;
	uint64_t cancelled_request_count;
	uint64_t emitted_token_count;
};

static uint32_t SparkModelBatchMultiplyFits(uint32_t left,uint32_t right)
{
	return(left == 0u || right <= UINT32_MAX / left ? 1u : 0u);
}

static uint32_t *SparkModelBatchRequestTokens(
	SparkModelBatchEngine *engine,
	uint32_t request_slot)
{
	return(&engine->request_token_storage[(uint64_t)request_slot * engine->max_context_tokens]);
}

static uint32_t *SparkModelBatchSubmissionRequestSlots(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmissionState *submission)
{
	return(&engine->submission_request_slots[(uint64_t)submission->slot_index * engine->max_active_sequence_count]);
}

static uint32_t *SparkModelBatchSubmissionPrefillCounts(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmissionState *submission)
{
	return(&engine->submission_prefill_counts[(uint64_t)submission->slot_index * engine->max_active_sequence_count]);
}

static SparkModelBatchRequestHandle SparkModelBatchMakeHandle(
	uint32_t slot,
	uint32_t generation)
{
	return(((uint64_t)generation << 32u) | ((uint64_t)slot + 1u));
}

static uint32_t SparkModelBatchHandleSlot(
	SparkModelBatchRequestHandle handle)
{
	uint32_t encoded;
	encoded = (uint32_t)handle;
	return(encoded != 0u ? encoded - 1u : SPARK_MODEL_BATCH_NO_SLOT);
}

static SparkModelBatchRequestState *SparkModelBatchFindRequest(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestHandle handle)
{
	SparkModelBatchRequestState *request;
	uint32_t slot;
	slot = SparkModelBatchHandleSlot(handle);
	if ( engine == 0 || slot >= engine->request_capacity )
		return(0);
	request = &engine->requests[slot];
	return(request->state != SPARK_MODEL_BATCH_REQUEST_FREE && request->handle == handle ? request : 0);
}

static SparkModelBatchSubmissionState *SparkModelBatchFindSubmission(
	SparkModelBatchEngine *engine,
	uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<engine->submission_capacity; index++)
		if ( engine->submissions[index].active != 0u && engine->submissions[index].submission_id == submission_id )
			return(&engine->submissions[index]);
	return(0);
}

static SparkModelBatchSubmissionState *SparkModelBatchReserveSubmission(
	SparkModelBatchEngine *engine,
	uint32_t work_kind)
{
	SparkModelBatchSubmissionState *submission;
	uint32_t index;
	for (index=0u; index<engine->submission_capacity; index++)
	{
		submission = &engine->submissions[index];
		if ( submission->active == 0u )
		{
			memset(submission,0,sizeof(*submission));
			submission->active = 1u;
			submission->slot_index = index;
			submission->work_kind = work_kind;
			submission->result_status = SPARK_STATUS_PENDING;
			return(submission);
		}
	}
	return(0);
}

static void SparkModelBatchReleaseSubmission(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission)
{
	memset(submission,0,sizeof(*submission));
	engine->inflight_submission_count--;
}

static void SparkModelBatchEmit(
	SparkModelBatchEngine *engine,
	const SparkModelBatchRequestState *request,
	uint32_t kind,
	SparkStatus status,
	uint32_t flags,
	uint32_t token_id)
{
	SparkModelBatchEvent event;
	memset(&event,0,sizeof(event));
	event.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	event.descriptor_bytes = SPARK_MODEL_BATCH_EVENT_BYTES;
	event.kind = kind;
	event.flags = flags;
	event.status = status;
	event.token_id = token_id;
	event.token_index = request->generated_token_count != 0u ? request->generated_token_count - 1u : 0u;
	event.generated_token_count = request->generated_token_count;
	event.request_id = request->request_id;
	event.sequence_id = request->sequence_id;
	event.request_handle = request->handle;
	engine->event_function(engine->event_context,&event);
}

static void SparkModelBatchFreeRequest(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	uint32_t slot,generation;
	slot = (uint32_t)(request - engine->requests);
	generation = request->generation;
	memset(request,0,sizeof(*request));
	request->generation = generation;
	request->next_free_slot = engine->free_request_head;
	engine->free_request_head = slot;
	engine->live_request_count--;
}

static void SparkModelBatchEmitTerminal(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request)
{
	SparkModelBatchRequestState snapshot;
	uint32_t kind;
	SparkStatus status;
	snapshot = *request;
	kind = request->terminal_event_kind;
	status = (SparkStatus)request->terminal_status;
	SparkModelBatchFreeRequest(engine,request);
	if ( kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
		engine->completed_request_count++;
	else if ( kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED )
		engine->cancelled_request_count++;
	SparkModelBatchEmit(engine,&snapshot,kind,status,0u,0u);
}

static void SparkModelBatchQueueTerminal(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	uint32_t kind,
	SparkStatus status)
{
	request->terminal_event_kind = kind;
	request->terminal_status = status;
	if ( status == SPARK_STATUS_OK && engine->failed_status == SPARK_STATUS_OK && request->resident_bound != 0u && engine->adapter_descriptor->resident_sequence_slot_reuse == SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE )
	{
		request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE;
		return;
	}
	SparkModelBatchEmitTerminal(engine,request);
}

static uint32_t SparkModelBatchTokenIsStop(
	const SparkModelBatchEngine *engine,
	uint32_t token_id)
{
	uint32_t index;
	for (index=0u; index<engine->stop_token_count; index++)
		if ( engine->stop_token_ids[index] == token_id )
			return(1u);
	return(0u);
}

static SparkStatus SparkModelBatchAcceptToken(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	uint32_t token_id)
{
	uint32_t stop;
	uint32_t *tokens;
	if ( request->generated_token_count >= request->output_token_budget || request->prompt_token_count + request->generated_token_count >= engine->max_context_tokens )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	tokens = SparkModelBatchRequestTokens(engine,(uint32_t)(request - engine->requests));
	tokens[request->prompt_token_count + request->generated_token_count] = token_id;
	request->generated_token_count++;
	engine->emitted_token_count++;
	stop = SparkModelBatchTokenIsStop(engine,token_id);
	request->state = SPARK_MODEL_BATCH_REQUEST_COMPLETING;
	SparkModelBatchEmit(engine,request,SPARK_MODEL_BATCH_EVENT_TOKEN,SPARK_STATUS_OK,stop != 0u ? SPARK_MODEL_BATCH_EVENT_FLAG_STOP_TOKEN : 0u,token_id);
	if ( request->cancel_pending != 0u )
		SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED,SPARK_STATUS_OK);
	else if ( stop != 0u || request->generated_token_count >= request->output_token_budget )
		SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED,SPARK_STATUS_OK);
	else
		request->state = SPARK_MODEL_BATCH_REQUEST_READY_DECODE;
	return(SPARK_STATUS_OK);
}

static void SparkModelBatchSetFailed(
	SparkModelBatchEngine *engine,
	SparkStatus status)
{
	if ( engine->failed_status == SPARK_STATUS_OK )
		engine->failed_status = status;
	engine->admission_open = 0u;
}

static void SparkModelBatchFailRequest(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestState *request,
	SparkStatus status)
{
	SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_ERROR,status);
}

static void SparkModelBatchRestoreRejectedRequest(
	SparkModelBatchRequestState *request,
	uint32_t work_kind)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL;
	else if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		request->state = SPARK_MODEL_BATCH_REQUEST_READY_DECODE;
	else
		request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE;
}

static void SparkModelBatchHandleRejected(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	SparkStatus status)
{
	uint32_t *request_slots;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[request_slots[lane]];
		if ( status == SPARK_STATUS_BUSY )
			SparkModelBatchRestoreRejectedRequest(request,submission->work_kind);
		else
			SparkModelBatchFailRequest(engine,request,status);
	}
}

static SparkStatus SparkModelBatchHandlePrefillCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	const SparkModelServingCompletion *completion)
{
	uint32_t *request_slots,*prefill_counts;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	prefill_counts = SparkModelBatchSubmissionPrefillCounts(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[request_slots[lane]];
		request->computed_prompt_token_count += prefill_counts[lane];
		request->resident_bound = 1u;
		if ( request->computed_prompt_token_count < request->prompt_token_count )
			request->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL;
		else if ( SparkModelBatchAcceptToken(engine,request,completion->token_ids[lane]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelBatchHandleDecodeCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	const SparkModelServingCompletion *completion)
{
	uint32_t *request_slots;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[request_slots[lane]];
		request->resident_bound = 1u;
		if ( SparkModelBatchAcceptToken(engine,request,completion->token_ids[lane]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static void SparkModelBatchHandleReleaseCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission)
{
	uint32_t *request_slots;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		request = &engine->requests[request_slots[lane]];
		request->resident_bound = 0u;
		SparkModelBatchEmitTerminal(engine,request);
	}
}

static void SparkModelBatchSubmitResult(
	void *result_context,
	uint64_t submission_id,
	SparkStatus status)
{
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmissionState *submission;
	engine = (SparkModelBatchEngine *)result_context;
	submission = engine != 0 ? SparkModelBatchFindSubmission(engine,submission_id) : 0;
	if ( submission == 0 || submission->result_received != 0u )
	{
		if ( engine != 0 )
			SparkModelBatchSetFailed(engine,SPARK_STATUS_SCHEMA_ERROR);
		return;
	}
	submission->result_received = 1u;
	submission->result_status = status;
	submission->admitted = status == SPARK_STATUS_OK ? 1u : 0u;
}

static SparkStatus SparkModelBatchApplyCompletion(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	const SparkModelServingCompletion *completion)
{
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(SparkModelBatchHandlePrefillCompletion(engine,submission,completion));
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(SparkModelBatchHandleDecodeCompletion(engine,submission,completion));
	SparkModelBatchHandleReleaseCompletion(engine,submission);
	return(SPARK_STATUS_OK);
}

static void SparkModelBatchFailSubmissionRequests(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *submission,
	SparkStatus status)
{
	uint32_t *request_slots;
	uint32_t lane;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,submission);
	for (lane=0u; lane<submission->lane_count; lane++)
		SparkModelBatchFailRequest(engine,&engine->requests[request_slots[lane]],status);
}

static void SparkModelBatchCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmissionState *submission;
	SparkStatus status;
	engine = (SparkModelBatchEngine *)completion_context;
	submission = engine != 0 && completion != 0 ? SparkModelBatchFindSubmission(engine,completion->submission_id) : 0;
	if ( submission == 0 || submission->result_received == 0u )
	{
		if ( engine != 0 )
			SparkModelBatchSetFailed(engine,SPARK_STATUS_SCHEMA_ERROR);
		return;
	}
	status = (SparkStatus)completion->status;
	if ( submission->admitted == 0u || status != SPARK_STATUS_OK )
	{
		status = submission->admitted == 0u ? (SparkStatus)submission->result_status : status;
		if ( submission->admitted != 0u )
			SparkModelBatchSetFailed(engine,status);
		SparkModelBatchHandleRejected(engine,submission,status);
	}
	else if ( SparkModelBatchApplyCompletion(engine,submission,completion) != SPARK_STATUS_OK )
	{
		SparkModelBatchSetFailed(engine,SPARK_STATUS_CAPACITY_EXCEEDED);
		SparkModelBatchFailSubmissionRequests(engine,submission,SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	SparkModelBatchReleaseSubmission(engine,submission);
}

static SparkStatus SparkModelBatchValidateConfiguration(
	const SparkModelBatchEngineConfiguration *configuration)
{
	uint32_t left,right;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_BATCH_ENGINE_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( configuration->flags != 0u || configuration->connect_timeout_ms == 0u || configuration->request_capacity == 0u || configuration->max_context_tokens < 2u || configuration->max_prefill_rows_per_submission == 0u || configuration->maximum_messages_per_rank_per_progress == 0u || configuration->stop_token_count > SPARK_MODEL_BATCH_ENGINE_MAX_STOP_TOKEN_COUNT || configuration->deployment == 0 || configuration->runtime_root == 0 || configuration->event_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (left=0u; left<configuration->stop_token_count; left++)
		for (right=left + 1u; right<configuration->stop_token_count; right++)
			if ( configuration->stop_token_ids[left] == configuration->stop_token_ids[right] )
				return(SPARK_STATUS_DUPLICATE);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelBatchAllocate(
	SparkModelBatchEngine *engine)
{
	uint32_t request_tokens,submission_lanes;
	if ( SparkModelBatchMultiplyFits(engine->request_capacity,engine->max_context_tokens) == 0u || SparkModelBatchMultiplyFits(engine->submission_capacity,engine->max_active_sequence_count) == 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	request_tokens = engine->request_capacity * engine->max_context_tokens;
	submission_lanes = engine->submission_capacity * engine->max_active_sequence_count;
	engine->requests = (SparkModelBatchRequestState *)calloc(engine->request_capacity,sizeof(engine->requests[0]));
	engine->submissions = (SparkModelBatchSubmissionState *)calloc(engine->submission_capacity,sizeof(engine->submissions[0]));
	engine->request_token_storage = (uint32_t *)calloc(request_tokens,sizeof(engine->request_token_storage[0]));
	engine->submission_request_slots = (uint32_t *)calloc(submission_lanes,sizeof(engine->submission_request_slots[0]));
	engine->submission_prefill_counts = (uint32_t *)calloc(submission_lanes,sizeof(engine->submission_prefill_counts[0]));
	engine->scratch_lanes = (SparkModelServingLane *)calloc(engine->max_active_sequence_count,sizeof(engine->scratch_lanes[0]));
	engine->scratch_token_ids = (uint32_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_token_ids[0]));
	engine->scratch_row_lane_indices = (uint32_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_row_lane_indices[0]));
	engine->scratch_row_positions = (uint64_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_row_positions[0]));
	engine->scratch_row_sequence_ids = (uint64_t *)calloc(engine->scratch_row_capacity,sizeof(engine->scratch_row_sequence_ids[0]));
	engine->scratch_request_slots = (uint32_t *)calloc(engine->max_active_sequence_count,sizeof(engine->scratch_request_slots[0]));
	engine->scratch_prefill_counts = (uint32_t *)calloc(engine->max_active_sequence_count,sizeof(engine->scratch_prefill_counts[0]));
	return(engine->requests != 0 && engine->submissions != 0 && engine->request_token_storage != 0 && engine->submission_request_slots != 0 && engine->submission_prefill_counts != 0 && engine->scratch_lanes != 0 && engine->scratch_token_ids != 0 && engine->scratch_row_lane_indices != 0 && engine->scratch_row_positions != 0 && engine->scratch_row_sequence_ids != 0 && engine->scratch_request_slots != 0 && engine->scratch_prefill_counts != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkModelBatchInitializeFreeList(
	SparkModelBatchEngine *engine)
{
	uint32_t index;
	engine->free_request_head = 0u;
	for (index=0u; index<engine->request_capacity; index++)
		engine->requests[index].next_free_slot = index + 1u < engine->request_capacity ? index + 1u : SPARK_MODEL_BATCH_NO_SLOT;
}

static SparkStatus SparkModelBatchConnectPipeline(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine *engine)
{
	SparkModelPipelineClientConfiguration pipeline_configuration;
	memset(&pipeline_configuration,0,sizeof(pipeline_configuration));
	pipeline_configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	pipeline_configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	pipeline_configuration.connect_timeout_ms = configuration->connect_timeout_ms;
	pipeline_configuration.deployment = configuration->deployment;
	pipeline_configuration.runtime_root = configuration->runtime_root;
	pipeline_configuration.submit_result_function = SparkModelBatchSubmitResult;
	pipeline_configuration.submit_result_context = engine;
	pipeline_configuration.completion_function = SparkModelBatchCompletion;
	pipeline_configuration.completion_context = engine;
	return(SparkModelPipelineClientConnect(&pipeline_configuration,&engine->pipeline));
}

static SparkStatus SparkModelBatchInitialize(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine *engine)
{
	const SparkModelServingRuntimeLimits *limits;
	SparkStatus status;
	limits = &configuration->deployment->runtime_limits;
	engine->request_capacity = configuration->request_capacity;
	engine->max_context_tokens = configuration->max_context_tokens;
	engine->max_prefill_rows = configuration->max_prefill_rows_per_submission;
	engine->max_active_sequence_count = limits->max_active_sequence_count;
	engine->scratch_row_capacity = engine->max_prefill_rows > engine->max_active_sequence_count ? engine->max_prefill_rows : engine->max_active_sequence_count;
	engine->submission_capacity = limits->max_inflight_submission_count;
	engine->maximum_messages_per_rank = configuration->maximum_messages_per_rank_per_progress;
	engine->stop_token_count = configuration->stop_token_count;
	memcpy(engine->stop_token_ids,configuration->stop_token_ids,sizeof(engine->stop_token_ids));
	engine->event_function = configuration->event_function;
	engine->event_context = configuration->event_context;
	engine->admission_open = 1u;
	engine->next_work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	if ( engine->request_capacity > limits->resident_sequence_capacity ||
		engine->max_prefill_rows > limits->max_input_row_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkModelBatchAllocate(engine);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelBatchConnectPipeline(configuration,engine);
	if ( status != SPARK_STATUS_OK )
		return(status);
	engine->adapter_descriptor = SparkModelPipelineClientGetAdapterDescriptor(engine->pipeline);
	if ( engine->adapter_descriptor == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkModelBatchInitializeFreeList(engine);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineConnect(
	const SparkModelBatchEngineConfiguration *configuration,
	SparkModelBatchEngine **engine_out)
{
	SparkModelBatchEngine *engine;
	SparkStatus status;
	if ( engine_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*engine_out = 0;
	status = SparkModelBatchValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	engine = (SparkModelBatchEngine *)calloc(1u,sizeof(*engine));
	if ( engine == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkModelBatchInitialize(configuration,engine);
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkModelBatchEngineDestroy(engine);
		return(status);
	}
	*engine_out = engine;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineDestroy(SparkModelBatchEngine *engine)
{
	SparkModelPipelineClientView pipeline_view;
	SparkStatus status;
	if ( engine == 0 )
		return(SPARK_STATUS_OK);
	if ( engine->live_request_count != 0u || engine->inflight_submission_count != 0u )
		return(SPARK_STATUS_BUSY);
	if ( engine->pipeline != 0 )
	{
		status = SparkModelPipelineClientGetView(engine->pipeline,&pipeline_view);
		if ( status != SPARK_STATUS_OK || pipeline_view.active_transaction_count != 0u )
			return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_BUSY);
	}
	SparkModelPipelineClientDestroy(engine->pipeline);
	free(engine->scratch_prefill_counts);
	free(engine->scratch_request_slots);
	free(engine->scratch_row_sequence_ids);
	free(engine->scratch_row_positions);
	free(engine->scratch_row_lane_indices);
	free(engine->scratch_token_ids);
	free(engine->scratch_lanes);
	free(engine->submission_prefill_counts);
	free(engine->submission_request_slots);
	free(engine->request_token_storage);
	free(engine->submissions);
	free(engine->requests);
	free(engine);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelBatchRequestIdExists(
	const SparkModelBatchEngine *engine,
	uint64_t request_id,
	uint64_t sequence_id)
{
	uint32_t index;
	for (index=0u; index<engine->request_capacity; index++)
		if ( engine->requests[index].state != SPARK_MODEL_BATCH_REQUEST_FREE && (engine->requests[index].request_id == request_id || engine->requests[index].sequence_id == sequence_id) )
			return(1u);
	return(0u);
}

static SparkStatus SparkModelBatchValidateSubmit(
	const SparkModelBatchEngine *engine,
	const SparkModelBatchSubmitRequest *request)
{
	if ( engine == 0 || request == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->abi_version != SPARK_MODEL_BATCH_ENGINE_ABI_VERSION || request->descriptor_bytes != SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( request->reserved0 != 0u || request->request_id == 0u || request->sequence_id == 0u || request->prompt_token_ids == 0 || request->prompt_token_count == 0u || request->output_token_budget == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->prompt_token_count > engine->max_context_tokens || request->output_token_budget > engine->max_context_tokens - request->prompt_token_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkModelBatchRequestIdExists(engine,request->request_id,request->sequence_id) != 0u )
		return(SPARK_STATUS_DUPLICATE);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineSubmit(
	SparkModelBatchEngine *engine,
	const SparkModelBatchSubmitRequest *request,
	SparkModelBatchRequestHandle *request_handle_out)
{
	SparkModelBatchRequestState *state;
	SparkStatus status;
	uint32_t slot;
	if ( request_handle_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*request_handle_out = SPARK_MODEL_BATCH_ENGINE_INVALID_REQUEST_HANDLE;
	status = SparkModelBatchValidateSubmit(engine,request);
	if ( status != SPARK_STATUS_OK || engine->admission_open == 0u || engine->failed_status != SPARK_STATUS_OK || engine->free_request_head == SPARK_MODEL_BATCH_NO_SLOT )
		return(status != SPARK_STATUS_OK ? status : engine->failed_status != SPARK_STATUS_OK ? (SparkStatus)engine->failed_status : SPARK_STATUS_BUSY);
	slot = engine->free_request_head;
	state = &engine->requests[slot];
	engine->free_request_head = state->next_free_slot;
	state->generation++;
	if ( state->generation == 0u )
		state->generation = 1u;
	state->state = SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL;
	state->priority = request->priority;
	state->prompt_token_count = request->prompt_token_count;
	state->output_token_budget = request->output_token_budget;
	state->request_id = request->request_id;
	state->sequence_id = request->sequence_id;
	state->handle = SparkModelBatchMakeHandle(slot,state->generation);
	memcpy(SparkModelBatchRequestTokens(engine,slot),request->prompt_token_ids,(size_t)request->prompt_token_count * sizeof(uint32_t));
	engine->live_request_count++;
	engine->submitted_request_count++;
	*request_handle_out = state->handle;
	SparkModelBatchEmit(engine,state,SPARK_MODEL_BATCH_EVENT_REQUEST_ACCEPTED,SPARK_STATUS_OK,0u,0u);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineCancel(
	SparkModelBatchEngine *engine,
	SparkModelBatchRequestHandle request_handle)
{
	SparkModelBatchRequestState *request;
	request = SparkModelBatchFindRequest(engine,request_handle);
	if ( request == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( request->state == SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_COMPLETING )
	{
		request->cancel_pending = 1u;
		return(SPARK_STATUS_PENDING);
	}
	if ( request->state == SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE )
	{
		request->terminal_event_kind = SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED;
		request->terminal_status = SPARK_STATUS_OK;
		return(SPARK_STATUS_OK);
	}
	SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelBatchStateForWork(uint32_t work_kind)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(SPARK_MODEL_BATCH_REQUEST_READY_DECODE);
	return(SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE);
}

static uint32_t SparkModelBatchInflightStateForWork(uint32_t work_kind)
{
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		return(SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT);
	return(SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT);
}

static uint32_t SparkModelBatchSelectRequests(
	SparkModelBatchEngine *engine,
	uint32_t work_kind)
{
	uint32_t selected,index,slot,state,row_limit;
	selected = 0u;
	state = SparkModelBatchStateForWork(work_kind);
	row_limit = work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? engine->max_prefill_rows : engine->max_active_sequence_count;
	for (index=0u; index<engine->request_capacity && selected<engine->max_active_sequence_count && selected<row_limit; index++)
	{
		slot = (engine->next_request_scan + index) % engine->request_capacity;
		if ( engine->requests[slot].state == state )
			engine->scratch_request_slots[selected++] = slot;
	}
	if ( selected != 0u )
		engine->next_request_scan = (engine->scratch_request_slots[selected - 1u] + 1u) % engine->request_capacity;
	return(selected);
}

static uint32_t SparkModelBatchAssignPrefillCounts(
	SparkModelBatchEngine *engine,
	uint32_t lane_count)
{
	uint32_t lane,remaining_rows;
	remaining_rows = engine->max_prefill_rows - lane_count;
	for (lane=0u; lane<lane_count; lane++)
		engine->scratch_prefill_counts[lane] = 1u;
	for (lane=0u; lane<lane_count && remaining_rows!=0u; lane++)
	{
		SparkModelBatchRequestState *request;
		uint32_t remaining_prompt,extra;
		request = &engine->requests[engine->scratch_request_slots[lane]];
		remaining_prompt = request->prompt_token_count - request->computed_prompt_token_count;
		extra = remaining_prompt - 1u < remaining_rows ? remaining_prompt - 1u : remaining_rows;
		engine->scratch_prefill_counts[lane] += extra;
		remaining_rows -= extra;
	}
	return(engine->max_prefill_rows - remaining_rows);
}

static void SparkModelBatchInitializeSubmission(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission,
	uint32_t work_kind,
	uint32_t lane_count)
{
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = work_kind;
	submission->submission_id = engine->next_submission_id;
	submission->request_id = engine->next_submission_id;
	submission->sequence_id = engine->next_submission_id;
	submission->control_generation = 1u;
	submission->transaction_id = engine->next_submission_id;
	submission->dispatch_generation = engine->next_submission_id;
	submission->request_generation = 1u;
	submission->step_generation = engine->next_submission_id;
	submission->residency.word0 = engine->next_submission_id;
	submission->residency.word1 = engine->next_submission_id ^ UINT64_C(0x535041524b504950);
	submission->residency.generation = engine->next_submission_id;
	submission->residency.owner = 1u;
	submission->active_sequence_count = lane_count;
	submission->lane_count = lane_count;
	submission->lanes = engine->scratch_lanes;
}

static void SparkModelBatchInitializeLane(
	SparkModelBatchEngine *engine,
	SparkModelServingLane *lane,
	uint32_t request_slot,
	uint64_t sequence_position,
	uint32_t context_token_count,
	uint32_t input_token_id)
{
	SparkModelBatchRequestState *request;
	request = &engine->requests[request_slot];
	memset(lane,0,sizeof(*lane));
	lane->request_id = request->request_id;
	lane->request_generation = request->generation;
	lane->step_generation = engine->next_submission_id;
	lane->sequence_id = request->sequence_id;
	lane->sequence_position = sequence_position;
	lane->resident_sequence_slot = request_slot;
	lane->context_token_count = context_token_count;
	lane->input_token_id = input_token_id;
}

static void SparkModelBatchBuildPrefillRows(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission,
	uint32_t lane_count)
{
	uint32_t lane,row,source,slot,wave,maximum;
	maximum = 0u;
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		uint32_t *tokens;
		slot = engine->scratch_request_slots[lane];
		request = &engine->requests[slot];
		tokens = SparkModelBatchRequestTokens(engine,slot);
		SparkModelBatchInitializeLane(engine,&engine->scratch_lanes[lane],slot,request->computed_prompt_token_count,request->computed_prompt_token_count + engine->scratch_prefill_counts[lane],tokens[request->computed_prompt_token_count]);
		if ( engine->scratch_prefill_counts[lane] > maximum )
			maximum = engine->scratch_prefill_counts[lane];
	}
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<lane_count; lane++)
			if ( engine->scratch_prefill_counts[lane] > wave )
		{
			SparkModelBatchRequestState *request;
			uint32_t *tokens;
			slot = engine->scratch_request_slots[lane];
			request = &engine->requests[slot];
			tokens = SparkModelBatchRequestTokens(engine,slot);
			source = request->computed_prompt_token_count + wave;
			engine->scratch_token_ids[row] = tokens[source];
			engine->scratch_row_lane_indices[row] = lane;
			engine->scratch_row_positions[row] = source;
			engine->scratch_row_sequence_ids[row] = request->sequence_id;
			row++;
		}
	submission->row_count = row;
	submission->token_count = row;
	submission->new_token_count = row;
}

static void SparkModelBatchBuildDecodeRows(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission,
	uint32_t lane_count)
{
	uint32_t lane,slot,position;
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		uint32_t *tokens;
		slot = engine->scratch_request_slots[lane];
		request = &engine->requests[slot];
		tokens = SparkModelBatchRequestTokens(engine,slot);
		position = request->prompt_token_count + request->generated_token_count - 1u;
		SparkModelBatchInitializeLane(engine,&engine->scratch_lanes[lane],slot,position,position + 1u,tokens[position]);
		engine->scratch_token_ids[lane] = tokens[position];
		engine->scratch_row_lane_indices[lane] = lane;
		engine->scratch_row_positions[lane] = position;
		engine->scratch_row_sequence_ids[lane] = request->sequence_id;
	}
	submission->row_count = lane_count;
	submission->token_count = lane_count;
	submission->new_token_count = lane_count;
}

static void SparkModelBatchBuildReleaseLanes(
	SparkModelBatchEngine *engine,
	uint32_t lane_count)
{
	uint32_t lane,slot,position;
	for (lane=0u; lane<lane_count; lane++)
	{
		SparkModelBatchRequestState *request;
		slot = engine->scratch_request_slots[lane];
		request = &engine->requests[slot];
		position = request->prompt_token_count + request->generated_token_count;
		SparkModelBatchInitializeLane(engine,&engine->scratch_lanes[lane],slot,position,position,0u);
	}
}

static void SparkModelBatchFinishSubmissionShape(
	SparkModelBatchEngine *engine,
	SparkModelServingSubmission *submission)
{
	uint32_t lane;
	submission->token_ids = submission->row_count != 0u ? engine->scratch_token_ids : 0;
	submission->row_lane_indices = submission->row_count != 0u ? engine->scratch_row_lane_indices : 0;
	submission->row_positions = submission->row_count != 0u ? engine->scratch_row_positions : 0;
	submission->row_sequence_ids = submission->row_count != 0u ? engine->scratch_row_sequence_ids : 0;
	for (lane=0u; lane<submission->lane_count; lane++)
		if ( engine->requests[engine->scratch_request_slots[lane]].priority > submission->priority )
			submission->priority = engine->requests[engine->scratch_request_slots[lane]].priority;
}

static uint32_t SparkModelBatchBuildSubmission(
	SparkModelBatchEngine *engine,
	uint32_t work_kind,
	SparkModelServingSubmission *submission)
{
	uint32_t lane_count;
	lane_count = SparkModelBatchSelectRequests(engine,work_kind);
	if ( lane_count == 0u )
		return(0u);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		(void)SparkModelBatchAssignPrefillCounts(engine,lane_count);
	SparkModelBatchInitializeSubmission(engine,submission,work_kind,lane_count);
	if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		SparkModelBatchBuildPrefillRows(engine,submission,lane_count);
	else if ( work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		SparkModelBatchBuildDecodeRows(engine,submission,lane_count);
	else
		SparkModelBatchBuildReleaseLanes(engine,lane_count);
	SparkModelBatchFinishSubmissionShape(engine,submission);
	return(lane_count);
}

static void SparkModelBatchRecordSubmission(
	SparkModelBatchEngine *engine,
	SparkModelBatchSubmissionState *state,
	uint32_t lane_count)
{
	uint32_t *request_slots,*prefill_counts;
	uint32_t lane,inflight_state;
	request_slots = SparkModelBatchSubmissionRequestSlots(engine,state);
	prefill_counts = SparkModelBatchSubmissionPrefillCounts(engine,state);
	state->lane_count = lane_count;
	state->submission_id = engine->next_submission_id;
	inflight_state = SparkModelBatchInflightStateForWork(state->work_kind);
	for (lane=0u; lane<lane_count; lane++)
	{
		request_slots[lane] = engine->scratch_request_slots[lane];
		prefill_counts[lane] = state->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? engine->scratch_prefill_counts[lane] : 0u;
		engine->requests[request_slots[lane]].state = inflight_state;
	}
	engine->inflight_submission_count++;
}

static SparkStatus SparkModelBatchDispatchKind(
	SparkModelBatchEngine *engine,
	uint32_t work_kind,
	uint32_t *dispatched_out)
{
	SparkModelBatchSubmissionState *state;
	SparkModelServingSubmission submission;
	SparkStatus status;
	uint32_t lane_count;
	*dispatched_out = 0u;
	state = SparkModelBatchReserveSubmission(engine,work_kind);
	if ( state == 0 )
		return(SPARK_STATUS_BUSY);
	engine->next_submission_id++;
	if ( engine->next_submission_id == 0u )
	{
		state->active = 0u;
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	lane_count = SparkModelBatchBuildSubmission(engine,work_kind,&submission);
	if ( lane_count == 0u )
	{
		state->active = 0u;
		return(SPARK_STATUS_NOT_FOUND);
	}
	status = SparkModelPipelineClientSubmit(engine->pipeline,&submission);
	if ( status != SPARK_STATUS_OK )
	{
		state->active = 0u;
		return(status);
	}
	SparkModelBatchRecordSubmission(engine,state,lane_count);
	*dispatched_out = 1u;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkModelBatchHasState(
	const SparkModelBatchEngine *engine,
	uint32_t state)
{
	uint32_t index;
	for (index=0u; index<engine->request_capacity; index++)
		if ( engine->requests[index].state == state )
			return(1u);
	return(0u);
}

static uint32_t SparkModelBatchChooseWorkKind(
	SparkModelBatchEngine *engine)
{
	uint32_t kind;
	if ( SparkModelBatchHasState(engine,SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE) != 0u )
		return(SPARK_MODEL_SERVING_WORK_KIND_RELEASE);
	kind = engine->next_work_kind;
	if ( kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL && SparkModelBatchHasState(engine,SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL) != 0u )
		engine->next_work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	else if ( SparkModelBatchHasState(engine,SPARK_MODEL_BATCH_REQUEST_READY_DECODE) != 0u )
	{
		kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
		engine->next_work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	}
	else if ( SparkModelBatchHasState(engine,SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL) != 0u )
		kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	else
		kind = 0u;
	return(kind);
}

static void SparkModelBatchFailIdleRequests(
	SparkModelBatchEngine *engine,
	SparkStatus status)
{
	uint32_t index,state;
	for (index=0u; index<engine->request_capacity; index++)
	{
		state = engine->requests[index].state;
		if ( state != SPARK_MODEL_BATCH_REQUEST_FREE && state != SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT && state != SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT && state != SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT )
			SparkModelBatchFailRequest(engine,&engine->requests[index],status);
	}
}

SparkStatus SparkModelBatchEngineProgress(
	SparkModelBatchEngine *engine,
	uint32_t maximum_new_submission_count)
{
	SparkStatus status;
	uint32_t dispatched,kind,step;
	if ( engine == 0 || maximum_new_submission_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelPipelineClientProgress(engine->pipeline,engine->maximum_messages_per_rank);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelBatchSetFailed(engine,status);
		SparkModelBatchFailIdleRequests(engine,status);
		return(status);
	}
	if ( engine->failed_status != SPARK_STATUS_OK )
		return((SparkStatus)engine->failed_status);
	for (step=0u; step<maximum_new_submission_count && engine->inflight_submission_count<engine->submission_capacity; step++)
	{
		kind = SparkModelBatchChooseWorkKind(engine);
		if ( kind == 0u )
			break;
		status = SparkModelBatchDispatchKind(engine,kind,&dispatched);
		if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_NOT_FOUND )
			break;
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelBatchSetFailed(engine,status);
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineCloseAdmission(
	SparkModelBatchEngine *engine)
{
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	engine->admission_open = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelBatchEngineBeginShutdown(
	SparkModelBatchEngine *engine)
{
	SparkModelBatchRequestState *request;
	uint32_t index;
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	engine->admission_open = 0u;
	for (index=0u; index<engine->request_capacity; index++)
	{
		request = &engine->requests[index];
		if ( request->state == SPARK_MODEL_BATCH_REQUEST_FREE )
			continue;
		if ( request->state == SPARK_MODEL_BATCH_REQUEST_PREFILL_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_DECODE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_RELEASE_INFLIGHT || request->state == SPARK_MODEL_BATCH_REQUEST_COMPLETING )
			request->cancel_pending = 1u;
		else if ( request->state == SPARK_MODEL_BATCH_REQUEST_QUEUED_RELEASE )
		{
			request->terminal_event_kind = SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED;
			request->terminal_status = SPARK_STATUS_OK;
		}
		else
			SparkModelBatchQueueTerminal(engine,request,SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED,SPARK_STATUS_OK);
	}
	return(engine->live_request_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_PENDING);
}

SparkStatus SparkModelBatchEngineGetPollDescriptors(
	const SparkModelBatchEngine *engine,
	SparkModelResidentClientPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out)
{
	if ( engine == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkModelPipelineClientGetPollDescriptors(engine->pipeline,descriptors,descriptor_capacity,descriptor_count_out));
}

static void SparkModelBatchCountStates(
	const SparkModelBatchEngine *engine,
	SparkModelBatchEngineView *view)
{
	uint32_t index;
	for (index=0u; index<engine->request_capacity; index++)
	{
		if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_QUEUED_PREFILL )
			view->queued_prefill_count++;
		if ( engine->requests[index].state == SPARK_MODEL_BATCH_REQUEST_READY_DECODE )
			view->ready_decode_count++;
	}
}

SparkStatus SparkModelBatchEngineGetView(
	const SparkModelBatchEngine *engine,
	SparkModelBatchEngineView *view)
{
	SparkStatus status;
	if ( engine == 0 || view == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	view->descriptor_bytes = SPARK_MODEL_BATCH_ENGINE_VIEW_BYTES;
	view->admission_open = engine->admission_open;
	view->request_capacity = engine->request_capacity;
	view->live_request_count = engine->live_request_count;
	view->inflight_submission_count = engine->inflight_submission_count;
	view->failed_status = engine->failed_status;
	view->submitted_request_count = engine->submitted_request_count;
	view->completed_request_count = engine->completed_request_count;
	view->cancelled_request_count = engine->cancelled_request_count;
	view->emitted_token_count = engine->emitted_token_count;
	SparkModelBatchCountStates(engine,view);
	status = SparkModelPipelineClientGetView(engine->pipeline,&view->pipeline);
	return(status);
}

const SparkModelServingAdapterDescriptor *SparkModelBatchEngineGetAdapterDescriptor(
	const SparkModelBatchEngine *engine)
{
	return(engine != 0 ? engine->adapter_descriptor : 0);
}
