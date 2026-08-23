#include "sparkpipe/spark_model_pipeline_client.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model_continuation_lease.h"
#include "spark_filesystem.h"

typedef struct SparkModelPipelineLeaseSlot
{
	uint64_t request_id;
	uint64_t request_generation;
	uint64_t sequence_id;
	SparkModelContinuationLease lease;
} SparkModelPipelineLeaseSlot;

typedef struct SparkModelPipelineTransaction
{
	uint32_t active;
	uint32_t work_kind;
	uint32_t active_sequence_count;
	uint32_t row_count;
	uint32_t tokens_per_sequence;
	uint32_t result_reported;
	uint32_t status;
	uint32_t result_mask;
	uint32_t prepared_mask;
	uint32_t decision_expected_mask;
	uint32_t decision_result_mask;
	uint32_t decision_kind;
	uint32_t completion_mask;
	uint32_t continued;
	uint32_t lane_count;
	uint64_t submitted_time_ns;
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
	SparkModelServingCompletion final_completion;
} SparkModelPipelineTransaction;

typedef struct SparkModelPipelineRankContext
{
	struct SparkModelPipelineClient *pipeline;
	uint32_t stage_index;
} SparkModelPipelineRankContext;

struct SparkModelPipelineClient
{
	uint32_t rank_count;
	uint32_t transaction_capacity;
	uint32_t active_transaction_count;
	uint32_t failed_status;
	uint32_t failed_stage_index;
	uint32_t all_rank_mask;
	uint32_t active_continue_lease_count;
	uint64_t lease_generation;
	uint64_t last_submission_id;
	uint64_t submitted_count;
	uint64_t continued_count;
	uint64_t admitted_count;
	uint64_t rejected_count;
	uint64_t completed_count;
	SparkModelServingAdapterDynamicLibrary adapter_library;
	SparkModelServingRuntimeLimits runtime_limits;
	const SparkModelServingAdapterDescriptor *adapter_descriptor;
	SparkModelResidentSubmitResultFunction submit_result_function;
	void *submit_result_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelPipelineStageCompletionFunction stage_completion_function;
	void *stage_completion_context;
	SparkModelResidentClient **clients;
	SparkModelPipelineRankContext *rank_contexts;
	SparkModelPipelineTransaction *transactions;
	SparkModelServingLane *transaction_lanes;
	SparkModelPipelineLeaseSlot *lease_slots;
};

static SparkModelServingLane *SparkModelPipelineClientTransactionLanes(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	uint64_t index;
	index = (uint64_t)(transaction - pipeline->transactions) *
		pipeline->runtime_limits.max_active_sequence_count;
	return(&pipeline->transaction_lanes[index]);
}

static uint64_t SparkModelPipelineClientMonotonicNanoseconds(void)
{
	struct timespec timestamp;
	if ( clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0 )
		return(0u);
	return(((uint64_t)timestamp.tv_sec * UINT64_C(1000000000)) + (uint64_t)timestamp.tv_nsec);
}

static SparkModelPipelineTransaction *SparkModelPipelineClientFind(
	SparkModelPipelineClient *pipeline,
	uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<pipeline->transaction_capacity; index++)
		if ( pipeline->transactions[index].active != 0u && pipeline->transactions[index].submission_id == submission_id )
			return(&pipeline->transactions[index]);
	return(0);
}

static SparkModelPipelineTransaction *SparkModelPipelineClientReserve(
	SparkModelPipelineClient *pipeline,
	const SparkModelServingSubmission *submission)
{
	SparkModelPipelineTransaction *transaction;
	SparkModelServingLane *lanes;
	uint32_t index;
	for (index=0u; index<pipeline->transaction_capacity; index++)
	{
		transaction = &pipeline->transactions[index];
		if ( transaction->active == 0u )
		{
			memset(transaction,0,sizeof(*transaction));
			transaction->active = 1u;
			transaction->work_kind = submission->work_kind;
			transaction->active_sequence_count = submission->active_sequence_count;
			transaction->row_count = submission->row_count;
			transaction->tokens_per_sequence = submission->tokens_per_sequence;
			transaction->lane_count = submission->active_sequence_count;
			transaction->status = SPARK_STATUS_OK;
			if ( pipeline->stage_completion_function != 0 )
				transaction->submitted_time_ns = SparkModelPipelineClientMonotonicNanoseconds();
			transaction->submission_id = submission->submission_id;
			transaction->request_id = submission->request_id;
			transaction->sequence_id = submission->sequence_id;
			transaction->sequence_position = submission->sequence_position;
			transaction->control_generation = submission->control_generation;
			transaction->transaction_id = submission->transaction_id;
			transaction->dispatch_generation = submission->dispatch_generation;
			transaction->request_generation = submission->request_generation;
			transaction->step_generation = submission->step_generation;
			transaction->residency = submission->residency;
			lanes = SparkModelPipelineClientTransactionLanes(pipeline,transaction);
			memcpy(lanes,submission->lanes,submission->active_sequence_count *
				sizeof(lanes[0]));
			pipeline->active_transaction_count++;
			return(transaction);
		}
	}
	return(0);
}

static void SparkModelPipelineClientRelease(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	memset(transaction,0,sizeof(*transaction));
	pipeline->active_transaction_count--;
}

static void SparkModelPipelineClientRecordFailure(
	SparkModelPipelineTransaction *transaction,
	SparkStatus status)
{
	if ( transaction->status == SPARK_STATUS_OK )
		transaction->status = (uint32_t)status;
}

static void SparkModelPipelineClientSetFailure(
	SparkModelPipelineClient *pipeline,
	SparkStatus status,
	uint32_t stage_index)
{
	uint32_t rank,slot;
	if ( pipeline->failed_status != SPARK_STATUS_OK )
		return;
	pipeline->failed_status = status;
	pipeline->failed_stage_index = stage_index;
	for (rank=0u; rank<pipeline->rank_count; rank++)
		SparkModelResidentClientFailStop(pipeline->clients[rank]);
	for (slot=0u; slot<pipeline->runtime_limits.resident_sequence_capacity;
		slot++)
	{
		memset(&pipeline->lease_slots[slot],0,
			sizeof(pipeline->lease_slots[slot]));
	}
	pipeline->active_continue_lease_count = 0u;
}

static uint32_t SparkModelPipelineClientLeaseMatches(
	const SparkModelPipelineClient *pipeline,
	const SparkModelServingLane *lane,
	uint64_t control_generation)
{
	const SparkModelPipelineLeaseSlot *slot;
	SparkStatus status;
	if ( lane->resident_sequence_slot >=
		pipeline->runtime_limits.resident_sequence_capacity )
		return(0u);
	slot = &pipeline->lease_slots[lane->resident_sequence_slot];
	if ( slot->request_id != lane->request_id ||
		slot->request_generation != lane->request_generation ||
		slot->sequence_id != lane->sequence_id )
		return(0u);
	status = SparkModelContinuationLeaseValidate(&slot->lease,
		pipeline->lease_generation,control_generation,lane->sequence_position,
		lane->step_generation);
	return(status == SPARK_STATUS_OK ? 1u : 0u);
}

static uint32_t SparkModelPipelineClientCanContinue(
	const SparkModelPipelineClient *pipeline,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane;
	if ( (pipeline->adapter_descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) == 0u )
		return(0u);
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( SparkModelPipelineClientLeaseMatches(pipeline,
			&submission->lanes[lane],submission->control_generation) == 0u )
			return(0u);
	return(1u);
}

static void SparkModelPipelineClientInvalidateLease(
	SparkModelPipelineClient *pipeline,
	uint32_t resident_sequence_slot)
{
	SparkModelPipelineLeaseSlot *slot;
	slot = &pipeline->lease_slots[resident_sequence_slot];
	if ( SparkModelContinuationLeaseIsActive(&slot->lease) != 0u )
		pipeline->active_continue_lease_count--;
	memset(slot,0,sizeof(*slot));
}

static SparkStatus SparkModelPipelineClientUpdateLeases(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	SparkModelPipelineLeaseSlot *slot;
	SparkModelServingLane *lanes;
	SparkStatus status;
	uint64_t next_sequence_position;
	uint32_t lane;
	if ( (pipeline->adapter_descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE) == 0u )
		return(SPARK_STATUS_OK);
	lanes = SparkModelPipelineClientTransactionLanes(pipeline,transaction);
	for (lane=0u; lane<transaction->lane_count; lane++)
	{
		slot = &pipeline->lease_slots[lanes[lane].resident_sequence_slot];
		if ( transaction->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		{
			SparkModelPipelineClientInvalidateLease(pipeline,
				lanes[lane].resident_sequence_slot);
			continue;
		}
		if ( SparkModelContinuationLeaseIsActive(&slot->lease) == 0u )
			pipeline->active_continue_lease_count++;
		slot->request_id = lanes[lane].request_id;
		slot->request_generation = lanes[lane].request_generation;
		slot->sequence_id = lanes[lane].sequence_id;
		next_sequence_position = lanes[lane].context_token_count;
		if ( transaction->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		{
			uint32_t completed_tokens;
			/* The lease must advance by the COMPLETION's emitted count
			 * (1 + accepted), not the submission's chain width (8). A
			 * partial-accept verify burst emits fewer tokens than the
			 * admitted chain, so the client lease must mirror the
			 * residentd's completion-derived advance. */
			completed_tokens = transaction->final_completion.tokens_per_sequence;
			if ( completed_tokens == 0u )
				completed_tokens = transaction->tokens_per_sequence;
			status = SparkModelContinuationLeaseDecodePosition(
				lanes[lane].context_token_count,
				completed_tokens,&next_sequence_position);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		status = SparkModelContinuationLeaseEstablish(&slot->lease,
			pipeline->lease_generation,transaction->control_generation,
			next_sequence_position,
			lanes[lane].step_generation);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static void SparkModelPipelineClientReportResult(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	if ( transaction->result_reported != 0u ||
		transaction->result_mask != pipeline->all_rank_mask )
		return;
	if ( transaction->continued == 0u && (transaction->decision_kind == 0u ||
		transaction->decision_result_mask != transaction->decision_expected_mask) )
		return;
	transaction->result_reported = 1u;
	if ( transaction->status == SPARK_STATUS_OK )
		pipeline->admitted_count++;
	else
		pipeline->rejected_count++;
	if ( pipeline->submit_result_function != 0 )
		pipeline->submit_result_function(pipeline->submit_result_context,transaction->submission_id,(SparkStatus)transaction->status);
}

static void SparkModelPipelineClientReportCompletion(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	SparkModelServingCompletion completion;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkStatus lease_status;
	if ( transaction->result_reported == 0u || transaction->completion_mask != pipeline->all_rank_mask )
		return;
	if ( transaction->status == SPARK_STATUS_OK )
	{
		lease_status = SparkModelPipelineClientUpdateLeases(pipeline,transaction);
		if ( lease_status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientRecordFailure(transaction,lease_status);
			SparkModelPipelineClientSetFailure(pipeline,lease_status,
				SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX);
		}
	}
	if ( transaction->status == SPARK_STATUS_OK )
		completion = transaction->final_completion;
	else
	{
		memset(&completion,0,sizeof(completion));
		completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
		completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
		completion.status = transaction->status;
		completion.submission_id = transaction->submission_id;
		completion.request_id = transaction->request_id;
		completion.sequence_id = transaction->sequence_id;
		completion.sequence_position = transaction->sequence_position;
		completion.control_generation = transaction->control_generation;
		completion.transaction_id = transaction->transaction_id;
		completion.dispatch_generation = transaction->dispatch_generation;
		completion.request_generation = transaction->request_generation;
		completion.step_generation = transaction->step_generation;
	}
	completion_function = pipeline->completion_function;
	completion_context = pipeline->completion_context;
	pipeline->completed_count++;
	SparkModelPipelineClientRelease(pipeline,transaction);
	if ( completion_function != 0 )
		completion_function(completion_context,&completion);
}

static SparkStatus SparkModelPipelineClientResolveAdmission(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	SparkStatus status;
	uint32_t rank;
	if ( transaction->result_mask != pipeline->all_rank_mask || transaction->decision_kind != 0u )
		return(SPARK_STATUS_OK);
	if ( transaction->continued != 0u )
	{
		if ( transaction->status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientSetFailure(pipeline,
				(SparkStatus)transaction->status,
				SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX);
			return((SparkStatus)transaction->status);
		}
		SparkModelPipelineClientReportResult(pipeline,transaction);
		SparkModelPipelineClientReportCompletion(pipeline,transaction);
		return(SPARK_STATUS_OK);
	}
	status = SPARK_STATUS_OK;
	if ( transaction->status == SPARK_STATUS_OK )
	{
		for (rank=0u; status==SPARK_STATUS_OK && rank<pipeline->rank_count;
			rank++)
			status = SparkModelResidentClientCanQueueDecision(
				pipeline->clients[rank],transaction->submission_id,
				SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientRecordFailure(transaction,status);
			SparkModelPipelineClientSetFailure(pipeline,status,rank - 1u);
			return(status);
		}
		transaction->decision_kind = SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT;
		transaction->decision_expected_mask = pipeline->all_rank_mask;
		for (rank=pipeline->rank_count; status==SPARK_STATUS_OK && rank!=0u;
			rank--)
			status = SparkModelResidentClientCommit(pipeline->clients[rank - 1u],transaction->submission_id);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientRecordFailure(transaction,status);
			SparkModelPipelineClientSetFailure(pipeline,status,rank);
			return(status);
		}
	}
	else
	{
		for (rank=0u; status==SPARK_STATUS_OK && rank<pipeline->rank_count;
			rank++)
			if ( (transaction->prepared_mask & (UINT32_C(1) << rank)) != 0u )
				status = SparkModelResidentClientCanQueueDecision(
					pipeline->clients[rank],transaction->submission_id,
					SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientSetFailure(pipeline,status,rank - 1u);
			return(status);
		}
		transaction->decision_kind = SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT;
		transaction->decision_expected_mask = transaction->prepared_mask;
		for (rank=0u; status==SPARK_STATUS_OK && rank<pipeline->rank_count;
			rank++)
			if ( (transaction->prepared_mask & (UINT32_C(1) << rank)) != 0u )
				status = SparkModelResidentClientAbort(pipeline->clients[rank],transaction->submission_id);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientSetFailure(pipeline,status,rank - 1u);
			return(status);
		}
	}
	if ( transaction->decision_expected_mask == 0u )
	{
		SparkModelPipelineClientReportResult(pipeline,transaction);
		if ( transaction->status != SPARK_STATUS_OK )
		{
			transaction->completion_mask = pipeline->all_rank_mask;
			SparkModelPipelineClientReportCompletion(pipeline,transaction);
		}
	}
	return(SPARK_STATUS_OK);
}

static void SparkModelPipelineClientRankDecisionResult(
	void *result_context,
	uint64_t submission_id,
	uint32_t decision_kind,
	SparkStatus status)
{
	SparkModelPipelineRankContext *context;
	SparkModelPipelineTransaction *transaction;
	uint32_t rank_mask;
	context = (SparkModelPipelineRankContext *)result_context;
	transaction = context != 0 ? SparkModelPipelineClientFind(context->pipeline,submission_id) : 0;
	rank_mask = context != 0 ? UINT32_C(1) << context->stage_index : 0u;
	if ( transaction == 0 || decision_kind != transaction->decision_kind || (transaction->decision_expected_mask & rank_mask) == 0u || (transaction->decision_result_mask & rank_mask) != 0u )
	{
		if ( context != 0 )
			SparkModelPipelineClientSetFailure(context->pipeline,SPARK_STATUS_SCHEMA_ERROR,context->stage_index);
		return;
	}
	transaction->decision_result_mask |= rank_mask;
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientRecordFailure(transaction,status);
		SparkModelPipelineClientSetFailure(context->pipeline,status,context->stage_index);
	}
	if ( transaction->decision_result_mask != transaction->decision_expected_mask )
		return;
	SparkModelPipelineClientReportResult(context->pipeline,transaction);
	if ( transaction->status != SPARK_STATUS_OK )
		transaction->completion_mask = context->pipeline->all_rank_mask;
	SparkModelPipelineClientReportCompletion(context->pipeline,transaction);
}

static void SparkModelPipelineClientRankResult(
	void *result_context,
	uint64_t submission_id,
	SparkStatus status)
{
	SparkModelPipelineRankContext *context;
	SparkModelPipelineTransaction *transaction;
	uint32_t rank_mask;
	context = (SparkModelPipelineRankContext *)result_context;
	transaction = context != 0 ? SparkModelPipelineClientFind(context->pipeline,submission_id) : 0;
	rank_mask = context != 0 ? UINT32_C(1) << context->stage_index : 0u;
	if ( transaction == 0 || (transaction->result_mask & rank_mask) != 0u )
	{
		if ( context != 0 )
			SparkModelPipelineClientSetFailure(context->pipeline,SPARK_STATUS_SCHEMA_ERROR,context->stage_index);
		return;
	}
	transaction->result_mask |= rank_mask;
	if ( status == SPARK_STATUS_OK )
		transaction->prepared_mask |= rank_mask;
	else
	{
		SparkModelPipelineClientRecordFailure(transaction,status);
		if ( transaction->continued != 0u )
			SparkModelPipelineClientSetFailure(context->pipeline,status,
				context->stage_index);
	}
	(void)SparkModelPipelineClientResolveAdmission(context->pipeline,transaction);
}

static SparkStatus SparkModelPipelineClientValidateRankCompletion(
	const SparkModelPipelineClient *pipeline,
	const SparkModelPipelineTransaction *transaction,
	uint32_t stage_index,
	const SparkModelServingCompletion *completion)
{
	return(SparkModelServingAdapterValidateStageCompletion(pipeline->adapter_descriptor,stage_index,transaction->work_kind,transaction->active_sequence_count,transaction->tokens_per_sequence,&transaction->residency,completion));
}

static void SparkModelPipelineClientReportStageCompletion(
	SparkModelPipelineRankContext *context,
	const SparkModelPipelineTransaction *transaction,
	const SparkModelServingCompletion *completion,
	SparkStatus status)
{
	SparkModelPipelineStageCompletion stage_completion;
	SparkModelPipelineClient *pipeline;
	uint64_t completed_time_ns;
	pipeline = context->pipeline;
	if ( pipeline->stage_completion_function == 0 )
		return;
	memset(&stage_completion,0,sizeof(stage_completion));
	stage_completion.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	stage_completion.descriptor_bytes = SPARK_MODEL_PIPELINE_STAGE_COMPLETION_BYTES;
	stage_completion.stage_index = context->stage_index;
	stage_completion.work_kind = transaction->work_kind;
	stage_completion.active_sequence_count = transaction->active_sequence_count;
	stage_completion.row_count = transaction->row_count;
	stage_completion.status = (uint32_t)status;
	stage_completion.submission_id = completion->submission_id;
	stage_completion.queue_delay_ns = completion->queue_delay_ns;
	stage_completion.service_time_ns = completion->service_time_ns;
	stage_completion.device_memcpy_bytes = completion->device_memcpy_bytes;
	stage_completion.host_staging_bytes = completion->host_staging_bytes;
	completed_time_ns = SparkModelPipelineClientMonotonicNanoseconds();
	if ( completed_time_ns != 0u )
	{
		stage_completion.flags |= SPARK_MODEL_PIPELINE_STAGE_COMPLETION_FLAG_CLIENT_COMPLETION_TIME_VALID;
		stage_completion.client_completion_time_ns = completed_time_ns;
	}
	if ( transaction->submitted_time_ns != 0u && completed_time_ns >= transaction->submitted_time_ns )
	{
		stage_completion.flags |= SPARK_MODEL_PIPELINE_STAGE_COMPLETION_FLAG_CLIENT_ELAPSED_VALID;
		stage_completion.client_elapsed_ns = completed_time_ns - transaction->submitted_time_ns;
	}
	pipeline->stage_completion_function(pipeline->stage_completion_context,&stage_completion);
}

static void SparkModelPipelineClientRankCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	SparkModelPipelineRankContext *context;
	SparkModelPipelineTransaction *transaction;
	SparkStatus stage_status,validation_status;
	uint32_t final_rank,rank_mask;
	context = (SparkModelPipelineRankContext *)completion_context;
	transaction = context != 0 && completion != 0 ? SparkModelPipelineClientFind(context->pipeline,completion->submission_id) : 0;
	rank_mask = context != 0 ? UINT32_C(1) << context->stage_index : 0u;
	if ( transaction == 0 || (transaction->continued == 0u &&
		(transaction->decision_kind != SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT ||
		 transaction->decision_expected_mask != context->pipeline->all_rank_mask)) ||
		(transaction->result_mask & rank_mask) == 0u ||
		(transaction->prepared_mask & rank_mask) == 0u ||
		(transaction->completion_mask & rank_mask) != 0u )
	{
		if ( context != 0 )
			SparkModelPipelineClientSetFailure(context->pipeline,SPARK_STATUS_SCHEMA_ERROR,context->stage_index);
		return;
	}
	stage_status = (SparkStatus)completion->status;
	if ( completion->request_id != transaction->request_id || completion->sequence_id != transaction->sequence_id || completion->sequence_position != transaction->sequence_position || completion->control_generation != transaction->control_generation || completion->transaction_id != transaction->transaction_id || completion->dispatch_generation != transaction->dispatch_generation || completion->request_generation != transaction->request_generation || completion->step_generation != transaction->step_generation )
	{
		SparkModelPipelineClientRecordFailure(transaction,SPARK_STATUS_SCHEMA_ERROR);
		SparkModelPipelineClientSetFailure(context->pipeline,SPARK_STATUS_SCHEMA_ERROR,context->stage_index);
		stage_status = SPARK_STATUS_SCHEMA_ERROR;
	}
	final_rank = context->pipeline->rank_count - 1u;
	validation_status = SparkModelPipelineClientValidateRankCompletion(context->pipeline,transaction,context->stage_index,completion);
	if ( validation_status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientRecordFailure(transaction,validation_status);
		SparkModelPipelineClientSetFailure(context->pipeline,validation_status,context->stage_index);
		if ( stage_status == SPARK_STATUS_OK )
			stage_status = validation_status;
	}
	if ( completion->status != SPARK_STATUS_OK )
		SparkModelPipelineClientRecordFailure(transaction,(SparkStatus)completion->status);
	SparkModelPipelineClientReportStageCompletion(context,transaction,completion,stage_status);
	if ( context->stage_index == final_rank )
		transaction->final_completion = *completion;
	transaction->completion_mask |= rank_mask;
	SparkModelPipelineClientReportCompletion(context->pipeline,transaction);
}

static SparkStatus SparkModelPipelineClientValidateConfiguration(
	const SparkModelPipelineClientConfiguration *configuration)
{
	if ( configuration == 0 )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( configuration->abi_version != SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( configuration->deployment == 0 || configuration->runtime_root == 0 || configuration->connect_timeout_ms == 0u || configuration->completion_function == 0 )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkModelPipelineClientInitializeState(
	const SparkModelPipelineClientConfiguration *configuration,
	SparkModelPipelineClient *pipeline)
{
	char adapter_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	SparkStatus status;
	status = SparkResolveRuntimePath(configuration->runtime_root,configuration->deployment->adapter_shared_object_path,adapter_path,sizeof(adapter_path));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterLoadInterfaceFromSharedObject(adapter_path,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&pipeline->adapter_library);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentValidateForAdapter(configuration->deployment,pipeline->adapter_library.adapter_interface.descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pipeline->rank_count = configuration->deployment->node_count;
	pipeline->transaction_capacity = configuration->deployment->runtime_limits.max_inflight_submission_count;
	pipeline->lease_generation = 1u;
	pipeline->failed_stage_index = SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX;
	pipeline->all_rank_mask = (UINT32_C(1) << pipeline->rank_count) - 1u;
	pipeline->runtime_limits = configuration->deployment->runtime_limits;
	pipeline->adapter_descriptor = pipeline->adapter_library.adapter_interface.descriptor;
	pipeline->submit_result_function = configuration->submit_result_function;
	pipeline->submit_result_context = configuration->submit_result_context;
	pipeline->completion_function = configuration->completion_function;
	pipeline->completion_context = configuration->completion_context;
	pipeline->stage_completion_function = configuration->stage_completion_function;
	pipeline->stage_completion_context = configuration->stage_completion_context;
	pipeline->clients = (SparkModelResidentClient **)calloc(pipeline->rank_count,sizeof(pipeline->clients[0]));
	pipeline->rank_contexts = (SparkModelPipelineRankContext *)calloc(pipeline->rank_count,sizeof(pipeline->rank_contexts[0]));
	pipeline->transactions = (SparkModelPipelineTransaction *)calloc(pipeline->transaction_capacity,sizeof(pipeline->transactions[0]));
	pipeline->transaction_lanes = (SparkModelServingLane *)calloc(
		(uint64_t)pipeline->transaction_capacity *
		pipeline->runtime_limits.max_active_sequence_count,
		sizeof(pipeline->transaction_lanes[0]));
	pipeline->lease_slots = (SparkModelPipelineLeaseSlot *)calloc(
		pipeline->runtime_limits.resident_sequence_capacity,
		sizeof(pipeline->lease_slots[0]));
	return(pipeline->clients != 0 && pipeline->rank_contexts != 0 &&
		pipeline->transactions != 0 && pipeline->transaction_lanes != 0 &&
		pipeline->lease_slots != 0 ? SPARK_STATUS_OK :
		SPARK_STATUS_CAPACITY_EXCEEDED);
}

static SparkStatus SparkModelPipelineClientConnectRank(
	const SparkModelPipelineClientConfiguration *configuration,
	SparkModelPipelineClient *pipeline,
	uint32_t stage)
{
	SparkModelResidentClientConfiguration client_configuration;
	const SparkModelResidentDeploymentNode *node;
	node = SparkModelResidentDeploymentFindStage(configuration->deployment,stage);
	if ( node == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	pipeline->rank_contexts[stage].pipeline = pipeline;
	pipeline->rank_contexts[stage].stage_index = stage;
	memset(&client_configuration,0,sizeof(client_configuration));
	client_configuration.abi_version = SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION;
	client_configuration.descriptor_bytes = SPARK_MODEL_RESIDENT_CLIENT_CONFIGURATION_BYTES;
	client_configuration.rank_index = node->rank_index;
	client_configuration.stage_index = node->stage_index;
	client_configuration.connect_timeout_ms = configuration->connect_timeout_ms;
	client_configuration.runtime_limits = configuration->deployment->runtime_limits;
	client_configuration.endpoint = node->control_endpoint;
	client_configuration.adapter_descriptor = pipeline->adapter_descriptor;
	client_configuration.submit_result_function = SparkModelPipelineClientRankResult;
	client_configuration.submit_result_context = &pipeline->rank_contexts[stage];
	client_configuration.decision_result_function = SparkModelPipelineClientRankDecisionResult;
	client_configuration.decision_result_context = &pipeline->rank_contexts[stage];
	client_configuration.completion_function = SparkModelPipelineClientRankCompletion;
	client_configuration.completion_context = &pipeline->rank_contexts[stage];
	return(SparkModelResidentClientConnect(&client_configuration,&pipeline->clients[stage]));
}

SparkStatus SparkModelPipelineClientConnect(
	const SparkModelPipelineClientConfiguration *configuration,
	SparkModelPipelineClient **pipeline_out)
{
	SparkModelPipelineClient *pipeline;
	SparkStatus status;
	uint32_t stage;
	if ( pipeline_out == 0 )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	*pipeline_out = 0;
	status = SparkModelPipelineClientValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pipeline = (SparkModelPipelineClient *)calloc(1u,sizeof(*pipeline));
	if ( pipeline == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkModelPipelineClientInitializeState(configuration,pipeline);
	for (stage=0u; status==SPARK_STATUS_OK && stage<pipeline->rank_count; stage++)
		status = SparkModelPipelineClientConnectRank(configuration,pipeline,stage);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientDestroy(pipeline);
		return(status);
	}
	*pipeline_out = pipeline;
	return(SPARK_STATUS_OK);
}

void SparkModelPipelineClientDestroy(SparkModelPipelineClient *pipeline)
{
	uint32_t rank;
	if ( pipeline == 0 )
		return;
	if ( pipeline->clients != 0 )
		for (rank=0u; rank<pipeline->rank_count; rank++)
			SparkModelResidentClientDestroy(pipeline->clients[rank]);
	free(pipeline->lease_slots);
	free(pipeline->transaction_lanes);
	free(pipeline->transactions);
	free(pipeline->rank_contexts);
	free(pipeline->clients);
	SparkModelServingAdapterUnloadInterface(&pipeline->adapter_library);
	free(pipeline);
}

static SparkStatus SparkModelPipelineClientPreflight(
	SparkModelPipelineClient *pipeline,
	uint32_t *failed_stage_index_out)
{
	SparkModelResidentClientView view;
	SparkStatus status;
	uint32_t rank;
	*failed_stage_index_out = SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX;
	for (rank=0u; rank<pipeline->rank_count; rank++)
	{
		status = SparkModelResidentClientGetView(pipeline->clients[rank],&view);
		if ( status != SPARK_STATUS_OK || view.connected == 0u )
		{
			*failed_stage_index_out = rank;
			return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_IO_ERROR);
		}
		if ( view.queued_message_count >= view.queue_capacity || view.pending_submission_count >= view.queue_capacity )
			return(SPARK_STATUS_BUSY);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelPipelineClientSubmit(
	SparkModelPipelineClient *pipeline,
	const SparkModelServingSubmission *submission)
{
	SparkModelPipelineTransaction *transaction;
	SparkStatus status;
	uint32_t continuation,failed_stage_index,rank;
	if ( pipeline == 0 || submission == 0 )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( pipeline->failed_status != SPARK_STATUS_OK )
		return((SparkStatus)pipeline->failed_status);
	status = SparkModelServingAdapterValidateRuntimeSubmission(pipeline->adapter_descriptor,&pipeline->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->submission_id <= pipeline->last_submission_id )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	status = SparkModelPipelineClientPreflight(pipeline,&failed_stage_index);
	if ( status != SPARK_STATUS_OK )
	{
		if ( failed_stage_index != SPARK_MODEL_PIPELINE_CLIENT_INVALID_STAGE_INDEX )
			SparkModelPipelineClientSetFailure(pipeline,status,failed_stage_index);
		return(status);
	}
	continuation = SparkModelPipelineClientCanContinue(pipeline,submission);
	if ( continuation != 0u )
	{
		for (rank=0u; status==SPARK_STATUS_OK && rank<pipeline->rank_count;
			rank++)
			status = SparkModelResidentClientCanQueueContinuation(
				pipeline->clients[rank],submission);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	transaction = SparkModelPipelineClientReserve(pipeline,submission);
	if ( transaction == 0 )
		return(SPARK_STATUS_BUSY);
	transaction->continued = continuation;
	for (rank=0u; rank<pipeline->rank_count; rank++)
	{
		status = continuation != 0u ? SparkModelResidentClientContinue(
			pipeline->clients[rank],submission) : SparkModelResidentClientPrepare(
			pipeline->clients[rank],submission);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientRecordFailure(transaction,status);
			SparkModelPipelineClientSetFailure(pipeline,status,rank);
			break;
		}
	}
	pipeline->last_submission_id = submission->submission_id;
	pipeline->submitted_count++;
	if ( continuation != 0u )
		pipeline->continued_count++;
	return(SPARK_STATUS_OK);
}

static void SparkModelPipelineClientFailTransactions(
	SparkModelPipelineClient *pipeline,
	SparkStatus status)
{
	SparkModelPipelineTransaction *transaction;
	SparkModelServingCompletion completion;
	uint64_t submission_id;
	uint32_t report_result;
	uint32_t index;
	for (index=0u; index<pipeline->transaction_capacity; index++)
	{
		transaction = &pipeline->transactions[index];
		if ( transaction->active == 0u )
			continue;
		submission_id = transaction->submission_id;
		report_result = transaction->result_reported == 0u ? 1u : 0u;
		if ( report_result != 0u )
			pipeline->rejected_count++;
		memset(&completion,0,sizeof(completion));
		completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
		completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
		completion.status = status;
		completion.submission_id = submission_id;
		completion.request_id = transaction->request_id;
		completion.sequence_id = transaction->sequence_id;
		completion.sequence_position = transaction->sequence_position;
		completion.control_generation = transaction->control_generation;
		completion.transaction_id = transaction->transaction_id;
		completion.dispatch_generation = transaction->dispatch_generation;
		completion.request_generation = transaction->request_generation;
		completion.step_generation = transaction->step_generation;
		pipeline->completed_count++;
		SparkModelPipelineClientRelease(pipeline,transaction);
		if ( report_result != 0u && pipeline->submit_result_function != 0 )
			pipeline->submit_result_function(pipeline->submit_result_context,submission_id,status);
		pipeline->completion_function(pipeline->completion_context,&completion);
	}
}

SparkStatus SparkModelPipelineClientProgress(
	SparkModelPipelineClient *pipeline,
	uint32_t maximum_message_count_per_rank)
{
	SparkStatus status;
	uint32_t rank;
	if ( pipeline == 0 || maximum_message_count_per_rank == 0u )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	if ( pipeline->failed_status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientFailTransactions(pipeline,(SparkStatus)pipeline->failed_status);
		return((SparkStatus)pipeline->failed_status);
	}
	status = SPARK_STATUS_OK;
	for (rank=pipeline->rank_count; status==SPARK_STATUS_OK && pipeline->failed_status==SPARK_STATUS_OK && rank!=0u; rank--)
		status = SparkModelResidentClientProgress(pipeline->clients[rank - 1u],maximum_message_count_per_rank);
	if ( status != SPARK_STATUS_OK || pipeline->failed_status != SPARK_STATUS_OK )
	{
		if ( pipeline->failed_status == SPARK_STATUS_OK )
			SparkModelPipelineClientSetFailure(pipeline,status,rank);
		SparkModelPipelineClientFailTransactions(pipeline,(SparkStatus)pipeline->failed_status);
		return((SparkStatus)pipeline->failed_status);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelPipelineClientGetPollDescriptors(
	const SparkModelPipelineClient *pipeline,
	SparkModelResidentClientPollDescriptor *descriptors,
	uint32_t descriptor_capacity,
	uint32_t *descriptor_count_out)
{
	SparkStatus status;
	uint32_t rank;
	if ( pipeline == 0 || descriptors == 0 || descriptor_count_out == 0 || descriptor_capacity < pipeline->rank_count )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	status = SPARK_STATUS_OK;
	for (rank=0u; status==SPARK_STATUS_OK && rank<pipeline->rank_count; rank++)
		status = SparkModelResidentClientGetPollDescriptor(pipeline->clients[rank],&descriptors[rank]);
	*descriptor_count_out = status == SPARK_STATUS_OK ? pipeline->rank_count : 0u;
	return(status);
}

SparkStatus SparkModelPipelineClientGetView(
	const SparkModelPipelineClient *pipeline,
	SparkModelPipelineClientView *view)
{
	SparkModelResidentClientView client_view;
	uint32_t rank;
	if ( pipeline == 0 || view == 0 )
		do { fprintf(stderr,"PC_TRACE %d\n",__LINE__); return(SPARK_STATUS_INVALID_ARGUMENT); } while(0);
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	view->descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_VIEW_BYTES;
	view->rank_count = pipeline->rank_count;
	view->active_transaction_count = pipeline->active_transaction_count;
	view->transaction_capacity = pipeline->transaction_capacity;
	view->failed_status = pipeline->failed_status;
	view->failed_stage_index = pipeline->failed_stage_index;
	view->active_continue_lease_count = pipeline->active_continue_lease_count;
	view->submitted_count = pipeline->submitted_count;
	view->continued_count = pipeline->continued_count;
	view->admitted_count = pipeline->admitted_count;
	view->rejected_count = pipeline->rejected_count;
	view->completed_count = pipeline->completed_count;
	for (rank=0u; rank<pipeline->rank_count; rank++)
		if ( SparkModelResidentClientGetView(pipeline->clients[rank],&client_view) == SPARK_STATUS_OK && client_view.connected != 0u )
			view->connected_rank_count++;
	return(SPARK_STATUS_OK);
}

const SparkModelServingAdapterDescriptor *SparkModelPipelineClientGetAdapterDescriptor(
	const SparkModelPipelineClient *pipeline)
{
	return(pipeline != 0 ? pipeline->adapter_descriptor : 0);
}
