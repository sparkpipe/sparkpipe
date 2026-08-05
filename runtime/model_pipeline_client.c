#include "sparkpipe/spark_model_pipeline_client.h"

#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"

typedef struct SparkModelPipelineTransaction
{
	uint32_t active;
	uint32_t work_kind;
	uint32_t active_sequence_count;
	uint32_t result_reported;
	uint32_t status;
	uint32_t result_mask;
	uint32_t prepared_mask;
	uint32_t decision_expected_mask;
	uint32_t decision_result_mask;
	uint32_t decision_kind;
	uint32_t completion_mask;
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
	uint32_t all_rank_mask;
	uint64_t last_submission_id;
	uint64_t submitted_count;
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
	SparkModelResidentClient **clients;
	SparkModelPipelineRankContext *rank_contexts;
	SparkModelPipelineTransaction *transactions;
};

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
			transaction->status = SPARK_STATUS_OK;
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

static void SparkModelPipelineClientReportResult(
	SparkModelPipelineClient *pipeline,
	SparkModelPipelineTransaction *transaction)
{
	if ( transaction->result_reported != 0u || transaction->result_mask != pipeline->all_rank_mask || transaction->decision_kind == 0u || transaction->decision_result_mask != transaction->decision_expected_mask )
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
	if ( transaction->result_reported == 0u || transaction->completion_mask != pipeline->all_rank_mask )
		return;
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
	status = SPARK_STATUS_OK;
	if ( transaction->status == SPARK_STATUS_OK )
	{
		transaction->decision_kind = SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT;
		transaction->decision_expected_mask = pipeline->all_rank_mask;
		for (rank=pipeline->rank_count; status==SPARK_STATUS_OK && rank!=0u; rank--)
			status = SparkModelResidentClientCommit(pipeline->clients[rank - 1u],transaction->submission_id);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientRecordFailure(transaction,status);
			pipeline->failed_status = status;
			return(status);
		}
	}
	else
	{
		transaction->decision_kind = SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT;
		transaction->decision_expected_mask = transaction->prepared_mask;
		for (rank=0u; status==SPARK_STATUS_OK && rank<pipeline->rank_count; rank++)
			if ( (transaction->prepared_mask & (UINT32_C(1) << rank)) != 0u )
				status = SparkModelResidentClientAbort(pipeline->clients[rank],transaction->submission_id);
		if ( status != SPARK_STATUS_OK )
		{
			pipeline->failed_status = status;
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
		if ( context != 0 && context->pipeline->failed_status == SPARK_STATUS_OK )
			context->pipeline->failed_status = SPARK_STATUS_SCHEMA_ERROR;
		return;
	}
	transaction->decision_result_mask |= rank_mask;
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientRecordFailure(transaction,status);
		context->pipeline->failed_status = status;
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
		if ( context != 0 && context->pipeline->failed_status == SPARK_STATUS_OK )
			context->pipeline->failed_status = SPARK_STATUS_SCHEMA_ERROR;
		return;
	}
	transaction->result_mask |= rank_mask;
	if ( status == SPARK_STATUS_OK )
		transaction->prepared_mask |= rank_mask;
	else
		SparkModelPipelineClientRecordFailure(transaction,status);
	(void)SparkModelPipelineClientResolveAdmission(context->pipeline,transaction);
}

static SparkStatus SparkModelPipelineClientValidateRankCompletion(
	const SparkModelPipelineClient *pipeline,
	const SparkModelPipelineTransaction *transaction,
	uint32_t stage_index,
	const SparkModelServingCompletion *completion)
{
	return(SparkModelServingAdapterValidateStageCompletion(pipeline->adapter_descriptor,stage_index,transaction->work_kind,transaction->active_sequence_count,&transaction->residency,completion));
}

static void SparkModelPipelineClientRankCompletion(
	void *completion_context,
	const SparkModelServingCompletion *completion)
{
	SparkModelPipelineRankContext *context;
	SparkModelPipelineTransaction *transaction;
	SparkStatus validation_status;
	uint32_t final_rank,rank_mask;
	context = (SparkModelPipelineRankContext *)completion_context;
	transaction = context != 0 && completion != 0 ? SparkModelPipelineClientFind(context->pipeline,completion->submission_id) : 0;
	rank_mask = context != 0 ? UINT32_C(1) << context->stage_index : 0u;
	if ( transaction == 0 || transaction->decision_kind != SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT || transaction->decision_expected_mask != context->pipeline->all_rank_mask || (transaction->result_mask & rank_mask) == 0u || (transaction->prepared_mask & rank_mask) == 0u || (transaction->completion_mask & rank_mask) != 0u )
	{
		if ( context != 0 && context->pipeline->failed_status == SPARK_STATUS_OK )
			context->pipeline->failed_status = SPARK_STATUS_SCHEMA_ERROR;
		return;
	}
	if ( completion->request_id != transaction->request_id || completion->sequence_id != transaction->sequence_id || completion->sequence_position != transaction->sequence_position || completion->control_generation != transaction->control_generation || completion->transaction_id != transaction->transaction_id || completion->dispatch_generation != transaction->dispatch_generation || completion->request_generation != transaction->request_generation || completion->step_generation != transaction->step_generation )
	{
		SparkModelPipelineClientRecordFailure(transaction,SPARK_STATUS_SCHEMA_ERROR);
		context->pipeline->failed_status = SPARK_STATUS_SCHEMA_ERROR;
	}
	final_rank = context->pipeline->rank_count - 1u;
	validation_status = SparkModelPipelineClientValidateRankCompletion(context->pipeline,transaction,context->stage_index,completion);
	if ( validation_status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientRecordFailure(transaction,validation_status);
		context->pipeline->failed_status = validation_status;
	}
	if ( completion->status != SPARK_STATUS_OK )
		SparkModelPipelineClientRecordFailure(transaction,(SparkStatus)completion->status);
	if ( context->stage_index == final_rank )
		transaction->final_completion = *completion;
	transaction->completion_mask |= rank_mask;
	SparkModelPipelineClientReportCompletion(context->pipeline,transaction);
}

static SparkStatus SparkModelPipelineClientValidateConfiguration(
	const SparkModelPipelineClientConfiguration *configuration)
{
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( configuration->deployment == 0 || configuration->runtime_root == 0 || configuration->connect_timeout_ms == 0u || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
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
		status = SparkModelServingAdapterLoadInterfaceFromSharedObject(adapter_path,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT,&pipeline->adapter_library);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentValidateForAdapter(configuration->deployment,pipeline->adapter_library.adapter_interface.descriptor);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pipeline->rank_count = configuration->deployment->node_count;
	pipeline->transaction_capacity = configuration->deployment->runtime_limits.max_inflight_submission_count;
	pipeline->all_rank_mask = (UINT32_C(1) << pipeline->rank_count) - 1u;
	pipeline->runtime_limits = configuration->deployment->runtime_limits;
	pipeline->adapter_descriptor = pipeline->adapter_library.adapter_interface.descriptor;
	pipeline->submit_result_function = configuration->submit_result_function;
	pipeline->submit_result_context = configuration->submit_result_context;
	pipeline->completion_function = configuration->completion_function;
	pipeline->completion_context = configuration->completion_context;
	pipeline->clients = (SparkModelResidentClient **)calloc(pipeline->rank_count,sizeof(pipeline->clients[0]));
	pipeline->rank_contexts = (SparkModelPipelineRankContext *)calloc(pipeline->rank_count,sizeof(pipeline->rank_contexts[0]));
	pipeline->transactions = (SparkModelPipelineTransaction *)calloc(pipeline->transaction_capacity,sizeof(pipeline->transactions[0]));
	return(pipeline->clients != 0 && pipeline->rank_contexts != 0 && pipeline->transactions != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED);
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
		return(SPARK_STATUS_INVALID_ARGUMENT);
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
	free(pipeline->transactions);
	free(pipeline->rank_contexts);
	free(pipeline->clients);
	SparkModelServingAdapterUnloadInterface(&pipeline->adapter_library);
	free(pipeline);
}

static SparkStatus SparkModelPipelineClientPreflight(
	SparkModelPipelineClient *pipeline)
{
	SparkModelResidentClientView view;
	SparkStatus status;
	uint32_t rank;
	for (rank=0u; rank<pipeline->rank_count; rank++)
	{
		status = SparkModelResidentClientGetView(pipeline->clients[rank],&view);
		if ( status != SPARK_STATUS_OK || view.connected == 0u )
			return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_IO_ERROR);
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
	uint32_t rank;
	if ( pipeline == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pipeline->failed_status != SPARK_STATUS_OK )
		return((SparkStatus)pipeline->failed_status);
	status = SparkModelServingAdapterValidateRuntimeSubmission(pipeline->adapter_descriptor,&pipeline->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->submission_id <= pipeline->last_submission_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelPipelineClientPreflight(pipeline);
	if ( status != SPARK_STATUS_OK )
		return(status);
	transaction = SparkModelPipelineClientReserve(pipeline,submission);
	if ( transaction == 0 )
		return(SPARK_STATUS_BUSY);
	for (rank=0u; rank<pipeline->rank_count; rank++)
	{
		status = SparkModelResidentClientPrepare(pipeline->clients[rank],submission);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelPipelineClientRecordFailure(transaction,status);
			pipeline->failed_status = status;
			break;
		}
	}
	pipeline->last_submission_id = submission->submission_id;
	pipeline->submitted_count++;
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
		return(SPARK_STATUS_INVALID_ARGUMENT);
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
			pipeline->failed_status = status;
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
		return(SPARK_STATUS_INVALID_ARGUMENT);
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
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	view->descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_VIEW_BYTES;
	view->rank_count = pipeline->rank_count;
	view->active_transaction_count = pipeline->active_transaction_count;
	view->transaction_capacity = pipeline->transaction_capacity;
	view->failed_status = pipeline->failed_status;
	view->submitted_count = pipeline->submitted_count;
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
