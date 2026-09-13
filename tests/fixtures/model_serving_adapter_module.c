#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_model_serving_adapter.h"

typedef struct TestModelServingPrepared
{
	uint32_t active;
	uint32_t committed;
	uint32_t busy_returned;
	uint64_t submission_id;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
} TestModelServingPrepared;

typedef struct TestModelServingState
{
	SparkModelServingRuntimeLimits runtime_limits;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	uint32_t stage_index;
	uint32_t quiescing;
	uint32_t continuation_busy_returned;
	uint64_t submitted_count;
	uint64_t completed_count;
	uint64_t rejected_count;
	TestModelServingPrepared prepared[4];
} TestModelServingState;

static const SparkModelServingAdapterDescriptor TestModelServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN,
	.stage_count = 3u,
	.layer_count = 7u,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = 64u,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_INT8,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 4u,
	.max_active_sequence_count = 32u,
	.max_input_row_count = 256u,
	.max_resident_sequence_count = 256u,
	.max_output_token_count = 256u,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE,
	.adapter_id = "test.model.serving.adapter.v1",
	.model_id = "test/model",
	.model_revision = "test-revision",
	.driver_program_name = "resident_decode",
	.artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	.stage_layer_counts = {2u,3u,2u},
	.boundary_sideband_kinds = {1u,0u,0u},
	.boundary_sideband_bytes_per_sequence = {16u,0u,0u},
	.minimum_efficient_submission_row_count = 16u,
	.cache_block_token_count = 4u
};

static SparkStatus TestModelServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&TestModelServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= TestModelServingDescriptor.stage_count || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,TestModelServingDescriptor.driver_program_name) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	TestModelServingState *state;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = TestModelServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (TestModelServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->runtime_limits = configuration->runtime_limits;
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	state->stage_index = configuration->stage_index;
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static void TestModelServingDestroy(void *adapter_state)
{
	free(adapter_state);
}

static SparkStatus TestModelServingValidateBoundaries(
	const TestModelServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t required_bytes;
	uint64_t input_sideband_bytes,output_sideband_bytes;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(submission->hidden_input_address == 0 && submission->boundary_sideband_input_address == 0 && submission->hidden_output_address == 0 && submission->boundary_sideband_output_address == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	required_bytes = (uint64_t)submission->row_count * TestModelServingDescriptor.boundary_element_count * TestModelServingDescriptor.boundary_element_bytes;
	input_sideband_bytes = state->stage_index != 0u ? (uint64_t)submission->row_count * TestModelServingDescriptor.boundary_sideband_bytes_per_sequence[state->stage_index - 1u] : 0u;
	output_sideband_bytes = state->stage_index + 1u < TestModelServingDescriptor.stage_count ? (uint64_t)submission->row_count * TestModelServingDescriptor.boundary_sideband_bytes_per_sequence[state->stage_index] : 0u;
	if ( state->stage_index == 0u && submission->hidden_input_address != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->stage_index != 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes != required_bytes) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (submission->boundary_sideband_input_address != 0) != (input_sideband_bytes != 0u) || submission->boundary_sideband_input_bytes != input_sideband_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->stage_index + 1u == TestModelServingDescriptor.stage_count && submission->hidden_output_address != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->stage_index + 1u < TestModelServingDescriptor.stage_count && (submission->hidden_output_address == 0 || submission->hidden_output_bytes != required_bytes) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (submission->boundary_sideband_output_address != 0) != (output_sideband_bytes != 0u) || submission->boundary_sideband_output_bytes != output_sideband_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	TestModelServingState *state;
	SparkStatus status;
	state = (TestModelServingState *)adapter_state;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&TestModelServingDescriptor,&state->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->model_extension_kind == 77u && submission->model_extension_bytes == 1u )
		return(state->stage_index == 1u ? SPARK_STATUS_UNSUPPORTED : SPARK_STATUS_OK);
	if ( submission->model_extension_kind == 88u && submission->model_extension_bytes == 1u )
		return(SPARK_STATUS_OK);
	if ( submission->model_extension_kind == 99u && submission->model_extension_bytes == 1u )
		return(SPARK_STATUS_OK);
	if ( submission->model_extension_kind == 98u && submission->model_extension_bytes == 1u )
		return(SPARK_STATUS_OK);
	if ( submission->model_extension_bytes != 0u || submission->model_extension_kind != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static void TestModelServingBuildCompletion(
	const TestModelServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelServingCompletion *completion)
{
	uint32_t lane,step,token_index;
	memset(completion,0,sizeof(*completion));
	completion->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion->status = SPARK_STATUS_OK;
	completion->submission_id = submission->submission_id;
	completion->request_id = submission->request_id;
	completion->sequence_id = submission->sequence_id;
	completion->sequence_position = submission->sequence_position;
	completion->control_generation = submission->control_generation;
	completion->transaction_id = submission->transaction_id;
	completion->dispatch_generation = submission->dispatch_generation;
	completion->request_generation = submission->request_generation;
	completion->step_generation = submission->step_generation;
	completion->residency = submission->residency;
	completion->queue_delay_ns = state->stage_index + 1u;
	completion->service_time_ns = (uint64_t)(state->stage_index + 1u) * 10u;
	completion->device_memcpy_bytes = (uint64_t)(state->stage_index + 1u) * 100u;
	completion->host_staging_bytes = (uint64_t)(state->stage_index + 1u) * 1000u;
	if ( state->stage_index + 1u != TestModelServingDescriptor.stage_count || submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE || submission->model_extension_kind == 88u )
		return;
	completion->completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
	completion->tokens_per_sequence = submission->tokens_per_sequence;
	completion->token_count = submission->active_sequence_count *
		submission->tokens_per_sequence;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		for (step=0u; step<submission->tokens_per_sequence; step++)
		{
			token_index = lane * submission->tokens_per_sequence + step;
			completion->token_ids[token_index] =
				(submission->lanes[lane].flags &
				 SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u ?
				(submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ?
				 4203u : 4200u) + lane + step : 0u;
		}
}

static uint32_t TestModelServingPreparedIdentityMatches(
	const TestModelServingPrepared *prepared,
	const SparkModelServingSubmission *submission)
{
	return(prepared->active != 0u &&
		prepared->submission_id == submission->submission_id &&
		prepared->control_generation == submission->control_generation &&
		prepared->transaction_id == submission->transaction_id &&
		prepared->dispatch_generation == submission->dispatch_generation &&
		prepared->request_generation == submission->request_generation &&
		prepared->step_generation == submission->step_generation ? 1u : 0u);
}

static SparkStatus TestModelServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	TestModelServingState *state;
	SparkModelServingCompletion completion;
	SparkStatus status;
	uint32_t index;
	state = (TestModelServingState *)adapter_state;
	status = TestModelServingValidateSubmission(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = TestModelServingValidateBoundaries(state,submission);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state != 0 )
			state->rejected_count++;
		return(status);
	}
	for (index=0u; index<sizeof(state->prepared) / sizeof(state->prepared[0]);
		index++)
		if ( state->prepared[index].active != 0u &&
			state->prepared[index].submission_id == submission->submission_id )
			break;
	if ( index == sizeof(state->prepared) / sizeof(state->prepared[0]) ||
		TestModelServingPreparedIdentityMatches(&state->prepared[index],
			submission) == 0u ||
		state->prepared[index].committed == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( submission->model_extension_kind == 99u &&
		state->prepared[index].busy_returned == 0u )
	{
		state->prepared[index].busy_returned = 1u;
		return(SPARK_STATUS_BUSY);
	}
	state->submitted_count++;
	if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE && submission->hidden_output_address != 0 )
	{
		if ( submission->hidden_input_address != 0 )
			memcpy(submission->hidden_output_address,submission->hidden_input_address,(size_t)submission->hidden_output_bytes);
		else
			memset(submission->hidden_output_address,0x2c,(size_t)submission->hidden_output_bytes);
	}
	if ( submission->boundary_sideband_output_address != 0 )
		memset(submission->boundary_sideband_output_address,0x5a,(size_t)submission->boundary_sideband_output_bytes);
	TestModelServingBuildCompletion(state,submission,&completion);
	state->completed_count++;
	state->completion_function(state->completion_context,&completion);
	memset(&state->prepared[index],0,sizeof(state->prepared[index]));
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelServingPrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submissions,
	uint32_t submission_count)
{
	TestModelServingState *state;
	SparkStatus status;
	uint32_t free_index,index;
	if ( adapter_state == 0 || submissions == 0 || submission_count != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (TestModelServingState *)adapter_state;
	status = TestModelServingValidateSubmission(adapter_state,submissions);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submissions->model_extension_kind == 98u &&
		state->continuation_busy_returned == 0u )
	{
		state->continuation_busy_returned = 1u;
		return(SPARK_STATUS_BUSY);
	}
	free_index = UINT32_MAX;
	for (index=0u; index<sizeof(state->prepared) / sizeof(state->prepared[0]);
		index++)
	{
		if ( state->prepared[index].active == 0u && free_index == UINT32_MAX )
			free_index = index;
		if ( state->prepared[index].active != 0u &&
			state->prepared[index].submission_id == submissions->submission_id )
			return(TestModelServingPreparedIdentityMatches(
				&state->prepared[index],submissions) == 0u ?
				SPARK_STATUS_SCHEMA_ERROR :
				state->prepared[index].committed == 0u ? SPARK_STATUS_OK :
				SPARK_STATUS_DUPLICATE);
	}
	if ( free_index == UINT32_MAX )
		return(SPARK_STATUS_BUSY);
	state->prepared[free_index].active = 1u;
	state->prepared[free_index].submission_id = submissions->submission_id;
	state->prepared[free_index].control_generation =
		submissions->control_generation;
	state->prepared[free_index].transaction_id = submissions->transaction_id;
	state->prepared[free_index].dispatch_generation =
		submissions->dispatch_generation;
	state->prepared[free_index].request_generation =
		submissions->request_generation;
	state->prepared[free_index].step_generation = submissions->step_generation;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelServingResolvePrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submission,
	uint32_t resolution)
{
	TestModelServingState *state;
	uint32_t index;
	if ( adapter_state == 0 || submission == 0 ||
		(resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		 resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (TestModelServingState *)adapter_state;
	for (index=0u; index<sizeof(state->prepared) / sizeof(state->prepared[0]);
		index++)
		if ( state->prepared[index].active != 0u &&
			state->prepared[index].submission_id == submission->submission_id )
			break;
	if ( index == sizeof(state->prepared) / sizeof(state->prepared[0]) ||
		state->prepared[index].committed != 0u ||
		TestModelServingPreparedIdentityMatches(&state->prepared[index],
			submission) == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT )
		state->prepared[index].committed = 1u;
	else
		memset(&state->prepared[index],0,sizeof(state->prepared[index]));
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	return(adapter_state != 0 && maximum_step_count != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus TestModelServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	TestModelServingState *state;
	state = (TestModelServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestModelServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	TestModelServingState *state;
	state = (TestModelServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	snapshot->available_submission_count = state->quiescing == 0u ? state->runtime_limits.max_inflight_submission_count : 0u;
	snapshot->submitted_count = state->submitted_count;
	snapshot->completed_count = state->completed_count;
	snapshot->rejected_count = state->rejected_count;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface TestModelServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &TestModelServingDescriptor,
	.initialize = TestModelServingInitialize,
	.destroy = TestModelServingDestroy,
	.validate_submission = TestModelServingValidateSubmission,
	.submit = TestModelServingSubmit,
	.prefetch = TestModelServingPrefetch,
	.resolve_prefetch = TestModelServingResolvePrefetch,
	.progress = TestModelServingProgress,
	.quiesce = TestModelServingQuiesce,
	.snapshot = TestModelServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&TestModelServingInterface);
}
