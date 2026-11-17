#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_model_serving_adapter.h"

#define HOST_TURN_RANKS 16u
#define HOST_TURN_PREPARED 64u

typedef struct HostTurnPrepared
{
	uint32_t active;
	uint32_t committed;
	uint64_t submission_id;
} HostTurnPrepared;

typedef struct HostTurnState
{
	SparkModelServingRuntimeLimits runtime_limits;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	uint32_t stage_index;
	uint32_t frame_us;
	uint64_t reset_generation;
	uint64_t submitted_count;
	uint64_t completed_count;
	HostTurnPrepared prepared[HOST_TURN_PREPARED];
} HostTurnState;

static const SparkModelServingAdapterDescriptor HostTurnDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS,
	.stage_count = HOST_TURN_RANKS,
	.layer_count = 64u,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = 5120u,
	.boundary_element_bytes = 2u,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = 16u,
	.max_active_sequence_count = 128u,
	.max_input_row_count = 2048u,
	.max_resident_sequence_count = 128u,
	.max_output_token_count = 1024u,
	.max_speculative_token_count = 0u,
	.adapter_id = "host.turn.bench.tp16.v1",
	.model_id = "host-turn-bench/model",
	.model_revision = "bench",
	.driver_program_name = "resident_decode",
	.artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
	.stage_layer_counts = {64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u},
	.minimum_efficient_submission_row_count = 1u,
	.cache_block_token_count = 64u,
	.parallel_group_size = 0u,
	.cache_checkpoint_token_count = 0u,
	.max_verify_row_count = 0u,
	.tree_slot_grain = 0u
};

static SparkStatus HostTurnInitialize(const SparkModelServingAdapterConfiguration *configuration,void **adapter_state)
{
	HostTurnState *state;
	FILE *stream;
	uint32_t parsed;
	if ( configuration == 0 || adapter_state == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = (HostTurnState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->runtime_limits = configuration->runtime_limits;
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	state->stage_index = configuration->stage_index;
	stream = configuration->adapter_configuration_path != 0 ? fopen(configuration->adapter_configuration_path,"rb") : 0;
	parsed = stream != 0 && fscanf(stream,"{\"frame_us\":%u}",&state->frame_us) == 1 ? 1u : 0u;
	if ( stream != 0 )
		fclose(stream);
	if ( parsed == 0u )
	{
		free(state);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static void HostTurnDestroy(void *adapter_state)
{
	free(adapter_state);
}

static SparkStatus HostTurnValidateSubmission(void *adapter_state,const SparkModelServingSubmission *submission)
{
	HostTurnState *state = (HostTurnState *)adapter_state;
	if ( state == 0 || submission == 0 || submission->control_generation < state->reset_generation )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) != 0u &&
		(submission->row_count == 0u || submission->row_count > state->runtime_limits.max_input_row_count || submission->row_sequence_ids == 0 || submission->lanes == 0) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static HostTurnPrepared *HostTurnFind(HostTurnState *state,uint64_t submission_id)
{
	uint32_t index;
	for (index=0u; index<HOST_TURN_PREPARED; index++)
		if ( state->prepared[index].active != 0u && state->prepared[index].submission_id == submission_id )
			return(&state->prepared[index]);
	return(0);
}

static SparkStatus HostTurnPrefetch(void *adapter_state,const SparkModelServingSubmission *submissions,uint32_t submission_count)
{
	HostTurnState *state = (HostTurnState *)adapter_state;
	uint32_t index;
	SparkStatus status;
	if ( state == 0 || submissions == 0 || submission_count != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = HostTurnValidateSubmission(adapter_state,submissions);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( HostTurnFind(state,submissions->submission_id) != 0 )
		return(SPARK_STATUS_OK);
	for (index=0u; index<HOST_TURN_PREPARED; index++)
		if ( state->prepared[index].active == 0u )
		{
			state->prepared[index].active = 1u;
			state->prepared[index].committed = 0u;
			state->prepared[index].submission_id = submissions->submission_id;
			return(SPARK_STATUS_OK);
		}
	return(SPARK_STATUS_BUSY);
}

static SparkStatus HostTurnResolvePrefetch(void *adapter_state,const SparkModelServingSubmission *submission,uint32_t resolution)
{
	HostTurnState *state = (HostTurnState *)adapter_state;
	HostTurnPrepared *prepared;
	if ( state == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	prepared = HostTurnFind(state,submission->submission_id);
	if ( prepared == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT )
		prepared->committed = 1u;
	else
		memset(prepared,0,sizeof(*prepared));
	return(SPARK_STATUS_OK);
}

static SparkStatus HostTurnSubmit(void *adapter_state,const SparkModelServingSubmission *submission)
{
	HostTurnState *state = (HostTurnState *)adapter_state;
	SparkModelServingCompletion completion;
	HostTurnPrepared *prepared;
	struct timespec started,finished;
	uint32_t lane,step,token_index;
	SparkStatus status;
	status = HostTurnValidateSubmission(adapter_state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	prepared = HostTurnFind(state,submission->submission_id);
	if ( prepared == 0 || prepared->committed == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	memset(prepared,0,sizeof(*prepared));
	state->submitted_count++;
	clock_gettime(CLOCK_MONOTONIC,&started);
	if ( state->frame_us != 0u )
	{
		struct timespec delay;
		delay.tv_sec = state->frame_us / 1000000u;
		delay.tv_nsec = (long)(state->frame_us % 1000000u) * 1000L;
		nanosleep(&delay,0);
	}
	clock_gettime(CLOCK_MONOTONIC,&finished);
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = SPARK_STATUS_OK;
	completion.submission_id = submission->submission_id;
	completion.request_id = submission->request_id;
	completion.sequence_id = submission->sequence_id;
	completion.sequence_position = submission->sequence_position;
	completion.control_generation = submission->control_generation;
	completion.transaction_id = submission->transaction_id;
	completion.dispatch_generation = submission->dispatch_generation;
	completion.request_generation = submission->request_generation;
	completion.step_generation = submission->step_generation;
	completion.residency = submission->residency;
	completion.service_time_ns = (uint64_t)(finished.tv_sec - started.tv_sec) * UINT64_C(1000000000) + (uint64_t)(finished.tv_nsec - started.tv_nsec) + 1u;
	if ( state->stage_index + 1u == HostTurnDescriptor.stage_count && SparkModelServingWorkKindUsesRows(submission->work_kind) != 0u )
	{
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		completion.tokens_per_sequence = submission->tokens_per_sequence;
		completion.token_count = submission->active_sequence_count * submission->tokens_per_sequence;
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			for (step=0u; step<submission->tokens_per_sequence; step++)
			{
				token_index = lane * submission->tokens_per_sequence + step;
				completion.token_ids[token_index] = (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u ?
					1000u + (uint32_t)((submission->lanes[lane].sequence_position + step + lane) % 4096u) : 0u;
			}
	}
	state->completed_count++;
	state->completion_function(state->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

static SparkStatus HostTurnProgress(void *adapter_state,uint32_t maximum_step_count)
{
	(void)adapter_state;
	(void)maximum_step_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus HostTurnQuiesce(void *adapter_state,uint64_t deadline_time_ns)
{
	(void)adapter_state;
	(void)deadline_time_ns;
	return(SPARK_STATUS_OK);
}

static SparkStatus HostTurnSnapshot(void *adapter_state,SparkModelServingAdapterSnapshot *snapshot)
{
	HostTurnState *state = (HostTurnState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	snapshot->available_submission_count = state->runtime_limits.max_inflight_submission_count;
	snapshot->submitted_count = state->submitted_count;
	snapshot->completed_count = state->completed_count;
	snapshot->max_sequence_positions = 131072u;
	return(SPARK_STATUS_OK);
}

static SparkStatus HostTurnReset(void *adapter_state,uint64_t control_generation)
{
	HostTurnState *state = (HostTurnState *)adapter_state;
	if ( state == 0 || control_generation == 0u || control_generation <= state->reset_generation )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(state->prepared,0,sizeof(state->prepared));
	state->reset_generation = control_generation;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface HostTurnInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &HostTurnDescriptor,
	.initialize = HostTurnInitialize,
	.destroy = HostTurnDestroy,
	.validate_submission = HostTurnValidateSubmission,
	.submit = HostTurnSubmit,
	.prefetch = HostTurnPrefetch,
	.resolve_prefetch = HostTurnResolvePrefetch,
	.progress = HostTurnProgress,
	.quiesce = HostTurnQuiesce,
	.snapshot = HostTurnSnapshot,
	.reset = HostTurnReset
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&HostTurnInterface);
}
