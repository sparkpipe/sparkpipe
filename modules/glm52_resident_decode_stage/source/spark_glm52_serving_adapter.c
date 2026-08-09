#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm52_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_driver_support.h"

#ifndef GLM52_EXPERT_WEIGHT_CODEC
#error "GLM52_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM52_EXPERT_CODEC_NAME
#error "GLM52_EXPERT_CODEC_NAME must name the exact package expert codec"
#endif
#ifndef GLM52_MODEL_REVISION
#error "GLM52_MODEL_REVISION must name the exact source snapshot"
#endif
#ifndef GLM52_CONTRACT_SHA256
#error "GLM52_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_GLM52_SERVING_ADAPTER_ID \
	"spark.glm52.serving-adapter.pp13.expert_" GLM52_EXPERT_CODEC_NAME ".v1"
#define SPARK_GLM52_SERVING_MODEL_ID "zai-org/GLM-5.2"
#define SPARK_GLM52_SERVING_DRIVER_MODEL_ID \
	"zai.glm-5.2.resident-decode-stage-firmware"
#define SPARK_GLM52_SERVING_STAGE_NAME "glm52_resident_decode_stage"
#define SPARK_GLM52_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_GLM52_SERVING_TARGET \
	"cuda.sm121.glm52.resident_decode_stage.bf16.expert_" GLM52_EXPERT_CODEC_NAME
#define SPARK_GLM52_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL)

static const char *const SparkGlm52ServingConfigurationMembers[] =
{
	"schema_version",
	"model_revision",
	"expert_weight_codec",
	"stage_pack_path",
	"max_sequence_positions",
	"execution_row_capacity"
};

typedef struct SparkGlm52ServingPending
{
	struct SparkGlm52ServingState *owner;
	uint32_t active;
	uint32_t row_count;
	uint32_t lane_count;
	uint32_t active_sequence_count;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	uint32_t last_row_by_lane[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
} SparkGlm52ServingPending;

typedef struct SparkGlm52ServingState
{
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkGlm52ResidentDecodeStageNodeContext node_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	SparkModelServingWakeFunction wake_function;
	void *wake_context;
	void *execution_stream;
	char stage_pack_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t stage_index;
	uint32_t pipeline_slot_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t quiescing;
	uint64_t orphan_completion_count;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkGlm52ServingPending pending[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkGlm52ServingState;

static const SparkModelServingAdapterDescriptor SparkGlm52ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV,
	.stage_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT,
	.layer_count = SPARK_GLM52_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT,
	.boundary_element_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequence_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.adapter_id = SPARK_GLM52_SERVING_ADAPTER_ID,
	.model_id = SPARK_GLM52_SERVING_MODEL_ID,
	.model_revision = GLM52_MODEL_REVISION,
	.driver_program_name = SPARK_GLM52_SERVING_PROGRAM_NAME,
	.artifact_sha256 = GLM52_CONTRACT_SHA256,
	.stage_layer_counts = {6u,6u,6u,6u,6u,6u,6u,6u,6u,6u,6u,6u,6u},
	.boundary_sideband_kinds = {0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_KIND,0u},
	.boundary_sideband_bytes_per_sequence = {0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW,0u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW,0u}
};

static int32_t SparkGlm52ServingJsonMember(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,root,name));
}

static SparkStatus SparkGlm52ServingJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkGlm52ServingJsonMember(document,root,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkGlm52ServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkGlm52ServingState *state,
	uint32_t *max_sequence_positions,
	uint32_t *execution_row_capacity)
{
	SparkJsonDocument document;
	char *relative_stage_pack_path;
	uint32_t schema_version;
	int32_t root,token;
	SparkStatus status;
	relative_stage_pack_path = 0;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkGlm52ServingConfigurationMembers,(uint32_t)(sizeof(SparkGlm52ServingConfigurationMembers) / sizeof(SparkGlm52ServingConfigurationMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_GLM52_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkGlm52ServingJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM52_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkGlm52ServingJsonMember(&document,root,"expert_weight_codec") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM52_EXPERT_CODEC_NAME)) )
		status = SPARK_STATUS_TARGET_MISMATCH;
	token = status == SPARK_STATUS_OK ? SparkGlm52ServingJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingJsonUnsigned(&document,root,"execution_row_capacity",execution_row_capacity);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	return(status);
}

static SparkStatus SparkGlm52ServingValidateRowOrder(
	const SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t last_position[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->node_context.max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
		counts[lane]++;
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	maximum = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= submission->row_count || submission->row_lane_indices[row++] != lane) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == submission->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkGlm52ServingPending *SparkGlm52ServingReservePending(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm52ServingPending *pending;
	uint32_t index,lane,row;
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		pending = &state->pending[index];
		if ( pending->active == 0u )
		{
			memset(pending,0,sizeof(*pending));
			pending->owner = state;
			pending->active = 1u;
			pending->row_count = submission->row_count;
			pending->lane_count = submission->lane_count;
			pending->active_sequence_count = submission->active_sequence_count;
			pending->submission_id = submission->submission_id;
			pending->request_id = submission->request_id;
			pending->sequence_id = submission->sequence_id;
			pending->sequence_position = submission->sequence_position;
			pending->control_generation = submission->control_generation;
			pending->transaction_id = submission->transaction_id;
			pending->dispatch_generation = submission->dispatch_generation;
			pending->request_generation = submission->request_generation;
			pending->step_generation = submission->step_generation;
			for (row=0u; row<submission->row_count; row++)
			{
				lane = submission->row_lane_indices[row];
				pending->last_row_by_lane[lane] = row;
				pending->resident_slots[row] = submission->lanes[lane].resident_sequence_slot;
			}
			return(pending);
		}
	}
	return(0);
}

static void SparkGlm52ServingOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkGlm52ServingState *state;
	(void)driver_completion;
	state = (SparkGlm52ServingState *)completion_context;
	if ( state != 0 )
		state->orphan_completion_count++;
}

static void SparkGlm52ServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkGlm52ServingPending *pending;
	SparkGlm52ServingState *state;
	SparkModelServingCompletion completion;
	uint32_t index,matches;
	pending = (SparkGlm52ServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || pending->active == 0u || driver_completion == 0 )
		return;
	matches = driver_completion->request_id == pending->request_id && driver_completion->sequence_id == pending->sequence_id && driver_completion->sequence_position == pending->sequence_position && driver_completion->program_id == state->program->program_id;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = matches != 0u ? (uint32_t)driver_completion->status : SPARK_STATUS_SCHEMA_ERROR;
	completion.submission_id = pending->submission_id;
	completion.request_id = pending->request_id;
	completion.sequence_id = pending->sequence_id;
	completion.sequence_position = pending->sequence_position;
	completion.control_generation = pending->control_generation;
	completion.transaction_id = pending->transaction_id;
	completion.dispatch_generation = pending->dispatch_generation;
	completion.request_generation = pending->request_generation;
	completion.step_generation = pending->step_generation;
	completion.accepted_token_count = driver_completion->accepted_token_count;
	completion.queue_delay_ns = driver_completion->queue_delay_ns;
	completion.service_time_ns = driver_completion->service_time_ns;
	completion.device_memcpy_bytes = driver_completion->device_memcpy_bytes;
	completion.host_staging_bytes = driver_completion->host_staging_bytes;
	if ( matches != 0u )
		completion.residency = driver_completion->residency;
	else
		state->orphan_completion_count++;
	if ( state->stage_index + 1u == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT && completion.status == SPARK_STATUS_OK )
	{
		completion.token_count = pending->active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index]];
	}
	pending->active = 0u;
	state->completion_function(state->completion_context,&completion);
}

static void SparkGlm52ServingDriverWake(void *wake_context)
{
	SparkGlm52ServingState *state;
	state = (SparkGlm52ServingState *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}

static uint32_t SparkGlm52ServingAvailableSubmissionCount(
	const SparkGlm52ServingState *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += state->pending[index].active == 0u ? 1u : 0u;
	return(available);
}

static void SparkGlm52ServingDestroy(void *adapter_state)
{
	SparkGlm52ServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkGlm52ServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		return;
	if ( state->driver.interface != 0 && state->driver.interface->snapshot != 0 && state->driver_instance != 0 && state->program != 0 )
	{
		memset(&snapshot,0,sizeof(snapshot));
		if ( state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot) != SPARK_STATUS_OK || snapshot.active_submission_count != 0u )
			return;
	}
	if ( state->driver.interface != 0 && state->driver.interface->destroy != 0 && state->driver_instance != 0 )
		state->driver.interface->destroy(state->driver_instance);
	SparkUnloadModelDriver(&state->driver);
	free(state);
}

static SparkStatus SparkGlm52ServingLoadDriver(
	SparkGlm52ServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	const SparkModelDriverDescriptor *descriptor;
	SparkModelDriverCreateRequest request;
	char error_buffer[512];
	SparkStatus status;
	SparkLoadedModelDriverReset(&state->driver);
	status = SparkLoadModelDriver(configuration->driver_shared_object_path,configuration->node_target,&state->driver,error_buffer,sizeof(error_buffer));
	if ( status != SPARK_STATUS_OK )
		return(status);
	descriptor = state->driver.interface->descriptor;
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_GLM52_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,GLM52_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_GLM52_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_GLM52_SERVING_TARGET) != 0 || strcmp(descriptor->model_description_sha256,GLM52_CONTRACT_SHA256) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,configuration->driver_program_name);
	if ( state->program == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( state->driver.interface->admit == 0 || state->program->submit == 0 || SparkModelDriverProgramSupportsRuntimeLimits(state->program,SPARK_GLM52_SERVING_REQUIRED_PROGRAM_FLAGS,state->pipeline_slot_count,state->max_active_sequence_count,state->max_input_row_count,state->resident_sequence_capacity) == 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = configuration->node_id;
	request.node_target = configuration->node_target;
	request.node_context = &state->node_context;
	request.execution_stream = configuration->execution_stream;
	request.completion_function = SparkGlm52ServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkGlm52ServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	return(status == SPARK_STATUS_OK && state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status);
}

static SparkStatus SparkGlm52ServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SparkGlm52ServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_GLM52_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkGlm52ServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkGlm52ServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkGlm52ServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->stage_index = configuration->stage_index;
	state->pipeline_slot_count = configuration->runtime_limits.max_inflight_submission_count;
	state->max_active_sequence_count = configuration->runtime_limits.max_active_sequence_count;
	state->max_input_row_count = configuration->runtime_limits.max_input_row_count;
	state->resident_sequence_capacity = configuration->runtime_limits.resident_sequence_capacity;
	state->runtime_limits = configuration->runtime_limits;
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	state->wake_function = configuration->wake_function;
	state->wake_context = configuration->wake_context;
	state->execution_stream = configuration->execution_stream;
	status = SparkGlm52ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > state->resident_sequence_capacity) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT;
		state->node_context.stage_index = state->stage_index;
		state->node_context.first_layer_index = SparkGlm52ResidentDecodeStageFirstLayer(state->stage_index);
		state->node_context.layer_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE;
		state->node_context.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->pipeline_slot_count;
		state->node_context.max_sequence_positions = max_sequence_positions;
		state->node_context.execution_row_capacity = execution_row_capacity;
		state->node_context.stage_pack_path = state->stage_pack_path;
		state->node_context.model_revision = GLM52_MODEL_REVISION;
		status = SparkGlm52ServingLoadDriver(state,configuration);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm52ServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingValidateBoundaries(
	const SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes,sideband_bytes;
	uint32_t sideband_input,sideband_output;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	sideband_bytes = (uint64_t)submission->row_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	sideband_input = SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index);
	sideband_output = SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index);
	if ( (state->stage_index != 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes < boundary_bytes)) || (state->stage_index == 0u && (submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u)) || (state->stage_index + 1u < SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT && (submission->hidden_output_address == 0 || submission->hidden_output_bytes < boundary_bytes)) || (state->stage_index + 1u == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT && (submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (sideband_input != 0u && (submission->boundary_sideband_input_address == 0 || submission->boundary_sideband_input_bytes < sideband_bytes)) || (sideband_input == 0u && (submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u)) || (sideband_output != 0u && (submission->boundary_sideband_output_address == 0 || submission->boundary_sideband_output_bytes < sideband_bytes)) || (sideband_output == 0u && (submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm52ServingState *state;
	SparkStatus status;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkGlm52ServingDescriptor,&state->runtime_limits,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingValidateBoundaries(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingValidateRowOrder(state,submission);
	if ( status == SPARK_STATUS_OK && submission->model_extension_bytes != 0u )
		status = SPARK_STATUS_UNSUPPORTED;
	return(status);
}

static void SparkGlm52ServingBuildFrame(
	const SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkGlm52ServingPending *pending,
	SparkGlm52ResidentDecodeStageBatchView *batch,
	SparkGlm52ResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffer,
	SparkModelDriverFrame *frame)
{
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = submission->row_count;
	batch->active_sequence_count = submission->active_sequence_count;
	batch->token_ids = state->stage_index == 0u ? submission->token_ids : 0;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = submission->row_positions;
	batch->row_sequence_ids = submission->row_sequence_ids;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	context->flags |= state->stage_index != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	context->flags |= state->stage_index + 1u < SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	context->flags |= SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u;
	context->flags |= SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u;
	context->batch = batch;
	context->hidden_input_bf16 = submission->hidden_input_address;
	context->hidden_input_bytes = submission->hidden_input_bytes;
	context->hidden_output_bf16 = submission->hidden_output_address;
	context->hidden_output_bytes = submission->hidden_output_bytes;
	context->sideband_input = submission->boundary_sideband_input_address;
	context->sideband_input_bytes = submission->boundary_sideband_input_bytes;
	context->sideband_output = submission->boundary_sideband_output_address;
	context->sideband_output_bytes = submission->boundary_sideband_output_bytes;
	memset(buffer,0,sizeof(*buffer));
	if ( state->stage_index + 1u == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT )
	{
		buffer->flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		buffer->address = pending->output_token_ids;
		buffer->bytes = (uint64_t)submission->row_count * sizeof(uint32_t);
	}
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = submission->sequence_id;
	frame->sequence_position = submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = submission->active_sequence_count;
	frame->new_token_count = submission->row_count;
	frame->priority = submission->priority;
	frame->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = state->stage_index + 1u == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT ? buffer : 0;
	frame->buffer_count = state->stage_index + 1u == SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT ? 1u : 0u;
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkGlm52ServingDriverCompletion;
	frame->completion_context = pending;
}

static SparkStatus SparkGlm52ServingAdmit(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = state->program->program_id;
	request.request_id = submission->request_id;
	request.sequence_id = submission->sequence_id;
	request.sequence_position = submission->sequence_position;
	request.deadline_time_ns = submission->deadline_time_ns;
	request.active_slot_count = submission->active_sequence_count;
	request.new_token_count = submission->row_count;
	request.priority = submission->priority;
	request.frame_flags = frame->flags;
	request.residency = submission->residency;
	SparkModelDriverInitializeAdmissionDecision(&decision);
	status = state->driver.interface->admit(state->driver_instance,&request,&decision);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkModelDriverAdmissionDecisionIsValid(&decision) == 0u )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( decision.accepted == 0u )
		return(decision.rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY ? SPARK_STATUS_BUSY : SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SparkModelDriverApplyAdmissionDecision(&decision,frame));
}

static SparkStatus SparkGlm52ServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm52ServingState *state;
	SparkGlm52ServingPending *pending;
	SparkGlm52ResidentDecodeStageBatchView batch;
	SparkGlm52ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffer;
	SparkModelDriverFrame frame;
	SparkStatus status;
	state = (SparkGlm52ServingState *)adapter_state;
	status = SparkGlm52ServingValidateSubmission(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkGlm52ServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	SparkGlm52ServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
	status = SparkGlm52ServingAdmit(state,submission,&frame);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status != SPARK_STATUS_OK )
		pending->active = 0u;
	return(status);
}

static SparkStatus SparkGlm52ServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkGlm52ServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkGlm52ServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SparkGlm52ServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

static SparkStatus SparkGlm52ServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkGlm52ServingState *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkGlm52ServingAvailableSubmissionCount(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SparkGlm52ServingAvailableSubmissionCount(state);
	snapshot->submitted_count = driver_snapshot.submitted_count;
	snapshot->completed_count = driver_snapshot.completed_count;
	snapshot->rejected_count = driver_snapshot.rejected_count + state->orphan_completion_count;
	snapshot->resident_sequence_count = driver_snapshot.resident_sequence_count;
	snapshot->resident_token_count = driver_snapshot.resident_token_count;
	snapshot->kv_token_capacity = driver_snapshot.kv_token_capacity;
	snapshot->device_memcpy_bytes_per_submit = driver_snapshot.device_memcpy_bytes_per_submit;
	snapshot->host_staging_bytes_per_submit = driver_snapshot.host_staging_bytes_per_submit;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface SparkGlm52ServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkGlm52ServingDescriptor,
	.initialize = SparkGlm52ServingInitialize,
	.destroy = SparkGlm52ServingDestroy,
	.validate_submission = SparkGlm52ServingValidateSubmission,
	.submit = SparkGlm52ServingSubmit,
	.progress = SparkGlm52ServingProgress,
	.quiesce = SparkGlm52ServingQuiesce,
	.snapshot = SparkGlm52ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkGlm52ServingInterface);
}
