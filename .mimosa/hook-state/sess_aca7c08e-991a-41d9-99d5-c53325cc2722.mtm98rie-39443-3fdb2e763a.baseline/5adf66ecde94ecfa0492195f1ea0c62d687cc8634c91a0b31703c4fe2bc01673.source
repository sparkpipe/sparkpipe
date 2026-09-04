#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm52_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_serving_adapter_template.h"

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
	"spark.glm52.serving-adapter.tp8.expert_" GLM52_EXPERT_CODEC_NAME ".v1"
#define SPARK_GLM52_SERVING_STAGE_COUNT 8u
#define SPARK_GLM52_SERVING_TP_DEGREE 8u
#define SPARK_GLM52_SERVING_STAGE_LAYERS \
	{78u,78u,78u,78u,78u,78u,78u,78u}
#define SPARK_GLM52_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
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
	"execution_row_capacity",
	"decode_split_context_threshold",
	"tp_degree",
	"tp_rank",
	"tp_collective"
};

typedef struct SparkGlm52ServingPending
{
	struct SparkGlm52ServingState *owner;
	SparkServingAdapterPendingCommon common;
	uint32_t last_row_by_lane[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	SparkModelDriverCacheLane cache_lanes[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t cache_lane_count;
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
	uint16_t tp_listen_port;
	uint16_t tp_peer_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t tp_collective_backend_kind;
	uint64_t tp_collective_identifier;
	SparkTpDeviceCollectiveTopology tp_collective_topology;
	char tp_collective_backend_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t tp_collective_control_port_base;
	SparkModelServingRuntimeLimits runtime_limits;
	SparkModelDriverCacheLane prefetch_lanes[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkGlm52ServingPending pending[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkGlm52ServingState;

static const SparkModelServingAdapterDescriptor SparkGlm52ServingDescriptor =
{
	SPARK_SERVING_ADAPTER_DESCRIPTOR_IDENTITY(
		SPARK_GLM52_SERVING_ADAPTER_ID,
		SPARK_GLM52_SERVING_MODEL_ID,
		GLM52_MODEL_REVISION,
		SPARK_GLM52_SERVING_PROGRAM_NAME,
		GLM52_CONTRACT_SHA256),
	.capability_flags = SPARK_SERVING_ADAPTER_CAPABILITY_CHAIN(
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION |
		SPARK_GLM52_SERVING_TOPOLOGY_FLAG |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV),
	.stage_count = SPARK_GLM52_SERVING_STAGE_COUNT,
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
	.cache_block_token_count = 64u,
	.stage_layer_counts = SPARK_GLM52_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds = {0u},
	.boundary_sideband_bytes_per_sequence = {0u}
};

static SparkStatus SparkGlm52ServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkGlm52ServingState *state,
	uint32_t tp_degree)
{
	SparkTpCollectiveConfigPolicy policy;
	SparkTpCollectiveAdapterConfig config;
	SparkStatus status;
	policy.peer_count = tp_degree;
	policy.allow_zero_collective_identifier = 1u;
	policy.require_contiguous_peer_ports = 1u;
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_RECURSIVE_DOUBLING_ONLY;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_ZERO_REQUIRED;
	memset(&config,0,sizeof(config));
	config.backend_module_path_buffer = state->tp_collective_backend_path;
	config.backend_module_path_bytes = sizeof(state->tp_collective_backend_path);
	status = SparkServingAdapterTemplateLoadTpCollective(document,root,
		runtime_root,&policy,&config);
	if ( status == SPARK_STATUS_OK )
	{
		state->tp_collective_backend_kind = config.backend_kind;
		state->tp_collective_identifier = config.collective_identifier;
		state->tp_listen_port = config.listen_port;
		memcpy(state->tp_peer_ports,config.peer_ports,sizeof(state->tp_peer_ports));
		state->tp_connect_timeout_milli = config.connect_timeout_milli;
		state->tp_operation_timeout_milli = config.operation_timeout_milli;
		state->tp_collective_control_port_base = config.control_port_base;
		state->tp_collective_topology = config.topology;
	}
	(void)fprintf(stderr,"GLM52-ADAPTER LoadTpCollective rc=%d backend=%u\n",(int)status,config.backend_kind);
	return(status);
}

static SparkStatus SparkGlm52ServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkGlm52ServingState *state,
	uint32_t *max_sequence_positions,
	uint32_t *execution_row_capacity,
	uint32_t *decode_split_context_threshold,
	uint32_t *tp_degree,
	uint32_t *tp_rank)
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
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_GLM52_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM52_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"expert_weight_codec") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM52_EXPERT_CODEC_NAME)) )
		status = SPARK_STATUS_TARGET_MISMATCH;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"execution_row_capacity",execution_row_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"decode_split_context_threshold",decode_split_context_threshold);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"tp_degree",tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"tp_rank",tp_rank);
	if ( status == SPARK_STATUS_OK && (*tp_degree == 0u || *tp_rank >= *tp_degree) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingLoadTpCollective(&document,root,runtime_root,state,*tp_degree);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	(void)fprintf(stderr,"GLM52-ADAPTER LoadConfiguration rc=%d\n",(int)status);
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
	uint32_t row;
	pending = (SparkGlm52ServingPending *)
		SparkServingAdapterTemplateReservePending(state->pending,
			sizeof(*pending),
			(uint32_t)offsetof(SparkGlm52ServingPending,common),
			state->pipeline_slot_count,
			(uint32_t)offsetof(SparkGlm52ServingPending,last_row_by_lane),
			submission);
	if ( pending == 0 )
		return(0);
	pending->owner = state;
	pending->common.active = 1u;
	for (row=0u; row<submission->row_count; row++)
		pending->resident_slots[row] =
			submission->lanes[submission->row_lane_indices[row]].resident_sequence_slot;
	return(pending);
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
	if ( state == 0 || pending->common.active == 0u || driver_completion == 0 )
		return;
	matches = driver_completion->request_id == pending->common.request_id && driver_completion->sequence_id == pending->common.sequence_id && driver_completion->sequence_position == pending->common.sequence_position && driver_completion->program_id == state->program->program_id;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = matches != 0u ? (uint32_t)driver_completion->status : SPARK_STATUS_SCHEMA_ERROR;
	completion.submission_id = pending->common.submission_id;
	completion.request_id = pending->common.request_id;
	completion.sequence_id = pending->common.sequence_id;
	completion.sequence_position = pending->common.sequence_position;
	completion.control_generation = pending->common.control_generation;
	completion.transaction_id = pending->common.transaction_id;
	completion.dispatch_generation = pending->common.dispatch_generation;
	completion.request_generation = pending->common.request_generation;
	completion.step_generation = pending->common.step_generation;
	completion.accepted_token_count = driver_completion->accepted_token_count;
	completion.queue_delay_ns = driver_completion->queue_delay_ns;
	completion.service_time_ns = driver_completion->service_time_ns;
	completion.device_memcpy_bytes = driver_completion->device_memcpy_bytes;
	completion.host_staging_bytes = driver_completion->host_staging_bytes;
	if ( matches != 0u )
		completion.residency = driver_completion->residency;
	else
		state->orphan_completion_count++;

	if ( state->stage_index + 1u == SPARK_GLM52_SERVING_STAGE_COUNT && completion.status == SPARK_STATUS_OK )
	{
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->common.active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index]];
	}
	pending->common.active = 0u;
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
		available += state->pending[index].common.active == 0u ? 1u : 0u;
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

static SparkStatus SparkGlm52ServingAcceptsProgram(
	const SparkModelDriverProgramDescriptor *program,
	void *accept_context)
{
	SparkGlm52ServingState *state;
	state = (SparkGlm52ServingState *)accept_context;
	if ( SparkModelDriverProgramSupportsRuntimeLimits(program,SPARK_GLM52_SERVING_REQUIRED_PROGRAM_FLAGS,state->pipeline_slot_count,state->max_active_sequence_count,state->max_input_row_count,state->resident_sequence_capacity) == 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingLoadDriver(
	SparkGlm52ServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkServingAdapterDriverRequest request;
	const SparkModelDriverProgramDescriptor *program;
	SparkStatus status;
	request.contract.driver_model_id = SPARK_GLM52_SERVING_DRIVER_MODEL_ID;
	request.contract.driver_model_revision = GLM52_MODEL_REVISION;
	request.contract.driver_stage_name = SPARK_GLM52_SERVING_STAGE_NAME;
	request.contract.driver_target = SPARK_GLM52_SERVING_TARGET;
	request.contract.model_description_sha256 = GLM52_CONTRACT_SHA256;
	request.node_context = &state->node_context;
	request.completion_context = state;
	request.completion_function = SparkGlm52ServingOrphanDriverCompletion;
	request.wake_function = SparkGlm52ServingDriverWake;
	program = 0;
	status = SparkServingAdapterTemplateLoadDriver(&request,configuration,
		&state->driver,&program,SparkGlm52ServingAcceptsProgram,state,
		&state->driver_instance);
	state->program = program;
	(void)fprintf(stderr,"GLM52-ADAPTER LoadDriver rc=%d\n",(int)status);
	return(status);
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
	if ( configuration->stage_index >= SPARK_GLM52_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_GLM52_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkGlm52ServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
	uint32_t decode_split_context_threshold;
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
	status = SparkGlm52ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&decode_split_context_threshold,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > state->resident_sequence_capacity || decode_split_context_threshold > max_sequence_positions) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_GLM52_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT;
		state->node_context.stage_index = 0u;
		state->node_context.first_layer_index = 0u;
		state->node_context.layer_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE;
		state->node_context.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->pipeline_slot_count;
		state->node_context.max_sequence_positions = max_sequence_positions;
		state->node_context.execution_row_capacity = execution_row_capacity;
		state->node_context.decode_split_context_threshold = decode_split_context_threshold;
		state->node_context.tp_degree = tp_degree;
		state->node_context.tp_rank = tp_rank;
		state->node_context.stage_pack_path = state->stage_pack_path;
		state->node_context.model_revision = GLM52_MODEL_REVISION;
		state->node_context.tp_collective_backend_kind = state->tp_collective_backend_kind;
		state->node_context.tp_collective_identifier = state->tp_collective_identifier;
		state->node_context.tp_connect_timeout_milli = state->tp_connect_timeout_milli;
		state->node_context.tp_operation_timeout_milli = state->tp_operation_timeout_milli;
		state->node_context.tp_collective_control_port_base = state->tp_collective_control_port_base;
		state->node_context.tp_collective_topology = state->tp_collective_topology;
		state->node_context.tp_collective_backend_module_path = state->tp_collective_backend_path;
		state->node_context.kv_backing_directory = configuration->kv_backing_directory;
		state->node_context.kv_backing_maximum_bytes = configuration->kv_backing_maximum_bytes;
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
	uint64_t boundary_bytes;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	if ( submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)boundary_bytes;
	(void)state;
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
	batch->token_ids = submission->token_ids;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = submission->row_positions;
	batch->row_sequence_ids = submission->row_sequence_ids;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
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
	buffer->flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	buffer->address = pending->output_token_ids;
	buffer->bytes = (uint64_t)submission->row_count * sizeof(uint32_t);
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = submission->sequence_id;
	frame->sequence_position = submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = submission->active_sequence_count;
	frame->new_token_count = submission->row_count;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = buffer;
	frame->buffer_count = 1u;
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkGlm52ServingDriverCompletion;
	frame->completion_context = pending;
}

static SparkStatus SparkGlm52ServingAdmit(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame,
	SparkModelDriverCacheLane *cache_lanes,
	uint32_t admission_flags)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	status = SparkAdmissionRequestFromSubmission(
		state->program->program_id,submission,cache_lanes,admission_flags,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkAdmissionEvaluateAndApply(
		state->driver.interface,state->driver_instance,&request,frame,&decision));
}

static SparkStatus SparkGlm52ServingPrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submissions,
	uint32_t submission_count)
{
	SparkGlm52ServingState *state;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t cache_lane_count,index;
	SparkStatus status;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 || submissions == 0 || submission_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<submission_count; index++)
	{
		status = SparkGlm52ServingValidateSubmission(state,&submissions[index]);
		if ( status == SPARK_STATUS_OK && submissions[index].work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			continue;
		if ( status == SPARK_STATUS_OK )
			status = SparkModelServingAdapterBuildDriverCacheLanes(&submissions[index],
				state->prefetch_lanes,
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
				&cache_lane_count);
		if ( status == SPARK_STATUS_OK )
		{
			status = SparkAdmissionRequestFromSubmission(state->program->program_id,
				&submissions[index],state->prefetch_lanes,
				SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,&request);
			if ( status == SPARK_STATUS_OK )
				status = SparkAdmissionEvaluate(state->driver.interface,
					state->driver_instance,&request,&decision);
		}
	}
	return(status);
}

static SparkStatus SparkGlm52ServingResolvePrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submission,
	uint32_t resolution)
{
	SparkGlm52ServingState *state;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t admission_flag,cache_lane_count;
	SparkStatus status;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 || submission == 0 ||
		(resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		 resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm52ServingValidateSubmission(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(SPARK_STATUS_OK);
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission,
		state->prefetch_lanes,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
		&cache_lane_count);
	if ( status != SPARK_STATUS_OK || cache_lane_count != submission->active_sequence_count )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INTERNAL_ERROR);
	admission_flag = resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT :
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	status = SparkAdmissionRequestFromSubmission(state->program->program_id,
		submission,state->prefetch_lanes,admission_flag,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkAdmissionEvaluate(state->driver.interface,
		state->driver_instance,&request,&decision));
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
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission,
		pending->cache_lanes,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
		&pending->cache_lane_count);
	if ( status != SPARK_STATUS_OK )
	{
		pending->common.active = 0u;
		return(status);
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		SparkGlm52ServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
		frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
		frame.cache_lane_count = pending->cache_lane_count;
		frame.cache_lanes = pending->cache_lanes;
		status = SparkGlm52ServingAdmit(state,submission,&frame,pending->cache_lanes,0u);
		if ( status == SPARK_STATUS_OK )
			status = state->program->submit(state->driver_instance,&frame);
		if ( status != SPARK_STATUS_OK )
			pending->common.active = 0u;
		return(status);
	}
	SparkGlm52ServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
	frame.cache_lane_count = pending->cache_lane_count;
	frame.cache_lanes = pending->cache_lanes;
	status = SparkGlm52ServingAdmit(state,submission,&frame,pending->cache_lanes,0u);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status != SPARK_STATUS_OK )
		pending->common.active = 0u;
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
	.prefetch = SparkGlm52ServingPrefetch,
	.resolve_prefetch = SparkGlm52ServingResolvePrefetch,
	.progress = SparkGlm52ServingProgress,
	.quiesce = SparkGlm52ServingQuiesce,
	.snapshot = SparkGlm52ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkGlm52ServingInterface);
}
