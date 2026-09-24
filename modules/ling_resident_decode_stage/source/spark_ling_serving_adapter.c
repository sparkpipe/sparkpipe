#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_ling_kv_geometry.h"
#include "sparkpipe/spark_ling_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_ling_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#define SPARK_FAMILY_CAMEL Ling
#define SPARK_FAMILY_UPPER LING
#define SPARK_FAMILY_LOWER ling

#include "sparkpipe/family/spark_family.h"

#ifndef LING_EXPERT_WEIGHT_CODEC
#error "LING_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef LING_EXPERT_CODEC_NAME
#error "LING_EXPERT_CODEC_NAME must name the exact package expert codec"
#endif
#ifndef LING_MODEL_REVISION
#error "LING_MODEL_REVISION must name the exact source snapshot"
#endif
#ifndef LING_CONTRACT_SHA256
#error "LING_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_LING_SERVING_ADAPTER_ID \
	"spark.ling.serving-adapter.tp16.expert_" LING_EXPERT_CODEC_NAME ".v1"
#define SPARK_LING_SERVING_STAGE_COUNT 16u
#define SPARK_LING_SERVING_TP_DEGREE 16u
#define SPARK_LING_SERVING_STAGE_LAYERS \
	{42u,42u,42u,42u,42u,42u,42u,42u,42u,42u,42u,42u,42u,42u,42u,42u}
#define SPARK_LING_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
#define SPARK_LING_SERVING_MODEL_ID "inclusionAI/Ling-3.0-flash"
#define SPARK_LING_SERVING_DRIVER_MODEL_ID \
	"inclusion.ling-3.0-flash.resident-decode-stage-firmware"
#define SPARK_LING_SERVING_STAGE_NAME "ling_resident_decode_stage"
#define SPARK_LING_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_LING_SERVING_TARGET \
	"cuda.sm121.ling.resident_decode_stage.bf16.expert_" LING_EXPERT_CODEC_NAME
#define SPARK_LING_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL)

static const char *const SparkLingServingConfigurationMembers[] =
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

typedef struct SparkLingServingPending
{
	struct SparkLingServingState *owner;
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
	uint32_t last_row_by_lane[SPARK_LING_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_LING_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_LING_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
} SparkLingServingPending;

typedef struct SparkLingServingState
{
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkLingResidentDecodeStageNodeContext node_context;
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
	atomic_uint reset_active;
	atomic_uint_fast64_t reset_generation;
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
	SparkLingServingPending pending[SPARK_LING_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkLingServingState;

static const SparkModelServingAdapterDescriptor SparkLingServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_LING_SERVING_TOPOLOGY_FLAG |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION,
	.stage_count = SPARK_LING_SERVING_STAGE_COUNT,
	.layer_count = SPARK_LING_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_LING_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT,
	.boundary_element_bytes = SPARK_LING_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = LING_EXPERT_WEIGHT_CODEC,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = SPARK_LING_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_LING_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_LING_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequence_count = SPARK_LING_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_LING_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.adapter_id = SPARK_LING_SERVING_ADAPTER_ID,
	.model_id = SPARK_LING_SERVING_MODEL_ID,
	.model_revision = LING_MODEL_REVISION,
	.driver_program_name = SPARK_LING_SERVING_PROGRAM_NAME,
	.artifact_sha256 = LING_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_LING_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds = {0u},
	.boundary_sideband_bytes_per_sequence = {0u},
	.cache_block_token_count = SPARK_LING_KV_BLOCK_TOKEN_COUNT
};

static SparkStatus SparkLingServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkLingServingState *state,
	uint32_t tp_degree)
{
	SparkTpCollectiveConfigPolicy policy;
	SparkTpCollectiveAdapterConfig config;
	SparkStatus status;
	policy.peer_count = tp_degree;
	policy.allow_zero_collective_identifier = 1u;
	policy.require_contiguous_peer_ports = 1u;
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_ADAPTIVE_COMBOS;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_MASK_CONDITIONAL;
	policy.require_session_ports = 1u;
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
		memcpy(state->tp_peer_ports,config.peer_ports,
			sizeof(state->tp_peer_ports));
		state->tp_connect_timeout_milli = config.connect_timeout_milli;
		state->tp_operation_timeout_milli = config.operation_timeout_milli;
		state->tp_collective_control_port_base = config.control_port_base;
		state->tp_collective_topology = config.topology;
		memcpy(state->node_context.tp_collective_session_ports_hc,
			config.session_ports_hc,
			sizeof(state->node_context.tp_collective_session_ports_hc));
	}
	return(status);
}

static SparkStatus SparkLingServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkLingServingState *state,
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
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkLingServingConfigurationMembers,(uint32_t)(sizeof(SparkLingServingConfigurationMembers) / sizeof(SparkLingServingConfigurationMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_LING_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,LING_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"expert_weight_codec") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,LING_EXPERT_CODEC_NAME)) )
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
		status = SparkLingServingLoadTpCollective(&document,root,runtime_root,state,*tp_degree);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(status);
}

static SparkLingServingPending *SparkLingServingReservePending(
	SparkLingServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkLingServingPending *pending;
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

static void SparkLingServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkLingServingPending *pending;
	SparkLingServingState *state;
	SparkModelServingCompletion completion;
	uint32_t index,matches;
	pending = (SparkLingServingPending *)completion_context;
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
	if ( completion.status != SPARK_STATUS_OK )
	{
		completion.accepted_token_count = 0u;
		completion.completion_flags = 0u;
	}
	if ( completion.status == SPARK_STATUS_OK )
	{
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index]];
	}
	pending->active = 0u;
	state->completion_function(state->completion_context,&completion);
}

static uint32_t SparkLingServingAvailableSubmissionCount(
	const SparkLingServingState *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += state->pending[index].active == 0u ? 1u : 0u;
	return(available);
}

static void SparkLingServingDestroy(void *adapter_state)
{
	SparkLingServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkLingServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
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

#include "sparkpipe/family/serving/spark_serving_orphan_driver_completion.h"

static SparkStatus SparkLingServingLoadDriver(
	SparkLingServingState *state,
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
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_LING_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,LING_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_LING_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_LING_SERVING_TARGET) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,configuration->driver_program_name);
	if ( state->program == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( state->driver.interface->admit == 0 || state->program->submit == 0 || SparkModelDriverProgramSupportsRuntimeLimits(state->program,SPARK_LING_SERVING_REQUIRED_PROGRAM_FLAGS,state->pipeline_slot_count,state->max_active_sequence_count,state->max_input_row_count,state->resident_sequence_capacity) == 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	SparkModelDriverInitializeCreateRequest(&request);
	request.node_id = configuration->node_id;
	request.node_target = configuration->node_target;
	request.node_context = &state->node_context;
	request.kv_logical_page_capacity =
		configuration->runtime_limits.kv_logical_page_capacity;
	request.kv_physical_page_capacity =
		configuration->runtime_limits.kv_physical_page_capacity;
	request.kv_backing_directory = configuration->kv_backing_directory;
	request.kv_backing_maximum_bytes =
		configuration->kv_backing_maximum_bytes;
	request.execution_stream = configuration->execution_stream;
	request.completion_function = SparkLingServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkLingServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	if ( status != SPARK_STATUS_OK )
		(void)fprintf(stderr,"LNG-T1ADAPTER-DIAG create=%d stage=%u layers=%u slots=%u cap=%u pos=%u rows=%u thr=%u tp=%u/%u codec=%u rev=%s pack=%s\n",
		    (int)status,state->node_context.stage_count,state->node_context.layer_count,state->node_context.pipeline_slot_count,state->node_context.resident_sequence_capacity,state->node_context.max_sequence_positions,state->node_context.execution_row_capacity,state->node_context.decode_split_context_threshold,state->node_context.tp_degree,state->node_context.tp_rank,state->node_context.expert_weight_codec,state->node_context.model_revision == 0 ? "(null)" : state->node_context.model_revision,state->node_context.stage_pack_path == 0 ? "(null)" : state->node_context.stage_pack_path);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK);
}

static SparkStatus SparkLingServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SparkLingServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_LING_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_LING_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLingServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkLingServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
	uint32_t decode_split_context_threshold;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkLingServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkLingServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	atomic_init(&state->reset_active,0u);
	atomic_init(&state->reset_generation,0u);
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
	status = SparkLingServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&decode_split_context_threshold,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_LING_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > SPARK_LING_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || decode_split_context_threshold > max_sequence_positions) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_LING_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_LING_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_LING_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_LING_RESIDENT_DECODE_STAGE_STAGE_COUNT;
		state->node_context.stage_index = 0u;
		state->node_context.first_layer_index = 0u;
		state->node_context.layer_count = SPARK_LING_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE;
		state->node_context.expert_weight_codec = LING_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->pipeline_slot_count;
		state->node_context.max_sequence_positions = max_sequence_positions;
		state->node_context.execution_row_capacity = execution_row_capacity;
		state->node_context.decode_split_context_threshold = decode_split_context_threshold;
		state->node_context.tp_degree = tp_degree;
		state->node_context.tp_rank = tp_rank;
		state->node_context.stage_pack_path = state->stage_pack_path;
		state->node_context.model_revision = LING_MODEL_REVISION;
		state->node_context.tp_collective_backend_kind = state->tp_collective_backend_kind;
		state->node_context.tp_collective_identifier = state->tp_collective_identifier;
		state->node_context.tp_connect_timeout_milli = state->tp_connect_timeout_milli;
		state->node_context.tp_operation_timeout_milli = state->tp_operation_timeout_milli;
		state->node_context.tp_collective_control_port_base = state->tp_collective_control_port_base;
		state->node_context.tp_collective_topology = state->tp_collective_topology;
		state->node_context.tp_collective_backend_module_path = state->tp_collective_backend_path;
		state->node_context.kv_backing_directory = configuration->kv_backing_directory;
		state->node_context.kv_backing_maximum_bytes = configuration->kv_backing_maximum_bytes;
		status = SparkLingServingLoadDriver(state,configuration);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkLingServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkLingServingSubmissionStale(
	const SparkLingServingState *state,
	const SparkModelServingSubmission *submission)
{
	if ( submission == 0 )
		return(0u);
	return(submission->control_generation <
		atomic_load_explicit(&state->reset_generation,memory_order_acquire) ?
		1u : 0u);
}

#include "sparkpipe/family/serving/spark_serving_validate_row_order.h"

static SparkStatus SparkLingServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkLingServingState *state;
	SparkStatus status;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	if ( SparkLingServingSubmissionStale(state,submission) != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkLingServingDescriptor,&state->runtime_limits,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkLingServingValidateBoundaries(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkLingServingValidateRowOrder(state,submission);
	if ( status == SPARK_STATUS_OK && submission->model_extension_bytes != 0u )
	{
		status = SPARK_STATUS_UNSUPPORTED;
	}
	return(status);
}

static _Thread_local SparkModelDriverCacheLane SparkLingServingCacheScratch[SPARK_LING_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

#include "sparkpipe/family/serving/spark_serving_cache_context.h"

static SparkStatus SparkLingServingPrefetch(void *adapter_state,
	const SparkModelServingSubmission *submissions,uint32_t count)
{
	SparkLingServingState *state;
	SparkServingCacheAdmission cache;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 || state->program == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	cache = SparkLingServingCacheContext(state,SparkLingServingCacheScratch);
	return(SparkServingCacheAdmissionRun(&cache,submissions,count,
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE));
}

static SparkStatus SparkLingServingResolvePrefetch(void *adapter_state,
	const SparkModelServingSubmission *submission,uint32_t resolution)
{
	SparkLingServingState *state;
	SparkServingCacheAdmission cache;
	uint32_t flags;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 || state->program == 0 ||
		(resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		 resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	flags = resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT :
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	cache = SparkLingServingCacheContext(state,SparkLingServingCacheScratch);
	return(SparkServingCacheAdmissionRun(&cache,submission,1u,flags));
}

static void SparkLingServingBuildFrame(
	const SparkLingServingState *state,
	const SparkModelServingSubmission *submission,
	SparkLingServingPending *pending,
	SparkLingResidentDecodeStageBatchView *batch,
	SparkLingResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffer,
	SparkModelDriverFrame *frame)
{
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_LING_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = submission->row_count;
	batch->active_sequence_count = submission->active_sequence_count;
	batch->token_ids = submission->token_ids;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = submission->row_positions;
	batch->row_sequence_ids = submission->row_sequence_ids;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_LING_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_LING_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	context->batch = batch;
	context->hidden_input_bf16 = submission->hidden_input_address;
	context->hidden_input_bytes = submission->hidden_input_bytes;
	context->hidden_output_bf16 = submission->hidden_output_address;
	context->hidden_output_bytes = submission->hidden_output_bytes;
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
	frame->completion_function = SparkLingServingDriverCompletion;
	frame->completion_context = pending;
}

static SparkStatus SparkLingServingAdmit(
	SparkLingServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	status = SparkAdmissionRequestFromSubmission(
		state->program->program_id,submission,0,0u,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkAdmissionEvaluateAndApply(
		state->driver.interface,state->driver_instance,&request,frame,&decision));
}

static SparkStatus SparkLingServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkLingServingState *state;
	SparkLingServingPending *pending;
	SparkLingResidentDecodeStageBatchView batch;
	SparkLingResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffer;
	SparkModelDriverFrame frame;
	SparkStatus status;
	state = (SparkLingServingState *)adapter_state;
	status = SparkLingServingValidateSubmission(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkLingServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	SparkLingServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
	status = SparkLingServingAdmit(state,submission,&frame);
	if ( status == SPARK_STATUS_OK )
	{
		status = state->program->submit(state->driver_instance,&frame);
	}
	if ( status != SPARK_STATUS_OK )
		pending->active = 0u;
	return(status);
}

static SparkStatus SparkLingServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkLingServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkLingServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SparkLingServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

#include "sparkpipe/family/serving/spark_serving_reset_control.h"

static SparkStatus SparkLingServingReset(void *adapter_state,
	uint64_t control_generation)
{
	SparkLingServingState *state;
	uint32_t expected = 0u;
	SparkStatus status;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( atomic_compare_exchange_strong_explicit(&state->reset_active,&expected,1u,
		memory_order_acquire,memory_order_relaxed) == 0 )
		return(SPARK_STATUS_BUSY);
	status = SparkLingServingResetControl(state,control_generation);
	atomic_store_explicit(&state->reset_active,0u,memory_order_release);
	return(status);
}

static SparkStatus SparkLingServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkLingServingState *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SparkLingServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkLingServingAvailableSubmissionCount(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SparkLingServingAvailableSubmissionCount(state);
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

static const SparkModelServingAdapterInterface SparkLingServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkLingServingDescriptor,
	.initialize = SparkLingServingInitialize,
	.destroy = SparkLingServingDestroy,
	.validate_submission = SparkLingServingValidateSubmission,
	.submit = SparkLingServingSubmit,
	.prefetch = SparkLingServingPrefetch,
	.resolve_prefetch = SparkLingServingResolvePrefetch,
	.progress = SparkLingServingProgress,
	.quiesce = SparkLingServingQuiesce,
	.snapshot = SparkLingServingSnapshot,
	.reset = SparkLingServingReset
};

#include "sparkpipe/family/serving/spark_serving_get_interface.h"
