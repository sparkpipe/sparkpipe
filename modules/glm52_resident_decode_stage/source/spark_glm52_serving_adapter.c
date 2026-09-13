#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm52_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_glm52_dspark.h"
#include "runtime/adapter_common.h"

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









#ifndef SPARK_GLM52_SERVING_STAGE_COUNT
#define SPARK_GLM52_SERVING_STAGE_COUNT 8u
#endif
#ifndef SPARK_GLM52_SERVING_TP_DEGREE
#define SPARK_GLM52_SERVING_TP_DEGREE 8u
#endif
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
	"tp_degree",
	"tp_rank",






	"speculation_enabled",
	"speculation_draft_count",
	"dspark_pack_path"
};

typedef struct SparkGlm52ServingPending
{


	SparkAdapterPendingCore core;
	struct SparkGlm52ServingState *owner;
	uint32_t last_row_by_lane[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];



	uint32_t spec_active;
	uint32_t spec_verify;
	uint32_t spec_rows_per_lane;
	uint32_t spec_row_token_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t spec_draft_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
	uint32_t spec_accept_counts[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t spec_row_positions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint64_t spec_row_sequence_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	SparkGlm52ResidentDecodeStageDsparkDraftView spec_view;
	SparkGlm52ResidentDecodeStageBatchView spec_batch;
	SparkGlm52ResidentDecodeStageFrameContext spec_context;
	SparkModelDriverBuffer spec_buffer;
	SparkModelDriverFrame spec_frame;
} SparkGlm52ServingPending;









typedef struct SparkGlm52ServingLaneSpec
{
	_Atomic uint32_t lock;
	uint64_t sequence_id;
	uint64_t tap_generation;
	uint32_t have_drafts;
	uint32_t draft_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
	uint32_t last_accepted_depth;
	uint8_t pending_stamp;
} SparkGlm52ServingLaneSpec;

typedef struct SparkGlm52ServingState
{




	SparkAdapterCommonState common;
	SparkGlm52ResidentDecodeStageNodeContext node_context;
	char stage_pack_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t stage_index;
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


	uint32_t speculate;
	uint32_t speculate_draft_rows;
	char dspark_pack_path[SPARK_INTERNAL_PATH_BYTES];
	SparkGlm52ServingLaneSpec spec_lanes[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *spec_stamp_scratch;
	SparkGlm52DsparkSpeculator speculator;





	_Atomic uint32_t speculator_lock;
	SparkGlm52DsparkSequenceState spec_sequence_states[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkGlm52ServingPending pending[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkGlm52ServingState;

static const SparkModelServingAdapterDescriptor SparkGlm52ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_GLM52_SERVING_TOPOLOGY_FLAG | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION,
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
	.max_speculative_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.adapter_id = SPARK_GLM52_SERVING_ADAPTER_ID,
	.model_id = SPARK_GLM52_SERVING_MODEL_ID,
	.model_revision = GLM52_MODEL_REVISION,
	.driver_program_name = SPARK_GLM52_SERVING_PROGRAM_NAME,
	.artifact_sha256 = GLM52_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_GLM52_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds = {0u},
	.boundary_sideband_bytes_per_sequence = {0u}
};








static const SparkAdapterTpCollectivePolicy SparkGlm52TpCollectivePolicy = {
	SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING, 1u, 1u
};

static SparkStatus SparkGlm52ServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkGlm52ServingState *state,
	uint32_t tp_degree)
{
	SparkAdapterTpCollectiveParsed tp;
	SparkStatus status;
	status = SparkAdapterLoadTpCollective(document,root,runtime_root,
		tp_degree,1u,&SparkGlm52TpCollectivePolicy,&tp);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_listen_port = tp.listen_port;
	state->tp_connect_timeout_milli = tp.connect_timeout_milli;
	state->tp_operation_timeout_milli = tp.operation_timeout_milli;
	state->tp_collective_backend_kind = tp.backend_kind;
	state->tp_collective_identifier = tp.collective_identifier;
	memcpy(state->tp_peer_ports,tp.peer_ports,sizeof(tp.peer_ports));
	state->tp_collective_control_port_base = tp.control_port_base;
	memcpy(state->tp_collective_backend_path,tp.backend_module_path,
		SPARK_INTERNAL_PATH_BYTES);
	state->tp_collective_topology = tp.topology;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkGlm52ServingState *state,
	uint32_t *max_sequence_positions,
	uint32_t *execution_row_capacity,
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
		status = SparkJsonGetUInt32Member(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_GLM52_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM52_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"expert_weight_codec") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM52_EXPERT_CODEC_NAME)) )
		status = SPARK_STATUS_TARGET_MISMATCH;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"max_sequence_positions",max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"execution_row_capacity",execution_row_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"tp_degree",tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"tp_rank",tp_rank);
	if ( status == SPARK_STATUS_OK && (*tp_degree == 0u || *tp_rank >= *tp_degree) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && *tp_degree != 1u )
	{
		if ( SparkJsonFindObjectMember(&document,root,"tp_collective") < 0 )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52ServingLoadTpCollective(&document,root,runtime_root,state,*tp_degree);
	}
	if ( status == SPARK_STATUS_OK )
	{
		token = SparkJsonFindObjectMember(&document,root,"speculation_enabled");
		if ( status == SPARK_STATUS_OK && token >= 0 )
		{
			bool enabled = false;
			if ( SparkJsonGetBoolean(&document,token,&enabled) == SPARK_STATUS_OK )
				state->speculate = enabled ? 1u : 0u;
			else
				status = SPARK_STATUS_SCHEMA_ERROR;
		}
		token = SparkJsonFindObjectMember(&document,root,"speculation_draft_count");
		if ( status == SPARK_STATUS_OK && token >= 0 )
			status = SparkJsonGetUInt32(&document,token,&state->speculate_draft_rows);
		token = SparkJsonFindObjectMember(&document,root,"dspark_pack_path");
		if ( status == SPARK_STATUS_OK && token >= 0 )
		{
			char *recorded = 0;
			status = SparkJsonCopyString(&document,token,&recorded);
			if ( status == SPARK_STATUS_OK )
			{
				if ( strlen(recorded) >= sizeof(state->dspark_pack_path) )
					status = SPARK_STATUS_SCHEMA_ERROR;
				else
					memcpy(state->dspark_pack_path,recorded,strlen(recorded) + 1u);
			}
			free(recorded);
		}
	}
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
	uint32_t occurrence_counts[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t wave_last_rows[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRowLayoutDenseLaneContext dense;
	uint32_t lane,row;
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->node_context.max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	dense.lane_count = submission->active_sequence_count;
	return(SparkRowLayoutValidateRoundMajor(submission->row_count,submission->active_sequence_count,submission->row_lane_indices,SparkRowLayoutDenseLaneOrdinal,&dense,occurrence_counts,wave_last_rows));
}

static SparkGlm52ServingPending *SparkGlm52ServingReservePending(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm52ServingPending *pending;
	int32_t index;
	index = SparkAdapterPendingClaim(state->common.pending,
		sizeof(SparkGlm52ServingPending),state->common.pipeline_slot_count);
	if ( index < 0 )
		return(0);
	pending = &state->pending[index];
	memset(pending,0,sizeof(*pending));
	pending->owner = state;

	pending->core.active = 1u;
	SparkAdapterPendingCapture(&pending->core,submission);

	SparkAdapterCaptureLastRowByLane(submission,pending->last_row_by_lane);
	SparkAdapterCaptureResidentSlotsPerRow(submission,pending->resident_slots);
	return(pending);
}



















static void SparkGlm52ServingCompleteSpeculative(
	SparkGlm52ServingPending *pending,
	SparkGlm52ServingState *state,
	SparkModelServingCompletion *completion)
{
	SparkGlm52ResidentDecodeStageDsparkDraftView *view = &pending->spec_view;
	uint32_t lane,index,rows_per_lane,minimum_committed;
	rows_per_lane = pending->spec_rows_per_lane;
	minimum_committed = rows_per_lane;




	SparkAdapterSpinLockAcquire(&state->speculator_lock);
	for (lane=0u; lane<pending->core.active_sequence_count && pending->spec_verify != 0u; lane++)
	{
		uint32_t slot = pending->resident_slots[lane];
		SparkGlm52ServingLaneSpec *spec = &state->spec_lanes[slot];
		uint32_t draft_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
		uint32_t emissions[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT];
		SparkGlm52DsparkVerifyResult verify_result;
		if ( view->verified_row_count != rows_per_lane )
		{
			completion->status = SPARK_STATUS_INTERNAL_ERROR;
			SparkAdapterSpinLockRelease(&state->speculator_lock);
			return;
		}
		SparkAdapterSpinLockAcquire(&spec->lock);
		for (index=1u; index<rows_per_lane; index++)
			draft_ids[index - 1u] = spec->draft_ids[index];
		SparkAdapterSpinLockRelease(&spec->lock);
		for (index=0u; index<rows_per_lane; index++)
			emissions[index] = pending->output_token_ids[index * pending->core.active_sequence_count + lane];
		if ( SparkGlm52DsparkResolveVerifierTokens(draft_ids,rows_per_lane - 1u,emissions,rows_per_lane,&verify_result) != SPARK_STATUS_OK ||
			verify_result.committed_token_count != view->accepted_token_counts[lane] + 1u )
		{
			completion->status = SPARK_STATUS_INTERNAL_ERROR;
			return;
		}
		spec->last_accepted_depth = view->accepted_token_counts[lane];
		spec->pending_stamp = 1u;
		SparkAdapterSpinLockRelease(&spec->lock);
		if ( verify_result.committed_token_count < minimum_committed )
			minimum_committed = verify_result.committed_token_count;
		(void)SparkGlm52DsparkCompleteVerify(&state->speculator,spec->sequence_id,&verify_result);
	}
	for (lane=0u; lane<pending->core.active_sequence_count; lane++)
	{
		uint32_t slot = pending->resident_slots[lane];
		SparkGlm52ServingLaneSpec *spec = &state->spec_lanes[slot];
		uint32_t index;
		if ( view->draft_token_count > 0u )
		{
			SparkAdapterSpinLockAcquire(&spec->lock);
			spec->sequence_id = pending->spec_row_sequence_ids[lane];
			for (index=0u; index<SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT; index++)
				spec->draft_ids[index] = index < view->draft_token_count ? view->draft_token_ids[lane * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT + index] : 0u;
			spec->have_drafts = 1u;
			(void)SparkGlm52DsparkMarkVerifierTapsReady(&state->speculator,pending->core.identity.request_id,spec->sequence_id,pending->core.identity.sequence_position,&spec->tap_generation);
			SparkAdapterSpinLockRelease(&spec->lock);
		}
	}
	if ( pending->spec_verify != 0u )
	{
		completion->tokens_per_sequence = minimum_committed;
		completion->token_count = pending->core.active_sequence_count * minimum_committed;
		completion->accepted_token_count = 0u;
		completion->completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (lane=0u; lane<pending->core.active_sequence_count; lane++)
		{
			uint32_t slot = pending->resident_slots[lane];
			uint32_t committed;
			SparkAdapterSpinLockAcquire(&state->spec_lanes[slot].lock);
			committed = state->spec_lanes[slot].last_accepted_depth + 1u;
			SparkAdapterSpinLockRelease(&state->spec_lanes[slot].lock);
			completion->accepted_token_count += committed;
			for (index=0u; index<minimum_committed; index++)
				completion->token_ids[lane * minimum_committed + index] = pending->output_token_ids[index * pending->core.active_sequence_count + lane];
		}
	}
	SparkAdapterSpinLockRelease(&state->speculator_lock);
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
	if ( state == 0 || pending->core.active == 0u || driver_completion == 0 )
		return;
	matches = SparkAdapterDriverCompletionMatches(driver_completion,
		pending->core.identity.request_id,pending->core.identity.sequence_id,
		pending->core.identity.sequence_position,state->common.program->program_id);
	SparkAdapterBuildCompletionHeader(&completion,&pending->core);
	completion.status = matches != 0u ? (uint32_t)driver_completion->status : SPARK_STATUS_SCHEMA_ERROR;

	completion.accepted_token_count = driver_completion->accepted_token_count;
	completion.queue_delay_ns = driver_completion->queue_delay_ns;
	completion.service_time_ns = driver_completion->service_time_ns;
	completion.device_memcpy_bytes = driver_completion->device_memcpy_bytes;
	completion.host_staging_bytes = driver_completion->host_staging_bytes;
	if ( matches != 0u )
		completion.residency = driver_completion->residency;
	else
		state->common.orphan_completion_count++;
	if ( state->stage_index + 1u == SPARK_GLM52_SERVING_STAGE_COUNT && completion.status == SPARK_STATUS_OK && pending->spec_active != 0u )
		SparkGlm52ServingCompleteSpeculative(pending,state,&completion);
	if ( state->stage_index + 1u == SPARK_GLM52_SERVING_STAGE_COUNT && completion.status == SPARK_STATUS_OK && pending->spec_active == 0u )
	{
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->core.active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index]];
	}
	pending->core.active = 0u;
	state->common.sink.function(state->common.sink.context,&completion);
}


static void SparkGlm52ServingDestroy(void *adapter_state)
{
	SparkGlm52ServingState *state;
	state = (SparkGlm52ServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkAdapterDestroyReady(&state->common) == 0u )
		return;
	SparkAdapterTeardownDriver(&state->common);
	free(state->spec_stamp_scratch);
	free(state);
}

static SparkStatus SparkGlm52ServingLoadDriver(
	SparkGlm52ServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{



	SparkAdapterDriverContract contract;
	SparkStatus status;
	contract.model_id = SPARK_GLM52_SERVING_DRIVER_MODEL_ID;
	contract.model_revision = GLM52_MODEL_REVISION;
	contract.stage_name = SPARK_GLM52_SERVING_STAGE_NAME;
	contract.target = SPARK_GLM52_SERVING_TARGET;
	contract.description_sha256 = GLM52_CONTRACT_SHA256;
	contract.required_program_flags = SPARK_GLM52_SERVING_REQUIRED_PROGRAM_FLAGS;
	contract.check_kind = 0u;
	contract.node_context = &state->node_context;
	status = SparkAdapterLoadDriver(&state->common,configuration,&contract);
	return(status);
}

static SparkStatus SparkGlm52ServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	return(SparkAdapterValidateConfiguration(&SparkGlm52ServingDescriptor,
		configuration,SPARK_GLM52_SERVING_PROGRAM_NAME,
		SPARK_GLM52_SERVING_STAGE_COUNT));
}










static SparkStatus SparkGlm52ServingPolicyDraftFunction(
	void *context,
	const SparkGlm52DsparkDraftRequest *request,
	SparkGlm52DsparkDraftResult *result);

static SparkStatus SparkGlm52ServingArmSpeculation(SparkGlm52ServingState *state,uint32_t tp_degree)
{
	SparkGlm52DsparkSpeculatorConfiguration configuration;
	SparkGlm52DsparkModelContract contract;
	const char *kill_switch;
	uint32_t lane;
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	kill_switch = getenv("SPARK_GLM52_SERVING_SPECULATE");
	if ( kill_switch != 0 && strcmp(kill_switch,"0") == 0 )
		state->speculate = 0u;
	if ( state->speculate == 0u )
		return(SPARK_STATUS_OK);
	if ( SPARK_GLM52_SERVING_TP_DEGREE != 1u || SPARK_GLM52_SERVING_STAGE_COUNT != 1u || tp_degree != 1u )
	{
		(void)fprintf(stderr,"GLM52-ADAPTER speculation_refused tp_degree=%u stage_count=%u reason=draft_transport_not_wired_for_fanout\n",SPARK_GLM52_SERVING_TP_DEGREE,SPARK_GLM52_SERVING_STAGE_COUNT);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	if ( state->speculate_draft_rows == 0u )
		state->speculate_draft_rows = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT;
	if ( state->speculate_draft_rows < 2u || state->speculate_draft_rows > SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	state->spec_stamp_scratch = (uint32_t *)malloc((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint32_t));
	if ( state->spec_stamp_scratch == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkGlm52DsparkBuildDefaultModelContract(&contract) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_DSPARK_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_GLM52_DSPARK_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.sequence_state_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	configuration.default_speculative_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT;
	configuration.minimum_confidence_milli = SPARK_GLM52_DSPARK_DEFAULT_MIN_CONFIDENCE_MILLI;
	configuration.realtime_minimum_confidence_milli = SPARK_GLM52_DSPARK_DEFAULT_REALTIME_MIN_CONFIDENCE_MILLI;
	configuration.sequence_states = state->spec_sequence_states;
	configuration.draft_function = SparkGlm52ServingPolicyDraftFunction;
	configuration.draft_context = state;
	configuration.model_contract = &contract;
	status = SparkGlm52DsparkInitialize(&state->speculator,&configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (lane=0u; lane<SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
		state->spec_lanes[lane].sequence_id = 0u;
	return(SPARK_STATUS_OK);
}





static SparkStatus SparkGlm52ServingPolicyDraftFunction(
	void *context,
	const SparkGlm52DsparkDraftRequest *request,
	SparkGlm52DsparkDraftResult *result)
{
	SparkGlm52ServingState *state;
	state = (SparkGlm52ServingState *)context;
	if ( state == 0 || request == 0 || result == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	uint32_t lane;
	for (lane=0u; lane<SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		SparkGlm52ServingLaneSpec *spec = &state->spec_lanes[lane];
		uint32_t count,index;
		int matches;


		SparkAdapterSpinLockAcquire(&spec->lock);
		matches = spec->have_drafts != 0u && spec->sequence_id == request->sequence_id;
		if ( matches != 0 )
		{
			count = request->requested_token_count;
			if ( count > SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT )
				count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT;
			result->token_count = count;
			for (index=0u; index<count; index++)
			{
				result->token_ids[index] = spec->draft_ids[index];
				result->confidence_milli[index] = SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE;
			}
		}
		SparkAdapterSpinLockRelease(&spec->lock);
		if ( matches != 0 )
			return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}







static SparkStatus SparkGlm52ServingStageSpeculatorEnvironment(
	const SparkGlm52ServingState *state)
{
	char path[SPARK_INTERNAL_PATH_BYTES];
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->speculate == 0u || SPARK_GLM52_SERVING_TP_DEGREE != 1u ||
		SPARK_GLM52_SERVING_STAGE_COUNT != 1u )
		return(SPARK_STATUS_OK);
	{
		const char *kill_switch = getenv("SPARK_GLM52_SERVING_SPECULATE");
		if ( kill_switch != 0 && strcmp(kill_switch,"0") == 0 )
			return(SPARK_STATUS_OK);
	}
	if ( state->dspark_pack_path[0] == '\0' )
	{
		(void)fprintf(stderr,"GLM52-ADAPTER speculation_refused reason=dspark_pack_path_missing\n");
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( SparkAdapterSetEnvironmentText("SPARK_GLM52_STAGE_SPECULATOR","1") != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	(void)snprintf(path,sizeof(path),"%s/manifest.json",state->dspark_pack_path);
	if ( SparkAdapterSetEnvironmentText("SPARK_GLM52_DSPARK_MANIFEST",path) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	(void)snprintf(path,sizeof(path),"%s/config.json",state->dspark_pack_path);
	if ( SparkAdapterSetEnvironmentText("SPARK_GLM52_DSPARK_CONFIG",path) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	(void)snprintf(path,sizeof(path),"%s/model.safetensors",state->dspark_pack_path);
	if ( SparkAdapterSetEnvironmentText("SPARK_GLM52_DSPARK_SAFETENSORS",path) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkGlm52ServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
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
	(void)SparkAdapterInitializePrologue(&state->common,configuration);
	state->common.pending = &state->pending[0].core;
	state->common.pending_element_bytes = sizeof(state->pending[0]);
	status = SparkGlm52ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > state->common.resident_sequence_capacity) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_GLM52_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingStageSpeculatorEnvironment(state);
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT;


		state->node_context.stage_index = 0u;
		state->node_context.first_layer_index = 0u;
		state->node_context.layer_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_LAYERS_PER_STAGE;
		state->node_context.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->common.resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->common.pipeline_slot_count;
		state->node_context.max_sequence_positions = max_sequence_positions;
		state->node_context.execution_row_capacity = execution_row_capacity;
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
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingArmSpeculation(state,tp_degree);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm52ServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingValidateBoundaries(
	const SparkModelServingSubmission *submission)
{



	if ( submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
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




	status = SparkAdapterValidateSubmissionOpen(&state->common,
		&SparkGlm52ServingDescriptor,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ServingValidateBoundaries(submission);
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
	frame->program_id = state->common.program->program_id;
	frame->execution_stream = state->common.execution_stream;
	frame->buffers = buffer;
	frame->buffer_count = 1u;
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkGlm52ServingDriverCompletion;
	frame->completion_context = pending;
}







static SparkStatus SparkGlm52ServingSubmitSpeculativeFrame(
	SparkGlm52ServingState *state,
	SparkModelDriverFrame *frame)
{
	return(SparkAdapterAdmitFrame(&state->common.driver,
		state->common.driver_instance,state->common.program,0,frame,1u));
}

















static SparkStatus SparkGlm52ServingSubmitSpeculative(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkGlm52ServingPending *pending)
{
	SparkGlm52ResidentDecodeStageBatchView *batch;
	SparkGlm52ResidentDecodeStageFrameContext *context;
	SparkGlm52ResidentDecodeStageDsparkDraftView *view;
	SparkModelDriverBuffer *buffer;
	SparkModelDriverFrame *frame;
	uint32_t lane,row,rows_per_lane,all_have_drafts;
	all_have_drafts = 1u;




	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		int usable;
		if ( slot >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		{
			all_have_drafts = 0u;
			continue;
		}
		SparkAdapterSpinLockAcquire(&state->spec_lanes[slot].lock);
		usable = state->spec_lanes[slot].have_drafts != 0u &&
			state->spec_lanes[slot].sequence_id == submission->row_sequence_ids[submission->row_lane_indices[lane]];
		SparkAdapterSpinLockRelease(&state->spec_lanes[slot].lock);
		if ( !usable )
			all_have_drafts = 0u;
	}
	rows_per_lane = all_have_drafts != 0u ? state->speculate_draft_rows : 1u;
	pending->spec_active = 1u;
	pending->spec_verify = all_have_drafts;
	pending->spec_rows_per_lane = rows_per_lane;
	batch = &pending->spec_batch;
	batch->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->active_sequence_count = submission->active_sequence_count;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = pending->spec_row_positions;
	batch->row_sequence_ids = pending->spec_row_sequence_ids;
	batch->token_ids = pending->spec_row_token_ids;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		uint64_t position = submission->row_positions[submission->row_lane_indices[lane]];
		uint64_t sequence = submission->row_sequence_ids[submission->row_lane_indices[lane]];
		uint32_t token = submission->token_ids[submission->row_lane_indices[lane]];
		for (row=0u; row<rows_per_lane; row++)
		{
			uint32_t expanded = row * submission->active_sequence_count + lane;
			pending->resident_slots[expanded] = slot;
			pending->spec_row_positions[expanded] = position + row;
			pending->spec_row_sequence_ids[expanded] = sequence;
			if ( row != 0u )
				SparkAdapterSpinLockAcquire(&state->spec_lanes[slot].lock);
			pending->spec_row_token_ids[expanded] = row == 0u ? token :
				state->spec_lanes[slot].draft_ids[row];
			if ( row != 0u )
				SparkAdapterSpinLockRelease(&state->spec_lanes[slot].lock);
		}
	}
	batch->row_count = rows_per_lane * submission->active_sequence_count;
	view = &pending->spec_view;
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_VIEW_ABI_VERSION;
	view->descriptor_bytes = sizeof(*view);
	view->requested_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT;
	view->sequence_id = submission->sequence_id;
	view->base_position = batch->row_positions[0];
	view->draft_token_ids = pending->spec_draft_ids;
	view->accepted_token_counts = pending->spec_accept_counts;
	context = &pending->spec_context;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->batch = batch;
	context->dspark_draft = view;
	if ( all_have_drafts != 0u )
	{
		context->flags = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
		context->previous_verify_accepts = state->spec_stamp_scratch;
		for (row=0u; row<batch->row_count; row++)
		{
			uint32_t slot = pending->resident_slots[row];
			uint32_t depth;
			SparkAdapterSpinLockAcquire(&state->spec_lanes[slot].lock);
			depth = state->spec_lanes[slot].pending_stamp != 0u ?
				state->spec_lanes[slot].last_accepted_depth : 0u;
			SparkAdapterSpinLockRelease(&state->spec_lanes[slot].lock);
			state->spec_stamp_scratch[row] = depth;
		}
	}
	else
	{
		context->flags = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
		context->previous_verify_accepts = 0;






		for (lane=0u; lane<SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
		{
			SparkAdapterSpinLockAcquire(&state->spec_lanes[lane].lock);
			state->spec_lanes[lane].have_drafts = 0u;
			SparkAdapterSpinLockRelease(&state->spec_lanes[lane].lock);
		}
	}
	buffer = &pending->spec_buffer;
	memset(buffer,0,sizeof(*buffer));
	buffer->flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
	buffer->address = pending->output_token_ids;
	buffer->bytes = (uint64_t)batch->row_count * sizeof(uint32_t);
	frame = &pending->spec_frame;
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = submission->sequence_id;
	frame->sequence_position = submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = submission->active_sequence_count;
	frame->new_token_count = batch->row_count;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->common.program->program_id;
	frame->execution_stream = state->common.execution_stream;
	frame->buffers = buffer;
	frame->buffer_count = 1u;
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkGlm52ServingDriverCompletion;
	frame->completion_context = pending;
	return(SparkGlm52ServingSubmitSpeculativeFrame(state,frame));
}










static SparkStatus SparkGlm52ServingPrepareVerifyDrafts(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm52DsparkDraftRequest request;
	SparkGlm52DsparkDraftResult draft;
	uint32_t lane;
	SparkStatus status;




	SparkAdapterSpinLockAcquire(&state->speculator_lock);
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		SparkGlm52ServingLaneSpec *spec;
		if ( slot >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		{
			SparkAdapterSpinLockRelease(&state->speculator_lock);
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
		spec = &state->spec_lanes[slot];
		SparkAdapterSpinLockAcquire(&spec->lock);
		if ( spec->have_drafts == 0u )
		{
			SparkAdapterSpinLockRelease(&spec->lock);
			continue;
		}
		request.tap_generation = spec->tap_generation;
		SparkAdapterSpinLockRelease(&spec->lock);
		memset(&request,0,sizeof(request));
		request.abi_version = SPARK_DSPARK_ABI_VERSION;
		request.descriptor_bytes = SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
		request.requested_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT;
		request.sequence_id = submission->row_sequence_ids[submission->row_lane_indices[lane]];
		status = SparkGlm52DsparkEnsureDraft(&state->speculator,&request);
		if ( status == SPARK_STATUS_NOT_FOUND )
		{
			SparkAdapterSpinLockAcquire(&spec->lock);
			spec->have_drafts = 0u;
			SparkAdapterSpinLockRelease(&spec->lock);
			continue;
		}
		if ( status != SPARK_STATUS_OK )
		{
			SparkAdapterSpinLockRelease(&state->speculator_lock);
			return(status);
		}
		status = SparkGlm52DsparkGetDraft(&state->speculator,request.sequence_id,&draft);
		if ( status != SPARK_STATUS_OK )
		{
			SparkAdapterSpinLockRelease(&state->speculator_lock);
			return(status);
		}


		SparkAdapterSpinLockAcquire(&spec->lock);
		if ( draft.token_count < state->speculate_draft_rows )
		{
			spec->have_drafts = 0u;
			SparkAdapterSpinLockRelease(&spec->lock);
			continue;
		}
		memcpy(spec->draft_ids,draft.token_ids,sizeof(spec->draft_ids));
		spec->sequence_id = request.sequence_id;
		spec->have_drafts = 1u;
		SparkAdapterSpinLockRelease(&spec->lock);
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm52ServingForgetAllSpeculation(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		SparkGlm52ServingLaneSpec *spec;
		if ( slot >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
			continue;
		spec = &state->spec_lanes[slot];




		SparkAdapterSpinLockAcquire(&state->speculator_lock);
		if ( spec->sequence_id != 0u )
			(void)SparkGlm52DsparkCancelSequence(&state->speculator,spec->sequence_id);
		SparkAdapterSpinLockRelease(&state->speculator_lock);
		SparkAdapterSpinLockAcquire(&spec->lock);
		spec->sequence_id = 0u;
		spec->tap_generation = 0u;
		spec->have_drafts = 0u;
		spec->last_accepted_depth = 0u;
		spec->pending_stamp = 0u;
		memset(spec->draft_ids,0,sizeof(spec->draft_ids));
		SparkAdapterSpinLockRelease(&spec->lock);
	}
}

static SparkStatus SparkGlm52ServingAdmit(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission,
	SparkModelDriverFrame *frame)
{
	return(SparkAdapterAdmitFrame(&state->common.driver,
		state->common.driver_instance,state->common.program,submission,
		frame,0u));
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
	if ( state->speculate != 0u && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE && submission->active_sequence_count >= 1u )
	{
		status = SparkGlm52ServingPrepareVerifyDrafts(state,submission);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52ServingSubmitSpeculative(state,submission,pending);
		if ( status != SPARK_STATUS_OK )
			pending->core.active = 0u;
		return(status);
	}
	if ( state->speculate != 0u && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		SparkGlm52ServingForgetAllSpeculation(state,submission);
	SparkGlm52ServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
	status = SparkGlm52ServingAdmit(state,submission,&frame);
	if ( status == SPARK_STATUS_OK )
		status = state->common.program->submit(state->common.driver_instance,&frame);
	if ( status != SPARK_STATUS_OK )
		pending->core.active = 0u;
	return(status);
}



static SparkStatus SparkGlm52ServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkGlm52ServingState *state;
	state = (SparkGlm52ServingState *)adapter_state;
	return(SparkAdapterQuiesce(state != 0 ? &state->common : 0,deadline_time_ns));
}

static SparkStatus SparkGlm52ServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkGlm52ServingState *state;
	state = (SparkGlm52ServingState *)adapter_state;
	return(SparkAdapterSnapshot(state != 0 ? &state->common : 0,snapshot));
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
	.progress = SparkModelServingAdapterStreamOrderedProgress,
	.quiesce = SparkGlm52ServingQuiesce,
	.snapshot = SparkGlm52ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkGlm52ServingInterface);
}
