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
/* Deployment-facing geometry: 8 flat ranks, one per TP rank, single PP
 * stage. The residentd fans each submission out to every rank
 * (PARALLEL_FANOUT) and the firmware stage stays STAGE_COUNT=1; the
 * adapter maps flat rank -> tp_rank and pins the firmware stage to 0.
 * The qwen36 TP1-build doctrine applies verbatim: a single-rank serving
 * shape is a compile-time variant (-DSPARK_GLM52_SERVING_STAGE_COUNT=1
 * -DSPARK_GLM52_SERVING_TP_DEGREE=1), and DFlash2 speculation is gated to
 * exactly that shape - drafts materialize inside the one rank's module, so
 * a fanout deployment would need a draft transport that does not exist. */
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
	/* v4: DFlash2 speculator stanza. enabled/draft_count ride every config
	 * so a deployment declares its speculation posture explicitly; the
	 * drafter artifact paths stay with the module's environment contract
	 * (dspark_pack_path records the reference for the manifest audit).
	 * tp_collective is REQUIRED for tp_degree > 1 and FORBIDDEN at 1 - it
	 * left the exact-member list so the single-rank shape can omit it. */
	"speculation_enabled",
	"speculation_draft_count",
	"dspark_pack_path"
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
	/* Speculation frame payload (spec_active != 0u): the expanded verify /
	 * draft-after batch lives here because the async chain reads it after
	 * submit() returns, exactly like resident_slots above. */
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

/* Per resident-slot speculation state: the latest drafts awaiting verify,
 * the accept depth the NEXT submission must stamp, and the policy tap
 * generation handed out when those drafts landed. */
typedef struct SparkGlm52ServingLaneSpec
{
	uint64_t sequence_id;
	uint64_t tap_generation;
	uint32_t have_drafts;
	uint32_t draft_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
	uint32_t last_accepted_depth;
	uint8_t pending_stamp;
} SparkGlm52ServingLaneSpec;

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
	/* DFlash2 speculation (see SPARK_GLM52_SERVING_TP_DEGREE note: armed
	 * only for the single-rank build). */
	uint32_t speculate;
	uint32_t speculate_draft_rows;
	char dspark_pack_path[SPARK_INTERNAL_PATH_BYTES];
	SparkGlm52ServingLaneSpec spec_lanes[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *spec_stamp_scratch;
	SparkGlm52DsparkSpeculator speculator;
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

/* Unsigned member reads go through the shared SparkJsonGetUInt32Member
 * (runtime/json.c) - the naming/audit round retired this adapter's private
 * copy, which every serving adapter had duplicated byte-for-byte. */

static SparkStatus SparkGlm52ServingLoadTpAlgorithms(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,mask;
	token = SparkJsonFindObjectMember(document,object,"algorithms");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	mask = 0u;
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		if ( SparkJsonStringEquals(document,element,"recursive_doubling") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING;
		else if ( SparkJsonStringEquals(document,element,
				"counter_rotating_split_ring") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING;
		else if ( SparkJsonStringEquals(document,element,"direct_all_to_all") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
		else
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	/* The collective implements split-ring and direct-all-to-all only at
	 * tp_degree 4, so TP8 runs recursive_doubling alone and the two
	 * algorithm-specific thresholds must be zero. */
	if ( count != 1u || mask != SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING )
		return(SPARK_STATUS_SCHEMA_ERROR);
	topology->algorithm_mask = mask;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingLoadTpStepRails(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,value;
	SparkStatus status;
	token = SparkJsonFindObjectMember(document,object,"step_rail_indices");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) ||
		SparkJsonGetArrayElementCount(document,token) !=
			SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonGetUInt32(document,element,&value);
		if ( status != SPARK_STATUS_OK || value >=
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
			return(SPARK_STATUS_SCHEMA_ERROR);
		topology->step_rail_indices[index] = value;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingLoadTpRailHosts(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology,
	uint32_t tp_degree)
{
	int32_t element,host_element,token;
	uint32_t host_count,index,rail;
	char *host;
	SparkStatus status;
	token = SparkJsonFindObjectMember(document,object,"rail_peer_hosts");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) ||
		SparkJsonGetArrayElementCount(document,token) !=
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	topology->rail_count = SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT;
	for (rail=0u; rail<topology->rail_count; rail++)
	{
		element = SparkJsonGetArrayElement(document,token,rail);
		if ( element < 0 ||
			!SparkJsonTokenIsType(document,element,SPARK_JSON_TOKEN_ARRAY) )
			return(SPARK_STATUS_SCHEMA_ERROR);
		host_count = SparkJsonGetArrayElementCount(document,element);
		if ( host_count != tp_degree )
			return(SPARK_STATUS_SCHEMA_ERROR);
		for (index=0u; index<host_count; index++)
		{
			host_element = SparkJsonGetArrayElement(document,element,index);
			host = 0;
			status = host_element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
				SparkJsonCopyString(document,host_element,&host);
			if ( status == SPARK_STATUS_OK )
				status = SparkCopyString(
					topology->rail_rank_hosts[rail][index],
					SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,host);
			free(host);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ServingValidateTpCollectiveMembers(
	const SparkJsonDocument *document,
	int32_t object,
	uint32_t backend_kind)
{
	static const char *const base_members[] =
	{
		"backend","backend_module_path","collective_identifier",
		"listen_port","connect_timeout_milli","operation_timeout_milli",
		"peer_hosts","peer_ports"
	};
	static const char *const adaptive_members[] =
	{
		"backend","backend_module_path","collective_identifier",
		"listen_port","connect_timeout_milli","operation_timeout_milli",
		"peer_hosts","peer_ports","algorithms",
		"direct_all_to_all_max_payload_bytes",
		"split_ring_min_payload_bytes","rail_peer_hosts",
		"step_rail_indices"
	};
	const char *const *members;
	uint32_t member_count;
	members = backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT ?
		adaptive_members : base_members;
	member_count = backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT ?
		(uint32_t)(sizeof(adaptive_members) / sizeof(adaptive_members[0])) :
		(uint32_t)(sizeof(base_members) / sizeof(base_members[0]));
	return(SparkJsonValidateObjectMembersExact(document,object,members,
		member_count));
}

static SparkStatus SparkGlm52ServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkGlm52ServingState *state,
	uint32_t tp_degree)
{
	int32_t object,token,element;
	uint32_t count,index,port;
	uint64_t collective_identifier;
	char *host,*relative_backend_path;
	SparkStatus status;
	if ( document == 0 || runtime_root == 0 || state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&state->tp_collective_topology,0,
		sizeof(state->tp_collective_topology));
	state->tp_collective_topology.abi_version =
		SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	state->tp_collective_topology.descriptor_bytes =
		SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	object = SparkJsonFindObjectMember(document,root,"tp_collective");
	if ( object < 0 || !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkJsonFindObjectMember(document,object,"backend");
	if ( token < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkJsonStringEquals(document,token,"nccl") )
		state->tp_collective_backend_kind =
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	else if ( SparkJsonStringEquals(document,token,"hidden_transport") )
		state->tp_collective_backend_kind =
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	else
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkGlm52ServingValidateTpCollectiveMembers(document,object,
		state->tp_collective_backend_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	relative_backend_path = 0;
	token = SparkJsonFindObjectMember(document,object,"backend_module_path");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonCopyString(document,token,&relative_backend_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_backend_path,
			state->tp_collective_backend_path,
			sizeof(state->tp_collective_backend_path));
	free(relative_backend_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	token = SparkJsonFindObjectMember(document,object,"collective_identifier");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt64(document,token,&collective_identifier);
	if ( status != SPARK_STATUS_OK )
		return(status);
	/* A zero identifier is not a configuration: the module would elide every
	 * reduce and serve rank-local math on a TP-degree pack. */
	if ( collective_identifier == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	state->tp_collective_identifier = collective_identifier;
	status = SparkJsonGetUInt32Member(document,object,"listen_port",&port);
	if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	state->tp_listen_port = (uint16_t)port;
	status = SparkJsonGetUInt32Member(document,object,"connect_timeout_milli",&state->tp_connect_timeout_milli);
	if ( status != SPARK_STATUS_OK || state->tp_connect_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	status = SparkJsonGetUInt32Member(document,object,"operation_timeout_milli",&state->tp_operation_timeout_milli);
	if ( status != SPARK_STATUS_OK || state->tp_operation_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	token = SparkJsonFindObjectMember(document,object,"peer_hosts");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != tp_degree )
		return(SPARK_STATUS_SCHEMA_ERROR);
	state->tp_collective_topology.rank_count = count;
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		host = 0;
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(document,element,&host);
		if ( status == SPARK_STATUS_OK )
			status = SparkCopyString(
				state->tp_collective_topology.rank_hosts[index],
				SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES,host);
		free(host);
		if ( status != SPARK_STATUS_OK ||
			state->tp_collective_topology.rank_hosts[index][0] == '\0' )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	}
	token = SparkJsonFindObjectMember(document,object,"peer_ports");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != tp_degree )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,element,&port);
		if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
		state->tp_peer_ports[index] = (uint16_t)port;
	}
	state->tp_collective_control_port_base = state->tp_peer_ports[0];
	for (index=1u; index<count; index++)
	{
		if ( state->tp_peer_ports[index] !=
			(uint16_t)(state->tp_collective_control_port_base + index) )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( state->tp_collective_backend_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		status = SparkGlm52ServingLoadTpAlgorithms(document,object,
			&state->tp_collective_topology);
		if ( status == SPARK_STATUS_OK )
			status = SparkJsonGetUInt32Member(document,object,
				"direct_all_to_all_max_payload_bytes",
				&state->tp_collective_topology.direct_all_to_all_max_payload_bytes);
		if ( status == SPARK_STATUS_OK )
			status = SparkJsonGetUInt32Member(document,object,
				"split_ring_min_payload_bytes",
				&state->tp_collective_topology.split_ring_min_payload_bytes);
		if ( status == SPARK_STATUS_OK &&
			(state->tp_collective_topology.direct_all_to_all_max_payload_bytes != 0u ||
			 state->tp_collective_topology.split_ring_min_payload_bytes != 0u) )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52ServingLoadTpRailHosts(document,object,
				&state->tp_collective_topology,tp_degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm52ServingLoadTpStepRails(document,object,
				&state->tp_collective_topology);
	}
	(void)fprintf(stderr,"GLM52-ADAPTER LoadTpCollective rc=%d backend=%u\n",(int)status,state->tp_collective_backend_kind);
	return(status);
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
	(void)fprintf(stderr,"GLM52-ADAPTER LoadConfiguration rc=%d\n",(int)status);
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


/*
 * Speculative completion: credit the accepted emissions, park the fresh
 * drafts for the next round, and close the policy round.
 *
 * Verify rounds: the module computed each lane's accept depth from the same
 * emissions this code reads; the policy's ResolveVerifierTokens recomputes
 * it from (drafts, emissions) as an independent check - a divergence is an
 * internal error, never a silent mis-credit. Committed tokens per lane are
 * the first `committed` emissions; a multi-lane round credits the MINIMUM
 * across lanes so the interleaved token_ids layout (lane * tps + step)
 * stays dense, exactly qwen36's min-accepted doctrine. Non-verify rounds
 * (draft-after decode) credit one token and only refresh the drafts.
 */
static void SparkGlm52ServingCompleteSpeculative(
	SparkGlm52ServingPending *pending,
	SparkGlm52ServingState *state,
	SparkModelServingCompletion *completion)
{
	SparkGlm52ResidentDecodeStageDsparkDraftView *view = &pending->spec_view;
	uint32_t lane,index,rows_per_lane,minimum_committed;
	rows_per_lane = pending->spec_rows_per_lane;
	minimum_committed = rows_per_lane;
	for (lane=0u; lane<pending->active_sequence_count && pending->spec_verify != 0u; lane++)
	{
		uint32_t slot = pending->resident_slots[lane];
		SparkGlm52ServingLaneSpec *spec = &state->spec_lanes[slot];
		uint32_t draft_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT];
		uint32_t emissions[SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_VERIFY_ROW_COUNT];
		SparkGlm52DsparkVerifyResult verify_result;
		if ( view->verified_row_count != rows_per_lane )
		{
			completion->status = SPARK_STATUS_INTERNAL_ERROR;
			return;
		}
		for (index=1u; index<rows_per_lane; index++)
			draft_ids[index - 1u] = spec->draft_ids[index];
		for (index=0u; index<rows_per_lane; index++)
			emissions[index] = pending->output_token_ids[index * pending->active_sequence_count + lane];
		if ( SparkGlm52DsparkResolveVerifierTokens(draft_ids,rows_per_lane - 1u,emissions,rows_per_lane,&verify_result) != SPARK_STATUS_OK ||
			verify_result.committed_token_count != view->accepted_token_counts[lane] + 1u )
		{
			completion->status = SPARK_STATUS_INTERNAL_ERROR;
			return;
		}
		spec->last_accepted_depth = view->accepted_token_counts[lane];
		spec->pending_stamp = 1u;
		if ( verify_result.committed_token_count < minimum_committed )
			minimum_committed = verify_result.committed_token_count;
		(void)SparkGlm52DsparkCompleteVerify(&state->speculator,spec->sequence_id,&verify_result);
	}
	for (lane=0u; lane<pending->active_sequence_count; lane++)
	{
		uint32_t slot = pending->resident_slots[lane];
		SparkGlm52ServingLaneSpec *spec = &state->spec_lanes[slot];
		uint32_t index;
		if ( view->draft_token_count > 0u )
		{
			spec->sequence_id = pending->spec_row_sequence_ids[lane];
			for (index=0u; index<SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT; index++)
				spec->draft_ids[index] = index < view->draft_token_count ? view->draft_token_ids[lane * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT + index] : 0u;
			spec->have_drafts = 1u;
			(void)SparkGlm52DsparkMarkVerifierTapsReady(&state->speculator,pending->request_id,spec->sequence_id,pending->sequence_position,&spec->tap_generation);
		}
	}
	if ( pending->spec_verify != 0u )
	{
		completion->tokens_per_sequence = minimum_committed;
		completion->token_count = pending->active_sequence_count * minimum_committed;
		completion->accepted_token_count = 0u;
		completion->completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (lane=0u; lane<pending->active_sequence_count; lane++)
		{
			uint32_t committed = state->spec_lanes[pending->resident_slots[lane]].last_accepted_depth + 1u;
			completion->accepted_token_count += committed;
			for (index=0u; index<minimum_committed; index++)
				completion->token_ids[lane * minimum_committed + index] = pending->output_token_ids[index * pending->active_sequence_count + lane];
		}
	}
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
	if ( state->stage_index + 1u == SPARK_GLM52_SERVING_STAGE_COUNT && completion.status == SPARK_STATUS_OK && pending->spec_active != 0u )
		SparkGlm52ServingCompleteSpeculative(pending,state,&completion);
	if ( state->stage_index + 1u == SPARK_GLM52_SERVING_STAGE_COUNT && completion.status == SPARK_STATUS_OK && pending->spec_active == 0u )
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
	free(state->spec_stamp_scratch);
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
	request.kv_logical_page_capacity =
		configuration->runtime_limits.kv_logical_page_capacity;
	request.kv_physical_page_capacity =
		configuration->runtime_limits.kv_physical_page_capacity;
	request.kv_backing_directory = configuration->kv_backing_directory;
	request.kv_backing_maximum_bytes =
		configuration->kv_backing_maximum_bytes;
	request.execution_stream = configuration->execution_stream;
	request.completion_function = SparkGlm52ServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkGlm52ServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	(void)fprintf(stderr,"GLM52-ADAPTER LoadDriver rc=%d\n",(int)status);
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
	if ( configuration->stage_index >= SPARK_GLM52_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_GLM52_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

/*
 * Arm DFlash2 speculation. The single-rank build is a hard gate, not a
 * preference: drafts materialize inside THIS process's module (the drafter
 * archive links into the firmware stage), so a fanout deployment would need
 * a draft-transport layer that does not exist yet - refuse loudly instead of
 * drafting on one rank and silently verifying garbage on the others. The
 * draft-row count is the verify depth: rows = [anchor, d_1..d_{rows-1}] out
 * of the drafter's 7 emitted ids whose first slot restates the anchor.
 */
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
	(void)fprintf(stderr,"GLM52-ADAPTER speculation_ready draft_rows=%u pack=%s\n",state->speculate_draft_rows,state->dspark_pack_path[0] != '\0' ? state->dspark_pack_path : "(module env)");
	return(SPARK_STATUS_OK);
}

/* Policy transport stub: the drafter runs inside the module behind a frame,
 * so by the time the adapter calls EnsureDraft for a lane the ids already
 * sit in that lane's spec record. Confidence reports full until the draft
 * view carries the backend's per-token confidence upstream. */
static SparkStatus SparkGlm52ServingPolicyDraftFunction(
	void *context,
	const SparkGlm52DsparkDraftRequest *request,
	SparkGlm52DsparkDraftResult *result)
{
	SparkGlm52ServingState *state;
	uint32_t index;
	state = (SparkGlm52ServingState *)context;
	if ( state == 0 || request == 0 || result == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; index++)
	{
		SparkGlm52ServingLaneSpec *spec = &state->spec_lanes[index];
		if ( spec->have_drafts != 0u && spec->sequence_id == request->sequence_id )
		{
			uint32_t count;
			count = request->requested_token_count;
			if ( count > SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT )
				count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT;
			result->token_count = count;
			for (index=0u; index<count; index++)
			{
				result->token_ids[index] = spec->draft_ids[index];
				result->confidence_milli[index] = SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE;
			}
			return(SPARK_STATUS_OK);
		}
	}
	return(SPARK_STATUS_NOT_FOUND);
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
	status = SparkGlm52ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > state->resident_sequence_capacity) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_GLM52_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT;
		/* Firmware stage is always 0; the deployment's flat rank index is the
		 * TP rank, cross-checked against the stage config below. */
		state->node_context.stage_index = 0u;
		state->node_context.first_layer_index = 0u;
		state->node_context.layer_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_LAYERS_PER_STAGE;
		state->node_context.expert_weight_codec = GLM52_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->pipeline_slot_count;
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
	/* Every TP8 fanout rank runs the full single-stage firmware: it owns the
	 * embedding and the head, so it accepts no hidden boundaries and no DSA
	 * sidebands. */
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
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkGlm52ServingDescriptor,&state->runtime_limits,submission);
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
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = buffer;
	frame->buffer_count = 1u;
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkGlm52ServingDriverCompletion;
	frame->completion_context = pending;
}


/* ---- DFlash2 speculative decode (single-rank build) ---------------- */

/* Admission evaluates the EXPANDED frame (the module sees rows_per_lane x
 * lanes), never the one-row engine submission - same doctrine as qwen36's
 * SparkAdmissionRequestFromFrame path. */
static SparkStatus SparkGlm52ServingSubmitSpeculativeFrame(
	SparkGlm52ServingState *state,
	SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	status = SparkAdmissionRequestFromFrame(state->program->program_id,frame,0,0u,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkAdmissionEvaluateAndApply(state->driver.interface,state->driver_instance,&request,frame,&decision);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,frame);
	return(status);
}

/*
 * Build ONE driver frame out of a one-row-per-lane decode submission.
 *
 * With drafts on every lane: a SPECULATIVE_VERIFY prefill-kind frame of
 * speculate_draft_rows rows per lane - row 0 restates the engine's token
 * (the previous round's committed emission), rows 1.. carry the drafter's
 * proposals d_1..d_{rows-1}; d_0 is never fed (it predicts the position the
 * committed token already occupies - qwen36's MTP doctrine). The accept
 * stamps of the PREVIOUS verify round ride along for the lanes that owe
 * them; the module reconciles its position books from them before the
 * ordinary continuity check runs.
 *
 * Without drafts (bootstrap or post-fallback): a plain one-row decode frame
 * with DSPARK_DRAFT_AFTER, which stages this round's anchor into the
 * drafter and produces the block the NEXT submission verifies.
 */
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
		if ( slot >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || state->spec_lanes[slot].have_drafts == 0u || state->spec_lanes[slot].sequence_id != submission->row_sequence_ids[submission->row_lane_indices[lane]] )
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
			pending->spec_row_token_ids[expanded] = row == 0u ? token :
				state->spec_lanes[slot].draft_ids[row];
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
			state->spec_stamp_scratch[row] = state->spec_lanes[slot].pending_stamp != 0u ?
				state->spec_lanes[slot].last_accepted_depth : 0u;
		}
	}
	else
	{
		context->flags = SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
		context->previous_verify_accepts = 0;
		/* This round moves every lane's anchor: stale drafts would verify
		 * against a context that no longer exists. Drop them. */
		for (lane=0u; lane<SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
			state->spec_lanes[lane].have_drafts = 0u;
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
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = buffer;
	frame->buffer_count = 1u;
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SparkGlm52ServingDriverCompletion;
	frame->completion_context = pending;
	return(SparkGlm52ServingSubmitSpeculativeFrame(state,frame));
}


/*
 * Draft retrieval for the lanes of this submission: drive the neutral
 * policy (taps were marked ready when the drafting frame completed), pull
 * the verified draft through GetDraft, and park it in the lane's spec
 * record for the frame builder. A lane whose policy state says "no draft"
 * simply reports no drafts - the caller falls back to a plain drafting
 * round for EVERY lane, keeping the batch shape uniform.
 */
static SparkStatus SparkGlm52ServingPrepareVerifyDrafts(
	SparkGlm52ServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm52DsparkDraftRequest request;
	SparkGlm52DsparkDraftResult draft;
	uint32_t lane;
	SparkStatus status;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		SparkGlm52ServingLaneSpec *spec;
		if ( slot >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		spec = &state->spec_lanes[slot];
		if ( spec->have_drafts == 0u )
			continue;
		memset(&request,0,sizeof(request));
		request.abi_version = SPARK_DSPARK_ABI_VERSION;
		request.descriptor_bytes = SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
		request.requested_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSPARK_DRAFT_TOKEN_COUNT;
		request.sequence_id = submission->row_sequence_ids[submission->row_lane_indices[lane]];
		request.tap_generation = spec->tap_generation;
		status = SparkGlm52DsparkEnsureDraft(&state->speculator,&request);
		if ( status == SPARK_STATUS_NOT_FOUND )
		{
			spec->have_drafts = 0u;
			continue;
		}
		if ( status != SPARK_STATUS_OK )
			return(status);
		status = SparkGlm52DsparkGetDraft(&state->speculator,request.sequence_id,&draft);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( draft.token_count < state->speculate_draft_rows )
		{
			/* Policy (confidence) truncated below the verify depth: not a
			 * failure, just nothing worth verifying this round. */
			spec->have_drafts = 0u;
			continue;
		}
		memcpy(spec->draft_ids,draft.token_ids,sizeof(spec->draft_ids));
		spec->sequence_id = request.sequence_id;
		spec->have_drafts = 1u;
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
		if ( spec->sequence_id != 0u )
			(void)SparkGlm52DsparkCancelSequence(&state->speculator,spec->sequence_id);
		memset(spec,0,sizeof(*spec));
	}
}

static SparkStatus SparkGlm52ServingAdmit(
	SparkGlm52ServingState *state,
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
			pending->active = 0u;
		return(status);
	}
	if ( state->speculate != 0u && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		SparkGlm52ServingForgetAllSpeculation(state,submission);
	SparkGlm52ServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
	status = SparkGlm52ServingAdmit(state,submission,&frame);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status != SPARK_STATUS_OK )
		pending->active = 0u;
	return(status);
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
	.progress = SparkModelServingAdapterStreamOrderedProgress,
	.quiesce = SparkGlm52ServingQuiesce,
	.snapshot = SparkGlm52ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkGlm52ServingInterface);
}
