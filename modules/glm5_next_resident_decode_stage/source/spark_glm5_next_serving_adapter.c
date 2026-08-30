#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm5_next_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_model_driver_support.h"

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM5_NEXT_EXPERT_CODEC_NAME
#error "GLM5_NEXT_EXPERT_CODEC_NAME must name the exact package expert codec"
#endif
#ifndef GLM5_NEXT_MODEL_REVISION
#error "GLM5_NEXT_MODEL_REVISION must name the exact source snapshot"
#endif
#ifndef GLM5_NEXT_CONTRACT_SHA256
#error "GLM5_NEXT_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_GLM5_NEXT_SERVING_ADAPTER_ID \
	"spark.glm5_next.serving-adapter.tp8.expert_" GLM5_NEXT_EXPERT_CODEC_NAME ".v1"
/* Deployment-facing geometry: 8 flat ranks, one per TP rank, single PP
 * stage. The residentd fans each submission out to every rank
 * (PARALLEL_FANOUT) and the firmware stage stays STAGE_COUNT=1; the
 * adapter maps flat rank -> tp_rank and pins the firmware stage to 0. */
/* glm5_next: TP16 whole-stack - every rank holds all 45 weight layers
 * (the deployment counts each rank as a stage; PARALLEL_FANOUT fans each
 * submission to all 16, exactly glm52's TP8 shape at 16). The MTP layer
 * rides the spec path. NOTE: HIDDEN_TRANSPORT must NOT be declared with
 * FANOUT unless HYBRID_TP_PP - the descriptor validator rejects that
 * combination (found at bring-up); the TP collective rides the node
 * context, not the adapter capability bits. */
#define SPARK_GLM5_NEXT_SERVING_STAGE_COUNT 16u
#define SPARK_GLM5_NEXT_SERVING_TP_DEGREE 16u
#define SPARK_GLM5_NEXT_SERVING_STAGE_LAYERS \
	{45u,45u,45u,45u,45u,45u,45u,45u,45u,45u,45u,45u,45u,45u,45u,45u}
#define SPARK_GLM5_NEXT_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
#define SPARK_GLM5_NEXT_SERVING_MODEL_ID "zai-org/GLM-5.3-Flash"
#define SPARK_GLM5_NEXT_SERVING_DRIVER_MODEL_ID \
	"zai.glm-5.3-flash.resident-decode-stage-firmware"
#define SPARK_GLM5_NEXT_SERVING_STAGE_NAME "glm5_next_resident_decode_stage"
#define SPARK_GLM5_NEXT_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_GLM5_NEXT_SERVING_TARGET \
	"cuda.sm121.glm5_next.resident_decode_stage.bf16.expert_" GLM5_NEXT_EXPERT_CODEC_NAME
#define SPARK_GLM5_NEXT_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL)

static const char *const SparkGlm5NextServingConfigurationMembers[] =
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

typedef struct SparkGlm5NextServingPending
{
	struct SparkGlm5NextServingState *owner;
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
	uint32_t last_row_by_lane[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
} SparkGlm5NextServingPending;

typedef struct SparkGlm5NextServingState
{
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkGlm5NextResidentDecodeStageNodeContext node_context;
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
	SparkGlm5NextServingPending pending[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkGlm5NextServingState;

static const SparkModelServingAdapterDescriptor SparkGlm5NextServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		SPARK_GLM5_NEXT_SERVING_TOPOLOGY_FLAG |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE,
	.stage_count = SPARK_GLM5_NEXT_SERVING_STAGE_COUNT,
	.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT,
	.boundary_element_bytes = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_inflight_submission_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequence_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.cache_block_token_count = 64u, /* the module's KV block geometry
	 * (SparkCeilDivU32(max_sequence_positions, 64) pages; the arena's
	 * block_token_count) — 0 left the engine's prefill spans uncapped and
	 * the row budget alone shaped every pass (the prefill-slowness lever) */
	.max_output_token_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = 0u,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_AT_POSITION_ZERO,
	.adapter_id = SPARK_GLM5_NEXT_SERVING_ADAPTER_ID,
	.model_id = SPARK_GLM5_NEXT_SERVING_MODEL_ID,
	.model_revision = GLM5_NEXT_MODEL_REVISION,
	.driver_program_name = SPARK_GLM5_NEXT_SERVING_PROGRAM_NAME,
	.artifact_sha256 = GLM5_NEXT_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_GLM5_NEXT_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds = {0u},
	.boundary_sideband_bytes_per_sequence = {0u}
};

static int32_t SparkGlm5NextServingJsonMember(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,root,name));
}

static SparkStatus SparkGlm5NextServingJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkGlm5NextServingJsonMember(document,root,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkGlm5NextServingLoadTpAlgorithms(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,mask;
	token = SparkGlm5NextServingJsonMember(document,object,"algorithms");
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

static SparkStatus SparkGlm5NextServingLoadTpStepRails(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,value;
	SparkStatus status;
	token = SparkGlm5NextServingJsonMember(document,object,"step_rail_indices");
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

static SparkStatus SparkGlm5NextServingLoadTpRailHosts(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology,
	uint32_t tp_degree)
{
	int32_t element,host_element,token;
	uint32_t host_count,index,rail;
	char *host;
	SparkStatus status;
	token = SparkGlm5NextServingJsonMember(document,object,"rail_peer_hosts");
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

static SparkStatus SparkGlm5NextServingValidateTpCollectiveMembers(
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

static SparkStatus SparkGlm5NextServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkGlm5NextServingState *state,
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
	object = SparkGlm5NextServingJsonMember(document,root,"tp_collective");
	if ( object < 0 || !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkGlm5NextServingJsonMember(document,object,"backend");
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
	status = SparkGlm5NextServingValidateTpCollectiveMembers(document,object,
		state->tp_collective_backend_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	relative_backend_path = 0;
	token = SparkGlm5NextServingJsonMember(document,object,"backend_module_path");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonCopyString(document,token,&relative_backend_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_backend_path,
			state->tp_collective_backend_path,
			sizeof(state->tp_collective_backend_path));
	free(relative_backend_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	token = SparkGlm5NextServingJsonMember(document,object,"collective_identifier");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt64(document,token,&collective_identifier);
	if ( status != SPARK_STATUS_OK )
		return(status);
	/* Identifier zero is the degraded single-rank bringup mode: the module
	 * keeps the pack's tp geometry but runs with every reduce elided. */
	state->tp_collective_identifier = collective_identifier;
	status = SparkGlm5NextServingJsonUnsigned(document,object,"listen_port",&port);
	if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	state->tp_listen_port = (uint16_t)port;
	status = SparkGlm5NextServingJsonUnsigned(document,object,"connect_timeout_milli",&state->tp_connect_timeout_milli);
	if ( status != SPARK_STATUS_OK || state->tp_connect_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	status = SparkGlm5NextServingJsonUnsigned(document,object,"operation_timeout_milli",&state->tp_operation_timeout_milli);
	if ( status != SPARK_STATUS_OK || state->tp_operation_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	token = SparkGlm5NextServingJsonMember(document,object,"peer_hosts");
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
	token = SparkGlm5NextServingJsonMember(document,object,"peer_ports");
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
		status = SparkGlm5NextServingLoadTpAlgorithms(document,object,
			&state->tp_collective_topology);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextServingJsonUnsigned(document,object,
				"direct_all_to_all_max_payload_bytes",
				&state->tp_collective_topology.direct_all_to_all_max_payload_bytes);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextServingJsonUnsigned(document,object,
				"split_ring_min_payload_bytes",
				&state->tp_collective_topology.split_ring_min_payload_bytes);
		if ( status == SPARK_STATUS_OK &&
			(state->tp_collective_topology.direct_all_to_all_max_payload_bytes != 0u ||
			 state->tp_collective_topology.split_ring_min_payload_bytes != 0u) )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextServingLoadTpRailHosts(document,object,
				&state->tp_collective_topology,tp_degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextServingLoadTpStepRails(document,object,
				&state->tp_collective_topology);
	}
	(void)fprintf(stderr,"GLM5_NEXT-ADAPTER LoadTpCollective rc=%d backend=%u\n",(int)status,state->tp_collective_backend_kind);
	return(status);
}

static SparkStatus SparkGlm5NextServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkGlm5NextServingState *state,
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
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkGlm5NextServingConfigurationMembers,(uint32_t)(sizeof(SparkGlm5NextServingConfigurationMembers) / sizeof(SparkGlm5NextServingConfigurationMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_GLM5_NEXT_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkGlm5NextServingJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM5_NEXT_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkGlm5NextServingJsonMember(&document,root,"expert_weight_codec") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,GLM5_NEXT_EXPERT_CODEC_NAME)) )
		status = SPARK_STATUS_TARGET_MISMATCH;
	token = status == SPARK_STATUS_OK ? SparkGlm5NextServingJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingJsonUnsigned(&document,root,"execution_row_capacity",execution_row_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingJsonUnsigned(&document,root,"decode_split_context_threshold",decode_split_context_threshold);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingJsonUnsigned(&document,root,"tp_degree",tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingJsonUnsigned(&document,root,"tp_rank",tp_rank);
	if ( status == SPARK_STATUS_OK && (*tp_degree == 0u || *tp_rank >= *tp_degree) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingLoadTpCollective(&document,root,runtime_root,state,*tp_degree);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	(void)fprintf(stderr,"GLM5_NEXT-ADAPTER LoadConfiguration rc=%d\n",(int)status);
	return(status);
}

static SparkStatus SparkGlm5NextServingValidateRowOrder(
	const SparkGlm5NextServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t last_position[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
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

static SparkGlm5NextServingPending *SparkGlm5NextServingReservePending(
	SparkGlm5NextServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm5NextServingPending *pending;
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

static void SparkGlm5NextServingOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkGlm5NextServingState *state;
	(void)driver_completion;
	state = (SparkGlm5NextServingState *)completion_context;
	if ( state != 0 )
		state->orphan_completion_count++;
}

static void SparkGlm5NextServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkGlm5NextServingPending *pending;
	SparkGlm5NextServingState *state;
	SparkModelServingCompletion completion;
	uint32_t index,matches,raw_accepted;
	pending = (SparkGlm5NextServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || pending->active == 0u || driver_completion == 0 )
		return;
	raw_accepted = driver_completion->accepted_token_count;
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
		/* Wire contract (SparkModelServingAdapterValidateCompletion): a
		 * non-OK completion must carry completion_flags == 0 and
		 * accepted_token_count == 0, or the residentd rejects the STRUCT
		 * with INVALID_ARGUMENT before reading the driver's true status
		 * — masking the real failure as status 1/reason 2. */
		completion.accepted_token_count = 0u;
		completion.completion_flags = 0u;
	}
	if ( completion.status == SPARK_STATUS_OK )
	{
		/* Pure TP fanout (this adapter's only topology): every rank runs the
		 * whole stack and the head reduce is a U64-max argmax reduction, so
		 * every rank holds the SAME global token. The validator admits a
		 * token payload from every stage of a parallel fanout, and the
		 * client face is whichever rank the API connects to (rank 0), so
		 * the old final-stage-only gate left that rank's clients with
		 * status-only completions forever. */
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index]];
	}
	fprintf(stderr,"G5N-DBG completion emit: sub %llu status %u flags %u tokcnt %u tps %u acc %u raw_acc %u ext %u resid_zero %d\n",
		(unsigned long long)completion.submission_id,(unsigned)completion.status,
		(unsigned)completion.completion_flags,(unsigned)completion.token_count,
		(unsigned)completion.tokens_per_sequence,
		(unsigned)completion.accepted_token_count,
		(unsigned)raw_accepted,
		(unsigned)completion.model_extension_bytes,
		(int)(completion.residency.word0 == 0u));
	pending->active = 0u;
	state->completion_function(state->completion_context,&completion);
}

static void SparkGlm5NextServingDriverWake(void *wake_context)
{
	SparkGlm5NextServingState *state;
	state = (SparkGlm5NextServingState *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}

static uint32_t SparkGlm5NextServingAvailableSubmissionCount(
	const SparkGlm5NextServingState *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += state->pending[index].active == 0u ? 1u : 0u;
	return(available);
}

static void SparkGlm5NextServingDestroy(void *adapter_state)
{
	SparkGlm5NextServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkGlm5NextServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkGlm5NextServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
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

static SparkStatus SparkGlm5NextServingLoadDriver(
	SparkGlm5NextServingState *state,
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
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_GLM5_NEXT_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,GLM5_NEXT_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_GLM5_NEXT_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_GLM5_NEXT_SERVING_TARGET) != 0 || strcmp(descriptor->model_description_sha256,GLM5_NEXT_CONTRACT_SHA256) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,configuration->driver_program_name);
	if ( state->program == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( state->driver.interface->admit == 0 || state->program->submit == 0 || SparkModelDriverProgramSupportsRuntimeLimits(state->program,SPARK_GLM5_NEXT_SERVING_REQUIRED_PROGRAM_FLAGS,state->pipeline_slot_count,state->max_active_sequence_count,state->max_input_row_count,state->resident_sequence_capacity) == 0u )
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
	request.completion_function = SparkGlm5NextServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkGlm5NextServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	(void)fprintf(stderr,"GLM5_NEXT-ADAPTER LoadDriver rc=%d\n",(int)status);
	return(status == SPARK_STATUS_OK && state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status);
}

static SparkStatus SparkGlm5NextServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SparkGlm5NextServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_GLM5_NEXT_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_GLM5_NEXT_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkGlm5NextServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
	uint32_t decode_split_context_threshold;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkGlm5NextServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkGlm5NextServingState *)calloc(1u,sizeof(*state));
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
	status = SparkGlm5NextServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&decode_split_context_threshold,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > state->resident_sequence_capacity || decode_split_context_threshold > max_sequence_positions) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_GLM5_NEXT_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_STAGE_COUNT;
		/* Firmware stage is always 0; the deployment's flat rank index is the
		 * TP rank, cross-checked against the stage config below. */
		state->node_context.stage_index = 0u;
		state->node_context.first_layer_index = 0u;
		state->node_context.layer_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE;
		state->node_context.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->pipeline_slot_count;
		state->node_context.max_sequence_positions = max_sequence_positions;
		state->node_context.execution_row_capacity = execution_row_capacity;
		state->node_context.decode_split_context_threshold = decode_split_context_threshold;
		state->node_context.tp_degree = tp_degree;
		state->node_context.tp_rank = tp_rank;
		state->node_context.stage_pack_path = state->stage_pack_path;
		state->node_context.model_revision = GLM5_NEXT_MODEL_REVISION;
		state->node_context.tp_collective_backend_kind = state->tp_collective_backend_kind;
		state->node_context.tp_collective_identifier = state->tp_collective_identifier;
		state->node_context.tp_connect_timeout_milli = state->tp_connect_timeout_milli;
		state->node_context.tp_operation_timeout_milli = state->tp_operation_timeout_milli;
		state->node_context.tp_collective_control_port_base = state->tp_collective_control_port_base;
		state->node_context.tp_collective_topology = state->tp_collective_topology;
		state->node_context.tp_collective_backend_module_path = state->tp_collective_backend_path;
		state->node_context.kv_backing_directory = configuration->kv_backing_directory;
		state->node_context.kv_backing_maximum_bytes = configuration->kv_backing_maximum_bytes;
		status = SparkGlm5NextServingLoadDriver(state,configuration);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextServingValidateBoundaries(
	const SparkGlm5NextServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	/* Every TP8 fanout rank runs the full single-stage firmware: it owns the
	 * embedding and the head, so it accepts no hidden boundaries and no DSA
	 * sidebands. */
	if ( submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)boundary_bytes;
	(void)state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm5NextServingState *state;
	SparkStatus status;
	state = (SparkGlm5NextServingState *)adapter_state;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkGlm5NextServingDescriptor,&state->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"G5N-DBG validate: runtime_submission -> %d (kind %u rows %u lanes %u ext %u)\n",
			(int)status,submission->work_kind,submission->row_count,submission->active_sequence_count,submission->model_extension_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingValidateBoundaries(state,submission);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"G5N-DBG validate: boundaries -> %d\n",(int)status);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingValidateRowOrder(state,submission);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"G5N-DBG validate: row_order -> %d\n",(int)status);
	if ( status == SPARK_STATUS_OK && submission->model_extension_bytes != 0u )
	{
		fprintf(stderr,"G5N-DBG validate: model_extension_bytes=%u kind=%u\n",
			submission->model_extension_bytes,submission->model_extension_kind);
		status = SPARK_STATUS_UNSUPPORTED;
	}
	return(status);
}

static void SparkGlm5NextServingBuildFrame(
	const SparkGlm5NextServingState *state,
	const SparkModelServingSubmission *submission,
	SparkGlm5NextServingPending *pending,
	SparkGlm5NextResidentDecodeStageBatchView *batch,
	SparkGlm5NextResidentDecodeStageFrameContext *context,
	SparkModelDriverBuffer *buffer,
	SparkModelDriverFrame *frame)
{
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = submission->row_count;
	batch->active_sequence_count = submission->active_sequence_count;
	batch->token_ids = submission->token_ids;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = submission->row_positions;
	batch->row_sequence_ids = submission->row_sequence_ids;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
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
	frame->completion_function = SparkGlm5NextServingDriverCompletion;
	frame->completion_context = pending;
}

static SparkStatus SparkGlm5NextServingAdmit(
	SparkGlm5NextServingState *state,
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

static SparkStatus SparkGlm5NextServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm5NextServingState *state;
	SparkGlm5NextServingPending *pending;
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffer;
	SparkModelDriverFrame frame;
	SparkStatus status;
	state = (SparkGlm5NextServingState *)adapter_state;
	status = SparkGlm5NextServingValidateSubmission(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkGlm5NextServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	SparkGlm5NextServingBuildFrame(state,submission,pending,&batch,&context,&buffer,&frame);
	status = SparkGlm5NextServingAdmit(state,submission,&frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"G5N-DBG submit: admit -> %d\n",(int)status);
	if ( status == SPARK_STATUS_OK )
	{
		status = state->program->submit(state->driver_instance,&frame);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"G5N-DBG submit: program->submit -> %d\n",(int)status);
	}
	if ( status != SPARK_STATUS_OK )
		pending->active = 0u;
	return(status);
}

static SparkStatus SparkGlm5NextServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkGlm5NextServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkGlm5NextServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SparkGlm5NextServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SparkGlm5NextServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

static SparkStatus SparkGlm5NextServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkGlm5NextServingState *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm5NextServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkGlm5NextServingAvailableSubmissionCount(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SparkGlm5NextServingAvailableSubmissionCount(state);
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

static const SparkModelServingAdapterInterface SparkGlm5NextServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkGlm5NextServingDescriptor,
	.initialize = SparkGlm5NextServingInitialize,
	.destroy = SparkGlm5NextServingDestroy,
	.validate_submission = SparkGlm5NextServingValidateSubmission,
	.submit = SparkGlm5NextServingSubmit,
	.progress = SparkGlm5NextServingProgress,
	.quiesce = SparkGlm5NextServingQuiesce,
	.snapshot = SparkGlm5NextServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkGlm5NextServingInterface);
}
