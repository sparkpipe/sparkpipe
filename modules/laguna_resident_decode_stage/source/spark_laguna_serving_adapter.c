#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_laguna_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_laguna_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#include "sparkpipe/spark_model_driver_support.h"

#ifndef LAGUNA_EXPERT_WEIGHT_CODEC
#error "LAGUNA_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef LAGUNA_EXPERT_CODEC_NAME
#error "LAGUNA_EXPERT_CODEC_NAME must name the exact package expert codec"
#endif
#ifndef LAGUNA_MODEL_REVISION
#error "LAGUNA_MODEL_REVISION must name the exact source snapshot"
#endif
#ifndef LAGUNA_CONTRACT_SHA256
#error "LAGUNA_CONTRACT_SHA256 must identify the exact package contract"
#endif

#define SPARK_LAGUNA_SERVING_ADAPTER_ID \
	"spark.laguna.serving-adapter.tp8.expert_" LAGUNA_EXPERT_CODEC_NAME ".v1"
#define SPARK_LAGUNA_SERVING_STAGE_COUNT 16u
#define SPARK_LAGUNA_SERVING_TP_DEGREE 8u
#define SPARK_LAGUNA_SERVING_PIPELINE_STAGES 2u
#define SPARK_LAGUNA_SERVING_STAGE_LAYERS \
	{24u,24u,24u,24u,24u,24u,24u,24u,24u,24u,24u,24u,24u,24u,24u,24u}
#define SPARK_LAGUNA_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
#define SPARK_LAGUNA_SERVING_MODEL_ID "poolside/Laguna-S-2.1"
#define SPARK_LAGUNA_SERVING_DRIVER_MODEL_ID \
	"laguna.laguna-s-2.1.resident-decode-stage-firmware"
#define SPARK_LAGUNA_SERVING_STAGE_NAME "laguna_resident_decode_stage"
#define SPARK_LAGUNA_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_LAGUNA_SERVING_TARGET \
	"cuda.sm121.laguna.resident_decode_stage.bf16.expert_" LAGUNA_EXPERT_CODEC_NAME
#define SPARK_LAGUNA_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL)


static const char *const SparkLagunaServingConfigurationMembers[] =
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

typedef struct SparkLagunaServingPending
{
	struct SparkLagunaServingState *owner;
	atomic_uint active;
	uint32_t row_count;
	uint32_t lane_count;
	uint32_t active_sequence_count;
	uint32_t work_kind;
	uint64_t submission_id;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	SparkLagunaResidentDecodeStageBatchView batch;
	SparkLagunaResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffer;
	SparkModelDriverFrame frame;
	SparkModelDriverCacheLane cache_lanes[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_row_by_lane[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t input_token_ids[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint64_t row_positions[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint64_t row_sequence_ids[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
} SparkLagunaServingPending;

typedef struct SparkLagunaServingState
{
	SparkLoadedModelDriver driver;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkLagunaResidentDecodeStageNodeContext node_context;
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
	atomic_uint quiescing;
	atomic_uint reset_active;
	atomic_uint_fast64_t reset_generation;
	atomic_uint_fast64_t orphan_completion_count;
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
	SparkLagunaServingPending pending[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkLagunaServingState;

static _Thread_local SparkModelDriverCacheLane SparkLagunaServingCacheScratch[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

static const SparkModelServingAdapterDescriptor SparkLagunaServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_LAGUNA_SERVING_TOPOLOGY_FLAG |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE,
	.stage_count = SPARK_LAGUNA_SERVING_STAGE_COUNT,
	.layer_count = SPARK_LAGUNA_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT,
	.boundary_element_bytes = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = LAGUNA_EXPERT_WEIGHT_CODEC,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.cache_block_token_count = SPARK_LAGUNA_MODEL_KV_PAGE_SLOTS,
	.max_inflight_submission_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequence_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.adapter_id = SPARK_LAGUNA_SERVING_ADAPTER_ID,
	.model_id = SPARK_LAGUNA_SERVING_MODEL_ID,
	.model_revision = LAGUNA_MODEL_REVISION,
	.driver_program_name = SPARK_LAGUNA_SERVING_PROGRAM_NAME,
	.artifact_sha256 = LAGUNA_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_LAGUNA_SERVING_STAGE_LAYERS,
	.boundary_sideband_kinds = {0u},
	.boundary_sideband_bytes_per_sequence = {0u}
};

static int32_t SparkLagunaServingJsonMember(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,root,name));
}

static SparkStatus SparkLagunaServingJsonUnsigned(
	const SparkJsonDocument *document,
	int32_t root,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkLagunaServingJsonMember(document,root,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkLagunaServingLoadTpAlgorithms(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,mask;
	token = SparkLagunaServingJsonMember(document,object,"algorithms");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
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
		else if ( SparkJsonStringEquals(document,element,"tree") )
			mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
		else
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( count == 1u && mask == SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING )
	{
	}
	else if ( count == 1u && mask == SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL )
	{
	}
	else if ( count == 2u && mask == (SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING |
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL) )
	{
	}
	else if ( count == 1u && mask == SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE )
	{
	}
	else if ( count == 2u && mask == (SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE |
		SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING) )
	{
	}
	else
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	topology->algorithm_mask = mask;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingLoadTpStepRails(
	const SparkJsonDocument *document,
	int32_t object,
	uint32_t tp_degree,
	SparkTpDeviceCollectiveTopology *topology)
{
	int32_t element,token;
	uint32_t count,index,value;
	SparkStatus status;
	token = SparkLagunaServingJsonMember(document,object,"step_rail_indices");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != SPARK_TP_DEVICE_COLLECTIVE_SPLIT_RING_ROUTE_COUNT &&
		count != tp_degree )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonGetUInt32(document,element,&value);
		if ( status != SPARK_STATUS_OK || value >=
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		topology->step_rail_indices[index] = value;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingLoadTpRailHosts(
	const SparkJsonDocument *document,
	int32_t object,
	SparkTpDeviceCollectiveTopology *topology,
	uint32_t tp_degree)
{
	int32_t element,host_element,token;
	uint32_t host_count,index,rail;
	char *host;
	SparkStatus status;
	token = SparkLagunaServingJsonMember(document,object,"rail_peer_hosts");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) ||
		SparkJsonGetArrayElementCount(document,token) !=
			SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	topology->rail_count = SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT;
	for (rail=0u; rail<topology->rail_count; rail++)
	{
		element = SparkJsonGetArrayElement(document,token,rail);
		if ( element < 0 ||
			!SparkJsonTokenIsType(document,element,SPARK_JSON_TOKEN_ARRAY) )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		host_count = SparkJsonGetArrayElementCount(document,element);
		if ( host_count != tp_degree )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
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

static SparkStatus SparkLagunaServingLoadSessionPorts(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint16_t table[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE]
		[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE],
	uint32_t tp_degree)
{
	int32_t token,element,cell;
	uint32_t row,column,port,count;
	SparkStatus status;
	token = SparkLagunaServingJsonMember(document,object,name);
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != tp_degree )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (row=0u; row<count; row++)
	{
		element = SparkJsonGetArrayElement(document,token,row);
		if ( element < 0 ||
			!SparkJsonTokenIsType(document,element,SPARK_JSON_TOKEN_ARRAY) ||
			SparkJsonGetArrayElementCount(document,element) != tp_degree )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for (column=0u; column<count; column++)
		{
			cell = SparkJsonGetArrayElement(document,element,column);
			status = cell < 0 ? SPARK_STATUS_SCHEMA_ERROR :
				SparkJsonGetUInt32(document,cell,&port);
			if ( status != SPARK_STATUS_OK || port > UINT16_MAX ||
				(row == column ? port != 0u : port == 0u) )
				return(status == SPARK_STATUS_OK ?
					SPARK_STATUS_SCHEMA_ERROR : status);
			table[row][column] = (uint16_t)port;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingValidateTpCollectiveMembers(
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
		"step_rail_indices","session_ports","session_ports_hc"
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

static SparkStatus SparkLagunaServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkLagunaServingState *state,
	uint32_t tp_degree)
{
	int32_t object,token,element;
	uint32_t count,index,port;
	uint64_t collective_identifier;
	char *host,*relative_backend_path;
	SparkStatus status;
	if ( document == 0 || runtime_root == 0 || state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&state->tp_collective_topology,0,
		sizeof(state->tp_collective_topology));
	state->tp_collective_topology.abi_version =
		SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	state->tp_collective_topology.descriptor_bytes =
		SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	object = SparkLagunaServingJsonMember(document,root,"tp_collective");
	if ( object < 0 || !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkLagunaServingJsonMember(document,object,"backend");
	if ( token < 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkJsonStringEquals(document,token,"nccl") )
		state->tp_collective_backend_kind =
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	else if ( SparkJsonStringEquals(document,token,"hidden_transport") )
		state->tp_collective_backend_kind =
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	else
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkLagunaServingValidateTpCollectiveMembers(document,object,
		state->tp_collective_backend_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	relative_backend_path = 0;
	token = SparkLagunaServingJsonMember(document,object,"backend_module_path");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonCopyString(document,token,&relative_backend_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_backend_path,
			state->tp_collective_backend_path,
			sizeof(state->tp_collective_backend_path));
	free(relative_backend_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	token = SparkLagunaServingJsonMember(document,object,"collective_identifier");
	status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt64(document,token,&collective_identifier);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_collective_identifier = collective_identifier;
	status = SparkLagunaServingJsonUnsigned(document,object,"listen_port",&port);
	if ( status != SPARK_STATUS_OK || port == 0u || port > UINT16_MAX )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	state->tp_listen_port = (uint16_t)port;
	status = SparkLagunaServingJsonUnsigned(document,object,"connect_timeout_milli",&state->tp_connect_timeout_milli);
	if ( status != SPARK_STATUS_OK || state->tp_connect_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	status = SparkLagunaServingJsonUnsigned(document,object,"operation_timeout_milli",&state->tp_operation_timeout_milli);
	if ( status != SPARK_STATUS_OK || state->tp_operation_timeout_milli == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
	token = SparkLagunaServingJsonMember(document,object,"peer_hosts");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != tp_degree )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
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
	token = SparkLagunaServingJsonMember(document,object,"peer_ports");
	if ( token < 0 || !SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != tp_degree )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
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
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( state->tp_collective_backend_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		status = SparkLagunaServingLoadTpAlgorithms(document,object,
			&state->tp_collective_topology);
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaServingLoadSessionPorts(document,object,
				"session_ports",state->tp_collective_topology.session_ports,
				tp_degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaServingJsonUnsigned(document,object,
				"direct_all_to_all_max_payload_bytes",
				&state->tp_collective_topology.direct_all_to_all_max_payload_bytes);
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaServingJsonUnsigned(document,object,
				"split_ring_min_payload_bytes",
				&state->tp_collective_topology.split_ring_min_payload_bytes);
		if ( status == SPARK_STATUS_OK &&
			(state->tp_collective_topology.algorithm_mask &
				SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL) == 0u &&
			state->tp_collective_topology.direct_all_to_all_max_payload_bytes != 0u )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK &&
			(state->tp_collective_topology.algorithm_mask &
				SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_COUNTER_ROTATING_SPLIT_RING) == 0u &&
			state->tp_collective_topology.split_ring_min_payload_bytes != 0u )
			status = SPARK_STATUS_SCHEMA_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaServingLoadTpRailHosts(document,object,
				&state->tp_collective_topology,tp_degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaServingLoadTpStepRails(document,object,
				tp_degree,&state->tp_collective_topology);
	}
	(void)fprintf(stderr,"LAGUNA-ADAPTER LoadTpCollective rc=%d backend=%u\n",(int)status,state->tp_collective_backend_kind);
	return(status);
}

static SparkStatus SparkLagunaServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkLagunaServingState *state,
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
		status = SparkJsonValidateObjectMembersExact(&document,root,SparkLagunaServingConfigurationMembers,(uint32_t)(sizeof(SparkLagunaServingConfigurationMembers) / sizeof(SparkLagunaServingConfigurationMembers[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_LAGUNA_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkLagunaServingJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,LAGUNA_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkLagunaServingJsonMember(&document,root,"expert_weight_codec") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,LAGUNA_EXPERT_CODEC_NAME)) )
		status = SPARK_STATUS_TARGET_MISMATCH;
	token = status == SPARK_STATUS_OK ? SparkLagunaServingJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingJsonUnsigned(&document,root,"execution_row_capacity",execution_row_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingJsonUnsigned(&document,root,"decode_split_context_threshold",decode_split_context_threshold);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingJsonUnsigned(&document,root,"tp_degree",tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingJsonUnsigned(&document,root,"tp_rank",tp_rank);
	if ( status == SPARK_STATUS_OK && (*tp_degree == 0u || *tp_rank >= *tp_degree) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingLoadTpCollective(&document,root,runtime_root,state,*tp_degree);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	(void)fprintf(stderr,"LAGUNA-ADAPTER LoadConfiguration rc=%d\n",(int)status);
	return(status);
}

static SparkStatus SparkLagunaServingValidateRowOrder(
	const SparkLagunaServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t last_position[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->node_context.max_sequence_positions )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == submission->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkLagunaServingPending *SparkLagunaServingReservePending(
	SparkLagunaServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkLagunaServingPending *pending;
	uint32_t index,lane,row,expected;
	if ( atomic_load_explicit(&state->quiescing,memory_order_acquire) != 0u )
		return(0);
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		pending = &state->pending[index];
		expected = 0u;
		if ( atomic_compare_exchange_strong_explicit(&pending->active,&expected,1u,memory_order_acquire,memory_order_relaxed) != 0 )
		{
			if ( atomic_load_explicit(&state->quiescing,memory_order_acquire) != 0u )
			{
				atomic_store_explicit(&pending->active,0u,memory_order_release);
				return(0);
			}
			pending->owner = state;
			pending->row_count = submission->row_count;
			pending->lane_count = submission->lane_count;
			pending->active_sequence_count = submission->active_sequence_count;
			pending->work_kind = submission->work_kind;
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
				pending->input_token_ids[row] = submission->token_ids[row];
				pending->row_positions[row] = submission->row_positions[row];
				pending->row_sequence_ids[row] = submission->row_sequence_ids[row];
			}
			return(pending);
		}
	}
	return(0);
}

static void SparkLagunaServingOrphanDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkLagunaServingState *state;
	(void)driver_completion;
	state = (SparkLagunaServingState *)completion_context;
	if ( state != 0 )
		atomic_fetch_add_explicit(&state->orphan_completion_count,1u,memory_order_relaxed);
}

static void SparkLagunaServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkLagunaServingPending *pending;
	SparkLagunaServingState *state;
	SparkModelServingCompletion completion;
	uint32_t index,matches;
	pending = (SparkLagunaServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || atomic_load_explicit(&pending->active,memory_order_acquire) == 0u || driver_completion == 0 )
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
		atomic_fetch_add_explicit(&state->orphan_completion_count,1u,memory_order_relaxed);
	if ( completion.status != SPARK_STATUS_OK )
	{
		completion.accepted_token_count = 0u;
		completion.completion_flags = 0u;
	}
	if ( completion.status == SPARK_STATUS_OK && pending->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		uint32_t burst = driver_completion->tokens_per_sequence != 0u ?
			driver_completion->tokens_per_sequence : 1u;
		if ( burst > 1u )
		{
			completion.status = SPARK_STATUS_SCHEMA_ERROR;
			completion.accepted_token_count = 0u;
			completion.completion_flags = 0u;
			atomic_store_explicit(&pending->active,0u,memory_order_release);
			state->completion_function(state->completion_context,&completion);
			return;
		}
		completion.tokens_per_sequence = burst;
		completion.token_count = pending->active_sequence_count * burst;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		if ( burst == 1u )
			for (index=0u; index<completion.token_count; index++)
				completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index]];
		else
			for (index=0u; index<completion.token_count; index++)
				completion.token_ids[index] = pending->output_token_ids[index];
	}
	atomic_store_explicit(&pending->active,0u,memory_order_release);
	state->completion_function(state->completion_context,&completion);
}

static void SparkLagunaServingDriverWake(void *wake_context)
{
	SparkLagunaServingState *state;
	state = (SparkLagunaServingState *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}

static uint32_t SparkLagunaServingAvailableSubmissionCount(
	const SparkLagunaServingState *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += atomic_load_explicit(&state->pending[index].active,memory_order_acquire) == 0u ? 1u : 0u;
	return(available);
}

static void SparkLagunaServingDestroy(void *adapter_state)
{
	SparkLagunaServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SparkLagunaServingState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkLagunaServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
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

static SparkStatus SparkLagunaServingLoadDriver(
	SparkLagunaServingState *state,
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
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_LAGUNA_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,LAGUNA_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_LAGUNA_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_LAGUNA_SERVING_TARGET) != 0 )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	state->program = SparkFindLoadedModelDriverProgram(&state->driver,configuration->driver_program_name);
	if ( state->program == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( state->driver.interface->admit == 0 || state->program->submit == 0 || SparkModelDriverProgramSupportsRuntimeLimits(state->program,SPARK_LAGUNA_SERVING_REQUIRED_PROGRAM_FLAGS,state->pipeline_slot_count,state->max_active_sequence_count,state->max_input_row_count,state->resident_sequence_capacity) == 0u )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
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
	request.completion_function = SparkLagunaServingOrphanDriverCompletion;
	request.completion_context = state;
	request.wake_function = SparkLagunaServingDriverWake;
	request.wake_context = state;
	status = state->driver.interface->create(&request,&state->driver_instance);
	(void)fprintf(stderr,"LAGUNA-ADAPTER LoadDriver rc=%d\n",(int)status);
	return(status == SPARK_STATUS_OK && state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status);
}

static SparkStatus SparkLagunaServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SparkLagunaServingDescriptor,&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration->stage_index >= SPARK_LAGUNA_SERVING_STAGE_COUNT || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_LAGUNA_SERVING_PROGRAM_NAME) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkLagunaServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
	uint32_t decode_split_context_threshold,index;
	SparkStatus status;
	if ( adapter_state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkLagunaServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkLagunaServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	atomic_init(&state->orphan_completion_count,0u);
	atomic_init(&state->quiescing,0u);
	atomic_init(&state->reset_active,0u);
	atomic_init(&state->reset_generation,0u);
	for (index=0u; index<SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; index++)
		atomic_init(&state->pending[index].active,0u);
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
	status = SparkLagunaServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&decode_split_context_threshold,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_LAGUNA_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || decode_split_context_threshold > max_sequence_positions) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_LAGUNA_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_LAGUNA_SERVING_PIPELINE_STAGES;
		state->node_context.stage_index = state->stage_index / SPARK_LAGUNA_SERVING_TP_DEGREE;
		state->node_context.first_layer_index = state->node_context.stage_index * SPARK_LAGUNA_MODEL_STAGE_LAYER_COUNT(SPARK_LAGUNA_SERVING_PIPELINE_STAGES,0u);
		state->node_context.layer_count = SPARK_LAGUNA_MODEL_STAGE_LAYER_COUNT(SPARK_LAGUNA_SERVING_PIPELINE_STAGES,0u);
		state->node_context.expert_weight_codec = LAGUNA_EXPERT_WEIGHT_CODEC;
		state->node_context.resident_sequence_capacity = state->resident_sequence_capacity;
		state->node_context.pipeline_slot_count = state->pipeline_slot_count;
		state->node_context.max_sequence_positions = max_sequence_positions;
		state->node_context.execution_row_capacity = execution_row_capacity;
		state->node_context.decode_split_context_threshold = decode_split_context_threshold;
		state->node_context.tp_degree = tp_degree;
		state->node_context.tp_rank = tp_rank;
		state->node_context.flags = 0u;
		state->node_context.stage_pack_path = state->stage_pack_path;
		state->node_context.model_revision = LAGUNA_MODEL_REVISION;
		state->node_context.tp_collective_backend_kind = state->tp_collective_backend_kind;
		state->node_context.tp_collective_identifier = state->tp_collective_identifier;
		state->node_context.tp_connect_timeout_milli = state->tp_connect_timeout_milli;
		state->node_context.tp_operation_timeout_milli = state->tp_operation_timeout_milli;
		state->node_context.tp_collective_control_port_base = state->tp_collective_control_port_base;
		state->node_context.tp_collective_topology = state->tp_collective_topology;
		state->node_context.tp_collective_backend_module_path = state->tp_collective_backend_path;
		state->node_context.kv_backing_directory = configuration->kv_backing_directory;
		state->node_context.kv_backing_maximum_bytes = configuration->kv_backing_maximum_bytes;
		status = SparkLagunaServingLoadDriver(state,configuration);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkLagunaServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingValidateBoundaries(
	const SparkLagunaServingState *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	if ( submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)boundary_bytes;
	(void)state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkLagunaServingState *state;
	SparkStatus status;
	state = (SparkLagunaServingState *)adapter_state;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	if ( submission != 0 && submission->control_generation < atomic_load_explicit(&state->reset_generation,memory_order_acquire) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkLagunaServingDescriptor,&state->runtime_limits,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingValidateBoundaries(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaServingValidateRowOrder(state,submission);
	if ( status == SPARK_STATUS_OK && submission->model_extension_bytes != 0u )
		status = SPARK_STATUS_UNSUPPORTED;
	return(status);
}

static SparkServingCacheAdmission SparkLagunaServingCacheContext(SparkLagunaServingState *state,SparkModelDriverCacheLane *lanes)
{
	SparkServingCacheAdmission cache;
	cache.program_id = state->program->program_id;
	cache.lane_capacity = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	cache.lanes = lanes;
	cache.driver = state->driver.interface;
	cache.driver_instance = state->driver_instance;
	cache.validate = SparkLagunaServingValidateSubmission;
	cache.adapter_state = state;
	return(cache);
}

static SparkStatus SparkLagunaServingPrefetch(void *adapter_state,const SparkModelServingSubmission *submissions,uint32_t count)
{
	SparkLagunaServingState *state;
	SparkServingCacheAdmission cache;
	state = (SparkLagunaServingState *)adapter_state;
	if ( state == 0 || state->program == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	cache = SparkLagunaServingCacheContext(state,SparkLagunaServingCacheScratch);
	return(SparkServingCacheAdmissionRun(&cache,submissions,count,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE));
}

static SparkStatus SparkLagunaServingResolvePrefetch(void *adapter_state,const SparkModelServingSubmission *submission,uint32_t resolution)
{
	SparkLagunaServingState *state;
	SparkServingCacheAdmission cache;
	uint32_t flags;
	state = (SparkLagunaServingState *)adapter_state;
	if ( state == 0 || state->program == 0 || (resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT && resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	flags = resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	cache = SparkLagunaServingCacheContext(state,SparkLagunaServingCacheScratch);
	return(SparkServingCacheAdmissionRun(&cache,submission,1u,flags));
}

static void SparkLagunaServingBuildFrame(
	const SparkLagunaServingState *state,
	const SparkModelServingSubmission *submission,
	SparkLagunaServingPending *pending)
{
	SparkLagunaResidentDecodeStageBatchView *batch = &pending->batch;
	SparkLagunaResidentDecodeStageFrameContext *context = &pending->context;
	SparkModelDriverBuffer *buffer = &pending->buffer;
	SparkModelDriverFrame *frame = &pending->frame;
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = submission->row_count;
	batch->active_sequence_count = submission->active_sequence_count;
	batch->token_ids = pending->input_token_ids;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = pending->row_positions;
	batch->row_sequence_ids = pending->row_sequence_ids;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = sizeof(*context);
	context->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
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
	frame->completion_function = SparkLagunaServingDriverCompletion;
	frame->completion_context = pending;
}

static SparkStatus SparkLagunaServingAdmit(
	SparkLagunaServingState *state,
	const SparkModelServingSubmission *submission,
	SparkLagunaServingPending *pending,
	SparkModelDriverFrame *frame)
{
	SparkServingCacheAdmission cache;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	cache = SparkLagunaServingCacheContext(state,pending->cache_lanes);
	status = SparkServingCacheBuildRequest(&cache,submission,0u,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	frame->cache_lanes = pending->cache_lanes;
	frame->cache_lane_count = request.cache_lane_count;
	return(SparkAdmissionEvaluateAndApply(state->driver.interface,state->driver_instance,&request,frame,&decision));
}

static SparkStatus SparkLagunaServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkLagunaServingState *state;
	SparkLagunaServingPending *pending;
	SparkModelDriverCompletion released = {0};
	SparkStatus status;
	state = (SparkLagunaServingState *)adapter_state;
	status = SparkLagunaServingValidateSubmission(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SparkLagunaServingReservePending(state,submission);
	if ( pending == 0 )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	SparkLagunaServingBuildFrame(state,submission,pending);
	status = SparkLagunaServingAdmit(state,submission,pending,&pending->frame);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"G5N-DBG submit: admit -> %d\n",(int)status);
	if ( status == SPARK_STATUS_OK )
	{
		if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		{
			released.request_id = pending->request_id;
			released.sequence_id = pending->sequence_id;
			released.sequence_position = pending->sequence_position;
			released.program_id = state->program->program_id;
			released.residency = submission->residency;
			SparkLagunaServingDriverCompletion(pending,&released);
		}
		else
			status = state->program->submit(state->driver_instance,&pending->frame);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr,"G5N-DBG submit: program->submit -> %d\n",(int)status);
	}
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&pending->active,0u,memory_order_release);
	return(status);
}

static SparkStatus SparkLagunaServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	(void)maximum_step_count;
	return(adapter_state != 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkLagunaServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkLagunaServingState *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SparkLagunaServingState *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SparkLagunaServingAvailableSubmissionCount(state) != state->pipeline_slot_count )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}

static SparkStatus SparkLagunaServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkLagunaServingState *state;
	SparkModelDriverRuntimeSnapshot driver_snapshot;
	uint32_t available;
	SparkStatus status;
	state = (SparkLagunaServingState *)adapter_state;
	if ( state == 0 || snapshot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&driver_snapshot,0,sizeof(driver_snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&driver_snapshot);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	available = SparkLagunaServingAvailableSubmissionCount(state);
	if ( available > driver_snapshot.available_dispatch_slot_count )
		available = driver_snapshot.available_dispatch_slot_count;
	snapshot->available_submission_count = state->quiescing == 0u ? available : 0u;
	snapshot->active_submission_count = state->pipeline_slot_count - SparkLagunaServingAvailableSubmissionCount(state);
	snapshot->submitted_count = driver_snapshot.submitted_count;
	snapshot->completed_count = driver_snapshot.completed_count;
	snapshot->rejected_count = driver_snapshot.rejected_count + atomic_load_explicit(&state->orphan_completion_count,memory_order_relaxed);
	snapshot->resident_sequence_count = driver_snapshot.resident_sequence_count;
	snapshot->resident_token_count = driver_snapshot.resident_token_count;
	snapshot->kv_token_capacity = driver_snapshot.kv_token_capacity;
	snapshot->device_memcpy_bytes_per_submit = driver_snapshot.device_memcpy_bytes_per_submit;
	snapshot->host_staging_bytes_per_submit = driver_snapshot.host_staging_bytes_per_submit;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaServingResetControl(void *adapter_state,uint64_t control_generation)
{
	SparkLagunaServingState *state = (SparkLagunaServingState *)adapter_state;
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	if ( state == 0 || control_generation == 0u || control_generation <= atomic_load_explicit(&state->reset_generation,memory_order_acquire) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkLagunaServingQuiesce(state,UINT64_MAX);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.descriptor_bytes = sizeof(request);
	request.program_id = state->program->program_id;
	request.control_generation = control_generation;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	SparkModelDriverInitializeAdmissionDecision(&decision);
	status = state->driver.interface->admit(state->driver_instance,&request,&decision);
	if ( status == SPARK_STATUS_OK && decision.accepted == 0u )
		status = SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
	{
		atomic_store_explicit(&state->reset_generation,control_generation,memory_order_release);
		atomic_store_explicit(&state->quiescing,0u,memory_order_release);
	}
	return(status);
}

static SparkStatus SparkLagunaServingReset(void *adapter_state,uint64_t control_generation)
{
	SparkLagunaServingState *state = (SparkLagunaServingState *)adapter_state;
	uint32_t expected = 0u;
	SparkStatus status;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( atomic_compare_exchange_strong_explicit(&state->reset_active,&expected,1u,memory_order_acquire,memory_order_relaxed) == 0 )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	status = SparkLagunaServingResetControl(state,control_generation);
	atomic_store_explicit(&state->reset_active,0u,memory_order_release);
	return(status);
}

static const SparkModelServingAdapterInterface SparkLagunaServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkLagunaServingDescriptor,
	.initialize = SparkLagunaServingInitialize,
	.destroy = SparkLagunaServingDestroy,
	.validate_submission = SparkLagunaServingValidateSubmission,
	.submit = SparkLagunaServingSubmit,
	.prefetch = SparkLagunaServingPrefetch,
	.resolve_prefetch = SparkLagunaServingResolvePrefetch,
	.progress = SparkLagunaServingProgress,
	.quiesce = SparkLagunaServingQuiesce,
	.snapshot = SparkLagunaServingSnapshot,
	.reset = SparkLagunaServingReset
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkLagunaServingInterface);
}
