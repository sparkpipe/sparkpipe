#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_parallel_shape.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_runner.h"
#include "sparkpipe/spark_dsv4_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "runtime/adapter_common.h"

#if SPARK_DSV4_SERVING_TOPOLOGY == 404
#if defined(SPARK_DSV4_PRO_BUILD)
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.pro.serving-adapter.tp4-pp4.v1"
#else
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.flash-0731.serving-adapter.tp4-pp4.v1"
#endif
#define SPARK_DSV4_SERVING_STAGE_COUNT 16u
#define SPARK_DSV4_SERVING_TP_DEGREE 4u
#define SPARK_DSV4_SERVING_PP_STAGE_COUNT 4u
#define SPARK_DSV4_SERVING_HYBRID 1u
#define SPARK_DSV4_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
#define SPARK_DSV4_SERVING_EXTRA_CAPABILITY \
	(SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HYBRID_TP_PP | \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT)
#define SPARK_DSV4_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#define SPARK_DSV4_SERVING_STAGE_LAYERS \
	{16u,16u,16u,16u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u}
#define SPARK_DSV4_SERVING_PIPELINE_SLOT_COUNT_MAX 4u
#elif SPARK_DSV4_SERVING_TOPOLOGY == 16
#if defined(SPARK_DSV4_PRO_BUILD)
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.pro.serving-adapter.tp16.v1"
#else
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.flash-0731.serving-adapter.tp16.v1"
#endif
#define SPARK_DSV4_SERVING_STAGE_COUNT 16u
#define SPARK_DSV4_SERVING_TP_DEGREE 16u
#define SPARK_DSV4_SERVING_PP_STAGE_COUNT 1u
#define SPARK_DSV4_SERVING_HYBRID 0u
#define SPARK_DSV4_SERVING_EXTRA_CAPABILITY 0u
#define SPARK_DSV4_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
#define SPARK_DSV4_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#define SPARK_DSV4_SERVING_STAGE_LAYERS \
	{SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT}
#define SPARK_DSV4_SERVING_PIPELINE_SLOT_COUNT_MAX 16u
#elif SPARK_DSV4_SERVING_TOPOLOGY == 4
#if defined(SPARK_DSV4_PRO_BUILD)
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.pro.serving-adapter.tp4.v1"
#else
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.flash-0731.serving-adapter.tp4.v1"
#endif
#define SPARK_DSV4_SERVING_STAGE_COUNT 4u
#define SPARK_DSV4_SERVING_TP_DEGREE 4u
#define SPARK_DSV4_SERVING_PP_STAGE_COUNT 1u
#define SPARK_DSV4_SERVING_HYBRID 0u
#define SPARK_DSV4_SERVING_EXTRA_CAPABILITY 0u
#define SPARK_DSV4_SERVING_TOPOLOGY_FLAG \
	SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT
#define SPARK_DSV4_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#define SPARK_DSV4_SERVING_STAGE_LAYERS \
	{SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT, \
	 SPARK_DSV4_MODEL_LAYER_COUNT,SPARK_DSV4_MODEL_LAYER_COUNT}
#define SPARK_DSV4_SERVING_PIPELINE_SLOT_COUNT_MAX 4u
#elif SPARK_DSV4_SERVING_TOPOLOGY == 13
#define SPARK_DSV4_SERVING_ADAPTER_ID \
	"spark.dsv4.flash-0731.serving-adapter.pp13.v2"
#define SPARK_DSV4_SERVING_STAGE_COUNT 13u
#define SPARK_DSV4_SERVING_TP_DEGREE 1u
#define SPARK_DSV4_SERVING_PP_STAGE_COUNT 13u
#define SPARK_DSV4_SERVING_HYBRID 0u
#define SPARK_DSV4_SERVING_EXTRA_CAPABILITY 0u
#define SPARK_DSV4_SERVING_TOPOLOGY_FLAG 0u
#define SPARK_DSV4_SERVING_REQUIRED_PROGRAM_FLAGS \
	(SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
	 SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)
#define SPARK_DSV4_SERVING_STAGE_LAYERS \
	{3u,3u,3u,3u,3u,3u,3u,4u,4u,4u,4u,4u,2u}
#define SPARK_DSV4_SERVING_PP13_PIPELINE_SLOT_COUNT_MAX \
	SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT
#define SPARK_DSV4_SERVING_PIPELINE_SLOT_COUNT_MAX \
	SPARK_DSV4_SERVING_PP13_PIPELINE_SLOT_COUNT_MAX
#else
#error "unsupported SPARK_DSV4_SERVING_TOPOLOGY"
#endif
#define SPARK_DSV4_SERVING_MODEL_ID SPARK_DSV4_MODEL_ID
#define SPARK_DSV4_SERVING_MODEL_REVISION SPARK_DSV4_MODEL_SOURCE_REVISION
#if SPARK_BATCH_BUCKET == 1u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B1
#elif SPARK_BATCH_BUCKET == 6u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B6
#elif SPARK_BATCH_BUCKET == 8u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B8
#elif SPARK_BATCH_BUCKET == 9u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B9
#elif SPARK_BATCH_BUCKET == 16u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B16
#elif SPARK_BATCH_BUCKET == 11u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B11
#elif SPARK_BATCH_BUCKET == 32u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B32
#elif SPARK_BATCH_BUCKET == 64u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B64
#elif SPARK_BATCH_BUCKET == 1024u
#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 \
	SPARK_DSV4_MODEL_DESCRIPTION_SHA256
#else
#error "serving adapter requires a generated batch-specific model description"
#endif
#define SPARK_DSV4_SERVING_DRIVER_MODEL_ID SPARK_DSV4_MODEL_DRIVER_MODEL_ID
#define SPARK_DSV4_SERVING_DRIVER_MODEL_REVISION SPARK_DSV4_MODEL_DRIVER_REVISION
#define SPARK_DSV4_SERVING_DRIVER_STAGE_NAME "dsv4_resident_decode_stage"
#define SPARK_DSV4_SERVING_PROGRAM_NAME "resident_decode"
#define SPARK_DSV4_SERVING_CHAIN_CAPABILITY \
	(SPARK_DSV4_SERVING_PP_STAGE_COUNT == 1u && \
	 SPARK_DSV4_SERVING_TP_DEGREE > 1u ? \
	 SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN : 0u)
#define SPARK_DSV4_SERVING_CHAIN_DEPTH \
	(SPARK_DSV4_SERVING_PP_STAGE_COUNT == 1u && \
	 SPARK_DSV4_SERVING_TP_DEGREE > 1u ? \
	 SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE : 1u)
#define SPARK_DSV4_SERVING_OUTPUT_TOKEN_CAPACITY \
	(SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT <= \
	 SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT / \
	 SPARK_DSV4_SERVING_CHAIN_DEPTH ? \
	 SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * \
	 SPARK_DSV4_SERVING_CHAIN_DEPTH : \
	 SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT)

/* Optional members stay last so the exact-member check can clip them. */
static const char *const SparkDsv4ServingConfigurationMembersBase[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions"
};

static const char *const SparkDsv4ServingConfigurationMembersPp[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"cuda_graph_count"
};

static const char *const SparkDsv4ServingConfigurationMembersTp[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"cuda_graph_count_by_pp_stage",
	"tp_collective"
};

typedef struct SparkDsv4ServingPending
{
	/* Shape + nine-field submission identity echo: captured once at claim
	 * by SparkAdapterPendingCapture (adapter_common.h). */
	SparkAdapterPendingCore core;
	struct SparkDsv4ServingAdapterState *owner;
	uint32_t emit_count;
	uint32_t cache_lane_count;
	uint32_t tokens_per_sequence;
	uint32_t last_row_by_lane[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t emit_row_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t emit_lane_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_row_lane_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t output_token_ids[SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT];
	SparkModelDriverCacheLane cache_lanes[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkDsv4ServingPending;

typedef struct SparkDsv4ServingAdapterState
{
	/* Counters, routes, the loaded-driver handle, residency inputs, the
	 * pending table pointer, and the quiescing latch: the prologue every
	 * family duplicated now lives in SparkAdapterCommonState.
	 * (adapter_common.h). */
	SparkAdapterCommonState common;
	SparkDsv4ResidentDecodeStageNodeContext node_context;
	SparkDsv4StageRunner runner;
	char stage_pack_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t stage_index;
	uint16_t tp_listen_port;
	uint16_t tp_peer_ports[SPARK_DSV4_RESIDENT_DECODE_STAGE_TP_PEER_COUNT];
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t tp_collective_backend_kind;
	uint64_t tp_collective_identifier;
	SparkTpDeviceCollectiveTopology tp_collective_topology;
	char tp_collective_backend_path[SPARK_INTERNAL_PATH_BYTES];
	uint32_t tp_collective_control_port_base;
	SparkDsv4ServingPending pending[SPARK_DSV4_SERVING_PIPELINE_SLOT_COUNT_MAX];
} SparkDsv4ServingAdapterState;

static uint32_t SparkDsv4ServingPpStageIndex(uint32_t world_rank)
{
	return(SPARK_DSV4_SERVING_HYBRID != 0u ?
		world_rank / SPARK_DSV4_SERVING_TP_DEGREE : world_rank);
}

static uint32_t SparkDsv4ServingTpRank(uint32_t world_rank)
{
	return(SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ?
		world_rank % SPARK_DSV4_SERVING_TP_DEGREE : 0u);
}

static _Thread_local SparkModelDriverCacheLane SparkDsv4ServingPrefetchLanes[
	SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

static const SparkModelServingAdapterDescriptor SparkDsv4ServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RELEASE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION | SPARK_DSV4_SERVING_TOPOLOGY_FLAG | SPARK_DSV4_SERVING_EXTRA_CAPABILITY | (SPARK_DSV4_SERVING_TOPOLOGY_FLAG == 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT : 0u) | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE | SPARK_DSV4_SERVING_CHAIN_CAPABILITY,
	.stage_count = SPARK_DSV4_SERVING_STAGE_COUNT,
	.layer_count = SPARK_DSV4_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,
	.boundary_element_bytes = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC,
	.expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC,
	.kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC,
	.max_inflight_submission_count = SPARK_DSV4_SERVING_PIPELINE_SLOT_COUNT_MAX,
	.max_active_sequence_count = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequence_count = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_DSV4_SERVING_OUTPUT_TOKEN_CAPACITY,
	.max_speculative_token_count = SPARK_DSV4_MODEL_DSPARK_SPEC_STEP,
	.resident_sequence_slot_reuse = SPARK_MODEL_SERVING_SLOT_REUSE_REQUIRES_RELEASE,
	.adapter_id = SPARK_DSV4_SERVING_ADAPTER_ID,
	.model_id = SPARK_DSV4_SERVING_MODEL_ID,
	.model_revision = SPARK_DSV4_SERVING_MODEL_REVISION,
	.driver_program_name = SPARK_DSV4_SERVING_PROGRAM_NAME,
	.artifact_sha256 = SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256,
	.stage_layer_counts = SPARK_DSV4_SERVING_STAGE_LAYERS,
	.minimum_efficient_submission_row_count = SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? 1u : 16u,
	.cache_block_token_count =
		SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS,
	.parallel_group_size = SPARK_DSV4_SERVING_HYBRID != 0u ?
		SPARK_DSV4_SERVING_TP_DEGREE : 0u
};

/* tp_collective stanza policy: all three known algorithms required;
 * adaptive thresholds nonzero AND strictly ordered (d2a < split_ring).
 * Port-span validation lives in the shared parser. */
static const SparkAdapterTpCollectivePolicy SparkDsv4TpCollectivePolicy = {
	SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS, 3u, 0u
};

static SparkStatus SparkDsv4ServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkDsv4ServingAdapterState *state)
{
	SparkAdapterTpCollectiveParsed tp;
	SparkStatus status;
	status = SparkAdapterLoadTpCollective(document,root,runtime_root,
		SPARK_DSV4_SERVING_STAGE_COUNT,1u,
		&SparkDsv4TpCollectivePolicy,&tp);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_listen_port = tp.listen_port;
	state->tp_connect_timeout_milli = tp.connect_timeout_milli;
	state->tp_operation_timeout_milli = tp.operation_timeout_milli;
	state->tp_collective_backend_kind = tp.backend_kind;
	state->tp_collective_identifier = tp.collective_identifier;
	memcpy(state->tp_peer_ports,tp.peer_ports,
		sizeof(tp.peer_ports[0]) * SPARK_DSV4_SERVING_STAGE_COUNT);
	state->tp_collective_control_port_base = tp.control_port_base;
	memcpy(state->tp_collective_backend_path,tp.backend_module_path,
		SPARK_INTERNAL_PATH_BYTES);
	state->tp_collective_topology = tp.topology;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ServingLoadTpGraphCounts(
	const SparkJsonDocument *document,
	int32_t root,
	const SparkDsv4ServingAdapterState *state,
	uint32_t *cuda_graph_count)
{
	int32_t element,token;
	uint32_t count,descriptor_index,expected,index,pp_stage_index,value;
	SparkStatus status;
	if ( document == 0 || state == 0 || cuda_graph_count == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	token = SparkJsonFindObjectMember(document,root,
		"cuda_graph_count_by_pp_stage");
	if ( token < 0 ||
		!SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = SparkJsonGetArrayElementCount(document,token);
	if ( count != SPARK_DSV4_SERVING_PP_STAGE_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	*cuda_graph_count = 0u;
	pp_stage_index = SPARK_DSV4_SERVING_HYBRID != 0u ?
		SparkDsv4ServingPpStageIndex(state->stage_index) : 0u;
	for (index=0u; index<count; index++)
	{
		element = SparkJsonGetArrayElement(document,token,index);
		status = element < 0 ? SPARK_STATUS_SCHEMA_ERROR :
			SparkJsonGetUInt32(document,element,&value);
		descriptor_index = SPARK_DSV4_SERVING_HYBRID != 0u ?
			index * SPARK_DSV4_SERVING_TP_DEGREE : 0u;
		expected = SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(
			SparkDsv4ServingDescriptor.stage_layer_counts[descriptor_index]);
		if ( status != SPARK_STATUS_OK || expected == 0u || value != expected )
			return(status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status);
		if ( index == pp_stage_index )
			*cuda_graph_count = value;
	}
	return(*cuda_graph_count != 0u ? SPARK_STATUS_OK :
		SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkDsv4ServingLoadConfiguration(
	const char *path,
	const char *runtime_root,
	SparkDsv4ServingAdapterState *state,
	uint32_t *max_sequence_positions,
	uint32_t *cuda_graph_count)
{
	SparkJsonDocument document;
	int32_t root,token;
	const char *const *members;
	uint32_t schema_version,member_count,has_cuda_graphs,has_tp_collective;
	char *relative_stage_pack_path;
	SparkStatus status;
	relative_stage_pack_path = 0;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	has_cuda_graphs = SparkJsonFindObjectMember(&document,root,
		"cuda_graph_count") >= 0 ? 1u : 0u;
	has_tp_collective = SparkJsonFindObjectMember(&document,root,"tp_collective") >= 0 ? 1u : 0u;
	if ( SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u )
	{
		members = SparkDsv4ServingConfigurationMembersTp;
		member_count = (uint32_t)(sizeof(SparkDsv4ServingConfigurationMembersTp) /
			sizeof(SparkDsv4ServingConfigurationMembersTp[0]));
		if ( has_tp_collective == 0u )
			status = SPARK_STATUS_SCHEMA_ERROR;
	}
	else
	{
		members = has_cuda_graphs != 0u ? SparkDsv4ServingConfigurationMembersPp : SparkDsv4ServingConfigurationMembersBase;
		member_count = has_cuda_graphs != 0u ? (uint32_t)(sizeof(SparkDsv4ServingConfigurationMembersPp) / sizeof(SparkDsv4ServingConfigurationMembersPp[0])) : (uint32_t)(sizeof(SparkDsv4ServingConfigurationMembersBase) / sizeof(SparkDsv4ServingConfigurationMembersBase[0]));
		if ( has_tp_collective != 0u )
			status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(&document,root,members,member_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_DSV4_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,SPARK_DSV4_SERVING_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkJsonFindObjectMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonGetUInt32Member(&document,root,"max_sequence_positions",max_sequence_positions);
	*cuda_graph_count = 0u;
	if ( status == SPARK_STATUS_OK && SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u )
		status = SparkDsv4ServingLoadTpGraphCounts(&document,root,state,
			cuda_graph_count);
	else
	{
		token = status == SPARK_STATUS_OK ?
			SparkJsonFindObjectMember(&document,root,"cuda_graph_count") : -1;
		if ( status == SPARK_STATUS_OK && token >= 0 )
			status = SparkJsonGetUInt32(&document,token,cuda_graph_count);
		if ( status == SPARK_STATUS_OK && *cuda_graph_count >
			SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT )
			status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK && SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u )
		status = SparkDsv4ServingLoadTpCollective(&document,root,runtime_root,
			state);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	return(status);
}

static uint32_t SparkDsv4ServingFirstLayer(uint32_t stage_index)
{
	uint32_t index,first_layer;
	first_layer = 0u;
	for (index=0u; index<stage_index; index++)
		first_layer += SparkDsv4ServingDescriptor.stage_layer_counts[index];
	return(first_layer);
}

static SparkStatus SparkDsv4ServingValidateRowOrder(
	const SparkDsv4ServingAdapterState *state,
	const SparkModelServingSubmission *submission)
{
	SparkRowLayoutDenseLaneContext dense;
	uint8_t seen[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t last_position[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t occurrences[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_rows[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane,row;
	for (row=0u; row<submission->row_count; row++)
		if ( submission->row_positions[row] >= state->node_context.max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(SPARK_STATUS_OK);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	memset(seen,0,sizeof(seen));
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( seen[lane] != 0u && (last_position[lane] == UINT64_MAX || submission->row_positions[row] != last_position[lane] + 1u) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
	}
	dense.lane_count = submission->active_sequence_count;
	return(SparkRowLayoutValidateRoundMajor(submission->row_count,submission->active_sequence_count,submission->row_lane_indices,SparkRowLayoutDenseLaneOrdinal,&dense,occurrences,last_rows));
}

static SparkStatus SparkDsv4ServingReservePending(
	SparkDsv4ServingAdapterState *state,
	const SparkModelServingSubmission *submission,
	SparkDsv4ServingPending **pending_out)
{
	SparkDsv4ServingPending *pending;
	int32_t index;
	SparkStatus status;
	if ( pending_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*pending_out = 0;
	index = SparkAdapterPendingClaim(state->common.pending,
		sizeof(SparkDsv4ServingPending),
		state->common.pipeline_slot_count);
	if ( index < 0 )
		return(SPARK_STATUS_BUSY);
	pending = &state->pending[index];
	memset(pending,0,sizeof(*pending));
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission,pending->cache_lanes,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&pending->cache_lane_count);
	if ( status == SPARK_STATUS_OK &&
		submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		status = SparkModelServingAdapterSelectEmitRows(submission,pending->emit_row_indices,pending->emit_lane_indices,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&pending->emit_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending->owner = state;
	SparkAdapterPendingCapture(&pending->core,submission);
	pending->tokens_per_sequence = submission->tokens_per_sequence;
	/* The two row-major captures the old single loop made in one pass:
	 * last row seen per lane (identity echo) and each row's own lane slot. */
	SparkAdapterCaptureLastRowByLane(submission,pending->last_row_by_lane);
	SparkAdapterCaptureResidentSlotsPerRow(submission,pending->resident_row_lane_indices);
	/* Late active-stamp preserved: dsv4 stamps active after the payload. */
	pending->core.active = 1u;
	*pending_out = pending;
	return(SPARK_STATUS_OK);
}

/* Pre-route driver completions bump the shared orphan counter through
 * SparkAdapterOrphanDriverCompletion and wakes forward through
 * SparkAdapterDispatchWake; both are handed to the driver at create time
 * with the common state as their context. */

static void SparkDsv4ServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkDsv4ServingAdapterState *state;
	SparkDsv4ServingPending *pending;
	SparkModelServingCompletion completion;
	uint32_t matches;
	uint32_t index;
	pending = (SparkDsv4ServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || pending->core.active == 0u || driver_completion == 0 )
		return;
	matches = SparkAdapterDriverCompletionMatches(driver_completion,
		pending->core.identity.request_id,pending->core.identity.sequence_id,
		pending->core.identity.sequence_position,state->common.program->program_id);
	SparkAdapterBuildCompletionHeader(&completion,&pending->core);
	completion.status = matches != 0u ? (uint32_t)driver_completion->status : SPARK_STATUS_SCHEMA_ERROR;
	/* Identity echo + ABI stamp come from the shared header builder. */
	if ( matches != 0u )
		completion.residency = driver_completion->residency;
	completion.accepted_token_count = driver_completion->accepted_token_count;
	/* DSpark verify (DECODE) frames emit 1..tokens_per_sequence tokens
	 * depending on acceptance; the module's completion carries the actual
	 * count. PREFILL/RELEASE completions carry tokens_per_sequence == 0u by
	 * contract and must NOT be fenced by this gate (a RELEASE would otherwise
	 * be rejected as SCHEMA_ERROR). */
	if ( matches != 0u &&
		pending->core.work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE &&
		(driver_completion->tokens_per_sequence == 0u ||
		 driver_completion->tokens_per_sequence >
			pending->tokens_per_sequence +
			SparkDsv4ServingDescriptor.max_speculative_token_count) )
		completion.status = SPARK_STATUS_SCHEMA_ERROR;
	completion.queue_delay_ns = driver_completion->queue_delay_ns;
	completion.service_time_ns = driver_completion->service_time_ns;
	completion.device_memcpy_bytes = driver_completion->device_memcpy_bytes;
	completion.host_staging_bytes = driver_completion->host_staging_bytes;
	if ( driver_completion->status != SPARK_STATUS_OK )
		fprintf(stderr,"dsv4_adapter driver_completion status=%s stage=%u submission=%llu accepted=%u\n",SparkStatusToString((SparkStatus)driver_completion->status),state->stage_index,(unsigned long long)pending->core.identity.submission_id,driver_completion->accepted_token_count);
	if ( matches == 0u )
		state->common.orphan_completion_count++;
	if ( state->stage_index + 1u == SPARK_DSV4_SERVING_STAGE_COUNT &&
		pending->core.work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE &&
		completion.status == SPARK_STATUS_OK )
	{
		completion.tokens_per_sequence = driver_completion->tokens_per_sequence;
		completion.token_count = pending->core.active_sequence_count *
			completion.tokens_per_sequence;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		if ( pending->core.work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
			for (index=0u; index<pending->core.active_sequence_count; index++)
				completion.token_ids[index] =
					pending->output_token_ids[index];
		else
			memcpy(completion.token_ids,pending->output_token_ids,
				(uint64_t)completion.token_count * sizeof(uint32_t));
	}
	pending->core.active = 0u;
	state->common.sink.function(state->common.sink.context,&completion);
}


static void SparkDsv4ServingDestroy(void *adapter_state)
{
	SparkDsv4ServingAdapterState *state;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	if ( state == 0 )
		return;
	if ( SparkAdapterDestroyReady(&state->common) == 0u )
		return;
	SparkAdapterTeardownDriver(&state->common);
	free(state);
}

static SparkStatus SparkDsv4ServingLoadDriver(
	SparkDsv4ServingAdapterState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	/* Kind 0 = the runtime-limits capability school; the target compare is
	 * left NULL - this family never pinned it. */
	SparkAdapterDriverContract contract;
	contract.model_id = SPARK_DSV4_SERVING_DRIVER_MODEL_ID;
	contract.model_revision = SPARK_DSV4_SERVING_DRIVER_MODEL_REVISION;
	contract.stage_name = SPARK_DSV4_SERVING_DRIVER_STAGE_NAME;
	contract.target = 0;
	contract.description_sha256 = SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256;
	contract.required_program_flags = SPARK_DSV4_SERVING_REQUIRED_PROGRAM_FLAGS;
	contract.check_kind = 0u;
	contract.node_context = &state->node_context;
	return(SparkAdapterLoadDriver(&state->common,configuration,&contract));
}

static SparkStatus SparkDsv4ServingInitializeRunner(
	SparkDsv4ServingAdapterState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkDsv4StageRunnerConfiguration runner_configuration;
	uint32_t pp_stage_index;
	memset(&runner_configuration,0,sizeof(runner_configuration));
	runner_configuration.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
	runner_configuration.descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES;
	runner_configuration.flags = SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION;
	pp_stage_index = SparkDsv4ServingPpStageIndex(state->stage_index);
	if ( SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u )
		runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_TENSOR_PARALLEL;
	if ( SPARK_DSV4_SERVING_HYBRID != 0u )
	{
		runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_HYBRID_TP_PP;
		runner_configuration.parallel_group_size = SPARK_DSV4_SERVING_TP_DEGREE;
		if ( pp_stage_index != 0u )
			runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY;
		if ( pp_stage_index + 1u < SPARK_DSV4_SERVING_PP_STAGE_COUNT )
			runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY;
		if ( state->stage_index + 1u == SPARK_DSV4_SERVING_STAGE_COUNT )
			runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_FINAL_TP_RANK;
	}
	else if ( SPARK_DSV4_SERVING_TOPOLOGY_FLAG == 0u )
	{
		if ( state->stage_index != 0u )
			runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY;
		if ( state->stage_index + 1u < SPARK_DSV4_SERVING_STAGE_COUNT )
			runner_configuration.flags |= SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY;
	}
	runner_configuration.stage_index = state->stage_index;
	runner_configuration.stage_count = SPARK_DSV4_SERVING_STAGE_COUNT;
	runner_configuration.max_active_sequence_count = state->common.max_active_sequence_count;
	runner_configuration.max_input_row_count = state->common.max_input_row_count;
	runner_configuration.resident_sequence_capacity = state->common.resident_sequence_capacity;
	runner_configuration.driver_interface = state->common.driver.interface;
	runner_configuration.driver_instance = state->common.driver_instance;
	runner_configuration.program = state->common.program;
	runner_configuration.execution_stream = configuration->execution_stream;
	return(SparkDsv4StageRunnerInitialize(&state->runner,&runner_configuration));
}

static SparkStatus SparkDsv4ServingValidateConfiguration(
	const SparkModelServingAdapterConfiguration *configuration)
{
	return(SparkAdapterValidateConfiguration(&SparkDsv4ServingDescriptor,
		configuration,SPARK_DSV4_SERVING_PROGRAM_NAME,
		SPARK_DSV4_SERVING_STAGE_COUNT));
}

static void SparkDsv4ServingInitializeState(
	SparkDsv4ServingAdapterState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	state->stage_index = configuration->stage_index;
	(void)SparkAdapterInitializePrologue(&state->common,configuration);
	/* The pending table is the family's own array; the shared helpers walk
	 * it through the core-stride view. */
	state->common.pending = &state->pending[0].core;
	state->common.pending_element_bytes = sizeof(state->pending[0]);
}

static SparkStatus SparkDsv4ServingInitializeNodeContext(
	SparkDsv4ServingAdapterState *state,
	uint32_t max_sequence_positions,
	uint32_t cuda_graph_count)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig tp_config;
	state->node_context.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	state->node_context.descriptor_bytes = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
	state->node_context.flags = SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_TENSOR_PARALLEL : 0u;
	if ( SPARK_DSV4_SERVING_HYBRID != 0u )
		state->node_context.flags |= SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_PIPELINE_PARALLEL;
	state->node_context.stage_count = SPARK_DSV4_SERVING_HYBRID != 0u ? SPARK_DSV4_SERVING_PP_STAGE_COUNT : SPARK_DSV4_SERVING_STAGE_COUNT;
	state->node_context.stage_index = SPARK_DSV4_SERVING_HYBRID != 0u ? SparkDsv4ServingPpStageIndex(state->stage_index) : state->stage_index;
	state->node_context.first_layer_index = SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? 0u : SparkDsv4ServingFirstLayer(state->stage_index);
	state->node_context.layer_count = SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? SPARK_DSV4_MODEL_LAYER_COUNT : SparkDsv4ServingDescriptor.stage_layer_counts[state->stage_index];
	state->node_context.resident_sequence_capacity = state->common.resident_sequence_capacity;
	state->node_context.pipeline_slot_count = state->common.pipeline_slot_count;
	state->node_context.max_sequence_positions = max_sequence_positions;
	state->node_context.linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC;
	state->node_context.expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC;
	state->node_context.kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC;
	state->node_context.tp_degree = SPARK_DSV4_SERVING_TP_DEGREE;
	state->node_context.tp_rank = SparkDsv4ServingTpRank(state->stage_index);
	state->node_context.tp_configuration_hash = 0u;
	memset(&shape,0,sizeof(shape));
	shape.abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
	shape.tp_degree = state->node_context.tp_degree;
	shape.tp_rank = state->node_context.tp_rank;
	shape.pp_stage_count = SPARK_DSV4_SERVING_HYBRID != 0u ?
		SPARK_DSV4_SERVING_PP_STAGE_COUNT :
		(SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? 1u :
		 SPARK_DSV4_SERVING_STAGE_COUNT);
	shape.pp_stage_index = SPARK_DSV4_SERVING_HYBRID != 0u ?
		state->node_context.stage_index :
		(SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? 0u : state->stage_index);
	if ( SparkDsv4TpDeriveNodeConfig(&shape,&tp_config) != SPARK_STATUS_OK )
		return(SPARK_STATUS_VALIDATION_FAILED);
	state->node_context.tp_configuration_hash = SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u ? tp_config.configuration_hash : 0u;
	if ( SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u )
	{
		state->node_context.first_layer_index = tp_config.first_layer_index;
		state->node_context.layer_count = tp_config.layer_count;
	}
	state->node_context.world_size = tp_config.world_size;
	state->node_context.world_rank = tp_config.world_rank;
	state->node_context.pp_stage_count = shape.pp_stage_count;
	state->node_context.pp_stage_index = shape.pp_stage_index;
	if ( SPARK_DSV4_SERVING_TOPOLOGY_FLAG != 0u )
	{
		state->node_context.tp_listen_port = state->tp_listen_port;
		memcpy(state->node_context.tp_peer_ports,state->tp_peer_ports,sizeof(state->tp_peer_ports));
		state->node_context.tp_connect_timeout_milli = state->tp_connect_timeout_milli;
		state->node_context.tp_operation_timeout_milli = state->tp_operation_timeout_milli;
		state->node_context.tp_collective_backend_kind =
			state->tp_collective_backend_kind;
		state->node_context.tp_collective_identifier = state->tp_collective_identifier;
		state->node_context.tp_collective_topology =
			state->tp_collective_topology;
		state->node_context.tp_collective_backend_module_path =
			state->tp_collective_backend_path;
		state->node_context.tp_collective_control_port_base =
			state->tp_collective_control_port_base;
		if ( SPARK_DSV4_SERVING_HYBRID != 0u )
		{
			uint32_t group_first_rank,index;
			group_first_rank = state->node_context.pp_stage_index * SPARK_DSV4_SERVING_TP_DEGREE;
			memset(state->node_context.tp_peer_ports,0,sizeof(state->node_context.tp_peer_ports));
			for (index=0u; index<SPARK_DSV4_SERVING_TP_DEGREE; index++)
				state->node_context.tp_peer_ports[index] = state->tp_peer_ports[group_first_rank + index];
			if ( SparkTpDeviceCollectiveSliceTopology(
					&state->tp_collective_topology,group_first_rank,
					SPARK_DSV4_SERVING_TP_DEGREE,
					&state->node_context.tp_collective_topology) !=
					SPARK_STATUS_OK )
				return(SPARK_STATUS_VALIDATION_FAILED);
			state->node_context.tp_collective_control_port_base =
				state->tp_peer_ports[group_first_rank];
			state->node_context.tp_collective_identifier ^= (uint64_t)state->node_context.pp_stage_index << 32u;
		}
	}
	/* Zero keeps the eager decode path; the deployment opts into capture. */
	state->node_context.cuda_graph_count = cuda_graph_count;
	state->node_context.stage_pack_path = state->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkDsv4ServingAdapterState *state;
	uint32_t max_sequence_positions,cuda_graph_count;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkDsv4ServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SparkDsv4ServingAdapterState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkDsv4ServingInitializeState(state,configuration);
	status = SparkDsv4ServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&cuda_graph_count);
		if ( status == SPARK_STATUS_OK && (max_sequence_positions < SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO || max_sequence_positions > SPARK_DSV4_MODEL_MAX_POSITIONS) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkDsv4ServingInitializeNodeContext(state,
			max_sequence_positions,cuda_graph_count);
		if ( status == SPARK_STATUS_OK )
			status = SparkDsv4ServingLoadDriver(state,configuration);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ServingInitializeRunner(state,configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv4ServingDestroy(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ServingValidateSubmissionBase(
	SparkDsv4ServingAdapterState *state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	/* Presence + quiescing gate + runtime-submission validation (shared). */
	status = SparkAdapterValidateSubmissionOpen(&state->common,
		&SparkDsv4ServingDescriptor,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkDsv4ServingValidateRowOrder(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->model_extension_bytes != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ServingValidateSubmission(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkDsv4ServingAdapterState *state;
	uint32_t emit_count;
	SparkStatus status;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	status = SparkDsv4ServingValidateSubmissionBase(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(SPARK_STATUS_OK);
	return(SparkModelServingAdapterSelectEmitRows(submission,0,0,0u,&emit_count));
}

static SparkStatus SparkDsv4ServingPrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submissions,
	uint32_t submission_count)
{
	SparkDsv4ServingAdapterState *state;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t cache_lane_count,index;
	SparkStatus status;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	if ( state == 0 || submissions == 0 || submission_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<submission_count; index++)
	{
		status = SparkDsv4ServingValidateSubmissionBase(state,&submissions[index]);
		if ( status == SPARK_STATUS_OK && submissions[index].work_kind ==
			SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			continue;
		if ( status == SPARK_STATUS_OK )
			status = SparkModelServingAdapterBuildDriverCacheLanes(
				&submissions[index],SparkDsv4ServingPrefetchLanes,
				SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
				&cache_lane_count);
		if ( status == SPARK_STATUS_OK )
		{
			status = SparkAdmissionRequestFromSubmission(state->common.program->program_id,
				&submissions[index],SparkDsv4ServingPrefetchLanes,
				SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,&request);
			if ( status == SPARK_STATUS_OK )
				status = SparkAdmissionEvaluate(state->common.driver.interface,
					state->common.driver_instance,&request,&decision);
		}
	}
	return(status);
}

static SparkStatus SparkDsv4ServingResolvePrefetch(
	void *adapter_state,
	const SparkModelServingSubmission *submission,
	uint32_t resolution)
{
	SparkDsv4ServingAdapterState *state;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t admission_flag,cache_lane_count;
	SparkStatus status;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	if ( state == 0 || submission == 0 ||
		(resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		 resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkDsv4ServingValidateSubmissionBase(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(SPARK_STATUS_OK);
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission,
		SparkDsv4ServingPrefetchLanes,
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
		&cache_lane_count);
	if ( status != SPARK_STATUS_OK || cache_lane_count !=
		submission->active_sequence_count )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INTERNAL_ERROR);
	admission_flag = resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT :
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	status = SparkAdmissionRequestFromSubmission(state->common.program->program_id,
		submission,SparkDsv4ServingPrefetchLanes,admission_flag,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkAdmissionEvaluate(state->common.driver.interface,
		state->common.driver_instance,&request,&decision));
}

static SparkStatus SparkDsv4ServingSubmitRelease(
	SparkDsv4ServingAdapterState *state,
	const SparkModelServingSubmission *submission,
	SparkDsv4ServingPending *pending)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverFrame frame;
	SparkStatus status;
	status = SparkAdmissionRequestFromSubmission(state->common.program->program_id,
		submission,pending->cache_lanes,0u,&request);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkAdmissionEvaluate(state->common.driver.interface,
		state->common.driver_instance,&request,&decision);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&frame,0,sizeof(frame));
	frame.request_id = submission->request_id;
	frame.sequence_id = submission->sequence_id;
	frame.sequence_position = submission->sequence_position;
	frame.deadline_time_ns = submission->deadline_time_ns;
	frame.active_slot_count = submission->active_sequence_count;
	frame.priority = submission->priority;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	frame.driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame.program_id = state->common.program->program_id;
	frame.execution_stream = state->runner.execution_stream;
	frame.cache_lane_count = pending->cache_lane_count;
	frame.cache_lanes = pending->cache_lanes;
	frame.residency = submission->residency;
	frame.completion_function = SparkDsv4ServingDriverCompletion;
	frame.completion_context = pending;
	status = SparkModelDriverApplyAdmissionDecision(&decision,&frame);
	if ( status == SPARK_STATUS_OK )
		status = state->common.program->submit(state->common.driver_instance,&frame);
	return(status);
}

static SparkStatus SparkDsv4ServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkDsv4ServingAdapterState *state;
	SparkDsv4ServingPending *pending;
	SparkDsv4StageRunnerDispatch dispatch;
	SparkStatus status;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	status = SparkDsv4ServingValidateSubmissionBase(state,submission);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_adapter submit_validate status=%s stage=%u submission=%llu kind=%u rows=%u lanes=%u\n",SparkStatusToString(status),state != 0 ? state->stage_index : UINT32_MAX,submission != 0 ? (unsigned long long)submission->submission_id : 0ull,submission != 0 ? submission->work_kind : 0u,submission != 0 ? submission->row_count : 0u,submission != 0 ? submission->active_sequence_count : 0u);
		return(status);
	}
	status = SparkDsv4ServingReservePending(state,submission,&pending);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_adapter reserve_pending status=%s stage=%u submission=%llu\n",SparkStatusToString(status),state->stage_index,(unsigned long long)submission->submission_id);
		return(status);
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
	{
		status = SparkDsv4ServingSubmitRelease(state,submission,pending);
		if ( status != SPARK_STATUS_OK )
			pending->core.active = 0u;
		return(status);
	}
	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version = SPARK_DSV4_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = SPARK_DSV4_STAGE_RUNNER_DISPATCH_BYTES;
	dispatch.flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL : 0u;
	dispatch.priority = submission->priority;
	dispatch.request_id = submission->request_id;
	dispatch.sequence_id = submission->sequence_id;
	dispatch.sequence_position = submission->sequence_position;
	dispatch.deadline_time_ns = submission->deadline_time_ns;
	dispatch.submission_id = submission->submission_id;
	dispatch.control_generation = submission->control_generation;
	dispatch.transaction_id = submission->transaction_id;
	dispatch.dispatch_generation = submission->dispatch_generation;
	dispatch.request_generation = submission->request_generation;
	dispatch.step_generation = submission->step_generation;
	dispatch.active_sequence_count = submission->active_sequence_count;
	dispatch.new_token_count = submission->new_token_count;
	dispatch.row_count = submission->row_count;
	dispatch.lane_count = submission->lane_count;
	dispatch.cache_lane_count = pending->cache_lane_count;
	dispatch.tokens_per_sequence = submission->tokens_per_sequence;
	dispatch.cache_lanes = pending->cache_lanes;
	dispatch.token_ids = submission->token_ids;
	dispatch.row_lane_indices = pending->resident_row_lane_indices;
	dispatch.row_positions = submission->row_positions;
	dispatch.row_sequence_ids = submission->row_sequence_ids;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
	{
		dispatch.emit_count = pending->emit_count;
		if ( pending->emit_count != 0u )
		{
			dispatch.emit_row_indices = pending->emit_row_indices;
			dispatch.emit_lane_indices = pending->emit_lane_indices;
		}
	}
	dispatch.output_token_ids = state->runner.owns_final_head != 0u ? pending->output_token_ids : 0;
	dispatch.hidden_input_bf16 = submission->hidden_input_address;
	dispatch.hidden_input_bytes = submission->hidden_input_bytes;
	dispatch.hidden_output_bf16 = submission->hidden_output_address;
	dispatch.hidden_output_bytes = submission->hidden_output_bytes;
	dispatch.residency = submission->residency;
	dispatch.completion_function = SparkDsv4ServingDriverCompletion;
	dispatch.completion_context = pending;
	status = SparkDsv4StageRunnerSubmit(&state->runner,&dispatch);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv4StageRunnerStats stats;
		memset(&stats,0,sizeof(stats));
		(void)SparkDsv4StageRunnerGetStats(&state->runner,&stats);
		fprintf(stderr,"dsv4_adapter runner_submit status=%s stage=%u submission=%llu kind=%u rows=%u lanes=%u cache_lanes=%u last=%u rejected=%llu admitted=%llu\n",SparkStatusToString(status),state->stage_index,(unsigned long long)submission->submission_id,submission->work_kind,submission->row_count,submission->active_sequence_count,dispatch.cache_lane_count,stats.last_status,(unsigned long long)stats.rejected_count,(unsigned long long)stats.admitted_count);
	}
	if ( status != SPARK_STATUS_OK )
		pending->core.active = 0u;
	return(status);
}

/* Progress is the shared stream-ordered stub; quiesce and snapshot are
 * the shared lifecycle bodies over the common state. */

static SparkStatus SparkDsv4ServingQuiesce(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SparkDsv4ServingAdapterState *state;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	return(SparkAdapterQuiesce(state != 0 ? &state->common : 0,deadline_time_ns));
}

static SparkStatus SparkDsv4ServingSnapshot(
	void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkDsv4ServingAdapterState *state;
	state = (SparkDsv4ServingAdapterState *)adapter_state;
	return(SparkAdapterSnapshot(state != 0 ? &state->common : 0,snapshot));
}

static const SparkModelServingAdapterInterface SparkDsv4ServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkDsv4ServingDescriptor,
	.initialize = SparkDsv4ServingInitialize,
	.destroy = SparkDsv4ServingDestroy,
	.validate_submission = SparkDsv4ServingValidateSubmission,
	.submit = SparkDsv4ServingSubmit,
	.prefetch = SparkDsv4ServingPrefetch,
	.resolve_prefetch = SparkDsv4ServingResolvePrefetch,
	.progress = SparkModelServingAdapterStreamOrderedProgress,
	.quiesce = SparkDsv4ServingQuiesce,
	.snapshot = SparkDsv4ServingSnapshot
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SparkDsv4ServingInterface);
}
