#include <stdlib.h>
#include "sparkpipe/spark_error_site.h"
#include <stdatomic.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_driver_loader.h"
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_glm5_next_serving_adapter.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_speculation_seam.h"
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next

#include "sparkpipe/family/spark_family.h"

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

#define SPARK_GLM5_NEXT_SERVING_MTP_ENV "SPARK_GLM5_NEXT_MTP"
#define SPARK_GLM5_NEXT_SERVING_SPECULATORS_ENV "SPARK_GLM5_NEXT_SPECULATORS"
#define SPARK_GLM5_NEXT_SERVING_AVAILABLE_SOURCES \
	(SPARK_SPECULATION_SEAM_SOURCE_MTP | \
	 SPARK_SPECULATION_SEAM_REMOTE_TAP_FREE_SOURCES)
#define SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_TIME_BUDGET_MS 20u
#define SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_MAX_DEPTH 16u
#define SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_MAX_NODE_COUNT 64u
#define SPARK_GLM5_NEXT_SERVING_SEAM_CONNECT_TIMEOUT_MS 1000u
#define SPARK_GLM5_NEXT_SERVING_SEAM_IO_TIMEOUT_MS 30000u
#define SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_LAYER_COUNT 1u

static SparkStatus SparkGlm5NextServingResolveSpeculationControl(
	uint32_t available_sources,
	const char **control_value,
	uint32_t *enabled_sources)
{
	const char *mtp_value,*mask_value;
	SparkStatus status;
	mtp_value = getenv(SPARK_GLM5_NEXT_SERVING_MTP_ENV);
	mask_value = getenv(SPARK_GLM5_NEXT_SERVING_SPECULATORS_ENV);
	if ( mtp_value != 0 && mask_value != 0 )
	{
		(void)fprintf(stderr,"GLM5_NEXT-ADAPTER %s and %s are both present: set only %s (speculation source mask)\n",SPARK_GLM5_NEXT_SERVING_MTP_ENV,SPARK_GLM5_NEXT_SERVING_SPECULATORS_ENV,SPARK_GLM5_NEXT_SERVING_SPECULATORS_ENV);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( mtp_value != 0 )
	{
		if ( mtp_value[0] == '0' && mtp_value[1] == '\0' )
		{
			*control_value = "0";
			*enabled_sources = 0u;
			return(SPARK_STATUS_OK);
		}
		if ( mtp_value[0] == '1' && mtp_value[1] == '\0' )
		{
			*control_value = "0x1";
			*enabled_sources = SPARK_SPECULATION_SEAM_SOURCE_MTP & available_sources;
			return(SPARK_STATUS_OK);
		}
		(void)fprintf(stderr,"GLM5_NEXT-ADAPTER %s must be exactly 0 or 1 (or use %s speculation source mask)\n",SPARK_GLM5_NEXT_SERVING_MTP_ENV,SPARK_GLM5_NEXT_SERVING_SPECULATORS_ENV);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	*control_value = mask_value;
	if ( mask_value == 0 || (mask_value[0] == '1' && mask_value[1] == '\0') )
	{
		*enabled_sources = 0u;
		return(SPARK_STATUS_OK);
	}
	status = SparkSpeculationSeamParseControl(mask_value,available_sources,enabled_sources);
	if ( status != SPARK_STATUS_OK )
	{
		(void)fprintf(stderr,"GLM5_NEXT-ADAPTER %s control value rejected: status=%d available=0x%x\n",SPARK_GLM5_NEXT_SERVING_SPECULATORS_ENV,(int)status,available_sources);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

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

#define SPARK_GLM5_NEXT_SERVING_CONFIGURATION_MEMBERS_BASE (sizeof(SparkGlm5NextServingConfigurationMembers) / sizeof(SparkGlm5NextServingConfigurationMembers[0]))
#define SPARK_GLM5_NEXT_SERVING_CONFIGURATION_MEMBERS_MAX (SPARK_GLM5_NEXT_SERVING_CONFIGURATION_MEMBERS_BASE + 3u)

static uint32_t SparkGlm5NextServingConfigurationList(uint32_t bridge,uint32_t index_cp,const char **list)
{
	uint32_t count;
	for (count=0u; count<SPARK_GLM5_NEXT_SERVING_CONFIGURATION_MEMBERS_BASE; count++)
		list[count] = SparkGlm5NextServingConfigurationMembers[count];
	if ( bridge != 0u )
	{
		list[count++] = "draft_bridge_host";
		list[count++] = "draft_bridge_port";
	}
	if ( index_cp != 0u )
		list[count++] = "dsa_index_context_parallel";
	return(count);
}

typedef struct SparkGlm5NextServingPending
{
	struct SparkGlm5NextServingState *owner;
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
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffer;
	SparkModelDriverFrame frame;
	SparkModelDriverCacheLane cache_lanes[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_row_by_lane[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t resident_slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint32_t input_token_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint64_t row_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	uint64_t row_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
	SparkRowSampling row_sampling[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT];
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
	uint32_t mtp_enabled;
	uint32_t index_cp;
	SparkSpeculationSeam *speculation_seam;
	char *bridge_host;
	uint32_t bridge_port;
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
	SparkGlm5NextServingPending pending[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
} SparkGlm5NextServingState;

static _Thread_local SparkModelDriverCacheLane SparkGlm5NextServingCacheScratch[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];

static const SparkModelServingAdapterDescriptor SparkGlm5NextServingDescriptor =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES,
	.capability_flags = SPARK_GLM5_NEXT_SERVING_TOPOLOGY_FLAG |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_ASYNC_COMPLETION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING,
	.stage_count = SPARK_GLM5_NEXT_SERVING_STAGE_COUNT,
	.layer_count = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT,
	.boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16,
	.boundary_element_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT,
	.boundary_element_bytes = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = GLM5_NEXT_EXPERT_WEIGHT_CODEC,
	.kv_cache_codec = SPARK_WEIGHT_CODEC_BF16,
	.cache_block_token_count = SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS,
	.max_inflight_submission_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
	.max_active_sequence_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_input_row_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,
	.max_resident_sequence_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_output_token_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
	.max_speculative_token_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,
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

static SparkStatus SparkGlm5NextServingLoadTpCollective(
	const SparkJsonDocument *document,
	int32_t root,
	const char *runtime_root,
	SparkGlm5NextServingState *state,
	uint32_t tp_degree)
{
	SparkTpCollectiveConfigPolicy policy;
	SparkTpCollectiveAdapterConfig config;
	SparkStatus status;
	if ( document == 0 || runtime_root == 0 || state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&policy,0,sizeof(policy));
	policy.peer_count = tp_degree;
	policy.allow_zero_collective_identifier = 1u;
	policy.require_contiguous_peer_ports = 1u;
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_ADAPTIVE_COMBOS;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_MASK_CONDITIONAL;
	policy.require_session_ports = 1u;
	memset(&config,0,sizeof(config));
	config.backend_module_path_buffer = state->tp_collective_backend_path;
	config.backend_module_path_bytes = sizeof(state->tp_collective_backend_path);
	status = SparkServingAdapterTemplateLoadTpCollective(document,root,runtime_root,&policy,&config);
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
		memcpy(state->node_context.tp_collective_session_ports_hc,config.session_ports_hc,sizeof(state->node_context.tp_collective_session_ports_hc));
	}
	(void)fprintf(stderr,"GLM5_NEXT-ADAPTER LoadTpCollective rc=%d backend=%u\n",(int)status,state->tp_collective_backend_kind);
	SPARK_RETURN(status);
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
	const char *members[SPARK_GLM5_NEXT_SERVING_CONFIGURATION_MEMBERS_MAX];
	char *relative_stage_pack_path;
	uint32_t schema_version;
	int32_t root,token,index_cp_token;
	int32_t bridge_host_token,bridge_port_token;
	SparkStatus status;
	relative_stage_pack_path = 0;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	bridge_host_token = status == SPARK_STATUS_OK ? SparkGlm5NextServingJsonMember(&document,root,"draft_bridge_host") : -1;
	bridge_port_token = status == SPARK_STATUS_OK ? SparkGlm5NextServingJsonMember(&document,root,"draft_bridge_port") : -1;
	index_cp_token = status == SPARK_STATUS_OK ? SparkGlm5NextServingJsonMember(&document,root,"dsa_index_context_parallel") : -1;
	if ( status == SPARK_STATUS_OK && (bridge_host_token < 0) != (bridge_port_token < 0) )
	{
		(void)fprintf(stderr,"GLM5_NEXT-ADAPTER draft_bridge_host and draft_bridge_port must both be present or both absent\n");
		status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(&document,root,members,SparkGlm5NextServingConfigurationList(bridge_host_token >= 0 ? 1u : 0u,index_cp_token >= 0 ? 1u : 0u,members));
	state->index_cp = 0u;
	if ( status == SPARK_STATUS_OK && index_cp_token >= 0 )
		status = SparkJsonGetUInt32(&document,index_cp_token,&state->index_cp);
	if ( status == SPARK_STATUS_OK && state->index_cp > 1u )
		status = SPARK_STATUS_SCHEMA_ERROR;
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
	if ( status == SPARK_STATUS_OK && bridge_host_token >= 0 )
	{
		status = SparkJsonCopyString(&document,bridge_host_token,&state->bridge_host);
		if ( status == SPARK_STATUS_OK )
			status = SparkJsonGetUInt32(&document,bridge_port_token,&state->bridge_port);
	}
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	(void)fprintf(stderr,"GLM5_NEXT-ADAPTER LoadConfiguration rc=%d\n",(int)status);
	SPARK_RETURN(status);
}

static void SparkGlm5NextServingModelContract(SparkSpeculationModelContract *contract)
{
	contract->abi_version = SPARK_SPECULATION_ABI_VERSION;
	contract->descriptor_bytes = SPARK_SPECULATION_MODEL_CONTRACT_DESCRIPTOR_BYTES;
	contract->verifier_hidden_dtype = SPARK_SPECULATION_VERIFIER_HIDDEN_DTYPE_BF16;
	contract->draft_dtype = SPARK_SPECULATION_DRAFT_DTYPE_BF16;
	contract->draft_layer_count = SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_LAYER_COUNT;
	contract->block_size = SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS;
	contract->hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	contract->intermediate_dimension = SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION;
	contract->attention_head_count = SPARK_GLM5_NEXT_MODEL_HEAD_COUNT;
	contract->kv_head_count = SPARK_GLM5_NEXT_MODEL_HEAD_COUNT;
	contract->head_dimension = SPARK_GLM5_NEXT_MODEL_VALUE_HEAD_DIMENSION;
	contract->vocab_size = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	contract->draft_vocab_size = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	contract->markov_rank = 0u;
	contract->maximum_speculative_token_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	contract->verifier_accept_k = 1u;
	contract->aux_layer_count = 0u;
	contract->enable_confidence_head = 0u;
	contract->confidence_head_with_markov = 0u;
}

static SparkStatus SparkGlm5NextServingInitializeSpeculationSeam(
	SparkGlm5NextServingState *state,
	uint32_t max_sequence_positions)
{
	SparkSpeculationSeamConfiguration seam_configuration;
	const char *control_value;
	uint32_t available_sources;
	uint32_t enabled_sources;
	SparkStatus status;
	available_sources = SPARK_GLM5_NEXT_SERVING_AVAILABLE_SOURCES;
	if ( state->bridge_host == 0 )
		available_sources &= ~SPARK_SPECULATION_SEAM_REMOTE_SOURCES;
	control_value = 0;
	enabled_sources = 0u;
	status = SparkGlm5NextServingResolveSpeculationControl(available_sources,&control_value,&enabled_sources);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->mtp_enabled = (enabled_sources & SPARK_SPECULATION_SEAM_SOURCE_MTP) != 0u ? 1u : 0u;
	memset(&seam_configuration,0,sizeof(seam_configuration));
	seam_configuration.abi_version = SPARK_SPECULATION_SEAM_ABI_VERSION;
	seam_configuration.descriptor_bytes = SPARK_SPECULATION_SEAM_DESCRIPTOR_BYTES;
	seam_configuration.available_source_mask = available_sources;
	seam_configuration.default_source_mask = 0u;
	seam_configuration.default_speculative_token_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	seam_configuration.lane_count = state->max_active_sequence_count;
	seam_configuration.max_committed_token_count = max_sequence_positions;
	seam_configuration.max_tap_row_count = 0u;
	seam_configuration.draft_time_budget_ms = SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_TIME_BUDGET_MS;
	seam_configuration.draft_max_depth = SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_MAX_DEPTH;
	seam_configuration.draft_max_node_count = SPARK_GLM5_NEXT_SERVING_SEAM_DRAFT_MAX_NODE_COUNT;
	seam_configuration.connect_timeout_ms = SPARK_GLM5_NEXT_SERVING_SEAM_CONNECT_TIMEOUT_MS;
	seam_configuration.io_timeout_ms = SPARK_GLM5_NEXT_SERVING_SEAM_IO_TIMEOUT_MS;
	seam_configuration.control_value = control_value;
	seam_configuration.bridge_host = state->bridge_host;
	seam_configuration.bridge_port = state->bridge_port;
	memcpy(seam_configuration.target_model,SPARK_GLM5_NEXT_SERVING_MODEL_ID,sizeof(SPARK_GLM5_NEXT_SERVING_MODEL_ID));
	SparkGlm5NextServingModelContract(&seam_configuration.model_contract);
	status = SparkSpeculationSeamInitialize(&seam_configuration,&state->speculation_seam);
	if ( status != SPARK_STATUS_OK )
	{
		(void)fprintf(stderr,"GLM5_NEXT-ADAPTER speculation seam init failed: status=%d\n",(int)status);
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextServingBurstLimit(const SparkGlm5NextServingState *state,uint32_t work_kind,uint32_t active_sequence_count,uint32_t tokens_per_sequence)
{
	if ( work_kind != SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(1u);
	if ( tokens_per_sequence > 1u )
		return(tokens_per_sequence);
	return(state->mtp_enabled != 0u && active_sequence_count == 1u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u : 1u);
}

static void SparkGlm5NextServingDriverCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkGlm5NextServingPending *pending;
	SparkGlm5NextServingState *state;
	SparkModelServingCompletion completion;
	uint32_t index,matches;
	pending = (SparkGlm5NextServingPending *)completion_context;
	state = pending != 0 ? pending->owner : 0;
	if ( state == 0 || driver_completion == 0 )
		return;
	if ( atomic_load_explicit(&pending->active,memory_order_acquire) == 0u )
	{
		SparkModelServingCompletion orphan;
		memset(&orphan,0,sizeof(orphan));
		orphan.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
		orphan.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
		orphan.status = SPARK_STATUS_NOT_FOUND;
		orphan.submission_id = pending->submission_id;
		orphan.request_id = pending->request_id;
		orphan.sequence_id = pending->sequence_id;
		orphan.sequence_position = pending->sequence_position;
		orphan.control_generation = pending->control_generation;
		orphan.transaction_id = pending->transaction_id;
		orphan.dispatch_generation = pending->dispatch_generation;
		orphan.request_generation = pending->request_generation;
		orphan.step_generation = pending->step_generation;
		fprintf(stderr,
			"PENDING-INACTIVE-COMPLETION id=%llu — pending cleared under the completion; delivering NOT_FOUND so the route releases (wedge #26)\n",
			(unsigned long long)pending->submission_id);
		state->completion_function(state->completion_context,&orphan);
		return;
	}
	matches = driver_completion->request_id == pending->request_id && driver_completion->sequence_id == pending->sequence_id && driver_completion->sequence_position == pending->sequence_position && driver_completion->program_id == state->program->program_id;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = matches != 0u ? SparkModelServingCompletionStatus((uint32_t)driver_completion->status) : SPARK_STATUS_SCHEMA_ERROR;
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
	if ( completion.status == SPARK_STATUS_OK && pending->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE && pending->work_kind != SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
	{
		uint32_t burst = driver_completion->tokens_per_sequence != 0u ?
			driver_completion->tokens_per_sequence : 1u;
		if ( burst > SparkGlm5NextServingBurstLimit(state,pending->work_kind,pending->active_sequence_count,pending->frame.tokens_per_sequence) )
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
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[pending->last_row_by_lane[index / burst] * burst + index % burst];
	}
	atomic_store_explicit(&pending->active,0u,memory_order_release);
	state->completion_function(state->completion_context,&completion);
}

#include "sparkpipe/family/serving/spark_serving_reserve_pending.h"

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
	SparkSpeculationSeamDestroy(state->speculation_seam);
	free(state->bridge_host);
	free(state);
}

#include "sparkpipe/family/serving/spark_serving_orphan_driver_completion_atomic.h"
#include "sparkpipe/family/serving/spark_serving_driver_wake.h"

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
		SPARK_RETURN(status);
	descriptor = state->driver.interface->descriptor;
	if ( descriptor == 0 || strcmp(descriptor->model_id,SPARK_GLM5_NEXT_SERVING_DRIVER_MODEL_ID) != 0 || strcmp(descriptor->model_revision,GLM5_NEXT_MODEL_REVISION) != 0 || strcmp(descriptor->stage_name,SPARK_GLM5_NEXT_SERVING_STAGE_NAME) != 0 || strcmp(descriptor->target,SPARK_GLM5_NEXT_SERVING_TARGET) != 0 )
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
	if ( status != SPARK_STATUS_OK )
		(void)fprintf(stderr,"G5N-T1ADAPTER-DIAG create=%d stage=%u first=%u layers=%u slots=%u cap=%u pos=%u rows=%u thr=%u tp=%u/%u codec=%u flags=%u rev=%s pack=%s\n",
		    (int)status,state->node_context.stage_count,state->node_context.first_layer_index,state->node_context.layer_count,state->node_context.pipeline_slot_count,state->node_context.resident_sequence_capacity,state->node_context.max_sequence_positions,state->node_context.execution_row_capacity,state->node_context.decode_split_context_threshold,state->node_context.tp_degree,state->node_context.tp_rank,state->node_context.expert_weight_codec,state->node_context.flags,state->node_context.model_revision == 0 ? "(null)" : state->node_context.model_revision,state->node_context.stage_pack_path == 0 ? "(null)" : state->node_context.stage_pack_path);
	(void)fprintf(stderr,"GLM5_NEXT-ADAPTER LoadDriver rc=%d max_sequence_positions=%u execution_row_capacity=%u decode_split_context_threshold=%u resident_sequence_capacity=%u kv_pages=%u/%u\n",(int)status,state->node_context.max_sequence_positions,state->node_context.execution_row_capacity,state->node_context.decode_split_context_threshold,state->node_context.resident_sequence_capacity,configuration->runtime_limits.kv_logical_page_capacity,configuration->runtime_limits.kv_physical_page_capacity);
	return(status == SPARK_STATUS_OK && state->driver_instance == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status);
}

#include "sparkpipe/family/serving/spark_serving_validate_configuration.h"

static SparkStatus SparkGlm5NextServingInitialize(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SparkGlm5NextServingState *state;
	uint32_t max_sequence_positions,execution_row_capacity,tp_degree,tp_rank;
	uint32_t decode_split_context_threshold,index;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SparkGlm5NextServingValidateConfiguration(configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state = (SparkGlm5NextServingState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	atomic_init(&state->orphan_completion_count,0u);
	atomic_init(&state->quiescing,0u);
	atomic_init(&state->reset_active,0u);
	atomic_init(&state->reset_generation,0u);
	for (index=0u; index<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; index++)
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
	status = SparkGlm5NextServingLoadConfiguration(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions,&execution_row_capacity,&decode_split_context_threshold,&tp_degree,&tp_rank);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS || execution_row_capacity == 0u || execution_row_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || decode_split_context_threshold > max_sequence_positions) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK && (tp_rank != configuration->stage_index || tp_degree != SPARK_GLM5_NEXT_SERVING_TP_DEGREE) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingInitializeSpeculationSeam(state,max_sequence_positions);
	if ( status == SPARK_STATUS_OK )
	{
		state->node_context.abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
		state->node_context.descriptor_bytes = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
		state->node_context.stage_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_STAGE_COUNT;
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
		state->node_context.flags = 0u;
		if ( state->mtp_enabled != 0u )
			state->node_context.flags |= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP;
		if ( state->index_cp != 0u )
			state->node_context.flags |= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_INDEX_CP;
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
		SPARK_RETURN(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/serving/spark_serving_validate_row_order.h"

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
	if ( submission != 0 && submission->control_generation < atomic_load_explicit(&state->reset_generation,memory_order_acquire) )
	{
		fprintf(stderr,"ADMIT9-RESETGEN submission=%llu control_generation=%llu reset_generation=%llu\n",
			(unsigned long long)submission->submission_id,
			(unsigned long long)submission->control_generation,
			(unsigned long long)atomic_load_explicit(&state->reset_generation,memory_order_acquire));
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SparkGlm5NextServingDescriptor,&state->runtime_limits,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingValidateBoundaries(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextServingValidateRowOrder(state,submission);
	if ( status == SPARK_STATUS_OK && submission->model_extension_bytes != 0u )
		status = SPARK_STATUS_UNSUPPORTED;
	SPARK_RETURN(status);
}

#include "sparkpipe/family/serving/spark_serving_cache_context.h"

#include "sparkpipe/family/serving/spark_serving_prefetch.h"
#include "sparkpipe/family/serving/spark_serving_abort_unexecuted.h"

static void SparkGlm5NextServingBuildFrame(
	const SparkGlm5NextServingState *state,
	const SparkModelServingSubmission *submission,
	SparkGlm5NextServingPending *pending)
{
	SparkGlm5NextResidentDecodeStageBatchView *batch = &pending->batch;
	SparkGlm5NextResidentDecodeStageFrameContext *context = &pending->context;
	SparkModelDriverBuffer *buffer = &pending->buffer;
	SparkModelDriverFrame *frame = &pending->frame;
	uint32_t row;
	for (row=0u; SparkModelServingWorkKindUsesRows(submission->work_kind) != 0u && row<submission->row_count; row++)
		pending->row_sampling[row] = submission->lanes[submission->row_lane_indices[row]].sampling;
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = submission->row_count;
	batch->active_sequence_count = submission->active_sequence_count;
	batch->token_ids = pending->input_token_ids;
	batch->row_resident_slots = pending->resident_slots;
	batch->row_positions = pending->row_positions;
	batch->row_sequence_ids = pending->row_sequence_ids;
	batch->row_sampling = pending->row_sampling;
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
	buffer->bytes = (uint64_t)submission->row_count * SparkGlm5NextServingBurstLimit(state,submission->work_kind,submission->active_sequence_count,submission->tokens_per_sequence) * sizeof(uint32_t);
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = submission->sequence_id;
	frame->sequence_position = submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = submission->active_sequence_count;
	frame->new_token_count = submission->row_count;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH ? SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH : 0u;
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
	SparkGlm5NextServingPending *pending,
	SparkModelDriverFrame *frame)
{
	SparkServingCacheAdmission cache;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	cache = SparkGlm5NextServingCacheContext(state,pending->cache_lanes);
	status = SparkServingCacheBuildRequest(&cache,submission,0u,&request);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	frame->cache_lanes = pending->cache_lanes;
	frame->cache_lane_count = request.cache_lane_count;
	return(SparkAdmissionEvaluateAndApply(state->driver.interface,state->driver_instance,&request,frame,&decision));
}

static SparkStatus SparkGlm5NextServingSubmit(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkGlm5NextServingState *state;
	SparkGlm5NextServingPending *pending;
	SparkModelDriverCompletion released = {0};
	SparkStatus status;
	state = (SparkGlm5NextServingState *)adapter_state;
	status = SparkGlm5NextServingValidateSubmission(state,submission);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	pending = SparkGlm5NextServingReservePending(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	SparkGlm5NextServingBuildFrame(state,submission,pending);
	status = SparkGlm5NextServingAdmit(state,submission,pending,&pending->frame);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
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
			SparkGlm5NextServingDriverCompletion(pending,&released);
		}
		else
			status = state->program->submit(state->driver_instance,&pending->frame);
		if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
			fprintf(stderr,"G5N-DBG submit: program->submit -> %d\n",(int)status);
	}
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&pending->active,0u,memory_order_release);
	SPARK_RETURN(SparkGlm5NextServingAbortUnexecuted(state,submission,status));
}

static SparkStatus SparkGlm5NextServingProgress(
	void *adapter_state,
	uint32_t maximum_step_count)
{
	SparkGlm5NextServingState *state = adapter_state;
	SparkModelDriverRuntimeSnapshot snapshot;
	(void)maximum_step_count;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&snapshot,0,sizeof(snapshot));
	return(state->driver.interface->snapshot(state->driver_instance,
		state->program->program_id,&snapshot));
}

#include "sparkpipe/family/serving/spark_serving_snapshot.h"

#include "sparkpipe/family/serving/spark_serving_quiesce.h"

static SparkStatus SparkGlm5NextServingResetControl(void *adapter_state,uint64_t control_generation)
{
	SparkGlm5NextServingState *state = (SparkGlm5NextServingState *)adapter_state;
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	if ( state == 0 || control_generation == 0u || control_generation <= atomic_load_explicit(&state->reset_generation,memory_order_acquire) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm5NextServingQuiesce(state,UINT64_MAX);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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
	SPARK_RETURN(status);
}

#include "sparkpipe/family/serving/spark_serving_reset.h"

static const SparkModelServingAdapterInterface SparkGlm5NextServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SparkGlm5NextServingDescriptor,
	.initialize = SparkGlm5NextServingInitialize,
	.destroy = SparkGlm5NextServingDestroy,
	.validate_submission = SparkGlm5NextServingValidateSubmission,
	.submit = SparkGlm5NextServingSubmit,
	.prefetch = SparkGlm5NextServingPrefetch,
	.resolve_prefetch = SparkGlm5NextServingResolvePrefetch,
	.progress = SparkGlm5NextServingProgress,
	.quiesce = SparkGlm5NextServingQuiesce,
	.snapshot = SparkGlm5NextServingSnapshot,
	.reset = SparkGlm5NextServingReset
};

#include "sparkpipe/family/serving/spark_serving_get_interface.h"
